"""C13 smvla protocol (slow): the real ``smvla_server._serve`` / ``_handler`` run on a loopback port, and the real
``smvla_client`` connects through the real ``WSPolicyConn`` to run full episodes.

``SMVLAPolicyHost`` is constructed with ``object.__new__``; the three model pieces (buffer factory, batched,
normalize_state) are replaced with CPU stubs, while the other methods (``new_episode`` reseeding, ``observe``,
``infer``) are the real implementations. No weights are read and CUDA is not initialized.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import numpy as np
import pytest

import eval_fakes as F

pytestmark = pytest.mark.slow


class _Buf:
    image_keys = ["observation.images.front", "observation.images.wrist"]

    def __init__(self):
        self.frames = []

    def reset(self):
        self.frames.clear()

    def observe(self, full):
        self.frames.append(full)

    def _prepare_inputs(self, instruction):
        return {"n": len(self.frames), "instruction": instruction,
                "last": F.sha_bytes(self.frames[-1]["observation.images.front"]) if self.frames else ""}


class _Batched:
    device = "cpu"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def generate_batch(self, processed, states):
        self.calls.append(processed[0])
        if self.fail:
            raise RuntimeError("stub inference failed")
        seed = processed[0]["n"] * 1000 + len(processed[0]["instruction"])
        a = np.random.default_rng(seed).uniform(-1, 1, size=(F.CHUNK_ROWS, 8)).astype(np.float32)
        return [(a, f"sub{processed[0]['n']}")]


def _host(fail=False):
    srv = F.smvla_server()
    h = object.__new__(srv.SMVLAPolicyHost)
    h.buffer_factory = _Buf
    h.batched = _Batched(fail)
    h.normalize_state = None
    h.to_full, h.state_norm = srv.make_closures(h.buffer_factory, None, h.batched)
    h.load_s = 0.0
    srv.reseed()
    h.rng_ref = srv.rng_digest()
    h.metadata = {"policy": "smvla", "fake_host": True, "rng_ref": h.rng_ref}
    return h


@pytest.fixture
def served():
    """Starts the real _serve in an event loop on a separate thread; returns (host factory setter, port)."""
    srv = F.smvla_server()
    holder = {}

    def start(host):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        fut = asyncio.run_coroutine_threadsafe(srv._serve(host, "127.0.0.1", port), loop)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        holder.update(loop=loop, fut=fut, thread=t)
        return port

    yield start
    if holder:
        async def _stop():
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        loop = holder["loop"]
        asyncio.run_coroutine_threadsafe(_stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        holder["thread"].join(timeout=5)
        loop.close()


class _Session:
    def __init__(self, plan, ep=4):
        self.env = F.FakeEnv("T", ep, plan)
        self.steps = 0

    def reset(self):
        return self.env.reset()

    def step(self, a):
        self.steps += 1
        return self.env.step(a)

    def close(self):
        pass


def _episode(port, plan, ident=None, rec=None):
    sm = F.smvla_client()
    sess = _Session(plan)
    rec = rec or F.FakeRecorder("/nonexistent-not-written", {})
    res = sm.run_episode(sess, ident or {"task": "T", "source_episode": None, "seed": 9},
                         {"host": "127.0.0.1", "port": port}, rec)
    return res, sess, rec


def test_full_episode_over_real_websocket(served):
    host = _host()
    port = served(host)
    res, sess, rec = _episode(port, F.Plan(success_at=20))
    assert res["status"] == "success" and res["steps"] == 20 and res["infra"] is False, res["error"]
    assert res["protocol"]["sha_mismatch"] == 0 and res["protocol"]["frames_sent"] == F.N_RESET_FRAMES + 20
    assert res["server_meta"]["fake_host"] is True
    # at the first inference the server buffer holds exactly the reset frames; at the second, reset frames + one
    # execution segment of frames
    assert [c["n"] for c in host.batched.calls] == [F.N_RESET_FRAMES, F.N_RESET_FRAMES + F.smvla_server().EXECUTE_HORIZON]
    assert host.batched.calls[0]["instruction"] == "goal-T-4"
    rng = [e for e in rec.events if e["kind"] == "server_rng"][0]["rng"]
    assert rng == host.rng_ref  # after each episode reset the RNG state equals the clean post-load reference


def test_two_episodes_each_reset_fresh(served):
    host = _host()
    port = served(host)
    r1, s1, _ = _episode(port, F.Plan(success_at=5))
    r2, s2, _ = _episode(port, F.Plan(success_at=5))
    assert r1["status"] == r2["status"] == "success"
    assert [c["n"] for c in host.batched.calls] == [F.N_RESET_FRAMES, F.N_RESET_FRAMES]
    for a, b in zip(s1.env.actions, s2.env.actions):
        assert np.array_equal(a, b)


def test_server_inference_error_is_infra_server_error(served):
    port = served(_host(fail=True))
    res, sess, _ = _episode(port, F.Plan(success_at=5))
    assert res["status"] == "error" and res["infra"] is True and res["infra_reason"] == "server_error"
    assert "stub inference failed" in res["error"] and sess.env.n == 0


@pytest.mark.parametrize("msg,needle", [({"bogus": {}}, "unknown message keys"),
                                         ({"infer": {"instruction": "x", "state": np.zeros(8, np.float32)}},
                                          "infer before reset"),
                                         ({"observe": {"frames": []}}, "observe before reset")])
def test_protocol_violations_get_error_reply(served, msg, needle):
    sm = F.smvla_client()
    port = served(_host())
    conn = sm.WSPolicyConn("127.0.0.1", port)
    try:
        reply, _, _ = conn.call(msg)
    finally:
        conn.close()
    assert needle in reply["error"]


def test_connection_refused_is_infra():
    sm = F.smvla_client()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    sess = _Session(F.Plan(success_at=1))
    res = sm.run_episode(sess, {"task": "T", "source_episode": None, "seed": 1}, {"host": "127.0.0.1", "port": port})
    assert res["status"] == "error" and res["infra"] is True and res["infra_reason"].startswith("connection:")
