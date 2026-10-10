"""S2：真实回环传输配合假环境，核对分块终点、审计字段及包装参数契约。"""
from __future__ import annotations

import contextlib
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

import eval_fakes as F
from robomme_ood_eval.models import framesamp_modul as mc
from robomme_ood_eval.servers import policy_server_wrap as wrap
from robomme_ood_eval.timing import ChunkTimer, attach_policy_timing


D_ADD, D_INF = 10.0, 50.0
ACTIONS = np.arange(20 * 7, dtype=np.float32).reshape(20, 7)


@contextlib.contextmanager
def fake_server(*, mode="ok", audit=True, large=False):
    """监听随机空闲端口；不导入仿真器或触发任何真实环境 reset。"""
    from openpi_client import msgpack_numpy
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    def handler(ws):
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack({"fake": True}))
        count = 0
        try:
            while True:
                request = msgpack_numpy.unpackb(ws.recv())
                if request.get("reset"):
                    out = {"reset_finished": True}
                elif request.get("add_buffer"):
                    time.sleep(D_ADD / 1000)
                    out = {"add_buffer_finished": True}
                else:
                    count += 1
                    time.sleep(D_INF / 1000)
                    if mode == "error" and count == 2:
                        ws.send("fake inference error")
                        continue
                    out = {} if mode == "missing_actions" and count == 2 else {"actions": ACTIONS}
                    if audit:
                        out[wrap.AUDIT_KEY] = {"server_timing": {"infer_ms": D_INF, "gpu": "fake"}}
                    if large:
                        out["padding"] = b"x" * (2 * 1024 * 1024)
                ws.send(packer.pack(out))
        except ConnectionClosed:
            pass

    with serve(handler, "127.0.0.1", 0, compression=None, max_size=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1]
        finally:
            server.shutdown()
            thread.join(5)
            assert not thread.is_alive()


class Session:
    def __init__(self, success_at=40):
        self.env = F.FakeEnv("T", 0, F.Plan(success_at=success_at))
        self.steps = 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        self.steps += 1
        return self.env.step(action)


def measured_loop(port, *, callback=None, timer=None):
    session, transport = Session(), {}
    client = mc.make_recording_client("127.0.0.1", port, None, transport)
    result = mc.evaluate_one(lambda: client, session.step,
                             lambda: mc.pre_traj_from_reset(*session.reset()),
                             timer=timer, on_decision=callback)
    return result, client, session, transport


def test_framesamp_chunk_timing_fake_ws():
    with fake_server() as port:
        result = mc.run_episode(Session(), {"task": "T"}, {"port": port}, None)
    timing = result["timing"]
    assert (result["steps"], result["decisions"], len(timing["chunks"])) == (40, 3, 3)
    for chunk in timing["chunks"]:
        assert D_INF <= chunk["action_rtt_ms"] <= D_INF + 40
        assert D_ADD <= chunk["add_buffer_rtt_ms"] <= D_ADD + 40
        assert chunk["decision_wall_ms"] >= chunk["action_rtt_ms"] + chunk["add_buffer_rtt_ms"]
        assert chunk["server_infer_ms"] == D_INF and chunk["lang_ms"] == 0
    assert timing["conservation"]["violations"] == 0 and timing["gpu_name"] == "fake"
    slack = min(c["decision_wall_ms"] - c["action_rtt_ms"] - c["add_buffer_rtt_ms"] for c in timing["chunks"])
    print("CHUNK_TIMING=PASS model=framesamp decisions=3 chunks=3 missing=0 steps=40")
    print("WALL_DECOMP=PASS model=framesamp kind=serial chunks=3 violations=0 "
          f"overhead_pct_p50={timing['conservation']['overhead_pct_p50']:.6f} "
          f"min_slack_ms={slack:.3f} lang_ms=0 server_infer_present=3/3")


def test_roundtrip_contract_and_audit_clear():
    with fake_server() as port:
        client = mc.make_recording_client("127.0.0.1", port, None, {})
        try:
            client.reset()
            client.add_buffer({"add_buffer": True, "images": np.zeros((1, 1, 2, 2, 3), np.uint8),
                               "state": np.zeros((1, 8), np.float32), "exec_start_idx": 0})
            out = client.infer({"prompt": "p", "observation/image": np.zeros((2, 2, 3), np.uint8),
                                "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
                                "observation/state": np.zeros(8, np.float32)})
            assert wrap.AUDIT_KEY not in out
            assert set(client._last_rtt) == {"reset", "add_buffer", "infer"}
            assert D_INF <= client._last_rtt["infer"] <= D_INF + 40
            assert client._last_server_timing == {"infer_ms": D_INF, "gpu": "fake"}
            client.reset()
            assert client._last_server_timing is None and client._last_audit is None
        finally:
            client._ws.close()
    with fake_server(audit=False) as port:
        _, client, _, _ = measured_loop(port)
        assert client._last_server_timing is None


def test_framesamp_on_decision_signature_unchanged():
    seen = []
    server = F.FakePolicyServer()
    session = Session()
    result = mc.evaluate_one(lambda: F.FakeMMEVLAWebsocketClient(server), session.step,
                             lambda: mc.pre_traj_from_reset(*session.reset()), timer=None,
                             on_decision=lambda index, actions: seen.append((index, actions)))
    assert [index for index, _ in seen] == [0, 1, 2]
    assert all(isinstance(actions, np.ndarray) for _, actions in seen)
    assert set(result) == {"status", "task_success", "steps", "error", "decisions", "infra", "infra_reason", "env_exception"}


@pytest.mark.parametrize("mode, error", [("error", "RuntimeError"), ("missing_actions", "KeyError")])
def test_framesamp_error_no_chunk(mode, error):
    timer = ChunkTimer()
    with fake_server(mode=mode) as port:
        result, client, _, _ = measured_loop(port, timer=timer)
    assert result["status"] == "error" and error in result["error"]
    assert result["decisions"] == len(timer.chunks) == 1
    if mode == "error":
        assert client._last_server_timing is None
    timing = attach_policy_timing({}, timer, gpu_name=None, model_kind="serial")
    assert timing["dangling_start"] and len(timing["chunks"]) == 1


def test_framesamp_chunk_end_at_recv(monkeypatch):
    """人工延迟解包和录像回调，精确终点仍是 recv，不依赖网络调度阈值。"""
    from openpi_client import msgpack_numpy

    original_unpack = msgpack_numpy.unpackb
    main_thread = threading.get_ident()

    def slow_unpack(data):
        if threading.get_ident() == main_thread and len(data) > 1024 * 1024:
            time.sleep(.03)
        return original_unpack(data)

    monkeypatch.setattr(msgpack_numpy, "unpackb", slow_unpack)
    timer = ChunkTimer()
    ends, starts = [], []
    real_start, real_close = timer.start, timer.close

    def start(*args, **kwargs):
        real_start(*args, **kwargs)
        starts.append(timer._t_obs)

    def close(**kwargs):
        ends.append(kwargs["t"])
        return real_close(**kwargs)

    monkeypatch.setattr(timer, "start", start)
    monkeypatch.setattr(timer, "close", close)
    with fake_server(large=True) as port:
        result, client, _, transport = measured_loop(port, timer=timer, callback=lambda *_: time.sleep(.03))
    assert result["status"] == "success" and len(timer.chunks) == 3
    for chunk, t0, t1 in zip(timer.chunks, starts, ends):
        assert chunk["decision_wall_ms"] == pytest.approx((t1 - t0) * 1000, abs=.00051)
        assert chunk["decision_wall_ms"] <= chunk["add_buffer_rtt_ms"] + chunk["action_rtt_ms"] + 10
    assert ends[-1] == client._last_recv_t
    assert all(row["unpack_s"] >= .03 for row in transport["per_msg"] if row["kind"] == "infer")


def test_audited_policy_server_timing(monkeypatch):
    monkeypatch.setattr(wrap, "_gpu_name", lambda: "fake")

    class Inner:
        def infer(self, obs):
            time.sleep(.02)
            return {"actions": ACTIONS, "payload": obs}

    original = {"actions": ACTIONS, "payload": {"prompt": "p"}}
    out = wrap.AuditedPolicy(Inner()).infer(original["payload"])
    audit = out.pop(wrap.AUDIT_KEY)
    assert audit["server_timing"]["infer_ms"] >= 20 and audit["server_timing"]["gpu"] == "fake"
    from openpi_client import msgpack_numpy
    assert msgpack_numpy.Packer().pack(out) == msgpack_numpy.Packer().pack(original)
    assert wrap._ACTIVE.sink is None

    class Broken:
        def infer(self, obs):
            raise ValueError("fake")

    with pytest.raises(ValueError, match="fake"):
        wrap.AuditedPolicy(Broken()).infer({})
    assert wrap._ACTIVE.sink is None


def test_gpu_name_cached_once(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(
        devices=lambda: calls.append(1) or [types.SimpleNamespace(device_kind="fake")]))
    wrap._gpu_name.cache_clear()
    try:
        assert wrap._gpu_name() == wrap._gpu_name() == "fake" and calls == [1]
    finally:
        wrap._gpu_name.cache_clear()


def test_serve_root_override(tmp_path, monkeypatch):
    root = tmp_path / "fake"
    (root / "scripts").mkdir(parents=True)
    (root / wrap.SERVE_REL).write_text("VALUE = 7\n")
    for flags in ([wrap.SERVE_ROOT_FLAG, str(root)], [f"{wrap.SERVE_ROOT_FLAG}={root}"]):
        assert wrap.split_serve_root([*flags, "--seed=7"]) == (str(root), ["--seed=7"])
    assert wrap.split_wrapper_args(["--sgeval-metadata-out=m.json", "--seed=7"]) == ("m.json", ["--seed=7"])
    assert wrap.mme_vla_root(str(root)) == root
    monkeypatch.setitem(sys.modules, "serve_policy", None)
    assert Path(wrap.load_serve_policy(root).__file__) == root / wrap.SERVE_REL
    monkeypatch.chdir(root)
    assert wrap.mme_vla_root() == root
    monkeypatch.chdir(tmp_path)
    fallback = tmp_path / "third_party" / "mme-vla"
    (fallback / "scripts").mkdir(parents=True)
    (fallback / wrap.SERVE_REL).write_text("VALUE = 8\n")
    monkeypatch.setenv("SGEVAL_THIRD_PARTY", str(fallback.parent))
    assert wrap.mme_vla_root() == fallback
    with pytest.raises(FileNotFoundError):
        wrap.mme_vla_root(str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        wrap.split_serve_root([wrap.SERVE_ROOT_FLAG])
    parsed = []
    sp = types.SimpleNamespace(Args=object)
    monkeypatch.setattr(wrap, "prepare_sys_path", lambda _: None)
    monkeypatch.setattr(wrap, "load_serve_policy", lambda actual: sp if actual == root else None)
    monkeypatch.setitem(sys.modules, "tyro", types.SimpleNamespace(cli=lambda cls, args: parsed.append(args) or "args"))
    monkeypatch.setattr(wrap, "run", lambda *a, **k: k)
    result = wrap.main(["wrapper", wrap.SERVE_ROOT_FLAG, str(root), "--sgeval-metadata-out=m.json", "--seed=7"])
    assert parsed == [["--seed=7"]] and result["serve_file"] == root / wrap.SERVE_REL


def test_obs_eq_audit_off(monkeypatch):
    from openpi_client import msgpack_numpy

    policy = types.SimpleNamespace(infer=lambda obs: {"actions": ACTIONS})
    args = types.SimpleNamespace(seed=7)
    sp = types.SimpleNamespace(create_policy=lambda _: policy)
    sp.main = lambda actual: sp.create_policy(actual)
    monkeypatch.setenv("SGEVAL_AUDIT", "0")
    off = wrap.run(sp, args, meta_path=None, argv_full=[])
    assert off is policy
    plain = msgpack_numpy.Packer().pack(policy.infer({}))
    assert msgpack_numpy.Packer().pack(off.infer({})) == plain
    monkeypatch.setenv("SGEVAL_AUDIT", "1")
    monkeypatch.setattr(wrap, "install_tokenizer_spy", lambda _: None)
    monkeypatch.setattr(wrap, "_gpu_name", lambda: "fake")
    on = wrap.run(sp, args, meta_path=None, argv_full=[], tokenizer_cls=object)
    reply = on.infer({})
    assert set(reply) == {"actions", wrap.AUDIT_KEY}
    reply.pop(wrap.AUDIT_KEY)
    assert msgpack_numpy.Packer().pack(reply) == plain
    print("OBS_EQ=PASS route=policy_server_wrap audit_off_bytes_equal=1 audit_off_extra_keys=0 audit_on_action_bytes_equal=1")
