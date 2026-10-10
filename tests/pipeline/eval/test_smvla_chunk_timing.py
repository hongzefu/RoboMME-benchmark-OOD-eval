"""以真实两端包装器、假会话及假服务覆盖分块计时；不运行仿真。"""
import copy
import time
import types

import numpy as np
import pytest

import eval_fakes as F
from robomme_ood_eval.timing import ChunkTimer, attach_policy_timing


class Session:
    def __init__(self, plan):
        self.env = F.FakeEnv("T", 4, plan)
        self.steps = 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        self.steps += 1
        return self.env.step(action)

    def close(self):
        pass


class Conn(F.FakeSmvlaConn):
    def __init__(self, *, first_obs=.02, fail_infer=None, bad=None, audit=True):
        super().__init__(F.FakePolicyServer())
        self.metadata["gpu_name"] = "CPU-test"
        self.first_obs = first_obs
        self.infers = 0
        self.observes = 0
        self.fail_infer = fail_infer
        self.bad = bad
        self.audit = audit

    def call(self, msg):
        kind = next(iter(msg))
        if kind == "observe":
            self.observes += 1
            time.sleep(self.first_obs if self.observes == 1 else .02)
        if kind == "infer":
            self.infers += 1
            time.sleep(.05)
            if self.infers == self.fail_infer:
                raise F.smvla_client().ServerError("假推理错误")
        reply, raw, rep_raw = super().call(msg)
        if kind == "infer":
            reply["infer_ms"] = 40
            if self.audit:
                reply["_sgeval_audit"] = {"timing": {"lang_ms": 30, "action_ms": 10}}
            if self.infers == 2 and self.bad == "missing":
                reply.pop("actions")
            if self.infers == 2 and self.bad == "mismatch":
                reply["actions"] = reply["actions"] + 1
        return reply, raw, rep_raw


def episode(*, conn=None, plan=None):
    conn = conn or Conn()
    out = F.smvla_client().run_episode(Session(plan or F.Plan(success_at=40)),
                                     {"task": "T", "seed": 9, "source_episode": None}, {},
                                     conn=conn, record_frames=False)
    return out, conn


def test_chunk_per_decision():
    out, conn = episode()
    chunks = out["timing"]["chunks"]
    assert out["status"] == "success", out["error"]
    assert len(chunks) == conn.infers == 3
    minimum = min(c["decision_wall_ms"] for c in chunks)
    assert minimum >= 70
    assert len(out["timing"]["infer_ms"]) == len(out["timing"]["rtt_ms"]) == 3
    assert len(out["timing"]["observe_ms"]) == 4
    print(f"CHUNK_TIMING=PASS model=smvla decisions=3 missing=0 expected=3 min_wall_ms={minimum:.3f} min_wall_ok=1 first_includes_observe=1")


def test_first_decision_starts_before_reset_observe():
    out, _ = episode(conn=Conn(first_obs=.2))
    first, second = out["timing"]["chunks"][:2]
    assert first["decision_wall_ms"] >= 250
    assert first["observe_rtt_ms"] >= 200
    assert second["observe_rtt_ms"] < 100


class Head:
    def __init__(self):
        self.calls = []

    def sample(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        time.sleep(.01)
        return np.arange(160, dtype=np.float32).reshape(20, 8)


class Batched:
    def __init__(self):
        self.model = types.SimpleNamespace(action_head=Head())

    def generate_batch(self, processed, states):
        time.sleep(.03)
        return [(self.model.action_head.sample("x", num_steps=3, state=None), "sub")]


def host():
    srv = F.smvla_server()
    h = object.__new__(srv.SMVLAPolicyHost)
    h.batched = Batched()
    h.state_norm = lambda state: None
    return h, types.SimpleNamespace(_prepare_inputs=lambda instruction: instruction)


def test_lang_split_real_proxy(monkeypatch):
    monkeypatch.setenv("SGEVAL_AUDIT", "1")
    h, buf = host()
    assert h._install_timed_sample() is True
    reply = h.infer(buf, "goal", np.zeros(8, np.float32))
    split = reply["_sgeval_audit"]["timing"]
    assert split["lang_ms"] == pytest.approx(30, abs=5)
    assert split["action_ms"] == pytest.approx(10, abs=5)
    delta = abs(split["lang_ms"] + split["action_ms"] - reply["infer_ms"]) / reply["infer_ms"] * 100
    assert delta < 1 and split["sample_calls"] == 1
    assert h.batched.model.action_head.calls == [(("x",), {"num_steps": 3, "state": None})]
    print(f"LANG_SPLIT=PASS model=smvla calls=1 delta_pct={delta:.6f} delta_ok=1 kwargs_passthrough=1")
    monkeypatch.setenv("SGEVAL_AUDIT", "0")
    h, buf = host()
    assert h._install_timed_sample() is False
    assert "_sgeval_audit" not in h.infer(buf, "goal", np.zeros(8, np.float32))


def test_audit_off_no_proxy_bytes_identical(monkeypatch):
    from openpi_client import msgpack_numpy

    srv = F.smvla_server()
    monkeypatch.setenv("SGEVAL_AUDIT", "0")
    h, buf = host()
    orig = h.batched.model.action_head.sample
    # 固化时钟使旧包装器与新包装器的计时值可直接逐字节比较。
    h.batched.model.action_head.sample = orig
    monkeypatch.setattr(srv.time, "monotonic", lambda: 123.0)
    assert h._install_timed_sample() is False
    assert h.batched.model.action_head.sample is orig and not hasattr(h, "_split")
    state = np.zeros(8, np.float32)
    got = h.infer(buf, "goal", state)
    actions = np.arange(160, dtype=np.float32).reshape(20, 8)
    expected = {"actions": actions[:16], "actions_full": actions, "subtask": "sub", "infer_ms": 0.0,
                "recv_state_sha": srv.array_sha(state), "recv_instruction_sha": srv.sha256_bytes(b"goal")}
    packer = msgpack_numpy.Packer()
    assert set(got) == set(expected)
    assert packer.pack(got) == packer.pack(expected)
    print("OBS_EQ=PASS route=smvla_server audit_off_identical=1 proxy_installed=0")
    monkeypatch.setenv("SGEVAL_AUDIT", "1")
    h, _ = host()
    assert h._install_timed_sample() is True
    assert isinstance(h.batched.model.action_head.sample, srv._TimedSample)


def test_wall_decomposition():
    out, _ = episode()
    timing = out["timing"]
    violations = sum(c["decision_wall_ms"] < c["observe_rtt_ms"] + c["action_rtt_ms"] - 1
                     or c["lang_ms"] > c["server_infer_ms"] + 1
                     or c["server_infer_ms"] > c["action_rtt_ms"] + 1 for c in timing["chunks"])
    c = timing["conservation"]
    assert violations == c["violations"] == c["missing_fields"] == 0
    assert c["expected"] == c["checked"] == len(timing["chunks"])
    t = ChunkTimer()
    t.chunks = copy.deepcopy(timing["chunks"])
    t.chunks[0].pop("observe_rtt_ms")
    bad = attach_policy_timing({}, t, gpu_name=None, model_kind="smvla")["conservation"]
    assert bad["missing_fields"] == bad["violations"] == 1
    print(f"WALL_DECOMP=PASS model=smvla kind=smvla chunks=3 violations=0 overhead_pct_p50={c['overhead_pct_p50']:.6f} order_violations=0 missing_fields=0 checked=3 expected=3")


def test_infer_error_no_chunk():
    out, _ = episode(conn=Conn(fail_infer=3))
    assert out["status"] == "error"
    assert len(out["timing"]["chunks"]) == 2 and out["timing"]["orphan_lang_ms"] == 0


def test_legacy_lists_kept_and_audit_unavailable():
    out, _ = episode(conn=Conn(audit=False))
    t = out["timing"]
    assert len(t["infer_ms"]) == len(t["rtt_ms"]) == 3 and len(t["observe_ms"]) == 4
    assert t["lang_calls"] == [] and not t["lang_split_available"]
    assert all(c["lang_ms"] == 0 for c in t["chunks"])


def test_step_error_episode():
    out, conn = episode(plan=F.Plan(raise_at=33, raise_exc=lambda: RuntimeError("假步错误")))
    assert out["status"] == "error" and out["steps"] == 32
    assert len(out["timing"]["chunks"]) == conn.infers == 3
    assert not out["timing"]["dangling_start"]


@pytest.mark.parametrize("bad", ["missing", "mismatch"])
def test_protocol_failure_no_chunk(bad):
    out, _ = episode(conn=Conn(bad=bad))
    assert out["status"] == "error" and out["protocol"]["broken"]
    assert len(out["timing"]["chunks"]) == 1


def test_reply_clock_excludes_ledger(monkeypatch):
    sm = F.smvla_client()
    orig = sm.PolicyTrace.lang_audit

    def slow_audit(self, *args, **kwargs):
        time.sleep(.1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(sm.PolicyTrace, "lang_audit", slow_audit)
    out, _ = episode(plan=F.Plan(success_at=1))
    assert out["timing"]["chunks"][0]["decision_wall_ms"] < 150


def test_failed_calls_and_dangling():
    out, _ = episode(conn=Conn(fail_infer=3))
    failed = out["timing"]["failed_calls"]
    assert len(failed) == 1 and failed[0]["kind"] == "infer" and failed[0]["error"]
    assert out["timing"]["chunk_summary"]["n_decisions"] == 2
    normal, _ = episode()
    assert normal["timing"]["dangling_start"] and normal["timing"]["conservation"]["missing_fields"] == 0
