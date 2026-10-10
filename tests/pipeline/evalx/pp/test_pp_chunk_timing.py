"""PP 块计时：真实第三方调度、CPU 假模型、独立节拍与字节基准。"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

from test_pp_server_wrap import _child_env, _pp_python, _pp_root, pytest

TREE = Path(__file__).resolve().parents[4]
SNAPSHOT = Path(__file__).parent / "fixtures" / "pp_server_wrap_pre_timing.py"
BASE = "3560a7a3d598dfdb1c6d2e76070b85c7351b676a"
SCHEDULES = [
    {"name": "main", "h": 5, "p1": 5, "p2": 7, "delay": 0, "steps": 40},
    {"name": "orphan", "h": 5, "p1": 5, "p2": 7, "delay": 0, "steps": 9},
    {"name": "hold", "h": 5, "p1": 5, "p2": 7, "delay": 2, "steps": 11},
    {"name": "h1", "h": 1, "p1": 1, "p2": 7, "delay": 0, "steps": 7},
    {"name": "h1_stale", "h": 1, "p1": 3, "p2": 7, "delay": 0, "steps": 8},
    {"name": "fresh_last", "h": 5, "p1": 5, "p2": 7, "delay": 0, "steps": 11},
]


def _child_main():
    import types

    import numpy as np
    import torch
    import ponderpounce.eval.robomme_server as rs
    from robomme_ood_eval.models.pp import TracedConnection
    from robomme_ood_eval.timing import attach_policy_timing

    assert torch.cuda.is_available() is False
    assert Path(sys.modules[TracedConnection.__module__].__file__).resolve() == TREE / "src/robomme_ood_eval/models/pp.py"

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    wrap = load(TREE / "src/robomme_ood_eval/servers/pp_server_wrap.py", "timed_wrap")
    old = load(SNAPSHOT, "old_wrap")

    class StubS2Context:
        def __init__(self, **kwargs):
            time.sleep(.050)
            self.n = 0

        def fire(self, pils):
            time.sleep(.080)
            n = self.n
            self.n += 1
            return rs.S2ContextResult(subgoal_tokens=None, subgoal_text=f"目标{n}",
                                      cognition=torch.full((2, 4), float(n)), kind="stub",
                                      gate_score=None, n_input_frames=1)

    class StubS1:
        def __init__(self, h):
            self.action_chunk_size, self.action_dim, self.noise_spec = h, 8, (h, 8)
            self.null_cognition = torch.zeros(2, 4)

        def predict_action(self, **kw):
            time.sleep(.030)
            return (kw["noise"][0] * .05 + kw["cognition"].mean() * .1).unsqueeze(0)

    def transform(pil):
        return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255).permute(2, 0, 1)

    def build(cls, sch):
        srv = cls.__new__(cls)
        super(rs.PonderPounceRoboMMEServer, srv).__init__()
        srv._device, srv._dtype = torch.device("cpu"), torch.float32
        srv._dt_ns = 50 * rs.NS_PER_MS
        srv._s1_period_ns, srv._s2_period_ns = sch["p1"] * srv._dt_ns, sch["p2"] * srv._dt_ns
        srv._wait_first_cognition = bool(sch["delay"])
        srv._camera_keys = ("agentview", "wrist")
        srv._seed = 0
        srv._episodes, srv._episode_counts = {}, {}
        srv._s2_delay_ns = sch["delay"] * srv._dt_ns
        srv._model, srv._norm_stats = None, None
        srv._s1 = StubS1(sch["h"])
        srv._s2 = types.SimpleNamespace(num_cognition_tokens=2)
        srv._s2_processor, srv._s2_max_new_tokens, srv._subgoal_grounded = None, 40, False
        srv._s1_transform = transform
        srv._adapter = rs.ObsAdapter(camera_keys=srv._camera_keys, proprio_dim=8, max_demo_frames=0,
                                     norm_stats=None, demo_fps=0, env_fps=20)
        srv._s1_tokenizer, srv._s1_max_token_len, srv._s1_num_camera_slots = None, 0, 0
        return srv

    class Ctx:
        session_id = "T|0|0"

        async def send_action(self, action):
            self.action = action

    def arr(a):
        a = np.ascontiguousarray(a)
        return [a.dtype.str, list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]

    async def run(cls, sch, enabled):
        os.environ["SGEVAL_AUDIT"] = "1" if enabled else "0"
        srv, ctx = build(cls, sch), Ctx()
        await srv.on_episode_start({"task": {"name": "T"}}, ctx)
        states, frames, audit = [], [], []

        class Conn:
            async def act(self, obs):
                await srv.on_observation(obs, ctx)
                frames.append(pickle.dumps(ctx.action, protocol=5).hex())
                audit.append(ctx.action.get("_sgeval_audit"))
                return dict(ctx.action)

        tc = TracedConnection(Conn())
        for step in range(sch["steps"]):
            obs = {"images": {cam: np.full((8, 8, 3), step, np.uint8) for cam in srv._camera_keys},
                   "states": np.arange(8, dtype=np.float32), "task_description": "pick up the cube"}
            action = await tc.act(obs)
            ep = srv._episodes[ctx.session_id]
            if not enabled:
                assert not hasattr(ep, "_sgeval_s2_ms") and not hasattr(ep, "_sgeval_s2_fired")
                assert ep.chunk is None or not hasattr(ep.chunk, "_sgeval_s1_ms")
            states.append({"actions": arr(action["actions"]), "cursor": ep.chunk.cursor if ep.chunk else None,
                           "chunk": arr(ep.chunk.actions.numpy()) if ep.chunk else None,
                           "rng": arr(ep.noise_rng.get_state().numpy()),
                           "n_s1": ep.n_s1_fires, "n_s2": ep.n_s2_fires, "tick": ep.tick,
                           "s1_next": ep.s1_next_fire_ns, "s2_next": ep.s2_next_fire_ns})
        timing = {}
        attach_policy_timing(timing, tc.timer, gpu_name=tc.gpu_name, model_kind="pp")
        timing.update(step_rtt_ms=tc.step_rtt_ms, fresh_mismatch=tc.fresh_mismatch,
                      pp_timing_missing=tc.timing_missing)
        await srv.on_episode_end({}, ctx)
        return {"states": states, "frames": frames, "audit": audit, "timing": timing}

    rs.SoftS2SessionContext = StubS2Context
    records = []
    for sch in SCHEDULES:
        parent = asyncio.run(run(rs.PonderPounceRoboMMEServer, sch, False))
        on = asyncio.run(run(wrap.SubgoalReportingServer, sch, True))
        off = asyncio.run(run(wrap.SubgoalReportingServer, sch, False))
        previous = asyncio.run(run(old.SubgoalReportingServer, sch, False))
        assert parent["states"] == on["states"] == off["states"] == previous["states"]
        assert off["frames"] == previous["frames"]
        assert all(a is None for a in off["audit"])
        # 审计关不得引入计时私有属性或调用同步、硬件查询。
        def forbidden():
            raise AssertionError("审计关调用了计时同步或硬件查询")
        original_sync, original_gpu = wrap._cuda_sync, wrap._gpu_name
        wrap._cuda_sync = wrap._gpu_name = forbidden
        try:
            off_again = asyncio.run(run(wrap.SubgoalReportingServer, sch, False))
            assert off_again["frames"] == previous["frames"]
        finally:
            wrap._cuda_sync, wrap._gpu_name = original_sync, original_gpu
        records.append({"schedule": sch, "on": on, "off_timing": off["timing"]})
    print("CHILD_RESULT " + json.dumps(records), flush=True)


_CACHE = None


def _records():
    global _CACHE
    if _CACHE is None:
        out = subprocess.run([_pp_python(), str(Path(__file__).resolve()), "--child"], cwd=TREE,
                             env=_child_env(), capture_output=True, text=True, timeout=180)
        lines = [line for line in out.stdout.splitlines() if line.startswith("CHILD_RESULT ")]
        assert out.returncode == 0 and lines, f"子进程失败 rc={out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-4000:]}"
        _CACHE = json.loads(lines[-1].removeprefix("CHILD_RESULT "))
    return _CACHE


def test_snapshot_is_fixed_original():
    out = subprocess.run(["git", "show", f"{BASE}:src/robomme_ood_eval/servers/pp_server_wrap.py"],
                         cwd=TREE, capture_output=True, timeout=20)
    assert out.returncode == 0
    assert SNAPSHOT.read_bytes().split(b"\n", 2)[2].rstrip(b"\n") == out.stdout.rstrip(b"\n")


def test_chunk_fresh_timing_language_conservation_and_obs_eq():
    calls = decisions = 0
    max_resid = 0.0
    for rec in _records():
        sch, on = rec["schedule"], rec["on"]
        timing, states = on["timing"], on["states"]
        chunks = timing["chunks"]
        # 从节拍独立推导：不使用被测包装器的 fresh 标记。
        first = 1 + sch["delay"]
        fresh_ticks = list(range(first, sch["steps"] + 1, sch["p1"]))
        s2_ticks = list(range(1, sch["steps"] + 1, sch["p2"]))
        assert [c["env_step"] + 1 for c in chunks] == fresh_ticks
        assert len(chunks) == states[-1]["n_s1"] == len(fresh_ticks)
        assert len(timing["lang_calls"]) == states[-1]["n_s2"] == len(s2_ticks)
        assert timing["pp_timing_missing"] == 0
        assert len(timing["step_rtt_ms"]) == sch["steps"]
        assert timing["dangling_start"] is (sch["steps"] not in fresh_ticks)
        want_mismatch = sch["steps"] - len(fresh_ticks) if sch["h"] == 1 else 0
        assert timing["fresh_mismatch"] == want_mismatch
        previous_tick = 0
        for c, tick in zip(chunks, fresh_ticks):
            expected_calls = [t for t in s2_ticks if previous_tick < t <= tick]
            expected_ms = 80 * len(expected_calls) + (50 if 1 in expected_calls else 0)
            assert c["n_lang"] == len(expected_calls)
            assert abs(c["lang_ms"] - expected_ms) < 20
            assert c["server_infer_ms"] >= 29
            assert abs(c["decision_wall_ms"] - c["step_rtt_ms"] - c["bucket_ms"]) <= .003
            assert abs(c["action_rtt_ms"] - c["step_rtt_ms"] + c["s2_infer_ms_at_fresh"]) <= .003
            resid = abs(c["decision_wall_ms"] - c["action_rtt_ms"] - c["lang_ms"])
            assert resid <= .003
            max_resid = max(max_resid, resid)
            previous_tick = tick
        lang_total = sum(c["ms"] for c in timing["lang_calls"])
        assigned = sum(c["lang_ms"] for c in chunks) + timing["orphan_lang_ms"]
        delta = abs(assigned - lang_total)
        assert delta <= .003
        conservation = timing["conservation"]
        assert conservation["violations"] == conservation["missing_fields"] == 0
        assert conservation["checked"] == len(chunks)
        assert abs(conservation["overhead_pct_p50"]) <= .001
        for tick in s2_ticks:
            audit = on["audit"][tick - 1]
            outer = audit["pp_timing"]["s2_infer_ms"]
            inner = audit["pp_generation"][0]["s2_fire_ms"]
            assert inner <= outer
            assert abs(inner - 80) < 20
            if tick == 1:
                assert outer - inner >= 45
        if sch["name"] == "hold":
            assert all(not a["pp_timing"]["chunk_fresh"] for a in on["audit"][:2])
            assert chunks[0]["s2_infer_ms_at_fresh"] == 0
            assert 125 <= chunks[0]["lang_ms"] < 150
        if sch["name"] == "orphan":
            assert 75 <= timing["orphan_lang_ms"] < 100
        off = rec["off_timing"]
        assert off["chunks"] == [] and len(off["step_rtt_ms"]) == sch["steps"]
        assert off["pp_timing_missing"] == sch["steps"]
        decisions += len(chunks)
        calls += len(timing["lang_calls"])
        print(f"PP_CHUNK_FRESH=PASS period={sch['p1']} fires={len(chunks)} schedule={sch['name']}")
        print(f"LANG_SPLIT=PASS model=pp calls={len(timing['lang_calls'])} delta_pct={delta / lang_total * 100:.6f} "
              f"orphan_ms={timing['orphan_lang_ms']} delta_ms={delta:.6f} schedule={sch['name']}")
    print(f"CHUNK_TIMING=PASS model=pp decisions={decisions} missing=0")
    print(f"WALL_DECOMP=PASS model=pp kind=pp chunks={decisions} violations=0 "
          f"overhead_pct_p50=0 max_resid_ms={max_resid:.6f}")
    print(f"OBS_EQ=PASS schedules={len(SCHEDULES)} audit_on_off_mismatch=0 off_bytes_equal={len(SCHEDULES)}")


def test_episode_reconnect_and_exception_keep_timing():
    from pp_fakes import FakeConn, FakeEnv, FakeSession
    from robomme_ood_eval.models import pp

    class AuditedConn(FakeConn):
        async def act(self, obs):
            action = await super().act(obs)
            time.sleep(.002)
            action[pp.AUDIT_KEY] = {"pp_timing": {
                "chunk_fresh": self.n_actions % 2 == 1, "n_s1_fires": (self.n_actions + 1) // 2,
                "s1_infer_ms": .2, "s2_fired": True, "s2_infer_ms": .5,
                "n_s2_fires": self.n_actions, "gpu": None}}
            return action

    identity = {"task": "VideoUnmask", "tier": "xhard1", "seed": 0, "builder_episode": 0}
    conn = AuditedConn(closed_at={3})
    result = pp.run_episode(FakeSession(FakeEnv("VideoUnmask", 0, done_at=5)), identity,
                            {"port": 1, "max_steps": 5}, None, connection_factory=lambda *a, **k: conn)
    assert result["status"] == "success", result
    timing = result["timing"]
    assert timing["reconnect_steps"] == [3] and timing["reconnected"] is True
    assert len(timing["step_rtt_ms"]) == result["decisions"] == 5
    assert len(timing["chunks"]) == 3 and timing["pp_timing_missing"] == 0
    assert len(timing["lang_calls"]) == 5
    conn = AuditedConn(raise_at={4: RuntimeError("预期失败")})
    result = pp.run_episode(FakeSession(FakeEnv("VideoUnmask", 0)), identity,
                            {"port": 1, "max_steps": 5}, None, connection_factory=lambda *a, **k: conn)
    assert result["status"] == "error" and result["decisions"] == 4
    assert len(result["timing"]["step_rtt_ms"]) == 4
    assert len(result["timing"]["chunks"]) == 2
    assert result["timing"]["orphan_lang_ms"] == .5
    assert result["timing"]["reconnected"] is False
    print("CHUNK_TIMING=PASS model=pp reconnect_steps=3 retry_rtt_excluded=1 exception_preserved=1")


if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    _child_main()
