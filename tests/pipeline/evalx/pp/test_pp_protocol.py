"""PonderPounce 新侧客户端与原侧驱动（S4）的日常门禁用例：全部 CPU 替身、无网络、无仿真。

期望消息序列一律手写：依据 vla-eval 0.7.0 源码事实（``SyncEpisodeRunner.run_episode`` 的帧顺序、
``RoboMMEBenchmark.make_obs`` 的观测键与 dtype、``RoboMMEBenchmark.step`` 的展平取动作），不调用被测代码生成期望。
用真实 vla-eval 源码跑 ``SyncEpisodeRunner`` 的对拍在同目录 ``test_pp_protocol_wire.py``（slow）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pp_fakes import (PP_BENCHMARK_LITERAL, REPO, TASK_GOAL, FakeConn, FakeEnv, FakeSession, action_for,
                      compare_frames, sha256_file)
from tests._support.loaders import load_script, script_path

pp = load_script("eval-official/pp_client.py")
orig = load_script("eval-official/pp_official_runner.py")
tw = load_script("eval-official/trace_writer.py")

TASK = "PickXtimes"
SRC, SEED = 7, 123457
XH = {"task": TASK, "tier": "xhard0", "seed": SEED, "source_episode": SRC, "builder_episode": 1,
      "key": f"{TASK}_xhard0_{SEED}", "candidate": None, "spec_sha256": None}
SID = f"{TASK}|{SRC}|{SEED}"


def run_new(env: FakeEnv, *, max_steps: int, ident: dict = XH, conn: FakeConn | None = None, step_cap=None,
            conn_info_extra: dict | None = None, recorder=None):
    conn = conn or FakeConn()
    made = {}

    def factory(url, timeout):
        made.update(url=url, timeout=timeout)
        return conn

    session = FakeSession(env, step_cap=step_cap)
    ci = {"host": "127.0.0.1", "port": 18310, "max_steps": max_steps, **(conn_info_extra or {})}
    res = pp.run_episode(session, ident, ci, recorder, connection_factory=factory)
    return res, conn, session, made


def hand_expected(env: FakeEnv, n_actions: int, *, success: bool, end: bool = True, sid: str = SID,
                  episode_idx: int = SRC) -> list:
    """按源码事实手写的期望帧序列（HELLO → EPISODE_START → (OBSERVATION, ACTION)×n → EPISODE_END）。"""
    d = env.demo

    def obs_at(k: int, first: bool) -> dict:
        f = env.frames_at(k)
        o = {"images": {"agentview": f["front"], "wrist": f["wrist"]}, "task_description": TASK_GOAL,
             "states": np.concatenate([f["joint"], f["grip"][:1]]).astype(np.float32)}
        if first:
            o["video_history"] = [env.frames_at(j)["front"] for j in range(d)]
            o["wrist_video_history"] = [env.frames_at(j)["wrist"] for j in range(d)]
            o["episode_restart"] = True
        return o

    frames = [("hello", {"benchmark": PP_BENCHMARK_LITERAL}),
              ("episode_start", {"task": {"name": TASK, "env_id": TASK, "episode_idx": episode_idx},
                                 "recording": {"sid": sid, "eid": sid, "eval_id": "", "db_path": ""}})]
    for i in range(n_actions):
        frames += [("observation", obs_at(d + i, i == 0)), ("action", {"actions": action_for(i)})]
    if end:
        frames.append(("episode_end", {"metrics": {"success": success}, "steps": n_actions, "elapsed_sec": 0.0}))
    return frames


def test_protocol_sequence_matches_handwritten_sync_runner_order():
    env = FakeEnv(TASK, SRC, demo=3, done_at=5)
    res, conn, session, made = run_new(env, max_steps=1300)
    want = hand_expected(FakeEnv(TASK, SRC, demo=3), 5, success=True)
    frames_diff, order_diff, notes = compare_frames(conn.log, want)
    assert (frames_diff, order_diff) == (0, 0), notes
    assert made == {"url": "ws://127.0.0.1:18310", "timeout": 300.0}
    # 交给环境的动作：展平后的 Python float 列表前 8 维
    assert len(env.actions) == 5
    for i, a in enumerate(env.actions):
        assert isinstance(a, list) and all(type(x) is float for x in a)
        assert a == [float(x) for x in action_for(i).flatten()[:8]]
    assert res["status"] == "success" and res["task_success"] is True and res["steps"] == 5
    assert res["sid"] == SID and res["eid"] == SID and res["demo_frames"] == 3 and res["frames_sent"] == 7
    assert conn.closed == 1
    print(f"PP_PROTOCOL=PASS frames_diff={frames_diff} order_diff={order_diff} source=handwritten frames={len(want)}")


def test_frame_comparator_is_not_vacuous():
    """比对器自检：dtype、shape、键、sid、帧序任一改动都必须被计数。"""
    env = FakeEnv(TASK, SRC, demo=3, done_at=2)
    _, conn, _, _ = run_new(env, max_steps=1300)
    base = hand_expected(FakeEnv(TASK, SRC, demo=3), 2, success=True)
    assert compare_frames(conn.log, base)[:2] == (0, 0)
    bad = [list(f) for f in base]
    bad[2][1] = dict(bad[2][1], states=bad[2][1]["states"].astype(np.float64))
    assert compare_frames(conn.log, [tuple(f) for f in bad])[0] == 1
    bad = [list(f) for f in base]
    bad[1][1] = {"task": bad[1][1]["task"], "recording": dict(bad[1][1]["recording"], sid="other")}
    assert compare_frames(conn.log, [tuple(f) for f in bad])[0] == 1
    bad = [list(f) for f in base]
    bad[2][1] = {k: v for k, v in bad[2][1].items() if k != "episode_restart"}
    assert compare_frames(conn.log, [tuple(f) for f in bad])[0] == 1
    swapped = base[:2] + [base[3], base[2]] + base[4:]
    assert compare_frames(conn.log, swapped)[1] == 2
    assert compare_frames(conn.log, base[:-1])[1] == 1


def test_xhard0_exactly_1300_actions_and_no_trailing_observation():
    env = FakeEnv(TASK, SRC, demo=2, done_at=None)
    res, conn, session, _ = run_new(env, max_steps=1300)
    kinds = [t for t, _ in conn.log]
    assert kinds.count("observation") == 1300 and kinds.count("action") == 1300
    assert kinds[-2:] == ["action", "episode_end"]
    assert len(env.actions) == 1300 and session.steps == 1300
    assert conn.log[-1][1]["steps"] == 1300 and conn.log[-1][1]["metrics"] == {"success": False}
    assert res["status"] == "timeout" and res["steps"] == 1300 and res["decisions"] == 1300


def test_v9_strict_cap_loop_matches_cap_and_cap_hit_is_timeout_without_episode_end():
    env = FakeEnv(TASK, SRC, demo=1)
    v9 = dict(XH, tier="xhard1", source_episode=3)
    res, conn, session, _ = run_new(env, max_steps=1600, ident=v9, step_cap=1600)
    assert len(env.actions) == 1600 and not session.cap_hit and res["status"] == "timeout"
    assert conn.log[1][1]["recording"]["sid"] == f"{TASK}|xhard1|{SEED}"
    # step_cap 小于循环上限（配置错配）：第 cap+1 次 step 不进环境，记 timeout，不发 EPISODE_END
    env2 = FakeEnv(TASK, SRC, demo=1)
    res2, conn2, s2, _ = run_new(env2, max_steps=10, ident=v9, step_cap=4)
    assert len(env2.actions) == 4 and s2.cap_hit and res2["status"] == "timeout" and res2["error"] is None
    assert "episode_end" not in [t for t, _ in conn2.log]


def test_fixed_sid_rules():
    assert pp.fixed_sid(XH) == SID
    assert pp.fixed_sid(XH, "hard-verify") == SID
    assert pp.fixed_sid(dict(XH, tier="xhard2", source_episode=11), "ood") == f"{TASK}|xhard2|{SEED}"
    with pytest.raises(ValueError):
        pp.fixed_sid(dict(XH, tier="xhard1"), "hard-verify")
    with pytest.raises(ValueError):
        pp.fixed_sid(dict(XH, source_episode=True))
    with pytest.raises(ValueError):
        pp.fixed_sid(dict(XH, source_episode=None))
    # 局号：两侧都发官方局号 source_episode（不是 builder_episode）
    env = FakeEnv(TASK, SRC, done_at=1)
    _, conn, _, _ = run_new(env, max_steps=5, conn_info_extra={"dataset": "hard-verify"})
    start = conn.log[1][1]
    assert start == {"task": {"name": TASK, "env_id": TASK, "episode_idx": SRC},
                     "recording": {"sid": SID, "eid": SID, "eval_id": "", "db_path": ""}}
    print("PP_FIXED_SID=PASS xhard0=task|source_episode|seed v9=task|tier|seed")


def test_reconnect_within_episode_never_resends_episode_start():
    env = FakeEnv(TASK, SRC, demo=2, done_at=6)
    conn = FakeConn(closed_at={3})
    res, conn, _, _ = run_new(env, max_steps=1300, conn=conn)
    kinds = [t for t, _ in conn.log]
    assert kinds.count("episode_start") == 1 and conn.reconnects == 1 and res["reconnects"] == 1
    obs = [p for t, p in conn.log if t == "observation"]
    assert len(obs) == 7  # 第 4 条观测（下标 3）断线后原样重发一次
    assert obs[3] is obs[4]
    assert res["status"] == "success" and len(env.actions) == 6
    # 断线次数超过上限：本局 error（基础设施），不发 EPISODE_END
    env2 = FakeEnv(TASK, SRC, demo=2, done_at=6)
    res2, conn2, _, _ = run_new(env2, max_steps=1300, conn=FakeConn(closed_at={2, 3}))
    assert res2["status"] == "error" and res2["infra"] is True and res2["infra_reason"] == "pp_connection_closed"
    assert [t for t, _ in conn2.log].count("episode_start") == 1 and "episode_end" not in [t for t, _ in conn2.log]
    print("PP_RECONNECT=PASS episode_start_resent=0")


def test_server_error_and_timeout_are_infra_errors():
    for exc, reason in ((TimeoutError("act"), "pp_act_timeout"), (RuntimeError("Server error: {'x': 1}"),
                                                                    "pp_server_error")):
        env = FakeEnv(TASK, SRC, done_at=9)
        res, conn, _, _ = run_new(env, max_steps=20, conn=FakeConn(raise_at={2: exc}))
        assert res["status"] == "error" and res["infra"] is True and res["infra_reason"] == reason
        assert len(env.actions) == 2 and conn.closed == 1


def test_env_error_status_and_empty_obs_end_episode_like_official():
    env = FakeEnv(TASK, SRC, demo=1, none_obs_at=3)
    res, conn, _, _ = run_new(env, max_steps=50)
    assert len(env.actions) == 3 and res["status"] == "error" and res["error"] == "env_status=error"
    assert conn.log[-1][0] == "episode_end" and conn.log[-1][1]["steps"] == 3
    env2 = FakeEnv(TASK, SRC, demo=1, truncate_at=4)
    res2, _, _, _ = run_new(env2, max_steps=50)
    assert res2["status"] == "timeout" and len(env2.actions) == 4


def test_trace_rows_and_canonical_bytes(tmp_path):
    p = tmp_path / "ep" / "trace.jsonl"
    env = FakeEnv(TASK, SRC, demo=3, done_at=4)
    res, conn, _, _ = run_new(env, max_steps=1300, conn_info_extra={"trace_path": str(p)})
    rows = tw.read_trace(p)
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "header" and kinds[1] == "demo" and kinds[-1] == "end"
    assert rows[0]["route"] == "pp-new" and rows[0]["identity"]["sid"] == SID and rows[0]["max_steps"] == 1300
    assert rows[1]["frames"] == 3 and rows[1]["texts"] == [TASK_GOAL]
    reqs = [r for r in rows if r["kind"] == "request"]
    assert [r["name"] for r in reqs] == ["episode_start"] + ["observation"] * 4 + ["episode_end"]
    assert sum(r["kind"] == "response" for r in rows) == 4 and sum(r["kind"] == "step" for r in rows) == 4
    assert rows[-1]["exec_steps"] == 4 and rows[-1]["status"] == "success"
    step1 = [r for r in rows if r["kind"] == "step"][0]
    assert step1["action"]["dtype"] == "<f8" and step1["action"]["shape"] == [8]
    # 再跑一遍：墙钟不同，但规范化字节逐帧相同（EPISODE_END 去掉 elapsed_sec、键排序）
    p2 = tmp_path / "ep2" / "trace.jsonl"
    run_new(FakeEnv(TASK, SRC, demo=3, done_at=4), max_steps=1300, conn_info_extra={"trace_path": str(p2)})
    sha1 = [r["sha256"] for r in rows if r["kind"] == "request"]
    sha2 = [r["sha256"] for r in tw.read_trace(p2) if r["kind"] == "request"]
    assert sha1 == sha2
    # 键顺序不影响规范化字节；dtype 改变会影响
    a = {"x": np.zeros(2, np.float32), "y": 1}
    assert pp.canonical_frame_bytes("observation", a) == pp.canonical_frame_bytes("observation", {"y": 1, "x": a["x"]})
    assert pp.canonical_frame_bytes("observation", a) != pp.canonical_frame_bytes(
        "observation", {"x": np.zeros(2, np.float64), "y": 1})


def test_trace_path_resolution(tmp_path):
    class Rec:
        out_dir = tmp_path / "rec" / "K.a2"

    ep_dir = tmp_path / "tr" / f"{XH['key']}.a2"  # env_client 给的每局目录，不再套一层
    assert pp.resolve_trace_path(XH, {"trace_path": "/x/t.jsonl", "trace_dir": str(ep_dir)}, Rec()) == Path("/x/t.jsonl")
    assert pp.resolve_trace_path(XH, {"trace_dir": str(ep_dir)}, Rec()) == ep_dir / "trace.jsonl"
    assert pp.resolve_trace_path(XH, {"trace_dir": str(ep_dir)}, None) == ep_dir / "trace.jsonl"
    assert pp.resolve_trace_path(XH, {"trace_dir": None}, Rec()) == Rec.out_dir / "trace.jsonl"
    assert pp.resolve_trace_path(XH, {}, Rec()) == Rec.out_dir / "trace.jsonl"
    assert pp.resolve_trace_path(XH, {"trace_dir": None}, None) is None
    # 端到端：每局目录事先不存在，由写轨迹时建好父目录；目录下只有 trace.jsonl
    assert not ep_dir.exists()
    run_new(FakeEnv(TASK, SRC, done_at=2), max_steps=10,
            conn_info_extra={"trace_dir": str(ep_dir), "episode_tag": ep_dir.name})
    assert sorted(p.name for p in ep_dir.iterdir()) == ["trace.jsonl"]
    assert tw.read_trace(ep_dir / "trace.jsonl")[-1]["exec_steps"] == 2


def test_module_import_does_not_require_vla_eval():
    """拆仓后 pp 客户端是评估包模块 ``robomme_hard_eval.models.pp``：导入不需要 vla_eval，也不拉进仿真包。"""
    code = ("import sys; sys.modules['vla_eval'] = None; sys.path.insert(0, %r); import robomme_hard_eval.models.pp; "
            "assert 'robomme_hard' not in sys.modules; print('PP_IMPORT_OK')" % str(REPO / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "PP_IMPORT_OK" in out.stdout, out.stderr


def test_official_sources_pin_protocol_constants():
    """只读引用主检出子模块的官方源码（worktree 内子模块目录为空），打印所读文件的 sha256。"""
    root = Path(os.environ.get("SGEVAL_THIRD_PARTY", REPO / "third_party")) / "PonderPounce"
    cfg, server = root / "configs" / "robomme.yaml", root / "ponderpounce" / "eval" / "robomme_server.py"
    assert cfg.is_file() and server.is_file(), f"缺官方源码（设 SGEVAL_THIRD_PARTY）：{root}"
    for f in (cfg, server):
        print(f"PP_SOURCE {f} sha256={sha256_file(f)}")
    text = cfg.read_text(encoding="utf-8")
    assert "timeout: 300.0" in text and pp.PP_TIMEOUT_S == 300.0
    assert f"benchmark: {pp.PP_BENCHMARK}" in text and pp.PP_BENCHMARK == PP_BENCHMARK_LITERAL
    assert "max_steps: 1300" in text and "action_space: joint_angle" in text
    src = server.read_text(encoding="utf-8")
    assert 'zlib.crc32(f"{self._seed}:{sid}:{n}".encode())' in src
    assert "sid = ctx.session_id" in src and "self._episode_counts[sid] = n + 1" in src


# ── 原侧驱动（替身 benchmark 与手写的 SyncEpisodeRunner 流程） ─────────────


class _Res:
    def __init__(self, obs, info, done):
        self.obs, self.info, self.done = obs, info, done


class FakeRoboBench:
    """RoboMMEBenchmark 的最小替身：reset 关旧环境、建新环境；step 把展平列表交给 ``self._env.step``。"""

    env_kwargs: dict = {}
    built: list = []

    def __init__(self, tasks, action_space, max_steps):
        FakeRoboBench.built.append((tuple(tasks), action_space, max_steps))
        self._env = None
        self._task_description = ""
        self.cleaned = 0

    def reset(self, task):
        if self._env is not None:
            self._env.close()
        self._env = FakeEnv(task["env_id"], task["episode_idx"], **self.env_kwargs.get(task["episode_idx"], {}))
        obs, info = self._env.reset()
        self._task_description = info["task_goal"][0]
        return obs

    def step(self, action):
        obs, _r, term, trunc, info = self._env.step(list(action))
        return _Res(obs, info, bool(term) or bool(trunc) or info.get("status") == "error")

    def cleanup(self):
        self.cleaned += 1


class HandSyncRunner:
    """按 SyncEpisodeRunner 源码事实手写的流程（reset → EPISODE_START → act/step/break → EPISODE_END）。"""

    async def run_episode(self, bench, task, conn, *, max_steps, recorder):
        bench.reset(task)
        await conn.start_episode({"task": task, "recording": {"sid": recorder.sid, "eid": recorder.eid,
                                                              "eval_id": recorder.eval_id, "db_path": recorder.db_path}})
        step = -1
        res = None
        for step in range(max_steps):
            a = await conn.act({"step": step})
            res = bench.step(a["actions"].flatten().tolist())
            if res.done:
                break
        await conn.end_episode({"metrics": {"success": bool(res and res.info.get("status") == "success")},
                                "steps": step + 1, "elapsed_sec": 0.0})
        return {"steps": step + 1}


class _BaseRec:
    def __init__(self):
        pass


def _orig_rows():
    return [dict(XH, source_episode=3, seed=11, key=f"{TASK}_xhard0_11"),
            dict(XH, source_episode=7, seed=22, key=f"{TASK}_xhard0_22")]


def _run_orig(tmp_path, conn, *, max_steps=6, kwargs=None):
    import anyio

    FakeRoboBench.env_kwargs = kwargs or {3: {"demo": 2, "done_at": 4}, 7: {"demo": 2}}
    FakeRoboBench.built = []
    checks = []
    summary = anyio.run(lambda: orig.run_shard(
        _orig_rows(), out_dir=tmp_path / "out", url="ws://127.0.0.1:1", bench_cls=orig.make_tracing_bench_class(
            FakeRoboBench), runner=HandSyncRunner(), conn_factory=lambda url, timeout: conn,
        recorder_cls=orig.make_fixed_sid_recorder_class(_BaseRec), max_steps=max_steps,
        import_check=lambda: checks.append(1)))
    rows = [json.loads(x) for x in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    return summary, rows, checks


def test_orig_runner_rows_frames_and_trace(tmp_path):
    conn = FakeConn()
    summary, rows, checks = _run_orig(tmp_path, conn)
    assert summary == {"episodes": 2, "errors": 0, "aborted": False} and len(checks) == 2
    assert FakeRoboBench.built == [((TASK,), "joint_angle", 6)]
    starts = [p for t, p in conn.log if t == "episode_start"]
    assert [s["recording"]["sid"] for s in starts] == [f"{TASK}|3|11", f"{TASK}|7|22"]
    assert starts[0]["task"] == {"name": TASK, "env_id": TASK, "episode_idx": 3}
    r0, r1 = rows
    assert (r0["status"], r0["exec_steps"], r0["demo_frames"]) == ("success", 4, 2)
    assert (r1["status"], r1["exec_steps"], r1["steps"]) == ("timeout", 6, 6)
    for r in rows:
        assert r["side"] == "orig" and r["policy"] == "pp" and r["dataset"] == "hard-verify"
        assert r["max_steps"] == 6 and r["sid"] == r["eid"] and r["infra"] is False
    # 原始帧：reset 全部帧（演示 2 + 初始 1）+ 每步 1 帧
    for r, n in ((r0, 2 + 1 + 4), (r1, 2 + 1 + 6)):
        fdir = Path(r["frames_dir"])
        meta = json.loads((fdir / "frames.json").read_text())
        assert meta["streams"]["front"] == {"width": 8, "height": 8, "count": n}
        assert meta["streams"]["wrist"]["count"] == n and meta["demo_frames"] == 2 and meta["init_frames"] == 1
        assert (fdir / "front.rgb24").stat().st_size == n * 8 * 8 * 3
        assert r["video_frames"] == {"front": n, "wrist": n}
    # 首帧字节即假环境第 0 帧；第 4 帧（初始帧之后第 1 步）为 step 1 的画面
    raw = (Path(r0["frames_dir"]) / "front.rgb24").read_bytes()
    ref = FakeEnv(TASK, 3, demo=2)
    assert raw[:192] == ref.frames_at(0)["front"].tobytes() and raw[3 * 192:4 * 192] == ref.frames_at(3)["front"].tobytes()
    tr = tw.read_trace(r0["trace_path"])
    assert tr[0]["route"] == "pp-orig" and tr[1]["frames"] == 2 and tr[-1]["exec_steps"] == 4
    steps = [x for x in tr if x["kind"] == "step"]
    assert steps[0]["action"]["shape"] == [8]


def test_orig_runner_timeout_reconnects_and_continues(tmp_path):
    conn = FakeConn(raise_at={1: TimeoutError("act timeout")})
    summary, rows, _ = _run_orig(tmp_path, conn)
    assert summary["episodes"] == 2 and summary["errors"] == 1 and conn.reconnects == 1
    assert rows[0]["status"] == "error" and rows[0]["infra"] is True and rows[0]["infra_reason"] == "pp_act_timeout"
    assert rows[1]["status"] == "timeout" and rows[1]["exec_steps"] == 6


def test_orig_runner_unreachable_aborts_shard(tmp_path):
    conn = FakeConn(raise_at={0: ConnectionError("unreachable")})
    summary, rows, _ = _run_orig(tmp_path, conn)
    assert summary["aborted"] is True and len(rows) == 1 and rows[0]["infra_reason"] == "pp_unreachable"


def test_orig_load_shard_rejects_non_xhard0_and_duplicates(tmp_path):
    p = tmp_path / "shard-00.json"
    p.write_text(json.dumps(_orig_rows()))
    assert [r["key"] for r in orig.load_shard(p, only=f"{TASK}_xhard0_22")] == [f"{TASK}_xhard0_22"]
    p.write_text(json.dumps([dict(XH, tier="xhard1")]))
    with pytest.raises(ValueError):
        orig.load_shard(p)
    p.write_text(json.dumps([XH, XH]))
    with pytest.raises(ValueError):
        orig.load_shard(p)


def test_orig_runner_imports_only_official_robomme():
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run([sys.executable, str(script_path("eval-official/pp_official_runner.py")),
                          "--check-imports"], capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    line = [x for x in out.stdout.splitlines() if x.startswith("OFFICIAL_IMPORTS=")][-1]
    assert line.startswith("OFFICIAL_IMPORTS=PASS") and line.endswith("robomme_hard_imported=0")
    assert f"robomme={REPO / 'src' / 'robomme'}" in line
    print(f"PP_ORIG_IMPORTS=PASS {line}")
