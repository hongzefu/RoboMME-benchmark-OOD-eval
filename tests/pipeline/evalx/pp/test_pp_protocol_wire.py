"""PonderPounce 协议对拍（slow）：真实 vla-eval 0.7.0 源码 vs 新侧 ``pp_client``，本地回环 websocket 假服务。

期望序列的独立来源：按路径加载 vla-eval 源码（``SGEVAL_VLA_EVAL_ROOT``，缺省为主检出 client-env 的
site-packages），用它的 ``SyncEpisodeRunner`` + ``RoboMMEBenchmark``（环境构建换成确定性假 builder）+
``Connection`` 跑一局，假服务逐帧记下收到的协议帧；再用同一假服务跑新侧 ``pp_client.run_episode``（同样用真实
``Connection``）与原侧 ``pp_official_runner.run_shard``，三份帧序列逐帧比对（帧类型、顺序、观测键、dtype/shape、
字节、sid、局号、seq）。vla-eval 源码缺失时测试失败，不跳过。依赖本机 artifacts，故标 slow、不进日常门禁。

第二阶段 S5（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S5」）在本文件追加日常门禁用例
（全部 CPU 替身、无网络；只有上面那条真实 vla-eval 对拍标 slow）：

- ``pp_subgoal_to_official`` 坐标换算：``at [612, 247]`` → ``at <63, 156>``、0～255 全量往返、多组坐标、越界夹取、``None``；
- 第二阶段新侧（``pp_phase2``）：外壳回包的 ``subgoal`` 进轨迹（换算后文本 + 原文）、标准答案改记 history 行、
  C1／C2／C3／C4／C6／C8 落地，并过 ``trace_contract.assert_renderable``／``assert_counts_consistent``；
- 原侧（``pp_official_runner.run_shard``）与开关关闭的新侧：与 ``BASE`` 版 ``pp_client``（``git show``）逐字节比
  轨迹、原始帧、结果行与协议帧序列。
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from pp_fakes import REPO, FakeConn, FakeEnv, FakeSession, action_for, compare_frames
from tests._support.loaders import load_script
from tests.pipeline.evalx.report import trace_contract as tc

CLIENT_ENV_SITE = "artifacts/sg-evaluation/venvs/client-env/lib/python3.11/site-packages"
TASK, SRC, SEED = "VideoUnmask", 5, 98765
XH = {"task": TASK, "tier": "xhard0", "seed": SEED, "source_episode": SRC, "builder_episode": 0,
      "key": f"{TASK}_xhard0_{SEED}", "candidate": None, "spec_sha256": None}
SID = f"{TASK}|{SRC}|{SEED}"
READ_FILES = ("runners/sync_runner.py", "connection.py", "benchmarks/robomme/benchmark.py", "protocol/messages.py",
              "model_servers/serve.py")


def _vla_eval_root() -> Path:
    """``SGEVAL_VLA_EVAL_ROOT``；缺省取主检出（worktree 内按 git 公共目录回到主检出）的 client-env site-packages。"""
    env = os.environ.get("SGEVAL_VLA_EVAL_ROOT")
    if env:
        return Path(env)
    out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                         capture_output=True, text=True, timeout=30)
    main = Path(out.stdout.strip()).parent if out.returncode == 0 and out.stdout.strip() else REPO
    return main / CLIENT_ENV_SITE


@pytest.fixture(scope="module")
def vla(tmp_path_factory):
    root = _vla_eval_root()
    pkg = root / "vla_eval"
    if not (pkg / "runners" / "sync_runner.py").is_file():
        pytest.fail(f"vla-eval 源码不在场（设 SGEVAL_VLA_EVAL_ROOT）：{root}")
    for rel in READ_FILES:
        print(f"VLA_EVAL_SOURCE {pkg / rel} sha256={hashlib.sha256((pkg / rel).read_bytes()).hexdigest()}")
    link_dir = tmp_path_factory.mktemp("vla_eval_link")
    (link_dir / "vla_eval").symlink_to(pkg, target_is_directory=True)
    sys.path.insert(0, str(link_dir))
    from vla_eval._version import __version__
    from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark
    from vla_eval.connection import Connection
    from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
    from vla_eval.recording import NullEpisodeRecorder
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    assert __version__.startswith("0.7.0"), __version__
    RoboMMEBenchmark.configure_render("gpu")
    return types.SimpleNamespace(RoboMMEBenchmark=RoboMMEBenchmark, Connection=Connection, Message=Message,
                                 MessageType=MessageType, pack=pack_message, unpack=unpack_message,
                                 NullEpisodeRecorder=NullEpisodeRecorder, SyncEpisodeRunner=SyncEpisodeRunner)


class LoopbackServer:
    """回环 websocket 假服务：逐帧记录 ``(type, payload, seq)``；HELLO 回 HELLO，OBSERVATION 回 ACTION（seq 回显），
    EPISODE_START／EPISODE_END 不回复（与 vla-eval serve.py 成功路径一致）。每条连接各记一份。"""

    def __init__(self, vla):
        self.vla = vla
        self.conns: list[list[tuple]] = []
        self.port = None
        self._ready = threading.Event()
        self._loop = None
        self._stop = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(30)

    def _run(self):
        import websockets.asyncio.server as wss

        async def handler(ws):
            log: list[tuple] = []
            self.conns.append(log)
            n = 0
            async for raw in ws:
                m = self.vla.unpack(raw)
                log.append((m.type.value, m.payload, m.seq))
                if m.type == self.vla.MessageType.HELLO:
                    await ws.send(self.vla.pack(self.vla.Message(type=self.vla.MessageType.HELLO,
                                                                 payload={"model_server": "fake"}, seq=m.seq)))
                elif m.type == self.vla.MessageType.OBSERVATION:
                    await ws.send(self.vla.pack(self.vla.Message(type=self.vla.MessageType.ACTION,
                                                                 payload={"actions": action_for(n)}, seq=m.seq)))
                    n += 1

        async def main():
            self._stop = asyncio.Event()
            async with wss.serve(handler, "127.0.0.1", 0, compression=None, max_size=None, ping_interval=None) as srv:
                self.port = srv.sockets[0].getsockname()[1]
                self._ready.set()
                await self._stop.wait()

        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(main())

    def close(self):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(10)


def _fake_robomme_modules(monkeypatch, env_kwargs: dict, built: list):
    """把 ``robomme.robomme_env``／``robomme.env_record_wrapper`` 换成假模块：RoboMMEBenchmark.reset 原样运行，
    环境由假 builder 按 ``(env_id, episode_idx)`` 构建。"""
    reg = types.ModuleType("robomme.robomme_env")

    class FakeBuilder:
        def __init__(self, env_id, dataset, action_space, gui_render, max_steps):
            built.append({"env_id": env_id, "dataset": dataset, "action_space": action_space,
                          "gui_render": gui_render, "max_steps": max_steps})
            self.env_id = env_id

        def make_env_for_episode(self, idx):
            return FakeEnv(self.env_id, idx, **env_kwargs)

    wrap = types.ModuleType("robomme.env_record_wrapper")
    wrap.BenchmarkEnvBuilder = FakeBuilder
    monkeypatch.setitem(sys.modules, "robomme.robomme_env", reg)
    monkeypatch.setitem(sys.modules, "robomme.env_record_wrapper", wrap)


def _reference_run(vla, port: int, max_steps: int) -> None:
    """独立来源：真实 SyncEpisodeRunner + RoboMMEBenchmark + Connection；记录器为测试内手写的固定 sid 子类。"""
    import anyio

    class Rec(vla.NullEpisodeRecorder):
        is_active = property(lambda self: True)
        sid = property(lambda self: SID)
        eid = property(lambda self: SID)
        eval_id = property(lambda self: "")
        db_path = property(lambda self: "")

    async def go():
        conn = vla.Connection(f"ws://127.0.0.1:{port}", timeout=300.0)
        await conn.connect(benchmark="vla_eval.benchmarks.robomme.benchmark:RoboMMEBenchmark")
        bench = vla.RoboMMEBenchmark(tasks=[TASK], action_space="joint_angle", max_steps=max_steps)
        try:
            await vla.SyncEpisodeRunner().run_episode(bench, {"name": TASK, "env_id": TASK, "episode_idx": SRC},
                                                      conn, max_steps=max_steps, recorder=Rec())
        finally:
            bench.cleanup()
            await conn.close()

    anyio.run(go)


def _strip(log):
    return [(t, p) for t, p, _ in log]


@pytest.mark.slow
@pytest.mark.parametrize("env_kwargs,max_steps,n_actions", [({"demo": 4, "done_at": 7}, 1300, 7),
                                                             ({"demo": 3}, 1300, 1300)])
def test_new_and_orig_frames_identical_to_vla_eval_sync_runner(vla, monkeypatch, tmp_path, env_kwargs, max_steps,
                                                               n_actions):
    pp = load_script("eval-official/pp_client.py")
    orig = load_script("eval-official/pp_official_runner.py")
    built: list = []
    _fake_robomme_modules(monkeypatch, env_kwargs, built)
    srv = LoopbackServer(vla)
    try:
        _reference_run(vla, srv.port, max_steps)
        env = FakeEnv(TASK, SRC, **env_kwargs)
        res = pp.run_episode(FakeSession(env), XH, {"host": "127.0.0.1", "port": srv.port, "max_steps": max_steps,
                                                    "dataset": "hard-verify",
                                                    "trace_path": str(tmp_path / "new" / "trace.jsonl")},
                             None, connection_factory=lambda url, timeout: vla.Connection(url, timeout=timeout))
        import anyio

        summary = anyio.run(lambda: orig.run_shard(
            [XH], out_dir=tmp_path / "orig", url=f"ws://127.0.0.1:{srv.port}",
            bench_cls=orig.make_tracing_bench_class(vla.RoboMMEBenchmark), runner=vla.SyncEpisodeRunner(),
            conn_factory=lambda url, timeout: vla.Connection(url, timeout=timeout),
            recorder_cls=orig.make_fixed_sid_recorder_class(vla.NullEpisodeRecorder), max_steps=max_steps))
    finally:
        srv.close()
    assert len(srv.conns) == 3
    ref, new, org = srv.conns
    # seq 也逐帧相同（每局新连接、单局分片）
    assert [s for *_, s in ref] == [s for *_, s in new] == [s for *_, s in org]
    fd_new, od_new, notes_new = compare_frames(_strip(new), _strip(ref))
    fd_org, od_org, notes_org = compare_frames(_strip(org), _strip(ref))
    assert (fd_new, od_new) == (0, 0), notes_new
    assert (fd_org, od_org) == (0, 0), notes_org
    kinds = [t for t, *_ in ref]
    assert kinds.count("observation") == n_actions and kinds[1] == "episode_start" and kinds[-1] == "episode_end"
    assert ref[1][1]["recording"]["sid"] == SID and ref[1][1]["task"]["episode_idx"] == SRC
    first = ref[2][1]
    assert set(first) == {"images", "task_description", "states", "video_history", "wrist_video_history",
                          "episode_restart"}
    assert first["states"].dtype == np.float32 and first["states"].shape == (8,)
    assert first["images"]["agentview"].dtype == np.uint8 and first["images"]["agentview"].shape == (8, 8, 3)
    assert len(first["video_history"]) == env_kwargs["demo"]
    assert all(b == {"env_id": TASK, "dataset": "test", "action_space": "joint_angle", "gui_render": False,
                     "max_steps": max_steps} for b in built) and len(built) == 2
    # 结局与步数：新侧、原侧一致
    rows = [json.loads(x) for x in (tmp_path / "orig" / "results.jsonl").read_text().splitlines()]
    assert summary["episodes"] == 1 and rows[0]["status"] == res["status"]
    assert rows[0]["exec_steps"] == res["steps"] == n_actions
    # 两侧轨迹的请求哈希、动作、画面逐行相同
    tw = load_script("eval-official/trace_writer.py")
    tn = tw.read_trace(tmp_path / "new" / "trace.jsonl")
    to = tw.read_trace(rows[0]["trace_path"])
    pick = lambda rs: [(r["kind"], r.get("sha256"), r.get("front_sha256"), (r.get("action") or {}).get("sha256"),  # noqa: E731
                        (r.get("actions") or {}).get("sha256")) for r in rs if r["kind"] in ("request", "response", "step")]
    assert pick(tn) == pick(to)
    assert [r for r in tn if r["kind"] == "demo"] == [r for r in to if r["kind"] == "demo"]
    print(f"PP_PROTOCOL=PASS frames_diff={fd_new + fd_org} order_diff={od_new + od_org} source=vla_eval_sync_runner "
          f"frames={len(ref)} actions={n_actions}")


# ── 第二阶段 S5：子目标坐标换算、外壳回包进轨迹、原侧与 BASE 逐字节一致（日常门禁，CPU 替身） ──────────

BASE_SHA = os.environ.get("SGEVAL_PP_BASE", "b869a3df9e7406b8f5458697f22656165b9c50b3")
PH_TASK, PH_SRC, PH_SEED = "PickXtimes", 7, 123457
PH = {"task": PH_TASK, "tier": "xhard0", "seed": PH_SEED, "source_episode": PH_SRC, "builder_episode": 1,
      "key": f"{PH_TASK}_xhard0_{PH_SEED}", "candidate": None, "spec_sha256": None}


def _pp():
    return load_script("eval-official/pp_client.py")


def _tw():
    return load_script("eval-official/trace_writer.py")


def test_pp_subgoal_to_official_examples():
    f = _pp().pp_subgoal_to_official
    assert f("pick up the cube at [612, 247]") == "pick up the cube at <63, 156>"
    # 多组坐标、空白与小数
    assert (f("pick the red cube at [0, 1000] and put it at[1000,0] then press at [ 500.0 , 500 ]")
            == "pick the red cube at <255, 0> and put it at <0, 255> then press at <128, 128>")
    # 越界夹到 0～255
    assert f("move at [-50, 1200]") == "move at <255, 0>"
    assert f("move at [2000, -3]") == "move at <0, 255>"
    # 没有坐标的文本原样；None 原样（C7 等待中）；不是「at [x, y]」形式的方括号不动
    assert f("open the drawer") == "open the drawer"
    assert f("") == ""
    assert f(None) is None
    assert f("cube [612, 247]") == "cube [612, 247]"


def test_pp_subgoal_to_official_roundtrip_0_255_and_monotonic():
    import re as _re

    f = _pp().pp_subgoal_to_official

    def back(x: int, y: int) -> tuple[int, int]:
        m = _re.fullmatch(r"at <(\d+), (\d+)>", f(f"at [{x}, {y}]"))
        return int(m.group(1)), int(m.group(2))  # (行 y', 列 x')

    for v in range(256):  # 官方 0～255 → 模型 0～1000 → 换回恰好是 v（行列两个维度都查）
        x = round(v * 1000 / 255)
        assert back(x, 0) == (0, v), v
        assert back(0, x) == (v, 0), v
    vals = [back(x, 0)[1] for x in range(1001)]
    assert vals[0] == 0 and vals[-1] == 255 and all(b >= a for a, b in zip(vals, vals[1:]))
    assert all(0 <= v <= 255 for v in vals) and len(set(vals)) == 256


class SubgoalConn(FakeConn):
    """外壳服务的替身：ACTION 回包另带 ``subgoal``（``schedule[i]`` 为第 i 个回包的子目标，越界取最后一个）。"""

    def __init__(self, schedule, **kw):
        super().__init__(**kw)
        self.schedule = list(schedule)

    async def act(self, obs):
        a = await super().act(obs)
        i = self.n_actions - 1
        a["subgoal"] = self.schedule[min(i, len(self.schedule) - 1)]
        return a


SCHEDULE = [None, None, "pick up the cube at [612, 247]", "pick up the cube at [612, 247]", "put it at [10, 990]",
            "put it at [10, 990]"]


def _run_phase2(tmp_path, env, *, conn=None, session=None, attempt=2, max_steps=1300, step_cap=None):
    pp = _pp()
    ep = tmp_path / f"{PH['key']}.a{attempt}"
    conn = conn or SubgoalConn(SCHEDULE)
    session = session or FakeSession(env, step_cap=step_cap)
    ci = {"host": "127.0.0.1", "port": 18310, "max_steps": max_steps, "dataset": "hard-verify", "pp_phase2": True,
          "trace_dir": str(ep), "episode_tag": ep.name}
    res = pp.run_episode(session, PH, ci, None, connection_factory=lambda url, timeout: conn)
    return res, ep, conn, session


def test_phase2_model_subgoal_goes_into_trace_and_is_renderable(tmp_path):
    tw = _tw()
    env = FakeEnv(PH_TASK, PH_SRC, demo=3, done_at=6)
    res, ep, conn, session = _run_phase2(tmp_path, env)
    assert res["status"] == "success" and res["steps"] == 6 and res["demo_frames"] == 3
    rows = tw.read_trace(ep / "trace.jsonl")
    header, demo, end = rows[0], rows[1], rows[-1]
    assert header["route"] == "pp/new"
    assert header["identity"]["attempt"] == 2 and header["identity"]["key"] == PH["key"]
    # C2：演示段含初始帧
    assert demo["kind"] == "demo" and demo["frames"] == 4 and len(demo["states"]) == 4
    ref = FakeEnv(PH_TASK, PH_SRC, demo=3)
    assert demo["front_sha256"] == [tw.image_sha256(ref.frames_at(k)["front"]) for k in range(4)]
    steps = [r for r in rows if r["kind"] == "step"]
    assert [s["subgoal"] for s in steps] == [None, None, "pick up the cube at <63, 156>",
                                             "pick up the cube at <63, 156>", "put it at <252, 3>",
                                             "put it at <252, 3>"]
    assert [s["subgoal_raw"] for s in steps] == SCHEDULE
    # 标准答案不再进 subgoal，改记同步号的 history 行
    hist = [r for r in rows if r["kind"] == "history"]
    assert [(h["start"], h["end"], h["note"]) for h in hist] == [(k, k, f"oracle_simple_subgoal:sg{k}")
                                                                 for k in range(1, 7)]
    # C3／C8
    assert (end["status"], end["terminal_reason"], end["exit_reason"]) == ("success", "success", "env_done")
    assert (end["demo_frames"], end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == (3, 6, 6, 10)
    assert "no_frame" not in end
    # C4：arrays.npz 与 trace 动作逐字节一致，且就是交给环境的值
    with np.load(ep / "arrays.npz") as arr:
        # 原动作键每步一键；冻结说明四.3 起 TraceWriter 另收观测步状态 exec_state__*（6 步都有观测）
        assert sorted(k for k in arr.files if k.startswith("exec_action__")) == [f"exec_action__{i:05d}" for i in range(6)]
        assert sorted(k for k in arr.files if k.startswith("exec_state__")) == [f"exec_state__{i:05d}" for i in range(6)]
        assert len(arr.files) == 12
        for s in steps:
            a = arr[f"exec_action__{s['step'] - 1:05d}"]
            assert a.dtype == np.float64 and a.shape == (8,)
            assert hashlib.sha256(a.tobytes()).hexdigest() == s["action"]["sha256"]
            assert a.tolist() == [float(x) for x in env.actions[s["step"] - 1]]
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": session.steps, "status": res["status"]})
    # 协议帧本身不变：与开关关闭、服务端不带 subgoal 的同一局逐帧相同
    plain = FakeConn()
    _pp().run_episode(FakeSession(FakeEnv(PH_TASK, PH_SRC, demo=3, done_at=6)), PH,
                      {"host": "127.0.0.1", "port": 18310, "max_steps": 1300, "dataset": "hard-verify",
                       "pp_phase2": False}, None, connection_factory=lambda url, timeout: plain)
    sent = lambda log: [(t, p) for t, p in log if t != "action"]  # noqa: E731
    fd, od, notes = compare_frames(sent(conn.log), sent(plain.log))
    assert (fd, od) == (0, 0), notes
    print(f"PP_NEW_PHASE2=PASS route=pp/new steps={len(steps)} subgoal_steps={sum(s['subgoal'] is not None for s in steps)}")


def test_phase2_strict_cap_timeout_is_renderable(tmp_path):
    env = FakeEnv(PH_TASK, PH_SRC, demo=2)
    res, ep, _, session = _run_phase2(tmp_path, env, max_steps=5, step_cap=5)
    end = _tw().read_trace(ep / "trace.jsonl")[-1]
    assert res["status"] == "timeout" and session.steps == 5
    assert (end["status"], end["terminal_reason"], end["steps_attempted"], end["frames_recorded"]) == (
        "timeout", "timeout", 5, 8)
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": session.steps, "status": res["status"]})


def test_phase2_missing_obs_and_env_exception_keep_step_numbers(tmp_path):
    tw = _tw()
    # 空观测步（环境报 error、观测为 None）：保留步号、动作与原因
    env = FakeEnv(PH_TASK, PH_SRC, demo=1, none_obs_at=3)
    res, ep, _, session = _run_phase2(tmp_path / "a", env)
    rows = tw.read_trace(ep / "trace.jsonl")
    steps = [r for r in rows if r["kind"] == "step"]
    assert res["status"] == "error" and [s["step"] for s in steps] == [1, 2, 3]
    assert steps[2]["observed"] is False and steps[2]["missing_reason"] == "obs_none"
    assert steps[2]["subgoal"] == "pick up the cube at <63, 156>" and steps[2]["action"]["shape"] == [8]
    end = rows[-1]
    assert (end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == (3, 2, 1 + 1 + 2)
    assert tc.contract_problems(ep) == []
    tc.assert_counts_consistent(ep, {"exec_steps": session.steps, "status": res["status"]})

    # 环境 step 抛异常：EnvSession 已把这一步计入 exec_steps，轨迹同样留一行缺观测步
    class Boom(FakeSession):
        def step(self, action):
            if self.steps == 1:
                self.steps += 1  # 与 EnvSession 一致：进入环境后异常也计步
                raise RuntimeError("physx exploded")
            return super().step(action)

    env2 = FakeEnv(PH_TASK, PH_SRC, demo=1)
    res2, ep2, _, s2 = _run_phase2(tmp_path / "b", env2, session=Boom(env2))
    rows2 = tw.read_trace(ep2 / "trace.jsonl")
    steps2 = [r for r in rows2 if r["kind"] == "step"]
    assert res2["status"] == "error" and res2["steps"] == 1 and s2.steps == 2
    assert steps2[1]["observed"] is False and steps2[1]["missing_reason"] == "env_exception:RuntimeError"
    assert rows2[-1]["steps_attempted"] == 2 and rows2[-1]["steps_observed"] == 1
    assert tc.contract_problems(ep2) == []
    tc.assert_counts_consistent(ep2, {"exec_steps": s2.steps, "status": res2["status"]})


def test_phase2_without_server_subgoal_is_infra_error_not_oracle(tmp_path):
    env = FakeEnv(PH_TASK, PH_SRC, demo=2, done_at=4)
    res, ep, _, _ = _run_phase2(tmp_path, env, conn=FakeConn())
    assert res["status"] == "error" and res["infra"] is True and res["infra_reason"] == "pp_subgoal_missing"
    assert env.actions == []  # 第一个回包就判外壳未生效，不把动作交给环境
    rows = _tw().read_trace(ep / "trace.jsonl")
    assert not [r for r in rows if r["kind"] == "step"] and rows[-1]["steps_attempted"] == 0
    assert tc.contract_problems(ep) == []


def test_phase2_reset_failure_is_no_frame(tmp_path):
    class BadReset(FakeSession):
        def reset(self):
            raise RuntimeError("vulkan")

    env = FakeEnv(PH_TASK, PH_SRC, demo=2)
    res, ep, _, _ = _run_phase2(tmp_path, env, session=BadReset(env))
    end = _tw().read_trace(ep / "trace.jsonl")[-1]
    assert res["status"] == "error" and end["no_frame"] is True
    assert (end["demo_frames"], end["frames_recorded"], end["steps_attempted"]) == (0, 0, 0)
    assert tc.contract_problems(ep) == []
    tc.assert_renderable(ep)  # 无帧 error 局只核契约、不调重绘器


def test_phase2_switch_from_env_var(monkeypatch):
    pp = _pp()
    monkeypatch.delenv("SGEVAL_PP_SERVER_WRAP", raising=False)
    assert pp.phase2_enabled({}) is False
    monkeypatch.setenv("SGEVAL_PP_SERVER_WRAP", "1")
    assert pp.phase2_enabled({}) is True and pp.phase2_enabled({"pp_phase2": False}) is False
    monkeypatch.setenv("SGEVAL_PP_SERVER_WRAP", "0")
    assert pp.phase2_enabled({}) is False and pp.phase2_enabled({"pp_phase2": True}) is True


def test_traced_connection_last_subgoal():
    import anyio

    pp = _pp()

    async def go():
        t1 = pp.TracedConnection(SubgoalConn([None, "a at [612, 247]"]))
        assert type(t1.last_subgoal).__name__ == "_Unset"
        await t1.act({})
        assert t1.last_subgoal is None
        await t1.act({})
        assert t1.last_subgoal == "a at [612, 247]"
        t2 = pp.TracedConnection(FakeConn())
        await t2.act({})
        assert type(t2.last_subgoal).__name__ == "_Unset"

    anyio.run(go)


# ── 原侧与开关关闭的新侧：对 BASE 版 pp_client 逐字节 ─────────────────────────────


def _load_pp_from(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("pp_client", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_base_pp(tmp_path):
    """``git show <BASE>:scripts/eval-official/pp_client.py`` 写到临时目录后按模块名 ``pp_client`` 载入。"""
    out = subprocess.run(["git", "show", f"{BASE_SHA}:scripts/eval-official/pp_client.py"], cwd=REPO,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode(errors="replace")
    d = tmp_path / "base_src"
    d.mkdir()
    (d / "pp_client.py").write_bytes(out.stdout)
    return _load_pp_from(d / "pp_client.py")


class _Res:
    def __init__(self, obs, info, done):
        self.obs, self.info, self.done = obs, info, done


class _Bench:
    """RoboMMEBenchmark 的最小替身（同 test_pp_protocol.FakeRoboBench）。"""

    env_kwargs: dict = {}

    def __init__(self, tasks, action_space, max_steps):
        self._env = None

    def reset(self, task):
        if self._env is not None:
            self._env.close()
        self._env = FakeEnv(task["env_id"], task["episode_idx"], **self.env_kwargs.get(task["episode_idx"], {}))
        obs, _info = self._env.reset()
        return obs

    def step(self, action):
        obs, _r, term, trunc, info = self._env.step(list(action))
        return _Res(obs, info, bool(term) or bool(trunc) or info.get("status") == "error")

    def cleanup(self):
        pass


class _HandRunner:
    """按 SyncEpisodeRunner 源码事实手写的流程（同 test_pp_protocol.HandSyncRunner）。"""

    async def run_episode(self, bench, task, conn, *, max_steps, recorder):
        bench.reset(task)
        await conn.start_episode({"task": task, "recording": {"sid": recorder.sid, "eid": recorder.eid,
                                                              "eval_id": recorder.eval_id, "db_path": recorder.db_path}})
        step, res = -1, None
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


def _orig_run(pp_mod, out_dir, conn, monkeypatch):
    """以给定 pp_client 模块跑原侧 run_shard（pp_official_runner 原文件，其 ``import pp_client`` 解析到 ``pp_mod``）。"""
    import anyio

    monkeypatch.setitem(sys.modules, "pp_client", pp_mod)
    orig = load_script("eval-official/pp_official_runner.py", fresh=True)
    assert orig.pp_client is pp_mod
    _Bench.env_kwargs = {3: {"demo": 2, "done_at": 4}, 7: {"demo": 2, "none_obs_at": 3}, 9: {"demo": 1}}
    rows = [dict(PH, source_episode=3, seed=11, key=f"{PH_TASK}_xhard0_11"),
            dict(PH, source_episode=7, seed=22, key=f"{PH_TASK}_xhard0_22"),
            dict(PH, source_episode=9, seed=33, key=f"{PH_TASK}_xhard0_33")]
    anyio.run(lambda: orig.run_shard(
        rows, out_dir=out_dir, url="ws://127.0.0.1:1", bench_cls=orig.make_tracing_bench_class(_Bench),
        runner=_HandRunner(), conn_factory=lambda url, timeout: conn,
        recorder_cls=orig.make_fixed_sid_recorder_class(_BaseRec), max_steps=6))
    return [json.loads(x) for x in (out_dir / "results.jsonl").read_text().splitlines()]


def _norm_row(row: dict, root: Path) -> dict:
    return {k: (v.replace(str(root), "<ROOT>") if isinstance(v, str) else v) for k, v in row.items() if k != "wall_s"}


@pytest.mark.parametrize("server_sends_subgoal", [False, True])
def test_orig_side_serialized_output_identical_to_base(tmp_path, monkeypatch, server_sends_subgoal):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("SGEVAL_PP_SERVER_WRAP", raising=False)
    base = _load_base_pp(tmp_path)
    cur = _load_pp_from(REPO / "scripts" / "eval-official" / "pp_client.py")
    assert cur.TRACE_SCHEMA_ROUTE_ORIG == base.TRACE_SCHEMA_ROUTE_ORIG == "pp-orig"
    files = {}
    for name, mod in (("base", base), ("cur", cur)):
        conn = SubgoalConn(SCHEDULE) if server_sends_subgoal else FakeConn()
        out = tmp_path / name
        rows = _orig_run(mod, out, conn, monkeypatch)
        files[name] = ({str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*"))
                        if p.is_file() and p.name != "results.jsonl"}, [_norm_row(r, out) for r in rows], conn.log)
    (fb, rb, lb), (fc, rc, lc) = files["base"], files["cur"]
    assert sorted(fb) == sorted(fc)
    assert sum(k.endswith("trace.jsonl") for k in fb) == 3 and any(k.endswith(".rgb24") for k in fb)
    diff = [k for k in fb if fb[k] != fc[k]]
    assert diff == [], diff
    assert rb == rc and len(rb) == 3
    assert len(lb) == len(lc) and compare_frames(lb, lc)[:2] == (0, 0)
    print(f"ORIG_SIDE_UNCHANGED=PASS route=pp-orig files={len(fb)} rows={len(rb)} subgoal_reply={server_sends_subgoal}")


@pytest.mark.parametrize("env_kwargs,step_cap", [({"demo": 3, "done_at": 5}, None), ({"demo": 1, "none_obs_at": 2}, None),
                                                 ({"demo": 2}, 4)])
def test_new_side_switch_off_identical_to_base(tmp_path, monkeypatch, env_kwargs, step_cap):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("SGEVAL_PP_SERVER_WRAP", raising=False)
    base = _load_base_pp(tmp_path)
    # BASE 的 pp_client 经 load_trace_writer() 取的是**当前** trace_writer，而当前 TraceWriter 缺省收集完整数组
    # （冻结说明四.3，BASE 时代的 TraceWriter 没有这一功能）：BASE 侧换成 collect_arrays=False 的同一个类，
    # 还原 BASE 时代的写盘行为，原断言（局目录只有 trace.jsonl、trace 逐字节相同）照旧成立。
    real_tw = base.load_trace_writer()
    monkeypatch.setattr(base, "load_trace_writer", lambda: types.SimpleNamespace(
        TraceWriter=functools.partial(real_tw.TraceWriter, collect_arrays=False)))
    cur = _load_pp_from(REPO / "scripts" / "eval-official" / "pp_client.py")
    out = {}
    for name, mod in (("base", base), ("cur", cur)):
        ep = tmp_path / name / f"{PH['key']}.a1"
        conn = SubgoalConn(SCHEDULE)  # 即使服务端带 subgoal，开关关时也与 BASE 相同
        res = mod.run_episode(FakeSession(FakeEnv(PH_TASK, PH_SRC, **env_kwargs), step_cap=step_cap), PH,
                              {"host": "127.0.0.1", "port": 18310, "max_steps": 1300, "dataset": "hard-verify",
                               "trace_dir": str(ep), "episode_tag": ep.name},
                              None, connection_factory=lambda url, timeout, c=conn: c)
        res = {k: v for k, v in res.items() if k != "timing"}
        out[name] = (res, sorted(p.name for p in ep.iterdir()), (ep / "trace.jsonl").read_bytes(), conn.log)
    (rb, lsb, tb, lb), (rc, lsc, tcur, lc) = out["base"], out["cur"]
    assert rb == rc and lsb == lsc == ["trace.jsonl"] and tb == tcur
    assert compare_frames(lb, lc)[:2] == (0, 0)
