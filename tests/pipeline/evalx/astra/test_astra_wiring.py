"""Astra 新侧接线：驱动把 Astra 的 ``runner.episode`` 接到真实 S1 builder 上（环境打桩、零外联）。

第一阶段（S5）判定行：``ASTRA_WIRING=PASS api_calls=0 builder_dataset_ok=1``（由 ``test_astra_wiring_xhard0`` 打印）。

第二阶段（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节 S3）在本文件补：

- 记录契约：``route=astra/new``、identity 的 ``builder_episode``／``key``／``attempt=1``、演示段含初始帧（C2）、
  结束原因只取 success／fail／timeout／error（C3）、三分计数（C8）；局目录 ``<key>.a1/`` 过 S0 测试助手
  ``assert_renderable``／``assert_counts_consistent``；帧数口径：两路无损流帧数 = ``frames_recorded``。
- ``--key`` 映射：局目录名 ``<key>.a1`` ↔ trace ``identity.key``／``attempt`` ↔ 重绘工具的官方局号 ``3a1``。
- 启动器：source 席位函数库后 ``MAX_STEPS``、trap、shell 选项不变，否则 RUN_BLOCKED；收尾显式
  重绘 → 转码 → 验收，生成一份官方视频与一份普通视频，重绘失败时保留原始帧。
- 费用硬上限（零外联夹具）：首次大输入、输入增长、usage 缺失、STOP 在等待中到达、守卫崩溃、429 重试复检；
  被拒请求一律不触发实际发送；局数硬上限 2。

启动器收尾用例：仓库里有真实 S2b 脚本（``seat_media_lib.sh``、``official_media_check.py``，真实库再调 S2a 重绘器
``--source raw``）时优先用真实脚本，否则退回本文件的桩，并断言／打印 ``ASTRA_MEDIA_MODE=real|stub``。桩与 S2b 真实
签名一致（``transcode_episode_dir [--keep-raw] <局目录>`` 位置敏感、验收四个必需参数含 ``--dataset``、无帧局 NO_FRAME）；
桩 ``render_official_dir`` 先把无损流拼成临时 ``episode.mp4``，再调用真实重绘工具（官方 ``RolloutRecorder`` 原类）。
"""
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import numpy as np
import pytest

from astra_fakes import (FakeEnv, FakeMonitor, FakeResponder, FakeVLA, NetCounter, astra_session, load_guard,
                         make_args, make_deps, print_upstream_digests, recording_builder_cls, third_party, write_cases)
from tests._support.loaders import load_script
from tests.pipeline.evalx.report import trace_contract as tc

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "eval-official" / "run_astra.sh"
RENDER = REPO / "scripts" / "eval-official" / "render_official_video.py"
ALL_TASKS = ["BinFill", "StopCube", "PickXtimes", "SwingXtimes", "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap",
             "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube",
             "InsertPeg", "PatternLock", "RouteStick"]
FAKE_KEY = "sk-test-placeholder-not-a-real-key"


def _trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_dir(output: Path, task: str, ep: int) -> Path:
    """``results/<task>/ep<NNN>/`` 下恰一个 ``<key>.a1``。"""
    dirs = [p for p in (Path(output) / task / f"ep{ep:03d}").iterdir() if p.is_dir() and ".a" in p.name]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


class _NullWriter:
    def append_data(self, frame):
        pass

    def close(self):
        pass


# ── 局清单（第一阶段，口径不变） ──────────────────────────────────────────

def test_prepare_hard0_local_zero_is_official_episode_3():
    """第二档身份：16 任务各取官方 test episode 3，即 hard-verify 本地局号 0（手写期望，不读被测代码的表）。"""
    with astra_session() as (mod, astra):
        assert list(astra.core.TASKS) == ALL_TASKS
        doc = mod.prepare_cases(recording_builder_cls(), "hard-verify", ALL_TASKS, source_episodes=[3])
    assert doc["dataset"] == "hard-verify"
    assert len(doc["cases"]) == 16
    for case in doc["cases"]:
        assert case["episode"] == 0 and case["source_episode"] == 3 and case["tier"] == "xhard0"


def test_prepare_v9_connectivity_is_videounmask_xhard1_first():
    with astra_session() as (mod, _astra):
        cls = recording_builder_cls()
        doc = mod.prepare_cases(cls, "ood", ["VideoUnmask"], tier="xhard1", index=0)
        builder = cls("VideoUnmask", dataset="ood", action_space="joint_angle", gui_render=False, max_steps=1600)
    (case,) = doc["cases"]
    first_xhard1 = min(ep for ep in range(builder.get_episode_num()) if builder.resolve_identity(ep)["tier"] == "xhard1")
    assert case["task"] == "VideoUnmask" and case["tier"] == "xhard1" and case["episode"] == first_xhard1
    assert case["seed"] == builder.resolve_identity(first_xhard1)["seed"]


# ── 正常局：接线 + 记录契约 + 帧数口径 ──────────────────────────────────────

def test_astra_wiring_xhard0(tmp_path, monkeypatch):
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        print_upstream_digests(mod)
        tasks = ["VideoUnmask", "BinFill"]
        cls = recording_builder_cls(lambda b, ep: FakeEnv(terminal_step=40))
        doc = mod.prepare_cases(cls, "hard-verify", tasks, source_episodes=[3])
        cls.constructed.clear()
        cases = write_cases(tmp_path / "cases.json", doc)
        args = make_args(tmp_path, cases, max_steps=1300)
        checks, monitor, vla = [], FakeMonitor(), FakeVLA()
        responder = FakeResponder(astra.champ)
        deps = make_deps(astra, cls, monitor=monitor, vla=vla, responder=responder, check_calls=checks)
        out = mod.run_cases(args, deps)
        seeds = {t: cls(t, dataset="hard-verify", action_space="joint_angle", gui_render=False,
                        max_steps=1300).resolve_identity(0)["seed"] for t in tasks}

    # ① validate_checkpoints 以启动参数调用一次
    assert checks == [(args.vla_checkpoint, args.monitor_adapter)]
    # builder：真实 S1 类，构造参数逐项手写期望
    assert cls.constructed[:2] == [
        {"env_id": t, "dataset": "hard-verify", "action_space": "joint_angle", "gui_render": False, "max_steps": 1300}
        for t in tasks]
    assert [c["episode"] for c in cls.make_calls] == [0, 0]
    assert all(c["args"] == () and c["kwargs"] == {} for c in cls.make_calls), "步数不得逐局覆盖"
    builder = cls("VideoUnmask", dataset="hard-verify", action_space="joint_angle", gui_render=False, max_steps=1300)
    assert builder.dataset == "hard-verify" and builder.max_steps_without_demonstration == 1302
    assert builder.resolve_identity(0)["source_episode"] == 3
    builder_dataset_ok = 1

    run = Path(args.output).parent
    assert run.parent.name == "group_0"
    results = out["results"]
    assert [r["status"] for r in results] == ["success", "success"]
    for task in tasks:
        ep = run / "results" / task / "ep000"
        # Astra 自己的产物照旧
        for name in ("identity.json", "decisions.jsonl", "actions.npy", "result.json", "rollout.mp4"):
            assert (ep / name).is_file(), name
        assert (ep / "monitor_inputs").is_dir()
        identity = json.loads((ep / "identity.json").read_text())
        assert identity["dataset"] == "hard-verify" and identity["episode"] == 0
        assert identity["seed_and_difficulty"][1] == "xhard0"
        assert np.load(ep / "actions.npy").shape == (40, 8)
        # <key>.a1：key 手写期望 <task>_<tier>_<seed>
        key = f"{task}_xhard0_{seeds[task]}"
        a_dir = _episode_dir(run / "results", task, 0)
        assert a_dir.name == f"{key}.a1"
        result = json.loads((ep / "result.json").read_text())
        assert result["dataset"] == "hard-verify" and result["steps"] == 40 and result["planner_calls"] == 1
        assert result["key"] == key and result["attempt"] == 1 and result["episode_dir"] == f"{key}.a1"
        assert result["media_dir"] == f"{key}.a1/media" and result["route"] == "astra/new"
        assert result["exec_steps"] == 40 and result["steps_observed"] == 40 and result["recorder_verify"] == "PASS"
        # 记录契约（C1～C8）
        tc.assert_renderable(a_dir)
        tc.assert_counts_consistent(a_dir, {"exec_steps": result["exec_steps"], "status": result["status"]})
        rows = _trace(a_dir / "trace.jsonl")
        header, demo, end = rows[0], next(r for r in rows if r["kind"] == "demo"), rows[-1]
        assert header["route"] == "astra/new" and header["max_steps"] == 1300
        ident = header["identity"]
        assert ident["source_episode"] == 3 and ident["dataset"] == "hard-verify" and ident["builder_episode"] == 0
        assert ident["key"] == key and ident["attempt"] == 1 and ident["tier"] == "xhard0"
        # C2：FakeEnv reset 给 3 帧演示 + 1 帧初始画面，全部记入
        assert demo["frames"] == 4 and len(demo["states"]) == 4 and demo["texts"] == ["pick up the cube"]
        assert end["demo_frames"] == 3 and end["status"] == end["terminal_reason"] == "success"
        assert end["steps_attempted"] == end["steps_observed"] == end["exec_steps"] == 40
        assert end["frames_recorded"] == 3 + 1 + 40 and end["no_frame"] is False
        steps = [r for r in rows if r["kind"] == "step"]
        assert [r["step"] for r in steps] == list(range(1, 41)) and all(r["subgoal"] for r in steps)
        names = [r["name"] for r in rows if r["kind"] == "request"]
        assert names.count("planner") == 1 and names.count("vla_infer") == 3 and names.count("monitor") == 2
        assert names[0] == "vla_reset"
        assert sum(1 for r in rows if r["kind"] == "response") == 3
        # 帧数口径：两路无损流（已从 media/ 移到局目录，C5）帧数 = frames_recorded；原动作每步一键
        for stream in ("front", "wrist"):
            assert (a_dir / f"{stream}.mkv").is_file() and not (a_dir / "media" / f"{stream}.mkv").exists()
            assert _jsonl_rows(a_dir / f"frames-{stream}.jsonl") == end["frames_recorded"]
        with np.load(a_dir / "arrays.npz") as arr:
            # 原动作键（write_exec_actions）照旧每步一键；冻结说明四.3 起 TraceWriter 另收观测步状态 exec_state__*
            assert sorted(k for k in arr.files if k.startswith("exec_action__")) == \
                [f"exec_action__{i:05d}" for i in range(40)]
            assert sorted(k for k in arr.files if k.startswith("exec_state__")) == \
                [f"exec_state__{i:05d}" for i in range(40)]
            assert len(arr.files) == 80
            assert arr["exec_action__00000"].dtype == np.float32 and arr["exec_action__00000"].shape == (8,)
        assert end["arrays"]["path"] == "arrays.npz" and "error" not in end["arrays"]
        assert end["arrays"]["action_keys"] == 40 and end["arrays"]["state_keys"] == 40
        assert end["arrays"]["missing_state_steps"] == []
        meta = json.loads((a_dir / "media" / "meta.json").read_text())
        assert meta["never_degrade"] is True and meta["level"] == 0 and meta["key"] == key
        assert meta["builder_episode"] == 0 and meta["attempt"] == 1 and meta["dataset"] == "hard-verify"
        rsum = json.loads((a_dir / "media" / "summary.json").read_text())
        assert rsum["RECORDER_VERIFY"] == "PASS" and rsum["frames"] == 2 * end["frames_recorded"]
        assert rsum["summary"]["exec_steps"] == 40
    # planner_calls 为分片共用的一个目录，按请求 uuid 分子目录
    calls = sorted(p for p in (run / "planner_calls").iterdir())
    assert len(calls) == 2 and all((p / "request.json").is_file() and (p / "response.json").is_file() for p in calls)
    assert {json.loads((p / "request.json").read_text())["episode"] for p in calls} == {0}
    # ⑥ 正常结束写 PILOT_FINISHED.json
    finished = json.loads((run / "results" / "PILOT_FINISHED.json").read_text())
    assert finished["dataset"] == "hard-verify" and finished["tasks"] == tasks
    assert responder.calls == 2 and vla.resets == 2
    assert net.calls == 0
    print(f"ASTRA_WIRING=PASS api_calls={net.calls} builder_dataset_ok={builder_dataset_ok}")


def test_v9_connectivity_runs_exactly_1600_steps(tmp_path, monkeypatch):
    """OOD（原 V9）连通局：第三阶段起 ood ↔ 1800（strict）；环境永不结束时循环恰好执行 1800 步后记 timeout
    （C3：不再写 loop_exit），Astra 自己的循环在 cap 处停、strict 守卫不触发（``cap_hit=False``）。

    函数名里的 1600 是历史名：它登记在总表 ``tests/contract/benchmark_contracts.json`` 的 C13-ASTRA-WIRING 条目里，
    改名须主会话同步总表，故本轮保留原名、只改内容。"""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        cls = recording_builder_cls(lambda b, ep: FakeEnv(terminal_step=None))
        doc = mod.prepare_cases(cls, "ood", ["VideoUnmask"], tier="xhard1", index=0)
        cls.constructed.clear()
        cases = write_cases(tmp_path / "cases.json", doc)
        args = make_args(tmp_path, cases, max_steps=1800)
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        deps = make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=FakeResponder(astra.champ),
                         check_calls=[])
        out = mod.run_cases(args, deps)
    assert cls.constructed == [{"env_id": "VideoUnmask", "dataset": "ood", "action_space": "joint_angle",
                                "gui_render": False, "max_steps": 1800}]
    (result,) = out["results"]
    assert result["status"] == "timeout" and result["steps"] == 1800 and result["exec_steps"] == 1800
    assert result["strict_cap"] is True and result["effective_cap"] == 1800 and result["cap_hit"] is False
    episode = doc["cases"][0]["episode"]
    assert cls.make_calls[0]["episode"] == episode
    a_dir = _episode_dir(Path(args.output), "VideoUnmask", episode)
    rows = _trace(a_dir / "trace.jsonl")
    end = rows[-1]
    assert end["exec_steps"] == 1800 and end["terminal_reason"] == end["status"] == "timeout"
    assert end["frames_recorded"] == 3 + 1 + 1800 and end["omitted_timeout_frames"] == 0
    assert end["strict_cap"] is True and end["effective_cap"] == 1800 and end["cap_hit"] is False
    ident = rows[0]["identity"]
    assert ident["tier"] == "xhard1" and ident["builder_episode"] == episode and ident.get("source_episode") is None
    assert a_dir.name == f"VideoUnmask_xhard1_{ident['seed']}.a1"
    tc.assert_renderable(a_dir)
    tc.assert_counts_consistent(a_dir, {"exec_steps": result["exec_steps"], "status": result["status"]})
    assert _jsonl_rows(a_dir / "frames-front.jsonl") == end["frames_recorded"]
    assert net.calls == 0


# ── 异常局：缺观测步、无帧局（C3、C8） ────────────────────────────────────

class _RaisingEnv(FakeEnv):
    """第 ``at`` 步 ``env.step`` 抛异常（动作已交给环境，但没有观测）。"""

    def __init__(self, at: int) -> None:
        super().__init__(terminal_step=None)
        self.at = at

    def step(self, action):
        if self.t + 1 == self.at:
            self.t += 1
            raise RuntimeError("fake simulator crash inside step")
        return super().step(action)


class _ObsNoneEnv(FakeEnv):
    def __init__(self, at: int) -> None:
        super().__init__(terminal_step=None)
        self.at = at

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return (None if self.t == self.at else obs), r, term, trunc, info


@pytest.mark.parametrize("mode", ["raise", "obs_none"])
def test_missing_observation_step_counts(tmp_path, monkeypatch, mode):
    """第 5 步无观测：步行保留步号、动作与原因（observed=false），attempted=5、observed=4，终态 error；
    frames_recorded = 演示 3 + 1 + 4；结果行 exec_steps 与 trace 对账。"""
    NetCounter().install(monkeypatch)
    env_cls = _RaisingEnv if mode == "raise" else _ObsNoneEnv
    with astra_session() as (mod, astra):
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        cls = recording_builder_cls(lambda b, ep: env_cls(5))
        doc = mod.prepare_cases(cls, "hard-verify", ["BinFill"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        out = mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(),
                                            responder=FakeResponder(astra.champ), check_calls=[]))
    (result,) = out["results"]
    assert result["status"] == "error" and result["exec_steps"] == 5 and result["steps_observed"] == 4
    a_dir = _episode_dir(Path(args.output), "BinFill", 0)
    assert tc.contract_problems(a_dir) == []
    tc.assert_counts_consistent(a_dir, {"exec_steps": result["exec_steps"], "status": result["status"]})
    rows = _trace(a_dir / "trace.jsonl")
    end = rows[-1]
    assert end["status"] == end["terminal_reason"] == "error" and end["no_frame"] is False
    assert end["steps_attempted"] == 5 and end["steps_observed"] == 4 and end["frames_recorded"] == 3 + 1 + 4
    last = [r for r in rows if r["kind"] == "step"][-1]
    assert last["step"] == 5 and last["observed"] is False and last["action"] is not None
    assert last["missing_reason"] == ("env_step_exception:RuntimeError" if mode == "raise" else "obs_none")
    assert _jsonl_rows(a_dir / "frames-front.jsonl") == end["frames_recorded"]
    with np.load(a_dir / "arrays.npz") as arr:
        # 动作每个 attempted 步都有键；第 5 步无观测：不补零状态，记入 end.arrays.missing_state_steps（冻结说明四.3）
        assert sorted(k for k in arr.files if k.startswith("exec_action__")) == \
            [f"exec_action__{i:05d}" for i in range(5)]
        assert sorted(k for k in arr.files if k.startswith("exec_state__")) == \
            [f"exec_state__{i:05d}" for i in range(4)]
        assert len(arr.files) == 9
    assert end["arrays"]["action_keys"] == 5 and end["arrays"]["state_keys"] == 4
    assert end["arrays"]["missing_state_steps"] == [5] and "error" not in end["arrays"]


def test_env_build_failure_is_no_frame_error(tmp_path, monkeypatch):
    """环境都没建起来：error 局 no_frame=true、frames_recorded=0，media/ 不建（录制器只在首次 reset 时建）。"""
    NetCounter().install(monkeypatch)

    def boom(builder, ep):
        raise RuntimeError("fake simulator start failure")

    with astra_session() as (mod, astra):
        cls = recording_builder_cls(boom)
        doc = mod.prepare_cases(cls, "hard-verify", ["BinFill"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        out = mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(),
                                            responder=FakeResponder(astra.champ), check_calls=[]))
    (result,) = out["results"]
    a_dir = _episode_dir(Path(args.output), "BinFill", 0)
    end = _trace(a_dir / "trace.jsonl")[-1]
    assert result["status"] == "error" and result["exec_steps"] == 0 and result["frames_recorded"] == 0
    assert end["no_frame"] is True and end["frames_recorded"] == 0 and end["demo_frames"] == 0
    assert tc.contract_problems(a_dir) == []
    tc.assert_renderable(a_dir)  # 无帧局只核契约、不调重绘器
    assert not (a_dir / "media").exists() and not (a_dir / "arrays.npz").exists()


def test_media_dir_must_be_new_and_empty(tmp_path):
    """``media/`` 已存在（哪怕是空的）也拒绝：不覆盖任何已有证据。"""
    with astra_session() as (mod, _astra):
        made = []
        media = tmp_path / "x.a1" / "media"
        media.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            mod._open_media(lambda d, m: made.append(d), media, {})
    assert made == []


# ── --key 映射 ───────────────────────────────────────────────────────────

def test_key_mapping_dir_trace_and_official_episode_id(tmp_path, monkeypatch):
    """局目录名 ``<key>.a1`` ↔ trace identity ↔ 重绘工具的官方局号（``<source_episode>a<attempt>``）↔ 验收输入行。"""
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        cls = recording_builder_cls(lambda b, ep: FakeEnv(terminal_step=12))
        doc = mod.prepare_cases(cls, "hard-verify", ["VideoUnmask"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(),
                                      responder=FakeResponder(astra.champ), check_calls=[]))
        a_dir = _episode_dir(Path(args.output), "VideoUnmask", 0)
        manifest, ledger = mod.media_inputs(a_dir)
    render = load_script("eval-official/render_official_video.py")
    trace = render.load_trace(a_dir / "trace.jsonl")
    assert render._episode_id(trace, a_dir) == "3a1"
    key = trace.identity["key"]
    assert a_dir.name == f"{key}.a1" and manifest["key"] == key and manifest["builder_episode"] == 0
    # 账本按 AttemptLedger 口径：attempt_start + accept，accepted_attempt_id == attempt_id，attempt_no = 目录名 .a1
    start, accept = ledger
    assert start["kind"] == "attempt_start" and accept["kind"] == "accept"
    assert accept["accepted_attempt_id"] == accept["attempt_id"] == start["attempt_id"]
    assert accept["attempt_no"] == start["attempt_no"] == 1 and accept["key"] == key
    assert accept["episode_dir"] == str(a_dir.resolve()) and accept["dataset"] == "hard-verify"
    assert manifest["route"] == accept["route"] == "astra/new" and manifest["dataset"] == "hard-verify"
    # 目录名与 key 不符：重绘工具与验收输入都拒绝
    wrong = a_dir.with_name(f"VideoUnmask_xhard0_999999.a1")
    os.rename(a_dir, wrong)
    with pytest.raises(ValueError):
        render._episode_id(render.load_trace(wrong / "trace.jsonl"), wrong)
    with astra_session() as (mod, _astra):
        with pytest.raises(ValueError, match="不符"):
            mod.media_inputs(wrong)


# ── 第一阶段拒绝用例（口径不变） ────────────────────────────────────────

@pytest.mark.parametrize("dataset,max_steps", [("hard-verify", 1600), ("ood", 1300), ("ood", 1600)])
def test_step_cap_pairing_blocks_before_any_side_effect(tmp_path, dataset, max_steps):
    with astra_session() as (mod, astra):
        cls = recording_builder_cls()
        doc = (mod.prepare_cases(cls, dataset, ["VideoUnmask"], source_episodes=[3]) if dataset == "hard-verify"
               else mod.prepare_cases(cls, dataset, ["VideoUnmask"], tier="xhard1", index=0))
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=max_steps)
        responder = FakeResponder(astra.champ)
        with pytest.raises(ValueError, match="step_cap_pairing"):
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder,
                                          check_calls=[]))
    assert not Path(args.output).exists() and responder.calls == 0


def test_layout_requires_group_dir(tmp_path):
    with astra_session() as (mod, astra):
        cls = recording_builder_cls()
        doc = mod.prepare_cases(cls, "hard-verify", ["VideoUnmask"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300, group="not_a_group")
        with pytest.raises(ValueError, match="reason=layout"):
            mod.run_cases(args, make_deps(astra, cls, check_calls=[]))
    assert not Path(args.output).exists()


@pytest.mark.parametrize("existing", ["output", "spool"])
def test_existing_output_or_spool_is_refused(tmp_path, existing):
    """② 两个目录都必须是新的（上游 main() 的 mkdir(exist_ok=False) 口径）。"""
    with astra_session() as (mod, astra):
        cls = recording_builder_cls()
        doc = mod.prepare_cases(cls, "hard-verify", ["VideoUnmask"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        Path(getattr(args, existing)).mkdir(parents=True)
        responder = FakeResponder(astra.champ)
        with pytest.raises(ValueError, match="existing evidence is preserved"):
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder,
                                          check_calls=[]))
    assert responder.calls == 0


def test_real_validate_checkpoints_rejects_fake_dirs(tmp_path):
    """① 用上游真实 validate_checkpoints：假目录在任何目录创建与请求之前被拒。"""
    with astra_session() as (mod, astra):
        cls = recording_builder_cls()
        doc = mod.prepare_cases(cls, "hard-verify", ["VideoUnmask"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        responder = FakeResponder(astra.champ)
        with pytest.raises(ValueError, match="Missing checkpoint component"):
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder))
    assert not Path(args.output).exists() and responder.calls == 0


def test_identity_mismatch_blocks_before_requests(tmp_path):
    with astra_session() as (mod, astra):
        cls = recording_builder_cls()
        doc = mod.prepare_cases(cls, "hard-verify", ["VideoUnmask"], source_episodes=[3])
        doc["cases"][0]["source_episode"] = 7
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        responder = FakeResponder(astra.champ)
        with pytest.raises(ValueError, match="身份不符"):
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder,
                                          check_calls=[]))
    assert responder.calls == 0


def test_env_sources_point_to_repo_src():
    with astra_session() as (mod, _astra):
        files = mod.assert_env_sources()
    src = str(mod.REPO_SRC.resolve())
    assert files["robomme_hard"].startswith(src) and files["robomme"].startswith(src)


# ── 费用硬上限：零外联夹具 ───────────────────────────────────────────────

PRICES = {"unit": "usd_per_1m_tokens", "input": 2.0, "cached_input": 0.5, "output": 8.0}
#: 2048 个输出 token 的最坏价：2048 × 8 / 1e6
OUT_WORST = 2048 * 8.0 / 1e6


class FakeUrlopen:
    """``urllib.request.urlopen`` 的替身：计数，按序返回 Responses API 样式的回包（或抛 HTTPError）。"""

    def __init__(self, plan=None) -> None:
        self.calls = 0
        self.plan = list(plan or [])

    def __call__(self, request, timeout=None):
        self.calls += 1
        item = self.plan.pop(0) if self.plan else {"input_tokens": 100}
        if isinstance(item, Exception):
            raise item
        body = {"status": "completed", "model": "gpt-6-astra", "reasoning": {"effort": "medium"},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}
        if item is not None:
            body["usage"] = {"input_tokens": item["input_tokens"], "output_tokens": 50,
                             "input_tokens_details": {"cached_tokens": 0},
                             "output_tokens_details": {"reasoning_tokens": 10}}
        return _Resp(body)


class _Resp:
    def __init__(self, body):
        self._b = json.dumps(body).encode()
        self.headers = {"x-request-id": "req-fake"}

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_429():
    headers = Message()
    headers["Retry-After"] = "1"
    return urllib.error.HTTPError("https://example.invalid", 429, "rate", headers,
                                  io.BytesIO(b'{"error":{"code":"rate_limit_exceeded"}}'))


class Clock:
    """假单调钟：``sleep`` 只推进时间并执行回调（模拟等待期间守卫写 STOP）。"""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list = []
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += max(0.0, s)
        if self.on_sleep is not None:
            self.on_sleep(s)


class GuardFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.guard = load_guard()
        self.group = tmp_path / "group_0"
        self.spool = self.group / "run" / "planner_calls"
        self.spool.mkdir(parents=True)
        self.state = tmp_path / "guard" / "ledger.state.json"
        self.ledger = self.guard.new_ledger()
        self.prices = dict(PRICES, cached_input=0.5, reasoning_billed_separately=False)
        self.n = 0

    def round(self) -> dict:
        return self.guard.guard_round([self.group], self.ledger, self.prices, 5.0, 2048, self.state)

    def request(self, text_bytes: int, images: int = 0) -> Path:
        """仿 Astra ``Planner`` 的 spool 布局写一份请求（文本 + 256×256 PNG）。"""
        from PIL import Image
        self.n += 1
        out = self.spool / f"{self.n:032x}"
        out.mkdir()
        names = []
        for i in range(images):
            Image.fromarray(np.full((256, 256, 3), i, np.uint8)).save(out / f"{i}.png")
            names.append(f"{i}.png")
        (out / "request.json").write_text(json.dumps({"images": names, "task": "BinFill", "episode": 0}))
        (out / "prompt.txt").write_text("x" * text_bytes)
        return out

    def reservations(self) -> dict:
        return self.guard.load_reservations(self.state)["reservations"]


def _client(mod, astra, fx: GuardFixture, clock: Clock, gate_clock=None):
    gate = mod.CostGate(fx.state, clock=gate_clock or time.time)
    cls = mod.guarded_client_class(astra.api_client)
    return cls(FAKE_KEY, gate, sleep=clock.sleep, monotonic=clock.monotonic), gate


def _resp(out: Path) -> dict:
    return json.loads((out / "response.json").read_text())


def test_guard_defaults_hard_cap_and_constants(tmp_path, capsys):
    guard = load_guard()
    args = guard.build_parser().parse_args(["--root", "r", "--prices", "p", "--ledger", "l"])
    assert args.cap == 5.0 and args.interval == 2.0 and guard.HARD_CAP_USD == 5.0
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps(PRICES))
    base = ["--root", str(tmp_path), "--prices", str(prices), "--ledger", str(tmp_path / "l.json"), "--once"]
    assert guard.main(base + ["--cap", "5.01"]) == 2 and "ASTRA_COST_BLOCKED" in capsys.readouterr().err
    assert guard.main(base + ["--interval", "6"]) == 2
    assert guard.main(base) == 0
    state = json.loads((tmp_path / "l.state.json").read_text())
    assert state["cap"] == 5.0 and state["stop"] is False and state["schema"] == guard.STATE_SCHEMA
    with astra_session() as (mod, _astra):
        assert mod.GUARD_HEARTBEAT_TIMEOUT_S == guard.HEARTBEAT_TIMEOUT_S == 10.0
        assert mod.guard_module().ASTRA_MAX_EPISODES == 2


def test_first_large_input_refused_without_sending(tmp_path, monkeypatch):
    """首次请求输入就大到「本次最坏」超过 5 美元：发送前拒发，urlopen 0 次；守卫把它计 0、不停机；随后小请求照常。"""
    fake = FakeUrlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        big = fx.request(2_600_000)  # 2.6M 字节 × 2 美元/百万 ≥ 5.2 美元
        client(big)
        assert fake.calls == 0
        refused = _resp(big)
        assert refused["status"] == "error" and "cost reservation refused" in refused["error"]
        assert (big / "guard_refused.json").is_file() and big.name not in fx.reservations()
        summary = fx.round()
        assert summary["refused"] == 1 and summary["usd"] == 0 and summary["stop"] is False
        assert fx.ledger["calls"][big.name]["missing"] == ["refused"]
        small = fx.request(1000, images=2)
        client(small)
    assert fake.calls == 1 and _resp(small)["status"] == "ok"
    res = fx.reservations()[small.name]
    # 估计：1000 字节 + 2 张 256×256（high detail 上界各 255）+ 64
    assert res["input_tokens"] == 1000 + 2 * 255 + 64 and res["state"] == "sent"
    assert res["usd"] == pytest.approx(res["input_tokens"] * 2.0 / 1e6 + OUT_WORST)


def test_input_growth_accumulates_reservations_until_refused(tmp_path, monkeypatch):
    """输入逐次增长、守卫尚未计入回包：未结预留累计，第 4 次（累计会超过 5 美元）在发送前被拒；
    守卫计入实际 usage 后预留结清，同样大小的请求再次放行。"""
    fake = FakeUrlopen([{"input_tokens": 1000}] * 5)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    sizes = [400_000, 700_000, 1_000_000, 1_300_000]
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        outs = []
        for size in sizes:
            outs.append(fx.request(size))
            client(outs[-1])
        assert fake.calls == 3
        assert [_resp(o)["status"] for o in outs] == ["ok", "ok", "ok", "error"]
        assert "cost reservation refused" in _resp(outs[3])["error"]
        worst = [fx.reservations()[o.name]["usd"] for o in outs[:3]]
        assert worst == sorted(worst) and worst[0] < worst[1] < worst[2]
        assert sum(worst) == pytest.approx(sum((s + 64) * 2.0 / 1e6 + OUT_WORST for s in sizes[:3]))
        summary = fx.round()  # 守卫计入三次实际 usage（各 1000 输入），预留结清
        assert summary["calls"] == 4 and summary["refused"] == 1 and summary["stop"] is False
        retry = fx.request(1_300_000)
        client(retry)
    assert fake.calls == 4 and _resp(retry)["status"] == "ok"
    assert clock.slept and all(s >= 0 for s in clock.slept), "发送间隔照第三方 20 秒节奏等待"


def test_missing_usage_writes_stop_and_blocks_next_send(tmp_path, monkeypatch):
    """回包缺 usage：守卫视为超限写 STOP（reason=astra_usage_missing），下一次请求发送前即被拒。"""
    fake = FakeUrlopen([None])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, gate = _client(mod, astra, fx, clock)
        first = fx.request(1000)
        client(first)
        assert fake.calls == 1 and _resp(first)["status"] == "ok" and _resp(first)["usage"] is None
        summary = fx.round()
        assert summary["stop"] is True and summary["reason"] == "astra_usage_missing" and summary["unknown_usage"] == 1
        stop = json.loads((fx.group / "STOP.json").read_text())
        assert stop["reason"] == "astra_usage_missing"
        with pytest.raises(mod.GuardRefused, match="stopped"):
            gate.read_state()
        second = fx.request(1000)
        client(second)
    assert fake.calls == 1
    assert _resp(second)["status"] == "error" and "Host requested stop" in _resp(second)["error"]


def test_stop_arriving_during_wait_blocks_send(tmp_path, monkeypatch):
    """预留通过后进入第三方 20 秒发送间隔，等待期间守卫写 STOP：等待结束复检即拒发，预留释放。"""
    fake = FakeUrlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        client.next_request_at = clock.now + 20
        clock.on_sleep = lambda s: (fx.group / "STOP.json").write_text('{"reason": "astra_cost_cap"}')
        out = fx.request(1000)
        client(out)
    assert clock.slept == [20]
    assert fake.calls == 0 and "Host requested stop" in _resp(out)["error"]
    assert fx.reservations()[out.name]["state"] == "released" and (out / "guard_refused.json").is_file()


def test_retry_after_429_is_gated_again(tmp_path, monkeypatch):
    """429 有界重试：每次重试前重新过闸（复用同一份预留）；重试等待中 STOP 到达则不再发第二次。"""
    fake = FakeUrlopen([_http_429(), {"input_tokens": 10}])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        ok = fx.request(1000)
        client(ok)
        assert fake.calls == 2 and _resp(ok)["status"] == "ok"
        assert list(fx.reservations()) == [ok.name] and (ok / "http_error_00.json").is_file()
        fake.plan = [_http_429(), {"input_tokens": 10}]
        stopped = fx.request(1000)
        client.next_request_at = clock.now  # 首次发送不等待；只有 429 退避（≥10 秒）期间写 STOP
        clock.on_sleep = lambda s: s >= 10 and (fx.group / "STOP.json").write_text("{}")
        client(stopped)
    assert fake.calls == 3, "第二个请求只发出第一次（429），重试前见到 STOP 即停"
    assert "Host requested stop" in _resp(stopped)["error"]


@pytest.mark.parametrize("failure", ["stale", "missing", "exited"])
def test_guard_unavailable_refuses_without_sending(tmp_path, monkeypatch, failure):
    """守卫失联：心跳超过 10 秒、状态文件不存在、守卫已退出——三种都在发送前拒发。"""
    fake = FakeUrlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    gate_clock = time.time
    if failure == "stale":
        gate_clock = lambda: time.time() + 10.5  # noqa: E731
    elif failure == "missing":
        fx.state.unlink()
    else:
        state = json.loads(fx.state.read_text())
        state["exited"] = True
        fx.state.write_text(json.dumps(state))
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, Clock(), gate_clock=gate_clock)
        out = fx.request(1000)
        client(out)
    assert fake.calls == 0
    assert _resp(out)["status"] == "error" and "cost guard unavailable" in _resp(out)["error"]


def test_real_guard_process_crash_heartbeat_goes_stale(tmp_path):
    """真实守卫进程（--interval 0.2）被 SIGKILL：心跳停更，最后一次心跳 10 秒后 runner 拒发、之前仍放行。"""
    fx = GuardFixture(tmp_path)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps(PRICES))
    guard_py = REPO / "scripts" / "eval-official" / "astra_cost_guard.py"
    proc = subprocess.Popen([sys.executable, str(guard_py), "--root", str(fx.group), "--prices", str(prices),
                             "--ledger", str(tmp_path / "guard" / "ledger.json"), "--state", str(fx.state),
                             "--interval", "0.2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 30
        beats = set()
        while time.time() < deadline and len(beats) < 2:
            if fx.state.is_file():
                try:
                    beats.add(json.loads(fx.state.read_text())["heartbeat"])
                except ValueError:
                    pass
            time.sleep(0.05)
        assert len(beats) >= 2, "守卫心跳未更新"
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    last = json.loads(fx.state.read_text())
    assert last["exited"] is False  # 崩溃不会写 exited
    with astra_session() as (mod, _astra):
        assert mod.CostGate(fx.state, clock=lambda: last["heartbeat"] + 5).read_state()["cap"] == 5.0
        with pytest.raises(mod.GuardRefused, match="heartbeat"):
            mod.CostGate(fx.state, clock=lambda: last["heartbeat"] + 10.5).read_state()


def test_runner_cap_never_exceeds_hard_cap(tmp_path):
    """守卫状态里的 cap 被改大也不放宽：生效上限 = min(状态 cap, 5)。"""
    fx = GuardFixture(tmp_path)
    fx.round()
    state = json.loads(fx.state.read_text())
    state["cap"] = 50.0
    fx.state.write_text(json.dumps(state))
    with astra_session() as (mod, _astra):
        gate = mod.CostGate(fx.state)
        assert gate.cap(gate.read_state()) == 5.0
        payload = {"input": [{"content": [{"type": "input_text", "text": "x" * 2_500_000}]}], "max_output_tokens": 2048}
        with pytest.raises(mod.GuardRefused, match="reservation refused"):
            gate.reserve("r" * 32, payload)


def test_episode_cap_two_across_runs(tmp_path):
    """局数硬上限 2：同一守卫预留文件跨 RUN 计数，同一局重复登记不重复计，第 3 局拒绝；
    带 episode_gate 的 run_cases 遇到 3 局清单在建目录与任何请求之前拒绝。"""
    fx = GuardFixture(tmp_path)
    fx.round()
    with astra_session() as (mod, astra):
        gate = mod.CostGate(fx.state)
        assert gate.register_episode("hard-verify:BinFill:0") == 1
        assert gate.register_episode("hard-verify:BinFill:0") == 1
        assert mod.CostGate(fx.state).register_episode("hard-verify:VideoUnmask:0") == 2
        with pytest.raises(mod.AstraStop) as info:
            mod.CostGate(fx.state).register_episode("hard-verify:MoveCube:0")
        assert info.value.reason == "episode_cap"
        cls = recording_builder_cls()
        doc = mod.prepare_cases(cls, "hard-verify", ["BinFill", "VideoUnmask", "MoveCube"], source_episodes=[3])
        args = make_args(tmp_path / "r2", write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        responder = FakeResponder(astra.champ)
        deps = make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder, check_calls=[])
        deps.episode_gate = gate
        with pytest.raises(ValueError, match="astra_episode_cap"):
            mod.run_cases(args, deps)
    assert not Path(args.output).exists() and responder.calls == 0


def test_default_deps_requires_guard_state():
    with astra_session() as (mod, astra):
        with pytest.raises(ValueError, match="cost_guard"):
            mod.default_deps(astra, 0, None)


# ── 启动器：source 不变量与收尾链（bash） ────────────────────────────────
#
# 收尾用例优先用仓库里的真实 S2b 脚本（scripts/eval-official/seat_media_lib.sh、official_media_check.py；合入工作
# 分支后存在，真实 render_official_dir 再调 S2a 重绘器 --source raw）；不存在时退回下面的桩。桩与真实脚本语义一致：
# - render_official_dir <局目录>：恰 1 个位置参数且为目录，否则打印 OFFICIAL_RENDER=FAIL stage=input 返回 2；
#   无帧 error 局（end.no_frame）打印 OFFICIAL_RENDER=NO_FRAME 返回 0；成功返回 0，失败返回 2。
# - transcode_episode_dir [--keep-raw] <局目录>：--keep-raw 只认第一个位置参数；其后恰 1 个目录参数，否则返回 2；
#   无原始帧返回 0（result=none）；--keep-raw 时转码成功也不删原始帧（raw_kept=true）。
# - official_media_check.py 全量模式：--manifest、--ledger、--root、--dataset 缺一即退出 2（同 argparse.error）；
#   账本按 AttemptLedger 口径取第一条 accept 行（accepted_attempt_id、attempt_no，attempt_id 须相同）；
#   dataset、route 与 trace 不符即 FAIL；无帧 error 局计 no_frame_error 不计 fail。

REAL_LIB = REPO / "scripts" / "eval-official" / "seat_media_lib.sh"
REAL_CHECK = REPO / "scripts" / "eval-official" / "official_media_check.py"

STUB_LIB = """
render_official_dir() {
  if [[ $# != 1 || ! -d "$1" ]]; then echo "OFFICIAL_RENDER=FAIL dir=${1:-} stage=input reason=no_dir"; return 2; fi
  "$(tool_py)" "$STUB_HELPER" render "$1" || return 2
}
transcode_episode_dir() {
  local keep=0
  if [[ "${1:-}" == "--keep-raw" ]]; then keep=1; shift; fi
  if [[ $# != 1 || ! -d "$1" ]]; then echo "REC_TRANSCODE dir=${1:-} result=fail detail=bad_args"; return 2; fi
  "$(tool_py)" "$STUB_HELPER" transcode "$1" "$keep"
}
"""

STUB_HELPER = r'''
"""桩：模拟 S2b 的 render_official_dir／transcode_episode_dir 与 official_media_check.py（语义见测试文件注释）。"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["STUB_SCRIPTS"])
import recorder  # noqa: E402


def end_row(d: Path) -> dict:
    return json.loads([ln for ln in (d / "trace.jsonl").read_text().splitlines() if ln.strip()][-1])


def mkmp4(d: Path, out: Path) -> int:
    _, front = recorder.load_frames(d, "front")
    _, wrist = recorder.load_frames(d, "wrist")
    frames = [np.concatenate([f, w], axis=1) for f, w in zip(front, wrist)]
    h, w = frames[0].shape[:2]
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:0", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(out)], input=b"".join(f.tobytes() for f in frames))
    return len(frames) if p.returncode == 0 else -1


cmd = sys.argv[1]
if cmd == "render":
    d = Path(sys.argv[2])
    end = end_row(d)
    if end.get("status") == "error" and end.get("no_frame") is True:
        print(f"OFFICIAL_RENDER=NO_FRAME dir={d.name} source_kind=none")
        sys.exit(0)
    if os.environ.get("STUB_RENDER_FAIL"):
        print(f"OFFICIAL_RENDER=FAIL dir={d.name} stage=render reason=stub")
        sys.exit(2)
    tmp = d / "episode.mp4"
    assert not tmp.exists()
    mkmp4(d, tmp)
    rc = subprocess.run([sys.executable, os.environ["STUB_RENDER_TOOL"], str(d), "--official-root",
                         os.environ["STUB_OFFICIAL_ROOT"], "--jobs", "1", "--threads", "1"]).returncode
    tmp.unlink()
    sys.exit(0 if rc == 0 else 2)
elif cmd == "transcode":
    d, keep = Path(sys.argv[2]), sys.argv[3] == "1"
    if not (d / "front.mkv").is_file():
        print(f"REC_TRANSCODE dir={d.name} result=none")
        sys.exit(0)
    n = mkmp4(d, d / "episode.mp4")
    if not keep:
        for name in ("front.mkv", "wrist.mkv"):
            (d / name).unlink()
    print(f"REC_TRANSCODE dir={d.name} kind=new frames={n} result=ok raw_kept={str(keep).lower()}")
elif cmd == "check":
    ap = argparse.ArgumentParser()
    for a in ("--manifest", "--ledger", "--root"):
        ap.add_argument(a, action="append", default=[])
    ap.add_argument("--dataset")
    ap.add_argument("--route")
    ap.add_argument("--out")
    args = ap.parse_args(sys.argv[2:])
    if not args.manifest or not args.ledger or not args.root or not args.dataset:
        ap.error("全量验收须给 --manifest、--ledger、--root、--dataset（各可重复）")
    row = json.loads(Path(args.manifest[0]).read_text().splitlines()[0])
    accepts = [r for r in map(json.loads, Path(args.ledger[0]).read_text().splitlines()) if r.get("kind") == "accept"
               and r.get("key") == row["key"]]
    reasons = []
    if not accepts or accepts[0].get("attempt_id") != accepts[0].get("accepted_attempt_id"):
        reasons.append("no_accepted_attempt")
        n = None
    else:
        n = accepts[0]["attempt_no"]
    d = Path(args.root[0]) / f"{row['key']}.a{n}"
    rows = [json.loads(ln) for ln in (d / "trace.jsonl").read_text().splitlines() if ln.strip()] if d.is_dir() else []
    if not rows:
        reasons.append("dir_missing")
    else:
        ident, end = rows[0]["identity"], rows[-1]
        if ident.get("dataset") != args.dataset:
            reasons.append("identity:dataset")
        if args.route and rows[0]["route"] != args.route:
            reasons.append("route")
        mp4 = sorted((d / "official").glob("*.mp4")) if (d / "official").is_dir() else []
        no_frame = end.get("status") == "error" and end.get("no_frame") is True
        if not no_frame and len(mp4) != 1:
            reasons.append(f"official_count={len(mp4)}")
    status = "fail" if reasons else ("no_frame_error" if rows and no_frame else "pass")
    Path(args.out).write_text(json.dumps({"key": row["key"], "status": status, "reasons": reasons}) + "\n")
    print(f"OFFICIAL_MEDIA={'FAIL' if reasons else 'PASS'} total=1 skip=0 fail={int(bool(reasons))} "
          f"no_frame_error={int(status == 'no_frame_error')}")
    sys.exit(1 if reasons else 0)
'''


def _media_mode() -> str:
    return "real" if REAL_LIB.is_file() and REAL_CHECK.is_file() else "stub"


def _write_stubs(tmp_path: Path, lib_text: str | None = None, *, prefer_real: bool = False) -> dict:
    """环境：``prefer_real`` 且仓库里有真实 S2b 脚本时用真实脚本（``env['MEDIA_MODE']='real'``），否则用桩。
    ``lib_text`` 给出时总是用这段文本当函数库（source 副作用用例）。"""
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "helper.py").write_text(STUB_HELPER)
    check = stub_dir / "official_media_check.py"
    check.write_text(f"import runpy, sys\nsys.argv = [sys.argv[0], 'check'] + sys.argv[1:]\n"
                     f"runpy.run_path({str(stub_dir / 'helper.py')!r}, run_name='__main__')\n")
    mode = _media_mode() if prefer_real and lib_text is None else "stub"
    lib = stub_dir / "seat_media_lib.sh"
    lib.write_text(lib_text if lib_text is not None else STUB_LIB)
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "STUB_RENDER_FAIL")}
    env.update(SIM_PYTHON=sys.executable, SEAT_MEDIA_LIB=str(REAL_LIB if mode == "real" else lib),
               OFFICIAL_MEDIA_CHECK=str(REAL_CHECK if mode == "real" else check), STUB_HELPER=str(stub_dir / "helper.py"),
               STUB_SCRIPTS=str(REPO / "scripts" / "eval-official"), STUB_RENDER_TOOL=str(RENDER),
               STUB_OFFICIAL_ROOT=str(third_party().parent.parent), MAX_STEPS="1300", MEDIA_MODE=mode)
    return env


def _finish(run: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), "--finish", str(run)], capture_output=True, text=True, env=env,
                          timeout=240)


def test_launcher_static_order_and_explicit_interpreters():
    """主流程：守卫状态必给、解释器显式设、source 在起服务与设 trap 之前；check／run 都带 --guard-state；
    收尾三步顺序为 render → transcode → 验收，调用形状与 S2b 真实签名一致。"""
    text = SCRIPT.read_text()
    assert 'BENCH_PY="${BENCH_PY:-$SIM_PYTHON}"' in text and 'TOOL_PY="${TOOL_PY:-$BENCH_PY}"' in text
    assert "export BENCH_PY TOOL_PY" in text
    main = text[text.index('if [[ $# != 2 ]]; then\n  echo \'Usage: bash scripts/eval-official/run_astra.sh '
                           '--policy-seed <n> CASES'):]
    i_seed = main.index("RUN_BLOCKED reason=policy_seed")
    i_guard = main.index(': "${ASTRA_GUARD_STATE:?')
    assert i_seed < i_guard, "模型 seed 缺失须在任何副作用之前拒绝"
    assert main.count('--policy-seed "$POLICY_SEED"') == 2, "check 与 run 都转发模型 seed"
    i_source = main.index("source_media_lib || exit $?")
    i_check = main.index(" check --cases ")
    i_vla = main.index("scripts/serve_policy.py")
    i_trap = main.index("trap cleanup EXIT")
    i_finish = main.index('finish_run "$RUN"')
    assert i_guard < i_source < i_check < i_vla < i_trap < i_finish
    assert main.count('--guard-state "$ASTRA_GUARD_STATE"') == 2
    fe = text[text.index("finish_episode() {"):text.index("finish_run() {")]
    assert fe.index('render_official_dir "$d"') < fe.index('transcode_episode_dir "$d"') < fe.index("astra_media_check")
    assert 'transcode_episode_dir --keep-raw "$d"' in fe and '"$d" --keep-raw' not in text
    assert "official-render.failed" in fe
    mc = text[text.index("astra_media_check() {"):text.index("finish_episode() {")]
    for flag in ("--manifest", "--ledger", "--root", '--dataset "$dataset"', "--route astra/new", "--out"):
        assert flag in mc, flag


def test_real_media_scripts_interface_if_present():
    """仓库里已有真实 S2b 脚本时（合入后），核对本启动器依赖的接口仍在：函数名与 --keep-raw 首参、
    official_media_check.py 的四个必需参数。没有时打印未验证说明（桩用例覆盖同一语义）。"""
    if _media_mode() == "stub":
        print("ASTRA_MEDIA_MODE=stub（S2b 真实脚本不在本检出，接口由桩用例覆盖）")
        return
    lib = REAL_LIB.read_text()
    assert "render_official_dir()" in lib and "transcode_episode_dir()" in lib
    assert 'if [[ "${1:-}" == "--keep-raw" ]]; then keep=1; shift; fi' in lib
    proc = subprocess.run([sys.executable, str(REAL_CHECK), "--manifest", "x", "--ledger", "y", "--root", "z"],
                          capture_output=True, text=True)
    assert proc.returncode == 2 and "--dataset" in proc.stderr
    print("ASTRA_MEDIA_MODE=real")


@pytest.mark.slow
@pytest.mark.parametrize("lib,what", [
    ("MAX_STEPS=1600\n", "MAX_STEPS"),
    ("trap 'echo hijack' EXIT\n", "trap"),
    ("set +u\n", "shell_options"),
    ("render_official_dir() { :; }\n", "function=transcode_episode_dir"),
])
def test_source_media_lib_side_effects_blocked(tmp_path, lib, what):
    """source 席位函数库后 MAX_STEPS、trap、shell 选项必须不变、两个函数必须在；否则 RUN_BLOCKED 退出 3。"""
    run = tmp_path / "group_0" / "run"
    (run / "results").mkdir(parents=True)
    text = lib if what.startswith("function") else STUB_LIB + lib
    proc = _finish(run, _write_stubs(tmp_path, text))
    assert proc.returncode == 3, proc.stderr
    assert "RUN_BLOCKED" in proc.stderr and what in proc.stderr


@pytest.mark.slow
def test_source_media_lib_benign_keeps_max_steps_and_trap(tmp_path):
    """桩库与（若在）真实库各 source 一次：MAX_STEPS 与 trap 不变；库文件缺失即 RUN_BLOCKED。"""
    run = tmp_path / "group_0" / "run"
    (run / "results").mkdir(parents=True)
    libs = [_write_stubs(tmp_path)] + ([_write_stubs(tmp_path, prefer_real=True)] if _media_mode() == "real" else [])
    for env in libs:
        proc = _finish(run, env)
        assert proc.returncode == 0, proc.stderr
        assert "ASTRA_MEDIA_LIB=OK" in proc.stdout and "max_steps=1300 traps_unchanged=1" in proc.stdout
        assert "ASTRA_FINISH_SUMMARY" in proc.stdout and "total=0 fail=0" in proc.stdout
    env = _write_stubs(tmp_path)
    env["SEAT_MEDIA_LIB"] = str(tmp_path / "absent.sh")
    proc = _finish(run, env)
    assert proc.returncode == 3 and "missing_dependency missing=S2b" in proc.stderr


@pytest.mark.slow
def test_stub_semantics_match_real_signatures(tmp_path):
    """桩的位置敏感性与参数校验：--keep-raw 放在目录之后、验收缺 --dataset，都与真实脚本一样失败。"""
    env = _write_stubs(tmp_path)
    d = tmp_path / "X_xhard0_1.a1"
    d.mkdir()
    script = f'tool_py() {{ echo "$TOOL_PY"; }}; source "$SEAT_MEDIA_LIB"; transcode_episode_dir "{d}" --keep-raw'
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**env, "TOOL_PY": sys.executable})
    assert proc.returncode == 2 and "bad_args" in proc.stdout
    proc = subprocess.run([sys.executable, env["OFFICIAL_MEDIA_CHECK"], "--manifest", "m", "--ledger", "l", "--root",
                           str(tmp_path)], capture_output=True, text=True, env=env)
    assert proc.returncode == 2 and "--dataset" in proc.stderr


def _cpu_run(tmp_path: Path, monkeypatch, task: str = "VideoUnmask", steps: int = 20, env_plan=None
             ) -> tuple[Path, Path, int]:
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        cls = recording_builder_cls(env_plan or (lambda b, ep: FakeEnv(terminal_step=steps)))
        doc = mod.prepare_cases(cls, "hard-verify", [task], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(),
                                      responder=FakeResponder(astra.champ), check_calls=[]))
    run = Path(args.output).parent
    a_dir = _episode_dir(Path(args.output), task, 0)
    return run, a_dir, _trace(a_dir / "trace.jsonl")[-1]["frames_recorded"]


def _count_frames(mp4: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True)
    return int(out.stdout.strip())


@pytest.mark.slow
def test_launcher_finish_makes_official_and_plain_video(tmp_path, monkeypatch):
    """CPU 夹具：真实 runner 出局目录 → 真实启动器 ``--finish`` 收尾 → 一份官方视频（官方原类重绘）与一份普通视频
    ``episode.mp4``，原始帧转码后删除，官方视频验收通过。仓库有真实 S2b 脚本时用真实脚本，否则用桩（断言用的哪种）。"""
    run, a_dir, frames = _cpu_run(tmp_path, monkeypatch)
    env = _write_stubs(tmp_path, prefer_real=True)
    assert env["MEDIA_MODE"] == _media_mode()
    assert env["SEAT_MEDIA_LIB"] == str(REAL_LIB if env["MEDIA_MODE"] == "real" else tmp_path / "stubs" / "seat_media_lib.sh")
    print(f"ASTRA_MEDIA_MODE={env['MEDIA_MODE']}")
    proc = _finish(run, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OFFICIAL_RENDER=PASS" in proc.stdout and "OFFICIAL_MEDIA=PASS" in proc.stdout
    assert f"ASTRA_FINISH dir={a_dir} render_rc=0 transcode_rc=0 media_check_rc=0" in proc.stdout
    assert "ASTRA_FINISH_SUMMARY" in proc.stdout and "total=1 fail=0" in proc.stdout
    official = sorted((a_dir / "official").glob("*.mp4"))
    assert len(official) == 1 and "VideoUnmask_ep3a1_success_" in official[0].name
    assert _count_frames(official[0]) == frames
    assert (a_dir / "episode.mp4").is_file() and _count_frames(a_dir / "episode.mp4") == frames
    assert not (a_dir / "front.mkv").exists() and not (a_dir / "official-render.failed").exists()
    key = a_dir.name.rsplit(".a", 1)[0]
    ledger = [json.loads(x) for x in (run / "official-media" / f"{key}.ledger.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in ledger] == ["attempt_start", "accept"]


@pytest.mark.slow
def test_launcher_render_failure_keeps_raw_frames(tmp_path, monkeypatch):
    """重绘失败：写 official-render.failed，转码以 ``--keep-raw <局目录>`` 调用（原始帧保留、普通视频照出），
    本局计失败、退出 5。转码用真实库（若在）或桩，重绘函数被覆写为必失败。"""
    run, a_dir, frames = _cpu_run(tmp_path, monkeypatch, task="BinFill", steps=8)
    env = _write_stubs(tmp_path, prefer_real=True)
    base_lib = env["SEAT_MEDIA_LIB"]
    failing = tmp_path / "stubs" / "failing_render_lib.sh"
    failing.write_text(f'source "{base_lib}"\n'
                       'render_official_dir() { echo "OFFICIAL_RENDER=FAIL dir=$(basename "$1") stage=render '
                       'reason=forced"; return 2; }\n')
    env["SEAT_MEDIA_LIB"] = str(failing)
    before = {n: (a_dir / n).read_bytes() for n in ("front.mkv", "wrist.mkv")}
    proc = _finish(run, env)
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert (a_dir / "official-render.failed").is_file() and "render_rc=2" in proc.stdout
    assert "raw_kept=true" in proc.stdout or json.loads((a_dir / "transcode.json").read_text()).get("raw_kept") is True
    assert {n: (a_dir / n).read_bytes() for n in before} == before, "原始帧必须原样保留"
    assert (a_dir / "episode.mp4").is_file() and _count_frames(a_dir / "episode.mp4") == frames
    assert "ASTRA_FINISH_SUMMARY" in proc.stdout and "total=1 fail=1" in proc.stdout


@pytest.mark.slow
def test_launcher_finish_no_frame_episode(tmp_path, monkeypatch):
    """环境没建起来的无帧 error 局：重绘返回 NO_FRAME（0）、转码无媒体（0）、验收计 no_frame_error 不计 fail。"""

    def boom(builder, ep):
        raise RuntimeError("fake simulator start failure")

    run, a_dir, frames = _cpu_run(tmp_path, monkeypatch, task="BinFill", env_plan=boom)
    assert frames == 0
    env = _write_stubs(tmp_path, prefer_real=True)
    proc = _finish(run, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NO_FRAME" in proc.stdout and "no_frame_error=1" in proc.stdout
    assert "total=1 fail=0" in proc.stdout and not list((a_dir / "official").glob("*.mp4") if (a_dir / "official").is_dir() else [])
