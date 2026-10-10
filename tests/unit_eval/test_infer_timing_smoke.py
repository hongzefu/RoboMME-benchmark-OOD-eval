"""单块停止与共享预算：使用会话替身，验证真实序列化账本，不启动环境。"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from robomme_ood_eval import episode, timing
from robomme_ood_eval.models import framesamp_modul as FM, smvla as SM
from robomme_ood_eval.session import NullRecorder

ROOT = Path(__file__).resolve().parents[2]


def load_smoke():
    spec = importlib.util.spec_from_file_location("it_smoke_test", ROOT / "dev-scripts/checks/infer_timing_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs():
    return {"front_rgb_list": [np.zeros((2, 2, 3), np.uint8)],
            "wrist_rgb_list": [np.zeros((2, 2, 3), np.uint8)],
            "joint_state_list": [np.zeros(7, np.float32)],
            "gripper_state_list": [np.zeros(2, np.float32)]}


def _ongoing(action=None):
    return _obs(), 0.0, False, False, {"status": "ongoing", "action": action}


@pytest.mark.parametrize("audit", [False, True])
def test_single_chunk_stops_without_extra_environment_step(monkeypatch, audit):
    module = load_smoke()
    class FakeSession:
        def __init__(self):
            self._cap_hit = False
        def step(self, action):
            return _ongoing(action)
    ep = SimpleNamespace(EnvSession=FakeSession, StepCapReached=episode.StepCapReached)
    # 作用域结束恢复本进程计时方法，避免污染其它测试。
    monkeypatch.setattr(timing.ChunkTimer, "close", timing.ChunkTimer.close)
    state = module.install_stop(ep, timing, audit=audit)
    session = ep.EnvSession()
    if audit:
        session.step("hold")
        timer = timing.ChunkTimer(clock=lambda: 1.0)
        timer.start(0.0)
        timer.close(decision=0, env_step=1, action_rtt_ms=1, server_infer_ms=1)
    obs, reward, terminated, truncated, info = session.step("action")
    assert obs is not None and reward == 0.0 and not terminated and truncated
    assert info["status"] == "timeout" and info["single_chunk_smoke"]["task_success"] is False
    with pytest.raises(RuntimeError, match="驱动未遵守冒烟截断"):
        session.step("extra")
    assert not session._cap_hit and state["stopped"]
    assert state["steps"] == (2 if audit else 1)
    assert state["steps_after_chunk"] == int(audit)


@pytest.mark.parametrize("formal", [False, True])
def test_main_charges_build_reset_and_preserves_formal_loop(tmp_path, monkeypatch, formal):
    module = load_smoke()
    out = tmp_path / "out"
    ledger_path = tmp_path / "shared.jsonl"
    monkeypatch.setenv("ROBOMME_EVAL_ROOT", str(ROOT))
    monkeypatch.setenv("IT_BUDGET_LEDGER", str(ledger_path))
    monkeypatch.setenv("SGEVAL_AUDIT", "1")
    monkeypatch.setattr(sys, "argv", ["entry", "--model", "astra", "--dataset", "hard-verify",
                                     "--tasks", "VideoUnmask", "--episodes", "0:1", "--out", str(out)]
                                    + (["--it-formal"] if formal else []))
    class FakeSession:
        def __init__(self, *, claim_reset):
            self._cap_hit = False
            claim_reset("build")
            claim_reset("reset")
        def step(self, action):
            return _ongoing(action)
    monkeypatch.setattr(episode, "EnvSession", FakeSession)
    monkeypatch.setattr(timing.ChunkTimer, "close", timing.ChunkTimer.close)
    def fake_entry(path, *, run_name):
        assert Path(path) == ROOT / "scripts/evaluate.py" and run_name == "__main__"
        assert "--it-formal" not in sys.argv
        session = episode.EnvSession()
        timer = timing.ChunkTimer(clock=lambda: 1.0)
        timer.start(0.0)
        timer.close(decision=0, env_step=0, action_rtt_ms=1, server_infer_ms=1)
        first = session.step("first")
        if formal:
            assert first[3] is False and "single_chunk_smoke" not in first[4]
            assert session.step("second")[4]["action"] == "second"
        else:
            assert first[3] is True and first[4]["status"] == "timeout"
            with pytest.raises(RuntimeError, match="驱动未遵守冒烟截断"):
                session.step("second")
    monkeypatch.setattr(module.runpy, "run_path", fake_entry)
    module.main()
    budget = sys.modules["it_shared_budget"].BudgetLedger(ledger_path, trajectory_cap=27, reset_cap=54,
                                                         astra_cap=3, shared_infra_cap=6, expired_cap=0,
                                                         planned_first_tries=21)
    ok, lines = budget.report_lines()
    assert ok
    assert any("trajectories=1/27" in line and "resets=2/54" in line for line in lines)
    print(f"SINGLE_CHUNK_BUDGET=PASS mode={'formal' if formal else 'smoke'} trajectories=1 resets=2 real_resets=0")


def test_interrupted_same_out_charges_separate_attempts(tmp_path, monkeypatch):
    module = load_smoke()
    ledger_path = tmp_path / "shared.jsonl"
    out = tmp_path / "not_created"
    monkeypatch.setenv("ROBOMME_EVAL_ROOT", str(ROOT))
    monkeypatch.setenv("IT_BUDGET_LEDGER", str(ledger_path))
    argv = ["entry", "--model", "astra", "--dataset", "hard-verify", "--tasks", "VideoUnmask",
            "--episodes", "0:1", "--out", str(out), "--it-formal"]
    def interrupted(*args, **kwargs):
        raise RuntimeError("模拟创建输出前中断")
    monkeypatch.setattr(module.runpy, "run_path", interrupted)
    monkeypatch.setattr(episode, "EnvSession", episode.EnvSession)
    for _ in range(3):
        monkeypatch.setattr(sys, "argv", list(argv))
        with pytest.raises(RuntimeError, match="模拟创建输出前中断"):
            module.main()
    budget = sys.modules["it_shared_budget"].BudgetLedger(ledger_path, trajectory_cap=27, reset_cap=54,
                                                         astra_cap=3, shared_infra_cap=6, expired_cap=0,
                                                         planned_first_tries=21)
    assert budget._load().trajectories == 3 and budget._load().astra == 3
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(RuntimeError, match="astra=3/3") as refusal:
        module.main()
    assert refusal.value.reason == "astra_cap"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(ValueError, match="输出目录已存在"):
        module.main()
    assert budget._load().trajectories == 3
    print("SMOKE_ATTEMPTS=PASS interrupted=3 charged=3 fourth_refused=1 existing_out_refused=1 real_resets=0")


class _Recorder(NullRecorder):
    """记录 CPU 替身的真实会话事件，供核对原始终态与私有截断标记。"""
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)


class _Env:
    """不创建仿真，只返回固定观测或故意触发错误。"""
    def __init__(self, behavior="ongoing"):
        self.n = 0
        self.behavior = behavior

    def reset(self):
        return _obs(), {"status": "ongoing", "task_goal": "test"}

    def step(self, action):
        self.n += 1
        if self.behavior == "raise":
            raise AttributeError("真实环境异常不能当成冒烟成功")
        if self.behavior == "none":
            return None, 0.0, False, False, {"status": "ongoing"}
        if self.behavior in ("success", "fail", "error"):
            return _obs(), 0.0, True, False, {"status": self.behavior, "error_message": "原始错误"}
        return _ongoing(action)

    def close(self):
        pass


class _FrameClient:
    """假模型消息；推理本体与 FrameSamp 驱动来自生产代码。"""
    def __init__(self, audit, *, fail_infer=False):
        self.calls = []
        self.fail_infer = fail_infer
        self._last_rtt = {"infer": 0.0, "add_buffer": 0.0}
        self._last_server_timing = {"infer_ms": 0.0, "gpu": "fake"} if audit else None
        self._ws = SimpleNamespace(close=lambda: None)

    def reset(self):
        self.calls.append("reset")
        return {"reset_finished": True}

    def add_buffer(self, buffer):
        self.calls.append("add_buffer")
        return {"add_buffer_finished": True}

    def infer(self, observation):
        self.calls.append("infer")
        if self.fail_infer:
            raise AttributeError("真实模型异常不能当成冒烟成功")
        return {"actions": np.zeros((20, 8), np.float32)}


class _SmvlaConn:
    """为真实 SimpleMemVLA 驱动提供完整的请求指纹协议。"""
    metadata = {"gpu_name": "fake"}

    def __init__(self, audit):
        self.audit = audit
        self.calls = []

    def call(self, msg):
        raw = pickle.dumps(msg)
        kind = next(iter(msg))
        self.calls.append(kind)
        body = msg[kind]
        reply = {"req_sha": hashlib.sha256(raw).hexdigest()}
        if kind == "reset":
            reply["rng"] = "fake"
        elif kind == "observe":
            reply.update(n=len(body["frames"]), observe_time_ms=0.0,
                         frame_sha=[{cam: SM.frame_sha(fr[cam]) for cam in ("front", "wrist")}
                                    for fr in body["frames"]])
        else:
            actions = np.zeros((16, 8), np.float32)
            reply.update(recv_state_sha=SM.array_sha(body["state"]),
                         recv_instruction_sha=hashlib.sha256(body["instruction"].encode()).hexdigest(),
                         infer_ms=0.0, actions=actions, actions_full=actions, subtask="test")
            if self.audit:
                reply[SM.AUDIT_KEY] = {"timing": {"lang_ms": 0.0, "action_ms": 0.0}}
        return reply, raw, pickle.dumps(reply)


def _drive(tmp_path, monkeypatch, *, model, audit, behavior="ongoing", fail_infer=False):
    """真实会话、真实模型驱动、真实 trace；只有环境和模型通信是 CPU 替身。"""
    module = load_smoke()
    monkeypatch.setenv("SGEVAL_AUDIT", "1" if audit else "0")
    monkeypatch.setattr(timing.ChunkTimer, "close", timing.ChunkTimer.close)
    ep = SimpleNamespace(EnvSession=episode.EnvSession)
    state = module.install_stop(ep, timing, audit=audit)
    env = _Env(behavior)
    rec = _Recorder()
    builder = SimpleNamespace(make_env_for_episode=lambda ep: env)
    sess = ep.EnvSession("VideoUnmask", 0, dataset="hard-verify", max_steps=1300, recorder=rec, builder=builder)
    ident = {"task": "VideoUnmask", "key": "fake", "episode": 0, "seed": 0, "source_episode": 0}
    conn_info = {"host": "unused", "port": 1, "trace_path": str(tmp_path / "trace.jsonl"), "max_steps": 1300}
    if model == "framesamp":
        client = _FrameClient(audit, fail_infer=fail_infer)
        monkeypatch.setattr(FM, "make_recording_client", lambda *args: client)
        result = FM.run_episode(sess, ident, conn_info, rec)
    else:
        client = _SmvlaConn(audit)
        result = SM.run_episode(sess, ident, conn_info, rec, conn=client)
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    return result, sess, env, rec, client, state, trace


@pytest.mark.parametrize("model", ["framesamp", "smvla"])
@pytest.mark.parametrize("audit", [True, False])
def test_real_drivers_stop_cleanly_after_one_actual_step(tmp_path, monkeypatch, model, audit):
    """反例回归：真实驱动不再把外层停止包装成属性错误或漏记已执行步。"""
    result, sess, env, rec, client, state, trace = _drive(tmp_path, monkeypatch, model=model, audit=audit)
    assert result["status"] == "timeout" and result["task_success"] is False, result["error"]
    assert result["error"] is None and result.get("env_exception") is None
    assert result["steps"] == sess.steps == env.n == state["steps"] == 1
    assert result["decisions"] == client.calls.count("infer") == len(result["timing"]["chunks"]) == 1
    assert not sess.cap_hit and state["stopped"] and state["stop_reason"] == "first_chunk"
    original_steps = [e for e in rec.events if e["kind"] == "env_step"]
    stops = [e for e in rec.events if e["kind"] == "single_chunk_smoke_stop"]
    assert len(original_steps) == len(stops) == 1
    assert original_steps[0]["status"] == "ongoing" and original_steps[0]["truncated"] is False
    assert stops[0]["steps"] == 1 and stops[0]["task_success"] is False
    assert trace[-1]["steps_attempted"] == trace[-1]["steps_observed"] == 1
    assert trace[-1]["status"] == "timeout" and trace[-1]["cap_hit"] is False
    assert trace[-1]["observer_hook_errors"] == 0
    # 序列化后再按真实外层分类，三个步数来源仍一致，且不是任务成功。
    classified = episode.EpisodeResult(dataset="hard-verify", task="VideoUnmask", episode=0, source_episode=0,
                                       tier="xhard0", seed=0, candidate=None, spec_sha256=None, key="fake",
                                       max_steps=1300, strict_cap=False, attempt=1, policy_seed=0, model=model,
                                       policy_label=model)
    episode._classify(classified, json.loads(json.dumps(result)), sess)
    assert classified.steps == classified.exec_steps == 1 and classified.task_success == 0
    assert classified.status == "timeout" and classified.error is None
    print(f"SMOKE_DRIVER_STOP=PASS model={model} audit={int(audit)} actual_steps=1 result_steps=1 infer=1 "
          "task_success=0 trace_steps=1")


@pytest.mark.parametrize("model", ["framesamp", "smvla"])
@pytest.mark.parametrize("behavior", ["raise", "none", "error", "success", "fail"])
def test_real_driver_errors_and_natural_terminal_are_not_smoke_success(tmp_path, monkeypatch, model, behavior):
    """真实异常、空观测与自然终态沿原路径处理，私有入口不改写成控制超时。"""
    result, sess, env, rec, client, state, trace = _drive(tmp_path, monkeypatch, model=model, audit=True,
                                                      behavior=behavior)
    assert env.n == sess.steps == 1 and client.calls.count("infer") == 1
    assert not state["stopped"] and state["stop_reason"] is None and not sess.cap_hit
    assert not any(e["kind"] == "single_chunk_smoke_stop" for e in rec.events)
    if behavior in ("success", "fail"):
        assert result["status"] == behavior and result["task_success"] == (behavior == "success")
    else:
        assert result["status"] == "error" and result["task_success"] is False and result["error"]
        if behavior == "raise":
            assert "真实环境异常不能当成冒烟成功" in (result["error"] + (result.get("env_exception") or ""))


def test_real_framesamp_model_attribute_error_is_not_hidden(tmp_path, monkeypatch):
    """普通模型 AttributeError 保持失败，不能按错误类型宽泛地当成停止成功。"""
    result, sess, env, rec, client, state, trace = _drive(tmp_path, monkeypatch, model="framesamp", audit=True,
                                                      fail_infer=True)
    assert result["status"] == "error" and result["task_success"] is False
    assert result["error"] == "AttributeError: 真实模型异常不能当成冒烟成功"
    assert sess.steps == env.n == state["steps"] == 0 and not state["stopped"]
    assert result["timing"]["chunks"] == [] and not state["chunk"]


def test_audit_without_chunk_stops_at_sixteen_and_reports_missing_chunk(monkeypatch):
    """首块前的保护上限仍为十六步；保护退出明确标记没有合法块，不冒充块通过。"""
    module = load_smoke()
    class Session:
        def step(self, action):
            return _ongoing(action)
    ep = SimpleNamespace(EnvSession=Session)
    monkeypatch.setattr(timing.ChunkTimer, "close", timing.ChunkTimer.close)
    state = module.install_stop(ep, timing, audit=True)
    session = ep.EnvSession()
    for step in range(16):
        out = session.step(step)
        assert out[3] == (step == 15)
    assert state["steps"] == 16 and not state["chunk"] and state["steps_after_chunk"] == 0
    assert state["stopped"] and state["stop_reason"] == "no_chunk_step_limit"
    assert out[4]["single_chunk_smoke"]["chunk"] is False
    with pytest.raises(RuntimeError, match="驱动未遵守冒烟截断"):
        session.step("extra")
    assert state["steps"] == 16


def test_main_missing_chunk_fails_and_still_commits_budget(tmp_path, monkeypatch):
    """审计开但没有合法块时主入口非正常退出，已消耗的预算仍如实结算。"""
    module = load_smoke()
    ledger_path = tmp_path / "shared.jsonl"
    monkeypatch.setenv("ROBOMME_EVAL_ROOT", str(ROOT))
    monkeypatch.setenv("IT_BUDGET_LEDGER", str(ledger_path))
    monkeypatch.setenv("SGEVAL_AUDIT", "1")
    monkeypatch.setattr(sys, "argv", ["entry", "--model", "astra", "--dataset", "hard-verify",
                                     "--tasks", "VideoUnmask", "--episodes", "0:1", "--out", str(tmp_path / "out")])
    class Session:
        def __init__(self, *, claim_reset):
            claim_reset("build")
            claim_reset("reset")
        def step(self, action):
            return _ongoing(action)
    monkeypatch.setattr(episode, "EnvSession", Session)
    monkeypatch.setattr(timing.ChunkTimer, "close", timing.ChunkTimer.close)
    def entry_without_chunk(*args, **kwargs):
        session = episode.EnvSession()
        for i in range(16):
            out = session.step(i)
        assert out[3] is True and out[4]["single_chunk_smoke"]["chunk"] is False
    monkeypatch.setattr(module.runpy, "run_path", entry_without_chunk)
    with pytest.raises(RuntimeError, match="SINGLE_CHUNK_SMOKE=FAIL"):
        module.main()
    budget = sys.modules["it_shared_budget"].BudgetLedger(ledger_path, trajectory_cap=27, reset_cap=54,
                                                         astra_cap=3, shared_infra_cap=6, expired_cap=0,
                                                         planned_first_tries=21)
    state = budget._load()
    assert state.trajectories == 1 and state.resets == 2 and state.astra == 1
    print("SMOKE_MISSING_CHUNK=PASS actual_steps=16 smoke_result=FAIL budget_preserved=1 real_resets=0")
