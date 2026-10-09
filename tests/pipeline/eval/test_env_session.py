"""C13 ``robomme_hard_eval.session.EnvSession``（拆仓前在 env_client.py）：一局环境的 build／reset／step／close 与录制、额度、步数上限。

假环境与假 builder 的行为由本文件手写；核对的是 EnvSession 交给环境与录制器的东西。
"""
from __future__ import annotations

import numpy as np
import pytest

import eval_fakes as F


class _Builder:
    def __init__(self, plan=None):
        self.calls = []
        self.env = None
        self.plan = plan or F.Plan(success_at=3)

    def make_env_for_episode(self, ep, max_steps=None):
        self.calls.append((ep, max_steps))
        self.env = F.FakeEnv("T", ep, self.plan)
        return self.env


def _session(**kw):
    ec = F.env_session()
    b = kw.pop("builder", None) or _Builder(kw.pop("plan", None))
    rec = kw.pop("recorder", None) or F.FakeRecorder("/nonexistent-not-written", {})
    return ec.EnvSession("T", 5, builder=b, recorder=rec, **kw), b, rec


def test_reset_returns_env_output_unchanged_and_switches_phase():
    s, b, rec = _session(max_steps=37)
    obs, info = s.reset()
    want = F.obs_of(F.reset_values(5))
    for k in want:
        assert all(np.array_equal(x, y) for x, y in zip(obs[k], want[k]))
    assert info["task_goal"][0] == "goal-T-5" and s.task_goal == "goal-T-5"
    assert b.calls == [(5, None)]  # build 一次；步数不逐局传，由 builder 构造参数决定
    assert rec.phases == ["reset", "reset", "run"]  # build、reset 期间只入队，reset 返回后切 run
    assert rec.frames == {"front": F.N_RESET_FRAMES, "wrist": F.N_RESET_FRAMES}
    assert s.timing["demo_frames"] == F.N_RESET_FRAMES - 1


def test_step_passes_action_unchanged_and_records_exec_action():
    s, b, rec = _session()
    s.reset()
    a = np.arange(8, dtype=np.float64) / 7
    out = s.step(a)
    assert np.array_equal(b.env.actions[0], a) and b.env.actions[0].dtype == np.float64
    assert out[4]["status"] == "ongoing" and s.steps == 1
    assert rec.arrays["exec_action"] == 1
    ev = [e for e in rec.events if e["kind"] == "env_step_action"][0]
    assert ev["dtype"] == a.dtype.str and ev["shape"] == [8]


def test_step_exception_propagates_and_counts_step():
    s, b, rec = _session(plan=F.Plan(raise_at=1, raise_exc=lambda: ValueError("坏动作")))
    s.reset()
    with pytest.raises(ValueError, match="坏动作"):
        s.step(np.zeros(8))
    assert s.steps == 1
    assert [e["kind"] for e in rec.events][-1] == "env_step_exception"


def test_step_cap_stops_before_env():
    ec = F.env_session()
    s, b, rec = _session(plan=F.Plan(), step_cap=4)
    s.reset()
    for _ in range(4):
        s.step(np.zeros(8))
    with pytest.raises(ec.StepCapReached):
        s.step(np.zeros(8))
    assert b.env.n == 4 and s.steps == 4 and s.cap_hit is True


def test_success_on_last_allowed_step_is_returned():
    s, b, rec = _session(plan=F.Plan(success_at=4), step_cap=4)
    s.reset()
    for _ in range(3):
        s.step(np.zeros(8))
    out = s.step(np.zeros(8))
    assert out[2] is True and out[4]["status"] == "success" and s.cap_hit is False


def test_claim_reset_order_and_budget_flag():
    ec = F.env_session()
    claims = []

    def claim(what):
        if len(claims) >= 1:
            raise ec.ResetBudgetExhausted("额度用尽")
        claims.append(what)

    s, b, rec = _session(claim_reset=claim)
    with pytest.raises(ec.ResetBudgetExhausted):
        s.reset()
    assert claims == ["build"] and s.budget_exhausted is True and s.reset_calls == 1
    assert b.env.resets == 0  # 额度被拒时 reset 不进入环境


def test_recorder_failure_becomes_recorder_error():
    ec = F.env_session()
    rec = F.FakeRecorder("/nonexistent-not-written", {}, fail_on="add_frames")
    s, b, _ = _session(recorder=rec)
    with pytest.raises(ec.RecorderError, match="No space left"):
        s.reset()


def test_obs_none_step_is_recorded_without_frames():
    class NoneObsEnv(F.FakeEnv):
        def step(self, action):
            self.n += 1
            return None, 0.0, True, False, {"status": "error", "error_message": "IK"}

    b = _Builder()
    b.make_env_for_episode = lambda ep, max_steps=None: NoneObsEnv("T", ep, F.Plan())
    s, _, rec = _session(builder=b)
    s.reset()
    out = s.step(np.zeros(8))
    assert out[0] is None
    ev = [e for e in rec.events if e["kind"] == "env_step"][0]
    assert ev["obs_none"] is True and ev["status"] == "error"


def test_close_is_safe_and_releases_env():
    s, b, rec = _session()
    s.reset()
    s.step(np.zeros(8))
    s.close()
    assert b.env.closed is True and s.env is None
    assert s.timing["step_n"] == 1
    s.close()  # 再次 close 不抛


def test_own_builder_uses_dataset_and_max_steps():
    """不注入 builder 时按 dataset 与 max_steps 自建真实 builder（只解析身份，不建场景）；缺 max_steps 即拒绝。"""
    ec = F.env_session()
    s = ec.EnvSession("PickXtimes", 0, max_steps=1300, dataset="hard-verify")
    assert s.builder.dataset == "hard-verify"
    # 拆仓后 EnvSession 不再有 identity()（身份解析归 episode.make_spec），直接核自建 builder 的解析结果
    assert s.builder.resolve_identity(0)["tier"] == "xhard0"
    s9 = ec.EnvSession("PickXtimes", 0, max_steps=1600)  # 默认 ood（V9 不变）
    assert s9.builder.dataset == "ood"
    with pytest.raises(ValueError, match="max_steps"):
        _ = ec.EnvSession("PickXtimes", 0, dataset="hard-verify").builder
    with pytest.raises(ValueError):
        ec.EnvSession("PickXtimes", 0, max_steps=1300, dataset="test")
