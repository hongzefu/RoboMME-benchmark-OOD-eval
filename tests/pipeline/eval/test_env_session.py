"""C13 ``robomme_ood_eval.session.EnvSession`` (in env_client.py before the repo split): build / reset / step / close of
one episode's environment, plus recording, budget and step cap.

The fake environment and fake builder behavior is hand-written in this file; the tests check what EnvSession hands to
the environment and the recorder.
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
    assert b.calls == [(5, None)]  # built once; the step cap comes from the builder's constructor, not per episode
    assert rec.phases == ["reset", "reset", "run"]  # queued only during build/reset; switches to run after reset
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
    s, b, rec = _session(plan=F.Plan(raise_at=1, raise_exc=lambda: ValueError("bad action")))
    s.reset()
    with pytest.raises(ValueError, match="bad action"):
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
            raise ec.ResetBudgetExhausted("budget exhausted")
        claims.append(what)

    s, b, rec = _session(claim_reset=claim)
    with pytest.raises(ec.ResetBudgetExhausted):
        s.reset()
    assert claims == ["build"] and s.budget_exhausted is True and s.reset_calls == 1
    assert b.env.resets == 0  # when the budget is refused, reset never reaches the environment


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
    s.close()  # a second close does not raise


def test_own_builder_uses_dataset_and_max_steps():
    """Without an injected builder, a real builder is built from dataset and max_steps (identity resolution only, no
    scene); a missing max_steps is rejected."""
    ec = F.env_session()
    s = ec.EnvSession("PickXtimes", 0, max_steps=1300, dataset="hard-verify")
    assert s.builder.dataset == "hard-verify"
    # After the repo split EnvSession no longer has identity() (identity resolution moved to episode.make_spec);
    # check the self-built builder's resolution directly
    assert s.builder.resolve_identity(0)["tier"] == "xhard0"
    s9 = ec.EnvSession("PickXtimes", 0, max_steps=1600)  # defaults to ood (V9 unchanged)
    assert s9.builder.dataset == "ood"
    with pytest.raises(ValueError, match="max_steps"):
        _ = ec.EnvSession("PickXtimes", 0, dataset="hard-verify").builder
    with pytest.raises(ValueError):
        ec.EnvSession("PickXtimes", 0, max_steps=1300, dataset="test")
