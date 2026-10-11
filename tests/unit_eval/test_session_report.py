"""``EnvSession`` (budget ledger, strict truncation, hooks, read-only attributes, idempotent close) and ``report.summarize``."""
from __future__ import annotations

import json

import numpy as np
import pytest

from robomme_ood_eval import report
from robomme_ood_eval.session import EnvSession, ResetBudgetExhausted, StepCapReached
from tests.unit_eval import fakes


def make_session(**kw):
    b = fakes.FakeBuilder("VideoUnmask", "ood", 1800, finish_at=kw.pop("finish_at", None))
    return EnvSession("VideoUnmask", 0, dataset="ood", builder=b, **kw), b


def test_dataset_must_be_known():
    with pytest.raises(ValueError):
        EnvSession("T", 0, dataset="test")


@pytest.mark.parametrize("observed", [False, True, np.array([False]), np.array([True]), None, "false", 2])
def test_real_info_success_is_observed_and_serialized_without_changing_output(tmp_path, observed):
    import json
    class Recorder(fakes.FakeRecorder):
        def __init__(self):
            super().__init__(None, {})
            self.events = []
        def add_event(self, event):
            self.events.append(event)
    class Env(fakes.FakeEnv):
        def reset(self):
            obs, info = super().reset()
            info["success"] = observed
            self.returned_info = info
            return obs, info
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info["success"] = observed
            self.returned_info = info
            return obs, reward, terminated, truncated, info
    rec = Recorder()
    s, builder = make_session(recorder=rec)
    env = Env(0, finish_at=None)
    builder.make_env_for_episode = lambda _: env
    assert s.task_success is None
    _obs, info = s.reset()
    assert info is env.returned_info and s.info is info
    expected = int(observed.item() if isinstance(observed, np.ndarray) else observed) if \
               isinstance(observed, (bool, np.ndarray)) else None
    assert s.task_success == expected
    returned = s.step(np.zeros(8))
    assert returned[4] is env.returned_info and s.task_success == expected
    path = tmp_path / "events.json"
    path.write_text(json.dumps(rec.events))
    events = json.loads(path.read_text())
    assert [e["task_success"] for e in events if e["kind"] in ("env_reset", "env_step")] == [expected, expected]
    with pytest.raises(AttributeError):
        s.task_success = 0
    s.close()


def test_build_reset_step_counts_and_readonly_props():
    rec = fakes.FakeRecorder(None, {})
    progress, first = [], []
    s, b = make_session(recorder=rec, progress_cb=progress.append, progress_every=2,
                        first_step_cb=lambda: first.append(1))
    assert s.recorder is rec and s.env is None
    s.build()
    s.build()  # built only once
    obs, info = s.reset()
    assert s.task_goal == fakes.GOAL and s.info is info and s.reset_calls == 2
    for _ in range(4):
        s.step(np.zeros(8, np.float32))
    assert s.steps == 4 and first == [1] and progress == [2, 4] and s.info["status"] == "ongoing"
    with pytest.raises(AttributeError):
        s.steps = 0  # type: ignore[misc]
    s.close()
    s.close()  # idempotent
    assert b.envs[0].closed == 1 and s.env is None and s.timing["step_n"] == 4


def test_strict_cap_blocks_next_step():
    s, b = make_session(step_cap=3)
    s.reset()
    for _ in range(3):
        s.step(np.zeros(8))
    with pytest.raises(StepCapReached):
        s.step(np.zeros(8))
    assert s.cap_hit and s.steps == 3 and b.envs[0].n == 3


def test_ledger_claim_before_build_and_reset():
    class Ledger:
        def __init__(self):
            self.whats = []

        def claim(self, what):
            if len(self.whats) >= 1:
                raise ResetBudgetExhausted("over budget")
            self.whats.append(what)

    led = Ledger()
    s, _ = make_session(ledger=led)
    s.build()
    with pytest.raises(ResetBudgetExhausted):
        s.reset()
    assert led.whats == ["build"] and s.reset_calls == 1 and s.budget_exhausted


def test_external_exception_with_budget_attr_marks_exhausted():
    class Exhausted(RuntimeError):
        budget_exhausted = True

    def claim(what):
        raise Exhausted("shared ledger over budget")

    s, _ = make_session(claim_reset=claim)
    with pytest.raises(Exhausted):
        s.build()
    assert s.budget_exhausted and s.reset_calls == 0


def test_report_summarize(tmp_path):
    rows = [
        {"key": "A_x_1", "task": "VideoUnmask", "status": "success", "infra": False, "model": "dummy",
         "policy_label": "dummy", "dataset": "ood", "policy_seed": 0},
        {"key": "A_x_2", "task": "VideoUnmask", "status": "fail", "infra": False},
        {"key": "A_x_3", "task": "VideoUnmask", "status": "fail", "infra": True},  # infrastructure episodes are excluded from the denominator
        {"key": "B_x_1", "task": "BinFill", "status": "timeout", "infra": False},
        {"key": "B_x_1", "task": "BinFill", "status": "success", "infra": False},  # for a repeated key the last row wins
        {"key": "C_x_1", "task": "StopCube", "status": "timeout", "infra": False},
    ]
    p = tmp_path / "results.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"half', encoding="utf-8")
    log = report.summarize(tmp_path)
    assert log["schema"] == "robomme-ood-eval-log/2" and log["speed"] is None
    assert json.loads((tmp_path / "log.json").read_text()) == log
    assert log["tasks"]["VideoUnmask"] == {"episodes": 3, "counted": 2, "success": 1, "fail": 1, "timeout": 0,
                                           "infra": 1, "success_rate": 0.5}
    assert log["tasks"]["BinFill"]["success_rate"] == 1.0 and log["tasks"]["StopCube"]["success_rate"] == 0.0
    assert log["total_success_rate"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert list(log["tasks"]) == ["BinFill", "StopCube", "VideoUnmask"]  # official task order
    assert log["episodes"] == 5 and log["infra"] == 1 and log["success"] == 2 and len(log["tasks_missing"]) == 13
    assert log["model"] == "dummy" and log["dataset"] == "ood"


def test_report_empty(tmp_path):
    log = report.summarize(tmp_path / "results.jsonl")
    assert log["total_success_rate"] is None and log["episodes"] == 0 and log["counted"] == 0
