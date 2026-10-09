"""S0 contract helper self-test: compliant fixtures pass, each counterexample is reported; the new trace_writer API
does not change old-format bytes by default."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests._support.loaders import load_script
from tests.pipeline.evalx.report import trace_contract as tc


def _tw():
    return load_script("eval-official/trace_writer.py")


def _img(k: int, cam: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), (k * 7 + cam * 3) % 251, dtype=np.uint8)


def _st(k: int) -> np.ndarray:
    return np.arange(8, dtype=np.float32) + np.float32(k)


def _write(root, *, route="smvla/new", status="fail", n=3, missing=(), no_frame=False, key="T_xhard0_7",
           attempt=1, action_dtype=np.float32, end_extra=None, arrays=True):
    t = _tw()
    ep = root / f"{key}.a{attempt}"
    ident = {"task": "T", "tier": "xhard0", "seed": 7, "dataset": "hard-verify", "source_episode": 3,
             "key": key, "attempt": attempt}
    # collect_arrays=False: this fixture decides per C4 whether to write arrays.npz (the stage 3 TraceWriter writes
    # it automatically by default)
    w = t.TraceWriter(ep / "trace.jsonl", route=route, identity=ident, max_steps=1300, collect_arrays=False)
    payload = {}
    if no_frame:
        w.log_demo([], [])
        w.close(status="error", terminal_reason="error", demo_frames=0, no_frame=True,
                steps_attempted=0, steps_observed=0, frames_recorded=0)
        return ep
    w.log_demo([_img(-1), _img(0)], [_img(-1, 1), _img(0, 1)], [_st(-1), _st(0)], ["goal"])
    for s in range(1, n + 1):
        a = (np.arange(8) * 0.5 + s).astype(action_dtype)
        payload[f"exec_action__{s - 1:05d}"] = a
        if s in missing:
            w.log_missing_step(step=s, action=a, reason="env_exception")
        else:
            w.log_step(step=s, front=_img(s), wrist=_img(s, 1), state=_st(s), action=a, subgoal=None,
                       terminated=False, truncated=False, status=None)
    obs = n - len(missing)
    extra = {"demo_frames": 1, "steps_attempted": n, "steps_observed": obs, "frames_recorded": 2 + obs}
    extra.update(end_extra or {})
    w.close(status=status, terminal_reason=status, **extra)
    if arrays and np.dtype(action_dtype) != np.dtype("<f4"):
        np.savez(ep / "arrays.npz", **payload)
    return ep


def test_compliant_episode_renders(tmp_path):
    ep = _write(tmp_path)
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": 3, "status": "fail"})


def test_float64_actions_need_arrays(tmp_path):
    ep = _write(tmp_path, action_dtype=np.float64)
    tc.assert_renderable(ep)
    ep2 = _write(tmp_path / "b", action_dtype=np.float64, arrays=False)
    assert any("C4" in p and "arrays.npz" in p for p in tc.contract_problems(ep2))


def test_no_frame_error_episode_is_contract_ok(tmp_path):
    ep = _write(tmp_path, no_frame=True)
    tc.assert_renderable(ep)  # no-frame episodes only check the contract


@pytest.mark.parametrize("kwargs,needle", [
    ({"route": "pp-new"}, "C1"),
    ({"key": "X_xhard0_1", "attempt": 2}, None),  # self-consistent, should pass
    ({"end_extra": {"frames_recorded": 99}}, "C8 frames_recorded"),
    ({"end_extra": {"steps_observed": 1}}, "C8 steps_observed"),
    ({"status": "env_done"}, "C3"),
    ({"end_extra": {"observer_hook_errors": -1}}, "C11"),
])
def test_contract_negatives(tmp_path, kwargs, needle):
    ep = _write(tmp_path, **kwargs)
    probs = tc.contract_problems(ep)
    if needle is None:
        assert probs == []
    else:
        assert any(needle in p for p in probs), probs


def test_dir_name_mismatch_and_counts_vs_result_row(tmp_path):
    ep = _write(tmp_path)
    moved = ep.with_name("T_xhard0_7.a2")
    ep.rename(moved)
    assert any("C6" in p for p in tc.contract_problems(moved))
    ep = _write(tmp_path / "c")
    with pytest.raises(AssertionError, match="exec_steps"):
        tc.assert_counts_consistent(ep, {"exec_steps": 4})


def test_missing_step_keeps_number_action_and_reason(tmp_path):
    ep = _write(tmp_path, missing=(2,), n=3)
    rows = _tw().read_trace(ep / "trace.jsonl")
    st2 = [r for r in rows if r["kind"] == "step"][1]
    assert st2["step"] == 2 and st2["observed"] is False and st2["missing_reason"] == "env_exception"
    assert st2["front_sha256"] is None and st2["action"] is not None
    assert st2["terminated"] == "NOT_OBSERVED" and st2["truncated"] == "NOT_OBSERVED"
    assert tc.contract_problems(ep) == []
    tc.assert_counts_consistent(ep, {"exec_steps": 3})


def test_default_log_step_bytes_unchanged(tmp_path):
    """Without the new parameters, step rows have the same key set and values as the old format (no new keys such as
    observed / missing_reason)."""
    t = _tw()
    w = t.TraceWriter(tmp_path / "trace.jsonl", route="r", identity={}, max_steps=5)
    w.log_step(step=1, front=_img(1), wrist=_img(1, 1), state=_st(1), action=_st(1), subgoal="x",
               terminated=1, truncated=0, status=None)
    w.close(status="fail")
    row = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[2])
    assert sorted(row) == sorted(["kind", "step", "front_sha256", "wrist_sha256", "state", "action", "subgoal",
                                  "terminated", "truncated", "status"])
    assert row["terminated"] is True and row["truncated"] is False
    assert repr(t.UNSET) == "UNSET" and not t.UNSET and t.UNSET is type(t.UNSET)()
