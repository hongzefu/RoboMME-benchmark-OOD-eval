"""C13-SG-TRACE: row order, fields and hashing conventions of each episode's ``trace.jsonl`` (plan part two, 1.7).
Expected values are hand-written."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import sgx_report_fixtures as F


def test_row_order_and_fields(tmp_path):
    t = F.tw()
    p = tmp_path / "ep" / "trace.jsonl"
    ident = {"task": "VideoUnmask", "source_episode": 3, "seed": 11, "attempt": 1}
    with t.TraceWriter(p, route="new", identity=ident, max_steps=1300) as w:
        w.log_demo([F.frame(0)], [F.frame(0, 1)], [F.state(0)], ["watch the demo"])
        w.log_request("reset", b"abc", step=0)
        w.log_response(np.zeros((2, 8), np.float32), step=0)
        w.log_history(1, 2, note="buf")
        w.log_step(step=1, front=F.frame(1), wrist=F.frame(1, 1), state=F.state(1), action=F.action(1),
                   subgoal="pick up", terminated=False, truncated=False, status="ongoing")
        w.log_step(step=2, front=F.frame(2), wrist=None, state=F.state(2), action=F.action(2),
                   subgoal="pick up", terminated=True, truncated=False, status="success")
        w.close(status="success", terminal_reason="env_terminated", note="x")
    rows = t.read_trace(p)
    assert [r["kind"] for r in rows] == ["header", "demo", "request", "response", "history", "step", "step", "end"]
    h = rows[0]
    assert h["schema"] == "sgeval-trace/1" and h["identity"] == ident and h["max_steps"] == 1300 and h["route"] == "new"
    assert rows[1]["frames"] == 1 and rows[1]["texts"] == ["watch the demo"]
    assert rows[2] == {"kind": "request", "name": "reset", "step": 0, "sha256": hashlib.sha256(b"abc").hexdigest(),
                       "nbytes": 3}
    assert rows[3]["actions"]["dtype"] == "<f4" and rows[3]["actions"]["shape"] == [2, 8]
    assert rows[6]["wrist_sha256"] is None and rows[6]["terminated"] is True
    # Stage 3 (freeze note 4.3): collect_arrays is on by default, so the end row gains an arrays summary; all other
    # fields are unchanged
    assert {k: v for k, v in rows[-1].items() if k != "arrays"} == {
        "kind": "end", "status": "success", "exec_steps": 2, "terminal_reason": "env_terminated", "note": "x"}
    assert rows[-1]["arrays"] == {"path": "arrays.npz", "action_keys": 2, "state_keys": 2, "missing_state_steps": []}
    assert t.validate_trace(rows) == []
    assert t.subgoal_sequence(rows) == ["pick up"]


def test_array_hash_keeps_dtype_and_shape():
    t = F.tw()
    a64 = np.arange(8, dtype=np.float64)
    a32 = a64.astype(np.float32)
    r64, r32 = t.array_record(a64), t.array_record(a32)
    # Raw dtype bytes are hashed directly without casting to float32: same values in different dtypes hash differently
    assert r64["sha256"] == hashlib.sha256(a64.tobytes()).hexdigest()
    assert r64["sha256"] != r32["sha256"] and r64["dtype"] == "<f8" and r32["dtype"] == "<f4"
    assert r64["f32hex"] == r32["f32hex"]  # human-readable field is equal and not used for verdicts
    assert t.array_record(a32.reshape(2, 4))["shape"] == [2, 4]
    assert t.image_sha256(None) is None
    img = F.frame(1)
    assert t.image_sha256(img) != t.image_sha256(img.reshape(16, 3))  # shape is part of the frame hash
    assert t.array_record(None) is None


def test_flip_one_bit_changes_hash():
    t = F.tw()
    a = F.action(3)
    assert t.array_record(a)["sha256"] != t.array_record(F.flip_bit(a))["sha256"]


def test_canonical_bytes_deterministic_and_dtype_sensitive():
    t = F.tw()
    x = {"b": np.ones(3, np.float32), "a": [1, 2.5, "s", None, True], "c": b"\x00"}
    y = {"c": b"\x00", "a": [1, 2.5, "s", None, True], "b": np.ones(3, np.float32)}
    assert t.canonical_bytes(x) == t.canonical_bytes(y)
    assert t.canonical_bytes(x) != t.canonical_bytes({**x, "b": np.ones(3, np.float64)})
    # Adjacent floats (differing only in the last bit) must also be distinguished
    assert t.canonical_bytes({"v": 0.1}) != t.canonical_bytes({"v": float(np.nextafter(0.1, 1.0))})


def test_write_after_close_raises_and_exit_closes_with_error(tmp_path):
    t = F.tw()
    w = t.TraceWriter(tmp_path / "a" / "trace.jsonl", route="r", identity={}, max_steps=5)
    w.close(status="fail")
    with pytest.raises(RuntimeError):
        w.log_request("infer", b"", step=0)
    p = tmp_path / "b" / "trace.jsonl"
    with pytest.raises(ValueError):
        with t.TraceWriter(p, route="r", identity={}, max_steps=5):
            raise ValueError("boom")
    rows = t.read_trace(p)
    assert rows[-1]["status"] == "error" and rows[-1]["terminal_reason"] == "exception:ValueError"
    assert [r["kind"] for r in rows] == ["header", "demo", "end"]
    assert t.validate_trace(rows) == []


def test_validate_trace_reports_structure_problems():
    t = F.tw()
    good = [{"kind": "header", "schema": "sgeval-trace/1"}, {"kind": "demo"},
            {"kind": "step", "step": 1}, {"kind": "step", "step": 2}, {"kind": "end", "exec_steps": 2}]
    assert t.validate_trace(good) == []
    gap = [*good[:2], {"kind": "step", "step": 1}, {"kind": "step", "step": 3}, {"kind": "end", "exec_steps": 3}]
    assert "steps are not contiguous from 1" in t.validate_trace(gap)
    noend = good[:-1]
    assert "end is not the unique last line" in t.validate_trace(noend)
    bad_exec = [*good[:-1], {"kind": "end", "exec_steps": 5}]
    assert "end.exec_steps does not match the last step" in t.validate_trace(bad_exec)
    assert t.validate_trace([]) == ["empty trace"]
    assert "demo is not before the first step" in t.validate_trace([good[0], good[2], good[1], good[3], good[4]])
    # The GroundSG official loop resets the policy before taking the initial observation: request before demo is
    # compliant (12.454)
    assert t.validate_trace([good[0], {"kind": "request", "step": 0}, good[1], *good[2:]]) == []
    assert json.dumps(good)  # the fixture itself is serializable
