"""Test helpers for the stage 2 shared contract C1-C11 (1005-eval-video-phase2-all-models-rerun-plan.md part two,
section 0, S0).

Tests for each route subtask (S1, S3, S4, S5, S7) call these on the episode directories they write:

- ``assert_renderable(ep_dir)``: checks ``trace.jsonl`` (and ``arrays.npz``) rule by rule against the contract; for
  episodes with frames, it also has the rerender tool ``robomme_ood_eval.record.official_render.load_trace`` read the
  record once to confirm the official-layout video can be drawn from it. No-frame ``error`` episodes
  (``end.no_frame=true``) only get the contract check; the rerenderer is not called.
- ``assert_counts_consistent(ep_dir, result_row)``: reconciles the C8 three-way counts with the result row's
  ``exec_steps``.

The contract text is the module docstring of ``src/robomme_ood_eval/record/trace_writer.py`` in the eval package; the
checks here rely only on that text and do not read constants of the route under test.
Production modules are always loaded by path via ``tests._support.loaders.load_script``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tests._support.loaders import load_script

TERMINALS = ("success", "fail", "timeout", "error")
NEW_ROUTES = re.compile(r"^(groundsg/[A-Za-z0-9_.-]+|pp|astra|smvla|perceptual-framesamp-modul)/(new|orig)$")
EP_DIR = re.compile(r"^(?P<key>.+)\.a(?P<attempt>\d+)$")
NOT_OBSERVED = "NOT_OBSERVED"


def _tw():
    return load_script("eval-official/trace_writer.py")


def _render():
    from robomme_ood_eval.record import official_render

    return official_render


def _fail(ep_dir: Path, msg: str) -> None:
    raise AssertionError(f"{ep_dir}: {msg}")


def contract_problems(ep_dir: str | Path) -> list[str]:
    """Return the list of contract problems (empty means compliant); the pure-check part of ``assert_renderable``,
    so counterexample tests can assert on each item."""
    ep_dir = Path(ep_dir)
    tpath = ep_dir / "trace.jsonl"
    if not tpath.is_file():
        return ["missing trace.jsonl (C5)"]
    rows = _tw().read_trace(tpath)
    problems = list(_tw().validate_trace(rows))
    if problems:
        return problems
    header, end = rows[0], rows[-1]
    demo = next(r for r in rows if r["kind"] == "demo")
    steps = [r for r in rows if r["kind"] == "step"]

    route = header.get("route")
    if not isinstance(route, str) or not NEW_ROUTES.match(route):
        problems.append(f"C1 invalid route: {route!r}")

    ident = header.get("identity") or {}
    for k in ("task", "tier", "seed", "dataset", "key", "attempt"):
        if ident.get(k) is None:
            problems.append(f"C6 identity missing {k}")
    if ident.get("source_episode") is None and ident.get("builder_episode") is None:
        problems.append("C6 identity missing source_episode / builder_episode")
    m = EP_DIR.match(ep_dir.name)
    if m and ident.get("key") is not None:
        if str(ident["key"]) != m["key"]:
            problems.append(f"C6 identity.key={ident['key']!r} does not match directory name {ep_dir.name!r}")
        if ident.get("attempt") is not None and int(ident["attempt"]) != int(m["attempt"]):
            problems.append(f"C6 identity.attempt={ident['attempt']} does not match directory name {ep_dir.name!r}")

    status, reason = end.get("status"), end.get("terminal_reason")
    if status not in TERMINALS or reason not in TERMINALS:
        problems.append(f"C3 invalid terminal state: status={status!r} terminal_reason={reason!r}")
    elif status != reason and (reason, status) != ("error", "timeout"):
        problems.append(f"C3 conflicting terminal state: status={status!r} terminal_reason={reason!r}")
    no_frame = bool(end.get("no_frame"))
    if no_frame and status != "error":
        problems.append("C3 only error episodes may set no_frame")

    demo_frames = end.get("demo_frames")
    if type(demo_frames) is not int or demo_frames < 0:
        problems.append(f"C2 end.demo_frames missing or invalid: {demo_frames!r}")
    elif not no_frame:
        if demo.get("frames") != demo_frames + 1:
            problems.append(f"C2 demo.frames={demo.get('frames')} != end.demo_frames+1={demo_frames + 1}")
        if len(demo.get("states") or []) != demo.get("frames"):
            problems.append("C2 demo state count differs from frame count")
        texts = demo.get("texts")
        if not isinstance(texts, list) or len(texts) != 1 or not str(texts[0]).strip():
            problems.append("C2 demo.texts must be a single non-empty task goal")

    for st in steps:
        for k in ("terminated", "truncated"):
            if not isinstance(st.get(k), bool) and st.get(k) != NOT_OBSERVED:
                problems.append(f"C9 step {st['step']} {k} must be a bool or NOT_OBSERVED")
        if not isinstance(st.get("subgoal"), (str, type(None))):
            problems.append(f"C7 step {st['step']} subgoal must be text or None")
        if st.get("action") is None:
            problems.append(f"C4 step {st['step']} missing action")
        if st.get("observed") is False:
            if not st.get("missing_reason"):
                problems.append(f"C8 step {st['step']} has no observation but no reason")
        elif not st.get("front_sha256") or not st.get("wrist_sha256") or st.get("state") is None:
            problems.append(f"C4 step {st['step']} missing frame or state "
                            "(steps without observation need observed=false)")

    observed = sum(1 for st in steps if st.get("observed") is not False)
    for k in ("steps_attempted", "steps_observed", "frames_recorded"):
        if type(end.get(k)) is not int:
            problems.append(f"C8 end missing {k}")
    if not any(p.startswith("C8 end missing") for p in problems):
        omitted = int(end.get("omitted_timeout_frames", 0) or 0)
        if end["steps_attempted"] != len(steps) or end["steps_attempted"] != end.get("exec_steps"):
            problems.append(f"C8 steps_attempted={end['steps_attempted']} does not match step rows {len(steps)} "
                            "/ exec_steps")
        if end["steps_observed"] != observed:
            problems.append(f"C8 steps_observed={end['steps_observed']} does not match observed steps {observed}")
        if not no_frame and type(demo_frames) is int:
            expect = demo_frames + 1 + end["steps_observed"] - omitted
            if end["frames_recorded"] != expect:
                problems.append(f"C8 frames_recorded={end['frames_recorded']} != {expect}")
        if no_frame and end["frames_recorded"] != 0:
            problems.append("C8 frames_recorded must be 0 for a no_frame episode")

    if "observer_hook_errors" in end and (type(end["observer_hook_errors"]) is not int or end["observer_hook_errors"] < 0):
        problems.append("C11 observer_hook_errors must be a non-negative integer")

    arrays_path = ep_dir / "arrays.npz"
    non_f32 = [st for st in steps if st.get("action") and np.dtype(st["action"]["dtype"]) != np.dtype("<f4")]
    if non_f32 and not arrays_path.is_file():
        problems.append("C4 non-float32 actions present but arrays.npz is missing")
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as arr:
            keys = set(arr.files)
            for st in steps:
                key = f"exec_action__{st['step'] - 1:05d}"
                if key not in keys:
                    problems.append(f"C4 arrays.npz missing {key}")
                    continue
                a = arr[key]
                rec = st["action"]
                if a.dtype.str != rec["dtype"] or list(a.shape) != rec["shape"]:
                    problems.append(f"C4 {key} dtype/shape does not match trace")
    return problems


def assert_renderable(ep_dir: str | Path) -> None:
    """Contract check + for episodes with frames, have the rerender tool ``load_trace`` read the record once."""
    ep_dir = Path(ep_dir)
    problems = contract_problems(ep_dir)
    if problems:
        _fail(ep_dir, "; ".join(problems))
    end = _tw().read_trace(ep_dir / "trace.jsonl")[-1]
    if end.get("no_frame"):
        return
    try:
        _render().load_trace(ep_dir / "trace.jsonl")
    except Exception as e:  # rejected by the rerenderer means not renderable
        _fail(ep_dir, f"rerender tool load_trace rejected it: {type(e).__name__}: {e}")


def assert_counts_consistent(ep_dir: str | Path, result_row: dict) -> None:
    """C8: ``end`` three-way counts are self-consistent, ``steps_attempted`` equals the result row's ``exec_steps``,
    and the terminal state matches the result row."""
    ep_dir = Path(ep_dir)
    rows = _tw().read_trace(ep_dir / "trace.jsonl")
    end = rows[-1]
    problems = [p for p in contract_problems(ep_dir) if p.startswith("C8")]
    if problems:
        _fail(ep_dir, "; ".join(problems))
    if "exec_steps" in result_row and end["steps_attempted"] != int(result_row["exec_steps"]):
        _fail(ep_dir, f"steps_attempted={end['steps_attempted']} != result row exec_steps={result_row['exec_steps']}")
    if "status" in result_row and result_row["status"] != end["status"]:
        _fail(ep_dir, f"result row status={result_row['status']!r} does not match trace end.status={end['status']!r}")
