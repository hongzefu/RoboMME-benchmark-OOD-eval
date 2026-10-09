"""C13: the error branch of the two official evaluation entries (``scripts/evaluation.py`` and
``scripts/evaluation_ood.py`` in the benchmark submodule), actually executed in a test subprocess. After the repo split
both entries live in the submodule (the evaluation repo's ``scripts/`` only keeps ``evaluate.py``); when the submodule
is not checked out (git worktree), the same files are located via the installed ``robomme`` package.

The entry scripts run verbatim as ``__main__`` (``entry_bootstrap.py`` only swaps the builder for a CPU stub environment
in the subprocess's memory).

Registered semantic issue (current behavior locked, contract marked conditional): the error branch never assigns
``outcome``:
- error on the first episode: saving the video reads the undefined ``outcome`` and the process exits with NameError
  (env already closed, video not saved);
- previous episode reached a terminal status, this one errors: the video file name reuses the previous ``outcome``.
Error episodes always count as failures and are included in the success-rate denominator.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import eval_fakes as F

ENTRIES = ("evaluation.py", "evaluation_ood.py")
BOOT = Path(__file__).resolve().parent / "entry_bootstrap.py"


def _entry_path(entry: str) -> Path:
    import importlib.util

    own = F.REPO / "third_party" / "robomme_benchmark" / "scripts" / entry
    if own.is_file():
        return own
    spec = importlib.util.find_spec("robomme")
    return Path(list(spec.submodule_search_locations)[0]).parents[1] / "scripts" / entry


def _run(tmp_path, entry, scen):
    sp = tmp_path / "scen.json"
    sp.write_text(json.dumps(scen), encoding="utf-8")
    log = tmp_path / "events.jsonl"
    p = subprocess.run([sys.executable, str(BOOT), str(_entry_path(entry)), str(sp), str(log)],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120)
    return p, F.read_jsonl(log)


@pytest.mark.parametrize("entry", ENTRIES)
def test_error_on_first_episode_crashes_after_close(tmp_path, entry):
    p, ev = _run(tmp_path, entry, {"tasks": ["T"], "episodes": 1, "plans": {"T/0": ["ongoing", "error"]}})
    assert p.returncode != 0
    assert "NameError" in p.stderr and "outcome" in p.stderr
    kinds = [e["kind"] for e in ev]
    assert kinds == ["builder", "reset", "step", "step", "close"]  # env is closed before the crash
    assert "save" not in kinds


@pytest.mark.parametrize("entry", ENTRIES)
def test_error_after_terminal_reuses_stale_outcome_and_counts_failure(tmp_path, entry):
    scen = {"tasks": ["T"], "episodes": 3,
            "plans": {"T/0": ["ongoing", "success"], "T/1": ["error"], "T/2": ["fail"]}}
    p, ev = _run(tmp_path, entry, scen)
    assert p.returncode == 0, p.stderr
    saves = [e for e in ev if e["kind"] == "save"]
    assert len(saves) == 3
    assert "T_ep_0_success_" in saves[0]["path"]
    assert "T_ep_1_success_" in saves[1]["path"]  # the error episode reuses the previous outcome (current behavior locked)
    assert "T_ep_2_fail_" in saves[2]["path"]
    assert [e["frames"] for e in saves] == [3, 2, 2]  # 2 reset frames + non-terminal steps before the terminal one
    assert [e["ep"] for e in ev if e["kind"] == "close"] == [0, 1, 2]
    assert "Success rate: 0.3333333333333333" in p.stdout  # 1 success / 3 episodes (error counts as failure)
    assert all(e["action_shape"] == [8] for e in ev if e["kind"] == "step")
