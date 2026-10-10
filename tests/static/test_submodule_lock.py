"""L0: benchmark submodule lock and "the eval repo does not copy benchmark code" (C18-SUBMODULE-LOCK).

Before the repo split, ``test_upstream_bytes.py`` and ``test_copies_vs_upstream.py`` guarded the benchmark repo's
``src/robomme`` (byte-identical to official ``1fadc0ec``) and the whitelisted copy in ``src/robomme_ood``; after the
split both live only in the benchmark repo and are guarded by that repo's own tests of the same name and its
``BENCH_UPSTREAM`` verdict line. The eval repo only needs to pin two things:

1. ``third_party/robomme_benchmark`` is a submodule registered in ``.gitmodules`` (URL pointing at the benchmark repo),
   and the gitlink in the HEAD tree equals the lock value ``BENCH_LOCK`` recorded in this file (commit 1.16, "bump the
   benchmark submodule to the locked release v1.0-ood (a5efb99)"). Upgrading the submodule must update it here too;
   it must never drift silently. When the submodule is checked out, its HEAD must also equal the gitlink.
2. The eval repo holds no copy of ``src/robomme`` or ``src/robomme_ood`` (neither tracked by git nor on disk);
   ``robomme`` / ``robomme_ood`` may only resolve from the submodule's ``src/`` (located via ``find_spec`` only,
   without executing package code).
"""
from __future__ import annotations

import configparser
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from tests._support.loaders import REPO

SUBMODULE = "third_party/robomme_benchmark"
BENCH_URL = "https://github.com/hongzefu/RoboMME-benchmark-OOD.git"
#: Locked commit of the benchmark submodule (written in commit 1.7; change together with the gitlink when upgrading)
BENCH_LOCK = "51e05feb05c61b9a5a25cef3ba189ad625f2cd83"


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=60).stdout


def gitlink(path: str = SUBMODULE) -> tuple[str, str]:
    """(mode, sha) of ``path`` in the HEAD tree."""
    line = _git("ls-tree", "HEAD", "--", path).strip()
    mode, kind, sha = line.split("\t")[0].split()
    assert kind == "commit", line
    return mode, sha


def test_gitmodules_declares_benchmark_submodule():
    cp = configparser.ConfigParser()
    cp.read_string((REPO / ".gitmodules").read_text(encoding="utf-8"))
    sec = f'submodule "{SUBMODULE}"'
    assert cp.has_section(sec), cp.sections()
    assert cp.get(sec, "path") == SUBMODULE and cp.get(sec, "url") == BENCH_URL


def test_gitlink_equals_recorded_lock():
    mode, sha = gitlink()
    assert mode == "160000" and re.fullmatch(r"[0-9a-f]{40}", sha), (mode, sha)
    assert sha == BENCH_LOCK, f"submodule gitlink={sha} does not match the recorded lock {BENCH_LOCK} (update BENCH_LOCK when upgrading the submodule)"


def test_checked_out_submodule_matches_gitlink():
    """When the submodule is checked out (main checkout) its HEAD equals the gitlink; in a worktree the submodule directory is empty and only the gitlink is checked (previous test)."""
    sub = REPO / SUBMODULE
    if not (sub / ".git").exists():
        pytest.skip("Not verified: submodule not checked out (worktree); only the gitlink is checked (test_gitlink_equals_recorded_lock)")
    assert _git("rev-parse", "HEAD", cwd=sub).strip() == gitlink()[1]


def test_no_benchmark_package_copies_in_eval_repo():
    for pkg in ("src/robomme", "src/robomme_ood"):
        assert _git("ls-files", "--", pkg).strip() == "", f"eval repo tracks {pkg}"
        assert not (REPO / pkg).exists(), f"eval repo has {pkg} on disk"


def test_benchmark_packages_resolve_to_submodule_src():
    for name in ("robomme", "robomme_ood"):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin, name
        origin = Path(spec.origin).resolve().as_posix()
        assert f"/{SUBMODULE}/src/{name}/" in origin, (name, origin)
        assert not origin.startswith((REPO / "src").resolve().as_posix() + "/"), (name, origin)
