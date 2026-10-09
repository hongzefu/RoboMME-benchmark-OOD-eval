"""L0：benchmark 子模块锁定与「评估仓不复制 benchmark 代码」（C18-SUBMODULE-LOCK）。

拆仓前的 ``test_upstream_bytes.py``、``test_copies_vs_upstream.py`` 守护的是 benchmark 仓的 ``src/robomme``（与官方
``1fadc0ec`` 逐字节相同）与 ``src/robomme_hard`` 的白名单副本；拆仓后这两份代码只在 benchmark 仓里，由 benchmark 仓自己的
同名测试与 ``BENCH_UPSTREAM`` 判定行守护。评估仓这一侧只需钉死两件事：

1. ``third_party/robomme_benchmark`` 是 ``.gitmodules`` 登记的子模块（URL 为 benchmark 仓），HEAD 树里的 gitlink 等于本文件
   记下的锁定值 ``BENCH_LOCK``（1.7 提交「benchmark 子模块升到候选版本 v1.0-ood-rc1（d22fe20）」）——升级子模块时必须同步
   改这里，不能悄悄漂移；子模块已检出时，检出的 HEAD 也必须等于 gitlink；
2. 评估仓里没有 ``src/robomme``、``src/robomme_hard`` 的副本（既不被 git 跟踪，也不在磁盘上），``robomme``／``robomme_hard``
   只能从子模块的 ``src/`` 解析（只用 ``find_spec`` 找位置，不执行包代码）。
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
#: benchmark 子模块的锁定提交（1.7 提交写入；升级子模块时与 gitlink 一起改）
BENCH_LOCK = "d22fe20dc7750e6db10601453206210270912e7d"


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=60).stdout


def gitlink(path: str = SUBMODULE) -> tuple[str, str]:
    """HEAD 树里 ``path`` 的 (mode, sha)。"""
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
    assert sha == BENCH_LOCK, f"子模块 gitlink={sha} 与记录的锁定值 {BENCH_LOCK} 不符（升级子模块须同步改 BENCH_LOCK）"


def test_checked_out_submodule_matches_gitlink():
    """子模块已检出（主检出）时检出的 HEAD 等于 gitlink；worktree 里子模块目录为空，只核 gitlink（上一条）。"""
    sub = REPO / SUBMODULE
    if not (sub / ".git").exists():
        pytest.skip("未验证：子模块未检出（worktree），只核 gitlink（test_gitlink_equals_recorded_lock）")
    assert _git("rev-parse", "HEAD", cwd=sub).strip() == gitlink()[1]


def test_no_benchmark_package_copies_in_eval_repo():
    for pkg in ("src/robomme", "src/robomme_hard"):
        assert _git("ls-files", "--", pkg).strip() == "", f"评估仓跟踪了 {pkg}"
        assert not (REPO / pkg).exists(), f"评估仓磁盘上有 {pkg}"


def test_benchmark_packages_resolve_to_submodule_src():
    for name in ("robomme", "robomme_hard"):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin, name
        origin = Path(spec.origin).resolve().as_posix()
        assert f"/{SUBMODULE}/src/{name}/" in origin, (name, origin)
        assert not origin.startswith((REPO / "src").resolve().as_posix() + "/"), (name, origin)
