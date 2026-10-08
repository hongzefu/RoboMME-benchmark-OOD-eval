"""C13 两个官方评估入口（benchmark 子模块的 ``scripts/evaluation.py`` 与 ``scripts/evaluation_ood.py``）的 error 分支，
以测试子进程实际执行。拆仓后两个入口都在子模块里（评估仓 ``scripts/`` 只留 ``evaluate.py``）；子模块未检出（git
worktree）时按已安装 ``robomme`` 包的位置找同一份文件。

入口脚本一字不改地以 ``__main__`` 运行（``entry_bootstrap.py`` 只在子进程内存里把 builder 换成 CPU 替身环境）。

已登记的语义问题（锁定现状，契约记 conditional）：error 分支不给 ``outcome`` 赋值——
- 第一局就 error：保存录像时读未定义的 ``outcome``，进程以 NameError 退出（env 已 close，录像未保存）；
- 前一局有终态、本局 error：录像文件名沿用上一局的 ``outcome``。
error 局一律记失败、计入成功率分母。
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
    assert kinds == ["builder", "reset", "step", "step", "close"]  # env 在崩溃前已 close
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
    assert "T_ep_1_success_" in saves[1]["path"]  # error 局沿用上一局 outcome（锁定现状）
    assert "T_ep_2_fail_" in saves[2]["path"]
    assert [e["frames"] for e in saves] == [3, 2, 2]  # reset 2 帧 + 终态前的非终态步
    assert [e["ep"] for e in ev if e["kind"] == "close"] == [0, 1, 2]
    assert "Success rate: 0.3333333333333333" in p.stdout  # 1 成功 / 3 局（error 计失败）
    assert all(e["action_shape"] == [8] for e in ev if e["kind"] == "step")
