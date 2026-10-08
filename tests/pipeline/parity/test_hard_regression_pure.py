"""C15.17 ``hard_regression.py`` 的纯函数：记录点路径的静态收集（``record_path_patterns``／``is_record_path``）、
规格逐叶比对（``compare_specs``）、规格根档序的输入校验、``--episodes`` 局号区间与按数据集名推每任务局数。
（reset-replay 判定行与 ``--out`` 校验随 ``reset-replay`` 子命令在拆仓时删除。）

期望由本文件手写的小输入推出：路径正则按 AST 规则手写，判定行字段按手写的目标与记录行手数。
"""
from __future__ import annotations

import json

import pytest

import parity_fixtures as F


@pytest.fixture(scope="module")
def hr():
    return F.hard_regression()


# ── 记录点路径 ──────────────────────────────────────────────────────────────


def _env_tree(root):
    (root / "utils").mkdir(parents=True)
    (root / "T.py").write_text(
        "class X:\n"
        "    def f(self, rec, i, p, name):\n"
        "        rec.record('a.b', 1)\n"
        "        rec.record(f'cells.{i}.x', 2)\n"
        "        rec.record(f'{p}.placed', 3)\n"      # 整条以变量开头：无法静态归类，不收
        "        rec.other('z.z', 1)\n"               # 不是 record 方法
        "        rec.record(name)\n"                  # 非常量、非 f-string：不收
        "        rec.record()\n",                     # 无参数：不收
        encoding="utf-8")
    (root / "utils" / "u.py").write_text("def g(r):\n    r.record('u.k', 0)\n", encoding="utf-8")
    (root / "Other.py").write_text("def h(r):\n    r.record('other.only', 0)\n", encoding="utf-8")  # 别的任务不混入
    return root


def test_record_path_patterns_static_collection(hr, tmp_path):
    root = _env_tree(tmp_path / "env")
    pats = hr.record_path_patterns("T", root=root)
    assert pats == {r"a\.b", r"cells\.[^.]+\.x", r"u\.k"}
    assert hr.record_path_patterns("T", root=root, method="other") == {r"z\.z"}
    assert hr.is_record_path("cells.3.x", pats) and hr.is_record_path("cells.3.x.deep", pats)  # 前缀即其下全部
    assert not hr.is_record_path("cells.3.y", pats) and not hr.is_record_path("a.bc", pats)
    assert not hr.is_record_path("cells.3.4.x", pats)  # 占位符只匹配一个路径段


def test_compare_specs_splits_record_within_tol_from_injected(hr):
    tol = hr._hard_specs().RECORDED_FLOAT_TOL
    pats = {r"a\.b", r"cells\.[^.]+\.x"}
    s4 = {"provenance": {"by": "s4"}, "identity": {"seed": 1}, "a": {"b": 1.0}, "cells": {"3": {"x": 0.5}}, "c": 1,
          "same": [1, 2]}
    hard = {"provenance": {"by": "hard"}, "identity": {"seed": 2}, "a": {"b": 1.0 + tol / 2},
            "cells": {"3": {"x": 0.5 + 10 * tol}}, "c": 2, "d": 0, "same": [1, 2]}
    r = hr.compare_specs(s4, hard, pats)
    # provenance／identity 不比；a.b 是记录点且差在容差内；cells.3.x 记录点但超差、c 非记录点、d 只在一侧 → 回注差
    assert (r["injected"], r["within"], r["paths"]) == (3, 1, ["c", "cells.3.x", "d"])
    assert r["max_abs"] == pytest.approx(tol / 2)
    # 同一路径另被登记为「值点」（value_patterns）时，即便在容差内也算回注差
    r = hr.compare_specs(s4, hard, pats, value_patterns={r"a\.b"})
    assert (r["injected"], r["within"]) == (4, 0) and "a.b" in r["paths"]
    assert hr.compare_specs(s4, dict(s4), pats) == {"injected": 0, "within": 0, "max_abs": 0.0, "paths": []}


# ── 输入校验 ────────────────────────────────────────────────────────────────


def test_specs_tiers_detects_v4_header(hr, tmp_path, capsys):
    hs = hr._hs_light()
    root = tmp_path / "root"
    root.mkdir()
    assert hr.specs_tiers(str(root)) == (hs.TIERS, False)  # 空根
    (root / hs.TIERS[0]).mkdir()
    (root / hs.TIERS[0] / "specs.jsonl").write_text("坏的首行\n", encoding="utf-8")
    assert hr.specs_tiers(str(root))[1] is False  # 首行读不出：跳过
    (root / hs.TIERS[1]).mkdir()
    (root / hs.TIERS[1] / "specs.jsonl").write_text(json.dumps({"schema": hs.SCHEMA}) + "\n", encoding="utf-8")
    assert hr.specs_tiers(str(root)) == (hs.TIERS, True)  # 只含部分档的局部根也判 /4
    capsys.readouterr()


def test_episodes_half_open_range(hr):
    """``--episodes a:b`` 是 builder 局号半开区间；不给即全部；越界、倒序、格式错一律报错（不静默截断）。"""
    assert hr.parse_episodes("0:2", 12) == [0, 1]
    assert hr.parse_episodes("11:12", 12) == [11]
    assert hr.parse_episodes(None, 3) == [0, 1, 2]
    for bad in ("2:2", "3:1", "-1:2", "0:13", "0", "a:b"):
        with pytest.raises(SystemExit):
            hr.parse_episodes(bad, 12)


def test_expected_episodes_by_dataset_name(hr):
    """按数据集名取局：hard-verify 只有 xhard0（XHARD0_PER_TASK），ood 只有新值档（格表在该任务的局数之和）。"""
    hs = type("HS", (), {"XHARD0_PER_TASK": 5, "EXPECTED_CELLS": {("A", "xhard1"): 3, ("A", "xhard2"): 4,
                                                                   ("B", "xhard1"): 9}})
    assert hr.expected_episodes("A", hs, "hard-verify") == 5
    assert hr.expected_episodes("A", hs, "ood") == 7
    assert hr.expected_episodes("A", hs, "ood", {("A", "xhard4"): 2}) == 2
    with pytest.raises(ValueError, match="未知数据集"):
        hr.expected_episodes("A", hs, "test")
    real = hr._hs_light()
    assert hr.expected_episodes(real.ALL_TASKS[0], real, "hard-verify") == real.XHARD0_PER_TASK
