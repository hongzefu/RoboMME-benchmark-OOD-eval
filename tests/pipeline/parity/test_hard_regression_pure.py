"""C15.17 ``hard_regression.py`` 的纯函数：记录点路径的静态收集（``record_path_patterns``／``is_record_path``）、
规格逐叶比对（``compare_specs``）、reset-replay 判定行（``replay_verdict``）、``--out`` 与规格根档序的输入校验。

期望由本文件手写的小输入推出：路径正则按 AST 规则手写，判定行字段按手写的目标与记录行手数。
"""
from __future__ import annotations

import json
import re

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


# ── reset-replay 判定行 ──────────────────────────────────────────────────────


def _targets():
    return [{"task": "A", "tier": "xhard1", "seed": 1, "spec_sha256": "s1"},
            {"task": "B", "tier": "xhard2", "seed": 2, "spec_sha256": "s2"}]


def _row(t, **binding):
    b = {"injected_mismatch": 0, "layout_drift": 0, "unused": 0, "mode": "replay", "spec_sha256": t["spec_sha256"]}
    b.update(binding)
    return dict(t, ok=True, binding=b)


TABLE = {("A", "xhard1"): 1, ("B", "xhard2"): 1}


def _verdict(hr, rows, *, version="v9", table=TABLE, limited=False, targets=None):
    ok, line = hr.replay_verdict(targets or _targets(), rows, version=version, table=table, limited=limited)
    return ok, dict(t.split("=", 1) for t in line.split()), line


def test_replay_verdict_pass(hr):
    ok, f, line = _verdict(hr, [_row(t) for t in _targets()])
    assert ok and line == ("V9_RESET_REPLAY=PASS shape=cells2 resets=2 replay=2 injected_mismatch=0 layout_drift=0 "
                           "spec_bound=2 unused=0 layout_hit_bad=0 errors=0")


@pytest.mark.parametrize("case,field,want", [
    ("error", "errors", "1"),
    ("injected", "injected_mismatch", "1"),
    ("drift", "layout_drift", "2"),
    ("unused", "unused", "1"),
    ("not_replay", "replay", "1"),
    ("spec_other", "spec_bound", "1"),
    ("hit_bad", "layout_hit_bad", "1"),
    ("missing_row", "resets", "1"),
])
def test_replay_verdict_each_failure(hr, case, field, want):
    t = _targets()
    rows = [_row(t[0]), _row(t[1])]
    if case == "error":
        rows[1] = dict(t[1], ok=False, error="RuntimeError: x")
    elif case == "injected":
        rows[0] = _row(t[0], injected_mismatch=1)
    elif case == "drift":
        rows[0] = _row(t[0], layout_drift=2)
    elif case == "unused":
        rows[1] = _row(t[1], unused=1)
    elif case == "not_replay":
        rows[1] = _row(t[1], mode="sample")
    elif case == "spec_other":
        rows[1] = _row(t[1], spec_sha256="其他规格")
    elif case == "hit_bad":
        rows[0] = _row(t[0], layered=True, layout_hit=["p1"], layout_paths_expected=["p1", "p2"])
    else:
        rows = rows[:1]
    ok, f, _ = _verdict(hr, rows)
    assert not ok and f["V9_RESET_REPLAY"] == "FAIL" and f[field] == want


def test_replay_verdict_cell_count_and_versions(hr):
    rows = [_row(t) for t in _targets()]
    big = {**TABLE, ("C", "xhard3"): 1}
    ok, _, _ = _verdict(hr, rows, table=big)
    assert not ok  # 未截断时目标数必须等于完整格表格数，缺格不得静默少测
    ok, _, _ = _verdict(hr, rows, table=big, limited=True)
    assert ok  # --limit 截断时不要求
    # 非 /4 口径不核 spec_bound，判定行名随版本
    loose = [dict(r, spec_sha256=None) for r in rows]
    ok, f, line = _verdict(hr, loose, version="v7", table=None)
    assert ok and line.startswith("V7_RESET_REPLAY=PASS ") and f["spec_bound"] == "0"
    ok, _, _ = _verdict(hr, loose, version="v9")
    assert not ok


# ── 输入校验 ────────────────────────────────────────────────────────────────


def test_replay_out_must_be_file(hr, tmp_path):
    with pytest.raises(SystemExit, match="必须是 jsonl 文件路径"):
        hr._replay_out_path(str(tmp_path))
    with pytest.raises(SystemExit, match="必须是 jsonl 文件路径"):
        hr._replay_out_path("artifacts/x/")
    assert hr._replay_out_path(str(tmp_path / "r.jsonl")) == tmp_path / "r.jsonl"
    assert hr.replay_key({"task": "A", "tier": "x", "seed": 1}) == ("A", "x", 1, None)  # 旧记录无 spec_sha256 永不命中


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
    with pytest.raises(SystemExit, match="规格根不是"):
        hr.specs_version(str(tmp_path / "empty-root"))
    capsys.readouterr()


def test_xhard0_switch_guard(hr):
    on = type("HS", (), {"XHARD0_IN_TEST_HARD": True, "XHARD0_PER_TASK": 12})
    off = type("HS", (), {"XHARD0_IN_TEST_HARD": False, "xhard0_prefix": staticmethod(lambda: 0)})
    legacy = type("HS", (), {"XHARD0_PER_TASK": 7})  # 旧 hard_specs 无开关：视为开、前置局数取 XHARD0_PER_TASK
    hr._require_xhard0_in_test_hard(on)
    hr._require_xhard0_in_test_hard(legacy)
    with pytest.raises(SystemExit, match="xhard0 已退出 ood"):
        hr._require_xhard0_in_test_hard(off)
    assert (hr._xhard0_prefix(on), hr._xhard0_prefix(off), hr._xhard0_prefix(legacy)) == (12, 0, 7)
    assert re.fullmatch(r"\d+", str(hr._xhard0_prefix(hr._hs_light())))
