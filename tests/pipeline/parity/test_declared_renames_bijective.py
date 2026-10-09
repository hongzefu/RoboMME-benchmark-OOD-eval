"""C15 ``hard_regression.XHARD0_DECLARED_RENAMES`` 的加载期校验：每个分区的改名映射必须是置换（双射、值集合 = 键集合）。

非置换映射会让「按映射改名后逐键相等」失去意义：两个名改成同一个名会把两个实体并成一个（后写的覆盖先写的），
只改一边会凭空造出对侧没有的名字。这里两层证明：
- 直接调真实校验函数 ``_check_renames_bijective``：现行映射通过；三类非置换各自被拒；
- 隔离副本（源码文本替换后写到 tmp 另执行，不改主检出、不登记 sys.modules）：把现行映射换成非置换，
  模块加载本身就抛 ValueError——证明校验确实接在模块加载时，而不只是一个没人调的函数。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import parity_fixtures as F
from tests._support.dev_loaders import script_path

#: 现行映射在源码里的写法（隔离副本据此替换；形态变了先报「需同步」）
DECLARED = '{"button_left": "button_right", "button_right": "button_left"}'


@pytest.fixture(scope="module")
def R():
    return F.hard_regression()


def test_current_declared_renames_are_permutations(R):
    R._check_renames_bijective(R.XHARD0_DECLARED_RENAMES)
    for sections in R.XHARD0_DECLARED_RENAMES.values():
        for mapping in sections.values():
            assert sorted(mapping) == sorted(mapping.values())


@pytest.mark.parametrize("mapping", [
    {"a": "c", "b": "c"},          # 两个名并成一个：非单射
    {"a": "b"},                    # 只改一边：值集合 ≠ 键集合
    {"a": "b", "b": "x"},          # 单射但改出键集合之外的名
])
def test_non_permutation_rejected(R, mapping):
    with pytest.raises(ValueError, match="不是置换"):
        R._check_renames_bijective({"T": {"articulations": mapping}})


def test_identity_and_cycle_accepted(R):
    R._check_renames_bijective({"T": {"actors": {"a": "a"}, "articulations": {"a": "b", "b": "c", "c": "a"}}})


def test_module_load_rejects_non_permutation(tmp_path, monkeypatch):
    src = script_path("parity/hard_regression.py").read_text(encoding="utf-8")
    assert src.count(DECLARED) == 1, "现行改名映射的源码形态变了，本用例需同步"
    # 隔离副本放在 tmp/dev-scripts/parity/ 下，同目录带一份 _common.py（模块顶层按同目录模块名 import _common）；
    # sys.path 的改动用例结束由 monkeypatch 还原
    copy = tmp_path / "dev-scripts" / "parity" / "hard_regression_bad.py"
    copy.parent.mkdir(parents=True)
    (copy.parent / "_common.py").write_text(script_path("parity/_common.py").read_text(encoding="utf-8"),
                                            encoding="utf-8")
    copy.write_text(src.replace(DECLARED, '{"button_left": "button_right", "button_right": "button_right"}'),
                    encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(copy.parent), *sys.path])
    spec = importlib.util.spec_from_file_location("_t6_hard_regression_bad_renames", copy)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ValueError, match="不是置换"):
        spec.loader.exec_module(module)
    # 对照：同一流程只是不改映射时照常加载
    good = tmp_path / "dev-scripts" / "parity" / "hard_regression_good.py"
    good.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_t6_hard_regression_good_renames", good)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
