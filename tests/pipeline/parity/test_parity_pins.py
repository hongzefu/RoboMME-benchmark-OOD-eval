"""C15 闸门口径钉值：``noise_gate`` 的常量与钉值文件 ``parity_pins.py`` 逐个相等；canonical sha 固定向量。

其余对拍用例为了不写字面值（R8）从模块读这些常量、只测判定逻辑；本文件负责抓「改口径」本身。
"""
from __future__ import annotations

import copy

import parity_fixtures as F
import parity_pins as PIN


def test_gen_regress闸门常量与钉值相等():
    ng = F.noise_gate()
    assert ng.NOISE_MAX == PIN.NOISE_MAX
    assert ng.FLIP_RERUN_MAX == PIN.FLIP_RERUN_MAX
    assert ng.RERUN_MIN == PIN.RERUN_MIN
    assert ng.PRECOND_WORKERS == PIN.PRECOND_WORKERS
    assert ng.PRECOND_GPU == PIN.PRECOND_GPU
    assert set(ng.NOISE_MAX) == set(ng.REF_SETS)


def test_canonical_sha固定向量_noise_gate与gate_set同口径():
    obj = copy.deepcopy(PIN.CANONICAL_VECTOR_OBJ)
    ng = F.noise_gate()
    assert ng.canonical_json({k: v for k, v in obj.items() if k != "sha256"}) == PIN.CANONICAL_VECTOR_TEXT
    assert ng.payload_sha256(obj) == PIN.CANONICAL_VECTOR_SHA256
    gs = F.load_script("parity/gate_set.py")
    assert gs.payload_sha256(obj) == PIN.CANONICAL_VECTOR_SHA256
    # 顶层 sha256 键的取值不参与；任一内容改动都改变 digest
    assert ng.payload_sha256(dict(obj, sha256="其他")) == PIN.CANONICAL_VECTOR_SHA256
    assert ng.payload_sha256(dict(obj, note="中文 uni")) != PIN.CANONICAL_VECTOR_SHA256
    assert obj == PIN.CANONICAL_VECTOR_OBJ  # 不改入参
