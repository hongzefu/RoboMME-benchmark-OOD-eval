"""C15 对拍容差文件 ``scripts/configs/hard-parity-tolerances.json`` 自洽（只读）。

文件由 ``hard_parity.py compare --pair O:P --calibrate`` 写入：阈值 = 原始最大值 × 1.5（帧数向上取整）与下界取大。
这里不复刻该公式，而是把文件里记录的 ``raw_max`` 当作一批「校准对」重新喂真实 ``calibrate``，要求它产出的
四个阈值与文件里的逐个相等、且未触合理性上界；``load_tolerances`` 读出的四项与文件一致。
改阈值不改 ``raw_max``（或反之）都会让本用例失败——这正是 R21「改阈值须改本文件并写进留档」的机检面。
"""
from __future__ import annotations

import json

import pytest

import parity_fixtures as F

PATH = F.REPO / "scripts" / "configs" / "hard-parity-tolerances.json"


@pytest.fixture(scope="module")
def hp():
    return F.hard_parity()


def test_容差文件由原始最大值重算可得(hp):
    raw = PATH.read_bytes()
    doc = json.loads(raw)
    pair = {"action_max": doc["raw_max"]["action_max"], "state_max": doc["raw_max"]["state_max"],
            "image_mad": doc["raw_max"]["image_mad"], "frames_diff": doc["raw_max"]["frames_max"], "error": None}
    zero = {"action_max": 0.0, "state_max": 0.0, "image_mad": 0.0, "frames_diff": 0, "error": None}
    ok, info = hp.calibrate([pair, zero, {"error": "打不开的对不参与"}])
    assert ok and info["ceiling_hit"] == []
    for key in hp.TOL_KEYS.values():
        assert info["payload"][key] == pytest.approx(doc[key], rel=0, abs=1e-12), key
    assert doc["calibrated_from"] == "O:P native" and isinstance(doc["n"], int) and doc["n"] > 0
    assert PATH.read_bytes() == raw


def test_load_tolerances与文件一致_缺文件即拒(hp, tmp_path, monkeypatch):
    doc = json.loads(PATH.read_text(encoding="utf-8"))
    got = hp.load_tolerances()
    assert set(got) == set(hp.TOL_KEYS) and all(got[k] == float(doc[v]) for k, v in hp.TOL_KEYS.items())
    monkeypatch.setattr(hp, "TOLERANCES", tmp_path / "none.json")
    with pytest.raises(hp.ParityError):
        hp.load_tolerances()


def test_calibrate_超合理性上界即不通过(hp):
    huge = {"action_max": 1e6, "state_max": 0.0, "image_mad": 0.0, "frames_diff": 0, "error": None}
    ok, info = hp.calibrate([huge])
    assert not ok and info["ceiling_hit"] == ["action_max"]
