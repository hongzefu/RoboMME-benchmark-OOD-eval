"""C15 原始 train 身份冻结与对拍参数校验 ``scripts/parity/train_split_parity.py``（纯函数，不读官方仓库、不起仿真）。

- ``recovery_mode``：官方 ``EpisodeJob.recovery_mode`` 规则的手写表；
- ``_validate_records``：官方 metadata 结构的逐条拒绝；
- ``cross_check_identity``：清单与官方原始字节逐条＋集合双向比较（换 seed、换顺序、重复补齐都要计 mismatch）；
- ``select_subset``：每任务每难度按官方原顺序取前 per_cell 条，不足即拒；
- ``_parse_paths``／``_parse_shard``：参数校验。
``compare_h5_pair`` 的比较范围在 ``test_h5_comparators.py``。
"""
from __future__ import annotations

import copy
import json

import pytest

import parity_fixtures as F


@pytest.fixture(scope="module")
def tsp():
    return F.train_split()


def test_recovery_mode手写表(tsp):
    assert [tsp.recovery_mode(ep) for ep in range(8)] == ["z", "z", "z", "xy", "xy", "xy", None, None]


BASE = {"env_id": "BinFill", "record_count": 2, "records": [
    {"task": "BinFill", "episode": 0, "seed": 4000, "difficulty": "easy"},
    {"task": "BinFill", "episode": 1, "seed": 4100, "difficulty": "medium"}]}


def _mut(fn):
    obj = copy.deepcopy(BASE)
    fn(obj)
    return obj


@pytest.mark.parametrize("name,obj", [
    ("顶层不是对象", []),
    ("env_id 与文件名不符", _mut(lambda o: o.update(env_id="PickXtimes"))),
    ("record_count 与实际不符", _mut(lambda o: o.update(record_count=3))),
    ("条数不足", _mut(lambda o: (o["records"].pop(), o.update(record_count=1)))),
    ("多一个字段", _mut(lambda o: o["records"][0].update(attempt=0))),
    ("task 字段错", _mut(lambda o: o["records"][0].update(task="X"))),
    ("episode 非整数", _mut(lambda o: o["records"][0].update(episode=0.0))),
    ("seed 是布尔", _mut(lambda o: o["records"][0].update(seed=True))),
    ("难度非法", _mut(lambda o: o["records"][0].update(difficulty="xhard1"))),
    ("episode 重复", _mut(lambda o: o["records"][1].update(episode=0))),
    ("episode 不是 0..n-1", _mut(lambda o: o["records"][1].update(episode=5))),
])
def test_官方metadata结构逐条拒绝(tsp, name, obj):
    assert len(tsp._validate_records("BinFill", copy.deepcopy(BASE), 2)) == 2
    with pytest.raises(tsp.IdentityFreezeError):
        tsp._validate_records("BinFill", obj, 2)


def _meta(records):
    raw = json.dumps({"env_id": "BinFill", "record_count": len(records), "records": records}).encode()
    return {"BinFill": {"raw": raw}}


def test_清单与官方逐条双向比较(tsp):
    recs = BASE["records"]
    meta = _meta(recs)
    rows = [dict(r) for r in recs]
    assert tsp.cross_check_identity(rows, meta, tasks=("BinFill",)) == 0
    replaced = [dict(recs[0]), dict(recs[1], seed=4101)]  # 用公式值替换实际 seed
    assert tsp.cross_check_identity(replaced, meta, tasks=("BinFill",)) > 0
    swapped = [dict(recs[1]), dict(recs[0])]  # 集合相同、顺序不同
    assert tsp.cross_check_identity(swapped, meta, tasks=("BinFill",)) > 0
    padded = [dict(recs[0]), dict(recs[0])]  # 遗漏一条、用重复行补齐
    assert tsp.cross_check_identity(padded, meta, tasks=("BinFill",)) > 0
    assert tsp.cross_check_identity(rows[:1], meta, tasks=("BinFill",)) > 0


def test_子集按官方原顺序取前per_cell条_不足即拒(tsp):
    rows = [{"task": "BinFill", "episode": e, "seed": e, "difficulty": d}
            for e, d in enumerate(["hard", "easy", "medium", "easy", "hard", "medium", "easy", "medium", "hard"])]
    subset, eps = tsp.select_subset(rows, per_cell=2, tasks=("BinFill",))
    # 手写期望：easy 取 1、3；medium 取 2、5；hard 取 0、4（难度序 easy→medium→hard，各自保持原顺序）
    assert [r["episode"] for r in subset] == [1, 3, 2, 5, 0, 4]
    assert eps == {"BinFill": [0, 1, 2, 3, 4, 5]}
    with pytest.raises(tsp.IdentityFreezeError):
        tsp.select_subset(rows, per_cell=4, tasks=("BinFill",))


def test_参数校验(tsp):
    for bad in ("A1,A1", "A3", "", " , "):
        with pytest.raises(tsp.IdentityFreezeError):
            tsp._parse_paths(bad)
    assert tsp._parse_paths("D,A1") == ["A1", "D"]
    for bad in ("5/4", "1-4", "0/4", "1/0", "a/b"):
        with pytest.raises(tsp.IdentityFreezeError):
            tsp._parse_shard(bad)
    assert tsp._parse_shard("2/4") == (2, 4) and tsp._parse_shard(None) is None
