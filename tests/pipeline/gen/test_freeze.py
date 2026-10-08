"""``_freeze_min``：规格抽签封存退役后保留的两个纯函数（旧仓 ``_freeze.py`` 逐字摘出）。

* ``_movecube_way``：录像局用的 MoveCube 运动方式取 ``initializations`` 里序号最大（按整数比较）那次的 ``way_idx``；
* ``write_jsonl_exclusive``：目标已存在即拒、不覆盖、不留临时文件；写出的每行是 ``hard_specs.canonical_json``，
  写出的签名规格经真实 ``hard_specs.load_specs`` 读回逐行相等。

期望值手写（哪一次的 way、文件内容），不从被测函数推出。
"""
from __future__ import annotations

import json

import pytest

from gen_world import FM, H, fake_spec, freeze_file, read_rows


def test_movecube_way_reads_last_initialization():
    spec = fake_spec("MoveCube", "xhard4", 0, 0, way=2)  # 构造期 initializations.0 写的是另一个值
    assert spec["initializations"]["0"]["way_idx"] != 2
    assert FM._movecube_way(spec) == 2
    spec["initializations"]["10"] = {"way_idx": 0}  # 序号按整数比较：10 > 1（字符串比较会取到 "1"）
    assert FM._movecube_way(spec) == 0
    assert FM._movecube_way({"spec_kind": "x"}) is None
    assert FM._movecube_way({"initializations": {}}) is None
    assert FM._movecube_way({"initializations": {"3": {"other": 1}}}) is None
    assert FM._movecube_way(None) is None


def test_write_exclusive_canonical_lines_and_refuses_existing(tmp_path):
    path = tmp_path / "xhard1" / "specs.jsonl"
    records = [{"b": 1, "a": "中文"}, {"z": [1.5, None, True]}]
    FM.write_jsonl_exclusive(path, records)
    assert path.read_text(encoding="utf-8") == '{"a":"中文","b":1}\n{"z":[1.5,null,true]}\n'
    before = path.read_bytes()
    with pytest.raises(H.SpecsError, match="禁止覆盖"):
        FM.write_jsonl_exclusive(path, [{"other": 1}])
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["specs.jsonl"]  # 不留临时文件


def test_signed_specs_roundtrip_through_load_specs(tmp_path):
    """测试侧签名器 + 真实 ``write_jsonl_exclusive`` 落盘的 /4 文件，经真实 ``hard_specs.load_specs`` 读回逐行相等。"""
    path = tmp_path / "xhard4" / "specs.jsonl"
    header, rows = freeze_file(path, "xhard4", {"MoveCube": (3, 6)}, ways={"MoveCube": lambda e: e % 3})
    loaded_header, loaded_rows = H.load_specs(path, expected_cells=H.resolve_cell_table({("MoveCube", "xhard4"): 3}),
                                              check_fingerprint=False)
    assert (loaded_header, loaded_rows) == (header, rows) == read_rows(path)
    assert [FM._movecube_way(r["spec"]) for r in rows] == [0, 1, 2, 0, 1, 2]
    assert [r["candidate"] for r in rows if r["selected"]] == [0, 1, 2]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["schema"] == H.SCHEMA
