"""C15 ``noise_gate.py gen-compare``：两侧生成 h5 逐局归入互斥五类（byte_equal／diverge／gen_fail／structural／unknown）。

正例覆盖五类各至少一局；负例：侧里重复行（含「遗漏身份用重复行补齐」）、档位冲突、h5 打不开、dtype 改变、帧号不连续、
记录 sha 与文件不符、缺文件、基础设施失败、内容全同但字节不同；白名单多态字段 ``action/waypoint_action`` 的放行与拒绝；
三种侧格式（生成输出根、results.jsonl 根、v8-delivery 清单）与身份清单四种格式（含空文件拒收）。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

import parity_fixtures as F

T = "PickXtimes"


@pytest.fixture(scope="module")
def ng():
    return F.noise_gate()


def side(root: Path, entries: dict[int, dict], tier: str = "xhard1") -> Path:
    """entries[seed] = {h5_kw: write_h5 参数 | None 表示失败, ok, error_type, tier, raw: 原样覆盖行}。"""
    lines = []
    for seed, e in entries.items():
        if e.get("skip"):
            continue
        rel = f"episodes/{T}_episode_{seed}/hdf5_files/x.h5"
        line = F.id_line(T, e.get("tier", tier), seed, ok=e.get("ok", True), error_type=e.get("error_type"))
        if e.get("ok", True):
            path = root / rel
            if "bytes" in e:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(e["bytes"])
            else:
                F.write_h5(path, seed=seed, **e.get("h5_kw", {}))
            line.update(path=rel, sha256=F.sha256(path))
        line.update(e.get("raw", {}))
        lines.append(line)
        if e.get("dup"):
            lines.append(dict(line))
    return F.write_run(root, lines, workers=4, gpu="NVIDIA A40")


def ids(seeds, tier="xhard1"):
    return [{"id": f"{T}|{tier}|{s}", "task": T, "tier": tier, "seed": s} for s in seeds]


def classify(ng, tmp_path, ref_e: dict, new_e: dict, seeds=None, tier="xhard1"):
    r, n = side(tmp_path / "ref", ref_e, tier), side(tmp_path / "new", new_e, tier)
    rows, counts = ng.gen_compare(r, n, ids(seeds if seeds is not None else sorted(ref_e), tier))
    return {x["seed"]: x for x in rows}, counts


def test_五类各至少一例_互斥且完备(ng, tmp_path):
    ref = {s: {} for s in range(6)}
    new = {0: {}, 1: {"h5_kw": {"joint_offset_from": 2}}, 2: {"ok": False, "error_type": "DatasetGenerationError"},
           3: {"h5_kw": {"drop": (1, "obs/gripper_state")}}, 4: {"ok": False, "error_type": "TimeoutError"},
           5: {"skip": True}}
    by, counts = classify(ng, tmp_path, ref, new)
    want = {0: ("byte_equal", None), 1: ("diverge", None), 2: ("gen_fail", "DatasetGenerationError"),
            3: ("structural", f"episode_3:frame1:schema_new"), 4: ("unknown", None), 5: ("unknown", "missing_new")}
    for seed, (cls, reason) in want.items():
        assert by[seed]["class"] == cls, (seed, by[seed])
        if reason:
            assert by[seed]["reason"] == reason
    assert by[1]["first_divergence"] == 2 and by[1]["action_max"] == pytest.approx(0.5)
    assert by[2]["fail_side"] == "new" and by[4]["reason"].startswith("infra_new:TimeoutError")
    assert counts == {"byte_equal": 1, "diverge": 1, "gen_fail": 1, "structural": 1, "unknown": 2}
    assert sum(counts.values()) == len(ref) and all(r["class"] in ng.GEN_CLASSES for r in by.values())


def test_参照侧失败_两侧都失败(ng, tmp_path):
    ref = {0: {"ok": False, "error_type": "PlannerExhausted"}, 1: {"ok": False, "error_type": "PlannerExhausted"}}
    new = {0: {}, 1: {"ok": False, "error_type": "DatasetGenerationError"}}
    by, _ = classify(ng, tmp_path, ref, new)
    assert (by[0]["class"], by[0]["fail_side"]) == ("gen_fail", "ref")
    assert (by[1]["class"], by[1]["fail_side"]) == ("gen_fail", "both")


def test_侧里身份重复_遗漏身份用重复行补齐(ng, tmp_path):
    """M15c：新侧去掉 seed 2、把 seed 1 写两遍，行数不变——两局都不得判 byte_equal。"""
    ref = {0: {}, 1: {}, 2: {}}
    new = {0: {}, 1: {"dup": True}, 2: {"skip": True}}
    by, counts = classify(ng, tmp_path, ref, new)
    assert (by[1]["class"], by[1]["reason"]) == ("unknown", "duplicate_new")
    assert (by[2]["class"], by[2]["reason"]) == ("unknown", "missing_new")
    assert counts["byte_equal"] == 1


def test_档位冲突(ng, tmp_path):
    by, _ = classify(ng, tmp_path, {0: {}}, {0: {"raw": {"tier": "xhard2"}}})
    assert (by[0]["class"], by[0]["reason"]) == ("unknown", "tier_mismatch_new:xhard2")


def test_xhard0身份接受官方难度名hard_其余档不接受(ng, tmp_path):
    by, _ = classify(ng, tmp_path, {0: {"raw": {"tier": "hard"}}}, {0: {}}, tier="xhard0")
    assert by[0]["class"] == "byte_equal"
    by, _ = classify(ng, tmp_path / "b", {0: {"raw": {"tier": "hard"}}}, {0: {}}, tier="xhard1")
    assert by[0]["reason"] == "tier_mismatch_ref:hard"


def test_h5打不开(ng, tmp_path):
    by, _ = classify(ng, tmp_path, {0: {}}, {0: {"bytes": b"\x89HDF-broken" * 10}})
    assert by[0]["class"] == "unknown" and by[0]["reason"].startswith("invalid_h5_new:open_error")


def test_dtype改变与帧号不连续判结构不同(ng, tmp_path):
    by, _ = classify(ng, tmp_path, {0: {}, 1: {}},
                     {0: {"h5_kw": {"joint_dtype": "float32"}}, 1: {"h5_kw": {"frame_ids": [0, 1, 2, 5]}}})
    assert (by[0]["class"], by[0]["reason"]) == ("structural", "episode_0:signature")
    assert (by[1]["class"], by[1]["reason"]) == ("structural", "episode_1:frames_not_contiguous")


def test_记录sha与文件不符_缺文件(ng, tmp_path):
    by, _ = classify(ng, tmp_path, {0: {}, 1: {}}, {0: {"raw": {"sha256": "0" * 64}},
                                                    1: {"raw": {"path": "episodes/none/x.h5"}}})
    assert (by[0]["class"], by[0]["reason"]) == ("unknown", "sha_recorded_mismatch_new")
    assert (by[1]["class"], by[1]["reason"]) == ("unknown", "missing_file_new")
    # --trust-recorded-sha：不重算、直接信记录（弱）——记录的 sha 不同就去比内容，内容全同仍归原因不明
    rows, _ = ng.gen_compare(tmp_path / "ref", tmp_path / "new", ids([0]), rehash=False)
    assert (rows[0]["class"], rows[0]["reason"]) == ("unknown", "content_equal_bytes_differ")


def test_内容全同但字节不同判原因不明(ng, tmp_path):
    r = side(tmp_path / "ref", {0: {}})
    new_root = tmp_path / "new"
    path = new_root / f"episodes/{T}_episode_0/hdf5_files/x.h5"
    src = F.write_h5(tmp_path / "src.h5", seed=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 同一内容换文件格式版本（libver=latest）逐对象复制：内容全同、字节不同
    with h5py.File(src, "r") as a, h5py.File(path, "w", libver="latest") as b:
        for key in a:
            a.copy(a[key], b, name=key)
    F.write_run(new_root, [F.id_line(T, "xhard1", 0, sha=F.sha256(path), path=str(path.relative_to(new_root)))],
                workers=4, gpu="NVIDIA A40")
    assert F.sha256(path) != F.sha256(tmp_path / "ref" / f"episodes/{T}_episode_0/hdf5_files/x.h5")
    rows, _ = ng.gen_compare(r, new_root, ids([0]))
    assert (rows[0]["class"], rows[0]["reason"]) == ("unknown", "content_equal_bytes_differ")


def test_交换两个身份的产物_总数不变也不判相同(ng, tmp_path):
    """M10：新侧把 seed 0 与 seed 1 的 h5 对调（局数、成功数都不变）。"""
    r = side(tmp_path / "ref", {0: {}, 1: {}})
    new_root = tmp_path / "new"
    lines = []
    for seed, src in ((0, 1), (1, 0)):
        rel = f"episodes/{T}_episode_{seed}/hdf5_files/x.h5"
        F.write_h5(new_root / rel, seed=src)
        lines.append(F.id_line(T, "xhard1", seed, sha=F.sha256(new_root / rel), path=rel))
    F.write_run(new_root, lines, workers=4, gpu="NVIDIA A40")
    rows, counts = ng.gen_compare(r, new_root, ids([0, 1]))
    assert counts["byte_equal"] == 0 and {x["class"] for x in rows} == {"structural"}


# ── 白名单多态字段 action/waypoint_action ─────────────────────────────────────────────


def test_白名单只登记waypoint一个字段且只允许两种签名(ng):
    assert set(ng.POLYMORPHIC_KEYS) == {"action/waypoint_action"}
    assert ng.POLYMORPHIC_KEYS["action/waypoint_action"] == frozenset({("float32", (7,)), ("float64", (7,))})


@pytest.mark.parametrize("ref_wp,new_wp,new_kw,cls,reason_tail", [
    # 占位（float32 NaN）与实值（float64）随轨迹变化：分叉后签名不同按内容差异计
    ({0: "nan", 1: "real", 2: "real", 3: "nan"}, {0: "nan", 1: "real", 2: "nan", 3: "real"},
     {"joint_offset_from": 2}, "diverge", None),
    # 一侧全程实值、另一侧后半段占位
    ({0: "real", 1: "real", 2: "real", 3: "real"}, {0: "real", 1: "real", 2: "nan", 3: "nan"},
     {"joint_offset_from": 2}, "diverge", None),
    # 白名单集合外签名（int64）
    ({0: "real", 1: "real", 2: "real", 3: "real"}, {0: "real", 1: "real", 2: "real", 3: "int"},
     {"joint_offset_from": 2}, "structural", "frame3:polymorphic_signature_new"),
    # 白名单字段某帧不存在
    ({0: "real", 1: "real", 2: "real", 3: "real"}, {0: "real", 1: "real", 3: "real"},
     {"joint_offset_from": 2}, "structural", "frame2:polymorphic_signature_new"),
    # 第 0 帧签名不同
    ({0: "real", 1: "real", 2: "real", 3: "real"}, {0: "nan", 1: "real", 2: "real", 3: "real"},
     {}, "structural", "frame0:schema"),
    # 非白名单字段只在一侧出现
    ({}, {0: "real", 1: "real", 2: "real", 3: "real"}, {}, "structural", "dataset_universe"),
])
def test_白名单字段的放行与拒绝(ng, tmp_path, ref_wp, new_wp, new_kw, cls, reason_tail):
    by, _ = classify(ng, tmp_path, {0: {"h5_kw": {"waypoint": ref_wp}}},
                     {0: {"h5_kw": {"waypoint": new_wp, **new_kw}}})
    row = by[0]
    assert row["class"] == cls, row
    if reason_tail:
        assert row["reason"].endswith(reason_tail)
    else:
        assert row["first_divergence"] == 2 and row["variable_keys"] == ["action/waypoint_action"]


# ── 侧格式与身份清单格式 ───────────────────────────────────────────────────────────────


def test_交付清单侧与results侧(ng, tmp_path):
    h5 = F.write_h5(tmp_path / "d" / "e0.h5", seed=0)
    delivery = tmp_path / "delivery.json"
    delivery.write_text(json.dumps({"schema": "v8-delivery/1", "rows": [
        {"task": T, "tier": "xhard1", "seed": 0, "h5": "d/e0.h5", "h5_sha256": F.sha256(h5)}]}), encoding="utf-8")
    side_d = ng.load_side(delivery)
    assert side_d[(T, 0)]["status"] == "ok" and Path(side_d[(T, 0)]["h5"]) == h5.resolve()
    res_root = tmp_path / "res"
    F.write_jsonl(res_root / "results.jsonl", [
        {"kind": "result", "record": {"task": T, "seed": 0, "tier": "xhard1", "ok": False, "error_type": "X"}},
        {"kind": "result", "record": {"task": T, "seed": 0, "tier": "xhard1", "ok": True, "h5": str(h5),
                                      "h5_sha256": F.sha256(h5)}},
        {"kind": "progress"}])
    side_r = ng.load_side(res_root)
    assert side_r[(T, 0)]["status"] == "ok" and "duplicate" not in side_r[(T, 0)]  # 重试按最后一行
    rows, counts = ng.gen_compare(delivery, res_root, ids([0]))
    assert counts["byte_equal"] == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "rows": []}), encoding="utf-8")
    with pytest.raises(ng.GateError):
        ng.load_side(bad)
    with pytest.raises(ng.GateError):
        ng.load_side(tmp_path / "nothing")
    (tmp_path / "emptydir").mkdir()
    with pytest.raises(ng.GateError):
        ng.load_side(tmp_path / "emptydir")


def test_身份清单四种格式_重复与空文件拒收(ng, tmp_path):
    lst = tmp_path / "d0.json"
    lst.write_text(json.dumps([{"task": "A", "seed": 1, "source_episode": 3, "builder_episode": 0}]))
    got = ng.load_identities(lst)
    assert got == [{"id": "A|xhard0|1", "task": "A", "tier": "xhard0", "seed": 1, "source_episode": 3,
                    "builder_episode": 0}]
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"rows": [{"task": "A", "difficulty": "xhard2", "seed": 5}]}))
    assert ng.load_identities(rows)[0]["id"] == "A|xhard2|5"
    jl = F.write_jsonl(tmp_path / "x.jsonl", [{"task": "A", "tier": "xhard3", "seed": 6}])
    assert ng.load_identities(jl)[0]["id"] == "A|xhard3|6"
    gate = ng.load_identities(F.REPO / "scripts" / "configs" / "gate-set-xhard0-48.json")
    assert gate and all(g["tier"] == "xhard0" and "builder_episode" in g for g in gate)
    dup = F.write_jsonl(tmp_path / "dup.jsonl", [{"task": "A", "seed": 5}, {"task": "A", "seed": 5}])
    with pytest.raises(ng.GateError, match="重复"):
        ng.load_identities(dup)
    for name, text in (("empty.json", ""), ("empty.jsonl", ""), ("obj.json", "{}")):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        if name == "empty.jsonl":
            assert ng.load_identities(p) == []  # 空 jsonl 读成空清单；由下游「空集拒收」兜底（见 gen-regress）
        else:
            with pytest.raises(Exception):  # noqa: B017 冻结文件被清空：不得读成空清单
                ng.load_identities(p)


def test_冻结检查集被清空_不得读成空清单(ng, tmp_path):
    """M15b：gate-set 冻结文件被清空（零字节）或只剩 {}，读身份必须报错。"""
    for text in ("", "{}", json.dumps({"schema": "gate-set-v9/2", "rows": [], "sha256": "0" * 64})):
        p = tmp_path / "gate.json"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(Exception):  # noqa: B017
            ng.load_identities(p)


# ── CLI 与读回 ─────────────────────────────────────────────────────────────────────


def test_cli_gen_compare_写meta与逐局行_读回校验(ng, tmp_path, capsys):
    r = side(tmp_path / "ref", {0: {}, 1: {}})
    n = side(tmp_path / "new", {0: {}, 1: {"h5_kw": {"joint_offset_from": 1}}})
    idf = F.write_jsonl(tmp_path / "ids.jsonl", [{"task": T, "tier": "xhard1", "seed": s} for s in (0, 1)])
    out = tmp_path / "cmp.jsonl"
    assert ng.main(["gen-compare", "--ref", str(r), "--new", str(n), "--identities", str(idf), "--out", str(out)]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line == "GEN_PAIR=INFO ref=ref new=new n=2 byte_equal=1 diverge=1 gen_fail=0 structural=0 unknown=0"
    meta, rows = ng.read_gen_pairs(out)
    assert meta["schema"] == "noise-gen-compare/1" and meta["n"] == 2 and len(rows) == 2
    assert ng.gen_counts(rows) == {"byte_equal": 1, "diverge": 1, "gen_fail": 0, "structural": 0, "unknown": 0,
                                   "diff": 1, "n": 2}
    lines = out.read_text(encoding="utf-8").splitlines()
    for bad in (lines[1:],                                                         # 缺 meta
                [lines[0], lines[1], lines[1]],                                    # 身份重复
                [lines[0], lines[1].replace('"class": "byte_equal"', '"class": "same"')]):  # 类别非法
        out.write_text("\n".join(bad) + "\n", encoding="utf-8")
        with pytest.raises(ng.GateError):
            ng.read_gen_pairs(out)
