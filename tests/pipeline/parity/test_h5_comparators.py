"""C15 三个 h5 比较器的比较范围：同一批合成变体同时喂

- ``noise_gate.classify_pair``／``compare_h5``（生成噪声五类，读两侧「合法 h5」后逐帧结构＋内容）；
- ``train_split_parity.compare_h5_pair``（整文件 sha → 全字段逐字节，含根属性 F-8）；
- ``hard_parity.pair_metrics`` ＋ ``classify_terminal``（setup／schema／四项容差指标／首个分叉步）。

每个变体只改一处，逐格钉死「谁能看见、谁看不见」。看不见的格子是**现状记录**（比较器按设计只管那几项），
不是放行：例如 hard_parity 只读 joint_action／joint_state／gripper_state／两路 RGB，根属性与新增字段对它不可见。
F-5：空文件、非 h5 文件在 noise_gate 里不得判 byte_equal（「产物合法」与「两份相同」分开判）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

import parity_fixtures as F

SEED = 7
EP = f"episode_{SEED}"


@pytest.fixture(scope="module")
def mods():
    return F.noise_gate(), F.train_split(), F.hard_parity()


def pair(tmp_path: Path, left_kw: dict, right_kw: dict, *, copy: bool = False) -> tuple[Path, Path]:
    left = F.write_h5(tmp_path / "l.h5", seed=SEED, **left_kw)
    right = tmp_path / "r.h5"
    if copy:
        right.write_bytes(left.read_bytes())
    else:
        F.write_h5(right, seed=SEED, **right_kw)
    return left, right


def noise_class(ng, left: Path, right: Path) -> dict:
    ident = {"id": f"T|xhard1|{SEED}", "task": "T", "tier": "xhard1", "seed": SEED}
    side = lambda p: {"h5": str(p), "sha256": None, "status": "ok", "tier": None}  # noqa: E731
    return ng.classify_pair(ident, side(left), side(right))


def train_paths(res: dict) -> set[tuple[str, str]]:
    return {(m["path"], m["kind"]) for m in res["mismatches"]}


# ── 同一批变体，三个比较器逐个钉死 ────────────────────────────────────────────────────


def test_相同字节_三者都判相同(mods, tmp_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, {}, copy=True)
    assert noise_class(ng, left, right)["class"] == "byte_equal"
    assert ts.compare_h5_pair(left, right)["sha_equal"] == 1
    m = hp.pair_metrics(str(left), str(right))
    assert (m["error"], m["setup_equal"], m["schema_equal"], m["contiguous"], m["first_divergence"]) == \
        (None, True, True, True, None)


@pytest.mark.parametrize("where,kw,noise_reason,train_path", [
    ("根属性（F-8）", {"root_attrs": {"producer": "x"}}, "file_attrs", "/"),
    ("episode 组属性", {"episode_attrs": {"note": "x"}}, f"{EP}:attrs", EP),
    ("setup 组属性", {"setup_attrs": {"note": "x"}}, f"{EP}/setup:schema", f"{EP}/setup"),
])
def test_属性变化_noise与train看得见_hard看不见(mods, tmp_path, where, kw, noise_reason, train_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, kw)
    got = noise_class(ng, left, right)
    assert (got["class"], got["reason"]) == ("structural", noise_reason), where
    res = ts.compare_h5_pair(left, right)
    assert res["sha_equal"] == 0 and (train_path, "attr_keys") in train_paths(res) and res["field_mismatch"] >= 1
    m = hp.pair_metrics(str(left), str(right))
    assert (m["setup_equal"], m["schema_equal"], m["first_divergence"]) == (True, True, None)  # 现状：盲区


def test_属性同键不同值_train按值报(mods, tmp_path):
    _ng, ts, _hp = mods
    left, right = pair(tmp_path, {"root_attrs": {"producer": "x"}}, {"root_attrs": {"producer": "y"}})
    assert ("/@producer", "attr_value") in train_paths(ts.compare_h5_pair(left, right))


def test_两侧都新增同一字段且只在该字段不同(mods, tmp_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {"extra_field": {}}, {"extra_field": {2: 1.0}})
    got = noise_class(ng, left, right)
    assert (got["class"], got["first_divergence"]) == ("diverge", 2)
    res = ts.compare_h5_pair(left, right)
    assert train_paths(res) == {(f"{EP}/timestep_2/obs/extra", "value")}
    m = hp.pair_metrics(str(left), str(right))
    assert (m["schema_equal"], m["first_divergence"], m["action_max"], m["state_max"]) == (True, None, 0.0, 0.0)
    # hard_parity 看不见这一字段：两侧都成功、sha 不同 → 终态记为「噪声」而非结构不同（现状记录）
    key = ("T", "xhard1", SEED)
    denom = {"in_all": {key}, "dup_keys": set(), "delivery_sha": {}, "frozen_sha": {}}
    a = {"success": True, "path": "l.h5", "sha256": F.sha256(left)}
    b = {"success": True, "path": "r.h5", "sha256": F.sha256(right)}
    assert hp.classify_terminal(key, a, b, m, denom, True, False) == "noise"


def test_中间帧缺一个字段(mods, tmp_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, {"drop": (2, "info/is_completed")})
    got = noise_class(ng, left, right)
    assert got["class"] == "structural" and got["reason"] == f"{EP}:frame2:schema_new"
    res = ts.compare_h5_pair(left, right)
    assert res["missing_right"] == [f"{EP}/timestep_2/info/is_completed"] and res["field_mismatch"] >= 1
    m = hp.pair_metrics(str(left), str(right))
    assert m["schema_equal"] is True  # 现状：schema_of 把各帧折叠成集合，单帧缺字段看不见
    # 缺的是 pair_metrics 要读的字段时直接报错（判定层不等）
    left2, right2 = pair(tmp_path / "b", {}, {"drop": (2, "obs/joint_state")})
    assert hp.pair_metrics(str(left2), str(right2))["error"] is not None


def test_dtype改变(mods, tmp_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, {"joint_dtype": "float32"})
    got = noise_class(ng, left, right)
    assert (got["class"], got["reason"]) == ("structural", f"{EP}:signature")
    res = ts.compare_h5_pair(left, right)
    assert {k for _p, k in train_paths(res)} == {"dtype_or_shape"}
    assert hp.pair_metrics(str(left), str(right))["schema_equal"] is False


def test_帧号不连续(mods, tmp_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, {"frame_ids": [0, 1, 3, 4]})
    got = noise_class(ng, left, right)
    assert (got["class"], got["reason"]) == ("structural", f"{EP}:frames_not_contiguous")
    assert ts.compare_h5_pair(left, right)["field_mismatch"] > 0
    assert hp.pair_metrics(str(left), str(right))["contiguous"] is False


@pytest.mark.parametrize("kw,frame,field", [
    ({"joint_offset_from": 2}, 2, "action/joint_action"),
    ({"state_offset_from": 1}, 1, "obs/joint_state"),
    ({"gripper_offset_from": 3}, 3, "obs/gripper_state"),
])
def test_某一数据集从某帧起分叉_三者都抓到且分叉步一致(mods, tmp_path, kw, frame, field):
    """M15a：比较器跳过任何一个数据集，对应参数化实例都会失败。"""
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, kw)
    got = noise_class(ng, left, right)
    assert (got["class"], got["first_divergence"]) == ("diverge", frame)
    res = ts.compare_h5_pair(left, right)
    assert {p for p, _k in train_paths(res)} == {f"{EP}/timestep_{i}/{field}" for i in range(frame, 4)}
    m = hp.pair_metrics(str(left), str(right))
    assert m["first_divergence"] == frame
    want = {"action/joint_action": "action_max"}.get(field, "state_max")
    assert m[want] == pytest.approx(0.5) and m["setup_equal"] and m["schema_equal"]


def test_第0帧内容不同(mods, tmp_path):
    ng, _ts, hp = mods
    left, right = pair(tmp_path, {}, {"frame0_pixel": True})
    got = noise_class(ng, left, right)
    assert (got["class"], got["reason"]) == ("structural", f"{EP}:frame0_content")
    m = hp.pair_metrics(str(left), str(right))
    assert m["first_divergence"] == 0 and m["image_mad"] > 0


def test_setup取值不同(mods, tmp_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, {"setup_seed": SEED + 1})
    got = noise_class(ng, left, right)
    assert (got["class"], got["reason"]) == ("structural", f"{EP}/setup/seed:value")
    assert (f"{EP}/setup/seed", "value") in train_paths(ts.compare_h5_pair(left, right))
    assert hp.pair_metrics(str(left), str(right))["setup_equal"] is False


def test_只多一帧(mods, tmp_path):
    ng, ts, hp = mods
    left, right = pair(tmp_path, {}, {"frames": 5})
    got = noise_class(ng, left, right)
    assert (got["class"], got["first_divergence"], got["frames_diff"]) == ("diverge", 3, 1)
    res = ts.compare_h5_pair(left, right)
    assert res["timestep_count"] == {"left": 4, "right": 5}
    m = hp.pair_metrics(str(left), str(right))
    assert (m["frames_diff"], m["common"]) == (1, 4)


@pytest.mark.parametrize("payload,reason", [(b"", "empty"), (b"not an hdf5 file\n" * 4, "open_error:")],
                         ids=["空文件", "非h5"])
def test_F5_空文件或非h5_两份字节相同也不判byte_equal(mods, tmp_path, payload, reason):
    ng, ts, hp = mods
    left, right = tmp_path / "l.h5", tmp_path / "r.h5"
    left.write_bytes(payload)
    right.write_bytes(payload)
    got = noise_class(ng, left, right)
    assert got["class"] == "unknown" and got["reason"].startswith(f"invalid_h5_ref:{reason}")
    # train 比较器只判「两份相同」，不判产物合法（现状记录：两个空文件 sha_equal=1）
    assert ts.compare_h5_pair(left, right)["sha_equal"] == 1
    assert hp.pair_metrics(str(left), str(right))["error"] is not None


def test_无顶层键的h5不合法(mods, tmp_path):
    import h5py

    ng, _ts, _hp = mods
    left = tmp_path / "l.h5"
    with h5py.File(left, "w"):
        pass
    right = tmp_path / "r.h5"
    right.write_bytes(left.read_bytes())
    got = noise_class(ng, left, right)
    assert (got["class"], got["reason"]) == ("unknown", "invalid_h5_ref:no_keys")


def test_NaN按位相同不算差异(mods, tmp_path):
    ng, ts, _hp = mods
    left, right = pair(tmp_path, {"waypoint": {1: "nan"}, "root_attrs": {"r": 1}},
                       {"waypoint": {1: "nan"}, "root_attrs": {"r": 1}, "frames": 4})
    # 两份内容相同（含 NaN 占位），字节也相同：train 逐字节相同
    assert ts.compare_h5_pair(left, right)["field_mismatch"] == 0
    assert noise_class(ng, left, right)["class"] == "byte_equal"


# ── hard_parity 五终态（classify_terminal）的负例 ─────────────────────────────────────


@pytest.fixture()
def term(mods, tmp_path):
    _ng, _ts, hp = mods
    left, right = pair(tmp_path, {}, {}, copy=True)
    m = hp.pair_metrics(str(left), str(right))
    key = ("T", "xhard1", SEED)
    sha = F.sha256(left)
    denom = {"in_all": {key}, "dup_keys": set(), "delivery_sha": {key: sha}, "frozen_sha": {key: sha}}
    ok = {"success": True, "path": "x.h5", "sha256": sha}
    return hp, key, m, denom, ok


def test_五终态_正例与每条结构不同路径(term):
    hp, key, m, denom, ok = term
    fail = {"success": False, "path": None, "sha256": None}
    assert hp.classify_terminal(key, ok, dict(ok), m, denom, True, False) == "byte_equal"
    assert hp.classify_terminal(key, ok, dict(ok, sha256="0" * 64), m, denom, True, False) == "noise"
    assert hp.classify_terminal(key, ok, fail, m, denom, True, False) == "h2_fail"
    assert hp.classify_terminal(key, fail, ok, m, denom, True, False) == "flipped"
    assert hp.classify_terminal(key, fail, fail, m, denom, True, False) == "structural"
    other = ("T", "xhard1", SEED + 1)
    assert hp.classify_terminal(other, ok, ok, m, denom, True, False) == "structural"       # 不在四方交集
    assert hp.classify_terminal(key, ok, ok, m, dict(denom, dup_keys={key}), True, False) == "structural"
    assert hp.classify_terminal(key, ok, ok, m, denom, False, False) == "structural"       # 包归属不符
    assert hp.classify_terminal(key, ok, ok, m, denom, True, True) == "structural"         # 恢复模式不符
    bad = dict(denom, delivery_sha={key: "1" * 64})
    assert hp.classify_terminal(key, ok, ok, m, bad, True, False) == "structural"          # gen1 与交付清单不符
    for broken in ({"error": "x"}, {"setup_equal": False}, {"schema_equal": False}, {"contiguous": False}):
        assert hp.classify_terminal(key, ok, dict(ok, sha256="0" * 64), dict(m, **broken), denom, True, False) \
            == "structural", broken
