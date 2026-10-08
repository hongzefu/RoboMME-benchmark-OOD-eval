"""C15 ``hard_regression.py env-digest-compare``（ENV_DIGEST_PARITY，只报告）的无仿真夹具：合成 rows.jsonl + npz，
覆盖全同、仅改名、同键互换值（真差异）、演示帧差、帧数差、reset 初始帧走图像路径、状态值差、键集合差、
量不出来（形状不符、npz 缺失）、行数不齐、错行与半行、常驻模式拒绝续跑。纯 CPU，不建环境。

逐层摘要由生产写入侧的 ``_env_digest``／``_env_json_digest``／``_env_layer`` 生成（它们是夹具的「写者」，
被测的是读者 ``cmd_env_digest_compare``）；量值期望由原始数组手算。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import parity_fixtures as F

R = F.hard_regression()


def _layers_from(arrays: dict[str, np.ndarray], identity: dict) -> dict:
    """按工具同一口径由原始数组算逐层摘要。"""
    keys: dict[str, dict[str, str]] = {name: {} for name in R.ENV_DIGEST_LAYERS}
    for full, value in arrays.items():
        layer, key = full.split("/", 1)
        keys[layer][key] = R._env_digest(value)
    keys["identity"] = {k: R._env_json_digest(v) for k, v in identity.items()}
    return {name: R._env_layer(k) for name, k in keys.items()}


def _base_arrays(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "pre_demo_state/actors/cube_0": rng.normal(size=13).astype(np.float32),
        "pre_demo_state/actors/button_left": rng.normal(size=13).astype(np.float32),
        "pre_demo_state/articulations/panda": rng.normal(size=31).astype(np.float32),
        "demo_frames/front": rng.integers(0, 256, (5, 8, 8, 3), dtype=np.uint8),
        "demo_frames/wrist": rng.integers(0, 256, (5, 8, 8, 3), dtype=np.uint8),
        "reset_obs/front_init": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
        "reset_obs/joint_state_list": rng.normal(size=(6, 7)),
        "post_demo_state/actors/cube_0": rng.normal(size=13).astype(np.float32),
        "post_demo_state/actors/button_left": rng.normal(size=13).astype(np.float32),
        "step_frames/front": rng.integers(0, 256, (3, 8, 8, 3), dtype=np.uint8),
        "step_obs/joint_state_list": rng.normal(size=(3, 7)),
        "step_state/actors/cube_0": rng.normal(size=(3, 13)).astype(np.float32),
        "step_status/action": np.zeros(8),
    }


def _write_cell(root: Path, name: str, idents: list[dict]) -> Path:
    """idents: [{task, source_episode, seed, builder_episode, arrays, identity}] → <root>/<name>/rows.jsonl + arrays/*.npz。"""
    cell = root / name
    (cell / "arrays").mkdir(parents=True)
    with (cell / "rows.jsonl").open("w") as stream:
        for index, ident in enumerate(idents):
            npz = f"arrays/{ident['task']}-ep{ident['source_episode']}.npz"
            np.savez_compressed(cell / npz, **ident["arrays"])
            row = {k: ident[k] for k in ("task", "source_episode", "seed", "builder_episode")}
            row.update({"order_index": index, "error": None, "npz": npz,
                        "layers": _layers_from(ident["arrays"], ident["identity"])})
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return cell


def _ident(ep: int, arrays: dict | None = None, identity: dict | None = None) -> dict:
    return {"task": "PickXtimes", "source_episode": ep, "seed": 510000 + ep * 100, "builder_episode": (ep - 3) // 4,
            "arrays": arrays if arrays is not None else _base_arrays(ep),
            "identity": identity or {"seed": 510000 + ep * 100, "tier": "xhard0", "task_goal": ["pick"]}}


def _run(tmp_path: Path, a: Path, b: Path, capsys) -> tuple[dict, str]:
    out = tmp_path / "cmp.json"
    args = type("A", (), {"a": str(a), "b": str(b), "out": str(out)})()
    assert R.cmd_env_digest_compare(args) == 0
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("ENV_DIGEST_PARITY ")]
    assert len(line) == 1
    return json.loads(out.read_text()), line[0]


def test_all_equal(tmp_path, capsys):
    a = _write_cell(tmp_path, "a", [_ident(3), _ident(7)])
    b = _write_cell(tmp_path, "b", [_ident(3), _ident(7)])
    summary, line = _run(tmp_path, a, b, capsys)
    assert summary["compared"] == 2 and summary["first_diff"] == "-"
    assert all(v == 2 for v in summary["layer_equal"].values())
    # 全同时摘要相等、不读数组：量值为 n/a 而不是假 0
    assert "first_diff=- " in line and "state_max_abs=n/a " in line and "image_mad=n/a " in line
    assert "unmeasured=0" in line and "npz_missing=0" in line
    assert "layer_equal=identity:2,pre_demo_state:2,demo_frames:2" in line


def test_rename_only(tmp_path, capsys):
    arrays = _base_arrays(3)
    renamed = dict(arrays)
    renamed["pre_demo_state/actors/button_right"] = renamed.pop("pre_demo_state/actors/button_left")
    a = _write_cell(tmp_path, "a", [_ident(3, arrays)])
    b = _write_cell(tmp_path, "b", [_ident(3, renamed)])
    summary, line = _run(tmp_path, a, b, capsys)
    layer = summary["details"][0]["layers"]["pre_demo_state"]
    assert layer["name_only"] and layer["key_set_diff"]
    assert summary["first_diff"] == "-" and summary["name_only"]["pre_demo_state"] == 1
    assert "name_only=1" in line


def test_demo_frame_diff(tmp_path, capsys):
    """逐帧 MAD：只改第 2 帧一个像素 → 最大逐帧 MAD = 差值 / 单帧元素数，第一个不等帧 = 2，不等帧数 = 1。"""
    arrays = _base_arrays(3)
    changed = dict(arrays)
    frames = arrays["demo_frames/front"].copy()
    frames[2, 0, 0, 0] = (int(frames[2, 0, 0, 0]) + 51) % 256
    changed["demo_frames/front"] = frames
    a = _write_cell(tmp_path, "a", [_ident(3, arrays)])
    b = _write_cell(tmp_path, "b", [_ident(3, changed)])
    summary, line = _run(tmp_path, a, b, capsys)
    assert summary["first_diff"] == "demo_frames"
    delta = abs(int(frames[2, 0, 0, 0]) - int(arrays["demo_frames/front"][2, 0, 0, 0]))
    per_frame = delta / frames[0].size
    assert summary["image_mad"] == pytest.approx(per_frame)
    assert summary["image_mad_unit"] == pytest.approx(per_frame / 255)
    assert summary["image_mad_mean"] == pytest.approx(per_frame / len(frames))
    assert summary["image_diff_frames"] == 1 and summary["image_first_diff_frame"] == "demo_frames/front@2"
    assert summary["layer_equal"]["demo_frames"] == 0 and summary["layer_equal"]["pre_demo_state"] == 1
    assert summary["state_max_abs"] is None
    assert "first_diff=demo_frames" in line and "image_first_diff_frame=demo_frames/front@2" in line


def test_demo_frame_count_diff(tmp_path, capsys):
    """演示帧数不同：只比公共前缀，记两边帧数。"""
    arrays = _base_arrays(3)
    short = dict(arrays)
    short["demo_frames/front"] = arrays["demo_frames/front"][:3]
    a = _write_cell(tmp_path, "a", [_ident(3, arrays)])
    b = _write_cell(tmp_path, "b", [_ident(3, short)])
    summary, _ = _run(tmp_path, a, b, capsys)
    img = summary["details"][0]["layers"]["demo_frames"]["image"]["keys"]["front"]
    assert img["frames"] == [5, 3] and img["diff_frames"] == 0 and img["per_frame_mad_max"] == 0.0


def test_reset_init_frame_uses_image_path(tmp_path, capsys):
    """reset_obs 的 uint8 初始帧走图像路径（逐帧 MAD），不进 obs_max_abs。"""
    arrays = _base_arrays(3)
    changed = dict(arrays)
    init = arrays["reset_obs/front_init"].copy()
    init[...] = 255 - init
    changed["reset_obs/front_init"] = init
    a = _write_cell(tmp_path, "a", [_ident(3, arrays)])
    b = _write_cell(tmp_path, "b", [_ident(3, changed)])
    summary, _ = _run(tmp_path, a, b, capsys)
    expected = float(np.abs(init.astype(np.int16) - arrays["reset_obs/front_init"].astype(np.int16)).mean())
    assert summary["first_diff"] == "reset_obs"
    assert summary["image_mad"] == pytest.approx(expected) and summary["obs_max_abs"] is None


def test_same_keys_swapped_values_is_real_diff(tmp_path, capsys):
    """两个同形 actor 互换位姿：键集合相同、值多重集相同，但这是真差异，不许判成仅改名。"""
    arrays = _base_arrays(3)
    swapped = dict(arrays)
    swapped["pre_demo_state/actors/cube_0"] = arrays["pre_demo_state/actors/button_left"]
    swapped["pre_demo_state/actors/button_left"] = arrays["pre_demo_state/actors/cube_0"]
    a = _write_cell(tmp_path, "a", [_ident(3, arrays)])
    b = _write_cell(tmp_path, "b", [_ident(3, swapped)])
    summary, line = _run(tmp_path, a, b, capsys)
    layer = summary["details"][0]["layers"]["pre_demo_state"]
    assert not layer.get("name_only") and layer["diff_key_count"] == 2
    assert summary["first_diff"] == "pre_demo_state" and summary["name_only"]["pre_demo_state"] == 0
    expected = float(np.abs(arrays["pre_demo_state/actors/cube_0"].astype(np.float64)
                            - arrays["pre_demo_state/actors/button_left"].astype(np.float64)).max())
    assert summary["state_max_abs"] == pytest.approx(expected)
    assert "name_only=0" in line


def test_unmeasured_shape_mismatch_and_npz_missing(tmp_path, capsys):
    """形状不符、npz 缺失都量不出来：记 n/a 并计数，不填 0。"""
    arrays = _base_arrays(3)
    changed = dict(arrays)
    changed["post_demo_state/actors/cube_0"] = np.zeros(14, dtype=np.float32)
    a = _write_cell(tmp_path, "a", [_ident(3, arrays), _ident(7)])
    b_arrays7 = dict(_base_arrays(7))
    b_arrays7["step_state/actors/cube_0"] = b_arrays7["step_state/actors/cube_0"] + 1
    b = _write_cell(tmp_path, "b", [_ident(3, changed), _ident(7, b_arrays7)])
    (b / "arrays" / "PickXtimes-ep7.npz").unlink()
    summary, line = _run(tmp_path, a, b, capsys)
    assert summary["state_max_abs"] is None and summary["npz_missing"] == 1 and summary["unmeasured"] == 2
    assert "state_max_abs=n/a" in line and "npz_missing=1" in line and "unmeasured=2" in line


def test_resident_resume_refused(tmp_path, capsys):
    """--resident 且 rows.jsonl 已有行：默认拒绝续跑（返回 2、不起子进程）。"""
    idents = tmp_path / "ids.json"
    idents.write_text(json.dumps([{"task": "PickXtimes", "source_episode": 3, "seed": 510300, "builder_episode": 0},
                                  {"task": "PickXtimes", "source_episode": 7, "seed": 510700, "builder_episode": 1}]))
    _write_cell(tmp_path, "cell", [_ident(3)])
    args = type("A", (), {"identities": str(idents), "out": str(tmp_path), "cell": "cell", "resident": True,
                          "reverse": False, "limit": 0, "gpu": None, "fixed_steps": 30,
                          "allow_resume_resident": False})()
    assert R.cmd_env_digest(args) == 2
    out = capsys.readouterr().out
    assert "ENV_DIGEST_RESUME_REFUSED" in out and "resume_generation=1" in out
    assert R._env_resume_generation(tmp_path / "cell") == 1 and R._env_resume_generation(tmp_path / "none") == 0


def test_state_value_diff(tmp_path, capsys):
    arrays = _base_arrays(3)
    changed = dict(arrays)
    changed["post_demo_state/actors/cube_0"] = arrays["post_demo_state/actors/cube_0"] + np.float32(0.25)
    a = _write_cell(tmp_path, "a", [_ident(3, arrays)])
    b = _write_cell(tmp_path, "b", [_ident(3, changed)])
    summary, _ = _run(tmp_path, a, b, capsys)
    assert summary["first_diff"] == "post_demo_state"
    assert summary["state_max_abs"] == pytest.approx(0.25, rel=1e-5)


def test_key_set_diff(tmp_path, capsys):
    arrays = _base_arrays(3)
    extra = dict(arrays)
    extra["step_obs/eef_state_list"] = np.ones((3, 6))
    a = _write_cell(tmp_path, "a", [_ident(3, arrays)])
    b = _write_cell(tmp_path, "b", [_ident(3, extra)])
    summary, line = _run(tmp_path, a, b, capsys)
    layer = summary["details"][0]["layers"]["step_obs"]
    assert layer["key_set_diff"] and layer["only_b"] == ["eef_state_list"] and not layer.get("name_only")
    assert summary["first_diff"] == "step_obs" and summary["key_set_diff"] == 1
    assert "key_set_diff=1" in line


def test_row_count_mismatch(tmp_path, capsys):
    a = _write_cell(tmp_path, "a", [_ident(3), _ident(7), _ident(11)])
    b = _write_cell(tmp_path, "b", [_ident(7)])
    summary, line = _run(tmp_path, a, b, capsys)
    assert summary["compared"] == 1 and summary["rows_a"] == 3 and summary["rows_b"] == 1
    assert len(summary["missing_in_b"]) == 2 and summary["missing_in_a"] == []
    assert "compared=1" in line and "missing_in_b=2" in line


def test_error_and_half_rows_ignored(tmp_path, capsys):
    """有错的行与被杀进程留下的半行不算完成（续跑会重做）。"""
    a = _write_cell(tmp_path, "a", [_ident(3)])
    with (a / "rows.jsonl").open("a") as stream:
        stream.write(json.dumps({"task": "PickXtimes", "source_episode": 7, "seed": 510700, "builder_episode": 1,
                                 "error": "RuntimeError: x"}) + "\n")
        stream.write('{"task": "PickX')
    assert list(R._env_rows(a)) == ["PickXtimes/ep3/seed510300/b0"]
