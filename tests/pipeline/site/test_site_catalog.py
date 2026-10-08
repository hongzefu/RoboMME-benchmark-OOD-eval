"""C17 站点目录、逐段 subgoal 与语义差异：真实 ``site_catalog.main／build_catalog``、``subgoal_lengths.main／read_episode／label``、
``semantic_diff.build／main``。

- 规格：包内真实 V9 规格切出的小子表（StopCube／PickXtimes 的 xhard1 与 MoveCube 的 xhard4，整格配额）进门禁；
  59 格全量（43 个新值格 + 16 个 xhard0 格）标 ``slow``。配置维度对的是真实交付行的取值，负例在真实行上改一个值再重签。
- 身份表独立给出（新值局取自规格的交付行、xhard0 每任务 12 个合成 seed），目录必须与之逐一对应。
- 逐段：合成局逐帧 subgoal 为 a×3、b×2、a′×2（前 2 帧演示），期望段、模板首次／后续与执行步手算。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from site_world import CAT, DEMO, H, SUBGOALS, catalog_inputs, write_h5
from tests._support.loaders import load_script

SG = load_script("injection-dev/site/subgoal_lengths.py")
SEM = load_script("injection-dev/site/semantic_diff.py")
SMALL = {("StopCube", "xhard1"): H.V9_CELLS[("StopCube", "xhard1")],
         ("PickXtimes", "xhard1"): H.V9_CELLS[("PickXtimes", "xhard1")],
         ("MoveCube", "xhard4"): H.V9_CELLS[("MoveCube", "xhard4")]}
N_X0 = len(H.ALL_TASKS) * H.XHARD0_PER_TASK
EXEC = len(SUBGOALS) - DEMO
#: xhard0 的执行步上限（手写钉值；按档步数查表已从 hard_specs 删除）
XHARD0_STEP_CAP = 1300


def _argv(src: dict, out: Path, *extra) -> list[str]:
    argv = ["--out", str(out)]
    for key in ("specs_root", "delivery", "identities", "xhard0_gen", "path_base"):
        argv += [f"--{key.replace('_', '-')}", str(src[key])]
    if src.get("cells_json") is not None:
        argv += ["--cells-json", str(src["cells_json"])]
    return [*argv, *extra]


def _line(capsys, name="V8_SITE_CATALOG") -> dict[str, str]:
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith(f"{name}=")][-1]
    head, *rest = line.split()
    return {"verdict": head.split("=", 1)[1], **dict(p.split("=", 1) for p in rest)}


@pytest.fixture
def small(tmp_path):
    return catalog_inputs(tmp_path / "in", SMALL)


def _check_catalog(src: dict, out: Path, kv: dict, cells: dict):
    n_new = sum(cells.values())
    assert kv["verdict"] == "PASS" and kv["identities"] == kv["expected"] == str(n_new + N_X0)
    assert (kv["gen_v8"], kv["gen_xhard0_new"], kv["gen_xhard0_old"]) == (str(n_new), str(N_X0), str(N_X0))
    assert (kv["config_mismatch"], kv["problems"], kv["eval_unevaluated"]) == ("0", "0", str(n_new + N_X0))
    cat = json.loads((out / "catalog.json").read_text())
    media = json.loads((out / "media-private.json").read_text())
    assert cat["schema"] == CAT.SITE_CATALOG_SCHEMA and cat["eval"]["status"] == "unevaluated"
    assert [t["id"] for t in cat["tasks"]] == list(CAT.NAMES)
    ids = [json.loads(x) for x in Path(src["identities"]).read_text().splitlines()]
    got = {(tier, t["id"], ep["seed"]) for t in cat["tasks"] for tier, cell in t["tiers"].items()
           for ep in cell["episodes"]}
    assert got == {(r["tier"], r["task"], r["seed"]) for r in ids}
    assert {(t["id"], tier) for t in cat["tasks"] for tier in t["tiers"] if tier != "xhard0"} == set(cells)
    for t in cat["tasks"]:
        for tier, cell in t["tiers"].items():
            assert [ep["idx"] for ep in cell["episodes"]] == list(range(1, len(cell["episodes"]) + 1))
            assert cell["eval_status"] == "unevaluated"
            for ep in cell["episodes"]:
                assert ep["eval_status"] == "unevaluated" and ep["eval"] == {}
                gen = ep["gen"]
                if tier == "xhard0":
                    assert set(gen) == {"new", "old"} and gen["new"]["demo_frames"] == DEMO
                else:
                    assert (gen["new"]["frames"], gen["new"]["exec_steps"], gen["new"]["demo_frames"]) == \
                        (len(SUBGOALS), EXEC, DEMO)
                    assert gen["new"]["media"] in media
    assert all(Path(p).is_file() and p.endswith(".mp4") for p in media.values())
    return cat


def test_small_catalog_matches_independent_identity_table(small, tmp_path, capsys):
    out = tmp_path / "site"
    assert CAT.main(_argv(small, out)) == 0
    cat = _check_catalog(small, out, _line(capsys), SMALL)
    pick = next(t for t in cat["tasks"] if t["id"] == "PickXtimes")["tiers"]["xhard1"]["config"]
    assert [c["dim"] for c in pick] == [d for d, _ in CAT.DIMS["PickXtimes"]]
    assert all(len(c["values"]) == 1 for c in pick)  # 定值维度：整格只有一个取值
    move = next(t for t in cat["tasks"] if t["id"] == "MoveCube")["tiers"]["xhard4"]["config"]
    assert move == [{"dim": None, "values": [], "text": CAT.NO_DIM_TEXT}]


@pytest.mark.slow
def test_full_59_cells_catalog(tmp_path, capsys):
    src = catalog_inputs(tmp_path / "in", dict(H.V9_CELLS))
    src["cells_json"] = None  # 走 --cells v9 完整表
    out = tmp_path / "site"
    assert CAT.main(_argv(src, out)) == 0
    cat = _check_catalog(src, out, _line(capsys), dict(H.V9_CELLS))
    assert sum(len(t["tiers"]) for t in cat["tasks"]) == len(H.V9_CELLS) + len(H.ALL_TASKS)


def test_config_mismatch_on_edited_real_row(tmp_path, capsys):
    def edit(tier, rows):
        if tier == "xhard1":
            row = next(r for r in rows if r["task"] == "PickXtimes" and r["selected"])
            row["spec"]["objects"]["num_repeats"] += 1

    src = catalog_inputs(tmp_path / "in", SMALL, edit=edit)
    out = tmp_path / "site"
    assert CAT.main(_argv(src, out)) == 1
    kv = _line(capsys)
    assert kv["verdict"] == "FAIL" and kv["config_mismatch"] == "1"
    assert not (out / "catalog.json").exists()  # FAIL 不写产物


def _drop(rows, pred):
    hit = next(i for i, r in enumerate(rows) if pred(r))
    return rows[:hit] + rows[hit + 1:]


@pytest.mark.parametrize("break_it, needle", [
    ("drop_identity", "身份清单新值局与 delivery 不一致"),
    ("drop_x0", "xhard0 身份应为"),
    ("dup_identity", "不唯一"),
    ("dup_delivery", "delivery 身份重复"),
    ("no_exec_steps", "delivery 行缺"),
    ("drop_x0_gen", "xhard0 O 缺生成记录"),
])
def test_identity_and_delivery_negatives(small, break_it, needle):
    ids_path, dpath = Path(small["identities"]), Path(small["delivery"])
    ids = [json.loads(x) for x in ids_path.read_text().splitlines()]
    delivery = json.loads(dpath.read_text())
    if break_it == "drop_identity":
        ids = _drop(ids, lambda r: r["tier"] == "xhard1")
    elif break_it == "drop_x0":
        ids = _drop(ids, lambda r: r["tier"] == "xhard0" and r["task"] == "BinFill")
    elif break_it == "dup_identity":
        ids.append(dict(ids[0]))
    elif break_it == "dup_delivery":
        delivery["rows"].append(dict(delivery["rows"][0]))
    elif break_it == "no_exec_steps":
        delivery["rows"][0].pop("exec_steps")
    elif break_it == "drop_x0_gen":
        man = Path(small["xhard0_gen"]) / "manifest-O.jsonl"
        man.write_text("".join(man.read_text().splitlines(keepends=True)[1:]))
    ids_path.write_text("".join(json.dumps(r) + "\n" for r in ids))
    dpath.write_text(json.dumps(delivery))
    _, _, stats = CAT.build_catalog(small)
    assert any(needle in p for p in stats["problems"]), stats["problems"][:5]


def test_positive_counterpart_has_no_problems(small):
    _, media, stats = CAT.build_catalog(small)
    assert stats["problems"] == [] and stats["identities"] == stats["expected"]


def test_refuses_to_overwrite_existing_outputs(small, tmp_path, capsys):
    out = tmp_path / "site"
    assert CAT.main(_argv(small, out)) == 0
    before = (out / "catalog.json").read_bytes()
    capsys.readouterr()
    assert CAT.main(_argv(small, out)) == 1
    assert (out / "catalog.json").read_bytes() == before


def test_v9_eval_mode_needs_reused(small):
    with pytest.raises(ValueError, match="同时给"):
        CAT.build_catalog(dict(small, eval_reuse=Path("/x")))


def test_pick_gen_video_rules(tmp_path):
    ep = tmp_path / "ep"
    (ep / "hdf5_files").mkdir(parents=True)
    (ep / "videos").mkdir()
    h5 = ep / "hdf5_files" / "a.h5"
    row = {"seed": 7, "task": "StopCube", "tier": "xhard1", "episode": 0}
    for name in ("FAILED_x_seed7_a.mp4", "success_NO_OBJECT_seed7_b.mp4", "ok_seed77_c.mp4"):
        (ep / "videos" / name).write_bytes(b"x")
    with pytest.raises(ValueError, match="找不到生成视频"):
        CAT.pick_gen_video(h5, row, None)
    (ep / "videos" / "ok_seed7_d.mp4").write_bytes(b"x")
    assert CAT.pick_gen_video(h5, row, None).name == "ok_seed7_d.mp4"
    (ep / "videos" / "again_seed7_e.mp4").write_bytes(b"x")
    with pytest.raises(ValueError, match="有 2 个"):
        CAT.pick_gen_video(h5, row, None)


def test_table1_ok_handles_points_and_ranges():
    lo, hi = CAT.TABLE1["RouteStick"]["路线段数"]["xhard1"]
    assert CAT.table1_ok("RouteStick", "路线段数", "xhard1", lo)
    assert CAT.table1_ok("RouteStick", "路线段数", "xhard1", hi)
    assert not CAT.table1_ok("RouteStick", "路线段数", "xhard1", hi + 1)
    assert not CAT.table1_ok("RouteStick", "路线段数", "xhard1", lo - 1)
    assert not CAT.table1_ok("RouteStick", "路线段数", "xhard1", None)
    assert not CAT.table1_ok("MoveCube", "x", "xhard4", 1)


# ── 逐段 subgoal：连续段与演示过滤 ─────────────────────────────────────


def test_read_episode_segments_and_demo(tmp_path):
    h5 = write_h5(tmp_path / "a.h5", ["pick the red cube"] * 2 + ["press"] + ["pick the red cube", "pick the blue cube"],
                  demo=2, goal=["goal one", "goal two"])
    rec = SG.read_episode(str(h5))
    assert rec == {"segs": [["pick the red cube", 2], ["press", 1], ["pick the red cube", 1], ["pick the blue cube", 1]],
                   "frames": 5, "demo": 2, "goal": ["goal one", "goal two"], "difficulty": "t7"}
    assert [(s["tpl"], s["kind"], s["frames"]) for s in SG.label(rec["segs"])] == [
        ("pick the 〈颜色〉 cube", "首次", 2), ("press", "首次", 1), ("pick the 〈颜色〉 cube", "后续", 1),
        ("pick the 〈颜色〉 cube", "后续", 1)]
    empty = tmp_path / "empty.h5"
    import h5py
    h5py.File(empty, "w").close()
    assert SG.read_episode(str(empty)) == {"segs": [], "frames": 0, "demo": 0, "goal": [], "difficulty": None}


def _subgoals(src, out, capsys):
    rc = SG.main(["--site-dir", str(out), "--specs-root", str(src["specs_root"]), "--delivery", str(src["delivery"]),
                  "--xhard0-gen", str(src["xhard0_gen"]), "--path-base", str(src["path_base"]), "--workers", "1"])
    return rc, _line(capsys, "V8_SUBGOALS")


def test_subgoal_lengths_end_to_end(small, tmp_path, capsys):
    out = tmp_path / "site"
    assert CAT.main(_argv(small, out)) == 0
    capsys.readouterr()
    rc, kv = _subgoals(small, out, capsys)
    assert rc == 0 and kv["verdict"] == "PASS" and kv["exec_mismatch"] == "0" and kv["missing"] == "0"
    assert kv["xhard0_same"] == f"{N_X0}/{N_X0}"
    sg = json.loads((out / "subgoals.json").read_text())
    stop = sg["oracle"]["StopCube"]["xhard1"]
    assert (stop["n"], stop["mean"], stop["min"], stop["max"]) == (SMALL[("StopCube", "xhard1")], EXEC, EXEC, EXEC)
    assert (stop["demo_min"], stop["total_max"], stop["max_steps"]) == (DEMO, len(SUBGOALS), H.EXEC_CAP)
    assert sg["oracle"]["BinFill"]["xhard0"]["max_steps"] == XHARD0_STEP_CAP
    ep = sg["episodes"]["StopCube"]["xhard1"]["1"]["new"]
    assert [(s["text"], s["frames"], s["kind"]) for s in ep] == [
        ("pick up the red cube", 3, "首次"), ("press the button", 2, "首次"), ("pick up the blue cube", 2, "后续")]


def test_subgoal_exec_mismatch_fails_without_output(small, tmp_path, capsys):
    out = tmp_path / "site"
    assert CAT.main(_argv(small, out)) == 0
    capsys.readouterr()
    dpath = Path(small["delivery"])
    delivery = json.loads(dpath.read_text())
    delivery["rows"][0]["exec_steps"] += 1
    dpath.write_text(json.dumps(delivery))
    rc, kv = _subgoals(small, out, capsys)
    assert rc == 1 and kv["verdict"] == "FAIL" and kv["exec_mismatch"] == "1"
    assert not (out / "subgoals.json").exists()


# ── 语义差异：只比句式不比参数 ─────────────────────────────────────────


def _ep(goal, subs):
    return {"goal": goal, "new": [{"text": t, "tpl": tpl, "kind": kind, "frames": 1} for t, tpl, kind in subs]}


X0_GOAL = "pick up the red cube, then pick up the container hiding the blue cube"
PICK = ("pick up the container", "pick up the container", "首次")
PUT = ("put down the container", "put down the container", "首次")
SG_DOC = {"episodes": {
    "VideoUnmask": {
        "xhard0": {"1": _ep([X0_GOAL], [PICK, PUT])},
        # 只换颜色：句式相同 → 仅参数变化
        "xhard1": {"1": _ep(["pick up the green cube, then pick up the container hiding the red cube"], [PICK, PUT])},
        # goal 多一句、「put down」多重复一次 → 有调整；重复只记 sub_repeat_only
        "xhard2": {"1": _ep([X0_GOAL + ", next pick up another container hiding the green cube"],
                            [PICK, PUT, ("put down the container", "put down the container", "后续")])},
    },
    "StopCube": {
        "xhard0": {"1": _ep(["press the button when the cube stops"], [("wait", "wait", "首次")])},
        # 新增 subgoal 类型，但 NOTES 没有人工说明 → note_missing
        "xhard1": {"1": _ep(["press the button when the cube stops"], [("wait", "wait", "首次"),
                                                                       ("tap twice", "tap twice", "首次")])},
    },
}}


def test_semantic_labels_follow_real_differences():
    out, stats = SEM.build(SG_DOC)
    vu = out["tasks"]["VideoUnmask"]["tiers"]
    assert (vu["xhard1"]["changed"], vu["xhard1"]["label"]) == (False, "仅参数变化")
    assert vu["xhard2"]["changed"] and vu["xhard2"]["label"] == "有语义调整"
    assert [g["example"] for g in vu["xhard2"]["goal_added"]] == [SG_DOC["episodes"]["VideoUnmask"]["xhard2"]["1"]["goal"][0]]
    assert vu["xhard2"]["sub_added"] == [] and vu["xhard2"]["sub_repeat_only"] == [
        {"tpl": "put down the container", "kind": "后续", "segments": 1}]
    assert vu["xhard2"]["note"] == SEM.NOTES[("VideoUnmask", "xhard2")]
    assert out["episodes"]["VideoUnmask"]["xhard2"]["1"] == {"goal": [True], "sub": [False, False, False]}
    sc = out["tasks"]["StopCube"]["tiers"]["xhard1"]
    assert sc["changed"] and [s["tpl"] for s in sc["sub_added"]] == ["tap twice"] and sc["note"] is None
    assert sc["sub_pairs"] == [{"before": None, "after": {"tpl": "tap twice", "example": "tap twice"}}]
    assert out["episodes"]["StopCube"]["xhard1"]["1"]["sub"] == [False, True]
    assert stats == {"cells": 3, "changed": 2, "param_only": 1, "note_missing": 1, "tasks": 2}


def test_semantic_main_fails_on_missing_note(tmp_path, capsys):
    (tmp_path / "subgoals.json").write_text(json.dumps(SG_DOC))
    assert SEM.main(["--site-dir", str(tmp_path)]) == 1
    assert _line(capsys, "V8_SEMANTIC")["note_missing"] == "1"
    doc = {"episodes": {"VideoUnmask": SG_DOC["episodes"]["VideoUnmask"]}}
    (tmp_path / "subgoals.json").write_text(json.dumps(doc))
    assert SEM.main(["--site-dir", str(tmp_path)]) == 0
    # semantic_diff 没有单独的 schema 常量：以生产 build 产出的 schema 为准（不在测试里写字面值）
    assert json.loads((tmp_path / "semantic.json").read_text())["schema"] == SEM.build({"episodes": {}})[0]["schema"]


def test_norm_goal_erases_parameters_only():
    assert SEM.norm_goal("Pick up TWO red cubes 3 times") == SEM.norm_goal("pick up one blue cube 2 times")
    assert SEM.norm_goal("pick up the cube") != SEM.norm_goal("put down the cube")
