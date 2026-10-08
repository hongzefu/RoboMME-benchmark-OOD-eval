"""C12 第一阶段：抽签循环 → 封存（``hard-specs/4``）→ 排他落盘，以及 ``freeze_specs.py --dry-run`` 的预算打印。

真实调用 ``_draw.draw_task``／``merge_task_rows``、``freeze_specs.draw_rows_by_task``／``plan_v8``／``main``、
``_freeze.freeze``／``stratified_select``／``write_jsonl_exclusive``、``hard_specs.load_specs``。抽签只替换
``draw_one`` 注入点（CPU 替身，不 reset）。期望值手写：哪一局失败、选中哪些候选、每方式取哪几号。
"""
from __future__ import annotations

import concurrent.futures
import json
import sys

import pytest

from gen_world import D, F, FS, H, R, FakeDraw, draw, fake_spec, freeze_file, header_parts

MC = ("MoveCube", "xhard4")


# ── 抽签循环与封存 ──────────────────────────────────────────────────


def test_draw_then_freeze_signs_v4_file(tmp_path):
    # StopCube xhard1：配额 2、候选 4；候选 1 的第一次 reset 被拒（attempt 0 失败、attempt 1 成功）
    path = tmp_path / "xhard1" / "specs.jsonl"
    drafts, fake = draw("xhard1", {"StopCube": (2, 4)}, fail={("StopCube", 1): 1})
    assert [(c[2], c[3]) for c in fake.calls] == [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0)]
    header, rows = F.freeze(drafts, header_parts("xhard1", ["StopCube"], drafts), {"StopCube": (0, 1)},
                            {"StopCube": 4}, schema=H.SCHEMA)
    F.write_jsonl_exclusive(path, [header, *rows])
    got_header, got_rows = H.load_specs(path, check_fingerprint=False)
    assert got_header == header and got_rows == rows
    assert header["schema"] == H.SCHEMA and header["layout_rule"] == {"mode": "independent"}
    assert header["exec_cap"] == H.EXEC_CAP and header["delivery_per_cell"] == {"StopCube": 2}
    assert header["per_env"] == {"StopCube": 4} and header["select_rule"] == {"StopCube": [0, 1]}
    per_env = header["draw_stats"]["freeze_per_env"]["StopCube"]
    assert per_env == {"attempted": 5, "candidates": 4, "initial_selected": [0, 1], "candidates_requested": 4}
    # 行：candidate == episode，seed 是替身真正收到的那次成功尝试的 seed（不按公式重算）
    ok_seed = {(c[2]): c[1] for c in fake.calls if not (c[2] == 1 and c[3] == 0)}
    assert [(r["candidate"], r["attempt"], r["seed"], r["selected"], r["tried"], r["rollout"], r["layout_parent"])
            for r in rows] == [(0, 0, ok_seed[0], True, False, None, None), (1, 1, ok_seed[1], True, False, None, None),
                               (2, 0, ok_seed[2], False, False, None, None), (3, 0, ok_seed[3], False, False, None, None)]
    assert len({r["seed"] for r in rows}) == 4


def test_draw_stops_at_reset_budget_and_records_shortfall():
    # 候选 3、reset 预算 4：候选 1 一直被拒 → 4 次尝试后停止，只攒到 1 个成功候选
    fake = FakeDraw("xhard1", fail={("StopCube", 1): 99})
    rows = D.draw_task("StopCube", {}, 3, 4, fake, "xhard1", H.seed_rule_for("xhard1", "v8"))
    assert [(r["episode"], r["attempt"], r["reset_ok"]) for r in rows] == [(0, 0, True), (1, 0, False), (1, 1, False),
                                                                          (1, 2, False)]
    stats = D.draw_stats(rows)
    assert stats["per_task"]["StopCube"] == {"attempted": 4, "ok": 1, "fail_class": {"ResetRejected": 3}}
    assert stats["attempted"] == 4 and stats["ok"] == 1


def test_multi_worker_draw_equals_single_worker():
    # 多 worker 路径（executor_factory 注入线程池）与单 worker 行序、内容逐行相同；合并按任务序
    tasks = ["StopCube", "SwingXtimes"]
    fail = {("SwingXtimes", 0): 2}
    kw = dict(difficulty="xhard1", seed_rule=H.seed_rule_for("xhard1", "v8"))
    one, s1 = FS.draw_rows_by_task(tasks, {t: {} for t in tasks}, {t: 2 for t in tasks}, {t: 9 for t in tasks}, 1,
                                   draw_one=FakeDraw("xhard1", fail), **kw)
    many, s2 = FS.draw_rows_by_task(tasks, {t: {} for t in tasks}, {t: 2 for t in tasks}, {t: 9 for t in tasks}, 2,
                                    draw_one=FakeDraw("xhard1", fail),
                                    executor_factory=lambda n: concurrent.futures.ThreadPoolExecutor(n), **kw)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]  # noqa: E731
    assert strip(one) == strip(many) and s1 == s2
    assert [r["task"] for r in one] == ["StopCube"] * 2 + ["SwingXtimes"] * 4


def test_merge_rejects_rows_out_of_draw_order():
    rule = H.seed_rule_for("xhard1", "v8")
    rows = D.draw_task("StopCube", {}, 2, 9, FakeDraw("xhard1"), "xhard1", rule)
    with pytest.raises(H.SpecsError, match="行序与抽签规则不符"):
        D.merge_task_rows(["StopCube"], {"StopCube": list(reversed(rows))}, rule)
    with pytest.raises(H.SpecsError, match="缺少环境"):
        D.merge_task_rows(["StopCube", "SwingXtimes"], {"StopCube": rows}, rule)
    assert D.merge_task_rows(["StopCube"], {"StopCube": rows}, rule) == rows


# ── 拒绝分支 ──────────────────────────────────────────────────────


def _good_drafts():
    drafts, _ = draw("xhard1", {"StopCube": (2, 3)})
    return drafts


def _tamper(kind):
    drafts = _good_drafts()
    parts = header_parts("xhard1", ["StopCube"], drafts)
    schema, select = H.SCHEMA, {"StopCube": (0, 1)}
    if kind == "schema":
        schema = "hard-specs/3"
    elif kind == "layout_rule":
        parts["layout_rule"] = {"mode": "derived"}
    elif kind == "tier":
        parts["difficulty"] = "xhard0"
    elif kind == "spec_sha":
        drafts[1]["spec"] = dict(drafts[1]["spec"], objects={"probe": -1})
    elif kind == "draft_tier":
        drafts[0]["difficulty"] = "xhard2"
    elif kind == "seed":
        drafts[2]["seed"] += 1
    elif kind == "gap":
        del drafts[1]
    elif kind == "quota":
        select = {"StopCube": (0, 1, 2, 3)}
    return drafts, parts, schema, select


@pytest.mark.parametrize("kind, needle", [
    ("schema", "未知 schema"), ("layout_rule", "不接受调用方给的 layout_rule"), ("tier", "未知档位"),
    ("spec_sha", "规格散列不符"), ("draft_tier", "档位与 header 不符"), ("seed", "seed 与公式不符"),
    ("gap", "编号不连续"), ("quota", "选不满配额"),
])
def test_freeze_rejects(kind, needle):
    drafts, parts, schema, select = _tamper(kind)
    with pytest.raises(H.SpecsError, match=needle):
        F.freeze(drafts, parts, select, 3, schema=schema)


def test_freeze_accepts_untampered_counterpart():
    # 拒绝分支的正例对照：同一组草稿不改任何东西即通过
    drafts, parts, schema, select = _tamper("none")
    header, rows = F.freeze(drafts, parts, select, 3, schema=schema)
    assert [r["candidate"] for r in rows if r["selected"]] == [0, 1]


def test_write_exclusive_refuses_existing(tmp_path):
    path = tmp_path / "xhard1" / "specs.jsonl"
    freeze_file(path, "xhard1", {"StopCube": (1, 1)})
    before = path.read_bytes()
    with pytest.raises(H.SpecsError, match="禁止覆盖"):
        freeze_file(path, "xhard1", {"StopCube": (1, 2)})
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["specs.jsonl"]  # 不留临时文件


# ── 分层选签 ──────────────────────────────────────────────────────


def _mc_rows(ways: list[int]):
    return [{"episode": e, "spec": fake_spec("MoveCube", "xhard4", e, 0, w)} for e, w in enumerate(ways)]


@pytest.mark.parametrize("ways, select, expected", [
    # 每种方式取编号最小的一个：way1→0、way0→2、way2→3
    ([1, 1, 0, 2, 0, 2], (0, 1, 2), [0, 2, 3]),
    # 没有 way2：way0→0、way1→3，再按 select 补 1
    ([0, 0, 0, 1, 1, 1], (0, 1, 2), [0, 1, 3]),
    # 配额 2 < 方式数：只取前两种方式（way0→1、way1→0）
    ([1, 0, 2], (0, 1), [0, 1]),
])
def test_movecube_stratified_one_per_way(ways, select, expected):
    assert F.stratified_select("MoveCube", "xhard4", _mc_rows(ways), select) == expected


def test_stratified_is_plain_select_for_other_tasks():
    rows = [{"episode": e, "spec": fake_spec("StopCube", "xhard1", e, 0)} for e in range(5)]
    assert F.stratified_select("StopCube", "xhard1", rows, (1, 3)) == [1, 3]
    assert F.stratified_select("StopCube", "xhard1", rows, (1, 3, 4), quota=2) == [1, 3]


def test_movecube_way_reads_last_initialization():
    spec = fake_spec("MoveCube", "xhard4", 0, 0, way=2)  # 构造期 initializations.0 写的是另一个值
    assert F._movecube_way(spec) == 2
    spec["initializations"]["10"] = {"way_idx": 0}  # 序号按整数比较：10 > 1
    assert F._movecube_way(spec) == 0
    assert F._movecube_way({"spec_kind": "x"}) is None


def test_v9_movecube_per_way_quota_through_freeze():
    quota = H.V9_CELLS[MC]
    by_way = F.V9_MOVECUBE_QUOTA_BY_WAY
    drafts, _ = draw("xhard4", {"MoveCube": (quota, 80)}, ways={"MoveCube": lambda e: e % 3})
    header, rows = F.freeze(drafts, header_parts("xhard4", ["MoveCube"], drafts), {"MoveCube": tuple(range(quota))},
                            {"MoveCube": 80}, schema=H.SCHEMA)
    want = sorted(e for w in (0, 1, 2) for e in [x for x in range(80) if x % 3 == w][:by_way[w]])
    assert [r["candidate"] for r in rows if r["selected"]] == want
    assert header["select_rule"] == {"MoveCube": want}
    stats = header["draw_stats"]["freeze_per_env"]["MoveCube"]
    # 80 个候选按 e % 3 分：way0 27 个（0,3,…,78）、way1 27 个、way2 26 个
    assert stats["candidates_by_way"] == {"0": 27, "1": 27, "2": 26}
    assert stats["quota_by_way"] == {str(w): n for w, n in by_way.items()}


def test_v9_movecube_way_short_is_rejected_with_counts():
    quota = H.V9_CELLS[MC]
    by_way = F.V9_MOVECUBE_QUOTA_BY_WAY
    # way2 只有 10 个候选（0..9），其余 70 个在 way0／way1 交替
    drafts, _ = draw("xhard4", {"MoveCube": (quota, 80)}, ways={"MoveCube": lambda e: 2 if e < 10 else e % 2})
    with pytest.raises(H.SpecsError, match=f"way2 候选 10 配额 {by_way[2]} 差 {by_way[2] - 10}"):
        F.freeze(drafts, header_parts("xhard4", ["MoveCube"], drafts), {"MoveCube": tuple(range(quota))}, 80,
                 schema=H.SCHEMA)


@pytest.mark.parametrize("task, ways_quota, quota, needle", [
    ("StopCube", {0: 1, 1: 1, 2: 1}, 3, "只用于 MoveCube"),
    ("MoveCube", {0: 1, 1: 1}, 2, "须为"),
    ("MoveCube", {0: 1, 1: -1, 2: 2}, 2, "须为"),
    ("MoveCube", {0: 1, 1: 1, 2: 1}, 4, "合计 3 ≠ 格配额 4"),
])
def test_quota_by_way_argument_checks(task, ways_quota, quota, needle):
    rows = _mc_rows([0, 1, 2, 0, 1, 2])
    with pytest.raises(H.SpecsError, match=needle):
        F.stratified_select(task, "xhard4", rows, tuple(range(quota)), quota, ways_quota)


# ── freeze_specs.py 计划与 --dry-run ─────────────────────────────────


def _main(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["freeze_specs.py", *argv])
    code = FS.main()
    return code, capsys.readouterr().out


def _cells(out: str) -> dict[str, dict[str, str]]:
    rows = {}
    for line in out.splitlines():
        if line.startswith("FREEZE_CELL "):
            kv = dict(p.split("=", 1) for p in line.split()[1:])
            rows[kv["task"]] = kv
    return rows


def test_dry_run_candidates_default_to_quota(monkeypatch, capsys, tmp_path):
    out_path = tmp_path / "frozen" / "xhard4" / "specs.jsonl"
    code, out = _main(monkeypatch, capsys, "--tier", "xhard4", "--cells", "v9smoke", "--out", str(out_path),
                      "--max-reset-attempts", "7", "--task-max-reset-attempts", "MoveCube@xhard4=3,InsertPeg@xhard1=9",
                      "--dry-run")
    assert code == 0
    cells = _cells(out)
    assert set(cells) == {t for t, tier in R.V9_SMOKE_CELLS if tier == "xhard4"}
    for task, kv in cells.items():
        quota = R.V9_SMOKE_CELLS[(task, "xhard4")]
        assert (int(kv["quota"]), int(kv["candidates"]), int(kv["spare"])) == (quota, quota, 0)  # W4：候选数缺省等于局数
    # --task-max-reset-attempts 带 @档 的条目只对该档生效：InsertPeg@xhard1 不作用于 xhard4
    assert cells["MoveCube"]["reset_cap"] == "3" and cells["InsertPeg"]["reset_cap"] == "7"
    plan = next(l for l in out.splitlines() if l.startswith("FREEZE_PLAN "))
    assert "reset_budget<=10" in plan and f"quota={sum(R.V9_SMOKE_CELLS.values())}" in plan
    assert not out_path.parent.parent.exists()  # dry-run 不写盘、不建目录


def test_dry_run_default_reset_cap_and_way_quota(monkeypatch, capsys, tmp_path):
    code, out = _main(monkeypatch, capsys, "--tier", "xhard4", "--cells", "v9shard1", "--candidates-per-env",
                      "MoveCube=80", "--out", str(tmp_path / "s.jsonl"), "--dry-run")
    assert code == 0
    mc = _cells(out)["MoveCube"]
    # 缺省 reset 上限手算：⌈80 ÷ 0.97 × 1.5⌉ = ⌈123.71⌉ = 124（MoveCube 不在实测接受率表里，取缺省 0.97）
    assert mc["reset_cap"] == "124" and mc["candidates"] == "80"
    assert int(mc["spare"]) == 80 - H.V9_CELLS[MC]
    assert mc["quota_by_way"] == F.format_quota_by_way(F.V9_MOVECUBE_QUOTA_BY_WAY)
    assert mc["select"] == f"0..{H.V9_CELLS[MC] - 1}"


def test_existing_out_refused_even_for_dry_run(monkeypatch, capsys, tmp_path):
    out_path = tmp_path / "specs.jsonl"
    out_path.write_text("占位\n")
    with pytest.raises(SystemExit, match="已存在"):
        _main(monkeypatch, capsys, "--tier", "xhard4", "--cells", "v9smoke", "--out", str(out_path), "--dry-run")
    assert out_path.read_text() == "占位\n"


@pytest.mark.parametrize("kwargs, err, needle", [
    (dict(tasks_arg="StopCube"), H.SpecsError, "不在 xhard4 交付格里"),
    (dict(select_arg="0..3"), H.SpecsError, "≠ 格表配额"),
    (dict(candidates_arg="MoveCube=0"), H.SpecsError, "不足以覆盖选取"),
    (dict(task_reset_arg="StopCube=5"), H.SpecsError, "本档未抽的任务"),
    (dict(candidates_arg="StopCube=5"), H.SpecsError, "逐任务条目"),
])
def test_plan_v8_rejects(kwargs, err, needle):
    base = dict(tasks_arg="all", candidates_arg=None, select_arg="default", max_reset_arg=None, task_reset_arg=None)
    with pytest.raises(err, match=needle):
        FS.plan_v8("xhard4", dict(R.V9_SMOKE_CELLS), **{**base, **kwargs})


def test_plan_v8_positive_counterpart():
    plan = FS.plan_v8("xhard4", dict(R.V9_SMOKE_CELLS), "all", None, "default", 5, None)
    assert plan["candidates"] == plan["quota"] and set(plan["reset_caps"].values()) == {5}
    with pytest.raises(H.SpecsError, match="没有 xhard1 档的格"):
        FS.plan_v8("xhard1", dict(R.V9_SMOKE_CELLS), "all", None, "default", 5, None)


def test_cells_names_reject_removed_v8_tables():
    for name in ("full", "smoke"):
        with pytest.raises(R.RolloutError, match="已删除"):
            R.resolve_cells(name)
    assert R.resolve_cells("v9") == H.V9_CELLS


def test_cells_json_roundtrip_and_bounds(tmp_path):
    path = tmp_path / "cells.json"
    path.write_text(json.dumps({"StopCube@xhard1": 2, "MoveCube@xhard4": 1}))
    cells = R.resolve_cells(path)
    assert cells == {("StopCube", "xhard1"): 2, ("MoveCube", "xhard4"): 1}
    assert list(cells) == [("StopCube", "xhard1"), ("MoveCube", "xhard4")]  # 按档序再按任务序
    over = H.V9_CELLS[("StopCube", "xhard1")] + 1
    for bad in ({"StopCube@xhard1": over}, {"StopCube@xhard1": 0}, {"StopCube@xhard1": True},
                {"InsertPeg@xhard1": 1}):
        path.write_text(json.dumps(bad))
        with pytest.raises(R.RolloutError):
            R.resolve_cells(path)


def test_parse_by_task_helpers():
    tasks = ["A", "B"]
    assert F.parse_int_by_task(None, tasks, {"A": 1, "B": 2}, "x") == {"A": 1, "B": 2}
    assert F.parse_int_by_task("4", tasks, {"A": 1, "B": 2}, "x") == {"A": 4, "B": 4}
    assert F.parse_int_by_task("B=7", tasks, {"A": 1, "B": 2}, "x") == {"A": 1, "B": 7}
    assert F.parse_select_by_task("B=2..4,A=1", tasks, {"A": (0,), "B": (0,)}) == {"A": (1,), "B": (2, 3, 4)}
    assert F.parse_select_by_task("1,3", tasks, {"A": (0,), "B": (0,)}) == {"A": (1, 3), "B": (1, 3)}
    with pytest.raises(H.SpecsError):
        F.parse_select_by_task("C=1", tasks, {"A": (0,), "B": (0,)})
    with pytest.raises(H.SpecsError):
        D.parse_task_max_reset_attempts("A=x", "xhard1")


def test_extract_builds_sampling_from_native_blocks_without_env():
    # ①定规则：只读任务模块的 native_blocks，不建环境（资源守卫会拒绝任何场景构建）
    ex = FS._extract
    one = ex.build_sampling(["StopCube", "MoveCube"])
    assert set(one) == {"tasks"} and set(one["tasks"]) == {"StopCube", "MoveCube"}
    assert all(set(v) == {"decision", "native"} for v in one["tasks"].values())
    assert H.digest(ex.build_sampling(["StopCube", "MoveCube"])) == H.digest(one)  # 不抽随机数，两次逐字相同
    with pytest.raises(ModuleNotFoundError):
        ex.build_sampling(["NoSuchTask"])
