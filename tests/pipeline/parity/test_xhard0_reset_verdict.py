"""C15 ``hard_regression.py xhard0-reset-parity`` 的判定层（XHARD0_RESET_PARITY／XHARD0_DEMO_DIFF）。

GPU 探针 ``_run_probe`` 换成返回手写探针结果的替身（monkeypatch 模块属性，不注入 sys.modules）；真实的
``cmd_xhard0_reset_parity`` 逐字段比、仅改名判定、合并上一轮、写报告，再由真实 ``_xhard0_reset_verdict`` 出判定行。
每任务局数取一个小的测试值（不是业务常量）；任务清单取生产的 16 任务规范序。
"""
from __future__ import annotations

import copy
import json
import types

import pytest

import parity_fixtures as F

PER_TASK = 2


@pytest.fixture(scope="module")
def R():
    return F.hard_regression()


@pytest.fixture(scope="module")
def tasks():
    return tuple(F.hard_parity().hard_specs_light().ALL_TASKS)


def probe_row(side: str, task: str, ep: int, idx: int) -> dict:
    seed = 500000 + 100 * idx
    return {"episode": ep, "seed": seed, "difficulty": "hard" if side == "official" else "xhard0",
            "make_kwargs": {"env_id": task, "seed": seed}, "wrapper_chain": ["A", "B", task],
            "pre_demo_state": {"actors": {"cube_0": [0.1, 0.2], "button_left": [0.3, 0.4]},
                               "articulations": {"panda": [1.0]}},
            "task_goal": [f"do {task}"], "choices": [], "demo_frames": 5, "demo_digest": "d", "post_state": "p"}


def run(R, tasks, tmp_path, monkeypatch, capsys, mutate=None, only_tasks=None, merge_with=None):
    """mutate(side, task, idx, row) 就地改替身返回值。返回 (退出码, 判定行, 报告行)。"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = {"rows": [{"task": t, "episode": 3 + 4 * i} for t in tasks for i in range(PER_TASK)]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))

    def fake_probe(side, task, src, eps, gpu):
        out = []
        for idx, ep in enumerate(eps):
            row = probe_row(side, task, ep, idx)
            if mutate:
                mutate(side, task, idx, row)
            out.append(row)
        return out

    monkeypatch.setattr(R, "_run_probe", fake_probe)
    monkeypatch.setattr(R, "_hard_specs", lambda: types.SimpleNamespace(ALL_TASKS=tasks, XHARD0_PER_TASK=PER_TASK))
    monkeypatch.setattr(R, "_require_xhard0_in_test_hard", lambda hs: None)
    args = types.SimpleNamespace(manifest=str(tmp_path / "m.json"), out=str(tmp_path / "out"), src_root=str(tmp_path),
                                 gpu="0", tasks=",".join(only_tasks) if only_tasks else None,
                                 merge_with=merge_with, report_name="r.jsonl")
    rc = R.cmd_xhard0_reset_parity(args)
    out = capsys.readouterr().out.strip().splitlines()
    report = [json.loads(t) for t in (tmp_path / "out" / "r.jsonl").read_text().splitlines()]
    return rc, out, report


def verdict_line(out):
    return next(t for t in out if t.startswith("XHARD0_RESET_PARITY="))


def test_两侧逐位相同_PASS(R, tasks, tmp_path, monkeypatch, capsys):
    rc, out, report = run(R, tasks, tmp_path, monkeypatch, capsys)
    n = len(tasks) * PER_TASK
    assert rc == 0 and len(report) == n
    line = verdict_line(out)
    assert line.startswith("XHARD0_RESET_PARITY=PASS ") and line.endswith(f" compared={n} det_diff=0 name_only=0 first_det_diff=-")
    assert out[-1] == f"XHARD0_DEMO_DIFF=INFO frames_equal={n} max_frame_diff=0 demo_equal={n} post_equal={n}"
    assert sum(t.startswith("XHARD0_RESET_TASK ") for t in out) == len(tasks)


@pytest.mark.parametrize("field,value", [
    ("seed", 1), ("make_kwargs", {"env_id": "X"}), ("wrapper_chain", ["A"]), ("task_goal", ["other"]),
    ("choices", ["a"]), ("pre_demo_state", {"actors": {"cube_0": [9.0, 0.2], "button_left": [0.3, 0.4]},
                                            "articulations": {"panda": [1.0]}}),
])
def test_确定性层任一字段不同即FAIL(R, tasks, tmp_path, monkeypatch, capsys, field, value):
    target = tasks[3]

    def mutate(side, task, idx, row):
        if side == "hard" and task == target and idx == 1:
            row[field] = value
    rc, out, report = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    line = verdict_line(out)
    assert rc == 1 and line.startswith("XHARD0_RESET_PARITY=FAIL ") and " det_diff=1 " in line
    assert f"first_det_diff={target}/ep7:['{field}']" in line
    assert sum(bool(r["det_bad"]) for r in report) == 1


def test_官方侧难度名不是hard也算不一致(R, tasks, tmp_path, monkeypatch, capsys):
    def mutate(side, task, idx, row):
        if side == "official" and task == tasks[0] and idx == 0:
            row["difficulty"] = "medium"
    rc, out, _ = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    assert rc == 1 and "['difficulty']" in verdict_line(out)


def test_演示前状态只是实体改名_记name_only不计差异(R, tasks, tmp_path, monkeypatch, capsys):
    target = tasks[5]

    def mutate(side, task, idx, row):
        if side == "hard" and task == target:
            actors = row["pre_demo_state"]["actors"]
            actors["button_right"] = actors.pop("button_left")
    rc, out, report = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    line = verdict_line(out)
    assert rc == 0 and f" name_only={PER_TASK} " in line and f"name_only_tasks=['{target}']" in line
    assert all(r["det_bad"] == [] for r in report)


def test_同键互换取值是真差异_FAIL(R, tasks, tmp_path, monkeypatch, capsys):
    """两个同形 actor 键集合不变、只互换取值：去名后多重集相同，但没有任何改名，必须计真差异（与
    ``env-digest-compare`` 的判法一致），判定行 FAIL、不记 name_only。"""
    def mutate(side, task, idx, row):
        if side == "hard" and task == tasks[0] and idx == 0:
            actors = row["pre_demo_state"]["actors"]
            actors["cube_0"], actors["button_left"] = actors["button_left"], actors["cube_0"]
    rc, out, report = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    line = verdict_line(out)
    assert rc == 1 and line.startswith("XHARD0_RESET_PARITY=FAIL ")
    assert " det_diff=1 name_only=0 " in line and f"first_det_diff={tasks[0]}/ep3:['pre_demo_state']" in line
    assert [r["name_only"] for r in report].count(True) == 0


BUS = "ButtonUnmaskSwap"


def _swap_buttons(side, row, extra_swap=False):
    """官方侧 button_left／button_right 在 hard 侧名字对调（F3），状态数值相同。"""
    left, right = [0.5, 0.6], [0.7, 0.8]
    row["pre_demo_state"]["articulations"].update(
        {"button_left": left, "button_right": right} if side == "official" else {"button_left": right, "button_right": left})
    if extra_swap and side == "hard":
        actors = row["pre_demo_state"]["actors"]
        actors["cube_0"], actors["button_left"] = actors["button_left"], actors["cube_0"]


def test_已声明的BUS左右按钮对调_记name_only(R, tasks, tmp_path, monkeypatch, capsys):
    """键集合相同、取值互换，但正是 XHARD0_DECLARED_RENAMES 声明的 BUS 对调：按映射改名后逐键相等 ⇒ name_only。"""
    assert BUS in tasks and R.XHARD0_DECLARED_RENAMES[BUS]

    def mutate(side, task, idx, row):
        if task == BUS:
            _swap_buttons(side, row)
    rc, out, report = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    line = verdict_line(out)
    assert rc == 0 and f" det_diff=0 name_only={PER_TASK} " in line and f"name_only_tasks=['{BUS}']" in line


def test_同样的左右对调出现在未声明任务_真差异(R, tasks, tmp_path, monkeypatch, capsys):
    other = next(t for t in tasks if t != BUS)

    def mutate(side, task, idx, row):
        if task == other and idx == 0:
            _swap_buttons(side, row)
    rc, out, _ = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    assert rc == 1 and " det_diff=1 name_only=0 " in verdict_line(out)


def test_BUS声明对调之外另有同键互换_真差异(R, tasks, tmp_path, monkeypatch, capsys):
    def mutate(side, task, idx, row):
        if task == BUS:
            _swap_buttons(side, row, extra_swap=(idx == 0))
    rc, out, _ = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    line = verdict_line(out)
    assert rc == 1 and f" det_diff=1 name_only={PER_TASK - 1} " in line


def test_改名同时取值也变_是真差异_FAIL(R, tasks, tmp_path, monkeypatch, capsys):
    """键集合确实不同（有改名），但去名后多重集也不同：不是仅改名，计真差异。"""
    def mutate(side, task, idx, row):
        if side == "hard" and task == tasks[2] and idx == 1:
            actors = row["pre_demo_state"]["actors"]
            actors["button_right"] = [9.9, 0.4]
            del actors["button_left"]
    rc, out, _ = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    line = verdict_line(out)
    assert rc == 1 and " det_diff=1 name_only=0 " in line


def test_仅改名判定辅助_正例与负例(R):
    base = {"actors": {"a": [1], "b": [2]}, "articulations": {"panda": [0]}}
    renamed = {"actors": {"a": [1], "c": [2]}, "articulations": {"panda": [0]}}
    swapped = {"actors": {"a": [2], "b": [1]}, "articulations": {"panda": [0]}}
    assert R._pre_state_name_only(base, renamed) is True
    assert R._pre_state_name_only(base, swapped) is False
    assert R._pre_state_name_only(base, base) is False
    assert R._pre_state_name_only(base, {"actors": {"a": [1], "c": [3]}, "articulations": {"panda": [0]}}) is False
    swap_map = {"actors": {"a": "b", "b": "a"}}
    assert R._pre_state_name_only(base, swapped, swap_map) is True
    assert R._pre_state_name_only(base, {"actors": {"a": [2], "b": [9]}, "articulations": {"panda": [0]}}, swap_map) is False


def test_演示层差异只报告(R, tasks, tmp_path, monkeypatch, capsys):
    def mutate(side, task, idx, row):
        if side == "hard" and task == tasks[1] and idx == 0:
            row.update(demo_frames=8, demo_digest="x", post_state="y")
    rc, out, _ = run(R, tasks, tmp_path, monkeypatch, capsys, mutate)
    n = len(tasks) * PER_TASK
    assert rc == 0
    assert out[-1] == (f"XHARD0_DEMO_DIFF=INFO frames_equal={n - 1} max_frame_diff=3 demo_equal={n - 1} "
                       f"post_equal={n - 1}")


def test_局数不齐即FAIL_合并上一轮补齐后PASS(R, tasks, tmp_path, monkeypatch, capsys):
    rc, out, report = run(R, tasks, tmp_path / "a", monkeypatch, capsys, only_tasks=tasks[:2])
    assert rc == 1 and f"compared={2 * PER_TASK} " in verdict_line(out)
    full_rc, _out, full = run(R, tasks, tmp_path / "b", monkeypatch, capsys)
    assert full_rc == 0
    prev = tmp_path / "prev.jsonl"
    stale = copy.deepcopy(full)
    for r in stale:
        if r["task"] == tasks[0]:
            r["det_bad"] = ["seed"]  # 上一轮这个任务坏了；本轮重跑它，应整段替换
    prev.write_text("".join(json.dumps(r) + "\n" for r in stale))
    rc, out, report = run(R, tasks, tmp_path / "c", monkeypatch, capsys, only_tasks=tasks[:1], merge_with=str(prev))
    assert rc == 0 and len(report) == len(tasks) * PER_TASK and " det_diff=0 " in verdict_line(out)


def test_判定函数_局数多一局也FAIL(R, tasks):
    rows = [{"task": t, "source_episode": i, "det_bad": [], "name_only": False, "demo_frames": [1, 1],
             "demo_equal": True, "post_equal": True} for t in tasks for i in range(PER_TASK)]
    n = len(tasks)
    assert R._xhard0_reset_verdict(rows, PER_TASK, n_tasks=n) == 0
    assert R._xhard0_reset_verdict(rows + [dict(rows[0])], PER_TASK, n_tasks=n) == 1
    assert R._xhard0_reset_verdict(rows[:-1], PER_TASK, n_tasks=n) == 1
    assert R._xhard0_reset_verdict([], PER_TASK, n_tasks=n) == 1


def test_期望局数随任务清单长度变化(R, tasks, tmp_path, monkeypatch, capsys):
    """期望局数 = 任务清单长度 × 每任务局数，不写死 16：任务清单缩到 3 个时，3×PER_TASK 局即 PASS。"""
    short = tasks[:3]
    rc, out, report = run(R, short, tmp_path, monkeypatch, capsys)
    line = verdict_line(out)
    assert rc == 0 and len(report) == 3 * PER_TASK and line.startswith("XHARD0_RESET_PARITY=PASS ")
    assert f" shape=3x1x{PER_TASK} " in line
    rows = [{"task": t, "source_episode": i, "det_bad": [], "name_only": False, "demo_frames": [1, 1],
             "demo_equal": True, "post_equal": True} for t in short for i in range(PER_TASK)]
    assert R._xhard0_reset_verdict(rows, PER_TASK, n_tasks=3) == 0
    assert R._xhard0_reset_verdict(rows, PER_TASK, n_tasks=len(tasks)) == 1
