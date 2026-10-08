"""C12／C13 交界：``export_eval_identities.py`` 经真实 ``BenchmarkEnvBuilder(ood)`` 逐局列身份，对包内真实 V9 规格跑。

独立期望：新值身份集合直接从包内五份 ``specs.jsonl`` 的交付行（selected 且 rollout ok）读出，不经 builder；
xhard0 开关打开时 192 局官方路线清单按贪心均衡切片（手算小例子核贪心规则）。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest

from tests._support.loaders import load_script

E = load_script("injection-dev/export_eval_identities.py")
H = E.hard_specs
PACKAGED = Path(H.__file__).resolve().parents[1] / "env_metadata" / "ood"


def _delivered_rows():
    rows = []
    for tier in H.TIERS:
        lines = (PACKAGED / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            r = json.loads(line)
            if r["selected"] and (r["rollout"] or {}).get("status") == "ok":
                rows.append({"task": r["task"], "tier": r["tier"], "episode": r["episode"], "seed": r["seed"],
                             "candidate": r["candidate"]})
    return rows


@pytest.fixture(scope="module")
def delivered():
    return _delivered_rows()


def isolate_specs_root_env(mp) -> None:
    """main 会把 --specs-root 写进进程环境。``delenv(raising=False)`` 在变量本不存在时不登记，单用它会让写入泄漏
    到后续用例；先 setenv（登记原状：存在则记原值、不存在则记「不存在」）再 delenv，用例结束即复原。"""
    mp.setenv(H.SPECS_ROOT_ENV, "t7-placeholder")
    mp.delenv(H.SPECS_ROOT_ENV)


@pytest.fixture
def env(monkeypatch):
    isolate_specs_root_env(monkeypatch)
    return monkeypatch


def _write(path: Path, rows) -> Path:
    path.write_text(json.dumps({"schema": "v8-delivery/1", "rows": rows}), encoding="utf-8")
    return path


def _lines(capsys) -> str:
    return next(l for l in capsys.readouterr().out.splitlines() if l.startswith("EVAL_IDENTITY_EXPORT="))


def test_export_matches_packaged_delivery(env, delivered, tmp_path, capsys):
    out = tmp_path / "ids.jsonl"
    official = tmp_path / "official.jsonl"
    rc = E.main(["--specs-root", str(PACKAGED), "--delivery", str(_write(tmp_path / "d.json", delivered)),
                 "--out", str(out), "--official-out", str(official)])
    line = _lines(capsys)
    assert rc == 0 and line.startswith("EVAL_IDENTITY_EXPORT=PASS ") and "official=skipped" in line
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == len(delivered)
    assert {(r["task"], r["tier"], r["seed"]) for r in rows} == {(r["task"], r["tier"], r["seed"]) for r in delivered}
    assert all(set(r) == {"task", "episode", "tier", "seed", "candidate", "source_episode", "round", "shard"} for r in rows)
    assert all(r["round"] is None and r["shard"] is None for r in rows)
    # 每任务 episode 连续 0..n-1，且任务按 16 任务规范序排列
    per_task = Counter(r["task"] for r in rows)
    assert [r["task"] for r in rows] == [t for t in H.ALL_TASKS for _ in range(per_task[t])]
    for task in H.ALL_TASKS:
        assert [r["episode"] for r in rows if r["task"] == task] == list(range(per_task[task]))
    cand = {(r["task"], r["tier"], r["seed"]): r["candidate"] for r in delivered}
    assert all(r["candidate"] == cand[(r["task"], r["tier"], r["seed"])] for r in rows)
    assert not official.exists()  # 开关关：官方路线清单不写


def test_export_detects_delivery_mismatch(env, delivered, tmp_path, capsys):
    rc = E.main(["--specs-root", str(PACKAGED), "--delivery", str(_write(tmp_path / "d.json", delivered[1:])),
                 "--out", str(tmp_path / "ids.jsonl")])
    line = _lines(capsys)
    assert rc == 1 and line.startswith("EVAL_IDENTITY_EXPORT=FAIL ") and "delivery_mismatch=1 " in line


def test_v9_requires_delivery(env, tmp_path):
    with pytest.raises(SystemExit):
        E.main(["--specs-root", str(PACKAGED), "--out", str(tmp_path / "ids.jsonl")])
    assert not (tmp_path / "ids.jsonl").exists()


def test_xhard0_switch_on_adds_official_route(env, delivered, tmp_path, capsys):
    env.setattr(H, "XHARD0_IN_TEST_HARD", True)
    d = _write(tmp_path / "d.json", delivered)
    with pytest.raises(SystemExit):  # 开关开时必须显式给 --official-out（默认值是 V8 根）
        E.main(["--specs-root", str(PACKAGED), "--delivery", str(d), "--out", str(tmp_path / "x.jsonl")])
    official = tmp_path / "official.jsonl"
    rc = E.main(["--specs-root", str(PACKAGED), "--delivery", str(d), "--out", str(tmp_path / "ids.jsonl"),
                 "--official-out", str(official)])
    assert rc == 0, _lines(capsys)
    rows = [json.loads(x) for x in (tmp_path / "ids.jsonl").read_text().splitlines()]
    x0 = [r for r in rows if r["tier"] == H.XHARD0]
    assert Counter(r["task"] for r in x0) == {t: H.XHARD0_PER_TASK for t in H.ALL_TASKS}
    assert len(rows) == len(delivered) + len(x0)
    off = [json.loads(x) for x in official.read_text().splitlines()]
    assert {(r["task"], r["seed"]) for r in off} == {(r["task"], r["seed"]) for r in x0}
    assert {r["shard"] for r in off} == set(range(E.SHARDS))
    assert all(r["source_episode"] in H.XHARD0_EPISODES for r in off)


def test_balance_greedy_by_hand():
    # 三局两片：按估计用时降序逐局给当前最轻的片（同轻取片号小）
    rows = [{"task": "StopCube", "episode": 0}, {"task": "BinFill", "episode": 0}, {"task": "PickHighlight", "episode": 0}]
    assert E.TASK_SECONDS["PickHighlight"] > E.TASK_SECONDS["BinFill"] > E.TASK_SECONDS["StopCube"]
    E.balance(rows, 2)
    assert {r["task"]: r["shard"] for r in rows} == {"PickHighlight": 0, "BinFill": 1, "StopCube": 1}
    same = [{"task": "StopCube", "episode": 1}, {"task": "StopCube", "episode": 0}]
    E.balance(same, 2)
    assert [(r["episode"], r["shard"]) for r in same] == [(1, 1), (0, 0)]


def test_check_rows_negatives():
    cells = {("StopCube", "xhard1"): 2}
    good = [{"task": "StopCube", "tier": "xhard1", "seed": s, "round": None, "shard": None} for s in (1, 2)]
    ok, facts = E.check_rows(good, [], cells, {("StopCube", "xhard1", 1), ("StopCube", "xhard1", 2)})
    assert ok and facts["cell_mismatch"] == 0 and facts["delivery_mismatch"] == 0
    assert not E.check_rows([dict(good[0], round=3), good[1]], [], cells)[0]
    ok, facts = E.check_rows(good[:1], [], cells)
    assert not ok and facts["cell_mismatch"] == 1
    ok, facts = E.check_rows(good + [{"task": "BinFill", "tier": "xhard1", "seed": 3}], [], cells)
    assert not ok and facts["cell_mismatch"] == 1  # 表外格也计
    assert not E.check_rows(good, [{"task": "StopCube"}], cells)[0]  # 开关关时官方行必须为 0


def test_specs_root_env_restored_after_main(delivered, tmp_path, capsys):
    # 夹具的复原效果本身：在独立的 MonkeyPatch 上下文里跑 main，退出上下文后环境变量回到调用前（不依赖用例顺序）
    before = os.environ.get(H.SPECS_ROOT_ENV)
    with pytest.MonkeyPatch.context() as mp:
        isolate_specs_root_env(mp)
        assert E.main(["--specs-root", str(PACKAGED), "--delivery", str(_write(tmp_path / "d.json", delivered)),
                       "--out", str(tmp_path / "ids.jsonl")]) == 0
        assert os.environ[H.SPECS_ROOT_ENV] == str(PACKAGED.resolve())  # main 确实写了
    assert os.environ.get(H.SPECS_ROOT_ENV) == before
    capsys.readouterr()
