"""``export_eval_identities.py`` 经真实 ``BenchmarkEnvBuilder(task, dataset)`` 逐局列身份，对包内真实规格跑。

独立期望：ood 的新值身份集合直接从包内五份 ``specs.jsonl`` 的交付行（selected 且 rollout ok）读出，不经 builder；
hard-verify 每任务恰 12 局 xhard0、官方原号恰为 3, 7, …, 47（手写）。拆仓后 xhard0 只按数据集名出现在 hard-verify，
不再前置到 ood（旧开关用例随之删除）。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tests._support.dev_loaders import load_script

E = load_script("injection-dev/export_eval_identities.py")
H = E.hard_specs()
PACKAGED = Path(H.__file__).resolve().parents[1] / "env_metadata" / "ood"
#: 官方 test 的 hard 子集原 episode 号（手写，不读被测代码）
XHARD0_SOURCE = tuple(range(3, 48, 4))


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


def _write(path: Path, rows) -> Path:
    path.write_text(json.dumps({"schema": "v8-delivery/1", "rows": rows}), encoding="utf-8")
    return path


def _line(capsys) -> str:
    return next(l for l in capsys.readouterr().out.splitlines() if l.startswith("EVAL_IDENTITY_EXPORT="))


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_split_accept_two_rows(tmp_path, capsys):
    """S5 runbook：VideoUnmask × {hard-verify 局 0（官方原号 3）、ood 局 0} 恰两行。"""
    out = tmp_path / "ids.jsonl"
    rc = E.main(["--dataset", "hard-verify,ood", "--tasks", "VideoUnmask", "--episodes", "0:1", "--out", str(out)])
    line = _line(capsys)
    assert rc == 0 and line.startswith("EVAL_IDENTITY_EXPORT=PASS ") and " episodes=2 " in line
    assert " hard_verify=1 ood=1 xhard0=1 " in line
    rows = _read(out)
    assert [(r["dataset"], r["task"], r["builder_episode"]) for r in rows] == [("hard-verify", "VideoUnmask", 0),
                                                                              ("ood", "VideoUnmask", 0)]
    hv, ood = rows
    assert hv["tier"] == "xhard0" and hv["source_episode"] == 3 and hv["candidate"] is None and hv["spec_sha256"] is None
    assert ood["tier"] != "xhard0" and ood["source_episode"] is None and isinstance(ood["candidate"], int)
    assert len(ood["spec_sha256"]) == 64
    for r in rows:
        assert set(r) == set(E.ROW_KEYS) and r["episode"] == r["builder_episode"]
        assert r["key"] == f"{r['task']}_{r['tier']}_{r['seed']}"
    # 与执行侧 builder 的解析一致（同一 hard_specs）
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    for r in rows:
        got = BenchmarkEnvBuilder("VideoUnmask", dataset=r["dataset"]).resolve_identity(0)
        assert (got["tier"], got["seed"], got.get("source_episode"), got.get("spec_sha256")) == \
               (r["tier"], r["seed"], r["source_episode"], r["spec_sha256"])


def test_full_export_matches_packaged_delivery(delivered, tmp_path, capsys):
    out = tmp_path / "ids.jsonl"
    rc = E.main(["--out", str(out), "--delivery", str(_write(tmp_path / "d.json", delivered))])
    line = _line(capsys)
    assert rc == 0 and line.startswith("EVAL_IDENTITY_EXPORT=PASS ") and "delivery_mismatch=0 " in line
    rows = _read(out)
    hv = [r for r in rows if r["dataset"] == "hard-verify"]
    ood = [r for r in rows if r["dataset"] == "ood"]
    assert [r["dataset"] for r in rows] == ["hard-verify"] * len(hv) + ["ood"] * len(ood)  # 按 --dataset 顺序写出
    assert len(hv) == 16 * 12 and all(r["tier"] == "xhard0" for r in hv)
    for task in H.ALL_TASKS:
        assert [r["source_episode"] for r in hv if r["task"] == task] == list(XHARD0_SOURCE)
        assert [r["builder_episode"] for r in hv if r["task"] == task] == list(range(12))
    assert not [r for r in ood if r["tier"] == "xhard0"]
    assert len(ood) == len(delivered)
    assert {(r["task"], r["tier"], r["seed"]) for r in ood} == {(r["task"], r["tier"], r["seed"]) for r in delivered}
    cand = {(r["task"], r["tier"], r["seed"]): r["candidate"] for r in delivered}
    assert all(r["candidate"] == cand[(r["task"], r["tier"], r["seed"])] for r in ood)
    per_task = Counter(r["task"] for r in ood)
    assert [r["task"] for r in ood] == [t for t in H.ALL_TASKS for _ in range(per_task[t])]
    for task in H.ALL_TASKS:
        assert [r["builder_episode"] for r in ood if r["task"] == task] == list(range(per_task[task]))


def test_full_export_detects_delivery_mismatch(delivered, tmp_path, capsys):
    rc = E.main(["--dataset", "ood", "--out", str(tmp_path / "ids.jsonl"),
                 "--delivery", str(_write(tmp_path / "d.json", delivered[1:]))])
    line = _line(capsys)
    assert rc == 1 and line.startswith("EVAL_IDENTITY_EXPORT=FAIL ") and "delivery_mismatch=1 " in line


def test_check_rows_negatives(tmp_path, capsys):
    E.main(["--dataset", "hard-verify,ood", "--tasks", "VideoUnmask", "--episodes", "0:1",
            "--out", str(tmp_path / "ids.jsonl")])
    capsys.readouterr()
    good = _read(tmp_path / "ids.jsonl")
    exp = {"hard-verify": 1, "ood": 1}
    assert E.check_rows(good, exp, H, full=False)[0]
    hv, ood = good
    cases = {
        "hard-verify 行带 candidate": [dict(hv, candidate=1), ood],
        "hard-verify 行原号不在官方 hard 子集": [dict(hv, source_episode=4), ood],
        "ood 行是 xhard0": [hv, dict(ood, tier="xhard0", key=f"{ood['task']}_xhard0_{ood['seed']}")],
        "ood 行缺指纹": [hv, dict(ood, spec_sha256=None)],
        "key 不自洽": [dict(hv, key="x"), ood],
        "缺一行": [hv],
        "多一个键": [dict(hv, extra=1), ood],
        "重复": [hv, ood, ood],
    }
    for name, rows in cases.items():
        assert not E.check_rows(rows, exp, H, full=False)[0], name


@pytest.mark.parametrize("argv", [["--dataset", "test", "--out", "x"], ["--tasks", "NoSuchTask", "--out", "x"],
                                  ["--episodes", "3:1", "--out", "x"], ["--dataset", "ood,ood", "--out", "x"]],
                         ids=["bad_dataset", "bad_task", "empty_range", "dup_dataset"])
def test_bad_args_exit_2(argv):
    with pytest.raises(SystemExit) as ei:
        E.main(argv)
    assert ei.value.code == 2
