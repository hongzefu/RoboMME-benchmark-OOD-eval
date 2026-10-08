"""C12 分片与聚合：真实 ``split_v8`` → 片内 ``run_continue_v8``（FakeRunner）→ ``aggregate_v8`` 写 ``delivery.json``；
``generate_h5.py`` 的 A40 闸门与 aggregate 子命令。

期望值手写：每格局数、哪格 PASS／pending／exhausted／shortfall、h5 坏在哪一种；计数键零值必须显式写出（M17）。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from gen_world import GH, R, FakeRunner, continue_kwargs, freeze_root, read_rows, write_h5

SC, SW, MC = ("StopCube", "xhard1"), ("SwingXtimes", "xhard1"), ("MoveCube", "xhard4")
WAYS = {"MoveCube": lambda e: e % 3}
SHARD_A = {SC: 2, SW: 1}
SHARD_B = {MC: 3}


@pytest.fixture
def frozen(tmp_path):
    return freeze_root(tmp_path / "frozen", {SC: (2, 4), SW: (1, 2), MC: (3, 6)}, ways=WAYS)


def _ids(root: Path, tiers) -> set[tuple]:
    out = set()
    for tier in tiers:
        _, rows = read_rows(root / tier / "specs.jsonl")
        out |= {(r["task"], r["tier"], r["candidate"], r["seed"], r["spec_sha256"]) for r in rows}
    return out


# ── 分片 ──────────────────────────────────────────────────────────


def test_split_is_disjoint_and_complete(frozen, tmp_path):
    meta_a = R.split_v8(frozen, SHARD_A, tmp_path / "a", label="a")
    meta_b = R.split_v8(frozen, SHARD_B, tmp_path / "b", label="b")
    got_a = _ids(tmp_path / "a" / "specs", ["xhard1"])
    got_b = _ids(tmp_path / "b" / "specs", ["xhard4"])
    assert got_a & got_b == set()
    assert got_a | got_b == _ids(frozen, ["xhard1", "xhard4"])
    for meta, tiers in ((meta_a, ["xhard1"]), (meta_b, ["xhard4"])):
        assert set(meta["sources"]) == set(tiers)
        for tier in tiers:
            src_header, _ = read_rows(frozen / tier / "specs.jsonl")
            assert meta["sources"][tier]["identity_sha256"] == src_header["identity_sha256"]
            assert meta["sources"][tier]["file_sha256"] == \
                hashlib.sha256((frozen / tier / "specs.jsonl").read_bytes()).hexdigest()
    sub, _ = read_rows(tmp_path / "a" / "specs" / "xhard1" / "specs.jsonl")
    assert sub["tasks"] == ["StopCube", "SwingXtimes"] and sub["delivery_per_cell"] == {"StopCube": 2, "SwingXtimes": 1}
    assert json.loads((tmp_path / "a" / R.SHARD_META).read_text())["cells"] == {"StopCube@xhard1": 2, "SwingXtimes@xhard1": 1}


def test_split_rejections(frozen, tmp_path, monkeypatch):
    R.split_v8(frozen, SHARD_A, tmp_path / "a", label="a")
    with pytest.raises(R.RolloutError, match="禁止覆盖"):
        R.split_v8(frozen, SHARD_A, tmp_path / "a", label="a")
    with pytest.raises(R.RolloutError, match="冻结配额 2 ≠ 格表 1"):
        R.split_v8(frozen, {SC: 1}, tmp_path / "c", label="c")
    FakeRunner().install(monkeypatch)
    R.run_continue_v8(frozen, {SC: 2, SW: 1, MC: 3}, tmp_path / "run", **continue_kwargs(tmp_path))
    with pytest.raises(R.RolloutError, match="已跑过"):
        R.split_v8(frozen, SHARD_A, tmp_path / "d", label="d")


# ── 聚合四种状态与计数序列化 ─────────────────────────────────────────


def test_shard_run_delivers_and_serializes_zero_counts(frozen, tmp_path, monkeypatch):
    shard = tmp_path / "a"
    R.split_v8(frozen, SHARD_A, shard, label="a")
    FakeRunner({("StopCube", 0): ["fail"]}).install(monkeypatch)
    summary = R.run_continue_v8(shard / "specs", SHARD_A, shard, **continue_kwargs(tmp_path))
    report = json.loads((shard / "delivery.json").read_text())  # 读回落盘形态，而不是内存对象
    assert report["schema"] == R.V8_DELIVERY_SCHEMA
    assert report["line"].startswith("V8_DELIVERY_SET=PASS ") and summary["delivery_set"] == report["line"]
    counts = report["counts"]
    assert set(counts) == set(R.V8_TOTAL_COUNT_KEYS) and all(type(v) is int for v in counts.values())
    # 手算：StopCube 2 + SwingXtimes 1 = 3 局；StopCube 候选 0 失败、递补 2
    assert (counts["expected"], counts["delivered"], counts["failed"], counts["backfills"]) == (3, 3, 1, 1)
    assert (counts["candidates"], counts["tried"], counts["spares_left"]) == (6, 4, 2)
    for key in ("exec_over_cap", "infra_retries", "pending", "bad_h5", "exhausted_cells", "pending_cells", "failed_cells"):
        assert counts[key] == 0, key
    assert "expected=3 " in report["line"] and "failed=1 " in report["line"]
    for cell in report["cells"].values():
        assert set(R.V8_CELL_COUNT_KEYS) <= set(cell) and cell["status"] == "PASS" and cell["reason"] is None
    rows = {(r["task"], r["candidate"]): r for r in report["rows"]}
    assert set(rows) == {("StopCube", 1), ("StopCube", 2), ("SwingXtimes", 0)}
    r = rows[("StopCube", 2)]
    assert (r["role"], r["initial_selected"], r["exec_steps"], r["frames"]) == ("backfill", False, 4, 5)
    assert (shard / r["path"]).resolve() == Path(r["h5"]).resolve()
    assert Path(r["video"]).name.endswith(".mp4") and f"_seed{r['seed']}_" in Path(r["video"]).name


def test_pending_cells_when_nothing_ran(frozen, tmp_path):
    ledgers = tmp_path / "ledger"
    ledgers.mkdir()
    (ledgers / R.LEDGER_NAME).write_text("")
    report = R.aggregate_v8(frozen, {SC: 2, SW: 1, MC: 3}, [ledgers], tmp_path / "d.json")
    assert {k: (c["status"], c["reason"]) for k, c in report["cells"].items()} == {
        "StopCube/xhard1": ("FAIL", "pending"), "SwingXtimes/xhard1": ("FAIL", "pending"),
        "MoveCube/xhard4": ("FAIL", "pending")}
    assert report["counts"]["pending_cells"] == 3 and report["counts"]["expected"] == 6
    assert report["counts"]["delivered"] == 0 and report["line"].startswith("V8_DELIVERY_SET=FAIL ")


def test_movecube_same_way_exhausted_even_with_spares(frozen, tmp_path, monkeypatch):
    shard = tmp_path / "b"
    R.split_v8(frozen, SHARD_B, shard, label="b")
    FakeRunner({("MoveCube", 1): ["fail"], ("MoveCube", 4): ["fail"]}).install(monkeypatch)
    R.run_continue_v8(shard / "specs", SHARD_B, shard, **continue_kwargs(tmp_path))
    cell = json.loads((shard / "delivery.json").read_text())["cells"]["MoveCube/xhard4"]
    assert (cell["status"], cell["reason"], cell["delivered"], cell["spares_left"]) == ("FAIL", "exhausted", 2, 2)


def test_shortfall_when_failure_not_backfilled(tmp_path):
    root = freeze_root(tmp_path / "frozen", {SC: (2, 4)})
    path = root / "xhard1" / "specs.jsonl"
    header, rows = read_rows(path)
    h5 = write_h5(tmp_path / "ep1.h5", 3, 0)
    rows[0].update(tried=True, selected=False, rollout={"status": "failed", "error_type": "TaskFailed"})
    rows[1].update(tried=True, rollout={"status": "ok", "h5_path": str(h5),
                                        "h5_sha256": hashlib.sha256(h5.read_bytes()).hexdigest()})
    R.write_back(path, header, rows, R.file_sha256(path))
    (tmp_path / R.LEDGER_NAME).write_text("")
    cell = R.aggregate_v8(root, {SC: 2}, [tmp_path], tmp_path / "d.json")["cells"]["StopCube/xhard1"]
    assert (cell["status"], cell["reason"], cell["delivered"], cell["spares_left"]) == ("FAIL", "shortfall", 1, 2)
    assert cell["expected"] == 2  # 分母是格表局数，不是结果行数（M17）


def test_aggregate_requires_ledger_and_refuses_overwrite(frozen, tmp_path):
    with pytest.raises(R.RolloutError, match="缺少尝试账本"):
        R.aggregate_v8(frozen, {SC: 2, SW: 1, MC: 3}, [tmp_path / "none"], tmp_path / "d.json")
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(R.RolloutError, match="已存在"):
        R.aggregate_v8(frozen, {SC: 2, SW: 1, MC: 3}, [tmp_path], tmp_path / "x.json")
    assert (tmp_path / "x.json").read_text() == "{}"


def test_aggregate_rejects_infra_from_foreign_cells(frozen, tmp_path):
    (tmp_path / R.LEDGER_NAME).write_text(json.dumps({"kind": "infra_retry", "task": "BinFill", "tier": "xhard1",
                                                      "candidate": 0, "round": 0}) + "\n")
    with pytest.raises(R.RolloutError, match="格表之外"):
        R.aggregate_v8(frozen, {SC: 2, SW: 1, MC: 3}, [tmp_path], tmp_path / "d.json")


# ── 整树搬迁后的 rebase ─────────────────────────────────────────────


@pytest.fixture
def moved(frozen, tmp_path, monkeypatch):
    """片 A 跑完后整树挪到新位置（模拟 GL NFS → 本机）。"""
    old = tmp_path / "gl" / "a"
    R.split_v8(frozen, SHARD_A, old, label="a")
    FakeRunner().install(monkeypatch)
    R.run_continue_v8(old / "specs", SHARD_A, old, **continue_kwargs(tmp_path))
    new = tmp_path / "local" / "a"
    new.parent.mkdir()
    shutil.move(str(old), str(new))
    return old, new


def _agg(new, rebase, out):
    return R.aggregate_v8(new / "specs", SHARD_A, [new], out, rebase=R.parse_rebase(rebase))


def test_rebase_rewrites_paths_and_rechecks_sha(moved, tmp_path):
    old, new = moved
    report = _agg(new, [f"{old}={new}"], tmp_path / "d.json")
    assert report["line"].startswith("V8_DELIVERY_SET=PASS ")
    assert all(Path(r["h5"]).is_relative_to(new) for r in report["rows"])
    assert all(Path(r["video"]).is_relative_to(new) for r in report["rows"])


def test_rebase_errors(moved, tmp_path):
    old, new = moved
    no_rebase = R.aggregate_v8(new / "specs", SHARD_A, [new], tmp_path / "plain.json")
    assert {r["problem"] for r in no_rebase["bad_rows"]} == {"h5_missing"}
    unmapped = _agg(new, [f"{tmp_path / 'elsewhere'}={new}"], tmp_path / "u.json")
    assert {r["problem"] for r in unmapped["bad_rows"]} == {"h5_unmapped"}
    h5s = sorted(new.rglob("*.h5"))
    h5s[0].unlink()
    with open(h5s[1], "ab") as fh:
        fh.write(b"\0")
    mixed = _agg(new, [f"{old}={new}"], tmp_path / "m.json")
    assert sorted(r["problem"] for r in mixed["bad_rows"]) == ["h5_missing", "h5_sha_mismatch"]
    assert mixed["counts"]["bad_h5"] == 2 and mixed["line"].startswith("V8_DELIVERY_SET=FAIL ")
    for bad in (["no-equals"], ["=x"], ["x="]):
        with pytest.raises(R.RolloutError, match="--rebase"):
            R.parse_rebase(bad)


def test_rebase_prefers_longest_prefix_and_whole_segments():
    rb = R.parse_rebase(["/a=/x", "/a/b=/y"])
    assert R.rebase_path("/a/b/c.h5", rb) == "/y/c.h5"
    assert R.rebase_path("/a/c.h5", rb) == "/x/c.h5"
    assert R.rebase_path("/ab/c.h5", rb) is None  # 只认整段目录前缀


# ── generate_h5.py：A40 闸门与 aggregate 子命令 ─────────────────────────


def _fake_tools(monkeypatch, gpu_lines: str):
    def run(argv, **kw):
        out = gpu_lines if argv[0] == "nvidia-smi" else "f00dfeed\n"
        return types.SimpleNamespace(stdout=out, returncode=0)

    monkeypatch.setattr(GH, "subprocess", types.SimpleNamespace(run=run))


def _gh(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["generate_h5.py", *argv])
    return GH.main()


def test_generation_refuses_non_a40(monkeypatch, tmp_path):
    _fake_tools(monkeypatch, "NVIDIA RTX 6000 Ada Generation, 550.54\n")
    out = tmp_path / "gen"
    with pytest.raises(SystemExit, match="只许在 A40"):
        _gh(monkeypatch, "--mode", "continue", "--specs", str(tmp_path), "--output", str(out))
    assert not out.exists()


def test_a40_and_dev_smoke_pass_the_gate(monkeypatch, tmp_path):
    # 两张卡：CUDA_VISIBLE_DEVICES=1 取第二行（A40）；过闸后因规格根为空在下一道检查停下，launch 记录已写
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    _fake_tools(monkeypatch, "NVIDIA RTX 6000 Ada Generation, 550.54\nNVIDIA A40, 550.54\n")
    out = tmp_path / "gen"
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="没有任何"):
        _gh(monkeypatch, "--mode", "continue", "--specs", str(tmp_path / "empty"), "--output", str(out))
    (launch,) = out.glob("launch-*.json")
    facts = json.loads(launch.read_text())
    assert (facts["gpu_model"], facts["src_commit"], facts["dev_smoke"]) == ("NVIDIA A40", "f00dfeed", False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit, match="没有任何"):
        _gh(monkeypatch, "--mode", "continue", "--specs", str(tmp_path / "empty"), "--output", str(tmp_path / "g2"),
            "--dev-smoke")


def test_cli_aggregate_exit_codes(frozen, tmp_path, monkeypatch):
    _fake_tools(monkeypatch, "")
    shard = tmp_path / "a"
    R.split_v8(frozen, SHARD_A, shard, label="a")
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(R.cells_json(SHARD_A)))
    (shard / R.LEDGER_NAME).write_text("")
    assert _gh(monkeypatch, "--mode", "aggregate", "--specs", str(shard / "specs"), "--cells", str(cells),
               "--shards", str(shard), "--output", str(tmp_path / "o1")) == 1  # 未跑：pending
    FakeRunner().install(monkeypatch)
    (shard / R.LEDGER_NAME).unlink()
    R.run_continue_v8(shard / "specs", SHARD_A, shard, **continue_kwargs(tmp_path))
    assert _gh(monkeypatch, "--mode", "aggregate", "--specs", str(shard / "specs"), "--cells", str(cells),
               "--shards", str(shard), "--output", str(tmp_path / "o2")) == 0
    report = json.loads((tmp_path / "o2" / "delivery.json").read_text())
    assert report["code_baseline"] == "f00dfeed" and report["cells_source"] == str(cells)
