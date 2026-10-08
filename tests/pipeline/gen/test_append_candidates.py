"""C12「追加」：某格备用耗尽后 ``append_candidates.py`` 单格补抽，再 ``run_continue_v8 --resume`` 续跑交付。

链路全部真实：冻结 → 分片 → 片内续跑（FakeRunner）到 exhausted → 补抽（``draw_one`` 注入 CPU 替身、
``sampling_check`` 注入空核对）→ 写回两份规格与 ``shard.json`` → 续跑只执行新递补的一局。期望手写。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_world import H, R, FakeDraw, FakeRunner, continue_kwargs, freeze_root, read_rows, state
from tests._support.loaders import load_script

AP = load_script("injection-dev/append_candidates.py")
S = "StopCube"
CELLS = {(S, "xhard1"): 2}


@pytest.fixture
def exhausted(tmp_path, monkeypatch):
    """配额 2、候选 3；候选 0 失败递补 2，2 也失败 → 该格 exhausted（交付 1）。"""
    frozen = freeze_root(tmp_path / "frozen", {(S, "xhard1"): (2, 3)})
    shard = tmp_path / "gen1" / "a"
    R.split_v8(frozen, CELLS, shard, label="a")
    FakeRunner({(S, 0): ["fail"], (S, 2): ["fail"]}).install(monkeypatch)
    R.run_continue_v8(shard / "specs", CELLS, shard, **continue_kwargs(tmp_path))
    cell = json.loads((shard / "delivery.json").read_text())["cells"][f"{S}/xhard1"]
    assert (cell["status"], cell["reason"]) == ("FAIL", "exhausted")
    return frozen, shard


def _main(frozen, shard, *extra, draw=None):
    argv = ["--frozen-root", str(frozen), "--shard-dir", str(shard), "--cell", f"{S}@xhard1", "--extra", "2",
            "--max-reset-attempts", "5", *extra]
    return AP.main(argv, draw_one=draw or FakeDraw("xhard1"), sampling_check=lambda task, sampling: None)


def _line(capsys):
    return [l for l in capsys.readouterr().out.splitlines() if l.startswith("APPEND_CANDIDATES=")][-1]


def test_append_then_resume_delivers(exhausted, monkeypatch, capsys):
    frozen, shard = exhausted
    f_path = frozen / "xhard1" / "specs.jsonl"
    old_lines = f_path.read_text().splitlines()[1:]
    old_identity = read_rows(f_path)[0]["identity_sha256"]
    draw = FakeDraw("xhard1", fail={(S, 3): 1})  # 新候选 3 第一次 reset 被拒
    assert _main(frozen, shard, draw=draw) == 0
    line = _line(capsys)
    assert line.startswith("APPEND_CANDIDATES=PASS ") and "per_env=3→5" in line and "backfill_selected=1" in line
    assert [(c[2], c[3]) for c in draw.calls] == [(3, 0), (3, 1), (4, 0)]
    # 冻结根：旧行逐字保留，末尾追加两行未选候选；签随 per_env 变化重算
    f_header, f_rows = H.load_specs(f_path, check_fingerprint=False)
    assert f_path.read_text().splitlines()[1:4] == old_lines
    assert [(r["candidate"], r["attempt"], r["selected"], r["tried"]) for r in f_rows[3:]] == [(3, 1, False, False),
                                                                                           (4, 0, False, False)]
    ok_seeds = {c[2]: c[1] for c in draw.calls if (c[2], c[3]) != (3, 0)}
    assert [r["seed"] for r in f_rows[3:]] == [ok_seeds[3], ok_seeds[4]]
    assert f_header["per_env"][S] == 5 and f_header["identity_sha256"] != old_identity
    # 片：缺额 1 → 编号最小的新候选 3 标 selected；来源 identity 跟上冻结根
    assert state(shard / "specs" / "xhard1" / "specs.jsonl")[(S, 3)] == (True, False, None)
    assert state(shard / "specs" / "xhard1" / "specs.jsonl")[(S, 4)] == (False, False, None)
    meta = json.loads((shard / R.SHARD_META).read_text())
    assert meta["sources"]["xhard1"]["identity_sha256"] == f_header["identity_sha256"]
    assert list(f_path.parent.glob("specs.jsonl.pre-append-*"))  # 写前备份
    fake = FakeRunner().install(monkeypatch)
    summary = R.run_continue_v8(shard / "specs", CELLS, shard, resume=True, **continue_kwargs(shard.parent))
    assert fake.executions == [(S, 3, "ok")]
    assert summary["delivery_set"].startswith("V8_DELIVERY_SET=PASS ")
    assert list(shard.glob("delivery.json.prev-*"))  # 上一份交付清单改名留证


def test_dry_run_writes_nothing(exhausted, capsys):
    frozen, shard = exhausted
    before = {p: p.read_bytes() for p in [frozen / "xhard1" / "specs.jsonl", shard / "specs" / "xhard1" / "specs.jsonl",
                                          shard / R.SHARD_META]}
    draw = FakeDraw("xhard1")
    assert _main(frozen, shard, "--dry-run", draw=draw) == 0
    assert "APPEND_DRY_RUN=PASS" in capsys.readouterr().out
    assert draw.calls == [] and all(p.read_bytes() == b for p, b in before.items())


def test_refuses_when_spares_or_pending_remain(tmp_path, capsys):
    frozen = freeze_root(tmp_path / "frozen", {(S, "xhard1"): (2, 3)})
    shard = tmp_path / "gen1" / "a"
    R.split_v8(frozen, CELLS, shard, label="a")
    before = (shard / "specs" / "xhard1" / "specs.jsonl").read_bytes()
    draw = FakeDraw("xhard1")
    assert _main(frozen, shard, draw=draw) == 1  # 刚切片：selected 未跑
    assert "APPEND_CANDIDATES=FAIL" in _line(capsys)
    assert draw.calls == [] and (shard / "specs" / "xhard1" / "specs.jsonl").read_bytes() == before


@pytest.mark.parametrize("breaker, needle", [
    ("lock", "lock 存在"),
    ("cell", "不是 v8 交付格"),
    ("budget", "须为正整数"),
])
def test_append_preconditions(exhausted, capsys, breaker, needle):
    frozen, shard = exhausted
    extra = []
    if breaker == "lock":
        Path(str(frozen / "xhard1" / "specs.jsonl") + ".lock").write_text("{}")
    argv = ["--frozen-root", str(frozen), "--shard-dir", str(shard), "--cell",
            "MoveCube@xhard1" if breaker == "cell" else f"{S}@xhard1", "--extra", "2",
            "--max-reset-attempts", "0" if breaker == "budget" else "5", *extra]
    draw = FakeDraw("xhard1")
    assert AP.main(argv, draw_one=draw, sampling_check=lambda t, s: None) == 1
    out = capsys.readouterr().out
    assert needle in out and draw.calls == []
