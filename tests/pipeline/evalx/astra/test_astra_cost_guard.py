"""S5 费用守卫：usage 计价、缺字段告警、账本跨根累计与去重、在途最坏投影、STOP.json 写入位置。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from astra_fakes import load_guard

PRICES = {"unit": "usd_per_1m_tokens", "input": 2.0, "cached_input": 0.5, "output": 8.0}


def _prices(tmp_path, **extra):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({**PRICES, **extra}))
    return load_guard().load_prices(path)


def _call(spool: Path, rid: str, usage=None, *, status="ok", started=True, responded=True):
    out = spool / rid
    out.mkdir(parents=True)
    (out / "request.json").write_text(json.dumps({"id": rid, "images": []}))
    if started:
        (out / "api_started.json").write_text("{}")
    if responded:
        body = {"status": status}
        if usage is not None:
            body["usage"] = usage
        (out / "response.json").write_text(json.dumps(body))
    return out


USAGE = {"input_tokens": 10_000, "output_tokens": 1_000, "input_tokens_details": {"cached_tokens": 4_000},
         "output_tokens_details": {"reasoning_tokens": 600}, "total_tokens": 11_000}


def test_price_usage_hand_computed(tmp_path):
    guard = load_guard()
    usd, tokens, warn = guard.price_usage(USAGE, _prices(tmp_path))
    # (10000-4000)*2 + 4000*0.5 + 1000*8 = 12000 + 2000 + 8000 = 22000 → /1e6
    assert usd == pytest.approx(0.022) and warn == []
    assert tokens == {"input_tokens": 10_000, "output_tokens": 1_000, "cached_tokens": 4_000, "reasoning_tokens": 600}
    usd2, _, _ = guard.price_usage(USAGE, _prices(tmp_path, reasoning_billed_separately=True))
    assert usd2 == pytest.approx(0.022 + 600 * 8 / 1e6)


def test_missing_fields_count_zero_and_warn(tmp_path):
    guard = load_guard()
    usd, tokens, warn = guard.price_usage({"input_tokens": 1_000_000}, _prices(tmp_path))
    assert usd == pytest.approx(2.0)
    assert sorted(warn) == ["input_tokens_details.cached_tokens", "output_tokens", "output_tokens_details.reasoning_tokens"]


def test_scan_counts_only_planner_spool_and_dedupes_across_roots_and_ledger(tmp_path):
    guard = load_guard()
    prices = _prices(tmp_path)
    local = tmp_path / "local" / "group_0" / "run" / "planner_calls"
    gl = tmp_path / "gl" / "group_1" / "run" / "planner_calls"
    _call(local, "a" * 32, USAGE)
    _call(gl, "b" * 32, USAGE)
    _call(gl, "c" * 32, None, status="error")  # HTTP 错误：无 usage
    _call(gl, "d" * 32, None, responded=False)  # 在途
    monitor = tmp_path / "gl" / "group_1" / "run" / "results" / "BinFill" / "ep000" / "monitor_inputs" / "t0016"
    monitor.mkdir(parents=True)
    (monitor / "response.json").write_text(json.dumps({"text": "true"}))  # 监视器响应：旁边无 request.json，不计

    ledger_path = tmp_path / "ledger.json"
    ledger = guard.load_ledger(ledger_path)
    s1 = guard.cycle([tmp_path / "local"], ledger, prices, 30.0, 2048)
    guard.save_ledger(ledger_path, ledger)
    assert s1["calls"] == 1 and s1["usd"] == pytest.approx(0.022)

    # 换一个进程（读回账本）再加上 GL 根：本机那一条不重复计
    ledger2 = guard.load_ledger(ledger_path)
    s2 = guard.cycle([tmp_path / "local", tmp_path / "gl"], ledger2, prices, 30.0, 2048)
    assert s2["calls"] == 3 and s2["usd"] == pytest.approx(0.044)
    assert s2["no_usage"] == 1 and s2["pending"] == 1
    assert s2["worst_request_usd"] == pytest.approx((10_000 * 2 + 2048 * 8) / 1e6)
    assert not s2["stop"] and not list(tmp_path.rglob("STOP.json"))


def test_cap_projection_writes_stop_in_every_group(tmp_path, capsys):
    guard = load_guard()
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(json.dumps(PRICES))
    root = tmp_path / "astra"
    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0},
             "output_tokens_details": {"reasoning_tokens": 0}}
    _call(root / "group_0" / "orig" / "planner_calls", "e" * 32, usage)
    (root / "group_1" / "new").mkdir(parents=True)
    ledger = tmp_path / "ledger.json"
    # 累计 2.0；最坏一次 = 1e6*2/1e6 + 2048*8/1e6 = 2.016384；投影 4.016384
    rc = guard.main(["--root", str(root), "--prices", str(prices_path), "--ledger", str(ledger), "--cap", "4.0", "--once"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ASTRA_COST usd=2.0000 calls=1" in out
    assert "ASTRA_COST=STOP usd=2.0000 cap=4" in out
    for group in ("group_0", "group_1"):
        stop = json.loads((root / group / "STOP.json").read_text())
        assert stop["reason"] == "astra_cost_cap" and stop["projected_usd"] == pytest.approx(4.016384)
    assert json.loads(ledger.read_text())["calls"]["e" * 32]["usd"] == pytest.approx(2.0)


def test_below_cap_no_stop_and_root_inside_group(tmp_path, capsys):
    """根直接给到 spool（group_* 在祖先上）也能定位停机口；未到线不写。"""
    guard = load_guard()
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(json.dumps(PRICES))
    spool = tmp_path / "group_0" / "run" / "planner_calls"
    _call(spool, "f" * 32, USAGE)
    rc = guard.main(["--root", str(spool), "--prices", str(prices_path), "--ledger", str(tmp_path / "l.json"), "--once"])
    assert rc == 0 and "ASTRA_COST=STOP" not in capsys.readouterr().out
    assert not (tmp_path / "group_0" / "STOP.json").exists()
    assert guard.group_dirs([spool]) == [(tmp_path / "group_0").resolve()]


def test_prices_must_be_given(tmp_path):
    guard = load_guard()
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"input": 1.0}))
    with pytest.raises(ValueError, match="output"):
        guard.load_prices(bad)
