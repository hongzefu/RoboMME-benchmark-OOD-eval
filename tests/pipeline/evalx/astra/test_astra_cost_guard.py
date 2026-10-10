"""S5 cost guard: usage pricing, missing-field warnings, cross-root ledger accumulation and dedup, worst-case
projection of in-flight calls, and where STOP.json is written."""
from __future__ import annotations

import json
import subprocess
import sys
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
    # (10000-4000)*2 + 4000*0.5 + 1000*8 = 12000 + 2000 + 8000 = 22000 -> /1e6
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
    _call(gl, "c" * 32, None, status="error")  # HTTP error: no usage
    _call(gl, "d" * 32, None, responded=False)  # in flight
    monitor = tmp_path / "gl" / "group_1" / "run" / "results" / "BinFill" / "ep000" / "monitor_inputs" / "t0016"
    monitor.mkdir(parents=True)
    (monitor / "response.json").write_text(json.dumps({"text": "true"}))  # monitor response: no request.json next to it, not counted

    ledger_path = tmp_path / "ledger.json"
    ledger = guard.load_ledger(ledger_path)
    s1 = guard.cycle([tmp_path / "local"], ledger, prices, 30.0, 2048)
    guard.save_ledger(ledger_path, ledger)
    assert s1["calls"] == 1 and s1["usd"] == pytest.approx(0.022)

    # New process (ledger read back) plus the GL root: the local call is not counted twice
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
    # Accumulated 2.0; worst single call = 1e6*2/1e6 + 2048*8/1e6 = 2.016384; projection 4.016384
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
    """With the root given directly as the spool (group_* is an ancestor) the stop location is still found; nothing
    is written below the cap."""
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


@pytest.mark.parametrize("value", [1, 2, 3])
def test_explicit_episode_limit_accepts_only_authorized_integers(value):
    """Explicit limits allow one to three episodes; the hard cost cap remains five USD."""
    guard = load_guard()
    assert guard.check_max_episodes(value) == value
    assert guard.ASTRA_MAX_EPISODES == 2 and guard.HARD_MAX_EPISODES == 3
    assert guard.HARD_CAP_USD == 5.0


@pytest.mark.parametrize("value", [0, -1, 4, True, False, 1.0, 2.5, "3", None])
def test_episode_limit_rejects_invalid_type_or_range(value):
    """Reject booleans, floats, strings and non-positive values without coercion."""
    with pytest.raises(ValueError, match="ASTRA_COST_BLOCKED.*max-episodes"):
        load_guard().check_max_episodes(value)


@pytest.mark.parametrize("value", [0, -1, 4, True, False, 1.0, 2.5, "3"])
@pytest.mark.parametrize("entry", ["guard_round", "write_state"])
def test_invalid_explicit_episode_limit_has_no_disk_side_effect(tmp_path, value, entry):
    """Reject invalid configuration before creating locks, state, cost files or stop markers."""
    guard = load_guard()
    root = tmp_path / "absent"
    state_path = root / "guard.state.json"
    ledger = guard.new_ledger()
    summary = guard.cycle([], ledger, PRICES, 5.0, 2048)
    with pytest.raises(ValueError, match="max-episodes"):
        if entry == "guard_round":
            guard.guard_round([root], ledger, PRICES, 5.0, 2048, state_path, max_episodes=value)
        else:
            guard.write_state(state_path, summary, ledger, PRICES, 2048, max_episodes=value)
    assert not root.exists() and ledger["calls"] == {}


@pytest.mark.parametrize("limit", [None, 3])
def test_heartbeat_and_final_state_keep_requested_episode_limit(tmp_path, limit):
    """Read back persisted states: default two or explicit three applies to heartbeat and final exit."""
    guard = load_guard()
    state_path = tmp_path / "guard.state.json"
    ledger = guard.new_ledger()
    summary = guard.guard_round([], ledger, PRICES, 5.0, 2048, state_path, clock=lambda: 10.0,
                                max_episodes=limit)
    heartbeat = json.loads(state_path.read_text())
    assert heartbeat["max_episodes"] == (2 if limit is None else 3)
    assert heartbeat["heartbeat"] == 10.0 and heartbeat["exited"] is False
    guard.write_state(state_path, summary, ledger, PRICES, 2048, exited=True, clock=lambda: 11.0,
                      max_episodes=limit)
    final = json.loads(state_path.read_text())
    assert final["max_episodes"] == heartbeat["max_episodes"]
    assert final["heartbeat"] == 11.0 and final["exited"] is True


def test_omitted_episode_limit_reads_dynamic_default(tmp_path, monkeypatch):
    """Preserve fixture overrides of the default constant instead of freezing it at function definition."""
    guard = load_guard()
    ledger = guard.new_ledger()
    state_path = tmp_path / "guard.state.json"
    for limit in (3, 10):
        monkeypatch.setattr(guard, "ASTRA_MAX_EPISODES", limit)
        guard.guard_round([], ledger, PRICES, 5.0, 2048, state_path)
        assert json.loads(state_path.read_text())["max_episodes"] == limit


@pytest.mark.parametrize("value", ["0", "-1", "4", "true", "2.5"])
def test_main_invalid_episode_limit_does_not_create_files(tmp_path, value):
    """Invalid CLI arguments fail before reading prices or writing files."""
    guard = load_guard()
    root = tmp_path / "absent"
    argv = ["--root", str(root), "--prices", str(root / "prices.json"), "--ledger", str(root / "ledger.json"),
            "--max-episodes", value, "--once"]
    if value in ("true", "2.5"):
        with pytest.raises(SystemExit) as exc:
            guard.main(argv)
        assert exc.value.code == 2
    else:
        assert guard.main(argv) == 2
    assert not root.exists()


@pytest.mark.parametrize("limit", [None, 3])
def test_main_once_serializes_default_or_explicit_episode_limit(tmp_path, limit):
    """The real single-round entry preserves the default two or explicit three in its state."""
    guard = load_guard()
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps(PRICES))
    ledger_path = tmp_path / "ledger.json"
    argv = ["--root", str(tmp_path), "--prices", str(price_path), "--ledger", str(ledger_path), "--once"]
    if limit is not None:
        argv += ["--max-episodes", str(limit)]
    assert guard.main(argv) == 0
    state = json.loads(guard.default_state_path(ledger_path).read_text())
    assert state["max_episodes"] == (2 if limit is None else 3)
    assert state["cap"] == 5.0 and state["exited"] is False


def test_main_shutdown_final_state_preserves_explicit_three(tmp_path, monkeypatch):
    """A simulated stop after the first round persists the explicit three on normal exit."""
    guard = load_guard()
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps(PRICES))
    ledger_path = tmp_path / "ledger.json"
    handlers = {}
    monkeypatch.setattr(guard.signal, "signal", lambda signum, handler: handlers.update({signum: handler}))
    original_round = guard.guard_round
    heartbeats = []

    def round_then_stop(*args, **kwargs):
        summary = original_round(*args, **kwargs)
        heartbeats.append(json.loads(guard.default_state_path(ledger_path).read_text()))
        handlers[guard.signal.SIGTERM](guard.signal.SIGTERM, None)
        return summary

    monkeypatch.setattr(guard, "guard_round", round_then_stop)
    assert guard.main(["--root", str(tmp_path), "--prices", str(price_path), "--ledger", str(ledger_path),
                       "--max-episodes", "3"]) == 0
    final = json.loads(guard.default_state_path(ledger_path).read_text())
    assert len(heartbeats) == 1 and heartbeats[0]["max_episodes"] == final["max_episodes"] == 3
    assert heartbeats[0]["exited"] is False and final["exited"] is True


def test_guard_process_restart_keeps_same_cumulative_ledger(tmp_path):
    """Two independent guard processes reuse one ledger without clearing episodes, reservations or costs."""
    guard = load_guard()
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps(PRICES))
    spool = tmp_path / "group_0" / "planner_calls"
    _call(spool, "first", USAGE)
    ledger_path = tmp_path / "ledger.json"
    state_path = guard.default_state_path(ledger_path)
    reservations = {"schema": guard.RESERVATIONS_SCHEMA,
                    "episodes": ["off", "on", "formal"],
                    "reservations": {"first": {"usd": 0.1, "state": "sent"},
                                     "pending": {"usd": 0.25, "state": "reserved"}}}
    guard.save_reservations(state_path, reservations)
    reservation_bytes = guard.reservations_path(state_path).read_bytes()
    command = [sys.executable, "-m", "robomme_ood_eval.servers.astra_cost_guard", "--root", str(spool),
               "--prices", str(price_path), "--ledger", str(ledger_path), "--max-episodes", "3", "--cap", "5",
               "--once"]
    for expected_calls in (1, 2):
        if expected_calls == 2:
            _call(spool, "second", USAGE)
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert completed.returncode == 0, completed.stderr
        ledger = guard.load_ledger(ledger_path)
        assert len(ledger["calls"]) == expected_calls
        state = json.loads(state_path.read_text())
        assert state["max_episodes"] == 3 and state["cap"] == 5.0
        assert state["committed_usd"] == pytest.approx(0.022 * expected_calls)
        assert state["projected_usd"] == pytest.approx(0.022 * expected_calls + 0.25)
        assert guard.reservations_path(state_path).read_bytes() == reservation_bytes
        assert guard.load_reservations(state_path) == reservations
    print("ASTRA_EPISODE_LIMIT=PASS default=2 explicit=3 hard_max=3 restart_calls=2 reservations_preserved=1")
