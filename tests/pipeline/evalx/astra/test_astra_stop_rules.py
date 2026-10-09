"""Astra stop rules (moved from the old ``run_cases`` into ``AstraPolicy``: judged after play returns, the next
``reset`` raises ``AstraStop``, the outer loop stops the whole batch): a single planner-service error stops; 3
consecutive non-planner errors stop; once the cost cap is hit ``group_0/STOP.json`` is written and no further episode
is started nor request sent.

Verdict line: ``ASTRA_STOP_RULES=PASS cases=3`` (printed by ``test_astra_stop_rules_summary`` once all three rules
pass).
"""
from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message

import pytest

from astra_fakes import FakeEnv, FakeMonitor, FakeResponder, FakeVLA, Harness, NetCounter, astra_session, load_guard

TASKS4 = ["BinFill", "PickXtimes", "VideoUnmask", "MoveCube"]
_PASSED: set = set()


def _astra_result(h, r) -> dict:
    return json.loads((h.astra_ep_dir(r) / "result.json").read_text())


@pytest.mark.parametrize("k", [1, 2])
def test_planner_error_on_episode_k_stops_and_sends_no_more(tmp_path, monkeypatch, k):
    """Rule 5: after the fake planner errors on episode k (``Planner bridge failed`` prefix) the whole batch stops; no
    more planner requests are sent and no more envs are built."""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        responder = FakeResponder(astra.champ, fail_on_episode_index=k)
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=40),
                    responder=responder, max_episodes=10)
        h.load()
        results, stop = h.batch("hard-verify", TASKS4)
        h.policy.close()
    assert isinstance(stop, mod.AstraStop) and stop.reason == "planner_error"
    assert responder.calls == k and h.ran() == TASKS4[:k] and len(results) == k
    failed = _astra_result(h, results[-1])
    assert failed["status"] == "error" and failed["error"].startswith("Planner bridge failed")
    assert results[-1].status == "fail" and results[-1].error_kind == "error"
    assert net.calls == 0
    _PASSED.add("planner_error")


def test_planner_call_limit_prefix_stops(tmp_path, monkeypatch):
    """Rule 5: ``Pilot planner-call`` prefix: hitting the per-episode planner-call cap stops the whole batch."""
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=None),
                    monitor=FakeMonitor(predictions=[True] * 4), max_planner_calls=1)
        h.load()
        results, stop = h.batch("hard-verify", TASKS4[:2])
        h.policy.close()
    assert stop.reason == "planner_error" and h.responder.calls == 1 and h.ran() == TASKS4[:1]
    assert _astra_result(h, results[0])["error"].startswith("Pilot planner-call")


def test_real_responses_client_http_error_stops_after_one_offline_attempt(tmp_path, monkeypatch):
    """Real upstream ``ResponsesClient`` (fake key) + urlopen fake raising HTTP 500: only one attempt, no network
    access, then the whole batch stops."""
    def http_500():
        headers = Message()
        return urllib.error.HTTPError("https://example.invalid", 500, "fake", headers, io.BytesIO(b'{"error":{}}'))

    net = NetCounter(raise_exc=http_500).install(monkeypatch)
    with astra_session() as (mod, astra):
        client = astra.api_client.ResponsesClient("sk-test-placeholder-not-a-real-key")
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=40),
                    responder=client)
        policy = h.load()
        results, stop = h.batch("hard-verify", TASKS4[:3])
        policy.close()
    assert stop.reason == "planner_error"
    assert net.calls == 1 and h.ran() == TASKS4[:1]
    (call,) = list(policy.spool.iterdir())
    assert json.loads((call / "response.json").read_text())["status"] == "error"


def test_three_consecutive_infra_errors_stop(tmp_path, monkeypatch):
    """Rule 4: 3 consecutive non-planner errors stop; an episode that ends normally in between resets the count (same
    convention as upstream main()). Error sources: simulator errors, VLA disconnects."""
    net = NetCounter().install(monkeypatch)
    tasks = ["BinFill", "PickXtimes", "VideoUnmask", "MoveCube", "InsertPeg", "PatternLock"]
    plan = {"BinFill": "step_error", "PickXtimes": "ok", "VideoUnmask": "step_error", "MoveCube": "step_error",
            "InsertPeg": "step_error", "PatternLock": "ok"}

    def env_plan(builder, ep):
        return FakeEnv(terminal_step=40, step_error_at=5 if plan[builder.env_id] == "step_error" else None)

    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=env_plan, max_episodes=10)
        h.load()
        results, stop = h.batch("hard-verify", tasks)
        errors_seen = h.policy._errors
        h.policy.close()
    assert stop.reason == "infra_errors" and errors_seen == 3
    assert h.ran() == tasks[:5], "PatternLock must not start after the 3rd consecutive error (InsertPeg)"
    assert [_astra_result(h, r)["status"] for r in results] == ["error", "success", "error", "error", "error"]
    assert [r.status for r in results] == ["fail", "success", "fail", "fail", "fail"]
    assert net.calls == 0
    _PASSED.add("infra_errors")


def test_vla_disconnect_counts_as_infra_error(tmp_path, monkeypatch):
    """A VLA disconnect (``runner.episode`` records error) counts as one non-planner error; a single one does not
    stop, the Policy continues with the next episode."""
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=40),
                    vla=FakeVLA(fail_at=2))
        h.load()
        results, stop = h.batch("hard-verify", ["BinFill", "VideoUnmask"])
        h.policy.close()
    assert stop is None and [r.status for r in results] == ["fail", "success"]
    assert "fake VLA websocket dropped" in results[0].error


def test_cost_cap_writes_stop_and_no_more_requests(tmp_path, monkeypatch):
    """The cost guard writes ``group_0/STOP.json`` at the cap: the next ``reset`` stops the whole batch (no env
    built), and the real ``ResponsesClient`` under ``group_0`` also refuses to send. The guard is the real ``cycle``
    (inflated unit prices, cap 30 USD, single call 20.08 USD, projection 40.16 > 30)."""
    net = NetCounter().install(monkeypatch)
    guard = load_guard()
    big = {"input_tokens": 2_000_000, "output_tokens": 2000, "input_tokens_details": {"cached_tokens": 0},
           "output_tokens_details": {"reasoning_tokens": 1500}}
    prices_doc = {"unit": "usd_per_1m_tokens", "input": 10.0, "cached_input": 1.0, "output": 40.0}
    summaries = []
    with astra_session() as (mod, astra):
        state = {}

        def after_write(out):
            summaries.append(guard.cycle([state["root"]], state["ledger"], state["prices"], 30.0, 2048))

        responder = FakeResponder(astra.champ, usage=big, after_write=after_write)
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=40),
                    responder=responder, prices=prices_doc)
        state.update(root=h.group, ledger=guard.new_ledger(), prices=guard.load_prices(h.prices_path))
        policy = h.load()
        results, stop = h.batch("hard-verify", TASKS4[:3])
        out = policy.spool / "manual"
        out.mkdir()
        (out / "request.json").write_text(json.dumps({"images": []}))
        (out / "prompt.txt").write_text("x")
        astra.api_client.ResponsesClient("sk-test-placeholder-not-a-real-key")(out)
        refused = json.loads((out / "response.json").read_text())
        policy.close()
    assert stop.reason == "host_stop" and len(results) == 1
    assert responder.calls == 1 and h.ran() == TASKS4[:1]
    assert summaries[0]["stop"] and summaries[0]["usd"] == pytest.approx(20.08)
    stop_doc = json.loads((h.group / "STOP.json").read_text())
    assert stop_doc["reason"] == "astra_cost_cap" and stop_doc["cap"] == 30.0
    assert refused["status"] == "error" and "Host requested stop" in refused["error"]
    assert net.calls == 0
    _PASSED.add("cost_stop")


def test_astra_stop_rules_summary():
    """Kept at the end of this file: prints the verdict line once all three stop-rule cases pass (fails when run
    alone, as a hint to run the whole file first)."""
    assert _PASSED == {"planner_error", "infra_errors", "cost_stop"}, _PASSED
    print(f"ASTRA_STOP_RULES=PASS cases={len(_PASSED)}")
