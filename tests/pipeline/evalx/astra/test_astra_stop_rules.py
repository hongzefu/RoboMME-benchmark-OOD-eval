"""S5 停机规则：规划服务出错单次即停、连续 3 局基础设施错误即停、费用到线写 STOP.json 后不再发请求。

判定行：``ASTRA_STOP_RULES=PASS cases=3``（由 ``test_astra_stop_rules_summary`` 在三条规则全部通过后打印）。
"""
from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from astra_fakes import (FakeEnv, FakeMonitor, FakeResponder, FakeVLA, NetCounter, astra_session, load_guard,
                         make_args, make_deps, recording_builder_cls, write_cases)

TASKS4 = ["BinFill", "PickXtimes", "VideoUnmask", "MoveCube"]
_PASSED: set = set()


class _NullWriter:
    def append_data(self, frame):
        pass

    def close(self):
        pass


def _setup(mod, astra, tmp_path, monkeypatch, tasks, env_plan=None):
    monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
    cls = recording_builder_cls(env_plan)
    doc = mod.prepare_cases(cls, "hard-verify", tasks, source_episodes=[3])
    args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
    return cls, args


def _ran(cls) -> list[str]:
    return [c["env_id"] for c in cls.make_calls]


@pytest.mark.parametrize("k", [1, 2])
def test_planner_error_on_episode_k_stops_and_sends_no_more(tmp_path, monkeypatch, k):
    """⑤ 第 k 局规划替身报错（``Planner bridge failed`` 前缀）后整个分片停，不再发任何规划请求。"""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        cls, args = _setup(mod, astra, tmp_path, monkeypatch, TASKS4, lambda b, ep: FakeEnv(terminal_step=40))
        responder = FakeResponder(astra.champ, fail_on_episode_index=k)
        with pytest.raises(mod.AstraStop) as info:
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder,
                                          check_calls=[]))
    assert info.value.reason == "planner_error"
    assert responder.calls == k and _ran(cls) == TASKS4[:k]
    failed = json.loads((Path(args.output) / TASKS4[k - 1] / "ep000" / "result.json").read_text())
    assert failed["status"] == "error" and failed["error"].startswith("Planner bridge failed")
    assert not (Path(args.output) / "PILOT_FINISHED.json").exists()
    assert net.calls == 0
    _PASSED.add("planner_error")


def test_planner_call_limit_prefix_stops(tmp_path, monkeypatch):
    """⑤ ``Pilot planner-call`` 前缀：单局规划次数到上限即停整个分片。"""
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        cls, args = _setup(mod, astra, tmp_path, monkeypatch, TASKS4[:2], lambda b, ep: FakeEnv(terminal_step=None))
        args.max_planner_calls = 1
        responder = FakeResponder(astra.champ)
        with pytest.raises(mod.AstraStop) as info:
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(predictions=[True] * 4), vla=FakeVLA(),
                                          responder=responder, check_calls=[]))
    assert info.value.reason == "planner_error" and responder.calls == 1 and _ran(cls) == TASKS4[:1]


def test_real_responses_client_http_error_stops_after_one_offline_attempt(tmp_path, monkeypatch):
    """真实 ``ResponsesClient``（假密钥）+ urlopen 替身抛 HTTP 500：只尝试一次、不外联，分片随即停。"""
    def http_500():
        headers = Message()
        return urllib.error.HTTPError("https://example.invalid", 500, "fake", headers, io.BytesIO(b'{"error":{}}'))

    net = NetCounter(raise_exc=http_500).install(monkeypatch)
    with astra_session() as (mod, astra):
        cls, args = _setup(mod, astra, tmp_path, monkeypatch, TASKS4[:3], lambda b, ep: FakeEnv(terminal_step=40))
        client = astra.api_client.ResponsesClient("sk-test-placeholder-not-a-real-key")
        with pytest.raises(mod.AstraStop) as info:
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=client,
                                          check_calls=[]))
    assert info.value.reason == "planner_error"
    assert net.calls == 1 and _ran(cls) == TASKS4[:1]
    (call,) = list(Path(args.spool).iterdir())
    assert json.loads((call / "response.json").read_text())["status"] == "error"


def test_three_consecutive_infra_errors_stop(tmp_path, monkeypatch):
    """④ 非规划类 error 连续 3 局即停；中间一局正常结束会清零（上游 main() 同口径）。"""
    net = NetCounter().install(monkeypatch)
    tasks = ["BinFill", "PickXtimes", "VideoUnmask", "MoveCube", "InsertPeg", "PatternLock"]
    plan = {"BinFill": "raise", "PickXtimes": "ok", "VideoUnmask": "raise", "MoveCube": "step_error",
            "InsertPeg": "raise", "PatternLock": "ok"}

    def env_plan(builder, ep):
        mode = plan[builder.env_id]
        if mode == "raise":
            raise RuntimeError("fake simulator start failure")
        return FakeEnv(terminal_step=40, step_error_at=5 if mode == "step_error" else None)

    with astra_session() as (mod, astra):
        cls, args = _setup(mod, astra, tmp_path, monkeypatch, tasks, env_plan)
        responder = FakeResponder(astra.champ)
        with pytest.raises(mod.AstraStop) as info:
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder,
                                          check_calls=[]))
    assert info.value.reason == "infra_errors"
    assert _ran(cls) == tasks[:5], "第 3 次连续错误（InsertPeg）后不得再开 PatternLock"
    statuses = [json.loads((Path(args.output) / t / "ep000" / "result.json").read_text())["status"] for t in tasks[:5]]
    assert statuses == ["error", "success", "error", "error", "error"]
    assert not (Path(args.output) / "PatternLock").exists()
    assert net.calls == 0
    _PASSED.add("infra_errors")


def test_cost_cap_writes_stop_and_no_more_requests(tmp_path, monkeypatch):
    """费用守卫到线写 ``group_0/STOP.json``：驱动下一局前停（③），``ResponsesClient`` 也拒绝再发。"""
    net = NetCounter().install(monkeypatch)
    guard = load_guard()
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(json.dumps({"unit": "usd_per_1m_tokens", "input": 10.0, "cached_input": 1.0, "output": 40.0}))
    prices = guard.load_prices(prices_path)
    ledger = guard.new_ledger()
    root = tmp_path / "group_0"
    big = {"input_tokens": 2_000_000, "output_tokens": 2000, "input_tokens_details": {"cached_tokens": 0},
           "output_tokens_details": {"reasoning_tokens": 1500}}
    summaries = []
    with astra_session() as (mod, astra):
        cls, args = _setup(mod, astra, tmp_path, monkeypatch, TASKS4[:3], lambda b, ep: FakeEnv(terminal_step=40))
        responder = FakeResponder(astra.champ, usage=big,
                                  after_write=lambda out: summaries.append(guard.cycle([root], ledger, prices, 30.0, 2048)))
        with pytest.raises(mod.AstraStop) as info:
            mod.run_cases(args, make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder,
                                          check_calls=[]))
        # 停机后真实 ResponsesClient 在 group_0 下也拒绝发送（不触网）
        out = Path(args.spool) / "manual"
        out.mkdir()
        (out / "request.json").write_text(json.dumps({"images": []}))
        (out / "prompt.txt").write_text("x")
        astra.api_client.ResponsesClient("sk-test-placeholder-not-a-real-key")(out)
        refused = json.loads((out / "response.json").read_text())
    assert info.value.reason == "host_stop"
    assert responder.calls == 1 and _ran(cls) == TASKS4[:1]
    assert summaries[0]["stop"] and summaries[0]["usd"] == pytest.approx(20.08)
    stop = json.loads((root / "STOP.json").read_text())
    assert stop["reason"] == "astra_cost_cap" and stop["cap"] == 30.0
    assert refused["status"] == "error" and "Host requested stop" in refused["error"]
    assert net.calls == 0
    _PASSED.add("cost_stop")


def test_astra_stop_rules_summary():
    """放在本文件末尾：三条停机规则的用例都通过后打印判定行（单独运行本用例时会失败，提示先跑全文件）。"""
    assert _PASSED == {"planner_error", "infra_errors", "cost_stop"}, _PASSED
    print(f"ASTRA_STOP_RULES=PASS cases={len(_PASSED)}")
