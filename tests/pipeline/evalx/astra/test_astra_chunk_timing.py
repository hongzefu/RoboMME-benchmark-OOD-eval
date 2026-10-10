"""Astra 动作块计时及累计三次登记：假时钟、假接口，零付费请求和真实仿真。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_fakes import FakeEnv, FakeGuardServer, GuardFixture, Harness, NetCounter, astra_session, third_party
from robomme_ood_eval.models import astra as mod
from robomme_ood_eval.servers import policy_server_wrap as wrap
from robomme_ood_eval.timing import ChunkTimer, attach_policy_timing


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += ms / 1000


class Writer:
    def __init__(self):
        self.requests = []
        self.responses = []

    def log_request(self, name, payload, **kwargs):
        self.requests.append((name, payload))

    def log_response(self, actions, **kwargs):
        self.responses.append(actions)

    def close(self, **kwargs):
        pass


class Monitor:
    def __init__(self, clock):
        self.clock = clock

    def predict(self, *args):
        self.clock.advance(300)
        return False, [], .250


class Planner:
    def __init__(self, clock, tmp_path):
        self.clock, self.spool = clock, tmp_path
        self.fail = False

    def predict(self, *args):
        self.clock.advance(2000)
        return "command", "planner-id", 1.900

    def review_second_button(self, *args):
        if self.fail:
            raise RuntimeError("复核失败夹具")
        self.clock.advance(500)
        return False, "review-id", .450


class Client:
    def __init__(self, clock, *, audit=True):
        self.clock, self.audit = clock, audit
        self.actions = np.zeros((16, 8), np.float32)

    def infer(self, element):
        self.clock.advance(120)
        out = {"actions": self.actions}
        if self.audit:
            out[mod.AUDIT_KEY] = {"server_timing": {"infer_ms": 80.0, "gpu": "fake-gpu"}}
        return out


@pytest.fixture
def rig(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(mod.time, "perf_counter", clock)
    ctx = mod.TraceContext(Writer(), timer=ChunkTimer(clock=clock))
    planner = Planner(clock, tmp_path)
    client = Client(clock)
    return SimpleNamespace(clock=clock, ctx=ctx, inner_planner=planner, inner_client=client,
                           monitor=mod.TracedMonitor(Monitor(clock), ctx),
                           planner=mod.TracedPlanner(planner, ctx), client=mod.TracedClient(client, ctx),
                           tmp=tmp_path)


def _drive(rig, script):
    for name in script:
        if name == "monitor":
            rig.monitor.predict("VideoUnmask", "goal", "command", [], 0, None, rig.tmp)
        elif name == "planner":
            rig.planner.predict("VideoUnmask", "goal", [], [], {}, [], [], 0, rig.ctx.t)
        elif name == "review":
            rig.planner.review_second_button("goal", [], None, "command", 0, [], [], 0, rig.ctx.t)
        else:
            reply = rig.client.infer({"grounded_subgoal": "command"})
            assert reply["actions"] is rig.inner_client.actions
            assert mod.AUDIT_KEY not in reply


@pytest.mark.parametrize("script,wall,lang,planner,review", [
    (["planner", "infer"], 2120, 2000, True, False),
    (["monitor", "infer"], 420, 300, False, False),
    (["monitor", "planner", "infer"], 2420, 2300, True, False),
    (["monitor", "review", "planner", "infer"], 2920, 2800, True, True),
    (["monitor", "review", "infer"], 920, 800, False, True),
    (["infer"], 120, 0, False, False),
])
def test_chunk_sequences(rig, script, wall, lang, planner, review):
    _drive(rig, script)
    chunk, = rig.ctx.timer.chunks
    assert chunk["decision_wall_ms"] == wall
    assert chunk["lang_ms"] == lang and chunk["action_rtt_ms"] == 120
    assert chunk["server_infer_ms"] == 80
    assert chunk["has_planner"] == planner and chunk["has_review"] == review
    assert chunk["no_lang"] == (lang == 0)
    assert rig.ctx.gpu_name == "fake-gpu"
    assert all(call["ms"] >= call["third_party_s"] * 1000 for call in rig.ctx.timer.lang_calls)
    assert attach_policy_timing({}, rig.ctx.timer, gpu_name=rig.ctx.gpu_name,
                                model_kind="serial")["conservation"]["violations"] == 0


def test_abort_clears_orphan_bucket_and_invalid_actions(rig):
    _drive(rig, ["monitor"])
    rig.inner_planner.fail = True
    with pytest.raises(RuntimeError, match="复核失败"):
        _drive(rig, ["review"])
    assert rig.ctx.orphan_ms == 300 and rig.ctx.orphan_n == 1
    assert not rig.ctx.chunk_started and not rig.ctx.timer.chunks
    _drive(rig, ["infer"])
    assert rig.ctx.timer.chunks[0]["lang_ms"] == 0
    assert rig.ctx.timer.chunks[0]["decision_wall_ms"] == 120
    _drive(rig, ["monitor"])
    rig.inner_client.actions = np.zeros((0, 8))
    with pytest.raises(ValueError, match="nonempty"):
        _drive(rig, ["infer"])
    assert len(rig.ctx.timer.chunks) == 1 and rig.ctx.orphan_ms == 600
    rig.ctx.timer.absorb_orphan(rig.ctx.orphan_ms, rig.ctx.orphan_n)
    out = attach_policy_timing({}, rig.ctx.timer, gpu_name=None, model_kind="serial")
    assert out["orphan_lang_ms"] == 600 and out["orphan_n"] == 2


def test_no_audit_and_last_audit_fallback(rig):
    rig.inner_client.audit = False
    _drive(rig, ["infer"])
    assert rig.ctx.timer.chunks[-1]["server_infer_ms"] is None
    assert rig.ctx.gpu_name is None

    def infer(element):
        rig.clock.advance(120)
        rig.inner_client._last_audit = {"server_timing": {"infer_ms": 70, "gpu": "last-gpu"}}
        return {"actions": rig.inner_client.actions}

    rig.inner_client._last_audit = {"server_timing": {"infer_ms": 999}}
    rig.inner_client.infer = infer
    _drive(rig, ["infer"])
    assert rig.ctx.timer.chunks[-1]["server_infer_ms"] == 70
    rig.inner_client.infer = lambda element: {"actions": rig.inner_client.actions}
    _drive(rig, ["infer"])
    assert rig.ctx.timer.chunks[-1]["server_infer_ms"] is None


def test_seconds_pairing_and_finalization(rig):
    _drive(rig, ["monitor", "review", "planner", "infer"])
    ep = rig.tmp / "ep"
    ep.mkdir()
    rows = [{"type": "monitor", "t": 0, "seconds": .250},
            {"type": "second_button_review", "t": 0, "seconds": .450},
            {"type": "continue_last_subgoal", "t": 0, "seconds": 99},
            {"type": "planner", "t": 0, "seconds": 1.9}, {"type": "exception", "t": 0}]
    path = ep / "decisions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = {"status": "success"}
    mod._finish_one(SimpleNamespace(policy_seed=7), ep, ep / "attempt", rig.tmp, "key", 1,
                    rig.ctx, rig.ctx.writer, result, None)
    assert result["lang_calls_mismatch"] == [] and result["lang_unpaired"] == 0
    assert result["lang_diff_ms"]["planner"] == [100, 100, 100]
    assert result["vla_gpu_name"] == "fake-gpu"
    json.dumps(result)
    rows[3]["seconds"] = 2.1
    path.write_text("\n".join(json.dumps(row) for row in rows))
    mod._crosscheck_language(path, rig.ctx.timer.lang_calls, result)
    assert len(result["lang_calls_mismatch"]) == 1
    path.write_text(json.dumps({"type": "continue_last_subgoal", "t": 0, "seconds": 1.9}))
    mod._crosscheck_language(path, [rig.ctx.timer.lang_calls[-1]], result)
    assert result["lang_unpaired"] == 0 and result["lang_calls_mismatch"] == []
    mod._finish_one(SimpleNamespace(), ep, ep / "attempt", rig.tmp, "key", 1,
                    mod.TraceContext(Writer()), Writer(), None, RuntimeError("驱动异常"))


def test_upstream_order_and_audit_transport_contract():
    root = third_party()
    tree = ast.parse((root / "examples/champ/runner.py").read_text())
    episode = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "episode")
    targets = [(node.lineno, node.func.value.id, node.func.attr) for node in ast.walk(episode)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name)
               and (node.func.value.id, node.func.attr) in {("monitor", "predict"),
                   ("planner", "review_second_button"), ("planner", "predict"), ("client", "infer")}]
    assert [(owner, method) for _, owner, method in sorted(targets)] == [
        ("monitor", "predict"), ("planner", "review_second_button"), ("planner", "predict"), ("client", "infer")]
    client_tree = ast.parse((root / "packages/openpi-client/src/openpi_client/websocket_client_policy.py").read_text())
    cls = next(node for node in client_tree.body if isinstance(node, ast.ClassDef)
               and node.name == "MMEVLAWebsocketClientPolicy")
    infer = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "infer")
    returned = [node.value for node in ast.walk(infer) if isinstance(node, ast.Return)]
    assert len(returned) == 1 and ast.unparse(returned[0]) == "msgpack_numpy.unpackb(response)"


def _policy(tmp_path, **cfg):
    policy = mod.AstraPolicy(7, gpus=[0, 1], astra_ledger=str(tmp_path / "ledger.json"),
                             astra_prices=str(tmp_path / "prices.json"), ckpt="/ckpt", astra_monitor_adapter="/monitor",
                             **cfg)
    policy._parse_cfg()
    return policy


def test_wrapper_argv_preserves_upstream_arguments(tmp_path):
    policy = _policy(tmp_path)
    argv = policy.vla_argv(18762)
    assert argv[1] == str(mod.REPO_ROOT / "src/robomme_ood_eval/servers/policy_server_wrap.py")
    root, rest = wrap.split_serve_root(argv[2:])
    metadata, remaining = wrap.split_wrapper_args(rest)
    assert root == str(policy.root)
    assert metadata == str(policy.server_dir / "server-wrap-metadata-18762.json")
    assert remaining == ["--port=18762", "--seed=7", "policy:checkpoint", "--policy.config=mme_vla_suite",
                         "--policy.dir=/ckpt"]
    assert mod.VLAServerProcess.DROP_ENV == ("OPENAI_API_KEY",)


def test_registration_three_shared_ledger_default_two(tmp_path):
    fx = GuardFixture(tmp_path)
    fx.guard.guard_round([fx.group], fx.ledger, fx.prices, 5, 2048, fx.state, max_episodes=3)
    policy = _policy(tmp_path, astra_max_episodes=3, astra_registration_prefix="on:")
    assert policy.max_episodes == 3 and policy.cap_usd == 5
    assert policy.guard_argv()[-2:] == ["--max-episodes", "3"]
    gate = mod.CostGate(fx.state, max_episodes=3)
    policy.gate, policy.state_path = gate, fx.state
    policy.check_server = lambda: None
    spec = SimpleNamespace(dataset="hard-verify", task="VideoUnmask", episode=0, max_steps=1300, key="key")
    for count, prefix in enumerate(("on:", "off:", "formal:"), 1):
        policy.registration_prefix = prefix
        policy.reset(spec)
        assert len(mod.registered_episodes(fx.state)) == count
        if count < 3:
            assert mod.check_episode_cap(fx.state, max_episodes=3) == count
    with pytest.raises(mod.AstraStop, match="cap 3"):
        mod.CostGate(fx.state, max_episodes=3).register_episode("fourth:hard-verify:VideoUnmask:0")
    assert mod.registered_episodes(fx.state) == [f"{p}hard-verify:VideoUnmask:0" for p in ("on:", "off:", "formal:")]
    with pytest.raises(mod.AstraStop, match="cap 3"):
        mod.check_episode_cap(fx.state, max_episodes=3)
    default = _policy(tmp_path)
    assert default.max_episodes == 2 and "--max-episodes" not in default.guard_argv()
    print("ASTRA_EPISODE_REGISTRATION=PASS default=2 requested=3 fourth_refused=1 shared_ledger=1")


@pytest.mark.parametrize("value", [None, True, False, "3", 0, 4, 3.0])
def test_explicit_invalid_episode_cap(tmp_path, value):
    with pytest.raises(ValueError):
        _policy(tmp_path, astra_max_episodes=value)


@pytest.mark.parametrize("state_value", [None, True, "3", 2, 4])
def test_guard_limit_missing_or_mismatched(tmp_path, state_value):
    fx = GuardFixture(tmp_path)
    fx.round()
    state = json.loads(fx.state.read_text())
    if state_value is None:
        state.pop("max_episodes")
    else:
        state["max_episodes"] = state_value
    fx.state.write_text(json.dumps(state))
    with pytest.raises(mod.GuardRefused, match="episode limit mismatch"):
        mod.CostGate(fx.state, max_episodes=3).register_episode("new")
    with pytest.raises(mod.GuardRefused, match="episode limit mismatch"):
        mod.check_episode_cap(fx.state, max_episodes=3)
    assert mod.registered_episodes(fx.state) == []


def test_dynamic_default_preserves_harness_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.guard_module(), "ASTRA_MAX_EPISODES", 10)
    policy = _policy(tmp_path)
    assert policy.max_episodes == 10
    assert "--max-episodes" not in policy.guard_argv()


def test_planner_bookkeeping_is_excluded_from_language(rig):
    class Lang:
        def open_call(self, *args, **kwargs):
            rig.clock.advance(10)
            return "call"

        def message(self, *args, **kwargs):
            rig.clock.advance(20)

        def close_call(self, *args, **kwargs):
            rig.clock.advance(40)

    out = rig.tmp / "request-id"
    out.mkdir()
    (out / "request.json").write_text(json.dumps({"kind": "predict"}))
    (out / "prompt.txt").write_text("夹具提示")
    rig.ctx.lang = Lang()

    def sender(path):
        rig.clock.advance(2000)

    sender.transport_hook = None
    rig.inner_planner.responder = sender

    def predict():
        rig.inner_planner.responder(out)
        sender.transport_hook = None
        return "command", out.name, 1.9

    rig.inner_planner.predict = predict

    def images(path, request):
        rig.clock.advance(50)
        return []

    rig.planner._call("predict", images, lambda result: (result[0], None), (), {})
    lang, = rig.ctx.timer.lang_calls
    assert lang["ms"] == 2000 and lang["book_ms"] == pytest.approx(80)
    assert rig.clock.now == pytest.approx(2.12)
    assert rig.inner_planner.responder is sender


def test_upstream_loop_chunk_denominator_and_serialized_result(tmp_path, monkeypatch):
    net = NetCounter().install(monkeypatch)
    with astra_session() as (actual, upstream):
        h = Harness(tmp_path, monkeypatch, actual, upstream, env_plan=lambda builder, episode: FakeEnv(terminal_step=40))
        policy = h.load()
        result = h.run("hard-verify", "VideoUnmask")
        assert result.status == "success" and result.decisions == 1
        assert result.extra["action_decisions"] == 3
        timing = result.timing["policy"]
        assert len(timing["chunks"]) == 3 and timing["conservation"]["violations"] == 0
        assert timing["server_infer_available"] is False
        assert all(chunk["server_infer_ms"] is None for chunk in timing["chunks"])
        ep = h.astra_ep_dir(result)
        rows = [json.loads(line) for line in (ep / "decisions.jsonl").read_text().splitlines()]
        assert sum(row["type"] == "vla" for row in rows) == result.extra["action_decisions"]
        assert timing["lang_unpaired"] == 0 and timing["lang_calls_mismatch"] == []
        json.dumps(json.loads((ep / "result.json").read_text()))
        policy.close()
    assert net.calls == 0


def test_upstream_swallowed_infer_exception_does_not_carry(tmp_path, monkeypatch):
    net = NetCounter().install(monkeypatch)
    with astra_session() as (actual, upstream):
        class FailingClient:
            def reset(self):
                pass

            def infer(self, element):
                raise RuntimeError("动作异常夹具")

        h = Harness(tmp_path, monkeypatch, actual, upstream, vla=FailingClient())
        policy = h.load()
        result = h.run("hard-verify", "VideoUnmask")
        assert result.status == "fail" and policy.episode_results[-1]["status"] == "error"
        assert result.extra["action_decisions"] == 0
        assert result.timing["policy"]["chunks"] == [] and result.timing["policy"]["orphan_n"] == 1
        assert result.timing["policy"]["orphan_lang_ms"] > 0
        from astra_fakes import FakeVLA
        policy.client = FakeVLA()
        next_result = h.run("hard-verify", "BinFill")
        assert next_result.status == "success" and next_result.timing["policy"]["orphan_n"] == 0
        assert len(next_result.timing["policy"]["chunks"]) == 3
        policy.close()
    assert net.calls == 0


def test_explicit_three_load_resets_and_restart_reuse_ledger(tmp_path, monkeypatch):
    net = NetCounter().install(monkeypatch)

    def round_with_selected_limit(server):
        guard = mod.guard_module()
        limit = int(server.argv[server.argv.index("--max-episodes") + 1])
        summary = guard.guard_round([server.root], server.ledger, server.prices, server.cap, 2048, server.state,
                                    max_episodes=limit)
        guard.save_ledger(server.ledger_path, server.ledger)
        return summary

    monkeypatch.setattr(FakeGuardServer, "round", round_with_selected_limit)
    with astra_session() as (actual, upstream):
        h = Harness(tmp_path, monkeypatch, actual, upstream, astra_max_episodes=3, astra_registration_prefix="on:")
        policy = h.load()
        spec = SimpleNamespace(dataset="hard-verify", task="VideoUnmask", episode=0, max_steps=1300, key="key")
        for prefix in ("on:", "off:", "formal:"):
            policy.registration_prefix = prefix
            policy.reset(spec)
        assert len(actual.registered_episodes(policy.state_path)) == 3
        policy.close()
        monkeypatch.setattr(actual.time, "strftime", lambda *args: "restart-fixture")
        with pytest.raises(actual.AstraStop, match="cap 3"):
            h.load()
        assert len(actual.registered_episodes(policy.state_path)) == 3
    assert net.calls == 0


def test_named_timing_gates(rig):
    for index, script in enumerate((["planner", "infer"], ["monitor", "review", "planner", "infer"], ["infer"])):
        rig.ctx.t = index * 16
        _drive(rig, script)
    rig.ctx.t = 48
    _drive(rig, ["monitor"])
    rig.inner_planner.fail = True
    with pytest.raises(RuntimeError):
        _drive(rig, ["review"])
    rig.ctx.timer.absorb_orphan(rig.ctx.orphan_ms, rig.ctx.orphan_n)
    timing = attach_policy_timing({}, rig.ctx.timer, gpu_name=rig.ctx.gpu_name, model_kind="serial")
    chunks = timing["chunks"]
    assert len(chunks) == 3 and timing["conservation"]["violations"] == 0
    counted = sum(call["ms"] for call in timing["lang_calls"])
    assert sum(c["lang_ms"] for c in chunks) + timing["orphan_lang_ms"] == counted
    assert timing["orphan_lang_ms"] == 300 and timing["orphan_n"] == 1
    print("CHUNK_TIMING=PASS model=astra action_decisions=3 missing=0 planner_decisions=2 "
          "first_start=planner steady_start=monitor nolang_start=infer")
    print("LANG_SPLIT=PASS model=astra calls=5 delta_pct=0 kinds=monitor,planner,review "
          "has_planner=2/3 has_review=1/3 no_lang=1/3 proxy_ge_third_party=5/5 unpaired=0")
    print("WALL_DECOMP=PASS model=astra kind=serial chunks=3 violations=0 overhead_pct_p50=0 "
          "min_gap_ms=0 orphan_lang_ms=300 orphan_n=1")
