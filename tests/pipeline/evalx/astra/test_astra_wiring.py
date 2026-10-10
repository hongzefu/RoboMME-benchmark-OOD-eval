"""Astra 接线：``AstraPolicy``（模型侧 4 方法）经外层 ``run_episode`` 跑子模块 ``runner.episode``（环境打桩、零外联）。

判定行：``ASTRA_WIRING=PASS api_calls=0 builder_dataset_ok=1``（由 ``test_astra_wiring_xhard0`` 打印）。

覆盖：

- 外层 builder 构造参数（dataset、action_space=joint_angle、gui_render=False、max_steps=1300／1800）与
  ``make_env_for_episode`` 只传局号；``SessionBuilder`` 把外层会话交给 ``runner.episode``，``close()`` 置空、环境只由外层关；
- S4 三件：``args.output`` = 本局 raw 目录下 ``astra/``（Astra 自写产物落在那里）、尝试目录名用 ``spec.attempt``、
  停机规则（另见 ``test_astra_stop_rules.py``）；
- 同一 Policy 先后跑 hard-verify（1300）与 ood（1800），``args.max_steps`` 随每局传入；
- ood strict cap：Astra 循环越界时外层会话与 ``TracedEnv`` 都在第 1801 步进环境之前拒绝，本局按 timeout 收尾；
- 环境源断言为 ``third_party/robomme_benchmark/src``；
- 费用硬上限（零外联夹具）：首次大输入、输入增长、usage 缺失、STOP 在等待中到达、守卫崩溃、429 重试复检；
  被拒请求一律不触发实际发送；局数硬上限 2（跨进程登记）。
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_fakes import (FAKE_KEY, FakeEnv, FakeRecorder, Harness, NetCounter,
                         astra_session, load_guard, print_upstream_digests, read_trace, third_party_dir)
# 费用硬上限的零外联夹具（test_astra_stage3 也用，放在公共替身 astra_fakes）
from astra_fakes import OUT_WORST, PRICES, Clock, FakeUrlopen, GuardFixture, _client, _http_429, _resp
from tests.pipeline.evalx.report import trace_contract as tc


def _find_episode(builder_cls, task: str, dataset: str, *, tier: str | None = None, source: int | None = None) -> int:
    b = builder_cls(task, dataset=dataset, action_space="joint_angle", gui_render=False,
                    max_steps={"hard-verify": 1300, "ood": 1800}[dataset])
    for ep in range(b.get_episode_num()):
        ident = b.resolve_identity(ep)
        if (tier is None or ident["tier"] == tier) and (source is None or ident.get("source_episode") == source):
            return ep
    raise AssertionError(f"{task} {dataset} 找不到 tier={tier} source={source}")


# ── 一局接线（hard-verify） ─────────────────────────────────────────────────

def test_astra_wiring_xhard0(tmp_path, monkeypatch):
    """两局 hard-verify（本地局号 0 = 官方 test episode 3）：外层 builder 构造参数逐项、Astra 自写产物落在本局
    raw 目录下 ``astra/``、尝试目录 ``<key>.a1``、trace 在 raw 目录（40 步、请求序列 vla_reset、planner×1、
    vla_infer×3、monitor×2）、Astra ``result.json`` 追加映射、环境只由外层关一次、规划替身恰 2 次、零外联。"""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        print_upstream_digests()
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=40))
        policy = h.load()
        results = [h.run("hard-verify", task, 0) for task in ("BinFill", "VideoUnmask")]
        policy.close()
    builder_ok = all(c["dataset"] == "hard-verify" and c["action_space"] == "joint_angle"
                     and c["gui_render"] is False and c["max_steps"] == 1300 for c in h.cls.constructed)
    assert builder_ok and {c["env_id"] for c in h.cls.constructed} == {"BinFill", "VideoUnmask"}
    assert [(c["env_id"], c["episode"], c["args"], c["kwargs"]) for c in h.cls.make_calls] == [
        ("BinFill", 0, (), {}), ("VideoUnmask", 0, (), {})]
    for r in results:
        assert r.status == "success" and r.task_success == 1 and r.exec_steps == 40 and r.source_episode == 3
        assert r.error is None and r.infra is False
        assert r.extra["planner_calls"] == 1 and r.decisions == 1 and r.extra["cloud_seed"] is None
        assert r.extra["astra_env_close_calls"] == 1 and r.extra["astra_dir"] == f"astra/{r.task}/ep000"
        raw = h.raw(r)
        ep_dir = h.astra_ep_dir(r)
        for name in ("identity.json", "decisions.jsonl", "actions.npy", "result.json", "review_protocol.json"):
            assert (ep_dir / name).is_file(), name
        assert (ep_dir / "monitor_inputs").is_dir()
        ident = json.loads((ep_dir / "identity.json").read_text())
        assert ident["dataset"] == "hard-verify" and ident["episode"] == 0
        astra_res = json.loads((ep_dir / "result.json").read_text())
        assert astra_res["status"] == "success" and astra_res["key"] == r.key and astra_res["attempt"] == 1
        assert astra_res["episode_dir"] == f"{r.key}.a1" and astra_res["trace"] == "../../../trace.jsonl"
        assert (h.attempt_dir(r) / "provenance.json").is_file()
        rows = read_trace(raw / "trace.jsonl")
        assert rows[0]["route"] == "astra/new" and rows[0]["identity"]["key"] == r.key
        assert rows[0]["identity"]["attempt"] == 1 and rows[0]["identity"]["builder_episode"] == 0
        assert sum(1 for x in rows if x["kind"] == "step") == 40
        reqs = [x["name"] for x in rows if x["kind"] == "request"]
        assert reqs == ["vla_reset", "planner", "vla_infer", "monitor", "vla_infer", "monitor", "vla_infer"]
        assert rows[-1]["status"] == "success" and rows[-1]["steps_attempted"] == 40
        assert tc.contract_problems(raw) == []  # 重绘工具 load_trace 另由 dev-scripts 侧负责
        tc.assert_counts_consistent(raw, {"exec_steps": r.exec_steps, "status": r.status})
        rec = next(x for x in FakeRecorder.instances if x.raw == raw)
        assert rec.frames["front"] == rec.frames["wrist"] == 3 + 1 + 40, "演示 3 帧 + 初始帧 + 40 步，经外层会话录"
    assert h.responder.calls == 2 and len(h.responder.seen) == 2
    assert policy.spool.name == "planner_calls" and policy.spool.parent.parent == h.group, "规划 spool 在 group_0 下"
    assert len([p for p in policy.spool.iterdir() if p.is_dir()]) == 2
    assert net.calls == 0
    print(f"ASTRA_WIRING=PASS api_calls={net.calls} builder_dataset_ok={int(builder_ok)}")


def test_env_closed_only_by_outer_session(tmp_path, monkeypatch):
    """``runner.episode`` 的 ``finally: env.close()`` 落在 ``SessionEnv.close``（置空）；真实环境只由外层
    ``session.close()`` 关一次。"""
    NetCounter().install(monkeypatch)
    envs = []

    def plan(b, ep):
        envs.append(FakeEnv(terminal_step=20))
        return envs[-1]

    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=plan)
        h.load()
        r = h.run("hard-verify", "BinFill", 0)
        h.policy.close()
    assert r.status == "success" and envs[0].close_calls == 1 and r.extra["astra_env_close_calls"] == 1


def test_session_builder_adapter():
    """``SessionBuilder``：只交本局局号、只交一次；``SessionEnv.close`` 置空；``resolve_episode`` 委托真 builder。"""
    with astra_session() as (mod, _astra):
        events = []
        session = SimpleNamespace(reset=lambda: events.append("reset") or ({}, {}),
                                  step=lambda a: events.append("step") or (None, 0, False, False, {}),
                                  close=lambda: events.append("session.close"), steps=0, step_cap=None)
        real = SimpleNamespace(resolve_episode=lambda ep: ("seed", ep))
        b = mod.SessionBuilder(session, real, 3)
        with pytest.raises(ValueError, match="only hands out this episode's environment"):
            b.make_env_for_episode(2)
        env = b.make_env_for_episode(3)
        env.reset()
        env.step(np.zeros(8, np.float32))
        env.close()
        assert events == ["reset", "step"] and env.close_calls == 1, "close 置空，不得调 session.close"
        assert env.steps == 0 and b.resolve_episode(3) == ("seed", 3)
        with pytest.raises(RuntimeError, match="only once per episode"):
            b.make_env_for_episode(3)


# ── 同一 Policy 跑两档；尝试号进目录名 ──────────────────────────────────────

def test_two_datasets_one_policy_max_steps_per_episode(tmp_path, monkeypatch):
    """同一 Policy 先跑 hard-verify 再跑 ood：``args.max_steps`` 逐局为 1300、1800，``args.output`` 恒为本局 raw 目录
    下的 ``astra/``；load 一次、reset 与 play 各两次；外层 builder 按数据集各建一个。"""
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=20))
        seen = []
        real_episode = astra.runner.episode

        def spy(args, task, ep, builder, *rest):
            seen.append((args.dataset, args.max_steps, Path(args.output), task, ep))
            return real_episode(args, task, ep, builder, *rest)

        monkeypatch.setattr(astra.runner, "episode", spy)
        policy = h.load()
        ood_ep = _find_episode(h.cls, "VideoUnmask", "ood", tier="xhard1")
        r1 = h.run("hard-verify", "BinFill", 0)
        r2 = h.run("ood", "VideoUnmask", ood_ep)
        policy.close()
    assert [(d, m) for d, m, *_ in seen] == [("hard-verify", 1300), ("ood", 1800)]
    assert seen[0][2] == h.raw(r1) / "astra" and seen[1][2] == h.raw(r2) / "astra"
    assert r1.status == r2.status == "success" and r2.strict_cap is True and r1.strict_cap is False
    assert [(c["env_id"], c["dataset"], c["max_steps"]) for c in h.cls.constructed[1:]] == [  # [0] 是找局号用的
        ("BinFill", "hard-verify", 1300), ("VideoUnmask", "ood", 1800)]
    assert policy.calls == {"load": 1, "reset": 2, "play": 2, "close": 1}
    assert len(h.servers()) == 2 and all(s.starts == 1 for s in h.servers().values())


def test_attempt_number_in_dir_trace_and_provenance(tmp_path, monkeypatch):
    """尝试号来自 ``spec.attempt``（不再写死 1）：尝试目录 ``<key>.a3``、trace identity、provenance、Astra 结果都记 3。"""
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=20))
        h.load()
        r = h.run("hard-verify", "BinFill", 0, attempt=3)
        h.policy.close()
    assert r.attempt == 3 and h.attempt_dir(r).name == f"{r.key}.a3"
    assert json.loads((h.attempt_dir(r) / "provenance.json").read_text())["attempt"] == 3
    assert read_trace(h.raw(r) / "trace.jsonl")[0]["identity"]["attempt"] == 3
    assert json.loads((h.astra_ep_dir(r) / "result.json").read_text())["attempt"] == 3


def test_existing_astra_ep_dir_refused_keeps_evidence(tmp_path, monkeypatch):
    """本局 raw 目录下 Astra 局目录已存在（上一尝试的证据）：play 拒绝、不覆盖、不发任何规划请求；外层记 error，
    Policy 继续下一局（单次非规划错误不停）。"""
    NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=20))
        policy = h.load()
        spec, _ = h.E.make_spec(policy, "hard-verify", "BinFill", 0, h.out)
        old = Path(spec.out_dir) / "astra" / "BinFill" / "ep000"
        old.mkdir(parents=True)
        (old / "marker.txt").write_text("上一尝试")
        r1 = h.run("hard-verify", "BinFill", 0)
        r2 = h.run("hard-verify", "VideoUnmask", 0)
        policy.close()
    assert r1.status == "fail" and r1.error_kind == "error" and "episode directory already exists" in r1.error
    assert (old / "marker.txt").read_text() == "上一尝试" and sorted(p.name for p in old.iterdir()) == ["marker.txt"]
    assert r2.status == "success" and h.responder.calls == 1


# ── ood strict cap 与 1800 步 ──────────────────────────────────────────────

def test_v9_connectivity_runs_exactly_1800_steps(tmp_path, monkeypatch):
    """ood（VideoUnmask xhard1 第 0 局）环境永不结束：Astra 循环恰好执行 1800 步，``terminal_reason=timeout``，
    ``cap_hit=false``（循环自己停，未触发 strict 拒绝）。"""
    NetCounter().install(monkeypatch)
    envs = []

    def plan(b, ep):
        envs.append(FakeEnv(terminal_step=None))
        return envs[-1]

    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=plan)
        h.load()
        ep = _find_episode(h.cls, "VideoUnmask", "ood", tier="xhard1")
        r = h.run("ood", "VideoUnmask", ep)
        h.policy.close()
    assert envs[0].steps_taken == 1800 and r.status == "timeout" and r.cap_hit is False and r.exec_steps == 1800
    end = read_trace(h.raw(r) / "trace.jsonl")[-1]
    assert end["status"] == end["terminal_reason"] == "timeout" and end["cap_hit"] is False
    assert end["steps_attempted"] == 1800


def test_strict_cap_overrun_finishes_as_timeout(tmp_path, monkeypatch):
    """让 Astra 自己的循环以为上限是 1900（模拟循环越界）：第 1801 次 ``step`` 在进环境之前被拒（``TracedEnv`` 先拒并
    交外层会话再拒一次，置 ``session.cap_hit``）；本局按 timeout 收尾、``cap_hit=True``、不计基础设施错误，
    trace／arrays 恰 1800 步，外层结果与 Astra 结果一致。"""
    net = NetCounter().install(monkeypatch)
    envs = []

    def plan(b, ep):
        envs.append(FakeEnv(terminal_step=None))
        return envs[-1]

    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=plan)
        real_episode = astra.runner.episode

        def overrun(run_args, *rest):
            return real_episode(SimpleNamespace(**{**vars(run_args), "max_steps": 1900}), *rest)

        monkeypatch.setattr(astra.runner, "episode", overrun)
        h.load()
        ep = _find_episode(h.cls, "VideoUnmask", "ood", tier="xhard1")
        r = h.run("ood", "VideoUnmask", ep)
        stop = h.policy._stop
        h.policy.close()
    assert envs[0].steps_taken == 1800
    assert r.status == "timeout" and r.cap_hit is True and r.exec_steps == 1800 and r.strict_cap is True
    assert stop is None and h.policy._errors == 0, "strict cap 不计基础设施错误"
    raw = h.raw(r)
    rows = read_trace(raw / "trace.jsonl")
    end = rows[-1]
    assert end["status"] == end["terminal_reason"] == "timeout" and end["cap_hit"] is True
    assert end["steps_attempted"] == 1800 and sum(1 for x in rows if x["kind"] == "step") == 1800
    with np.load(raw / "arrays.npz") as arr:
        acts = sorted(k for k in arr.files if k.startswith("exec_action__"))
        assert acts[-1] == "exec_action__01799" and len(acts) == 1800
    assert tc.contract_problems(raw) == []  # 重绘工具 load_trace 另由 dev-scripts 侧负责
    tc.assert_counts_consistent(raw, {"exec_steps": r.exec_steps, "status": r.status})
    saved = json.loads((h.astra_ep_dir(r) / "result.json").read_text())
    assert saved["status"] == "timeout" and saved["cap_hit"] is True and saved["effective_cap"] == 1800
    rec = next(x for x in FakeRecorder.instances if x.raw == raw)
    assert any(e.get("kind") == "step_cap_reached" for e in rec.events)
    assert net.calls == 0
    print("EVAL_CAP=PASS route=astra max_steps=1800 rejected_step=1801")


def test_step_cap_pairing_blocks_in_reset(tmp_path, monkeypatch):
    """数据集与步数配对不符（模拟外层给错 ``spec.max_steps``）：``reset`` 在登记局数与任何请求之前拒绝。"""
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        policy = h.load()
        spec, _ = h.E.make_spec(policy, "hard-verify", "BinFill", 0, h.out)
        import dataclasses
        bad = dataclasses.replace(spec, max_steps=1600)
        with pytest.raises(ValueError, match="step_cap_pairing"):
            policy.reset(bad)
        assert mod.registered_episodes(policy.state_path) == []
        policy.close()
    assert h.responder.calls == 0


def test_env_sources_point_to_benchmark_submodule():
    """环境源必须是 ``third_party/robomme_benchmark/src``；``benchmark_src()`` 跟随 ``SGEVAL_THIRD_PARTY``。"""
    with astra_session() as (mod, _astra):
        files = mod.assert_env_sources()
        src = str(mod.benchmark_src().resolve())
    assert src == str((third_party_dir() / "robomme_benchmark" / "src").resolve())
    assert files["robomme_ood"].startswith(src) and files["robomme"].startswith(src)


# ── 费用硬上限（零外联夹具在 astra_fakes） ───────────────────────────────────────────────

def test_guard_defaults_hard_cap_and_constants(tmp_path, capsys):
    guard = load_guard()
    args = guard.build_parser().parse_args(["--root", "r", "--prices", "p", "--ledger", "l"])
    assert args.cap == 5.0 and args.interval == 2.0 and guard.HARD_CAP_USD == 5.0
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps(PRICES))
    base = ["--root", str(tmp_path), "--prices", str(prices), "--ledger", str(tmp_path / "l.json"), "--once"]
    assert guard.main(base + ["--cap", "5.01"]) == 2 and "ASTRA_COST_BLOCKED" in capsys.readouterr().err
    assert guard.main(base + ["--interval", "6"]) == 2
    assert guard.main(base) == 0
    state = json.loads((tmp_path / "l.state.json").read_text())
    assert state["cap"] == 5.0 and state["stop"] is False and state["schema"] == guard.STATE_SCHEMA
    with astra_session() as (mod, _astra):
        assert mod.GUARD_HEARTBEAT_TIMEOUT_S == guard.HEARTBEAT_TIMEOUT_S == 10.0
        assert mod.guard_module().ASTRA_MAX_EPISODES == 2


def test_first_large_input_refused_without_sending(tmp_path, monkeypatch):
    """首次请求输入就大到「本次最坏」超过 5 美元：发送前拒发，urlopen 0 次；守卫把它计 0、不停机；随后小请求照常。"""
    fake = FakeUrlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        big = fx.request(2_600_000)  # 2.6M 字节 × 2 美元/百万 ≥ 5.2 美元
        client(big)
        assert fake.calls == 0
        refused = _resp(big)
        assert refused["status"] == "error" and "cost reservation refused" in refused["error"]
        assert (big / "guard_refused.json").is_file() and big.name not in fx.reservations()
        summary = fx.round()
        assert summary["refused"] == 1 and summary["usd"] == 0 and summary["stop"] is False
        assert fx.ledger["calls"][big.name]["missing"] == ["refused"]
        small = fx.request(1000, images=2)
        client(small)
    assert fake.calls == 1 and _resp(small)["status"] == "ok"
    res = fx.reservations()[small.name]
    # 估计：1000 字节 + 2 张 256×256（high detail 上界各 255）+ 64
    assert res["input_tokens"] == 1000 + 2 * 255 + 64 and res["state"] == "sent"
    assert res["usd"] == pytest.approx(res["input_tokens"] * 2.0 / 1e6 + OUT_WORST)


def test_input_growth_accumulates_reservations_until_refused(tmp_path, monkeypatch):
    """输入逐次增长、守卫尚未计入回包：未结预留累计，第 4 次（累计会超过 5 美元）在发送前被拒；
    守卫计入实际 usage 后预留结清，同样大小的请求再次放行。"""
    fake = FakeUrlopen([{"input_tokens": 1000}] * 5)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    sizes = [400_000, 700_000, 1_000_000, 1_300_000]
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        outs = []
        for size in sizes:
            outs.append(fx.request(size))
            client(outs[-1])
        assert fake.calls == 3
        assert [_resp(o)["status"] for o in outs] == ["ok", "ok", "ok", "error"]
        assert "cost reservation refused" in _resp(outs[3])["error"]
        worst = [fx.reservations()[o.name]["usd"] for o in outs[:3]]
        assert worst == sorted(worst) and worst[0] < worst[1] < worst[2]
        assert sum(worst) == pytest.approx(sum((s + 64) * 2.0 / 1e6 + OUT_WORST for s in sizes[:3]))
        summary = fx.round()  # 守卫计入三次实际 usage（各 1000 输入），预留结清
        assert summary["calls"] == 4 and summary["refused"] == 1 and summary["stop"] is False
        retry = fx.request(1_300_000)
        client(retry)
    assert fake.calls == 4 and _resp(retry)["status"] == "ok"
    assert clock.slept and all(s >= 0 for s in clock.slept), "发送间隔照第三方 20 秒节奏等待"


def test_missing_usage_writes_stop_and_blocks_next_send(tmp_path, monkeypatch):
    """回包缺 usage：守卫视为超限写 STOP（reason=astra_usage_missing），下一次请求发送前即被拒。"""
    fake = FakeUrlopen([None])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, gate = _client(mod, astra, fx, clock)
        first = fx.request(1000)
        client(first)
        assert fake.calls == 1 and _resp(first)["status"] == "ok" and _resp(first)["usage"] is None
        summary = fx.round()
        assert summary["stop"] is True and summary["reason"] == "astra_usage_missing" and summary["unknown_usage"] == 1
        stop = json.loads((fx.group / "STOP.json").read_text())
        assert stop["reason"] == "astra_usage_missing"
        with pytest.raises(mod.GuardRefused, match="stopped"):
            gate.read_state()
        second = fx.request(1000)
        client(second)
    assert fake.calls == 1
    assert _resp(second)["status"] == "error" and "Host requested stop" in _resp(second)["error"]


def test_stop_arriving_during_wait_blocks_send(tmp_path, monkeypatch):
    """预留通过后进入第三方 20 秒发送间隔，等待期间守卫写 STOP：等待结束复检即拒发，预留释放。"""
    fake = FakeUrlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        client.next_request_at = clock.now + 20
        clock.on_sleep = lambda s: (fx.group / "STOP.json").write_text('{"reason": "astra_cost_cap"}')
        out = fx.request(1000)
        client(out)
    assert clock.slept == [20]
    assert fake.calls == 0 and "Host requested stop" in _resp(out)["error"]
    assert fx.reservations()[out.name]["state"] == "released" and (out / "guard_refused.json").is_file()


def test_retry_after_429_is_gated_again(tmp_path, monkeypatch):
    """429 有界重试：每次重试前重新过闸（复用同一份预留）；重试等待中 STOP 到达则不再发第二次。"""
    fake = FakeUrlopen([_http_429(), {"input_tokens": 10}])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, clock)
        ok = fx.request(1000)
        client(ok)
        assert fake.calls == 2 and _resp(ok)["status"] == "ok"
        assert list(fx.reservations()) == [ok.name] and (ok / "http_error_00.json").is_file()
        fake.plan = [_http_429(), {"input_tokens": 10}]
        stopped = fx.request(1000)
        client.next_request_at = clock.now  # 首次发送不等待；只有 429 退避（≥10 秒）期间写 STOP
        clock.on_sleep = lambda s: s >= 10 and (fx.group / "STOP.json").write_text("{}")
        client(stopped)
    assert fake.calls == 3, "第二个请求只发出第一次（429），重试前见到 STOP 即停"
    assert "Host requested stop" in _resp(stopped)["error"]


@pytest.mark.parametrize("failure", ["stale", "missing", "exited"])
def test_guard_unavailable_refuses_without_sending(tmp_path, monkeypatch, failure):
    """守卫失联：心跳超过 10 秒、状态文件不存在、守卫已退出——三种都在发送前拒发。"""
    fake = FakeUrlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    gate_clock = time.time
    if failure == "stale":
        gate_clock = lambda: time.time() + 10.5  # noqa: E731
    elif failure == "missing":
        fx.state.unlink()
    else:
        state = json.loads(fx.state.read_text())
        state["exited"] = True
        fx.state.write_text(json.dumps(state))
    with astra_session() as (mod, astra):
        client, _gate = _client(mod, astra, fx, Clock(), gate_clock=gate_clock)
        out = fx.request(1000)
        client(out)
    assert fake.calls == 0
    assert _resp(out)["status"] == "error" and "cost guard unavailable" in _resp(out)["error"]


def test_real_guard_process_crash_heartbeat_goes_stale(tmp_path):
    """真实守卫进程（--interval 0.2）被 SIGKILL：心跳停更，最后一次心跳 10 秒后 runner 拒发、之前仍放行。"""
    fx = GuardFixture(tmp_path)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps(PRICES))
    guard_py = Path(load_guard().__file__)
    proc = subprocess.Popen([sys.executable, str(guard_py), "--root", str(fx.group), "--prices", str(prices),
                             "--ledger", str(tmp_path / "guard" / "ledger.json"), "--state", str(fx.state),
                             "--interval", "0.2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 30
        beats = set()
        while time.time() < deadline and len(beats) < 2:
            if fx.state.is_file():
                try:
                    beats.add(json.loads(fx.state.read_text())["heartbeat"])
                except ValueError:
                    pass
            time.sleep(0.05)
        assert len(beats) >= 2, "守卫心跳未更新"
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    last = json.loads(fx.state.read_text())
    assert last["exited"] is False  # 崩溃不会写 exited
    with astra_session() as (mod, _astra):
        assert mod.CostGate(fx.state, clock=lambda: last["heartbeat"] + 5).read_state()["cap"] == 5.0
        with pytest.raises(mod.GuardRefused, match="heartbeat"):
            mod.CostGate(fx.state, clock=lambda: last["heartbeat"] + 10.5).read_state()


def test_runner_cap_never_exceeds_hard_cap(tmp_path):
    """守卫状态里的 cap 被改大也不放宽：生效上限 = min(状态 cap, 5)。"""
    fx = GuardFixture(tmp_path)
    fx.round()
    state = json.loads(fx.state.read_text())
    state["cap"] = 50.0
    fx.state.write_text(json.dumps(state))
    with astra_session() as (mod, _astra):
        gate = mod.CostGate(fx.state)
        assert gate.cap(gate.read_state()) == 5.0
        payload = {"input": [{"content": [{"type": "input_text", "text": "x" * 2_500_000}]}], "max_output_tokens": 2048}
        with pytest.raises(mod.GuardRefused, match="reservation refused"):
            gate.reserve("r" * 32, payload)


def test_episode_cap_two_across_runs(tmp_path):
    """局数硬上限 2：同一守卫预留文件跨进程计数，同一局重复登记不重复计，第 3 局拒绝；``check_episode_cap`` 在
    已登记 2 局时直接报停。"""
    fx = GuardFixture(tmp_path)
    fx.round()
    with astra_session() as (mod, _astra):
        gate = mod.CostGate(fx.state)
        assert gate.register_episode("hard-verify:BinFill:0") == 1
        assert gate.register_episode("hard-verify:BinFill:0") == 1
        assert mod.check_episode_cap(fx.state) == 1
        assert mod.CostGate(fx.state).register_episode("hard-verify:VideoUnmask:0") == 2
        with pytest.raises(mod.AstraStop) as info:
            mod.CostGate(fx.state).register_episode("hard-verify:MoveCube:0")
        assert info.value.reason == "episode_cap"
        with pytest.raises(mod.AstraStop) as info2:
            mod.check_episode_cap(fx.state)
        assert info2.value.reason == "episode_cap"
        from robomme_ood_eval.policy import AstraStop as BaseStop
        assert isinstance(info.value, BaseStop), "外层 evaluate.py 按基类整批停"
