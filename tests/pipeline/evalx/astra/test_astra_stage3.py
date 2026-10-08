"""Astra 的第三阶段功能（接到 ``AstraPolicy`` 之后）：1800 步 strict cap、模型 seed、语言账本、数组合并写。

全部零外联、零费用、零 GPU：环境、VLA、规划、监视、两个服务端都是替身（见 ``astra_fakes.Harness``）。

判定行（由对应用例打印）：
- ``POLICY_SEEDS=PASS route=astra seeds=0,7,42 server_seed_ok=3 trace_ok=3 cloud_seed=null``
- ``ASTRA_LANG_IO=PASS planner=<n> monitor=<n> action=<n> unresolved_steps=0 open_calls=0 image_ref_unresolved=0``
（``EVAL_CAP=PASS route=astra …`` 由 ``test_astra_wiring.py::test_strict_cap_overrun_finishes_as_timeout`` 打印。）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astra_fakes import (FakeEnv, FakeMonitor, FakeResponder, Harness, NetCounter, astra_session, ensure_language_log,
                         read_language, read_trace, template_text, third_party)
from tests.pipeline.evalx.report import trace_contract as tc


def _trace(path: Path) -> list[dict]:
    return read_trace(path)


def _ctx(mod, tmp_path: Path, *, cap: int, strict: bool, lang=None):
    from robomme_hard_eval.record import trace_writer as tw
    writer = tw.TraceWriter(tmp_path / "trace.jsonl", route="astra/new",
                            identity={"task": "BinFill", "dataset": "ood", "key": "k", "attempt": 1}, max_steps=cap)
    return mod.TraceContext(writer, effective_cap=cap, strict_cap=strict, lang=lang), writer


# ── 1800 步 strict cap（TracedEnv 自身；外层会话联动见 test_astra_wiring.py） ─────────

def test_ood_strict_cap_step_1800_runs_and_1801_never_reaches_env(tmp_path):
    """ood（strict，cap 1800）：第 1800 步照常进真实环境并落 trace；第 1801 次 ``step`` 在计数与动作追加之前被拒，
    真实环境步数仍 1800、trace 不多一行、``cap_hit=True``、抛 ``StepCapReached``。"""
    with astra_session() as (mod, _astra):
        assert mod.DATASET_STEP_PAIRING == {"hard-verify": 1300, "ood": 1800}
        assert mod.DATASET_STRICT_CAP == {"hard-verify": False, "ood": True}
        ctx, writer = _ctx(mod, tmp_path, cap=1800, strict=True)
        inner = FakeEnv(terminal_step=None)
        env = mod.TracedEnv(inner, ctx)
        env.reset()
        action = np.zeros(8, dtype=np.float32)
        for _ in range(1800):
            env.step(action)
        rows_before = (tmp_path / "trace.jsonl").read_text()
        assert inner.steps_taken == 1800 and ctx.attempted == 1800 and not ctx.cap_hit
        with pytest.raises(mod.StepCapReached, match="STEP_CAP exec_steps=1800 cap=1800"):
            env.step(action)
        assert inner.steps_taken == 1800, "第 1801 步不得进入真实环境"
        assert ctx.attempted == 1800 and len(ctx.actions) == 1800 and ctx.cap_hit is True
        assert (tmp_path / "trace.jsonl").read_text() == rows_before, "被拒的第 1801 步不得多落一行"
        writer.close(status="timeout", terminal_reason="timeout")
    steps = [r for r in _trace(tmp_path / "trace.jsonl") if r["kind"] == "step"]
    assert [r["step"] for r in steps][-1] == 1800 and len(steps) == 1800




def test_hard_verify_is_not_strict(tmp_path):
    """hard-verify（cap 1300）保持非 strict：第 1301 次 step 照常交给底层环境（截断靠环境自己）。"""
    with astra_session() as (mod, _astra):
        ctx, writer = _ctx(mod, tmp_path, cap=1300, strict=mod.strict_cap_of("hard-verify"))
        inner = FakeEnv(terminal_step=None)
        env = mod.TracedEnv(inner, ctx)
        env.reset()
        for _ in range(1301):
            env.step(np.zeros(8, dtype=np.float32))
        writer.close(status="timeout", terminal_reason="timeout")
        assert inner.steps_taken == 1301 and ctx.cap_hit is False
        assert mod.strict_cap_of("hard-verify") is False and mod.strict_cap_of("ood") is True


# ── 模型 seed ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, "", "abc", "-1", "7.5", True])
def test_policy_seed_missing_or_bad_blocks(value):
    with astra_session() as (mod, _astra):
        with pytest.raises(ValueError, match="RUN_BLOCKED reason=policy_seed"):
            mod.check_policy_seed(value)
        assert [mod.check_policy_seed(v) for v in (0, "7", 42)] == [0, 7, 42]



@pytest.mark.parametrize("seed", [-1])
def test_policy_seed_bad_blocks_before_any_server(tmp_path, monkeypatch, seed):
    """负数模型种子：``load`` 在起任何服务之前 ``RUN_BLOCKED reason=policy_seed``。"""
    from astra_fakes import FakeServer
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        with pytest.raises(ValueError, match="RUN_BLOCKED reason=policy_seed"):
            h.load(seed=seed)
    assert FakeServer.instances == []


def test_policy_seeds_service_command_and_trace(tmp_path, monkeypatch):
    """seed 0／7／42：① VLA 服务 ``--seed=<seed>``（不再是 42 常量），服务端元数据与 ``server_seed`` 反查一致；
    ② 记进 trace identity／end、外层结果、Astra ``result.json``、provenance；``cloud_seed`` 恒为 null（不伪造）。"""
    net = NetCounter().install(monkeypatch)
    server_ok = trace_ok = 0
    for seed in (0, 7, 42):
        sub = tmp_path / f"s{seed}"
        sub.mkdir()
        with astra_session() as (mod, astra):
            h = Harness(sub, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=20))
            policy = h.load(seed=seed)
            vla = h.servers()["astra-vla"]
            r = h.run("hard-verify", "BinFill", 0)
            policy.close()
        assert f"--seed={seed}" in vla.argv and not (seed != 42 and "--seed=42" in vla.argv)
        assert vla.policy_seed == seed and vla.metadata["policy_seed"] == seed and r.server_seed == seed
        server_ok += 1
        rows = _trace(h.raw(r) / "trace.jsonl")
        assert rows[0]["identity"]["policy_seed"] == seed
        if "policy_seed" in rows[0]:
            assert rows[0]["policy_seed"] == seed
        assert rows[-1]["policy_seed"] == seed and rows[-1]["cloud_seed"] is None
        assert rows[-1]["strict_cap"] is False and rows[-1]["effective_cap"] == 1300
        assert r.policy_seed == seed and r.extra["cloud_seed"] is None
        saved = json.loads((h.astra_ep_dir(r) / "result.json").read_text())
        assert saved["policy_seed"] == seed and saved["cloud_seed"] is None
        prov = json.loads((h.attempt_dir(r) / "provenance.json").read_text())
        assert prov["policy_seed"] == prov["server_seed"] == seed and prov["cloud_seed"] is None
        trace_ok += 1
    assert net.calls == 0
    print(f"POLICY_SEEDS=PASS route=astra seeds=0,7,42 server_seed_ok={server_ok} trace_ok={trace_ok} cloud_seed=null")


# ── 语言账本 ─────────────────────────────────────────────────────────────

def _calls(rows: list[dict]) -> dict:
    """按 call_id 聚合：open、messages、close。"""
    calls: dict = {}
    for r in rows:
        if r["kind"] == "call_open":
            calls[r["call_id"]] = {"open": r, "messages": [], "close": None}
        elif r["kind"] == "message":
            calls[r["call_id"]]["messages"].append(r)
        elif r["kind"] == "call_close":
            calls[r["call_id"]]["close"] = r
    return calls


def _known_shas(rows: list[dict]) -> tuple[dict, set]:
    """trace 里可解析的画面 sha：exec 帧 k（0 = demo 末帧，k = step k）与 demo 帧 i，外加所有 wrist sha。"""
    demo = next(r for r in rows if r["kind"] == "demo")
    exec_front = {0: demo["front_sha256"][-1]}
    for r in rows:
        if r["kind"] == "step":
            exec_front[r["step"]] = r["front_sha256"]
    wrists = set(demo["wrist_sha256"]) | {r["wrist_sha256"] for r in rows if r["kind"] == "step"}
    demo_front = dict(enumerate(demo["front_sha256"][:-1]))
    return {"exec": exec_front, "demo": demo_front}, wrists


def _unresolved_images(images: list, fronts: dict, wrists: set) -> int:
    bad = 0
    for img in images:
        for src in img["sources"]:
            if src.get("cam") == "wrist":
                ok = src.get("raw_sha256") in wrists
            else:
                ok = fronts[src["phase"]].get(src["frame_idx"]) == src.get("raw_sha256")
            bad += 0 if ok and src.get("raw_sha256") else 1
    return bad




def test_language_log_planner_monitor_action(tmp_path, monkeypatch):
    """两局（VideoUnmask 有演示拼图、ButtonUnmask 有执行记忆拼图）：planner／monitor／action_model 三类调用齐全、
    输入原文与 spool／上游输入契约逐字相同、附图引用全部能在 trace 里解析、每个执行步可追溯到 action_model 调用、
    第二次规划回不合模板的原文时记 ``fallback=continue_last``；来源清单含 prompts 的 sha256。"""
    net = NetCounter().install(monkeypatch)
    tasks = ["VideoUnmask", "ButtonUnmask"]
    with astra_session() as (mod, astra):
        kind = ensure_language_log(monkeypatch)
        import input_contract
        texts = [template_text(astra.champ, "VideoUnmask"), "I cannot comply.",
                 template_text(astra.champ, "ButtonUnmask"), "I cannot comply."]
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=60),
                    monitor=FakeMonitor(predictions=[False, True, False, True]),
                    responder=FakeResponder(astra.champ, texts=texts))
        policy = h.load()
        results = [h.run("hard-verify", t, 0) for t in tasks]
        policy.close()
        normal = input_contract.NORMAL
    assert [r.status for r in results] == ["success", "success"]
    totals = {"planner": 0, "monitor": 0, "action_model": 0}
    unresolved_steps = open_calls = image_bad = 0
    spool = policy.spool
    for task, r in zip(tasks, results):
        raw = h.raw(r)
        rows = _trace(raw / "trace.jsonl")
        lang_rows = read_language(raw / "language.jsonl")
        calls = _calls(lang_rows)
        fronts, wrists = _known_shas(rows)
        by_model: dict = {}
        for cid, c in calls.items():
            by_model.setdefault(c["open"]["model"], []).append(c)
            open_calls += c["close"] is None
            ins = [m for m in c["messages"] if m["dir"] == "in"]
            assert ins, f"调用 {cid} 没有输入"
            pos_in = lang_rows.index(ins[-1])
            assert c["close"] is None or pos_in < lang_rows.index(c["close"])
            for m in ins:
                image_bad += _unresolved_images(m.get("images") or [], fronts, wrists)
        planner = by_model["planner"]
        assert len(planner) == r.extra["planner_calls"] == 2
        first, second = planner
        (msg_in,) = [m for m in first["messages"] if m["dir"] == "in"]
        rid = first["open"]["params"]["request_id"]
        assert msg_in["role"] == "user" and msg_in["text"] == (spool / rid / "prompt.txt").read_text()
        req = json.loads((spool / rid / "request.json").read_text())
        assert [i["slot"] for i in msg_in["images"]] == list(range(len(req["images"])))
        assert msg_in["images"][0]["ref"] == "current" and msg_in["images"][1]["ref"] == "keyframe"
        assert first["open"]["params"]["cloud_seed"] is None and first["open"]["transport_attempt"] == 0
        assert first["close"]["status"] == "reply" and first["close"]["parsed"] == texts[0 if task == tasks[0] else 2]
        assert first["close"]["fallback"] is None
        assert [m["text"] for m in second["messages"] if m["dir"] == "out"] == ["I cannot comply."]
        assert second["close"]["parsed"] is None and second["close"]["fallback"] == "continue_last"
        refs = {i["ref"] for c in planner for m in c["messages"] for i in (m.get("images") or [])}
        assert ("demo_sheet" in refs) == (task == "VideoUnmask") and ("memory_sheet" in refs) == (task == "ButtonUnmask")
        monitors = by_model["monitor"]
        assert len(monitors) == r.extra["monitor_calls"] and monitors
        for c in monitors:
            sys_m, user_m = [m for m in c["messages"] if m["dir"] == "in"]
            assert sys_m["role"] == "system" and sys_m["text"] == normal and sys_m["message_index"] == 0
            assert user_m["role"] == "user" and len(user_m["images"]) == 10
            assert [i["ref"] for i in user_m["images"]] == ["recent"] * 8 + ["command_start", "wrist"]
            assert "Current grounded subgoal:" in user_m["text"]
            (reply,) = [m for m in c["messages"] if m["dir"] == "out"]
            assert reply["text"] in ("true", "false") and c["close"]["parsed"] == (reply["text"] == "true")
        actions = by_model["action_model"]
        assert len(actions) == sum(1 for x in rows if x["kind"] == "request" and x["name"] == "vla_infer")
        goal = next(x for x in rows if x["kind"] == "demo")["texts"][0]
        for c in actions:
            (m,) = c["messages"]
            assert m["role"] == "fields" and m["text"]["prompt"] == goal and m["text"]["grounded_subgoal"]
            assert c["close"]["status"] == "reply" and c["close"]["server_final_text"] is None
        action_ids = {c["open"]["call_id"] for c in actions}
        for x in rows:
            if x["kind"] == "step":
                ok = x.get("source_call_id") in action_ids and 0 <= x.get("chunk_index", -1) < 16
                unresolved_steps += 0 if ok else 1
        for k in totals:
            totals[k] += len(by_model.get(k, []))
        prov = json.loads((h.attempt_dir(r) / "provenance.json").read_text())
        champ = third_party() / "examples" / "champ"
        assert prov["prompts"]["prompts/index.json"] == mod.file_sha256(champ / "prompts" / "index.json")
        assert prov["prompts"][f"prompts/{task}.md"] == mod.file_sha256(champ / "prompts" / f"{task}.md")
        assert prov["language_log"] == "language.jsonl" and rows[-1]["language"] == "language.jsonl"
        assert tc.contract_problems(raw) == []
    assert unresolved_steps == 0 and open_calls == 0 and image_bad == 0
    assert net.calls == 0
    print(f"ASTRA_LANG_IO=PASS planner={totals['planner']} monitor={totals['monitor']} action={totals['action_model']} "
          f"unresolved_steps={unresolved_steps} open_calls={open_calls} image_ref_unresolved={image_bad} log={kind}")


def test_language_input_persisted_before_send_failure(tmp_path, monkeypatch):
    """假 planner 在「发送时」抛异常（不写 response.json）：prompt 全文与附图引用已先落盘，调用以 ``status=error`` 收尾；
    按停机规则⑤（Planner 前缀）整批停，没有任何外联。"""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        ensure_language_log(monkeypatch)
        responder = FakeResponder(astra.champ, raise_on_send=RuntimeError("Planner API fake transport failure"))
        h = Harness(tmp_path, monkeypatch, mod, astra, env_plan=lambda b, ep: FakeEnv(terminal_step=40),
                    responder=responder)
        policy = h.load()
        results, stop = h.batch("hard-verify", ["BinFill", "VideoUnmask"])
        policy.close()
    assert stop.reason == "planner_error" and len(results) == 1
    calls = _calls(read_language(h.raw(results[0]) / "language.jsonl"))
    (call,) = [c for c in calls.values() if c["open"]["model"] == "planner"]
    (msg_in,) = call["messages"]
    rid = call["open"]["params"]["request_id"]
    assert msg_in["dir"] == "in" and msg_in["text"] == (policy.spool / rid / "prompt.txt").read_text()
    assert msg_in["images"] and not (policy.spool / rid / "response.json").exists()
    assert call["close"]["status"] == "error"
    assert net.calls == 0

def test_transport_retry_opens_new_call_with_attempt(tmp_path, monkeypatch):
    """真实 ``GuardedResponsesClient``（零外联替身）遇 429 后重试成功：语言账本两个 planner 调用，
    ``transport_attempt`` 0（error）与 1（reply），输入原文相同；费用守卫行为不变（urlopen 恰 2 次、同一份预留）。"""
    import urllib.request
    from test_astra_wiring import Clock, FakeUrlopen, GuardFixture, _client, _http_429, _resp
    fake = FakeUrlopen([_http_429(), {"input_tokens": 10}])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        ensure_language_log(monkeypatch)
        from robomme_hard_eval.record import trace_writer as tw
        client, _gate = _client(mod, astra, fx, clock)
        lang = tw.LanguageLog(tmp_path / "lang" / "language.jsonl")
        ctx = mod.TraceContext(None, lang=lang)
        call = mod.PlannerLanguageCall(ctx, lambda out, req: [], "predict")
        out = fx.request(1000)
        call.wrap(client)(out)
        call.finish(parsed="x", fallback=None)
        lang.close()
        assert client.transport_hook is None, "调用结束后回调还原"
    assert fake.calls == 2 and _resp(out)["status"] == "ok"
    assert list(fx.reservations()) == [out.name]
    calls = list(_calls(read_language(tmp_path / "lang" / "language.jsonl")).values())
    assert [c["open"]["transport_attempt"] for c in calls] == [0, 1]
    assert [c["close"]["status"] for c in calls] == ["error", "reply"]
    ins = [[m["text"] for m in c["messages"] if m["dir"] == "in"] for c in calls]
    assert ins[0] == ins[1] == [(out / "prompt.txt").read_text()]
    assert [m["text"] for m in calls[1]["messages"] if m["dir"] == "out"] == ["ok"]


# ── arrays.npz 合并写 ─────────────────────────────────────────────────────

def test_write_exec_actions_uses_merge_write_npz_when_present(tmp_path, monkeypatch):
    """``trace_writer.merge_write_npz`` 存在时 ``write_exec_actions`` 只经它写（键 ``exec_action__%05d``）；
    不存在时保持旧的 ``np.savez``。"""
    with astra_session() as (mod, _astra):
        from robomme_hard_eval.record import trace_writer as tw
        actions = [np.full(8, i, dtype=np.float32) for i in range(3)]
        seen = []
        real = getattr(tw, "merge_write_npz", None)

        def spy(path, mapping):
            seen.append((Path(path), sorted(mapping)))
            if real is not None:
                return real(path, mapping)
            np.savez(path, **mapping)

        monkeypatch.setattr(tw, "merge_write_npz", spy, raising=False)
        (tmp_path / "a").mkdir()
        mod.write_exec_actions(tmp_path / "a" / "arrays.npz", actions)
        assert seen == [(tmp_path / "a" / "arrays.npz", [f"exec_action__{i:05d}" for i in range(3)])]
        monkeypatch.delattr(tw, "merge_write_npz", raising=False)
        (tmp_path / "b").mkdir()
        mod.write_exec_actions(tmp_path / "b" / "arrays.npz", actions)
        with np.load(tmp_path / "b" / "arrays.npz") as arr:
            assert sorted(arr.files) == [f"exec_action__{i:05d}" for i in range(3)]
            assert arr["exec_action__00002"].dtype == np.float32
