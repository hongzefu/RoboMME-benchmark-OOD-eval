"""GroundSG 第三阶段接线（1006 计划八.3、八.10 第 3、8 条、八.11；接口冻结说明 2.2～2.4、三、五）。

* 语言账本：R6 的 ``trace_writer.LanguageLog`` 尚未合入，这里按冻结说明第五节的签名手写一个严格替身（只收冻结的
  关键字参数，名字写错即 TypeError），装到被测模块实际用的那份 ``trace_writer`` 上；核 QwenVL／MemER 每次提问一次
  ``subgoal_model`` 调用（MemER 重问各自一次、带 retry）、QwenVL keep_period 复用记 reuse、Oracle 记
  ``subgoal_source: oracle``、动作服务回包 ``_sgeval_audit`` 交给官方代码前被 pop 掉并记账、执行步关联
  ``source_call_id``／``chunk_index``、附图哈希能在轨迹帧哈希里找到、没有悬空调用。
* 原侧共享账本：手写假账本按调用顺序记账（含冻结说明三.3 的 token 幂等语义），核
  reserve → claim_reset(build) → claim_reset(reset) → commit；第 2 次尝试先 claim_retry；名额被拒不建局目录；
  轨迹额度不足停片；同一 token 续跑不重复扣额。
* 入口：``official_hard_runner.main`` 的 policy_seed／budget_args／MemER adapter 配对／步数配对（参数核对先于任何导入，
  进程内调用）；``run_official_hard.sh`` 的必填项与驱动命令转发。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import groundsg_fakes as F
from tests._support.loaders import REPO

EO = REPO / "scripts" / "eval-official"
NOTE = "Your previous reply was not valid JSON. Reply with the JSON object only."
BAD = "not json at all"


# ---------------------------------------------------------------- 语言账本替身（冻结说明第五节签名）


class StrictLanguageLog:
    """``trace_writer.LanguageLog`` 的替身：方法与关键字参数照冻结说明逐字（多一个、错一个名字都会 TypeError）。"""

    instances: list["StrictLanguageLog"] = []

    def __init__(self, path):
        self.path = Path(path)
        self.rows: list[dict] = []
        self.open: dict[str, int] = {}
        self.n = 0
        self.closed = False
        StrictLanguageLog.instances.append(self)

    def open_call(self, model, step, *, params=None, transport_attempt=0, retry=0):
        assert model in ("subgoal_model", "action_model", "planner", "monitor") and not self.closed
        cid = f"c{self.n:04d}"
        self.n += 1
        self.open[cid] = 0
        self.rows.append({"kind": "call_open", "call_id": cid, "model": model, "step": step, "params": params,
                          "transport_attempt": transport_attempt, "retry": retry})
        return cid

    def message(self, call_id, *, dir, role, text, images=None, channel=None, token_ids=None, mask=None,
                tokenizer=None, truncated=None, demo_video=None):
        assert call_id in self.open and dir in ("in", "out")
        idx = self.open[call_id]
        self.open[call_id] += 1
        self.rows.append({"kind": "message", "call_id": call_id, "message_index": idx, "dir": dir, "role": role,
                          "text": text, "images": images, "channel": channel, "token_ids": token_ids, "mask": mask,
                          "tokenizer": tokenizer, "truncated": truncated, "demo_video": demo_video})
        return idx

    def close_call(self, call_id, *, status, parsed=None, fallback=None, server_final_text=None,
                   server_truncated=None):
        assert call_id in self.open and status in ("reply", "error", "cancelled")
        assert fallback in (None, "last_valid", "model_response_error", "continue_last")
        del self.open[call_id]
        self.rows.append({"kind": "call_close", "call_id": call_id, "status": status, "parsed": parsed,
                          "fallback": fallback, "server_final_text": server_final_text,
                          "server_truncated": server_truncated})

    def reuse(self, step, reused_call_id):
        self.rows.append({"kind": "reuse", "step": step, "reused_call_id": reused_call_id, "reused_previous": True})

    def close(self):
        for cid in list(self.open):
            self.close_call(cid, status="cancelled")
        self.closed = True

    # 便于断言
    def calls(self, model=None) -> list[dict]:
        out = []
        for r in self.rows:
            if r["kind"] == "call_open" and (model is None or r["model"] == model):
                cid = r["call_id"]
                out.append({"open": r, "messages": [m for m in self.rows if m["kind"] == "message"
                                                    and m["call_id"] == cid],
                            "close": next(c for c in self.rows if c["kind"] == "call_close" and c["call_id"] == cid)})
        return out


@pytest.fixture
def lang(monkeypatch):
    StrictLanguageLog.instances = []
    tw = F.groundsg_client().trace_writer
    monkeypatch.setattr(tw, "LanguageLog", StrictLanguageLog, raising=False)
    return StrictLanguageLog


class AuditServer(F.FakeServer):
    """动作服务替身：infer 回包另带 ``_sgeval_audit``（R3 外壳的冻结格式）。"""

    def handle(self, obj):
        out = super().handle(obj)
        if "actions" in out:
            out["_sgeval_audit"] = {"channels": [
                {"channel": "task", "text": f"Task: {obj.get('prompt')};", "token_ids": [1, 2], "mask": [1, 1],
                 "tokenizer": "fake", "truncated": False},
                {"channel": "symbolic", "text": f"Current Subgoal: {obj.get('grounded_subgoal')};",
                 "token_ids": [3], "mask": [1], "tokenizer": "fake", "truncated": True}],
                "server_final_text": f"Task: {obj.get('prompt')};\nCurrent Subgoal: {obj.get('grounded_subgoal')};\nAction: ",
                "pp_generation": None}
        return out


def _trace_hashes(rows) -> set:
    demo = next(r for r in rows if r["kind"] == "demo")
    hs = set(demo["front_sha256"]) | set(demo["wrist_sha256"])
    for r in rows:
        if r["kind"] == "step":
            hs |= {r.get("front_sha256"), r.get("wrist_sha256")}
    return hs - {None}


def _check_common(log: StrictLanguageLog, trace_path: str) -> None:
    """无悬空调用；每个执行步都能追溯到一个 action_model 调用；附图哈希都在轨迹帧哈希里。"""
    assert log.closed and log.open == {}
    rows = F.read_trace(trace_path)
    action_ids = {c["open"]["call_id"] for c in log.calls("action_model")}
    steps = [r for r in rows if r["kind"] == "step"]
    assert steps and all(r["source_call_id"] in action_ids for r in steps)
    # 每个动作块内 chunk_index 从 0 连续
    by_call: dict[str, list[int]] = {}
    for r in steps:
        by_call.setdefault(r["source_call_id"], []).append(r["chunk_index"])
    assert all(v == list(range(len(v))) for v in by_call.values())
    hs = _trace_hashes(rows)
    for m in log.rows:
        if m["kind"] == "message":
            for img in m["images"] or []:
                assert img["raw_sha256"] in hs, img


def test_memer_language_log_retries_and_audit(tmp_path, lang):
    """MemER 首问「坏、坏、好」：三次 subgoal_model 调用 retry=0,1,2，前两次 parsed=None 关闭，第三次带 parsed；
    重问 user 原文末尾带提醒句、temperature 0.7；附图引用为 recent、frame_idx=0；有演示视频记 demo[0:2]。"""
    swift = F.FakeSwift(script=[BAD, BAD, '{"current_subtask": "go <|box_start|>(500,500)<|box_end|>", '
                                          '"keyframe_positions": []}'])
    side = F.NewSide(F.MEMER, 60, tmp_path, F.World(default=F.Plan(success_at=20)), swift=swift,
                     server=AuditServer())
    res = side.run(F.identity())
    assert res["status"] == "success" and res["language"]["subgoal_calls"] == 4  # 首问 3 次 + 第 16 步 1 次
    (log,) = lang.instances
    assert log.path == Path(res["trace_path"]).parent / "language.jsonl"
    sub = log.calls("subgoal_model")
    assert [c["open"]["retry"] for c in sub] == [0, 1, 2, 0]
    assert [c["open"]["params"]["temperature"] for c in sub] == [0, 0.7, 0.7, 0]
    assert all(c["open"]["params"]["adapter"] == F.MEMER_ADAPTER and c["open"]["params"]["policy_seed"] == 7
               and c["open"]["params"]["memer_compat_sha256"] for c in sub)
    assert [c["close"]["parsed"] for c in sub[:2]] == [None, None]
    assert sub[2]["close"]["parsed"]["subgoal"] == "go <128, 128>" and sub[2]["close"]["fallback"] is None
    for c in sub:
        roles = [(m["dir"], m["role"]) for m in c["messages"]]
        assert roles == [("in", "system"), ("in", "user"), ("out", "assistant")]
    u0, u1 = sub[0]["messages"][1], sub[1]["messages"][1]
    assert u1["text"] == u0["text"] + "\n" + NOTE and u0["demo_video"] == "demo[0:2]"
    assert [(i["ref"], i["frame_idx"], i["cam"]) for i in u0["images"]] == [("recent", 0, "front")]
    assert [m["text"] for m in (c["messages"][2] for c in sub[:3])] == [BAD, BAD, swift_reply_of(sub[2])]
    # 动作模型：in 为结构化字段，子目标来自子目标模型；审计键被 pop 掉、记进调用
    act = log.calls("action_model")
    assert len(act) == 2 and all(c["close"]["server_final_text"].endswith("Action: ") for c in act)
    f0 = act[0]["messages"][0]
    assert f0["role"] == "fields" and f0["text"]["subgoal_source"] == "subgoal_model"
    assert f0["text"]["grounded_subgoal"] == "go <128, 128>" and f0["text"]["subgoal_call_id"] == sub[2]["open"]["call_id"]
    assert [m["channel"] for m in act[0]["messages"][1:]] == ["task", "symbolic"]
    assert act[0]["close"]["server_truncated"] is True
    _check_common(log, res["trace_path"])


def swift_reply_of(call) -> str:
    return call["messages"][2]["text"]


def test_memer_language_log_fallback_and_named_error(tmp_path, lang):
    """第二次提问「坏坏坏」且已有上一次：fallback=last_valid；新局首问「坏坏坏」：fallback=model_response_error。"""
    good = '{"current_subtask": "first", "keyframe_positions": []}'
    swift = F.FakeSwift(script=[good, BAD, BAD, BAD, BAD, BAD, BAD])
    side = F.NewSide(F.MEMER, 60, tmp_path, F.World(plans={3: F.Plan(success_at=20), 7: F.Plan(success_at=20)}),
                     swift=swift)
    r1 = side.run(F.identity())
    assert r1["status"] == "success"
    sub = lang.instances[0].calls("subgoal_model")
    assert [c["open"]["retry"] for c in sub] == [0, 0, 1, 2]
    assert sub[3]["close"]["fallback"] == "last_valid" and sub[3]["close"]["parsed"]["subgoal"] == "first"
    r2 = side.run(F.identity(source_episode=7, builder_episode=1, seed=510700))
    assert (r2["status"], r2["error_kind"]) == ("error", "model_response_error")
    sub2 = lang.instances[1].calls("subgoal_model")
    assert [c["open"]["retry"] for c in sub2] == [0, 1, 2]
    assert sub2[-1]["close"]["fallback"] == "model_response_error" and lang.instances[1].open == {}


def test_qwenvl_reuse_and_oracle_source(tmp_path, lang):
    """QwenVL：ButtonUnmask 的 keep_period=90，第 16、32 步复用上一回复，记 reuse 指向首问调用；
    Oracle：没有 subgoal_model 调用，动作字段 subgoal_source=oracle；旧服务无审计键 → server_final_text=None。"""
    ident = F.identity(task="ButtonUnmask")
    side = F.NewSide(F.QWENVL, 60, tmp_path / "q", F.World(default=F.Plan(success_at=40)))
    res = side.run(ident)
    assert res["status"] == "success"
    log = lang.instances[-1]
    sub = log.calls("subgoal_model")
    reuses = [r for r in log.rows if r["kind"] == "reuse"]
    assert len(sub) == 1 and [r["step"] for r in reuses] == [16, 32]
    assert all(r["reused_call_id"] == sub[0]["open"]["call_id"] for r in reuses)
    assert [(i["ref"], i["frame_idx"]) for i in sub[0]["messages"][1]["images"]] == [("current", 0)]
    assert sub[0]["close"]["parsed"]["subgoal"] == side.server.log[2][2]["grounded_subgoal"]
    _check_common(log, res["trace_path"])
    side_o = F.NewSide(F.ORACLE, 60, tmp_path / "o", F.World(default=F.Plan(success_at=20)))
    res_o = side_o.run(F.identity())
    log_o = lang.instances[-1]
    assert log_o.calls("subgoal_model") == []
    act = log_o.calls("action_model")
    assert [c["messages"][0]["text"]["subgoal_source"] for c in act] == ["oracle", "oracle"]
    assert [c["messages"][0]["text"]["grounded_subgoal"] for c in act] == [F.subgoal_at(0), F.subgoal_at(16)]
    assert all(c["close"]["server_final_text"] is None and c["close"]["status"] == "reply" for c in act)
    _check_common(log_o, res_o["trace_path"])


def test_orig_side_writes_language_log(tmp_path, lang):
    orig = F.OrigSide(F.MEMER, 60, tmp_path, F.World(default=F.Plan(success_at=20)))
    row = orig.run(F.identity())
    assert row["status"] == "success"
    (log,) = lang.instances
    assert log.path == Path(row["ep_dir"]) / "language.jsonl"
    assert len(log.calls("subgoal_model")) == 2 and len(log.calls("action_model")) == 2
    _check_common(log, row["trace_path"])


def test_audit_key_popped_without_language_log(tmp_path):
    """语言账本没有时审计键也照样在交给官方代码前 pop 掉，官方循环拿到的回包与旧服务相同。"""
    mc = F.groundsg_client()
    tap = mc.EpisodeTap(None)
    server = AuditServer()
    tc = mc.TracingClient(F.FakeClient(server), tap)
    tc.reset()
    obs = {"observation/image": F.frame(1), "observation/wrist_image": F.frame(2),
           "observation/state": F.frame(3)[0, 0, :].astype("float32"), "prompt": "p"}
    out = tc.infer(obs)
    assert set(out) == {"actions"}


class RecordingLikeClient(F.FakeClient):
    """模仿 ``framesamp_modul_client.RecordingClient``：``_roundtrip`` 里先把审计键 pop 掉、存进 ``_last_audit``，
    交给上层的回包已不含审计键（真实 smoke 里语言账本 ``server_final_text`` 全为 null 的来源）。"""

    def __init__(self, server):
        super().__init__(server)
        self._last_audit = None
        self.seen: list[dict] = []

    def infer(self, obs):
        out = self.server.handle(obs)
        self._last_audit = out.pop(mc_audit_key(), None) if isinstance(out, dict) else None
        self.seen.append(dict(out))
        return out


def mc_audit_key() -> str:
    return F.groundsg_client().AUDIT_KEY


class ToggleAuditServer(AuditServer):
    """第一次 infer 回包带审计块，之后不带（核第二次不误用第一次的块）。"""

    def __init__(self):
        super().__init__()
        self.n_infer = 0

    def handle(self, obj):
        out = super().handle(obj)
        if "actions" in out:
            self.n_infer += 1
            if self.n_infer > 1:
                out.pop("_sgeval_audit", None)
        return out


def test_audit_read_back_from_recording_client_last_audit(tmp_path):
    """内层客户端已在 ``_roundtrip`` 里 pop 审计键时，``TracingClient`` 回落读 ``_last_audit``：语言账本记 task／symbolic
    两条 channel 消息、``server_final_text`` 非空；交给官方代码的 dict 不含审计键；第二次 infer（回包无审计块）调用前
    清零，不会误用第一次的块。"""
    mc = F.groundsg_client()
    tap = mc.EpisodeTap(None)
    log = StrictLanguageLog(tmp_path / "language.jsonl")
    tap.lang = mc.LangTap(log, tap, variant="memer")
    inner = RecordingLikeClient(ToggleAuditServer())
    tc = mc.TracingClient(inner, tap)
    tc.reset()
    obs = {"observation/image": F.frame(1), "observation/wrist_image": F.frame(2),
           "observation/state": F.frame(3)[0, 0, :].astype("float32"), "prompt": "p", "grounded_subgoal": "g"}
    out1 = tc.infer(obs)
    assert mc.AUDIT_KEY not in out1 and set(out1) == {"actions"}
    out2 = tc.infer(obs)
    assert mc.AUDIT_KEY not in out2
    assert all(mc.AUDIT_KEY not in s for s in inner.seen)
    calls = log.calls("action_model")
    assert len(calls) == 2
    first, second = calls
    chans = [m for m in first["messages"] if m["channel"] is not None]
    assert [m["channel"] for m in chans] == ["task", "symbolic"]
    assert chans[0]["text"] == "Task: p;" and chans[1]["token_ids"] == [3]
    assert first["close"]["server_final_text"] == "Task: p;\nCurrent Subgoal: g;\nAction: "
    assert first["close"]["server_truncated"] is True
    assert [m for m in second["messages"] if m["channel"] is not None] == []
    assert second["close"]["server_final_text"] is None and inner._last_audit is None


def test_step_rows_unchanged_without_language_log(tmp_path, monkeypatch):
    """语言账本不存在（显式去掉）时 step 行仍是旧 9 键（不加 source_call_id／chunk_index）。"""
    tw = F.groundsg_client().trace_writer
    monkeypatch.setattr(tw, "LanguageLog", None, raising=False)
    side = F.NewSide(F.ORACLE, 60, tmp_path, F.World(default=F.Plan(success_at=5)))
    res = side.run(F.identity())
    step = next(r for r in F.read_trace(res["trace_path"]) if r["kind"] == "step")
    assert set(step) == {"kind", "step", "front_sha256", "wrist_sha256", "state", "action", "subgoal", "terminated",
                         "truncated", "status"}
    assert not (Path(res["trace_path"]).parent / "language.jsonl").exists() and "language" not in res


# ---------------------------------------------------------------- 原侧共享账本（冻结说明三）


class FakeLedger:
    """冻结说明三的接口替身：按调用顺序记账；reserve／claim_retry 按 token 幂等（同 token 返回同 rid／True、不记新行）。"""

    def __init__(self, *, deny_retry: bool = False, exhaust_after: int | None = None):
        self.calls: list[tuple] = []
        self.rows: list[dict] = []
        self.tokens: dict[str, str] = {}
        self.retry_tokens: set[str] = set()
        self.deny_retry, self.exhaust_after = deny_retry, exhaust_after

    def reserve(self, *, resets, route, key, token, kind_of_try, **extra):
        self.calls.append(("reserve", token, kind_of_try, resets, route, key))
        if token in self.tokens:
            return self.tokens[token]
        if self.exhaust_after is not None and len(self.tokens) >= self.exhaust_after:
            e = RuntimeError("trajectories exhausted")
            e.budget_exhausted = True
            raise e
        rid = f"rid{len(self.tokens)}"
        self.tokens[token] = rid
        self.rows.append({"kind": "reserve", "rid": rid, "token": token, "kind_of_try": kind_of_try})
        return rid

    def claim_retry(self, *, route, key, interrupt, token, **extra):
        self.calls.append(("claim_retry", token, interrupt, route, key))
        if token in self.retry_tokens:
            return True
        if self.deny_retry:
            return False
        self.retry_tokens.add(token)
        self.rows.append({"kind": "retry_claim", "token": token, "interrupt": interrupt})
        return True

    def claim_reset(self, rid, what, **extra):
        self.calls.append(("claim_reset", rid, what))
        self.rows.append({"kind": "reset_claim", "rid": rid, "what": what})

    def commit(self, rid, *, resets=None, **extra):
        self.calls.append(("commit", rid, extra.get("status"), extra.get("infra")))
        self.rows.append({"kind": "commit", "rid": rid})

    def release(self, rid, **extra):
        self.calls.append(("release", rid, extra.get("status")))
        self.rows.append({"kind": "release", "rid": rid})


ROUTE = f"groundsg/{F.ORACLE}/seed7/orig"


def _rows2():
    return [F.identity(), F.identity(source_episode=7, builder_episode=1, seed=510700)]


def test_orig_budget_call_sequence_first_try(tmp_path):
    orig = F.OrigSide(F.ORACLE, 60, tmp_path, F.World(default=F.Plan(success_at=5)))
    led = FakeLedger()
    budget = orig.ohr.OrigBudget(led, variant=F.ORACLE, policy_seed=7)
    assert budget.route == ROUTE
    summary = orig.ohr.run_shard(orig.ctx, _rows2(), out=tmp_path / "out", budget=budget)
    assert summary["episodes"] == 2 and not summary["budget_blocked"]
    k1, k2 = (r["key"] for r in _rows2())
    assert led.calls == [
        ("reserve", f"{ROUTE}|{k1}|a1", "first", 2, ROUTE, k1), ("claim_reset", "rid0", "build"),
        ("claim_reset", "rid0", "reset"), ("commit", "rid0", "success", False),
        ("reserve", f"{ROUTE}|{k2}|a1", "first", 2, ROUTE, k2), ("claim_reset", "rid1", "build"),
        ("claim_reset", "rid1", "reset"), ("commit", "rid1", "success", False)]
    got = [json.loads(x) for x in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert [(g["budget_token"], g["budget_rid"], g["policy_seed"]) for g in got] == [
        (f"{ROUTE}|{k1}|a1", "rid0", 7), (f"{ROUTE}|{k2}|a1", "rid1", 7)]


def test_orig_budget_retry_is_idempotent_by_token(tmp_path):
    """第 2 次尝试（infra 重试）：先 claim_retry 再 reserve(kind_of_try=recovery)；崩溃续跑（同 route／key／attempt）
    用同一 token，账本不多记一行。"""
    led = FakeLedger()
    k1 = F.identity()["key"]
    for i in (1, 2):  # 第二遍模拟续跑：新的输出目录、同一账本
        orig = F.OrigSide(F.ORACLE, 60, tmp_path / f"run{i}", F.World(default=F.Plan(success_at=5)))
        budget = orig.ohr.OrigBudget(led, variant=F.ORACLE, policy_seed=7)
        orig.ohr.run_shard(orig.ctx, [F.identity()], out=tmp_path / f"run{i}", attempt=2, budget=budget)
    tok = f"{ROUTE}|{k1}|a2"
    seq = [c[:3] for c in led.calls if c[0] in ("claim_retry", "reserve")]
    assert seq == [("claim_retry", tok, "infra"), ("reserve", tok, "recovery")] * 2
    assert [r["kind"] for r in led.rows if r["kind"] in ("retry_claim", "reserve")] == ["retry_claim", "reserve"]


def test_orig_budget_retry_denied_and_exhausted(tmp_path):
    orig = F.OrigSide(F.ORACLE, 60, tmp_path, F.World(default=F.Plan(success_at=5)))
    led = FakeLedger(deny_retry=True)
    budget = orig.ohr.OrigBudget(led, variant=F.ORACLE, policy_seed=7)
    summary = orig.ohr.run_shard(orig.ctx, [F.identity()], out=tmp_path / "d", attempt=2, budget=budget)
    assert summary["retry_denied"] == 1 and summary["episodes"] == 0
    assert [c[0] for c in led.calls] == ["claim_retry"]  # 拒了就不预约
    (row,) = [json.loads(x) for x in (tmp_path / "d" / "results.jsonl").read_text().splitlines()]
    assert (row["status"], row["infra"], row["infra_reason"], row["attempt"]) == \
        ("error", True, "budget_retry_denied", 2)
    assert not (tmp_path / "d" / f"{row['key']}.a2").exists()
    led2 = FakeLedger(exhaust_after=1)
    budget2 = orig.ohr.OrigBudget(led2, variant=F.ORACLE, policy_seed=7)
    s2 = orig.ohr.run_shard(orig.ctx, _rows2(), out=tmp_path / "e", budget=budget2)
    assert s2["budget_blocked"] is True and s2["episodes"] == 1
    k2 = _rows2()[1]["key"]
    assert not (tmp_path / "e" / f"{k2}.a1").exists()


def test_orig_budget_releases_when_never_built(tmp_path, monkeypatch):
    orig = F.OrigSide(F.ORACLE, 60, tmp_path, F.World(default=F.Plan(success_at=5)))

    def no_runner(*a, **kw):
        raise RuntimeError("builder ctor failed")

    monkeypatch.setattr(orig.ohr, "runner_for", no_runner)
    led = FakeLedger()
    orig.ohr.run_shard(orig.ctx, [F.identity()], out=tmp_path / "r",
                       budget=orig.ohr.OrigBudget(led, variant=F.ORACLE, policy_seed=7))
    assert [c[0] for c in led.calls] == ["reserve", "release"]


def test_orig_budget_route_carries_variant_and_seed():
    ohr = F.official_hard_runner()
    for v in F.VARIANTS:
        for s in (0, 7, 42):
            assert ohr.budget_route(v, s) == f"groundsg/{v}/seed{s}/orig"
            assert ohr.budget_token(ohr.budget_route(v, s), "K", 2) == f"groundsg/{v}/seed{s}/orig|K|a2"


# ---------------------------------------------------------------- 入口参数


GOOD_ARGV = ["--shard", "/nonexistent/shard.json", "--out", "/nonexistent/out", "--port", "1",
             "--variant", F.ORACLE, "--policy-seed", "7", "--budget-ledger", "/nonexistent/l.jsonl",
             "--trajectory-cap", "870", "--shared-infra-cap", "50", "--expired-cap", "50",
             "--planned-first-tries", "821"]


def _drop(argv, flag):
    i = argv.index(flag)
    return argv[:i] + argv[i + 2:]


def _set(argv, flag, value):
    i = argv.index(flag)
    return argv[:i + 1] + [value] + argv[i + 2:]


def test_runner_cli_blocks_before_any_import(capsys):
    """参数核对先于导入：缺 policy_seed、缺任一预算参数、MemER adapter 配错、步数非 1300 → 退出 3 与具名原因。"""
    ohr = F.official_hard_runner()
    cases = [(_drop(GOOD_ARGV, "--policy-seed"), "RUN_BLOCKED reason=policy_seed"),
             (_set(GOOD_ARGV, "--policy-seed", "-1"), "RUN_BLOCKED reason=policy_seed"),
             (_set(GOOD_ARGV, "--variant", F.MEMER), "GROUNDSG_ORIG_BLOCKED reason=args --memer-adapter"),
             (GOOD_ARGV + ["--memer-adapter", "/m"], "GROUNDSG_ORIG_BLOCKED reason=args --memer-adapter"),
             (_set(GOOD_ARGV, "--variant", F.MEMER) + ["--memer-adapter", "/m", "--qwenvl-groundsg-adapter", "/q"],
              "GROUNDSG_ORIG_BLOCKED reason=args --qwenvl-groundsg-adapter"),
             (GOOD_ARGV + ["--max-steps", "1600"], "RUN_BLOCKED reason=step_cap_pairing")]
    for flag in ("--budget-ledger", "--trajectory-cap", "--shared-infra-cap", "--expired-cap",
                 "--planned-first-tries"):
        cases.append((_drop(GOOD_ARGV, flag), f"RUN_BLOCKED reason=budget_args missing={flag}"))
    for argv, want in cases:
        assert ohr.main(argv) == 3, argv
        out = capsys.readouterr().out
        assert want in out, (argv, out)


def test_run_official_hard_requires_stage3_args(tmp_path):
    """启动器：缺 --policy-seed 或任一预算参数 → RUN_BLOCKED、退出 3（不回落默认值）。"""
    base = ["bash", str(EO / "run_official_hard.sh"), "--run-name", "R", "--seat", "01", "--repo", str(REPO),
            "--stage", str(tmp_path / "stage"), "--shard", str(tmp_path / "s.json"), "--policy", "groundsg",
            "--groundsg-variant", F.MEMER, "--memer-adapter", str(tmp_path), "--openpi-data-home", "/o",
            "--tokenizer-sha256", "x", "--dataset", "hard-verify", "--max-steps", "1300",
            "--infra-retry-budget", "1", "--local-root", str(tmp_path / "local")]
    budget = ["--budget-ledger", str(tmp_path / "l.jsonl"), "--trajectory-cap", "870", "--shared-infra-cap", "50",
              "--expired-cap", "50", "--planned-first-tries", "821"]
    env = {k: v for k, v in os.environ.items() if k not in ("POLICY_SEED", "MEMER_ADAPTER")}
    cases = [(base + budget, "RUN_BLOCKED reason=policy_seed"),
             (base + ["--policy-seed", "x"] + budget, "RUN_BLOCKED reason=policy_seed"),
             (base + ["--policy-seed", "7"] + _drop(budget, "--expired-cap"),
              "RUN_BLOCKED reason=budget_args missing=EXPIRED_CAP"),
             (base + ["--policy-seed", "7"] + _set(budget, "--trajectory-cap", "abc"),
              "RUN_BLOCKED reason=budget_args missing=TRAJECTORY_CAP=abc")]
    for argv, want in cases:
        p = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)
        assert p.returncode == 3, (argv, p.stdout, p.stderr)
        assert want in p.stdout and p.stdout.rstrip().splitlines()[-1] == "EXIT_CODE=3", p.stdout
    assert not (tmp_path / "stage").exists()


def test_build_runner_cmd_forwards_memer_seed_and_budget():
    script = r'''
set -u
source "$EO/run_official_hard.sh"
POLICY=groundsg; GROUNDSG_VARIANT=ground-sg-memer; QWENVL_ADAPTER=""; MEMER_ADAPTER=/ck/memer
SGEVAL_CLIENT_PY=/py/client; SHARD=/s.json; RUN_OUT=/run; MAX_STEPS=1300; DATASET=hard-verify
POLICY_SEED=42; BUDGET_LEDGER=/l.jsonl; TRAJECTORY_CAP=870; SHARED_INFRA_CAP=50; EXPIRED_CAP=50; PLANNED_FIRST_TRIES=821
build_runner_cmd 2 k1 18555
for a in "${RUN_ARGV[@]}"; do printf 'RUN %s\n' "$a"; done
'''
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, EO=str(EO)))
    run = [x[4:] for x in p.stdout.splitlines() if x.startswith("RUN ")]
    assert run[:2] == ["/py/client", "scripts/eval-official/official_hard_runner.py"], p.stdout + p.stderr

    def opt(name):
        return run[run.index(name) + 1] if name in run else None

    assert (opt("--variant"), opt("--memer-adapter"), opt("--qwenvl-groundsg-adapter")) == \
        (F.MEMER, "/ck/memer", None)
    assert (opt("--policy-seed"), opt("--budget-ledger"), opt("--trajectory-cap"), opt("--shared-infra-cap"),
            opt("--expired-cap"), opt("--planned-first-tries")) == ("42", "/l.jsonl", "870", "50", "50", "821")
    # 驱动的解析器认这些参数（同名同义）
    args = F.official_hard_runner().build_parser().parse_args(run[2:])
    assert (args.policy_seed, args.memer_adapter, args.trajectory_cap, args.planned_first_tries) == ("42", "/ck/memer",
                                                                                                    870, 821)
