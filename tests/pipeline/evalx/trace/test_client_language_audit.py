"""C13-SG-CLIENT-LANGUAGE：三个新侧客户端（FrameSamp+Modulation、SimpleMemVLA、PonderPounce）的语言账本接线与
审计键剥离（冻结说明第五节，计划八.11「一局里各路线会有什么记录」；R6）。

假服务分「带 ``_sgeval_audit``」与「不带」两种跑同一局：交给环境的动作字节、发给服务的请求必须完全相同（OBS_EQ
雏形）；``language.jsonl`` 的调用、消息顺序与字段按计划八.11 手写期望核对。不起网络、不起仿真。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests._support.loaders import load_script
from tests.pipeline.eval import eval_fakes as F
from tests.pipeline.evalx.pp import pp_fakes as P

AUDIT_KEY = "_sgeval_audit"
IDENT = {"task": "T", "tier": "xhard0", "seed": 1, "source_episode": 2, "builder_episode": 2, "key": "T_xhard0_1"}
CHANNELS = [
    {"channel": "task", "text": "goal-T-2", "token_ids": [5, 6, 7], "mask": [1, 1, 1], "tokenizer": "paligemma",
     "truncated": False},
    {"channel": "symbolic", "text": "Current Subgoal: none", "token_ids": [9], "mask": [1], "tokenizer": "paligemma",
     "truncated": True},
]
FINAL_TEXT = "Task: goal-T-2;\nAction: "


def _tw():
    return load_script("eval-official/trace_writer.py")


def _lines(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _audit(**extra) -> dict:
    return {"channels": [dict(c) for c in CHANNELS], "server_final_text": FINAL_TEXT, "pp_generation": None, **extra}


def _calls(rows: list[dict]) -> dict[str, dict]:
    """按 call_id 归并：{call_id: {"open": 行, "msgs": [行…], "close": 行}}，并核行序（open 在前、close 在后）。"""
    out: dict[str, dict] = {}
    for r in rows:
        if r["kind"] == "call_open":
            assert r["call_id"] not in out
            out[r["call_id"]] = {"open": r, "msgs": [], "close": None}
        elif r["kind"] == "message":
            c = out[r["call_id"]]
            assert c["close"] is None and r["message_index"] == len(c["msgs"])
            c["msgs"].append(r)
        elif r["kind"] == "call_close":
            assert out[r["call_id"]]["close"] is None
            out[r["call_id"]]["close"] = r
    assert all(c["close"] is not None for c in out.values())
    return out


def _frame_shas(trace_rows: list[dict]) -> set[str]:
    demo = next(r for r in trace_rows if r["kind"] == "demo")
    shas = set(demo["front_sha256"]) | set(demo["wrist_sha256"])
    for r in trace_rows:
        if r["kind"] == "step":
            shas |= {r["front_sha256"], r["wrist_sha256"]}
    return shas - {None}


def _image_ref_resolves(img: dict, trace_rows: list[dict]) -> bool:
    """附图引用按 phase／frame_idx／cam 落到 trace 的那一帧，且 raw_sha256 相等。"""
    key = "front_sha256" if img["cam"] == "front" else "wrist_sha256"
    if img["phase"] == "demo":
        demo = next(r for r in trace_rows if r["kind"] == "demo")
        return demo[key][img["frame_idx"]] == img["raw_sha256"]
    st = next(r for r in trace_rows if r["kind"] == "step" and r["step"] == img["frame_idx"])
    return st[key] == img["raw_sha256"]


def _norm(x):
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


# ---------------------------------------------------------------- 环境会话（真实 EnvSession + 假 builder）


class _B:
    def __init__(self, plan):
        self.plan = plan
        self.env = None

    def make_env_for_episode(self, ep, max_steps=None):
        self.env = F.FakeEnv("T", ep, self.plan)
        return self.env


def _session(plan):
    b = _B(plan)
    from robomme_ood_eval.session import EnvSession

    return EnvSession("T", 2, recorder=None, builder=b), b


def _conn_info(tmp_path, name, **kw):
    tag = f"{IDENT['key']}.a1"
    ep = tmp_path / name / tag
    return dict({"trace_dir": str(ep), "episode_tag": tag, "dataset": "hard-verify"}, **kw), ep


# ---------------------------------------------------------------- SimpleMemVLA


class _AuditSmvlaConn(F.FakeSmvlaConn):
    """回包在 infer 时多带 ``_sgeval_audit``（服务外壳行为）；其余字节与父类相同。"""

    def __init__(self, server, with_audit: bool):
        super().__init__(server)
        self.with_audit = with_audit
        self.raws: list[bytes] = []

    def call(self, msg):
        rep, raw, rep_raw = super().call(msg)
        self.raws.append(raw)
        if self.with_audit and "actions_full" in rep:
            rep = dict(rep, **{AUDIT_KEY: _audit(server_final_text="The overall task is: goal-T-2. Identify the "
                                                                    "current sub-task.")})
        return rep, raw, rep_raw


def _smvla_run(tmp_path, name, with_audit):
    sm = F.smvla_client()
    server = F.FakePolicyServer()
    sess, b = _session(F.Plan(success_at=20))
    ci, ep = _conn_info(tmp_path, name)
    conn = _AuditSmvlaConn(server, with_audit)
    res = sm.run_episode(sess, dict(IDENT), ci, None, conn=conn)
    return res, b, server, conn, ep


def test_smvla_audit_does_not_change_actions_and_language_rows(tmp_path):
    ra, ba, sa, ca, epa = _smvla_run(tmp_path, "audit", True)
    rn, bn, sn, cn, epn = _smvla_run(tmp_path, "plain", False)
    assert ra["status"] == rn["status"] == "success" and ra["decisions"] == rn["decisions"] == 2
    # OBS_EQ 雏形：交给环境的动作字节、发给服务的请求字节与服务收到的内容完全相同
    assert [a.tobytes() for a in ba.env.actions] == [a.tobytes() for a in bn.env.actions]
    assert ca.raws == cn.raws and _norm(sa.log) == _norm(sn.log)
    for ep, with_audit in ((epa, True), (epn, False)):
        rows = _lines(ep / "language.jsonl")
        calls = _calls(rows)
        assert len(calls) == 2 and all(c["open"]["model"] == "action_model" for c in calls.values())
        assert [c["open"]["step"] for c in calls.values()] == [0, 16]
        for c in calls.values():
            m = c["msgs"]
            assert (m[0]["dir"], m[0]["role"], m[0]["text"]) == ("in", "fields", {"instruction": "goal-T-2"})
            if with_audit:
                assert [(x["dir"], x["role"], x["channel"], x["text"], x["token_ids"], x["truncated"]) for x in m[1:3]] \
                    == [("in", "fields", ch["channel"], ch["text"], ch["token_ids"], ch["truncated"]) for ch in CHANNELS]
                assert c["close"]["server_final_text"].startswith("The overall task is: goal-T-2")
                assert c["close"]["server_truncated"] is True
            else:
                assert len(m) == 2 and c["close"]["server_final_text"] is None
            assert (m[-1]["dir"], m[-1]["role"], m[-1]["text"]) == ("out", "assistant", "s")
            assert (c["close"]["status"], c["close"]["parsed"]) == ("reply", "s")
        trace = _lines(ep / "trace.jsonl")
        steps = [r for r in trace if r["kind"] == "step"]
        ids = list(calls)
        assert [(s["source_call_id"], s["chunk_index"]) for s in steps] == \
            [(ids[(s["step"] - 1) // 16], (s["step"] - 1) % 16) for s in steps]
        assert AUDIT_KEY not in (ep / "trace.jsonl").read_text(encoding="utf-8")


# ---------------------------------------------------------------- FrameSamp+Modulation


class _AuditFramesampClient(F.FakeMMEVLAWebsocketClient):
    def __init__(self, server, with_audit: bool):
        super().__init__(server)
        self.with_audit = with_audit

    def infer(self, element):
        out = super().infer(element)
        if self.with_audit:
            out = dict(out, **{AUDIT_KEY: _audit()})
        return out


def _framesamp_run(tmp_path, monkeypatch, name, with_audit, traced=True):
    mc = F.framesamp_modul_client()
    server = F.FakePolicyServer()
    monkeypatch.setattr(mc, "make_recording_client",
                        lambda host, port, recorder, timing: _AuditFramesampClient(server, with_audit))
    sess, b = _session(F.Plan(success_at=20))
    ci, ep = _conn_info(tmp_path, name, port=1, max_steps=1300)
    if not traced:
        ci = {"port": 1, "max_steps": 1300}
    res = mc.run_episode(sess, dict(IDENT), ci, None)
    return res, b, server, ep


def test_framesamp_audit_does_not_change_actions_and_language_rows(tmp_path, monkeypatch):
    ra, ba, sa, epa = _framesamp_run(tmp_path, monkeypatch, "audit", True)
    rn, bn, sn, epn = _framesamp_run(tmp_path, monkeypatch, "plain", False)
    ru, bu, su, _ = _framesamp_run(tmp_path, monkeypatch, "untraced", True, traced=False)
    assert ra["status"] == rn["status"] == ru["status"] == "success" and ra["decisions"] == 2
    acts = [a.tobytes() for a in ba.env.actions]
    assert acts == [a.tobytes() for a in bn.env.actions] == [a.tobytes() for a in bu.env.actions]
    assert _norm(sa.log) == _norm(sn.log) == _norm(su.log)
    for ep, with_audit in ((epa, True), (epn, False)):
        trace = _lines(ep / "trace.jsonl")
        calls = _calls(_lines(ep / "language.jsonl"))
        assert len(calls) == 2 and [c["open"]["step"] for c in calls.values()] == [0, 16]
        for c in calls.values():
            assert c["open"]["model"] == "action_model"
            m0 = c["msgs"][0]
            assert (m0["dir"], m0["role"], m0["text"]) == ("in", "fields", {"prompt": "goal-T-2"})
            assert [(i["slot"], i["ref"], i["cam"]) for i in m0["images"]] == [(0, "current", "front"),
                                                                              (1, "wrist", "wrist")]
            assert all(_image_ref_resolves(i, trace) for i in m0["images"])
            assert all(i["raw_sha256"] in _frame_shas(trace) for i in m0["images"])
            assert all(m["dir"] == "in" for m in c["msgs"])  # FrameSamp+Modulation 无文字输出
            if with_audit:
                assert [(m["channel"], m["text"], m["mask"]) for m in c["msgs"][1:]] == \
                    [(ch["channel"], ch["text"], ch["mask"]) for ch in CHANNELS]
                assert (c["close"]["server_final_text"], c["close"]["server_truncated"]) == (FINAL_TEXT, True)
            else:
                assert len(c["msgs"]) == 1 and c["close"]["server_final_text"] is None
            assert c["close"]["status"] == "reply"
        first = next(iter(calls.values()))["msgs"][0]["images"][0]
        assert (first["phase"], first["frame_idx"]) == ("demo", F.N_RESET_FRAMES - 1)
        steps = [r for r in trace if r["kind"] == "step"]
        ids = list(calls)
        assert [(s["source_call_id"], s["chunk_index"]) for s in steps] == \
            [(ids[(s["step"] - 1) // 16], (s["step"] - 1) % 16) for s in steps]


# ---------------------------------------------------------------- PonderPounce


GEN_T = {"context": "<|im_start|>user\npick up the cube", "text": "think… pick up the cube at [500, 250]",
         "reasoning": "think…", "subgoal_raw": "pick up the cube at [500, 250]", "kind": "transition", "committed": True}
GEN_N = {"context": "<|im_start|>user\npick up the cube", "text": None, "reasoning": None, "subgoal_raw": None,
         "kind": "nontransition", "committed": False}


class _AuditPPConn(P.FakeConn):
    """第二阶段外壳回包：带 ``subgoal``；可选带 ``_sgeval_audit``（第 0、2 个回包带 S2 transition／nontransition 生成块）。"""

    def __init__(self, with_audit: bool):
        super().__init__()
        self.with_audit = with_audit

    async def act(self, obs):
        a = dict(await super().act(obs))  # 复制：父类记录的协议帧保持原样
        n = self.n_actions - 1
        a["subgoal"] = "pick up the cube at [500, 250]"
        if self.with_audit:
            gen = {0: GEN_T, 2: GEN_N}.get(n)
            a[AUDIT_KEY] = _audit(pp_generation=None if gen is None else dict(gen))
        return a


def _pp_run(tmp_path, name, with_audit):
    pp = load_script("eval-official/pp_client.py")
    conn = _AuditPPConn(with_audit)
    sess = P.FakeSession(P.FakeEnv("PickXtimes", 7, demo=2, done_at=4))
    ident = {"task": "PickXtimes", "tier": "xhard0", "seed": 9, "source_episode": 7, "builder_episode": 1,
             "key": "PickXtimes_xhard0_9"}
    tag = f"{ident['key']}.a1"
    ep = tmp_path / name / tag
    ci = {"host": "127.0.0.1", "port": 1, "max_steps": 1300, "dataset": "hard-verify", "pp_phase2": True,
          "trace_dir": str(ep), "episode_tag": tag}
    res = pp.run_episode(sess, ident, ci, None, connection_factory=lambda url, timeout: conn)
    return res, sess, conn, ep


def test_pp_audit_does_not_change_actions_and_language_rows(tmp_path):
    ra, sa, ca, epa = _pp_run(tmp_path, "audit", True)
    rn, sn, cn, epn = _pp_run(tmp_path, "plain", False)
    assert ra["status"] == rn["status"] == "success" and ra["steps"] == 4
    assert sa.env.actions == sn.env.actions and all(isinstance(x, float) for x in sa.env.actions[0])
    assert P.compare_frames(ca.log, cn.log)[:2] == (0, 0)  # 发给服务的协议帧逐帧相同
    for ep, with_audit in ((epa, True), (epn, False)):
        trace = _lines(ep / "trace.jsonl")
        calls = _calls(_lines(ep / "language.jsonl"))
        act_calls = [c for c in calls.values() if c["open"]["model"] == "action_model"]
        sg_calls = [c for c in calls.values() if c["open"]["model"] == "subgoal_model"]
        assert [c["open"]["step"] for c in act_calls] == [0, 1, 2, 3]
        for c in act_calls:
            m0 = c["msgs"][0]
            assert (m0["dir"], m0["role"], m0["text"]) == ("in", "fields", {"task_description": P.TASK_GOAL})
            assert all(_image_ref_resolves(i, trace) for i in m0["images"]) and len(m0["images"]) == 2
            assert c["close"]["server_final_text"] == (FINAL_TEXT if with_audit else None)
        steps = [r for r in trace if r["kind"] == "step"]
        assert [(s["source_call_id"], s["chunk_index"]) for s in steps] == \
            [(c["open"]["call_id"], 0) for c in act_calls]
        if not with_audit:
            assert sg_calls == []
            continue
        assert [c["open"]["step"] for c in sg_calls] == [0, 2]
        t_call, n_call = sg_calls
        assert [(m["dir"], m["role"], m["text"]) for m in t_call["msgs"]] == [
            ("in", "user", GEN_T["context"]), ("out", "assistant", GEN_T["text"])]
        p = t_call["close"]["parsed"]
        assert (p["subgoal"], p["subgoal_raw"], p["kind"], p["committed"], p["text_output"]) == (
            "pick up the cube at <64, 128>", GEN_T["subgoal_raw"], "transition", True, True)
        assert p["reasoning"] == "think…" and p["generation"] == GEN_T
        assert [(m["dir"], m["text"]) for m in n_call["msgs"]] == [("in", GEN_N["context"]), ("out", None)]
        pn = n_call["close"]["parsed"]
        assert (pn["kind"], pn["text_output"], pn["committed"], pn["subgoal"]) == ("nontransition", False, False, None)
        end = trace[-1]
        assert end["language_hook_errors"] == 0 and end["arrays"]["action_keys"] == 4


def test_pp_act_failure_closes_call_with_error_and_retries_get_transport_attempt(tmp_path):
    """同一局断线重发：第一次调用记 error，重发的调用 transport_attempt=1；之后的调用回到 0。"""
    pp = load_script("eval-official/pp_client.py")

    class _Closed(_AuditPPConn):
        def __init__(self):
            super().__init__(True)
            self.closed_at = {1}

    conn = _Closed()
    sess = P.FakeSession(P.FakeEnv("PickXtimes", 7, demo=2, done_at=3))
    ident = {"task": "PickXtimes", "tier": "xhard0", "seed": 9, "source_episode": 7, "builder_episode": 1,
             "key": "PickXtimes_xhard0_9"}
    ep = tmp_path / "PickXtimes_xhard0_9.a1"
    ci = {"port": 1, "max_steps": 1300, "dataset": "hard-verify", "pp_phase2": True, "trace_dir": str(ep),
          "episode_tag": ep.name, "pp_max_reconnects": 1}
    res = pp.run_episode(sess, ident, ci, None, connection_factory=lambda url, timeout: conn)
    assert res["status"] == "success" and res["reconnects"] == 1
    act_calls = [c for c in _calls(_lines(ep / "language.jsonl")).values() if c["open"]["model"] == "action_model"]
    assert [(c["open"]["step"], c["open"]["transport_attempt"], c["close"]["status"]) for c in act_calls] == [
        (0, 0, "reply"), (1, 0, "error"), (1, 1, "reply"), (2, 0, "reply")]


# ── FIX-3：S2 输入（外壳从上下文 token 解码的 input_text）进 subgoal_model 调用的 in 消息 ─────────────────


class _S2InputPPConn(_AuditPPConn):
    """外壳回包的 S2 生成块带 ``input_text``／``input_images``（形状照 ``pp_server_wrap._S2Watch._decode_input``）：
    第 0 次回包是首次 fire（任务前缀 + 2 张演示图 + 本次两路观测图），第 2 次回包是之后的 fire（回灌的上一子目标 +
    cognition 占位 + 本次两路观测图）。像素哈希按外壳同一算法对客户端发出的原始帧计算。"""

    def __init__(self, with_inputs: bool):
        super().__init__(True)
        self.with_inputs = with_inputs

    async def act(self, obs):
        a = await super().act(obs)
        n = self.n_actions - 1
        gen = (a[AUDIT_KEY] or {}).get("pp_generation")
        if self.with_inputs and gen is not None:
            h = _tw().image_sha256
            obs_imgs = [{"source": "obs", "cam_key": c, "cam_slot": s, "pixel_sha256": h(obs["images"][c])}
                        for s, c in enumerate(("agentview", "wrist"))]
            if n == 0:
                demo = [{"source": "demo", "demo_pos": i, "pixel_sha256": h(f)}
                        for i, f in enumerate(obs["video_history"])]
                imgs, text = demo + obs_imgs, "<|im_start|>user\npick up the cube" + "".join(
                    f"<image:{k}>" for k in range(len(demo) + 2))
            else:
                imgs, text = obs_imgs, "<|fim_prefix|>pick up the cube at [500, 250]<|fim_pad|><image:0><image:1>"
            imgs = [dict(d, index=k, n_tokens=4) for k, d in enumerate(imgs)]
            gen = dict(gen, input_text=text, input_images=imgs, input_token_count=len(text),
                       input_decoded_from_tokens=True)
            a[AUDIT_KEY] = dict(a[AUDIT_KEY], pp_generation=gen)
        return a


def _pp_s2_run(tmp_path, name, with_inputs):
    pp = load_script("eval-official/pp_client.py")
    conn = _S2InputPPConn(with_inputs)
    sess = P.FakeSession(P.FakeEnv("PickXtimes", 7, demo=2, done_at=4))
    ident = {"task": "PickXtimes", "tier": "xhard0", "seed": 9, "source_episode": 7, "builder_episode": 1,
             "key": "PickXtimes_xhard0_9"}
    tag = f"{ident['key']}.a1"
    ep = tmp_path / name / tag
    ci = {"host": "127.0.0.1", "port": 1, "max_steps": 1300, "dataset": "hard-verify", "pp_phase2": True,
          "trace_dir": str(ep), "episode_tag": tag}
    res = pp.run_episode(sess, ident, ci, None, connection_factory=lambda url, timeout: conn)
    return res, sess, conn, ep


@pytest.mark.slow  # 核心短测时长控制（主会话 2026-10-07）：在 -m "" 下跑
def test_pp_s2_input_text_lands_as_in_message_with_resolvable_frames(tmp_path):
    """块带 ``input_text`` 时 subgoal_model 调用先写 ``dir=in role=user``（文字原样、附图按帧号落到 trace 那一帧：
    演示图 → demo 行 front 第 i 帧、``ref=keyframe``；观测图 → 本步帧），``parsed.input_decoded_from_tokens=True``，
    ``generation`` 不重复存 ``input_text``；动作与协议帧与不带输入的回包逐字节相同；LANG_IO 检查器零异常。"""
    ra, sa, ca, epa = _pp_s2_run(tmp_path, "inputs", True)
    rn, sn, cn, epn = _pp_s2_run(tmp_path, "plain", False)
    assert ra["status"] == rn["status"] == "success"
    assert sa.env.actions == sn.env.actions
    assert P.compare_frames(ca.log, cn.log)[:2] == (0, 0)
    trace = _lines(epa / "trace.jsonl")
    sg = [c for c in _calls(_lines(epa / "language.jsonl")).values() if c["open"]["model"] == "subgoal_model"]
    assert [c["open"]["step"] for c in sg] == [0, 2]
    first, later = sg
    m_in, m_out = first["msgs"]
    assert (m_in["dir"], m_in["role"]) == ("in", "user") and m_out["dir"] == "out"
    assert m_in["text"] == "<|im_start|>user\npick up the cube<image:0><image:1><image:2><image:3>"
    assert [(i["slot"], i["ref"], i["phase"], i["frame_idx"], i["cam"]) for i in m_in["images"]] == [
        (0, "keyframe", "demo", 0, "front"), (1, "keyframe", "demo", 1, "front"),
        (2, "current", "demo", 2, "front"), (3, "wrist", "demo", 2, "wrist")]
    assert m_in["demo_video"] == "demo[0:2]"
    assert all(_image_ref_resolves(i, trace) for i in m_in["images"])
    p = first["close"]["parsed"]
    assert p["input_decoded_from_tokens"] is True and "input_text" not in p["generation"]
    assert p["input_image_check"] == {"n": 4, "sha_mismatch": 0, "demo_unmatched": 0}
    li = later["msgs"][0]
    assert li["text"].startswith("<|fim_prefix|>pick up the cube at [500, 250]")
    assert [(i["slot"], i["phase"], i["frame_idx"], i["cam"]) for i in li["images"]] == [
        (0, "exec", 2, "front"), (1, "exec", 2, "wrist")]
    assert all(_image_ref_resolves(i, trace) for i in li["images"]) and li["demo_video"] is None
    # 没有 input_text 的块：与之前相同（in 消息仍按 context／prompt，parsed 不带新键）
    sg_plain = [c for c in _calls(_lines(epn / "language.jsonl")).values() if c["open"]["model"] == "subgoal_model"]
    assert all("input_decoded_from_tokens" not in c["close"]["parsed"] for c in sg_plain)
    assert [m["text"] for m in sg_plain[0]["msgs"] if m["dir"] == "in"] == [GEN_T["context"]]
    chk = load_script("eval-official/lang_io_check.py").check_episode(epa)
    assert all(v == 0 for v in chk["counts"].values()), chk
    print(f"PP_S2_INPUT_CLIENT=PASS subgoal_calls={len(sg)} image_ref_unresolved=0")
