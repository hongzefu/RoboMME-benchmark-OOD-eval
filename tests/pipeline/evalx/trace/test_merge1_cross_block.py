"""C13-MERGE1-CROSS：第三阶段五块（R6／R2／R3／R7／R5）合并后的跨块接口对齐（MERGE-1 整合，接口冻结说明三、四、五节）。

各块在同一基点上按冻结说明并行写成，单块测试只能用替身对接对方接口；这里一律用**两边的真实代码**各跑一次：

- 数组合并写（清单 B）：R2 ``orig_observer/step_arrays.StepArrays.save`` 与 R5 ``astra_hard_runner.write_exec_actions`` 都经
  R6 ``trace_writer.merge_write_npz`` 写同一 ``arrays.npz``；与 ``TraceWriter.close`` 不论谁先写，``exec_action__*`` 与
  ``exec_state__*`` 都在、``end.arrays`` 无 error。
- 共享预算账本（清单 C）：原侧 R2 ``official_hard_runner.open_budget_ledger``＋``OrigBudget`` 与新侧 R3
  ``env_client.AttemptLedger``（``budget_caps`` 构造参数）打开同一个真实 ``budget_ledger.BudgetLedger`` 文件：同 config 不报
  ``BudgetConfigMismatch``、首试与恢复计数正确、同 token 续跑不重复扣、config 不一致即拒。
- 服务外壳回包 → 客户端语言账本（清单 D、K）：R3 ``pp_server_wrap`` 的真实 ``_S2Watch``＋``_audit_block`` 产出的
  ``_sgeval_audit`` 交给 R6 ``pp_client.TracedConnection``，``language.jsonl`` 里 S2 生成块（``subgoal_model`` 调用）非空、
  键对得上，S1 通道的 ``text_reconstructed`` 记进该调用 ``call_close.parsed``；R3 ``policy_server_wrap.build_audit`` 的
  通道字段经 R6 ``audit_channel_messages`` 原样落行。
- 目录锁超时（清单 H）：``merge_write_npz`` 等锁超过 ``MERGE_LOCK_TIMEOUT_S`` 抛 ``TimeoutError``，``TraceWriter.close``
  记进 ``end.arrays.error``、不挂住收尾。

纯 CPU，不起网络、不起仿真、不加载模型。
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tests._support.loaders import load_script
from tests.pipeline.evalx.astra.astra_fakes import astra_session
from tests.pipeline.evalx.groundsg import groundsg_fakes as G
from tests.pipeline.evalx.pp import pp_fakes as P


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _arrays(p: Path) -> dict:
    with np.load(p) as z:
        return {k: (z[k].dtype.str, z[k].shape, z[k].tobytes()) for k in z.files}


def _trace_three_steps_one_missing(tw, d: Path, actions: list) -> None:
    """3 个观测步 + 第 4 步无观测（动作照收，状态不补零）；收尾写 arrays.npz。"""
    w = tw.TraceWriter(d / "trace.jsonl", route="x/new", identity={"key": "K"}, max_steps=10, policy_seed=7)
    for i, a in enumerate(actions[:3]):
        w.log_step(step=i + 1, front=None, wrist=None, state=np.full(8, i, dtype=np.float32), action=a,
                   subgoal=None, terminated=False, truncated=False, status="ongoing")
    w.log_missing_step(step=4, action=actions[3], reason="obs_none")
    w.close(status="error", terminal_reason="error")


def _assert_union(d: Path, action_dtype: str) -> None:
    arr = _arrays(d / "arrays.npz")
    assert sorted(k for k in arr if k.startswith("exec_action__")) == [f"exec_action__{i:05d}" for i in range(4)]
    assert sorted(k for k in arr if k.startswith("exec_state__")) == [f"exec_state__{i:05d}" for i in range(3)]
    assert {arr[f"exec_action__{i:05d}"][0] for i in range(4)} == {action_dtype}
    end = _rows(d / "trace.jsonl")[-1]
    assert end["arrays"] == {"path": "arrays.npz", "action_keys": 4, "state_keys": 3, "missing_state_steps": [4]}


# ── 清单 B：R2／R5 的数组写法与 TraceWriter.close 任意先后 ─────────────────────────────


@pytest.mark.parametrize("arrays_first", [True, False])
def test_step_arrays_and_trace_writer_any_order_keep_states(tmp_path, monkeypatch, arrays_first):
    tw = load_script("eval-official/trace_writer.py")
    monkeypatch.setitem(sys.modules, "trace_writer", tw)  # step_arrays 按此别名取 trace_writer（同 _obs_common 约定）
    sa = load_script("eval-official/orig_observer/step_arrays.py", fresh=True)
    actions = [np.arange(8, dtype=np.float64) + i for i in range(4)]
    s = sa.StepArrays()
    for i, a in enumerate(actions):
        s.add(i, a)
    if arrays_first:
        s.save(tmp_path / "arrays.npz")
    _trace_three_steps_one_missing(tw, tmp_path, actions)
    if not arrays_first:
        s.save(tmp_path / "arrays.npz")
    _assert_union(tmp_path, "<f8")


@pytest.mark.parametrize("arrays_first", [True, False])
def test_astra_write_exec_actions_and_trace_writer_any_order_keep_states(tmp_path, arrays_first):
    with astra_session() as (mod, _astra):
        import trace_writer as tw  # 与 write_exec_actions 内部 import 的是同一个模块对象

        assert callable(getattr(tw, "merge_write_npz", None))
        actions = [np.full(8, i, dtype=np.float32) for i in range(4)]
        if arrays_first:
            mod.write_exec_actions(tmp_path / "arrays.npz", actions)
        _trace_three_steps_one_missing(tw, tmp_path, actions)
        if not arrays_first:
            mod.write_exec_actions(tmp_path / "arrays.npz", actions)
    _assert_union(tmp_path, "<f4")


# ── 清单 H：目录锁超时 ─────────────────────────────────────────────────────────


def test_merge_write_npz_lock_timeout_recorded_in_end_arrays(tmp_path, monkeypatch):
    tw = load_script("eval-official/trace_writer.py", fresh=True)
    monkeypatch.setattr(tw, "MERGE_LOCK_TIMEOUT_S", 0.3)
    d = tmp_path / "ep"
    d.mkdir()
    held = tw._lock_dir(d)  # 另一写者占着目录锁（同进程另开一个文件描述，flock 同样互斥）
    try:
        with pytest.raises(TimeoutError):
            tw.merge_write_npz(d / "arrays.npz", {"exec_action__00000": np.zeros(8)})
        _trace_three_steps_one_missing(tw, d, [np.zeros(8)] * 4)
    finally:
        tw._unlock_dir(held)
    end = _rows(d / "trace.jsonl")[-1]
    assert end["arrays"]["error"].startswith("TimeoutError") and not (d / "arrays.npz").exists()
    # 锁释放后照常写
    tw.merge_write_npz(d / "arrays.npz", {"exec_action__00000": np.zeros(8)})
    assert list(_arrays(d / "arrays.npz")) == ["exec_action__00000"]


# ── 清单 C：原侧＋新侧共用同一个真实共享账本 ─────────────────────────────────────


CAPS = {"trajectory_cap": 6, "shared_infra_cap": 2, "expired_cap": 2, "planned_first_tries": 4}


def test_orig_and_new_side_share_one_real_budget_ledger(tmp_path):
    ohr, ec = G.official_hard_runner(), G.env_client()
    bl = load_script("eval-official/budget_ledger.py")
    book = tmp_path / "budget" / "budget-ledger.jsonl"
    book.parent.mkdir()
    # 原侧：official_hard_runner 的 CLI 参数名 → open_budget_ledger（R2）
    orig_args = argparse.Namespace(budget_ledger=str(book), **CAPS)
    orig = ohr.OrigBudget(ohr.open_budget_ledger(orig_args), variant=G.ORACLE, policy_seed=G.POLICY_SEED)
    assert orig.route == f"groundsg/{G.ORACLE}/seed7/orig"
    # 新侧：env_client 的 CLI 参数名 → budget_caps → AttemptLedger 打开同一账本（R3）
    new_args = argparse.Namespace(budget_ledger=str(book), **CAPS)
    new_route = f"groundsg/{G.ORACLE}/seed7/new"
    new = ec.AttemptLedger(tmp_path / "attempts.jsonl", seat="00", policy="groundsg", shared=str(book),
                           route=new_route, caps=ec.budget_caps(new_args))
    assert new.shared is not None and type(new.shared).__name__ == "BudgetLedger"

    def new_reserve(key: str, attempt: int) -> str:  # 与 SeatRunner._reserve 同参数（token 同式）
        return new.shared.reserve(resets=ec.NEW_SIDE_RESETS_PER_ATTEMPT, route=new_route, key=key,
                                  token=bl.make_token(new_route, key, attempt),
                                  kind_of_try="first" if attempt == 1 else "recovery")

    # 首试：原侧两个、新侧两个（共 4 = planned_first_tries）
    h1 = orig.begin("K1", 1)
    h2 = orig.begin("K2", 1)
    r1 = new_reserve("K1", 1)
    new_reserve("K2", 1)
    # 同 token 续跑：同一 rid、不写新行
    n_rows = len(_rows(book))
    assert orig.begin("K1", 1)["rid"] == h1["rid"] and new_reserve("K1", 1) == r1
    assert len(_rows(book)) == n_rows
    # 原侧一局真实 build 后收尾、一局没 build 就退回
    orig.claimer(h1)("build")
    orig.settle(h1, {"status": "success", "infra": False})
    orig.settle(h2, {"status": "error", "infra": True})  # resets_claimed=0 → release
    st = bl.BudgetLedger(book, **CAPS).state()
    assert (st.trajectories, st.first_started, st.recovery_used) == (3, 3, 0)
    # 恢复：原侧 K2 第 2 次（claim_retry + recovery reserve），新侧也来一次；recovery 合计受 6-4=2 约束
    h2b = orig.begin("K2", 2)
    assert new.allow_retry("K1", token=bl.make_token(new_route, "K1", 2)) is True
    new_reserve("K1", 2)
    st = bl.BudgetLedger(book, **CAPS).state()
    assert (st.trajectories, st.first_started, st.recovery_used, len(st.retries)) == (5, 3, 2, 2)
    # 同 token 再领重试与再预约：幂等
    assert orig.begin("K2", 2)["rid"] == h2b["rid"]
    assert len(bl.BudgetLedger(book, **CAPS).state().retries) == 2
    # 恢复额度用满：第三次恢复被拒（未开始的首试 1 个仍保留）
    with pytest.raises(Exception) as ei:
        new_reserve("K3", 2)
    assert getattr(ei.value, "budget_exhausted", False) and ei.value.reason in ("reserved_for_first_tries",
                                                                                   "recovery_cap")
    # 账本恰好一行 config，两侧参数一致；换一组参数打开即拒
    rows = _rows(book)
    assert [r["kind"] for r in rows].count("config") == 1 and rows[0]["kind"] == "config"
    assert {k: rows[0][k] for k in CAPS} == CAPS
    with pytest.raises(Exception) as ei:  # 原侧经自己的 load_sibling 载入 budget_ledger，按类名识别
        ohr.open_budget_ledger(argparse.Namespace(budget_ledger=str(book), **dict(CAPS, planned_first_tries=5))) \
            .check()
    assert type(ei.value).__name__ == "BudgetConfigMismatch"
    print(f"BUDGET_SHARED_E2E=PASS trajectories=5 first=3 recovery=2 retries=2 config_rows=1")


# ── 清单 D、K：服务外壳审计键 → 客户端语言账本 ──────────────────────────────────


def _load_pp_server_wrap(monkeypatch):
    """pp_server_wrap 顶层 import 父类服务；这里给空父类与 build_pi0_prompt 桩（外壳自己的代码全部真实执行）。"""
    rs = types.ModuleType("ponderpounce.eval.robomme_server")
    rs.PonderPounceRoboMMEServer = type("PonderPounceRoboMMEServer", (), {})
    prompt = types.ModuleType("ponderpounce.pi05.prompt")
    prompt.build_pi0_prompt = lambda task, state: f"Task: {task};\nAction: "
    for name, m in (("ponderpounce", types.ModuleType("ponderpounce")),
                    ("ponderpounce.eval", types.ModuleType("ponderpounce.eval")),
                    ("ponderpounce.eval.robomme_server", rs),
                    ("ponderpounce.pi05", types.ModuleType("ponderpounce.pi05")),
                    ("ponderpounce.pi05.prompt", prompt)):
        monkeypatch.setitem(sys.modules, name, m)
    return load_script("eval-official/pp_server_wrap.py", fresh=True)


class _FakeS2Ctx:
    """S2 上下文桩：按脚本给 fire 结果（transition 带子目标与 reasoning，nontransition 空）。"""

    def __init__(self):
        self._context_len, self.n = 10, 0

    def fire(self):
        self.n += 1
        self._context_len += 3
        if self.n == 1:
            return types.SimpleNamespace(subgoal_text="pick up the cube at [500, 250]", reasoning_text="think",
                                         subgoal_tokens=[1, 2, 3], kind="transition", gate_score=0.9,
                                         n_input_frames=self.n)
        return types.SimpleNamespace(subgoal_text="", reasoning_text="", subgoal_tokens=None, kind="nontransition",
                                     gate_score=0.1, n_input_frames=self.n)


class _WrapAuditConn(P.FakeConn):
    """每个 ACTION 回包附上 pp_server_wrap 真实 ``_audit_block`` 产出的审计键：第 0、2 次回包前各 fire 一次 S2。"""

    def __init__(self, wrap):
        super().__init__()
        self.srv = wrap.SubgoalReportingServer()
        self.ctx = _FakeS2Ctx()
        self.ep = types.SimpleNamespace(
            session=types.SimpleNamespace(s1_prompt_ids=np.array([5, 6, 7]), s1_prompt_mask=np.array([1, 1, 0]),
                                          task_text="pick up the cube"),
            active_subgoal=None, n_s2_fires=0, n_s1_fires=0)
        self.ep._sgeval_generations = []
        wrap._S2Watch(self.ctx, self.ep._sgeval_generations)
        self.audits = []

    async def act(self, obs):
        a = dict(await super().act(obs))
        n = self.n_actions - 1
        if n in (0, 2):
            self.ctx.fire()
            self.ep.n_s2_fires += 1
        a["subgoal"] = "pick up the cube at [500, 250]"
        audit = self.srv._audit_block(self.ep)
        self.audits.append(audit)
        a["_sgeval_audit"] = audit
        return a


def test_pp_server_wrap_generation_and_text_flag_reach_language_log(tmp_path, monkeypatch):
    wrap = _load_pp_server_wrap(monkeypatch)
    pp = load_script("eval-official/pp_client.py")
    conn = _WrapAuditConn(wrap)
    sess = P.FakeSession(P.FakeEnv("PickXtimes", 7, demo=2, done_at=4))
    ident = {"task": "PickXtimes", "tier": "xhard0", "seed": 9, "source_episode": 7, "builder_episode": 1,
             "key": "PickXtimes_xhard0_9"}
    ep = tmp_path / f"{ident['key']}.a1"
    ci = {"host": "127.0.0.1", "port": 1, "max_steps": 1300, "dataset": "hard-verify", "pp_phase2": True,
          "trace_dir": str(ep), "episode_tag": ep.name, "policy_seed": 7}
    res = pp.run_episode(sess, ident, ci, None, connection_factory=lambda url, timeout: conn)
    assert res["status"] == "success" and res["steps"] == 4
    # 外壳一侧：没有 fire 的回包 pp_generation 为 None，有 fire 的是一块的列表
    assert [a["pp_generation"] is None for a in conn.audits] == [False, True, False, True]
    assert all(a["channels"][0]["text_reconstructed"] is True for a in conn.audits)
    rows = _rows(ep / "language.jsonl")
    opens = [r for r in rows if r["kind"] == "call_open"]
    closes = {r["call_id"]: r for r in rows if r["kind"] == "call_close"}
    msgs = [r for r in rows if r["kind"] == "message"]
    sg = [r for r in opens if r["model"] == "subgoal_model"]
    act = [r for r in opens if r["model"] == "action_model"]
    assert [r["step"] for r in act] == [0, 1, 2, 3] and [r["step"] for r in sg] == [0, 2]
    # S2 transition 块非空：out 文字 = 子目标原文，parsed 有换算后子目标、reasoning、kind、committed 与 params
    t_close, n_close = closes[sg[0]["call_id"]], closes[sg[1]["call_id"]]
    t_out = [m for m in msgs if m["call_id"] == sg[0]["call_id"] and m["dir"] == "out"]
    assert [m["text"] for m in t_out] == ["pick up the cube at [500, 250]"]
    tp = t_close["parsed"]
    assert (tp["subgoal"], tp["reasoning"], tp["kind"], tp["committed"], tp["text_output"]) == \
        ("pick up the cube at <64, 128>", "think", "transition", True, True)
    assert sg[0]["params"]["fire_index"] == 0 and sg[0]["params"]["gate_score"] == 0.9
    assert sg[0]["params"]["context_delta"] == 3
    # nontransition：显式无文字输出
    assert (n_close["parsed"]["kind"], n_close["parsed"]["text_output"], n_close["parsed"]["committed"]) == \
        ("nontransition", False, False)
    # S1 通道：token_ids／mask 原样、text_reconstructed 落在该 action_model 调用的 call_close.parsed
    for o in act:
        ch = [m for m in msgs if m["call_id"] == o["call_id"] and m.get("channel") == "task"]
        assert len(ch) == 1 and ch[0]["token_ids"] == [5, 6, 7] and ch[0]["mask"] == [1, 1, 0]
        assert ch[0]["text"] == "Task: pick up the cube;\nAction: "
        assert closes[o["call_id"]]["parsed"] == {"text_reconstructed_channels": ["task"]}
        assert closes[o["call_id"]]["server_final_text"] == "Task: pick up the cube;\nAction: "
    # 审计键在交给环境前已 pop（环境收到的动作是 8 维 float）
    assert all(len(a) == 8 for a in sess.env.actions)


def test_policy_server_wrap_channels_land_in_language_log(tmp_path):
    psw = load_script("eval-official/policy_server_wrap.py")
    tw = load_script("eval-official/trace_writer.py")
    channels = [{"channel": "task", "text": "pick up the cube", "token_ids": [1, 2], "mask": [True, True],
                 "tokenizer": "paligemma_tokenizer.model max_len=48", "truncated": False},
                {"channel": "symbolic", "text": "Current Subgoal: pick red", "token_ids": [3], "mask": [True],
                 "tokenizer": "paligemma_tokenizer.model max_len=48", "truncated": True}]
    audit = psw.build_audit(channels)
    lang = tw.LanguageLog(tmp_path / "language.jsonl")
    cid = lang.open_call("action_model", 0)
    final, trunc = tw.audit_channel_messages(lang, cid, audit)
    lang.close_call(cid, status="reply", server_final_text=final, server_truncated=trunc)
    lang.close()
    msgs = [r for r in _rows(tmp_path / "language.jsonl") if r["kind"] == "message"]
    assert [(m["channel"], m["text"], m["token_ids"], m["mask"], m["tokenizer"], m["truncated"]) for m in msgs] == \
        [(c["channel"], c["text"], c["token_ids"], c["mask"], c["tokenizer"], c["truncated"]) for c in channels]
    assert (final, trunc) == ("Current Subgoal: pick red", True)
