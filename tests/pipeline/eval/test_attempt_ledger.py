"""C13 持久尝试账本 ``seat.AttemptLedger``（拆仓后在 ``dev-scripts/gl/seat.py``）：一切计数从账本内容推出，进程重启（重新打开）不刷新。

期望全部由本文件手写的事件序列推出：reset 额度、作废尝试、infra 重试计数、最终结局判定、唯一 accept 与 late、
预算提升、悬空尝试。
"""
from __future__ import annotations

import json

import pytest

import eval_fakes_dev as F


def _ledger(tmp_path, **kw):
    ec = F.env_client()
    return ec.AttemptLedger(tmp_path / "x.ledger.jsonl", seat="s00", policy="perceptual-framesamp-modul", **kw)


def _row(key, aid, no, **kw):
    return {"key": key, "attempt_id": aid, "attempt_no": no, **kw}


def test_reset_budget_counts_claims_and_survives_reopen(tmp_path):
    ec = F.env_client()
    led = _ledger(tmp_path)
    led.start(3, 1)
    led.attempt_start(key="k", attempt_id="a1", attempt_no=1, retry=False)
    led.claim_reset(key="k", attempt_id="a1", attempt_no=1, what="build")
    led.claim_reset(key="k", attempt_id="a1", attempt_no=1, what="reset")
    assert led.reset_left() == 1
    again = _ledger(tmp_path)  # 模拟进程重启
    again.start(3, 1)
    assert again.reset_claims == 2 and again.reset_left() == 1
    again.claim_reset(key="k", attempt_id="a2", attempt_no=2, what="build")
    with pytest.raises(ec.ResetBudgetExhausted):
        again.claim_reset(key="k", attempt_id="a2", attempt_no=2, what="reset")
    claims = [r for r in F.read_jsonl(tmp_path / "x.ledger.jsonl") if r["kind"] == "reset_claim"]
    assert [c["n"] for c in claims] == [1, 2, 3]  # 被拒的那次不写行


def test_budget_raise_only_when_cli_exceeds_history(tmp_path):
    led = _ledger(tmp_path)
    led.start(5, 1)
    _ledger(tmp_path).start(5, 1)  # 相同：不记提升
    _ledger(tmp_path).start(4, 1)  # 更小：不记提升
    _ledger(tmp_path).start(9, 1, reason="用户批准")
    raises = [r for r in F.read_jsonl(tmp_path / "x.ledger.jsonl") if r["kind"] == "budget_raise"]
    assert [(r["from"], r["to"], r["reason"]) for r in raises] == [(5, 9, "用户批准")]


@pytest.mark.parametrize("record,final", [
    ({"status": "success"}, True),
    ({"status": "fail"}, True),
    ({"status": "timeout"}, True),
    ({"status": "error", "infra": False}, True),
    ({"status": "error", "infra": True}, False),
    ({"status": "error", "infra": False, "budget_exhausted": True}, False),
    ({"status": "error", "infra": False, "run_blocked": True}, False),
    ({"status": "ongoing"}, False),
    ({"status": None}, False),
])
def test_is_final_table(record, final):
    assert F.env_client().AttemptLedger.is_final(record) is final


def test_single_accept_and_late(tmp_path):
    led = _ledger(tmp_path)
    led.start(10, 2)
    for aid, no, status, infra in (("a1", 1, "error", True), ("a2", 2, "fail", False), ("a3", 3, "success", False)):
        led.attempt_start(key="k", attempt_id=aid, attempt_no=no, retry=no > 1)
        late = led.attempt_end(_row("k", aid, no, status=status, infra=infra))
        assert late is (aid == "a3")
    rows = F.read_jsonl(tmp_path / "x.ledger.jsonl")
    acc = [r for r in rows if r["kind"] == "accept"]
    assert [(r["accepted_attempt_id"], r["status"]) for r in acc] == [("a2", "fail")]
    ends = {r["attempt_id"]: r["late"] for r in rows if r["kind"] == "attempt_end"}
    assert ends == {"a1": False, "a2": False, "a3": True}
    assert _ledger(tmp_path).accepted == {"k": "a2"}


def test_void_attempts_do_not_count(tmp_path):
    """额度耗尽而作废的尝试不计入每身份 2 次上限，也不计入 infra 重试额度。"""
    led = _ledger(tmp_path)
    led.start(10, 1)
    led.attempt_start(key="k", attempt_id="v1", attempt_no=1, retry=False)
    led.attempt_end(_row("k", "v1", 1, status="error", budget_exhausted=True))
    led.attempt_start(key="k", attempt_id="v2", attempt_no=2, retry=True)
    led.attempt_end(_row("k", "v2", 2, status="error", budget_exhausted=True))
    assert led.attempts_total("k") == 2 and led.attempts_used("k") == 0 and led.infra_retries_used() == 0
    led.attempt_start(key="k", attempt_id="i1", attempt_no=3, retry=False)
    led.attempt_end(_row("k", "i1", 3, status="error", infra=True))
    led.attempt_start(key="k", attempt_id="i2", attempt_no=4, retry=True)
    assert led.attempts_used("k") == 2 and led.infra_retries_used() == 1 and led.infra_retries_left() == 0


def test_last_end_final_and_dangling(tmp_path):
    led = _ledger(tmp_path)
    led.start(10, 1)
    led.attempt_start(key="a", attempt_id="a1", attempt_no=1, retry=False)
    led.attempt_end(_row("a", "a1", 1, status="error", infra=True))
    assert led.last_end_final("a") is False
    led.attempt_start(key="b", attempt_id="b1", attempt_no=1, retry=False)
    led.attempt_end(_row("b", "b1", 1, status="error", infra=False))
    assert led.last_end_final("b") is True
    led.attempt_start(key="c", attempt_id="c1", attempt_no=1, retry=False)  # 无 attempt_end：进程被杀
    assert [r["attempt_id"] for r in led.dangling()] == ["c1"]
    assert led.last_end_final("c") is False


LABEL = "perceptual-framesamp-modul"


def _seat_ledger(stage, **kw):
    ec = F.env_client()
    d = F.seat_dir(stage, LABEL)
    led = ec.AttemptLedger(d / f"{LABEL}.ledger.jsonl", seat="s00", policy=LABEL, shared=None, **kw)
    led.start(100, 10)
    return led


def test_recover_dangling_from_result_json(tmp_path):
    """悬空尝试（进程被杀）：attempt_start 记下的本局 result.json 在且尝试号相同就按它补 attempt_end、accept 与队列
    accepted 标记；没有就记基础设施错误（不 accept）。"""
    task, tier = F.v9_cells_sorted()[0]
    a, b = F.packaged_identity(task, tier, 0), F.packaged_identity(task, tier, 1)
    stage = tmp_path / "stage"
    led = _seat_ledger(stage)
    res_a = tmp_path / "ra" / "result.json"
    res_a.parent.mkdir()
    res_a.write_text(json.dumps({"key": a["key"], "attempt": 1, "status": "fail", "infra": False, "exec_steps": 5}))
    led.attempt_start(key=a["key"], attempt_id="x1", attempt_no=1, retry=False, identity=a, result_path=str(res_a))
    led.attempt_start(key=b["key"], attempt_id="x2", attempt_no=1, retry=False, identity=b,
                      result_path=str(tmp_path / "nope" / "result.json"))
    runner = F.make_runner(stage, LABEL, F.fake_seat_policy(), F.World())
    try:
        assert runner.recover_dangling() == 2
    finally:
        F.episode_mod().BUILDER_FACTORY = runner._fake_restore
    rows = F.seat_ledger(stage, LABEL)
    ends = {r["attempt_id"]: r for r in rows if r["kind"] == "attempt_end"}
    assert ends["x1"]["status"] == "fail" and ends["x1"]["recovered"] is True
    assert ends["x2"]["status"] == "error" and ends["x2"]["infra"] is True
    assert [r["accepted_attempt_id"] for r in rows if r["kind"] == "accept"] == ["x1"]
    acc = sorted(p.name for p in (F.queue_dir(stage, LABEL) / "accepted").glob("*.json"))
    assert acc == [f"ood__{a['key']}.json"]


def test_resume_skips_accepted_and_exhausted(tmp_path, capsys):
    """重启后：已 accepted 的身份、尝试名额用满的身份都跳过，不再建环境。b 两次 infra 用满仍无 accepted：两次运行都
    读回队列得 missing=1，打印 RUN_INCOMPLETE、退出码 6。"""
    (t1, tier1), (t2, tier2) = F.v9_cells_sorted()[:2]
    a, b = F.packaged_identity(t1, tier1, 0), F.packaged_identity(t2, tier2, 0)
    world = F.World({(b["task"], b["builder_episode"]): [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2"))]})
    stage = tmp_path / "stage"
    assert F.run_rows(F.make_runner(stage, LABEL, F.fake_seat_policy(), world), [a, b]) == 6
    n_envs = len(world.envs)
    assert n_envs == 3  # a 一次成功；b 两次 infra
    assert F.run_rows(F.make_runner(stage, LABEL, F.fake_seat_policy(), world), [a, b]) == 6
    assert len(world.envs) == n_envs
    out = capsys.readouterr().out
    assert out.count("RUN_INCOMPLETE") == 2 and f"missing=1 first=ood:{b['key']}" in out


def test_all_accepted_returns_zero(tmp_path):
    task, tier = F.v9_cells_sorted()[0]
    a = F.packaged_identity(task, tier, 0)
    assert F.run_rows(F.make_runner(tmp_path / "stage", LABEL, F.fake_seat_policy(), F.World()), [a]) == 0


@pytest.mark.parametrize("status,infra,recovered", [("fail", False, True), ("error", False, True),
                                                    ("error", True, False)],
                         ids=["fail", "ordinary_error", "infra_error"])
def test_crash_window_final_end_without_accept_gets_accept(tmp_path, capsys, status, infra, recovered):
    """attempt_end 已写、accept 未写时进程死掉：恢复时对「最后一次 attempt_end 是最终结局」的身份补写本地 accept 与
    队列 accepted 标记（recovered=true），不再建环境；最后一次是 infra 错误的不补，照常重试一次。"""
    task, tier = F.v9_cells_sorted()[0]
    a = F.packaged_identity(task, tier, 0)
    stage = tmp_path / "stage"
    led = _seat_ledger(stage)
    led.attempt_start(key=a["key"], attempt_id="x1", attempt_no=1, retry=False, identity=a)
    led.append({"kind": "attempt_end", "key": a["key"], "attempt_id": "x1", "attempt_no": 1, "status": status,
                "infra": infra, "cap_hit": False, "exec_steps": 3, "budget_exhausted": False, "late": False})
    # 队列里该尝试的领取文件已收尾（同一席位旧进程留下）
    q = F.env_client().DynamicQueue(F.queue_dir(stage, LABEL), seat="other", wall_s=60)
    q.end_claim(q.create_claim(a, 1), "infra" if infra else status)
    world = F.World()
    rc = F.run_rows(F.make_runner(stage, LABEL, F.fake_seat_policy(), world), [a])
    assert rc == 0
    rows = F.seat_ledger(stage, LABEL)
    acc = [r for r in rows if r["kind"] == "accept"]
    assert len(acc) == 1
    if recovered:
        assert acc[0]["accepted_attempt_id"] == "x1" and acc[0]["recovered"] is True and acc[0]["status"] == status
        assert world.envs == []
        assert f"LEDGER_RECOVER_ACCEPT key={a['key']} attempt_id=x1" in capsys.readouterr().out
    else:
        assert acc[0]["accepted_attempt_id"] != "x1" and "recovered" not in acc[0]
        assert len(world.envs) == 1
