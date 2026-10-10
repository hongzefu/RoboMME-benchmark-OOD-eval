"""席位贯通（核心）：包内真实身份 → 真实 ``seat.py`` 的 ``SeatRunner``（常驻假模型 + 动态队列 + CPU 假环境）→ 评估包
``run_episode`` 的 ``result.json`` → 本席位账本、队列 ``accepted/`` 标记与席位结果行，逐项核对尝试数、接受数与退出码。

拆仓后报告不再从席位目录读（``eval_report`` 的旧布局不变、不在此贯通），本文件改为核新产物树与队列；期望写成本文件里
的手写表，不调用被测逻辑生成期望。另有常驻 A→B→A：同一个 Policy 实例先后跑 A、B、A，``reset`` 三次，第三局的执行
动作必须与第一局逐位相同，且 B 局 server 收到的帧只来自 B。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import eval_fakes_dev as F

LABEL = "perceptual-framesamp-modul"  # 队列与席位目录标签（任意已登记模型名）


def _svulkan():
    return RuntimeError("svulkan2: vk::Device lost（替身基础设施故障）")


def _ik():
    return RuntimeError("IK 求解失败（替身环境错误）")


# 期望表：plans 为同一身份逐次尝试的环境行为；status 为权威 result.json 的终态（error 已并入 fail）；
# attempts／accepted 为本席位账本 attempt_start 行数与队列 accepted 标记数；exit 为 SeatRunner.run 返回值
# （6 = 跑完仍有身份无 accepted、已无尝试名额；5 = reset 额度耗尽）；retries 为每身份基础设施重试次数。
SCENARIOS = {
    "success": dict(plans=[F.Plan(success_at=3)], status="success", attempts=1, accepted=1, exit=0),
    "fail": dict(plans=[F.Plan(fail_at=2)], status="fail", attempts=1, accepted=1, exit=0),
    "cap_timeout": dict(plans=[F.Plan()], status="timeout", attempts=1, accepted=1, exit=0),
    "step_error": dict(plans=[F.Plan(raise_at=2, raise_exc=_ik)], status="fail", attempts=1, accepted=1, exit=0),
    "infra_retry_once": dict(plans=[F.Plan(raise_at=2, raise_exc=_svulkan), F.Plan(success_at=3)], status="success",
                             attempts=2, accepted=1, exit=0),
    "infra_no_retry": dict(plans=[F.Plan(raise_at=2, raise_exc=_svulkan)], status=None, attempts=1, accepted=0,
                           exit=6, retries=0),
    "server_disconnect": dict(plans=[F.Plan(success_at=3)], server="infer_disconnect", status=None, attempts=2,
                              accepted=0, exit=6),
    "reset_budget": dict(plans=[F.Plan(success_at=3)], reset_budget=1, status=None, attempts=1, accepted=0, exit=5),
}


def _ident():
    task, tier = F.v9_cells_sorted()[0]
    return F.packaged_identity(task, tier, 0)


def test_env_build_guard_settles_then_stops(tmp_path, capsys, monkeypatch):
    """构建失败保存真实结果并结清 rid，关闭策略，第二身份完全不尝试。"""
    # 复用生产席位打开账本的模块，避免重复导入数据类。
    bm = F.env_client().load_sibling("budget_ledger")
    shared = bm.BudgetLedger(tmp_path / "budget.jsonl")
    a = _ident()
    task, tier = F.v9_cells_sorted()[1]
    b = F.packaged_identity(task, tier, 0)
    server = F.FakePolicyServer()
    runner = F.make_runner(tmp_path / "stage", LABEL, F.fake_seat_policy(server), F.World(),
                           budget_ledger=shared, stop_on_env_build_error=True, infra_retries=19)
    closed = []
    monkeypatch.setattr(runner.fake_policy, "close", lambda: closed.append(True))
    def broken(*_args):
        raise RuntimeError("svulkan2 构建失败")
    monkeypatch.setattr(F.HybridBuilder, "make_env_for_episode", broken)
    assert F.run_rows(runner, [a, b]) == 3
    rows = F.seat_results(tmp_path / "stage", LABEL)
    assert len(rows) == 1 and rows[0]["infra_reason"] == "env_build"
    assert Path(rows[0]["result"]).is_file()
    assert len(shared.state().reserves) == len(shared.state().commits) == 1
    assert not shared.state().retries
    ledger = F.seat_ledger(tmp_path / "stage", LABEL)
    assert sum(r["kind"] == "attempt_end" for r in ledger) == 1
    assert json.loads(next((F.queue_dir(tmp_path / "stage", LABEL) / "claims").glob("*.json")).read_text())["ended"]
    assert closed == [True]
    assert "RUN_BLOCKED reason=env_build" in capsys.readouterr().out


@pytest.mark.parametrize("first", [F.Plan(fail_at=2), F.Plan()])
def test_env_build_guard_keeps_normal_terminal_results(tmp_path, first):
    """正常失败或截断仍接受并继续下一身份，重启不重跑已接受身份。"""
    a = _ident()
    task, tier = F.v9_cells_sorted()[1]
    b = F.packaged_identity(task, tier, 0)
    world = F.World({(a["task"], a["builder_episode"]): [first],
                     (b["task"], b["builder_episode"]): [F.Plan(success_at=2)]})
    runner = F.make_runner(tmp_path / "stage", LABEL, F.fake_seat_policy(), world,
                           stop_on_env_build_error=True, infra_retries=19)
    assert F.run_rows(runner, [a, b]) == 0
    rows = F.seat_results(tmp_path / "stage", LABEL)
    assert [r["status"] for r in rows] == ["fail" if first.fail_at else "timeout", "success"]
    assert all(r["accepted"] and not r["infra"] for r in rows)
    again = F.make_runner(tmp_path / "stage", LABEL, F.fake_seat_policy(), F.World(),
                          stop_on_env_build_error=True, infra_retries=19)
    assert F.run_rows(again, [a, b]) == 0
    assert len(F.seat_results(tmp_path / "stage", LABEL)) == 2


def _run(tmp_path, name: str):
    sc = SCENARIOS[name]
    ident = _ident()
    world = F.World({(ident["task"], ident["builder_episode"]): list(sc["plans"])})
    server = F.FakePolicyServer(fail_on=sc.get("server"))
    stage = tmp_path / "stage"
    runner = F.make_runner(stage, LABEL, F.fake_seat_policy(server), world, reset_budget=sc.get("reset_budget", 100),
                           infra_retries=sc.get("retries", 1))
    rc = F.run_rows(runner, [ident])
    return dict(sc=sc, ident=ident, world=world, server=server, rc=rc, stage=stage, runner=runner,
                ledger=F.seat_ledger(stage, LABEL), results=F.seat_results(stage, LABEL),
                accepted=sorted((F.queue_dir(stage, LABEL) / "accepted").glob("*.json")),
                claims=sorted((F.queue_dir(stage, LABEL) / "claims").glob("*.json")))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_seat_runner_scenarios(tmp_path, name):
    r = _run(tmp_path, name)
    sc, ident = r["sc"], r["ident"]
    assert r["rc"] == sc["exit"]
    kinds = [x["kind"] for x in r["ledger"]]
    assert kinds.count("attempt_start") == sc["attempts"] == kinds.count("attempt_end")
    assert kinds.count("accept") == sc["accepted"]
    assert len(r["accepted"]) == sc["accepted"]
    assert len(r["results"]) == sc["attempts"]
    assert [x["attempt"] for x in r["results"]] == list(range(1, sc["attempts"] + 1))
    for row in r["results"]:
        assert row["key"] == ident["key"] and row["dataset"] == "ood" and row["policy_seed"] == 7
    if sc["accepted"]:
        acc = json.loads(r["accepted"][0].read_text())
        assert acc["key"] == ident["key"] and acc["status"] == sc["status"] and acc["attempt"] == sc["attempts"]
        res = json.loads(Path(acc["result"]).read_text())  # 权威 result.json
        assert res["status"] == sc["status"] and res["attempt"] == sc["attempts"]
        assert res["task_success"] == int(sc["status"] == "success")
        assert res["key"] == ident["key"] and res["spec_sha256"] == ident["spec_sha256"]
    # 领取文件：作废的（额度耗尽）删掉，其余全部记收尾，不留在跑状态
    for c in r["claims"]:
        assert json.loads(c.read_text())["ended"] is True
    if name == "reset_budget":
        assert r["claims"] == []


def test_cap_timeout_never_steps_past_cap(tmp_path):
    """ood 严格截断 1800：环境恰好执行 1800 步，第 1801 次 step 不进入环境；按 timeout 计、不算基础设施。"""
    r = _run(tmp_path, "cap_timeout")
    (env,) = r["world"].envs
    assert env.n == 1800
    (row,) = r["results"]
    assert row["status"] == "timeout" and row["cap_hit"] is True and row["infra"] is False
    assert row["exec_steps"] == 1800
    assert r["world"].builders[0].max_steps == 1800


def test_infra_retry_reruns_same_identity_once(tmp_path):
    r = _run(tmp_path, "infra_retry_once")
    first, second = r["results"]
    assert first["infra"] is True and first["final"] is False and first["accepted"] is False
    assert second["status"] == "success" and second["final"] is True and second["accepted"] is True
    starts = [x for x in r["ledger"] if x["kind"] == "attempt_start"]
    assert [x["retry"] for x in starts] == [False, True]
    (acc,) = [x for x in r["ledger"] if x["kind"] == "accept"]
    assert acc["accepted_attempt_id"] == second["attempt_id"]
    assert len(r["world"].envs) == 2 and all(e.closed for e in r["world"].envs)
    claims = [json.loads(c.read_text()) for c in r["claims"]]
    assert [(c["attempt"], c["end_status"]) for c in claims] == [(1, "infra"), (2, "success")]


def test_infra_without_retry_is_missing_with_exit_6(tmp_path, capsys):
    """本轮口径：基础设施重试 0 次，失败即停，身份计 missing，退出 6。"""
    r = _run(tmp_path, "infra_no_retry")
    (row,) = r["results"]
    assert row["infra"] is True and row["final"] is False
    assert len(r["world"].envs) == 1
    out = capsys.readouterr().out
    assert "RUN_INCOMPLETE" in out and f"first=ood:{r['ident']['key']}" in out


def test_ordinary_error_is_final_and_not_retried(tmp_path):
    r = _run(tmp_path, "step_error")
    (row,) = r["results"]
    assert row["status"] == "fail" and row["error_kind"] == "error" and row["infra"] is False and row["final"]
    assert len(r["world"].envs) == 1


def test_reset_budget_exhaustion_stops_with_exit_5(tmp_path, capsys):
    """本席位 reset 额度 1：build 领到唯一一次额度，reset 被拒 → 尝试作废（不 accept、删领取文件）、以 5 退出。"""
    r = _run(tmp_path, "reset_budget")
    (row,) = r["results"]
    assert row["budget_exhausted"] is True and row["void"] is True
    claims = [x for x in r["ledger"] if x["kind"] == "reset_claim"]
    assert [c["what"] for c in claims] == ["build"]
    assert r["world"].envs[0].resets == 0
    assert "RESET_BUDGET_EXHAUSTED" in capsys.readouterr().out


def test_resume_skips_accepted_without_loading(tmp_path, capsys):
    """全部已 accepted 时再跑：不加载模型、不建环境、退出 0。"""
    r = _run(tmp_path, "success")
    calls = []
    world2 = F.World()
    runner = F.make_runner(r["stage"], LABEL, F.fake_seat_policy(), world2)
    runner.policy_factory = lambda *a, **k: calls.append(1)
    assert F.run_rows(runner, [r["ident"]]) == 0
    assert calls == [] and world2.envs == []
    assert "claimable=0" in capsys.readouterr().out


# ---------------------------------------------------------------- 常驻 A→B→A


def _abA(tmp_path):
    (ta, tier_a), (tb, tier_b) = F.v9_cells_sorted()[:2]
    a, b = F.packaged_identity(ta, tier_a, 0), F.packaged_identity(tb, tier_b, 0)
    world = F.World(default=F.Plan(success_at=40))
    server = F.FakePolicyServer()
    pol = F.fake_seat_policy(server)
    stage = tmp_path / "stage"
    ec = F.env_client()
    runner = F.make_runner(stage, LABEL, pol, world)
    try:
        recs = []
        for ident, n in ((a, 1), (b, 1), (a, 2)):
            c = ec.Claim(ident=ident, attempt=n, path=runner.queue.create_claim(ident, n), token="t", rid=None,
                         retry=n > 1)
            recs.append(runner.run_claim(pol, c))
    finally:
        F.episode_mod().BUILDER_FACTORY = runner._fake_restore
        F.episode_mod().clear_builders()
    return a, b, world, server, pol, recs


def test_abA_resident_policy_third_episode_identical_to_first(tmp_path):
    a, b, world, server, pol, recs = _abA(tmp_path)
    assert [r["status"] for r in recs] == ["success"] * 3
    assert pol.calls["reset"] == 3 and pol.calls["play"] == 3 and pol.calls["close"] == 0
    env_a1, env_b, env_a2 = world.envs
    assert (env_a1.ep, env_b.ep, env_a2.ep) == (a["builder_episode"], b["builder_episode"], a["builder_episode"])
    assert len(env_a1.actions) == len(env_a2.actions) == 40
    for x, y in zip(env_a1.actions, env_a2.actions):
        assert np.array_equal(x, y)
    assert not np.array_equal(env_b.actions[0], env_a1.actions[0])
    resets = [i for i, (k, _) in enumerate(server.log) if k == "reset"]
    assert len(resets) == 3
    for i, ident in zip(resets, (a, b, a)):
        kind, payload = server.log[i + 1]
        assert kind == "observe"
        assert payload["frames"] == [F.sha_bytes(F.frame(v)) for v in F.reset_values(ident["builder_episode"])]


def test_abA_ledger_marks_repeat_as_late(tmp_path):
    a, b, world, server, pol, recs = _abA(tmp_path)
    assert [r["late"] for r in recs] == [False, False, True]
    ledger = F.seat_ledger(tmp_path / "stage", LABEL)
    acc = [x for x in ledger if x["kind"] == "accept"]
    assert sorted(x["key"] for x in acc) == sorted([a["key"], b["key"]])
    assert [x["accepted_attempt_id"] for x in acc if x["key"] == a["key"]] == [recs[0]["attempt_id"]]
    # 队列：A 的第二次终态不再落 accepted（第一份即权威）
    assert [r["accepted"] for r in recs] == [True, True, False]
