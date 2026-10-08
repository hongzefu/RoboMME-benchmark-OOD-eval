"""C13 评估贯通（核心）：包内真实身份 → 执行清单 → 真实 ``SeatRunner``（真 perceptual-framesamp-modul／smvla 客户端 + 假连接 + CPU 假环境）
→ 结果行与持久账本 → 真实 ``eval_report.main`` 出报告，逐项核对分母、结局计数与判定行。

七种回合各自独立跑一个运行根（两种策略各一遍）；期望写成本文件里的手写表，不调用被测逻辑生成期望。
另有 A→B→A 连续三局：同一常驻客户端先后跑 A、B、A，第三局的执行动作必须与第一局逐位相同，且 B 局 server
收到的帧只来自 B（植入 M13：删掉策略 reset 必须被抓）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import eval_fakes as F

POLICIES = ("perceptual-framesamp-modul", "smvla")


def _svulkan():
    return RuntimeError("svulkan2: vk::Device lost（替身基础设施故障）")


def _ik():
    return RuntimeError("IK 求解失败（替身环境错误）")


# 期望表：plans 为同一身份逐次尝试的环境行为；outcome 为报告里该身份的结局；
# attempts／accepts 为账本 attempt_start／accept 行数；abandoned 为报告单列的废弃尝试数；
# cov／rep 为两行判定的 PASS 与否；exit 为客户端退出码（run_identities 返回值或 SystemExit 码：6 = 跑完仍有身份
# 无权威终态，5 = reset 额度耗尽）。
SCENARIOS = {
    "success": dict(plans=[F.Plan(success_at=3)], outcome="success", attempts=1, accepts=1, abandoned=0,
                    cov=True, rep=True, exit=0),
    "fail": dict(plans=[F.Plan(fail_at=2)], outcome="fail", attempts=1, accepts=1, abandoned=0,
                 cov=True, rep=True, exit=0),
    "cap_timeout": dict(plans=[F.Plan()], outcome="timeout", attempts=1, accepts=1, abandoned=0,
                        cov=True, rep=True, exit=0),
    "step_error": dict(plans=[F.Plan(raise_at=2, raise_exc=_ik)], outcome="error", attempts=1, accepts=1,
                       abandoned=0, cov=True, rep=True, exit=0),
    "step_infra_retry": dict(plans=[F.Plan(raise_at=2, raise_exc=_svulkan), F.Plan(success_at=3)],
                             outcome="success", attempts=2, accepts=1, abandoned=1, cov=True, rep=True, exit=0),
    "server_disconnect": dict(plans=[F.Plan(success_at=3)], server="infer_disconnect", outcome="missing",
                              attempts=2, accepts=0, abandoned=2, cov=False, rep=True, exit=6),
    "reset_budget": dict(plans=[F.Plan(success_at=3)], reset_budget=1, outcome="missing", attempts=1, accepts=0,
                         abandoned=1, cov=False, rep=True, exit=5),
}
OUTCOME_KEYS = {"success", "fail", "timeout", "error", "missing", "conflict"}


def _ident():
    task, tier = F.v9_cells_sorted()[0]
    return F.packaged_identity(task, tier, 0)


def _run_scenario(tmp_path, monkeypatch, capsys, policy: str, name: str):
    sc = SCENARIOS[name]
    ident = _ident()
    world = F.World({(ident["task"], ident["builder_episode"]): list(sc["plans"])})
    server = F.FakePolicyServer(fail_on=sc.get("server"))
    stage = tmp_path / "stage"
    runner = F.make_runner(stage, policy, F.policy_module(policy, monkeypatch, server), world,
                           reset_budget=sc.get("reset_budget", 100))
    rc = F.run_rows(runner, [ident])
    manifest = F.write_manifest(tmp_path / "manifest" / "manifest.json", [ident])
    rrc, lines, rep = F.run_report(capsys, manifest, stage, [policy], tmp_path / "report", "--expect-total", "1")
    out = stage / "s00" / policy
    return dict(sc=sc, ident=ident, world=world, server=server, rc=rc, rrc=rrc, lines=lines, rep=rep,
                results=F.read_jsonl(out / "results.jsonl"), ledger=F.read_jsonl(out / f"{policy}.ledger.jsonl"))


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("policy", POLICIES)
def test_seat_runner_to_report(tmp_path, monkeypatch, capsys, policy, name):
    r = _run_scenario(tmp_path, monkeypatch, capsys, policy, name)
    sc, ident, rep = r["sc"], r["ident"], r["rep"]
    assert r["rc"] == sc["exit"]

    # 账本：尝试数与唯一接受
    kinds = [x["kind"] for x in r["ledger"]]
    assert kinds.count("attempt_start") == sc["attempts"]
    assert kinds.count("accept") == sc["accepts"]
    assert kinds.count("attempt_end") == sc["attempts"]

    # 结果行：每次尝试一行，task_success 独立于 status 只在 success 时为真
    assert len(r["results"]) == sc["attempts"]
    assert [x["attempt_no"] for x in r["results"]] == list(range(1, sc["attempts"] + 1))
    for row in r["results"]:
        assert row["task_success"] is (row["status"] == "success")
        assert row["key"] == ident["key"] and row["spec_sha256"] == ident["spec_sha256"]

    # 报告：分母固定为清单身份数，六种结局键齐全且只有期望那一格为 1
    pp = rep["per_policy"][policy]
    assert pp["denominator"] == 1
    assert set(pp["outcomes"]) == OUTCOME_KEYS
    assert pp["outcomes"] == {k: int(k == sc["outcome"]) for k in OUTCOME_KEYS}
    assert len(pp["abandoned"]) == sc["abandoned"]
    assert pp["accepted"] == sc["accepts"]
    cov, rp = F.verdict(r["lines"], "V8_EVAL_COVERAGE"), F.verdict(r["lines"], "V8_EVAL_REPORT")
    assert cov[""] == ("PASS" if sc["cov"] else "FAIL")
    assert rp[""] == ("PASS" if sc["rep"] else "FAIL")
    assert cov["missing"] == str(int(sc["outcome"] == "missing"))
    assert cov["error_final"] == str(int(sc["outcome"] == "error"))
    assert rp["exec_over_cap"] == "0"
    assert r["rrc"] == (0 if sc["cov"] and sc["rep"] else 1)
    cell = f"{ident['task']}@{ident['tier']}"
    assert pp["cells"][cell]["denominator"] == 1
    assert pp["cells"][cell]["success"] == int(sc["outcome"] == "success")


@pytest.mark.parametrize("policy", POLICIES)
def test_cap_timeout_never_steps_past_cap(tmp_path, monkeypatch, capsys, policy):
    """ood（--max-steps 1600 --strict-cap）上限超时：环境恰好执行 1600 步，第 1601 次 step 不进入环境；
    按 timeout 计、不算基础设施。"""
    r = _run_scenario(tmp_path, monkeypatch, capsys, policy, "cap_timeout")
    cap = 1600
    (env,) = r["world"].envs
    assert env.n == cap
    (row,) = r["results"]
    assert row["status"] == "timeout" and row["cap_hit"] is True and row["infra"] is False
    assert row["exec_steps"] == cap and row["max_steps"] == row["effective_max_steps"] == cap
    assert row["dataset"] == "ood" and row["strict_cap"] is True
    assert r["world"].builders[0].max_steps == cap  # builder 构造参数即 --max-steps
    assert r["world"].make_calls[0][2] is None  # make_env_for_episode 不再逐局传步数


@pytest.mark.parametrize("policy", POLICIES)
def test_infra_retry_reruns_same_identity_once(tmp_path, monkeypatch, capsys, policy):
    """基础设施异常：同一身份立即重试一次，第二次尝试 retry=true，第一次记 infra 且不被接受。"""
    r = _run_scenario(tmp_path, monkeypatch, capsys, policy, "step_infra_retry")
    first, second = r["results"]
    assert first["status"] == "error" and first["infra"] is True
    assert second["status"] == "success" and second["infra"] is False
    starts = [x for x in r["ledger"] if x["kind"] == "attempt_start"]
    assert [x["retry"] for x in starts] == [False, True]
    (acc,) = [x for x in r["ledger"] if x["kind"] == "accept"]
    assert acc["accepted_attempt_id"] == second["attempt_id"]
    assert len(r["world"].envs) == 2 and all(e.closed for e in r["world"].envs)


@pytest.mark.parametrize("policy", POLICIES)
def test_ordinary_error_is_final_and_not_retried(tmp_path, monkeypatch, capsys, policy):
    """step 抛非基础设施异常：记 error、infra=false、被接受为终局、不重试；报告计 error_final。"""
    r = _run_scenario(tmp_path, monkeypatch, capsys, policy, "step_error")
    (row,) = r["results"]
    assert row["status"] == "error" and row["infra"] is False
    assert len(r["world"].envs) == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_disconnect_exhausts_two_attempts_then_missing(tmp_path, monkeypatch, capsys, policy):
    r = _run_scenario(tmp_path, monkeypatch, capsys, policy, "server_disconnect")
    assert all(x["status"] == "error" and x["infra"] is True for x in r["results"])
    assert r["rep"]["per_policy"][policy]["budget"]["infra_retries_total"] == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_reset_budget_exhaustion_stops_with_exit_5(tmp_path, monkeypatch, capsys, policy):
    """额度 1：build 领到唯一一次额度，reset 被拒 → 尝试作废（不 accept）、客户端以 5 退出。"""
    r = _run_scenario(tmp_path, monkeypatch, capsys, policy, "reset_budget")
    (row,) = r["results"]
    assert row["budget_exhausted"] is True and row["infra"] is False and row["status"] == "error"
    claims = [x for x in r["ledger"] if x["kind"] == "reset_claim"]
    assert [c["what"] for c in claims] == ["build"]
    assert r["world"].envs[0].resets == 0  # 被拒的 reset 没有进入环境


# ---------------------------------------------------------------- A→B→A 跨局隔离（M13）


def _abA(tmp_path, monkeypatch, policy):
    (ta, tier_a), (tb, tier_b) = F.v9_cells_sorted()[:2]
    a, b = F.packaged_identity(ta, tier_a, 0), F.packaged_identity(tb, tier_b, 0)
    world = F.World(default=F.Plan(success_at=40))
    server = F.FakePolicyServer()
    runner = F.make_runner(tmp_path / "stage", policy, F.policy_module(policy, monkeypatch, server), world)
    recs = [runner.run_one(a, attempt=1), runner.run_one(b, attempt=1), runner.run_one(a, attempt=2)]
    return a, b, world, server, recs


@pytest.mark.parametrize("policy", POLICIES)
def test_abA_third_episode_identical_to_first(tmp_path, monkeypatch, policy):
    a, b, world, server, recs = _abA(tmp_path, monkeypatch, policy)
    assert [r["status"] for r in recs] == ["success"] * 3
    env_a1, env_b, env_a2 = world.envs
    assert (env_a1.ep, env_b.ep, env_a2.ep) == (a["builder_episode"], b["builder_episode"], a["builder_episode"])
    # 第三局（A 再跑）执行的每一个动作与第一局逐位相同
    assert len(env_a1.actions) == len(env_a2.actions) == 40
    for x, y in zip(env_a1.actions, env_a2.actions):
        assert np.array_equal(x, y)
    # B 局与 A 局的动作不同（替身确实依赖本局输入，等式不是平凡成立）
    assert not np.array_equal(env_b.actions[0], env_a1.actions[0])
    # 每局 server 先收到 reset，且 reset 后收到的第一批帧恰为本局 reset 帧
    resets = [i for i, (k, _) in enumerate(server.log) if k == "reset"]
    assert len(resets) == 3
    for i, ident in zip(resets, (a, b, a)):
        kind, payload = server.log[i + 1]
        assert kind == "observe"
        want = [F.sha_bytes(F.frame(v)) for v in F.reset_values(ident["builder_episode"])]
        assert payload["frames"] == want


@pytest.mark.parametrize("policy", POLICIES)
def test_abA_ledger_marks_repeat_as_late(tmp_path, monkeypatch, policy):
    """同一身份第二次终态记 late，账本只有一条该身份的 accept（唯一接受）。"""
    a, b, world, server, recs = _abA(tmp_path, monkeypatch, policy)
    assert [r["late"] for r in recs] == [False, False, True]
    ledger = F.read_jsonl(tmp_path / "stage" / "s00" / policy / f"{policy}.ledger.jsonl")
    acc = [x for x in ledger if x["kind"] == "accept"]
    assert sorted(x["key"] for x in acc) == sorted([a["key"], b["key"]])
    assert [x["accepted_attempt_id"] for x in acc if x["key"] == a["key"]] == [recs[0]["attempt_id"]]
