"""S8 预算与额度（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S8 预算与额度」、六节、R2、R6、R11）。

被测：``dev-scripts/gl/budget_ledger.py``（共享账本与 CLI）与 ``dev-scripts/gl/seat.py`` 的共享模式。期望值一律由本文件
手写的事件序列推出：

* 并发争抢最后一个额度仅一方成功（线程与 CLI 子进程两种）；
* 前两次 reset 失败、第三次成功的夹具记满 6 次 reset；
* 余额不足时在进入 attempt 前拒绝（不写 attempt_start、不建环境、退出码 5）；
* 到期（有 Slurm 证据）与故障分账，两者都占每身份 2 次名额；重启换节点不刷新；
* Astra 第 3 局拒绝；
* 正常 fail／timeout／非 infra error 不重评；
* 拆仓验收（R5）：reset 计量是硬上限，预约与实领超额一律拒绝、不写行；``reset_cap`` 进 config 行。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import eval_fakes as F
from tests._support.loaders import load_script, script_path


def bl_mod():
    return load_script("eval-official/budget_ledger.py")


def _cli(ledger: Path, *argv: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, **(env_extra or {}))
    env.pop("SGEVAL_BUDGET_LEDGER", None)
    return subprocess.run([sys.executable, str(script_path("eval-official/budget_ledger.py")), "--ledger", str(ledger),
                           *argv], capture_output=True, text=True, env=env, timeout=60)


@pytest.fixture(autouse=True)
def _no_gate_env(monkeypatch):
    """测试默认不带门控环境变量（模拟 BASE 环境）；需要时各用例自行设置。"""
    for k in ("SGEVAL_BUDGET_LEDGER", "SGEVAL_EXPIRED_JOBS", "SLURM_JOB_ID", "SLURM_JOB_END_TIME"):
        monkeypatch.delenv(k, raising=False)


# ───────────────────────────── 共享账本本体 ─────────────────────────────


def test_concurrent_last_trajectory_only_one_wins(tmp_path):
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    bm.BudgetLedger(path, trajectory_cap=3).reserve(resets=2)
    bm.BudgetLedger(path, trajectory_cap=3).reserve(resets=2)
    n = 8
    barrier = threading.Barrier(n)
    wins, losses = [], []

    def worker():
        led = bm.BudgetLedger(path, trajectory_cap=3)  # 每个线程独立打开（独立文件描述）
        barrier.wait()
        try:
            wins.append(led.reserve(resets=2, route="perceptual-framesamp-modul/new"))
        except bm.BudgetExhausted:
            losses.append(1)

    ts = [threading.Thread(target=worker) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(wins) == 1 and len(losses) == n - 1
    assert bm.BudgetLedger(path, trajectory_cap=3).state().trajectories == 3  # 第三阶段：构造参数须与 config 行一致


def test_concurrent_last_shared_infra_only_one_wins(tmp_path):
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    n = 6
    barrier = threading.Barrier(n)
    got = []

    def worker(i):
        led = bm.BudgetLedger(path, shared_infra_cap=1)
        barrier.wait()
        got.append(led.claim_retry(route="smvla/orig" if i % 2 else "smvla/new", key=f"k{i}", interrupt="infra"))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(got) == [False] * (n - 1) + [True]


def test_concurrent_cli_reserve_only_one_wins(tmp_path):
    """两个 CLI 进程同时争最后一个轨迹名额：一个退出码 0，一个退出码 5（RUN_BLOCKED reason=budget）。"""
    path = tmp_path / "budget.jsonl"
    assert _cli(path, "--trajectory-cap", "2", "reserve", "--resets", "6").returncode == 0
    script = str(script_path("eval-official/budget_ledger.py"))
    env = {k: v for k, v in os.environ.items() if k != "SGEVAL_BUDGET_LEDGER"}
    procs = [subprocess.Popen([sys.executable, script, "--ledger", str(path), "--trajectory-cap", "2", "reserve",
                               "--resets", "6", "--route", "smvla/orig", "--key", f"k{i}"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for i in range(2)]
    outs = [(p.wait(timeout=60), p.stdout.read()) for p in procs]
    codes = sorted(c for c, _ in outs)
    assert codes == [0, 5]
    assert any("RUN_BLOCKED reason=budget" in o for _, o in outs)
    assert any(o.strip().splitlines()[-1].startswith("BUDGET_RESERVE ") for _, o in outs)


def test_reset_cap_is_hard(tmp_path, capsys):
    """R5：计量额度硬上限。预约后计量超额 → BudgetExhausted(reason=reset_cap)、不写行；预约数之内的实领不增加总数、
    照常记；超出预约数的实领使总数超额 → 同样拒绝、不写行。旧名 reset_soft_cap 仍可用、同为硬上限。"""
    bm = bl_mod()
    path = tmp_path / "b.jsonl"
    led = bm.BudgetLedger(path, reset_cap=5)
    rid = led.reserve(resets=4)
    with pytest.raises(bm.BudgetExhausted) as ei:
        led.reserve(resets=2)  # 4 + 2 > 5
    assert ei.value.reason == "reset_cap" and "BUDGET_REJECT resets=6 cap=5" in capsys.readouterr().err
    for what in ("build", "reset", "build", "reset"):
        led.claim_reset(rid, what)  # 预约数之内
    led.claim_reset(rid, "build")  # 第 5 次：总数 5，等于上限照记
    with pytest.raises(bm.BudgetExhausted):
        led.claim_reset(rid, "reset")  # 第 6 次：超额
    rows = F.read_jsonl(path)
    assert [r["kind"] for r in rows].count("reserve") == 1 and [r["kind"] for r in rows].count("reset_claim") == 5
    config = next(r for r in rows if r["kind"] == "config")
    assert config["reset_cap"] == 5 and config["schema"] == "sgeval-budget/3"
    with pytest.raises(bm.BudgetConfigMismatch):
        bm.BudgetLedger(path, reset_cap=60).check()  # 入口口径不一致即拒
    ok, lines = bm.BudgetLedger(path, reset_soft_cap=5).report_lines()
    assert ok and lines[-1] == "BUDGET_ENFORCEMENT=PASS trajectories=1/6366 resets=5/5 astra=0/2 shared_infra=0/50"


def test_cli_reserve_rejects_over_reset_cap(tmp_path):
    path = tmp_path / "b.jsonl"
    assert _cli(path, "--reset-cap", "3", "reserve", "--resets", "2").returncode == 0
    r = _cli(path, "--reset-cap", "3", "reserve", "--resets", "2")
    assert r.returncode == 5 and "RUN_BLOCKED reason=budget" in r.stdout and "budget_reason=reset_cap" in r.stdout


def test_astra_third_episode_rejected(tmp_path):
    bm = bl_mod()
    path = tmp_path / "b.jsonl"
    led = bm.BudgetLedger(path)
    led.reserve(resets=2, route="astra/new", astra=True)
    led.reserve(resets=2, route="astra/new", astra=True)
    with pytest.raises(bm.BudgetExhausted):
        bm.BudgetLedger(path).reserve(resets=2, route="astra/new", astra=True)
    r = _cli(path, "reserve", "--resets", "2", "--route", "astra/new", "--astra")
    assert r.returncode == 5 and "RUN_BLOCKED reason=budget" in r.stdout
    led.reserve(resets=2, route="perceptual-framesamp-modul/new")  # 非 Astra 不受影响
    last = _cli(path, "report").stdout.strip().splitlines()[-1]
    assert last == "BUDGET_ENFORCEMENT=PASS trajectories=3/6366 resets=6/141430 astra=2/2 shared_infra=0/50"


def test_release_returns_trajectory_but_keeps_claimed_resets(tmp_path):
    bm = bl_mod()
    led = bm.BudgetLedger(tmp_path / "b.jsonl", trajectory_cap=1)
    rid = led.reserve(resets=6)
    led.claim_reset(rid, "build")
    led.release(rid)
    st = led.state()
    assert st.trajectories == 0 and st.resets == 1
    led.reserve(resets=2)  # 名额已退回
    with pytest.raises(bm.BudgetExhausted):
        led.reserve(resets=2)


def test_cli_reserve_commit_release_report(tmp_path):
    """原侧启动器口径（R11）：SimpleMemVLA 每局 reserve --resets 6、FrameSamp+Modulation 每局 --resets 2；commit 可写实际数。"""
    path = tmp_path / "b.jsonl"
    r1 = _cli(path, "reserve", "--resets", "6", "--route", "smvla/orig", "--key", "a")
    r2 = _cli(path, "reserve", "--resets", "2", "--route", "perceptual-framesamp-modul/orig", "--key", "b")
    r3 = _cli(path, "reserve", "--resets", "2", "--route", "perceptual-framesamp-modul/orig", "--key", "c")
    rid1, rid2, rid3 = (r.stdout.strip().split("rid=")[-1] for r in (r1, r2, r3))
    assert _cli(path, "commit", "--id", rid1).returncode == 0
    assert _cli(path, "commit", "--id", rid2, "--resets", "4").returncode == 0
    assert _cli(path, "release", "--id", rid3).returncode == 0
    assert _cli(path, "claim-retry", "--route", "perceptual-framesamp-modul/orig", "--key", "b", "--interrupt", "infra").returncode == 0
    assert _cli(path, "claim-retry", "--route", "perceptual-framesamp-modul/orig", "--key", "b", "--interrupt", "expired").returncode == 5
    rep = _cli(path, "report")
    assert rep.returncode == 0
    assert rep.stdout.strip().splitlines()[-1] == (
        "BUDGET_ENFORCEMENT=PASS trajectories=2/6366 resets=10/141430 astra=0/2 shared_infra=1/50")


def test_report_fails_on_torn_row(tmp_path):
    path = tmp_path / "b.jsonl"
    bl_mod().BudgetLedger(path).reserve(resets=2)
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"kind": "reserve", "rid"')  # 写入中被杀的半行
    rep = _cli(path, "report")
    assert rep.returncode == 1
    assert rep.stdout.strip().splitlines()[-1].startswith("BUDGET_ENFORCEMENT=FAIL ")


def test_retry_quota_survives_restart_and_node_change(tmp_path, monkeypatch):
    """额度只从账本推出：换进程、换节点（主机名不同）都不刷新；同一路线同一身份至多 1 次重试（infra 与 expired 合计）。"""
    bm = bl_mod()
    path = tmp_path / "b.jsonl"
    assert bm.BudgetLedger(path).claim_retry(route="pp/new", key="k", interrupt="expired") is True
    monkeypatch.setattr("socket.gethostname", lambda: "gl-other-node")
    again = bm.BudgetLedger(path)
    assert again.claim_retry(route="pp/new", key="k", interrupt="infra") is False
    assert again.claim_retry(route="pp/orig", key="k", interrupt="infra") is True  # 另一侧是另一身份
    st = again.state()
    assert (st.retries_of("expired"), st.retries_of("infra")) == (1, 1)
    hosts = {r["host"] for r in F.read_jsonl(path)}
    assert "gl-other-node" in hosts


# ───────────────────────────── EnvSession 门控 ─────────────────────────────


class _ResetFailEnv:
    def __init__(self, fail: bool):
        self.fail = fail

    def reset(self):
        if self.fail:
            raise RuntimeError("reset 失败（夹具）")
        return F.obs_of([1, 2, 3]), {"task_goal": ["g"], "status": "ongoing"}

    def close(self):
        pass


class _Builder:
    def __init__(self, fails: int):
        self.fails = fails
        self.made = 0

    def make_env_for_episode(self, ep):
        self.made += 1
        return _ResetFailEnv(self.made <= self.fails)


def test_two_failed_resets_then_success_counts_six(tmp_path):
    """前两次尝试 reset 失败、第三次成功：每次尝试 build 与 reset 各领 1 次 → 共享账本记满 6（= 原侧每局预约数）。"""
    from robomme_ood_eval.session import EnvSession

    bm = bl_mod()
    led = bm.BudgetLedger(tmp_path / "b.jsonl")
    rid = led.reserve(resets=6, route="smvla/orig", key="k")
    builder = _Builder(fails=2)
    outcomes = []
    for _ in range(3):
        s = EnvSession("PickXtimes", 0, builder=builder, budget_claim=lambda what: led.claim_reset(rid, what))
        try:
            s.reset()
            outcomes.append("ok")
        except RuntimeError:
            outcomes.append("fail")
        assert s.reset_calls == 2
        s.close()
    led.commit(rid)
    assert outcomes == ["fail", "fail", "ok"]
    st = led.state()
    assert st.claimed[rid] == 6 and st.resets_of(rid) == 6 and st.resets == 6
    whats = [r["what"] for r in F.read_jsonl(tmp_path / "b.jsonl") if r["kind"] == "reset_claim"]
    assert whats == ["build", "reset"] * 3


def test_envsession_default_does_not_touch_budget(tmp_path):
    from robomme_ood_eval.session import EnvSession

    s = EnvSession("PickXtimes", 0, builder=_Builder(fails=0))
    assert s.budget_claim is None
    s.reset()
    assert s.reset_calls == 2
    assert not list(tmp_path.iterdir())


# ───────────────────────────── SeatRunner 共享模式 ─────────────────────────────

LABEL = "perceptual-framesamp-modul"
ROUTE = f"{LABEL}/seed7/new"


def _ident(i=0):
    task, tier = F.v9_cells_sorted()[i]
    return F.packaged_identity(task, tier, 0)


def _runner(tmp_path, world, budget_ledger, **kw):
    return F.make_runner(tmp_path / "stage", LABEL, F.fake_seat_policy(), world, budget_ledger=budget_ledger, **kw)


def test_insufficient_budget_rejected_before_claim(tmp_path, capsys):
    """领取前先预约：名额已满即 RUN_BLOCKED reason=budget、退出 5；不写 attempt_start、不建领取文件、不建环境。"""
    bm = bl_mod()
    shared = bm.BudgetLedger(tmp_path / "budget.jsonl", trajectory_cap=1)
    shared.reserve(resets=6, route="smvla/orig", key="other")  # 名额已满
    world = F.World()
    assert F.run_rows(_runner(tmp_path, world, shared), [_ident()]) == 5
    rows = F.seat_ledger(tmp_path / "stage", LABEL)
    assert not [r for r in rows if r["kind"] in ("attempt_start", "reset_claim", "attempt_end")]
    assert world.envs == []
    assert not list((F.queue_dir(tmp_path / "stage", LABEL) / "claims").glob("*.json"))
    assert "RUN_BLOCKED reason=budget policy=perceptual-framesamp-modul" in capsys.readouterr().out


def test_shared_mode_infra_then_success(tmp_path):
    """共享模式：首试 infra 错误 → 向共享账本原子领 infra 名额 → 重试成功；两次尝试各预约一条轨迹（首试 first、重试
    recovery，token 为 <路线>|<key>|a<N>）、build+reset 各记一次；两次都 commit；report 末行 PASS。"""
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    a = _ident()
    world = F.World({(a["task"], a["builder_episode"]): [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2")),
                                                         F.Plan(success_at=3)]})
    assert F.run_rows(_runner(tmp_path, world, str(path)), [a]) == 0
    st = bm.BudgetLedger(path).state()
    assert st.trajectories == 2 and len(st.commits) == 2 and st.resets == 4
    assert [(r["route"], r["key"], r["interrupt"]) for r in st.retries] == [(ROUTE, a["key"], "infra")]
    res = [st.reserves[r] for r in st.reserves]
    assert [(r["kind_of_try"], r["token"]) for r in res] == [("first", f"{ROUTE}|{a['key']}|a1"),
                                                             ("recovery", f"{ROUTE}|{a['key']}|a2")]
    local = F.seat_ledger(tmp_path / "stage", LABEL)
    starts = [r for r in local if r["kind"] == "attempt_start"]
    assert [r.get("interrupt") for r in starts] == [None, "infra"]
    assert all(r["route"] == ROUTE and r["budget_rid"] in st.reserves for r in starts)
    ok, lines = bm.BudgetLedger(path).report_lines()
    assert ok and lines[-1] == "BUDGET_ENFORCEMENT=PASS trajectories=2/6366 resets=4/141430 astra=0/2 shared_infra=1/50"


def test_reset_cap_rejects_before_claim(tmp_path, capsys):
    """计量额度硬上限（R5）：共享账本已用 2、上限 3，本局预约 2 次 reset 计量会超额 → 领取前拒绝、退出 5，不建环境。"""
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    shared = bm.BudgetLedger(path, reset_cap=3)
    shared.reserve(resets=2, route="smvla/orig", key="other")
    world = F.World()
    assert F.run_rows(_runner(tmp_path, world, shared), [_ident()]) == 5
    assert world.envs == [] and bm.BudgetLedger(path, reset_cap=3).state().trajectories == 1
    out = capsys.readouterr().out
    assert "RUN_BLOCKED reason=budget" in out and "budget_reason=reset_cap" in out


def test_env_var_gate_opens_shared_mode(tmp_path, monkeypatch):
    from tests._support.loaders import load_script as _ls

    path = tmp_path / "budget.jsonl"
    monkeypatch.setenv("SGEVAL_BUDGET_LEDGER", str(path))
    runner = F.make_runner(tmp_path / "stage", LABEL, F.fake_seat_policy(), F.World())
    # make_runner 缺省显式关闭共享（budget_ledger=None）；不给 shared 时由环境变量门控打开
    ec = _ls("eval-official/env_client.py")
    gated = ec.AttemptLedger(tmp_path / "x.jsonl", seat="s", policy=LABEL)
    assert runner.ledger.shared is None and gated.shared is not None


def test_expired_and_infra_counted_separately(tmp_path, monkeypatch):
    """两个悬空尝试：一个所在作业在到期清单里（expired），一个没有证据（infra）。恢复后分别记账；重试时分别占
    到期续跑与共享 infra 额度；两者都占每身份 2 次名额（重试后该身份 attempts_used == 2）。"""
    ec, bm = F.env_client(), bl_mod()
    path = tmp_path / "budget.jsonl"
    a, b = _ident(0), _ident(1)
    stage = tmp_path / "stage"
    pre = ec.AttemptLedger(F.seat_dir(stage, LABEL) / f"{LABEL}.ledger.jsonl", seat="s00", policy=LABEL,
                           shared=str(path), route=ROUTE)
    pre.start(100, 10)
    q = ec.DynamicQueue(F.queue_dir(stage, LABEL), seat="s00", wall_s=60)
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    pre.attempt_start(key=a["key"], attempt_id="xa", attempt_no=1, retry=False, identity=a,
                      claim=str(q.create_claim(a, 1)))
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    pre.attempt_start(key=b["key"], attempt_id="xb", attempt_no=1, retry=False, identity=b,
                      claim=str(q.create_claim(b, 1)))
    monkeypatch.setenv("SLURM_JOB_ID", "333")
    jobs = tmp_path / "expired-jobs.txt"
    jobs.write_text("111.batch\n999\n", encoding="utf-8")
    monkeypatch.setenv("SGEVAL_EXPIRED_JOBS", str(jobs))
    world = F.World()
    runner = _runner(tmp_path, world, str(path))
    assert runner.recover_dangling() == 2
    ends = {r["attempt_id"]: r for r in F.seat_ledger(stage, LABEL) if r["kind"] == "attempt_end"}
    assert ends["xa"]["interrupt"] == "expired" and "sacct_timeout job=111" in ends["xa"]["interrupt_evidence"]
    assert ends["xb"]["interrupt"] == "infra" and ends["xb"]["interrupt_evidence"] == "no_expiry_evidence"
    assert runner.ledger.interrupt_counts() == {"infra": 1, "expired": 1}
    assert F.run_rows(runner, [a, b]) == 0
    st = bm.BudgetLedger(path).state()
    assert (st.retries_of("expired"), st.retries_of("infra")) == (1, 1)
    led = runner.ledger
    assert led.attempts_used(a["key"]) == 2 and led.attempts_used(b["key"]) == 2
    assert len(world.envs) == 2


def test_classify_by_slurm_end_time(tmp_path):
    ec = F.env_client()
    led = ec.AttemptLedger(tmp_path / "x.jsonl", seat="s", policy=LABEL, shared=None, expired_jobs=[])
    start = {"attempt_id": "a", "t": 1000.0, "slurm_job_id": "5", "slurm_end_time": 2000}
    led.last_t["a"] = 1950.0
    assert led.classify_interrupt(start, now=2100.0)[0] == "expired"
    assert led.classify_interrupt(start, now=1990.0)[0] == "infra"  # 结束时刻未到
    led.last_t["a"] = 500.0
    assert led.classify_interrupt(start, now=2100.0)[0] == "infra"  # 早在到期前就停了：不是到期


def test_second_interrupt_exhausts_identity(tmp_path):
    """首试 infra、重试又 infra：该身份 2 次名额用满，不再领第三次（missing，退出码 6）。"""
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    a = _ident()
    world = F.World({(a["task"], a["builder_episode"]): [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2"))]})
    assert F.run_rows(_runner(tmp_path, world, str(path)), [a]) == 6
    assert len(world.envs) == 2
    assert len(bm.BudgetLedger(path).state().retries) == 1


def test_zero_retries_never_claims_retry(tmp_path):
    """本轮口径（--infra-retries 0）：infra 失败不领重试名额、不预约第二条轨迹。"""
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    a = _ident()
    world = F.World({(a["task"], a["builder_episode"]): [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2"))]})
    assert F.run_rows(_runner(tmp_path, world, str(path), infra_retries=0), [a]) == 6
    st = bm.BudgetLedger(path).state()
    assert st.retries == [] and st.trajectories == 1 and len(world.envs) == 1


@pytest.mark.parametrize("status", ["success", "fail", "timeout"])
def test_normal_outcomes_not_rerun_in_shared_mode(tmp_path, status):
    """队列里已 accepted（任一席位）的身份：不领取、不预约、不建环境。"""
    ec, bm = F.env_client(), bl_mod()
    path = tmp_path / "budget.jsonl"
    a = _ident()
    q = ec.DynamicQueue(F.queue_dir(tmp_path / "stage", LABEL), seat="other-seat", wall_s=60)
    assert q.accept(a, attempt=1, status=status, result=None)
    world = F.World()
    assert F.run_rows(_runner(tmp_path, world, str(path)), [a]) == 0
    assert world.envs == []
    st = bm.BudgetLedger(path).state()
    assert st.retries == [] and st.trajectories == 0
