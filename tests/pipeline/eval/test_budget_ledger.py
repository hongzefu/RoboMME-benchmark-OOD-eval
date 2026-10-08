"""S8 预算与额度（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S8 预算与额度」、六节、R2、R6、R11）。

被测：``scripts/eval-official/budget_ledger.py``（共享账本与 CLI）与 ``env_client.py`` 的共享模式门控。期望值一律由本文件
手写的事件序列推出：

* 并发争抢最后一个额度仅一方成功（线程与 CLI 子进程两种）；
* 前两次 reset 失败、第三次成功的夹具记满 6 次 reset；
* 余额不足时在进入 attempt 前拒绝（不写 attempt_start、不建环境、退出码 5）；
* 到期（有 Slurm 证据）与故障分账，两者都占每身份 2 次名额；重启换节点不刷新；
* Astra 第 3 局拒绝；
* 正常 fail／timeout／非 infra error 不重评；
* 默认不传新参数时 ``AttemptLedger`` 与 ``SeatRunner`` 的账本与 BASE 逐字节相同。
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

import eval_fakes as F
from tests._support.loaders import REPO, load_script, script_path

BASE = "b869a3df9e7406b8f5458697f22656165b9c50b3"


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


def test_reset_soft_cap_only_warns(tmp_path, capsys):
    bm = bl_mod()
    led = bm.BudgetLedger(tmp_path / "b.jsonl", reset_soft_cap=5)
    led.reserve(resets=4)
    rid = led.reserve(resets=4)  # 8 > 5：只告警
    led.claim_reset(rid, "reset")
    err = capsys.readouterr().err
    assert "BUDGET_WARN resets=8 soft=5" in err
    ok, lines = led.report_lines()
    assert ok and lines[-1] == ("BUDGET_ENFORCEMENT=PASS trajectories=2/6366 resets=8/5 astra=0/2 "
                                "shared_infra=0/50")


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
    ec, bm = F.env_client(), bl_mod()
    led = bm.BudgetLedger(tmp_path / "b.jsonl")
    rid = led.reserve(resets=6, route="smvla/orig", key="k")
    builder = _Builder(fails=2)
    outcomes = []
    for _ in range(3):
        s = ec.EnvSession("PickXtimes", 0, builder=builder, budget_claim=lambda what: led.claim_reset(rid, what))
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
    ec = F.env_client()
    s = ec.EnvSession("PickXtimes", 0, builder=_Builder(fails=0))
    assert s.budget_claim is None
    s.reset()
    assert s.reset_calls == 2
    assert not list(tmp_path.iterdir())


# ───────────────────────────── SeatRunner 共享模式 ─────────────────────────────


def _ident(i=0):
    task, tier = F.v9_cells_sorted()[i]
    return F.packaged_identity(task, tier, 0)


def test_insufficient_budget_rejected_before_attempt(tmp_path, monkeypatch, capsys):
    bm = bl_mod()
    shared = bm.BudgetLedger(tmp_path / "budget.jsonl", trajectory_cap=1)
    shared.reserve(resets=6, route="smvla/orig", key="other")  # 名额已满
    world = F.World()
    runner = F.make_runner(tmp_path, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                           budget_ledger=shared)
    assert F.run_rows(runner, [_ident()]) == 5
    rows = F.read_jsonl(tmp_path / "s00" / "perceptual-framesamp-modul" / "perceptual-framesamp-modul.ledger.jsonl")
    assert not [r for r in rows if r["kind"] in ("attempt_start", "reset_claim", "attempt_end")]
    assert world.envs == []
    assert not (tmp_path / "s00" / "perceptual-framesamp-modul" / "results.jsonl").exists()
    assert "RUN_BLOCKED reason=budget policy=perceptual-framesamp-modul" in capsys.readouterr().out


def test_shared_mode_infra_then_success(tmp_path, monkeypatch):
    """共享模式：首试 infra 错误 → 向共享账本原子领 infra 名额 → 重试成功；两次尝试各预约一条轨迹、build+reset
    各记一次；report 末行 PASS。"""
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    a = _ident()
    world = F.World({(a["task"], a["builder_episode"]): [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2")),
                                                         F.Plan(success_at=3)]})
    runner = F.make_runner(tmp_path, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                           budget_ledger=str(path))
    assert F.run_rows(runner, [a]) == 0
    st = bm.BudgetLedger(path).state()
    assert st.trajectories == 2 and len(st.commits) == 2 and st.resets == 4
    assert [(r["route"], r["key"], r["interrupt"]) for r in st.retries] == [("perceptual-framesamp-modul/new", a["key"], "infra")]
    local = F.read_jsonl(tmp_path / "s00" / "perceptual-framesamp-modul" / "perceptual-framesamp-modul.ledger.jsonl")
    starts = [r for r in local if r["kind"] == "attempt_start"]
    assert [r.get("interrupt") for r in starts] == [None, "infra"]
    assert all(r["route"] == "perceptual-framesamp-modul/new" and r["budget_rid"] in st.reserves for r in starts)
    ok, lines = bm.BudgetLedger(path).report_lines()
    assert ok and lines[-1] == "BUDGET_ENFORCEMENT=PASS trajectories=2/6366 resets=4/141430 astra=0/2 shared_infra=1/50"


def test_env_var_gate_opens_shared_mode(tmp_path, monkeypatch):
    path = tmp_path / "budget.jsonl"
    monkeypatch.setenv("SGEVAL_BUDGET_LEDGER", str(path))
    runner = F.make_runner(tmp_path, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), F.World())
    assert runner.ledger.shared is not None
    assert F.run_rows(runner, [_ident()]) == 0
    assert bl_mod().BudgetLedger(path).state().trajectories == 1


def test_expired_and_infra_counted_separately(tmp_path, monkeypatch):
    """两个悬空尝试：一个所在作业在到期清单里（expired），一个没有证据（infra）。恢复后分别记账；重试时分别占
    到期续跑与共享 infra 额度；两者都占每身份 2 次名额（重试后该身份 attempts_used == 2）。"""
    ec, bm = F.env_client(), bl_mod()
    path = tmp_path / "budget.jsonl"
    a, b = _ident(0), _ident(1)
    out = tmp_path / "s00" / "perceptual-framesamp-modul"
    pre = ec.AttemptLedger(out / "perceptual-framesamp-modul.ledger.jsonl", seat="s00", policy="perceptual-framesamp-modul", shared=str(path), route="perceptual-framesamp-modul/new")
    pre.start(100, 10)
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    pre.attempt_start(key=a["key"], attempt_id="xa", attempt_no=1, retry=False)
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    pre.attempt_start(key=b["key"], attempt_id="xb", attempt_no=1, retry=False)
    monkeypatch.setenv("SLURM_JOB_ID", "333")
    jobs = tmp_path / "expired-jobs.txt"
    jobs.write_text("111.batch\n999\n", encoding="utf-8")
    monkeypatch.setenv("SGEVAL_EXPIRED_JOBS", str(jobs))
    world = F.World()
    runner = F.make_runner(tmp_path, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                           budget_ledger=str(path))
    assert runner.recover_dangling() == 2
    ends = {r["attempt_id"]: r for r in F.read_jsonl(out / "perceptual-framesamp-modul.ledger.jsonl") if r["kind"] == "attempt_end"}
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
    led = ec.AttemptLedger(tmp_path / "x.jsonl", seat="s", policy="perceptual-framesamp-modul", shared=None, expired_jobs=[])
    start = {"attempt_id": "a", "t": 1000.0, "slurm_job_id": "5", "slurm_end_time": 2000}
    led.last_t["a"] = 1950.0
    assert led.classify_interrupt(start, now=2100.0)[0] == "expired"
    assert led.classify_interrupt(start, now=1990.0)[0] == "infra"  # 结束时刻未到
    led.last_t["a"] = 500.0
    assert led.classify_interrupt(start, now=2100.0)[0] == "infra"  # 早在到期前就停了：不是到期


def test_second_interrupt_exhausts_identity(tmp_path, monkeypatch):
    """首试到期中断、重试又 infra 失败：该身份 2 次名额用满，不再领第三次（missing，退出码 6）。"""
    bm = bl_mod()
    path = tmp_path / "budget.jsonl"
    a = _ident()
    world = F.World({(a["task"], a["builder_episode"]): [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2"))]})
    runner = F.make_runner(tmp_path, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                           budget_ledger=str(path))
    assert F.run_rows(runner, [a]) == 6
    assert len(world.envs) == 2
    assert len(bm.BudgetLedger(path).state().retries) == 1


@pytest.mark.parametrize("status,infra", [("fail", False), ("timeout", False), ("error", False)])
def test_normal_outcomes_not_rerun_in_shared_mode(tmp_path, monkeypatch, status, infra):
    ec, bm = F.env_client(), bl_mod()
    path = tmp_path / "budget.jsonl"
    a = _ident()
    out = tmp_path / "s00" / "perceptual-framesamp-modul"
    led = ec.AttemptLedger(out / "perceptual-framesamp-modul.ledger.jsonl", seat="s00", policy="perceptual-framesamp-modul", shared=str(path))
    led.start(100, 10)
    led.attempt_start(key=a["key"], attempt_id="x1", attempt_no=1, retry=False)
    led.attempt_end({"key": a["key"], "attempt_id": "x1", "attempt_no": 1, "status": status, "infra": infra})
    world = F.World()
    runner = F.make_runner(tmp_path, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                           budget_ledger=str(path))
    assert F.run_rows(runner, [a]) == 0
    assert world.envs == []
    st = bm.BudgetLedger(path).state()
    assert st.retries == [] and st.trajectories == 0


# ───────────────────────────── 默认行为与 BASE 逐字节相同（R2） ─────────────────────────────


def _base_env_client(tmp_path: Path):
    try:
        src = subprocess.run(["git", "show", f"{BASE}:scripts/eval-official/env_client.py"], cwd=REPO,
                             capture_output=True, text=True, check=True, timeout=60).stdout
    except (subprocess.SubprocessError, OSError) as e:
        pytest.skip(f"取不到 BASE 版本：{e}")
    d = tmp_path / "base-src" / "scripts" / "eval-official"
    d.mkdir(parents=True)
    p = d / "env_client.py"
    p.write_text(src, encoding="utf-8")
    name = f"_base_env_client_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _det(monkeypatch):
    """固定时间与 uuid：两边账本里的 t 与 attempt_id 可逐字节比较。"""
    clock = itertools.count(1_700_000_000)
    ids = itertools.count(1)
    monkeypatch.setattr("time.time", lambda: float(next(clock)))
    monkeypatch.setattr("uuid.uuid4", lambda: uuid.UUID(int=next(ids)))


def _ledger_script(mod, path: Path):
    led = mod.AttemptLedger(path, seat="s00", policy="perceptual-framesamp-modul")
    led.start(5, 1)
    led.attempt_start(key="k", attempt_id="a1", attempt_no=1, retry=False, task="T")
    led.claim_reset(key="k", attempt_id="a1", attempt_no=1, what="build")
    led.claim_reset(key="k", attempt_id="a1", attempt_no=1, what="reset")
    led.attempt_end({"key": "k", "attempt_id": "a1", "attempt_no": 1, "status": "error", "infra": True})
    led.attempt_start(key="k", attempt_id="a2", attempt_no=2, retry=True)
    led.attempt_end({"key": "k", "attempt_id": "a2", "attempt_no": 2, "status": "fail", "infra": False,
                     "recovered": True})
    led.attempt_start(key="j", attempt_id="b1", attempt_no=1, retry=False)
    mod.AttemptLedger(path, seat="s00", policy="perceptual-framesamp-modul").start(9, 1, reason="raise")
    again = mod.AttemptLedger(path, seat="s00", policy="perceptual-framesamp-modul")
    return (again.reset_left(), again.infra_retries_left(), again.attempts_used("k"), again.accepted,
            [r["attempt_id"] for r in again.dangling()])


def test_default_attempt_ledger_bytes_equal_base(tmp_path, monkeypatch):
    base = _base_env_client(tmp_path)
    cur = F.env_client()
    _det(monkeypatch)
    r_base = _ledger_script(base, tmp_path / "base.jsonl")
    _det(monkeypatch)
    r_cur = _ledger_script(cur, tmp_path / "cur.jsonl")
    assert r_base == r_cur
    assert (tmp_path / "base.jsonl").read_bytes() == (tmp_path / "cur.jsonl").read_bytes()


def _legacy_dataset(name: str) -> str:
    """改名前（BASE）接口里的数据集名：取自 official_defs 别名表（BASE 的 env_client 只认旧名）。"""
    defs = load_script("eval-official/official_defs.py")
    return next(o for o, n in defs.LEGACY_DATASET_ALIASES.items() if n == name)


def _runner_of(mod, stage: Path, policy_mod, world, *, dataset: str = "ood"):
    canon_ds = load_script("eval-official/official_defs.py").canonical_dataset  # builder 只认官方名
    out = stage / "s00" / "perceptual-framesamp-modul"
    args = F.seat_args(out, "perceptual-framesamp-modul", ledger=out / "perceptual-framesamp-modul.ledger.jsonl",
                       infra_retry_budget=1, dataset=dataset)
    return mod.SeatRunner(args, policy_mod=policy_mod, recorder_factory=lambda d, m: F.FakeRecorder(d, m, world),
                          builder_factory=lambda task, dataset, ms: F.HybridBuilder(task, ms, world, canon_ds(dataset)),
                          proc_info={"gpu_name": "fake", "gpu_uuid": "fake", "git_commit": "0" * 40,
                                     "git_dirty": False, "init_timing": {}})


def test_default_seat_runner_ledger_bytes_equal_base(tmp_path, monkeypatch):
    """不给 --budget-ledger、环境变量为空：同一身份清单（含 infra 重试与额度用尽）跑出的账本与 BASE 逐字节相同。"""
    base = _base_env_client(tmp_path)
    cur = F.env_client()
    a, b, c = _ident(0), _ident(1), _ident(2)
    boom = [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2"))]

    old_ds = _legacy_dataset("ood")

    def run(mod, stage, dataset):
        _det(monkeypatch)
        world = F.World({(b["task"], b["builder_episode"]): list(boom), (c["task"], c["builder_episode"]): list(boom)})
        runner = _runner_of(mod, stage, F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                            dataset=dataset)
        return F.run_rows(runner, [a, b, c]), len(world.envs)

    # BASE 只认改名前的数据集名：以旧名驱动（R1 只改名，账本行不含数据集名，仍须逐字节相同）
    assert run(base, tmp_path / "base", old_ds) == run(cur, tmp_path / "cur", "ood")
    lb = (tmp_path / "base" / "s00" / "perceptual-framesamp-modul" / "perceptual-framesamp-modul.ledger.jsonl").read_bytes()
    lc = (tmp_path / "cur" / "s00" / "perceptual-framesamp-modul" / "perceptual-framesamp-modul.ledger.jsonl").read_bytes()
    # 账本行带数据集名：BASE 侧的旧名换成官方名（只换这一个 JSON 字符串值）
    lb = lb.replace(f'"{old_ds}"'.encode(), b'"ood"')
    assert b"budget_rid" not in lc and b"interrupt" not in lc and b"token" not in lc
    # 第三阶段 progress.json 多了具名阶段写入（各取一次 time.time()），固定时钟下行内 t 的数值会错开：
    # 去掉 t 后逐字节相同（行的顺序、键、其余值全部不变）
    def _no_t(raw: bytes) -> list[bytes]:
        return [re.sub(rb'"t": [0-9.]+(, )?', b"", line) for line in raw.splitlines()]
    assert _no_t(lb) == _no_t(lc)
