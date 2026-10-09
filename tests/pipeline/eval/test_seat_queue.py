"""GL 席位动态队列（``dev-scripts/gl/seat.py`` 的 ``DynamicQueue``／``SeatRunner``）的核心机制，用注入的 ``now`` 与
``hard_exit`` 钉死（拆分方案第二部分 §二 seat.py 行）：

a. 心跳未超 2 倍墙钟的领取不被别的席位领走；
b. 超 2 倍墙钟被回收、计一次尝试；缺省 0 次重试时该身份 exhausted、退出 6；被回收尝试的共享账本 rid 当即结算；
c. 同席位名的旧进程留下的未收尾领取当即回收（不等过期）；
d. 两个席位并发争同一身份只有一方领到（身份锁 flock + O_EXCL）；
e. ``SeatRunner._hard_exit``：先写本地 attempt_end、commit 共享账本 rid、领取文件记收尾，再退 75；共享账本无悬空 rid；
f. 心跳与结算互斥：已收尾／已作废的领取文件不会被心跳写回在跑状态（F3）。

期望值全部手写；不跑仿真，假 builder 走 ``eval_fakes.HybridBuilder``。
"""
from __future__ import annotations

import json
import os
import threading
import time

import pytest

import eval_fakes_dev as F

LABEL = "perceptual-framesamp-modul"
T0 = 1_000_000.0


def _bl():
    from tests._support.dev_loaders import load_script

    return load_script("eval-official/budget_ledger.py")


def _ident(i: int = 0):
    task, tier = F.v9_cells_sorted()[i]
    return F.packaged_identity(task, tier, 0)


class Clock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _runner(tmp_path, seat, clock, world=None, **kw):
    return F.make_runner(tmp_path / "stage", LABEL, F.fake_seat_policy(), world or F.World(), seat_dir=seat,
                         now=clock, **kw)


def _restore(runner):
    F.episode_mod().BUILDER_FACTORY = runner._fake_restore
    F.episode_mod().clear_builders()


def _claims(tmp_path):
    return sorted((F.queue_dir(tmp_path / "stage", LABEL) / "claims").glob("*.json"))


def _open_rids(path) -> set:
    """共享账本里未结算（既未 commit 也未 release）的 rid。"""
    st = _bl().BudgetLedger(path).state()
    return {r for r in st.reserves if r not in st.commits and r not in st.releases}


def test_a_fresh_heartbeat_is_not_taken_by_other_seat(tmp_path, capsys):
    ident = _ident()
    a = _runner(tmp_path, "A", Clock())
    try:
        c = a.claim(ident)
    finally:
        _restore(a)
    assert c.attempt == 1
    before = c.path.read_bytes()
    world = F.World()
    b = _runner(tmp_path, "B", Clock(T0 + 2 * a.wall_s - 1), world)  # 差 1 秒才过期
    assert F.run_rows(b, [ident]) == 0
    assert world.envs == [] and c.path.read_bytes() == before and len(_claims(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "claimable=0" in out and "SEAT_IDLE" in out and "running_elsewhere=1" in out


def test_b_expired_claim_reclaimed_counts_attempt_and_exhausts_with_zero_retries(tmp_path, capsys):
    ident = _ident()
    book = tmp_path / "budget.jsonl"
    a = _runner(tmp_path, "A", Clock(), budget_ledger=str(book), infra_retries=0)
    try:
        c = a.claim(ident)
    finally:
        _restore(a)
    assert c.rid and json.loads(c.path.read_text())["rid"] == c.rid  # 预约的 rid 立即记进领取文件
    assert _open_rids(book) == {c.rid}
    world = F.World()
    b = _runner(tmp_path, "B", Clock(T0 + 2 * a.wall_s + 1), world, budget_ledger=str(book), infra_retries=0)
    assert F.run_rows(b, [ident]) == 6
    doc = json.loads(c.path.read_text())
    assert doc["ended"] is True and doc["end_status"] == "reclaimed" and doc["reclaim_reason"] == "expired"
    assert doc["reclaimed_by"] == "B" and len(_claims(tmp_path)) == 1  # 回收计一次尝试，0 次重试不再领第二次
    assert world.envs == []
    assert _open_rids(book) == set()  # 被回收尝试的 rid 已结算（F4）
    st = _bl().BudgetLedger(book).state()
    assert st.commits[c.rid]["infra"] is True
    out = capsys.readouterr().out
    assert "QUEUE_RECLAIM" in out and "reason=expired" in out and "RUN_INCOMPLETE" in out


def test_b2_expired_claim_reclaimed_then_retried_when_allowed(tmp_path):
    ident = _ident()
    a = _runner(tmp_path, "A", Clock())
    try:
        a.claim(ident)
    finally:
        _restore(a)
    world = F.World()
    b = _runner(tmp_path, "B", Clock(T0 + 2 * a.wall_s + 1), world, infra_retries=1)
    assert F.run_rows(b, [ident]) == 0
    assert [json.loads(p.read_text())["attempt"] for p in _claims(tmp_path)] == [1, 2]
    assert len(world.envs) == 1
    starts = [r for r in F.seat_ledger(tmp_path / "stage", LABEL, seat="B") if r["kind"] == "attempt_start"]
    assert [(r["attempt_no"], r["retry"]) for r in starts] == [(2, True)]


def test_c_same_seat_stale_claim_reclaimed_immediately(tmp_path, capsys):
    ident = _ident()
    clock = Clock()
    q = F.env_client().DynamicQueue(F.queue_dir(tmp_path / "stage", LABEL), seat="A", wall_s=60, now=clock)
    p = q.create_claim(ident, 1)
    doc = json.loads(p.read_text())
    doc["pid"] = 999_999_999  # 同席位名的旧进程（lease 保证同名席位只有一个进程）
    p.write_text(json.dumps(doc))
    world = F.World()
    r = _runner(tmp_path, "A", clock, world, infra_retries=1)  # 时钟未动：心跳完全新鲜
    assert F.run_rows(r, [ident]) == 0
    assert json.loads(p.read_text())["reclaim_reason"] == "infra" and len(world.envs) == 1
    assert "reason=stale_same_seat" in capsys.readouterr().out


def test_d_concurrent_claims_only_one_wins(tmp_path):
    ident = _ident()
    clock = Clock()
    runners = [_runner(tmp_path, f"S{i}", clock) for i in range(6)]
    try:
        barrier = threading.Barrier(len(runners))
        got = []

        def go(r):
            barrier.wait()
            got.append(r.claim(ident))

        ts = [threading.Thread(target=go, args=(r,)) for r in runners]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        _restore(runners[0])
    wins = [g for g in got if not isinstance(g, str)]
    assert len(wins) == 1 and sorted(g for g in got if isinstance(g, str)) == ["live"] * 5
    assert len(_claims(tmp_path)) == 1
    # O_EXCL 本身：同一尝试号第二次创建失败
    q = runners[0].queue
    assert q.create_claim(ident, 1) is None


def test_e_hard_exit_settles_before_exit_75(tmp_path, capsys):
    ident = _ident()
    book = tmp_path / "budget.jsonl"
    seen = {}
    holder = {}

    def fake_exit(code):  # 退出那一刻核对：attempt_end、commit、领取收尾都已落盘
        r, c = holder["runner"], holder["claim"]
        led = F.seat_ledger(tmp_path / "stage", LABEL, seat="A")
        seen.update(code=code, ends=[x for x in led if x["kind"] == "attempt_end"], open=_open_rids(book),
                    claim=json.loads(c.path.read_text()))

    r = _runner(tmp_path, "A", Clock(), budget_ledger=str(book), hard_exit=fake_exit)
    try:
        c = r.claim(ident)
        holder.update(runner=r, claim=c)
        r.ledger.attempt_start(key=c.key, attempt_id="aid1", attempt_no=1, retry=False, identity=ident,
                               budget_rid=c.rid, claim=str(c.path))
        r._current, r._current_attempt_id = c, "aid1"
        assert _open_rids(book) == {c.rid}
        r._hard_exit(75)
    finally:
        _restore(r)
    assert seen["code"] == 75
    (end,) = seen["ends"]
    assert end["attempt_id"] == "aid1" and end["infra"] is True and end["infra_reason"] == "watchdog_exit"
    assert seen["open"] == set()  # 共享账本无悬空 rid
    assert seen["claim"]["ended"] is True and seen["claim"]["end_status"] == "infra_timeout"
    assert r._current is None
    assert "SEAT_WATCHDOG_EXIT" in capsys.readouterr().out
    # 重启后 recover_dangling 不再把它当悬空（已有 attempt_end）
    r2 = _runner(tmp_path, "A", Clock(), budget_ledger=str(book))
    try:
        assert r2.recover_dangling() == 0
    finally:
        _restore(r2)


def test_f_heartbeat_never_revives_ended_or_voided_claim(tmp_path):
    ident = _ident()
    clock = Clock()
    r = _runner(tmp_path, "A", clock)
    try:
        c = r.claim(ident)
        r._current = c
        clock.t += 5
        r.heartbeat_once()
        assert json.loads(c.path.read_text())["t_heartbeat"] == T0 + 5
        # 结算持锁：心跳线程等锁，拿到锁时当前领取已清空，不写
        with r._lock:
            t = threading.Thread(target=r.heartbeat_once)
            t.start()
            time.sleep(0.2)
            assert t.is_alive()  # 心跳被锁挡住
            r.queue.end_claim(c.path, "success")
            r._current = None
        t.join(5)
        doc = json.loads(c.path.read_text())
        assert doc["ended"] is True and doc["t_heartbeat"] == T0 + 5
        # 即便还挂着当前领取：已收尾的不写、已作废（删除）的不重建
        r._current = c
        clock.t += 5
        r.heartbeat_once()
        assert json.loads(c.path.read_text())["t_heartbeat"] == T0 + 5
        r.queue.void_claim(c.path)
        r.heartbeat_once()
        assert not c.path.exists()
    finally:
        _restore(r)
