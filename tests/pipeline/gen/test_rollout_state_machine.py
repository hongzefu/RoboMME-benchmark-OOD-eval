"""C12 第二阶段状态机（``hard-specs/4``）：真实 ``_rollout.run_continue`` + 真实 ``run_batch``，runner 由 FakeRunner 冒充。

每个用例的期望都是手写的事件序列：哪一局在哪一轮执行、什么结局、递补到哪个候选、最后每行的
``selected／tried／rollout.status``；不从被测函数推出。覆盖：失败递补、基础设施重试（上限 1 次）、执行步上限、
崩溃恢复不重复派发（M18）、恢复歧义、超配额、回写身份变化、锁、MoveCube 同方式递补。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gen_world import H, R, Crash, FakeRunner, continue_kwargs, freeze_file, install_run, read_rows, state

S = "StopCube"


@pytest.fixture
def specs(tmp_path):
    """StopCube xhard1：配额 2、候选 4（全部一次 reset 成功）。"""
    path = tmp_path / "frozen" / "xhard1" / "specs.jsonl"
    freeze_file(path, "xhard1", {S: (2, 4)})
    return path


def _run(specs: Path, out: Path, **kw):
    return R.run_continue(specs, out, ledger=out / R.LEDGER_NAME, **continue_kwargs(out.parent), **kw)


def _ledger(out: Path):
    return [(e["kind"], e["task"], e["candidate"], e["round"]) for e in R.read_ledger(out / R.LEDGER_NAME)]


def test_failure_is_backfilled_with_next_untried_candidate(monkeypatch, specs, tmp_path):
    frozen_header, _ = read_rows(specs)
    fake = FakeRunner({(S, 0): ["fail"]}).install(monkeypatch)
    out = tmp_path / "out"
    summary = _run(specs, out)
    assert fake.executions == [(S, 0, "fail"), (S, 1, "ok"), (S, 2, "ok")]
    assert state(specs) == {(S, 0): (False, True, "failed"), (S, 1): (True, True, "ok"),
                            (S, 2): (True, True, "ok"), (S, 3): (False, False, None)}
    assert _ledger(out) == [("result", S, 0, 0), ("result", S, 1, 0), ("result", S, 2, 1)]
    assert summary["attempted"] == 3 and summary["rounds"] == 2 and summary["infra_retries"] == 0
    assert summary["delivered"] == 2
    header, rows = H.load_specs(specs, check_fingerprint=False)
    assert header["identity_sha256"] == frozen_header["identity_sha256"]  # 签不变
    assert header["delivery_sha256"] != frozen_header["delivery_sha256"]
    by = {r["candidate"]: r["rollout"] for r in rows}
    assert by[2]["role"] == "backfill" and by[1]["role"] == "selected" and by[0]["error_type"] == "TaskFailed"
    h5 = Path(by[2]["h5_path"])
    assert h5.parent == out / "episodes" / "xhard1" / f"{S}_episode_2" / "hdf5_files"
    assert by[2]["h5_sha256"] == hashlib.sha256(h5.read_bytes()).hexdigest()
    assert (by[2]["frames"], by[2]["exec_steps"]) == (5, 4)  # 5 帧、1 帧演示 → 执行 4 步
    assert not Path(str(specs) + ".lock").exists()
    assert json.loads((out / "summary.json").read_text()) == summary


def test_infra_failure_retried_once_then_succeeds(monkeypatch, specs, tmp_path):
    fake = FakeRunner({(S, 0): ["infra", "ok"]}).install(monkeypatch)
    out = tmp_path / "out"
    summary = _run(specs, out)
    assert fake.executions == [(S, 0, "infra"), (S, 1, "ok"), (S, 0, "ok")]
    assert state(specs)[(S, 0)] == (True, True, "ok") and state(specs)[(S, 2)] == (False, False, None)
    assert _ledger(out) == [("infra_retry", S, 0, 0), ("result", S, 1, 0), ("result", S, 0, 1)]
    assert summary["attempted"] == 3 and summary["infra_retries"] == 1


def test_second_infra_failure_is_final_and_backfilled(monkeypatch, specs, tmp_path):
    # 第一次是错误文本命中 svulkan2 的基础设施失败（按文本归类），第二次超过每身份 1 次重试上限 → 记失败并递补
    fake = FakeRunner({(S, 0): ["vulkan", "infra"]}).install(monkeypatch)
    out = tmp_path / "out"
    _run(specs, out)
    assert fake.executions == [(S, 0, "vulkan"), (S, 1, "ok"), (S, 0, "infra"), (S, 2, "ok")]
    _, rows = read_rows(specs)
    assert rows[0]["rollout"]["status"] == "failed" and rows[0]["rollout"]["error_type"] == "BrokenProcessPool"
    assert state(specs)[(S, 2)] == (True, True, "ok")


def test_missing_result_counts_as_runner_crash(monkeypatch, specs, tmp_path):
    fake = FakeRunner({(S, 1): ["none"]}).install(monkeypatch)
    out = tmp_path / "out"
    summary = _run(specs, out)
    assert fake.executions == [(S, 0, "ok"), (S, 1, "none"), (S, 1, "ok")]
    assert _ledger(out) == [("infra_retry", S, 1, 0), ("result", S, 0, 0), ("result", S, 1, 1)]
    assert R.read_ledger(out / R.LEDGER_NAME)[0]["error_type"] == "RunnerCrash"
    assert summary["infra_retries"] == 1


def test_task_failure_is_not_retried():
    # 判定器负例：普通任务失败不是基础设施失败；ok 结果永远不是
    assert not R.is_infra({"ok": False, "error_type": "TaskFailed", "error": "plan failed"})
    assert not R.is_infra({"ok": True, "error_type": "BrokenProcessPool"})
    assert R.is_infra({"ok": False, "error_type": "ValueError", "error": "CUDA error: device lost"})


# ── 执行步上限 ──────────────────────────────────────────────────────


def test_exec_cap_boundary_through_real_h5(monkeypatch, tmp_path):
    # 候选 0：执行步 = 上限 + 1（演示 5 帧不计）→ 超限失败、删媒体、递补；候选 1：执行步恰为上限 → 交付
    specs = tmp_path / "frozen" / "xhard1" / "specs.jsonl"
    freeze_file(specs, "xhard1", {S: (1, 3)})
    cap = H.EXEC_CAP
    fake = FakeRunner({(S, 0): [("ok", cap + 6, 5)], (S, 1): [("ok", cap + 5, 5)]}).install(monkeypatch)
    out = tmp_path / "out"
    _run(specs, out)
    assert fake.executions == [(S, 0, "ok"), (S, 1, "ok")]
    _, rows = read_rows(specs)
    r0, r1 = rows[0]["rollout"], rows[1]["rollout"]
    assert (r0["status"], r0["error_type"], r0["exec_steps"], r0["purged_files"]) == ("failed", "exec_over_cap", cap + 1, 2)
    assert (r1["status"], r1["exec_steps"], r1["frames"]) == ("ok", cap, cap + 5)
    ep0 = out / "episodes" / "xhard1" / f"{S}_episode_0"
    assert sorted(p.name for p in ep0.rglob("*") if p.is_file()) == ["spec_replay.json"]  # h5、mp4 已删，其余留证
    assert list((out / "episodes" / "xhard1" / f"{S}_episode_1").rglob("*.h5"))
    entry = R.read_ledger(out / R.LEDGER_NAME)[0]
    assert entry["record"]["error_type"] == "exec_over_cap"


@pytest.mark.parametrize("record, kind", [
    ({"ok": True, "h5": None, "exec_steps": None}, "h5_missing"),
    ({"ok": True, "h5": "/x.h5", "exec_steps": None}, "exec_steps_missing"),
])
def test_ok_without_h5_facts_is_not_delivered(record, kind, tmp_path):
    rec = dict(record)
    R.enforce_exec_cap(rec, {"tier": "xhard1", "task": S, "episode": 0}, tmp_path, 3)
    assert rec["ok"] is False and rec["error_type"] == kind


def test_exec_cap_unit_boundary(tmp_path):
    row = {"tier": "xhard1", "task": S, "episode": 0}
    at = {"ok": True, "h5": "/x.h5", "exec_steps": 3}
    R.enforce_exec_cap(at, row, tmp_path, 3, purge=False)
    assert at["ok"] is True
    over = {"ok": True, "h5": "/x.h5", "exec_steps": 4}
    R.enforce_exec_cap(over, row, tmp_path, 3, purge=False)
    assert over["error_type"] == "exec_over_cap" and over["purge_pending"] is True


def test_purge_refuses_h5_outside_episode_dir(tmp_path):
    outside = tmp_path / "elsewhere" / "a.h5"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    with pytest.raises(R.RolloutError, match="不在该局目录内"):
        R.purge_episode_media(tmp_path / "out", {"tier": "xhard1", "task": S, "episode": 0}, str(outside))
    assert outside.exists()


# ── 崩溃恢复（M18）与恢复歧义 ─────────────────────────────────────────


def test_crash_between_rounds_resumes_without_redispatch(monkeypatch, specs, tmp_path):
    out = tmp_path / "out"
    before = specs.read_bytes()
    first = FakeRunner({(S, 0): ["fail"]}, crash_on_call=2).install(monkeypatch)
    with pytest.raises(Crash):
        _run(specs, out)
    assert first.executions == [(S, 0, "fail"), (S, 1, "ok")]
    assert specs.read_bytes() == before  # 回写前中断：规格文件逐字节不变
    assert not Path(str(specs) + ".lock").exists()
    assert _ledger(out) == [("result", S, 0, 0), ("result", S, 1, 0)]
    with pytest.raises(R.RolloutError, match="已有运行痕迹"):
        _run(specs, out)
    # 续跑：新替身对候选 0 会返回 ok——若重复派发已记账的身份，交付集合就会被改写
    second = FakeRunner().install(monkeypatch)
    summary = _run(specs, out, resume=True)
    assert second.executions == [(S, 2, "ok")]
    assert state(specs) == {(S, 0): (False, True, "failed"), (S, 1): (True, True, "ok"),
                            (S, 2): (True, True, "ok"), (S, 3): (False, False, None)}
    assert _ledger(out) == [("result", S, 0, 0), ("result", S, 1, 0), ("result", S, 2, 1)]
    assert summary["attempted"] == 3  # 累计预算 = 两次进程实际执行之和，不重复计
    assert len(first.executions) + len(second.executions) == 3


def test_crash_inside_round_resumes_from_partial(monkeypatch, specs, tmp_path):
    out = tmp_path / "out"
    first = FakeRunner(crash_on_call=1, crash_after_jobs=1).install(monkeypatch)
    with pytest.raises(Crash):
        _run(specs, out)
    assert first.executions == [(S, 0, "ok")] and _ledger(out) == []
    second = FakeRunner().install(monkeypatch)
    _run(specs, out, resume=True)
    assert second.executions == [(S, 1, "ok")]  # 候选 0 已在 partial 里，不再执行
    assert "--resume" in second.calls[0]
    assert state(specs)[(S, 0)] == (True, True, "ok") and state(specs)[(S, 1)] == (True, True, "ok")


def test_h5_without_partial_record_is_ambiguous(monkeypatch, specs, tmp_path):
    out = tmp_path / "out"
    FakeRunner(crash_on_call=1, crash_after_jobs=1).install(monkeypatch)
    with pytest.raises(Crash):
        _run(specs, out)
    (out / "_rounds" / "xhard1_round_00" / "results.partial.jsonl").unlink()  # h5 已写、partial 未落的窗口
    before = specs.read_bytes()
    fake = FakeRunner().install(monkeypatch)
    with pytest.raises(R.RolloutError, match=r"恢复歧义.*xhard1/StopCube/0"):
        _run(specs, out, resume=True)
    assert fake.executions == [] and specs.read_bytes() == before
    assert not Path(str(specs) + ".lock").exists()


def test_aside_dirs_do_not_count_as_ambiguous(tmp_path):
    aside = tmp_path / "episodes" / "xhard1" / f"{S}_episode_0.aside123" / "hdf5_files"
    aside.mkdir(parents=True)
    (aside / "a.h5").write_bytes(b"x")
    assert R.unknown_identities(tmp_path) == []
    live = tmp_path / "episodes" / "xhard1" / f"{S}_episode_1" / "hdf5_files"
    live.mkdir(parents=True)
    (live / "b.h5").write_bytes(b"x")
    assert R.unknown_identities(tmp_path) == ["xhard1/StopCube/1"]


# ── 配额、回写、锁 ──────────────────────────────────────────────────


def test_plan_pending_rejects_over_quota(specs):
    _, rows = read_rows(specs)
    assert [r["candidate"] for r in R.plan_pending(rows, {S: 2})] == [0, 1]
    rows[2]["selected"] = True
    with pytest.raises(R.RolloutError, match="超过 delivery_per_cell"):
        R.plan_pending(rows, {S: 2})


def test_plan_pending_redo_resets_rollout(specs):
    _, rows = read_rows(specs)
    rows[3]["rollout"], rows[3]["selected"] = {"status": "failed"}, False
    assert [r["candidate"] for r in R.plan_pending(rows, {S: 3}, redo={(S, 3)})] == [0, 1, 3]
    assert rows[3]["rollout"] is None and rows[3]["selected"] is True


def test_write_back_refuses_identity_change(specs):
    header, rows = read_rows(specs)
    sha = R.file_sha256(specs)
    rows[0]["seed"] += 1
    with pytest.raises(R.RolloutError, match="identity_sha256 变了"):
        R.write_back(specs, header, rows, sha)
    assert R.file_sha256(specs) == sha


def test_write_back_refuses_concurrent_edit(monkeypatch, specs, tmp_path):
    fake = FakeRunner()

    def editing_run(argv, **kw):
        result = fake.run(argv, **kw)
        with specs.open("a") as fh:  # 运行期间有人改了规格文件
            fh.write("\n")
        return result

    install_run(monkeypatch, editing_run)
    with pytest.raises(R.RolloutError, match="被他人改过"):
        _run(specs, tmp_path / "out")
    assert specs.read_bytes().endswith(b"\n\n")  # 未被回写覆盖


def test_existing_lock_refuses_before_any_work(monkeypatch, specs, tmp_path):
    lock = Path(str(specs) + ".lock")
    lock.write_text('{"pid": 1}')
    fake = FakeRunner().install(monkeypatch)
    with pytest.raises(R.RolloutError, match="锁已存在"):
        _run(specs, tmp_path / "out")
    assert fake.calls == [] and lock.read_text() == '{"pid": 1}'
    assert not (tmp_path / "out").exists()


# ── MoveCube 同方式递补 ──────────────────────────────────────────────


def test_movecube_backfills_only_within_same_way(monkeypatch, tmp_path):
    # 候选 0..5 的运动方式 = 编号 % 3；配额 3（非 V9 的 50，走一方式一个的分层）→ 初选 0、1、2
    specs = tmp_path / "frozen" / "xhard4" / "specs.jsonl"
    freeze_file(specs, "xhard4", {"MoveCube": (3, 6)}, ways={"MoveCube": lambda e: e % 3})
    fake = FakeRunner({("MoveCube", 1): ["fail"], ("MoveCube", 4): ["fail"]}).install(monkeypatch)
    out = tmp_path / "out"
    _run(specs, out)
    # way1 的候选 1 失败 → 递补同方式的 4（不是编号最小的 3）；4 也失败 → 已无 way1 备用，不跨方式递补
    assert fake.executions == [("MoveCube", 0, "ok"), ("MoveCube", 1, "fail"), ("MoveCube", 2, "ok"),
                               ("MoveCube", 4, "fail")]
    assert state(specs) == {("MoveCube", 0): (True, True, "ok"), ("MoveCube", 1): (False, True, "failed"),
                            ("MoveCube", 2): (True, True, "ok"), ("MoveCube", 3): (False, False, None),
                            ("MoveCube", 4): (False, True, "failed"), ("MoveCube", 5): (False, False, None)}


def test_same_way_rule_only_for_v4_new_tiers():
    assert R.same_way_tasks_for({"schema": H.SCHEMA, "difficulty": "xhard4", "tasks": ["MoveCube", "InsertPeg"]}) \
        == frozenset({"MoveCube"})
    assert R.same_way_tasks_for({"schema": "hard-specs/3", "difficulty": "xhard4", "tasks": ["MoveCube"]}) == frozenset()
    assert R.same_way_tasks_for({"schema": H.SCHEMA, "difficulty": "xhard4", "tasks": ["InsertPeg"]}) == frozenset()
