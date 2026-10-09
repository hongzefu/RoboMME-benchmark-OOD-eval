"""第三阶段跨块反例（R4 查缺补漏）：真实路线产物喂真实检查器、换种子不误用别组结果、预算上限三入口一致。

依据：``1006-rename-official-names-and-stage3-eval-plan.md`` 第二部分二节 R4 行（真实触发缺 seed、少 state、篡改数组字节、
坏账本、重复启动、错 attempt 发布、语言账本悬空调用与图引用哈希对不上、恢复请求挤占首试额度）、三节「闸门与 CPU
runbook」第二段（cap 在 reserve／retry／report 三入口一致读取、到期接续计入恢复合计、第三次 attempt 拒绝）、八.3（每个
``(模型, policy_seed)`` 独立目录与账本路线，更换种子不误用其他组的 accepted 结果）、八.11（``LANG_IO``）；接口
``docs/plans/1006-stage3-interface-freeze.md`` 三、四、五节。

已有用例覆盖的反例（见交回报告的覆盖审计表）不在此重复；这里只补：

1. 各块的检查器（``trace_arrays_check``、``lang_io_check``）此前只喂手写夹具。本文件让六条席位路线的**真实客户端**
   经 GL 席位 ``SeatRunner``（常驻 Policy）产出 trace／arrays.npz／language.jsonl（另加 3-tier Astra 一局，经
   ``AstraPolicy``），不做任何规范化直接判 PASS，再在
   副本上逐个注入：篡改一个动作字节、给缺观测步补零、删一个观测步的状态、改一个附图引用哈希、删一个调用的关闭行，
   各自被对应计数拦下。
2. 换种子：同一身份在种子 7 已 accepted，种子 42 用独立目录与同一共享预算账本时照常重新跑（账本 token 路线各带种子）；
   种子 42 误用种子 7 的尝试账本时拒跑（``RUN_BLOCKED reason=route_mismatch``、退出 3、不写新行）；全新 stage 与
   同路线续跑不误拒；缺 ``route`` 的历史账本行按 ``legacy`` 拒。
3. 预算上限：同一份 config 下 reserve（首试／恢复）、claim_retry（infra／expired）、report 三个入口读到同一组上限；
   到期接续与 infra 重试合计受 ``trajectory_cap − planned_first_tries`` 约束；同一身份第二次领重试（第三次 attempt）被拒；
   构造参数与账本 config 不一致时三个入口一致拒绝。

R4 当初钉住的两条缺陷均已在 1006 MERGE-2 修复，本文件改为断言修复后的行为：

* D1 ``lang_io_check`` 把执行段首帧（reset 初始画面，记在 demo 行末帧）的 ``phase=exec, frame_idx=0`` 图引用判为悬空
  ——由 FIX-1（按 ``(phase, cam, frame_idx)`` 精确比对、exec 帧 0 映射 demo 末帧）修复；
* D2 ``env_client`` 尝试账本被另一种子复用时把别组 accepted 当作已完成跳过——由 ``SeatRunner._check_route`` 修复；
  拆仓后席位按 ``seed<n>`` 分目录，复用只可能来自手工混入，路线核对照旧拦。
"""
from __future__ import annotations

import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

import eval_fakes_dev as F
import test_stage3_seven_routes as R
from tests._support.dev_loaders import load_script
from tests.pipeline.evalx.groundsg import groundsg_fakes as G

SEAT_ROUTES = R.SEAT_ROUTES


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (*R.GROUNDSG_ENV, "SGEVAL_BUDGET_LEDGER", "SGEVAL_EXPIRED_JOBS", "SLURM_JOB_ID", "SLURM_JOB_END_TIME",
              "SGEVAL_AUDIT", "POLICY_SEED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SGEVAL_PP_SERVER_WRAP", "1")
    return monkeypatch


def _checkers():
    return load_script("eval-official/trace_arrays_check.py"), load_script("eval-official/lang_io_check.py")


def _verdict(capsys, fn, argv) -> tuple[int, dict, str]:
    capsys.readouterr()
    rc = fn(argv)
    out = capsys.readouterr().out
    line = out.strip().splitlines()[-1]
    head, *rest = line.split()
    return rc, {"": head.split("=", 1)[1], **dict(x.split("=", 1) for x in rest)}, out


# ═════════════════════════════════ 1 真实路线产物 → 检查器 ═════════════════════════════════


def _run_routes(tmp: Path, monkeypatch) -> tuple[Path, dict]:
    """六条路线各跑一局成功（20 步）；PP 与 GroundSG Oracle 另各跑一局第 5 步环境异常（缺观测步）。产物收进同一根。"""
    root = tmp / "traces"
    plans = [(route, G.Plan(success_at=20), "ok") for route in SEAT_ROUTES]
    plans += [("pp", G.Plan(raise_at=5), "raise"), ("groundsg-oracle", G.Plan(raise_at=5), "raise")]
    where: dict = {}
    for route, plan, tag in plans:
        sub = tmp / f"{route}-{tag}"
        world = R.World(plan)
        with monkeypatch.context() as mp:
            runner = R.seat_runner(route, sub, world, mp, seed=7, max_steps=R.CAP, spy={})
            ident = R.ood_identity()
            assert F.run_rows(runner, [ident]) == 0
        (row,) = F.read_jsonl(runner.results_path)
        # 拆仓后 error 并入 fail、原终态记 error_kind（已定口径第 5 条）
        want = ("success", None) if tag == "ok" else ("fail", "error")
        assert (row["status"], row.get("error_kind")) == want, (route, tag, row.get("error"))
        src = R.raw_dir(runner, ident)  # 本局 raw 目录：trace.jsonl、arrays.npz、language.jsonl 都在这里
        dst = root / route / tag / f"{ident['key']}.a1"
        shutil.copytree(src, dst)
        where[(route, tag)] = dst
    # 3-tier Astra：零外联全替身 AstraPolicy 跑一局（hard-verify BinFill，20 步成功），本局 raw 目录同样收进根
    with monkeypatch.context() as mp:
        h, r, net = R.astra_run(tmp / "astra", mp, seed=7, dataset="hard-verify", task="BinFill")
        assert r.status == "success" and net.calls == 0
    dst = root / "astra" / "ok" / f"{r.key}.a1"
    shutil.copytree(h.raw(r), dst)
    where[("astra", "ok")] = dst
    return root, where


@pytest.mark.slow
def test_real_route_outputs_pass_checkers_and_injected_faults_fail(tmp_path, monkeypatch, capsys):
    ta, li = _checkers()
    root, where = _run_routes(tmp_path, monkeypatch)
    n = len(where)
    rc, v, out = _verdict(capsys, ta.main, [str(root)])
    assert rc == 0 and v[""] == "PASS" and v["episodes"] == str(n), out
    # LANG_IO：R4-D1 已由 FIX-1 修复（图引用按 (phase, cam, frame_idx) 精确比对、exec 帧 0 映射 demo 末帧），
    # 真实九局产物不做任何规范化直接 PASS
    rc, v, out = _verdict(capsys, li.main, ["--root", str(root)])
    assert rc == 0 and v[""] == "PASS" and v["episodes"] == str(n), out
    assert v["image_ref_unresolved"] == "0" and set(x for k, x in v.items() if k not in ("", "episodes")) == {"0"}, out
    # 缺观测步不补零：两条路线的异常步都没有状态键、end 行如实列出
    for route in ("pp", "groundsg-oracle"):
        d = where[(route, "raise")]
        end = F.read_jsonl(d / "trace.jsonl")[-1]
        assert end["arrays"]["missing_state_steps"] == [5], (route, end["arrays"])
        with np.load(d / "arrays.npz") as z:
            assert "exec_action__00004" in z.files and "exec_state__00004" not in z.files, (route, sorted(z.files))

    def fresh(name: str) -> tuple[Path, dict]:
        r2 = tmp_path / name
        shutil.copytree(root, r2)
        return r2, {k: r2 / v.relative_to(root) for k, v in where.items()}

    def rewrite_npz(path: Path, edit) -> None:
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        edit(d)
        np.savez(path, **d)

    hits = {}
    # ① 篡改一个动作字节（GroundSG MemER 局）→ tampered=1
    r2, w2 = fresh("tamper-action")

    def flip(d):
        a = d["exec_action__00002"].copy()
        a.view(np.uint8)[0] ^= 1
        d["exec_action__00002"] = a
    rewrite_npz(w2[("groundsg-memer", "ok")] / "arrays.npz", flip)
    rc, v, _ = _verdict(capsys, ta.main, [str(r2)])
    assert rc == 1 and v["tampered"] == "1" and v["attempted_steps_missing"] == "0", v
    hits["tampered_action"] = 1
    # ② 给缺观测步补零（PP 异常局第 5 步）→ tampered=1（不补零）
    r3, w3 = fresh("zero-fill")

    def zero_fill(d):
        d["exec_state__00004"] = np.zeros_like(d["exec_state__00003"])
    rewrite_npz(w3[("pp", "raise")] / "arrays.npz", zero_fill)
    rc, v, _ = _verdict(capsys, ta.main, [str(r3)])
    assert rc == 1 and v["tampered"] == "1", v
    hits["zero_filled_state"] = 1
    # ③ 少 state：删掉一个观测步的状态（SimpleMemVLA 局第 3 步）→ observed_state_missing=1
    r4, w4 = fresh("drop-state")
    rewrite_npz(w4[("smvla", "ok")] / "arrays.npz", lambda d: d.pop("exec_state__00002"))
    rc, v, _ = _verdict(capsys, ta.main, [str(r4)])
    assert rc == 1 and v["observed_state_missing"] == "1", v
    hits["observed_state_missing"] = 1
    # ④ 图引用哈希对不上：改 QwenVL 局第一个带图消息的一个 raw_sha256 → image_ref_unresolved≥1
    r5, w5 = fresh("image-ref")
    lp = w5[("groundsg-qwenvl", "ok")] / "language.jsonl"
    rows = F.read_jsonl(lp)
    i = next(k for k, r in enumerate(rows) if r.get("kind") == "message" and r.get("images"))
    img = rows[i]["images"][0]
    img["raw_sha256"] = ("0" if img["raw_sha256"][0] != "0" else "1") + img["raw_sha256"][1:]
    lp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    rc, v, _ = _verdict(capsys, li.main, ["--root", str(r5)])
    assert rc == 1 and int(v["image_ref_unresolved"]) >= 1 and v["open_calls"] == "0", v
    hits["image_ref_unresolved"] = int(v["image_ref_unresolved"])
    # ⑤ 悬空调用：删掉 FrameSamp+Modulation 局最后一个调用的关闭行 → open_calls=1
    r6, w6 = fresh("dangling")
    lp = w6[("perceptual-framesamp-modul", "ok")] / "language.jsonl"
    rows = F.read_jsonl(lp)
    j = max(k for k, r in enumerate(rows) if r.get("kind") == "call_close")
    lp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for k, r in enumerate(rows) if k != j),
                  encoding="utf-8")
    rc, v, _ = _verdict(capsys, li.main, ["--root", str(r6)])
    assert rc == 1 and v["open_calls"] == "1", v
    hits["open_calls"] = 1
    capsys.readouterr()
    print(f"CROSS_BLOCK_CHECKERS=PASS routes={len(SEAT_ROUTES) + 1} episodes={n} trace_arrays=PASS "
          f"lang_io=PASS " + " ".join(f"{k}={x}" for k, x in hits.items()))


# ═════════════════════════════════ 2 换种子不误用别组 accepted ═════════════════════════════════


def _one_step_policy(calls: list):
    def run_episode(session, identity, conn_info, recorder):
        calls.append(conn_info["policy_seed"])
        session.reset()
        session.step([0.0] * 8)
        return {"status": "fail", "steps": 1, "infra": False, "error": None}
    return types.SimpleNamespace(run_episode=run_episode)


def _seed_runner(stage: Path, seed: int, calls: list, ledger: Path, world, **kw):
    """拆仓后的席位：同一 stage 下按 ``(模型, policy_seed)`` 分目录（``queue/pp/seed<n>/``、``seats/pp/seed<n>/<席位>/``），
    共享预算账本路线 ``pp/seed<n>/new``。"""
    return F.make_runner(stage, "pp", _one_step_policy(calls), world, policy_seed=seed, budget_ledger=str(ledger),
                         **R.CAPS, reset_cap=141430, **kw)


def test_seed_switch_same_stage_reruns_and_routes_stay_separate(tmp_path):
    """种子 7 先跑完（accepted）；种子 42 用同一 stage、同一共享预算账本：按种子分目录，身份照常重跑（不误用种子 7 的
    accepted），账本 route／token 各带自己的种子，两组结果的 policy_seed 各是各的。"""
    ledger = tmp_path / "budget.jsonl"
    world = F.World()
    ident = F.hard0_identity("PickXtimes", 0)
    stage = tmp_path / "stage"
    calls: list = []
    assert F.run_rows(_seed_runner(stage, 7, calls, ledger, world), [ident]) == 0
    assert F.run_rows(_seed_runner(stage, 42, calls, ledger, world), [ident]) == 0
    assert calls == [7, 42] and len(world.envs) == 2
    (row7,) = F.seat_results(stage, "pp", seed=7)
    (row42,) = F.seat_results(stage, "pp", seed=42)
    assert (row7["policy_seed"], row42["policy_seed"]) == (7, 42)
    assert row7["budget_token"] == f"pp/seed7/new|{ident['key']}|a1"
    assert row42["budget_token"] == f"pp/seed42/new|{ident['key']}|a1"
    reserves = [(r["route"], r["kind_of_try"]) for r in F.read_jsonl(ledger) if r["kind"] == "reserve"]
    assert reserves == [("pp/seed7/new", "first"), ("pp/seed42/new", "first")]
    for seed in (7, 42):
        assert (F.queue_dir(stage, "pp", seed=seed) / "accepted" / f"hard-verify__{ident['key']}.json").is_file()


def _seat_ledger_path(stage: Path, seed: int) -> Path:
    return F.seat_dir(stage, "pp", seed=seed) / "pp.ledger.jsonl"


def test_seat_ledger_with_other_seed_route_is_blocked(tmp_path, capsys):
    """路线核对仍在：本席位尝试账本里混进别的种子的 ``attempt_start``（route 含 ``seed7``）时，种子 42 的席位在取得
    lease 后、领局之前打印 ``RUN_BLOCKED reason=route_mismatch … found=pp/seed7/new want=pp/seed42/new``、退出 3，不跑、
    不写任何结果行与预算行。"""
    ledger = tmp_path / "budget.jsonl"
    world = F.World()
    ident = F.hard0_identity("PickXtimes", 0)
    stage = tmp_path / "stage"
    lp = _seat_ledger_path(stage, 42)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({"t": 1.0, "seat": "s00", "policy": "pp", "kind": "attempt_start", "key": ident["key"],
                              "attempt_id": "x", "attempt_no": 1, "retry": False, "route": "pp/seed7/new"}) + "\n",
                  encoding="utf-8")
    calls: list = []
    r42 = _seed_runner(stage, 42, calls, ledger, world)
    budget_before = F.read_jsonl(ledger)
    capsys.readouterr()
    rc = F.run_rows(r42, [ident])
    out = capsys.readouterr().out
    assert rc == 3 and calls == [], (rc, calls, out)
    line = next(x for x in out.splitlines() if x.startswith("RUN_BLOCKED"))
    assert "reason=route_mismatch" in line and f"ledger={lp}" in line, line
    assert "found=pp/seed7/new" in line and "want=pp/seed42/new" in line, line
    assert "RUN_PLAN" not in out, out
    assert F.seat_results(stage, "pp", seed=42) == []
    assert [x["kind"] for x in F.read_jsonl(lp) if x["kind"] not in ("budget", "budget_raise")] == ["attempt_start"]
    assert [r for r in F.read_jsonl(ledger) if r["kind"] != "config"] == \
        [r for r in budget_before if r["kind"] != "config"]
    print("SEED_SWITCH_ROUTE_GUARD=PASS rc=3 reason=route_mismatch new_result_rows=0 new_attempt_rows=0")


def test_route_guard_allows_fresh_stage_and_same_route_resume(tmp_path, capsys):
    """路线核对不误拒：全新 stage（空账本）首跑照常；同一路线续跑（同种子、同账本）照常，已 accepted 的跳过；第二个身份
    在续跑中照常开跑。"""
    ledger = tmp_path / "budget.jsonl"
    world = F.World()
    a, b = F.hard0_identity("PickXtimes", 0), F.hard0_identity("PickXtimes", 1)
    stage = tmp_path / "stage"
    calls: list = []
    assert F.run_rows(_seed_runner(stage, 7, calls, ledger, world), [a]) == 0
    capsys.readouterr()
    assert F.run_rows(_seed_runner(stage, 7, calls, ledger, world), [a, b]) == 0
    out = capsys.readouterr().out
    assert "RUN_BLOCKED" not in out and "total=2 claimable=1" in out, out
    assert calls == [7, 7]
    routes = {r.get("route") for r in F.read_jsonl(_seat_ledger_path(stage, 7)) if r.get("kind") == "attempt_start"}
    assert routes == {"pp/seed7/new"}
    print("ROUTE_GUARD_NO_FALSE_BLOCK=PASS fresh_stage=run same_route_resume=run")


def test_route_guard_rejects_legacy_ledger_rows(tmp_path, capsys):
    """历史账本里没有 ``route`` 字段的 ``attempt_start`` 行视为 ``legacy``，同样拒跑（``found=legacy``）。"""
    ledger = tmp_path / "budget.jsonl"
    world = F.World()
    ident = F.hard0_identity("PickXtimes", 0)
    stage = tmp_path / "stage"
    lp = _seat_ledger_path(stage, 7)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({"t": 1.0, "seat": "s00", "policy": "pp", "kind": "attempt_start", "key": "old",
                              "attempt_id": "old-a1", "attempt_no": 1, "retry": False}) + "\n", encoding="utf-8")
    calls: list = []
    capsys.readouterr()
    rc = F.run_rows(_seed_runner(stage, 7, calls, ledger, world), [ident])
    out = capsys.readouterr().out
    assert rc == 3 and calls == [] and "reason=route_mismatch" in out and "found=legacy" in out, out
    assert "want=pp/seed7/new" in out, out
    assert F.seat_results(stage, "pp") == []
    assert [x["kind"] for x in F.read_jsonl(lp) if x["kind"] != "budget"] == ["attempt_start"]
    print("ROUTE_GUARD_LEGACY=PASS rc=3 found=legacy")


# ═════════════════════════════════ 3 预算上限三入口一致 ═════════════════════════════════


def test_budget_caps_consistent_across_reserve_retry_report(tmp_path):
    """小账本（轨迹 12、首试 9 → 恢复合计 3；infra 2、到期 2）：恢复按 SeatRunner 的次序「先 claim_retry 再 reserve
    recovery」进行，到期接续与 infra 重试共用合计 3；第 4 次恢复在 retry 与 reserve 两个入口同时被拒，同一身份第二次领重试
    （第三次 attempt）被拒；report 的上限与两个入口一致；换一组构造参数时三个入口一致拒绝。"""
    bm = load_script("eval-official/budget_ledger.py")
    caps = {"trajectory_cap": 12, "shared_infra_cap": 2, "expired_cap": 2, "planned_first_tries": 9}
    led = bm.BudgetLedger(tmp_path / "b.jsonl", **caps)
    assert led.recovery_cap == 3
    for i in range(3):  # 三份首试先开（还剩 6 份未开始，恢复不得挤占）
        led.reserve(resets=0, route="r/seed7/new", key=f"k{i}", token=f"r/seed7/new|k{i}|a1", kind_of_try="first")
    plan = [("k0", "expired"), ("k1", "infra"), ("k2", "expired")]
    for key, it in plan:
        tok = f"r/seed7/new|{key}|a2"
        assert led.claim_retry(route="r/seed7/new", key=key, interrupt=it, token=tok) is True
        led.reserve(resets=0, route="r/seed7/new", key=key, token=tok, kind_of_try="recovery")
    st = led.state()
    assert (st.recovery_used, len(st.retries), st.retries_of("expired"), st.retries_of("infra")) == (3, 3, 2, 1)
    # 第 4 次恢复：retry 入口（infra 分项尚余 1，但合计 3 已满）与 reserve 入口都拒
    assert led.claim_retry(route="r/seed7/new", key="k9", interrupt="infra") is False
    with pytest.raises(bm.BudgetExhausted) as ei:
        led.reserve(resets=0, route="r/seed7/new", key="k9", kind_of_try="recovery")
    assert ei.value.reason in ("recovery_cap", "reserved_for_first_tries")
    # 同一身份第三次 attempt：该身份已领过一次重试，再领被拒（与合计无关的独立约束；换宽松账本单独验）
    led_b = bm.BudgetLedger(tmp_path / "b2.jsonl", trajectory_cap=12, shared_infra_cap=5, expired_cap=5,
                            planned_first_tries=0)
    assert led_b.claim_retry(route="r/seed7/new", key="kk", interrupt="infra") is True
    assert led_b.claim_retry(route="r/seed7/new", key="kk", interrupt="expired") is False
    # 余下 6 份首试照样全部可预约，之后轨迹满
    for i in range(3, 9):
        led.reserve(resets=0, route="r/seed7/new", key=f"k{i}", kind_of_try="first")
    with pytest.raises(bm.BudgetExhausted) as ei:
        led.reserve(resets=0, route="r/seed7/new", key="extra", kind_of_try="first")
    assert ei.value.reason == "trajectory_cap"
    ok, lines = led.report_lines()
    detail = next(x for x in lines if x.startswith("BUDGET_DETAIL"))
    assert ok and lines[-1].startswith("BUDGET_ENFORCEMENT=PASS trajectories=12/12 ")
    assert "first_started=9/9 recovery=3/3" in detail and "expired=2/2" in detail and "shared_infra=1/2" in lines[-1]
    # 构造参数改一项：reserve、claim_retry 抛 BudgetConfigMismatch，report 判 config_mismatch FAIL（不补写 config 行）
    n_rows = len(F.read_jsonl(tmp_path / "b.jsonl"))
    for k in caps:
        other = bm.BudgetLedger(tmp_path / "b.jsonl", **dict(caps, **{k: caps[k] + 1}))
        with pytest.raises(bm.BudgetConfigMismatch):
            other.reserve(resets=0, route="r/seed7/new", key="z", kind_of_try="first")
        with pytest.raises(bm.BudgetConfigMismatch):
            other.claim_retry(route="r/seed7/new", key="z", interrupt="infra")
        ok2, lines2 = other.report_lines()
        assert not ok2 and any("config_mismatch" in x for x in lines2), (k, lines2)
    assert len(F.read_jsonl(tmp_path / "b.jsonl")) == n_rows
    print("BUDGET_CAP_ENTRIES=PASS entries=reserve,claim_retry,report recovery_cap=3 expired_counted=2 "
          "third_attempt_rejected=1 config_mismatch_rejected_at=3")
