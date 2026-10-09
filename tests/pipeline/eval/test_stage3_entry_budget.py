"""第三阶段 R3：共享入口、预算账本、服务种子与外壳、1800 步、执行期限（1006 计划第二部分一节 R3 行、八.9、八.10 第
3～5 条、八.12；接口冻结说明二、三、五、七、八节）。纯 CPU：不起仿真、不加载权重、不联网；期望值一律由本文件手写。

判定行（本文件打印原文）：

* ``EVAL_CAP=PASS``：ood 严格上限 1800——第 1800 次 step 照常进环境、第 1801 次在进入环境前被拒（``cap_hit``、终态
  timeout，局结果 ``max_steps=1800 strict_cap=true``）；hard-verify 不 strict（第 1301 次照常进环境，局结果
  ``max_steps=1300 strict_cap=false``）；步数口径只由数据集决定（``EpisodeSpec``），没有可配错的参数。
* ``POLICY_SEEDS=PASS``：种子 0／7／42 贯通到四类服务命令（MME-VLA 外壳 ``--seed=``、MemER 同一命令、smvla
  ``--policy-seed``、pp ``--args.seed``）、席位交给 ``load_policy`` 的种子与变体参数、结果行 ``policy_seed``／
  ``server_seed``（从服务元数据反查）、账本路线 ``…/seed<n>/new``；缺失即 ``RUN_BLOCKED reason=policy_seed``；smvla 每局
  reseed 用该种子。
* ``BUDGET_ENFORCEMENT=PASS``：坏行即拒、补换行、config 不一致拒、同 token 幂等不重复扣、恢复挤占首试额度被拒
  （``planned_first_tries=821, trajectory_cap=870`` 下第 50 次恢复被拒、821 份首试仍全部可预约）、lease 被占
  ``RUN_BLOCKED``、缺预算参数 ``RUN_BLOCKED``、不给 ``--reset-budget`` 时只计量不拦、retry／reserve／attempt_start 共用
  同一 token。
* ``DEADLINES=PASS``：首推、媒体收尾两处期限到时以 ``infra_reason=deadline_<phase>`` 记局结果、席位先结算
  （``watchdog_exit``）再退 75，重起客户端重试一次即成功；席位与单局两份 ``progress.json`` 都记具名阶段。拆仓后上下文
  加载（``load_policy``）不再有单独期限。
* ``LEASE_BY_DATASET=PASS``：lease 按「路线 + 席位名」，两个席位同路线可并行、同席位名重复启动被拦；两个数据集同在
  一个动态队列里，靠队列键区分。
* ``OBS_EQ=PASS``：``SGEVAL_AUDIT=0/1`` 两种下 MME-VLA 外壳（假模型）与 smvla 服务的回包除审计键外逐字节相同、随机数
  状态相同、推理次数相同；多推理一次的坏外壳被比较器查出；审计里的通道原文、token id、mask、截断与真实分词一致。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

import eval_fakes as F
from robomme_hard_eval import episode as E
from robomme_hard_eval import servers as S
from tests._support.loaders import load_script

#: 旧仓 run_seat.sh 不迁（拆分方案第二部分 §二）：起服务端的命令与种子核对改由 ``robomme_hard_eval.servers`` 与各模型
#: Policy 子类承担，本文件对应断言改接到那里；席位脚本的预算参数核对与无进展检测归 GL 席位（dev-scripts/gl）。
EO = F.REPO / "scripts" / "eval-official"
CAPS = {"trajectory_cap": 870, "shared_infra_cap": 50, "expired_cap": 50, "planned_first_tries": 821}
CAP_ARGV = ["--trajectory-cap", "870", "--shared-infra-cap", "50", "--expired-cap", "50", "--planned-first-tries", "821"]
SEEDS = (0, 7, 42)
HARD0_TASK = "PickXtimes"


def bl():
    return load_script("eval-official/budget_ledger.py")


def _bash(script: str, **env) -> subprocess.CompletedProcess:
    e = dict(os.environ, EO=str(EO), **{k: str(v) for k, v in env.items()})
    e.pop("POLICY_SEED", None)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=60)


@pytest.fixture(autouse=True)
def _no_gate_env(monkeypatch):
    for k in ("SGEVAL_BUDGET_LEDGER", "SGEVAL_EXPIRED_JOBS", "SLURM_JOB_ID", "SLURM_JOB_END_TIME", "SGEVAL_AUDIT"):
        monkeypatch.delenv(k, raising=False)


class _Builder:
    """只给 EnvSession 用的最小 builder：每次 make_env_for_episode 新建一个假环境。"""

    def __init__(self, world: F.World, task: str = "T"):
        self.world, self.task = world, task

    def make_env_for_episode(self, ep):
        return self.world.new_env(self.task, int(ep))


def _step_until_cap_policy():
    def run_episode(session, identity, conn_info, recorder):
        session.reset()
        try:
            while True:
                session.step(np.zeros(8, dtype=np.float32))
        except F.env_session().StepCapReached:
            pass
        return {"status": "fail", "steps": session.steps, "infra": False, "error": None}
    return types.SimpleNamespace(run_episode=run_episode)


def _short_policy():
    """reset 后逐步执行到环境报终态（至多 5 步）。"""
    def run_episode(session, identity, conn_info, recorder):
        try:
            session.reset()
            status = "fail"
            for _ in range(5):
                *_, info = session.step(np.zeros(8, dtype=np.float32))
                if info["status"] in ("success", "fail"):
                    status = info["status"]
                    break
        except Exception as e:  # noqa: BLE001 与真实客户端同：异常记为非基础设施错误，额度耗尽由 SeatRunner 改判
            return {"status": "error", "steps": session.steps, "infra": False, "error": f"{type(e).__name__}: {e}"}
        return {"status": status, "steps": session.steps, "infra": False, "error": None}
    return types.SimpleNamespace(run_episode=run_episode)


# ═════════════════════════════════ EVAL_CAP ═════════════════════════════════


def test_eval_cap_1800_strict_and_hard_verify_not_strict(tmp_path):
    ec = F.env_session()
    world = F.World(default=F.Plan())  # 永不终止
    a = np.zeros(8, dtype=np.float32)
    s = ec.EnvSession("T", 0, builder=_Builder(world), step_cap=1800)
    s.reset()
    for _ in range(1800):
        s.step(a)
    env = world.envs[-1]
    assert env.n == 1800 and not s.cap_hit  # 第 1800 次照常进环境
    with pytest.raises(ec.StepCapReached):
        s.step(a)
    assert env.n == 1800 and s.cap_hit and s.steps == 1800  # 第 1801 次没进环境
    h = ec.EnvSession("T", 1, builder=_Builder(world), step_cap=None, dataset="hard-verify")
    h.reset()
    for _ in range(1301):
        h.step(a)
    assert world.envs[-1].n == 1301 and not h.cap_hit  # hard-verify 不 strict

    # 配对（旧 run_seat.sh::step_cap_pairing 的职责）：新接口里步数上限与是否严格截断只由数据集决定，没有可配错的参数；
    # 交给模型的 conn_info 只在 ood 下带 effective_cap=1800
    want = {"ood": (1800, True, 1800), "hard-verify": (1300, False, None)}
    pol = _seed_policy("pp", 7, tmp_path)
    bad = 0
    for d, (ms, strict, eff) in want.items():
        spec = _fake_spec(d, pol, tmp_path)
        conn = pol._conn_info(spec)
        bad += int(E.DATASET_MAX_STEPS[d] != ms or (spec.max_steps, spec.strict_cap) != (ms, strict)
                   or (conn["max_steps"], conn["strict_cap"], conn["effective_cap"]) != (ms, strict, eff))
    with pytest.raises(ValueError):
        E.EpisodeSpec(**{**vars(_fake_spec("ood", pol, tmp_path)), "dataset": "test-hard"})
    assert bad == 0 and set(E.DATASET_MAX_STEPS) == set(want)
    # SeatRunner（拆仓后常驻 Policy + 动态队列）：ood 1800 strict → timeout、cap_hit、第 1801 步不进环境；局结果
    # result.json 记 max_steps=1800、strict_cap=true。拆仓后席位不再有 effective_cap／seat_info（步数口径归 EpisodeSpec），
    # 改核局结果里的 max_steps／strict_cap 两键
    task, tier = F.v9_cells_sorted()[0]
    ident = F.packaged_identity(task, tier, 0)
    world_ood = F.World(default=F.Plan())
    runner = F.make_runner(tmp_path / "ood", "pp", _step_until_cap_policy(), world_ood, policy_seed=7)
    assert F.run_rows(runner, [ident]) == 0
    (row,) = F.read_jsonl(runner.results_path)
    assert (row["status"], row["cap_hit"], row["exec_steps"]) == ("timeout", True, 1800)
    assert world_ood.envs[-1].n == 1800
    res = json.loads(Path(row["result"]).read_text())
    assert (res["max_steps"], res["strict_cap"], res["status"], res["cap_hit"]) == (1800, True, "timeout", True)
    hv = F.make_runner(tmp_path / "hv", "pp", _short_policy(), F.World(), policy_seed=7)
    assert F.run_rows(hv, [F.hard0_identity(HARD0_TASK, 0)]) == 0
    (hrow,) = F.read_jsonl(hv.results_path)
    hres = json.loads(Path(hrow["result"]).read_text())
    assert (hres["max_steps"], hres["strict_cap"], hres["dataset"]) == (1300, False, "hard-verify")

    print(f"EVAL_CAP=PASS ood_cap=1800 step_1800_entered=1 step_1801_entered=0 hard_verify_strict=0 pairing_cases={len(want)}")


# ═════════════════════════════════ POLICY_SEEDS ═════════════════════════════════

#: 各服务端命令的手写期望用到的固定参数（旧 LIB_SRV 里的 MME_VLA_PY／SMVLA_PY／PP_PY 与各 ckpt）
SRV_CFG = {"perceptual-framesamp-modul": {"mme_vla_py": "/py/mme-vla", "ckpt": "/ck/fsm"},
           "groundsg": {"mme_vla_py": "/py/mme-vla", "ckpt": "/ck/sg"},
           "smvla": {"smvla_py": "/py/smvla", "ckpt": "/ck/smvla"},
           "pp": {"pp_py": "/py/pp", "ckpt": "/ck/pp"}}


def _seed_policy(pol, seed, tmp_path, *, variant="", memer="", wrap="0"):
    """按旧 LIB_SRV 的口径建 Policy 实例（不 load、不起服务），端口固定 18123，元数据目录 ``<tmp>/o/<label>``。"""
    from robomme_hard_eval.models import resolve

    cfg = dict(SRV_CFG[pol], port=18123, openpi_data_home="/openpi", preflight=False, ckpt_fingerprint=False)
    if pol == "groundsg":
        cfg.update(groundsg_variant=variant, memer_adapter_path=memer or None)
    if pol == "pp":
        cfg["pp_server_wrap"] = wrap == "1"
    p = resolve(pol)(policy_seed=seed, **cfg)
    p.metadata_dir = tmp_path / "o" / p.label
    p._pick_port()
    return p


def _fake_spec(dataset, pol, tmp_path):
    ms = E.DATASET_MAX_STEPS[dataset]
    return E.EpisodeSpec(dataset=dataset, task=HARD0_TASK, episode=0, source_episode=3 if dataset == "hard-verify"
                         else None, tier="xhard0" if dataset == "hard-verify" else "xhard2", seed=11, candidate=None,
                         spec_sha256=None, key=f"{HARD0_TASK}_x_11", max_steps=ms, strict_cap=dataset == "ood",
                         attempt=1, policy_seed=pol.policy_seed, out_dir=str(tmp_path / "raw"))


def _srv_cli(pol, seed, tmp_path, *, variant="", memer="", wrap="0"):
    """服务端 argv（各模型 ``*_server_spec``，即旧 ``build_server_cmd`` 的 Python 版）与交给客户端的 conn_info。"""
    from robomme_hard_eval.models import framesamp_modul as fm
    from robomme_hard_eval.models import pp as ppm
    from robomme_hard_eval.models import smvla as smm

    p = _seed_policy(pol, seed, tmp_path, variant=variant, memer=memer, wrap=wrap)
    ck = SRV_CFG[pol]["ckpt"]
    build = {"perceptual-framesamp-modul": fm.mme_vla_server_spec, "groundsg": fm.mme_vla_server_spec,
             "smvla": smm.smvla_server_spec, "pp": ppm.pp_server_spec}[pol]
    argv, _env, _cwd = build(p, ck)
    return p, argv, p._conn_info(_fake_spec("ood", p, tmp_path))


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_policy_seeds_reach_servers_clients_seat_info_and_results(tmp_path):
    ec = F.env_client()
    srv_mod = F.smvla_server()
    from robomme_hard_eval import servers as S

    wrap = str(S.SERVERS_DIR / "policy_server_wrap.py")
    o = tmp_path / "o"
    routes = mismatches = 0
    for seed in SEEDS:
        s = str(seed)
        cases = [
            ("perceptual-framesamp-modul", "", "", "0",
             lambda srv, s=s: srv[:2] == ["/py/mme-vla", wrap] and f"--seed={s}" in srv
             and f"--sgeval-metadata-out={o}/perceptual-framesamp-modul/server-wrap-metadata-18123.json" in srv),
            ("groundsg", "ground-sg-oracle", "", "0",
             lambda srv, s=s: srv[:2] == ["/py/mme-vla", wrap] and f"--seed={s}" in srv),
            ("groundsg", "ground-sg-memer", "/ad/memer", "0",  # MemER 的动作服务命令与 GroundSG 相同
             lambda srv, s=s: srv[:2] == ["/py/mme-vla", wrap] and f"--seed={s}" in srv and "--policy.dir=/ck/sg" in srv),
            ("smvla", "", "", "0",
             lambda srv, s=s: _opt(srv, "--policy-seed") == s
             and _opt(srv, "--metadata_out") == f"{o}/smvla/server-wrap-metadata-18123.json"),
            ("pp", "", "", "0", lambda srv, s=s: srv[:3] == ["/py/pp", "-m", "ponderpounce.eval.robomme_server"]
             and _opt(srv, "--args.seed") == s),
            ("pp", "", "", "1", lambda srv, s=s: srv[1].endswith("pp_server_wrap.py") and _opt(srv, "--args.seed") == s
             and f"--sgeval-metadata-out={o}/pp/server-wrap-metadata-18123.json" in srv),
        ]
        for pol, variant, memer, wrap_on, ok in cases:
            p, srv, conn = _srv_cli(pol, seed, tmp_path, variant=variant, memer=memer, wrap=wrap_on)
            routes += 1
            good = ok(srv) and conn["policy_seed"] == seed and p.policy_seed == seed \
                and not any(a in ("--seed=7", "--seed=0") for a in srv if a != f"--seed={s}")
            if pol == "groundsg":
                good = good and p.variant == variant and (p.memer_adapter or "") == memer \
                    and p.label == f"groundsg-{variant}"
            mismatches += int(not good)
            assert good, (pol, variant, wrap_on, seed, srv, conn)
        # 进程内（拆仓后）：种子与变体经 load_policy(model, seed, **cfg) 交给常驻 Policy；conn_info、结果行、账本路线、
        # server_seed 反查（服务端元数据的 policy_seed，Policy.server_seed）。旧 make_policy_context(seat_info) 已不存在。
        calls = {"factory": [], "conn": []}

        def run_episode(session, identity, conn_info, recorder):
            calls["conn"].append({k: v for k, v in conn_info.items() if k != "policy_context"})
            session.reset()
            *_, info = session.step([0.0] * 8)
            return {"status": "fail", "steps": 1, "infra": False, "error": None, "memer_compat_sha256": "c" * 64}

        adapter = tmp_path / f"memer-{seed}"
        adapter.mkdir()
        ledger = tmp_path / f"budget-{seed}.jsonl"
        stage = tmp_path / f"stage-{seed}"
        runner = F.make_runner(stage, "groundsg", types.SimpleNamespace(run_episode=run_episode), F.World(),
                               groundsg_variant="ground-sg-memer", memer_adapter=str(adapter), policy_seed=seed,
                               budget_ledger=str(ledger), **CAPS)
        pol = runner.fake_policy
        srv = types.SimpleNamespace(metadata={"policy_seed": seed}, left_line=lambda: "", check=lambda: None,
                                    stop=lambda **k: "stopped")  # 假服务端：只带元数据（种子反查用）
        pol.servers = lambda srv=srv: [srv]

        def factory(model, pseed, **cfg):
            calls["factory"].append((model, pseed, cfg))
            return pol
        runner.policy_factory = factory
        ident = F.hard0_identity(HARD0_TASK, 0)
        assert F.run_rows(runner, [ident]) == 0
        runner.close()
        assert calls["factory"] == [("groundsg", seed, {"groundsg_variant": "ground-sg-memer",
                                                         "memer_adapter": str(adapter)})]
        assert calls["conn"][0]["policy_seed"] == seed and calls["conn"][0]["dataset"] == "hard-verify"
        (row,) = F.read_jsonl(runner.results_path)
        route = f"groundsg/ground-sg-memer/seed{seed}/new"
        assert runner.route == route and runner.label == "groundsg-ground-sg-memer"
        res = json.loads(Path(row["result"]).read_text())
        assert (row["policy_seed"], res["policy_seed"], res["server_seed"], res["memer_compat_sha256"]) == \
            (seed, seed, seed, "c" * 64)
        assert row["budget_token"] == f"{route}|{ident['key']}|a1" and row["error_kind"] is None
        res = [r for r in F.read_jsonl(ledger) if r["kind"] == "reserve"]
        assert [(r["route"], r["token"], r["kind_of_try"]) for r in res] == [(route, row["budget_token"], "first")]
        # smvla 服务：每局 reseed 用该种子（object.__new__ 构造的 host + 真实 new_episode）
        host = object.__new__(srv_mod.SMVLAPolicyHost)
        host.policy_seed = seed
        host.buffer_factory = type("B", (), {"reset": lambda self: None, "image_keys": []})
        host.new_episode()
        got = srv_mod.rng_digest()
        srv_mod.reseed(seed)
        assert got == srv_mod.rng_digest()
    # 缺失 → RUN_BLOCKED reason=policy_seed（CLI 与席位脚本两处）
    base = ["run", "--identities", "x.json", "--seat", "s", "--out", str(tmp_path / "o"), "--policy", "pp",
            "--budget-ledger", "b.jsonl", "--reset-cap", "20", *CAP_ARGV]
    assert ec.entry_blockers(ec.build_parser().parse_args(base))[0] == "policy_seed"
    assert ec.entry_blockers(ec.build_parser().parse_args(base + ["--policy-seed", "-1"]))[0] == "policy_seed"
    assert ec.entry_blockers(ec.build_parser().parse_args(base + ["--policy-seed", "42"])) is None
    # 服务端一侧（旧 policy_seed_check／build_server_cmd）：四个 Policy 拒负种子，文字为 RUN_BLOCKED reason=policy_seed
    blocked = 0
    for pol in SRV_CFG:
        with pytest.raises(S.PreflightError, match="RUN_BLOCKED reason=policy_seed"):
            _seed_policy(pol, -1, tmp_path, variant="ground-sg-oracle")
        blocked += 1
    assert blocked == 4
    print(f"POLICY_SEEDS=PASS seeds={','.join(map(str, SEEDS))} server_routes={routes} mismatch={mismatches} "
          f"missing_blocked=1")


def test_variant_pairing_memer_cli(tmp_path, capsys):
    ec = F.env_client()
    adapter = tmp_path / "memer"
    adapter.mkdir()
    base = ["run", "--identities", "x.json", "--seat", "s", "--out", str(tmp_path / "o"), "--policy", "groundsg",
            "--budget-ledger", "b.jsonl", "--reset-cap", "20", *CAP_ARGV, "--policy-seed", "7"]
    cases = {
        "memer_no_adapter": (["--groundsg-variant", "ground-sg-memer"], "--memer-adapter"),
        "memer_missing_dir": (["--groundsg-variant", "ground-sg-memer", "--memer-adapter", str(tmp_path / "nope")], "目录不存在"),
        "memer_with_qwenvl": (["--groundsg-variant", "ground-sg-memer", "--memer-adapter", str(adapter),
                               "--qwenvl-groundsg-adapter", str(adapter)], "--qwenvl-groundsg-adapter"),
        "oracle_with_memer": (["--groundsg-variant", "ground-sg-oracle", "--memer-adapter", str(adapter)], "--memer-adapter"),
    }
    for name, (extra, why) in cases.items():
        rc = ec.main(base + extra)
        out = capsys.readouterr().out
        assert rc == 3 and "RUN_BLOCKED reason=variant_pairing" in out and why in out, (name, out)
    ok = ec.build_parser().parse_args(base + ["--groundsg-variant", "ground-sg-memer", "--memer-adapter", str(adapter)])
    assert ec.entry_blockers(ok) is None and ec.variant_problems(ok, check_dirs=True) == []


# ═════════════════════════════════ BUDGET_ENFORCEMENT ═════════════════════════════════


def test_budget_enforcement_counterexamples(tmp_path, monkeypatch, capsys):
    bm, ec = bl(), F.env_client()
    cases = 0
    # 1 坏行即拒（读写前）：reserve 抛 LedgerCorrupt，CLI 退出 3，report 判 FAIL 不抛
    p1 = tmp_path / "corrupt.jsonl"
    bm.BudgetLedger(p1, **CAPS).reserve(resets=2, route="r", key="k")
    with open(p1, "a", encoding="utf-8") as fh:
        fh.write('{"kind": "reserve", "rid"')
    with pytest.raises(bm.LedgerCorrupt):
        bm.BudgetLedger(p1, **CAPS).reserve(resets=2, route="r", key="k2")
    with pytest.raises(bm.LedgerCorrupt):
        bm.BudgetLedger(p1, **CAPS).claim_retry(route="r", key="k", interrupt="infra")
    assert bm.main(["--ledger", str(p1), *CAP_ARGV, "reserve", "--resets", "2"]) == 3
    assert "RUN_BLOCKED reason=ledger_corrupt" in capsys.readouterr().out
    ok, lines = bm.BudgetLedger(p1, **CAPS).report_lines()
    assert not ok and "bad_rows=1" in " ".join(lines)
    cases += 1
    # 2 补换行：末行缺换行（完整 JSON）时追加前先补一个换行，两行都可解析
    p2 = tmp_path / "pad.jsonl"
    led2 = bm.BudgetLedger(p2, **CAPS)
    led2.check()
    raw = p2.read_bytes().rstrip(b"\n")
    p2.write_bytes(raw)
    led2.reserve(resets=1, route="r", key="k")
    lines2 = p2.read_bytes().split(b"\n")
    assert lines2[-1] == b"" and all(json.loads(x) for x in lines2[:-1]) and len(lines2) == 3
    cases += 1
    # 3 config 不一致：首次打开写 config 行；构造参数改一项即拒（CLI 退出 3）
    p3 = tmp_path / "cfg.jsonl"
    bm.BudgetLedger(p3, **CAPS).check()
    (cfg,) = [r for r in F.read_jsonl(p3) if r["kind"] == "config"]
    assert {k: cfg[k] for k in CAPS} == CAPS and cfg["schema"] == "sgeval-budget/3"
    for k in CAPS:
        with pytest.raises(bm.BudgetConfigMismatch):
            bm.BudgetLedger(p3, **dict(CAPS, **{k: CAPS[k] + 1})).reserve(resets=1)
    with pytest.raises(bm.BudgetConfigMismatch):
        bm.BudgetLedger(p3).state()  # 回落常量默认值同样被拒
    args = [*CAP_ARGV]
    args[1] = "871"
    assert bm.main(["--ledger", str(p3), *args, "report"]) == 1
    assert bm.main(["--ledger", str(p3), *args, "reserve", "--resets", "1"]) == 3
    assert "RUN_BLOCKED reason=budget_config" in capsys.readouterr().out
    cases += 1
    # 4 同 token 幂等：reserve／claim_retry 重放不写新行、不重复扣
    p4 = tmp_path / "token.jsonl"
    led4 = bm.BudgetLedger(p4, **CAPS)
    tok = bm.make_token("pp/seed7/new", "K", 2)
    assert tok == "pp/seed7/new|K|a2"
    assert led4.claim_retry(route="pp/seed7/new", key="K", interrupt="infra", token=tok) is True
    assert led4.claim_retry(route="pp/seed7/new", key="K", interrupt="infra", token=tok) is True
    r1 = led4.reserve(resets=2, route="pp/seed7/new", key="K", token=tok, kind_of_try="recovery")
    r2 = led4.reserve(resets=2, route="pp/seed7/new", key="K", token=tok, kind_of_try="recovery")
    st = led4.state()
    assert r1 == r2 and st.trajectories == 1 and len(st.retries) == 1
    led4.release(r1)  # 退回后同 token 可重新预约（新 rid）
    assert led4.reserve(resets=2, route="pp/seed7/new", key="K", token=tok, kind_of_try="recovery") != r1
    cases += 1
    # 5 首试保留额度：planned=821、cap=870 下恢复最多 49 次，第 50 次被拒；随后 821 份首试仍全部可预约
    p5 = tmp_path / "reserve.jsonl"
    led5 = bm.BudgetLedger(p5, **CAPS)
    for i in range(49):
        led5.reserve(resets=0, route="r", key=f"rec{i}", token=f"r|rec{i}|a2", kind_of_try="recovery")
    with pytest.raises(bm.BudgetExhausted) as ei:
        led5.reserve(resets=0, route="r", key="rec49", token="r|rec49|a2", kind_of_try="recovery")
    assert ei.value.reason == "reserved_for_first_tries"
    assert led5.claim_retry(route="r", key="rec49", interrupt="infra") is False  # 重试领取同样受约束
    for i in range(821):
        led5.reserve(resets=0, route="r", key=f"first{i}", token=f"r|first{i}|a1", kind_of_try="first")
    with pytest.raises(bm.BudgetExhausted) as ei:
        led5.reserve(resets=0, route="r", key="first821", kind_of_try="first")
    assert ei.value.reason == "trajectory_cap"
    st5 = led5.state()
    assert (st5.trajectories, st5.first_started, st5.recovery_used) == (870, 821, 49)
    ok5, lines5 = led5.report_lines()
    assert ok5 and lines5[-1].startswith("BUDGET_ENFORCEMENT=PASS trajectories=870/870 ")
    # infra 与 expired 合计受 cap−planned=49 约束（分项各自另受 50）
    p5b = tmp_path / "retry.jsonl"
    led5b = bm.BudgetLedger(p5b, trajectory_cap=870, shared_infra_cap=50, expired_cap=50, planned_first_tries=821)
    for i in range(821):  # 首试全部开始后，只剩恢复额度
        led5b.reserve(resets=0, route="r", key=f"f{i}", kind_of_try="first")
    got = [led5b.claim_retry(route="r", key=f"k{i}", interrupt="infra" if i % 2 else "expired") for i in range(50)]
    assert got == [True] * 49 + [False]
    cases += 1
    # 6 lease 被占：同一 shard 第二个持有者拿不到；SeatRunner 读待跑身份前取 lease，被占即 RUN_BLOCKED 退出 3
    p6 = tmp_path / "lease" / "budget.jsonl"
    led6 = bm.BudgetLedger(p6, **CAPS)
    with led6.lease("pp_seed7_new--shard-00"):
        with pytest.raises(bm.LeaseHeld):
            with bm.BudgetLedger(p6, **CAPS).lease("pp_seed7_new--shard-00"):
                pass
    with led6.lease("pp_seed7_new--shard-00"):  # 释放后可再取
        pass
    world = F.World()
    task, tier = F.v9_cells_sorted()[0]
    ident = F.packaged_identity(task, tier, 0)
    runner = F.make_runner(tmp_path / "lease-stage", "pp", _short_policy(), world, policy_seed=7,
                           budget_ledger=str(p6), **CAPS)
    sid = ec.shard_id_of(runner.ledger.route, "s00")
    with led6.lease(sid):
        assert F.run_rows(runner, [ident]) == 3
    assert "RUN_BLOCKED reason=lease_held" in capsys.readouterr().out and world.envs == []
    assert not runner.results_path.exists()
    cases += 1
    # 7 缺预算参数：CLI 拦 reason=budget_args（不回落常量默认值）；席位脚本同样拦
    base = ["run", "--identities", "x.json", "--seat", "s", "--out", str(tmp_path / "o7"), "--policy", "pp",
            "--policy-seed", "7"]
    full = base + ["--budget-ledger", "b.jsonl", "--reset-cap", "20", *CAP_ARGV]
    assert ec.entry_blockers(ec.build_parser().parse_args(full)) is None
    for i in range(0, len(full) - len(base), 2):
        argv = full[:len(base) + i] + full[len(base) + i + 2:]
        blk = ec.entry_blockers(ec.build_parser().parse_args(argv))
        assert blk is not None and blk[0] == "budget_args", argv
        assert ec.main(argv) == 3
        assert "RUN_BLOCKED reason=budget_args" in capsys.readouterr().out
    # 旧 run_seat.sh::budget_args_check 的同项核对（不迁，席位入口只剩上面的 entry_blockers）见
    # test_seat_shell_budget_args_check（待 GL 席位脚本合并后改接）
    cases += 1
    # 8 不给 --reset-budget：只计量（reset_claim 照写、budget 行 reset_budget=null），多局多次 reset 也不停；给 1 仍硬拦
    rows = [F.packaged_identity(task, tier, k) for k in range(3)]
    stage8 = tmp_path / "meter"
    r8 = F.make_runner(stage8, "pp", _short_policy(), F.World(default=F.Plan(fail_at=2)), reset_budget=None,
                       policy_seed=7)
    assert F.run_rows(r8, rows) == 0
    led_rows = F.read_jsonl(r8.ledger.path)
    assert [r["reset_budget"] for r in led_rows if r["kind"] == "budget"] == [None]
    assert sum(r["kind"] == "reset_claim" for r in led_rows) == 6 and r8.ledger.reset_left() == float("inf")
    r8b = F.make_runner(tmp_path / "hard", "pp", _short_policy(), F.World(default=F.Plan(fail_at=2)),
                        reset_budget=1, policy_seed=7)
    assert F.run_rows(r8b, rows) == 5
    cases += 1
    # 9 同一 token 贯穿 claim_retry／reserve／attempt_start（共享模式，首试 infra → 重试成功）
    p9 = tmp_path / "shared9.jsonl"
    a9 = rows[0]
    world9 = F.World({(a9["task"], a9["builder_episode"]): [F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("svulkan2")),
                                                            F.Plan(success_at=2)]})

    def run9(session, identity, conn_info, recorder):
        session.reset()
        try:
            for _ in range(5):
                *_, info = session.step(np.zeros(8, dtype=np.float32))
                if info["status"] == "success":
                    return {"status": "success", "steps": session.steps, "infra": False, "error": None}
        except RuntimeError as e:
            return {"status": "error", "steps": session.steps, "infra": True, "infra_reason": "env_step",
                    "error": str(e)}
        return {"status": "fail", "steps": session.steps, "infra": False, "error": None}

    r9 = F.make_runner(tmp_path / "s9", "pp", types.SimpleNamespace(run_episode=run9), world9, policy_seed=7,
                       budget_ledger=str(p9), **CAPS)
    assert F.run_rows(r9, [a9]) == 0
    route = "pp/seed7/new"
    toks = [f"{route}|{a9['key']}|a1", f"{route}|{a9['key']}|a2"]
    starts = [r for r in F.read_jsonl(r9.ledger.path) if r["kind"] == "attempt_start"]
    shared = F.read_jsonl(p9)
    assert [r["token"] for r in starts] == toks
    assert [(r["token"], r["kind_of_try"]) for r in shared if r["kind"] == "reserve"] == \
        [(toks[0], "first"), (toks[1], "recovery")]
    assert [r["token"] for r in shared if r["kind"] == "retry_claim"] == [toks[1]]
    assert [r["budget_token"] for r in F.read_jsonl(r9.results_path)] == toks
    # 崩溃窗口重放：同一 token 再领重试、再预约，不写新行
    led9 = bm.BudgetLedger(p9, **CAPS)
    assert led9.claim_retry(route=route, key=a9["key"], interrupt="infra", token=toks[1]) is True
    n_before = len(F.read_jsonl(p9))
    rid_again = led9.reserve(resets=2, route=route, key=a9["key"], token=toks[1], kind_of_try="recovery")
    assert rid_again == [r for r in shared if r["kind"] == "reserve"][1]["rid"] and len(F.read_jsonl(p9)) == n_before
    cases += 1
    print(f"BUDGET_ENFORCEMENT=PASS cases={cases} corrupt_rejected=1 newline_padded=1 config_mismatch_rejected=1 "
          f"token_idempotent=1 recovery_50th_rejected=1 first_tries_821_reservable=1 lease_held_blocked=1 "
          f"budget_args_blocked=1 reset_meter_only=1 shared_token=1")


# ═════════════════════════════════ DEADLINES ═════════════════════════════════


class _Died(SystemExit):
    """模拟看门狗 ``os._exit(75)`` 之后进程已死：被卡住的主线程醒来即退出，不再走后续收尾。"""


def _deadline_runner(tmp_path, phase: str, world=None, *, stage=None):
    """第一次尝试在指定阶段卡住（等事件），期限 0.3 s；注入的 hard_exit 记下退出码、放行卡住的线程，卡住处醒来后抛
    ``_Died``（等价于进程已被 ``os._exit`` 杀掉）。拆仓后单局看门狗只管首推理与媒体收尾两个阶段期限（另有整局墙钟），
    上下文加载（``load_policy``）不再有单独期限。"""
    ev = threading.Event()
    exits: list[int] = []
    calls = {"n": 0}

    def run_episode(session, identity, conn_info, recorder):
        calls["n"] += 1
        session.reset()
        if phase == "first_infer" and calls["n"] == 1:
            assert ev.wait(10)
            raise _Died(75)
        *_, info = session.step([0.0] * 8)
        return {"status": "fail", "steps": 1, "infra": False, "error": None}

    mod = types.SimpleNamespace(run_episode=run_episode)
    world = world or F.World()

    class _Slow(F.FakeRecorder):
        n = 0

        def close(self, summary):
            _Slow.n += 1
            if phase == "media_finalize" and _Slow.n == 1:
                assert ev.wait(10)
                raise _Died(75)
            return super().close(summary)

    runner = F.make_runner(stage or tmp_path, "pp", mod, world, policy_seed=7,
                           recorder_factory=lambda d, m: _Slow(d, m, world),
                           hard_exit=lambda code: (exits.append(code), ev.set()))
    runner.episode_kwargs.update(first_infer_s=0.3, media_s=0.3)
    return runner, exits


@pytest.mark.parametrize("phase", ["first_infer", "media_finalize"])
def test_deadline_ends_episode_as_infra_and_retry_succeeds(tmp_path, capsys, phase):
    world = F.World()
    runner, exits = _deadline_runner(tmp_path, phase, world)
    ident = F.hard0_identity(HARD0_TASK, 0)
    with pytest.raises(_Died):  # 第一次进程：看门狗到点、席位先结算再退 75
        F.run_rows(runner, [ident])
    assert exits == [75]
    out = capsys.readouterr().out
    assert f"INFRA_TIMEOUT phase={phase} key={ident['key']}" in out and "SEAT_WATCHDOG_EXIT" in out
    # 重起客户端（同一席位名）：上一次尝试已由 _hard_exit 写 attempt_end，不算悬空；按 1 次基础设施重试再跑一次即成功
    again, exits2 = _deadline_runner(tmp_path, "none", world)
    assert F.run_rows(again, [ident]) == 0 and exits2 == []
    rows = F.read_jsonl(again.results_path)
    assert [(r["attempt"], r["status"], r["infra"], r.get("infra_reason")) for r in rows] == \
        [(1, "error", True, "watchdog_exit"), (2, "fail", False, None)]
    ep_rows = F.read_jsonl(Path(rows[-1]["result"]).parents[2] / "results.jsonl")
    assert [(r["attempt"], r["status"], r["infra"], r["infra_reason"], r["error_kind"]) for r in ep_rows] == \
        [(1, "fail", True, f"deadline_{phase}", "error"), (2, "fail", False, None, None)]
    led = F.read_jsonl(again.ledger.path)
    assert [r["kind"] for r in led if r["kind"] in ("attempt_start", "attempt_end", "accept")] == \
        ["attempt_start", "attempt_end", "attempt_start", "attempt_end", "accept"]
    prog = json.loads(again.progress_path.read_text())
    assert prog["phase"] == "claim" and prog["episodes_done"] == 1 and prog["policy_seed"] == 7


def test_progress_phases_and_seat_idle_reads_named_progress(tmp_path, monkeypatch):
    """两份 progress.json 都经过具名阶段：席位层（``seats/…/progress.json``）context_load → claim → episode → done →
    claim（再扫一遍无可领），单局层（``rollouts/…/progress.json``）reset → first_infer → episode → media_finalize → done。"""
    seen, ep_seen = [], []
    runner = F.make_runner(tmp_path / "p", "pp", _short_policy(), F.World(default=F.Plan(fail_at=3)), policy_seed=7)
    orig = runner.progress

    def spy(phase, **kw):
        orig(phase, **kw)
        seen.append(json.loads(runner.progress_path.read_text())["phase"])
    runner.progress = spy
    orig_ep = E._progress

    def ep_spy(spec, phase, step=0):
        orig_ep(spec, phase, step)
        ep_seen.append(phase)
    monkeypatch.setattr(E, "_progress", ep_spy)
    assert F.run_rows(runner, [F.hard0_identity(HARD0_TASK, 0)]) == 0
    dedup = [p for i, p in enumerate(seen) if i == 0 or p != seen[i - 1]]
    assert dedup == ["context_load", "claim", "episode", "done", "claim"]
    assert set(seen) <= set(F.env_client().PHASES)
    ep_dedup = [p for i, p in enumerate(ep_seen) if i == 0 or p != ep_seen[i - 1]]
    assert ep_dedup == ["reset", "first_infer", "episode", "media_finalize", "done"]

    print("DEADLINES=PASS phases=first_infer,media_finalize exit_code=75 retry_ok=1 named_progress=1")


@pytest.mark.skip(reason="旧 run_seat.sh 不迁（拆分方案第二部分 §二）；席位脚本的无进展检测归 dev-scripts/gl，待其合并后改接")
def test_seat_shell_idle_reads_named_progress(tmp_path):
    """席位脚本的无进展计时只认 phase／identity／step 变化（原断言原样保留，等 GL 席位脚本给出新落点后改接）。"""
    d = tmp_path / "seat"
    d.mkdir()
    pj, log = d / "progress.json", d / "client.log"
    script = r'''
source "$EO/run_seat.sh"; TOOL_PY="$PY"
PROG_FLOOR=$(( $(ts) - 1000 ))
w() { printf '{"phase": "%s", "identity": "k", "attempt_no": 1, "step": %s, "episodes_done": 0, "t": %s}' "$1" "$2" "$3" > "$PJ"; }
w episode 16 $(( $(ts) - 500 )); progress_idle "$PJ"; echo "A=$IDLE"
w episode 16 $(ts); touch "$LOG"; progress_idle "$PJ"; echo "B=$IDLE"
w episode 32 $(( $(ts) - 5 )); progress_idle "$PJ"; echo "C=$IDLE"
echo "S=$(idle_s "$PJ")"
'''
    p = _bash(script, PY=sys.executable, PJ=pj, LOG=log)
    vals = dict(re.findall(r"^([ABCS])=(\d+)$", p.stdout, re.M))
    assert {k: int(v) for k, v in vals.items()}.keys() == {"A", "B", "C", "S"}, p.stdout + p.stderr
    a, b, c = int(vals["A"]), int(vals["B"]), int(vals["C"])
    assert 495 <= a <= 520 and b >= a and 4 <= c <= 30  # 同签名重写不重计时；步数变化才算进展


@pytest.mark.skip(reason="旧 run_seat.sh 不迁；预算参数核对的席位脚本版归 dev-scripts/gl，待其合并后改接")
def test_seat_shell_budget_args_check():
    p = _bash('source "$EO/run_seat.sh"; BUDGET_LEDGER=/b; TRAJECTORY_CAP=870; SHARED_INFRA_CAP=50; EXPIRED_CAP=""; '
              'PLANNED_FIRST_TRIES=821; budget_args_check; echo "RC=$?"')
    assert "RC=3" in p.stdout and "RUN_BLOCKED reason=budget_args" in p.stdout and "--expired-cap" in p.stdout


# ═════════════════════════════════ OBS_EQ ═════════════════════════════════


class _SPProc:
    """sentencepiece 处理器替身：每个字符一个 id（bos=1）。"""

    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [ord(ch) % 97 + 2 for ch in text]


def _tokenizer_cls():
    """与三方 PaligemmaTokenizer.tokenize 同形（字符串拼法、补齐、截断）的替身类；每次新建一个类（各跑一套补丁）。"""

    class Tok:
        def __init__(self, max_len=64):
            self._max_len = max_len
            self._tokenizer = _SPProc()

        def tokenize(self, prompt, state=None, subgoal=None):
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            if subgoal is not None:
                sg = subgoal.strip().replace("_", " ").replace("\n", " ")
                tokens = self._tokenizer.encode(f"Task: {cleaned};\nCurrent Subgoal: {sg};\nAction: ", add_bos=True)
            else:
                tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")
            n = len(tokens)
            if n < self._max_len:
                mask = [True] * n + [False] * (self._max_len - n)
                tokens = tokens + [False] * (self._max_len - n)
            else:
                tokens, mask = tokens[: self._max_len], [True] * self._max_len
            return np.asarray(tokens), np.asarray(mask)
    return Tok


OBS_SEQ = [{"prompt": "pick_cube", "subgoal": "grasp", "state": [0.1] * 8},
           {"prompt": "pick_cube", "subgoal": "lift the cube high above the table now", "state": [0.2] * 8},
           {"prompt": "stack all the blocks", "subgoal": None, "state": [0.3] * 8}]


def _fake_sp(tok_cls, sink: dict):
    """三方 serve_policy 模块替身：create_policy 建「每次 infer 两路分词 + 用自己的 RNG 采样」的假模型；main 依次推理。"""

    class Policy:
        def __init__(self, seed):
            self.tok = tok_cls()
            self.rng = np.random.default_rng(seed)
            self.calls = 0
            self.metadata = {"fake": True}

        def infer(self, obs):
            self.calls += 1
            t, m = self.tok.tokenize(obs["prompt"], None)
            out = {"state": np.asarray(obs["state"], dtype=np.float32)}
            if obs["subgoal"] is not None:
                st, sm = self.tok.tokenize(prompt=obs["prompt"], subgoal=obs["subgoal"], state=None)
                out["sym"] = st[:6].astype(np.int64)
            noise = self.rng.standard_normal(4).astype(np.float32)
            out["actions"] = noise + t[:4].astype(np.float32) * 0.01
            return out

        def reset(self):
            pass

    sp = types.SimpleNamespace()
    sp.create_policy = lambda args: Policy(args.seed)

    def main(args):
        pol = sp.create_policy(args)
        sink["outs"] = [pol.infer(o) for o in OBS_SEQ]
        sink["rng"] = pol.rng.bit_generator.state
        sink["calls"] = pol.calls
        sink["policy"] = pol
    sp.main = main
    return sp


def _run_wrap(wrap, monkeypatch, audit: str, tmp_path, *, mutant=False):
    """以外壳 ``run`` 起替身服务（每次新建分词类，补丁互不影响）；返回 (推理产物, 服务元数据)。"""
    monkeypatch.setenv("SGEVAL_AUDIT", audit)
    sink: dict = {}
    tok = _tokenizer_cls()
    sp = _fake_sp(tok, sink)
    if mutant:  # 坏外壳：多推理一次（必须被比较器查出）
        class Twice(wrap.AuditedPolicy):
            def infer(self, obs):
                self._inner.infer(obs)
                return super().infer(obs)
        monkeypatch.setattr(wrap, "AuditedPolicy", Twice)
    meta = tmp_path / f"meta-{audit}-{int(mutant)}.json"
    args = types.SimpleNamespace(seed=42, port=18123)
    wrap.run(sp, args, meta_path=str(meta), argv_full=["policy_server_wrap.py", "--seed=42"], tokenizer_cls=tok)
    return sink, json.loads(meta.read_text())


def _strip(outs):
    return pickle.dumps([{k: v for k, v in o.items() if k != "_sgeval_audit"} for o in outs])


def test_obs_eq_policy_server_wrap_and_smvla(tmp_path, monkeypatch):
    wrap = load_script("eval-official/policy_server_wrap.py", fresh=True)
    on, meta_on = _run_wrap(wrap, monkeypatch, "1", tmp_path)
    off, meta_off = _run_wrap(wrap, monkeypatch, "0", tmp_path)
    # 除审计键外逐字节相同；随机数状态、推理次数相同；关时没有审计键、开时每次都有
    assert _strip(on["outs"]) == _strip(off["outs"]) and on["rng"] == off["rng"] and on["calls"] == off["calls"] == 3
    assert all("_sgeval_audit" not in o for o in off["outs"]) and all("_sgeval_audit" in o for o in on["outs"])
    assert type(off["policy"]).__name__ == "Policy" and type(on["policy"]).__name__ == "AuditedPolicy"
    # 审计内容与真实分词一致
    a0, a1, a2 = (o["_sgeval_audit"] for o in on["outs"])
    tok = _tokenizer_cls()()
    for aud, obs in zip((a0, a1, a2), OBS_SEQ):
        want = [("task", "pick cube\n" if obs["prompt"] == "pick_cube" else "stack all the blocks\n",
                 tok.tokenize(obs["prompt"], None))]
        if obs["subgoal"] is not None:
            sg = obs["subgoal"]
            want.append(("symbolic", f"Task: {obs['prompt'].replace('_', ' ')};\nCurrent Subgoal: {sg};\nAction: ",
                         tok.tokenize(prompt=obs["prompt"], subgoal=sg)))
        got = aud["channels"]
        assert [c["channel"] for c in got] == [w[0] for w in want]
        for c, (_, text, (ids, mask)) in zip(got, want):
            assert c["text"] == text and c["token_ids"] == [int(x) for x in ids] and c["mask"] == [bool(x) for x in mask]
        assert aud["server_final_text"] == want[-1][1] and aud["pp_generation"] is None
    assert [c["truncated"] for c in a0["channels"]] == [False, False]
    assert [c["truncated"] for c in a1["channels"]] == [False, True]  # 长子目标超过 max_len=64 → 截断
    assert meta_on["policy_seed"] == 42 and meta_on["port"] == 18123 and meta_on["audit"] is True
    assert meta_off["audit"] is False and meta_on["argv"] == ["policy_server_wrap.py", "--seed=42"]
    # 比较器自检：多推理一次的坏外壳被查出（随机数状态、推理次数、动作字节都不同）
    bad, _ = _run_wrap(wrap, monkeypatch, "1", tmp_path, mutant=True)
    assert bad["rng"] != off["rng"] and bad["calls"] != off["calls"] and _strip(bad["outs"]) != _strip(off["outs"])
    monkeypatch.undo()
    # 参数拆分：外壳自用参数摘掉、其余原样
    meta, rest = wrap.split_wrapper_args(["--sgeval-metadata-out=/m.json", "--seed=7", "--port=1", "policy:checkpoint"])
    assert meta == "/m.json" and rest == ["--seed=7", "--port=1", "policy:checkpoint"]

    # smvla：模板化完整 prompt 进审计键；SGEVAL_AUDIT=0/1 回包除审计键外逐字节相同、随机数状态相同
    torch = pytest.importorskip("torch", reason="未验证：torch 未安装")
    msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy", reason="未验证：openpi_client 未安装")
    srv = F.smvla_server()
    template = "<|im_start|>system\nfixed 20 Hz template<|im_end|>\n<|im_start|>user\nThe overall task is: {}<|im_end|>"

    class Proc:
        tokenizer = types.SimpleNamespace(name_or_path="qwen3-vl-fake")

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return template.format(messages)

    class Buf:
        image_keys = ["observation.images.front", "observation.images.wrist"]

        def __init__(self):
            self.processor = Proc()
            self.cache = None

        def reset(self):
            self.cache = None

        def observe(self, full):
            pass

        def _prepare_inputs(self, instruction):
            if self.cache is None or self.cache[0] != instruction:  # 上游提示缓存：同一指令只套一次模板
                self.cache = (instruction, self.processor.apply_chat_template(instruction))
            return {"n": len(self.cache[1])}

    class Batched:
        device = "cpu"

        def generate_batch(self, processed, states):
            noise = torch.randn(3)  # DiT 采样消耗 torch 随机数
            a = np.arange(F.CHUNK_ROWS * 8, dtype=np.float32).reshape(F.CHUNK_ROWS, 8) + float(noise.sum())
            return [(a, f"sub{processed[0]['n']}")]

    def run(audit):
        os.environ["SGEVAL_AUDIT"] = audit
        try:
            h = object.__new__(srv.SMVLAPolicyHost)
            h.policy_seed, h.buffer_factory, h.batched, h.normalize_state = 7, Buf, Batched(), None
            h.to_full, h.state_norm = srv.make_closures(Buf, None, h.batched)
            buf = h.new_episode()
            replies = [h.infer(buf, ins, np.full(8, 0.5, dtype=np.float32)) for ins in ("抓起方块", "抓起方块", "放下")]
            return replies, srv.rng_digest()
        finally:
            os.environ.pop("SGEVAL_AUDIT", None)

    import unittest.mock as um
    with um.patch.object(srv.time, "monotonic", lambda: 100.0):  # infer_ms 固定，回包可逐字节比
        r_on, rng_on = run("1")
        r_off, rng_off = run("0")
    pk = msgpack_numpy.Packer()
    strip = lambda rs: [pk.pack({k: v for k, v in r.items() if k != "_sgeval_audit"}) for r in rs]  # noqa: E731
    assert strip(r_on) == strip(r_off) and rng_on == rng_off
    assert all("_sgeval_audit" not in r for r in r_off)
    texts = [r["_sgeval_audit"]["channels"][0]["text"] for r in r_on]
    assert texts == [template.format("抓起方块"), template.format("抓起方块"), template.format("放下")]
    assert all(r["_sgeval_audit"]["server_final_text"] == t for r, t in zip(r_on, texts))
    assert r_on[0]["_sgeval_audit"]["channels"][0]["tokenizer"] == "qwen3-vl-fake"
    print("OBS_EQ=PASS routes=2 (policy_server_wrap,smvla_server) audit_on_off_mismatch=0 rng_diff=0 "
          "extra_infer=0 mutant_double_infer_detected=1")


# ═══════════════════════════ LEASE_BY_DATASET（1007 FIX-4） ═══════════════════════════


def test_lease_id_includes_dataset_and_shard_content(tmp_path, capsys):
    """拆仓后身份不再按分片切给席位（共享动态队列），lease 只保证「同一路线同一席位名同时只有一个客户端」：lease id 为
    ``<路线>--seat-<席位名>``；两个数据集同在一个队列里，靠队列键 ``<数据集>__<key>`` 区分。同路线不同席位两边都拿得到
    lease、都能跑；同一席位名重复启动，第二个 ``RUN_BLOCKED reason=lease_held``、退出 3、不建环境。（用例名沿用旧口径，
    契约总表按 nodeid 引用。）"""
    bm, ec = bl(), F.env_client()
    route = "groundsg/ground-sg-memer/seed7/new"
    assert ec.shard_id_of(route, "s00") == "groundsg_ground-sg-memer_seed7_new--seat-s00"
    assert ec.shard_id_of(route, "s01") != ec.shard_id_of(route, "s00")
    assert ec.shard_id_of("pp/seed7/new", "host a/1") == "pp_seed7_new--seat-host_a_1"
    # 两个数据集同 key 的身份在同一队列里互不相撞
    task, tier = F.v9_cells_sorted()[0]
    ident = F.packaged_identity(task, tier, 0)
    hv_ident = F.hard0_identity(HARD0_TASK, 0)
    assert ec.queue_key(ident).startswith("ood__") and ec.queue_key(dict(ident, dataset="hard-verify")) != ec.queue_key(ident)

    # 账本层：同路线两个席位的 lease 可同时持有
    p = tmp_path / "shared" / "budget.jsonl"
    with bm.BudgetLedger(p, **CAPS).lease(ec.shard_id_of("pp/seed7/new", "s01")):
        with bm.BudgetLedger(p, **CAPS).lease(ec.shard_id_of("pp/seed7/new", "s00")):
            pass

    # SeatRunner 层：s01 持有 lease 时，s00 照常跑完（同一队列，两数据集各一局）
    stage = tmp_path / "stage"
    s01 = F.make_runner(stage, "pp", _short_policy(), F.World(), seat_dir="s01", policy_seed=7, budget_ledger=str(p),
                        **CAPS)
    s01._open_shared_budget()
    try:
        world_a = F.World()
        a = F.make_runner(stage, "pp", _short_policy(), world_a, seat_dir="s00", policy_seed=7, budget_ledger=str(p),
                          **CAPS)
        assert F.run_rows(a, [ident, hv_ident]) == 0
        out = capsys.readouterr().out
        want = ec.shard_id_of("pp/seed7/new", "s00")
        assert "RUN_BLOCKED" not in out and f"BUDGET_LEASE shard={want}" in out
        assert len(world_a.envs) == 2
        # 同一席位名 s00 重复启动：a 的 lease 已在 run 结束时释放；另起一个持有者占住 s00，再起 b 即被拦
        with bm.BudgetLedger(p, **CAPS).lease(want):
            world_b = F.World()
            b = F.make_runner(tmp_path / "stage-b", "pp", _short_policy(), world_b, seat_dir="s00", policy_seed=7,
                              budget_ledger=str(p), **CAPS)
            assert F.run_rows(b, [ident]) == 3
            assert "RUN_BLOCKED reason=lease_held" in capsys.readouterr().out and world_b.envs == []
            assert not b.results_path.exists()
    finally:
        s01._release_lease()
    print("LEASE_BY_DATASET=PASS two_seats_same_route=both_ok same_seat_dup=blocked queue_key_by_dataset=1")
