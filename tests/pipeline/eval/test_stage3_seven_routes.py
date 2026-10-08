"""第三阶段七路线 CPU 参数化（R4 查缺补漏）：模型种子 0／7／42 与 ood 1800 步边界逐路线真实触发。

依据：``1006-rename-official-names-and-stage3-eval-plan.md`` 第一部分二「模型 seed」「1800 步」、第二部分三节「闸门与
CPU runbook」第二段（七路线都进 1800 边界与 0／7／42 参数化测试，Astra 走零外联夹具）、八.3；接口
``docs/plans/1006-stage3-interface-freeze.md`` 2.1、2.2、四、七节。

七路线：FrameSamp+Modulation、SimpleMemVLA、PonderPounce、GroundSG Oracle、GroundSG QwenVL、MemER（GroundSG 第三变体）、
3-tier Astra。前六条走生产 ``env_client.SeatRunner``（真实 ``EnvSession``、真实客户端模块，只把服务连接、swift 与环境换成
CPU 替身）；Astra 不经过 ``EnvSession``，走 ``astra_hard_runner.run_cases``（真实 ``TracedEnv`` 守卫）与 ``run_astra.sh``
（服务与仿真解释器是只记参数的桩）。全部零 GPU、零外联、零费用、不加载权重。

每条路线、每个种子核三处（期望值由本文件手写，不调用被测函数生成）：

1. **服务命令**：``run_seat.sh::build_server_cmd`` 的 argv 里种子取值等于 ``--policy-seed``（MME-VLA 外壳
   ``--seed=<s>``、smvla ``--policy-seed <s>``、pp ``--args.seed <s>``；Astra 由 ``run_astra.sh`` 起的 VLA 服务 ``--seed=<s>``），
   且没有残留旧常量（7／0／42）；
2. **该路线真正用的随机状态**：MME-VLA 外壳把 argv 的 ``--seed`` 交给三方 ``create_policy``（假 ``serve_policy`` 记下实参）
   并写服务元数据；GroundSG 三变体 ``Args.model_seed``、QwenVL／MemER 构造预测器前 ``seed_everything(<s>)``（Oracle 不调）；
   smvla 服务每局 ``reseed(<s>)``（随机数摘要与手动重置后相同）；
3. **结果与 trace**：结果行 ``policy_seed``、``server_seed``（服务元数据反查）、``budget_token`` 的路线含 ``seed<s>``；
   trace header／identity／end 都记 ``policy_seed``。

1800 边界：``ood``、``--max-steps 1800 --strict-cap``、环境永不终止；第 1800 步进入真实（假）环境，第 1801 步不进入；
结果行 ``status=timeout``、``exec_steps=1800``、``effective_cap=1800``；trace 恰 1800 个 step 行、end 行
``steps_attempted=1800``（有该字段的路线）。PonderPounce 自身循环在 1800 退出，可以不置 ``cap_hit``（计划八.3 末段）；
SimpleMemVLA 的理论循环上界 ``115×16=1840`` 越过 1800，由 ``EnvSession`` 守卫拒第 1801 步。

判定行（``-s`` 可见）：

* ``POLICY_SEEDS=PASS models=7 seeds=0,7,42 cases=21 cpu_only=1``
* ``EVAL_CAP=PASS models=7 dataset=ood max_steps=1800 rejected_step=1801``

1800 步的七路线用例单文件耗时超过 10 s，按测试规约标 ``slow``（``pytest -m slow tests/pipeline/eval`` 执行）。
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import eval_fakes as F
from tests._support.loaders import load_script
from tests.pipeline.evalx.astra import astra_fakes as A
from tests.pipeline.evalx.groundsg import groundsg_fakes as G
from tests.pipeline.evalx.pp import pp_fakes as P

EO = F.REPO / "scripts" / "eval-official"
SEEDS = (0, 7, 42)
CAP = 1800
CAPS = {"trajectory_cap": 870, "shared_infra_cap": 50, "expired_cap": 50, "planned_first_tries": 821}
#: 七条路线（报告名 → SeatRunner 的 --policy、--groundsg-variant）；astra 不走 SeatRunner
ROUTES = {
    "perceptual-framesamp-modul": ("perceptual-framesamp-modul", None),
    "smvla": ("smvla", None),
    "pp": ("pp", None),
    "groundsg-oracle": ("groundsg", G.ORACLE),
    "groundsg-qwenvl": ("groundsg", G.QWENVL),
    "groundsg-memer": ("groundsg", G.MEMER),
    "astra": ("astra", None),
}
SEAT_ROUTES = tuple(r for r in ROUTES if r != "astra")
GROUNDSG_ENV = ("IMAGE_MAX_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "FPS_MAX_FRAMES", "USE_HF", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """被测代码会写 os.environ（官方 GroundSG 定义、swift 开关）；先经 monkeypatch 登记，用例结束后恢复。"""
    for k in (*GROUNDSG_ENV, "SGEVAL_BUDGET_LEDGER", "SGEVAL_EXPIRED_JOBS", "SLURM_JOB_ID", "SLURM_JOB_END_TIME",
              "SGEVAL_AUDIT", "POLICY_SEED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SGEVAL_PP_SERVER_WRAP", "1")  # 本轮 PP 走外壳（回包带 subgoal），trace 记第二阶段字段
    return monkeypatch


# ───────────────────────────── 服务命令（run_seat.sh::build_server_cmd） ─────────────────────────────

LIB_SRV = r'''
set -u
source "$EO/run_seat.sh"
OUT=/o; GPU=0; MME_VLA_PY=/py/mme-vla; PP_PY=/py/pp; SMVLA_PY=/py/smvla; OPENPI_HOME=/openpi; PP_CKPT=/ck/pp
FRAMESAMP_MODUL_CKPT=/ck/fsm; GROUNDSG_CKPT=/ck/sg; SMVLA_CKPT=/ck/smvla; GROUNDSG_VARIANT="$W_VARIANT"
MEMER_ADAPTER="${W_MEMER:-}"; QWENVL_ADAPTER="${W_QWENVL:-}"; POLICY_SEED="$W_SEED"; SGEVAL_PP_SERVER_WRAP=1
build_server_cmd "$W_POL" 18123; echo "BUILD_RC=$?"
for a in "${SRV_ARGV[@]}"; do printf 'SRV %s\n' "$a"; done
'''


def server_argv(route: str, seed: int) -> list[str]:
    pol, variant = ROUTES[route]
    env = dict(os.environ, EO=str(EO), W_POL=pol, W_SEED=str(seed), W_VARIANT=variant or "",
               W_MEMER=G.MEMER_ADAPTER if variant == G.MEMER else "", W_QWENVL=G.ADAPTER if variant == G.QWENVL else "")
    p = subprocess.run(["bash", "-c", LIB_SRV], capture_output=True, text=True, env=env, timeout=60)
    assert "BUILD_RC=0" in p.stdout, (route, seed, p.stdout, p.stderr[-400:])
    return [x[4:] for x in p.stdout.splitlines() if x.startswith("SRV ")]


def _opt(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def seed_in_server_argv(route: str, argv: list[str]) -> int:
    """（手写）按路线从服务 argv 取出种子；同一 argv 里只许出现一处种子参数。"""
    pol, _ = ROUTES[route]
    if pol in ("perceptual-framesamp-modul", "groundsg"):
        vals = [a.split("=", 1)[1] for a in argv if a.startswith("--seed=")]
        assert argv[1].endswith("policy_server_wrap.py") and len(vals) == 1, argv
        return int(vals[0])
    if pol == "smvla":
        assert argv.count("--policy-seed") == 1, argv
        return int(_opt(argv, "--policy-seed"))
    assert pol == "pp" and argv[1].endswith("pp_server_wrap.py") and argv.count("--args.seed") == 1, argv
    return int(_opt(argv, "--args.seed"))


# ───────────────────────────── 服务侧随机状态（外壳与 smvla） ─────────────────────────────


def wrap_create_policy_seed(argv: list[str], meta_path: Path) -> int:
    """以生产外壳 ``policy_server_wrap.run`` 起假三方服务：argv 去掉外壳自用参数后按 ``--seed=`` 交给 ``create_policy``；
    返回假 ``create_policy`` 实收的种子，并写服务元数据（供结果行 ``server_seed`` 反查）。"""
    wrap = load_script("eval-official/policy_server_wrap.py")
    meta, rest = wrap.split_wrapper_args(argv[2:])
    assert meta is not None and meta.endswith("server-metadata-18123.json")
    seen: dict = {}

    def create_policy(a):
        seen["seed"] = a.seed
        return types.SimpleNamespace(metadata={})

    sp = types.SimpleNamespace(create_policy=create_policy, main=lambda a: sp.create_policy(a))
    seed = int(next(x.split("=", 1)[1] for x in rest if x.startswith("--seed=")))
    tok = type("Tok", (), {"tokenize": lambda self, prompt, state=None, subgoal=None: (np.zeros(1), np.ones(1, bool))})
    wrap.run(sp, types.SimpleNamespace(seed=seed, port=18123), meta_path=str(meta_path), argv_full=list(argv[1:]),
             tokenizer_cls=tok)
    return int(seen["seed"])


def smvla_reseed_matches(seed: int) -> bool:
    """smvla 服务每局 ``new_episode`` 用 ``policy_seed`` reseed：随机数摘要与手动 ``reseed(seed)`` 后相同，且与另一种子不同。"""
    srv = F.smvla_server()
    host = object.__new__(srv.SMVLAPolicyHost)
    host.policy_seed = seed
    host.buffer_factory = type("B", (), {"reset": lambda self: None, "image_keys": []})
    host.new_episode()
    got = srv.rng_digest()
    srv.reseed(seed)
    same = got == srv.rng_digest()
    srv.reseed(seed + 1)
    return same and got != srv.rng_digest()


# ───────────────────────────── SeatRunner 夹具（六条 env_client 路线） ─────────────────────────────


class World:
    """SeatRunner 用的假世界：环境取 GroundSG 替身（帧宽 256，官方叠字录像可写；观测键与真实环境相同，六条路线通用）。"""

    def __init__(self, plan: G.Plan):
        self.plan = plan
        self.envs: list = []
        self.builders: list = []
        self.make_calls: list = []
        self.recorders: list = []

    def new_env(self, task: str, ep: int):
        env = G.FakeEnv(task, ep, self.plan)
        self.envs.append(env)
        return env


class PPConn(P.FakeConn):
    """PP 外壳回包：动作块之外带 ``subgoal``（外壳关时没有这个键，客户端会判外壳未生效）。"""

    async def act(self, obs):
        a = await super().act(obs)
        a["subgoal"] = f"pick at [{self.n_actions % 1000}, 500]"
        return a


def policy_module(route: str, monkeypatch, spy: dict):
    """路线 → 交给 SeatRunner 的策略模块（真实客户端模块，只换连接工厂）。"""
    pol, variant = ROUTES[route]
    if pol == "perceptual-framesamp-modul":
        return F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer())
    if pol == "smvla":
        return F.smvla_policy(F.FakePolicyServer())
    if pol == "pp":
        pc = load_script("eval-official/pp_client.py")
        return types.SimpleNamespace(run_episode=functools.partial(pc.run_episode,
                                                                    connection_factory=lambda url, t: PPConn(url, t)))
    mc = G.groundsg_client()
    od = mc.official_defs
    real = od.seed_everything

    def seed_spy(s, _real=real):
        spy.setdefault("seed_everything", []).append(s)
        return _real(s)

    monkeypatch.setattr(od, "seed_everything", seed_spy)
    server, swift = G.FakeServer(), G.FakeSwift()

    def make_policy_context(seat_info):
        ctx = mc.make_policy_context(seat_info, client_factory=lambda h, p, ep: G.FakeClient(server),
                                     qwen_extra=swift.names)
        spy["model_seed"] = ctx["args"].model_seed
        return ctx

    return types.SimpleNamespace(make_policy_context=make_policy_context, run_episode=mc.run_episode)


def seat_runner(route: str, tmp: Path, world: World, monkeypatch, *, seed: int, max_steps: int, spy: dict,
                budget_ledger: Path | None = None, strict_cap: bool = True):
    pol, variant = ROUTES[route]
    kw: dict = dict(dataset="ood", max_steps=max_steps, strict_cap=strict_cap, policy_seed=seed,
                    trace_root=str(tmp / "trace"), reset_budget=None)
    if pol == "groundsg":
        kw.update(groundsg_variant=variant, policy_dir=f"groundsg-{variant}",
                  qwenvl_groundsg_adapter=G.ADAPTER if variant == G.QWENVL else None,
                  memer_adapter=G.MEMER_ADAPTER if variant == G.MEMER else None)
    if budget_ledger is not None:
        kw.update(budget_ledger=str(budget_ledger), **CAPS)
    stage = tmp / "stage"
    return F.make_runner(stage, pol, policy_module(route, monkeypatch, spy), world, **kw)


def ood_identity() -> dict:
    task, tier = F.v9_cells_sorted()[0]
    return F.packaged_identity(task, tier, 0)


def read_trace(runner, ident: dict) -> list[dict]:
    p = Path(runner.trace_root) / f"{ident['key']}.a1" / "trace.jsonl"
    return F.read_jsonl(p)


def route_label(route: str, seed: int) -> str:
    pol, variant = ROUTES[route]
    head = f"groundsg/{variant}" if pol == "groundsg" else pol
    return f"{head}/seed{seed}/new"


# ═════════════════════════════════ POLICY_SEEDS ═════════════════════════════════


def _seat_seed_case(route: str, seed: int, tmp: Path, monkeypatch) -> list[str]:
    """一条 env_client 路线、一个种子；返回不符项（空表示通过）。"""
    bad: list[str] = []
    pol, variant = ROUTES[route]
    argv = server_argv(route, seed)
    if seed_in_server_argv(route, argv) != seed:
        bad.append(f"server_argv {argv}")
    spy: dict = {}
    world = World(G.Plan(success_at=20))
    ledger = tmp / "budget.jsonl"
    runner = seat_runner(route, tmp, world, monkeypatch, seed=seed, max_steps=CAP, spy=spy, budget_ledger=ledger)
    meta = runner.server_metadata_path()
    meta.parent.mkdir(parents=True, exist_ok=True)
    if pol in ("perceptual-framesamp-modul", "groundsg"):
        if wrap_create_policy_seed(argv, meta) != seed:
            bad.append("wrap.create_policy seed")
    else:  # smvla／pp 服务端元数据由真实服务（smvla_server serve／pp_server_wrap）写；这里按其格式写入 argv 里的种子
        meta.write_text(json.dumps({"policy_seed": seed_in_server_argv(route, argv), "argv": argv}))
    if pol == "smvla" and not smvla_reseed_matches(seed):
        bad.append("smvla reseed")
    ident = ood_identity()
    rc = F.run_rows(runner, [ident])
    runner.close()
    rows = F.read_jsonl(runner.results_path)
    if rc != 0 or len(rows) != 1:
        return bad + [f"rc={rc} rows={len(rows)} {[r.get('error') for r in rows]}"]
    row = rows[0]
    if row["status"] != "success":
        bad.append(f"status={row['status']} error={row.get('error')}")
    route_name = route_label(route, seed)
    want = {"policy_seed": seed, "server_seed": seed, "budget_token": f"{route_name}|{ident['key']}|a1",
            "effective_cap": CAP}
    bad += [f"row.{k}={row.get(k)!r}" for k, v in want.items() if row.get(k) != v]
    if pol == "groundsg":
        if row.get("policy_variant") != variant:
            bad.append(f"row.policy_variant={row.get('policy_variant')}")
        if spy.get("model_seed") != seed:
            bad.append(f"Args.model_seed={spy.get('model_seed')}")
        want_se = [] if variant == G.ORACLE else [seed]
        if spy.get("seed_everything", []) != want_se:
            bad.append(f"seed_everything={spy.get('seed_everything')}")
    res = [r for r in F.read_jsonl(ledger) if r["kind"] == "reserve"]
    if [r["route"] for r in res] != [route_name]:
        bad.append(f"ledger routes={[r['route'] for r in res]}")
    tr = read_trace(runner, ident)
    if not tr:
        return bad + ["trace missing"]
    head, end = tr[0], tr[-1]
    if head.get("policy_seed") != seed or head.get("identity", {}).get("policy_seed") != seed \
            or end.get("policy_seed") != seed:
        bad.append(f"trace seeds header={head.get('policy_seed')} identity={head.get('identity', {}).get('policy_seed')} "
                   f"end={end.get('policy_seed')}")
    return bad


class _NullWriter:
    def append_data(self, frame):
        pass

    def close(self):
        pass


STUB_SIM = r"""#!/usr/bin/env bash
for a in "$@"; do
  if [[ "$a" == *astra_hard_runner.py ]]; then printf '%s\n' "$*" >> "$STUB_LOG"; exit 0; fi
done
if [[ "${1:-}" == "-" && "${2:-}" =~ ^[0-9]+$ ]]; then cat >/dev/null; exit 0; fi
exec "$REAL_PY" "$@"
"""
STUB_VLA = r"""#!/usr/bin/env bash
if [[ "${1:-}" == "-" ]]; then cat >/dev/null; exit 0; fi
printf '%s\n' "$*" > "$VLA_LOG"
"""


def astra_launch(tmp: Path, seed: int) -> tuple[subprocess.CompletedProcess, Path, Path, Path]:
    """``run_astra.sh`` 以桩解释器起：VLA 服务只记 argv、runner 只记命令行；零外联（假 key 不会被用到）。"""
    stubs = tmp / "stubs"
    stubs.mkdir(parents=True, exist_ok=True)
    sim, vla, lib = stubs / "sim.sh", stubs / "vla.sh", stubs / "lib.sh"
    sim.write_text(STUB_SIM)
    vla.write_text(STUB_VLA)
    lib.write_text("render_official_dir() { :; }\ntranscode_episode_dir() { :; }\n")
    sim.chmod(0o755)
    vla.chmod(0o755)
    (tmp / "astra-root").mkdir(exist_ok=True)
    cases = A.write_cases(tmp / "cases.json", {"dataset": "ood", "cases": []})
    run = tmp / "group_0" / "run"
    log, vla_log = tmp / "runner-argv.log", tmp / "vla-argv.log"
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.update(VLA_PYTHON=str(vla), SIM_PYTHON=str(sim), REAL_PY=sys.executable, STUB_LOG=str(log),
               VLA_LOG=str(vla_log), VLA_CHECKPOINT=str(tmp / "ckpt"), MONITOR_BASE=str(tmp / "base"),
               MONITOR_ADAPTER=str(tmp / "adapter"), MAX_STEPS=str(CAP),
               OPENAI_API_KEY="sk-test-placeholder-not-a-real-key", ASTRA_GUARD_STATE=str(tmp / "g.json"),
               ASTRA_ROOT=str(tmp / "astra-root"), SEAT_MEDIA_LIB=str(lib), VLA_GPU="0", MONITOR_GPU="1",
               PORT="18999")
    proc = subprocess.run(["bash", str(EO / "run_astra.sh"), "--policy-seed", str(seed), str(cases), str(run)],
                          capture_output=True, text=True, env=env, timeout=120)
    return proc, run, log, vla_log


def _a_dir(output: Path, task: str, ep: int) -> Path:
    dirs = [p for p in (Path(output) / task / f"ep{ep:03d}").iterdir() if p.is_dir() and ".a" in p.name]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _astra_seed_case(seed: int, tmp: Path, monkeypatch) -> list[str]:
    bad: list[str] = []
    proc, run, log, vla_log = astra_launch(tmp, seed)
    if proc.returncode != 0:
        return [f"run_astra.sh rc={proc.returncode} {proc.stderr[-300:]}"]
    argv = vla_log.read_text().split()
    seeds = [a for a in argv if a.startswith("--seed=")]
    if seeds != [f"--seed={seed}"]:
        bad.append(f"vla argv seeds={seeds}")
    calls = log.read_text().splitlines()
    seeded = [c for c in calls if f" check " in f" {c} " or f" run " in f" {c} "]  # summarize 不读种子
    if len(seeded) != 2 or not all(c.endswith(f"--policy-seed {seed}") or f"--policy-seed {seed} " in c
                                   for c in seeded):
        bad.append(f"runner check/run calls={seeded}")
    meta = json.loads((run / "server-metadata-18999.json").read_text())
    if meta.get("policy_seed") != seed or meta.get("cloud_seed") is not None:
        bad.append(f"server meta={meta}")
    net = A.NetCounter().install(monkeypatch)
    with A.astra_session() as (mod, astra):
        cls = A.recording_builder_cls(lambda b, ep: A.FakeEnv(terminal_step=20))
        doc = mod.prepare_cases(cls, "hard-verify", ["BinFill"], source_episodes=[3])
        args = A.make_args(tmp, A.write_cases(tmp / "c.json", doc), max_steps=1300, policy_seed=seed)
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        deps = A.make_deps(astra, cls, monitor=A.FakeMonitor(), vla=A.FakeVLA(),
                           responder=A.FakeResponder(astra.champ), check_calls=[])
        (result,) = mod.run_cases(args, deps)["results"]
    a_dir = _a_dir(Path(args.output), "BinFill", 0)
    rows = F.read_jsonl(a_dir / "trace.jsonl")
    if rows[0]["identity"].get("policy_seed") != seed or rows[-1].get("policy_seed") != seed:
        bad.append("astra trace policy_seed")
    if rows[0].get("policy_seed", seed) != seed:
        bad.append("astra trace header policy_seed")
    if result.get("policy_seed") != seed or result.get("cloud_seed") is not None:
        bad.append(f"astra result policy_seed={result.get('policy_seed')}")
    prov = json.loads((a_dir / "provenance.json").read_text())
    if prov.get("server_seed") != seed:
        bad.append(f"astra provenance server_seed={prov.get('server_seed')}")
    if net.calls != 0:
        bad.append(f"net calls={net.calls}")
    return bad


@pytest.mark.slow
def test_policy_seeds_seven_routes_three_seeds(tmp_path, monkeypatch):
    """七路线 × 种子 0／7／42 共 21 例：种子到达服务命令、该路线真正用的随机状态、结果行与 trace。"""
    failures: dict = {}
    cases = 0
    for route in ROUTES:
        for seed in SEEDS:
            sub = tmp_path / route / f"seed{seed}"
            sub.mkdir(parents=True)
            with monkeypatch.context() as mp:
                bad = _astra_seed_case(seed, sub, mp) if route == "astra" else _seat_seed_case(route, seed, sub, mp)
            cases += 1
            print(f"POLICY_SEED_CASE route={route} seed={seed} {'ok' if not bad else 'BAD ' + '; '.join(bad)}")
            if bad:
                failures[(route, seed)] = bad
    assert not failures, failures
    assert cases == len(ROUTES) * len(SEEDS) == 21
    print(f"POLICY_SEEDS=PASS models={len(ROUTES)} seeds={','.join(map(str, SEEDS))} cases={cases} cpu_only=1")


# ═════════════════════════════════ EVAL_CAP ═════════════════════════════════


def _seat_cap_case(route: str, tmp: Path, monkeypatch) -> tuple[list[str], dict]:
    pol, variant = ROUTES[route]
    world = World(G.Plan())  # 永不终止
    runner = seat_runner(route, tmp, world, monkeypatch, seed=7, max_steps=CAP, spy={})
    ident = ood_identity()
    rc = F.run_rows(runner, [ident])
    rows = F.read_jsonl(runner.results_path)
    bad: list[str] = []
    if rc != 0 or len(rows) != 1:
        return [f"rc={rc} rows={[(r.get('status'), r.get('error')) for r in rows]}"], {}
    row = rows[0]
    (env,) = world.envs
    info = {"env_steps": len(env.actions), "status": row["status"], "exec_steps": row["exec_steps"],
            "cap_hit": row["cap_hit"], "effective_cap": row["effective_cap"]}
    if len(env.actions) != CAP:
        bad.append(f"环境实收 {len(env.actions)} 步（应恰 {CAP}：第 {CAP} 步进入、第 {CAP + 1} 步不进入）")
    if (row["status"], row["exec_steps"], row["effective_cap"], row["infra"]) != ("timeout", CAP, CAP, False):
        bad.append(f"row={info} infra={row['infra']} error={row.get('error')}")
    if pol != "pp" and not row["cap_hit"]:  # PP 自身循环在 1800 退出、不触发守卫（计划八.3）
        bad.append("cap_hit=False（第 1801 步应被 EnvSession 守卫拒绝）")
    tr = read_trace(runner, ident)
    steps = [r for r in tr if r.get("kind") == "step"]
    if len(steps) != CAP or steps[-1].get("step") != CAP:
        bad.append(f"trace step rows={len(steps)} last={steps[-1].get('step') if steps else None}")
    end = tr[-1]
    if end.get("status") != "timeout":
        bad.append(f"trace end status={end.get('status')}")
    if "steps_attempted" in end and end["steps_attempted"] != CAP:
        bad.append(f"trace end steps_attempted={end['steps_attempted']}")
    if tr[0].get("effective_cap") != CAP:
        bad.append(f"trace header effective_cap={tr[0].get('effective_cap')}")
    return bad, info


def _astra_cap_case(tmp: Path, monkeypatch) -> tuple[list[str], dict]:
    """Astra：自身循环被改成以为上限 1900（模拟越界），入口守卫在第 1801 步前拒绝。"""
    net = A.NetCounter().install(monkeypatch)
    with A.astra_session() as (mod, astra):
        envs = []

        def plan(builder, ep):
            envs.append(A.FakeEnv(terminal_step=None))
            return envs[-1]

        cls = A.recording_builder_cls(plan)
        doc = mod.prepare_cases(cls, "ood", ["VideoUnmask"], tier="xhard1", index=0)
        args = A.make_args(tmp, A.write_cases(tmp / "cases.json", doc), max_steps=CAP, policy_seed=7)
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        real_episode = astra.runner.episode

        def overrun(run_args, *rest):
            return real_episode(types.SimpleNamespace(**{**vars(run_args), "max_steps": CAP + 100}), *rest)

        monkeypatch.setattr(astra.runner, "episode", overrun)
        deps = A.make_deps(astra, cls, monitor=A.FakeMonitor(), vla=A.FakeVLA(),
                           responder=A.FakeResponder(astra.champ), check_calls=[])
        (result,) = mod.run_cases(args, deps)["results"]
    info = {"env_steps": envs[0].steps_taken, "status": result["status"], "exec_steps": result["exec_steps"],
            "cap_hit": result["cap_hit"], "effective_cap": result["effective_cap"]}
    bad = []
    if envs[0].steps_taken != CAP:
        bad.append(f"环境实收 {envs[0].steps_taken} 步")
    if (result["status"], result["exec_steps"], result["cap_hit"], result["effective_cap"]) != ("timeout", CAP, True, CAP):
        bad.append(f"result={info}")
    a_dir = _a_dir(Path(args.output), "VideoUnmask", doc["cases"][0]["episode"])
    steps = [r for r in F.read_jsonl(a_dir / "trace.jsonl") if r["kind"] == "step"]
    if len(steps) != CAP:
        bad.append(f"trace step rows={len(steps)}")
    if net.calls:
        bad.append(f"net calls={net.calls}")
    return bad, info


@pytest.mark.slow
def test_eval_cap_1800_seven_routes(tmp_path, monkeypatch):
    """七路线 ood 1800：第 1800 步进入环境、第 1801 步不进入，终态 timeout、exec_steps=1800、effective_cap=1800。"""
    failures: dict = {}
    for route in ROUTES:
        sub = tmp_path / route
        sub.mkdir()
        with monkeypatch.context() as mp:
            bad, info = _astra_cap_case(sub, mp) if route == "astra" else _seat_cap_case(route, sub, mp)
        print(f"EVAL_CAP_CASE route={route} {info} {'ok' if not bad else 'BAD ' + '; '.join(bad)}")
        if bad:
            failures[route] = bad
    assert not failures, failures
    # 对照（证明本判据有区分力）：同一 FrameSamp+Modulation 路线去掉 strict cap，客户端自己的循环会把第 1801 步送进环境
    ctl_world = World(G.Plan())
    ctl = seat_runner("perceptual-framesamp-modul", tmp_path / "control", ctl_world, monkeypatch, seed=7,
                      max_steps=CAP, spy={}, strict_cap=False)
    assert F.run_rows(ctl, [ood_identity()]) == 0
    assert len(ctl_world.envs[0].actions) == CAP + 1, "对照失效：不带 strict cap 时第 1801 步应进入环境"
    print(f"EVAL_CAP=PASS models={len(ROUTES)} dataset=ood max_steps={CAP} rejected_step={CAP + 1} "
          f"control_without_strict_cap_entered={CAP + 1}")


# ═════════════════════════════════ OBS_EQ（观察关闭／开启等价 + 三类突变被拒） ═════════════════════════════════
#
# 计划第二部分三节：观察开关两侧「实际输入 token id／mask、动作字节、RNG、模型调用次数」逐字节相同，只允许审计字段不同；
# 多抽随机数、多推理一次、改一个 token 的反例必须被拒。``test_stage3_entry_budget.py::test_obs_eq_policy_server_wrap_and_smvla``
# 已核两种外壳的开关等价与「多推理一次」；本节补：同一个比较器对三类突变都给出不等（外壳与 smvla 服务各三类）。
# 覆盖路线：MME-VLA 外壳 ``policy_server_wrap.py`` 承载 FrameSamp+Modulation 与 GroundSG 三变体（Oracle／QwenVL／MemER）的
# 动作服务（上面 POLICY_SEEDS 用例逐路线核了服务 argv[1] 是该外壳），smvla 服务承载 SimpleMemVLA，共 5 条路线；PonderPounce
# 外壳的同类对照在 ``tests/pipeline/evalx/pp/test_pp_server_wrap.py``（``PP_AUDIT_OBS_EQ``，需 client-env 子进程）；Astra 的
# VLA 服务直接起三方 ``serve_policy.py``、没有服务端观察层，不适用。

OBS_ROUTES = ("perceptual-framesamp-modul", "groundsg-oracle", "groundsg-qwenvl", "groundsg-memer", "smvla")


class _SPProc:
    """sentencepiece 处理器替身：每个字符一个 id（bos=1）。"""

    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [ord(ch) % 97 + 2 for ch in text]


def _tok_cls():
    """与三方 PaligemmaTokenizer.tokenize 同签名的替身类（每次新建，外壳的观察补丁各套各的）。"""

    class Tok:
        def __init__(self, max_len=48):
            self._max_len = max_len
            self._tokenizer = _SPProc()

        def tokenize(self, prompt, state=None, subgoal=None):
            text = prompt if subgoal is None else f"Task: {prompt};\nCurrent Subgoal: {subgoal};\nAction: "
            ids = self._tokenizer.encode(text, add_bos=True)[: self._max_len]
            mask = [True] * len(ids) + [False] * (self._max_len - len(ids))
            return np.asarray(ids + [0] * (self._max_len - len(ids))), np.asarray(mask)
    return Tok


OBS_SEQ = [{"prompt": "pick cube", "subgoal": "grasp the red cube"}, {"prompt": "pick cube", "subgoal": None},
           {"prompt": "stack all blocks", "subgoal": "place on top"}]


def _wrap_run(monkeypatch, audit: str, mutant: str | None, tmp: Path) -> dict:
    """生产外壳 ``run`` 起假三方服务（每次新载一份外壳、新建分词类）；返回模型实际收到的 token、回包、RNG、调用次数。"""
    monkeypatch.setenv("SGEVAL_AUDIT", audit)
    wrap = load_script("eval-official/policy_server_wrap.py", fresh=True)
    tok = _tok_cls()
    sink: dict = {}

    class Policy:
        def __init__(self, seed):
            self.tok, self.rng, self.calls, self.inputs, self.metadata = tok(), np.random.default_rng(seed), 0, [], {}

        def infer(self, obs):
            self.calls += 1
            ids, mask = self.tok.tokenize(obs["prompt"], None)
            self.inputs.append((np.asarray(ids).tobytes(), np.asarray(mask).tobytes()))
            if obs["subgoal"] is not None:
                sids, smask = self.tok.tokenize(prompt=obs["prompt"], subgoal=obs["subgoal"], state=None)
                self.inputs.append((np.asarray(sids).tobytes(), np.asarray(smask).tobytes()))
                ids = sids
            noise = self.rng.standard_normal(4)
            return {"actions": (noise + np.asarray(ids[:4], dtype=np.float64) * 0.01).astype(np.float32)}

    sp = types.SimpleNamespace(create_policy=lambda a: Policy(a.seed))

    def main(a):
        pol = sp.create_policy(a)
        sink["outs"] = [pol.infer(o) for o in OBS_SEQ]
        inner = pol.__dict__.get("_inner", pol)
        sink.update(rng=inner.rng.bit_generator.state, calls=inner.calls, inputs=list(inner.inputs))
    sp.main = main

    if mutant == "extra_rng":  # 外壳在观察时多抽一次模型的随机数
        class ExtraRng(wrap.AuditedPolicy):
            def infer(self, obs):
                self._inner.rng.standard_normal(1)
                return super().infer(obs)
        monkeypatch.setattr(wrap, "AuditedPolicy", ExtraRng)
    elif mutant == "extra_infer":  # 外壳多推理一次
        class Twice(wrap.AuditedPolicy):
            def infer(self, obs):
                self._inner.infer(obs)
                return super().infer(obs)
        monkeypatch.setattr(wrap, "AuditedPolicy", Twice)
    elif mutant == "token_change":  # 外壳的分词观察改了一个 token
        real_spy = wrap.install_tokenizer_spy

        def bad_spy(cls):
            real_spy(cls)
            inner_tokenize = cls.tokenize

            def tokenize(self, *a, **k):
                ids, mask = inner_tokenize(self, *a, **k)
                ids = np.array(ids, copy=True)
                ids[1] += 1
                return ids, mask
            cls.tokenize = tokenize
        monkeypatch.setattr(wrap, "install_tokenizer_spy", bad_spy)
    wrap.run(sp, types.SimpleNamespace(seed=7, port=1), meta_path=str(tmp / f"meta-{audit}-{mutant}.json"),
             argv_full=["policy_server_wrap.py", "--seed=7"], tokenizer_cls=tok)
    sink["outs_bytes"] = [{k: np.asarray(v).tobytes() for k, v in o.items() if k != "_sgeval_audit"}
                          for o in sink["outs"]]
    return sink


def obs_eq_problems(on: dict, off: dict) -> list[str]:
    """（手写）比较器：开／关两侧在模型实际输入、动作字节、RNG、调用次数四项上的不等项；空表示等价。"""
    pairs = (("inputs", on["inputs"], off["inputs"]), ("actions", on["outs_bytes"], off["outs_bytes"]),
             ("rng", on["rng"], off["rng"]), ("calls", on["calls"], off["calls"]))
    return [name for name, a, b in pairs if a != b]


def _smvla_run(audit: str, mutant: str | None) -> dict:
    """smvla 服务（真实 ``SMVLAPolicyHost.infer``／``new_episode``；模型与 processor 为替身）。"""
    import unittest.mock as um

    import torch

    srv = F.smvla_server()
    template = "<|im_start|>user\nThe overall task is: {}<|im_end|>"
    seen: dict = {"inputs": [], "calls": 0}

    class Proc:
        tokenizer = types.SimpleNamespace(name_or_path="qwen3-vl-fake")

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return template.format(messages)

    class Buf:
        image_keys = ["observation.images.front", "observation.images.wrist"]

        def __init__(self):
            self.processor, self.cache = Proc(), None

        def reset(self):
            self.cache = None

        def observe(self, full):
            pass

        def _prepare_inputs(self, instruction):
            if self.cache is None or self.cache[0] != instruction:  # 上游提示缓存：同一指令只套一次模板
                self.cache = (instruction, self.processor.apply_chat_template(instruction))
            return {"text": self.cache[1]}

    class Batched:
        device = "cpu"

        def generate_batch(self, processed, states):
            seen["calls"] += 1
            seen["inputs"].append(processed[0]["text"])
            noise = torch.randn(3)  # DiT 采样消耗 torch 随机数
            a = np.arange(F.CHUNK_ROWS * 8, dtype=np.float32).reshape(F.CHUNK_ROWS, 8) + float(noise.sum())
            return [(a, f"sub{len(processed[0]['text'])}")]

    host_cls = srv.SMVLAPolicyHost
    patches = []
    if mutant == "extra_rng":  # 观察时多抽一次 torch 随机数
        real_w0 = host_cls._watch_prompt
        patches.append(um.patch.object(host_cls, "_watch_prompt", lambda self, buf: (torch.randn(1), real_w0(self, buf))))
    elif mutant == "extra_infer":  # 组审计块时多推理一次
        real_b = host_cls._audit_block

        def twice(self, buf):
            self.batched.generate_batch([buf._prepare_inputs("抓起方块")], [None])
            return real_b(self, buf)
        patches.append(um.patch.object(host_cls, "_audit_block", twice))
    elif mutant == "token_change":  # 观察挂钩改了模板化 prompt 的一个字符
        real_w = host_cls._watch_prompt

        def bad_watch(self, buf):
            real_w(self, buf)
            proc = buf.processor
            if not getattr(proc, "_mut", False):
                inner = proc.apply_chat_template
                proc.apply_chat_template = lambda *a, **k: inner(*a, **k) + "!"
                proc._mut = True
        patches.append(um.patch.object(host_cls, "_watch_prompt", bad_watch))
    patches.append(um.patch.object(srv.time, "monotonic", lambda: 100.0))  # infer_ms 固定，回包可逐字节比
    old = os.environ.get("SGEVAL_AUDIT")
    os.environ["SGEVAL_AUDIT"] = audit
    try:
        for p in patches:
            p.start()
        h = object.__new__(host_cls)
        h.policy_seed, h.buffer_factory, h.batched, h.normalize_state = 7, Buf, Batched(), None
        h.to_full, h.state_norm = srv.make_closures(Buf, None, h.batched)
        buf = h.new_episode()
        replies = [h.infer(buf, ins, np.full(8, 0.5, dtype=np.float32)) for ins in ("抓起方块", "抓起方块", "放下")]
        rng = srv.rng_digest()
    finally:
        for p in reversed(patches):
            p.stop()
        if old is None:
            os.environ.pop("SGEVAL_AUDIT", None)
        else:
            os.environ["SGEVAL_AUDIT"] = old
    outs = [{k: np.asarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v) for k, v in r.items()
             if k != "_sgeval_audit"} for r in replies]
    return {"inputs": seen["inputs"], "outs_bytes": outs, "rng": rng, "calls": seen["calls"]}


def test_obs_eq_on_off_equal_and_three_mutants_rejected(tmp_path, monkeypatch):
    """观察开／关两侧四项全等；多抽随机数、多推理一次、改一个 token 三类突变各自被比较器拒（外壳与 smvla 各三类）。"""
    rejected = 0
    off = _wrap_run(monkeypatch, "0", None, tmp_path)
    on = _wrap_run(monkeypatch, "1", None, tmp_path)
    assert obs_eq_problems(on, off) == [], obs_eq_problems(on, off)
    assert off["calls"] == 3 and len(off["inputs"]) == 5
    want = {"extra_rng": {"rng", "actions"}, "extra_infer": {"calls", "rng"}, "token_change": {"inputs", "actions"}}
    for mutant, must in want.items():
        got = set(obs_eq_problems(_wrap_run(monkeypatch, "1", mutant, tmp_path), off))
        assert must <= got, (mutant, got)
        rejected += 1
    monkeypatch.delenv("SGEVAL_AUDIT", raising=False)
    s_off, s_on = _smvla_run("0", None), _smvla_run("1", None)
    assert obs_eq_problems(s_on, s_off) == [] and s_off["calls"] == 3
    want_s = {"extra_rng": {"rng", "actions"}, "extra_infer": {"calls", "rng"}, "token_change": {"inputs"}}
    for mutant, must in want_s.items():
        got = set(obs_eq_problems(_smvla_run("1", mutant), s_off))
        assert must <= got, ("smvla", mutant, got)
        rejected += 1
    print(f"OBS_EQ=PASS routes={len(OBS_ROUTES)} ({','.join(OBS_ROUTES)}) wrappers=policy_server_wrap,smvla_server "
          f"on_off_mismatch=0 mutants_rejected={rejected}/6 pp=see_PP_AUDIT_OBS_EQ astra=no_server_observer")
