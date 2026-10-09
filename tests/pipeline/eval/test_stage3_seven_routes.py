"""第三阶段七路线 CPU 参数化（R4 查缺补漏）：模型种子 0／7／42 与 ood 1800 步边界逐路线真实触发。

依据：``1006-rename-official-names-and-stage3-eval-plan.md`` 第一部分二「模型 seed」「1800 步」、第二部分三节「闸门与
CPU runbook」第二段（七路线都进 1800 边界与 0／7／42 参数化测试，Astra 走零外联夹具）、八.3；接口
``docs/plans/1006-stage3-interface-freeze.md`` 2.1、2.2、四、七节。

七路线：FrameSamp+Modulation、SimpleMemVLA、PonderPounce、GroundSG Oracle、GroundSG QwenVL、MemER（GroundSG 第三变体）、
3-tier Astra。前六条走生产 GL 席位 ``dev-scripts/gl/seat.py::SeatRunner``（常驻 Policy，真实 ``EnvSession``、真实客户端
模块经 ``eval_fakes.ModulePolicy`` 包成 Policy，只把服务连接、swift 与环境换成 CPU 替身）；Astra 走 ``load_policy("astra")``
的 ``AstraPolicy``（真实 ``TracedEnv`` 守卫，服务端与 VLA 等为 ``astra_fakes.Harness`` 的替身，拆仓后 ``run_astra.sh`` 不迁）。全部零 GPU、零外联、零费用、不加载权重。

每条路线、每个种子核三处（期望值由本文件手写，不调用被测函数生成）：

1. **服务命令**：各模型 Policy 的服务端命令（``models/*::*_server_spec``，即旧 ``run_seat.sh::build_server_cmd`` 的
   Python 版；``run_seat.sh`` 不迁）的 argv 里种子取值等于 ``policy_seed``（MME-VLA 外壳
   ``--seed=<s>``、smvla ``--policy-seed <s>``、pp ``--args.seed <s>``；Astra 由 ``AstraPolicy.load`` 起的 VLA 服务 ``--seed=<s>``），
   且没有残留旧常量（7／0／42）；
2. **该路线真正用的随机状态**：MME-VLA 外壳把 argv 的 ``--seed`` 交给三方 ``create_policy``（假 ``serve_policy`` 记下实参）
   并写服务元数据；GroundSG 三变体 ``Args.model_seed``、QwenVL／MemER 构造预测器前 ``seed_everything(<s>)``（Oracle 不调）；
   smvla 服务每局 ``reseed(<s>)``（随机数摘要与手动重置后相同）；
3. **结果与 trace**：结果行 ``policy_seed``、``server_seed``（服务元数据反查）、``budget_token`` 的路线含 ``seed<s>``；
   trace header／identity／end 都记 ``policy_seed``。

1800 边界：``ood``（拆仓后上限与 strict 只由数据集决定）、环境永不终止；第 1800 步进入真实（假）环境，第 1801 步不进入；
席位结果行 ``status=timeout``、``exec_steps=1800``，局结果 ``max_steps=1800``、``strict_cap=true``，trace 头
``effective_cap=1800``；trace 恰 1800 个 step 行、end 行
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
import types
from pathlib import Path

import numpy as np
import pytest

import eval_fakes as F
from tests._support.loaders import load_script
from tests.pipeline.evalx.astra import astra_fakes as A
from tests.pipeline.evalx.groundsg import groundsg_fakes as G
from tests.pipeline.evalx.pp import pp_fakes as P

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


# ───────────────────────────── 服务命令（各模型 ``*_server_spec``） ─────────────────────────────

#: 旧 LIB_SRV 的固定参数（解释器与 ckpt 都是只记不跑的假路径）
SRV_CFG = {"perceptual-framesamp-modul": {"mme_vla_py": "/py/mme-vla", "ckpt": "/ck/fsm"},
           "groundsg": {"mme_vla_py": "/py/mme-vla", "ckpt": "/ck/sg"},
           "smvla": {"smvla_py": "/py/smvla", "ckpt": "/ck/smvla"},
           "pp": {"pp_py": "/py/pp", "ckpt": "/ck/pp", "pp_server_wrap": True}}


def server_argv(route: str, seed: int) -> list[str]:
    """该路线 Policy 的服务端 argv（端口 18123，不起服务）。"""
    from robomme_ood_eval.models import framesamp_modul as fm
    from robomme_ood_eval.models import pp as ppm
    from robomme_ood_eval.models import resolve
    from robomme_ood_eval.models import smvla as smm

    pol, variant = ROUTES[route]
    cfg = dict(SRV_CFG[pol], port=18123, openpi_data_home="/openpi", preflight=False, ckpt_fingerprint=False,
               server_dir="/o")
    if pol == "groundsg":
        cfg.update(groundsg_variant=variant, memer_adapter_path=G.MEMER_ADAPTER if variant == G.MEMER else None,
                   qwenvl_groundSG_adapter_path=G.ADAPTER if variant == G.QWENVL else None)
    p = resolve(pol)(policy_seed=seed, **cfg)
    p._pick_port()
    build = {"perceptual-framesamp-modul": fm.mme_vla_server_spec, "groundsg": fm.mme_vla_server_spec,
             "smvla": smm.smvla_server_spec, "pp": ppm.pp_server_spec}[pol]
    argv, _env, _cwd = build(p, SRV_CFG[pol]["ckpt"])
    return argv


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
    assert meta is not None and meta.endswith("server-wrap-metadata-18123.json")
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

    # 拆仓后经 eval_fakes.ModulePolicy 包成常驻 Policy：变体与 adapter 由 conn_extra 并入上下文入参与每局 conn_info
    # （与真实 GroundSGPolicy.load／play 交给官方代码的键相同）
    extra = {"groundsg_variant": variant,
             "qwenvl_groundSG_adapter_path": G.ADAPTER if variant == G.QWENVL else None,
             "memer_adapter_path": G.MEMER_ADAPTER if variant == G.MEMER else None}
    return types.SimpleNamespace(make_policy_context=make_policy_context, run_episode=mc.run_episode,
                                 conn_extra=extra)


def seat_runner(route: str, tmp: Path, world: World, monkeypatch, *, seed: int, max_steps: int, spy: dict,
                budget_ledger: Path | None = None):
    """拆仓后的 GL 席位（常驻 Policy + 动态队列）。步数上限与是否 strict 只由身份行的数据集决定（ood 1800 strict、
    hard-verify 1300 不 strict），``max_steps`` 只作调用方核对用。"""
    pol, variant = ROUTES[route]
    assert max_steps == CAP
    kw: dict = dict(policy_seed=seed, reset_budget=None)
    if pol == "groundsg":
        kw.update(groundsg_variant=variant,
                  qwenvl_groundsg_adapter=G.ADAPTER if variant == G.QWENVL else None,
                  memer_adapter=G.MEMER_ADAPTER if variant == G.MEMER else None)
    if budget_ledger is not None:
        kw.update(budget_ledger=str(budget_ledger), **CAPS)
    stage = tmp / "stage"
    return F.make_runner(stage, pol, policy_module(route, monkeypatch, spy), world, **kw)


def ood_identity() -> dict:
    task, tier = F.v9_cells_sorted()[0]
    return F.packaged_identity(task, tier, 0)


def raw_dir(runner, ident: dict, attempt: int = 1) -> Path:
    """该身份该次尝试的本局 raw 目录（拆仓后轨迹、录制器产物与 result.json 都在这里）。"""
    rows = [r for r in F.read_jsonl(runner.results_path) if r["key"] == ident["key"] and r["attempt"] == attempt]
    assert rows, (ident["key"], attempt)
    return Path(rows[-1]["result"]).parent


def read_result(runner, ident: dict) -> dict:
    return json.loads((raw_dir(runner, ident) / "result.json").read_text(encoding="utf-8"))


def read_trace(runner, ident: dict) -> list[dict]:
    return F.read_jsonl(raw_dir(runner, ident) / "trace.jsonl")


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
    meta = tmp / "server-metadata.json"  # 拆仓后服务端元数据由 Policy 持有的服务端进程读，这里按其格式写一份交给假服务端
    if pol in ("perceptual-framesamp-modul", "groundsg"):
        if wrap_create_policy_seed(argv, meta) != seed:
            bad.append("wrap.create_policy seed")
    else:  # smvla／pp 服务端元数据由真实服务（smvla_server serve／pp_server_wrap）写；这里按其格式写入 argv 里的种子
        meta.write_text(json.dumps({"policy_seed": seed_in_server_argv(route, argv), "argv": argv}))
    if pol == "smvla" and not smvla_reseed_matches(seed):
        bad.append("smvla reseed")
    srv = types.SimpleNamespace(metadata=json.loads(meta.read_text(encoding="utf-8")), check=lambda: None,
                                stop=lambda **k: "stopped", left_line=lambda: "")
    runner.fake_policy.servers = lambda: [srv]  # Policy.server_seed 从服务端元数据反查
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
    want = {"policy_seed": seed, "budget_token": f"{route_name}|{ident['key']}|a1"}
    bad += [f"row.{k}={row.get(k)!r}" for k, v in want.items() if row.get(k) != v]
    # 局结果 result.json：种子与服务端反查种子、步数口径（拆仓后 effective_cap 不进结果行，改核 max_steps／strict_cap）
    res = read_result(runner, ident)
    want_res = {"policy_seed": seed, "server_seed": seed, "max_steps": CAP, "strict_cap": True}
    bad += [f"result.{k}={res.get(k)!r}" for k, v in want_res.items() if res.get(k) != v]
    if pol == "groundsg":
        if res.get("policy_variant") != variant:
            bad.append(f"result.policy_variant={res.get('policy_variant')}")
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


def find_episode(builder_cls, task: str, dataset: str, *, tier: str | None = None) -> int:
    """真实身份解析里第一个符合档位的 builder 局号（ood 的 xhard1 局用）。"""
    b = builder_cls(task, dataset=dataset, action_space="joint_angle", gui_render=False,
                    max_steps={"hard-verify": 1300, "ood": CAP}[dataset])
    for ep in range(b.get_episode_num()):
        if tier is None or b.resolve_identity(ep)["tier"] == tier:
            return ep
    raise AssertionError(f"{task} {dataset} 找不到 tier={tier}")


def astra_run(tmp: Path, monkeypatch, *, seed: int, dataset: str, task: str, tier: str | None = None,
              env_plan=None, overrun: int | None = None):
    """拆仓后的 Astra 路线：经 ``load_policy("astra", seed)`` 建全替身 ``AstraPolicy``（服务端、VLA、规划、监视都是
    ``astra_fakes.Harness`` 的替身，零外联、零 GPU），外层 ``run_episode`` 跑一局。``overrun`` 给出时把 Astra 自己的循环
    上限改大（模拟越界）。返回 (harness, 结果, 外联计数器)。旧 ``run_astra.sh``／``astra_hard_runner.run_cases`` 已不迁。"""
    net = A.NetCounter().install(monkeypatch)
    with A.astra_session() as (mod, astra):
        h = A.Harness(tmp, monkeypatch, mod, astra, env_plan=env_plan or (lambda b, ep: A.FakeEnv(terminal_step=20)))
        if overrun is not None:
            real_episode = astra.runner.episode

            def over(run_args, *rest):
                return real_episode(types.SimpleNamespace(**{**vars(run_args), "max_steps": overrun}), *rest)

            monkeypatch.setattr(astra.runner, "episode", over)
        h.load(seed=seed)
        ep = find_episode(h.cls, task, dataset, tier=tier) if tier else 0
        r = h.run(dataset, task, ep)
        h.policy.close()
    return h, r, net


# ═════════════════════════════════ POLICY_SEEDS（Astra） ═════════════════════════════════


def _astra_seed_case(seed: int, tmp: Path, monkeypatch) -> list[str]:
    bad: list[str] = []
    h, r, net = astra_run(tmp, monkeypatch, seed=seed, dataset="hard-verify", task="BinFill")
    vla = h.servers()["astra-vla"]
    seeds = [x for x in vla.argv if str(x).startswith("--seed=")]
    if seeds != [f"--seed={seed}"]:
        bad.append(f"vla argv seeds={seeds}")
    if vla.metadata.get("policy_seed") != seed or vla.metadata.get("cloud_seed") is not None:
        bad.append(f"server meta={vla.metadata}")
    if r.server_seed != seed or r.policy_seed != seed:
        bad.append(f"result server_seed={r.server_seed} policy_seed={r.policy_seed}")
    rows = F.read_jsonl(h.raw(r) / "trace.jsonl")
    if rows[0]["identity"].get("policy_seed") != seed or rows[-1].get("policy_seed") != seed:
        bad.append("astra trace policy_seed")
    if rows[0].get("policy_seed", seed) != seed:
        bad.append("astra trace header policy_seed")
    saved = json.loads((h.astra_ep_dir(r) / "result.json").read_text())
    if saved.get("policy_seed") != seed or saved.get("cloud_seed") is not None:
        bad.append(f"astra result policy_seed={saved.get('policy_seed')}")
    prov = json.loads((h.attempt_dir(r) / "provenance.json").read_text())
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
    res = read_result(runner, ident)
    info = {"env_steps": len(env.actions), "status": row["status"], "exec_steps": row["exec_steps"],
            "cap_hit": row["cap_hit"], "max_steps": res.get("max_steps"), "strict_cap": res.get("strict_cap")}
    if len(env.actions) != CAP:
        bad.append(f"环境实收 {len(env.actions)} 步（应恰 {CAP}：第 {CAP} 步进入、第 {CAP + 1} 步不进入）")
    if (row["status"], row["exec_steps"], res.get("max_steps"), res.get("strict_cap"), row["infra"]) != \
            ("timeout", CAP, CAP, True, False):
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
    envs: list = []

    def plan(builder, ep):
        envs.append(A.FakeEnv(terminal_step=None))
        return envs[-1]

    h, r, net = astra_run(tmp, monkeypatch, seed=7, dataset="ood", task="VideoUnmask", tier="xhard1", env_plan=plan,
                          overrun=CAP + 100)
    tr = F.read_jsonl(h.raw(r) / "trace.jsonl")
    info = {"env_steps": envs[0].steps_taken, "status": r.status, "exec_steps": r.exec_steps, "cap_hit": r.cap_hit,
            "max_steps": r.max_steps, "strict_cap": r.strict_cap, "trace_effective_cap": tr[-1].get("effective_cap")}
    bad = []
    if envs[0].steps_taken != CAP:
        bad.append(f"环境实收 {envs[0].steps_taken} 步")
    if (r.status, r.exec_steps, r.cap_hit, r.max_steps, r.strict_cap) != ("timeout", CAP, True, CAP, True):
        bad.append(f"result={info}")
    if tr[-1].get("effective_cap") != CAP:
        bad.append(f"trace end effective_cap={tr[-1].get('effective_cap')}")
    steps = [x for x in tr if x["kind"] == "step"]
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
    # 对照（证明本判据有区分力）：同一 FrameSamp+Modulation 路线跑不 strict 的 hard-verify（上限 1300），客户端自己的
    # 循环会把第 1301 步送进环境（拆仓后 strict 只由数据集决定，不再有单独的开关）
    hv_cap = 1300
    ctl_world = World(G.Plan())
    ctl = seat_runner("perceptual-framesamp-modul", tmp_path / "control", ctl_world, monkeypatch, seed=7,
                      max_steps=CAP, spy={})
    assert F.run_rows(ctl, [F.hard0_identity("PickXtimes", 0)]) == 0
    assert len(ctl_world.envs[0].actions) == hv_cap + 1, "对照失效：不 strict 时第 1301 步应进入环境"
    print(f"EVAL_CAP=PASS models={len(ROUTES)} dataset=ood max_steps={CAP} rejected_step={CAP + 1} "
          f"control_hard_verify_not_strict_entered={hv_cap + 1}")


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
