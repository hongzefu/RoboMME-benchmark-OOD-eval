#!/usr/bin/env python3
"""GroundSG 原侧（官方循环 + 官方 builder）驱动（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.3、1.7；子任务 S3）。

独立进程，解释器用客户端扩展环境（client-env）。官方 ``eval.py`` 经 ``subgoal_predictor.py`` 无条件导入 gemini
（``google.generativeai``）与 memer，所以**不整模块 import** 官方文件：与新侧一样用
``official_defs.extract_defs`` 摘取 ``EpisodeEvaluator``、``Args``、``EnvRunner`` 与所选变体的预测器原文。
``sys.path`` 只加官方 ``examples/robomme``、benchmark 子模块的 ``src``（官方 ``robomme`` 在
``third_party/robomme_benchmark/src/robomme/``）与评估包 ``src``（同用的模块按文件路径加载，旧名→新文件见
``EVAL_MODULE_FILES``）；启动时断言 ``robomme.__file__`` 在子模块 ``src/robomme/`` 下、``"robomme_hard" not in
sys.modules``，每局结束再查一次。

流程：对 hard0 分片（``eval_manifest.py --mode hard0`` 产出的 ``shard-NN.json``）的每一行，官方
``EnvRunner(task, video_dir, max_steps=--max-steps)``（每任务一个，与官方 ``evaluate`` 相同）→
``make_env(source_episode)`` → ``EpisodeEvaluator.eval_each_episode`` → ``close_env``。预测器与评估器整个进程只建
一次（与官方相同，QwenVL 模型只加载一次）。

外围记录全部是委托包装，不改官方 ``robomme`` 的任何方法、不 monkeypatch：

* ``TapRunner`` 包住官方 ``EnvRunner`` 实例的 ``get_init_obs``／``step``（其余属性原样转发）；
* ``EnvTap`` 包住 ``EnvRunner.env`` 这个对象：``step`` 原样转发并记下 ``terminated``／``truncated``，其余属性转发；
* 客户端经 ``eval.py`` 的 ``_websocket_client_policy`` 名字注入 ``groundsg_client.TracingClient``（包住真实
  ``MMEVLAWebsocketClientPolicy``）。

每局目录 ``<out>/<key>.a<attempt>/``（已存在即报错，分片开跑前统一检查）::

    trace.jsonl                 # trace_writer，route groundsg/<variant>/orig；字段与新侧同一套代码写出
    frames/front.rgb24          # 逐帧 H×W×3 uint8 原始字节顺序拼接（ffmpeg rawvideo rgb24）
    frames/wrist.rgb24
    frames/frames.json          # {"pix_fmt":"rgb24","streams":{"front":{"width","height","count"},"wrist":{…}},
                                #  "demo_frames","init_frames","exec_steps","missing_steps","order"}
    official/<官方叠字视频>.mp4  # 第三阶段：官方循环自己写的视频，核验帧数后保留（归档先于清理），另写 provenance.json
    official/provenance.json
    language.jsonl              # 语言账本（trace_writer.LanguageLog 存在时；接口冻结说明五）
    ep<source_episode>_{QwenVL,MemER}_log.jsonl   # QwenVL／MemER 变体：官方子目标模型日志（局末从临时目录归档）

第三阶段（1006-rename-official-names-and-stage3-eval-plan.md 八.3、八.10 第 3、8 条；接口冻结说明 2.2～2.4、三、四、五）：

* 变体放行 ``ground-sg-memer``（``--memer-adapter``，与 ``--qwenvl-groundsg-adapter`` 互斥配对）；
* ``--policy-seed`` 必填（缺失 ``RUN_BLOCKED reason=policy_seed``、退出 3），贯通 ``make_args(model_seed=)``、
  ``seed_everything``（QwenVL／MemER 构造前）、trace identity／end 与结果行；
* 共享预算账本：``--budget-ledger --trajectory-cap --shared-infra-cap --expired-cap --planned-first-tries`` 必填（缺任一
  ``RUN_BLOCKED reason=budget_args``、退出 3，不回落账本常量默认值）；直接 ``load_sibling("budget_ledger")`` 调
  ``reserve``／``claim_reset``／``claim_retry``／``commit``／``release``，route ``groundsg/<variant>/seed<n>/orig``，
  幂等 token ``<route>|<key>|a<attempt>``（``OrigBudget``）。轨迹额度不足 ``RUN_BLOCKED reason=budget``、退出 5；
  第 2 次尝试的重试名额领不到时该身份记一行 ``infra_reason=budget_retry_denied``、不建局目录、不跑；
* trace identity 补 ``attempt``、``policy_seed``，end 补 ``steps_attempted／steps_observed／frames_recorded``（与新侧
  同一套 ``episode_counts``），缺观测步按 C8 用 ``log_missing_step``；
* 数据集只跑 ``hard-verify``，``--max-steps`` 只许 1300（``RUN_BLOCKED reason=step_cap_pairing``）。

帧顺序：``get_init_obs`` 返回的全部帧（``demo_frames`` 个演示帧 + 1 个初始帧），之后每执行一步追加该步观测；某步
环境抛异常（官方 ``EnvRunner.step`` 返回 None 三元组）时不追加、步号记入 ``missing_steps``。
``count = demo_frames + 1 + exec_steps - len(missing_steps)``。官方循环 ``count > max_steps`` 才判超时，所以超时局
执行第 ``max_steps+1`` 步，外围帧含这一步（xhard0：1300 → 执行 1301 步、``count = demo_frames + 1302``）；官方
循环自己写的叠字 mp4 只录到第 ``max_steps`` 步（第三阶段起保留进 ``official/``）。Qwen／MemER 临时目录在每局目录下的
``{qwen,memer}-tmp/hard-verify/<key>.a<attempt>/``，局末删除。

结果行（``<out>/results.jsonl``，每局一行，追加并 fsync）：分片行的身份字段 + ``side="orig"``、``policy="groundsg"``、
``policy_variant``、``dataset="hard-verify"``、``attempt``、``status``、``task_success``、``exec_steps``、``steps``、
``demo_frames``、``max_steps``、``effective_max_steps``、``success_flag``、``decisions``、``error``、``infra``、
``infra_reason``、``ep_dir``、``trace_path``、``frames_dir``、``video_frames``（``{"front":n,"wrist":n}``）、
``wall_s``、``official_sha256``；第三阶段另记 ``policy_seed``、``effective_cap``、``memer_compat_sha256``（MemER）、
``error_kind``、``budget_token``、``budget_rid``、``official_source``／``official_videos`` 与三分计数。官方 ``unknown`` 记 ``status="error"``、``error="success_flag=unknown"``，**不中止**
（官方 ``evaluate`` 在此中止整个评估；这里只记该局，属记录在案的偏离）。

用法::

    python official_hard_runner.py --shard <shard-NN.json> --out <目录> --port <端口> --policy-seed <n> \\
        --variant {ground-sg-oracle,ground-sg-qwenvl,ground-sg-memer} \\
        [--qwenvl-groundsg-adapter <路径> | --memer-adapter <路径>] \\
        --budget-ledger <账本> --trajectory-cap <n> --shared-infra-cap <n> --expired-cap <n> --planned-first-tries <n> \\
        [--host 127.0.0.1] [--max-steps 1300] [--attempt 1] [--only <key,…>] [--no-frames]
    python official_hard_runner.py --check-imports [--variant …]   # 只做导入断言，打印 OFFICIAL_IMPORTS=PASS …

退出码：0 分片跑完（含 error 局）；2 服务不可达中止；3 参数、分片、导入断言或账本配置不合格；5 轨迹预算不足。
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
# 本目录不进 sys.path（同目录模块一律按文件路径加载）
sys.path[:] = [p for p in sys.path if p and str(_Path(p).resolve()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
#: benchmark 子模块的源码根（官方 ``robomme`` 与 ``robomme_hard`` 都在这里）
BENCH_SRC = REPO / "third_party" / "robomme_benchmark" / "src"
#: 评估包源码根
EVAL_SRC = REPO / "src"
#: 旧仓评估目录下的模块名 → 拆仓后的文件（别名照旧）
EVAL_MODULE_FILES = {
    "official_defs": EVAL_SRC / "robomme_hard_eval" / "models" / "_official_defs.py",
    "groundsg_client": EVAL_SRC / "robomme_hard_eval" / "models" / "groundsg.py",
    "framesamp_modul_client": EVAL_SRC / "robomme_hard_eval" / "models" / "framesamp_modul.py",
    "smvla_client": EVAL_SRC / "robomme_hard_eval" / "models" / "smvla.py",
    "trace_writer": EVAL_SRC / "robomme_hard_eval" / "record" / "trace_writer.py",
    "recorder": EVAL_SRC / "robomme_hard_eval" / "record" / "recorder.py",
    "budget_ledger": REPO / "dev-scripts" / "gl" / "budget_ledger.py",
}
SIDE = "orig"
POLICY = "groundsg"
DATASET = "hard-verify"
XHARD0 = "xhard0"
MAX_STEPS = 1300
EXIT_UNREACHABLE = 2
EXIT_BAD_INPUT = 3
EXIT_BUDGET = 5
#: 原侧每次尝试预约的 reset 计量（官方 make_env 建环境与 get_init_obs 的 reset 各 1 次；与新侧同口径）
ORIG_RESETS_PER_ATTEMPT = 2
#: 第三阶段原侧预算参数（接口冻结说明 2.4；全部必填，不回落账本常量默认值）
BUDGET_ARGS = ("budget_ledger", "trajectory_cap", "shared_infra_cap", "expired_cap", "planned_first_tries")
#: 服务可达探测：每局建客户端前 TCP 探测，累计等这么久仍连不上即判「服务不可达」（官方客户端自己会无限重试）
PROBE_TIMEOUT_S = 300.0


def _load(name: str):
    """按文件路径加载旧名 ``name`` 对应的模块（``EVAL_MODULE_FILES``；别名与模型客户端的 ``load_sibling`` 相同，已加载
    则复用）。评估包 ``src`` 追加到 ``sys.path`` 末尾，供拆仓后模块里的 ``robomme_hard_eval.*`` 包内导入使用。"""
    if name in sys.modules:
        return sys.modules[name]
    if str(EVAL_SRC) not in sys.path:
        sys.path.append(str(EVAL_SRC))
    spec = importlib.util.spec_from_file_location(name, EVAL_MODULE_FILES[name])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


official_defs = _load("official_defs")
trace_writer = _load("trace_writer")  # 先于客户端登记：客户端按旧名 load_sibling 取同一份
groundsg = _load("groundsg_client")


class ServerUnreachable(ConnectionError):
    """模型服务在 ``PROBE_TIMEOUT_S`` 内连不上：本局记 error 并停止整个分片（退出码 2）。"""


def dumps(obj: Any) -> str:
    def default(o):
        try:
            import numpy as np

            if isinstance(o, np.generic):
                return o.item()
        except ImportError:
            pass
        return str(o)

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=default)


# ── 启动：路径与导入断言 ────────────────────────────────────────────────────


def setup_paths() -> list[str]:
    """``sys.path`` 只加官方 ``examples/robomme`` 与 benchmark 子模块 ``src``（放最前）。返回加入的两项。"""
    added = [str(official_defs.official_robomme_dir()), str(BENCH_SRC)]
    for p in reversed(added):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return added


def assert_no_robomme_hard() -> None:
    bad = sorted(m for m in sys.modules if m == "robomme_hard" or m.startswith("robomme_hard."))
    if bad:
        raise AssertionError(f"原侧进程导入了 robomme_hard：{bad[:5]}")


def import_official_env() -> dict:
    """照官方 ``env_runner.py`` 头部导入环境注册与 builder，并断言来源：``robomme`` 在 benchmark 子模块 ``src/robomme/``，
    ``robomme_hard`` 未被导入。返回 ``{"BenchmarkEnvBuilder", "robomme_file"}``。"""
    import robomme

    want = (BENCH_SRC / "robomme").resolve()
    got = Path(robomme.__file__).resolve()
    if want not in got.parents:
        raise AssertionError(f"robomme 不在 benchmark 子模块 src/robomme 下：{got}")
    import robomme.robomme_env  # noqa: F401 环境注册（官方 from robomme.robomme_env import *）
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    assert_no_robomme_hard()
    return {"BenchmarkEnvBuilder": BenchmarkEnvBuilder, "robomme_file": str(got)}


# ── 分片 ───────────────────────────────────────────────────────────────────


def load_shard(path: Path, only: str | None = None) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"分片应为非空 JSON 数组：{path}")
    keys = set()
    for i, r in enumerate(rows):
        for k in ("task", "tier", "seed", "source_episode", "key"):
            if k not in r:
                raise ValueError(f"分片第 {i} 行缺 {k}")
        if r["tier"] != XHARD0:
            raise ValueError(f"分片第 {i} 行 tier={r['tier']!r}，原侧只跑 {XHARD0}")
        if not isinstance(r["source_episode"], int) or isinstance(r["source_episode"], bool):
            raise ValueError(f"分片第 {i} 行 source_episode={r['source_episode']!r} 须为整数")
        if r["key"] in keys:
            raise ValueError(f"分片 key 重复：{r['key']}")
        keys.add(r["key"])
    if only:
        want = set(only.split(","))
        rows = [r for r in rows if r["key"] in want]
    return rows


# ── 委托包装 ────────────────────────────────────────────────────────────────


class EnvTap:
    """包住 ``EnvRunner.env`` 对象：``step`` 原样转发并记下 ``terminated``／``truncated``；其余属性转发。"""

    def __init__(self, env: Any):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "last", None)

    def step(self, action):
        object.__setattr__(self, "last", None)
        out = self._env.step(action)
        object.__setattr__(self, "last", (out[2], out[3]))
        return out

    def __getattr__(self, name):
        return getattr(self._env, name)


class TapRunner:
    """包住官方 ``EnvRunner`` 实例：``get_init_obs``／``step`` 先调官方原方法再把结果交给 ``EpisodeTap``；其余属性
    （``env_id``、``episode_id``、``task_goal``、``difficulty``、``info``、两个 oracle 属性）原样转发。
    ``before_reset``（可选）在官方 ``get_init_obs`` 真正 reset 之前调用（共享账本 ``claim_reset`` 计量）。"""

    def __init__(self, inner: Any, tap: Any, before_reset: Callable[[], Any] | None = None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tap", tap)
        object.__setattr__(self, "_before_reset", before_reset)

    def get_init_obs(self):
        if self._before_reset is not None:
            self._before_reset()
        pre = self._inner.get_init_obs()
        self._tap.on_reset(pre)
        return pre

    def step(self, action):
        env = self._inner.env
        out = self._inner.step(action)
        obs3, stop, flag = out
        tt = env.last if isinstance(env, EnvTap) else None
        term, trunc = tt if tt is not None else (None, None)
        self._tap.on_step(action, obs3, stop, flag, term, trunc)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


# ── 驱动 ────────────────────────────────────────────────────────────────────


def probe_server(host: str, port: int, timeout_s: float | None = None) -> None:
    """TCP 探测模型服务；``timeout_s``（默认取模块常量 ``PROBE_TIMEOUT_S``）内连不上抛 ``ServerUnreachable``。"""
    timeout_s = PROBE_TIMEOUT_S if timeout_s is None else float(timeout_s)
    t_end = time.monotonic() + timeout_s
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=5):
                return
        except OSError as e:
            if time.monotonic() >= t_end:
                raise ServerUnreachable(f"{host}:{port} 在 {timeout_s:.0f}s 内连不上：{e}") from e
            time.sleep(min(2.0, max(0.05, t_end - time.monotonic())))


def default_client_factory(host: str, port: int, episode: dict) -> Any:
    """原侧默认：先探测可达，再建真实 ``openpi_client`` 的 ``MMEVLAWebsocketClientPolicy``（与官方相同，不加记录）。"""
    probe_server(host, port)
    from openpi_client import websocket_client_policy

    return websocket_client_policy.MMEVLAWebsocketClientPolicy(host, port)


def make_context(variant: str, *, host: str, port: int, max_steps: int, policy_seed: Any,
                 adapter: str | None = None, memer_adapter: str | None = None,
                 builder_cls: Any, scratch_root: Path, client_factory: Callable | None = None,
                 qwen_extra: dict | None = None) -> dict:
    """进程内只建一次：官方定义（含 ``EnvRunner``）、``Args``（``model_seed=policy_seed``）、预测器（QwenVL／MemER 在此
    ``seed_everything`` 后加载模型）、评估器。"""
    seed = official_defs.check_policy_seed(policy_seed)
    ctx: dict[str, Any] = {"variant": variant, "episode": None, "policy_seed": seed,
                           "client_factory": client_factory or default_client_factory}

    def ws_factory(h, p):
        ep = ctx["episode"]
        client = groundsg.TracingClient(ctx["client_factory"](h, p, ep), ep["tap"])
        ep["clients"].append(client)
        return client

    defs = official_defs.load_groundsg(variant, with_env_runner=True,
                                       env_runner_extra={"BenchmarkEnvBuilder": builder_cls},
                                       ws_module=official_defs.ws_shim(ws_factory), qwen_extra=qwen_extra)
    base = Path(scratch_root) / "qwen-tmp"
    args = official_defs.make_args(defs, variant=variant, host=host, port=port, max_steps=max_steps, model_seed=seed,
                                   adapter_path=adapter, memer_adapter_path=memer_adapter, save_dir=str(base))
    predictor = official_defs.build_predictor(defs, args, base)
    ctx.update(defs=defs, args=args, predictor=predictor, evaluator=defs["EpisodeEvaluator"](args, base),
               official_sha256=dict(defs["sha256"]), runners={},
               memer_compat_sha256=defs.get("memer_compat_sha256"))
    return ctx


# ── 共享预算账本（接口冻结说明三） ─────────────────────────────────────────────


class RetryDenied(RuntimeError):
    """第 2 次尝试领不到共享重试名额（infra 额度或每身份次数用尽）：该身份不跑，记一行后继续下一身份。"""


def budget_route(variant: str, policy_seed: int) -> str:
    """原侧账本 route：``groundsg/<variant>/seed<policy_seed>/orig``（接口冻结说明三.2）。"""
    return f"{POLICY}/{variant}/seed{int(policy_seed)}/{SIDE}"


def budget_token(route: str, key: str, attempt: int) -> str:
    """幂等 token：``<route>|<key>|a<attempt>``（接口冻结说明三.3；崩溃续跑用同一 token 恢复，不重复扣额）。"""
    return f"{route}|{key}|a{int(attempt)}"


class OrigBudget:
    """原侧接共享账本（``budget_ledger.BudgetLedger`` 或测试替身；只用冻结说明列出的方法与参数名）。

    每个身份尝试：第 2 次起先 ``claim_retry(route=, key=, interrupt="infra", token=)``（领不到抛 ``RetryDenied``）→
    ``reserve(resets=2, route=, key=, token=, kind_of_try="first"|"recovery")`` 取 rid（额度不足由账本抛
    ``BudgetExhausted``，按 ``budget_exhausted`` 属性识别，交调用方停片退出 5）→ 每次真实 build／reset 前
    ``claim_reset(rid, what, route=)`` → 局末 ``commit(rid, status=, infra=)``；没进到建环境就失败的 ``release``。"""

    def __init__(self, ledger: Any, *, variant: str, policy_seed: int):
        self.ledger = ledger
        self.route = budget_route(variant, policy_seed)

    def begin(self, key: str, attempt: int) -> dict:
        token = budget_token(self.route, key, attempt)
        if int(attempt) >= 2:
            ok = self.ledger.claim_retry(route=self.route, key=key, interrupt="infra", token=token,
                                         side=SIDE, attempt_no=int(attempt))
            if not ok:
                raise RetryDenied(f"claim_retry 拒绝 route={self.route} key={key} attempt={attempt}")
        rid = self.ledger.reserve(resets=ORIG_RESETS_PER_ATTEMPT, route=self.route, key=key, token=token,
                                  kind_of_try="first" if int(attempt) == 1 else "recovery",
                                  attempt_no=int(attempt), side=SIDE, policy=POLICY, astra=False)
        return {"token": token, "rid": rid, "resets_claimed": 0}

    def claimer(self, handle: dict) -> Callable[[str], None]:
        def claim(what: str) -> None:
            handle["resets_claimed"] += 1
            self.ledger.claim_reset(handle["rid"], what, route=self.route)

        return claim

    def settle(self, handle: dict, rec: dict) -> None:
        if handle.get("resets_claimed", 0) == 0:  # 一次 build 都没到：未进入执行，退回轨迹名额
            self.ledger.release(handle["rid"], status=rec.get("status"))
        else:
            self.ledger.commit(handle["rid"], status=rec.get("status"), infra=bool(rec.get("infra")))


def runner_for(ctx: dict, task: str, video_dir: Path) -> Any:
    """每任务一个官方 ``EnvRunner``（与官方 ``evaluate`` 相同，builder 建一次跨局复用）。"""
    if task not in ctx["runners"]:
        ctx["runners"][task] = ctx["defs"]["EnvRunner"](task, video_dir, max_steps=ctx["args"].max_steps)
    return ctx["runners"][task]


def ep_dir_of(out: Path, row: dict, attempt: int) -> Path:
    return Path(out) / f"{row['key']}.a{int(attempt)}"


def run_identity(ctx: dict, row: dict, *, out: Path, attempt: int = 1, write_frames: bool = True,
                 budget_claim: Callable[[str], Any] | None = None, budget: dict | None = None) -> dict:
    """跑一个 hard0 身份，返回结果行（不写 results.jsonl）。每局目录已存在即抛 ``FileExistsError``。

    ``budget_claim(what)``：每次真实 build（官方 ``make_env``）／reset（官方 ``get_init_obs``）之前调用（共享账本
    ``claim_reset``）；``budget``：``OrigBudget.begin`` 的返回（token、rid 记进结果行）。"""
    variant = ctx["variant"]
    max_steps = int(ctx["args"].max_steps)
    policy_seed = ctx.get("policy_seed")
    tag = f"{row['key']}.a{int(attempt)}"
    ep_dir = ep_dir_of(out, row, attempt)
    ep_dir.mkdir(parents=True, exist_ok=False)
    tpath = ep_dir / "trace.jsonl"
    route = f"groundsg/{variant}/orig"
    tid = {k: row.get(k) for k in ("task", "tier", "seed", "source_episode", "builder_episode", "key")}
    tid["dataset"] = DATASET
    tid["attempt"] = int(attempt)  # C6：与局目录名 <key>.a<N> 的 N 一致
    tid["policy_seed"] = policy_seed  # 接口冻结说明四.1
    trace = groundsg.make_trace(tpath, route=route, identity=tid, max_steps=max_steps, policy_seed=policy_seed,
                                effective_cap=max_steps)
    frames = groundsg.RawFrameWriter(ep_dir / "frames") if write_frames else None
    tap = groundsg.EpisodeTap(trace, frames, missing_step_contract=True)
    video_dir = ep_dir / "official-video"
    t0 = time.perf_counter()
    runner = None
    difficulty = None
    lang_log = groundsg.open_language_log(tpath)
    prov = {"identity": dict(tid), "route": route, "dataset": DATASET, "attempt": int(attempt), "episode_tag": tag,
            "official_episode_id": int(row["source_episode"]), "official_sha256": dict(ctx.get("official_sha256") or {}),
            "policy_seed": policy_seed, "policy_variant": variant, "side": SIDE}
    if variant == official_defs.VARIANT_MEMER:
        prov["memer_compat_sha256"] = ctx.get("memer_compat_sha256")
    try:
        runner = runner_for(ctx, row["task"], video_dir)
        runner.video_save_dir = video_dir
        if budget_claim is not None:
            budget_claim("build")
        runner.make_env(int(row["source_episode"]))
        difficulty = getattr(runner, "difficulty", None)
        runner.env = EnvTap(runner.env)
        before = (lambda: budget_claim("reset")) if budget_claim is not None else None
        res = groundsg.run_official_episode(ctx, TapRunner(runner, tap, before_reset=before), tap, dataset=DATASET,
                                         episode_tag=tag, scratch=ep_dir, archive_dir=ep_dir, keep_official=True,
                                         official_provenance=prov, language_log=lang_log)
    except Exception as e:  # noqa: BLE001 官方 EnvRunner 构建／make_env 失败（官方 evaluate 同样整局记 error）
        err = f"{type(e).__name__}: {e}"[:800]
        infra = groundsg.classify_infra(err)
        res = {"status": "error", "task_success": False, "steps": tap.steps, "error": err, "success_flag": "error",
               "decisions": tap.decisions, "infra": True, "infra_reason": infra or "env_build",
               "env_exception": None, "exception": type(e).__name__, "qwen_log": None, "timing": {},
               "official_videos": [], "official_source": "none",
               "official_save_error": f"官方循环未开始：{err}"[:800],
               **groundsg.episode_counts(tap, "error", type(e).__name__, max_steps)}
        if "policy_seed" in ctx:
            res.update(policy_seed=policy_seed, policy_variant=variant, error_kind=None, subgoal_log=None)
            if variant == official_defs.VARIANT_MEMER:
                res["memer_compat_sha256"] = ctx.get("memer_compat_sha256")
    finally:
        if lang_log is not None:
            lang_log.close()
        if runner is not None and getattr(runner, "env", None) is not None:
            try:
                runner.close_env()
            except Exception as e:  # noqa: BLE001
                print(f"close_env error: {e!r}", flush=True)
    if res.get("exception") == "ServerUnreachable":
        res.update(infra=True, infra_reason="groundsg_unreachable")
    shutil.rmtree(video_dir, ignore_errors=True)  # 官方视频已在 run_official_episode 里核验后搬进 official/
    wall = time.perf_counter() - t0
    demo = 0 if res.get("no_frame") else tap.demo_frames  # C3：无帧 error 局 demo_frames 记 0
    extra = {k: res[k] for k in ("steps_attempted", "steps_observed", "frames_recorded", "omitted_timeout_frames")
             if k in res}
    if res.get("no_frame"):
        extra["no_frame"] = True
    # C3：terminal_reason 取终态（与新侧同口径），官方原返回值另记 success_flag
    trace.close(status=res["status"], terminal_reason=res["status"], side=SIDE, demo_frames=demo,
                decisions=res.get("decisions"), success_flag=res.get("success_flag"),
                official_source=res.get("official_source"),
                official_videos=[Path(x).name for x in res.get("official_videos") or []], policy_seed=policy_seed,
                **extra)
    fsum = frames.close() if frames is not None else None
    out_row = dict(row)
    out_row.update(res)
    out_row.update(side=SIDE, policy=POLICY, policy_variant=variant, dataset=DATASET, attempt=int(attempt),
                   attempt_no=int(attempt), task_success=res["status"] == "success", exec_steps=tap.steps,
                   steps=tap.steps, demo_frames=demo, max_steps=max_steps, effective_max_steps=max_steps,
                   effective_cap=max_steps, difficulty=difficulty, ep_dir=str(ep_dir), trace_path=str(tpath),
                   frames_dir=None if frames is None else str(frames.dir),
                   video_frames=None if fsum is None else {s: v["count"] for s, v in fsum["streams"].items()},
                   wall_s=round(wall, 3), host=socket.gethostname(), policy_seed=policy_seed,
                   budget_token=(budget or {}).get("token"), budget_rid=(budget or {}).get("rid"))
    return out_row


def append_result(path: Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def retry_denied_row(ctx: dict, row: dict, *, attempt: int, token: str, detail: str) -> dict:
    """重试名额被共享账本拒绝：不建局目录、不跑，只记一行（infra，``plan_round`` 据此不再排这一身份）。"""
    out = dict(row)
    out.update(side=SIDE, policy=POLICY, policy_variant=ctx["variant"], dataset=DATASET, attempt=int(attempt),
               attempt_no=int(attempt), status="error", task_success=False, exec_steps=0, steps=0, infra=True,
               infra_reason="budget_retry_denied", error=detail[:800], policy_seed=ctx.get("policy_seed"),
               budget_token=token, budget_rid=None, ep_dir=None, host=socket.gethostname())
    return out


def run_shard(ctx: dict, rows: list[dict], *, out: Path, attempt: int = 1, write_frames: bool = True,
              import_check: Callable[[], Any] | None = None, budget: OrigBudget | None = None) -> dict:
    """逐身份跑；返回 ``{"episodes","errors","aborted","budget_blocked",<各终态计数>}``。服务不可达即停止（aborted）；
    给了 ``budget``（``OrigBudget``）时每个身份先过共享账本，轨迹额度不足即停止（budget_blocked）。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    summary = {"episodes": 0, "errors": 0, "aborted": False, "budget_blocked": False, "retry_denied": 0,
               "success": 0, "fail": 0, "timeout": 0}
    for row in rows:
        handle = None
        if budget is not None:
            try:
                handle = budget.begin(row["key"], attempt)
            except RetryDenied as e:
                rec = retry_denied_row(ctx, row, attempt=attempt, token=budget_token(budget.route, row["key"], attempt),
                                       detail=str(e))
                append_result(results, rec)
                summary["retry_denied"] += 1
                print(f"BUDGET_RETRY_DENIED side={SIDE} route={budget.route} key={row['key']} attempt={attempt}",
                      flush=True)
                continue
            except Exception as e:  # noqa: BLE001 budget_ledger.BudgetExhausted（按属性识别，可能来自另一份模块副本）
                if not getattr(e, "budget_exhausted", False):
                    raise
                print(f"RUN_BLOCKED reason=budget side={SIDE} route={budget.route} key={row['key']} detail={e}",
                      flush=True)
                summary["budget_blocked"] = True
                break
        try:
            rec = run_identity(ctx, row, out=out, attempt=attempt, write_frames=write_frames,
                               budget_claim=budget.claimer(handle) if handle is not None else None, budget=handle)
        except BaseException:
            if handle is not None:
                budget.settle(handle, {"status": "error", "infra": True})
            raise
        append_result(results, rec)
        if handle is not None:
            budget.settle(handle, rec)
        summary["episodes"] += 1
        summary["errors"] += int(rec["status"] == "error")
        if rec["status"] in ("success", "fail", "timeout"):
            summary[rec["status"]] += 1
        print(f"EPISODE_DONE side={SIDE} policy={POLICY} variant={ctx['variant']} dataset={DATASET} key={rec['key']} "
              f"status={rec['status']} exec_steps={rec['exec_steps']} demo_frames={rec['demo_frames']} "
              f"video_frames={(rec['video_frames'] or {}).get('front')} infra={rec['infra']} wall_s={rec['wall_s']:.1f}",
              flush=True)
        if import_check is not None:
            import_check()
        if rec.get("exception") == "ServerUnreachable":
            summary["aborted"] = True
            break
    return summary


def cmd_check_imports(args) -> int:
    """只做导入断言：路径、``robomme`` 来源、所选（默认全部三个）变体的官方定义可摘取；不建环境、不连服务、不加载模型。"""
    added = setup_paths()
    info = import_official_env()
    variants = [args.variant] if args.variant else list(official_defs.VARIANTS)
    for v in variants:
        defs = official_defs.load_groundsg(v, with_env_runner=True,
                                           env_runner_extra={"BenchmarkEnvBuilder": info["BenchmarkEnvBuilder"]},
                                           ws_module=official_defs.ws_shim(lambda h, p: None))
        names = sorted(k for k in defs["predictor"] if k.endswith("Predictor"))
        print(f"OFFICIAL_DEFS variant={v} predictors={','.join(names)} "
              + " ".join(f"{k}={s}" for k, s in sorted(defs["sha256"].items())), flush=True)
    assert_no_robomme_hard()
    leaked = sorted(m for m in ("subgoal_predictor", "eval", "env_runner", "utils", "google.generativeai", "swift",
                                "subgoal_prediction") if m in sys.modules)
    if leaked:
        raise AssertionError(f"官方模块被整模块导入：{leaked}")
    print(f"OFFICIAL_IMPORTS=PASS robomme={info['robomme_file']} robomme_hard_imported=0 "
          f"official_modules_imported=0 variants={len(variants)} sys_path_added={','.join(added)}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GroundSG 原侧驱动（官方循环 + 官方 builder 跑 hard0 分片）")
    ap.add_argument("--check-imports", action="store_true", help="只做导入断言")
    ap.add_argument("--shard", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--only", default=None, help="逗号分隔的 key，只跑这些")
    ap.add_argument("--no-frames", action="store_true", help="不写原始帧")
    ap.add_argument("--variant", choices=list(official_defs.VARIANTS))
    ap.add_argument("--qwenvl-groundsg-adapter", default=None, help="ground-sg-qwenvl 必填（其余变体不许给）")
    ap.add_argument("--memer-adapter", default=None, help="ground-sg-memer 必填（其余变体不许给）")
    ap.add_argument("--policy-seed", default=None, help="模型种子（必填，非负整数；接口冻结说明 2.2）")
    ap.add_argument("--budget-ledger", default=None, help="共享预算账本路径（必填）")
    ap.add_argument("--trajectory-cap", type=int, default=None, help="轨迹硬上限（必填）")
    ap.add_argument("--shared-infra-cap", type=int, default=None, help="共享基础设施重试上限（必填）")
    ap.add_argument("--expired-cap", type=int, default=None, help="到期接续上限（必填）")
    ap.add_argument("--planned-first-tries", type=int, default=None, help="计划首试数（必填，账本为其保留额度）")
    return ap


def open_budget_ledger(args) -> Any:
    """按冻结说明三的参数名打开共享账本（``budget_ledger.BudgetLedger``，不传任何默认值以外的常量）。"""
    budget_ledger = _load("budget_ledger")
    return budget_ledger.BudgetLedger(args.budget_ledger, trajectory_cap=int(args.trajectory_cap),
                                      shared_infra_cap=int(args.shared_infra_cap), expired_cap=int(args.expired_cap),
                                      planned_first_tries=int(args.planned_first_tries))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_imports:
        try:
            return cmd_check_imports(args)
        except (AssertionError, FileNotFoundError, KeyError) as e:
            print(f"OFFICIAL_IMPORTS=FAIL reason={e}", flush=True)
            return EXIT_BAD_INPUT
    # 参数核对先于任何导入（不改 sys.path、不导入 robomme）：缺参数、配对、policy_seed、预算五参数、步数配对
    miss = [n for n, v in (("--shard", args.shard), ("--out", args.out), ("--port", args.port),
                           ("--variant", args.variant)) if v is None]
    if miss:
        print(f"GROUNDSG_ORIG_BLOCKED reason=args missing={' '.join(miss)}", flush=True)
        return EXIT_BAD_INPUT
    if (args.variant == official_defs.VARIANT_QWENVL) != bool(args.qwenvl_groundsg_adapter):
        print("GROUNDSG_ORIG_BLOCKED reason=args --qwenvl-groundsg-adapter 仅且必须与 ground-sg-qwenvl 同用", flush=True)
        return EXIT_BAD_INPUT
    if (args.variant == official_defs.VARIANT_MEMER) != bool(args.memer_adapter):
        print("GROUNDSG_ORIG_BLOCKED reason=args --memer-adapter 仅且必须与 ground-sg-memer 同用", flush=True)
        return EXIT_BAD_INPUT
    try:
        policy_seed = official_defs.check_policy_seed(args.policy_seed)
    except ValueError:
        print(f"RUN_BLOCKED reason=policy_seed value={args.policy_seed!r}（--policy-seed 必填，非负整数）", flush=True)
        return EXIT_BAD_INPUT
    bmiss = [f"--{n.replace('_', '-')}" for n in BUDGET_ARGS if getattr(args, n) in (None, "")]
    if bmiss:
        print(f"RUN_BLOCKED reason=budget_args missing={' '.join(bmiss)}", flush=True)
        return EXIT_BAD_INPUT
    if args.max_steps != MAX_STEPS:
        print(f"RUN_BLOCKED reason=step_cap_pairing dataset={DATASET} max_steps={args.max_steps} "
              f"expect_max_steps={MAX_STEPS}（原侧只跑 hard-verify）", flush=True)
        return EXIT_BAD_INPUT
    try:
        added = setup_paths()
        info = import_official_env()
    except (AssertionError, FileNotFoundError, KeyError) as e:
        print(f"OFFICIAL_IMPORTS=FAIL reason={e}", flush=True)
        return EXIT_BAD_INPUT
    try:
        rows = load_shard(args.shard, args.only)
    except (ValueError, OSError) as e:
        print(f"GROUNDSG_ORIG_BLOCKED reason=shard detail={e}", flush=True)
        return EXIT_BAD_INPUT
    exists = [str(ep_dir_of(args.out, r, args.attempt)) for r in rows if ep_dir_of(args.out, r, args.attempt).exists()]
    if exists:
        print(f"GROUNDSG_ORIG_BLOCKED reason=ep_dir_exists n={len(exists)} first={exists[0]}", flush=True)
        return EXIT_BAD_INPUT
    try:
        ledger = open_budget_ledger(args)
    except Exception as e:  # noqa: BLE001 账本配置行不一致／坏行即拒（接口冻结说明三.1、三.6），按类名识别
        if type(e).__name__ in ("BudgetConfigMismatch", "LedgerCorrupt"):
            print(f"RUN_BLOCKED reason=budget_config detail={type(e).__name__}: {e}", flush=True)
            return EXIT_BAD_INPUT
        raise
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = make_context(args.variant, host=args.host, port=args.port, max_steps=args.max_steps, policy_seed=policy_seed,
                       adapter=args.qwenvl_groundsg_adapter, memer_adapter=args.memer_adapter,
                       builder_cls=info["BenchmarkEnvBuilder"], scratch_root=args.out)
    budget = OrigBudget(ledger, variant=args.variant, policy_seed=policy_seed)
    url = f"ws://{args.host}:{args.port}"
    print(f"GROUNDSG_ORIG_START shard={args.shard} variant={args.variant} episodes={len(rows)} url={url} "
          f"max_steps={args.max_steps} policy_seed={policy_seed} budget_route={budget.route} "
          f"memer_compat_sha256={ctx.get('memer_compat_sha256') or 'none'} robomme={info['robomme_file']} "
          f"sys_path_added={','.join(added)} "
          + " ".join(f"{k}={s}" for k, s in sorted(ctx["official_sha256"].items())), flush=True)
    summary = run_shard(ctx, rows, out=args.out, attempt=args.attempt, write_frames=not args.no_frames,
                        import_check=assert_no_robomme_hard, budget=budget)
    print(f"GROUNDSG_ORIG_DONE shard={args.shard} variant={args.variant} episodes={summary['episodes']} "
          f"errors={summary['errors']} success={summary['success']} fail={summary['fail']} "
          f"timeout={summary['timeout']} aborted={int(summary['aborted'])} "
          f"budget_blocked={int(summary['budget_blocked'])} retry_denied={summary['retry_denied']}", flush=True)
    if summary["budget_blocked"]:
        return EXIT_BUDGET
    return EXIT_UNREACHABLE if summary["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
