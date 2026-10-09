"""Astra 模型：把 Astra-on-RoboMME 的单局循环 ``runner.episode`` 接到评估仓「模型侧 4 个方法」上（拆分方案 §三 Astra 行）。

``AstraPolicy`` 的四个方法（外层 ``load_policy``／``run_episode`` 调用，接口见 ``robomme_ood_eval.policy``）：

- ``load()``（进程级一次）：核模型种子与两张卡；``bootstrap`` 子模块 ``examples/champ``；断言环境源是
  ``third_party/robomme_benchmark/src``；``validate_checkpoints``；查 ``group_*/STOP.json``（已停即拒，账本保留）；
  用 ``ServerProcess`` 起**费用守卫** ``servers/astra_cost_guard.py --cap <astra_cap_usd≤5>``（账本 ``astra_ledger``、
  单价 ``astra_prices``，就绪 = 守卫心跳状态新鲜）与 **Astra VLA 服务端**（子模块 ``scripts/serve_policy.py``，
  ``--seed=<policy_seed>``，``CUDA_VISIBLE_DEVICES=gpus[0]``，就绪 = 端口在听）；本进程钉到 ``gpus[1]`` 加载
  ``runner.Monitor``；建一个 ``GuardedResponsesClient``（20 秒发送间隔靠同一实例维持）与一个 ``core.Planner``。
- ``reset(spec)``（每局第一句；不向服务端发消息、不碰环境）：探两个服务端存活；上一局留下的停机条件、
  ``group_*/STOP.json`` 任一成立即抛 ``AstraStop``；``CostGate.register_episode`` 跨进程登记局数（第 3 局即拒）。
- ``play(session, spec, recorder)``：经 ``SessionBuilder`` 把外层 ``EnvSession`` 交给子模块
  ``runner.episode(args, task, ep, builder, monitor, planner, client)``，外面套 ``Traced*`` 记录包装写本局
  ``trace.jsonl``／``language.jsonl``；``args.max_steps`` 随每局 ``spec.max_steps`` 传入（同一 Policy 可先后跑 1300、
  1800 两档）。返回后按上游 ``main()`` 的停机规则记下停机条件，由下一局 ``reset`` 抛出。
- ``close()``：先停 VLA 服务端进程组、再停守卫（守卫收 TERM 后写 ``exited=true``），释放监视模型。

停机规则（从旧 ``run_cases`` 搬进 Policy；play 返回后判定，下一局 ``reset`` 抛 ``AstraStop``，外层整批停）：
① 错误信息以 ``PLANNER_STOP_PREFIXES`` 开头的局单次即停；② ``status=="error"`` 且不是规划输出不合模板
（``core.is_planner_failure``）的局连续累计 3 次即停（正常结局清零）；③ 已登记局数到 ``ASTRA_MAX_EPISODES`` 即停；
另 ④ ``group_*/STOP.json``（守卫到线或人工写）在每局 ``reset`` 与每次发送前都查。

产物（S4 三件）：

1. ``args.output`` = ``<spec.out_dir>/astra``：Astra 自写的 ``<task>/ep<NNN>/{identity.json, decisions.jsonl,
   rollout.mp4, actions.npy, result.json, monitor_inputs/}`` 留在那里，不与外层产物树冲突；
2. 尝试目录名用 ``spec.attempt``：``<task>/ep<NNN>/<key>.a<attempt>/provenance.json``（不再写死 ``a1``）；
3. 停机规则如上。

``trace.jsonl``、``language.jsonl``、``arrays.npz`` 写在 ``spec.out_dir``（外层出官方版式视频读这里的 trace；
``arrays.npz`` 经 ``trace_writer.merge_write_npz`` 与外层录制器的同名键逐项核对合并）。原始帧、逐步数组、事件由外层
``EnvSession`` 写进外层录制器，本模块不再自建录制器。

费用口径（用户：Astra 独立费用账本、5 美元硬上限）：守卫 ``--cap`` 默认且最大 5 美元（``HARD_CAP_USD``），
``GuardedResponsesClient`` 每次真正发送前同步读守卫状态、原子预留单次最坏费用，守卫失联（心跳超过 10 秒）、STOP、
预留失败任一即拒发；局数硬上限 2，跨进程登记在守卫预留文件里。账本、预留文件与 ``STOP.json`` 都在
``astra_ledger`` 所在目录下长期保留，绝不为绕额度新建账本。

密钥只从环境变量 ``OPENAI_API_KEY`` 读，交给 ``GuardedResponsesClient``；本模块不读密钥文件、不打印密钥；VLA 服务端
进程的环境里去掉这个变量（与上游 ``run.sh`` 的 ``env -u OPENAI_API_KEY`` 同义）。

模型 seed：``policy_seed`` 作为 VLA 服务 ``--seed``（替代上游固定 42）；云端 planner／monitor 无 seed 接口，只在
trace、``result.json``、来源清单里记 ``policy_seed`` 与 ``cloud_seed=null``，不伪造。

步数：``ood ↔ 1800``（strict：外层 ``EnvSession`` 与 ``TracedEnv`` 都在第 1801 次 ``step`` 进环境之前拒绝，本局按
``timeout`` 收尾）、``hard-verify ↔ 1300``（非 strict）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from robomme_ood_eval.policy import AstraStop as _PolicyAstraStop
from robomme_ood_eval.policy import Policy, Ready, ServerProcess, pick_port

HERE = Path(__file__).resolve().parent
#: 评估仓根（``src/robomme_ood_eval/models`` 往上三层）
REPO_ROOT = HERE.parents[2]

#: 新侧只接受这两个数据集；与每局 ``spec.max_steps`` 做配对一致性检查
DATASET_STEP_PAIRING = {"hard-verify": 1300, "ood": 1800}
#: ood 为 strict（第 1801 次 step 在进入真实环境前被拒），hard-verify 不变（截断靠底层环境）
DATASET_STRICT_CAP = {"hard-verify": False, "ood": True}
#: 上游 ``api_client.ResponsesClient`` 请求体里固定的输出上限（规划与复审共用；只用于语言账本 params）
PLANNER_MAX_OUTPUT_TOKENS = 2048
#: 上游 ``runner.Monitor`` 的 ``RequestConfig(max_tokens=8, temperature=0)``（只用于语言账本 params）
MONITOR_PARAMS = {"temperature": 0, "max_tokens": 8}
#: 来源清单 provenance.json 的文件名（尝试目录 ``<key>.a<attempt>/`` 下）
PROVENANCE_FILE = "provenance.json"
#: 上游 ``main()`` 第⑤项：错误信息以这三个前缀开头的局，单次即停
PLANNER_STOP_PREFIXES = ("Planner API", "Pilot planner-call", "Planner bridge failed")
#: 上游 ``main()`` 第④项：非规划类 error 连续累计到此数即停
INFRA_ERROR_LIMIT = 3
#: ``ResponsesClient._send`` 只认这两个目录名下的 STOP.json
GROUP_DIR_NAMES = ("group_0", "group_1")
#: 启动时打印 sha256 的 Astra 源文件（留档用）
ASTRA_SOURCE_FILES = ("runner.py", "core.py", "api_client.py", "input_contract.py", "release_utils.py",
                      "train_entry.py", "weights.json")
#: 新侧路线名
ROUTE = "astra/new"
#: Astra 自写产物在本局 raw 目录下的子目录名（S4 第①件）
ASTRA_SUBDIR = "astra"
#: 终态
TERMINALS = ("success", "fail", "timeout", "error")
#: 守卫心跳超时（秒）：超过即视为守卫失联、拒发（与 astra_cost_guard.HEARTBEAT_TIMEOUT_S 相同，由测试核对）
GUARD_HEARTBEAT_TIMEOUT_S = 10.0
#: 第三方 ResponsesClient 两次发送之间的固定间隔（秒；上游 _send 里的 20）
SEND_INTERVAL_S = 20
#: VLA 服务端口缺省基数（与上游 run.sh 的 18762 相同）
DEFAULT_PORT_BASE = 18762
#: 上游 ``runner.episode`` 的单局规划次数上限缺省（上游 main() 的 ``--max-planner-calls`` 默认 24）
DEFAULT_MAX_PLANNER_CALLS = 24
#: VLA 服务就绪等待上限（秒；旧 run_astra.sh 为 15 分钟）
VLA_READY_TIMEOUT_S = 900.0
#: 守卫就绪等待上限（秒）
GUARD_READY_TIMEOUT_S = 60.0
#: Astra 单局墙钟（秒）：20 秒发送间隔 × 至多 24 次规划 + 监视与仿真，按 1800 缺省偏紧，放宽到 3600
ASTRA_EPISODE_WALL_S = 3600.0


class AstraStop(_PolicyAstraStop):
    """Astra 报停（上游 ``main()`` 那几处 ``raise RuntimeError`` 的同义异常）；``reason`` 供测试与日志判读。
    是 ``robomme_ood_eval.policy.AstraStop`` 的子类：外层 ``scripts/evaluate.py`` 据此整批停（退出 3）。"""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class StepCapReached(RuntimeError):
    """strict cap：已执行 ``effective_cap`` 步后再调用 ``step``，不进入真实环境（与 ``session.StepCapReached`` 同义）。
    继承 ``RuntimeError``：Astra ``runner.episode`` 的 ``except Exception`` 接住它，本模块收尾时改记 ``timeout``。"""


# ── 路径与上游模块 ─────────────────────────────────────────────────────────

def third_party_root() -> Path:
    """第三方子模块根：``SGEVAL_THIRD_PARTY``（worktree 测试只读引用主检出）> 本仓库 ``third_party``。"""
    third = os.environ.get("SGEVAL_THIRD_PARTY")
    return Path(third).resolve() if third else (REPO_ROOT / "third_party").resolve()


def astra_root(explicit: str | None = None) -> Path:
    """Astra 子模块根：显式参数（cfg ``astra_root``）> ``<third_party>/Astra-on-RoboMME``。"""
    if explicit:
        return Path(explicit).resolve()
    return third_party_root() / "Astra-on-RoboMME"


def benchmark_src() -> Path:
    """环境源唯一允许的位置：``<third_party>/robomme_benchmark/src``（benchmark 仓子模块）。"""
    return third_party_root() / "robomme_benchmark" / "src"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(root: Path) -> SimpleNamespace:
    """``sys.path`` 加 Astra ``examples/champ`` 与 ``packages/openpi-client/src``，导入 Astra 的四个模块并返回。

    不再把本模块所在目录放进 ``sys.path``（评估仓里那是 ``models/``，会遮蔽同名模块）；``trace_writer`` 与守卫
    一律按包路径导入。"""
    champ = root / "examples" / "champ"
    if not (champ / "runner.py").is_file():
        raise FileNotFoundError(f"找不到 Astra 源码 {champ}/runner.py（子模块未初始化？）")
    for entry in (str(root / "packages" / "openpi-client" / "src"), str(champ)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    import api_client  # noqa: PLC0415  Astra 的直连 Responses API 客户端
    import core  # noqa: PLC0415
    import release_utils  # noqa: PLC0415
    import runner as astra_runner  # noqa: PLC0415  Astra 的单局循环（模块级 episode 函数）
    return SimpleNamespace(root=root, champ=champ, runner=astra_runner, core=core,
                           release_utils=release_utils, api_client=api_client)


def source_digests(champ: Path) -> dict[str, str]:
    return {name: file_sha256(champ / name) for name in ASTRA_SOURCE_FILES if (champ / name).is_file()}


def assert_env_sources() -> dict[str, str]:
    """环境源必须是 ``third_party/robomme_benchmark/src``：``robomme`` 与 ``robomme_hard`` 都不得来自 Astra 嵌套的
    官方副本或别处。"""
    import robomme  # noqa: PLC0415
    import robomme_hard  # noqa: PLC0415
    want = str(benchmark_src().resolve()) + os.sep
    files = {"robomme_hard": str(Path(robomme_hard.__file__).resolve()),
             "robomme": str(Path(robomme.__file__).resolve())}
    for name, path in files.items():
        if not path.startswith(want):
            raise RuntimeError(f"{name} 不是来自 third_party/robomme_benchmark/src（{want}）：{path}")
    print(f"ASTRA_ENV_SOURCE robomme_hard={files['robomme_hard']} robomme={files['robomme']}", flush=True)
    return files


def check_pairing(dataset: str, max_steps: int) -> None:
    expected = DATASET_STEP_PAIRING.get(dataset)
    if expected is None:
        raise ValueError(f"RUN_BLOCKED reason=dataset dataset={dataset!r}（只接受 {sorted(DATASET_STEP_PAIRING)}）")
    if int(max_steps) != expected:
        raise ValueError(f"RUN_BLOCKED reason=step_cap_pairing dataset={dataset} max_steps={max_steps}（应为 {expected}）")


def check_policy_seed(value) -> int:
    """``policy_seed`` 必填、非负整数；缺失或非法即 ``RUN_BLOCKED reason=policy_seed``（不回落旧默认 42）。"""
    if value is None or isinstance(value, bool):
        raise ValueError("RUN_BLOCKED reason=policy_seed 必须显式给模型种子（非负整数，不回落任何旧默认值）")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.strip().isdigit():
        seed = int(value.strip())
    else:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed policy_seed={value!r} 须为非负整数")
    if seed < 0:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed policy_seed={value!r} 须为非负整数")
    return seed


# ── 轨迹委托（trace_writer） ───────────────────────────────────────────────

def _canonical(obj: Any) -> bytes:
    """规范化字节：数组一律换成 ``array_record``（dtype、shape、sha256），其余按 JSON 排序键序列化。"""
    import numpy as np  # noqa: PLC0415
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415

    def conv(value):
        if isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")):
            rec = tw.array_record(value)
            return {"dtype": rec["dtype"], "shape": rec["shape"], "sha256": rec["sha256"]}
        if isinstance(value, dict):
            return {str(k): conv(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [conv(v) for v in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    return json.dumps(conv(obj), sort_keys=True, ensure_ascii=False).encode()


class TraceContext:
    """一局的共享状态：当前步号与子目标（VLA 请求里的 ``grounded_subgoal``）、C8 三分计数、原动作、录制器。

    ``open_recorder``：无参可调用，首次 reset 时由 ``TracedEnv`` 调用一次建录制器（评估仓的 ``run_one`` 恒传 ``None``：录制归外层 ``EnvSession``）；
    为 ``None`` 时不录（只供只测轨迹的单元测试）。

    第三阶段：``effective_cap``／``strict_cap``（strict 时 ``TracedEnv.step`` 在计数前拒第 cap+1 步，置 ``cap_hit``）；
    ``lang``：``trace_writer.LanguageLog`` 实例或 ``None``（不记语言账本）；``source_call_id``／``chunk_index``：
    当前动作块来自哪次 ``action_model`` 调用、下一步是块内第几个动作；``action_params``／``monitor_params``：
    语言账本 ``open_call`` 的 ``params``。"""

    def __init__(self, writer, open_recorder: Callable | None = None, *, effective_cap: int | None = None,
                 strict_cap: bool = False, lang=None, action_params: dict | None = None,
                 monitor_params: dict | None = None) -> None:
        self.writer = writer
        self.t = 0
        self.subgoal: str | None = None
        self.attempted = 0  # C8 交给环境的步数（含异常步）
        self.observed = 0  # C8 返回有效观测的步数
        self.demo_frames: int | None = None  # 演示帧数（不含初始帧）；None = 还没 reset
        self.actions: list = []  # 每个执行步实际交给环境的动作（原 dtype／shape／bytes）
        self.recorder = None
        self._open_recorder = open_recorder
        self.effective_cap = None if effective_cap is None else int(effective_cap)
        self.strict_cap = bool(strict_cap)
        self.cap_hit = False
        self.lang = lang
        self.source_call_id: str | None = None
        self.chunk_index = 0
        self.action_params = dict(action_params or {})
        self.monitor_params = dict(monitor_params or {})
        self._sha_cache: dict = {}

    def ensure_recorder(self):
        if self.recorder is None and self._open_recorder is not None:
            self.recorder = self._open_recorder()
        return self.recorder

    def frame_sha(self, phase: str, idx: int, frames) -> str | None:
        """Astra ``frames``／``demo`` 列表第 ``idx`` 帧的原像素 sha256（与 trace 的 ``front_sha256`` 同算法），按帧缓存。"""
        from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
        key = (phase, int(idx))
        if key not in self._sha_cache:
            self._sha_cache[key] = tw.image_sha256(frames[idx]) if 0 <= idx < len(frames) else None
        return self._sha_cache[key]

    def step_link(self) -> dict:
        """执行步关联（冻结说明五）：开了语言账本时给 ``log_step`` 的 ``source_call_id``／``chunk_index``，否则空
        （不传时 step 行键集合与旧格式逐字节相同）。每取一次块内序号加一。"""
        if self.lang is None:
            return {}
        link = {"source_call_id": self.source_call_id, "chunk_index": self.chunk_index}
        self.chunk_index += 1
        return link


def _pack_state(obs, i: int = -1):
    import numpy as np  # noqa: PLC0415
    return np.concatenate([np.asarray(obs["joint_state_list"][i]),
                           np.asarray(obs["gripper_state_list"][i])[:1]]).astype(np.float32)


def _goal_text(goal) -> str | None:
    """与 Astra ``runner.episode`` 取任务目标同式：列表取第一个。"""
    if isinstance(goal, list):
        return goal[0] if goal else None
    return goal


class TracedEnv:
    """包住 builder 给的环境：reset 记演示段（C2），step 记执行后的画面、状态、动作与终止标志（C4、C8）。

    录制器只用 ``add_frames``／``add_array``：演示段全部帧（含初始帧）与每个有效观测步的最后一帧各进一次，
    故两路流的帧数 = ``frames_recorded`` = 演示帧数 + 1 + 有效观测步数。

    strict cap（ood）：守卫在计数与动作追加**之前**——已执行 ``effective_cap`` 步再调用即不进真实环境、不记录器、
    不落 trace 行，置 ``cap_hit`` 并抛 ``StepCapReached``。内层是外层 ``EnvSession``（经 ``SessionEnv``）且它的
    ``step_cap`` 同样到顶时，先交给它拒一次（它在进环境之前抛，并置 ``session.cap_hit``）。

    评估仓里 ``ctx.recorder`` 恒为 ``None``：帧、数组、事件由外层 ``EnvSession`` 写进外层录制器，这里只写 trace。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def reset(self, *a, **k):
        import numpy as np  # noqa: PLC0415
        obs, info = self._inner.reset(*a, **k)
        ctx = self._ctx
        rec = ctx.ensure_recorder()
        fronts = [np.asarray(x, dtype=np.uint8) for x in obs.get("front_rgb_list", [])]
        wrists = [np.asarray(x, dtype=np.uint8) for x in obs.get("wrist_rgb_list", [])]
        n = min(len(obs.get("joint_state_list", [])), len(obs.get("gripper_state_list", [])))
        states = [_pack_state(obs, i) for i in range(n)]
        if rec is not None and fronts:
            rec.add_frames("front", np.stack(fronts), tag="reset")
            rec.add_frames("wrist", np.stack(wrists), tag="reset")
            for key in ("joint_state_list", "gripper_state_list"):
                if obs.get(key) is not None and len(obs[key]):
                    rec.add_array(f"reset_{key[:-5]}", np.stack([np.asarray(x) for x in obs[key]]))
        goal = _goal_text(info.get("task_goal"))
        ctx.writer.log_demo(fronts, wrists, states, [goal] if goal is not None else [])
        ctx.demo_frames = max(len(fronts) - 1, 0)
        return obs, info

    def step(self, action):
        import numpy as np  # noqa: PLC0415
        ctx = self._ctx
        if ctx.strict_cap and ctx.effective_cap is not None and ctx.attempted >= ctx.effective_cap:
            ctx.cap_hit = True
            rec = ctx.recorder
            if rec is not None and hasattr(rec, "add_event"):
                rec.add_event({"kind": "step_cap_reached", "step": ctx.attempted, "cap": ctx.effective_cap})
            print(f"ASTRA_STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap} rejected_step={ctx.attempted + 1}",
                  flush=True)
            if _session_would_refuse(self._inner):
                # 外层 EnvSession 自带同一上限：交给它拒（不进真实环境），它置 ``session.cap_hit`` 并记录器事件，
                # 外层结果的 cap_hit／timeout 与本驱动一致
                try:
                    self._inner.step(action)
                except Exception as exc:  # noqa: BLE001 EnvSession.StepCapReached
                    raise StepCapReached(f"STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap}") from exc
            raise StepCapReached(f"STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap}")
        rec = ctx.ensure_recorder()
        ctx.attempted += 1
        ctx.t = n = ctx.attempted
        a = np.array(action, copy=True)
        ctx.actions.append(a)
        link = ctx.step_link()
        if rec is not None:
            rec.add_array("exec_action", a, step=n - 1)
        try:
            out = self._inner.step(action)
        except BaseException as exc:
            ctx.writer.log_missing_step(step=n, action=a, reason=f"env_step_exception:{type(exc).__name__}",
                                        subgoal=ctx.subgoal, **link)
            raise
        obs, reward, terminated, truncated, info = out
        status = info.get("status") if isinstance(info, dict) else None
        if obs is None or not obs.get("front_rgb_list") or not obs.get("wrist_rgb_list"):
            ctx.writer.log_missing_step(step=n, action=a, reason="obs_none", subgoal=ctx.subgoal, **link)
            return out
        front = np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8)
        wrist = np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8)
        if rec is not None:
            rec.add_frames("front", front, tag=f"step{n}")
            rec.add_frames("wrist", wrist, tag=f"step{n}")
            rec.add_array("joint_state", np.asarray(obs["joint_state_list"][-1]), step=n - 1)
            rec.add_array("gripper_state", np.asarray(obs["gripper_state_list"][-1]), step=n - 1)
        ctx.observed += 1
        ctx.writer.log_step(step=n, front=front, wrist=wrist, state=_pack_state(obs), action=a, subgoal=ctx.subgoal,
                            terminated=terminated, truncated=truncated, status=status, **link)
        return out

    def close(self):
        return self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _session_would_refuse(inner) -> bool:
    """内层是带 ``step_cap`` 的外层会话、且已执行步数到顶（再调 ``step`` 会在进环境之前被它拒）。"""
    cap = getattr(inner, "step_cap", None)
    steps = getattr(inner, "steps", None)
    return isinstance(cap, int) and isinstance(steps, int) and steps >= cap


class TracedBuilder:
    """只暴露 ``runner.episode`` 用到的两个方法；``make_env_for_episode`` 只传局号（评估仓里内层是 ``SessionBuilder``）。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def make_env_for_episode(self, episode):
        return TracedEnv(self._inner.make_env_for_episode(episode), self._ctx)

    def resolve_episode(self, episode):
        return self._inner.resolve_episode(episode)


# ── 语言账本（冻结说明五；LanguageLog 由 R6 的 trace_writer 提供） ─────────────

#: 附图变换描述：无损 PNG（规划／监视的单帧附件）、``core.sheets`` 拼图（JPEG q92）、websocket 原数组（VLA）
PNG_TRANSFORM = {"resize": None, "crop": None, "layout": None, "encode": "png"}
SHEET_TRANSFORM = {"resize": None, "crop": None, "layout": "core.sheets 4x4 grid, 256x280 cell, frame label",
                   "encode": "jpeg q92"}
RAW_TRANSFORM = {"resize": None, "crop": None, "layout": None, "encode": "msgpack_numpy"}
#: Astra 拼图每页帧数（``core.sheets``）
SHEET_PAGE = 16


def language_log_cls():
    """可选探测：``trace_writer.LanguageLog``（R6）；不存在时返回 ``None``（不记语言账本、不报错）。"""
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    return getattr(tw, "LanguageLog", None)


def _image_ref(slot: int, ref: str, phase: str, frame_idx, cam: str, raw_sha: str | None, *, transform: dict,
               sources: list | None = None, encoded: Path | None = None) -> dict:
    """附图引用（不存图片本身）：单帧的 ``sources`` 为它自己；拼图给有序来源帧列表、``frame_idx=None``。"""
    if sources is None:
        sources = [{"phase": phase, "frame_idx": frame_idx, "cam": cam, "raw_sha256": raw_sha}]
    enc = file_sha256(encoded) if encoded is not None and Path(encoded).is_file() else None
    return {"slot": slot, "ref": ref, "phase": phase, "frame_idx": frame_idx, "cam": cam, "raw_sha256": raw_sha,
            "sources": sources, "transform": dict(transform), "encoded_sha256": enc}


def _sheet_sources(ctx: TraceContext, phase: str, indices: list, frames) -> list:
    return [{"phase": phase, "frame_idx": int(i), "cam": "front", "raw_sha256": ctx.frame_sha(phase, int(i), frames)}
            for i in indices]


def planner_images(ctx: TraceContext, out: Path, request: dict, frames, demo, *, wrist=None,
                   command_start: int | None = None) -> list:
    """按 ``request.json`` 的 ``images`` 顺序给规划／复审请求的附图引用。

    exec 帧号 = Astra ``frames`` 下标（0 = reset 后初始帧，即 trace demo 段末帧；k = 第 k 步执行后，即 trace step k）；
    demo 帧号 = 演示段下标。拼图页 p 覆盖下标列表的第 ``16p`` 至 ``16p+15`` 个。"""
    now = len(frames) - 1
    memory = [int(i) for i in request.get("memory_frame_ids") or []]
    demo_idx = list(range(len(demo or [])))
    images = []
    for slot, name in enumerate(request.get("images") or []):
        path = out / name
        if name == "current.png":
            images.append(_image_ref(slot, "current", "exec", now, "front", ctx.frame_sha("exec", now, frames),
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name == "execution_start.png":
            images.append(_image_ref(slot, "keyframe", "exec", 0, "front", ctx.frame_sha("exec", 0, frames),
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name == "command_start.png" and command_start is not None:
            images.append(_image_ref(slot, "command_start", "exec", int(command_start), "front",
                                     ctx.frame_sha("exec", int(command_start), frames), transform=PNG_TRANSFORM,
                                     encoded=path))
        elif name == "current_wrist.png":
            from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
            images.append(_image_ref(slot, "wrist", "exec", now, "wrist",
                                     tw.image_sha256(wrist) if wrist is not None else None,
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name.startswith(("demo_", "memory_")):
            phase, ref = ("demo", "demo_sheet") if name.startswith("demo_") else ("exec", "memory_sheet")
            page = int(Path(name).stem.split("_")[1])
            pool = demo_idx if phase == "demo" else memory
            src_frames = demo if phase == "demo" else frames
            idx = pool[page * SHEET_PAGE:(page + 1) * SHEET_PAGE]
            images.append(_image_ref(slot, ref, phase, None, "front", None, transform=SHEET_TRANSFORM,
                                     sources=_sheet_sources(ctx, phase, idx, src_frames), encoded=path))
        else:  # 未知附件：只记文件哈希，不猜来源
            images.append(_image_ref(slot, None, "exec", None, "front", None, transform=PNG_TRANSFORM, sources=[],
                                     encoded=path))
    return images


class PlannerLanguageCall:
    """一次规划（或第二按钮复审）请求的语言账本记录器。

    ``wrap(responder)`` 返回给 Astra ``Planner`` 用的发送方：Planner 写完 ``prompt.txt``／``request.json``／附图后调用它，
    它**先**开调用、写 ``in`` 消息（prompt 全文 + 附图引用）落盘，再交给真实发送方；发送方若支持
    ``transport_hook``（本文件的 ``GuardedResponsesClient``），每次传输重试（attempt ≥ 1）把上一个调用记 ``error``、
    另开一个 ``transport_attempt=k`` 的调用并重写 ``in`` 消息。``finish`` 在 Planner 返回或抛错后写回复原文与收尾。"""

    def __init__(self, ctx: TraceContext, image_fn: Callable[[Path, dict], list], kind: str) -> None:
        self.ctx = ctx
        self.image_fn = image_fn
        self.kind = kind
        self.call_id: str | None = None
        self.out: Path | None = None
        self.prompt: str | None = None
        self.images: list | None = None
        self.params: dict | None = None
        self.step = ctx.t

    def _open(self, transport_attempt: int) -> None:
        lang = self.ctx.lang
        self.call_id = lang.open_call("planner", self.step, params=self.params, transport_attempt=transport_attempt)
        lang.message(self.call_id, dir="in", role="user", text=self.prompt, images=self.images)

    def wrap(self, inner: Callable) -> Callable:
        def responder(out):
            out = Path(out)
            self.out = out
            request = json.loads((out / "request.json").read_text())
            self.prompt = (out / "prompt.txt").read_text()
            self.images = self.image_fn(out, request)
            self.params = {"temperature": None, "max_tokens": PLANNER_MAX_OUTPUT_TOKENS,
                           "model_id": request.get("model"), "adapter_sha": None, "effort": request.get("effort"),
                           "kind": request.get("kind", self.kind), "request_id": out.name, "cloud_seed": None,
                           "policy_seed": self.ctx.action_params.get("policy_seed")}
            self._open(0)  # 发送前落盘
            has_hook = hasattr(inner, "transport_hook")
            if has_hook:
                previous = inner.transport_hook
                inner.transport_hook = self.transport
            try:
                return inner(out)
            finally:
                if has_hook:
                    inner.transport_hook = previous
        return responder

    def transport(self, out, attempt: int) -> None:
        """``GuardedResponsesClient._send`` 每次传输尝试前调用；attempt 0 即 ``wrap`` 已开的那个调用。"""
        if int(attempt) == 0 or self.call_id is None:
            return
        self.ctx.lang.close_call(self.call_id, status="error")
        self._open(int(attempt))

    def finish(self, *, parsed=None, fallback=None, failed: bool = False) -> None:
        if self.call_id is None:
            return
        lang = self.ctx.lang
        response = None
        if self.out is not None and (self.out / "response.json").is_file():
            try:
                response = json.loads((self.out / "response.json").read_text())
            except ValueError:
                response = None
        replied = bool(response) and response.get("status") == "ok"
        if replied:
            lang.message(self.call_id, dir="out", role="assistant", text=response.get("text"))
            if failed and fallback is None:  # 有回复但不合约定（如复审不是 true/false）
                fallback = "model_response_error"
        lang.close_call(self.call_id, status="reply" if replied else "error", parsed=parsed, fallback=fallback)
        self.call_id = None


def _monitor_images(ctx: TraceContext, ids, frames, command_start: int, wrist) -> list:
    """监视器 10 张图（``input_contract.build_input`` 顺序）：最近 8 帧、本条命令起点帧、当前腕部帧。"""
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    images = [_image_ref(slot, "recent", "exec", int(fid), "front", ctx.frame_sha("exec", int(fid), frames),
                         transform=PNG_TRANSFORM) for slot, fid in enumerate(ids or [])]
    images.append(_image_ref(len(images), "command_start", "exec", int(command_start), "front",
                             ctx.frame_sha("exec", int(command_start), frames), transform=PNG_TRANSFORM))
    images.append(_image_ref(len(images), "wrist", "exec", len(frames) - 1, "wrist",
                             tw.image_sha256(wrist) if wrist is not None else None, transform=PNG_TRANSFORM))
    return images


class TracedClient:
    """包住 VLA websocket 客户端：记每次 ``infer`` 的规范化请求与完整动作块；开了语言账本时每次 ``infer`` 记一个
    ``action_model`` 调用（``in``：``prompt``／``grounded_subgoal``／``simple_subgoal`` 字段原文与两张当前帧引用，
    发送前落盘；Astra 的 VLA 服务无审计回包，``server_final_text=None``）。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def reset(self):
        self._ctx.writer.log_request("vla_reset", b"", step=self._ctx.t)
        return self._inner.reset()

    def _open_lang(self, element) -> str | None:
        ctx = self._ctx
        if ctx.lang is None:
            return None
        from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
        call_id = ctx.lang.open_call("action_model", ctx.t, params=dict(ctx.action_params))
        fields = {k: element.get(k) for k in ("prompt", "grounded_subgoal", "simple_subgoal") if k in element}
        images = [_image_ref(0, "current", "exec", ctx.t, "front", tw.image_sha256(element.get("observation/image")),
                             transform=RAW_TRANSFORM),
                  _image_ref(1, "wrist", "exec", ctx.t, "wrist",
                             tw.image_sha256(element.get("observation/wrist_image")), transform=RAW_TRANSFORM)]
        ctx.lang.message(call_id, dir="in", role="fields", text=fields, images=images)
        return call_id

    def infer(self, element):
        ctx = self._ctx
        ctx.subgoal = element.get("grounded_subgoal")
        call_id = self._open_lang(element)
        ctx.source_call_id, ctx.chunk_index = call_id, 0
        ctx.writer.log_request("vla_infer", _canonical(element), step=ctx.t)
        try:
            out = self._inner.infer(element)
        except BaseException:
            if call_id is not None:
                ctx.lang.close_call(call_id, status="error")
            raise
        if call_id is not None:
            ctx.lang.close_call(call_id, status="reply", server_final_text=None, server_truncated=None)
        ctx.writer.log_response(out.get("actions"), step=ctx.t)
        return out


def _spool_payload(out: Path) -> bytes:
    """规划请求的规范化字节：``request.json``（去掉 id／created）+ ``prompt.txt`` + 各附图字节的 sha256。"""
    request = json.loads((out / "request.json").read_text())
    images = {name: file_sha256(out / name) for name in request.get("images", []) if (out / name).is_file()}
    request = {k: v for k, v in request.items() if k not in ("id", "created")}
    prompt = (out / "prompt.txt").read_text() if (out / "prompt.txt").is_file() else ""
    return json.dumps({"request": request, "prompt": prompt, "images": images}, sort_keys=True,
                      ensure_ascii=False).encode()


class TracedPlanner:
    """包住 Astra ``Planner``：每次规划／复审请求按 spool 目录内容记一行 request；开了语言账本时临时把
    ``Planner.responder`` 换成 ``PlannerLanguageCall.wrap`` 的发送方（调用结束即还原）。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def _log(self, name: str, rid: str) -> None:
        out = Path(self._inner.spool) / rid
        if (out / "request.json").is_file():
            self._ctx.writer.log_request(name, _spool_payload(out), step=self._ctx.t)

    def _call(self, method: str, image_fn: Callable, parse: Callable, a, k):
        ctx = self._ctx
        fn = getattr(self._inner, method)
        responder = getattr(self._inner, "responder", None)
        if ctx.lang is None or responder is None:
            return fn(*a, **k)
        call = PlannerLanguageCall(ctx, image_fn, method)
        self._inner.responder = call.wrap(responder)
        try:
            result = fn(*a, **k)
        except BaseException:
            call.finish(failed=True)
            raise
        finally:
            self._inner.responder = responder
        parsed, fallback = parse(result)
        call.finish(parsed=parsed, fallback=fallback)
        return result

    def predict(self, task, goal, frames, demo, memory, completed, issued, episode, t):
        a = (task, goal, frames, demo, memory, completed, issued, episode, t)
        result = self._call(
            "predict", lambda out, req: planner_images(self._ctx, out, req, frames, demo),
            lambda r: (r[0], None if r[0] is not None else "continue_last"), a, {})
        self._log("planner", result[1])
        return result

    def review_second_button(self, goal, frames, wrist, subgoal, command_start, completed, issued, episode, t):
        a = (goal, frames, wrist, subgoal, command_start, completed, issued, episode, t)
        result = self._call(
            "review_second_button",
            lambda out, req: planner_images(self._ctx, out, req, frames, [], wrist=wrist, command_start=command_start),
            lambda r: (r[0], None), a, {})
        self._log("planner_review", result[1])
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TracedMonitor:
    """包住监视器：记 ``input.json``（去掉图片路径，换成各图 sha256）。开了语言账本时每次 ``predict`` 记一个
    ``monitor`` 调用：推理**之前**按上游 ``input_contract.from_observations`` 落 system、user（10 张图引用），
    之后写回复原文（``response.json`` 的 ``text``）与 ``parsed`` 布尔。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def _open_lang(self, task, goal, subgoal, frames, command_start, wrist) -> str | None:
        ctx = self._ctx
        if ctx.lang is None:
            return None
        try:
            import input_contract  # noqa: PLC0415  Astra 的监视器输入契约（与 Monitor.predict 第一行同一函数）
            sample, ids = input_contract.from_observations(task, goal, subgoal, frames, command_start, wrist)
            system, user = sample["messages"][0]["content"], sample["messages"][1]["content"]
        except Exception:  # noqa: BLE001 构造失败时上游 predict 也会同样失败；仍先落一条可读的输入
            ids, system = None, None
            user = json.dumps({"task": task, "goal": goal, "subgoal": subgoal, "command_start": command_start},
                              ensure_ascii=False)
        call_id = ctx.lang.open_call("monitor", ctx.t, params=dict(ctx.monitor_params))
        ctx.lang.message(call_id, dir="in", role="system", text=system)
        images = _monitor_images(ctx, ids, frames, command_start, wrist) if ids is not None else []
        ctx.lang.message(call_id, dir="in", role="user", text=user, images=images)
        return call_id

    def _close_lang(self, call_id: str, out: Path, pred, failed: bool) -> None:
        lang = self._ctx.lang
        text = None
        response = Path(out) / "response.json"
        if response.is_file():
            try:
                text = json.loads(response.read_text()).get("text")
            except ValueError:
                text = None
        if text is not None:
            lang.message(call_id, dir="out", role="assistant", text=text)
        if failed:
            lang.close_call(call_id, status="reply" if text is not None else "error", parsed=None,
                            fallback="model_response_error" if text is not None else None)
        else:
            lang.close_call(call_id, status="reply", parsed=pred)

    def predict(self, task, goal, subgoal, frames, command_start, wrist, out):
        call_id = self._open_lang(task, goal, subgoal, frames, command_start, wrist)
        try:
            result = self._inner.predict(task, goal, subgoal, frames, command_start, wrist, out)
        except BaseException:
            if call_id is not None:
                self._close_lang(call_id, out, None, failed=True)
            raise
        if call_id is not None:
            self._close_lang(call_id, out, result[0], failed=False)
        sample_path = Path(out) / "input.json"
        if sample_path.is_file():
            sample = json.loads(sample_path.read_text())
            sample["images"] = [file_sha256(Path(p)) if Path(p).is_file() else None for p in sample.get("images", [])]
            payload = json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()
        else:
            payload = _canonical({"task": task, "goal": goal, "subgoal": subgoal, "command_start": command_start})
        self._ctx.writer.log_request("monitor", payload, step=self._ctx.t)
        return result



# ── 一局：runner.episode + Traced* 包装 + 收尾 ─────────────────────────────

def terminal_of(result: dict | None) -> str:
    """Astra 结果的 ``status`` 已是 success／fail／timeout／error（``runner.episode`` 把其他环境状态归为 error）；
    循环走满 ``max_steps`` 与环境报 timeout 都是 ``timeout``；其余（含驱动异常、无结果）一律 ``error``。"""
    status = (result or {}).get("status")
    return status if status in TERMINALS else "error"


def episode_key(task: str, identity: dict) -> str:
    """局 key：``<task>_<tier>_<seed>``（与 ``EpisodeSpec.key`` 同式）。"""
    return f"{task}_{identity['tier']}_{int(identity['seed'])}"


def strict_cap_of(dataset: str) -> bool:
    """ood 为 strict，hard-verify 不是；未知数据集按非 strict（配对检查早已拒绝）。"""
    return bool(DATASET_STRICT_CAP.get(dataset, False))


def write_exec_actions(path: Path, actions: list) -> None:
    """每个执行步的原动作写 ``exec_action__%05d``（0 起步序号）；一旦写就每步都有键。

    ``arrays.npz`` 唯一允许的写法是 ``trace_writer.merge_write_npz``（与 ``TraceWriter.close`` 和外层录制器写的同键
    逐项核对后合并、原子替换）；该函数不存在时保持旧的 ``np.savez``。"""
    import numpy as np  # noqa: PLC0415
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    if not actions:
        return
    mapping = {f"exec_action__{i:05d}": a for i, a in enumerate(actions)}
    merge = getattr(tw, "merge_write_npz", None)
    if merge is not None:
        merge(Path(path), mapping)
    else:
        np.savez(path, **mapping)


def prompt_digests(champ: Path) -> dict[str, str]:
    """``prompts/*.md`` 与 ``prompts/index.json`` 的 sha256（键为相对 ``examples/champ`` 的路径）。"""
    prompts = Path(champ) / "prompts"
    files = sorted(prompts.glob("*.md")) + ([prompts / "index.json"] if (prompts / "index.json").is_file() else [])
    return {str(p.relative_to(champ)): file_sha256(p) for p in files}


def write_provenance(path: Path, *, astra, args, key: str, strict_cap: bool, language: bool, attempt: int) -> dict:
    """尝试目录来源清单：Astra 源码与 prompts 的 sha256、模型 seed（云端无 seed 接口，``cloud_seed=null``）、实际 cap。"""
    doc = {"route": ROUTE, "key": key, "attempt": int(attempt), "dataset": args.dataset,
           "policy_seed": getattr(args, "policy_seed", None), "server_seed": getattr(args, "policy_seed", None),
           "cloud_seed": None, "cloud_seed_note": "planner/monitor cloud API has no seed interface; not fabricated",
           "effective_cap": int(args.max_steps), "strict_cap": bool(strict_cap),
           "astra_root": str(getattr(astra, "root", "")), "astra_sources": source_digests(astra.champ),
           "prompts": prompt_digests(astra.champ), "language_log": "language.jsonl" if language else None}
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return doc


def group_stop_file(group_dir: Path) -> Path:
    """停机口 ``group_*/STOP.json``（费用守卫到线写；``ResponsesClient._send`` 与每局 ``reset`` 都查）。"""
    return Path(group_dir) / "STOP.json"


def run_one(args, task, ep, identity, builder, monitor, planner, client, astra, *, attempt: int, trace_dir: Path,
            strict_cap: bool | None = None) -> dict:
    """一局：建尝试目录 ``<args.output>/<task>/ep<NNN>/<key>.a<attempt>/`` 与 ``trace_dir/trace.jsonl``，经委托包装调
    Astra 的 ``runner.episode``；``finally`` 里统一收尾：写 ``arrays.npz`` → 语言账本收尾 → trace ``close`` →
    Astra ``result.json`` 追加映射。驱动异常（``runner.episode`` 自己不接的 ``BaseException``）照样收尾后再抛。"""
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    ep_dir = Path(args.output) / task / f"ep{ep:03d}"
    key = episode_key(task, identity)
    a_dir = ep_dir / f"{key}.a{int(attempt)}"
    a_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    policy_seed = getattr(args, "policy_seed", None)
    strict = strict_cap_of(args.dataset) if strict_cap is None else bool(strict_cap)
    trace_identity = {"task": task, "dataset": args.dataset, **identity, "builder_episode": int(ep), "key": key,
                      "attempt": int(attempt), "policy_seed": policy_seed}
    header_extra = _supported_kwargs(tw.TraceWriter, policy_seed=policy_seed, effective_cap=int(args.max_steps),
                                     strict_cap=strict)
    writer = tw.TraceWriter(trace_dir / "trace.jsonl", route=ROUTE, identity=trace_identity, max_steps=args.max_steps,
                            **header_extra)
    lang_cls = language_log_cls()
    lang = lang_cls(trace_dir / "language.jsonl") if lang_cls is not None else None
    write_provenance(a_dir / PROVENANCE_FILE, astra=astra, args=args, key=key, strict_cap=strict,
                     language=lang is not None, attempt=attempt)
    ctx = TraceContext(writer, open_recorder=None, effective_cap=int(args.max_steps), strict_cap=strict, lang=lang,
                       action_params={"temperature": None, "max_tokens": None, "model_id": "mme_vla_suite",
                                      "adapter_sha": None, "checkpoint": str(args.vla_checkpoint),
                                      "policy_seed": policy_seed},
                       monitor_params={**MONITOR_PARAMS, "model_id": getattr(args, "monitor_base", None),
                                       "adapter": str(args.monitor_adapter), "adapter_sha": None})
    result: dict | None = None
    driver_error: BaseException | None = None
    try:
        result = astra.runner.episode(args, task, ep, TracedBuilder(builder, ctx), TracedMonitor(monitor, ctx),
                                      TracedPlanner(planner, ctx), TracedClient(client, ctx))
        return result
    except BaseException as exc:
        driver_error = exc
        raise
    finally:
        _finish_one(args, ep_dir, a_dir, trace_dir, key, int(attempt), ctx, writer, result, driver_error)


def _supported_kwargs(fn: Callable, **candidates) -> dict:
    """可选探测：只把 ``fn`` 签名里显式声明的关键字传过去。"""
    import inspect  # noqa: PLC0415
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in candidates.items() if k in params}


def _finish_one(args, ep_dir: Path, a_dir: Path, trace_dir: Path, key: str, attempt: int, ctx: TraceContext,
                writer, result: dict | None, driver_error: BaseException | None) -> None:
    terminal = terminal_of(result) if driver_error is None else "error"
    if ctx.cap_hit and driver_error is None:  # strict cap 拒第 cap+1 步：Astra 记的 error 改为 timeout
        terminal = "timeout"
    has_demo = ctx.demo_frames is not None
    no_frame = not has_demo
    demo_frames = ctx.demo_frames if has_demo else 0
    frames_recorded = demo_frames + 1 + ctx.observed if has_demo else 0
    write_exec_actions(trace_dir / "arrays.npz", ctx.actions)
    if ctx.lang is not None:
        ctx.lang.close()  # 仍未关闭的调用补 cancelled
    extra = {"policy_seed": getattr(args, "policy_seed", None), "cloud_seed": None, "strict_cap": ctx.strict_cap,
             "effective_cap": ctx.effective_cap, "cap_hit": ctx.cap_hit,
             "language": "language.jsonl" if ctx.lang is not None else None, "provenance": PROVENANCE_FILE}
    if result is not None:
        extra.update({k: result.get(k) for k in ("planner_calls", "monitor_calls", "review_calls", "error")})
    if driver_error is not None:
        extra["driver_exception"] = f"{type(driver_error).__name__}: {driver_error}"[:800]
    if no_frame and terminal != "error":  # 理论上不会发生：没 reset 却正常结束；按 error 记，避免假终态
        terminal = "error"
    writer.close(status=terminal, terminal_reason=terminal, demo_frames=demo_frames, steps_attempted=ctx.attempted,
                 steps_observed=ctx.observed, frames_recorded=frames_recorded, omitted_timeout_frames=0,
                 no_frame=no_frame, **extra)
    if result is not None:
        rel = lambda p: os.path.relpath(p, ep_dir)  # noqa: E731
        if ctx.cap_hit and driver_error is None:
            result["status"] = "timeout"  # 停机规则②不把 strict cap 计为基础设施错误
        result.update(route=ROUTE, key=key, attempt=int(attempt), episode_dir=a_dir.name,
                      trace=rel(trace_dir / "trace.jsonl"), exec_steps=ctx.attempted, steps_observed=ctx.observed,
                      frames_recorded=frames_recorded, demo_frames=demo_frames, terminal_reason=terminal,
                      policy_seed=getattr(args, "policy_seed", None), cloud_seed=None,
                      strict_cap=ctx.strict_cap, effective_cap=ctx.effective_cap, cap_hit=ctx.cap_hit)
        if (ep_dir / "result.json").is_file():
            from core import atomic_json  # noqa: PLC0415  Astra 的原子写（与它写 result.json 同一函数）
            atomic_json(ep_dir / "result.json", result)


# ── 费用硬上限：守卫状态同步预留 + 子类化 ResponsesClient ─────────────────────

_GUARD_MOD = None


def guard_module():
    """费用守卫模块 ``robomme_ood_eval.servers.astra_cost_guard``（预留协议、锁、文件名的唯一来源）。"""
    global _GUARD_MOD
    if _GUARD_MOD is None:
        from robomme_ood_eval.servers import astra_cost_guard  # noqa: PLC0415
        _GUARD_MOD = astra_cost_guard
    return _GUARD_MOD


class GuardRefused(RuntimeError):
    """发送前拒发（守卫失联、守卫已停、预留失败、局数到顶）。消息以 ``Planner API`` 开头：经 Planner 包成
    ``Planner bridge failed`` 后同样触发停机规则①（``PLANNER_STOP_PREFIXES``）单次即停。"""


def _image_tokens_upper(width: int, height: int) -> int:
    """单张 ``detail=high`` 图片输入 token 的保守上界：取两种公开计价口径的较大者。

    ① 512 切块：先缩到 2048×2048 内、再把短边缩到 768，``85 + 170 × 块数``；
    ② 32 像素小块：小块数（上限 1536）× 2.5（各型号乘数的最大值取整上浮）。"""
    import math  # noqa: PLC0415
    w, h = max(int(width), 1), max(int(height), 1)
    scale = min(1.0, 2048 / max(w, h))
    w1, h1 = w * scale, h * scale
    scale2 = min(1.0, 768 / min(w1, h1))
    w2, h2 = w1 * scale2, h1 * scale2
    tiles = math.ceil(w2 / 512) * math.ceil(h2 / 512)
    patches = min(math.ceil(w / 32) * math.ceil(h / 32), 1536)
    return max(85 + 170 * tiles, math.ceil(patches * 2.5))


def _image_size(data_url: str) -> tuple[int, int]:
    """从 ``data:<mime>;base64,<...>`` 解出图片尺寸；解不出按 2048×2048（最坏）计。"""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=False)
        from PIL import Image  # noqa: PLC0415
        with Image.open(io.BytesIO(raw)) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return 2048, 2048


def estimate_request(payload: dict) -> dict:
    """按本次实际请求估输入 token 上界：文本按 UTF-8 字节数（每 token 至少 1 字节），图片按 ``_image_tokens_upper``，
    另加 64 的格式开销；输出按请求自带的 ``max_output_tokens``。"""
    text_bytes, images = 0, []
    for item in payload.get("input") or []:
        for part in item.get("content") or []:
            if part.get("type") == "input_text":
                text_bytes += len(str(part.get("text", "")).encode("utf-8"))
            elif part.get("type") == "input_image":
                images.append(_image_tokens_upper(*_image_size(str(part.get("image_url", "")))))
    return {"input_tokens": text_bytes + sum(images) + 64, "text_bytes": text_bytes, "images": len(images),
            "image_tokens": sum(images), "max_output_tokens": int(payload.get("max_output_tokens") or 0)}


class CostGate:
    """runner 一侧的守卫协议：发送前同步读守卫状态、原子预留单次最坏费用；登记局数。

    状态文件由 ``astra_cost_guard.py --state`` 每轮写；预留文件与锁由守卫模块定名。任何读不到、过期（心跳超过
    ``GUARD_HEARTBEAT_TIMEOUT_S``）、守卫已退出或已停的情况都拒发——宁停不发。"""

    def __init__(self, state_path: str | Path, *, clock: Callable[[], float] = time.time,
                 heartbeat_timeout: float = GUARD_HEARTBEAT_TIMEOUT_S) -> None:
        self.state_path = Path(state_path)
        self.clock = clock
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.guard = guard_module()

    def read_state(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError) as exc:
            raise GuardRefused(f"Planner API cost guard unavailable: state unreadable ({type(exc).__name__})") from None
        if state.get("schema") != self.guard.STATE_SCHEMA:
            raise GuardRefused(f"Planner API cost guard unavailable: schema {state.get('schema')!r}")
        age = self.clock() - float(state.get("heartbeat") or 0)
        if state.get("exited"):
            raise GuardRefused("Planner API cost guard unavailable: guard exited")
        if not -1.0 <= age <= self.heartbeat_timeout:
            raise GuardRefused(f"Planner API cost guard unavailable: heartbeat age {age:.1f}s > {self.heartbeat_timeout:g}s")
        if state.get("stop"):
            raise GuardRefused(f"Planner API cost guard stopped: {state.get('reason')}")
        return state

    def cap(self, state: dict) -> float:
        """生效上限 = min(守卫的 --cap, 本轮硬上限)；守卫状态被改大也不放宽。"""
        return min(float(state.get("cap") or 0.0), float(self.guard.HARD_CAP_USD))

    def worst_usd(self, state: dict, estimate: dict) -> float:
        prices = state["prices"]
        out_tokens = max(int(estimate["max_output_tokens"]), int(state.get("max_output_tokens") or 0))
        usd = (estimate["input_tokens"] * float(prices["input"]) + out_tokens * float(prices["output"])) / 1e6
        if prices.get("reasoning_billed_separately"):
            usd += out_tokens * float(prices["output"]) / 1e6
        return usd

    def reserve(self, rid: str, payload: dict) -> dict:
        """锁内：读状态 → 已计 + 未结预留 + 本次最坏 > 上限即拒；否则记预留（同一 rid 重试不重复计，取较大者）。"""
        estimate = estimate_request(payload)
        g = self.guard
        with g.locked(self.state_path):
            state = self.read_state()
            doc = g.load_reservations(self.state_path)
            worst = self.worst_usd(state, estimate)
            existing = doc["reservations"].get(rid)
            others = g.outstanding_reservations(
                {"reservations": {k: v for k, v in doc["reservations"].items() if k != rid}}, state.get("counted") or [])
            committed = float(state.get("committed_usd") or 0.0)
            amount = max(worst, float(existing["usd"])) if existing and existing.get("state") != "released" else worst
            cap = self.cap(state)
            if committed + others + amount > cap:
                raise GuardRefused(f"Planner API cost reservation refused: committed={committed:.4f} "
                                   f"reserved={others:.4f} worst={amount:.4f} cap={cap:g}")
            doc["reservations"][rid] = {"usd": amount, "state": "reserved", "time": self.clock(), **estimate}
            g.save_reservations(self.state_path, doc)
        return {"usd": amount, **estimate}

    def mark(self, rid: str, state_name: str) -> None:
        g = self.guard
        with g.locked(self.state_path):
            doc = g.load_reservations(self.state_path)
            if rid in doc["reservations"]:
                doc["reservations"][rid]["state"] = state_name
                doc["reservations"][rid][f"{state_name}_at"] = self.clock()
                g.save_reservations(self.state_path, doc)

    def register_episode(self, episode_id: str) -> int:
        """局数硬上限：同一预留文件里跨 RUN 登记；已登记的同一局不重复计；第 3 局拒绝。"""
        g = self.guard
        with g.locked(self.state_path):
            self.read_state()
            doc = g.load_reservations(self.state_path)
            if episode_id not in doc["episodes"]:
                if len(doc["episodes"]) >= g.ASTRA_MAX_EPISODES:
                    print(f"ASTRA_STOP reason=episode_cap episode={episode_id} used={len(doc['episodes'])}", flush=True)
                    raise AstraStop("episode_cap", f"Astra episode cap {g.ASTRA_MAX_EPISODES} reached")
                doc["episodes"].append(episode_id)
                g.save_reservations(self.state_path, doc)
            return len(doc["episodes"])


def registered_episodes(state_path: str | Path) -> list:
    """预留文件里已登记的局（跨进程、跨 RUN 累计；账本保留，不新建绕额度）。"""
    g = guard_module()
    with g.locked(Path(state_path)):
        return list(g.load_reservations(Path(state_path))["episodes"])


def check_episode_cap(state_path: str | Path) -> int:
    """局数硬上限：已登记局数到 ``ASTRA_MAX_EPISODES`` 即抛 ``AstraStop(episode_cap)``；返回已登记局数。"""
    limit = guard_module().ASTRA_MAX_EPISODES
    used = len(registered_episodes(state_path))
    if used >= limit:
        print(f"ASTRA_STOP reason=episode_cap used={used} cap={limit}", flush=True)
        raise AstraStop("episode_cap", f"Astra episode cap {limit} reached")
    return used


_GUARDED_CLASSES: dict = {}


def guarded_client_class(api_client):
    """返回第三方 ``api_client.ResponsesClient`` 的子类（按基类缓存）；第三方文件零改动。

    覆写 ``_send``：每一次真正 ``urlopen`` 之前依次 ① 查 ``group_*/STOP.json``；② 同步读守卫状态并原子预留
    本次最坏费用（同一请求的 429 重试复用同一份预留）；③ 等待第三方的 20 秒发送间隔；④ 等待后再查一次
    STOP 与守卫状态。任一不通过即写 ``guard_refused.json``、释放预留并抛错（第三方 ``__call__`` 把它记成
    ``status=error`` 的 ``response.json``）。其余（HTTP 错误记录、429 有界退避、脱敏）与第三方 ``_send`` 逐项同式。"""
    base = api_client.ResponsesClient
    if base in _GUARDED_CLASSES:
        return _GUARDED_CLASSES[base]

    class GuardedResponsesClient(base):
        #: 语言账本的传输尝试回调 ``(out, attempt)``（``PlannerLanguageCall.transport``）；``None`` 时不调用。
        #: 只记录、不改变发送与费用守卫的任何判断。
        transport_hook = None

        def __init__(self, key, gate: CostGate, *, sleep: Callable[[float], None] = time.sleep,
                     monotonic: Callable[[], float] = time.monotonic) -> None:
            super().__init__(key)
            self.gate = gate
            self._sleep = sleep
            self._monotonic = monotonic

        @staticmethod
        def _stop_requested(out: Path) -> bool:
            return any(p.name in GROUP_DIR_NAMES and (p / "STOP.json").exists() for p in out.parents)

        def _refuse(self, out: Path, rid: str, message: str):
            api_client.atomic_json(out / guard_module().REFUSED_MARKER, {"time": time.time(), "reason": message})
            try:
                self.gate.mark(rid, "released")
            except Exception:  # noqa: BLE001 守卫已失联时释放也可能失败；拒发本身不受影响
                pass
            print(f"ASTRA_GUARD_REFUSED request={rid} reason={json.dumps(message, ensure_ascii=False)}", flush=True)
            raise RuntimeError(message)

        def _gate_before_send(self, request, out: Path, rid: str) -> None:
            if self._stop_requested(out):
                self._refuse(out, rid, "Host requested stop; no new API request")
            try:
                self.gate.reserve(rid, json.loads(request.data))
            except GuardRefused as exc:
                self._refuse(out, rid, str(exc))
            self._sleep(max(0, self.next_request_at - self._monotonic()))
            if self._stop_requested(out):  # 等待间隔里 STOP 可能已到
                self._refuse(out, rid, "Host requested stop; no new API request")
            try:
                self.gate.read_state()
            except GuardRefused as exc:
                self._refuse(out, rid, str(exc))

        def _send(self, request, out):
            import random  # noqa: PLC0415
            import urllib.error  # noqa: PLC0415
            import urllib.request  # noqa: PLC0415
            from email.utils import parsedate_to_datetime  # noqa: PLC0415
            out = Path(out)
            rid = out.name
            for attempt in range(9):
                if self.transport_hook is not None:  # 语言账本：每次传输尝试的输入在过闸与发送之前落盘
                    self.transport_hook(out, attempt)
                self._gate_before_send(request, out, rid)
                self.next_request_at = self._monotonic() + SEND_INTERVAL_S
                self.gate.mark(rid, "sent")
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        return json.loads(response.read()), response.headers.get("x-request-id")
                except urllib.error.HTTPError as error:
                    body = error.read().decode("utf-8", errors="replace")
                    try:
                        detail = json.loads(body).get("error", {})
                    except ValueError:
                        detail = {}
                    code = detail.get("code") if isinstance(detail, dict) else None
                    kind = detail.get("type") if isinstance(detail, dict) else None
                    headers = {k: v for k, v in error.headers.items()
                               if k.lower() == "retry-after" or k.lower() == "x-request-id"
                               or k.lower().startswith("x-ratelimit-")}
                    api_client.atomic_json(out / f"http_error_{attempt:02d}.json",
                                           json.loads(self.scrub(json.dumps({"status": error.code, "body": body,
                                                                             "headers": headers, "time": time.time()}))))
                    retryable = error.code == 429 and (code in ("rate_limit_exceeded", "slow_down")
                                                       or kind == "rate_limit_error")
                    if not retryable or attempt == 8:
                        raise RuntimeError(self.scrub(f"HTTP {error.code}: {body}")) from None
                    delay = min(120, 10 * 2 ** attempt)
                    retry_after = error.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            try:
                                delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                            except (TypeError, ValueError, OverflowError):
                                pass
                    self._sleep(delay + random.uniform(0, 3))

    _GUARDED_CLASSES[base] = GuardedResponsesClient
    return GuardedResponsesClient



# ── 两个服务端：费用守卫与 VLA ─────────────────────────────────────────────

class GuardProcess(ServerProcess):
    """费用守卫 ``servers/astra_cost_guard.py`` 的进程（不听端口）。

    ``port`` 只是元数据文件名里的标号（取 VLA 端口 + 1，``pick_port`` 已保证该号空闲）；就绪 = 守卫的心跳状态文件
    可读、schema 对、未退出、心跳新鲜（``CostGate.read_state`` 不抛 ``GuardRefused``；守卫已写 STOP 也算已起来）；
    停止时不查显存（守卫不占卡）。收 TERM 后守卫写 ``exited=true``，runner 立即拒发。"""

    def __init__(self, *a, state_path: str | Path, **k) -> None:
        super().__init__(*a, **k)
        self.state_path = Path(state_path)

    def _ready_now(self) -> bool:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return False
        g = guard_module()
        if state.get("schema") != g.STATE_SCHEMA or state.get("exited"):
            return False
        return abs(time.time() - float(state.get("heartbeat") or 0)) <= GUARD_HEARTBEAT_TIMEOUT_S

    def stop(self, *, grace_s: float = 60.0, check_gpu: bool = False) -> str:
        return super().stop(grace_s=grace_s, check_gpu=check_gpu)


class VLAServerProcess(ServerProcess):
    """Astra VLA 服务端（子模块 ``scripts/serve_policy.py``）：子进程环境里去掉 ``OPENAI_API_KEY``（VLA 不需要规划
    接口的密钥；与上游 ``run.sh`` 的 ``env -u OPENAI_API_KEY`` 同义），其余同 ``ServerProcess``。"""

    DROP_ENV = ("OPENAI_API_KEY",)

    def _child_env(self) -> dict:
        env = super()._child_env()
        for k in self.DROP_ENV:
            env.pop(k, None)
        return env


# ── 外层会话 → runner.episode 的 builder ────────────────────────────────────

class SessionEnv:
    """把外层 ``EnvSession`` 包成 ``runner.episode`` 眼里的环境：``reset``／``step`` 原样转给会话；``close()`` 置空
    （``runner.episode`` 的 ``finally`` 会关环境，但环境归外层关，模型侧不得调 ``session.close()``）。"""

    def __init__(self, session) -> None:
        self._session = session
        self.close_calls = 0

    def reset(self, *a, **k):
        return self._session.reset(*a, **k)

    def step(self, action):
        return self._session.step(action)

    def close(self) -> None:
        self.close_calls += 1  # 置空：只计次，不关

    def __getattr__(self, name):
        return getattr(self._session, name)


class SessionBuilder:
    """给 ``runner.episode`` 的 builder：``make_env_for_episode`` 返回外层会话（经 ``SessionEnv``，``close()`` 置空），
    只认本局局号、只交一次；``resolve_episode`` 委托真 builder（``runner.episode`` 写 ``identity.json`` 用）。"""

    def __init__(self, session, real_builder, episode: int) -> None:
        self._session = session
        self._builder = real_builder
        self._episode = int(episode)
        self.env: SessionEnv | None = None

    def make_env_for_episode(self, episode):
        if int(episode) != self._episode:
            raise ValueError(f"SessionBuilder 只交本局环境：要 {episode}，本局 {self._episode}")
        if self.env is not None:
            raise RuntimeError("SessionBuilder 一局只交一次环境")
        self.env = SessionEnv(self._session)
        return self.env

    def resolve_episode(self, episode):
        return self._builder.resolve_episode(episode)


def pin_visible_gpu(gpu) -> str:
    """把本进程钉到一张卡（监视模型 ``device_map={'':0}`` 即这张卡；仿真渲染同卡，与旧 run_astra.sh 的
    ``CUDA_VISIBLE_DEVICES=$MONITOR_GPU`` 同义）。CUDA 已初始化且可见卡不同时拒绝（改环境变量已不生效）。"""
    want = str(gpu)
    torch = sys.modules.get("torch")
    try:
        inited = bool(torch is not None and torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001
        inited = False
    if inited and os.environ.get("CUDA_VISIBLE_DEVICES") != want:
        raise RuntimeError(f"CUDA 已在本进程初始化（CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}），"
                           f"无法再把监视模型钉到第 {want} 张卡")
    os.environ["CUDA_VISIBLE_DEVICES"] = want
    return want


# ── AstraPolicy ─────────────────────────────────────────────────────────────

class AstraPolicy(Policy):
    """Astra 的模型侧四方法（见模块文档串）。

    ``cfg``（``scripts/evaluate.py`` 的模型参数，连字符换下划线后透传）：

    ========================  ============================================================================
    ``gpus``                  必给两张不同的卡：第一张起 VLA 服务端，第二张给本进程（监视模型与仿真）
    ``astra_ledger``          必给：费用账本 JSON（同目录放守卫状态、预留文件、``group_0/``；长期保留）
    ``astra_prices``          必给：单价配置 JSON（美元／百万 token，守卫 ``--prices``）
    ``astra_cap_usd``         费用上限，缺省 5；大于 5 直接拒绝（守卫 ``HARD_CAP_USD``）
    ``ckpt``                  必给：VLA 权重目录（``…/symbolic-grounded-subgoal/79999``）；别名 ``astra_vla_checkpoint``
    ``astra_monitor_adapter`` 必给：监视器 LoRA（``checkpoint-2246``）
    ``astra_monitor_base``    监视器底座（缺省上游 ``runner.BASE``）
    ``astra_vla_python``      VLA 服务端解释器（缺省 ``third_party/mme-vla/.venv/bin/python``）
    ``astra_root``            Astra 子模块根（缺省 ``third_party/Astra-on-RoboMME``）
    ``astra_group_dir``       停机口目录（缺省 ``<账本目录>/group_0``；名字须为 group_0／group_1）
    ``port_base``             VLA 端口基数（缺省 18762）
    ``max_planner_calls``     单局规划上限（缺省 24）
    ``astra_guard_interval``  守卫扫描间隔秒（缺省 2）
    ========================  ============================================================================
    """

    model = "astra"
    episode_wall_s = ASTRA_EPISODE_WALL_S
    #: 测试替换成假服务端类
    guard_server_cls = GuardProcess
    vla_server_cls = VLAServerProcess

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        self.guard: ServerProcess | None = None
        self.astra: SimpleNamespace | None = None
        self.gate: CostGate | None = None
        self.monitor = self.client = self.planner = self.responder = None
        self.run_dir: Path | None = None
        self.spool: Path | None = None
        self.port: int | None = None
        self._errors = 0
        self._stop: AstraStop | None = None
        self.episode_results: list[dict] = []

    # ── 配置 ──────────────────────────────────────────────────────────
    def _cfg_path(self, *names: str, required: bool = True) -> Path | None:
        for n in names:
            v = self.cfg.get(n)
            if v not in (None, ""):
                return Path(os.path.abspath(Path(str(v)).expanduser()))  # 不 resolve：venv 解释器是符号链接
        if required:
            flags = " 或 ".join("--" + n.replace("_", "-") for n in names)
            raise ValueError(f"RUN_BLOCKED reason=astra_cfg Astra 必须给 {flags}")
        return None

    def _parse_cfg(self) -> None:
        self.policy_seed = check_policy_seed(self.policy_seed)
        gpus = self.cfg.get("gpus")
        if isinstance(gpus, (int, str)):
            gpus = [int(x) for x in str(gpus).split(",") if str(x).strip()]
        gpus = [int(x) for x in (gpus or [])]
        if len(gpus) < 2 or gpus[0] == gpus[1]:
            raise ValueError(f"RUN_BLOCKED reason=astra_gpus Astra 需要两张不同的卡（--gpus <VLA 卡>,<监视卡>），得到 {gpus}")
        self.vla_gpu, self.monitor_gpu = gpus[0], gpus[1]
        self.ledger = self._cfg_path("astra_ledger")
        self.prices = self._cfg_path("astra_prices")
        g = guard_module()
        self.cap_usd = g.check_cap(float(self.cfg.get("astra_cap_usd", g.HARD_CAP_USD)))
        self.state_path = g.default_state_path(self.ledger)
        self.group_dir = self._cfg_path("astra_group_dir", required=False) or self.ledger.parent / "group_0"
        if self.group_dir.name not in GROUP_DIR_NAMES:
            raise ValueError(f"RUN_BLOCKED reason=layout astra_group_dir 须叫 {GROUP_DIR_NAMES} 之一：{self.group_dir}")
        self.vla_checkpoint = self._cfg_path("ckpt", "astra_vla_checkpoint")
        self.monitor_adapter = self._cfg_path("astra_monitor_adapter")
        self.monitor_base = self.cfg.get("astra_monitor_base")
        self.root = astra_root(self.cfg.get("astra_root"))
        self.vla_python = (self._cfg_path("astra_vla_python", required=False)
                           or third_party_root() / "mme-vla" / ".venv" / "bin" / "python")
        self.port_base = int(self.cfg.get("port_base") or DEFAULT_PORT_BASE)
        self.max_planner_calls = int(self.cfg.get("max_planner_calls") or DEFAULT_MAX_PLANNER_CALLS)
        self.guard_interval = float(self.cfg.get("astra_guard_interval") or guard_module().DEFAULT_INTERVAL_S)
        self.server_dir = self.ledger.parent / "servers"

    # ── 可替换的构造（测试注入替身） ────────────────────────────────
    def make_monitor(self, base, adapter):
        return self.astra.runner.Monitor(base, adapter)

    def make_client(self, port: int):
        from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy  # noqa: PLC0415
        import openpi_client  # noqa: PLC0415
        print(f"ASTRA_VLA_CLIENT openpi_client={openpi_client.__file__} port={port}", flush=True)
        return MMEVLAWebsocketClientPolicy("127.0.0.1", port)

    def make_responder(self, key: str, gate: "CostGate"):
        return guarded_client_class(self.astra.api_client)(key, gate)

    # ── 服务端命令 ────────────────────────────────────────────────────
    def guard_argv(self) -> list[str]:
        return [sys.executable, str(Path(guard_module().__file__).resolve()), "--root", str(self.group_dir),
                "--prices", str(self.prices), "--ledger", str(self.ledger), "--state", str(self.state_path),
                "--cap", f"{self.cap_usd:g}", "--interval", f"{self.guard_interval:g}"]

    def vla_argv(self, port: int) -> list[str]:
        """照抄上游 ``run.sh``，只把 ``--seed`` 由固定 42 换成 ``policy_seed``。"""
        return [str(self.vla_python), "scripts/serve_policy.py", f"--port={port}", f"--seed={self.policy_seed}",
                "policy:checkpoint", "--policy.config=mme_vla_suite", f"--policy.dir={self.vla_checkpoint}"]

    def vla_env(self) -> dict:
        """上游 ``run.sh`` 的环境变量；PYTHONPATH 第四段（环境源）换成 ``third_party/robomme_benchmark/src``。"""
        root = self.root
        return {"PYTHONPATH": os.pathsep.join([str(root / "examples" / "champ"), str(root / "src"),
                                               str(root / "packages" / "openpi-client" / "src"),
                                               str(benchmark_src())]),
                "OPENPI_DATA_HOME": os.environ.get("OPENPI_DATA_HOME", str(Path.home() / ".cache" / "openpi")),
                "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false", "USE_HF": "1", "IMAGE_MAX_TOKEN_NUM": "128",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def _vla_port(self) -> int:
        """有本基数的元数据（看门狗退出后留下的服务端）就沿用该端口，交 ``start()`` 去 attach；否则取空闲端口。"""
        if (self.server_dir / f"server-metadata-{self.port_base}.json").is_file():
            return self.port_base
        return pick_port(self.port_base)

    # ── 四个方法 ──────────────────────────────────────────────────────
    def load(self) -> None:
        self._parse_cfg()
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("RUN_BLOCKED reason=astra_key 环境变量 OPENAI_API_KEY 为空（密钥只从环境变量读）")
        self.astra = bootstrap(self.root)
        print("ASTRA_SOURCE root=" + str(self.root) + " " + " ".join(
            f"{k}={v}" for k, v in source_digests(self.astra.champ).items()), flush=True)
        assert_env_sources()
        if self.monitor_base is None:
            self.monitor_base = self.astra.runner.BASE
        self.astra.release_utils.validate_checkpoints(str(self.vla_checkpoint), str(self.monitor_adapter))
        stop = group_stop_file(self.group_dir)
        if stop.exists():  # 已停：账本保留，不新建账本绕额度
            print(f"ASTRA_STOP reason=host_stop stop={stop}（load 前已存在）", flush=True)
            raise AstraStop("host_stop", f"STOP.json 已存在：{stop}")
        self.group_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.group_dir / f"run-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        self.spool = self.run_dir / "planner_calls"
        self.run_dir.mkdir(parents=False, exist_ok=False)
        self.port = self._vla_port()
        # ① 费用守卫（先起：之后任何付费请求都要它的心跳与预留）
        self.guard = self.guard_server_cls(self.guard_argv(), cwd=str(REPO_ROOT), port=self.port + 1,
                                           metadata_dir=self.server_dir, name="astra-guard",
                                           log_path=self.server_dir / "astra-guard.log",
                                           ready_timeout_s=GUARD_READY_TIMEOUT_S, state_path=self.state_path)
        self.guard.start()
        self.gate = CostGate(self.state_path)
        state = self.gate.read_state()
        used = check_episode_cap(self.state_path)
        print(f"ASTRA_GUARD=PASS cap={self.gate.cap(state):g} committed={float(state.get('committed_usd') or 0):.4f} "
              f"episodes_used={used} ledger={self.ledger}", flush=True)
        # ② VLA 服务端（第一张卡）
        self.server = self.vla_server_cls(self.vla_argv(self.port), env=self.vla_env(), cwd=str(self.root),
                                          gpu=self.vla_gpu, ready=[Ready.port()], port=self.port,
                                          metadata_dir=self.server_dir, policy_seed=self.policy_seed,
                                          ckpt=str(self.vla_checkpoint), name="astra-vla",
                                          ready_timeout_s=VLA_READY_TIMEOUT_S)
        self.server.start()
        # ③ 本进程：第二张卡加载监视模型；VLA 客户端、规划客户端（同一实例维持 20 秒间隔）与 Planner
        pin_visible_gpu(self.monitor_gpu)
        self.monitor = self.make_monitor(self.monitor_base, str(self.monitor_adapter))
        self.client = self.make_client(self.port)
        self.responder = self.make_responder(key, self.gate)
        self.planner = self.astra.core.Planner(str(self.spool), responder=self.responder)
        print(f"ASTRA_LOAD=PASS policy_seed={self.policy_seed} server_seed={self.policy_seed} cloud_seed=null "
              f"vla_gpu={self.vla_gpu} monitor_gpu={self.monitor_gpu} port={self.port} run_dir={self.run_dir}",
              flush=True)

    def servers(self) -> list[ServerProcess]:
        return [s for s in (self.server, self.guard) if s is not None]

    def reset(self, spec) -> None:
        """局前拒绝点（不发服务端消息、不碰环境）：上一局留下的停机条件 → STOP.json → 两个服务端探活 →
        数据集与步数配对 → 跨进程登记局数（第 3 局即拒）。"""
        if self._stop is not None:
            raise self._stop
        stop = group_stop_file(self.group_dir)
        if stop.exists():
            print(f"ASTRA_STOP reason=host_stop key={spec.key}", flush=True)
            self._stop = AstraStop("host_stop", "Host requested stop before next episode")
            raise self._stop
        self.check_server()
        check_pairing(spec.dataset, spec.max_steps)
        self.gate.register_episode(f"{spec.dataset}:{spec.task}:{spec.episode}")

    def episode_args(self, spec) -> SimpleNamespace:
        """``runner.episode`` 的 ``args``：``output`` 指到本局 raw 目录下的 ``astra/``，``max_steps`` 随本局。"""
        return SimpleNamespace(output=str(Path(spec.out_dir) / ASTRA_SUBDIR), dataset=spec.dataset,
                               max_steps=int(spec.max_steps), max_planner_calls=self.max_planner_calls,
                               vla_checkpoint=str(self.vla_checkpoint), monitor_adapter=str(self.monitor_adapter),
                               monitor_base=self.monitor_base, policy_seed=self.policy_seed, port=self.port)

    def play(self, session, spec, recorder) -> dict:
        from robomme_ood_eval import episode as E  # noqa: PLC0415
        check_pairing(spec.dataset, spec.max_steps)
        args = self.episode_args(spec)
        ep_dir = Path(args.output) / spec.task / f"ep{int(spec.episode):03d}"
        if ep_dir.exists():
            raise RuntimeError(f"Astra 局目录已存在（保留上一尝试的证据，不覆盖）：{ep_dir}")
        identity = {k: v for k, v in spec.identity().items() if k not in ("task", "dataset", "attempt", "key")}
        builder = SessionBuilder(session, E.builder_for(spec.task, spec.dataset), spec.episode)
        try:
            result = run_one(args, spec.task, int(spec.episode), identity, builder, self.monitor, self.planner,
                             self.client, self.astra, attempt=spec.attempt, trace_dir=Path(spec.out_dir),
                             strict_cap=spec.strict_cap)
        except Exception as exc:  # 驱动异常：计一次非规划错误，照样原样上抛（外层记 error）
            self._after_episode({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, spec)
            raise
        self._after_episode(result, spec)
        status = terminal_of(result)
        out = {"status": status, "task_success": int(status == "success"), "steps": int(result.get("steps") or 0),
               "error": result.get("error"), "infra": False, "infra_reason": None,
               "decisions": int(result.get("planner_calls") or 0),
               "planner_calls": int(result.get("planner_calls") or 0),
               "monitor_calls": int(result.get("monitor_calls") or 0),
               "review_calls": int(result.get("review_calls") or 0),
               "astra_dir": os.path.relpath(ep_dir, spec.out_dir), "cloud_seed": None,
               "timing": {"astra_seconds": result.get("seconds")}}
        if builder.env is not None:
            out["astra_env_close_calls"] = builder.env.close_calls
        return out

    def _after_episode(self, result: dict, spec) -> None:
        """上游 ``main()`` 第④⑤项与局数上限：play 返回后判定，记下停机条件，下一局 ``reset`` 抛出。"""
        self.episode_results.append(result)
        failed = result.get("status") == "error" and not self.astra.core.is_planner_failure(result)
        self._errors = self._errors + 1 if failed else 0
        if result.get("status") == "error" and str(result.get("error") or "").startswith(PLANNER_STOP_PREFIXES):
            print(f"ASTRA_STOP reason=planner_error key={spec.key}", flush=True)
            self._stop = AstraStop("planner_error", "Stopping pilot after planner service or budget error")
        elif self._errors >= INFRA_ERROR_LIMIT:
            print(f"ASTRA_STOP reason=infra_errors key={spec.key} errors={self._errors}", flush=True)
            self._stop = AstraStop("infra_errors", "Three consecutive infrastructure/protocol errors; stopping shard")
        else:
            try:
                check_episode_cap(self.state_path)
            except AstraStop as exc:
                self._stop = exc

    def close(self) -> None:
        """先停 VLA 服务端进程组、再停守卫（守卫写 exited=true）；释放监视模型。幂等。"""
        if self.closed:
            return
        self.closed = True
        for srv in self.servers():
            try:
                srv.stop()
            except Exception as exc:  # noqa: BLE001 一个停不掉不妨碍停另一个
                print(f"ASTRA_CLOSE_WARN name={getattr(srv, 'name', '?')} error={type(exc).__name__}: {exc}", flush=True)
        self.monitor = self.client = self.planner = self.responder = None


__all__ = ["AstraPolicy", "AstraStop", "StepCapReached", "SessionBuilder", "SessionEnv", "GuardProcess",
           "VLAServerProcess", "CostGate", "GuardRefused", "guarded_client_class", "run_one", "check_pairing",
           "check_policy_seed", "bootstrap", "assert_env_sources"]
