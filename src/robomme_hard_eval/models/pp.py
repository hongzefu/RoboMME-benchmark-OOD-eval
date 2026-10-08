#!/usr/bin/env python3
"""PonderPounce 新侧客户端（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.4、1.7；子任务 S4）。

``run_episode(session, identity, conn_info, recorder) -> dict`` 由 ``env_client.SeatRunner.run_one`` 调用
（``--policy pp`` 时按 ``load_sibling("pp_client")`` 加载）。环境侧是本仓库 ``EnvSession``（``robomme_hard``），
模型侧是 vla-eval 0.7.0 协议的 PonderPounce 服务（``python -m ponderpounce.eval.robomme_server``）。

逐项照 vla-eval 0.7.0 源码复刻（``runners/sync_runner.py::SyncEpisodeRunner.run_episode``、
``benchmarks/robomme/benchmark.py::RoboMMEBenchmark``、``benchmarks/base.py::StepBenchmark``）：

1. 连接：``vla_eval.connection.Connection(url, timeout=300.0)``，``connect(benchmark=PP_BENCHMARK)`` 完成 HELLO 握手
   （PonderPounce 官方 ``configs/robomme.yaml`` 的 ``server.timeout`` 即 300.0）。
2. ``session.reset()`` → 首条观测（``make_obs`` 同式：``images.agentview``／``images.wrist``、``task_description``、
   ``states``（7 关节 float64 + 夹爪第 1 维，拼接后转 float32），首条另带 ``video_history = front_rgb_list[:-1]``、
   ``wrist_video_history = wrist_rgb_list[:-1]``、``episode_restart=True``）。
3. ``EPISODE_START``（服务端成功时不回复）：``{"task": {"name","env_id","episode_idx"},
   "recording": {"sid","eid","eval_id": "","db_path": ""}}``；``sid`` 固定（见 ``fixed_sid``），``eid`` 与 ``sid`` 相同。
4. ``for step in range(max_steps)``：``act(obs)`` → 动作 ``actions`` 展平为 Python float 列表、取前 8 维交给
   ``session.step`` → ``terminated or truncated or info["status"] == "error"`` 即 ``break``（最后一帧不再发送）
   → 否则按 ``make_obs`` 打包下一帧。上限恰好 ``max_steps`` 个动作（xhard0 为 1300；V9 由 session 的 strict cap 管，
   循环上限与之相等）。
5. 正常走完循环才发 ``EPISODE_END``：``{"metrics": {"success": status=="success"}, "steps": step+1,
   "elapsed_sec": ...}``；中途异常不发（与 SyncEpisodeRunner 异常上抛时一致），连接一律关闭。

固定 sid（计划 1.4「固定 sid」行）：xhard0 为 ``<task>|<source_episode>|<seed>``，V9 为 ``<task>|<tier>|<seed>``。
PonderPounce 的噪声种子是 ``crc32(f"{seed}:{sid}:{n}")``，两侧各起自己的服务进程、同一 sid 在一个进程内只用一次，
保证 ``n=0``。基础设施重试由席位脚本「重启服务再重发同一 sid」完成；本驱动内的 ``reconnect`` 只用于同一局内
``ConnectionClosed`` 断线（至多 ``conn_info["pp_max_reconnects"]`` 次，缺省 1），**不重发 EPISODE_START**——
注意 vla-eval 服务端对新连接分配新的会话 id，PonderPounce 收到无 EPISODE_START 的观测会回 ERROR，
本局随之记 ``error``（``infra=True``），由席位脚本按基础设施故障处理。

轨迹（计划 1.7）：每局一个 ``trace.jsonl``（``trace_writer.TraceWriter``，route ``pp-new``）。``request`` 行为每个
局内协议帧（EPISODE_START／OBSERVATION／EPISODE_END，不含连接级 HELLO）的规范化字节 sha256（``canonical_frame_bytes``：
键排序的 msgpack、数组按原始 dtype/shape 的原始字节、EPISODE_END 去掉墙钟字段 ``elapsed_sec``）；``response`` 行为
服务返回的完整动作块；``step`` 行为执行后的画面、状态、交给环境的 8 维动作与终态。原侧 ``pp_official_runner.py``
用本模块同一批辅助函数写轨迹，两侧字段口径一致。

``conn_info`` 读取的键：``host``（缺省 127.0.0.1）、``port``（必需）、``max_steps``（必需）、``dataset``（可选，
``hard-verify`` 时要求 ``tier == "xhard0"``）、``trace_path``／``trace_dir``（可选；``trace_dir`` 为每局目录，
见 ``resolve_trace_path``）、
``pp_max_reconnects``（可选，缺省 1）、``pp_phase2``（可选，见下）、``attempt``／``episode_tag``（可选，第二阶段身份用）。

第二阶段新侧（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S5」、〇节 R2 与 C1～C11）：
新行为一律由开关打开，默认（开关关）写出的协议帧与 ``trace.jsonl`` 与 ``BASE`` 逐字节相同。开关取
``conn_info["pp_phase2"]``（显式给出时以它为准），否则环境变量 ``SGEVAL_PP_SERVER_WRAP=1``（席位脚本以外壳
``pp_server_wrap.py`` 起服务时同时置位）。打开后：

- 服务端外壳在每个 ACTION 回包里带 ``"subgoal"``（模型自己的子目标文本，等待中为 ``None``）；
  ``TracedConnection.act`` 存进 ``last_subgoal``；回包缺这个键即判外壳未生效，本局记 ``error``
  （``infra_reason=pp_subgoal_missing``），不拿环境标准答案冒充。
- 逐步 ``subgoal`` 写模型子目标经 ``pp_subgoal_to_official`` 换算坐标后的文本（原文另记 ``subgoal_raw``）；
  环境标准答案 ``info["simple_subgoal_online"]`` 改记同步号的 ``history`` 行，``note=oracle_simple_subgoal:<文本>``。
- C1 route ``pp/new``；C2 演示段记全部 reset 帧（含初始帧），``end.demo_frames`` 为不含初始帧的演示帧数；
  C3 ``terminal_reason`` 与 ``status`` 同取 ``success``／``fail``／``timeout``／``error``（旧口径的退出原因另记
  ``exit_reason``），reset 之前就失败的局记 ``no_frame=true``；C4 交给环境的 float64 动作原值与观测步状态由
  ``TraceWriter`` 收集，收尾经 ``trace_writer.merge_write_npz`` 写同目录 ``arrays.npz``（``exec_action__%05d``／
  ``exec_state__%05d``；与同目录录像器的同名键同值合并，先后任意都不覆盖；第三阶段起本模块不再直接 ``np.savez``）；
  C6 identity 加 ``attempt``；C8 ``end`` 写 ``steps_attempted``／``steps_observed``／``frames_recorded``，
  环境异常步与空观测步用 ``log_missing_step`` 保留步号与动作。

第三阶段语言账本（冻结说明第五节；只在第二阶段开关打开且有轨迹落点时记，轨迹同目录 ``language.jsonl``）：
每个观测一个 ``action_model`` 调用（发送前写 ``in``：``role=fields`` 的 ``task_description`` + 当前前视／腕部帧引用；
回包后 pop 掉 ``_sgeval_audit``，逐通道写分词消息，``close_call`` 记 ``server_final_text``）；审计块带
``pp_generation`` 时另开 ``subgoal_model`` 调用记 S2 完整生成块（见 ``TracedConnection._log_generation``；块带外壳
解码的 S2 输入 ``input_text`` 时先记 ``in`` 消息与附图引用，回包后写，属第五节同类例外）。执行步
``source_call_id`` 指向本步动作来自的 ``action_model`` 调用、``chunk_index=0``。审计键在动作交给环境之前一律 pop 掉，
不改动作、RNG 与请求字节。

原侧 ``pp_official_runner.py``（R1 零改动）只调用本模块的默认路径：``TRACE_SCHEMA_ROUTE_ORIG`` 仍为 ``pp-orig``、
``trace_reset``／``trace_step`` 不传新参数，原侧序列化输出与 ``BASE`` 逐字节相同。

导入期只依赖标准库与 numpy；``vla_eval``（客户端扩展环境 client-env 内）、``anyio``、``msgpack``、``websockets``
都在用到时才导入。单测通过 ``run_episode(..., connection_factory=...)`` 注入替身连接。
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

#: HELLO 握手里带的 benchmark 名，与 PonderPounce 官方 configs/robomme.yaml 的 ``benchmark`` 相同
PP_BENCHMARK = "vla_eval.benchmarks.robomme.benchmark:RoboMMEBenchmark"
#: 每次 recv 的超时（秒），与官方 configs/robomme.yaml 的 ``server.timeout`` 相同
PP_TIMEOUT_S = 300.0
#: 交给环境的动作维数（7 关节 + 1 夹爪）
PP_ACTION_DIMS = 8
#: 同一局内 ConnectionClosed 后的重连次数上限（缺省）
PP_MAX_RECONNECTS = 1
XHARD0 = "xhard0"
HARD_VERIFY = "hard-verify"
TRACE_SCHEMA_ROUTE_NEW = "pp-new"
#: 原侧 route；``pp_official_runner.py``（R1 零改动）引用本常量，保持旧值使原侧输出与 BASE 逐字节相同
TRACE_SCHEMA_ROUTE_ORIG = "pp-orig"
#: 第二阶段新侧 route（C1）
TRACE_ROUTE_NEW_C1 = "pp/new"
#: 第二阶段开关的环境变量（与席位脚本起外壳服务的开关同名）
PHASE2_ENV = "SGEVAL_PP_SERVER_WRAP"
#: 服务端外壳回包里子目标的键（与 pp_server_wrap.SUBGOAL_KEY 相同）
SUBGOAL_KEY = "subgoal"
#: 服务外壳回包审计键（冻结说明第五节；与 trace_writer.AUDIT_KEY 相同），交给环境前 pop 掉
AUDIT_KEY = "_sgeval_audit"
#: 环境标准答案在 history 行 note 里的前缀
ORACLE_NOTE_PREFIX = "oracle_simple_subgoal:"
#: 协议帧类型（与 vla_eval.protocol.messages.MessageType 的取值相同）
HELLO, OBSERVATION, ACTION, EPISODE_START, EPISODE_END, ERROR = (
    "hello", "observation", "action", "episode_start", "episode_end", "error")


def load_trace_writer():
    """评估包的 ``record.trace_writer`` 模块（包内 import，已导入则复用；单测可整体替换本函数）。"""
    from robomme_hard_eval.record import trace_writer

    return trace_writer


#: 「未提供」哨兵（R2／C7）：即 ``trace_writer.UNSET``，与 ``None``（模型等待中）区分
UNSET = load_trace_writer().UNSET


def _is_unset(x: Any) -> bool:
    """``UNSET`` 判定（按类名，兼容测试里按路径另载的 trace_writer 副本）。"""
    return x is UNSET or type(x).__name__ == "_Unset"


# ── 第二阶段：开关、子目标坐标换算 ─────────────────────────────────────────────


def phase2_enabled(conn_info: dict) -> bool:
    """第二阶段新侧开关：``conn_info["pp_phase2"]`` 显式给出时以它为准，否则看环境变量 ``SGEVAL_PP_SERVER_WRAP``。"""
    if conn_info.get("pp_phase2") is not None:
        return bool(conn_info["pp_phase2"])
    return os.environ.get(PHASE2_ENV, "") == "1"


_AT_POINT = re.compile(r"\bat\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def _to_255(v: str | float) -> int:
    """0～1000 → 0～255：``v*255/1000`` 四舍五入（0.5 进位，不用银行家舍入），再夹到 0～255。"""
    return max(0, min(255, int(math.floor(float(v) * 255.0 / 1000.0 + 0.5))))


def pp_subgoal_to_official(text: str | None) -> str | None:
    """PonderPounce 子目标里的每组 ``at [x, y]``（0～1000，先 x 后 y）换成官方版式 ``at <y', x'>``（0～255，先行后列）。

    ``y' = round(y*255/1000)``、``x' = round(x*255/1000)``，夹到 0～255；其余文字原样保留；``None`` 原样返回
    （C7 等待中）。例：``"pick up the cube at [612, 247]"`` → ``"pick up the cube at <63, 156>"``。"""
    if text is None:
        return None
    return _AT_POINT.sub(lambda m: f"at <{_to_255(m.group(2))}, {_to_255(m.group(1))}>", str(text))


def attempt_of(identity: dict, conn_info: dict, trace_path: Path | None) -> int | None:
    """C6 尝试号：``conn_info["attempt"]`` → ``episode_tag``（``<key>.a<N>``）→ 轨迹所在目录名 → ``identity["attempt"]``。"""
    if conn_info.get("attempt") is not None:
        return int(conn_info["attempt"])
    for name in (conn_info.get("episode_tag"), trace_path.parent.name if trace_path is not None else None):
        m = re.match(r"^.+\.a(\d+)$", str(name or ""))
        if m:
            return int(m.group(1))
    att = identity.get("attempt")
    return int(att) if att is not None else None


class SubgoalMissing(RuntimeError):
    """第二阶段开着，但 ACTION 回包没有 ``subgoal`` 键（服务端外壳未生效）。"""


# ── 身份与协议载荷 ──────────────────────────────────────────────────────────


def fixed_sid(identity: dict, dataset: str | None = None) -> str:
    """固定 sid：xhard0 ``<task>|<source_episode>|<seed>``；V9 ``<task>|<tier>|<seed>``。

    ``dataset == "hard-verify"`` 时身份必须是 xhard0，否则 ``ValueError``（防止 V9 身份混进 hard0 分片）。"""
    task, tier, seed = identity["task"], identity.get("tier"), identity["seed"]
    if dataset == HARD_VERIFY and tier != XHARD0:
        raise ValueError(f"dataset={dataset} 但身份 tier={tier!r}，不是 {XHARD0}")
    if tier == XHARD0:
        src = identity["source_episode"]
        if not isinstance(src, int) or isinstance(src, bool):
            raise ValueError(f"xhard0 身份的 source_episode 必须是整数：{src!r}")
        return f"{task}|{int(src)}|{int(seed)}"
    if not tier:
        raise ValueError(f"身份缺 tier：{identity!r}")
    return f"{task}|{tier}|{int(seed)}"


def episode_idx_of(identity: dict) -> int:
    """EPISODE_START 的 ``task.episode_idx``：取官方局号 ``source_episode``（两侧一致）；没有时退回 ``builder_episode``。"""
    src = identity.get("source_episode")
    if isinstance(src, int) and not isinstance(src, bool):
        return int(src)
    return int(identity["builder_episode"])


def episode_start_payload(task: str, episode_idx: int, sid: str, eid: str) -> dict:
    """与 ``SyncEpisodeRunner`` 在「记录器 is_active」时发出的 EPISODE_START 载荷同构。"""
    return {"task": {"name": task, "env_id": task, "episode_idx": int(episode_idx)},
            "recording": {"sid": sid, "eid": eid, "eval_id": "", "db_path": ""}}


def episode_end_payload(success: bool, steps: int, elapsed_sec: float) -> dict:
    return {"metrics": {"success": bool(success)}, "steps": int(steps), "elapsed_sec": round(float(elapsed_sec), 3)}


def task_description_of(info: dict) -> str:
    """``RoboMMEBenchmark.reset`` 同式：``info["task_goal"]`` 为列表取第 0 个，否则 ``str``。"""
    goal = info["task_goal"]
    return goal[0] if isinstance(goal, list) else str(goal)


def state8(joint: Any, gripper: Any) -> np.ndarray:
    """``make_obs`` 同式的 8 维状态：float64 的 7 关节 + 夹爪第 1 维，拼接后转 float32。"""
    j = np.asarray(joint, dtype=np.float64)
    g = np.asarray(gripper, dtype=np.float64)[:1]
    return np.concatenate([j, g]).astype(np.float32)


class ObsPacker:
    """逐局一个，复刻 ``RoboMMEBenchmark.make_obs``（send_wrist_image／send_state／send_video_history 均为缺省 True，
    send_subgoal 为缺省 False）。演示视频只随第一条非空观测发送一次，之后清空。"""

    def __init__(self) -> None:
        self.task_description = ""
        self._video: list = []
        self._wrist_video: list = []

    def on_reset(self, raw_obs: dict, info: dict) -> None:
        self._video = list(raw_obs["front_rgb_list"][:-1])
        self._wrist_video = list(raw_obs.get("wrist_rgb_list", [])[:-1])
        self.task_description = task_description_of(info)

    @property
    def demo_frames(self) -> int:
        return len(self._video)

    def make(self, raw_obs: Any) -> dict:
        if not raw_obs:
            return {"images": {}, "task_description": self.task_description}
        front_list = raw_obs.get("front_rgb_list", [])
        if not front_list:
            return {"images": {}, "task_description": self.task_description}
        obs: dict[str, Any] = {"images": {"agentview": front_list[-1]}, "task_description": self.task_description}
        wrist_list = raw_obs.get("wrist_rgb_list")
        if wrist_list:
            obs["images"]["wrist"] = wrist_list[-1]
        obs["states"] = state8(raw_obs["joint_state_list"][-1], raw_obs["gripper_state_list"][-1])
        if self._video:
            obs["video_history"] = list(self._video)
            if self._wrist_video:
                obs["wrist_video_history"] = list(self._wrist_video)
            obs["episode_restart"] = True
            self._video = []
            self._wrist_video = []
        return obs


def exec_action8(action_payload: dict) -> list[float]:
    """``RoboMMEBenchmark.step`` 同式取 ``actions``（缺则 ``action``）并展平为 Python float 列表，再取前 8 维
    （计划 1.4：``actions[0][:8]``；``(1, D)`` 展平后的前 8 个即第 0 行前 8 维）。"""
    raw = action_payload.get("actions", action_payload.get("action"))
    if raw is None:
        raise ValueError("Action dict must contain 'actions' or 'action' key")
    if hasattr(raw, "flatten"):
        flat = raw.flatten().tolist()
    elif not isinstance(raw, list):
        flat = list(raw)
    else:
        flat = raw
    return [float(x) for x in flat[:PP_ACTION_DIMS]]


def step_done(terminated: Any, truncated: Any, info: dict) -> bool:
    """``RoboMMEBenchmark.step`` 的 done 判据。"""
    return bool(terminated) or bool(truncated) or (isinstance(info, dict) and info.get("status") == "error")


def terminal_status(done: bool, truncated: bool, info: dict | None) -> tuple[str, str | None]:
    """局末状态映射：``success``／``fail``／``timeout`` 原样；环境 ``error`` 记 error；被截断且无明确状态记 timeout；
    循环跑满 ``max_steps`` 仍未结束记 timeout；其余（``ongoing``／``unknown`` 等）记 error + ``success_flag=<值>``。"""
    st = (info or {}).get("status") if isinstance(info, dict) else None
    if not done:
        return "timeout", None
    if st in ("success", "fail", "timeout"):
        return st, None
    if st == "error":
        return "error", "env_status=error"
    if truncated:
        return "timeout", None
    return "error", f"success_flag={st}"


# ── 规范化字节与轨迹 ──────────────────────────────────────────────────────


def _normalize(obj: Any) -> Any:
    """键排序、数组转「dtype/shape/原始字节」字典、numpy 标量转 Python 标量（不经 PNG，保证确定）。"""
    if isinstance(obj, dict):
        return {str(k): _normalize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {"__ndarray__": True, "dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def canonical_frame_bytes(msg_type: str, payload: Any) -> bytes:
    """一个协议帧的规范化字节（轨迹 request 行的哈希来源）。EPISODE_END 去掉墙钟 ``elapsed_sec``；
    帧头的 ``seq``／``timestamp`` 不计入。"""
    import msgpack

    if msg_type == EPISODE_END and isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if k != "elapsed_sec"}
    return msgpack.packb({"type": str(msg_type), "payload": _normalize(payload)}, use_bin_type=True)


class NullTrace:
    """不写轨迹时的空实现（接口同 trace_writer.TraceWriter）。"""

    exec_steps = 0

    def log_demo(self, *a, **k): pass
    def log_request(self, *a, **k): pass
    def log_response(self, *a, **k): pass
    def log_history(self, *a, **k): pass
    def log_step(self, *a, **k): pass
    def log_missing_step(self, *a, **k): pass
    def close(self, *a, **k): pass


def trace_reset(trace, raw_obs: dict, task_description: str, *, include_initial: bool = False) -> int:
    """演示行：``video_history``（= ``front_rgb_list[:-1]``）逐帧画面哈希、腕部同位帧、逐帧 8 维状态与指令文本。
    返回演示帧数（不含初始帧）。

    ``include_initial=True``（第二阶段新侧，C2）：演示行记 reset 返回的全部帧（含最后一帧初始画面），
    ``frames == len(states) ==`` 返回值 ``+ 1``。缺省 False 与 BASE 逐字节相同（原侧不传）。"""
    if include_initial:
        fronts = list(raw_obs["front_rgb_list"])
        wrists = list(raw_obs.get("wrist_rgb_list", []))
        joints = list(raw_obs.get("joint_state_list", []) or [])
        grips = list(raw_obs.get("gripper_state_list", []) or [])
        states = [state8(j, g) for j, g in zip(joints, grips)]
        trace.log_demo(fronts, wrists, states=states, texts=[task_description])
        return max(0, len(fronts) - 1)
    fronts = list(raw_obs["front_rgb_list"][:-1])
    wrists = list(raw_obs.get("wrist_rgb_list", [])[:-1])
    joints = list(raw_obs.get("joint_state_list", []) or [])
    grips = list(raw_obs.get("gripper_state_list", []) or [])
    states = [state8(j, g) for j, g in zip(joints[:-1], grips[:-1])]
    trace.log_demo(fronts, wrists, states=states, texts=[task_description])
    return len(fronts)


def trace_step(trace, step_no: int, out: tuple, action8: list[float], subgoal: Any = UNSET,
               link: dict | None = None) -> bool:
    """执行完第 ``step_no`` 步（从 1 计）的一行；``out`` 为环境 ``step`` 的五元组。返回本步是否有完整观测。

    ``subgoal`` 缺省（``UNSET``）：逐步 ``subgoal`` 记环境标准答案 ``info["simple_subgoal_online"]``，写出与 BASE
    逐字节相同（原侧只走这条路）。显式传入模型子目标（文本，或 ``None``＝模型等待中）：``subgoal`` 记
    ``pp_subgoal_to_official`` 换算后的文本、``subgoal_raw`` 记原文；标准答案改记同步号 ``history`` 行
    （``note=oracle_simple_subgoal:<文本>``，标准答案为空的步不记）；没有完整观测的步改用 ``log_missing_step``（C8）。"""
    obs, _reward, terminated, truncated, info = out
    info = info if isinstance(info, dict) else {}
    front = wrist = state = None
    if isinstance(obs, dict) and obs.get("front_rgb_list"):
        front = obs["front_rgb_list"][-1]
        if obs.get("wrist_rgb_list"):
            wrist = obs["wrist_rgb_list"][-1]
        if obs.get("joint_state_list") and obs.get("gripper_state_list"):
            state = state8(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
    oracle = info.get("simple_subgoal_online")
    observed = front is not None and wrist is not None and state is not None
    if _is_unset(subgoal):
        trace.log_step(step=step_no, front=front, wrist=wrist, state=state,
                       action=np.asarray(action8, dtype=np.float64), subgoal=None if oracle is None else str(oracle),
                       terminated=bool(terminated), truncated=bool(truncated), status=info.get("status"))
        return observed
    raw = None if subgoal is None else str(subgoal)
    official = pp_subgoal_to_official(raw)
    action = np.asarray(action8, dtype=np.float64)
    link = {k: v for k, v in (link or {}).items() if v is not None}  # 第三阶段 source_call_id／chunk_index
    if observed:
        trace.log_step(step=step_no, front=front, wrist=wrist, state=state, action=action, subgoal=official,
                       terminated=bool(terminated), truncated=bool(truncated), status=info.get("status"),
                       subgoal_raw=raw, **link)
    else:
        trace.log_missing_step(step=step_no, action=action, reason="obs_none" if obs is None else "obs_incomplete",
                               subgoal=official, subgoal_raw=raw, env_status=info.get("status"), **link)
    if oracle is not None:
        trace.log_history(step_no, step_no, note=f"{ORACLE_NOTE_PREFIX}{oracle}")
    return observed


class TracedConnection:
    """包住 vla-eval ``Connection``：局内每个协议帧写轨迹 request／response 行并计数；其余属性透传。

    ``SyncEpisodeRunner`` 只调用 ``start_episode``／``act``／``end_episode``，原侧把本包装直接交给它。

    ``last_subgoal``：最近一次 ACTION 回包里 ``"subgoal"`` 的值（服务端外壳 ``pp_server_wrap.py`` 才有；文本或
    ``None``＝模型等待中）；回包没有这个键时为 ``UNSET``。只读记录，不影响轨迹与返回值。"""

    def __init__(self, conn: Any, trace=None, lang=None) -> None:
        self._conn = conn
        self.trace = trace if trace is not None else NullTrace()
        self.frames_sent = 0
        self.actions_received = 0
        self.last_subgoal: Any = UNSET
        # 第三阶段语言账本（冻结说明第五节；None 时不记，原侧与第二阶段开关关时都不传）
        self.lang = lang
        self.lang_errors = 0
        self.last_call_id: str | None = None  # 最近一次成功回包的 action_model 调用（执行步的 source_call_id）
        self.last_audit: Any = None           # 最近一次回包里 pop 出的 ``_sgeval_audit``
        self.demo_index: int | None = None    # 演示段末帧（初始画面）在 trace demo 行里的下标
        self._failed_step: int | None = None
        self._transport_attempt = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def start_episode(self, config: dict) -> None:
        self.trace.log_request(EPISODE_START, canonical_frame_bytes(EPISODE_START, config), step=0)
        self.frames_sent += 1
        await self._conn.start_episode(config)

    # -- 语言账本辅助：一律吞异常、计 lang_errors，不改请求、动作与控制流 --
    def _lang(self, fn: str, *a, **k) -> Any:
        if self.lang is None:
            return None
        try:
            return getattr(self.lang, fn)(*a, **k)
        except Exception as e:  # noqa: BLE001
            self.lang_errors += 1
            print(f"TRACE_HOOK_ERROR route=pp/new where=language.{fn} {type(e).__name__}: {e}"[:600], flush=True)
            return None

    def _image_refs(self, obs: dict, step: int) -> list | None:
        imgs = obs.get("images") if isinstance(obs, dict) else None
        if not imgs:
            return None
        h = load_trace_writer().image_sha256
        phase, idx = ("demo", self.demo_index) if step == 0 else ("exec", step)
        refs = []
        for slot, (key, ref, cam) in enumerate((("agentview", "current", "front"), ("wrist", "wrist", "wrist"))):
            if key in imgs:
                refs.append({"slot": slot, "ref": ref, "phase": phase, "frame_idx": idx, "cam": cam,
                             "raw_sha256": h(imgs[key]), "sources": None, "transform": None, "encoded_sha256": None})
        return refs or None

    def _log_generation(self, step: int, gen: Any, obs: Any = None) -> None:
        """S2 完整生成块记成一个 ``subgoal_model`` 调用（服务端内部推理，事后记账）。

        读取的键（R3 外壳写入，缺键记 None）：``context``／``prompt``（S2 输入全文，含回灌历史）、``images``（附图引用）、
        ``text``（完整生成块原文）、``reasoning``、``subgoal_raw``（``at [x, y]`` 原文）、``kind``（``transition``／
        ``nontransition``）、``committed``（提交或回滚）。``parsed`` 记换算后子目标与整块原样；nontransition 显式记
        ``text_output=False``、``out`` 文字为 None。

        FIX-3：块带 ``input_text``（外壳从 S2 上下文 token 解码的输入文字）时，先写 ``dir=in role=user`` 消息（附图按
        ``input_images`` 解析成帧引用，见 ``_s2_input_images``），``parsed`` 另记 ``input_decoded_from_tokens=True`` 与
        ``input_image_check``，``parsed.generation`` 去掉 ``input_text``（不重复存）；没有该键时与之前逐字节相同。"""
        if not isinstance(gen, dict):
            return
        cid = self._lang("open_call", "subgoal_model", step, params=gen.get("params"))
        if cid is None:
            return
        decoded = isinstance(gen.get("input_text"), str)
        check = None
        if decoded:
            # FIX-3：S2 输入是服务外壳从上下文 token 解码出的文字（本次 fire 前新增的段：首次含任务前缀与演示图，之后含
            # 回灌的上一子目标与 cognition 占位、本次观测图）。它来自回包，只能在回包后写——与冻结说明第五节已登记的
            # 「带 channel 的分词通道消息」同类的「发送前落盘」例外；仍记 dir=in（描述的是进模型的内容）。
            try:
                imgs, demo_video, check = self._s2_input_images(gen.get("input_images"), obs, step)
            except Exception as e:  # noqa: BLE001 账本辅助一律不影响控制流
                self.lang_errors += 1
                print(f"TRACE_HOOK_ERROR route=pp/new where=language.s2_images {type(e).__name__}: {e}"[:600],
                      flush=True)
                imgs, demo_video, check = None, None, None
            self._lang("message", cid, dir="in", role="user", text=gen["input_text"], images=imgs,
                       demo_video=demo_video)
        else:
            ctx = gen.get("context", gen.get("prompt"))
            if ctx is not None or gen.get("images"):
                self._lang("message", cid, dir="in", role="user", text=ctx, images=gen.get("images"))
        kind = gen.get("kind", gen.get("transition"))
        nontransition = kind in ("nontransition", False)
        raw = gen.get("subgoal_raw")
        text = None if nontransition else gen.get("text", raw)
        self._lang("message", cid, dir="out", role="assistant", text=text)
        parsed = {"subgoal": pp_subgoal_to_official(None if raw is None else str(raw)), "subgoal_raw": raw,
                  "reasoning": gen.get("reasoning"), "kind": kind, "committed": gen.get("committed"),
                  "text_output": not nontransition and text is not None, "generation": gen}
        if decoded:  # 行 schema 冻结：标志与图核对结果落在 parsed；输入原文已在 in 消息里，generation 不再重复
            parsed["generation"] = {k: v for k, v in gen.items() if k != "input_text"}
            parsed["input_decoded_from_tokens"] = True
            parsed["input_image_check"] = check
        self._lang("close_call", cid, status="reply", parsed=parsed)

    def _s2_input_images(self, descs: Any, obs: Any, step: int) -> tuple[list | None, str | None, dict]:
        """S2 输入第 k 个 ``<image:k>`` → 冻结说明的 images 元素（slot=k）。

        - ``source=obs``：本次观测（S2 fire 发生在处理这条观测时，生成块随它的回包返回），帧号与 ``_image_refs`` 同口径
          （step 0 为 demo 段末帧 ``demo_index``，k≥1 为执行第 k 步后），``raw_sha256`` 用客户端对发出原始帧的哈希
          （即 trace 帧哈希），与外壳的 ``pixel_sha256`` 核对；
        - ``source=demo``：S2 前缀里的演示图（服务端按 demo_fps 抽帧）；在本次观测的 ``video_history``（= trace demo
          行前 N 帧）里按像素哈希顺序匹配出帧号，``ref=keyframe``、``cam=front``；匹配不上时帧号 None、``raw_sha256``
          取外壳哈希（检查器会如实报 unresolved）。
        返回 ``(images, demo_video, check)``；check = ``{"n", "sha_mismatch", "demo_unmatched"}``。"""
        check = {"n": 0, "sha_mismatch": 0, "demo_unmatched": 0}
        if not isinstance(descs, list) or not descs:
            return None, None, check
        h = load_trace_writer().image_sha256
        imgs = obs.get("images") if isinstance(obs, dict) else None
        imgs = imgs if isinstance(imgs, dict) else {}
        phase, idx = ("demo", self.demo_index) if step == 0 else ("exec", step)
        vh = obs.get("video_history") if isinstance(obs, dict) else None
        vh_hashes = [h(f) for f in vh] if isinstance(vh, list) and any(
            isinstance(d, dict) and d.get("source") == "demo" for d in descs) else []
        cams = {"agentview": ("current", "front"), "wrist": ("wrist", "wrist")}
        out, demo_idx, nxt = [], [], 0
        for k, d in enumerate(descs):
            d = d if isinstance(d, dict) else {}
            px = d.get("pixel_sha256")
            check["n"] += 1
            if d.get("source") == "obs":
                ref, cam = cams.get(str(d.get("cam_key")), ("current", None))
                raw = h(imgs[d["cam_key"]]) if d.get("cam_key") in imgs else px
                check["sha_mismatch"] += int(px is not None and raw != px)
                out.append({"slot": k, "ref": ref, "phase": phase, "frame_idx": idx, "cam": cam, "raw_sha256": raw,
                            "sources": None, "transform": None, "encoded_sha256": None})
                continue
            fi = None
            if px is not None:
                for i in list(range(nxt, len(vh_hashes))) + list(range(0, min(nxt, len(vh_hashes)))):
                    if vh_hashes[i] == px:
                        fi = i
                        break
            if fi is None:
                check["demo_unmatched"] += 1
            else:
                nxt = fi + 1
                demo_idx.append(fi)
            out.append({"slot": k, "ref": "keyframe", "phase": "demo", "frame_idx": fi, "cam": "front",
                        "raw_sha256": px if fi is None else vh_hashes[fi], "sources": None, "transform": None,
                        "encoded_sha256": None})
        demo_video = f"demo[{min(demo_idx)}:{max(demo_idx) + 1}]" if demo_idx else None
        return out, demo_video, check

    async def act(self, obs: dict) -> dict:
        step = self.actions_received
        self.trace.log_request(OBSERVATION, canonical_frame_bytes(OBSERVATION, obs), step=step)
        self.frames_sent += 1
        cid = None
        if self.lang is not None:  # 发送前落盘：每个观测一个 action_model 调用
            self._transport_attempt = self._transport_attempt + 1 if self._failed_step == step else 0
            cid = self._lang("open_call", "action_model", step, transport_attempt=self._transport_attempt)
            if cid is not None:
                fields = {"task_description": obs.get("task_description")} if isinstance(obs, dict) else None
                self._lang("message", cid, dir="in", role="fields", text=fields, images=self._image_refs(obs, step))
        try:
            action = await self._conn.act(obs)
        except BaseException:
            if cid is not None:
                self._lang("close_call", cid, status="error")
            self._failed_step = step
            raise
        self._failed_step = None
        # 第三阶段：服务外壳审计键在动作交给环境之前 pop 掉（只进语言账本）
        self.last_audit = action.pop(AUDIT_KEY, None) if isinstance(action, dict) else None
        self.actions_received += 1
        if isinstance(action, dict) and SUBGOAL_KEY in action:
            sg = action[SUBGOAL_KEY]
            self.last_subgoal = None if sg is None else str(sg)
        else:
            self.last_subgoal = UNSET
        raw = action.get("actions", action.get("action")) if isinstance(action, dict) else None
        self.trace.log_response(raw, step=step)
        if cid is not None:
            final_text = final_trunc = None
            if self.lang is not None:
                try:
                    final_text, final_trunc = load_trace_writer().audit_channel_messages(self.lang, cid,
                                                                                         self.last_audit)
                except Exception as e:  # noqa: BLE001
                    self.lang_errors += 1
                    print(f"TRACE_HOOK_ERROR route=pp/new where=language.audit {type(e).__name__}: {e}"[:600],
                          flush=True)
            # 服务外壳标了 text_reconstructed 的通道（S1 prompt 原文是重拼的，token_ids／mask 才是真实截获）：message 行
            # schema 冻结、没有自由字段，标志落在本调用 call_close.parsed（无标志时 parsed 仍为 None，与之前相同）
            chans = self.last_audit.get("channels") if isinstance(self.last_audit, dict) else None
            recon = sorted({str(c.get("channel")) for c in (chans or [])
                            if isinstance(c, dict) and c.get("text_reconstructed")})
            self._lang("close_call", cid, status="reply", server_final_text=final_text,
                       server_truncated=final_trunc,
                       parsed={"text_reconstructed_channels": recon} if recon else None)
            gen = self.last_audit.get("pp_generation") if isinstance(self.last_audit, dict) else None
            for g in (gen if isinstance(gen, list) else [gen] if gen is not None else []):
                self._log_generation(step, g, obs)
        self.last_call_id = cid
        return action

    async def end_episode(self, result: dict) -> None:
        self.trace.log_request(EPISODE_END, canonical_frame_bytes(EPISODE_END, result), step=self.actions_received)
        self.frames_sent += 1
        await self._conn.end_episode(result)


def resolve_trace_path(identity: dict, conn_info: dict, recorder: Any) -> Path | None:
    """轨迹文件位置：``conn_info["trace_path"]``；否则 ``conn_info["trace_dir"]``（env_client 给的**每局目录**
    ``<trace-root>/<key>.a<attempt>``，SeatRunner 不预先建，由 TraceWriter 建父目录；未给 ``--trace-root`` 时为
    None）下的 ``trace.jsonl``；否则录像器目录下 ``trace.jsonl``；都没有则不写轨迹。"""
    if conn_info.get("trace_path"):
        return Path(conn_info["trace_path"])
    if conn_info.get("trace_dir"):
        return Path(conn_info["trace_dir"]) / "trace.jsonl"
    rec_dir = getattr(recorder, "out_dir", None)
    if rec_dir:
        return Path(rec_dir) / "trace.jsonl"
    return None


def is_connection_closed(exc: BaseException) -> bool:
    try:
        import websockets.exceptions as wse
    except Exception:  # noqa: BLE001
        return type(exc).__name__ == "ConnectionClosed"
    return isinstance(exc, wse.ConnectionClosed)


def classify_exception(exc: BaseException) -> tuple[str, str | None, bool]:
    """``(status, infra_reason, infra)``：连接／服务类为基础设施故障，``StepCapReached`` 记 timeout，其余记 error。"""
    name = type(exc).__name__
    if name == "StepCapReached":
        return "timeout", None, False
    if name == "SubgoalMissing":
        return "error", "pp_subgoal_missing", True
    if name == "RecorderError":
        return "error", "recorder", True
    if isinstance(exc, ConnectionError):
        return "error", "pp_unreachable", True
    if isinstance(exc, TimeoutError):
        return "error", "pp_act_timeout", True
    if is_connection_closed(exc):
        return "error", "pp_connection_closed", True
    if isinstance(exc, RuntimeError) and str(exc).startswith("Server error"):
        return "error", "pp_server_error", True
    return "error", None, False


def default_connection_factory(url: str, timeout: float) -> Any:
    """运行时才导入 vla-eval（客户端扩展环境 client-env 内）。"""
    from vla_eval.connection import Connection

    return Connection(url, timeout=timeout)


# ── 新侧一局 ───────────────────────────────────────────────────────────────


class _EnvError(Exception):
    """包住 ``session.step``／``session.reset`` 里环境侧抛出的异常，与连接异常区分。"""

    def __init__(self, inner: BaseException):
        super().__init__(f"{type(inner).__name__}: {inner}")
        self.inner = inner


def run_episode(session, identity: dict, conn_info: dict, recorder, *,
                connection_factory: Callable[[str, float], Any] | None = None) -> dict:
    """env_client 调用入口：一局 PonderPounce。``session`` 为已 build 的 EnvSession（或同接口替身）。"""
    import anyio

    return anyio.run(_run_episode_async, session, identity, conn_info, recorder,
                     connection_factory or default_connection_factory)


async def _run_episode_async(session, identity: dict, conn_info: dict, recorder, connection_factory) -> dict:
    timing: dict[str, Any] = {}
    max_steps = int(conn_info["max_steps"])
    dataset = conn_info.get("dataset")
    sid = fixed_sid(identity, dataset)
    eid = sid
    task = identity["task"]
    ep_idx = episode_idx_of(identity)
    max_reconnects = int(conn_info.get("pp_max_reconnects", PP_MAX_RECONNECTS))
    url = f"ws://{conn_info.get('host', '127.0.0.1')}:{int(conn_info['port'])}"
    trace_path = resolve_trace_path(identity, conn_info, recorder)
    phase2 = phase2_enabled(conn_info)
    if trace_path is not None:
        TraceWriter = load_trace_writer().TraceWriter
        ident_t = {"task": task, "tier": identity.get("tier"), "seed": identity.get("seed"),
                   "source_episode": identity.get("source_episode"),
                   "builder_episode": identity.get("builder_episode"),
                   "key": identity.get("key"), "dataset": dataset, "sid": sid,
                   "episode_idx": ep_idx, "side": "new"}
        # 开关关（旧口径）时不收集完整数组：局目录只有 trace.jsonl、end 行无 arrays 摘要，与 BASE 逐字节相同
        tw_kw: dict[str, Any] = {"collect_arrays": phase2}
        if phase2:  # C6：加尝试号（开关关时 identity 与 BASE 逐字节相同）
            ident_t["attempt"] = attempt_of(identity, conn_info, trace_path)
            seed = conn_info.get("policy_seed", identity.get("policy_seed"))
            if seed is not None:  # 冻结说明四.1：identity／header 记 policy_seed；未给时不写
                ident_t["policy_seed"] = int(seed)
                tw_kw["policy_seed"] = int(seed)
            if conn_info.get("effective_cap") is not None:
                tw_kw["effective_cap"] = int(conn_info["effective_cap"])
        trace = TraceWriter(trace_path, route=TRACE_ROUTE_NEW_C1 if phase2 else TRACE_SCHEMA_ROUTE_NEW,
                            max_steps=max_steps, identity=ident_t, **tw_kw)
    else:
        trace = NullTrace()
    lang = None
    if phase2 and trace_path is not None:  # 第三阶段语言账本：轨迹同目录 language.jsonl
        try:
            LanguageLog = getattr(load_trace_writer(), "LanguageLog", None)
            lang = LanguageLog(trace_path.parent / "language.jsonl") if LanguageLog is not None else None
        except Exception as e:  # noqa: BLE001
            lang = None
            print(f"TRACE_HOOK_ERROR route=pp/new where=language.init {type(e).__name__}: {e}"[:600], flush=True)

    result: dict[str, Any] = {"side": "new", "sid": sid, "eid": eid, "episode_idx": ep_idx, "max_steps": max_steps,
                              "reconnects": 0, "frames_sent": 0, "decisions": 0, "steps": 0, "demo_frames": None}
    packer = ObsPacker()
    conn = connection_factory(url, PP_TIMEOUT_S)
    tconn = TracedConnection(conn, trace, lang=lang)
    status, error, infra, infra_reason, env_exc = "error", None, False, None, None
    executed = 0
    attempted = observed = 0  # C8（只在第二阶段写进 end）
    # C4：交给环境的动作原值由 TraceWriter 逐步收集，收尾经 merge_write_npz 写 arrays.npz（第三阶段，不再直接 np.savez）
    model_subgoal: Any = UNSET  # 第二阶段：本步回包里的模型子目标
    trace_reason = None
    t_start = time.perf_counter()
    try:
        t0 = time.perf_counter()
        await conn.connect(benchmark=PP_BENCHMARK)
        timing["connect_s"] = time.perf_counter() - t0

        bench_t0 = time.monotonic()  # StepBenchmark.start_episode 在 reset 之前取 _t0
        try:
            raw_obs, info = session.reset()
        except Exception as e:  # noqa: BLE001
            raise _EnvError(e) from e
        packer.on_reset(raw_obs, info)
        result["demo_frames"] = trace_reset(trace, raw_obs, packer.task_description, include_initial=phase2)
        tconn.demo_index = result["demo_frames"]  # include_initial 时初始画面在 demo 行的下标 = 演示帧数
        obs = packer.make(raw_obs)
        start = episode_start_payload(task, ep_idx, sid, eid)
        await tconn.start_episode(start)
        _rec_event(recorder, {"kind": "pp_episode_start", "sid": sid, "eid": eid, "task": task,
                              "episode_idx": ep_idx, "max_steps": max_steps, "url": url})

        last_info: dict = {}
        done = truncated = False
        step = -1
        for step in range(max_steps):
            action = await _act_with_reconnect(tconn, conn, obs, result, max_reconnects, recorder)
            raw = action.get("actions", action.get("action")) if isinstance(action, dict) else None
            if raw is not None:
                _rec_array(recorder, "model_action", np.array(raw, copy=True), step)
            a8 = exec_action8(action)
            if phase2:
                if _is_unset(tconn.last_subgoal):
                    raise SubgoalMissing(f"第二阶段开着但第 {step + 1} 个 ACTION 回包没有 {SUBGOAL_KEY!r} 键"
                                         "（服务端外壳 pp_server_wrap.py 未生效）")
                model_subgoal = tconn.last_subgoal
            link = {"source_call_id": tconn.last_call_id, "chunk_index": 0} \
                if tconn.last_call_id is not None else {}
            try:
                out = session.step(a8)
            except Exception as e:  # noqa: BLE001
                if phase2 and type(e).__name__ != "StepCapReached":  # 已进入环境的异常步（C8：含异常步）
                    attempted += 1
                    trace.log_missing_step(step=attempted, action=np.asarray(a8, dtype=np.float64),
                                           reason=f"env_exception:{type(e).__name__}",
                                           subgoal=pp_subgoal_to_official(model_subgoal), subgoal_raw=model_subgoal,
                                           **link)
                raise _EnvError(e) from e
            executed += 1
            if phase2:
                attempted += 1
                observed += bool(trace_step(trace, executed, out, a8, subgoal=model_subgoal, link=link))
            else:
                trace_step(trace, executed, out, a8)
            raw_obs, _reward, terminated, truncated, last_info = out
            last_info = last_info if isinstance(last_info, dict) else {}
            done = step_done(terminated, truncated, last_info)
            if done:
                break
            obs = packer.make(raw_obs)

        status, error = terminal_status(done, bool(truncated), last_info)
        trace_reason = "env_done" if done else "loop_exit"
        elapsed = time.monotonic() - bench_t0
        await tconn.end_episode(episode_end_payload(last_info.get("status") == "success", step + 1, elapsed))
    except _EnvError as e:
        inner = e.inner
        status, infra_reason, infra = classify_exception(inner)
        error = None if status == "timeout" else f"{type(inner).__name__}: {inner}"[:800]
        env_exc = f"{type(inner).__name__}: {inner}"[:800]
        trace_reason = "step_cap" if status == "timeout" else "env_exception"
    except Exception as e:  # noqa: BLE001 连接、服务或其他异常
        status, infra_reason, infra = classify_exception(e)
        error = f"{type(e).__name__}: {e}"[:800]
        trace_reason = f"exception:{type(e).__name__}"
    finally:
        try:
            await conn.close()
        except Exception:  # noqa: BLE001
            pass
    timing["episode_s"] = time.perf_counter() - t_start
    result.update(status=status, task_success=status == "success", steps=executed, error=error, infra=bool(infra),
                  infra_reason=infra_reason, env_exception=env_exc, frames_sent=tconn.frames_sent,
                  decisions=tconn.actions_received, timing=timing)
    if not phase2:
        trace.close(status=status, terminal_reason=trace_reason, sid=sid, frames_sent=tconn.frames_sent,
                    reconnects=result["reconnects"])
    else:
        no_frame = result["demo_frames"] is None  # reset 之前就失败：没有任何画面（C3）
        demo = 0 if no_frame else int(result["demo_frames"])
        end_extra: dict[str, Any] = {
            "sid": sid, "frames_sent": tconn.frames_sent, "reconnects": result["reconnects"], "side": "new",
            "exit_reason": trace_reason, "demo_frames": demo, "steps_attempted": attempted,
            "steps_observed": observed, "frames_recorded": 0 if no_frame else demo + 1 + observed}
        if no_frame:
            end_extra["no_frame"] = True
        if lang is not None:  # 先关语言账本（悬空调用补 cancelled）
            try:
                lang.close()
            except Exception as e:  # noqa: BLE001
                tconn.lang_errors += 1
                print(f"TRACE_HOOK_ERROR route=pp/new where=language.close {type(e).__name__}: {e}"[:600], flush=True)
            end_extra["language_hook_errors"] = tconn.lang_errors
        # arrays.npz：TraceWriter.close 经 merge_write_npz 写（同目录录像器的同名键同值合并、分目录各写各的），
        # 写盘失败记 end.arrays.error，由 TRACE_ARRAYS 检查器判失败
        trace.close(status=status, terminal_reason=status, **end_extra)
    _rec_event(recorder, {"kind": "pp_episode_end", "sid": sid, "status": status, "steps": executed,
                          "frames_sent": tconn.frames_sent, "reconnects": result["reconnects"], "error": error})
    return result


async def _act_with_reconnect(tconn: TracedConnection, conn: Any, obs: dict, result: dict, max_reconnects: int,
                              recorder) -> dict:
    """同一局内 ConnectionClosed：``reconnect()``（含 HELLO）后重发同一条观测，不重发 EPISODE_START。"""
    while True:
        try:
            return await tconn.act(obs)
        except Exception as e:  # noqa: BLE001
            if not is_connection_closed(e) or result["reconnects"] >= max_reconnects:
                raise
            result["reconnects"] += 1
            _rec_event(recorder, {"kind": "pp_reconnect", "n": result["reconnects"], "error": repr(e)[:400],
                                  "decision": tconn.actions_received})
            await conn.reconnect()


def _same_dir(a: Any, b: Any) -> bool:
    """两个目录是否同一路径（``b`` 为空时 False）；用于判断轨迹是否就写在录像器目录里。"""
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:  # noqa: BLE001
        return False


def _rec_event(recorder, event: dict) -> None:
    if recorder is not None and hasattr(recorder, "add_event"):
        recorder.add_event(event)


def _rec_array(recorder, name: str, arr: np.ndarray, step: int) -> None:
    if recorder is not None and hasattr(recorder, "add_array"):
        recorder.add_array(name, arr, step=step)


# ── 新接口：模型侧 4 个方法（拆分方案 §三「五个模型各自怎么落」） ─────────────────────

from robomme_hard_eval import servers as _servers  # noqa: E402
from robomme_hard_eval.policy import Ready  # noqa: E402


def pp_server_spec(policy, ckpt: Any) -> tuple[list, dict, Path]:
    """PonderPounce 服务端的命令、环境与 cwd，照抄旧 ``run_seat.sh::build_server_cmd`` 的 pp 分支：cwd 为
    ``third_party/PonderPounce``，HF 离线（``HF_HOME`` 沿用调用方环境）；外壳打开（缺省，旧 GL 启动器
    ``SGEVAL_PP_SERVER_WRAP=1``）时以绝对路径起本包 ``servers/pp_server_wrap.py``，否则 ``-m
    ponderpounce.eval.robomme_server``；其后 ``--args.checkpoint_path --args.seed --args.device cuda:0 --port``。
    解释器取 ``cfg["pp_py"]`` → 环境变量 ``PP_PY`` → ``third_party/PonderPounce/.venv/bin/python``。"""
    S = _servers
    cfg = policy.cfg
    sub = policy.root / "third_party" / "PonderPounce"
    py = S.interpreter(cfg, "pp_py", "PP_PY", sub / ".venv" / "bin" / "python")
    env = {"PYTHONUNBUFFERED": "1",
           "HF_HUB_OFFLINE": str(cfg.get("hf_hub_offline", os.environ.get("HF_HUB_OFFLINE", "1"))),
           "TRANSFORMERS_OFFLINE": str(cfg.get("transformers_offline", os.environ.get("TRANSFORMERS_OFFLINE", "1"))),
           PHASE2_ENV: "1" if policy.server_wrap else "0"}
    if policy.server_wrap:
        argv = [str(py), str(S.SERVERS_DIR / "pp_server_wrap.py"), f"--sgeval-metadata-out={policy.wrap_meta}"]
    else:
        argv = [str(py), "-m", "ponderpounce.eval.robomme_server"]
    argv += ["--args.checkpoint_path", str(ckpt), "--args.seed", str(int(policy.policy_seed)), "--args.device",
             "cuda:0", "--port", str(int(policy.port))]
    return argv, env, sub


class PPPolicy(_servers.ServedPolicy):
    """PonderPounce（``pp``）。

    * ``load``：预检（子模块、解释器、ckpt 与 ``norm_stats.json``、外壳）→ 起（或 attach）服务端，就绪 =
      ``GET /health`` 返回 200；不加预热；核本进程能导入 ``vla_eval``（客户端须在 ``envs/client-env`` 里跑）；
    * ``reset(spec)``：不发消息、不碰环境——探活后算本局 sid ``<task>|<source_episode 或 tier>|<seed>``；本服务进程
      已用过它（只会在同一身份重跑、``spec.attempt > 1`` 时出现）才重起服务端（S3），保证服务端按
      ``crc32(f"{seed}:{sid}:{n}")`` 取噪声时 ``n`` 从 0 开始；
    * ``play``：原样调本模块 ``run_episode``（每局照旧发 ``EPISODE_START``，断线只在本局内重连一次），结果另记
      ``pp_sid`` 与 ``pp_sid_use_index``（该 sid 在本服务进程里的使用序号）；
    * ``close``：停服务端进程组（基类）。

    必填 cfg：``ckpt``（PonderPounce 没有缺省 ckpt）。可选 ``pp_py``、``pp_server_wrap``（缺省开）、
    ``pp_max_reconnects``（缺省 1）、``hf_hub_offline``、``transformers_offline``；``connection_factory`` 只供单测注入。"""

    model = "pp"

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        self.server_wrap = _servers.flag_on(self.cfg.get("pp_server_wrap", True))
        self.ckpt = self.cfg.get("ckpt") or _servers.DEFAULT_CKPTS[self.model]
        self.sid_uses: dict[str, int] = {}   # 本服务进程里每个 sid 已发过 EPISODE_START 的次数
        self.server_restarts = 0
        self.current_sid: str | None = None
        self._spec_argv: tuple | None = None

    def load(self) -> None:
        S = _servers
        if not self.ckpt:
            raise S.PreflightError("RUN_BLOCKED reason=pp_ckpt_missing ckpt=unset（PonderPounce 须显式给 ckpt）")
        self._pick_port()
        argv, env, cwd = pp_server_spec(self, self.ckpt)
        if self.preflight:
            S.preflight_pp(self.root, self.ckpt, Path(argv[0]), server_wrap=self.server_wrap, seed=self.policy_seed)
            import importlib.util

            if self.cfg.get("connection_factory") is None and importlib.util.find_spec("vla_eval") is None:
                raise S.PreflightError("RUN_BLOCKED reason=client_env detail=本进程导入不了 vla_eval（PonderPounce 客户端须用 "
                                       "envs/client-env/.venv 的解释器跑）")
        self._spec_argv = (argv, env, cwd)
        self._launch(argv, env, cwd, [Ready.health()], self.ckpt)
        self._fingerprint(self.ckpt)

    def _restart_server(self, why: str) -> None:
        """重起服务端（同端口、同命令）：停旧进程组、新起并等 ``/health``，清空本进程 sid 使用表。"""
        argv, env, cwd = self._spec_argv
        print(f"PP_SERVER_RESTART reason={why} port={self.port}", flush=True)
        if self.server is not None:
            self.server.stop()
        self._launch(argv, env, cwd, [Ready.health()], self.ckpt)
        self.sid_uses.clear()
        self.server_restarts += 1

    def reset(self, spec) -> None:
        super().reset(spec)
        sid = fixed_sid(spec.identity(), spec.dataset)
        if self.sid_uses.get(sid, 0) > 0:
            self._restart_server(f"sid_reused sid={sid} attempt={spec.attempt}")
        self.current_sid = sid

    def play(self, session, spec, recorder) -> dict:
        sid = fixed_sid(spec.identity(), spec.dataset)
        use_index = self.sid_uses.get(sid, 0)
        conn = self._conn_info(spec)
        conn.update(pp_phase2=self.server_wrap, pp_max_reconnects=int(self.cfg.get("pp_max_reconnects",
                                                                                   PP_MAX_RECONNECTS)))
        try:
            res = run_episode(session, spec.identity(), conn, recorder,
                              connection_factory=self.cfg.get("connection_factory"))
        finally:
            self.sid_uses[sid] = use_index + 1  # 保守计：本局可能已发出 EPISODE_START
        res.update(pp_sid=sid, pp_sid_use_index=use_index, pp_server_restarts=self.server_restarts)
        return self._finish_play(res)
