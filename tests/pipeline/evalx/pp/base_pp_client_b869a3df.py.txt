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
``test-hard0`` 时要求 ``tier == "xhard0"``）、``trace_path``／``trace_dir``（可选；``trace_dir`` 为每局目录，
见 ``resolve_trace_path``）、
``pp_max_reconnects``（可选，缺省 1）。

导入期只依赖标准库与 numpy；``vla_eval``（客户端扩展环境 client-env 内）、``anyio``、``msgpack``、``websockets``
都在用到时才导入。单测通过 ``run_episode(..., connection_factory=...)`` 注入替身连接。
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
# 本目录只挂在 sys.path 末尾，防止同目录模块遮蔽标准库
sys.path[:] = [p for p in sys.path if p and str(_Path(p).resolve()) != _HERE] + [_HERE]

import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402

#: HELLO 握手里带的 benchmark 名，与 PonderPounce 官方 configs/robomme.yaml 的 ``benchmark`` 相同
PP_BENCHMARK = "vla_eval.benchmarks.robomme.benchmark:RoboMMEBenchmark"
#: 每次 recv 的超时（秒），与官方 configs/robomme.yaml 的 ``server.timeout`` 相同
PP_TIMEOUT_S = 300.0
#: 交给环境的动作维数（7 关节 + 1 夹爪）
PP_ACTION_DIMS = 8
#: 同一局内 ConnectionClosed 后的重连次数上限（缺省）
PP_MAX_RECONNECTS = 1
XHARD0 = "xhard0"
TEST_HARD0 = "test-hard0"
TRACE_SCHEMA_ROUTE_NEW = "pp-new"
TRACE_SCHEMA_ROUTE_ORIG = "pp-orig"
#: 协议帧类型（与 vla_eval.protocol.messages.MessageType 的取值相同）
HELLO, OBSERVATION, ACTION, EPISODE_START, EPISODE_END, ERROR = (
    "hello", "observation", "action", "episode_start", "episode_end", "error")


# ── 身份与协议载荷 ──────────────────────────────────────────────────────────


def fixed_sid(identity: dict, dataset: str | None = None) -> str:
    """固定 sid：xhard0 ``<task>|<source_episode>|<seed>``；V9 ``<task>|<tier>|<seed>``。

    ``dataset == "test-hard0"`` 时身份必须是 xhard0，否则 ``ValueError``（防止 V9 身份混进 hard0 分片）。"""
    task, tier, seed = identity["task"], identity.get("tier"), identity["seed"]
    if dataset == TEST_HARD0 and tier != XHARD0:
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
    def close(self, *a, **k): pass


def trace_reset(trace, raw_obs: dict, task_description: str) -> int:
    """演示行：``video_history``（= ``front_rgb_list[:-1]``）逐帧画面哈希、腕部同位帧、逐帧 8 维状态与指令文本。
    返回演示帧数。"""
    fronts = list(raw_obs["front_rgb_list"][:-1])
    wrists = list(raw_obs.get("wrist_rgb_list", [])[:-1])
    joints = list(raw_obs.get("joint_state_list", []) or [])
    grips = list(raw_obs.get("gripper_state_list", []) or [])
    states = [state8(j, g) for j, g in zip(joints[:-1], grips[:-1])]
    trace.log_demo(fronts, wrists, states=states, texts=[task_description])
    return len(fronts)


def trace_step(trace, step_no: int, out: tuple, action8: list[float]) -> None:
    """执行完第 ``step_no`` 步（从 1 计）的一行；``out`` 为环境 ``step`` 的五元组。"""
    obs, _reward, terminated, truncated, info = out
    info = info if isinstance(info, dict) else {}
    front = wrist = state = None
    if isinstance(obs, dict) and obs.get("front_rgb_list"):
        front = obs["front_rgb_list"][-1]
        if obs.get("wrist_rgb_list"):
            wrist = obs["wrist_rgb_list"][-1]
        if obs.get("joint_state_list") and obs.get("gripper_state_list"):
            state = state8(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
    subgoal = info.get("simple_subgoal_online")
    trace.log_step(step=step_no, front=front, wrist=wrist, state=state,
                   action=np.asarray(action8, dtype=np.float64), subgoal=None if subgoal is None else str(subgoal),
                   terminated=bool(terminated), truncated=bool(truncated), status=info.get("status"))


class TracedConnection:
    """包住 vla-eval ``Connection``：局内每个协议帧写轨迹 request／response 行并计数；其余属性透传。

    ``SyncEpisodeRunner`` 只调用 ``start_episode``／``act``／``end_episode``，原侧把本包装直接交给它。"""

    def __init__(self, conn: Any, trace=None) -> None:
        self._conn = conn
        self.trace = trace if trace is not None else NullTrace()
        self.frames_sent = 0
        self.actions_received = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def start_episode(self, config: dict) -> None:
        self.trace.log_request(EPISODE_START, canonical_frame_bytes(EPISODE_START, config), step=0)
        self.frames_sent += 1
        await self._conn.start_episode(config)

    async def act(self, obs: dict) -> dict:
        step = self.actions_received
        self.trace.log_request(OBSERVATION, canonical_frame_bytes(OBSERVATION, obs), step=step)
        self.frames_sent += 1
        action = await self._conn.act(obs)
        self.actions_received += 1
        raw = action.get("actions", action.get("action")) if isinstance(action, dict) else None
        self.trace.log_response(raw, step=step)
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


def load_trace_writer():
    """同目录 ``trace_writer`` 模块：已导入则复用，否则按文件路径加载（不依赖调用时的 sys.path）。"""
    mod = sys.modules.get("trace_writer")
    if mod is not None:
        return mod
    import importlib.util

    spec = importlib.util.spec_from_file_location("trace_writer", Path(_HERE) / "trace_writer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trace_writer"] = mod
    spec.loader.exec_module(mod)
    return mod


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
    if trace_path is not None:
        TraceWriter = load_trace_writer().TraceWriter
        trace = TraceWriter(trace_path, route=TRACE_SCHEMA_ROUTE_NEW, max_steps=max_steps,
                            identity={"task": task, "tier": identity.get("tier"), "seed": identity.get("seed"),
                                      "source_episode": identity.get("source_episode"),
                                      "builder_episode": identity.get("builder_episode"),
                                      "key": identity.get("key"), "dataset": dataset, "sid": sid,
                                      "episode_idx": ep_idx, "side": "new"})
    else:
        trace = NullTrace()

    result: dict[str, Any] = {"side": "new", "sid": sid, "eid": eid, "episode_idx": ep_idx, "max_steps": max_steps,
                              "reconnects": 0, "frames_sent": 0, "decisions": 0, "steps": 0, "demo_frames": None}
    packer = ObsPacker()
    conn = connection_factory(url, PP_TIMEOUT_S)
    tconn = TracedConnection(conn, trace)
    status, error, infra, infra_reason, env_exc = "error", None, False, None, None
    executed = 0
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
        result["demo_frames"] = trace_reset(trace, raw_obs, packer.task_description)
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
            try:
                out = session.step(a8)
            except Exception as e:  # noqa: BLE001
                raise _EnvError(e) from e
            executed += 1
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
    trace.close(status=status, terminal_reason=trace_reason, sid=sid, frames_sent=tconn.frames_sent,
                reconnects=result["reconnects"])
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


def _rec_event(recorder, event: dict) -> None:
    if recorder is not None and hasattr(recorder, "add_event"):
        recorder.add_event(event)


def _rec_array(recorder, name: str, arr: np.ndarray, step: int) -> None:
    if recorder is not None and hasattr(recorder, "add_array"):
        recorder.add_array(name, arr, step=step)
