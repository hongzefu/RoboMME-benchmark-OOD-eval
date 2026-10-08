"""SimpleMemVLA 新接口客户端（v7.5eval 方案 §2.3；运行在 benchmark .venv）。

``run_episode(session, identity, conn_info, recorder)`` 逐行复现旧官方
``4e0c04f robomme_sim/eval_success.py::run_group``（组大小 1、details 模式）与
``evaluate_manifest`` 的逐局收尾，以及 ``robomme_sim/robomme_env.py::SimEnvService`` 的
reset/step 打包语义；环境由 ``env_client.EnvSession`` 在本进程建，策略在 smvla_server 进程。

与旧官方的对应（行号指 SimpleMemVLA-official-xhard0 @4e0c04f）：
- reset：SimEnvService.reset（重试 reset_retries=2 次，帧/状态为空算失败，instruction 取
  info["task_goal"][0]）→ 发 reset + observe(全部演示帧 + 初始帧)，cur_state = states[-1] (float32)。
- 决策循环：hard_bound = ceil(max_steps/16)+2（1300 步时 84）；每次决策 infer → chunk =
  actions[:16] → SimEnvService.step 逐行 ``np.asarray(row, float64).reshape(-1)[:8]``，遇
  done / error / obs None 即停；返回帧一律 observe；steps += consumed。
- 终态：环境报 done 取环境 status（error 时带 error_message）；step 抛异常记
  ``step_exc: <e>``（steps 不加本块）；决策用尽而环境未报终态记 timeout，error 为
  ``策略循环 hard_bound=84 用尽，环境未报终态``（steps 通常为 1344）；其余异常记 error + traceback。

纯函数层（encode_frames / encode_states / instruction_from_info / step_chunk / hard_bound）与
IO 层（WSPolicyConn）分开，单测用假 session / 假连接直接驱动。

第二阶段 S4（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S4」，契约 C1～C11 见
``trace_writer.py`` 模块文档串）：每局另写 ``trace.jsonl``（route ``smvla/new``），落点沿用
``groundsg_client.trace_location`` 的约定（``trace_path`` → ``<trace_dir>/trace.jsonl`` → ``<recorder.out_dir>/trace.jsonl``，
都没有则不写）。``PolicyTrace`` 是 SimpleMemVLA／FrameSamp+Modulation 两条新侧路线共用的记录器（``framesamp_modul_client`` 按文件路径复用）：

- 演示段按 C2 记全部 reset 帧（含初始帧），收尾 ``demo_frames = 帧数 - 1``；
- 每个交给环境的步一行：有观测记 ``log_step``（一步多帧取最后一帧），没有有效观测（step 抛异常且环境侧已计步、
  ``obs is None``、``status == "error"``）记 ``log_missing_step``；``steps_attempted`` 以环境会话的 ``steps``
  增量为准，与结果行 ``exec_steps`` 同口径（C8）；
- 子目标：SimpleMemVLA 取决策回包 ``subtask``（``smvla_server.py`` ``infer`` 回包键），FrameSamp+Modulation 全程 ``None``（C7）；
- 请求／响应（C10）：SimpleMemVLA 记逻辑输入（指令、状态、上次推理以来 observe 的帧哈希序列）与完整动作块
  ``actions_full``；FrameSamp+Modulation 记原始 msgpack 帧字节的 sha256 与 ``infer`` 回包动作块；
- strict-cap（``StepCapReached`` 或会话 ``cap_hit``）一律按 ``timeout`` 收尾（C3）；
- 完整数值（第三阶段，冻结说明第四节）：每步动作／状态原值由 ``TraceWriter`` 收集，收尾经
  ``trace_writer.merge_write_npz`` 写轨迹目录 ``arrays.npz``（``exec_action__%05d``／``exec_state__%05d``）；与录像器
  同目录时与其 ``exec_action__%05d`` 同键同值合并，先后任意都不覆盖；本模块不再直接 ``np.savez``；
- 语言账本（第三阶段，冻结说明第五节）：轨迹同目录 ``language.jsonl``。SimpleMemVLA 每次决策一个
  ``action_model`` 调用：发送前写 ``in``（``role=fields``，任务目标），回包后写服务外壳审计块的逐通道分词与
  ``out``（``role=assistant``，子任务原文），``close_call`` 记 ``server_final_text``（服务端模板化 prompt，缺审计块记
  None）；执行步 ``source_call_id``／``chunk_index`` 指向该调用。回包里的 ``_sgeval_audit`` 一律先 pop 掉，不进
  动作、不改请求字节；
- 记录器内任何异常只计 ``observer_hook_errors`` 并打印 ``TRACE_HOOK_ERROR``，不改变发给服务的请求、交给环境的
  动作与控制流（后续 ``CLIENT_REPLAY_EQ`` 回放比对）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

MAX_STEPS = 1300
EXECUTE_HORIZON = 16
RESET_RETRIES = 2
FINAL_STATUSES = ("success", "fail", "timeout")
CAM_FRONT = "front"
CAM_WRIST = "wrist"


class ProtocolError(RuntimeError):
    """消息协议损坏（回包 sha 与发出内容不符、回包缺键）。属运行阻塞。"""


class ServerError(RuntimeError):
    """server 回包 {"error": traceback}：server 侧异常，不是环境结局。"""


# 基础设施故障标记（与 framesamp_modul_client.INFRA_MARKERS 同表）：只决定 infra 标记（env_client 据此重试），不改 status。
INFRA_MARKERS = ("RecorderError", "svulkan2", "EXCLUSIVE", "Vulkan", "vk::", "out of memory", "RESOURCE_EXHAUSTED",
                 "CUDA_ERROR", "ConnectionClosed", "ConnectionRefused", "InvalidStatus", "Connection reset")
ENV_RESET_INFRA_MARKERS = ("svulkan2", "EXCLUSIVE", "Vulkan", "vk::")
SERVER_OOM_MARKERS = ("out of memory", "OutOfMemory", "CUDA", "CUBLAS", "cuDNN")


def _marker(text: str | None, markers) -> str | None:
    for m in markers:
        if m in (text or ""):
            return m
    return None


def classify_exception(e: BaseException, tb: str) -> str | None:
    """整局兜底异常 → infra_reason（None 表示非基础设施）。"""
    if isinstance(e, ServerError):
        return "server_oom" if _marker(str(e), SERVER_OOM_MARKERS) else "server_error"
    if isinstance(e, ProtocolError):
        return "protocol"
    try:
        from websockets.exceptions import ConnectionClosed, InvalidHandshake
        if isinstance(e, (ConnectionClosed, InvalidHandshake)):
            return f"connection:{type(e).__name__}"
    except ImportError:
        pass
    if isinstance(e, (OSError, TimeoutError)):  # ConnectionRefusedError / reset / 连接超时
        return f"connection:{type(e).__name__}"
    m = _marker(tb, INFRA_MARKERS)
    return f"marker:{m}" if m else None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frame_sha(frame: np.ndarray) -> str:
    """单帧指纹（与 recorder.frame_sha256、smvla_server.frame_sha 同一定义）。"""
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def array_sha(arr: np.ndarray) -> str:
    """数值数组指纹（与 recorder.array_sha256、smvla_server.array_sha 同一定义）。"""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(a.tobytes())
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    return h.hexdigest()


def hard_bound(max_steps: int = MAX_STEPS, execute_horizon: int = EXECUTE_HORIZON) -> int:
    """照抄 run_group：``max(1, -(-int(args.max_steps) // max(1, args.execute_horizon))) + 2``。"""
    return max(1, -(-int(max_steps) // max(1, execute_horizon))) + 2


def timeout_error(hb: int) -> str:
    return f"策略循环 hard_bound={hb} 用尽，环境未报终态"


# ---- 以下五个函数逐行照抄 robomme_sim/robomme_env.py（c564c17 = 4e0c04f）第 33–75 行 ----
def _to_uint8_hwc(img) -> np.ndarray:
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.uint8, copy=False)


def _to_f32(vec) -> np.ndarray:
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def encode_frames(obs: dict) -> list[dict[str, np.ndarray]]:
    fronts = obs["front_rgb_list"]
    wrists = obs["wrist_rgb_list"]
    return [
        {CAM_FRONT: _to_uint8_hwc(f), CAM_WRIST: _to_uint8_hwc(w)}
        for f, w in zip(fronts, wrists)
    ]


def encode_states(obs: dict) -> list[np.ndarray]:
    joints = obs["joint_state_list"]
    grippers = obs["gripper_state_list"]
    states = []
    for j, g in zip(joints, grippers):
        jv = _to_f32(j)[:7]
        gv = _to_f32(g)
        g0 = gv[:1] if gv.size else np.zeros(1, dtype=np.float32)
        states.append(np.concatenate([jv, g0]).astype(np.float32))
    return states


def _scalar(x) -> float:
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            return float(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x).reshape(-1)[0])
    return float(x)
# ---- 照抄结束 ----


def instruction_from_info(info: dict, task_name: str) -> str:
    """照抄 SimEnvService.reset 取 instruction 的分支。"""
    task_goal = info.get("task_goal")
    if isinstance(task_goal, (list, tuple)) and task_goal:
        return str(task_goal[0])
    return str(task_goal) if task_goal else f"Complete the {task_name} task."


def reset_session(session, task_name: str, retries: int = RESET_RETRIES) -> dict:
    """照抄 SimEnvService.reset：最多 retries+1 次；返回 {ok, instruction, frames, states} 或 {ok: False, reason}。"""
    last_err = "unknown"
    for _attempt in range(retries + 1):
        try:
            obs, info = session.reset()
            frames = encode_frames(obs)
            states = encode_states(obs)
            if not frames or not states:
                raise RuntimeError("reset returned no frames")
            return {"ok": True, "instruction": instruction_from_info(info or {}, task_name),
                    "frames": frames, "states": states, "attempts": _attempt + 1}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            try:  # 旧：出错后 env.close() 并置空，下一次重建
                session.close()
            except Exception:
                pass
    return {"ok": False, "reason": f"reset_failed: {last_err}"}


def step_chunk(session, action_chunk, on_exec=None, on_obs=None) -> dict:
    """照抄 SimEnvService.step + RoboMMESimEnv.step_one（逐行执行、遇 done/error/obs None 即停）。

    on_exec(act8) 在每行交给环境前回调（记录实际执行动作）。
    on_obs(step, front, wrist, state, terminated, truncated, *, status, error) 在每行 ``session.step`` 返回后回调
    （S4 逐步记录）：``step`` 为本块内第几步（从 1 计，即当时的 consumed）；一步返回多帧时取最后一帧的前视、腕部
    画面与 8 维状态；``status == "error"`` 或 ``obs is None`` 时画面与状态传 ``None``（缺观测步）。``session.step``
    抛异常时不回调（由调用方按环境侧计步处理）。回调只读，不改变本函数的返回值与控制流。
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    frames, states, consumed = [], [], 0
    error = None
    status = "ongoing"
    done = False
    for action in action_chunk:
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] < 8:
            raise ValueError(f"Expected an 8-dim joint_angle action, got shape {act.shape}")
        if on_exec is not None:
            on_exec(act[:8])
        obs, _reward, terminated, truncated, info = session.step(act[:8])
        term = bool(_scalar(terminated)) if terminated is not None else False
        trunc = bool(_scalar(truncated)) if truncated is not None else False
        info = info if isinstance(info, dict) else {}
        status = str(info.get("status", "ongoing"))
        done = term or trunc or status in ("success", "fail", "timeout", "error")
        consumed += 1
        if status == "error" or obs is None:
            error = str(info.get("error_message", "env step error"))
            if on_obs is not None:
                on_obs(consumed, None, None, None, term, trunc, status=status, error=error)
            break
        fr = encode_frames(obs)
        st = encode_states(obs)
        frames.extend(fr)
        states.extend(st)
        if on_obs is not None:
            on_obs(consumed, fr[-1][CAM_FRONT] if fr else None, fr[-1][CAM_WRIST] if fr else None,
                   st[-1] if st else None, term, trunc, status=status, error=None)
        if done:
            break
    return {
        "frames": frames,
        "states": states,
        "consumed": consumed,
        "done": bool(done or error is not None),
        "success": status == "success",
        "status": status,
        "error_message": error,
    }


class WSPolicyConn:
    """到 smvla_server 的同步 websocket 连接（msgpack_numpy；compression=None、max_size=None、不走代理）。"""

    def __init__(self, host: str, port: int, open_timeout: float = 600.0):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        self._m = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._ws = connect(f"ws://{host}:{int(port)}", compression=None, max_size=None, proxy=None,
                           open_timeout=open_timeout, ping_interval=None, ping_timeout=None)
        self.metadata = self._m.unpackb(self._ws.recv())

    def call(self, msg: dict) -> tuple[dict, bytes, bytes]:
        raw = self._packer.pack(msg)
        self._ws.send(raw)
        rep_raw = self._ws.recv()
        if isinstance(rep_raw, str):
            raise ProtocolError(f"server 返回文本帧：{rep_raw[:2000]}")
        return self._m.unpackb(rep_raw), raw, rep_raw

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


class _NullRecorder:
    def set_phase(self, phase):
        pass

    def add_frames(self, stream, frames, *, tag=""):
        return []

    def add_array(self, name, arr, *, step=None):
        pass

    def add_event(self, event):
        pass


def episode_key(identity: dict) -> str:
    return f"{identity['task']}/{identity['source_episode']}/{identity['seed']}"


# ---- S4：两条新侧路线（smvla／perceptual-framesamp-modul）共用的逐局轨迹记录器 ----

TRACE_TERMINALS = ("success", "fail", "timeout", "error")
#: arrays.npz 的执行动作键（C4）；与 recorder._write_arrays 的 f"{name}__{k:05d}"（name="exec_action"）逐字同名
ACTION_KEY = "exec_action__%05d"
#: 服务外壳回包审计键（冻结说明第五节；与 trace_writer.AUDIT_KEY 相同）
AUDIT_KEY = "_sgeval_audit"
_TAG_ATTEMPT = re.compile(r"\.a(\d+)$")


def load_sibling(name: str):
    """按文件路径加载本目录下的模块（别名与 env_client／groundsg_client 的 load_sibling 相同，已加载则复用）。"""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def episode_attempt(tag: str | None) -> int:
    """局目录名 ``<key>.a<N>`` 的尝试号 N；无后缀记 1（与 groundsg_client 的 ``f"{key}.a1"`` 缺省一致）。"""
    m = _TAG_ATTEMPT.search(str(tag or ""))
    return int(m.group(1)) if m else 1


def recorder_writes_arrays(recorder: Any) -> bool:
    """录像器会在 ``out_dir`` 写 ``arrays.npz``（含 ``exec_action__%05d``）：真实 ``recorder.EpisodeRecorder`` 即如此。"""
    return recorder is not None and getattr(recorder, "out_dir", None) is not None and \
        callable(getattr(recorder, "add_array", None))


class PolicyTrace:
    """一局新侧轨迹（route ``smvla/new``／``perceptual-framesamp-modul/new``）。``enabled`` 为假时（无落点）所有方法都是空操作。

    所有记录方法吞掉自身异常、计 ``hook_errors``（收尾写 ``end.observer_hook_errors``，C11），绝不改变调用方的
    请求、动作与控制流；动作一律先复制再记录。
    """

    def __init__(self, route: str, identity: dict, conn_info: dict, recorder: Any, *, max_steps: int,
                 recorder_has_actions: bool, omit_overflow_frame: bool):
        self.route = route
        self.w = None
        self.path: Path | None = None
        self.hook_errors = 0
        self.steps = 0          # steps_attempted
        self.observed = 0       # steps_observed
        self.demo_frames: int | None = None
        self.actions: list[np.ndarray | None] = []
        self.cap_hit = False
        self.max_steps = int(max_steps)
        self.recorder_has_actions = bool(recorder_has_actions)
        self.omit_overflow_frame = bool(omit_overflow_frame)
        self.encodings: set[str] = set()
        self.pending_frames: list[list[str | None]] = []
        self._tw = None
        self.lang = None        # 第三阶段语言账本（trace 同目录 language.jsonl）
        self.src_call: str | None = None  # 当前动作块来自哪次调用（log_step 的 source_call_id）
        self.chunk_pos = 0      # 当前动作块内下一步的序号（log_step 的 chunk_index）
        try:
            ci = conn_info or {}
            path = load_sibling("groundsg_client").trace_location(ci, recorder)
            if path is None:
                return
            self._tw = load_sibling("trace_writer")
            tag = ci.get("episode_tag") or f"{identity.get('key')}.a1"
            ident = {k: identity.get(k) for k in ("task", "tier", "seed", "source_episode", "builder_episode", "key")}
            ident.update(dataset=ci.get("dataset"), attempt=episode_attempt(tag))
            seed = ci.get("policy_seed", identity.get("policy_seed"))
            cap = ci.get("effective_cap")
            kw: dict[str, Any] = {}
            if seed is not None:  # 冻结说明四.1：identity／header 记 policy_seed；未给时不写，旧字节不变
                ident["policy_seed"] = int(seed)
                kw["policy_seed"] = int(seed)
            if cap is not None:
                kw["effective_cap"] = int(cap)
            self.w = self._tw.TraceWriter(path, route=route, identity=ident, max_steps=self.max_steps, **kw)
            self.path = Path(path)
        except Exception as e:  # noqa: BLE001 记录器建不起来：只告警，不影响本局
            self.w = None
            self._err("init", e)
            return
        try:
            LanguageLog = getattr(self._tw, "LanguageLog", None)
            if LanguageLog is not None:
                self.lang = LanguageLog(self.path.parent / "language.jsonl")
        except Exception as e:  # noqa: BLE001
            self.lang = None
            self._err("language.init", e)

    @property
    def enabled(self) -> bool:
        return self.w is not None

    def _err(self, where: str, e: BaseException) -> None:
        self.hook_errors += 1
        print(f"TRACE_HOOK_ERROR route={self.route} where={where} {type(e).__name__}: {e}"[:600], flush=True)

    @staticmethod
    def _copy(a: Any) -> np.ndarray | None:
        return None if a is None else np.array(a, copy=True)

    # -- 演示、请求、响应、历史 --
    def demo(self, fronts, wrists, states, goal) -> None:
        if not self.enabled:
            return
        try:
            self.demo_frames = len(fronts) - 1
            self.w.log_demo(list(fronts), list(wrists), list(states), [goal])
        except Exception as e:  # noqa: BLE001
            self._err("demo", e)

    def note_frames(self, frames) -> None:
        """SimpleMemVLA：记下已 observe 的帧（前视、腕部画面哈希），并入下一次 infer 的逻辑输入（C10）。"""
        if not self.enabled:
            return
        try:
            h = self._tw.image_sha256
            self.pending_frames.extend([[h(fr[CAM_FRONT]), h(fr[CAM_WRIST])] for fr in frames])
        except Exception as e:  # noqa: BLE001
            self._err("note_frames", e)

    def logical_request(self, name: str, instruction: str, state: Any) -> None:
        """SimpleMemVLA 的逻辑输入行：指令、状态、上次推理以来的帧哈希序列（C10，不比通信字节）。"""
        if not self.enabled:
            return
        try:
            obj = {"instruction": instruction, "state": np.array(state, copy=True), "frames": self.pending_frames}
            self.encodings.add("logical")
            self.w.log_request(name, self._tw.canonical_bytes(obj), step=self.steps)
            self.pending_frames = []
        except Exception as e:  # noqa: BLE001
            self._err("request", e)

    def raw_request(self, name: str, obj: Any, raw: bytes | None) -> None:
        """FrameSamp+Modulation：原始 msgpack 帧字节的 sha256（拿不到原始字节时退回 ``canonical_bytes``，收尾标 ``request_encoding``）。"""
        if not self.enabled:
            return
        try:
            if raw is not None:
                payload, enc = bytes(raw), "msgpack"
            else:
                payload, enc = self._tw.canonical_bytes(obj), "canonical"
            self.encodings.add(enc)
            self.w.log_request(name, payload, step=self.steps)
        except Exception as e:  # noqa: BLE001
            self._err("request", e)

    def response(self, actions: Any) -> None:
        if not self.enabled:
            return
        try:
            self.w.log_response(self._copy(actions), step=self.steps)
        except Exception as e:  # noqa: BLE001
            self._err("response", e)

    def history(self, start: int, end: int, note: str) -> None:
        if not self.enabled:
            return
        try:
            self.w.log_history(start, end, note=note)
        except Exception as e:  # noqa: BLE001
            self._err("history", e)

    # -- 第三阶段语言账本（冻结说明第五节）：全部吞异常、计 hook_errors，不改请求、动作与控制流 --
    def lang_open(self, model: str, *, step: int | None = None, **kw: Any) -> str | None:
        """开一次模型调用；``step`` 缺省取已交给环境的步数。账本不可用时返回 None（其余 lang_* 对 None 是空操作）。"""
        if self.lang is None:
            return None
        try:
            return self.lang.open_call(model, self.steps if step is None else int(step), **kw)
        except Exception as e:  # noqa: BLE001
            self._err("language.open", e)
            return None

    def lang_msg(self, call_id: str | None, **kw: Any) -> None:
        if self.lang is None or call_id is None:
            return
        try:
            self.lang.message(call_id, **kw)
        except Exception as e:  # noqa: BLE001
            self._err("language.message", e)

    def lang_audit(self, call_id: str | None, audit: Any) -> tuple[Any, Any]:
        """服务外壳审计块的逐通道分词写成 ``in`` 消息，返回 ``(server_final_text, server_truncated)``；缺审计块记 None。"""
        if self.lang is None or call_id is None:
            return None, None
        try:
            return self._tw.audit_channel_messages(self.lang, call_id, audit)
        except Exception as e:  # noqa: BLE001
            self._err("language.audit", e)
            return None, None

    def lang_close(self, call_id: str | None, **kw: Any) -> None:
        if self.lang is None or call_id is None:
            return
        try:
            self.lang.close_call(call_id, **kw)
        except Exception as e:  # noqa: BLE001
            self._err("language.close", e)

    def image_ref(self, slot: int, ref: str, cam: str, img: Any) -> dict | None:
        """附图引用：当前帧在 trace 里的位置（演示段末帧或第 N 步执行后）与原始帧 sha256（与 trace 同一算法）。"""
        if not self.enabled:
            return None
        try:
            if self.steps == 0:
                phase, idx = "demo", (self.demo_frames if self.demo_frames is not None else None)
            else:
                phase, idx = "exec", self.steps
            return {"slot": int(slot), "ref": ref, "phase": phase, "frame_idx": idx, "cam": cam,
                    "raw_sha256": self._tw.image_sha256(img), "sources": None, "transform": None,
                    "encoded_sha256": None}
        except Exception as e:  # noqa: BLE001
            self._err("language.image_ref", e)
            return None

    def begin_chunk(self, call_id: str | None) -> None:
        """之后执行的步都来自 ``call_id`` 这次调用，块内序号从 0 重新计。"""
        self.src_call = call_id
        self.chunk_pos = 0

    def _link(self) -> dict:
        if self.src_call is None:
            return {}
        out = {"source_call_id": self.src_call, "chunk_index": self.chunk_pos}
        self.chunk_pos += 1
        return out

    # -- 执行步 --
    def step(self, action, front, wrist, state, *, subgoal, terminated, truncated, status) -> None:
        """有效观测步（C4）：执行后的画面、状态、实际交给环境的动作与当步子目标。"""
        if not self.enabled:
            return
        try:
            a = self._copy(action)
            self.steps += 1
            self.observed += 1
            self.actions.append(a)
            self.w.log_step(step=self.steps, front=front, wrist=wrist, state=state, action=a, subgoal=subgoal,
                            terminated=bool(terminated), truncated=bool(truncated), status=status, **self._link())
        except Exception as e:  # noqa: BLE001
            self._err("step", e)

    def missing(self, action, reason: str, subgoal=None) -> None:
        """缺观测步（C8）：保留步号、动作与原因。"""
        if not self.enabled:
            return
        try:
            a = self._copy(action)
            self.steps += 1
            self.actions.append(a)
            self.w.log_missing_step(step=self.steps, action=a, reason=str(reason)[:300], subgoal=subgoal,
                                    **self._link())
        except Exception as e:  # noqa: BLE001
            self._err("missing_step", e)

    def step_exception(self, action, exc: BaseException, before: int | None, after: int | None, *,
                       logged: int = 0, subgoal=None) -> None:
        """``session.step`` 抛异常后：按环境会话 ``steps`` 的增量补记缺观测步（与结果行 ``exec_steps`` 同口径）。

        ``StepCapReached``（strict-cap）不进环境、不计步，只标 ``cap_hit``；拿不到会话计步时其余异常按 1 步计。"""
        if not self.enabled:
            return
        try:
            name = type(exc).__name__
            if name == "StepCapReached":
                self.cap_hit = True
            if before is not None and after is not None:
                extra = int(after) - int(before) - int(logged)
            else:
                extra = 0 if name == "StepCapReached" else 1
            for _ in range(max(0, extra)):
                self.missing(action, f"step_exception: {name}: {exc}", subgoal)
        except Exception as e:  # noqa: BLE001
            self._err("step_exception", e)

    # -- 收尾 --
    # 第三阶段：arrays.npz 不再由本类直接 np.savez；TraceWriter 逐步收集 exec_action／exec_state，close 时经
    # trace_writer.merge_write_npz 合并写盘（与同目录录像器先后任意都不覆盖；分目录时各写各的），end.arrays 为摘要。

    def close(self, status: str | None, *, cap_hit: bool = False, **extra: Any) -> None:
        """按 C2、C3、C8 收尾：strict-cap 记 ``timeout``；无演示帧的 error 局记 ``no_frame``。"""
        if not self.enabled:
            return
        try:
            self.cap_hit = self.cap_hit or bool(cap_hit)
            st = status if status in TRACE_TERMINALS else "error"
            if self.cap_hit:
                st = "timeout"
            no_frame = self.demo_frames is None
            demo = 0 if no_frame else int(self.demo_frames)
            # 官方循环超过 max_steps 时先 break、最后一步不录（只对 FrameSamp+Modulation 这类「第 max_steps+1 步判超时」的路线）
            omitted = int(self.omit_overflow_frame and st == "timeout" and not self.cap_hit and
                          self.steps == self.max_steps + 1)
            frames = 0 if no_frame else demo + 1 + self.observed - omitted
            if self.lang is not None:  # 先关语言账本（悬空调用补 cancelled），其异常也计进本局 hook_errors
                try:
                    self.lang.close()
                except Exception as e:  # noqa: BLE001
                    self._err("language.close", e)
            enc = sorted(self.encodings)
            self.w.close(status=st, terminal_reason=st, side="new", demo_frames=demo, no_frame=no_frame,
                         steps_attempted=self.steps, steps_observed=self.observed, frames_recorded=frames,
                         omitted_timeout_frames=omitted, cap_hit=self.cap_hit,
                         request_encoding=enc[0] if len(enc) == 1 else ("mixed" if enc else None),
                         observer_hook_errors=self.hook_errors, **extra)
        except Exception as e:  # noqa: BLE001
            self._err("close", e)


def run_episode(session, identity: dict, conn_info: dict, recorder=None, *, conn=None,
                max_steps: int = MAX_STEPS, execute_horizon: int = EXECUTE_HORIZON,
                reset_retries: int = RESET_RETRIES, record_frames: bool | None = None) -> dict:
    """跑一局；返回 {status, task_success, steps, error, decisions, timing, server_meta, protocol}。

    conn：测试注入的假连接（需有 metadata、call(msg)->(reply, raw, rep_raw)、close()）；为 None
    时按 conn_info={"host","port"} 建 WSPolicyConn。session 由调用方构造，本函数不 close。
    """
    rec = recorder if recorder is not None else _NullRecorder()
    # env_client.EnvSession 持有同一 recorder 时，原始帧、exec_action 与 reset/run 阶段已由环境侧记录，
    # 本函数只记策略侧（消息指纹、state、model_action、决策事件），避免重复。
    env_records = recorder is not None and getattr(session, "recorder", None) is recorder
    if record_frames is None:
        record_frames = not env_records
    task = identity["task"]
    hb = hard_bound(max_steps, execute_horizon)
    t_start = time.monotonic()
    timing: dict[str, Any] = {"reset_env_s": None, "infer_ms": [], "rtt_ms": [], "step_env_s": 0.0,
                              "observe_ms": [], "connect_s": None}
    out = {"status": "error", "task_success": False, "steps": 0, "error": None, "decisions": 0,
           "infra": False, "infra_reason": None,
           "hard_bound": hb, "timing": timing, "server_meta": None,
           "protocol": {"messages": 0, "frames_sent": 0, "sha_mismatch": 0}}
    proto = out["protocol"]
    frame_idx = {"front": 0, "wrist": 0}
    own_conn = conn is None
    # S4：逐局轨迹（无落点时 enabled=False，以下记录调用全是空操作，客户端行为与 BASE 相同）。
    # 轨迹 max_steps 记客户端循环的真实步数上界 hard_bound×execute_horizon（1300 步时 84×16=1344）：决策用尽的
    # timeout 局正常执行到该上界、每步都录，不存在「第 max_steps+1 步不录」；--max-steps 另记 end.episode_max_steps。
    trace = PolicyTrace("smvla/new", identity, conn_info or {}, recorder, max_steps=hb * int(execute_horizon),
                        recorder_has_actions=recorder_writes_arrays(recorder), omit_overflow_frame=False)

    def record_frames_(frames, tag):
        if not record_frames or not frames:
            return None
        idx = {}
        for cam in (CAM_FRONT, CAM_WRIST):
            idx[cam] = rec.add_frames(cam, np.stack([fr[cam] for fr in frames]), tag=tag)
        return {k: [v[0], v[-1]] if v else [] for k, v in idx.items()}

    last_audit: list[Any] = [None]  # 最近一次回包里 pop 出的服务外壳审计块

    def call(kind: str, msg: dict, extra: dict | None = None) -> dict:
        t0 = time.monotonic()
        reply, raw, rep_raw = conn.call(msg)
        rtt = (time.monotonic() - t0) * 1000.0
        proto["messages"] += 1
        ev = {"kind": "msg", "type": kind, "send_sha256": sha256_bytes(raw), "send_bytes": len(raw),
              "recv_sha256": sha256_bytes(rep_raw), "recv_bytes": len(rep_raw), "rtt_ms": round(rtt, 3)}
        if extra:
            ev.update(extra)
        rec.add_event(ev)
        if kind == "infer":
            timing["rtt_ms"].append(round(rtt, 3))
        if not isinstance(reply, dict):
            raise ProtocolError(f"{kind} 回包不是 dict：{type(reply)}")
        # 第三阶段：服务外壳审计键先 pop 掉（只进语言账本，不进动作与后续校验）
        last_audit[0] = reply.pop(AUDIT_KEY, None)
        if "error" in reply:
            raise ServerError(f"server 报错（{kind}）：{reply['error']}")
        if reply.get("req_sha") != sha256_bytes(raw):
            proto["sha_mismatch"] += 1
            raise ProtocolError(f"{kind} 回包 req_sha 与发出字节不符")
        return reply

    def observe(frames, tag):
        if not frames:
            return
        idx = record_frames_(frames, tag)
        payload = [{CAM_FRONT: fr[CAM_FRONT], CAM_WRIST: fr[CAM_WRIST]} for fr in frames]
        reply = call("observe", {"observe": {"frames": payload}}, {"n_frames": len(frames), "tag": tag,
                                                                   "frame_idx": idx})
        sent = [{CAM_FRONT: frame_sha(fr[CAM_FRONT]), CAM_WRIST: frame_sha(fr[CAM_WRIST])} for fr in frames]
        if reply.get("n") != len(frames) or reply.get("frame_sha") != sent:
            proto["sha_mismatch"] += 1
            raise ProtocolError(f"observe 回包帧指纹与发出不符（发 {len(frames)} 帧，回 {reply.get('n')}）")
        proto["frames_sent"] += len(frames)
        timing["observe_ms"].append(round(float(reply.get("observe_time_ms", 0.0)), 3))
        trace.note_frames(frames)

    status = None
    error = None
    steps = 0
    try:
        # ---- reset（旧：pool.reset → SimEnvService.reset）----
        if not env_records:
            rec.set_phase("reset")
        t0 = time.monotonic()
        r = reset_session(session, task, retries=reset_retries)
        timing["reset_env_s"] = round(time.monotonic() - t0, 3)
        if not env_records:
            rec.set_phase("run")
        if not r.get("ok"):
            status, error = "error", f"reset 失败：{r}"[:2000]
            m = _marker(r.get("reason"), ENV_RESET_INFRA_MARKERS)
            if m:
                out.update(infra=True, infra_reason=f"env_reset:{m}")
            rec.add_event({"kind": "reset_failed", "reason": r.get("reason")})
        else:
            rec.add_event({"kind": "reset_ok", "attempts": r["attempts"], "n_frames": len(r["frames"]),
                           "instruction": r["instruction"]})
            if trace.enabled:  # C2：全部 reset 帧（演示 + 初始帧）与对应 8 维状态
                trace.demo([fr[CAM_FRONT] for fr in r["frames"]], [fr[CAM_WRIST] for fr in r["frames"]],
                           r["states"], r["instruction"])
            if own_conn:
                t0 = time.monotonic()
                conn = WSPolicyConn(conn_info["host"], conn_info["port"])
                timing["connect_s"] = round(time.monotonic() - t0, 3)
            out["server_meta"] = getattr(conn, "metadata", None)
            reply = call("reset", {"reset": {"episode_key": episode_key(identity)}})
            rec.add_event({"kind": "server_rng", "rng": reply.get("rng")})
            instruction = r["instruction"]
            observe(r["frames"], "reset")
            cur_state = np.asarray(r["states"][-1], dtype=np.float32)
            for i, s in enumerate(r["states"]):
                rec.add_array("reset_state", np.asarray(s, dtype=np.float32), step=i - len(r["states"]) + 1)

            success = False
            ended = False
            decisions = 0
            while decisions < hb:
                decisions += 1
                if success:  # 旧：act = [g for g in active if not success[g]]; if not act: break
                    break
                rec.add_array("state", cur_state, step=steps)
                trace.logical_request("infer", instruction, cur_state)
                # 第三阶段语言账本：每次决策一个 action_model 调用，in 消息在发送前落盘
                cid = trace.lang_open("action_model", params={"decision": decisions - 1})
                trace.lang_msg(cid, dir="in", role="fields", text={"instruction": instruction})
                try:
                    reply = call("infer", {"infer": {"instruction": instruction, "state": cur_state}},
                                 {"decision": decisions - 1, "step": steps, "state_sha": array_sha(cur_state)})
                    if reply.get("recv_state_sha") != array_sha(cur_state) or \
                            reply.get("recv_instruction_sha") != sha256_bytes(instruction.encode("utf-8")):
                        proto["sha_mismatch"] += 1
                        raise ProtocolError("infer 回包的状态/指令指纹与发出不符")
                except BaseException:
                    trace.lang_close(cid, status="error")
                    raise
                final_text, final_trunc = trace.lang_audit(cid, last_audit[0])
                trace.lang_msg(cid, dir="out", role="assistant", text=reply.get("subtask"))
                trace.lang_close(cid, status="reply", parsed=reply.get("subtask"), server_final_text=final_text,
                                 server_truncated=final_trunc)
                trace.begin_chunk(cid)
                timing["infer_ms"].append(round(float(reply["infer_ms"]), 3))
                actions_full = np.asarray(reply["actions_full"])
                rec.add_array("model_action", actions_full, step=steps)
                chunk = np.asarray(reply["actions"])[:execute_horizon]
                if not np.array_equal(chunk, actions_full[:execute_horizon]):
                    raise ProtocolError("infer 回包 actions 与 actions_full[:16] 不符")
                rec.add_event({"kind": "decision", "decision": decisions - 1, "step": steps,
                               "subtask": reply.get("subtask"), "infer_ms": round(float(reply["infer_ms"]), 3),
                               "model_action_sha": array_sha(actions_full)})

                trace.response(actions_full)
                subtask = reply.get("subtask")  # 当前子目标（C4、C7）：本次决策回包的子任务文本
                subgoal = subtask if subtask is None or isinstance(subtask, str) else str(subtask)

                exec_step = [steps]
                last_act = [None]  # S4：本块最近一行交给环境的动作（异常步补记用）
                obs_seen = [0]

                def on_exec(act8, _s=exec_step, _last=last_act):
                    if not env_records:
                        rec.add_array("exec_action", act8, step=_s[0])
                    _s[0] += 1
                    _last[0] = act8

                def on_obs(_i, front, wrist, state, terminated, truncated, *, status=None, error=None,
                           _last=last_act, _seen=obs_seen, _sg=subgoal):
                    _seen[0] += 1
                    if front is None:
                        why = f"env_status_error: {error}" if status == "error" else "obs_none"
                        trace.missing(_last[0], why, _sg)
                    else:
                        trace.step(_last[0], front, wrist, state, subgoal=_sg, terminated=terminated,
                                   truncated=truncated, status=status)

                sess_before = getattr(session, "steps", None) if trace.enabled else None
                t0 = time.monotonic()
                try:
                    res = step_chunk(session, chunk, on_exec=on_exec, on_obs=on_obs if trace.enabled else None)
                except Exception as e:  # 旧：InProcSimPool.step 捕获为 {"error": f"step_exc: {e}"}
                    res = {"error": f"step_exc: {e}"}
                    trace.step_exception(last_act[0], e, sess_before, getattr(session, "steps", None),
                                         logged=obs_seen[0], subgoal=subgoal)
                timing["step_env_s"] += time.monotonic() - t0
                if "error" in res:
                    status, error, ended = "error", str(res.get("error")), True
                    m = _marker(error, INFRA_MARKERS)
                    if m:
                        out.update(infra=True, infra_reason=f"env_step:{m}")
                    rec.add_event({"kind": "step_exc", "error": error})
                    break
                observe(res["frames"], "step")
                if res.get("states"):
                    cur_state = np.asarray(res["states"][-1], dtype=np.float32)
                success = bool(res.get("success", False))
                steps += int(res.get("consumed", 0) or 0)
                rec.add_event({"kind": "chunk_done", "decision": decisions - 1, "consumed": res["consumed"],
                               "steps": steps, "status": res["status"], "done": res["done"]})
                if res.get("done", False):
                    status = str(res.get("status") or ("success" if success else "fail"))
                    error = res.get("error_message")
                    ended = True
                    break
            out["decisions"] = decisions
            if not ended:
                status = "success" if success else "timeout"
                error = None if success else timeout_error(hb)
    except Exception as e:
        # 旧：evaluate_manifest 捕获 run_group 异常，error 取 traceback 末 2000 字符，steps 保留
        status = "error"
        tb = traceback.format_exc()
        error = tb[-2000:]
        if isinstance(e, ProtocolError):
            proto["broken"] = True
        reason = classify_exception(e, tb)
        if reason:
            out.update(infra=True, infra_reason=reason)
    finally:
        if own_conn and conn is not None:
            conn.close()
    if status not in FINAL_STATUSES + ("error",):
        error = f"未知终态 {status}；{error}"
        status = "error"
    timing["step_env_s"] = round(timing["step_env_s"], 3)
    timing["episode_s"] = round(time.monotonic() - t_start, 3)
    out.update(status=status, task_success=status == "success", steps=steps, error=error)
    rec.add_event({"kind": "episode_end", "status": status, "steps": steps, "error": error,
                   "decisions": out["decisions"]})
    if trace.enabled:  # C2、C3、C8：strict-cap 以会话 cap_hit 为准记 timeout（与 env_client._classify 同口径）
        trace.close(status, cap_hit=bool(getattr(session, "cap_hit", False)), decisions=out["decisions"],
                    episode_max_steps=int(max_steps), step_bound=hb * int(execute_horizon),
                    session_steps=getattr(session, "steps", None))
        out["trace_path"] = str(trace.path)
    return out
