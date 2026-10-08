#!/usr/bin/env python3
"""v7.5eval 新接口的 MME-VLA 客户端（0929-v7.5eval-restructure-plan.md §2.2、§3.2 第 1 步 3、第 3 步）。

逐行照抄旧官方客户端（MME-VLA ``ecf086c`` ``examples/robomme/{utils.py,env_runner.py,eval.py}`` 与官方历史成绩
客户端 ``927c56d`` ``eval.py::evaluate_manifest``）的循环语义，不 import 子模块的 ``examples/``：

* 每局新建 websocket（``openpi_client`` 的 ``MMEVLAWebsocketClientPolicy``），开局 ``client.reset()``；
* reset 返回的全部演示帧 + 初始帧进缓存，``exec_start_idx = len(image_buffer) - 1``；
* 动作计划为空 → ``add_buffer(pack_buffer(front 缓存, state 缓存, exec_start_idx))`` → ``infer`` → 取前 16 个 →
  清缓存；每执行一步 ``count += 1``，``count > 1300`` 即 timeout（步数 1301），且这一判断先于终态判断；
* 终态映射：``success/fail/timeout`` 原样，其他（如 ``ongoing``/``unknown``）记 ``error`` + ``success_flag=<值>``；
  整局任何异常记 ``error`` + ``<异常类>: <消息>``，步数取异常前最后一次 ``count``。
  ``env.step`` 抛异常时照 ``EnvRunner.step`` 返回 ``(None,)*3, True, "error"``，随后 ``add_observation`` 对 None
  调 ``.copy()`` 抛 ``AttributeError``——旧官方就是这样落成 error 的，这里原样保留；``env.step`` 不抛异常但返回
  ``obs=None`` 时，旧官方在 try 之外 ``obs["front_rgb_list"]`` 抛 ``TypeError``、``count`` 不加，这里也原样保留。

纯函数层（``pack_state``、``pack_buffer``、``EpisodeState``、``pre_traj_from_reset``、``EnvRunnerShim``、
``run_loop``、``evaluate_one``）不碰网络与仿真，单测用合成观测直接断言消息序列。

另有两个子命令（只在跑通核对时用）：``relay`` 起一个逐消息透明的 websocket 中继并记每条消息 sha256；
``transport-check`` 比对客户端事件与中继日志，输出 ``TRANSPORT=PASS frames=<n> mismatch=0``。
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
# 本目录只挂在 sys.path 末尾，防止同目录模块遮蔽标准库
sys.path[:] = [p for p in sys.path if p and str(_Path(p).resolve()) != _HERE] + [_HERE]

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Tuple  # noqa: E402

import numpy as np  # noqa: E402

MAX_STEPS = 1300
OBS_HORIZON = 16
#: 服务外壳回包审计键（冻结说明第五节；与 trace_writer.AUDIT_KEY 相同）
AUDIT_KEY = "_sgeval_audit"
NORMAL = ("success", "fail", "timeout")
INFRA_MARKERS = ("RecorderError", "svulkan2", "EXCLUSIVE", "Vulkan", "vk::", "out of memory", "RESOURCE_EXHAUSTED",
                 "CUDA_ERROR", "ConnectionClosed", "ConnectionRefused", "InvalidStatus", "Connection reset")


def sha(arr: Any) -> str:
    """数组字节的 sha256（C 连续）；bytes/str 直接取。"""
    if isinstance(arr, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(arr)).hexdigest()
    if isinstance(arr, str):
        return hashlib.sha256(arr.encode("utf-8")).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ── 照抄 examples/robomme/env_runner.py ────────────────────────────────────


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space（照抄 env_runner.pack_state）
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


def pre_traj_from_reset(obs: dict, info: dict) -> dict[str, Any]:
    """照抄 ``EnvRunner.get_init_obs`` 的 reset 之后部分（reset 本身由 EnvSession 做）。"""
    if isinstance(info["task_goal"], list):
        task_goal = info["task_goal"][0]
    else:
        task_goal = info["task_goal"]
    images = obs["front_rgb_list"]
    wrist_images = obs["wrist_rgb_list"]
    states = [pack_state(joint_state, gripper_state) for joint_state, gripper_state in
              zip(obs["joint_state_list"], obs["gripper_state_list"])]
    return {"images": images, "wrist_images": wrist_images, "states": states, "task_goal": task_goal}


class EnvRunnerShim:
    """照抄 ``EnvRunner.step``：``self.env.step`` 换成 ``step_fn``（EnvSession.step），其余逐行相同。"""

    def __init__(self, step_fn: Callable[[Any], tuple]):
        self._step_fn = step_fn
        self.info: dict | None = None
        self.last_exception: BaseException | None = None

    def step(self, action: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], bool, str]:
        try:
            obs, _, terminated, truncated, self.info = self._step_fn(action)
        except Exception as e:
            if type(e).__name__ == "RecorderError":  # 录制器故障不是环境错误：上抛，整局记 error + infra（新接口专有）
                raise
            print(f"Error: {e}")
            self.last_exception = e
            return (None, None, None), True, "error"

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        joint_state = obs["joint_state_list"][-1]
        gripper_state = obs["gripper_state_list"][-1]
        state = pack_state(joint_state, gripper_state)

        outcome = self.info.get("status", "unknown")
        stop = terminated or truncated

        return (img, wrist_img, state), stop, outcome


# ── 照抄 examples/robomme/utils.py ────────────────────────────────────────


def pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    image_output = np.stack(image_buffer, axis=0).astype(np.uint8)[:, None]
    state_output = np.stack(state_buffer, axis=0).astype(np.float32)
    return {
        "images": image_output,
        "state": state_output,
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }


class EpisodeState:
    def __init__(self):
        self.image_buffer = []
        self.wrist_image_buffer = []
        self.state_buffer = []
        self.action_plan = collections.deque()
        self.count = 0
        self.exec_start_idx = 0

    def add_observation(self, img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray):
        self.image_buffer.append(img.copy())
        self.wrist_image_buffer.append(wrist_img.copy())
        self.state_buffer.append(state.copy())

    def clear_buffers(self):
        self.image_buffer.clear()
        self.wrist_image_buffer.clear()
        self.state_buffer.clear()
        self.exec_start_idx = 0

    def get_current_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]


# ── 照抄 eval.py::EpisodeEvaluator（去掉子目标预测与视频，官方评估时它们都是空操作）─────────


class _Progress:
    """记录 ``last_steps``（旧官方 ``self.last_steps``），供异常时取步数。"""

    def __init__(self):
        self.last_steps = 0
        self.decisions = 0


def run_loop(client, runner: EnvRunnerShim, reset_fn: Callable[[], dict], progress: _Progress, *,
             max_steps: int = MAX_STEPS, obs_horizon: int = OBS_HORIZON,
             on_decision: Callable[[int, np.ndarray], None] | None = None) -> str:
    """``eval_each_episode`` 的循环：返回 success_flag（异常直接上抛，由 ``evaluate_one`` 映射）。

    ``reset_fn`` 在 ``client.reset()`` 之后调用（与旧官方「先连 server、reset 策略，再 reset 环境」同序），
    返回 ``pre_traj``。"""
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)

    epstate = EpisodeState()
    pre_traj = reset_fn()
    task_goal = pre_traj["task_goal"]
    epstate.image_buffer.extend(pre_traj["images"])
    epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
    epstate.state_buffer.extend(pre_traj["states"])
    epstate.exec_start_idx = len(epstate.image_buffer) - 1

    img, wrist_img, robot_state = epstate.get_current_obs()
    prompt = task_goal
    success_flag = "unknown"

    while True:
        if not epstate.action_plan:
            resp = client.add_buffer(pack_buffer(
                epstate.image_buffer,
                epstate.state_buffer,
                epstate.exec_start_idx,
            ))
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.1)
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": robot_state,
                "prompt": prompt,
            }
            actions = client.infer(element)["actions"]
            if on_decision is not None:
                on_decision(progress.decisions, actions)
            progress.decisions += 1
            action_chunk = actions[:obs_horizon]
            epstate.action_plan.extend(action_chunk)
            epstate.clear_buffers()

        action = epstate.action_plan.popleft()
        obs, stop_flag, success_flag = runner.step(action)
        epstate.count += 1

        progress.last_steps = epstate.count
        if epstate.count > max_steps:
            success_flag = "timeout"
            break

        img, wrist_img, robot_state = obs

        epstate.add_observation(img, wrist_img, robot_state)

        if stop_flag:
            break

    return success_flag


def classify_infra(error: str | None, runner: EnvRunnerShim | None) -> str | None:
    """判断 error 是否属于基础设施故障（可重试原身份，计入重试额度）；只影响 ``infra`` 标记，不改 status。"""
    texts = [error or ""]
    if runner is not None and runner.last_exception is not None:
        texts.append(f"{type(runner.last_exception).__name__}: {runner.last_exception}")
    for text in texts:
        for marker in INFRA_MARKERS:
            if marker in text:
                return marker
    return None


def evaluate_one(client_factory: Callable[[], Any], step_fn: Callable[[Any], tuple], reset_fn: Callable[[], dict], *,
                 max_steps: int = MAX_STEPS, obs_horizon: int = OBS_HORIZON,
                 on_decision: Callable[[int, np.ndarray], None] | None = None) -> dict:
    """照抄 ``evaluate_manifest`` 单局部分的终态映射。client 连接也在 try 里（旧官方同样）。"""
    progress = _Progress()
    runner = EnvRunnerShim(step_fn)
    error = None
    client = None
    try:
        client = client_factory()
        success_flag = run_loop(client, runner, reset_fn, progress, max_steps=max_steps, obs_horizon=obs_horizon,
                                on_decision=on_decision)
    except Exception as e:  # noqa: BLE001 与旧官方同样整局兜底
        print(f"Error evaluating episode: {e}")
        success_flag, error = "error", f"{type(e).__name__}: {e}"
    finally:
        if client is not None:
            try:
                client._ws.close()
            except Exception:  # noqa: BLE001
                pass
    status = success_flag if success_flag in NORMAL else "error"
    if status == "error" and error is None:
        error = f"success_flag={success_flag}"
    infra = classify_infra(error, runner) if status == "error" else None
    env_exc = runner.last_exception
    return {"status": status, "task_success": status == "success", "steps": progress.last_steps, "error": error,
            "decisions": progress.decisions, "infra": infra is not None, "infra_reason": infra,
            "env_exception": None if env_exc is None else f"{type(env_exc).__name__}: {env_exc}"[:800]}


# ── 带记录与测速的 websocket 客户端 ────────────────────────────────────────


def payload_digest(obj: dict) -> dict:
    """出站消息里的逐帧／逐数组 sha256（客户端与中继两侧用同一函数）。"""
    out: dict[str, Any] = {}
    if obj.get("reset", False):
        out["kind"] = "reset"
    elif obj.get("add_buffer", False):
        out["kind"] = "add_buffer"
        imgs = np.asarray(obj["images"])
        out["frames"] = [sha(imgs[i, 0]) for i in range(imgs.shape[0])]
        out["states"] = [sha(np.asarray(obj["state"])[i]) for i in range(len(obj["state"]))]
        out["exec_start_idx"] = int(obj["exec_start_idx"])
        out["shape"] = list(imgs.shape)
    else:
        out["kind"] = "infer"
        out["image"] = sha(obj["observation/image"])
        out["wrist"] = sha(obj["observation/wrist_image"])
        out["state"] = sha(obj["observation/state"])
        out["prompt"] = obj.get("prompt")
    return out


def response_digest(obj: Any) -> dict:
    """入站消息摘要：infer 回复记 actions 的 sha256、dtype、shape。"""
    if not isinstance(obj, dict):
        return {"kind": "other"}
    if "actions" in obj:
        a = np.asarray(obj["actions"])
        return {"kind": "actions", "actions": sha(a), "dtype": a.dtype.str, "shape": list(a.shape),
                "infer_time_ms": float(obj.get("infer_time_ms", float("nan")))}
    if obj.get("reset_finished"):
        return {"kind": "reset_finished", "server_ms": float(obj.get("reset_time_ms", 0.0))}
    if obj.get("add_buffer_finished"):
        return {"kind": "add_buffer_finished", "server_ms": float(obj.get("add_buffer_time_ms", 0.0))}
    return {"kind": "metadata"}


def make_recording_client(host: str, port: int, recorder, timing: dict):
    """``MMEVLAWebsocketClientPolicy`` 的子类：收发逻辑与父类逐行相同（pack→send→recv→str 即报错→unpackb），
    只在两侧加 sha256 记录与计时。连接参数与父类 ``_wait_for_server`` 相同。"""
    import websockets.sync.client
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy

    class RecordingClient(MMEVLAWebsocketClientPolicy):
        def __init__(self):
            self._seq = 0
            # S4：可选的原始字节观察钩子 raw_hook(obj, sent_bytes, recv_bytes)，每次往返成功后调用（只读，异常吞掉）
            self._raw_hook = None
            self._last_audit = None  # 第三阶段：最近一次回包里 pop 出的服务外壳审计块
            super().__init__(host, port)

        def _wait_for_server(self):
            t0 = time.perf_counter()
            while True:
                try:
                    headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                    conn = websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None, additional_headers=headers,
                        ping_timeout=600, open_timeout=60, close_timeout=60
                    )
                    raw = conn.recv()
                    metadata = msgpack_numpy.unpackb(raw)
                    timing["connect_s"] = time.perf_counter() - t0
                    self._event({"kind": "ws_recv", "seq": -1, "msg": "metadata", "sha": sha(raw), "len": len(raw)})
                    return conn, metadata
                except ConnectionRefusedError:
                    time.sleep(5)

        def _event(self, ev: dict) -> None:
            if recorder is not None:
                recorder.add_event(ev)

        def _roundtrip(self, obj: dict) -> dict:
            t0 = time.perf_counter()
            data = self._packer.pack(obj)
            t1 = time.perf_counter()
            self._ws.send(data)
            response = self._ws.recv()
            t2 = time.perf_counter()
            if isinstance(response, str):
                self._event({"kind": "ws_recv", "seq": self._seq, "msg": "error_text", "sha": sha(response)})
                raise RuntimeError(f"Error in inference server:\n{response}")
            out = msgpack_numpy.unpackb(response)
            # 第三阶段：服务外壳审计键在回包交给官方循环（进而交给环境）之前 pop 掉，另存给语言账本
            self._last_audit = out.pop(AUDIT_KEY, None) if isinstance(out, dict) else None
            t3 = time.perf_counter()
            dig = payload_digest(obj)
            rdig = response_digest(out)
            server_ms = rdig.get("server_ms", rdig.get("infer_time_ms", 0.0))
            self._event({"kind": "ws_send", "seq": self._seq, "sha": sha(data), "len": len(data), "payload": dig})
            self._event({"kind": "ws_recv", "seq": self._seq, "sha": sha(response), "len": len(response),
                         "payload": rdig, "pack_s": t1 - t0, "rtt_s": t2 - t1, "unpack_s": t3 - t2})
            per = timing.setdefault("per_msg", [])
            per.append({"seq": self._seq, "kind": dig["kind"], "pack_s": t1 - t0, "rtt_s": t2 - t1,
                        "unpack_s": t3 - t2, "server_ms": server_ms, "bytes": len(data)})
            self._seq += 1
            hook = self._raw_hook
            if hook is not None:
                try:
                    hook(obj, data, response)
                except Exception:  # noqa: BLE001 观察钩子不得影响收发
                    pass
            return out

        def infer(self, obs):  # noqa: D401
            return self._roundtrip(obs)

        def reset(self):
            return self._roundtrip({"reset": True})

        def add_buffer(self, buffer):
            return self._roundtrip(buffer)

    return RecordingClient()


def summarize_timing(timing: dict) -> dict:
    """把逐消息计时汇总成 首次/第 2、3 次/稳态均值/P95（推理只算 infer 消息）。"""
    per = timing.pop("per_msg", [])
    out = {k: v for k, v in timing.items()}
    for kind in ("reset", "add_buffer", "infer"):
        rows = [r for r in per if r["kind"] == kind]
        if not rows:
            continue
        rtt = np.array([r["rtt_s"] for r in rows])
        srv = np.array([r["server_ms"] for r in rows]) / 1000.0
        pack = np.array([r["pack_s"] for r in rows])
        unpack = np.array([r["unpack_s"] for r in rows])
        d = {"n": len(rows), "rtt_first_s": float(rtt[0]), "rtt_mean_s": float(rtt.mean()),
             "rtt_p95_s": float(np.percentile(rtt, 95)), "server_mean_s": float(srv.mean()),
             "network_mean_s": float((rtt - srv).mean()), "pack_mean_s": float(pack.mean()),
             "unpack_mean_s": float(unpack.mean()), "bytes_mean": float(np.mean([r["bytes"] for r in rows]))}
        if kind == "infer":
            d["server_first_s"] = float(srv[0])
            d["server_2_s"] = float(srv[1]) if len(srv) > 1 else None
            d["server_3_s"] = float(srv[2]) if len(srv) > 2 else None
            steady = srv[3:] if len(srv) > 3 else srv
            d["server_steady_mean_s"] = float(steady.mean())
            d["server_steady_p95_s"] = float(np.percentile(steady, 95))
        out[kind] = d
    return out


def _load_sibling(name: str):
    """按文件路径加载本目录下的模块（与 env_client／groundsg_client 的 load_sibling 同名注册，已加载则复用）。"""
    if name in sys.modules:
        return sys.modules[name]
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, Path(_HERE) / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TracedClient:
    """S4：包住 MME-VLA websocket 客户端（或替身）的 ``reset``／``add_buffer``／``infer``，原样转发后记请求与回包（C10）。

    请求记原始 msgpack 帧字节的 sha256（经 ``RecordingClient._raw_hook`` 拿到实际发出的字节；替身没有该钩子时退回
    ``canonical_bytes``）；``add_buffer`` 另记历史边界行（覆盖的步号区间）；``infer`` 回包记完整动作块。转发的对象与
    返回值不做任何改动，请求失败（抛异常）时不记该行。"""

    def __init__(self, inner: Any, trace: Any):
        self._inner = inner
        self._trace = trace
        self._raw: tuple | None = None
        self._history_from = 0
        if hasattr(inner, "_raw_hook"):
            inner._raw_hook = self._on_raw

    def _on_raw(self, obj, data, response) -> None:
        self._raw = (data, response)

    def _record(self, name: str, obj: Any, out: Any) -> None:
        raw, self._raw = self._raw, None
        tr = self._trace
        tr.raw_request(name, obj, raw[0] if raw is not None else None)
        try:
            if name == "add_buffer":
                n = len(obj["images"]) if isinstance(obj, dict) and "images" in obj else 0
                tr.history(self._history_from, tr.steps,
                           note=f"add_buffer frames={n} exec_start_idx={obj.get('exec_start_idx')}")
                self._history_from = tr.steps
            elif name == "infer":
                tr.response(out["actions"])
        except Exception as e:  # noqa: BLE001
            tr._err(f"client.{name}", e)

    def reset(self):
        self._raw = None
        out = self._inner.reset()
        self._record("reset", {"reset": True}, out)
        return out

    def add_buffer(self, buffer):
        self._raw = None
        out = self._inner.add_buffer(buffer)
        self._record("add_buffer", buffer, out)
        return out

    def _image_refs(self, obs: Any) -> list | None:
        """当前前视、腕部两张帧的引用（``raw_sha256`` 与 trace 同算法，可在 demo 末帧或上一步 step 行里找到）。"""
        if not isinstance(obs, dict):
            return None
        tr = self._trace
        refs = [tr.image_ref(0, "current", "front", obs.get("observation/image")),
                tr.image_ref(1, "wrist", "wrist", obs.get("observation/wrist_image"))]
        return [r for r in refs if r is not None] or None

    def infer(self, obs):
        """第三阶段语言账本：每个推理步一个 ``action_model`` 调用。发送前写 ``in``（``role=fields``：结构化字段原文 +
        两张当前帧引用）；回包后 pop 掉 ``_sgeval_audit``（真实客户端已在 ``_roundtrip`` 里 pop 并存进
        ``_last_audit``），逐通道写分词消息，``close_call`` 记 ``server_final_text``；之后执行的步关联到本调用。"""
        self._raw = None
        tr = self._trace
        cid = tr.lang_open("action_model")
        fields = {k: v for k, v in obs.items() if isinstance(v, (str, int, float, bool)) or v is None} \
            if isinstance(obs, dict) else None
        tr.lang_msg(cid, dir="in", role="fields", text=fields, images=self._image_refs(obs))
        if hasattr(self._inner, "_last_audit"):
            self._inner._last_audit = None
        try:
            out = self._inner.infer(obs)
        except BaseException:
            tr.lang_close(cid, status="error")
            raise
        audit = out.pop(AUDIT_KEY, None) if isinstance(out, dict) else None
        if audit is None:
            audit = getattr(self._inner, "_last_audit", None)
        final_text, final_trunc = tr.lang_audit(cid, audit)
        tr.lang_close(cid, status="reply", server_final_text=final_text, server_truncated=final_trunc)
        tr.begin_chunk(cid)
        self._record("infer", obs, out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def traced_step_fn(session: Any, trace: Any) -> Callable[[Any], tuple]:
    """S4：包一层 ``session.step`` 拿完整五元组记逐步轨迹；动作对象原样交给环境，返回值与异常原样透出。

    有观测：最后一帧前视、腕部画面与 ``pack_state`` 状态（与 ``EnvRunnerShim.step`` 同算法），子目标 ``None``（C7）；
    ``obs is None``：缺观测步；抛异常：按会话 ``steps`` 增量记缺观测步（即 ``EnvRunnerShim`` 返回 ``(None,)*3`` 的步），
    ``StepCapReached`` 不进环境、不计步、只标 ``cap_hit``。"""

    def step(action):
        before = getattr(session, "steps", None)
        try:
            out = session.step(action)
        except Exception as e:
            trace.step_exception(action, e, before, getattr(session, "steps", None))
            raise
        try:
            obs, _r, terminated, truncated, info = out
            status = info.get("status", "unknown") if isinstance(info, dict) else None
            if obs is None:
                trace.missing(action, "obs_none")
            else:
                try:
                    img = obs["front_rgb_list"][-1]
                    wrist = obs["wrist_rgb_list"][-1]
                    state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
                except Exception as e:  # noqa: BLE001 观测不可读：照常透出，由官方循环自己报错
                    trace._err("step.obs", e)
                    trace.missing(action, f"obs_unreadable: {type(e).__name__}: {e}")
                else:
                    trace.step(action, img, wrist, state, subgoal=None, terminated=terminated,
                               truncated=truncated, status=status)
        except Exception as e:  # noqa: BLE001
            trace._err("step", e)
        return out

    return step


def run_episode(session, identity: dict, conn_info: dict, recorder) -> dict:
    """env_client 调用入口：一局 FrameSamp+Modulation。``session`` 为已 build 的 EnvSession。

    S4：有轨迹落点（``groundsg_client.trace_location``）时写 ``trace.jsonl``（route ``perceptual-framesamp-modul/new``）：``reset_fn`` 外包一层记
    演示（C2），``session.step`` 外包一层记逐步（C4、C8），客户端外包一层记请求与回包（C10），收尾按 C2、C3、C8。
    无落点时三层都不包，与 BASE 行为相同。"""
    timing: dict[str, Any] = {}
    max_steps = int(conn_info.get("max_steps", MAX_STEPS))
    sm = _load_sibling("smvla_client")  # 共用的 PolicyTrace
    trace = sm.PolicyTrace("perceptual-framesamp-modul/new", identity, conn_info, recorder, max_steps=max_steps,
                           recorder_has_actions=sm.recorder_writes_arrays(recorder) and
                           getattr(session, "recorder", None) is recorder,
                           omit_overflow_frame=True)

    def client_factory():
        client = make_recording_client(conn_info.get("host", "127.0.0.1"), int(conn_info["port"]), recorder, timing)
        return TracedClient(client, trace) if trace.enabled else client

    def reset_fn():
        obs, info = session.reset()
        pre = pre_traj_from_reset(obs, info)
        if trace.enabled:
            trace.demo(pre["images"], pre["wrist_images"], pre["states"], pre["task_goal"])
        return pre

    def on_decision(idx: int, actions: np.ndarray) -> None:
        if recorder is not None:
            recorder.add_array("model_action", np.array(actions, copy=True), step=idx)
            a = np.asarray(actions)
            recorder.add_event({"kind": "model_action", "decision": idx, "sha": sha(a),
                                "row_sha": [sha(a[i]) for i in range(a.shape[0])],
                                "dtype": np.asarray(actions).dtype.str, "shape": list(np.asarray(actions).shape),
                                "exec_n": min(OBS_HORIZON, len(actions)), "env_step": session.steps})

    step_fn = traced_step_fn(session, trace) if trace.enabled else session.step
    t0 = time.perf_counter()
    res = evaluate_one(client_factory, step_fn, reset_fn, max_steps=max_steps, on_decision=on_decision)
    timing["episode_s"] = time.perf_counter() - t0
    res["timing"] = summarize_timing(timing)
    if trace.enabled:
        trace.close(res["status"], cap_hit=bool(getattr(session, "cap_hit", False)), decisions=res["decisions"],
                    session_steps=getattr(session, "steps", None))
        res["trace_path"] = str(trace.path)
    return res


# ── 中继与传输核对（跑通核对用）──────────────────────────────────────────


def cmd_relay(args) -> int:
    """逐消息透明中继：客户端连 listen 端口，中继连 upstream；每条消息原样转发（不开压缩、max_size=None），
    日志记方向、序号、sha256、长度与解包后的逐帧摘要。"""
    import asyncio

    import websockets
    import websockets.asyncio.client as wsc
    import websockets.asyncio.server as wss
    from openpi_client import msgpack_numpy

    log = open(args.log, "a", encoding="utf-8")
    conn_id = [0]

    def write(rec):
        log.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        log.flush()

    async def handler(client_ws):
        cid = conn_id[0]
        conn_id[0] += 1
        async with wsc.connect(f"ws://127.0.0.1:{args.upstream}", compression=None, max_size=None,
                               ping_timeout=600, open_timeout=60, close_timeout=60) as up:
            seq = {"c2s": 0, "s2c": -1}

            async def pump(src, dst, direction):
                try:
                    async for msg in src:
                        rec = {"conn": cid, "dir": direction, "seq": seq[direction], "sha": sha(msg),
                               "len": len(msg), "t": time.time()}
                        if not isinstance(msg, str):
                            try:
                                obj = msgpack_numpy.unpackb(msg)
                                rec["payload"] = (payload_digest(obj) if direction == "c2s" else response_digest(obj))
                            except Exception as e:  # noqa: BLE001
                                rec["decode_error"] = repr(e)
                        write(rec)
                        seq[direction] += 1
                        await dst.send(msg)
                except websockets.ConnectionClosed:
                    pass
                finally:
                    await dst.close()

            await asyncio.gather(pump(client_ws, up, "c2s"), pump(up, client_ws, "s2c"))

    async def main():
        async with wss.serve(handler, "127.0.0.1", args.listen, compression=None, max_size=None):
            print(f"RELAY_READY listen={args.listen} upstream={args.upstream}", flush=True)
            await asyncio.Future()

    asyncio.run(main())
    return 0


def transport_check(events: list[dict], relay: list[dict]) -> dict:
    """客户端事件 vs 中继日志：逐消息 sha 相同、顺序相同；env 帧／状态 == 出站载荷；回复动作 == 执行动作。"""
    mismatch, frames, notes = 0, 0, []
    sends = [e for e in events if e.get("kind") == "ws_send"]
    recvs = [e for e in events if e.get("kind") == "ws_recv"]
    r_c2s = [r for r in relay if r["dir"] == "c2s"]
    r_s2c = [r for r in relay if r["dir"] == "s2c"]
    if len(sends) != len(r_c2s) or len(recvs) != len(r_s2c):
        mismatch += 1
        notes.append(f"count client send/recv={len(sends)}/{len(recvs)} relay={len(r_c2s)}/{len(r_s2c)}")
    for a, b in zip(sends, r_c2s):
        if a["sha"] != b["sha"] or a["len"] != b["len"] or a["payload"] != b.get("payload"):
            mismatch += 1
            notes.append(f"c2s seq={a['seq']}")
    for a, b in zip(recvs, r_s2c):
        if a["sha"] != b["sha"]:
            mismatch += 1
            notes.append(f"s2c seq={a['seq']}")
    # env 帧与出站载荷逐帧比：add_buffer 的 frames 必须等于上一次决策以来 env 交出的 front 帧序列
    env_front: list[str] = []
    env_wrist: list[str] = []
    env_state: list[str] = []
    goal = None
    exec_actions: list[str] = []
    model_rows: list[str] = []
    for e in events:
        k = e.get("kind")
        if k == "env_reset":
            env_front, env_wrist, env_state = list(e["front"]), list(e["wrist"]), list(e["state8"])
            goal = e.get("task_goal")
        elif k == "env_step" and e.get("front") is not None:
            env_front.append(e["front"][-1])
            env_wrist.append(e["wrist"][-1])
            env_state.append(e["state8"])
        elif k == "env_step_action":
            exec_actions.append(e["sha"])
        elif k == "model_action":
            model_rows.extend(e.get("row_sha", [])[: e.get("exec_n", OBS_HORIZON)])
        elif k == "ws_send":
            p = e["payload"]
            if p["kind"] == "add_buffer":
                frames += len(p["frames"])
                if p["frames"] != env_front or p["states"] != env_state:
                    mismatch += 1
                    notes.append(f"add_buffer seq={e['seq']} frames/state != env")
                env_front_last, env_wrist_last, env_state_last = env_front[-1], env_wrist[-1], env_state[-1]
                env_front, env_wrist, env_state = [], [], []
            elif p["kind"] == "infer":
                frames += 2
                if (p["image"], p["wrist"], p["state"]) != (env_front_last, env_wrist_last, env_state_last):
                    mismatch += 1
                    notes.append(f"infer seq={e['seq']} obs != env")
                if goal is not None and p["prompt"] != goal:
                    mismatch += 1
                    notes.append(f"infer seq={e['seq']} prompt")
    n = min(len(exec_actions), len(model_rows))
    bad_exec = sum(1 for i in range(n) if exec_actions[i] != model_rows[i])
    if bad_exec or len(exec_actions) > len(model_rows):
        mismatch += bad_exec + max(0, len(exec_actions) - len(model_rows))
        notes.append(f"exec_action != model rows: {bad_exec}")
    return {"mismatch": mismatch, "frames": frames, "messages": len(sends) + len(recvs),
            "exec_actions": len(exec_actions), "notes": notes[:20]}


def cmd_transport_check(args) -> int:
    events = [json.loads(line) for line in Path(args.events).read_text().splitlines() if line.strip()]
    relay = [json.loads(line) for line in Path(args.relay_log).read_text().splitlines() if line.strip()]
    if args.conn is not None:
        relay = [r for r in relay if r["conn"] == args.conn]
    res = transport_check(events, relay)
    ok = res["mismatch"] == 0 and res["frames"] > 0
    print(f"TRANSPORT={'PASS' if ok else 'FAIL'} frames={res['frames']} mismatch={res['mismatch']} "
          f"messages={res['messages']} exec_actions={res['exec_actions']}")
    for note in res["notes"]:
        print(f"  note: {note}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v7.5eval FrameSamp+Modulation 客户端辅助子命令")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("relay")
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", type=int, required=True)
    p.add_argument("--log", required=True)
    p.set_defaults(func=cmd_relay)
    p = sub.add_parser("transport-check")
    p.add_argument("--events", required=True)
    p.add_argument("--relay-log", required=True)
    p.add_argument("--conn", type=int, default=None)
    p.set_defaults(func=cmd_transport_check)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
