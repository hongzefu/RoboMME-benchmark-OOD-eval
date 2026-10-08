"""PonderPounce 接入测试的替身（S4）：确定性假环境、EnvSession 替身、进程内假连接、逐帧比对。

全部是 CPU 替身：不建仿真场景、不加载权重、不连外网。假环境的画面、状态、指令只由 ``(task, episode_idx)``
决定，新侧（``pp_client`` + EnvSession 替身）与原侧（vla-eval ``RoboMMEBenchmark``，经假 ``BenchmarkEnvBuilder``）
拿到逐字节相同的原始观测，消息序列的差异只可能来自驱动本身。
"""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
PP_BENCHMARK_LITERAL = "vla_eval.benchmarks.robomme.benchmark:RoboMMEBenchmark"
TASK_GOAL = "pick up the cube"
H = W = 8


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StepCapReached(RuntimeError):
    """与 env_client.StepCapReached 同名的替身（pp_client 按类名识别）。"""


class FakeEnv:
    """确定性假环境：reset 给 ``demo`` 个演示帧 + 1 个初始帧；每步 1 帧；第 ``done_at`` 步给终态。"""

    def __init__(self, task: str, episode_idx: int, *, demo: int = 3, done_at: int | None = None,
                 done_status: str = "success", none_obs_at: int | None = None, truncate_at: int | None = None):
        self.task, self.episode_idx = task, int(episode_idx)
        self.demo, self.done_at, self.done_status = demo, done_at, done_status
        self.none_obs_at, self.truncate_at = none_obs_at, truncate_at
        self.rng_seed = zlib.crc32(f"{task}:{episode_idx}".encode())
        self.t = 0
        self.actions: list[Any] = []
        self.closed = False

    def _frame(self, k: int, cam: int) -> np.ndarray:
        rng = np.random.default_rng([self.rng_seed, k, cam])
        return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    def _joint(self, k: int) -> np.ndarray:
        return np.random.default_rng([self.rng_seed, k, 7]).normal(size=7).astype(np.float64)

    def _grip(self, k: int) -> np.ndarray:
        return np.random.default_rng([self.rng_seed, k, 9]).normal(size=2).astype(np.float64)

    def frames_at(self, k: int) -> dict:
        return {"front": self._frame(k, 0), "wrist": self._frame(k, 1), "joint": self._joint(k), "grip": self._grip(k)}

    def reset(self):
        n = self.demo + 1
        obs = {"front_rgb_list": [self._frame(k, 0) for k in range(n)],
               "wrist_rgb_list": [self._frame(k, 1) for k in range(n)],
               "joint_state_list": [self._joint(k) for k in range(n)],
               "gripper_state_list": [self._grip(k) for k in range(n)],
               "eef_state_list": [np.zeros(7) for _ in range(n)]}
        self.t = 0
        return obs, {"task_goal": [TASK_GOAL], "status": "ongoing"}

    def step(self, action):
        self.actions.append(action)
        self.t += 1
        k = self.demo + self.t
        done = self.done_at is not None and self.t >= self.done_at
        trunc = self.truncate_at is not None and self.t >= self.truncate_at
        status = self.done_status if done else ("timeout" if trunc else "ongoing")
        info = {"status": status, "simple_subgoal_online": f"sg{self.t}", "grounded_subgoal_online": f"gsg{self.t}"}
        if self.none_obs_at is not None and self.t == self.none_obs_at:
            return None, 0.0, False, False, {"status": "error"}
        obs = {"front_rgb_list": [self._frame(k, 0)], "wrist_rgb_list": [self._frame(k, 1)],
               "joint_state_list": [self._joint(k)], "gripper_state_list": [self._grip(k)],
               "eef_state_list": [np.zeros(7)]}
        return obs, 0.0, done, trunc, info

    def close(self):
        self.closed = True


class FakeSession:
    """EnvSession 的最小替身：``reset()`` → ``(obs, info)``；``step`` 原样转发、计步，可选 strict cap。"""

    def __init__(self, env: FakeEnv, *, step_cap: int | None = None):
        self.env = env
        self.step_cap = step_cap
        self.steps = 0
        self.cap_hit = False
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return self.env.reset()

    def step(self, action):
        if self.step_cap is not None and self.steps >= self.step_cap:
            self.cap_hit = True
            raise StepCapReached(f"STEP_CAP exec_steps={self.steps} cap={self.step_cap}")
        out = self.env.step(action)
        self.steps += 1
        return out


def action_for(n: int, dim: int = 10) -> np.ndarray:
    """第 n 次请求的确定性动作块：float32 (1, dim)。"""
    return (np.arange(dim, dtype=np.float32).reshape(1, dim) * np.float32(0.001) + np.float32(n) * np.float32(0.01))


class FakeConn:
    """进程内假连接（接口同 vla_eval.connection.Connection 的用到部分），按调用顺序记录协议帧。

    ``closed_at``：第 i 次 ``act``（从 0 计，含重发）抛 websockets ``ConnectionClosed``。
    ``raise_at``：第 i 次 ``act`` 抛给定异常。"""

    def __init__(self, url: str = "ws://fake", timeout: float = 0.0, *, closed_at=(), raise_at=None):
        self.url, self.timeout = url, timeout
        self.log: list[tuple[str, Any]] = []
        self.closed_at = set(closed_at)
        self.raise_at = dict(raise_at or {})
        self.n_act_calls = 0
        self.n_actions = 0
        self.reconnects = 0
        self.closed = 0

    async def connect(self, *, benchmark=None):
        self.log.append(("hello", {"benchmark": benchmark}))

    async def reconnect(self):
        self.reconnects += 1
        self.log.append(("hello", {"benchmark": "<reconnect>"}))

    async def start_episode(self, config):
        self.log.append(("episode_start", config))

    async def act(self, obs):
        i = self.n_act_calls
        self.n_act_calls += 1
        self.log.append(("observation", obs))
        if i in self.closed_at:
            import websockets.exceptions as wse

            raise wse.ConnectionClosed(None, None)
        if i in self.raise_at:
            raise self.raise_at[i]
        a = {"actions": action_for(self.n_actions)}
        self.n_actions += 1
        self.log.append(("action", a))
        return a

    async def end_episode(self, result):
        self.log.append(("episode_end", result))

    async def close(self):
        self.closed += 1


# ── 逐帧比对 ───────────────────────────────────────────────────────────────


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            out.append(f"{path}: 类型 {type(a).__name__} != {type(b).__name__}")
        elif a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            out.append(f"{path}: 数组 {a.dtype}{a.shape} != {b.dtype}{b.shape} 或字节不同")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: 键 {sorted(a)} != {sorted(b)}")
            return
        for k in a:
            _diff(a[k], b[k], f"{path}.{k}", out)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
        return
    if type(a) is not type(b) or a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def strip_wallclock(frame: tuple[str, Any]) -> tuple[str, Any]:
    """EPISODE_END 的 ``elapsed_sec`` 是墙钟，比对前去掉（只断言它存在且为数）。"""
    t, p = frame
    if t == "episode_end" and isinstance(p, dict):
        assert isinstance(p.get("elapsed_sec"), float), p
        p = {k: v for k, v in p.items() if k != "elapsed_sec"}
    return t, p


def compare_frames(got: list[tuple[str, Any]], want: list[tuple[str, Any]]) -> tuple[int, int, list[str]]:
    """返回 ``(frames_diff, order_diff, notes)``：order_diff 为帧类型序列不同的位置数（含长度差），
    frames_diff 为类型相同但载荷不同的帧数。"""
    notes: list[str] = []
    gt, wt = [t for t, _ in got], [t for t, _ in want]
    order_diff = abs(len(gt) - len(wt)) + sum(1 for x, y in zip(gt, wt) if x != y)
    if order_diff:
        notes.append(f"类型序列不同：got={gt[:12]}… want={wt[:12]}…")
    frames_diff = 0
    for i, (g, w) in enumerate(zip(got, want)):
        if g[0] != w[0]:
            continue
        d: list[str] = []
        _diff(strip_wallclock(g)[1], strip_wallclock(w)[1], f"[{i}]{g[0]}", d)
        if d:
            frames_diff += 1
            notes.extend(d[:3])
    return frames_diff, order_diff, notes
