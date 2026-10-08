"""外层交给 ``Policy.play`` 的环境会话 ``EnvSession``（抽自 ``dev-scripts/gl/seat.py::EnvSession``，拆分方案 §三）。

一局一个实例，由 ``episode.run_episode`` 建、``build()``、交给模型的 ``play``、最后由外层 ``close()``：

* ``build()``：``builder.make_env_for_episode(builder_episode)``，计 1 次 reset 预算；
* ``reset()``：原样返回 ``env.reset()`` 的 ``(obs, info)``，计 1 次 reset 预算；返回后把演示段帧、状态与事件写进录制器；
* ``step(action)``：action 原样交给 ``env.step``，记录实际交出去的数组（``exec_action``）与返回的当前帧、状态、终态；
  ``strict_cap`` 时第 ``max_steps+1`` 次调用不进环境，置 ``cap_hit`` 并抛 ``StepCapReached``；
* ``close()``：只由外层调用（模型侧不得关环境），幂等。

预算：``ledger``（任意带 ``claim(what)`` 方法的对象，超额抛 ``ResetBudgetExhausted`` 或带 ``budget_exhausted``
属性的异常）与 ``claim_reset(what)``（GL 席位的持久账本回调）在每次实际 build／reset 之前各领一次；两者都不给时不计额度，
只计 ``reset_calls``。只读属性：``env``、``info``、``steps``、``cap_hit``、``reset_calls``、``recorder``。
钩子：``first_step_cb``（本局第一次 ``step`` 调用时触发一次，结束首推理计时）、``progress_cb(steps)``（每
``progress_every`` 步一次）。
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

#: 两个评估数据集
OOD = "ood"
HARD_VERIFY = "hard-verify"
DATASETS = (OOD, HARD_VERIFY)


class RecorderError(RuntimeError):
    """录制器（含磁盘写满）出错：归为基础设施故障（infra=True），不当成环境错误。"""


class StepCapReached(RuntimeError):
    """``strict_cap``：已执行 ``max_steps`` 步仍未结束，第 ``max_steps+1`` 次 ``step`` 不进入环境（外层收为 timeout）。"""


class ResetBudgetExhausted(RuntimeError):
    """reset 额度（build 与 reset 各算一次）已用尽。"""

    budget_exhausted = True


def sha(arr: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def state8(joint, gripper):
    """与旧官方 ``pack_state`` 同式的 8 维状态（7 关节 + 夹爪第一维，float32）。"""
    import numpy as np

    return np.concatenate([np.asarray(joint), np.asarray(gripper)[:1]], axis=0, dtype=np.float32)


def _scalar(x):
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return None


class _GuardedRecorder:
    """录制器代理：任何录制调用抛出的异常都转成 RecorderError（保留原因）。"""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if not callable(fn):
            return fn

        def call(*a, **k):
            try:
                return fn(*a, **k)
            except RecorderError:
                raise
            except Exception as e:  # noqa: BLE001
                raise RecorderError(f"{name}: {type(e).__name__}: {e}") from e

        return call


class NullRecorder:
    """不录制时的空录制器（接口同 ``record.recorder.EpisodeRecorder``）。"""

    def set_phase(self, phase):
        pass

    def add_frames(self, stream, frames, *, tag=""):
        return []

    def add_array(self, name, arr, *, step=None):
        pass

    def add_event(self, event):
        pass

    def close(self, summary):
        return {"RECORDER_VERIFY": "SKIP"}


class EnvSession:
    """一局环境。``builder`` 由外层按 ``(task, dataset)`` 缓存后传入（与官方「每任务一个 builder」一致）。"""

    def __init__(self, task: str, builder_episode: int, *, dataset: str = OOD, max_steps: int | None = None,
                 recorder=None, builder=None, progress_cb: Callable[[int], None] | None = None,
                 progress_every: int = 16, step_cap: int | None = None, ledger: Any = None,
                 claim_reset: Callable[[str], None] | None = None, budget_claim: Callable[[str], None] | None = None,
                 first_step_cb: Callable[[], None] | None = None):
        if dataset not in DATASETS:
            raise ValueError(f"dataset={dataset!r} 不是 {DATASETS} 之一")
        self.task = task
        self.dataset = dataset
        self.builder_episode = int(builder_episode)
        # max_steps 只在需要自建 builder 时用（构造参数）；不逐局传给 make_env_for_episode
        self.max_steps = None if max_steps is None else int(max_steps)
        self.step_cap = None if step_cap is None else int(step_cap)
        self.ledger = ledger
        self.claim_reset = claim_reset
        self.budget_claim = budget_claim
        self.first_step_cb = first_step_cb
        self.progress_cb = progress_cb
        self.progress_every = max(1, int(progress_every))
        self.recorder = recorder if recorder is not None else NullRecorder()
        # 内部录制一律经代理；对外仍暴露原录制器（SimpleMemVLA 客户端靠 ``session.recorder is recorder`` 判断）
        self._rec = _GuardedRecorder(self.recorder)
        self._builder = builder
        self._env = None
        self._info: Any = None
        self._steps = 0
        self._cap_hit = False
        self._reset_calls = 0
        self._closed = False
        self.budget_exhausted = False
        self.timing: dict[str, Any] = {}
        self._step_s: list[float] = []
        self._rec_s = 0.0
        self.task_goal = None

    # ── 只读属性 ─────────────────────────────────────────────────────
    @property
    def env(self):
        return self._env

    @property
    def info(self):
        """最近一次 reset／step 返回的 info（GroundSG Oracle 每步读其中的 ``grounded_subgoal_online``）。"""
        return self._info

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def cap_hit(self) -> bool:
        return self._cap_hit

    @property
    def reset_calls(self) -> int:
        return self._reset_calls

    @property
    def builder(self):
        if self._builder is None:
            from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

            if self.max_steps is None:
                raise ValueError("EnvSession 自建 builder 必须给 max_steps（步数上限由入口按数据集给出）")
            t0 = time.perf_counter()
            self._builder = BenchmarkEnvBuilder(env_id=self.task, dataset=self.dataset, action_space="joint_angle",
                                                max_steps=self.max_steps)
            self.timing["builder_init_s"] = time.perf_counter() - t0
        return self._builder

    # ── 预算 ─────────────────────────────────────────────────────────
    def _claim(self, what: str) -> None:
        """每次实际 build／reset 前领一次额度，领到后才计入 reset_calls。"""
        try:
            if self.ledger is not None:
                self.ledger.claim(what)
            if self.claim_reset is not None:
                self.claim_reset(what)
        except Exception as e:  # noqa: BLE001 额度耗尽（可能来自另一份模块副本，按属性识别）
            if isinstance(e, ResetBudgetExhausted) or getattr(e, "budget_exhausted", False):
                self.budget_exhausted = True
            raise
        if self.budget_claim is not None:
            self.budget_claim(what)
        self._reset_calls += 1

    # ── 环境 ─────────────────────────────────────────────────────────
    def build(self) -> None:
        """``make_env_for_episode(builder_episode)``：步数上限由 builder 构造参数 ``max_steps`` 决定（与官方相同）。"""
        if self._env is not None:
            return
        if self._closed:
            raise RuntimeError("EnvSession 已 close，不能再 build")
        self._claim("build")
        self._rec.set_phase("reset")
        t0 = time.perf_counter()
        self._env = self.builder.make_env_for_episode(self.builder_episode)
        self.timing["env_build_s"] = time.perf_counter() - t0

    def reset(self):
        """原样返回 ``env.reset()`` 的 ``(obs, info)``；返回后记录演示段帧与数组（录制器此时仍在 reset 阶段，只入队）。"""
        import numpy as np

        self.build()
        self._claim("reset")
        self._rec.set_phase("reset")
        t0 = time.perf_counter()
        obs, info = self._env.reset()
        self.timing["reset_s"] = time.perf_counter() - t0
        self._info = info
        t1 = time.perf_counter()
        goal = info.get("task_goal") if isinstance(info, dict) else None
        self.task_goal = goal[0] if isinstance(goal, list) else goal
        front = np.stack(obs["front_rgb_list"])
        wrist = np.stack(obs["wrist_rgb_list"])
        self._rec.add_frames("front", front, tag="reset")
        self._rec.add_frames("wrist", wrist, tag="reset")
        for key in ("joint_state_list", "gripper_state_list", "eef_state_list"):
            if key in obs and obs[key] is not None and len(obs[key]):
                self._rec.add_array(f"reset_{key[:-5]}", np.stack([np.asarray(x) for x in obs[key]]))
        s8 = [sha(state8(j, g)) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])]
        self._rec.add_event({"kind": "env_reset", "task": self.task, "builder_episode": self.builder_episode,
                             "frames": int(front.shape[0]), "demo_frames": int(front.shape[0]) - 1,
                             "front": [sha(f) for f in front], "wrist": [sha(f) for f in wrist], "state8": s8,
                             "task_goal": self.task_goal,
                             "status": info.get("status") if isinstance(info, dict) else None,
                             "reset_s": self.timing["reset_s"]})
        self._rec_s += time.perf_counter() - t1
        self.timing["demo_frames"] = int(front.shape[0]) - 1
        self._rec.set_phase("run")
        return obs, info

    def step(self, action):
        """action 原样交给 ``env.step``；记录交出去的数组与返回的当前帧、状态、终态。"""
        import numpy as np

        if self.first_step_cb is not None:
            cb, self.first_step_cb = self.first_step_cb, None
            cb()
        if self.step_cap is not None and self._steps >= self.step_cap:
            self._cap_hit = True
            self._rec.add_event({"kind": "step_cap_reached", "step": self._steps, "cap": self.step_cap})
            raise StepCapReached(f"STEP_CAP exec_steps={self._steps} cap={self.step_cap}")
        t0 = time.perf_counter()
        a = np.array(action, copy=True)
        self._rec.add_array("exec_action", a, step=self._steps)
        self._rec.add_event({"kind": "env_step_action", "step": self._steps, "sha": sha(a), "dtype": a.dtype.str,
                             "shape": list(a.shape)})
        t1 = time.perf_counter()
        try:
            out = self._env.step(action)
        except Exception as e:
            self._step_s.append(time.perf_counter() - t1)
            self._rec.add_event({"kind": "env_step_exception", "step": self._steps,
                                 "error": f"{type(e).__name__}: {e}"[:800]})
            self._steps += 1
            raise
        t2 = time.perf_counter()
        self._step_s.append(t2 - t1)
        obs, reward, terminated, truncated, info = out
        self._info = info
        status = info.get("status") if isinstance(info, dict) else None
        ev: dict[str, Any] = {"kind": "env_step", "step": self._steps, "terminated": bool(terminated),
                              "truncated": bool(truncated), "status": status, "reward": _scalar(reward)}
        if obs is None:
            ev.update(front=None, wrist=None, obs_none=True)
        else:
            front = np.stack(obs["front_rgb_list"])
            wrist = np.stack(obs["wrist_rgb_list"])
            self._rec.add_frames("front", front, tag=f"step{self._steps}")
            self._rec.add_frames("wrist", wrist, tag=f"step{self._steps}")
            joint, grip = obs["joint_state_list"][-1], obs["gripper_state_list"][-1]
            self._rec.add_array("joint_state", np.asarray(joint), step=self._steps)
            self._rec.add_array("gripper_state", np.asarray(grip), step=self._steps)
            if obs.get("eef_state_list"):
                self._rec.add_array("eef_state", np.asarray(obs["eef_state_list"][-1]), step=self._steps)
            ev.update(front=[sha(f) for f in front], wrist=[sha(f) for f in wrist], state8=sha(state8(joint, grip)))
        self._rec.add_event(ev)
        self._steps += 1
        self._rec_s += (t1 - t0) + (time.perf_counter() - t2)
        if self.progress_cb is not None and self._steps % self.progress_every == 0:
            self.progress_cb(self._steps)
        return out

    def close(self) -> None:
        """关环境并汇总逐步计时；只由外层调用，幂等。"""
        import numpy as np

        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            t0 = time.perf_counter()
            try:
                self._env.close()
            finally:
                self.timing["env_close_s"] = time.perf_counter() - t0
                self._env = None
        if self._step_s:
            s = np.array(self._step_s)
            self.timing.update(step_n=int(s.size), step_total_s=float(s.sum()), step_mean_s=float(s.mean()),
                               step_p50_s=float(np.percentile(s, 50)), step_p95_s=float(np.percentile(s, 95)),
                               step_first_s=float(s[0]))
        self.timing["record_overhead_s"] = self._rec_s
