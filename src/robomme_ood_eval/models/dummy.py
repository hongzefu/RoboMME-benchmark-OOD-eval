"""随机动作 dummy 模型（冒烟与单测用；动作口径照官方 ``scripts/evaluation.py::DummyModel``）。

无服务端：``load`` 不做事；``reset(spec)`` 只按 ``policy_seed`` 重设本局随机数（不碰环境）；``play`` 用
``session.reset()``／``session.step()`` 跑到环境终止或步数上限，同时写本局 ``trace.jsonl``（路线 ``dummy/new``），
供官方版式重绘与产物核对使用。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from robomme_ood_eval.policy import Policy
from robomme_ood_eval.record.trace_writer import TraceWriter

#: 官方 DummyModel 的基准关节动作（7 关节 + 夹爪）
BASE_ACTION = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0], dtype=np.float32)
NOISE_STD = 0.01


def _state8(obs: dict, i: int = -1) -> np.ndarray:
    joint = np.asarray(obs["joint_state_list"][i])
    grip = np.asarray(obs["gripper_state_list"][i])
    return np.concatenate([joint, grip[:1]], axis=0, dtype=np.float32)


class DummyPolicy(Policy):
    model = "dummy"

    def __init__(self, policy_seed: int, **cfg):
        super().__init__(policy_seed, **cfg)
        self._rng = np.random.default_rng(self.policy_seed)

    def reset(self, spec) -> None:
        """局前：只重设本局随机数（同一 policy_seed 下每局动作序列相同），不碰环境、无服务端消息。"""
        super().reset(spec)
        self._rng = np.random.default_rng(self.policy_seed)

    def predict(self) -> np.ndarray:
        noise = self._rng.normal(0.0, NOISE_STD, BASE_ACTION.shape)
        noise[..., -1:] = 0.0  # 夹爪不加噪
        return BASE_ACTION + noise

    def play(self, session, spec, recorder) -> dict:
        trace = TraceWriter(Path(spec.out_dir) / "trace.jsonl", route="dummy/new", identity=spec.identity(),
                            max_steps=spec.max_steps, policy_seed=spec.policy_seed,
                            effective_cap=spec.max_steps if spec.strict_cap else None)
        status, error, steps, info = "error", None, 0, {}
        demo_frames = 0
        try:
            obs, info = session.reset()
            goal = info.get("task_goal") if isinstance(info, dict) else None
            goal = goal[0] if isinstance(goal, list) else goal
            n = len(obs["front_rgb_list"])
            demo_frames = n - 1
            trace.log_demo(list(obs["front_rgb_list"]), list(obs["wrist_rgb_list"]),
                           states=[_state8(obs, i) for i in range(n)], texts=[str(goal or "")])
            while True:
                action = self.predict()
                obs, _reward, terminated, truncated, info = session.step(action)
                steps += 1
                st = info.get("status") if isinstance(info, dict) else None
                trace.log_step(step=steps, front=obs["front_rgb_list"][-1], wrist=obs["wrist_rgb_list"][-1],
                               state=_state8(obs), action=action, subgoal=None, terminated=bool(terminated),
                               truncated=bool(truncated), status=st)
                if st == "error":
                    status, error = "error", f"env status=error: {info.get('error_message')}"
                    break
                if terminated or truncated:
                    status = "success" if st == "success" else ("timeout" if st == "timeout" else "fail")
                    break
        except BaseException as e:
            from robomme_ood_eval.session import StepCapReached

            status = "timeout" if isinstance(e, StepCapReached) else "error"
            trace.close(status=status, terminal_reason=status, demo_frames=demo_frames, cap_hit=session.cap_hit,
                        error=f"{type(e).__name__}: {e}"[:400])
            raise
        trace.close(status=status, terminal_reason=status, demo_frames=demo_frames, cap_hit=session.cap_hit)
        return {"status": status, "task_success": int(status == "success"), "steps": steps, "error": error,
                "infra": False, "infra_reason": None, "decisions": steps}
