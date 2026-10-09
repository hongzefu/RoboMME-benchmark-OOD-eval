"""Random-action dummy model (for smoke runs and unit tests; actions follow the official
``scripts/evaluation.py::DummyModel``).

No server: ``load`` does nothing; ``reset(spec)`` only reseeds this episode's RNG from ``policy_seed`` (it never
touches the environment); ``play`` drives ``session.reset()`` / ``session.step()`` until the environment terminates
or the step cap is hit, and writes this episode's ``trace.jsonl`` (route ``dummy/new``) for official-layout
re-rendering and output checks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from robomme_ood_eval.policy import Policy
from robomme_ood_eval.record.trace_writer import TraceWriter

#: base joint action of the official DummyModel (7 joints + gripper)
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
        """Before the episode: only reseed the RNG (same action sequence every episode for one policy_seed); never
        touches the environment and sends no server message."""
        super().reset(spec)
        self._rng = np.random.default_rng(self.policy_seed)

    def predict(self) -> np.ndarray:
        noise = self._rng.normal(0.0, NOISE_STD, BASE_ACTION.shape)
        noise[..., -1:] = 0.0  # no noise on the gripper
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
