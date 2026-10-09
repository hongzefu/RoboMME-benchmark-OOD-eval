"""Test doubles for the outer-interface unit tests: fake builder / fake env (no simulation), fake recorder, and a
fake Policy with configurable behavior.

``EVENTS`` is a global event sequence used to pin the call order (reset before build, play after build, teardown
session.close -> recorder.close -> result.json).
"""
from __future__ import annotations

import time

import numpy as np

from robomme_ood_eval.policy import Policy

EVENTS: list[str] = []
H = W = 256
GOAL = "pick up the red cube"


def frame(i: int, cam: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    f = np.zeros((H, W, 3), np.uint8)
    f[..., 0] = (xx + 7 * i) % 256
    f[..., 1] = (yy + 3 * cam) % 256
    f[..., 2] = 60 + 40 * cam
    f[40:80, (10 + 4 * i) % 200:(50 + 4 * i) % 200 or 200] = (255, 255, 255)
    return f


def obs_of(frames: list[int], joint_scale: float = 0.0) -> dict:
    return {"front_rgb_list": [frame(i, 0) for i in frames], "wrist_rgb_list": [frame(i, 1) for i in frames],
            "joint_state_list": [np.full(7, joint_scale + 0.01 * i, np.float64) for i in frames],
            "gripper_state_list": [np.array([0.04, 0.04], np.float64) for _ in frames]}


class FakeEnv:
    """reset yields 3 frames (2 demo frames + the initial frame); step ``finish_at`` terminates with ``finish_status``; ``None`` never terminates."""

    def __init__(self, episode: int, finish_at: int | None = 4, finish_status: str = "success", demo: int = 2):
        self.episode = episode
        self.finish_at, self.finish_status, self.demo = finish_at, finish_status, demo
        self.n = 0
        self.closed = 0

    def reset(self):
        EVENTS.append("env.reset")
        self.n = 0
        return obs_of(list(range(self.demo + 1))), {"task_goal": [GOAL, "alt"], "status": "ongoing"}

    def step(self, action):
        self.n += 1
        done = self.finish_at is not None and self.n >= self.finish_at
        info = {"status": self.finish_status if done else "ongoing", "grounded_subgoal_online": None}
        return obs_of([self.demo + self.n]), 0.0, done, False, info

    def close(self):
        EVENTS.append("env.close")
        self.closed += 1


class FakeBuilder:
    """``resolve_identity`` returns an identity per dataset; ``make_env_for_episode`` records an event."""

    instances: list["FakeBuilder"] = []

    def __init__(self, task: str, dataset: str, max_steps: int, *, n_episodes: int = 3, fail_build: bool = False,
                 finish_at: int | None = 4, finish_status: str = "success"):
        self.task, self.dataset, self.max_steps = task, dataset, max_steps
        self.n_episodes, self.fail_build = n_episodes, fail_build
        self.finish_at, self.finish_status = finish_at, finish_status
        self.envs: list[FakeEnv] = []
        FakeBuilder.instances.append(self)

    def resolve_identity(self, episode: int) -> dict:
        if not 0 <= episode < self.n_episodes:
            raise IndexError(episode)
        if self.dataset == "hard-verify":
            return {"episode": episode, "tier": "xhard0", "candidate": None, "seed": 500 + episode,
                    "source_dataset": "test", "source_episode": 3 + episode, "spec_sha256": None, "source_run": None}
        return {"episode": episode, "tier": "xhard2", "candidate": 10 + episode, "seed": 9000 + episode,
                "spec_sha256": f"{episode:064x}", "source_run": None}

    def get_episode_num(self) -> int:
        return self.n_episodes

    def make_env_for_episode(self, episode: int):
        EVENTS.append("builder.make_env")
        if self.fail_build:
            raise RuntimeError("vulkan device lost")
        env = FakeEnv(episode, self.finish_at, self.finish_status)
        self.envs.append(env)
        return env


class FakeRecorder:
    def __init__(self, raw, meta):
        EVENTS.append("recorder.init")
        self.raw, self.meta = raw, meta
        self.frames = 0
        self.closed = 0

    def set_phase(self, phase):
        pass

    def add_frames(self, stream, frames, *, tag=""):
        self.frames += len(frames)
        return []

    def add_array(self, name, arr, *, step=None):
        pass

    def add_event(self, event):
        pass

    def close(self, summary):
        EVENTS.append("recorder.close")
        self.closed += 1
        return {"RECORDER_VERIFY": "PASS", "summary": summary}


class FakeServer:
    def __init__(self):
        self.pid, self.port = 4242, 18080
        self.metadata = {"policy_seed": 7}

    def left_line(self) -> str:
        return f"SERVER_LEFT pid={self.pid} port={self.port} metadata=/x/server-metadata-{self.port}.json stop=\"x\""

    def check(self):
        pass

    def stop(self, **k):
        EVENTS.append("server.stop")


class FakePolicy(Policy):
    """``behavior``: success / fail_status / raise / stepcap (keep stepping until StepCapReached propagates) /
    swallow_cap (swallow StepCapReached and return fail) / return_error / bad_output / sleep (sleep ``sleep_s`` then
    return success) / touch_close (illegally call session.close)."""

    model = "fake"

    def __init__(self, policy_seed: int, behavior: str = "success", sleep_s: float = 0.0, with_server: bool = False,
                 **cfg):
        super().__init__(policy_seed, **cfg)
        self.behavior, self.sleep_s = behavior, sleep_s
        self.seen_specs = []
        self.session_seen = None
        if with_server:
            self.server = FakeServer()

    def load(self):
        EVENTS.append("policy.load")

    def reset(self, spec):
        EVENTS.append("policy.reset")
        self.seen_specs.append(spec)

    def play(self, session, spec, recorder):
        EVENTS.append("policy.play")
        self.session_seen = session
        assert session.env is not None, "env must be built before play (M2)"
        assert session.recorder is recorder
        b = self.behavior
        if b == "raise":
            raise ValueError("model blew up")
        if b == "bad_output":
            return {"status": "success"}
        if b == "sleep":
            time.sleep(self.sleep_s)
        session.reset()
        steps = 0
        if b in ("stepcap", "swallow_cap"):
            from robomme_ood_eval.session import StepCapReached

            try:
                while True:
                    session.step(np.zeros(8, np.float32))
                    steps += 1
            except StepCapReached:
                if b == "stepcap":
                    raise
                return {"status": "fail", "task_success": 0, "steps": steps, "error": None, "infra": False,
                        "infra_reason": None}
        while True:
            _o, _r, term, trunc, info = session.step(np.zeros(8, np.float32))
            steps += 1
            if term or trunc:
                break
        if b == "return_error":
            return {"status": "error", "task_success": 0, "steps": steps, "error": "IK failed", "infra": False,
                    "infra_reason": None}
        if b == "fail_status":
            return {"status": "fail", "task_success": 0, "steps": steps, "error": None, "infra": False,
                    "infra_reason": None, "pp_sid_use_index": 0}
        return {"status": "success", "task_success": 1, "steps": steps, "error": None, "infra": False,
                "infra_reason": None, "decisions": 2, "pp_sid_use_index": 1}

    def close(self):
        EVENTS.append("policy.close")
        super().close()
