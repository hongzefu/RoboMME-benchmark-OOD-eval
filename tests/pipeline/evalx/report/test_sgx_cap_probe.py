"""C13-SG-CAP-PROBE：步数到顶的三种循环口径（CPU 假环境，不建真实仿真）。期望值手写：
mme-vla 1301／loop_count，range 1300／loop_exit，strict 1600／strict_cap；环境先截断时如实报 env_truncated。"""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from tests._support.loaders import load_script


def C():
    return load_script("eval-official/cap_probe.py")


class FakeEnv:
    """仿 DemonstrationWrapper：非演示步计数 ≥ ``cap`` 时 truncated；可在第 ``terminate_at`` 步 terminated。"""

    def __init__(self, cap: int, terminate_at: int | None = None):
        self.cap = cap
        self.terminate_at = terminate_at
        self.steps = 0
        self.actions: list[np.ndarray] = []
        self.closed = False

    def _obs(self):
        j = np.linspace(0.1, 0.7, 7) + 0.0
        return {"joint_state_list": [j * 0, j], "gripper_state_list": [np.zeros(2), np.array([0.04, 0.04])]}

    def reset(self):
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action))
        term = self.terminate_at is not None and self.steps >= self.terminate_at
        trunc = self.steps >= self.cap
        return self._obs(), 0.0, np.array([term]), np.array([trunc]), {}

    def close(self):
        self.closed = True


class FakeBuilder:
    def __init__(self, max_steps: int, *, cap_override: int | None = None, terminate_at: int | None = None):
        self.max_steps_without_demonstration = max_steps + 2 if cap_override is None else cap_override
        self.terminate_at = terminate_at
        self.env = None
        self.episodes: list[int] = []

    def make_env_for_episode(self, ep):
        self.episodes.append(ep)
        self.env = FakeEnv(self.max_steps_without_demonstration, self.terminate_at)
        return self.env


def _hold(o):
    return C().hold_action(o)


@pytest.mark.parametrize("loop,max_steps,steps,reason", [
    ("mme-vla", 1300, 1301, "loop_count"),
    ("range", 1300, 1300, "loop_exit"),
    ("strict", 1600, 1600, "strict_cap"),
])
def test_loops_against_builder_cap(loop, max_steps, steps, reason):
    c = C()
    b = FakeBuilder(max_steps)
    env = b.make_env_for_episode(0)
    obs, _ = env.reset()
    res = c.run_loop(env, obs, loop, max_steps, c.hold_action)
    assert (res["exec_steps"], res["terminal_reason"]) == (steps, reason)
    assert env.steps == steps  # strict：第 1601 次不进环境
    line = c.verdict_line(dataset="hard-verify", max_steps=max_steps, builder_cap=b.max_steps_without_demonstration,
                          loop=loop, res=res, source="run", official=False)
    assert line.startswith(f"STEP_CAP=PASS dataset=hard-verify max_steps={max_steps} builder_cap={max_steps + 2} "
                           f"loop={loop} exec_steps={steps} terminal_reason={reason}")
    assert "builder_cap_ok=1" in line


def test_env_truncates_first_is_reported_honestly():
    c = C()
    b = FakeBuilder(1300, cap_override=1290)  # 环境上限比循环口径小
    env = b.make_env_for_episode(0)
    obs, _ = env.reset()
    res = c.run_loop(env, obs, "mme-vla", 1300, c.hold_action)
    assert (res["exec_steps"], res["terminal_reason"]) == (1290, "env_truncated")
    line = c.verdict_line(dataset="hard-verify", max_steps=1300, builder_cap=1290, loop="mme-vla", res=res, source="run",
                          official=False)
    assert line.startswith("STEP_CAP=FAIL ") and "exec_steps=1290 terminal_reason=env_truncated" in line
    assert "builder_cap_ok=0" in line
    env2 = FakeBuilder(1600, terminate_at=37).make_env_for_episode(0)
    obs, _ = env2.reset()
    assert c.run_loop(env2, obs, "strict", 1600, c.hold_action)["terminal_reason"] == "env_terminated"


def test_derive_range_from_mme_vla_prefix():
    c = C()
    flags = [(False, False)] * 1300 + [(False, False)]
    assert c.derive_range(flags, 1300) == {"exec_steps": 1300, "terminal_reason": "loop_exit"}
    flags2 = [(False, False)] * 99 + [(False, True)] + [(False, False)] * 1201
    assert c.derive_range(flags2, 1300) == {"exec_steps": 100, "terminal_reason": "env_truncated"}
    assert c.derive_range([(False, False)] * 10, 1300)["terminal_reason"] == "prefix_too_short"


def test_hold_action_keeps_joints_and_gripper():
    c = C()
    obs = FakeEnv(5)._obs()
    a = c.hold_action(obs)
    assert a.shape == (8,) and np.array_equal(a[:7], np.linspace(0.1, 0.7, 7)) and a[7] == 1.0
    obs["gripper_state_list"][-1] = np.array([0.001, 0.001])
    assert c.hold_action(obs)[7] == -1.0
    assert c.hold_action(obs, "0.5")[7] == 0.5


def test_main_mme_vla_with_derive_range(capsys):
    c = C()
    b = FakeBuilder(1300)
    rc = c.main(["--dataset", "hard-verify", "--max-steps", "1300", "--task", "VideoUnmask", "--episode", "0",
                 "--loop", "mme-vla", "--derive-range"], builder_factory=lambda a: b)
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and b.episodes == [0] and b.env.closed
    assert out[0].startswith("CAP_PROBE builder.max_steps_without_demonstration=1302 max_steps=1300 dataset=hard-verify")
    run = [x for x in out if x.startswith("STEP_CAP=")]
    assert run[0].startswith("STEP_CAP=PASS dataset=hard-verify max_steps=1300 builder_cap=1302 loop=mme-vla "
                             "exec_steps=1301 terminal_reason=loop_count") and run[0].endswith("source=run")
    assert run[1].startswith("STEP_CAP=PASS dataset=hard-verify max_steps=1300 builder_cap=1302 loop=range "
                             "exec_steps=1300 terminal_reason=loop_exit") and run[1].endswith("source=prefix")
    # 所有动作都是「保持当前关节位置」
    assert all(np.array_equal(x[:7], np.linspace(0.1, 0.7, 7)) for x in b.env.actions)


def test_main_strict_official_label_and_rejects_bad_combo(capsys):
    c = C()
    b = FakeBuilder(1600)
    seen: list[argparse.Namespace] = []

    def fac(a):
        seen.append(a)
        return b

    rc = c.main(["--official", "--max-steps", "1600", "--task", "VideoUnmask", "--episode", "3", "--loop", "strict"],
                builder_factory=fac)
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("STEP_CAP=")][0]
    assert rc == 0 and seen[0].official and line.endswith("official=1") and "dataset=test " in line
    with pytest.raises(SystemExit):
        c.main(["--max-steps", "1300", "--task", "T", "--episode", "0", "--loop", "range", "--derive-range"],
               builder_factory=fac)
