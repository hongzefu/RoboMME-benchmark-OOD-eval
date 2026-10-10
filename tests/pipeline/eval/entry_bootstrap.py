"""Bootstrap script for the test subprocess: runs the official ``scripts/evaluation.py`` or
``scripts/evaluation_ood.py`` (benchmark submodule) unchanged as ``__main__``.

Only in this subprocess's memory, the four ``BenchmarkEnvBuilder`` methods used by the entry are replaced with a CPU
stub environment and ``imageio.mimsave`` is replaced with an event logger. No source file is modified and nothing is
persisted (P2 exemption for temporary in-process test stubs); the entry script itself runs verbatim.

Usage: python entry_bootstrap.py <entry path> <scenario json> <event log jsonl>
Scenario json: {"tasks": [...], "episodes": n, "plans": {"<task>/<ep>": ["ongoing", "success" | "fail" | "error", ...]}}
"""
from __future__ import annotations

import importlib
import json
import runpy
import sys

import numpy as np

entry, scen_path, log_path = sys.argv[1:4]
with open(scen_path, encoding="utf-8") as fh:
    SCEN = json.load(fh)
LOG = open(log_path, "a", encoding="utf-8")


def emit(**kw):
    LOG.write(json.dumps(kw, ensure_ascii=False) + "\n")
    LOG.flush()


def _obs(n):
    f = np.zeros((8, 8, 3), dtype=np.uint8)
    return {"front_rgb_list": [f] * n, "wrist_rgb_list": [f] * n}


class FakeEnv:
    def __init__(self, task, ep):
        self.task, self.ep = task, ep
        self.plan = list(SCEN["plans"][f"{task}/{ep}"])

    def reset(self):
        emit(kind="reset", task=self.task, ep=self.ep)
        return _obs(2), {"task_goal": [f"goal {self.task} {self.ep}"], "status": "ongoing"}

    def step(self, action):
        status = self.plan.pop(0)
        emit(kind="step", task=self.task, ep=self.ep, status=status, action_shape=list(np.shape(action)))
        if status == "error":
            return None, 0.0, True, False, {"status": "error", "error_message": "IK has no solution (stub)"}
        done = status in ("success", "fail")
        return _obs(1), 0.0, done, False, {"status": status}

    def close(self):
        emit(kind="close", task=self.task, ep=self.ep)


wrapper = ("robomme_ood.env_record_wrapper" if entry.endswith(("evaluation_ood.py", "evaluation_hard.py"))
           else "robomme.env_record_wrapper")
Builder = importlib.import_module(wrapper).BenchmarkEnvBuilder


def _init(self, env_id, dataset, action_space, max_steps, **kw):
    self.env_id = env_id
    emit(kind="builder", env_id=env_id, dataset=dataset, action_space=action_space)


Builder.__init__ = _init
Builder.get_task_list = classmethod(lambda cls: list(SCEN["tasks"]))
Builder.get_episode_num = lambda self: int(SCEN["episodes"])
Builder.make_env_for_episode = lambda self, ep, **kw: FakeEnv(self.env_id, ep)

import imageio  # noqa: E402

imageio.mimsave = lambda path, frames, fps=30: emit(kind="save", path=str(path), frames=len(frames))

sys.argv = [entry]
runpy.run_path(entry, run_name="__main__")
