"""测试子进程的引导脚本：以 ``__main__`` 原样执行 ``scripts/evaluation.py`` 或 ``scripts/evaluation_hard.py``。

只在本子进程内存里把入口用到的 ``BenchmarkEnvBuilder`` 四个方法换成 CPU 替身环境、把 ``imageio.mimsave`` 换成记事件，
不改任何源文件、不落盘（P2 对测试进程内临时替身的豁免）；入口脚本本身一字不改地执行。

用法：python entry_bootstrap.py <入口路径> <场景 json> <事件日志 jsonl>
场景 json：{"tasks": [...], "episodes": n, "plans": {"<task>/<ep>": ["ongoing", "success" | "fail" | "error", ...]}}
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
            return None, 0.0, True, False, {"status": "error", "error_message": "IK 无解（替身）"}
        done = status in ("success", "fail")
        return _obs(1), 0.0, done, False, {"status": status}

    def close(self):
        emit(kind="close", task=self.task, ep=self.ep)


wrapper = "robomme_hard.env_record_wrapper" if entry.endswith("evaluation_hard.py") else "robomme.env_record_wrapper"
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
