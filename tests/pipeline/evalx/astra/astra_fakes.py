"""Astra 测试的公共替身：规划、监视、VLA、环境、两个服务端全部替身，零外联、零费用、零 GPU。

被测对象是评估仓的 ``robomme_ood_eval.models.astra.AstraPolicy``（模型侧 4 方法），外层用真实的
``robomme_ood_eval.episode.run_episode``（builder 是真实 ``BenchmarkEnvBuilder`` 的子类，只把
``make_env_for_episode`` 打桩成 ``FakeEnv``；录制器是 ``FakeRecorder``，不出网站视频）。

Astra 上游源码只读引用 ``$SGEVAL_THIRD_PARTY/Astra-on-RoboMME``；未设时取当前检出的 ``third_party``，worktree 里
子模块为空时退回主检出（``git rev-parse --git-common-dir`` 的上一层）；都不在时直接失败（不 skip）。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from tests._support.loaders import REPO

ASTRA_MODULES = ("runner", "core", "api_client", "release_utils", "input_contract", "train_entry")
#: 上游 runner.episode 默认的单局规划次数上限
MAX_PLANNER_CALLS = 24
FAKE_KEY = "sk-test-placeholder-not-a-real-key"
PRICES = {"unit": "usd_per_1m_tokens", "input": 2.0, "cached_input": 0.5, "output": 8.0}


def _main_checkout() -> Path | None:
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(common).parent if common else None


def third_party_dir() -> Path:
    """第三方子模块目录：``SGEVAL_THIRD_PARTY`` > 当前检出 ``third_party``（子模块已初始化时）> 主检出 ``third_party``。"""
    value = os.environ.get("SGEVAL_THIRD_PARTY")
    if value:
        return Path(value)
    here = REPO / "third_party"
    if (here / "Astra-on-RoboMME" / "examples" / "champ" / "runner.py").is_file():
        return here
    main = _main_checkout()
    return (main / "third_party") if main is not None else here


def third_party() -> Path:
    root = third_party_dir() / "Astra-on-RoboMME"
    assert (root / "examples" / "champ" / "runner.py").is_file(), f"Astra 上游源码不在 {root}"
    return root


def print_upstream_digests() -> dict:
    champ = third_party() / "examples" / "champ"
    files = ("run.sh", "runner.py", "api_client.py", "core.py", "input_contract.py", "release_utils.py",
             "prepare_cases.py", "weights.json")
    digests = {}
    for name in files:
        digests[name] = hashlib.sha256((champ / name).read_bytes()).hexdigest()
        print(f"ASTRA_UPSTREAM_READ {champ / name} sha256={digests[name]}")
    return digests


def load_runner():
    from robomme_ood_eval.models import astra

    return astra


def load_guard():
    from robomme_ood_eval.servers import astra_cost_guard

    return astra_cost_guard


@contextlib.contextmanager
def astra_session():
    """导入 Astra 上游模块并在退出时恢复 ``sys.path`` 与 ``sys.modules``，避免污染同一 pytest 会话的其他测试。
    ``SGEVAL_THIRD_PARTY`` 在会话内指向 ``third_party_dir()``（被测模块按它找子模块与环境源）。"""
    saved_path = list(sys.path)
    saved_mods = {name: sys.modules.get(name) for name in (*ASTRA_MODULES, "openpi_client")}
    saved_env = os.environ.get("SGEVAL_THIRD_PARTY")
    os.environ["SGEVAL_THIRD_PARTY"] = str(third_party_dir())
    mod = load_runner()
    try:
        astra = mod.bootstrap(third_party())
        yield mod, astra
    finally:
        sys.path[:] = saved_path
        for name, old in saved_mods.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        if saved_env is None:
            os.environ.pop("SGEVAL_THIRD_PARTY", None)
        else:
            os.environ["SGEVAL_THIRD_PARTY"] = saved_env


class NetCounter:
    """把 ``urllib.request.urlopen`` 换成计数后抛错的替身：``api_calls`` 由它实测。"""

    def __init__(self, raise_exc=None) -> None:
        self.calls = 0
        self.raise_exc = raise_exc

    def __call__(self, *a, **k):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc()
        raise RuntimeError("测试中禁止外联：urlopen 被调用")

    def install(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", self)
        return self


# ── 环境替身 ─────────────────────────────────────────────────────────────

def _frame(value: int) -> np.ndarray:
    return np.full((256, 256, 3), value % 251, dtype=np.uint8)


class FakeEnv:
    """最小 RoboMME 环境：``terminal_step`` 步后 success；``None`` 则永不结束（由循环上限截住）。"""

    def __init__(self, terminal_step: int | None = 40, demo_frames: int = 3, step_error_at: int | None = None) -> None:
        self.t = 0
        self.terminal_step = terminal_step
        self.demo_frames = demo_frames
        self.step_error_at = step_error_at
        self.closed = False
        self.close_calls = 0
        self.steps_taken = 0

    def _obs(self, n: int) -> dict:
        return {
            "front_rgb_list": [_frame(self.t * 7 + i) for i in range(n)],
            "wrist_rgb_list": [_frame(self.t * 11 + i + 1) for i in range(n)],
            "joint_state_list": [np.full(7, self.t + i, dtype=np.float64) for i in range(n)],
            "gripper_state_list": [np.full(2, 0.5, dtype=np.float64) for _ in range(n)],
        }

    def reset(self):
        return self._obs(self.demo_frames + 1), {"task_goal": ["pick up the cube"], "status": "ongoing"}

    def step(self, action):
        self.t += 1
        self.steps_taken += 1
        if self.step_error_at is not None and self.t == self.step_error_at:
            return self._obs(1), 0.0, False, False, {"status": "error", "error_message": "sim crash (fake)"}
        done = self.terminal_step is not None and self.t >= self.terminal_step
        return self._obs(1), 0.0, done, False, {"status": "success" if done else "ongoing"}

    def close(self):
        self.closed = True
        self.close_calls += 1


def recording_builder_cls(env_plan=None):
    """真实 ``BenchmarkEnvBuilder`` 的子类：构造参数照常走真实校验并被记录；``make_env_for_episode`` 打桩。

    ``env_plan``：可调用 ``(builder, episode) -> FakeEnv``，或抛异常模拟基础设施故障。
    """
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    class RecordingBuilder(BenchmarkEnvBuilder):
        constructed: list = []
        make_calls: list = []

        def __init__(self, env_id, **kwargs):
            RecordingBuilder.constructed.append({"env_id": env_id, **kwargs})
            super().__init__(env_id, **kwargs)
            self.envs = []

        def make_env_for_episode(self, episode_idx, *args, **kwargs):
            RecordingBuilder.make_calls.append({"env_id": self.env_id, "episode": episode_idx,
                                                "args": args, "kwargs": kwargs})
            env = env_plan(self, episode_idx) if env_plan else FakeEnv()
            self.envs.append(env)
            return env

    RecordingBuilder.constructed = []
    RecordingBuilder.make_calls = []
    return RecordingBuilder


class FakeRecorder:
    """外层录制器替身（接口同 ``record.recorder.EpisodeRecorder``）：只计数。"""

    instances: list = []

    def __init__(self, raw, meta):
        self.raw, self.meta = Path(raw), meta
        self.frames = {"front": 0, "wrist": 0}
        self.arrays: list = []
        self.events: list = []
        self.summary = None
        FakeRecorder.instances.append(self)

    def set_phase(self, phase):
        pass

    def add_frames(self, stream, frames, *, tag=""):
        self.frames[stream] += len(frames) if getattr(frames, "ndim", 4) == 4 else 1
        return []

    def add_array(self, name, arr, *, step=None):
        self.arrays.append((name, step))

    def add_event(self, event):
        self.events.append(event)

    def close(self, summary):
        self.summary = summary
        return {"RECORDER_VERIFY": "PASS"}


class NullWriter:
    """``imageio.get_writer`` 替身（Astra 自写的 rollout.mp4 不真编码）。"""

    def append_data(self, frame):
        pass

    def close(self):
        pass


# ── 监视器、VLA、规划应答替身 ───────────────────────────────────────────

class FakeMonitor:
    """与 Astra ``Monitor.predict`` 同签名；写 ``input.json`` 与一张图，按序列返回布尔预测（默认恒 False）。"""

    def __init__(self, predictions=None) -> None:
        self.calls = 0
        self.predictions = list(predictions or [])

    def predict(self, task, goal, subgoal, frames, command_start, wrist, out):
        from PIL import Image
        out = Path(out)
        Image.fromarray(np.asarray(frames[-1])).save(out / "0.png")
        (out / "input.json").write_text(json.dumps({"task": task, "goal": goal, "subgoal": subgoal,
                                                    "command_start": command_start, "images": [str(out / "0.png")]}))
        pred = self.predictions[self.calls] if self.calls < len(self.predictions) else False
        # 与上游 Monitor.predict 同：回复原文写 response.json（语言账本读它）
        (out / "response.json").write_text(json.dumps({"text": "true" if pred else "false"}))
        self.calls += 1
        return pred, [0], 0.0


class FakeVLA:
    def __init__(self, fail_at: int | None = None) -> None:
        self.infers = 0
        self.resets = 0
        self.fail_at = fail_at

    def reset(self):
        self.resets += 1

    def infer(self, element):
        self.infers += 1
        assert element["observation/image"].shape == (256, 256, 3)
        if self.fail_at is not None and self.infers == self.fail_at:
            raise ConnectionError("fake VLA websocket dropped")
        return {"actions": np.zeros((16, 8), dtype=np.float32)}


def template_text(champ: Path, task: str) -> str:
    templates = json.loads((champ / "prompts" / "index.json").read_text())[task]["templates"]
    import re
    text = re.sub(r"\{[^}]+\}", "red", templates[0])
    return text.replace("<y, x>", "<100, 100>")


class FakeResponder:
    """Astra ``Planner`` 的 ``responder`` 替身：直接写 ``response.json``（OpenAI usage 原样结构），不联网。

    ``fail_on_episode_index``：第 k 次出现的新局号（按请求顺序，从 1 计）上返回 ``status=error``。
    ``after_write``：每次写完后的回调（模拟同时在跑的费用守卫）。
    """

    def __init__(self, champ: Path, usage=None, fail_on_episode_index: int | None = None, after_write=None,
                 raise_on_send: BaseException | None = None, texts: list | None = None) -> None:
        self.champ = champ
        self.raise_on_send = raise_on_send  # 「发送时」抛异常（不写 response.json），验发送前落盘
        self.texts = list(texts or [])  # 依次替换回复原文（用完回落模板句）
        self.calls = 0
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 100,
                               "input_tokens_details": {"cached_tokens": 0},
                               "output_tokens_details": {"reasoning_tokens": 50}}
        self.fail_on = fail_on_episode_index
        self.after_write = after_write
        self.seen = []

    def __call__(self, out):
        out = Path(out)
        self.calls += 1
        request = json.loads((out / "request.json").read_text())
        key = (request["task"], request["episode"])
        if key not in self.seen:
            self.seen.append(key)
        if self.raise_on_send is not None:
            raise self.raise_on_send
        if self.fail_on is not None and len(self.seen) == self.fail_on:
            body = {"status": "error", "error": "HTTP 500: fake planner outage"}
        else:
            text = self.texts.pop(0) if self.texts else template_text(self.champ, request["task"])
            body = {"status": "ok", "text": text, "model": "gpt-6-astra",
                    "effort": "medium", "response_status": "completed", "usage": self.usage}
        (out / "response.json").write_text(json.dumps(body))
        if self.after_write is not None:
            self.after_write(out)


# ── 两个服务端替身 ─────────────────────────────────────────────────────────

class FakeServer:
    """``ServerProcess`` 替身：只记构造参数与 start／check／stop 次数，不起进程。``LOG`` 记全局先后次序。"""

    instances: list = []
    LOG: list = []

    def __init__(self, argv, env=None, cwd=None, gpu=None, ready=None, *, port, metadata_dir, policy_seed=None,
                 ckpt=None, log_path=None, name="server", ready_timeout_s=0.0, ready_poll_s=0.5, state_path=None):
        self.argv, self.env, self.cwd, self.gpu, self.ready = list(argv), dict(env or {}), cwd, gpu, ready
        self.port, self.metadata_dir, self.policy_seed, self.ckpt = port, Path(metadata_dir), policy_seed, ckpt
        self.log_path, self.name, self.ready_timeout_s, self.state_path = log_path, name, ready_timeout_s, state_path
        self.starts = self.stops = self.checks = 0
        self.dead = False
        self.pid = None
        self.metadata = None
        FakeServer.instances.append(self)

    def start(self):
        self.starts += 1
        self.pid = 40000 + len(FakeServer.instances)
        self.metadata = {"pid": self.pid, "policy_seed": self.policy_seed, "argv": self.argv, "port": self.port}
        FakeServer.LOG.append(("start", self.name))
        return self

    def alive(self) -> bool:
        return not self.dead

    def check(self):
        self.checks += 1
        if self.dead:
            from robomme_ood_eval.policy import ServerDead
            raise ServerDead(f"{self.name} 服务端已退出（替身）")

    def stop(self, **k):
        self.stops += 1
        FakeServer.LOG.append(("stop", self.name))
        return "term"

    def left_line(self) -> str:
        return f"SERVER_LEFT pid={self.pid} port={self.port} metadata=x stop=\"x\""


def _argv_value(argv: list, flag: str) -> str:
    return argv[argv.index(flag) + 1]


class FakeGuardServer(FakeServer):
    """费用守卫替身：不起进程，``start``／``check`` 在进程内跑一轮真实 ``guard_round``（扫 spool、写心跳状态与
    STOP），等同于一个刚好在这一刻扫过一轮的守卫。参数从守卫 argv 里取（``--root``／``--prices``／``--ledger``／
    ``--state``／``--cap``），因此同时核对了 argv 的形状。"""

    def start(self):
        super().start()
        guard = load_guard()
        self.root = Path(_argv_value(self.argv, "--root"))
        self.prices = guard.load_prices(Path(_argv_value(self.argv, "--prices")))
        self.ledger_path = Path(_argv_value(self.argv, "--ledger"))
        self.state = Path(_argv_value(self.argv, "--state"))
        self.cap = float(_argv_value(self.argv, "--cap"))
        self.ledger = guard.load_ledger(self.ledger_path)
        self.round()
        return self

    def round(self) -> dict:
        guard = load_guard()
        summary = guard.guard_round([self.root], self.ledger, self.prices, self.cap, 2048, self.state)
        guard.save_ledger(self.ledger_path, self.ledger)
        return summary

    def check(self):
        super().check()
        self.round()


# ── AstraPolicy 夹具 ────────────────────────────────────────────────────────

class Harness:
    """搭一个全替身的 ``AstraPolicy``（经 ``load_policy("astra", …)``）与外层 ``run_episode``。

    替身：两个服务端（``FakeServer``／``FakeGuardServer``）、``validate_checkpoints``、监视器、VLA 客户端、规划应答
    （``responder=None`` 时不替换，用真实 ``GuardedResponsesClient``，须配 ``NetCounter``）、外层 builder 与录制器。"""

    def __init__(self, tmp_path: Path, monkeypatch, mod, astra, *, env_plan=None, monitor=None, vla=None,
                 responder="fake", prices=None, max_episodes: int | None = None, **cfg) -> None:
        from robomme_ood_eval import episode as E

        self.tmp, self.mod, self.astra, self.E = Path(tmp_path), mod, astra, E
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: NullWriter())
        self.check_calls: list = []
        monkeypatch.setattr(astra.release_utils, "validate_checkpoints",
                            lambda v, m: self.check_calls.append((v, m)))
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "set-by-test")
        monkeypatch.setattr(FakeServer, "instances", [])
        monkeypatch.setattr(FakeServer, "LOG", [])
        monkeypatch.setattr(mod.AstraPolicy, "guard_server_cls", FakeGuardServer)
        monkeypatch.setattr(mod.AstraPolicy, "vla_server_cls", FakeServer)
        self.monitor = monitor if monitor is not None else FakeMonitor()
        self.vla = vla if vla is not None else FakeVLA()
        self.monitor_calls: list = []

        def make_monitor(policy, base, adapter):
            self.monitor_calls.append((base, adapter, os.environ.get("CUDA_VISIBLE_DEVICES")))
            return self.monitor

        monkeypatch.setattr(mod.AstraPolicy, "make_monitor", make_monitor)
        monkeypatch.setattr(mod.AstraPolicy, "make_client", lambda policy, port: self.vla)
        if responder == "fake":
            responder = FakeResponder(astra.champ)
        self.responder = responder
        if responder is not None:
            monkeypatch.setattr(mod.AstraPolicy, "make_responder", lambda policy, key, gate: self.responder)
        if max_episodes is not None:
            monkeypatch.setattr(mod.guard_module(), "ASTRA_MAX_EPISODES", int(max_episodes))
        self.cls = recording_builder_cls(env_plan)
        monkeypatch.setattr(E, "BUILDER_FACTORY",
                            lambda task, dataset, ms: self.cls(task, dataset=dataset, action_space="joint_angle",
                                                               gui_render=False, max_steps=ms))
        monkeypatch.setattr(E, "_BUILDERS", {})
        monkeypatch.setattr(FakeRecorder, "instances", [])
        work = self.tmp / "astra-cost"
        work.mkdir(parents=True, exist_ok=True)
        self.prices_path = work / "prices.json"
        self.prices_path.write_text(json.dumps(prices or PRICES))
        self.ledger = work / "ledger.json"
        self.out = self.tmp / "out"
        self.cfg = {"gpus": [0, 1], "astra_ledger": str(self.ledger), "astra_prices": str(self.prices_path),
                    "ckpt": str(self.tmp / "ckpt" / "symbolic-grounded-subgoal" / "79999"),
                    "astra_monitor_adapter": str(self.tmp / "ckpt" / "monitor"), "astra_monitor_base": "fake-base",
                    "astra_vla_python": str(self.tmp / "vla-venv" / "bin" / "python"), **cfg}
        self.policy = None

    @property
    def group(self) -> Path:
        return self.ledger.parent / "group_0"

    def load(self, seed: int = 7):
        from robomme_ood_eval.policy import load_policy

        self.policy = load_policy("astra", seed, **self.cfg)
        return self.policy

    def servers(self) -> dict:
        return {s.name: s for s in FakeServer.instances}

    def run(self, dataset: str, task: str, episode: int = 0, *, attempt: int = 1):
        return self.E.run_episode(self.policy, dataset, task, episode, self.out, attempt=attempt,
                                  recorder_factory=FakeRecorder, render=False)

    def batch(self, dataset: str, tasks: list, episode: int = 0):
        """仿 ``scripts/evaluate.py`` 主循环：逐局 ``run_episode``，``AstraStop`` 即整批停。返回 (结果列表, 停机异常)。"""
        from robomme_ood_eval.policy import AstraStop

        results = []
        try:
            for task in tasks:
                results.append(self.run(dataset, task, episode))
        except AstraStop as exc:
            return results, exc
        return results, None

    def raw(self, result) -> Path:
        return self.out / "rollouts" / "astra" / result.dataset / f"seed{result.policy_seed}" / result.raw_dir

    def astra_ep_dir(self, result) -> Path:
        return self.raw(result) / "astra" / result.task / f"ep{result.episode:03d}"

    def attempt_dir(self, result) -> Path:
        return self.astra_ep_dir(result) / f"{result.key}.a{result.attempt}"

    def ran(self) -> list:
        return [c["env_id"] for c in self.cls.make_calls]


def read_trace(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ── 语言账本 ────────────────────────────────────────────────────────────

def ensure_language_log(monkeypatch) -> str:
    """评估仓的 ``trace_writer.LanguageLog`` 恒存在：返回 ``real``（保留旧接口以免改动调用点）。"""
    from robomme_ood_eval.record import trace_writer as tw

    assert getattr(tw, "LanguageLog", None) is not None
    return "real"


def read_language(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
