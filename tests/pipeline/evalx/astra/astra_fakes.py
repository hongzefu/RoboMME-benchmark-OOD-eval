"""Shared fakes for the Astra tests: planner, monitor, VLA, environment and both servers are all faked.
No network access, no cost, no GPU.

The unit under test is the eval repo's ``robomme_ood_eval.models.astra.AstraPolicy`` (the 4 model-side methods).
The outer layer is the real ``robomme_ood_eval.episode.run_episode`` (the builder is a subclass of the real
``BenchmarkEnvBuilder`` with only ``make_env_for_episode`` stubbed to return ``FakeEnv``; the recorder is
``FakeRecorder`` and produces no site videos).

Astra upstream sources are referenced read-only from ``$SGEVAL_THIRD_PARTY/Astra-on-RoboMME``. When unset, the
current checkout's ``third_party`` is used; if the submodule is empty (e.g. inside a worktree), fall back to the main
checkout (the parent of ``git rev-parse --git-common-dir``). If none exists, fail outright (no skip).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import numpy as np

from tests._support.loaders import REPO

ASTRA_MODULES = ("runner", "core", "api_client", "release_utils", "input_contract", "train_entry")
#: Upstream runner.episode default cap on planner calls per episode
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
    """Third-party submodule dir: ``SGEVAL_THIRD_PARTY`` > current checkout ``third_party`` (if the submodule is
    initialized) > main checkout ``third_party``."""
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
    assert (root / "examples" / "champ" / "runner.py").is_file(), f"Astra upstream sources not found at {root}"
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
    """Import the Astra upstream modules and restore ``sys.path`` and ``sys.modules`` on exit, so other tests in
    the same pytest session are not polluted. Inside the session ``SGEVAL_THIRD_PARTY`` points at
    ``third_party_dir()`` (the module under test uses it to locate submodules and environment sources)."""
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
    """Replace ``urllib.request.urlopen`` with a fake that counts and then raises; ``api_calls`` is measured by it."""

    def __init__(self, raise_exc=None) -> None:
        self.calls = 0
        self.raise_exc = raise_exc

    def __call__(self, *a, **k):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc()
        raise RuntimeError("network access is forbidden in tests: urlopen was called")

    def install(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", self)
        return self


# -- Environment fakes ---------------------------------------------------

def _frame(value: int) -> np.ndarray:
    return np.full((256, 256, 3), value % 251, dtype=np.uint8)


class FakeEnv:
    """Minimal RoboMME env: success after ``terminal_step`` steps; ``None`` never ends (cut off by the loop cap)."""

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
    """Subclass of the real ``BenchmarkEnvBuilder``: constructor args go through real validation and are
    recorded; ``make_env_for_episode`` is stubbed.

    ``env_plan``: callable ``(builder, episode) -> FakeEnv``, or raise to simulate an infrastructure failure.
    """
    from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder

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
    """Fake outer recorder (same interface as ``record.recorder.EpisodeRecorder``): only counts."""

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
    """Fake ``imageio.get_writer`` (Astra's own rollout.mp4 is not actually encoded)."""

    def append_data(self, frame):
        pass

    def close(self):
        pass


# -- Monitor, VLA and planner-responder fakes ------------------------------

class FakeMonitor:
    """Same signature as Astra ``Monitor.predict``; writes ``input.json`` and one image, returns boolean
    predictions in sequence (always False by default)."""

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
        # Same as upstream Monitor.predict: raw reply goes to response.json (the language log reads it)
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
    """Fake ``responder`` for the Astra ``Planner``: writes ``response.json`` directly (OpenAI usage structure
    verbatim), no network.

    ``fail_on_episode_index``: return ``status=error`` on the k-th distinct episode seen (request order, 1-based).
    ``after_write``: callback after every write (simulates a concurrently running cost guard).
    """

    def __init__(self, champ: Path, usage=None, fail_on_episode_index: int | None = None, after_write=None,
                 raise_on_send: BaseException | None = None, texts: list | None = None) -> None:
        self.champ = champ
        self.raise_on_send = raise_on_send  # raise "at send time" (no response.json written) to verify persist-before-send
        self.texts = list(texts or [])  # replace reply texts in order (falls back to the template sentence when exhausted)
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


# -- Fakes for the two servers --------------------------------------------

class FakeServer:
    """Fake ``ServerProcess``: records constructor args and start/check/stop counts, spawns no process.
    ``LOG`` records the global order of events."""

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
            raise ServerDead(f"{self.name} server has exited (fake)")

    def stop(self, **k):
        self.stops += 1
        FakeServer.LOG.append(("stop", self.name))
        return "term"

    def left_line(self) -> str:
        return f"SERVER_LEFT pid={self.pid} port={self.port} metadata=x stop=\"x\""


def _argv_value(argv: list, flag: str) -> str:
    return argv[argv.index(flag) + 1]


class FakeGuardServer(FakeServer):
    """Fake cost guard: spawns no process; ``start``/``check`` run one real ``guard_round`` in-process (scan the
    spool, write heartbeat state and STOP), equivalent to a guard that happened to sweep exactly at this moment.
    Parameters are read from the guard argv (``--root``/``--prices``/``--ledger``/``--state``/``--cap``), which
    also checks the argv shape."""

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


# -- AstraPolicy harness --------------------------------------------------

class Harness:
    """Build a fully faked ``AstraPolicy`` (via ``load_policy("astra", ...)``) plus the outer ``run_episode``.

    Fakes: both servers (``FakeServer``/``FakeGuardServer``), ``validate_checkpoints``, monitor, VLA client, planner
    responder (with ``responder=None`` it is not replaced and the real ``GuardedResponsesClient`` is used, which must
    be paired with ``NetCounter``), outer builder and recorder."""

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
        """Mimic the ``scripts/evaluate.py`` main loop: ``run_episode`` per episode, ``AstraStop`` stops the whole
        batch. Returns (results, stop exception)."""
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


# -- Language log ----------------------------------------------------------

def ensure_language_log(monkeypatch) -> str:
    """The eval repo's ``trace_writer.LanguageLog`` always exists: returns ``real`` (old interface kept so call
    sites need not change)."""
    from robomme_ood_eval.record import trace_writer as tw

    assert getattr(tw, "LanguageLog", None) is not None
    return "real"


def read_language(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# -- Hard cost cap: zero-network fixtures (moved from test_astra_wiring.py, shared with test_astra_stage3.py;
#    PRICES is at the top of the file) ----------------------------------------------------------------------

#: Worst-case price of 2048 output tokens: 2048 * 8 / 1e6
OUT_WORST = 2048 * 8.0 / 1e6


class FakeUrlopen:
    """Fake ``urllib.request.urlopen``: counts calls and returns Responses-API-style replies in order (or raises
    HTTPError)."""

    def __init__(self, plan=None) -> None:
        self.calls = 0
        self.plan = list(plan or [])

    def __call__(self, request, timeout=None):
        self.calls += 1
        item = self.plan.pop(0) if self.plan else {"input_tokens": 100}
        if isinstance(item, Exception):
            raise item
        body = {"status": "completed", "model": "gpt-6-astra", "reasoning": {"effort": "medium"},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}
        if item is not None:
            body["usage"] = {"input_tokens": item["input_tokens"], "output_tokens": 50,
                             "input_tokens_details": {"cached_tokens": 0},
                             "output_tokens_details": {"reasoning_tokens": 10}}
        return _Resp(body)


class _Resp:
    def __init__(self, body):
        self._b = json.dumps(body).encode()
        self.headers = {"x-request-id": "req-fake"}

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_429():
    headers = Message()
    headers["Retry-After"] = "1"
    return urllib.error.HTTPError("https://example.invalid", 429, "rate", headers,
                                  io.BytesIO(b'{"error":{"code":"rate_limit_exceeded"}}'))


class Clock:
    """Fake monotonic clock: ``sleep`` only advances time and runs a callback (simulates the guard writing STOP
    while waiting)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list = []
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += max(0.0, s)
        if self.on_sleep is not None:
            self.on_sleep(s)


class GuardFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.guard = load_guard()
        self.group = tmp_path / "group_0"
        self.spool = self.group / "run" / "planner_calls"
        self.spool.mkdir(parents=True)
        self.state = tmp_path / "guard" / "ledger.state.json"
        self.ledger = self.guard.new_ledger()
        self.prices = dict(PRICES, cached_input=0.5, reasoning_billed_separately=False)
        self.n = 0

    def round(self) -> dict:
        return self.guard.guard_round([self.group], self.ledger, self.prices, 5.0, 2048, self.state)

    def request(self, text_bytes: int, images: int = 0) -> Path:
        """Write one request in the Astra ``Planner`` spool layout (text + 256x256 PNGs)."""
        from PIL import Image
        self.n += 1
        out = self.spool / f"{self.n:032x}"
        out.mkdir()
        names = []
        for i in range(images):
            Image.fromarray(np.full((256, 256, 3), i, np.uint8)).save(out / f"{i}.png")
            names.append(f"{i}.png")
        (out / "request.json").write_text(json.dumps({"images": names, "task": "BinFill", "episode": 0}))
        (out / "prompt.txt").write_text("x" * text_bytes)
        return out

    def reservations(self) -> dict:
        return self.guard.load_reservations(self.state)["reservations"]


def _client(mod, astra, fx: GuardFixture, clock: Clock, gate_clock=None):
    gate = mod.CostGate(fx.state, clock=gate_clock or time.time)
    cls = mod.guarded_client_class(astra.api_client)
    return cls(FAKE_KEY, gate, sleep=clock.sleep, monotonic=clock.monotonic), gate


def _resp(out: Path) -> dict:
    return json.loads((out / "response.json").read_text())
