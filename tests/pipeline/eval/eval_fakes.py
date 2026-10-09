"""Shared fakes for the evaluation pipeline tests (C13): fake environment, hybrid builder, fake recorder, fake policy
server, and fake connections for the two protocols.

Plan detail 4.5.4 originally placed this at ``tests/_support/eval_fakes.py``; ``tests/_support/`` belongs to the main
session, so per the assignment table this module lives in ``tests/pipeline/eval/``.

Design rules:
- Production modules are always loaded by path via ``tests._support.loaders.load_script`` (same as the production
  entry); no stubs are injected into sys.modules.
- Identity resolution uses the real ``robomme_hard`` ``BenchmarkEnvBuilder`` (packaged specs); only
  ``make_env_for_episode`` is replaced with a CPU fake environment, so identity checks (``check_identity``) use real
  resolution results. No simulation scene is built.
- The fake policy server's actions are generated deterministically from a digest of "all frame fingerprints received
  since this episode's reset + current state + instruction": an episode without a reset would compute different
  actions from the previous episode's buffer, which makes cross-episode isolation (A->B->A) observable.
- The smvla protocol fingerprint functions come from the real ``smvla_server.py`` (the server-side implementation);
  they are not re-implemented in the tests.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import pickle
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tests._support.loaders import REPO, load_script

HW = 4  # fake frame side length (pixels); the shape plays no role in the logic under test
#: Step caps (split plan, settled rule 3), hand-written per the convention rather than read from the code under test:
#: ood is 1800 with strict truncation, hard-verify is 1300 without truncation
V9_MAX_STEPS = 1800
HARD0_MAX_STEPS = 1300
N_RESET_FRAMES = 3  # frames returned by the fake env's reset (2 demo frames + 1 initial frame)
#: action rows returned per inference by the fake server (more than the execution segment, to check that only the
#: first rows are executed)
CHUNK_ROWS = 20


# ---------------------------------------------------------------- production modules


def env_session():
    """After the repo split, ``EnvSession``, ``NullRecorder``, ``StepCapReached``, ``RecorderError`` and
    ``ResetBudgetExhausted`` live in the evaluation package ``robomme_ood_eval.session`` (formerly env_client.py)."""
    from robomme_ood_eval import session

    return session


def framesamp_modul_client():
    return load_script("eval-official/framesamp_modul_client.py")


def smvla_client():
    return load_script("eval-official/smvla_client.py")


def smvla_server():
    return load_script("eval-official/smvla_server.py")


def hard_specs():
    from robomme_hard.env_record_wrapper import hard_specs as hs

    return hs


def real_builder(task: str, max_steps: int | None = None, dataset: str = "ood"):
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    kw = {} if max_steps is None else {"max_steps": int(max_steps)}
    return BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="joint_angle", **kw)


# ---------------------------------------------------------------- observations and fake environment


def frame(v: int) -> np.ndarray:
    return np.full((HW, HW, 3), int(v) % 256, dtype=np.uint8)


def sha_bytes(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def obs_of(vals: list[int]) -> dict:
    """Observation with the same keys as the real environment: five lists of equal length; 7-D joints, 2-D gripper,
    6-D end effector."""
    return {
        "front_rgb_list": [frame(v) for v in vals],
        "wrist_rgb_list": [frame(v + 1) for v in vals],
        "joint_state_list": [np.full(7, v / 10.0, dtype=np.float64) for v in vals],
        "gripper_state_list": [np.array([v / 100.0, v / 100.0], dtype=np.float64) for v in vals],
        "eef_state_list": [np.zeros(6, dtype=np.float64) for _ in vals],
    }


def reset_values(ep: int) -> list[int]:
    base = (ep * 7) % 150
    return [base + i for i in range(N_RESET_FRAMES)]


@dataclasses.dataclass
class Plan:
    """Fake environment behavior within one attempt: step n reports a success/fail terminal status or raises;
    if all are None the episode never terminates."""

    success_at: int | None = None
    fail_at: int | None = None
    raise_at: int | None = None
    raise_exc: Callable[[], BaseException] | None = None


class FakeEnv:
    def __init__(self, task: str, ep: int, plan: Plan):
        self.task, self.ep, self.plan = task, ep, plan
        self.n = 0
        self.actions: list[np.ndarray] = []
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1
        return obs_of(reset_values(self.ep)), {"task_goal": [f"goal-{self.task}-{self.ep}", "alt"], "status": "ongoing"}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        self.n += 1
        if self.plan.raise_at == self.n:
            raise self.plan.raise_exc()
        status = "ongoing"
        if self.plan.success_at == self.n:
            status = "success"
        elif self.plan.fail_at == self.n:
            status = "fail"
        terminated = status in ("success", "fail")
        return obs_of([100 + (self.ep * 3 + self.n) % 100]), 0.0, terminated, False, {"status": status}

    def close(self):
        self.closed = True


class World:
    """Fake world shared by a group of tests: hands out a Plan per attempt keyed by (task, builder_episode) and records
    all fake environments and builder calls."""

    def __init__(self, plans: dict[tuple[str, int], list[Plan]] | None = None, default: Plan | None = None):
        self.plans = {k: list(v) for k, v in (plans or {}).items()}
        self.default = default or Plan(success_at=3)
        self.envs: list[FakeEnv] = []
        self.make_calls: list[tuple] = []
        self.builders: list[Any] = []
        self.recorders: list["FakeRecorder"] = []

    def new_env(self, task: str, ep: int) -> FakeEnv:
        lst = self.plans.get((task, ep))
        plan = (lst.pop(0) if len(lst) > 1 else lst[0]) if lst else self.default
        env = FakeEnv(task, ep, plan)
        self.envs.append(env)
        return env

    def envs_of(self, task: str, ep: int) -> list[FakeEnv]:
        return [e for e in self.envs if (e.task, e.ep) == (task, ep)]


class HybridBuilder:
    """Real builder identity resolution + CPU fake environment (no simulation scene is built). ``dataset`` is passed
    to the real builder unchanged."""

    def __init__(self, task: str, max_steps: int | None, world: World, dataset: str = "ood"):
        self.task, self.max_steps, self.world, self.dataset = task, max_steps, world, dataset
        self.real = real_builder(task, max_steps, dataset)
        world.builders.append(self)

    def resolve_identity(self, ep):
        return self.real.resolve_identity(ep)

    def make_env_for_episode(self, ep, max_steps=None):
        self.world.make_calls.append((self.task, int(ep), max_steps, self.max_steps))
        return self.world.new_env(self.task, int(ep))


class FakeRecorder:
    """Same interface as ``recorder.EpisodeRecorder``; on close it writes the three placeholder media files the report
    check needs."""

    def __init__(self, rec_dir, meta, world: World | None = None, *, fail_on: str | None = None):
        self.rec_dir = Path(rec_dir)
        self.meta = dict(meta)
        self.phases: list[str] = []
        self.frames: Counter = Counter()
        self.arrays: Counter = Counter()
        self.events: list[dict] = []
        self.fail_on = fail_on
        self.closed_with = None
        if world is not None:
            world.recorders.append(self)

    def _maybe_fail(self, name):
        if self.fail_on == name:
            raise OSError(28, "No space left on device")

    def set_phase(self, phase):
        self._maybe_fail("set_phase")
        self.phases.append(phase)

    def add_frames(self, stream, frames, *, tag=""):
        self._maybe_fail("add_frames")
        n = int(np.asarray(frames).shape[0])
        self.frames[stream] += n
        return list(range(n))

    def add_array(self, name, arr, *, step=None):
        self._maybe_fail("add_array")
        self.arrays[name] += 1

    def add_event(self, event):
        self.events.append(dict(event))

    def close(self, summary):
        self._maybe_fail("close")
        self.closed_with = dict(summary)
        self.rec_dir.mkdir(parents=True, exist_ok=True)
        for name in ("front.mkv", "wrist.mkv"):
            (self.rec_dir / name).write_bytes(b"placeholder")
        (self.rec_dir / "summary.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
        return {"RECORDER_VERIFY": "PASS"}


# ---------------------------------------------------------------- fake policy server and connections for the two protocols


def connection_closed():
    from websockets.exceptions import ConnectionClosed

    return ConnectionClosed(None, None)


class FakePolicyServer:
    """Policy state shared across connections. ``fail_on``: None / "infer_disconnect" (connection drops during
    inference) / "server_error" (replies with an error)."""

    def __init__(self, *, fail_on: str | None = None):
        self.fail_on = fail_on
        self.buffer: list[str] = []
        self.log: list[tuple[str, Any]] = []

    def reset(self, key=None):
        self.buffer = []
        self.log.append(("reset", key))

    def observe(self, shas: list[str], extra=None):
        self.buffer.extend(shas)
        self.log.append(("observe", {"frames": list(shas), **(extra or {})}))

    def actions(self, state: np.ndarray, prompt: str) -> np.ndarray:
        self.log.append(("infer", {"state": np.array(state, copy=True), "prompt": prompt,
                                   "buffer_len": len(self.buffer)}))
        if self.fail_on == "infer_disconnect":
            raise connection_closed()
        h = hashlib.sha256("|".join(self.buffer).encode() + np.asarray(state, np.float32).tobytes()
                           + prompt.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        a = rng.uniform(-1.0, 1.0, size=(CHUNK_ROWS, 8)).astype(np.float32)
        self.log[-1][1]["actions"] = a.copy()
        return a

    def kinds(self) -> list[str]:
        return [k for k, _ in self.log]


class _FakeWS:
    def __init__(self, server):
        self.server = server

    def close(self):
        self.server.log.append(("close", None))


class FakeMMEVLAWebsocketClient:
    """Protocol stub for ``MMEVLAWebsocketClientPolicy``: three message kinds, reset / add_buffer / infer."""

    def __init__(self, server: FakePolicyServer):
        self.server = server
        self._ws = _FakeWS(server)

    def reset(self):
        self.server.reset()
        return {"reset_finished": True}

    def add_buffer(self, buf):
        imgs = np.asarray(buf["images"])
        self.server.observe([sha_bytes(imgs[i, 0]) for i in range(imgs.shape[0])],
                            {"exec_start_idx": int(buf["exec_start_idx"]), "shape": list(imgs.shape),
                             "states": [np.array(s, copy=True) for s in buf["state"]]})
        return {"add_buffer_finished": True}

    def infer(self, element):
        a = self.server.actions(np.asarray(element["observation/state"]), str(element["prompt"]))
        self.server.log[-1][1].update(image=sha_bytes(element["observation/image"]),
                                      wrist=sha_bytes(element["observation/wrist_image"]))
        return {"actions": a}


class FakeSmvlaConn:
    """Fake connection for the smvla protocol (same interface as ``smvla_client.WSPolicyConn``); reply fingerprints
    use the real ``smvla_server`` functions."""

    def __init__(self, server: FakePolicyServer, *, tamper: str | None = None):
        self.server = server
        self.metadata = {"policy": "smvla", "fake": True}
        self.tamper = tamper  # None / "req_sha" / "frame_sha"
        self.n = 0
        self.closed = False

    def call(self, msg: dict):
        srv = smvla_server()
        raw = pickle.dumps((self.n, msg), protocol=4)
        self.n += 1
        kind = next(iter(msg))
        if kind == "reset":
            self.server.reset(msg["reset"]["episode_key"])
            rep = {"reset_finished": True, "rng": {"fake": True}}
        elif kind == "observe":
            frs = msg["observe"]["frames"]
            shas = [{k: srv.frame_sha(v) for k, v in sorted(f.items())} for f in frs]
            self.server.observe([d["front"] for d in shas])
            if self.tamper == "frame_sha" and shas:
                shas = shas[:-1]
            rep = {"observe_finished": True, "n": len(frs), "frame_sha": shas}
        elif kind == "infer":
            if self.server.fail_on == "server_error":
                rep = {"error": "Traceback: RuntimeError: fake server internal error"}
                return rep, raw, b"e" + raw
            p = msg["infer"]
            full = self.server.actions(np.asarray(p["state"]), str(p["instruction"]))
            rep = {"actions": full[:srv.EXECUTE_HORIZON], "actions_full": full, "subtask": "s", "infer_ms": 1.0,
                   "recv_state_sha": srv.array_sha(np.asarray(p["state"])),
                   "recv_instruction_sha": srv.sha256_bytes(str(p["instruction"]).encode("utf-8"))}
        else:
            raise AssertionError(f"unknown message {kind}")
        rep["req_sha"] = srv.sha256_bytes(raw)
        if self.tamper == "req_sha":
            rep["req_sha"] = "0" * 64
        return rep, raw, b"r" + raw

    def close(self):
        self.closed = True


def framesamp_modul_policy(monkeypatch, server: FakePolicyServer):
    """Real framesamp_modul_client module with only the websocket-client factory replaced by a fake client (a new
    client per episode, same as the real behavior)."""
    mc = framesamp_modul_client()
    monkeypatch.setattr(mc, "make_recording_client", lambda host, port, recorder, timing: FakeMMEVLAWebsocketClient(server))
    return mc


def smvla_policy(server: FakePolicyServer, **conn_kw):
    """Real smvla_client.run_episode with a fake connection injected; keeps its keyword signature (SeatRunner relies on
    it to pass max_steps / reset_retries)."""
    sm = smvla_client()
    return types.SimpleNamespace(run_episode=functools.partial(sm.run_episode, conn=FakeSmvlaConn(server, **conn_kw)))


def policy_module(name: str, monkeypatch, server: FakePolicyServer):
    return framesamp_modul_policy(monkeypatch, server) if name == "perceptual-framesamp-modul" else smvla_policy(server)


# ---------------------------------------------------------------- identity


def tier_cap(tier: str) -> int:
    """Step cap for the tier per the launch convention (hand-written constants): xhard0 uses hard-verify's 1300, the
    other tiers use ood's 1600."""
    return HARD0_MAX_STEPS if tier == "xhard0" else V9_MAX_STEPS


@functools.lru_cache(maxsize=None)
def _resolved(task: str) -> tuple[tuple[int, dict], ...]:
    """Per-episode resolution by the real ood builder (after the repo split ood has no xhard0; episodes 0-49)."""
    b = real_builder(task)
    return tuple((ep, b.resolve_identity(ep)) for ep in range(b.get_episode_num()))


def packaged_identity(task: str, tier: str, k: int = 0) -> dict:
    """Real packaged identity (the k-th episode of this tier in ood per the real builder) -> execution identity row
    (field contract C1 + dataset; key hand-written per the contract)."""
    hits = [(ep, ident) for ep, ident in _resolved(task) if ident["tier"] == tier]
    ep, ident = hits[k]
    return {"dataset": "ood", "task": task, "tier": tier, "seed": int(ident["seed"]), "candidate": ident["candidate"],
            "builder_episode": ep, "source_episode": None, "spec_sha256": ident["spec_sha256"],
            "key": f"{task}_{tier}_{int(ident['seed'])}"}


@functools.lru_cache(maxsize=None)
def _resolved_hard0(task: str) -> tuple[tuple[int, dict], ...]:
    b = real_builder(task, dataset="hard-verify")
    return tuple((ep, b.resolve_identity(ep)) for ep in range(b.get_episode_num()))


def hard0_identity(task: str, k: int = 0) -> dict:
    """Execution identity row for the k-th episode in hard-verify (same field contract as C1; candidate / spec_sha256
    are null; key hand-written per the contract)."""
    ep, ident = _resolved_hard0(task)[k]
    return {"dataset": "hard-verify", "task": task, "tier": "xhard0", "seed": int(ident["seed"]), "candidate": None,
            "builder_episode": ep, "source_episode": int(ident["source_episode"]), "spec_sha256": None,
            "key": f"{task}_xhard0_{int(ident['seed'])}"}


def v9_cells_sorted() -> list[tuple[str, str]]:
    return sorted(hard_specs().V9_CELLS)


def episode_mod():
    from robomme_ood_eval import episode as E

    return E


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


__all__ = [n for n in dir() if not n.startswith("_")] + ["REPO"]
