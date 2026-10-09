"""Astra model: connects Astra-on-RoboMME's per-episode loop ``runner.episode`` to the evaluation repo's four
model-side methods.

The four methods of ``AstraPolicy`` (called by the outer ``load_policy`` / ``run_episode``; interface in
``robomme_ood_eval.policy``):

- ``load()`` (once per process): check the model seed and the two GPUs; ``bootstrap`` the submodule's
  ``examples/champ``; assert the environment source is ``third_party/robomme_benchmark/src``;
  ``validate_checkpoints``; check ``group_*/STOP.json`` (refuse if already stopped; the ledger is kept); start via
  ``ServerProcess`` the **cost guard** ``servers/astra_cost_guard.py --cap <astra_cap_usd<=5>`` (ledger
  ``astra_ledger``, unit prices ``astra_prices``, ready = guard heartbeat state is fresh) and the **Astra VLA server**
  (the submodule's ``scripts/serve_policy.py``, ``--seed=<policy_seed>``, ``CUDA_VISIBLE_DEVICES=gpus[0]``, ready =
  port listening); pin this process to ``gpus[1]`` and load ``runner.Monitor``; build one ``GuardedResponsesClient``
  (the 20-second send interval is kept by the same instance) and one ``core.Planner``.
- ``reset(spec)`` (first statement of each episode; sends no server message and never touches the environment):
  probe that both servers are alive; raise ``AstraStop`` if a stop condition left by the previous episode or
  ``group_*/STOP.json`` holds; ``CostGate.register_episode`` registers the episode count across processes (the third
  episode is refused).
- ``play(session, spec, recorder)``: hand the outer ``EnvSession`` via ``SessionBuilder`` to the submodule's
  ``runner.episode(args, task, ep, builder, monitor, planner, client)``, wrapped in ``Traced*`` recorders that write
  this episode's ``trace.jsonl`` / ``language.jsonl``; ``args.max_steps`` follows each episode's ``spec.max_steps``
  (one Policy can run both the 1300 and 1800 caps in turn). Afterwards the stop condition is recorded per the
  upstream ``main()`` stop rules and raised by the next episode's ``reset``.
- ``close()``: stop the VLA server process group first, then the guard (which writes ``exited=true`` after
  receiving TERM), and release the monitor model.

Stop rules (moved from the old ``run_cases`` into the Policy; judged after play returns, the next episode's
``reset`` raises ``AstraStop`` and the outer loop stops the whole batch):
(1) an episode whose error message starts with one of ``PLANNER_STOP_PREFIXES`` stops immediately; (2) 3 consecutive
episodes with ``status=="error"`` that are not a planner output template mismatch (``core.is_planner_failure``) stop
(a normal outcome resets the count); (3) stop when the registered episode count reaches ``ASTRA_MAX_EPISODES``;
and (4) ``group_*/STOP.json`` (written by the guard at the limit, or manually) is checked at every episode ``reset``
and before every send.

Outputs:

1. ``args.output`` = ``<spec.out_dir>/astra``: Astra's own ``<task>/ep<NNN>/{identity.json, decisions.jsonl,
   rollout.mp4, actions.npy, result.json, monitor_inputs/}`` stays there, without clashing with the outer output
   tree;
2. the attempt directory name uses ``spec.attempt``: ``<task>/ep<NNN>/<key>.a<attempt>/provenance.json`` (no longer
   hard-coded to ``a1``);
3. stop rules as above.

``trace.jsonl``, ``language.jsonl`` and ``arrays.npz`` are written in ``spec.out_dir`` (the outer loop renders the
official-layout video from the trace here; ``arrays.npz`` is merged via ``trace_writer.merge_write_npz`` with
same-named keys of the outer recorder, checked item by item). Raw frames, per-step arrays and events are written by
the outer ``EnvSession`` into the outer recorder; this module no longer creates its own recorder.

Cost policy (Astra has its own cost ledger with a hard 5 USD limit): the guard's ``--cap`` defaults to and is at most
5 USD (``HARD_CAP_USD``); ``GuardedResponsesClient`` synchronously reads the guard state before every real send and
atomically reserves the single-request worst-case cost, refusing to send when the guard is lost (heartbeat older
than 10 seconds), STOP is set, or the reservation fails; the hard episode limit is 2, registered across processes in
the guard reservation file. The ledger, reservation file and ``STOP.json`` all live long-term in the directory of
``astra_ledger``; a new ledger is never created to get around the limit.

The API key is read only from the environment variable ``OPENAI_API_KEY`` and handed to ``GuardedResponsesClient``;
this module never reads key files or prints the key; the variable is removed from the VLA server process
environment (same as the upstream ``run.sh`` ``env -u OPENAI_API_KEY``).

Model seed: ``policy_seed`` becomes the VLA server ``--seed`` (replacing upstream's fixed 42); the cloud planner /
monitor have no seed interface, so trace, ``result.json`` and the provenance manifest only record ``policy_seed`` and
``cloud_seed=null``, never fabricated.

Steps: ``ood <-> 1800`` (strict: both the outer ``EnvSession`` and ``TracedEnv`` refuse the 1801st ``step`` before it
reaches the environment, and the episode finalizes as ``timeout``), ``hard-verify <-> 1300`` (not strict).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from robomme_ood_eval.policy import AstraStop as _PolicyAstraStop
from robomme_ood_eval.policy import Policy, Ready, ServerProcess, pick_port

HERE = Path(__file__).resolve().parent
#: evaluation repo root (three levels above ``src/robomme_ood_eval/models``)
REPO_ROOT = HERE.parents[2]

#: the new side only accepts these two datasets; consistency-checked against each episode's ``spec.max_steps``
DATASET_STEP_PAIRING = {"hard-verify": 1300, "ood": 1800}
#: ood is strict (the 1801st step is refused before reaching the real environment); hard-verify is unchanged
#: (truncation by the underlying environment)
DATASET_STRICT_CAP = {"hard-verify": False, "ood": True}
#: fixed output limit in the request body of upstream ``api_client.ResponsesClient`` (shared by planning and review;
#: only used for language ledger params)
PLANNER_MAX_OUTPUT_TOKENS = 2048
#: ``RequestConfig(max_tokens=8, temperature=0)`` of upstream ``runner.Monitor`` (only used for language ledger params)
MONITOR_PARAMS = {"temperature": 0, "max_tokens": 8}
#: file name of the provenance manifest (under the attempt directory ``<key>.a<attempt>/``)
PROVENANCE_FILE = "provenance.json"
#: upstream ``main()`` item 5: an episode whose error message starts with one of these three prefixes stops at once
PLANNER_STOP_PREFIXES = ("Planner API", "Pilot planner-call", "Planner bridge failed")
#: upstream ``main()`` item 4: stop when non-planner errors accumulate consecutively to this count
INFRA_ERROR_LIMIT = 3
#: ``ResponsesClient._send`` only honors STOP.json under these two directory names
GROUP_DIR_NAMES = ("group_0", "group_1")
#: Astra source files whose sha256 is printed at startup (for the record)
ASTRA_SOURCE_FILES = ("runner.py", "core.py", "api_client.py", "input_contract.py", "release_utils.py",
                      "train_entry.py", "weights.json")
#: new-side route name
ROUTE = "astra/new"
#: subdirectory name of Astra's own outputs under this episode's raw directory
ASTRA_SUBDIR = "astra"
#: terminal statuses
TERMINALS = ("success", "fail", "timeout", "error")
#: guard heartbeat timeout (seconds): beyond it the guard is considered lost and sends are refused (same as
#: astra_cost_guard.HEARTBEAT_TIMEOUT_S, checked by tests)
GUARD_HEARTBEAT_TIMEOUT_S = 10.0
#: fixed interval between two sends of the third-party ResponsesClient (seconds; the 20 in upstream _send)
SEND_INTERVAL_S = 20
#: default VLA server port base (same as 18762 in upstream run.sh)
DEFAULT_PORT_BASE = 18762
#: default per-episode planning call limit of upstream ``runner.episode`` (upstream main()'s ``--max-planner-calls``
#: defaults to 24)
DEFAULT_MAX_PLANNER_CALLS = 24
#: VLA server readiness timeout (seconds; the old launcher used 15 minutes)
VLA_READY_TIMEOUT_S = 900.0
#: guard readiness timeout (seconds)
GUARD_READY_TIMEOUT_S = 60.0
#: Astra per-episode wall clock (seconds): 20-second send interval x at most 24 planning calls + monitoring and
#: simulation is tight against the default 1800, so it is relaxed to 3600
ASTRA_EPISODE_WALL_S = 3600.0


class AstraStop(_PolicyAstraStop):
    """Astra stop (equivalent to the few ``raise RuntimeError`` in upstream ``main()``); ``reason`` is for tests and
    log reading. A subclass of ``robomme_ood_eval.policy.AstraStop``: the outer ``scripts/evaluate.py`` stops the
    whole batch on it (exit 3)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class StepCapReached(RuntimeError):
    """strict cap: calling ``step`` after ``effective_cap`` steps were executed never reaches the real environment
    (same meaning as ``session.StepCapReached``). Inherits ``RuntimeError``: Astra ``runner.episode``'s
    ``except Exception`` catches it, and this module records ``timeout`` at finalization."""


# -- paths and upstream modules ----------------------------------------------------------

def third_party_root() -> Path:
    """Third-party submodule root: ``SGEVAL_THIRD_PARTY`` (e.g. to reference a populated checkout read-only from a
    worktree) > this repo's ``third_party``."""
    third = os.environ.get("SGEVAL_THIRD_PARTY")
    return Path(third).resolve() if third else (REPO_ROOT / "third_party").resolve()


def astra_root(explicit: str | None = None) -> Path:
    """Astra submodule root: explicit argument (cfg ``astra_root``) > ``<third_party>/Astra-on-RoboMME``."""
    if explicit:
        return Path(explicit).resolve()
    return third_party_root() / "Astra-on-RoboMME"


def benchmark_src() -> Path:
    """The only allowed environment source: ``<third_party>/robomme_benchmark/src`` (the benchmark submodule)."""
    return third_party_root() / "robomme_benchmark" / "src"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(root: Path) -> SimpleNamespace:
    """Add Astra's ``examples/champ`` and ``packages/openpi-client/src`` to ``sys.path``, import Astra's four modules
    and return them.

    This module's own directory is no longer put on ``sys.path`` (in the evaluation repo that is ``models/``, which
    would shadow same-named modules); ``trace_writer`` and the guard are always imported by package path."""
    champ = root / "examples" / "champ"
    if not (champ / "runner.py").is_file():
        raise FileNotFoundError(f"Astra source not found: {champ}/runner.py (submodule not initialized?)")
    for entry in (str(root / "packages" / "openpi-client" / "src"), str(champ)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    import api_client  # noqa: PLC0415  Astra's direct Responses API client
    import core  # noqa: PLC0415
    import release_utils  # noqa: PLC0415
    import runner as astra_runner  # noqa: PLC0415  Astra's per-episode loop (module-level episode function)
    return SimpleNamespace(root=root, champ=champ, runner=astra_runner, core=core,
                           release_utils=release_utils, api_client=api_client)


def source_digests(champ: Path) -> dict[str, str]:
    return {name: file_sha256(champ / name) for name in ASTRA_SOURCE_FILES if (champ / name).is_file()}


def assert_env_sources() -> dict[str, str]:
    """The environment source must be ``third_party/robomme_benchmark/src``: neither ``robomme`` nor ``robomme_hard``
    may come from the official copy nested in Astra or anywhere else."""
    import robomme  # noqa: PLC0415
    import robomme_hard  # noqa: PLC0415
    want = str(benchmark_src().resolve()) + os.sep
    files = {"robomme_hard": str(Path(robomme_hard.__file__).resolve()),
             "robomme": str(Path(robomme.__file__).resolve())}
    for name, path in files.items():
        if not path.startswith(want):
            raise RuntimeError(f"{name} does not come from third_party/robomme_benchmark/src ({want}): {path}")
    print(f"ASTRA_ENV_SOURCE robomme_hard={files['robomme_hard']} robomme={files['robomme']}", flush=True)
    return files


def check_pairing(dataset: str, max_steps: int) -> None:
    expected = DATASET_STEP_PAIRING.get(dataset)
    if expected is None:
        raise ValueError(f"RUN_BLOCKED reason=dataset dataset={dataset!r} (only {sorted(DATASET_STEP_PAIRING)} are accepted)")
    if int(max_steps) != expected:
        raise ValueError(f"RUN_BLOCKED reason=step_cap_pairing dataset={dataset} max_steps={max_steps} (expected {expected})")


def check_policy_seed(value) -> int:
    """``policy_seed`` is required, a non-negative integer; missing or invalid means ``RUN_BLOCKED reason=policy_seed``
    (no fallback to the old default 42)."""
    if value is None or isinstance(value, bool):
        raise ValueError("RUN_BLOCKED reason=policy_seed the model seed must be given explicitly (non-negative integer, "
                         "no fallback to any old default)")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.strip().isdigit():
        seed = int(value.strip())
    else:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed policy_seed={value!r} must be a non-negative integer")
    if seed < 0:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed policy_seed={value!r} must be a non-negative integer")
    return seed


# -- trace delegation (trace_writer) -----------------------------------------------------

def _canonical(obj: Any) -> bytes:
    """Normalized bytes: arrays always become ``array_record`` (dtype, shape, sha256), everything else is serialized
    as JSON with sorted keys."""
    import numpy as np  # noqa: PLC0415
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415

    def conv(value):
        if isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")):
            rec = tw.array_record(value)
            return {"dtype": rec["dtype"], "shape": rec["shape"], "sha256": rec["sha256"]}
        if isinstance(value, dict):
            return {str(k): conv(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [conv(v) for v in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    return json.dumps(conv(obj), sort_keys=True, ensure_ascii=False).encode()


class TraceContext:
    """Shared state of one episode: current step number and subgoal (``grounded_subgoal`` in VLA requests), the three
    C8 counts, original actions, recorder.

    ``open_recorder``: a no-argument callable invoked once by ``TracedEnv`` at the first reset to create a recorder
    (the evaluation repo's ``run_one`` always passes ``None``: recording belongs to the outer ``EnvSession``); with
    ``None`` nothing is recorded (only for unit tests that test the trace alone).

    ``effective_cap`` / ``strict_cap`` (when strict, ``TracedEnv.step`` refuses step cap+1 before counting and sets
    ``cap_hit``); ``lang``: a ``trace_writer.LanguageLog`` instance or ``None`` (no language ledger);
    ``source_call_id`` / ``chunk_index``: which ``action_model`` call the current action chunk came from and the index
    of the next action within the chunk; ``action_params`` / ``monitor_params``: ``params`` for the language ledger's
    ``open_call``."""

    def __init__(self, writer, open_recorder: Callable | None = None, *, effective_cap: int | None = None,
                 strict_cap: bool = False, lang=None, action_params: dict | None = None,
                 monitor_params: dict | None = None) -> None:
        self.writer = writer
        self.t = 0
        self.subgoal: str | None = None
        self.attempted = 0  # C8 steps given to the environment (including exception steps)
        self.observed = 0  # C8 steps that returned a valid observation
        self.demo_frames: int | None = None  # demo frame count (excluding the initial frame); None = not reset yet
        self.actions: list = []  # actions actually given to the environment per executed step (original dtype / shape / bytes)
        self.recorder = None
        self._open_recorder = open_recorder
        self.effective_cap = None if effective_cap is None else int(effective_cap)
        self.strict_cap = bool(strict_cap)
        self.cap_hit = False
        self.lang = lang
        self.source_call_id: str | None = None
        self.chunk_index = 0
        self.action_params = dict(action_params or {})
        self.monitor_params = dict(monitor_params or {})
        self._sha_cache: dict = {}

    def ensure_recorder(self):
        if self.recorder is None and self._open_recorder is not None:
            self.recorder = self._open_recorder()
        return self.recorder

    def frame_sha(self, phase: str, idx: int, frames) -> str | None:
        """Raw pixel sha256 of frame ``idx`` of Astra's ``frames`` / ``demo`` list (same algorithm as the trace's
        ``front_sha256``), cached per frame."""
        from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
        key = (phase, int(idx))
        if key not in self._sha_cache:
            self._sha_cache[key] = tw.image_sha256(frames[idx]) if 0 <= idx < len(frames) else None
        return self._sha_cache[key]

    def step_link(self) -> dict:
        """Executed-step link: with the language ledger open, gives ``log_step``'s ``source_call_id`` /
        ``chunk_index``, otherwise empty (without them the step line's key set is byte-identical to the old format).
        Each call increments the in-chunk index."""
        if self.lang is None:
            return {}
        link = {"source_call_id": self.source_call_id, "chunk_index": self.chunk_index}
        self.chunk_index += 1
        return link


def _pack_state(obs, i: int = -1):
    import numpy as np  # noqa: PLC0415
    return np.concatenate([np.asarray(obs["joint_state_list"][i]),
                           np.asarray(obs["gripper_state_list"][i])[:1]]).astype(np.float32)


def _goal_text(goal) -> str | None:
    """Same as how Astra ``runner.episode`` takes the task goal: the first element of a list."""
    if isinstance(goal, list):
        return goal[0] if goal else None
    return goal


class TracedEnv:
    """Wraps the environment given by the builder: reset records the demo segment (C2), step records the post-step
    image, state, action and termination flags (C4, C8).

    The recorder only uses ``add_frames`` / ``add_array``: every frame of the demo segment (including the initial
    frame) and the last frame of each valid observation step go in once each, so the frame count of both streams =
    ``frames_recorded`` = demo frames + 1 + valid observation steps.

    strict cap (ood): the guard comes **before** counting and appending the action -- a call after ``effective_cap``
    executed steps never reaches the real environment, is not recorded and writes no trace line; it sets ``cap_hit``
    and raises ``StepCapReached``. When the inner environment is the outer ``EnvSession`` (via ``SessionEnv``) and its
    ``step_cap`` is reached as well, it is handed to it to refuse once (it raises before reaching the environment
    and sets ``session.cap_hit``).

    In the evaluation repo ``ctx.recorder`` is always ``None``: frames, arrays and events are written by the outer
    ``EnvSession`` into the outer recorder; only the trace is written here."""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def reset(self, *a, **k):
        import numpy as np  # noqa: PLC0415
        obs, info = self._inner.reset(*a, **k)
        ctx = self._ctx
        rec = ctx.ensure_recorder()
        fronts = [np.asarray(x, dtype=np.uint8) for x in obs.get("front_rgb_list", [])]
        wrists = [np.asarray(x, dtype=np.uint8) for x in obs.get("wrist_rgb_list", [])]
        n = min(len(obs.get("joint_state_list", [])), len(obs.get("gripper_state_list", [])))
        states = [_pack_state(obs, i) for i in range(n)]
        if rec is not None and fronts:
            rec.add_frames("front", np.stack(fronts), tag="reset")
            rec.add_frames("wrist", np.stack(wrists), tag="reset")
            for key in ("joint_state_list", "gripper_state_list"):
                if obs.get(key) is not None and len(obs[key]):
                    rec.add_array(f"reset_{key[:-5]}", np.stack([np.asarray(x) for x in obs[key]]))
        goal = _goal_text(info.get("task_goal"))
        ctx.writer.log_demo(fronts, wrists, states, [goal] if goal is not None else [])
        ctx.demo_frames = max(len(fronts) - 1, 0)
        return obs, info

    def step(self, action):
        import numpy as np  # noqa: PLC0415
        ctx = self._ctx
        if ctx.strict_cap and ctx.effective_cap is not None and ctx.attempted >= ctx.effective_cap:
            ctx.cap_hit = True
            rec = ctx.recorder
            if rec is not None and hasattr(rec, "add_event"):
                rec.add_event({"kind": "step_cap_reached", "step": ctx.attempted, "cap": ctx.effective_cap})
            print(f"ASTRA_STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap} rejected_step={ctx.attempted + 1}",
                  flush=True)
            if _session_would_refuse(self._inner):
                # the outer EnvSession has the same cap: let it refuse (never reaching the real environment); it sets
                # ``session.cap_hit`` and records the event, so the outer result's cap_hit / timeout match this driver
                try:
                    self._inner.step(action)
                except Exception as exc:  # noqa: BLE001 EnvSession.StepCapReached
                    raise StepCapReached(f"STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap}") from exc
            raise StepCapReached(f"STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap}")
        rec = ctx.ensure_recorder()
        ctx.attempted += 1
        ctx.t = n = ctx.attempted
        a = np.array(action, copy=True)
        ctx.actions.append(a)
        link = ctx.step_link()
        if rec is not None:
            rec.add_array("exec_action", a, step=n - 1)
        try:
            out = self._inner.step(action)
        except BaseException as exc:
            ctx.writer.log_missing_step(step=n, action=a, reason=f"env_step_exception:{type(exc).__name__}",
                                        subgoal=ctx.subgoal, **link)
            raise
        obs, reward, terminated, truncated, info = out
        status = info.get("status") if isinstance(info, dict) else None
        if obs is None or not obs.get("front_rgb_list") or not obs.get("wrist_rgb_list"):
            ctx.writer.log_missing_step(step=n, action=a, reason="obs_none", subgoal=ctx.subgoal, **link)
            return out
        front = np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8)
        wrist = np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8)
        if rec is not None:
            rec.add_frames("front", front, tag=f"step{n}")
            rec.add_frames("wrist", wrist, tag=f"step{n}")
            rec.add_array("joint_state", np.asarray(obs["joint_state_list"][-1]), step=n - 1)
            rec.add_array("gripper_state", np.asarray(obs["gripper_state_list"][-1]), step=n - 1)
        ctx.observed += 1
        ctx.writer.log_step(step=n, front=front, wrist=wrist, state=_pack_state(obs), action=a, subgoal=ctx.subgoal,
                            terminated=terminated, truncated=truncated, status=status, **link)
        return out

    def close(self):
        return self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _session_would_refuse(inner) -> bool:
    """The inner environment is an outer session with ``step_cap`` whose executed step count is at the cap (another
    ``step`` call would be refused by it before reaching the environment)."""
    cap = getattr(inner, "step_cap", None)
    steps = getattr(inner, "steps", None)
    return isinstance(cap, int) and isinstance(steps, int) and steps >= cap


class TracedBuilder:
    """Only exposes the two methods ``runner.episode`` uses; ``make_env_for_episode`` only passes the episode index
    (in the evaluation repo the inner builder is ``SessionBuilder``)."""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def make_env_for_episode(self, episode):
        return TracedEnv(self._inner.make_env_for_episode(episode), self._ctx)

    def resolve_episode(self, episode):
        return self._inner.resolve_episode(episode)


# -- language ledger (LanguageLog is provided by trace_writer) ----------------------------

#: image transform descriptions: lossless PNG (single-frame attachments of planning / monitoring), ``core.sheets``
#: contact sheets (JPEG q92), raw websocket arrays (VLA)
PNG_TRANSFORM = {"resize": None, "crop": None, "layout": None, "encode": "png"}
SHEET_TRANSFORM = {"resize": None, "crop": None, "layout": "core.sheets 4x4 grid, 256x280 cell, frame label",
                   "encode": "jpeg q92"}
RAW_TRANSFORM = {"resize": None, "crop": None, "layout": None, "encode": "msgpack_numpy"}
#: frames per page of Astra contact sheets (``core.sheets``)
SHEET_PAGE = 16


def language_log_cls():
    """Optional probe: ``trace_writer.LanguageLog``; returns ``None`` when missing (no language ledger, no error)."""
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    return getattr(tw, "LanguageLog", None)


def _image_ref(slot: int, ref: str, phase: str, frame_idx, cam: str, raw_sha: str | None, *, transform: dict,
               sources: list | None = None, encoded: Path | None = None) -> dict:
    """Image reference (the image itself is not stored): a single frame's ``sources`` is itself; a contact sheet gives
    an ordered list of source frames and ``frame_idx=None``."""
    if sources is None:
        sources = [{"phase": phase, "frame_idx": frame_idx, "cam": cam, "raw_sha256": raw_sha}]
    enc = file_sha256(encoded) if encoded is not None and Path(encoded).is_file() else None
    return {"slot": slot, "ref": ref, "phase": phase, "frame_idx": frame_idx, "cam": cam, "raw_sha256": raw_sha,
            "sources": sources, "transform": dict(transform), "encoded_sha256": enc}


def _sheet_sources(ctx: TraceContext, phase: str, indices: list, frames) -> list:
    return [{"phase": phase, "frame_idx": int(i), "cam": "front", "raw_sha256": ctx.frame_sha(phase, int(i), frames)}
            for i in indices]


def planner_images(ctx: TraceContext, out: Path, request: dict, frames, demo, *, wrist=None,
                   command_start: int | None = None) -> list:
    """Image references of a planning / review request, in the order of ``images`` in ``request.json``.

    exec frame number = index into Astra ``frames`` (0 = the initial frame after reset, i.e. the last frame of the
    trace demo segment; k = after executing step k, i.e. trace step k); demo frame number = index into the demo
    segment. Contact sheet page p covers items ``16p`` to ``16p+15`` of the index list."""
    now = len(frames) - 1
    memory = [int(i) for i in request.get("memory_frame_ids") or []]
    demo_idx = list(range(len(demo or [])))
    images = []
    for slot, name in enumerate(request.get("images") or []):
        path = out / name
        if name == "current.png":
            images.append(_image_ref(slot, "current", "exec", now, "front", ctx.frame_sha("exec", now, frames),
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name == "execution_start.png":
            images.append(_image_ref(slot, "keyframe", "exec", 0, "front", ctx.frame_sha("exec", 0, frames),
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name == "command_start.png" and command_start is not None:
            images.append(_image_ref(slot, "command_start", "exec", int(command_start), "front",
                                     ctx.frame_sha("exec", int(command_start), frames), transform=PNG_TRANSFORM,
                                     encoded=path))
        elif name == "current_wrist.png":
            from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
            images.append(_image_ref(slot, "wrist", "exec", now, "wrist",
                                     tw.image_sha256(wrist) if wrist is not None else None,
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name.startswith(("demo_", "memory_")):
            phase, ref = ("demo", "demo_sheet") if name.startswith("demo_") else ("exec", "memory_sheet")
            page = int(Path(name).stem.split("_")[1])
            pool = demo_idx if phase == "demo" else memory
            src_frames = demo if phase == "demo" else frames
            idx = pool[page * SHEET_PAGE:(page + 1) * SHEET_PAGE]
            images.append(_image_ref(slot, ref, phase, None, "front", None, transform=SHEET_TRANSFORM,
                                     sources=_sheet_sources(ctx, phase, idx, src_frames), encoded=path))
        else:  # unknown attachment: only record the file hash, never guess the source
            images.append(_image_ref(slot, None, "exec", None, "front", None, transform=PNG_TRANSFORM, sources=[],
                                     encoded=path))
    return images


class PlannerLanguageCall:
    """Language ledger recorder for one planning (or second-button review) request.

    ``wrap(responder)`` returns the sender used by Astra's ``Planner``: the Planner calls it after writing
    ``prompt.txt`` / ``request.json`` / images, and it **first** opens the call and persists the ``in`` message (full
    prompt + image references) before handing over to the real sender; if the sender supports ``transport_hook``
    (this file's ``GuardedResponsesClient``), every transport retry (attempt >= 1) records the previous call as
    ``error``, opens another call with ``transport_attempt=k`` and rewrites the ``in`` message. ``finish`` writes the
    raw reply and finalizes after the Planner returns or raises."""

    def __init__(self, ctx: TraceContext, image_fn: Callable[[Path, dict], list], kind: str) -> None:
        self.ctx = ctx
        self.image_fn = image_fn
        self.kind = kind
        self.call_id: str | None = None
        self.out: Path | None = None
        self.prompt: str | None = None
        self.images: list | None = None
        self.params: dict | None = None
        self.step = ctx.t

    def _open(self, transport_attempt: int) -> None:
        lang = self.ctx.lang
        self.call_id = lang.open_call("planner", self.step, params=self.params, transport_attempt=transport_attempt)
        lang.message(self.call_id, dir="in", role="user", text=self.prompt, images=self.images)

    def wrap(self, inner: Callable) -> Callable:
        def responder(out):
            out = Path(out)
            self.out = out
            request = json.loads((out / "request.json").read_text())
            self.prompt = (out / "prompt.txt").read_text()
            self.images = self.image_fn(out, request)
            self.params = {"temperature": None, "max_tokens": PLANNER_MAX_OUTPUT_TOKENS,
                           "model_id": request.get("model"), "adapter_sha": None, "effort": request.get("effort"),
                           "kind": request.get("kind", self.kind), "request_id": out.name, "cloud_seed": None,
                           "policy_seed": self.ctx.action_params.get("policy_seed")}
            self._open(0)  # persist before sending
            has_hook = hasattr(inner, "transport_hook")
            if has_hook:
                previous = inner.transport_hook
                inner.transport_hook = self.transport
            try:
                return inner(out)
            finally:
                if has_hook:
                    inner.transport_hook = previous
        return responder

    def transport(self, out, attempt: int) -> None:
        """Called by ``GuardedResponsesClient._send`` before each transport attempt; attempt 0 is the call already
        opened by ``wrap``."""
        if int(attempt) == 0 or self.call_id is None:
            return
        self.ctx.lang.close_call(self.call_id, status="error")
        self._open(int(attempt))

    def finish(self, *, parsed=None, fallback=None, failed: bool = False) -> None:
        if self.call_id is None:
            return
        lang = self.ctx.lang
        response = None
        if self.out is not None and (self.out / "response.json").is_file():
            try:
                response = json.loads((self.out / "response.json").read_text())
            except ValueError:
                response = None
        replied = bool(response) and response.get("status") == "ok"
        if replied:
            lang.message(self.call_id, dir="out", role="assistant", text=response.get("text"))
            if failed and fallback is None:  # a reply that violates the convention (e.g. a review that is not true/false)
                fallback = "model_response_error"
        lang.close_call(self.call_id, status="reply" if replied else "error", parsed=parsed, fallback=fallback)
        self.call_id = None


def _monitor_images(ctx: TraceContext, ids, frames, command_start: int, wrist) -> list:
    """The monitor's 10 images (order of ``input_contract.build_input``): the latest 8 frames, the start frame of the
    current command, and the current wrist frame."""
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    images = [_image_ref(slot, "recent", "exec", int(fid), "front", ctx.frame_sha("exec", int(fid), frames),
                         transform=PNG_TRANSFORM) for slot, fid in enumerate(ids or [])]
    images.append(_image_ref(len(images), "command_start", "exec", int(command_start), "front",
                             ctx.frame_sha("exec", int(command_start), frames), transform=PNG_TRANSFORM))
    images.append(_image_ref(len(images), "wrist", "exec", len(frames) - 1, "wrist",
                             tw.image_sha256(wrist) if wrist is not None else None, transform=PNG_TRANSFORM))
    return images


class TracedClient:
    """Wraps the VLA websocket client: records the normalized request and full action chunk of every ``infer``; with
    the language ledger open each ``infer`` records one ``action_model`` call (``in``: raw ``prompt`` /
    ``grounded_subgoal`` / ``simple_subgoal`` fields and references to the two current frames, persisted before
    sending; Astra's VLA server has no audit reply, so ``server_final_text=None``)."""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def reset(self):
        self._ctx.writer.log_request("vla_reset", b"", step=self._ctx.t)
        return self._inner.reset()

    def _open_lang(self, element) -> str | None:
        ctx = self._ctx
        if ctx.lang is None:
            return None
        from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
        call_id = ctx.lang.open_call("action_model", ctx.t, params=dict(ctx.action_params))
        fields = {k: element.get(k) for k in ("prompt", "grounded_subgoal", "simple_subgoal") if k in element}
        images = [_image_ref(0, "current", "exec", ctx.t, "front", tw.image_sha256(element.get("observation/image")),
                             transform=RAW_TRANSFORM),
                  _image_ref(1, "wrist", "exec", ctx.t, "wrist",
                             tw.image_sha256(element.get("observation/wrist_image")), transform=RAW_TRANSFORM)]
        ctx.lang.message(call_id, dir="in", role="fields", text=fields, images=images)
        return call_id

    def infer(self, element):
        ctx = self._ctx
        ctx.subgoal = element.get("grounded_subgoal")
        call_id = self._open_lang(element)
        ctx.source_call_id, ctx.chunk_index = call_id, 0
        ctx.writer.log_request("vla_infer", _canonical(element), step=ctx.t)
        try:
            out = self._inner.infer(element)
        except BaseException:
            if call_id is not None:
                ctx.lang.close_call(call_id, status="error")
            raise
        if call_id is not None:
            ctx.lang.close_call(call_id, status="reply", server_final_text=None, server_truncated=None)
        ctx.writer.log_response(out.get("actions"), step=ctx.t)
        return out


def _spool_payload(out: Path) -> bytes:
    """Normalized bytes of a planning request: ``request.json`` (without id / created) + ``prompt.txt`` + the sha256 of
    each image's bytes."""
    request = json.loads((out / "request.json").read_text())
    images = {name: file_sha256(out / name) for name in request.get("images", []) if (out / name).is_file()}
    request = {k: v for k, v in request.items() if k not in ("id", "created")}
    prompt = (out / "prompt.txt").read_text() if (out / "prompt.txt").is_file() else ""
    return json.dumps({"request": request, "prompt": prompt, "images": images}, sort_keys=True,
                      ensure_ascii=False).encode()


class TracedPlanner:
    """Wraps Astra's ``Planner``: every planning / review request records one request line from the spool directory
    content; with the language ledger open ``Planner.responder`` is temporarily replaced by the sender from
    ``PlannerLanguageCall.wrap`` (restored when the call ends)."""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def _log(self, name: str, rid: str) -> None:
        out = Path(self._inner.spool) / rid
        if (out / "request.json").is_file():
            self._ctx.writer.log_request(name, _spool_payload(out), step=self._ctx.t)

    def _call(self, method: str, image_fn: Callable, parse: Callable, a, k):
        ctx = self._ctx
        fn = getattr(self._inner, method)
        responder = getattr(self._inner, "responder", None)
        if ctx.lang is None or responder is None:
            return fn(*a, **k)
        call = PlannerLanguageCall(ctx, image_fn, method)
        self._inner.responder = call.wrap(responder)
        try:
            result = fn(*a, **k)
        except BaseException:
            call.finish(failed=True)
            raise
        finally:
            self._inner.responder = responder
        parsed, fallback = parse(result)
        call.finish(parsed=parsed, fallback=fallback)
        return result

    def predict(self, task, goal, frames, demo, memory, completed, issued, episode, t):
        a = (task, goal, frames, demo, memory, completed, issued, episode, t)
        result = self._call(
            "predict", lambda out, req: planner_images(self._ctx, out, req, frames, demo),
            lambda r: (r[0], None if r[0] is not None else "continue_last"), a, {})
        self._log("planner", result[1])
        return result

    def review_second_button(self, goal, frames, wrist, subgoal, command_start, completed, issued, episode, t):
        a = (goal, frames, wrist, subgoal, command_start, completed, issued, episode, t)
        result = self._call(
            "review_second_button",
            lambda out, req: planner_images(self._ctx, out, req, frames, [], wrist=wrist, command_start=command_start),
            lambda r: (r[0], None), a, {})
        self._log("planner_review", result[1])
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TracedMonitor:
    """Wraps the monitor: records ``input.json`` (image paths replaced by each image's sha256). With the language
    ledger open each ``predict`` records one ``monitor`` call: **before** inference, system and user (10 image
    references) are persisted per upstream ``input_contract.from_observations``; afterwards the raw reply
    (``text`` of ``response.json``) and the ``parsed`` boolean are written."""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def _open_lang(self, task, goal, subgoal, frames, command_start, wrist) -> str | None:
        ctx = self._ctx
        if ctx.lang is None:
            return None
        try:
            import input_contract  # noqa: PLC0415  Astra's monitor input contract (same function as the first line of Monitor.predict)
            sample, ids = input_contract.from_observations(task, goal, subgoal, frames, command_start, wrist)
            system, user = sample["messages"][0]["content"], sample["messages"][1]["content"]
        except Exception:  # noqa: BLE001 if construction fails upstream predict fails the same way; still persist a readable input first
            ids, system = None, None
            user = json.dumps({"task": task, "goal": goal, "subgoal": subgoal, "command_start": command_start},
                              ensure_ascii=False)
        call_id = ctx.lang.open_call("monitor", ctx.t, params=dict(ctx.monitor_params))
        ctx.lang.message(call_id, dir="in", role="system", text=system)
        images = _monitor_images(ctx, ids, frames, command_start, wrist) if ids is not None else []
        ctx.lang.message(call_id, dir="in", role="user", text=user, images=images)
        return call_id

    def _close_lang(self, call_id: str, out: Path, pred, failed: bool) -> None:
        lang = self._ctx.lang
        text = None
        response = Path(out) / "response.json"
        if response.is_file():
            try:
                text = json.loads(response.read_text()).get("text")
            except ValueError:
                text = None
        if text is not None:
            lang.message(call_id, dir="out", role="assistant", text=text)
        if failed:
            lang.close_call(call_id, status="reply" if text is not None else "error", parsed=None,
                            fallback="model_response_error" if text is not None else None)
        else:
            lang.close_call(call_id, status="reply", parsed=pred)

    def predict(self, task, goal, subgoal, frames, command_start, wrist, out):
        call_id = self._open_lang(task, goal, subgoal, frames, command_start, wrist)
        try:
            result = self._inner.predict(task, goal, subgoal, frames, command_start, wrist, out)
        except BaseException:
            if call_id is not None:
                self._close_lang(call_id, out, None, failed=True)
            raise
        if call_id is not None:
            self._close_lang(call_id, out, result[0], failed=False)
        sample_path = Path(out) / "input.json"
        if sample_path.is_file():
            sample = json.loads(sample_path.read_text())
            sample["images"] = [file_sha256(Path(p)) if Path(p).is_file() else None for p in sample.get("images", [])]
            payload = json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()
        else:
            payload = _canonical({"task": task, "goal": goal, "subgoal": subgoal, "command_start": command_start})
        self._ctx.writer.log_request("monitor", payload, step=self._ctx.t)
        return result



# -- one episode: runner.episode + Traced* wrappers + finalization -----------------------

def terminal_of(result: dict | None) -> str:
    """The ``status`` of an Astra result is already success / fail / timeout / error (``runner.episode`` folds other
    environment statuses into error); a loop that ran all ``max_steps`` and an environment-reported timeout are both
    ``timeout``; everything else (driver exceptions, no result) is ``error``."""
    status = (result or {}).get("status")
    return status if status in TERMINALS else "error"


def episode_key(task: str, identity: dict) -> str:
    """Episode key: ``<task>_<tier>_<seed>`` (same form as ``EpisodeSpec.key``)."""
    return f"{task}_{identity['tier']}_{int(identity['seed'])}"


def strict_cap_of(dataset: str) -> bool:
    """ood is strict, hard-verify is not; unknown datasets count as not strict (the pairing check already refused
    them)."""
    return bool(DATASET_STRICT_CAP.get(dataset, False))


def write_exec_actions(path: Path, actions: list) -> None:
    """Write the original action of each executed step as ``exec_action__%05d`` (0-based step index); once written
    every step has a key.

    The only allowed way to write ``arrays.npz`` is ``trace_writer.merge_write_npz`` (merged with the same keys
    written by ``TraceWriter.close`` and the outer recorder after checking item by item, replaced atomically); without
    that function the old ``np.savez`` is kept."""
    import numpy as np  # noqa: PLC0415
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    if not actions:
        return
    mapping = {f"exec_action__{i:05d}": a for i, a in enumerate(actions)}
    merge = getattr(tw, "merge_write_npz", None)
    if merge is not None:
        merge(Path(path), mapping)
    else:
        np.savez(path, **mapping)


def prompt_digests(champ: Path) -> dict[str, str]:
    """sha256 of ``prompts/*.md`` and ``prompts/index.json`` (keys are paths relative to ``examples/champ``)."""
    prompts = Path(champ) / "prompts"
    files = sorted(prompts.glob("*.md")) + ([prompts / "index.json"] if (prompts / "index.json").is_file() else [])
    return {str(p.relative_to(champ)): file_sha256(p) for p in files}


def write_provenance(path: Path, *, astra, args, key: str, strict_cap: bool, language: bool, attempt: int) -> dict:
    """Attempt-directory provenance manifest: sha256 of the Astra source and prompts, model seed (the cloud has no
    seed interface, ``cloud_seed=null``), effective cap."""
    doc = {"route": ROUTE, "key": key, "attempt": int(attempt), "dataset": args.dataset,
           "policy_seed": getattr(args, "policy_seed", None), "server_seed": getattr(args, "policy_seed", None),
           "cloud_seed": None, "cloud_seed_note": "planner/monitor cloud API has no seed interface; not fabricated",
           "effective_cap": int(args.max_steps), "strict_cap": bool(strict_cap),
           "astra_root": str(getattr(astra, "root", "")), "astra_sources": source_digests(astra.champ),
           "prompts": prompt_digests(astra.champ), "language_log": "language.jsonl" if language else None}
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return doc


def group_stop_file(group_dir: Path) -> Path:
    """Stop point ``group_*/STOP.json`` (written by the cost guard at the limit; checked by ``ResponsesClient._send``
    and every episode ``reset``)."""
    return Path(group_dir) / "STOP.json"


def run_one(args, task, ep, identity, builder, monitor, planner, client, astra, *, attempt: int, trace_dir: Path,
            strict_cap: bool | None = None) -> dict:
    """One episode: create the attempt directory ``<args.output>/<task>/ep<NNN>/<key>.a<attempt>/`` and
    ``trace_dir/trace.jsonl``, call Astra's ``runner.episode`` through the delegating wrappers, and finalize uniformly
    in ``finally``: write ``arrays.npz`` -> finalize the language ledger -> trace ``close`` -> append the mapping to
    Astra's ``result.json``. Driver exceptions (``BaseException`` not caught by ``runner.episode`` itself) are
    re-raised after finalization."""
    from robomme_ood_eval.record import trace_writer as tw  # noqa: PLC0415
    ep_dir = Path(args.output) / task / f"ep{ep:03d}"
    key = episode_key(task, identity)
    a_dir = ep_dir / f"{key}.a{int(attempt)}"
    a_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    policy_seed = getattr(args, "policy_seed", None)
    strict = strict_cap_of(args.dataset) if strict_cap is None else bool(strict_cap)
    trace_identity = {"task": task, "dataset": args.dataset, **identity, "builder_episode": int(ep), "key": key,
                      "attempt": int(attempt), "policy_seed": policy_seed}
    header_extra = _supported_kwargs(tw.TraceWriter, policy_seed=policy_seed, effective_cap=int(args.max_steps),
                                     strict_cap=strict)
    writer = tw.TraceWriter(trace_dir / "trace.jsonl", route=ROUTE, identity=trace_identity, max_steps=args.max_steps,
                            **header_extra)
    lang_cls = language_log_cls()
    lang = lang_cls(trace_dir / "language.jsonl") if lang_cls is not None else None
    write_provenance(a_dir / PROVENANCE_FILE, astra=astra, args=args, key=key, strict_cap=strict,
                     language=lang is not None, attempt=attempt)
    ctx = TraceContext(writer, open_recorder=None, effective_cap=int(args.max_steps), strict_cap=strict, lang=lang,
                       action_params={"temperature": None, "max_tokens": None, "model_id": "mme_vla_suite",
                                      "adapter_sha": None, "checkpoint": str(args.vla_checkpoint),
                                      "policy_seed": policy_seed},
                       monitor_params={**MONITOR_PARAMS, "model_id": getattr(args, "monitor_base", None),
                                       "adapter": str(args.monitor_adapter), "adapter_sha": None})
    result: dict | None = None
    driver_error: BaseException | None = None
    try:
        result = astra.runner.episode(args, task, ep, TracedBuilder(builder, ctx), TracedMonitor(monitor, ctx),
                                      TracedPlanner(planner, ctx), TracedClient(client, ctx))
        return result
    except BaseException as exc:
        driver_error = exc
        raise
    finally:
        _finish_one(args, ep_dir, a_dir, trace_dir, key, int(attempt), ctx, writer, result, driver_error)


def _supported_kwargs(fn: Callable, **candidates) -> dict:
    """Optional probe: only pass the keywords explicitly declared in ``fn``'s signature."""
    import inspect  # noqa: PLC0415
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in candidates.items() if k in params}


def _finish_one(args, ep_dir: Path, a_dir: Path, trace_dir: Path, key: str, attempt: int, ctx: TraceContext,
                writer, result: dict | None, driver_error: BaseException | None) -> None:
    terminal = terminal_of(result) if driver_error is None else "error"
    if ctx.cap_hit and driver_error is None:  # strict cap refused step cap+1: the error Astra recorded becomes timeout
        terminal = "timeout"
    has_demo = ctx.demo_frames is not None
    no_frame = not has_demo
    demo_frames = ctx.demo_frames if has_demo else 0
    frames_recorded = demo_frames + 1 + ctx.observed if has_demo else 0
    write_exec_actions(trace_dir / "arrays.npz", ctx.actions)
    if ctx.lang is not None:
        ctx.lang.close()  # calls still open get cancelled
    extra = {"policy_seed": getattr(args, "policy_seed", None), "cloud_seed": None, "strict_cap": ctx.strict_cap,
             "effective_cap": ctx.effective_cap, "cap_hit": ctx.cap_hit,
             "language": "language.jsonl" if ctx.lang is not None else None, "provenance": PROVENANCE_FILE}
    if result is not None:
        extra.update({k: result.get(k) for k in ("planner_calls", "monitor_calls", "review_calls", "error")})
    if driver_error is not None:
        extra["driver_exception"] = f"{type(driver_error).__name__}: {driver_error}"[:800]
    if no_frame and terminal != "error":  # should not happen in theory: ended normally without reset; record error to avoid a fake terminal status
        terminal = "error"
    writer.close(status=terminal, terminal_reason=terminal, demo_frames=demo_frames, steps_attempted=ctx.attempted,
                 steps_observed=ctx.observed, frames_recorded=frames_recorded, omitted_timeout_frames=0,
                 no_frame=no_frame, **extra)
    if result is not None:
        rel = lambda p: os.path.relpath(p, ep_dir)  # noqa: E731
        if ctx.cap_hit and driver_error is None:
            result["status"] = "timeout"  # stop rule (2) does not count a strict cap as an infrastructure error
        result.update(route=ROUTE, key=key, attempt=int(attempt), episode_dir=a_dir.name,
                      trace=rel(trace_dir / "trace.jsonl"), exec_steps=ctx.attempted, steps_observed=ctx.observed,
                      frames_recorded=frames_recorded, demo_frames=demo_frames, terminal_reason=terminal,
                      policy_seed=getattr(args, "policy_seed", None), cloud_seed=None,
                      strict_cap=ctx.strict_cap, effective_cap=ctx.effective_cap, cap_hit=ctx.cap_hit)
        if (ep_dir / "result.json").is_file():
            from core import atomic_json  # noqa: PLC0415  Astra's atomic write (the same function it uses for result.json)
            atomic_json(ep_dir / "result.json", result)


# -- hard cost limit: synchronous reservation against guard state + ResponsesClient subclass --

_GUARD_MOD = None


def guard_module():
    """The cost guard module ``robomme_ood_eval.servers.astra_cost_guard`` (the single source of the reservation
    protocol, locks and file names)."""
    global _GUARD_MOD
    if _GUARD_MOD is None:
        from robomme_ood_eval.servers import astra_cost_guard  # noqa: PLC0415
        _GUARD_MOD = astra_cost_guard
    return _GUARD_MOD


class GuardRefused(RuntimeError):
    """Refused before sending (guard lost, guard stopped, reservation failed, episode limit reached). The message
    starts with ``Planner API``: once wrapped by the Planner into ``Planner bridge failed`` it likewise triggers stop
    rule (1) (``PLANNER_STOP_PREFIXES``) and stops at once."""


def _image_tokens_upper(width: int, height: int) -> int:
    """Conservative upper bound of input tokens for one ``detail=high`` image: the larger of two public pricing
    schemes.

    (1) 512 tiles: first fit within 2048x2048, then scale the short side to 768, ``85 + 170 x tiles``;
    (2) 32-pixel patches: number of patches (capped at 1536) x 2.5 (the largest per-model multiplier, rounded up)."""
    import math  # noqa: PLC0415
    w, h = max(int(width), 1), max(int(height), 1)
    scale = min(1.0, 2048 / max(w, h))
    w1, h1 = w * scale, h * scale
    scale2 = min(1.0, 768 / min(w1, h1))
    w2, h2 = w1 * scale2, h1 * scale2
    tiles = math.ceil(w2 / 512) * math.ceil(h2 / 512)
    patches = min(math.ceil(w / 32) * math.ceil(h / 32), 1536)
    return max(85 + 170 * tiles, math.ceil(patches * 2.5))


def _image_size(data_url: str) -> tuple[int, int]:
    """Decode the image size from ``data:<mime>;base64,<...>``; if it cannot be decoded assume 2048x2048 (worst
    case)."""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=False)
        from PIL import Image  # noqa: PLC0415
        with Image.open(io.BytesIO(raw)) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return 2048, 2048


def estimate_request(payload: dict) -> dict:
    """Estimate an upper bound on input tokens from the actual request: text by UTF-8 byte count (at least 1 byte per
    token), images by ``_image_tokens_upper``, plus 64 for formatting overhead; output by the request's own
    ``max_output_tokens``."""
    text_bytes, images = 0, []
    for item in payload.get("input") or []:
        for part in item.get("content") or []:
            if part.get("type") == "input_text":
                text_bytes += len(str(part.get("text", "")).encode("utf-8"))
            elif part.get("type") == "input_image":
                images.append(_image_tokens_upper(*_image_size(str(part.get("image_url", "")))))
    return {"input_tokens": text_bytes + sum(images) + 64, "text_bytes": text_bytes, "images": len(images),
            "image_tokens": sum(images), "max_output_tokens": int(payload.get("max_output_tokens") or 0)}


class CostGate:
    """The runner side of the guard protocol: synchronously read the guard state and atomically reserve the
    single-request worst-case cost before sending; register episodes.

    The state file is written every round by ``astra_cost_guard.py --state``; the reservation file and lock are named
    by the guard module. Sending is refused whenever the state cannot be read, is stale (heartbeat older than
    ``GUARD_HEARTBEAT_TIMEOUT_S``), or the guard has exited or stopped -- better to stop than to send."""

    def __init__(self, state_path: str | Path, *, clock: Callable[[], float] = time.time,
                 heartbeat_timeout: float = GUARD_HEARTBEAT_TIMEOUT_S) -> None:
        self.state_path = Path(state_path)
        self.clock = clock
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.guard = guard_module()

    def read_state(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError) as exc:
            raise GuardRefused(f"Planner API cost guard unavailable: state unreadable ({type(exc).__name__})") from None
        if state.get("schema") != self.guard.STATE_SCHEMA:
            raise GuardRefused(f"Planner API cost guard unavailable: schema {state.get('schema')!r}")
        age = self.clock() - float(state.get("heartbeat") or 0)
        if state.get("exited"):
            raise GuardRefused("Planner API cost guard unavailable: guard exited")
        if not -1.0 <= age <= self.heartbeat_timeout:
            raise GuardRefused(f"Planner API cost guard unavailable: heartbeat age {age:.1f}s > {self.heartbeat_timeout:g}s")
        if state.get("stop"):
            raise GuardRefused(f"Planner API cost guard stopped: {state.get('reason')}")
        return state

    def cap(self, state: dict) -> float:
        """Effective limit = min(the guard's --cap, the hard limit); a guard state edited upward never relaxes it."""
        return min(float(state.get("cap") or 0.0), float(self.guard.HARD_CAP_USD))

    def worst_usd(self, state: dict, estimate: dict) -> float:
        prices = state["prices"]
        out_tokens = max(int(estimate["max_output_tokens"]), int(state.get("max_output_tokens") or 0))
        usd = (estimate["input_tokens"] * float(prices["input"]) + out_tokens * float(prices["output"])) / 1e6
        if prices.get("reasoning_billed_separately"):
            usd += out_tokens * float(prices["output"]) / 1e6
        return usd

    def reserve(self, rid: str, payload: dict) -> dict:
        """Inside the lock: read the state -> refuse if counted + outstanding reservations + this worst case > limit;
        otherwise record the reservation (retries of the same rid are not counted twice; the larger is kept)."""
        estimate = estimate_request(payload)
        g = self.guard
        with g.locked(self.state_path):
            state = self.read_state()
            doc = g.load_reservations(self.state_path)
            worst = self.worst_usd(state, estimate)
            existing = doc["reservations"].get(rid)
            others = g.outstanding_reservations(
                {"reservations": {k: v for k, v in doc["reservations"].items() if k != rid}}, state.get("counted") or [])
            committed = float(state.get("committed_usd") or 0.0)
            amount = max(worst, float(existing["usd"])) if existing and existing.get("state") != "released" else worst
            cap = self.cap(state)
            if committed + others + amount > cap:
                raise GuardRefused(f"Planner API cost reservation refused: committed={committed:.4f} "
                                   f"reserved={others:.4f} worst={amount:.4f} cap={cap:g}")
            doc["reservations"][rid] = {"usd": amount, "state": "reserved", "time": self.clock(), **estimate}
            g.save_reservations(self.state_path, doc)
        return {"usd": amount, **estimate}

    def mark(self, rid: str, state_name: str) -> None:
        g = self.guard
        with g.locked(self.state_path):
            doc = g.load_reservations(self.state_path)
            if rid in doc["reservations"]:
                doc["reservations"][rid]["state"] = state_name
                doc["reservations"][rid][f"{state_name}_at"] = self.clock()
                g.save_reservations(self.state_path, doc)

    def register_episode(self, episode_id: str) -> int:
        """Hard episode limit: registered in the same reservation file across runs; the same episode is not counted
        twice; the third episode is refused."""
        g = self.guard
        with g.locked(self.state_path):
            self.read_state()
            doc = g.load_reservations(self.state_path)
            if episode_id not in doc["episodes"]:
                if len(doc["episodes"]) >= g.ASTRA_MAX_EPISODES:
                    print(f"ASTRA_STOP reason=episode_cap episode={episode_id} used={len(doc['episodes'])}", flush=True)
                    raise AstraStop("episode_cap", f"Astra episode cap {g.ASTRA_MAX_EPISODES} reached")
                doc["episodes"].append(episode_id)
                g.save_reservations(self.state_path, doc)
            return len(doc["episodes"])


def registered_episodes(state_path: str | Path) -> list:
    """Episodes registered in the reservation file (cumulative across processes and runs; the ledger is kept and no
    new one is created to get around the limit)."""
    g = guard_module()
    with g.locked(Path(state_path)):
        return list(g.load_reservations(Path(state_path))["episodes"])


def check_episode_cap(state_path: str | Path) -> int:
    """Hard episode limit: raise ``AstraStop(episode_cap)`` when the registered count reaches
    ``ASTRA_MAX_EPISODES``; returns the registered count."""
    limit = guard_module().ASTRA_MAX_EPISODES
    used = len(registered_episodes(state_path))
    if used >= limit:
        print(f"ASTRA_STOP reason=episode_cap used={used} cap={limit}", flush=True)
        raise AstraStop("episode_cap", f"Astra episode cap {limit} reached")
    return used


_GUARDED_CLASSES: dict = {}


def guarded_client_class(api_client):
    """Return a subclass of the third-party ``api_client.ResponsesClient`` (cached per base class); the third-party
    file is untouched.

    Overrides ``_send``: before every real ``urlopen``, in order, (1) check ``group_*/STOP.json``; (2) synchronously
    read the guard state and atomically reserve this request's worst-case cost (429 retries of the same request reuse
    the same reservation); (3) wait for the third party's 20-second send interval; (4) after waiting, check STOP and
    the guard state again. Any failure writes ``guard_refused.json``, releases the reservation and raises (the
    third-party ``__call__`` records it as a ``status=error`` ``response.json``). Everything else (HTTP error
    recording, bounded 429 backoff, redaction) matches the third-party ``_send`` item by item."""
    base = api_client.ResponsesClient
    if base in _GUARDED_CLASSES:
        return _GUARDED_CLASSES[base]

    class GuardedResponsesClient(base):
        #: language ledger transport-attempt callback ``(out, attempt)`` (``PlannerLanguageCall.transport``); not called
        #: when ``None``. Only records; never changes any decision of sending or the cost guard.
        transport_hook = None

        def __init__(self, key, gate: CostGate, *, sleep: Callable[[float], None] = time.sleep,
                     monotonic: Callable[[], float] = time.monotonic) -> None:
            super().__init__(key)
            self.gate = gate
            self._sleep = sleep
            self._monotonic = monotonic

        @staticmethod
        def _stop_requested(out: Path) -> bool:
            return any(p.name in GROUP_DIR_NAMES and (p / "STOP.json").exists() for p in out.parents)

        def _refuse(self, out: Path, rid: str, message: str):
            api_client.atomic_json(out / guard_module().REFUSED_MARKER, {"time": time.time(), "reason": message})
            try:
                self.gate.mark(rid, "released")
            except Exception:  # noqa: BLE001 releasing may fail too when the guard is lost; the refusal itself is unaffected
                pass
            print(f"ASTRA_GUARD_REFUSED request={rid} reason={json.dumps(message, ensure_ascii=False)}", flush=True)
            raise RuntimeError(message)

        def _gate_before_send(self, request, out: Path, rid: str) -> None:
            if self._stop_requested(out):
                self._refuse(out, rid, "Host requested stop; no new API request")
            try:
                self.gate.reserve(rid, json.loads(request.data))
            except GuardRefused as exc:
                self._refuse(out, rid, str(exc))
            self._sleep(max(0, self.next_request_at - self._monotonic()))
            if self._stop_requested(out):  # STOP may have arrived during the wait
                self._refuse(out, rid, "Host requested stop; no new API request")
            try:
                self.gate.read_state()
            except GuardRefused as exc:
                self._refuse(out, rid, str(exc))

        def _send(self, request, out):
            import random  # noqa: PLC0415
            import urllib.error  # noqa: PLC0415
            import urllib.request  # noqa: PLC0415
            from email.utils import parsedate_to_datetime  # noqa: PLC0415
            out = Path(out)
            rid = out.name
            for attempt in range(9):
                if self.transport_hook is not None:  # language ledger: each transport attempt's input is persisted before the gate and the send
                    self.transport_hook(out, attempt)
                self._gate_before_send(request, out, rid)
                self.next_request_at = self._monotonic() + SEND_INTERVAL_S
                self.gate.mark(rid, "sent")
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        return json.loads(response.read()), response.headers.get("x-request-id")
                except urllib.error.HTTPError as error:
                    body = error.read().decode("utf-8", errors="replace")
                    try:
                        detail = json.loads(body).get("error", {})
                    except ValueError:
                        detail = {}
                    code = detail.get("code") if isinstance(detail, dict) else None
                    kind = detail.get("type") if isinstance(detail, dict) else None
                    headers = {k: v for k, v in error.headers.items()
                               if k.lower() == "retry-after" or k.lower() == "x-request-id"
                               or k.lower().startswith("x-ratelimit-")}
                    api_client.atomic_json(out / f"http_error_{attempt:02d}.json",
                                           json.loads(self.scrub(json.dumps({"status": error.code, "body": body,
                                                                             "headers": headers, "time": time.time()}))))
                    retryable = error.code == 429 and (code in ("rate_limit_exceeded", "slow_down")
                                                       or kind == "rate_limit_error")
                    if not retryable or attempt == 8:
                        raise RuntimeError(self.scrub(f"HTTP {error.code}: {body}")) from None
                    delay = min(120, 10 * 2 ** attempt)
                    retry_after = error.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            try:
                                delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                            except (TypeError, ValueError, OverflowError):
                                pass
                    self._sleep(delay + random.uniform(0, 3))

    _GUARDED_CLASSES[base] = GuardedResponsesClient
    return GuardedResponsesClient



# -- the two servers: cost guard and VLA -------------------------------------------------

class GuardProcess(ServerProcess):
    """Process of the cost guard ``servers/astra_cost_guard.py`` (does not listen on a port).

    ``port`` is only a label in the metadata file name (VLA port + 1, which ``pick_port`` already guarantees is free);
    ready = the guard's heartbeat state file is readable, has the right schema, has not exited and the heartbeat is
    fresh (``CostGate.read_state`` does not raise ``GuardRefused``; a guard that already wrote STOP also counts as
    started); stopping does not check GPU memory (the guard uses no GPU). After TERM the guard writes
    ``exited=true`` and the runner refuses immediately."""

    def __init__(self, *a, state_path: str | Path, **k) -> None:
        super().__init__(*a, **k)
        self.state_path = Path(state_path)

    def _ready_now(self) -> bool:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return False
        g = guard_module()
        if state.get("schema") != g.STATE_SCHEMA or state.get("exited"):
            return False
        return abs(time.time() - float(state.get("heartbeat") or 0)) <= GUARD_HEARTBEAT_TIMEOUT_S

    def stop(self, *, grace_s: float = 60.0, check_gpu: bool = False) -> str:
        return super().stop(grace_s=grace_s, check_gpu=check_gpu)


class VLAServerProcess(ServerProcess):
    """Astra VLA server (the submodule's ``scripts/serve_policy.py``): ``OPENAI_API_KEY`` is removed from the child
    environment (the VLA does not need the planning API key; same as the upstream ``run.sh`` ``env -u
    OPENAI_API_KEY``); otherwise same as ``ServerProcess``."""

    DROP_ENV = ("OPENAI_API_KEY",)

    def _child_env(self) -> dict:
        env = super()._child_env()
        for k in self.DROP_ENV:
            env.pop(k, None)
        return env


# -- outer session -> builder for runner.episode -----------------------------------------

class SessionEnv:
    """Wraps the outer ``EnvSession`` as the environment ``runner.episode`` sees: ``reset`` / ``step`` are forwarded
    unchanged to the session; ``close()`` is a no-op (``runner.episode``'s ``finally`` closes the environment, but the
    environment belongs to the outer loop and the model side must not call ``session.close()``)."""

    def __init__(self, session) -> None:
        self._session = session
        self.close_calls = 0

    def reset(self, *a, **k):
        return self._session.reset(*a, **k)

    def step(self, action):
        return self._session.step(action)

    def close(self) -> None:
        self.close_calls += 1  # no-op: only count, never close

    def __getattr__(self, name):
        return getattr(self._session, name)


class SessionBuilder:
    """Builder for ``runner.episode``: ``make_env_for_episode`` returns the outer session (via ``SessionEnv``,
    ``close()`` a no-op), accepting only this episode's index and handing it out only once; ``resolve_episode``
    delegates to the real builder (used by ``runner.episode`` to write ``identity.json``)."""

    def __init__(self, session, real_builder, episode: int) -> None:
        self._session = session
        self._builder = real_builder
        self._episode = int(episode)
        self.env: SessionEnv | None = None

    def make_env_for_episode(self, episode):
        if int(episode) != self._episode:
            raise ValueError(f"SessionBuilder only hands out this episode's environment: requested {episode}, this episode {self._episode}")
        if self.env is not None:
            raise RuntimeError("SessionBuilder hands out the environment only once per episode")
        self.env = SessionEnv(self._session)
        return self.env

    def resolve_episode(self, episode):
        return self._builder.resolve_episode(episode)


def pin_visible_gpu(gpu) -> str:
    """Pin this process to one GPU (the monitor model's ``device_map={'':0}`` is this GPU; simulation rendering uses
    the same GPU, like the old launcher's ``CUDA_VISIBLE_DEVICES=$MONITOR_GPU``). Refuses when CUDA is already
    initialized with a different visible GPU (changing the environment variable no longer has an effect)."""
    want = str(gpu)
    torch = sys.modules.get("torch")
    try:
        inited = bool(torch is not None and torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001
        inited = False
    if inited and os.environ.get("CUDA_VISIBLE_DEVICES") != want:
        raise RuntimeError(f"CUDA is already initialized in this process (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); "
                           f"cannot pin the monitor model to GPU {want}")
    os.environ["CUDA_VISIBLE_DEVICES"] = want
    return want


# ── AstraPolicy ─────────────────────────────────────────────────────────────

class AstraPolicy(Policy):
    """Astra's four model-side methods (see the module docstring).

    ``cfg`` (model options of ``scripts/evaluate.py``, passed through with hyphens converted to underscores):

    ========================  ============================================================================
    ``gpus``                  required, two different GPUs: the first runs the VLA server, the second this
                              process (monitor model and simulation)
    ``astra_ledger``          required: cost ledger JSON (guard state, reservation file and ``group_0/`` live in
                              the same directory; kept long-term)
    ``astra_prices``          required: unit price config JSON (USD / million tokens, guard ``--prices``)
    ``astra_cap_usd``         cost limit, default 5; anything above 5 is refused (guard ``HARD_CAP_USD``)
    ``ckpt``                  required: VLA weight directory (``.../symbolic-grounded-subgoal/79999``); alias
                              ``astra_vla_checkpoint``
    ``astra_monitor_adapter`` required: monitor LoRA (``checkpoint-2246``)
    ``astra_monitor_base``    monitor base model (default: upstream ``runner.BASE``)
    ``astra_vla_python``      VLA server interpreter (default ``third_party/mme-vla/.venv/bin/python``)
    ``astra_root``            Astra submodule root (default ``third_party/Astra-on-RoboMME``)
    ``astra_group_dir``       stop-point directory (default ``<ledger dir>/group_0``; must be named group_0 /
                              group_1)
    ``port_base``             VLA port base (default 18762)
    ``max_planner_calls``     per-episode planning limit (default 24)
    ``astra_guard_interval``  guard scan interval in seconds (default 2)
    ========================  ============================================================================
    """

    model = "astra"
    episode_wall_s = ASTRA_EPISODE_WALL_S
    #: replaced by fake server classes in tests
    guard_server_cls = GuardProcess
    vla_server_cls = VLAServerProcess

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        self.guard: ServerProcess | None = None
        self.astra: SimpleNamespace | None = None
        self.gate: CostGate | None = None
        self.monitor = self.client = self.planner = self.responder = None
        self.run_dir: Path | None = None
        self.spool: Path | None = None
        self.port: int | None = None
        self._errors = 0
        self._stop: AstraStop | None = None
        self.episode_results: list[dict] = []

    # -- configuration ---------------------------------------------------------
    def _cfg_path(self, *names: str, required: bool = True) -> Path | None:
        for n in names:
            v = self.cfg.get(n)
            if v not in (None, ""):
                return Path(os.path.abspath(Path(str(v)).expanduser()))  # no resolve: the venv interpreter is a symlink
        if required:
            flags = " or ".join("--" + n.replace("_", "-") for n in names)
            raise ValueError(f"RUN_BLOCKED reason=astra_cfg Astra requires {flags}")
        return None

    def _parse_cfg(self) -> None:
        self.policy_seed = check_policy_seed(self.policy_seed)
        gpus = self.cfg.get("gpus")
        if isinstance(gpus, (int, str)):
            gpus = [int(x) for x in str(gpus).split(",") if str(x).strip()]
        gpus = [int(x) for x in (gpus or [])]
        if len(gpus) < 2 or gpus[0] == gpus[1]:
            raise ValueError(f"RUN_BLOCKED reason=astra_gpus Astra needs two different GPUs (--gpus <VLA GPU>,<monitor GPU>), got {gpus}")
        self.vla_gpu, self.monitor_gpu = gpus[0], gpus[1]
        self.ledger = self._cfg_path("astra_ledger")
        self.prices = self._cfg_path("astra_prices")
        g = guard_module()
        self.cap_usd = g.check_cap(float(self.cfg.get("astra_cap_usd", g.HARD_CAP_USD)))
        self.state_path = g.default_state_path(self.ledger)
        self.group_dir = self._cfg_path("astra_group_dir", required=False) or self.ledger.parent / "group_0"
        if self.group_dir.name not in GROUP_DIR_NAMES:
            raise ValueError(f"RUN_BLOCKED reason=layout astra_group_dir must be named one of {GROUP_DIR_NAMES}: {self.group_dir}")
        self.vla_checkpoint = self._cfg_path("ckpt", "astra_vla_checkpoint")
        self.monitor_adapter = self._cfg_path("astra_monitor_adapter")
        self.monitor_base = self.cfg.get("astra_monitor_base")
        self.root = astra_root(self.cfg.get("astra_root"))
        self.vla_python = (self._cfg_path("astra_vla_python", required=False)
                           or third_party_root() / "mme-vla" / ".venv" / "bin" / "python")
        self.port_base = int(self.cfg.get("port_base") or DEFAULT_PORT_BASE)
        self.max_planner_calls = int(self.cfg.get("max_planner_calls") or DEFAULT_MAX_PLANNER_CALLS)
        self.guard_interval = float(self.cfg.get("astra_guard_interval") or guard_module().DEFAULT_INTERVAL_S)
        self.server_dir = self.ledger.parent / "servers"

    # -- replaceable constructors (tests inject stand-ins) ---------------------
    def make_monitor(self, base, adapter):
        return self.astra.runner.Monitor(base, adapter)

    def make_client(self, port: int):
        from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy  # noqa: PLC0415
        import openpi_client  # noqa: PLC0415
        print(f"ASTRA_VLA_CLIENT openpi_client={openpi_client.__file__} port={port}", flush=True)
        return MMEVLAWebsocketClientPolicy("127.0.0.1", port)

    def make_responder(self, key: str, gate: "CostGate"):
        return guarded_client_class(self.astra.api_client)(key, gate)

    # -- server commands -------------------------------------------------------
    def guard_argv(self) -> list[str]:
        return [sys.executable, str(Path(guard_module().__file__).resolve()), "--root", str(self.group_dir),
                "--prices", str(self.prices), "--ledger", str(self.ledger), "--state", str(self.state_path),
                "--cap", f"{self.cap_usd:g}", "--interval", f"{self.guard_interval:g}"]

    def vla_argv(self, port: int) -> list[str]:
        """Copied from upstream ``run.sh``, only replacing the fixed ``--seed`` 42 with ``policy_seed``."""
        return [str(self.vla_python), "scripts/serve_policy.py", f"--port={port}", f"--seed={self.policy_seed}",
                "policy:checkpoint", "--policy.config=mme_vla_suite", f"--policy.dir={self.vla_checkpoint}"]

    def vla_env(self) -> dict:
        """Environment variables of upstream ``run.sh``; the fourth PYTHONPATH entry (environment source) is replaced
        by ``third_party/robomme_benchmark/src``."""
        root = self.root
        return {"PYTHONPATH": os.pathsep.join([str(root / "examples" / "champ"), str(root / "src"),
                                               str(root / "packages" / "openpi-client" / "src"),
                                               str(benchmark_src())]),
                "OPENPI_DATA_HOME": os.environ.get("OPENPI_DATA_HOME", str(Path.home() / ".cache" / "openpi")),
                "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false", "USE_HF": "1", "IMAGE_MAX_TOKEN_NUM": "128",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def _vla_port(self) -> int:
        """If metadata for this base exists (a server left behind after a watchdog exit), keep that port and let
        ``start()`` attach; otherwise take a free port."""
        if (self.server_dir / f"server-metadata-{self.port_base}.json").is_file():
            return self.port_base
        return pick_port(self.port_base)

    # -- the four methods ------------------------------------------------------
    def load(self) -> None:
        self._parse_cfg()
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("RUN_BLOCKED reason=astra_key environment variable OPENAI_API_KEY is empty (the key is only "
                             "read from the environment)")
        self.astra = bootstrap(self.root)
        print("ASTRA_SOURCE root=" + str(self.root) + " " + " ".join(
            f"{k}={v}" for k, v in source_digests(self.astra.champ).items()), flush=True)
        assert_env_sources()
        if self.monitor_base is None:
            self.monitor_base = self.astra.runner.BASE
        self.astra.release_utils.validate_checkpoints(str(self.vla_checkpoint), str(self.monitor_adapter))
        stop = group_stop_file(self.group_dir)
        if stop.exists():  # already stopped: keep the ledger, never create a new one to get around the limit
            print(f"ASTRA_STOP reason=host_stop stop={stop} (already present before load)", flush=True)
            raise AstraStop("host_stop", f"STOP.json already exists: {stop}")
        self.group_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.group_dir / f"run-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        self.spool = self.run_dir / "planner_calls"
        self.run_dir.mkdir(parents=False, exist_ok=False)
        self.port = self._vla_port()
        # (1) cost guard (started first: every paid request afterwards needs its heartbeat and reservations)
        self.guard = self.guard_server_cls(self.guard_argv(), cwd=str(REPO_ROOT), port=self.port + 1,
                                           metadata_dir=self.server_dir, name="astra-guard",
                                           log_path=self.server_dir / "astra-guard.log",
                                           ready_timeout_s=GUARD_READY_TIMEOUT_S, state_path=self.state_path)
        self.guard.start()
        self.gate = CostGate(self.state_path)
        state = self.gate.read_state()
        used = check_episode_cap(self.state_path)
        print(f"ASTRA_GUARD=PASS cap={self.gate.cap(state):g} committed={float(state.get('committed_usd') or 0):.4f} "
              f"episodes_used={used} ledger={self.ledger}", flush=True)
        # (2) VLA server (first GPU)
        self.server = self.vla_server_cls(self.vla_argv(self.port), env=self.vla_env(), cwd=str(self.root),
                                          gpu=self.vla_gpu, ready=[Ready.port()], port=self.port,
                                          metadata_dir=self.server_dir, policy_seed=self.policy_seed,
                                          ckpt=str(self.vla_checkpoint), name="astra-vla",
                                          ready_timeout_s=VLA_READY_TIMEOUT_S)
        self.server.start()
        # (3) this process: load the monitor model on the second GPU; VLA client, planning client (the same instance keeps the 20-second interval) and Planner
        pin_visible_gpu(self.monitor_gpu)
        self.monitor = self.make_monitor(self.monitor_base, str(self.monitor_adapter))
        self.client = self.make_client(self.port)
        self.responder = self.make_responder(key, self.gate)
        self.planner = self.astra.core.Planner(str(self.spool), responder=self.responder)
        print(f"ASTRA_LOAD=PASS policy_seed={self.policy_seed} server_seed={self.policy_seed} cloud_seed=null "
              f"vla_gpu={self.vla_gpu} monitor_gpu={self.monitor_gpu} port={self.port} run_dir={self.run_dir}",
              flush=True)

    def servers(self) -> list[ServerProcess]:
        return [s for s in (self.server, self.guard) if s is not None]

    def reset(self, spec) -> None:
        """Pre-episode refusal point (no server message, no environment access): stop condition left by the previous
        episode -> STOP.json -> liveness of both servers -> dataset / step-cap pairing -> cross-process episode
        registration (the third episode is refused)."""
        if self._stop is not None:
            raise self._stop
        stop = group_stop_file(self.group_dir)
        if stop.exists():
            print(f"ASTRA_STOP reason=host_stop key={spec.key}", flush=True)
            self._stop = AstraStop("host_stop", "Host requested stop before next episode")
            raise self._stop
        self.check_server()
        check_pairing(spec.dataset, spec.max_steps)
        self.gate.register_episode(f"{spec.dataset}:{spec.task}:{spec.episode}")

    def episode_args(self, spec) -> SimpleNamespace:
        """``args`` for ``runner.episode``: ``output`` points at ``astra/`` under this episode's raw directory, and
        ``max_steps`` follows this episode."""
        return SimpleNamespace(output=str(Path(spec.out_dir) / ASTRA_SUBDIR), dataset=spec.dataset,
                               max_steps=int(spec.max_steps), max_planner_calls=self.max_planner_calls,
                               vla_checkpoint=str(self.vla_checkpoint), monitor_adapter=str(self.monitor_adapter),
                               monitor_base=self.monitor_base, policy_seed=self.policy_seed, port=self.port)

    def play(self, session, spec, recorder) -> dict:
        from robomme_ood_eval import episode as E  # noqa: PLC0415
        check_pairing(spec.dataset, spec.max_steps)
        args = self.episode_args(spec)
        ep_dir = Path(args.output) / spec.task / f"ep{int(spec.episode):03d}"
        if ep_dir.exists():
            raise RuntimeError(f"Astra episode directory already exists (keeping the previous attempt's evidence, not overwriting): {ep_dir}")
        identity = {k: v for k, v in spec.identity().items() if k not in ("task", "dataset", "attempt", "key")}
        builder = SessionBuilder(session, E.builder_for(spec.task, spec.dataset), spec.episode)
        try:
            result = run_one(args, spec.task, int(spec.episode), identity, builder, self.monitor, self.planner,
                             self.client, self.astra, attempt=spec.attempt, trace_dir=Path(spec.out_dir),
                             strict_cap=spec.strict_cap)
        except Exception as exc:  # driver exception: count one non-planner error and re-raise unchanged (the outer loop records error)
            self._after_episode({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, spec)
            raise
        self._after_episode(result, spec)
        status = terminal_of(result)
        out = {"status": status, "task_success": int(status == "success"), "steps": int(result.get("steps") or 0),
               "error": result.get("error"), "infra": False, "infra_reason": None,
               "decisions": int(result.get("planner_calls") or 0),
               "planner_calls": int(result.get("planner_calls") or 0),
               "monitor_calls": int(result.get("monitor_calls") or 0),
               "review_calls": int(result.get("review_calls") or 0),
               "astra_dir": os.path.relpath(ep_dir, spec.out_dir), "cloud_seed": None,
               "timing": {"astra_seconds": result.get("seconds")}}
        if builder.env is not None:
            out["astra_env_close_calls"] = builder.env.close_calls
        return out

    def _after_episode(self, result: dict, spec) -> None:
        """Upstream ``main()`` items 4 and 5 plus the episode limit: judged after play returns, the stop condition is
        recorded and raised by the next episode's ``reset``."""
        self.episode_results.append(result)
        failed = result.get("status") == "error" and not self.astra.core.is_planner_failure(result)
        self._errors = self._errors + 1 if failed else 0
        if result.get("status") == "error" and str(result.get("error") or "").startswith(PLANNER_STOP_PREFIXES):
            print(f"ASTRA_STOP reason=planner_error key={spec.key}", flush=True)
            self._stop = AstraStop("planner_error", "Stopping pilot after planner service or budget error")
        elif self._errors >= INFRA_ERROR_LIMIT:
            print(f"ASTRA_STOP reason=infra_errors key={spec.key} errors={self._errors}", flush=True)
            self._stop = AstraStop("infra_errors", "Three consecutive infrastructure/protocol errors; stopping shard")
        else:
            try:
                check_episode_cap(self.state_path)
            except AstraStop as exc:
                self._stop = exc

    def close(self) -> None:
        """Stop the VLA server process group first, then the guard (which writes exited=true); release the monitor
        model. Idempotent."""
        if self.closed:
            return
        self.closed = True
        for srv in self.servers():
            try:
                srv.stop()
            except Exception as exc:  # noqa: BLE001 failing to stop one must not prevent stopping the other
                print(f"ASTRA_CLOSE_WARN name={getattr(srv, 'name', '?')} error={type(exc).__name__}: {exc}", flush=True)
        self.monitor = self.client = self.planner = self.responder = None


__all__ = ["AstraPolicy", "AstraStop", "StepCapReached", "SessionBuilder", "SessionEnv", "GuardProcess",
           "VLAServerProcess", "CostGate", "GuardRefused", "guarded_client_class", "run_one", "check_pairing",
           "check_policy_seed", "bootstrap", "assert_env_sources"]
