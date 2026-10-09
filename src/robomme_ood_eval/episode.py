"""The outer once-per-episode ``run_episode`` and its two data structures ``EpisodeSpec`` and ``EpisodeResult``.

Order of ``run_episode(policy, dataset, task, episode, out_dir, expect=None) -> EpisodeResult`` (each step pinned by
unit tests):

1. Build the ``EpisodeSpec`` and check the identity: ``BenchmarkEnvBuilder.resolve_identity(episode)`` (builders are
   cached in-process per ``(task, dataset)``); when ``expect`` (the identity row from the queue) is given, compare
   tier / seed / candidate / spec_sha256 / source_episode key by key via ``check_identity``; on mismatch write a
   result with ``error=IDENTITY_MISMATCH`` and raise ``IdentityMismatch``;
2. ``policy.reset(spec)``: if it raises ``ServerDead`` / ``AstraStop``, write no result and re-raise unchanged (the
   episode has not started; a resumed run will redo it);
3. Create the recorder and the ``EnvSession`` and call ``session.build()`` first; on failure record
   ``infra=True infra_reason=env_build`` and skip play. Arm the per-episode watchdog (wall clock, first-inference
   deadline, media-finalize deadline);
4. ``policy.play(session, spec, recorder)``;
5. ``finally``: ``session.close()`` -> ``recorder.close()`` -> (when there is a trace) render the official-layout
   site video -> write ``result.json`` and append to ``results.jsonl``. **Never calls ``policy.load()``.**

Terminal status: ``error`` is folded into ``fail`` with the original value kept in ``error_kind``;
``StepCapReached`` and ``session.cap_hit`` are recorded as ``timeout``; count fields are always written explicitly,
including zeros. When the watchdog fires it first writes an ``INFRA_TIMEOUT`` result, prints
``SERVER_LEFT pid=... port=... metadata=... stop="scripts/evaluate.py --stop-server ..."``, then calls
``os._exit(75)`` -- only the client is killed; the server subprocess is left alone.

Output tree: under ``<out>/rollouts/<model>/<dataset>/seed<policy_seed>/`` there are ``results.jsonl``,
``log.json`` (written by ``report``), ``progress.json``,
``videos/<Task>_ep<N>_<success|fail|timeout>_<task_goal>_<tier>.mp4`` and ``raw/<Task>_ep<N>_<tier>/`` (the
recorder's ``front.mkv`` / ``wrist.mkv`` / ``arrays.npz`` / ``meta.json`` / ``events.jsonl`` / ``frames-*.jsonl``,
the model's ``trace.jsonl``, and ``result.json``). ``ep<N>``: ood uses the builder episode index 0-49, hard-verify
uses the original official episode number (``source_episode``).
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from robomme_ood_eval.session import (DATASETS, HARD_VERIFY, OOD, EnvSession, RecorderError, ResetBudgetExhausted,
                                       StepCapReached)

#: step cap per dataset; ood truncates strictly (step 1801 never reaches the environment)
DATASET_MAX_STEPS = {HARD_VERIFY: 1300, OOD: 1800}
XHARD0 = "xhard0"
#: the 16 official tasks (same order as the official ``TASK_NAME_LIST``)
TASKS = ("BinFill", "StopCube", "PickXtimes", "SwingXtimes", "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap",
         "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube",
         "InsertPeg", "PatternLock", "RouteStick")
#: default per-episode wall clock (seconds): per model, 1800 otherwise; the first episode in a process gets an
#: extra FIRST_EPISODE_EXTRA_S
DEFAULT_WALL_S = {"smvla": 900.0, "perceptual-framesamp-modul": 1200.0}
FALLBACK_WALL_S = 1800.0
FIRST_EPISODE_EXTRA_S = 600.0
FIRST_INFER_DEADLINE_S = 1800.0
MEDIA_DEADLINE_S = 1200.0
EXIT_WALL = 75
#: terminal statuses play may return; only the first three are persisted
PLAY_STATUSES = ("success", "fail", "timeout", "error")
TERMINAL_STATUSES = ("success", "fail", "timeout")
#: required PlayOutput fields
PLAY_REQUIRED = ("status", "task_success", "steps", "error", "infra", "infra_reason")
#: watchdog phase -> infra_reason
WATCH_REASONS = {"episode_wall": "episode_wall", "first_infer": "deadline_first_infer",
                 "media_finalize": "deadline_media_finalize"}
#: hard-exit function (os._exit in production; unit tests replace it with a non-exiting recorder)
HARD_EXIT: Callable[[int], Any] = os._exit


class IdentityMismatch(RuntimeError):
    """The identity resolved by the builder does not match the queue identity row: the result has been written and
    the caller stops the whole batch."""


class PlayContractError(ValueError):
    """The value returned by ``play`` violates the ``PlayOutput`` contract."""


# -- builder cache -------------------------------------------------------------------

#: replaceable in unit tests: ``(task, dataset, max_steps) -> builder``; None builds a real ``BenchmarkEnvBuilder``
BUILDER_FACTORY: Callable[[str, str, int], Any] | None = None
_BUILDERS: dict[tuple[str, str], Any] = {}


def builder_for(task: str, dataset: str):
    """Cache builders in-process per ``(task, dataset)`` (step cap from ``DATASET_MAX_STEPS[dataset]``)."""
    if dataset not in DATASET_MAX_STEPS:
        raise ValueError(f"dataset={dataset!r} is not one of {tuple(DATASET_MAX_STEPS)}")
    ck = (task, dataset)
    if ck not in _BUILDERS:
        ms = DATASET_MAX_STEPS[dataset]
        if BUILDER_FACTORY is not None:
            _BUILDERS[ck] = BUILDER_FACTORY(task, dataset, ms)
        else:
            from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

            _BUILDERS[ck] = BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="joint_angle",
                                                max_steps=ms)
    return _BUILDERS[ck]


def clear_builders() -> None:
    _BUILDERS.clear()


# -- identity check (extracted from seat.py::check_identity) -------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _same_int_or_null(a: Any, b: Any) -> bool:
    return (a is None and b is None) or (_is_int(a) and _is_int(b) and a == b)


def check_identity(resolved: dict, want: dict, *, dataset: str = OOD) -> str | None:
    """The identity resolved by the builder must match the queue identity row; returns a description on mismatch.

    ood: tier / seed / candidate / spec_sha256 strictly equal key by key (candidate may be empty); if the identity
    row gives source_episode it must match too. hard-verify: tier is xhard0 on both sides, seed and source_episode
    are equal integers, and candidate and spec_sha256 are null on both sides."""
    bad = []
    if resolved.get("tier") != want.get("tier"):
        bad.append(f"tier builder={resolved.get('tier')} want={want.get('tier')}")
    if dataset == HARD_VERIFY:
        if want.get("tier") != XHARD0:
            bad.append(f"tier want={want.get('tier')} is not {XHARD0}")
        for k in ("seed", "source_episode"):
            rv, wv = resolved.get(k), want.get(k)
            if not (_is_int(rv) and _is_int(wv) and rv == wv):
                bad.append(f"{k} builder={rv!r} want={wv!r}")
        for k in ("candidate", "spec_sha256"):
            if resolved.get(k) is not None or want.get(k) is not None:
                bad.append(f"{k} builder={resolved.get(k)!r} want={want.get(k)!r} must both be null")
        return "; ".join(bad) or None
    for k in ("seed", "candidate"):
        rv, wv = resolved.get(k), want.get(k)
        if not _same_int_or_null(rv, wv):
            bad.append(f"{k} builder={rv!r} want={wv!r}")
    if resolved.get("spec_sha256") != want.get("spec_sha256"):
        bad.append(f"spec_sha256 builder={resolved.get('spec_sha256')} want={want.get('spec_sha256')}")
    if "source_episode" in want and not _same_int_or_null(resolved.get("source_episode"), want.get("source_episode")):
        bad.append(f"source_episode builder={resolved.get('source_episode')!r} want={want.get('source_episode')!r}")
    return "; ".join(bad) or None


# -- data structures -----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """One episode's identity (immutable, built by the outer loop; from ``BenchmarkEnvBuilder.resolve_identity``).

    ``episode``: builder episode index (ood 0-49, hard-verify 0-11); ``source_episode``: original official number of
    an xhard0 episode (None for new-value tiers); ``key``: ``<Task>_<tier>_<seed>``; ``max_steps``: 1300 / 1800;
    ``strict_cap``: true for ood; ``out_dir``: this episode's raw output directory ``raw/<Task>_ep<N>_<tier>/``
    (where the model writes ``trace.jsonl``)."""

    dataset: str
    task: str
    episode: int
    source_episode: int | None
    tier: str
    seed: int
    candidate: int | None
    spec_sha256: str | None
    key: str
    max_steps: int
    strict_cap: bool
    attempt: int
    policy_seed: int
    out_dir: str

    def __post_init__(self):
        if self.dataset not in DATASET_MAX_STEPS:
            raise ValueError(f"dataset={self.dataset!r} is not one of {tuple(DATASET_MAX_STEPS)}")

    @property
    def ep_label(self) -> int:
        """``ep<N>`` in file names: ood uses the builder episode index, hard-verify the original official episode
        number."""
        if self.dataset == HARD_VERIFY:
            if self.source_episode is None:
                raise ValueError("hard-verify episode is missing source_episode")
            return int(self.source_episode)
        return int(self.episode)

    @property
    def raw_name(self) -> str:
        return f"{self.task}_ep{self.ep_label}_{self.tier}"

    def identity(self) -> dict:
        """Identity fields written to the result row and the trace header."""
        return {"dataset": self.dataset, "task": self.task, "episode": self.episode, "builder_episode": self.episode,
                "source_episode": self.source_episode, "tier": self.tier, "seed": self.seed,
                "candidate": self.candidate, "spec_sha256": self.spec_sha256, "key": self.key,
                "max_steps": self.max_steps, "strict_cap": self.strict_cap, "attempt": self.attempt,
                "policy_seed": self.policy_seed}


@dataclasses.dataclass
class EpisodeResult:
    """Result returned by ``run_episode`` and written to ``result.json`` (the fields are the contract).

    ``status`` is one of success / fail / timeout; ``error_kind`` records the original terminal status folded into
    fail (``error``), None otherwise; the count fields ``task_success`` / ``steps`` / ``exec_steps`` /
    ``reset_calls`` / ``decisions`` / ``demo_frames`` are always written explicitly; model-specific fields in
    ``extra`` (e.g. PonderPounce's ``pp_sid_use_index``) are flattened to the top level by ``to_dict``."""

    dataset: str
    task: str
    episode: int
    source_episode: int | None
    tier: str
    seed: int
    candidate: int | None
    spec_sha256: str | None
    key: str
    max_steps: int
    strict_cap: bool
    attempt: int
    policy_seed: int
    model: str
    policy_label: str
    server_seed: int | None = None
    status: str = "fail"
    task_success: int = 0
    steps: int = 0
    exec_steps: int = 0
    cap_hit: bool = False
    reset_calls: int = 0
    demo_frames: int = 0
    decisions: int = 0
    error: str | None = None
    error_kind: str | None = None
    infra: bool = False
    infra_reason: str | None = None
    budget_exhausted: bool = False
    run_blocked: bool = False
    recorder_verify: str | None = None
    raw_dir: str | None = None
    video: str | None = None
    video_error: str | None = None
    task_goal: str | None = None
    timing: dict = dataclasses.field(default_factory=dict)
    extra: dict = dataclasses.field(default_factory=dict)
    host: str = dataclasses.field(default_factory=socket.gethostname)
    t_start: float = 0.0
    t_end: float = 0.0

    @classmethod
    def from_spec(cls, spec: EpisodeSpec, policy) -> "EpisodeResult":
        ident = spec.identity()
        ident.pop("builder_episode")
        return cls(**ident, model=str(getattr(policy, "model", "?")),
                   policy_label=str(getattr(policy, "label", None) or getattr(policy, "model", "?")))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        extra = d.pop("extra") or {}
        for k, v in extra.items():
            d.setdefault(k, v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeResult":
        names = {f.name for f in dataclasses.fields(cls)}
        core = {k: v for k, v in d.items() if k in names and k != "extra"}
        return cls(**core, extra={k: v for k, v in d.items() if k not in names})


def normalize_play_output(out: Any) -> dict:
    """Check the ``PlayOutput`` contract: a dict with all required fields and a valid status; otherwise raise
    ``PlayContractError``."""
    if not isinstance(out, dict):
        raise PlayContractError(f"play must return a dict, got {type(out).__name__}")
    missing = [k for k in PLAY_REQUIRED if k not in out]
    if missing:
        raise PlayContractError(f"PlayOutput is missing required fields {missing}")
    if out["status"] not in PLAY_STATUSES:
        raise PlayContractError(f"PlayOutput.status={out['status']!r} is not in {PLAY_STATUSES}")
    return dict(out)


# -- paths ---------------------------------------------------------------------------


def run_dir(out_dir: str | Path, policy_label: str, dataset: str, policy_seed: int) -> Path:
    """``<out>/rollouts/<model>/<dataset>/seed<policy_seed>/``."""
    return Path(out_dir) / "rollouts" / str(policy_label) / dataset / f"seed{int(policy_seed)}"


def _label(policy) -> str:
    return str(getattr(policy, "label", None) or getattr(policy, "model", "model"))


def make_spec(policy, dataset: str, task: str, episode: int, out_dir: str | Path, *, attempt: int = 1
              ) -> tuple[EpisodeSpec, dict]:
    """Resolve the identity and build the ``EpisodeSpec``; returns ``(spec, resolved)``. A dataset that is neither
    of the two raises ``ValueError`` directly."""
    if dataset not in DATASET_MAX_STEPS:
        raise ValueError(f"dataset={dataset!r} is not one of {tuple(DATASET_MAX_STEPS)}")
    resolved = dict(builder_for(task, dataset).resolve_identity(int(episode)))
    tier, seed = resolved["tier"], int(resolved["seed"])
    src = resolved.get("source_episode")
    rdir = run_dir(out_dir, _label(policy), dataset, policy.policy_seed)
    spec = EpisodeSpec(dataset=dataset, task=task, episode=int(episode),
                       source_episode=None if src is None else int(src), tier=tier, seed=seed,
                       candidate=resolved.get("candidate"), spec_sha256=resolved.get("spec_sha256"),
                       key=f"{task}_{tier}_{seed}", max_steps=DATASET_MAX_STEPS[dataset],
                       strict_cap=dataset == OOD, attempt=int(attempt), policy_seed=int(policy.policy_seed),
                       out_dir="")
    spec = dataclasses.replace(spec, out_dir=str(rdir / "raw" / spec.raw_name))
    return spec, resolved


def result_path(policy, dataset: str, task: str, episode: int, out_dir: str | Path) -> Path:
    """Path of this episode's ``result.json`` (used to skip on resume; only resolves the identity, never touches the
    environment or the model)."""
    spec, _ = make_spec(policy, dataset, task, episode, out_dir)
    return Path(spec.out_dir) / "result.json"


# -- writing -------------------------------------------------------------------------


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def append_jsonl(path: Path, row: dict) -> None:
    """Append one line and fsync; if the previous line is a half line left by a crash, add a newline first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((dumps(row) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(dumps(obj) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_result(result: EpisodeResult, spec: EpisodeSpec) -> None:
    raw = Path(spec.out_dir)
    rdir = raw.parent.parent
    d = result.to_dict()
    write_json_atomic(raw / "result.json", d)
    append_jsonl(rdir / "results.jsonl", d)


def _progress(spec: EpisodeSpec, phase: str, step: int = 0) -> None:
    rdir = Path(spec.out_dir).parent.parent
    try:
        write_json_atomic(rdir / "progress.json", {"pid": os.getpid(), "host": socket.gethostname(),
                                                   "key": spec.key, "task": spec.task, "episode": spec.episode,
                                                   "dataset": spec.dataset, "phase": phase, "step": int(step),
                                                   "t": time.time()})
    except OSError:
        pass


# -- watchdog ------------------------------------------------------------------------


class _Watchdog:
    """Per-episode watchdog: one absolute-deadline timer per phase; when it fires, the timer thread writes an
    ``INFRA_TIMEOUT`` result, prints ``SERVER_LEFT``, then calls ``HARD_EXIT(75)`` (only the client is killed)."""

    def __init__(self, policy, spec: EpisodeSpec, result: EpisodeResult, state: dict):
        self.policy, self.spec, self.result, self.state = policy, spec, result, state
        self.lock = threading.Lock()
        self.timers: dict[str, threading.Timer] = {}

    def arm(self, phase: str, limit: float | None) -> None:
        if not limit or limit <= 0:
            return
        t = threading.Timer(float(limit), self._fire, args=(phase, float(limit)))
        t.daemon = True
        self.timers[phase] = t
        t.start()

    def disarm(self, phase: str) -> None:
        with self.lock:
            self.state[f"{phase}_done"] = True
        t = self.timers.pop(phase, None)
        if t is not None:
            t.cancel()

    def disarm_all(self) -> None:
        for p in list(self.timers):
            self.disarm(p)

    def _fire(self, phase: str, limit: float) -> None:
        with self.lock:
            if self.state.get(f"{phase}_done") or self.state.get("fired"):
                return
            self.state["fired"] = phase
            sess = self.state.get("session")
            res = dataclasses.replace(self.result)
            res.status, res.task_success, res.error_kind = "fail", 0, "error"
            res.infra, res.infra_reason = True, WATCH_REASONS.get(phase, f"deadline_{phase}")
            res.error = f"INFRA_TIMEOUT phase={phase} limit_s={limit:.0f}"
            res.exec_steps = int(getattr(sess, "steps", 0) or 0)
            res.reset_calls = int(getattr(sess, "reset_calls", 0) or 0)
            res.cap_hit = bool(getattr(sess, "cap_hit", False))
            res.demo_frames = int(((getattr(sess, "timing", None) or {}).get("demo_frames")) or 0)
            res.t_end = time.time()
            self.state["timeout_result"] = res
            try:
                _write_result(res, self.spec)
            finally:
                print(f"INFRA_TIMEOUT phase={phase} key={self.spec.key} limit_s={limit:.0f}", flush=True)
                for srv in _servers_of(self.policy):
                    print(srv.left_line(), flush=True)
        HARD_EXIT(EXIT_WALL)


def _servers_of(policy) -> list:
    fn = getattr(policy, "servers", None)
    try:
        return list(fn()) if callable(fn) else []
    except Exception:  # noqa: BLE001
        return []


def wall_limit(policy, first: bool, override: float | None = None) -> float:
    base = override if override is not None else (getattr(policy, "episode_wall_s", None)
                                                  or DEFAULT_WALL_S.get(getattr(policy, "model", ""), FALLBACK_WALL_S))
    return float(base) + (FIRST_EPISODE_EXTRA_S if first else 0.0)


# ── run_episode ──────────────────────────────────────────────────────────────


def _default_recorder(raw: Path, meta: dict):
    from robomme_ood_eval.record.recorder import EpisodeRecorder

    return EpisodeRecorder(raw, meta, overwrite=True)


def _render(spec: EpisodeSpec, result: EpisodeResult, official_root: str | Path | None) -> None:
    """When ``trace.jsonl`` exists, render the official-layout site video into ``videos/``; failures only record
    ``video_error`` and never change the terminal status."""
    raw = Path(spec.out_dir)
    if not (raw / "trace.jsonl").is_file():
        result.video_error = "no_trace"
        return
    try:
        from robomme_ood_eval.record import official_render

        out = official_render.render_video(raw, raw.parent.parent / "videos", episode_id=spec.ep_label,
                                           terminal=result.status, official_root=official_root)
        result.video = None if out is None else str(Path(out).relative_to(raw.parent.parent))
        if out is None:
            result.video_error = "no_frame"
    except Exception as e:  # noqa: BLE001
        result.video_error = f"{type(e).__name__}: {e}"[:800]


def run_episode(policy, dataset: str, task: str, episode: int, out_dir: str | Path, expect: dict | None = None, *,
                attempt: int = 1, ledger: Any = None, recorder_factory: Callable[[Path, dict], Any] | None = None,
                render: bool = True, official_root: str | Path | None = None, wall_s: float | None = None,
                first_infer_s: float | None = FIRST_INFER_DEADLINE_S,
                media_s: float | None = MEDIA_DEADLINE_S) -> EpisodeResult:
    """Run one episode (order described in the module docstring). ``out_dir`` is the output root (containing
    ``rollouts/<model>/<dataset>/seed<n>/``).

    Keyword arguments are only for the cluster seat runner and unit tests: ``attempt`` (attempt number), ``ledger``
    (reset-budget ledger with ``claim(what)``), ``recorder_factory(raw_dir, meta)`` (default: the real
    ``EpisodeRecorder``), ``render`` (whether to render site videos), ``official_root`` (repo root holding the
    official ``RolloutRecorder``), and the three watchdog deadlines (None or 0 disables one)."""
    t_start = time.time()
    spec, resolved = make_spec(policy, dataset, task, episode, out_dir, attempt=attempt)
    raw = Path(spec.out_dir)
    result = EpisodeResult.from_spec(spec, policy)
    result.t_start = t_start
    result.raw_dir = str(raw.relative_to(raw.parent.parent))
    result.server_seed = policy.server_seed() if callable(getattr(policy, "server_seed", None)) else None
    # (1) identity
    if expect is not None:
        bad = check_identity(resolved, expect, dataset=dataset)
        if bad:
            result.status, result.error_kind, result.run_blocked = "fail", "error", True
            result.error = f"IDENTITY_MISMATCH {bad}"[:800]
            result.extra["resolved_identity"] = resolved
            result.t_end = time.time()
            _write_result(result, spec)
            print(f"RUN_BLOCKED reason=identity key={spec.key} dataset={dataset} detail={bad}", flush=True)
            raise IdentityMismatch(bad)
    # (2) pre-episode: client state, session identifiers, liveness probe, pre-episode refusals (ServerDead /
    # AstraStop are re-raised unchanged and no result is written)
    _progress(spec, "reset")
    policy.reset(spec)
    # (3) recorder + EnvSession + watchdog
    state: dict[str, Any] = {}
    dog = _Watchdog(policy, spec, result, state)
    first = int(getattr(policy, "episodes_run", 0) or 0) == 0
    meta = {"identity": spec.identity(), "model": result.model, "policy_label": result.policy_label,
            "resolved_identity": resolved}
    try:
        recorder = (recorder_factory or _default_recorder)(raw, meta)
    except Exception as e:  # noqa: BLE001 recorder cannot be created (e.g. disk full) = infrastructure failure
        result.status, result.error_kind, result.infra, result.infra_reason = "fail", "error", True, "recorder"
        result.error = f"RecorderError: init: {type(e).__name__}: {e}"[:800]
        result.t_end = time.time()
        _write_result(result, spec)
        return result

    def on_first_step():
        dog.disarm("first_infer")
        _progress(spec, "episode")

    session = EnvSession(task, spec.episode, dataset=dataset, max_steps=spec.max_steps, recorder=recorder,
                         builder=builder_for(task, dataset), progress_cb=lambda s: _progress(spec, "episode", s),
                         step_cap=spec.max_steps if spec.strict_cap else None, ledger=ledger,
                         first_step_cb=on_first_step)
    state["session"] = session
    dog.arm("episode_wall", wall_limit(policy, first, wall_s))
    dog.arm("first_infer", first_infer_s)
    _progress(spec, "first_infer")
    t0 = time.perf_counter()
    out: dict = {}
    budget_exc: BaseException | None = None
    try:
        try:
            session.build()
        except Exception as e:  # noqa: BLE001 build failure (e.g. Vulkan) is an infrastructure failure; skip play
            if session.budget_exhausted:
                budget_exc = e
                out = {"status": "error", "error": f"{type(e).__name__}: {e}"[:800], "infra": False,
                       "infra_reason": None}
            else:
                out = {"status": "error", "error": f"{type(e).__name__}: {e}"[:800], "infra": True,
                       "infra_reason": "env_build"}
        else:
            # ④ play
            try:
                out = normalize_play_output(policy.play(session, spec, recorder))
            except StepCapReached as e:
                out = {"status": "timeout", "error": str(e), "infra": False, "infra_reason": None}
            except RecorderError as e:
                out = {"status": "error", "error": f"RecorderError: {e}"[:800], "infra": True,
                       "infra_reason": "recorder"}
            except Exception as e:  # noqa: BLE001 ordinary model-side exception: record error, still finalize; the Policy moves on to the next episode
                if session.budget_exhausted:
                    budget_exc = e
                out = {"status": "error", "error": f"{type(e).__name__}: {e}"[:800], "infra": False,
                       "infra_reason": None, "traceback": traceback.format_exc()[-4000:]}
    except BaseException:
        # KeyboardInterrupt / SystemExit: finalize only, write no result
        dog.disarm_all()
        _safe(session.close)
        _safe(lambda: recorder.close({"status": "interrupted", "steps": session.steps}))
        raise
    finally:
        dog.disarm("first_infer")
        dog.disarm("episode_wall")
    if state.get("fired"):
        return state["timeout_result"]
    wall = time.perf_counter() - t0
    # (5) finalize: session.close -> recorder.close -> video -> result.json
    close_error = None
    try:
        session.close()
    except Exception as e:  # noqa: BLE001
        close_error = repr(e)[:800]
    _progress(spec, "media_finalize", session.steps)
    dog.arm("media_finalize", media_s)
    rsum: dict | None = None
    try:
        rsum = recorder.close({"status": out.get("status"), "steps": out.get("steps"), "exec_steps": session.steps,
                               "cap_hit": session.cap_hit})
    except Exception as e:  # noqa: BLE001 finalize failure (e.g. disk full) = infrastructure failure
        rsum = {"RECORDER_VERIFY": "ERROR", "error": f"{type(e).__name__}: {e}"[:800]}
        out.update(infra=True, infra_reason="recorder_close")
    _classify(result, out, session)
    if close_error:
        result.extra["close_error"] = close_error
    result.recorder_verify = (rsum or {}).get("RECORDER_VERIFY")
    if render:
        _render(spec, result, official_root)
    dog.disarm("media_finalize")
    if state.get("fired"):
        return state["timeout_result"]
    result.timing = {"episode_wall_s": round(wall, 3), "first_episode_in_process": first,
                     "env": dict(session.timing), "policy": out.get("timing"), "recorder": rsum}
    result.task_goal = session.task_goal
    result.t_end = time.time()
    _write_result(result, spec)
    policy.episodes_run = int(getattr(policy, "episodes_run", 0) or 0) + 1
    _progress(spec, "done", session.steps)
    print(f"EPISODE_DONE model={result.model} dataset={dataset} key={spec.key} ep={spec.ep_label} "
          f"status={result.status} exec_steps={result.exec_steps} cap_hit={result.cap_hit} infra={result.infra}",
          flush=True)
    if budget_exc is not None:
        raise budget_exc
    return result


def _safe(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


def _classify(result: EpisodeResult, out: dict, session: EnvSession) -> None:
    """Execution counts follow the environment side; budget exhaustion and step-cap classification override what
    the model reports; error is folded into fail."""
    status = out.get("status", "error")
    result.steps = int(out.get("steps") or 0)
    result.exec_steps = int(session.steps)
    result.cap_hit = bool(session.cap_hit)
    result.reset_calls = int(session.reset_calls)
    result.demo_frames = int(session.timing.get("demo_frames") or 0)
    result.decisions = int(out.get("decisions") or 0)
    result.error = out.get("error")
    result.infra = bool(out.get("infra"))
    result.infra_reason = out.get("infra_reason")
    for k, v in out.items():
        if k not in PLAY_REQUIRED and k not in ("decisions", "timing", "traceback"):
            result.extra[k] = v
    if out.get("traceback"):
        result.extra["traceback"] = out["traceback"]
    if session.budget_exhausted:
        result.budget_exhausted = True
        result.infra, result.infra_reason = False, None
        status = "error"
    elif session.cap_hit:
        if status != "timeout":
            result.extra["client_status"] = status
        status = "timeout"
        result.error = result.error or f"STEP_CAP exec_steps={session.steps} cap={session.step_cap} not successful, counted as timeout"
    if status == "error":
        result.status, result.error_kind = "fail", "error"
    else:
        result.status = status
    result.task_success = int(result.status == "success")


__all__ = ["EpisodeSpec", "EpisodeResult", "run_episode", "make_spec", "result_path", "run_dir", "builder_for",
           "clear_builders", "check_identity", "normalize_play_output", "DATASET_MAX_STEPS", "TASKS",
           "DEFAULT_WALL_S", "IdentityMismatch", "PlayContractError", "PLAY_REQUIRED", "DATASETS"]
