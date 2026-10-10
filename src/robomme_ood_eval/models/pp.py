#!/usr/bin/env python3
"""PonderPounce new-side client.

``run_episode(session, identity, conn_info, recorder) -> dict`` is called by the seat runner's ``run_one`` (loaded
via ``load_sibling("pp_client")`` for ``--policy pp``). The environment side is this repo's ``EnvSession``
(``robomme_ood``); the model side is a PonderPounce server speaking the vla-eval 0.7.0 protocol
(``python -m ponderpounce.eval.robomme_server``).

Reproduced item by item from the vla-eval 0.7.0 source (``runners/sync_runner.py::SyncEpisodeRunner.run_episode``,
``benchmarks/robomme/benchmark.py::RoboMMEBenchmark``, ``benchmarks/base.py::StepBenchmark``):

1. Connect: ``vla_eval.connection.Connection(url, timeout=300.0)``, ``connect(benchmark=PP_BENCHMARK)`` completes the
   HELLO handshake (``server.timeout`` in PonderPounce's official ``configs/robomme.yaml`` is 300.0).
2. ``session.reset()`` -> first observation (same as ``make_obs``: ``images.agentview`` / ``images.wrist``,
   ``task_description``, ``states`` (7 joints float64 + gripper dim 1, concatenated and cast to float32); the first
   one additionally carries ``video_history = front_rgb_list[:-1]``, ``wrist_video_history = wrist_rgb_list[:-1]``,
   ``episode_restart=True``).
3. ``EPISODE_START`` (no reply when the server succeeds): ``{"task": {"name","env_id","episode_idx"},
   "recording": {"sid","eid","eval_id": "","db_path": ""}}``; ``sid`` is fixed (see ``fixed_sid``), ``eid`` equals
   ``sid``.
4. ``for step in range(max_steps)``: ``act(obs)`` -> the action ``actions`` is flattened to a list of Python floats
   and its first 8 dims are given to ``session.step`` -> ``terminated or truncated or info["status"] == "error"``
   means ``break`` (the last frame is not sent) -> otherwise the next frame is packed with ``make_obs``. At most
   exactly ``max_steps`` actions (1300 for xhard0; for the OOD dataset the session's strict cap applies and the loop
   bound equals it).
5. ``EPISODE_END`` is sent only when the loop completes normally: ``{"metrics": {"success": status=="success"},
   "steps": step+1, "elapsed_sec": ...}``; not sent on an exception midway (consistent with SyncEpisodeRunner
   re-raising), and the connection is always closed.

Fixed sid: ``<task>|<source_episode>|<seed>`` for xhard0, ``<task>|<tier>|<seed>`` for OOD tiers. PonderPounce's
noise seed is ``crc32(f"{seed}:{sid}:{n}")``; each side starts its own server process and a sid is used only once
per process, guaranteeing ``n=0``. Infrastructure retries are done by the seat runner "restarting the server and
resending the same sid"; the ``reconnect`` in this driver is only for a ``ConnectionClosed`` within the same episode
(at most ``conn_info["pp_max_reconnects"]`` times, default 1) and **does not resend EPISODE_START** -- note that the
vla-eval server assigns a new session id to a new connection, and PonderPounce replies ERROR to observations without
EPISODE_START, so the episode then records ``error`` (``infra=True``) and the seat runner handles it as an
infrastructure failure.

Trace: one ``trace.jsonl`` per episode (``trace_writer.TraceWriter``, route ``pp-new``). ``request`` lines are the
sha256 of the normalized bytes of every in-episode protocol frame (EPISODE_START / OBSERVATION / EPISODE_END, not the
connection-level HELLO) (``canonical_frame_bytes``: key-sorted msgpack, arrays as raw bytes in original dtype/shape,
EPISODE_END without the wall-clock field ``elapsed_sec``); ``response`` lines are the full action chunk returned by the
server; ``step`` lines are the post-step images, state, the 8-d action given to the environment and terminal flags.
The original side ``pp_official_runner.py`` writes its trace with the same helpers from this module, so the fields
match on both sides.

Keys read from ``conn_info``: ``host`` (default 127.0.0.1), ``port`` (required), ``max_steps`` (required),
``dataset`` (optional; ``hard-verify`` requires ``tier == "xhard0"``), ``trace_path`` / ``trace_dir`` (optional;
``trace_dir`` is the per-episode directory, see ``resolve_trace_path``), ``pp_max_reconnects`` (optional, default 1),
``pp_phase2`` (optional, see below), ``attempt`` / ``episode_tag`` (optional, for identity).

Extended new-side behavior (shared contracts C1-C11): all new behavior is turned on by a switch; by default (switch
off) the protocol frames and ``trace.jsonl`` written are byte-identical to ``BASE``. The switch is
``conn_info["pp_phase2"]`` (authoritative when given explicitly), otherwise the environment variable
``SGEVAL_PP_SERVER_WRAP=1`` (set by the seat runner when starting the server with the ``pp_server_wrap.py``
wrapper). When on:

- the server wrapper attaches ``"subgoal"`` to every ACTION reply (the model's own subgoal text, ``None`` while
  waiting); ``TracedConnection.act`` stores it in ``last_subgoal``; a reply without this key means the wrapper is not
  in effect, and the episode records ``error`` (``infra_reason=pp_subgoal_missing``) instead of passing off the
  environment's ground truth as the subgoal.
- per-step ``subgoal`` records the model subgoal text after coordinate conversion by ``pp_subgoal_to_official`` (the
  raw text is recorded as ``subgoal_raw``); the environment ground truth ``info["simple_subgoal_online"]`` is recorded
  instead as a ``history`` line with the same step number, ``note=oracle_simple_subgoal:<text>``.
- C1 route ``pp/new``; C2 the demo segment records all reset frames (including the initial frame), and
  ``end.demo_frames`` is the demo frame count without the initial frame; C3 ``terminal_reason`` and ``status`` both
  take ``success`` / ``fail`` / ``timeout`` / ``error`` (the old exit reason is recorded separately as
  ``exit_reason``), and episodes that fail before reset record ``no_frame=true``; C4 the original float64 actions
  given to the environment and the states of observed steps are collected by ``TraceWriter`` and at finalization
  written via ``trace_writer.merge_write_npz`` to ``arrays.npz`` in the same directory (``exec_action__%05d`` /
  ``exec_state__%05d``; merged with same-named keys of a recorder in the same directory with equal values, neither
  overwriting the other in any order; this module no longer calls ``np.savez`` directly); C6 identity adds
  ``attempt``; C8 ``end`` writes ``steps_attempted`` / ``steps_observed`` / ``frames_recorded``, and environment
  exception steps and empty-observation steps keep their step number and action via ``log_missing_step``.

Language ledger (only recorded with the switch on and a trace location; ``language.jsonl`` next to the trace): one
``action_model`` call per observation (``in`` written before sending: ``role=fields`` ``task_description`` +
references to the current front / wrist frames; after the reply ``_sgeval_audit`` is popped, per-channel tokenization
messages are written, and ``close_call`` records ``server_final_text``); when the audit block carries
``pp_generation`` a ``subgoal_model`` call is opened to record the full S2 generation block (see
``TracedConnection._log_generation``; when the block carries the S2 input ``input_text`` decoded by the wrapper, the
``in`` message and image references are written first -- written after the reply, the same kind of exception as
other tokenization channel messages). Executed steps point via ``source_call_id`` to the ``action_model`` call their
action came from, with ``chunk_index=0``. The audit key is always popped before actions reach the environment and
never changes actions, RNG or request bytes.

The original side ``pp_official_runner.py`` (unchanged) only calls the default paths of this module:
``TRACE_SCHEMA_ROUTE_ORIG`` is still ``pp-orig`` and ``trace_reset`` / ``trace_step`` get no new arguments, so the
original side's serialized output is byte-identical to ``BASE``.

Timing records every successful observation RTT. A fresh S1 chunk requires both cursor=1 and an increased S1 fire
count. For its RTT R, same-step S2 time s, and earlier nonfresh S2 bucket B, the recorded wall is the synthetic
R+B, action RTT is R-s, and language time is B+s. Initial hold time enters B; a final unassigned bucket becomes
orphan language time. Failed transport attempts have no RTT entry, and any reconnect marks the entire episode
for exclusion by timing reports. Timing is independent of the language ledger.

At import time this only depends on the standard library and numpy; ``vla_eval`` (in the client extension
environment client-env), ``anyio``, ``msgpack`` and ``websockets`` are imported only when used. Unit tests inject
stand-in connections via ``run_episode(..., connection_factory=...)``.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robomme_ood_eval.timing import ChunkTimer, attach_policy_timing

#: benchmark name sent in the HELLO handshake, same as ``benchmark`` in PonderPounce's official configs/robomme.yaml
PP_BENCHMARK = "vla_eval.benchmarks.robomme.benchmark:RoboMMEBenchmark"
#: timeout of each recv (seconds), same as ``server.timeout`` in the official configs/robomme.yaml
PP_TIMEOUT_S = 300.0
#: action dims given to the environment (7 joints + 1 gripper)
PP_ACTION_DIMS = 8
#: default upper bound on reconnects after ConnectionClosed within one episode
PP_MAX_RECONNECTS = 1
XHARD0 = "xhard0"
HARD_VERIFY = "hard-verify"
TRACE_SCHEMA_ROUTE_NEW = "pp-new"
#: original-side route; referenced by ``pp_official_runner.py`` (unchanged), keeping the old value so the original
#: side's output stays byte-identical to BASE
TRACE_SCHEMA_ROUTE_ORIG = "pp-orig"
#: extended new-side route (C1)
TRACE_ROUTE_NEW_C1 = "pp/new"
#: environment variable of the extended-behavior switch (same name as the seat runner's switch for starting the
#: wrapper server)
PHASE2_ENV = "SGEVAL_PP_SERVER_WRAP"
#: subgoal key in server wrapper replies (same as pp_server_wrap.SUBGOAL_KEY)
SUBGOAL_KEY = "subgoal"
#: server wrapper reply audit key (same as trace_writer.AUDIT_KEY), popped before reaching the environment
AUDIT_KEY = "_sgeval_audit"
#: prefix of the environment ground truth in the note of history lines
ORACLE_NOTE_PREFIX = "oracle_simple_subgoal:"
#: protocol frame types (same values as vla_eval.protocol.messages.MessageType)
HELLO, OBSERVATION, ACTION, EPISODE_START, EPISODE_END, ERROR = (
    "hello", "observation", "action", "episode_start", "episode_end", "error")


def load_trace_writer():
    """The evaluation package's ``record.trace_writer`` module (package import, reused if already imported; unit tests
    may replace this function entirely)."""
    from robomme_ood_eval.record import trace_writer

    return trace_writer


#: "not provided" sentinel (C7): i.e. ``trace_writer.UNSET``, distinct from ``None`` (model is waiting)
UNSET = load_trace_writer().UNSET


def _is_unset(x: Any) -> bool:
    """``UNSET`` check (by class name, compatible with trace_writer copies loaded by path in tests)."""
    return x is UNSET or type(x).__name__ == "_Unset"


# -- extended behavior: switch, subgoal coordinate conversion --------------------------


def phase2_enabled(conn_info: dict) -> bool:
    """Extended new-side switch: ``conn_info["pp_phase2"]`` when given explicitly, otherwise the environment variable
    ``SGEVAL_PP_SERVER_WRAP``."""
    if conn_info.get("pp_phase2") is not None:
        return bool(conn_info["pp_phase2"])
    return os.environ.get(PHASE2_ENV, "") == "1"


_AT_POINT = re.compile(r"\bat\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def _to_255(v: str | float) -> int:
    """0-1000 -> 0-255: ``v*255/1000`` rounded half up (no banker's rounding), then clamped to 0-255."""
    return max(0, min(255, int(math.floor(float(v) * 255.0 / 1000.0 + 0.5))))


def pp_subgoal_to_official(text: str | None) -> str | None:
    """Convert every ``at [x, y]`` (0-1000, x first) in a PonderPounce subgoal into the official layout ``at <y', x'>``
    (0-255, row first).

    ``y' = round(y*255/1000)``, ``x' = round(x*255/1000)``, clamped to 0-255; other text is kept unchanged; ``None``
    is returned unchanged (C7 waiting). Example: ``"pick up the cube at [612, 247]"`` ->
    ``"pick up the cube at <63, 156>"``."""
    if text is None:
        return None
    return _AT_POINT.sub(lambda m: f"at <{_to_255(m.group(2))}, {_to_255(m.group(1))}>", str(text))


def attempt_of(identity: dict, conn_info: dict, trace_path: Path | None) -> int | None:
    """C6 attempt number: ``conn_info["attempt"]`` -> ``episode_tag`` (``<key>.a<N>``) -> name of the trace's
    directory -> ``identity["attempt"]``."""
    if conn_info.get("attempt") is not None:
        return int(conn_info["attempt"])
    for name in (conn_info.get("episode_tag"), trace_path.parent.name if trace_path is not None else None):
        m = re.match(r"^.+\.a(\d+)$", str(name or ""))
        if m:
            return int(m.group(1))
    att = identity.get("attempt")
    return int(att) if att is not None else None


class SubgoalMissing(RuntimeError):
    """The extended switch is on, but the ACTION reply has no ``subgoal`` key (the server wrapper is not in
    effect)."""


# -- identity and protocol payloads ----------------------------------------------------


def fixed_sid(identity: dict, dataset: str | None = None) -> str:
    """Fixed sid: xhard0 ``<task>|<source_episode>|<seed>``; OOD tiers ``<task>|<tier>|<seed>``.

    With ``dataset == "hard-verify"`` the identity must be xhard0, otherwise ``ValueError`` (prevents OOD identities
    from mixing into hard-verify shards)."""
    task, tier, seed = identity["task"], identity.get("tier"), identity["seed"]
    if dataset == HARD_VERIFY and tier != XHARD0:
        raise ValueError(f"dataset={dataset} but identity tier={tier!r} is not {XHARD0}")
    if tier == XHARD0:
        src = identity["source_episode"]
        if not isinstance(src, int) or isinstance(src, bool):
            raise ValueError(f"source_episode of an xhard0 identity must be an integer: {src!r}")
        return f"{task}|{int(src)}|{int(seed)}"
    if not tier:
        raise ValueError(f"identity is missing tier: {identity!r}")
    return f"{task}|{tier}|{int(seed)}"


def episode_idx_of(identity: dict) -> int:
    """``task.episode_idx`` of EPISODE_START: the official episode number ``source_episode`` (same on both sides);
    falls back to ``builder_episode``."""
    src = identity.get("source_episode")
    if isinstance(src, int) and not isinstance(src, bool):
        return int(src)
    return int(identity["builder_episode"])


def episode_start_payload(task: str, episode_idx: int, sid: str, eid: str) -> dict:
    """Same structure as the EPISODE_START payload ``SyncEpisodeRunner`` sends when "the recorder is_active"."""
    return {"task": {"name": task, "env_id": task, "episode_idx": int(episode_idx)},
            "recording": {"sid": sid, "eid": eid, "eval_id": "", "db_path": ""}}


def episode_end_payload(success: bool, steps: int, elapsed_sec: float) -> dict:
    return {"metrics": {"success": bool(success)}, "steps": int(steps), "elapsed_sec": round(float(elapsed_sec), 3)}


def task_description_of(info: dict) -> str:
    """Same as ``RoboMMEBenchmark.reset``: take element 0 when ``info["task_goal"]`` is a list, otherwise ``str``."""
    goal = info["task_goal"]
    return goal[0] if isinstance(goal, list) else str(goal)


def state8(joint: Any, gripper: Any) -> np.ndarray:
    """8-d state as in ``make_obs``: float64 7 joints + gripper dim 1, concatenated and cast to float32."""
    j = np.asarray(joint, dtype=np.float64)
    g = np.asarray(gripper, dtype=np.float64)[:1]
    return np.concatenate([j, g]).astype(np.float32)


class ObsPacker:
    """One per episode, reproducing ``RoboMMEBenchmark.make_obs`` (send_wrist_image / send_state /
    send_video_history all default True, send_subgoal default False). The demo video is sent only once with the
    first non-empty observation and cleared afterwards."""

    def __init__(self) -> None:
        self.task_description = ""
        self._video: list = []
        self._wrist_video: list = []

    def on_reset(self, raw_obs: dict, info: dict) -> None:
        self._video = list(raw_obs["front_rgb_list"][:-1])
        self._wrist_video = list(raw_obs.get("wrist_rgb_list", [])[:-1])
        self.task_description = task_description_of(info)

    @property
    def demo_frames(self) -> int:
        return len(self._video)

    def make(self, raw_obs: Any) -> dict:
        if not raw_obs:
            return {"images": {}, "task_description": self.task_description}
        front_list = raw_obs.get("front_rgb_list", [])
        if not front_list:
            return {"images": {}, "task_description": self.task_description}
        obs: dict[str, Any] = {"images": {"agentview": front_list[-1]}, "task_description": self.task_description}
        wrist_list = raw_obs.get("wrist_rgb_list")
        if wrist_list:
            obs["images"]["wrist"] = wrist_list[-1]
        obs["states"] = state8(raw_obs["joint_state_list"][-1], raw_obs["gripper_state_list"][-1])
        if self._video:
            obs["video_history"] = list(self._video)
            if self._wrist_video:
                obs["wrist_video_history"] = list(self._wrist_video)
            obs["episode_restart"] = True
            self._video = []
            self._wrist_video = []
        return obs


def exec_action8(action_payload: dict) -> list[float]:
    """Same as ``RoboMMEBenchmark.step``: take ``actions`` (or ``action`` if missing), flatten to a list of Python
    floats, then take the first 8 dims (``actions[0][:8]``; the first 8 after flattening ``(1, D)`` are the first 8
    dims of row 0)."""
    raw = action_payload.get("actions", action_payload.get("action"))
    if raw is None:
        raise ValueError("Action dict must contain 'actions' or 'action' key")
    if hasattr(raw, "flatten"):
        flat = raw.flatten().tolist()
    elif not isinstance(raw, list):
        flat = list(raw)
    else:
        flat = raw
    return [float(x) for x in flat[:PP_ACTION_DIMS]]


def step_done(terminated: Any, truncated: Any, info: dict) -> bool:
    """The done criterion of ``RoboMMEBenchmark.step``."""
    return bool(terminated) or bool(truncated) or (isinstance(info, dict) and info.get("status") == "error")


def terminal_status(done: bool, truncated: bool, info: dict | None) -> tuple[str, str | None]:
    """End-of-episode status mapping: ``success`` / ``fail`` / ``timeout`` unchanged; environment ``error`` records
    error; truncated without an explicit status records timeout; a loop that ran all ``max_steps`` without ending
    records timeout; anything else (``ongoing`` / ``unknown`` etc.) records error + ``success_flag=<value>``."""
    st = (info or {}).get("status") if isinstance(info, dict) else None
    if not done:
        return "timeout", None
    if st in ("success", "fail", "timeout"):
        return st, None
    if st == "error":
        return "error", "env_status=error"
    if truncated:
        return "timeout", None
    return "error", f"success_flag={st}"


# -- normalized bytes and trace --------------------------------------------------------


def _normalize(obj: Any) -> Any:
    """Sort keys, turn arrays into "dtype/shape/raw bytes" dicts, numpy scalars into Python scalars (no PNG, so it is
    deterministic)."""
    if isinstance(obj, dict):
        return {str(k): _normalize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {"__ndarray__": True, "dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def canonical_frame_bytes(msg_type: str, payload: Any) -> bytes:
    """Normalized bytes of one protocol frame (the hash source of trace request lines). EPISODE_END drops the
    wall-clock ``elapsed_sec``; the frame header's ``seq`` / ``timestamp`` are not included."""
    import msgpack

    if msg_type == EPISODE_END and isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if k != "elapsed_sec"}
    return msgpack.packb({"type": str(msg_type), "payload": _normalize(payload)}, use_bin_type=True)


class NullTrace:
    """No-op implementation when no trace is written (same interface as trace_writer.TraceWriter)."""

    exec_steps = 0

    def log_demo(self, *a, **k): pass
    def log_request(self, *a, **k): pass
    def log_response(self, *a, **k): pass
    def log_history(self, *a, **k): pass
    def log_step(self, *a, **k): pass
    def log_missing_step(self, *a, **k): pass
    def close(self, *a, **k): pass


def trace_reset(trace, raw_obs: dict, task_description: str, *, include_initial: bool = False) -> int:
    """Demo line: per-frame image hashes of ``video_history`` (= ``front_rgb_list[:-1]``), wrist frames at the same
    positions, per-frame 8-d states and the instruction text. Returns the demo frame count (excluding the initial
    frame).

    ``include_initial=True`` (extended new side, C2): the demo line records all frames returned by reset (including
    the final initial frame), ``frames == len(states) ==`` return value ``+ 1``. The default False is byte-identical
    to BASE (the original side does not pass it)."""
    if include_initial:
        fronts = list(raw_obs["front_rgb_list"])
        wrists = list(raw_obs.get("wrist_rgb_list", []))
        joints = list(raw_obs.get("joint_state_list", []) or [])
        grips = list(raw_obs.get("gripper_state_list", []) or [])
        states = [state8(j, g) for j, g in zip(joints, grips)]
        trace.log_demo(fronts, wrists, states=states, texts=[task_description])
        return max(0, len(fronts) - 1)
    fronts = list(raw_obs["front_rgb_list"][:-1])
    wrists = list(raw_obs.get("wrist_rgb_list", [])[:-1])
    joints = list(raw_obs.get("joint_state_list", []) or [])
    grips = list(raw_obs.get("gripper_state_list", []) or [])
    states = [state8(j, g) for j, g in zip(joints[:-1], grips[:-1])]
    trace.log_demo(fronts, wrists, states=states, texts=[task_description])
    return len(fronts)


def trace_step(trace, step_no: int, out: tuple, action8: list[float], subgoal: Any = UNSET,
               link: dict | None = None) -> bool:
    """One line after executing step ``step_no`` (from 1); ``out`` is the 5-tuple of the environment ``step``. Returns
    whether this step has a complete observation.

    ``subgoal`` default (``UNSET``): per-step ``subgoal`` records the environment ground truth
    ``info["simple_subgoal_online"]``, byte-identical to BASE (the original side only takes this path). With an
    explicit model subgoal (text, or ``None`` = model is waiting): ``subgoal`` records the text converted by
    ``pp_subgoal_to_official`` and ``subgoal_raw`` the raw text; the ground truth is recorded instead as a ``history``
    line with the same step number (``note=oracle_simple_subgoal:<text>``, not recorded for steps with an empty ground
    truth); steps without a complete observation use ``log_missing_step`` instead (C8)."""
    obs, _reward, terminated, truncated, info = out
    info = info if isinstance(info, dict) else {}
    front = wrist = state = None
    if isinstance(obs, dict) and obs.get("front_rgb_list"):
        front = obs["front_rgb_list"][-1]
        if obs.get("wrist_rgb_list"):
            wrist = obs["wrist_rgb_list"][-1]
        if obs.get("joint_state_list") and obs.get("gripper_state_list"):
            state = state8(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
    oracle = info.get("simple_subgoal_online")
    observed = front is not None and wrist is not None and state is not None
    if _is_unset(subgoal):
        trace.log_step(step=step_no, front=front, wrist=wrist, state=state,
                       action=np.asarray(action8, dtype=np.float64), subgoal=None if oracle is None else str(oracle),
                       terminated=bool(terminated), truncated=bool(truncated), status=info.get("status"))
        return observed
    raw = None if subgoal is None else str(subgoal)
    official = pp_subgoal_to_official(raw)
    action = np.asarray(action8, dtype=np.float64)
    link = {k: v for k, v in (link or {}).items() if v is not None}  # source_call_id / chunk_index
    if observed:
        trace.log_step(step=step_no, front=front, wrist=wrist, state=state, action=action, subgoal=official,
                       terminated=bool(terminated), truncated=bool(truncated), status=info.get("status"),
                       subgoal_raw=raw, **link)
    else:
        trace.log_missing_step(step=step_no, action=action, reason="obs_none" if obs is None else "obs_incomplete",
                               subgoal=official, subgoal_raw=raw, env_status=info.get("status"), **link)
    if oracle is not None:
        trace.log_history(step_no, step_no, note=f"{ORACLE_NOTE_PREFIX}{oracle}")
    return observed


class TracedConnection:
    """Wraps the vla-eval ``Connection``: every in-episode protocol frame writes trace request / response lines and
    is counted; other attributes pass through.

    ``SyncEpisodeRunner`` only calls ``start_episode`` / ``act`` / ``end_episode``; the original side hands this
    wrapper to it directly.

    ``last_subgoal``: the ``"subgoal"`` value of the most recent ACTION reply (only present with the server wrapper
    ``pp_server_wrap.py``; text or ``None`` = model is waiting); ``UNSET`` when the reply lacks the key. A read-only
    record that does not affect the trace or return values."""

    def __init__(self, conn: Any, trace=None, lang=None) -> None:
        self._conn = conn
        self.trace = trace if trace is not None else NullTrace()
        self.frames_sent = 0
        self.actions_received = 0
        self.last_subgoal: Any = UNSET
        # language ledger (None means nothing is recorded; not passed on the original side or with the switch off)
        self.lang = lang
        self.lang_errors = 0
        self.last_call_id: str | None = None  # action_model call of the most recent successful reply (source_call_id of executed steps)
        self.last_audit: Any = None           # ``_sgeval_audit`` popped from the most recent reply
        self.demo_index: int | None = None    # index of the last demo frame (initial image) in the trace demo line
        self._failed_step: int | None = None
        self._transport_attempt = 0
        self.timer = ChunkTimer()
        self.step_rtt_ms: list[float] = []
        self._bucket_ms = 0.0
        self._n_fresh = 0
        self._last_n_s1 = 0
        self.gpu_name = None
        self.timing_missing = 0
        self.fresh_mismatch = 0
        self.reconnect_steps: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def start_episode(self, config: dict) -> None:
        self.trace.log_request(EPISODE_START, canonical_frame_bytes(EPISODE_START, config), step=0)
        self.frames_sent += 1
        await self._conn.start_episode(config)

    # -- language ledger helpers: always swallow exceptions and count lang_errors, never changing requests, actions or control flow --
    def _lang(self, fn: str, *a, **k) -> Any:
        if self.lang is None:
            return None
        try:
            return getattr(self.lang, fn)(*a, **k)
        except Exception as e:  # noqa: BLE001
            self.lang_errors += 1
            print(f"TRACE_HOOK_ERROR route=pp/new where=language.{fn} {type(e).__name__}: {e}"[:600], flush=True)
            return None

    def _image_refs(self, obs: dict, step: int) -> list | None:
        imgs = obs.get("images") if isinstance(obs, dict) else None
        if not imgs:
            return None
        h = load_trace_writer().image_sha256
        phase, idx = ("demo", self.demo_index) if step == 0 else ("exec", step)
        refs = []
        for slot, (key, ref, cam) in enumerate((("agentview", "current", "front"), ("wrist", "wrist", "wrist"))):
            if key in imgs:
                refs.append({"slot": slot, "ref": ref, "phase": phase, "frame_idx": idx, "cam": cam,
                             "raw_sha256": h(imgs[key]), "sources": None, "transform": None, "encoded_sha256": None})
        return refs or None

    def _log_generation(self, step: int, gen: Any, obs: Any = None) -> None:
        """Record a full S2 generation block as one ``subgoal_model`` call (server-internal inference, recorded after
        the fact).

        Keys read (written by the wrapper; None when missing): ``context`` / ``prompt`` (full S2 input text,
        including fed-back history), ``images`` (image references), ``text`` (raw full generation block),
        ``reasoning``, ``subgoal_raw`` (raw ``at [x, y]`` text), ``kind`` (``transition`` / ``nontransition``),
        ``committed`` (committed or rolled back). ``parsed`` records the converted subgoal and the whole block
        unchanged; nontransition explicitly records ``text_output=False`` and the ``out`` text as None.

        When the block carries ``input_text`` (input text decoded by the wrapper from S2 context tokens), a
        ``dir=in role=user`` message is written first (images resolved into frame references from ``input_images``,
        see ``_s2_input_images``), ``parsed`` additionally records ``input_decoded_from_tokens=True`` and
        ``input_image_check``, and ``parsed.generation`` drops ``input_text`` (not stored twice); without the key the
        output is byte-identical to before."""
        if not isinstance(gen, dict):
            return
        cid = self._lang("open_call", "subgoal_model", step, params=gen.get("params"))
        if cid is None:
            return
        decoded = isinstance(gen.get("input_text"), str)
        check = None
        if decoded:
            # the S2 input is text decoded by the server wrapper from context tokens (segments added before this
            # fire: the first one includes the task prefix and demo images, later ones the fed-back previous subgoal,
            # cognition placeholders and this observation image). It comes from the reply, so it can only be written
            # after the reply -- the same kind of exception to "persist before sending" as the tokenization channel
            # messages; it is still recorded as dir=in (it describes what went into the model).
            try:
                imgs, demo_video, check = self._s2_input_images(gen.get("input_images"), obs, step)
            except Exception as e:  # noqa: BLE001 ledger helpers never affect control flow
                self.lang_errors += 1
                print(f"TRACE_HOOK_ERROR route=pp/new where=language.s2_images {type(e).__name__}: {e}"[:600],
                      flush=True)
                imgs, demo_video, check = None, None, None
            self._lang("message", cid, dir="in", role="user", text=gen["input_text"], images=imgs,
                       demo_video=demo_video)
        else:
            ctx = gen.get("context", gen.get("prompt"))
            if ctx is not None or gen.get("images"):
                self._lang("message", cid, dir="in", role="user", text=ctx, images=gen.get("images"))
        kind = gen.get("kind", gen.get("transition"))
        nontransition = kind in ("nontransition", False)
        raw = gen.get("subgoal_raw")
        text = None if nontransition else gen.get("text", raw)
        self._lang("message", cid, dir="out", role="assistant", text=text)
        parsed = {"subgoal": pp_subgoal_to_official(None if raw is None else str(raw)), "subgoal_raw": raw,
                  "reasoning": gen.get("reasoning"), "kind": kind, "committed": gen.get("committed"),
                  "text_output": not nontransition and text is not None, "generation": gen}
        if decoded:  # the line schema is frozen: flags and the image check go into parsed; the raw input is already in the in message, so generation does not repeat it
            parsed["generation"] = {k: v for k, v in gen.items() if k != "input_text"}
            parsed["input_decoded_from_tokens"] = True
            parsed["input_image_check"] = check
        self._lang("close_call", cid, status="reply", parsed=parsed)

    def _s2_input_images(self, descs: Any, obs: Any, step: int) -> tuple[list | None, str | None, dict]:
        """The k-th ``<image:k>`` of the S2 input -> an images element (slot=k).

        - ``source=obs``: this observation (S2 fires while processing this observation and the generation block comes
          back with its reply); the frame number follows ``_image_refs`` (step 0 is the last demo frame
          ``demo_index``, k>=1 is after executing step k), and ``raw_sha256`` uses the client's hash of the raw frame
          it sent (i.e. the trace frame hash), checked against the wrapper's ``pixel_sha256``;
        - ``source=demo``: demo images in the S2 prefix (sampled by the server at demo_fps); the frame number is
          matched in order by pixel hash within this observation's ``video_history`` (= the first N frames of the
          trace demo line), with ``ref=keyframe``, ``cam=front``; when unmatched the frame number is None and
          ``raw_sha256`` takes the wrapper hash (the checker reports it as unresolved).
        Returns ``(images, demo_video, check)``; check = ``{"n", "sha_mismatch", "demo_unmatched"}``."""
        check = {"n": 0, "sha_mismatch": 0, "demo_unmatched": 0}
        if not isinstance(descs, list) or not descs:
            return None, None, check
        h = load_trace_writer().image_sha256
        imgs = obs.get("images") if isinstance(obs, dict) else None
        imgs = imgs if isinstance(imgs, dict) else {}
        phase, idx = ("demo", self.demo_index) if step == 0 else ("exec", step)
        vh = obs.get("video_history") if isinstance(obs, dict) else None
        vh_hashes = [h(f) for f in vh] if isinstance(vh, list) and any(
            isinstance(d, dict) and d.get("source") == "demo" for d in descs) else []
        cams = {"agentview": ("current", "front"), "wrist": ("wrist", "wrist")}
        out, demo_idx, nxt = [], [], 0
        for k, d in enumerate(descs):
            d = d if isinstance(d, dict) else {}
            px = d.get("pixel_sha256")
            check["n"] += 1
            if d.get("source") == "obs":
                ref, cam = cams.get(str(d.get("cam_key")), ("current", None))
                raw = h(imgs[d["cam_key"]]) if d.get("cam_key") in imgs else px
                check["sha_mismatch"] += int(px is not None and raw != px)
                out.append({"slot": k, "ref": ref, "phase": phase, "frame_idx": idx, "cam": cam, "raw_sha256": raw,
                            "sources": None, "transform": None, "encoded_sha256": None})
                continue
            fi = None
            if px is not None:
                for i in list(range(nxt, len(vh_hashes))) + list(range(0, min(nxt, len(vh_hashes)))):
                    if vh_hashes[i] == px:
                        fi = i
                        break
            if fi is None:
                check["demo_unmatched"] += 1
            else:
                nxt = fi + 1
                demo_idx.append(fi)
            out.append({"slot": k, "ref": "keyframe", "phase": "demo", "frame_idx": fi, "cam": "front",
                        "raw_sha256": px if fi is None else vh_hashes[fi], "sources": None, "transform": None,
                        "encoded_sha256": None})
        demo_video = f"demo[{min(demo_idx)}:{max(demo_idx) + 1}]" if demo_idx else None
        return out, demo_video, check

    async def act(self, obs: dict) -> dict:
        step = self.actions_received
        self.trace.log_request(OBSERVATION, canonical_frame_bytes(OBSERVATION, obs), step=step)
        self.frames_sent += 1
        cid = None
        if self.lang is not None:  # persist before sending: one action_model call per observation
            self._transport_attempt = self._transport_attempt + 1 if self._failed_step == step else 0
            cid = self._lang("open_call", "action_model", step, transport_attempt=self._transport_attempt)
            if cid is not None:
                fields = {"task_description": obs.get("task_description")} if isinstance(obs, dict) else None
                self._lang("message", cid, dir="in", role="fields", text=fields, images=self._image_refs(obs, step))
        t0 = time.perf_counter()
        self.timer.start(t0)
        try:
            action = await self._conn.act(obs)
        except BaseException:
            if cid is not None:
                self._lang("close_call", cid, status="error")
            self._failed_step = step
            raise
        self._failed_step = None
        # pop the server wrapper audit key before actions reach the environment (it only goes to the language ledger)
        self.last_audit = action.pop(AUDIT_KEY, None) if isinstance(action, dict) else None
        rtt = (time.perf_counter() - t0) * 1000
        self.step_rtt_ms.append(rtt)
        self._account(step, rtt)
        self.actions_received += 1
        if isinstance(action, dict) and SUBGOAL_KEY in action:
            sg = action[SUBGOAL_KEY]
            self.last_subgoal = None if sg is None else str(sg)
        else:
            self.last_subgoal = UNSET
        raw = action.get("actions", action.get("action")) if isinstance(action, dict) else None
        self.trace.log_response(raw, step=step)
        if cid is not None:
            final_text = final_trunc = None
            if self.lang is not None:
                try:
                    final_text, final_trunc = load_trace_writer().audit_channel_messages(self.lang, cid,
                                                                                         self.last_audit)
                except Exception as e:  # noqa: BLE001
                    self.lang_errors += 1
                    print(f"TRACE_HOOK_ERROR route=pp/new where=language.audit {type(e).__name__}: {e}"[:600],
                          flush=True)
            # channels the server wrapper marked text_reconstructed (the S1 raw prompt is rebuilt; only token_ids /
            # mask are truly captured): the message line schema is frozen with no free fields, so the flag goes into
            # this call's call_close.parsed (without the flag parsed stays None, as before)
            chans = self.last_audit.get("channels") if isinstance(self.last_audit, dict) else None
            recon = sorted({str(c.get("channel")) for c in (chans or [])
                            if isinstance(c, dict) and c.get("text_reconstructed")})
            self._lang("close_call", cid, status="reply", server_final_text=final_text,
                       server_truncated=final_trunc,
                       parsed={"text_reconstructed_channels": recon} if recon else None)
            gen = self.last_audit.get("pp_generation") if isinstance(self.last_audit, dict) else None
            for g in (gen if isinstance(gen, list) else [gen] if gen is not None else []):
                self._log_generation(step, g, obs)
        self.last_call_id = cid
        return action

    def _account(self, step: int, rtt: float) -> None:
        """Assign S2 time to the next fresh S1 chunk, including initial hold steps."""
        pt = self.last_audit.get("pp_timing") if isinstance(self.last_audit, dict) else None
        if not isinstance(pt, dict):
            self.timing_missing += 1
            return
        s2 = pt["s2_infer_ms"] or 0.0
        if pt["s2_fired"]:
            self.timer.add_lang(s2, kind="s2", env_step=step, s2_fired=True)
        fresh = pt["chunk_fresh"] and pt["n_s1_fires"] > self._last_n_s1
        if pt["chunk_fresh"] and not fresh:
            self.fresh_mismatch += 1
        if fresh:
            self.timer.close(decision=self._n_fresh, env_step=step, action_rtt_ms=rtt - s2,
                             server_infer_ms=pt["s1_infer_ms"], decision_wall_ms=rtt + self._bucket_ms,
                             extra={"step_rtt_ms": rtt, "s2_infer_ms_at_fresh": s2,
                                    "bucket_ms": self._bucket_ms, "n_s2_fires": pt["n_s2_fires"]})
            self._n_fresh += 1
            self._bucket_ms = 0.0
            self._last_n_s1 = pt["n_s1_fires"]
        else:
            self._bucket_ms += s2
        self.gpu_name = pt.get("gpu") or self.gpu_name

    async def end_episode(self, result: dict) -> None:
        self.trace.log_request(EPISODE_END, canonical_frame_bytes(EPISODE_END, result), step=self.actions_received)
        self.frames_sent += 1
        await self._conn.end_episode(result)


def resolve_trace_path(identity: dict, conn_info: dict, recorder: Any) -> Path | None:
    """Trace file location: ``conn_info["trace_path"]``; otherwise ``trace.jsonl`` under ``conn_info["trace_dir"]``
    (the **per-episode directory** ``<trace-root>/<key>.a<attempt>`` given by the seat runner, not created in advance;
    TraceWriter creates the parent; None when ``--trace-root`` is not given); otherwise ``trace.jsonl`` in the
    recorder directory; if none applies no trace is written."""
    if conn_info.get("trace_path"):
        return Path(conn_info["trace_path"])
    if conn_info.get("trace_dir"):
        return Path(conn_info["trace_dir"]) / "trace.jsonl"
    rec_dir = getattr(recorder, "out_dir", None)
    if rec_dir:
        return Path(rec_dir) / "trace.jsonl"
    return None


def is_connection_closed(exc: BaseException) -> bool:
    try:
        import websockets.exceptions as wse
    except Exception:  # noqa: BLE001
        return type(exc).__name__ == "ConnectionClosed"
    return isinstance(exc, wse.ConnectionClosed)


def classify_exception(exc: BaseException) -> tuple[str, str | None, bool]:
    """``(status, infra_reason, infra)``: connection / server errors are infrastructure failures, ``StepCapReached``
    records timeout, everything else records error."""
    name = type(exc).__name__
    if name == "StepCapReached":
        return "timeout", None, False
    if name == "SubgoalMissing":
        return "error", "pp_subgoal_missing", True
    if name == "RecorderError":
        return "error", "recorder", True
    if isinstance(exc, ConnectionError):
        return "error", "pp_unreachable", True
    if isinstance(exc, TimeoutError):
        return "error", "pp_act_timeout", True
    if is_connection_closed(exc):
        return "error", "pp_connection_closed", True
    if isinstance(exc, RuntimeError) and str(exc).startswith("Server error"):
        return "error", "pp_server_error", True
    return "error", None, False


def default_connection_factory(url: str, timeout: float) -> Any:
    """Import vla-eval only at runtime (in the client extension environment client-env)."""
    from vla_eval.connection import Connection

    return Connection(url, timeout=timeout)


# -- one new-side episode ----------------------------------------------------------------


class _EnvError(Exception):
    """Wraps exceptions raised on the environment side in ``session.step`` / ``session.reset``, to tell them apart
    from connection exceptions."""

    def __init__(self, inner: BaseException):
        super().__init__(f"{type(inner).__name__}: {inner}")
        self.inner = inner


def run_episode(session, identity: dict, conn_info: dict, recorder, *,
                connection_factory: Callable[[str, float], Any] | None = None) -> dict:
    """Entry point called by the seat runner: one PonderPounce episode. ``session`` is an already built EnvSession
    (or a stand-in with the same interface)."""
    import anyio

    return anyio.run(_run_episode_async, session, identity, conn_info, recorder,
                     connection_factory or default_connection_factory)


async def _run_episode_async(session, identity: dict, conn_info: dict, recorder, connection_factory) -> dict:
    timing: dict[str, Any] = {}
    max_steps = int(conn_info["max_steps"])
    dataset = conn_info.get("dataset")
    sid = fixed_sid(identity, dataset)
    eid = sid
    task = identity["task"]
    ep_idx = episode_idx_of(identity)
    max_reconnects = int(conn_info.get("pp_max_reconnects", PP_MAX_RECONNECTS))
    url = f"ws://{conn_info.get('host', '127.0.0.1')}:{int(conn_info['port'])}"
    trace_path = resolve_trace_path(identity, conn_info, recorder)
    phase2 = phase2_enabled(conn_info)
    if trace_path is not None:
        TraceWriter = load_trace_writer().TraceWriter
        ident_t = {"task": task, "tier": identity.get("tier"), "seed": identity.get("seed"),
                   "source_episode": identity.get("source_episode"),
                   "builder_episode": identity.get("builder_episode"),
                   "key": identity.get("key"), "dataset": dataset, "sid": sid,
                   "episode_idx": ep_idx, "side": "new"}
        # with the switch off (old behavior) full arrays are not collected: the episode directory only has trace.jsonl and the end line has no arrays summary, byte-identical to BASE
        tw_kw: dict[str, Any] = {"collect_arrays": phase2}
        if phase2:  # C6: add the attempt number (with the switch off identity is byte-identical to BASE)
            ident_t["attempt"] = attempt_of(identity, conn_info, trace_path)
            seed = conn_info.get("policy_seed", identity.get("policy_seed"))
            if seed is not None:  # identity / header record policy_seed; not written when absent
                ident_t["policy_seed"] = int(seed)
                tw_kw["policy_seed"] = int(seed)
            if conn_info.get("effective_cap") is not None:
                tw_kw["effective_cap"] = int(conn_info["effective_cap"])
        trace = TraceWriter(trace_path, route=TRACE_ROUTE_NEW_C1 if phase2 else TRACE_SCHEMA_ROUTE_NEW,
                            max_steps=max_steps, identity=ident_t, **tw_kw)
    else:
        trace = NullTrace()
    lang = None
    if phase2 and trace_path is not None:  # language ledger: language.jsonl next to the trace
        try:
            LanguageLog = getattr(load_trace_writer(), "LanguageLog", None)
            lang = LanguageLog(trace_path.parent / "language.jsonl") if LanguageLog is not None else None
        except Exception as e:  # noqa: BLE001
            lang = None
            print(f"TRACE_HOOK_ERROR route=pp/new where=language.init {type(e).__name__}: {e}"[:600], flush=True)

    result: dict[str, Any] = {"side": "new", "sid": sid, "eid": eid, "episode_idx": ep_idx, "max_steps": max_steps,
                              "reconnects": 0, "frames_sent": 0, "decisions": 0, "steps": 0, "demo_frames": None}
    packer = ObsPacker()
    conn = connection_factory(url, PP_TIMEOUT_S)
    tconn = TracedConnection(conn, trace, lang=lang)
    status, error, infra, infra_reason, env_exc = "error", None, False, None, None
    executed = 0
    attempted = observed = 0  # C8 (only written into end with the switch on)
    # C4: original actions given to the environment are collected per step by TraceWriter and written to arrays.npz via merge_write_npz at finalization (no direct np.savez)
    model_subgoal: Any = UNSET  # the model subgoal in this step's reply
    trace_reason = None
    t_start = time.perf_counter()
    try:
        t0 = time.perf_counter()
        await conn.connect(benchmark=PP_BENCHMARK)
        timing["connect_s"] = time.perf_counter() - t0

        bench_t0 = time.monotonic()  # StepBenchmark.start_episode takes _t0 before reset
        try:
            raw_obs, info = session.reset()
        except Exception as e:  # noqa: BLE001
            raise _EnvError(e) from e
        packer.on_reset(raw_obs, info)
        result["demo_frames"] = trace_reset(trace, raw_obs, packer.task_description, include_initial=phase2)
        tconn.demo_index = result["demo_frames"]  # with include_initial the initial image's index in the demo line = demo frame count
        obs = packer.make(raw_obs)
        start = episode_start_payload(task, ep_idx, sid, eid)
        await tconn.start_episode(start)
        _rec_event(recorder, {"kind": "pp_episode_start", "sid": sid, "eid": eid, "task": task,
                              "episode_idx": ep_idx, "max_steps": max_steps, "url": url})

        last_info: dict = {}
        done = truncated = False
        step = -1
        for step in range(max_steps):
            action = await _act_with_reconnect(tconn, conn, obs, result, max_reconnects, recorder)
            raw = action.get("actions", action.get("action")) if isinstance(action, dict) else None
            if raw is not None:
                _rec_array(recorder, "model_action", np.array(raw, copy=True), step)
            a8 = exec_action8(action)
            if phase2:
                if _is_unset(tconn.last_subgoal):
                    raise SubgoalMissing(f"extended switch is on but ACTION reply {step + 1} has no {SUBGOAL_KEY!r} key"
                                         " (the server wrapper pp_server_wrap.py is not in effect)")
                model_subgoal = tconn.last_subgoal
            link = {"source_call_id": tconn.last_call_id, "chunk_index": 0} \
                if tconn.last_call_id is not None else {}
            try:
                out = session.step(a8)
            except Exception as e:  # noqa: BLE001
                if phase2 and type(e).__name__ != "StepCapReached":  # an exception step that reached the environment (C8: includes exception steps)
                    attempted += 1
                    trace.log_missing_step(step=attempted, action=np.asarray(a8, dtype=np.float64),
                                           reason=f"env_exception:{type(e).__name__}",
                                           subgoal=pp_subgoal_to_official(model_subgoal), subgoal_raw=model_subgoal,
                                           **link)
                raise _EnvError(e) from e
            executed += 1
            if phase2:
                attempted += 1
                observed += bool(trace_step(trace, executed, out, a8, subgoal=model_subgoal, link=link))
            else:
                trace_step(trace, executed, out, a8)
            raw_obs, _reward, terminated, truncated, last_info = out
            last_info = last_info if isinstance(last_info, dict) else {}
            done = step_done(terminated, truncated, last_info)
            if done:
                break
            obs = packer.make(raw_obs)

        status, error = terminal_status(done, bool(truncated), last_info)
        trace_reason = "env_done" if done else "loop_exit"
        elapsed = time.monotonic() - bench_t0
        await tconn.end_episode(episode_end_payload(last_info.get("status") == "success", step + 1, elapsed))
    except _EnvError as e:
        inner = e.inner
        status, infra_reason, infra = classify_exception(inner)
        error = None if status == "timeout" else f"{type(inner).__name__}: {inner}"[:800]
        env_exc = f"{type(inner).__name__}: {inner}"[:800]
        trace_reason = "step_cap" if status == "timeout" else "env_exception"
    except Exception as e:  # noqa: BLE001 connection, server or other exceptions
        status, infra_reason, infra = classify_exception(e)
        error = f"{type(e).__name__}: {e}"[:800]
        trace_reason = f"exception:{type(e).__name__}"
    finally:
        try:
            await conn.close()
        except Exception:  # noqa: BLE001
            pass
    timing["episode_s"] = time.perf_counter() - t_start
    attach_policy_timing(timing, tconn.timer, gpu_name=tconn.gpu_name, model_kind="pp")
    timing.update(step_rtt_ms=tconn.step_rtt_ms, pp_timing_missing=tconn.timing_missing,
                  reconnect_steps=tconn.reconnect_steps, reconnected=bool(tconn.reconnect_steps),
                  fresh_mismatch=tconn.fresh_mismatch)
    result.update(status=status, task_success=status == "success", steps=executed, error=error, infra=bool(infra),
                  infra_reason=infra_reason, env_exception=env_exc, frames_sent=tconn.frames_sent,
                  decisions=tconn.actions_received, timing=timing)
    if not phase2:
        trace.close(status=status, terminal_reason=trace_reason, sid=sid, frames_sent=tconn.frames_sent,
                    reconnects=result["reconnects"])
    else:
        no_frame = result["demo_frames"] is None  # failed before reset: no image at all (C3)
        demo = 0 if no_frame else int(result["demo_frames"])
        end_extra: dict[str, Any] = {
            "sid": sid, "frames_sent": tconn.frames_sent, "reconnects": result["reconnects"], "side": "new",
            "exit_reason": trace_reason, "demo_frames": demo, "steps_attempted": attempted,
            "steps_observed": observed, "frames_recorded": 0 if no_frame else demo + 1 + observed}
        if no_frame:
            end_extra["no_frame"] = True
        if lang is not None:  # close the language ledger first (dangling calls get cancelled)
            try:
                lang.close()
            except Exception as e:  # noqa: BLE001
                tconn.lang_errors += 1
                print(f"TRACE_HOOK_ERROR route=pp/new where=language.close {type(e).__name__}: {e}"[:600], flush=True)
            end_extra["language_hook_errors"] = tconn.lang_errors
        # arrays.npz: written by TraceWriter.close via merge_write_npz (same-named keys of a recorder in the same
        # directory merge with equal values; separate directories each write their own); a write failure is recorded
        # in end.arrays.error and judged as a failure by the TRACE_ARRAYS checker
        trace.close(status=status, terminal_reason=status, **end_extra)
    _rec_event(recorder, {"kind": "pp_episode_end", "sid": sid, "status": status, "steps": executed,
                          "frames_sent": tconn.frames_sent, "reconnects": result["reconnects"], "error": error})
    return result


async def _act_with_reconnect(tconn: TracedConnection, conn: Any, obs: dict, result: dict, max_reconnects: int,
                              recorder) -> dict:
    """ConnectionClosed within one episode: ``reconnect()`` (including HELLO), then resend the same observation,
    without resending EPISODE_START."""
    while True:
        try:
            return await tconn.act(obs)
        except Exception as e:  # noqa: BLE001
            if not is_connection_closed(e) or result["reconnects"] >= max_reconnects:
                raise
            result["reconnects"] += 1
            tconn.reconnect_steps.append(tconn.actions_received)
            _rec_event(recorder, {"kind": "pp_reconnect", "n": result["reconnects"], "error": repr(e)[:400],
                                  "decision": tconn.actions_received})
            await conn.reconnect()


def _same_dir(a: Any, b: Any) -> bool:
    """Whether two directories are the same path (False when ``b`` is empty); used to tell whether the trace is
    written in the recorder directory."""
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:  # noqa: BLE001
        return False


def _rec_event(recorder, event: dict) -> None:
    if recorder is not None and hasattr(recorder, "add_event"):
        recorder.add_event(event)


def _rec_array(recorder, name: str, arr: np.ndarray, step: int) -> None:
    if recorder is not None and hasattr(recorder, "add_array"):
        recorder.add_array(name, arr, step=step)


# -- new interface: the four model-side methods --------------------------------------

from robomme_ood_eval import servers as _servers  # noqa: E402
from robomme_ood_eval.policy import Ready  # noqa: E402


def pp_server_spec(policy, ckpt: Any) -> tuple[list, dict, Path]:
    """Command, environment and cwd of the PonderPounce server, copied from the pp branch of the legacy
    ``run_seat.sh::build_server_cmd``: cwd is ``third_party/PonderPounce``, HF offline (``HF_HOME`` inherited from the
    caller's environment); with the wrapper on (default; the legacy launcher's ``SGEVAL_PP_SERVER_WRAP=1``) this
    package's ``servers/pp_server_wrap.py`` is started by absolute path, otherwise ``-m
    ponderpounce.eval.robomme_server``; followed by ``--args.checkpoint_path --args.seed --args.device cuda:0
    --port``. The interpreter comes from ``cfg["pp_py"]`` -> env var ``PP_PY`` ->
    ``third_party/PonderPounce/.venv/bin/python``."""
    S = _servers
    cfg = policy.cfg
    sub = policy.root / "third_party" / "PonderPounce"
    py = S.interpreter(cfg, "pp_py", "PP_PY", sub / ".venv" / "bin" / "python")
    env = {"PYTHONUNBUFFERED": "1",
           "HF_HUB_OFFLINE": str(cfg.get("hf_hub_offline", os.environ.get("HF_HUB_OFFLINE", "1"))),
           "TRANSFORMERS_OFFLINE": str(cfg.get("transformers_offline", os.environ.get("TRANSFORMERS_OFFLINE", "1"))),
           PHASE2_ENV: "1" if policy.server_wrap else "0"}
    if policy.server_wrap:
        argv = [str(py), str(S.SERVERS_DIR / "pp_server_wrap.py"), f"--sgeval-metadata-out={policy.wrap_meta}"]
    else:
        argv = [str(py), "-m", "ponderpounce.eval.robomme_server"]
    argv += ["--args.checkpoint_path", str(ckpt), "--args.seed", str(int(policy.policy_seed)), "--args.device",
             "cuda:0", "--port", str(int(policy.port))]
    return argv, env, sub


class PPPolicy(_servers.ServedPolicy):
    """PonderPounce (``pp``).

    * ``load``: preflight (submodule, interpreter, ckpt with ``norm_stats.json``, wrapper) -> start (or attach) the
      server, ready = ``GET /health`` returns 200; no warm-up; checks this process can import ``vla_eval`` (the
      client must run in ``envs/client-env``);
    * ``reset(spec)``: sends no message and never touches the environment -- after the liveness probe computes this
      episode's sid ``<task>|<source_episode or tier>|<seed>``; only if this server process has already used it (which
      only happens when rerunning the same identity, ``spec.attempt > 1``) is the server restarted, guaranteeing
      ``n`` starts at 0 when the server derives noise from ``crc32(f"{seed}:{sid}:{n}")``;
    * ``play``: calls this module's ``run_episode`` unchanged (``EPISODE_START`` is still sent every episode, and a
      dropped connection is reconnected only once within the episode); the result additionally records ``pp_sid``
      and ``pp_sid_use_index`` (the use index of that sid in this server process);
    * ``close``: stop the server process group (base class).

    Required cfg: ``ckpt`` (PonderPounce has no default ckpt). Optional ``pp_py``, ``pp_server_wrap`` (on by
    default), ``pp_max_reconnects`` (default 1), ``hf_hub_offline``, ``transformers_offline``;
    ``connection_factory`` is only for unit test injection."""

    model = "pp"
    requires_ckpt = True

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        self.server_wrap = _servers.flag_on(self.cfg.get("pp_server_wrap", True))
        self.ckpt = self.cfg.get("ckpt")  # no default checkpoint; load() rejects a missing one
        self.sid_uses: dict[str, int] = {}   # how many times each sid has sent EPISODE_START in this server process
        self.server_restarts = 0
        self.current_sid: str | None = None
        self._spec_argv: tuple | None = None

    def load(self) -> None:
        S = _servers
        if not self.ckpt:
            raise S.PreflightError("RUN_BLOCKED reason=pp_ckpt_missing ckpt=unset (PonderPounce requires an explicit ckpt)")
        self._pick_port()
        argv, env, cwd = pp_server_spec(self, self.ckpt)
        if self.preflight:
            S.preflight_pp(self.root, self.ckpt, Path(argv[0]), server_wrap=self.server_wrap, seed=self.policy_seed)
            import importlib.util

            if self.cfg.get("connection_factory") is None and importlib.util.find_spec("vla_eval") is None:
                raise S.PreflightError("RUN_BLOCKED reason=client_env detail=this process cannot import vla_eval (the "
                                       "PonderPounce client must run with the envs/client-env/.venv interpreter)")
        self._spec_argv = (argv, env, cwd)
        self._launch(argv, env, cwd, [Ready.health()], self.ckpt)
        self._fingerprint(self.ckpt)

    def _restart_server(self, why: str) -> None:
        """Restart the server (same port, same command): stop the old process group, start a new one and wait for
        ``/health``, and clear this process's sid usage table."""
        argv, env, cwd = self._spec_argv
        print(f"PP_SERVER_RESTART reason={why} port={self.port}", flush=True)
        if self.server is not None:
            self.server.stop()
        self._launch(argv, env, cwd, [Ready.health()], self.ckpt)
        self.sid_uses.clear()
        self.server_restarts += 1

    def reset(self, spec) -> None:
        super().reset(spec)
        sid = fixed_sid(spec.identity(), spec.dataset)
        if self.sid_uses.get(sid, 0) > 0:
            self._restart_server(f"sid_reused sid={sid} attempt={spec.attempt}")
        self.current_sid = sid

    def play(self, session, spec, recorder) -> dict:
        sid = fixed_sid(spec.identity(), spec.dataset)
        use_index = self.sid_uses.get(sid, 0)
        conn = self._conn_info(spec)
        conn.update(pp_phase2=self.server_wrap, pp_max_reconnects=int(self.cfg.get("pp_max_reconnects",
                                                                                   PP_MAX_RECONNECTS)))
        try:
            res = run_episode(session, spec.identity(), conn, recorder,
                              connection_factory=self.cfg.get("connection_factory"))
        finally:
            self.sid_uses[sid] = use_index + 1  # counted conservatively: this episode may already have sent EPISODE_START
        res.update(pp_sid=sid, pp_sid_use_index=use_index, pp_server_restarts=self.server_restarts)
        return self._finish_play(res)
