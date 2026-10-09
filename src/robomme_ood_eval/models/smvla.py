"""SimpleMemVLA client for the new interface (runs in the benchmark .venv).

``run_episode(session, identity, conn_info, recorder)`` reproduces line by line the old official
``4e0c04f robomme_sim/eval_success.py::run_group`` (group size 1, details mode), the per-episode finalization of
``evaluate_manifest``, and the reset/step packing semantics of ``robomme_sim/robomme_env.py::SimEnvService``; the
environment is built in this process by the environment session, and the policy runs in the smvla_server process.

Correspondence with the old official code (line numbers refer to SimpleMemVLA-official-xhard0 @4e0c04f):
- reset: SimEnvService.reset (reset_retries=2 retries; empty frames/states count as failure; instruction is
  info["task_goal"][0]) -> send reset + observe(all demo frames + the initial frame), cur_state = states[-1]
  (float32).
- Decision loop: hard_bound = ceil(max_steps/16)+2 (84 for 1300 steps); each decision infer -> chunk =
  actions[:16] -> SimEnvService.step row by row ``np.asarray(row, float64).reshape(-1)[:8]``, stopping on
  done / error / obs None; returned frames are always observed; steps += consumed.
- Terminal status: when the environment reports done, take the environment status (with error_message on error);
  a step exception records ``step_exc: <e>`` (steps not incremented by this chunk); when decisions run out without a
  terminal status from the environment, record timeout with error
  ``policy loop hard_bound=84 exhausted, environment reported no terminal status`` (steps usually 1344); other
  exceptions record error + traceback.

The pure-function layer (encode_frames / encode_states / instruction_from_info / step_chunk / hard_bound) is separate
from the IO layer (WSPolicyConn); unit tests drive them directly with fake sessions / fake connections.

Trace (shared contracts C1-C11 in the ``trace_writer.py`` module docstring): each episode also writes ``trace.jsonl``
(route ``smvla/new``), located by the ``groundsg_client.trace_location`` convention (``trace_path`` ->
``<trace_dir>/trace.jsonl`` -> ``<recorder.out_dir>/trace.jsonl``; nothing is written if none is given).
``PolicyTrace`` is the recorder shared by the two new-side routes SimpleMemVLA / FrameSamp+Modulation
(``framesamp_modul_client`` reuses it by file path):

- the demo segment records all reset frames per C2 (including the initial frame), finalizing with
  ``demo_frames = frame count - 1``;
- one line per step given to the environment: with an observation ``log_step`` (the last frame when one step
  returns several), without a valid observation (step raised and the environment side counted the step,
  ``obs is None``, ``status == "error"``) ``log_missing_step``; ``steps_attempted`` follows the increase of the
  environment session's ``steps``, same basis as the result row ``exec_steps`` (C8);
- subgoal: SimpleMemVLA takes ``subtask`` from the decision reply (the ``infer`` reply key of ``smvla_server.py``),
  FrameSamp+Modulation is ``None`` throughout (C7);
- requests / responses (C10): SimpleMemVLA records the logical input (instruction, state, sequence of hashes of the
  frames observed since the previous inference) and the full action chunk ``actions_full``; FrameSamp+Modulation
  records the sha256 of the raw msgpack frame bytes and the ``infer`` reply action chunk;
- a strict-cap hit (``StepCapReached`` or the session's ``cap_hit``) always finalizes as ``timeout`` (C3);
- full numeric values: per-step original actions / states are collected by ``TraceWriter`` and at finalization
  written via ``trace_writer.merge_write_npz`` to ``arrays.npz`` in the trace directory
  (``exec_action__%05d`` / ``exec_state__%05d``); when in the same directory as the recorder they merge with its
  ``exec_action__%05d`` (same keys, same values) and neither overwrites the other in any order; this module no longer
  calls ``np.savez`` directly;
- language ledger: ``language.jsonl`` next to the trace. SimpleMemVLA has one ``action_model`` call per decision:
  ``in`` (``role=fields``, the task goal) is written before sending; after the reply, the per-channel tokenization
  of the server wrapper audit block and ``out`` (``role=assistant``, raw subtask text) are written, and
  ``close_call`` records ``server_final_text`` (the server's templated prompt; None without an audit block); executed
  steps point to the call via ``source_call_id`` / ``chunk_index``. ``_sgeval_audit`` in replies is always popped
  first and never reaches actions or request bytes;
- any exception inside the recorder only increments ``observer_hook_errors`` and prints ``TRACE_HOOK_ERROR``, never
  changing the requests sent to the server, the actions given to the environment or the control flow (checked later
  by ``CLIENT_REPLAY_EQ`` replay comparison).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

MAX_STEPS = 1300
EXECUTE_HORIZON = 16
RESET_RETRIES = 2
FINAL_STATUSES = ("success", "fail", "timeout")
CAM_FRONT = "front"
CAM_WRIST = "wrist"


class ProtocolError(RuntimeError):
    """Message protocol broken (reply sha does not match what was sent, reply missing keys). Blocks the run."""


class ServerError(RuntimeError):
    """Server replied {"error": traceback}: a server-side exception, not an environment outcome."""


# infrastructure-failure markers (same table as framesamp_modul_client.INFRA_MARKERS): only decide the infra flag
# (the outer loop retries based on it); status is unchanged.
INFRA_MARKERS = ("RecorderError", "svulkan2", "EXCLUSIVE", "Vulkan", "vk::", "out of memory", "RESOURCE_EXHAUSTED",
                 "CUDA_ERROR", "ConnectionClosed", "ConnectionRefused", "InvalidStatus", "Connection reset")
ENV_RESET_INFRA_MARKERS = ("svulkan2", "EXCLUSIVE", "Vulkan", "vk::")
SERVER_OOM_MARKERS = ("out of memory", "OutOfMemory", "CUDA", "CUBLAS", "cuDNN")


def _marker(text: str | None, markers) -> str | None:
    for m in markers:
        if m in (text or ""):
            return m
    return None


def classify_exception(e: BaseException, tb: str) -> str | None:
    """Episode-level catch-all exception -> infra_reason (None means not infrastructure)."""
    if isinstance(e, ServerError):
        return "server_oom" if _marker(str(e), SERVER_OOM_MARKERS) else "server_error"
    if isinstance(e, ProtocolError):
        return "protocol"
    try:
        from websockets.exceptions import ConnectionClosed, InvalidHandshake
        if isinstance(e, (ConnectionClosed, InvalidHandshake)):
            return f"connection:{type(e).__name__}"
    except ImportError:
        pass
    if isinstance(e, (OSError, TimeoutError)):  # ConnectionRefusedError / reset / connection timeout
        return f"connection:{type(e).__name__}"
    m = _marker(tb, INFRA_MARKERS)
    return f"marker:{m}" if m else None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frame_sha(frame: np.ndarray) -> str:
    """Single-frame fingerprint (same definition as recorder.frame_sha256 and smvla_server.frame_sha)."""
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def array_sha(arr: np.ndarray) -> str:
    """Numeric array fingerprint (same definition as recorder.array_sha256 and smvla_server.array_sha)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(a.tobytes())
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    return h.hexdigest()


def hard_bound(max_steps: int = MAX_STEPS, execute_horizon: int = EXECUTE_HORIZON) -> int:
    """Copied from run_group: ``max(1, -(-int(args.max_steps) // max(1, args.execute_horizon))) + 2``."""
    return max(1, -(-int(max_steps) // max(1, execute_horizon))) + 2


def timeout_error(hb: int) -> str:
    return f"policy loop hard_bound={hb} exhausted, environment reported no terminal status"


# ---- the following five functions are copied line by line from robomme_sim/robomme_env.py (c564c17 = 4e0c04f) lines 33-75 ----
def _to_uint8_hwc(img) -> np.ndarray:
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.uint8, copy=False)


def _to_f32(vec) -> np.ndarray:
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def encode_frames(obs: dict) -> list[dict[str, np.ndarray]]:
    fronts = obs["front_rgb_list"]
    wrists = obs["wrist_rgb_list"]
    return [
        {CAM_FRONT: _to_uint8_hwc(f), CAM_WRIST: _to_uint8_hwc(w)}
        for f, w in zip(fronts, wrists)
    ]


def encode_states(obs: dict) -> list[np.ndarray]:
    joints = obs["joint_state_list"]
    grippers = obs["gripper_state_list"]
    states = []
    for j, g in zip(joints, grippers):
        jv = _to_f32(j)[:7]
        gv = _to_f32(g)
        g0 = gv[:1] if gv.size else np.zeros(1, dtype=np.float32)
        states.append(np.concatenate([jv, g0]).astype(np.float32))
    return states


def _scalar(x) -> float:
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            return float(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x).reshape(-1)[0])
    return float(x)
# ---- end of copy ----


def instruction_from_info(info: dict, task_name: str) -> str:
    """Copied from the instruction branch of SimEnvService.reset."""
    task_goal = info.get("task_goal")
    if isinstance(task_goal, (list, tuple)) and task_goal:
        return str(task_goal[0])
    return str(task_goal) if task_goal else f"Complete the {task_name} task."


def reset_session(session, task_name: str, retries: int = RESET_RETRIES) -> dict:
    """Copied from SimEnvService.reset: at most retries+1 attempts; returns {ok, instruction, frames, states} or
    {ok: False, reason}."""
    last_err = "unknown"
    for _attempt in range(retries + 1):
        try:
            obs, info = session.reset()
            frames = encode_frames(obs)
            states = encode_states(obs)
            if not frames or not states:
                raise RuntimeError("reset returned no frames")
            return {"ok": True, "instruction": instruction_from_info(info or {}, task_name),
                    "frames": frames, "states": states, "attempts": _attempt + 1}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            try:  # old behavior: env.close() after an error and clear it; the next attempt rebuilds
                session.close()
            except Exception:
                pass
    return {"ok": False, "reason": f"reset_failed: {last_err}"}


def step_chunk(session, action_chunk, on_exec=None, on_obs=None) -> dict:
    """Copied from SimEnvService.step + RoboMMESimEnv.step_one (row by row, stopping on done/error/obs None).

    on_exec(act8) is called before each row is given to the environment (records the action actually executed).
    on_obs(step, front, wrist, state, terminated, truncated, *, status, error) is called after each ``session.step``
    returns (per-step recording): ``step`` is the step index within this chunk (from 1, i.e. the current consumed);
    when one step returns several frames the front, wrist images and 8-d state of the last frame are used; when
    ``status == "error"`` or ``obs is None`` images and state are passed as ``None`` (step without observation). Not
    called when ``session.step`` raises (the caller handles it by environment-side step counting). The callbacks are
    read-only and never change this function's return value or control flow.
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    frames, states, consumed = [], [], 0
    error = None
    status = "ongoing"
    done = False
    for action in action_chunk:
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] < 8:
            raise ValueError(f"Expected an 8-dim joint_angle action, got shape {act.shape}")
        if on_exec is not None:
            on_exec(act[:8])
        obs, _reward, terminated, truncated, info = session.step(act[:8])
        term = bool(_scalar(terminated)) if terminated is not None else False
        trunc = bool(_scalar(truncated)) if truncated is not None else False
        info = info if isinstance(info, dict) else {}
        status = str(info.get("status", "ongoing"))
        done = term or trunc or status in ("success", "fail", "timeout", "error")
        consumed += 1
        if status == "error" or obs is None:
            error = str(info.get("error_message", "env step error"))
            if on_obs is not None:
                on_obs(consumed, None, None, None, term, trunc, status=status, error=error)
            break
        fr = encode_frames(obs)
        st = encode_states(obs)
        frames.extend(fr)
        states.extend(st)
        if on_obs is not None:
            on_obs(consumed, fr[-1][CAM_FRONT] if fr else None, fr[-1][CAM_WRIST] if fr else None,
                   st[-1] if st else None, term, trunc, status=status, error=None)
        if done:
            break
    return {
        "frames": frames,
        "states": states,
        "consumed": consumed,
        "done": bool(done or error is not None),
        "success": status == "success",
        "status": status,
        "error_message": error,
    }


class WSPolicyConn:
    """Synchronous websocket connection to smvla_server (msgpack_numpy; compression=None, max_size=None, no proxy)."""

    def __init__(self, host: str, port: int, open_timeout: float = 600.0):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        self._m = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._ws = connect(f"ws://{host}:{int(port)}", compression=None, max_size=None, proxy=None,
                           open_timeout=open_timeout, ping_interval=None, ping_timeout=None)
        self.metadata = self._m.unpackb(self._ws.recv())

    def call(self, msg: dict) -> tuple[dict, bytes, bytes]:
        raw = self._packer.pack(msg)
        self._ws.send(raw)
        rep_raw = self._ws.recv()
        if isinstance(rep_raw, str):
            raise ProtocolError(f"server returned a text frame: {rep_raw[:2000]}")
        return self._m.unpackb(rep_raw), raw, rep_raw

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


class _NullRecorder:
    def set_phase(self, phase):
        pass

    def add_frames(self, stream, frames, *, tag=""):
        return []

    def add_array(self, name, arr, *, step=None):
        pass

    def add_event(self, event):
        pass


def episode_key(identity: dict) -> str:
    return f"{identity['task']}/{identity['source_episode']}/{identity['seed']}"


# ---- per-episode trace recorder shared by the two new-side routes (smvla / perceptual-framesamp-modul) ----

TRACE_TERMINALS = ("success", "fail", "timeout", "error")
#: executed-action key in arrays.npz (C4); literally the same name as recorder._write_arrays' f"{name}__{k:05d}"
#: (name="exec_action")
ACTION_KEY = "exec_action__%05d"
#: server wrapper reply audit key (same as trace_writer.AUDIT_KEY)
AUDIT_KEY = "_sgeval_audit"
_TAG_ATTEMPT = re.compile(r"\.a(\d+)$")


#: legacy sibling module name -> module in the evaluation package (no longer loaded from each other by file path)
SIBLINGS = {"smvla_client": "robomme_ood_eval.models.smvla",
            "groundsg_client": "robomme_ood_eval.models.groundsg",
            "framesamp_modul_client": "robomme_ood_eval.models.framesamp_modul",
            "official_defs": "robomme_ood_eval.models._official_defs",
            "trace_writer": "robomme_ood_eval.record.trace_writer"}


def load_sibling(name: str):
    """Return the evaluation-package module corresponding to a legacy sibling module (package import; reused if
    already imported)."""
    if name not in SIBLINGS:
        raise ModuleNotFoundError(f"unknown sibling module {name!r}; choices: {', '.join(SIBLINGS)}")
    return importlib.import_module(SIBLINGS[name])


def episode_attempt(tag: str | None) -> int:
    """Attempt number N from an episode directory name ``<key>.a<N>``; 1 without a suffix (consistent with the
    groundsg_client default ``f"{key}.a1"``)."""
    m = _TAG_ATTEMPT.search(str(tag or ""))
    return int(m.group(1)) if m else 1


def recorder_writes_arrays(recorder: Any) -> bool:
    """Whether the recorder writes ``arrays.npz`` (with ``exec_action__%05d``) in ``out_dir``: the real
    ``recorder.EpisodeRecorder`` does."""
    return recorder is not None and getattr(recorder, "out_dir", None) is not None and \
        callable(getattr(recorder, "add_array", None))


class PolicyTrace:
    """One episode's new-side trace (route ``smvla/new`` / ``perceptual-framesamp-modul/new``). When ``enabled`` is
    false (no location) every method is a no-op.

    Every recording method swallows its own exceptions and increments ``hook_errors`` (written to
    ``end.observer_hook_errors`` at finalization, C11), never changing the caller's requests, actions or control
    flow; actions are always copied before recording.
    """

    def __init__(self, route: str, identity: dict, conn_info: dict, recorder: Any, *, max_steps: int,
                 recorder_has_actions: bool, omit_overflow_frame: bool):
        self.route = route
        self.w = None
        self.path: Path | None = None
        self.hook_errors = 0
        self.steps = 0          # steps_attempted
        self.observed = 0       # steps_observed
        self.demo_frames: int | None = None
        self.actions: list[np.ndarray | None] = []
        self.cap_hit = False
        self.max_steps = int(max_steps)
        self.recorder_has_actions = bool(recorder_has_actions)
        self.omit_overflow_frame = bool(omit_overflow_frame)
        self.encodings: set[str] = set()
        self.pending_frames: list[list[str | None]] = []
        self._tw = None
        self.lang = None        # language ledger (language.jsonl next to the trace)
        self.src_call: str | None = None  # which call the current action chunk came from (log_step's source_call_id)
        self.chunk_pos = 0      # index of the next step within the current action chunk (log_step's chunk_index)
        try:
            ci = conn_info or {}
            path = load_sibling("groundsg_client").trace_location(ci, recorder)
            if path is None:
                return
            self._tw = load_sibling("trace_writer")
            tag = ci.get("episode_tag") or f"{identity.get('key')}.a1"
            ident = {k: identity.get(k) for k in ("task", "tier", "seed", "source_episode", "builder_episode", "key")}
            ident.update(dataset=ci.get("dataset"), attempt=episode_attempt(tag))
            seed = ci.get("policy_seed", identity.get("policy_seed"))
            cap = ci.get("effective_cap")
            kw: dict[str, Any] = {}
            if seed is not None:  # identity / header record policy_seed; not written when absent, so old bytes are unchanged
                ident["policy_seed"] = int(seed)
                kw["policy_seed"] = int(seed)
            if cap is not None:
                kw["effective_cap"] = int(cap)
            self.w = self._tw.TraceWriter(path, route=route, identity=ident, max_steps=self.max_steps, **kw)
            self.path = Path(path)
        except Exception as e:  # noqa: BLE001 recorder cannot be created: warn only, the episode is unaffected
            self.w = None
            self._err("init", e)
            return
        try:
            LanguageLog = getattr(self._tw, "LanguageLog", None)
            if LanguageLog is not None:
                self.lang = LanguageLog(self.path.parent / "language.jsonl")
        except Exception as e:  # noqa: BLE001
            self.lang = None
            self._err("language.init", e)

    @property
    def enabled(self) -> bool:
        return self.w is not None

    def _err(self, where: str, e: BaseException) -> None:
        self.hook_errors += 1
        print(f"TRACE_HOOK_ERROR route={self.route} where={where} {type(e).__name__}: {e}"[:600], flush=True)

    @staticmethod
    def _copy(a: Any) -> np.ndarray | None:
        return None if a is None else np.array(a, copy=True)

    # -- demo, requests, responses, history --
    def demo(self, fronts, wrists, states, goal) -> None:
        if not self.enabled:
            return
        try:
            self.demo_frames = len(fronts) - 1
            self.w.log_demo(list(fronts), list(wrists), list(states), [goal])
        except Exception as e:  # noqa: BLE001
            self._err("demo", e)

    def note_frames(self, frames) -> None:
        """SimpleMemVLA: record the observed frames (front, wrist image hashes), merged into the next infer's logical
        input (C10)."""
        if not self.enabled:
            return
        try:
            h = self._tw.image_sha256
            self.pending_frames.extend([[h(fr[CAM_FRONT]), h(fr[CAM_WRIST])] for fr in frames])
        except Exception as e:  # noqa: BLE001
            self._err("note_frames", e)

    def logical_request(self, name: str, instruction: str, state: Any) -> None:
        """SimpleMemVLA logical-input line: instruction, state, frame-hash sequence since the previous inference (C10;
        communication bytes are not compared)."""
        if not self.enabled:
            return
        try:
            obj = {"instruction": instruction, "state": np.array(state, copy=True), "frames": self.pending_frames}
            self.encodings.add("logical")
            self.w.log_request(name, self._tw.canonical_bytes(obj), step=self.steps)
            self.pending_frames = []
        except Exception as e:  # noqa: BLE001
            self._err("request", e)

    def raw_request(self, name: str, obj: Any, raw: bytes | None) -> None:
        """FrameSamp+Modulation: sha256 of the raw msgpack frame bytes (falls back to ``canonical_bytes`` when the raw
        bytes are unavailable, marked ``request_encoding`` at finalization)."""
        if not self.enabled:
            return
        try:
            if raw is not None:
                payload, enc = bytes(raw), "msgpack"
            else:
                payload, enc = self._tw.canonical_bytes(obj), "canonical"
            self.encodings.add(enc)
            self.w.log_request(name, payload, step=self.steps)
        except Exception as e:  # noqa: BLE001
            self._err("request", e)

    def response(self, actions: Any) -> None:
        if not self.enabled:
            return
        try:
            self.w.log_response(self._copy(actions), step=self.steps)
        except Exception as e:  # noqa: BLE001
            self._err("response", e)

    def history(self, start: int, end: int, note: str) -> None:
        if not self.enabled:
            return
        try:
            self.w.log_history(start, end, note=note)
        except Exception as e:  # noqa: BLE001
            self._err("history", e)

    # -- language ledger: every method swallows exceptions and increments hook_errors, never changing requests, actions or control flow --
    def lang_open(self, model: str, *, step: int | None = None, **kw: Any) -> str | None:
        """Open one model call; ``step`` defaults to the number of steps already given to the environment. Returns
        None when the ledger is unavailable (the other lang_* methods are no-ops on None)."""
        if self.lang is None:
            return None
        try:
            return self.lang.open_call(model, self.steps if step is None else int(step), **kw)
        except Exception as e:  # noqa: BLE001
            self._err("language.open", e)
            return None

    def lang_msg(self, call_id: str | None, **kw: Any) -> None:
        if self.lang is None or call_id is None:
            return
        try:
            self.lang.message(call_id, **kw)
        except Exception as e:  # noqa: BLE001
            self._err("language.message", e)

    def lang_audit(self, call_id: str | None, audit: Any) -> tuple[Any, Any]:
        """Write the per-channel tokenization of the server wrapper audit block as ``in`` messages and return
        ``(server_final_text, server_truncated)``; None without an audit block."""
        if self.lang is None or call_id is None:
            return None, None
        try:
            return self._tw.audit_channel_messages(self.lang, call_id, audit)
        except Exception as e:  # noqa: BLE001
            self._err("language.audit", e)
            return None, None

    def lang_close(self, call_id: str | None, **kw: Any) -> None:
        if self.lang is None or call_id is None:
            return
        try:
            self.lang.close_call(call_id, **kw)
        except Exception as e:  # noqa: BLE001
            self._err("language.close", e)

    def image_ref(self, slot: int, ref: str, cam: str, img: Any) -> dict | None:
        """Image reference: the current frame's position in the trace (last demo frame or after step N) and the raw
        frame sha256 (same algorithm as the trace)."""
        if not self.enabled:
            return None
        try:
            if self.steps == 0:
                phase, idx = "demo", (self.demo_frames if self.demo_frames is not None else None)
            else:
                phase, idx = "exec", self.steps
            return {"slot": int(slot), "ref": ref, "phase": phase, "frame_idx": idx, "cam": cam,
                    "raw_sha256": self._tw.image_sha256(img), "sources": None, "transform": None,
                    "encoded_sha256": None}
        except Exception as e:  # noqa: BLE001
            self._err("language.image_ref", e)
            return None

    def begin_chunk(self, call_id: str | None) -> None:
        """Steps executed from now on come from the call ``call_id``; the in-chunk index restarts at 0."""
        self.src_call = call_id
        self.chunk_pos = 0

    def _link(self) -> dict:
        if self.src_call is None:
            return {}
        out = {"source_call_id": self.src_call, "chunk_index": self.chunk_pos}
        self.chunk_pos += 1
        return out

    # -- executed steps --
    def step(self, action, front, wrist, state, *, subgoal, terminated, truncated, status) -> None:
        """Step with a valid observation (C4): post-step images, state, the action actually given to the environment,
        and the current subgoal."""
        if not self.enabled:
            return
        try:
            a = self._copy(action)
            self.steps += 1
            self.observed += 1
            self.actions.append(a)
            self.w.log_step(step=self.steps, front=front, wrist=wrist, state=state, action=a, subgoal=subgoal,
                            terminated=bool(terminated), truncated=bool(truncated), status=status, **self._link())
        except Exception as e:  # noqa: BLE001
            self._err("step", e)

    def missing(self, action, reason: str, subgoal=None) -> None:
        """Step without observation (C8): keeps step number, action and reason."""
        if not self.enabled:
            return
        try:
            a = self._copy(action)
            self.steps += 1
            self.actions.append(a)
            self.w.log_missing_step(step=self.steps, action=a, reason=str(reason)[:300], subgoal=subgoal,
                                    **self._link())
        except Exception as e:  # noqa: BLE001
            self._err("missing_step", e)

    def step_exception(self, action, exc: BaseException, before: int | None, after: int | None, *,
                       logged: int = 0, subgoal=None) -> None:
        """After ``session.step`` raised: add steps without observation according to the increase of the environment
        session's ``steps`` (same basis as the result row ``exec_steps``).

        ``StepCapReached`` (strict-cap) never reaches the environment and is not counted; it only sets ``cap_hit``;
        when the session step count is unavailable, other exceptions count as 1 step."""
        if not self.enabled:
            return
        try:
            name = type(exc).__name__
            if name == "StepCapReached":
                self.cap_hit = True
            if before is not None and after is not None:
                extra = int(after) - int(before) - int(logged)
            else:
                extra = 0 if name == "StepCapReached" else 1
            for _ in range(max(0, extra)):
                self.missing(action, f"step_exception: {name}: {exc}", subgoal)
        except Exception as e:  # noqa: BLE001
            self._err("step_exception", e)

    # -- finalization --
    # arrays.npz is no longer written by this class via np.savez; TraceWriter collects exec_action / exec_state per
    # step and close writes them via trace_writer.merge_write_npz (in any order relative to a recorder in the same
    # directory, neither overwrites the other; separate directories each write their own); end.arrays is a summary.

    def close(self, status: str | None, *, cap_hit: bool = False, **extra: Any) -> None:
        """Finalize per C2, C3, C8: strict-cap records ``timeout``; an error episode without demo frames records
        ``no_frame``."""
        if not self.enabled:
            return
        try:
            self.cap_hit = self.cap_hit or bool(cap_hit)
            st = status if status in TRACE_TERMINALS else "error"
            if self.cap_hit:
                st = "timeout"
            no_frame = self.demo_frames is None
            demo = 0 if no_frame else int(self.demo_frames)
            # the official loop breaks before recording the last step when exceeding max_steps (only for routes such as FrameSamp+Modulation that "judge timeout at step max_steps+1")
            omitted = int(self.omit_overflow_frame and st == "timeout" and not self.cap_hit and
                          self.steps == self.max_steps + 1)
            frames = 0 if no_frame else demo + 1 + self.observed - omitted
            if self.lang is not None:  # close the language ledger first (dangling calls get cancelled); its exceptions also count toward this episode's hook_errors
                try:
                    self.lang.close()
                except Exception as e:  # noqa: BLE001
                    self._err("language.close", e)
            enc = sorted(self.encodings)
            self.w.close(status=st, terminal_reason=st, side="new", demo_frames=demo, no_frame=no_frame,
                         steps_attempted=self.steps, steps_observed=self.observed, frames_recorded=frames,
                         omitted_timeout_frames=omitted, cap_hit=self.cap_hit,
                         request_encoding=enc[0] if len(enc) == 1 else ("mixed" if enc else None),
                         observer_hook_errors=self.hook_errors, **extra)
        except Exception as e:  # noqa: BLE001
            self._err("close", e)


def run_episode(session, identity: dict, conn_info: dict, recorder=None, *, conn=None,
                max_steps: int = MAX_STEPS, execute_horizon: int = EXECUTE_HORIZON,
                reset_retries: int = RESET_RETRIES, record_frames: bool | None = None) -> dict:
    """Run one episode; returns {status, task_success, steps, error, decisions, timing, server_meta, protocol}.

    conn: a fake connection injected by tests (needs metadata, call(msg)->(reply, raw, rep_raw), close()); when None
    a WSPolicyConn is built from conn_info={"host","port"}. The session is built by the caller; this function does
    not close it.
    """
    rec = recorder if recorder is not None else _NullRecorder()
    # when the environment session holds the same recorder, raw frames, exec_action and the reset/run phases are
    # already recorded on the environment side; this function only records the policy side (message fingerprints,
    # state, model_action, decision events) to avoid duplication.
    env_records = recorder is not None and getattr(session, "recorder", None) is recorder
    if record_frames is None:
        record_frames = not env_records
    task = identity["task"]
    hb = hard_bound(max_steps, execute_horizon)
    t_start = time.monotonic()
    timing: dict[str, Any] = {"reset_env_s": None, "infer_ms": [], "rtt_ms": [], "step_env_s": 0.0,
                              "observe_ms": [], "connect_s": None}
    out = {"status": "error", "task_success": False, "steps": 0, "error": None, "decisions": 0,
           "infra": False, "infra_reason": None,
           "hard_bound": hb, "timing": timing, "server_meta": None,
           "protocol": {"messages": 0, "frames_sent": 0, "sha_mismatch": 0}}
    proto = out["protocol"]
    frame_idx = {"front": 0, "wrist": 0}
    own_conn = conn is None
    # per-episode trace (enabled=False without a location, in which case all recording calls below are no-ops and
    # client behavior is unchanged). The trace's max_steps records the real upper bound of the client loop,
    # hard_bound x execute_horizon (84 x 16 = 1344 for 1300 steps): a timeout episode that runs out of decisions
    # executes up to that bound with every step recorded, so there is no "step max_steps+1 not recorded"; --max-steps
    # is recorded separately as end.episode_max_steps.
    trace = PolicyTrace("smvla/new", identity, conn_info or {}, recorder, max_steps=hb * int(execute_horizon),
                        recorder_has_actions=recorder_writes_arrays(recorder), omit_overflow_frame=False)

    def record_frames_(frames, tag):
        if not record_frames or not frames:
            return None
        idx = {}
        for cam in (CAM_FRONT, CAM_WRIST):
            idx[cam] = rec.add_frames(cam, np.stack([fr[cam] for fr in frames]), tag=tag)
        return {k: [v[0], v[-1]] if v else [] for k, v in idx.items()}

    last_audit: list[Any] = [None]  # server wrapper audit block popped from the most recent reply

    def call(kind: str, msg: dict, extra: dict | None = None) -> dict:
        t0 = time.monotonic()
        reply, raw, rep_raw = conn.call(msg)
        rtt = (time.monotonic() - t0) * 1000.0
        proto["messages"] += 1
        ev = {"kind": "msg", "type": kind, "send_sha256": sha256_bytes(raw), "send_bytes": len(raw),
              "recv_sha256": sha256_bytes(rep_raw), "recv_bytes": len(rep_raw), "rtt_ms": round(rtt, 3)}
        if extra:
            ev.update(extra)
        rec.add_event(ev)
        if kind == "infer":
            timing["rtt_ms"].append(round(rtt, 3))
        if not isinstance(reply, dict):
            raise ProtocolError(f"{kind} reply is not a dict: {type(reply)}")
        # pop the server wrapper audit key first (it only goes to the language ledger, never to actions or later checks)
        last_audit[0] = reply.pop(AUDIT_KEY, None)
        if "error" in reply:
            raise ServerError(f"server error ({kind}): {reply['error']}")
        if reply.get("req_sha") != sha256_bytes(raw):
            proto["sha_mismatch"] += 1
            raise ProtocolError(f"{kind} reply req_sha does not match the bytes sent")
        return reply

    def observe(frames, tag):
        if not frames:
            return
        idx = record_frames_(frames, tag)
        payload = [{CAM_FRONT: fr[CAM_FRONT], CAM_WRIST: fr[CAM_WRIST]} for fr in frames]
        reply = call("observe", {"observe": {"frames": payload}}, {"n_frames": len(frames), "tag": tag,
                                                                   "frame_idx": idx})
        sent = [{CAM_FRONT: frame_sha(fr[CAM_FRONT]), CAM_WRIST: frame_sha(fr[CAM_WRIST])} for fr in frames]
        if reply.get("n") != len(frames) or reply.get("frame_sha") != sent:
            proto["sha_mismatch"] += 1
            raise ProtocolError(f"observe reply frame fingerprints do not match what was sent (sent {len(frames)} frames, got {reply.get('n')})")
        proto["frames_sent"] += len(frames)
        timing["observe_ms"].append(round(float(reply.get("observe_time_ms", 0.0)), 3))
        trace.note_frames(frames)

    status = None
    error = None
    steps = 0
    try:
        # ---- reset (old: pool.reset -> SimEnvService.reset) ----
        if not env_records:
            rec.set_phase("reset")
        t0 = time.monotonic()
        r = reset_session(session, task, retries=reset_retries)
        timing["reset_env_s"] = round(time.monotonic() - t0, 3)
        if not env_records:
            rec.set_phase("run")
        if not r.get("ok"):
            status, error = "error", f"reset failed: {r}"[:2000]
            m = _marker(r.get("reason"), ENV_RESET_INFRA_MARKERS)
            if m:
                out.update(infra=True, infra_reason=f"env_reset:{m}")
            rec.add_event({"kind": "reset_failed", "reason": r.get("reason")})
        else:
            rec.add_event({"kind": "reset_ok", "attempts": r["attempts"], "n_frames": len(r["frames"]),
                           "instruction": r["instruction"]})
            if trace.enabled:  # C2: all reset frames (demo + initial frame) and the matching 8-d states
                trace.demo([fr[CAM_FRONT] for fr in r["frames"]], [fr[CAM_WRIST] for fr in r["frames"]],
                           r["states"], r["instruction"])
            if own_conn:
                t0 = time.monotonic()
                conn = WSPolicyConn(conn_info["host"], conn_info["port"])
                timing["connect_s"] = round(time.monotonic() - t0, 3)
            out["server_meta"] = getattr(conn, "metadata", None)
            reply = call("reset", {"reset": {"episode_key": episode_key(identity)}})
            rec.add_event({"kind": "server_rng", "rng": reply.get("rng")})
            instruction = r["instruction"]
            observe(r["frames"], "reset")
            cur_state = np.asarray(r["states"][-1], dtype=np.float32)
            for i, s in enumerate(r["states"]):
                rec.add_array("reset_state", np.asarray(s, dtype=np.float32), step=i - len(r["states"]) + 1)

            success = False
            ended = False
            decisions = 0
            while decisions < hb:
                decisions += 1
                if success:  # old: act = [g for g in active if not success[g]]; if not act: break
                    break
                rec.add_array("state", cur_state, step=steps)
                trace.logical_request("infer", instruction, cur_state)
                # language ledger: one action_model call per decision; the in message is persisted before sending
                cid = trace.lang_open("action_model", params={"decision": decisions - 1})
                trace.lang_msg(cid, dir="in", role="fields", text={"instruction": instruction})
                try:
                    reply = call("infer", {"infer": {"instruction": instruction, "state": cur_state}},
                                 {"decision": decisions - 1, "step": steps, "state_sha": array_sha(cur_state)})
                    if reply.get("recv_state_sha") != array_sha(cur_state) or \
                            reply.get("recv_instruction_sha") != sha256_bytes(instruction.encode("utf-8")):
                        proto["sha_mismatch"] += 1
                        raise ProtocolError("infer reply state/instruction fingerprints do not match what was sent")
                except BaseException:
                    trace.lang_close(cid, status="error")
                    raise
                final_text, final_trunc = trace.lang_audit(cid, last_audit[0])
                trace.lang_msg(cid, dir="out", role="assistant", text=reply.get("subtask"))
                trace.lang_close(cid, status="reply", parsed=reply.get("subtask"), server_final_text=final_text,
                                 server_truncated=final_trunc)
                trace.begin_chunk(cid)
                timing["infer_ms"].append(round(float(reply["infer_ms"]), 3))
                actions_full = np.asarray(reply["actions_full"])
                rec.add_array("model_action", actions_full, step=steps)
                chunk = np.asarray(reply["actions"])[:execute_horizon]
                if not np.array_equal(chunk, actions_full[:execute_horizon]):
                    raise ProtocolError("infer reply actions do not match actions_full[:16]")
                rec.add_event({"kind": "decision", "decision": decisions - 1, "step": steps,
                               "subtask": reply.get("subtask"), "infer_ms": round(float(reply["infer_ms"]), 3),
                               "model_action_sha": array_sha(actions_full)})

                trace.response(actions_full)
                subtask = reply.get("subtask")  # current subgoal (C4, C7): the subtask text of this decision's reply
                subgoal = subtask if subtask is None or isinstance(subtask, str) else str(subtask)

                exec_step = [steps]
                last_act = [None]  # the last row of this chunk given to the environment (for recording a step that raised)
                obs_seen = [0]

                def on_exec(act8, _s=exec_step, _last=last_act):
                    if not env_records:
                        rec.add_array("exec_action", act8, step=_s[0])
                    _s[0] += 1
                    _last[0] = act8

                def on_obs(_i, front, wrist, state, terminated, truncated, *, status=None, error=None,
                           _last=last_act, _seen=obs_seen, _sg=subgoal):
                    _seen[0] += 1
                    if front is None:
                        why = f"env_status_error: {error}" if status == "error" else "obs_none"
                        trace.missing(_last[0], why, _sg)
                    else:
                        trace.step(_last[0], front, wrist, state, subgoal=_sg, terminated=terminated,
                                   truncated=truncated, status=status)

                sess_before = getattr(session, "steps", None) if trace.enabled else None
                t0 = time.monotonic()
                try:
                    res = step_chunk(session, chunk, on_exec=on_exec, on_obs=on_obs if trace.enabled else None)
                except Exception as e:  # old: InProcSimPool.step catches it as {"error": f"step_exc: {e}"}
                    res = {"error": f"step_exc: {e}"}
                    trace.step_exception(last_act[0], e, sess_before, getattr(session, "steps", None),
                                         logged=obs_seen[0], subgoal=subgoal)
                timing["step_env_s"] += time.monotonic() - t0
                if "error" in res:
                    status, error, ended = "error", str(res.get("error")), True
                    m = _marker(error, INFRA_MARKERS)
                    if m:
                        out.update(infra=True, infra_reason=f"env_step:{m}")
                    rec.add_event({"kind": "step_exc", "error": error})
                    break
                observe(res["frames"], "step")
                if res.get("states"):
                    cur_state = np.asarray(res["states"][-1], dtype=np.float32)
                success = bool(res.get("success", False))
                steps += int(res.get("consumed", 0) or 0)
                rec.add_event({"kind": "chunk_done", "decision": decisions - 1, "consumed": res["consumed"],
                               "steps": steps, "status": res["status"], "done": res["done"]})
                if res.get("done", False):
                    status = str(res.get("status") or ("success" if success else "fail"))
                    error = res.get("error_message")
                    ended = True
                    break
            out["decisions"] = decisions
            if not ended:
                status = "success" if success else "timeout"
                error = None if success else timeout_error(hb)
    except Exception as e:
        # old: evaluate_manifest catches run_group exceptions; error takes the last 2000 characters of the traceback, steps kept
        status = "error"
        tb = traceback.format_exc()
        error = tb[-2000:]
        if isinstance(e, ProtocolError):
            proto["broken"] = True
        reason = classify_exception(e, tb)
        if reason:
            out.update(infra=True, infra_reason=reason)
    finally:
        if own_conn and conn is not None:
            conn.close()
    if status not in FINAL_STATUSES + ("error",):
        error = f"unknown terminal status {status}; {error}"
        status = "error"
    timing["step_env_s"] = round(timing["step_env_s"], 3)
    timing["episode_s"] = round(time.monotonic() - t_start, 3)
    out.update(status=status, task_success=status == "success", steps=steps, error=error)
    rec.add_event({"kind": "episode_end", "status": status, "steps": steps, "error": error,
                   "decisions": out["decisions"]})
    if trace.enabled:  # C2, C3, C8: strict-cap follows the session's cap_hit and records timeout (same basis as the outer _classify)
        trace.close(status, cap_hit=bool(getattr(session, "cap_hit", False)), decisions=out["decisions"],
                    episode_max_steps=int(max_steps), step_bound=hb * int(execute_horizon),
                    session_steps=getattr(session, "steps", None))
        out["trace_path"] = str(trace.path)
    return out


# -- new interface: the four model-side methods --------------------------------------

from robomme_ood_eval import servers as _servers  # noqa: E402
from robomme_ood_eval.policy import Ready  # noqa: E402


def smvla_server_spec(policy, ckpt: Any) -> tuple[list, dict, Path]:
    """Command, environment and cwd of the SimpleMemVLA server, copied from the smvla branch of the legacy
    ``run_seat.sh::build_server_cmd``:
    ``<smvla-env interpreter> servers/smvla_server.py serve --port --ckpt --warmup [--det] --policy-seed
    --metadata_out``, with cwd at the evaluation repo root and fixed variables such as ``OMP_NUM_THREADS=1``. The
    interpreter comes from ``cfg["smvla_py"]`` -> env var ``SMVLA_PY`` -> ``envs/smvla-env/.venv/bin/python``."""
    S = _servers
    cfg = policy.cfg
    py = S.interpreter(cfg, "smvla_py", "SMVLA_PY", policy.root / "envs" / "smvla-env" / ".venv" / "bin" / "python")
    det = S.flag_on(cfg.get("det", False))
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"} if det else {}
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", MPLBACKEND="Agg")
    argv = [str(py), str(S.SERVERS_DIR / "smvla_server.py"), "serve", "--port", str(int(policy.port)), "--ckpt",
            str(ckpt), "--warmup", *(["--det"] if det else []), "--policy-seed", str(int(policy.policy_seed)),
            "--metadata_out", str(policy.wrap_meta)]
    return argv, env, policy.root


class SmvlaPolicy(_servers.ServedPolicy):
    """SimpleMemVLA (``smvla``).

    * ``load``: start ``servers/smvla_server.py`` with the ``envs/smvla-env`` interpreter (the server does its own
      ``--warmup``; ready = port listening), check ``policy_seed`` in the server wrapper metadata, and fingerprint the
      ckpt in the background;
    * ``reset(spec)``: sends no message and never touches the environment -- only probes liveness and records this
      episode's identity;
    * ``play``: calls this module's ``run_episode`` unchanged (``max_steps=spec.max_steps``, ``reset_retries=0``, as
      in the old seat runner): first ``session.reset()``, then connect to the server and send ``reset`` with
      ``episode_key`` (triggering ``reseed``), keeping the old order; the session is handed over via
      ``SessionNoClose`` so the old code's ``env.close()`` after a failed reset does not close the outer
      environment;
    * ``close``: stop the server process group (base class).

    cfg: ``ckpt`` (required; there is no default), optional ``smvla_py``, ``det``. The client process should be
    started with ``OMP_NUM_THREADS=1`` (as the old seat runner did; setting it after numpy is loaded has no effect,
    so this only checks and warns)."""

    model = "smvla"
    requires_ckpt = True

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        self.ckpt: Path | None = None
        self.current_spec = None

    def load(self) -> None:
        S = _servers
        self.ckpt = S.require_ckpt(self.cfg, self.model)
        self._pick_port()
        argv, env, cwd = smvla_server_spec(self, self.ckpt)
        if self.preflight:
            py = Path(argv[0])
            if not (py.is_file() and os.access(py, os.X_OK)):
                raise S.PreflightError(f"RUN_BLOCKED reason=smvla_venv_missing {py}")
            if not self.ckpt.is_dir():
                raise S.PreflightError(f"RUN_BLOCKED reason=smvla_ckpt_missing ckpt={self.ckpt}")
        omp = os.environ.get("OMP_NUM_THREADS")
        if omp != "1":
            print(f"SMVLA_CLIENT_OMP=WARN OMP_NUM_THREADS={omp!r} (the old seat client used 1)", flush=True)
        self._launch(argv, env, cwd, [Ready.port()], self.ckpt)
        self._fingerprint(self.ckpt)

    def reset(self, spec) -> None:
        super().reset(spec)
        self.current_spec = spec

    def play(self, session, spec, recorder) -> dict:
        res = run_episode(_servers.SessionNoClose(session), spec.identity(), self._conn_info(spec), recorder,
                          max_steps=int(spec.max_steps), reset_retries=0)
        return self._finish_play(res)
