#!/usr/bin/env python3
"""GroundSG (MME-VLA symbolic-grounded-subgoal) new-side client.

The seat runner (``--policy groundsg --groundsg-variant {ground-sg-oracle,ground-sg-qwenvl,ground-sg-memer}``) loads
this module via ``load_sibling("groundsg_client")`` and calls ``run_episode(session, identity, conn_info, recorder)``.

The loop itself is **not rewritten**: ``official_defs.extract_defs`` extracts the verbatim ``EpisodeEvaluator`` and
``Args`` from the official ``eval.py``, and the official ``EpisodeEvaluator.eval_each_episode`` is called directly;
this module only provides a runner adapter object (``SessionRunner``) that delegates the official ``EnvRunner``
interface (``env_id``, ``episode_id``, ``task_goal``, ``difficulty``, ``info``, ``get_init_obs()``, ``step()``,
``simple_subgoal_oracle``, ``grounded_subgoal_oracle``) to this repo's ``EnvSession`` ``reset()`` / ``step()``,
syncing ``info`` every step. The bodies of ``get_init_obs`` / ``step`` match the official ``EnvRunner`` line by line,
with only ``self.env`` replaced by ``EnvSession``; the following three exceptions are not swallowed as environment
errors but re-raised unchanged: ``StepCapReached`` (``--strict-cap``; caught by this module and returned, and
``run_one`` records timeout from ``cap_hit``), ``RecorderError`` (infrastructure), ``ResetBudgetExhausted``.

The subgoal predictor only extracts the classes the chosen variant needs: Oracle takes ``OracleSubgoalPredictor``
(does not read qwenvl/api.py and does not import swift); QwenVL takes ``QwenVLSubgoalPredictor`` and
``Qwen3VLModel`` (not Gemini / MemER); MemER takes ``MemERSubgoalPredictor`` and ``Qwen3VLModelMemER`` with the
compatibility layer applied (see the ``official_defs`` module docstring). Before construction it asserts exactly one
of ``use_oracle`` / ``use_qwenvl`` / ``use_memer`` is true. The predictor and evaluator live in ``policy_context``
and are built once per seat (``make_policy_context``), same as the official "one predictor per evaluation, reused
across episodes".

Seed, scratch areas, language ledger and result fields:

* ``seat_info["policy_seed"]`` is required (non-negative integer) and written explicitly into ``Args.model_seed``
  via ``official_defs.make_args(model_seed=)``; ``seed_everything`` runs before constructing the QwenVL / MemER
  predictor; the MemER adapter comes from ``seat_info["memer_adapter_path"]``;
* subgoal model scratch areas: QwenVL ``<trace_dir>/qwen-tmp/...``, MemER ``<trace_dir>/memer-tmp/...``; at the end
  of the episode ``ep*_QwenVL_log.jsonl`` / ``ep*_MemER_log.jsonl`` are archived into the episode directory and the
  scratch area is deleted entirely (also on exceptions);
* language ledger ``language.jsonl`` (``trace_writer.LanguageLog``, optional probe: nothing is recorded and no error
  raised when the class is missing): QwenVL / MemER record one ``subgoal_model`` call per question (system, raw user
  text, image references; each MemER re-ask is a separate call with ``retry``), with the raw reply and ``parsed``;
  QwenVL keep_period reuse steps record ``reuse``; every action inference is one ``action_model`` call (structured
  fields; Oracle carries ``subgoal_source: oracle``); ``_sgeval_audit`` in action server replies is popped before
  reaching the official code and recorded in that call (``None`` when missing); executed steps point to the source
  call via ``source_call_id`` / ``chunk_index``;
* result rows record ``policy_seed``, ``policy_variant``, ``memer_compat_sha256`` (MemER), ``error_kind`` (MemER
  three bad replies with no previous valid subgoal -> ``model_response_error``, ``status=error``, not
  infrastructure, not rerun).

Per-episode peripheral recording (same code on both sides; the original side ``official_hard_runner.py`` also uses
``EpisodeTap`` / ``TracingClient`` from here):

* ``trace.jsonl`` (``trace_writer.TraceWriter``): location ``conn_info["trace_path"]`` first, otherwise
  ``<trace_dir>/trace.jsonl``, otherwise ``<recorder.out_dir>/trace.jsonl``; nothing is written if none applies;
* every request sent to the model (``reset`` / ``add_buffer`` / ``infer``) records its sha256 after normalization by
  ``official_defs.canonical_bytes``; ``infer`` replies record the full action chunk;
* Qwen / MemER scratch directory ``<trace_dir>/{qwen,memer}-tmp/<dataset>/<episode_tag>/`` (in a temporary directory
  when there is no ``trace_dir``); at the end of the episode ``ep<id>_{QwenVL,MemER}_log.jsonl`` is archived next to
  the trace and the scratch directory deleted entirely (also on ``unknown`` / early exit on exceptions);
* the overlay mp4 written by the official loop itself: both sides default to ``keep_official=True``, and after
  verification it is moved into ``<episode dir>/official/``. New-side raw-only delivery disables this archive,
  while the native temporary encoding and cleanup remain unchanged.

Terminal status: official ``success`` / ``fail`` / ``timeout`` are kept unchanged; ``unknown`` (and other non-terminal
values) record ``status="error"``, ``error="success_flag=<value>"`` without stopping the seat; exceptions raised by
the official loop record ``status="error"`` + ``<exception class>: <message>``, with ``infra`` decided by
``framesamp_modul_client.INFRA_MARKERS``.

Official overlay video retention (additions only):

* ``run_official_episode(..., keep_official=UNSET)``: without it the behavior is byte-identical to before (the
  original side ``official_hard_runner`` does not pass it). With a true value: before the call the evaluator
  instance's ``init_episode`` is wrapped to capture the official ``(task_goal, recorder)`` (restored in
  ``finally``); normal episodes are saved by the official ``save_video``; on ``StepCapReached`` / other exceptions /
  ``unknown`` with no mp4 in ``video_dir``, **the official recorder's own** ``save_video`` is called to salvage it
  using the official file name format (truncated by ``official_safe_filename``); failure before ``init_episode``
  returns: ``none`` when no frame was recorded, ``partial`` when reset frames were obtained but the recorder was not
  handed over (only the reason is recorded; it never poses as a complete video). Before deleting ``video_dir``,
  ``finally`` runs ``keep_official_videos`` (full decode, frame count, exactly one file), moves it into
  ``<archive_dir>/official/`` and writes ``provenance.json``; verification or salvage failures only record
  ``official_save_error``, and raw frames are kept. The return value additionally has ``official_videos``,
  ``official_source`` (``official`` / ``official-salvaged`` / ``partial`` / ``none``), ``official_save_error`` and the
  three C8 counts.
* The new-side ``run_episode`` passes ``groundsg_keep_official`` (default True); False marks
  ``official_source=disabled`` and ``official_videos=[]``. The trace is finalized per the shared contract: C6
  identity adds ``attempt``, steps without observation use ``log_missing_step`` (C8), ``end`` writes
  ``steps_attempted`` / ``steps_observed`` / ``frames_recorded`` / ``omitted_timeout_frames``, ``terminal_reason``
  takes the terminal status (C3; the official raw value is recorded separately as ``success_flag``), and frameless
  episodes get ``no_frame=true``.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robomme_ood_eval import timing as chunk_timing

NORMAL = ("success", "fail", "timeout")
#: exceptions not swallowed as environment errors (matched by class name to avoid importing env_client)
PASS_THROUGH = ("StepCapReached", "RecorderError", "ResetBudgetExhausted")
INFRA_MARKERS = ("RecorderError", "svulkan2", "EXCLUSIVE", "Vulkan", "vk::", "out of memory", "RESOURCE_EXHAUSTED",
                 "CUDA_ERROR", "ConnectionClosed", "ConnectionRefused", "InvalidStatus", "Connection reset")


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


official_defs = load_sibling("official_defs")
trace_writer = load_sibling("trace_writer")
_UNSET = trace_writer.UNSET  # "not provided" sentinel for new optional parameters

#: subdirectory for official overlay videos and identity manifests (official-layout files always go into official/)
OFFICIAL_VIDEO_SUBDIR = "official"
OFFICIAL_SOURCES = ("official", "official-salvaged", "partial", "none")
PROVENANCE_SCHEMA = "official-video-provenance/1"
FRAMES_BASIS = "demo_frames + 1 + steps_observed - omitted_timeout_frames"


def classify_infra(*texts: str | None) -> str | None:
    for text in texts:
        for marker in INFRA_MARKERS:
            if marker in (text or ""):
                return marker
    return None


# -- peripheral recording: trace, raw frames, requests --------------------------------


class RawFrameWriter:
    """One raw-frame directory per original-side episode (same format as ``pp_official_runner.RawFrameWriter``,
    transcoded by the same launcher pipeline):

    ``front.rgb24`` / ``wrist.rgb24`` concatenate raw H x W x 3 uint8 bytes per frame (ffmpeg rawvideo rgb24);
    ``frames.json`` is ``{"pix_fmt":"rgb24","streams":{"front":{"width","height","count"},"wrist":{...}},
    "demo_frames","init_frames","exec_steps","missing_steps","order"}``. Frame order: all frames returned by reset
    (demo + initial), then the observation of each executed step; when a step has no observation (environment step
    raised) nothing is appended and the step number goes into ``missing_steps``.
    ``count = demo_frames + init_frames + exec_steps - len(missing_steps)``."""

    STREAMS = ("front", "wrist")

    def __init__(self, frames_dir: str | Path):
        self.dir = Path(frames_dir)
        self.dir.mkdir(parents=True, exist_ok=False)
        self._fh = {s: (self.dir / f"{s}.rgb24").open("wb") for s in self.STREAMS}
        self.shape: dict[str, tuple | None] = {s: None for s in self.STREAMS}
        self.count = {s: 0 for s in self.STREAMS}
        self.meta: dict[str, Any] = {"demo_frames": None, "init_frames": 0, "exec_steps": 0, "missing_steps": []}
        self._closed = False

    def write(self, stream: str, frame: Any) -> None:
        a = np.ascontiguousarray(np.asarray(frame))
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"{stream} frames must be H x W x 3 uint8, got {a.dtype} {a.shape}")
        if self.shape[stream] is None:
            self.shape[stream] = a.shape
        elif self.shape[stream] != a.shape:
            raise ValueError(f"{stream} frame size changed: {self.shape[stream]} -> {a.shape}")
        self._fh[stream].write(a.tobytes())
        self.count[stream] += 1

    def close(self) -> dict:
        if self._closed:
            return self.summary()
        for fh in self._fh.values():
            fh.close()
        self._closed = True
        summ = self.summary()
        (self.dir / "frames.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                              encoding="utf-8")
        return summ

    def summary(self) -> dict:
        streams = {}
        for s in self.STREAMS:
            shp = self.shape[s]
            streams[s] = {"width": None if shp is None else int(shp[1]), "height": None if shp is None else int(shp[0]),
                          "count": self.count[s]}
        return {"pix_fmt": "rgb24", "streams": streams, "order": "reset_all_then_per_step_last", **self.meta}


class EpisodeTap:
    """Peripheral recorder of one episode (shared by the new and original sides). All callbacks run outside the
    environment / model calls and only read data, never modify it.

    * ``on_reset(pre_traj)``: after ``get_init_obs`` returns, record the demo (all reset frames, states, goal text);
    * ``on_request`` / ``on_response``: client requests and replies;
    * ``on_step(action, obs3, stop, flag, terminated, truncated)``: every executed step (including steps that raised,
      where ``obs3`` is a None triple).

    ``missing_step_contract`` (default ``UNSET`` = old behavior, byte-identical): when true, steps without observation
    are recorded per C8 with ``TraceWriter.log_missing_step`` (``observed=false``, ``missing_reason``); only the new
    side turns it on. ``missing_steps`` only keeps the step numbers without observation in memory (for C8 counts)
    and does not affect any output.
    """

    def __init__(self, trace: Any | None, frames: RawFrameWriter | None = None, *,
                 missing_step_contract: Any = _UNSET):
        self.trace = trace
        self.frames = frames
        self.missing_step_contract = missing_step_contract is not _UNSET and bool(missing_step_contract)
        self.missing_steps: list[int] = []
        self.steps = 0
        self.decisions = 0
        self.last_subgoal: str | None = None
        self.history_from = 0
        self.demo_frames: int | None = None
        self.task_goal: str | None = None
        self.requests: list[str] = []  # sha256 of normalized request bytes (for tests and summaries)
        self.lang: LangTap | None = None  # language ledger wiring (attached by run_official_episode when needed)
        self.chunk_call: str | None = None  # source call of the current action chunk (action_model call_id)
        self.chunk_pos = 0

    def on_reset(self, pre: dict) -> None:
        imgs, wrists, states = list(pre["images"]), list(pre["wrist_images"]), list(pre["states"])
        self.demo_frames = len(imgs) - 1
        self.task_goal = pre.get("task_goal")
        if self.trace is not None:
            self.trace.log_demo(imgs, wrists, states, [self.task_goal])
        if self.frames is not None:
            for f in imgs:
                self.frames.write("front", f)
            for w in wrists:
                self.frames.write("wrist", w)
            self.frames.meta.update(demo_frames=len(imgs) - 1, init_frames=1 if imgs else 0)

    def on_request(self, name: str, obj: Any) -> None:
        payload = official_defs.canonical_bytes(obj)
        self.requests.append(trace_writer.bytes_record(payload)["sha256"])
        if self.trace is not None:
            self.trace.log_request(name, payload, step=self.steps)
        if name == "add_buffer" and self.trace is not None:
            n = len(obj["images"]) if isinstance(obj, dict) and "images" in obj else 0
            self.trace.log_history(self.history_from, self.steps,
                                   note=f"add_buffer frames={n} exec_start_idx={obj.get('exec_start_idx')}")
            self.history_from = self.steps
        if name == "infer":
            self.decisions += 1
            self.last_subgoal = obj.get("grounded_subgoal") if isinstance(obj, dict) else None

    def on_response(self, actions: Any, call_id: str | None = None) -> None:
        if self.trace is not None:
            self.trace.log_response(actions, step=self.steps)
        if self.lang is not None:
            self.chunk_call, self.chunk_pos = call_id, 0

    def _source_kw(self) -> dict:
        """With the language ledger open, executed steps link to the action's source call; without it no key is
        added (the step line keeps the old 9 keys)."""
        if self.lang is None or self.chunk_call is None:
            return {}
        kw = {"source_call_id": self.chunk_call, "chunk_index": self.chunk_pos}
        self.chunk_pos += 1
        return kw

    def on_step(self, action: Any, obs3: tuple, stop: bool, flag: str, terminated: Any, truncated: Any,
                reason: str | None = None) -> None:
        self.steps += 1
        img, wrist, state = obs3
        if img is None:
            self.missing_steps.append(self.steps)
        if self.frames is not None:
            self.frames.meta["exec_steps"] = self.steps
            if img is not None:
                self.frames.write("front", img)
                if wrist is not None:
                    self.frames.write("wrist", wrist)
            else:
                self.frames.meta["missing_steps"].append(self.steps)
        src = self._source_kw()
        if self.trace is not None and img is None and self.missing_step_contract:
            self.trace.log_missing_step(step=self.steps, action=action, subgoal=self.last_subgoal,
                                        reason=reason or f"no_observation status={flag}", **src)
        elif self.trace is not None:
            self.trace.log_step(step=self.steps, front=img, wrist=wrist, state=state, action=action,
                                subgoal=self.last_subgoal, terminated=bool(terminated), truncated=bool(truncated),
                                status=flag, **src)


class TracingClient:
    """Wraps ``reset`` / ``add_buffer`` / ``infer`` of ``MMEVLAWebsocketClientPolicy`` (or a stand-in): record the
    request first, then forward unchanged.

    The server wrapper audit key ``_sgeval_audit`` in ``infer`` replies is always popped before reaching the official
    code; with the language ledger open it is recorded in this ``action_model`` call (``server_final_text=None`` when
    the key is missing)."""

    def __init__(self, inner: Any, tap: EpisodeTap, timer=None, timing=None):
        self._inner = inner
        self._tap = tap
        self._timer, self._timing = timer, timing

    def reset(self):
        self._tap.on_request("reset", {"reset": True})
        return self._inner.reset()

    def add_buffer(self, buffer):
        self._tap.on_request("add_buffer", buffer)
        return self._inner.add_buffer(buffer)

    def infer(self, obs):
        self._tap.on_request("infer", obs)
        lang = self._tap.lang
        cid = lang.action_open(obs) if lang is not None else None
        # the real inner client (framesamp_modul_client.RecordingClient) already pops the audit key in _roundtrip and
        # stores it in _last_audit: clear it before the call so the previous one is not reused when this reply has
        # no audit block
        if hasattr(self._inner, "_last_audit"):
            self._inner._last_audit = None
        if self._timer is not None:
            for name, value in (("_last_rtt", {}), ("_last_server_timing", None), ("_last_recv_t", None)):
                if hasattr(self._inner, name):
                    setattr(self._inner, name, value)
        try:
            out = self._inner.infer(obs)
            t_ret = time.perf_counter() if self._timer is not None else None
        except BaseException:
            if lang is not None:
                lang.close(cid, status="error")
            raise
        audit = out.pop(AUDIT_KEY, None) if isinstance(out, dict) else None
        if audit is None:
            audit = getattr(self._inner, "_last_audit", None)
        if self._timer is not None and isinstance(out, dict) and "actions" in out:
            rtts = getattr(self._inner, "_last_rtt", {})
            rtt = rtts.get("infer") if isinstance(rtts, dict) else None
            server = getattr(self._inner, "_last_server_timing", None)
            missing = rtt is None or not hasattr(self._inner, "_last_server_timing")
            if missing and self._timing is not None:
                self._timing["timing_missing"] = self._timing.get("timing_missing", 0) + 1
            recv_t = getattr(self._inner, "_last_recv_t", None)
            self._timer.close(t=recv_t if recv_t is not None else t_ret,
                              decision=self._tap.decisions - 1, env_step=self._tap.steps,
                              action_rtt_ms=rtt,
                              server_infer_ms=server.get("infer_ms") if isinstance(server, dict) else None)
        if lang is not None:
            lang.action_close(cid, audit)
        self._tap.on_response(out["actions"], call_id=cid)
        return out

    def close(self) -> None:
        ws = getattr(self._inner, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


#: server wrapper reply audit key (written by the wrapper; the client pops it before handing to the official code)
AUDIT_KEY = "_sgeval_audit"


def open_language_log(trace_path: str | Path | None) -> Any:
    """Language ledger ``<trace directory>/language.jsonl``: opened when ``trace_writer.LanguageLog`` exists and there
    is a trace location, otherwise None (optional probe: nothing is recorded and no error raised)."""
    cls = getattr(trace_writer, "LanguageLog", None)
    if cls is None or trace_path is None:
        return None
    return cls(Path(trace_path).parent / "language.jsonl")


_STEP_IMG_RE = re.compile(r"step_(\d+)_image\.png$")


def _png_sha256(path: str) -> str | None:
    """Hash of a subgoal model image (the png saved by the official code) read back, computed like
    ``trace_writer.image_sha256``; png is lossless, so it equals the trace frame hash."""
    try:
        import imageio.v2 as iio

        return trace_writer.image_sha256(np.asarray(iio.imread(path)))
    except Exception:  # noqa: BLE001 unreadable: record None only; evaluation is unaffected
        return None


def _request_fields(request: Any) -> dict:
    """messages / images / videos / objects of a swift ``InferRequest`` (or test stand-in) (read-only, taken before
    sending)."""
    if hasattr(request, "kw"):
        def get(k):
            return request.kw.get(k)
    else:
        def get(k):
            return getattr(request, k, None)
    return {"messages": list(get("messages") or []), "images": list(get("images") or []),
            "videos": list(get("videos") or []), "objects": get("objects")}


def _config_fields(request_config: Any) -> dict:
    kw = getattr(request_config, "kw", None)
    if isinstance(kw, dict):
        return kw
    return {k: getattr(request_config, k, None) for k in ("temperature", "max_tokens")}


class LangTap:
    """Language ledger wiring of one episode (shared by the new and original sides).

    * ``subgoal_open`` / ``subgoal_reply``: one call per real ``engine.infer`` of the subgoal model (QwenVL / MemER);
      ``in`` is written before sending (system, raw user text + image references + demo video segment), ``out`` is
      the raw reply; earlier calls within the same ``get_subgoal`` (bad replies before a MemER re-ask) are closed
      with ``parsed=None`` at the next question, and the last one is closed with ``parsed`` and ``fallback`` after
      ``get_subgoal`` returns; a ``get_subgoal`` without a real question (QwenVL keep_period reuse) records
      ``reuse``;
    * ``action_open`` / ``action_close``: one ``action_model`` call per ``infer`` of the action server; ``in`` holds
      the structured fields (``prompt``, ``grounded_subgoal``, ``simple_subgoal``, ``subgoal_source``,
      ``subgoal_call_id``) and references to the current front / wrist frames (written before sending); the raw
      tokenization channels of the reply audit key are added after the reply arrives (server view), and
      ``server_final_text`` / ``server_truncated`` go into ``call_close``.

    Image ``frame_idx`` convention: number of executed steps (0 = the initial frame after reset, i.e. the last demo
    frame; n = the observation after step n), with ``raw_sha256`` on the same basis as ``front_sha256`` of the trace
    ``step`` line (or the last ``demo`` frame).
    """

    def __init__(self, log: Any, tap: EpisodeTap, *, variant: str, api: Any = None, params: dict | None = None,
                 timer=None):
        self.log, self.tap, self.variant, self.api = log, tap, variant, api
        self.params = dict(params or {})
        self.pending: list[str] = []  # subgoal calls opened but not yet closed in this get_subgoal
        self.last_subgoal_call: str | None = None
        self.subgoal_calls = 0
        self.action_calls = 0
        self.reuses = 0
        self.timer = timer
        self._open_t: dict[str, tuple[float, int]] = {}

    @property
    def memer(self) -> bool:
        return self.variant == official_defs.VARIANT_MEMER

    # -- common --------------------------------------------------------------
    def close(self, call_id: str | None, **kw) -> None:
        if call_id is None:
            return
        self.log.close_call(call_id, **kw)
        self._open_t.pop(call_id, None)
        if call_id in self.pending:
            self.pending.remove(call_id)

    # -- subgoal model ---------------------------------------------------------
    def _subgoal_images(self, paths: list[str]) -> list[dict]:
        n_recent = len(getattr(self.api, "current_execution_frame_paths", None) or []) if self.memer else len(paths)
        n_key = max(0, len(paths) - n_recent)
        out = []
        for i, path in enumerate(paths):
            m = _STEP_IMG_RE.search(str(path))
            ref = "keyframe" if i < n_key else ("recent" if self.memer else "current")
            out.append({"slot": i, "ref": ref, "phase": "exec", "frame_idx": int(m.group(1)) if m else None,
                        "cam": "front", "raw_sha256": _png_sha256(path), "sources": [Path(str(path)).name],
                        "transform": {"encode": "png"}, "encoded_sha256": None})
        return out

    def subgoal_open(self, request: Any, request_config: Any) -> str:
        for cid in list(self.pending):  # the previous (bad) reply is closed before re-asking: the model replied, but invalidly
            self.close(cid, status="reply", parsed=None, fallback=None)
        f = _request_fields(request)
        cfg = _config_fields(request_config)
        params = dict(self.params, temperature=cfg.get("temperature"), max_tokens=cfg.get("max_tokens"))
        if f["objects"] is not None:
            params["objects"] = f["objects"]
        retry = int(getattr(self.api, "_memer_retry", 0) or 0) if self.memer else 0
        cid = self.log.open_call("subgoal_model", int(self.tap.steps), params=params, retry=retry)
        demo = f"demo[0:{self.tap.demo_frames}]" if f["videos"] and self.tap.demo_frames is not None else None
        for msg in f["messages"]:
            if msg.get("role") == "system":
                self.log.message(cid, dir="in", role="system", text=msg.get("content"))
            else:
                self.log.message(cid, dir="in", role=msg.get("role"), text=msg.get("content"),
                                 images=self._subgoal_images(f["images"]), demo_video=demo)
        self.pending.append(cid)
        if self.timer is not None:
            self._open_t[cid] = (time.perf_counter(), retry)
        self.subgoal_calls += 1
        return cid

    def subgoal_reply(self, call_id: str, out: Any) -> None:
        try:
            text = out[0].choices[0].message.content
        except Exception:  # noqa: BLE001
            text = None
        self.log.message(call_id, dir="out", role="assistant", text=text)
        opened = self._open_t.pop(call_id, None)
        if self.timer is not None and opened is not None:
            t0, retry = opened
            self.timer.add_lang((time.perf_counter() - t0) * 1000, kind="ask", env_step=self.tap.steps,
                                retry=retry, call_id=call_id)

    def subgoal_begin(self) -> None:
        for cid in list(self.pending):
            self.close(cid, status="cancelled")

    def subgoal_end(self, value: Any, exc: BaseException | None = None) -> None:
        """Finalize after ``get_subgoal`` returns (or raises): close the last call with parsed / fallback; record
        reuse when there was no real question."""
        if not self.pending:
            if exc is None and self.last_subgoal_call is not None:
                self.log.reuse(int(self.tap.steps), self.last_subgoal_call)
                self.reuses += 1
            return
        last = self.pending[-1]
        for cid in self.pending[:-1]:
            self.close(cid, status="reply", parsed=None, fallback=None)
        if exc is not None:
            if type(exc).__name__ == "MemERResponseError":
                self.close(last, status="reply", parsed=None, fallback="model_response_error")
            else:
                self.close(last, status="error", parsed=None, fallback=None)
            return
        fb = getattr(self.api, "_memer_fallback", None) if self.memer else None
        parsed: dict[str, Any] = {"subgoal": value}
        if self.memer:
            parsed["keyframe_positions"] = getattr(self.api, "_memer_last_positions", None)
            parsed["key_frame_ids"] = sorted(getattr(self.api, "key_frame_paths", None) or {})
        if fb == "last_valid":  # the third one is bad too: reuse the previous valid subgoal (parsed records the reused value, marked fallback)
            parsed = {"subgoal": value, "reused_from_last_valid": True}
        self.close(last, status="reply", parsed=parsed, fallback=fb)
        self.last_subgoal_call = last

    # -- action model ----------------------------------------------------------
    def action_open(self, obs: Any) -> str:
        obs = obs if isinstance(obs, dict) else {}
        step = int(self.tap.steps)
        oracle = self.variant == official_defs.VARIANT_ORACLE
        fields = {"prompt": obs.get("prompt"), "grounded_subgoal": obs.get("grounded_subgoal"),
                  "simple_subgoal": obs.get("simple_subgoal"),
                  "subgoal_source": "oracle" if oracle else "subgoal_model",
                  "subgoal_call_id": None if oracle else self.last_subgoal_call}
        imgs = []
        for key, ref, cam in (("observation/image", "current", "front"), ("observation/wrist_image", "wrist", "wrist")):
            if key in obs:
                imgs.append({"slot": len(imgs), "ref": ref, "phase": "exec", "frame_idx": step, "cam": cam,
                             "raw_sha256": trace_writer.image_sha256(obs[key]), "sources": [], "transform": None,
                             "encoded_sha256": None})
        cid = self.log.open_call("action_model", step, params=None)
        self.log.message(cid, dir="in", role="fields", text=fields, images=imgs)
        self.action_calls += 1
        return cid

    def action_close(self, call_id: str, audit: Any) -> None:
        audit = audit if isinstance(audit, dict) else {}
        chans = [c for c in (audit.get("channels") or []) if isinstance(c, dict)]
        for ch in chans:
            self.log.message(call_id, dir="in", role="fields", text=ch.get("text"), channel=ch.get("channel"),
                             token_ids=ch.get("token_ids"), mask=ch.get("mask"), tokenizer=ch.get("tokenizer"),
                             truncated=ch.get("truncated"))
        truncated = any(bool(c.get("truncated")) for c in chans) if chans else None
        self.close(call_id, status="reply", server_final_text=audit.get("server_final_text"),
                   server_truncated=truncated)

    def summary(self) -> dict:
        return {"subgoal_calls": self.subgoal_calls, "action_calls": self.action_calls, "reuses": self.reuses}


class _EngineTap:
    """Delegating wrapper of the subgoal model ``engine``: records the language ledger before and after ``infer``;
    other attributes are forwarded unchanged."""

    def __init__(self, inner: Any, lang: LangTap):
        self._inner, self._lang = inner, lang

    def infer(self, reqs, request_config=None):
        cid = self._lang.subgoal_open(reqs[0], request_config)
        try:
            out = self._inner.infer(reqs, request_config=request_config)
        except BaseException:
            self._lang.close(cid, status="error")
            raise
        self._lang.subgoal_reply(cid, out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _timing_cuda():
    """Use an already imported CUDA backend; CPU timing never imports or synchronizes CUDA."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    return cuda if cuda is not None and cuda.is_available() else None


class _CountingEngine:
    """Count actual asks independently of the optional language ledger."""

    def __init__(self, inner):
        self._inner, self.n = inner, 0

    def infer(self, *args, **kwargs):
        self.n += 1
        return self._inner.infer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TimedPredictor:
    """Episode-local evaluator wrapper; the context retains the original predictor."""

    def __init__(self, inner, timer, *, variant, counter=None, tap=None):
        self._inner, self._timer, self._variant = inner, timer, variant
        self._counter, self._tap = counter, tap

    def get_subgoal(self, count, current_subgoal, last_subgoal):
        self._timer.start()
        if self._variant == official_defs.VARIANT_ORACLE:
            return self._inner.get_subgoal(count, current_subgoal, last_subgoal)
        n0 = self._counter.n if self._counter is not None else 0
        cuda = _timing_cuda()
        if cuda is not None:
            cuda.synchronize()
        t0, exc = time.perf_counter(), None
        try:
            return self._inner.get_subgoal(count, current_subgoal, last_subgoal)
        except BaseException as e:
            exc = e
            raise
        finally:
            if cuda is not None:
                cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            extra = {}
            if self._counter is not None:
                asks = self._counter.n - n0
                extra.update(tries=asks, reuse=asks == 0)
            if self._variant == official_defs.VARIANT_MEMER:
                extra["fallback"] = getattr(getattr(self._inner, "api", None), "_memer_fallback", None)
            self._timer.add_lang(ms, kind="subgoal", env_step=self._tap.steps if self._tap is not None else count,
                                 error=type(exc).__name__ if exc else None, **extra)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def install_language(predictor: Any, lang: LangTap) -> Callable[[], None]:
    """Attach the language ledger to the predictor (wrapping ``get_subgoal`` as an instance attribute and replacing
    ``api.engine`` with the delegating wrapper); returns a restore function. Oracle has no subgoal model and is not
    wrapped."""
    restore: list[Callable[[], None]] = []
    if type(predictor).__name__ not in SUBGOAL_TMP:
        return lambda: None
    api = getattr(predictor, "api", None)
    if api is not None and hasattr(api, "engine"):
        inner = api.engine
        api.engine = _EngineTap(inner, lang)
        restore.append(lambda: setattr(api, "engine", inner))
    had = "get_subgoal" in vars(predictor)
    saved = vars(predictor).get("get_subgoal")
    orig = predictor.get_subgoal

    def get_subgoal(count, current_subgoal, last_subgoal):
        lang.subgoal_begin()
        try:
            out = orig(count, current_subgoal, last_subgoal)
        except BaseException as e:
            lang.subgoal_end(None, exc=e)
            raise
        lang.subgoal_end(out[0] if isinstance(out, tuple) else out)
        return out

    predictor.get_subgoal = get_subgoal

    def _undo():
        if had:
            predictor.get_subgoal = saved
        else:
            vars(predictor).pop("get_subgoal", None)

    restore.append(_undo)

    def undo_all():
        for fn in reversed(restore):
            fn()

    return undo_all


def episode_scratch(trace_dir: str | None) -> tuple[Path, bool]:
    """This episode's scratch area: ``trace_dir`` if given (returns False: not deleted entirely), otherwise a new
    temporary directory (returns True: deleted entirely at the end of the episode)."""
    if trace_dir:
        p = Path(trace_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p, False
    return Path(tempfile.mkdtemp(prefix="groundsg-")), True


#: predictors with a subgoal model: scratch directory name and the infix of the official log name
#: (``ep<id>_<infix>_log.jsonl``)
SUBGOAL_TMP = {"QwenVLSubgoalPredictor": ("qwen-tmp", "QwenVL"), "MemERSubgoalPredictor": ("memer-tmp", "MemER")}


def qwen_begin(predictor: Any, scratch: Path, dataset: str, episode_tag: str) -> Path | None:
    """QwenVL / MemER predictors: point the official ``save_dir`` at
    ``<scratch>/{qwen,memer}-tmp/<dataset>/<episode_tag>/`` (the official code appends ``<env_name>/ep<episode_id>``,
    and both logs are written to ``<env_name>/ep<id>_{QwenVL,MemER}_log.jsonl``). Returns None for Oracle."""
    kind = SUBGOAL_TMP.get(type(predictor).__name__)
    if kind is None:
        return None
    base = scratch / kind[0] / str(dataset) / str(episode_tag)
    base.mkdir(parents=True, exist_ok=True)
    predictor.save_dir = base
    predictor.episode_dir = None
    return base


def qwen_end(predictor: Any, base: Path | None, archive_dir: Path | None) -> str | None:
    """End of episode (called on normal end, ``unknown`` and exceptions): archive ``ep<id>_{QwenVL,MemER}_log.jsonl``
    into ``archive_dir``, delete the scratch directory entirely, and remove the empty ``<dataset>`` and
    ``{qwen,memer}-tmp`` parents level by level. Returns the archived path."""
    if base is None:
        return None
    archived = None
    logs = sorted([*base.glob("*/ep*_QwenVL_log.jsonl"), *base.glob("*/ep*_MemER_log.jsonl")])
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for log in logs:
            dst = archive_dir / log.name
            shutil.move(str(log), dst)
            archived = str(dst)
    shutil.rmtree(base, ignore_errors=True)
    predictor.episode_dir = None
    p = base.parent
    for _ in range(2):  # the two levels <dataset> and {qwen,memer}-tmp, removed only when empty
        try:
            p.rmdir()
        except OSError:
            break
        p = p.parent
    return archived


def trace_location(conn_info: dict, recorder: Any) -> Path | None:
    """Trace location: ``trace_path`` -> ``<trace_dir>/trace.jsonl`` -> ``<recorder.out_dir>/trace.jsonl`` -> not
    written."""
    if conn_info.get("trace_path"):
        return Path(conn_info["trace_path"])
    if conn_info.get("trace_dir"):
        return Path(conn_info["trace_dir"]) / "trace.jsonl"
    out = getattr(recorder, "out_dir", None)
    return Path(out) / "trace.jsonl" if out else None


def map_flag(flag: str) -> tuple[str, str | None]:
    """Official return value -> (status, error)."""
    if flag in NORMAL:
        return flag, None
    return "error", f"success_flag={flag}"


# -- new-side runner adapter -----------------------------------------------------------


def official_episode_id(identity: dict, episode_tag: str) -> str:
    """Short episode id handed to the official loop: ``<source_episode>a<attempt>`` (builder_episode when there is no
    source episode number), e.g. ``3a1``.

    Same magnitude as the original side's official ``EnvRunner.episode_id`` (an integer source episode number), so
    overlay video file names do not exceed 255 bytes."""
    src = identity.get("source_episode")
    if src is None:
        src = identity.get("builder_episode")
    att = episode_tag.rsplit(".a", 1)[1] if ".a" in episode_tag else "1"
    return f"{src}a{att}"


class SessionRunner:
    """The official ``EnvRunner`` interface delegated to ``EnvSession``. ``get_init_obs`` / ``step`` match the official
    code line by line."""

    def __init__(self, session: Any, episode_tag: str, pack_state: Callable, tap: EpisodeTap,
                 official_episode_id: str | None = None):
        self._session = session
        self._pack_state = pack_state
        self._tap = tap
        self.env_id = session.task
        # episode_id handed to the official loop: the official eval.py uses it to build the overlay video file name
        # ``{env_id}_ep{episode_id}_{flag}_{task_goal}_{difficulty}.mp4``; with a long task goal (e.g. SwingXtimes)
        # using episode_tag (``<key>.a<n>``) would exceed the 255-byte file name limit, and ffmpeg could not open the
        # output (Broken pipe); so the short id ``official_episode_id`` is used, while this repo's own directories,
        # Qwen scratch directories and traces still use episode_tag.
        self.episode_id = official_episode_id or episode_tag
        self.task_goal: str = ""
        self.info: dict | None = None
        self.last_exception: BaseException | None = None
        env = getattr(session, "env", None)
        self.difficulty = getattr(getattr(env, "unwrapped", None), "difficulty", None)

    def get_init_obs(self) -> dict:
        obs, self.info = self._session.reset()
        if isinstance(self.info["task_goal"], list):
            self.task_goal = self.info["task_goal"][0]
        else:
            self.task_goal = self.info["task_goal"]
        images = obs["front_rgb_list"]
        wrist_images = obs["wrist_rgb_list"]
        states = [self._pack_state(joint_state, gripper_state) for joint_state, gripper_state in
                  zip(obs["joint_state_list"], obs["gripper_state_list"])]
        pre = {"images": images, "wrist_images": wrist_images, "states": states, "task_goal": self.task_goal}
        self._tap.on_reset(pre)
        return pre

    def step(self, action: np.ndarray):
        try:
            obs, _, terminated, truncated, self.info = self._session.step(action)
        except Exception as e:
            if type(e).__name__ in PASS_THROUGH:
                raise
            print(f"Error: {e}")
            self.last_exception = e
            self._tap.on_step(action, (None, None, None), True, "error", None, None,
                              reason=f"{type(e).__name__}: {e}"[:400])
            return (None, None, None), True, "error"

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        joint_state = obs["joint_state_list"][-1]
        gripper_state = obs["gripper_state_list"][-1]
        state = self._pack_state(joint_state, gripper_state)

        outcome = self.info.get("status", "unknown")
        stop = terminated or truncated

        self._tap.on_step(action, (img, wrist_img, state), stop, outcome, terminated, truncated)
        return (img, wrist_img, state), stop, outcome

    @property
    def simple_subgoal_oracle(self) -> str:
        return self.info["simple_subgoal_online"]

    @property
    def grounded_subgoal_oracle(self) -> str:
        return self.info["grounded_subgoal_online"]


# -- seat context ----------------------------------------------------------------------


def default_client_factory(host: str, port: int, episode: dict) -> Any:
    """New-side default: ``framesamp_modul_client.make_recording_client`` (a ``MMEVLAWebsocketClientPolicy``
    subclass with send/receive identical to the parent line by line, additionally recording per-message sha256 in
    recorder events)."""
    framesamp_modul = load_sibling("framesamp_modul_client")
    return framesamp_modul.make_recording_client(host, int(port), episode.get("recorder"), episode.setdefault("timing", {}))


def make_policy_context(seat_info: dict, *, client_factory: Callable | None = None, qwen_extra: dict | None = None,
                        save_dir: str | Path | None = None) -> dict:
    """Built once per seat: official definitions, ``Args``, the subgoal predictor (QwenVL / MemER load their model
    here), ``EpisodeEvaluator``.

    ``seat_info`` reads ``groundsg_variant``, ``policy_seed`` (required, non-negative integer; missing raises
    ``ValueError`` whose text contains ``RUN_BLOCKED reason=policy_seed``), ``qwenvl_groundSG_adapter_path``
    (QwenVL), ``memer_adapter_path`` (MemER). ``client_factory(host, port, episode) -> client`` and ``qwen_extra``
    (stand-ins for the three swift names) are only for test injection."""
    variant = seat_info.get("groundsg_variant")
    if variant not in official_defs.VARIANTS:
        raise ValueError(f"groundsg_variant={variant!r} is not one of {official_defs.VARIANTS}")
    policy_seed = official_defs.check_policy_seed(seat_info.get("policy_seed"))
    ctx: dict[str, Any] = {"variant": variant, "seat_info": dict(seat_info), "episode": None,
                           "client_factory": client_factory or default_client_factory, "policy_seed": policy_seed}

    def ws_factory(host, port):
        ep = ctx["episode"]
        client = TracingClient(ctx["client_factory"](host, port, ep), ep["tap"],
                               timer=ep["chunk_timer"], timing=ep["timing"])
        ep["clients"].append(client)
        return client

    defs = official_defs.load_groundsg(variant, with_env_runner=False, ws_module=official_defs.ws_shim(ws_factory),
                                       qwen_extra=qwen_extra)
    base = Path(save_dir) if save_dir else Path(seat_info.get("trace_root") or seat_info.get("out") or
                                                 tempfile.gettempdir()) / "qwen-tmp"
    args = official_defs.make_args(defs, variant=variant, host=seat_info.get("host", "127.0.0.1"),
                                   port=int(seat_info["port"]), max_steps=int(seat_info["max_steps"]),
                                   model_seed=policy_seed,
                                   adapter_path=seat_info.get("qwenvl_groundSG_adapter_path"),
                                   memer_adapter_path=seat_info.get("memer_adapter_path"), save_dir=str(base))
    t0 = time.perf_counter()
    predictor = official_defs.build_predictor(defs, args, base)
    ctx.update(defs=defs, args=args, predictor=predictor, evaluator=defs["EpisodeEvaluator"](args, base),
               predictor_init_s=time.perf_counter() - t0, official_sha256=dict(defs["sha256"]),
               memer_compat_sha256=defs.get("memer_compat_sha256"))
    print(f"GROUNDSG_CONTEXT variant={variant} max_steps={args.max_steps} predictor={type(predictor).__name__} "
          f"policy_seed={args.model_seed} memer_compat_sha256={ctx['memer_compat_sha256'] or 'none'} "
          f"init_s={ctx['predictor_init_s']:.1f}", flush=True)
    return ctx


def language_params(ctx: dict) -> dict:
    """Fixed decoding parameters of subgoal model calls (temperature and max_tokens are added per call)."""
    args = ctx.get("args")
    variant = ctx.get("variant")
    adapter = None
    if variant == official_defs.VARIANT_QWENVL:
        adapter = getattr(args, "qwenvl_groundSG_adapter_path", None)
    elif variant == official_defs.VARIANT_MEMER:
        adapter = getattr(args, "memer_adapter_path", None)
    out = {"model_id": "Qwen/Qwen3-VL-4B-Instruct" if adapter else None, "adapter": adapter,
           "policy_seed": getattr(args, "model_seed", None)}
    if variant == official_defs.VARIANT_MEMER:
        out["memer_compat_sha256"] = ctx.get("memer_compat_sha256")
    return out


def make_trace(tpath: Path, *, route: str, identity: dict, max_steps: int, policy_seed: int | None,
               effective_cap: Any = None) -> Any:
    """``TraceWriter``: the header's ``policy_seed`` / ``effective_cap`` are passed only if the signature accepts them
    -- when the current ``TraceWriter`` lacks these two parameters they are not passed (identity and the end line
    still record ``policy_seed``)."""
    kw: dict[str, Any] = {}
    try:
        params = inspect.signature(trace_writer.TraceWriter).parameters
    except (TypeError, ValueError):
        params = {}
    if "policy_seed" in params:
        kw["policy_seed"] = policy_seed
    if "effective_cap" in params and effective_cap is not None:
        kw["effective_cap"] = effective_cap
    return trace_writer.TraceWriter(tpath, route=route, identity=identity, max_steps=max_steps, **kw)


def close_policy_context(ctx: Any) -> None:
    """Seat teardown: release the predictor (QwenVL engine) and the evaluator."""
    if not isinstance(ctx, dict):
        return
    for k in ("predictor", "evaluator"):
        ctx.pop(k, None)
    try:
        import gc

        gc.collect()
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# -- official overlay video retention (new side only) ----------------------------------


class OfficialVideoRejected(ValueError):
    """The official overlay video failed verification (count, decodability, frame count, target directory) and is not
    moved into ``official/``."""


def official_safe_filename(full_name: str) -> str:
    """Same as ``render_official_video.safe_filename``: strip ``/``, ``\\`` and NUL; when longer than 255 bytes,
    truncate and append a digest of the full name.

    Duplicated here (6 lines) rather than imported across files to avoid depending on the re-render tool; the two
    rules must stay identical."""
    sanitized = re.sub(r"[/\\\x00]", "_", full_name)
    if sanitized == full_name and len(sanitized.encode()) <= 255:
        return sanitized
    suffix = "__" + hashlib.sha256(full_name.encode()).hexdigest()[:16] + ".mp4"
    stem = sanitized.removesuffix(".mp4").encode()[:255 - len(suffix.encode())].decode(errors="ignore")
    return stem + suffix


def official_video_name(runner: Any, flag: str, task_goal: str) -> str:
    """File name format of the official ``eval_each_episode``
    ``{env_id}_ep{episode_id}_{flag}_{task_goal}_{difficulty}.mp4`` (after truncation)."""
    return official_safe_filename(
        f"{runner.env_id}_ep{runner.episode_id}_{flag}_{task_goal}_{runner.difficulty}.mp4")


def ffmpeg_exe() -> str:
    """The ffmpeg used by the official ``imageio.mimsave``: ``IMAGEIO_FFMPEG_EXE`` first, otherwise the binary bundled
    with imageio-ffmpeg."""
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_video_frames(path: str | Path, ffmpeg: str | None = None) -> int:
    """Fully decode the first video stream and return its frame count; a non-zero ffmpeg exit code or any
    error-level message raises ``OfficialVideoRejected``."""
    cmd = [ffmpeg or ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
           "-map", "0:v:0", "-f", "framemd5", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    err = proc.stderr.decode(errors="replace").strip()
    if proc.returncode != 0 or err:
        raise OfficialVideoRejected(f"decode failed rc={proc.returncode}: {err[:300]}")
    return sum(1 for ln in proc.stdout.decode(errors="replace").splitlines() if ln.strip() and not ln.startswith("#"))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def keep_official_videos(video_dir: str | Path, dst: str | Path, *, expected_frames: int,
                         provenance: dict | None = None, ffmpeg: str | None = None) -> list[str]:
    """Move the official overlay video in ``video_dir`` into ``dst`` (i.e. ``<episode dir>/official/``) and write
    ``provenance.json``.

    Checked before moving: exactly one ``*.mp4`` (not a symlink), full decode without errors, frame count equal to
    ``expected_frames`` (= ``frames_recorded``), ``dst`` absent or empty. Any failure raises
    ``OfficialVideoRejected`` and nothing is moved or written. On success prints ``OFFICIAL_VIDEO=KEPT`` and returns
    the list of moved paths (exactly 1). ``provenance`` is the identity, route, terminal status etc. given by the
    caller; this function adds ``video`` (file name, sha256, bytes) and ``frames`` (decoded frames, expected frames,
    basis)."""
    video_dir, dst = Path(video_dir), Path(dst)
    mp4s = sorted(video_dir.glob("*.mp4")) if video_dir.is_dir() else []
    if len(mp4s) != 1:
        raise OfficialVideoRejected(f"expected exactly 1 official video, got {len(mp4s)}: {[p.name for p in mp4s]}")
    src = mp4s[0]
    if src.is_symlink() or not src.is_file():
        raise OfficialVideoRejected(f"official video is not a regular file: {src.name}")
    frames = decode_video_frames(src, ffmpeg)
    if frames != int(expected_frames):
        raise OfficialVideoRejected(f"frame count mismatch: decoded {frames} != frames_recorded {int(expected_frames)}")
    if dst.is_symlink() or (dst.exists() and (not dst.is_dir() or any(dst.iterdir()))):
        raise OfficialVideoRejected(f"target directory exists and is not empty: {dst}")
    sha, nbytes = _file_sha256(src), src.stat().st_size
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / src.name
    shutil.move(str(src), str(target))
    prov = dict(provenance or {})
    prov["schema"] = PROVENANCE_SCHEMA
    prov["video"] = {"name": target.name, "sha256": sha, "bytes": nbytes}
    frame_info = dict(prov.get("frames") or {})
    frame_info.update(decoded=frames, expected=int(expected_frames), basis=FRAMES_BASIS)
    prov["frames"] = frame_info
    tmp = dst / ".provenance.json.tmp"
    tmp.write_text(json.dumps(prov, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, dst / "provenance.json")
    print(f"OFFICIAL_VIDEO=KEPT source={prov.get('official_source')} frames={frames} sha256={sha[:16]} "
          f"name={target.name}", flush=True)
    return [str(target)]


def episode_counts(tap: EpisodeTap, flag: Any, exc_name: str | None, max_steps: int | None) -> dict:
    """The three C8 counts (derived from ``EpisodeTap`` records and the official loop semantics).

    When the official ``count > max_steps`` it ``break``s before ``record``: the last step of a natural-timeout
    episode has an observation but does not enter the official video, recorded as ``omitted_timeout_frames=1`` (0 if
    that step had no observation anyway). Episodes where ``get_init_obs`` never returned are ``no_frame``."""
    attempted = int(tap.steps)
    observed = attempted - len(tap.missing_steps)
    no_frame = tap.demo_frames is None
    omitted = int(flag == "timeout" and exc_name is None and max_steps is not None and attempted > int(max_steps)
                  and attempted not in tap.missing_steps)
    frames = 0 if no_frame else int(tap.demo_frames) + 1 + observed - omitted
    return {"steps_attempted": attempted, "steps_observed": observed, "frames_recorded": frames,
            "omitted_timeout_frames": omitted, "no_frame": no_frame}


def _status_of(flag: Any, error: str | None, exc_name: str | None) -> tuple[str, str | None]:
    if exc_name == "StepCapReached":
        return "timeout", error
    if error is None:
        return map_flag(flag)
    return "error", error


def _finish_official(captured: dict, runner: Any, tap: EpisodeTap, video_dir: Path, archive_dir: Path | None, *,
                     flag: Any, error: str | None, exc_name: str | None, counts: dict, provenance: dict) -> dict:
    """Finalization with ``keep_official`` on (called before deleting ``video_dir``); this function never raises and
    records every failure in its return value."""
    out: dict[str, Any] = {"official_videos": [], "official_source": "none", "official_save_error": None}
    rec = captured.get("recorder")
    if rec is None:
        if captured.get("init_started") and tap.demo_frames is not None:
            # reset frames were obtained but the official init_episode never handed over the recorder: partial frames do not make a video; only record the reason (never pose as a complete video)
            out["official_source"] = "partial"
            out["official_save_error"] = (f"init_episode failed midway: {tap.demo_frames + 1} reset frames obtained, "
                                          f"official recorder not handed over, no video; {error or ''}")[:800]
        else:
            out["official_save_error"] = f"failed to get the initial observation before or during init_episode, no frame recorded; {error or ''}"[:800]
        return out
    try:
        status, _ = _status_of(flag, error, exc_name)
        has_mp4 = video_dir.is_dir() and any(video_dir.glob("*.mp4"))
        source = "official"
        if not has_mp4:
            if exc_name is None and flag != "unknown":
                raise OfficialVideoRejected(f"official loop ended normally (flag={flag}) but wrote no mp4")
            label = "timeout" if exc_name == "StepCapReached" else ("unknown" if flag == "unknown" else "error")
            name = official_video_name(runner, label, captured.get("task_goal"))
            try:
                rec.save_video(name)  # the official instance's own save_video (never copied or rewritten)
            except Exception as e:  # noqa: BLE001
                raise OfficialVideoRejected(f"salvage save_video failed: {type(e).__name__}: {e}") from e
            source = "official-salvaged"
        if archive_dir is None:
            raise OfficialVideoRejected("no episode directory (archive_dir=None); nowhere to keep it")
        prov = dict(provenance)
        prov.update(official_source=source,
                    terminal={"status": status, "success_flag": flag, "error": error, "exception": exc_name},
                    frames={k: counts[k] for k in ("steps_attempted", "steps_observed", "frames_recorded",
                                                   "omitted_timeout_frames")} | {"demo_frames": tap.demo_frames})
        out["official_videos"] = keep_official_videos(video_dir, Path(archive_dir) / OFFICIAL_VIDEO_SUBDIR,
                                                      expected_frames=counts["frames_recorded"], provenance=prov)
        out["official_source"] = source
    except Exception as e:  # noqa: BLE001 only record the reason; raw frames are kept
        out["official_save_error"] = f"{type(e).__name__}: {e}"[:800]
        print(f"OFFICIAL_VIDEO=REJECTED reason={out['official_save_error'][:200]!r}", flush=True)
    return out


def run_official_episode(ctx: dict, runner: Any, tap: EpisodeTap, *, dataset: str, episode_tag: str,
                         scratch: Path, archive_dir: Path | None, recorder: Any = None,
                         keep_official: Any = _UNSET, official_provenance: Any = _UNSET,
                         language_log: Any = _UNSET) -> dict:
    """Shared by both sides: run one ``eval_each_episode`` on the official evaluator in ``ctx``, catch exceptions,
    clean up temporary directories, and return the terminal-status dict.

    Without ``keep_official`` the video behavior is unchanged; all episodes add chunk and language timing to the
    existing ``timing`` result. With a true value the
    official overlay video is kept (see the module docstring), and ``official_provenance`` is the identity, route
    etc. written into ``provenance.json`` (dict). When ``language_log`` (the return value of ``open_language_log``) is
    not None the language ledger is wired via ``LangTap``, and the predictor and engine are restored at the end of the
    episode (the caller is responsible for ``close``-ing the ledger).

    MemER three bad replies with no previous valid subgoal (``MemERResponseError``): ``status=error``,
    ``error_kind=model_response_error``, not infrastructure (not rerun)."""
    keep = keep_official is not _UNSET and bool(keep_official)
    predictor, evaluator = ctx["predictor"], ctx["evaluator"]
    ctx["episode"] = {"tap": tap, "clients": [], "recorder": recorder, "timing": {}}
    ep = ctx["episode"]
    timer = chunk_timing.ChunkTimer()
    ep["chunk_timer"] = timer
    ep["timing"]["timing_missing"] = 0
    video_dir = scratch / "official-video"
    qbase = qwen_begin(predictor, scratch, dataset, episode_tag)
    lang = None
    undo_lang = None
    if language_log is not _UNSET and language_log is not None:
        lang = LangTap(language_log, tap, variant=ctx.get("variant"), api=getattr(predictor, "api", None),
                       params=language_params(ctx), timer=timer)
        tap.lang = lang
        undo_lang = install_language(predictor, lang)
    counter = None
    counted_engine = None
    api = getattr(predictor, "api", None)
    if ctx.get("variant") != official_defs.VARIANT_ORACLE and hasattr(api, "engine"):
        counted_engine = api.engine
        counter = api.engine = _CountingEngine(counted_engine)
    timed = _TimedPredictor(predictor, timer, variant=ctx.get("variant"), counter=counter, tap=tap)
    error = None
    exc_name = None
    flag = None
    captured: dict[str, Any] = {}
    official: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    had_attr = "init_episode" in vars(evaluator)
    saved_attr = vars(evaluator).get("init_episode")
    if keep:
        orig_init = evaluator.init_episode

        def init_episode_capture(env_runner, epstate, video_save_dir):
            captured["init_started"] = True
            out = orig_init(env_runner, epstate, video_save_dir)
            captured["task_goal"], captured["recorder"] = out
            return out

        evaluator.init_episode = init_episode_capture
    t0 = time.perf_counter()
    try:
        flag = evaluator.eval_each_episode(runner, timed, video_dir)
    except Exception as e:  # noqa: BLE001 recorded as error, same as the episode-level catch-all of the official evaluate
        exc_name = type(e).__name__
        print(f"Error evaluating episode {episode_tag}: {e}")
        flag, error = "error", f"{exc_name}: {e}"[:800]
    finally:
        if keep:
            if had_attr:
                evaluator.init_episode = saved_attr
            else:
                vars(evaluator).pop("init_episode", None)
        if counted_engine is not None:
            api.engine = counted_engine
        if undo_lang is not None:
            undo_lang()
        tap.lang = None
        for c in ctx["episode"]["clients"]:
            c.close()
        qlog = qwen_end(predictor, qbase, archive_dir)
        if keep:
            counts = episode_counts(tap, flag, exc_name, getattr(ctx.get("args"), "max_steps", None))
            prov = dict(official_provenance) if isinstance(official_provenance, dict) else {}
            prov.setdefault("official_sha256", dict(ctx.get("official_sha256") or {}))
            official = _finish_official(captured, runner, tap, video_dir, archive_dir, flag=flag, error=error,
                                        exc_name=exc_name, counts=counts, provenance=prov)
        shutil.rmtree(video_dir, ignore_errors=True)  # the official overlay mp4 is not delivered (with keep it was already moved into official/)
    wall = time.perf_counter() - t0
    status, error = _status_of(flag, error, exc_name)
    env_exc = getattr(runner, "last_exception", None)
    env_exc_s = None if env_exc is None else f"{type(env_exc).__name__}: {env_exc}"[:800]
    infra = classify_infra(error, env_exc_s) if status == "error" else None
    timing = dict(ctx["episode"].get("timing") or {})
    if "per_msg" in timing:  # per-message timing of the new-side recording client: summarized on the same basis as framesamp_modul_client
        timing = load_sibling("framesamp_modul_client").summarize_timing(timing)
    timing["episode_s"] = wall
    cuda = _timing_cuda()
    chunk_timing.attach_policy_timing(timing, timer,
                                     gpu_name=cuda.get_device_name(0) if cuda is not None else None,
                                     model_kind="serial")
    ctx["episode"] = None
    res = {"status": status, "task_success": status == "success", "steps": tap.steps, "error": error,
           "success_flag": flag, "decisions": tap.decisions, "infra": infra is not None, "infra_reason": infra,
           "env_exception": env_exc_s, "exception": exc_name, "qwen_log": qlog, "timing": timing,
           "official_sha256": dict(ctx.get("official_sha256") or {})}
    if "policy_seed" in ctx:  # newer fields; when an old context has no policy_seed the result row is as before
        res.update(policy_seed=ctx["policy_seed"], policy_variant=ctx.get("variant"), subgoal_log=qlog,
                   error_kind=error_kind_of(exc_name, status, infra))
        if ctx.get("variant") == official_defs.VARIANT_MEMER:
            res["memer_compat_sha256"] = ctx.get("memer_compat_sha256")
        if lang is not None:
            res["language"] = lang.summary()
    if keep:
        res.update(counts)
        res.update(official)
    return res


def error_kind_of(exc_name: str | None, status: str, infra: Any) -> str | None:
    """Named error: MemER three bad replies with no previous valid subgoal -> ``model_response_error``; None
    otherwise."""
    if status == "error" and not infra and exc_name == "MemERResponseError":
        return official_defs.MemERResponseError.error_kind
    return None


def run_episode(session, identity: dict, conn_info: dict, recorder) -> dict:
    """Entry point called by the seat runner: one GroundSG (new side) episode. ``session`` is an already built
    EnvSession. ``groundsg_keep_official`` defaults to True; False disables archived MP4 delivery only."""
    keep_official = conn_info.get("groundsg_keep_official", True)
    if not isinstance(keep_official, bool):
        raise ValueError("groundsg_keep_official must be a boolean")
    ctx = conn_info.get("policy_context")
    if not isinstance(ctx, dict) or "evaluator" not in ctx:
        raise RuntimeError("groundsg needs the policy_context built by make_policy_context first")
    variant = conn_info.get("groundsg_variant")
    if variant != ctx["variant"]:
        raise ValueError(f"conn_info groundsg_variant={variant!r} does not match policy_context {ctx['variant']!r}")
    max_steps = int(conn_info["max_steps"])
    if int(ctx["args"].max_steps) != max_steps:
        raise ValueError(f"conn_info max_steps={max_steps} does not match policy_context {ctx['args'].max_steps}")
    dataset = conn_info.get("dataset")
    tag = conn_info.get("episode_tag") or f"{identity.get('key')}.a1"
    attempt = attempt_of(tag)
    tpath = trace_location(conn_info, recorder)
    route = f"groundsg/{variant}/new"
    ident = {k: identity.get(k) for k in ("task", "tier", "seed", "source_episode", "builder_episode", "key")}
    ident["dataset"] = dataset
    ident["attempt"] = attempt  # C6: = the N in the episode directory name <key>.a<N> (the attempt number of the ledger's accepted_attempt_id)
    policy_seed = ctx.get("policy_seed")
    ident["policy_seed"] = policy_seed  # identity must contain attempt and policy_seed
    trace = (make_trace(tpath, route=route, identity=ident, max_steps=max_steps, policy_seed=policy_seed,
                        effective_cap=conn_info.get("effective_cap")) if tpath is not None else None)
    tap = EpisodeTap(trace, missing_step_contract=True)
    runner = SessionRunner(session, tag, ctx["defs"]["pack_state"], tap,
                           official_episode_id=official_episode_id(identity, tag))
    scratch, own_scratch = episode_scratch(conn_info.get("trace_dir"))
    archive_dir = tpath.parent if tpath is not None else None
    prov = {"identity": dict(ident), "route": route, "dataset": dataset, "attempt": attempt, "episode_tag": tag,
            "official_episode_id": runner.episode_id, "official_sha256": dict(ctx.get("official_sha256") or {}),
            "policy_seed": policy_seed, "policy_variant": variant}
    if variant == official_defs.VARIANT_MEMER:
        prov["memer_compat_sha256"] = ctx.get("memer_compat_sha256")
    lang_log = open_language_log(tpath)
    try:
        res = run_official_episode(ctx, runner, tap, dataset=dataset, episode_tag=tag, scratch=scratch,
                                   archive_dir=archive_dir, recorder=recorder, keep_official=keep_official,
                                   official_provenance=prov, language_log=lang_log)
    finally:
        if lang_log is not None:
            lang_log.close()
        if own_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    if not keep_official:
        # Keep native evaluation and temporary encoding; only disable archived MP4 delivery.
        res.update(episode_counts(tap, res["success_flag"], res["exception"], max_steps))
        res.update(official_source="disabled", official_videos=[], official_save_error=None)
    demo = (getattr(session, "timing", None) or {}).get("demo_frames", tap.demo_frames)
    if res.get("no_frame"):
        demo = 0  # C3: a frameless error episode records demo_frames as 0
    if trace is not None:
        extra = {k: res[k] for k in ("steps_attempted", "steps_observed", "frames_recorded", "omitted_timeout_frames")}
        if res.get("no_frame"):
            extra["no_frame"] = True
        # C3: terminal_reason takes the terminal status (timeout for strict-cap, error for unknown); the official raw return value is recorded separately as success_flag
        trace.close(status=res["status"], terminal_reason=res["status"], side="new", demo_frames=demo,
                    decisions=res["decisions"], success_flag=res["success_flag"],
                    official_source=res["official_source"],
                    official_videos=[Path(p).name for p in res["official_videos"]], policy_seed=policy_seed,
                    **extra)
    res.update(side="new", demo_frames=demo, max_steps=max_steps, policy_variant=variant,
               trace_path=str(tpath) if tpath is not None else None)
    return res


def attempt_of(episode_tag: str) -> int:
    """Attempt number N from an episode directory name ``<key>.a<N>``; 1 without ``.a<N>`` (same basis as
    ``official_episode_id``)."""
    m = re.search(r"\.a(\d+)$", str(episode_tag))
    return int(m.group(1)) if m else 1


# -- new interface: the four model-side methods --------------------------------------

from robomme_ood_eval import servers as _servers  # noqa: E402
from robomme_ood_eval.episode import DATASET_MAX_STEPS  # noqa: E402

#: synthetic subgoal given to the action server during warm-up (only so the input shape with a subgoal gets compiled
#: once)
WARMUP_SUBGOAL = "pick up the cube at <128, 128>"


class GroundSGPolicy(_servers.ServedPolicy):
    """GroundSG (``groundsg``; e.g. ``groundsg_variant=ground-sg-oracle``).

    * ``load``: variant pairing -> tokenizer gate -> MME-VLA preflight -> start (or attach) a server of the same shape
      as FrameSamp+Modulation (different ckpt and ``history_config='symbolic-grounded-subgoal.yaml'``) -> the added
      warm-up -> ``make_policy_context`` builds the official definitions and subgoal predictor once
      (``_official_defs.load_groundsg`` + ``build_predictor``). The official ``Args`` hard-codes ``max_steps``, so one
      ``EpisodeEvaluator`` is built per dataset cap (1300, 1800), **sharing the same predictor**
      (``subgoal_predictor.py`` does not read ``max_steps``);
    * ``reset(spec)``: sends no message and never touches the environment -- probes liveness, checks there is an
      evaluator for this episode's step cap, and records this episode's identity;
    * ``play``: switches the context to the ``(Args, EpisodeEvaluator)`` for ``spec.max_steps`` and calls this
      module's ``run_episode`` unchanged (official ``eval_each_episode``: the server ``reset`` comes before
      ``session.reset()``; Oracle reads ``session.info["grounded_subgoal_online"]`` every step, synced via
      ``SessionRunner``); the episode's scratch area is created under ``cfg["work_dir"]`` (default: the system temp
      directory) and deleted entirely at the end of the episode; the trace, language ledger, official overlay video
      (``official/``) and subgoal model logs all go to this episode's raw directory;
    * ``close``: stop the server process group and release the predictor and evaluators.

    ``label`` is ``groundsg-<variant>`` (the model directory name in the output tree). Required cfg: ``ckpt`` (no
    default), ``groundsg_variant``, ``openpi_data_home``, ``tokenizer_sha256`` (when preflight is on); QwenVL / MemER
    additionally need ``qwenvl_groundSG_adapter_path`` / ``memer_adapter_path``. Optional ``mme_vla_py``,
    ``compile_cache``, ``det``, ``warmup``, ``work_dir``; ``client_factory`` and ``qwen_extra`` are only for unit
    tests to inject stand-ins."""

    model = "groundsg"
    requires_ckpt = True

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        self.variant = self.cfg.get("groundsg_variant")
        self.label = f"groundsg-{self.variant}" if self.variant else "groundsg"
        self.qwenvl_adapter = self.cfg.get("qwenvl_groundSG_adapter_path") or self.cfg.get("qwenvl_groundsg_adapter")
        self.memer_adapter = self.cfg.get("memer_adapter_path") or self.cfg.get("memer_adapter")
        self.ckpt: Path | None = None
        self.ctx: dict | None = None
        self.evaluators: dict[int, tuple[Any, Any]] = {}
        self.current_spec = None
        self.keep_official = self.cfg.get("groundsg_keep_official", True)
        if not isinstance(self.keep_official, bool):
            raise ValueError("groundsg_keep_official must be a boolean")

    def load(self) -> None:
        S = _servers
        if self.preflight or self.variant not in S.GROUNDSG_VARIANTS:
            S.variant_pairing(self.variant, self.qwenvl_adapter, self.memer_adapter)
        for k, v in {"USE_HF": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
            os.environ.setdefault(k, v)  # legacy seat client environment (the QwenVL predictor always runs HF offline)
        fm = load_sibling("framesamp_modul_client")
        self.ckpt = fm.load_mme_vla_server(self, self.model)
        if S.flag_on(self.cfg.get("warmup", True)):
            self.load_info["warmup"] = fm.warmup_server(self.host, int(self.port), subgoal=WARMUP_SUBGOAL,
                                                        frames=int(self.cfg.get("warmup_frames", fm.WARMUP_FRAMES)))
        work = self.cfg.get("work_dir")
        self.save_dir = Path(work) / "groundsg-qwen-tmp" if work else Path(tempfile.gettempdir()) / "qwen-tmp"
        steps = sorted(set(DATASET_MAX_STEPS.values()))
        seat = {"groundsg_variant": self.variant, "policy_seed": self.policy_seed, "host": self.host,
                "port": int(self.port), "max_steps": steps[0], "qwenvl_groundSG_adapter_path": self.qwenvl_adapter,
                "memer_adapter_path": self.memer_adapter}
        t0 = time.perf_counter()
        ctx = make_policy_context(seat, client_factory=self.cfg.get("client_factory"),
                                  qwen_extra=self.cfg.get("qwen_extra"), save_dir=self.save_dir)
        self.evaluators[int(ctx["args"].max_steps)] = (ctx["args"], ctx["evaluator"])
        for ms in steps[1:]:  # second evaluator: same official definitions and predictor, only Args.max_steps differs
            args = official_defs.make_args(ctx["defs"], variant=self.variant, host=self.host, port=int(self.port),
                                           max_steps=int(ms), model_seed=ctx["policy_seed"],
                                           adapter_path=self.qwenvl_adapter, memer_adapter_path=self.memer_adapter,
                                           save_dir=str(self.save_dir))
            self.evaluators[int(ms)] = (args, ctx["defs"]["EpisodeEvaluator"](args, self.save_dir))
        self.ctx = ctx
        self.load_info.update(context_s=round(time.perf_counter() - t0, 3), variant=self.variant,
                              evaluators=sorted(self.evaluators))
        print(f"GROUNDSG_EVALUATORS variant={self.variant} max_steps={sorted(self.evaluators)} "
              f"predictor={type(ctx['predictor']).__name__} shared_predictor=1", flush=True)

    def reset(self, spec) -> None:
        super().reset(spec)
        if int(spec.max_steps) not in self.evaluators:
            raise ValueError(f"GroundSG has no evaluator for max_steps={spec.max_steps} (built: {sorted(self.evaluators)})")
        self.current_spec = spec

    def play(self, session, spec, recorder) -> dict:
        ctx = self.ctx
        ctx["args"], ctx["evaluator"] = self.evaluators[int(spec.max_steps)]
        work = self.cfg.get("work_dir")
        if work:
            Path(work).mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="groundsg-", dir=work or None))
        conn = self._conn_info(spec)
        conn.update(groundsg_variant=self.variant, policy_context=ctx, trace_dir=str(scratch),
                    qwenvl_groundSG_adapter_path=self.qwenvl_adapter, memer_adapter_path=self.memer_adapter,
                    groundsg_keep_official=self.keep_official)
        try:
            res = run_episode(session, spec.identity(), conn, recorder)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        res.pop("trace_dir", None)
        return self._finish_play(res)

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self.ctx is not None:
                close_policy_context(self.ctx)
                self.ctx = None
            self.evaluators.clear()
