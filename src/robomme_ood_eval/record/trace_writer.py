"""Per-episode trace ``trace.jsonl``.

The original- and new-side drivers call this module with the signatures and field conventions defined here; field
names and signatures are stable. The trace comparison tool ``gate2_compare.py`` only reads the fields written here.

File format: one ``trace.jsonl`` per episode, one JSON object per line, ``kind`` distinguishes line types:

- ``header`` (exactly one, the first line): ``route``, ``identity`` (task / source_episode / seed / tier / dataset
  etc., supplied by the caller), ``max_steps``, ``schema``.
- ``demo`` (exactly one, before the first step; only header / request / response / history may precede it -- the
  GroundSG official loop resets the policy before taking the initial observation, so request lines may come before
  demo; ``frames=0`` when there is no demo): demo frame count, per-frame front / wrist image sha256, and the states
  and texts of the demo phase.
- ``request``: sha256 and byte count of the normalized bytes of every request sent to the model (GroundSG
  ``reset`` / ``add_buffer`` / ``infer``, every PonderPounce protocol frame, Astra planning / monitoring requests),
  plus ``step`` (the step before which it happened).
- ``response``: the full action chunk returned by the model (``array_record``).
- ``step``: one line per executed step: ``step`` (from 1), ``front_sha256``, ``wrist_sha256``, ``state``, ``action``
  (all ``array_record``: original dtype, shape, sha256, plus the hex of float32 x 8 for human reading),
  ``subgoal``, ``terminated``, ``truncated``, ``status``. Image hashes are computed before images enter the video
  encoder.
- ``history``: history buffer boundaries (e.g. the step range covered by FrameSamp+Modulation ``add_buffer``).
- ``end`` (exactly one, the last line): ``status``, ``exec_steps``, ``terminal_reason`` and caller-supplied fields.

``identical_trace`` means only that the fields above are equal item by item (reports state this coverage).

Helpers:

- ``canonical_bytes(obj)``: normalize a request object (mix of dict / list / array / scalar) into deterministic
  bytes for callers that have no ready serialized bytes to pass to ``log_request``; arrays are represented by
  original dtype, shape and sha256, without converting to float32 first.
- ``validate_trace(rows)``: line-order and structure self-check (header first, demo unique and before the first
  step, end unique and last, steps contiguous from 1, ``end.exec_steps`` equal to the last step); returns a list of
  problems, empty means valid.
- ``find_traces(root)``, ``subgoal_sequence(rows)``: used by ``gate2_compare.py``.
- ``TraceWriter`` raises ``RuntimeError`` on any write after ``close`` (so late writes after finalization are never
  silently lost).
- Suggested identity fields: ``task``, ``source_episode``, ``seed``, ``tier``, ``dataset`` and ``attempt`` (attempt
  number) in ``identity``; ``gate2_compare`` pairs by ``(task, source_episode, seed)`` first and, when one identity
  has several traces, matches result rows by ``attempt``.

Shared contracts C1-C11 (checked item by item by a test helper):

- C1 ``route``: new side is always ``<model>/new`` (``groundsg/<variant>/new``, ``pp/new``, ``astra/new``,
  ``smvla/new``, ``perceptual-framesamp-modul/new``), original side ``<model>/orig`` (``groundsg/<variant>/orig``,
  ``pp/orig``, ``smvla/orig``, ``perceptual-framesamp-modul/orig``).
- C2 the demo segment records all reset frames (including the final initial frame), ``demo.frames ==
  len(demo.states)``; finalize with ``close(..., demo_frames=<demo frame count excluding the initial frame>)`` so
  that ``demo.frames == end.demo_frames + 1``.
- C3 ``end.status`` and ``end.terminal_reason`` take ``success`` / ``fail`` / ``timeout`` / ``error``; a strict-cap
  hit is always ``timeout``; ``error`` episodes may have no frames (``end.no_frame=true``, in which case the demo
  segment may be empty and ``demo_frames`` is 0), and the re-renderer records the reason for frameless episodes
  without producing a video.
- C4 every step records the post-step image, state, action and the current subgoal; actions are recorded with the
  original dtype / shape / bytes actually given to the environment, never converted for the contract; original
  values of non-float32 actions go to ``arrays.npz`` in the same directory under key ``exec_action__%05d`` (0-based
  step index, i.e. ``step - 1``), and once ``arrays.npz`` is written every executed step must have a key.
- C5 ``trace.jsonl``, ``arrays.npz`` and raw frames live in the same episode directory.
- C6 ``identity`` contains ``task``, ``tier``, ``seed``, ``dataset``, ``source_episode`` or ``builder_episode``, a
  ``key`` consistent with the episode directory name ``<key>.a<N>``, and ``attempt`` (= the attempt number N of the
  ledger's ``accepted_attempt_id``).
- C7 a ``None`` subgoal means the model is waiting (the official recording shows the ``[initializing...]``
  placeholder), and the trace keeps the original ``None``; routes without subgoals (FrameSamp+Modulation) are
  ``None`` throughout.
- C8 three counts, written into ``end``: ``steps_attempted`` (steps given to the environment, including steps that
  raised; = result row ``exec_steps``), ``steps_observed`` (steps that returned a valid observation),
  ``frames_recorded`` (frames actually recorded by the official recorder
  = ``demo_frames + 1 + steps_observed - omitted_timeout_frames``); steps without observation are kept via
  ``log_missing_step`` with step number, action and reason (``observed=false``), never deleted or filled in.
- C9 fields that cannot be observed on both sides are written as ``NOT_OBSERVED`` (e.g. ``terminated`` /
  ``truncated`` on the original side); the comparator never treats two ``NOT_OBSERVED`` as equal and reports
  ``not_observed=<n>`` per dimension.
- C10 requests / responses / history boundaries define the "common logical input" per model: GroundSG,
  PonderPounce and FrameSamp+Modulation use the same protocol on both sides and compare raw hashes; SimpleMemVLA is
  embedded on the original side and uses websocket on the new side, so only the logical input (instruction, state,
  frame-hash sequence) and full action chunks are compared.
- C11 ``end.observer_hook_errors=<n>`` (mandatory for read-only observer routes); when greater than 0 that episode's
  ``TRACE_COMPLETE`` counts as failed, without affecting the task score.

To implement C3, C8 and C9, the constants ``NOT_OBSERVED`` and ``UNSET`` were added (additions only): ``log_step``'s
``terminated`` / ``truncated`` accept ``NOT_OBSERVED`` and write it unchanged and accept extra fields ``**extra``,
and ``log_missing_step`` was added; without these new values the written bytes are exactly as before. New optional
parameters of shared functions always use ``UNSET`` as the "not provided" sentinel; ``None`` only means "model is
waiting".

Later additions (additions only):

- ``header`` optionally gains ``policy_seed`` and ``effective_cap`` (not written when the constructor arguments are
  ``None``, so old bytes are unchanged); when ``policy_seed`` is given the ``end`` line carries it too.
- ``log_step`` / ``log_missing_step`` optionally gain ``source_call_id`` and ``chunk_index`` (link to the language
  ledger; without them the step line is still the old 9 keys + old extra keys).
- Full arrays: constructor argument ``arrays_path`` (default ``<trace directory>/arrays.npz``) and switch
  ``collect_arrays`` (default True). Every attempted step collects ``exec_action__%05d`` (key number = step-1), and
  observed steps with a state also collect ``exec_state__%05d``; steps without observation are not zero-filled and
  their numbers go to ``end.arrays.missing_state_steps``. ``close()`` writes via ``merge_write_npz`` (in any order
  relative to other writers such as the recorder, neither overwrites the other), and the ``end`` line holds an
  ``arrays`` summary; merge conflicts do not raise but are recorded in ``arrays.error``.
- ``merge_write_npz(path, mapping)``: the only allowed way to write ``arrays.npz`` (same-key dtype / shape / sha256
  must all match, otherwise ``ArraysConflict``; directory-level exclusive lock + ``path.tmp`` + ``os.replace``
  atomic write).
- ``LanguageLog(path)``: per-episode ``language.jsonl`` language ledger (four line kinds: call_open / message /
  call_close / reuse).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "sgeval-trace/1"
NOT_OBSERVED = "NOT_OBSERVED"  # C9: field not observable on this side


class _Unset:
    """"Not provided" sentinel (C7): distinct from ``None`` (model is waiting)."""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


def _flag(x: Any) -> Any:
    return NOT_OBSERVED if isinstance(x, str) and x == NOT_OBSERVED else bool(x)


def image_sha256(img: Any) -> str | None:
    """sha256 of an image: computed over the contiguous bytes in original dtype and shape; ``None`` returns ``None``."""
    if img is None:
        return None
    arr = np.ascontiguousarray(np.asarray(img))
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}|{arr.shape}|".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def array_record(x: Any) -> dict | None:
    """Identity record of an array: ``{"dtype","shape","sha256","f32hex"}``; hashed without converting to float32
    first.

    ``f32hex`` is for human reading only: hex of the first 8 values after flattening and converting to float32; it is
    not used for equality.
    """
    if x is None:
        return None
    arr = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256()
    h.update(arr.tobytes())
    flat = arr.reshape(-1)
    head = flat[:8].astype(np.float32) if flat.dtype.kind in "fiub" else np.zeros(0, np.float32)
    return {"dtype": arr.dtype.str, "shape": list(arr.shape), "sha256": h.hexdigest(), "f32hex": head.tobytes().hex()}


def bytes_record(payload: bytes) -> dict:
    """Normalized byte record of a request: ``{"sha256","nbytes"}``."""
    return {"sha256": hashlib.sha256(payload).hexdigest(), "nbytes": len(payload)}


ACTION_KEY = "exec_action__%05d"  # key number = step - 1 (C4)
STATE_KEY = "exec_state__%05d"    # key number = step - 1; only observed steps with a state have one


class ArraysConflict(ValueError):
    """``merge_write_npz`` found a same-key array whose dtype / shape / sha256 differs from the existing file."""


def _array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


#: upper bound (seconds) for ``merge_write_npz`` waiting on the directory lock; on timeout raises ``TimeoutError``
#: (``TraceWriter.close`` records it in ``end.arrays.error``)
MERGE_LOCK_TIMEOUT_S = 600.0
_MERGE_LOCK_POLL_S = 0.05


def _lock_dir(directory: Path, timeout_s: float | None = None):
    """Directory-level exclusive lock (``fcntl.flock`` on the directory itself, leaving no lock file in the episode
    directory); returns None (no locking) without fcntl.

    **Local disks only**: per-episode outputs of formal runs are written on node-local disk (not NFS); ``flock``
    semantics on NFS are unreliable, so do not put episode directories on a network disk and rely on this lock.
    Polls non-blockingly and raises ``TimeoutError`` after ``timeout_s`` (default ``MERGE_LOCK_TIMEOUT_S``) without
    the lock, so finalization never hangs forever."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover non-POSIX
        return None
    timeout_s = MERGE_LOCK_TIMEOUT_S if timeout_s is None else float(timeout_s)
    deadline = time.monotonic() + timeout_s
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"merge_write_npz timed out waiting for the directory lock after {timeout_s:g}s: {directory}") from None
                time.sleep(_MERGE_LOCK_POLL_S)
    except BaseException:
        os.close(fd)
        raise


def _unlock_dir(fd) -> None:
    if fd is None:
        return
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def merge_write_npz(path: str | Path, mapping: dict) -> None:
    """The only allowed way to write ``arrays.npz``: read the existing file -> same keys must match in dtype / shape /
    sha256 (otherwise raise ``ArraysConflict`` and write nothing) -> merge -> write ``<path>.tmp`` then
    ``os.replace``.

    Concurrent writers in the same directory are serialized by a directory-level ``flock`` (local disks only; waiting
    longer than ``MERGE_LOCK_TIMEOUT_S`` raises ``TimeoutError``); no file is created when ``mapping`` is empty and
    the file does not exist; nothing is rewritten when there are no new keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = {str(k): np.array(v, copy=True) for k, v in dict(mapping).items()}
    for k, a in new.items():
        if a.dtype.hasobject:
            raise TypeError(f"merge_write_npz refuses object arrays: {k}")
    fd = _lock_dir(path.parent)
    try:
        existing: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                existing = {k: z[k] for k in z.files}
        conflicts = []
        for k, a in new.items():
            old = existing.get(k)
            if old is None:
                continue
            if old.dtype.str != a.dtype.str or tuple(old.shape) != tuple(a.shape) or \
                    _array_sha256(old) != _array_sha256(a):
                conflicts.append(f"{k}: existing {old.dtype.str}{list(old.shape)} new {a.dtype.str}{list(a.shape)}")
        if conflicts:
            raise ArraysConflict(f"{path} has {len(conflicts)} same-key conflicts: " + "; ".join(conflicts[:5]))
        added = [k for k in new if k not in existing]
        if not added:
            return
        merged = {**existing, **{k: new[k] for k in added}}
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:  # pass a file object so np.savez does not append a .npz suffix on its own
                np.savez(fh, **merged)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:  # failed halfway: keep the old file intact and leave no partial temp file
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
    finally:
        _unlock_dir(fd)


class TraceWriter:
    """One instance per episode; lines are appended in call order and ``close`` writes the ``end`` line. Exiting the
    context manager without close finalizes with ``status="error"``."""

    def __init__(self, path: str | Path, *, route: str, identity: dict, max_steps: int,
                 policy_seed: int | None = None, effective_cap: int | None = None,
                 arrays_path: str | Path | None = None, collect_arrays: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._closed = False
        self.exec_steps = 0
        self.policy_seed = None if policy_seed is None else int(policy_seed)
        self.arrays_path = Path(arrays_path) if arrays_path is not None else self.path.parent / "arrays.npz"
        self.collect_arrays = bool(collect_arrays)
        self._actions: dict[int, np.ndarray] = {}
        self._states: dict[int, np.ndarray] = {}
        self._missing_state: set[int] = set()
        header = {"kind": "header", "schema": SCHEMA, "route": route, "identity": identity, "max_steps": int(max_steps)}
        if policy_seed is not None:
            header["policy_seed"] = int(policy_seed)
        if effective_cap is not None:
            header["effective_cap"] = int(effective_cap)
        self._write(header)
        self._demo_written = False

    def _write(self, row: dict) -> None:
        if self._closed:
            raise RuntimeError(f"trace already finalized; refusing to append a {row.get('kind')} line: {self.path}")
        self._fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def log_demo(self, fronts: list, wrists: list, states: list | None = None, texts: list | None = None) -> None:
        """Demo phase (called exactly once): per-frame image hashes, states and texts."""
        assert not self._demo_written, "log_demo may be called only once"
        self._demo_written = True
        self._write({
            "kind": "demo",
            "frames": len(fronts),
            "front_sha256": [image_sha256(f) for f in fronts],
            "wrist_sha256": [image_sha256(w) for w in wrists],
            "states": [array_record(s) for s in (states or [])],
            "texts": list(texts or []),
        })

    def log_request(self, name: str, payload: bytes, *, step: int) -> None:
        """One request sent to the model; ``payload`` is the caller's normalized bytes (e.g. the msgpack
        serialization)."""
        self._write({"kind": "request", "name": name, "step": int(step), **bytes_record(payload)})

    def log_response(self, actions: Any, *, step: int) -> None:
        """The full action chunk returned by the model."""
        self._write({"kind": "response", "step": int(step), "actions": array_record(actions)})

    def log_history(self, start_step: int, end_step: int, *, note: str = "") -> None:
        """History buffer boundary (inclusive step range)."""
        self._write({"kind": "history", "start": int(start_step), "end": int(end_step), "note": note})

    def log_step(self, *, step: int, front: Any, wrist: Any, state: Any, action: Any, subgoal: str | None,
                 terminated: bool, truncated: bool, status: str | None, source_call_id: str | None = None,
                 chunk_index: int | None = None, **extra: Any) -> None:
        """One line after executing step ``step`` (from 1); images are the post-step observation.

        ``terminated`` / ``truncated`` may be ``NOT_OBSERVED`` (C9); ``extra`` is merged into the line unchanged
        (without it the line is byte-identical to the old format). ``source_call_id`` / ``chunk_index``: which call in
        the language ledger this step's action came from and its index within the action chunk; ``None`` is not
        written. With ``collect_arrays`` on, the step's original action (``exec_action__%05d``) and state
        (``exec_state__%05d``, not zero-filled when there is no state) are also collected.
        """
        if not self._demo_written:
            self.log_demo([], [])
        self.exec_steps = max(self.exec_steps, int(step))
        row = {
            "kind": "step", "step": int(step),
            "front_sha256": image_sha256(front), "wrist_sha256": image_sha256(wrist),
            "state": array_record(state), "action": array_record(action),
            "subgoal": subgoal, "terminated": _flag(terminated), "truncated": _flag(truncated), "status": status,
            **extra,
        }
        if source_call_id is not None:
            row["source_call_id"] = source_call_id
        if chunk_index is not None:
            row["chunk_index"] = int(chunk_index)
        self._write(row)
        if self.collect_arrays:
            k = int(step) - 1
            if action is not None:
                self._actions[k] = np.array(action, copy=True)
            if state is not None:
                self._states[k] = np.array(state, copy=True)
                self._missing_state.discard(int(step))
            else:
                self._missing_state.add(int(step))

    def log_missing_step(self, *, step: int, action: Any, reason: str, subgoal: str | None = None,
                         source_call_id: str | None = None, chunk_index: int | None = None, **extra: Any) -> None:
        """C8: a step whose action was given to the environment but returned no valid observation; keeps step number,
        action and reason, with images and state recorded as ``None``."""
        self.log_step(step=step, front=None, wrist=None, state=None, action=action, subgoal=subgoal,
                      terminated=NOT_OBSERVED, truncated=NOT_OBSERVED, status=None,
                      source_call_id=source_call_id, chunk_index=chunk_index,
                      observed=False, missing_reason=str(reason), **extra)

    def _arrays_summary(self) -> dict:
        """Write ``arrays.npz`` at finalization and return the ``end.arrays`` summary; write failures (including
        ``ArraysConflict``) are recorded in ``error`` and not raised."""
        try:
            rel = os.path.relpath(self.arrays_path, self.path.parent)
        except ValueError:  # pragma: no cover different drive letters
            rel = str(self.arrays_path)
        summary: dict[str, Any] = {"path": rel.replace(os.sep, "/"), "action_keys": len(self._actions),
                                   "state_keys": len(self._states),
                                   "missing_state_steps": sorted(self._missing_state)}
        mapping = {ACTION_KEY % k: a for k, a in sorted(self._actions.items())}
        mapping.update({STATE_KEY % k: a for k, a in sorted(self._states.items())})
        if mapping:
            try:
                merge_write_npz(self.arrays_path, mapping)
            except Exception as e:  # noqa: BLE001 finalization never loses the end line because of an array write failure; checkers judge FAIL from error
                summary["error"] = f"{type(e).__name__}: {e}"[:600]
        return summary

    def close(self, *, status: str, terminal_reason: str | None = None, **extra: Any) -> None:
        if self._closed:
            return
        if not self._demo_written:
            self.log_demo([], [])
        row = {"kind": "end", "status": status, "exec_steps": self.exec_steps,
               "terminal_reason": terminal_reason, **extra}
        if self.policy_seed is not None and "policy_seed" not in extra:
            row["policy_seed"] = self.policy_seed
        if self.collect_arrays:
            if "arrays" in extra:  # a caller's legacy arrays string yields to the summary; the original value is kept separately
                row["arrays_caller"] = extra["arrays"]
            row["arrays"] = self._arrays_summary()
        self._write(row)
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close(status="error", terminal_reason=f"exception:{exc_type.__name__}" if exc_type else "unclosed")


# -- language ledger language.jsonl --------------------------------------------------

LANG_MODELS = ("subgoal_model", "action_model", "planner", "monitor")
LANG_DIRS = ("in", "out")
LANG_ROLES = ("system", "user", "assistant", "fields")
LANG_STATUSES = ("reply", "error", "cancelled")
LANG_FALLBACKS = (None, "last_valid", "model_response_error", "continue_last")
#: audit key in server wrapper replies (written by the wrapper; the client pops it before giving actions to the
#: environment)
AUDIT_KEY = "_sgeval_audit"


def _jsonable(x: Any) -> Any:
    """Convert numpy scalars / arrays, bytes, tuples and Paths into JSON-writable values (text unchanged, never
    truncated)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bytes, bytearray, memoryview)):
        b = bytes(x)
        return {"__bytes_sha256__": hashlib.sha256(b).hexdigest(), "nbytes": len(b)}
    if isinstance(x, Path):
        return str(x)
    return x


class LanguageLog:
    """One ``language.jsonl`` per episode: accounts for real model calls.

    Every line uses ``ensure_ascii=False`` and is flushed immediately; callers must write ``dir="in"`` messages
    before actually sending them (writing persists them). ``close()`` adds ``call_close status=cancelled`` for calls
    still open and then closes the file; writing after finalization raises ``RuntimeError``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._closed = False
        self._seq = 0
        self._open: dict[str, int] = {}  # open call -> next message index
        self._known: set[str] = set()

    @staticmethod
    def _ts() -> float:
        return round(time.time(), 6)

    def _write(self, row: dict) -> None:
        if self._closed:
            raise RuntimeError(f"language ledger already finalized; refusing to append a {row.get('kind')} line: {self.path}")
        self._fh.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        self._fh.flush()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def open_calls(self) -> list[str]:
        return list(self._open)

    def open_call(self, model: str, step: int, *, params: dict | None = None, transport_attempt: int = 0,
                  retry: int = 0) -> str:
        if model not in LANG_MODELS:
            raise ValueError(f"unknown model={model!r}, expected one of {LANG_MODELS}")
        call_id = f"c{self._seq + 1:05d}"
        self._write({"kind": "call_open", "call_id": call_id, "model": model, "step": int(step), "params": params,
                     "transport_attempt": int(transport_attempt), "retry": int(retry), "ts": self._ts()})
        self._seq += 1
        self._open[call_id] = 0
        self._known.add(call_id)
        return call_id

    def message(self, call_id: str, *, dir: str, role: str, text: Any, images: list | None = None,  # noqa: A002
                channel: str | None = None, token_ids: Any = None, mask: Any = None, tokenizer: Any = None,
                truncated: Any = None, demo_video: Any = None) -> int:
        if call_id not in self._open:
            raise ValueError(f"call {call_id!r} is not open or already closed")
        if dir not in LANG_DIRS:
            raise ValueError(f"unknown dir={dir!r}")
        if role not in LANG_ROLES:
            raise ValueError(f"unknown role={role!r}")
        idx = self._open[call_id]
        self._write({"kind": "message", "call_id": call_id, "message_index": idx, "dir": dir, "role": role,
                     "text": text, "images": images, "channel": channel, "token_ids": token_ids, "mask": mask,
                     "tokenizer": tokenizer, "truncated": truncated, "demo_video": demo_video, "ts": self._ts()})
        self._open[call_id] = idx + 1
        return idx

    def close_call(self, call_id: str, *, status: str, parsed: Any = None, fallback: str | None = None,
                   server_final_text: Any = None, server_truncated: Any = None) -> None:
        if call_id not in self._open:
            raise ValueError(f"call {call_id!r} is not open or already closed")
        if status not in LANG_STATUSES:
            raise ValueError(f"unknown status={status!r}")
        if fallback not in LANG_FALLBACKS:
            raise ValueError(f"unknown fallback={fallback!r}")
        self._write({"kind": "call_close", "call_id": call_id, "status": status, "parsed": parsed,
                     "fallback": fallback, "server_final_text": server_final_text,
                     "server_truncated": server_truncated, "ts": self._ts()})
        del self._open[call_id]

    def reuse(self, step: int, reused_call_id: str) -> None:
        if reused_call_id not in self._known:
            raise ValueError(f"reused call {reused_call_id!r} does not exist")
        self._write({"kind": "reuse", "step": int(step), "reused_call_id": reused_call_id, "reused_previous": True})

    def close(self) -> None:
        if self._closed:
            return
        try:
            for cid in list(self._open):
                self.close_call(cid, status="cancelled")
        finally:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "LanguageLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_language(path: str | Path) -> list[dict]:
    """Read back one episode's language ledger."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def audit_channel_messages(lang: "LanguageLog", call_id: str, audit: Any) -> tuple[Any, Any]:
    """Write the per-channel tokenization of the server wrapper audit block ``{"channels":[...],
    "server_final_text", ...}`` as ``dir=in role=fields`` messages; returns ``(server_final_text,
    server_truncated)``. When ``audit`` is None or not a dict nothing is written and ``(None, None)`` is returned."""
    if not isinstance(audit, dict):
        return None, None
    truncs = []
    for ch in audit.get("channels") or []:
        if not isinstance(ch, dict):
            continue
        truncs.append(ch.get("truncated"))
        lang.message(call_id, dir="in", role="fields", text=ch.get("text"), channel=ch.get("channel"),
                     token_ids=ch.get("token_ids"), mask=ch.get("mask"), tokenizer=ch.get("tokenizer"),
                     truncated=ch.get("truncated"))
    known = [t for t in truncs if t is not None]
    server_truncated = audit.get("server_truncated", any(bool(t) for t in known) if known else None)
    return audit.get("server_final_text"), server_truncated


def read_trace(path: str | Path) -> list[dict]:
    """Read back one episode's trace (for gate2_compare and tests)."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- normalized bytes, structure self-check, lookup ----------------------------------


def _canon(obj: Any) -> Any:
    """Recursive normalization: arrays -> original dtype/shape/sha256 record; bytes -> sha256; dict keys to strings
    (json then sorts by key)."""
    if isinstance(obj, np.ndarray) or isinstance(obj, np.generic):
        rec = array_record(obj)
        return {"__array__": [rec["dtype"], rec["shape"], rec["sha256"]]}
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"__bytes__": hashlib.sha256(bytes(obj)).hexdigest()}
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, float):
        # floats use an exact round-trip (hex) to avoid json implementation differences
        return {"__float__": float.hex(obj)}
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return {"__repr__": repr(obj)}


def canonical_bytes(obj: Any) -> bytes:
    """Normalized bytes of a request object: the same content always gives the same bytes; arrays are recorded by
    original dtype and shape, not converted to float32."""
    return json.dumps(_canon(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def validate_trace(rows: list[dict]) -> list[str]:
    """Structure self-check; returns a list of problems (empty means valid)."""
    problems: list[str] = []
    if not rows:
        return ["empty trace"]
    if rows[0].get("kind") != "header":
        problems.append("first line is not header")
    if rows[0].get("schema") != SCHEMA:
        problems.append(f"schema is not {SCHEMA}")
    kinds = [r.get("kind") for r in rows]
    if kinds.count("header") != 1:
        problems.append(f"header line count {kinds.count('header')}")
    if kinds.count("demo") != 1:
        problems.append(f"demo line count {kinds.count('demo')}")
    else:
        i = kinds.index("demo")
        if any(k not in ("header", "request", "response", "history") for k in kinds[1:i]):
            problems.append("demo is not before the first step")
    if kinds.count("end") != 1 or kinds[-1] != "end":
        problems.append("end is not the unique last line")
    steps = [int(r["step"]) for r in rows if r.get("kind") == "step"]
    if steps != list(range(1, len(steps) + 1)):
        problems.append("steps are not contiguous from 1")
    if rows[-1].get("kind") == "end" and int(rows[-1].get("exec_steps", -1)) != (steps[-1] if steps else 0):
        problems.append("end.exec_steps does not match the last step")
    return problems


def find_traces(root: str | Path) -> list[Path]:
    """Recursively find ``trace.jsonl`` (returned sorted, deterministic)."""
    return sorted(Path(root).rglob("trace.jsonl"))


def subgoal_sequence(rows: list[dict]) -> list[str]:
    """Per-step ``subgoal`` sequence with consecutive duplicates and empty values removed (for Astra mode
    comparison)."""
    seq: list[str] = []
    for r in rows:
        if r.get("kind") != "step":
            continue
        s = r.get("subgoal")
        if s is None or s == "":
            continue
        if not seq or seq[-1] != s:
            seq.append(s)
    return seq
