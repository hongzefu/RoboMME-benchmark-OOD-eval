"""Per-episode video store (EpisodeRecorder): raw arrays are never degraded; images from both cameras are encoded
asynchronously as lossy AV1 4:4:4.

Since 2026-10-08 raw frames are stored as AV1: the encoding parameters are fixed in the module constant
``AV1_ENCODE_ARGS`` (``libaom-av1 -cpu-used 4 -crf 24 -b:v 0 -pix_fmt yuv444p -g 300 -r 30 -row-mt 1 -threads 2``,
RGB->YUV with bt709 full range), muxed as MKV, and ``meta.json`` records ``raw_codec="av1-yuv444p"`` (legacy outputs
lack this field and are treated as ``ffv1``). Lossy encoding no longer requires "decoded bytes equal the pre-encode
sha256": final verification checks three things instead -- frame count, timestamps (per-frame pts exactly i/fps) and
complete decodability -- and the raw spool is deleted only after it passes; on failure the stream is re-encoded once
synchronously from the raw spool. Actions and states are stored byte-exact in ``arrays.npz`` and can be restored
exactly; images can only be restored approximately.

Design notes:

- Each frame's sha256 (pre-encode bytes) is computed on the calling thread and recorded per stream in
  ``frames-<stream>.jsonl`` (idx, sha256, tag, enc); within one episode and stream, frames with identical sha256
  are encoded only once (``enc`` points at the already encoded frame), and expanding by ``enc`` after decoding
  yields the per-frame images.
- Frame bytes go to a writer thread through a bounded queue: when the queue is full ``add_frames`` blocks
  (backpressure) and never drops frames; the writer thread first appends the raw bytes to
  ``out_dir/.spool/<stream>.raw`` and then feeds them through a pipe to an ffmpeg subprocess (encoding runs in a
  separate process; CPU affinity is set by ``V75_ENCODE_CPUS``).
- During ``set_phase("reset")`` frames are only queued and spooled, not fed to ffmpeg (asynchronous encoding would
  steal CPU and change the wall-clock-timed RRT planning); after switching back to ``"run"`` the writer thread
  catches up from the raw spool.
- ``close()``: the sentinel is enqueued with a deadline (``close_deadline_s``, defaulting to the environment
  variable ``V75_RECORDER_JOIN_S``, then 900 s); if the queue is full and ffmpeg is stuck on stdin, the encoder
  subprocess is killed when the deadline passes to unblock it, so it never waits forever. Then encoding is finalized
  -> frame count / timestamp / decodability checks.
- Degradation only looks at free space on the disk holding ``V75_DATA_ROOT`` (two thresholds, 600 / 150 GiB), and
  prints one ``STORAGE_DEGRADE level= free_gib=`` line whenever the level changes; when ``meta["baseline"]`` or
  ``meta["never_degrade"]`` is true the level is always 0. Under AV1 levels 0 and 1 encode identically; level 2 keeps
  only the first and last 50 image frames (the sha256 list stays complete).

This file depends only on numpy and the standard library and must import under Python 3.10 (SimpleMemVLA venv) and
3.11 (benchmark and the legacy FrameSamp+Modulation client venv).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

LEVEL_THRESHOLDS_GIB = (600.0, 150.0)  # below the first -> level 1, below the second -> level 2
LEVEL2_KEEP = 50  # image frames kept at each end at level 2
_SENTINEL = object()
_LAST_LEVEL_LOCK = threading.Lock()
_LAST_LEVEL = 0  # degradation level last decided in this process (to print only when the level changes)
_FFMPEG_CACHE: dict[str, Any] = {}
#: raw-frame encoding (must not be changed): libaom-av1, 4:4:4, crf 24, speed 4, GOP 300, 30 fps, row multithreading,
#: 2 threads
RAW_CODEC = "av1-yuv444p"
#: codec of legacy outputs (without the raw_codec field)
LEGACY_RAW_CODEC = "ffv1"
FPS = 30
AV1_ENCODER = "libaom-av1"
AV1_ENCODE_ARGS = ("-c:v", "libaom-av1", "-cpu-used", "4", "-crf", "24", "-b:v", "0", "-pix_fmt", "yuv444p",
                   "-g", "300", "-r", str(FPS), "-row-mt", "1", "-threads", "2")
#: RGB->YUV with bt709 full range (encoder side), with the same tags in the bitstream so the decoder restores by tag
AV1_COLOR_FILTER = "scale=out_color_matrix=bt709:out_range=full"
AV1_COLOR_TAGS = ("-color_range", "pc", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709")
#: color restoration when decoding the AV1 raw stream back to rgb24 (symmetric to the encoder side)
AV1_DECODE_FILTER = "scale=in_color_matrix=bt709:in_range=full,format=rgb24"
#: default deadline (seconds) for close() waiting on the writer thread
DEFAULT_CLOSE_DEADLINE_S = 900.0


def _trace_writer():
    """The sibling ``trace_writer`` (the only implementation of ``merge_write_npz``): reuse it if already imported,
    otherwise load it by file path."""
    mod = sys.modules.get("trace_writer")
    if mod is not None and hasattr(mod, "merge_write_npz"):
        return mod
    import importlib.util

    # if the registered module of that name lacks the function (an old copy), load under a private name instead of
    # replacing a module someone else imported
    name = "trace_writer" if mod is None else "_recorder_trace_writer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "trace_writer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dumps(obj: Any) -> str:
    """Project-wide jsonl line format."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def sha256_bytes(b: bytes | memoryview) -> str:
    return hashlib.sha256(b).hexdigest()


def frame_sha256(frame: np.ndarray) -> str:
    """sha256 of one frame (H,W,3 uint8): computed over C-contiguous bytes, consistent with the proxy, client hooks
    and comparison tools."""
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """sha256 of a numeric array: bytes + dtype + shape all go into the digest."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(a.tobytes())
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- ffmpeg and degradation level

def find_ffmpeg() -> str:
    """Find an ffmpeg that supports libaom-av1 encoding: V75_FFMPEG > /usr/bin/ffmpeg > PATH > the one bundled with
    imageio_ffmpeg. Raises if none is found."""
    if "exe" in _FFMPEG_CACHE:
        return _FFMPEG_CACHE["exe"]
    cands: list[str] = []
    if os.environ.get("V75_FFMPEG"):
        cands.append(os.environ["V75_FFMPEG"])
    cands.append("/usr/bin/ffmpeg")
    w = shutil.which("ffmpeg")
    if w:
        cands.append(w)
    try:
        import imageio_ffmpeg  # type: ignore

        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for c in cands:
        if not (c and os.path.isfile(c) and os.access(c, os.X_OK)):
            continue
        try:
            out = subprocess.run([c, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60).stdout
        except Exception:
            continue
        if f" {AV1_ENCODER} " in out:
            _FFMPEG_CACHE["exe"] = c
            _FFMPEG_CACHE["libx264"] = " libx264 " in out
            _FFMPEG_CACHE["ffv1"] = " ffv1 " in out
            return c
    raise RuntimeError(f"no ffmpeg supporting {AV1_ENCODER} found; candidates: {cands}")


class RecorderError(RuntimeError):
    """Recording failure (writer thread died, encoder misconfigured, ...). Hooks catch it and mark this episode's
    recording FAIL; the evaluation itself is unaffected."""


def parse_cpu_list(text: str) -> set[int]:
    """Parse a taskset-style CPU list: "3", "2-3", "1,5-6"."""
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def check_encode_cpus() -> str:
    """Check that V75_ENCODE_CPUS is a subset of this process's CPU affinity; raise RecorderError if invalid (fail
    early). Returns the normalized value or an empty string."""
    cpus = os.environ.get("V75_ENCODE_CPUS", "").strip()
    if not cpus:
        return ""
    try:
        want = parse_cpu_list(cpus)
    except ValueError as e:
        raise RecorderError(f"cannot parse V75_ENCODE_CPUS: {cpus!r} ({e})")
    allowed = os.sched_getaffinity(0)
    if not want or not want <= allowed:
        raise RecorderError(f"V75_ENCODE_CPUS={cpus} is not within this process's CPU affinity {sorted(allowed)}")
    if not shutil.which("taskset"):
        raise RecorderError("V75_ENCODE_CPUS is set but taskset was not found")
    return cpus


def _cpu_prefix() -> list[str]:
    """CPU affinity prefix for encode / decode subprocesses (V75_ENCODE_CPUS, e.g. "3" or "2-3"; validated when the
    recorder is initialized)."""
    cpus = os.environ.get("V75_ENCODE_CPUS", "").strip()
    if cpus and shutil.which("taskset"):
        return ["taskset", "-c", cpus]
    return []


def free_gib_default(path: str | Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(str(p)).free / 2**30


def degrade_level(free_gib: float) -> int:
    if free_gib < LEVEL_THRESHOLDS_GIB[1]:
        return 2
    if free_gib < LEVEL_THRESHOLDS_GIB[0]:
        return 1
    return 0


def _note_level(level: int, free_gib: float) -> None:
    """Print a STORAGE_DEGRADE line when the level differs from the previous one (deduplicated per process)."""
    global _LAST_LEVEL
    with _LAST_LEVEL_LOCK:
        if level != _LAST_LEVEL:
            print(f"STORAGE_DEGRADE level={level} free_gib={free_gib:.1f}", flush=True)
            _LAST_LEVEL = level


class _Empty(Exception):
    pass


class _BoundedQueue:
    """Bounded blocking queue (not the stdlib queue: a same-named queue.py in an old evaluation script directory used
    to shadow it when on sys.path[0], so this approach is kept)."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._items: collections.deque = collections.deque()
        self._cv = threading.Condition()

    def put(self, item: Any, timeout: float | None = None) -> bool:
        """Put an item; wait when the queue is full (backpressure). With a timeout, return False on timeout so the
        caller can check whether the writer thread is still alive."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while len(self._items) >= self.maxsize:
                if deadline is None:
                    self._cv.wait()
                else:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    self._cv.wait(left)
            self._items.append(item)
            self._cv.notify_all()
            return True

    def get(self, timeout: float) -> Any:
        with self._cv:
            if not self._items:
                self._cv.wait(timeout)
            if not self._items:
                raise _Empty
            item = self._items.popleft()
            self._cv.notify_all()
            return item


# ---------------------------------------------------------------- per-stream state

class _Stream:
    """One camera stream: per-frame records (written by the calling thread under the recorder lock) and encoding state
    (owned by the writer thread)."""

    def __init__(self, name: str, shape: tuple, spool_dir: Path):
        self.name = name
        self.shape = shape  # (H, W, C)
        self.frame_bytes = int(np.prod(shape))
        self.records: list[dict] = []  # per frame {idx, sha256, tag, enc}
        self.sha2enc: dict[str, int] = {}
        self.enc_sha: list[str] = []  # encode index -> sha256
        self.n_enqueued = 0
        self.tail: collections.deque = collections.deque(maxlen=LEVEL2_KEEP)  # level 2: buffered tail (idx, sha, bytes)
        # owned by the writer thread from here on
        self.spool_path = spool_dir / f"{name}.raw"
        self.spool_w = None
        self.n_spooled = 0
        self.n_fed = 0
        self.proc: subprocess.Popen | None = None
        self.proc_rusage_cpu = 0.0
        self.reordered = 0
        self.encode_error: str | None = None


class EpisodeRecorder:
    """Recorder for one episode, thread safe. See the module docstring for usage; ``close()`` must be called once."""

    def __init__(self, out_dir: str | Path, meta: dict, *, lossless: bool = True, encode_async: bool = True,
                 queue_frames: int = 64, fps: int = 30,
                 free_gib_fn: Callable[[str], float] | None = None,
                 encode_delay_s: float = 0.0, overwrite: bool = False, close_deadline_s: float | None = None):
        self.out_dir = Path(out_dir)
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"recording directory is not empty; refusing to overwrite (pass overwrite=True "
                                      f"explicitly): {self.out_dir}")
            shutil.rmtree(self.out_dir)
        self.encode_cpus = check_encode_cpus()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.spool_dir = self.out_dir / ".spool"
        self.spool_dir.mkdir(exist_ok=True)
        self.meta = dict(meta)
        self.fps = int(fps)
        self.encode_async = bool(encode_async)
        self._encode_delay_s = float(encode_delay_s)  # only for unit tests simulating slow encoding
        watch = os.environ.get("V75_DATA_ROOT") or str(self.out_dir)
        force_lossless = bool(self.meta.get("baseline") or self.meta.get("never_degrade"))
        free = (free_gib_fn or free_gib_default)(watch)
        level = degrade_level(free)
        _note_level(level, free)
        if force_lossless:
            level = 0
        elif not lossless:
            level = max(level, 1)
        self.level = level
        self.free_gib = free
        self.ffmpeg = find_ffmpeg()
        if close_deadline_s is None:
            close_deadline_s = float(os.environ.get("V75_RECORDER_JOIN_S", DEFAULT_CLOSE_DEADLINE_S))
        self.close_deadline_s = float(close_deadline_s)
        self._lock = threading.RLock()
        self._streams: dict[str, _Stream] = {}
        self._arrays: list[tuple[str, int, int | None, np.ndarray]] = []
        self._array_counts: dict[str, int] = collections.defaultdict(int)
        self._event_seq = 0
        self._events_fh = open(self.out_dir / "events.jsonl", "w", encoding="utf-8")
        self._q = _BoundedQueue(max(1, int(queue_frames)))
        self._run_phase = threading.Event()  # set = run (may encode); cleared = reset (queue only)
        self._phase = "reset"
        self._queue_wait_s = 0.0
        self._closed = False
        self._writer_error: str | None = None  # writer thread exception (e.g. ENOSPC); once set, add_* raise RecorderError
        self._discarded = 0  # queued frames discarded after the writer thread failed (counted as dropped)
        self._t0 = time.time()
        self.meta.update(level=self.level, free_gib=round(free, 1), watch_path=watch,
                         codec=RAW_CODEC, raw_codec=RAW_CODEC, encode_args=list(AV1_ENCODE_ARGS),
                         color=AV1_COLOR_FILTER, ffmpeg=self.ffmpeg,
                         fps=self.fps, encode_cpus=self.encode_cpus or None, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=os.getpid())
        (self.out_dir / "meta.json").write_text(dumps(self.meta) + "\n", encoding="utf-8")
        self._writer = None
        if self.encode_async:
            self._writer = threading.Thread(target=self._writer_loop, name="v75-recorder-writer", daemon=True)
            self._writer.start()

    # ------------------------------------------------------------ public interface

    def set_phase(self, phase: str) -> None:
        if phase not in ("reset", "run"):
            raise ValueError(f"phase must be reset / run: {phase}")
        with self._lock:
            self._phase = phase
        if phase == "run":
            self._run_phase.set()
        else:
            self._run_phase.clear()
        self.add_event({"kind": "phase", "phase": phase})

    def add_frames(self, stream: str, frames, *, tag: str = "") -> list[int]:
        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.ndim != 4 or arr.dtype != np.uint8:
            raise ValueError(f"add_frames needs uint8 (N,H,W,C)/(H,W,C), got {arr.dtype} {arr.shape}")
        idxs: list[int] = []
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder is already closed")
            self._check_writer()
            st = self._streams.get(stream)
            if st is None:
                st = _Stream(stream, tuple(arr.shape[1:]), self.spool_dir)
                self._streams[stream] = st
            if tuple(arr.shape[1:]) != st.shape:
                raise ValueError(f"stream {stream} frame shape changed: {st.shape} -> {arr.shape[1:]}")
            for i in range(arr.shape[0]):
                b = np.ascontiguousarray(arr[i]).tobytes()
                sha = sha256_bytes(b)
                idx = len(st.records)
                rec = {"idx": idx, "sha256": sha, "tag": tag, "enc": None}
                st.records.append(rec)
                idxs.append(idx)
                if sha in st.sha2enc:
                    rec["enc"] = st.sha2enc[sha]
                elif self.level >= 2 and idx >= LEVEL2_KEEP:
                    st.tail.append((idx, sha, b))  # whether it belongs to the last 50 frames is decided at close
                else:
                    rec["enc"] = self._enqueue(st, sha, b)
        return idxs

    def add_array(self, name: str, arr, *, step: int | None = None) -> None:
        a = np.array(arr, copy=True)  # copy host data; never keep a reference to the caller's array
        with self._lock:
            self._check_writer()
            k = self._array_counts[name]
            self._array_counts[name] += 1
            self._arrays.append((name, k, step, a))

    def add_event(self, event: dict) -> None:
        with self._lock:
            ev = dict(event)
            ev.setdefault("t", round(time.time() - self._t0, 6))
            ev["seq"] = self._event_seq
            self._event_seq += 1
            self._events_fh.write(dumps(ev) + "\n")
            self._events_fh.flush()

    def close(self, summary: dict) -> dict:
        t_close = time.time()
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder closed twice")
            # level 2: the frames in the tail buffer are the last 50 frames; encode them now
            try:
                for st in self._streams.values():
                    for idx, sha, b in list(st.tail):
                        rec = st.records[idx]
                        rec["enc"] = st.sha2enc[sha] if sha in st.sha2enc else self._enqueue(st, sha, b)
                    st.tail.clear()
            except RecorderError as e:  # the writer thread already failed: close neither raises nor blocks; the result is FAIL
                self._writer_error = self._writer_error or str(e)
            self._closed = True
        self.set_phase_run_for_close()
        if self._writer is not None:
            # enqueuing the sentinel and waiting for the writer thread share one deadline: with a full queue and
            # ffmpeg stuck on stdin the writer thread can never drain the queue, and the old "join only after a
            # successful enqueue" looped forever. When the deadline passes, kill the encoder subprocess first (the
            # writer thread's stdin.write then errors out), then give the writer 30 s to drain the queue and exit on
            # the sentinel.
            deadline = time.monotonic() + self.close_deadline_s
            sent = False
            while self._writer.is_alive() and time.monotonic() < deadline:
                if self._q.put(_SENTINEL, timeout=min(1.0, max(0.01, deadline - time.monotonic()))):
                    sent = True
                    break
            if sent:
                self._writer.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._writer.is_alive():  # writer thread stuck (e.g. ffmpeg not reading the pipe): kill encoders, record FAIL, do not block
                self._writer_error = self._writer_error or "writer thread timed out during close"
                self._kill_encoders()
                if not sent:
                    sent = self._q.put(_SENTINEL, timeout=30.0)
                self._writer.join(timeout=30)
        else:
            try:
                self._drain_sync()
            except Exception as e:
                self._writer_error = f"{type(e).__name__}: {e}"
        encode_cpu = sum(st.proc_rusage_cpu for st in self._streams.values())
        # decode check; on mismatch re-encode once from the raw spool
        t_verify = time.time()
        verify: dict[str, dict] = {}
        reencoded = 0
        for st in self._streams.values():
            v = self._verify_stream(st)
            # after a writer close timeout (encoder already killed) do not re-encode: it may hang too, and close must
            # be bounded
            if (v["mismatch"] or v["error"]) and not self._writer_error:
                reencoded += 1
                try:
                    self._reencode_from_spool(st)
                except Exception as e:  # raw spool missing / corrupt: record the error, close does not raise
                    st.encode_error = f"re-encode failed {type(e).__name__}: {e}"
                encode_cpu += st.proc_rusage_cpu
                v = self._verify_stream(st)
                v["reencoded"] = True
            verify[st.name] = v
        verify_s = time.time() - t_verify
        frames = sum(len(st.records) for st in self._streams.values())
        dropped = sum(st.n_enqueued - st.n_spooled for st in self._streams.values())
        if self._writer_error:
            errors_writer = [f"writer: {self._writer_error}"]
        else:
            errors_writer = []
        reordered = sum(st.reordered for st in self._streams.values())
        mismatch = sum(v["mismatch"] for v in verify.values())
        errors = errors_writer + [f"{k}: {v['error']}" for k, v in verify.items() if v["error"]]
        ok = mismatch == 0 and dropped == 0 and reordered == 0 and not errors and self._writer_ok()
        # write per-frame manifests, raw arrays and summary
        for st in self._streams.values():
            with open(self.out_dir / f"frames-{st.name}.jsonl", "w", encoding="utf-8") as fh:
                for rec in st.records:
                    fh.write(dumps(rec) + "\n")
        try:
            self._write_arrays()
        except Exception as e:  # noqa: BLE001 array write failed (including key conflicts with the trace): do not raise, record FAIL
            errors.append(f"arrays: {type(e).__name__}: {e}"[:800])
            ok = False
        with self._lock:
            self._events_fh.close()
        nbytes = sum(p.stat().st_size for p in self.out_dir.glob("*.mkv"))
        if ok:
            shutil.rmtree(self.spool_dir, ignore_errors=True)
        for st in self._streams.values():
            ep = self._err_path(st)
            if ep.exists() and ep.stat().st_size == 0:
                ep.unlink()
        res = {
            "RECORDER_VERIFY": "PASS" if ok else "FAIL",
            "frames": frames,
            "encoded_frames": sum(st.n_enqueued for st in self._streams.values()),
            "decode_mismatch": mismatch,
            "dropped": dropped,
            "reordered": reordered,
            "bytes": nbytes,
            "level": self.level,
            "lossless": False,
            "raw_codec": RAW_CODEC,
            "encode_cpu_s": round(encode_cpu, 3),
            "queue_wait_s": round(self._queue_wait_s, 3),
            "verify_s": round(verify_s, 3),
            "finalize_s": round(time.time() - t_close, 3),
            "reencoded_streams": reencoded,
            "discarded_after_writer_error": self._discarded,
            "errors": errors,
            "streams": {k: {**v, "frames": len(self._streams[k].records),
                            "encoded": self._streams[k].n_enqueued} for k, v in verify.items()},
            "summary": summary,
        }
        (self.out_dir / "summary.json").write_text(dumps(res) + "\n", encoding="utf-8")
        return res

    # ------------------------------------------------------------ internals: enqueue, writer thread, encoding

    def _writer_ok(self) -> bool:
        return self._writer is None or not self._writer.is_alive()

    def _kill_encoders(self) -> None:
        for st in self._streams.values():
            proc = st.proc
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def set_phase_run_for_close(self) -> None:
        """Force encoding to be allowed during close (no phase event is written; the event file is closed later)."""
        self._run_phase.set()

    def _enqueue(self, st: _Stream, sha: str, b: bytes) -> int:
        """Called with the lock held: assign an encode index and enqueue, blocking (backpressure when full)."""
        enc = st.n_enqueued
        st.n_enqueued += 1
        st.sha2enc[sha] = enc
        st.enc_sha.append(sha)
        item = (st.name, enc, b)
        if self._writer is None:
            self._spool_write(st, enc, b)
            return enc
        t = time.perf_counter()
        try:
            while not self._q.put(item, timeout=1.0):
                self._check_writer()
        finally:
            self._queue_wait_s += time.perf_counter() - t
        return enc

    def _check_writer(self) -> None:
        """Raise RecorderError if the writer thread has failed or died (outside close), so callers never block
        forever."""
        if self._writer_error:
            raise RecorderError(f"recorder writer thread failed: {self._writer_error}")
        if self._writer is not None and not self._writer.is_alive() and not self._closed:
            self._writer_error = "recorder writer thread has exited"
            raise RecorderError(self._writer_error)

    def _spool_write(self, st: _Stream, enc: int, b: bytes) -> None:
        if st.spool_w is None:
            st.spool_w = open(st.spool_path, "ab")
        if enc != st.n_spooled:
            st.reordered += 1
        st.spool_w.write(b)
        st.n_spooled += 1

    def _writer_loop(self) -> None:
        pending = True
        while True:
            try:
                item = self._q.get(timeout=0.05 if pending else 0.5)
            except _Empty:
                item = None
            if item is _SENTINEL:
                break
            if self._writer_error:  # already failed: keep draining the queue so callers unblock; count as discarded
                if item is not None:
                    self._discarded += 1
                continue
            try:
                if item is not None:
                    name, enc, b = item
                    self._spool_write(self._streams[name], enc, b)
                pending = False
                if self._run_phase.is_set():
                    pending = self._feed_some(max_frames=8)
            except BaseException as e:  # e.g. ENOSPC: record the error; add_* will raise RecorderError afterwards
                self._writer_error = f"{type(e).__name__}: {e}"
        # finalize: feed everything, close the pipes, wait for ffmpeg
        for st in list(self._streams.values()):
            try:
                self._finish_stream(st)
            except BaseException as e:
                st.encode_error = st.encode_error or f"{type(e).__name__}: {e}"

    def _drain_sync(self) -> None:
        for st in list(self._streams.values()):
            self._finish_stream(st)

    def _ffmpeg_encode_cmd(self, st: _Stream, out: Path) -> list[str]:
        h, w, c = st.shape
        if c != 3:
            raise ValueError(f"only 3-channel images are supported: {st.shape}")
        return av1_encode_cmd(self.ffmpeg, w, h, out, fps=self.fps, prefix=_cpu_prefix())

    def _err_path(self, st: _Stream) -> Path:
        return self.out_dir / f".ffmpeg-{st.name}.log"

    def _start_proc(self, st: _Stream) -> None:
        out = self.out_dir / f"{st.name}.mkv"
        with open(self._err_path(st), "ab") as err:  # stderr goes to a file, not a pipe (a full pipe would deadlock)
            st.proc = subprocess.Popen(self._ffmpeg_encode_cmd(st, out), stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=err)

    def _feed_some(self, max_frames: int) -> bool:
        """Feed at most max_frames frames from the raw spool to ffmpeg; returns whether anything is left."""
        left = False
        for st in list(self._streams.values()):
            if st.encode_error:
                continue
            n = min(st.n_spooled - st.n_fed, max_frames)
            if n <= 0:
                continue
            try:
                self._feed(st, n)
            except Exception as e:  # encoding failure does not affect recording itself; close re-encodes from the raw spool
                st.encode_error = f"{type(e).__name__}: {e}"
                continue
            if st.n_spooled > st.n_fed:
                left = True
        return left

    def _feed(self, st: _Stream, n: int) -> None:
        if st.spool_w is not None:
            st.spool_w.flush()
        if st.proc is None:
            self._start_proc(st)
        with open(st.spool_path, "rb") as fh:
            fh.seek(st.n_fed * st.frame_bytes)
            data = fh.read(n * st.frame_bytes)
        if len(data) != n * st.frame_bytes:
            raise IOError(f"raw spool read-back length mismatch {len(data)} != {n * st.frame_bytes}")
        if self._encode_delay_s:
            time.sleep(self._encode_delay_s * n)
        st.proc.stdin.write(data)
        st.n_fed += n

    def _wait_proc(self, proc: subprocess.Popen, err_path: Path) -> tuple[int, float, str]:
        _pid, status, ru = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        try:
            err = err_path.read_text(errors="replace")[-2000:]
        except OSError:
            err = ""
        return proc.returncode, ru.ru_utime + ru.ru_stime, err

    def _finish_stream(self, st: _Stream) -> None:
        if st.spool_w is not None:
            st.spool_w.flush()
            st.spool_w.close()
            st.spool_w = None
        if st.encode_error is None:
            try:
                while st.n_fed < st.n_spooled:
                    self._feed(st, min(64, st.n_spooled - st.n_fed))
            except Exception as e:
                st.encode_error = f"{type(e).__name__}: {e}"
        if st.proc is not None:
            try:
                st.proc.stdin.close()
            except Exception:
                pass
            rc, cpu, err = self._wait_proc(st.proc, self._err_path(st))
            st.proc_rusage_cpu = cpu
            if rc != 0 and st.encode_error is None:
                st.encode_error = f"ffmpeg rc={rc}: {err}"
            st.proc = None

    def _reencode_from_spool(self, st: _Stream) -> None:
        """Synchronously re-encode a whole stream from the raw spool (called on encode failure or decode mismatch;
        the environment is not re-run)."""
        out = self.out_dir / f"{st.name}.mkv"
        with open(st.spool_path, "rb") as fh, open(self._err_path(st), "ab") as ef:
            p = subprocess.Popen(self._ffmpeg_encode_cmd(st, out), stdin=fh,
                                 stdout=subprocess.DEVNULL, stderr=ef)
            rc, cpu, err = self._wait_proc(p, self._err_path(st))
        st.proc_rusage_cpu = cpu
        st.encode_error = None if rc == 0 else f"re-encode ffmpeg rc={rc}: {err}"

    def _verify_stream(self, st: _Stream) -> dict:
        """AV1 is lossy: decode completely once (framemd5), check the frame count equals the number of encode indices
        and that per-frame timestamps are exactly i/fps; bytes are no longer compared."""
        res = {"mismatch": 0, "decoded": 0, "error": st.encode_error, "timestamps_ok": False}
        if st.n_enqueued == 0:
            return res
        out = self.out_dir / f"{st.name}.mkv"
        if not out.exists():
            res["error"] = res["error"] or "mkv does not exist"
            res["mismatch"] = st.n_enqueued
            return res
        try:
            frames = probe_decoded_frames(self.ffmpeg, out)
        except Exception as e:
            res["error"] = f"decode failed {type(e).__name__}: {e}"
            res["mismatch"] = st.n_enqueued
            return res
        res["decoded"] = len(frames)
        res["mismatch"] = abs(len(frames) - st.n_enqueued)
        bad_ts = [i for i, t in enumerate(frames) if abs(t - i / self.fps) > 0.5 / self.fps]
        res["timestamps_ok"] = not bad_ts
        if bad_ts:
            res["error"] = f"timestamp mismatch on {len(bad_ts)} frames (first idx={bad_ts[0]} t={frames[bad_ts[0]]:.4f})"
        elif res["mismatch"] == 0:
            res["error"] = None  # passing frame count, timestamps and complete decoding is authoritative
        return res

    def _write_arrays(self) -> None:
        with self._lock:
            items = list(self._arrays)
        if not items:
            return
        payload = {f"{name}__{k:05d}": a for name, k, _s, a in items}
        # the only write path is merge_write_npz, so finalizing this and the sibling trace in either order never
        # overwrites the other; a same-key conflict raises ArraysConflict, which close() records in errors
        # (RECORDER_VERIFY=FAIL)
        _trace_writer().merge_write_npz(self.out_dir / "arrays.npz", payload)
        with open(self.out_dir / "arrays-index.jsonl", "w", encoding="utf-8") as fh:
            for name, k, step, a in items:
                fh.write(dumps({"key": f"{name}__{k:05d}", "name": name, "k": k, "step": step,
                                "dtype": a.dtype.str, "shape": list(a.shape), "sha256": array_sha256(a)}) + "\n")


# ---------------------------------------------------------------- read-back tools (for comparison)

def av1_encode_cmd(ffmpeg: str, w: int, h: int, out: str | Path, *, fps: int = FPS,
                   prefix: list[str] | None = None) -> list[str]:
    """Full command for rgb24 raw frames (piped input) -> AV1 4:4:4 MKV (fixed parameters)."""
    return list(prefix or []) + [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                                 "-i", "pipe:0", "-an", "-vf", AV1_COLOR_FILTER, *AV1_ENCODE_ARGS,
                                 *AV1_COLOR_TAGS, str(out)]


def probe_decoded_frames(ffmpeg: str, path: str | Path) -> list[float]:
    """Decode completely once (``-f framemd5``, ffmpeg only) and return per-frame display times (seconds); raises
    RuntimeError on decode failure."""
    cmd = _cpu_prefix() + [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(path),
                           "-map", "0:v:0", "-f", "framemd5", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed rc={p.returncode}: {p.stderr.decode(errors='replace')[-1000:]}")
    tb = None
    out: list[float] = []
    for line in p.stdout.decode(errors="replace").splitlines():
        if line.startswith("#tb 0:"):
            num, den = line.split(":", 1)[1].strip().split("/")
            tb = int(num) / int(den)
        elif line and not line.startswith("#"):
            parts = [x.strip() for x in line.split(",")]
            if parts[0] != "0":
                continue
            if tb is None:
                raise RuntimeError("framemd5 output is missing the #tb line")
            out.append(int(parts[2]) * tb)
    return out


def raw_codec_of(out_dir: str | Path) -> str:
    """Read ``raw_codec`` from the episode directory's ``meta.json``; without the field (legacy outputs) it is
    ``ffv1``."""
    try:
        meta = json.loads((Path(out_dir) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LEGACY_RAW_CODEC
    return str(meta.get("raw_codec") or LEGACY_RAW_CODEC)


def decode_raw_frames(ffmpeg: str, path: str | Path, shape: tuple, *, raw_codec: str = RAW_CODEC) -> list[bytes]:
    """Decode an mkv into a list of rgb24 raw frame bytes; AV1 streams restore color as bt709 full range, legacy FFV1
    (gbrp) is converted directly."""
    h, w, c = shape
    conv = ["-vf", AV1_DECODE_FILTER] if raw_codec != LEGACY_RAW_CODEC else []
    cmd = _cpu_prefix() + [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(path), *conv,
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed rc={p.returncode}: {p.stderr.decode(errors='replace')[-1000:]}")
    fb = h * w * c
    data = p.stdout
    if len(data) % fb:
        raise RuntimeError(f"decoded byte count {len(data)} is not a multiple of the frame size {fb}")
    return [data[i:i + fb] for i in range(0, len(data), fb)]


def load_frames(out_dir: str | Path, stream: str) -> tuple[list[dict], list[np.ndarray | None]]:
    """Read back one stream: returns (per-frame records, per-frame images); frames not stored at level 2 are None.
    The decoder is chosen by ``raw_codec`` in ``meta.json``."""
    out_dir = Path(out_dir)
    recs = [json.loads(l) for l in (out_dir / f"frames-{stream}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    mkv = out_dir / f"{stream}.mkv"
    if not recs or not mkv.exists():
        return recs, [None] * len(recs)
    # getting the shape via ffprobe is too heavy and cannot be known from the first frame outside meta; parse the
    # resolution from ffmpeg's banner instead
    probe = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(mkv)], capture_output=True, text=True).stderr
    import re

    m = re.search(r"Video: .*?, (\d+)x(\d+)", probe)
    if not m:
        raise RuntimeError(f"cannot read the resolution of {mkv}")
    w, h = int(m.group(1)), int(m.group(2))
    dec = decode_raw_frames(find_ffmpeg(), mkv, (h, w, 3), raw_codec=raw_codec_of(out_dir))
    imgs = [np.frombuffer(dec[r["enc"]], np.uint8).reshape(h, w, 3) if r["enc"] is not None else None for r in recs]
    return recs, imgs


def verdict_line(res: dict) -> str:
    return (f"RECORDER_VERIFY={res['RECORDER_VERIFY']} frames={res['frames']} decode_mismatch={res['decode_mismatch']} "
            f"dropped={res['dropped']} reordered={res['reordered']} bytes={res['bytes']} level={res['level']}")


def _main(argv: list[str]) -> int:
    """Command line: ``recorder.py load <out_dir> <stream>`` reads back and checks (for debugging). Legacy FFV1
    outputs check per-frame sha256; AV1 is lossy, so it only checks that every record yields an image."""
    if len(argv) == 3 and argv[0] == "load":
        recs, imgs = load_frames(argv[1], argv[2])
        lossy = raw_codec_of(argv[1]) != LEGACY_RAW_CODEC
        bad = sum(1 for r, im in zip(recs, imgs)
                  if im is not None and not lossy and frame_sha256(im) != r["sha256"])
        kept = sum(im is not None for im in imgs)
        print(f"RECORDER_LOAD={'PASS' if bad == 0 else 'FAIL'} frames={len(recs)} kept={kept} mismatch={bad}")
        return 0 if bad == 0 else 1
    print("usage: recorder.py load <out_dir> <stream>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
