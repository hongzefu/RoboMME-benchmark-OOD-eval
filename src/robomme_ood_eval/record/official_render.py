"""Official-layout rendering library (a thin command-line wrapper around this module also exists).

Calls the official ``RolloutRecorder`` offline on existing raw frames and traces; never starts a simulation and never
modifies the source videos. The official ``RolloutRecorder`` is loaded only via ``importlib`` from the pinned original
file ``third_party/mme-vla/examples/robomme/utils.py``, never copied or rewritten; all drawing (text panel, red frame
around the demo segment) belongs to the original class, and this module only encodes the frames it draws into a
site video.

Pipeline: read trace (shared contracts C3 / C4 / C8) -> select source -> decode -> verify -> official recorder draws
each frame -> encode -> re-check.

Three sources:
- ``raw-new``: ``front.mkv`` / ``wrist.mkv`` + ``frames-<stream>.jsonl`` (idx, sha256, tag, enc) written by
  ``recorder.py``, **read per version according to ``raw_codec`` in the episode directory's ``meta.json``**:
  * ``av1-yuv444p`` (since 2026-10-08): lossy; checks the encoder is av1 and that the decoded frame count exactly
    covers the index enc values; **decoded byte sha256 is no longer verified**
    (``frame_hash_check.mode=skipped-lossy-av1``);
  * no ``raw_codec`` (legacy FFV1 output): as before, checks losslessness, that decoded frame byte sha256 equals the
    index, and the trace frame hashes;
- ``raw-orig``: ``frames/{front,wrist}.rgb24`` + ``frames/frames.json`` (original-side observer);
- ``mp4``: ``episode.mp4`` in the episode directory (512x256 side by side, lossy, only the frame count is checked).

Site video: ``libaom-av1``, ``-pix_fmt yuv420p``, 30 fps, MP4 container (``-movflags +faststart``); the size is
rounded up to a multiple of 16 and scaled using the same rule as the official ``save_video`` (imageio,
macro_block_size=16).

Original action values are restored from ``arrays.npz`` (C4); lossy f32hex must never stand in for non-float32
actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np

from robomme_ood_eval.record import trace_writer as _tw
from robomme_ood_eval.record.recorder import AV1_DECODE_FILTER, LEGACY_RAW_CODEC, RAW_CODEC, find_ffmpeg

PKG_ROOT = Path(__file__).resolve().parents[1]
#: eval repo root (three levels above src/robomme_ood_eval/record/); the official RolloutRecorder lives under its
#: third_party/mme-vla
REPO = Path(__file__).resolve().parents[3]
OFFICIAL_UTILS_REL = "third_party/mme-vla/examples/robomme/utils.py"
OFFICIAL_EVAL_REL = "third_party/mme-vla/examples/robomme/eval.py"
#: environment variable that overrides the official repo root (point it at a checkout whose submodules are
#: populated, e.g. when they are empty in a worktree)
ENV_OFFICIAL_ROOT = "ROBOMME_EVAL_OFFICIAL_ROOT"
SCHEMA = "official-rerender/2"
TERMINALS = ("success", "fail", "timeout", "error")
STREAMS = ("front", "wrist")
SOURCE_MODES = ("auto", "mp4", "raw")
RAW_KINDS = ("raw-new", "raw-orig")
FPS = 30
#: site video encoding parameters (AV1 4:2:0 in MP4, faststart)
WEB_ENCODE_ARGS = ("-c:v", "libaom-av1", "-cpu-used", "4", "-crf", "24", "-b:v", "0", "-pix_fmt", "yuv420p",
                   "-row-mt", "1", "-threads", "2", "-r", str(FPS), "-movflags", "+faststart")
MACRO_BLOCK = 16


def default_official_root() -> Path:
    env = os.environ.get(ENV_OFFICIAL_ROOT)
    return Path(env) if env else REPO


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_official(repo_root: Path | None = None):
    """Execute the pinned official utils file directly; returns (RolloutRecorder, TASK_WITH_VIDEO_DEMO)."""
    path = Path(repo_root or default_official_root()) / OFFICIAL_UTILS_REL
    if not path.is_file():
        raise FileNotFoundError(f"official RolloutRecorder not found: {path} (submodule not initialized? you can set "
                                f"{ENV_OFFICIAL_ROOT})")
    mod = _load(path, "official_robomme_utils")
    return mod.RolloutRecorder, mod.TASK_WITH_VIDEO_DEMO


ENV_FFPROBE = "ROBOMME_FFPROBE"


def find_ffprobe(ffmpeg: str | None = None) -> str:
    """ffprobe: the ROBOMME_FFPROBE environment variable first; then the one next to ffmpeg (if ffmpeg is not given,
    it is located with the recorder's find_ffmpeg rules, honoring V75_FFMPEG); finally PATH. Some compute nodes have
    no ffprobe on PATH and imageio only bundles ffmpeg, so it must be found via the first two rules."""
    env = os.environ.get(ENV_FFPROBE)
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    if not ffmpeg:
        try:
            from .recorder import find_ffmpeg
            ffmpeg = find_ffmpeg()
        except Exception:  # noqa: BLE001 -- fall back to PATH when ffmpeg cannot be found
            ffmpeg = None
    if ffmpeg:
        cand = Path(ffmpeg).with_name("ffprobe")
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    w = shutil.which("ffprobe")
    if not w:
        raise RuntimeError(f"ffprobe not found (set {ENV_FFPROBE}, or point V75_FFMPEG at an ffmpeg with ffprobe in "
                           f"the same directory)")
    return w


def f32_from_record(rec: dict) -> np.ndarray:
    """Only complete, finite 8-d numeric records may enter the official text panel."""
    if not isinstance(rec, dict) or rec.get("shape") != [8]:
        raise ValueError("array shape must be [8]")
    if np.dtype(rec.get("dtype", "O")).kind not in "fiub":
        raise ValueError("array dtype must be numeric")
    raw = bytes.fromhex(rec["f32hex"])
    if len(raw) != 32:
        raise ValueError("f32hex must record all 8 float32 values")
    a = np.frombuffer(raw, dtype="<f4").copy()
    if not np.isfinite(a).all():
        raise ValueError("array contains non-finite values")
    if np.dtype(rec["dtype"]) == np.dtype("<f4") and hashlib.sha256(raw).hexdigest() != rec.get("sha256"):
        raise ValueError("float32 array bytes do not match trace.sha256")
    return a


@dataclass
class TraceData:
    identity: dict
    route: str
    task_goal: str
    demo_frames: int
    init_states: list
    init_subgoal: str | None
    steps: list
    terminal_reason: str
    end_status: str
    max_steps: int
    omitted_timeout_frames: int
    no_frame: bool = False
    no_frame_reason: str | None = None
    demo_front_sha256: list = field(default_factory=list)
    demo_wrist_sha256: list = field(default_factory=list)
    policy_seed: int | None = None
    cap_hit: bool = False

    @property
    def named_terminal(self) -> str:
        """Terminal status used in file names: a strict-cap hit or ``status=timeout`` is always ``timeout``."""
        if self.cap_hit or self.end_status == "timeout":
            return "timeout"
        return self.terminal_reason

    @property
    def observed_steps(self) -> list:
        return [st for st in self.steps if st.get("observed", True)]

    @property
    def missing_steps(self) -> list:
        return [st["step"] for st in self.steps if not st.get("observed", True)]

    @property
    def source_frames(self) -> int:
        return len(self.init_states) + len(self.observed_steps)

    @property
    def output_frames(self) -> int:
        return self.source_frames - self.omitted_timeout_frames

    def frame_hashes(self, stream: str) -> list:
        demo = self.demo_front_sha256 if stream == "front" else self.demo_wrist_sha256
        return list(demo) + [st[f"{stream}_sha256"] for st in self.observed_steps]


def _check_counts(end: dict, *, attempted: int, observed: int, frames: int, omitted: int) -> None:
    for key, expect in (("steps_attempted", attempted), ("steps_observed", observed),
                        ("frames_recorded", frames), ("omitted_timeout_frames", omitted)):
        if key in end and end[key] != expect:
            raise ValueError(f"C8 end.{key}={end[key]!r} does not match the value derived from the trace {expect}")


def load_trace(path: Path, arrays_path: Path | None = None) -> TraceData:
    """Check the sequence and required fields first, then decode; demo.frames includes the initial frame."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("trace is empty or has a line that is not an object")
    if any(r.get("kind") not in ("header", "demo", "step", "request", "response", "history", "end") for r in rows):
        raise ValueError("trace contains an unknown line kind")
    problems = _tw.validate_trace(rows)
    if problems:
        raise ValueError("; ".join(problems))
    header, end = rows[0], rows[-1]
    demo = next(r for r in rows if r["kind"] == "demo")
    identity = header["identity"]
    if not isinstance(identity, dict) or not all(identity.get(k) is not None for k in ("task", "tier", "seed", "dataset")):
        raise ValueError("trace identity is missing task/tier/seed/dataset")
    route = header["route"]
    if not isinstance(route, str) or route.rsplit("/", 1)[-1] not in ("new", "orig"):
        raise ValueError("route must be explicitly new or orig")
    terminal = end["terminal_reason"]
    end_status = end.get("status")
    policy_seed = header.get("policy_seed", identity.get("policy_seed"))
    if policy_seed is not None and (type(policy_seed) is not int or policy_seed < 0):
        raise ValueError(f"policy_seed must be a non-negative integer: {policy_seed!r}")
    if (identity.get("policy_seed") is not None and header.get("policy_seed") is not None
            and identity["policy_seed"] != header["policy_seed"]):
        raise ValueError("trace header.policy_seed does not match identity.policy_seed")
    cap_hit = end.get("cap_hit") is True
    if (end_status not in TERMINALS or terminal not in TERMINALS or
            (terminal != end_status and (terminal, end_status) != ("error", "timeout"))):
        raise ValueError("abnormal or conflicting terminal status is not supported")
    if cap_hit and end_status not in ("timeout", "error"):
        raise ValueError(f"end.cap_hit=true but status={end_status} (a strict-cap hit can only be timeout)")
    max_steps = header["max_steps"]
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    raw_steps = [r for r in rows if r["kind"] == "step"]
    texts = demo.get("texts")
    if end.get("no_frame") is True:
        if end_status != "error":
            raise ValueError("C3 only error episodes may be no_frame")
        if demo.get("frames") or any(st.get("observed") is not False for st in raw_steps):
            raise ValueError("C3 a no_frame episode must have no frames at all")
        _check_counts(end, attempted=len(raw_steps), observed=0, frames=0, omitted=0)
        goal = texts[0] if isinstance(texts, list) and texts and isinstance(texts[0], str) else ""
        reason = end.get("no_frame_reason") or end.get("error") or "no_frame"
        steps = [{**st, "observed": False} for st in raw_steps]
        return TraceData(identity, route, goal, int(end.get("demo_frames") or 0), [], None, steps, terminal,
                         end_status, max_steps, 0, no_frame=True, no_frame_reason=str(reason),
                         policy_seed=policy_seed, cap_hit=cap_hit)
    frames, demo_frames = demo["frames"], end["demo_frames"]
    if type(frames) is not int or type(demo_frames) is not int or demo_frames < 0 or frames != demo_frames + 1:
        raise ValueError("demo.frames must equal end.demo_frames + 1")
    if any(not isinstance(demo.get(k), list) or len(demo[k]) != frames for k in ("states", "front_sha256", "wrist_sha256")):
        raise ValueError("initial states and frame hashes are incomplete")
    if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str) or not texts[0].strip():
        raise ValueError("demo.texts must contain exactly one non-empty task goal")
    init = [f32_from_record(s) for s in demo["states"]]
    if any(np.dtype(s["dtype"]) != np.dtype("<f4") for s in demo["states"]):
        raise ValueError("initial state is not float32; f32hex cannot prove the full original array")
    steps = []
    arrays_path = arrays_path or Path(path).with_name("arrays.npz")
    arrays = np.load(arrays_path, allow_pickle=False) if arrays_path.exists() else None
    try:
        for st in raw_steps:
            if type(st["step"]) is not int:
                raise ValueError("step must be an integer")
            if not isinstance(st.get("subgoal"), (str, type(None))):
                raise ValueError("subgoal must be text or None")
            observed = st.get("observed", True)
            if observed is not True and observed is not False:
                raise ValueError("observed must be a boolean")
            if observed:
                if not st.get("front_sha256") or not st.get("wrist_sha256"):
                    raise ValueError("executed step has incomplete frames and cannot be paired with the video (a step "
                                     "without observation must set observed=false)")
                if not isinstance(st.get("state"), dict) or np.dtype(st["state"]["dtype"]) != np.dtype("<f4"):
                    raise ValueError("executed state is not float32; f32hex cannot prove the full original array")
            elif not st.get("missing_reason"):
                raise ValueError(f"C8 step {st['step']} has no observation but no reason was written")
            action_rec = st.get("action")
            if not isinstance(action_rec, dict):
                raise ValueError(f"C4 step {st['step']} has no action")
            action = f32_from_record(action_rec)
            key = f"exec_action__{st['step'] - 1:05d}"
            if arrays is not None:
                if key not in arrays.files:
                    raise ValueError(f"C4 arrays.npz is missing {key}")
                action = arrays[key]
                if (list(action.shape) != action_rec["shape"] or action.dtype.str != action_rec["dtype"] or
                        hashlib.sha256(action.tobytes()).hexdigest() != action_rec["sha256"] or not np.isfinite(action).all()):
                    raise ValueError(f"original action {key} does not match trace dtype/shape/sha256")
            elif np.dtype(action_rec["dtype"]) != np.dtype("<f4"):
                raise ValueError("original actions arrays.npz missing; refusing to lose original array precision via "
                                 "f32hex")
            state = f32_from_record(st["state"]) if observed else None
            steps.append({**st, "observed": observed, "state": state, "action": action})
    finally:
        if arrays is not None:
            arrays.close()
    if len(steps) > max_steps + 1 or (len(steps) > max_steps and end_status != "timeout"):
        raise ValueError("executed steps exceed the official cap and are not a single final timeout step")
    omitted = int(terminal == "timeout" and end_status == "timeout" and len(steps) == max_steps + 1
                  and steps[-1]["observed"])
    observed_n = sum(st["observed"] for st in steps)
    _check_counts(end, attempted=len(steps), observed=observed_n, frames=frames + observed_n - omitted, omitted=omitted)
    subgoal = "[initializing...]" if any(st["subgoal"] is not None for st in steps) or "ground-sg" in route else None
    return TraceData(identity, route, texts[0], demo_frames, init, subgoal, steps, terminal, end_status, max_steps,
                     omitted, demo_front_sha256=list(demo["front_sha256"]), demo_wrist_sha256=list(demo["wrist_sha256"]),
                     policy_seed=policy_seed, cap_hit=cap_hit)


def feed_official_recorder(rec, front, wrist, trace: TraceData, task: str, video_demo_tasks,
                           max_output_bytes: int | None = None) -> None:
    """Same order as the official init_episode / eval_each_episode; all drawing belongs to the original class."""
    if len(front) != trace.source_frames or len(wrist) != trace.source_frames:
        raise ValueError(f"frame_count: input {len(front)}/{len(wrist)}, trace {trace.source_frames}")
    front, wrist = np.asarray(front), np.asarray(wrist)
    if front.dtype != np.uint8 or wrist.dtype != np.uint8 or front.ndim != 4 or front.shape != wrist.shape or front.shape[-1] != 3:
        raise ValueError("front/wrist must be uint8 arrays of the same N x H x W x 3 shape")
    n_init = len(trace.init_states)
    recorded_bytes = 0

    def record(**kwargs):
        nonlocal recorded_bytes
        rec.record(**kwargs)
        recorded_bytes += rec.total_images[-1].nbytes
        if max_output_bytes is not None and recorded_bytes > max_output_bytes:
            raise ValueError("actual frame memory of the official text panel exceeds this episode's budget")

    for i, state in enumerate(trace.init_states):
        record(image=front[i].copy(), wrist_image=wrist[i].copy(), state=state,
               is_video_demo=task in video_demo_tasks and i < n_init - 1, subgoal=trace.init_subgoal)
    observed = trace.observed_steps
    for k, st in enumerate(observed[:len(observed) - trace.omitted_timeout_frames]):
        subgoal = st["subgoal"] if st["subgoal"] is not None else trace.init_subgoal
        record(image=front[n_init + k].copy(), wrist_image=wrist[n_init + k].copy(),
               state=st["state"], action=st["action"], subgoal=subgoal)


def fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": Path(path).stat().st_size}


def probe_video(ffprobe: str, path: Path) -> dict:
    proc = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                           "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
                           "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(proc.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("video must have exactly one video stream")
    s = streams[0]
    return {"width": int(s["width"]), "height": int(s["height"]), "frames": int(s["nb_read_frames"]),
            "fps": str(Fraction(s["avg_frame_rate"])), "codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt")}


def probe_stream(ffprobe: str, path: Path) -> dict:
    """Codec, pixel format, size and fully decoded frame count of a raw stream."""
    proc = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries",
                           "stream=codec_type,codec_name,pix_fmt,width,height,nb_read_frames", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"raw stream unreadable: {path.name}: {proc.stderr.strip()[:200]}")
    streams = [s for s in json.loads(proc.stdout).get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"raw stream must have exactly one video stream: {path.name}")
    s = streams[0]
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"), "width": int(s["width"]),
            "height": int(s["height"]), "frames": int(s["nb_read_frames"])}


def _ffmpeg_rawvideo(ffmpeg: str, path: Path, n: int, h: int, w: int, vf: str | None = None) -> np.ndarray:
    """Stream-decode to rgb24 into a preallocated array; the frame count must be exactly n. ``vf`` is a color
    restoration filter (used for AV1)."""
    images = np.empty((n, h, w, 3), dtype=np.uint8)
    conv = ["-vf", vf] if vf else []
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen([ffmpeg, "-v", "error", "-threads", "1", "-i", str(path), *conv, "-f", "rawvideo",
                                 "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"], stdout=subprocess.PIPE, stderr=err)
        try:
            for arr in images:
                view = memoryview(arr).cast("B")
                offset = 0
                while offset < len(view):
                    got = proc.stdout.readinto(view[offset:])
                    if not got:
                        raise ValueError("decoded frames ended early")
                    offset += got
            if proc.stdout.read(1):
                raise ValueError("decoded frame count exceeds ffprobe")
            code = proc.wait()
            if code:
                err.seek(0)
                raise ValueError(f"ffmpeg exited {code}: {err.read().decode(errors='replace')}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
    return images


def decode_mp4(ffmpeg: str, mp4: Path, info: dict | None = None):
    info = info or probe_video(find_ffprobe(ffmpeg), mp4)
    if (info["width"], info["height"]) != (512, 256):
        raise ValueError("source video must be 512x256")
    images = _ffmpeg_rawvideo(ffmpeg, mp4, info["frames"], 256, 512)
    return images[:, :, :256], images[:, :, 256:]


def safe_filename(full_name: str) -> str:
    """Write the full official semantics to the sidecar; append a fixed digest when the name is invalid for the
    file system or too long."""
    sanitized = re.sub(r"[/\\\x00]", "_", full_name)
    if sanitized == full_name and len(sanitized.encode()) <= 255:
        return sanitized
    suffix = "__" + hashlib.sha256(full_name.encode()).hexdigest()[:16] + ".mp4"
    stem = sanitized.removesuffix(".mp4").encode()[:255 - len(suffix.encode())].decode(errors="ignore")
    return stem + suffix


def official_episode_id(identity: dict, episode_tag: str) -> str:
    """Short episode id ``<source_episode>a<attempt>`` for a legacy episode directory (``<key>.a<N>``), same form as
    ``models/groundsg.py``."""
    src = identity.get("source_episode")
    if src is None:
        src = identity.get("builder_episode")
    att = episode_tag.rsplit(".a", 1)[1] if ".a" in episode_tag else "1"
    return f"{src}a{att}"


def _episode_tag(trace: TraceData, ep_dir: Path, key: str | None = None) -> str:
    """Legacy episode directory tag ``<key>.a<N>``: the directory name by default; when the name does not contain
    the key, ``key`` gives it explicitly (C6)."""
    identity = trace.identity
    name = ep_dir.name
    if key is not None:
        if not key or any(c in key for c in "/\\\x00"):
            raise ValueError("--key is invalid")
        if identity.get("key") is not None and str(identity["key"]) != key:
            raise ValueError(f"--key={key} does not match trace.identity.key={identity['key']}")
        name = f"{key}.a{int(identity.get('attempt') or 1)}"
    if identity.get("key") and not re.fullmatch(re.escape(str(identity["key"])) + r"\.a[1-9]\d*", name):
        raise ValueError("episode directory does not match trace.identity.key")
    m = re.fullmatch(r".+\.a([1-9]\d*)", name)
    if identity.get("attempt") is not None and m and int(m[1]) != int(identity["attempt"]):
        raise ValueError(f"episode directory attempt a{m[1]} does not match trace.identity.attempt={identity['attempt']}")
    return name


def _episode_id(trace: TraceData, ep_dir: Path, key: str | None = None) -> str:
    identity = trace.identity
    tag = _episode_tag(trace, ep_dir, key)
    if trace.route.endswith("/new"):
        if identity.get("source_episode") is None and identity.get("builder_episode") is None:
            raise ValueError("new-side identity has neither source_episode nor builder_episode")
        return official_episode_id(identity, tag)
    if identity.get("source_episode") is None:
        raise ValueError("original side is missing source_episode")
    return str(identity["source_episode"])


# -- source selection and decoding ---------------------------------------------------


@dataclass
class Source:
    """Frame source of one episode; ``streams`` / ``index`` map "path relative to the episode directory -> absolute
    path"; ``raw_codec`` only matters for raw-new."""

    kind: str
    streams: dict
    index: dict
    raw_codec: str | None = None

    def media(self, ep_dir: Path) -> dict:
        return {"streams": {rel: fingerprint(p) for rel, p in self.streams.items()},
                "index": {rel: fingerprint(p) for rel, p in self.index.items()}}


def _rel(ep_dir: Path, p: Path) -> str:
    return p.relative_to(ep_dir).as_posix()


def _raw_candidates(ep_dir: Path) -> list:
    out = []
    for base in (ep_dir, ep_dir / "media"):
        if any((base / f"{s}.mkv").exists() for s in STREAMS):
            out.append(("raw-new", base))
    if any((ep_dir / "frames" / f"{s}.rgb24").exists() for s in STREAMS):
        out.append(("raw-orig", ep_dir / "frames"))
    return out


def _meta_raw_codec(meta_path: Path | None) -> str:
    if meta_path is None or not meta_path.is_file():
        return LEGACY_RAW_CODEC
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        raise ValueError(f"corrupt index: {meta_path.name} is not JSON") from None
    return str(meta.get("raw_codec") or LEGACY_RAW_CODEC)


def select_source(ep_dir: Path, mode: str) -> Source:
    """``auto``: raw if raw streams exist, otherwise mp4; ``raw`` fails when raw streams are missing and never falls
    back to mp4."""
    if mode not in SOURCE_MODES:
        raise ValueError(f"--source must be one of {SOURCE_MODES}")
    cands = _raw_candidates(ep_dir)
    if len(cands) > 1:
        raise ValueError("raw frame source is ambiguous: " + ",".join(f"{k}@{_rel(ep_dir, b) or '.'}" for k, b in cands))
    mp4 = ep_dir / "episode.mp4"
    if mode == "mp4" or (mode == "auto" and not cands):
        if not mp4.is_file():
            raise ValueError("no usable source: episode.mp4 missing" + ("" if mode == "mp4" else ", and no raw frames"))
        return Source("mp4", {_rel(ep_dir, mp4): mp4}, {})
    if not cands:
        raise ValueError("raw mode is missing raw frames (no fallback to mp4)")
    kind, base = cands[0]
    streams, index = {}, {}
    for s in STREAMS:
        if kind == "raw-new":
            stream, idx = base / f"{s}.mkv", base / f"frames-{s}.jsonl"
        else:
            stream, idx = base / f"{s}.rgb24", None
        if not stream.is_file() or stream.is_symlink():
            raise ValueError(f"raw stream missing {_rel(ep_dir, stream)}")
        streams[_rel(ep_dir, stream)] = stream
        if idx is not None:
            if not idx.is_file():
                raise ValueError(f"raw index missing {_rel(ep_dir, idx)}")
            index[_rel(ep_dir, idx)] = idx
    if kind == "raw-new":
        meta = base / "meta.json"
        if meta.is_file():
            index[_rel(ep_dir, meta)] = meta
        return Source(kind, streams, index, raw_codec=_meta_raw_codec(meta if meta.is_file() else None))
    meta = base / "frames.json"
    if not meta.is_file():
        raise ValueError(f"raw index missing {_rel(ep_dir, meta)}")
    index[_rel(ep_dir, meta)] = meta
    return Source(kind, streams, index)


def read_frame_index(path: Path) -> list:
    """Read ``frames-<stream>.jsonl``: idx unique and contiguous 0..n-1, sha256 complete, enc a non-negative integer
    (an empty enc means the degraded level 2)."""
    rows = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            raise ValueError(f"corrupt index: {path.name} line {ln} is not JSON") from None
        if (not isinstance(r, dict) or type(r.get("idx")) is not int or not isinstance(r.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", r["sha256"])):
            raise ValueError(f"corrupt index: {path.name} line {ln} is missing idx/sha256")
        if r.get("enc") is None:
            raise ValueError(f"lossy degradation: {path.name} line {ln} has empty enc (degraded level keeping only first "
                             f"and last frames)")
        if type(r["enc"]) is not int or r["enc"] < 0:
            raise ValueError(f"corrupt index: {path.name} line {ln} has an invalid enc")
        rows.append(r)
    idxs = [r["idx"] for r in rows]
    if len(set(idxs)) != len(idxs):
        raise ValueError(f"duplicate index: {path.name}")
    if sorted(idxs) != list(range(len(idxs))):
        raise ValueError(f"corrupt index: {path.name} idx is not contiguous 0..n-1")
    return sorted(rows, key=lambda r: r["idx"])


def _select_rows(rows: list, trace: TraceData, stream: str) -> list:
    """When the index row count equals the source frame count they map one to one; otherwise select by tag (demo
    segment ``reset``; step k takes the last frame of ``step{k-1}``)."""
    if len(rows) == trace.source_frames:
        return rows
    reset = [r for r in rows if r.get("tag") == "reset"]
    if len(reset) != len(trace.init_states):
        raise ValueError(f"missing frames: {stream} index has {len(rows)} rows, demo segment {len(reset)} frames, trace "
                         f"needs {trace.source_frames} frames")
    by_tag: dict = {}
    for r in rows:
        by_tag[str(r.get("tag"))] = r
    out = list(reset)
    for st in trace.observed_steps:
        r = by_tag.get(f"step{st['step'] - 1}")
        if r is None:
            raise ValueError(f"missing frames: {stream} step {st['step']} has no frame in the index")
        out.append(r)
    return out


def _verify_trace_hashes(trace: TraceData, stream: str, frames: np.ndarray) -> int:
    expect = trace.frame_hashes(stream)
    if len(expect) != len(frames):
        raise ValueError(f"missing frames: {stream} source {len(frames)} frames, trace {len(expect)} frames")
    for i, (img, want) in enumerate(zip(frames, expect)):
        if _tw.image_sha256(img) != want:
            raise ValueError(f"per-frame hash mismatch: {stream} frame {i} differs from the trace record (swapped "
                             f"stream or misalignment)")
    return len(frames)


def decode_raw_new(ffmpeg: str, ep_dir: Path, src: Source, trace: TraceData) -> tuple:
    """Read recorder raw streams per ``raw_codec`` version and expand them back to per-frame images using the index
    enc values. Returns (front, wrist, detail).

    * ``av1-yuv444p``: checks codec av1 / pixel format yuv444p, that the decoded frame count exactly covers the index
      enc values, and that the frame count matches the trace; lossy, so bytes are not checked;
    * ``ffv1`` (legacy output): checks losslessness, that decoded frame byte sha256 equals the index, and the trace
      frame hashes."""
    ffprobe = find_ffprobe(ffmpeg)
    codec = src.raw_codec or LEGACY_RAW_CODEC
    if codec not in (RAW_CODEC, LEGACY_RAW_CODEC):
        raise ValueError(f"unknown raw_codec={codec!r}")
    lossy = codec == RAW_CODEC
    meta_rel = next((r for r in src.index if r.endswith("meta.json")), None)
    if meta_rel is not None and not lossy:
        meta = json.loads(src.index[meta_rel].read_text(encoding="utf-8"))
        if meta.get("level", 0) != 0 or meta.get("codec", "ffv1") != "ffv1":
            raise ValueError(f"lossy degradation: {meta_rel} level={meta.get('level')} codec={meta.get('codec')}")
    if not lossy:
        summary = next(iter(src.streams.values())).parent / "summary.json"
        if summary.is_file():
            try:
                lossless = json.loads(summary.read_text(encoding="utf-8")).get("lossless")
            except ValueError:
                lossless = None
            if lossless is False:
                raise ValueError("lossy degradation: summary.json lossless=false")
    out, checked = {}, {}
    for s in STREAMS:
        stream = next(p for r, p in src.streams.items() if r.endswith(f"{s}.mkv"))
        idx_path = next(p for r, p in src.index.items() if r.endswith(f"frames-{s}.jsonl"))
        rows = read_frame_index(idx_path)
        info = probe_stream(ffprobe, stream)
        if lossy:
            if info["codec"] != "av1" or info["pix_fmt"] != "yuv444p":
                raise ValueError(f"{stream.name} codec {info['codec']}/{info['pix_fmt']} does not match raw_codec={codec}")
        elif info["codec"] != "ffv1":
            raise ValueError(f"lossy degradation: {stream.name} codec {info['codec']} is not ffv1")
        encs = sorted({r["enc"] for r in rows})
        if encs != list(range(info["frames"])):
            raise ValueError(f"corrupt index: {s} stream decodes to {info['frames']} frames, index enc covers {len(encs)}"
                             f" (max {encs[-1] if encs else None}); swapped stream or wrong enc")
        decoded = _ffmpeg_rawvideo(ffmpeg, stream, info["frames"], info["height"], info["width"],
                                   vf=AV1_DECODE_FILTER if lossy else None)
        if not lossy:
            enc_sha = [hashlib.sha256(f.tobytes()).hexdigest() for f in decoded]
            bad = [r["idx"] for r in rows if enc_sha[r["enc"]] != r["sha256"]]
            if bad:
                raise ValueError(f"decoded frames do not match index sha256: {s} idx={bad[:5]} (swapped stream or wrong enc)")
        picked = _select_rows(rows, trace, s)
        frames = decoded[[r["enc"] for r in picked]]
        del decoded
        if lossy:
            if len(frames) != trace.source_frames:
                raise ValueError(f"missing frames: {s} source {len(frames)} frames, trace {trace.source_frames} frames")
            checked[s] = len(frames)
        else:
            checked[s] = _verify_trace_hashes(trace, s, frames)
        out[s] = frames
    if out["front"].shape != out["wrist"].shape:
        raise ValueError("front/wrist sizes differ")
    return out["front"], out["wrist"], {"index_rows": checked, "raw_codec": codec, "lossy": lossy}


def decode_raw_orig(ep_dir: Path, src: Source, trace: TraceData) -> tuple:
    """Slice ``frames/*.rgb24`` into frames using the size and count in ``frames.json``; the frame count must exactly
    equal the trace source frame count."""
    meta_path = next(p for r, p in src.index.items() if r.endswith("frames.json"))
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        raise ValueError("corrupt index: frames/frames.json is not JSON") from None
    if meta.get("pix_fmt", "rgb24") != "rgb24":
        raise ValueError(f"lossy degradation: pix_fmt={meta.get('pix_fmt')}")
    if meta.get("demo_frames") is not None and meta["demo_frames"] != trace.demo_frames:
        raise ValueError(f"frames.json demo_frames={meta['demo_frames']} does not match trace {trace.demo_frames}")
    if meta.get("missing_steps") is not None and list(meta["missing_steps"]) != trace.missing_steps:
        raise ValueError(f"frames.json missing_steps does not match the trace's unobserved steps {trace.missing_steps}")
    out = {}
    for s in STREAMS:
        st = (meta.get("streams") or {}).get(s)
        if not isinstance(st, dict) or not all(type(st.get(k)) is int and st[k] > 0 for k in ("width", "height", "count")):
            raise ValueError(f"corrupt index: frames.json is missing width/height/count for {s}")
        w, h, n = st["width"], st["height"], st["count"]
        if n != trace.source_frames:
            raise ValueError(f"missing frames: {s} count={n}, trace needs {trace.source_frames} frames")
        path = next(p for r, p in src.streams.items() if r.endswith(f"{s}.rgb24"))
        if path.stat().st_size != n * w * h * 3:
            raise ValueError(f"missing frames: {s}.rgb24 byte count {path.stat().st_size} != {n}x{w}x{h}x3")
        out[s] = np.fromfile(path, dtype=np.uint8).reshape(n, h, w, 3)
    if out["front"].shape != out["wrist"].shape:
        raise ValueError("front/wrist sizes differ")
    checked = {s: _verify_trace_hashes(trace, s, out[s]) for s in STREAMS}
    return out["front"], out["wrist"], {"index_rows": checked}


def _source_frame_bytes(ffmpeg: str, src: Source) -> int:
    if src.kind == "mp4":
        return 512 * 256 * 3
    if src.kind == "raw-orig":
        meta = json.loads(next(p for r, p in src.index.items() if r.endswith("frames.json")).read_text(encoding="utf-8"))
        return sum(int(meta["streams"][s]["width"]) * int(meta["streams"][s]["height"]) * 3 for s in STREAMS)
    ffprobe = find_ffprobe(ffmpeg)
    total = 0
    for p in src.streams.values():
        proc = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                               "-of", "json", str(p)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValueError(f"raw stream unreadable: {p.name}")
        s = json.loads(proc.stdout)["streams"][0]
        total += int(s["width"]) * int(s["height"]) * 3
    return total


# -- site video encoding -------------------------------------------------------------


def web_size(width: int, height: int) -> tuple:
    """Same as imageio (macro_block_size=16): round width and height each up to a multiple of 16."""
    return ((width + MACRO_BLOCK - 1) // MACRO_BLOCK * MACRO_BLOCK,
            (height + MACRO_BLOCK - 1) // MACRO_BLOCK * MACRO_BLOCK)


def encode_web_mp4(frames: list, out: Path, *, ffmpeg: str | None = None, fps: int = FPS) -> None:
    """Per-frame RGB drawn by the official recorder -> site video (AV1 4:2:0 MP4, faststart, 30 fps). When the size
    is not a multiple of 16 it is scaled to the rounded size with ``scale``, like the official ``save_video``."""
    if not frames:
        raise ValueError("no frames to encode")
    h, w = frames[0].shape[:2]
    if any(f.shape != frames[0].shape or f.dtype != np.uint8 for f in frames):
        raise ValueError("per-frame size or dtype is inconsistent")
    ffmpeg = ffmpeg or find_ffmpeg()
    ow, oh = web_size(w, h)
    vf = ["-vf", f"scale={ow}:{oh}"] if (ow, oh) != (w, h) else []
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0", "-an", *vf, *WEB_ENCODE_ARGS, "-f", "mp4", str(out)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        code = proc.wait()
        if code:
            err.seek(0)
            raise RuntimeError(f"site video encoding failed rc={code}: {err.read().decode(errors='replace')[-800:]}")


# -- render one episode --------------------------------------------------------------


def _base_context(trace: TraceData, *, episode_id: str, episode_tag: str, full_name: str | None,
                  safe_name: str | None, out_rel: str | None, source_trace: dict, source_arrays: dict | None,
                  official_root: Path, ffmpeg: str) -> dict:
    import cv2

    observed_n = len(trace.observed_steps)
    official_eval = official_root / OFFICIAL_EVAL_REL
    return {"schema": SCHEMA, "identity": trace.identity, "episode_id": episode_id, "episode_tag": episode_tag,
            "route": trace.route, "task_goal": trace.task_goal, "terminal_reason": trace.named_terminal,
            "trace_terminal_reason": trace.terminal_reason, "cap_hit": trace.cap_hit, "policy_seed": trace.policy_seed,
            "status": trace.end_status, "no_frame": trace.no_frame,
            "source_frames": trace.source_frames, "frames": trace.output_frames, "demo_frames": trace.demo_frames,
            "exec_steps": len(trace.steps), "steps_attempted": len(trace.steps), "steps_observed": observed_n,
            "missing_steps": trace.missing_steps, "frames_recorded": trace.output_frames,
            "recorded_exec_steps": observed_n - trace.omitted_timeout_frames,
            "omitted_timeout_frames": trace.omitted_timeout_frames, "full_name": full_name, "safe_name": safe_name,
            "out_rel": out_rel, "source_trace": source_trace, "source_arrays": source_arrays,
            "tool": fingerprint(Path(__file__)),
            "official": fingerprint(official_root / OFFICIAL_UTILS_REL),
            "official_eval": fingerprint(official_eval) if official_eval.is_file() else None,
            "trace_reader": fingerprint(Path(_tw.__file__)), "ffmpeg": fingerprint(Path(ffmpeg)),
            "web_encode_args": list(WEB_ENCODE_ARGS),
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__}}


def _safe_rel(rel: str) -> bool:
    p = Path(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts


def _source_still_valid(old: dict, ep_dir: Path, requested: str) -> bool:
    kind = old.get("source_kind") or ("mp4" if old.get("source_mp4") else None)
    if requested == "mp4" and kind != "mp4" or requested == "raw" and kind not in RAW_KINDS:
        return False
    media = old.get("source_media")
    if media is None and kind == "mp4":
        media = {"streams": {"episode.mp4": old.get("source_mp4")}, "index": {}}
    if not isinstance(media, dict) or not isinstance(media.get("streams"), dict) or not isinstance(media.get("index"), dict):
        return False
    for rel, fp in media["index"].items():
        p = ep_dir / rel
        if not _safe_rel(rel) or not p.is_file() or fingerprint(p) != fp:
            return False
    for rel, fp in media["streams"].items():
        p = ep_dir / rel
        if not _safe_rel(rel):
            return False
        if p.exists():
            if fingerprint(p) != fp:
                return False
        elif kind not in RAW_KINDS:
            return False
    return True


def _try_reuse(sidecar: Path, output: Path, base: dict, ep_dir: Path, requested: str, ffprobe: str,
               trace: TraceData) -> dict | None:
    try:
        old = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    except (ValueError, OSError):
        return None
    if not all(old.get(k) == v for k, v in base.items()):
        return None
    if trace.no_frame:
        if old.get("render_status") in ("no_frame",) and not output.exists():
            return {**old, "render_status": "no_frame", "reused": True, "dir": ep_dir.name, "out": None}
        return None
    if not output.is_file() or not _source_still_valid(old, ep_dir, requested):
        return None
    actual = probe_video(ffprobe, output)
    if (actual["frames"] == trace.output_frames and actual["fps"] == "30" and
            all(old.get(k) == v for k, v in actual.items()) and old.get("output_fingerprint") == fingerprint(output)):
        return {**old, "render_status": "reused", "dir": ep_dir.name, "out": str(output)}
    return None


def _write_sidecar(out_dir: Path, sidecar: Path, result: dict) -> None:
    side_temp = out_dir / f".render-{uuid.uuid4().hex}.json"
    side_temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(side_temp, sidecar)


def render_episode(ep_dir: Path, *, official_root: Path | None = None, ffmpeg: str | None = None,
                   out_subdir: str = "official", prefix: str = "official-rerender__", overwrite: bool = False,
                   max_memory_mib: int = 6144, source: str = "auto", key: str | None = None,
                   output: Path | None = None, sidecar: Path | None = None, episode_id: str | int | None = None,
                   terminal: str | None = None) -> dict:
    """Atomically produce the official-layout video and the identity manifest ``render.json`` for each episode; an
    old result is reused only when the whole-pipeline fingerprint matches.

    Default (legacy CLI behavior): the video goes to
    ``<ep_dir>/<out_subdir>/<prefix><Task>_ep<id>_<terminal>_<goal>_<tier>.mp4``, with the episode id derived from
    the legacy episode directory name ``<key>.a<N>``. The new output tree is produced via ``render_video``, which
    passes ``output`` (target video path), ``sidecar``, ``episode_id`` and ``terminal`` explicitly (the result row's
    terminal status, overriding the one derived from the trace) and no longer requires the directory to be named
    ``<key>.a<N>``. Frameless error episodes only write the sidecar (``render_status=no_frame`` and the reason), no
    video."""
    ep_dir = Path(ep_dir).resolve()
    official_root = Path(official_root or default_official_root()).resolve()
    if Path(out_subdir).name != out_subdir or out_subdir in ("", ".", "..") or any(c in prefix for c in "/\\\x00"):
        raise ValueError("output subdirectory or prefix is unsafe")
    if source not in SOURCE_MODES:
        raise ValueError(f"--source must be one of {SOURCE_MODES}")
    if terminal is not None and terminal not in ("success", "fail", "timeout"):
        raise ValueError(f"terminal must be success / fail / timeout: {terminal!r}")
    trace_path, arrays_path = ep_dir / "trace.jsonl", ep_dir / "arrays.npz"
    source_trace = fingerprint(trace_path)
    source_arrays = fingerprint(arrays_path) if arrays_path.exists() else None
    trace = load_trace(trace_path)
    if episode_id is None:
        episode_tag = _episode_tag(trace, ep_dir, key)
        episode_id = _episode_id(trace, ep_dir, key)
    else:
        episode_tag, episode_id = ep_dir.name, str(episode_id)
    named = terminal or trace.named_terminal
    if output is None:
        out_dir = ep_dir / out_subdir
    else:
        output = Path(output).resolve()
        out_dir = output.parent
    if out_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if trace.no_frame:
        full_name = safe_name = None
        out_path = out_dir / ".no-frame-placeholder.mp4"
    else:
        full_name = (f"{'' if output is not None else prefix}{trace.identity['task']}_ep{episode_id}_{named}_"
                     f"{trace.task_goal}_{trace.identity['tier']}.mp4")
        safe_name = output.name if output is not None else safe_filename(full_name)
        out_path = output if output is not None else out_dir / safe_name
    sidecar = Path(sidecar) if sidecar is not None else out_dir / "render.json"
    if out_path.is_symlink() or sidecar.is_symlink():
        raise ValueError("output video and manifest must not be symlinks")
    ffmpeg = str(Path(ffmpeg or find_ffmpeg()).resolve())
    ffprobe = find_ffprobe(ffmpeg)
    out_rel = None if safe_name is None else (f"{out_subdir}/{safe_name}" if output is None else safe_name)
    base = _base_context(trace, episode_id=str(episode_id), episode_tag=episode_tag, full_name=full_name,
                         safe_name=safe_name, out_rel=out_rel, source_trace=source_trace,
                         source_arrays=source_arrays, official_root=official_root, ffmpeg=ffmpeg)
    base["terminal_reason"] = named
    if out_path.exists() or sidecar.exists():
        reused = _try_reuse(sidecar, out_path, base, ep_dir, source, ffprobe, trace)
        if reused is not None:
            return reused
        if not overwrite:
            raise ValueError("existing output failed provenance and full-video verification; refusing to overwrite, "
                             "pass overwrite explicitly")
    if trace.no_frame:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {**base, "source_kind": "none", "source_media": None, "source_mp4": None, "dir": ep_dir.name,
                  "render_status": "no_frame", "no_frame_reason": trace.no_frame_reason, "out": None}
        _write_sidecar(sidecar.parent, sidecar, result)
        return result
    src = select_source(ep_dir, source)
    media = src.media(ep_dir)
    context = {**base, "source_kind": src.kind, "source_media": media, "raw_codec": src.raw_codec,
               "source_mp4": media["streams"].get("episode.mp4") if src.kind == "mp4" else None}
    info = None
    if src.kind == "mp4":
        info = probe_video(ffprobe, src.streams["episode.mp4"])
        if {k: info[k] for k in ("width", "height", "frames", "fps")} != {
                "width": 512, "height": 256, "frames": trace.source_frames, "fps": "30"}:
            raise ValueError(f"source video frames/fps/size mismatch: {info}; expected frames={trace.source_frames} fps=30 size=512x256")
    frame_src_bytes = _source_frame_bytes(ffmpeg, src)
    src_bytes = trace.source_frames * frame_src_bytes * (2 if src.kind == "raw-new" else 1)
    Recorder, demo_tasks = load_official(official_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".official-rec-", dir=out_dir))
    try:
        rec = Recorder(scratch, trace.task_goal, fps=FPS)
        dummy = np.zeros((256, 256, 3), np.uint8)
        rec.record(dummy, dummy, trace.init_states[0], subgoal=trace.init_subgoal)
        frame_bytes = rec.total_images[0].nbytes
        estimated = int((src_bytes + trace.output_frames * frame_bytes) * 1.35 + 64 * 1024**2)
        rec.total_images.clear()
        if estimated > max_memory_mib * 1024**2:
            raise ValueError(f"estimated memory {estimated / 1024**2:.0f} MiB exceeds the limit {max_memory_mib} MiB")
        if src.kind == "mp4":
            front, wrist = decode_mp4(ffmpeg, src.streams["episode.mp4"], info)
            hash_check = {"mode": "skipped-lossy-mp4"}
        elif src.kind == "raw-new":
            front, wrist, detail = decode_raw_new(ffmpeg, ep_dir, src, trace)
            hash_check = ({"mode": "skipped-lossy-av1", "frames": detail["index_rows"]} if detail["lossy"]
                          else {"mode": "verified", "frames": detail["index_rows"], "mismatch": 0})
        else:
            front, wrist, detail = decode_raw_orig(ep_dir, src, trace)
            hash_check = {"mode": "verified", "frames": detail["index_rows"], "mismatch": 0}
        output_budget = int((max_memory_mib * 1024**2 - 64 * 1024**2) / 1.35 - src_bytes)
        feed_official_recorder(rec, front, wrist, trace, trace.identity["task"], demo_tasks, output_budget)
        del front, wrist
        if len(rec.total_images) != trace.output_frames or len({a.shape for a in rec.total_images}) != 1:
            raise ValueError("official recorder frame count or per-frame size is inconsistent")
        temp = out_dir / f".render-{uuid.uuid4().hex}.mp4"
        try:
            encode_web_mp4(rec.total_images, temp, ffmpeg=ffmpeg)
            actual = probe_video(ffprobe, temp)
            shape = rec.total_images[0].shape
            expected_size = web_size(shape[1], shape[0])
            if (actual["frames"] != trace.output_frames or actual["fps"] != "30"
                    or (actual["width"], actual["height"]) != expected_size
                    or actual["codec"] != "av1" or actual["pix_fmt"] != "yuv420p"):
                raise ValueError(f"output integrity mismatch: {actual}")
            if (fingerprint(trace_path) != source_trace or
                    (fingerprint(arrays_path) if arrays_path.exists() else None) != source_arrays or
                    src.media(ep_dir) != media):
                raise ValueError("source files changed during re-rendering")
            result = {**context, **actual, "dir": ep_dir.name, "render_status": "rendered", "out": str(out_path),
                      "estimated_memory_mib": round(estimated / 1024**2), "output_fingerprint": fingerprint(temp),
                      "input_array_precision": "verified-original", "frame_hash_check": hash_check}
            os.replace(temp, out_path)
            _write_sidecar(sidecar.parent, sidecar, result)
            return result
        finally:
            temp.unlink(missing_ok=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def render_video(raw_dir: Path, videos_dir: Path, *, episode_id: int | str, terminal: str,
                 official_root: Path | None = None, ffmpeg: str | None = None) -> Path | None:
    """For the new output tree: render ``raw/<Task>_ep<N>_<tier>/`` into
    ``videos/<Task>_ep<N>_<success|fail|timeout>_<task_goal>_<tier>.mp4``, with the manifest written to
    ``render.json`` in the episode directory; an existing video with the same name is always overwritten (outputs
    follow this run's result). Returns None for frameless episodes."""
    raw_dir = Path(raw_dir)
    trace = load_trace(raw_dir / "trace.jsonl")
    if trace.no_frame:
        render_episode(raw_dir, official_root=official_root, ffmpeg=ffmpeg, overwrite=True,
                       sidecar=raw_dir / "render.json", output=Path(videos_dir) / ".no-frame.mp4",
                       episode_id=episode_id, terminal=terminal)
        return None
    name = safe_filename(f"{trace.identity['task']}_ep{episode_id}_{terminal}_{trace.task_goal}_"
                         f"{trace.identity['tier']}.mp4")
    out = Path(videos_dir) / name
    render_episode(raw_dir, official_root=official_root, ffmpeg=ffmpeg, overwrite=True, source="raw",
                   sidecar=raw_dir / "render.json", output=out, episode_id=episode_id, terminal=terminal)
    return out
