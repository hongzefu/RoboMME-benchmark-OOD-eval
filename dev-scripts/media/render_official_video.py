#!/usr/bin/env python3
"""从已有视频与轨迹离线调用官方录像器；不启动仿真，不修改源视频。

逐帧一致的对象是同一组解码画面与经 trace 指纹核对的原数组。
float64 动作从 arrays.npz 恢复，禁止用有舍入损失的 f32hex 替代。

第二阶段（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节 S2a）重构为
「来源选择 → 解码 → 核验 → 录像 → 复核」：

- 来源三种（``--source {auto,mp4,raw}``）：
  ``mp4``——局目录 ``episode.mp4``（512×256 左右拼接，有损，只核帧数，不核逐帧哈希）；
  ``raw-new``——``recorder.py`` 写的 ``front.mkv``／``wrist.mkv``（FFV1）+ ``frames-<stream>.jsonl``
  （idx、sha256、tag、enc；同一流重复帧只编码一份，按 ``enc`` 展开回逐帧原图），在局目录或其 ``media/`` 下；
  ``raw-orig``——``frames/{front,wrist}.rgb24`` + ``frames/frames.json``（rgb24、各流 width/height/count）。
  展开规则与 ``run_seat.sh::transcode_episode_dir`` 文档串一致。``auto`` 有原始流（``*.mkv`` 或
  ``frames/*.rgb24``）即走 raw，否则走 mp4；``raw`` 模式下缺流、索引损坏、重复索引、有损降级一律失败，
  不退回 mp4。
- raw 来源逐帧核验：raw-new 先核解码画面字节 sha256 等于索引 sha256（调包流、错 enc 在此暴露），
  两种 raw 再核选出的逐帧画面 ``trace_writer.image_sha256`` 等于 trace 记录的画面哈希。
- trace 按共享契约 C3（放行 ``status=error``；``end.no_frame=true`` 的无帧 error 局返回 ``no_frame``、
  ``render.json`` 记原因、不出视频）、C4（动作按原 dtype 核对 ``arrays.npz``）、C8（帧数由观测步推导并与
  ``end.frames_recorded`` 对账；``observed=false`` 的缺观测步不补帧）读取。
- 1006 第三阶段：文件名与 ``render.json`` 的 ``terminal_reason`` 用「命名终态」——strict-cap 命中（``end.cap_hit``）或
  ``end.status=timeout`` 一律 ``timeout``（旧口径 ``terminal_reason=error`` 的超时局不再出现 ``error`` 命名），trace 原值另记
  ``trace_terminal_reason``；``render.json`` 记 ``policy_seed``（trace header／identity，旧轨迹为 null）与 ``cap_hit``。
- 每路流与索引文件的 sha256 以局目录相对路径进 ``render.json`` 与复用上下文；raw 流在转码后被删时，
  索引仍一致即可复用（目录搬到 NFS 后仍可核），流文件若仍在则必须逐字节一致。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "official-rerender/1"
TERMINALS = ("success", "fail", "timeout", "error")
STREAMS = ("front", "wrist")
SOURCE_MODES = ("auto", "mp4", "raw")
RAW_KINDS = ("raw-new", "raw-orig")
# 复用判定时单独核对的来源字段（其余上下文字段必须逐项相等）
SOURCE_KEYS = ("source_kind", "source_media", "source_mp4")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"不能加载模块：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _trace_writer():
    return _load(REPO / "scripts/eval-official/trace_writer.py", "rerender_trace_writer")


def load_official(repo_root: Path):
    """直接执行锁定官方 utils 原文件；返回原类与原任务清单。"""
    path = Path(repo_root) / "third_party/mme-vla/examples/robomme/utils.py"
    mod = _load(path, "official_robomme_utils")
    return mod.RolloutRecorder, mod.TASK_WITH_VIDEO_DEMO


def f32_from_record(rec: dict) -> np.ndarray:
    """只有完整、有限的 8 维数值记录才能进入官方文字区。"""
    if not isinstance(rec, dict) or rec.get("shape") != [8]:
        raise ValueError("数组 shape 必须为 [8]")
    if np.dtype(rec.get("dtype", "O")).kind not in "fiub":
        raise ValueError("数组 dtype 必须为数值类型")
    raw = bytes.fromhex(rec["f32hex"])
    if len(raw) != 32:
        raise ValueError("f32hex 必须完整记录 8 个 float32")
    a = np.frombuffer(raw, dtype="<f4").copy()
    if not np.isfinite(a).all():
        raise ValueError("数组包含非有限值")
    if np.dtype(rec["dtype"]) == np.dtype("<f4") and hashlib.sha256(raw).hexdigest() != rec.get("sha256"):
        raise ValueError("float32 数组字节与 trace.sha256 不符")
    return a


@dataclass
class TraceData:
    identity: dict
    route: str
    task_goal: str
    demo_frames: int
    init_states: list[np.ndarray]
    init_subgoal: str | None
    steps: list[dict]
    terminal_reason: str
    end_status: str
    max_steps: int
    omitted_timeout_frames: int
    # C3：无帧 error 局
    no_frame: bool = False
    no_frame_reason: str | None = None
    # raw 来源逐帧核验用：演示段（含初始帧）的 front／wrist 画面哈希
    demo_front_sha256: list = field(default_factory=list)
    demo_wrist_sha256: list = field(default_factory=list)
    # 1006 第三阶段：模型种子（trace header／identity 的 policy_seed，旧轨迹没有为 None）与 strict-cap 命中
    policy_seed: int | None = None
    cap_hit: bool = False

    @property
    def named_terminal(self) -> str:
        """文件名用的终态：strict-cap 命中（``end.cap_hit``）或 ``status=timeout`` 一律 ``timeout``
        （旧口径 ``terminal_reason=error``／``status=timeout`` 的超时局本版不再命名 ``error``）。"""
        if self.cap_hit or self.end_status == "timeout":
            return "timeout"
        return self.terminal_reason

    @property
    def observed_steps(self) -> list[dict]:
        """C8：返回有效观测的执行步（缺观测步 ``observed=false`` 不进视频、不补帧）。"""
        return [st for st in self.steps if st.get("observed", True)]

    @property
    def missing_steps(self) -> list[int]:
        return [st["step"] for st in self.steps if not st.get("observed", True)]

    @property
    def source_frames(self) -> int:
        return len(self.init_states) + len(self.observed_steps)

    @property
    def output_frames(self) -> int:
        return self.source_frames - self.omitted_timeout_frames

    def frame_hashes(self, stream: str) -> list:
        """来源逐帧（演示段 + 观测步）应有的画面哈希，顺序与原始帧一致。"""
        demo = self.demo_front_sha256 if stream == "front" else self.demo_wrist_sha256
        return list(demo) + [st[f"{stream}_sha256"] for st in self.observed_steps]


def _check_counts(end: dict, *, attempted: int, observed: int, frames: int, omitted: int) -> None:
    """C8：``end`` 里写了的三分计数必须与 trace 推导值一致（旧轨迹不写则不核）。"""
    for key, expect in (("steps_attempted", attempted), ("steps_observed", observed),
                        ("frames_recorded", frames), ("omitted_timeout_frames", omitted)):
        if key in end and end[key] != expect:
            raise ValueError(f"C8 end.{key}={end[key]!r} 与 trace 推导值 {expect} 不符")


def load_trace(path: Path, arrays_path: Path | None = None) -> TraceData:
    """先核对序列与必需字段，再解码；demo.frames 包括初始帧。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("轨迹为空或行不是对象")
    if any(r.get("kind") not in ("header", "demo", "step", "request", "response", "history", "end") for r in rows):
        raise ValueError("轨迹包含未知行类型")
    writer = _trace_writer()
    problems = writer.validate_trace(rows)
    if problems:
        raise ValueError("；".join(problems))
    header, end = rows[0], rows[-1]
    demo = next(r for r in rows if r["kind"] == "demo")
    identity = header["identity"]
    if not isinstance(identity, dict) or not all(identity.get(k) is not None for k in ("task", "tier", "seed", "dataset")):
        raise ValueError("轨迹身份缺少 task/tier/seed/dataset")
    route = header["route"]
    if not isinstance(route, str) or route.rsplit("/", 1)[-1] not in ("new", "orig"):
        raise ValueError("route 必须明确 new 或 orig")
    terminal = end["terminal_reason"]
    end_status = end.get("status")
    policy_seed = header.get("policy_seed", identity.get("policy_seed"))
    if policy_seed is not None and (type(policy_seed) is not int or policy_seed < 0):
        raise ValueError(f"policy_seed 必须为非负整数：{policy_seed!r}")
    if (identity.get("policy_seed") is not None and header.get("policy_seed") is not None
            and identity["policy_seed"] != header["policy_seed"]):
        raise ValueError("trace header.policy_seed 与 identity.policy_seed 不符")
    cap_hit = end.get("cap_hit") is True
    # C3：四种终态；成功字段与终态字段一致，唯一例外是旧口径的 error/timeout
    if (end_status not in TERMINALS or terminal not in TERMINALS or
            (terminal != end_status and (terminal, end_status) != ("error", "timeout"))):
        raise ValueError("不支持非正常终态或终态冲突")
    if cap_hit and end_status not in ("timeout", "error"):
        raise ValueError(f"end.cap_hit=true 但 status={end_status}（strict-cap 命中只能是 timeout）")
    max_steps = header["max_steps"]
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps 必须为正整数")
    raw_steps = [r for r in rows if r["kind"] == "step"]
    texts = demo.get("texts")
    if end.get("no_frame") is True:
        # C3：无帧 error 局只核终态与计数，不读画面；重绘器记原因、不出视频
        if end_status != "error":
            raise ValueError("C3 只有 error 局允许 no_frame")
        if demo.get("frames") or any(st.get("observed") is not False for st in raw_steps):
            raise ValueError("C3 no_frame 局不应有任何画面")
        _check_counts(end, attempted=len(raw_steps), observed=0, frames=0, omitted=0)
        goal = texts[0] if isinstance(texts, list) and texts and isinstance(texts[0], str) else ""
        reason = end.get("no_frame_reason") or end.get("error") or "no_frame"
        steps = [{**st, "observed": False} for st in raw_steps]
        return TraceData(identity, route, goal, int(end.get("demo_frames") or 0), [], None, steps, terminal,
                         end_status, max_steps, 0, no_frame=True, no_frame_reason=str(reason),
                         policy_seed=policy_seed, cap_hit=cap_hit)
    frames, demo_frames = demo["frames"], end["demo_frames"]
    if type(frames) is not int or type(demo_frames) is not int or demo_frames < 0 or frames != demo_frames + 1:
        raise ValueError("demo.frames 必须等于 end.demo_frames + 1")
    if any(not isinstance(demo.get(k), list) or len(demo[k]) != frames for k in ("states", "front_sha256", "wrist_sha256")):
        raise ValueError("初始状态与画面哈希数量不完整")
    if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str) or not texts[0].strip():
        raise ValueError("demo.texts 必须包含唯一非空任务目标")
    init = [f32_from_record(s) for s in demo["states"]]
    if any(np.dtype(s["dtype"]) != np.dtype("<f4") for s in demo["states"]):
        raise ValueError("初始状态非 float32，f32hex 不能证明完整原数组")
    steps = []
    arrays_path = arrays_path or Path(path).with_name("arrays.npz")
    arrays = np.load(arrays_path, allow_pickle=False) if arrays_path.exists() else None
    try:
        for st in raw_steps:
            if type(st["step"]) is not int:
                raise ValueError("step 必须为整数")
            if not isinstance(st.get("subgoal"), (str, type(None))):
                raise ValueError("subgoal 必须为文本或 None")
            observed = st.get("observed", True)
            if observed is not True and observed is not False:
                raise ValueError("observed 只能是布尔值")
            if observed:
                if not st.get("front_sha256") or not st.get("wrist_sha256"):
                    raise ValueError("执行步没有完整画面，不能与视频配对（缺观测步须 observed=false）")
                if not isinstance(st.get("state"), dict) or np.dtype(st["state"]["dtype"]) != np.dtype("<f4"):
                    raise ValueError("执行状态非 float32，f32hex 不能证明完整原数组")
            elif not st.get("missing_reason"):
                raise ValueError(f"C8 第 {st['step']} 步缺观测但没写原因")
            action_rec = st.get("action")
            if not isinstance(action_rec, dict):
                raise ValueError(f"C4 第 {st['step']} 步缺动作")
            action = f32_from_record(action_rec)
            key = f"exec_action__{st['step'] - 1:05d}"
            if arrays is not None:
                if key not in arrays.files:
                    raise ValueError(f"C4 arrays.npz 缺 {key}")
                action = arrays[key]
                if (list(action.shape) != action_rec["shape"] or action.dtype.str != action_rec["dtype"] or
                        hashlib.sha256(action.tobytes()).hexdigest() != action_rec["sha256"] or not np.isfinite(action).all()):
                    raise ValueError(f"原动作 {key} 与 trace dtype/shape/sha256 不符")
            elif np.dtype(action_rec["dtype"]) != np.dtype("<f4"):
                raise ValueError("缺少原动作 arrays.npz，拒绝以 f32hex 丢失原数组精度")
            state = f32_from_record(st["state"]) if observed else None
            steps.append({**st, "observed": observed, "state": state, "action": action})
    finally:
        if arrays is not None:
            arrays.close()
    if len(steps) > max_steps + 1 or (len(steps) > max_steps and end_status != "timeout"):
        raise ValueError("执行步超出官方上限且不是唯一超时末步")
    # 官方 eval_each_episode 在超过 max_steps 时先 break，最后一步不调用 record；该步无观测时本就没有帧可省。
    omitted = int(terminal == "timeout" and end_status == "timeout" and len(steps) == max_steps + 1
                  and steps[-1]["observed"])
    observed_n = sum(st["observed"] for st in steps)
    _check_counts(end, attempted=len(steps), observed=observed_n, frames=frames + observed_n - omitted, omitted=omitted)
    subgoal = "[initializing...]" if any(st["subgoal"] is not None for st in steps) or "ground-sg" in route else None
    return TraceData(identity, route, texts[0], demo_frames, init, subgoal, steps, terminal, end_status, max_steps,
                     omitted, demo_front_sha256=list(demo["front_sha256"]), demo_wrist_sha256=list(demo["wrist_sha256"]),
                     policy_seed=policy_seed, cap_hit=cap_hit)


def feed_official_recorder(rec, front, wrist, trace: TraceData, task: str, video_demo_tasks=None,
                           max_output_bytes: int | None = None) -> None:
    """顺序与官方 init_episode / eval_each_episode 相同，所有绘图归原类。"""
    if len(front) != trace.source_frames or len(wrist) != trace.source_frames:
        raise ValueError(f"frame_count：输入 {len(front)}/{len(wrist)}，轨迹 {trace.source_frames}")
    front, wrist = np.asarray(front), np.asarray(wrist)
    if front.dtype != np.uint8 or wrist.dtype != np.uint8 or front.ndim != 4 or front.shape != wrist.shape or front.shape[-1] != 3:
        raise ValueError("front/wrist 必须为同尺寸的 N×H×W×3 uint8 数组")
    if video_demo_tasks is None:
        _, video_demo_tasks = load_official(REPO)
    n_init = len(trace.init_states)
    recorded_bytes = 0

    def record(**kwargs):
        nonlocal recorded_bytes
        rec.record(**kwargs)
        recorded_bytes += rec.total_images[-1].nbytes
        if max_output_bytes is not None and recorded_bytes > max_output_bytes:
            raise ValueError("官方文字区的实际帧内存超过本局预算")

    for i, state in enumerate(trace.init_states):
        record(image=front[i].copy(), wrist_image=wrist[i].copy(), state=state,
               is_video_demo=task in video_demo_tasks and i < n_init - 1, subgoal=trace.init_subgoal)
    observed = trace.observed_steps
    for k, st in enumerate(observed[:len(observed) - trace.omitted_timeout_frames]):
        # C7：有子目标的路线里 None（模型等待中）以 [initializing...] 占位；无子目标路线 init_subgoal 为 None，原样
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
                           "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)],
                          capture_output=True, text=True, check=True)
    streams = json.loads(proc.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("视频必须有唯一画面流")
    s = streams[0]
    return {"width": int(s["width"]), "height": int(s["height"]), "frames": int(s["nb_read_frames"]),
            "fps": str(Fraction(s["avg_frame_rate"]))}


def probe_stream(ffprobe: str, path: Path) -> dict:
    """原始流的编码器、尺寸与完整解码帧数（raw-new 用来核 FFV1 无损与帧数）。"""
    proc = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries",
                           "stream=codec_type,codec_name,width,height,nb_read_frames", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"原始流不可读：{path.name}：{proc.stderr.strip()[:200]}")
    streams = [s for s in json.loads(proc.stdout).get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"原始流必须有唯一画面流：{path.name}")
    s = streams[0]
    return {"codec": s.get("codec_name"), "width": int(s["width"]), "height": int(s["height"]),
            "frames": int(s["nb_read_frames"])}


def _ffmpeg_rawvideo(ffmpeg: str, path: Path, n: int, h: int, w: int) -> np.ndarray:
    """流式解码为 rgb24，读入预分配数组，避免整段 stdout 再复制一份大缓冲；帧数必须恰为 n。"""
    images = np.empty((n, h, w, 3), dtype=np.uint8)
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen([ffmpeg, "-v", "error", "-threads", "1", "-i", str(path), "-f", "rawvideo",
                                 "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"], stdout=subprocess.PIPE, stderr=err)
        try:
            for arr in images:
                view = memoryview(arr).cast("B")
                offset = 0
                while offset < len(view):
                    got = proc.stdout.readinto(view[offset:])
                    if not got:
                        raise ValueError("解码帧提前结束")
                    offset += got
            if proc.stdout.read(1):
                raise ValueError("解码帧数超过 ffprobe")
            code = proc.wait()
            if code:
                err.seek(0)
                raise ValueError(f"ffmpeg 退出 {code}：{err.read().decode(errors='replace')}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
    return images


def decode_mp4(ffmpeg: str, mp4: Path, info: dict | None = None):
    """流式读入预分配数组，避免整段 stdout 再复制一份数 GB 缓冲。"""
    info = info or probe_video(str(Path(ffmpeg).with_name("ffprobe")), mp4)
    if (info["width"], info["height"]) != (512, 256):
        raise ValueError("源视频必须为 512×256")
    images = _ffmpeg_rawvideo(ffmpeg, mp4, info["frames"], 256, 512)
    return images[:, :, :256], images[:, :, 256:]


def safe_filename(full_name: str) -> str:
    """完整官方语义写入 sidecar；文件系统不合法或超长时附固定摘要。"""
    sanitized = re.sub(r"[/\\\x00]", "_", full_name)
    if sanitized == full_name and len(sanitized.encode()) <= 255:
        return sanitized
    suffix = "__" + hashlib.sha256(full_name.encode()).hexdigest()[:16] + ".mp4"
    stem = sanitized.removesuffix(".mp4").encode()[:255 - len(suffix.encode())].decode(errors="ignore")
    return stem + suffix


def _episode_tag(trace: TraceData, ep_dir: Path, key: str | None = None) -> str:
    """局标签 ``<key>.a<N>``：默认取目录名；Astra 等目录名不含 key 时由 ``--key`` 显式给出（C6）。"""
    identity = trace.identity
    name = ep_dir.name
    if key is not None:
        if not key or any(c in key for c in "/\\\x00"):
            raise ValueError("--key 不合法")
        if identity.get("key") is not None and str(identity["key"]) != key:
            raise ValueError(f"--key={key} 与 trace.identity.key={identity['key']} 不符")
        name = f"{key}.a{int(identity.get('attempt') or 1)}"
    if identity.get("key") and not re.fullmatch(re.escape(str(identity["key"])) + r"\.a[1-9]\d*", name):
        raise ValueError("局目录与 trace.identity.key 不符")
    m = re.fullmatch(r".+\.a([1-9]\d*)", name)
    if identity.get("attempt") is not None and m and int(m[1]) != int(identity["attempt"]):
        raise ValueError(f"局目录尝试号 a{m[1]} 与 trace.identity.attempt={identity['attempt']} 不符")
    return name


def _episode_id(trace: TraceData, ep_dir: Path, key: str | None = None) -> str:
    identity = trace.identity
    tag = _episode_tag(trace, ep_dir, key)
    if trace.route.endswith("/new"):
        if identity.get("source_episode") is None and identity.get("builder_episode") is None:
            raise ValueError("新侧身份没有 source_episode 或 builder_episode")
        mc = _load(REPO / "scripts/eval-official/groundsg_client.py", "rerender_groundsg_client")
        return mc.official_episode_id(identity, tag)
    if identity.get("source_episode") is None:
        raise ValueError("原侧缺少 source_episode")
    return str(identity["source_episode"])


# ── 来源选择与解码 ─────────────────────────────────────────────────────────


@dataclass
class Source:
    """一局的画面来源；``streams``／``index`` 为「局目录相对路径 → 绝对路径」。"""

    kind: str
    streams: dict[str, Path]
    index: dict[str, Path]

    def media(self, ep_dir: Path) -> dict:
        return {"streams": {rel: fingerprint(p) for rel, p in self.streams.items()},
                "index": {rel: fingerprint(p) for rel, p in self.index.items()}}


def _rel(ep_dir: Path, p: Path) -> str:
    return p.relative_to(ep_dir).as_posix()


def _raw_candidates(ep_dir: Path) -> list[tuple[str, Path]]:
    """只看原始流本身是否还在（转码后 frames-*.jsonl／frames.json 保留、流文件已删，同 transcode_episode_dir）。"""
    out = []
    for base in (ep_dir, ep_dir / "media"):
        if any((base / f"{s}.mkv").exists() for s in STREAMS):
            out.append(("raw-new", base))
    if any((ep_dir / "frames" / f"{s}.rgb24").exists() for s in STREAMS):
        out.append(("raw-orig", ep_dir / "frames"))
    return out


def select_source(ep_dir: Path, mode: str) -> Source:
    """``auto``：有原始流即 raw，否则 mp4；``raw`` 缺原始流即失败，不退回 mp4。"""
    if mode not in SOURCE_MODES:
        raise ValueError(f"--source 只能是 {SOURCE_MODES}")
    cands = _raw_candidates(ep_dir)
    if len(cands) > 1:
        raise ValueError("原始帧来源不唯一：" + ",".join(f"{k}@{_rel(ep_dir, b) or '.'}" for k, b in cands))
    mp4 = ep_dir / "episode.mp4"
    if mode == "mp4" or (mode == "auto" and not cands):
        if not mp4.is_file():
            raise ValueError("没有可用来源：缺 episode.mp4" + ("" if mode == "mp4" else "，也没有原始帧"))
        return Source("mp4", {_rel(ep_dir, mp4): mp4}, {})
    if not cands:
        raise ValueError("raw 模式缺原始帧（不退回 mp4）")
    kind, base = cands[0]
    streams, index = {}, {}
    for s in STREAMS:
        if kind == "raw-new":
            stream, idx = base / f"{s}.mkv", base / f"frames-{s}.jsonl"
        else:
            stream, idx = base / f"{s}.rgb24", None
        if not stream.is_file() or stream.is_symlink():
            raise ValueError(f"raw 缺流 {_rel(ep_dir, stream)}")
        streams[_rel(ep_dir, stream)] = stream
        if idx is not None:
            if not idx.is_file():
                raise ValueError(f"raw 缺索引 {_rel(ep_dir, idx)}")
            index[_rel(ep_dir, idx)] = idx
    if kind == "raw-new":
        if (base / "meta.json").is_file():
            index[_rel(ep_dir, base / "meta.json")] = base / "meta.json"
    else:
        meta = base / "frames.json"
        if not meta.is_file():
            raise ValueError(f"raw 缺索引 {_rel(ep_dir, meta)}")
        index[_rel(ep_dir, meta)] = meta
    return Source(kind, streams, index)


def read_frame_index(path: Path) -> list[dict]:
    """读 ``frames-<stream>.jsonl``：idx 唯一且 0..n-1 连续、sha256 完整、enc 为非负整数（enc 为空即 2 档有损降级）。"""
    rows = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            raise ValueError(f"索引损坏：{path.name} 第 {ln} 行不是 JSON") from None
        if (not isinstance(r, dict) or type(r.get("idx")) is not int or not isinstance(r.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", r["sha256"])):
            raise ValueError(f"索引损坏：{path.name} 第 {ln} 行缺 idx/sha256")
        if r.get("enc") is None:
            raise ValueError(f"有损降级：{path.name} 第 {ln} 行 enc 为空（只留首尾帧的降级档）")
        if type(r["enc"]) is not int or r["enc"] < 0:
            raise ValueError(f"索引损坏：{path.name} 第 {ln} 行 enc 非法")
        rows.append(r)
    idxs = [r["idx"] for r in rows]
    if len(set(idxs)) != len(idxs):
        raise ValueError(f"重复索引：{path.name}")
    if sorted(idxs) != list(range(len(idxs))):
        raise ValueError(f"索引损坏：{path.name} idx 不是 0..n-1 连续")
    return sorted(rows, key=lambda r: r["idx"])


def _select_rows(rows: list[dict], trace: TraceData, stream: str) -> list[dict]:
    """把索引行映射到来源逐帧：行数恰等于来源帧数时一一对应；否则按 env_client 的 tag
    （演示段 ``reset``，第 k 步 ``step{k-1}`` 取该步最后一帧）选出，缺任何一帧即失败。"""
    if len(rows) == trace.source_frames:
        return rows
    reset = [r for r in rows if r.get("tag") == "reset"]
    if len(reset) != len(trace.init_states):
        raise ValueError(f"缺帧：{stream} 索引 {len(rows)} 行，演示段 {len(reset)} 帧，trace 需 {trace.source_frames} 帧")
    by_tag: dict[str, dict] = {}
    for r in rows:
        by_tag[str(r.get("tag"))] = r
    out = list(reset)
    for st in trace.observed_steps:
        r = by_tag.get(f"step{st['step'] - 1}")
        if r is None:
            raise ValueError(f"缺帧：{stream} 第 {st['step']} 步在索引里没有画面")
        out.append(r)
    return out


def _verify_trace_hashes(trace: TraceData, stream: str, frames: np.ndarray) -> int:
    """逐帧核对解码画面的 ``image_sha256`` 等于 trace 记录；返回核对帧数。"""
    tw = _trace_writer()
    expect = trace.frame_hashes(stream)
    if len(expect) != len(frames):
        raise ValueError(f"缺帧：{stream} 来源 {len(frames)} 帧，trace {len(expect)} 帧")
    for i, (img, want) in enumerate(zip(frames, expect)):
        if tw.image_sha256(img) != want:
            raise ValueError(f"逐帧哈希不符：{stream} 第 {i} 帧与 trace 记录不同（调包流或错位）")
    return len(frames)


def decode_raw_new(ffmpeg: str, ep_dir: Path, src: Source, trace: TraceData) -> tuple[np.ndarray, np.ndarray, dict]:
    """FFV1 流按索引 enc 展开回逐帧原图；核无损、核索引 sha256、核 trace 画面哈希。"""
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    meta_rel = next((r for r in src.index if r.endswith("meta.json")), None)
    if meta_rel is not None:
        try:
            meta = json.loads(src.index[meta_rel].read_text(encoding="utf-8"))
        except ValueError:
            raise ValueError(f"索引损坏：{meta_rel} 不是 JSON") from None
        if meta.get("level", 0) != 0 or meta.get("codec", "ffv1") != "ffv1":
            raise ValueError(f"有损降级：{meta_rel} level={meta.get('level')} codec={meta.get('codec')}")
    summary = next(iter(src.streams.values())).parent / "summary.json"
    if summary.is_file():
        try:
            lossless = json.loads(summary.read_text(encoding="utf-8")).get("lossless")
        except ValueError:
            lossless = None  # 摘要不可读不作判据；无损性以下面的编码器与逐帧 sha256 为准
        if lossless is False:
            raise ValueError("有损降级：summary.json lossless=false")
    out, checked = {}, {}
    for s in STREAMS:
        stream = next(p for r, p in src.streams.items() if r.endswith(f"{s}.mkv"))
        idx_path = next(p for r, p in src.index.items() if r.endswith(f"frames-{s}.jsonl"))
        rows = read_frame_index(idx_path)
        info = probe_stream(ffprobe, stream)
        if info["codec"] != "ffv1":
            raise ValueError(f"有损降级：{stream.name} 编码器 {info['codec']} 不是 ffv1")
        encs = sorted({r["enc"] for r in rows})
        if encs != list(range(info["frames"])):
            raise ValueError(f"索引损坏：{s} 流解码 {info['frames']} 帧，索引 enc 覆盖 {len(encs)} 个"
                             f"（最大 {encs[-1] if encs else None}），调包流或错 enc")
        decoded = _ffmpeg_rawvideo(ffmpeg, stream, info["frames"], info["height"], info["width"])
        enc_sha = [hashlib.sha256(f.tobytes()).hexdigest() for f in decoded]
        bad = [r["idx"] for r in rows if enc_sha[r["enc"]] != r["sha256"]]
        if bad:
            raise ValueError(f"解码画面与索引 sha256 不符：{s} idx={bad[:5]}（调包流或错 enc）")
        picked = _select_rows(rows, trace, s)
        frames = decoded[[r["enc"] for r in picked]]
        del decoded
        checked[s] = _verify_trace_hashes(trace, s, frames)
        out[s] = frames
    if out["front"].shape != out["wrist"].shape:
        raise ValueError("front/wrist 尺寸不同")
    return out["front"], out["wrist"], {"index_rows": checked}


def decode_raw_orig(ep_dir: Path, src: Source, trace: TraceData) -> tuple[np.ndarray, np.ndarray, dict]:
    """``frames/*.rgb24`` 按 ``frames.json`` 的尺寸与帧数切帧；帧数必须恰等于 trace 来源帧数。"""
    meta_path = next(p for r, p in src.index.items() if r.endswith("frames.json"))
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        raise ValueError("索引损坏：frames/frames.json 不是 JSON") from None
    if meta.get("pix_fmt", "rgb24") != "rgb24":
        raise ValueError(f"有损降级：pix_fmt={meta.get('pix_fmt')}")
    if meta.get("demo_frames") is not None and meta["demo_frames"] != trace.demo_frames:
        raise ValueError(f"frames.json demo_frames={meta['demo_frames']} 与 trace {trace.demo_frames} 不符")
    if meta.get("missing_steps") is not None and list(meta["missing_steps"]) != trace.missing_steps:
        raise ValueError(f"frames.json missing_steps 与 trace 缺观测步 {trace.missing_steps} 不符")
    out = {}
    for s in STREAMS:
        st = (meta.get("streams") or {}).get(s)
        if not isinstance(st, dict) or not all(type(st.get(k)) is int and st[k] > 0 for k in ("width", "height", "count")):
            raise ValueError(f"索引损坏：frames.json 缺 {s} 的 width/height/count")
        w, h, n = st["width"], st["height"], st["count"]
        if n != trace.source_frames:
            raise ValueError(f"缺帧：{s} count={n}，trace 需 {trace.source_frames} 帧")
        path = next(p for r, p in src.streams.items() if r.endswith(f"{s}.rgb24"))
        if path.stat().st_size != n * w * h * 3:
            raise ValueError(f"缺帧：{s}.rgb24 字节数 {path.stat().st_size} != {n}×{w}×{h}×3")
        out[s] = np.fromfile(path, dtype=np.uint8).reshape(n, h, w, 3)
    if out["front"].shape != out["wrist"].shape:
        raise ValueError("front/wrist 尺寸不同")
    checked = {s: _verify_trace_hashes(trace, s, out[s]) for s in STREAMS}
    return out["front"], out["wrist"], {"index_rows": checked}


def _source_frame_bytes(ffmpeg: str, src: Source) -> int:
    """解码前估算来源每帧（front + wrist）字节数，用于内存预算。"""
    if src.kind == "mp4":
        return 512 * 256 * 3
    if src.kind == "raw-orig":
        meta = json.loads(next(p for r, p in src.index.items() if r.endswith("frames.json")).read_text(encoding="utf-8"))
        return sum(int(meta["streams"][s]["width"]) * int(meta["streams"][s]["height"]) * 3 for s in STREAMS)
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    total = 0
    for p in src.streams.values():
        proc = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                               "-of", "json", str(p)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValueError(f"原始流不可读：{p.name}")
        s = json.loads(proc.stdout)["streams"][0]
        total += int(s["width"]) * int(s["height"]) * 3
    return total


# ── 复用判定 ───────────────────────────────────────────────────────────────


def _safe_rel(rel: str) -> bool:
    p = Path(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts


def _source_still_valid(old: dict, ep_dir: Path, requested: str) -> bool:
    """旧结果的来源仍可核：索引文件必须在且逐字节一致；流文件若在必须一致，raw 流转码后被删允许。"""
    kind = old.get("source_kind") or ("mp4" if old.get("source_mp4") else None)
    if requested == "mp4" and kind != "mp4" or requested == "raw" and kind not in RAW_KINDS:
        return False
    media = old.get("source_media")
    if media is None and kind == "mp4":  # 第一阶段 render.json：只有 source_mp4
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


def _configure_threads(threads: int, cpus: list[int], worker_index: int) -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    import cv2
    cv2.setNumThreads(threads)
    # imageio 的官方 save_video 不接受额外 ffmpeg 参数；子进程继承 CPU 亲和性限制编码线程。
    if hasattr(os, "sched_setaffinity"):
        start = worker_index * threads
        os.sched_setaffinity(0, set(cpus[start:start + threads]))


def _init_worker(threads: int, cpus: list[int], counter) -> None:
    """共享计数器分配唯一工作进程编号；每个进程只配置一次，不按 PID 碰撞。"""
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    _configure_threads(threads, cpus, index)


def render_episode(ep_dir: Path, *, official_root: Path = REPO, ffmpeg: str = "/usr/bin/ffmpeg",
                   out_subdir: str = "official", prefix: str = "official-rerender__", overwrite: bool = False,
                   max_memory_mib: int = 6144, source: str = "auto", key: str | None = None) -> dict:
    """每局原子产出视频与身份清单；旧结果只有全链路指纹一致才复用。

    流程：读 trace（C3/C4/C8）→ 复用判定 → 来源选择 → 解码 → 逐帧核验 → 官方录像 → 输出复核。
    无帧 error 局只写 ``render.json``（``render_status=no_frame`` 与原因），不出视频。
    """
    ep_dir, official_root = Path(ep_dir).resolve(), Path(official_root).resolve()
    if Path(out_subdir).name != out_subdir or out_subdir in ("", ".", "..") or any(c in prefix for c in "/\\\x00"):
        raise ValueError("输出子目录或前缀不安全")
    if source not in SOURCE_MODES:
        raise ValueError(f"--source 只能是 {SOURCE_MODES}")
    trace_path, arrays_path = ep_dir / "trace.jsonl", ep_dir / "arrays.npz"
    source_trace = fingerprint(trace_path)
    source_arrays = fingerprint(arrays_path) if arrays_path.exists() else None
    trace = load_trace(trace_path)
    episode_tag = _episode_tag(trace, ep_dir, key)
    episode_id = _episode_id(trace, ep_dir, key)
    out_dir = ep_dir / out_subdir
    if out_dir.is_symlink():
        raise ValueError("输出目录不能为符号链接")
    if trace.no_frame:
        full_name = safe_name = None
        output = out_dir / ".no-frame-placeholder.mp4"  # 只用于「不应存在视频」的判定
    else:
        full_name = (f"{prefix}{trace.identity['task']}_ep{episode_id}_{trace.named_terminal}_{trace.task_goal}_"
                     f"{trace.identity['tier']}.mp4")
        safe_name = safe_filename(full_name)
        output = out_dir / safe_name
    sidecar = out_dir / "render.json"
    if output.is_symlink() or sidecar.is_symlink():
        raise ValueError("输出视频与清单不能为符号链接")
    ffmpeg = str(Path(ffmpeg).resolve())
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg
    import cv2
    import imageio
    observed_n = len(trace.observed_steps)
    base = {"schema": SCHEMA, "identity": trace.identity, "episode_id": episode_id, "episode_tag": episode_tag,
            "route": trace.route, "task_goal": trace.task_goal, "terminal_reason": trace.named_terminal,
            "trace_terminal_reason": trace.terminal_reason, "cap_hit": trace.cap_hit, "policy_seed": trace.policy_seed,
            "status": trace.end_status, "no_frame": trace.no_frame,
            "source_frames": trace.source_frames, "frames": trace.output_frames, "demo_frames": trace.demo_frames,
            "exec_steps": len(trace.steps), "steps_attempted": len(trace.steps), "steps_observed": observed_n,
            "missing_steps": trace.missing_steps, "frames_recorded": trace.output_frames,
            "recorded_exec_steps": observed_n - trace.omitted_timeout_frames,
            "omitted_timeout_frames": trace.omitted_timeout_frames, "full_name": full_name, "safe_name": safe_name,
            "out_rel": None if safe_name is None else f"{out_subdir}/{safe_name}",
            "source_trace": source_trace, "source_arrays": source_arrays,
            "tool": fingerprint(Path(__file__)),
            "official": fingerprint(official_root / "third_party/mme-vla/examples/robomme/utils.py"),
            "official_eval": fingerprint(official_root / "third_party/mme-vla/examples/robomme/eval.py"),
            "episode_id_helper": fingerprint(REPO / "scripts/eval-official/groundsg_client.py"),
            "trace_reader": fingerprint(REPO / "scripts/eval-official/trace_writer.py"), "ffmpeg": fingerprint(Path(ffmpeg)),
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "imageio": imageio.__version__}}
    if output.exists() or sidecar.exists():
        reused = _try_reuse(sidecar, output, base, ep_dir, source, ffprobe, trace)
        if reused is not None:
            return reused
        if not overwrite:
            raise ValueError("已有输出未通过 provenance 与完整视频验证；拒绝覆盖，请显式使用 --overwrite")
    if trace.no_frame:
        out_dir.mkdir(exist_ok=True)
        result = {**base, "source_kind": "none", "source_media": None, "source_mp4": None, "dir": ep_dir.name,
                  "render_status": "no_frame", "no_frame_reason": trace.no_frame_reason, "out": None}
        _write_sidecar(out_dir, sidecar, result)
        return result
    src = select_source(ep_dir, source)
    media = src.media(ep_dir)
    context = {**base, "source_kind": src.kind, "source_media": media,
               "source_mp4": media["streams"].get("episode.mp4") if src.kind == "mp4" else None}
    info = None
    if src.kind == "mp4":
        mp4 = src.streams["episode.mp4"]
        info = probe_video(ffprobe, mp4)
        if info != {"width": 512, "height": 256, "frames": trace.source_frames, "fps": "30"}:
            raise ValueError(f"源视频帧数/帧率/尺寸不符：{info}；期望 frames={trace.source_frames} fps=30 size=512x256")
    frame_src_bytes = _source_frame_bytes(ffmpeg, src)
    # raw-new 解码时还要容纳去重后的编码帧，按来源帧数再计一份
    src_bytes = trace.source_frames * frame_src_bytes * (2 if src.kind == "raw-new" else 1)
    Recorder, demo_tasks = load_official(official_root)
    out_dir.mkdir(exist_ok=True)
    rec = Recorder(out_dir, trace.task_goal, fps=30)
    # 先绘单帧估计官方版式高度，源解码 + 全部官方帧 + 编码余量一起计入预算。
    dummy = np.zeros((256, 256, 3), np.uint8)
    rec.record(dummy, dummy, trace.init_states[0], subgoal=trace.init_subgoal)
    frame_bytes = rec.total_images[0].nbytes
    estimated = int((src_bytes + trace.output_frames * frame_bytes) * 1.35 + 64 * 1024**2)
    rec.total_images.clear()
    if estimated > max_memory_mib * 1024**2:
        raise ValueError(f"预计内存 {estimated / 1024**2:.0f} MiB 超过上限 {max_memory_mib} MiB")
    if src.kind == "mp4":
        front, wrist = decode_mp4(ffmpeg, src.streams["episode.mp4"], info)
        hash_check = {"mode": "skipped-lossy-mp4"}
    elif src.kind == "raw-new":
        front, wrist, detail = decode_raw_new(ffmpeg, ep_dir, src, trace)
        hash_check = {"mode": "verified", "frames": detail["index_rows"], "mismatch": 0}
    else:
        front, wrist, detail = decode_raw_orig(ep_dir, src, trace)
        hash_check = {"mode": "verified", "frames": detail["index_rows"], "mismatch": 0}
    output_budget = int((max_memory_mib * 1024**2 - 64 * 1024**2) / 1.35 - src_bytes)
    feed_official_recorder(rec, front, wrist, trace, trace.identity["task"], demo_tasks, output_budget)
    del front, wrist
    if len(rec.total_images) != trace.output_frames or len({a.shape for a in rec.total_images}) != 1:
        raise ValueError("官方录像器帧数或逐帧尺寸不一致")
    temp = out_dir / f".render-{uuid.uuid4().hex}.mp4"
    try:
        rec.save_video(temp.name)
        actual = probe_video(ffprobe, temp)
        shape = rec.total_images[0].shape
        expected_size = ((shape[1] + 15) // 16 * 16, (shape[0] + 15) // 16 * 16)
        if actual["frames"] != trace.output_frames or actual["fps"] != "30" or (actual["width"], actual["height"]) != expected_size:
            raise ValueError(f"输出完整性不符：{actual}")
        if (fingerprint(trace_path) != source_trace or
                (fingerprint(arrays_path) if arrays_path.exists() else None) != source_arrays or
                src.media(ep_dir) != media):
            raise ValueError("源文件在重绘期间发生变化")
        result = {**context, **actual, "dir": ep_dir.name, "render_status": "rendered", "out": str(output),
                  "estimated_memory_mib": round(estimated / 1024**2), "output_fingerprint": fingerprint(temp),
                  "input_array_precision": "verified-original", "frame_hash_check": hash_check}
        os.replace(temp, output)
        _write_sidecar(out_dir, sidecar, result)
        return result
    finally:
        temp.unlink(missing_ok=True)


def _worker(ep_dir: str, opts: dict) -> dict:
    try:
        return {"ok": True, **render_episode(Path(ep_dir), **opts)}
    except Exception as exc:
        return {"ok": False, "dir": Path(ep_dir).name, "reason": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--label", default="groundsg-ground-sg-oracle")
    parser.add_argument("--official-root", type=Path, default=REPO)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/usr/bin/ffmpeg")
    parser.add_argument("--out-subdir", default="official")
    parser.add_argument("--prefix", default="official-rerender__")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-memory-mib", type=int, default=6144)
    parser.add_argument("--source", choices=SOURCE_MODES, default="auto",
                        help="画面来源：auto 有原始流即 raw 否则 mp4；raw 缺流或索引不合格即失败、不退回 mp4")
    parser.add_argument("--key", help="目录名不含 key 时（如 Astra）显式给出局 key，只允许单个局目录")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.threads < 1 or args.max_memory_mib < 128 or bool(args.root) == bool(args.episodes):
        parser.error("必须选择局目录或 --root，且资源参数必须为正数")
    if args.key is not None and (args.root or len(args.episodes) != 1):
        parser.error("--key 只能配合单个局目录使用")
    if args.root:
        root = args.root.resolve()
        dirs = sorted(p.parent for p in root.rglob("trace.jsonl")
                      if args.label in p.parts and not any(part.startswith(".") for part in p.relative_to(root).parts))
    else:
        dirs = [p.resolve() for p in args.episodes]
    if not dirs or len(set(dirs)) != len(dirs):
        parser.error("没有局目录或包含重复目录")
    opts = {k: getattr(args, k) for k in ("official_root", "ffmpeg", "out_subdir", "prefix", "overwrite", "max_memory_mib",
                                          "source", "key")}
    available = next((int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:")), args.max_memory_mib * 1024**2)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    if args.threads > len(cpus):
        parser.error("--threads 超过可用 CPU 数")
    jobs = min(args.jobs, len(dirs), max(1, int(available * 0.6) // (args.max_memory_mib * 1024**2)), len(cpus) // args.threads)
    print(f"OFFICIAL_RENDER_RESOURCES jobs={jobs} requested_jobs={args.jobs} threads={args.threads} max_memory_mib={args.max_memory_mib}", flush=True)
    ok = fail = reused = no_frame = 0
    counter = multiprocessing.Value("i", 0)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(args.threads, cpus, counter)) as pool:
        futures = {pool.submit(_worker, str(p), dict(opts)): p for p in dirs}
        for f in as_completed(futures):
            try:
                result = f.result()
            except Exception as exc:
                result = {"ok": False, "dir": futures[f].name, "reason": f"{type(exc).__name__}: {exc}"}
            if result["ok"] and result["render_status"] == "no_frame":
                ok += 1
                no_frame += 1
                reused += bool(result.get("reused"))
                print(f"OFFICIAL_RENDER=NO_FRAME dir={result['dir']} source_kind=none terminal_reason={result['terminal_reason']} end_status={result['status']} steps={result['exec_steps']} reason={json.dumps(result['no_frame_reason'], ensure_ascii=False)}", flush=True)
            elif result["ok"]:
                ok += 1
                reused += result["render_status"] == "reused"
                print(f"OFFICIAL_RENDER=PASS dir={result['dir']} frames={result['frames']} demo={result['demo_frames']} steps={result['exec_steps']} omitted={result['omitted_timeout_frames']} size={result['width']}x{result['height']} status={result['render_status']} source_kind={result.get('source_kind', 'mp4')} terminal_reason={result['terminal_reason']} end_status={result['status']} out={json.dumps(result['out'])}", flush=True)
            else:
                fail += 1
                print(f"OFFICIAL_RENDER=FAIL dir={result['dir']} reason={json.dumps(result['reason'], ensure_ascii=False)}", flush=True)
    print(f"OFFICIAL_RENDER_SUMMARY={'PASS' if fail == 0 else 'FAIL'} total={len(dirs)} ok={ok} fail={fail} reused={reused} no_frame={no_frame}", flush=True)
    return int(fail > 0)


if __name__ == "__main__":
    raise SystemExit(main())
