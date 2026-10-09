"""官方版式渲染库（抽自 ``dev-scripts/media/render_official_video.py`` 的库部分；该脚本由 E4 改成本模块的薄 CLI）。

从已有原始帧与轨迹离线调用官方 ``RolloutRecorder``；不启动仿真，不修改源视频。官方 ``RolloutRecorder`` 只经
``importlib`` 从锁定的 ``third_party/mme-vla/examples/robomme/utils.py`` 原文件加载，不复制不改写（R3）；所有绘图
（文字区、演示段红框）都归原类，本模块只把它画好的帧编码成网站视频。

流程：读 trace（共享契约 C3／C4／C8）→ 来源选择 → 解码 → 核验 → 官方录像器逐帧绘制 → 编码 → 复核。

来源三种：
- ``raw-new``：``recorder.py`` 写的 ``front.mkv``／``wrist.mkv`` + ``frames-<stream>.jsonl``（idx、sha256、tag、enc），
  **按局目录 ``meta.json`` 的 ``raw_codec`` 分版本读**：
  * ``av1-yuv444p``（2026-10-08 起）：有损，核编码器为 av1、解码帧数恰覆盖索引 enc，**不再校验解码字节 sha256**
    （``frame_hash_check.mode=skipped-lossy-av1``）；
  * 没有 ``raw_codec``（旧 FFV1 产物）：照旧核无损、核解码画面字节 sha256 等于索引、核 trace 画面哈希；
- ``raw-orig``：``frames/{front,wrist}.rgb24`` + ``frames/frames.json``（原侧观察器）；
- ``mp4``：局目录 ``episode.mp4``（512×256 左右拼接，有损，只核帧数）。

网站视频（R8）：``libaom-av1``、``-pix_fmt yuv420p``、30 fps、封装 MP4（``-movflags +faststart``）；尺寸按官方
``save_video``（imageio，macro_block_size=16）同样的规则向上取到 16 的倍数并缩放。

动作原值从 ``arrays.npz`` 恢复（C4），禁止用有舍入损失的 f32hex 替代非 float32 动作。
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

from robomme_hard_eval.record import trace_writer as _tw
from robomme_hard_eval.record.recorder import AV1_DECODE_FILTER, LEGACY_RAW_CODEC, RAW_CODEC, find_ffmpeg

PKG_ROOT = Path(__file__).resolve().parents[1]
#: eval 仓根（src/robomme_hard_eval/record/ 往上三级）；官方 RolloutRecorder 在其 third_party/mme-vla 下
REPO = Path(__file__).resolve().parents[3]
OFFICIAL_UTILS_REL = "third_party/mme-vla/examples/robomme/utils.py"
OFFICIAL_EVAL_REL = "third_party/mme-vla/examples/robomme/eval.py"
#: 环境变量可覆盖官方仓根（worktree 里子模块为空时指向主检出）
ENV_OFFICIAL_ROOT = "ROBOMME_EVAL_OFFICIAL_ROOT"
SCHEMA = "official-rerender/2"
TERMINALS = ("success", "fail", "timeout", "error")
STREAMS = ("front", "wrist")
SOURCE_MODES = ("auto", "mp4", "raw")
RAW_KINDS = ("raw-new", "raw-orig")
FPS = 30
#: 网站视频编码参数（R8：AV1 4:2:0 封装 MP4，faststart）
WEB_ENCODE_ARGS = ("-c:v", "libaom-av1", "-cpu-used", "4", "-crf", "24", "-b:v", "0", "-pix_fmt", "yuv420p",
                   "-row-mt", "1", "-threads", "2", "-r", str(FPS), "-movflags", "+faststart")
MACRO_BLOCK = 16


def default_official_root() -> Path:
    env = os.environ.get(ENV_OFFICIAL_ROOT)
    return Path(env) if env else REPO


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"不能加载模块：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_official(repo_root: Path | None = None):
    """直接执行锁定官方 utils 原文件；返回 (RolloutRecorder, TASK_WITH_VIDEO_DEMO)。"""
    path = Path(repo_root or default_official_root()) / OFFICIAL_UTILS_REL
    if not path.is_file():
        raise FileNotFoundError(f"找不到官方 RolloutRecorder：{path}（子模块未初始化？可设 {ENV_OFFICIAL_ROOT}）")
    mod = _load(path, "official_robomme_utils")
    return mod.RolloutRecorder, mod.TASK_WITH_VIDEO_DEMO


ENV_FFPROBE = "ROBOMME_FFPROBE"


def find_ffprobe(ffmpeg: str | None = None) -> str:
    """ffprobe：环境变量 ROBOMME_FFPROBE 优先；其次与 ffmpeg 同目录者（未给 ffmpeg 时按录制器 find_ffmpeg 的规则定位，
    认 V75_FFMPEG）；最后 PATH。GL 计算节点 PATH 上没有 ffprobe、imageio 自带的只有 ffmpeg，须经前两条找到。"""
    env = os.environ.get(ENV_FFPROBE)
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    if not ffmpeg:
        try:
            from .recorder import find_ffmpeg
            ffmpeg = find_ffmpeg()
        except Exception:  # noqa: BLE001 — 找不到 ffmpeg 时退回 PATH
            ffmpeg = None
    if ffmpeg:
        cand = Path(ffmpeg).with_name("ffprobe")
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    w = shutil.which("ffprobe")
    if not w:
        raise RuntimeError(f"找不到 ffprobe（可设 {ENV_FFPROBE}，或让 V75_FFMPEG 指向同目录带 ffprobe 的 ffmpeg）")
    return w


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
        """文件名用的终态：strict-cap 命中或 ``status=timeout`` 一律 ``timeout``。"""
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
            raise ValueError(f"C8 end.{key}={end[key]!r} 与 trace 推导值 {expect} 不符")


def load_trace(path: Path, arrays_path: Path | None = None) -> TraceData:
    """先核对序列与必需字段，再解码；demo.frames 包括初始帧。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("轨迹为空或行不是对象")
    if any(r.get("kind") not in ("header", "demo", "step", "request", "response", "history", "end") for r in rows):
        raise ValueError("轨迹包含未知行类型")
    problems = _tw.validate_trace(rows)
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
    """顺序与官方 init_episode / eval_each_episode 相同，所有绘图归原类。"""
    if len(front) != trace.source_frames or len(wrist) != trace.source_frames:
        raise ValueError(f"frame_count：输入 {len(front)}/{len(wrist)}，轨迹 {trace.source_frames}")
    front, wrist = np.asarray(front), np.asarray(wrist)
    if front.dtype != np.uint8 or wrist.dtype != np.uint8 or front.ndim != 4 or front.shape != wrist.shape or front.shape[-1] != 3:
        raise ValueError("front/wrist 必须为同尺寸的 N×H×W×3 uint8 数组")
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
        raise ValueError("视频必须有唯一画面流")
    s = streams[0]
    return {"width": int(s["width"]), "height": int(s["height"]), "frames": int(s["nb_read_frames"]),
            "fps": str(Fraction(s["avg_frame_rate"])), "codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt")}


def probe_stream(ffprobe: str, path: Path) -> dict:
    """原始流的编码器、像素格式、尺寸与完整解码帧数。"""
    proc = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries",
                           "stream=codec_type,codec_name,pix_fmt,width,height,nb_read_frames", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"原始流不可读：{path.name}：{proc.stderr.strip()[:200]}")
    streams = [s for s in json.loads(proc.stdout).get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"原始流必须有唯一画面流：{path.name}")
    s = streams[0]
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"), "width": int(s["width"]),
            "height": int(s["height"]), "frames": int(s["nb_read_frames"])}


def _ffmpeg_rawvideo(ffmpeg: str, path: Path, n: int, h: int, w: int, vf: str | None = None) -> np.ndarray:
    """流式解码为 rgb24，读入预分配数组；帧数必须恰为 n。``vf`` 为色彩还原滤镜（AV1 用）。"""
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
    info = info or probe_video(find_ffprobe(ffmpeg), mp4)
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


def official_episode_id(identity: dict, episode_tag: str) -> str:
    """旧局目录（``<key>.a<N>``）的短局号 ``<source_episode>a<attempt>``，与 ``models/groundsg.py`` 同式。"""
    src = identity.get("source_episode")
    if src is None:
        src = identity.get("builder_episode")
    att = episode_tag.rsplit(".a", 1)[1] if ".a" in episode_tag else "1"
    return f"{src}a{att}"


def _episode_tag(trace: TraceData, ep_dir: Path, key: str | None = None) -> str:
    """旧局目录标签 ``<key>.a<N>``：默认取目录名；目录名不含 key 时由 ``key`` 显式给出（C6）。"""
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
        return official_episode_id(identity, tag)
    if identity.get("source_episode") is None:
        raise ValueError("原侧缺少 source_episode")
    return str(identity["source_episode"])


# ── 来源选择与解码 ─────────────────────────────────────────────────────────


@dataclass
class Source:
    """一局的画面来源；``streams``／``index`` 为「局目录相对路径 → 绝对路径」；``raw_codec`` 只对 raw-new 有意义。"""

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
        raise ValueError(f"索引损坏：{meta_path.name} 不是 JSON") from None
    return str(meta.get("raw_codec") or LEGACY_RAW_CODEC)


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
        meta = base / "meta.json"
        if meta.is_file():
            index[_rel(ep_dir, meta)] = meta
        return Source(kind, streams, index, raw_codec=_meta_raw_codec(meta if meta.is_file() else None))
    meta = base / "frames.json"
    if not meta.is_file():
        raise ValueError(f"raw 缺索引 {_rel(ep_dir, meta)}")
    index[_rel(ep_dir, meta)] = meta
    return Source(kind, streams, index)


def read_frame_index(path: Path) -> list:
    """读 ``frames-<stream>.jsonl``：idx 唯一且 0..n-1 连续、sha256 完整、enc 为非负整数（enc 为空即 2 档降级）。"""
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


def _select_rows(rows: list, trace: TraceData, stream: str) -> list:
    """索引行数恰等于来源帧数时一一对应；否则按 tag（演示段 ``reset``，第 k 步 ``step{k-1}`` 取最后一帧）选出。"""
    if len(rows) == trace.source_frames:
        return rows
    reset = [r for r in rows if r.get("tag") == "reset"]
    if len(reset) != len(trace.init_states):
        raise ValueError(f"缺帧：{stream} 索引 {len(rows)} 行，演示段 {len(reset)} 帧，trace 需 {trace.source_frames} 帧")
    by_tag: dict = {}
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
    expect = trace.frame_hashes(stream)
    if len(expect) != len(frames):
        raise ValueError(f"缺帧：{stream} 来源 {len(frames)} 帧，trace {len(expect)} 帧")
    for i, (img, want) in enumerate(zip(frames, expect)):
        if _tw.image_sha256(img) != want:
            raise ValueError(f"逐帧哈希不符：{stream} 第 {i} 帧与 trace 记录不同（调包流或错位）")
    return len(frames)


def decode_raw_new(ffmpeg: str, ep_dir: Path, src: Source, trace: TraceData) -> tuple:
    """按 ``raw_codec`` 分版本读 recorder 原始流，按索引 enc 展开回逐帧画面。返回 (front, wrist, detail)。

    * ``av1-yuv444p``：核编码器 av1／像素格式 yuv444p、解码帧数恰覆盖索引 enc、逐帧数与 trace 对上；有损，不核字节；
    * ``ffv1``（旧产物）：核无损、核解码画面字节 sha256 等于索引、核 trace 画面哈希。"""
    ffprobe = find_ffprobe(ffmpeg)
    codec = src.raw_codec or LEGACY_RAW_CODEC
    if codec not in (RAW_CODEC, LEGACY_RAW_CODEC):
        raise ValueError(f"不认识的 raw_codec={codec!r}")
    lossy = codec == RAW_CODEC
    meta_rel = next((r for r in src.index if r.endswith("meta.json")), None)
    if meta_rel is not None and not lossy:
        meta = json.loads(src.index[meta_rel].read_text(encoding="utf-8"))
        if meta.get("level", 0) != 0 or meta.get("codec", "ffv1") != "ffv1":
            raise ValueError(f"有损降级：{meta_rel} level={meta.get('level')} codec={meta.get('codec')}")
    if not lossy:
        summary = next(iter(src.streams.values())).parent / "summary.json"
        if summary.is_file():
            try:
                lossless = json.loads(summary.read_text(encoding="utf-8")).get("lossless")
            except ValueError:
                lossless = None
            if lossless is False:
                raise ValueError("有损降级：summary.json lossless=false")
    out, checked = {}, {}
    for s in STREAMS:
        stream = next(p for r, p in src.streams.items() if r.endswith(f"{s}.mkv"))
        idx_path = next(p for r, p in src.index.items() if r.endswith(f"frames-{s}.jsonl"))
        rows = read_frame_index(idx_path)
        info = probe_stream(ffprobe, stream)
        if lossy:
            if info["codec"] != "av1" or info["pix_fmt"] != "yuv444p":
                raise ValueError(f"{stream.name} 编码 {info['codec']}/{info['pix_fmt']} 与 raw_codec={codec} 不符")
        elif info["codec"] != "ffv1":
            raise ValueError(f"有损降级：{stream.name} 编码器 {info['codec']} 不是 ffv1")
        encs = sorted({r["enc"] for r in rows})
        if encs != list(range(info["frames"])):
            raise ValueError(f"索引损坏：{s} 流解码 {info['frames']} 帧，索引 enc 覆盖 {len(encs)} 个"
                             f"（最大 {encs[-1] if encs else None}），调包流或错 enc")
        decoded = _ffmpeg_rawvideo(ffmpeg, stream, info["frames"], info["height"], info["width"],
                                   vf=AV1_DECODE_FILTER if lossy else None)
        if not lossy:
            enc_sha = [hashlib.sha256(f.tobytes()).hexdigest() for f in decoded]
            bad = [r["idx"] for r in rows if enc_sha[r["enc"]] != r["sha256"]]
            if bad:
                raise ValueError(f"解码画面与索引 sha256 不符：{s} idx={bad[:5]}（调包流或错 enc）")
        picked = _select_rows(rows, trace, s)
        frames = decoded[[r["enc"] for r in picked]]
        del decoded
        if lossy:
            if len(frames) != trace.source_frames:
                raise ValueError(f"缺帧：{s} 来源 {len(frames)} 帧，trace {trace.source_frames} 帧")
            checked[s] = len(frames)
        else:
            checked[s] = _verify_trace_hashes(trace, s, frames)
        out[s] = frames
    if out["front"].shape != out["wrist"].shape:
        raise ValueError("front/wrist 尺寸不同")
    return out["front"], out["wrist"], {"index_rows": checked, "raw_codec": codec, "lossy": lossy}


def decode_raw_orig(ep_dir: Path, src: Source, trace: TraceData) -> tuple:
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
            raise ValueError(f"原始流不可读：{p.name}")
        s = json.loads(proc.stdout)["streams"][0]
        total += int(s["width"]) * int(s["height"]) * 3
    return total


# ── 网站视频编码 ─────────────────────────────────────────────────────────────


def web_size(width: int, height: int) -> tuple:
    """与 imageio（macro_block_size=16）相同：宽高各自向上取到 16 的倍数。"""
    return ((width + MACRO_BLOCK - 1) // MACRO_BLOCK * MACRO_BLOCK,
            (height + MACRO_BLOCK - 1) // MACRO_BLOCK * MACRO_BLOCK)


def encode_web_mp4(frames: list, out: Path, *, ffmpeg: str | None = None, fps: int = FPS) -> None:
    """官方录像器画好的逐帧 RGB → 网站视频（AV1 4:2:0 MP4，faststart，30 fps）。尺寸不是 16 的倍数时与官方
    ``save_video`` 一样用 ``scale`` 缩放到取整后的尺寸。"""
    if not frames:
        raise ValueError("没有可编码的帧")
    h, w = frames[0].shape[:2]
    if any(f.shape != frames[0].shape or f.dtype != np.uint8 for f in frames):
        raise ValueError("逐帧尺寸或 dtype 不一致")
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
            raise RuntimeError(f"网站视频编码失败 rc={code}：{err.read().decode(errors='replace')[-800:]}")


# ── 渲染一局 ─────────────────────────────────────────────────────────────────


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
    """每局原子产出官方版式视频与身份清单 ``render.json``；旧结果只有全链路指纹一致才复用。

    缺省（旧 CLI 口径）：视频写 ``<ep_dir>/<out_subdir>/<prefix><Task>_ep<局号>_<终态>_<目标>_<tier>.mp4``，局号由旧
    局目录名 ``<key>.a<N>`` 推出。新产物树由 ``render_video`` 调用：显式给 ``output``（目标视频路径）、``sidecar``、
    ``episode_id`` 与 ``terminal``（结果行终态，覆盖 trace 推出的命名终态），不再要求目录名形如 ``<key>.a<N>``。
    无帧 error 局只写 sidecar（``render_status=no_frame`` 与原因），不出视频。"""
    ep_dir = Path(ep_dir).resolve()
    official_root = Path(official_root or default_official_root()).resolve()
    if Path(out_subdir).name != out_subdir or out_subdir in ("", ".", "..") or any(c in prefix for c in "/\\\x00"):
        raise ValueError("输出子目录或前缀不安全")
    if source not in SOURCE_MODES:
        raise ValueError(f"--source 只能是 {SOURCE_MODES}")
    if terminal is not None and terminal not in ("success", "fail", "timeout"):
        raise ValueError(f"terminal 只能是 success／fail／timeout：{terminal!r}")
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
        raise ValueError("输出目录不能为符号链接")
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
        raise ValueError("输出视频与清单不能为符号链接")
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
            raise ValueError("已有输出未通过 provenance 与完整视频验证；拒绝覆盖，请显式使用 overwrite")
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
            raise ValueError(f"源视频帧数/帧率/尺寸不符：{info}；期望 frames={trace.source_frames} fps=30 size=512x256")
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
            raise ValueError(f"预计内存 {estimated / 1024**2:.0f} MiB 超过上限 {max_memory_mib} MiB")
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
            raise ValueError("官方录像器帧数或逐帧尺寸不一致")
        temp = out_dir / f".render-{uuid.uuid4().hex}.mp4"
        try:
            encode_web_mp4(rec.total_images, temp, ffmpeg=ffmpeg)
            actual = probe_video(ffprobe, temp)
            shape = rec.total_images[0].shape
            expected_size = web_size(shape[1], shape[0])
            if (actual["frames"] != trace.output_frames or actual["fps"] != "30"
                    or (actual["width"], actual["height"]) != expected_size
                    or actual["codec"] != "av1" or actual["pix_fmt"] != "yuv420p"):
                raise ValueError(f"输出完整性不符：{actual}")
            if (fingerprint(trace_path) != source_trace or
                    (fingerprint(arrays_path) if arrays_path.exists() else None) != source_arrays or
                    src.media(ep_dir) != media):
                raise ValueError("源文件在重绘期间发生变化")
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
    """新产物树用：把 ``raw/<Task>_ep<N>_<tier>/`` 渲成
    ``videos/<Task>_ep<N>_<success|fail|timeout>_<task_goal>_<tier>.mp4``，清单写在局目录 ``render.json``；
    已有同名视频一律覆盖（产物以本次结果为准）。无帧局返回 None。"""
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
