"""逐局视频存储器（EpisodeRecorder）：原始数组永不降级，双相机图像 AV1 4:4:4 有损异步编码。

2026-10-08 拆分方案（§四「换 AV1 要改的地方」、红线 R8）起原始帧改存 AV1：编码参数固定为模块常量
``AV1_ENCODE_ARGS``（``libaom-av1 -cpu-used 4 -crf 24 -b:v 0 -pix_fmt yuv444p -g 300 -r 30 -row-mt 1 -threads 2``，
RGB→YUV 用 bt709 全范围），封装 MKV，``meta.json`` 记 ``raw_codec="av1-yuv444p"``（旧产物没有该字段，视为
``ffv1``）。有损编码不再要求「解码字节等于编码前 sha256」：收尾核验改为帧数、时间戳（逐帧 pts 恰为 i/fps）与可完整
解码三项，核验通过后才删原始块；不通过时从原始块同步重编码一次。动作与状态按原字节存 ``arrays.npz``，可精确恢复；
图像只能近似恢复。

设计要点（方案 §2.6、共享契约 recorder.py 一节）：

- 每帧在调用线程里算 sha256（编码前字节），逐流记 ``frames-<stream>.jsonl``（idx、sha256、tag、enc）；同一局
  同一流里 sha256 相同的帧只编码一份（``enc`` 指向已编码的那一帧），解码后按 ``enc`` 展开即得逐帧画面。
- 帧字节经有界队列交给写线程：队列满时 ``add_frames`` 阻塞（反压），从不丢帧；写线程先把原始字节追加到
  ``out_dir/.spool/<stream>.raw``，再经管道喂给 ffmpeg 子进程编码（编码在独立进程里，CPU 亲和由
  ``V75_ENCODE_CPUS`` 指定）。
- ``set_phase("reset")`` 期间只入队、落盘，不喂 ffmpeg（异步编码抢 CPU 会改变按墙钟计时的 RRT 规划）；
  切回 ``"run"`` 后写线程从原始块里补喂。
- ``close()``：哨兵入队带截止时间（``close_deadline_s``，缺省取环境变量 ``V75_RECORDER_JOIN_S``，再缺省 900 s）；
  队列满且 ffmpeg 卡在 stdin 时到点先杀编码子进程解开阻塞，不会无限等待。随后收尾编码 → 帧数／时间戳／可解码核验。
- 降级只看 ``V75_DATA_ROOT`` 所在盘的可用空间（600／150 GiB 两档），每次档位变化打印一行
  ``STORAGE_DEGRADE level= free_gib=``；``meta["baseline"]`` 或 ``meta["never_degrade"]`` 为真时始终 0 档。
  AV1 下 0 档与 1 档编码相同；2 档只留首尾各 50 帧图像（sha256 列表仍完整）。

本文件只依赖 numpy 与标准库，须能在 Python 3.10（SimpleMemVLA venv）与 3.11（benchmark、旧 FrameSamp+Modulation 客户端 venv）下导入。
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

LEVEL_THRESHOLDS_GIB = (600.0, 150.0)  # 低于前者进 1 档，低于后者进 2 档
LEVEL2_KEEP = 50  # 2 档时首尾各保留的图像帧数
_SENTINEL = object()
_LAST_LEVEL_LOCK = threading.Lock()
_LAST_LEVEL = 0  # 本进程上一次判定的降级档位（用于只在档位变化时打印一行）
_FFMPEG_CACHE: dict[str, Any] = {}
#: 原始帧编码（R8，不得改动）：libaom-av1、4:4:4、crf 24、速度档 4、GOP 300、30 fps、行多线程、2 线程
RAW_CODEC = "av1-yuv444p"
#: 旧产物（没有 raw_codec 字段）的编码
LEGACY_RAW_CODEC = "ffv1"
FPS = 30
AV1_ENCODER = "libaom-av1"
AV1_ENCODE_ARGS = ("-c:v", "libaom-av1", "-cpu-used", "4", "-crf", "24", "-b:v", "0", "-pix_fmt", "yuv444p",
                   "-g", "300", "-r", str(FPS), "-row-mt", "1", "-threads", "2")
#: RGB→YUV 用 bt709 全范围（编码侧），并在码流里打同样的标记，解码侧按标记还原
AV1_COLOR_FILTER = "scale=out_color_matrix=bt709:out_range=full"
AV1_COLOR_TAGS = ("-color_range", "pc", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709")
#: 解码 AV1 原始流回 rgb24 时的色彩还原（与编码侧对称）
AV1_DECODE_FILTER = "scale=in_color_matrix=bt709:in_range=full,format=rgb24"
#: close() 收尾等待写线程的缺省截止时间（秒）
DEFAULT_CLOSE_DEADLINE_S = 900.0


def _trace_writer():
    """同目录 ``trace_writer``（``merge_write_npz`` 的唯一实现）：已导入则复用，否则按文件路径加载。"""
    mod = sys.modules.get("trace_writer")
    if mod is not None and hasattr(mod, "merge_write_npz"):
        return mod
    import importlib.util

    # 已注册的同名模块缺该函数（旧副本）时另起私有名加载，不顶替别人已导入的模块
    name = "trace_writer" if mod is None else "_recorder_trace_writer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "trace_writer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dumps(obj: Any) -> str:
    """全项目统一的 jsonl 行格式。"""
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
    """单帧（H,W,3 uint8）的 sha256：按 C 连续字节计算，与代理、客户端钩子、对拍工具口径一致。"""
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """数值数组的 sha256：字节 + dtype + shape，一起进摘要。"""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(a.tobytes())
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- ffmpeg 与降级判定

def find_ffmpeg() -> str:
    """找支持 libaom-av1 编码的 ffmpeg：V75_FFMPEG > /usr/bin/ffmpeg > PATH > imageio_ffmpeg 自带。找不到即报错。"""
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
    raise RuntimeError(f"找不到支持 {AV1_ENCODER} 的 ffmpeg，候选：{cands}")


class RecorderError(RuntimeError):
    """录制失败（写线程死亡、编码核配置错误等）。钩子捕获后该局录制记 FAIL，评估本身不受影响。"""


def parse_cpu_list(text: str) -> set[int]:
    """解析 taskset 风格的 CPU 列表："3"、"2-3"、"1,5-6"。"""
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
    """校验 V75_ENCODE_CPUS 是本进程 CPU 亲和集合的子集；不合法即 RecorderError（尽早失败）。返回规范值或空串。"""
    cpus = os.environ.get("V75_ENCODE_CPUS", "").strip()
    if not cpus:
        return ""
    try:
        want = parse_cpu_list(cpus)
    except ValueError as e:
        raise RecorderError(f"V75_ENCODE_CPUS 无法解析：{cpus!r}（{e}）")
    allowed = os.sched_getaffinity(0)
    if not want or not want <= allowed:
        raise RecorderError(f"V75_ENCODE_CPUS={cpus} 不在本进程 CPU 亲和集合 {sorted(allowed)} 之内")
    if not shutil.which("taskset"):
        raise RecorderError("设了 V75_ENCODE_CPUS 但找不到 taskset")
    return cpus


def _cpu_prefix() -> list[str]:
    """编码／解码子进程的 CPU 亲和前缀（V75_ENCODE_CPUS，例如 "3" 或 "2-3"；已在 recorder 初始化时校验）。"""
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
    """档位与上一次不同就打印一行 STORAGE_DEGRADE（进程内去重）。"""
    global _LAST_LEVEL
    with _LAST_LEVEL_LOCK:
        if level != _LAST_LEVEL:
            print(f"STORAGE_DEGRADE level={level} free_gib={free_gib:.1f}", flush=True)
            _LAST_LEVEL = level


class _Empty(Exception):
    pass


class _BoundedQueue:
    """有界阻塞队列（不用标准库 queue：拆仓前旧评估脚本目录里的同名 queue.py 在 sys.path[0] 时会遮蔽它，沿用此写法）。"""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._items: collections.deque = collections.deque()
        self._cv = threading.Condition()

    def put(self, item: Any, timeout: float | None = None) -> bool:
        """放入；队列满时等待（反压）。给了 timeout 时超时返回 False，由调用方检查写线程是否还活着。"""
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


# ---------------------------------------------------------------- 单流状态

class _Stream:
    """一路相机流：逐帧记录（调用线程写，持 recorder 锁）与编码状态（写线程独占）。"""

    def __init__(self, name: str, shape: tuple, spool_dir: Path):
        self.name = name
        self.shape = shape  # (H, W, C)
        self.frame_bytes = int(np.prod(shape))
        self.records: list[dict] = []  # 逐帧 {idx, sha256, tag, enc}
        self.sha2enc: dict[str, int] = {}
        self.enc_sha: list[str] = []  # 编码序号 → sha256
        self.n_enqueued = 0
        self.tail: collections.deque = collections.deque(maxlen=LEVEL2_KEEP)  # 2 档：暂存尾部 (idx, sha, bytes)
        # 以下由写线程独占
        self.spool_path = spool_dir / f"{name}.raw"
        self.spool_w = None
        self.n_spooled = 0
        self.n_fed = 0
        self.proc: subprocess.Popen | None = None
        self.proc_rusage_cpu = 0.0
        self.reordered = 0
        self.encode_error: str | None = None


class EpisodeRecorder:
    """一局的录制器，线程安全。用法见模块说明；``close()`` 必须调用一次。"""

    def __init__(self, out_dir: str | Path, meta: dict, *, lossless: bool = True, encode_async: bool = True,
                 queue_frames: int = 64, fps: int = 30,
                 free_gib_fn: Callable[[str], float] | None = None,
                 encode_delay_s: float = 0.0, overwrite: bool = False, close_deadline_s: float | None = None):
        self.out_dir = Path(out_dir)
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"录制目录非空，拒绝覆盖（需显式 overwrite=True）：{self.out_dir}")
            shutil.rmtree(self.out_dir)
        self.encode_cpus = check_encode_cpus()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.spool_dir = self.out_dir / ".spool"
        self.spool_dir.mkdir(exist_ok=True)
        self.meta = dict(meta)
        self.fps = int(fps)
        self.encode_async = bool(encode_async)
        self._encode_delay_s = float(encode_delay_s)  # 仅供单测模拟慢编码
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
        self._run_phase = threading.Event()  # 置位 = run（可编码）；清位 = reset（只入队）
        self._phase = "reset"
        self._queue_wait_s = 0.0
        self._closed = False
        self._writer_error: str | None = None  # 写线程异常（如 ENOSPC）；一旦置位，add_* 抛 RecorderError
        self._discarded = 0  # 写线程失败后丢弃的入队帧数（计入 dropped）
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

    # ------------------------------------------------------------ 公共接口

    def set_phase(self, phase: str) -> None:
        if phase not in ("reset", "run"):
            raise ValueError(f"phase 只能是 reset／run：{phase}")
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
            raise ValueError(f"add_frames 需 uint8 (N,H,W,C)/(H,W,C)，得到 {arr.dtype} {arr.shape}")
        idxs: list[int] = []
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder 已 close")
            self._check_writer()
            st = self._streams.get(stream)
            if st is None:
                st = _Stream(stream, tuple(arr.shape[1:]), self.spool_dir)
                self._streams[stream] = st
            if tuple(arr.shape[1:]) != st.shape:
                raise ValueError(f"流 {stream} 帧形状变化：{st.shape} → {arr.shape[1:]}")
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
                    st.tail.append((idx, sha, b))  # 收尾时再决定是否属于末 50 帧
                else:
                    rec["enc"] = self._enqueue(st, sha, b)
        return idxs

    def add_array(self, name: str, arr, *, step: int | None = None) -> None:
        a = np.array(arr, copy=True)  # 复制主机端数据，不持有调用方的引用
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
                raise RuntimeError("recorder 重复 close")
            # 2 档：尾部暂存里的帧即末 50 帧，此时补编码
            try:
                for st in self._streams.values():
                    for idx, sha, b in list(st.tail):
                        rec = st.records[idx]
                        rec["enc"] = st.sha2enc[sha] if sha in st.sha2enc else self._enqueue(st, sha, b)
                    st.tail.clear()
            except RecorderError as e:  # 写线程已失败：close 不抛、不阻塞，结果记 FAIL
                self._writer_error = self._writer_error or str(e)
            self._closed = True
        self.set_phase_run_for_close()
        if self._writer is not None:
            # 哨兵入队与等待写线程共用一个截止时间：队列满、ffmpeg 卡在 stdin 时写线程永远取不走队列，
            # 旧写法的「入队成功才 join」会无限循环。到点先杀编码子进程（写线程的 stdin.write 随即报错返回），
            # 再给 30 s 让写线程把队列取空、收到哨兵退出。
            deadline = time.monotonic() + self.close_deadline_s
            sent = False
            while self._writer.is_alive() and time.monotonic() < deadline:
                if self._q.put(_SENTINEL, timeout=min(1.0, max(0.01, deadline - time.monotonic()))):
                    sent = True
                    break
            if sent:
                self._writer.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._writer.is_alive():  # 写线程卡死（如 ffmpeg 不读管道）：杀编码子进程，记 FAIL，不阻塞
                self._writer_error = self._writer_error or "写线程收尾超时"
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
        # 解码核对；不符就从原始块重编码一次
        t_verify = time.time()
        verify: dict[str, dict] = {}
        reencoded = 0
        for st in self._streams.values():
            v = self._verify_stream(st)
            # 写线程收尾超时（编码子进程已被杀）时不再重编码：重编码同样可能卡住，收尾必须有界
            if (v["mismatch"] or v["error"]) and not self._writer_error:
                reencoded += 1
                try:
                    self._reencode_from_spool(st)
                except Exception as e:  # 原始块缺失／损坏：记错误，close 不抛
                    st.encode_error = f"重编码失败 {type(e).__name__}: {e}"
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
        # 逐帧清单、原始数组、摘要落盘
        for st in self._streams.values():
            with open(self.out_dir / f"frames-{st.name}.jsonl", "w", encoding="utf-8") as fh:
                for rec in st.records:
                    fh.write(dumps(rec) + "\n")
        try:
            self._write_arrays()
        except Exception as e:  # noqa: BLE001 数组写盘失败（含与 trace 同键不一致）：不抛，记 FAIL
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

    # ------------------------------------------------------------ 内部：入队、写线程、编码

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
        """收尾时强制允许编码（不写 phase 事件，事件文件稍后关闭）。"""
        self._run_phase.set()

    def _enqueue(self, st: _Stream, sha: str, b: bytes) -> int:
        """持锁调用：分配编码序号并阻塞入队（队列满即反压）。"""
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
        """写线程已失败或已死（非收尾）即抛 RecorderError，调用方不会永久阻塞。"""
        if self._writer_error:
            raise RecorderError(f"录制写线程失败：{self._writer_error}")
        if self._writer is not None and not self._writer.is_alive() and not self._closed:
            self._writer_error = "录制写线程已退出"
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
            if self._writer_error:  # 已失败：继续取走队列元素让调用方解除阻塞，计为丢弃
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
            except BaseException as e:  # 例如 ENOSPC：记下错误，之后 add_* 抛 RecorderError
                self._writer_error = f"{type(e).__name__}: {e}"
        # 收尾：全部喂完、关管道、等 ffmpeg
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
            raise ValueError(f"只支持 3 通道图像：{st.shape}")
        return av1_encode_cmd(self.ffmpeg, w, h, out, fps=self.fps, prefix=_cpu_prefix())

    def _err_path(self, st: _Stream) -> Path:
        return self.out_dir / f".ffmpeg-{st.name}.log"

    def _start_proc(self, st: _Stream) -> None:
        out = self.out_dir / f"{st.name}.mkv"
        with open(self._err_path(st), "ab") as err:  # stderr 写文件，不用管道（避免管道写满卡死）
            st.proc = subprocess.Popen(self._ffmpeg_encode_cmd(st, out), stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=err)

    def _feed_some(self, max_frames: int) -> bool:
        """从原始块往 ffmpeg 补喂至多 max_frames 帧；返回是否还有没喂完的。"""
        left = False
        for st in list(self._streams.values()):
            if st.encode_error:
                continue
            n = min(st.n_spooled - st.n_fed, max_frames)
            if n <= 0:
                continue
            try:
                self._feed(st, n)
            except Exception as e:  # 编码失败不影响录制本身；close 时从原始块重编码
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
            raise IOError(f"原始块读回长度不符 {len(data)} != {n * st.frame_bytes}")
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
        """从原始块同步重编码整路流（编码失败或解码不符时调用，不重跑环境）。"""
        out = self.out_dir / f"{st.name}.mkv"
        with open(st.spool_path, "rb") as fh, open(self._err_path(st), "ab") as ef:
            p = subprocess.Popen(self._ffmpeg_encode_cmd(st, out), stdin=fh,
                                 stdout=subprocess.DEVNULL, stderr=ef)
            rc, cpu, err = self._wait_proc(p, self._err_path(st))
        st.proc_rusage_cpu = cpu
        st.encode_error = None if rc == 0 else f"重编码 ffmpeg rc={rc}: {err}"

    def _verify_stream(self, st: _Stream) -> dict:
        """AV1 有损：完整解码一遍（framemd5），核帧数等于编码序号数、逐帧时间戳恰为 i/fps；不再比字节。"""
        res = {"mismatch": 0, "decoded": 0, "error": st.encode_error, "timestamps_ok": False}
        if st.n_enqueued == 0:
            return res
        out = self.out_dir / f"{st.name}.mkv"
        if not out.exists():
            res["error"] = res["error"] or "mkv 不存在"
            res["mismatch"] = st.n_enqueued
            return res
        try:
            frames = probe_decoded_frames(self.ffmpeg, out)
        except Exception as e:
            res["error"] = f"解码失败 {type(e).__name__}: {e}"
            res["mismatch"] = st.n_enqueued
            return res
        res["decoded"] = len(frames)
        res["mismatch"] = abs(len(frames) - st.n_enqueued)
        bad_ts = [i for i, t in enumerate(frames) if abs(t - i / self.fps) > 0.5 / self.fps]
        res["timestamps_ok"] = not bad_ts
        if bad_ts:
            res["error"] = f"时间戳不符 {len(bad_ts)} 帧（首个 idx={bad_ts[0]} t={frames[bad_ts[0]]:.4f}）"
        elif res["mismatch"] == 0:
            res["error"] = None  # 帧数、时间戳、可完整解码三项通过即以此为准
        return res

    def _write_arrays(self) -> None:
        with self._lock:
            items = list(self._arrays)
        if not items:
            return
        payload = {f"{name}__{k:05d}": a for name, k, _s, a in items}
        # 第三阶段（冻结说明四.4）：唯一写法 merge_write_npz，与同目录 trace 的收尾先后任意都不互相覆盖；
        # 同键不一致抛 ArraysConflict，由 close() 记进 errors（RECORDER_VERIFY=FAIL）
        _trace_writer().merge_write_npz(self.out_dir / "arrays.npz", payload)
        with open(self.out_dir / "arrays-index.jsonl", "w", encoding="utf-8") as fh:
            for name, k, step, a in items:
                fh.write(dumps({"key": f"{name}__{k:05d}", "name": name, "k": k, "step": step,
                                "dtype": a.dtype.str, "shape": list(a.shape), "sha256": array_sha256(a)}) + "\n")


# ---------------------------------------------------------------- 读回工具（对拍用）

def av1_encode_cmd(ffmpeg: str, w: int, h: int, out: str | Path, *, fps: int = FPS,
                   prefix: list[str] | None = None) -> list[str]:
    """rgb24 原始帧（管道输入）→ AV1 4:4:4 MKV 的完整命令（R8 参数）。"""
    return list(prefix or []) + [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                                 "-i", "pipe:0", "-an", "-vf", AV1_COLOR_FILTER, *AV1_ENCODE_ARGS,
                                 *AV1_COLOR_TAGS, str(out)]


def probe_decoded_frames(ffmpeg: str, path: str | Path) -> list[float]:
    """完整解码一遍（``-f framemd5``，只用 ffmpeg），返回逐帧显示时间（秒）；解码失败抛 RuntimeError。"""
    cmd = _cpu_prefix() + [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(path),
                           "-map", "0:v:0", "-f", "framemd5", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败 rc={p.returncode}: {p.stderr.decode(errors='replace')[-1000:]}")
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
                raise RuntimeError("framemd5 输出缺 #tb 行")
            out.append(int(parts[2]) * tb)
    return out


def raw_codec_of(out_dir: str | Path) -> str:
    """读局目录 ``meta.json`` 的 ``raw_codec``；没有该字段（旧产物）视为 ``ffv1``。"""
    try:
        meta = json.loads((Path(out_dir) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LEGACY_RAW_CODEC
    return str(meta.get("raw_codec") or LEGACY_RAW_CODEC)


def decode_raw_frames(ffmpeg: str, path: str | Path, shape: tuple, *, raw_codec: str = RAW_CODEC) -> list[bytes]:
    """把 mkv 解码成 rgb24 原始帧字节列表；AV1 流按 bt709 全范围还原色彩，旧 FFV1（gbrp）直接转。"""
    h, w, c = shape
    conv = ["-vf", AV1_DECODE_FILTER] if raw_codec != LEGACY_RAW_CODEC else []
    cmd = _cpu_prefix() + [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(path), *conv,
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败 rc={p.returncode}: {p.stderr.decode(errors='replace')[-1000:]}")
    fb = h * w * c
    data = p.stdout
    if len(data) % fb:
        raise RuntimeError(f"解码字节数 {len(data)} 不是帧大小 {fb} 的整数倍")
    return [data[i:i + fb] for i in range(0, len(data), fb)]


def load_frames(out_dir: str | Path, stream: str) -> tuple[list[dict], list[np.ndarray | None]]:
    """读回一路流：返回 (逐帧记录, 逐帧图像)；2 档里没存的帧为 None。按 ``meta.json`` 的 ``raw_codec`` 选解码方式。"""
    out_dir = Path(out_dir)
    recs = [json.loads(l) for l in (out_dir / f"frames-{stream}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    mkv = out_dir / f"{stream}.mkv"
    if not recs or not mkv.exists():
        return recs, [None] * len(recs)
    # 形状从 ffprobe 太重，直接用 meta 外的首帧无法得知；约定 256x256 以外时调用方传形状
    probe = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(mkv)], capture_output=True, text=True).stderr
    import re

    m = re.search(r"Video: .*?, (\d+)x(\d+)", probe)
    if not m:
        raise RuntimeError(f"读不出 {mkv} 的分辨率")
    w, h = int(m.group(1)), int(m.group(2))
    dec = decode_raw_frames(find_ffmpeg(), mkv, (h, w, 3), raw_codec=raw_codec_of(out_dir))
    imgs = [np.frombuffer(dec[r["enc"]], np.uint8).reshape(h, w, 3) if r["enc"] is not None else None for r in recs]
    return recs, imgs


def verdict_line(res: dict) -> str:
    return (f"RECORDER_VERIFY={res['RECORDER_VERIFY']} frames={res['frames']} decode_mismatch={res['decode_mismatch']} "
            f"dropped={res['dropped']} reordered={res['reordered']} bytes={res['bytes']} level={res['level']}")


def _main(argv: list[str]) -> int:
    """命令行：``recorder.py load <out_dir> <stream>`` 读回并核对（调试用）。旧 FFV1 产物核逐帧 sha256；AV1 有损，
    只核每条记录都能取到图像。"""
    if len(argv) == 3 and argv[0] == "load":
        recs, imgs = load_frames(argv[1], argv[2])
        lossy = raw_codec_of(argv[1]) != LEGACY_RAW_CODEC
        bad = sum(1 for r, im in zip(recs, imgs)
                  if im is not None and not lossy and frame_sha256(im) != r["sha256"])
        kept = sum(im is not None for im in imgs)
        print(f"RECORDER_LOAD={'PASS' if bad == 0 else 'FAIL'} frames={len(recs)} kept={kept} mismatch={bad}")
        return 0 if bad == 0 else 1
    print("用法：recorder.py load <out_dir> <stream>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
