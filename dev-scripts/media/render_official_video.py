#!/usr/bin/env python3
"""官方版式重绘的命令行入口：评估包 ``robomme_ood_eval.record.official_render`` 的薄 CLI。

拆仓（1008 拆分方案第二部分 §二 ``dev-scripts/media/`` 行）后，来源选择 → 解码 → 核验 → 官方录像器逐帧绘制 → 编码 →
复核的全部实现都在库里（R3：官方 ``RolloutRecorder`` 只从锁定的 ``third_party/mme-vla/examples/robomme/utils.py``
原文件加载）；本脚本只负责：挑局目录（位置参数或 ``--root`` + ``--label``）、按内存与 CPU 定并发、多进程调库函数
``render_episode``、逐局打印判定行与汇总行。库里的公开名字（``load_trace``、``feed_official_recorder``、
``select_source``、``safe_filename`` 等）在本模块重新导出，旧调用方按模块属性取用时不变。

网站视频编码按 R8（AV1 4:2:0 封装 MP4，faststart，30 fps）；原始帧按局目录 ``meta.json`` 的 ``raw_codec`` 分版本读
（AV1 有损不再校验解码字节 sha256，旧 FFV1 照旧逐帧核验）。

用法::

    python dev-scripts/media/render_official_video.py <局目录>... [--source {auto,mp4,raw}] [--out-subdir official]
    python dev-scripts/media/render_official_video.py --root <运行根> --label <路线名> [--jobs N --threads 2]

判定行：每局 ``OFFICIAL_RENDER=PASS|NO_FRAME|FAIL dir=…``，末行 ``OFFICIAL_RENDER_SUMMARY=PASS|FAIL total= ok= fail=
reused= no_frame=``。退出码 0 全部成功、1 有失败、2 参数错。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from robomme_ood_eval.record import official_render as _lib  # noqa: E402
from robomme_ood_eval.record.official_render import (  # noqa: E402,F401  重新导出：旧调用方按模块属性取用
    FPS, RAW_KINDS, SCHEMA, SOURCE_MODES, STREAMS, TERMINALS, WEB_ENCODE_ARGS, Source, TraceData, decode_mp4,
    decode_raw_new, decode_raw_orig, default_official_root, encode_web_mp4, f32_from_record, feed_official_recorder,
    find_ffprobe, fingerprint, load_official, load_trace, official_episode_id, probe_stream, probe_video,
    read_frame_index, render_episode, render_video, safe_filename, select_source, web_size)


def _configure_threads(threads: int, cpus: list[int], worker_index: int) -> None:
    """每个渲染进程只用 ``threads`` 个线程与对应的 CPU 子集（编码子进程继承亲和性）。"""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    try:
        import cv2

        cv2.setNumThreads(threads)
    except ImportError:  # pragma: no cover - 官方 utils 依赖 cv2，缺失时渲染本身会报错
        pass
    if hasattr(os, "sched_setaffinity") and cpus:
        start = worker_index * threads
        os.sched_setaffinity(0, set(cpus[start:start + threads]) or set(cpus))


def _init_worker(threads: int, cpus: list[int], counter) -> None:
    """共享计数器分配唯一工作进程编号；每个进程只配置一次，不按 PID 碰撞。"""
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    _configure_threads(threads, cpus, index)


def _worker(ep_dir: str, opts: dict) -> dict:
    try:
        return {"ok": True, **_lib.render_episode(Path(ep_dir), **opts)}
    except Exception as exc:  # noqa: BLE001 逐局失败如实记录
        return {"ok": False, "dir": Path(ep_dir).name, "reason": f"{type(exc).__name__}: {exc}"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*", type=Path, help="局目录（含 trace.jsonl）")
    parser.add_argument("--root", type=Path, help="按 --label 递归找局目录的根（与位置参数二选一）")
    parser.add_argument("--label", default="groundsg-ground-sg-oracle", help="--root 模式下路径里须含的路线名")
    parser.add_argument("--official-root", type=Path, default=None,
                        help=f"官方仓根（含 {_lib.OFFICIAL_UTILS_REL}）；缺省取环境变量 {_lib.ENV_OFFICIAL_ROOT} 或评估仓根")
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
    return parser


def main(argv=None) -> int:
    parser = build_parser()
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
    opts = {k: getattr(args, k) for k in ("official_root", "ffmpeg", "out_subdir", "prefix", "overwrite",
                                          "max_memory_mib", "source", "key")}
    available = next((int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:")), args.max_memory_mib * 1024**2)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    if args.threads > len(cpus):
        parser.error("--threads 超过可用 CPU 数")
    jobs = min(args.jobs, len(dirs), max(1, int(available * 0.6) // (args.max_memory_mib * 1024**2)),
               max(1, len(cpus) // args.threads))
    print(f"OFFICIAL_RENDER_RESOURCES jobs={jobs} requested_jobs={args.jobs} threads={args.threads} "
          f"max_memory_mib={args.max_memory_mib}", flush=True)
    ok = fail = reused = no_frame = 0
    counter = multiprocessing.Value("i", 0)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(args.threads, cpus, counter)) as pool:
        futures = {pool.submit(_worker, str(p), dict(opts)): p for p in dirs}
        for f in as_completed(futures):
            try:
                result = f.result()
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "dir": futures[f].name, "reason": f"{type(exc).__name__}: {exc}"}
            if result["ok"] and result["render_status"] == "no_frame":
                ok += 1
                no_frame += 1
                reused += bool(result.get("reused"))
                print(f"OFFICIAL_RENDER=NO_FRAME dir={result['dir']} source_kind=none "
                      f"terminal_reason={result['terminal_reason']} end_status={result['status']} "
                      f"steps={result['exec_steps']} reason={json.dumps(result['no_frame_reason'], ensure_ascii=False)}",
                      flush=True)
            elif result["ok"]:
                ok += 1
                reused += result["render_status"] == "reused"
                print(f"OFFICIAL_RENDER=PASS dir={result['dir']} frames={result['frames']} demo={result['demo_frames']} "
                      f"steps={result['exec_steps']} omitted={result['omitted_timeout_frames']} "
                      f"size={result['width']}x{result['height']} status={result['render_status']} "
                      f"source_kind={result.get('source_kind', 'mp4')} terminal_reason={result['terminal_reason']} "
                      f"end_status={result['status']} out={json.dumps(result['out'])}", flush=True)
            else:
                fail += 1
                print(f"OFFICIAL_RENDER=FAIL dir={result['dir']} reason={json.dumps(result['reason'], ensure_ascii=False)}",
                      flush=True)
    print(f"OFFICIAL_RENDER_SUMMARY={'PASS' if fail == 0 else 'FAIL'} total={len(dirs)} ok={ok} fail={fail} "
          f"reused={reused} no_frame={no_frame}", flush=True)
    return int(fail > 0)


if __name__ == "__main__":
    raise SystemExit(main())
