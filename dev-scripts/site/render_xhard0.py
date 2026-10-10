#!/usr/bin/env python3
"""v7 对照站点：从 xhard0 生成 h5 离线合成演示视频（0929 站点方案 §3；只读 h5，不跑仿真、不碰录像器）。

xhard0 新旧两个入口只留下 h5、没有 mp4：
- ``O``（旧入口，官方 ``robomme`` dataset-gen）：``artifacts/newtask-v7/parity/h5/O-xhard0-bucket``
- ``H``（新入口，``robomme_ood``）：``artifacts/newtask-v7/parity/h5/H-xhard0``

逐局按 ``timestep_<k>`` 升序读 ``obs/front_rgb`` 与 ``obs/wrist_rgb``（各 256×256×3 uint8），左右拼成 512×256，
左上角按 ``info/is_video_demo`` 写 ``DEMO``／``EXEC``，经 ffmpeg libx264（yuv420p、faststart）以 30 fps 输出到
``<out>/<side>/<task>/<task>_xhard0_<seed>.mp4``。真实渲染帧，原分辨率输出、不缩放。

已存在且帧数等于 timestep 数的视频直接跳过（可续跑）。生成失败留下的空 h5（VideoPlaceOrder seed 610701、611101，
官方原版即失败，两入口相同）记 ``generation_failed``、不出视频。逐局写 ``<out>/manifest-<side>.jsonl``
（task／seed／h5 相对路径／h5 sha256／帧数／mp4），结束打印
``XHARD0_RENDER=PASS|FAIL sides=<n> episodes=<每侧局数>x<n> frame_mismatch=<n>`` 与 ``EXIT_CODE=``。

    uv run --no-sync python dev-scripts/site/render_xhard0.py --side O --side H --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # dev-scripts/site/<本文件> → 仓根
SIDES = {
    "O": REPO_ROOT / "artifacts/newtask-v7/parity/h5/O-xhard0-bucket",
    "H": REPO_ROOT / "artifacts/newtask-v7/parity/h5/H-xhard0",
}
DEFAULT_OUT = REPO_ROOT / "artifacts/newtask-v7/site-media/xhard0-gen"
NAME = re.compile(r"(?P<task>[A-Za-z]+)_ep(?P<ep>\d+)_seed(?P<seed>\d+)\.h5\Z")
FPS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return -1


def timesteps(episode: h5py.Group) -> list[str]:
    keys = [k for k in episode if k.startswith("timestep_")]
    return sorted(keys, key=lambda k: int(k.split("_", 1)[1]))


def render_one(job: tuple[str, str, str]) -> dict:
    side, h5_path, out_root = job
    h5 = Path(h5_path)
    m = NAME.match(h5.name)
    task, seed = m["task"], int(m["seed"])
    out = Path(out_root) / side / task / f"{task}_xhard0_{seed}.mp4"
    with h5py.File(h5, "r") as handle:
        if not list(handle):
            # 生成失败留下的空 h5（官方原版在该 seed 上即 DatasetGenerationError，两个入口相同）：不出视频，照实记录
            return {"side": side, "task": task, "seed": seed, "episode": int(m["ep"]),
                    "h5": str(h5.relative_to(REPO_ROOT)), "h5_sha256": sha256(h5), "frames": 0, "demo_frames": 0,
                    "mp4": None, "mp4_frames": 0, "generation_failed": True}
        (ep_name,) = list(handle)
        episode = handle[ep_name]
        keys = timesteps(episode)
        record = {"side": side, "task": task, "seed": seed, "episode": int(m["ep"]),
                  "h5": str(h5.relative_to(REPO_ROOT)), "h5_sha256": sha256(h5), "frames": len(keys),
                  "demo_frames": 0, "mp4": str(out.relative_to(REPO_ROOT))}
        demo_flags = [bool(episode[k]["info/is_video_demo"][()]) for k in keys]
        record["demo_frames"] = int(sum(demo_flags))
        if out.exists() and probe_frames(out) == len(keys):
            record["skipped"] = True
            record["mp4_frames"] = len(keys)
            return record
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x256", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
        for key, demo in zip(keys, demo_flags):
            obs = episode[key]["obs"]
            frame = np.ascontiguousarray(np.concatenate([obs["front_rgb"][()], obs["wrist_rgb"][()]], axis=1))
            label = "DEMO" if demo else "EXEC"
            color = (255, 200, 0) if demo else (0, 230, 120)
            cv2.rectangle(frame, (0, 0), (58, 18), (0, 0, 0), thickness=-1)
            cv2.putText(frame, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg 失败：{h5}")
        tmp.replace(out)
    record["mp4_frames"] = probe_frames(out)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", action="append", choices=sorted(SIDES), required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="每侧只处理前 n 局（冒烟用）")
    args = ap.parse_args()
    args.out = args.out.resolve()

    per_side: dict[str, int] = {}
    failed: dict[str, int] = {}
    mismatch = 0
    for side in args.side:
        files = sorted(p for p in SIDES[side].rglob("*.h5") if NAME.match(p.name))
        if args.limit:
            files = files[: args.limit]
        per_side[side] = len(files)
        jobs = [(side, str(p), str(args.out)) for p in files]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(render_one, jobs), 1):
                rows.append(row)
                failed[side] = failed.get(side, 0) + bool(row.get("generation_failed"))
                if row["mp4_frames"] != row["frames"]:
                    mismatch += 1
                    print(f"# 帧数不符 {row['mp4']} h5={row['frames']} mp4={row['mp4_frames']}", flush=True)
                if i % 16 == 0 or i == len(jobs):
                    print(f"RENDER side={side} done={i}/{len(jobs)}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (r["task"], r["seed"]))
        (args.out / f"manifest-{side}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    counts = set(per_side.values())
    ok = mismatch == 0 and len(counts) == 1 and (args.limit or counts == {192})
    print(f"XHARD0_RENDER={'PASS' if ok else 'FAIL'} sides={len(per_side)} "
          f"episodes={'/'.join(str(v) for v in per_side.values())}x{len(per_side)} "
          f"rendered={'/'.join(str(per_side[k] - failed.get(k, 0)) for k in per_side)} "
          f"generation_failed={'/'.join(str(failed.get(k, 0)) for k in per_side)} frame_mismatch={mismatch}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    print(f"EXIT_CODE={code}", flush=True)
    sys.exit(code)
