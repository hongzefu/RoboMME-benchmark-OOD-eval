#!/usr/bin/env python3
"""v8 站点：把 V8 双模型评估录像（FFV1 无损 mkv）转成浏览器可播的 H.264 mp4（只读录像目录，不跑仿真）。

评估录像器（``src/robomme_ood_eval/record/recorder.py``）每局两路 ``front.mkv``／``wrist.mkv``，同一流里 sha256 相同的帧只编码
一份，``frames-<stream>.jsonl`` 逐帧记 ``idx``（原始帧序号）、``enc``（该帧在 mkv 里的编码序号）、``tag``（``reset``
＝初始观测与演示段，``step<k>``＝执行第 k 步）。直接转码 mkv 会丢掉重复帧、时间轴错位，所以逐局：

1. ffmpeg 把两路 mkv 各解码为 rgb24 数组（编码帧序列）；
2. 按 ``frames-*.jsonl`` 的 ``idx → enc`` 展开回逐帧原图；
3. 前视、腕部左右拼成 512×256，左上角按 tag 写 ``DEMO``（reset）／``EXEC``（step），与 v7 xhard0 合成视频同一画法；
4. libx264 ``-preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart`` 以 30 fps 输出到
   ``<run-dir>/site-media/<policy>/<tier>/<task>/<key>.mp4``。

输入是 ``<run-dir>/report/video-index.jsonl`` 里 ``accepted`` 的行（每身份唯一权威终态的录像目录）。已存在且帧数等于
展开帧数的 mp4 直接跳过（可续跑）。逐局写 ``<run-dir>/site-media/manifest.jsonl``，结束打印
``V8_EVAL_TRANSCODE=PASS|FAIL episodes=<n> frame_mismatch=<n> stream_len_mismatch=<n> failed=<n>`` 与 ``EXIT_CODE=``。

    uv run --no-sync python dev-scripts/site/eval_transcode.py \\
        --run-dir artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01 --workers 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # dev-scripts/site/<本文件> → 仓根
FPS = 30
SIDE = 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return -1


def decode(mkv: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mkv), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, SIDE, SIDE, 3)


def records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r["idx"]))
    return rows


def render_one(job: tuple[dict, str]) -> dict:
    row, out_root = job
    rec_dir = REPO_ROOT / row["path"]
    out = Path(out_root) / row["policy"] / row["tier"] / row["task"] / f"{row['key']}.mp4"
    res = {"key": row["key"], "policy": row["policy"], "tier": row["tier"], "task": row["task"],
           "status": row.get("status"), "rec_dir": row["path"], "mp4": str(out.relative_to(REPO_ROOT))}
    try:
        fr, wr = records(rec_dir / "frames-front.jsonl"), records(rec_dir / "frames-wrist.jsonl")
        n = min(len(fr), len(wr))
        res.update(frames=n, front_records=len(fr), wrist_records=len(wr),
                   demo_frames=sum(1 for r in fr[:n] if str(r.get("tag", "")).startswith("reset")))
        if out.exists() and probe_frames(out) == n:
            res.update(skipped=True, mp4_frames=n, mp4_sha256=sha256(out))
            return res
        front, wrist = decode(rec_dir / "front.mkv"), decode(rec_dir / "wrist.mkv")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * SIDE}x{SIDE}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
        for f, w in zip(fr[:n], wr[:n]):
            frame = np.ascontiguousarray(np.concatenate([front[int(f["enc"])], wrist[int(w["enc"])]], axis=1))
            demo = str(f.get("tag", "")).startswith("reset")
            color = (255, 200, 0) if demo else (0, 230, 120)
            cv2.rectangle(frame, (0, 0), (58, 18), (0, 0, 0), thickness=-1)
            cv2.putText(frame, "DEMO" if demo else "EXEC", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg 编码失败")
        tmp.replace(out)
        res.update(mp4_frames=probe_frames(out), mp4_sha256=sha256(out))
    except Exception as e:  # noqa: BLE001 逐局失败如实记录，不中断其余局
        res.update(error=f"{type(e).__name__}: {e}"[:500], mp4_frames=-1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 n 局（冒烟用）")
    args = ap.parse_args()
    run = args.run_dir.resolve()
    rows = [json.loads(line) for line in (run / "report" / "video-index.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("accepted")]
    rows.sort(key=lambda r: (r["policy"], r["tier"], r["task"], r["key"]))
    if args.limit:
        rows = rows[: args.limit]
    out_root = run / "site-media"
    out_root.mkdir(parents=True, exist_ok=True)
    done, mismatch, stream_mismatch, failed = [], 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(render_one, [(r, str(out_root)) for r in rows], chunksize=1), 1):
            done.append(res)
            if res.get("error"):
                failed += 1
                print(f"# 失败 {res['policy']} {res['key']}: {res['error']}", flush=True)
            elif res["mp4_frames"] != res["frames"]:
                mismatch += 1
                print(f"# 帧数不符 {res['mp4']} frames={res['frames']} mp4={res['mp4_frames']}", flush=True)
            if res.get("front_records") != res.get("wrist_records"):
                stream_mismatch += 1
            if i % 50 == 0 or i == len(rows):
                print(f"TRANSCODE done={i}/{len(rows)} failed={failed}", flush=True)
    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for res in sorted(done, key=lambda r: (r["policy"], r["tier"], r["task"], r["key"])):
            fh.write(json.dumps(res, ensure_ascii=False, sort_keys=True) + "\n")
    ok = not (mismatch or failed or stream_mismatch) and len(done) == len(rows)
    print(f"V8_EVAL_TRANSCODE={'PASS' if ok else 'FAIL'} episodes={len(done)} frame_mismatch={mismatch} "
          f"stream_len_mismatch={stream_mismatch} failed={failed}", flush=True)
    print(f"EXIT_CODE={0 if ok else 1}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
