"""评估视频搬运（V8／V9 评估录像）：NFS 暂存 → 本机 /data，与评估解耦、天然可续。

原 v7 单文件模式（``--policy``／``--records-glob``／``--identities``／``--tier``）已于维护计划 W2 删除，``--mode`` 只剩 ``v8``（缺省）。

``--mode v8``（1001-v8-post-evaluation-gl-plan.md 第一部分 §1 第 7 条、契约 C4 第二段；缺省即此模式）：

    python dev-scripts/gl/eval_video_mover.py --mode v8 --stage <NFS 运行根> \\
        --dest artifacts/v8-evaluation/<R>/videos [--policies smvla,perceptual-framesamp-modul] [--once] [--stop-file F] [--interval S]

扫 ``<stage>/sNN/<policy>/results.jsonl`` 的 V8 结果行（``v8: true``，含 error／infra／金丝雀尝试与迟到终态），按 ``rec_dir``
的目录名（``<key>.a<n>``、``<key>.canary.a<n>``）找 ``<stage>/sNN/<policy>/rec/<目录名>/``；只搬「结果行已写出」的目录（进行中的
不碰），且目录内文件 ``--stable-sec`` 秒无变化、没有 rsync 临时文件（``.<name>.<6 位随机>``）。录像器自己留下的点开头文件
（非空 ``.ffmpeg-<stream>.log``、close 失败时的 ``.spool/``）照常整目录搬；NFS 删除占位 ``.nfs*`` 不搬不删。
``rec/`` 下点开头的目录（节点同步的 ``.incoming/`` 等）不碰、不算孤儿；节点同步落的 ``rec/<目录名>.dupN/`` 随同一结果行照常搬，
目的地沿用 ``.dupN`` 名、不覆盖。
整目录先 ``rsync -a`` 到 ``<dest>/.incoming/<随机>/`` → 逐文件两端 sha256 相同 → 原子改名到 ``<dest>/<policy>/<tier>/<task>/<目录名>/``
（目标已存在且内容相同视为已搬；内容不同则落 ``<目标>.dupN``，本机已有副本一律不删）→ 只删核对过的源文件 → 往
``<dest>/moved.jsonl`` 追加一行（``mode: "v8"``，含逐文件 sha256；源目录删不掉记 ``src_left: true``）；已搬的源不再存在，可续。
sha 不符：只删临时目录、保留 NFS 源，计 sha_mismatch（常驻模式下一轮重试）。常驻模式每 ``--interval`` 秒打印
``VMOVE mode=v8 moved= pending= bytes= stage_bytes= sha_mismatch=``，``--stop-file`` 出现且无待搬时退出（结果行未写出的目录不碰）。
``--once``：搬到无待稳定目录后（单个目录持续不稳超过 ``--once-max-wait`` 秒即放弃、留在运行根计入 stage_left），再把
``<stage>/sNN/<policy>/rec/`` 下没有结果行的孤儿目录整目录搬到 ``<dest>/<policy>/_orphan/<sNN>/<目录名>/``，然后全量对账，打印
``V8_EVAL_VIDEOS=PASS|FAIL policies= expected= videos= missing= decode_fail= sha_mismatch= error_attempt_videos= stage_left= orphan_videos= error_final_no_video=``：
expected = 每模型账本 accepted 终态数之和（权威终态口径复用 同目录 ``eval_report.py``）；videos = 本机已有且
front.mkv、wrist.mkv 都能读出帧数（>0）的终态录像目录数；error_attempt_videos = 本机已有的非权威尝试（错误／重试／迟到）目录数；
stage_left = 对账后运行根里仍有可搬文件的录像目录数（rsync 失败、sha 不符、持续不稳、孤儿搬不走都在这里体现，>0 即 FAIL）；
orphan_videos = 本次搬进 ``_orphan`` 的目录数（搬前先打印 ``ORPHAN_CANDIDATE``；--once 应在全部席位结束后跑）；
error_final_no_video = 非 infra 错误终局无录像但结果行写明原因的数（不计 missing，与 eval_report 同一判定），
PASS 要求 videos + error_final_no_video = expected。常驻模式 --stop-file 出现后，持续不稳或持续搬运失败超过
``--once-max-wait`` 秒的目录放弃（打印 reason=unstable／move_failed），进程照常退出。
帧数优先用 ``ffprobe -count_frames``，没有 ffprobe 时用 ``imageio-ffmpeg`` 自带 ffmpeg 全解码计帧，都没有则判 decode_fail
（``reason=no_decoder``）。读得出的帧数按 (路径, 大小, mtime) 缓存在 ``<dest>/decode-cache.jsonl``。末行 ``EXIT_CODE=``（FAIL 为 1）。

``--layout sgeval``（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.6；缺省 ``--layout v8`` 时上面的行为一字不变）::

    python dev-scripts/gl/eval_video_mover.py --layout sgeval --stage <NFS 媒体根> \\
        --dest artifacts/sg-evaluation/<run>/videos [--min-free-gib 50] [--once] [--stop-file F] [--interval S]

``--stage`` 指 ``run_eval_gl.sh``／``run_official_hard.sh`` 的 ``--media-root``（缺省 ``<运行根>/media``），其下为输出键
``<policy>[-<variant>]/<dataset>/<side>/<key>.a<attempt>[.dupN]/``（side 取 orig／new；点开头的目录如 ``.incoming`` 不碰）。
每局目录由席位脚本原子发布，里面是单个 ``*.mp4``（已转码）、``trace.jsonl``、``summary.json``／``transcode.json`` 等小文件；
多于一个 mp4 的目录不搬（打印 ``MOVER_SKIP reason=multi_mp4``）。目录 ``--stable-sec`` 秒无变化后：整目录
``rsync -a`` 到 ``<dest>/.incoming/<随机>/`` → 逐文件两端 sha256 相同 → 原子改名到 ``<dest>/<同一输出键>/``（已存在且内容
相同视为已搬；内容不同落 ``.dupN``）→ 只删核对过的 NFS 源文件 → 往 ``<dest>/moved.jsonl`` 追加一行（``mode: "sgeval"``）。
停止条件：sha256 不一致（只删临时目录、保留 NFS 源）或本机盘（``--dest`` 所在）剩余低于 ``--min-free-gib`` 时立刻停止
搬运，打印 ``MOVER_STOP reason=sha_mismatch|low_disk …``，退出码 1。``--once`` 搬到无待搬目录后打印
``SGEVAL_MOVE=PASS|FAIL moved= left= skipped= bytes= stopped=0|1``（PASS 要求 left、skipped、stopped 都为 0）；常驻模式每
``--interval`` 秒打印 ``VMOVE mode=sgeval moved= pending= bytes= stage_bytes=``，``--stop-file`` 出现且无待搬时退出。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import uuid
import subprocess
import sys
import time
import threading
import fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ V8 模式

V8_MEDIA = ("front.mkv", "wrist.mkv")
_REPORT_MOD = None


def eval_report_mod():
    """复用同目录 ``eval_report.py`` 的读入与权威终态口径（accepted_attempt_id），两处不各写一套。"""
    global _REPORT_MOD
    if _REPORT_MOD is None:
        path = Path(__file__).resolve().parent / "eval_report.py"
        spec = importlib.util.spec_from_file_location("eval_report_for_mover", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _REPORT_MOD = mod
    return _REPORT_MOD


RSYNC_TMP = re.compile(r"^\.[^/]+\.[A-Za-z0-9]{6}$")  # rsync 写入中的临时文件：.<name>.<6 位随机>


def is_nfs_placeholder(p: Path) -> bool:
    """NFS 删除占位（silly rename）：不搬、不删、不算可搬文件。"""
    return p.name.startswith(".nfs")


def tree_files(root: Path) -> list[Path]:
    """可搬文件（递归，含录像器留下的点开头文件与 .spool/），不含 .nfs* 占位。"""
    out = []
    for p in root.rglob("*"):
        if (p.is_file() or p.is_symlink()) and not is_nfs_placeholder(p):
            out.append(p)
    return sorted(out)


def dir_snapshot(root: Path) -> tuple | None:
    """目录内可搬文件 (相对路径, 大小, mtime_ns)；有 rsync 临时文件时返回 None（仍在写）。"""
    snap = []
    for p in tree_files(root):
        if RSYNC_TMP.match(p.name):
            return None
        try:
            st = p.lstat()
        except FileNotFoundError:
            return None
        snap.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
    return tuple(snap)


def v8_attempt_dirs(stage: Path, policies: list[str]) -> list[dict]:
    """结果行已写出的全部 V8 尝试（含 error／infra／迟到），每个录像目录一项。"""
    rm = eval_report_mod()
    out, seen = [], set()
    for pol in policies:
        st = rm.load_policy(stage, pol)
        for row in st["results"]:
            if not row.get("v8"):
                continue
            name = rm.rec_name(row)
            if not name:
                continue
            rec = Path(row["_seat_dir"]) / pol / "rec"
            # 节点同步若目标已存在会落 rec/<name>.dupN/：同一结果行的副本照常搬，目的地沿用 .dupN 名
            dup_re = re.compile(rf"^{re.escape(name)}\.dup\d+$")
            names = [name] + (sorted(p.name for p in rec.glob(f"{glob.escape(name)}.dup*") if dup_re.match(p.name))
                              if rec.is_dir() else [])
            for nm in names:
                src = rec / nm
                if str(src) in seen:
                    continue
                seen.add(str(src))
                out.append({"policy": pol, "row": row, "name": nm, "src": src})
    return out


def v8_dest_dir(dest: Path, policy: str, row: dict, name: str) -> Path:
    return dest / policy / str(row.get("tier") or "_notier") / str(row.get("task")) / name


def dir_digests(root: Path) -> dict[str, str]:
    return {str(f.relative_to(root)): sha256(f) for f in tree_files(root)}


def v8_move_dir(item: dict, dest: Path, moved_log: Path) -> tuple[str, int]:
    """rsync 整目录到 <dest>/.incoming/<随机>/ → 逐文件 sha256 → 原子改名到目标 → 只删核对过的源文件 → 记 moved.jsonl。

    返回 (结果, 字节数)；结果 ∈ moved / rsync_fail / sha_mismatch。本机已有副本一律不删；sha 不符只删临时目录。
    item 为结果行尝试（row 有值）或孤儿目录（row 为 None、带 out）。
    """
    src, row = item["src"], item.get("row") or {}
    out = item.get("out") or v8_dest_dir(dest, item["policy"], row, item["name"])
    incoming = dest / ".incoming" / uuid.uuid4().hex
    incoming.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["rsync", "-a", "--exclude=.nfs*", f"{src}/", f"{incoming}/"], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"# rsync 失败 {src}: {proc.stderr.strip()[:300]}", flush=True)
        shutil.rmtree(incoming, ignore_errors=True)
        return "rsync_fail", 0
    files: dict[str, str] = {}
    nbytes = 0
    for f in tree_files(src):  # 核对清单：此刻源里的可搬文件；核对之后新出现的文件不删
        rel = str(f.relative_to(src))
        digest = sha256(f)
        tgt = incoming / rel
        if not tgt.is_file() or sha256(tgt) != digest:
            print(f"# sha256 不一致，保留 NFS 副本、只删临时目录 {src} 文件 {rel}", flush=True)
            shutil.rmtree(incoming, ignore_errors=True)
            return "sha_mismatch", 0
        files[rel] = digest
        nbytes += f.stat().st_size
    if not files:
        shutil.rmtree(incoming, ignore_errors=True)
        return "empty", 0
    out.parent.mkdir(parents=True, exist_ok=True)
    final = out
    if out.exists():
        if dir_digests(out) == files:  # 上次已拷到本机、删源前中断：内容相同，不再落第二份
            shutil.rmtree(incoming, ignore_errors=True)
        else:
            n = 1
            while Path(f"{out}.dup{n}").exists():
                n += 1
            final = Path(f"{out}.dup{n}")
            print(f"# 本机已有内容不同的 {out}，新版本落 {final.name}", flush=True)
            os.rename(incoming, final)
    else:
        os.rename(incoming, final)
    for rel in files:
        (src / rel).unlink(missing_ok=True)
    for d in sorted((p for p in src.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    src_left = False
    try:
        src.rmdir()
    except OSError:
        src_left = True  # 残留 .nfs* 占位或核对后新出现的文件
    with moved_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"mode": "v8", "policy": item["policy"], "key": row.get("key"),
                                 "task": row.get("task"), "tier": row.get("tier"), "seed": row.get("seed"),
                                 "attempt_id": row.get("attempt_id"), "attempt_no": row.get("attempt_no"),
                                 "status": row.get("status"), "infra": row.get("infra"),
                                 "canary": bool(row.get("canary")), "orphan": item.get("row") is None,
                                 "seat": row.get("_seat") or item.get("seat"),
                                 "src": str(src), "dest": str(final), "files": files, "bytes": nbytes,
                                 "src_left": src_left, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return "moved", nbytes


def stage_rec_dirs(stage: Path, policies: list[str]) -> list[tuple[str, str, Path]]:
    """运行根里仍有可搬文件的录像目录 (policy, sNN, 路径)；点开头的目录（节点同步的 .incoming 等）跳过、不算孤儿。"""
    out = []
    for pol in policies:
        for d in sorted(stage.glob(f"s*/{pol}/rec/*")):
            if d.name.startswith("."):
                continue
            if d.is_dir() and tree_files(d):
                out.append((pol, d.parents[2].name, d))
    return out


def pick_decoder() -> tuple[str, str | None]:
    """ffprobe 优先；没有就用 imageio-ffmpeg 自带的 ffmpeg（再退回 PATH 里的 ffmpeg）；都没有返回 none。"""
    probe = shutil.which("ffprobe")
    if probe:
        return "ffprobe", probe
    try:
        import imageio_ffmpeg  # type: ignore

        return "ffmpeg", imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pass
    ff = shutil.which("ffmpeg")
    if ff:
        return "ffmpeg", ff
    return "none", None


def count_frames(path: Path, decoder: tuple[str, str | None]) -> int | None:
    """完整解码视频流 0 并返回帧数；读不出返回 None。"""
    kind, exe = decoder
    try:
        if kind == "ffprobe":
            proc = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                   "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                                  capture_output=True, text=True, timeout=1800)
            txt = proc.stdout.strip().split("\n")[0].strip().rstrip(",")
            return int(txt) if proc.returncode == 0 and txt.isdigit() else None
        if kind == "ffmpeg":
            proc = subprocess.run([exe, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-",
                                   "-progress", "pipe:1"], capture_output=True, text=True, timeout=1800)
            frames = [l.split("=", 1)[1].strip() for l in proc.stdout.splitlines() if l.startswith("frame=")]
            return int(frames[-1]) if proc.returncode == 0 and frames and frames[-1].isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


class DecodeCache:
    """读得出的帧数按 (路径, 大小, mtime_ns) 缓存；读不出的不缓存、下次重试。"""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[tuple, int] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.data[(r["path"], r["size"], r["mtime_ns"])] = int(r["frames"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    def frames(self, f: Path, decoder) -> int | None:
        st = f.stat()
        k = (str(f.resolve()), st.st_size, st.st_mtime_ns)
        if self.data.get(k):
            return self.data[k]
        n = count_frames(f, decoder)
        if n:
            self.data[k] = n
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"path": k[0], "size": k[1], "mtime_ns": k[2], "frames": n}) + "\n")
        return n


def v8_verify(stage: Path, dest: Path, policies: list[str], workers: int) -> tuple:
    """本机对账：返回 (expected, videos, missing, decode_fail, error_attempt_videos, error_final_no_video, problems)。

    非 infra 错误终局无录像：结果行写明原因 → error_final_no_video（不计 missing）；无原因 → missing。
    判定与 eval_report 的 error_final_without_video_and_reason 共用 ``error_final_video_class``。"""
    rm = eval_report_mod()
    decoder = pick_decoder()
    if decoder[0] != "ffprobe":
        print(f"# 解码器：{decoder[0]} {decoder[1] or ''}（未找到 ffprobe）", flush=True)
    cache = DecodeCache(dest / "decode-cache.jsonl")
    expected = 0
    terminal_dirs: list[tuple[str, dict, Path]] = []
    error_attempt_videos = 0
    for pol in policies:
        an = rm.analyze_attempts(rm.load_policy(stage, pol))
        expected += len(an["accepted"])
        for row in an["accepted"].values():
            terminal_dirs.append((pol, row, v8_dest_dir(dest, pol, row, rm.rec_name(row) or "_norec")))
        acc_ids = {str(r.get("attempt_id")) for r in an["accepted"].values()}
        for aid, lst in an["by_attempt"].items():
            row = lst[-1]
            if aid in acc_ids or not rm.rec_name(row):
                continue
            if v8_dest_dir(dest, pol, row, rm.rec_name(row)).is_dir():
                error_attempt_videos += 1
    missing = decode_fail = error_final_no_video = 0
    problems: list[dict] = []
    to_decode = []
    for pol, row, d in terminal_dirs:
        has_media = d.is_dir() and all((d / m).is_file() for m in V8_MEDIA)
        if rm.error_final_video_class(row, has_media) == "explained":
            error_final_no_video += 1
            problems.append({"policy": pol, "key": row.get("key"), "problem": "error_final_no_video",
                             "reason": (rm.error_reason(row) or "")[:200]})
            continue
        if not has_media:
            missing += 1
            problems.append({"policy": pol, "key": row.get("key"), "problem": "missing", "dir": str(d)})
            continue
        to_decode.append((pol, row, d))

    def check(item):
        _, _, d = item
        if decoder[0] == "none":
            return item, {m: None for m in V8_MEDIA}
        return item, {m: cache.frames(d / m, decoder) for m in V8_MEDIA}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(check, to_decode))
    videos = 0
    for (pol, row, d), frames in results:
        if all(frames[m] for m in V8_MEDIA):
            videos += 1
        else:
            decode_fail += 1
            problems.append({"policy": pol, "key": row.get("key"), "problem": "decode_fail", "dir": str(d),
                             "frames": frames, "reason": "no_decoder" if decoder[0] == "none" else None})
    return expected, videos, missing, decode_fail, error_attempt_videos, error_final_no_video, problems


def v8_verdict(stage: Path, dest: Path, policies: list[str], *, sha_mismatch: int, stage_left: int,
               orphan_videos: int, workers: int) -> tuple[bool, str, list]:
    (expected, videos, missing, decode_fail, error_attempt_videos, error_final_no_video,
     problems) = v8_verify(stage, dest, policies, workers)
    ok = (expected > 0 and missing == 0 and decode_fail == 0 and sha_mismatch == 0 and stage_left == 0
          and videos + error_final_no_video == expected)
    line = (f"V8_EVAL_VIDEOS={'PASS' if ok else 'FAIL'} policies={len(policies)} expected={expected} videos={videos} "
            f"missing={missing} decode_fail={decode_fail} sha_mismatch={sha_mismatch} "
            f"error_attempt_videos={error_attempt_videos} stage_left={stage_left} orphan_videos={orphan_videos} "
            f"error_final_no_video={error_final_no_video}")
    return ok, line, problems


def v8_main(args) -> int:
    stage, dest = Path(args.stage).resolve(), Path(args.dest)
    policies = [p for p in args.policies.split(",") if p]
    dest.mkdir(parents=True, exist_ok=True)
    moved_log = dest / "moved.jsonl"
    seen: dict[str, tuple[tuple | None, float]] = {}
    first_pending: dict[str, float] = {}
    failed: set[str] = set()
    gave_up: set[str] = set()
    warned_orphans: set[str] = set()
    moved_bytes = moved_n = sha_mismatch = orphan_videos = 0
    last_report = 0.0
    while True:
        pending = 0
        items = v8_attempt_dirs(stage, policies)
        if args.once:  # 对账轮：没有结果行的孤儿录像目录也整目录搬到 _orphan
            known = {str(it["src"]) for it in items}
            for pol, seat, d in stage_rec_dirs(stage, policies):
                if str(d) not in known:
                    if str(d) not in warned_orphans:  # --once 应在全部席位结束后跑；此时仍无结果行的目录才当孤儿
                        warned_orphans.add(str(d))
                        print(f"ORPHAN_CANDIDATE policy={pol} seat={seat} dir={d}", flush=True)
                    items.append({"policy": pol, "row": None, "name": d.name, "src": d, "seat": seat,
                                  "out": dest / pol / "_orphan" / seat / d.name})
        for item in items:
            src = item["src"]
            key = str(src)
            if not src.is_dir() or not tree_files(src):
                continue  # 尚未同步到运行根、已搬走，或只剩 .nfs* 占位
            if (args.once and key in failed) or key in gave_up:
                continue  # 对账轮里同一目录只试一次；失败的留在运行根，计入 stage_left
            snap = dir_snapshot(src)
            now = time.time()
            first_pending.setdefault(key, now)
            # 收尾阶段（--once，或常驻模式 --stop-file 已出现）不无限等持续不稳的目录
            finishing = args.once or bool(args.stop_file and Path(args.stop_file).exists())
            if finishing and now - first_pending[key] > args.once_max_wait:
                why = "move_failed" if key in failed else "unstable"
                print(f"# 持续{'搬运失败' if why == 'move_failed' else '不稳'}超过 {args.once_max_wait:.0f}s，"
                      f"放弃并计入 stage_left reason={why}：{src}", flush=True)
                gave_up.add(key)
                continue
            prev = seen.get(key)
            if snap is None or prev is None or prev[0] != snap:
                seen[key] = (snap, now)
                pending += 1
                continue
            if now - prev[1] < args.stable_sec:
                pending += 1
                continue
            result, nbytes = v8_move_dir(item, dest, moved_log)
            seen.pop(key, None)
            if result in ("moved", "empty"):
                first_pending.pop(key, None)  # 搬运失败不清：常驻模式 --stop-file 后持续失败也能按时放弃
            if result == "moved":
                failed.discard(key)
                moved_n += 1
                moved_bytes += nbytes
                orphan_videos += item.get("row") is None
            elif result != "empty":
                sha_mismatch += result == "sha_mismatch"
                failed.add(key)
                if not args.once:
                    pending += 1
        now = time.time()
        stop = bool(args.stop_file and Path(args.stop_file).exists() and pending == 0)
        if args.once or stop or now - last_report >= args.interval:
            stage_bytes = sum(f.stat().st_size for pol in policies for f in stage.glob(f"s*/{pol}/rec/**/*")
                              if f.is_file()) if stage.exists() else 0
            print(f"VMOVE mode=v8 moved={moved_n} pending={pending} bytes={moved_bytes} stage_bytes={stage_bytes} "
                  f"sha_mismatch={sha_mismatch}", flush=True)
            last_report = now
        if args.once:
            if pending == 0:
                break
        elif stop:
            break
        time.sleep(max(0.05, min(args.stable_sec, args.interval)))
    rc = 0
    if args.once:
        left = stage_rec_dirs(stage, policies)
        for pol, seat, d in left:
            why = ("move_failed" if str(d) in failed else
                   "unstable" if str(d) in gave_up else "not_moved")
            print(f"# 运行根残留 {json.dumps({'policy': pol, 'seat': seat, 'dir': str(d), 'reason': why}, ensure_ascii=False)}",
                  flush=True)
        ok, line, problems = v8_verdict(stage, dest, policies, sha_mismatch=sha_mismatch, stage_left=len(left),
                                        orphan_videos=orphan_videos, workers=args.decode_workers)
        for p in problems[:50]:
            print(f"# {json.dumps(p, ensure_ascii=False)}", flush=True)
        print(line, flush=True)
        rc = 0 if ok else 1
    print(f"EXIT_CODE={rc}", flush=True)
    return rc


# ------------------------------------------------------------------ sgeval 布局

SGEVAL_SIDES = ("orig", "new")


def sgeval_dirs(stage: Path) -> list[tuple[str, Path]]:
    """媒体根下全部每局目录 (输出键, 路径)：<label>/<dataset>/<side>/<目录名>/；任何一层点开头的跳过。"""
    out = []
    if not stage.is_dir():
        return out
    for d in sorted(stage.glob("*/*/*/*")):
        rel = d.relative_to(stage)
        if not d.is_dir() or any(part.startswith(".") for part in rel.parts) or rel.parts[2] not in SGEVAL_SIDES:
            continue
        out.append((str(rel), d))
    return out


def free_gib(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / (1 << 30)


def sgeval_move_dir(rel: str, src: Path, dest: Path, moved_log: Path) -> tuple[str, int]:
    """rsync 整目录 → 逐文件 sha256 → 原子改名到 <dest>/<rel> → 只删核对过的源文件 → 记 moved.jsonl。
    返回 (结果, 字节数)；结果 ∈ moved / same / rsync_fail / sha_mismatch / empty。sha 不符只删临时目录、源不动。"""
    incoming = dest / ".incoming" / uuid.uuid4().hex
    incoming.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["rsync", "-a", "--exclude=.nfs*", f"{src}/", f"{incoming}/"], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"# rsync 失败 {src}: {proc.stderr.strip()[:300]}", flush=True)
        shutil.rmtree(incoming, ignore_errors=True)
        return "rsync_fail", 0
    files: dict[str, str] = {}
    nbytes = 0
    for f in tree_files(src):
        r = str(f.relative_to(src))
        digest = sha256(f)
        tgt = incoming / r
        if not tgt.is_file() or sha256(tgt) != digest:
            print(f"# sha256 不一致，保留 NFS 源、只删临时目录 {src} 文件 {r}", flush=True)
            shutil.rmtree(incoming, ignore_errors=True)
            return "sha_mismatch", 0
        files[r] = digest
        nbytes += f.stat().st_size
    if not files:
        shutil.rmtree(incoming, ignore_errors=True)
        return "empty", 0
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    final, result = out, "moved"
    if out.exists():
        if dir_digests(out) == files:  # 上次已拷到本机、删源前中断：内容相同，不再落第二份
            shutil.rmtree(incoming, ignore_errors=True)
            result = "same"
        else:
            n = 1
            while Path(f"{out}.dup{n}").exists():
                n += 1
            final = Path(f"{out}.dup{n}")
            print(f"# 本机已有内容不同的 {out}，新版本落 {final.name}", flush=True)
            os.rename(incoming, final)
    else:
        os.rename(incoming, final)
    for r in files:
        (src / r).unlink(missing_ok=True)
    for d in sorted((p for p in src.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    src_left = False
    try:
        src.rmdir()
    except OSError:
        src_left = True
    parts = Path(rel).parts
    with moved_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"mode": "sgeval", "key": rel, "label": parts[0], "dataset": parts[1], "side": parts[2],
                                 "episode": parts[3], "src": str(src), "dest": str(final), "files": files,
                                 "bytes": nbytes, "src_left": src_left, "result": result,
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return "moved", nbytes


def sgeval_main(args) -> int:
    stage, dest = Path(args.stage).resolve(), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    moved_log = dest / "moved.jsonl"
    seen: dict[str, tuple[tuple | None, float]] = {}
    skipped: set[str] = set()
    moved_n = moved_bytes = 0
    stop_reason = None
    last_report = 0.0
    while stop_reason is None:
        pending = 0
        for rel, src in sgeval_dirs(stage):
            if not src.is_dir() or not tree_files(src):
                continue
            mp4s = [p for p in src.glob("*.mp4")]
            if len(mp4s) > 1:
                if rel not in skipped:
                    skipped.add(rel)
                    print(f"MOVER_SKIP reason=multi_mp4 dir={rel} mp4={len(mp4s)}", flush=True)
                continue
            snap = dir_snapshot(src)
            now = time.time()
            prev = seen.get(rel)
            if snap is None or prev is None or prev[0] != snap:
                seen[rel] = (snap, now)
                pending += 1
                continue
            if now - prev[1] < args.stable_sec:
                pending += 1
                continue
            free = free_gib(dest)
            if free < args.min_free_gib:
                stop_reason = f"low_disk free_gib={free:.1f} min_free_gib={args.min_free_gib:g} dest={dest}"
                break
            result, nbytes = sgeval_move_dir(rel, src, dest, moved_log)
            seen.pop(rel, None)
            if result == "moved":
                moved_n += 1
                moved_bytes += nbytes
            elif result == "sha_mismatch":
                stop_reason = f"sha_mismatch dir={rel}"
                break
            elif result == "rsync_fail":
                pending += 1
        if stop_reason is not None:
            break
        now = time.time()
        stop = bool(args.stop_file and Path(args.stop_file).exists() and pending == 0)
        if args.once or stop or now - last_report >= args.interval:
            stage_bytes = sum(f.stat().st_size for _, d in sgeval_dirs(stage) for f in tree_files(d))
            print(f"VMOVE mode=sgeval moved={moved_n} pending={pending} bytes={moved_bytes} stage_bytes={stage_bytes}",
                  flush=True)
            last_report = now
        if (args.once and pending == 0) or stop:
            break
        time.sleep(max(0.05, min(args.stable_sec, args.interval)))
    if stop_reason is not None:
        print(f"MOVER_STOP reason={stop_reason}（停止搬运，NFS 源保留）", flush=True)
    left = sum(1 for _, d in sgeval_dirs(stage) if tree_files(d))
    rc = 0
    if args.once or stop_reason is not None:
        ok = stop_reason is None and left == 0 and not skipped
        print(f"SGEVAL_MOVE={'PASS' if ok else 'FAIL'} moved={moved_n} left={left} skipped={len(skipped)} "
              f"bytes={moved_bytes} stopped={int(stop_reason is not None)}", flush=True)
        rc = 0 if ok else 1
    print(f"EXIT_CODE={rc}", flush=True)
    return rc


RAW_SCHEMA = "raw-ood-v1"
RAW_CONTEXT = ("run", "run_name", "exec_commit", "config_sha256", "controller_job_id")


def raw_atomic(path: Path, value: dict) -> None:
    """同目录临时文件、落盘后替换，消费者不会读到半份 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def raw_relative(value: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("相对路径必须是字符串")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(p in ("..", ".") for p in value.split("/")):
        raise ValueError(f"非法相对路径：{value!r}")
    return path


def raw_tree(root: Path) -> dict:
    """禁止符号链接和特殊文件，完整集合不忽略隐藏文件。"""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"不是实体目录：{root}")
    out = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"非法文件类型：{path}")
        if path.is_file():
            out[str(path.relative_to(root))] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return out


def raw_signature(root: Path) -> tuple:
    """轮询只核文件集合与 stat，启动和最终验收才全量读取字节。"""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("缓存目录不再是实体目录")
    signature = []
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("缓存目录出现非法文件类型")
        if path.is_file():
            st = path.stat()
            signature.append((str(path.relative_to(root)), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino))
    return tuple(sorted(signature))


def raw_validate_publication(row: dict, context: dict) -> None:
    required = {"schema", *RAW_CONTEXT, "model", "shard_id", "identity_key", "relative_dir", "files", "delete_files"}
    if not required <= row.keys() or row["schema"] != RAW_SCHEMA:
        raise ValueError("发布项缺键或协议不符")
    if any(row[k] != context[k] for k in RAW_CONTEXT):
        raise ValueError("发布项版本或运行身份不符")
    raw_relative(row["relative_dir"])
    if not all(isinstance(row[k], str) and row[k] for k in ("model", "shard_id", "identity_key")):
        raise ValueError("发布项身份为空")
    if not isinstance(row["files"], dict) or not row["files"]:
        raise ValueError("发布项文件集合为空")
    for name, info in row["files"].items():
        raw_relative(name)
        if set(info) != {"size", "sha256"} or type(info["size"]) is not int or info["size"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", info["sha256"]):
            raise ValueError("发布项文件指纹非法")
    if not isinstance(row["delete_files"], list) or len(set(row["delete_files"])) != len(row["delete_files"]):
        raise ValueError("删除清单非法")
    for name in row["delete_files"]:
        if name not in row["files"] or name not in ("front.mkv", "wrist.mkv", "arrays.npz", "raw/front.mkv", "raw/wrist.mkv", "raw/arrays.npz"):
            raise ValueError("删除清单含小文件或非发布文件")


def raw_verify_media(directory: Path) -> dict:
    """完整解码并核帧索引与动作原字节，不启动仿真。"""
    import numpy as np
    raw = directory / "raw" if (directory / "raw").is_dir() else directory
    meta = json.loads((raw / "meta.json").read_text())
    if meta["raw_codec"] != "av1-yuv444p" or meta["level"] > 1:
        raise ValueError("录像编码或退化档不符")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("完整解码需要 ffmpeg 和 ffprobe")
    decoded = {}
    frame_indices = {}
    for stream in ("front", "wrist"):
        rows = [json.loads(line) for line in (raw / f"frames-{stream}.jsonl").read_text().splitlines()]
        if not rows or [r["idx"] for r in rows] != list(range(len(rows))) or any(type(r["enc"]) is not int or r["enc"] < 0 for r in rows):
            raise ValueError("帧索引不连续或缺失")
        encoded = max(r["enc"] for r in rows) + 1
        if set(r["enc"] for r in rows) != set(range(encoded)):
            raise ValueError("编码帧索引缺口")
        if any(not re.fullmatch(r"[a-f0-9]{64}", r["sha256"]) for r in rows):
            raise ValueError("帧索引哈希非法")
        frame_indices[stream] = rows
        video = raw / f"{stream}.mkv"
        proc = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(video), "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True, timeout=1800)
        if proc.returncode or proc.stderr.strip():
            raise ValueError(f"完整解码失败：{stream} {proc.stderr[:300]}")
        probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams", "-show_entries", "frame=best_effort_timestamp_time:stream=codec_name,pix_fmt", "-of", "json", str(video)], capture_output=True, text=True, timeout=1800)
        if probe.returncode:
            raise ValueError("视频探测失败")
        data = json.loads(probe.stdout)
        if len(data["frames"]) != encoded or data["streams"][0]["codec_name"] != "av1" or data["streams"][0]["pix_fmt"] != "yuv444p":
            raise ValueError("视频帧数或格式不符")
        if any(abs(float(f["best_effort_timestamp_time"]) - i / 30) > 0.5 / 30 for i, f in enumerate(data["frames"])):
            raise ValueError("视频时间戳不符")
        decoded[stream] = encoded
    rows = [json.loads(line) for line in (raw / "trace.jsonl").read_text().splitlines()]
    spec = importlib.util.spec_from_file_location("_raw_trace_checker", Path(__file__).resolve().parents[2] / "src/robomme_ood_eval/record/trace_writer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problems = module.validate_trace(rows)
    if problems:
        raise ValueError(f"轨迹结构不符：{problems}")
    if len(frame_indices["front"]) != len(frame_indices["wrist"]):
        raise ValueError("双视角逻辑帧数不符")
    with np.load(raw / "arrays.npz", allow_pickle=False) as arrays:
        for row in rows:
            if row["kind"] != "step":
                continue
            action = arrays[f"exec_action__{row['step'] - 1:05d}"]
            expected = row["action"]
            actual = module.array_record(action)
            if any(actual[k] != expected[k] for k in ("dtype", "shape", "sha256")):
                raise ValueError("动作数组与轨迹不符")
    result = json.loads((directory / "result.json").read_text())
    for key in ("dataset", "task", "tier", "seed", "key"):
        if result[key] != rows[0]["identity"][key]:
            raise ValueError("轨迹与结果身份不符")
    if type(result["task_success"]) not in (bool, int) or result["task_success"] not in (0, 1):
        raise ValueError("成功字段非法")
    if result["exec_steps"] != rows[-1]["exec_steps"] or result["error_kind"] is not None:
        raise ValueError("结果步骤或基础设施状态不符")
    return decoded


def raw_move(row: dict, stage: Path, dest: Path, *, min_free_gib: float = 50) -> dict:
    """先完整落本机再删已核验大文件；崩溃重启只恢复删除，不生成重复局。"""
    rel = raw_relative(row["relative_dir"])
    source, target = stage / rel, dest / rel
    for base, path in ((stage, source), (dest, target)):
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError("目录越界")
        if any(p.is_symlink() for p in [path, *path.parents] if p != base.parent):
            raise ValueError("目录链含符号链接")
    if free_gib(dest) < min_free_gib + sum(v["size"] for v in row["files"].values()) / (1 << 30):
        raise ValueError("low_disk")
    if target.exists():
        if raw_tree(target) != row["files"]:
            raise ValueError("本机同名目录内容冲突，禁止覆盖")
    else:
        if raw_tree(source) != row["files"]:
            raise ValueError("源文件集合或 SHA 不符")
        incoming = dest / ".incoming" / uuid.uuid4().hex
        incoming.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["rsync", "-a", f"{source}/", f"{incoming}/"], capture_output=True, text=True, timeout=1800)
        if proc.returncode:
            raise ValueError(f"rsync失败：{proc.stderr[:300]}")
        if raw_tree(incoming) != row["files"] or raw_tree(source) != row["files"]:
            raise ValueError("拷贝后文件集合或 SHA 不符")
        raw_verify_media(incoming)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("落地时同名冲突")
        os.rename(incoming, target)
    decoded = raw_verify_media(target)
    result_identity = json.loads((target / "result.json").read_text())["key"]
    if not result_identity or not row["identity_key"].endswith(":" + result_identity):
        raise ValueError("发布项与结果身份不符")
    # 每个删除再次核对，缺失只在本机完整副本已验证时允许。
    remaining = raw_tree(source)
    if any(name not in row["files"] or info != row["files"][name] for name, info in remaining.items()):
        raise ValueError("删除前源文件变化")
    if any(name not in remaining for name in set(row["files"]) - set(row["delete_files"])):
        raise ValueError("源小文件缺失")
    for name in row["delete_files"]:
        path = source / name
        if path.exists():
            if {"size": path.stat().st_size, "sha256": sha256(path)} != row["files"][name]:
                raise ValueError("删除前大文件变化")
            path.unlink()
    return {**row, "dest": str(target), "bytes": sum(v["size"] for v in row["files"].values()), "decoded": decoded, "trace_checked": True, "result": "moved"}


def raw_main(args) -> int:
    stage, dest = Path(args.stage).resolve(), Path(args.dest).resolve()
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    config_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if "config_sha256" in manifest and manifest["config_sha256"] != config_sha:
        raise ValueError("配置内哈希与原字节哈希冲突")
    manifest["config_sha256"] = config_sha
    context = {k: manifest[k] for k in RAW_CONTEXT}
    if context["run"] != context["run_name"]:
        raise ValueError("run 与 run_name 不一致")
    expected = manifest["expected_publications"]
    identities = manifest["expected_identities"]
    if not isinstance(identities, list) or len(set(identities)) != len(identities) or {hashlib.sha256(k.encode()).hexdigest() + ".json" for k in identities} != set(expected):
        raise ValueError("期望身份与发布文件集合不一致")
    if not isinstance(expected, list) or not expected or len(set(expected)) != len(expected) or any(not re.fullmatch(r"[a-f0-9]{64}\.json", name) for name in expected):
        raise ValueError("权威发布清单非法")
    dest.mkdir(parents=True, exist_ok=True)
    state = stage / "mover"
    state.mkdir(parents=True, exist_ok=True)
    heartbeat_path = Path(args.heartbeat_file) if args.heartbeat_file else state / "heartbeat.json"
    error_path = Path(args.error_file) if args.error_file else state / "error.json"
    stop_path = Path(args.stop_file) if args.stop_file else stage / "STOP"
    lock = (dest / ".raw-mover.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"schema": RAW_SCHEMA, **context, "time": time.time(), "pid": os.getpid(), "pending": 0, "moved": 0, "last_progress": time.time()}
    done = threading.Event()
    heartbeat_error = []
    def heartbeat():
        while not done.is_set():
            try:
                raw_atomic(heartbeat_path, {**status, "time": time.time()})
            except Exception as exc:
                heartbeat_error.append(exc)
                return
            done.wait(10)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    verified_receipts = {}
    try:
        while True:
            if heartbeat_error:
                raise RuntimeError(f"心跳写入失败：{heartbeat_error[0]}")
            if stop_path.exists():
                raise RuntimeError("收到全局 STOP")
            published = {p.name: p for p in (stage / "published").glob("*.json")}
            if set(published) - set(expected):
                raise ValueError("出现非权威发布项")
            receipts = {p.name for p in (state / "receipts").glob("*.json")}
            if receipts - set(published):
                raise ValueError("回执存在非发布身份")
            status["pending"] = 0
            for name, path in sorted(published.items()):
                row = json.loads(path.read_text())
                raw_validate_publication(row, context)
                if row["identity_key"] not in identities:
                    raise ValueError("发布身份不在权威清单")
                if name != hashlib.sha256(row["identity_key"].encode()).hexdigest() + ".json":
                    raise ValueError("发布文件名与身份不符")
                receipt = state / "receipts" / name
                if receipt.exists():
                    target = dest / raw_relative(row["relative_dir"])
                    signature = (sha256(path), sha256(receipt), raw_signature(target))
                    if name in verified_receipts:
                        if verified_receipts[name] != signature:
                            raise ValueError("已核验发布、回执或本机文件发生变化")
                        continue
                    prior = json.loads(receipt.read_text())
                    if any(prior[k] != row[k] for k in row) or prior["result"] != "moved" or prior["trace_checked"] is not True or raw_tree(dest / raw_relative(row["relative_dir"])) != row["files"]:
                        raise ValueError("已有回执内容或本机副本不符")
                    if prior["dest"] != str(dest / raw_relative(row["relative_dir"])) or prior["bytes"] != sum(v["size"] for v in row["files"].values()) or set(prior["decoded"]) != {"front", "wrist"} or any(type(n) is not int or n <= 0 for n in prior["decoded"].values()):
                        raise ValueError("已有回执路径、字节数或解码计数不符")
                    verified_receipts[name] = signature
                    continue
                status["pending"] += 1
                result = raw_move(row, stage, dest, min_free_gib=args.min_free_gib)
                raw_atomic(receipt, result)
                verified_receipts[name] = (sha256(path), sha256(receipt), raw_signature(dest / raw_relative(row["relative_dir"])))
                status["last_progress"] = time.time()
            status["moved"] = len(list((state / "receipts").glob("*.json")))
            status["pending"] = len(published) - status["moved"]
            if (stage / "gpu_work_complete.json").exists() or args.once:
                if not args.once:
                    marker = json.loads((stage / "gpu_work_complete.json").read_text())
                    if any(marker[k] != context[k] for k in RAW_CONTEXT):
                        raise ValueError("GPU完成标记版本不符")
                if set(published) != set(expected) or status["moved"] != len(expected):
                    raise ValueError("最终发布或回执集合不完整")
                for path in published.values():
                    row = json.loads(path.read_text())
                    if raw_tree(dest / raw_relative(row["relative_dir"])) != row["files"]:
                        raise ValueError("最终本机文件集合或 SHA 不符")
                    raw_verify_media(dest / raw_relative(row["relative_dir"]))
                    source_files = raw_tree(stage / raw_relative(row["relative_dir"]))
                    retained = {k: v for k, v in row["files"].items() if k not in row["delete_files"]}
                    if source_files != retained:
                        raise ValueError("最终源大文件未排空或小文件变化")
                report = {"schema": RAW_SCHEMA, **context, "expected": len(expected), "moved": len(expected), "missing": 0, "decode_fail": 0, "sha_mismatch": 0, "pending": 0, "time": time.time()}
                raw_atomic(state / "completed.json", report)
                print(f"RAW_DELIVERY=PASS expected={len(expected)} moved={len(expected)} missing=0 decode_fail=0 sha_mismatch=0 pending=0", flush=True)
                return 0
            done.wait(min(args.interval, 5))
    except Exception as exc:
        error = {"schema": RAW_SCHEMA, **context, "component": "mover", "error": type(exc).__name__, "message": str(exc), "time": time.time()}
        raw_atomic(error_path, error)
        if not stop_path.exists():
            raw_atomic(stop_path, error)
        print(f"RAW_DELIVERY=FAIL error={type(exc).__name__} message={exc}", flush=True)
        return 1
    finally:
        done.set()
        thread.join(timeout=11)
        lock.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="v8", choices=("v8",),
                    help="v8（缺省且唯一）：整目录搬 V8／V9 评估录像，见 v8_main")
    ap.add_argument("--manifest", help="raw-ood：冻结运行配置及权威发布项清单")
    ap.add_argument("--heartbeat-file", help="raw-ood：NFS 独立心跳路径")
    ap.add_argument("--error-file", help="raw-ood：NFS 结构化错误路径")
    ap.add_argument("--layout", default="v8", choices=("v8", "sgeval", "raw-ood"),
                    help="v8（缺省）：运行根 sNN/<policy>/rec/ 布局；sgeval：媒体根下 <label>/<dataset>/<side>/<key>.a<n>/，见 sgeval_main")
    ap.add_argument("--min-free-gib", type=float, default=50.0,
                    help="sgeval：--dest 所在盘剩余低于该值即停止搬运（MOVER_STOP reason=low_disk）")
    ap.add_argument("--policies", default="smvla,perceptual-framesamp-modul", help="v8 模式：要搬的模型（运行根下 sNN/<policy>/）")
    ap.add_argument("--decode-workers", type=int, default=4, help="v8 --once 解码核对并行数")
    ap.add_argument("--once-max-wait", type=float, default=300.0,
                    help="v8 --once：单个目录持续不稳超过该秒数即放弃（留在运行根、计入 stage_left）")
    ap.add_argument("--stage", required=True, help="NFS 视频暂存根（只用于统计积压字节数与越界检查）")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--stable-sec", type=float, default=10.0)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()
    if args.layout == "raw-ood":
        if not args.manifest:
            ap.error("raw-ood 必须提供 --manifest")
        try:
            return raw_main(args)
        except Exception as exc:
            # 配置与独占锁的前置失败也必须通知监督器，不能仅写标准错误。
            context = {}
            try:
                config = json.loads(Path(args.manifest).read_text())
                context = {k: config[k] for k in RAW_CONTEXT if k in config}
                context["config_sha256"] = sha256(Path(args.manifest))
            except Exception:
                pass
            error = {"schema": RAW_SCHEMA, **context, "component": "mover", "error": type(exc).__name__, "message": str(exc), "time": time.time()}
            raw_atomic(Path(args.error_file) if args.error_file else Path(args.stage) / "mover/error.json", error)
            stop = Path(args.stop_file) if args.stop_file else Path(args.stage) / "STOP"
            if not stop.exists():
                raw_atomic(stop, error)
            print(f"RAW_DELIVERY=FAIL error={type(exc).__name__} message={exc}", flush=True)
            return 1
    if args.layout == "sgeval":
        return sgeval_main(args)
    return v8_main(args)


if __name__ == "__main__":
    sys.exit(main())
