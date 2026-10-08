#!/usr/bin/env bash
# 席位媒体函数库（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S2b」）：只含函数，不设变量、不运行。
#
# 由 run_seat.sh（随之 run_eval_gl.sh／run_official_hard.sh／pair_seat.sh）与 run_astra.sh `source`；函数内用
# `${BASH_SOURCE[0]}` 定位本文件所在目录（即执行副本的 scripts/eval-official/），不依赖调用方的 REPO 变量。
#
#   transcode_episode_dir [--keep-raw] <局目录>
#       就地转码为 episode.mp4（实现原在 run_seat.sh，原样移入）；--keep-raw 时转码成功也不删原始帧
#       （官方重绘失败的局用，原始帧随目录发布以便事后重绘），transcode.json 记 raw_kept=true。
#   render_official_dir <局目录>
#       官方版式重绘：已有 official/*.mp4 时先完整核验（official_media_check.py --verify-dir：恰 1 个、完整解码、
#       帧数与 provenance 一致），通过打印 OFFICIAL_RENDER=KEPT；否则把旧 official/ 整体移到隐藏留证目录
#       .official-rejected-<时间>-<pid>/（不删），再调 render_official_video.py <局目录> --source raw --jobs 1
#       （渲染器内部写临时名后 os.replace 原子替换），渲染后再核验一次。返回 0 = KEPT／重绘通过／无帧 error 局，
#       非 0 = 失败（打印 OFFICIAL_RENDER=FAIL dir= stage= reason=）。
#       --source 由 S2a 新增：渲染器 --help 里没有 --source 时不硬调（旧渲染器只读 episode.mp4，而此时还没转码），
#       直接以 reason=renderer_no_source_raw 失败，由调用方保留原始帧。
#   seat_media_ffmpeg
#       ffmpeg 解析顺序与 transcode_episode_dir 内 ffmpeg_exe() 相同：SGEVAL_FFMPEG → V75_FFMPEG → /usr/bin/ffmpeg
#       → PATH → imageio_ffmpeg；找不到返回 1。
#
# 解释器：调用方定义了 tool_py（run_seat.sh）就用它，否则取 TOOL_PY → BENCH_PY → python3。

seat_media_py() {  # 本库内 Python 小工具的解释器
  if declare -F tool_py >/dev/null 2>&1; then tool_py; else echo "${TOOL_PY:-${BENCH_PY:-python3}}"; fi
}

seat_media_dir() {  # 本文件所在目录（绝对路径）
  (cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
}

seat_media_ffmpeg() {  # 打印 ffmpeg 路径；顺序同 transcode_episode_dir 的 ffmpeg_exe()
  local c
  for c in "${SGEVAL_FFMPEG:-}" "${V75_FFMPEG:-}" /usr/bin/ffmpeg "$(command -v ffmpeg 2>/dev/null)"; do
    if [[ -n "$c" && -f "$c" && -x "$c" ]]; then echo "$c"; return 0; fi
  done
  c="$("$(seat_media_py)" - <<'PY' 2>/dev/null
try:
    import imageio_ffmpeg
    print(imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    pass
PY
)"
  [[ -n "$c" ]] && { echo "$c"; return 0; }
  return 1
}

transcode_episode_dir() {  # [--keep-raw] <局目录>；返回值与打印行见 _seat_media_transcode
  _seat_media_transcode "$@"
}

_seat_media_transcode() {  # [--keep-raw] $1 = 每局目录。就地转码为 episode.mp4，帧数一致后删原始帧；打印 REC_TRANSCODE 行
  # 返回 0：ok／already／none（无媒体）／empty；1：帧数不符（原始帧保留、mp4 删除）；2：失败（原始帧保留）
  local keep=0
  if [[ "${1:-}" == "--keep-raw" ]]; then keep=1; shift; fi
  "$(seat_media_py)" - "$1" "$keep" <<'PY'
"""两种原始帧：
- 新侧（recorder.py）：front.mkv／wrist.mkv（FFV1，同一流里重复帧只编码一份）+ frames-<stream>.jsonl（idx→enc）。
  按 idx 展开回逐帧原图（同 scripts/injection-dev/site/eval_transcode.py），期望帧数 = 记录行数；
- 原侧（pp_official_runner.py／official_hard_runner.py）：frames/{front,wrist}.rgb24 + frames/frames.json
  （pix_fmt=rgb24、各流 width/height/count），期望帧数 = count。
两路左右拼接（高度不同则下方补黑），libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart，30 fps。
转码后完整解码数帧，与期望相等才删原始帧（新侧删 *.mkv 与 .spool/，原侧删 frames/*.rgb24，frames.json 与
frames-*.jsonl 保留），并写 transcode.json。第二个参数为 1（--keep-raw）时一律不删原始帧，transcode.json 记
raw_kept=true（官方重绘失败的局：原始帧随目录发布，留待事后重绘）。"""
import json, mmap, os, re, shutil, subprocess, sys
from pathlib import Path

FPS, MP4, STREAMS = 30, "episode.mp4", ("front", "wrist")
d = Path(sys.argv[1])
KEEP_RAW = len(sys.argv) > 2 and sys.argv[2] == "1"


def ffmpeg_exe():
    for c in (os.environ.get("SGEVAL_FFMPEG"), os.environ.get("V75_FFMPEG"), "/usr/bin/ffmpeg", shutil.which("ffmpeg")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def count_frames(ff, path):
    p = subprocess.run([ff, "-nostdin", "-v", "error", "-nostats", "-i", str(path), "-map", "0:v:0", "-f", "null", "-",
                        "-progress", "pipe:1"], capture_output=True, text=True)
    fr = [x.split("=", 1)[1].strip() for x in p.stdout.splitlines() if x.startswith("frame=")]
    return int(fr[-1]) if p.returncode == 0 and fr and fr[-1].isdigit() else -1


def video_size(ff, path):
    p = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True)
    for line in p.stderr.splitlines():
        if "Video:" in line:
            m = re.search(r",\s*(\d+)x(\d+)[\s,\[]", line + " ")
            if m:
                return int(m.group(1)), int(m.group(2))
    raise RuntimeError(f"读不出分辨率：{path}")


def emit(**kw):
    kw.setdefault("dir", d.name)
    if KEEP_RAW:
        kw["raw_kept"] = True
    (d / "transcode.json").write_text(json.dumps(kw, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print("REC_TRANSCODE " + " ".join(f"{k}={kw[k]}" for k in ("dir", "kind", "frames", "mp4_frames", "result")
                                       if k in kw) + (f" detail={kw['detail']}" if kw.get("detail") else "")
          + (" raw_kept=1" if KEEP_RAW else ""), flush=True)


def raw_paths(kind):
    if kind == "new":
        return [d / f"{s}.mkv" for s in STREAMS] + [d / ".spool"]
    return [d / "frames" / f"{s}.rgb24" for s in STREAMS]


def remove(paths):
    if KEEP_RAW:
        return
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def main():
    if not d.is_dir():
        print(f"REC_TRANSCODE dir={d.name} kind=none result=none detail=no_dir", flush=True)
        return 0
    # 只看原始媒体是否还在：转码成功后 frames-*.jsonl／frames.json 保留，但 *.mkv／*.rgb24 已删
    kind = "new" if any((d / f"{s}.mkv").exists() for s in STREAMS) else \
        "orig" if any((d / "frames" / f"{s}.rgb24").exists() for s in STREAMS) else "none"
    if kind == "none":
        if (d / MP4).exists():  # 上一次已转码（如周期同步被收尾打断后重入）：保留原 transcode.json
            print(f"REC_TRANSCODE dir={d.name} kind=none result=already", flush=True)
        else:
            print(f"REC_TRANSCODE dir={d.name} kind=none result=none", flush=True)
        return 0
    ff = ffmpeg_exe()
    if ff is None:
        emit(kind=kind, result="fail", detail="no_ffmpeg")
        return 2
    tmp_raw, parts = [], []
    try:
        if kind == "new":
            streams = [s for s in STREAMS if (d / f"{s}.mkv").exists()]
            recs = {}
            for s in streams:
                rows = [json.loads(x) for x in (d / f"frames-{s}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
                rows.sort(key=lambda r: int(r["idx"]))
                recs[s] = rows
            n = len(recs[streams[0]])
            if any(len(recs[s]) != n for s in streams):
                emit(kind=kind, frames=n, result="frame_mismatch",
                     detail="stream_len " + ",".join(f"{s}:{len(recs[s])}" for s in streams))
                return 1
            if n == 0:
                remove(raw_paths(kind))
                emit(kind=kind, frames=0, result="empty")
                return 0
            for s in streams:
                if any(r.get("enc") is None for r in recs[s]):
                    emit(kind=kind, frames=n, result="fail", detail=f"enc_none stream={s}")
                    return 2
                w, h = video_size(ff, d / f"{s}.mkv")
                raw = d / f".tc-{s}.raw"
                tmp_raw.append(raw)
                subprocess.run([ff, "-nostdin", "-v", "error", "-y", "-i", str(d / f"{s}.mkv"), "-f", "rawvideo",
                                "-pix_fmt", "rgb24", str(raw)], check=True)
                fsz = w * h * 3
                if raw.stat().st_size % fsz or max(int(r["enc"]) for r in recs[s]) >= raw.stat().st_size // fsz:
                    emit(kind=kind, frames=n, result="fail", detail=f"decoded_short stream={s}")
                    return 2
                fh = open(raw, "rb")
                parts.append((mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ), w, h, [int(r["enc"]) for r in recs[s]]))
        else:
            meta = json.loads((d / "frames" / "frames.json").read_text(encoding="utf-8"))
            if meta.get("pix_fmt", "rgb24") != "rgb24":
                emit(kind=kind, result="fail", detail=f"pix_fmt={meta.get('pix_fmt')}")
                return 2
            st = meta.get("streams") or {}
            streams = [s for s in STREAMS if s in st]
            counts = {s: int(st[s]["count"]) for s in streams}
            n = counts[streams[0]] if streams else 0
            if any(c != n for c in counts.values()):
                emit(kind=kind, frames=n, result="frame_mismatch",
                     detail="stream_len " + ",".join(f"{s}:{c}" for s, c in counts.items()))
                return 1
            if n == 0:
                remove(raw_paths(kind))
                emit(kind=kind, frames=0, result="empty")
                return 0
            for s in streams:
                w, h = int(st[s]["width"]), int(st[s]["height"])
                f = d / "frames" / f"{s}.rgb24"
                if f.stat().st_size != n * w * h * 3:
                    emit(kind=kind, frames=n, result="frame_mismatch",
                         detail=f"raw_size stream={s} bytes={f.stat().st_size} expect={n * w * h * 3}")
                    return 1
                fh = open(f, "rb")
                parts.append((mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ), w, h, list(range(n))))
        W = sum(p[1] for p in parts)
        H = max(p[2] for p in parts)
        out = d / MP4
        part = d / ".episode.part.mp4"
        proc = subprocess.Popen([ff, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", "-f", "mp4", str(part)], stdin=subprocess.PIPE)
        for i in range(n):
            if len(parts) == 1:
                mm, w, h, idx = parts[0]
                off = idx[i] * w * h * 3
                proc.stdin.write(mm[off:off + w * h * 3])
                continue
            buf = bytearray()
            for y in range(H):
                for mm, w, h, idx in parts:
                    if y < h:
                        off = idx[i] * w * h * 3 + y * w * 3
                        buf += mm[off:off + w * 3]
                    else:
                        buf += bytes(w * 3)
            proc.stdin.write(bytes(buf))
        proc.stdin.close()
        if proc.wait() != 0:
            part.unlink(missing_ok=True)
            emit(kind=kind, frames=n, result="fail", detail="encode_rc")
            return 2
        m = count_frames(ff, part)
        if m != n:
            part.unlink(missing_ok=True)
            emit(kind=kind, frames=n, mp4_frames=m, result="frame_mismatch")
            return 1
        part.replace(out)
        for mm, *_ in parts:
            mm.close()
        parts.clear()
        remove(raw_paths(kind))
        emit(kind=kind, frames=n, mp4_frames=m, result="ok", width=W, height=H, mp4_bytes=out.stat().st_size)
        return 0
    except Exception as e:  # noqa: BLE001 逐局失败如实记录，原始帧保留
        emit(kind=kind, result="fail", detail=f"{type(e).__name__}:{str(e)[:200]}".replace(" ", "_"))
        return 2
    finally:
        for mm, *_ in parts:
            try:
                mm.close()
            except Exception:  # noqa: BLE001
                pass
        for p in tmp_raw:
            p.unlink(missing_ok=True)


sys.exit(main())
PY
}

render_official_dir() {  # $1 = 局目录；返回 0 = KEPT／重绘通过／无帧 error 局，非 0 = 失败（调用方据此保留原始帧）
  local d="${1:-}" name lib py ff renderer out rc aside
  name="$(basename "$d")"
  if [[ -z "$d" || ! -d "$d" ]]; then
    echo "OFFICIAL_RENDER=FAIL dir=$name stage=input reason=no_dir"; return 2
  fi
  lib="$(seat_media_dir)"; py="$(seat_media_py)"
  if ! ff="$(seat_media_ffmpeg)"; then
    echo "OFFICIAL_RENDER=FAIL dir=$name stage=input reason=no_ffmpeg"; return 2
  fi
  if [[ ! -f "$lib/official_media_check.py" ]]; then
    echo "OFFICIAL_RENDER=FAIL dir=$name stage=input reason=media_check_missing path=$lib/official_media_check.py"; return 2
  fi
  # 已有官方视频：完整核验通过才保留（KEPT）；不通过的整目录移到隐藏留证目录，不删
  if compgen -G "$d/official/*.mp4" >/dev/null; then
    out="$("$py" "$lib/official_media_check.py" --verify-dir "$d" --ffmpeg "$ff" 2>&1)"; rc=$?
    [[ -n "$out" ]] && echo "$out"
    if (( rc == 0 )); then
      echo "OFFICIAL_RENDER=KEPT dir=$name"; return 0
    fi
    aside="$d/.official-rejected-$(date +%s)-$$"
    mv -T -- "$d/official" "$aside" || { echo "OFFICIAL_RENDER=FAIL dir=$name stage=kept_check reason=move_aside_failed"; return 2; }
    echo "OFFICIAL_RENDER_REJECTED dir=$name moved_to=$(basename "$aside")"
  fi
  renderer="$lib/render_official_video.py"
  if [[ ! -f "$renderer" ]]; then
    echo "OFFICIAL_RENDER=FAIL dir=$name stage=input reason=renderer_missing path=$renderer"; return 2
  fi
  if ! "$py" "$renderer" --help 2>/dev/null | grep -q -- '--source'; then
    echo "OFFICIAL_RENDER=FAIL dir=$name stage=input reason=renderer_no_source_raw（渲染器尚不支持 --source raw）"; return 2
  fi
  out="$("$py" "$renderer" "$d" --source raw --jobs 1 --ffmpeg "$ff" --out-subdir official 2>&1)"; rc=$?
  [[ -n "$out" ]] && echo "$out"
  if (( rc != 0 )); then
    echo "OFFICIAL_RENDER=FAIL dir=$name stage=render reason=renderer_rc_$rc"; return 2
  fi
  out="$("$py" "$lib/official_media_check.py" --verify-dir "$d" --ffmpeg "$ff" 2>&1)"; rc=$?
  [[ -n "$out" ]] && echo "$out"
  if (( rc != 0 )); then
    echo "OFFICIAL_RENDER=FAIL dir=$name stage=verify reason=verify_rc_$rc"; return 2
  fi
  return 0
}
