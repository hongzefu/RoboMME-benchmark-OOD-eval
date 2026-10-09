"""C13 评估视频搬运 ``dev-scripts/gl/eval_video_mover.py --mode v8 --once``（慢，真 ffmpeg 微型媒体）
与 ``eval_report`` 对 ``--videos`` 产物的判定（``verify_new_videos`` → ``V9_EVAL_VIDEOS`` 判定行）。

搬运脚本以测试子进程实际执行；运行根按生产布局手写（``eval_fakes.Stage``），录像目录里放 ffmpeg 现做的
几帧 ffv1 视频。覆盖：正例（原子搬运、逐文件 sha 一致、源删除、moved.jsonl 记录）、重复／覆盖（同内容再搬不落第二份，
不同内容落 ``.dup1`` 且原件不动）、缺件（少一路视频 → missing、坏视频 → decode_fail、搬后篡改 → sha_mismatch、
无搬运记录 → moved_record_absent）。缺 ffmpeg 时整文件记「未验证」。

``--layout sgeval``：按输出键 ``<label>/<dataset>/<side>/<key>.a<n>/`` 整目录搬、两端逐文件 sha256 相同才删 NFS 源；
本机盘剩余低于阈值（``low_disk``）、rsync 替身制造的 sha 不一致（``sha_mismatch``）都打 ``MOVER_STOP`` 并保留源；
多 mp4 目录跳过；发布中的 ``.incoming`` 不碰；缺省布局行为不变。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import eval_fakes as F

pytestmark = pytest.mark.slow

MOVER = F.REPO / "dev-scripts" / "gl" / "eval_video_mover.py"
POL = "perceptual-framesamp-modul"

if shutil.which("ffmpeg") is None:  # pragma: no cover
    pytest.skip("未验证：缺 ffmpeg", allow_module_level=True)


def _mkv(path: Path, frames: int, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={color}:size=16x16:rate=5", "-frames:v", str(frames), "-c:v", "ffv1", str(path)],
                   check=True)
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    return {"a3": _mkv(d / "a3.mkv", 3, "red"), "b4": _mkv(d / "b4.mkv", 4, "blue"),
            "c5": _mkv(d / "c5.mkv", 5, "green")}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ident(i: int) -> dict:
    task, tier = "TaskA", "xhard1"
    seed = 9_000_000 + i
    return {"task": task, "tier": tier, "seed": seed, "candidate": i, "builder_episode": i, "source_episode": None,
            "spec_sha256": hashlib.sha256(str(seed).encode()).hexdigest(), "effective_max_steps": 10,
            "key": f"{task}_{tier}_{seed}"}


def _put_media(st, ident, no, files: dict[str, Path | bytes]):
    d = st.dir / "rec" / f"{ident['key']}.a{no}"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        if isinstance(src, bytes):
            (d / name).write_bytes(src)
        else:
            shutil.copy2(src, d / name)
    (d / "summary.json").write_text('{"RECORDER_VERIFY": "PASS"}', encoding="utf-8")
    return d


def _mover(stage: Path, dest: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(MOVER), "--mode", "v8", "--stage", str(stage), "--dest", str(dest),
                        "--policies", POL, "--once", "--stable-sec", "0", "--interval", "0", "--decode-workers", "1",
                        "--once-max-wait", "30"], capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout


def _video_line(manifest: Path, stage: Path, videos: Path, n: int) -> dict:
    """真实 build_report + verify_new_videos + v9_lines 出 V9_EVAL_VIDEOS 行（新评 n 局、无复用）。"""
    er = F.eval_report()
    rep = er.build_report(manifest, stage, [POL], videos, partial=False, expect_total=n, cap=1 << 30,
                          keep_internal=True)
    vid = er.verify_new_videos(rep, manifest, [POL], videos)
    stub = {"new": n, "reused": 0, "total": n, "count_mismatch_detail": []}
    rep["v9"] = stub
    lines = er.v9_lines(rep, stub, vid, expect_new=n, expect_reused=0)
    return F.verdict(list(lines), "V9_EVAL_VIDEOS")


def _setup(tmp_path, media, n=2):
    stage = tmp_path / "stage"
    st = F.Stage(stage, POL)
    idents = [_ident(i) for i in range(n)]
    for i, ident in enumerate(idents):
        st.accepted(ident, f"a{i}", "success", media=False, rec_dir=str(st.dir / "rec" / f"{ident['key']}.a1"))
    manifest = F.write_manifest(tmp_path / "m" / "manifest.json", idents)
    return stage, st, idents, manifest


def test_positive_atomic_move_and_report(tmp_path, media):
    stage, st, idents, manifest = _setup(tmp_path, media)
    srcs = [_put_media(st, x, 1, {"front.mkv": media["a3"], "wrist.mkv": media["b4"]}) for x in idents]
    dest = tmp_path / "videos"
    rc, out = _mover(stage, dest)
    v = F.verdict(out.splitlines(), "V8_EVAL_VIDEOS")
    assert rc == 0, out
    assert v[""] == "PASS" and v["expected"] == "2" and v["videos"] == "2" and v["stage_left"] == "0"
    moved = F.read_jsonl(dest / "moved.jsonl")
    assert len(moved) == 2
    for x, src in zip(idents, srcs):
        d = dest / POL / x["tier"] / x["task"] / f"{x['key']}.a1"
        assert _sha(d / "front.mkv") == _sha(media["a3"]) and _sha(d / "wrist.mkv") == _sha(media["b4"])
        assert not src.exists()  # 源已删
        rec = next(r for r in moved if r["key"] == x["key"])
        assert rec["files"]["front.mkv"] == _sha(media["a3"]) and rec["dest"] == str(d)
    assert not any((dest / ".incoming").iterdir())
    vv = _video_line(manifest, stage, dest, 2)
    assert vv[""] == "PASS" and vv["expected"] == "2" and vv["videos"] == "2"


def test_rerun_same_content_no_dup_and_different_content_goes_to_dup1(tmp_path, media):
    stage, st, idents, manifest = _setup(tmp_path, media, n=1)
    (x,) = idents
    dest = tmp_path / "videos"
    _put_media(st, x, 1, {"front.mkv": media["a3"], "wrist.mkv": media["b4"]})
    assert _mover(stage, dest)[0] == 0
    target = dest / POL / x["tier"] / x["task"] / f"{x['key']}.a1"
    # 同内容再出现（删源前中断后的重跑）：不落第二份，源照删
    _put_media(st, x, 1, {"front.mkv": media["a3"], "wrist.mkv": media["b4"]})
    rc, out = _mover(stage, dest)
    assert rc == 0, out
    assert not Path(f"{target}.dup1").exists()
    # 不同内容：落 .dup1，原件一个字节不动
    _put_media(st, x, 1, {"front.mkv": media["c5"], "wrist.mkv": media["c5"]})
    rc, out = _mover(stage, dest)
    assert rc == 0, out
    dup = Path(f"{target}.dup1")
    assert _sha(dup / "front.mkv") == _sha(media["c5"])
    assert _sha(target / "front.mkv") == _sha(media["a3"]) and _sha(target / "wrist.mkv") == _sha(media["b4"])
    assert not (st.dir / "rec" / f"{x['key']}.a1").exists()


def test_missing_and_broken_media(tmp_path, media):
    stage, st, idents, manifest = _setup(tmp_path, media, n=2)
    a, b = idents
    _put_media(st, a, 1, {"front.mkv": media["a3"]})  # 少 wrist
    _put_media(st, b, 1, {"front.mkv": media["a3"], "wrist.mkv": b"not a video"})  # 坏视频
    dest = tmp_path / "videos"
    rc, out = _mover(stage, dest)
    v = F.verdict(out.splitlines(), "V8_EVAL_VIDEOS")
    assert rc == 1
    assert v[""] == "FAIL" and v["missing"] == "1" and v["decode_fail"] == "1" and v["videos"] == "0"
    vv = _video_line(manifest, stage, dest, 2)
    assert vv[""] == "FAIL" and vv["missing"] == "1" and vv["decode_fail"] == "1" and vv["videos"] == "0"


def test_report_catches_tamper_and_absent_move_record(tmp_path, media):
    stage, st, idents, manifest = _setup(tmp_path, media, n=2)
    for x in idents:
        _put_media(st, x, 1, {"front.mkv": media["a3"], "wrist.mkv": media["b4"]})
    dest = tmp_path / "videos"
    assert _mover(stage, dest)[0] == 0
    a, b = idents
    da = dest / POL / a["tier"] / a["task"] / f"{a['key']}.a1"
    shutil.copy2(media["c5"], da / "wrist.mkv")  # 搬后被替换：读得出帧但 sha 与搬运记录不符
    log = dest / "moved.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in F.read_jsonl(log) if r["key"] != b["key"]) + "\n",
                   encoding="utf-8")  # b 的搬运记录丢失
    vv = _video_line(manifest, stage, dest, 2)
    assert vv[""] == "FAIL" and vv["sha_mismatch"] == "1" and vv["moved_record_absent"] == "1" and vv["videos"] == "0"


# ---------------------------------------------------------------- --layout sgeval（1003 计划 1.6）

SG_KEYS = ("groundsg-ground-sg-oracle/hard-verify/orig/PickXtimes_xhard0_510300.a1",
           "groundsg-ground-sg-oracle/hard-verify/new/PickXtimes_xhard0_510300.a1",
           "pp/ood/new/PickXtimes_xhard1_16100000.a2")


def _sg_stage(tmp_path: Path, keys=SG_KEYS, mp4s: int = 1) -> Path:
    """按 run_eval_gl.sh／run_official_hard.sh 的发布布局手写媒体根；另放一个发布中的 .incoming 目录（不得碰）。"""
    stage = tmp_path / "media"
    for k in keys:
        d = stage / k
        d.mkdir(parents=True)
        for i in range(mp4s):
            (d / ("episode.mp4" if i == 0 else f"extra{i}.mp4")).write_bytes(f"mp4-{k}-{i}".encode())
        (d / "trace.jsonl").write_text(json.dumps({"kind": "header", "key": k}) + "\n", encoding="utf-8")
        (d / "summary.json").write_text('{"RECORDER_VERIFY": "PASS"}', encoding="utf-8")
    inc = stage / "pp" / "ood" / "new" / ".incoming" / "half"
    inc.mkdir(parents=True)
    (inc / "episode.mp4").write_bytes(b"partial")
    return stage


def _sg_mover(stage: Path, dest: Path, *extra: str, env=None) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(MOVER), "--layout", "sgeval", "--stage", str(stage), "--dest", str(dest),
                        "--once", "--stable-sec", "0", "--interval", "0", *extra], capture_output=True, text=True,
                       timeout=180, env=env)
    return p.returncode, p.stdout


def test_sgeval_moves_by_output_key_with_sha_and_deletes_source(tmp_path):
    stage = _sg_stage(tmp_path)
    want = {k: {f.name: _sha(f) for f in (stage / k).iterdir()} for k in SG_KEYS}
    dest = tmp_path / "videos"
    rc, out = _sg_mover(stage, dest, "--min-free-gib", "0")
    assert rc == 0, out
    v = F.verdict(out.splitlines(), "SGEVAL_MOVE")
    assert v[""] == "PASS" and v["moved"] == "3" and v["left"] == "0" and v["stopped"] == "0"
    moved = F.read_jsonl(dest / "moved.jsonl")
    assert sorted(r["key"] for r in moved) == sorted(SG_KEYS) and {r["mode"] for r in moved} == {"sgeval"}
    for k in SG_KEYS:
        assert {f.name: _sha(f) for f in (dest / k).iterdir()} == want[k]  # 两端逐文件 sha256 相同
        assert not (stage / k).exists()  # 核对过才删源
        assert next(r for r in moved if r["key"] == k)["files"] == want[k]
    assert (stage / "pp" / "ood" / "new" / ".incoming" / "half" / "episode.mp4").is_file()  # 发布中的不碰
    # 默认布局不受影响：同一媒体根用缺省（v8）布局扫不到任何结果行，不搬不删
    stage2 = _sg_stage(tmp_path / "again")
    p = subprocess.run([sys.executable, str(MOVER), "--stage", str(stage2), "--dest", str(tmp_path / "v8dest"),
                        "--policies", POL, "--once", "--stable-sec", "0", "--interval", "0"],
                       capture_output=True, text=True, timeout=180)
    assert "V8_EVAL_VIDEOS=FAIL" in p.stdout and all((stage2 / k / "episode.mp4").is_file() for k in SG_KEYS)


def test_sgeval_low_disk_stops_and_keeps_source(tmp_path):
    stage = _sg_stage(tmp_path)
    dest = tmp_path / "videos"
    rc, out = _sg_mover(stage, dest, "--min-free-gib", "1e9")
    assert rc == 1
    assert "MOVER_STOP reason=low_disk" in out
    v = F.verdict(out.splitlines(), "SGEVAL_MOVE")
    assert v[""] == "FAIL" and v["moved"] == "0" and v["stopped"] == "1" and v["left"] == "3"
    assert all((stage / k / "episode.mp4").is_file() for k in SG_KEYS)


def test_sgeval_sha_mismatch_stops_and_keeps_source(tmp_path):
    """rsync 替身：照常拷贝后往目的端 trace.jsonl 追加一个字节，模拟传输损坏。"""
    real = shutil.which("rsync")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "rsync"
    fake.write_text(f'#!/usr/bin/env bash\n"{real}" "$@" || exit $?\ndst="${{@: -1}}"\n'
                    f'printf x >> "${{dst%/}}/trace.jsonl"\n', encoding="utf-8")
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    stage = _sg_stage(tmp_path, keys=SG_KEYS[:1])
    dest = tmp_path / "videos"
    rc, out = _sg_mover(stage, dest, "--min-free-gib", "0", env=env)
    assert rc == 1, out
    assert f"MOVER_STOP reason=sha_mismatch dir={SG_KEYS[0]}" in out
    assert (stage / SG_KEYS[0] / "trace.jsonl").is_file() and not (dest / SG_KEYS[0]).exists()
    assert not any((dest / ".incoming").iterdir())  # 临时目录已删


def test_sgeval_multi_mp4_is_skipped(tmp_path):
    stage = _sg_stage(tmp_path, keys=SG_KEYS[:1], mp4s=2)
    rc, out = _sg_mover(stage, tmp_path / "videos", "--min-free-gib", "0")
    assert rc == 1 and f"MOVER_SKIP reason=multi_mp4 dir={SG_KEYS[0]}" in out
    v = F.verdict(out.splitlines(), "SGEVAL_MOVE")
    assert v[""] == "FAIL" and v["skipped"] == "1" and v["left"] == "1"
    assert (stage / SG_KEYS[0] / "episode.mp4").is_file()
