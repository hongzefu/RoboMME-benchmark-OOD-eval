"""录制器 AV1 4:4:4（R8）：真 ffmpeg 编几十帧小视频、ffprobe 核 av1／yuv444p、核帧数时间戳与可解码、
``meta.json`` 记 ``raw_codec``；``close()`` 截止时间在编码子进程卡死时不挂死。"""
from __future__ import annotations

import json
import shutil
import stat
import subprocess
import time

import numpy as np
import pytest

from robomme_ood_eval.record import recorder as R


def _ffmpeg_or_skip() -> str:
    try:
        return R.find_ffmpeg()
    except RuntimeError as e:
        pytest.skip(f"未验证：本机没有支持 libaom-av1 的 ffmpeg（{e}）")


def _frames(n: int, h: int = 64, w: int = 64, cam: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((n, h, w, 3), np.uint8)
    for i in range(n):
        out[i, ..., 0] = (xx * 4 + 5 * i) % 256
        out[i, ..., 1] = (yy * 4 + 40 * cam) % 256
        out[i, ..., 2] = 100
        out[i, 20:40, (2 * i) % 40:(2 * i) % 40 + 16] = (255, 0, 0)
    return out


def _probe(path) -> tuple:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("未验证：本机没有 ffprobe")
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt,color_range,color_space,width,height", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return (s["codec_name"], s["pix_fmt"], s.get("color_range"), s.get("color_space"), str(s["width"]),
            str(s["height"]))


def test_constants_are_r8():
    assert R.AV1_ENCODE_ARGS == ("-c:v", "libaom-av1", "-cpu-used", "4", "-crf", "24", "-b:v", "0", "-pix_fmt",
                                 "yuv444p", "-g", "300", "-r", "30", "-row-mt", "1", "-threads", "2")
    assert "bt709" in R.AV1_COLOR_FILTER and "out_range=full" in R.AV1_COLOR_FILTER
    assert R.RAW_CODEC == "av1-yuv444p" and R.LEGACY_RAW_CODEC == "ffv1"
    cmd = R.av1_encode_cmd("ffmpeg", 256, 256, "x.mkv")
    assert cmd[-1] == "x.mkv" and "libaom-av1" in cmd and "-s" in cmd and cmd[cmd.index("-s") + 1] == "256x256"


def test_encode_av1_yuv444p_and_verify(tmp_path):
    _ffmpeg_or_skip()
    front, wrist = _frames(40), _frames(40, cam=1)
    rec = R.EpisodeRecorder(tmp_path / "ep", {"identity": {"task": "T"}}, free_gib_fn=lambda _p: 10_000.0)
    rec.set_phase("reset")
    rec.add_frames("front", front[:5], tag="reset")
    rec.add_frames("wrist", wrist[:5], tag="reset")
    rec.set_phase("run")
    for i in range(5, 40):
        rec.add_frames("front", front[i], tag=f"step{i - 5}")
        rec.add_frames("wrist", wrist[i], tag=f"step{i - 5}")
        rec.add_array("exec_action", np.full(8, i, np.float64), step=i - 5)
    rec.add_frames("front", front[39], tag="dup")  # 重复帧只编码一份
    res = rec.close({"status": "success"})
    assert res["RECORDER_VERIFY"] == "PASS", res
    assert res["errors"] == [] and res["dropped"] == 0 and res["decode_mismatch"] == 0
    assert res["lossless"] is False and res["raw_codec"] == "av1-yuv444p"
    assert res["streams"]["front"]["decoded"] == 40 and res["streams"]["front"]["frames"] == 41
    assert res["streams"]["front"]["timestamps_ok"] is True
    ep = tmp_path / "ep"
    for s in ("front", "wrist"):
        codec, pix, rng, space, w, h = _probe(ep / f"{s}.mkv")
        assert (codec, pix, rng, space, w, h) == ("av1", "yuv444p", "pc", "bt709", "64", "64")
    meta = json.loads((ep / "meta.json").read_text())
    assert meta["raw_codec"] == "av1-yuv444p" and meta["encode_args"] == list(R.AV1_ENCODE_ARGS)
    assert R.raw_codec_of(ep) == "av1-yuv444p"
    assert not (ep / ".spool").exists()  # 核验通过才删原始块
    with np.load(ep / "arrays.npz") as z:
        assert z["exec_action__00000"].dtype == np.float64 and z["exec_action__00034"][0] == 39
    recs, imgs = R.load_frames(ep, "front")
    assert len(recs) == 41 and recs[-1]["enc"] == 39
    psnr = []
    for a, b in zip(front, imgs[:40]):
        mse = float(((a.astype(np.float64) - b) ** 2).mean())
        psnr.append(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))
    assert min(psnr) > 30.0, psnr  # 有损但色彩还原正确（bt709 全范围编解码对称）


def test_legacy_meta_without_raw_codec_is_ffv1(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"codec": "ffv1"}))
    assert R.raw_codec_of(tmp_path) == "ffv1"
    assert R.raw_codec_of(tmp_path / "missing") == "ffv1"


def test_verify_detects_frame_count_mismatch(tmp_path):
    _ffmpeg_or_skip()
    rec = R.EpisodeRecorder(tmp_path / "ep", {}, free_gib_fn=lambda _p: 10_000.0, encode_async=False)
    rec.add_frames("front", _frames(6))
    st = rec._streams["front"]
    res = rec.close({})
    assert res["RECORDER_VERIFY"] == "PASS"
    st.n_enqueued += 2  # 人为声称多编了 2 帧
    v = rec._verify_stream(st)
    assert v["mismatch"] == 2 and v["decoded"] == 6


def _stuck_ffmpeg(tmp_path):
    """假 ffmpeg：编码（读管道）时永远不读 stdin；解码调用立即失败。"""
    p = tmp_path / "fake-ffmpeg"
    p.write_text("#!/bin/sh\ncase \"$*\" in *pipe:0*) exec sleep 600;; esac\nexit 1\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def test_close_deadline_does_not_hang_when_encoder_stuck(tmp_path, monkeypatch):
    fake = _stuck_ffmpeg(tmp_path)
    monkeypatch.setattr(R, "find_ffmpeg", lambda: fake)
    rec = R.EpisodeRecorder(tmp_path / "ep", {}, free_gib_fn=lambda _p: 10_000.0, queue_frames=2,
                            close_deadline_s=2.0)
    rec.set_phase("run")
    big = _frames(3, 256, 256)  # 单帧 192 KiB > 管道缓冲，第一帧写入即卡住
    rec.add_frames("front", big[0])
    time.sleep(0.5)  # 写线程取走第一帧、卡在 ffmpeg stdin
    rec.add_frames("front", big[1])
    rec.add_frames("front", big[2])  # 队列（容量 2）此时已满
    t0 = time.monotonic()
    res = rec.close({})
    took = time.monotonic() - t0
    assert took < 40, took
    assert res["RECORDER_VERIFY"] == "FAIL"
    assert any("timed out during close" in e for e in res["errors"]), res["errors"]
    assert (tmp_path / "ep" / ".spool").exists()  # 核验未通过，原始块保留
