"""新侧交给官方循环的 episode_id 必须短：官方叠字视频文件名含整段任务目标，超过 255 字节即 ffmpeg Broken pipe。

2026-10-05 GL 与本机第二档实测：SwingXtimes xhard0 新侧 12 局全部 error（OSError: [Errno 32] Broken pipe），
当时 episode_id 用的是 ``SwingXtimes_xhard0_530300.a1``。期望值手写。
"""
from __future__ import annotations

from tests._support.loaders import load_script

SWING_GOAL = ("pick up the green cube, move it to the top of the right-side target, then move it to the top of the "
              "left-side target, repeating this back-and-forth motion three times, finally press the button to stop")


def _mc():
    return load_script("eval-official/groundsg_client.py")


def test_short_official_episode_id():
    mc = _mc()
    assert mc.official_episode_id({"source_episode": 3, "builder_episode": 0}, "SwingXtimes_xhard0_530300.a1") == "3a1"
    assert mc.official_episode_id({"source_episode": None, "builder_episode": 17}, "VideoUnmask_xhard1_16600000.a2") == "17a2"
    assert mc.official_episode_id({"source_episode": 7}, "no_suffix") == "7a1"


def test_official_video_filename_fits_255_bytes():
    mc = _mc()
    eid = mc.official_episode_id({"source_episode": 47, "builder_episode": 11}, "SwingXtimes_xhard0_534700.a9")
    for flag in ("success", "fail", "timeout", "unknown"):
        name = f"SwingXtimes_ep{eid}_{flag}_{SWING_GOAL}_hard.mp4"   # 官方 eval.py 的拼法
        assert len(name.encode()) <= 255, (flag, len(name.encode()))
    long_tag = "SwingXtimes_xhard0_534700.a9"
    assert len(f"SwingXtimes_ep{long_tag}_timeout_{SWING_GOAL}_hard.mp4".encode()) > 255   # 旧做法确实超限
