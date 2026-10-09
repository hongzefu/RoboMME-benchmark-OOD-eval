"""The episode_id the new side hands to the official loop must be short: the official overlay video filename contains
the full task goal, and going over 255 bytes makes ffmpeg fail with Broken pipe.

Observed 2026-10-05 on the cluster and locally (second tier): all 12 new-side SwingXtimes xhard0 episodes ended in
error (OSError: [Errno 32] Broken pipe) while episode_id was ``SwingXtimes_xhard0_530300.a1``. Expected values are
hand-written.
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
        name = f"SwingXtimes_ep{eid}_{flag}_{SWING_GOAL}_hard.mp4"   # same composition as official eval.py
        assert len(name.encode()) <= 255, (flag, len(name.encode()))
    long_tag = "SwingXtimes_xhard0_534700.a9"
    assert len(f"SwingXtimes_ep{long_tag}_timeout_{SWING_GOAL}_hard.mp4".encode()) > 255   # the old scheme really exceeds the limit
