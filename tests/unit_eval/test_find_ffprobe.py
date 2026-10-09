"""official_render.find_ffprobe 的查找顺序：ROBOMME_FFPROBE → 与 ffmpeg 同目录（未给 ffmpeg 时按录制器 find_ffmpeg，认 V75_FFMPEG）→ PATH。

2026-10-08 拆仓验收 S5：GL 计算节点 PATH 上没有 ffprobe，run_episode 调 render_video 时不传 ffmpeg，旧实现只查 PATH，
8 局网站视频全部记 video_error「找不到 ffprobe」。这里钉住修复后的三条规则与报错。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from robomme_hard_eval.record import official_render as R
from robomme_hard_eval.record import recorder as REC


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def no_path_ffprobe(monkeypatch, tmp_path):
    monkeypatch.delenv(R.ENV_FFPROBE, raising=False)
    monkeypatch.setattr(R.shutil, "which", lambda name: None)
    return tmp_path


def test_env_ffprobe_wins(no_path_ffprobe, monkeypatch):
    probe = _exe(no_path_ffprobe / "x" / "my-ffprobe")
    sib = _exe(no_path_ffprobe / "tools" / "ffprobe")
    monkeypatch.setenv(R.ENV_FFPROBE, str(probe))
    assert R.find_ffprobe(str(sib.with_name("ffmpeg"))) == str(probe)


def test_sibling_of_given_ffmpeg(no_path_ffprobe):
    sib = _exe(no_path_ffprobe / "tools" / "ffprobe")
    assert R.find_ffprobe(str(no_path_ffprobe / "tools" / "ffmpeg")) == str(sib)


def test_without_ffmpeg_uses_recorder_find_ffmpeg(no_path_ffprobe, monkeypatch):
    sib = _exe(no_path_ffprobe / "ff" / "ffprobe")
    monkeypatch.setattr(REC, "find_ffmpeg", lambda: str(no_path_ffprobe / "ff" / "ffmpeg"))
    assert R.find_ffprobe() == str(sib)


def test_not_found_raises_with_hint(no_path_ffprobe, monkeypatch):
    monkeypatch.setattr(REC, "find_ffmpeg", lambda: str(no_path_ffprobe / "none" / "ffmpeg"))
    with pytest.raises(RuntimeError, match=R.ENV_FFPROBE):
        R.find_ffprobe()


def test_env_pointing_to_missing_file_is_ignored(no_path_ffprobe, monkeypatch):
    sib = _exe(no_path_ffprobe / "tools" / "ffprobe")
    monkeypatch.setenv(R.ENV_FFPROBE, str(no_path_ffprobe / "missing"))
    assert R.find_ffprobe(str(no_path_ffprobe / "tools" / "ffmpeg")) == str(sib)
    assert not os.path.exists(no_path_ffprobe / "missing")
