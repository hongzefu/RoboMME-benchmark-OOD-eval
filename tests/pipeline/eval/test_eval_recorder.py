"""C13 evaluation recorder ``recorder.EpisodeRecorder`` (slow, real ffmpeg): with lossy AV1 4:4:4 encoding (split plan
red line R8), the read-back frame count, per-frame pre-encoding sha256 and dedup mapping are exact, the image is
approximate (PSNR > 30 dB), duplicate frames in a stream are encoded only once, the reset phase only queues, degrade
levels work, and overwriting is refused. Cases with the real recorder attached to a real ``SeatRunner`` are in
``test_eval_recorder_seat.py``.

The file name carries the ``eval_`` prefix: pytest imports test modules by file name by default, so this avoids
clashing with same-named files in other directories.
Without an ffmpeg that supports libaom-av1 the whole file is marked "Not verified".
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests._support.loaders import load_script

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rec_mod():
    mod = load_script("eval-official/recorder.py")
    try:
        mod.find_ffmpeg()
    except RuntimeError as e:
        pytest.skip(f"Not verified: {e}")
    return mod


def _frames(vals, hw=16):
    out = []
    for v in vals:
        f = np.zeros((hw, hw, 3), dtype=np.uint8)
        f[..., 0] = v
        f[: hw // 2, :, 1] = 255 - v  # top and bottom halves differ, so the frame is not constant
        f[:, : hw // 3, 2] = (v * 7) % 256
        out.append(f)
    return np.stack(out)


def _psnr(a, b) -> float:
    mse = float(((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def test_av1_roundtrip_and_dedup(tmp_path, rec_mod):
    """Lossy AV1 (R8): frame count, per-frame pre-encoding sha256, dedup mapping and raw arrays are exact; the image is
    only approximate (PSNR > 30 dB) and is no longer required to be byte-identical."""
    r = rec_mod.EpisodeRecorder(tmp_path / "ep", {"never_degrade": True}, free_gib_fn=lambda p: 1e6)
    r.set_phase("reset")
    front = _frames([1, 2, 3, 2], hw=64)  # frame 4 is byte-identical to frame 2
    idx = r.add_frames("front", front, tag="reset")
    r.set_phase("run")
    r.add_frames("front", _frames([9], hw=64), tag="step0")
    r.add_frames("wrist", _frames([5, 5], hw=64), tag="step0")
    a = np.arange(8, dtype=np.float32)
    r.add_array("exec_action", a, step=0)
    r.add_event({"kind": "note", "v": 1})
    res = r.close({"status": "success", "steps": 1})
    assert idx == [0, 1, 2, 3]
    assert res["RECORDER_VERIFY"] == "PASS" and res["level"] == 0
    assert res["lossless"] is False and res["raw_codec"] == rec_mod.RAW_CODEC == "av1-yuv444p"
    assert res["frames"] == 7 and res["encoded_frames"] == 4 + 1  # front: 4 after dedup, wrist: 1
    assert res["decode_mismatch"] == 0 and res["dropped"] == 0 and res["errors"] == []
    assert res["streams"]["front"]["timestamps_ok"] is True and res["streams"]["front"]["decoded"] == 4
    out = tmp_path / "ep"
    assert not (out / ".spool").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["raw_codec"] == "av1-yuv444p" and rec_mod.raw_codec_of(out) == "av1-yuv444p"
    recs, imgs = rec_mod.load_frames(out, "front")
    assert [x["enc"] for x in recs] == [0, 1, 2, 1, 3]
    allf = np.concatenate([front, _frames([9], hw=64)])
    assert [x["sha256"] for x in recs] == [rec_mod.frame_sha256(f) for f in allf]  # pre-encoding hashes exact
    psnr = [_psnr(want, got) for want, got in zip(allf, imgs)]
    assert len(psnr) == 5 and min(psnr) > 30.0, psnr
    wrecs, wimgs = rec_mod.load_frames(out, "wrist")
    assert [x["enc"] for x in wrecs] == [0, 0]
    assert min(_psnr(x, _frames([5], hw=64)[0]) for x in wimgs) > 30.0
    idx_rows = [json.loads(x) for x in (out / "arrays-index.jsonl").read_text().splitlines()]
    assert idx_rows[0]["name"] == "exec_action" and idx_rows[0]["dtype"] == a.dtype.str
    arrs = np.load(out / "arrays.npz")
    assert np.array_equal(arrs[idx_rows[0]["key"]], a)
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    assert [e["phase"] for e in events if e["kind"] == "phase"] == ["reset", "run"]
    assert json.loads((out / "summary.json").read_text())["summary"] == {"status": "success", "steps": 1}


def test_refuses_non_empty_dir(tmp_path, rec_mod):
    d = tmp_path / "ep"
    d.mkdir()
    (d / "x").write_text("1")
    with pytest.raises(FileExistsError):
        rec_mod.EpisodeRecorder(d, {}, free_gib_fn=lambda p: 1e6)


@pytest.mark.parametrize("free,meta,level", [(1e6, {}, 0), (100.0, {"never_degrade": True}, 0),
                                             (100.0, {"baseline": True}, 0)])
def test_degrade_level_respects_never_degrade(tmp_path, rec_mod, free, meta, level):
    r = rec_mod.EpisodeRecorder(tmp_path / "ep", meta, free_gib_fn=lambda p: free)
    r.add_frames("front", _frames([1]))
    res = r.close({})
    assert res["level"] == level and res["RECORDER_VERIFY"] == "PASS"


def test_level2_keeps_head_and_tail_only(tmp_path, rec_mod):
    if not rec_mod._FFMPEG_CACHE.get("libx264"):
        pytest.skip("Not verified: ffmpeg lacks libx264, cannot encode degrade level 2")
    keep = rec_mod.LEVEL2_KEEP
    n = 2 * keep + 5
    r = rec_mod.EpisodeRecorder(tmp_path / "ep", {}, free_gib_fn=lambda p: 1.0)
    r.add_frames("front", _frames(list(range(n))))
    res = r.close({})
    assert res["level"] == 2 and res["frames"] == n and res["encoded_frames"] == 2 * keep
    recs, _ = rec_mod.load_frames(tmp_path / "ep", "front")
    assert sum(x["enc"] is None for x in recs) == n - 2 * keep
    assert all(x["enc"] is None for x in recs[keep:n - keep])
