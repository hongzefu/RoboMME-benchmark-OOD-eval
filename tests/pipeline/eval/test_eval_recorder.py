"""C13 评估录制器 ``recorder.EpisodeRecorder``（慢，真 ffmpeg）：无损编码读回逐帧相同、同流重复帧只编一次、
reset 阶段只入队、降级档、拒绝覆盖；以及真实录制器接在真实 ``SeatRunner`` 上、产物直接喂报告。

文件名带 ``eval_`` 前缀：pytest 默认按文件名导入测试模块，避免与其他目录的同名文件冲突。
缺支持 ffv1 的 ffmpeg 时整文件记「未验证」。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import eval_fakes as F
from tests._support.loaders import load_script

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rec_mod():
    mod = load_script("eval-official/recorder.py")
    try:
        mod.find_ffmpeg()
    except RuntimeError as e:
        pytest.skip(f"未验证：{e}")
    return mod


def _frames(vals, hw=16):
    out = []
    for v in vals:
        f = np.zeros((hw, hw, 3), dtype=np.uint8)
        f[..., 0] = v
        f[: hw // 2, :, 1] = 255 - v  # 上下半区不同，避免整帧常数
        f[:, : hw // 3, 2] = (v * 7) % 256
        out.append(f)
    return np.stack(out)


def test_lossless_roundtrip_and_dedup(tmp_path, rec_mod):
    r = rec_mod.EpisodeRecorder(tmp_path / "ep", {"never_degrade": True}, free_gib_fn=lambda p: 1e6)
    r.set_phase("reset")
    front = _frames([1, 2, 3, 2])  # 第 4 帧与第 2 帧逐字节相同
    idx = r.add_frames("front", front, tag="reset")
    r.set_phase("run")
    r.add_frames("front", _frames([9]), tag="step0")
    r.add_frames("wrist", _frames([5, 5]), tag="step0")
    a = np.arange(8, dtype=np.float32)
    r.add_array("exec_action", a, step=0)
    r.add_event({"kind": "note", "v": 1})
    res = r.close({"status": "success", "steps": 1})
    assert idx == [0, 1, 2, 3]
    assert res["RECORDER_VERIFY"] == "PASS" and res["level"] == 0 and res["lossless"] is True
    assert res["frames"] == 7 and res["encoded_frames"] == 4 + 1  # front 去重后 4、wrist 1
    assert res["decode_mismatch"] == 0 and res["dropped"] == 0 and res["errors"] == []
    out = tmp_path / "ep"
    assert not (out / ".spool").exists()
    recs, imgs = rec_mod.load_frames(out, "front")
    assert [x["enc"] for x in recs] == [0, 1, 2, 1, 3]
    for want, got in zip(np.concatenate([front, _frames([9])]), imgs):
        assert np.array_equal(want, got)
    _, wimgs = rec_mod.load_frames(out, "wrist")
    assert all(np.array_equal(x, _frames([5])[0]) for x in wimgs)
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
        pytest.skip("未验证：ffmpeg 不支持 libx264，2 档降级无法编码")
    keep = rec_mod.LEVEL2_KEEP
    n = 2 * keep + 5
    r = rec_mod.EpisodeRecorder(tmp_path / "ep", {}, free_gib_fn=lambda p: 1.0)
    r.add_frames("front", _frames(list(range(n))))
    res = r.close({})
    assert res["level"] == 2 and res["frames"] == n and res["encoded_frames"] == 2 * keep
    recs, _ = rec_mod.load_frames(tmp_path / "ep", "front")
    assert sum(x["enc"] is None for x in recs) == n - 2 * keep
    assert all(x["enc"] is None for x in recs[keep:n - keep])


def test_seat_runner_with_real_recorder_feeds_report(tmp_path, monkeypatch, capsys, rec_mod):
    task, tier = F.v9_cells_sorted()[0]
    ident = F.packaged_identity(task, tier, 0)
    world = F.World({(task, ident["builder_episode"]): [F.Plan(success_at=20)]})
    ec = F.env_client()
    out = tmp_path / "stage" / "s00" / "perceptual-framesamp-modul"
    args = F.seat_args(out, "perceptual-framesamp-modul", ledger=out / "perceptual-framesamp-modul.ledger.jsonl")
    runner = ec.SeatRunner(args, policy_mod=F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()),
                           recorder_factory=lambda d, m: rec_mod.EpisodeRecorder(d, m, free_gib_fn=lambda p: 1e6),
                           builder_factory=lambda t, ms: F.HybridBuilder(t, ms, world),
                           proc_info={"init_timing": {}})
    assert F.run_rows(runner, [ident]) == 0
    (row,) = F.read_jsonl(out / "results.jsonl")
    assert row["status"] == "success" and row["recorder_verify"] == "PASS"
    summary = json.loads((out / "rec" / f"{ident['key']}.a1" / "summary.json").read_text())
    # reset 3 帧 + 20 步各 1 帧
    assert summary["frames"] == 2 * (F.N_RESET_FRAMES + 20)
    manifest = F.write_manifest(tmp_path / "m" / "manifest.json", [ident])
    rc, lines, rep = F.run_report(capsys, manifest, tmp_path / "stage", ["perceptual-framesamp-modul"], tmp_path / "rep", "--expect-total", "1")
    assert rc == 0 and F.verdict(lines, "V8_EVAL_REPORT")["media_unexplained"] == "0"
