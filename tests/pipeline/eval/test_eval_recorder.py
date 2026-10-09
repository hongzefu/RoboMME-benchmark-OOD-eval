"""C13 评估录制器 ``recorder.EpisodeRecorder``（慢，真 ffmpeg）：AV1 4:4:4 有损编码（拆分方案红线 R8）读回帧数、
逐帧编码前 sha256 与去重映射精确、画面近似（PSNR > 30 dB）、同流重复帧只编一次、reset 阶段只入队、降级档、拒绝覆盖；
以及真实录制器接在真实 ``SeatRunner`` 上、产物直接喂报告。

文件名带 ``eval_`` 前缀：pytest 默认按文件名导入测试模块，避免与其他目录的同名文件冲突。
缺支持 libaom-av1 的 ffmpeg 时整文件记「未验证」。
"""
from __future__ import annotations

import json
from pathlib import Path

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


def _psnr(a, b) -> float:
    mse = float(((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def test_av1_roundtrip_and_dedup(tmp_path, rec_mod):
    """AV1 有损（R8）：帧数、逐帧编码前 sha256、去重映射与原始数组精确；画面只近似（PSNR > 30 dB），不再要求逐字节。"""
    r = rec_mod.EpisodeRecorder(tmp_path / "ep", {"never_degrade": True}, free_gib_fn=lambda p: 1e6)
    r.set_phase("reset")
    front = _frames([1, 2, 3, 2], hw=64)  # 第 4 帧与第 2 帧逐字节相同
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
    assert res["frames"] == 7 and res["encoded_frames"] == 4 + 1  # front 去重后 4、wrist 1
    assert res["decode_mismatch"] == 0 and res["dropped"] == 0 and res["errors"] == []
    assert res["streams"]["front"]["timestamps_ok"] is True and res["streams"]["front"]["decoded"] == 4
    out = tmp_path / "ep"
    assert not (out / ".spool").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["raw_codec"] == "av1-yuv444p" and rec_mod.raw_codec_of(out) == "av1-yuv444p"
    recs, imgs = rec_mod.load_frames(out, "front")
    assert [x["enc"] for x in recs] == [0, 1, 2, 1, 3]
    allf = np.concatenate([front, _frames([9], hw=64)])
    assert [x["sha256"] for x in recs] == [rec_mod.frame_sha256(f) for f in allf]  # 编码前字节的哈希仍逐帧精确
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
    """拆仓后：GL 席位（常驻 Policy + 动态队列）用真实 AV1 录制器跑一局：局结果 ``recorder_verify=PASS``、录制器
    ``summary.json`` 帧数 = 两路 ×（reset 帧 + 20 步各 1 帧）；本局结果行喂评估包汇总 ``robomme_hard_eval.report``
    （拆仓后的汇总入口，读 ``rollouts/<模型>/<数据集>/seed<n>/results.jsonl``；旧 ``eval_report`` 读的 ``sNN/<policy>/``
    布局新席位不再产出）。"""
    from robomme_hard_eval import report

    task, tier = F.v9_cells_sorted()[0]
    ident = F.packaged_identity(task, tier, 0)
    world = F.World({(task, ident["builder_episode"]): [F.Plan(success_at=20)]})
    runner = F.make_runner(tmp_path / "stage", "perceptual-framesamp-modul",
                           F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                           recorder_factory=lambda d, m: rec_mod.EpisodeRecorder(d, m, free_gib_fn=lambda p: 1e6))
    assert F.run_rows(runner, [ident]) == 0
    (row,) = F.read_jsonl(runner.results_path)
    raw = Path(row["result"]).parent
    res = json.loads((raw / "result.json").read_text())
    assert row["status"] == "success" and res["recorder_verify"] == "PASS"
    summary = json.loads((raw / "summary.json").read_text())
    # reset 3 帧 + 20 步各 1 帧，front、wrist 两路
    assert summary["frames"] == 2 * (F.N_RESET_FRAMES + 20) and summary["RECORDER_VERIFY"] == "PASS"
    capsys.readouterr()
    log = report.summarize(raw.parent.parent)
    assert (log["episodes"], log["counted"], log["success"], log["infra"]) == (1, 1, 1, 0)
    assert log["tasks"][task]["success_rate"] == 1.0 and log["total_success_rate"] == 1.0
    assert "EVAL_LOG " in capsys.readouterr().out and (raw.parent.parent / "log.json").is_file()
