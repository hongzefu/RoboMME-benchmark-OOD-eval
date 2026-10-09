"""C13 评估录制器接在真实 ``SeatRunner`` 上（慢，真 ffmpeg）：GL 席位用真实 AV1 录制器跑一局、产物直接喂报告。

由 ``test_eval_recorder.py`` 拆出（席位层 ``SeatRunner`` 只在 dev 侧）。缺支持 libaom-av1 的 ffmpeg 时整文件记「未验证」。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import eval_fakes_dev as F
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


def test_seat_runner_with_real_recorder_feeds_report(tmp_path, monkeypatch, capsys, rec_mod):
    """拆仓后：GL 席位（常驻 Policy + 动态队列）用真实 AV1 录制器跑一局：局结果 ``recorder_verify=PASS``、录制器
    ``summary.json`` 帧数 = 两路 ×（reset 帧 + 20 步各 1 帧）；本局结果行喂评估包汇总 ``robomme_ood_eval.report``
    （拆仓后的汇总入口，读 ``rollouts/<模型>/<数据集>/seed<n>/results.jsonl``；旧 ``eval_report`` 读的 ``sNN/<policy>/``
    布局新席位不再产出）。"""
    from robomme_ood_eval import report

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
