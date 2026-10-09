"""``scripts/evaluate.py`` 端到端（CPU）：dummy 模型 + 假环境 + 真录制器（AV1）+ 真官方 RolloutRecorder 渲染。

与 S2 的 ``EVAL_EPISODE`` 判定同形：一次调用 ``--dataset hard-verify,ood --episodes 0:1``，同一个 Policy 先后跑两档，
核 loads=1 resets=2 closes=1、每局 raw 产物（front／wrist mkv 为 av1／yuv444p，arrays.npz、trace.jsonl、
result.json）、网站视频（av1／yuv420p）与 log.json；重跑全部跳过。真实仿真冒烟由主会话合并后跑。"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from robomme_ood_eval import episode as E
from robomme_ood_eval import policy as P
from robomme_ood_eval.record import official_render, recorder
from tests.unit_eval import fakes

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "evaluate.py"


def official_root() -> Path | None:
    """官方 RolloutRecorder 所在仓根：本检出子模块有就用本检出，否则（worktree 子模块为空）用主检出。"""
    if (REPO / official_render.OFFICIAL_UTILS_REL).is_file():
        return REPO
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    main = Path(common).parent
    return main if (main / official_render.OFFICIAL_UTILS_REL).is_file() else None


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_cli_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe(path: Path) -> tuple:
    out = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt", "-of", "json", str(path)], capture_output=True, text=True,
                         check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["codec_name"], s["pix_fmt"]


def test_help_prints():
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--model" in r.stdout and "--stop-server" in r.stdout and "--episodes" in r.stdout


def test_scripts_dir_has_only_evaluate():
    assert sorted(p.name for p in (REPO / "scripts").iterdir()) == ["evaluate.py"]


def test_parse_args_and_cfg_passthrough():
    mod = load_script()
    assert mod.parse_episodes("0:2") == (0, 2)
    with pytest.raises(Exception):
        mod.parse_episodes("3:1")
    args, rest = mod.build_parser().parse_known_args(
        ["--model", "groundsg", "--groundsg-variant", "ground-sg-oracle", "--dataset", "ood", "--seed", "7",
         "--out", "/tmp/x", "--gpus", "0,1", "--cfg", "warmup=false", "--port-base", "18000", "--foo-bar", "3",
         "--flag"])
    cfg = mod.make_cfg(args, mod.parse_extra(rest))
    assert cfg == {"groundsg_variant": "ground-sg-oracle", "port_base": 18000, "gpus": [0, 1], "warmup": False,
                   "foo_bar": 3, "flag": True}


def test_stop_server_flag(tmp_path, monkeypatch):
    mod = load_script()
    seen = []
    monkeypatch.setattr(P.ServerProcess, "stop_by_metadata", staticmethod(lambda p: seen.append(p) or "term"))
    assert mod.main(["--stop-server", str(tmp_path / "server-metadata-1.json")]) == 0
    assert seen == [tmp_path / "server-metadata-1.json"]


def test_dummy_end_to_end_two_datasets_resident(tmp_path, monkeypatch, capsys):
    try:
        recorder.find_ffmpeg()
    except RuntimeError as e:
        pytest.skip(f"未验证：本机没有支持 libaom-av1 的 ffmpeg（{e}）")
    if shutil.which("ffprobe") is None:
        pytest.skip("未验证：本机没有 ffprobe")
    root = official_root()
    if root is None:
        pytest.skip("未验证：找不到官方 RolloutRecorder（mme-vla 子模块未初始化）")
    monkeypatch.setenv(official_render.ENV_OFFICIAL_ROOT, str(root))
    fakes.EVENTS.clear()
    E.clear_builders()
    monkeypatch.setattr(E, "BUILDER_FACTORY", fakes.FakeBuilder)
    mod = load_script()
    seen = []
    real = P.load_policy

    def spy(*a, **k):
        p = real(*a, **k)
        seen.append(p)
        return p
    monkeypatch.setattr(P, "load_policy", spy)
    argv = ["--model", "dummy", "--dataset", "hard-verify,ood", "--tasks", "VideoUnmask", "--episodes", "0:1",
            "--seed", "0", "--out", str(tmp_path)]
    assert mod.main(argv) == 0
    (p,) = seen
    assert (p.calls["load"], p.calls["reset"], p.calls["close"]) == (1, 2, 1)
    out = capsys.readouterr().out
    assert "EVAL_DONE model=dummy" in out and "episodes_run=2" in out
    files_ok = 0
    for dataset, ep_label, tier in (("hard-verify", 3, "xhard0"), ("ood", 0, "xhard2")):
        rdir = tmp_path / "rollouts" / "dummy" / dataset / "seed0"
        raw = rdir / "raw" / f"VideoUnmask_ep{ep_label}_{tier}"
        for name in ("front.mkv", "wrist.mkv", "arrays.npz", "trace.jsonl", "result.json", "meta.json",
                     "events.jsonl", "frames-front.jsonl", "frames-wrist.jsonl"):
            assert (raw / name).is_file(), (dataset, name)
        assert probe(raw / "front.mkv") == ("av1", "yuv444p") and probe(raw / "wrist.mkv") == ("av1", "yuv444p")
        res = json.loads((raw / "result.json").read_text())
        assert res["status"] in ("success", "fail", "timeout") and res["recorder_verify"] == "PASS"
        assert res["video_error"] is None, res["video_error"]
        video = rdir / res["video"]
        assert video.name == f"VideoUnmask_ep{ep_label}_{res['status']}_{fakes.GOAL}_{tier}.mp4"
        assert probe(video) == ("av1", "yuv420p")
        log = json.loads((rdir / "log.json").read_text())
        assert log["episodes"] == 1 and "VideoUnmask" in log["tasks"] and log["total_success_rate"] is not None
        files_ok += 1
    assert files_ok == 2
    # 续跑：已有 result.json 的局全部跳过，Policy 照样只 load／close 一次
    seen.clear()
    assert mod.main(argv) == 0
    assert seen[0].calls["reset"] == 0 and seen[0].calls["load"] == 1 and seen[0].calls["close"] == 1
    assert "skipped=2" in capsys.readouterr().out


def test_official_render_reads_by_raw_codec(tmp_path, monkeypatch):
    """同一局原始产物：meta.json 记 av1-yuv444p 时按有损版读（跳过字节 sha）；去掉 raw_codec（视为旧 FFV1 产物）
    时走旧版核验，AV1 产物被拒为「有损降级」。"""
    try:
        recorder.find_ffmpeg()
    except RuntimeError as e:
        pytest.skip(f"未验证：本机没有支持 libaom-av1 的 ffmpeg（{e}）")
    if shutil.which("ffprobe") is None:
        pytest.skip("未验证：本机没有 ffprobe")
    E.clear_builders()
    monkeypatch.setattr(E, "BUILDER_FACTORY", fakes.FakeBuilder)
    with P.load_policy("dummy", 0) as p:
        res = E.run_episode(p, "ood", "VideoUnmask", 0, tmp_path, render=False)
    raw = tmp_path / "rollouts/dummy/ood/seed0" / res.raw_dir
    trace = official_render.load_trace(raw / "trace.jsonl")
    src = official_render.select_source(raw, "raw")
    assert src.kind == "raw-new" and src.raw_codec == "av1-yuv444p"
    front, wrist, detail = official_render.decode_raw_new(recorder.find_ffmpeg(), raw, src, trace)
    assert detail["lossy"] and front.shape == (trace.source_frames, fakes.H, fakes.W, 3) == wrist.shape
    meta = json.loads((raw / "meta.json").read_text())
    meta.pop("raw_codec")
    meta["codec"] = "ffv1"
    (raw / "meta.json").write_text(json.dumps(meta))
    legacy = official_render.select_source(raw, "raw")
    assert legacy.raw_codec == "ffv1"
    with pytest.raises(ValueError, match="lossy degradation"):
        official_render.decode_raw_new(recorder.find_ffmpeg(), raw, legacy, trace)


def test_web_size_matches_imageio_macro_block():
    assert official_render.web_size(512, 454) == (512, 464)
    assert official_render.web_size(512, 464) == (512, 464)
    assert "yuv420p" in official_render.WEB_ENCODE_ARGS and "+faststart" in official_render.WEB_ENCODE_ARGS
