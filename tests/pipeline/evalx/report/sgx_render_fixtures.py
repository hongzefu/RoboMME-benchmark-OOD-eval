"""S2a 重绘工具的无损原始帧夹具（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节 S2a）。

局目录按生产格式写出：轨迹用真实 ``trace_writer.TraceWriter``，新侧原始帧用真实 ``recorder.EpisodeRecorder``
（拆仓后为 AV1 4:4:4 有损 + ``frames-<stream>.jsonl``，``meta.json`` 记 ``raw_codec``；调用顺序照 ``EnvSession``：reset
帧 tag ``reset``、第 k 步 tag ``step{k-1}``、每步先 ``add_array("exec_action")`` 再记画面，异常步不记画面）；
``raw="legacy"`` 按旧录制器的产物格式手写 FFV1 无损流（``meta.json`` 无 ``raw_codec``，同一流重复帧只编码一份，
索引 ``enc`` 指向首次出现的编码帧），供「旧产物照旧逐帧核验」的用例使用；原侧原始帧照
``official_hard_runner``／``pp_official_runner`` 文档串写 ``frames/{front,wrist}.rgb24`` + ``frames.json``。
期望计数由用例参数手算，不读被测重绘工具。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tests._support.loaders import load_script

H = W = 256

# 四种局（计划 S2a 测试清单）+ 正常局；missing 为缺观测步号（异常步／无观测返回），cap 为 strict-cap 步数
KINDS = {
    "normal": dict(status="success", steps=3, max_steps=10, missing=(), reason=None),
    "exception": dict(status="error", steps=3, max_steps=10, missing=(3,), reason="env_exception"),
    "obs_none": dict(status="fail", steps=4, max_steps=10, missing=(2,), reason="obs_none"),
    "natural_timeout": dict(status="timeout", steps=4, max_steps=3, missing=(), reason=None),
    "strict_cap": dict(status="timeout", steps=3, max_steps=10, missing=(), reason=None),
}


@dataclass
class Episode:
    dir: Path
    result_row: dict
    frames_recorded: int
    steps_observed: int


def _tw():
    return load_script("eval-official/trace_writer.py")


def _rec_mod():
    return load_script("eval-official/recorder.py")


def _img(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def _state(k: int) -> np.ndarray:
    return (np.arange(8, dtype=np.float32) + np.float32(k)) / np.float32(9)


def _action(k: int, dtype) -> np.ndarray:
    return ((np.arange(8) + k) / 11).astype(dtype)


def _write_legacy_ffv1(ep: Path, streams: dict[str, list[tuple[np.ndarray, str]]]) -> None:
    """旧录制器产物：每流一个 FFV1 MKV（只编码首次出现的画面）+ ``frames-<stream>.jsonl``（idx、sha256、tag、enc）
    + 无 ``raw_codec`` 字段的 ``meta.json``（level 0、codec ffv1）。"""
    ff = shutil.which("ffmpeg")
    ep.mkdir(parents=True, exist_ok=True)
    for name, items in streams.items():
        first: dict[str, int] = {}
        uniq: list[np.ndarray] = []
        rows = []
        for idx, (img, tag) in enumerate(items):
            sha = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
            if sha not in first:
                first[sha] = len(uniq)
                uniq.append(img)
            rows.append({"idx": idx, "sha256": sha, "tag": tag, "enc": first[sha]})
        subprocess.run([ff, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", "30", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(ep / f"{name}.mkv")],
                       input=b"".join(np.ascontiguousarray(x).tobytes() for x in uniq), check=True)
        (ep / f"frames-{name}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (ep / "meta.json").write_text(json.dumps({"never_degrade": True, "level": 0, "codec": "ffv1"}) + "\n")


def write_episode(root: Path, kind: str = "normal", *, raw: str = "new", demo: int = 2, task: str = "BinFill",
                  route: str = "groundsg/ground-sg-oracle/new", dir_name: str | None = None,
                  action_dtype=np.float64) -> Episode:
    """写一局：``raw`` 取 ``new``（recorder.py 的 AV1 流）、``legacy``（旧 FFV1 无损流）或 ``orig``（rgb24 + frames.json）；
    不写 episode.mp4。"""
    spec = KINDS[kind]
    key = f"{task}_xhard0_{100 + len(kind)}"
    ep = Path(root) / (dir_name or f"{key}.a1")
    identity = {"task": task, "tier": "xhard0", "seed": 100 + len(kind), "dataset": "hard-verify", "source_episode": 3,
                "builder_episode": 0, "key": key, "attempt": 1}
    # 演示段前两帧相同，验证同一流重复帧只编码一份、按 enc 展开
    fronts = [_img(1)] + [_img(1)] + [_img(10 + i) for i in range(demo - 1)]
    wrists = [_img(2)] + [_img(2)] + [_img(20 + i) for i in range(demo - 1)]
    fronts, wrists = fronts[:demo + 1], wrists[:demo + 1]
    rec = None
    if raw == "new":
        rec = _rec_mod().EpisodeRecorder(ep, {"never_degrade": True, "task": task}, free_gib_fn=lambda _p: 1e6)
        rec.set_phase("reset")
        rec.add_frames("front", np.stack(fronts), tag="reset")
        rec.add_frames("wrist", np.stack(wrists), tag="reset")
        rec.set_phase("run")
    orig_front, orig_wrist = list(fronts), list(wrists)
    legacy = {"front": [(f, "reset") for f in fronts], "wrist": [(x, "reset") for x in wrists]}
    w = _tw().TraceWriter(ep / "trace.jsonl", route=route, identity=identity, max_steps=spec["max_steps"])
    w.log_demo(fronts, wrists, [_state(-k) for k in range(demo + 1)], ["put the cube in the bin"])
    actions = {}
    for s in range(1, spec["steps"] + 1):
        a = _action(s, action_dtype)
        actions[f"exec_action__{s - 1:05d}"] = a
        if rec is not None:
            rec.add_array("exec_action", a, step=s - 1)
        subgoal = None if s == 1 else f"pick up the cube at <{s}, {s + 1}>"
        if s in spec["missing"]:
            w.log_missing_step(step=s, action=a, reason=spec["reason"], subgoal=subgoal)
            continue
        f, wr = _img(100 + s), _img(200 + s)
        if rec is not None:
            rec.add_frames("front", f, tag=f"step{s - 1}")
            rec.add_frames("wrist", wr, tag=f"step{s - 1}")
        orig_front.append(f)
        orig_wrist.append(wr)
        legacy["front"].append((f, f"step{s - 1}"))
        legacy["wrist"].append((wr, f"step{s - 1}"))
        w.log_step(step=s, front=f, wrist=wr, state=_state(s), action=a, subgoal=subgoal,
                   terminated=s == spec["steps"] and spec["status"] in ("success", "fail"), truncated=False,
                   status=spec["status"] if s == spec["steps"] else "unknown")
    observed = spec["steps"] - len(spec["missing"])
    omitted = int(kind == "natural_timeout")
    frames_recorded = demo + 1 + observed - omitted
    w.close(status=spec["status"], terminal_reason=spec["status"], demo_frames=demo, steps_attempted=spec["steps"],
            steps_observed=observed, frames_recorded=frames_recorded, omitted_timeout_frames=omitted)
    if rec is not None:
        res = rec.close({"status": spec["status"]})
        assert res["RECORDER_VERIFY"] == "PASS", res
    elif raw == "legacy":
        np.savez(ep / "arrays.npz", **actions)
        _write_legacy_ffv1(ep, legacy)
    else:
        np.savez(ep / "arrays.npz", **actions)
        fd = ep / "frames"
        fd.mkdir()
        for s, frames in (("front", orig_front), ("wrist", orig_wrist)):
            (fd / f"{s}.rgb24").write_bytes(b"".join(np.ascontiguousarray(x).tobytes() for x in frames))
        meta = {"pix_fmt": "rgb24", "order": "reset_all_then_per_step_last", "demo_frames": demo, "init_frames": 1,
                "exec_steps": spec["steps"], "missing_steps": list(spec["missing"]),
                "streams": {s: {"width": W, "height": H, "count": len(orig_front)} for s in ("front", "wrist")}}
        (fd / "frames.json").write_text(json.dumps(meta, sort_keys=True) + "\n")
    return Episode(ep, {"exec_steps": spec["steps"], "status": spec["status"]}, frames_recorded, observed)


def write_no_frame(root: Path) -> Path:
    """C3：reset 前就失败的 error 局，没有任何画面。"""
    key = "BinFill_xhard0_900"
    ep = Path(root) / f"{key}.a1"
    identity = {"task": "BinFill", "tier": "xhard0", "seed": 900, "dataset": "hard-verify", "source_episode": 3,
                "builder_episode": 0, "key": key, "attempt": 1}
    w = _tw().TraceWriter(ep / "trace.jsonl", route="smvla/new", identity=identity, max_steps=10)
    w.log_demo([], [])
    w.close(status="error", terminal_reason="error", demo_frames=0, no_frame=True, steps_attempted=0,
            steps_observed=0, frames_recorded=0, error="reset_exception: RuntimeError")
    return ep
