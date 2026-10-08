"""官方原类逐位对照与真实编解码；覆盖存量 strict cap、精度恢复及坏轨迹拒绝。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tests._support.loaders import load_script


def _module():
    return load_script("eval-official/render_official_video.py")


def _official_root(repo_root):
    # worktree 子模块未初始化时只读引用主检出；测试不初始化或修改 gitlink。
    if (repo_root / "third_party/mme-vla/examples/robomme/utils.py").is_file():
        return repo_root
    return Path(os.environ["SGEVAL_THIRD_PARTY"]).parent


def _array(a):
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "f32hex": a.astype("<f4").tobytes().hex()}


def _rows(*, demo=2, steps=3, max_steps=10, terminal="success", status=None, task="VideoUnmask", action_dtype="<f4"):
    identity = {"task": task, "tier": "xhard0", "seed": 123, "dataset": "ood", "source_episode": 3,
                "builder_episode": 0, "key": f"{task}_xhard0_123"}
    rows = [{"kind": "header", "schema": "sgeval-trace/1", "route": "groundsg/ground-sg-oracle/new",
             "identity": identity, "max_steps": max_steps},
            {"kind": "demo", "frames": demo + 1, "front_sha256": ["front"] * (demo + 1),
             "wrist_sha256": ["wrist"] * (demo + 1), "states": [_array(np.arange(8, dtype="<f4") / 9)] * (demo + 1),
             "texts": ["press button"]}]
    for step in range(1, steps + 1):
        rows.append({"kind": "step", "step": step, "state": _array(np.arange(8, dtype="<f4") / 7),
                     "action": _array(np.arange(8, dtype=action_dtype) / 11), "front_sha256": "front",
                     "wrist_sha256": "wrist", "subgoal": f"move to <3, 5> {step}", "terminated": False,
                     "truncated": False, "status": "unknown"})
    rows.append({"kind": "end", "demo_frames": demo, "exec_steps": steps, "terminal_reason": terminal,
                 "status": status or terminal})
    return rows


def _trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return _module().load_trace(path)


@pytest.mark.parametrize("task", ["VideoUnmask", "BinFill"])
def test_feed_matches_official_recorder_bitwise(tmp_path, repo_root, task):
    mod = _module()
    trace = _trace(tmp_path, _rows(task=task))
    Recorder, demo_tasks = mod.load_official(_official_root(repo_root))
    left, right = Recorder(tmp_path / "left", trace.task_goal), Recorder(tmp_path / "right", trace.task_goal)
    frames = np.random.default_rng(42).integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
    wrists = np.random.default_rng(43).integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
    # 独立对照按官方 init_episode 的 3 个输入与执行循环的 3 个输入直接调原类。
    for i in range(3):
        left.record(frames[i].copy(), wrists[i].copy(), trace.init_states[i].copy(),
                    is_video_demo=task in demo_tasks and i < 2, subgoal="[initializing...]")
    for i in range(3):
        st = trace.steps[i]
        left.record(frames[i + 3].copy(), wrists[i + 3].copy(), st["state"].copy(),
                    action=st["action"].copy(), subgoal=st["subgoal"])
    mod.feed_official_recorder(right, frames, wrists, trace, task, demo_tasks)
    assert len(left.total_images) == len(right.total_images) == 6
    assert all(np.array_equal(a, b) for a, b in zip(left.total_images, right.total_images))
    print("RENDER_PARITY=PASS frames=6 mismatch=0")


def test_f32hex_roundtrip_and_shape_guard():
    mod = _module()
    a = np.linspace(-1, 1, 8, dtype="<f4")
    assert np.array_equal(mod.f32_from_record(_array(a)), a)
    for bad in ({**_array(a), "shape": [1, 8]}, {**_array(a), "f32hex": "00"},
                {**_array(a), "sha256": "0" * 64}, _array(np.full(8, np.nan, dtype="<f4"))):
        with pytest.raises(ValueError):
            mod.f32_from_record(bad)


def test_restore_float64_action_with_verified_npz(tmp_path):
    mod = _module()
    rows = _rows(demo=0, steps=1, action_dtype="<f8")
    action = np.array([0, 0.8404500172406898, 0, 0, 0, 0, 0, 0], dtype="<f8")
    rows[2]["action"] = _array(action)
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="丢失原数组精度"):
        mod.load_trace(path)
    np.savez(tmp_path / "arrays.npz", exec_action__00000=action)
    trace = mod.load_trace(path)
    assert trace.steps[0]["action"].dtype == np.dtype("<f8")
    assert np.array_equal(trace.steps[0]["action"], action)
    assert f"{trace.steps[0]['action'][1]:.4f}" == "0.8405"
    assert f"{mod.f32_from_record(_array(action))[1]:.4f}" == "0.8404"
    np.savez(tmp_path / "arrays.npz", exec_action__00000=action + 1)
    with pytest.raises(ValueError, match="sha256"):
        mod.load_trace(path)


@pytest.mark.parametrize("terminal,status,steps,omitted", [("timeout", "timeout", 3, 1),
                                                           ("error", "timeout", 2, 0),
                                                           ("timeout", "timeout", 2, 0)])
def test_timeout_only_omits_official_unrecorded_overrun(tmp_path, repo_root, terminal, status, steps, omitted):
    mod = _module()
    trace = _trace(tmp_path, _rows(demo=0, steps=steps, max_steps=2, terminal=terminal, status=status))
    Recorder, demo_tasks = mod.load_official(_official_root(repo_root))
    rec = Recorder(tmp_path / "out", trace.task_goal)
    images = np.zeros((steps + 1, 8, 8, 3), np.uint8)
    mod.feed_official_recorder(rec, images, images, trace, "VideoUnmask", demo_tasks)
    assert trace.omitted_timeout_frames == omitted
    assert len(rec.total_images) == steps + 1 - omitted
    assert trace.terminal_reason == terminal and trace.end_status == status


def test_no_demo_no_subgoal_and_wrong_frame_count(tmp_path, repo_root):
    mod = _module()
    rows = _rows(demo=0, steps=1, task="BinFill")
    rows[0]["route"] = "perceptual-framesamp-modul/new"
    rows[2]["subgoal"] = None
    trace = _trace(tmp_path, rows)
    assert trace.init_subgoal is None and trace.demo_frames == 0 and trace.source_frames == 2
    Recorder, demo_tasks = mod.load_official(_official_root(repo_root))
    rec = Recorder(tmp_path / "out", trace.task_goal)
    images = np.zeros((2, 8, 8, 3), np.uint8)
    mod.feed_official_recorder(rec, images, images, trace, "BinFill", demo_tasks)
    assert len(rec.total_images) == 2
    with pytest.raises(ValueError, match="frame_count"):
        mod.feed_official_recorder(rec, images[:1], images, trace, "BinFill", demo_tasks)


@pytest.mark.parametrize("defect", ["double_init", "reordered", "duplicate_end", "missing_state", "end_count", "bad_terminal"])
def test_malformed_trace_rejected(tmp_path, defect):
    rows = _rows()
    if defect == "double_init":
        rows[1]["frames"] += 1
    elif defect == "reordered":
        rows[2], rows[3] = rows[3], rows[2]
    elif defect == "duplicate_end":
        rows.append(rows[-1])
    elif defect == "missing_state":
        rows[1]["states"] = []
    elif defect == "end_count":
        rows[-1]["exec_steps"] -= 1
    else:
        rows[-1]["status"] = "success"
        rows[-1]["terminal_reason"] = "error"
    with pytest.raises(ValueError):
        _trace(tmp_path, rows)


def test_safe_filename_preserves_full_semantics_with_utf8_limit():
    mod = _module()
    short = "official-rerender__VideoUnmask_ep3a1_success_goal_xhard0.mp4"
    assert mod.safe_filename(short) == short
    a, b = "official-rerender__" + "目标" * 110 + "_xhard1.mp4", "official-rerender__" + "目标" * 110 + "_xhard2.mp4"
    assert len(mod.safe_filename(a).encode()) <= 255
    assert mod.safe_filename(a) != mod.safe_filename(b)
    assert "/" not in mod.safe_filename("goal/with/slash.mp4")


@pytest.mark.slow
def test_end_to_end_with_real_ffmpeg(tmp_path, repo_root, capsys):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("未验证：缺少 ffmpeg/ffprobe")
    mod = _module()
    ep = tmp_path / "VideoUnmask_xhard0_123.a1"
    ep.mkdir()
    rows = _rows(demo=1, steps=2)
    (ep / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=512x256:rate=30",
                    "-frames:v", "4", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(ep / "episode.mp4")], check=True)
    source_before = mod.fingerprint(ep / "episode.mp4")
    official = _official_root(repo_root)
    assert mod.main([str(ep), "--official-root", str(official)]) == 0
    assert "OFFICIAL_RENDER_SUMMARY=PASS total=1 ok=1 fail=0" in capsys.readouterr().out
    result = json.loads((ep / "official/render.json").read_text())
    assert result["safe_name"] == "official-rerender__VideoUnmask_ep3a1_success_press button_xhard0.mp4"
    assert result["frames"] == 4 and result["fps"] == "30"
    assert result["width"] == 512 and result["height"] == 512
    assert result["render_status"] == "rendered" and result["status"] == "success"
    assert mod.render_episode(ep, official_root=official)["render_status"] == "reused"
    result["source_trace"]["sha256"] = "0" * 64
    (ep / "official/render.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="拒绝覆盖"):
        mod.render_episode(ep, official_root=official)
    replaced = mod.render_episode(ep, official_root=official, overwrite=True)
    assert replaced["render_status"] == "rendered"
    assert mod.fingerprint(ep / "episode.mp4") == source_before


# ── 第二阶段 S2a：无损原始帧来源、逐帧哈希核验、C3／C4／C8 ─────────────────────────────

from tests.pipeline.evalx.report import sgx_render_fixtures as fx  # noqa: E402
from tests.pipeline.evalx.report import trace_contract as tc  # noqa: E402


def _need_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("未验证：缺少 ffmpeg/ffprobe")


def _render(ep, repo_root, **kw):
    return _module().render_episode(ep, official_root=_official_root(repo_root), ffmpeg=shutil.which("ffmpeg"), **kw)


def _probe_frames(path):
    return _module().probe_video(shutil.which("ffprobe"), path)["frames"]


def _index_rows(ep, stream="front"):
    return [json.loads(x) for x in (ep / f"frames-{stream}.jsonl").read_text().splitlines() if x.strip()]


def _write_index(ep, rows, stream="front"):
    (ep / f"frames-{stream}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


@pytest.mark.slow
@pytest.mark.parametrize("raw", ["new", "orig"])
def test_lossless_raw_without_mp4_renders(tmp_path, repo_root, capsys, raw):
    _need_ffmpeg()
    ep = fx.write_episode(tmp_path, "normal", raw=raw)
    assert not (ep.dir / "episode.mp4").exists()
    tc.assert_renderable(ep.dir)
    mod = _module()
    assert mod.main([str(ep.dir), "--official-root", str(_official_root(repo_root)), "--source", "raw"]) == 0
    out = capsys.readouterr().out
    kind = f"raw-{raw}"
    assert f"source_kind={kind}" in out and "OFFICIAL_RENDER_SUMMARY=PASS total=1 ok=1 fail=0" in out
    side = json.loads((ep.dir / "official/render.json").read_text())
    assert side["source_kind"] == kind and side["source_mp4"] is None
    assert side["frame_hash_check"]["mode"] == "verified"
    assert side["frame_hash_check"]["frames"] == {"front": ep.frames_recorded, "wrist": ep.frames_recorded}
    # 流与索引以局目录相对路径登记 sha256
    streams = sorted(side["source_media"]["streams"])
    assert streams == (["front.mkv", "wrist.mkv"] if raw == "new" else ["frames/front.rgb24", "frames/wrist.rgb24"])
    assert all(not Path(p).is_absolute() for p in side["source_media"]["index"])
    assert side["frames"] == side["frames_recorded"] == ep.frames_recorded
    assert _probe_frames(ep.dir / side["out_rel"]) == ep.frames_recorded
    # auto 在有原始流时也选 raw，且结果可复用
    assert _render(ep.dir, repo_root)["render_status"] == "reused"


@pytest.mark.slow
@pytest.mark.parametrize("defect", ["swap_streams", "wrong_enc", "missing_frame", "duplicate_idx", "lossy"])
def test_raw_new_defects_fail_without_fallback(tmp_path, repo_root, defect):
    _need_ffmpeg()
    ep = fx.write_episode(tmp_path, "normal").dir
    # 放一个合法 mp4 作诱饵：raw 失败也不得退回 mp4
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=512x256:rate=30",
                    "-frames:v", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(ep / "episode.mp4")], check=True)
    rows = _index_rows(ep)
    if defect == "swap_streams":
        (ep / "front.mkv").rename(ep / "tmp.mkv")
        (ep / "wrist.mkv").rename(ep / "front.mkv")
        (ep / "tmp.mkv").rename(ep / "wrist.mkv")
        match = "sha256 不符"
    elif defect == "wrong_enc":
        # 末两行 enc 互换：enc 集合仍完整覆盖解码帧，只能靠逐行 sha256 发现
        rows[-1]["enc"], rows[-2]["enc"] = rows[-2]["enc"], rows[-1]["enc"]
        _write_index(ep, rows)
        match = "sha256 不符"
    elif defect == "missing_frame":
        _write_index(ep, rows[:-1])
        match = "缺帧|索引损坏"
    elif defect == "duplicate_idx":
        _write_index(ep, rows + [rows[-1]])
        match = "重复索引"
    else:
        meta = json.loads((ep / "meta.json").read_text())
        meta.update(level=1, codec="libx264-crf18")
        (ep / "meta.json").write_text(json.dumps(meta))
        match = "有损降级"
    for source in ("raw", "auto"):
        with pytest.raises(ValueError, match=match):
            _render(ep, repo_root, source=source)
    assert not (ep / "official").exists() or not list((ep / "official").glob("*.mp4"))


@pytest.mark.slow
def test_raw_orig_missing_frame_and_raw_mode_needs_raw(tmp_path, repo_root):
    _need_ffmpeg()
    ep = fx.write_episode(tmp_path, "obs_none", raw="orig").dir
    # 原侧没有逐帧索引哈希：调包两路流只能靠 trace 画面哈希发现
    (ep / "frames/front.rgb24").rename(ep / "frames/tmp.rgb24")
    (ep / "frames/wrist.rgb24").rename(ep / "frames/front.rgb24")
    (ep / "frames/tmp.rgb24").rename(ep / "frames/wrist.rgb24")
    with pytest.raises(ValueError, match="逐帧哈希不符"):
        _render(ep, repo_root, source="raw")
    f = ep / "frames/front.rgb24"
    f.write_bytes(f.read_bytes()[:-fx.H * fx.W * 3])
    with pytest.raises(ValueError, match="缺帧"):
        _render(ep, repo_root, source="raw")
    only_mp4 = tmp_path / "mp4only"
    only_mp4.mkdir()
    ep2 = fx.write_episode(only_mp4, "normal", raw="orig").dir
    for p in ("frames/front.rgb24", "frames/wrist.rgb24"):
        (ep2 / p).unlink()
    (ep2 / "episode.mp4").write_bytes(b"")
    with pytest.raises(ValueError, match="raw 模式缺原始帧"):
        _render(ep2, repo_root, source="raw")


@pytest.mark.slow
def test_reuse_after_raw_deleted_but_not_after_raw_changed(tmp_path, repo_root):
    _need_ffmpeg()
    ep = fx.write_episode(tmp_path, "normal").dir
    first = _render(ep, repo_root, source="raw")
    assert first["render_status"] == "rendered"
    # 改 raw 后复用必须失败（流指纹不符），不加 --overwrite 拒绝覆盖
    keep = (ep / "front.mkv").read_bytes()
    (ep / "front.mkv").write_bytes(keep + b"\0")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        _render(ep, repo_root, source="raw")
    (ep / "front.mkv").write_bytes(keep)
    # 改索引同样不可复用
    idx = (ep / "frames-wrist.jsonl").read_text()
    (ep / "frames-wrist.jsonl").write_text(idx + "\n")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        _render(ep, repo_root)
    (ep / "frames-wrist.jsonl").write_text(idx)
    # 转码后删 raw（frames-*.jsonl 保留）：重入走复用；整个目录搬走后仍可核（相对路径）
    for p in ("front.mkv", "wrist.mkv"):
        (ep / p).unlink()
    moved = tmp_path / "nfs" / ep.name
    moved.parent.mkdir()
    ep.rename(moved)
    for source in ("raw", "auto"):
        again = _render(moved, repo_root, source=source)
        assert again["render_status"] == "reused" and again["output_fingerprint"] == first["output_fingerprint"]
    with pytest.raises(ValueError, match="拒绝覆盖"):
        _render(moved, repo_root, source="mp4")


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["exception", "obs_none", "natural_timeout", "strict_cap"])
def test_episode_kinds_result_trace_media_counts_agree(tmp_path, repo_root, kind):
    _need_ffmpeg()
    ep = fx.write_episode(tmp_path, kind)
    tc.assert_renderable(ep.dir)
    tc.assert_counts_consistent(ep.dir, ep.result_row)
    trace = _module().load_trace(ep.dir / "trace.jsonl")
    assert len(trace.steps) == ep.result_row["exec_steps"] and len(trace.observed_steps) == ep.steps_observed
    assert trace.output_frames == ep.frames_recorded
    result = _render(ep.dir, repo_root, source="raw")
    end = json.loads((ep.dir / "trace.jsonl").read_text().splitlines()[-1])
    assert result["status"] == ep.result_row["status"] == end["status"]
    assert result["exec_steps"] == result["steps_attempted"] == end["steps_attempted"] == ep.result_row["exec_steps"]
    assert result["steps_observed"] == end["steps_observed"]
    assert result["frames"] == end["frames_recorded"] == _probe_frames(Path(result["out"]))
    index_front = len(_index_rows(ep.dir))
    assert index_front == end["demo_frames"] + 1 + end["steps_observed"]  # 原始帧不含缺观测步
    assert result["omitted_timeout_frames"] == int(kind == "natural_timeout")
    assert result["missing_steps"] == list(fx.KINDS[kind]["missing"])


@pytest.mark.slow
def test_no_frame_error_episode_records_reason(tmp_path, repo_root, capsys):
    _need_ffmpeg()
    ep = fx.write_no_frame(tmp_path)
    tc.assert_renderable(ep)
    mod = _module()
    trace = mod.load_trace(ep / "trace.jsonl")
    assert trace.no_frame and trace.end_status == "error" and trace.output_frames == 0
    assert mod.main([str(ep), "--official-root", str(_official_root(repo_root))]) == 0
    out = capsys.readouterr().out
    assert "OFFICIAL_RENDER=NO_FRAME" in out and "no_frame=1" in out and "fail=0" in out
    side = json.loads((ep / "official/render.json").read_text())
    assert side["render_status"] == "no_frame" and "reset_exception" in side["no_frame_reason"]
    assert not list((ep / "official").glob("*.mp4"))
    again = _render(ep, repo_root)
    assert again["render_status"] == "no_frame" and again["reused"] is True


def test_error_status_allowed_and_count_mismatch_rejected(tmp_path):
    mod = _module()
    trace = _trace(tmp_path, _rows(terminal="error", status="error", steps=2))
    assert trace.end_status == trace.terminal_reason == "error" and trace.output_frames == 5
    rows = _rows(steps=2)
    rows[-1].update(steps_attempted=2, steps_observed=2, frames_recorded=99)
    with pytest.raises(ValueError, match="frames_recorded"):
        _trace(tmp_path, rows)
    rows = _rows(steps=2)
    rows[-1]["no_frame"] = True
    with pytest.raises(ValueError, match="no_frame"):
        _trace(tmp_path, rows)
    assert mod.TERMINALS == ("success", "fail", "timeout", "error")


@pytest.mark.slow
def test_key_option_for_dir_without_key(tmp_path, repo_root):
    _need_ffmpeg()
    ep = fx.write_episode(tmp_path, "normal", dir_name="ep000").dir
    with pytest.raises(ValueError, match="局目录与 trace.identity.key 不符"):
        _render(ep, repo_root)
    with pytest.raises(ValueError, match="--key"):
        _render(ep, repo_root, key="Other_xhard0_1")
    key = json.loads((ep / "trace.jsonl").read_text().splitlines()[0])["identity"]["key"]
    mod = _module()
    assert mod.main([str(ep), "--official-root", str(_official_root(repo_root)), "--key", key]) == 0
    side = json.loads((ep / "official/render.json").read_text())
    assert side["episode_tag"] == f"{key}.a1" and side["episode_id"] == "3a1"
    with pytest.raises(SystemExit):
        mod.main([str(ep), str(ep.parent), "--key", key])


# ── 1006 第三阶段：strict-cap 局命名 timeout、render.json 记 policy_seed ─────────────────────────


def test_strict_cap_named_timeout_and_policy_seed_parsed(tmp_path):
    mod = _module()
    rows = _rows(demo=0, steps=2, max_steps=2, terminal="error", status="timeout")  # 旧口径：terminal_reason=error
    rows[0]["policy_seed"] = 7
    rows[0]["identity"]["policy_seed"] = 7
    rows[-1]["cap_hit"] = True
    trace = _trace(tmp_path, rows)
    assert trace.terminal_reason == "error" and trace.named_terminal == "timeout"
    assert trace.policy_seed == 7 and trace.cap_hit is True
    assert _trace(tmp_path, _rows(steps=2)).policy_seed is None  # 旧轨迹没有种子：记 None，不补
    bad = _rows(steps=2)
    bad[-1]["cap_hit"] = True  # success 局不可能 strict-cap 命中
    with pytest.raises(ValueError, match="cap_hit"):
        _trace(tmp_path, bad)
    bad = _rows(steps=2)
    bad[0]["policy_seed"], bad[0]["identity"]["policy_seed"] = 7, 42
    with pytest.raises(ValueError, match="policy_seed"):
        _trace(tmp_path, bad)
    bad = _rows(steps=2)
    bad[0]["policy_seed"] = -1
    with pytest.raises(ValueError, match="policy_seed"):
        _trace(tmp_path, bad)


@pytest.mark.slow
def test_strict_cap_render_file_named_timeout(tmp_path, repo_root):
    _need_ffmpeg()
    ep = fx.write_episode(tmp_path, "strict_cap")
    path = ep.dir / "trace.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["policy_seed"] = 7
    rows[-1]["terminal_reason"] = "error"  # 旧口径的 strict-cap 超时局
    rows[-1]["cap_hit"] = True
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = _render(ep.dir, repo_root, source="raw")
    assert "_timeout_" in result["safe_name"] and "_error_" not in result["safe_name"]
    side = json.loads((ep.dir / "official/render.json").read_text())
    assert side["terminal_reason"] == "timeout" and side["trace_terminal_reason"] == "error"
    assert side["policy_seed"] == 7 and side["cap_hit"] is True
