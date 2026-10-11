"""原始录像搬运真实字节、崩溃窗口与错误封闭测试；不触发环境。"""
import hashlib
import importlib.util
import json
import subprocess
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("raw_mover_test", ROOT / "dev-scripts/gl/eval_video_mover.py")
MOVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOVER)
CONTEXT = {"run": "fixture", "run_name": "fixture", "exec_commit": "a" * 40,
           "config_sha256": "b" * 64, "controller_job_id": "123"}


@pytest.fixture
def publication(tmp_path):
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    episode = stage / "rollouts/smvla/ood/seed7/raw/VideoUnmask_ep0_xhard1"
    dest.mkdir()
    # 真实生产器生成非重复帧、原数组和序列化轨迹，全程不创建环境。
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / "src/robomme_ood_eval/record" / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    recorder_module = load("fixture_recorder", "recorder.py")
    trace_module = load("fixture_trace", "trace_writer.py")
    recorder = recorder_module.EpisodeRecorder(episode, {}, encode_async=False, free_gib_fn=lambda _: 1000)
    action = np.zeros(8, dtype=np.float32)
    state = np.arange(5, dtype=np.float64)
    identity = {"dataset": "ood", "task": "VideoUnmask", "tier": "xhard1", "seed": 0, "key": "key"}
    trace = trace_module.TraceWriter(episode / "trace.jsonl", route="fixture", identity=identity, max_steps=1)
    reset_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    step_frame = np.full((16, 16, 3), 170, dtype=np.uint8)
    trace.log_demo([reset_frame], [reset_frame], states=[state])
    trace.log_step(step=1, front=step_frame, wrist=step_frame, state=state, action=action,
                   subgoal=None, terminated=False, truncated=False, status=None)
    trace.close(status="done")
    for stream in ("front", "wrist"):
        recorder.add_frames(stream, reset_frame, tag="reset")
        recorder.add_frames(stream, step_frame, tag="step0")
    recorder.add_array("reset_state", state, step=0)
    recorder.add_array("exec_action", action, step=0)
    recorder.add_array("exec_state", state, step=0)
    recorder.add_event({"kind": "env_reset", "frames": 1, "front": [hashlib.sha256(reset_frame.tobytes()).hexdigest()], "wrist": [hashlib.sha256(reset_frame.tobytes()).hexdigest()]})
    recorder.add_event({"kind": "env_step", "step": 0, "front": [hashlib.sha256(step_frame.tobytes()).hexdigest()], "wrist": [hashlib.sha256(step_frame.tobytes()).hexdigest()]})
    recorder.close({})
    MOVER.raw_atomic(episode / "result.json", {**identity, "task_success": False, "exec_steps": 1, "error_kind": None, "infra": False, "recorder_verify": "PASS", "demo_frames": 0, "model": "smvla", "policy_label": "smvla", "policy_seed": 7, "episode": 0})
    row = {"schema": MOVER.RAW_SCHEMA, **CONTEXT, "model": "smvla", "shard_id": "0", "identity_key": "smvla:ood:key", "relative_dir": str(episode.relative_to(stage)), "files": MOVER.raw_tree(episode), "delete_files": ["front.mkv", "wrist.mkv", "arrays.npz"]}
    return stage, dest, episode, row


def test_real_move_retains_small_files_and_recovers_delete_window(publication):
    stage, dest, source, row = publication
    MOVER.raw_validate_publication(row, CONTEXT)
    receipt = MOVER.raw_move(row, stage, dest, min_free_gib=0)
    assert receipt["decoded"] == {"front": 2, "wrist": 2}
    assert MOVER.raw_tree(dest / row["relative_dir"]) == row["files"]
    assert set(MOVER.raw_tree(source)) == set(row["files"]) - set(row["delete_files"])
    assert MOVER.raw_move(row, stage, dest, min_free_gib=0)["result"] == "moved"


def test_hash_failure_preserves_source(publication):
    stage, dest, source, row = publication
    (source / "front.mkv").write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA"):
        MOVER.raw_move(row, stage, dest, min_free_gib=0)
    assert (source / "arrays.npz").exists()
    assert not (dest / row["relative_dir"]).exists()


def test_name_conflict_and_low_disk_preserve_source(publication, monkeypatch):
    stage, dest, source, row = publication
    monkeypatch.setattr(MOVER, "free_gib", lambda _: 0)
    with pytest.raises(ValueError, match="low_disk"):
        MOVER.raw_move(row, stage, dest)
    monkeypatch.setattr(MOVER, "free_gib", lambda _: 1000)
    target = dest / row["relative_dir"]
    target.mkdir(parents=True)
    (target / "other").write_text("原有用户文件")
    with pytest.raises(ValueError, match="冲突"):
        MOVER.raw_move(row, stage, dest)
    assert (source / "front.mkv").exists()
    assert (target / "other").read_text() == "原有用户文件"


def test_incomplete_decode_and_action_corruption(publication):
    _, _, source, _ = publication
    (source / "front.mkv").write_bytes(b"bad")
    with pytest.raises(ValueError, match="解码"):
        MOVER.raw_verify_media(source)
    (source / "front.mkv").write_bytes((source / "wrist.mkv").read_bytes())
    np.savez(source / "arrays.npz", exec_action__00000=np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="数组"):
        MOVER.raw_verify_media(source)


def test_missing_key_delete_small_and_symlink_rejected(publication):
    stage, dest, source, row = publication
    missing = dict(row)
    del missing["files"]
    with pytest.raises(ValueError, match="缺键"):
        MOVER.raw_validate_publication(missing, CONTEXT)
    with pytest.raises(ValueError, match="小文件"):
        MOVER.raw_validate_publication({**row, "delete_files": ["result.json"]}, CONTEXT)
    (source / "link").symlink_to(source / "result.json")
    with pytest.raises(ValueError, match="文件类型"):
        MOVER.raw_move(row, stage, dest, min_free_gib=0)


def test_serialized_completion_and_structured_error(publication):
    stage, dest, _, row = publication
    name = hashlib.sha256(row["identity_key"].encode()).hexdigest() + ".json"
    manifest = stage / "config.json"
    MOVER.raw_atomic(manifest, {**{k: v for k, v in CONTEXT.items() if k != "config_sha256"}, "expected_publications": [name], "expected_identities": [row["identity_key"]]})
    row["config_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    MOVER.raw_atomic(stage / "published" / name, row)
    args = SimpleNamespace(stage=str(stage), dest=str(dest), manifest=str(manifest), min_free_gib=0,
                           heartbeat_file=str(stage / "control/heartbeat.json"), error_file=str(stage / "control/error.json"), stop_file=str(stage / "control/STOP"), once=True, interval=.01)
    assert MOVER.raw_main(args) == 0
    completed = json.loads((stage / "mover/completed.json").read_text())
    assert {k: completed[k] for k in ("missing", "decode_fail", "sha_mismatch", "pending")} == {"missing": 0, "decode_fail": 0, "sha_mismatch": 0, "pending": 0}
    receipt = stage / "mover/receipts" / name
    bad = json.loads(receipt.read_text())
    del bad["trace_checked"]
    MOVER.raw_atomic(receipt, bad)
    assert MOVER.raw_main(args) == 1
    assert json.loads((stage / "control/error.json").read_text())["component"] == "mover"
    assert not (stage / "control/STOP").exists()
    assert not (stage / "STOP").exists()


def test_independent_heartbeat_during_blocking_move(publication, monkeypatch):
    stage, dest, _, row = publication
    name = hashlib.sha256(row["identity_key"].encode()).hexdigest() + ".json"
    manifest = stage / "config.json"
    MOVER.raw_atomic(manifest, {**{k: v for k, v in CONTEXT.items() if k != "config_sha256"}, "expected_publications": [name], "expected_identities": [row["identity_key"]]})
    row["config_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    MOVER.raw_atomic(stage / "published" / name, row)
    heartbeat = stage / "mover/heartbeat.json"
    original = MOVER.raw_move
    def blocked(*args, **kwargs):
        deadline = time.monotonic() + 2
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        initial = json.loads(heartbeat.read_text())["time"]
        # 模拟长 rsync，主线程完全阻塞时心跳仍由独立线程更新。
        time.sleep(10.2)
        assert json.loads(heartbeat.read_text())["time"] > initial
        return original(*args, **kwargs)
    monkeypatch.setattr(MOVER, "raw_move", blocked)
    args = SimpleNamespace(stage=str(stage), dest=str(dest), manifest=str(manifest), min_free_gib=0,
                           heartbeat_file=None, error_file=None, stop_file=None, once=True, interval=.01)
    assert MOVER.raw_main(args) == 0


def test_extra_file_and_frame_index_mismatch_fail(publication):
    stage, dest, source, row = publication
    (source / "extra").write_text("额外文件")
    with pytest.raises(ValueError, match="集合"):
        MOVER.raw_move(row, stage, dest, min_free_gib=0)
    (source / "frames-front.jsonl").write_text(json.dumps({"idx": 0, "enc": 1, "sha256": "c" * 64}) + "\n")
    with pytest.raises(ValueError, match="缺口"):
        MOVER.raw_verify_media(source)


def test_polling_does_not_rehash_delivered_large_files(publication, monkeypatch):
    stage, dest, _, row = publication
    name = hashlib.sha256(row["identity_key"].encode()).hexdigest() + ".json"
    manifest = stage / "config.json"
    MOVER.raw_atomic(manifest, {**{k: v for k, v in CONTEXT.items() if k != "config_sha256"}, "expected_publications": [name], "expected_identities": [row["identity_key"]]})
    row["config_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    MOVER.raw_atomic(stage / "published" / name, row)
    signature = MOVER.raw_signature
    calls = []
    hashes = []
    tree = MOVER.raw_tree
    def tracked_tree(path):
        if path == dest / row["relative_dir"]:
            hashes.append(path)
        return tree(path)
    def tracked_signature(path):
        calls.append(path)
        if len(calls) == 4:
            MOVER.raw_atomic(stage / "gpu_work_complete.json", {k: row[k] for k in MOVER.RAW_CONTEXT})
        return signature(path)
    monkeypatch.setattr(MOVER, "raw_tree", tracked_tree)
    monkeypatch.setattr(MOVER, "raw_signature", tracked_signature)
    args = SimpleNamespace(stage=str(stage), dest=str(dest), manifest=str(manifest), min_free_gib=0,
                           heartbeat_file=None, error_file=None, stop_file=None, once=False, interval=.01)
    assert MOVER.raw_main(args) == 0
    assert len(calls) == 4
    assert len(hashes) == 1  # 三次轮询不重哈希，最后完整核验一次。


@pytest.mark.parametrize("field,value", [("infra", True), ("recorder_verify", "FAIL")])
def test_result_infrastructure_and_recorder_failure(publication, field, value):
    _, _, source, _ = publication
    result = json.loads((source / "result.json").read_text())
    result[field] = value
    MOVER.raw_atomic(source / "result.json", result)
    with pytest.raises(ValueError, match="验收失败"):
        MOVER.raw_verify_media(source)


def test_summary_failure_and_both_views_missing_tail(publication):
    _, _, source, _ = publication
    summary = json.loads((source / "summary.json").read_text())
    summary["dropped"] = 1
    MOVER.raw_atomic(source / "summary.json", summary)
    with pytest.raises(ValueError, match="汇总存在失败"):
        MOVER.raw_verify_media(source)
    summary["dropped"] = 0
    MOVER.raw_atomic(source / "summary.json", summary)
    for stream in ("front", "wrist"):
        path = source / f"frames-{stream}.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="图像事件不符|视频帧数"):
        MOVER.raw_verify_media(source)


def test_cross_model_same_key_and_wrong_directory_fail(publication):
    stage, dest, source, row = publication
    wrong = {**row, "model": "pp", "identity_key": "pp:ood:key"}
    with pytest.raises(ValueError, match="目录不符"):
        MOVER.raw_validate_publication(wrong, CONTEXT)
    with pytest.raises(ValueError, match="身份不符"):
        MOVER.raw_move(wrong, stage, dest, min_free_gib=0)
    assert (source / "front.mkv").exists()


def test_delete_recovery_needs_only_floor_space(publication, monkeypatch):
    stage, dest, _, row = publication
    MOVER.raw_move(row, stage, dest, min_free_gib=0)
    monkeypatch.setattr(MOVER, "free_gib", lambda _: 50)
    assert MOVER.raw_move(row, stage, dest)["result"] == "moved"
    monkeypatch.setattr(MOVER, "free_gib", lambda _: 49.9)
    with pytest.raises(ValueError, match="low_disk"):
        MOVER.raw_move(row, stage, dest)


def test_mover_error_is_independent_and_preserves_first_cause(publication):
    stage, dest, source, row = publication
    name = hashlib.sha256(row["identity_key"].encode()).hexdigest() + ".json"
    manifest = stage / "config.json"
    MOVER.raw_atomic(manifest, {**{k: v for k, v in CONTEXT.items() if k != "config_sha256"}, "expected_publications": [name], "expected_identities": [row["identity_key"]]})
    row["config_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    MOVER.raw_atomic(stage / "published" / name, row)
    (source / "front.mkv").write_bytes("损坏的源".encode())
    error_path, stop = stage / "control/mover-error.json", stage / "control/STOP"
    args = SimpleNamespace(stage=str(stage), dest=str(dest), manifest=str(manifest), min_free_gib=0,
                           heartbeat_file=None, error_file=str(error_path), stop_file=str(stop), once=True, interval=.01)
    assert MOVER.raw_main(args) == 1
    original = error_path.read_bytes()
    report = json.loads(original)
    assert report["component"] == "mover"
    assert report["config_sha256"] == row["config_sha256"]
    assert "SHA" in report["message"]
    assert not stop.exists() and not (stage / "STOP").exists()
    assert (source / "arrays.npz").exists()
    assert not (dest / row["relative_dir"]).exists()
    # 独立消费真实序列化报告，确认供监督器判断自身错误，而无调度停止指令。
    assert report["schema"] == MOVER.RAW_SCHEMA and report["run_name"] == CONTEXT["run_name"]
    assert "stop_gpu" not in report and "cancel_jobs" not in report
    MOVER.raw_report_error(error_path, {**report, "message": "后续报告写入错误"})
    assert error_path.read_bytes() == original


def test_cli_preflight_failure_does_not_write_stop(tmp_path):
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    stage.mkdir()
    manifest = stage / "broken-config.json"
    manifest.write_text("{破损配置")
    error, stop = stage / "control/mover-error.json", stage / "control/STOP"
    proc = subprocess.run([sys.executable, str(ROOT / "dev-scripts/gl/eval_video_mover.py"),
                           "--layout", "raw-ood", "--stage", str(stage), "--dest", str(dest),
                           "--manifest", str(manifest), "--error-file", str(error), "--stop-file", str(stop), "--once"], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 1
    report = json.loads(error.read_text())
    assert report["error"] == "JSONDecodeError" and report["component"] == "mover"
    assert "RAW_DELIVERY=FAIL" in proc.stdout
    assert not stop.exists() and not (stage / "STOP").exists()


def test_manual_stop_is_preserved_as_input(publication):
    stage, dest, source, row = publication
    name = hashlib.sha256(row["identity_key"].encode()).hexdigest() + ".json"
    manifest = stage / "config.json"
    MOVER.raw_atomic(manifest, {**{k: v for k, v in CONTEXT.items() if k != "config_sha256"}, "expected_publications": [name], "expected_identities": [row["identity_key"]]})
    stop = stage / "control/STOP"
    MOVER.raw_atomic(stop, {"reason": "用户手动停止"})
    stop_bytes = stop.read_bytes()
    args = SimpleNamespace(stage=str(stage), dest=str(dest), manifest=str(manifest), min_free_gib=0,
                           heartbeat_file=None, error_file=None, stop_file=str(stop), once=True, interval=.01)
    assert MOVER.raw_main(args) == 1
    assert stop.read_bytes() == stop_bytes
    assert (source / "front.mkv").exists()
    assert "人工 STOP" in json.loads((stage / "mover/error.json").read_text())["message"]
    paused = json.loads((stage / "mover/paused.json").read_text())
    assert paused["paused"] is True and paused["unfinished"] == 1
    assert not (stage / "mover/completed.json").exists()


@pytest.mark.parametrize("mutation", ["index_missing", "state_missing", "state_changed", "unknown_key", "summary_error", "summary_count", "missing_unknown", "demo_state_changed", "enc_collision", "enc_order"])
def test_complete_arrays_and_encoded_frame_bijection(publication, mutation):
    _, _, source, _ = publication
    if mutation == "index_missing":
        (source / "arrays-index.jsonl").unlink()
    elif mutation in ("state_missing", "state_changed", "unknown_key"):
        with np.load(source / "arrays.npz") as existing:
            payload = {k: existing[k] for k in existing.files}
        if mutation == "state_missing":
            payload.pop("exec_state__00000")
        elif mutation == "state_changed":
            payload["exec_state__00000"] = np.ones(5, dtype=np.float64)
        else:
            payload["unknown__00000"] = np.zeros(1)
        np.savez(source / "arrays.npz", **payload)
    elif mutation.startswith("enc_"):
        path = source / "frames-front.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "enc_collision":
            rows[1]["enc"] = 0
        else:
            rows[0]["enc"], rows[1]["enc"] = 1, 0
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    else:
        path = source / "trace.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "summary_error":
            rows[-1]["arrays"]["error"] = "原数组写入失败"
        elif mutation == "summary_count":
            rows[-1]["arrays"]["state_keys"] = 0
        elif mutation == "demo_state_changed":
            next(r for r in rows if r["kind"] == "demo")["states"][0]["sha256"] = "f" * 64
        else:
            rows[-1]["arrays"]["missing_state_steps"] = [1]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        MOVER.raw_verify_media(source)


def test_explicit_missing_state_is_distinct_from_complete_state(publication, capsys):
    _, _, source, _ = publication
    MOVER.raw_verify_media(source)
    assert "level=complete_state" in capsys.readouterr().out
    with np.load(source / "arrays.npz") as existing:
        payload = {k: existing[k] for k in existing.files if k != "exec_state__00000"}
    np.savez(source / "arrays.npz", **payload)
    index_path = source / "arrays-index.jsonl"
    entries = [json.loads(line) for line in index_path.read_text().splitlines()]
    index_path.write_text("\n".join(json.dumps(r) for r in entries if r["name"] != "exec_state") + "\n")
    path = source / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(r for r in rows if r["kind"] == "step")["state"] = None
    rows[-1]["arrays"].update(state_keys=0, missing_state_steps=[1])
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    MOVER.raw_verify_media(source)
    assert "level=declared_missing_state" in capsys.readouterr().out


@pytest.mark.parametrize("mutation", ["schema", "bytes_bool", "decoded_bool", "trace_false", "extra", "dest"])
def test_shared_receipt_validator_rejects_malformed_delivery(publication, mutation):
    stage, dest, _, row = publication
    receipt = MOVER.raw_move(row, stage, dest, min_free_gib=0)
    if mutation == "schema":
        receipt["schema"] = "wrong"
    elif mutation == "bytes_bool":
        receipt["bytes"] = True
    elif mutation == "decoded_bool":
        receipt["decoded"]["front"] = True
    elif mutation == "trace_false":
        receipt["trace_checked"] = False
    elif mutation == "extra":
        receipt["unknown"] = 1
    else:
        receipt["dest"] = str(dest / "wrong")
    with pytest.raises(ValueError):
        MOVER.raw_validate_receipt(receipt, row, CONTEXT, dest)


def test_crash_between_large_file_deletions_resumes_without_recopy(publication, monkeypatch):
    stage, dest, source, row = publication
    original = Path.unlink
    def interrupted(path, *args, **kwargs):
        if path == source / "wrist.mkv":
            raise OSError("模拟删除窗口中断")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(OSError, match="中断"):
        MOVER.raw_move(row, stage, dest, min_free_gib=0)
    assert not (source / "front.mkv").exists()
    assert (source / "wrist.mkv").exists() and (source / "arrays.npz").exists()
    target = dest / row["relative_dir"]
    signature = MOVER.raw_signature(target)
    monkeypatch.setattr(Path, "unlink", original)
    receipt = MOVER.raw_move(row, stage, dest, min_free_gib=0)
    MOVER.raw_validate_receipt(receipt, row, CONTEXT, dest)
    assert MOVER.raw_signature(target) == signature
    assert MOVER.raw_tree(source) == {k: v for k, v in row["files"].items() if k not in row["delete_files"]}


def test_mover_forwards_complete_manifest_to_shared_context_helper(publication, monkeypatch):
    import raw_ood_context
    stage, dest, _, row = publication
    name = hashlib.sha256(row["identity_key"].encode()).hexdigest() + ".json"
    manifest = stage / "config.json"
    MOVER.raw_atomic(manifest, {**{k: v for k, v in CONTEXT.items() if k != "config_sha256"}, "expected_publications": [name], "expected_identities": [row["identity_key"]], "full_config_marker": "必须保留"})
    row["config_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    MOVER.raw_atomic(stage / "published" / name, row)
    original, calls = raw_ood_context.match_context, []
    def checked(publication, context):
        assert context["full_config_marker"] == "必须保留"
        calls.append(context)
        return original(publication, context)
    monkeypatch.setattr(raw_ood_context, "match_context", checked)
    args = SimpleNamespace(stage=str(stage), dest=str(dest), manifest=str(manifest), min_free_gib=0,
                           heartbeat_file=None, error_file=None, stop_file=None, once=True, interval=.01)
    assert MOVER.raw_main(args) == 0 and len(calls) >= 3


def test_error_report_write_failure_keeps_original_diagnostic(tmp_path, monkeypatch, capsys):
    def fail_write(*_):
        raise OSError("报告盘不可写")
    monkeypatch.setattr(MOVER, "raw_atomic", fail_write)
    MOVER.raw_report_error(tmp_path / "mover-error.json", {"error": "ValueError", "message": "首因：源 SHA 不符"})
    output = capsys.readouterr().out
    assert "RAW_DELIVERY=FAIL error=ValueError message=首因：源 SHA 不符" in output
    assert "RAW_ERROR_WRITE_FAILED" in output and "报告盘不可写" in output
    assert not (tmp_path / "STOP").exists()
