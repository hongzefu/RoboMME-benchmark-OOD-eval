"""原始录像搬运真实字节、崩溃窗口与错误封闭测试；不触发环境。"""
import hashlib
import importlib.util
import json
import subprocess
import time
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
    episode = stage / "rollouts/model/ood/seed7/raw/VideoUnmask_ep0_xhard1"
    episode.mkdir(parents=True)
    dest.mkdir()
    # 真 AV1 单帧解码，而非用文件存在代替解码。
    proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=black:s=16x16:r=30", "-frames:v", "1", "-c:v", "libaom-av1", "-cpu-used", "8", "-pix_fmt", "yuv444p", str(episode / "front.mkv")], capture_output=True)
    assert proc.returncode == 0, proc.stderr
    (episode / "wrist.mkv").write_bytes((episode / "front.mkv").read_bytes())
    MOVER.raw_atomic(episode / "meta.json", {"raw_codec": "av1-yuv444p", "level": 0})
    for stream in ("front", "wrist"):
        (episode / f"frames-{stream}.jsonl").write_text(json.dumps({"idx": 0, "enc": 0, "sha256": "c" * 64}) + "\n")
    action = np.zeros(8, dtype=np.float32)
    np.savez(episode / "arrays.npz", exec_action__00000=action)
    identity = {"dataset": "ood", "task": "VideoUnmask", "tier": "xhard1", "seed": 0, "key": "key"}
    rows = [{"kind": "header", "schema": "sgeval-trace/1", "identity": identity}, {"kind": "demo"},
            {"kind": "step", "step": 1, "action": {"dtype": action.dtype.str, "shape": [8], "sha256": hashlib.sha256(action.tobytes()).hexdigest(), "f32hex": action.tobytes().hex()}},
            {"kind": "end", "exec_steps": 1}]
    (episode / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    MOVER.raw_atomic(episode / "result.json", {**identity, "task_success": False, "exec_steps": 1, "error_kind": None})
    row = {"schema": MOVER.RAW_SCHEMA, **CONTEXT, "model": "model", "shard_id": "0", "identity_key": "model:key", "relative_dir": str(episode.relative_to(stage)), "files": MOVER.raw_tree(episode), "delete_files": ["front.mkv", "wrist.mkv", "arrays.npz"]}
    return stage, dest, episode, row


def test_real_move_retains_small_files_and_recovers_delete_window(publication):
    stage, dest, source, row = publication
    MOVER.raw_validate_publication(row, CONTEXT)
    receipt = MOVER.raw_move(row, stage, dest, min_free_gib=0)
    assert receipt["decoded"] == {"front": 1, "wrist": 1}
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
    with pytest.raises(ValueError, match="动作"):
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
    assert (stage / "control/STOP").exists()


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
