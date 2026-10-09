"""C13-SG-VIDEO-LAYOUT：按上游 mme-vla 布局发布账本接受的官方版式视频（1006 计划八.7、接口冻结说明第六节）。

夹具全部手写：运行根 ``<run>/s00/<目录名>/results.jsonl`` 与账本、局目录 ``<run>/<目录名>/<dataset>/new/<key>.a<N>/``
（``trace.jsonl`` header／demo／end、``official/`` 里的假 mp4 字节与重绘器 ``render.json``）。发布不解码视频，只核字节与
sha256，所以 mp4 用固定字节即可。期望文件名、索引列与判定行直接写在用例里。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests._support.dev_loaders import load_script

GOAL = "pick the cube"
COLUMNS = ["model_id", "policy_seed", "dataset", "side", "key", "accepted_attempt_id", "episode_id", "terminal",
           "src_rel", "src_sha256", "dst_name"]


def P():
    return load_script("eval-official/publish_videos.py")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Run:
    """一个运行根：逐局写局目录、结果行与账本行。"""

    def __init__(self, root: Path, *, policy="groundsg", variant="ground-sg-qwenvl", dataset="ood"):
        self.root, self.policy, self.variant, self.dataset = Path(root), policy, variant, dataset
        dirname = f"{policy}-{variant}" if variant else policy
        self.seat = self.root / "s00" / dirname
        self.media = self.root / dirname / dataset / "new"
        self.seat.mkdir(parents=True, exist_ok=True)
        self.media.mkdir(parents=True, exist_ok=True)

    def _append(self, name, row):
        with open(self.seat / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def episode(self, k: int, *, attempt=1, status="success", accept=True, infra=False, cap_hit=False,
                video: bytes | None = None, no_frame=False, policy_seed=7, row_seed=7, end_extra=None,
                tier="xhard1", task="PickXtimes") -> Path:
        env_seed = 7_000_000 + k
        key = f"{task}_{tier}_{env_seed}"
        ident = {"task": task, "tier": tier, "seed": env_seed, "dataset": self.dataset, "key": key,
                 "builder_episode": k, "source_episode": 100 + k, "attempt": attempt}
        d = self.media / f"{key}.a{attempt}"
        (d / "official").mkdir(parents=True)
        header = {"kind": "header", "route": f"{self.policy}/new", "identity": ident, "max_steps": 1800}
        if policy_seed is not None:
            header["policy_seed"] = policy_seed
        end_status = "timeout" if cap_hit else status
        end = {"kind": "end", "status": end_status, "terminal_reason": "error" if cap_hit else status,
               "frames_recorded": 0 if no_frame else 5, **({"cap_hit": True} if cap_hit else {}),
               **({"no_frame": True} if no_frame else {}), **(end_extra or {})}
        rows = [header, {"kind": "demo", "texts": [GOAL]}, end]
        (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        if not no_frame:
            data = video if video is not None else f"mp4-{key}-a{attempt}".encode()
            v = d / "official" / f"official-rerender__{task}_ep{k}a{attempt}_{status}_{GOAL}_{tier}.mp4"
            v.write_bytes(data)
            (d / "official" / "render.json").write_text(json.dumps(
                {"identity": ident, "task_goal": GOAL, "policy_seed": policy_seed,
                 "output_fingerprint": {"sha256": _sha(data), "size": len(data)}}), encoding="utf-8")
        aid = f"{key}-a{attempt}"
        row = {"policy": self.policy, "dataset": self.dataset, "key": key, "task": task, "tier": tier, "seed": env_seed,
               "identity": {"tier": tier, "seed": env_seed, "builder_episode": k}, "attempt_id": aid,
               "attempt_no": attempt, "attempt": attempt, "status": "timeout" if cap_hit else status,
               "cap_hit": cap_hit, "infra": infra, "canary": False, "late": False}
        if self.variant:
            row["policy_variant"] = self.variant
        if row_seed is not None:
            row["policy_seed"] = row_seed
        self._append(f"{self.policy}.ledger.jsonl", {"kind": "attempt_start", "key": key, "attempt_id": aid,
                                                     "attempt_no": attempt, "policy": self.policy})
        self._append("results.jsonl", row)
        if accept:
            self._append(f"{self.policy}.ledger.jsonl", {"kind": "accept", "key": key, "attempt_id": aid,
                                                         "attempt_no": attempt, "accepted_attempt_id": aid,
                                                         "policy": self.policy})
        return d


def _publish(capsys, run: Run, out: Path, *extra, model="groundsg", variant="ground-sg-qwenvl", dataset="ood"):
    argv = ["--run-root", str(run.root), "--out-root", str(out), "--model-id", model, "--policy-seed", "7",
            "--dataset", dataset, "--side", "new", *extra]
    if variant:
        argv += ["--variant", variant]
    capsys.readouterr()
    rc = P().main(argv)
    lines = capsys.readouterr().out.strip().splitlines()
    return rc, lines[-1], lines


def _kv(line: str) -> dict:
    head, *rest = line.split()
    return {"": head.split("=", 1)[1], **dict(x.split("=", 1) for x in rest if "=" in x)}


def _index(vdir: Path) -> list[dict]:
    lines = (vdir / "index.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == COLUMNS
    return [dict(zip(COLUMNS, x.split("\t"))) for x in lines[1:]]


def test_publish_layout_three_terminals_index_and_idempotent(tmp_path, capsys):
    run = Run(tmp_path / "run")
    e0 = run.episode(0, status="success")
    run.episode(1, status="fail")
    run.episode(2, cap_hit=True)  # strict-cap 命中：trace terminal_reason=error、status=timeout → 命名 timeout
    out = tmp_path / "pub"
    rc, line, _ = _publish(capsys, run, out)
    print(line)
    vdir = out.resolve() / "groundsg" / "seed7" / "qwenvl" / "videos"
    assert rc == 0 and line == ("VIDEO_LAYOUT=PASS model=groundsg seed=7 videos=3 no_frame_error=0 error_named=0 "
                                f"accepted=3 published=3 reused=0 problems=0 mode=publish dir={vdir}")
    names = sorted(p.name for p in vdir.glob("*.mp4"))
    assert names == [f"PickXtimes_ep0_success_{GOAL}_xhard1.mp4", f"PickXtimes_ep1_fail_{GOAL}_xhard1.mp4",
                     f"PickXtimes_ep2_timeout_{GOAL}_xhard1.mp4"]
    assert not any("_error_" in n for n in names)
    src = next((e0 / "official").glob("*.mp4"))
    dst = vdir / names[0]
    assert os.stat(src).st_ino == os.stat(dst).st_ino  # 默认硬链接；局目录原位不动
    assert src.exists() and (e0 / "trace.jsonl").exists()
    idx = {r["key"]: r for r in _index(vdir)}
    r0 = idx["PickXtimes_xhard1_7000000"]
    assert r0 == {"model_id": "groundsg", "policy_seed": "7", "dataset": "ood", "side": "new",
                  "key": "PickXtimes_xhard1_7000000", "accepted_attempt_id": "PickXtimes_xhard1_7000000-a1",
                  "episode_id": "0", "terminal": "success",
                  "src_rel": f"groundsg-ground-sg-qwenvl/ood/new/PickXtimes_xhard1_7000000.a1/official/{src.name}",
                  "src_sha256": _sha(src.read_bytes()), "dst_name": names[0]}
    assert idx["PickXtimes_xhard1_7000002"]["terminal"] == "timeout"
    # 重复发布：同名同 sha 幂等跳过
    before = (vdir / "index.tsv").read_bytes()
    rc, line, _ = _publish(capsys, run, out)
    print(line)
    assert rc == 0 and _kv(line)["published"] == "0" and _kv(line)["reused"] == "3" and _kv(line)[""] == "PASS"
    assert (vdir / "index.tsv").read_bytes() == before
    rc, line, _ = _publish(capsys, run, out, "--verify")
    assert rc == 0 and _kv(line)[""] == "PASS" and _kv(line)["mode"] == "verify"


def test_only_accepted_attempt_is_published(tmp_path, capsys):
    """infra a1（有视频、未接受）＋ accepted a2：只发 a2，索引记 a2 的 accepted_attempt_id，文件名不带 attempt。"""
    run = Run(tmp_path / "run")
    run.episode(0, attempt=1, status="fail", accept=False, infra=True, video=b"attempt-1-video")
    run.episode(0, attempt=2, status="success", video=b"attempt-2-video")
    out = tmp_path / "pub"
    rc, line, _ = _publish(capsys, run, out)
    print(line)
    vdir = out / "groundsg" / "seed7" / "qwenvl" / "videos"
    assert rc == 0 and _kv(line)["videos"] == "1" and _kv(line)["published"] == "1"
    (only,) = list(vdir.glob("*.mp4"))
    assert only.name == f"PickXtimes_ep0_success_{GOAL}_xhard1.mp4"
    assert only.read_bytes() == b"attempt-2-video"
    (row,) = _index(vdir)
    assert row["accepted_attempt_id"] == "PickXtimes_xhard1_7000000-a2" and row["src_sha256"] == _sha(b"attempt-2-video")


def test_same_name_different_sha_fails_and_verify_catches_tamper(tmp_path, capsys):
    run = Run(tmp_path / "run")
    e0 = run.episode(0, video=b"first")
    out = tmp_path / "pub"
    assert _publish(capsys, run, out)[0] == 0
    vdir = out / "groundsg" / "seed7" / "qwenvl" / "videos"
    dst = vdir / f"PickXtimes_ep0_success_{GOAL}_xhard1.mp4"
    # 局目录里的视频被换成另一份（sidecar 一并改，局内自洽）：同名异 sha → conflict，不覆盖
    src = next((e0 / "official").glob("*.mp4"))
    src.unlink()  # 断开硬链接，避免改动传到已发布文件
    src.write_bytes(b"second")
    side = json.loads((e0 / "official" / "render.json").read_text())
    side["output_fingerprint"]["sha256"] = _sha(b"second")
    (e0 / "official" / "render.json").write_text(json.dumps(side))
    rc, line, lines = _publish(capsys, run, out)
    print(line)
    assert rc == 1 and _kv(line)[""] == "FAIL" and _kv(line)["problems"] == "1"
    assert any('"kind": "conflict"' in x for x in lines) and dst.read_bytes() == b"first"
    # verify：已发布文件与期望 sha 不符
    rc, line, _ = _publish(capsys, run, out, "--verify")
    assert rc == 1 and _kv(line)[""] == "FAIL"


def test_all_86_identities_no_frame(tmp_path, capsys):
    """86 个身份全是无帧 error：只进索引（dst_name 为空）、不出视频；accepted = 0 + 86 = 86。"""
    run = Run(tmp_path / "run")
    for k in range(86):
        run.episode(k, status="error", no_frame=True)
    out = tmp_path / "pub"
    rc, line, _ = _publish(capsys, run, out, "--expect-total", "86")
    print(line)
    kv = _kv(line)
    assert rc == 0 and (kv[""], kv["videos"], kv["no_frame_error"], kv["accepted"], kv["error_named"]) == \
        ("PASS", "0", "86", "86", "0")
    vdir = out / "groundsg" / "seed7" / "qwenvl" / "videos"
    rows = _index(vdir)
    assert len(rows) == 86 and all(r["dst_name"] == "" and r["terminal"] == "error" for r in rows)
    assert not list(vdir.glob("*.mp4"))
    rc, line, _ = _publish(capsys, run, out, "--expect-total", "87", "--verify")
    assert rc == 1 and _kv(line)[""] == "FAIL"


def test_error_with_frames_is_error_named_and_not_published(tmp_path, capsys):
    run = Run(tmp_path / "run")
    run.episode(0, status="success")
    run.episode(1, status="error")  # 有帧有视频的 error 局：本版不允许 error 命名
    out = tmp_path / "pub"
    rc, line, _ = _publish(capsys, run, out)
    print(line)
    kv = _kv(line)
    assert rc == 1 and kv[""] == "FAIL" and kv["error_named"] == "1" and kv["videos"] == "1"
    vdir = out / "groundsg" / "seed7" / "qwenvl" / "videos"
    assert not any("_error_" in p.name for p in vdir.glob("*.mp4"))


def test_policy_seed_mismatch_and_unproven(tmp_path, capsys):
    run = Run(tmp_path / "run")
    run.episode(0, policy_seed=42, row_seed=7)            # trace 与 sidecar 写了 42
    run.episode(1, policy_seed=None, row_seed=None)        # 一处都没写：不补成已证种子
    rc, line, lines = _publish(capsys, run, tmp_path / "pub")
    print(line)
    assert rc == 1 and _kv(line)["problems"] == "2"
    assert any('"seed_mismatch"' in x for x in lines) and any('"seed_unproven"' in x for x in lines)


def test_hard_verify_source_episode_copy_mode_and_no_variant(tmp_path, capsys):
    run = Run(tmp_path / "run", policy="pp", variant=None, dataset="hard-verify")
    e0 = run.episode(0, tier="xhard0")
    out = tmp_path / "pub"
    rc, line, _ = _publish(capsys, run, out, "--mode", "copy", model="pp", variant=None, dataset="hard-verify")
    print(line)
    vdir = out.resolve() / "pp" / "seed7" / "videos"
    assert rc == 0 and line.endswith(f"dir={vdir}")
    dst = vdir / f"PickXtimes_ep100_success_{GOAL}_xhard0.mp4"  # hard-verify 用官方源局号
    src = next((e0 / "official").glob("*.mp4"))
    assert dst.read_bytes() == src.read_bytes() and os.stat(dst).st_ino != os.stat(src).st_ino
    with pytest.raises(SystemExit):  # groundsg 必须给变体，其他模型不得给
        P().main(["--run-root", str(run.root), "--model-id", "groundsg", "--policy-seed", "7", "--dataset", "ood",
                  "--side", "new"])
    with pytest.raises(SystemExit):
        P().main(["--run-root", str(run.root), "--model-id", "pp", "--variant", "ground-sg-oracle",
                  "--policy-seed", "7", "--dataset", "ood", "--side", "new"])


def test_verify_rejects_error_named_and_extra_files(tmp_path, capsys):
    run = Run(tmp_path / "run")
    run.episode(0)
    out = tmp_path / "pub"
    assert _publish(capsys, run, out)[0] == 0
    vdir = out / "groundsg" / "seed7" / "qwenvl" / "videos"
    (vdir / f"PickXtimes_ep9_error_{GOAL}_xhard1.mp4").write_bytes(b"old")
    rc, line, lines = _publish(capsys, run, out, "--verify")
    print(line)
    kv = _kv(line)
    assert rc == 1 and kv["error_named"] == "1" and any('"extra_files"' in x for x in lines)


def test_long_goal_truncated_with_hash(tmp_path, capsys):
    run = Run(tmp_path / "run")
    e0 = run.episode(0)
    long_goal = "目标" * 120
    rows = [json.loads(x) for x in (e0 / "trace.jsonl").read_text().splitlines()]
    side = json.loads((e0 / "official" / "render.json").read_text())
    side["task_goal"] = long_goal
    (e0 / "official" / "render.json").write_text(json.dumps(side, ensure_ascii=False))
    rc, line, _ = _publish(capsys, run, tmp_path / "pub")
    vdir = tmp_path / "pub" / "groundsg" / "seed7" / "qwenvl" / "videos"
    (dst,) = list(vdir.glob("*.mp4"))
    assert rc == 0 and len(dst.name.encode()) <= 255 and dst.name.endswith(".mp4") and "__" in dst.name
    assert rows[0]["kind"] == "header"
