"""C15.16 ``hard_parity.py`` 的 generate／publish／anchor／binding／import-delivery／export-xhard0-manifest 子命令与
compare 的参数闸门、校准与超容差分类，经真实 CLI 入口 ``hard_parity.main([...])`` 跑。

外部依赖一律在测试进程内替换 ``hard_parity`` 模块自己的 ``subprocess`` 名字（scripts/ 侧的临时 patch，不碰 src/robomme）：

- ``nvidia-smi``、``git rev-parse``（src_root 提交）、``versions()`` 子进程：回固定文本；
- 运行器（``train_split_runner.py``／``generate_h5.py``）：按命令行里的 jobs／identities 在输出根写微型 h5 与
  ``results.partial.jsonl``，由真实 ``Mover`` 线程扫成 identities；
- HF CLI（``hf buckets list／sync／cp``）：list 回预置文本、cp 从本地对应文件拷贝（可篡改或缺失）。

锚点登记表 ``ANCHORS`` 与容差文件 ``TOLERANCES`` 都改指 tmp 下的副本，不写仓库。期望（行数、计数、判定行字段）由本文件
手写的输入推出。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import parity_fixtures as F


@pytest.fixture(scope="module")
def hp():
    return F.hard_parity()


def fields(line: str) -> dict[str, str]:
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


def _cp(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class FakeSub:
    """替身 ``subprocess``：只认 hard_parity 会起的几类命令，其余一律报错（防止漏网的真实外部调用）。"""

    STDOUT = subprocess.STDOUT

    def __init__(self, *, gpu: str = "NVIDIA A40, 550.54.15", runner=None, hf=None, versions_out='{"python": "3.11.0"}'):
        self.gpu, self.runner, self.hf, self.versions_out = gpu, runner, hf, versions_out
        self.calls: list[tuple[list[str], dict]] = []

    def run(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, kw))
        if cmd[0] == "nvidia-smi":
            return _cp(self.gpu + "\n")
        if cmd[0] == "git":
            return _cp("f" * 40 + "\n")
        if len(cmd) > 2 and cmd[1] == "-c":
            return _cp(self.versions_out + "\n" if self.versions_out else "", stderr="版本探针失败")
        if "buckets" in cmd:
            return self.hf(cmd)
        if self.runner is not None:
            return self.runner(cmd, kw)
        raise AssertionError(f"意外的外部命令：{cmd}")


# ── generate ─────────────────────────────────────────────────────────────


def _native_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"rows_total": len(rows), "rows": [
        {"task": r["task"], "difficulty": r["tier"], "episode": r["episode"], "seed": r["seed"]} for r in rows]}))
    return path


ROWS = [{"task": "PickXtimes", "tier": "easy", "episode": e, "seed": 7000 + e} for e in range(3)]


def _runner_writes(skip: set[int] = frozenset(), module="robomme.robomme_env.PickXtimes"):
    """运行器替身：按 jobs.json 逐局写 h5 + results.partial.jsonl；skip 里的 seed 不产出（模拟漏局）。"""
    def runner(cmd, kw):
        jobs_path = Path(cmd[cmd.index("--jobs-json") + 1])
        out = jobs_path.parent.parent
        jobs = json.loads(jobs_path.read_text())
        partial = out / "_runner" / "results.partial.jsonl"
        for job in jobs:
            if job["seed"] in skip:
                continue
            F.write_h5(Path(job["worker_dir"]) / "hdf5_files" / "x.h5", seed=job["seed"] % 97, frames=2)
            rec = {"task": job["task"], "episode": job["episode"], "seed": job["seed"], "difficulty": job["difficulty"],
                   "ok": True, "env_package": kw["env"].get("ROBOMME_ENV_PACKAGE"), "env_module": module}
            with partial.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        (out / "_runner" / "results.json").write_text(json.dumps(
            {"robomme_module": "/src/src/robomme/__init__.py", "worker": "official._worker"}))
        return _cp()
    return runner


def _gen_argv(manifest: Path, out: Path, *, side="O", tier="native", extra=()):
    return ["generate", "--side", side, "--tier", tier, "--manifest", str(manifest), "--src-root", str(out.parent / "src"),
            "--workers", "2", "--out", str(out), *extra]


def test_generate_native_records_every_identity(hp, tmp_path, monkeypatch, capsys):
    fake = FakeSub(runner=_runner_writes())
    monkeypatch.setattr(hp, "subprocess", fake)
    manifest = _native_manifest(tmp_path / "m.json", ROWS)
    out = tmp_path / "O-native"
    assert hp.main(_gen_argv(manifest, out)) == 0
    text = capsys.readouterr().out
    line = next(t for t in text.splitlines() if t.startswith("GENERATE="))
    assert fields(line)["GENERATE"] == "PASS" and (fields(line)["rows"], fields(line)["recorded"],
                                                   fields(line)["success"]) == ("3", "3", "3")
    assert text.count("EPISODE_DONE side=O easy/PickXtimes/") == 3
    ids = [json.loads(t) for t in (out / "identities.jsonl").read_text().splitlines()]
    assert sorted(r["seed"] for r in ids) == [7000, 7001, 7002]
    assert all(r["path"] and r["sha256"] == F.sha256(out / r["path"]) and r["frames"] == 2 for r in ids)
    launch = json.loads(next(out.glob("launch-*.json")).read_text())
    assert (launch["schema"], launch["side"], launch["rows"], launch["env_package"]) == (F.LAUNCH_SCHEMA, "O", 3, "robomme")
    assert launch["gpu_model"] == "NVIDIA A40" and launch["src_commit"] == "f" * 40 and launch["dev_smoke"] is False
    runner_cmd, kw = next((c, k) for c, k in fake.calls if str(hp.RUNNER) in c)
    assert "--force-mirror" not in runner_cmd and kw["env"]["ROBOMME_ENV_PACKAGE"] == "robomme"
    assert "PYTHONPATH" not in kw["env"]


def test_generate_h_side_xhard0_and_missing_episode_fails(hp, tmp_path, monkeypatch, capsys):
    """H 侧：--force-mirror、ROBOMME_ENV_PACKAGE=robomme_hard、xhard0 按数据集名走 hard-verify 路由；漏一局 → GENERATE=FAIL。"""
    rows = [dict(r, tier="hard") for r in ROWS]
    fake = FakeSub(runner=_runner_writes(skip={7001}, module="robomme_hard.robomme_env.PickXtimes"))
    monkeypatch.setattr(hp, "subprocess", fake)
    manifest = _native_manifest(tmp_path / "m.json", rows)
    out = tmp_path / "H-xhard0"
    assert hp.main(_gen_argv(manifest, out, side="H", tier="xhard0")) == 1
    f = fields(next(t for t in capsys.readouterr().out.splitlines() if t.startswith("GENERATE=")))
    assert f["GENERATE"] == "FAIL" and (f["rows"], f["recorded"]) == ("3", "2")
    cmd, kw = next((c, k) for c, k in fake.calls if str(hp.RUNNER) in c)
    assert kw["env"]["ROBOMME_ENV_PACKAGE"] == "robomme_hard"
    for flag in ("--force-mirror", "--identity-source", "--xhard0-manifest", "--builder-route"):
        assert flag in cmd
    assert cmd[cmd.index("--builder-route") + 1] == "hard-verify"


def test_generate_smoke_and_dev_smoke_labels(hp, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hp, "subprocess", FakeSub(gpu="NVIDIA RTX 6000 Ada Generation, 550.54.15",
                                                  runner=_runner_writes()))
    manifest = _native_manifest(tmp_path / "m.json", ROWS)
    assert hp.main(_gen_argv(manifest, tmp_path / "a", extra=("--dev-smoke", "--smoke", "1"))) == 0
    f = fields(next(t for t in capsys.readouterr().out.splitlines() if "_SMOKE=" in t))
    assert f["NATIVE_SMOKE"] == "PASS" and f["rows"] == "1"


@pytest.mark.parametrize("case,needle", [
    ("not_a40", "只许在 A40"),
    ("v9_not_h2", "只生成 H2"),
    ("xhard0_side_p", "xhard0 只生成 O、H"),
    ("native_identities", "--identities 只用于"),
    ("out_not_empty", "已存在且非空"),
])
def test_generate_gates(hp, tmp_path, monkeypatch, case, needle):
    gpu = "NVIDIA RTX 6000 Ada Generation, 1" if case == "not_a40" else "NVIDIA A40, 1"
    monkeypatch.setattr(hp, "subprocess", FakeSub(gpu=gpu))  # 闸门在起运行器之前；运行器被调用即 AssertionError
    manifest = _native_manifest(tmp_path / "m.json", ROWS)
    out = tmp_path / "out"
    side, tier, extra = "O", "native", []
    if case == "v9_not_h2":
        manifest = tmp_path / "d.json"
        manifest.write_text(json.dumps({"schema": "v8-delivery/1", "rows": []}))
        side, tier = "H", "v9"
    elif case == "xhard0_side_p":
        side, tier = "P", "xhard0"
    elif case == "native_identities":
        extra = ["--identities", str(F.write_jsonl(tmp_path / "i.jsonl", [{"task": "PickXtimes", "tier": "easy",
                                                                            "seed": 7000}]))]
    elif case == "out_not_empty":
        out.mkdir()
        (out / "old.txt").write_text("x")
    with pytest.raises(hp.ParityError, match=needle):
        hp.main(_gen_argv(manifest, out, side=side, tier=tier, extra=extra))


def test_generate_bad_expect_ref_leaves_no_files(hp, tmp_path, monkeypatch):
    monkeypatch.setattr(hp, "subprocess", FakeSub())
    bad = tmp_path / "ref.json"
    bad.write_text(json.dumps({"schema": "noise-ref/0"}))
    out = tmp_path / "out"
    with pytest.raises(hp.ParityError, match="--expect-ref 参照不可用"):
        hp.main(_gen_argv(_native_manifest(tmp_path / "m.json", ROWS), out, extra=("--expect-ref", str(bad))))
    assert not out.exists()


def _v9_delivery(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"schema": "v8-delivery/1", "rows": [
        {"task": r["task"], "tier": r["tier"], "episode": r["episode"], "seed": r["seed"], "candidate": r["episode"],
         "h5_sha256": "0" * 64} for r in rows]}))
    return path


V9_ROWS = [{"task": "MoveCube", "tier": "xhard4", "episode": e, "seed": 9100 + e} for e in range(3)]


def _replay_writes(cmd, kw):
    """generate_h5 --mode replay 替身：按 --identities 逐局写 h5 与 partial 行。"""
    out = Path(cmd[cmd.index("--output") + 1])
    ids = [json.loads(t) for t in Path(cmd[cmd.index("--identities") + 1]).read_text().splitlines()]
    partial = out / "results.partial.jsonl"
    for r in ids:
        wdir = out / "episodes" / r["tier"] / f"{r['task']}_episode_{r['seed']}"
        F.write_h5(wdir / "hdf5_files" / "x.h5", seed=r["seed"] % 97, frames=2)
        with partial.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"task": r["task"], "episode": r["seed"], "seed": r["seed"], "difficulty": r["tier"],
                                 "ok": True, "env_module": "robomme_hard.robomme_env.MoveCube"}) + "\n")
    return _cp()


def test_generate_v9_h2_replays_identity_subset(hp, tmp_path, monkeypatch, capsys):
    fake = FakeSub(runner=_replay_writes)
    monkeypatch.setattr(hp, "subprocess", fake)
    delivery = _v9_delivery(tmp_path / "d.json", V9_ROWS)
    ids = F.write_jsonl(tmp_path / "ids.jsonl", [{"task": r["task"], "tier": r["tier"], "seed": r["seed"]}
                                                 for r in V9_ROWS[:2]])
    out = tmp_path / "H2-v9"
    argv = _gen_argv(delivery, out, side="H2", tier="v9",
                     extra=("--identities", str(ids), "--specs-root", str(tmp_path / "specs")))
    assert hp.main(argv) == 0
    f = fields(next(t for t in capsys.readouterr().out.splitlines() if t.startswith("GENERATE=")))
    assert (f["GENERATE"], f["rows"], f["recorded"]) == ("PASS", "2", "2")
    cmd, _ = next((c, k) for c, k in fake.calls if str(hp.GENERATE_H5) in c)
    assert cmd[cmd.index("--mode") + 1] == "replay" and cmd[cmd.index("--specs") + 1] == str(tmp_path / "specs")
    assert [json.loads(t)["seed"] for t in (out / "_identities.jsonl").read_text().splitlines()] == [9100, 9101]


@pytest.mark.parametrize("case", ["stray", "no_specs"])
def test_generate_v9_gates(hp, tmp_path, monkeypatch, case):
    monkeypatch.setattr(hp, "subprocess", FakeSub())
    delivery = _v9_delivery(tmp_path / "d.json", V9_ROWS)
    extra = []
    if case == "stray":
        ids = F.write_jsonl(tmp_path / "ids.jsonl", [{"task": "MoveCube", "tier": "xhard4", "seed": 1}])
        extra = ["--identities", str(ids), "--specs-root", str(tmp_path / "s")]
    with pytest.raises(hp.ParityError, match="交付清单之外" if case == "stray" else "须给 --specs-root"):
        hp.main(_gen_argv(delivery, tmp_path / "out", side="H2", tier="v9", extra=extra))


def test_gpu_facts_and_versions(hp, monkeypatch):
    fake = FakeSub(gpu="NVIDIA A40, 550.1\nNVIDIA RTX 6000 Ada Generation, 550.2")
    monkeypatch.setattr(hp, "subprocess", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert hp.gpu_facts() == {"gpu_model": "NVIDIA RTX 6000 Ada Generation", "driver": "550.2"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")  # 非单个数字：取第一行
    assert hp.gpu_facts()["gpu_model"] == "NVIDIA A40"
    fake.gpu = ""
    assert hp.gpu_facts() == {"gpu_model": "unknown", "driver": "unknown"}
    assert hp.versions() == {"python": "3.11.0"}
    fake.versions_out = ""
    assert hp.versions() == {"error": "版本探针失败"}


# ── publish ──────────────────────────────────────────────────────────────


def _local_side(root: Path, n: int = 2) -> Path:
    lines = []
    for i in range(n):
        rel = f"episodes/T_episode_{i}/hdf5_files/x.h5"
        F.write_h5(root / rel, seed=i + 1, frames=2)
        lines.append(F.id_line("T", "easy", 100 + i, sha=F.sha256(root / rel), path=rel))
    lines.append(F.id_line("T", "easy", 999, ok=False))  # 失败局没有文件，不进 SHA256SUMS
    F.write_run(root, lines, workers=4, gpu="NVIDIA A40", host="gl1001")
    (root / "launch-1.json").write_text(json.dumps({"schema": F.LAUNCH_SCHEMA, "src_commit": "c" * 40,
                                                    "gpu_model": "NVIDIA A40", "driver": "1", "host": "gl1001",
                                                    "slurm_job": "123", "workers": 4, "manifest_sha256": "m" * 64}))
    return root


class FakeHF:
    def __init__(self, local: Path, *, listing: str = "", tamper: str | None = None, missing: str | None = None):
        self.local, self.listing, self.tamper, self.missing = local, listing, tamper, missing
        self.synced = []

    def __call__(self, cmd):
        op = cmd[cmd.index("buckets") + 1]
        if op == "list":
            return _cp(self.listing)
        if op == "sync":
            self.synced.append(cmd)
            return _cp()
        if op == "cp":
            rel = cmd[-2].split("/P-test/native/", 1)[1]
            if rel != self.missing:
                data = b"tampered" if rel == self.tamper else (self.local / rel).read_bytes()
                Path(cmd[-1]).write_bytes(data)
            return _cp()
        raise AssertionError(cmd)


def _pub_argv(local: Path, tmp: Path, *extra):
    return ["publish", "--side", "P", "--tier", "native", "--prefix", "P-test", "--local", str(local),
            "--readback-tmp", str(tmp), *extra]


def test_publish_readback_pass_and_manifest(hp, tmp_path, monkeypatch, capsys):
    local = _local_side(tmp_path / "P-native")
    hf = FakeHF(local)
    monkeypatch.setattr(hp, "subprocess", FakeSub(hf=hf))
    assert hp.main(_pub_argv(local, tmp_path)) == 0
    f = fields(capsys.readouterr().out.strip().splitlines()[-1])
    assert (f["BUCKET_SYNC"], f["objects"], f["readback_sha_equal"], f["mismatch"]) == ("PASS", "2", "2", "0")
    sums = (local / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 2 and all(s.split("  ")[0] == F.sha256(local / s.split("  ")[1]) for s in sums)
    m = json.loads((local / "manifest.json").read_text())
    assert (m["side"], m["tag"], m["objects"], m["job_ids"], m["nodes"], m["src_commit"]) == \
        ("P", "pre-hard-split", 2, ["123"], ["gl1001"], "c" * 40)
    assert len(hf.synced) == 1 and (local / "bucket-list.txt").exists()


@pytest.mark.parametrize("bad", ["tamper", "missing"])
def test_publish_readback_mismatch_fails(hp, tmp_path, monkeypatch, capsys, bad):
    local = _local_side(tmp_path / "P-native")
    rel = "episodes/T_episode_1/hdf5_files/x.h5"
    monkeypatch.setattr(hp, "subprocess", FakeSub(hf=FakeHF(local, **{bad: rel})))
    assert hp.main(_pub_argv(local, tmp_path)) == 1
    f = fields(capsys.readouterr().out.strip().splitlines()[-1])
    assert (f["BUCKET_SYNC"], f["mismatch"], f["readback_sha_equal"]) == ("FAIL", "1", "1")


def test_publish_gates_and_dry_run(hp, tmp_path, monkeypatch, capsys):
    local = _local_side(tmp_path / "P-native")
    # 远端目录已有对象：只增不改，拒传（除非 --resume-upload）
    monkeypatch.setattr(hp, "subprocess", FakeSub(hf=FakeHF(local, listing="ID  size\nx.h5  10\n")))
    with pytest.raises(hp.ParityError, match="只增不改"):
        hp.main(_pub_argv(local, tmp_path))
    assert hp.main(_pub_argv(local, tmp_path, "--resume-upload", "--dry-run")) == 0
    assert capsys.readouterr().out.startswith("PUBLISH_DRY_RUN remote=hf://buckets/")
    # 「(empty)」不算已有对象
    monkeypatch.setattr(hp, "subprocess", FakeSub(hf=FakeHF(local, listing="(empty)\n")))
    assert hp.main(_pub_argv(local, tmp_path, "--dry-run")) == 0
    # 本地副本被改：sha 与 identities 不符即停
    (local / "episodes/T_episode_0/hdf5_files/x.h5").write_bytes(b"x")
    with pytest.raises(hp.ParityError, match="本地副本 sha"):
        hp.main(_pub_argv(local, tmp_path, "--dry-run"))
    with pytest.raises(hp.ParityError, match="不存在"):
        hp.main(_pub_argv(tmp_path / "nope", tmp_path))


# ── anchor／binding ──────────────────────────────────────────────────────


TAG = "parity-anchor-v6"


def _tag_commit() -> str:
    out = subprocess.run(["git", "-C", str(F.REPO), "rev-parse", "--verify", "--quiet", f"{TAG}^{{commit}}"],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        pytest.skip(f"未验证：本检出没有 tag {TAG}")
    return out


def _side(root: Path, side: str, rows: list[dict], *, gpu="NVIDIA A40", driver="1", worker=None, module=None):
    lines = []
    for r in rows:
        rel = f"episodes/{r['task']}_episode_{r['episode']}/hdf5_files/x.h5"
        F.write_h5(root / rel, seed=r["seed"] % 97, frames=2)
        extra = {"episode": r["episode"], "recovery_mode": None}
        if worker:
            extra["worker"] = worker
        if module:
            extra["env_module"] = module
        if side == "O":
            extra["robomme_module"] = "/o/src/robomme/__init__.py"
        lines.append(F.id_line(r["task"], r["tier"], r["seed"], sha=F.sha256(root / rel), path=rel, **extra))
    return F.write_run(root, lines, workers=4, gpu=gpu, driver=driver)


def _register(hp, h5_root, commit, segments="native:P-native", *extra):
    return hp.main(["anchor", "register", "--tag", TAG, "--commit", commit, "--segments", segments,
                    "--h5-root", str(h5_root), *extra])


def test_anchor_register_check_and_tamper(hp, tmp_path, monkeypatch, capsys):
    commit = _tag_commit()
    monkeypatch.setattr(hp, "ANCHORS", tmp_path / "anchors.json")
    h5 = tmp_path / "h5"
    _side(h5 / "P-native", "P", ROWS, worker="train_split_worker.run_one", module="robomme_hard.robomme_env.PickXtimes")
    assert _register(hp, h5, commit) == 0
    f = fields(capsys.readouterr().out.strip().splitlines()[-1])
    assert (f["PARITY_ANCHOR"], f["cached"], f["sha_bad"], f["commit"]) == ("PASS", "3", "0", commit[:12])
    seg = json.loads((tmp_path / "anchors.json").read_text())["anchors"][TAG]["segments"]["native"]
    assert seg["rows"] == 3 and seg["worker"] == ["train_split_worker.run_one"] and seg["env_module_prefix"] == ["robomme_hard"]
    # 锚点不移动：重复登记拒绝
    with pytest.raises(hp.ParityError, match="已登记"):
        _register(hp, h5, commit)
    # 产物被改：逐局 sha 重算不符 → FAIL
    victim = next((h5 / "P-native").rglob("*.h5"))
    victim.write_bytes(b"changed")
    assert hp.main(["anchor", "check", "--tag", TAG, "--h5-root", str(h5)]) == 1
    f = fields(capsys.readouterr().out.strip())
    assert f["PARITY_ANCHOR"] == "FAIL" and f["sha_bad"] == "1"
    # 拉回目录不在：记缺目录
    shutil.rmtree(h5 / "P-native")
    ok, line, _ = hp.anchor_check(TAG, h5)
    assert not ok and "缺本机拉回目录" in line


def test_anchor_wrong_commit_unregistered_and_gpu_mix(hp, tmp_path, monkeypatch, capsys):
    _tag_commit()
    monkeypatch.setattr(hp, "ANCHORS", tmp_path / "anchors.json")
    h5 = tmp_path / "h5"
    _side(h5 / "P-native", "P", ROWS, worker="w")
    assert _register(hp, h5, "0" * 40) == 1  # 登记的 commit 与 tag 不符
    assert "tag 指向" in capsys.readouterr().out
    ok, line, _ = hp.anchor_check("no-such-tag", h5)
    assert not ok and line == "PARITY_ANCHOR=FAIL tag=no-such-tag reason=未登记"
    _side(h5 / "P-x0", "P", ROWS, worker="w", gpu="NVIDIA RTX 6000 Ada Generation")
    with pytest.raises(hp.ParityError, match="GPU／驱动不一致"):
        _register(hp, h5, "0" * 40, "native:P-native,xhard0:P-x0", "--replace")


def test_compare_with_p_anchor_and_binding(hp, tmp_path, monkeypatch, capsys):
    commit = _tag_commit()
    monkeypatch.setattr(hp, "ANCHORS", tmp_path / "anchors.json")
    h5 = tmp_path / "h5"
    _side(h5 / "P-native", "P", ROWS, worker="train_split_worker.run_one", module="robomme_hard.robomme_env.PickXtimes")
    _side(h5 / "H-native", "H", ROWS, module="robomme_hard.robomme_env.PickXtimes")
    _side(h5 / "O-native", "O", ROWS, worker="official._worker")
    assert _register(hp, h5, commit) == 0
    manifest = _native_manifest(tmp_path / "m.json", ROWS)
    capsys.readouterr()
    argv = ["compare", "--pair", "P:H", "--tier", "native", "--manifest", str(manifest), "--p-anchor", TAG,
            "--h5-root", str(h5), "--compare-root", str(tmp_path / "cmp"), "--workers", "2"]
    assert hp.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("PARITY_ANCHOR=PASS ")
    f = fields(next(t for t in out if t.startswith("PARITY_P_H=")))
    assert (f["PARITY_P_H"], f["binding_ok"], f["compared"]) == ("PASS", "3", "3")
    # 比对目录已存在：不给 --run-name 即拒；给了另起子目录
    with pytest.raises(hp.ParityError, match="--run-name"):
        hp.main(argv)
    assert hp.main(argv + ["--run-name", "r2"]) == 0 and (tmp_path / "cmp" / "PH-native" / "r2" / "summary.json").is_file()
    # 锚点没有该档的段
    xmanifest = _native_manifest(tmp_path / "x.json", [dict(r, tier="hard") for r in ROWS])
    with pytest.raises(hp.ParityError, match="没有 xhard0 段"):
        hp.main(["compare", "--pair", "P:H", "--tier", "xhard0", "--manifest", str(xmanifest), "--p-anchor", TAG,
                 "--h5-root", str(h5), "--compare-root", str(tmp_path / "cmpx"), "--workers", "2"])
    # 未登记锚点：判定行 FAIL、退出 1
    capsys.readouterr()
    assert hp.main(["compare", "--pair", "P:H", "--tier", "native", "--manifest", str(manifest), "--p-anchor", "no-such",
                    "--h5-root", str(h5), "--compare-root", str(tmp_path / "c3"), "--workers", "2"]) == 1
    assert capsys.readouterr().out.startswith("PARITY_ANCHOR=FAIL tag=no-such reason=未登记")
    # binding：三侧包归属
    assert hp.main(["binding", "--h5-root", str(h5), "--p-anchor", TAG]) == 0
    assert capsys.readouterr().out.strip() == \
        "ENV_PACKAGE_BINDING=PASS sides=3 O=robomme P=robomme_hard H=robomme_hard mismatch=0"
    # 负例：H 侧一局的环境模块不属于 robomme_hard
    lines = [json.loads(t) for t in (h5 / "H-native" / "identities.jsonl").read_text().splitlines()]
    lines[0]["env_module"] = "robomme.robomme_env.PickXtimes"
    F.write_jsonl(h5 / "H-native" / "identities.jsonl", lines)
    assert hp.main(["binding", "--h5-root", str(h5), "--p-anchor", TAG]) == 1
    assert capsys.readouterr().out.strip().endswith("mismatch=1")


def test_binding_rules_per_side(hp):
    o_ok = {"worker": "official._worker", "robomme_module": "/o/src/robomme/__init__.py"}
    assert hp._binding_ok("O", o_ok) and not hp._binding_ok("O", dict(o_ok, worker="train_split_worker.run_one"))
    assert not hp._binding_ok("O", dict(o_ok, robomme_module="/o/src/robomme_hard/robomme/__init__.py"))
    assert hp._binding_ok("H2", {"env_module": "robomme_hard.x"}) and not hp._binding_ok("H2", {"env_module": None})
    assert not hp._binding_ok("H", None)
    anchor = {"segments": {"native": {"worker": ["w"], "env_module_prefix": ["robomme_hard"]}}}
    assert hp._binding_ok("P", {"worker": "w", "env_module": "robomme_hard.a"}, anchor, "native")
    assert not hp._binding_ok("P", {"worker": "v", "env_module": "robomme_hard.a"}, anchor, "native")
    assert not hp._binding_ok("P", {"worker": "w", "env_module": "robomme.a"}, anchor, "native")


# ── compare：闸门、校准、超容差分类 ─────────────────────────────────────────────


def _pair_world(tmp: Path, *, right_offset_from: int | None = None, pair="O:H"):
    rows = ROWS
    tmp.mkdir(parents=True, exist_ok=True)
    left_side, right_side = pair.split(":")
    left = tmp / f"{left_side}-native"
    right = tmp / f"{right_side}-native"
    for side, root, off in ((left_side, left, None), (right_side, right, right_offset_from)):
        lines = []
        for r in rows:
            rel = f"episodes/{r['task']}_episode_{r['episode']}/hdf5_files/x.h5"
            F.write_h5(root / rel, seed=r["seed"] % 97, frames=4, joint_offset_from=off, offset=0.5)
            extra = ({"worker": "official._worker", "robomme_module": "/o/src/robomme/__init__.py"} if side in ("O", "P")
                     else {"env_module": "robomme_hard.robomme_env.PickXtimes"})
            lines.append(F.id_line(r["task"], r["tier"], r["seed"], sha=F.sha256(root / rel), path=rel, **extra))
        F.write_run(root, lines, workers=4, gpu="NVIDIA A40")
    manifest = _native_manifest(tmp / "m.json", rows)
    return ["compare", "--pair", pair, "--tier", "native", "--manifest", str(manifest), "--left", str(left),
            "--right", str(right), "--compare-root", str(tmp / "cmp"), "--workers", "2"]


@pytest.mark.parametrize("offset_from,cls", [(1, "noise"), (0, "fail")])
def test_compare_tolerance_over_classified_by_first_divergence(hp, tmp_path, capsys, offset_from, cls):
    """右侧从第 offset_from 帧起动作加 0.5（远超容差）：首个分叉步 >0 归噪声、=0 归失败；3 局全超 → 5% 硬线必中。"""
    assert hp.main(_pair_world(tmp_path, right_offset_from=offset_from)) == 1
    out = capsys.readouterr().out.splitlines()
    f = fields(next(t for t in out if t.startswith("PARITY_O_H=")))
    assert f["PARITY_O_H"] == "FAIL" and f["hard_line_5pct"] == "HIT" and f["over_total"] == "3"
    assert (f["noise"], f["tol_over"]) == (("3", "0") if cls == "noise" else ("0", "3"))
    assert f["sha_equal"] == "0" and f["identity_equal"] == "3"
    summary = json.loads((tmp_path / "cmp" / "OH-native" / "summary.json").read_text())
    assert {h["class"] for h in summary["tol_hits"]} == {cls}
    assert {h["first_divergence"] for h in summary["tol_hits"]} == {offset_from}
    ref = fields(next(t for t in out if t.startswith("PARITY_REFERENCE=")))
    assert ref["first_divergence_min"] == str(offset_from)


def test_compare_calibrate_writes_tolerance_file(hp, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hp, "TOLERANCES", tmp_path / "tol.json")
    argv = _pair_world(tmp_path, pair="O:P") + ["--calibrate"]
    assert hp.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    calib = fields(next(t for t in out if t.startswith("PARITY_TOL_CALIB=")))
    doc_text = (tmp_path / "tol.json").read_text()
    doc = json.loads(doc_text)
    import hashlib
    assert calib["PARITY_TOL_CALIB"] == "PASS" and calib["n"] == "3"
    assert calib["tol_file_sha"] == hashlib.sha256(doc_text.encode()).hexdigest()[:12]
    assert doc["raw_max"] == {"action_max": 0.0, "state_max": 0.0, "image_mad": 0.0, "frames_max": 0.0}
    assert doc["calibrated_from"] == "O:P native" and doc["n"] == 3
    assert fields(next(t for t in out if t.startswith("PARITY_O_P=")))["PARITY_O_P"] == "PASS"
    # 校准只许 O:P
    with pytest.raises(hp.ParityError, match="--calibrate 只允许 O:P"):
        hp.main(_pair_world(tmp_path / "b") + ["--calibrate"])


def test_compare_v9_requires_specs_root_and_cells_subset(hp, tmp_path):
    delivery = _v9_delivery(tmp_path / "d.json", V9_ROWS)
    with pytest.raises(hp.ParityError, match="须给 --specs-root"):
        hp.main(["compare", "--pair", "H:H2", "--tier", "v9", "--manifest", str(delivery)])
    with pytest.raises(hp.ParityError, match="与 --tier v9 不符|格表非法"):
        hp.compare_cells(types.SimpleNamespace(tier="v9", cells='{"NoSuchTask/xhard1": 1}'))


# ── import-delivery ───────────────────────────────────────────────────────


def _delivery_world(tmp: Path, *, bad_sha: bool = False, drop_path: bool = False):
    base = tmp / "gen1"
    rows = []
    for r in V9_ROWS:
        h5 = F.write_h5(base / "h5" / f"{r['seed']}.h5", seed=r["seed"] % 97, frames=2)
        row = {"task": r["task"], "tier": r["tier"], "episode": r["episode"], "seed": r["seed"], "candidate": r["episode"],
               "path": f"h5/{r['seed']}.h5", "h5_sha256": F.sha256(h5), "frames": 2, "env_module": "robomme_hard.x"}
        rows.append(row)
    if bad_sha:
        rows[0]["h5_sha256"] = "1" * 64
    if drop_path:
        rows[1].pop("path")
    path = base / "delivery.json"
    path.write_text(json.dumps({"schema": "v8-delivery/1", "rows": rows, "counts": {}}))
    (base / "launch-1.json").write_text("{}")
    return path


def test_import_delivery_symlinks_and_sha(hp, tmp_path, capsys):
    d = _delivery_world(tmp_path)
    h5_root = tmp_path / "h5"
    assert hp.main(["import-delivery", "--delivery", str(d), "--h5-root", str(h5_root)]) == 0
    assert capsys.readouterr().out.startswith("IMPORT_DELIVERY=PASS tier=v9 rows=3 sha_mismatch=0")
    out = h5_root / "H-v9"
    lines = [json.loads(t) for t in (out / "identities.jsonl").read_text().splitlines()]
    assert [l["seed"] for l in lines] == [9100, 9101, 9102] and all((out / l["path"]).is_symlink() for l in lines)
    assert (out / "launch-1.json").is_file()
    with pytest.raises(hp.ParityError, match="已存在"):
        hp.main(["import-delivery", "--delivery", str(d), "--h5-root", str(h5_root)])


@pytest.mark.parametrize("kw", [{"bad_sha": True}, {"drop_path": True}])
def test_import_delivery_bad_rows_counted(hp, tmp_path, capsys, kw):
    d = _delivery_world(tmp_path, **kw)
    assert hp.main(["import-delivery", "--delivery", str(d), "--h5-root", str(tmp_path / "h5")]) == 1
    assert capsys.readouterr().out.startswith("IMPORT_DELIVERY=FAIL tier=v9 ")


def test_import_delivery_gates(hp, tmp_path):
    d = _delivery_world(tmp_path)
    ids = F.write_jsonl(tmp_path / "ids.jsonl", [{"task": "MoveCube", "tier": "xhard4", "seed": 1}])
    with pytest.raises(hp.ParityError, match="交付清单之外"):
        hp.main(["import-delivery", "--delivery", str(d), "--identities", str(ids), "--h5-root", str(tmp_path / "a")])
    wrong = tmp_path / "w.json"
    wrong.write_text(json.dumps({"schema": "v7-delivery/1", "rows": []}))
    with pytest.raises(hp.ParityError, match="只接受 v8-delivery/1"):
        hp.main(["import-delivery", "--delivery", str(wrong), "--h5-root", str(tmp_path / "b")])


# ── 读取适配：交付清单、身份子集、格表、侧目录补齐 ─────────────────────────────────


def test_read_delivery_list_payload_and_cell_list(hp, tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"task": "A", "difficulty": "xhard1", "episode": 2, "seed": 5, "h5": "/abs/x.h5",
                              "sha256": "s"}]))
    d = hp.read_delivery(p)
    assert d["rows"][0]["tier"] == "xhard1" and d["rows"][0]["candidate"] == 2 and d["rows"][0]["h5_sha256"] == "s"
    assert hp.delivery_h5(d, d["rows"][0]) == Path("/abs/x.h5")
    assert d["counts"] == {k: None for k in hp.DELIVERY_COUNT_KEYS}  # 没显式写 → None（调用方据此判「未写零」）
    p.write_text(json.dumps({"schema": "v8-delivery/1", "totals": {"failed": 0}, "rows": [],
                             "cells": [{"task": "A", "difficulty": "xhard2", "n": 1}]}))
    d = hp.read_delivery(p)
    assert d["cells"] == {"A/xhard2": {"task": "A", "difficulty": "xhard2", "n": 1}} and d["counts"]["failed"] == 0
    assert hp.delivery_h5(d, {"path": None}) is None


def test_read_identity_subset_forms(hp, tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"delivered": [{"task": "A", "tier": "xhard1", "seed": 1, "source": "v8-reuse"},
                                           {"task": "B", "difficulty": "xhard2", "seed": "2", "source": "v9-new"}]}))
    assert hp.read_identity_subset(p) == {("B", "xhard2", 2)}  # 带 source 只取 v9-new
    p.write_text(json.dumps([{"task": "A", "tier": "xhard1", "seed": 1}]))
    assert hp.read_identity_subset(p) == {("A", "xhard1", 1)}
    p.write_text(json.dumps({"rows": [{"task": "A", "tier": "x", "seed": 1, "source": "v8-reuse"}]}))
    with pytest.raises(hp.ParityError, match="子集为空"):
        hp.read_identity_subset(p)


def test_parse_cells_forms_and_rejections(hp, tmp_path):
    hs = hp.hard_specs_light()
    (k1, n1), (k2, n2) = list(hs.V9_CELLS.items())[:2]
    want = {k1: 1, k2: n2}
    forms = [
        {f"{k1[0]}/{k1[1]}": 1, f"{k2[0]}@{k2[1]}": n2},
        {k1[0]: {k1[1]: 1}, **({k2[0]: {k2[1]: n2}} if k2[0] != k1[0] else {})},
        [[k1[0], k1[1], 1], [k2[0], k2[1], n2]],
        [{"task": k1[0], "tier": k1[1], "n": 1}, {"task": k2[0], "tier": k2[1], "count": n2}],
        {"cells": [[k1[0], k1[1], 1], [k2[0], k2[1], n2]]},
    ]
    if k2[0] == k1[0]:
        forms[1] = {k1[0]: {k1[1]: 1, k2[1]: n2}}
    for form in forms:
        assert hp.parse_cells_versioned(json.dumps(form), hs) == (want, "v9"), form
    f = tmp_path / "cells.json"
    f.write_text(json.dumps(forms[0]))
    assert hp.parse_cells(str(f), hs) == want  # 文件路径同样接受
    assert hp.parse_cells_versioned("v9full", hs) == (dict(hs.V9_CELLS), "v9")
    smoke, ver = hp.parse_cells_versioned("v9smoke", hs)
    assert ver == "v9" and all(k in hs.V9_CELLS and 0 < n <= hs.V9_CELLS[k] for k, n in smoke.items())
    shard, _ = hp.parse_cells_versioned("v9shard1", hs)
    assert shard and {k[0] for k in shard} == set(hp.V9_SHARD_TASKS["shard1"])
    assert sum(shard.values()) == sum(n for k, n in hs.V9_CELLS.items() if k[0] in hp.V9_SHARD_TASKS["shard1"])
    full, ver = hp.parse_cells_versioned(None, hs)
    assert full == dict(hs.EXPECTED_CELLS) and ver == hp.expected_version(hs)
    for bad in ("not-json{", json.dumps({f"{k1[0]}/{k1[1]}": n1 + 1}), json.dumps({"NoTask/xhard1": 1}),
                json.dumps({f"{k1[0]}/{k1[1]}": True}), json.dumps({})):
        with pytest.raises(hp.ParityError):
            hp.parse_cells_versioned(bad, hs)
    with pytest.raises(hp.ParityError, match="未登记的交付格表"):
        hp.table_version({k1: 1}, hs)


def test_root_cell_table_reads_headers(hp, tmp_path):
    hs = hp.hard_specs_light()
    assert hp.root_cell_table(tmp_path, hs) is None  # 空根
    tier = hs.TIERS[0]
    (tmp_path / tier).mkdir()
    task = next(k[0] for k in hs.V9_CELLS if k[1] == tier)
    (tmp_path / tier / "specs.jsonl").write_text(json.dumps({"schema": hs.SCHEMA, "difficulty": tier,
                                                             "delivery_per_cell": {task: 1}}) + "\n")
    tier2 = hs.TIERS[1]
    (tmp_path / tier2).mkdir()
    (tmp_path / tier2 / "specs.jsonl").write_text("坏的首行\n")  # 首行读不出：跳过
    assert hp.root_cell_table(tmp_path, hs) == ("v9", hs.CELL_TABLES["v9"])


def test_hard_specs_light_loads_file_without_package(hp, monkeypatch):
    """包已导入时复用包内模块；未导入时按文件单独加载一份（不经包 __init__，不连带导入仿真），且只加载一次。"""
    name = "robomme_hard.env_record_wrapper.hard_specs"
    pkg = sys.modules.get(name)
    if pkg is not None:
        assert hp.hard_specs_light() is pkg
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(hp._common, "_HARD_SPECS_LIGHT", None)
    light = hp.hard_specs_light()
    assert light is not pkg and light.__name__ == "_hard_specs_light"
    assert name not in sys.modules and hp.hard_specs_light() is light
    if pkg is not None:
        assert light.V9_CELLS == pkg.V9_CELLS


def test_side_lines_fill_robomme_module(hp, tmp_path):
    """有 _runner/results.json：只补空缺不覆盖；无（bucket 拉回）：按 launch 的 src_root 推出，只补官方 _worker 的行。"""
    a = tmp_path / "a"
    F.write_jsonl(a / "identities.jsonl", [{"task": "T", "tier": "e", "seed": 1, "robomme_module": "keep"},
                                           {"task": "T", "tier": "e", "seed": 2}])
    (a / "_runner").mkdir()
    (a / "_runner" / "results.json").write_text(json.dumps({"robomme_module": "probe", "worker": "official._worker"}))
    got = hp.side_lines(a)
    assert [g["robomme_module"] for g in got] == ["keep", "probe"] and got[1]["worker"] == "official._worker"
    b = tmp_path / "b"
    F.write_jsonl(b / "identities.jsonl", [{"task": "T", "tier": "e", "seed": 1, "worker": "official._worker"},
                                           {"task": "T", "tier": "e", "seed": 2, "worker": "other"}])
    (b / "launch-1.json").write_text(json.dumps({"env_package": "robomme", "src_root": "/o"}))
    got = hp.side_lines(b)
    assert got[0]["robomme_module"] == "/o/src/robomme/__init__.py" and got[0]["robomme_module_source"] == "launch"
    assert got[1].get("robomme_module") is None


# ── export-xhard0-manifest ────────────────────────────────────────────────


def test_export_xhard0_manifest_cross_checks_builder(hp, tmp_path, monkeypatch, capsys):
    bench = F.bench_root()
    rows, _ = hp.xhard0_records(bench)
    builder = [{"task": r["task"], "episode": r["episode"], "seed": r["seed"]} for r in rows]
    monkeypatch.setattr(hp, "_builder_xhard0_rows", lambda: builder)
    out = tmp_path / "x0.json"
    assert hp.main(["export-xhard0-manifest", "--src-root", str(bench), "--out", str(out)]) == 0
    f = fields(capsys.readouterr().out.strip())
    assert (f["XHARD0_IDENTITY"], f["identities"], f["missing"], f["extra"]) == ("PASS", str(len(rows)), "0", "0")
    m = json.loads(out.read_text())
    assert m["rows_total"] == len(rows) and sum(m["recovery_config_counts"].values()) == len(rows)
    # builder 少一条：交叉核对 FAIL
    monkeypatch.setattr(hp, "_builder_xhard0_rows", lambda: builder[1:])
    assert hp.main(["export-xhard0-manifest", "--src-root", str(bench), "--out", str(out)]) == 1
    f = fields(capsys.readouterr().out.strip().split(" problems=")[0])
    assert (f["XHARD0_IDENTITY"], f["missing"], f["extra"]) == ("FAIL", "1", "0")
