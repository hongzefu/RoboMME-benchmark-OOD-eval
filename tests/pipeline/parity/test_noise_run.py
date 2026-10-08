"""C15／C16 噪声基线单遍运行包装的 Python 逻辑 ``scripts/parity/noise_run.py``：``preflight``（RUN_FRESH、资产、
BUDGET、来源报告）与 ``finish``。``ship`` 的贯通在 ``test_mover_ship_pull.py``；壳脚本在 ``test_noise_run_shell.py``（slow）。

全部在临时目录：nvidia-smi 用 ``NOISE_RUN_NVIDIA_SMI`` 指向假脚本（不碰真实 GPU）；预算上限是本用例自定的小数，
不是业务常量。账本里历史的 ``eval``／``digest`` 行（噪声基线阶段登记过）照常计入 total。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import parity_fixtures as F

CAPS = {"gen": {"attempts": 10, "resets": 30, "retries": 2},
        "total": {"attempts": 15, "resets": 40, "retries": 3}}


@pytest.fixture(scope="module")
def nr():
    return F.noise_run()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    fake = tmp_path / "fake-nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'NVIDIA A40, 595.71.05, GPU-aaaa-bbbb'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("NOISE_RUN_NVIDIA_SMI", str(fake))
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps(CAPS))
    return {"tmp": tmp_path, "caps": caps, "ledger": tmp_path / "ledger" / "budget.jsonl", "smi": fake}


def preflight(nr, env, name, *, kind="gen", attempts=2, resets=6, retries=0, out_root=None, extra=()):
    out_root = out_root or env["tmp"] / "runs" / name
    prov = env["tmp"] / "prov" / f"{name}.json"
    argv = ["preflight", "--pass", name, "--kind", kind, "--out-root", str(out_root),
            "--budget-ledger", str(env["ledger"]), "--budget-caps", str(env["caps"]),
            "--attempts", str(attempts), "--resets", str(resets), "--retries", str(retries),
            "--max-steps", "777", "--server-args", "--port 8000", "--provenance-out", str(prov), *extra]
    return nr.main(argv), prov


def ledger_rows(env):
    return [json.loads(t) for t in env["ledger"].read_text().splitlines()]


# ── RUN_FRESH ───────────────────────────────────────────────────────────────────


def test_输出根不存在或为空目录可跑(nr, env, capsys):
    rc, prov = preflight(nr, env, "p1")
    assert rc == 0 and "RUN_FRESH=PASS" in capsys.readouterr().out
    assert json.loads(prov.read_text())["out_root_state"] == "absent"
    empty = env["tmp"] / "empty"
    empty.mkdir()
    rc, prov = preflight(nr, env, "p2", out_root=empty)
    assert rc == 0 and json.loads(prov.read_text())["out_root_state"] == "empty_dir"


@pytest.mark.parametrize("make", ["nonempty", "file"])
def test_输出根非空或是文件即拒_且不占预算(nr, env, capsys, make):
    root = env["tmp"] / "used"
    if make == "nonempty":
        root.mkdir()
        (root / "identities.jsonl").write_text("{}\n")
    else:
        root.write_text("x")
    rc, prov = preflight(nr, env, "p1", out_root=root)
    out = capsys.readouterr().out
    assert rc == nr.EXIT_FRESH and "RUN_FRESH=FAIL reason=out_root_not_empty" in out
    assert out.strip().splitlines()[-1].startswith("NOISE_PREFLIGHT=FAIL pass=p1 kind=gen")
    assert not prov.exists() and not env["ledger"].exists()


def test_kind只认gen(nr, env):
    with pytest.raises(SystemExit):
        preflight(nr, env, "e1", kind="eval")
    assert not env["ledger"].exists()


# ── BUDGET ──────────────────────────────────────────────────────────────────────


def test_预算按kind与total累加_恰好到线可跑_超一即拒(nr, env, capsys):
    assert preflight(nr, env, "g1", attempts=5, resets=15)[0] == 0
    assert preflight(nr, env, "g2", attempts=5, resets=15)[0] == 0  # gen 恰好 10／30
    rc, prov = preflight(nr, env, "g3", attempts=1, resets=0)
    out = capsys.readouterr().out
    assert rc == nr.EXIT_BUDGET and "BUDGET=FAIL reason=over_cap" in out and "gen.attempts=11>10" in out
    assert not prov.exists()
    rows = ledger_rows(env)
    assert [r["pass"] for r in rows] == ["g1", "g2"]
    assert all(set(r) == {"t", "pass", "kind", "attempts", "resets", "retries", "event"} for r in rows)


def test_账本历史eval与digest行计入total(nr, env, capsys):
    env["ledger"].parent.mkdir(parents=True)
    hist = [{"t": 1.0, "pass": "ev-old", "kind": "eval", "attempts": 6, "resets": 6, "retries": 0, "event": "reserve"},
            {"t": 2.0, "pass": "ev-old", "kind": "eval", "attempts": 8, "resets": 6, "retries": 0, "event": "finish"},
            {"t": 3.0, "pass": "dg-old", "kind": "digest", "attempts": 2, "resets": 2, "retries": 0,
             "event": "reserve"}]
    env["ledger"].write_text("".join(json.dumps(r) + "\n" for r in hist))
    # 历史 total = eval 取较大的实际数 8 + digest 2 = 10；gen 再要 6 → total 16 > 15
    rc, _ = preflight(nr, env, "g1", attempts=6, resets=0)
    out = capsys.readouterr().out
    assert rc == nr.EXIT_BUDGET and "total.attempts=16>15" in out
    rc, _ = preflight(nr, env, "g2", attempts=5, resets=0)
    out = capsys.readouterr().out
    assert rc == 0 and "total_attempts_after=15/15" in out


def test_同名pass拒(nr, env, capsys):
    rc, prov = preflight(nr, env, "dup")
    assert rc == 0
    prov.unlink()  # 绕过「来源报告已存在」那道检查，专测账本同名拒绝
    rc, _ = preflight(nr, env, "dup", out_root=env["tmp"] / "other")
    assert rc == nr.EXIT_BUDGET and "reason=duplicate_pass" in capsys.readouterr().out


def test_finish实际数大于登记时按实际计费(nr, env, capsys):
    rc, prov = preflight(nr, env, "g1", attempts=2, resets=6)
    assert rc == 0
    assert nr.main(["finish", "--pass", "g1", "--provenance", str(prov), "--exit-code", "0",
                    "--actual-attempts", "9", "--actual-resets", "unknown"]) == nr.EXIT_OVERRUN
    assert "BUDGET=FAIL reason=actual_exceeds_reserved" in capsys.readouterr().out
    assert preflight(nr, env, "g2", attempts=2, resets=0)[0] == nr.EXIT_BUDGET  # 已计费 9，再要 2 超 10


def test_账本fsync_损坏_缺上限(nr, env, capsys, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(nr.os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    assert preflight(nr, env, "g1")[0] == 0 and calls
    env["ledger"].write_text('{"pass": "x", "event": "reserve"\n')
    assert preflight(nr, env, "g2", out_root=env["tmp"] / "r2")[0] == nr.EXIT_BUDGET
    assert "ledger_corrupt" in capsys.readouterr().out
    env["ledger"].write_text("")
    env["caps"].write_text(json.dumps({"gen": CAPS["gen"]}))
    assert preflight(nr, env, "g3", out_root=env["tmp"] / "r3")[0] == nr.EXIT_BUDGET
    assert "cap_missing scope=total" in capsys.readouterr().out


# ── 资产 ────────────────────────────────────────────────────────────────────────


def make_asset(root: Path) -> dict[str, str]:
    (root / "sub").mkdir(parents=True)
    (root / "config.json").write_bytes(b'{"a": 1}')
    (root / "sub" / "w.bin").write_bytes(b"\x00\x01weights")
    return {"./config.json": sha(b'{"a": 1}'), "./sub/w.bin": sha(b"\x00\x01weights")}


def write_lock(path: Path, name: str, files: dict[str, str]) -> Path:
    path.write_text(json.dumps({"assets": {name: {"files": files}}}))
    return path


def test_资产相同PASS_排除顶层cache与DS_Store(nr, env, capsys):
    root = env["tmp"] / "ckpt"
    files = make_asset(root)
    (root / ".cache" / "hf").mkdir(parents=True)
    (root / ".cache" / "hf" / "x").write_text("m")
    (root / "sub" / ".DS_Store").write_text("d")
    (root / "sub" / ".cache").mkdir()
    (root / "sub" / ".cache" / "keep").write_bytes(b"k")  # 非顶层 .cache 不排除
    files["./sub/.cache/keep"] = sha(b"k")
    lock = write_lock(env["tmp"] / "lock.json", "smvla", files)
    rc, prov = preflight(nr, env, "g1", extra=("--assets-lock", str(lock), "--asset-dir", f"smvla={root}"))
    out = capsys.readouterr().out
    assert rc == 0 and "ASSET name=smvla files=3 locked=3 missing=0 extra=0 mismatch=0" in out and "assets=PASS" in out
    a = json.loads(prov.read_text())["assets_sha"]["smvla"]
    assert a["verdict"] == "PASS" and a["files_sha256"] == a["locked_files_sha256"]


@pytest.mark.parametrize("change,field", [("missing", "missing=1"), ("extra", "extra=1"), ("byte", "mismatch=1"),
                                          ("not_in_lock", "error=not_in_lock")])
def test_资产不符即FAIL且不占预算(nr, env, capsys, change, field):
    root = env["tmp"] / "ckpt"
    files = make_asset(root)
    lock = write_lock(env["tmp"] / "lock.json", "other" if change == "not_in_lock" else "smvla", files)
    if change == "missing":
        (root / "sub" / "w.bin").unlink()
    elif change == "extra":
        (root / "new.bin").write_bytes(b"x")
    elif change == "byte":
        (root / "sub" / "w.bin").write_bytes(b"\x00\x02weights")
    rc, prov = preflight(nr, env, "g1", extra=("--assets-lock", str(lock), "--asset-dir", f"smvla={root}"))
    out = capsys.readouterr().out
    assert rc == nr.EXIT_ASSETS and field in out and "reason=assets" in out
    assert not prov.exists() and not env["ledger"].exists()


def test_tokenizer单独核sha(nr, env, capsys):
    tok = env["tmp"] / "tokenizer.model"
    tok.write_bytes(b"tok")
    rc, prov = preflight(nr, env, "g1", extra=("--tokenizer", str(tok), "--tokenizer-sha256", sha(b"tok")))
    assert rc == 0 and "match=1" in capsys.readouterr().out
    rc, _ = preflight(nr, env, "g2", extra=("--tokenizer", str(tok), "--tokenizer-sha256", sha(b"zz")))
    assert rc == nr.EXIT_ASSETS


def test_资产过滤规则与真实find一致(nr, env):
    root = env["tmp"] / "ckpt"
    make_asset(root)
    (root / ".cache" / "a").mkdir(parents=True)
    (root / ".cache" / "a" / "b").write_text("x")
    (root / "sub" / ".DS_Store").write_text("d")
    (root / "lnk").symlink_to(root / "config.json")
    (root / "dangling").symlink_to(env["tmp"] / "nope")
    out = subprocess.run(["find", "-L", ".", "-type", "f", "!", "-path", "./.cache/*", "!", "-name", ".DS_Store"],
                         cwd=root, capture_output=True, text=True, check=True).stdout.split()
    assert sorted(out) == sorted(nr.list_asset_files(root))


# ── 来源报告与 finish ─────────────────────────────────────────────────────────────


def test_来源报告指纹可重算_finish后不变_重复finish拒(nr, env, capsys, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    rc, prov = preflight(nr, env, "g1")
    out = capsys.readouterr().out
    assert rc == 0
    rep = json.loads(prov.read_text())
    assert (rep["gpu_model"], rep["driver"], rep["gpu_uuid"]) == ("NVIDIA A40", "595.71.05", "GPU-aaaa-bbbb")
    assert rep["slurm_job_id"] == "12345" and rep["max_steps"] == 777 and rep["server_args"] == "--port 8000"
    assert rep["budget"]["reserved"] == {"attempts": 2, "resets": 6, "retries": 0}
    body = {k: v for k, v in rep.items() if k != "fingerprint"}
    want = sha(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert rep["fingerprint"] == want
    assert out.strip().splitlines()[-1] == f"NOISE_PREFLIGHT=PASS pass=g1 kind=gen assets=SKIP fingerprint={want[:12]}"
    env_keys = ("commit", "git_dirty", "policy_commits", "assets_sha", "tokenizer_sha256", "driver", "gpu_model",
                "max_steps", "server_args")
    assert rep["env_fingerprint"] == sha(json.dumps({k: rep.get(k) for k in env_keys}, ensure_ascii=False,
                                                    sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert nr.main(["finish", "--pass", "g1", "--provenance", str(prov), "--exit-code", "1",
                    "--actual-attempts", "2", "--actual-resets", "unknown"]) == 0
    rep2 = json.loads(prov.read_text())
    assert rep2["exit_code"] == 1 and rep2["actual_attempts"] == 2 and rep2["actual_resets"] is None
    assert rep2["fingerprint"] == want == nr.fingerprint_of(rep2)
    fin = [r for r in ledger_rows(env) if r["event"] == "finish"]
    assert len(fin) == 1 and fin[0]["attempts"] == 2 and fin[0]["resets"] is None
    assert nr.main(["finish", "--pass", "g1", "--provenance", str(prov), "--exit-code", "0",
                    "--actual-attempts", "2", "--actual-resets", "0"]) != 0
    assert nr.main(["finish", "--pass", "zz", "--provenance", str(prov), "--exit-code", "0",
                    "--actual-attempts", "1", "--actual-resets", "1"]) == nr.EXIT_USAGE


def test_nvidia_smi不可用时为null_来源报告已存在即拒(nr, env, monkeypatch):
    monkeypatch.setenv("NOISE_RUN_NVIDIA_SMI", str(env["tmp"] / "no-such-smi"))
    rc, prov = preflight(nr, env, "g1")
    assert rc == 0
    rep = json.loads(prov.read_text())
    assert rep["driver"] is None and rep["gpu_model"] is None and rep["gpu_uuid"] is None
    assert preflight(nr, env, "g1", out_root=env["tmp"] / "x")[0] == nr.EXIT_FRESH


@pytest.mark.parametrize("value", ["-1", "x"])
def test_finish实际数非法即拒(nr, env, value):
    rc, prov = preflight(nr, env, "g1")
    assert nr.main(["finish", "--pass", "g1", "--provenance", str(prov), "--exit-code", "0",
                    "--actual-attempts", value, "--actual-resets", "0"]) == nr.EXIT_USAGE
