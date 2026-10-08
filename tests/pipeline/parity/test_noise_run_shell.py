"""C16 壳脚本 ``scripts/parity/noise_run_gl.sh``：真 bash 子进程 + 假真实命令（``true``／``false``／``sleep``）。

标 ``slow``（起 bash、等信号），不进日常门禁。核：preflight 不过不执行真实命令也不写 finish；真实命令退出码进
``EXIT_CODE=`` 尾行与账本 finish 行；导出的线程环境变量；TERM 中断时收掉真实命令仍写 finish 与 ``EXIT_CODE=143``；
``--kind`` 只认 gen；``bash -n`` 语法。
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import parity_fixtures as F

pytestmark = pytest.mark.slow

WRAPPER = F.REPO / "scripts" / "parity" / "noise_run_gl.sh"
CAPS = {"gen": {"attempts": 10, "resets": 30, "retries": 2}, "total": {"attempts": 15, "resets": 40, "retries": 3}}


@pytest.fixture()
def env(tmp_path):
    fake = tmp_path / "fake-nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'NVIDIA A40, 595.71.05, GPU-aaaa'\n")
    fake.chmod(0o755)
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps(CAPS))
    return {"tmp": tmp_path, "caps": caps, "ledger": tmp_path / "ledger" / "budget.jsonl", "smi": fake}


def wrapper_cmd(env, name, real_cmd, *, out_root=None, kind="gen", extra=()):
    log = env["tmp"] / "logs" / f"{name}.log"
    prov = env["tmp"] / "prov" / f"{name}.json"
    out_root = out_root or env["tmp"] / "runs" / name
    cmd = ["bash", str(WRAPPER), "--pass", name, "--kind", kind, "--out-root", str(out_root), "--log", str(log),
           "--provenance-out", str(prov), "--budget-ledger", str(env["ledger"]), "--budget-caps", str(env["caps"]),
           "--attempts", "1", "--resets", "3", "--retries", "0", *extra, "--", *real_cmd]
    e = dict(os.environ, PY=sys.executable, NOISE_RUN_NVIDIA_SMI=str(env["smi"]))
    return cmd, e, log, prov


def run(env, name, real_cmd, **kw):
    cmd, e, log, prov = wrapper_cmd(env, name, real_cmd, **kw)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=120)
    return proc, log, prov


def ledger(env):
    return [json.loads(t) for t in env["ledger"].read_text().splitlines()]


@pytest.mark.parametrize("real,code", [(["true"], 0), (["false"], 1)])
def test_端到端写EXIT_CODE与finish行(env, real, code):
    proc, log, prov = run(env, f"w{code}", real, extra=("--actual-attempts", "1"))
    assert proc.returncode == code, proc.stdout + proc.stderr
    lines = log.read_text().splitlines()
    assert lines[-1] == f"EXIT_CODE={code}"
    assert any(ln.startswith("NOISE_PREFLIGHT=PASS") for ln in lines)
    assert any(ln.startswith(f"NOISE_FINISH=PASS pass=w{code} exit_code={code}") for ln in lines)
    fin = [r for r in ledger(env) if r["event"] == "finish"]
    assert len(fin) == 1 and fin[0]["exit_code"] == code and fin[0]["attempts"] == 1
    rep = json.loads(prov.read_text())
    assert rep["exit_code"] == code and rep["real_command"].strip() == " ".join(real)


def test_真实命令输出进日志且线程变量已导出(env):
    proc, log, _ = run(env, "wenv", ["bash", "-c", 'echo "HELLO omp=$OMP_NUM_THREADS mkl=$MKL_NUM_THREADS"'])
    assert proc.returncode == 0
    assert "HELLO omp=1 mkl=1" in log.read_text()


def test_preflight不过不执行真实命令且不写finish(env):
    root = env["tmp"] / "used"
    root.mkdir()
    (root / "x").write_text("1")
    marker = env["tmp"] / "ran"
    proc, log, prov = run(env, "wbad", ["touch", str(marker)], out_root=root)
    nr = F.noise_run()
    assert proc.returncode == nr.EXIT_FRESH
    text = log.read_text()
    assert "RUN_FRESH=FAIL reason=out_root_not_empty" in text
    assert text.splitlines()[-1] == f"EXIT_CODE={nr.EXIT_FRESH}"
    assert not marker.exists() and not prov.exists() and not env["ledger"].exists()


def test_kind只认gen且缺真实命令即拒(env):
    proc, _log, _prov = run(env, "wk", ["true"], kind="eval")
    assert proc.returncode == 2 and "EXIT_CODE=2" in proc.stdout
    cmd, e, _log, _prov = wrapper_cmd(env, "wn", [])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=60)
    assert proc.returncode == 2 and not env["ledger"].exists()


def test_TERM中断时收掉真实命令_照写finish与EXIT_CODE(env):
    pidfile = env["tmp"] / "child.pid"
    code = f"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));print('begin',flush=True);time.sleep(60)"
    cmd, e, log, _prov = wrapper_cmd(env, "wsig", [sys.executable, "-c", code])
    proc = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 30
    while not (log.exists() and "begin" in log.read_text()):
        assert time.time() < deadline and proc.poll() is None
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode == 143
    lines = log.read_text().splitlines()
    assert lines[-1] == "EXIT_CODE=143"
    assert any(ln.startswith("NOISE_FINISH=PASS pass=wsig exit_code=143") for ln in lines)
    child = int(pidfile.read_text())
    deadline = time.time() + 5
    while Path(f"/proc/{child}").exists() and "zombie" not in Path(f"/proc/{child}/status").read_text().lower():
        assert time.time() < deadline, "真实命令没有被收掉"
        time.sleep(0.1)


def test_bash语法检查():
    assert subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0
