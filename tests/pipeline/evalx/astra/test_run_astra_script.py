"""S5 新侧启动器 ``run_astra.sh``：语法、照抄项静态核对、三项拒绝行为（不起任何服务、不读密钥文件）。"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from astra_fakes import third_party

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "eval-official" / "run_astra.sh"
SENTINEL = "sk-test-sentinel-not-a-real-key"


def test_bash_syntax():
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_copied_items_and_differences_static():
    """逐项照抄 run.sh 的环境变量与 VLA 启动命令；PYTHONPATH 只换第四段；无看门狗；不读密钥文件。"""
    text = SCRIPT.read_text()
    upstream = (third_party() / "examples" / "champ" / "run.sh").read_text()
    copied = [
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1",
        "export TOKENIZERS_PARALLELISM=false USE_HF=1 IMAGE_MAX_TOKEN_NUM=128",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        'env -u OPENAI_API_KEY CUDA_VISIBLE_DEVICES="$VLA_GPU" "$VLA_PYTHON" scripts/serve_policy.py',
        "[[ \"$VLA_GPU\" != \"$MONITOR_GPU\" ]] || { echo 'Use distinct GPUs for VLA and monitor' >&2; exit 2; }",
        "[[ ! -e \"$RUN\" ]] || { echo 'Use a new run directory to preserve existing evidence' >&2; exit 2; }",
        'VLA_GPU=${VLA_GPU:-0}', 'MONITOR_GPU=${MONITOR_GPU:-1}', 'PORT=${PORT:-18762}',
    ]
    for line in copied:
        assert line in upstream, f"上游已变：{line}"
        assert line in text, f"未照抄：{line}"
    # 第三阶段唯一改动：服务 --seed 由上游固定 42 换成 --policy-seed 的值（R5）
    up_seed = '--port="$PORT" --seed=42 policy:checkpoint --policy.config=mme_vla_suite'
    assert up_seed in upstream, "上游已变：VLA 启动命令"
    assert up_seed.replace("--seed=42", '--seed="$POLICY_SEED"') in text
    assert "--seed=42" not in text
    up_pp = re.search(r'export PYTHONPATH="([^"]+)"', upstream).group(1).split(":")
    new_pp = re.search(r'export PYTHONPATH="([^"]+)"', text).group(1).split(":")
    assert [p.replace("$REPO", "$ASTRA") for p in up_pp[:3]] == new_pp[:3]
    assert up_pp[3] == "$REPO/third_party/robomme_benchmark/src" and new_pp[3] == "$REPO/src"
    assert "NO_PROXY=127.0.0.1,localhost" in text
    assert "NO_PROGRESS" not in text and "watchdog" not in text.lower()
    assert "openai.key" not in text and "--key-file" not in text
    assert not re.search(r"echo[^\n]*OPENAI_API_KEY", text)
    assert "trap cleanup EXIT" in text and " check --cases " in text


def _env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.update(VLA_PYTHON=sys.executable, SIM_PYTHON=sys.executable, VLA_CHECKPOINT=str(tmp_path / "vla"),
               MONITOR_BASE=str(tmp_path / "base"), MONITOR_ADAPTER=str(tmp_path / "adapter"), MAX_STEPS="1300",
               OPENAI_API_KEY=SENTINEL, SGEVAL_THIRD_PARTY=str(third_party().parent))
    env.update(extra)
    return env


def _run(tmp_path, run_dir, **extra):
    cases = tmp_path / "cases.json"
    cases.write_text('{"dataset": "hard-verify", "cases": []}')
    proc = subprocess.run(["bash", str(SCRIPT), "--policy-seed", "7", str(cases), str(run_dir)], capture_output=True,
                          text=True, env=_env(tmp_path, **extra), timeout=60)
    assert SENTINEL not in proc.stdout + proc.stderr, "密钥不得出现在输出里"
    return proc


@pytest.mark.slow
def test_same_gpu_exits_2(tmp_path):
    run_dir = tmp_path / "group_0" / "run"
    proc = _run(tmp_path, run_dir, VLA_GPU="0", MONITOR_GPU="0")
    assert proc.returncode == 2 and "Use distinct GPUs" in proc.stderr
    assert not run_dir.exists()


@pytest.mark.slow
def test_existing_run_dir_refused(tmp_path):
    run_dir = tmp_path / "group_0" / "run"
    run_dir.mkdir(parents=True)
    proc = _run(tmp_path, run_dir)
    assert proc.returncode == 2 and "Use a new run directory" in proc.stderr
    assert list(run_dir.iterdir()) == []


@pytest.mark.slow
def test_run_dir_outside_group_refused(tmp_path):
    run_dir = tmp_path / "elsewhere" / "run"
    proc = _run(tmp_path, run_dir)
    assert proc.returncode == 2 and "reason=layout" in proc.stderr
    assert not run_dir.exists()


@pytest.mark.slow
def test_missing_key_refused_without_leaking(tmp_path):
    run_dir = tmp_path / "group_0" / "run"
    proc = _run(tmp_path, run_dir, OPENAI_API_KEY="")
    assert proc.returncode != 0 and "OPENAI_API_KEY" in proc.stderr and not run_dir.exists()
