#!/usr/bin/env python3
"""run_shard.sh 夹具用的桩 scripts/evaluate.py：按 FIXTURE_SCENARIO 模拟真实入口的几种退出路径。

只用标准库，不跑模型、不跑仿真、不连服务端；由 shard_fixture.sh 复制到临时目录的
code/scripts/evaluate.py，经 SHARD_CODE 覆盖让 run_shard.sh 调用。
每局只写一个 <Task>_ep0_xhard1 目录：有 result.json = 已完成局；只有 trace.jsonl = 半截局。

环境变量：
  FIXTURE_SCENARIO  ok | wall75 | envbuild | infra2 | cap | noprog | term
  FIXTURE_CNT       调用计数文件（跨模型、跨尝试累加，用于「第 N 次调用」类场景）
  FIXTURE_PIDFILE   term 场景下写入本进程 pid，供外部核对 TERM 后子进程已收
"""
import json
import os
import pathlib
import sys
import time

argv = sys.argv[1:]


def opt(name):
    return argv[argv.index(name) + 1] if name in argv else None


tasks = opt("--tasks").split(",")
out = pathlib.Path(opt("--out"))
model = opt("--model")
variant = opt("--groundsg-variant")
label = f"groundsg-{variant}" if variant else model
raw = out / "rollouts" / label / "ood" / "seed7" / "raw"
raw.mkdir(parents=True, exist_ok=True)

scenario = os.environ.get("FIXTURE_SCENARIO", "ok")
cnt = pathlib.Path(os.environ["FIXTURE_CNT"])
n = int(cnt.read_text() or 0) if cnt.exists() else 0
cnt.write_text(str(n + 1))


def write(task, infra_reason=None):
    """写一局完整结果（infra_reason 非空即 infra 局）。"""
    d = raw / f"{task}_ep0_xhard1"
    d.mkdir(exist_ok=True)
    (d / "result.json").write_text(json.dumps({
        "task": task, "infra": bool(infra_reason), "infra_reason": infra_reason,
        "status": "fail" if infra_reason else "success"}))


def half(task):
    """留下一个没有 result.json 的半截局目录（模拟进程在局中被打断）。"""
    d = raw / f"{task}_ep0_xhard1"
    d.mkdir(exist_ok=True)
    (d / "trace.jsonl").write_text("x\n")


if scenario == "term" and label == "pp":
    pathlib.Path(os.environ["FIXTURE_PIDFILE"]).write_text(str(os.getpid()))
    print(f"STUB_SLEEP pid={os.getpid()}", flush=True)
    time.sleep(60)
    sys.exit(0)

for t in tasks:
    if (raw / f"{t}_ep0_xhard1" / "result.json").exists():
        print(f"EPISODE_SKIP task={t} (result.json exists)", flush=True)
        continue
    if scenario == "wall75" and n == 0 and label == "groundsg-ground-sg-oracle":
        half(t)  # 首次调用：墙钟超时硬退出，当前局成半截
        print("SERVER_LEFT", flush=True)
        os._exit(75)
    if scenario == "envbuild" and label == "pp":
        write(t, "env_build")
        time.sleep(0.2)
        continue
    if scenario == "infra2" and label == "pp":
        write(t, "recorder")
        time.sleep(0.2)
        continue
    if scenario == "cap" and label in ("pp", "smvla"):
        half(t)  # 每次调用都在首局被打断（模拟服务端反复死），半截局累计直到超封顶
        os._exit(3)
    if scenario == "noprog" and label == "pp" and n <= 2:
        time.sleep(20)  # 静默空转：活着但不写任何文件，等看门杀
        os._exit(1)
    write(t)
print("EVAL_DONE", flush=True)
sys.exit(0)
