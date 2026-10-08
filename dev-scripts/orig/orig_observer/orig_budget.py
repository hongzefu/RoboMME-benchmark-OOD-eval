#!/usr/bin/env python3
"""原侧启动器的逐局预算预约（计划第二部分一节 S7、R6、R11；调用 S8 ``budget_ledger.py`` CLI，只读其接口）。

原版评估在一个进程里逐局跑完一片，启动器无法在每局开头插入调用，所以按「遍」结算：

- ``prepare``（每遍评估起跑前）：本片尚无终态行（success／fail／timeout）的每个身份——若上一遍留有未结的 rid，先
  ``release``（重跑不把同一局重复计为多条轨迹），再 ``reserve --resets <n> --route <route> --key <key>`` 取新 rid；
  任一 ``reserve`` 失败即打印 ``RUN_BLOCKED reason=budget`` 并以 5 退出（已预约的保留在状态文件里，下次照样先 release）。
- ``settle``（每遍评估结束后）：已有终态行且 rid 未结的身份 ``commit --id <rid>``。

账本 CLI：``BUDGET_LEDGER_CMD``（空格切分）或缺省 ``<python> <eval-official>/budget_ledger.py``；顶层参数
``BUDGET_LEDGER_ARGS``（如 ``--ledger L``）放在子命令**之前**；不给 ``--ledger`` 时由账本脚本自己读环境变量
``SGEVAL_BUDGET_LEDGER``。rid 取 ``reserve`` 标准输出末行的 ``rid=<rid>``。

状态文件（``--state``，JSON）：``{key: {"rid", "state": reserved|committed|released, "history": [...]}}``，原子写。
末行：``BUDGET_PREPARE episodes=<n> reserved=<n> released=<n>`` 或 ``BUDGET_SETTLE committed=<n> open=<n>``。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
FINAL = ("success", "fail", "timeout")
EXIT_BUDGET = 5


def ledger_cmd() -> list[str]:
    cmd = os.environ.get("BUDGET_LEDGER_CMD", "").strip()
    if cmd:
        return shlex.split(cmd)
    path = EVAL_DIR / "budget_ledger.py"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return [os.environ.get("BUDGET_PY") or sys.executable, str(path)]


def call(sub: list[str]) -> subprocess.CompletedProcess:
    top = shlex.split(os.environ.get("BUDGET_LEDGER_ARGS", ""))
    return subprocess.run(ledger_cmd() + top + sub, capture_output=True, text=True)


def identities(manifest: str, shard: int, only_tasks: str | None) -> list[tuple[str, int, int]]:
    want = set(only_tasks.split(",")) if only_tasks else None
    out = []
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["shard"]) != int(shard) or (want is not None and r["task"] not in want):
            continue
        out.append((r["task"], int(r["source_episode"]), int(r["seed"])))
    return sorted(out)


def finished(log: str) -> set[tuple[str, int]]:
    done = set()
    p = Path(log)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") in FINAL:
                    done.add((r["task"], int(r["source_episode"])))
    return done


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _rid_of(stdout: str) -> str | None:
    lines = [x for x in stdout.splitlines() if x.strip()]
    if not lines:
        return None
    last = lines[-1].split()[-1]
    return last[4:] if last.startswith("rid=") and len(last) > 4 else None


def _fail(msg: str, proc: subprocess.CompletedProcess | None = None) -> None:
    if proc is not None:
        for line in (proc.stdout + proc.stderr).splitlines()[-5:]:
            print(f"BUDGET_LEDGER {line}", flush=True)
    print(msg, flush=True)


def prepare(a) -> int:
    path = Path(a.state)
    st = load_state(path)
    done = finished(a.episode_log)
    pend = [(t, s, sd) for t, s, sd in identities(a.manifest, a.shard, a.only_tasks) if (t, s) not in done]
    reserved = released = 0
    for task, src, seed in pend:
        key = f"{task}_xhard0_{seed}"
        ent = st.setdefault(key, {"rid": None, "state": None, "history": []})
        if ent["rid"] and ent["state"] == "reserved":
            p = call(["release", "--id", ent["rid"]])
            if p.returncode != 0:
                save_state(path, st)
                _fail(f"RUN_BLOCKED reason=budget_release_failed key={key} rid={ent['rid']}", p)
                return EXIT_BUDGET
            ent["history"].append({"rid": ent["rid"], "state": "released"})
            ent["state"] = "released"
            released += 1
        p = call(["reserve", "--resets", str(a.resets), "--route", a.route, "--key", key])
        rid = _rid_of(p.stdout) if p.returncode == 0 else None
        if rid is None:
            save_state(path, st)
            _fail(f"RUN_BLOCKED reason=budget reserved={reserved}/{len(pend)} resets_per_episode={a.resets} key={key}", p)
            return EXIT_BUDGET
        ent.update(rid=rid, state="reserved")
        ent["history"].append({"rid": rid, "state": "reserved"})
        reserved += 1
        save_state(path, st)
    save_state(path, st)
    print(f"BUDGET_PREPARE episodes={len(pend)} reserved={reserved} released={released} "
          f"resets_per_episode={a.resets} route={a.route}", flush=True)
    return 0


def settle(a) -> int:
    path = Path(a.state)
    st = load_state(path)
    done = finished(a.episode_log)
    committed = 0
    rc = 0
    for task, src, seed in identities(a.manifest, a.shard, a.only_tasks):
        key = f"{task}_xhard0_{seed}"
        ent = st.get(key)
        if not ent or ent.get("state") != "reserved" or (task, src) not in done:
            continue
        p = call(["commit", "--id", ent["rid"]])
        if p.returncode != 0:
            _fail(f"BUDGET_COMMIT_FAILED key={key} rid={ent['rid']}", p)
            rc = 1
            continue
        ent["state"] = "committed"
        ent["history"].append({"rid": ent["rid"], "state": "committed"})
        committed += 1
    save_state(path, st)
    still = sum(1 for e in st.values() if e.get("state") == "reserved")
    print(f"BUDGET_SETTLE committed={committed} open={still}", flush=True)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="原侧逐局预算预约（S8 账本 CLI）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "settle"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--shard", type=int, required=True)
        p.add_argument("--episode-log", required=True)
        p.add_argument("--state", required=True)
        p.add_argument("--only-tasks", default=None)
        if name == "prepare":
            p.add_argument("--resets", type=int, required=True)
            p.add_argument("--route", required=True)
    a = ap.parse_args(argv)
    try:
        return prepare(a) if a.cmd == "prepare" else settle(a)
    except FileNotFoundError as e:
        print(f"RUN_BLOCKED reason=budget_ledger_missing path={e}", flush=True)
        return EXIT_BUDGET


if __name__ == "__main__":
    sys.exit(main())
