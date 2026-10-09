"""``results.jsonl`` → ``log.json``（拆分方案 §四；汇总口径参照 ``dev-scripts/gl/eval_report.py``）。

``log.json``：每个出现过的任务各自 ``success_rate``（该任务成功局数 ／ 计入局数），``total_success_rate`` 为各任务
成功率的平均（不是总局数加权）。计入局 = 非基础设施局（``infra`` 为假）；基础设施局单列 ``infra``、不进分母。
同一 ``key`` 有多行时取最后一行（续跑补写）。计数字段一律显式写 0（P4）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from robomme_ood_eval.episode import TASKS

SCHEMA = "robomme-ood-eval-log/1"


def read_results(path: str | Path) -> list[dict]:
    """读逐局结果行（跳过崩溃留下的半行）。"""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_by_key(rows: Iterable[dict]) -> list[dict]:
    last: dict[str, dict] = {}
    for r in rows:
        k = r.get("key") or f"{r.get('task')}_{r.get('tier')}_{r.get('seed')}"
        last.pop(k, None)
        last[k] = r
    return list(last.values())


def summarize_rows(rows: Iterable[dict]) -> dict:
    """按任务汇总；返回 ``log.json`` 内容。"""
    rows = _latest_by_key(rows)
    per: dict[str, dict] = {}
    for r in rows:
        t = str(r.get("task"))
        d = per.setdefault(t, {"episodes": 0, "counted": 0, "success": 0, "fail": 0, "timeout": 0, "infra": 0})
        d["episodes"] += 1
        if r.get("infra"):
            d["infra"] += 1
            continue
        d["counted"] += 1
        st = r.get("status")
        if st == "success":
            d["success"] += 1
        elif st == "timeout":
            d["timeout"] += 1
        else:
            d["fail"] += 1
    order = [t for t in TASKS if t in per] + sorted(t for t in per if t not in TASKS)
    tasks = {}
    rates = []
    for t in order:
        d = per[t]
        rate = d["success"] / d["counted"] if d["counted"] else None
        tasks[t] = {**d, "success_rate": rate}
        if rate is not None:
            rates.append(rate)
    first = rows[0] if rows else {}
    return {
        "schema": SCHEMA,
        "model": first.get("model"),
        "policy_label": first.get("policy_label"),
        "dataset": first.get("dataset"),
        "policy_seed": first.get("policy_seed"),
        "tasks": tasks,
        "total_success_rate": (sum(rates) / len(rates)) if rates else None,
        "tasks_evaluated": len(rates),
        "tasks_missing": [t for t in TASKS if t not in per],
        "episodes": len(rows),
        "counted": sum(d["counted"] for d in per.values()),
        "success": sum(d["success"] for d in per.values()),
        "infra": sum(d["infra"] for d in per.values()),
    }


def summarize(results: str | Path) -> dict:
    """``results`` 可以是 ``results.jsonl`` 或其所在目录；在同目录写 ``log.json`` 并返回内容。"""
    p = Path(results)
    if p.is_dir():
        p = p / "results.jsonl"
    log = summarize_rows(read_results(p))
    out = p.parent / "log.json"
    tmp = out.with_name(".log.json.tmp")
    tmp.write_text(json.dumps(log, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    rate = log["total_success_rate"]
    print(f"EVAL_LOG path={out} episodes={log['episodes']} counted={log['counted']} success={log['success']} "
          f"infra={log['infra']} tasks={log['tasks_evaluated']} "
          f"total_success_rate={'NA' if rate is None else f'{rate:.4f}'}", flush=True)
    return log
