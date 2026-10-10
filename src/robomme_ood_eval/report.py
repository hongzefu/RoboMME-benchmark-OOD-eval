"""``results.jsonl`` -> ``log.json``.

``log.json`` holds a ``success_rate`` per task that appears (successful episodes / counted episodes of that task);
``total_success_rate`` is the mean of the per-task rates (not weighted by episode count). Counted episodes are the
non-infrastructure ones (``infra`` false); infrastructure episodes are reported separately as ``infra`` and are not
in the denominator. When one ``key`` has several rows the last one wins (resumed runs append). Count fields are
always written explicitly, including zeros.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from robomme_ood_eval.episode import TASKS
from robomme_ood_eval.timing import merge_summaries

SCHEMA = "robomme-ood-eval-log/2"


def read_results(path: str | Path) -> list[dict]:
    """Read per-episode result rows (skipping a half-written last line left by a crash)."""
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


def speed_from_rows(rows: Iterable[dict], *, include_empty: bool = False) -> dict | None:
    """Aggregate latest identities by policy variant and GPU, excluding abnormal episodes.

    Missing old timing is counted separately from unavailable conservation evidence.
    The shared timing helper owns phase preservation, weighting and percentiles.
    """
    by_policy: dict[str, list[dict]] = {}
    for row in rows:
        label = row.get("policy_label") or row.get("model") or row.get("policy") or "unknown"
        variant = row.get("policy_variant")
        if variant and not row.get("policy_label"):
            label = f"{label}:{variant}"
        by_policy.setdefault(str(label), []).append(row)
    groups: dict[tuple, dict] = {}
    has_timing = False
    totals = {k: 0 for k in ("missing", "excluded_infra", "excluded_timeout", "excluded_reconnected", "excluded_status")}
    for label, policy_rows in by_policy.items():
        for row in _latest_by_key(policy_rows):
            timing = (row.get("timing") or {}).get("policy") or {}
            has_timing |= timing.get("chunk_summary") is not None
            group = groups.setdefault((label, timing.get("gpu_name")), {
                "policy_label": label, "gpu_name": timing.get("gpu_name"), "inputs": [], "identities": [], "layer_identities": {},
                **{k: 0 for k in totals}})
            if row.get("infra"):
                reason = "excluded_infra"
            elif row.get("status") == "timeout":
                reason = "excluded_timeout"
            elif timing.get("reconnected") is True and (
                    row.get("model") == "pp" or row.get("policy") == "pp" or label == "pp" or label.startswith("pp:")):
                reason = "excluded_reconnected"
            elif row.get("status") not in ("success", "fail", "failure"):
                reason = "excluded_status"
            elif timing.get("chunk_summary") is None:
                reason = "missing"
            else:
                group["inputs"].append(timing)
                episode = next((row[k] for k in ("episode", "builder_episode", "source_episode", "seed")
                                if row.get(k) is not None), None)
                identity = {"task": row.get("task"), "tier": row.get("tier"), "episode": episode}
                group["identities"].append(identity)
                for layer, stats in timing["chunk_summary"].get("by_planner_review", {}).items():
                    if stats.get("n_decisions", 0):
                        group["layer_identities"].setdefault(layer, []).append(identity)
                continue
            group[reason] += 1
            totals[reason] += 1
    if not has_timing and not include_empty:
        return None
    merged = []
    for group in groups.values():
        inputs = group.pop("inputs")
        merged.append({**merge_summaries(inputs), **group})
    return {"groups": merged, **totals}


def summarize_rows(rows: Iterable[dict]) -> dict:
    """Aggregate per task; returns the ``log.json`` content."""
    rows = list(rows)
    speed = speed_from_rows(rows)
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
        "speed": speed,
    }


def summarize(results: str | Path) -> dict:
    """``results`` may be ``results.jsonl`` or its directory; writes ``log.json`` next to it and returns the content."""
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
