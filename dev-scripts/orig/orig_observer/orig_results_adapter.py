#!/usr/bin/env python3
"""原侧逐局日志 → 唯一 attempt 映射 ``orig-attempts.json``（计划第二部分一节 S7，审计第 12、14 条；供 S6
``gate2_compare.py --orig-attempts``）。

输入：
- 原版逐局日志：SimpleMemVLA ``--episode_log``（``episodes-shardXXofYY.jsonl``）或 FrameSamp+Modulation ``episodes.jsonl``；每行至少
  ``task``、``source_episode``、``seed``、``status``，每局（含 error 后的重评）追加一行；
- 观测器录制根：逐局目录 ``<key>.a<N>/trace.jsonl``（``N`` 为该身份在录制根下的开局顺序号，含中途被杀的局）。

规则：
1. 每个 ``(task, source_episode)`` 取**最后一条终态行**（success／fail／timeout）为权威；没有终态行时取最后一行
   （``status=error``，``unresolved=true``）。
2. 局目录与日志行的对应：该身份**带 ``end`` 行**的 trace 按 ``N`` 升序，与该身份日志行按出现顺序一一对应；没有
   ``end`` 行（进程中途被杀，钩子没收尾）或缺 trace 的目录记 ``orphan``（有目录无终态行，不进比较）；完整 trace 多于
   日志行的、排在末尾的多余目录也记 ``orphan``（收尾后、写日志前进程退出）。
3. 权威行的 ``attempt`` = 对上的局目录号 ``N``（与 trace ``identity.attempt``、目录名 ``.a<N>`` 一致，C6）；对不上
   局目录记 ``missing_trace``；对上了但 trace ``end.status`` 与日志 ``status`` 不同记 ``status_mismatch``；同一身份
   日志行多于完整 trace 记 ``ambiguous``（无法确定哪一行缺 trace）。

输出 ``orig-attempts.json``：``{"schema":"orig-attempts/1","policy","route","identities":[...],"orphans":[...],
"counts":{...}}``，``identities`` 每项 ``{task, source_episode, seed, key, status, attempt, ep_dir, trace_path,
row_index, rows_total, terminal_rows, unresolved, trace_status, problem}``；末行
``ORIG_ATTEMPTS=PASS|FAIL identities=<n> resolved=<n> unresolved=<n> orphan=<n> missing_trace=<n> ambiguous=<n>
status_mismatch=<n>``（``missing_trace``／``ambiguous``／``status_mismatch`` 任一非 0 即 FAIL；``unresolved`` 与
``orphan`` 只报告不判 FAIL）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "orig-attempts/1"
FINAL = ("success", "fail", "timeout")
EP_DIR_RE = re.compile(r"^(?P<key>.+)\.a(?P<attempt>\d+)$")
ROUTES = {"smvla": "smvla/orig", "perceptual-framesamp-modul": "perceptual-framesamp-modul/orig"}


def official_defs():
    """评估包里的别名表 ``robomme_ood_eval.models._official_defs``（旧名别名表的唯一来源；模块名 ``official_defs``，
    已加载则复用同一模块）。"""
    import importlib.util

    mod = sys.modules.get("official_defs")
    if mod is None or not hasattr(mod, "canonical_row"):
        path = Path(__file__).resolve().parents[3] / "src" / "robomme_ood_eval" / "models" / "_official_defs.py"
        spec = importlib.util.spec_from_file_location("official_defs", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["official_defs"] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop("official_defs", None)
            raise
    return mod


def key_of(task: str, seed: int) -> str:
    return f"{task}_xhard0_{int(seed)}"


def read_log(paths: list[str | Path]) -> list[dict]:
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(official_defs().canonical_row(json.loads(line)))
    return rows


def _trace_end(tpath: Path) -> dict | None:
    """trace 的 end 行（没有或读不了返回 None）；同时返回 header 身份供核对。"""
    try:
        lines = [x for x in tpath.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        last = json.loads(lines[-1])
        head = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    if last.get("kind") != "end" or head.get("kind") != "header":
        return None
    return {"end": last, "identity": head.get("identity") or {}, "route": official_defs().canonical_route(head.get("route"))}


def scan_episodes(rec_root: str | Path) -> dict[str, list[dict]]:
    """录制根下的局目录：key → [{attempt, dir, trace, complete, end_status}]，按 attempt 升序。"""
    out: dict[str, list[dict]] = defaultdict(list)
    root = Path(rec_root)
    if not root.is_dir():
        return {}
    for d in root.iterdir():
        m = EP_DIR_RE.match(d.name)
        if not (m and d.is_dir()):
            continue
        t = d / "trace.jsonl"
        info = _trace_end(t) if t.exists() else None
        out[m["key"]].append({"attempt": int(m["attempt"]), "dir": str(d), "trace": str(t) if t.exists() else None,
                              "complete": info is not None,
                              "end_status": None if info is None else info["end"].get("status"),
                              "identity_attempt": None if info is None else info["identity"].get("attempt")})
    for v in out.values():
        v.sort(key=lambda x: x["attempt"])
    return dict(out)


def build(log_rows: list[dict], rec_root: str | Path, *, policy: str) -> dict:
    policy = official_defs().canonical_policy(policy)  # 历史调用方的旧标签只读兼容；写出只用官方名
    by_ident: dict[tuple, list[dict]] = defaultdict(list)
    for r in log_rows:
        by_ident[(r["task"], int(r["source_episode"]))].append(r)
    eps = scan_episodes(rec_root)
    identities, orphans = [], []
    counts = {"identities": 0, "resolved": 0, "unresolved": 0, "orphan": 0, "missing_trace": 0, "ambiguous": 0,
              "status_mismatch": 0}
    used_keys = set()
    for (task, src), rows in sorted(by_ident.items()):
        seed = int(rows[-1]["seed"])
        key = key_of(task, seed)
        used_keys.add(key)
        dirs = eps.get(key, [])
        complete = [d for d in dirs if d["complete"]]
        for d in dirs:
            if not d["complete"]:
                orphans.append({"key": key, "attempt": d["attempt"], "dir": d["dir"], "reason": "no_end_line"})
        for d in complete[len(rows):]:
            orphans.append({"key": key, "attempt": d["attempt"], "dir": d["dir"], "reason": "no_log_row"})
        term_idx = [i for i, r in enumerate(rows) if r.get("status") in FINAL]
        idx = term_idx[-1] if term_idx else len(rows) - 1
        row = rows[idx]
        unresolved = not term_idx
        item = {"task": task, "source_episode": src, "seed": seed, "key": key,
                "status": row.get("status") if not unresolved else "error", "attempt": None, "ep_dir": None,
                "trace_path": None, "row_index": idx, "rows_total": len(rows), "terminal_rows": len(term_idx),
                "unresolved": unresolved, "trace_status": None, "problem": None,
                "orig_row_attempt": row.get("attempt")}
        if len(complete) < len(rows):
            # 日志行多于完整 trace：只有缺口全在权威行之后时才能确定对应
            if idx < len(complete):
                counts["ambiguous"] += 1
                item["problem"] = "ambiguous"
            else:
                counts["missing_trace"] += 1
                item["problem"] = "missing_trace"
        if item["problem"] is None:
            d = complete[idx]
            item.update(attempt=d["attempt"], ep_dir=d["dir"], trace_path=d["trace"], trace_status=d["end_status"])
            if d["end_status"] != row.get("status"):
                counts["status_mismatch"] += 1
                item["problem"] = "status_mismatch"
            elif d["identity_attempt"] is not None and int(d["identity_attempt"]) != d["attempt"]:
                counts["status_mismatch"] += 1
                item["problem"] = "identity_attempt_mismatch"
        counts["identities"] += 1
        counts["unresolved" if unresolved else "resolved"] += 1
        identities.append(item)
    for key, dirs in sorted(eps.items()):
        if key in used_keys:
            continue
        for d in dirs:
            orphans.append({"key": key, "attempt": d["attempt"], "dir": d["dir"], "reason": "identity_not_in_log"})
    counts["orphan"] = len(orphans)
    ok = counts["missing_trace"] == 0 and counts["ambiguous"] == 0 and counts["status_mismatch"] == 0
    return {"schema": SCHEMA, "policy": policy, "route": ROUTES.get(policy), "rec_root": str(rec_root),
            "identities": identities, "orphans": orphans, "counts": counts, "ORIG_ATTEMPTS": "PASS" if ok else "FAIL"}


def verdict_line(res: dict) -> str:
    c = res["counts"]
    return (f"ORIG_ATTEMPTS={res['ORIG_ATTEMPTS']} identities={c['identities']} resolved={c['resolved']} "
            f"unresolved={c['unresolved']} orphan={c['orphan']} missing_trace={c['missing_trace']} "
            f"ambiguous={c['ambiguous']} status_mismatch={c['status_mismatch']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="原侧逐局日志 → 唯一 attempt 映射 orig-attempts.json")
    ap.add_argument("--policy", required=True, choices=sorted(ROUTES))
    ap.add_argument("--episode-log", nargs="+", required=True, help="原版逐局日志（可多份，按给出顺序拼接）")
    ap.add_argument("--rec-root", required=True, help="观测器录制根（含 <key>.a<N>/）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = build(read_log(a.episode_log), a.rec_root, policy=a.policy)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(verdict_line(res), flush=True)
    return 0 if res["ORIG_ATTEMPTS"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
