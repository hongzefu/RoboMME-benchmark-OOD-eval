"""四模型评估总报告（计划第二部分 1.7）：各模型 × 数据集 × 入口（新侧／原侧）的覆盖与成功率，并入第二档差异与预算账本。

输入 ``--sets <sets.json>``，格式::

    {"sets": [
      {"policy": "groundsg", "variant": "ground-sg-oracle", "dataset": "hard-verify", "side": "new", "site": "gl",
       "results": ["<结果 jsonl 路径或 glob>", ...], "expect_total": 192,
       "manifest": "<可选：身份清单 json/jsonl，含 task/tier/seed>"},
      ...]}

- 每个集合内按身份 ``(task, tier, seed)`` 取最终行（``canary``／``infra``／``late`` 为真的行不算）；``tier`` 取结果行
  ``tier``，缺省取 ``identity.tier``，xhard0 结果行即 ``xhard0``。同一身份多条最终行且终态不同记
  ``conflicting``，终态相同记 ``duplicate``（都算不完整）。
- 有 ``manifest`` 时缺失／多余按身份逐个算；否则 ``missing = max(0, expect_total - 覆盖数)``。
- 成功率 = ``status == "success"`` 的最终行 / 覆盖数，另按任务、按档分列；终态分布按 ``status`` 计。
- ``--gate2 <gate2_compare 输出 json>``（可多份）：并入第二档差异摘要。
- 预算：``--ledger``（env_client 账本 jsonl，可 glob、可多份）时，尝试数 = ``attempt_start`` 行数，reset 数 =
  ``reset_claim`` 行数；不给账本时从结果行算（尝试数 = 非金丝雀结果行数，reset 数 = ``reset_calls`` 之和）。
  上限由 ``--max-attempts``、``--max-resets`` 给出。

判定行::

    MODEL_EVAL_REPORT=PASS|PARTIAL sets=<n> complete=<n> incomplete=<n> episodes=<n> gate2=<n>
    EVAL_BUDGET=PASS|FAIL attempts=<n><=<上限> resets=<n><=<上限> source=ledger|results

``PARTIAL`` 表示有集合覆盖不全或有重复／冲突，报告照样写出（不阻塞）；未给上限时 ``EVAL_BUDGET=NA``。

第三阶段矩阵（1006 计划第二部分八.6、四节第 7 步；只在任一集合写了 ``policy_seed`` 或给了 ``--policy-seed`` 时输出）：
每个集合是一个组合（模型 × 变体 × 数据集 × 入口），集合的 ``policy_seed`` 缺省取 ``--policy-seed``；集合内每条最终行的
``policy_seed`` 必须等于它（缺字段也算不符，不把历史缺字段补成已证种子），计 ``seed_mismatch``。按种子各出一行::

    STAGE3_MATRIX=PASS|FAIL policy_seed=<s> combinations=<n> unique_terminal=<n> incomplete=<n> seed_mismatch=<n>
        duplicate_combination=<n>

``unique_terminal``：各组合里恰有一条最终行、终态为 success／fail／timeout／error 的身份数之和。PASS 要求组合数 > 0、
每个组合完整（同 ``complete``）、``seed_mismatch == 0``、同一种子下组合标签不重复。任一行 FAIL 时退出码 1。

用法::

    python dev-scripts/checks/model_eval_report.py --sets sets.json [--gate2 g.json ...] \
        [--ledger '<glob>' ...] [--max-attempts N --max-resets M] --out-json report.json --out-md report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        out += [Path(h) for h in hits] if hits else ([Path(pat)] if Path(pat).exists() else [])
    seen: set[Path] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def official_defs():
    """评估包里的别名表 ``robomme_ood_eval.models._official_defs``（旧名别名表的唯一来源；模块名 ``official_defs``，
    已加载则复用同一模块）。"""
    import importlib.util

    mod = sys.modules.get("official_defs")
    if mod is None or not hasattr(mod, "canonical_row"):
        path = Path(__file__).resolve().parents[2] / "src" / "robomme_ood_eval" / "models" / "_official_defs.py"
        spec = importlib.util.spec_from_file_location("official_defs", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["official_defs"] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop("official_defs", None)
            raise
    return mod


def _jsonl(path: Path) -> list[dict]:
    """JSON 数组／对象或 JSONL；历史行的旧策略标签／数据集名／路线映射成官方名。"""
    canon = official_defs().canonical_row
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("rows", data.get("identities", []))
        return [canon(r) for r in rows]
    return [canon(json.loads(x)) for x in text.splitlines() if x.strip()]


def tier_of(row: dict) -> str | None:
    if row.get("tier") is not None:
        return row["tier"]
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    return ident.get("tier")


def key_of(row: dict) -> tuple:
    seed = row.get("seed")
    return (row.get("task"), tier_of(row), None if seed is None else int(seed))


def is_final(row: dict) -> bool:
    return not (row.get("canary") or row.get("infra") or row.get("late"))


def label_of(s: dict) -> str:
    pol = s["policy"] + (f"-{s['variant']}" if s.get("variant") else "")
    return f"{pol}/{s['dataset']}/{s.get('side', 'new')}" + (f"@{s['site']}" if s.get("site") else "")


def _rate(succ: int, n: int) -> float | None:
    return round(succ / n, 4) if n else None


def summarize_set(spec: dict, policy_seed: int | None = None) -> dict:
    defs = official_defs()
    seed = spec.get("policy_seed", policy_seed)
    spec = {**spec, "policy": defs.canonical_policy(spec["policy"]), "dataset": defs.canonical_dataset(spec["dataset"])}
    files = _paths(list(spec.get("results") or []))
    rows = [r for p in files for r in _jsonl(p)]
    finals: dict[tuple, list[dict]] = defaultdict(list)
    attempts = resets = 0
    for r in rows:
        if r.get("canary"):
            continue
        attempts += 1
        resets += int(r.get("reset_calls") or 0)
        if is_final(r):
            finals[key_of(r)].append(r)
    conflicting = sum(1 for v in finals.values() if len({x.get("status") for x in v}) > 1)
    duplicate = sum(1 for v in finals.values() if len(v) > 1 and len({x.get("status") for x in v}) == 1)
    chosen = {k: v[0] for k, v in finals.items()}
    expect = spec.get("expect_total")
    if spec.get("manifest"):
        want = {key_of(r) for r in _jsonl(Path(spec["manifest"]))}
        missing = len(want - set(chosen))
        extra = len(set(chosen) - want)
        expect = len(want) if expect is None else expect
    else:
        missing = max(0, int(expect) - len(chosen)) if expect is not None else 0
        extra = max(0, len(chosen) - int(expect)) if expect is not None else 0
    status = Counter(r.get("status") for r in chosen.values())
    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for k, r in chosen.items():
        ok = int(r.get("status") == "success")
        by_task[k[0]][0] += ok
        by_task[k[0]][1] += 1
        by_tier[str(k[1])][0] += ok
        by_tier[str(k[1])][1] += 1
    n = len(chosen)
    seed_mismatch = 0
    if seed is not None:
        seed_mismatch = sum(1 for v in finals.values() for r in v if r.get("policy_seed") != seed)
    unique_terminal = sum(1 for v in finals.values()
                          if len(v) == 1 and v[0].get("status") in ("success", "fail", "timeout", "error"))
    complete = missing == 0 and extra == 0 and conflicting == 0 and duplicate == 0 and bool(files) and \
        (expect is None or n == int(expect))
    return {
        "label": label_of(spec), "policy": spec["policy"], "variant": spec.get("variant"), "dataset": spec["dataset"],
        "side": spec.get("side", "new"), "site": spec.get("site"), "files": [str(p) for p in files],
        "expect_total": expect, "covered": n, "missing": missing, "extra": extra, "duplicate": duplicate,
        "conflicting": conflicting, "complete": complete, "status": dict(sorted(status.items(), key=str)),
        "success": status.get("success", 0), "success_rate": _rate(status.get("success", 0), n),
        "by_task": {t: {"success": s, "n": m, "rate": _rate(s, m)} for t, (s, m) in sorted(by_task.items())},
        "by_tier": {t: {"success": s, "n": m, "rate": _rate(s, m)} for t, (s, m) in sorted(by_tier.items())},
        "attempts": attempts, "resets": resets,
        "policy_seed": seed, "seed_mismatch": seed_mismatch, "unique_terminal": unique_terminal,
    }


def read_ledgers(patterns: list[str]) -> dict:
    files = _paths(patterns)
    kinds: Counter = Counter()
    for p in files:
        for r in _jsonl(p):
            kinds[r.get("kind")] += 1
    return {"files": [str(p) for p in files], "attempts": kinds.get("attempt_start", 0),
            "resets": kinds.get("reset_claim", 0)}


def read_gate2(paths: list[str]) -> list[dict]:
    out = []
    for p in _paths(paths):
        s = json.loads(Path(p).read_text(encoding="utf-8")).get("summary", {})
        out.append({"file": str(p), **{k: v for k, v in s.items() if not isinstance(v, list)}})
    return out


def stage3_matrix(summaries: list[dict]) -> list[dict]:
    """按 ``policy_seed`` 分组的第三阶段矩阵判定（见模块说明）；没有任何种子信息返回空表。"""
    groups: dict = defaultdict(list)
    for s in summaries:
        groups[s["policy_seed"]].append(s)
    out = []
    for seed, lst in sorted(groups.items(), key=lambda kv: str(kv[0])):
        labels = Counter(s["label"] for s in lst)
        res = {"policy_seed": seed, "combinations": len(lst),
               "unique_terminal": sum(s["unique_terminal"] for s in lst),
               "incomplete": sum(1 for s in lst if not s["complete"]),
               "seed_mismatch": sum(s["seed_mismatch"] for s in lst),
               "duplicate_combination": sum(n - 1 for n in labels.values() if n > 1),
               "labels": [s["label"] for s in lst]}
        res["verdict"] = "PASS" if (seed is not None and res["combinations"] > 0 and res["incomplete"] == 0
                                    and res["seed_mismatch"] == 0 and res["duplicate_combination"] == 0) else "FAIL"
        out.append(res)
    return out


def build_report(sets: list[dict], *, gate2: list[str] = (), ledgers: list[str] = (), max_attempts: int | None = None,
                 max_resets: int | None = None, policy_seed: int | None = None) -> dict:
    summaries = [summarize_set(s, policy_seed) for s in sets]
    g2 = read_gate2(list(gate2))
    if ledgers:
        led = read_ledgers(list(ledgers))
        budget = {"source": "ledger", "attempts": led["attempts"], "resets": led["resets"], "files": led["files"]}
    else:
        budget = {"source": "results", "attempts": sum(s["attempts"] for s in summaries),
                  "resets": sum(s["resets"] for s in summaries)}
    budget.update(max_attempts=max_attempts, max_resets=max_resets)
    if max_attempts is None or max_resets is None:
        budget["verdict"] = "NA"
    else:
        budget["verdict"] = "PASS" if budget["attempts"] <= max_attempts and budget["resets"] <= max_resets else "FAIL"
    incomplete = sum(1 for s in summaries if not s["complete"])
    rep = {"sets": summaries, "gate2": g2, "budget": budget,
           "verdict": "PASS" if summaries and incomplete == 0 else "PARTIAL",
           "complete": len(summaries) - incomplete, "incomplete": incomplete,
           "episodes": sum(s["covered"] for s in summaries)}
    if policy_seed is not None or any("policy_seed" in s for s in sets):
        rep["stage3"] = stage3_matrix(summaries)
    return rep


def verdict_lines(rep: dict) -> list[str]:
    b = rep["budget"]
    l1 = (f"MODEL_EVAL_REPORT={rep['verdict']} sets={len(rep['sets'])} complete={rep['complete']} "
          f"incomplete={rep['incomplete']} episodes={rep['episodes']} gate2={len(rep['gate2'])}")
    if b["verdict"] == "NA":
        l2 = f"EVAL_BUDGET=NA attempts={b['attempts']} resets={b['resets']} source={b['source']}"
    else:
        l2 = (f"EVAL_BUDGET={b['verdict']} attempts={b['attempts']}<={b['max_attempts']} "
              f"resets={b['resets']}<={b['max_resets']} source={b['source']}")
    out = [l1, l2]
    for m in rep.get("stage3") or []:
        out.append(f"STAGE3_MATRIX={m['verdict']} policy_seed={'NA' if m['policy_seed'] is None else m['policy_seed']} "
                   f"combinations={m['combinations']} unique_terminal={m['unique_terminal']} "
                   f"incomplete={m['incomplete']} seed_mismatch={m['seed_mismatch']} "
                   f"duplicate_combination={m['duplicate_combination']}")
    return out


def to_markdown(rep: dict) -> str:
    stage3 = rep.get("stage3")
    L = ["# 四模型评估总报告", "", "## 覆盖与成功率", ""]
    if stage3:
        L += ["| 集合 | 模型种子 | 期望 | 覆盖 | 缺失 | 多余 | 重复 | 冲突 | 成功 | 成功率 | 终态分布 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    else:
        L += ["| 集合 | 期望 | 覆盖 | 缺失 | 多余 | 重复 | 冲突 | 成功 | 成功率 | 终态分布 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in rep["sets"]:
        dist = "，".join(f"{k}:{v}" for k, v in s["status"].items())
        seed_cell = f" {s['policy_seed']} |" if stage3 else ""
        L.append(f"| {s['label']} |{seed_cell} {s['expect_total']} | {s['covered']} | {s['missing']} | {s['extra']} | "
                 f"{s['duplicate']} | {s['conflicting']} | {s['success']} | {s['success_rate']} | {dist} |")
    for s in rep["sets"]:
        L += ["", f"### {s['label']}：按档", "", "| 档 | 成功 | 局数 | 成功率 |", "|---|---|---|---|"]
        L += [f"| {t} | {v['success']} | {v['n']} | {v['rate']} |" for t, v in s["by_tier"].items()]
        L += ["", f"### {s['label']}：按任务", "", "| 任务 | 成功 | 局数 | 成功率 |", "|---|---|---|---|"]
        L += [f"| {t} | {v['success']} | {v['n']} | {v['rate']} |" for t, v in s["by_task"].items()]
    if rep["gate2"]:
        L += ["", "## 第二档差异", "", "| 模型 | 地点 | 判定 | 配对 | 同终态 | 逐项相同 | 启动首局相同 |", "|---|---|---|---|---|---|---|"]
        for g in rep["gate2"]:
            fe = f"{g['first_episode_identical']}/{g['server_epochs']}" if "first_episode_identical" in g else "—"
            L.append(f"| {g.get('policy')} | {g.get('site') or 'gl'} | {g.get('verdict')} | {g.get('compared')} | "
                     f"{g.get('same_terminal')} | {g.get('identical_trace', '—')} | {fe} |")
    b = rep["budget"]
    L += ["", "## 预算", "", f"- 来源：{b['source']}；尝试 {b['attempts']}（上限 {b['max_attempts']}），"
          f"reset {b['resets']}（上限 {b['max_resets']}），判定 {b['verdict']}。", ""]
    L += ["", "```", *verdict_lines(rep), "```", ""]
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="四模型评估总报告")
    ap.add_argument("--sets", required=True, help="集合配置 json（格式见模块说明）")
    ap.add_argument("--gate2", nargs="*", default=[])
    ap.add_argument("--ledger", nargs="*", default=[])
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--max-resets", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--policy-seed", type=int, default=None,
                    help="第三阶段矩阵：集合未写 policy_seed 时的缺省种子；给出即输出 STAGE3_MATRIX 行")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = json.loads(Path(args.sets).read_text(encoding="utf-8"))
    sets = cfg["sets"] if isinstance(cfg, dict) else cfg
    rep = build_report(sets, gate2=args.gate2, ledgers=args.ledger, max_attempts=args.max_attempts,
                       max_resets=args.max_resets, policy_seed=args.policy_seed)
    lines = verdict_lines(rep)
    rep["lines"] = lines
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(to_markdown(rep), encoding="utf-8")
    for line in lines:
        print(line, flush=True)
    stage3_fail = any(m["verdict"] != "PASS" for m in rep.get("stage3") or [])
    return 1 if rep["budget"]["verdict"] == "FAIL" or stage3_fail else 0


if __name__ == "__main__":
    sys.exit(main())
