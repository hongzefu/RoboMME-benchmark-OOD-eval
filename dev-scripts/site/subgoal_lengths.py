#!/usr/bin/env python3
"""v8 站点的逐段 subgoal 帧数与 oracle 步数统计（只读 h5 与 v8 规格，不读评估记录、不读任何计划表格）。

由 ``subgoal_lengths.py`` 改写（v8 方案第一部分 §2.2）：

- **h5 来源**：与 ``site_catalog.py`` 同一套 ``(tier, task, seed)`` 键。xhard1～5 取 ``delivery.json``
  （schema ``v8-delivery/1``）逐局的 h5 路径；xhard0 取 ``xhard0-gen/manifest-{H,O}.jsonl`` 的 ``h5`` 字段
  （新入口 H / 官方旧入口 O）。
- **段**：逐帧读 ``timestep_<k>/info/simple_subgoal``，文字连续相同的帧算一段；模板把序数词、颜色、数字抹成占位符，
  按本局首次 / 后续出现分开。
- **oracle 步数**：执行步 = 帧数 − ``info/is_video_demo`` 帧数；逐格给均值／最小～最大／局数；
  xhard1～5 逐局与 ``delivery.json`` 的 ``exec_steps`` 相等（不等即 FAIL）。
- **执行步上限**：xhard1～5 取 v8 规格 header ``exec_cap``（``hard-specs/4``），xhard0 取
  本地常量 ``XHARD0_STEP_CAP``（1300，与官方 scripts/evaluation.py 的默认步数相同）；不读 ``eval.*.max_steps``（v8 评估置空）。
- **逐局 task goal**：``setup/task_goal`` 全部措辞与 ``setup/difficulty``；配置在目录里（规格逐局抽值），本文件不重复。

输出 ``<site-dir>/subgoals.json``（schema ``v8-subgoals/1``，``open("x")`` 写入、拒绝覆盖），由
``site_server.py`` 的 ``/api/subgoals`` 提供给页面。末行打印
``V8_SUBGOALS=PASS|FAIL h5=<n> cells=<n> missing=<n> exec_mismatch=<n> xhard0_same=<a>/<b> goals=<n>``。

    uv run --no-sync python scripts/injection-dev/site/subgoal_lengths.py --site-dir artifacts/newtask-v8/site \\
      --specs-root artifacts/newtask-v8/specs-root --delivery artifacts/newtask-v8/gen1/delivery.json \\
      --xhard0-gen artifacts/newtask-v7/site-media/xhard0-gen
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("site_catalog", HERE / "site_catalog.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

XHARD0_SIDES = (("new", "H"), ("old", "O"))
ORDINALS = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
            "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth last").split()
COLORS = "red green blue yellow purple orange pink white black brown gray grey cyan magenta".split()
WORD = re.compile(r"\b(" + "|".join(ORDINALS + COLORS) + r")\b")
NUM = re.compile(r"\d+")
RATIO, FRAMES = 0.25, 15


def template(text: str) -> str:
    return NUM.sub("N", WORD.sub(lambda m: "〈序数〉" if m.group(1) in ORDINALS else "〈颜色〉", text))


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def read_episode(path: str) -> dict:
    """一局的逐段 subgoal、总帧数、视频演示帧数（``info/is_video_demo``）、task goal 全部措辞与 difficulty。"""
    with h5py.File(path, "r") as f:
        if not list(f):  # 官方原版就生成失败的空 h5
            return {"segs": [], "frames": 0, "demo": 0, "goal": [], "difficulty": None}
        g = f[list(f)[0]]
        steps = sorted((k for k in g if k.startswith("timestep_")), key=lambda k: int(k[9:]))
        segs: list[list] = []
        demo = 0
        for k in steps:
            info = g[k]["info"]
            s = _text(info["simple_subgoal"][()])
            demo += bool(info["is_video_demo"][()]) if "is_video_demo" in info else 0
            if segs and segs[-1][0] == s:
                segs[-1][1] += 1
            else:
                segs.append([s, 1])
        setup = g["setup"] if "setup" in g else {}
        goal = [_text(x) for x in setup["task_goal"][()]] if "task_goal" in setup else []
        diff = _text(setup["difficulty"][()]) if "difficulty" in setup else None
    return {"segs": segs, "frames": len(steps), "demo": demo, "goal": goal, "difficulty": diff}


def label(segs: list[list]) -> list[dict]:
    seen: dict[str, int] = defaultdict(int)
    out = []
    for text, n in segs:
        t = template(text)
        seen[t] += 1
        out.append({"text": text, "frames": n, "tpl": t, "kind": "首次" if seen[t] == 1 else "后续"})
    return out


def med(xs):
    return statistics.median(xs) if xs else None


#: xhard0 的执行步上限（按档步数查表已从 hard_specs 删除，评估入口按数据集传 max_steps）
XHARD0_STEP_CAP = 1300


def tier_caps(specs_root: Path, tiers) -> dict[str, int | None]:
    """xhard1～5：规格 header 的 ``exec_cap``；xhard0：本地常量 ``XHARD0_STEP_CAP``（1300）。"""
    caps: dict[str, int | None] = {"xhard0": XHARD0_STEP_CAP}
    for tier in tiers:
        if tier == "xhard0":
            continue
        path = Path(specs_root) / tier / "specs.jsonl"
        with path.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
        caps[tier] = header.get("exec_cap")
    return caps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-dir", type=Path, required=True)
    ap.add_argument("--specs-root", type=Path, default=C.DEFAULT_SOURCES["specs_root"])
    ap.add_argument("--delivery", type=Path, default=C.DEFAULT_SOURCES["delivery"])
    ap.add_argument("--xhard0-gen", type=Path, default=C.DEFAULT_SOURCES["xhard0_gen"])
    ap.add_argument("--path-base", type=Path, default=C.REPO_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    cat = json.loads((args.site_dir / "catalog.json").read_text(encoding="utf-8"))
    delivery = C.load_delivery(args.delivery)
    gen = {(r["tier"], r["task"], int(r["seed"])): r for r in delivery["rows"]}
    xgen = {side: {(r["task"], int(r["seed"])): r for r in C.jsonl(args.xhard0_gen / f"manifest-{side}.jsonl")}
            for side in ("H", "O")}

    jobs, problems = [], []
    for t in cat["tasks"]:
        for tier, cell in t["tiers"].items():
            for ep in cell["episodes"]:
                if tier == "xhard0":
                    for entry, side in XHARD0_SIDES:
                        row = xgen[side].get((t["id"], ep["seed"]))
                        if row is None:
                            problems.append(f"找不到 xhard0 {side} h5：{t['id']}/{ep['seed']}")
                            continue
                        jobs.append((t["id"], tier, ep["idx"], ep["seed"], entry, str(C.resolve(row["h5"], args.path_base)), None))
                else:
                    row = gen.get((tier, t["id"], ep["seed"]))
                    if row is None:
                        problems.append(f"delivery 缺行：{tier}/{t['id']}/{ep['seed']}")
                        continue
                    jobs.append((t["id"], tier, ep["idx"], ep["seed"], "new",
                                 str(C.delivery_h5(row, args.delivery, args.path_base)), row.get("exec_steps")))
    missing_files = [j for j in jobs if not Path(j[5]).is_file()]
    for j in missing_files:
        problems.append(f"h5 不存在：{j[1]}/{j[0]}/{j[3]}/{j[4]} {j[5]}")
    jobs = [j for j in jobs if Path(j[5]).is_file()]
    if args.workers <= 1:  # 串行（测试与小目录）
        reads = [read_episode(j[5]) for j in jobs]
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            reads = list(ex.map(read_episode, [j[5] for j in jobs], chunksize=8))
    caps = tier_caps(args.specs_root, sorted({tier for t in cat["tasks"] for tier in t["tiers"]}))

    episodes: dict = defaultdict(lambda: defaultdict(dict))
    steps: dict = defaultdict(lambda: defaultdict(list))  # 任务 → 档 → [(执行步, 演示帧)]
    goals = exec_mismatch = over_cap = 0
    for (task, tier, idx, seed, entry, _, exec_steps), r in zip(jobs, reads):
        rec = episodes[task][tier].setdefault(str(idx), {"seed": seed})
        rec[entry] = label(r["segs"])
        if entry == "new":
            rec.update(frames=r["frames"], demo=r["demo"], goal=r["goal"], difficulty=r["difficulty"])
            goals += bool(r["goal"])
            if r["frames"]:
                steps[task][tier].append((r["frames"] - r["demo"], r["demo"]))
            if tier != "xhard0":
                if exec_steps is None or exec_steps != r["frames"] - r["demo"]:
                    exec_mismatch += 1
                    problems.append(f"执行步与 delivery 不符：{tier}/{task}/{seed} h5={r['frames'] - r['demo']} "
                                    f"delivery={exec_steps}")
                if caps.get(tier) is not None and r["frames"] - r["demo"] > caps[tier]:
                    over_cap += 1
                    problems.append(f"执行步超过上限：{tier}/{task}/{seed} {r['frames'] - r['demo']} > {caps[tier]}")

    oracle: dict = defaultdict(dict)
    want_cells = {(t["id"], tier) for t in cat["tasks"] for tier in t["tiers"]}
    for task, tiers in steps.items():
        for tier, xs in tiers.items():
            ex_ = [a for a, _ in xs]
            oracle[task][tier] = {"n": len(xs), "mean": round(statistics.mean(ex_), 1), "min": min(ex_),
                                  "max": max(ex_), "demo_mean": round(statistics.mean(b for _, b in xs), 1),
                                  "demo_min": min(b for _, b in xs), "demo_max": max(b for _, b in xs),
                                  "total_mean": round(statistics.mean(a + b for a, b in xs), 1),
                                  "total_min": min(a + b for a, b in xs), "total_max": max(a + b for a, b in xs),
                                  "max_steps": caps.get(tier)}
    missing_cells = sorted(f"{task}/{tier}" for task, tier in want_cells if tier not in oracle.get(task, {}))
    for key in missing_cells:
        problems.append(f"oracle 缺格：{key}")
    missing_eps = sum(1 for t in cat["tasks"] for tier, cell in t["tiers"].items() for ep in cell["episodes"]
                      if "new" not in episodes.get(t["id"], {}).get(tier, {}).get(str(ep["idx"]), {}))

    x0_same = x0_total = 0
    for task, tiers in episodes.items():
        for rec in tiers.get("xhard0", {}).values():
            x0_total += 1
            x0_same += rec.get("new") == rec.get("old")
    if x0_same != x0_total:
        problems.append(f"xhard0 新旧入口逐段不同 {x0_total - x0_same} 局")

    summary = {}
    for task, tiers in episodes.items():
        pool: dict = defaultdict(lambda: defaultdict(list))
        order: list = []
        for tier in cat["tiers"]:
            for rec in tiers.get(tier, {}).values():
                for s in rec.get("new", []):
                    key = (s["tpl"], s["kind"])
                    if key not in order:
                        order.append(key)
                    pool[key][tier].append(s["frames"])
        rows = []
        for key in order:
            base = med(pool[key].get("xhard0", []))
            cells = {}
            for tier, xs in pool[key].items():
                m = med(xs)
                verdict = "基准" if tier == "xhard0" else (
                    "新增" if base is None else ("不一致" if abs(m - base) > max(RATIO * base, FRAMES) else "一致"))
                cells[tier] = {"n": len(xs), "median": m, "min": min(xs), "max": max(xs), "verdict": verdict}
            rows.append({"tpl": key[0], "kind": key[1], "tiers": cells})
        summary[task] = rows

    out = {"schema": "v8-subgoals/1", "oracle": oracle, "caps": caps,
           "rule": f"同任务同模板，某档中位数与 xhard0 中位数相差超过 {int(RATIO * 100)}% 且超过 {FRAMES} 帧记为不一致；"
                   "xhard0 没有的模板记为新增。段 = simple_subgoal 文字连续相同的帧；模板 = 抹去序数词、颜色、数字，"
                   "并按本局首次 / 后续出现分开。",
           "xhard0_new_old_same": [x0_same, x0_total], "summary": summary, "episodes": episodes}
    status = "PASS" if not problems and not missing_eps else "FAIL"
    for p in problems[:30]:
        print(f"# {p}")
    if status == "PASS":
        with open(args.site_dir / "subgoals.json", "x", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    n_cells = sum(len(v) for v in oracle.values())
    print(f"V8_SUBGOALS={status} h5={len(jobs)} cells={n_cells} missing={len(missing_files) + missing_eps + len(missing_cells)} "
          f"exec_mismatch={exec_mismatch} over_cap={over_cap} xhard0_same={x0_same}/{x0_total} goals={goals}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
