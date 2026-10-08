#!/usr/bin/env python3
"""v8 站点：xhard1～5 相对 xhard0 的 task goal／subgoal 语义调整（用户 2026-10-02「Go和SubGo有哪些语义上的调整」，标注在网页上）。

只读 ``<site-dir>/subgoals.json``（``subgoal_lengths.py`` 从真实 h5 逐局提取的 task goal 措辞与逐段 subgoal），写
``<site-dir>/semantic.json``（schema ``v8-semantic/1``），由站点 ``/api/semantic`` 提供给页面。

判定口径（只比句式，不比参数）：

- **goal**：每条 task goal 措辞先归一化——序数词（first/second/…/last/next/another）→〈序数〉、颜色词→〈颜色〉、
  数量词与数字→〈数〉、``times``→次、复数名词并为单数、「〈数〉〈颜色〉cube, …and …」整段并为〈数〉〈颜色〉cube列表；
  某档出现 xhard0 没有的归一化句式记「新增句式」，xhard0 有而该档一条都没有的记「不再出现的句式」。
- **subgoal**：按 ``subgoals.json`` 已有的同类模板 ``tpl``（序数、颜色已换成占位符）比较；模板 xhard0 没有的记「新增类型」，
  xhard0 有而该档没有的记「不再出现的类型」；模板相同、只是首次／后续不同（例如三目标时多一次 ``put down the container``）
  记「仅重复次数变化」，不算语义调整。
- 一格「有语义调整」＝ goal 或 subgoal 有新增／不再出现；否则为「仅参数变化」（数量、次数、颜色、序数不同，句式相同）。

每格附人工核读过原文的中文说明（``NOTES``）；逐局给出每条 goal、每段 subgoal 是否属于新增句式／类型，供页面高亮。
有调整的格另给「调整前后」对照（用户 2026-10-02「有语义调整的需要把语义调整前后的都写上」）：
``goal_before``／``goal_after``（xhard0 与该档全部 goal 句式，各附一句原文，标保留／去掉／新增）、
``sub_pairs``（不再出现的 subgoal 类型与新增类型按文字相似度配对，配不上的单列）、
``example``（xhard0 第 1 局与该档一局——优先含新增 subgoal 的局——的 task goal 与整条 subgoal 序列，新句式／新类型标出）。
末行打印 ``V8_SEMANTIC=PASS|FAIL tasks=<n> cells=<n> changed=<n> param_only=<n> note_missing=<n>``。

    uv run --no-sync python scripts/injection-dev/site/semantic_diff.py --site-dir artifacts/newtask-v8/site-eval
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ORD = r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|final|next|another"
COL = r"red|green|blue|yellow|purple|orange|cyan|magenta|white|black|pink|brown|gray|grey"
NUM = r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|once|twice|thrice|\d+"

#: 人工核读原文后的说明：(任务, 档) → 中文说明；未列出的「有调整」格判 note_missing。
_UNMASK3 = ("找 2 个目标容器变为找 3 个：goal 在两句之间多出「next pick up another container hiding the 〈颜色〉 cube」；"
            "subgoal 无新类型，只是「put down the container」多重复一次。")
NOTES = {
    ("PickHighlight", "xhard1"): "goal 两种措辞都重写：要求「one by one／one at a time」逐个抓取；第一种措辞补上结尾「finally press the button to stop」；"
                                 "xhard0 带拼写错误「highlighteted」的那句不再出现。subgoal「pick up the 〈序数〉 highlighted cube, which is 〈颜色〉」"
                                 "改为不再说颜色的「pick up the 〈序数〉 highlighted cube」，并在最后新增一次「press the button」。",
    ("VideoUnmask", "xhard2"): _UNMASK3, ("VideoUnmask", "xhard3"): _UNMASK3, ("VideoUnmask", "xhard4"): _UNMASK3,
    ("ButtonUnmask", "xhard2"): _UNMASK3, ("ButtonUnmask", "xhard3"): _UNMASK3, ("ButtonUnmask", "xhard4"): _UNMASK3,
    ("VideoUnmaskSwap", "xhard2"): _UNMASK3,
    ("ButtonUnmaskSwap", "xhard1"): "goal 不变；subgoal 新增「wait for the containers to finish swapping」（等待容器换位结束）。",
    ("ButtonUnmaskSwap", "xhard2"): _UNMASK3[:-1] + "；另新增 subgoal「wait for the containers to finish swapping」（等待容器换位结束）。",
    ("VideoRepick", "xhard1"): "goal 不再出现表示「再抓一次」的「again」措辞，一律写明次数（twice／two times 等），因为新档重抓至少 2 次；"
                               "subgoal 新增等待段「static」。",
    ("VideoPlaceButton", "xhard1"): "goal 只剩「where it was 〈序数〉 placed before／after the button was pressed」一种句式（必须指明第几次放置），"
                                    "xhard0 的「right before／after」「placed immediately」「previously placed」等措辞不再出现；"
                                    "subgoal「drop the cube onto table」改为「put the cube back to its original position」。",
    ("VideoPlaceOrder", "xhard1"): "goal 不变；subgoal「drop the cube onto table」改为「put the cube back to its original position」，且可多次出现。",
}
for _task in ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"):
    NOTES[(_task, "xhard2")] = NOTES[(_task, "xhard1")]


def norm_goal(text: str) -> str:
    t = text.lower()
    t = re.sub(rf"\b({ORD})\b", "〈序数〉", t)
    t = re.sub(rf"\b({COL})\b", "〈颜色〉", t)
    t = re.sub(rf"\b({NUM})\b", "〈数〉", t)
    t = re.sub(r"\btimes?\b", "次", t)
    for plural, single in (("cubes", "cube"), ("containers", "container"), ("buttons", "button")):
        t = re.sub(rf"\b{plural}\b", single, t)
    t = re.sub(r"〈数〉 〈颜色〉 cube(?:, 〈数〉 〈颜色〉 cube)*(?: and 〈数〉 〈颜色〉 cube)?", "〈数〉〈颜色〉cube列表", t)
    return re.sub(r"\s+", " ", t).strip()


def pair_subs(removed: list[dict], added: list[dict]) -> list[dict]:
    """不再出现的类型与新增类型按模板文字相似度配对（≥0.5 才配），其余单列（调整前或调整后为空）。"""
    pairs, left = [], list(added)
    for r in removed:
        best = max(left, key=lambda a: difflib.SequenceMatcher(None, r["tpl"], a["tpl"]).ratio(), default=None)
        if best is not None and difflib.SequenceMatcher(None, r["tpl"], best["tpl"]).ratio() >= 0.5:
            left.remove(best)
            pairs.append({"before": {"tpl": r["tpl"], "example": r["example"]}, "after": {"tpl": best["tpl"], "example": best["example"]}})
        else:
            pairs.append({"before": {"tpl": r["tpl"], "example": r["example"]}, "after": None})
    pairs += [{"before": None, "after": {"tpl": a["tpl"], "example": a["example"]}} for a in left]
    return pairs


def example(base: dict, eps: dict, g0: set, s0: set) -> dict:
    """前后各一局：xhard0 第 1 局；该档优先取含新增 subgoal 类型的第一局，否则第 1 局。"""
    first = lambda d: d[min(d, key=int)] if d else {}
    b_idx = min(base, key=int) if base else None
    a_idx = next((i for i in sorted(eps, key=int) if any(s["tpl"] not in s0 for s in eps[i].get("new") or [])),
                 min(eps, key=int) if eps else None)
    b, a = (base.get(b_idx) or {}), (eps.get(a_idx) or {})
    return {
        "before_ep": b_idx, "before_goal": b.get("goal") or [], "before_seq": [x["text"] for x in b.get("new") or []],
        "after_ep": a_idx, "after_goal": a.get("goal") or [], "after_goal_new": [norm_goal(x) not in g0 for x in a.get("goal") or []],
        "after_seq": [x["text"] for x in a.get("new") or []], "after_seq_new": [x["tpl"] not in s0 for x in a.get("new") or []],
    }


def build(sg: dict) -> tuple[dict, dict]:
    out = {"schema": "v8-semantic/1", "rule": __doc__.split("判定口径")[1].split("每格附")[0].strip(), "tasks": {}, "episodes": {}}
    stats = Counter()
    for task, tiers in sg["episodes"].items():
        base = tiers.get("xhard0") or {}
        g0 = {norm_goal(x) for ep in base.values() for x in ep.get("goal") or []}
        s0 = {s["tpl"] for ep in base.values() for s in ep.get("new") or []}
        k0 = {(s["tpl"], s["kind"]) for ep in base.values() for s in ep.get("new") or []}
        g0_ex = {norm_goal(x): x for ep in base.values() for x in ep.get("goal") or []}
        s0_ex = {s["tpl"]: s["text"] for ep in base.values() for s in ep.get("new") or []}
        tinfo, einfo = {}, {}
        for tier, eps in tiers.items():
            if tier == "xhard0":
                continue
            g_ex, s_ex, s_eps, rep = {}, {}, Counter(), Counter()
            einfo[tier] = {}
            for idx, ep in eps.items():
                goals = ep.get("goal") or []
                segs = ep.get("new") or []
                for x in goals:
                    g_ex.setdefault(norm_goal(x), x)
                seen = set()
                for s in segs:
                    s_ex.setdefault(s["tpl"], s["text"])
                    if s["tpl"] not in s0 and s["tpl"] not in seen:
                        s_eps[s["tpl"]] += 1
                        seen.add(s["tpl"])
                    if s["tpl"] in s0 and (s["tpl"], s["kind"]) not in k0:
                        rep[(s["tpl"], s["kind"])] += 1
                einfo[tier][idx] = {"goal": [norm_goal(x) not in g0 for x in goals],
                                    "sub": [s["tpl"] not in s0 for s in segs]}
            g1, s1 = set(g_ex), set(s_ex)
            cell = {
                "goal_added": [{"tpl": t, "example": g_ex[t]} for t in sorted(g1 - g0)],
                "goal_removed": [{"tpl": t, "example": g0_ex[t]} for t in sorted(g0 - g1)],
                "sub_added": [{"tpl": t, "example": s_ex[t], "episodes": s_eps[t], "of": len(eps)} for t in sorted(s1 - s0)],
                "sub_removed": [{"tpl": t, "example": s0_ex[t]} for t in sorted(s0 - s1)],
                "sub_repeat_only": [{"tpl": t, "kind": k, "segments": n} for (t, k), n in sorted(rep.items())],
            }
            changed = any(cell[k] for k in ("goal_added", "goal_removed", "sub_added", "sub_removed"))
            cell["changed"] = changed
            if changed:
                cell["goal_before"] = [{"tpl": t, "example": g0_ex[t], "state": "kept" if t in g1 else "removed"} for t in sorted(g0)]
                cell["goal_after"] = [{"tpl": t, "example": g_ex[t], "state": "kept" if t in g0 else "added"} for t in sorted(g1)]
                cell["sub_pairs"] = pair_subs(cell["sub_removed"], cell["sub_added"])
                cell["example"] = example(base, eps, g0, s0)
            cell["label"] = "有语义调整" if changed else "仅参数变化"
            note = NOTES.get((task, tier))
            if changed and not note:
                stats["note_missing"] += 1
            cell["note"] = note if changed else "goal 句式与 subgoal 类型与 xhard0 相同，只是数量、次数、颜色或序数等参数不同。"
            stats["cells"] += 1
            stats["changed" if changed else "param_only"] += 1
            tinfo[tier] = cell
        out["tasks"][task] = {"xhard0_goal_tpls": sorted(g0), "xhard0_sub_tpls": sorted(s0), "tiers": tinfo}
        out["episodes"][task] = einfo
    stats["tasks"] = len(out["tasks"])
    return out, dict(stats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    sg = json.loads((args.site_dir / "subgoals.json").read_text(encoding="utf-8"))
    out, st = build(sg)
    for task, info in out["tasks"].items():
        for tier, cell in info["tiers"].items():
            if cell["changed"]:
                print(f"# {task}/{tier} G+{len(cell['goal_added'])} G-{len(cell['goal_removed'])} "
                      f"S+{len(cell['sub_added'])} S-{len(cell['sub_removed'])}", flush=True)
    (args.site_dir / "semantic.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    ok = st.get("note_missing", 0) == 0 and st.get("tasks") == len(sg["episodes"])
    print(f"V8_SEMANTIC={'PASS' if ok else 'FAIL'} tasks={st.get('tasks', 0)} cells={st.get('cells', 0)} "
          f"changed={st.get('changed', 0)} param_only={st.get('param_only', 0)} note_missing={st.get('note_missing', 0)}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
