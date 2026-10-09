#!/usr/bin/env python3
"""语言账本验收 ``LANG_IO``（1006 计划第二部分八.11「验收」、接口冻结说明第五节）。纯 CPU、只用标准库、只读。

读每个局目录的 ``trace.jsonl`` 与同目录 ``language.jsonl``（``trace_writer.LanguageLog`` 写的行：``call_open``／
``message``／``call_close``／``reuse``），逐局核：

1. 每个执行步（trace ``step`` 行）能追溯到来源调用：``source_call_id`` 指向本局已打开的调用、且该调用发生在这一步执行之前
   （调用的 ``step`` ≤ 该步 − 1）；没有 ``source_call_id`` 的步须有 ``reuse`` 行（``step`` = 该步 − 1）指向已有调用。
   追溯不到计 ``unresolved_steps``；trace 有执行步却没有 ``language.jsonl`` 计 ``language_missing``，其执行步全部计入
   ``unresolved_steps``。
2. 每个调用恰以 ``reply``／``error``／``cancelled`` 之一关闭一次；未关闭计 ``open_calls``。
3. 有序性（``order_bad``）：消息必须在所属调用打开之后、关闭之前；``message_index`` 在调用内从 0 连续递增（文件顺序）；
   ``system`` 只能是第 0 条；``dir=out`` 之后不能再有 ``dir=in``；``call_id`` 不重复；``reuse`` 指向已有调用。
4. 每个附图引用 ``images[].raw_sha256``（及 ``sources`` 里的 ``raw_sha256``）能在本局 trace 的帧哈希里找到：
   ``phase=demo`` 查 demo 段、``phase=exec`` 查执行步，``cam`` 区分 front／wrist；找不到计 ``image_ref_unresolved``
   （引用错 attempt 的账本在这里暴露）。帧号口径（各客户端一致）：``phase=demo`` 的 ``frame_idx=i`` 是 demo 行
   ``<cam>_sha256[i]``；``phase=exec`` 的 ``frame_idx=k`` 是已执行步数——k≥1 对 trace ``step=k`` 行（第 k 步执行后的
   观测），k=0 是 reset 后的初始帧，即 demo 行 ``<cam>_sha256[-1]``（演示段末帧）。带整数 ``frame_idx`` 的引用按帧号
   精确比对那一帧；没有帧号（拼图本身、旧账本）的退回「该 phase／cam 的任一帧」（exec 集合含演示段末帧）。
5. 同一步多一次调用（``extra_calls``）：trace 有 ``response`` 行时，每一步以 ``reply`` 关闭的 ``action_model`` 调用数必须
   等于该步的 ``response`` 行数。
6. 缺回复（``reply_missing``）：``subgoal_model``／``planner``／``monitor`` 调用以 ``reply`` 关闭却没有 ``dir=out`` 消息。
7. 重问（``retry_mismatch``）：同一模型的 ``retry`` 只能是 0 或上一次 + 1（≤2）；局目录里有归档的
   ``ep*_MemER_log.jsonl`` 时，其请求行数必须等于 ``subgoal_model`` 调用数、``{"response": …}`` 行数必须等于以 ``reply``
   关闭的 ``subgoal_model`` 调用数。
8. 外壳最终文字（``server_text_empty``）：该局任一 ``action_model`` 调用关闭时带了 ``server_final_text``（说明该路线有服务
   外壳），或给了 ``--require-server-text``，则每个以 ``reply`` 关闭的 ``action_model`` 调用都必须带非空 ``server_final_text``。
9. system 一致（``system_inconsistent``）：所有被核的局里，同一（模型, trace header ``route``）的 ``system`` 消息原文必须
   相同（按两者分组后与组内多数不同的消息逐条计数；同一 ``--root`` 混放 QwenVL 与 MemER 等不同路线时各比各的）；``--expect-system <model>=<sha256>`` 时另须等于给定哈希（文字 UTF-8 字节的 sha256）。

判定行（末行）::

    LANG_IO=PASS|FAIL episodes=<n> unresolved_steps=<n> open_calls=<n> image_ref_unresolved=<n> order_bad=<n>
        extra_calls=<n> reply_missing=<n> retry_mismatch=<n> server_text_empty=<n> system_inconsistent=<n>
        language_missing=<n> bad_rows=<n>

任一计数 > 0 或 ``episodes=0`` 即 FAIL，退出 1。用法::

    python dev-scripts/checks/lang_io_check.py <局目录>... | --root <根> [--require-server-text]
        [--expect-system subgoal_model=<sha256> ...] [--out-json <报告>]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

STATUSES = ("reply", "error", "cancelled")
REPLY_MODELS = ("subgoal_model", "planner", "monitor")
COUNT_KEYS = ("unresolved_steps", "open_calls", "image_ref_unresolved", "order_bad", "extra_calls", "reply_missing",
              "retry_mismatch", "server_text_empty", "system_inconsistent", "language_missing", "bad_rows")
LANG_FILE = "language.jsonl"


def _jsonl(path: Path) -> tuple[list[dict], int]:
    rows, bad = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            rows.append(r)
        else:
            bad += 1
    return rows, bad


def find_episodes(root: Path) -> list[Path]:
    out = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if "trace.jsonl" in files:
            out.append(Path(cur))
    return sorted(out)


def frame_sets(trace: list[dict]) -> dict:
    """``{(phase, cam): {sha256}}``（无帧号引用用）与 ``{(phase, cam, frame_idx): sha256}``（有帧号引用用）合在一个
    字典里：phase ∈ demo／exec，cam ∈ front／wrist。exec 帧号 0 = 演示段末帧（reset 后初始帧），k≥1 = ``step=k`` 行。"""
    sets: dict = defaultdict(set)
    for r in trace:
        if r.get("kind") == "demo":
            for cam in ("front", "wrist"):
                shas = [x if isinstance(x, str) else None for x in r.get(f"{cam}_sha256") or []]
                sets[("demo", cam)].update(x for x in shas if x)
                for i, x in enumerate(shas):
                    if x:
                        sets[("demo", cam, i)] = x
                if shas and shas[-1]:
                    sets[("exec", cam)].add(shas[-1])
                    sets[("exec", cam, 0)] = shas[-1]
        elif r.get("kind") == "step":
            k = r.get("step")
            for cam in ("front", "wrist"):
                v = r.get(f"{cam}_sha256")
                if isinstance(v, str):
                    sets[("exec", cam)].add(v)
                    if isinstance(k, int) and k >= 1:
                        sets[("exec", cam, k)] = v
    return sets


def image_refs(img: dict) -> list[tuple[str | None, str | None, int | None, str]]:
    """一个图引用里要核的 (phase, cam, frame_idx, sha256)：本身的 ``raw_sha256`` 与 ``sources`` 里每个来源帧。
    来源帧是对象时用它自己的 ``frame_idx``（缺省为 None，不继承图本身的帧号——拼图本身 ``frame_idx=None``）；
    来源是 64 位哈希字符串时沿用图本身的 phase／cam、帧号 None。"""
    out = []
    if isinstance(img.get("raw_sha256"), str):
        out.append((img.get("phase"), img.get("cam"), img.get("frame_idx"), img["raw_sha256"]))
    for s in img.get("sources") or []:
        if isinstance(s, dict) and isinstance(s.get("raw_sha256"), str):
            out.append((s.get("phase", img.get("phase")), s.get("cam", img.get("cam")), s.get("frame_idx"),
                        s["raw_sha256"]))
        elif isinstance(s, str) and len(s) == 64:
            out.append((img.get("phase"), img.get("cam"), None, s))
    return out


def ref_found(sets: dict, phase: str | None, cam: str | None, sha: str, frame_idx: int | None = None) -> bool:
    """有整数帧号且 phase 明确时按帧号精确比对；否则查该 phase／cam 的任一帧。"""
    phases = (phase,) if phase in ("demo", "exec") else ("demo", "exec")
    cams = (cam,) if cam in ("front", "wrist") else ("front", "wrist")
    if isinstance(frame_idx, int) and not isinstance(frame_idx, bool) and phase in ("demo", "exec"):
        return any(sets.get((phase, c, frame_idx)) == sha for c in cams)
    return any(sha in sets.get((p, c), ()) for p in phases for c in cams)


def memer_log_counts(ep: Path) -> tuple[int, int] | None:
    """归档的 ``ep*_MemER_log.jsonl``（局目录内递归）→ (请求行数, 回复行数)；没有返回 None。上游每次提问先追加请求
    字典、收到回复后追加 ``{"response": …}``（``api_memer.py``），两种行可能写在同一物理行，故按 JSON 对象切分。"""
    logs = sorted(ep.rglob("ep*_MemER_log.jsonl"))
    if not logs:
        return None
    req = resp = 0
    dec = json.JSONDecoder()
    for p in logs:
        text = p.read_text(encoding="utf-8")
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            try:
                obj, i = dec.raw_decode(text, i)
            except ValueError:
                break
            if isinstance(obj, dict) and set(obj) == {"response"}:
                resp += 1
            else:
                req += 1
    return req, resp


def check_episode(ep: Path, *, require_server_text: bool = False) -> dict:
    c = Counter({k: 0 for k in COUNT_KEYS})
    problems: list[str] = []
    systems: list[tuple[str, str]] = []

    def bad(kind: str, detail: str, n: int = 1) -> None:
        c[kind] += n
        if len(problems) < 50:
            problems.append(f"{kind}:{detail}")

    trace, tb = _jsonl(ep / "trace.jsonl")
    c["bad_rows"] += tb
    route = next((r for r in trace if r.get("kind") == "header"), {}).get("route")  # system 一致按路线分组
    steps = [r for r in trace if r.get("kind") == "step"]
    responses = Counter(int(r["step"]) for r in trace if r.get("kind") == "response" and isinstance(r.get("step"), int))
    lp = ep / LANG_FILE
    if not lp.is_file():
        if steps:
            bad("language_missing", str(lp))
            bad("unresolved_steps", "无 language.jsonl", len(steps))
        return {"dir": str(ep), "counts": dict(c), "problems": problems, "systems": systems, "route": route}
    lang, lb = _jsonl(lp)
    if lb:
        bad("bad_rows", f"language.jsonl 坏行 {lb}", lb)
    sets = frame_sets(trace)
    calls: dict = {}
    reuse_by_step: dict = {}
    retry_prev: dict = {}
    for r in lang:
        kind = r.get("kind")
        cid = r.get("call_id")
        if kind == "call_open":
            if cid in calls:
                bad("order_bad", f"call_id 重复 {cid}")
                continue
            model = r.get("model")
            retry = r.get("retry", 0)
            prev = retry_prev.get(model)
            if not isinstance(retry, int) or retry < 0 or retry > 2 or (retry > 0 and prev != retry - 1):
                bad("retry_mismatch", f"{cid} model={model} retry={retry} 上一次={prev}")
            retry_prev[model] = retry
            calls[cid] = {"model": model, "step": r.get("step"), "msgs": [], "closed": None, "close": None,
                          "seen_out": False}
        elif kind == "message":
            call = calls.get(cid)
            if call is None or call["closed"] is not None:
                bad("order_bad", f"消息不在打开的调用内 call_id={cid}")
                continue
            mi = r.get("message_index")
            if mi != len(call["msgs"]):
                bad("order_bad", f"{cid} message_index={mi} 期望 {len(call['msgs'])}")
            if r.get("role") == "system" and len(call["msgs"]) != 0:
                bad("order_bad", f"{cid} system 不是第 0 条")
            if r.get("dir") == "in" and call["seen_out"]:
                bad("order_bad", f"{cid} 回复之后又有输入")
            if r.get("dir") == "out":
                call["seen_out"] = True
            if r.get("role") == "system":
                systems.append((str(call["model"]), r.get("text") if isinstance(r.get("text"), str)
                                else json.dumps(r.get("text"), ensure_ascii=False, sort_keys=True)))
            call["msgs"].append(r)
            for img in r.get("images") or []:
                if not isinstance(img, dict):
                    bad("image_ref_unresolved", f"{cid} 图引用不是对象")
                    continue
                refs = image_refs(img)
                if not refs:
                    bad("image_ref_unresolved", f"{cid} 图引用缺 raw_sha256")
                for phase, cam, idx, sha in refs:
                    if not ref_found(sets, phase, cam, sha, idx):
                        bad("image_ref_unresolved", f"{cid} {phase}/{cam}#{idx} {sha[:12]}")
        elif kind == "call_close":
            call = calls.get(cid)
            if call is None or call["closed"] is not None:
                bad("order_bad", f"关闭未打开或已关闭的调用 {cid}")
                continue
            if r.get("status") not in STATUSES:
                bad("order_bad", f"{cid} status={r.get('status')}")
            call["closed"] = r.get("status")
            call["close"] = r
        elif kind == "reuse":
            if r.get("reused_call_id") not in calls:
                bad("order_bad", f"reuse 指向不存在的调用 {r.get('reused_call_id')}")
            else:
                reuse_by_step[r.get("step")] = r.get("reused_call_id")
        else:
            bad("bad_rows", f"未知行 kind={kind}")
    for cid, call in calls.items():
        if call["closed"] is None:
            bad("open_calls", str(cid))
        elif call["closed"] == "reply" and call["model"] in REPLY_MODELS and not call["seen_out"]:
            bad("reply_missing", f"{cid} model={call['model']}")
    # 执行步追溯
    for st in steps:
        s = st.get("step")
        sid = st.get("source_call_id")
        call = calls.get(sid) if sid is not None else None
        if call is not None:
            if isinstance(call["step"], int) and isinstance(s, int) and call["step"] > s - 1:
                bad("unresolved_steps", f"第 {s} 步指向之后才发生的调用 {sid}")
            continue
        if sid is None and isinstance(s, int) and reuse_by_step.get(s - 1) in calls:
            continue
        bad("unresolved_steps", f"第 {s} 步 source_call_id={sid}")
    # 同一步多一次调用
    if responses:
        per_step = Counter(call["step"] for call in calls.values()
                           if call["model"] == "action_model" and call["closed"] == "reply")
        for s in sorted(set(per_step) | set(responses), key=str):
            if per_step.get(s, 0) != responses.get(s, 0):
                bad("extra_calls", f"第 {s} 步 action_model 回复 {per_step.get(s, 0)} 次，trace response {responses.get(s, 0)} 行",
                    abs(per_step.get(s, 0) - responses.get(s, 0)))
    # 外壳最终文字
    am = [call for call in calls.values() if call["model"] == "action_model"]
    has_shell = require_server_text or any((call["close"] or {}).get("server_final_text") is not None for call in am)
    if has_shell:
        for cid, call in calls.items():
            if call["model"] == "action_model" and call["closed"] == "reply":
                t = (call["close"] or {}).get("server_final_text")
                if not (isinstance(t, str) and t.strip()):
                    bad("server_text_empty", str(cid))
    # MemER 归档日志
    mc = memer_log_counts(ep)
    if mc is not None:
        sg = [call for call in calls.values() if call["model"] == "subgoal_model"]
        req, resp = mc
        replied = sum(1 for call in sg if call["closed"] == "reply")
        if req != len(sg) or resp != replied:
            bad("retry_mismatch", f"MemER 日志 请求 {req}／回复 {resp}，语言账本 subgoal_model 调用 {len(sg)}／回复 {replied}")
    return {"dir": str(ep), "counts": dict(c), "problems": problems, "systems": systems, "route": route}


def check(episodes: list[Path], *, require_server_text: bool = False, expect_system: dict | None = None) -> dict:
    per = [check_episode(ep, require_server_text=require_server_text) for ep in episodes]
    total = Counter({k: 0 for k in COUNT_KEYS})
    for r in per:
        total.update(r["counts"])
    # system 一致：按（模型, trace header route）分组，组内全部 system 原文与组内多数相同；给了哈希时另须等于它
    by_group: dict = defaultdict(list)
    for r in per:
        for model, text in r["systems"]:
            by_group[(model, r.get("route"))].append((r, text))
    sys_problems = []
    for (model, route), items in by_group.items():
        cnt = Counter(t for _, t in items)
        major = cnt.most_common(1)[0][0]
        want = (expect_system or {}).get(model)
        for r, t in items:
            sha = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if t != major or (want is not None and sha != want):
                total["system_inconsistent"] += 1
                sys_problems.append({"dir": r["dir"], "model": model, "route": route, "sha256": sha})
    for r in per:
        r.pop("systems")
        r.pop("route", None)
    ok = len(per) > 0 and all(total[k] == 0 for k in COUNT_KEYS)
    line = (f"LANG_IO={'PASS' if ok else 'FAIL'} episodes={len(per)} "
            + " ".join(f"{k}={total[k]}" for k in COUNT_KEYS))
    return {"verdict": "PASS" if ok else "FAIL", "line": line, "counts": dict(total), "episodes": per,
            "system_problems": sys_problems}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="*", type=Path)
    ap.add_argument("--root", type=Path, action="append", default=[])
    ap.add_argument("--require-server-text", action="store_true")
    ap.add_argument("--expect-system", action="append", default=[], help="<model>=<sha256>，可重复")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)
    eps = list(args.episodes)
    for root in args.root:
        eps += find_episodes(root)
    if not eps and not args.root:
        ap.error("需要局目录或 --root")
    expect = {}
    for item in args.expect_system:
        model, _, sha = item.partition("=")
        if not model or len(sha) != 64:
            ap.error(f"--expect-system 格式为 <model>=<sha256>：{item}")
        expect[model] = sha.lower()
    res = check(sorted(set(eps)), require_server_text=args.require_server_text, expect_system=expect)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for r in res["episodes"]:
        if r["problems"]:
            print(f"LANG_IO_EPISODE dir={r['dir']} problems={json.dumps(r['problems'][:5], ensure_ascii=False)}",
                  flush=True)
    print(res["line"], flush=True)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
