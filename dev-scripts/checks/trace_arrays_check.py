#!/usr/bin/env python3
"""完整数值闸门：逐局核对 ``trace.jsonl`` 与同局 ``arrays.npz``（``docs/plans/1006-stage3-interface-freeze.md`` 第四节）。

用法::

    python dev-scripts/checks/trace_arrays_check.py <局目录或其上级目录> [...] [--json <摘要输出>]

每个找到的 ``trace.jsonl`` 算一局；数组文件取 ``end.arrays.path``（相对轨迹目录），缺省同目录 ``arrays.npz``。逐局核：

- ``attempted_steps_missing``：每个 step 行（attempted 步，含 ``observed=false`` 的缺观测步）都要有
  ``exec_action__%05d``（键号 = step−1）；step 行 ``action`` 为空也算缺。
- ``observed_state_missing``：step 行 ``state`` 非空的观测步都要有 ``exec_state__%05d``；缺观测步（``state`` 为空）
  不得有该键（不补零），且步号须恰在 ``end.arrays.missing_state_steps`` 里。
- ``tampered``：数组的 dtype、shape、sha256（原始字节）须与 step 行 ``action``／``state`` 的 ``array_record`` 逐项相等。
- ``dtype_mixed``：同一局的 ``exec_action`` 键之间（以及 ``exec_state`` 键之间）dtype 不一致，如 float32／float64 混用。
- ``unreadable``：数组文件缺失（有 attempted 步时）、读不出、或 ``end.arrays.error`` 非空（收尾合并冲突等）。
- ``summary_mismatch``：``end.arrays`` 摘要（``action_keys``／``state_keys``／``missing_state_steps``）与轨迹不符。

判定行（末行）::

    TRACE_ARRAYS=PASS|FAIL episodes=<n> attempted_steps_missing=<n> observed_state_missing=<n> tampered=<n>
        dtype_mixed=<n> unreadable=<n> summary_mismatch=<n> extra_action_keys=<n>

前四项之外的计数任一大于 0 也判 FAIL（``extra_action_keys`` 只报不判：同目录录像器可能多记 strict-cap 之外的键）；
没有任何局判 FAIL。退出码：PASS 0，FAIL 1。只依赖标准库与 numpy。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ACTION_KEY = "exec_action__%05d"
STATE_KEY = "exec_state__%05d"
_ACTION_RE = re.compile(r"^exec_action__(\d{5})$")
_STATE_RE = re.compile(r"^exec_state__(\d{5})$")
COUNT_KEYS = ("attempted_steps_missing", "observed_state_missing", "tampered", "dtype_mixed", "unreadable",
              "summary_mismatch", "extra_action_keys")
FAIL_KEYS = COUNT_KEYS[:-1]


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _same(rec: dict | None, a: np.ndarray) -> bool:
    """数组与 trace 里的 ``array_record``（dtype、shape、原始字节 sha256）是否逐项相等。"""
    if not isinstance(rec, dict):
        return False
    return rec.get("dtype") == a.dtype.str and list(rec.get("shape") or []) == list(a.shape) and \
        rec.get("sha256") == _sha(a)


def check_episode(trace_path: str | Path) -> dict:
    """一局的计数与问题列表。"""
    trace_path = Path(trace_path)
    out: dict[str, Any] = {k: 0 for k in COUNT_KEYS}
    out.update(trace=str(trace_path), problems=[])
    probs: list[str] = out["problems"]
    try:
        rows = _read_rows(trace_path)
    except Exception as e:  # noqa: BLE001
        out["unreadable"] += 1
        probs.append(f"trace 读不出：{type(e).__name__}: {e}")
        return out
    steps = [r for r in rows if r.get("kind") == "step"]
    end = rows[-1] if rows and rows[-1].get("kind") == "end" else {}
    summ = end.get("arrays") if isinstance(end.get("arrays"), dict) else None
    rel = (summ or {}).get("path") or "arrays.npz"
    arr_path = trace_path.parent / rel
    if summ is not None and summ.get("error"):
        out["unreadable"] += 1
        probs.append(f"end.arrays.error：{summ['error']}")
    arrays: dict[str, np.ndarray] = {}
    if arr_path.is_file():
        try:
            with np.load(arr_path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001
            out["unreadable"] += 1
            probs.append(f"{arr_path.name} 读不出：{type(e).__name__}: {e}")
    elif steps:
        out["unreadable"] += 1
        probs.append(f"缺 {rel}（有 {len(steps)} 个 attempted 步）")

    no_state_steps: list[int] = []
    for r in steps:
        n = int(r["step"])
        ak, sk = ACTION_KEY % (n - 1), STATE_KEY % (n - 1)
        if r.get("action") is None or ak not in arrays:
            out["attempted_steps_missing"] += 1
            probs.append(f"step {n} 缺 {ak}")
        elif not _same(r.get("action"), arrays[ak]):
            out["tampered"] += 1
            probs.append(f"step {n} {ak} 与 trace 动作记录不符")
        if r.get("state") is None:
            no_state_steps.append(n)
            if sk in arrays:
                out["tampered"] += 1
                probs.append(f"缺观测步 {n} 不应有 {sk}（不补零）")
        elif sk not in arrays:
            out["observed_state_missing"] += 1
            probs.append(f"观测步 {n} 缺 {sk}")
        elif not _same(r.get("state"), arrays[sk]):
            out["tampered"] += 1
            probs.append(f"step {n} {sk} 与 trace 状态记录不符")

    for regex, name in ((_ACTION_RE, "exec_action"), (_STATE_RE, "exec_state")):
        dts = sorted({arrays[k].dtype.str for k in arrays if regex.match(k)})
        if len(dts) > 1:
            out["dtype_mixed"] += 1
            probs.append(f"{name} dtype 混用：{dts}")
    step_nums = {int(r["step"]) for r in steps}
    extra = [k for k in arrays if _ACTION_RE.match(k) and int(_ACTION_RE.match(k).group(1)) + 1 not in step_nums]
    out["extra_action_keys"] = len(extra)

    if summ is not None:
        n_obs = sum(1 for r in steps if r.get("state") is not None)
        n_act = sum(1 for r in steps if r.get("action") is not None)
        if summ.get("action_keys") != n_act or summ.get("state_keys") != n_obs or \
                sorted(summ.get("missing_state_steps") or []) != sorted(no_state_steps):
            out["summary_mismatch"] += 1
            probs.append(f"end.arrays 摘要与轨迹不符：{ {k: summ.get(k) for k in ('action_keys', 'state_keys')} } "
                         f"轨迹 action={n_act} state={n_obs} 缺状态步={no_state_steps}")
    out["episodes"] = 1
    return out


def find_traces(roots: list[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        p = Path(root)
        if p.is_file() and p.name == "trace.jsonl":
            found.add(p)
        elif p.is_dir():
            found.update(p.rglob("trace.jsonl"))
    return sorted(found)


def check(roots: list[str | Path]) -> dict:
    traces = find_traces(roots)
    total: dict[str, Any] = {k: 0 for k in COUNT_KEYS}
    eps = []
    for t in traces:
        r = check_episode(t)
        eps.append(r)
        for k in COUNT_KEYS:
            total[k] += int(r[k])
    ok = bool(traces) and all(total[k] == 0 for k in FAIL_KEYS)
    total.update(ok=ok, episodes=len(traces), details=eps)
    return total


def verdict_line(res: dict) -> str:
    return (f"TRACE_ARRAYS={'PASS' if res['ok'] else 'FAIL'} episodes={res['episodes']} "
            + " ".join(f"{k}={res[k]}" for k in COUNT_KEYS))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+", help="局目录（含 trace.jsonl）或其上级目录，递归查找")
    ap.add_argument("--json", default=None, help="把逐局问题与计数写成 JSON")
    ap.add_argument("--max-problems", type=int, default=20, help="打印的问题条数上限")
    args = ap.parse_args(argv)
    res = check(args.roots)
    shown = 0
    for ep in res["details"]:
        for p in ep["problems"]:
            if shown >= args.max_problems:
                break
            print(f"TRACE_ARRAYS_PROBLEM trace={ep['trace']} {p}")
            shown += 1
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(verdict_line(res), flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
