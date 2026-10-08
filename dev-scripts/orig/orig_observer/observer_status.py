#!/usr/bin/env python3
"""原侧观测器的完整性判定：写 ``observer-status.json`` 并打印末行 ``OBSERVER_COMPLETE=…``（计划第二部分一节 S7，
审计第 13 条「透明检查失败独立传播、日志封口后再对账」）。

只读判定，**与评估退出码分开**：本脚本恒以 0 退出，启动器的 ``EXIT_CODE=`` 保持原语义（任务成绩不因观测器失败
而改变）；观测器失败只体现在 ``OBSERVER_COMPLETE=FAIL`` 与 ``observer-status.json``。

判定（全部满足才 PASS）：
- 透明性报告（仅 FrameSamp+Modulation）：``--report`` 文件存在且可解析（否则 ``report=missing``）、检查器退出码为 0 或 1（其余视为
  崩溃，``report=crashed``）、``OBSERVER_TRANSPARENT=PASS``（含清单身份覆盖与代理日志封口）、``mismatch=0``；
- 代理收尾（仅 FrameSamp+Modulation）：``--sealed yes``（代理写了 ``proxy-<pid>.done``）且未被 ``kill -9``（``--proxy-force-killed 0``）；
- 钩子异常：``<rec_root>/hook-errors.jsonl`` 行数为 0；所有完整 trace 的 ``end.observer_hook_errors`` 为 0；
- attempt 映射：给了 ``--episode-log`` 时，``orig_results_adapter.build`` 判 ``ORIG_ATTEMPTS=PASS``（同时写
  ``<rec_root>/orig-attempts.json``）。

末行：``OBSERVER_COMPLETE=PASS|FAIL episodes=<完整 trace 局数> conns=<客户端连接数> mismatch=<n> hook_errors=<n>
report=<ok|missing|crashed>``（SimpleMemVLA 无代理，``conns=0 mismatch=0 report=ok``）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_OBS_DIR = str(Path(__file__).resolve().parent)
if _OBS_DIR not in sys.path:
    sys.path.append(_OBS_DIR)
import orig_results_adapter as ORA  # noqa: E402


def _lines(p: Path) -> int:
    try:
        return sum(1 for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
    except OSError:
        return 0


def trace_summary(rec_root: Path) -> dict:
    complete, incomplete, hook_sum = 0, 0, 0
    for d in sorted(rec_root.glob("*.a*")):
        if not (d.is_dir() and ORA.EP_DIR_RE.match(d.name)):
            continue
        info = ORA._trace_end(d / "trace.jsonl") if (d / "trace.jsonl").exists() else None
        if info is None:
            incomplete += 1
            continue
        complete += 1
        v = info["end"].get("observer_hook_errors")
        hook_sum += int(v) if isinstance(v, int) else 1  # 缺字段也算一次异常（C11 必写）
    return {"episodes": complete, "incomplete": incomplete, "trace_hook_errors": hook_sum}


def evaluate(rec_root: str | Path, *, policy: str, report: str | None = None, checker_rc: int | None = None,
             sealed: str = "yes", proxy_force_killed: int = 0, episode_logs: list[str] | None = None) -> dict:
    rec_root = Path(rec_root)
    policy = ORA.official_defs().canonical_policy(policy)  # 历史调用方的旧标签只读兼容；写出只用官方名
    reasons: list[str] = []
    ts = trace_summary(rec_root)
    hook_file = _lines(rec_root / "hook-errors.jsonl")
    hook_errors = max(hook_file, ts["trace_hook_errors"])
    if hook_errors:
        reasons.append(f"hook_errors={hook_errors}")
    conns, mismatch, rep_state, transparent = 0, 0, "ok", None
    if policy == "perceptual-framesamp-modul":
        rep = None
        if report and Path(report).is_file():
            try:
                rep = json.loads(Path(report).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rep = None
                rep_state = "crashed"
        else:
            rep_state = "missing"
        if checker_rc is not None and checker_rc not in (0, 1):
            rep_state = "crashed"
        if rep_state == "ok" and not isinstance(rep, dict):
            rep_state = "crashed"
        if rep_state == "ok":
            conns = int(rep.get("conns", 0) or 0)
            mismatch = int(rep.get("mismatch", 0) or 0)
            transparent = rep.get("OBSERVER_TRANSPARENT")
            if transparent != "PASS":
                reasons.append(f"OBSERVER_TRANSPARENT={transparent}")
            if mismatch:
                reasons.append(f"mismatch={mismatch}")
        else:
            reasons.append(f"report={rep_state}")
        if sealed != "yes":
            reasons.append(f"sealed={sealed}")
        if int(proxy_force_killed):
            reasons.append("proxy_force_killed=1")
    attempts = None
    if episode_logs:
        res = ORA.build(ORA.read_log(episode_logs), rec_root, policy=policy)
        (rec_root / "orig-attempts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True)
                                                     + "\n", encoding="utf-8")
        attempts = ORA.verdict_line(res)
        if res["ORIG_ATTEMPTS"] != "PASS":
            reasons.append("orig_attempts=FAIL")
    ok = not reasons
    return {"OBSERVER_COMPLETE": "PASS" if ok else "FAIL", "policy": policy, "episodes": ts["episodes"],
            "incomplete_traces": ts["incomplete"], "conns": conns, "mismatch": mismatch, "hook_errors": hook_errors,
            "report": rep_state, "transparent": transparent, "sealed": sealed,
            "proxy_force_killed": int(proxy_force_killed), "checker_rc": checker_rc, "orig_attempts": attempts,
            "reasons": reasons}


def verdict_line(st: dict) -> str:
    return (f"OBSERVER_COMPLETE={st['OBSERVER_COMPLETE']} episodes={st['episodes']} conns={st['conns']} "
            f"mismatch={st['mismatch']} hook_errors={st['hook_errors']} report={st['report']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="原侧观测器完整性判定（不影响评估退出码）")
    ap.add_argument("--rec-root", required=True)
    ap.add_argument("--policy", required=True, choices=("smvla", "perceptual-framesamp-modul"))
    ap.add_argument("--report", default=None, help="transparency_check.py --out 写出的报告（FrameSamp+Modulation）")
    ap.add_argument("--checker-rc", type=int, default=None, help="transparency_check.py 的退出码（FrameSamp+Modulation）")
    ap.add_argument("--sealed", default="yes", choices=("yes", "no", "timeout"))
    ap.add_argument("--proxy-force-killed", type=int, default=0)
    ap.add_argument("--episode-log", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="缺省 <rec_root>/observer-status.json")
    a = ap.parse_args(argv)
    try:
        st = evaluate(a.rec_root, policy=a.policy, report=a.report, checker_rc=a.checker_rc, sealed=a.sealed,
                      proxy_force_killed=a.proxy_force_killed, episode_logs=a.episode_log)
    except Exception as e:  # noqa: BLE001 判定本身出错也只判 FAIL，不改评估退出码
        st = {"OBSERVER_COMPLETE": "FAIL", "policy": a.policy, "episodes": 0, "conns": 0, "mismatch": 0,
              "hook_errors": 0, "report": "crashed", "reasons": [f"observer_status_error={type(e).__name__}: {e}"]}
    out = Path(a.out) if a.out else Path(a.rec_root) / "observer-status.json"
    try:
        out.write_text(json.dumps(st, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"OBSERVER_STATUS_WRITE_ERROR {e}", flush=True)
    print(verdict_line(st), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
