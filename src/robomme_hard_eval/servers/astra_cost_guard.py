"""3-tier Astra 费用守卫：常驻汇总两侧全部规划请求的 ``usage`` × 单价，到线即在 ``group_*/`` 下写 ``STOP.json``。

计划：``1003-oracle-subgoal-groundsg-eval-plan.md`` 第二部分 1.5「费用上限」与红线 R4。

- 扫描对象：每个 ``--root`` 下（递归）凡是与 ``request.json`` 同目录的 ``response.json``——这正是 Astra
  ``Planner`` 的 spool 布局 ``<spool>/<uuid>/{request.json,response.json}``；监视器的
  ``monitor_inputs/tNNNN/response.json`` 旁边没有 ``request.json``，不计。``ResponsesClient`` 写的
  ``response.json`` 里 ``usage`` 是 OpenAI Responses API 原样的 usage 对象。
- 计价：``(input_tokens - cached_tokens) × input + cached_tokens × cached_input + output_tokens × output``，
  单价单位「美元 / 百万 token」，由 ``--prices`` 配置文件给（不写死）。``reasoning_tokens`` 已含在
  ``output_tokens`` 里，只统计不另计价（配置 ``"reasoning_billed_separately": true`` 时另按 output 单价加计）。
  缺字段按 0 计，并计入 ``missing_fields`` 告警计数；``status=error`` 且无 ``usage`` 的响应计入 ``no_usage``。
- 累计账本：``--ledger`` 指向的 JSON 先读入（本机预检的账本拷到 GL 后接着累计），按请求 uuid 去重，
  每轮扫描后原子写回。
- 到线判定：``累计 + max(1, 在途数) × 一次请求最坏花费 > --cap``（在途＝有 ``api_started.json`` 而无
  ``response.json``；最坏花费＝已见最大输入 token × input 单价 + ``--max-output-tokens``(2048) × output 单价）。
  到线时在每个 ``group_*`` 目录（``--root`` 的祖先或子孙中名为 ``group_<数字>`` 的目录）写 ``STOP.json``，
  打印 ``ASTRA_COST=STOP usd=… cap=…``。Astra 自带两个停机口都读它：``ResponsesClient._send`` 每次发送前、
  驱动每局开跑前。
- 输出：累计变化时（及首轮）打印 ``ASTRA_COST usd=<累计> calls=<n> …``。

单价配置示例（``--prices``）::

    {"model": "gpt-6-astra", "unit": "usd_per_1m_tokens",
     "input": 1.25, "cached_input": 0.125, "output": 10.0,
     "source": "<OpenAI 价目页 URL>", "checked_at": "2026-10-05"}

（数值仅为格式示意，开工后查官方价目填写并记入 launch.md。）

第二阶段硬上限（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节 S3、审计第 19 条、红线 R6）：

- ``--cap`` 默认且最大为 ``HARD_CAP_USD``（本轮固定 5 美元），给更大的值直接拒绝启动（退出码 2）；
  ``--interval`` 默认 2 秒（须不超过心跳超时 10 秒的一半）。
- 缺 usage 视为超限：非 error 的规划响应缺 ``usage`` 或缺 ``input_tokens``／``output_tokens`` 即写 STOP
  （``reason=astra_usage_missing``）。``status=error`` 且无 usage 的响应（HTTP 错误、超时等，可能计费也可能
  未计费）按「一次请求最坏花费」计入投影（``unknown_charge_usd``）而不直接停——HTTP 错误本身不计费，按最坏价
  计入已是保守口径；同目录有 ``guard_refused.json`` 的（runner 发送前就拒发、从未外联）计 0。
- 心跳与同步预留：每轮把状态原子写进 ``--state``（默认 ``<账本名>.state.json``）：``heartbeat``（墙钟秒）、
  ``committed_usd``（已计 + error 无 usage 的最坏计入）、``counted``（已计入账本的请求 uuid）、``prices``、
  ``max_output_tokens``、``cap``、``stop``。``astra_hard_runner.py`` 的 ``GuardedResponsesClient`` 每次真正发送前
  同步读它，并在 ``<state>.reservations.json``（``fcntl`` 锁 ``<state>.lock``）里原子预留单次最坏费用；守卫把
  尚未计入账本的预留也算进投影。守卫正常退出时写 ``exited=true``；崩溃则心跳停更，runner 10 秒后即拒发。
- 局数硬上限 ``ASTRA_MAX_EPISODES=2``：runner 每局开跑前在同一预留文件的 ``episodes`` 列表里登记，跨 RUN 累计。
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

LEDGER_SCHEMA = "astra-cost-ledger/1"
GROUP_RE = re.compile(r"^group_\d+$")
#: 有效单价单位
PRICE_UNIT = "usd_per_1m_tokens"
#: 本轮固定的费用硬上限（美元；R6：Astra 本机 smoke ≤2 局、5 美元硬上限）；--cap 不得超过它
HARD_CAP_USD = 5.0
#: 守卫扫描间隔默认值（秒）
DEFAULT_INTERVAL_S = 2.0
#: runner 判定守卫失联的心跳超时（秒）；--interval 不得超过它的一半
HEARTBEAT_TIMEOUT_S = 10.0
#: Astra 局数硬上限（R6）；runner 经预留文件的 episodes 列表跨 RUN 计数
ASTRA_MAX_EPISODES = 2
STATE_SCHEMA = "astra-guard-state/1"
RESERVATIONS_SCHEMA = "astra-reservations/1"
#: runner 发送前拒发时写在请求目录里的标记；守卫见到它就把该请求计 0（从未外联）
REFUSED_MARKER = "guard_refused.json"


def load_prices(path: Path) -> dict:
    prices = json.loads(Path(path).read_text())
    if prices.get("unit", PRICE_UNIT) != PRICE_UNIT:
        raise ValueError(f"单价单位须为 {PRICE_UNIT}，实为 {prices.get('unit')!r}")
    for key in ("input", "output"):
        if not isinstance(prices.get(key), (int, float)) or prices[key] < 0:
            raise ValueError(f"单价配置缺少非负数值 {key!r}")
    prices.setdefault("cached_input", prices["input"])
    prices.setdefault("reasoning_billed_separately", False)
    return prices


def _get(usage: dict, *keys, warn: list) -> int:
    value = usage
    for key in keys:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            warn.append(".".join(keys))
            return 0
        value = value[key]
    try:
        return int(value)
    except (TypeError, ValueError):
        warn.append(".".join(keys))
        return 0


def price_usage(usage: dict, prices: dict) -> tuple[float, dict, list]:
    """一次请求的花费；返回 ``(usd, tokens, 缺失字段列表)``。"""
    warn: list = []
    tokens = {
        "input_tokens": _get(usage, "input_tokens", warn=warn),
        "output_tokens": _get(usage, "output_tokens", warn=warn),
        "cached_tokens": _get(usage, "input_tokens_details", "cached_tokens", warn=warn),
        "reasoning_tokens": _get(usage, "output_tokens_details", "reasoning_tokens", warn=warn),
    }
    cached = min(tokens["cached_tokens"], tokens["input_tokens"])
    usd = ((tokens["input_tokens"] - cached) * prices["input"] + cached * prices["cached_input"]
           + tokens["output_tokens"] * prices["output"]) / 1e6
    if prices.get("reasoning_billed_separately"):
        usd += tokens["reasoning_tokens"] * prices["output"] / 1e6
    return usd, tokens, warn


def new_ledger() -> dict:
    return {"schema": LEDGER_SCHEMA, "calls": {}}


def load_ledger(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return new_ledger()
    ledger = json.loads(Path(path).read_text())
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise ValueError(f"账本 schema 应为 {LEDGER_SCHEMA}：{path}")
    return ledger


def save_ledger(path: Path, ledger: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def scan(roots: list[Path], ledger: dict, prices: dict) -> dict:
    """扫描全部根，把新出现的规划响应计入账本；返回本轮统计（在途数、本轮新增等）。"""
    pending = 0
    added = 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for request in root.rglob("request.json"):
            call_dir = request.parent
            response = call_dir / "response.json"
            if not response.is_file():
                if (call_dir / "api_started.json").is_file():
                    pending += 1
                continue
            rid = call_dir.name
            if rid in ledger["calls"]:
                continue
            try:
                body = json.loads(response.read_text())
            except (OSError, ValueError):
                continue  # 正在写（Astra 用原子替换，理论上不会出现）；下一轮再读
            usage = body.get("usage")
            if (call_dir / REFUSED_MARKER).is_file() and body.get("status") == "error" and not isinstance(usage, dict):
                usd, tokens, warn = 0.0, {}, ["refused"]  # runner 发送前拒发：从未外联，不计费
            elif isinstance(usage, dict):
                usd, tokens, warn = price_usage(usage, prices)
            else:
                usd, tokens, warn = 0.0, {}, ["usage"]
            ledger["calls"][rid] = {"usd": usd, **tokens, "missing": warn, "status": body.get("status"),
                                    "path": str(response), "counted_at": time.time()}
            added += 1
    return {"pending": pending, "added": added}


def unknown_usage(call: dict) -> bool:
    """非 error 的响应却拿不到计费依据（usage 整体缺失，或缺 input_tokens／output_tokens）：视为超限。"""
    missing = call.get("missing") or []
    if call.get("status") == "error" or missing == ["refused"]:
        return False
    return "usage" in missing or "input_tokens" in missing or "output_tokens" in missing


def totals(ledger: dict) -> dict:
    calls = ledger["calls"].values()
    return {
        "usd": sum(c["usd"] for c in calls),
        "calls": len(ledger["calls"]),
        "max_input": max((c.get("input_tokens", 0) for c in calls), default=0),
        "missing_fields": sum(1 for c in calls if c.get("missing") and c["missing"] not in (["usage"], ["refused"])),
        "no_usage": sum(1 for c in calls if c.get("missing") == ["usage"]),
        "unknown_usage": sum(1 for c in calls if unknown_usage(c)),
        "error_no_usage": sum(1 for c in calls if c.get("missing") == ["usage"] and c.get("status") == "error"),
        "refused": sum(1 for c in calls if c.get("missing") == ["refused"]),
    }


def worst_request_usd(max_input: int, prices: dict, max_output_tokens: int) -> float:
    return (max_input * prices["input"] + max_output_tokens * prices["output"]) / 1e6


def group_dirs(roots: list[Path]) -> list[Path]:
    """当前全部 ``group_*`` 目录：各根自身及其祖先中名为 group_<n> 的，加上根下（≤3 层）的。"""
    found: set[Path] = set()
    for root in roots:
        root = Path(root).resolve()
        for candidate in (root, *root.parents):
            if GROUP_RE.match(candidate.name):
                found.add(candidate)
        if root.is_dir():
            for depth in ("*", "*/*", "*/*/*"):
                for path in root.glob(depth):
                    if path.is_dir() and GROUP_RE.match(path.name):
                        found.add(path)
    return sorted(found)


def write_stops(roots: list[Path], summary: dict) -> list[Path]:
    written = []
    reason = summary.get("reason") or "astra_cost_cap"
    for group in group_dirs(roots):
        stop = group / "STOP.json"
        if stop.exists():
            continue
        tmp = group / "STOP.json.tmp"
        tmp.write_text(json.dumps({**summary, "reason": reason, "time": time.time()}, indent=2) + "\n")
        os.replace(tmp, stop)
        written.append(stop)
    return written


def outstanding_reservations(reservations: dict | None, counted) -> float:
    """尚未计入账本的预留（runner 已预留、守卫还没扫到响应）之和；已释放（拒发）的不计。"""
    if not reservations:
        return 0.0
    counted = set(counted)
    return sum(float(r.get("usd", 0.0)) for rid, r in (reservations.get("reservations") or {}).items()
               if rid not in counted and r.get("state") in ("reserved", "sent"))


def cycle(roots: list[Path], ledger: dict, prices: dict, cap: float, max_output_tokens: int,
          reservations: dict | None = None) -> dict:
    """一轮：扫描、算累计与投影、到线写 STOP。返回本轮摘要。

    投影 = 已计 + error 无 usage 按最坏价计入 + max(未结预留之和, max(1, 在途数) × 一次最坏花费)；
    非 error 响应缺 usage 一律 STOP（``reason=astra_usage_missing``）。"""
    stats = scan(roots, ledger, prices)
    tot = totals(ledger)
    worst = worst_request_usd(tot["max_input"], prices, max_output_tokens)
    unknown_charge = tot["error_no_usage"] * worst
    committed = tot["usd"] + unknown_charge
    reserved = outstanding_reservations(reservations, ledger["calls"])
    projected = committed + max(reserved, max(1, stats["pending"]) * worst)
    summary = {"usd": round(tot["usd"], 6), "calls": tot["calls"], "pending": stats["pending"],
               "worst_request_usd": round(worst, 6), "projected_usd": round(projected, 6), "cap": cap,
               "missing_fields": tot["missing_fields"], "no_usage": tot["no_usage"], "added": stats["added"],
               "unknown_usage": tot["unknown_usage"], "unknown_charge_usd": round(unknown_charge, 6),
               "committed_usd": round(committed, 6), "reserved_usd": round(reserved, 6), "refused": tot["refused"]}
    summary["stop"] = projected > cap or tot["unknown_usage"] > 0
    summary["reason"] = "astra_usage_missing" if tot["unknown_usage"] > 0 else "astra_cost_cap"
    summary["stops_written"] = [str(p) for p in write_stops(roots, summary)] if summary["stop"] else []
    return summary


# ── 心跳状态与同步预留（runner 与守卫共用的协议） ───────────────────────────

def default_state_path(ledger_path: Path) -> Path:
    ledger_path = Path(ledger_path)
    return ledger_path.with_name(ledger_path.stem + ".state.json")


def reservations_path(state_path: Path) -> Path:
    state_path = Path(state_path)
    return state_path.with_name(state_path.name + ".reservations.json")


def lock_path(state_path: Path) -> Path:
    state_path = Path(state_path)
    return state_path.with_name(state_path.name + ".lock")


@contextlib.contextmanager
def locked(state_path: Path):
    """预留文件的排他锁（``fcntl.flock``）；runner 预留与守卫一轮扫描都在锁内，保证「读—判—写」原子。"""
    path = lock_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_reservations(state_path: Path) -> dict:
    path = reservations_path(state_path)
    if not path.is_file():
        return {"schema": RESERVATIONS_SCHEMA, "reservations": {}, "episodes": []}
    doc = json.loads(path.read_text())
    if doc.get("schema") != RESERVATIONS_SCHEMA:
        raise ValueError(f"预留文件 schema 应为 {RESERVATIONS_SCHEMA}：{path}")
    doc.setdefault("reservations", {})
    doc.setdefault("episodes", [])
    return doc


def save_reservations(state_path: Path, doc: dict) -> None:
    atomic_write_json(reservations_path(state_path), doc)


def write_state(state_path: Path, summary: dict, ledger: dict, prices: dict, max_output_tokens: int, *,
                exited: bool = False, clock=time.time) -> dict:
    state = {"schema": STATE_SCHEMA, "pid": os.getpid(), "heartbeat": clock(), "exited": bool(exited),
             "cap": float(summary["cap"]), "hard_cap": HARD_CAP_USD, "committed_usd": summary["committed_usd"],
             "usd": summary["usd"], "projected_usd": summary["projected_usd"], "stop": bool(summary["stop"]),
             "reason": summary.get("reason"), "unknown_usage": summary.get("unknown_usage", 0),
             "counted": sorted(ledger["calls"]), "prices": prices, "max_output_tokens": int(max_output_tokens),
             "max_episodes": ASTRA_MAX_EPISODES}
    atomic_write_json(Path(state_path), state)
    return state


def guard_round(roots: list[Path], ledger: dict, prices: dict, cap: float, max_output_tokens: int,
                state_path: Path, *, clock=time.time) -> dict:
    """守卫一轮：锁内读预留 → 扫描与投影（到线写 STOP）→ 写心跳状态。``main`` 与测试共用。"""
    with locked(state_path):
        reservations = load_reservations(state_path)
        summary = cycle(roots, ledger, prices, cap, max_output_tokens, reservations)
        write_state(state_path, summary, ledger, prices, max_output_tokens, clock=clock)
    return summary


def check_cap(cap: float) -> float:
    cap = float(cap)
    if not 0 < cap <= HARD_CAP_USD:
        raise ValueError(f"ASTRA_COST_BLOCKED --cap={cap:g} 超出本轮硬上限 {HARD_CAP_USD:g} 美元（R6）")
    return cap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3-tier Astra 费用守卫（每 --interval 秒汇总一次）")
    parser.add_argument("--root", action="append", required=True,
                        help="spool 根（可多个，本机预检与 GL 各一个）；也可直接给 group_*/ 或其上层目录")
    parser.add_argument("--prices", required=True, help="单价配置 JSON（美元/百万 token）")
    parser.add_argument("--ledger", required=True, help="累计账本 JSON：存在则先读入，每轮写回")
    parser.add_argument("--cap", type=float, default=HARD_CAP_USD,
                        help=f"美元；默认且最大 {HARD_CAP_USD:g}（本轮硬上限，R6）")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help=f"秒；不得超过心跳超时 {HEARTBEAT_TIMEOUT_S:g} 秒的一半")
    parser.add_argument("--state", default=None, help="心跳状态文件（runner 发送前同步读）；默认 <账本名>.state.json")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Astra 每次请求的 max_output_tokens")
    parser.add_argument("--once", action="store_true", help="只跑一轮（测试与手动核账用）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_cap(args.cap)
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    if not 0 < args.interval <= HEARTBEAT_TIMEOUT_S / 2:
        print(f"ASTRA_COST_BLOCKED --interval={args.interval:g} 须在 (0, {HEARTBEAT_TIMEOUT_S / 2:g}] 秒内",
              file=sys.stderr, flush=True)
        return 2
    prices = load_prices(Path(args.prices))
    ledger_path = Path(args.ledger)
    state_path = Path(args.state) if args.state else default_state_path(ledger_path)
    ledger = load_ledger(ledger_path)
    ledger["prices"] = prices
    roots = [Path(r) for r in args.root]
    running = {"go": True}

    def _stop(signum, frame):  # noqa: ARG001
        running["go"] = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    last = None
    stop_seen = False
    summary = None
    while running["go"]:
        summary = guard_round(roots, ledger, prices, args.cap, args.max_output_tokens, state_path)
        save_ledger(ledger_path, ledger)
        key = (summary["calls"], summary["pending"], summary["usd"])
        if key != last:
            print(f"ASTRA_COST usd={summary['usd']:.4f} calls={summary['calls']} pending={summary['pending']} "
                  f"projected={summary['projected_usd']:.4f} cap={args.cap:g}", flush=True)
            if summary["missing_fields"] or summary["no_usage"]:
                print(f"ASTRA_COST_WARN missing_fields={summary['missing_fields']} no_usage={summary['no_usage']}",
                      flush=True)
            last = key
        if summary["stop"] and (summary["stops_written"] or not stop_seen):
            print(f"ASTRA_COST=STOP usd={summary['usd']:.4f} cap={args.cap:g} projected={summary['projected_usd']:.4f} "
                  f"stop_files={len(summary['stops_written'])} reason={summary['reason']}", flush=True)
            stop_seen = True
        if args.once:
            break
        deadline = time.monotonic() + args.interval
        while running["go"] and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    if summary is not None and not args.once:
        # 正常退出：写 exited=true，runner 立即拒发（不等心跳超时）
        with locked(state_path):
            write_state(state_path, summary, ledger, prices, args.max_output_tokens, exited=True)
        print(f"ASTRA_COST_GUARD_EXIT state={state_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
