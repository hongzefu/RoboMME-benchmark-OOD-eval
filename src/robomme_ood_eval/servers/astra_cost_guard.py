"""3-tier Astra cost guard: continuously sums ``usage`` x unit price over all planning requests on both sides, and
writes ``STOP.json`` under ``group_*/`` when the limit is reached.

- What is scanned: under each ``--root`` (recursively), every ``response.json`` in the same directory as a
  ``request.json`` -- exactly the Astra ``Planner`` spool layout ``<spool>/<uuid>/{request.json,response.json}``;
  the monitor's ``monitor_inputs/tNNNN/response.json`` has no ``request.json`` next to it and is not counted. In the
  ``response.json`` written by ``ResponsesClient``, ``usage`` is the OpenAI Responses API usage object verbatim.
- Pricing: ``(input_tokens - cached_tokens) x input + cached_tokens x cached_input + output_tokens x output``, with
  unit prices in "USD per million tokens" supplied by the ``--prices`` config file (never hard-coded).
  ``reasoning_tokens`` are already included in ``output_tokens`` and are only counted, not priced separately (with
  ``"reasoning_billed_separately": true`` in the config they are additionally charged at the output price). Missing
  fields count as 0 and increment the ``missing_fields`` warning counter; responses with ``status=error`` and no
  ``usage`` count toward ``no_usage``.
- Cumulative ledger: the JSON at ``--ledger`` is read first (a ledger from a local pre-check can be copied to the
  cluster and continued), deduplicated by request uuid, and atomically written back after every scan.
- Limit check: ``total + max(1, in_flight) x worst-case cost of one request > --cap`` (in flight = has
  ``api_started.json`` but no ``response.json``; worst-case cost = largest input tokens seen x input price +
  ``--max-output-tokens`` (2048) x output price). When the limit is reached, ``STOP.json`` is written in every
  ``group_*`` directory (directories named ``group_<number>`` among the ancestors or descendants of ``--root``) and
  ``ASTRA_COST=STOP usd=... cap=...`` is printed. Both of Astra's built-in stop points read it:
  ``ResponsesClient._send`` before each send, and ``AstraPolicy.reset`` before each episode (in this evaluation repo
  the guard is started and stopped by ``AstraPolicy.load`` via ``ServerProcess``, no longer started manually first).
- Output: when the total changes (and on the first round) prints ``ASTRA_COST usd=<total> calls=<n> ...``.

Example price config (``--prices``)::

    {"model": "gpt-6-astra", "unit": "usd_per_1m_tokens",
     "input": 1.25, "cached_input": 0.125, "output": 10.0,
     "source": "<OpenAI pricing page URL>", "checked_at": "2026-10-05"}

(The values only illustrate the format; look up the official prices before running and record them.)

Hard limits:

- ``--cap`` defaults to and is at most ``HARD_CAP_USD`` (fixed at 5 USD); a larger value refuses to start (exit
  code 2); ``--interval`` defaults to 2 seconds (must be at most half the 10-second heartbeat timeout).
- Missing usage counts as over the limit: a non-error planning response without ``usage`` or without
  ``input_tokens`` / ``output_tokens`` writes STOP (``reason=astra_usage_missing``). Responses with
  ``status=error`` and no usage (HTTP errors, timeouts, ...; possibly billed, possibly not) are added to the
  projection at the "worst-case cost of one request" (``unknown_charge_usd``) rather than stopping directly -- HTTP
  errors themselves are not billed, so counting them at the worst-case price is already conservative; those with a
  ``guard_refused.json`` in the same directory (refused by the runner before sending, never left the machine) count
  as 0.
- Heartbeat and synchronous reservation: every round the state is atomically written to ``--state`` (default
  ``<ledger name>.state.json``): ``heartbeat`` (wall-clock seconds), ``committed_usd`` (counted + worst-case charge
  for error responses without usage), ``counted`` (request uuids already in the ledger), ``prices``,
  ``max_output_tokens``, ``cap``, ``stop``. ``GuardedResponsesClient`` in ``models/astra.py`` reads it synchronously
  before every real send and atomically reserves the single-request worst-case cost in
  ``<state>.reservations.json`` (``fcntl`` lock ``<state>.lock``); the guard also includes reservations not yet in
  the ledger in its projection. On a normal exit the guard writes ``exited=true``; if it crashes the heartbeat stops
  updating and the runner refuses to send after 10 seconds.
- Hard episode limit ``ASTRA_MAX_EPISODES=2``: before each episode the runner registers it in the ``episodes`` list
  of the same reservation file, cumulative across runs.
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
#: valid unit-price unit
PRICE_UNIT = "usd_per_1m_tokens"
#: fixed hard cost limit (USD; Astra local smoke runs are limited to <= 2 episodes and 5 USD); --cap must not exceed it
HARD_CAP_USD = 5.0
#: default guard scan interval (seconds)
DEFAULT_INTERVAL_S = 2.0
#: heartbeat timeout (seconds) after which the runner treats the guard as lost; --interval must not exceed half of it
HEARTBEAT_TIMEOUT_S = 10.0
#: hard Astra episode limit; the runner counts across runs via the episodes list of the reservation file
ASTRA_MAX_EPISODES = 2
STATE_SCHEMA = "astra-guard-state/1"
RESERVATIONS_SCHEMA = "astra-reservations/1"
#: marker written in the request directory when the runner refuses before sending; the guard counts such requests as
#: 0 (they never left the machine)
REFUSED_MARKER = "guard_refused.json"


def load_prices(path: Path) -> dict:
    prices = json.loads(Path(path).read_text())
    if prices.get("unit", PRICE_UNIT) != PRICE_UNIT:
        raise ValueError(f"unit price unit must be {PRICE_UNIT}, got {prices.get('unit')!r}")
    for key in ("input", "output"):
        if not isinstance(prices.get(key), (int, float)) or prices[key] < 0:
            raise ValueError(f"price config lacks a non-negative number for {key!r}")
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
    """Cost of one request; returns ``(usd, tokens, list of missing fields)``."""
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
        raise ValueError(f"ledger schema must be {LEDGER_SCHEMA}: {path}")
    return ledger


def save_ledger(path: Path, ledger: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def scan(roots: list[Path], ledger: dict, prices: dict) -> dict:
    """Scan all roots and add newly seen planning responses to the ledger; returns this round's stats (in flight,
    newly added, ...)."""
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
                continue  # being written (Astra uses atomic replace, so in theory this never happens); read next round
            usage = body.get("usage")
            if (call_dir / REFUSED_MARKER).is_file() and body.get("status") == "error" and not isinstance(usage, dict):
                usd, tokens, warn = 0.0, {}, ["refused"]  # refused by the runner before sending: never left the machine, not billed
            elif isinstance(usage, dict):
                usd, tokens, warn = price_usage(usage, prices)
            else:
                usd, tokens, warn = 0.0, {}, ["usage"]
            ledger["calls"][rid] = {"usd": usd, **tokens, "missing": warn, "status": body.get("status"),
                                    "path": str(response), "counted_at": time.time()}
            added += 1
    return {"pending": pending, "added": added}


def unknown_usage(call: dict) -> bool:
    """A non-error response without a billing basis (usage missing entirely, or missing input_tokens /
    output_tokens): treated as over the limit."""
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
    """All current ``group_*`` directories: each root itself and its ancestors named group_<n>, plus those under the
    root (<= 3 levels)."""
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
    """Sum of reservations not yet in the ledger (reserved by the runner, response not yet scanned by the guard);
    released (refused) ones are not counted."""
    if not reservations:
        return 0.0
    counted = set(counted)
    return sum(float(r.get("usd", 0.0)) for rid, r in (reservations.get("reservations") or {}).items()
               if rid not in counted and r.get("state") in ("reserved", "sent"))


def cycle(roots: list[Path], ledger: dict, prices: dict, cap: float, max_output_tokens: int,
          reservations: dict | None = None) -> dict:
    """One round: scan, compute the total and projection, write STOP when the limit is reached. Returns this round's
    summary.

    projection = counted + error responses without usage at the worst-case price + max(sum of outstanding
    reservations, max(1, in_flight) x worst-case cost of one request); a non-error response without usage always
    STOPs (``reason=astra_usage_missing``)."""
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


# -- heartbeat state and synchronous reservation (protocol shared by runner and guard) --

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
    """Exclusive lock on the reservation file (``fcntl.flock``); runner reservations and each guard scan run inside
    the lock, so "read - decide - write" is atomic."""
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
        raise ValueError(f"reservation file schema must be {RESERVATIONS_SCHEMA}: {path}")
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
    """One guard round: read reservations inside the lock -> scan and project (write STOP when the limit is reached)
    -> write the heartbeat state. Shared by ``main`` and tests."""
    with locked(state_path):
        reservations = load_reservations(state_path)
        summary = cycle(roots, ledger, prices, cap, max_output_tokens, reservations)
        write_state(state_path, summary, ledger, prices, max_output_tokens, clock=clock)
    return summary


def check_cap(cap: float) -> float:
    cap = float(cap)
    if not 0 < cap <= HARD_CAP_USD:
        raise ValueError(f"ASTRA_COST_BLOCKED --cap={cap:g} exceeds the hard limit of {HARD_CAP_USD:g} USD")
    return cap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3-tier Astra cost guard (aggregates once every --interval seconds)")
    parser.add_argument("--root", action="append", required=True,
                        help="spool root (repeatable, e.g. one for a local pre-check and one for the cluster); may "
                             "also be a group_*/ directory or its parent")
    parser.add_argument("--prices", required=True, help="unit price config JSON (USD per million tokens)")
    parser.add_argument("--ledger", required=True, help="cumulative ledger JSON: read first if it exists, written "
                                                        "back every round")
    parser.add_argument("--cap", type=float, default=HARD_CAP_USD,
                        help=f"USD; default and maximum {HARD_CAP_USD:g} (hard limit)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help=f"seconds; must not exceed half the heartbeat timeout of {HEARTBEAT_TIMEOUT_S:g} seconds")
    parser.add_argument("--state", default=None, help="heartbeat state file (read synchronously by the runner before "
                                                      "sending); default <ledger name>.state.json")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="max_output_tokens of each Astra request")
    parser.add_argument("--once", action="store_true", help="run a single round (for tests and manual reconciliation)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_cap(args.cap)
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    if not 0 < args.interval <= HEARTBEAT_TIMEOUT_S / 2:
        print(f"ASTRA_COST_BLOCKED --interval={args.interval:g} must be within (0, {HEARTBEAT_TIMEOUT_S / 2:g}] seconds",
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
        # normal exit: write exited=true so the runner refuses immediately (without waiting for the heartbeat timeout)
        with locked(state_path):
            write_state(state_path, summary, ledger, prices, args.max_output_tokens, exited=True)
        print(f"ASTRA_COST_GUARD_EXIT state={state_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
