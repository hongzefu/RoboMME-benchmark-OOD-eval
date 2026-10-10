"""Single-threaded decision timing and aggregation, in milliseconds.

The timer observes calls; it does not synchronize devices or query hardware.
Language within a forward pass is a subset of server inference, not another RTT.
"""
from __future__ import annotations

import copy
import math
import statistics
import time

LANG_KINDS = frozenset({"subgoal", "s2", "planner", "monitor", "review", "lang_in_forward"})
DETAIL_KINDS = frozenset({"ask"})
MODEL_KINDS = ("serial", "pp", "smvla")
SERIAL_TOL = 0.20
PP_TOL_MS = 1.0
CHUNK_FIELDS = ("decision_wall_ms", "action_rtt_ms", "server_infer_ms", "lang_ms")
RESERVED_KEYS = frozenset((*CHUNK_FIELDS, "decision", "env_step", "n_lang", "phase"))
PLANNER_LAYERS = ("planner+review", "planner_only", "review_only", "neither", "no_lang")


def _ms(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("timing must be finite and nonnegative")
    return value


def _phase(index):
    return ("first", "second", "third")[index] if index < 3 else "steady"


class ChunkTimer:
    """A single-threaded timer. Repeated start replaces only the start time."""

    def __init__(self, *, clock=time.perf_counter):
        self.chunks = []
        self.lang_calls = []
        self._clock = clock
        self._t_obs = None
        self._bucket_ms = 0.0
        self._bucket_n = 0
        self._orphan_ms = 0.0
        self._orphan_n = 0
        self._attached = None

    def start(self, t=None):
        self._t_obs = self._clock() if t is None else float(t)

    def add_lang(self, ms, *, kind, env_step, **extra):
        if kind not in LANG_KINDS | DETAIL_KINDS:
            raise ValueError(f"unknown language timing kind: {kind}")
        if {"index", "env_step", "ms", "kind"}.intersection(extra):
            raise ValueError("language timing extra uses reserved keys")
        ms = _ms(ms)
        self.lang_calls.append({"index": len(self.lang_calls), "env_step": int(env_step),
                                "ms": round(ms, 3), "kind": kind,
                                **{k: v for k, v in extra.items() if v is not None}})
        if kind in LANG_KINDS:
            self._bucket_ms += ms
            self._bucket_n += 1

    def close(self, *, decision, env_step, action_rtt_ms, server_infer_ms, extra=None,
              t=None, decision_wall_ms=None):
        extra = dict(extra or {})
        if RESERVED_KEYS.intersection(extra):
            raise ValueError("chunk timing extra uses reserved keys")
        if decision_wall_ms is None:
            if self._t_obs is None:
                raise RuntimeError("ChunkTimer.close without start")
            decision_wall_ms = ((self._clock() if t is None else float(t)) - self._t_obs) * 1000
        values = {"decision_wall_ms": _ms(decision_wall_ms), "lang_ms": self._bucket_ms,
                  "action_rtt_ms": None if action_rtt_ms is None else _ms(action_rtt_ms),
                  "server_infer_ms": None if server_infer_ms is None else _ms(server_infer_ms)}
        chunk = {"decision": int(decision), "env_step": int(env_step), "n_lang": self._bucket_n,
                 "phase": _phase(len(self.chunks)),
                 **{k: None if v is None else round(v, 3) for k, v in values.items()}, **extra}
        for key, value in list(extra.items()):
            if key.endswith("_ms") and value is not None:
                chunk[key] = round(_ms(value), 3)
        if "observe_rtt_ms" in chunk and chunk["observe_rtt_ms"] is not None and action_rtt_ms is not None:
            chunk["slack_ms"] = round(values["decision_wall_ms"] - chunk["observe_rtt_ms"] - action_rtt_ms, 3)
        self.chunks.append(chunk)
        self._t_obs = None
        self._bucket_ms = 0.0
        self._bucket_n = 0
        return chunk

    def absorb_orphan(self, ms, n):
        ms = _ms(ms)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("orphan count must be a nonnegative integer")
        self._orphan_ms += ms
        self._orphan_n += n

    def orphan(self):
        out = {"orphan_lang_ms": round(self._bucket_ms + self._orphan_ms, 3),
               "n": self._bucket_n + self._orphan_n}
        self._bucket_ms = self._orphan_ms = 0.0
        self._bucket_n = self._orphan_n = 0
        return out


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _field_summary(chunks, field):
    vals = [c[field] for c in chunks if c.get(field) is not None]
    if not vals:
        return {"n": 0}
    steady = [c[field] for c in chunks if c.get("phase") == "steady" and c.get(field) is not None]
    selected = steady if any(c.get("phase") == "steady" for c in chunks) else vals
    return {"n": len(vals),
            **{phase: next((c[field] for c in chunks if c.get("phase") == phase and c.get(field) is not None), None)
               for phase in ("first", "second", "third")},
            "steady_mean": statistics.mean(selected) if selected else None, "steady_p50": _percentile(selected, .50),
            "steady_p95": _percentile(selected, .95)}


def summarize_chunks(chunks, fields=CHUNK_FIELDS):
    # Assign missing phases before filtering; existing phases survive stratification.
    chunks = [{**c, "phase": c.get("phase", _phase(i))} for i, c in enumerate(chunks)]
    return {"n_decisions": len(chunks),
            "steady_short": not any(c["phase"] == "steady" for c in chunks),
            **{field: _field_summary(chunks, field) for field in fields}}


def _layer(chunk):
    if chunk.get("lang_ms", 0) == 0:
        return "no_lang"
    planner, review = bool(chunk.get("has_planner")), bool(chunk.get("has_review"))
    return "planner+review" if planner and review else "planner_only" if planner else "review_only" if review else "neither"


def _conservation(timer, kind):
    out = {"kind": kind, "expected": len(timer.chunks), "checked": 0,
           "violations": 0, "missing_fields": 0, "overhead_pct_p50": None}
    overhead = []
    for c in timer.chunks:
        wall, rtt, lang = c["decision_wall_ms"], c["action_rtt_ms"], c["lang_ms"]
        if rtt is None:
            out["missing_fields"] += 1
            continue
        if kind == "smvla":
            obs, infer = c.get("observe_rtt_ms"), c.get("server_infer_ms")
            if obs is None or infer is None:
                out["missing_fields"] += 1
                out["violations"] += 1
                continue
            bad = wall < obs + rtt - 1 or lang > infer + 1 or infer > rtt + 1
            base = obs + rtt
        else:
            base = rtt + lang
            bad = abs(wall - base) > PP_TOL_MS if kind == "pp" else wall < base - 1
        out["checked"] += 1
        out["violations"] += int(bad)
        if wall:
            overhead.append((wall - base) / wall)
    out["overhead_pct_p50"] = _percentile(overhead, .5)
    if kind == "smvla":
        # Calls are assigned to chunks by their counted-call order, not env_step,
        # which may repeat across decisions in lightweight sessions.
        counted = [c for c in timer.lang_calls if c["kind"] in LANG_KINDS]
        offset = 0
        available = True
        for c in timer.chunks:
            calls = counted[offset:offset + c["n_lang"]]
            available &= any(x["kind"] == "lang_in_forward" for x in calls)
            offset += c["n_lang"]
        out["lang_split_available"] = available
    return out


def attach_policy_timing(timing, timer, *, gpu_name, model_kind):
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"unknown model timing kind: {model_kind}")
    if timer._attached is None:
        fields = CHUNK_FIELDS + (("observe_rtt_ms", "slack_ms") if model_kind == "smvla" else ())
        summary = summarize_chunks(timer.chunks, fields)
        if any("has_planner" in c for c in timer.chunks):
            summary["by_planner_review"] = {
                layer: summarize_chunks(sub) if sub else {"n": 0}
                for layer in PLANNER_LAYERS
                for sub in [[c for c in timer.chunks if _layer(c) == layer]]}
        lang_summary = {}
        for kind in sorted({c["kind"] for c in timer.lang_calls}):
            sub = [c for c in timer.lang_calls if c["kind"] == kind]
            lang_summary[kind] = _field_summary([{**c, "phase": _phase(i)} for i, c in enumerate(sub)], "ms")
        lang_summary.update(total_ms_counted=sum(c["ms"] for c in timer.lang_calls if c["kind"] in LANG_KINDS),
                            n_ask=sum(c["kind"] == "ask" for c in timer.lang_calls))
        orphan = timer.orphan()
        data = {"chunks": timer.chunks, "chunk_summary": summary, "lang_calls": timer.lang_calls,
                "lang_summary": lang_summary, "gpu_name": gpu_name, "orphan_lang_ms": orphan["orphan_lang_ms"],
                "orphan_n": orphan["n"], "dangling_start": timer._t_obs is not None,
                "conservation": _conservation(timer, model_kind)}
        if model_kind == "smvla":
            data["lang_split_available"] = data["conservation"]["lang_split_available"]
        timer._attached = copy.deepcopy(data)
    timing.update(copy.deepcopy(timer._attached))
    return timing


def _conservation_counts(policy_timing):
    """Preserve available counts and fail closed when evidence is incomplete."""
    keys = ("violations", "missing_fields", "checked", "expected")
    evidence = policy_timing.get("conservation")
    evidence = evidence if isinstance(evidence, dict) else {}
    counts = {}
    incomplete = False
    for key in keys:
        value = evidence.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            incomplete = True
        else:
            counts[key] = value
    decisions = policy_timing["chunk_summary"].get("n_decisions", 0)
    if "expected" not in counts:
        counts["expected"] = decisions
    elif counts["expected"] != decisions:
        incomplete = True
        counts["expected"] = max(counts["expected"], decisions)
    # Zeros here mean no known violations/comparisons, never successful evidence:
    # an absent or malformed count always contributes a missing-evidence count.
    out = {key: counts.get(key, 0) for key in keys}
    if incomplete:
        out["missing_fields"] += max(1, decisions)
    return out


def merge_summaries(summaries):
    """Merge one GPU group of policy timings, using raw steady values if present.

    With summary-only inputs, means are weighted by the original phase count;
    percentiles are approximate and explicitly marked. Callers own exclusions.
    Absent conservation counts contribute missing evidence, so they cannot pass.
    """
    summaries = [p for p in summaries if p.get("chunk_summary") is not None]
    gpus = {p.get("gpu_name") for p in summaries}
    if len(gpus) > 1:
        raise ValueError("timing summaries must be grouped by gpu_name")
    result = {"decisions": sum(p["chunk_summary"].get("n_decisions", 0) for p in summaries),
              "gpu_name": next(iter(gpus), None), "episodes_with_timing": len(summaries),
              "steady_short_eps": sum(bool(p["chunk_summary"].get("steady_short")) for p in summaries),
              "approx": any("chunks" not in p for p in summaries)}
    result["n_decisions"] = result["decisions"]
    evidence = [_conservation_counts(p) for p in summaries]
    for key in ("violations", "missing_fields", "checked", "expected"):
        result[key] = sum(counts[key] for counts in evidence)
    for field in CHUNK_FIELDS:
        first = []
        total, weight = 0.0, 0
        raw = []
        approx50, approx95 = [], []
        for p in summaries:
            stats = p["chunk_summary"].get(field, {})
            if stats.get("first") is not None:
                first.append(stats["first"])
            if "chunks" in p:
                chunks = [{**c, "phase": c.get("phase", _phase(i))} for i, c in enumerate(p["chunks"])]
                all_vals = [c[field] for c in chunks if c.get(field) is not None]
                vals = [c[field] for c in chunks if c["phase"] == "steady" and c.get(field) is not None] if any(c["phase"] == "steady" for c in chunks) else all_vals
                total += sum(vals)
                weight += len(vals)
                raw.extend(vals)
                approx50.extend(vals)
                approx95.extend(vals)
            else:
                n = stats.get("n", 0)
                count = n if p["chunk_summary"].get("steady_short") else max(0, n - sum(stats.get(k) is not None for k in ("first", "second", "third")))
                if count and stats.get("steady_mean") is not None:
                    total += stats["steady_mean"] * count
                    weight += count
                    approx50.extend([stats.get("steady_p50", stats["steady_mean"])] * count)
                    approx95.extend([stats.get("steady_p95", stats["steady_mean"])] * count)
        stats = {"first_mean": statistics.mean(first) if first else None,
                 "steady_mean": total / weight if weight else None,
                 "steady_p50": statistics.mean(approx50) if result["approx"] and approx50 else _percentile(raw, .5),
                 "steady_p95": statistics.mean(approx95) if result["approx"] and approx95 else _percentile(raw, .95),
                 "steady_n": weight}
        result[field] = stats
        if field != "decision_wall_ms":
            result[field.removesuffix("_ms") + "_steady_mean_ms"] = stats["steady_mean"]
    if any("by_planner_review" in p["chunk_summary"] for p in summaries):
        result["by_planner_review"] = {}
        for layer in PLANNER_LAYERS:
            inputs = []
            for p in summaries:
                sub = p["chunk_summary"].get("by_planner_review", {}).get(layer, {"n": 0})
                if not sub.get("n_decisions", 0):
                    continue
                q = {"gpu_name": p.get("gpu_name"), "chunk_summary": sub}
                if "chunks" in p:
                    q["chunks"] = [c for c in p["chunks"] if _layer(c) == layer]
                    conservation = p.get("conservation")
                    kind = conservation.get("kind") if isinstance(conservation, dict) else None
                    if kind in MODEL_KINDS:
                        timer = ChunkTimer()
                        timer.chunks = q["chunks"]
                        q["conservation"] = _conservation(timer, kind)
                # Without raw chunks, episode-level conservation cannot be
                # attributed to this stratum. Deliberately leave it unavailable:
                # recursive aggregation marks missing evidence and approx=True.
                inputs.append(q)
            result["by_planner_review"][layer] = merge_summaries(inputs) if inputs else {"n": 0}
    return result
