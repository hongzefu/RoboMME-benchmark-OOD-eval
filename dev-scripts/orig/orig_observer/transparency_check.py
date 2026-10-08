"""代理透明性对账（FrameSamp+Modulation 原侧观测器）：客户端逐条 send/recv 的 sha256 与代理入口／出口逐条对照（方案第二部分 §2 ``OBSERVER_TRANSPARENT``）。

对账口径：
- 客户端 ``send`` 第 i 条 ↔ 代理同一连接 ``c2s`` 第 i 条（代理收到并原样转发给 server 的那条）；
- 代理 ``s2c`` 第 i 条（从 server 收到并原样转发）↔ 客户端 ``recv`` 第 i 条；
- 比对帧类型、字节长度、sha256 与顺序。连接按内容配对（客户端第 k 条连接 ↔ 代理里消息序列与之最吻合的连接），
  不依赖进程内连接序号（每遍重评会重起客户端进程，代理常驻）。

S7 新增（计划第二部分一节 S7，审计第 13 条）：
- ``--manifest <清单> [--shard <i>] [--only-tasks a,b]``：本片每个清单身份（``<task>_xhard0_<seed>``）必须对应到至少一条
  客户端连接（``client-transport-*.jsonl`` 的 ``episode`` 字段为局目录名 ``<key>.a<N>``）与至少一局带 ``end`` 行的
  ``trace.jsonl``（在 ``--episodes`` 目录下，缺省即 ``--rec-root``），缺即 FAIL（``missing_conn``／``missing_trace``）；
- 日志封口：``<rec_root>/proxy/proxy-<pid>.jsonl`` 每份都必须有同名 ``proxy-<pid>.done``，否则 FAIL（``unsealed``）。

用法：python transparency_check.py --rec-root <REC_ROOT> [--out report.json] [--manifest M --shard I] [--episodes DIR]
输出一行：``OBSERVER_TRANSPARENT=PASS|FAIL messages=<n> mismatch=<m> conns=<k> unmatched_conns=<u> proxy_only_conns=<x>
missing_conn=<a> missing_trace=<b> unsealed=<c>``
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

EP_DIR_RE = re.compile(r"^(?P<key>.+)\.a(?P<attempt>\d+)$")


def _read(paths):
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def load_client(rec_root: Path) -> dict:
    """{(pid, conn): {"send": [...], "recv": [...]}}，每条 (idx, type, len, sha256)。"""
    out: dict = collections.defaultdict(lambda: {"send": [], "recv": []})
    for r in _read(sorted(rec_root.glob("client-transport-*.jsonl"))):
        out[(r["pid"], r["conn"])][r["dir"]].append((r["idx"], r["type"], r["len"], r["sha256"]))
    return dict(out)


def load_proxy(rec_root: Path) -> dict:
    """{(proxy_file, conn): {"c2s": [...], "s2c": [...]}}。"""
    out: dict = collections.defaultdict(lambda: {"c2s": [], "s2c": []})
    for p in sorted((rec_root / "proxy").glob("proxy-*.jsonl")):
        for r in _read([p]):
            if r.get("kind") == "msg":
                out[(p.name, r["conn"])][r["dir"]].append((r["idx"], r["type"], r["len"], r["sha256"]))
    return dict(out)


def _cmp(a: list, b: list) -> tuple[int, int]:
    """(对上的条数, 不符条数)；按 idx 排序后逐位比，长度差计入不符。"""
    a = sorted(a)
    b = sorted(b)
    ok = sum(1 for x, y in zip(a, b) if x == y)
    bad = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    # idx 必须是 0..n-1 连续（顺序不乱、不缺）
    bad += sum(1 for i, x in enumerate(a) if x[0] != i) + sum(1 for i, y in enumerate(b) if y[0] != i)
    return ok, bad


def manifest_keys(path: str | Path, shard: int | None = None, only_tasks: str | None = None) -> list[str]:
    """清单本片身份键 ``<task>_xhard0_<seed>``（排序、去重）。"""
    want = set(only_tasks.split(",")) if only_tasks else None
    keys = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if shard is not None and int(r.get("shard", -1)) != int(shard):
            continue
        if want is not None and r["task"] not in want:
            continue
        keys.add(f"{r['task']}_xhard0_{int(r['seed'])}")
    return sorted(keys)


def _key_of(name) -> str | None:
    m = EP_DIR_RE.match(str(name or ""))
    return m["key"] if m else None


def coverage(rec_root: Path, keys: list[str], episodes: Path | None = None) -> dict:
    """每个身份键：有无客户端连接、有无完整 trace（带 end 行）。"""
    conn_keys = set()
    for r in _read(sorted(rec_root.glob("client-transport-*.jsonl"))):
        k = _key_of(r.get("episode"))
        if k:
            conn_keys.add(k)
    trace_keys = set()
    root = Path(episodes) if episodes else rec_root
    for t in sorted(root.glob("*.a*/trace.jsonl")):
        k = _key_of(t.parent.name)
        if not k:
            continue
        try:
            lines = [x for x in t.read_text(encoding="utf-8").splitlines() if x.strip()]
            if lines and json.loads(lines[-1]).get("kind") == "end":
                trace_keys.add(k)
        except Exception:  # noqa: BLE001 坏文件按缺 trace 计
            continue
    return {"missing_conn": [k for k in keys if k not in conn_keys],
            "missing_trace": [k for k in keys if k not in trace_keys]}


def unsealed(rec_root: Path) -> list[str]:
    """没有 ``.done`` 封口标记的代理日志。"""
    out = []
    for p in sorted((rec_root / "proxy").glob("proxy-*.jsonl")):
        if not p.with_suffix(".done").exists():
            out.append(p.name)
    return out


def check(rec_root: Path, keys: list[str] | None = None, episodes: Path | None = None) -> dict:
    client = load_client(rec_root)
    proxy = load_proxy(rec_root)
    used = set()
    pairs, messages, mismatch, unmatched = [], 0, 0, 0
    for ck in sorted(client):
        c = client[ck]
        best, best_score = None, None
        for pk, p in proxy.items():
            if pk in used:
                continue
            ok1, bad1 = _cmp(c["send"], p["c2s"])
            ok2, bad2 = _cmp(p["s2c"], c["recv"])
            score = (ok1 + ok2, -(bad1 + bad2))
            if best_score is None or score > best_score:
                best, best_score = pk, score
        n = len(c["send"]) + len(c["recv"])
        if best is None or best_score[0] == 0:
            unmatched += 1
            mismatch += n
            pairs.append({"client": list(ck), "proxy": None, "messages": n})
            continue
        used.add(best)
        p = proxy[best]
        ok1, bad1 = _cmp(c["send"], p["c2s"])
        ok2, bad2 = _cmp(p["s2c"], c["recv"])
        messages += ok1 + ok2
        mismatch += bad1 + bad2
        pairs.append({"client": list(ck), "proxy": list(best), "c2s_ok": ok1, "c2s_bad": bad1,
                      "s2c_ok": ok2, "s2c_bad": bad2})
    extra = [list(k) for k in proxy if k not in used]
    cov = coverage(rec_root, keys, episodes) if keys is not None else {"missing_conn": [], "missing_trace": []}
    uns = unsealed(rec_root)
    ok = mismatch == 0 and unmatched == 0 and messages > 0 and not extra  # 代理多出的连接也算不透明
    ok = ok and not cov["missing_conn"] and not cov["missing_trace"] and not uns
    return {"OBSERVER_TRANSPARENT": "PASS" if ok else "FAIL", "messages": messages, "mismatch": mismatch,
            "conns": len(client), "unmatched_conns": unmatched, "proxy_only_conns": extra, "pairs": pairs,
            "manifest_keys": None if keys is None else len(keys), "missing_conn": cov["missing_conn"],
            "missing_trace": cov["missing_trace"], "unsealed": uns}


def verdict_line(rep: dict) -> str:
    return (f"OBSERVER_TRANSPARENT={rep['OBSERVER_TRANSPARENT']} messages={rep['messages']} "
            f"mismatch={rep['mismatch']} conns={rep['conns']} unmatched_conns={rep['unmatched_conns']} "
            f"proxy_only_conns={len(rep['proxy_only_conns'])} missing_conn={len(rep.get('missing_conn') or [])} "
            f"missing_trace={len(rep.get('missing_trace') or [])} unsealed={len(rep.get('unsealed') or [])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="代理透明性对账")
    ap.add_argument("--rec-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None, help="清单 jsonl；给出即核对本片每个身份都有连接与 trace")
    ap.add_argument("--shard", type=int, default=None, help="只取清单里 shard 等于此值的行")
    ap.add_argument("--only-tasks", default=None, help="逗号分隔，只核对这些任务（与评估的 only_tasks 一致）")
    ap.add_argument("--episodes", default=None, help="逐局目录 <key>.a<N>/ 所在根（缺省 = --rec-root）")
    a = ap.parse_args(argv)
    keys = manifest_keys(a.manifest, a.shard, a.only_tasks) if a.manifest else None
    rep = check(Path(a.rec_root), keys, Path(a.episodes) if a.episodes else None)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(verdict_line(rep), flush=True)
    return 0 if rep["OBSERVER_TRANSPARENT"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
