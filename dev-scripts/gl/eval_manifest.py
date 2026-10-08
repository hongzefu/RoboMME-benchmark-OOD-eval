#!/usr/bin/env python3
"""V9 双模型评估执行清单（1002-newtask-v9-movecube-region-800-plan.md 第二部分 §2.1、§2.2 第 7 条；字段契约沿用
1001-v8-post-evaluation-gl-plan.md 的 C1）。

    python scripts/eval-official/eval_manifest.py --identities <eval-identities-992.jsonl> \
        --delivery <newtask-v9/delivery/delivery.local.json> \
        --exclude-evaluated <V8 manifest.json> --shards 10 --out-dir <dir>

V8 1070 局的构建分支（不带 ``--exclude-evaluated``）已删除；``--exclude-evaluated`` 现为必填。V8 manifest 只作为
已评身份的对账输入读取（``load_v8_manifest``）。

格表不取 ``EXPECTED_CELLS``，而由 ``--delivery`` 逐格计数推出，且必须与 ``hard_specs`` 登记的某张完整交付格表
（``EXPECTED_CELLS``／``CELL_TABLES``）逐格相等（V9 即 43 格 800 局）。五步，任何一步不过即打印
``V9_EVAL_SHARDS=FAIL ...`` 并以非零退出（不过不写产物；写出后读回核对不过则删掉本次写出的 manifest.json、
shard-NN.json 与 reused.json，不留半成品）：

1. 核源集：``export_eval_identities.py`` 的身份行（xhard0 + 新值），逐格对格表；xhard0 行 candidate=null、
   source_episode 为整数，新值行 candidate 为整数、source_episode=null；(task, episode) 唯一。
2. 筛新值行，xhard0 在这一步丢弃并计数。
3. 按 ``(task, tier, seed, candidate)`` 连交付行补 ``spec_sha256``——**不用交付行的 episode**
   （那是格内序号，不是 builder episode）；连不上（missing）或一对多（duplicate）即停；交付行未被用到记 extra。
4. 核执行清单：格数、行数与格表相等，xhard0=0、指纹齐全、``key`` 唯一。
5. 按四元组 ``(task, tier, seed, spec_sha256)`` 与 V8 manifest ``rows`` 比对：命中即复用（V9 MoveCube 与 V8 同 seed
   不同布局，靠 ``spec_sha256`` 区分，不命中），不命中即新评。新评集合必须恰为 ``V9_NEW_RULE``（MoveCube xhard4
   全格 + InsertPeg xhard4 候选号 ≥ 29），否则 FAIL。

然后按任务历史耗时（``TASK_SECONDS``，档位无关）对新评行做最长处理时间优先的贪心均衡切片，两模型用同一分片；
核对分片两两不交、并集等于新评集合后写：

- ``<out-dir>/manifest.json``：``{schema, source_sha256, delivery_sha256, xhard0_dropped, total, cells, shards, rows,
  reused}``，``reused = {path, sha256, count, v8_manifest, v8_manifest_sha256}``；
- ``<out-dir>/shard-NN.json``：执行身份行数组，每行
  ``{task, tier, seed, candidate, builder_episode, source_episode, spec_sha256, key}``（步数上限不进身份行），
  ``key = f"{task}_{tier}_{seed}"``；
- ``<out-dir>/reused.json``：复用集合（schema ``v9-eval-reused/1``，rows 按 (task, tier, seed) 排序，``v8_key`` 取
  V8 manifest 行的 ``key`` 原值）。

末行 ``V9_EVAL_SHARDS=PASS shards=10 total=80 cells=2 reused=720 reused_sha256=<前 12 位> missing=0 extra=0 duplicate=0 xhard0=0``。
只用标准库，``hard_specs`` 按文件路径加载（不 import robomme_hard 包，不触发 sapien）。

``--mode``（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.2）：

- ``v9-new``（默认）：上述五步，行为与判定行不变。
- ``v9-full``：``--identities`` + ``--delivery``，只做第 1～4 步、不剔除已评身份，全部执行行（800 局）切片；末行
  ``EVAL_SHARDS=PASS mode=v9-full total=800 cells=43 missing=0 extra=0 duplicate=0 xhard0=0``。
- ``hard0``：不读交付清单，直接由 ``BenchmarkEnvBuilder(task, dataset="hard-verify")`` 枚举 16 任务、每任务前
  ``--per-task N``（默认 12）局（此模式会 import robomme_hard）；行键同 ``SHARD_ROW_KEYS``、``spec_sha256=None``、
  ``candidate=None``、``key=<task>_xhard0_<seed>``；末行 ``EVAL_SHARDS=PASS mode=hard0 total=<16×N> xhard0=<16×N> ...``。
  ``--pair-shards``：原侧与新侧共用同一份 ``shard-NN.json``（同一身份两侧同分片、分片内同序）。

``xhard0==0`` 的要求只在 V9 两种模式（``check_exec``）里保留。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
HARD_SPECS_PATH = REPO / "src" / "robomme_hard" / "env_record_wrapper" / "hard_specs.py"

SCHEMA = "v8-eval-manifest/1"
REUSED_SCHEMA = "v9-eval-reused/1"
REUSED_FILE = "reused.json"
#: V9 新评集合规则（第一部分 §1 第 4、7 条）：{(task, tier): 最小候选号}；MoveCube 全部重抽（候选号 ≥ 0），
#: InsertPeg 只评追加的候选号 ≥ 29 的局（V8 已评的 20 局候选号 < 29）。
V9_NEW_RULE = {("MoveCube", "xhard4"): 0, ("InsertPeg", "xhard4"): 29}
DELIVERY_SCHEMA = "v8-delivery/1"
SOURCE_KEYS = ("task", "episode", "tier", "seed", "candidate", "source_episode", "round", "shard")
#: 执行身份行字段；与 env_client.V8_IDENTITY_KEYS 同步（步数上限不进身份行，由入口按数据集给 --max-steps）
SHARD_ROW_KEYS = ("task", "tier", "seed", "candidate", "builder_episode", "source_episode", "spec_sha256", "key")
DEFAULT_SHARDS = 10
#: hard0 模式每任务默认取前 12 局（hard-verify 每任务恰 12 局）
DEFAULT_PER_TASK = 12

#: 每任务平均单局用时（秒），复制自 ``scripts/injection-dev/export_eval_identities.py::TASK_SECONDS``
#: （该脚本 import ``_common`` 改 sys.path，不便跨目录 import；数值逐字相同，测试核对两者一致）。
TASK_SECONDS = {
    "BinFill": 224.0, "ButtonUnmask": 116.5, "ButtonUnmaskSwap": 73.4, "InsertPeg": 152.2, "MoveCube": 180.3,
    "PatternLock": 79.8, "PickHighlight": 246.7, "PickXtimes": 114.2, "RouteStick": 74.1, "StopCube": 64.9,
    "SwingXtimes": 90.8, "VideoPlaceButton": 58.0, "VideoPlaceOrder": 61.7, "VideoRepick": 51.9, "VideoUnmask": 97.4,
    "VideoUnmaskSwap": 50.5,
}


class ManifestError(RuntimeError):
    """四步转换中的任何不符：带计数，main 据此打印 FAIL 行。"""

    def __init__(self, stage: str, detail: str, **counts):
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail
        self.counts = counts


def load_hard_specs():
    """按文件路径加载 ``hard_specs``（只依赖标准库）。"""
    name = "_v8_manifest_hard_specs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARD_SPECS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def v8_key(row: dict) -> str:
    return f"{row['task']}_{row['tier']}_{int(row['seed'])}"


def cell_name(task: str, tier: str) -> str:
    return f"{task}@{tier}"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ManifestError("source", f"第 {i} 行不是 JSON：{e}") from e
    return rows


# ── 第 1 步：核源集 ──────────────────────────────────────────────────────────


def check_source(rows: list[dict], hs, cells: dict | None = None) -> dict:
    cells = hs.EXPECTED_CELLS if cells is None else cells
    # xhard0 行数随开关 XHARD0_IN_TEST_HARD：开为 16 × 12、关为 0（与 export_eval_identities.xhard0_total 同口径）；
    # 旧写法写死 16 × XHARD0_PER_TASK，开关关（V9 默认）时会把导出器的 800 行源集误判为缺 192 行
    xhard0_expected = len(hs.ALL_TASKS) * hs.xhard0_prefix()
    total_expected = xhard0_expected + sum(cells.values())
    bad = []
    seen = set()
    per_cell: dict[tuple[str, str], int] = defaultdict(int)
    xhard0 = 0
    for i, r in enumerate(rows):
        if set(r) != set(SOURCE_KEYS):
            bad.append(f"row{i} keys={sorted(r)}")
            continue
        tk = (r["task"], r["episode"])
        if tk in seen:
            bad.append(f"row{i} duplicate (task,episode)={tk}")
        seen.add(tk)
        if not _is_int(r["episode"]) or not _is_int(r["seed"]):
            bad.append(f"row{i} episode/seed 非整数")
        if r["tier"] == hs.XHARD0:
            xhard0 += 1
            if r["candidate"] is not None or not _is_int(r["source_episode"]):
                bad.append(f"row{i} xhard0 行 candidate 须为 null、source_episode 须为整数")
        else:
            if not _is_int(r["candidate"]) or r["source_episode"] is not None:
                bad.append(f"row{i} 新值行 candidate 须为整数、source_episode 须为 null")
            per_cell[(r["task"], r["tier"])] += 1
    cell_mismatch = sum(per_cell.get(k, 0) != n for k, n in cells.items()) + sum(k not in cells for k in per_cell)
    if bad or len(rows) != total_expected or xhard0 != xhard0_expected or cell_mismatch:
        raise ManifestError("source", f"源集不符 rows={len(rows)} expected={total_expected} xhard0={xhard0} "
                                      f"expected_xhard0={xhard0_expected} cell_mismatch={cell_mismatch} "
                                      f"bad={bad[:5]}", total=len(rows), xhard0=xhard0, cells=len(per_cell))
    return {"rows": len(rows), "xhard0": xhard0, "cells": len(per_cell)}


# ── 第 2、3 步：筛新值、连交付补指纹 ─────────────────────────────────────────


def filter_new(rows: list[dict], hs) -> tuple[list[dict], int]:
    new = [r for r in rows if r["tier"] != hs.XHARD0]
    return new, len(rows) - len(new)


def join_delivery(new_rows: list[dict], delivery: dict, hs) -> list[dict]:
    if delivery.get("schema") != DELIVERY_SCHEMA:
        raise ManifestError("delivery", f"交付 schema={delivery.get('schema')!r}，期望 {DELIVERY_SCHEMA}")
    index: dict[tuple, list[dict]] = defaultdict(list)
    for d in delivery.get("rows") or []:
        index[(d["task"], d["tier"], int(d["seed"]), d["candidate"])].append(d)
    out, missing, duplicate = [], [], []
    used: set[tuple] = set()
    for r in new_rows:
        jk = (r["task"], r["tier"], int(r["seed"]), r["candidate"])
        hits = index.get(jk, [])
        if not hits:
            missing.append(jk)
            continue
        if len(hits) > 1:
            duplicate.append(jk)
            continue
        if jk in used:  # 两条身份行连到同一交付行，也是一对多
            duplicate.append(jk)
            continue
        used.add(jk)
        sha = hits[0].get("spec_sha256")
        out.append({"task": r["task"], "tier": r["tier"], "seed": int(r["seed"]), "candidate": int(r["candidate"]),
                    "builder_episode": int(r["episode"]), "source_episode": r["source_episode"], "spec_sha256": sha,
                    "key": v8_key(r)})
    extra = [k for k in index if k not in used]
    if missing or duplicate or extra:
        raise ManifestError("join", f"连接交付行失败 missing={missing[:5]} duplicate={duplicate[:5]} extra={extra[:5]}",
                            missing=len(missing), duplicate=len(duplicate), extra=len(extra))
    return out


# ── 第 4 步：核执行清单 ──────────────────────────────────────────────────────


def check_exec(rows: list[dict], hs, cells: dict | None = None) -> dict:
    """V9 两种模式（v9-new／v9-full）的执行清单核对：格数、行数与格表相等，xhard0=0、指纹齐全、key 唯一。"""
    cells = hs.EXPECTED_CELLS if cells is None else cells
    per_cell: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        per_cell[(r["task"], r["tier"])] += 1
    xhard0 = sum(r["tier"] == hs.XHARD0 for r in rows)
    no_sha = sum(not (isinstance(r["spec_sha256"], str) and len(r["spec_sha256"]) == 64
                      and all(c in "0123456789abcdef" for c in r["spec_sha256"])) for r in rows)
    keys = [r["key"] for r in rows]
    dup = len(keys) - len(set(keys))
    cell_mismatch = sum(per_cell.get(k, 0) != n for k, n in cells.items()) + sum(k not in cells for k in per_cell)
    total = sum(cells.values())
    if len(rows) != total or xhard0 or no_sha or dup or cell_mismatch or len(per_cell) != len(cells):
        raise ManifestError("exec", f"执行清单不符 rows={len(rows)} expected={total} xhard0={xhard0} no_sha={no_sha} "
                                    f"dup_key={dup} cell_mismatch={cell_mismatch}",
                            total=len(rows), xhard0=xhard0, duplicate=dup, cells=len(per_cell))
    return {"total": len(rows), "cells": len(per_cell), "xhard0": xhard0}


# ── 分片 ─────────────────────────────────────────────────────────────────────


def balance(rows: list[dict], shards: int) -> list[list[dict]]:
    """最长处理时间优先：按估计用时降序（同用时按 task、tier、builder_episode），逐局分给当前总用时最小的片。"""
    load = [0.0] * shards
    out: list[list[dict]] = [[] for _ in range(shards)]
    for r in sorted(rows, key=lambda r: (-TASK_SECONDS[r["task"]], r["task"], r["tier"], r["builder_episode"])):
        k = min(range(shards), key=lambda i: (load[i], i))
        out[k].append(r)
        load[k] += TASK_SECONDS[r["task"]]
    for part in out:
        part.sort(key=lambda r: (r["task"], r["tier"], r["builder_episode"]))
    return out


def check_shards(parts: list[list[dict]], rows: list[dict]) -> dict:
    want = {r["key"] for r in rows}
    seen: dict[str, int] = defaultdict(int)
    for p in parts:
        for r in p:
            seen[r["key"]] += 1
    return {"missing": len(want - set(seen)), "extra": len(set(seen) - want),
            "duplicate": sum(n - 1 for n in seen.values() if n > 1)}


# ── V9：剔除已评身份（第 5 步）───────────────────────────────────────────────


def quad(r: dict) -> tuple[str, str, int, str]:
    return (r["task"], r["tier"], int(r["seed"]), r["spec_sha256"])


def delivery_cells(delivery: dict, hs) -> dict[tuple[str, str], int]:
    """格表由交付清单逐格计数推出，且必须与登记的某张完整交付格表逐格相等（不写死局数）。"""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for d in delivery.get("rows") or []:
        counts[(d["task"], d["tier"])] += 1
    tables = [hs.EXPECTED_CELLS, *getattr(hs, "CELL_TABLES", {}).values()]
    for table in tables:
        if dict(counts) == dict(table):
            return dict(table)
    raise ManifestError("cells", f"交付清单逐格计数（{len(counts)} 格、{sum(counts.values())} 行）不等于任何登记的交付格表",
                        total=sum(counts.values()), cells=len(counts))


def load_v8_manifest(path: Path) -> dict[tuple, dict]:
    """V8 manifest ``rows`` 按四元组索引；四元组或 ``key`` 重复即停。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ManifestError("exclude", f"V8 manifest schema={doc.get('schema')!r}，期望 {SCHEMA}")
    index: dict[tuple, dict] = {}
    keys: set[str] = set()
    dup = 0
    for r in doc.get("rows") or []:
        q = quad(r)
        if q in index or r["key"] in keys:
            dup += 1
        index[q] = r
        keys.add(r["key"])
    if dup:
        raise ManifestError("exclude", f"V8 manifest 四元组或 key 重复 {dup} 处", duplicate=dup)
    return index


def in_new_rule(r: dict) -> bool:
    lo = V9_NEW_RULE.get((r["task"], r["tier"]))
    return lo is not None and int(r["candidate"]) >= lo


def split_evaluated(rows: list[dict], v8_index: dict[tuple, dict]) -> tuple[list[dict], list[dict]]:
    """第 5 步：四元组命中 V8 manifest → 复用（带 V8 ``key``），否则新评；新评集合必须恰为 ``V9_NEW_RULE``。"""
    reused, new = [], []
    for r in rows:
        hit = v8_index.get(quad(r))
        if hit is None:
            new.append(r)
        else:
            reused.append({"task": r["task"], "tier": r["tier"], "seed": int(r["seed"]),
                           "spec_sha256": r["spec_sha256"], "v8_key": hit["key"]})
    rule = {r["key"] for r in rows if in_new_rule(r)}
    got = {r["key"] for r in new}
    missing = sorted(rule - got)  # 规则内却命中 V8（被当成复用）
    extra = sorted(got - rule)    # 规则外却没命中 V8（本该复用）
    if missing or extra:
        raise ManifestError("exclude", f"新评集合不符 V9_NEW_RULE={sorted(V9_NEW_RULE.items())} "
                                       f"规则内被复用={missing[:5]} 规则外未命中 V8={extra[:5]}",
                            missing=len(missing), extra=len(extra), total=len(new), reused=len(reused))
    reused.sort(key=lambda x: (x["task"], x["tier"], x["seed"]))
    return reused, new


def reused_doc(reused: list[dict], v8_manifest: Path) -> dict:
    return {"schema": REUSED_SCHEMA, "v8_manifest": str(v8_manifest), "v8_manifest_sha256": file_sha256(v8_manifest),
            "count": len(reused), "rows": reused}


def dump_json(doc) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def v9_exec_rows(identities: Path, delivery_path: Path) -> tuple[Any, dict, list[dict], int]:
    """V9 两种模式共用的第 1～4 步：源集 → 筛新值 → 补指纹 → 核格表。返回 (hs, 格表, 执行行, 丢弃的 xhard0 数)。"""
    hs = load_hard_specs()
    delivery = json.loads(Path(delivery_path).read_text(encoding="utf-8"))
    cells_table = delivery_cells(delivery, hs)
    src = read_jsonl(identities)
    check_source(src, hs, cells_table)                     # 第 1 步
    new_all, dropped = filter_new(src, hs)                 # 第 2 步
    rows = join_delivery(new_all, delivery, hs)            # 第 3 步
    check_exec(rows, hs, cells_table)                      # 第 4 步
    return hs, cells_table, rows, dropped


def build_v9(identities: Path, delivery_path: Path, shards: int,
             v8_manifest: Path) -> tuple[dict, list[list[dict]], str]:
    """V9：源集 → 筛新值 → 补指纹 → 核格表 → 剔除已评 → 只切新评行。返回 (manifest, parts, reused.json 文本)。"""
    _, cells_table, rows, dropped = v9_exec_rows(identities, delivery_path)
    reused, new = split_evaluated(rows, load_v8_manifest(v8_manifest))  # 第 5 步
    if shards < 1:
        raise ManifestError("shards", f"shards={shards}")
    parts = balance(new, shards)
    chk = check_shards(parts, new)
    if any(chk.values()):
        raise ManifestError("shards", f"分片不符 {chk}", **chk)
    order = {k: i for i, k in enumerate(cells_table)}
    cells: dict[str, int] = defaultdict(int)
    for r in new:
        cells[cell_name(r["task"], r["tier"])] += 1
    manifest_rows = [dict(r, shard=f"{i:02d}") for i, p in enumerate(parts) for r in p]
    manifest_rows.sort(key=lambda r: (order[(r["task"], r["tier"])], r["builder_episode"]))
    reused_text = dump_json(reused_doc(reused, v8_manifest))
    manifest = {
        "schema": SCHEMA,
        "source_sha256": file_sha256(identities),
        "delivery_sha256": file_sha256(delivery_path),
        "xhard0_dropped": dropped,
        "total": len(new),
        "cells": {cell_name(*k): cells[cell_name(*k)] for k in cells_table if cells.get(cell_name(*k))},
        "shards": {f"{i:02d}": len(p) for i, p in enumerate(parts)},
        "rows": manifest_rows,
        "reused": {"path": REUSED_FILE, "sha256": hashlib.sha256(reused_text.encode("utf-8")).hexdigest(),
                   "count": len(reused), "v8_manifest": str(v8_manifest),
                   "v8_manifest_sha256": file_sha256(v8_manifest)},
    }
    return manifest, parts, reused_text


def split_shards(rows: list[dict], shards: int) -> list[list[dict]]:
    """按历史耗时均衡切片并核对两两不交、并集等于全集。"""
    if shards < 1:
        raise ManifestError("shards", f"shards={shards}")
    parts = balance(rows, shards)
    chk = check_shards(parts, rows)
    if any(chk.values()):
        raise ManifestError("shards", f"分片不符 {chk}", **chk)
    return parts


def v9_builder(task: str):
    """ood 的真实 builder（开关 XHARD0_IN_TEST_HARD 按当前进程取值；评估客户端默认关）。"""
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder(task, dataset="ood")


def verify_builder_episodes(rows: list[dict], builder_factory=None) -> None:
    """逐行用 builder 解析 ``builder_episode``，核对 tier／seed／candidate／spec_sha256 与身份行一致。

    身份清单的 ``episode`` 字段依导出时的 ``XHARD0_IN_TEST_HARD`` 开关而定（开时每任务前 12 局是 xhard0，V9 局号整体 +12）；
    与评估客户端所用 builder 的开关不一致时，局号会整体错位，客户端只能在开跑后以 IDENTITY_MISMATCH 拦下（2026-10-05 GL 实测）。
    这里在写分片前就拦住：任一行不符即 ManifestError(stage=builder)。"""
    make = builder_factory or v9_builder
    builders: dict = {}
    bad: list[str] = []
    for r in rows:
        b = builders.get(r["task"]) or builders.setdefault(r["task"], make(r["task"]))
        try:
            x = b.resolve_identity(int(r["builder_episode"]))
        except (KeyError, IndexError, ValueError):  # 局号越界（如整体错位后超出该任务局数）同样计为不符
            bad.append(f"{r['key']}@{r['builder_episode']}(越界)")
            continue
        got = (x.get("tier"), int(x["seed"]), x.get("candidate"), x.get("spec_sha256"))
        want = (r["tier"], int(r["seed"]), r.get("candidate"), r.get("spec_sha256"))
        if got != want:
            bad.append(f"{r['key']}@{r['builder_episode']}")
    if bad:
        raise ManifestError("builder", f"builder 解析与身份行不符 {len(bad)} 行，例 {bad[:3]}（身份清单导出时的 "
                            f"XHARD0_IN_TEST_HARD 开关与当前不一致？）", total=len(rows), mismatch=len(bad))


def build_v9_full(identities: Path, delivery_path: Path, shards: int,
                  builder_factory=None) -> tuple[dict, list[list[dict]]]:
    """V9 全量：第 1～4 步之后不剔除已评身份，全部执行行（V9 即 43 格 800 局）切片；写分片前逐行用 builder 核对局号。
    返回 (manifest, parts)。"""
    _, cells_table, rows, dropped = v9_exec_rows(identities, delivery_path)
    verify_builder_episodes(rows, builder_factory)
    parts = split_shards(rows, shards)
    order = {k: i for i, k in enumerate(cells_table)}
    cells: dict[str, int] = defaultdict(int)
    for r in rows:
        cells[cell_name(r["task"], r["tier"])] += 1
    manifest_rows = [dict(r, shard=f"{i:02d}") for i, p in enumerate(parts) for r in p]
    manifest_rows.sort(key=lambda r: (order[(r["task"], r["tier"])], r["builder_episode"]))
    manifest = {
        "schema": SCHEMA, "mode": "v9-full", "dataset": "ood",
        "source_sha256": file_sha256(identities), "delivery_sha256": file_sha256(delivery_path),
        "xhard0_dropped": dropped, "total": len(rows),
        "cells": {cell_name(*k): cells[cell_name(*k)] for k in cells_table if cells.get(cell_name(*k))},
        "shards": {f"{i:02d}": len(p) for i, p in enumerate(parts)}, "rows": manifest_rows,
    }
    return manifest, parts


def hard0_builder(task: str):
    """hard-verify 的真实 builder（只读官方 test 元数据的 hard 子集，不建仿真场景）。"""
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder(task, dataset="hard-verify")


def check_hard0(rows: list[dict], hs, per_task: int) -> dict:
    """hard0 执行清单核对：16 任务 × 每任务 N 行，全为 xhard0、candidate／spec_sha256 为 null、source_episode 为整数，
    key 与 (task, source_episode) 都唯一。"""
    per: dict[str, int] = defaultdict(int)
    bad = []
    for r in rows:
        per[r["task"]] += 1
        if r["tier"] != hs.XHARD0 or r["candidate"] is not None or r["spec_sha256"] is not None \
                or not _is_int(r["source_episode"]) or not _is_int(r["seed"]) or r["key"] != v8_key(r):
            bad.append(r["key"])
    keys = [r["key"] for r in rows]
    srcs = [(r["task"], r["source_episode"]) for r in rows]
    dup = (len(keys) - len(set(keys))) + (len(srcs) - len(set(srcs)))
    xhard0 = sum(r["tier"] == hs.XHARD0 for r in rows)
    total = len(hs.ALL_TASKS) * per_task
    cell_mismatch = sum(per.get(t, 0) != per_task for t in hs.ALL_TASKS) + sum(t not in hs.ALL_TASKS for t in per)
    if len(rows) != total or bad or dup or cell_mismatch or xhard0 != len(rows):
        raise ManifestError("hard0", f"hard0 清单不符 rows={len(rows)} expected={total} bad={bad[:5]} dup={dup} "
                                     f"cell_mismatch={cell_mismatch}",
                            total=len(rows), xhard0=xhard0, duplicate=dup, cells=len(per))
    return {"total": len(rows), "xhard0": xhard0, "cells": len(per)}


def build_hard0(per_task: int, shards: int, *, pair_shards: bool = False,
                builder_factory=None) -> tuple[dict, list[list[dict]]]:
    """hard0：直接由 ``BenchmarkEnvBuilder(task, dataset="hard-verify")`` 枚举（不读交付清单），每任务取前 N 局。

    行键同 ``SHARD_ROW_KEYS``：``spec_sha256=None``、``candidate=None``、``key=<task>_xhard0_<seed>``。
    ``pair_shards``：原侧（官方入口）与新侧（本仓库客户端）共用同一份 ``shard-NN.json``，同一身份两侧落在同一分片、
    分片内按 (task, builder_episode) 正序（两侧调用序列相同），manifest 记 ``paired_sides``。"""
    hs = load_hard_specs()
    if per_task < 1 or per_task > hs.XHARD0_PER_TASK:
        raise ManifestError("hard0", f"--per-task={per_task} 须在 1..{hs.XHARD0_PER_TASK}")
    make = builder_factory or hard0_builder
    rows: list[dict] = []
    for task in hs.ALL_TASKS:
        b = make(task)
        n = int(b.get_episode_num())
        if per_task > n:
            raise ManifestError("hard0", f"{task} hard-verify 只有 {n} 局，不够 --per-task={per_task}",
                                total=len(rows))
        for ep in range(per_task):
            ident = b.resolve_identity(ep)
            row = {"task": task, "tier": ident["tier"], "seed": int(ident["seed"]), "candidate": ident.get("candidate"),
                   "builder_episode": ep, "source_episode": ident.get("source_episode"),
                   "spec_sha256": ident.get("spec_sha256")}
            row["key"] = v8_key(row)
            rows.append(row)
    check_hard0(rows, hs, per_task)
    parts = split_shards(rows, shards)
    manifest_rows = [dict(r, shard=f"{i:02d}") for i, p in enumerate(parts) for r in p]
    order = {t: i for i, t in enumerate(hs.ALL_TASKS)}
    manifest_rows.sort(key=lambda r: (order[r["task"]], r["builder_episode"]))
    manifest = {
        "schema": SCHEMA, "mode": "hard0", "dataset": "hard-verify", "per_task": per_task, "total": len(rows),
        "cells": {cell_name(t, hs.XHARD0): per_task for t in hs.ALL_TASKS},
        "shards": {f"{i:02d}": len(p) for i, p in enumerate(parts)}, "rows": manifest_rows,
        "pair_shards": bool(pair_shards),
    }
    if pair_shards:
        manifest["paired_sides"] = ["orig", "new"]
    return manifest, parts


def output_paths(out_dir: Path, shards: int, reused: bool = False) -> list[Path]:
    extra = [out_dir / REUSED_FILE] if reused else []
    return [out_dir / f"shard-{i:02d}.json" for i in range(shards)] + [out_dir / "manifest.json"] + extra


def write_outputs(out_dir: Path, manifest: dict, parts: list[list[dict]], reused_text: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if reused_text is not None:
        (out_dir / REUSED_FILE).write_text(reused_text, encoding="utf-8")
    for i, p in enumerate(parts):
        doc = [{k: r[k] for k in SHARD_ROW_KEYS} for r in p]
        (out_dir / f"shard-{i:02d}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")


def remove_outputs(out_dir: Path, shards: int, reused: bool = False) -> int:
    """读回核对不过时删掉本次写出的产物（只删 manifest.json 与 shard-NN.json，V9 另删 reused.json；不碰目录里其他文件）。"""
    n = 0
    for path in output_paths(out_dir, shards, reused):
        if path.exists():
            path.unlink()
            n += 1
    return n


def verify_outputs(out_dir: Path, manifest: dict, shards: int, reused_text: str | None = None) -> dict:
    """从磁盘读回（JSON 往返）再核一次：分片两两不交、并集等于 manifest.rows；V9 另核 reused.json 与 manifest.reused。"""
    back = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    parts = [json.loads((out_dir / f"shard-{i:02d}.json").read_text(encoding="utf-8")) for i in range(shards)]
    chk = check_shards(parts, back["rows"])
    chk["roundtrip_mismatch"] = int(back != manifest) + sum(
        set(r) != set(SHARD_ROW_KEYS) for p in parts for r in p)
    if reused_text is not None:
        raw = (out_dir / REUSED_FILE).read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        meta = back.get("reused") or {}
        chk["reused_mismatch"] = int(raw != reused_text.encode("utf-8")) + int(
            hashlib.sha256(raw).hexdigest() != meta.get("sha256")) + int(
            doc.get("count") != len(doc.get("rows") or []) or doc.get("count") != meta.get("count")) + int(
            dump_json(doc) != raw.decode("utf-8"))
    return chk


MODES = ("v9-new", "v9-full", "hard0")


def _write_and_verify(out_dir: Path, manifest: dict, parts: list[list[dict]], shards: int,
                      reused_text: str | None) -> None:
    """写产物并读回核对；不过则删掉本次写出的产物并抛 ManifestError。"""
    write_outputs(out_dir, manifest, parts, reused_text)
    try:
        chk = verify_outputs(out_dir, manifest, shards, reused_text)
    except (OSError, ValueError, KeyError) as e:
        chk = {"verify_error": f"{type(e).__name__}: {e}"}
    if any(chk.values()):
        removed = remove_outputs(out_dir, shards, reused_text is not None)
        raise ManifestError("verify", f"读回核对不符 {chk}，已删本次产物 {removed} 个",
                            **{k: v for k, v in chk.items() if isinstance(v, int)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="v9-new", choices=list(MODES),
                    help="v9-new（默认，剔除 V8 已评、只切新评行）／v9-full（V9 全量 800 局）／hard0（hard-verify xhard0 局）")
    ap.add_argument("--identities", default=None, help="V9 两模式：export_eval_identities.py 的身份 JSONL（V9 992 行）")
    ap.add_argument("--delivery", default=None, help="V9 两模式：交付清单 delivery.local.json（schema v8-delivery/1）")
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude-evaluated", default=None, metavar="V8_MANIFEST",
                    help="v9-new 必填：按四元组 (task, tier, seed, spec_sha256) 剔除 V8 manifest 已评身份，只切新评行，"
                         "复用集合写 <out-dir>/reused.json")
    ap.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK, help="hard0：每任务取前 N 局（默认 12）")
    ap.add_argument("--pair-shards", action="store_true",
                    help="hard0：原侧与新侧共用同一份 shard-NN.json（同一身份两侧同分片、同序）")
    args = ap.parse_args(argv)
    if args.mode in ("v9-new", "v9-full"):
        need = [("--identities", args.identities), ("--delivery", args.delivery)]
        if args.mode == "v9-new":
            need.append(("--exclude-evaluated", args.exclude_evaluated))
        miss = [n for n, v in need if v is None]
        if miss:
            ap.error(f"--mode {args.mode} 必须给 {' '.join(miss)}")
        if args.pair_shards:
            ap.error("--pair-shards 只用于 --mode hard0")
    elif args.identities or args.delivery or args.exclude_evaluated:
        ap.error("--mode hard0 由 builder 直接枚举，不接受 --identities／--delivery／--exclude-evaluated")
    out_dir = Path(args.out_dir)
    if args.mode == "v9-new":
        return main_v9_new(args, out_dir)
    tag = "EVAL_SHARDS"
    try:
        if args.mode == "v9-full":
            manifest, parts = build_v9_full(Path(args.identities), Path(args.delivery), args.shards)
        else:
            manifest, parts = build_hard0(args.per_task, args.shards, pair_shards=args.pair_shards)
        _write_and_verify(out_dir, manifest, parts, args.shards, None)
    except ManifestError as e:
        c = e.counts
        print(f"{tag}=FAIL mode={args.mode} total={c.get('total', 'NA')} cells={c.get('cells', 'NA')} "
              f"missing={c.get('missing', 0)} extra={c.get('extra', 0)} duplicate={c.get('duplicate', 0)} "
              f"xhard0={c.get('xhard0', 'NA')} stage={e.stage} detail={e.detail}", flush=True)
        return 1
    for name, n in manifest["shards"].items():
        est = sum(TASK_SECONDS[r["task"]] for r in parts[int(name)])
        print(f"SHARD {name} rows={n} est_s={est:.0f}", flush=True)
    xhard0 = sum(r["tier"] == "xhard0" for r in manifest["rows"])
    if args.mode == "v9-full":
        print(f"{tag}=PASS mode=v9-full total={manifest['total']} cells={len(manifest['cells'])} missing=0 extra=0 "
              f"duplicate=0 xhard0={xhard0}", flush=True)
    else:
        if args.pair_shards:
            print(f"PAIR_SHARDS sides=orig,new shards={args.shards} files=shard-NN.json", flush=True)
        print(f"{tag}=PASS mode=hard0 total={manifest['total']} xhard0={xhard0} per_task={args.per_task} "
              f"cells={len(manifest['cells'])} missing=0 extra=0 duplicate=0", flush=True)
    return 0


def main_v9_new(args, out_dir: Path) -> int:
    """默认模式（行为与改动前相同）：剔除 V8 已评、只切新评行，判定行 V9_EVAL_SHARDS。"""
    tag = "V9_EVAL_SHARDS"
    try:
        manifest, parts, reused_text = build_v9(Path(args.identities), Path(args.delivery), args.shards,
                                                Path(args.exclude_evaluated))
        _write_and_verify(out_dir, manifest, parts, args.shards, reused_text)
    except ManifestError as e:
        c = e.counts
        print(f"{tag}=FAIL shards={args.shards} total={c.get('total', 'NA')} cells={c.get('cells', 'NA')} "
              f"reused={c.get('reused', 'NA')} missing={c.get('missing', 0)} extra={c.get('extra', 0)} "
              f"duplicate={c.get('duplicate', 0)} xhard0={c.get('xhard0', 'NA')} stage={e.stage} "
              f"detail={e.detail}", flush=True)
        return 1
    for name, n in manifest["shards"].items():
        est = sum(TASK_SECONDS[r["task"]] for r in parts[int(name)])
        print(f"SHARD {name} rows={n} est_s={est:.0f}", flush=True)
    meta = manifest["reused"]
    print(f"REUSED path={out_dir / REUSED_FILE} sha256={meta['sha256']} count={meta['count']} "
          f"v8_manifest_sha256={meta['v8_manifest_sha256']}", flush=True)
    print(f"{tag}=PASS shards={args.shards} total={manifest['total']} cells={len(manifest['cells'])} "
          f"reused={meta['count']} reused_sha256={meta['sha256'][:12]} missing=0 extra=0 duplicate=0 xhard0=0",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
