#!/usr/bin/env python3
"""V9 逐身份清单导出（v9 见 1002 方案 §2.4.2 第 7 步；格式沿用 v8 方案第二部分 §2.2 第 9 条，原 v7 版见 0928 方案第一部分 §5 第 3 条）。

- 经 ``robomme_hard`` 的 ``BenchmarkEnvBuilder(task, "ood")`` 逐任务列出全部局：总数与逐格局数由格表推出并核对
  （包内 ``EXPECTED_CELLS``＝V9_CELLS，43 格 800；开关打开时另加 xhard0 16 任务 × 1 档 × 12 局 = 192）。
  builder 读包内规格；``--specs-root`` 经 ``ROBOMME_HARD_SPECS_ROOT`` 覆盖。V8 的 1070 局格表与对应的 1262 局清单
  已于维护计划阶段 1b 删除（git 历史可取回）。
- ``round``／``shard`` 字段保留、一律置空（null）。
- 官方路线对照：xhard0 的 192 局（官方 test 的原 episode 号）仍按 ``TASK_SECONDS`` 贪心均衡切 10 片（3′ xhard0 评估用）。
- 行格式：``{task, episode, tier, seed, candidate, source_episode, round, shard}``；官方行 ``{task, source_episode, seed, shard}``。

**v9**（1002 方案）：总数与逐格局数改由格表推出——格表取 ``--specs-root`` 各档 header 的逐任务配额
（``hard_parity.root_cell_table``：完整 V9 根恰为 V9_CELLS），不给 ``--specs-root`` 时取包内 ``EXPECTED_CELLS``
（v9 阶段 3b 切换后即 V9）。V9：992 = 192 + 800，默认文件名 ``eval-identities-992.jsonl``，默认目录
``artifacts/newtask-v9/``。V9 运行时：

* ``--delivery`` 必给（v9 assemble 的 800 行 ``delivery.local.json``）：builder 列出的 800 个新值身份
  (task, tier, seed) 须与清单逐一相同（``delivery_mismatch``）；
* ``--official-out`` **必须显式给 V9 路径**（审计 10）：其默认值仍指向 V8 根
  ``artifacts/newtask-v8/eval-official-xhard0-192.jsonl``，V9 运行时用默认值即报错退出，避免覆写 V8 产物::

    uv run --no-sync python scripts/injection-dev/export_eval_identities.py \\
        --specs-root artifacts/newtask-v9/specs-root --delivery artifacts/newtask-v9/delivery/delivery.local.json \\
        --out artifacts/v9-evaluation/inputs/eval-identities-992.jsonl \\
        --official-out artifacts/v9-evaluation/inputs/eval-official-xhard0-192.jsonl

**xhard0 退出 OOD 数据集**（1002 号 xhard0 退出方案）：builder 前置的 xhard0 局数随开关
``hard_specs.XHARD0_IN_TEST_HARD``（环境变量 ``ROBOMME_HARD_XHARD0_IN_TEST_HARD=1`` 开）——开为每任务 12（共 192），
关（默认）为 0。关档时总数只剩新值格局数（V9 800 = 0 + 800，默认文件名 ``eval-identities-800.jsonl``），
xhard0 行须为 0；官方路线清单无行可写，``--official-out`` 跳过写文件并在判定行打 ``official=skipped``（仍可 PASS），
V9 也不再强制显式给 ``--official-out``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401  路径设置

import hard_parity  # noqa: E402  scripts/parity（_common 已加路径；纯标准库）
from robomme_hard.env_record_wrapper import hard_specs  # noqa: E402

#: 每任务平均单局用时（秒；v6 SimpleMemVLA 实测，docs/eval-doc/testhard-0928/records 1100 局 elapsed_s 均值，
#: 策略仓库内留档）。原名 V6_SECONDS，随 V6 删除改名（v8 方案第一部分 §2.5）；用于官方路线分片均衡。
TASK_SECONDS = {
    "BinFill": 224.0, "ButtonUnmask": 116.5, "ButtonUnmaskSwap": 73.4, "InsertPeg": 152.2, "MoveCube": 180.3,
    "PatternLock": 79.8, "PickHighlight": 246.7, "PickXtimes": 114.2, "RouteStick": 74.1, "StopCube": 64.9,
    "SwingXtimes": 90.8, "VideoPlaceButton": 58.0, "VideoPlaceOrder": 61.7, "VideoRepick": 51.9, "VideoUnmask": 97.4,
    "VideoUnmaskSwap": 50.5,
}
SHARDS = 10


def _xhard0_prefix() -> int:
    """builder 每任务前置的 xhard0 局数：开关开 12、关 0（S1 合并前的旧 hard_specs 无开关，按 12）。"""
    return getattr(hard_specs, "xhard0_prefix", lambda: hard_specs.XHARD0_PER_TASK)()


def xhard0_total() -> int:
    """xhard0 总局数：16 任务 × 前置局数（开关开 192、关 0）；运行时读开关，不在导入时冻结。"""
    return len(hard_specs.ALL_TASKS) * _xhard0_prefix()


#: 导入时刻的 xhard0 总数（兼容旧引用；判据一律走 :func:`xhard0_total`）
XHARD0_TOTAL = xhard0_total()
#: 官方路线 xhard0 清单的默认路径（V8 根；V9 必须显式给，审计 10）
OFFICIAL_OUT_DEFAULT = "artifacts/newtask-v8/eval-official-xhard0-192.jsonl"


def expected_total(cells: dict[tuple[str, str], int]) -> int:
    """格表推出的总局数：xhard0 16 × 前置局数 + 新值格局数之和（开关开：V9 992 = 192 + 800；
    开关关：V9 800）。"""
    return xhard0_total() + sum(cells.values())


def identities_name(cells: dict[tuple[str, str], int]) -> str:
    return f"eval-identities-{expected_total(cells)}.jsonl"


#: 包内当前格表（EXPECTED_CELLS）与导入时开关推出的总局数与默认文件名（V9：开关开 992、关 800）
EXPECTED_TOTAL = expected_total(hard_specs.EXPECTED_CELLS)
IDENTITIES_NAME = identities_name(hard_specs.EXPECTED_CELLS)


def balance(rows: list[dict], shards: int) -> None:
    """最长处理时间优先的贪心：按估计用时降序，逐局分给当前总用时最小的片（同用时取片号小者），就地写 ``shard``。"""
    load = [0.0] * shards
    for row in sorted(rows, key=lambda r: (-TASK_SECONDS[r["task"]], r["task"], r.get("episode", r.get("source_episode")))):
        k = min(range(shards), key=lambda i: (load[i], i))
        row["shard"] = k
        load[k] += TASK_SECONDS[row["task"]]


def check_rows(rows: list[dict], official: list[dict], cells: dict[tuple[str, str], int] | None = None,
               delivery_ids: set[tuple[str, str, int]] | None = None) -> tuple[bool, dict]:
    """总数 = :func:`expected_total`（开关开 V9 992，关 800）、xhard0 = :func:`xhard0_total`（192／0）、逐格局数等于格表（缺省 EXPECTED_CELLS）、
    round／shard 全空、官方行数 = xhard0 总数；给 ``delivery_ids`` 时新值身份 (task, tier, seed) 须与之逐一相同。"""
    cells = hard_specs.EXPECTED_CELLS if cells is None else cells
    per_cell: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r["tier"] != hard_specs.XHARD0:
            per_cell[(r["task"], r["tier"])] += 1
    tiers: dict[str, int] = defaultdict(int)
    for r in rows:
        tiers[r["tier"]] += 1
    facts = {"episodes": len(rows), "xhard0": tiers.get(hard_specs.XHARD0, 0), "cells": len(per_cell),
             "cell_mismatch": sum(per_cell.get(k, 0) != n for k, n in cells.items())
             + sum(k not in cells for k in per_cell),
             "round_or_shard_set": sum(r.get("round") is not None or r.get("shard") is not None for r in rows),
             "official": len(official), "tiers": dict(sorted(tiers.items())), "expected": expected_total(cells)}
    if delivery_ids is not None:
        new_ids = {(r["task"], r["tier"], int(r["seed"])) for r in rows if r["tier"] != hard_specs.XHARD0}
        facts["delivery_mismatch"] = len(new_ids ^ delivery_ids)
    x0_total = xhard0_total()
    ok = (facts["episodes"] == facts["expected"] and facts["xhard0"] == x0_total and facts["cell_mismatch"] == 0
          and facts["round_or_shard_set"] == 0 and facts["official"] == x0_total
          and facts.get("delivery_mismatch", 0) == 0)
    return ok, facts


def resolve_table(specs_root: str | None) -> tuple[str, dict[tuple[str, str], int]]:
    """格表：``--specs-root`` 的 header 逐任务配额推出（V9 根 → V9_CELLS），否则包内 EXPECTED_CELLS。"""
    if specs_root:
        found = hard_parity.root_cell_table(specs_root, hard_specs)
        if found is not None:
            return found
    return hard_parity.table_version(hard_specs.EXPECTED_CELLS, hard_specs), hard_specs.EXPECTED_CELLS


def read_delivery_ids(path: str) -> set[tuple[str, str, int]]:
    """交付清单（v8-delivery/1 或 v9 的 800 行清单）的新值身份集合 (task, tier, seed)。"""
    return {hard_parity.ident(r) for r in hard_parity.read_delivery(Path(path))["rows"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="逐身份清单输出；缺省 artifacts/newtask-<版本>/eval-identities-<总数>.jsonl"
                             "（V9 → newtask-v9/eval-identities-992.jsonl，开关关时 eval-identities-800.jsonl）")
    parser.add_argument("--official-out", default=OFFICIAL_OUT_DEFAULT,
                        help=f"官方路线 xhard0 192 局清单输出；默认 {OFFICIAL_OUT_DEFAULT} 是 V8 根——"
                             "V9 必须显式给（如 artifacts/v9-evaluation/inputs/eval-official-xhard0-192.jsonl），"
                             "V9 运行时用默认值即报错退出（审计 10）")
    parser.add_argument("--specs-root", default=None,
                        help="覆盖规格根（经 ROBOMME_HARD_SPECS_ROOT）；格表按其各档 header 的逐任务配额推出（V9 根 → 992）")
    parser.add_argument("--delivery", default=None,
                        help="交付清单 json（V9 必给：v9 assemble 的 800 行 delivery.local.json）；新值身份须与之逐一相同")
    args = parser.parse_args(argv)
    version, cells = resolve_table(args.specs_root)
    if version == "v9":
        if args.delivery is None:
            parser.error("V9 格表须给 --delivery（v9 assemble 的 800 行交付清单）")
        if args.official_out == OFFICIAL_OUT_DEFAULT and xhard0_total() > 0:
            parser.error(f"V9 格表须显式给 --official-out（默认值 {OFFICIAL_OUT_DEFAULT} 是 V8 根，不得覆写）")
    out = args.out or f"artifacts/newtask-{version}/{identities_name(cells)}"
    delivery_ids = read_delivery_ids(args.delivery) if args.delivery else None
    if args.specs_root:
        os.environ[hard_specs.SPECS_ROOT_ENV] = str(Path(args.specs_root).resolve())
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    rows = []
    for task in hard_specs.ALL_TASKS:
        builder = BenchmarkEnvBuilder(task, dataset="ood")
        items = []
        for ep in range(builder.get_episode_num()):
            ident = builder.resolve_identity(ep)
            items.append({"task": task, "episode": ep, "tier": ident["tier"], "seed": ident["seed"],
                          "candidate": ident.get("candidate"), "source_episode": ident.get("source_episode"),
                          "round": None, "shard": None})
        rows.extend(items)
    rows.sort(key=lambda r: (hard_specs.ALL_TASKS.index(r["task"]), r["episode"]))
    official = [{"task": r["task"], "source_episode": r["source_episode"], "seed": r["seed"]}
                for r in rows if r["tier"] == hard_specs.XHARD0]
    balance(official, SHARDS)
    official.sort(key=lambda r: (r["shard"], hard_specs.ALL_TASKS.index(r["task"]), r["source_episode"]))
    # 开关关闭（xhard0 不在 ood）时官方路线清单无行：跳过写文件，判定行打 official=skipped
    skip_official = xhard0_total() == 0
    writes = [(out, rows)] + ([] if skip_official else [(args.official_out, official)])
    for path, data in writes:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in data))
    ok, facts = check_rows(rows, official, cells, delivery_ids)
    print(f"EVAL_IDENTITY_EXPORT={'PASS' if ok else 'FAIL'} version={version} episodes={facts['episodes']} "
          f"expected={facts['expected']} "
          f"xhard0={facts['xhard0']} cells={facts['cells']} cell_mismatch={facts['cell_mismatch']} "
          + (f"delivery_mismatch={facts['delivery_mismatch']} " if "delivery_mismatch" in facts else "") +
          f"round_or_shard_set={facts['round_or_shard_set']} official={'skipped' if skip_official else facts['official']} shards={SHARDS} "
          f"tiers={facts['tiers']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
