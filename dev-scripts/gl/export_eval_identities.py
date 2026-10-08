#!/usr/bin/env python3
"""评估身份清单导出：经子模块 ``robomme_hard`` 的 ``BenchmarkEnvBuilder(task, dataset)`` 逐局列出执行身份行（JSONL）。

拆仓（1008 拆分方案第二部分 §二 ``dev-scripts/gl/`` 行、runbook S5）后 benchmark 只认两个评估数据集：

* ``hard-verify``：只含 xhard0，即官方 test 元数据里每任务 ``difficulty=="hard"`` 的 12 局（官方原 episode 3, 7, …,
  47）；builder 局号 0～11，``source_episode`` 为官方原号，``candidate``／``spec_sha256`` 为 null；
* ``ood``：五个新值档（V9 43 格 800 局），builder 局号每任务 0～49，``candidate`` 为整数、``spec_sha256`` 为 64 位串，
  ``source_episode`` 为 null。

xhard0 不再前置到 ood（旧的 xhard0 前置开关、规格根覆盖与官方路线清单参数已随拆仓删除），xhard0 期望行数一律按数据集名定。

行格式（GL 席位 ``seat.py`` 的动态队列与 ``eval_manifest.py`` 都直接读）::

    {"dataset", "task", "episode", "builder_episode", "tier", "seed", "candidate", "source_episode", "spec_sha256", "key"}

``episode`` 与 ``builder_episode`` 同值（builder 局号）；``key = <task>_<tier>_<seed>``。

用法::

    # S5 拆仓验收的两行身份：VideoUnmask hard-verify 局 0（官方原号 3）与 ood 局 0
    python dev-scripts/gl/export_eval_identities.py --dataset hard-verify,ood --tasks VideoUnmask --episodes 0:1 \\
        --out configs/split-accept-identities.jsonl
    # 全量：hard-verify 16 × 12 = 192 与 ood 800
    python dev-scripts/gl/export_eval_identities.py --dataset hard-verify,ood --out <路径>

核对（任一不过即 FAIL、退出 1，文件照写以便排查）：

* 行数等于「各数据集 Σ 任务 min(局号上界, 该任务局数) − 下界」；全量时 hard-verify = 16 × 12、ood 逐格等于包内
  ``EXPECTED_CELLS``（V9 43 格 800）；
* hard-verify 行全为 xhard0、candidate／spec_sha256 为 null、source_episode ∈ ``XHARD0_EPISODES``；ood 行无 xhard0、
  candidate 为整数、spec_sha256 为 64 位串；
* ``(dataset, key)`` 唯一；给 ``--delivery`` 时 ood 全量的新值身份 (task, tier, seed) 须与交付清单逐一相同。

末行 ``EVAL_IDENTITY_EXPORT=PASS|FAIL datasets=<…> episodes=<n> hard_verify=<n> ood=<n> xhard0=<n> tasks=<n>
cell_mismatch=<n> bad=<n> dup_keys=<n> [delivery_mismatch=<n>] count_mismatch=<n> out=<路径>``。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HARD_VERIFY = "hard-verify"
OOD = "ood"
DATASETS = (HARD_VERIFY, OOD)
#: 行字段
ROW_KEYS = ("dataset", "task", "episode", "builder_episode", "tier", "seed", "candidate", "source_episode",
            "spec_sha256", "key")
#: 每任务平均单局用时（秒；v6 SimpleMemVLA 实测，docs/eval-doc/testhard-0928/records 1100 局 elapsed_s 均值）。
#: ``eval_manifest.py`` 的分片均衡复制同一张表。
TASK_SECONDS = {
    "BinFill": 224.0, "ButtonUnmask": 116.5, "ButtonUnmaskSwap": 73.4, "InsertPeg": 152.2, "MoveCube": 180.3,
    "PatternLock": 79.8, "PickHighlight": 246.7, "PickXtimes": 114.2, "RouteStick": 74.1, "StopCube": 64.9,
    "SwingXtimes": 90.8, "VideoPlaceButton": 58.0, "VideoPlaceOrder": 61.7, "VideoRepick": 51.9, "VideoUnmask": 97.4,
    "VideoUnmaskSwap": 50.5,
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def hard_specs():
    from robomme_hard.env_record_wrapper import hard_specs as hs

    return hs


def make_builder(task: str, dataset: str):
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder(task, dataset=dataset)


def parse_episodes(text: str | None) -> tuple[int, int | None]:
    """``a:b`` 半开区间（``0:1`` 即局 0）；缺省全部。"""
    if not text:
        return 0, None
    a, sep, b = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"--episodes 须为 a:b 半开区间：{text!r}")
    lo, hi = int(a or 0), (int(b) if b else None)
    if lo < 0 or (hi is not None and hi <= lo):
        raise argparse.ArgumentTypeError(f"--episodes 区间为空或非法：{text!r}")
    return lo, hi


def identity_row(dataset: str, task: str, ep: int, ident: dict) -> dict:
    """builder 解析结果 → 执行身份行。"""
    src = ident.get("source_episode")
    return {"dataset": dataset, "task": task, "episode": int(ep), "builder_episode": int(ep), "tier": ident["tier"],
            "seed": int(ident["seed"]), "candidate": ident.get("candidate"),
            "source_episode": None if src is None else int(src), "spec_sha256": ident.get("spec_sha256"),
            "key": f"{task}_{ident['tier']}_{int(ident['seed'])}"}


def export_rows(datasets: list[str], tasks: list[str], lo: int, hi: int | None,
                builder_factory=make_builder) -> tuple[list[dict], dict]:
    """逐数据集、逐任务列身份行；返回 (行, 每数据集期望行数)。"""
    rows: list[dict] = []
    expected: dict[str, int] = {}
    for ds in datasets:
        n_ds = 0
        for task in tasks:
            b = builder_factory(task, ds)
            n = int(b.get_episode_num())
            top = n if hi is None else min(hi, n)
            for ep in range(lo, top):
                rows.append(identity_row(ds, task, ep, b.resolve_identity(ep)))
            n_ds += max(0, top - lo)
        expected[ds] = n_ds
    return rows, expected


def check_rows(rows: list[dict], expected: dict[str, int], hs, *, full: bool,
               delivery_ids: set[tuple[str, str, int]] | None = None) -> tuple[bool, dict]:
    """逐行与总数核对；``full``（全部任务、全部局号）时另按数据集名核 xhard0 总数与 ood 逐格局数。"""
    bad: list[str] = []
    per_ds: dict[str, int] = defaultdict(int)
    per_cell: dict[tuple[str, str], int] = defaultdict(int)
    xhard0 = 0
    for i, r in enumerate(rows):
        per_ds[r.get("dataset")] += 1
        if set(r) != set(ROW_KEYS):
            bad.append(f"row{i} keys")
            continue
        if r["key"] != f"{r['task']}_{r['tier']}_{r['seed']}" or r["episode"] != r["builder_episode"]:
            bad.append(f"row{i} key/episode")
        if r["tier"] == hs.XHARD0:
            xhard0 += 1
        if r["dataset"] == HARD_VERIFY:
            if (r["tier"] != hs.XHARD0 or r["candidate"] is not None or r["spec_sha256"] is not None
                    or r["source_episode"] not in tuple(hs.XHARD0_EPISODES)):
                bad.append(f"row{i} hard-verify 行须 xhard0、candidate／spec_sha256 为 null、source_episode 为官方 hard 局号")
        else:
            if (r["tier"] == hs.XHARD0 or not isinstance(r["candidate"], int) or r["source_episode"] is not None
                    or not isinstance(r["spec_sha256"], str) or not _SHA_RE.match(r["spec_sha256"])):
                bad.append(f"row{i} ood 行须为新值档、candidate 为整数、spec_sha256 为 64 位串、source_episode 为 null")
            per_cell[(r["task"], r["tier"])] += 1
    keys = [(r.get("dataset"), r.get("key")) for r in rows]
    facts = {"episodes": len(rows), "hard_verify": per_ds.get(HARD_VERIFY, 0), "ood": per_ds.get(OOD, 0),
             "xhard0": xhard0, "bad": len(bad), "dup_keys": len(keys) - len(set(keys)), "cell_mismatch": 0,
             "count_mismatch": sum(per_ds.get(ds, 0) != n for ds, n in expected.items()), "first_bad": bad[:3]}
    if full:
        if HARD_VERIFY in expected and per_ds.get(HARD_VERIFY, 0) != len(hs.ALL_TASKS) * int(hs.XHARD0_PER_TASK):
            facts["count_mismatch"] += 1
        if OOD in expected:
            cells = hs.EXPECTED_CELLS
            facts["cell_mismatch"] = (sum(per_cell.get(k, 0) != n for k, n in cells.items())
                                      + sum(k not in cells for k in per_cell))
            if delivery_ids is not None:
                new_ids = {(r["task"], r["tier"], int(r["seed"])) for r in rows if r.get("dataset") == OOD}
                facts["delivery_mismatch"] = len(new_ids ^ delivery_ids)
    ok = (not bad and facts["dup_keys"] == 0 and facts["count_mismatch"] == 0 and facts["cell_mismatch"] == 0
          and facts.get("delivery_mismatch", 0) == 0)
    return ok, facts


def read_delivery_ids(path: str) -> set[tuple[str, str, int]]:
    """交付清单（``{"rows": [...]}``，每行含 task／tier／seed）的新值身份集合 (task, tier, seed)。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(r["task"], r["tier"], int(r["seed"])) for r in doc["rows"]}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=",".join(DATASETS), help="逗号分隔：hard-verify、ood（缺省两者，按此顺序写出）")
    p.add_argument("--tasks", default=None, help="逗号分隔任务名；缺省官方 16 任务（规范序）")
    p.add_argument("--episodes", type=parse_episodes, default=(0, None), help="builder 局号半开区间 a:b；缺省全部")
    p.add_argument("--out", required=True, help="身份清单 JSONL 输出路径")
    p.add_argument("--delivery", default=None, help="可选：交付清单 json；ood 全量时新值身份须与之逐一相同")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    if not datasets or any(d not in DATASETS for d in datasets) or len(set(datasets)) != len(datasets):
        parser.error(f"--dataset 只能是 {'、'.join(DATASETS)}（可逗号分隔两者），得到 {args.dataset!r}")
    hs = hard_specs()
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else list(hs.ALL_TASKS)
    unknown = [t for t in tasks if t not in hs.ALL_TASKS]
    if unknown or len(set(tasks)) != len(tasks):
        parser.error(f"--tasks 含未知或重复任务：{unknown or tasks}")
    lo, hi = args.episodes
    full = args.tasks is None and lo == 0 and hi is None
    rows, expected = export_rows(datasets, tasks, lo, hi)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    delivery_ids = read_delivery_ids(args.delivery) if args.delivery else None
    ok, facts = check_rows(rows, expected, hs, full=full, delivery_ids=delivery_ids)
    print(f"EVAL_IDENTITY_EXPORT={'PASS' if ok else 'FAIL'} datasets={','.join(datasets)} episodes={facts['episodes']} "
          f"hard_verify={facts['hard_verify']} ood={facts['ood']} xhard0={facts['xhard0']} tasks={len(tasks)} "
          f"cell_mismatch={facts['cell_mismatch']} bad={facts['bad']} dup_keys={facts['dup_keys']} "
          + (f"delivery_mismatch={facts['delivery_mismatch']} " if "delivery_mismatch" in facts else "")
          + f"count_mismatch={facts['count_mismatch']} out={out}"
          + (f" first_bad={facts['first_bad']}" if facts["first_bad"] else ""), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
