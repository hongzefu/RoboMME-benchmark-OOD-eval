#!/usr/bin/env python3
"""第二阶段入口：只读 jsonl → 生成 h5 → 按状态机回写（0927 计划第一部分 §5.2；v8 方案第二部分 §2.2 第 7 条）。

    # 正常生产：只接受 hard-specs/4 规格根（<root>/<tier>/specs.jsonl），按 --cells 逐格跑（V8_DELIVERY_SET）
    uv run --no-sync python scripts/injection-dev/generate_h5.py --mode continue \\
        --specs <规格根> --cells v9smoke --output <输出目录> --workers 1 --gpu 0
    # 分片：切片 → 片内 continue → 聚合（V9 的 MoveCube 片即 --cells v9shard1）
    uv run --no-sync python scripts/injection-dev/generate_h5.py --mode split \\
        --specs <冻结根> --cells v9shard1 --output <gen1>/shard-movecube
    uv run --no-sync python scripts/injection-dev/generate_h5.py --mode continue \\
        --specs <gen1>/shard-movecube/specs --cells v9shard1 --output <gen1>/shard-movecube --workers 4 --gpu 0
    # 只重算聚合（规格根 + 各片账本目录）；整树搬迁后（GL NFS → 本机 /data）用 --rebase 换 h5／mp4 前缀并逐个核 sha256
    uv run --no-sync python scripts/injection-dev/generate_h5.py --mode aggregate --specs <规格根> --cells v9 \\
        --shards <账本目录,...> --rebase <旧前缀>=<新前缀> --output <目录> [--out <新 delivery.json 路径>]
    # 对拍专用：按身份清单只读重放，不递补、不回写（--specs 给 /4 规格根或单文件，缺省读包内）
    uv run --no-sync python scripts/injection-dev/generate_h5.py --mode replay \\
        --identities <gen1 的 delivery.json 或 jsonl> --specs <规格根> --output <输出目录>

- ``--cells``：``v9``（V9_CELLS 43 格 800，continue／aggregate 缺省；split 必填）、``v9shard1``、``v9smoke``
  或格表 JSON 路径（``{"Task@tier": 局数}``）。规格根的任务集合与逐格配额必须与格表相等。
- 退出码：continue／aggregate 认 ``V8_DELIVERY_SET=PASS``；其余按原口径。
- 维护计划 W2 起删除：单文件 ``/2`` continue（及其 ``--redo``／``--tasks``／``--self-check``）、``/3`` v7 规格根的
  continue 与 replay、``--mode merge``（V8 四席合并）。
- ``--resume``：沿用已有 ``--output`` 续跑；有 h5 却无 partial 记录的身份标 UNKNOWN 并停止；v8 另按账本
  ``<output>/results.jsonl`` 重放已有结果、基础设施重试计数跨重启保留。
- 环境包由 ``--pkg``（默认 robomme_hard）经 ``ROBOMME_ENV_PACKAGE`` 传给 worker。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _common  # noqa: F401  路径设置
import _rollout  # noqa: E402

from robomme_hard.env_record_wrapper import hard_specs  # noqa: E402


def _launch_facts(src_root: Path) -> dict[str, str]:
    """GPU 型号、驱动、src_commit、节点与作业号（写进 launch 记录，gen1／gen2 比对前核对同型号同驱动）。"""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if out and visible and visible.isdigit() and int(visible) < len(out):
        out = [out[int(visible)]]
    name, driver = (out[0].split(",")[0].strip(), out[0].split(",")[1].strip()) if out else ("unknown", "unknown")
    commit = subprocess.run(["git", "-C", str(src_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"gpu_model": name, "driver": driver, "src_commit": commit or "unknown", "host": os.uname().nodename,
            "slurm_job": os.environ.get("SLURM_JOB_ID")}


def _git_head(src_root: Path) -> str:
    return subprocess.run(["git", "-C", str(src_root), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip() or "unknown"


def _shard_dirs(text: str | None) -> list[Path]:
    dirs = [Path(item.strip()) for item in (text or "").split(",") if item.strip()]
    if not dirs:
        raise SystemExit("--shards 必须给出逗号分隔的分片输出目录")
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("continue", "replay", "split", "aggregate"))
    parser.add_argument("--specs", default=None,
                        help="continue：/4 规格根；split：/4 冻结根；aggregate：/4 规格根；"
                             "replay：/4 规格根或单文件，缺省读包内")
    parser.add_argument("--cells", default=None,
                        help="格表：v9／v9shard1／v9smoke／格表 JSON 路径（continue／split／aggregate；"
                             "continue／aggregate 缺省 v9，split 必填；V8 专用的 full／smoke 已删除）")
    parser.add_argument("--shards", default=None, help="aggregate：逗号分隔的分片输出目录（各含 specs/、shard.json、results.jsonl）")
    parser.add_argument("--rebase", action="append", default=[], metavar="OLD=NEW",
                        help="aggregate：整树搬迁后把规格里记录的 h5／mp4 路径前缀 OLD 换成 NEW（可重复；"
                             "替换后逐个核存在与 sha256，无前缀匹配或不符即该格 FAIL）")
    parser.add_argument("--out", default=None, help="aggregate：delivery.json 输出路径（缺省 <output>/delivery.json；已存在即拒绝）")
    parser.add_argument("--identities", default=None, help="replay 模式的身份清单（delivery.json、final-delivery.json 或 jsonl）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--pkg", default="robomme_hard", choices=("robomme", "robomme_hard"))
    parser.add_argument("--src-root", default=str(_common.REPO_ROOT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dev-smoke", action="store_true", help="放行非 A40（本机开发冒烟）；正式生成一律 A40（v7 D-9）")
    args = parser.parse_args()

    output = Path(args.output)
    src_root = Path(args.src_root).resolve()

    # ── 不起仿真的两个子模式：切片、聚合 ──
    if args.mode == "split":
        if not args.specs or not args.cells:
            raise SystemExit("split 模式必须给 --specs（冻结根）与 --cells（本片格表）")
        _rollout.split_v8(Path(args.specs), _rollout.resolve_cells(args.cells), output, label=str(args.cells)
                          if not Path(str(args.cells)).is_file() else Path(args.cells).stem)
        return 0
    if args.mode == "aggregate":
        if not args.specs:
            raise SystemExit("aggregate 模式必须给 --specs（v8 规格根）")
        report = _rollout.aggregate_v8(Path(args.specs), _rollout.resolve_cells(args.cells or "v9"),
                                       _shard_dirs(args.shards), Path(args.out) if args.out else output / "delivery.json",
                                       cells_label=str(args.cells or "v9"), code_baseline=_git_head(src_root),
                                       rebase=_rollout.parse_rebase(args.rebase))
        print(report["line"], flush=True)
        return 0 if report["line"].startswith("V8_DELIVERY_SET=PASS") else 1

    facts = _launch_facts(src_root)
    if not args.dev_smoke and "A40" not in facts["gpu_model"]:
        raise SystemExit(f"正式生成只许在 A40 上跑：当前 {facts}")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"launch-{int(time.time())}.json").write_text(json.dumps(
        {"schema": "generate-h5-launch/1", "mode": args.mode, "specs": args.specs, "identities": args.identities,
         "cells": args.cells, "workers": args.workers, "gpu": args.gpu, "pkg": args.pkg, "dev_smoke": args.dev_smoke,
         **facts}, ensure_ascii=False, indent=2) + "\n")
    specs_dir = Path(args.specs) if args.specs and Path(args.specs).is_dir() else None
    root_schema = _rollout.detect_root_schema(specs_dir) if specs_dir is not None else None
    if specs_dir is not None and root_schema is None:
        raise SystemExit(f"{specs_dir} 下没有任何 <tier>/specs.jsonl")
    if args.mode == "replay":
        if not args.identities:
            raise SystemExit("replay 模式必须给 --identities")
        specs_paths = None
        if specs_dir is not None:
            if root_schema != hard_specs.SCHEMA:
                raise SystemExit(f"{specs_dir} 的 schema 为 {root_schema}，replay 的规格根只支持 {hard_specs.SCHEMA}")
            # 规格根按 TIERS 读（含 xhard5）；给了 --cells 就按格表校验，否则逐档单文件校验
            if args.cells:
                tiers = list(_rollout.load_v8_root(specs_dir, _rollout.resolve_cells(args.cells)))
            else:
                tiers = [t for t in hard_specs.TIERS if (specs_dir / t / "specs.jsonl").is_file()]
                for tier in tiers:
                    _rollout.load_specs_any(specs_dir / tier / "specs.jsonl", check_fingerprint=False)
            specs_paths = {tier: specs_dir / tier / "specs.jsonl" for tier in tiers}
        elif args.specs:
            header, _ = _rollout.load_specs_any(args.specs, check_fingerprint=False)
            specs_paths = {header["difficulty"]: Path(args.specs)}
        summary = _rollout.run_replay(_rollout.load_identities(Path(args.identities)), output, src_root=src_root,
                                      workers=args.workers, gpu=args.gpu, pkg=args.pkg, resume=args.resume,
                                      specs_paths=specs_paths)
        print(f"GENERATE_REPLAY_DONE scheduled={summary['scheduled']} ok={summary['ok']} failed={summary['failed']} "
              f"out={output}")
        return 0
    if not args.specs:
        raise SystemExit("continue 模式必须给 --specs")
    if specs_dir is None or root_schema != hard_specs.SCHEMA:
        raise SystemExit(f"continue 只接受 {hard_specs.SCHEMA} 规格根（<root>/<tier>/specs.jsonl）配 --cells："
                         f"{args.specs}（schema={root_schema}）")
    # gen1：按格表逐档逐格跑，执行步超限过滤与同格递补，收尾写 delivery.json（V8_DELIVERY_SET）
    cells_label = str(args.cells or "v9")
    summary = _rollout.run_continue_v8(specs_dir, _rollout.resolve_cells(cells_label), output, src_root=src_root,
                                       workers=args.workers, gpu=args.gpu, pkg=args.pkg,
                                       code_baseline=facts["src_commit"], resume=args.resume,
                                       cells_label=cells_label)
    print(f"GENERATE_CONTINUE_DONE attempted={summary['attempted']} rounds={summary['rounds']} "
          f"infra_retries={summary['infra_retries']} delivered={summary['delivered']} "
          f"exec_over_cap={summary['exec_over_cap']} backfills={summary['backfills']} out={output}")
    return 0 if summary["delivery_set"].startswith("V8_DELIVERY_SET=PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
