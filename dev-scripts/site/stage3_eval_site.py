#!/usr/bin/env python3
"""第三阶段 OOD 视频站：按 Task 逐局并排显示生成真值、GroundSG+Oracle 与五个模型的评估视频，附成功率表。

只放视频与成功率，不放语言记录。版式沿用 ``official_overlay.html``（8083 Oracle 站）。五模型视频从 GL 运行根
（NFS）复制进本机媒体根，Oracle（1004 轮本机）与生成真值（V9 交付）本在 ``artifacts/`` 下直接引用。

    uv run --no-sync python scripts/injection-dev/site/stage3_eval_site.py --run-root <R> --site-dir <A>/site-stage3 --copy-root <A>/site-media
    uv run --no-sync python scripts/injection-dev/site/stage3_eval_site.py --serve --site-dir <A>/site-stage3 --port 8084
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
TASKS = ("BinFill", "PickXtimes", "SwingXtimes", "PickHighlight", "VideoUnmask", "ButtonUnmask",
         "VideoUnmaskSwap", "ButtonUnmaskSwap", "VideoRepick", "PatternLock", "RouteStick",
         "VideoPlaceButton", "VideoPlaceOrder", "MoveCube", "InsertPeg", "StopCube")
SCHEMA = "stage3-eval-site/2"
FINAL = ("success", "fail", "timeout")
# (站内 ID, 展示名, 发布目录, 发布清单 src_rel 的候选根)；PonderPounce 用 FIX-3 后的 r2 重跑，MemER 用已接受视图。
MODELS = (
    ("smvla", "SimpleMemVLA", "publish/smvla/seed7/videos", ("media/smvla/ood/new",)),
    ("pp", "PonderPounce", "publish-pp-r2/pp/seed7/videos", ("media-pp-r2/pp/ood/new",)),
    ("groundsg-qwenvl", "GroundSG+QwenVL", "publish/groundsg/seed7/qwenvl/videos",
     ("media/groundsg-ground-sg-qwenvl/ood/new",)),
    ("groundsg-memer", "MemER", "publish/groundsg/seed7/memer/videos", ("reports/memer/accepted-view",)),
    ("framesamp", "FrameSamp+Modulation", "publish/perceptual-framesamp-modul/seed7/videos",
     ("media/perceptual-framesamp-modul/ood/new",)),
)
ORACLE_SITE = ARTIFACTS / "sg-evaluation/sg-eval-gl-20261004-01/official-overlay/site-tasks"
GT_SITE = ARTIFACTS / "newtask-v9/site"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def media_id(kind: str, name: str) -> str:
    return f"{kind}_{hashlib.sha256(name.encode()).hexdigest()[:32]}"


def locate(root: Path, bases: tuple[str, ...], rel: str) -> Path:
    """src_rel 必须在候选根里恰好命中一个实体文件。"""
    hits = [root / base / rel for base in bases if (root / base / rel).is_file()]
    require(len(hits) == 1, f"源视频定位不唯一：{rel}（命中 {len(hits)}）")
    return hits[0]


def trace_ends(path: Path) -> tuple[dict, dict, dict]:
    """只读轨迹头部（header、demo）与末行（end），不整份读 NFS 上的大文件。"""
    header = demo = None
    with path.open(encoding="utf-8") as stream:
        for _, line in zip(range(20), stream):
            row = json.loads(line)
            if row.get("kind") == "header":
                header = row
            elif row.get("kind") == "demo":
                demo = row
                break
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        end = json.loads(stream.read().decode("utf-8").rstrip("\n").rsplit("\n", 1)[-1])
    require(header is not None and demo is not None and end.get("kind") == "end", f"轨迹首尾不完整：{path}")
    return header, demo, end


def copy_in(src: Path, dst: Path) -> Path:
    """NFS 视频复制进本机媒体根；已存在且字节数相同则复用。"""
    if not (dst.is_file() and dst.stat().st_size == src.stat().st_size):
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
    require(dst.stat().st_size == src.stat().st_size, f"复制后字节数不符：{dst}")
    return dst


def model_runs(root: Path, model: str, bases: tuple[str, ...], publish: Path, copy_root: Path,
               media: dict) -> dict:
    rows = list(csv.DictReader((publish / "index.tsv").open(encoding="utf-8"), delimiter="\t"))
    require(len(rows) == 86, f"{model} 发布清单局数 {len(rows)} 不是 86")
    runs = {}
    for row in rows:
        key, status = row["key"], row["terminal"]
        require(status in FINAL and key not in runs, f"{model} 终态或重复：{key}")
        official = locate(root, bases, row["src_rel"])
        # 发布目录是硬链接：同 inode 即已验收的那一份，不再全量哈希。
        require(official.stat().st_ino == (publish / row["dst_name"]).stat().st_ino, f"{model} 发布与源不是同一文件：{key}")
        episode_dir = official.parent.parent
        original = episode_dir / "episode.mp4"
        require(original.is_file() and original.stat().st_size > 0, f"{model} 缺原始视频：{key}")
        header, demo, end = trace_ends(episode_dir / "trace.jsonl")
        require(end.get("status") == status and header.get("identity", {}).get("key") == key,
                f"{model} 轨迹终态或身份与发布清单不符：{key}")
        name = f"{model}/{episode_dir.name}"
        run = {"attempt": episode_dir.name, "status": status, "exec_steps": end["exec_steps"],
               "demo_frames": end["demo_frames"], "frames": end["frames_recorded"],
               "task_goal": (demo.get("texts") or [""])[0]}
        for kind, src in (("official", official), ("original", original)):
            run[f"{kind}_media"] = media_id(kind, name)
            media[run[f"{kind}_media"]] = str(copy_in(src, copy_root / model / episode_dir.name / f"{kind}.mp4"))
        runs[key] = run
    return runs


def oracle_runs(keys: set, media: dict) -> dict:
    """1004 轮本机 GroundSG+Oracle（1600 步）：取 8083 站已逐局验全的目录与白名单，只留本轮 86 局。"""
    catalog = json.loads((ORACLE_SITE / "catalog.json").read_text(encoding="utf-8"))
    private = json.loads((ORACLE_SITE / "media-private.json").read_text(encoding="utf-8"))
    proof = json.loads((ORACLE_SITE / "success-private-proof.json").read_text(encoding="utf-8"))
    results = {}
    for line in Path(proof["v9"]["path"]).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        results[row["key"]] = row
    runs = {}
    for ep in catalog["episodes"]:
        key = f"{ep['task']}_{ep['tier']}_{ep['seed']}"
        if key not in keys:
            continue
        require(results[key]["status"] == ep["status"], f"Oracle 视频终态与权威结果不同：{key}")
        run = {"attempt": ep["id"], "status": ep["status"], "exec_steps": ep["exec_steps"],
               "demo_frames": ep["demo_frames"], "frames": ep["frames"], "task_goal": ep["task_goal"],
               "spec_sha256": results[key].get("spec_sha256")}
        for kind in ("official", "original"):
            path = Path(private[ep[f"{kind}_media"]])
            require(path.is_file(), f"Oracle 视频缺失：{key}")
            run[f"{kind}_media"] = media_id(kind, f"groundsg-oracle/{ep['id']}")
            media[run[f"{kind}_media"]] = str(path)
        runs[key] = run
    require(set(runs) == keys, f"Oracle 未覆盖全部局：缺 {len(keys - set(runs))}")
    return runs


def gt_runs(keys: set, media: dict) -> dict:
    """V9 生成真值：取 V9 站目录中 xhard1～5 的生成视频（gen.new），按任务／难度／种子对齐。"""
    catalog = json.loads((GT_SITE / "catalog.json").read_text(encoding="utf-8"))
    private = json.loads((GT_SITE / "media-private.json").read_text(encoding="utf-8"))
    runs = {}
    for task in catalog["tasks"]:
        for tier, block in task["tiers"].items():
            for ep in block["episodes"]:
                key = f"{task['id']}_{tier}_{ep['seed']}"
                if key not in keys:
                    continue
                gen = ep["gen"]["new"]
                path = Path(private[gen["media"]])
                require(path.is_file(), f"生成真值视频缺失：{key}")
                ident = media_id("gt", key)
                media[ident] = str(path)
                runs[key] = {"status": "success", "exec_steps": gen.get("exec_steps"), "frames": gen["frames"],
                             "demo_frames": gen["demo_frames"], "media": ident, "config": ep.get("config")}
    require(set(runs) == keys, f"生成真值未覆盖全部局：缺 {len(keys - set(runs))}")
    return runs


def build_site(root: Path, site_dir: Path, copy_root: Path) -> dict:
    root = root.resolve(strict=True)
    media, per_model, models = {}, {}, []
    for model, name, publish_rel, bases in MODELS:
        per_model[model] = model_runs(root, model, bases, root / publish_rel, copy_root, media)
        models.append({"id": model, "name": name, "note": "GL · 1800 步 · seed 7"})
    keys = set(per_model[MODELS[0][0]])
    require(all(set(runs) == keys for runs in per_model.values()), "五模型局集合不一致")
    per_model["groundsg-oracle"] = oracle_runs(keys, media)
    models.append({"id": "groundsg-oracle", "name": "GroundSG+Oracle",
                   "note": "1004 轮本机 · 1600 步 · 条件不同，仅作参考"})
    gt = gt_runs(keys, media)
    for model in models:
        runs = per_model[model["id"]].values()
        model["episodes"], model["success"] = len(runs), sum(r["status"] == "success" for r in runs)
    episodes = []
    for key in keys:
        task, tier, seed = key.rsplit("_", 2)
        goal = per_model[MODELS[0][0]][key]["task_goal"]
        episodes.append({"key": key, "task": task, "tier": tier, "seed": int(seed), "task_goal": goal, "gt": gt[key],
                         "runs": {m["id"]: per_model[m["id"]][key] for m in models}})
    episodes.sort(key=lambda e: (TASKS.index(e["task"]), e["tier"], e["seed"]))
    counts = Counter(e["task"] for e in episodes)
    require(set(counts) == set(TASKS), "任务覆盖不完整")
    rates = [{"task": t, **{m["id"]: {"success": sum(e["runs"][m["id"]]["status"] == "success"
                                                      for e in episodes if e["task"] == t), "total": counts[t]}
                             for m in models}} for t in TASKS]
    catalog = {
        "schema": SCHEMA, "scope": "sg-eval-gl-20261006-03 · 第三阶段 OOD",
        "notice": ("五个模型为 Great Lakes 第三阶段 OOD 正式评估：1800 步 strict、模型 seed 7、43 个任务难度格 × 每格 2 局 = 每模型 86 局；"
                   "每格 n=2，成绩只看大致排序。GroundSG+Oracle 为 1004 轮本机复刻（1600 步、同 86 局身份），条件不同，仅作参考。"
                   "生成真值为 V9 数据集生成时的专家轨迹录像。"),
        "counts": {"episodes": len(episodes), "models": len(models), "media": len(media)},
        "models": models, "tasks": [{"id": t, "episodes": counts[t]} for t in TASKS],
        "tiers": sorted({e["tier"] for e in episodes}), "episodes": episodes, "success_rates": rates,
    }
    site_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("catalog.json", catalog), ("media-private.json", media)):
        tmp = site_dir / f"{name}.part"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(site_dir / name)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, help="GL 运行根（含 publish、media、reports）")
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--copy-root", type=Path, help="五模型视频复制目标（须在 --media-root 内）")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--media-root", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    try:
        if args.serve:
            sys.path.insert(0, str(HERE))
            from site_server import create_server
            with create_server(args.host, args.port, site_dir=args.site_dir, media_root=args.media_root,
                               html_path=HERE / "stage3_eval_site.html") as server:
                print(f"STAGE3_SITE_READY port={server.server_port} media={len(server.files.media)}", flush=True)
                server.serve_forever()
            return 0
        require(args.run_root is not None and args.copy_root is not None, "构建目录需要 --run-root 与 --copy-root")
        catalog = build_site(args.run_root, args.site_dir, args.copy_root)
        c = catalog["counts"]
        print(f"STAGE3_SITE=PASS models={c['models']} episodes={c['episodes']} media={c['media']} "
              + " ".join(f"{m['id']}={m['success']}/{m['episodes']}" for m in catalog["models"]), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"STAGE3_SITE=FAIL reason={exc}", flush=True)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
