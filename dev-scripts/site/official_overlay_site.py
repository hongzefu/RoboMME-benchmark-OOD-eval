#!/usr/bin/env python3
"""构建本机 Oracle 官方版式视频目录，或复用白名单服务器托管该目录。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # dev-scripts/site → 仓根
TASKS = ("BinFill", "PickXtimes", "SwingXtimes", "PickHighlight", "VideoUnmask", "ButtonUnmask",
         "VideoUnmaskSwap", "ButtonUnmaskSwap", "VideoRepick", "PatternLock", "RouteStick",
         "VideoPlaceButton", "VideoPlaceOrder", "MoveCube", "InsertPeg", "StopCube")
SCHEMA = "official-overlay-site/1"
FINAL = ("success", "fail", "timeout")


def require(condition: bool, message: str) -> None:
    """未知或不完整的证据直接报错，不将缺项默认为通过。"""
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_inside(path: Path, root: Path) -> Path:
    """保持与服务器相同的实体目录边界，不接受符号链接媒体。"""
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), f"文件超出媒体根：{path.name}")
    require(all(not part.is_symlink() for part in (path, *path.parents) if part != root.parent),
            f"文件经过符号链接：{path.name}")
    return resolved


def reject_symlinks(path: Path) -> None:
    require(not any(component.is_symlink() for component in (path, *path.parents)), "媒体或站点目录经过符号链接")


def read_trace(path: Path) -> tuple[dict, dict, dict]:
    """核对真实序列化轨迹的首尾、唯一演示、连续步号与执行计数。"""
    header = demo = end = None
    count = 0
    ended = False
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            kind = row.get("kind")
            require(not ended, "end 后仍有轨迹记录")
            if number == 1:
                require(kind == "header" and row.get("schema") == "sgeval-trace/1", "轨迹首行或版本不符")
            if kind == "header":
                require(header is None, "轨迹重复 header")
                header = row
            elif kind == "demo":
                require(demo is None and count == 0, "轨迹重复 demo 或顺序不符")
                demo = row
            elif kind == "step":
                require(demo is not None and row.get("step") == count + 1, "轨迹 step 顺序不符")
                count += 1
            elif kind == "end":
                end, ended = row, True
            else:
                require(kind in ("request", "response", "history"), "轨迹出现未知记录")
    require(header is not None and demo is not None and end is not None, "轨迹缺少首尾或演示记录")
    require(type(end.get("exec_steps")) is int and end["exec_steps"] == count, "轨迹执行步数不符")
    require(type(demo.get("frames")) is int and demo["frames"] >= 0, "轨迹演示帧数不符")
    require(type(end.get("demo_frames")) is int and end["demo_frames"] + 1 == demo["frames"],
            "轨迹演示加初始帧数不符")
    require(end.get("status") in FINAL, "轨迹终态未完成")
    return header, demo, end


def check_fingerprint(record: dict, path: Path, *, full: bool) -> None:
    require(isinstance(record, dict), f"缺少输入指纹：{path.name}")
    require(record.get("size") == path.stat().st_size, f"输入字节数发生变化：{path.name}")
    require(isinstance(record.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
            f"输入 sha256 缺失：{path.name}")
    if full:
        require(record["sha256"] == sha256(path), f"输入 sha256 发生变化：{path.name}")


def episode_entry(directory: Path, root: Path) -> tuple[dict, dict]:
    trace = regular_inside(directory / "trace.jsonl", root)
    original = regular_inside(directory / "episode.mp4", root)
    sidecar_path = regular_inside(directory / "official/render.json", root)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    require(sidecar.get("schema") == "official-rerender/1" and sidecar.get("render_status") in ("rendered", "reused"),
            "重绘 sidecar 未通过")
    header, demo, end = read_trace(trace)
    identity = header.get("identity", {})
    task, tier, seed = identity.get("task"), identity.get("tier"), identity.get("seed")
    require(task in TASKS and tier in tuple(f"xhard{i}" for i in range(1, 6)) and type(seed) is int,
            "任务、难度或种子不符")
    require(identity.get("dataset") == "ood" and header.get("route") == "groundsg/ground-sg-oracle/new",
            "不是本机第三档 Oracle 新侧轨迹")
    require(sidecar.get("identity") == identity, "重绘与轨迹身份不同")
    require(sidecar.get("route") == header["route"], "重绘路线与轨迹不同")
    key = f"{task}_{tier}_{seed}"
    require(identity.get("key") == key and re.fullmatch(re.escape(key) + r"\.a\d+", directory.name) is not None,
            "目录与轨迹身份不同")
    official_files = list((directory / "official").glob("*.mp4"))
    require(len(official_files) == 1, "官方子目录必须恰好有一个 mp4")
    official = regular_inside(official_files[0], root)
    require(official.name.startswith("official-rerender__") and original.stat().st_size > 0 and official.stat().st_size > 0,
            "媒体为空或重绘命名不符")
    require(sidecar.get("out") == str(official), "sidecar 输出路径不符")
    require(sidecar.get("terminal_reason") == end["terminal_reason"] and sidecar.get("status") == end["status"],
            "重绘终态与轨迹不同")
    require(end["terminal_reason"] == end["status"] or (end["terminal_reason"], end["status"]) == ("error", "timeout"),
            "轨迹成功字段与终态字段冲突")
    require(str(sidecar.get("fps")) == "30", "重绘帧率不是 30 fps")
    for name, expected in (("demo_frames", end["demo_frames"]), ("exec_steps", end["exec_steps"])):
        require(sidecar.get(name) == expected, f"重绘 {name} 与轨迹不同")
    recorded, omitted = sidecar.get("recorded_exec_steps"), sidecar.get("omitted_timeout_frames")
    require(type(recorded) is int and type(omitted) is int and omitted in (0, 1), "重绘执行计数缺失")
    require(recorded + omitted == end["exec_steps"] and (not omitted or end["status"] == "timeout"),
            "重绘超时末帧口径不符")
    require(sidecar.get("source_frames") == demo["frames"] + end["exec_steps"] and
            sidecar.get("frames") == demo["frames"] + recorded, "重绘总帧数不符")
    require(sidecar.get("width") == 512 and type(sidecar.get("height")) is int and sidecar["height"] >= 256,
            "重绘画面尺寸不符")
    check_fingerprint(sidecar.get("source_trace"), trace, full=True)
    check_fingerprint(sidecar.get("source_mp4"), original, full=False)
    check_fingerprint(sidecar.get("output_fingerprint"), official, full=True)
    arrays = regular_inside(directory / "arrays.npz", root)
    check_fingerprint(sidecar.get("source_arrays"), arrays, full=True)
    require(isinstance(sidecar.get("task_goal"), str) and sidecar["task_goal"], "任务目标缺失")
    require(isinstance(demo.get("texts"), list) and demo["texts"] and sidecar["task_goal"] == demo["texts"][0],
            "重绘任务目标与轨迹不同")
    require(isinstance(sidecar.get("episode_id"), (str, int)), "官方局号缺失")
    media = {}
    media_suffix = hashlib.sha256(directory.name.encode("utf-8")).hexdigest()[:32]
    for kind, path in (("official", official), ("original", original)):
        media[f"{kind}_{media_suffix}"] = str(path)
    entry = {
        "id": directory.name, "task": task, "task_name": task, "tier": tier, "seed": seed,
        "episode_id": sidecar["episode_id"], "status": end["status"], "task_goal": sidecar["task_goal"],
        "demo_frames": end["demo_frames"], "exec_steps": end["exec_steps"], "recorded_exec_steps": recorded,
        "omitted_timeout_frames": omitted, "frames": sidecar["frames"], "fps": 30,
        "width": sidecar["width"], "height": sidecar["height"], "duration": round(sidecar["frames"] / 30, 3),
        "official_media": f"official_{media_suffix}", "original_media": f"original_{media_suffix}",
    }
    return entry, media


def read_results(path: Path, *, baseline: bool) -> tuple[dict, dict]:
    """只接受已合并的逐身份终态，不从多次尝试中自行挑选成功回合。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    wanted, per_task = (192, 12) if baseline else (800, 50)
    require(len(rows) == wanted, f"权威结果局数不符：{path.name}")
    records = {}
    for row in rows:
        task, tier, seed = row.get("task"), row.get("tier"), row.get("seed")
        require(task in TASKS and type(seed) is int, "权威结果身份缺失")
        require(tier == "xhard0" if baseline else tier in tuple(f"xhard{i}" for i in range(1, 6)),
                "权威结果难度不符")
        require(row.get("dataset") == ("hard-verify" if baseline else "ood"), "权威结果数据集不符")
        require(row.get("policy") == "groundsg" and row.get("policy_variant") == "ground-sg-oracle" and
                row.get("side") == "new" and row.get("host") == "sled-vail", "不是同本机 Oracle 新侧结果")
        require(row.get("infra") is False and row.get("status") in FINAL and
                type(row.get("task_success")) is bool and row["task_success"] == (row["status"] == "success"),
                "权威结果含基础设施失败或成功字段冲突")
        key = f"{task}_{tier}_{seed}"
        require(row.get("key") == key and key not in records, "权威结果 key 不符或重复")
        records[key] = row
    counts = Counter(row["task"] for row in records.values())
    require(set(counts) == set(TASKS) and set(counts.values()) == {per_task}, "权威结果任务覆盖不完整")
    rates = {task: {"success": sum(row["task_success"] for row in records.values() if row["task"] == task),
                    "total": counts[task]} for task in TASKS}
    return records, rates


def build_site(root: Path, site_dir: Path, expect_episodes: int, xhard0_results: Path, v9_results: Path) -> dict:
    """先逐局验全再写目录，任何缺项均不发布部分目录。"""
    reject_symlinks(root)
    reject_symlinks(site_dir)
    root = root.resolve(strict=True)
    require(root.is_dir(), "媒体根必须为实体目录")
    directories = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    require(len(directories) == expect_episodes, f"局目录数 {len(directories)} 与期望 {expect_episodes} 不同")
    entries, media, identities = [], {}, set()
    for directory in directories:
        try:
            entry, mapping = episode_entry(directory, root)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise ValueError(f"{directory.name}：{exc}") from exc
        identity = (entry["task"], entry["tier"], entry["seed"])
        require(identity not in identities, f"重复评估身份：{directory.name}")
        identities.add(identity)
        entries.append(entry)
        require(not set(media).intersection(mapping), "媒体 ID 重复")
        media.update(mapping)
    counts = Counter(row["task"] for row in entries)
    require(expect_episodes == 800 and set(counts) == set(TASKS) and set(counts.values()) == {50},
            "完整目录必须是 16 任务 × 50 局 = 800 局")
    entries.sort(key=lambda row: (TASKS.index(row["task"]), row["tier"], row["seed"]))
    baseline_records, baseline_rates = read_results(xhard0_results, baseline=True)
    v9_records, v9_rates = read_results(v9_results, baseline=False)
    entry_keys = {f"{row['task']}_{row['tier']}_{row['seed']}" for row in entries}
    require(entry_keys == set(v9_records), "视频身份全集与 V9 权威结果不同")
    for row in entries:
        result = v9_records[f"{row['task']}_{row['tier']}_{row['seed']}"]
        require(result["status"] == row["status"], "视频终态与 V9 权威结果不同")
    success_rates = [{"task": task, "v9": v9_rates[task], "xhard0": baseline_rates[task]} for task in TASKS]
    private_proof = {"schema": "official-overlay-success-proof/1",
                     "v9": {"path": str(v9_results.resolve(strict=True)), "sha256": sha256(v9_results),
                            "size": v9_results.stat().st_size, "episodes": len(v9_records)},
                     "xhard0": {"path": str(xhard0_results.resolve(strict=True)), "sha256": sha256(xhard0_results),
                                "size": xhard0_results.stat().st_size, "episodes": len(baseline_records)}}
    catalog = {
        "schema": SCHEMA, "scope": "本机 V9 第三档 GroundSG+Oracle 复刻",
        "notice": "本机 Oracle 复刻视频，供版式与行为检查；不代表 Great Lakes 正式评估成绩。",
        "counts": {"episodes": len(entries), "tasks": len(counts), "per_task": 50, "media": len(media)},
        "tasks": [{"id": task, "name": task, "episodes": counts[task]} for task in TASKS],
        "tiers": sorted({row["tier"] for row in entries}), "episodes": entries,
        "success_rates": success_rates,
        "validation": {"trace_sha256": "逐局全量核对", "arrays_sha256": "逐局全量核对",
                       "official_sha256": "逐局全量核对",
                       "source_mp4": "身份与字节数核对，未重复全量哈希"},
    }
    site_dir.mkdir(parents=True, exist_ok=True)
    require(not any((site_dir / name).exists() for name in ("catalog.json", "media-private.json", "success-private-proof.json")),
            "站点目录已有目录文件，拒绝覆盖")
    for name, data in (("catalog.json", catalog), ("media-private.json", media), ("success-private-proof.json", private_proof)):
        with (site_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="本机第三档 Oracle 媒体的 new 根目录")
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--expect-episodes", type=int, default=800)
    parser.add_argument("--xhard0-results", type=Path, help="上一版本机 Oracle 新侧的最终 192 局结果")
    parser.add_argument("--v9-results", type=Path, help="本机 V9 第三档 Oracle 最终 800 局结果")
    parser.add_argument("--serve", action="store_true", help="托管已构建的目录，不重建目录")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--media-root", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    try:
        if args.serve:
            from site_server import create_server
            with create_server(args.host, args.port, site_dir=args.site_dir, media_root=args.media_root,
                               html_path=HERE / "official_overlay.html") as server:
                print(f"OFFICIAL_SITE_READY port={server.server_port} media={len(server.files.media)}", flush=True)
                server.serve_forever()
            return 0
        require(args.root is not None, "构建目录需要 --root")
        require(args.xhard0_results is not None and args.v9_results is not None,
                "构建成功率表需要 --xhard0-results 与 --v9-results")
        catalog = build_site(args.root, args.site_dir, args.expect_episodes, args.xhard0_results, args.v9_results)
        print(f"OFFICIAL_SITE=PASS episodes={catalog['counts']['episodes']} tasks=16 per_task=50 media=1600", flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"OFFICIAL_SITE=FAIL reason={exc}", flush=True)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
