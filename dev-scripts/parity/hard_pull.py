#!/usr/bin/env python3
"""阶段 4 逐局拉取（sled-vail 上跑，0927 计划第一部分 §5.4「流转」）：NFS 暂存 → ``/data`` → 核 sha → 删 NFS 副本。

GL 节点上的 ``hard_parity.py generate`` 每局搬到 ``<stage>/<段>/episodes/…`` 后写 ``SHIPPED``（大文件 sha 清单）；
本脚本只拉带 ``SHIPPED`` 的局，拉回后逐文件重算 sha 与 ``SHIPPED`` 比对，一致才删 NFS 上该局的 h5／mp4
（小文件留着随段末 rsync）。段目录出现 ``SEGMENT_DONE`` 后拉回小文件（identities.jsonl 等），并核 identities 里
每局 h5 都在 ``/data`` 且 sha 相等，打 ``PULL_SEGMENT=PASS|FAIL``；全部段完成后退出并打 ``PULL_DONE``。
NFS 上只存在途的局；段结束确认目录下无大文件后 ``rmdir`` 空目录由 runbook 手工执行（显式逐目录）。

    uv run --no-sync python dev-scripts/parity/hard_pull.py --stage <NFS>/hs-stage --dest artifacts/newtask-v6/hard-split/h5 \
        --segments smoke-O-native,O-native,H-xhard,smoke-P-native,P-native,smoke-H-native,H-native

``--identities <jsonl>``（对拍细则 3.4「只回传翻转局」，如 ``gen-regress check --rerun-identities-out`` 写的重跑清单
或 ``generate --expect-ref`` 写的 ``flips.jsonl``；每行 ``{task, seed, ...}``，按 (task, seed) 匹配，档位名不参与）：
等段目录出现 ``SEGMENT_DONE`` 后读暂存里的 ``identities.jsonl``，只拉清单内且带 ``SHIPPED`` 的局；段末核对只覆盖
清单内的局（h5 在本机且 sha 相等、NFS 上这些局不残留大文件），清单外仍留在暂存的已搬运局只计数
（``not_selected``）、不拉不删。各遍顶层报告文件（``identities.jsonl``、``results.jsonl``、``summary.json``、
``launch-*.json`` 等）照常拉回。清单内某局在该段 identities 里没有、或在但没有 ``SHIPPED``（如 ``verdict=match``
按设计未复制）时只计数 ``absent``／``not_shipped``，不判失败——同一份清单会跨首跑与重跑多个段使用。
不传 ``--identities`` 时行为不变。

``generate --expect-ref`` 写出的段（identities 行带 ``verdict``）：``verdict == "match"`` 的局按设计不复制（节点上已删），
段末核对不要求其文件，计入 ``matched=``（判定行尾追加，只在段内有带 verdict 的行时出现）；其余行照旧要求 h5 在本机且
sha 相等。``--identities`` 允许空清单（零翻转时用）：不拉任何局，只拉段顶层小文件与报告并做段末核对。
行里没有 ``verdict`` 的旧格式段，行为与原先完全相同。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pull_episode(src: Path, dst: Path) -> tuple[bool, str]:
    shipped = json.loads((src / "SHIPPED").read_text())
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for rel, sha in shipped.items():
        if sha256_file(dst / rel) != sha:
            return False, f"sha 不符：{dst / rel}"
    for rel in shipped:
        (src / rel).unlink()
    (src / "PULLED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    return True, "ok"


def read_identity_keys(path: Path) -> set[tuple[str, int]]:
    """``--identities`` 清单：jsonl 每行 ``{task, seed, ...}``，返回 (task, seed) 集合。

    空清单合法（``--expect-ref`` 零翻转时）：不拉任何局，只拉段顶层小文件与报告。"""
    return {(str(r["task"]), int(r["seed"])) for r in
            (json.loads(t) for t in path.read_text(encoding="utf-8").splitlines() if t.strip())}


def selected_episodes(stage: Path, keys: set[tuple[str, int]]) -> tuple[list[Path], dict[str, int]]:
    """按暂存 identities.jsonl 把清单身份映射到局目录（``<局目录>/hdf5_files/<名>.h5`` 的上两级）。"""
    lines = [json.loads(t) for t in (stage / "identities.jsonl").read_text(encoding="utf-8").splitlines() if t.strip()]
    chosen, seen = [], set()
    stats = {"selected": 0, "absent": 0, "not_shipped": 0, "not_selected": 0}
    for line in lines:
        key = (str(line["task"]), int(line["seed"]))
        rel = line.get("path")
        ep = stage / Path(rel).parent.parent if rel else None
        if key in keys:
            seen.add(key)
            if ep is not None and (ep / "SHIPPED").is_file():
                chosen.append(ep)
                stats["selected"] += 1
            else:
                stats["not_shipped"] += 1
        elif ep is not None and (ep / "SHIPPED").is_file() and not (ep / "PULLED").exists():
            stats["not_selected"] += 1
    stats["absent"] = len(keys - seen)
    return chosen, stats


def finish_segment(stage: Path, dest: Path, only: list[Path] | None = None, stats: dict[str, int] | None = None) -> bool:
    for path in sorted(p for p in stage.iterdir() if p.is_file()):
        shutil.copy2(path, dest / path.name)
    for sub in ("_runner", "_rounds"):
        if (stage / sub).is_dir():
            shutil.copytree(stage / sub, dest / sub, dirs_exist_ok=True)
    lines = [json.loads(t) for t in (dest / "identities.jsonl").read_text().splitlines() if t.strip()] \
        if (dest / "identities.jsonl").exists() else []
    bad = 0
    # generate --expect-ref 的 match 局按设计没有复制到暂存：不要求文件，单独计数
    has_verdict = any("verdict" in line for line in lines)
    matched = sum(1 for line in lines if line.get("verdict") == "match")
    if only is None:
        for line in lines:
            if line.get("path") and line.get("verdict") != "match":
                local = dest / line["path"]
                bad += int(not local.is_file() or sha256_file(local) != line["sha256"])
        leftovers = [p for p in stage.rglob("*") if p.is_file() and p.suffix in (".h5", ".mp4")]
    else:
        # --identities：只核清单内且已搬运的局
        chosen = {str(ep.relative_to(stage)) for ep in only}
        for line in lines:
            rel = line.get("path")
            if rel and line.get("verdict") != "match" and str(Path(rel).parent.parent) in chosen:
                local = dest / rel
                bad += int(not local.is_file() or sha256_file(local) != line["sha256"])
        leftovers = [p for ep in only for p in ep.rglob("*") if p.is_file() and p.suffix in (".h5", ".mp4")]
    ok = bad == 0 and not leftovers and bool(lines)
    extra = "" if stats is None else " " + " ".join(f"{k}={v}" for k, v in stats.items())
    if has_verdict:
        extra += f" matched={matched}"
    print(f"PULL_SEGMENT={'PASS' if ok else 'FAIL'} segment={stage.name} identities={len(lines)} "
          f"h5={sum(1 for l in lines if l.get('path'))} sha_bad={bad} nfs_leftover_media={len(leftovers)}{extra}",
          flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--identities", type=Path, default=None,
                        help="只拉清单内的局（jsonl，每行 {task, seed}；按 (task, seed) 匹配；可为空清单＝只拉报告）；"
                             "不传时行为不变")
    args = parser.parse_args()
    segments = [s for s in args.segments.split(",") if s]
    keys = read_identity_keys(args.identities) if args.identities is not None else None
    done: dict[str, bool] = {}
    while len(done) < len(segments):
        for seg in segments:
            if seg in done:
                continue
            stage, dest = args.stage / seg, args.dest / seg
            if not stage.is_dir():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            if keys is not None:
                # 只拉清单内的局：等段末小文件（含 identities.jsonl）落到暂存后再按身份映射局目录
                if not (stage / "SEGMENT_DONE").exists():
                    continue
                chosen, stats = selected_episodes(stage, keys)
                for src in chosen:
                    if (src / "PULLED").exists():
                        continue
                    ok, note = pull_episode(src, dest / src.relative_to(stage))
                    print(f"PULLED {seg}/{src.relative_to(stage)} ok={int(ok)} {'' if ok else note}", flush=True)
                done[seg] = finish_segment(stage, dest, only=chosen, stats=stats)
                continue
            for marker in sorted(stage.glob("episodes/**/SHIPPED")):
                src = marker.parent
                if (src / "PULLED").exists():
                    continue
                ok, note = pull_episode(src, dest / src.relative_to(stage))
                print(f"PULLED {seg}/{src.relative_to(stage)} ok={int(ok)} {'' if ok else note}", flush=True)
            if (stage / "SEGMENT_DONE").exists() and not list(p for p in stage.glob("episodes/**/SHIPPED")
                                                            if not (p.parent / "PULLED").exists()):
                done[seg] = finish_segment(stage, dest)
        time.sleep(args.interval)
    ok = all(done.values())
    print(f"PULL_DONE={'PASS' if ok else 'FAIL'} segments={len(done)} failed={[s for s, v in done.items() if not v]}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
