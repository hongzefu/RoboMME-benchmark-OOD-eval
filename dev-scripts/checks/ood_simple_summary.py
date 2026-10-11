"""五模型 OOD 简化续跑的汇总（只读）：覆盖核对、成功率三级表、infra 单列、PP 上下文溢出重分类、
疑似 IO 误判清单，可选原始录像完整解码校验与 NFS 大文件删除清单（1011 最简四卡方案，计划第二部分 §一）。

用法（本机，产物已 rsync 回 /data）：
  uv run --no-sync python dev-scripts/checks/ood_simple_summary.py \
      --rollouts artifacts/ood-five-raw-seed7-20261010-02/gl/rollouts \
      --identities artifacts/ood-five-raw-seed7-20261010-02/control/identities.jsonl \
      [--verify-media 8] [--verify-limit K] [--emit-delete-list <文件>] [--nfs-rollouts <NFS rollouts 根>]

口径：
  - 只读 rollouts/<label>/ood/seed7/raw/*/result.json（每局独立原子写）；results.jsonl / progress.json / log.json
    是同模型四进程共写的文件，不作数。
  - 覆盖：每个模型的 result.json key 集合须与身份清单（800 行）完全相同、spec_sha256 逐局相同。
  - 成功率分母只含 infra=false 的局；infra 局单列。
  - PP 重分类（用户 2026-10-10「采用：上下文耗尽计模型失败 0」）：label=pp、infra_reason=pp_server_error 且
    error 含「context overflow」或「exceeds cap」的局，重分类后计入分母、成功记 0；重分类前后两组数字并列。
  - io_suspect：infra=false 但 error／env_exception／exception 含 OSError、Errno、Input/output error、
    Stale file handle、No space、Remote I/O 等字样的局，单列交用户（可能是被记成模型失败的基础设施故障）。
  - --verify-media N：对 infra=false 的局以 N 路并行调用 dev-scripts/gl/eval_video_mover.py::raw_verify_media；
    infra 局不进 RAW_VERIFY 分母（缺 frames-front.jsonl 必抛异常），以 infra_skipped 单列。
  - --emit-delete-list：只写出本次 RAW_VERIFY 通过局的 front.mkv／wrist.mkv／arrays.npz 的 NFS 绝对路径；
    必须同时给 --verify-media，清单文件不得落在 rollouts 树内。
判定行：COVERAGE=PASS|FAIL、SUMMARY=PASS|FAIL、RAW_VERIFY=PASS|FAIL（仅 --verify-media 时）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MOVER = REPO / "dev-scripts/gl/eval_video_mover.py"
MODELS = ("groundsg-ground-sg-oracle", "perceptual-framesamp-modul", "pp", "smvla", "groundsg-ground-sg-qwenvl")
FIELDS = ("key", "task", "tier", "spec_sha256", "status", "task_success", "infra", "infra_reason",
          "error", "env_exception", "exception", "t_start", "t_end")
# 真实样例（docs/validation/full-eval-20261010/records/pp-context-overflow/）：
#   RuntimeError: Server error: {... 'Ponder context overflow at step 1661: S2 context length 16439 exceeds cap 16384'}
PP_CTX = re.compile(r"context overflow|exceeds cap")
IO_PAT = re.compile(r"OSError|Errno|Input/output error|Stale file handle|No space|Remote I/O|Broken pipe")
MEDIA = ("front.mkv", "wrist.mkv", "arrays.npz")
# 本机副本与 NFS 副本只差仓库根前缀（产物树相对仓库根的路径相同）
LOCAL_REPO = "/data/hongzefu/RoboMME-benchmark-OOD-eval"
NFS_REPO = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval"


def load(rollouts: Path, labels):
    """读全部 result.json；解析失败的局记入 bad 列表（SUMMARY 判 FAIL），不中断。"""
    rows, bad = [], []
    for label in labels:
        for f in sorted((rollouts / label / "ood" / "seed7" / "raw").glob("*/result.json")):
            try:
                d = json.loads(f.read_text())
            except Exception as e:  # noqa: BLE001 —— 半写或损坏的 result.json 只记录，不中断汇总
                bad.append((str(f), f"{type(e).__name__}: {e}"))
                continue
            r = {k: d.get(k) for k in FIELDS}
            r["label"], r["dir"] = label, str(f.parent)
            rows.append(r)
    return rows, bad


def rate_table(rows, keyfn, head, title, reclass=False):
    """按 keyfn 分组输出 Markdown 表：成功 / 非 infra 分母 / 成功率 / infra 数。reclass=True 时 pp_ctx 局计模型失败。"""
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        a = agg[keyfn(r)]
        as_model_fail = reclass and r["pp_ctx"]
        if r["infra"] and not as_model_fail:
            a[2] += 1
        else:
            a[1] += 1
            a[0] += int(r["task_success"] == 1 and not as_model_fail)
    print(f"\n### {title}\n\n| {' | '.join(head)} | 成功 | 非infra局 | 成功率 | infra |\n|{'---|' * (len(head) + 4)}")
    for k in sorted(agg):
        s, n, i = agg[k]
        print(f"| {' | '.join(map(str, k))} | {s} | {n} | {(s / n * 100 if n else 0):.1f}% | {i} |")
    return agg


def round_robin(rows):
    """按 (label, status) 分组轮转排序，使 --verify-limit 抽样的前 K 局覆盖不同模型与状态。"""
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["label"], str(r["status"]), str(r["key"]))):
        groups[(r["label"], str(r["status"]))].append(r)
    out, qs = [], [groups[k] for k in sorted(groups)]
    while any(qs):
        for q in qs:
            if q:
                out.append(q.pop(0))
    return out


def verify_one(mover, r):
    try:
        mover.raw_verify_media(Path(r["dir"]))
        ok, err = 1, ""
    except Exception as e:  # noqa: BLE001 —— 任何异常都记为该局校验失败，逐局列出
        ok, err = 0, f"{type(e).__name__}: {str(e)[:160]}".replace("\n", " ")
    print(f"RAW_VERIFY_ITEM dir={r['dir']} ok={ok} err={err}", flush=True)
    return r, ok


def main():
    ap = argparse.ArgumentParser(description="五模型 OOD 简化续跑汇总（只读）")
    ap.add_argument("--rollouts", type=Path, required=True, help="gl/rollouts 根（本机副本）")
    ap.add_argument("--identities", type=Path, required=True, help="control/identities.jsonl（每模型 800 行）")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--verify-media", type=int, default=0, help="原始录像完整解码校验的并行度；0=不校验")
    ap.add_argument("--verify-limit", type=int, default=0, help="只校验轮转抽样的前 K 局（0=全部）；用于冒烟")
    ap.add_argument("--emit-delete-list", type=Path, help="写出已校验局三个大文件的 NFS 路径清单")
    ap.add_argument("--nfs-rollouts", help=f"NFS 侧 rollouts 根；缺省按 {LOCAL_REPO} → {NFS_REPO} 前缀替换推出")
    a = ap.parse_args()
    labels = [m for m in a.models.split(",") if m]
    rollouts = a.rollouts.resolve()

    ident = {}
    for line in a.identities.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            ident[d["key"]] = d["spec_sha256"]
    rows, bad = load(rollouts, labels)
    for r in rows:
        text = " ".join(str(r[k]) for k in ("error", "env_exception", "exception") if r[k])
        r["pp_ctx"] = (r["label"] == "pp" and bool(r["infra"]) and r["infra_reason"] == "pp_server_error"
                       and bool(PP_CTX.search(str(r["error"] or ""))))
        r["io"] = (not r["infra"]) and bool(IO_PAT.search(text))

    # 一、覆盖核对
    print("## 覆盖核对\n\n| 模型 | 期望 | found | missing | extra | sha_mismatch | dup |\n|---|---|---|---|---|---|---|")
    tot = defaultdict(int)
    mism = []
    for label in labels:
        keys = [r["key"] for r in rows if r["label"] == label]
        found = set(keys)
        miss, extra = len(set(ident) - found), len(found - set(ident))
        badsha = [r for r in rows if r["label"] == label and r["key"] in ident and r["spec_sha256"] != ident[r["key"]]]
        mism += badsha
        dup = len(keys) - len(found)
        print(f"| {label} | {len(ident)} | {len(found)} | {miss} | {extra} | {len(badsha)} | {dup} |")
        for k, v in (("expected", len(ident)), ("found", len(found)), ("missing", miss), ("extra", extra),
                     ("sha", len(badsha)), ("dup", dup)):
            tot[k] += v
    for r in mism[:5]:
        print(f"SHA_MISMATCH label={r['label']} key={r['key']} result={r['spec_sha256']} identities={ident[r['key']]}")
    for path, err in bad[:20]:
        print(f"RESULT_UNREADABLE path={path} err={err[:160]}")

    # 二、成功率
    rate_table(rows, lambda r: (r["label"], r["task"], r["tier"]), ["模型", "任务", "档"], "按 (模型, 任务, 档)")
    rate_table(rows, lambda r: (r["label"], r["task"]), ["模型", "任务"], "按 (模型, 任务)")
    rate_table(rows, lambda r: (r["label"],), ["模型"], "按模型")
    pp = [r for r in rows if r["label"] == "pp"]
    rate_table(pp, lambda r: ("pp（重分类前）",), ["口径"], "PP 重分类前（上下文溢出计 infra）")
    rate_table(pp, lambda r: ("pp（重分类后）",), ["口径"], "PP 重分类后（上下文溢出计模型失败 0）", reclass=True)

    # 三、infra 与疑似 IO 清单
    infra = [r for r in rows if r["infra"]]
    print(f"\n### infra 局（{len(infra)}）")
    for r in infra:
        print(f"INFRA label={r['label']} key={r['key']} reason={r['infra_reason']} pp_ctx={int(r['pp_ctx'])} "
              f"err={str(r['error'])[:120]}")
    io = [r for r in rows if r["io"]]
    print(f"\n### io_suspect（{len(io)}，infra=false 但错误文本像基础设施故障，交用户裁决）")
    for r in io:
        print(f"IO_SUSPECT label={r['label']} key={r['key']} status={r['status']} err={str(r['error'])[:120]} "
              f"env_exc={str(r['env_exception'])[:80]}")

    cov_ok = all(tot[k] == 0 for k in ("missing", "extra", "sha", "dup")) and not bad
    print(f"\nCOVERAGE={'PASS' if cov_ok else 'FAIL'} expected={tot['expected']} found={tot['found']} "
          f"missing={tot['missing']} extra={tot['extra']} sha_mismatch={tot['sha']} dup={tot['dup']} "
          f"unreadable={len(bad)}")
    cells = len({(r["label"], r["task"], r["tier"]) for r in rows})
    print(f"SUMMARY={'FAIL' if bad else 'PASS'} models={len({r['label'] for r in rows})} cells={cells} "
          f"infra={len(infra)} pp_ctx_reclass={sum(r['pp_ctx'] for r in rows)} io_suspect={len(io)} "
          f"unreadable={len(bad)}")

    # 四、原始录像校验（infra 局不进分母）
    passed = []
    if a.verify_media:
        spec = importlib.util.spec_from_file_location("eval_video_mover", MOVER)
        mover = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mover)
        cand = round_robin([r for r in rows if not r["infra"]])
        if a.verify_limit:
            cand = cand[:a.verify_limit]
        with ThreadPoolExecutor(a.verify_media) as ex:
            res = list(ex.map(lambda r: verify_one(mover, r), cand))
        passed = [r for r, ok in res if ok]
        failed = len(res) - len(passed)
        print(f"RAW_VERIFY={'PASS' if failed == 0 and res else 'FAIL'} verified={len(passed)} failed={failed} "
              f"infra_skipped={len(infra)} limit={a.verify_limit or 'all'}")

    # 五、NFS 大文件删除清单（只含本次校验通过的局；执行删除另须用户确认）
    if a.emit_delete_list:
        if not a.verify_media:
            sys.exit("--emit-delete-list 只能与 --verify-media 同用：未校验的局不得进删除清单")
        out = a.emit_delete_list.resolve()
        if rollouts == out or rollouts in out.parents:
            sys.exit("--emit-delete-list 不得写进 rollouts 树（本脚本对产物树只读）")
        src = str(rollouts)
        nfs = a.nfs_rollouts
        if not nfs:
            if not src.startswith(LOCAL_REPO + "/"):
                sys.exit(f"rollouts 不在 {LOCAL_REPO} 下，须显式给 --nfs-rollouts")
            nfs = NFS_REPO + src[len(LOCAL_REPO):]
        nfs = nfs.rstrip("/")
        lines = [f"{nfs}{r['dir'][len(src):]}/{m}" for r in passed for m in MEDIA]
        out.write_text("".join(f"{x}\n" for x in lines))
        print(f"DELETE_LIST path={out} lines={len(lines)} episodes={len(passed)} nfs_root={nfs}")


if __name__ == "__main__":
    main()
