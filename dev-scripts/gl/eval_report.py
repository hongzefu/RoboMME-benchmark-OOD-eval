"""V8 双模型评估汇总（1001-v8-post-evaluation-gl-plan.md 第一部分 §1 第 5、7 条、§4；契约 C4 第一段）。纯 CPU、只用标准库。

    python dev-scripts/gl/eval_report.py --manifest <manifest.json> --stage <运行根> \
        --policies smvla,perceptual-framesamp-modul --out <dir> [--videos <本机视频根>] [--partial] [--expect-total 1070] [--cap 1600]

输入（契约 C1～C3）：
- ``--manifest``：``eval_manifest.py`` 产出的 ``manifest.json``（``rows`` 为执行身份行，``key = f"{task}_{tier}_{seed}"``）；
  分片归属取行内 ``shard`` 字段，没有就读同目录 ``shard-NN.json``。
- ``--stage``：运行根，逐席 ``sNN/<policy>/results.jsonl`` 与账本 ``sNN/<policy>/<policy>.ledger.jsonl``
  （也兼容 ``sNN/<policy>.ledger.jsonl``）；录像目录 ``sNN/<policy>/rec/<key>.a<n>/``。
- ``--videos``（可选）：``eval_video_mover.py --mode v8`` 的本机目的根 ``<videos>/<policy>/<tier>/<task>/<key>.a<n>/``，
  已搬走的录像在这里找。

口径：
- 每身份唯一权威终态 = 账本 ``accept`` 行的 ``accepted_attempt_id`` 对应的结果行；不是「最后一条」。
- 合法权威结局：success／fail／timeout，或「非 infra 错误终局」（status=error、infra=false、非 budget_exhausted、
  非 run_blocked；记 error 列、占分母、不算成功，判定行追加 error_final=）。
- 迟到终态（同一身份 accept 之后再来的终态）与废弃尝试（未被接受的 error／infra 尝试；账本有 attempt_end 而
  results.jsonl 无对应行的悬空尝试也算）单列，不入分数。金丝雀行（canary）不进终态分母。
- 同一身份两条不同 ``accepted_attempt_id``、accept 指向 infra 错误或其他非终态行、或同一 attempt_id 的结果行自相矛盾
  → conflicting_terminal。
- 分母固定为 manifest 身份数（默认核对 1070）／模型；没有 accept 的身份一律记 missing（不进成功、照样占分母）。
- ``exec_steps`` 大于 ``--cap``（1600）的结果行计入 exec_over_cap，>0 即 FAIL。

产出 ``<out>/report.json``、``<out>/report.md``（中文）、``<out>/video-index.jsonl``；末两行判定：
``V8_EVAL_COVERAGE=PASS|FAIL policies= missing= extra= duplicate= conflicting_terminal= late_ignored= error_final=``（error_final 为契约字段后的追加字段：非 infra 错误终局数）
``V8_EVAL_REPORT=PASS|FAIL count_mismatch= media_unexplained= exec_over_cap=``
``--partial``：中途进度，不因缺失／媒体未就位判 FAIL（冲突、越限照判），另报已完成数、各格已完成与按已完成局墙钟估算的各席剩余时间。
退出码：两行都 PASS 为 0，否则 1。

V9 合并复用（1002-newtask-v9-movecube-region-800-plan.md 第二部分 §2.1、§2.2 第 7 条、§2.4.2 第 7 步）：

    python dev-scripts/gl/eval_report.py --manifest <V9 run>/manifest/manifest.json --stage <V9 运行根> \
        --videos <V9 本机视频根> --out <dir> \
        --reuse <V8 结果目录> --reuse-manifest <V8 manifest.json>

- ``--manifest`` 是 ``eval_manifest.py --exclude-evaluated`` 的新评清单（默认核对 80 行）；复用集合**只认**同目录
  ``reused.json``（sha256 须等于 manifest ``reused.sha256``，``v8_manifest_sha256`` 须等于 ``--reuse-manifest`` 现算值）。
- ``--reuse``：V8 运行根（含 ``sNN/<policy>/``）；给的是 V8 结果目录（如
  ``artifacts/v8-evaluation/<R>``）时自动取其下 ``nfs-records/run``。只读，不重算、不改写 V8 任何文件。
- 复用行：按 ``v8_key`` 找 V8 manifest 行，四元组 (task, tier, seed, spec_sha256) 须逐条相等；终态取 V8 账本
  ``accepted_attempt_id`` 指向的结果行（同 V8 口径 ``analyze_attempts``），结果行的 spec_sha256 也须相等；任一不符、
  V8 无 accepted 终态、或与新评行四元组重叠 → count_mismatch。
- 新评 80 局照 V8 口径出分表；另出 800 局总表（按任务、按档、按格，两模型成功率），写进 ``report.json`` 的 ``v9`` 与
  ``report.md``。720 局终态是 V8 当时跑的（R-7），报告注明。
- 视频：只核新评身份（模型数 × 新评行数）：本机 ``<videos>/<policy>/<tier>/<task>/<rec 目录名>/`` 的 front.mkv、wrist.mkv
  都在且读得出帧数（解码与缓存沿用 ``eval_video_mover.py`` 的 ``pick_decoder``／``DecodeCache``），``moved.jsonl``
  里该目录的逐文件 sha256 与现算一致；``moved.jsonl`` 无该目录记录的计 moved_record_absent（不计入 videos，>0 即 FAIL）；
  非 infra 错误终局无录像且写明原因计 error_final_no_video（同 V8）。
末三行：
``V9_EVAL_COVERAGE=PASS|FAIL policies= expected= missing= extra= duplicate= conflicting_terminal= error_final=``（error_final>0 即 FAIL）
``V9_EVAL_REPORT=PASS|FAIL total= new= reused= count_mismatch= media_unexplained=``（exec_over_cap>0 时追加该字段并 FAIL）
``V9_EVAL_VIDEOS=PASS|FAIL policies= expected= videos= missing= decode_fail= sha_mismatch= moved_record_absent=``
退出码：三行都 PASS 为 0，否则 1。不传 ``--reuse`` 时一切同 V8。
``--partial`` 与 ``--reuse`` 同用时：新评分表与进度估算照 ``--partial`` 口径出，但 V9 三行一律按严格口径判定（缺失按结局
计数、未搬视频与复用对齐照常计数），且三行都追加 ``partial=1`` 并一律判 FAIL——评估中途必然 FAIL；V9 判定行只以评估结束、
视频搬完后不带 ``--partial`` 的那次为准。
report.json 键的分工（站点 S1-E 依赖）：``per_policy.<p>.cells／tasks／tiers`` 只统计本次 manifest 的新评身份（与 V8 report
同口径）；800 局总表只在 ``v9.totals.<p>``（``tasks／tiers／cells／outcomes``），不混进 ``per_policy``。

带 ``--dataset {ood,hard-verify}``（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.2）：

    python dev-scripts/gl/eval_report.py --manifest <manifest.json> --stage <运行根> --dataset hard-verify \
        --policies perceptual-framesamp-modul,groundsg:ground-sg-oracle,pp --expect-total 192 --out <dir> [--videos <本机视频根>] [--side new]

- ``--policies`` 接受任意 ``<policy>[:<variant>]``，运行根目录 ``sNN/<policy>[-<variant>]/``，结果行按 ``policy`` 与
  ``policy_variant`` 过滤；``--expect-total`` 必须由调用方给出（800 或 192 等）。
- 结果行与清单的 ``dataset`` 必须等于 ``--dataset``（串了计 count_mismatch 与 dataset_crossed）；hard-verify 身份必备字段
  为 tier／seed／source_episode（不含 spec_sha256），不做 exec_over_cap（官方循环允许第 1301 步）。
- 视频在 ``<videos>/<policy>[-<variant>]/<dataset>/<side>/<key>.a<n>/``：转码后的 mp4（或 front.mkv＋wrist.mkv）读得出帧。
逐模型三行，全部 PASS 退出 0：
``EVAL_COVERAGE=PASS dataset=… policy=… expected=… missing=0 extra=0 duplicate=0 conflicting_terminal=0 error_final=0``
``EVAL_REPORT=PASS dataset=… policy=… count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=0|skip``
``EVAL_VIDEOS=PASS dataset=… policy=… expected=… videos=… missing=0 decode_fail=0``
``--partial`` 时三行一律 FAIL 并追加 ``partial=1``。

1006 第三阶段（seed 7、ood 1800 步；计划第二部分八.10 第 9 条、八.12 第 8 条）对带 ``--dataset`` 的口径追加：

- 真实步数上限（``--cap``，ood 给 1800）：每条结果行实际的 ``max_steps``／``effective_max_steps``／``effective_cap``
  （写了的都要等于 ``--cap``，一个都没写也算不符）必须等于 ``--cap``；能找到的局目录 ``trace.jsonl`` header
  写了 ``effective_cap`` 时只核 ``effective_cap == --cap``，不拿 header ``max_steps`` 比（SimpleMemVLA 的 header
  ``max_steps`` 记的是客户端理论动作上界 ``hard_bound(1800)×16 = 1840``，不是实际执行 cap；实际执行步数另由
  exec_over_cap 拦）；header 没写 ``effective_cap``（旧路线／旧口径）时仍核 header ``max_steps == --cap``。
  不符计 ``cap_mismatch``（EVAL_REPORT 追加 ``cap=`` 与该字段，>0 即 FAIL）——仍在 1600 截断的旧路线在
  ``--cap 1800`` 下不能误过。hard-verify 不做（官方循环口径不变）。
- 模型种子：按 ``(model, policy_seed)`` 分组报告，三行都带 ``policy_seed=<s>``。给 ``--policy-seed`` 时每条结果行的
  ``policy_seed`` 必须等于它（缺字段也算不符，不把历史缺字段补成已证种子）；不给时取结果行里唯一的种子，出现多个计
  ``policy_seed_mixed``。不符都计入 ``count_mismatch``。
- 无帧 error 例外（用户裁决「保留例外、报告单列」，三个检查器统一）：非 infra 错误终局、录像目录无视频且无帧（trace 末行
  ``no_frame`` 为真或 ``frames_recorded == 0``；找不到 trace 时以写明原因为准）的身份计 ``no_frame_error``，
  ``accepted = videos + no_frame_error``；EVAL_COVERAGE 的 ``error_final`` 只把无帧 error 之外的错误终局判 FAIL，
  两行都单列 ``no_frame_error``。有帧却没视频的错误终局照常计 ``missing``。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from robomme_ood_eval.report import speed_from_rows

TERMINAL = ("success", "fail", "timeout")
DEFAULT_CAP = 1600
DEFAULT_TOTAL = 1070
V9_DEFAULT_NEW = 80
V9_DEFAULT_REUSED = 720
REUSED_SCHEMA = "v9-eval-reused/1"
MEDIA_FILES = ("front.mkv", "wrist.mkv")


# ---------------------------------------------------------------- 读入

def hard_specs_path() -> Path:
    """子模块里 ``hard_specs.py`` 的位置：优先评估仓 ``third_party/robomme_benchmark/src/``；子模块未检出（如 git
    worktree）时按已安装 ``robomme_ood`` 包的位置找（``find_spec`` 只定位、不执行包的 ``__init__``）。"""
    p = Path(__file__).resolve().parents[2] / "third_party" / "robomme_benchmark" / "src" / "robomme_ood" / "env_record_wrapper" / "hard_specs.py"
    if p.is_file():
        return p
    import importlib.util

    spec = importlib.util.find_spec("robomme_ood")
    if spec is not None and spec.submodule_search_locations:
        cand = Path(list(spec.submodule_search_locations)[0]) / "env_record_wrapper" / "hard_specs.py"
        if cand.is_file():
            return cand
    return p


def official_defs():
    """评估包里的别名表 ``robomme_ood_eval.models._official_defs``（旧名别名表的唯一来源；模块名 ``official_defs``，
    已加载则复用同一模块）。"""
    import importlib.util

    mod = sys.modules.get("official_defs")
    if mod is None or not hasattr(mod, "canonical_row"):
        path = Path(__file__).resolve().parents[2] / "src" / "robomme_ood_eval" / "models" / "_official_defs.py"
        spec = importlib.util.spec_from_file_location("official_defs", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["official_defs"] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop("official_defs", None)
            raise
    return mod


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL；评估进程正在追加的半行（解析失败）跳过。历史行的旧策略标签／数据集名／路线映射成官方名。"""
    canon = official_defs().canonical_row
    out: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(canon(row))
    return out


def key_of(row: dict) -> str:
    if row.get("key"):
        return str(row["key"])
    return f"{row.get('task')}_{row.get('tier')}_{row.get('seed')}"


def seat_dirs(stage: Path) -> list[Path]:
    return sorted(p for p in stage.glob("s*") if p.is_dir() and p.name[1:].isdigit())


def parse_policy_spec(spec: str) -> tuple[str, str | None, str]:
    """``<policy>[:<variant>]`` → (策略名, 变体或 None, 运行根里的目录名 ``<policy>[-<variant>]``)。"""
    policy, _, variant = str(spec).partition(":")
    variant = variant or None
    return policy, variant, (f"{policy}-{variant}" if variant else policy)


def ledger_files(seat_dir: Path, policy: str, dirname: str | None = None) -> list[Path]:
    dirname = dirname or policy
    cands = [seat_dir / dirname / f"{policy}.ledger.jsonl", seat_dir / f"{dirname}.ledger.jsonl"]
    cands += sorted((seat_dir / dirname).glob("*.ledger.jsonl"))
    seen, out = set(), []
    for p in cands:
        if p.exists() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def load_policy(stage: Path, spec: str) -> dict:
    """读一个模型（``<policy>[:<variant>]``）全部席位的结果行与账本行；目录 ``sNN/<policy>[-<variant>]/``。
    结果行按 ``policy`` 与（给了变体时）``policy_variant`` 过滤。每行补 ``_seat``（目录名）与 ``_seat_dir``。"""
    policy, variant, dirname = parse_policy_spec(spec)
    results: list[dict] = []
    ledger: list[dict] = []
    defs = official_defs()
    #: 历史运行根里的旧目录名与账本文件名（读入后行内标签已映射成官方名）
    names = [(policy, dirname)] + list(zip(defs.legacy_labels(policy) or [policy], defs.legacy_labels(dirname)))
    for sd in seat_dirs(stage):
        pol_file, dirname_here = next(((p, d) for p, d in names if (sd / d).is_dir()), (policy, dirname))
        for row in read_jsonl(sd / dirname_here / "results.jsonl"):
            if row.get("policy") not in (None, policy):
                continue
            if variant is not None and row.get("policy_variant") not in (None, variant):
                continue
            row["_seat"], row["_seat_dir"] = sd.name, str(sd)
            results.append(row)
        for lp in ledger_files(sd, pol_file, dirname_here):
            for row in read_jsonl(lp):
                if row.get("policy") not in (None, policy):
                    continue
                row["_seat"] = sd.name
                ledger.append(row)
    return {"results": results, "ledger": ledger}


def rec_name(row: dict) -> str | None:
    if row.get("rec_dir"):
        return Path(str(row["rec_dir"])).name
    if row.get("attempt_no") is not None and (row.get("key") or row.get("task")):
        return f"{key_of(row)}.a{int(row['attempt_no'])}"
    return None


def is_error_final(row: dict) -> bool:
    """非 infra 错误是该身份的最终结局（E-A 会写 accept）：status=error、infra=false、非额度耗尽、非运行阻塞。"""
    return (row.get("status") == "error" and not row.get("infra") and not row.get("budget_exhausted")
            and not row.get("run_blocked"))


def error_reason(row: dict) -> str | None:
    why = row.get("error") or row.get("infra_reason")
    return str(why) if why else None


def error_final_video_class(row: dict, has_media: bool) -> str | None:
    """非 infra 错误终局的录像判定（报告与搬运脚本共用）：
    None = 不是错误终局或有录像（照常核对）；"explained" = 无录像但写明原因（不计缺失）；"unexplained" = 无录像无原因。"""
    if not is_error_final(row) or has_media:
        return None
    return "explained" if error_reason(row) else "unexplained"


def analyze_attempts(state: dict) -> dict:
    """按账本 accept 决定每身份唯一权威终态（与 manifest 无关的部分；搬运脚本也复用）。

    返回：
      by_attempt: attempt_id -> [结果行]（v8 行）
      accepted:   key -> 结果行（accept 指向且唯一、终态、无自相矛盾）
      conflicts:  [{key, reason, ...}]
      accept_no_row: [{key, attempt_id}]   accept 指向的结果行尚未出现
      duplicate:  重复计数（同 attempt_id 多条结果行、同一 accept 重复写）
      late:       [结果行]  accept 之外的终态行（迟到）
      abandoned:  [结果行]  非终态（error 等）且未被接受的尝试
    """
    # 金丝雀尝试（rec 目录 <key>.canary.a<n>）不进终态分母、不算迟到／废弃；录像仍由搬运脚本照搬
    rows = [r for r in state["results"] if r.get("v8") and not r.get("canary")]
    by_attempt: dict[str, list[dict]] = defaultdict(list)
    no_id: list[dict] = []
    for r in rows:
        if r.get("attempt_id"):
            by_attempt[str(r["attempt_id"])].append(r)
        else:
            no_id.append(r)
    duplicate = 0
    conflicts: list[dict] = []
    for aid, lst in by_attempt.items():
        if len(lst) > 1:
            duplicate += len(lst) - 1
    accepts: dict[str, list[str]] = defaultdict(list)
    for lr in state["ledger"]:
        if lr.get("kind") == "accept":
            aid = lr.get("accepted_attempt_id") or lr.get("attempt_id")
            accepts[str(lr.get("key"))].append(str(aid))
    late_ids = {str(lr.get("attempt_id")) for lr in state["ledger"]
                if lr.get("kind") == "attempt_end" and lr.get("late")}
    accepted: dict[str, dict] = {}
    accept_no_row: list[dict] = []
    for key, ids in accepts.items():
        uniq = sorted(set(ids))
        duplicate += len(ids) - len(uniq)
        if len(uniq) > 1:
            conflicts.append({"key": key, "reason": "multiple_accept", "attempt_ids": uniq})
            continue
        aid = uniq[0]
        lst = by_attempt.get(aid, [])
        if not lst:
            accept_no_row.append({"key": key, "attempt_id": aid})
            continue
        statuses = {r.get("status") for r in lst}
        if len(statuses) > 1:
            conflicts.append({"key": key, "reason": "attempt_rows_disagree", "attempt_id": aid,
                              "statuses": sorted(map(str, statuses))})
            continue
        row = lst[-1]
        if row.get("status") not in TERMINAL and not is_error_final(row):
            reason = "accept_infra_error" if row.get("status") == "error" and row.get("infra") else "accept_not_terminal"
            conflicts.append({"key": key, "reason": reason, "attempt_id": aid, "status": row.get("status")})
            continue
        if key_of(row) != key:
            conflicts.append({"key": key, "reason": "accept_key_mismatch", "attempt_id": aid, "row_key": key_of(row)})
            continue
        accepted[key] = row
    accepted_ids = {str(r["attempt_id"]) for r in accepted.values()}
    conflict_keys = {c["key"] for c in conflicts}
    late: list[dict] = []
    abandoned: list[dict] = []
    for aid, lst in by_attempt.items():
        if aid in accepted_ids:
            continue
        r = lst[-1]
        if r.get("status") in TERMINAL:
            if key_of(r) in conflict_keys:
                continue
            late.append(r)  # 未被接受的终态：迟到（账本 late=true）或尚未写 accept
        else:
            abandoned.append(r)
    for r in no_id:
        (late if r.get("status") in TERMINAL else abandoned).append(r)
    # 悬空尝试恢复：账本有 attempt_end 而 results.jsonl 无对应行 → 计入废弃尝试，不报错
    ended: dict[str, dict] = {}
    for lr in state["ledger"]:
        if lr.get("kind") == "attempt_end" and lr.get("attempt_id"):
            ended[str(lr["attempt_id"])] = lr
    pending_accept_ids = {a["attempt_id"] for a in accept_no_row}  # accept 已写、结果行未出现：只计冲突，不算废弃
    for aid, lr in ended.items():
        if aid in by_attempt or aid in accepted_ids or aid in pending_accept_ids:
            continue
        abandoned.append({"key": lr.get("key"), "attempt_id": aid, "attempt_no": lr.get("attempt_no"),
                          "status": lr.get("status") or "error", "infra": lr.get("infra"),
                          "infra_reason": lr.get("infra_reason") or lr.get("reason") or "ledger_only_attempt_end",
                          "ledger_only": True, "_seat": lr.get("_seat")})
    return {"rows": rows, "by_attempt": by_attempt, "accepted": accepted, "conflicts": conflicts,
            "accept_no_row": accept_no_row, "duplicate": duplicate, "late": late, "late_ids": late_ids,
            "abandoned": abandoned}


# ---------------------------------------------------------------- 媒体

def layout_names(dirname: str, dataset: str | None = None) -> list[tuple[str, str | None]]:
    """目录名与数据集目录名的候选：官方名在前，其后是历史运行根／视频根里的旧名组合（只读兼容）。"""
    defs = official_defs()
    dirs = [dirname] + defs.legacy_labels(dirname)
    dss = [dataset] + [old for old, new in defs.LEGACY_DATASET_ALIASES.items() if new == dataset]
    return [(d, ds) for d in dirs for ds in dss]


def find_media(row: dict, policy: str, videos: Path | None, *, dataset: str | None = None,
               side: str = "new") -> dict:
    """在运行根（尚未搬走）与本机视频根（已搬走）里找录像目录。``policy`` 为 ``<policy>[:<variant>]``。

    不带 ``dataset``（V8／V9 口径）：本机 ``<videos>/<目录名>/<tier>/<task>/<rec 目录名>/``；带 ``dataset``：本机
    ``<videos>/<目录名>/<dataset>/<side>/<rec 目录名>/``，再找结果行 ``rec_dir``。两种都最后找运行根
    ``sNN/<目录名>/rec/<rec 目录名>/``。"""
    dirname = parse_policy_spec(policy)[2]
    name = rec_name(row)
    if not name:
        return {"location": "absent", "path": None, "files": []}
    cands: list[tuple[str, Path]] = []
    names = layout_names(dirname, dataset)
    if videos is not None:
        for dn, ds in names:
            if dataset is None:
                cands.append(("local", videos / dn / str(row.get("tier") or "_notier") / str(row.get("task")) / name))
            else:
                cands.append(("local", videos / dn / ds / side / name))
    if dataset is not None and row.get("rec_dir"):
        cands.append(("stage", Path(str(row["rec_dir"]))))
    if row.get("_seat_dir"):
        for dn in dict.fromkeys(dn for dn, _ in names):
            cands.append(("stage", Path(row["_seat_dir"]) / dn / "rec" / name))
    for loc, p in cands:
        if p.is_dir():
            files = sorted(f.name for f in p.iterdir() if f.is_file())
            return {"location": loc, "path": str(p), "files": files}
    return {"location": "absent", "path": None, "files": []}


def media_files(m: dict, *, any_mp4: bool = False) -> list[str]:
    """录像目录里的视频文件：front.mkv 与 wrist.mkv 两路都在则取两路；``any_mp4`` 时另认转码后的 ``*.mp4``。"""
    if all(f in m["files"] for f in MEDIA_FILES):
        return list(MEDIA_FILES)
    if any_mp4:
        return [f for f in m["files"] if f.endswith(".mp4")]
    return []


def media_ok(m: dict, *, any_mp4: bool = False) -> bool:
    """V8／V9 口径要求 front.mkv、wrist.mkv、summary.json 齐全；``any_mp4``（带 --dataset）另认「转码后的 mp4 +
    summary.json」。"""
    return bool(media_files(m, any_mp4=any_mp4)) and "summary.json" in m["files"]


# ---------------------------------------------------------------- 主体

def load_manifest(path: Path) -> tuple[dict, list[dict], dict[str, str]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = list(doc.get("rows") or [])
    for r in rows:
        r["key"] = key_of(r)
    shard_of: dict[str, str] = {}
    for r in rows:
        if r.get("shard") is not None:
            shard_of[r["key"]] = f"{int(r['shard']):02d}" if str(r["shard"]).isdigit() else str(r["shard"])
    if not shard_of:
        for sp in sorted(path.parent.glob("shard-*.json")):
            sid = sp.stem.split("-", 1)[1]
            try:
                items = json.loads(sp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for it in items:
                shard_of[key_of(it)] = sid
    return doc, rows, shard_of


def load_shard_files(pattern: str | None) -> dict[str, str]:
    """shard-NN.json → {key: "NN"}；pattern 为空返回空表。"""
    out: dict[str, str] = {}
    if not pattern:
        return out
    import glob as _glob

    for sp in sorted(_glob.glob(pattern)):
        p = Path(sp)
        sid = p.stem.split("-", 1)[1] if "-" in p.stem else p.stem
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for it in items:
            out[key_of(it)] = sid
    return out


def cell_of(r: dict) -> str:
    return f"{r['task']}@{r['tier']}"


def _fmt_s(sec: float | None) -> str:
    if sec is None:
        return "未观测"
    h, rem = divmod(int(round(sec)), 3600)
    return f"{h}h{rem // 60:02d}m"


def speed_of(row: dict) -> dict | None:
    """只读新的策略计时；旧结果由调用方单列为缺失。"""
    timing = (row.get("timing") or {}).get("policy") or {}
    return timing if timing.get("chunk_summary") is not None else None


def _speed_mult(identities: list[dict]) -> str:
    triples = {(r.get("task"), r.get("tier"), r.get("episode")) for r in identities}
    tasks, tiers, episodes = (len({r[i] for r in triples}) for i in range(3))
    n = len(triples)
    if n and tasks * tiers * episodes == n:
        return f"{tasks} 任务 × {tiers} 档 × {episodes} 局 = {n}"
    return f"{n} 局（{tasks} 任务、{tiers} 档、局号 {episodes} 种，非满格／不齐）"


def _speed_section(rows_by_policy: dict[str, list[dict]]) -> dict:
    """与 log.json 共用分组、排除和原始稳态聚合；不改变成功率口径。"""
    rows = [{**r, "policy_label": r.get("policy_label") or label}
            for label, rows in rows_by_policy.items() for r in rows]
    speed = speed_from_rows(rows, include_empty=True)
    labels = {"planner+review": "规划+复核", "planner_only": "仅规划", "review_only": "仅复核",
              "neither": "仅监视", "no_lang": "无语言"}
    table = []
    for group in speed["groups"]:
        if not group["episodes_with_timing"]:
            continue
        layers = group.get("by_planner_review")
        sources = [(f"{group['policy_label']}·{labels[layer]}", sub, group["layer_identities"].get(layer, []))
                   for layer, sub in layers.items() if sub.get("decisions", 0)] if layers is not None else [
                       (group["policy_label"], group, group["identities"])]
        for label, summary, identities in sources:
            wall = summary["decision_wall_ms"]
            table.append({"policy_label": label, "gpu_name": group["gpu_name"], "mult": _speed_mult(identities),
                          **{k: summary[k] for k in ("decisions", "violations", "missing_fields", "checked", "expected",
                                                     "steady_short_eps", "approx", "episodes_with_timing")},
                          "wall_first_ms": wall["first_mean"], "wall_steady_mean_ms": wall["steady_mean"],
                          "wall_steady_p50_ms": wall["steady_p50"], "wall_steady_p95_ms": wall["steady_p95"],
                          **{k: summary[k] for k in ("action_rtt_steady_mean_ms", "lang_steady_mean_ms", "server_infer_steady_mean_ms")}})
    # 父组证据独立保留，不能被 Astra 原始 chunks 的子层重算补成通过；父子不相加。
    conservation = {k: sum(group[k] for group in speed["groups"])
                    for k in ("violations", "missing_fields", "checked", "expected")}
    return {"rows": table, "conservation": conservation,
            **{k: speed[k] for k in ("missing", "excluded_infra", "excluded_timeout", "excluded_reconnected", "excluded_status")}}


def speed_table(rows_by_policy: dict[str, list[dict]]) -> list[dict]:
    return _speed_section(rows_by_policy)["rows"]


def speed_table_from_results(paths: list[Path]) -> dict:
    """显式 results.jsonl 路径直接读取，不扫描席位目录。"""
    rows = [row for path in paths for row in read_jsonl(path)]
    by_policy = defaultdict(list)
    for row in rows:
        label = row.get("policy_label") or row.get("model") or row.get("policy") or "unknown"
        if row.get("policy_variant") and not row.get("policy_label"):
            label = f"{label}:{row['policy_variant']}"
        by_policy[label].append(row)
    return {"schema": "v8-eval-report/2", "speed": _speed_section(by_policy)}


def speed_line(speed: dict) -> str:
    totals = {k: sum(row[k] for row in speed["rows"]) for k in ("decisions", "violations", "missing_fields", "checked", "expected")}
    parent = speed.get("conservation", totals)
    passed = all(not counts["violations"] and not counts["missing_fields"] and counts["checked"] == counts["expected"]
                 for counts in (parent, totals))
    return (f"SPEED_TABLE={'PASS' if passed else 'FAIL'} rows={len(speed['rows'])} "
            f"decisions={totals['decisions']} violations={parent['violations']} missing={speed['missing']} "
            f"checked={parent['checked']} expected={parent['expected']} missing_fields={parent['missing_fields']} "
            f"layer_violations={totals['violations']} layer_missing_fields={totals['missing_fields']} "
            f"layer_checked={totals['checked']} layer_expected={totals['expected']}")


def _fmt_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} ms"


def _render_speed(speed: dict) -> list[str]:
    lines = ["## 推理速度", "", "| 模型 | GPU | decision 数（乘式） | 首次 | 稳态 mean | p50 | p95 | 动作往返 | 语言 | 服务端推理 | 守恒违例 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fields = ("wall_first_ms", "wall_steady_mean_ms", "wall_steady_p50_ms", "wall_steady_p95_ms",
              "action_rtt_steady_mean_ms", "lang_steady_mean_ms", "server_infer_steady_mean_ms")
    for row in speed["rows"]:
        label = row["policy_label"] + ("（分位数近似）" if row["approx"] else "")
        lines.append(f"| {label} | {row['gpu_name'] or '未知'} | {row['decisions']}（{row['mult']}） | "
                     + " | ".join(_fmt_ms(row[k]) for k in fields) + f" | {row['violations']} |")
    lines.extend(["", f"旧计时缺失 {speed['missing']} 局；排除 infra={speed['excluded_infra']}、timeout={speed['excluded_timeout']}、PP 重连={speed['excluded_reconnected']}、其他非正常终局={speed['excluded_status']}。",
                  "GL A40 与本机 RTX 6000 Ada 不混比；未知 GPU 单独分组。PP 的 decision_wall_ms 为 fresh 步往返与桶内 S2 的合成值。Astra 含规划行包含 SEND_INTERVAL_S=20。", ""])
    return lines


def wall_of(row: dict) -> float | None:
    t = row.get("timing") or {}
    v = t.get("episode_wall_s") if isinstance(t, dict) else None
    if v is None:
        v = row.get("wall_s")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


CAP_FIELDS = ("max_steps", "effective_max_steps", "effective_cap")


def read_trace_edges(d: Path | None) -> tuple[dict, dict]:
    """局目录 ``trace.jsonl`` 的 (header, end)；读不到返回两个空字典。"""
    if d is None:
        return {}, {}
    try:
        lines = [x for x in (Path(d) / "trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        head, end = (json.loads(lines[0]), json.loads(lines[-1])) if lines else ({}, {})
    except (OSError, ValueError):
        return {}, {}
    head = head if isinstance(head, dict) and head.get("kind") == "header" else {}
    end = end if isinstance(end, dict) and end.get("kind") == "end" else {}
    return head, end


def cap_problems(row: dict, cap: int, media: dict | None = None) -> list[str]:
    """结果行实际步数上限与 ``cap`` 不符的地方：写了的 ``max_steps``／``effective_max_steps``／``effective_cap``
    都要等于 cap、至少写一个；局目录找得到 trace 时，header 写了 ``effective_cap`` 就只核它等于 cap（header
    ``max_steps`` 可能是理论循环上界，如 SimpleMemVLA 的 115×16=1840），没写 ``effective_cap`` 才核 header
    ``max_steps`` 等于 cap（旧路线保护）。"""
    out = []
    present = {f: row.get(f) for f in CAP_FIELDS if row.get(f) is not None}
    if not present:
        out.append("result_cap_fields_absent")
    for f, v in present.items():
        try:
            ok = int(v) == int(cap)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(f"result.{f}={v}")
    path = (media or {}).get("path")
    head, _ = read_trace_edges(Path(path)) if path else ({}, {})
    f = "effective_cap" if head.get("effective_cap") is not None else "max_steps"
    if head.get(f) is not None and head.get(f) != cap:
        out.append(f"trace.header.{f}={head.get(f)}")
    return out


def no_frame_of(row: dict, d: Path | None) -> bool | None:
    """该错误终局是否无帧：结果行 ``no_frame`` 或局目录 trace 末行（``no_frame`` 为真或 ``frames_recorded == 0``）；
    两处都没有信息返回 None（由调用方按「写明原因」口径判）。"""
    if row.get("no_frame") is True:
        return True
    _, end = read_trace_edges(d)
    if not end:
        return None
    return end.get("no_frame") is True or end.get("frames_recorded") == 0


#: 每个数据集结果行必备的身份字段与逐键比对清单的字段（hard-verify 无规格指纹，改核 source_episode）
ID_REQUIRED = {None: ("tier", "seed", "spec_sha256"), "ood": ("tier", "seed", "spec_sha256"),
               "hard-verify": ("tier", "seed", "source_episode")}
ID_COMPARE = {None: ("tier", "seed", "candidate", "spec_sha256"),
              "ood": ("tier", "seed", "candidate", "spec_sha256"),
              "hard-verify": ("tier", "seed", "candidate", "source_episode")}
DATASETS = ("ood", "hard-verify")


def build_report(manifest_path: Path, stage: Path, policies: list[str], videos: Path | None, *,
                 partial: bool, expect_total: int, cap: int | None, shard_files: str | None = None,
                 keep_internal: bool = False, dataset: str | None = None, side: str = "new",
                 policy_seed: int | None = None) -> dict:
    """``dataset``（--dataset）为空时是 V8／V9 口径；给出时身份必备字段按数据集取、结果行与清单的数据集必须一致
    （串了计 count_mismatch 与 dataset_crossed），``cap`` 为 None 时不做越限判定（hard-verify）。"""
    doc, mrows, shard_of = load_manifest(manifest_path)
    mkeys = {r["key"]: r for r in mrows}
    count_mismatch: list[str] = []
    if len(mrows) != expect_total:
        count_mismatch.append(f"manifest rows={len(mrows)} != expect_total={expect_total}")
    if doc.get("total") is not None and int(doc["total"]) != len(mrows):
        count_mismatch.append(f"manifest total={doc['total']} != rows={len(mrows)}")
    if len(mkeys) != len(mrows):
        count_mismatch.append(f"manifest key 重复 {len(mrows) - len(mkeys)}")
    cell_den = Counter(cell_of(r) for r in mkeys.values())
    if doc.get("cells") and dict(doc["cells"]) != dict(cell_den):
        count_mismatch.append("manifest cells 与 rows 逐格计数不一致")
    task_den = Counter(r["task"] for r in mkeys.values())
    if dataset is not None:
        if official_defs().canonical_dataset(doc.get("dataset")) not in (None, dataset):
            count_mismatch.append(f"manifest dataset={doc.get('dataset')} != --dataset {dataset}")
        if dataset == "hard-verify":
            wrong = [k for k, r in mkeys.items() if r.get("tier") != "xhard0"]
        else:
            wrong = [k for k, r in mkeys.items() if r.get("tier") == "xhard0"]
        if wrong:
            count_mismatch.append(f"manifest 有 {len(wrong)} 行档位与 --dataset {dataset} 不符：{wrong[:5]}")
    cm_global = len(count_mismatch)
    id_required, id_compare = ID_REQUIRED[dataset], ID_COMPARE[dataset]

    cov = Counter()
    exec_over_cap: list[dict] = []
    cap_mismatch: list[dict] = []
    media_unexplained: list[dict] = []
    index: list[dict] = []
    per_policy: dict[str, Any] = {}
    observed_seat: dict[str, str] = {}
    speed_inputs = {}

    for pol in policies:
        cm_before = len(count_mismatch)
        st = load_policy(stage, pol)
        speed_inputs[pol] = st["results"]
        # 实际所在席位以运行根里观测到的目录为准（分片可能重分到别的 sNN），不依赖 manifest 的 shard 字段
        for r in st["results"] + st["ledger"]:
            if r.get("key") and r.get("_seat"):
                observed_seat[str(r["key"])] = r["_seat"][1:]
        an = analyze_attempts(st)
        acc = an["accepted"]
        conflict_n = len(an["conflicts"]) + (0 if partial else len(an["accept_no_row"]))
        seen_keys = {key_of(r) for r in an["rows"]} | {str(l.get("key")) for l in st["ledger"] if l.get("key")}
        extra = sorted(k for k in seen_keys if k not in mkeys)
        conflict_keys = {c["key"] for c in an["conflicts"]}
        if not partial:  # accept 已写、结果行未出现：正式汇总只计 conflicting_terminal，不重复计 missing
            conflict_keys |= {a["key"] for a in an["accept_no_row"]}
        outcome: dict[str, str] = {}
        for k in mkeys:
            if k in acc:
                outcome[k] = acc[k]["status"]  # success／fail／timeout，或非 infra 错误终局 error
            elif k in conflict_keys:
                outcome[k] = "conflict"
            else:
                outcome[k] = "missing"  # 无 accept：只有 infra 错误尝试或一次都没跑，错误尝试另列废弃
        missing = [k for k in mkeys if k not in acc and k not in conflict_keys]  # 冲突身份只计 conflicting_terminal
        cov["missing"] += 0 if partial else len(missing)
        cov["error_final"] += sum(1 for k in mkeys if k in acc and acc[k].get("status") == "error")
        cov["extra"] += len(extra)
        cov["duplicate"] += an["duplicate"]
        cov["conflicting_terminal"] += conflict_n
        cov["late_ignored"] += len(an["late"])

        # 一致性：成功字段、身份字段
        for k, r in acc.items():
            if k not in mkeys:
                continue
            if bool(r.get("task_success")) != (r.get("status") == "success"):
                count_mismatch.append(f"{pol} {k} task_success={r.get('task_success')} status={r.get('status')}")
            m = mkeys[k]
            ident = dict(r.get("identity") or {})
            for f in id_compare:
                mv = m.get(f)
                rv = r.get(f, ident.get(f))
                if mv is not None and rv is not None and str(mv) != str(rv):
                    count_mismatch.append(f"{pol} {k} {f} manifest={mv} result={rv}")
        oc = Counter(outcome.values())
        if sum(oc.values()) != len(mkeys):
            count_mismatch.append(f"{pol} 结局计数和 {sum(oc.values())} != 分母 {len(mkeys)}")

        # 必备字段：缺 exec_steps 或身份字段（tier／seed／spec_sha256，hard-verify 为 tier／seed／source_episode）的
        # 结果行不静默放过；带 --dataset 时结果行的 dataset 必须等于它（两个数据集不串）
        crossed = 0
        for r in an["rows"]:
            ident = dict(r.get("identity") or {})
            lacks = [f for f in id_required if r.get(f, ident.get(f)) is None]
            if dataset is not None and r.get("dataset") != dataset:
                crossed += 1
                count_mismatch.append(f"{pol} {key_of(r)} attempt_id={r.get('attempt_id')} "
                                      f"dataset={r.get('dataset')} != --dataset {dataset}")
            if "exec_steps" not in r or (r.get("status") in TERMINAL and r.get("exec_steps") is None):
                lacks.append("exec_steps")
            if lacks:
                count_mismatch.append(f"{pol} {key_of(r)} "
                                      f"attempt_id={r.get('attempt_id')} 缺字段 {','.join(lacks)}")

        # 越限（cap 为 None 即 hard-verify：官方循环允许第 max_steps+1 步，不判）
        for r in an["rows"]:
            es = r.get("exec_steps")
            if cap is not None and es is not None and int(es) > cap:
                exec_over_cap.append({"policy": pol, "key": key_of(r), "attempt_id": r.get("attempt_id"),
                                      "exec_steps": int(es)})

        # 1006：真实步数上限（只在带 --dataset 且做越限判定时核；ood 给 --cap 1800）
        if dataset is not None and cap is not None:
            for r in an["rows"]:
                bad = cap_problems(r, cap, find_media(r, pol, videos, dataset=dataset, side=side))
                if bad:
                    cap_mismatch.append({"policy": pol, "key": key_of(r), "attempt_id": r.get("attempt_id"),
                                         "problems": bad})

        # 1006：模型种子（按 (model, policy_seed) 分组；给了 --policy-seed 时逐行核）
        seeds = sorted({r.get("policy_seed") for r in an["rows"]}, key=str)
        if policy_seed is not None:
            for r in an["rows"]:
                if r.get("policy_seed") != policy_seed:
                    count_mismatch.append(f"{pol} {key_of(r)} attempt_id={r.get('attempt_id')} "
                                          f"policy_seed={r.get('policy_seed')} != --policy-seed {policy_seed}")
            group_seed = policy_seed
        else:
            known = [x for x in seeds if x is not None]
            if len(seeds) > 1:
                count_mismatch.append(f"{pol} policy_seed_mixed {seeds}")
            group_seed = known[0] if len(known) == 1 and len(seeds) == 1 else None

        # 媒体与视频索引（每次尝试一行）
        accepted_ids = {str(r.get("attempt_id")) for r in acc.values()}
        late_ids = {str(r.get("attempt_id")) for r in an["late"]}
        no_video_errors: list[dict] = []
        for aid, lst in sorted(an["by_attempt"].items(), key=lambda kv: (key_of(kv[1][-1]), kv[0])):
            r = lst[-1]
            m = find_media(r, pol, videos, dataset=dataset, side=side)
            mok = media_ok(m, any_mp4=dataset is not None)
            is_acc = aid in accepted_ids
            reason = None
            efc = error_final_video_class(r, mok) if is_acc else None
            if efc is not None:
                # 非 infra 错误终局无录像：写明原因即可（与搬运脚本 error_final_no_video 同一判定）
                why = error_reason(r)
                no_video_errors.append({"key": key_of(r), "attempt_id": aid, "status": "error",
                                        "reason": str(why)[:200] if why else None, "location": m["location"]})
                if efc == "unexplained" and not partial:
                    reason = "error_final_without_video_and_reason"
                    media_unexplained.append({"policy": pol, "key": key_of(r), "attempt_id": aid, "reason": reason,
                                              "path": m["path"]})
            elif is_acc:
                if not mok:
                    reason = "accepted_terminal_media_absent" if m["location"] == "absent" else "accepted_terminal_media_incomplete"
                    if not partial:
                        media_unexplained.append({"policy": pol, "key": key_of(r), "attempt_id": aid, "reason": reason,
                                                  "path": m["path"]})
                elif r.get("recorder_verify") not in (None, "PASS"):
                    reason = f"recorder_verify={r.get('recorder_verify')}"
                    media_unexplained.append({"policy": pol, "key": key_of(r), "attempt_id": aid, "reason": reason,
                                              "path": m["path"]})
            elif r.get("status") not in TERMINAL and not mok:
                why = r.get("infra_reason") or r.get("error")
                no_video_errors.append({"key": key_of(r), "attempt_id": aid, "status": r.get("status"),
                                        "reason": str(why)[:200] if why else None, "location": m["location"]})
                if not why and not partial:
                    media_unexplained.append({"policy": pol, "key": key_of(r), "attempt_id": aid,
                                              "reason": "error_without_video_and_reason", "path": m["path"]})
            index.append({"policy": pol, "key": key_of(r), "task": r.get("task"), "tier": r.get("tier"),
                          "seat": r.get("_seat"), "attempt_id": aid, "attempt_no": r.get("attempt_no"),
                          "status": r.get("status"), "accepted": is_acc, "late": aid in late_ids,
                          "infra": r.get("infra"), "rec_name": rec_name(r), "location": m["location"],
                          "path": m["path"], "files": m["files"], "recorder_verify": r.get("recorder_verify"),
                          "media_ok": mok, "issue": reason})

        # 分数
        cells: dict[str, dict] = {}
        for c, den in sorted(cell_den.items()):
            ks = [k for k in mkeys if cell_of(mkeys[k]) == c]
            cc = Counter(outcome[k] for k in ks)
            cells[c] = {"denominator": den, "done": sum(cc[s] for s in (*TERMINAL, "error")), "success": cc["success"],
                        "fail": cc["fail"], "timeout": cc["timeout"], "error": cc["error"], "missing": cc["missing"],
                        "conflict": cc["conflict"], "success_rate": cc["success"] / den if den else None}
        tasks: dict[str, dict] = {}
        for t, den in sorted(task_den.items()):
            s = sum(1 for k in mkeys if mkeys[k]["task"] == t and outcome[k] == "success")
            tasks[t] = {"denominator": den, "success": s, "success_rate": s / den if den else None}
        tiers: dict[str, dict] = {}
        for tier in sorted({str(r["tier"]) for r in mkeys.values()}):
            ks = [k for k in mkeys if str(mkeys[k]["tier"]) == tier]
            s = sum(1 for k in ks if outcome[k] == "success")
            tiers[tier] = {"denominator": len(ks), "success": s, "success_rate": s / len(ks) if ks else None}
        micro = oc["success"] / len(mkeys) if mkeys else None
        macro = statistics.fmean(v["success_rate"] for v in tasks.values()) if tasks else None

        # 预算（逐席）
        seats: dict[str, dict] = defaultdict(lambda: {"reset_claim": 0, "attempts": 0, "infra_retries": 0,
                                                     "reset_budget": None, "infra_retry_budget": None,
                                                     "budget_raise": [], "budget_exhausted": 0, "run_blocked": 0,
                                                     "infra_error_attempts": 0})
        for l in st["ledger"]:
            s = seats[l["_seat"]]
            kind = l.get("kind")
            if kind == "reset_claim":
                s["reset_claim"] += 1
            elif kind == "attempt_start":
                s["attempts"] += 1
                if int(l.get("attempt_no") or 1) >= 2:
                    s["infra_retries"] += 1
            elif kind == "budget":
                for f in ("reset_budget", "infra_retry_budget"):
                    if l.get(f) is not None:
                        s[f] = max(int(l[f]), s[f] or 0)
            elif kind == "budget_raise":
                s["budget_raise"].append({"from": l.get("from"), "to": l.get("to"), "reason": l.get("reason")})
        for r in an["rows"]:
            s = seats[r["_seat"]]
            s["budget_exhausted"] += bool(r.get("budget_exhausted"))
            s["run_blocked"] += bool(r.get("run_blocked"))
            s["infra_error_attempts"] += bool(r.get("infra")) and r.get("status") == "error"
        budget = {"seats": dict(sorted(seats.items())),
                  "reset_claim_total": sum(s["reset_claim"] for s in seats.values()),
                  "infra_retries_total": sum(s["infra_retries"] for s in seats.values()),
                  "attempts_total": sum(s["attempts"] for s in seats.values()),
                  "budget_raise_total": sum(len(s["budget_raise"]) for s in seats.values())}

        per_policy[pol] = {
            "denominator": len(mkeys), "outcomes": {s: oc[s] for s in (*TERMINAL, "error", "missing", "conflict")},
            "accepted": len(acc), "micro_success_rate": micro, "task_macro_success_rate": macro,
            "cells": cells, "tasks": tasks, "tiers": tiers, "budget": budget,
            "extra_keys": extra, "conflicts": an["conflicts"], "accept_without_row": an["accept_no_row"],
            "duplicate": an["duplicate"],
            "late": [{"key": key_of(r), "attempt_id": r.get("attempt_id"), "attempt_no": r.get("attempt_no"),
                      "status": r.get("status"), "ledger_late": str(r.get("attempt_id")) in an["late_ids"]}
                     for r in an["late"]],
            "abandoned": [{"key": key_of(r), "attempt_id": r.get("attempt_id"), "attempt_no": r.get("attempt_no"),
                           "status": r.get("status"), "infra": r.get("infra"),
                           "reason": str(r.get("infra_reason") or r.get("error") or "")[:200]} for r in an["abandoned"]],
            "errors_without_video": no_video_errors,
            # 该模型自己的判定计数（带 --dataset 时逐模型出判定行）；count_mismatch 含清单级的公共项
            "coverage": {"missing": len(missing), "extra": len(extra), "duplicate": an["duplicate"],
                         "conflicting_terminal": conflict_n, "late_ignored": len(an["late"]),
                         "error_final": sum(1 for k in mkeys if k in acc and acc[k].get("status") == "error")},
            "count_mismatch": cm_global + (len(count_mismatch) - cm_before),
            "dataset_crossed": crossed, "policy_seed": group_seed, "policy_seeds_seen": seeds,
            "_acc": acc, "_outcome": outcome,
        }

    # 中途进度：按已完成局墙钟估算各席剩余
    progress = None
    if partial:
        seat_of = {**shard_of, **load_shard_files(shard_files), **observed_seat}
        progress = estimate_progress(mkeys, seat_of, per_policy, policies, cell_den)
    if not keep_internal:
        for p in per_policy.values():
            p.pop("_acc")
            p.pop("_outcome")

    coverage_pass = all(cov[k] == 0 for k in ("missing", "extra", "duplicate", "conflicting_terminal"))
    report_pass = not count_mismatch and not media_unexplained and not exec_over_cap and not cap_mismatch
    return {
        "schema": "v8-eval-report/2", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "speed": _speed_section(speed_inputs),
        "manifest": str(manifest_path), "stage": str(stage), "videos": str(videos) if videos else None,
        "partial": partial, "cap": cap, "expect_total": expect_total, "policies": policies, "dataset": dataset,
        "side": side if dataset is not None else None,
        "coverage": {"pass": coverage_pass, **{k: cov[k] for k in
                                              ("missing", "extra", "duplicate", "conflicting_terminal", "late_ignored", "error_final")}},
        "report": {"pass": report_pass, "count_mismatch": len(count_mismatch),
                   "media_unexplained": len(media_unexplained), "exec_over_cap": len(exec_over_cap),
                   "cap_mismatch": len(cap_mismatch)},
        "count_mismatch_detail": count_mismatch, "media_unexplained_detail": media_unexplained,
        "exec_over_cap_detail": exec_over_cap, "cap_mismatch_detail": cap_mismatch, "policy_seed": policy_seed, "per_policy": per_policy, "progress": progress,
        "_index": index,
    }


def estimate_progress(mkeys: dict, shard_of: dict, per_policy: dict, policies: list[str], cell_den: Counter) -> dict:
    """各席剩余 = Σ 模型 Σ 该席未完成身份 × 该模型该任务已完成局平均墙钟（无观测时退回该模型全局平均并标出）。
    每席按 policies 顺序串行跑两个模型，总剩余 = 最慢席位剩余。"""
    out: dict[str, Any] = {"policies": {}, "seats": {}, "unobserved_cells": {}}
    est_task: dict[str, dict[str, float]] = {}
    est_pol: dict[str, float | None] = {}
    for pol in policies:
        acc = per_policy[pol]["_acc"]
        walls: dict[str, list[float]] = defaultdict(list)
        for k, r in acc.items():
            w = wall_of(r)
            if w is not None and k in mkeys:
                walls[mkeys[k]["task"]].append(w)
        est_task[pol] = {t: statistics.fmean(v) for t, v in walls.items()}
        allw = [w for v in walls.values() for w in v]
        est_pol[pol] = statistics.fmean(allw) if allw else None
        done = per_policy[pol]["accepted"]
        out["policies"][pol] = {"done": done, "denominator": len(mkeys),
                                "cells_done": {c: v["done"] for c, v in per_policy[pol]["cells"].items()},
                                "mean_wall_s": est_pol[pol]}
        out["unobserved_cells"][pol] = sorted(c for c, v in per_policy[pol]["cells"].items() if v["done"] == 0)
    seats = sorted(set(shard_of.values())) or ["?"]
    for sid in seats:
        total = 0.0
        unknown = False
        detail = {}
        for pol in policies:
            outcome = per_policy[pol]["_outcome"]
            rem = [k for k in mkeys if shard_of.get(k, "?") == sid and outcome[k] not in (*TERMINAL, "error")]
            secs = 0.0
            fallback = 0
            for k in rem:
                t = mkeys[k]["task"]
                if t in est_task[pol]:
                    secs += est_task[pol][t]
                elif est_pol[pol] is not None:
                    secs += est_pol[pol]
                    fallback += 1
                else:
                    unknown = True
            detail[pol] = {"remaining": len(rem), "est_s": None if (rem and est_pol[pol] is None) else secs,
                           "fallback_mean_used": fallback}
            total += secs
        out["seats"][sid] = {"est_remaining_s": None if unknown else total, "detail": detail}
    known = [v["est_remaining_s"] for v in out["seats"].values()]
    out["est_remaining_s"] = None if any(v is None for v in known) else (max(known) if known else 0.0)
    out["slowest_seat"] = (max(out["seats"], key=lambda s: out["seats"][s]["est_remaining_s"] or 0)
                           if out["seats"] and out["est_remaining_s"] is not None else None)
    return out


# ---------------------------------------------------------------- 输出

def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def render_md(rep: dict, *verdict_lines: str) -> str:
    L: list[str] = []
    ds = rep.get("dataset")
    title = f"# 评估汇总（dataset={ds}）" if ds else "# V8 双模型评估汇总"
    L.append(title + ("（中途进度）" if rep["partial"] else ""))
    L.append("")
    L.append(f"- 生成时间：{rep['generated_at']}")
    L.append(f"- manifest：`{rep['manifest']}`；运行根：`{rep['stage']}`；本机视频根：`{rep['videos']}`")
    cap_txt = (f"执行步上限 {rep['cap']}，超过即判 FAIL。" if rep["cap"] is not None
               else "hard-verify 不做执行步越限判定（官方循环允许第 max_steps+1 步）。")
    L.append(f"- 口径：每身份唯一权威终态取账本 `accept` 行的 `accepted_attempt_id`；迟到终态与废弃尝试单列、不入分数；"
             f"分母固定为 manifest 身份数（{rep['expect_total']}／模型）；{cap_txt}"
             + ("中途进度下「缺失」即尚未完成，不判 FAIL；判定行的 missing 不计。" if rep["partial"] and not ds else "")
             + ("中途进度下判定行一律 FAIL 并标 partial=1。" if rep["partial"] and ds else ""))
    L.append("")
    L.append("```")
    L.extend(verdict_lines)
    L.append("```")
    L.append("")
    L.append("## 总表")
    L.append("")
    L.append("| 模型 | 分母 | 已定终态 | 成功 | 失败 | timeout | error | 缺失 | 冲突 | 全局微平均 | 任务宏平均 | 迟到 | 废弃尝试 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pol, p in rep["per_policy"].items():
        o = p["outcomes"]
        L.append(f"| {pol} | {p['denominator']} | {p['accepted']} | {o['success']} | {o['fail']} | {o['timeout']} | "
                 f"{o['error']} | {o['missing']} | {o['conflict']} | {pct(p['micro_success_rate'])} | "
                 f"{pct(p['task_macro_success_rate'])} | {len(p['late'])} | {len(p['abandoned'])} |")
    L.append("")
    pols = list(rep["per_policy"])
    if "speed" in rep:
        L.extend(_render_speed(rep["speed"]))
    L.append("## 任务×档成功率（成功／分母；括号内 timeout／error／缺失）")
    L.append("")
    L.append("| 格 | " + " | ".join(pols) + " |")
    L.append("|---|" + "---:|" * len(pols))
    cells = sorted(next(iter(rep["per_policy"].values()))["cells"]) if pols else []
    for c in cells:
        cols = []
        for pol in pols:
            v = rep["per_policy"][pol]["cells"][c]
            cols.append(f"{v['success']}/{v['denominator']} {pct(v['success_rate'])} "
                        f"({v['timeout']}/{v['error']}/{v['missing']})")
        L.append(f"| {c} | " + " | ".join(cols) + " |")
    L.append("")
    L.append("## 按任务（跨档）与按档（跨任务）")
    L.append("")
    L.append("| 任务 | " + " | ".join(pols) + " |")
    L.append("|---|" + "---:|" * len(pols))
    tasks = sorted(next(iter(rep["per_policy"].values()))["tasks"]) if pols else []
    for t in tasks:
        L.append(f"| {t} | " + " | ".join(
            f"{rep['per_policy'][p]['tasks'][t]['success']}/{rep['per_policy'][p]['tasks'][t]['denominator']} "
            f"{pct(rep['per_policy'][p]['tasks'][t]['success_rate'])}" for p in pols) + " |")
    L.append("")
    L.append("| 档 | " + " | ".join(pols) + " |")
    L.append("|---|" + "---:|" * len(pols))
    tiers = sorted(next(iter(rep["per_policy"].values()))["tiers"]) if pols else []
    for t in tiers:
        L.append(f"| {t} | " + " | ".join(
            f"{rep['per_policy'][p]['tiers'][t]['success']}/{rep['per_policy'][p]['tiers'][t]['denominator']} "
            f"{pct(rep['per_policy'][p]['tiers'][t]['success_rate'])}" for p in pols) + " |")
    L.append("")
    L.append("## 预算（逐席）")
    L.append("")
    for pol, p in rep["per_policy"].items():
        b = p["budget"]
        L.append(f"**{pol}**：reset_claim 总数 {b['reset_claim_total']}，尝试 {b['attempts_total']}，"
                 f"infra 重试 {b['infra_retries_total']}，额度提升 {b['budget_raise_total']} 次。")
        L.append("")
        L.append("| 席 | reset_claim | reset 额度 | 尝试 | infra 重试 | infra 重试额度 | infra 错误尝试 | 额度耗尽 | 运行阻塞 | 额度提升 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for sid, s in b["seats"].items():
            raises = "；".join(f"{x['from']}→{x['to']}（{x['reason']}）" for x in s["budget_raise"]) or "—"
            L.append(f"| {sid} | {s['reset_claim']} | {s['reset_budget']} | {s['attempts']} | {s['infra_retries']} | "
                     f"{s['infra_retry_budget']} | {s['infra_error_attempts']} | {s['budget_exhausted']} | "
                     f"{s['run_blocked']} | {raises} |")
        L.append("")
    L.append("## 迟到终态与废弃尝试（不入分数）")
    L.append("")
    any_side = False
    for pol, p in rep["per_policy"].items():
        for x in p["late"]:
            any_side = True
            L.append(f"- {pol} 迟到 `{x['key']}` attempt_no={x['attempt_no']} status={x['status']} "
                     f"账本 late={x['ledger_late']}")
        for x in p["abandoned"]:
            any_side = True
            L.append(f"- {pol} 废弃 `{x['key']}` attempt_no={x['attempt_no']} status={x['status']} infra={x['infra']} "
                     f"原因：{x['reason'] or '（无）'}")
    if not any_side:
        L.append("- 无")
    L.append("")
    L.append("## 异常明细")
    L.append("")
    lines = []
    for pol, p in rep["per_policy"].items():
        for c in p["conflicts"]:
            lines.append(f"- {pol} 冲突终态 `{c['key']}`：{c['reason']} {json.dumps({k: v for k, v in c.items() if k not in ('key', 'reason')}, ensure_ascii=False)}")
        for c in p["accept_without_row"]:
            lines.append(f"- {pol} accept 指向的结果行未出现 `{c['key']}` attempt_id={c['attempt_id']}")
        if p["extra_keys"]:
            lines.append(f"- {pol} manifest 之外的身份 {len(p['extra_keys'])} 个：{', '.join(p['extra_keys'][:20])}")
        for e in p["errors_without_video"]:
            lines.append(f"- {pol} 无录像的错误尝试 `{e['key']}` status={e['status']}：{e['reason'] or '（未说明）'}")
    for x in rep["exec_over_cap_detail"]:
        lines.append(f"- exec_steps 越限 {x['policy']} `{x['key']}` exec_steps={x['exec_steps']} > {rep['cap']}")
    for x in (rep.get("dataset_videos") or {}).get("problems", [])[:200]:
        lines.append(f"- 视频 {x['policy']} `{x['key']}`：{x['problem']} {x.get('dir') or ''}")
    for x in rep["count_mismatch_detail"]:
        lines.append(f"- 计数不一致：{x}")
    for x in rep["media_unexplained_detail"]:
        lines.append(f"- 媒体无法解释 {x['policy']} `{x['key']}`：{x['reason']}")
    L.extend(lines or ["- 无"])
    L.append("")
    pr = rep.get("progress")
    if pr:
        L.append("## 中途进度")
        L.append("")
        for pol, v in pr["policies"].items():
            L.append(f"- {pol}：已完成 {v['done']}/{v['denominator']}，已完成局平均墙钟 {_fmt_s(v['mean_wall_s'])}")
            un = pr["unobserved_cells"][pol]
            L.append(f"  - 未观测格（尚无完成局，用该模型全局平均估算）{len(un)} 个：{', '.join(un) if un else '无'}")
        L.append("")
        L.append(f"预计剩余（最慢席位）：**{_fmt_s(pr['est_remaining_s'])}**（席 {pr['slowest_seat']}）")
        L.append("")
        L.append("| 席 | 预计剩余 | " + " | ".join(f"{p} 剩余局" for p in pols) + " |")
        L.append("|---|---:|" + "---:|" * len(pols))
        for sid, s in pr["seats"].items():
            L.append(f"| {sid} | {_fmt_s(s['est_remaining_s'])} | " +
                     " | ".join(str(s["detail"][p]["remaining"]) for p in pols) + " |")
        L.append("")
    L.append("视频索引见同目录 `video-index.jsonl`（每次尝试一行，含终态、迟到、错误尝试）。")
    return "\n".join(L) + "\n"


def lines_of(rep: dict) -> tuple[str, str]:
    c, r = rep["coverage"], rep["report"]
    cov_line = (f"V8_EVAL_COVERAGE={'PASS' if c['pass'] else 'FAIL'} policies={len(rep['policies'])} "
                f"missing={c['missing']} extra={c['extra']} duplicate={c['duplicate']} "
                f"conflicting_terminal={c['conflicting_terminal']} late_ignored={c['late_ignored']} error_final={c['error_final']}")
    rep_line = (f"V8_EVAL_REPORT={'PASS' if r['pass'] else 'FAIL'} count_mismatch={r['count_mismatch']} "
                f"media_unexplained={r['media_unexplained']} exec_over_cap={r['exec_over_cap']}")
    return cov_line, rep_line


# ---------------------------------------------------------------- 带 --dataset：逐模型判定行


def verify_dataset_videos(rep: dict, manifest_path: Path, policies: list[str], videos: Path | None,
                          dataset: str, side: str) -> dict:
    """逐模型核对本机视频根 ``<videos>/<policy>[-<variant>]/<dataset>/<side>/<key>.a<n>/``：每个清单身份的权威终态
    录像在、视频（转码后的 mp4，或 front.mkv＋wrist.mkv）都读得出帧。非 infra 错误终局无录像但写明原因计
    error_final_no_video（不算缺失）。未给 ``--videos`` 时全部计缺失并标 ``videos_root=absent``。"""
    _, mrows, _ = load_manifest(manifest_path)
    out: dict[str, Any] = {"per_policy": {}, "problems": [], "videos_root": str(videos) if videos else None}
    for pol in policies:
        dirname = parse_policy_spec(pol)[2]
        acc = rep["per_policy"][pol]["_acc"]
        res = {"expected": len(mrows), "videos": 0, "missing": 0, "decode_fail": 0, "error_final_no_video": 0,
               "no_frame_error": 0}
        for m in mrows:
            k = m["key"]
            row = acc.get(k)
            if row is None or videos is None:
                res["missing"] += 1
                out["problems"].append({"policy": pol, "key": k,
                                        "problem": "no_accepted_terminal" if row is None else "videos_root_absent"})
                continue
            ds_dirs = [videos / dn / ds / side / (rec_name(row) or "_norec") for dn, ds in layout_names(dirname, dataset)]
            d = next((x for x in ds_dirs if x.is_dir()), ds_dirs[0])
            files = sorted(f.name for f in d.iterdir() if f.is_file()) if d.is_dir() else []
            vids = media_files({"files": files}, any_mp4=True)
            if error_final_video_class(row, bool(vids)) == "explained":
                if no_frame_of(row, d if d.is_dir() else None) is False:
                    # 有帧的错误终局没有视频：不是无帧例外，照常计缺失
                    res["missing"] += 1
                    out["problems"].append({"policy": pol, "key": k, "problem": "error_with_frames_no_video",
                                            "dir": str(d)})
                    continue
                res["error_final_no_video"] += 1
                res["no_frame_error"] += 1
                continue
            if not vids:
                res["missing"] += 1
                out["problems"].append({"policy": pol, "key": k, "problem": "missing", "dir": str(d)})
                continue
            frames = {f: count_media_frames(d / f, videos) for f in vids}
            if not all(frames.values()):
                res["decode_fail"] += 1
                out["problems"].append({"policy": pol, "key": k, "problem": "decode_fail", "dir": str(d),
                                        "frames": frames})
                continue
            res["videos"] += 1
        res["pass"] = (videos is not None and res["expected"] > 0 and res["missing"] == 0
                       and res["decode_fail"] == 0 and res["videos"] + res["no_frame_error"] == res["expected"])
        res["accepted"] = res["videos"] + res["no_frame_error"]
        out["per_policy"][pol] = res
    return out


def dataset_lines(rep: dict, vid: dict) -> list[str]:
    """逐模型三行：EVAL_COVERAGE、EVAL_REPORT、EVAL_VIDEOS（--partial 时一律 FAIL 并追加 partial=1）。"""
    ds, partial = rep["dataset"], bool(rep["partial"])
    tail = " partial=1" if partial else ""
    cap_txt = "skip" if rep["cap"] is None else None
    eoc_by: Counter = Counter(x["policy"] for x in rep["exec_over_cap_detail"])
    mu_by: Counter = Counter(x["policy"] for x in rep["media_unexplained_detail"])
    out: list[str] = []
    cm_by: Counter = Counter(x["policy"] for x in rep.get("cap_mismatch_detail") or [])
    for pol, p in rep["per_policy"].items():
        c = p["coverage"]
        v = vid["per_policy"][pol]
        seed_txt = f" policy_seed={'NA' if p.get('policy_seed') is None else p['policy_seed']}"
        # 无帧 error 是用户裁决的具名例外：只有其余错误终局判 FAIL
        other_error = max(0, c["error_final"] - v["no_frame_error"])
        cov_pass = all(c[k] == 0 for k in ("missing", "extra", "duplicate", "conflicting_terminal")) \
            and other_error == 0 and not partial
        out.append(f"EVAL_COVERAGE={'PASS' if cov_pass else 'FAIL'} dataset={ds} policy={pol} expected={p['denominator']} "
                   f"missing={c['missing']} extra={c['extra']} duplicate={c['duplicate']} "
                   f"conflicting_terminal={c['conflicting_terminal']} error_final={c['error_final']} "
                   f"no_frame_error={v['no_frame_error']}{seed_txt}{tail}")
        eoc, cmm = eoc_by[pol], cm_by[pol]
        rep_pass = p["count_mismatch"] == 0 and mu_by[pol] == 0 and eoc == 0 and cmm == 0 and not partial
        out.append(f"EVAL_REPORT={'PASS' if rep_pass else 'FAIL'} dataset={ds} policy={pol} "
                   f"count_mismatch={p['count_mismatch']} dataset_crossed={p['dataset_crossed']} "
                   f"media_unexplained={mu_by[pol]} exec_over_cap={cap_txt or eoc}"
                   + ("" if cap_txt else f" cap={rep['cap']} cap_mismatch={cmm}") + f"{seed_txt}{tail}")
        vpass = v["pass"] and not partial
        out.append(f"EVAL_VIDEOS={'PASS' if vpass else 'FAIL'} dataset={ds} policy={pol} expected={v['expected']} "
                   f"accepted={v['accepted']} videos={v['videos']} no_frame_error={v['no_frame_error']} "
                   f"missing={v['missing']} decode_fail={v['decode_fail']}"
                   + ("" if vid["videos_root"] else " videos_root=absent") + f"{seed_txt}{tail}")
    return out


# ---------------------------------------------------------------- V9：合并 V8 复用

def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_reuse_stage(d: Path) -> Path:
    """V8 运行根：本身含 sNN/ 即用；否则取 V8 结果目录下的 ``nfs-records/run``（本机搬回的运行根）或 ``run``。"""
    for cand in (d, d / "nfs-records" / "run", d / "run"):
        if cand.is_dir() and seat_dirs(cand):
            return cand
    raise FileNotFoundError(f"--reuse {d} 下找不到含 sNN/ 的 V8 运行根（试过 ., nfs-records/run, run）")


def _quad(r: dict) -> tuple:
    ident = dict(r.get("identity") or {})
    return tuple(str(r.get(f, ident.get(f))) for f in ("task", "tier", "seed", "spec_sha256"))


def _rate_table(entries: list[tuple[str, str, str]], key_fn) -> dict[str, dict]:
    """entries = [(task, tier, outcome)] → {分组: {denominator, success, success_rate}}。"""
    den: Counter = Counter()
    suc: Counter = Counter()
    for task, tier, oc in entries:
        k = key_fn(task, tier)
        den[k] += 1
        suc[k] += oc == "success"
    return {k: {"denominator": den[k], "success": suc[k], "success_rate": suc[k] / den[k] if den[k] else None}
            for k in sorted(den)}


def build_reuse(rep: dict, manifest_path: Path, reuse_dir: Path, reuse_manifest: Path, policies: list[str],
                *, expect_reused: int, hs=None) -> dict:
    """读 reused.json（唯一复用依据）→ 对齐 V8 manifest → 取 V8 账本 accepted 终态 → 800 局总表。只读 V8。

    800 局总表另与交付格表 ``hard_specs.V9_CELLS`` 逐格比分母（每个模型各比一次）：缺格、多格或某格局数不等都记
    count_mismatch。``hs`` 缺省时按文件路径加载子模块 ``third_party/robomme_benchmark/src/robomme_ood/env_record_wrapper/hard_specs.py``（只依赖标准库，
    不 import robomme_ood 包、不触发 sapien）。"""
    mismatch: list[str] = []
    if hs is None:
        import importlib.util

        name = "_v8_report_hard_specs"
        hs = sys.modules.get(name)
        if hs is None:
            spec = importlib.util.spec_from_file_location(name, hard_specs_path())
            hs = importlib.util.module_from_spec(spec)
            sys.modules[name] = hs
            spec.loader.exec_module(hs)
    want_cells = {f"{t}@{tier}": int(n) for (t, tier), n in hs.V9_CELLS.items()}
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, new_rows, _ = load_manifest(manifest_path)
    meta = doc.get("reused") or {}
    rpath = manifest_path.parent / str(meta.get("path") or "reused.json")
    out: dict[str, Any] = {"reused_path": str(rpath), "reuse_dir": str(reuse_dir), "reuse_manifest": str(reuse_manifest)}
    if not meta:
        mismatch.append("manifest 无 reused 键（不是 --exclude-evaluated 产出的清单）")
    try:
        raw = rpath.read_bytes()
        rdoc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        mismatch.append(f"reused.json 读不出：{type(e).__name__}: {e}")
        raw, rdoc = b"", {"rows": []}
    out["reused_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
    if meta and out["reused_sha256"] != meta.get("sha256"):
        mismatch.append(f"reused.json sha256={out['reused_sha256'][:12]} != manifest.reused.sha256="
                        f"{str(meta.get('sha256'))[:12]}")
    if rdoc.get("schema") != REUSED_SCHEMA:
        mismatch.append(f"reused.json schema={rdoc.get('schema')!r}")
    rrows = list(rdoc.get("rows") or [])
    if not (rdoc.get("count") == len(rrows) == expect_reused) or (meta and meta.get("count") != len(rrows)):
        mismatch.append(f"复用行数 rows={len(rrows)} count={rdoc.get('count')} manifest={meta.get('count')} "
                        f"expect={expect_reused}")
    v8_sha = file_sha256(reuse_manifest)
    out["reuse_manifest_sha256"] = v8_sha
    if rdoc.get("v8_manifest_sha256") != v8_sha:
        mismatch.append(f"reused.json v8_manifest_sha256 与 --reuse-manifest 现算 {v8_sha[:12]} 不符")
    v8doc = json.loads(reuse_manifest.read_text(encoding="utf-8"))
    v8_by_key = {key_of(r): r for r in v8doc.get("rows") or []}
    rquads = Counter(_quad(r) for r in rrows)
    mismatch += [f"复用四元组重复 {q}" for q, n in rquads.items() if n > 1]
    rkeys = Counter(str(r.get("v8_key")) for r in rrows)
    mismatch += [f"复用 v8_key 重复 {k}" for k, n in rkeys.items() if n > 1]
    new_quads = {_quad(r) for r in new_rows}
    mismatch += [f"复用行与新评行四元组重叠 {q}" for q in rquads if q in new_quads]
    for r in rrows:
        m = v8_by_key.get(str(r.get("v8_key")))
        if m is None:
            mismatch.append(f"复用 {r.get('v8_key')} 不在 V8 manifest")
        elif _quad(m) != _quad(r):
            mismatch.append(f"复用 {r.get('v8_key')} 四元组与 V8 manifest 不符 reused={_quad(r)} v8={_quad(m)}")
    try:
        v8_stage = resolve_reuse_stage(reuse_dir)
    except FileNotFoundError as e:
        mismatch.append(str(e))
        v8_stage = None
    out["v8_stage"] = str(v8_stage) if v8_stage else None
    totals: dict[str, Any] = {}
    for pol in policies:
        acc = analyze_attempts(load_policy(v8_stage, pol))["accepted"] if v8_stage else {}
        reused_entries: list[tuple[str, str, str]] = []
        for r in rrows:
            k = str(r.get("v8_key"))
            row = acc.get(k)
            if row is None:
                mismatch.append(f"{pol} 复用 {k} V8 账本无 accepted 终态")
                oc = "missing"
            else:
                oc = str(row.get("status"))
                if _quad(row) != _quad(r):
                    mismatch.append(f"{pol} 复用 {k} V8 结果行四元组 {_quad(row)} 与 reused.json {_quad(r)} 不符")
            reused_entries.append((str(r.get("task")), str(r.get("tier")), oc))
        outcome = rep["per_policy"][pol]["_outcome"]
        new_entries = [(str(r["task"]), str(r["tier"]), outcome[r["key"]]) for r in new_rows]
        allv = new_entries + reused_entries
        oc_all = Counter(o for *_, o in allv)
        by_task = _rate_table(allv, lambda t, _tier: t)
        totals[pol] = {
            "denominator": len(allv), "new": len(new_entries), "reused": len(reused_entries),
            "outcomes": {s: oc_all[s] for s in (*TERMINAL, "error", "missing", "conflict")},
            "success": oc_all["success"],
            "micro_success_rate": oc_all["success"] / len(allv) if allv else None,
            "task_macro_success_rate": (statistics.fmean(v["success_rate"] for v in by_task.values())
                                        if by_task else None),
            "new_success": sum(o == "success" for *_, o in new_entries),
            "reused_success": sum(o == "success" for *_, o in reused_entries),
            "tasks": by_task,
            "tiers": _rate_table(allv, lambda _t, tier: tier),
            "cells": _rate_table(allv, lambda t, tier: f"{t}@{tier}"),
        }
        # 800 局总表逐格对 V9_CELLS：只核「新评 + 复用」总数挡不住格间此消彼长
        have_cells = {k: v["denominator"] for k, v in totals[pol]["cells"].items()}
        for cell in sorted(set(want_cells) | set(have_cells)):
            if have_cells.get(cell, 0) != want_cells.get(cell, 0):
                mismatch.append(f"{pol} 总表格 {cell} 局数={have_cells.get(cell, 0)} != V9_CELLS={want_cells.get(cell, 0)}")
    out.update({"new": len(new_rows), "reused": len(rrows), "total": len(new_rows) + len(rrows),
                "count_mismatch_detail": mismatch, "totals": totals})
    return out


_MOVER = None
_DECODE: dict[str, Any] = {}


def video_mover_mod():
    """只读复用同目录 ``eval_video_mover.py`` 的解码器选择与帧数缓存（V8 视频核对逻辑）。"""
    global _MOVER
    if _MOVER is None:
        import importlib.util

        path = Path(__file__).resolve().parent / "eval_video_mover.py"
        spec = importlib.util.spec_from_file_location("eval_video_mover_for_v9_report", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOVER = mod
    return _MOVER


def count_media_frames(path: Path, videos: Path) -> int | None:
    """front.mkv／wrist.mkv 帧数（读不出为 None）；缓存写 ``<videos>/decode-cache.jsonl``，与搬运脚本共用。"""
    mv = video_mover_mod()
    if "decoder" not in _DECODE:
        _DECODE["decoder"] = mv.pick_decoder()
    if _DECODE.get("root") != str(videos):
        _DECODE["root"], _DECODE["cache"] = str(videos), mv.DecodeCache(videos / "decode-cache.jsonl")
    if _DECODE["decoder"][0] == "none":
        return None
    return _DECODE["cache"].frames(path, _DECODE["decoder"])


def verify_new_videos(rep: dict, manifest_path: Path, policies: list[str], videos: Path | None) -> dict:
    """只核新评身份：expected = 模型数 × 新评行数；accepted 终态录像在本机、两路都读得出帧、sha256 与 moved.jsonl 一致。"""
    _, new_rows, _ = load_manifest(manifest_path)
    expected = len(policies) * len(new_rows)
    res = {"expected": expected, "videos": 0, "missing": 0, "decode_fail": 0, "sha_mismatch": 0,
           "error_final_no_video": 0, "moved_record_absent": 0, "problems": []}
    moved: dict[tuple, dict] = {}
    if videos is not None:
        for row in read_jsonl(videos / "moved.jsonl"):
            if row.get("mode") == "v8" and row.get("dest") and isinstance(row.get("files"), dict):
                moved[(str(row.get("policy")), Path(str(row["dest"])).name, str(row.get("tier")),
                       str(row.get("task")))] = row["files"]
    for pol in policies:
        acc = rep["per_policy"][pol]["_acc"]
        for m in new_rows:
            k = m["key"]
            row = acc.get(k)
            if row is None:
                res["missing"] += 1
                res["problems"].append({"policy": pol, "key": k, "problem": "no_accepted_terminal"})
                continue
            name = rec_name(row) or "_norec"
            d = (videos / pol / str(row.get("tier") or "_notier") / str(row.get("task")) / name) if videos else None
            has = d is not None and d.is_dir() and all((d / f).is_file() for f in MEDIA_FILES)
            if error_final_video_class(row, has) == "explained":
                res["error_final_no_video"] += 1
                continue
            if not has:
                res["missing"] += 1
                res["problems"].append({"policy": pol, "key": k, "problem": "missing", "dir": str(d)})
                continue
            frames = {f: count_media_frames(d / f, videos) for f in MEDIA_FILES}
            if not all(frames.values()):
                res["decode_fail"] += 1
                res["problems"].append({"policy": pol, "key": k, "problem": "decode_fail", "dir": str(d),
                                        "frames": frames})
                continue
            rec = moved.get((pol, name, str(row.get("tier")), str(row.get("task"))))
            if rec is None:  # 无搬运记录：sha 无从核对，不计入 videos，判定行 moved_record_absent>0 即 FAIL
                res["moved_record_absent"] += 1
                res["problems"].append({"policy": pol, "key": k, "problem": "moved_record_absent", "dir": str(d)})
                continue
            bad = [f for f, h in rec.items() if (d / f).is_file() and file_sha256(d / f) != h]
            if bad:
                res["sha_mismatch"] += 1
                res["problems"].append({"policy": pol, "key": k, "problem": "sha_mismatch", "files": bad})
                continue
            res["videos"] += 1
    res["pass"] = (expected > 0 and res["missing"] == 0 and res["decode_fail"] == 0 and res["sha_mismatch"] == 0
                   and res["moved_record_absent"] == 0
                   and res["videos"] + res["error_final_no_video"] == expected)
    return res


def v9_lines(rep: dict, v9: dict, vid: dict, *, expect_new: int, expect_reused: int) -> tuple[str, str, str]:
    c, r = rep["coverage"], rep["report"]
    partial = bool(rep["partial"])
    # 严格口径：缺失按结局计数（--partial 下 coverage.missing 记 0，这里不采用）；--partial 下三行一律 FAIL 并标 partial=1
    missing = sum(p["outcomes"]["missing"] for p in rep["per_policy"].values()) if partial else c["missing"]
    tail = " partial=1" if partial else ""
    cov_pass = c["pass"] and c["error_final"] == 0 and missing == 0 and not partial
    cov_line = (f"V9_EVAL_COVERAGE={'PASS' if cov_pass else 'FAIL'} policies={len(rep['policies'])} "
                f"expected={rep['v9']['new']} missing={missing} extra={c['extra']} duplicate={c['duplicate']} "
                f"conflicting_terminal={c['conflicting_terminal']} error_final={c['error_final']}{tail}")
    cm = r["count_mismatch"] + len(v9["count_mismatch_detail"])
    rep_pass = (cm == 0 and r["media_unexplained"] == 0 and r["exec_over_cap"] == 0 and not partial
                and v9["new"] == expect_new and v9["reused"] == expect_reused)
    rep_line = (f"V9_EVAL_REPORT={'PASS' if rep_pass else 'FAIL'} total={v9['total']} new={v9['new']} "
                f"reused={v9['reused']} count_mismatch={cm} media_unexplained={r['media_unexplained']}"
                + (f" exec_over_cap={r['exec_over_cap']}" if r["exec_over_cap"] else "") + tail)
    vid_line = (f"V9_EVAL_VIDEOS={'PASS' if vid['pass'] else 'FAIL'} policies={len(rep['policies'])} "
                f"expected={vid['expected']} videos={vid['videos']} missing={vid['missing']} "
                f"decode_fail={vid['decode_fail']} sha_mismatch={vid['sha_mismatch']} "
                f"moved_record_absent={vid['moved_record_absent']}"
                + (f" error_final_no_video={vid['error_final_no_video']}" if vid["error_final_no_video"] else "")
                + tail)
    if partial:
        vid_line = vid_line.replace("V9_EVAL_VIDEOS=PASS", "V9_EVAL_VIDEOS=FAIL", 1)
    return cov_line, rep_line, vid_line


def render_v9_md(rep: dict, lines: tuple[str, ...]) -> str:
    v9 = rep["v9"]
    pols = list(v9["totals"])
    L = ["# V9 双模型评估汇总" + ("（中途进度）" if rep["partial"] else ""), ""]
    L.append(f"- 新评 {v9['new']} 局（本次运行）+ 复用 {v9['reused']} 局（V8 评估，终态取 V8 账本 `accepted_attempt_id`）"
             f" = 总表 {v9['total']} 局。复用集合唯一依据：`{v9['reused_path']}`（sha256 `{v9['reused_sha256']}`）；"
             f"V8 运行根 `{v9['v8_stage']}`；V8 manifest `{v9['reuse_manifest']}`。")
    L.append(f"- 注意（R-7）：复用的 {v9['reused']} 局是 V8 2026-10-02 跑的结果，新评 {v9['new']} 局是之后跑的；权重、tokenizer、"
             "客户端钉在同一锁值，但 MME-VLA 同入口重跑本有翻转，总表新旧两部分不是同一时刻的采样。")
    L += ["", "```", *lines, "```", "", "## 800 局总表", ""]
    L.append("| 模型 | 分母 | 成功 | 失败 | timeout | error | 缺失 | 冲突 | 全局微平均 | 任务宏平均 | 新评成功 | 复用成功 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pol, t in v9["totals"].items():
        o = t["outcomes"]
        L.append(f"| {pol} | {t['denominator']} | {o['success']} | {o['fail']} | {o['timeout']} | {o['error']} | "
                 f"{o['missing']} | {o['conflict']} | {pct(t['micro_success_rate'])} | {pct(t['task_macro_success_rate'])} | "
                 f"{t['new_success']}/{t['new']} | {t['reused_success']}/{t['reused']} |")
    for title, field in (("按任务（跨档，800 局）", "tasks"), ("按档（跨任务，800 局）", "tiers"), ("按格（800 局）", "cells")):
        L += ["", f"## {title}", "", "| 分组 | " + " | ".join(pols) + " |", "|---|" + "---:|" * len(pols)]
        groups = sorted(next(iter(v9["totals"].values()))[field]) if pols else []
        for g in groups:
            L.append(f"| {g} | " + " | ".join(
                f"{v9['totals'][p][field][g]['success']}/{v9['totals'][p][field][g]['denominator']} "
                f"{pct(v9['totals'][p][field][g]['success_rate'])}" for p in pols) + " |")
    if v9["count_mismatch_detail"]:
        L += ["", "## 复用对齐不符", ""] + [f"- {x}" for x in v9["count_mismatch_detail"][:200]]
    vid = rep["v9_videos"]
    L += ["", "## 新评视频核对", "",
          f"期望 {vid['expected']}，通过 {vid['videos']}，缺失 {vid['missing']}，解码失败 {vid['decode_fail']}，"
          f"sha 不符 {vid['sha_mismatch']}，错误终局无录像（写明原因）{vid['error_final_no_video']}，"
          f"moved.jsonl 无记录 {vid['moved_record_absent']}。"]
    L += ["", "---", "", "以下为新评身份分表（V8 口径，分母为新评行数）。", ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--stage", help="运行根（含 sNN/<policy>/）")
    ap.add_argument("--results", nargs="+", help="显式 results.jsonl 路径；只产出速度表与守恒判定")
    ap.add_argument("--policies", default="smvla,perceptual-framesamp-modul",
                    help="逗号分隔的 <policy>[:<variant>]（如 perceptual-framesamp-modul,groundsg:ground-sg-oracle,pp）；运行根目录 sNN/<policy>[-<variant>]/")
    ap.add_argument("--dataset", default=None, choices=list(DATASETS),
                    help="逐模型出 EVAL_COVERAGE／EVAL_REPORT／EVAL_VIDEOS；hard-verify 身份必备字段不含 spec_sha256、"
                         "不做 exec_over_cap；须同给 --expect-total（800 或 192 等，由调用方给）")
    ap.add_argument("--side", default="new", choices=["new", "orig"],
                    help="带 --dataset 时本机视频根下的侧别目录（<videos>/<policy>[-<variant>]/<dataset>/<side>/）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default=None, help="本机视频根（eval_video_mover --mode v8 的 --dest）")
    ap.add_argument("--partial", action="store_true", help="中途进度：不因缺失／媒体未就位判 FAIL")
    ap.add_argument("--expect-total", type=int, default=None,
                    help=f"manifest 身份数（默认 {DEFAULT_TOTAL}；带 --reuse 时默认新评 {V9_DEFAULT_NEW}）")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--policy-seed", type=int, default=None,
                    help="带 --dataset：模型种子；每条结果行 policy_seed 必须等于它，判定行带 policy_seed=")
    ap.add_argument("--shard-files", default=None,
                    help="--partial 估算用：实际分片文件 glob（shard-NN.json → 席 NN），覆盖 manifest 的 shard 字段；"
                         "已观测到的身份一律以运行根里实际所在 sNN 为准")
    ap.add_argument("--reuse", default=None, help="V9：V8 结果目录或运行根（只读，终态取 V8 账本 accepted_attempt_id）")
    ap.add_argument("--reuse-manifest", default=None, help="V9：V8 manifest.json（与 reused.json 的 v8_manifest_sha256 核对）")
    ap.add_argument("--expect-reused", type=int, default=V9_DEFAULT_REUSED, help="V9：复用行数（默认 720）")
    args = ap.parse_args(argv)
    if args.results:
        rep = speed_table_from_results([Path(p) for p in args.results])
        line = speed_line(rep["speed"])
        rep["verdict_lines"] = [line]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out / "report.md").write_text("\n".join(_render_speed(rep["speed"]) + [f"`{line}`", ""]), encoding="utf-8")
        print(line, flush=True)
        return 0
    if not args.manifest or not args.stage:
        ap.error("非 --results 入口必须提供 --manifest 与 --stage")
    old = [p for p in args.policies.split(",") if p and official_defs().canonical_policy(p.partition(":")[0]) != p.partition(":")[0]]
    if old:
        ap.error(f"--policies 只接受官方名，收到旧名 {old}")
    v9 = args.reuse is not None
    if v9 != (args.reuse_manifest is not None):
        ap.error("--reuse 与 --reuse-manifest 须同时给出")
    if args.dataset is not None:
        if v9:
            ap.error("--dataset 不与 --reuse 同用")
        if args.expect_total is None:
            ap.error("--dataset 必须同给 --expect-total（期望身份数由调用方给出）")
        return main_dataset(args)
    expect_total = args.expect_total if args.expect_total is not None else (V9_DEFAULT_NEW if v9 else DEFAULT_TOTAL)
    policies = [p for p in args.policies.split(",") if p]
    videos = Path(args.videos) if args.videos else None
    rep = build_report(Path(args.manifest), Path(args.stage), policies, videos,
                       partial=args.partial, expect_total=expect_total, cap=args.cap,
                       shard_files=args.shard_files, keep_internal=v9)
    v9_out = None
    if v9:
        rep["v9"] = build_reuse(rep, Path(args.manifest), Path(args.reuse), Path(args.reuse_manifest), policies,
                                expect_reused=args.expect_reused)
        vid = verify_new_videos(rep, Path(args.manifest), policies, videos)
        rep["v9_videos"] = vid
        for p in rep["per_policy"].values():
            p.pop("_acc")
            p.pop("_outcome")
        v9_out = v9_lines(rep, rep["v9"], vid, expect_new=expect_total, expect_reused=args.expect_reused)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = rep.pop("_index")
    cov_line, rep_line = lines_of(rep)
    speed_verdict = speed_line(rep["speed"])
    rep["verdict_lines"] = [cov_line, rep_line, speed_verdict]
    if v9_out:
        rep["v9_verdict_lines"] = list(v9_out)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    with (out / "video-index.jsonl").open("w", encoding="utf-8") as fh:
        for row in index:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    md = render_md(rep, cov_line, rep_line, speed_verdict)
    if v9_out:
        md = render_v9_md(rep, v9_out) + md.replace("# V8 双模型评估汇总", "## 新评身份分表（V8 口径）", 1)
    (out / "report.md").write_text(md, encoding="utf-8")
    if rep.get("progress"):
        pr = rep["progress"]
        done = " ".join(f"{p}={v['done']}/{v['denominator']}" for p, v in pr["policies"].items())
        print(f"V8_EVAL_PROGRESS {done} est_remaining_s="
              f"{'unobserved' if pr['est_remaining_s'] is None else round(pr['est_remaining_s'])} "
              f"slowest_seat={pr['slowest_seat']}", flush=True)
    if v9_out:
        for line in v9_out:
            print(line, flush=True)
        print(speed_verdict, flush=True)
        return 0 if all(line.split()[0].endswith("=PASS") for line in v9_out) else 1
    print(cov_line, flush=True)
    print(rep_line, flush=True)
    print(speed_verdict, flush=True)
    return 0 if rep["coverage"]["pass"] and rep["report"]["pass"] else 1


def main_dataset(args) -> int:
    """带 --dataset：逐模型三行判定（EVAL_COVERAGE／EVAL_REPORT／EVAL_VIDEOS），全部 PASS 退出 0，否则 1。"""
    policies = [p for p in args.policies.split(",") if p]
    videos = Path(args.videos) if args.videos else None
    cap = None if args.dataset == "hard-verify" else args.cap
    rep = build_report(Path(args.manifest), Path(args.stage), policies, videos, partial=args.partial,
                       expect_total=args.expect_total, cap=cap, shard_files=args.shard_files, keep_internal=True,
                       dataset=args.dataset, side=args.side, policy_seed=args.policy_seed)
    vid = verify_dataset_videos(rep, Path(args.manifest), policies, videos, args.dataset, args.side)
    rep["dataset_videos"] = vid
    for p in rep["per_policy"].values():
        p.pop("_acc")
        p.pop("_outcome")
    lines = dataset_lines(rep, vid)
    speed_verdict = speed_line(rep["speed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = rep.pop("_index")
    rep["verdict_lines"] = [*lines, speed_verdict]
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    with (out / "video-index.jsonl").open("w", encoding="utf-8") as fh:
        for row in index:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "report.md").write_text(render_md(rep, *lines, speed_verdict), encoding="utf-8")
    if rep.get("progress"):
        pr = rep["progress"]
        done = " ".join(f"{p}={v['done']}/{v['denominator']}" for p, v in pr["policies"].items())
        print(f"EVAL_PROGRESS dataset={args.dataset} {done} est_remaining_s="
              f"{'unobserved' if pr['est_remaining_s'] is None else round(pr['est_remaining_s'])} "
              f"slowest_seat={pr['slowest_seat']}", flush=True)
    for line in lines:
        print(line, flush=True)
    print(speed_verdict, flush=True)
    return 0 if all(line.split()[0].endswith("=PASS") for line in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
