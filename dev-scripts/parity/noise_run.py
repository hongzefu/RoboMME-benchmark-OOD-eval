#!/usr/bin/env python3
"""噪声基线 GL 单遍运行包装的 Python 逻辑（1003-noise-baseline-plan.md 第一部分 3.3，第二部分 §一、§二 S3、§四、§五）。

每一遍生成都经同目录 ``noise_run_gl.sh`` 启动，壳脚本依次调本文件的 ``preflight`` → 真实命令 → ``finish``。
噪声工具只留生成这一条线（1003 代码测试维护计划「细则 2.5」：评估分片 ``eval-shard`` 已删，``--kind`` 只认 ``gen``；
账本里历史的 ``eval``／``digest`` 行仍照常累计）。三个子命令：

``preflight``  起跑前检查，任一不过即非零退出、不执行真实命令：

1. 输出根必须不存在或为空目录，否则 ``RUN_FRESH=FAIL reason=out_root_not_empty``（审查意见 6：复用同一运行根时
   评估客户端会跳过已接受的身份，旧结果会被当成新一遍）。本命令不创建输出根，由真实命令自己建。
2. 资产全量核对：每个 ``--asset-dir 名=路径`` 按建锁脚本的过滤规则
   ``find -L . -type f ! -path './.cache/*' ! -name .DS_Store``（跟随符号链接、排除顶层 ``.cache/`` 子树与任意层的
   ``.DS_Store``）逐文件 sha256，与资产锁 ``assets.<名>.files``（``./相对路径 -> sha256``）比，每资产一行
   ``ASSET name= files= locked= missing= extra= mismatch=``；tokenizer 单独核 sha256。全零才 ``assets=PASS``，
   一项都没给为 ``assets=SKIP``。
3. 预算：追加式 jsonl 账本，``fcntl.flock`` 加锁后读全部历史，按 kind 累加（每遍取「登记数」与「实际数」的较大者，
   实际数未知时按登记数计），加上本遍申请后任一项超过 ``--budget-caps`` 的 kind 上限或 ``total`` 上限即拒跑
   ``BUDGET=FAIL``；同名 pass 已登记也拒。通过后追加一行
   ``{"t","pass","kind","attempts","resets","retries","event":"reserve"}``，写后 flush + ``os.fsync``。
   资产核对放在预算登记之前：资产不过就不占预算。
4. 环境指纹：主仓库 commit 与 ``git status --porcelain`` 是否为空、各 ``--policy-repo`` 的 commit、
   ``nvidia-smi --query-gpu=name,driver_version,uuid``（可执行文件可由环境变量 ``NOISE_RUN_NVIDIA_SMI`` 指定，
   不可用时字段为 null）、hostname、``SLURM_JOB_ID``、python 版本。
5. 写来源报告 json（``--provenance-out``，已存在即拒，不覆盖），``fingerprint`` = 剔除 ``fingerprint`` 与
   finish 追加的键（``FINISH_KEYS``）后 canonical JSON 的 sha256——finish 追加字段后 fingerprint 不变、可重算；
   另有 ``env_fingerprint`` 只覆盖「换节点也应相同」的环境字段（commit、脏否、策略 commit、资产清单 sha、tokenizer、
   驱动、GPU 型号、步数上限、server 参数），供「已有正式评估那一遍能否并入」的指纹核对用。

   末行 ``NOISE_PREFLIGHT=PASS pass=<名> kind=<k> assets=<PASS|SKIP> fingerprint=<前12位>``。

``finish``  真实命令结束后：账本追加 ``event":"finish"`` 行（实际数，``unknown`` 记 null），来源报告原子改写追加
``finished_at``、``finished_at_iso``、``exit_code``、``actual_attempts``、``actual_resets``、``actual_retries``。
实际数超过登记数时打印 ``BUDGET=FAIL reason=actual_exceeds_reserved`` 并以 6 退出（记录照写）。

``ship``  ``hard_parity.py generate --stage`` 已经逐局把 h5／mp4 复制到暂存目录、核 h5 sha、写 ``SHIPPED`` 并删节点
大文件；本子命令只做完整性核对：节点输出根 ``identities.jsonl`` 里每条有 ``path`` 的局在暂存目录都有 ``SHIPPED``，
其中 h5 的 sha 与 identities 行相等，未被 ``hard_pull.py`` 拉走（无 ``PULLED``）的局逐文件重算 sha 与 ``SHIPPED``
相等，节点上不残留 h5／mp4。带 ``--expect-ref`` 生成时 ``Mover`` 对 ``verdict=match`` 的局（与噪声基线逐字节相同）
按设计不复制、直接删节点大文件，这类局不要求 ``SHIPPED``，计入 ``matched``（对拍细则 3.4）。``--finalize`` 时核对通过后把节点输出根的小文件（非 h5／mp4）复制到暂存目录并写
``SEGMENT_DONE``（与 v6 stage4 runbook 的 ``rsync --exclude`` + ``touch`` 等价），节点目录不删、由调用方显式删。
末行 ``NOISE_SHIP=PASS|FAIL episodes= shipped= matched= pulled= sha_bad= missing= node_leftover_media=``。

只用标准库。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: preflight 只接受生成遍（细则 2.5）
KINDS = ("gen",)
#: 账本里可能出现的历史 kind（噪声基线阶段登记过评估与环境摘要遍），累计预算时照常计入
LEDGER_KINDS = ("gen", "eval", "digest")
BUDGET_FIELDS = ("attempts", "resets", "retries")
#: finish 追加进来源报告的键；fingerprint 计算时剔除，保证 finish 前后 fingerprint 一致
FINISH_KEYS = ("finished_at", "finished_at_iso", "exit_code", "actual_attempts", "actual_resets", "actual_retries")
#: env_fingerprint 覆盖的字段：换节点、换遍也应相同的环境事实（不含遍名、时间、主机、GPU UUID、输出根）
ENV_KEYS = ("commit", "git_dirty", "policy_commits", "assets_sha", "tokenizer_sha256", "driver", "gpu_model",
            "max_steps", "server_args")
MEDIA_SUFFIXES = (".h5", ".mp4")

EXIT_FRESH = 4
EXIT_BUDGET = 5
EXIT_OVERRUN = 6
EXIT_ASSETS = 7
EXIT_USAGE = 2


class RunError(RuntimeError):
    """带退出码的失败；main 打印说明后以该码退出。"""

    def __init__(self, line: str, code: int):
        super().__init__(line)
        self.line = line
        self.code = code


# ── 通用小工具 ────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(doc: Any) -> str:
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint_of(report: dict) -> str:
    """来源报告指纹：剔除 ``fingerprint`` 与 ``FINISH_KEYS`` 后 canonical JSON 的 sha256。"""
    body = {k: v for k, v in report.items() if k != "fingerprint" and k not in FINISH_KEYS}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def env_fingerprint_of(report: dict) -> str:
    body = {k: report.get(k) for k in ENV_KEYS}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, doc: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def parse_pairs(items: list[str] | None, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise RunError(f"NOISE_PREFLIGHT=FAIL reason=bad_arg flag={flag} value={item!r}（须为 名=路径）", EXIT_USAGE)
        if name in out:
            raise RunError(f"NOISE_PREFLIGHT=FAIL reason=bad_arg flag={flag} duplicate_name={name}", EXIT_USAGE)
        out[name] = value
    return out


def run_text(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


# ── 输出根 ────────────────────────────────────────────────────────────────────


def check_out_root(out_root: Path) -> str:
    """不存在返回 absent，空目录返回 empty_dir，其余一律 RUN_FRESH=FAIL。"""
    if not out_root.exists() and not out_root.is_symlink():
        return "absent"
    if out_root.is_dir():
        if any(out_root.iterdir()):
            raise RunError(f"RUN_FRESH=FAIL reason=out_root_not_empty out_root={out_root}", EXIT_FRESH)
        return "empty_dir"
    raise RunError(f"RUN_FRESH=FAIL reason=out_root_not_empty out_root={out_root} detail=not_a_directory", EXIT_FRESH)


# ── 资产核对 ──────────────────────────────────────────────────────────────────


def list_asset_files(root: Path) -> dict[str, Path]:
    """等价于 ``cd root && find -L . -type f ! -path './.cache/*' ! -name .DS_Store``：返回 ``./相对路径 -> 实际路径``。

    跟随符号链接（目录与文件都跟随），断链不算文件；顶层 ``.cache`` 整棵子树排除（``-path './.cache/*'`` 的 ``*``
    可跨 ``/``），更深层的 ``.cache`` 不排除；任意层名为 ``.DS_Store`` 的文件排除。目录符号链接成环时同一真实目录只走一次。
    """
    out: dict[str, Path] = {}
    seen_dirs: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(real)
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            dirnames[:] = [d for d in dirnames if d != ".cache"]
        dirnames.sort()
        for name in sorted(filenames):
            if name == ".DS_Store":
                continue
            path = Path(dirpath) / name
            if not path.is_file():  # 断链或特殊文件：find -L -type f 不列
                continue
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            out["./" + rel.replace(os.sep, "/")] = path
    return out


def files_list_sha(files: dict[str, str]) -> str:
    """清单 sha：按相对路径字节序排序后逐行 ``<sha256>  <./相对路径>\\n`` 的 sha256（本地与锁两侧同一规则）。"""
    text = "".join(f"{files[k]}  {k}\n" for k in sorted(files))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_asset(name: str, root: Path, locked: dict[str, str]) -> dict[str, Any]:
    if not root.is_dir():
        return {"name": name, "root": str(root), "files": 0, "locked": len(locked), "missing": len(locked),
                "extra": 0, "mismatch": 0, "ok": False, "error": "root_not_dir", "files_sha256": None,
                "locked_files_sha256": files_list_sha(locked)}
    found = {rel: sha256_file(path) for rel, path in list_asset_files(root).items()}
    missing = sorted(set(locked) - set(found))
    extra = sorted(set(found) - set(locked))
    mismatch = sorted(k for k in set(found) & set(locked) if found[k] != locked[k])
    return {"name": name, "root": str(root), "files": len(found), "locked": len(locked), "missing": len(missing),
            "extra": len(extra), "mismatch": len(mismatch), "ok": not (missing or extra or mismatch),
            "files_sha256": files_list_sha(found), "locked_files_sha256": files_list_sha(locked),
            "detail": {"missing": missing[:10], "extra": extra[:10], "mismatch": mismatch[:10]}}


def check_assets(args) -> tuple[str, dict[str, Any], str | None]:
    """返回 (判定 PASS|SKIP|FAIL, 每资产结果, tokenizer 实测 sha)。"""
    dirs = parse_pairs(args.asset_dir, "--asset-dir")
    if dirs and not args.assets_lock:
        raise RunError("NOISE_PREFLIGHT=FAIL reason=bad_arg detail=--asset-dir 须同时给 --assets-lock", EXIT_USAGE)
    if bool(args.tokenizer) != bool(args.tokenizer_sha256):
        raise RunError("NOISE_PREFLIGHT=FAIL reason=bad_arg detail=--tokenizer 与 --tokenizer-sha256 须同时给", EXIT_USAGE)
    results: dict[str, Any] = {}
    ok = True
    if dirs:
        lock = json.loads(Path(args.assets_lock).read_text(encoding="utf-8"))
        assets = lock.get("assets") or {}
        for name, root in dirs.items():
            entry = assets.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("files"), dict):
                res = {"name": name, "root": root, "files": 0, "locked": 0, "missing": 0, "extra": 0, "mismatch": 0,
                       "ok": False, "error": "not_in_lock", "files_sha256": None, "locked_files_sha256": None}
            else:
                res = check_asset(name, Path(root), entry["files"])
            results[name] = res
            ok &= res["ok"]
            print(f"ASSET name={name} files={res['files']} locked={res['locked']} missing={res['missing']} "
                  f"extra={res['extra']} mismatch={res['mismatch']}"
                  + (f" error={res['error']}" if res.get("error") else ""), flush=True)
    tok_sha = None
    if args.tokenizer:
        tok = Path(args.tokenizer)
        tok_sha = sha256_file(tok) if tok.is_file() else None
        match = tok_sha is not None and tok_sha == args.tokenizer_sha256.lower()
        ok &= match
        print(f"TOKENIZER path={tok} sha256={tok_sha} expected={args.tokenizer_sha256} match={int(match)}", flush=True)
    if not dirs and not args.tokenizer:
        return "SKIP", results, tok_sha
    return ("PASS" if ok else "FAIL"), results, tok_sha


# ── 预算账本 ──────────────────────────────────────────────────────────────────


def read_ledger_locked(stream) -> list[dict]:
    stream.seek(0)
    rows = []
    for i, line in enumerate(stream.read().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunError(f"BUDGET=FAIL reason=ledger_corrupt line={i} detail={exc}", EXIT_BUDGET) from exc
        if not isinstance(row, dict) or row.get("event") not in ("reserve", "finish"):
            raise RunError(f"BUDGET=FAIL reason=ledger_corrupt line={i} detail=bad_event", EXIT_BUDGET)
        rows.append(row)
    return rows


def append_line_locked(stream, row: dict) -> None:
    stream.seek(0, os.SEEK_END)
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def charged_by_pass(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """每遍的计费量：登记数与实际数（若已知且更大）取较大者。"""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["event"] == "reserve":
            out[row["pass"]] = {"kind": row["kind"], **{f: int(row[f]) for f in BUDGET_FIELDS}}
    for row in rows:
        if row["event"] == "finish" and row.get("pass") in out:
            cur = out[row["pass"]]
            for f in BUDGET_FIELDS:
                v = row.get(f)
                if isinstance(v, int) and not isinstance(v, bool) and v > cur[f]:
                    cur[f] = v
    return out


def totals(charged: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    tot = {k: dict.fromkeys(BUDGET_FIELDS, 0) for k in (*LEDGER_KINDS, "total")}
    for c in charged.values():
        bucket = tot.setdefault(c["kind"], dict.fromkeys(BUDGET_FIELDS, 0))
        for f in BUDGET_FIELDS:
            bucket[f] += c[f]
            tot["total"][f] += c[f]
    return tot


def reserve_budget(ledger: Path, caps: dict, pass_name: str, kind: str, req: dict[str, int]) -> dict[str, Any]:
    for scope in (kind, "total"):
        cap = caps.get(scope)
        if not isinstance(cap, dict) or any(not isinstance(cap.get(f), int) for f in BUDGET_FIELDS):
            raise RunError(f"BUDGET=FAIL reason=cap_missing scope={scope}", EXIT_BUDGET)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            rows = read_ledger_locked(stream)
            if any(r.get("pass") == pass_name and r["event"] == "reserve" for r in rows):
                raise RunError(f"BUDGET=FAIL reason=duplicate_pass pass={pass_name}", EXIT_BUDGET)
            before = totals(charged_by_pass(rows))
            after = {s: dict(v) for s, v in before.items()}
            for f in BUDGET_FIELDS:
                after[kind][f] += req[f]
                after["total"][f] += req[f]
            over = [f"{s}.{f}={after[s][f]}>{caps[s][f]}" for s in (kind, "total") for f in BUDGET_FIELDS
                    if after[s][f] > caps[s][f]]
            if over:
                raise RunError(f"BUDGET=FAIL reason=over_cap pass={pass_name} kind={kind} " + " ".join(over),
                               EXIT_BUDGET)
            t = time.time()
            append_line_locked(stream, {"t": t, "pass": pass_name, "kind": kind, **req, "event": "reserve"})
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    print(f"BUDGET=PASS pass={pass_name} kind={kind} " +
          " ".join(f"{f}={req[f]} {kind}_{f}_after={after[kind][f]}/{caps[kind][f]}" for f in BUDGET_FIELDS) +
          " " + " ".join(f"total_{f}_after={after['total'][f]}/{caps['total'][f]}" for f in BUDGET_FIELDS),
          flush=True)
    return {"ledger": str(ledger.resolve()), "reserved": req, "caps": {kind: caps[kind], "total": caps["total"]},
            "totals_before": {kind: before[kind], "total": before["total"]},
            "totals_after": {kind: after[kind], "total": after["total"]}, "reserved_at": t}


# ── 环境指纹 ──────────────────────────────────────────────────────────────────


def git_facts(repo: Path) -> tuple[str | None, bool | None, int | None]:
    head = run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = run_text(["git", "-C", str(repo), "status", "--porcelain"])
    lines = None if status is None else [ln for ln in status.splitlines() if ln.strip()]
    return (head.strip() if head else None), (None if lines is None else bool(lines)), \
        (None if lines is None else len(lines))


def gpu_facts() -> dict[str, Any]:
    exe = os.environ.get("NOISE_RUN_NVIDIA_SMI") or shutil.which("nvidia-smi")
    rows: list[dict[str, str]] = []
    if exe:
        text = run_text([exe, "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"])
        for line in (text or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0]:
                rows.append({"name": parts[0], "driver": parts[1], "uuid": parts[2]})
    if not rows:
        return {"gpus": [], "gpu_model": None, "driver": None, "gpu_uuid": None}

    def uniq(key: str) -> str:
        vals = list(dict.fromkeys(r[key] for r in rows))
        return ";".join(vals)

    return {"gpus": rows, "gpu_model": uniq("name"), "driver": uniq("driver"),
            "gpu_uuid": ",".join(r["uuid"] for r in rows)}


# ── preflight ─────────────────────────────────────────────────────────────────


def cmd_preflight(args) -> int:
    started = time.time()
    out_root = Path(args.out_root)
    prov_out = Path(args.provenance_out)
    for name, value in (("attempts", args.attempts), ("resets", args.resets), ("retries", args.retries)):
        if value < 0:
            raise RunError(f"NOISE_PREFLIGHT=FAIL reason=bad_arg detail=--{name} 须为非负整数", EXIT_USAGE)
    if prov_out.exists():
        raise RunError(f"NOISE_PREFLIGHT=FAIL reason=provenance_exists path={prov_out}", EXIT_FRESH)
    out_state = check_out_root(out_root)
    print(f"RUN_FRESH=PASS out_root={out_root} state={out_state}", flush=True)
    policy_repos = parse_pairs(args.policy_repo, "--policy-repo")
    caps = json.loads(Path(args.budget_caps).read_text(encoding="utf-8"))

    verdict, asset_results, tok_sha = check_assets(args)
    if verdict == "FAIL":
        raise RunError(f"NOISE_PREFLIGHT=FAIL pass={args.pass_name} kind={args.kind} reason=assets", EXIT_ASSETS)

    commit, dirty, dirty_lines = git_facts(REPO)
    policy_commits, policy_dirty = {}, {}
    for name, path in sorted(policy_repos.items()):
        c, d, _ = git_facts(Path(path))
        policy_commits[name], policy_dirty[name] = c, d
    gpu = gpu_facts()

    req = {"attempts": args.attempts, "resets": args.resets, "retries": args.retries}
    budget = reserve_budget(Path(args.budget_ledger), caps, args.pass_name, args.kind, req)

    report: dict[str, Any] = {
        "schema": "noise-run-provenance/1",
        "pass": args.pass_name,
        "kind": args.kind,
        "started_at": started,
        "started_at_iso": iso(started),
        "commit": commit,
        "git_dirty": dirty,
        "git_dirty_lines": dirty_lines,
        "repo": str(REPO),
        "policy_commits": policy_commits,
        "policy_dirty": policy_dirty,
        "assets_verdict": verdict,
        "assets_lock": str(Path(args.assets_lock).resolve()) if args.assets_lock else None,
        "assets_lock_sha256": sha256_file(Path(args.assets_lock)) if args.assets_lock else None,
        "assets_sha": {name: {"root": r["root"], "files": r["files"], "files_sha256": r["files_sha256"],
                              "locked_files_sha256": r["locked_files_sha256"],
                              "verdict": "PASS" if r["ok"] else "FAIL"} for name, r in asset_results.items()},
        "tokenizer": str(args.tokenizer) if args.tokenizer else None,
        "tokenizer_sha256": tok_sha,
        "driver": gpu["driver"],
        "gpu_model": gpu["gpu_model"],
        "gpu_uuid": gpu["gpu_uuid"],
        "gpus": gpu["gpus"],
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "max_steps": args.max_steps,
        "server_args": args.server_args,
        "out_root": str(out_root.resolve()),
        "out_root_state": out_state,
        "real_command": args.real_command,
        "budget": budget,
    }
    report["env_fingerprint"] = env_fingerprint_of(report)
    report["fingerprint"] = fingerprint_of(report)
    prov_out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(prov_out, report)
    print(f"PROVENANCE path={prov_out} env_fingerprint={report['env_fingerprint'][:12]} commit={commit} "
          f"git_dirty={dirty} gpu_model={gpu['gpu_model']} driver={gpu['driver']} host={report['host']} "
          f"slurm_job_id={report['slurm_job_id']}", flush=True)
    print(f"NOISE_PREFLIGHT=PASS pass={args.pass_name} kind={args.kind} assets={verdict} "
          f"fingerprint={report['fingerprint'][:12]}", flush=True)
    return 0


# ── finish ────────────────────────────────────────────────────────────────────


def parse_count(value: str, flag: str) -> int | None:
    if value == "unknown":
        return None
    try:
        n = int(value)
    except ValueError as exc:
        raise RunError(f"NOISE_FINISH=FAIL reason=bad_arg flag={flag} value={value!r}", EXIT_USAGE) from exc
    if n < 0:
        raise RunError(f"NOISE_FINISH=FAIL reason=bad_arg flag={flag} value={value!r}", EXIT_USAGE)
    return n


def cmd_finish(args) -> int:
    prov = Path(args.provenance)
    report = json.loads(prov.read_text(encoding="utf-8"))
    if report.get("pass") != args.pass_name:
        raise RunError(f"NOISE_FINISH=FAIL reason=pass_mismatch provenance={report.get('pass')} arg={args.pass_name}",
                       EXIT_USAGE)
    if any(k in report for k in FINISH_KEYS):
        raise RunError(f"NOISE_FINISH=FAIL reason=already_finished pass={args.pass_name}", EXIT_USAGE)
    actual = {"attempts": parse_count(args.actual_attempts, "--actual-attempts"),
              "resets": parse_count(args.actual_resets, "--actual-resets"),
              "retries": parse_count(args.actual_retries, "--actual-retries")}
    ledger = Path(report["budget"]["ledger"])
    t = time.time()
    with open(ledger, "a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            rows = read_ledger_locked(stream)
            reserves = [r for r in rows if r.get("pass") == args.pass_name and r["event"] == "reserve"]
            if len(reserves) != 1:
                raise RunError(f"NOISE_FINISH=FAIL reason=reserve_count={len(reserves)} pass={args.pass_name}",
                               EXIT_BUDGET)
            if any(r.get("pass") == args.pass_name and r["event"] == "finish" for r in rows):
                raise RunError(f"NOISE_FINISH=FAIL reason=already_finished pass={args.pass_name}", EXIT_BUDGET)
            append_line_locked(stream, {"t": t, "pass": args.pass_name, "kind": report["kind"], **actual,
                                        "exit_code": args.exit_code, "event": "finish"})
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    report.update(finished_at=t, finished_at_iso=iso(t), exit_code=args.exit_code,
                  actual_attempts=actual["attempts"], actual_resets=actual["resets"], actual_retries=actual["retries"])
    write_json_atomic(prov, report)
    reserved = reserves[0]
    over = [f"{f}={actual[f]}>{reserved[f]}" for f in BUDGET_FIELDS
            if actual[f] is not None and actual[f] > int(reserved[f])]
    shown = " ".join(f"{f}={'unknown' if actual[f] is None else actual[f]}" for f in BUDGET_FIELDS)
    if over:
        print(f"BUDGET=FAIL reason=actual_exceeds_reserved pass={args.pass_name} " + " ".join(over), flush=True)
        print(f"NOISE_FINISH=PASS pass={args.pass_name} exit_code={args.exit_code} {shown} overrun=1", flush=True)
        return EXIT_OVERRUN
    print(f"NOISE_FINISH=PASS pass={args.pass_name} exit_code={args.exit_code} {shown} overrun=0", flush=True)
    return 0


# ── ship ──────────────────────────────────────────────────────────────────────


def cmd_ship(args) -> int:
    src, stage = Path(args.src), Path(args.stage)
    ident = src / "identities.jsonl"
    lines = [json.loads(t) for t in ident.read_text(encoding="utf-8").splitlines() if t.strip()] \
        if ident.is_file() else []
    episodes = [ln for ln in lines if ln.get("path")]
    shipped = pulled = sha_bad = missing = matched = 0
    notes: list[str] = []
    for ln in episodes:
        if ln.get("verdict") == "match":
            matched += 1  # --expect-ref：与噪声基线逐字节相同，按设计不复制（Mover 已删节点大文件）
            continue
        rel_h5 = Path(ln["path"])
        ep_rel = rel_h5.parent.parent  # <局目录>/hdf5_files/<名>.h5
        ep_stage = stage / ep_rel
        marker = ep_stage / "SHIPPED"
        if not marker.is_file():
            missing += 1
            notes.append(f"无 SHIPPED：{ep_rel}")
            continue
        shipped += 1
        listing = json.loads(marker.read_text(encoding="utf-8"))
        h5_key = str(rel_h5.relative_to(ep_rel))
        if listing.get(h5_key) != ln.get("sha256"):
            sha_bad += 1
            notes.append(f"SHIPPED 里 h5 sha 与 identities 不符：{ep_rel}")
            continue
        if (ep_stage / "PULLED").exists():
            pulled += 1
            continue
        for rel, sha in listing.items():
            path = ep_stage / rel
            if not path.is_file():
                missing += 1
                notes.append(f"暂存缺文件：{path}")
            elif sha256_file(path) != sha:
                sha_bad += 1
                notes.append(f"暂存 sha 不符：{path}")
    leftovers = [p for p in src.rglob("*") if p.is_file() and p.suffix in MEDIA_SUFFIXES] if src.is_dir() else []
    ok = bool(lines) and sha_bad == 0 and missing == 0 and not leftovers
    if not lines:
        notes.append(f"节点输出根无 identities.jsonl 或为空：{ident}")
    finalized = 0
    if ok and args.finalize:
        stage.mkdir(parents=True, exist_ok=True)
        for path in sorted(p for p in src.rglob("*") if p.is_file() and p.suffix not in MEDIA_SUFFIXES):
            target = stage / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (stage / "SEGMENT_DONE").write_text(iso(time.time()) + "\n", encoding="utf-8")
        finalized = 1
    for note in notes[:10]:
        print(f"# {note}", flush=True)
    print(f"NOISE_SHIP={'PASS' if ok else 'FAIL'} episodes={len(episodes)} identities={len(lines)} shipped={shipped} "
          f"matched={matched} pulled={pulled} sha_bad={sha_bad} missing={missing} node_leftover_media={len(leftovers)} "
          f"finalized={finalized}", flush=True)
    return 0 if ok else 1


# ── 入口 ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("preflight", help="起跑前检查：输出根为空、资产全量 sha、预算登记、环境指纹、来源报告")
    pf.add_argument("--pass", dest="pass_name", required=True)
    pf.add_argument("--kind", required=True, choices=KINDS)
    pf.add_argument("--out-root", required=True)
    pf.add_argument("--budget-ledger", required=True)
    pf.add_argument("--budget-caps", required=True)
    pf.add_argument("--attempts", type=int, required=True)
    pf.add_argument("--resets", type=int, required=True)
    pf.add_argument("--retries", type=int, required=True)
    pf.add_argument("--assets-lock", default=None)
    pf.add_argument("--asset-dir", action="append", default=None, metavar="名=路径")
    pf.add_argument("--tokenizer", default=None)
    pf.add_argument("--tokenizer-sha256", default=None)
    pf.add_argument("--policy-repo", action="append", default=None, metavar="名=路径")
    pf.add_argument("--max-steps", type=int, default=None)
    pf.add_argument("--server-args", default=None)
    pf.add_argument("--real-command", default=None, help="只记录：壳脚本传入的真实命令（shell 转义后的字符串）")
    pf.add_argument("--provenance-out", required=True)
    pf.set_defaults(func=cmd_preflight)

    fi = sub.add_parser("finish", help="真实命令结束后：账本追加 finish 行、来源报告追加结束信息")
    fi.add_argument("--pass", dest="pass_name", required=True)
    fi.add_argument("--provenance", required=True)
    fi.add_argument("--exit-code", type=int, required=True)
    fi.add_argument("--actual-attempts", required=True, help="整数或 unknown")
    fi.add_argument("--actual-resets", required=True, help="整数或 unknown")
    fi.add_argument("--actual-retries", default="unknown", help="整数或 unknown（默认 unknown）")
    fi.set_defaults(func=cmd_finish)

    sh = sub.add_parser("ship", help="核对 hard_parity generate --stage 的逐局暂存是否完整")
    sh.add_argument("--src", required=True, help="节点本地输出根（hard_parity generate --out）")
    sh.add_argument("--stage", required=True, help="NFS 暂存目录（hard_parity generate --stage）")
    sh.add_argument("--finalize", action="store_true", help="核对通过后复制小文件并写 SEGMENT_DONE")
    sh.set_defaults(func=cmd_ship)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RunError as exc:
        print(exc.line, flush=True)
        if args.cmd == "preflight" and not exc.line.startswith("NOISE_PREFLIGHT=FAIL"):
            print(f"NOISE_PREFLIGHT=FAIL pass={args.pass_name} kind={args.kind} code={exc.code}", flush=True)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
