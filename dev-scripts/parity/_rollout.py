"""第二阶段：只读 jsonl → 生成 h5 → 按状态机回写（0927 计划第一部分 §5.2）。

由 ``scripts/parity/v4_rollout.py`` 下沉：真正起环境、录 h5、跑规划的仍是 ``train_split_runner.py``
（``--identity-source formula --no-recovery``）→ ``train_split_worker.run_one``（``gym.make(..., sampling_config=,
native_episode_spec=)``），环境包由 ``ROBOMME_ENV_PACKAGE`` 决定（默认 robomme_hard）。

两种模式：

* ``continue``（正常生产）：进入即在 ``<specs>.lock`` 上 ``O_EXCL`` 取锁并持有到回写结束；待跑集 = 每格
  ``selected`` 且 ``rollout`` 缺失的行；某行失败 → ``selected=false``、``rollout.status="failed"``、``tried=true``，
  从同格 ``tried=false`` 且未选中的候选里按 ``candidate`` 升序递补一个；每格 ``selected`` 行数 ≤ ``delivery_per_cell``。
  基础设施失败每身份最多重跑 1 次（计入预算、记原因与次数）；任务失败不重试挑成功。
  全部批次结束后一次回写：重读 specs，整份文件 sha256 必须等于读入时；只改 ``selected``／``tried``／``rollout``；
  临时文件 + ``os.replace``；写后 ``load_specs`` 核 ``identity_sha256`` 未变并重算 ``delivery_sha256``。
* ``replay``（对拍专用）：只按给定身份清单逐局重放，不递补、不回写 jsonl，结果只写 ``--output``；
  清单与调度集合必须全等 → ``REPLAY_SET``。

恢复（``--resume``）：联合核对各轮 ``jobs.json``、``results.partial.jsonl``、局目录与 h5；局目录有完整 h5 而 partial
无记录的身份标 ``UNKNOWN`` 并列清单交用户，不直接重跑、不直接采纳。

v8（v8 方案第二部分 §2.2 第 7 条，``hard-specs/4``）：

* 执行步 ``exec_steps`` = h5 的 ``timestep_*`` 数 − ``info/is_video_demo`` 为真的帧数（口径同
  ``hard_regression.py::cmd_step_headroom``），由 ``h5_facts`` 算出写进结果段（不进签）；
* ``run_continue`` 读到 /4 header 时自动启用 ``exec_cap``：执行步超过上限的局在 ``apply_results`` 里记
  ``status=failed``、``error_type=exec_over_cap`` 并保留 ``exec_steps``，``selected=false``、``tried=true``，触发同格递补；
  其 h5／mp4 只在该局的显式目录 ``<output>/episodes/<tier>/<task>_episode_<n>/`` 里删除（不跨运行 glob）；
* ``plan_pending`` 接逐任务配额 ``delivery_per_cell[task]``；
* 尝试账本 ``<output>/results.jsonl``：每条最终结果与每次基础设施重试各追加一行，``--resume`` 时按账本重放恢复
  状态、基础设施重试计数跨重启保留（每身份 ≤ 1 次）；
* ``run_continue_v8``：按格表逐档调用 ``run_continue``，收尾 ``aggregate_v8`` 写 ``delivery.json``（``v8-delivery/1``）
  并打印 ``V8_DELIVERY_SET``；
* 生成分片：``split_v8``（冻结根 → 每片自己的 ``<片输出>/specs`` 规格根、锁、账本）→ 各片 ``run_continue_v8``
  → ``aggregate_v8`` 按片账本核全部格（V8 的四席分片表与 ``merge_v8`` 已于维护计划 W2 删除）。
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import _common  # noqa: F401  路径设置

from robomme_hard.env_record_wrapper import hard_specs  # noqa: E402

RUNNER = _common.REPO_ROOT / "scripts" / "parity" / "train_split_runner.py"
INFRA_TYPES = {"BrokenProcessPool", "TerminatedWorkerError", "TimeoutError", "RunnerCrash"}
INFRA_PATTERN = re.compile(r"svulkan2|vulkan|ErrorIncompatibleDriver|EXCLUSIVE|CUDA error|out of memory|"
                           r"Resource temporarily unavailable|Too many open files", re.I)


class RolloutError(RuntimeError):
    """锁、回写、恢复歧义或清单不符；一律停止。"""


def is_infra(result: dict[str, Any]) -> bool:
    if result.get("ok"):
        return False
    return result.get("error_type") in INFRA_TYPES or bool(INFRA_PATTERN.search(str(result.get("error") or "")))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── 锁 ─────────────────────────────────────────────────────────────────


class SpecsLock:
    """``<specs>.lock``：``O_EXCL`` 创建，记 pid／host／启动时间；已存在一律拒绝（不自动判陈旧）。"""

    def __init__(self, specs: Path):
        self.path = Path(str(specs) + ".lock")
        self.held = False

    def acquire(self) -> None:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            info = self.path.read_text(errors="replace") if self.path.exists() else ""
            raise RolloutError(f"锁已存在，拒绝并交用户：{self.path} {info.strip()}") from exc
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                       "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}, stream)
        self.held = True

    def release(self) -> None:
        if self.held and self.path.exists():
            self.path.unlink()
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


# ── 调 runner ──────────────────────────────────────────────────────────


def episode_dir(out_dir: Path, row: dict[str, Any]) -> Path:
    return out_dir / "episodes" / row["tier"] / f"{row['task']}_episode_{row['episode']}"


def h5_facts(path: Path) -> dict[str, Any]:
    import h5py

    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
            size += len(chunk)
    with h5py.File(path, "r") as handle:
        episode = handle[list(handle.keys())[0]]
        steps = [k for k in episode.keys() if k.startswith("timestep_")]
        # 执行步口径同 hard_regression.py::cmd_step_headroom：总帧 − info/is_video_demo 为真的帧
        demo = sum(bool(episode[k]["info/is_video_demo"][()]) for k in steps)
    return {"h5_sha256": digest.hexdigest(), "bytes": size, "frames": len(steps), "exec_steps": len(steps) - demo}


def run_batch(batch: list[dict[str, Any]], header: dict[str, Any], out_dir: Path, round_index: int, *,
              src_root: Path, workers: int, gpu: str, pkg: str, official_root: Path | None = None,
              resume: bool = False, runner_env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """一批身份（同一档）交给 runner，返回逐条结果（含回注绑定核验与 h5 事实）。"""
    tiers = {row["tier"] for row in batch}
    if len(tiers) != 1 or tiers != {header["difficulty"]}:
        raise RolloutError(f"一批只能是同一档且与 header 一致：{tiers}")
    work = out_dir / "_rounds" / f"{header['difficulty']}_round_{round_index:02d}"
    work.mkdir(parents=True, exist_ok=resume)
    done_here = set()
    if resume and (work / "results.partial.jsonl").exists():
        done_here = {(r["task"], int(r["episode"])) for r in map(json.loads, (work / "results.partial.jsonl").read_text().splitlines()) if r}
    jobs, specs = [], {}
    for row in batch:
        wdir = episode_dir(out_dir, row)
        if wdir.exists() and (row["task"], row["episode"]) not in done_here:
            # 官方 _worker 要求局目录不存在：上一次（基础设施失败或中断）的目录改名挪开留作证据，不删除
            aside = wdir.with_name(f"{wdir.name}.aside{int(time.time())}")
            wdir.rename(aside)
            print(f"# 局目录已存在，挪开留证：{aside}", flush=True)
        jobs.append({"task": row["task"], "episode": row["episode"], "seed": row["seed"], "attempt": row["attempt"],
                     "difficulty": row["tier"], "seed_rule": header["seed_rule"],
                     "worker_dir": str(episode_dir(out_dir, row))})
        specs[f"{row['task']}/{row['episode']}"] = row["spec"]
    tasks = sorted({row["task"] for row in batch})
    (work / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    (work / "sampling.json").write_text(json.dumps({"tasks": {t: header["sampling_config"][t] for t in tasks}},
                                                   ensure_ascii=False), encoding="utf-8")
    (work / "specs.json").write_text(json.dumps({"specs": specs}, ensure_ascii=False), encoding="utf-8")
    command = [
        sys.executable, str(RUNNER), "--src-root", str(src_root), "--jobs-json", str(work / "jobs.json"),
        "--results-json", str(work / "results.json"), "--workers", str(workers), "--gpu", str(gpu),
        "--sampling-config", str(work / "sampling.json"), "--episode-specs", str(work / "specs.json"),
        "--identity-source", "formula", "--no-recovery",
    ]
    if official_root is not None:
        command += ["--official-root", str(official_root)]
    if resume:
        command.append("--resume")
    env = dict(runner_env or os.environ)
    env.update(ROBOMME_ENV_PACKAGE=pkg, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    started = time.time()
    with (work / "runner.log").open("a", encoding="utf-8") as log:
        proc = subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    raw: dict[tuple[str, int], dict[str, Any]] = {}
    if (work / "results.json").exists():
        raw = {(r["task"], int(r["episode"])): r for r in json.loads((work / "results.json").read_text())["results"]}
    elif (work / "results.partial.jsonl").exists():
        for line in (work / "results.partial.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                raw[(r["task"], int(r["episode"]))] = r
    out = []
    for row in batch:
        result = raw.get((row["task"], row["episode"]))
        if result is None:
            result = {"ok": False, "error_type": "RunnerCrash", "error": f"runner exit={proc.returncode}，无该局结果"}
        wdir = episode_dir(out_dir, row)
        binding = None
        replay = wdir / "spec_replay.json"
        if replay.exists():
            payload = json.loads(replay.read_text(encoding="utf-8"))
            mismatches = payload.get("mismatches", [])
            binding = {"mismatch": len(mismatches),
                       "unattributed_mismatch": sum(1 for m in mismatches if not m.get("decision_key")),
                       "unused": len(payload.get("unused", [])), "value_points": payload.get("value_points"),
                       "layout_hit": payload.get("layout_hit", 0), "layout_drift": payload.get("layout_drift", 0),
                       "layout_overridden": payload.get("layout_overridden", 0)}
        h5 = sorted((wdir / "hdf5_files").glob("*.h5"))
        record = {
            "task": row["task"], "tier": row["tier"], "candidate": row["candidate"], "episode": row["episode"],
            "seed": row["seed"], "attempt": row["attempt"], "spec_sha256": row["spec_sha256"],
            "ok": bool(result.get("ok")), "error_type": result.get("error_type"),
            "error": (str(result.get("error") or ""))[:500] or None, "spec_binding": binding,
            "env_package": result.get("env_package"), "env_module": result.get("env_module"),
            "wrapper_modules": result.get("wrapper_modules"),
            "h5": str(h5[0]) if h5 else None, "round": round_index, "role": row.get("_role", "selected"),
        }
        if record["ok"] and h5:
            try:
                record.update(h5_facts(h5[0]))
            except FileNotFoundError:
                # hard_parity generate 的搬运线程（Mover）可能在列目录与打开之间把 h5 搬走或按 match 删掉；
                # 与「列目录时已被搬走」同等处理（h5 记 None、不回读事实），不让整批生成崩溃。
                record["h5"] = None
        out.append(record)
    print(f"ROUND {header['difficulty']}/{round_index} jobs={len(batch)} ok={sum(r['ok'] for r in out)} "
          f"infra={sum(is_infra(r) for r in out)} wall_s={time.time() - started:.0f}", flush=True)
    return out


def rollout_block(record: dict[str, Any], pkg: str, code_baseline: str, output: Path) -> dict[str, Any]:
    block = {
        "status": "ok" if record["ok"] else "failed", "round": record["round"], "role": record["role"],
        "env_package": pkg, "env_module": record.get("env_module"), "code_baseline": code_baseline,
        "source": str(output), "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if record["ok"]:
        block.update({k: record.get(k) for k in ("h5_sha256", "bytes", "frames")})
        block["h5_path"] = record.get("h5")
    else:
        block["error_type"] = record.get("error_type")
        if record.get("purged") is not None:
            block["purged_files"] = len(record["purged"])
    if record.get("exec_steps") is not None:
        block["exec_steps"] = int(record["exec_steps"])
    return block


def _has_run_traces(output: Path) -> bool:
    """输出目录已跑过（有局目录或轮次目录）才拒绝；调用方预先写入的 launch／清单等小文件不算。"""
    return (output / "episodes").exists() or (output / "_rounds").exists()


def unknown_identities(out_dir: Path) -> list[str]:
    """局目录有完整 h5、但各轮 partial 与 results.json 都没有记录的身份（恢复的歧义窗口）。"""
    recorded = set()
    for partial in out_dir.glob("_rounds/*/results.partial.jsonl"):
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recorded.add((r["task"], int(r["episode"]), str(r.get("difficulty"))))
    unknown = []
    for h5 in out_dir.glob("episodes/*/*/hdf5_files/*.h5"):
        if ".aside" in h5.parents[1].name:
            continue  # run_batch 挪开留证的旧局目录（``<局目录>.aside<时间戳>``）不参与恢复判定
        tier = h5.parents[2].name
        task, _, episode = h5.parents[1].name.rpartition("_episode_")
        if (task, int(episode), tier) not in recorded:
            unknown.append(f"{tier}/{task}/{episode}")
    return sorted(unknown)


# ── continue 模式：状态机 ──────────────────────────────────────────────────


def plan_pending(rows: list[dict[str, Any]], per_cell: int | dict[str, int], redo: set[tuple[str, int]] = frozenset(),
                 tasks: set[str] | None = None) -> list[dict[str, Any]]:
    """待跑集 = 每格 ``selected`` 且 ``rollout`` 缺失的行。``per_cell`` 为整数（/2、/3 全局配额）或
    ``{task: 配额}``（/4 逐任务配额）；任一格 selected 超过配额即拒绝。"""
    for row in rows:
        if (row["task"], row["candidate"]) in redo:
            row["rollout"], row["selected"] = None, True
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + int(row["selected"])
    quota = (lambda task: int(per_cell[task])) if isinstance(per_cell, dict) else (lambda task: int(per_cell))  # noqa: E731
    bad = {t: n for t, n in counts.items() if n > quota(t)}
    if bad:
        raise RolloutError(f"每格 selected 超过 delivery_per_cell={per_cell}：{bad}")
    return [row for row in rows if row["selected"] and row["rollout"] is None and (tasks is None or row["task"] in tasks)]


def purge_episode_media(output: Path, row: dict[str, Any], h5: str | None) -> list[str]:
    """只在该局的显式目录 ``<output>/episodes/<tier>/<task>_episode_<n>/`` 里删 h5 与 mp4（不跨运行 glob）；
    记录的 h5 路径不在该目录内即拒绝删除。返回删掉的文件路径。"""
    wdir = episode_dir(Path(output), row).resolve()
    if h5 and wdir not in Path(h5).resolve().parents:
        raise RolloutError(f"h5 不在该局目录内，拒绝删除：{h5}（期望在 {wdir} 下）")
    removed: list[str] = []
    if wdir.is_dir():
        for path in sorted(wdir.rglob("*")):
            if path.is_file() and path.suffix in (".h5", ".mp4"):
                path.unlink()
                removed.append(str(path))
    return removed


def enforce_exec_cap(record: dict[str, Any], row: dict[str, Any], output: Path, exec_cap: int, *,
                     purge: bool = True) -> None:
    """v8 执行步上限（就地改 ``record``）：ok 却无 h5／无 ``exec_steps`` 的局记失败（不让缺数据的局混进交付）；
    执行步 > ``exec_cap`` 的局记 ``exec_over_cap``、保留 ``exec_steps``，并删该局目录里的 h5／mp4。
    ``purge=False`` 时只标 ``record["purge_pending"] = True``，由调用方先追加账本再 ``purge_record_media``
    （崩溃在两步之间时 ``--resume`` 重放账本会补删，不会重跑该局）。"""
    if not record.get("ok"):
        return
    steps = record.get("exec_steps")
    if steps is None:
        kind = "h5_missing" if not record.get("h5") else "exec_steps_missing"
        record.update(ok=False, error_type=kind, error=f"{kind}：ok 结果缺 h5 或执行步，不进交付")
        return
    if int(steps) > int(exec_cap):
        record.update(ok=False, error_type="exec_over_cap", error=f"执行步 {steps} > 上限 {exec_cap}")
        if purge:
            record["purged"] = purge_episode_media(output, row, record.get("h5"))
        else:
            record["purge_pending"] = True


def purge_record_media(rows: list[dict[str, Any]], records: list[dict[str, Any]], output: Path) -> None:
    """对标了 ``purge_pending`` 的结果删该局显式目录里的 h5／mp4，并把删除数写进行的 rollout 段。"""
    by_key = {(r["task"], r["candidate"]): r for r in rows}
    for record in records:
        if record.get("purge_pending"):
            row = by_key[(record["task"], record["candidate"])]
            purged = purge_episode_media(output, row, record.get("h5"))
            if row.get("rollout") is not None:
                row["rollout"]["purged_files"] = len(purged)


def apply_results(rows: list[dict[str, Any]], results: list[dict[str, Any]], pkg: str, code_baseline: str,
                  output: Path, *, exec_cap: int | None = None, purge: bool = True,
                  same_way_tasks: frozenset[str] | set[str] = frozenset()) -> list[dict[str, Any]]:
    """按结果更新行；失败的行退选并递补同格下一个未试候选。返回新一轮待跑行。

    ``exec_cap``（v8）给出时先过 ``enforce_exec_cap``：超限局按失败处理（同格递补），``record`` 被就地改写；
    ``purge=False`` 时超限局的媒体留给调用方在追加账本之后删（见 ``purge_record_media``）。
    ``same_way_tasks``（v9 方案 §2.2 第 4 条：MoveCube）里的任务只从与失败局同一运动方式（``way_idx``）的未试备用
    里递补，没有同方式备用就不递补（该格最终记 ``exhausted``）；其他任务照旧按候选号取同格下一个。"""
    by_key = {(r["task"], r["candidate"]): r for r in rows}
    backfill = []
    for record in results:
        row = by_key[(record["task"], record["candidate"])]
        if exec_cap is not None:
            enforce_exec_cap(record, row, output, exec_cap, purge=purge)
        row["tried"] = True
        row["rollout"] = rollout_block(record, pkg, code_baseline, output)
        if not record["ok"]:
            row["selected"] = False
            spare = sorted((r for r in rows if r["task"] == row["task"] and not r["tried"] and not r["selected"]),
                           key=lambda r: r["candidate"])
            if row["task"] in same_way_tasks:
                way = row_way(row)
                spare = [r for r in spare if row_way(r) == way]
            if spare:
                spare[0]["selected"] = True
                spare[0]["_role"] = "backfill"
                backfill.append(spare[0])
    return backfill


def row_way(row: dict[str, Any]) -> int | None:
    """规格行的 MoveCube 运动方式（``_freeze._movecube_way``：最后一次 _initialize_episode 的 way_idx）；读不到为 None。"""
    from _freeze import _movecube_way  # noqa: PLC0415

    return _movecube_way(row.get("spec"))


#: v9 起按运动方式同方式递补的任务（v9 方案 §2.2 第 4 条：MoveCube 逐方式配额 17／17／16，递补不跨方式）
SAME_WAY_BACKFILL_TASKS = frozenset({"MoveCube"})


def same_way_tasks_for(header: dict[str, Any]) -> frozenset[str]:
    """/4 规格（新值档）里同方式递补的任务；/2、/3 一律空集（v7 及更早的递补规则逐字不变）。"""
    if header.get("schema") != hard_specs.SCHEMA or header.get("difficulty") not in hard_specs.TIERS:
        return frozenset()
    return frozenset(SAME_WAY_BACKFILL_TASKS & set(header.get("tasks") or ()))


def load_specs_any(path: Path, *, check_fingerprint: bool = False):
    """单文件读取：/4 文件的配额上限格表按 header 自带的逐任务配额推出（``hard_specs.header_cell_table``，
    V9 文件落 V9_CELLS）。"""
    with Path(path).open(encoding="utf-8") as stream:
        first = stream.readline()
    try:
        table = hard_specs.header_cell_table(json.loads(first))
    except ValueError:
        table = None  # 首行坏了：交给 load_specs 报具体错
    return hard_specs.load_specs(path, expected_cells=table, check_fingerprint=check_fingerprint)


def write_back(specs: Path, header: dict[str, Any], rows: list[dict[str, Any]], expected_file_sha: str) -> dict[str, Any]:
    """持锁回写：整份文件 sha 必须未变；只改结果段；临时文件 + os.replace；写后复核身份不变。"""
    if file_sha256(specs) != expected_file_sha:
        raise RolloutError(f"{specs} 在运行期间被他人改过（整份文件 sha 不符），中止回写")
    clean = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    new_header = dict(header)
    new_header["delivery_sha256"] = hard_specs.delivery_sha256(clean)
    if hard_specs.identity_sha256(new_header, clean) != header["identity_sha256"]:
        raise RolloutError("回写后 identity_sha256 变了——只允许改结果段")
    hard_specs.validate_specs(new_header, clean, expected_cells=hard_specs.header_cell_table(new_header))
    fd, name = tempfile.mkstemp(prefix=".hardspecs-", dir=specs.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for record in [new_header, *clean]:
            stream.write(hard_specs.canonical_json(record) + "\n")
    os.chmod(name, 0o644)
    os.replace(name, specs)
    header2, _ = load_specs_any(specs)
    if header2["identity_sha256"] != header["identity_sha256"]:
        raise RolloutError("回写后重读 identity_sha256 不符")
    return new_header


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """尝试账本（``<output>/results.jsonl``）：每行 ``{"kind": "result"|"infra_retry", tier, task, candidate, round, ...}``。"""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_ledger(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    with Path(path).open("a", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_continue(specs: Path, output: Path, *, src_root: Path, workers: int, gpu: str, pkg: str,
                 code_baseline: str, redo: set[tuple[str, int]] = frozenset(), resume: bool = False,
                 max_infra_retries: int = 1, batch_runner=None, tasks: set[str] | None = None,
                 ledger: Path | None = None, check_traces: bool = True,
                 same_way_tasks: frozenset[str] | set[str] | None = None) -> dict[str, Any]:
    """``batch_runner`` 只供纯 CPU 夹具注入假 runner（STATE_MACHINE）。

    ``same_way_tasks`` 缺省按 header 取 ``same_way_tasks_for``（/4 新值档的 MoveCube 同方式递补；/2、/3 为空），
    账本重放与正常轮次用同一规则。

    /4 规格自动启用 header ``exec_cap``（执行步超限按失败递补）与逐任务配额。``ledger``（v8 驱动传
    ``<output>/results.jsonl``）给出时：每条最终结果与每次基础设施重试追加进账本；``resume`` 且规格尚未回写
    （本档无 ``tried`` 行）时按账本顺序重放结果恢复状态，基础设施重试计数一律从账本累计（跨重启保留，每身份 ≤
    ``max_infra_retries``），轮次号接着账本最大轮次往后编。``check_traces=False`` 供多档驱动只在入口查一次运行痕迹。"""
    runner = batch_runner or (lambda batch, header, idx: run_batch(
        batch, header, output, idx, src_root=src_root, workers=workers, gpu=gpu, pkg=pkg, resume=resume))
    lock = SpecsLock(specs)
    lock.acquire()  # 锁先于一切：取锁失败的进程 worker 数为零、直接退出
    try:
        file_sha = file_sha256(specs)
        header, rows = load_specs_any(specs)
        exec_cap = int(header["exec_cap"]) if header["schema"] == hard_specs.SCHEMA else None
        same_way = same_way_tasks_for(header) if same_way_tasks is None else frozenset(same_way_tasks)
        if resume:
            unknown = unknown_identities(output)
            if unknown:
                raise RolloutError(f"恢复歧义：以下身份有 h5 但无 partial 记录，标 UNKNOWN 交用户：{unknown}")
        elif check_traces and _has_run_traces(output):
            raise RolloutError(f"{output} 已有运行痕迹（episodes/ 或 _rounds/）；续跑用 --resume")
        output.mkdir(parents=True, exist_ok=True)
        tier = header["difficulty"]
        infra_retries: dict[tuple[str, int], int] = {}
        attempted = 0
        round_index = 0
        if ledger is not None:
            entries = [e for e in read_ledger(ledger) if e["tier"] == tier]
            if entries and not resume:
                raise RolloutError(f"{ledger} 已有 {tier} 的记录；续跑用 --resume")
            for entry in entries:
                key = (entry["task"], int(entry["candidate"]))
                if entry["kind"] == "infra_retry":
                    infra_retries[key] = infra_retries.get(key, 0) + 1
            if entries and not any(r["tried"] for r in rows):
                # 规格尚未回写：按账本顺序逐条重放（与原轮次里的逐条处理顺序相同，递补选择因此逐一复现）
                for entry in entries:
                    if entry["kind"] == "result":
                        record = dict(entry["record"])
                        apply_results(rows, [record], pkg, code_baseline, output, same_way_tasks=same_way)
                        # 账本已记超限、但崩溃发生在删媒体之前：此处补删（幂等）
                        purge_record_media(rows, [record], output)
            attempted = len(entries)
            round_index = 1 + max((int(e["round"]) for e in entries), default=-1)
        pending = plan_pending(rows, header["delivery_per_cell"], redo, tasks)
        while pending:
            results = runner(pending, header, round_index)
            attempted += len(results)
            retry = []
            final = []
            infra_entries = []
            for record in results:
                key = (record["task"], record["candidate"])
                if is_infra(record) and infra_retries.get(key, 0) < max_infra_retries:
                    infra_retries[key] = infra_retries.get(key, 0) + 1
                    retry.append(next(r for r in pending if (r["task"], r["candidate"]) == key))
                    infra_entries.append({"kind": "infra_retry", "tier": tier, "task": key[0], "candidate": key[1],
                                          "round": round_index, "error_type": record.get("error_type"),
                                          "error": (str(record.get("error") or ""))[:300] or None})
                else:
                    final.append(record)
            backfill = apply_results(rows, final, pkg, code_baseline, output, exec_cap=exec_cap, purge=False,
                                     same_way_tasks=same_way)
            if ledger is not None:
                append_ledger(ledger, infra_entries + [
                    {"kind": "result", "tier": tier, "task": r["task"], "candidate": int(r["candidate"]),
                     "round": round_index, "record": r} for r in final])
            # 先落账本、再删超限局的媒体（N7：崩溃后 --resume 按账本重放，不会重跑该局）
            purge_record_media(rows, final, output)
            pending = retry + backfill
            round_index += 1
        header = write_back(specs, header, rows, file_sha)
        delivered = sum(hard_specs.delivered(r) for r in rows)
        summary = {"attempted": attempted, "rounds": round_index, "infra_retries": sum(infra_retries.values()),
                   "delivered": delivered, "delivery_sha256": header["delivery_sha256"]}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        lock.release()


# ── continue 模式（v8）：逐格配额、执行步上限、分片与聚合 ─────────────────────


V8_DELIVERY_SCHEMA = "v8-delivery/1"
V8_SHARD_SCHEMA = "v8-shard/1"
LEDGER_NAME = "results.jsonl"
SHARD_META = "shard.json"
#: v9 生成分片（v9 方案 §2.12、§2.4.2 第 4 条）：只有 MoveCube 一片走 freeze → split → continue；InsertPeg 不走 split，
#: 由 ``v9_subset_specs.py extend`` 直接产出片根；其余 14 任务是 V8 子集、不生成。``--cells v9shard1`` 取 V9_CELLS 的局数。
V9_SHARD_TASKS: dict[str, tuple[str, ...]] = {"shard1": ("MoveCube",)}
#: v9 2b 冒烟格表（v9 方案 §2.4.2 第 2 条）：MoveCube、InsertPeg 的 xhard4 各 1 局（``--cells v9smoke``）
V9_SMOKE_CELLS: dict[tuple[str, str], int] = {("MoveCube", "xhard4"): 1, ("InsertPeg", "xhard4"): 1}
assert all(task in hard_specs.ALL_TASKS for tasks in V9_SHARD_TASKS.values() for task in tasks)
assert all(0 < n <= hard_specs.V9_CELLS[k] for k, n in V9_SMOKE_CELLS.items())
#: ``--cells`` 的 v9 具名格表：``v9``（V9_CELLS 完整 43 格 800，freeze_specs／generate_h5 的缺省）、
#: ``v9shard1``（V9_SHARD_TASKS 的片，局数取 V9_CELLS）、``v9smoke``
V9_CELL_NAMES = ("v9", "v9smoke", *(f"v9{name}" for name in V9_SHARD_TASKS))
#: 逐格与全局计数键（全部显式写出，零值也写）
V8_CELL_COUNT_KEYS = ("expected", "candidates", "tried", "delivered", "failed", "exec_over_cap", "backfills",
                      "infra_retries", "spares_left", "pending", "bad_h5")
V8_TOTAL_COUNT_KEYS = V8_CELL_COUNT_KEYS + ("exhausted_cells", "pending_cells", "failed_cells")


def order_cells(cells: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    """格表按 (TIERS 档序, ALL_TASKS 任务序) 排序（输出与合并的确定性）。"""
    return {key: int(cells[key]) for key in sorted(cells, key=lambda k: (hard_specs.TIERS.index(k[1]),
                                                                         hard_specs.ALL_TASKS.index(k[0])))}


def cell_table(cells: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    """格表 → 作配额上限的完整交付格表（``hard_specs.resolve_cell_table``：EXPECTED_CELLS → CELL_TABLES 第一张覆盖的表）。"""
    return hard_specs.resolve_cell_table(cells)


def check_cells(cells: dict[tuple[str, str], int],
                table: dict[tuple[str, str], int] | None = None) -> dict[tuple[str, str], int]:
    """格表核对：键必须在交付格表里、局数 1..表值。``table`` 缺省按 ``cell_table(cells)`` 取（V9 子表落 V9_CELLS），
    显式给出时只按它核。"""
    if not isinstance(cells, dict) or not cells:
        raise RolloutError("格表必须是非空 {(task, tier): 局数}")
    table = cell_table(cells) if table is None else table
    stray = sorted(k for k in cells if k not in table)
    if stray:
        raise RolloutError(f"格表含 V9_CELLS 之外的格（不交付的格不抽、不生成）：{stray}")
    bad = {k: n for k, n in cells.items() if not isinstance(n, int) or isinstance(n, bool) or not
           0 < n <= table[k]}
    if bad:
        raise RolloutError(f"格表局数须为 1..表 2 配额的整数：{bad}")
    return order_cells(cells)


def resolve_cells(spec: str | Path) -> dict[tuple[str, str], int]:
    """``--cells``：``v9``（V9_CELLS 完整 43 格 800）／``v9shard1``（V9 MoveCube 一片，局数取 V9_CELLS）／``v9smoke``（V9 冒烟 2 格）／
    格表 JSON 路径（``{"Task@tier": 局数}`` 或 ``{"cells": [{"task", "tier", "count"}]}``）。
    V8 专用的 ``full``（V8 1070 局表）与 ``smoke``（V8 2b 冒烟 7 格）已于维护计划阶段 1b（W4）删除。"""
    text = str(spec)
    if text in ("full", "smoke"):
        raise RolloutError(f"--cells {text} 只服务 V8（1070 局表／2b 冒烟），已删除；请显式给 "
                           f"{'／'.join(V9_CELL_NAMES)} 或格表 JSON 路径")
    if text == "v9":
        return check_cells(dict(hard_specs.V9_CELLS), hard_specs.V9_CELLS)
    if text == "v9smoke":
        return check_cells(dict(V9_SMOKE_CELLS), hard_specs.V9_CELLS)
    if text.startswith("v9") and text[2:] in V9_SHARD_TASKS:
        return check_cells({k: n for k, n in hard_specs.V9_CELLS.items() if k[0] in V9_SHARD_TASKS[text[2:]]},
                           hard_specs.V9_CELLS)
    path = Path(text)
    if not path.is_file():
        raise RolloutError(f"--cells 须为 {'／'.join(V9_CELL_NAMES)} 或格表 JSON 路径：{text!r}")
    data = json.loads(path.read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], int] = {}
    items = data["cells"] if isinstance(data, dict) and "cells" in data else data
    if isinstance(items, dict):
        for key, n in items.items():
            task, _, tier = key.partition("@")
            cells[(task, tier)] = n
    else:
        for item in items:
            cells[(item["task"], item["tier"])] = item["count"]
    return check_cells(cells)


def cells_json(cells: dict[tuple[str, str], int]) -> dict[str, int]:
    """格表 → JSON 可写形态 ``{"Task@tier": 局数}``（``resolve_cells`` 可读回）。"""
    return {f"{task}@{tier}": n for (task, tier), n in order_cells(cells).items()}


def cell_tiers(cells: dict[tuple[str, str], int]) -> list[str]:
    return [tier for tier in hard_specs.TIERS if any(t == tier for _, t in cells)]


def compact_range(values) -> str:
    values = list(values)
    if values and values == list(range(values[0], values[0] + len(values))):
        return f"{values[0]}..{values[-1]}"
    return ",".join(str(v) for v in values)


def detect_root_schema(root: Path) -> str | None:
    """规格根的 schema：``<root>/<tier>/specs.jsonl`` 首行 header 的 schema；各档必须一致，否则拒绝。无文件返回 None。"""
    schemas = {}
    for tier in hard_specs.TIERS:
        path = Path(root) / tier / "specs.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                schemas[tier] = json.loads(stream.readline())["schema"]
    if not schemas:
        return None
    if len(set(schemas.values())) != 1:
        raise RolloutError(f"{root} 各档 schema 不一致：{schemas}")
    return next(iter(schemas.values()))


def _seed_disjoint(loaded: dict[str, tuple]) -> None:
    seeds: dict[str, dict[str, set[int]]] = {}
    for tier, (_, rows) in loaded.items():
        for row in rows:
            seeds.setdefault(row["task"], {}).setdefault(tier, set()).add(int(row["seed"]))
    for task, by_tier in seeds.items():
        names = list(by_tier)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if by_tier[a] & by_tier[b]:
                    raise RolloutError(f"{task} 在 {a} 与 {b} 的 seed 相交")


def load_v8_root(root: Path, cells: dict[tuple[str, str], int]) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """按格表读 v8 规格根：逐档 ``load_specs``（/4 校验）+ 档名、任务集合、逐格 ``delivery_per_cell`` 与格表相等、
    跨档 seed 不交。全部行都未试过（刚冻结／刚切片）时另走 ``hard_specs.load_specs_root`` 全量契约（含每格 selected
    数 == 格表）；跑过之后某格备用耗尽会让 selected 少于配额，这时只按单文件校验读，逐格成败交给聚合判定。"""
    cells = check_cells(cells)
    table = cell_table(cells)
    out: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for tier in cell_tiers(cells):
        path = Path(root) / tier / "specs.jsonl"
        if not path.is_file():
            raise RolloutError(f"v8 规格根缺少 {path}")
        header, rows = hard_specs.load_specs(path, expected_cells=table, check_fingerprint=False)
        if header["schema"] != hard_specs.SCHEMA or header["difficulty"] != tier:
            raise RolloutError(f"{path}：须为 {hard_specs.SCHEMA} 且档位 {tier}（实为 {header['schema']}／"
                               f"{header['difficulty']}）")
        want = {task for task, t in cells if t == tier}
        if set(header["tasks"]) != want:
            raise RolloutError(f"{tier} 任务集合与格表不符：缺少 {sorted(want - set(header['tasks']))}，"
                               f"多出 {sorted(set(header['tasks']) - want)}")
        for task in sorted(want):
            if header["delivery_per_cell"][task] != cells[(task, tier)]:
                raise RolloutError(f"{task}/{tier} delivery_per_cell={header['delivery_per_cell'][task]} ≠ 格表 "
                                   f"{cells[(task, tier)]}")
        out[tier] = (header, rows)
    _seed_disjoint(out)
    if not any(row["tried"] for _, rows in out.values() for row in rows):
        hard_specs.load_specs_root(root, cells, cell_table=table, check_fingerprint=False)
    return out


def _ledger_infra(dirs: list[Path]) -> tuple[dict[tuple[str, str], int], list[str]]:
    counts: dict[tuple[str, str], int] = {}
    paths = []
    for d in dirs:
        path = Path(d) / LEDGER_NAME
        if not path.is_file():
            raise RolloutError(f"缺少尝试账本 {path}（聚合要从账本读基础设施重试数）")
        paths.append(str(path))
        for entry in read_ledger(path):
            if entry["kind"] == "infra_retry":
                key = (entry["task"], entry["tier"])
                counts[key] = counts.get(key, 0) + 1
    return counts, paths


def parse_rebase(items) -> list[tuple[str, str]]:
    """``--rebase <旧前缀>=<新前缀>``（可重复）→ ``[(旧, 新)]``，按旧前缀长度降序（最长前缀优先匹配）。"""
    out = []
    for item in items or []:
        old, sep, new = str(item).partition("=")
        if not sep or not old or not new:
            raise RolloutError(f"--rebase 须为 <旧前缀>=<新前缀>：{item!r}")
        out.append((os.path.abspath(old), os.path.abspath(new)))
    return sorted(out, key=lambda p: -len(p[0]))


def rebase_path(path: str, rebase: list[tuple[str, str]]) -> str | None:
    """按前缀替换（只认整段目录前缀）；没有任何前缀匹配返回 None（调用方判 FAIL，不静默保留旧路径）。"""
    path = os.path.abspath(path)
    for old, new in rebase:
        if path == old or path.startswith(old.rstrip(os.sep) + os.sep):
            return new + path[len(old):]
    return None


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_v8(specs_root: Path, cells: dict[tuple[str, str], int], ledger_dirs: list[Path], out_path: Path, *,
                 cells_label: str = "custom", code_baseline: str | None = None,
                 rebase: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """聚合步：读格表涉及各档的规格结果段与各片账本，写 ``delivery.json``（``v8-delivery/1``）并返回判定行。

    逐格：``delivered``（selected 且 rollout ok）、``candidates``（该格规格行数）、``tried``、``failed``（含超限）、
    ``exec_over_cap``、``backfills``（被递补上来并试过的备用候选）、``infra_retries``（账本）、``spares_left``、
    ``pending``（selected 但未跑）。某格 delivered ≠ 局数 即该格 FAIL（备用耗尽记 ``exhausted``），其余格照常统计；
    全部格相等才 ``V8_DELIVERY_SET=PASS``。

    ``rows`` 只含交付局（selected 且 rollout ok），每行必有 ``task tier episode candidate(==episode) seed exec_steps
    frames``（总帧数含演示帧）、``h5``（h5 绝对路径）、``path``（同一 h5 相对 ``delivery.json`` 所在目录，整树搬迁后
    仍可用）、``h5_sha256``、``env_module``，可选 ``video``（mp4 绝对路径）。逐格统计在 ``cells``
    （``{"<task>/<tier>": {...}}``），全局计数在 ``counts``；两处计数键全部显式写零。

    h5 核对：每个交付行的 h5 必须存在，否则该行记 ``h5_missing``、不进 ``rows``、该格 FAIL。``rebase``（整树搬迁后，
    如 GL 的 NFS 输出 rsync 回本机）给出时，规格 rollout 段里记录的 h5 路径按前缀替换到新位置（无前缀匹配记
    ``h5_unmapped``），替换后逐个重算 sha256 必须等于记录的 ``h5_sha256``（不符记 ``h5_sha_mismatch``）；
    ``h5``／``video``／``path`` 都按新位置写。``out_path`` 已存在即拒绝（不静默覆盖）。"""
    cells = check_cells(cells)
    out_path = Path(out_path)
    if out_path.exists():
        raise RolloutError(f"{out_path} 已存在，拒绝覆盖；换一个 --out／--output 新路径")
    loaded = load_v8_root(Path(specs_root), cells)
    infra, ledgers = _ledger_infra([Path(d) for d in ledger_dirs])
    stray_infra = sorted(k for k in infra if k not in cells)
    if stray_infra:
        raise RolloutError(f"账本里有格表之外的基础设施重试：{stray_infra}")
    totals = {key: 0 for key in V8_TOTAL_COUNT_KEYS}
    cell_out: dict[str, dict[str, Any]] = {}
    rows_out, over_rows, problems, bad_rows = [], [], [], []
    for (task, tier), expected in cells.items():
        mine = [r for r in loaded[tier][1] if r["task"] == task]
        roll = lambda r: r["rollout"] or {}  # noqa: E731
        c = {
            "expected": expected,
            "candidates": len(mine),
            "tried": sum(r["tried"] for r in mine),
            "delivered": sum(hard_specs.delivered(r) for r in mine),
            "failed": sum(roll(r).get("status") == "failed" for r in mine),
            "exec_over_cap": sum(roll(r).get("error_type") == "exec_over_cap" for r in mine),
            "backfills": sum(r["tried"] and not r["initial_selected"] for r in mine),
            "infra_retries": infra.get((task, tier), 0),
            "spares_left": sum(not r["tried"] and not r["selected"] for r in mine),
            "pending": sum(r["selected"] and r["rollout"] is None for r in mine),
            "bad_h5": 0,
        }
        bad_kinds: list[str] = []
        for r in sorted(mine, key=lambda r: int(r["candidate"])):
            rb = roll(r)
            if rb.get("error_type") == "exec_over_cap":
                over_rows.append({"task": task, "tier": tier, "candidate": int(r["candidate"]), "seed": int(r["seed"]),
                                  "exec_steps": rb.get("exec_steps")})
            if not hard_specs.delivered(r):
                continue
            recorded = rb.get("h5_path")
            h5_abs, bad = (os.path.abspath(recorded) if recorded else None), None
            if h5_abs and rebase:
                h5_abs = rebase_path(h5_abs, rebase)
                if h5_abs is None:
                    bad = "h5_unmapped"
            if bad is None and (not h5_abs or not Path(h5_abs).is_file()):
                bad = "h5_missing"
            if bad is None and rebase and _stream_sha256(Path(h5_abs)) != rb.get("h5_sha256"):
                bad = "h5_sha_mismatch"
            if bad is not None:
                c["bad_h5"] += 1
                bad_kinds.append(bad)
                bad_rows.append({"task": task, "tier": tier, "candidate": int(r["candidate"]), "seed": int(r["seed"]),
                                 "recorded_h5": recorded, "h5": h5_abs, "problem": bad})
                continue
            # 两方统一口径（主会话 2026-10-01）：``h5`` = h5 绝对路径（站点 S4-A 读）；``path`` = 同一 h5 相对
            # delivery.json 所在目录（v7 delivery 的键名，hard_parity import-delivery／step-headroom 读）
            row_out = {"task": task, "tier": tier, "candidate": int(r["candidate"]), "seed": int(r["seed"]),
                       "episode": int(r["episode"]), "spec_sha256": r["spec_sha256"],
                       "h5_sha256": rb.get("h5_sha256"), "frames": rb.get("frames"),
                       "exec_steps": rb.get("exec_steps"), "h5": h5_abs,
                       "path": os.path.relpath(h5_abs, os.path.abspath(out_path.parent)) if h5_abs else None,
                       # env_module 取自 runner 结果（REGISTERED_ENVS[task].cls.__module__），与 v7 delivery_rows 同取法
                       "env_module": rb.get("env_module"), "recovery_mode": None,
                       "initial_selected": bool(r["initial_selected"]), "role": rb.get("role")}
            if h5_abs:
                # 可选 video：只在该局显式目录的 videos/ 下找含 _seed<seed>_ 的唯一 mp4（找不到或不唯一就不写）
                videos = sorted((Path(h5_abs).parents[1] / "videos").glob(f"*_seed{int(r['seed'])}_*.mp4"))
                if len(videos) == 1:
                    row_out["video"] = str(videos[0])
            rows_out.append(row_out)
        if c["delivered"] == expected and not c["bad_h5"]:
            status, reason = "PASS", None
        elif c["pending"]:
            status, reason = "FAIL", "pending"
            totals["pending_cells"] += 1
        elif c["delivered"] != expected and (c["spares_left"] == 0 or (
                task in same_way_tasks_for(loaded[tier][0]) and not _same_way_spares(mine))):
            # 同方式递补的任务（v9 MoveCube）：失败局所在方式已无未试备用，即使别的方式还有备用也记 exhausted
            status, reason = "FAIL", "exhausted"
            totals["exhausted_cells"] += 1
        elif c["delivered"] != expected:
            status, reason = "FAIL", "shortfall"
        else:
            status, reason = "FAIL", bad_kinds[0]
        if status == "FAIL":
            totals["failed_cells"] += 1
            problems.append(f"{task}/{tier}:{reason}")
        for key in V8_CELL_COUNT_KEYS:
            totals[key] += c[key]
        cell_out[f"{task}/{tier}"] = {"task": task, "tier": tier, "status": status, "reason": reason, **c}
    ok = totals["failed_cells"] == 0 and totals["delivered"] == sum(cells.values()) and not bad_rows
    n_tasks = len({task for task, _ in cells})
    line = (f"V8_DELIVERY_SET={'PASS' if ok else 'FAIL'} tasks={n_tasks} cells={len(cells)} total={totals['delivered']} "
            f"expected={totals['expected']} failed={totals['failed']} exec_over_cap={totals['exec_over_cap']} "
            f"backfills={totals['backfills']} infra_retries={totals['infra_retries']} "
            f"exhausted_cells={totals['exhausted_cells']} pending_cells={totals['pending_cells']} "
            f"bad_h5={totals['bad_h5']}"
            + ("" if ok else f" problems={';'.join(problems[:12])}" + (";..." if len(problems) > 12 else "")))
    report = {
        "schema": V8_DELIVERY_SCHEMA, "specs_root": str(specs_root), "cells_source": cells_label,
        "cells_table": cells_json(cells), "code_baseline": code_baseline, "exec_cap": hard_specs.EXEC_CAP,
        "ledgers": ledgers, "rebase": [list(p) for p in (rebase or [])], "tasks": n_tasks, "cell_count": len(cells),
        "counts": totals, "cells": cell_out, "rows": rows_out, "bad_rows": bad_rows,
        "exec_over_cap_rows": over_rows, "problems": problems, "line": line,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def _same_way_spares(rows: list[dict[str, Any]]) -> int:
    """同方式递补口径下仍可用的备用数：落在「失败数多于递补数」（有缺口未补上）的方式里的未试备用行数。"""
    failed: dict[int | None, int] = {}
    backfilled: dict[int | None, int] = {}
    for r in rows:
        way = row_way(r)
        if r["tried"] and not r["selected"] and (r["rollout"] or {}).get("status") == "failed":
            failed[way] = failed.get(way, 0) + 1
        if not r["initial_selected"] and (r["tried"] or r["selected"]):
            backfilled[way] = backfilled.get(way, 0) + 1
    short = {way for way, n in failed.items() if n > backfilled.get(way, 0)}
    return sum(1 for r in rows if not r["tried"] and not r["selected"] and row_way(r) in short)


def run_continue_v8(specs_root: Path, cells: dict[tuple[str, str], int], output: Path, *, src_root: Path,
                    workers: int, gpu: str, pkg: str, code_baseline: str, resume: bool = False,
                    max_infra_retries: int = 1, batch_runner=None, cells_label: str = "custom") -> dict[str, Any]:
    """v8 gen1（单片或冒烟）：按格表逐档 ``run_continue``（每档一个规格文件、一把锁；只跑格表在该档的任务，
    逐任务配额、执行步上限、同格递补），账本 ``<output>/results.jsonl``；收尾 ``aggregate_v8`` 写
    ``<output>/delivery.json`` 并打印 ``V8_DELIVERY_SET``。某格备用耗尽只让该格 FAIL，其余格照常跑完。

    递补规则（v9 方案 §2.2 第 4 条）：MoveCube 在新值档只从同一 ``way_idx`` 的未试备用里递补，没有同方式备用即
    该格 ``exhausted``（``run_continue`` 按 /4 header 自动启用，见 ``same_way_tasks_for``）；其他任务不变。"""
    specs_root, output = Path(specs_root), Path(output)
    cells = check_cells(cells)
    loaded = load_v8_root(specs_root, cells)
    ledger = output / LEDGER_NAME
    if not resume and (_has_run_traces(output) or ledger.exists()):
        raise RolloutError(f"{output} 已有运行痕迹（episodes/、_rounds/ 或账本）；续跑用 --resume")
    output.mkdir(parents=True, exist_ok=True)
    per_tier = {}
    for tier in loaded:
        tasks = {task for task, t in cells if t == tier}
        per_tier[tier] = run_continue(specs_root / tier / "specs.jsonl", output, src_root=src_root, workers=workers,
                                      gpu=gpu, pkg=pkg, code_baseline=code_baseline, resume=resume,
                                      max_infra_retries=max_infra_retries, batch_runner=batch_runner, tasks=tasks,
                                      ledger=ledger, check_traces=False)
    delivery = output / "delivery.json"
    if delivery.exists():
        # 只有 --resume 能走到这里：上一次的交付清单改名留证（显式打印），不静默覆盖
        aside = delivery.with_name(f"delivery.json.prev-{int(time.time())}")
        delivery.rename(aside)
        print(f"# 上一次的 delivery.json 改名留证：{aside}", flush=True)
    report = aggregate_v8(specs_root, cells, [output], delivery, cells_label=cells_label,
                          code_baseline=code_baseline)
    print(report["line"], flush=True)
    summary = {"attempted": sum(s["attempted"] for s in per_tier.values()),
               "rounds": sum(s["rounds"] for s in per_tier.values()),
               "infra_retries": report["counts"]["infra_retries"], "delivered": report["counts"]["delivered"],
               "exec_over_cap": report["counts"]["exec_over_cap"], "backfills": report["counts"]["backfills"],
               "per_tier": per_tier, "delivery_set": report["line"]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def split_v8(frozen_root: Path, cells: dict[tuple[str, str], int], shard_out: Path, *, label: str) -> dict[str, Any]:
    """分片：从冻结根取格表涉及的格，逐档写 ``<shard_out>/specs/<tier>/specs.jsonl``（只含本片任务，按 /4 重签）
    与 ``<shard_out>/shard.json``（格表、来源各档 identity 与文件 sha）。冻结根必须未跑过；行与 sampling 逐字
    取自冻结根，合并时据此核对。"""
    from _freeze import write_jsonl_exclusive  # noqa: PLC0415

    cells = check_cells(cells)
    table = cell_table(cells)
    shard_out = Path(shard_out)
    if (shard_out / "specs").exists() or (shard_out / SHARD_META).exists():
        raise RolloutError(f"{shard_out} 已有分片规格，禁止覆盖")
    sources = {}
    for tier in cell_tiers(cells):
        path = Path(frozen_root) / tier / "specs.jsonl"
        header, rows = hard_specs.load_specs(path, expected_cells=table, check_fingerprint=False)
        if header["schema"] != hard_specs.SCHEMA or header["difficulty"] != tier:
            raise RolloutError(f"{path} 不是 {tier} 的 {hard_specs.SCHEMA}")
        if any(r["tried"] for r in rows):
            raise RolloutError(f"{path} 已跑过（有 tried 行），只能从未跑过的冻结根切片")
        want = [task for task in header["tasks"] if (task, tier) in cells]
        missing = sorted({task for task, t in cells if t == tier} - set(want))
        if missing:
            raise RolloutError(f"{path} 缺本片任务 {missing}")
        for task in want:
            if header["delivery_per_cell"][task] != cells[(task, tier)]:
                raise RolloutError(f"{task}/{tier} 冻结配额 {header['delivery_per_cell'][task]} ≠ 格表 {cells[(task, tier)]}")
        sub = copy.deepcopy(header)
        sub["tasks"] = want
        for key in ("per_env", "select_rule", "delivery_per_cell"):
            sub[key] = {task: header[key][task] for task in want}
        sub["sampling_config"] = {task: copy.deepcopy(header["sampling_config"][task]) for task in want}
        sub["sampling_config_sha256"] = hard_specs.digest(sub["sampling_config"])
        sub["run_id"] = f"{header['run_id']}-{label}"
        sub["draw_stats"] = {"split": {"label": label, "source_identity_sha256": header["identity_sha256"],
                                       "source_file_sha256": file_sha256(path)}}
        sub_rows = [copy.deepcopy(r) for r in rows if r["task"] in want]
        sub["identity_sha256"] = hard_specs.identity_sha256(sub, sub_rows)
        sub["delivery_sha256"] = hard_specs.delivery_sha256(sub_rows)
        hard_specs.validate_specs(sub, sub_rows, expected_cells=table)
        write_jsonl_exclusive(shard_out / "specs" / tier / "specs.jsonl", [sub, *sub_rows])
        sources[tier] = {"identity_sha256": header["identity_sha256"], "file_sha256": file_sha256(path)}
    meta = {"schema": V8_SHARD_SCHEMA, "label": label, "frozen_root": str(frozen_root), "cells": cells_json(cells),
            "sources": sources}
    (shard_out / SHARD_META).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    load_v8_root(shard_out / "specs", cells)
    print(f"V8_SPLIT=PASS label={label} tiers={len(sources)} cells={len(cells)} total={sum(cells.values())} "
          f"out={shard_out}", flush=True)
    return meta


def resolve_cells_from_json(data: dict[str, int]) -> dict[tuple[str, str], int]:
    cells = {}
    for key, n in data.items():
        task, _, tier = key.partition("@")
        cells[(task, tier)] = n
    return check_cells(cells)


# ── replay 模式 ─────────────────────────────────────────────────────────


def load_identities(path: Path) -> list[dict[str, Any]]:
    """身份清单：S4 ``final-delivery.json``（取 ``successes``）、gen1 的 ``delivery.json``（v8-delivery/1，取 ``rows``）
    或每行 ``{task, tier|difficulty, seed}`` 的 jsonl。"""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".json":
        items = json.loads(text)
        if isinstance(items, dict) and items.get("schema") == V8_DELIVERY_SCHEMA:
            items = items["rows"]  # gen1 的 delivery.json：交付行即身份清单
        elif isinstance(items, dict):
            items = items.get("successes", items)
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [{"task": i["task"], "tier": i.get("tier", i.get("difficulty")), "seed": int(i["seed"])} for i in items]


def run_replay(identities: list[dict[str, Any]], output: Path, *, src_root: Path, workers: int, gpu: str, pkg: str,
               resume: bool = False, specs_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    wanted = {(i["task"], i["tier"], i["seed"]) for i in identities}
    if len(wanted) != len(identities):
        raise RolloutError("身份清单有重复")
    if _has_run_traces(output) and not resume:
        raise RolloutError(f"{output} 已有运行痕迹（episodes/ 或 _rounds/）；续跑用 --resume")
    output.mkdir(parents=True, exist_ok=True)
    scheduled: set[tuple[str, str, int]] = set()
    results: list[dict[str, Any]] = []
    # 缺省读包内规格（档位跟随包内 TIERS）；显式给了规格路径（/4 规格根含 xhard5——阶段 3b 换包前它不在
    # 全局 TIERS 里）就只按给定档位读，不再混读包内其他档
    specs_paths = dict(specs_paths or {})
    tiers = tuple(specs_paths) if specs_paths else hard_specs.TIERS
    for tier in tiers:
        path = specs_paths[tier] if specs_paths else hard_specs.packaged_specs_path(tier)
        header, rows = load_specs_any(path)
        batch = [dict(r, _role="replay") for r in rows if (r["task"], tier, int(r["seed"])) in wanted]
        if not batch:
            continue
        scheduled |= {(r["task"], tier, int(r["seed"])) for r in batch}
        results.extend(run_batch(batch, header, output, 0, src_root=src_root, workers=workers, gpu=gpu, pkg=pkg,
                                 resume=resume))
    equal = scheduled == wanted
    print(f"REPLAY_SET={'PASS' if equal else 'FAIL'} scheduled={len(scheduled)} equal={len(scheduled & wanted)}"
          + ("" if equal else f" missing={sorted(wanted - scheduled)[:5]}"), flush=True)
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for record in sorted(results, key=lambda r: (tiers.index(r["tier"]), r["task"], r["candidate"])):
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"scheduled": len(scheduled), "wanted": len(wanted), "equal": equal, "ok": sum(r["ok"] for r in results),
               "failed": sum(not r["ok"] for r in results)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not equal:
        raise RolloutError("REPLAY_SET 不符")
    return summary
