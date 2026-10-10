#!/usr/bin/env python3
"""GL 席位客户端：一个席位一个常驻 Policy，从共享动态队列领局，逐局调评估包的 ``run_episode``（1008 拆分方案 §三
「scripts/evaluate.py 的调用链」与第二部分 §二 ``dev-scripts/gl/seat.py`` 行）。

与本机入口 ``scripts/evaluate.py`` 跑同一段循环，只是第 2 步「局表」换成动态队列、``run_episode`` 多传 ``expect``
（队列身份行，逐键核身份）::

    seat.py run --policy <模型> [--groundsg-variant V] --policy-seed N --identities <清单> --budget-ledger <账本>
                --trajectory-cap N --reset-cap N --shared-infra-cap N --expired-cap N --planned-first-tries N
                [--out <产物根>] [--seat <席位名>] [--infra-retries 0] [模型参数…]
    │
    ├─ 0. 打开共享预算账本（坏行、config 不符即 RUN_BLOCKED）→ 取本席位排他 lease → 核尝试账本路线 → 恢复上次崩溃留下的
    │     悬空尝试（补 attempt_end、结算共享账本 rid、补 accept 标记）
    ├─ 1. 队列里还有可领的局才 ``policy = load_policy(model, seed, **cfg)``（进程级一次）
    ├─ 2. 循环：在身份锁下「预约共享账本 → O_EXCL 建领取文件」→ ``run_episode(policy, dataset, task, ep, out,
    │        expect=身份行, attempt=N, ledger=reset 计量)`` → 结算（attempt_end、commit、accepted 标记）
    ├─ 3. ``policy.close()``（with 退出时）
    └─ 4. 读回队列：仍无 accepted 且已无人可领的身份计 missing（>0 打印 RUN_INCOMPLETE、退出 6）

本文件只保留席位层：``SeatRunner``（领局、结算、恢复）、持久尝试账本 ``AttemptLedger``、动态队列 ``DynamicQueue``、
本席位 lease、``progress.json``、``recover_dangling``／``recover_crash_window``。``EnvSession``、单局流程与看门狗在
``robomme_ood_eval.session``／``robomme_ood_eval.episode``，模型在 ``robomme_ood_eval.models``（经 ``load_policy``）。

**动态队列**（``<queue>/``，缺省 ``<out>/queue/<标签>/seed<n>/``；同一策略同一种子的全部席位共用一份）：

* 身份清单（``export_eval_identities.py`` 的 JSONL，或 ``eval_manifest.py`` 的分片 JSON 数组）每行一个身份；队列键
  ``<数据集>__<key>``；
* 每身份每次尝试一个领取文件 ``claims/<队列键>.a<N>.json``（``O_EXCL`` 创建；内容 pid、host、seat、心跳时间
  ``t_heartbeat``、收尾后 ``ended``／``end_status``）；领取与预约在身份锁 ``locks/<队列键>.lock``（``flock``）下进行，
  并发争同一身份只有一方领到；
* ``run_episode`` 把 ``result.json`` 原子写完后，在 ``accepted/<队列键>.json`` 落标记（``O_EXCL``，第一份即权威）；
* 领取文件心跳超过「2 倍单局墙钟」未更新（进程已死）才可回收；同一席位名的旧进程留下的未收尾领取（本席位 lease
  保证同名席位只有一个进程）视为中断、当即回收；回收计一次尝试；每身份至多 ``1 + --infra-retries`` 次尝试，重试名额
  另向共享账本 ``claim_retry`` 原子领取；
* 领取前先在共享账本 ``reserve``（token ``<路线>|<key>|a<N>``，首试 ``first``、重试 ``recovery``），超额硬拒：
  打印 ``RUN_BLOCKED reason=budget``、退出 5、不建领取文件。

**墙钟超时不再留下悬空 rid**：``run_episode`` 的看门狗到点时经 ``episode.HARD_EXIT`` 退出；席位把它换成
``SeatRunner._hard_exit``——先写本地 attempt_end（infra）、``commit`` 共享账本 rid、给领取文件记收尾，再 ``os._exit(75)``；
万一这一步也没做完，重启后的 ``recover_dangling`` 照样结算（commit 幂等）。

退出码：0 全部身份已有 accepted（或其余身份正由别的席位跑）；3 运行阻塞（参数、身份不符、服务端死／不符、Astra 报停、
账本坏行／config 不符、lease 被占、路线不符）；5 预算额度不足（共享账本拒绝预约或 reset 计量耗尽）；6 跑完仍有身份无
accepted 且已无尝试名额（``RUN_INCOMPLETE``）；75 单局墙钟／阶段期限超时（基础设施超时，已记录；外层
``run_eval_gl.sh`` 按重起次数决定是否重起客户端，本轮为 0）。

本轮基础设施重试次数为 0（``--infra-retries`` 缺省 0）：正常任务失败不重跑；基础设施故障不重试，留给用户裁决。
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

XHARD0 = "xhard0"
#: 两个评估数据集：新值档与官方 test 的 hard 子集（xhard0）
OOD = "ood"
HARD_VERIFY = "hard-verify"
DATASETS = (OOD, HARD_VERIFY)
#: GroundSG 三个子目标来源变体
GROUNDSG_VARIANTS = ("ground-sg-oracle", "ground-sg-qwenvl", "ground-sg-memer")
#: 变体 → 必须且只能给的 adapter 参数（argparse dest, CLI 名）；oracle 两者都不许给
VARIANT_ADAPTERS = {"ground-sg-qwenvl": ("qwenvl_groundsg_adapter", "--qwenvl-groundsg-adapter"),
                    "ground-sg-memer": ("memer_adapter", "--memer-adapter")}
#: CLI 必填的预算参数（argparse dest, CLI 名）；缺任一即 RUN_BLOCKED reason=budget_args（不回落账本常量默认值）
BUDGET_ARGS = (("budget_ledger", "--budget-ledger"), ("trajectory_cap", "--trajectory-cap"),
               ("reset_cap", "--reset-cap"), ("shared_infra_cap", "--shared-infra-cap"),
               ("expired_cap", "--expired-cap"), ("planned_first_tries", "--planned-first-tries"))
#: 本席位 progress.json 的具名阶段
PHASES = ("context_load", "claim", "episode", "done", "finished")
EXIT_BLOCKED = 3
EXIT_BUDGET = 5
EXIT_INCOMPLETE = 6
EXIT_WALL = 75
DEFAULT_SHUFFLE_SEED = 20260930
TERMINAL_STATUSES = ("success", "fail", "timeout")
#: 本轮基础设施重试次数（拆分方案 R5：0 次，失败即停交用户）
DEFAULT_INFRA_RETRIES = 0
#: 领取文件心跳间隔（秒）与过期倍数（心跳超过「倍数 × 单局墙钟」未更新才可回收）
DEFAULT_HEARTBEAT_S = 30.0
EXPIRE_FACTOR = 2.0
#: 执行身份行必须恰有的字段（``dataset`` 另由行或 ``--dataset`` 给出）
V8_IDENTITY_KEYS = ("task", "tier", "seed", "candidate", "builder_episode", "source_episode", "spec_sha256", "key")
#: 「未提供」哨兵：可选参数缺省时取环境变量；None 表示显式关闭
_UNSET = object()
#: 共享预算账本的门控环境变量；值为账本路径，空或未设即关闭（CLI 的 --budget-ledger 必填，进程内调用才走它）
ENV_BUDGET_LEDGER = "SGEVAL_BUDGET_LEDGER"
#: Slurm 到期证据：文件路径，内容为到期（TIMEOUT）作业号
ENV_EXPIRED_JOBS = "SGEVAL_EXPIRED_JOBS"
#: 悬空尝试最后活动时刻距 Slurm 结束时刻在此秒数内且结束时刻已过 → 视为到期中断
EXPIRE_MARGIN_S = 900.0
#: 每次尝试预约的 reset 计量（build 与 reset 各 1 次）
NEW_SIDE_RESETS_PER_ATTEMPT = 2
INTERRUPTS = ("infra", "expired")


class SeatStop(Exception):
    """席位整批停下（运行阻塞 3、额度不足 5）：已记录，调用方按 ``code`` 退出。"""

    def __init__(self, code: int, msg: str = ""):
        super().__init__(msg)
        self.code = int(code)


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def load_sibling(name: str, alias: str | None = None):
    """按文件路径加载本目录下的模块（已加载则复用；加载失败不留空壳）。"""
    alias = alias or name
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return mod


def official_defs():
    """评估包的别名表 ``robomme_ood_eval/models/_official_defs.py``（模块名 ``official_defs``，已加载则复用）。"""
    mod = sys.modules.get("official_defs")
    if mod is not None and hasattr(mod, "canonical_row"):
        return mod
    path = REPO / "src" / "robomme_ood_eval" / "models" / "_official_defs.py"
    spec = importlib.util.spec_from_file_location("official_defs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["official_defs"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("official_defs", None)
        raise
    return mod


def read_results(path: Path) -> list[dict]:
    """读 JSONL 行；崩溃留下的半行跳过；历史行的旧策略标签／数据集名／路线经 ``canonical_row`` 映射成官方名。"""
    canon = official_defs().canonical_row
    rows = []
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(canon(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return rows


def append_result(path: Path, record: dict) -> None:
    """追加一行并 fsync；前一行是崩溃留下的半行时先补换行。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((dumps(record) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(dumps(obj) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


# ── 持久尝试账本 ────────────────────────────────────────────────────────────


def _int_or_none(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


class ResetBudgetExhausted(RuntimeError):
    """本席位账本的 reset 额度（build 与 reset 各算一次）已用尽（只在给了 ``--reset-budget`` 时可能）。"""

    budget_exhausted = True


def _open_shared(shared: Any, caps: dict | None = None):
    """共享账本门控：``_UNSET`` → 取环境变量 ``SGEVAL_BUDGET_LEDGER``（空即关闭）；None／空串 → 关闭；路径 → 打开
    ``budget_ledger.BudgetLedger``（``caps`` 给出 config 参数时一并传入）；其他对象（已打开的账本，单测注入）原样返回。"""
    if shared is _UNSET:
        shared = os.environ.get(ENV_BUDGET_LEDGER) or None
    if shared is None or shared == "":
        return None
    if isinstance(shared, (str, Path)):
        return load_sibling("budget_ledger").BudgetLedger(shared, **(caps or {}))
    return shared


class AttemptLedger:
    """本席位的持久尝试账本（JSONL，追加写 + fsync）。一个账本只有一个写者（一个席位名一个客户端进程，lease 保证）。

    行：``{"t","kind","key","attempt_id","attempt_no","seat","policy",...}``，kind ∈
    ``budget | budget_raise | attempt_start | reset_claim | attempt_end | accept``。全部计数从账本内容推出（进程重启
    读回，不刷新）：

    * reset 额度：``reset_claim`` 行数对 ``--reset-budget``（不给只计量）；
    * 作废尝试：``attempt_end.budget_exhausted=true`` 的尝试没有执行段，不计入尝试上限、也不计入 infra 重试额度；
    * infra 重试：``attempt_start.retry=true`` 且未作废的尝试数；
    * 本席位的 accept：每身份第一条 ``accept`` 行（跨席位的权威是队列 ``accepted/`` 标记）。

    共享模式（``shared`` 给出账本路径／``budget_ledger.BudgetLedger`` 对象，或缺省时环境变量 ``SGEVAL_BUDGET_LEDGER``
    非空；``shared=None`` 显式关闭）：infra 重试名额改由共享账本 ``claim_retry`` 原子领取；``attempt_start`` 额外记
    ``route``、``slurm_job_id``、``slurm_end_time``；``classify_interrupt`` 按证据把悬空尝试分为 ``expired`` 或 ``infra``。
    """

    def __init__(self, path: Path | str, *, seat: str, policy: str, shared: Any = _UNSET, route: Any = _UNSET,
                 expired_jobs: Any = _UNSET, caps: dict | None = None):
        self.path = Path(path)
        self.seat = str(seat)
        self.policy = str(policy)
        self.shared = _open_shared(shared, caps)
        self.route = f"{self.policy}/new" if route is _UNSET or route is None else str(route)
        self._expired_jobs = expired_jobs
        self.last_t: dict[str, float] = {}
        self.reset_budget: int | None = None
        self.infra_retry_budget: int | None = None
        self._reset_metering = False
        self.reset_claims = 0
        self.budget_hist_max: int | None = None
        self.starts: dict[str, list[dict]] = {}
        self.ended: dict[str, dict] = {}
        self.accepted: dict[str, str] = {}
        self._lock = threading.Lock()
        for row in read_results(self.path):
            self._apply(row)

    def _apply(self, row: dict) -> None:
        kind = row.get("kind")
        aid = row.get("attempt_id")
        if aid and isinstance(row.get("t"), (int, float)):
            self.last_t[aid] = max(self.last_t.get(aid, 0.0), float(row["t"]))
        if kind == "reset_claim":
            self.reset_claims += 1
        elif kind == "budget":
            if row.get("reset_budget") is not None:
                self.budget_hist_max = max(self.budget_hist_max or 0, int(row["reset_budget"]))
        elif kind == "budget_raise":
            self.budget_hist_max = max(self.budget_hist_max or 0, int(row["to"]))
        elif kind == "attempt_start":
            self.starts.setdefault(row["key"], []).append(row)
        elif kind == "attempt_end":
            self.ended[row["attempt_id"]] = row
        elif kind == "accept":
            self.accepted.setdefault(row["key"], row["accepted_attempt_id"])

    def _void(self, attempt_id: str) -> bool:
        end = self.ended.get(attempt_id)
        return bool(end and end.get("budget_exhausted"))

    def append(self, row: dict) -> dict:
        row = {"t": time.time(), "seat": self.seat, "policy": self.policy, **row}
        with self._lock:
            append_result(self.path, row)
            self._apply(row)
        return row

    def start(self, reset_budget: int | None, infra_retry_budget: int, *, reason: str = "cli_reset_budget") -> None:
        """进程启动：记预算；命令行额度大于账本历史最大值即记一次提升。reset_budget=None：只计量。"""
        reset_budget = None if reset_budget is None else int(reset_budget)
        infra_retry_budget = int(infra_retry_budget)
        prev = self.budget_hist_max
        if prev is not None and reset_budget is not None and reset_budget > prev:
            self.append({"kind": "budget_raise", "from": prev, "to": reset_budget, "reason": reason})
            print(f"RESET_BUDGET_RAISE from={prev} to={reset_budget} reason={reason}", flush=True)
        self.append({"kind": "budget", "reset_budget": reset_budget, "infra_retry_budget": infra_retry_budget,
                     "pid": os.getpid(), "host": socket.gethostname()})
        self.reset_budget, self.infra_retry_budget = reset_budget, infra_retry_budget
        self._reset_metering = reset_budget is None

    @property
    def reset_enforced(self) -> bool:
        return not self._reset_metering

    def reset_left(self) -> float:
        if not self.reset_enforced:
            return float("inf")
        return int(self.reset_budget or 0) - self.reset_claims

    def attempts_total(self, key: str) -> int:
        return len(self.starts.get(key, []))

    def attempts_used(self, key: str) -> int:
        return sum(not self._void(r["attempt_id"]) for r in self.starts.get(key, []))

    def infra_retries_used(self) -> int:
        return sum(bool(r.get("retry")) and not self._void(r["attempt_id"])
                   for rows in self.starts.values() for r in rows)

    def infra_retries_left(self) -> int:
        if self.shared is not None:
            return self.shared.shared_infra_cap - self.shared.state().retries_of("infra")
        return int(self.infra_retry_budget or 0) - self.infra_retries_used()

    def retry_interrupt(self, key: str) -> str:
        live = [r for r in self.starts.get(key, []) if not self._void(r["attempt_id"])]
        end = self.ended.get(live[-1]["attempt_id"]) if live else None
        it = (end or {}).get("interrupt")
        return it if it in INTERRUPTS else "infra"

    def interrupt_counts(self, key: str | None = None) -> dict[str, int]:
        out = {i: 0 for i in INTERRUPTS}
        for k, rows in self.starts.items():
            if key is not None and k != key:
                continue
            for r in rows:
                end = self.ended.get(r["attempt_id"])
                if end is None or self._void(r["attempt_id"]) or self.is_final(end):
                    continue
                it = end.get("interrupt")
                out[it if it in INTERRUPTS else "infra"] += 1
        return out

    def allow_retry(self, key: str, *, token: str | None = None, interrupt: str | None = None) -> bool:
        """是否可以对 key 再开一次重试：非共享模式按本地额度；共享模式向共享账本原子领一个名额（token 幂等）。"""
        if self.shared is None:
            return self.infra_retries_left() > 0
        kw = {"token": token} if token else {}
        return bool(self.shared.claim_retry(route=self.route, key=key,
                                            interrupt=interrupt or self.retry_interrupt(key),
                                            max_attempts=1 + int(self.infra_retry_budget or 0),
                                            seat=self.seat, policy=self.policy, **kw))

    def expired_jobs(self) -> set[str]:
        src = self._expired_jobs
        if src is _UNSET:
            src = os.environ.get(ENV_EXPIRED_JOBS) or None
        if src is None:
            return set()
        if isinstance(src, (str, Path)):
            p = Path(src)
            if not p.is_file():
                return set()
            toks = p.read_text(encoding="utf-8").replace(",", " ").split()
        else:
            toks = [str(x) for x in src]
        return {t.split(".")[0] for t in toks if t.strip()}

    def classify_interrupt(self, start: dict, *, now: float | None = None) -> tuple[str, str]:
        now = time.time() if now is None else float(now)
        job = start.get("slurm_job_id")
        if job and str(job).split(".")[0] in self.expired_jobs():
            return "expired", f"sacct_timeout job={job}"
        end_t = start.get("slurm_end_time")
        if isinstance(end_t, (int, float)) and now >= end_t:
            last = self.last_t.get(start["attempt_id"], float(start.get("t") or 0.0))
            if last >= end_t - EXPIRE_MARGIN_S:
                return "expired", f"slurm_end_time={int(end_t)} last_activity={last:.0f}"
        return "infra", "no_expiry_evidence"

    def last_end_final(self, key: str) -> bool:
        live = [r for r in self.starts.get(key, []) if not self._void(r["attempt_id"])]
        if not live:
            return False
        end = self.ended.get(live[-1]["attempt_id"])
        return bool(end) and self.is_final(end)

    def dangling(self) -> list[dict]:
        """有 attempt_start、无 attempt_end 的尝试（进程被杀等）。"""
        return [r for rows in self.starts.values() for r in rows if r["attempt_id"] not in self.ended]

    def final_without_accept(self) -> list[dict]:
        """崩溃窗口：最后一次未作废尝试的 attempt_end 已是最终结局、该身份却没有 accept。"""
        out = []
        for key, rows in self.starts.items():
            if key in self.accepted:
                continue
            live = [r for r in rows if not self._void(r["attempt_id"])]
            if not live:
                continue
            end = self.ended.get(live[-1]["attempt_id"])
            if end and self.is_final(end):
                out.append(end)
        return out

    def recover_accept(self, end: dict) -> dict:
        return self.append({"kind": "accept", "key": end["key"], "attempt_id": end["attempt_id"],
                            "attempt_no": end.get("attempt_no"), "accepted_attempt_id": end["attempt_id"],
                            "status": end.get("status"), "recovered": True})

    def claim_reset(self, *, key: str, attempt_id: str, attempt_no: int, what: str, canary: bool = False) -> None:
        with self._lock:
            if self.reset_enforced and self.reset_claims >= int(self.reset_budget or 0):
                raise ResetBudgetExhausted(f"reset 额度耗尽 claims={self.reset_claims} budget={self.reset_budget}")
            row = {"t": time.time(), "seat": self.seat, "policy": self.policy, "kind": "reset_claim", "key": key,
                   "attempt_id": attempt_id, "attempt_no": attempt_no, "what": what, "canary": bool(canary),
                   "n": self.reset_claims + 1}
            append_result(self.path, row)
            self._apply(row)

    def attempt_start(self, *, key: str, attempt_id: str, attempt_no: int, retry: bool, **extra) -> None:
        if self.shared is not None:
            extra = {"route": self.route, "host": socket.gethostname(),
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID") or None,
                     "slurm_end_time": _int_or_none(os.environ.get("SLURM_JOB_END_TIME")), **extra}
        self.append({"kind": "attempt_start", "key": key, "attempt_id": attempt_id, "attempt_no": attempt_no,
                     "retry": bool(retry), **extra})

    @staticmethod
    def is_final(record: dict) -> bool:
        """最终结局：success／fail／timeout，以及非基础设施错误（status=error、infra=false、非 budget_exhausted、
        非 run_blocked）。infra=true 的错误不是结局。"""
        status = record.get("status")
        if status in TERMINAL_STATUSES:
            return True
        return (status == "error" and not record.get("infra") and not record.get("budget_exhausted")
                and not record.get("run_blocked"))

    def is_late(self, record: dict) -> bool:
        return self.is_final(record) and record["key"] in self.accepted

    def attempt_end(self, record: dict) -> bool:
        """写 attempt_end；最终结局且本账本里该身份尚无 accept 时再写 accept。返回 late。"""
        key, status = record["key"], record.get("status")
        late = self.is_late(record)
        base = {"key": key, "attempt_id": record["attempt_id"], "attempt_no": record["attempt_no"]}
        self.append({"kind": "attempt_end", **base, "status": status, "infra": bool(record.get("infra")),
                     "cap_hit": bool(record.get("cap_hit")), "exec_steps": record.get("exec_steps"),
                     "budget_exhausted": bool(record.get("budget_exhausted")), "late": late,
                     **({"run_blocked": True} if record.get("run_blocked") else {}),
                     **({"infra_reason": record["infra_reason"]} if record.get("infra_reason") else {}),
                     **({"recovered": True} if record.get("recovered") else {}),
                     **({"interrupt": record["interrupt"], "interrupt_evidence": record.get("interrupt_evidence")}
                        if record.get("interrupt") else {})})
        if self.is_final(record) and not late:
            self.append({"kind": "accept", **base, "accepted_attempt_id": record["attempt_id"], "status": status})
        return late


# ── 身份 ────────────────────────────────────────────────────────────────────


def v8_key(row: dict) -> str:
    """身份键 ``<task>_<tier>_<seed>``：结果、账本、队列统一用它。"""
    return f"{row['task']}_{row['tier']}_{int(row['seed'])}"


def key_of(row: dict) -> str:
    if row.get("key"):
        return str(row["key"])
    return v8_key(row)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_identity(row: dict, dataset: str | None = None) -> str | None:
    """执行身份行的结构核对（字段齐全、nullable 严格、key 自洽）；不符返回说明。数据集取 ``dataset`` 参数，缺省取行内
    ``dataset`` 字段：

    * ood：spec_sha256 为 64 位串，candidate 为整数或 null；
    * hard-verify：tier=="xhard0"、candidate 与 spec_sha256 为 null、source_episode 为整数。"""
    dataset = dataset if dataset is not None else row.get("dataset")
    if dataset not in DATASETS:
        return f"dataset={dataset!r}"
    if row.get("dataset") is not None and row["dataset"] != dataset:
        return f"dataset 行内={row['dataset']!r} 期望={dataset!r}"
    bad = []
    missing = [k for k in V8_IDENTITY_KEYS if k not in row]
    if missing:
        return f"missing_fields={missing}"
    if not _is_int(row["seed"]) or not _is_int(row["builder_episode"]):
        bad.append("seed/builder_episode 非整数")
    if row["candidate"] is not None and not _is_int(row["candidate"]):
        bad.append(f"candidate={row['candidate']!r}")
    if row["source_episode"] is not None and not _is_int(row["source_episode"]):
        bad.append(f"source_episode={row['source_episode']!r}")
    if dataset == HARD_VERIFY:
        if row["tier"] != XHARD0:
            bad.append(f"tier={row['tier']!r} 不是 {XHARD0}")
        if row["candidate"] is not None:
            bad.append(f"candidate={row['candidate']!r} 须为 null")
        if row["spec_sha256"] is not None:
            bad.append(f"spec_sha256={row['spec_sha256']!r} 须为 null")
        if not _is_int(row["source_episode"]):
            bad.append(f"source_episode={row['source_episode']!r} 须为整数")
    elif not isinstance(row["spec_sha256"], str) or len(row["spec_sha256"]) != 64:
        bad.append(f"spec_sha256={row['spec_sha256']!r}")
    if not bad and row["key"] != v8_key(row):
        bad.append(f"key={row['key']} want={v8_key(row)}")
    return "; ".join(bad) or None


#: 旧名（第三阶段 env_client 的接口），两参数形式不变
validate_v8_identity = validate_identity


def order_identities(rows: list[dict], order: str, shuffle_seed: int) -> list[dict]:
    rows = list(rows)
    if order == "reverse":
        rows.reverse()
    elif order == "shuffle":
        rows.sort(key=lambda r: (r.get("dataset") or "", r["task"], r["tier"], int(r["seed"])))
        random.Random(shuffle_seed).shuffle(rows)
    return rows


def read_identity_file(path: Path) -> list[dict]:
    """身份清单：JSON 数组（``eval_manifest.py`` 的 shard-NN.json）或 JSONL（``export_eval_identities.py``）。"""
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        data = json.loads(text)
        return [r for r in data if isinstance(r, dict)]
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def identity_for_queue(row: dict, dataset: str | None) -> dict:
    """执行身份行（字段契约 C1）+ ``dataset``；``export_eval_identities`` 多出的 ``episode`` 字段去掉。"""
    d = {k: row[k] for k in V8_IDENTITY_KEYS if k in row}
    d["dataset"] = row.get("dataset") or dataset
    return d


def load_identities(args) -> list[dict]:
    """读身份清单，行内没有 ``dataset`` 时取 ``--dataset``；``--dataset`` 同时作过滤（只跑该数据集的行）；按
    ``--only`` 过滤、``--order`` 排序、``--limit`` 截断。每行须为合法执行身份行、(dataset, key) 不重复，否则运行阻塞。"""
    want_ds = getattr(args, "dataset", None)
    rows = []
    for r in read_identity_file(Path(args.identities)):
        if want_ds and r.get("dataset") not in (None, want_ds):
            continue
        rows.append(identity_for_queue(r, want_ds))
    bad = [(i, b) for i, r in enumerate(rows) for b in [validate_identity(r)] if b]
    keys = [(r.get("dataset"), key_of(r)) for r in rows]
    dup = len(keys) - len(set(keys))
    if bad or dup or not rows:
        print(f"RUN_BLOCKED reason=identities n_rows={len(rows)} n_bad={len(bad)} dup_keys={dup} first={bad[:3]}",
              flush=True)
        raise SeatStop(EXIT_BLOCKED, "identities")
    if getattr(args, "only", None):
        want = set(args.only.split(","))
        rows = [r for r in rows if key_of(r) in want]
    rows = order_identities(rows, getattr(args, "order", "forward"), getattr(args, "shuffle_seed", DEFAULT_SHUFFLE_SEED))
    if getattr(args, "limit", 0):
        rows = rows[: args.limit]
    return rows


# ── 路线、变体、预算参数 ─────────────────────────────────────────────────────


def policy_seed_of(args) -> int | None:
    v = getattr(args, "policy_seed", None)
    return None if v is None else int(v)


def policy_route(args) -> str:
    """共享账本路线名：``<模型>/seed<n>/new``，GroundSG 为 ``groundsg/<variant>/seed<n>/new``。"""
    seed = policy_seed_of(args)
    head = f"groundsg/{getattr(args, 'groundsg_variant', None)}" if args.policy == "groundsg" else args.policy
    return f"{head}/new" if seed is None else f"{head}/seed{seed}/new"


def queue_label(args) -> str:
    """队列与席位目录的标签：模型名，GroundSG 为 ``groundsg-<variant>``。"""
    return f"groundsg-{getattr(args, 'groundsg_variant', None)}" if args.policy == "groundsg" else str(args.policy)


def policy_variant_of(args) -> str | None:
    return getattr(args, "groundsg_variant", None) if args.policy == "groundsg" else None


def variant_problems(args, *, check_dirs: bool = False) -> list[str]:
    """GroundSG 变体与 adapter 的配对核对：qwenvl 必须且只能给 --qwenvl-groundsg-adapter，memer 必须且只能给
    --memer-adapter，oracle 两者都不许给；``check_dirs`` 时再核 adapter 目录存在。"""
    bad = []
    variant = getattr(args, "groundsg_variant", None)
    if args.policy == "groundsg":
        if variant not in GROUNDSG_VARIANTS:
            bad.append(f"--policy groundsg 必须给 --groundsg-variant {'／'.join(GROUNDSG_VARIANTS)}")
    elif variant is not None:
        bad.append("--groundsg-variant 只能与 --policy groundsg 同用")
    for v, (dest, flag) in VARIANT_ADAPTERS.items():
        given = getattr(args, dest, None)
        if variant == v and not given:
            bad.append(f"--groundsg-variant {v} 必须给 {flag}")
        if given and variant != v:
            bad.append(f"{flag} 只能与 --groundsg-variant {v} 同用")
        if check_dirs and given and variant == v and not Path(given).is_dir():
            bad.append(f"{flag} 目录不存在：{given}")
    return bad


def entry_blockers(args) -> tuple[str, str] | None:
    """CLI ``run`` 入口的具名拦截：``--policy-seed``（必填、非负整数）、六个预算参数（必填，不回落常量默认）、GroundSG
    变体与 adapter 配对（含目录存在）、``--infra-retries`` 非负。返回 (reason, detail) 或 None。"""
    seed = getattr(args, "policy_seed", None)
    if seed is None or not _is_int(seed) or seed < 0:
        return "policy_seed", f"--policy-seed 必填且为非负整数（现为 {seed!r}）"
    miss = [flag for dest, flag in BUDGET_ARGS if getattr(args, dest, None) in (None, "")]
    if miss:
        return "budget_args", f"必须给 {' '.join(miss)}（不回落 budget_ledger.py 常量默认值）"
    neg = [flag for dest, flag in BUDGET_ARGS[1:] if getattr(args, dest) < 0]
    if neg:
        return "budget_args", f"{' '.join(neg)} 须为非负整数"
    vp = variant_problems(args, check_dirs=True)
    if vp:
        return "variant_pairing", "; ".join(vp)
    if not 0 <= int(getattr(args, "infra_retries", 0) or 0) <= 19:
        return "args", "--infra-retries 须为 0 至 19 的整数（每身份至多 20 次尝试）"
    return None


def budget_caps(args) -> dict | None:
    """``BudgetLedger`` 的构造参数：轨迹、共享 infra、到期续跑、计划首试四项全部给出才返回（否则 None，沿用账本缺省）；
    ``reset_cap`` 给了才带上（原侧驱动的 ``open_budget_ledger`` 不传它，两侧同用账本缺省时 config 仍一致）。CLI 入口
    由 ``entry_blockers`` 要求六项都给。"""
    vals = {dest: getattr(args, dest, None) for dest, _ in BUDGET_ARGS[1:] if dest != "reset_cap"}
    if any(v is None for v in vals.values()):
        return None
    out = {k: int(v) for k, v in vals.items()}
    if getattr(args, "reset_cap", None) is not None:
        out["reset_cap"] = int(args.reset_cap)
    return out


def shard_id_of(route: str, seat: Any) -> str:
    """本席位排他 lease 的 id：``<路线>--seat-<席位名>``（非 [A-Za-z0-9._-] 换成 ``_``）。动态队列下身份不再按分片
    切给席位，lease 只保证同一席位名同时只有一个客户端进程（本席位的尝试账本只有一个写者）。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{route}--seat-{seat}")


def default_seat() -> str:
    """缺省席位名：``<主机>-<作业号或 local>-gpu<CUDA_VISIBLE_DEVICES>``（重起客户端时不变）。"""
    job = os.environ.get("SLURM_JOB_ID") or "local"
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "na").replace(",", "+")
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", f"{socket.gethostname().split('.')[0]}-{job}-gpu{gpu}")


# ── 动态队列 ────────────────────────────────────────────────────────────────


def queue_key(ident: dict) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{ident['dataset']}__{key_of(ident)}")


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


@dataclass
class Claim:
    """一次领取：身份、尝试号、领取文件、预约 rid 与 token、是否重试及其中断分类。"""

    ident: dict
    attempt: int
    path: Path
    token: str
    rid: str | None
    retry: bool
    interrupt: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return key_of(self.ident)


@dataclass
class QueueState:
    """一个身份在队列里的状态：已接受？最后一次领取（尝试号、内容、是否在跑／过期／本席位旧进程留下）。"""

    accepted: dict | None
    attempts: list[tuple[int, dict]]
    live: bool = False
    expired: bool = False
    stale_mine: bool = False

    @property
    def last(self) -> tuple[int, dict] | None:
        return self.attempts[-1] if self.attempts else None


class DynamicQueue:
    """共享动态队列（见模块文档）。``now`` 可注入（单测）。"""

    CLAIM_RE = re.compile(r"^(?P<q>.+)\.a(?P<n>[1-9]\d*)\.json$")

    def __init__(self, root: Path | str, *, seat: str, wall_s: float, max_attempts: int = 1,
                 now: Callable[[], float] = time.time):
        self.root = Path(root)
        self.claims = self.root / "claims"
        self.accepted_dir = self.root / "accepted"
        self.locks = self.root / "locks"
        for d in (self.claims, self.accepted_dir, self.locks):
            d.mkdir(parents=True, exist_ok=True)
        self.seat = str(seat)
        self.wall_s = float(wall_s)
        self.max_attempts = max(1, int(max_attempts))
        self.now = now

    @property
    def expire_s(self) -> float:
        return EXPIRE_FACTOR * self.wall_s

    @contextlib.contextmanager
    def lock(self, ident: dict):
        fd = os.open(self.locks / f"{queue_key(ident)}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def accepted_path(self, ident: dict) -> Path:
        return self.accepted_dir / f"{queue_key(ident)}.json"

    def claim_path(self, ident: dict, n: int) -> Path:
        return self.claims / f"{queue_key(ident)}.a{int(n)}.json"

    def state(self, ident: dict) -> QueueState:
        q = queue_key(ident)
        attempts = []
        for p in self.claims.glob(f"{q}.a*.json"):
            m = self.CLAIM_RE.match(p.name)
            if not m or m.group("q") != q:
                continue
            attempts.append((int(m.group("n")), read_json(p) or {"corrupt": True}))
        attempts.sort(key=lambda x: x[0])
        st = QueueState(accepted=read_json(self.accepted_path(ident)), attempts=attempts)
        if st.last is not None:
            _n, doc = st.last
            if not doc.get("ended"):
                beat = doc.get("t_heartbeat") or doc.get("t_claim") or 0.0
                mine = doc.get("seat") == self.seat and doc.get("pid") != os.getpid()
                if mine:
                    st.stale_mine = True
                elif self.now() - float(beat) > self.expire_s:
                    st.expired = True
                else:
                    st.live = True
        return st

    def claimable(self, ident: dict) -> bool:
        """只读预判（不加锁、不预约）：未接受，且没有在跑的领取，且还有尝试名额。"""
        st = self.state(ident)
        if st.accepted is not None or st.live:
            return False
        return st.last is None or st.last[0] < self.max_attempts

    def end_claim(self, path: Path, status: str, **extra) -> None:
        """给领取文件记收尾（原子替换内容）；文件已被作废删除时不做事。"""
        doc = read_json(path)
        if doc is None:
            return
        doc.update(ended=True, end_status=status, t_end=self.now(), **extra)
        write_json_atomic(path, doc)

    def void_claim(self, path: Path) -> None:
        """作废一次领取（局还没开始：服务端拒绝、额度不足、身份不符）：删领取文件，不计尝试。"""
        with contextlib.suppress(FileNotFoundError):
            Path(path).unlink()

    def heartbeat(self, path: Path) -> None:
        doc = read_json(path)
        if doc is None or doc.get("ended"):
            return
        doc["t_heartbeat"] = self.now()
        write_json_atomic(path, doc)

    def accept(self, ident: dict, *, attempt: int, status: str, result: str | None, **extra) -> bool:
        """``accepted/<队列键>.json``（O_EXCL，第一份即权威）；已存在返回 False。"""
        p = self.accepted_path(ident)
        doc = {"key": key_of(ident), "dataset": ident["dataset"], "attempt": int(attempt), "status": status,
               "result": result, "seat": self.seat, "host": socket.gethostname(), "pid": os.getpid(),
               "t": self.now(), **extra}
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dumps(doc) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True

    def create_claim(self, ident: dict, n: int, **extra) -> Path | None:
        p = self.claim_path(ident, n)
        t = self.now()
        doc = {"key": key_of(ident), "dataset": ident["dataset"], "attempt": int(n), "pid": os.getpid(),
               "host": socket.gethostname(), "seat": self.seat, "t_claim": t, "t_heartbeat": t, "ended": False,
               **extra}
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dumps(doc) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return p


class _ResetMeter:
    """交给 ``run_episode(ledger=...)`` 的 reset 计量：每次实际 build／reset 先记本席位账本，再记共享账本该 rid 名下；
    任一超额抛带 ``budget_exhausted`` 属性的异常（``EnvSession`` 据此标记额度耗尽）。"""

    def __init__(self, ledger: AttemptLedger, *, key: str, attempt_id: str, attempt_no: int, rid: str | None):
        self.ledger, self.key, self.attempt_id, self.attempt_no, self.rid = ledger, key, attempt_id, attempt_no, rid

    def claim(self, what: str) -> None:
        self.ledger.claim_reset(key=self.key, attempt_id=self.attempt_id, attempt_no=self.attempt_no, what=what)
        if self.ledger.shared is not None:
            self.ledger.shared.claim_reset(self.rid, what, route=self.ledger.route)


def _budget_exc(e: BaseException) -> bool:
    return bool(getattr(e, "budget_exhausted", False))


# ── 进程级信息 ──────────────────────────────────────────────────────────────


def gpu_info() -> dict:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip() if cvd else "0"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader", "-i", first],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        name, uid, drv = [x.strip() for x in out[0].split(",")]
        return {"gpu_name": name, "gpu_uuid": uid, "gpu_driver": drv, "cuda_visible_devices": cvd}
    except Exception as e:  # noqa: BLE001
        return {"gpu_name": None, "gpu_uuid": None, "gpu_driver": None, "cuda_visible_devices": cvd,
                "gpu_error": repr(e)}


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain", "--", "src", "dev-scripts", "scripts"))}


def package_info() -> dict:
    """评估包与子模块包的实际来源（editable 指向核对）。"""
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for name in ("robomme_ood_eval", "robomme_hard", "robomme"):
        spec = importlib.util.find_spec(name)
        out[f"{name}_file"] = None if spec is None else spec.origin
    return out


# ── 席位 ────────────────────────────────────────────────────────────────────


class SeatRunner:
    """一个席位上一个策略的常驻客户端。

    可注入（单测）：``policy_factory(model, seed, **cfg)``（缺省 ``load_policy``）、``episode_kwargs``（透传给
    ``run_episode`` 的关键字，如 ``recorder_factory``、``render``）、``hard_exit``（缺省 ``os._exit``）、``now``。
    """

    def __init__(self, args, *, policy_factory: Callable[..., Any] | None = None, episode_kwargs: dict | None = None,
                 hard_exit: Callable[[int], Any] | None = None, proc_info: dict | None = None,
                 now: Callable[[], float] = time.time, shared: Any = _UNSET):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.policy_seed = policy_seed_of(args)
        self.label = queue_label(args)
        self.seat = str(getattr(args, "seat", None) or default_seat())
        self.route = policy_route(args)
        self.seat_dir = self.out / "seats" / self.label / f"seed{self.policy_seed}" / self.seat
        self.seat_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.seat_dir / "progress.json"
        self.results_path = self.seat_dir / "seat-results.jsonl"
        self.policy_factory = policy_factory
        self.episode_kwargs = dict(episode_kwargs or {})
        self._exit = hard_exit or os._exit
        self.proc_info = proc_info or {}
        self.now = now
        retries = int(getattr(args, "infra_retries", DEFAULT_INFRA_RETRIES) or 0)
        from robomme_ood_eval import episode as E

        self.E = E
        wall = getattr(args, "wall_s", None)
        wall = float(wall) if wall else E.DEFAULT_WALL_S.get(args.policy, E.FALLBACK_WALL_S)
        self.wall_s = wall + E.FIRST_EPISODE_EXTRA_S
        qroot = Path(args.queue) if getattr(args, "queue", None) else self.out / "queue" / self.label / f"seed{self.policy_seed}"
        self.queue = DynamicQueue(qroot, seat=self.seat, wall_s=self.wall_s, max_attempts=1 + retries, now=now)
        if shared is _UNSET:
            shared = getattr(args, "budget_ledger", None) or _UNSET
        self.ledger = AttemptLedger(self.seat_dir / f"{self.label}.ledger.jsonl", seat=self.seat, policy=args.policy,
                                    route=self.route, caps=budget_caps(args), shared=shared)
        self.ledger.start(getattr(args, "reset_budget", None), retries)
        self.episodes_done = 0
        self._current: Claim | None = None
        self._current_attempt_id: str | None = None
        self._lock = threading.Lock()
        self._lease_cm = None
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self.heartbeat_s = float(getattr(args, "heartbeat_s", None) or DEFAULT_HEARTBEAT_S)
        self.policy = None

    # 进度：原子替换写本席位 progress.json（阶段、身份或局数变化时）
    def progress(self, phase: str, **extra) -> None:
        if phase not in PHASES:
            raise ValueError(f"phase={phase!r} 不是 {PHASES} 之一")
        cur = self._current
        doc = {"pid": os.getpid(), "host": socket.gethostname(), "seat": self.seat, "policy": self.args.policy,
               "label": self.label, "policy_seed": self.policy_seed, "phase": phase,
               "key": cur.key if cur else None, "dataset": cur.ident["dataset"] if cur else None,
               "attempt_no": cur.attempt if cur else None, "episodes_done": self.episodes_done, "t": time.time(),
               **extra}
        write_json_atomic(self.progress_path, doc)

    # ── 模型 ────────────────────────────────────────────────────────────
    def policy_cfg(self) -> dict:
        """交给 ``load_policy(**cfg)`` 的模型参数：GroundSG 变体与 adapter、GPU、显式模型参数与透传参数。"""
        a = self.args
        cfg: dict = {}
        if a.policy == "groundsg":
            cfg["groundsg_variant"] = a.groundsg_variant
            for dest, _flag in VARIANT_ADAPTERS.values():
                if getattr(a, dest, None):
                    cfg[dest] = getattr(a, dest)
        for k in ("ckpt", "port_base", "work_dir", "compile_cache"):
            v = getattr(a, k, None)
            if v is not None:
                cfg[k] = v
        if getattr(a, "gpus", None):
            cfg["gpus"] = [int(x) for x in str(a.gpus).split(",") if x.strip()]
        cfg.update(getattr(a, "extra_cfg", None) or {})
        return cfg

    def load(self):
        factory = self.policy_factory
        if factory is None:
            from robomme_ood_eval.policy import load_policy as factory  # noqa: N813
        return factory(self.args.policy, self.policy_seed, **self.policy_cfg())

    # ── 共享账本、lease、路线 ──────────────────────────────────────────
    def _open_shared_budget(self) -> None:
        shared = self.ledger.shared
        if shared is None:
            return
        bl = load_sibling("budget_ledger")
        try:
            if hasattr(shared, "check"):
                shared.check()
            if self._lease_cm is None and hasattr(shared, "lease"):
                sid = shard_id_of(self.route, self.seat)
                cm = shared.lease(sid)
                cm.__enter__()
                self._lease_cm = cm
                print(f"BUDGET_LEASE shard={sid} pid={os.getpid()}", flush=True)
        except Exception as e:  # noqa: BLE001 账本类可能来自另一份模块副本，按类名识别
            reason = {"LeaseHeld": "lease_held", "LedgerCorrupt": "ledger_corrupt",
                      "BudgetConfigMismatch": "budget_config"}.get(type(e).__name__)
            if reason is None and not isinstance(e, (bl.LeaseHeld, bl.LedgerCorrupt, bl.BudgetConfigMismatch)):
                raise
            print(f"RUN_BLOCKED reason={reason} policy={self.args.policy} seat={self.seat} detail={e}", flush=True)
            raise SeatStop(EXIT_BLOCKED, reason) from e

    def _release_lease(self) -> None:
        cm, self._lease_cm = self._lease_cm, None
        if cm is not None:
            cm.__exit__(None, None, None)

    def _check_route(self) -> None:
        """尝试账本里已有 ``attempt_start`` 行的 ``route`` 必须全部等于当前路线（含 ``seed<n>``）；不符 RUN_BLOCKED。"""
        led = self.ledger
        if led.shared is None:
            return
        found = []
        for starts in led.starts.values():
            for st in starts:
                tag = str(st.get("route") or "legacy")
                if tag != led.route and tag not in found:
                    found.append(tag)
        if found:
            print(f"RUN_BLOCKED reason=route_mismatch ledger={led.path} found={','.join(found)} want={led.route} "
                  f"policy={self.args.policy} seat={self.seat}", flush=True)
            raise SeatStop(EXIT_BLOCKED, "route_mismatch")

    # ── 恢复 ────────────────────────────────────────────────────────────
    def _ident_of_start(self, st: dict) -> dict | None:
        ident = st.get("identity")
        return dict(ident) if isinstance(ident, dict) else None

    def _settle_shared(self, rid: str | None, *, status: str | None, infra: bool, void: bool = False) -> None:
        shared = self.ledger.shared
        if rid is None or shared is None:
            return
        if void:
            shared.release(rid, status=status)
        else:
            shared.commit(rid, status=status, infra=bool(infra))

    def recover_dangling(self) -> int:
        """本席位账本里有 attempt_start 无 attempt_end 的尝试（进程被杀、看门狗退出前没写完）：本局 ``result.json`` 在且
        尝试号相同的按它补 attempt_end（终态且非基础设施 → accept 与队列 accepted 标记），否则记基础设施错误；共享账本
        rid 一律结算（commit 幂等），领取文件记收尾。"""
        n = 0
        for st in self.ledger.dangling():
            res = read_json(Path(st["result_path"])) if st.get("result_path") else None
            row = {"key": st["key"], "attempt_id": st["attempt_id"], "attempt_no": st["attempt_no"],
                   "status": "error", "infra": True, "exec_steps": None, "recovered": True}
            if res is not None and res.get("attempt") == st["attempt_no"] and res.get("key") == st["key"]:
                infra = bool(res.get("infra"))
                row.update(status="error" if infra else res.get("status"), infra=infra,
                           exec_steps=res.get("exec_steps"), cap_hit=res.get("cap_hit"),
                           budget_exhausted=bool(res.get("budget_exhausted")), infra_reason=res.get("infra_reason"))
            elif self.ledger.shared is not None:
                interrupt, evidence = self.ledger.classify_interrupt(st)
                row.update(interrupt=interrupt, interrupt_evidence=evidence)
            self.ledger.attempt_end(row)
            self._settle_shared(st.get("budget_rid"), status=row["status"], infra=row["infra"])
            ident = self._ident_of_start(st)
            if ident is not None:
                if AttemptLedger.is_final(row):
                    self.queue.accept(ident, attempt=st["attempt_no"], status=row["status"],
                                      result=st.get("result_path"), recovered=True)
                if st.get("claim"):
                    self.queue.end_claim(Path(st["claim"]), "recovered" if AttemptLedger.is_final(row) else "interrupted")
            print(f"LEDGER_RECOVER key={st['key']} attempt_id={st['attempt_id']} status={row.get('status')}", flush=True)
            n += 1
        return n

    def recover_crash_window(self) -> int:
        """「attempt_end 已写、accept 未写」时崩溃：补写本地 accept 与队列 accepted 标记。"""
        n = 0
        for end in self.ledger.final_without_accept():
            self.ledger.recover_accept(end)
            st = next((s for s in self.ledger.starts.get(end["key"], []) if s["attempt_id"] == end["attempt_id"]), {})
            ident = self._ident_of_start(st)
            if ident is not None:
                self.queue.accept(ident, attempt=end.get("attempt_no"), status=end.get("status"),
                                  result=st.get("result_path"), recovered=True)
            print(f"LEDGER_RECOVER_ACCEPT key={end['key']} attempt_id={end['attempt_id']} status={end.get('status')}",
                  flush=True)
            n += 1
        return n

    # ── 领取 ────────────────────────────────────────────────────────────
    def token_of(self, key: str, attempt_no: int) -> str:
        """幂等 token ``<路线>|<key>|a<attempt_no>``（与 budget_ledger.make_token 同式）。"""
        return f"{self.route}|{key}|a{int(attempt_no)}"

    def _reserve(self, ident: dict, n: int, token: str, retry: bool) -> str | None:
        shared = self.ledger.shared
        if shared is None:
            return None
        try:
            return shared.reserve(resets=NEW_SIDE_RESETS_PER_ATTEMPT, route=self.route, key=key_of(ident),
                                  attempt_no=n, seat=self.seat, policy=self.args.policy, astra=self.args.policy == "astra",
                                  token=token, kind_of_try="recovery" if retry else "first", dataset=ident["dataset"])
        except Exception as e:  # noqa: BLE001 BudgetExhausted（可能来自另一份模块副本，按属性识别）
            if not _budget_exc(e):
                raise
            print(f"RUN_BLOCKED reason=budget policy={self.args.policy} seat={self.seat} key={key_of(ident)} "
                  f"dataset={ident['dataset']} budget_reason={getattr(e, 'reason', None)} detail={e}", flush=True)
            raise SeatStop(EXIT_BUDGET, "budget") from e

    def _reclaim_locked(self, ident: dict, st: "QueueState") -> str:
        """（须在身份锁内）回收最后一次未收尾的领取（过期或本席位旧进程留下）：领取文件记收尾，其共享账本 rid 按基础设施
        中断结算（commit 幂等，已结算的不重复记）。返回中断分类。"""
        q = self.queue
        n_last, doc = st.last
        interrupt = "expired" if st.expired else "infra"
        q.end_claim(q.claim_path(ident, n_last), "reclaimed", reclaimed_by=self.seat, reclaim_reason=interrupt)
        with contextlib.suppress(Exception):
            self._settle_shared(doc.get("rid"), status="fail", infra=True)
        print(f"QUEUE_RECLAIM key={key_of(ident)} dataset={ident['dataset']} attempt={n_last} "
              f"reason={'expired' if st.expired else 'stale_same_seat'} seat={self.seat}", flush=True)
        return interrupt

    def reclaim_stale(self, rows: list[dict]) -> int:
        """开跑前扫一遍：过期或本席位旧进程留下的未收尾领取一律回收（即便该身份已无尝试名额，也要收尾并结算 rid）。"""
        n = 0
        for ident in rows:
            with self.queue.lock(ident):
                st = self.queue.state(ident)
                if st.accepted is None and st.last is not None and not st.last[1].get("ended") \
                        and (st.expired or st.stale_mine):
                    self._reclaim_locked(ident, st)
                    n += 1
        return n

    def claim(self, ident: dict) -> Claim | str:
        """在身份锁下：判可领 →（重试时领重试名额）→ 预约共享账本 → O_EXCL 建领取文件。返回 ``Claim`` 或跳过原因
        （``accepted``／``live``／``exhausted``／``retry_denied``／``lost``）。"""
        q = self.queue
        with q.lock(ident):
            st = q.state(ident)
            if st.accepted is not None:
                return "accepted"
            if st.live:
                return "live"
            last = st.last
            interrupt = None
            if last is not None:
                n_last, doc = last
                # 上一次已收尾而无 accepted：中断分类以本席位账本为准（恢复时按 Slurm 证据分了 expired／infra），
                # 别的席位留下的按领取文件记的回收原因，缺省 infra
                interrupt = (self.ledger.retry_interrupt(key_of(ident)) if self.ledger.starts.get(key_of(ident))
                             else doc.get("reclaim_reason") or "infra")
                if not doc.get("ended"):  # 过期或本席位旧进程留下：回收（计一次尝试）
                    interrupt = self._reclaim_locked(ident, st)
                if n_last >= q.max_attempts:
                    return "exhausted"
            n = 1 if last is None else last[0] + 1
            retry = n > 1
            token = self.token_of(key_of(ident), n)
            if retry and not self.ledger.allow_retry(key_of(ident), token=token, interrupt=interrupt or "infra"):
                print(f"INFRA_RETRY_DENIED policy={self.args.policy} key={key_of(ident)} attempt={n}", flush=True)
                return "retry_denied"
            rid = self._reserve(ident, n, token, retry)
            path = q.create_claim(ident, n, token=token, rid=rid, route=self.route)
            if path is None:  # 锁内不应发生；保守地退回预约
                self._settle_shared(rid, status=None, infra=False, void=True)
                return "lost"
            return Claim(ident=ident, attempt=n, path=path, token=token, rid=rid, retry=retry, interrupt=interrupt)

    # ── 心跳 ────────────────────────────────────────────────────────────
    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(self.heartbeat_s):
            self.heartbeat_once()

    def heartbeat_once(self) -> None:
        """在 ``self._lock`` 内重读当前领取再写心跳：与结算（``_end``／``_hard_exit`` 同样持锁写收尾或作废）互斥，
        不会把已收尾的领取文件写回在跑状态；``DynamicQueue.heartbeat`` 自身也对 ended／已删除不写。"""
        with self._lock:
            cur = self._current
            if cur is not None:
                with contextlib.suppress(Exception):
                    self.queue.heartbeat(cur.path)

    def _start_heartbeat(self) -> None:
        if self._hb_thread is None:
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="seat-heartbeat", daemon=True)
            self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        self._hb_stop.set()
        t, self._hb_thread = self._hb_thread, None
        if t is not None:
            t.join(timeout=5)

    # ── 看门狗退出前的结算（修「墙钟超时只 _finish 不 _settle」的缺口） ───────
    def _hard_exit(self, code: int) -> None:
        with self._lock:
            cur, aid = self._current, self._current_attempt_id
            if cur is not None and aid is not None:
                with contextlib.suppress(Exception):
                    self.ledger.attempt_end({"key": cur.key, "attempt_id": aid, "attempt_no": cur.attempt,
                                             "status": "error", "infra": True, "infra_reason": "watchdog_exit",
                                             "exec_steps": None})
                with contextlib.suppress(Exception):
                    self._settle_shared(cur.rid, status="fail", infra=True)
                with contextlib.suppress(Exception):
                    self.queue.end_claim(cur.path, "infra_timeout", exit_code=int(code))
                with contextlib.suppress(Exception):
                    append_result(self.results_path, {"key": cur.key, "dataset": cur.ident["dataset"],
                                                      "attempt": cur.attempt, "attempt_id": aid,
                                                      "status": "error", "infra": True,
                                                      "infra_reason": "watchdog_exit", "budget_rid": cur.rid,
                                                      "t": time.time()})
                print(f"SEAT_WATCHDOG_EXIT key={cur.key} dataset={cur.ident['dataset']} attempt={cur.attempt} "
                      f"code={code}（已写 attempt_end、结算共享账本、领取文件记收尾）", flush=True)
                self._current = self._current_attempt_id = None
        sys.stdout.flush()
        self._exit(code)

    # ── 一局 ────────────────────────────────────────────────────────────
    def run_claim(self, policy, c: Claim) -> dict:
        """跑一次领取；返回本席位结果行。运行阻塞／额度不足抛 ``SeatStop``（已记录）。"""
        E = self.E
        from robomme_ood_eval.policy import AstraStop, ServerDead, ServerMismatch

        ident = c.ident
        ds, task, ep = ident["dataset"], ident["task"], int(ident["builder_episode"])
        attempt_id = uuid.uuid4().hex
        result_path = str(E.result_path(policy, ds, task, ep, self.out))
        with self._lock:
            self._current, self._current_attempt_id = c, attempt_id
        self.ledger.attempt_start(key=c.key, attempt_id=attempt_id, attempt_no=c.attempt, retry=c.retry,
                                  task=task, tier=ident["tier"], dataset=ds, builder_episode=ep, identity=ident,
                                  budget_rid=c.rid, token=c.token, claim=str(c.path), result_path=result_path,
                                  **({"interrupt": c.interrupt} if c.retry and c.interrupt else {}))
        self.progress("episode")
        meter = _ResetMeter(self.ledger, key=c.key, attempt_id=attempt_id, attempt_no=c.attempt, rid=c.rid)
        base = {"key": c.key, "dataset": ds, "task": task, "tier": ident["tier"], "builder_episode": ep,
                "attempt": c.attempt, "attempt_id": attempt_id, "budget_rid": c.rid, "budget_token": c.token,
                "claim": str(c.path), "result": result_path, "seat": self.seat, "policy": self.args.policy,
                "policy_seed": self.policy_seed, "host": socket.gethostname()}
        stop: SeatStop | None = None
        try:
            res = E.run_episode(policy, ds, task, ep, self.out, expect=ident, attempt=c.attempt, ledger=meter,
                                **self._episode_kw())
        except E.IdentityMismatch as e:
            row = dict(base, status="error", run_blocked=True, infra=False, error=f"IDENTITY_MISMATCH {e}"[:800])
            self._end(c, attempt_id, row, void=True)
            raise SeatStop(EXIT_BLOCKED, "identity") from e
        except (ServerDead, ServerMismatch, AstraStop) as e:
            row = dict(base, status="error", run_blocked=True, infra=False, error=f"{type(e).__name__}: {e}"[:800])
            self._end(c, attempt_id, row, void=True)
            print(f"RUN_BLOCKED reason={type(e).__name__} policy={self.args.policy} key={c.key} detail={e}", flush=True)
            raise SeatStop(EXIT_BLOCKED, type(e).__name__) from e
        except BaseException as e:  # noqa: BLE001
            if not _budget_exc(e):
                with self._lock:
                    self._current = self._current_attempt_id = None
                raise
            row = dict(base, status="error", budget_exhausted=True, infra=False,
                       error=f"RESET_BUDGET_EXHAUSTED {type(e).__name__}: {e}"[:800])
            self._end(c, attempt_id, row, void=True)
            print(f"RESET_BUDGET_EXHAUSTED policy={self.args.policy} seat={self.seat} key={c.key} detail={e}", flush=True)
            raise SeatStop(EXIT_BUDGET, "reset_budget") from e
        d = res.to_dict()
        infra = bool(d.get("infra"))
        row = dict(base, status=d["status"], task_success=d.get("task_success"), infra=infra,
                   infra_reason=d.get("infra_reason"), error=d.get("error"), error_kind=d.get("error_kind"),
                   exec_steps=d.get("exec_steps"), cap_hit=d.get("cap_hit"), reset_calls=d.get("reset_calls"),
                   budget_exhausted=bool(d.get("budget_exhausted")), run_blocked=bool(d.get("run_blocked")),
                   video=d.get("video"), video_error=d.get("video_error"), raw_dir=d.get("raw_dir"))
        self._end(c, attempt_id, row, void=False)
        self.episodes_done += 1
        self.progress("done")
        if infra and d.get("infra_reason") == "env_build" and getattr(self.args, "stop_on_env_build_error", False):
            print(f"RUN_BLOCKED reason=env_build policy={self.args.policy} seat={self.seat} key={c.key} "
                  f"attempt={c.attempt} settled=1", flush=True)
            raise SeatStop(EXIT_BLOCKED, "env_build")
        if stop is not None:  # pragma: no cover - 预留
            raise stop
        return row

    def _episode_kw(self) -> dict:
        kw = dict(self.episode_kwargs)
        kw.setdefault("render", not getattr(self.args, "no_render", False))
        if getattr(self.args, "official_root", None):
            kw.setdefault("official_root", self.args.official_root)
        if getattr(self.args, "wall_s", None):
            kw.setdefault("wall_s", float(self.args.wall_s))
        return kw

    def _end(self, c: Claim, attempt_id: str, row: dict, *, void: bool) -> None:
        """结算一次尝试：本地 attempt_end（infra 错误按 status=error 记，不进 accept）、共享账本 commit／release、
        accepted 标记或领取文件收尾、本席位结果行。"""
        infra = bool(row.get("infra"))
        led_status = "error" if (infra or row.get("run_blocked") or row.get("budget_exhausted")) else row["status"]
        led_row = {"key": c.key, "attempt_id": attempt_id, "attempt_no": c.attempt, "status": led_status,
                   "infra": infra, "cap_hit": row.get("cap_hit"), "exec_steps": row.get("exec_steps"),
                   "budget_exhausted": bool(row.get("budget_exhausted")), "run_blocked": bool(row.get("run_blocked")),
                   "infra_reason": row.get("infra_reason")}
        final = AttemptLedger.is_final(led_row)
        with self._lock:
            late = self.ledger.attempt_end(led_row)
            self._settle_shared(c.rid, status=row["status"], infra=infra, void=void)
            accepted = False
            if final:
                accepted = self.queue.accept(c.ident, attempt=c.attempt, status=row["status"], result=row.get("result"),
                                             attempt_id=attempt_id)
            if void:
                self.queue.void_claim(c.path)
            else:
                self.queue.end_claim(c.path, row["status"] if final else "infra", final=final, accepted=accepted)
            row.update(late=late, final=final, accepted=accepted, void=void, t=time.time())
            append_result(self.results_path, row)
            self._current = self._current_attempt_id = None
        print(f"SEAT_EPISODE_DONE policy={self.args.policy} seat={self.seat} dataset={row['dataset']} key={c.key} "
              f"attempt={c.attempt} status={row['status']} infra={int(infra)} final={int(final)} "
              f"accepted={int(accepted)} void={int(void)}", flush=True)

    # ── 主循环 ──────────────────────────────────────────────────────────
    def summarize(self, rows: list[dict]) -> int:
        acc = live = missing = 0
        miss_keys = []
        for ident in rows:
            st = self.queue.state(ident)
            if st.accepted is not None:
                acc += 1
            elif st.live:
                live += 1
            else:  # 无 accepted、也无人在跑（尝试用满，或被额度／阻塞挡下未再领）
                missing += 1
                miss_keys.append(f"{ident['dataset']}:{key_of(ident)}")
        print(f"RUN_SUMMARY policy={self.args.policy} label={self.label} seat={self.seat} total={len(rows)} "
              f"accepted={acc} running_elsewhere={live} missing={missing} episodes_run={self.episodes_done}", flush=True)
        if missing:
            print(f"RUN_INCOMPLETE policy={self.args.policy} seat={self.seat} total={len(rows)} missing={missing} "
                  f"first={','.join(miss_keys[:5])}", flush=True)
            return EXIT_INCOMPLETE
        if live:
            print(f"SEAT_IDLE policy={self.args.policy} seat={self.seat} running_elsewhere={live}", flush=True)
        return 0

    def run(self, rows: list[dict]) -> int:
        """跑一份身份清单；返回退出码（见模块文档）。``SeatStop`` 转为其退出码。"""
        try:
            self._open_shared_budget()
            self._check_route()
            self.recover_dangling()
            self.recover_crash_window()
            self.reclaim_stale(rows)
            todo = [r for r in rows if self.queue.claimable(r)]
            print(f"RUN_PLAN policy={self.args.policy} label={self.label} seat={self.seat} total={len(rows)} "
                  f"claimable={len(todo)} max_attempts={self.queue.max_attempts} route={self.route} "
                  f"reset_left={self.ledger.reset_left()}", flush=True)
            if not todo:
                return self.summarize(rows)
            self.progress("context_load")
            self._start_heartbeat()
            prev_exit = self.E.HARD_EXIT
            self.E.HARD_EXIT = self._hard_exit
            try:
                policy = self.load()
                self.policy = policy
                try:
                    while True:
                        progressed = False
                        for ident in rows:
                            self.progress("claim")
                            c = self.claim(ident)
                            if isinstance(c, Claim):
                                self.run_claim(policy, c)
                                progressed = True
                                break  # 每跑完一局从清单头重新扫（保持清单顺序、及时处理重试）
                        if not progressed:
                            break
                finally:
                    policy.close()
            finally:
                self.E.HARD_EXIT = prev_exit
                self._stop_heartbeat()
            return self.summarize(rows)
        except SeatStop as e:
            return e.code
        finally:
            self._release_lease()

    def close(self) -> None:
        self._stop_heartbeat()
        self._release_lease()


# ── CLI ─────────────────────────────────────────────────────────────────────


def _coerce(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def parse_extra(tokens: list[str]) -> dict:
    """未声明的 ``--名 值``／``--开关`` 透传成模型参数（连字符换下划线），同 ``scripts/evaluate.py``。"""
    cfg: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--") or tok == "--":
            raise SystemExit(f"无法解析的参数：{tok!r}")
        name = tok[2:]
        if "=" in name:
            name, val = name.split("=", 1)
            cfg[name.replace("-", "_")] = _coerce(val)
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            cfg[name.replace("-", "_")] = _coerce(tokens[i + 1])
            i += 2
        else:
            cfg[name.replace("-", "_")] = True
            i += 1
    return cfg


def model_choices() -> tuple[str, ...]:
    try:
        from robomme_ood_eval.models import MODELS

        return tuple(MODELS)
    except Exception:  # noqa: BLE001  pragma: no cover
        return ("dummy", "perceptual-framesamp-modul", "groundsg", "smvla", "pp", "astra")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GL 席位客户端：常驻 Policy + 共享动态队列（ood／hard-verify）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="领局并逐局调 run_episode")
    p.add_argument("--policy", required=True, choices=list(model_choices()), help="模型注册名")
    p.add_argument("--groundsg-variant", default=None, choices=list(GROUNDSG_VARIANTS), help="--policy groundsg 必填")
    p.add_argument("--qwenvl-groundsg-adapter", default=None, help="--groundsg-variant ground-sg-qwenvl 必填")
    p.add_argument("--memer-adapter", default=None, help="--groundsg-variant ground-sg-memer 必填")
    p.add_argument("--policy-seed", type=int, default=None, help="模型种子（必填，非负整数）")
    p.add_argument("--identities", required=True, help="身份清单：export_eval_identities.py 的 JSONL 或分片 JSON 数组")
    p.add_argument("--dataset", default=None, choices=list(DATASETS),
                   help="可选：只跑该数据集的行；行内没有 dataset 字段时以它为准")
    p.add_argument("--out", required=True, help="产物根（其下 rollouts/、queue/、seats/）")
    p.add_argument("--queue", default=None, help="动态队列目录（缺省 <out>/queue/<标签>/seed<n>/）")
    p.add_argument("--seat", default=None, help="席位名（缺省 <主机>-<作业号>-gpu<CUDA_VISIBLE_DEVICES>，重起客户端不变）")
    p.add_argument("--infra-retries", type=int, default=DEFAULT_INFRA_RETRIES,
                   help="每身份基础设施重试次数（本轮 0：失败即停交用户）")
    p.add_argument("--stop-on-env-build-error", action="store_true",
                   help="环境构建基础设施失败时先完整结算本次尝试，再阻塞退出席位")
    p.add_argument("--reset-budget", type=int, default=None, help="可选：本席位账本的 reset 额度（不给只计量）")
    p.add_argument("--wall-s", type=float, default=None, help="单局墙钟（秒；缺省按模型取）")
    p.add_argument("--heartbeat-s", type=float, default=DEFAULT_HEARTBEAT_S, help="领取文件心跳间隔（秒）")
    p.add_argument("--no-render", action="store_true", help="不出官方版式网站视频")
    p.add_argument("--official-root", default=None, help="官方 RolloutRecorder 所在仓根（缺省评估仓根）")
    p.add_argument("--order", default="forward", choices=["forward", "reverse", "shuffle"])
    p.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    p.add_argument("--only", default=None, help="只跑这些 <task>_<tier>_<seed>（逗号分隔）")
    p.add_argument("--limit", type=int, default=0)
    g = p.add_argument_group("预算（六个参数必填，缺任一即 RUN_BLOCKED reason=budget_args）")
    g.add_argument("--budget-ledger", default=None, help="共享预算账本（budget_ledger.py 格式，NFS 上本轮一份）")
    g.add_argument("--trajectory-cap", type=int, default=None)
    g.add_argument("--reset-cap", type=int, default=None)
    g.add_argument("--shared-infra-cap", type=int, default=None)
    g.add_argument("--expired-cap", type=int, default=None)
    g.add_argument("--planned-first-tries", type=int, default=None)
    m = p.add_argument_group("模型参数（透传给 load_policy；另可给任意 --<名> <值> 或 --cfg 键=值）")
    m.add_argument("--gpus", default=None, help="逗号分隔 GPU 编号（占位 job 内只见本席一张卡：0）")
    m.add_argument("--ckpt", default=None)
    m.add_argument("--port-base", type=int, default=None)
    m.add_argument("--work-dir", default=None)
    m.add_argument("--compile-cache", default=None)
    m.add_argument("--cfg", action="append", default=[], metavar="键=值")
    p.set_defaults(func=cmd_run)
    s = sub.add_parser("status", help="只读：打印队列里各身份的状态")
    s.add_argument("--queue", required=True)
    s.add_argument("--identities", required=True)
    s.add_argument("--dataset", default=None, choices=list(DATASETS))
    s.set_defaults(func=cmd_status)
    return ap


def cmd_run(args) -> int:
    blk = entry_blockers(args)
    vp = variant_problems(args)
    if vp and (blk is None or blk[0] != "variant_pairing"):
        blk = ("variant_pairing", "; ".join(vp))
    if blk:
        print(f"RUN_BLOCKED reason={blk[0]} detail={blk[1]}", flush=True)
        return EXIT_BLOCKED
    for kv in args.cfg:
        if "=" not in kv:
            print(f"RUN_BLOCKED reason=args detail=--cfg 须为 键=值：{kv!r}", flush=True)
            return EXIT_BLOCKED
        k, v = kv.split("=", 1)
        args.extra_cfg[k.replace("-", "_")] = _coerce(v)
    try:
        rows = load_identities(args)
    except SeatStop as e:
        return e.code
    import signal

    def _on_term(signum, _frame):  # TERM／HUP 转成 SystemExit：with 退出时 policy.close() 停服务端
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_term)
    proc = {**gpu_info(), **git_info(), **package_info()}
    runner = SeatRunner(args, proc_info=proc)
    print(f"CLIENT_READY policy={args.policy} variant={policy_variant_of(args)} seat={runner.seat} "
          f"policy_seed={args.policy_seed} route={runner.route} identities={len(rows)} host={socket.gethostname()} "
          f"gpu={proc.get('gpu_name')} robomme_ood_eval={proc.get('robomme_ood_eval_file')} "
          f"robomme_hard={proc.get('robomme_hard_file')} git={str(proc.get('git_commit'))[:12]} "
          f"dirty={proc.get('git_dirty')}", flush=True)
    write_json_atomic(runner.seat_dir / f"process-{os.getpid()}.json", proc)
    try:
        rc = runner.run(rows)
    finally:
        runner.close()
    runner.progress("finished", rc=rc)
    print(f"全部完成 policy={args.policy} seat={runner.seat} episodes={runner.episodes_done} rc={rc}", flush=True)
    return rc


def cmd_status(args) -> int:
    rows = []
    for r in read_identity_file(Path(args.identities)):
        if args.dataset and r.get("dataset") not in (None, args.dataset):
            continue
        rows.append(identity_for_queue(r, args.dataset))
    q = DynamicQueue(args.queue, seat="status", wall_s=float("inf"))
    counts = {"accepted": 0, "live": 0, "open": 0}
    for ident in rows:
        st = q.state(ident)
        kind = "accepted" if st.accepted is not None else ("live" if st.live else "open")
        counts[kind] += 1
        print(f"QUEUE_ITEM dataset={ident['dataset']} key={key_of(ident)} state={kind} "
              f"attempts={len(st.attempts)}", flush=True)
    print(f"QUEUE_STATUS total={len(rows)} " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "run":
        args.extra_cfg = parse_extra(rest)
    elif rest:
        ap.error(f"未知参数：{rest}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
