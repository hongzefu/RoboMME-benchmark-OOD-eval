#!/usr/bin/env python3
"""第二阶段跨原侧／新侧共享的预算账本（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S8 预算与额度」、
六节预算、R6、R11）。

一个账本文件（JSONL，追加写 + fsync）由所有席位、两侧路线共同读写；每次领取都在 ``<账本>.lock`` 的独占
``fcntl.flock`` 下「读全量 → 判额度 → 追加一行」，所以并发争抢最后一个额度只有一方成功；全部计数从账本内容推出，
进程重启、换节点都不刷新额度。

行（``kind``）：

* ``reserve``：预约一局（一条轨迹）与 ``resets`` 次 reset 计量；字段 ``rid``／``route``／``key``／``astra``；
* ``commit``：该局收尾；可带实际 reset 次数 ``resets``（不带时按预约数与实领数取大）；
* ``release``：该局未进入执行（如开局前被拒），轨迹名额退回；已实领的 reset 仍计入；
* ``reset_claim``：新侧 ``EnvSession._claim`` 每次实际 build／reset 记一行（挂在某个 ``rid`` 下）；
* ``retry_claim``：基础设施重试名额；``interrupt`` ∈ {``infra``，``expired``}，分别计数——``infra`` 占共享 infra
  额度，``expired``（有 Slurm 到期证据）占到期续跑额度；同一 ``route``+``key`` 至多 ``V8_MAX_ATTEMPTS-1`` 次重试。

上限（六节）：轨迹 6366 硬上限；reset 计量 141430；Astra 局数 2 硬上限；共享 infra 重试 50、到期续跑 500（两者都在 6366
之内，按预约另计轨迹）。

拆仓验收（1008 拆分方案 R5）起 **reset 计量也是硬上限**：``reserve`` 预约后的计量总数、``claim_reset`` 实领后的计量总数
超过 ``reset_cap`` 一律抛 ``BudgetExhausted(reason="reset_cap")``、不写行（调用方在进入 attempt 前拒绝或按额度耗尽收尾），
不再只打印 ``BUDGET_WARN``。``reset_cap`` 进 config 行（schema ``sgeval-budget/3``），所有入口必须给同一组上限，否则
``BudgetConfigMismatch``。本轮拆仓验收的口径：轨迹 26、计量 60、共享 infra 0、到期续跑 0、计划首试 26（基础设施重试 0 次）。

第三阶段（1006 计划第二部分八.9、八.10 第 5 条、八.12 第 1 条；接口冻结说明第三节）在此之上：

* ``config`` 行：账本首次打开写一行 ``{"kind":"config","trajectory_cap","shared_infra_cap","expired_cap",
  "planned_first_tries","schema":"sgeval-budget/2"}``，之后每次打开比对，构造参数不一致抛 ``BudgetConfigMismatch``；
* 幂等 token（``<route>|<key>|a<attempt_no>``）：``reserve(..., token=)`` 同 token 已有未退回的预约即返回同一 rid、
  不写新行；``claim_retry(..., token=)`` 同 token 已领过即返回 True、不写新行；
* 首试保留额度：``reserve(..., kind_of_try="recovery")`` 只在「已用轨迹 + 1 + 未开始的计划首试数 <= 轨迹上限」且
  恢复合计不超过 ``trajectory_cap - planned_first_tries`` 时允许，否则 ``BudgetExhausted(reason=
  "reserved_for_first_tries"|"recovery_cap")``；``claim_retry`` 同样受这两条约束；
* 分片排他 ``lease(shard_id)``：``<账本>.lease.<shard_id>`` 上非阻塞独占 flock，拿不到抛 ``LeaseHeld``；
* 坏行即拒：每次读写前账本里有坏行（半行、未知 kind）即抛 ``LedgerCorrupt``；``_append`` 写前若文件末字节不是换行
  先补一个换行。``report`` 遇到坏行或配置不一致不抛，判 FAIL。

CLI::

    budget_ledger.py [--ledger PATH] [--trajectory-cap N --reset-cap N --shared-infra-cap N --expired-cap N --planned-first-tries N]
                     reserve --resets N [--route R] [--key K] [--astra] [--token T] [--kind-of-try first|recovery]
    budget_ledger.py [--ledger PATH] commit --id RID [--resets N]
    budget_ledger.py [--ledger PATH] release --id RID
    budget_ledger.py [--ledger PATH] claim-retry --route R --key K --interrupt infra|expired [--token T]
    budget_ledger.py [--ledger PATH] report

``--ledger`` 缺省取环境变量 ``SGEVAL_BUDGET_LEDGER``。``reserve`` 成功时末行打印 ``BUDGET_RESERVE rid=<rid> ...``
（标准输出最后一个字段即 rid，可被启动器取用）；额度不足打印 ``RUN_BLOCKED reason=budget ...`` 并以退出码 5 退出；
账本坏行或配置不一致打印 ``RUN_BLOCKED reason=ledger_corrupt|budget_config`` 并以退出码 3 退出。
``report`` 末行固定为 ``BUDGET_ENFORCEMENT=PASS|FAIL trajectories=<used>/<cap> resets=<used>/<cap> astra=<used>/2
shared_infra=<used>/<cap>``（格式不变；第三阶段的到期接续、恢复合计、首试计数记在 ``BUDGET_DETAIL`` 行）。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


#: 六节预算：轨迹硬上限、reset 计量软上限（十倍）、Astra 局数硬上限、共享 infra 重试、到期续跑
TRAJECTORY_CAP = 6366
RESET_CAP = 141430
#: 旧名（第二阶段曾是软上限）；现与 RESET_CAP 同值、同为硬上限
RESET_SOFT_CAP = RESET_CAP
ASTRA_CAP = 2
SHARED_INFRA_CAP = 50
EXPIRED_CAP = 500
#: 与 env_client.V8_MAX_ATTEMPTS 相同口径：每身份至多 2 次尝试（首试 + 1 次重试，infra 与 expired 合计）
V8_MAX_ATTEMPTS = 2
INTERRUPTS = ("infra", "expired")
ENV_LEDGER = "SGEVAL_BUDGET_LEDGER"
EXIT_BUDGET = 5
EXIT_BLOCKED = 3
#: 第三阶段账本模式（config 行的 schema）
SCHEMA = "sgeval-budget/3"
#: config 行里比对的构造参数（接口冻结说明第三节第 1 条；拆仓验收加 reset_cap）
CONFIG_KEYS = ("trajectory_cap", "reset_cap", "shared_infra_cap", "expired_cap", "planned_first_tries")
#: reserve 的尝试性质：首试或恢复（infra 重试／到期接续）
KINDS_OF_TRY = ("first", "recovery")
_SHARD_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class BudgetExhausted(RuntimeError):
    """硬上限（轨迹、reset 计量、Astra、共享 infra／到期续跑、每身份重试名额、首试保留额度）不足：调用方在进入
    attempt 前拒绝。``reason`` 为具名原因（``trajectory_cap``／``reset_cap``／``astra_cap``／``reserved_for_first_tries``／
    ``recovery_cap``）。"""

    budget_exhausted = True

    def __init__(self, msg: str = "", *, reason: str = "trajectory_cap"):
        super().__init__(msg)
        self.reason = reason


class BudgetConfigMismatch(RuntimeError):
    """构造参数与账本 config 行不一致（CLI 退出 3）。"""

    exit_code = EXIT_BLOCKED


class LedgerCorrupt(RuntimeError):
    """账本里有坏行（半行、未知 kind）：每次读写前即拒（CLI 退出 3）。"""

    exit_code = EXIT_BLOCKED


class LeaseHeld(RuntimeError):
    """分片排他 lease 已被另一进程持有（调用方打印 RUN_BLOCKED reason=lease_held、退出 3）。"""

    exit_code = EXIT_BLOCKED


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def make_token(route: str, key: str, attempt_no: int) -> str:
    """幂等 token：``<route>|<key>|a<attempt_no>``（接口冻结说明第三节第 3 条）。"""
    return f"{route}|{key}|a{int(attempt_no)}"


def _canonical_row(row: Any) -> Any:
    """历史账本行的旧路线名映射成官方名（别名表在评估包 ``robomme_ood_eval/models/_official_defs.py``，模块名
    ``official_defs``，已加载则复用同一模块）。"""
    mod = sys.modules.get("official_defs")
    if mod is None or not hasattr(mod, "canonical_row"):
        import importlib.util

        path = Path(__file__).resolve().parents[2] / "src" / "robomme_ood_eval" / "models" / "_official_defs.py"
        spec = importlib.util.spec_from_file_location("official_defs", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["official_defs"] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop("official_defs", None)
            raise
    return mod.canonical_row(row)


class BudgetState:
    """从账本行推出的计数（只读快照）。"""

    def __init__(self, rows: list[dict]):
        self.reserves: dict[str, dict] = {}
        self.commits: dict[str, dict] = {}
        self.releases: set[str] = set()
        self.claimed: dict[str, int] = {}  # rid -> reset_claim 行数
        self.orphan_claims = 0  # 无 rid 的 reset_claim
        self.retries: list[dict] = []
        self.retry_tokens: set[str] = set()
        self.config: dict | None = None
        self.bad_rows = 0
        for row in rows:
            kind = row.get("kind")
            rid = row.get("rid")
            if kind == "config":
                if self.config is None:
                    self.config = row
            elif kind == "reserve" and rid:
                self.reserves.setdefault(rid, row)
            elif kind == "commit" and rid:
                self.commits.setdefault(rid, row)
            elif kind == "release" and rid:
                self.releases.add(rid)
            elif kind == "reset_claim":
                if rid:
                    self.claimed[rid] = self.claimed.get(rid, 0) + 1
                else:
                    self.orphan_claims += 1
            elif kind == "retry_claim":
                self.retries.append(row)
                if row.get("token"):
                    self.retry_tokens.add(str(row["token"]))
            else:
                self.bad_rows += 1

    def live(self, rid: str) -> bool:
        return rid in self.reserves and rid not in self.releases

    def live_rid_of(self, token: str) -> str | None:
        """同 token 的未退回预约（最早一条）；没有返回 None。"""
        for rid, row in self.reserves.items():
            if row.get("token") == token and self.live(rid):
                return rid
        return None

    @property
    def trajectories(self) -> int:
        """未退回的预约数（进行中 + 已收尾）。"""
        return sum(self.live(r) for r in self.reserves)

    def tries_of(self, kind_of_try: str) -> int:
        """未退回、带该 ``kind_of_try`` 的预约数。"""
        return sum(self.live(r) and row.get("kind_of_try") == kind_of_try for r, row in self.reserves.items())

    @property
    def first_started(self) -> int:
        """已开始的首试数（未退回的 kind_of_try=first 预约）。"""
        return self.tries_of("first")

    @property
    def recovery_used(self) -> int:
        """已用的恢复轨迹数（未退回的 kind_of_try=recovery 预约）。"""
        return self.tries_of("recovery")

    @property
    def astra(self) -> int:
        return sum(self.live(r) and bool(row.get("astra")) for r, row in self.reserves.items())

    def resets_if_claimed(self, rid: str | None) -> int:
        """假如对 ``rid`` 再记一次实领，计量总数会是多少（预约数之内的实领不增加总数）。"""
        if rid is None or rid not in self.reserves:
            return self.resets + 1
        before = self.resets_of(rid)
        self.claimed[rid] = self.claimed.get(rid, 0) + 1
        try:
            after = self.resets_of(rid)
        finally:
            self.claimed[rid] -= 1
        return self.resets + (after - before)

    def resets_of(self, rid: str) -> int:
        """单局 reset 计量：退回的只计实领；收尾带实际数的取实际数与实领数之大；其余取预约数与实领数之大。"""
        claimed = self.claimed.get(rid, 0)
        if rid in self.releases:
            return claimed
        com = self.commits.get(rid)
        if com is not None and com.get("resets") is not None:
            return max(int(com["resets"]), claimed)
        return max(int(self.reserves[rid].get("resets") or 0), claimed)

    @property
    def resets(self) -> int:
        return sum(self.resets_of(r) for r in self.reserves) + self.orphan_claims

    def retries_of(self, interrupt: str) -> int:
        return sum(r.get("interrupt") == interrupt for r in self.retries)

    def retries_for(self, route: str, key: str) -> int:
        return sum(r.get("route") == route and r.get("key") == key for r in self.retries)


class BudgetLedger:
    """共享预算账本。``caps`` 可覆盖（单测用），缺省取六节常量；``planned_first_tries`` 缺省 None（不做首试保留）。
    第三阶段新侧与原侧由启动脚本显式给出 config 的参数，不回落常量默认值。``reset_soft_cap`` 是 ``reset_cap`` 的旧名
    （两者都给时以 ``reset_cap`` 为准），现为硬上限。"""

    def __init__(self, path: Path | str, *, trajectory_cap: int = TRAJECTORY_CAP, reset_cap: int | None = None,
                 reset_soft_cap: int | None = None, astra_cap: int = ASTRA_CAP, shared_infra_cap: int = SHARED_INFRA_CAP,
                 expired_cap: int = EXPIRED_CAP, planned_first_tries: int | None = None, warn_stream=None):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.trajectory_cap = int(trajectory_cap)
        self.reset_cap = int(reset_cap if reset_cap is not None else
                             (reset_soft_cap if reset_soft_cap is not None else RESET_CAP))
        self.astra_cap = int(astra_cap)
        self.shared_infra_cap = int(shared_infra_cap)
        self.expired_cap = int(expired_cap)
        self.planned_first_tries = None if planned_first_tries is None else int(planned_first_tries)
        if self.planned_first_tries is not None and not 0 <= self.planned_first_tries <= self.trajectory_cap:
            raise ValueError(f"planned_first_tries={self.planned_first_tries} 须在 0..{self.trajectory_cap}")
        self.warn_stream = warn_stream

    def config(self) -> dict:
        """本对象应有的 config 行内容（不含 t／host 等元字段）。"""
        return {"kind": "config", "trajectory_cap": self.trajectory_cap, "reset_cap": self.reset_cap,
                "shared_infra_cap": self.shared_infra_cap,
                "expired_cap": self.expired_cap, "planned_first_tries": self.planned_first_tries, "schema": SCHEMA}

    @property
    def recovery_cap(self) -> int | None:
        """恢复（infra 重试与到期接续）合计上限 = 轨迹上限 − 计划首试数；未给计划首试数时为 None。"""
        return None if self.planned_first_tries is None else self.trajectory_cap - self.planned_first_tries

    # ── 底层：文件锁 + 读全量 + 追加 ─────────────────────────────────────
    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"kind": "__bad__"})  # 半行（写入中被杀）：计 bad_rows，读写前即拒
                continue
            rows.append(_canonical_row(row) if isinstance(row, dict) else {"kind": "__bad__"})
        return rows

    def _append(self, row: dict) -> dict:
        row = {"t": time.time(), "host": socket.gethostname(), "pid": os.getpid(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **row}
        with open(self.path, "ab+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:  # 末字节不是换行（上次写入中被杀）：先补一个换行，新行不与半行粘连
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.seek(0, os.SEEK_END)
                    f.write(b"\n")
            f.seek(0, os.SEEK_END)
            f.write((_dumps(row) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        return row

    def _config_diff(self, st: BudgetState) -> dict:
        want, got = self.config(), st.config or {}
        return {k: (got.get(k), want[k]) for k in (*CONFIG_KEYS, "schema") if got.get(k) != want[k]}

    def _load(self) -> BudgetState:
        """读写前统一入口（须在锁内）：坏行即拒；无 config 行则写一行，有则比对，不一致即拒。"""
        st = BudgetState(self._read())
        if st.bad_rows:
            raise LedgerCorrupt(f"ledger={self.path} bad_rows={st.bad_rows}（坏行即拒，须人工核对后修复）")
        if st.config is None:
            st.config = self._append(self.config())
        else:
            diff = self._config_diff(st)
            if diff:
                raise BudgetConfigMismatch(f"ledger={self.path} config 行与构造参数不一致（账本值, 构造值）={diff}")
        return st

    def state(self) -> BudgetState:
        with self._locked():
            return self._load()

    def check(self) -> None:
        """打开时的一致性核对（坏行、config）；不通过抛 LedgerCorrupt／BudgetConfigMismatch。"""
        self.state()

    @property
    def reset_soft_cap(self) -> int:
        """旧名（只读）：等于 ``reset_cap``。"""
        return self.reset_cap

    def _check_resets(self, total: int) -> None:
        """计量总数超过 ``reset_cap`` 即硬拒（不写行）：打印 ``BUDGET_REJECT`` 并抛 ``BudgetExhausted(reason="reset_cap")``。"""
        if total > self.reset_cap:
            print(f"BUDGET_REJECT resets={total} cap={self.reset_cap}（硬上限，拒绝）",
                  file=self.warn_stream or sys.stderr, flush=True)
            raise BudgetExhausted(f"resets={total}/{self.reset_cap}", reason="reset_cap")

    def _recovery_block(self, st: BudgetState, *, recovery_in_use: int) -> str | None:
        """一次新的恢复会不会挤占未开始的计划首试额度、或超过恢复合计；返回具名原因，可以则 None。"""
        if self.planned_first_tries is None:
            return None
        pending_first = max(0, self.planned_first_tries - st.first_started)
        if st.trajectories + 1 + pending_first > self.trajectory_cap:
            return "reserved_for_first_tries"
        if recovery_in_use + 1 > self.recovery_cap:
            return "recovery_cap"
        return None

    # ── 分片排他 lease ─────────────────────────────────────────────────
    @contextmanager
    def lease(self, shard_id: str):
        """``<账本>.lease.<shard_id>`` 上 ``LOCK_EX|LOCK_NB``：进程存活期间持有，拿不到抛 LeaseHeld。"""
        shard_id = str(shard_id)
        if not _SHARD_RE.match(shard_id):
            raise ValueError(f"shard_id={shard_id!r} 只许字母数字 . _ -")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lp = self.path.with_name(f"{self.path.name}.lease.{shard_id}")
        fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            raise LeaseHeld(f"shard={shard_id} path={lp}") from e
        try:
            os.ftruncate(fd, 0)
            os.write(fd, (_dumps({"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}) + "\n").encode())
            yield lp
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # ── 领取 ────────────────────────────────────────────────────────────
    def reserve(self, *, resets: int, route: str | None = None, key: str | None = None, astra: bool = False,
                token: str | None = None, kind_of_try: str | None = None, **extra) -> str:
        """预约一局：轨迹、reset 计量与 Astra 都是硬上限（不足抛 BudgetExhausted，不写行）。返回 rid。

        ``token``：同 token 已有未退回的预约时返回同一 rid、不写新行（崩溃后续跑不重复扣额）。
        ``kind_of_try="recovery"``：另受首试保留额度与恢复合计约束（见模块文档）。"""
        resets = int(resets)
        if resets < 0:
            raise ValueError("resets 不能为负")
        if kind_of_try is not None and kind_of_try not in KINDS_OF_TRY:
            raise ValueError(f"kind_of_try={kind_of_try!r} 不是 {KINDS_OF_TRY} 之一")
        with self._locked():
            st = self._load()
            if token:
                rid = st.live_rid_of(token)
                if rid is not None:
                    return rid
            if st.trajectories >= self.trajectory_cap:
                raise BudgetExhausted(f"trajectories={st.trajectories}/{self.trajectory_cap}", reason="trajectory_cap")
            if astra and st.astra >= self.astra_cap:
                raise BudgetExhausted(f"astra={st.astra}/{self.astra_cap}", reason="astra_cap")
            if kind_of_try == "recovery":
                why = self._recovery_block(st, recovery_in_use=st.recovery_used)
                if why:
                    raise BudgetExhausted(
                        f"reason={why} trajectories={st.trajectories}/{self.trajectory_cap} "
                        f"first_started={st.first_started}/{self.planned_first_tries} "
                        f"recovery={st.recovery_used}/{self.recovery_cap}", reason=why)
            self._check_resets(st.resets + resets)
            rid = uuid.uuid4().hex
            row = {"kind": "reserve", "rid": rid, "resets": resets, "route": route, "key": key, "astra": bool(astra)}
            if token:
                row["token"] = token
            if kind_of_try:
                row["kind_of_try"] = kind_of_try
            self._append({**row, **extra})
            return rid

    def _need_rid(self, st: BudgetState, rid: str) -> None:
        if rid not in st.reserves:
            raise KeyError(f"未知 rid={rid}")

    def commit(self, rid: str, *, resets: int | None = None, **extra) -> None:
        with self._locked():
            st = self._load()
            self._need_rid(st, rid)
            if rid in st.commits or rid in st.releases:
                return  # 重复收尾：幂等
            self._append({"kind": "commit", "rid": rid, "resets": None if resets is None else int(resets), **extra})

    def release(self, rid: str, **extra) -> None:
        with self._locked():
            st = self._load()
            self._need_rid(st, rid)
            if rid in st.commits or rid in st.releases:
                return
            self._append({"kind": "release", "rid": rid, **extra})

    def claim_reset(self, rid: str | None, what: str, **extra) -> None:
        """记一次实际 build／reset；记完后计量总数会超过 ``reset_cap`` 即抛 ``BudgetExhausted(reason="reset_cap")``、
        不写行（预约数之内的实领不增加总数，不会被拒）。"""
        with self._locked():
            st = self._load()
            self._check_resets(st.resets_if_claimed(rid))
            self._append({"kind": "reset_claim", "rid": rid, "what": what, **extra})

    def claim_retry(self, *, route: str, key: str, interrupt: str, token: str | None = None, **extra) -> bool:
        """原子领一次基础设施重试名额；``infra`` 占共享 infra 额度、``expired`` 占到期续跑额度，同一 route+key 至多
        ``V8_MAX_ATTEMPTS-1`` 次；给了 planned_first_tries 时另受首试保留额度与恢复合计约束。成功返回 True，任一不足
        返回 False（不写行）。``token`` 已领过时直接返回 True、不写新行。"""
        if interrupt not in INTERRUPTS:
            raise ValueError(f"interrupt={interrupt!r} 不是 {INTERRUPTS} 之一")
        cap = self.shared_infra_cap if interrupt == "infra" else self.expired_cap
        with self._locked():
            st = self._load()
            if token and token in st.retry_tokens:
                return True
            if st.retries_of(interrupt) >= cap:
                return False
            if st.retries_for(route, key) >= V8_MAX_ATTEMPTS - 1:
                return False
            if self._recovery_block(st, recovery_in_use=len(st.retries)):
                return False
            row = {"kind": "retry_claim", "route": route, "key": key, "interrupt": interrupt}
            if token:
                row["token"] = token
            self._append({**row, **extra})
            return True

    # ── 汇总 ────────────────────────────────────────────────────────────
    def report_lines(self) -> tuple[bool, list[str]]:
        """汇总判定；坏行与 config 不一致不抛、记进 BUDGET_PROBLEMS 判 FAIL（也不补写 config 行）。"""
        with self._locked():
            st = BudgetState(self._read())
        problems = []
        if st.config is not None and self._config_diff(st):
            problems.append("config_mismatch")
        if st.trajectories > self.trajectory_cap:
            problems.append("trajectories_over_cap")
        if st.astra > self.astra_cap:
            problems.append("astra_over_cap")
        if st.resets > self.reset_cap:
            problems.append("resets_over_cap")
        if st.retries_of("infra") > self.shared_infra_cap:
            problems.append("shared_infra_over_cap")
        if st.retries_of("expired") > self.expired_cap:
            problems.append("expired_over_cap")
        if self.recovery_cap is not None and st.recovery_used > self.recovery_cap:
            problems.append("recovery_over_cap")
        if st.bad_rows:
            problems.append(f"bad_rows={st.bad_rows}")
        open_n = sum(st.live(r) and r not in st.commits for r in st.reserves)
        rec_cap = "na" if self.recovery_cap is None else self.recovery_cap
        lines = [f"BUDGET_DETAIL ledger={self.path} reserves={len(st.reserves)} committed={len(st.commits)} "
                 f"released={len(st.releases)} open={open_n} expired={st.retries_of('expired')}/{self.expired_cap} "
                 f"reset_cap_exceeded={int(st.resets > self.reset_cap)} "
                 f"first_started={st.first_started}/{self.planned_first_tries} recovery={st.recovery_used}/{rec_cap} "
                 f"config={'present' if st.config is not None else 'absent'}"]
        if problems:
            lines.append(f"BUDGET_PROBLEMS {' '.join(problems)}")
        ok = not problems
        lines.append(f"BUDGET_ENFORCEMENT={'PASS' if ok else 'FAIL'} trajectories={st.trajectories}/{self.trajectory_cap} "
                     f"resets={st.resets}/{self.reset_cap} astra={st.astra}/{self.astra_cap} "
                     f"shared_infra={st.retries_of('infra')}/{self.shared_infra_cap}")
        return ok, lines


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="共享预算账本（轨迹／reset／Astra／基础设施重试／首试保留）")
    ap.add_argument("--ledger", default=None, help=f"账本 JSONL；缺省取环境变量 {ENV_LEDGER}")
    ap.add_argument("--trajectory-cap", type=int, default=TRAJECTORY_CAP)
    ap.add_argument("--reset-cap", "--reset-soft-cap", dest="reset_cap", type=int, default=RESET_CAP,
                    help="reset 计量硬上限（--reset-soft-cap 为旧名）；进 config 行比对")
    ap.add_argument("--astra-cap", type=int, default=ASTRA_CAP)
    ap.add_argument("--shared-infra-cap", type=int, default=SHARED_INFRA_CAP)
    ap.add_argument("--expired-cap", type=int, default=EXPIRED_CAP)
    ap.add_argument("--planned-first-tries", type=int, default=None, help="计划首试数（首试保留额度；与 config 行比对）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reserve")
    p.add_argument("--resets", type=int, required=True)
    p.add_argument("--route", default=None)
    p.add_argument("--key", default=None)
    p.add_argument("--astra", action="store_true")
    p.add_argument("--token", default=None, help="幂等 token（<route>|<key>|a<attempt_no>）")
    p.add_argument("--kind-of-try", default=None, choices=list(KINDS_OF_TRY))
    p = sub.add_parser("commit")
    p.add_argument("--id", required=True)
    p.add_argument("--resets", type=int, default=None)
    p = sub.add_parser("release")
    p.add_argument("--id", required=True)
    p = sub.add_parser("claim-retry")
    p.add_argument("--route", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--interrupt", required=True, choices=list(INTERRUPTS))
    p.add_argument("--token", default=None, help="幂等 token（与随后的 reserve／attempt_start 同一个）")
    sub.add_parser("report")
    return ap


def _dispatch(led: BudgetLedger, args) -> int:
    if args.cmd == "reserve":
        try:
            rid = led.reserve(resets=args.resets, route=args.route, key=args.key, astra=args.astra, token=args.token,
                              kind_of_try=args.kind_of_try)
        except BudgetExhausted as e:
            print(f"RUN_BLOCKED reason=budget detail={e} budget_reason={e.reason} route={args.route} key={args.key}",
                  flush=True)
            return EXIT_BUDGET
        print(f"BUDGET_RESERVE resets={args.resets} route={args.route} key={args.key} astra={int(args.astra)} rid={rid}",
              flush=True)
        return 0
    if args.cmd == "commit":
        led.commit(args.id, resets=args.resets)
        print(f"BUDGET_COMMIT rid={args.id} resets={args.resets}", flush=True)
        return 0
    if args.cmd == "release":
        led.release(args.id)
        print(f"BUDGET_RELEASE rid={args.id}", flush=True)
        return 0
    if args.cmd == "claim-retry":
        ok = led.claim_retry(route=args.route, key=args.key, interrupt=args.interrupt, token=args.token)
        print(f"BUDGET_RETRY ok={int(ok)} route={args.route} key={args.key} interrupt={args.interrupt}", flush=True)
        return 0 if ok else EXIT_BUDGET
    ok, lines = led.report_lines()
    for line in lines:
        print(line, flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.ledger or os.environ.get(ENV_LEDGER)
    if not path:
        print(f"RUN_BLOCKED reason=budget detail=未给 --ledger 且 {ENV_LEDGER} 为空", flush=True)
        return EXIT_BLOCKED
    led = BudgetLedger(path, trajectory_cap=args.trajectory_cap, reset_cap=args.reset_cap,
                       astra_cap=args.astra_cap, shared_infra_cap=args.shared_infra_cap, expired_cap=args.expired_cap,
                       planned_first_tries=args.planned_first_tries)
    try:
        return _dispatch(led, args)
    except LedgerCorrupt as e:
        print(f"RUN_BLOCKED reason=ledger_corrupt detail={e}", flush=True)
        return EXIT_BLOCKED
    except BudgetConfigMismatch as e:
        print(f"RUN_BLOCKED reason=budget_config detail={e}", flush=True)
        return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
