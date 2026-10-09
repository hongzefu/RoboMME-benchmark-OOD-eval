"""外层每局一次的 ``run_episode`` 与它的两个数据结构 ``EpisodeSpec``、``EpisodeResult``（拆分方案 §三、§四）。

``run_episode(policy, dataset, task, episode, out_dir, expect=None) -> EpisodeResult`` 的顺序（单测逐步钉死）：

1. 造 ``EpisodeSpec`` 并核身份：``BenchmarkEnvBuilder.resolve_identity(episode)``（builder 按 ``(task, dataset)``
   进程内缓存）；给了 ``expect``（队列里的身份行）就按 ``check_identity`` 逐键比 tier／seed／candidate／
   spec_sha256／source_episode，不符写 ``error=IDENTITY_MISMATCH`` 的结果并抛 ``IdentityMismatch``（M1）；
2. ``policy.reset(spec)``：抛 ``ServerDead``／``AstraStop`` 时不写结果、原样上抛（局还没开始，续跑会重做这一局）；
3. 建录制器与 ``EnvSession``，先 ``session.build()``，失败记 ``infra=True infra_reason=env_build``、不进 play（M2）；
   挂单局看门狗（墙钟、首推理期限、媒体收尾期限）；
4. ``policy.play(session, spec, recorder)``；
5. ``finally``：``session.close()`` → ``recorder.close()`` →（有 trace 时）出官方版式网站视频 → 写 ``result.json``
   并追加 ``results.jsonl``。**绝不调 ``policy.load()``。**

终态（已定口径第 5 条）：``error`` 并入 ``fail``、原值进 ``error_kind``；``StepCapReached`` 与 ``session.cap_hit``
改记 ``timeout``；计数字段一律显式写 0（P4）。看门狗到点时先写 ``INFRA_TIMEOUT`` 结果、打印
``SERVER_LEFT pid=… port=… metadata=… stop="scripts/evaluate.py --stop-server …"``，再 ``os._exit(75)``——只杀客户端，
服务端子进程不动（S2）。

产物树（§四）：``<out>/rollouts/<模型>/<数据集>/seed<policy_seed>/`` 下 ``results.jsonl``、``log.json``（``report``
写）、``progress.json``、``videos/<Task>_ep<N>_<success|fail|timeout>_<task_goal>_<tier>.mp4``、
``raw/<Task>_ep<N>_<tier>/``（录制器的 ``front.mkv``／``wrist.mkv``／``arrays.npz``／``meta.json``／``events.jsonl``／
``frames-*.jsonl``，模型写的 ``trace.jsonl``，以及 ``result.json``）。``ep<N>``：ood 用 builder 局号 0～49，
hard-verify 用官方原 episode 号（``source_episode``）。
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from robomme_ood_eval.session import (DATASETS, HARD_VERIFY, OOD, EnvSession, RecorderError, ResetBudgetExhausted,
                                       StepCapReached)

#: 每个数据集的步数上限（已定口径第 3 条）；ood 严格截断（第 1801 步不进环境）
DATASET_MAX_STEPS = {HARD_VERIFY: 1300, OOD: 1800}
XHARD0 = "xhard0"
#: 官方 16 任务（顺序同官方 ``TASK_NAME_LIST``）
TASKS = ("BinFill", "StopCube", "PickXtimes", "SwingXtimes", "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap",
         "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube",
         "InsertPeg", "PatternLock", "RouteStick")
#: 单局墙钟缺省（秒）：按模型取，其余 1800；进程内第一局另加 FIRST_EPISODE_EXTRA_S
DEFAULT_WALL_S = {"smvla": 900.0, "perceptual-framesamp-modul": 1200.0}
FALLBACK_WALL_S = 1800.0
FIRST_EPISODE_EXTRA_S = 600.0
FIRST_INFER_DEADLINE_S = 1800.0
MEDIA_DEADLINE_S = 1200.0
EXIT_WALL = 75
#: play 可返回的终态；落盘终态只有前三个
PLAY_STATUSES = ("success", "fail", "timeout", "error")
TERMINAL_STATUSES = ("success", "fail", "timeout")
#: PlayOutput 必有字段
PLAY_REQUIRED = ("status", "task_success", "steps", "error", "infra", "infra_reason")
#: 看门狗阶段 → infra_reason
WATCH_REASONS = {"episode_wall": "episode_wall", "first_infer": "deadline_first_infer",
                 "media_finalize": "deadline_media_finalize"}
#: 硬退出函数（生产为 os._exit；单测替换成不退出的记录函数）
HARD_EXIT: Callable[[int], Any] = os._exit


class IdentityMismatch(RuntimeError):
    """builder 解析出的身份与队列身份行不符（M1）：结果已写，调用方整批停。"""


class PlayContractError(ValueError):
    """``play`` 返回值不满足 ``PlayOutput`` 契约。"""


# ── builder 缓存 ─────────────────────────────────────────────────────────────

#: 单测可替换：``(task, dataset, max_steps) -> builder``；None 时建真实 ``BenchmarkEnvBuilder``
BUILDER_FACTORY: Callable[[str, str, int], Any] | None = None
_BUILDERS: dict[tuple[str, str], Any] = {}


def builder_for(task: str, dataset: str):
    """按 ``(task, dataset)`` 进程内缓存 builder（步数上限取 ``DATASET_MAX_STEPS[dataset]``）。"""
    if dataset not in DATASET_MAX_STEPS:
        raise ValueError(f"dataset={dataset!r} 不是 {tuple(DATASET_MAX_STEPS)} 之一")
    ck = (task, dataset)
    if ck not in _BUILDERS:
        ms = DATASET_MAX_STEPS[dataset]
        if BUILDER_FACTORY is not None:
            _BUILDERS[ck] = BUILDER_FACTORY(task, dataset, ms)
        else:
            from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

            _BUILDERS[ck] = BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="joint_angle",
                                                max_steps=ms)
    return _BUILDERS[ck]


def clear_builders() -> None:
    _BUILDERS.clear()


# ── 身份核对（抽自 seat.py::check_identity） ───────────────────────────────


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _same_int_or_null(a: Any, b: Any) -> bool:
    return (a is None and b is None) or (_is_int(a) and _is_int(b) and a == b)


def check_identity(resolved: dict, want: dict, *, dataset: str = OOD) -> str | None:
    """builder 解析出的身份必须与队列身份行一致；不一致返回说明。

    ood：tier／seed／candidate／spec_sha256 逐键严格相等（candidate 可空）；若身份行给了 source_episode 也须相等。
    hard-verify：两边 tier 都是 xhard0，seed 与 source_episode 同为整数且相等，candidate 与 spec_sha256 两边都为 null。"""
    bad = []
    if resolved.get("tier") != want.get("tier"):
        bad.append(f"tier builder={resolved.get('tier')} want={want.get('tier')}")
    if dataset == HARD_VERIFY:
        if want.get("tier") != XHARD0:
            bad.append(f"tier want={want.get('tier')} 不是 {XHARD0}")
        for k in ("seed", "source_episode"):
            rv, wv = resolved.get(k), want.get(k)
            if not (_is_int(rv) and _is_int(wv) and rv == wv):
                bad.append(f"{k} builder={rv!r} want={wv!r}")
        for k in ("candidate", "spec_sha256"):
            if resolved.get(k) is not None or want.get(k) is not None:
                bad.append(f"{k} builder={resolved.get(k)!r} want={want.get(k)!r} 须都为 null")
        return "; ".join(bad) or None
    for k in ("seed", "candidate"):
        rv, wv = resolved.get(k), want.get(k)
        if not _same_int_or_null(rv, wv):
            bad.append(f"{k} builder={rv!r} want={wv!r}")
    if resolved.get("spec_sha256") != want.get("spec_sha256"):
        bad.append(f"spec_sha256 builder={resolved.get('spec_sha256')} want={want.get('spec_sha256')}")
    if "source_episode" in want and not _same_int_or_null(resolved.get("source_episode"), want.get("source_episode")):
        bad.append(f"source_episode builder={resolved.get('source_episode')!r} want={want.get('source_episode')!r}")
    return "; ".join(bad) or None


# ── 数据结构 ─────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """一局的身份（不可变，外层造；来自 ``BenchmarkEnvBuilder.resolve_identity``）。

    ``episode``：builder 局号（ood 0～49、hard-verify 0～11）；``source_episode``：xhard0 局的官方原号（新值档为
    None）；``key``：``<Task>_<tier>_<seed>``；``max_steps``：1300／1800；``strict_cap``：ood 为真；``out_dir``：本局原始
    产物目录 ``raw/<Task>_ep<N>_<tier>/``（模型的 ``trace.jsonl`` 写在这里）。"""

    dataset: str
    task: str
    episode: int
    source_episode: int | None
    tier: str
    seed: int
    candidate: int | None
    spec_sha256: str | None
    key: str
    max_steps: int
    strict_cap: bool
    attempt: int
    policy_seed: int
    out_dir: str

    def __post_init__(self):
        if self.dataset not in DATASET_MAX_STEPS:
            raise ValueError(f"dataset={self.dataset!r} 不是 {tuple(DATASET_MAX_STEPS)} 之一")

    @property
    def ep_label(self) -> int:
        """文件名里的 ``ep<N>``：ood 用 builder 局号，hard-verify 用官方原 episode 号。"""
        if self.dataset == HARD_VERIFY:
            if self.source_episode is None:
                raise ValueError("hard-verify 局缺 source_episode")
            return int(self.source_episode)
        return int(self.episode)

    @property
    def raw_name(self) -> str:
        return f"{self.task}_ep{self.ep_label}_{self.tier}"

    def identity(self) -> dict:
        """写进结果行与 trace header 的身份字段。"""
        return {"dataset": self.dataset, "task": self.task, "episode": self.episode, "builder_episode": self.episode,
                "source_episode": self.source_episode, "tier": self.tier, "seed": self.seed,
                "candidate": self.candidate, "spec_sha256": self.spec_sha256, "key": self.key,
                "max_steps": self.max_steps, "strict_cap": self.strict_cap, "attempt": self.attempt,
                "policy_seed": self.policy_seed}


@dataclasses.dataclass
class EpisodeResult:
    """``run_episode`` 返回并落 ``result.json`` 的结果（字段即契约）。

    ``status`` 只取 success／fail／timeout；``error_kind`` 记被并入 fail 的原终态（``error``），其余为 None；
    计数字段 ``task_success``／``steps``／``exec_steps``／``reset_calls``／``decisions``／``demo_frames`` 一律显式写数；
    ``extra`` 里的模型专属字段（如 PonderPounce 的 ``pp_sid_use_index``）在 ``to_dict`` 时平铺到顶层。"""

    dataset: str
    task: str
    episode: int
    source_episode: int | None
    tier: str
    seed: int
    candidate: int | None
    spec_sha256: str | None
    key: str
    max_steps: int
    strict_cap: bool
    attempt: int
    policy_seed: int
    model: str
    policy_label: str
    server_seed: int | None = None
    status: str = "fail"
    task_success: int = 0
    steps: int = 0
    exec_steps: int = 0
    cap_hit: bool = False
    reset_calls: int = 0
    demo_frames: int = 0
    decisions: int = 0
    error: str | None = None
    error_kind: str | None = None
    infra: bool = False
    infra_reason: str | None = None
    budget_exhausted: bool = False
    run_blocked: bool = False
    recorder_verify: str | None = None
    raw_dir: str | None = None
    video: str | None = None
    video_error: str | None = None
    task_goal: str | None = None
    timing: dict = dataclasses.field(default_factory=dict)
    extra: dict = dataclasses.field(default_factory=dict)
    host: str = dataclasses.field(default_factory=socket.gethostname)
    t_start: float = 0.0
    t_end: float = 0.0

    @classmethod
    def from_spec(cls, spec: EpisodeSpec, policy) -> "EpisodeResult":
        ident = spec.identity()
        ident.pop("builder_episode")
        return cls(**ident, model=str(getattr(policy, "model", "?")),
                   policy_label=str(getattr(policy, "label", None) or getattr(policy, "model", "?")))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        extra = d.pop("extra") or {}
        for k, v in extra.items():
            d.setdefault(k, v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeResult":
        names = {f.name for f in dataclasses.fields(cls)}
        core = {k: v for k, v in d.items() if k in names and k != "extra"}
        return cls(**core, extra={k: v for k, v in d.items() if k not in names})


def normalize_play_output(out: Any) -> dict:
    """核 ``PlayOutput`` 契约：dict、必有字段齐全、status 合法；不满足抛 ``PlayContractError``。"""
    if not isinstance(out, dict):
        raise PlayContractError(f"play 必须返回 dict，得到 {type(out).__name__}")
    missing = [k for k in PLAY_REQUIRED if k not in out]
    if missing:
        raise PlayContractError(f"PlayOutput 缺必有字段 {missing}")
    if out["status"] not in PLAY_STATUSES:
        raise PlayContractError(f"PlayOutput.status={out['status']!r} 不在 {PLAY_STATUSES}")
    return dict(out)


# ── 路径 ─────────────────────────────────────────────────────────────────────


def run_dir(out_dir: str | Path, policy_label: str, dataset: str, policy_seed: int) -> Path:
    """``<out>/rollouts/<模型>/<数据集>/seed<policy_seed>/``。"""
    return Path(out_dir) / "rollouts" / str(policy_label) / dataset / f"seed{int(policy_seed)}"


def _label(policy) -> str:
    return str(getattr(policy, "label", None) or getattr(policy, "model", "model"))


def make_spec(policy, dataset: str, task: str, episode: int, out_dir: str | Path, *, attempt: int = 1
              ) -> tuple[EpisodeSpec, dict]:
    """解析身份并造 ``EpisodeSpec``；返回 ``(spec, resolved)``。dataset 不是两者之一直接 ``ValueError``。"""
    if dataset not in DATASET_MAX_STEPS:
        raise ValueError(f"dataset={dataset!r} 不是 {tuple(DATASET_MAX_STEPS)} 之一")
    resolved = dict(builder_for(task, dataset).resolve_identity(int(episode)))
    tier, seed = resolved["tier"], int(resolved["seed"])
    src = resolved.get("source_episode")
    rdir = run_dir(out_dir, _label(policy), dataset, policy.policy_seed)
    spec = EpisodeSpec(dataset=dataset, task=task, episode=int(episode),
                       source_episode=None if src is None else int(src), tier=tier, seed=seed,
                       candidate=resolved.get("candidate"), spec_sha256=resolved.get("spec_sha256"),
                       key=f"{task}_{tier}_{seed}", max_steps=DATASET_MAX_STEPS[dataset],
                       strict_cap=dataset == OOD, attempt=int(attempt), policy_seed=int(policy.policy_seed),
                       out_dir="")
    spec = dataclasses.replace(spec, out_dir=str(rdir / "raw" / spec.raw_name))
    return spec, resolved


def result_path(policy, dataset: str, task: str, episode: int, out_dir: str | Path) -> Path:
    """该局 ``result.json`` 的路径（续跑跳过用；只解析身份，不碰环境与模型）。"""
    spec, _ = make_spec(policy, dataset, task, episode, out_dir)
    return Path(spec.out_dir) / "result.json"


# ── 写盘 ─────────────────────────────────────────────────────────────────────


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


def append_jsonl(path: Path, row: dict) -> None:
    """追加一行并 fsync；前一行是崩溃留下的半行时先补换行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((dumps(row) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(dumps(obj) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_result(result: EpisodeResult, spec: EpisodeSpec) -> None:
    raw = Path(spec.out_dir)
    rdir = raw.parent.parent
    d = result.to_dict()
    write_json_atomic(raw / "result.json", d)
    append_jsonl(rdir / "results.jsonl", d)


def _progress(spec: EpisodeSpec, phase: str, step: int = 0) -> None:
    rdir = Path(spec.out_dir).parent.parent
    try:
        write_json_atomic(rdir / "progress.json", {"pid": os.getpid(), "host": socket.gethostname(),
                                                   "key": spec.key, "task": spec.task, "episode": spec.episode,
                                                   "dataset": spec.dataset, "phase": phase, "step": int(step),
                                                   "t": time.time()})
    except OSError:
        pass


# ── 看门狗 ───────────────────────────────────────────────────────────────────


class _Watchdog:
    """单局看门狗：每个阶段一个绝对期限计时器；到点由计时器线程写 ``INFRA_TIMEOUT`` 结果、打印 ``SERVER_LEFT``，再
    ``HARD_EXIT(75)``（只杀客户端）。"""

    def __init__(self, policy, spec: EpisodeSpec, result: EpisodeResult, state: dict):
        self.policy, self.spec, self.result, self.state = policy, spec, result, state
        self.lock = threading.Lock()
        self.timers: dict[str, threading.Timer] = {}

    def arm(self, phase: str, limit: float | None) -> None:
        if not limit or limit <= 0:
            return
        t = threading.Timer(float(limit), self._fire, args=(phase, float(limit)))
        t.daemon = True
        self.timers[phase] = t
        t.start()

    def disarm(self, phase: str) -> None:
        with self.lock:
            self.state[f"{phase}_done"] = True
        t = self.timers.pop(phase, None)
        if t is not None:
            t.cancel()

    def disarm_all(self) -> None:
        for p in list(self.timers):
            self.disarm(p)

    def _fire(self, phase: str, limit: float) -> None:
        with self.lock:
            if self.state.get(f"{phase}_done") or self.state.get("fired"):
                return
            self.state["fired"] = phase
            sess = self.state.get("session")
            res = dataclasses.replace(self.result)
            res.status, res.task_success, res.error_kind = "fail", 0, "error"
            res.infra, res.infra_reason = True, WATCH_REASONS.get(phase, f"deadline_{phase}")
            res.error = f"INFRA_TIMEOUT phase={phase} limit_s={limit:.0f}"
            res.exec_steps = int(getattr(sess, "steps", 0) or 0)
            res.reset_calls = int(getattr(sess, "reset_calls", 0) or 0)
            res.cap_hit = bool(getattr(sess, "cap_hit", False))
            res.demo_frames = int(((getattr(sess, "timing", None) or {}).get("demo_frames")) or 0)
            res.t_end = time.time()
            self.state["timeout_result"] = res
            try:
                _write_result(res, self.spec)
            finally:
                print(f"INFRA_TIMEOUT phase={phase} key={self.spec.key} limit_s={limit:.0f}", flush=True)
                for srv in _servers_of(self.policy):
                    print(srv.left_line(), flush=True)
        HARD_EXIT(EXIT_WALL)


def _servers_of(policy) -> list:
    fn = getattr(policy, "servers", None)
    try:
        return list(fn()) if callable(fn) else []
    except Exception:  # noqa: BLE001
        return []


def wall_limit(policy, first: bool, override: float | None = None) -> float:
    base = override if override is not None else (getattr(policy, "episode_wall_s", None)
                                                  or DEFAULT_WALL_S.get(getattr(policy, "model", ""), FALLBACK_WALL_S))
    return float(base) + (FIRST_EPISODE_EXTRA_S if first else 0.0)


# ── run_episode ──────────────────────────────────────────────────────────────


def _default_recorder(raw: Path, meta: dict):
    from robomme_ood_eval.record.recorder import EpisodeRecorder

    return EpisodeRecorder(raw, meta, overwrite=True)


def _render(spec: EpisodeSpec, result: EpisodeResult, official_root: str | Path | None) -> None:
    """有 ``trace.jsonl`` 时出官方版式网站视频到 ``videos/``；失败只记 ``video_error``，不改终态。"""
    raw = Path(spec.out_dir)
    if not (raw / "trace.jsonl").is_file():
        result.video_error = "no_trace"
        return
    try:
        from robomme_ood_eval.record import official_render

        out = official_render.render_video(raw, raw.parent.parent / "videos", episode_id=spec.ep_label,
                                           terminal=result.status, official_root=official_root)
        result.video = None if out is None else str(Path(out).relative_to(raw.parent.parent))
        if out is None:
            result.video_error = "no_frame"
    except Exception as e:  # noqa: BLE001
        result.video_error = f"{type(e).__name__}: {e}"[:800]


def run_episode(policy, dataset: str, task: str, episode: int, out_dir: str | Path, expect: dict | None = None, *,
                attempt: int = 1, ledger: Any = None, recorder_factory: Callable[[Path, dict], Any] | None = None,
                render: bool = True, official_root: str | Path | None = None, wall_s: float | None = None,
                first_infer_s: float | None = FIRST_INFER_DEADLINE_S,
                media_s: float | None = MEDIA_DEADLINE_S) -> EpisodeResult:
    """跑一局（顺序见模块说明）。``out_dir`` 是产物根（其下 ``rollouts/<模型>/<数据集>/seed<n>/``）。

    关键字参数只供 GL 席位与单测：``attempt``（尝试号）、``ledger``（带 ``claim(what)`` 的 reset 预算账本）、
    ``recorder_factory(raw_dir, meta)``（缺省真实 ``EpisodeRecorder``）、``render``（是否出网站视频）、
    ``official_root``（官方 ``RolloutRecorder`` 所在仓根）、三个看门狗期限（None 或 0 不挂）。"""
    t_start = time.time()
    spec, resolved = make_spec(policy, dataset, task, episode, out_dir, attempt=attempt)
    raw = Path(spec.out_dir)
    result = EpisodeResult.from_spec(spec, policy)
    result.t_start = t_start
    result.raw_dir = str(raw.relative_to(raw.parent.parent))
    result.server_seed = policy.server_seed() if callable(getattr(policy, "server_seed", None)) else None
    # ① 身份
    if expect is not None:
        bad = check_identity(resolved, expect, dataset=dataset)
        if bad:
            result.status, result.error_kind, result.run_blocked = "fail", "error", True
            result.error = f"IDENTITY_MISMATCH {bad}"[:800]
            result.extra["resolved_identity"] = resolved
            result.t_end = time.time()
            _write_result(result, spec)
            print(f"RUN_BLOCKED reason=identity key={spec.key} dataset={dataset} detail={bad}", flush=True)
            raise IdentityMismatch(bad)
    # ② 局前：客户端状态、会话标识、探活、局前拒绝（ServerDead／AstraStop 原样上抛，不写结果）
    _progress(spec, "reset")
    policy.reset(spec)
    # ③ 录制器 + EnvSession + 看门狗
    state: dict[str, Any] = {}
    dog = _Watchdog(policy, spec, result, state)
    first = int(getattr(policy, "episodes_run", 0) or 0) == 0
    meta = {"identity": spec.identity(), "model": result.model, "policy_label": result.policy_label,
            "resolved_identity": resolved}
    try:
        recorder = (recorder_factory or _default_recorder)(raw, meta)
    except Exception as e:  # noqa: BLE001 录制器建不起来（如磁盘满）= 基础设施故障
        result.status, result.error_kind, result.infra, result.infra_reason = "fail", "error", True, "recorder"
        result.error = f"RecorderError: init: {type(e).__name__}: {e}"[:800]
        result.t_end = time.time()
        _write_result(result, spec)
        return result

    def on_first_step():
        dog.disarm("first_infer")
        _progress(spec, "episode")

    session = EnvSession(task, spec.episode, dataset=dataset, max_steps=spec.max_steps, recorder=recorder,
                         builder=builder_for(task, dataset), progress_cb=lambda s: _progress(spec, "episode", s),
                         step_cap=spec.max_steps if spec.strict_cap else None, ledger=ledger,
                         first_step_cb=on_first_step)
    state["session"] = session
    dog.arm("episode_wall", wall_limit(policy, first, wall_s))
    dog.arm("first_infer", first_infer_s)
    _progress(spec, "first_infer")
    t0 = time.perf_counter()
    out: dict = {}
    budget_exc: BaseException | None = None
    try:
        try:
            session.build()
        except Exception as e:  # noqa: BLE001 构建失败（如 Vulkan）按基础设施故障记，不进 play（M2）
            if session.budget_exhausted:
                budget_exc = e
                out = {"status": "error", "error": f"{type(e).__name__}: {e}"[:800], "infra": False,
                       "infra_reason": None}
            else:
                out = {"status": "error", "error": f"{type(e).__name__}: {e}"[:800], "infra": True,
                       "infra_reason": "env_build"}
        else:
            # ④ play
            try:
                out = normalize_play_output(policy.play(session, spec, recorder))
            except StepCapReached as e:
                out = {"status": "timeout", "error": str(e), "infra": False, "infra_reason": None}
            except RecorderError as e:
                out = {"status": "error", "error": f"RecorderError: {e}"[:800], "infra": True,
                       "infra_reason": "recorder"}
            except Exception as e:  # noqa: BLE001 模型侧普通异常：记 error，照样收尾，Policy 继续下一局
                if session.budget_exhausted:
                    budget_exc = e
                out = {"status": "error", "error": f"{type(e).__name__}: {e}"[:800], "infra": False,
                       "infra_reason": None, "traceback": traceback.format_exc()[-4000:]}
    except BaseException:
        # KeyboardInterrupt／SystemExit：只收尾不写结果
        dog.disarm_all()
        _safe(session.close)
        _safe(lambda: recorder.close({"status": "interrupted", "steps": session.steps}))
        raise
    finally:
        dog.disarm("first_infer")
        dog.disarm("episode_wall")
    if state.get("fired"):
        return state["timeout_result"]
    wall = time.perf_counter() - t0
    # ⑤ 收尾：session.close → recorder.close → 视频 → result.json
    close_error = None
    try:
        session.close()
    except Exception as e:  # noqa: BLE001
        close_error = repr(e)[:800]
    _progress(spec, "media_finalize", session.steps)
    dog.arm("media_finalize", media_s)
    rsum: dict | None = None
    try:
        rsum = recorder.close({"status": out.get("status"), "steps": out.get("steps"), "exec_steps": session.steps,
                               "cap_hit": session.cap_hit})
    except Exception as e:  # noqa: BLE001 收尾失败（如磁盘满）= 基础设施故障
        rsum = {"RECORDER_VERIFY": "ERROR", "error": f"{type(e).__name__}: {e}"[:800]}
        out.update(infra=True, infra_reason="recorder_close")
    _classify(result, out, session)
    if close_error:
        result.extra["close_error"] = close_error
    result.recorder_verify = (rsum or {}).get("RECORDER_VERIFY")
    if render:
        _render(spec, result, official_root)
    dog.disarm("media_finalize")
    if state.get("fired"):
        return state["timeout_result"]
    result.timing = {"episode_wall_s": round(wall, 3), "first_episode_in_process": first,
                     "env": dict(session.timing), "policy": out.get("timing"), "recorder": rsum}
    result.task_goal = session.task_goal
    result.t_end = time.time()
    _write_result(result, spec)
    policy.episodes_run = int(getattr(policy, "episodes_run", 0) or 0) + 1
    _progress(spec, "done", session.steps)
    print(f"EPISODE_DONE model={result.model} dataset={dataset} key={spec.key} ep={spec.ep_label} "
          f"status={result.status} exec_steps={result.exec_steps} cap_hit={result.cap_hit} infra={result.infra}",
          flush=True)
    if budget_exc is not None:
        raise budget_exc
    return result


def _safe(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


def _classify(result: EpisodeResult, out: dict, session: EnvSession) -> None:
    """执行段计数以环境侧为准；额度耗尽与步数到顶的分类覆盖模型自报；error 并入 fail。"""
    status = out.get("status", "error")
    result.steps = int(out.get("steps") or 0)
    result.exec_steps = int(session.steps)
    result.cap_hit = bool(session.cap_hit)
    result.reset_calls = int(session.reset_calls)
    result.demo_frames = int(session.timing.get("demo_frames") or 0)
    result.decisions = int(out.get("decisions") or 0)
    result.error = out.get("error")
    result.infra = bool(out.get("infra"))
    result.infra_reason = out.get("infra_reason")
    for k, v in out.items():
        if k not in PLAY_REQUIRED and k not in ("decisions", "timing", "traceback"):
            result.extra[k] = v
    if out.get("traceback"):
        result.extra["traceback"] = out["traceback"]
    if session.budget_exhausted:
        result.budget_exhausted = True
        result.infra, result.infra_reason = False, None
        status = "error"
    elif session.cap_hit:
        if status != "timeout":
            result.extra["client_status"] = status
        status = "timeout"
        result.error = result.error or f"STEP_CAP exec_steps={session.steps} cap={session.step_cap} 未成功，按 timeout 计"
    if status == "error":
        result.status, result.error_kind = "fail", "error"
    else:
        result.status = status
    result.task_success = int(result.status == "success")


__all__ = ["EpisodeSpec", "EpisodeResult", "run_episode", "make_spec", "result_path", "run_dir", "builder_for",
           "clear_builders", "check_identity", "normalize_play_output", "DATASET_MAX_STEPS", "TASKS",
           "DEFAULT_WALL_S", "IdentityMismatch", "PlayContractError", "PLAY_REQUIRED", "DATASETS"]
