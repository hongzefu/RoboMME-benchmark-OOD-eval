#!/usr/bin/env python3
"""v7.5eval 新接口的环境侧（0929-v7.5eval-restructure-plan.md §2.4、§3.2 第 1 步 3、§4、§5）。

两部分：

1. ``EnvSession``：身份 → ``BenchmarkEnvBuilder(dataset=<--dataset>).make_env_for_episode`` → reset/step。
   * ``reset()`` 原样返回 ``(obs, info)``；reset 期间录制器只入队（``set_phase("reset")``），返回后切 ``"run"``；
   * ``step(action)`` 把 action **原样**交给 ``env.step``，并记录实际交出去的数组（``exec_action``）与返回的当前帧、
     状态、终态字段；``env.step`` 抛出的异常原样上抛（由策略客户端按旧官方语义处理）；
   * 逐段计时：环境构建、reset、逐步 env 时间、录制开销、close。
2. ``run`` 子命令：常驻客户端进程。import 与 Vulkan 设备只建一次，逐身份建 EnvSession + EpisodeRecorder，调用
   策略模块（``POLICY_MODULES``：framesamp_modul_client／smvla_client／groundsg_client／pp_client）的
   ``run_episode(session, identity, conn_info, recorder) -> dict``，结果写 ``<out>/results.jsonl``（合同格式），
   进度心跳写 ``<out>/progress.json``。

数据集与步数（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.2）：``--dataset {ood,hard-verify}`` 与
``--max-steps`` 都必填、无默认值；步数上限不再按档查表，一律取 ``--max-steps``。启动约定：ood →
``--max-steps 1600 --strict-cap``；hard-verify → ``--max-steps 1300``、不带 ``--strict-cap``。``--strict-cap`` 打开时
``EnvSession.step`` 在已执行 ``max_steps`` 步后不再进入环境、抛 ``StepCapReached``，``run_one`` 收为 ``status=timeout``
（``cap_hit=true``）；不带时步数由策略客户端自己的循环决定（官方循环可执行第 ``max_steps+1`` 步）。

身份清单一律是 ``eval_manifest.py`` 产出的分片 JSON（``shard-NN.json``），身份模式由 ``--dataset`` 决定：
ood 逐键严格核对 tier／seed／candidate／spec_sha256（spec_sha256 为 64 位串）；hard-verify 要求
tier=="xhard0"、candidate 与 spec_sha256 为 null、source_episode 为整数且与 builder 解析结果相等。不符即
「运行阻塞」（写一条 ``run_blocked`` 记录、打印 ``RUN_BLOCKED``、退出码 3）。

持久账本（``--ledger``，JSONL + fsync；契约 C2）两个数据集都启用，与身份模式、``--strict-cap`` 解耦：
``EnvSession.build``／``reset`` 每次实际调用前领一次 reset 额度，耗尽抛 ``ResetBudgetExhausted`` → 退出码 5；infra
重试额度与每身份至多 2 次尝试都从账本统计，进程重启不刷新；第一次终态写 ``accept``，之后的终态记 ``late``；
「attempt_end 已写、accept 未写」的崩溃窗口在恢复时补写 ``accept``。

退出码：0 全部完成；3 运行阻塞；5 reset 额度耗尽（run_seat.sh 不得重启）；6 跑完仍有身份无权威终态（infra
重试额度或每身份 2 次尝试用尽，打印 ``RUN_INCOMPLETE``）；75 单局墙钟超时（基础设施超时，已记录，由 run_seat.sh 重起）。

第二阶段共享预算（1005 计划第二部分一节 S8；R2 门控）：只有给了 ``--budget-ledger``、或环境变量
``SGEVAL_BUDGET_LEDGER`` 非空、或构造 ``AttemptLedger(shared=...)`` 时才打开，否则行为与 BASE 相同。打开后：
每次尝试在写 ``attempt_start`` 前向共享账本（``budget_ledger.py``）预约一条轨迹（不足即 ``RUN_BLOCKED reason=budget``、
退出码 5、不进入 attempt），``EnvSession._claim`` 每次实际 build／reset 记进共享账本；基础设施重试名额改由共享账本
原子领取（跨原侧／新侧、跨席位、重启换节点不刷新），中断分 ``infra`` 与 ``expired``（有 Slurm 到期证据）分别计数，
两者都占每身份 ``V8_MAX_ATTEMPTS`` 名额。

第三阶段（1006 计划第二部分八.9、八.10 第 4／5 条；接口冻结说明第二、三、七、八节）：

* ``--policy-seed <n>`` 必填（CLI 缺失即 ``RUN_BLOCKED reason=policy_seed``、退出 3），进 ``seat_info``、结果行
  ``policy_seed``，共享账本路线 ``<policy_label>/seed<n>/new``（GroundSG 为 ``groundsg/<variant>/seed<n>/new``）；
  结果行 ``server_seed`` 从服务元数据 ``server-metadata-<port>.json`` 的 ``policy_seed`` 反查（``--server-metadata``）。
* GroundSG 第三个变体 ``ground-sg-memer`` 与 ``--memer-adapter <dir>`` 配对；``ground-sg-qwenvl`` 只配
  ``--qwenvl-groundsg-adapter``；错配或目录不存在 ``RUN_BLOCKED reason=variant_pairing``。
* 预算参数 ``--budget-ledger --trajectory-cap --shared-infra-cap --expired-cap --planned-first-tries`` 在 CLI 必填（缺
  任一 ``RUN_BLOCKED reason=budget_args``，不回落 ``budget_ledger.py`` 常量默认值）；``--reset-budget`` 改为可选：不给时
  只计量（``reset_claim`` 照写、共享账本照记），不抛 ``ResetBudgetExhausted``。打开账本先核坏行与 config（不符
  ``RUN_BLOCKED reason=ledger_corrupt|budget_config``），读取待跑身份之前取分片排他 lease（拿不到
  ``RUN_BLOCKED reason=lease_held``）；每次尝试的 ``claim_retry``／``reserve``／``attempt_start`` 共用同一 token
  ``<route>|<key>|a<attempt_no>``（结果行 ``budget_token``），首试 ``kind_of_try=first``、重试 ``recovery``。
* 尝试账本路线核对（1006 MERGE-2，修 R4-D2）：共享模式下取得 lease 之后、读取待跑身份之前，核对尝试账本已有
  ``attempt_start`` 行的 ``route``；存在与当前路线（含 ``seed<n>``）不同者、或缺 ``route`` 字段的旧行（``found=legacy``），
  即打印 ``RUN_BLOCKED reason=route_mismatch ledger=<路径> found=<已有route> want=<当前route>``、退出 3，
  不写任何结果行（避免把别组种子的 accepted 当作已完成跳过、同一目录混两组结果）。账本无 ``attempt_start`` 行时照常跑。
* 执行期限：``--context-deadline-s``（策略上下文加载）、``--first-infer-deadline-s``（本局第一次 ``step`` 之前）、
  ``--media-deadline-s``（录制器收尾）各有绝对期限，超期写本局结果 ``status=error infra=true
  infra_reason=deadline_<phase>``、打印 ``DEADLINE_EXCEEDED`` 并以 75 退出（run_seat.sh 重起客户端）。
  ``progress.json`` 记 ``phase``（``context_load／first_infer／episode／media_finalize／done／finished``）、
  ``identity``、``step``、``t``，只在这些量变化时更新，run_seat.sh 据此判无进展。
* 结果行与 progress 记 ``effective_cap``（``--strict-cap`` 时为 ``--max-steps``，否则 null）；ood 的启动约定改为
  ``--max-steps 1800 --strict-cap``（第 1800 次 step 照常进环境，第 1801 次在进入环境前被拒）。

本目录只挂在 ``sys.path`` 末尾（防止同目录模块遮蔽标准库），同目录模块按文件路径加载。
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p and str(_Path(p).resolve()) != _HERE] + [_HERE]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "src",):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

XHARD0 = "xhard0"
#: 两个评估数据集：新值档（V9）与官方 test 的 hard 子集（xhard0）
OOD = "ood"
HARD_VERIFY = "hard-verify"
DATASETS = (OOD, HARD_VERIFY)
#: 策略标签（官方名；FrameSamp+Modulation 为 ``perceptual-framesamp-modul``、GroundSG 为 ``groundsg``）
POLICIES = ("perceptual-framesamp-modul", "smvla", "groundsg", "pp")
#: 策略标签 → 同目录策略模块名（按 load_sibling(POLICY_MODULES[policy]) 加载）
POLICY_MODULES = {"perceptual-framesamp-modul": "framesamp_modul_client", "smvla": "smvla_client",
                  "groundsg": "groundsg_client", "pp": "pp_client"}
#: GroundSG 三个子目标来源变体（第三阶段加 MemER）
GROUNDSG_VARIANTS = ("ground-sg-oracle", "ground-sg-qwenvl", "ground-sg-memer")
#: 变体 → 必须且只能给的 adapter 参数（argparse dest, CLI 名）；oracle 两者都不许给
VARIANT_ADAPTERS = {"ground-sg-qwenvl": ("qwenvl_groundsg_adapter", "--qwenvl-groundsg-adapter"),
                    "ground-sg-memer": ("memer_adapter", "--memer-adapter")}
#: 第三阶段 CLI 必填的预算参数（argparse dest, CLI 名）；缺任一即 RUN_BLOCKED reason=budget_args
BUDGET_ARGS = (("budget_ledger", "--budget-ledger"), ("trajectory_cap", "--trajectory-cap"),
               ("shared_infra_cap", "--shared-infra-cap"), ("expired_cap", "--expired-cap"),
               ("planned_first_tries", "--planned-first-tries"))
#: progress.json 的具名阶段（接口冻结说明第八节）
PHASES = ("context_load", "first_infer", "episode", "media_finalize", "done", "finished")
#: 执行期限缺省值（秒）：上下文加载、首推、媒体收尾
DEFAULT_CONTEXT_DEADLINE_S = 1800.0
DEFAULT_FIRST_INFER_DEADLINE_S = 1800.0
DEFAULT_MEDIA_DEADLINE_S = 1200.0
EXIT_BLOCKED = 3
EXIT_BUDGET = 5
EXIT_INCOMPLETE = 6
EXIT_WALL = 75
DEFAULT_SHUFFLE_SEED = 20260930
TERMINAL_STATUSES = ("success", "fail", "timeout")
#: V8 每身份至多尝试次数（首试 + 1 次基础设施重试）
V8_MAX_ATTEMPTS = 2
#: 执行身份行（eval_manifest.py shard-NN.json 的元素）必须恰有的字段；与 eval_manifest.SHARD_ROW_KEYS 同步
V8_IDENTITY_KEYS = ("task", "tier", "seed", "candidate", "builder_episode", "source_episode", "spec_sha256", "key")
#: 「未提供」哨兵（R2）：可选新参数缺省时保持 BASE 行为；None 表示显式关闭
_UNSET = object()
#: S8 共享预算账本的门控环境变量；值为账本路径，空或未设即关闭
ENV_BUDGET_LEDGER = "SGEVAL_BUDGET_LEDGER"
#: Slurm 到期证据：文件路径，内容为到期（TIMEOUT）作业号（空白／逗号分隔，可直接存 ``sacct -X -n -s TO -o JobID``）
ENV_EXPIRED_JOBS = "SGEVAL_EXPIRED_JOBS"
#: 悬空尝试最后活动时刻距 Slurm 结束时刻在此秒数内且结束时刻已过 → 视为到期中断
EXPIRE_MARGIN_S = 900.0
#: 新侧每次尝试预约的 reset 计量（build 与 reset 各 1 次；六节口径）
NEW_SIDE_RESETS_PER_ATTEMPT = 2
INTERRUPTS = ("infra", "expired")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def load_sibling(name: str, alias: str | None = None):
    """按文件路径加载本目录下的模块。"""
    alias = alias or name
    if alias in sys.modules:
        return sys.modules[alias]
    path = Path(_HERE) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(arr: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def state8(joint, gripper):
    """与旧官方 ``pack_state`` 同式的 8 维状态（只用于摘要核对，不交给策略）。"""
    import numpy as np

    return np.concatenate([np.asarray(joint), np.asarray(gripper)[:1]], axis=0, dtype=np.float32)


class RecorderError(RuntimeError):
    """录制器（含磁盘写满）出错：归为基础设施故障（infra=True），不当成环境错误。"""


class StepCapReached(RuntimeError):
    """``--strict-cap``：执行段已满 ``step_cap`` 步仍未成功，第 ``step_cap+1`` 次 ``step`` 不进入环境（run_one 收为 timeout）。"""


class ResetBudgetExhausted(RuntimeError):
    """V8：持久账本里的 reset 额度（build 与 reset 各算一次）已用尽（客户端以退出码 5 停止）。"""


class _GuardedRecorder:
    """录制器代理：任何录制调用抛出的异常都转成 RecorderError（保留原因）。"""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if not callable(fn):
            return fn

        def call(*a, **k):
            try:
                return fn(*a, **k)
            except RecorderError:
                raise
            except Exception as e:  # noqa: BLE001
                raise RecorderError(f"{name}: {type(e).__name__}: {e}") from e

        return call


class NullRecorder:
    """``--no-record`` 或单测用的空录制器（接口同 recorder.EpisodeRecorder）。"""

    def set_phase(self, phase):
        pass

    def add_frames(self, stream, frames, *, tag=""):
        return []

    def add_array(self, name, arr, *, step=None):
        pass

    def add_event(self, event):
        pass

    def close(self, summary):
        return {"RECORDER_VERIFY": "SKIP"}


# ── EnvSession ─────────────────────────────────────────────────────────────


class EnvSession:
    """一局环境。``builder`` 可由常驻进程按任务缓存后传入（与旧官方「每任务一个 EnvRunner」一致）。"""

    def __init__(self, task: str, builder_episode: int, *, max_steps: int | None = None, recorder=None, builder=None,
                 progress_cb: Callable[[int], None] | None = None, progress_every: int = 16,
                 step_cap: int | None = None, claim_reset: Callable[[str], None] | None = None,
                 dataset: str = OOD, budget_claim: Any = _UNSET, first_step_cb: Callable[[], None] | None = None):
        self.task = task
        # 第三阶段：本局第一次调用 step（策略首推完成、拿到第一个动作）时调用一次，用于结束 first_infer 期限
        self.first_step_cb = first_step_cb
        # S8：budget_claim(what) 在本地额度领到后再记进共享预算账本（只告警、不拦）；缺省（_UNSET）或 None 时不记
        self.budget_claim = None if budget_claim is _UNSET else budget_claim
        self.builder_episode = int(builder_episode)
        # max_steps 只在需要自建 builder 时用（构造参数）；不再逐局传给 make_env_for_episode
        self.max_steps = None if max_steps is None else int(max_steps)
        if dataset not in DATASETS:
            raise ValueError(f"dataset={dataset!r} 不是 {DATASETS} 之一")
        self.dataset = dataset
        # step_cap=None 时不截断（不带 --strict-cap）；claim_reset(what) 在每次实际 build／reset 前调用，额度耗尽抛
        # ResetBudgetExhausted（None 时不领额度，单测用）
        self.step_cap = None if step_cap is None else int(step_cap)
        self.claim_reset = claim_reset
        self.cap_hit = False
        self.budget_exhausted = False
        self.reset_calls = 0
        self.recorder = recorder if recorder is not None else NullRecorder()
        self._rec = _GuardedRecorder(self.recorder)  # 内部录制一律经代理；对外仍暴露原录制器（策略客户端靠 is 判断）
        self._builder = builder
        self.env = None
        self.steps = 0
        self.progress_cb = progress_cb
        self.progress_every = max(1, int(progress_every))
        self.timing: dict[str, Any] = {}
        self._step_s: list[float] = []
        self._rec_s = 0.0
        self.task_goal = None

    @property
    def builder(self):
        if self._builder is None:
            from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

            if self.max_steps is None:
                raise ValueError("EnvSession 自建 builder 必须给 max_steps（步数上限由入口按数据集给出）")
            t0 = time.perf_counter()
            self._builder = BenchmarkEnvBuilder(env_id=self.task, dataset=self.dataset, action_space="joint_angle",
                                                max_steps=self.max_steps)
            self.timing["builder_init_s"] = time.perf_counter() - t0
        return self._builder

    def identity(self) -> dict:
        return self.builder.resolve_identity(self.builder_episode)

    def _claim(self, what: str) -> None:
        """V8：向持久账本领一次 reset 额度（build 与 reset 各算一次），领到后才计入 reset_calls。"""
        if self.claim_reset is not None:
            try:
                self.claim_reset(what)
            except ResetBudgetExhausted:
                self.budget_exhausted = True
                raise
        if self.budget_claim is not None:
            self.budget_claim(what)
        self.reset_calls += 1

    def build(self) -> None:
        """``make_env_for_episode(builder_episode)``：不逐局传步数，由 builder 构造参数 ``max_steps`` 决定
        （DemonstrationWrapper 内部 +2，与官方相同）。"""
        if self.env is not None:
            return
        self._claim("build")
        self._rec.set_phase("reset")
        t0 = time.perf_counter()
        self.env = self.builder.make_env_for_episode(self.builder_episode)
        self.timing["env_build_s"] = time.perf_counter() - t0

    def reset(self):
        """原样返回 ``env.reset()`` 的 ``(obs, info)``；返回后记录原始帧与数组（录制器此时仍在 reset 阶段，只入队）。"""
        import numpy as np

        self.build()
        self._claim("reset")
        self._rec.set_phase("reset")
        t0 = time.perf_counter()
        obs, info = self.env.reset()
        self.timing["reset_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        goal = info.get("task_goal") if isinstance(info, dict) else None
        self.task_goal = goal[0] if isinstance(goal, list) else goal
        front = np.stack(obs["front_rgb_list"])
        wrist = np.stack(obs["wrist_rgb_list"])
        self._rec.add_frames("front", front, tag="reset")
        self._rec.add_frames("wrist", wrist, tag="reset")
        for key in ("joint_state_list", "gripper_state_list", "eef_state_list"):
            if key in obs and obs[key] is not None and len(obs[key]):
                self._rec.add_array(f"reset_{key[:-5]}", np.stack([np.asarray(x) for x in obs[key]]))
        s8 = [sha(state8(j, g)) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])]
        self._rec.add_event({"kind": "env_reset", "task": self.task, "builder_episode": self.builder_episode,
                                 "frames": int(front.shape[0]), "demo_frames": int(front.shape[0]) - 1,
                                 "front": [sha(f) for f in front], "wrist": [sha(f) for f in wrist], "state8": s8,
                                 "task_goal": self.task_goal, "status": info.get("status") if isinstance(info, dict) else None,
                                 "reset_s": self.timing["reset_s"]})
        self._rec_s += time.perf_counter() - t1
        self.timing["demo_frames"] = int(front.shape[0]) - 1
        self._rec.set_phase("run")
        return obs, info

    def step(self, action):
        """action 原样交给 ``env.step``；记录交出去的数组与返回的当前帧、状态、终态。

        ``--strict-cap``（``step_cap`` 非空）：已执行 ``step_cap`` 步后再调用即不进入环境，置 ``cap_hit`` 并抛 ``StepCapReached``；
        第 ``step_cap`` 步及之前环境报的终态照常返回。"""
        import numpy as np

        if self.first_step_cb is not None:
            cb, self.first_step_cb = self.first_step_cb, None
            cb()
        if self.step_cap is not None and self.steps >= self.step_cap:
            self.cap_hit = True
            self._rec.add_event({"kind": "step_cap_reached", "step": self.steps, "cap": self.step_cap})
            raise StepCapReached(f"STEP_CAP exec_steps={self.steps} cap={self.step_cap}")
        t0 = time.perf_counter()
        a = np.array(action, copy=True)
        self._rec.add_array("exec_action", a, step=self.steps)
        self._rec.add_event({"kind": "env_step_action", "step": self.steps, "sha": sha(a), "dtype": a.dtype.str,
                                 "shape": list(a.shape)})
        t1 = time.perf_counter()
        try:
            out = self.env.step(action)
        except Exception as e:
            self._step_s.append(time.perf_counter() - t1)
            self._rec.add_event({"kind": "env_step_exception", "step": self.steps,
                                     "error": f"{type(e).__name__}: {e}"[:800]})
            self.steps += 1
            raise
        t2 = time.perf_counter()
        self._step_s.append(t2 - t1)
        obs, reward, terminated, truncated, info = out
        status = info.get("status") if isinstance(info, dict) else None
        ev: dict[str, Any] = {"kind": "env_step", "step": self.steps, "terminated": bool(terminated),
                              "truncated": bool(truncated), "status": status, "reward": _scalar(reward)}
        if obs is None:
            ev.update(front=None, wrist=None, obs_none=True)
        else:
            front = np.stack(obs["front_rgb_list"])
            wrist = np.stack(obs["wrist_rgb_list"])
            self._rec.add_frames("front", front, tag=f"step{self.steps}")
            self._rec.add_frames("wrist", wrist, tag=f"step{self.steps}")
            joint, grip = obs["joint_state_list"][-1], obs["gripper_state_list"][-1]
            self._rec.add_array("joint_state", np.asarray(joint), step=self.steps)
            self._rec.add_array("gripper_state", np.asarray(grip), step=self.steps)
            if obs.get("eef_state_list"):
                self._rec.add_array("eef_state", np.asarray(obs["eef_state_list"][-1]), step=self.steps)
            ev.update(front=[sha(f) for f in front], wrist=[sha(f) for f in wrist], state8=sha(state8(joint, grip)))
        self._rec.add_event(ev)
        self.steps += 1
        self._rec_s += (t1 - t0) + (time.perf_counter() - t2)
        if self.progress_cb is not None and self.steps % self.progress_every == 0:
            self.progress_cb(self.steps)
        return out

    def close(self) -> None:
        import numpy as np

        if self.env is not None:
            t0 = time.perf_counter()
            try:
                self.env.close()
            finally:
                self.timing["env_close_s"] = time.perf_counter() - t0
                self.env = None
        if self._step_s:
            s = np.array(self._step_s)
            self.timing.update(step_n=int(s.size), step_total_s=float(s.sum()), step_mean_s=float(s.mean()),
                               step_p50_s=float(np.percentile(s, 50)), step_p95_s=float(np.percentile(s, 95)),
                               step_first_s=float(s[0]))
        self.timing["record_overhead_s"] = self._rec_s


def _scalar(x):
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return None


# ── 进程级信息 ──────────────────────────────────────────────────────────────


def timed_imports() -> dict:
    """分记 torch / sapien / mani_skill / robomme_hard 的 import 时间（按依赖顺序先后 import，后者不含前者）。"""
    out: dict[str, Any] = {}
    for name in ("numpy", "torch", "sapien", "mani_skill", "gymnasium", "robomme_hard"):
        t0 = time.perf_counter()
        __import__(name)
        out[f"import_{name}_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    import robomme_hard.robomme_env  # noqa: F401 注册 16 个环境
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder  # noqa: F401

    out["import_robomme_hard_env_s"] = time.perf_counter() - t0
    return out


def env_versions() -> dict:
    import mani_skill
    import numpy
    import sapien
    import torch

    import robomme_hard

    return {"sapien": getattr(sapien, "__version__", None), "mani_skill": getattr(mani_skill, "__version__", None),
            "torch": torch.__version__, "numpy": numpy.__version__, "python": sys.version.split()[0],
            "robomme_hard_file": robomme_hard.__file__, "executable": sys.executable}


def gpu_info() -> dict:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip() if cvd else "0"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader", "-i", first],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        name, uuid, drv = [x.strip() for x in out[0].split(",")]
        return {"gpu_name": name, "gpu_uuid": uuid, "gpu_driver": drv, "cuda_visible_devices": cvd}
    except Exception as e:  # noqa: BLE001
        return {"gpu_name": None, "gpu_uuid": None, "gpu_driver": None, "cuda_visible_devices": cvd,
                "gpu_error": repr(e)}


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain", "--", "scripts/eval-official", "src"))}


def cpu_info() -> dict:
    try:
        model = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                      if l.startswith("model name")), None)
    except OSError:
        model = None
    return {"cpu_model": model, "affinity_n": len(os.sched_getaffinity(0)),
            "affinity": sorted(os.sched_getaffinity(0))}


# ── 身份 ────────────────────────────────────────────────────────────────────


def v8_key(row: dict) -> str:
    """V8 身份键：结果、账本、录像目录统一用它。"""
    return f"{row['task']}_{row['tier']}_{int(row['seed'])}"


def key_of(row: dict) -> str:
    """身份行与结果行带 ``key``（= ``<task>_<tier>_<seed>``）时直接用它，否则按契约现算。"""
    if row.get("key"):
        return str(row["key"])
    return v8_key(row)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _same_int_or_null(a: Any, b: Any) -> bool:
    """两边同为 null，或同为（非 bool 的）整数且相等。"""
    return (a is None and b is None) or (_is_int(a) and _is_int(b) and a == b)


def validate_v8_identity(row: dict, dataset: str = OOD) -> str | None:
    """执行身份行的结构核对（字段齐全、nullable 严格、key 自洽）；不符返回说明。身份模式由 ``dataset`` 决定：

    * ood：spec_sha256 为 64 位串，candidate 为整数或 null；
    * hard-verify：tier=="xhard0"、candidate 与 spec_sha256 为 null、source_episode 为整数。"""
    if dataset not in DATASETS:
        return f"dataset={dataset!r}"
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


def check_identity(resolved: dict, want: dict, *, dataset: str = OOD) -> str | None:
    """builder 解析出的身份必须与清单一致；不一致返回说明（运行阻塞）。

    ood：tier／seed／candidate／spec_sha256 逐键严格相等（candidate 可空，两边同为 null 或同一整数）。
    hard-verify：两边 tier 都是 xhard0，seed 相等，candidate 与 spec_sha256 两边都为 null，source_episode 两边同为
    整数且相等。"""
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
    if not (isinstance(want.get("spec_sha256"), str) and resolved.get("spec_sha256") == want.get("spec_sha256")):
        bad.append(f"spec_sha256 builder={resolved.get('spec_sha256')} want={want.get('spec_sha256')}")
    return "; ".join(bad) or None


def order_identities(rows: list[dict], order: str, shuffle_seed: int) -> list[dict]:
    rows = list(rows)
    if order == "reverse":
        rows.reverse()
    elif order == "shuffle":  # 新值局 source_episode 为空：按 (task, tier, seed) 定序后再打乱
        rows.sort(key=lambda r: (r["task"], r["tier"], int(r["seed"])))
        random.Random(shuffle_seed).shuffle(rows)
    return rows


def read_results(path: Path) -> list[dict]:
    """读逐局结果行；历史行里的旧策略标签／数据集名／路线经 ``official_defs.canonical_row`` 映射成官方名。"""
    canon = load_sibling("official_defs").canonical_row
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(canon(json.loads(line)))
            except json.JSONDecodeError:
                continue  # 崩溃留下的半行
    return rows


def append_result(path: Path, record: dict) -> None:
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


# ── V8 持久尝试账本 ─────────────────────────────────────────────────────────


def _int_or_none(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _open_shared(shared: Any, caps: dict | None = None):
    """S8 门控：``_UNSET`` → 取环境变量 ``SGEVAL_BUDGET_LEDGER``（空即关闭）；None／空串 → 关闭；路径 → 打开
    ``budget_ledger.BudgetLedger``（第三阶段 ``caps`` 给出 config 四参数时一并传入）；其他对象（已打开的账本，单测注入）
    原样返回。"""
    if shared is _UNSET:
        shared = os.environ.get(ENV_BUDGET_LEDGER) or None
    if shared is None or shared == "":
        return None
    if isinstance(shared, (str, Path)):
        return load_sibling("budget_ledger").BudgetLedger(shared, **(caps or {}))
    return shared


class AttemptLedger:
    """V8 持久尝试账本（JSONL，追加写 + fsync；契约 C2）。一个账本只有一个写者（一席一策略一个客户端进程）。

    行：``{"t","kind","key","attempt_id","attempt_no","seat","policy",...}``，kind ∈
    ``budget | budget_raise | attempt_start | reset_claim | attempt_end | accept``。
    全部计数从账本内容推出（进程重启读回，不刷新）：

    * reset 额度：``reset_claim`` 行数（历史账本里的金丝雀行同样计入）对 ``--reset-budget``；
    * 作废尝试：``attempt_end.budget_exhausted=true`` 的尝试没有执行段（额度在 build／reset 前就被拒），不计入每身份
      2 次上限、也不计入 infra 重试额度；
    * infra 重试：``attempt_start.retry=true`` 且未作废的尝试数，对 ``--infra-retry-budget``；
    * 权威终态：每身份第一条 ``accept`` 行的 ``accepted_attempt_id``；success／fail／timeout 与非 infra 错误
      （status=error、infra=false）都 accept，infra 错误不 accept（重试额度或 2 次用满时该身份无 accept，汇总计 missing）。

    S8 共享模式（``shared`` 给出账本路径／``budget_ledger.BudgetLedger`` 对象，或缺省时环境变量
    ``SGEVAL_BUDGET_LEDGER`` 非空；``shared=None`` 显式关闭）：

    * infra 重试名额改由共享账本 ``claim_retry`` 原子领取（跨原侧／新侧、跨席位，本地 ``--infra-retry-budget`` 不再
      适用）；中断分 ``infra`` 与 ``expired`` 分别计数，两者都占每身份 ``V8_MAX_ATTEMPTS`` 名额；
    * ``attempt_start`` 额外记 ``route``、``slurm_job_id``、``slurm_end_time``（到期证据的来源）与重试的 ``interrupt``；
    * ``classify_interrupt`` 按证据把悬空尝试分为 ``expired``（Slurm 到期）或 ``infra``（其他）。

    非共享模式下以上一概不发生，账本行与 BASE 逐字节相同。
    """

    def __init__(self, path: Path | str, *, seat: str, policy: str, shared: Any = _UNSET, route: Any = _UNSET,
                 expired_jobs: Any = _UNSET, caps: dict | None = None):
        self.path = Path(path)
        self.seat = str(seat)
        self.policy = str(policy)
        self.shared = _open_shared(shared, caps)
        self.route = f"{self.policy}/new" if route is _UNSET or route is None else str(route)
        self._expired_jobs = expired_jobs
        self.last_t: dict[str, float] = {}  # attempt_id -> 该尝试最后一行的时间（到期判定用）
        self.reset_budget: int | None = None
        self.infra_retry_budget: int | None = None
        self._reset_metering = False  # start(reset_budget=None)：只计量、不拦（未 start 时仍按 BASE 视额度为 0）
        self.reset_claims = 0
        self.budget_hist_max: int | None = None
        self.starts: dict[str, list[dict]] = {}  # key -> attempt_start 行（含作废）
        self.ended: dict[str, dict] = {}  # attempt_id -> attempt_end 行
        self.accepted: dict[str, str] = {}  # key -> accepted_attempt_id
        self._lock = threading.Lock()
        for row in read_results(self.path):
            self._apply(row)

    # 状态推导
    def _apply(self, row: dict) -> None:
        kind = row.get("kind")
        aid = row.get("attempt_id")
        if aid and isinstance(row.get("t"), (int, float)):
            self.last_t[aid] = max(self.last_t.get(aid, 0.0), float(row["t"]))
        if kind == "reset_claim":
            self.reset_claims += 1
        elif kind == "budget":
            if row.get("reset_budget") is not None:  # 第三阶段只计量模式写 null，不参与额度提升比较
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

    # 进程启动：记预算；命令行额度大于账本历史最大值即记一次提升。reset_budget=None：只计量（第三阶段缺省）
    def start(self, reset_budget: int | None, infra_retry_budget: int, *, reason: str = "cli_reset_budget") -> None:
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

    # 计数
    @property
    def reset_enforced(self) -> bool:
        """start 时给了 reset 额度才按额度硬拦；start(None)（第三阶段不给 --reset-budget）只计量。"""
        return not self._reset_metering

    def reset_left(self) -> float:
        """剩余 reset 额度；只计量模式恒为 inf（不拦）。"""
        if not self.reset_enforced:
            return float("inf")
        return int(self.reset_budget or 0) - self.reset_claims

    def attempts_total(self, key: str) -> int:
        """含作废尝试；新尝试编号 = 此数 + 1（录像目录名唯一）。"""
        return len(self.starts.get(key, []))

    def attempts_used(self, key: str) -> int:
        """不含作废尝试：对每身份 2 次上限。"""
        return sum(not self._void(r["attempt_id"]) for r in self.starts.get(key, []))

    def infra_retries_used(self) -> int:
        return sum(bool(r.get("retry")) and not self._void(r["attempt_id"])
                   for rows in self.starts.values() for r in rows)

    def infra_retries_left(self) -> int:
        if self.shared is not None:  # S8：共享 infra 额度余量（只读快照，真正领取走 allow_retry 的原子操作）
            return self.shared.shared_infra_cap - self.shared.state().retries_of("infra")
        return int(self.infra_retry_budget or 0) - self.infra_retries_used()

    # ── S8 共享模式 ──────────────────────────────────────────────────────
    def retry_interrupt(self, key: str) -> str:
        """最后一次未作废尝试的中断分类（attempt_end.interrupt；未记的一律 infra）。"""
        live = [r for r in self.starts.get(key, []) if not self._void(r["attempt_id"])]
        end = self.ended.get(live[-1]["attempt_id"]) if live else None
        it = (end or {}).get("interrupt")
        return it if it in INTERRUPTS else "infra"

    def interrupt_counts(self, key: str | None = None) -> dict[str, int]:
        """未作废、非最终结局（infra 错误）的尝试按中断分类计数；key=None 时统计整本账本。"""
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

    def allow_retry(self, key: str, *, token: str | None = None) -> bool:
        """是否可以对 key 再开一次重试。非共享模式即 BASE 的本地额度判断；共享模式向共享账本原子领一个名额
        （infra 或 expired，按上一尝试的中断分类），并发争抢最后一个名额只有一方成功。``token`` 与随后该尝试的
        reserve／attempt_start 同一个：崩溃后续跑以同一 token 再领时幂等返回 True、不重复扣额。"""
        if self.shared is None:
            return self.infra_retries_left() > 0
        kw = {"token": token} if token else {}
        return bool(self.shared.claim_retry(route=self.route, key=key, interrupt=self.retry_interrupt(key),
                                            seat=self.seat, policy=self.policy, **kw))

    def expired_jobs(self) -> set[str]:
        """Slurm 到期作业号集合：构造参数 ``expired_jobs`` 优先，否则读环境变量 ``SGEVAL_EXPIRED_JOBS`` 指向的文件。"""
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
        # sacct 可能带 <JobID>.batch／<JobID>_<array> 等后缀：取主作业号
        return {t.split(".")[0] for t in toks if t.strip()}

    def classify_interrupt(self, start: dict, *, now: float | None = None) -> tuple[str, str]:
        """按证据给悬空尝试分类：返回 (interrupt, evidence)。

        * 尝试记下的 ``slurm_job_id`` 在到期作业号集合里 → ``expired``（``sacct_timeout``）；
        * 尝试记下的 ``slurm_end_time`` 已过、且该尝试最后一行距它不超过 ``EXPIRE_MARGIN_S`` → ``expired``；
        * 其他一律 ``infra``（``no_expiry_evidence``）。"""
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
        """最后一次未作废尝试的 attempt_end 是否已是最终结局（非 infra 错误等）：是则该身份不再重跑。"""
        live = [r for r in self.starts.get(key, []) if not self._void(r["attempt_id"])]
        if not live:
            return False
        end = self.ended.get(live[-1]["attempt_id"])
        return bool(end) and self.is_final(end)

    def dangling(self) -> list[dict]:
        """有 attempt_start、无 attempt_end 的尝试（进程被杀等）。"""
        return [r for rows in self.starts.values() for r in rows if r["attempt_id"] not in self.ended]

    def final_without_accept(self) -> list[dict]:
        """崩溃窗口：最后一次未作废尝试的 attempt_end 已是最终结局、该身份却没有 accept（attempt_end 写完、accept
        写之前进程死掉）。返回这些 attempt_end 行。"""
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
        """为崩溃窗口里的身份补写 accept（指向最后一次、即第一次最终结局的尝试），带 ``recovered=true``。"""
        return self.append({"kind": "accept", "key": end["key"], "attempt_id": end["attempt_id"],
                            "attempt_no": end.get("attempt_no"), "accepted_attempt_id": end["attempt_id"],
                            "status": end.get("status"), "recovered": True})

    # 写入
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
        if self.shared is not None:  # S8：记下到期证据的来源（作业号、Slurm 结束时刻）与路线
            extra = {"route": self.route, "host": socket.gethostname(),
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID") or None,
                     "slurm_end_time": _int_or_none(os.environ.get("SLURM_JOB_END_TIME")), **extra}
        self.append({"kind": "attempt_start", "key": key, "attempt_id": attempt_id, "attempt_no": attempt_no,
                     "retry": bool(retry), **extra})

    @staticmethod
    def is_final(record: dict) -> bool:
        """该尝试是否构成身份的最终结局（主会话 2026-10-02 口径裁定）：success／fail／timeout，以及非基础设施错误
        （status=error、infra=false、非 budget_exhausted、非 run_blocked——如 reset 失败、环境自报 error、客户端
        TypeError 等）。infra=true 的错误不是结局（可在额度内重试；额度或 2 次用满则不 accept，汇总计 missing）。"""
        status = record.get("status")
        if status in TERMINAL_STATUSES:
            return True
        return (status == "error" and not record.get("infra") and not record.get("budget_exhausted")
                and not record.get("run_blocked"))

    def is_late(self, record: dict) -> bool:
        return self.is_final(record) and record["key"] in self.accepted

    def attempt_end(self, record: dict) -> bool:
        """写 attempt_end；最终结局且该身份尚无 accept 时再写 accept（status 可为 error）。返回 late。"""
        key, status = record["key"], record.get("status")
        late = self.is_late(record)
        base = {"key": key, "attempt_id": record["attempt_id"], "attempt_no": record["attempt_no"]}
        self.append({"kind": "attempt_end", **base, "status": status, "infra": bool(record.get("infra")),
                     "cap_hit": bool(record.get("cap_hit")), "exec_steps": record.get("exec_steps"),
                     "budget_exhausted": bool(record.get("budget_exhausted")), "late": late,
                     **({"recovered": True} if record.get("recovered") else {}),
                     **({"interrupt": record["interrupt"], "interrupt_evidence": record.get("interrupt_evidence")}
                        if record.get("interrupt") else {})})
        if self.is_final(record) and not late:
            self.append({"kind": "accept", **base, "accepted_attempt_id": record["attempt_id"], "status": status})
        return late


# ── 常驻客户端 ──────────────────────────────────────────────────────────────


def policy_seed_of(args) -> int | None:
    """``--policy-seed``（未给为 None）。"""
    v = getattr(args, "policy_seed", None)
    return None if v is None else int(v)


def policy_route(args) -> str:
    """共享账本路线名（接口冻结说明第三节第 2 条）：``<policy_label>/seed<policy_seed>/new``，GroundSG 为
    ``groundsg/<variant>/seed<policy_seed>/new``；未给 ``--policy-seed``（进程内单测）时沿用旧式 ``<policy>/new``。"""
    seed = policy_seed_of(args)
    head = f"groundsg/{getattr(args, 'groundsg_variant', None)}" if args.policy == "groundsg" else args.policy
    return f"{head}/new" if seed is None else f"{head}/seed{seed}/new"


def policy_variant_of(args) -> str | None:
    """结果行的 ``policy_variant``：groundsg 取 ``--groundsg-variant``，其余策略为 null。"""
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


def check_run_args(args, *, need_identities: bool = False) -> str | None:
    """``run`` 的参数组合核对；不符返回说明（cmd_run 打印 RUN_BLOCKED reason=args 并以 3 退出）。第三阶段起
    ``--reset-budget`` 可选（不给只计量）；``--policy-seed``／预算参数／变体配对的 CLI 拦截见 ``entry_blockers``。"""
    bad = []
    if getattr(args, "dataset", None) not in DATASETS:
        bad.append(f"--dataset 必须是 {'／'.join(DATASETS)} 之一（现为 {getattr(args, 'dataset', None)!r}）")
    ms = getattr(args, "max_steps", None)
    if not _is_int(ms) or ms <= 0:
        bad.append(f"--max-steps 必须是正整数（现为 {ms!r}）")
    miss = [n for n, v in (("--ledger", getattr(args, "ledger", None)),
                           ("--infra-retry-budget", getattr(args, "infra_retry_budget", None)),
                           *((("--identities", getattr(args, "identities", None)),) if need_identities else ()))
            if v is None]
    if miss:
        bad.append(f"必须给 {' '.join(miss)}")
    bad += variant_problems(args)
    return "; ".join(bad) or None


def entry_blockers(args) -> tuple[str, str] | None:
    """CLI ``run`` 入口的具名拦截（接口冻结说明 2.2／2.3／2.4）：依次核 ``--policy-seed``（必填、非负整数）、五个预算
    参数（必填，不回落常量默认）、GroundSG 变体与 adapter 配对（含目录存在）。返回 (reason, detail) 或 None。"""
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
    return None


def _arg_or(args, name: str, default: Any) -> Any:
    v = getattr(args, name, None)
    return default if v is None else v


def budget_caps(args) -> dict | None:
    """``--trajectory-cap --shared-infra-cap --expired-cap --planned-first-tries`` 全部给出时的构造参数；否则 None
    （进程内单测与旧调用沿用 budget_ledger 缺省）。"""
    vals = {dest: getattr(args, dest, None) for dest, _ in BUDGET_ARGS[1:]}
    if any(v is None for v in vals.values()):
        return None
    return {k: int(v) for k, v in vals.items()}


def shard_id_of(route: str, identities: Any, seat: Any, dataset: Any = None) -> str:
    """分片排他 lease 的 shard_id，非 [A-Za-z0-9._-] 一律换成 ``_``。

    * 有身份清单（生产 CLI ``--identities`` 必填）：``<route>--<dataset>--<清单文件名 stem>-h<内容 sha256 前 10 位>``
      （1007 FIX-4）。路线不含数据集，ood 与 hard-verify 的清单都叫 ``shard-NN.json``，只拼「路线 + 文件名」会让两个
      数据集的不同分片撞同一把锁（GL OOD MemER 第一片被本机 hard-verify 对拍挡下 ``RUN_BLOCKED reason=lease_held``）；
      并入数据集名与清单内容短哈希后，不同数据集或不同目录下的同名不同内容分片互不相干，同一份清单（同内容）重复启动
      仍得到同一 id、被 lease 拦下。清单读不到时哈希段记 ``-hNA``（仍含数据集名）。
    * 无身份清单（只出现在进程内调用）：沿用 ``<route>--seat-<席位>``，不并入数据集。"""
    if not identities:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{route}--seat-{seat}")
    p = Path(str(identities))
    try:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:10]
    except OSError:
        digest = "NA"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{route}--{dataset or 'NA'}--{p.stem}-h{digest}")


class SeatRunner:
    """一个席位上一个策略的常驻客户端。``policy_mod`` / ``recorder_factory`` / ``builder_factory`` 可注入（单测）。

    * ``builder_factory`` 按形参个数调用：3 个 → ``(task, dataset, max_steps)``；2 个 → ``(task, max_steps)``；
      1 个 → ``(task)``。不注入时建 ``BenchmarkEnvBuilder(env_id=task, dataset=--dataset, action_space="joint_angle",
      max_steps=--max-steps)``，按 ``(task, dataset)`` 缓存。
    * ``policy_context``：整席只建一次的进程内对象（不进任何 JSON），经 ``conn_info["policy_context"]`` 交给策略。
      策略模块定义了 ``make_policy_context(seat_info: dict)`` 就在第一局前调用一次、取其返回值，否则为空 dict（策略
      可自行往里缓存）；定义了 ``close_policy_context(ctx)`` 就在席位收尾（``close()``）时调用一次。
    """

    def __init__(self, args, *, policy_mod=None, recorder_factory=None, builder_factory=None, proc_info=None):
        self.args = args
        bad = check_run_args(args)
        if bad:
            raise ValueError(bad)
        self.dataset = args.dataset
        self.max_steps = int(args.max_steps)
        self.strict_cap = bool(getattr(args, "strict_cap", False))
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.results_path = self.out / "results.jsonl"
        self.progress_path = self.out / "progress.json"
        self.policy_mod = policy_mod
        self.recorder_factory = recorder_factory
        self.builder_factory = builder_factory
        self.builders: dict[tuple[str, str], Any] = {}
        self.proc_info = proc_info or {}
        self.episodes_done = 0
        self._lock = threading.Lock()
        self._current: dict | None = None
        self._built_once = False
        rec_root = getattr(args, "rec_root", None)
        self.rec_root = Path(rec_root) if rec_root else self.out / "rec"
        trace_root = getattr(args, "trace_root", None)
        self.trace_root = Path(trace_root) if trace_root else None
        self._policy_context: Any = None
        self._policy_context_ready = False
        # 第三阶段：模型种子、实际步数上限、执行期限（缺省见常量）；_hard_exit 只供单测替换（生产为 os._exit）
        self.policy_seed = policy_seed_of(args)
        self.effective_cap = self.max_steps if self.strict_cap else None
        self.context_deadline_s = float(_arg_or(args, "context_deadline_s", DEFAULT_CONTEXT_DEADLINE_S))
        self.first_infer_deadline_s = float(_arg_or(args, "first_infer_deadline_s", DEFAULT_FIRST_INFER_DEADLINE_S))
        self.media_deadline_s = float(_arg_or(args, "media_deadline_s", DEFAULT_MEDIA_DEADLINE_S))
        self._hard_exit: Callable[[int], Any] = os._exit
        self._phase = "context_load"
        self._lease_cm = None
        # S8：--budget-ledger 显式打开共享预算（第三阶段带 config 四参数）；不给时由 AttemptLedger 按环境变量
        # SGEVAL_BUDGET_LEDGER 门控
        shared_kw = {"shared": args.budget_ledger} if getattr(args, "budget_ledger", None) else {}
        self.ledger = AttemptLedger(args.ledger, seat=args.seat, policy=args.policy, route=policy_route(args),
                                    caps=budget_caps(args), **shared_kw)
        self.ledger.start(getattr(args, "reset_budget", None), args.infra_retry_budget,
                          reason=getattr(args, "budget_raise_reason", None) or "cli_reset_budget")

    # 进度：原子替换写 progress.json。只在阶段、身份或步数变化时调用（不做定时心跳），run_seat.sh 据此判无进展
    def progress(self, step: int = 0, *, phase: str | None = None, **extra) -> None:
        if phase is not None:
            if phase not in PHASES:
                raise ValueError(f"phase={phase!r} 不是 {PHASES} 之一")
            self._phase = phase
        cur = self._current or {}
        doc = {"pid": os.getpid(), "host": socket.gethostname(), "seat": self.args.seat, "policy": self.args.policy,
               "cond": self.args.cond, "dataset": self.dataset, "key": cur.get("key"), "identity": cur.get("key"),
               "attempt_no": cur.get("attempt_no"), "phase": self._phase, "step": step, "t": time.time(),
               "episodes_done": self.episodes_done, "effective_cap": self.effective_cap,
               "policy_seed": self.policy_seed, **extra}
        tmp = self.progress_path.with_suffix(".json.tmp")
        tmp.write_text(dumps(doc), encoding="utf-8")
        os.replace(tmp, self.progress_path)

    def builder_for(self, task: str):
        """按 (task, dataset) 缓存 builder；步数上限取 ``--max-steps``（不再按档查表）。"""
        ck = (task, self.dataset)
        if ck not in self.builders:
            if self.builder_factory is not None:
                try:
                    n_params = len(inspect.signature(self.builder_factory).parameters)
                except (TypeError, ValueError):
                    n_params = 1
                if n_params >= 3:
                    self.builders[ck] = self.builder_factory(task, self.dataset, self.max_steps)
                elif n_params == 2:
                    self.builders[ck] = self.builder_factory(task, self.max_steps)
                else:
                    self.builders[ck] = self.builder_factory(task)
            else:
                from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

                self.builders[ck] = BenchmarkEnvBuilder(env_id=task, dataset=self.dataset, action_space="joint_angle",
                                                        max_steps=self.max_steps)
        return self.builders[ck]

    def policy_kwargs(self, max_steps: int) -> dict:
        """策略 ``run_episode`` 签名里有 ``max_steps``／``reset_retries`` 关键字（smvla）就显式传
        ``max_steps=--max-steps``、``reset_retries=0``；没有（perceptual-framesamp-modul 等只读 conn_info）就不传。"""
        try:
            params = inspect.signature(self.policy_mod.run_episode).parameters
        except (TypeError, ValueError):
            return {}
        kw: dict[str, Any] = {}
        if "max_steps" in params:
            kw["max_steps"] = int(max_steps)
        if "reset_retries" in params:
            kw["reset_retries"] = 0
        return kw

    def seat_info(self) -> dict:
        """交给 ``make_policy_context`` 的席位信息（只读、只在进程内传，不落盘）；第三阶段加 ``policy_seed``、
        ``memer_adapter_path``、``budget_ledger``、``effective_cap``（接口冻结说明 2.5）。"""
        a = self.args
        return {"policy": a.policy, "seat": a.seat, "host": a.host, "port": a.port, "dataset": self.dataset,
                "max_steps": self.max_steps, "strict_cap": self.strict_cap,
                "groundsg_variant": getattr(a, "groundsg_variant", None),
                "qwenvl_groundSG_adapter_path": getattr(a, "qwenvl_groundsg_adapter", None),
                "memer_adapter_path": getattr(a, "memer_adapter", None),
                "policy_seed": self.policy_seed,
                "budget_ledger": getattr(a, "budget_ledger", None),
                "effective_cap": self.effective_cap,
                "trace_root": str(self.trace_root) if self.trace_root else None, "out": str(self.out)}

    def server_metadata_path(self) -> Path:
        """服务元数据：``--server-metadata`` 优先，否则 ``<out>/server-metadata-<port>.json``。"""
        p = getattr(self.args, "server_metadata", None)
        return Path(p) if p else self.out / f"server-metadata-{self.args.port}.json"

    def server_seed(self) -> int | None:
        """结果行 ``server_seed``：从服务启动时写的元数据反查 ``policy_seed``；读不到为 null（不伪造）。"""
        try:
            v = json.loads(self.server_metadata_path().read_text(encoding="utf-8")).get("policy_seed")
        except (OSError, ValueError, AttributeError):
            return None
        return v if _is_int(v) else None

    def policy_context(self) -> Any:
        """整席只建一次的策略上下文（进程内对象）。"""
        if not self._policy_context_ready:
            make = getattr(self.policy_mod, "make_policy_context", None)
            self._policy_context = make(self.seat_info()) if callable(make) else {}
            self._policy_context_ready = True
        return self._policy_context

    def close(self) -> None:
        """席位收尾：策略定义了 ``close_policy_context`` 就调一次（上下文从未建过则不调）；释放分片 lease。"""
        self._release_lease()
        if self._policy_context_ready:
            fn = getattr(self.policy_mod, "close_policy_context", None)
            if callable(fn):
                fn(self._policy_context)
            self._policy_context_ready = False
            self._policy_context = None

    def base_record(self, ident: dict, *, attempt: int) -> dict:
        # 结果行保留 v8=True（下游 eval_report／搬运脚本按它识别本格式）与 canary 字段（恒为 False）；
        # max_steps 与 effective_max_steps 下游站点仍在读，值都取 --max-steps
        ident_full = {"tier": ident["tier"], "seed": int(ident["seed"]), "candidate": ident["candidate"],
                      "spec_sha256": ident["spec_sha256"], "builder_episode": int(ident["builder_episode"]),
                      "source_episode": ident.get("source_episode")}
        return {"v8": True, "key": key_of(ident), "task": ident["task"], "tier": ident["tier"],
                "seed": int(ident["seed"]), "candidate": ident["candidate"], "spec_sha256": ident["spec_sha256"],
                "source_episode": ident.get("source_episode"), "identity": ident_full,
                "builder_episode": int(ident["builder_episode"]), "dataset": self.dataset,
                "policy": self.args.policy, "policy_variant": policy_variant_of(self.args),
                "policy_seed": self.policy_seed, "effective_cap": self.effective_cap,
                "server_seed": self.server_seed(), "budget_token": None, "error_kind": None,
                "strict_cap": self.strict_cap, "cond": self.args.cond, "seat": self.args.seat,
                "host": socket.gethostname(), "attempt": attempt, "attempt_no": attempt, "canary": False,
                "gpu_name": self.proc_info.get("gpu_name"), "gpu_uuid": self.proc_info.get("gpu_uuid"),
                "git_commit": self.proc_info.get("git_commit"), "git_dirty": self.proc_info.get("git_dirty"),
                "max_steps": self.max_steps, "effective_max_steps": self.max_steps}

    def _check(self, ident: dict) -> tuple[str | None, dict | None, Any]:
        """身份核对：结构 → builder 解析 → 逐键比对。返回 (不符说明, 解析结果, builder)。"""
        bad = validate_v8_identity(ident, self.dataset)
        if bad is not None:
            return bad, None, None
        builder = self.builder_for(ident["task"])
        try:
            resolved = builder.resolve_identity(int(ident["builder_episode"]))
        except Exception as e:  # noqa: BLE001 局号越界等（如把别的数据集的分片喂进来）
            return f"resolve_identity: {type(e).__name__}: {e}"[:800], None, builder
        return check_identity(resolved, ident, dataset=self.dataset), resolved, builder

    def run_one(self, ident: dict, *, attempt: int = 1, retry: bool = False) -> dict:
        key = key_of(ident)
        tag = f"{key}.a{attempt}"
        rec_dir = self.rec_root / tag
        eff = self.max_steps
        record = self.base_record(ident, attempt=attempt)
        record["rec_dir"] = str(rec_dir)
        attempt_id = uuid.uuid4().hex
        record.update(attempt_id=attempt_id, exec_steps=0, client_steps=None, chunks=None, hard_bound=None,
                      cap_hit=False, demo_frames=None, reset_calls=0, late=False)
        self._current = {"key": key, "t0": time.time(), "attempt_no": attempt}
        bad, resolved, builder = self._check(ident)
        if bad:
            record.update(status="error", task_success=False, steps=0, error=f"IDENTITY_MISMATCH {bad}",
                          run_blocked=True, infra=False, resolved_identity=resolved)
            append_result(self.results_path, record)
            print(f"RUN_BLOCKED reason=identity key={key} dataset={self.dataset} detail={bad}", flush=True)
            raise SystemExit(EXIT_BLOCKED)
        if self.ledger.reset_left() <= 0:  # 只在给了 --reset-budget 时可能：开局前就没有额度，不开尝试、直接停
            self._budget_stop(key)
        # 第三阶段：同一 token 贯穿 claim_retry（run_identities_v8）／reserve／attempt_start，崩溃续跑幂等
        token = self.attempt_token(key, attempt) if self.ledger.shared is not None else None
        record["budget_token"] = token
        rid = self._reserve(key, attempt_id, attempt, token=token, retry=retry)  # 共享模式先预约轨迹（不足即停）
        shared_extra = {} if rid is None else {"budget_rid": rid, "token": token,
                                                **({"interrupt": self.ledger.retry_interrupt(key)} if retry else {})}
        self.ledger.attempt_start(key=key, attempt_id=attempt_id, attempt_no=attempt, retry=retry,
                                  task=ident["task"], tier=ident["tier"], dataset=self.dataset,
                                  builder_episode=int(ident["builder_episode"]), **shared_extra)
        state: dict[str, Any] = {"finished": False, "session": None, "rid": rid}
        # 上下文加载（整席一次）：绝对期限，超期以 infra_reason=deadline_context_load 结束本局
        if not self._policy_context_ready:
            self.progress(0, phase="context_load")
        policy_context = self._run_phase(state, "context_load", self.context_deadline_s, record, self.policy_context)
        if state.get("deadline_record") is not None:  # 期限处理已写结果（生产里进程已退出，只有单测走到这里）
            return state["deadline_record"]

        def claim_reset(what, _k=key, _a=attempt_id, _n=attempt):
            self.ledger.claim_reset(key=_k, attempt_id=_a, attempt_no=_n, what=what)
        meta = dict(record, resolved_identity=resolved, env=self.proc_info.get("env"),
                    never_degrade=bool(self.args.never_degrade), baseline=bool(self.args.baseline))
        try:
            recorder = self.recorder_factory(rec_dir, meta) if self.recorder_factory else NullRecorder()
        except Exception as e:  # noqa: BLE001 录制器建不起来（如磁盘满）= 基础设施故障
            record.update(status="error", task_success=False, steps=0, infra=True, infra_reason="recorder",
                          error=f"RecorderError: init: {type(e).__name__}: {e}"[:800])
            self._finish(record)
            self._settle(rid, record)
            self._print_done(record)
            return record
        first_timer: list = []

        def on_first_step():  # 首推完成（拿到第一个动作）：结束 first_infer 期限，进入 episode 阶段
            self._end_phase(state, "first_infer", first_timer)
            self.progress(0, phase="episode")
        session = EnvSession(ident["task"], int(ident["builder_episode"]), max_steps=eff, recorder=recorder,
                             builder=builder, progress_cb=lambda s: self.progress(s),
                             step_cap=eff if self.strict_cap else None, claim_reset=claim_reset,
                             dataset=self.dataset, first_step_cb=on_first_step, **self._budget_kw(rid))
        state["session"] = session
        # conn_info：除 policy_context（进程内对象）外都是可序列化的标量；trace_dir 为本局轨迹目录（不预先建，
        # 由写轨迹的一方建），未给 --trace-root 时为 null
        trace_dir = self.trace_root / tag if self.trace_root is not None else None
        conn_info = {"host": self.args.host, "port": self.args.port, "max_steps": eff, "policy": self.args.policy,
                     "seat": self.args.seat, "dataset": self.dataset, "strict_cap": self.strict_cap,
                     "groundsg_variant": getattr(self.args, "groundsg_variant", None),
                     "qwenvl_groundSG_adapter_path": getattr(self.args, "qwenvl_groundsg_adapter", None),
                     "memer_adapter_path": getattr(self.args, "memer_adapter", None),
                     "policy_seed": self.policy_seed, "effective_cap": self.effective_cap,
                     "trace_root": str(self.trace_root) if self.trace_root is not None else None,
                     "trace_dir": str(trace_dir) if trace_dir is not None else None,
                     "episode_tag": tag, "rec_dir": str(rec_dir), "policy_context": policy_context}
        policy_kw = self.policy_kwargs(eff)
        limit = self.args.episode_wall_s + (self.args.first_extra_s if self.episodes_done == 0 else 0)
        # first_extra_s 只给「server 刚（重）起后的第一局」：由 run_seat.sh 在 server 新起时传 600，客户端单独重起时传 0
        timer = None
        if limit > 0:
            timer = threading.Timer(limit, self._on_wall_timeout, args=(record, limit, state))
            timer.daemon = True
            timer.start()
        t0 = time.perf_counter()
        first_build = not self._built_once
        self.progress(0, phase="first_infer")
        first_timer.append(self._arm(state, "first_infer", self.first_infer_deadline_s, record))
        try:
            try:
                session.build()
                self._built_once = True
            except Exception as e:  # noqa: BLE001 构建失败（如 Vulkan）按基础设施故障记；额度耗尽不算基础设施
                res = {"status": "error", "task_success": False, "steps": 0, "error": f"{type(e).__name__}: {e}"[:800],
                       "infra": not isinstance(e, ResetBudgetExhausted),
                       "infra_reason": None if isinstance(e, ResetBudgetExhausted) else "env_build"}
            else:
                res = self.policy_mod.run_episode(session, ident, conn_info, recorder, **policy_kw)
        finally:
            self._end_phase(state, "first_infer", first_timer)
            with self._lock:
                state["finished"] = True
            if timer is not None:
                timer.cancel()
        if state.get("deadline_record") is not None:
            return state["deadline_record"]
        try:
            session.close()
        except Exception as e:  # noqa: BLE001
            res.setdefault("close_error", repr(e))
        wall = time.perf_counter() - t0
        self.progress(session.steps, phase="media_finalize")

        def close_recorder():
            return recorder.close({"status": res.get("status"), "steps": res.get("steps"),
                                   "exec_steps": session.steps, "cap_hit": session.cap_hit})
        try:
            rsum = self._run_phase(state, "media_finalize", self.media_deadline_s, dict(record, **res), close_recorder)
        except Exception as e:  # noqa: BLE001 收尾失败（如磁盘满）= 基础设施故障
            rsum = {"RECORDER_VERIFY": "ERROR", "error": f"{type(e).__name__}: {e}"[:800]}
            res.update(infra=True, infra_reason="recorder_close")
        if state.get("deadline_record") is not None:
            return state["deadline_record"]
        if "RecorderError" in str(res.get("error") or "") or "RecorderError" in str(res.get("env_exception") or ""):
            res.update(infra=True, infra_reason="recorder")
        timing = {"episode_wall_s": wall, "first_build_in_process": first_build, "env": dict(session.timing),
                  "policy": res.pop("timing", None), "recorder": rsum, "episode_index_in_process": self.episodes_done}
        if self.episodes_done == 0:
            timing["process_init"] = self.proc_info.get("init_timing")
        record.update(res)
        self._classify(record, session, res)
        record["task_success"] = record.get("status") == "success"
        record["timing"] = timing
        record["recorder_verify"] = (rsum or {}).get("RECORDER_VERIFY")
        self._finish(record)
        self._settle(rid, record)
        self.episodes_done += 1
        self.progress(record.get("steps", 0), phase="done")
        self._print_done(record)
        if record.get("budget_exhausted"):
            self._budget_stop(key)
        return record

    # ── 第三阶段：幂等 token 与执行期限 ─────────────────────────────────
    def attempt_token(self, key: str, attempt_no: int) -> str:
        """幂等 token ``<route>|<key>|a<attempt_no>``（与 budget_ledger.make_token 同式）。"""
        return f"{self.ledger.route}|{key}|a{int(attempt_no)}"

    def _arm(self, state: dict, phase: str, limit: float, record: dict):
        """给某阶段挂绝对期限（limit<=0 不挂）；返回计时器。"""
        if not limit or limit <= 0:
            return None
        t = threading.Timer(limit, self._on_deadline, args=(phase, limit, record, state))
        t.daemon = True
        t.start()
        return t

    def _end_phase(self, state: dict, phase: str, timers: list) -> None:
        with self._lock:
            state[f"{phase}_done"] = True
        for t in timers:
            if t is not None:
                t.cancel()
        timers.clear()

    def _run_phase(self, state: dict, phase: str, limit: float, record: dict, fn: Callable[[], Any]) -> Any:
        """在绝对期限下执行 fn（主线程照常阻塞；期限到由计时器线程写结果并退出进程）。"""
        timers = [self._arm(state, phase, limit, record)]
        try:
            return fn()
        finally:
            self._end_phase(state, phase, timers)

    def _on_deadline(self, phase: str, limit: float, record: dict, state: dict) -> None:
        """期限到：本局记 ``status=error infra=true infra_reason=deadline_<phase>``（attempt_end 照写、共享预算收尾），
        打印 ``DEADLINE_EXCEEDED`` 后以 75 退出（与单局墙钟同一退出码，run_seat.sh 重起客户端）。"""
        with self._lock:
            if state.get(f"{phase}_done") or state.get("deadline_record") is not None:
                return
            sess = state.get("session")
            rec = dict(record, status="error", task_success=False, steps=None, infra=True,
                       infra_reason=f"deadline_{phase}", error=f"DEADLINE phase={phase} limit_s={limit:.0f}",
                       exec_steps=getattr(sess, "steps", 0), reset_calls=getattr(sess, "reset_calls", 0),
                       cap_hit=getattr(sess, "cap_hit", False),
                       demo_frames=(getattr(sess, "timing", None) or {}).get("demo_frames"))
            state["deadline_record"] = rec
            self._finish(rec)
            self._settle(state.get("rid"), rec)
        print(f"DEADLINE_EXCEEDED phase={phase} policy={self.args.policy} key={key_of(record)} limit_s={limit:.0f}",
              flush=True)
        sys.stdout.flush()
        self._hard_exit(EXIT_WALL)

    def _classify(self, record: dict, session: EnvSession, res: dict) -> None:
        """执行段步数以环境侧计数为准；客户端自报另记。额度耗尽与步数到顶（仅 --strict-cap）的分类覆盖客户端的记法。"""
        record.update(exec_steps=session.steps, client_steps=res.get("steps"), cap_hit=session.cap_hit,
                      chunks=res.get("decisions") if "hard_bound" in res else None,
                      hard_bound=res.get("hard_bound"), demo_frames=session.timing.get("demo_frames"),
                      reset_calls=session.reset_calls, effective_max_steps=self.max_steps)
        if session.budget_exhausted or session.cap_hit:
            if session.budget_exhausted:
                record.update(status="error", infra=False, infra_reason=None, budget_exhausted=True,
                              client_error=res.get("error"),
                              error=f"RESET_BUDGET_EXHAUSTED claims={self.ledger.reset_claims} "
                                    f"budget={self.ledger.reset_budget}")
            elif session.cap_hit:
                record.update(status="timeout", infra=False, infra_reason=None, client_status=res.get("status"),
                              client_error=res.get("error"),
                              error=f"STEP_CAP exec_steps={session.steps} cap={self.max_steps} 未成功，按 timeout 计")

    # ── S8 共享预算（非共享模式下全部为空操作） ─────────────────────────
    def _reserve(self, key: str, attempt_id: str, attempt: int, *, token: str | None = None,
                 retry: bool = False) -> str | None:
        """向共享账本预约一条轨迹（reset 计量 NEW_SIDE_RESETS_PER_ATTEMPT；首试 ``kind_of_try=first``、重试
        ``recovery``；同一 token 幂等）；不足打印 RUN_BLOCKED reason=budget 并以退出码 5 停止（attempt_start 尚未写，
        不进入 attempt）。非共享模式返回 None。"""
        shared = self.ledger.shared
        if shared is None:
            return None
        kw = {"token": token, "kind_of_try": "recovery" if retry else "first"} if token else {}
        try:
            return shared.reserve(resets=NEW_SIDE_RESETS_PER_ATTEMPT, route=self.ledger.route, key=key,
                                  attempt_id=attempt_id, attempt_no=attempt, seat=self.args.seat,
                                  policy=self.args.policy, astra=False, **kw)
        except Exception as e:  # noqa: BLE001 budget_ledger.BudgetExhausted（可能来自另一份模块副本，按属性识别）
            if not getattr(e, "budget_exhausted", False):
                raise
            print(f"RUN_BLOCKED reason=budget policy={self.args.policy} seat={self.args.seat} key={key} "
                  f"budget_reason={getattr(e, 'reason', None)} detail={e}", flush=True)
            raise SystemExit(EXIT_BUDGET) from e

    def _budget_kw(self, rid: str | None) -> dict:
        """EnvSession 的 budget_claim：每次实际 build／reset 记进共享账本该 rid 名下。"""
        if rid is None:
            return {}
        shared = self.ledger.shared
        return {"budget_claim": lambda what, _r=rid: shared.claim_reset(_r, what, route=self.ledger.route)}

    def _settle(self, rid: str | None, record: dict) -> None:
        """作废尝试（额度在 build／reset 前就被拒）退回轨迹名额，其余收尾。"""
        if rid is None:
            return
        if record.get("budget_exhausted"):
            self.ledger.shared.release(rid, status=record.get("status"))
        else:
            self.ledger.shared.commit(rid, status=record.get("status"), infra=bool(record.get("infra")))

    def _finish(self, record: dict) -> None:
        """写结果行，再写账本 attempt_end（终态且首个 → accept）。late 先算好写进结果行。"""
        record["late"] = self.ledger.is_late(record)
        append_result(self.results_path, record)
        self.ledger.attempt_end(record)

    def _print_done(self, record: dict) -> None:
        print(f"EPISODE_DONE policy={self.args.policy} dataset={self.dataset} key={record['key']} "
              f"status={record['status']} exec_steps={record.get('exec_steps')} cap_hit={record.get('cap_hit')} "
              f"infra={record.get('infra')} attempt_no={record.get('attempt_no')}", flush=True)

    def _budget_stop(self, key: str) -> None:
        print(f"RESET_BUDGET_EXHAUSTED policy={self.args.policy} seat={self.args.seat} key={key} "
              f"claims={self.ledger.reset_claims} budget={self.ledger.reset_budget}", flush=True)
        raise SystemExit(EXIT_BUDGET)

    def _on_wall_timeout(self, record: dict, limit: float, state: dict) -> None:
        with self._lock:
            if state["finished"]:
                return
            # 取 session 实际值；拿不到写 null（不写 0）
            sess = state.get("session")
            rec = dict(record, status="error", task_success=False, steps=None, infra=True, infra_reason="episode_wall",
                       error=f"INFRA_TIMEOUT episode_wall>{limit:.0f}s",
                       exec_steps=getattr(sess, "steps", None), reset_calls=getattr(sess, "reset_calls", None),
                       cap_hit=getattr(sess, "cap_hit", None),
                       demo_frames=(getattr(sess, "timing", None) or {}).get("demo_frames"))
            self._finish(rec)
            print(f"INFRA_TIMEOUT key={key_of(record)} limit_s={limit:.0f}", flush=True)
            sys.stdout.flush()
            os._exit(EXIT_WALL)

    # ── 身份清单来源 ────────────────────────────────────────────────────────
    def run_identities(self, rows: list[dict]) -> int:
        """跑一份身份清单；返回退出码（0 全部有权威终态；6 仍有身份无 accept）。共享模式先核账本（坏行、config），
        再在读取待跑身份之前取分片排他 lease（拿不到即 RUN_BLOCKED reason=lease_held、退出 3）；策略上下文改在第一局
        开局时于 context_load 期限下加载（整席仍只建一次）。"""
        self._open_shared_budget()
        self._check_route()
        return self.run_identities_v8(rows)

    def _check_route(self) -> None:
        """共享模式：尝试账本里已有 ``attempt_start`` 行的 ``route`` 必须全部等于当前路线（含 ``seed<n>``）；缺
        ``route`` 字段的旧行视为 ``legacy``、同样不符。不符打印 ``RUN_BLOCKED reason=route_mismatch`` 并以 3 退出（不写结果
        行、不改 accepted 计法）。账本没有任何 ``attempt_start`` 行（全新 stage）时不拦。非共享模式不记 route，不核。"""
        led = self.ledger
        if led.shared is None:
            return
        want = led.route
        found = []
        for starts in led.starts.values():
            for st in starts:
                r = st.get("route")
                tag = "legacy" if not r else str(r)
                if tag != want and tag not in found:
                    found.append(tag)
        if found:
            print(f"RUN_BLOCKED reason=route_mismatch ledger={led.path} found={','.join(found)} want={want} "
                  f"policy={self.args.policy} seat={self.args.seat}", flush=True)
            raise SystemExit(EXIT_BLOCKED)

    def _open_shared_budget(self) -> None:
        shared = self.ledger.shared
        if shared is None:
            return
        bl = load_sibling("budget_ledger")
        try:
            if hasattr(shared, "check"):
                shared.check()
            if self._lease_cm is None and hasattr(shared, "lease"):
                sid = shard_id_of(self.ledger.route, getattr(self.args, "identities", None), self.args.seat,
                                  getattr(self.args, "dataset", None))
                cm = shared.lease(sid)
                cm.__enter__()
                self._lease_cm = cm
                print(f"BUDGET_LEASE shard={sid} pid={os.getpid()}", flush=True)
        except Exception as e:  # noqa: BLE001 账本类可能来自另一份模块副本，按类名识别
            reason = {"LeaseHeld": "lease_held", "LedgerCorrupt": "ledger_corrupt",
                      "BudgetConfigMismatch": "budget_config"}.get(type(e).__name__)
            if reason is None and not isinstance(e, (bl.LeaseHeld, bl.LedgerCorrupt, bl.BudgetConfigMismatch)):
                raise
            print(f"RUN_BLOCKED reason={reason} policy={self.args.policy} seat={self.args.seat} detail={e}", flush=True)
            raise SystemExit(EXIT_BLOCKED) from e

    def _release_lease(self) -> None:
        cm, self._lease_cm = self._lease_cm, None
        if cm is not None:
            cm.__exit__(None, None, None)

    def recover_dangling(self) -> int:
        """账本里有 attempt_start 无 attempt_end 的尝试（进程被杀、写账本前崩溃）：结果行里找得到同 attempt_id 的
        就按结果行补 attempt_end（终态且首个 → accept），找不到就记为基础设施错误（未作废，占一次尝试）。"""
        by_id = {r.get("attempt_id"): r for r in read_results(self.results_path) if r.get("attempt_id")}
        n = 0
        for st in self.ledger.dangling():
            row = by_id.get(st["attempt_id"])
            if row is None:
                row = {"key": st["key"], "attempt_id": st["attempt_id"], "attempt_no": st["attempt_no"],
                       "status": "error", "infra": True, "exec_steps": None}
                if self.ledger.shared is not None:  # S8：按证据分 expired／infra，而不是一律 infra
                    interrupt, evidence = self.ledger.classify_interrupt(st)
                    row.update(interrupt=interrupt, interrupt_evidence=evidence)
            self.ledger.attempt_end(dict(row, recovered=True))
            print(f"LEDGER_RECOVER key={st['key']} attempt_id={st['attempt_id']} status={row.get('status')}", flush=True)
            n += 1
        return n

    def recover_crash_window(self) -> int:
        """「attempt_end 已写、accept 未写」时崩溃：最后一次未作废尝试已是最终结局而无 accept 的身份补写 accept。"""
        n = 0
        for end in self.ledger.final_without_accept():
            self.ledger.recover_accept(end)
            print(f"LEDGER_RECOVER_ACCEPT key={end['key']} attempt_id={end['attempt_id']} status={end.get('status')}",
                  flush=True)
            n += 1
        return n

    def run_identities_v8(self, rows: list[dict]) -> int:
        """已有 accept 的身份跳过；已用满 2 次尝试的跳过；最后一次未作废尝试已是最终结局（含非 infra 错误；账本缺
        accept 的先由 recover_crash_window 补写）的跳过、不占 infra 额度；只有最后一次为 infra 错误的才按 infra
        重试额度重跑。跑完读回账本：仍无 accept 的身份计 missing，>0 打印 RUN_INCOMPLETE 并返回 6。"""
        led = self.ledger
        self.recover_dangling()
        self.recover_crash_window()
        pending: list[dict] = []
        skip_acc = skip_full = skip_budget = 0
        for ident in rows:
            k = key_of(ident)
            if k in led.accepted or led.last_end_final(k):
                skip_acc += 1
            elif led.attempts_used(k) >= V8_MAX_ATTEMPTS:
                skip_full += 1
            else:
                pending.append(ident)
        print(f"RUN_PLAN policy={self.args.policy} dataset={self.dataset} total={len(rows)} "
              f"resume_skip={skip_acc + skip_full} accepted={skip_acc} attempts_full={skip_full} todo={len(pending)} "
              f"reset_left={led.reset_left()} infra_retry_left={led.infra_retries_left()}", flush=True)
        while pending:
            ident = pending.pop(0)
            k = key_of(ident)
            used = led.attempts_used(k)
            if k in led.accepted or led.last_end_final(k) or used >= V8_MAX_ATTEMPTS:
                continue
            retry = used >= 1
            attempt_no = led.attempts_total(k) + 1
            token = self.attempt_token(k, attempt_no) if led.shared is not None else None
            if retry and not led.allow_retry(k, token=token):
                skip_budget += 1
                print(f"INFRA_RETRY_BUDGET_EXHAUSTED policy={self.args.policy} key={k} "
                      f"used={led.infra_retries_used()} budget={led.infra_retry_budget}", flush=True)
                continue
            rec = self.run_one(ident, attempt=attempt_no, retry=retry)
            if rec.get("status") == "error" and rec.get("infra") and led.attempts_used(k) < V8_MAX_ATTEMPTS:
                pending.insert(0, ident)  # 原身份立即重试一次（额度在下一轮判断）
        if skip_budget:
            print(f"RUN_PARTIAL policy={self.args.policy} infra_retry_skipped={skip_budget}", flush=True)
        # 席位收尾读回账本真实缺失数（不只看是否跑完）：重试耗尽或 2 次用满仍无 accept 的身份
        missing = sorted({key_of(r) for r in rows} - set(led.accepted))
        if missing:
            print(f"RUN_INCOMPLETE policy={self.args.policy} seat={self.args.seat} dataset={self.dataset} "
                  f"total={len(rows)} missing={len(missing)} first={','.join(missing[:5])}", flush=True)
            return EXIT_INCOMPLETE
        return 0


def load_identities(args) -> list[dict]:
    """读身份清单 JSON 数组（``eval_manifest.py`` 的 shard-NN.json），按 ``--only`` 过滤、``--order`` 排序、
    ``--limit`` 截断。每行须为该数据集的执行身份行（字段齐全、nullable 严格、key 自洽、key 不重复），不符即运行阻塞。"""
    rows = json.loads(Path(args.identities).read_text(encoding="utf-8"))
    bad = [(i, b) for i, r in enumerate(rows) for b in [validate_v8_identity(r, args.dataset)] if b]
    keys = [key_of(r) for r in rows]
    dup = len(keys) - len(set(keys))
    if bad or dup:
        print(f"RUN_BLOCKED reason=identities dataset={args.dataset} n_bad={len(bad)} dup_keys={dup} "
              f"first={bad[:3]}", flush=True)
        raise SystemExit(EXIT_BLOCKED)
    if args.only:
        want = set(args.only.split(","))
        rows = [r for r in rows if key_of(r) in want]
    rows = order_identities(rows, args.order, args.shuffle_seed)
    if args.limit:
        rows = rows[: args.limit]
    return rows


def cmd_run(args) -> int:
    bad = check_run_args(args, need_identities=True)
    if bad:
        vp = variant_problems(args)
        if vp:  # 变体／adapter 错配先打具名原因（接口冻结说明 2.3），再打通用参数行
            print(f"RUN_BLOCKED reason=variant_pairing detail={'; '.join(vp)}", flush=True)
        print(f"RUN_BLOCKED reason=args detail={bad}", flush=True)
        return EXIT_BLOCKED
    blk = entry_blockers(args)
    if blk:
        print(f"RUN_BLOCKED reason={blk[0]} detail={blk[1]}", flush=True)
        return EXIT_BLOCKED
    t_proc = time.perf_counter()
    init = timed_imports()
    proc = {"init_timing": init, "env": env_versions(), **gpu_info(), **git_info(), **cpu_info()}
    policy_mod = load_sibling(POLICY_MODULES[args.policy])
    recorder_factory = None
    if not args.no_record:
        recorder_mod = load_sibling("recorder")

        def recorder_factory(rec_dir, meta):
            return recorder_mod.EpisodeRecorder(rec_dir, meta)
    init["process_ready_s"] = time.perf_counter() - t_proc
    print(f"CLIENT_READY policy={args.policy} variant={policy_variant_of(args)} seat={args.seat} cond={args.cond} "
          f"dataset={args.dataset} max_steps={args.max_steps} strict_cap={int(bool(args.strict_cap))} "
          f"policy_seed={args.policy_seed} route={policy_route(args)} "
          f"host={socket.gethostname()} gpu={proc.get('gpu_name')} sapien={proc['env']['sapien']} "
          f"torch={proc['env']['torch']} robomme_hard={proc['env']['robomme_hard_file']} "
          f"git={proc['git_commit'][:12]} dirty={proc['git_dirty']} init_s={init['process_ready_s']:.1f}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"process-{os.getpid()}.json").write_text(dumps(proc), encoding="utf-8")
    runner = SeatRunner(args, policy_mod=policy_mod, recorder_factory=recorder_factory, proc_info=proc)
    try:
        rc = runner.run_identities(load_identities(args))
    finally:
        runner.close()
    runner.progress(0, phase="finished")
    print(f"全部完成 policy={args.policy} seat={args.seat} dataset={args.dataset} episodes={runner.episodes_done} "
          f"rc={rc}", flush=True)
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="评估环境侧常驻客户端（ood／hard-verify）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--policy", required=True, choices=list(POLICIES),
                   help="策略标签；模块按 POLICY_MODULES 映射加载")
    p.add_argument("--identities", required=True, help="身份清单：eval_manifest.py 产出的 shard-NN.json")
    p.add_argument("--dataset", required=True, choices=list(DATASETS),
                   help="身份模式与 builder 数据集；ood 配 --max-steps 1800 --strict-cap，hard-verify 配 --max-steps 1300")
    p.add_argument("--max-steps", type=int, required=True, help="步数上限（无默认值，由入口按数据集给出）")
    p.add_argument("--strict-cap", action="store_true",
                   help="执行满 --max-steps 步仍未成功即停（第 max_steps+1 步不进环境，记 timeout、cap_hit=true）")
    p.add_argument("--groundsg-variant", default=None, choices=list(GROUNDSG_VARIANTS), help="--policy groundsg 必填：子目标来源")
    p.add_argument("--qwenvl-groundsg-adapter", default=None,
                   help="--groundsg-variant ground-sg-qwenvl 必填：QwenVL 子目标预测器的 adapter 目录")
    p.add_argument("--memer-adapter", default=None,
                   help="--groundsg-variant ground-sg-memer 必填：MemER 子目标预测器的 adapter 目录")
    p.add_argument("--policy-seed", type=int, default=None,
                   help="模型种子（必填，非负整数；缺失 RUN_BLOCKED reason=policy_seed）；进 seat_info、结果行与账本路线")
    p.add_argument("--server-metadata", default=None,
                   help="服务元数据 JSON（server-metadata-<port>.json，含 policy_seed）；缺省 <out>/server-metadata-<port>.json")
    p.add_argument("--context-deadline-s", type=float, default=DEFAULT_CONTEXT_DEADLINE_S,
                   help="策略上下文加载的绝对期限（秒，0=不限）；超期 infra_reason=deadline_context_load、退出 75")
    p.add_argument("--first-infer-deadline-s", type=float, default=DEFAULT_FIRST_INFER_DEADLINE_S,
                   help="本局开始到第一次 step 的绝对期限（秒，0=不限）；超期 infra_reason=deadline_first_infer、退出 75")
    p.add_argument("--media-deadline-s", type=float, default=DEFAULT_MEDIA_DEADLINE_S,
                   help="录制器收尾的绝对期限（秒，0=不限）；超期 infra_reason=deadline_media_finalize、退出 75")
    p.add_argument("--trace-root", default=None, help="每局轨迹根目录；本局目录为 <trace-root>/<key>.a<attempt>")
    p.add_argument("--cond", required=True, help="条件代号，如 E1／N")
    p.add_argument("--seat", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--order", default="forward", choices=["forward", "reverse", "shuffle"])
    p.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    p.add_argument("--only", default=None, help="只跑这些 <task>_<tier>_<seed>（逗号分隔）")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--episode-wall-s", type=float, default=0.0, help="单局墙钟上限（0=不限）；超时记基础设施超时并退出 75")
    p.add_argument("--first-extra-s", type=float, default=600.0,
                   help="本进程第一局额外放宽（首次推理编译）；run_seat.sh 只在 server 新（重）起后传 600，否则传 0")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--never-degrade", action="store_true")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--rec-root", default=None, help="录像目录根（默认 <out>/rec）；录像目录名 <key>.a<attempt_no>")
    led = p.add_argument_group("持久账本（契约 C2；--ledger／--infra-retry-budget 与五个预算参数必填，缺任一即 RUN_BLOCKED）")
    led.add_argument("--ledger", default=None, help="持久尝试账本 JSONL（追加写、fsync）")
    led.add_argument("--reset-budget", type=int, default=None,
                     help="可选：本账本可领的底层 reset 额度（build 与 reset 各算一次），给了才硬拦；不给只计量")
    led.add_argument("--infra-retry-budget", type=int, default=None,
                     help="本账本可用的基础设施重试局数（每身份至多重试 1 次；额度按模型共享，由主会话切给各席）")
    led.add_argument("--budget-raise-reason", default=None,
                     help="--reset-budget 大于账本历史最大值时写进 budget_raise 行与 RESET_BUDGET_RAISE 的原因（默认 cli_reset_budget）")
    led.add_argument("--budget-ledger", default=None,
                     help="共享预算账本（budget_ledger.py 格式）；CLI 必填（进程内 SeatRunner 不给时仍按环境变量 "
                          "SGEVAL_BUDGET_LEDGER 门控）")
    led.add_argument("--trajectory-cap", type=int, default=None, help="共享账本轨迹硬上限（CLI 必填，与 config 行比对）")
    led.add_argument("--shared-infra-cap", type=int, default=None, help="共享 infra 重试上限（CLI 必填）")
    led.add_argument("--expired-cap", type=int, default=None, help="到期接续上限（CLI 必填）")
    led.add_argument("--planned-first-tries", type=int, default=None, help="计划首试数（首试保留额度；CLI 必填）")
    p.set_defaults(func=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
