#!/usr/bin/env python3
"""C／D 路的 worker：官方 ``_worker`` 的最小镜像，只多传两个显式输入。

方案第三节要求 C 路在 B 的基础上显式传 ``sampling_config``、D 路再加 ``episode_spec``，
而官方 ``_worker`` 的 ``gym.make`` 参数表是写死的，无法从外部注入这两个键。

为把「镜像漂移」压到最小，本文件**只复制 `gym.make` 参数表那一段**，其余全部直接调用
官方模块里的同一批函数：``_planner_classes``、``_execute_tasks``、``_raw_summary``、
``STICK_TASKS`` 与几个异常类。官方源码本身不改、不打补丁。

与官方 ``_worker`` 的差异清单（除此之外逐句一致）：
1. ``kwargs`` 里按需加入 ``sampling_config`` 与 ``episode_spec``；两者都为 ``None`` 时
   参数表与官方完全相同，因此 B 路仍可继续用官方 ``_worker``。
2. 返回体多一个 ``inputs`` 字段，记录本局实际传入了哪两个显式输入的散列，供 G4 核验。
3. 环境包由环境变量 ``ROBOMME_ENV_PACKAGE``（``robomme`` 缺省／``robomme_ood``）决定，所有 ``robomme…``
   导入都经 ``importlib`` 按包名取；返回体多 ``env_package``、``env_module``（``REGISTERED_ENVS[task].cls.__module__``）
   与 ``wrapper_modules``，供 ``ENV_PACKAGE_BINDING`` 核验各侧实际加载的包（0927 计划 §3.3）。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _digest(payload: Any) -> str | None:
    if payload is None:
        return None
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def run_one(payload: tuple) -> dict[str, Any]:
    """在 spawn 子进程里跑一条身份；``payload=(job, sampling_config, episode_spec)``。"""
    # V4：payload 可带第 4 项 disable_recovery（runner --no-recovery）；三元组时行为与改动前逐字相同
    job, sampling_config, episode_spec = payload[:3]
    disable_recovery = bool(payload[3]) if len(payload) > 3 else False
    # xhard0（v7 方案第二部分 §1.5）：第 5 项 builder_route="hard-verify" 时 gym.make 实参取自 robomme_ood 构建器
    # dataset="hard-verify" 的条目（按数据集名取局：该数据集只有 xhard0，局 0～11 对应官方原 episode 3,7,…,47）
    builder_route = payload[4] if len(payload) > 4 else None
    import generate_dataset as official  # 官方固定源码，父进程已把其目录放进 sys.path

    os.environ["CUDA_VISIBLE_DEVICES"] = job.gpu
    worker_dir = Path(job.worker_dir)
    raw_path = worker_dir / "hdf5_files" / f"{job.task}_ep{job.episode}_seed{job.seed}.h5"
    record_env: Any | None = None
    env_package = os.environ.get("ROBOMME_ENV_PACKAGE", "robomme")
    binding: dict[str, Any] = {"env_package": env_package, "env_module": None, "wrapper_modules": None,
                               "builder_route": builder_route, "builder_episode": None, "builder_tier": None,
                               "recovery_mode": job.recovery_mode}
    caught: BaseException | None = None
    error_traceback: str | None = None
    try:
        source_root = Path(job.repo_root) / "src"
        if not source_root.is_dir():
            raise official.DatasetGenerationError(f"src does not exist: {source_root}")
        sys.path.insert(0, str(source_root))
        import gymnasium as gym
        import torch
        if env_package not in ("robomme", "robomme_ood", "robomme_hard"):
            raise ValueError(f"ROBOMME_ENV_PACKAGE 非法：{env_package}")
        importlib.import_module(f"{env_package}.robomme_env")  # 注册环境
        wrappers = importlib.import_module(f"{env_package}.env_record_wrapper")
        FailsafeTimeout, RobommeRecordWrapper = wrappers.FailsafeTimeout, wrappers.RobommeRecordWrapper  # noqa: F841
        importlib.import_module(f"{env_package}.robomme_env.utils.SceneGenerationError")
        fail_safe = importlib.import_module(f"{env_package}.robomme_env.utils.planner_fail_safe")
        FailAwarePandaArmMotionPlanningSolver = fail_safe.FailAwarePandaArmMotionPlanningSolver
        FailAwarePandaStickMotionPlanningSolver = fail_safe.FailAwarePandaStickMotionPlanningSolver
        ScrewPlanFailure = fail_safe.ScrewPlanFailure
        from mani_skill.utils.registration import REGISTERED_ENVS

        binding["env_module"] = REGISTERED_ENVS[job.task].cls.__module__
        binding["wrapper_modules"] = {"RobommeRecordWrapper": RobommeRecordWrapper.__module__,
                                      "planner_fail_safe": fail_safe.__name__}

        arm_cls, stick_cls = official._planner_classes(
            FailAwarePandaArmMotionPlanningSolver,
            FailAwarePandaStickMotionPlanningSolver,
            ScrewPlanFailure,
        )
        kwargs: dict[str, Any] = {
            "obs_mode": "rgb+depth+segmentation",
            "control_mode": "pd_joint_pos",
            "render_mode": "rgb_array",
            "reward_mode": "dense",
            "seed": job.seed,
            "difficulty": job.difficulty,
        }
        if job.recovery_mode is not None and not disable_recovery:
            kwargs["robomme_failure_recovery"] = True
            kwargs["robomme_failure_recovery_mode"] = job.recovery_mode
        # ── 与官方 _worker 的唯一参数差异 ──────────────────────────────────────
        if sampling_config is not None:
            kwargs["sampling_config"] = sampling_config
        if episode_spec is not None:
            # D 路走的是步 4 的原值回注通道 native_episode_spec：原抽样照常执行、
            # 用于建场景的值来自冻结规格；旧的 episode_spec 注入通道保持不变（红线 R9）。
            kwargs["native_episode_spec"] = episode_spec
        if builder_route is not None:
            if builder_route != "hard-verify" or env_package != "robomme_ood":
                raise ValueError(f"builder_route={builder_route!r} 只允许 hard-verify 且环境包为 robomme_ood")
            builder_mod = importlib.import_module("robomme_ood.env_record_wrapper.hard_builder")
            builder = builder_mod.BenchmarkEnvBuilder(job.task, dataset=builder_route)
            identities = [builder.resolve_identity(ep) for ep in range(builder.get_episode_num())]
            hits = [ep for ep, ident in enumerate(identities)
                    if ident["tier"] == "xhard0" and ident["seed"] == job.seed
                    and ident["source_episode"] == job.episode]
            if len(hits) != 1:
                raise ValueError(f"{builder_route} 构建器里找不到唯一的 xhard0 条目：{job.task} seed={job.seed} "
                                 f"ep={job.episode}（命中 {len(hits)} 条）")
            chosen = builder._hard_env_kwargs(hits[0])
            if set(chosen) != {"seed", "difficulty"} or chosen != {"seed": job.seed, "difficulty": job.difficulty}:
                raise ValueError(f"builder 路线的环境参数与官方 job 不符：{chosen}")
            kwargs.update(chosen)
            binding.update(builder_episode=hits[0], builder_tier="xhard0")
        # ─────────────────────────────────────────────────────────────────────
        worker_dir.mkdir(parents=True, exist_ok=False)
        base_env = gym.make(job.task, **kwargs)
        record_env = RobommeRecordWrapper(
            base_env,
            dataset=str(worker_dir),
            env_id=job.task,
            episode=job.episode,
            seed=job.seed,
            save_video=True,
        )
        record_env.reset()
        planner_kwargs: dict[str, Any] = {
            "debug": False,
            "vis": False,
            "base_pose": record_env.unwrapped.agent.robot.pose,
            "visualize_target_grasp_pose": False,
            "print_env_info": False,
        }
        if job.task in official.STICK_TASKS:
            planner_kwargs["joint_vel_limits"] = 0.3
            planner = stick_cls(record_env, **planner_kwargs)
        else:
            planner = arm_cls(record_env, **planner_kwargs)
        official._execute_tasks(record_env, planner, torch, job)
        # 只读导出／回注核验：C 路把本局规格封存，D 路把兼容核验结果落档
        recorder = getattr(record_env.unwrapped, "_spec", None)
        if recorder is not None:
            # P3：随机流轨迹。三路都写，因为 D 路的兼容抽样照常发生，
            # 比的是「调用序号 + 取值点签名 + 抽样结果」这条有序序列。
            base_env = record_env.unwrapped
            states = {}
            for attribute in ("generator", "_hb_generator"):
                generator = getattr(base_env, attribute, None)
                if generator is not None and hasattr(generator, "get_state"):
                    states[attribute] = hashlib.sha256(
                        bytes(generator.get_state().numpy().tobytes())
                    ).hexdigest()
            (worker_dir / "rng_trace.json").write_text(
                json.dumps(
                    {
                        "schema": "train-parity-rng-trace/1",
                        "mode": recorder.mode,
                        "calls": [
                            {"index": index, "path": item["path"],
                             "drawn": item.get("drawn", item.get("value")),
                             "source": item["source"]}
                            for index, item in enumerate(recorder.trace)
                        ],
                        "final_generator_states": states,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            recorder.identity.update(
                {"task": job.task, "episode": job.episode, "seed": job.seed,
                 "difficulty": job.difficulty, "recovery_mode": job.recovery_mode}
            )
            if recorder.mode == "export":
                (worker_dir / "episode_spec.json").write_text(
                    json.dumps(recorder.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                consumed = set(recorder.consumed_paths())
                leaves = set(recorder.leaf_paths())
                # 「有记录却没被消费」＝规格里存了值但这一局没走到那个调用点；
                # 这正是 G4 要抓的「规格没被真正消费」的另一面。
                unused = sorted(path for path in leaves if not any(
                    path == item or path.startswith(item + ".") for item in consumed
                ))
                (worker_dir / "spec_replay.json").write_text(
                    json.dumps(
                        {"mismatches": recorder.mismatches,
                         "value_points": len(recorder.trace),
                         "consumed": sorted(consumed),
                         "unused": unused,
                         # V7 分层回注（§1.5）：layout_drift 已计入 mismatches；layout_overridden 只报告
                         "layered": bool(getattr(recorder, "layered", False)),
                         "layout_hit": len({i["path"] for i in recorder.trace
                                            if i["path"] in (getattr(recorder, "_layered_hit", None) or ())}),
                         "layout_overridden": int(getattr(recorder, "layout_overridden", 0)),
                         "layout_drift": int(getattr(recorder, "layout_drift", 0))},
                        ensure_ascii=False, indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
    except Exception as exc:  # noqa: BLE001 官方同样先分类后兜底，这里合并但保留类型名
        caught = exc
        error_traceback = traceback.format_exc()
    finally:
        if record_env is not None:
            try:
                record_env.close()
            except Exception as close_exc:  # noqa: BLE001 与官方一致
                if caught is None:
                    caught = close_exc
                    error_traceback = traceback.format_exc()

    base = {
        "task": job.task,
        "episode": job.episode,
        "seed": job.seed,
        "difficulty": job.difficulty,
        "gpu": job.gpu,
        "recovery_mode": job.recovery_mode,
        "attempt_count": 1,
        **binding,
        "inputs": {
            "sampling_config_sha256": _digest(sampling_config),
            "episode_spec_sha256": _digest(episode_spec),
            "spec_mode": None if record_env is None or getattr(record_env.unwrapped, "_spec", None) is None
                          else record_env.unwrapped._spec.mode,
        },
    }
    if caught is not None:
        return {
            **base,
            "ok": False,
            "error_type": type(caught).__name__,
            "error": str(caught),
            "traceback": error_traceback,
        }
    try:
        return {**base, "ok": True, **official._raw_summary(raw_path, job)}
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
