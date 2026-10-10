"""步数到顶实测（计划第二部分 1.7、第四节「步数到顶实测」）：不加载任何模型，用「保持当前关节位置」的动作一直走到结束。

先打印 ``builder.max_steps_without_demonstration``（预期 ``max_steps + 2``，以实际为准如实打印），再按所选循环口径走：

- ``--loop strict``：本仓库 V9 的 ``EnvSession(step_cap=max_steps)`` 语义——已执行 ``max_steps`` 步后再要 step 即不进入
  环境、记 ``strict_cap``（本文件内按同一语义实现，不依赖 ``env_client.py`` 的构造签名）。预期 1600／``strict_cap``。
- ``--loop mme-vla``：MME-VLA 官方 ``eval_each_episode``（GroundSG／FrameSamp+Modulation 共用）——先 ``step``、``count += 1``，再判 ``count > max_steps`` 记
  ``loop_count``，否则 ``terminated or truncated`` 即停。预期 ``max_steps + 1``／``loop_count``。
- ``--loop range``：PonderPounce ``SyncEpisodeRunner``、Astra 的 ``for t in range(max_steps)``（环境结束即 break）。
  预期 ``max_steps``／``loop_exit``。

环境先于循环口径结束时如实报 ``env_terminated``／``env_truncated`` 与实际步数（判 FAIL，记入决策项）。

``--derive-range``（只配 ``--loop mme-vla``）：xhard0 只真跑一局 mme-vla（MME-VLA 官方 eval.py）循环，``range`` 的结论取这一局的前 ``max_steps`` 步
（空动作确定、前缀相同），另出一行判定、标 ``source=prefix``，不写成真实跑过两条循环。

判定行::

    STEP_CAP=PASS|FAIL dataset=<…> max_steps=<n> builder_cap=<n> loop=<…> exec_steps=<n> terminal_reason=<…>
        expected_steps=<n> expected_reason=<…> builder_cap_ok=<0|1> source=run|prefix [official=1]

用法::

    python dev-scripts/checks/cap_probe.py --dataset hard-verify --max-steps 1300 --task VideoUnmask --episode 0 \
        --loop mme-vla --derive-range
    python dev-scripts/checks/cap_probe.py --dataset ood --max-steps 1600 --task VideoUnmask --episode <局> --loop strict
    python dev-scripts/checks/cap_probe.py --official --max-steps 1300 --task VideoUnmask --episode 3 --loop mme-vla

``--official`` 用官方 ``robomme`` builder 与 ``dataset="test"``（``--episode`` 为官方局号）；否则用
``robomme_ood`` builder（``--episode`` 为 builder 局号）。
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable

import numpy as np

LOOPS = ("strict", "mme-vla", "range")


def expected(loop: str, max_steps: int) -> tuple[int, str]:
    return {"strict": (max_steps, "strict_cap"), "mme-vla": (max_steps + 1, "loop_count"),
            "range": (max_steps, "loop_exit")}[loop]


def _flag(x: Any) -> bool:
    """terminated／truncated 可能是 bool、numpy 或 torch 张量。"""
    try:
        return bool(np.asarray(x.cpu() if hasattr(x, "cpu") else x).any())
    except Exception:  # noqa: BLE001
        return bool(x)


def hold_action(obs: dict, gripper: str = "auto") -> np.ndarray:
    """保持当前关节位置：7 个关节取观测最后一帧，夹爪按当前开度给 ±1（``auto``：开度 > 0.02 视为张开）。"""
    joint = np.asarray(obs["joint_state_list"][-1], dtype=np.float64).reshape(-1)[:7]
    if gripper == "auto":
        g = np.asarray(obs["gripper_state_list"][-1], dtype=np.float64).reshape(-1)
        cmd = 1.0 if (g.size and float(g[0]) > 0.02) else -1.0
    else:
        cmd = float(gripper)
    return np.concatenate([joint, [cmd]])


def run_loop(env, obs: dict, loop: str, max_steps: int, act: Callable[[dict], np.ndarray],
             progress: Callable[[int], None] | None = None) -> dict:
    """按循环口径走到结束，返回 ``{"exec_steps", "terminal_reason", "flags"}``；``flags`` 为逐步 (terminated, truncated)。"""
    if loop not in LOOPS:
        raise ValueError(loop)
    flags: list[tuple[bool, bool]] = []
    steps = 0
    reason = None

    def _step(o):
        nonlocal steps
        out = env.step(act(o))
        steps += 1
        o2, _r, term, trunc, _info = out
        flags.append((_flag(term), _flag(trunc)))
        if progress is not None:
            progress(steps)
        return o2, flags[-1]

    if loop == "strict":
        while True:
            if steps >= max_steps:
                reason = "strict_cap"
                break
            obs, (term, trunc) = _step(obs)
            if term or trunc:
                reason = "env_terminated" if term else "env_truncated"
                break
    elif loop == "mme-vla":
        while True:
            obs, (term, trunc) = _step(obs)
            if steps > max_steps:
                reason = "loop_count"
                break
            if term or trunc:
                reason = "env_terminated" if term else "env_truncated"
                break
    else:
        for _t in range(max_steps):
            obs, (term, trunc) = _step(obs)
            if term or trunc:
                reason = "env_terminated" if term else "env_truncated"
                break
        else:
            reason = "loop_exit"
    return {"exec_steps": steps, "terminal_reason": reason, "flags": flags}


def derive_range(flags: list[tuple[bool, bool]], max_steps: int) -> dict:
    """由 mme-vla 那一局的逐步标志推出 range 口径的结果（前 ``max_steps`` 步前缀）。"""
    for i, (term, trunc) in enumerate(flags[:max_steps]):
        if term or trunc:
            return {"exec_steps": i + 1, "terminal_reason": "env_terminated" if term else "env_truncated"}
    if len(flags) < max_steps:
        return {"exec_steps": len(flags), "terminal_reason": "prefix_too_short"}
    return {"exec_steps": max_steps, "terminal_reason": "loop_exit"}


def verdict_line(*, dataset: str, max_steps: int, builder_cap: Any, loop: str, res: dict, source: str,
                 official: bool) -> str:
    exp_n, exp_r = expected(loop, max_steps)
    ok = res["exec_steps"] == exp_n and res["terminal_reason"] == exp_r
    cap_ok = int(builder_cap == max_steps + 2)
    line = (f"STEP_CAP={'PASS' if ok else 'FAIL'} dataset={dataset} max_steps={max_steps} builder_cap={builder_cap} "
            f"loop={loop} exec_steps={res['exec_steps']} terminal_reason={res['terminal_reason']} "
            f"expected_steps={exp_n} expected_reason={exp_r} builder_cap_ok={cap_ok} source={source}")
    return line + (" official=1" if official else "")


def make_builder(args):
    """建 builder（真实仿真；测试不走这里）。"""
    if args.official:
        import robomme.robomme_env  # noqa: F401 注册官方 16 个环境
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        return BenchmarkEnvBuilder(env_id=args.task, dataset="test", action_space="joint_angle", gui_render=False,
                                   max_steps=args.max_steps)
    import robomme_ood.robomme_env  # noqa: F401 注册 16 个环境
    from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder(env_id=args.task, dataset=args.dataset, action_space="joint_angle", gui_render=False,
                               max_steps=args.max_steps)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="步数到顶实测（不加载模型，保持当前关节位置）")
    ap.add_argument("--dataset", choices=["hard-verify", "ood"], default="hard-verify")
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--official", action="store_true", help="官方 robomme builder、dataset=test，--episode 为官方局号")
    ap.add_argument("--loop", choices=LOOPS, required=True)
    ap.add_argument("--derive-range", action="store_true", help="只配 --loop mme-vla：另出 range 口径（source=prefix）")
    ap.add_argument("--gripper", default="auto", help="夹爪指令：auto（按当前开度）或数值")
    return ap


def main(argv: list[str] | None = None, *, builder_factory: Callable | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.derive_range and args.loop != "mme-vla":
        raise SystemExit("--derive-range 只配 --loop mme-vla")
    builder = (builder_factory or make_builder)(args)
    builder_cap = getattr(builder, "max_steps_without_demonstration", None)
    ds = "test" if args.official else args.dataset
    print(f"CAP_PROBE builder.max_steps_without_demonstration={builder_cap} max_steps={args.max_steps} "
          f"dataset={ds} task={args.task} episode={args.episode}", flush=True)
    env = builder.make_env_for_episode(args.episode)
    t0 = time.perf_counter()
    try:
        obs, _info = env.reset()
        res = run_loop(env, obs, args.loop, args.max_steps, lambda o: hold_action(o, args.gripper),
                       progress=lambda n: (n % 200 == 0) and print(f"CAP_PROBE progress step={n}", flush=True))
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    print(f"CAP_PROBE wall_s={time.perf_counter() - t0:.1f}", flush=True)
    line = verdict_line(dataset=ds, max_steps=args.max_steps, builder_cap=builder_cap, loop=args.loop, res=res,
                        source="run", official=args.official)
    print(line, flush=True)
    rc = 0 if line.startswith("STEP_CAP=PASS") else 1
    if args.derive_range:
        r2 = derive_range(res["flags"], args.max_steps)
        print(verdict_line(dataset=ds, max_steps=args.max_steps, builder_cap=builder_cap, loop="range", res=r2,
                           source="prefix", official=args.official), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
