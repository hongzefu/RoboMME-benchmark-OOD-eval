#!/usr/bin/env python3
"""RoboMME OOD 评估本机入口（拆分方案 §三「scripts/evaluate.py 的调用链」）。

    scripts/evaluate.py --model <名> --dataset hard-verify[,ood] --seed N [--tasks A,B] [--episodes a:b] \\
        --out <目录> [--gpus 0[,1]] [模型参数…]

* ``--model`` ∈ {dummy, perceptual-framesamp-modul, groundsg, smvla, pp, astra}；``--seed`` 是模型种子 policy_seed；
* ``--episodes a:b`` 是 builder 局号半开区间（``0:2`` 即 0、1）；缺省该任务全部局；``--tasks`` 缺省官方 16 任务；
* 已有 ``result.json`` 的局直接跳过（可续跑）；
* ``--stop-server <metadata>``：只按元数据停掉看门狗退出后留下的服务端，然后退出（S2）；
* 模型参数：``--ckpt``、``--groundsg-variant``（如 ``ground-sg-oracle``）、``--port-base``、``--work-dir``、
  ``--compile-cache``，以及任意 ``--<名> <值>`` 或 ``--cfg 键=值``，一律透传给 ``load_policy(**cfg)``（连字符换下划线）。

主循环：``with load_policy(...) as p: for dataset: for task: for ep: run_episode(...)``，每个数据集跑完调
``report.summarize`` 写 ``log.json``。产物在 ``<out>/rollouts/<模型>/<数据集>/seed<seed>/``。

退出码：0 完成；2 参数错；3 运行阻塞（身份不符、服务端不符或已死、Astra 报停）；5 reset 额度耗尽。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

EXIT_BLOCKED = 3
EXIT_BUDGET = 5


def parse_episodes(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    a, sep, b = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"--episodes 须为 a:b 半开区间：{text!r}")
    lo, hi = int(a or 0), int(b)
    if lo < 0 or hi <= lo:
        raise argparse.ArgumentTypeError(f"--episodes 区间为空或非法：{text!r}")
    return lo, hi


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
    """把未声明的 ``--名 值`` ／ ``--开关`` 透传成 cfg（连字符换下划线）。"""
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


def build_parser() -> argparse.ArgumentParser:
    from robomme_ood_eval.models import MODELS

    p = argparse.ArgumentParser(prog="scripts/evaluate.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, help="模型注册名")
    p.add_argument("--dataset", help="hard-verify、ood，或逗号分隔两者（同一个 Policy 先后跑）")
    p.add_argument("--seed", type=int, help="模型种子 policy_seed（加载时定，属于 Policy）")
    p.add_argument("--tasks", help="逗号分隔任务名；缺省官方 16 任务")
    p.add_argument("--episodes", type=parse_episodes, help="builder 局号半开区间 a:b；缺省全部")
    p.add_argument("--out", type=Path, help="产物根目录")
    p.add_argument("--gpus", help="逗号分隔 GPU 编号（服务端放第一张；Astra 的监视模型放第二张）")
    p.add_argument("--stop-server", type=Path, metavar="METADATA", help="只按 server-metadata-<port>.json 停服务端后退出")
    g = p.add_argument_group("模型参数（透传给 load_policy）")
    g.add_argument("--ckpt")
    g.add_argument("--groundsg-variant")
    g.add_argument("--port-base", type=int)
    g.add_argument("--work-dir")
    g.add_argument("--compile-cache")
    g.add_argument("--cfg", action="append", default=[], metavar="键=值", help="任意模型参数，可重复")
    g.add_argument("--no-render", action="store_true", help="不出网站视频（只留原始产物）")
    return p


def make_cfg(args, extra: dict) -> dict:
    cfg: dict = {}
    for k in ("ckpt", "groundsg_variant", "port_base", "work_dir", "compile_cache"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.gpus:
        cfg["gpus"] = [int(x) for x in args.gpus.split(",") if x.strip()]
    for kv in args.cfg:
        if "=" not in kv:
            raise SystemExit(f"--cfg 须为 键=值：{kv!r}")
        k, v = kv.split("=", 1)
        cfg[k.replace("-", "_")] = _coerce(v)
    cfg.update(extra)
    return cfg


def episode_range(task: str, dataset: str, rng: tuple[int, int] | None) -> range:
    from robomme_ood_eval import episode as E

    n = int(E.builder_for(task, dataset).get_episode_num())
    if rng is None:
        return range(n)
    lo, hi = rng
    return range(lo, min(hi, n))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    from robomme_ood_eval.policy import ServerProcess

    if args.stop_server is not None:
        how = ServerProcess.stop_by_metadata(args.stop_server)
        return 0 if how in ("term", "kill", "gone") else EXIT_BLOCKED
    missing = [n for n in ("model", "dataset", "seed", "out") if getattr(args, n) is None]
    if missing:
        parser.error("必须给 " + " ".join(f"--{m}" for m in missing))
    from robomme_ood_eval import episode as E
    from robomme_ood_eval import report
    from robomme_ood_eval.policy import AstraStop, ServerDead, ServerMismatch, load_policy
    from robomme_ood_eval.session import ResetBudgetExhausted

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    bad = [d for d in datasets if d not in E.DATASET_MAX_STEPS]
    if bad or not datasets:
        parser.error(f"--dataset 只能是 {'、'.join(E.DATASET_MAX_STEPS)}（得到 {args.dataset!r}）")
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else list(E.TASKS)
    cfg = make_cfg(args, parse_extra(rest))
    code = 0
    done = skipped = 0
    try:
        with load_policy(args.model, args.seed, **cfg) as policy:
            for dataset in datasets:
                for task in tasks:
                    for ep in episode_range(task, dataset, args.episodes):
                        if E.result_path(policy, dataset, task, ep, args.out).is_file():
                            skipped += 1
                            print(f"EPISODE_SKIP dataset={dataset} task={task} episode={ep}（已有 result.json）",
                                  flush=True)
                            continue
                        E.run_episode(policy, dataset, task, ep, args.out, render=not args.no_render)
                        done += 1
                report.summarize(E.run_dir(args.out, policy.label, dataset, policy.policy_seed))
    except (E.IdentityMismatch, ServerDead, ServerMismatch, AstraStop) as e:
        print(f"RUN_BLOCKED reason={type(e).__name__} detail={e}", flush=True)
        code = EXIT_BLOCKED
    except ResetBudgetExhausted as e:
        print(f"RESET_BUDGET_EXHAUSTED detail={e}", flush=True)
        code = EXIT_BUDGET
    print(f"EVAL_DONE model={args.model} datasets={','.join(datasets)} seed={args.seed} episodes_run={done} "
          f"skipped={skipped} exit={code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
