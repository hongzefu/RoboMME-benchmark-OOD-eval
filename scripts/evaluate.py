#!/usr/bin/env python3
"""Local entry point for RoboMME OOD evaluation (split plan section 3, "call chain of scripts/evaluate.py").

    scripts/evaluate.py --model <name> --dataset hard-verify[,ood] --seed N [--tasks A,B] [--episodes a:b] \\
        --out <dir> [--gpus 0[,1]] [--ckpt <path>] [model options...]

* ``--model`` is one of {dummy, perceptual-framesamp-modul, groundsg, smvla, pp, astra}; ``--seed`` is the model
  seed policy_seed;
* ``--ckpt`` has no default: perceptual-framesamp-modul, groundsg, smvla and pp require it on every run
  (missing it is an argument error, exit code 2); dummy and astra do not use it;
* ``--episodes a:b`` is a half-open range of builder episode indices (``0:2`` means 0 and 1); default is every
  episode of the task; ``--tasks`` defaults to the 16 official tasks;
* episodes that already have a ``result.json`` are skipped (runs are resumable);
* ``--stop-server <metadata>``: only stop the server left behind after its watchdog exited, using the metadata
  file, then exit (S2);
* model options: ``--ckpt``, ``--groundsg-variant`` (e.g. ``ground-sg-oracle``), ``--port-base``, ``--work-dir``,
  ``--compile-cache``, plus any ``--<name> <value>`` or ``--cfg key=value``; all are passed through to
  ``load_policy(**cfg)`` (hyphens become underscores).

Main loop: ``with load_policy(...) as p: for dataset: for task: for ep: run_episode(...)``; after each dataset
``report.summarize`` writes ``log.json``. Outputs go to ``<out>/rollouts/<model>/<dataset>/seed<seed>/``.

Exit codes: 0 done; 2 argument error; 3 run blocked (identity mismatch, server mismatch or dead, Astra stop);
5 reset budget exhausted.
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
        raise argparse.ArgumentTypeError(f"--episodes must be a half-open range a:b: {text!r}")
    lo, hi = int(a or 0), int(b)
    if lo < 0 or hi <= lo:
        raise argparse.ArgumentTypeError(f"--episodes range is empty or invalid: {text!r}")
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
    """Pass undeclared ``--name value`` / ``--switch`` tokens through as cfg (hyphens become underscores)."""
    cfg: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--") or tok == "--":
            raise SystemExit(f"cannot parse argument: {tok!r}")
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
    from robomme_ood_eval.models import MODELS, MODELS_REQUIRING_CKPT

    p = argparse.ArgumentParser(prog="scripts/evaluate.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, help="registered model name")
    p.add_argument("--dataset", help="hard-verify, ood, or both comma-separated (run one after another by the same "
                                     "Policy)")
    p.add_argument("--seed", type=int, help="model seed policy_seed (fixed at load time, belongs to the Policy)")
    p.add_argument("--tasks", help="comma-separated task names; default is the 16 official tasks")
    p.add_argument("--episodes", type=parse_episodes, help="half-open range a:b of builder episode indices; "
                                                           "default is all")
    p.add_argument("--out", type=Path, help="output root directory")
    p.add_argument("--gpus", help="comma-separated GPU indices (the server goes on the first; Astra's monitor model "
                                  "goes on the second)")
    p.add_argument("--stop-server", type=Path, metavar="METADATA",
                   help="only stop the server described by server-metadata-<port>.json, then exit")
    g = p.add_argument_group("model options (passed through to load_policy)")
    g.add_argument("--ckpt", help="checkpoint path; no default, required for "
                                  + ", ".join(sorted(MODELS_REQUIRING_CKPT)))
    g.add_argument("--groundsg-variant")
    g.add_argument("--port-base", type=int)
    g.add_argument("--work-dir")
    g.add_argument("--compile-cache")
    g.add_argument("--cfg", action="append", default=[], metavar="KEY=VALUE",
                   help="any model option, may be repeated")
    g.add_argument("--no-render", action="store_true", help="do not render website videos (keep raw outputs only)")
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
            raise SystemExit(f"--cfg must be key=value: {kv!r}")
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
    from robomme_ood_eval.models import MODELS_REQUIRING_CKPT
    from robomme_ood_eval.policy import ServerProcess

    if args.stop_server is not None:
        how = ServerProcess.stop_by_metadata(args.stop_server)
        return 0 if how in ("term", "kill", "gone") else EXIT_BLOCKED
    missing = [n for n in ("model", "dataset", "seed", "out") if getattr(args, n) is None]
    if missing:
        parser.error("missing required arguments: " + " ".join(f"--{m}" for m in missing))
    from robomme_ood_eval import episode as E
    from robomme_ood_eval import report
    from robomme_ood_eval.policy import AstraStop, ServerDead, ServerMismatch, load_policy
    from robomme_ood_eval.session import ResetBudgetExhausted

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    bad = [d for d in datasets if d not in E.DATASET_MAX_STEPS]
    if bad or not datasets:
        parser.error(f"--dataset must be one of {', '.join(E.DATASET_MAX_STEPS)} (got {args.dataset!r})")
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else list(E.TASKS)
    cfg = make_cfg(args, parse_extra(rest))
    if args.model in MODELS_REQUIRING_CKPT and "ckpt" not in cfg:
        parser.error(f"--ckpt is required for model {args.model}")
    code = 0
    done = skipped = 0
    try:
        with load_policy(args.model, args.seed, **cfg) as policy:
            for dataset in datasets:
                for task in tasks:
                    for ep in episode_range(task, dataset, args.episodes):
                        if E.result_path(policy, dataset, task, ep, args.out).is_file():
                            skipped += 1
                            print(f"EPISODE_SKIP dataset={dataset} task={task} episode={ep} (result.json exists)",
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
