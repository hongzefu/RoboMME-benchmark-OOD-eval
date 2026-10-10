"""本轮动作块冒烟与本机 Astra 预算入口；保留正式身份与步数配置。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy
import sys
import uuid


def install_stop(episode, timing, *, audit: bool):
    """审计开时等待首个合法块再执行一步；审计关时最多执行一步。"""
    state = {"chunk": False, "steps_after_chunk": 0, "steps": 0}
    original_close = timing.ChunkTimer.close

    def close(timer, **kwargs):
        row = original_close(timer, **kwargs)
        state["chunk"] = True
        return row

    timing.ChunkTimer.close = close
    original_session = episode.EnvSession

    class SmokeSession(original_session):
        def step(self, action):
            # 仅限制评估仓外层会话，不覆盖 benchmark 环境或第三方源码。
            if ((state["chunk"] and state["steps_after_chunk"] >= 1)
                    or (not audit and state["steps"] >= 1)
                    or state["steps"] >= 16):
                self._cap_hit = True
                raise episode.StepCapReached("SINGLE_CHUNK_SMOKE: 外层主动停止")
            out = super().step(action)
            state["steps"] += 1
            if state["chunk"]:
                state["steps_after_chunk"] += 1
            return out

    episode.EnvSession = SmokeSession
    return state


def main():
    root = Path(os.environ["ROBOMME_EVAL_ROOT"]).resolve()
    expected = root / "src" / "robomme_ood_eval" / "__init__.py"
    import robomme_ood_eval
    from robomme_ood_eval import episode, timing

    if Path(robomme_ood_eval.__file__).resolve() != expected:
        raise RuntimeError(f"SMOKE_IMPORT=FAIL actual={robomme_ood_eval.__file__} expected={expected}")
    formal = "--it-formal" in sys.argv
    if formal:
        sys.argv.remove("--it-formal")
    model = sys.argv[sys.argv.index("--model") + 1]
    if formal and model != "astra":
        raise ValueError("本机正式预算入口只允许 Astra")
    for flag, allowed in (("--dataset", "hard-verify"), ("--tasks", "VideoUnmask"), ("--episodes", "0:1")):
        if sys.argv[sys.argv.index(flag) + 1] != allowed:
            raise ValueError(f"本轮入口要求 {flag}={allowed}")
    out = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    if out.exists() or out.is_symlink():
        raise ValueError(f"输出目录已存在，拒绝重新执行：{out}")
    ledger_path = Path(os.environ["IT_BUDGET_LEDGER"])
    module_spec = importlib.util.spec_from_file_location("it_shared_budget", root / "dev-scripts/gl/budget_ledger.py")
    budget_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = budget_module
    module_spec.loader.exec_module(budget_module)
    ledger = budget_module.BudgetLedger(ledger_path, trajectory_cap=27, reset_cap=54, astra_cap=3,
                                       shared_infra_cap=6, expired_cap=0, planned_first_tries=21)
    route = f"{'formal' if formal else 'smoke'}-{model}-{os.environ.get('SGEVAL_AUDIT', '1')}"
    # 每次启动单独扣额；预检中断且尚未创建输出目录时也不能复用旧预约。
    rid = ledger.reserve(resets=2, route=route, key=str(out), token=f"{out}:{uuid.uuid4().hex}", kind_of_try="first",
                         astra=model == "astra")
    claims = {"n": 0}
    original_session = episode.EnvSession

    class BudgetSession(original_session):
        def __init__(self, *args, **kwargs):
            def claim(what):
                ledger.claim_reset(rid, what, route=route)
                claims["n"] += 1
            kwargs["claim_reset"] = claim
            super().__init__(*args, **kwargs)

    episode.EnvSession = BudgetSession
    state = ({"steps": 0, "chunk": False, "steps_after_chunk": 0} if formal else
             install_stop(episode, timing, audit=os.environ.get("SGEVAL_AUDIT", "1") != "0"))
    print(f"SMOKE_IMPORT=PASS root={root} model={model} out={out}", flush=True)
    try:
        runpy.run_path(str(root / "scripts/evaluate.py"), run_name="__main__")
    finally:
        ledger.commit(rid, resets=claims["n"], route=route)
        if formal:
            print(f"FORMAL_BUDGET_DONE model={model} resets={claims['n']}", flush=True)
        else:
            print(f"SINGLE_CHUNK_STOP steps={state['steps']} chunk={int(state['chunk'])} "
                  f"steps_after_chunk={state['steps_after_chunk']} resets={claims['n']}", flush=True)


if __name__ == "__main__":
    main()
