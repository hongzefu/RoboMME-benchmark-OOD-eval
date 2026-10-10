"""单块停止与共享预算：使用会话替身，验证真实序列化账本，不启动环境。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robomme_ood_eval import episode, timing

ROOT = Path(__file__).resolve().parents[2]


def load_smoke():
    spec = importlib.util.spec_from_file_location("it_smoke_test", ROOT / "dev-scripts/checks/infer_timing_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("audit", [False, True])
def test_single_chunk_stops_without_extra_environment_step(monkeypatch, audit):
    module = load_smoke()
    class FakeSession:
        def __init__(self):
            self._cap_hit = False
        def step(self, action):
            return action
    ep = SimpleNamespace(EnvSession=FakeSession, StepCapReached=episode.StepCapReached)
    # 作用域结束恢复本进程计时方法，避免污染其它测试。
    monkeypatch.setattr(timing.ChunkTimer, "close", timing.ChunkTimer.close)
    state = module.install_stop(ep, timing, audit=audit)
    session = ep.EnvSession()
    if audit:
        session.step("hold")
        timer = timing.ChunkTimer(clock=lambda: 1.0)
        timer.start(0.0)
        timer.close(decision=0, env_step=1, action_rtt_ms=1, server_infer_ms=1)
    session.step("action")
    with pytest.raises(episode.StepCapReached):
        session.step("extra")
    assert session._cap_hit
    assert state["steps"] == (2 if audit else 1)
    assert state["steps_after_chunk"] == int(audit)


@pytest.mark.parametrize("formal", [False, True])
def test_main_charges_build_reset_and_preserves_formal_loop(tmp_path, monkeypatch, formal):
    module = load_smoke()
    out = tmp_path / "out"
    ledger_path = tmp_path / "shared.jsonl"
    monkeypatch.setenv("ROBOMME_EVAL_ROOT", str(ROOT))
    monkeypatch.setenv("IT_BUDGET_LEDGER", str(ledger_path))
    monkeypatch.setenv("SGEVAL_AUDIT", "1")
    monkeypatch.setattr(sys, "argv", ["entry", "--model", "astra", "--dataset", "hard-verify",
                                     "--tasks", "VideoUnmask", "--episodes", "0:1", "--out", str(out)]
                                    + (["--it-formal"] if formal else []))
    class FakeSession:
        def __init__(self, *, claim_reset):
            self._cap_hit = False
            claim_reset("build")
            claim_reset("reset")
        def step(self, action):
            return action
    monkeypatch.setattr(episode, "EnvSession", FakeSession)
    monkeypatch.setattr(timing.ChunkTimer, "close", timing.ChunkTimer.close)
    def fake_entry(path, *, run_name):
        assert Path(path) == ROOT / "scripts/evaluate.py" and run_name == "__main__"
        assert "--it-formal" not in sys.argv
        session = episode.EnvSession()
        timer = timing.ChunkTimer(clock=lambda: 1.0)
        timer.start(0.0)
        timer.close(decision=0, env_step=0, action_rtt_ms=1, server_infer_ms=1)
        session.step("first")
        if formal:
            assert session.step("second") == "second"
        else:
            with pytest.raises(episode.StepCapReached):
                session.step("second")
    monkeypatch.setattr(module.runpy, "run_path", fake_entry)
    module.main()
    budget = sys.modules["it_shared_budget"].BudgetLedger(ledger_path, trajectory_cap=27, reset_cap=54,
                                                         astra_cap=3, shared_infra_cap=6, expired_cap=0,
                                                         planned_first_tries=21)
    ok, lines = budget.report_lines()
    assert ok
    assert any("trajectories=1/27" in line and "resets=2/54" in line for line in lines)
    print(f"SINGLE_CHUNK_BUDGET=PASS mode={'formal' if formal else 'smoke'} trajectories=1 resets=2 real_resets=0")


def test_interrupted_same_out_charges_separate_attempts(tmp_path, monkeypatch):
    module = load_smoke()
    ledger_path = tmp_path / "shared.jsonl"
    out = tmp_path / "not_created"
    monkeypatch.setenv("ROBOMME_EVAL_ROOT", str(ROOT))
    monkeypatch.setenv("IT_BUDGET_LEDGER", str(ledger_path))
    argv = ["entry", "--model", "astra", "--dataset", "hard-verify", "--tasks", "VideoUnmask",
            "--episodes", "0:1", "--out", str(out), "--it-formal"]
    def interrupted(*args, **kwargs):
        raise RuntimeError("模拟创建输出前中断")
    monkeypatch.setattr(module.runpy, "run_path", interrupted)
    monkeypatch.setattr(episode, "EnvSession", episode.EnvSession)
    for _ in range(3):
        monkeypatch.setattr(sys, "argv", list(argv))
        with pytest.raises(RuntimeError, match="模拟创建输出前中断"):
            module.main()
    budget = sys.modules["it_shared_budget"].BudgetLedger(ledger_path, trajectory_cap=27, reset_cap=54,
                                                         astra_cap=3, shared_infra_cap=6, expired_cap=0,
                                                         planned_first_tries=21)
    assert budget._load().trajectories == 3 and budget._load().astra == 3
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(RuntimeError, match="astra=3/3") as refusal:
        module.main()
    assert refusal.value.reason == "astra_cap"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(ValueError, match="输出目录已存在"):
        module.main()
    assert budget._load().trajectories == 3
    print("SMOKE_ATTEMPTS=PASS interrupted=3 charged=3 fourth_refused=1 existing_out_refused=1 real_resets=0")
