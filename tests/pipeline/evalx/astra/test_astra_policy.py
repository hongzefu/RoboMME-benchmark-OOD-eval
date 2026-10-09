"""Process-level lifecycle of ``AstraPolicy``: load starts each of the two servers once, reset is the pre-episode
refusal point, close stops the process groups (no network, no GPU).

- ``load``: the cost guard (``servers/astra_cost_guard.py --cap 5``) and the VLA server (``scripts/serve_policy.py``,
  first GPU, ``--seed=<policy_seed>``, API key removed from its env) are each started once, guard first then VLA;
  this process is pinned to the second GPU before building the monitor model. Invalid config, an existing
  ``STOP.json`` or an exhausted episode budget are refused before any server starts (or right after the guard starts).
- ``reset``: sends no server messages and does not touch the env; refuses on STOP, a stop condition left by the
  previous episode, a dead server, or the 3rd episode.
- ``close``: VLA first, then guard; the real-subprocess case checks that the whole process group exits and the guard
  writes ``exited=true``.

Verdict line: ``ASTRA_POLICY_LIFECYCLE=PASS`` (printed by ``test_close_stops_real_process_groups``).
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import pytest

from astra_fakes import FAKE_KEY, FakeServer, Harness, NetCounter, astra_session, load_guard, third_party


def _spec(h, task="BinFill", episode=0, dataset="hard-verify"):
    spec, _ = h.E.make_spec(h.policy, dataset, task, episode, h.out)
    return spec


# -- load ------------------------------------------------------------------

def test_load_starts_guard_and_vla_once_and_close_stops(tmp_path, monkeypatch):
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        policy = h.load(seed=7)
        servers = h.servers()
        guard, vla = servers["astra-guard"], servers["astra-vla"]
        assert FakeServer.LOG == [("start", "astra-guard"), ("start", "astra-vla")], "guard first, then VLA"
        assert guard.starts == vla.starts == 1 and len(FakeServer.instances) == 2
        g = load_guard()
        state = g.default_state_path(h.ledger)
        assert guard.argv == [sys.executable, str(Path(g.__file__).resolve()), "--root", str(h.group),
                              "--prices", str(h.prices_path), "--ledger", str(h.ledger), "--state", str(state),
                              "--cap", "5", "--interval", "2"]
        assert guard.port == vla.port + 1 and guard.metadata_dir == vla.metadata_dir == h.ledger.parent / "servers"
        ckpt = h.cfg["ckpt"]
        assert vla.argv == [h.cfg["astra_vla_python"], "scripts/serve_policy.py", f"--port={policy.port}", "--seed=7",
                            "policy:checkpoint", "--policy.config=mme_vla_suite", f"--policy.dir={ckpt}"]
        assert vla.gpu == 0 and vla.cwd == str(third_party()) and vla.policy_seed == 7 and vla.ckpt == ckpt
        assert [r.kind for r in vla.ready] == ["port"] and "OPENAI_API_KEY" not in vla.env
        root = third_party()
        assert vla.env["PYTHONPATH"].split(os.pathsep) == [
            str(root / "examples" / "champ"), str(root / "src"), str(root / "packages" / "openpi-client" / "src"),
            str(mod.benchmark_src())]
        assert vla.env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" and vla.env["HF_HUB_OFFLINE"] == "1"
        assert h.monitor_calls == [("fake-base", h.cfg["astra_monitor_adapter"], "1")], "monitor model is built on the second GPU"
        assert h.check_calls == [(ckpt, h.cfg["astra_monitor_adapter"])]
        assert policy.server_seed() == 7 and policy.servers() == [vla, guard]
        assert policy.spool.parent.parent == h.group and policy.spool.name == "planner_calls"
        assert json.loads(state.read_text())["cap"] == 5.0
        policy.close()
        policy.close()
        assert FakeServer.LOG[2:] == [("stop", "astra-vla"), ("stop", "astra-guard")], "stop VLA first, then the guard"
        assert guard.stops == vla.stops == 1, "close is idempotent"
    assert policy.calls["load"] == 1 and policy.calls["reset"] == 0 and policy.calls["play"] == 0
    assert net.calls == 0


@pytest.mark.parametrize("override,match", [
    ({"gpus": [0]}, "astra_gpus"),
    ({"gpus": [1, 1]}, "astra_gpus"),
    ({"astra_cap_usd": 5.01}, "ASTRA_COST_BLOCKED"),
    ({"astra_ledger": None}, "--astra-ledger"),
    ({"astra_prices": None}, "--astra-prices"),
    ({"ckpt": None}, "--ckpt or --astra-vla-checkpoint"),
    ({"astra_monitor_adapter": None}, "--astra-monitor-adapter"),
    ({"astra_group_dir": "elsewhere"}, "layout"),
    ({"_no_key": True}, "astra_key"),
])
def test_load_rejects_bad_cfg_before_any_server(tmp_path, monkeypatch, override, match):
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        if override.pop("_no_key", False):
            monkeypatch.setenv("OPENAI_API_KEY", "")
        for k, v in override.items():
            if v is None:
                h.cfg.pop(k, None)
            else:
                h.cfg[k] = str(tmp_path / v) if k == "astra_group_dir" else v
        with pytest.raises(ValueError, match=match):
            h.load()
    assert FakeServer.instances == [] and h.responder.calls == 0


def test_load_refuses_when_stop_exists(tmp_path, monkeypatch):
    """``group_0/STOP.json`` already exists (guard hit the cap or written by hand): load refuses before starting any
    server; ledger and STOP are left untouched."""
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        h.group.mkdir(parents=True)
        (h.group / "STOP.json").write_text('{"reason": "astra_cost_cap"}')
        with pytest.raises(mod.AstraStop) as info:
            h.load()
    assert info.value.reason == "host_stop" and FakeServer.instances == []
    assert json.loads((h.group / "STOP.json").read_text())["reason"] == "astra_cost_cap"


def test_load_refuses_when_episode_cap_already_used(tmp_path, monkeypatch):
    """The reservations file already holds 2 episodes (accumulated across processes): refuse right after the guard
    starts and checks the books, without starting VLA; ``load_policy`` cleans up by stopping the guard."""
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        g = load_guard()
        state = g.default_state_path(h.ledger)
        g.save_reservations(state, {"schema": g.RESERVATIONS_SCHEMA, "reservations": {},
                                    "episodes": ["hard-verify:BinFill:0", "ood:BinFill:3"]})
        with pytest.raises(mod.AstraStop) as info:
            h.load()
    assert info.value.reason == "episode_cap"
    assert [s.name for s in FakeServer.instances] == ["astra-guard"] and FakeServer.instances[0].stops == 1


def test_pin_visible_gpu_refuses_after_cuda_init(monkeypatch):
    with astra_session() as (mod, _astra):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        fake_torch = type("T", (), {"cuda": type("C", (), {"is_initialized": staticmethod(lambda: True)})})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        with pytest.raises(RuntimeError, match="CUDA is already initialized in this process"):
            mod.pin_visible_gpu(1)
        assert mod.pin_visible_gpu(0) == "0"
        monkeypatch.setitem(sys.modules, "torch", type("T", (), {"cuda": type("C", (), {
            "is_initialized": staticmethod(lambda: False)})}))
        assert mod.pin_visible_gpu(1) == "1" and os.environ["CUDA_VISIBLE_DEVICES"] == "1"


# -- reset: pre-episode refusal point ---------------------------------------

def test_reset_registers_episode_without_server_messages(tmp_path, monkeypatch):
    """reset: probes both servers and registers the episode; does not touch the env (builder built none), sends no
    reset to VLA and no planner request. Re-registering the same episode is not double counted; the 3rd distinct
    episode raises ``AstraStop(episode_cap)``."""
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        policy = h.load()
        checks0 = {n: s.checks for n, s in h.servers().items()}
        policy.reset(_spec(h, "BinFill", 0))
        policy.reset(_spec(h, "BinFill", 0))
        assert mod.registered_episodes(policy.state_path) == ["hard-verify:BinFill:0"]
        assert all(s.checks == checks0[n] + 2 for n, s in h.servers().items())
        assert h.cls.make_calls == [] and h.vla.resets == 0 and h.vla.infers == 0 and h.responder.calls == 0
        policy.reset(_spec(h, "VideoUnmask", 0))
        with pytest.raises(mod.AstraStop) as info:
            policy.reset(_spec(h, "MoveCube", 0))
        assert info.value.reason == "episode_cap"
        policy.close()


def test_reset_dead_server_raises_server_dead(tmp_path, monkeypatch):
    from robomme_ood_eval.policy import ServerDead

    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        policy = h.load()
        h.servers()["astra-vla"].dead = True
        with pytest.raises(ServerDead):
            policy.reset(_spec(h))
        assert mod.registered_episodes(policy.state_path) == [], "no episode is registered when a server is dead"
        policy.close()


def test_reset_stop_file_and_pending_stop_are_sticky(tmp_path, monkeypatch):
    """STOP.json appears -> reset raises ``AstraStop(host_stop)``; even after STOP is deleted this Policy starts no
    more episodes (the whole batch stops). A stop condition left by play is likewise raised by the next reset."""
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        policy = h.load()
        (h.group / "STOP.json").write_text("{}")
        with pytest.raises(mod.AstraStop) as info:
            policy.reset(_spec(h))
        assert info.value.reason == "host_stop"
        (h.group / "STOP.json").unlink()
        with pytest.raises(mod.AstraStop):
            policy.reset(_spec(h))
        assert mod.registered_episodes(policy.state_path) == []
        policy._stop = mod.AstraStop("planner_error", "x")
        with pytest.raises(mod.AstraStop) as info2:
            policy.reset(_spec(h))
        assert info2.value.reason == "planner_error"
        policy.close()


def test_reset_pairing_mismatch(tmp_path, monkeypatch):
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        policy = h.load()
        with pytest.raises(ValueError, match="step_cap_pairing"):
            policy.reset(dataclasses.replace(_spec(h), max_steps=1800))
        policy.close()


# -- close: process groups of real subprocesses ----------------------------

_FAKE_VLA_SCRIPT = r"""
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
print("KEY=" + os.environ.get("OPENAI_API_KEY", "<none>"), "CVD=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""),
      "CHILD=%d" % child.pid, flush=True)
print("READY", flush=True)
time.sleep(600)
"""


def test_close_stops_real_process_groups(tmp_path, monkeypatch):
    """Real subprocesses: the guard is the real ``astra_cost_guard.py`` (``GuardProcess``, ready = fresh heartbeat);
    the VLA slot holds a fake server that spawns its own child (``VLAServerProcess``, no API key in env,
    ``CUDA_VISIBLE_DEVICES`` = first GPU). ``close`` stops the VLA process group first (grandchild included), then
    the guard (which writes ``exited=true``); both metadata files are removed."""
    from robomme_ood_eval import policy as P

    monkeypatch.setattr(P, "wait_gpu_free", lambda pid, **k: True)  # do not query nvidia-smi
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra, astra_guard_interval=0.5)
        policy = mod.AstraPolicy(7, **h.cfg)
        policy._parse_cfg()
        port = P.pick_port(None)
        guard = mod.GuardProcess(policy.guard_argv(), cwd=str(mod.REPO_ROOT), port=port + 1,
                                 metadata_dir=policy.server_dir, name="astra-guard",
                                 log_path=policy.server_dir / "astra-guard.log", ready_timeout_s=60,
                                 state_path=policy.state_path)
        vla = mod.VLAServerProcess([sys.executable, "-c", _FAKE_VLA_SCRIPT], env={"PYTHONPATH": ""}, gpu=0,
                                   ready=[P.Ready.log("READY")], port=port, metadata_dir=policy.server_dir,
                                   policy_seed=7, name="astra-vla", ready_timeout_s=60)
        policy.guard, policy.server = guard, vla
        try:
            guard.start()
            vla.start()
            assert guard.alive() and vla.alive()
            line = next(x for x in vla.log_path.read_text().splitlines() if x.startswith("KEY="))
            assert "KEY=<none>" in line and "CVD=0" in line, line
            child = int(line.rsplit("CHILD=", 1)[1])
            assert P.pid_alive(child)
            assert mod.CostGate(policy.state_path).read_state()["cap"] == 5.0
            meta = [guard.metadata_path, vla.metadata_path]
            assert all(p.is_file() for p in meta)
        finally:
            policy.close()
        deadline = time.time() + 10
        while P.pid_alive(child) and time.time() < deadline:
            time.sleep(0.05)
        assert not P.pid_alive(child), "the grandchild in the VLA process group must exit too"
        assert not guard.alive() and not vla.alive()
        assert json.loads(policy.state_path.read_text())["exited"] is True, "the guard writes exited=true after TERM"
        assert not any(p.exists() for p in meta)
        with pytest.raises(mod.GuardRefused, match="exited"):
            mod.CostGate(policy.state_path).read_state()
    print("ASTRA_POLICY_LIFECYCLE=PASS servers=2 group_stop=1 guard_exited=1")
