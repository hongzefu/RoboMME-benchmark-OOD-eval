"""Contract tests for the outer ``run_episode`` and the three data structures (split plan, section 3: "three data
structures" and "two process kinds and fault handling").

Everything uses a fake builder / fake env / fake recorder / fake Policy (``tests/unit_eval/fakes.py``); no simulation,
no GPU. Pins ``EpisodeSpec`` / ``EpisodeResult`` / ``PlayOutput`` field by field, the call order (reset before build,
play after build, teardown session.close -> recorder.close -> result.json), that ``load()`` is called only once and
the policy stays resident across two episodes, and the persisted final state for each kind of fault.
"""
from __future__ import annotations

import dataclasses
import json
from functools import partial
from pathlib import Path

import pytest

from robomme_ood_eval import episode as E
from robomme_ood_eval import models
from robomme_ood_eval.policy import load_policy
from tests.unit_eval import fakes

SPEC_FIELDS = ("dataset", "task", "episode", "source_episode", "tier", "seed", "candidate", "spec_sha256", "key",
               "max_steps", "strict_cap", "attempt", "policy_seed", "out_dir")
PLAY_REQUIRED = ("status", "task_success", "steps", "error", "infra", "infra_reason")
RESULT_COUNTERS = ("task_success", "steps", "exec_steps", "reset_calls", "demo_frames", "decisions")


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    fakes.EVENTS.clear()
    fakes.FakeBuilder.instances.clear()
    E.clear_builders()
    monkeypatch.setattr(E, "BUILDER_FACTORY", fakes.FakeBuilder)
    monkeypatch.setitem(models.REGISTRY, "fake", ("tests.unit_eval.fakes", "FakePolicy"))
    yield
    E.clear_builders()


def run(policy, dataset="ood", task="VideoUnmask", ep=0, out=None, **kw):
    kw.setdefault("recorder_factory", fakes.FakeRecorder)
    kw.setdefault("render", False)
    return E.run_episode(policy, dataset, task, ep, out, **kw)


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ── Data structure contracts ─────────────────────────────────────────────────────────────


def test_episode_spec_fields_and_frozen():
    assert tuple(f.name for f in dataclasses.fields(E.EpisodeSpec)) == SPEC_FIELDS
    spec = E.EpisodeSpec(dataset="ood", task="T", episode=1, source_episode=None, tier="xhard2", seed=9, candidate=3,
                         spec_sha256="a" * 64, key="T_xhard2_9", max_steps=1800, strict_cap=True, attempt=1,
                         policy_seed=7, out_dir="/x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.seed = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        dataclasses.replace(spec, dataset="test")


def test_dataset_max_steps_and_invalid_dataset(tmp_path):
    assert E.DATASET_MAX_STEPS == {"hard-verify": 1300, "ood": 1800}
    p = load_policy("fake", 7)
    with pytest.raises(ValueError):
        E.run_episode(p, "test", "VideoUnmask", 0, tmp_path)
    assert p.calls["reset"] == 0


@pytest.mark.parametrize("dataset", ["ood", "hard-verify"])
def test_make_spec_from_resolve_identity(tmp_path, dataset):
    p = load_policy("fake", 7)
    spec, resolved = E.make_spec(p, dataset, "VideoUnmask", 1, tmp_path)
    assert spec.max_steps == E.DATASET_MAX_STEPS[dataset]
    assert spec.strict_cap is (dataset == "ood")
    assert spec.key == f"VideoUnmask_{spec.tier}_{spec.seed}" and spec.policy_seed == 7 and spec.attempt == 1
    assert spec.tier == resolved["tier"] and spec.seed == resolved["seed"]
    if dataset == "ood":
        assert (spec.candidate, spec.spec_sha256, spec.source_episode) == (11, f"{1:064x}", None)
        assert spec.ep_label == 1
    else:
        assert (spec.candidate, spec.spec_sha256, spec.source_episode) == (None, None, 4)
        assert spec.ep_label == 4  # hard-verify ep<N> uses the original official episode number
    assert Path(spec.out_dir) == tmp_path / "rollouts" / "fake" / dataset / "seed7" / "raw" / spec.raw_name
    assert spec.raw_name == f"VideoUnmask_ep{spec.ep_label}_{spec.tier}"


def test_builder_cached_per_task_dataset(tmp_path):
    p = load_policy("fake", 7)
    run(p, out=tmp_path, ep=0)
    run(p, out=tmp_path, ep=1)
    run(p, dataset="hard-verify", out=tmp_path, ep=0)
    keys = [(b.task, b.dataset) for b in fakes.FakeBuilder.instances]
    assert keys == [("VideoUnmask", "ood"), ("VideoUnmask", "hard-verify")]


def test_play_output_contract_required_fields():
    out = {k: None for k in PLAY_REQUIRED} | {"status": "success"}
    assert E.normalize_play_output(out)["status"] == "success"
    for k in PLAY_REQUIRED:
        bad = dict(out)
        bad.pop(k)
        with pytest.raises(E.PlayContractError):
            E.normalize_play_output(bad)
    with pytest.raises(E.PlayContractError):
        E.normalize_play_output(dict(out, status="weird"))
    with pytest.raises(E.PlayContractError):
        E.normalize_play_output(["not", "a", "dict"])


# ── Call order and residency ───────────────────────────────────────────────────────────


def test_call_order_reset_build_play_close(tmp_path):
    p = load_policy("fake", 7)
    fakes.EVENTS.clear()
    res = run(p, out=tmp_path)
    ev = fakes.EVENTS
    assert ev.index("policy.reset") < ev.index("recorder.init") < ev.index("builder.make_env") < ev.index("policy.play")
    assert ev.index("policy.play") < ev.index("env.close") < ev.index("recorder.close")
    assert "policy.load" not in ev and "policy.close" not in ev  # run_episode never calls load / close
    assert (Path(tmp_path) / "rollouts/fake/ood/seed7" / res.raw_dir / "result.json").is_file()


def test_load_once_two_episodes_resident(tmp_path):
    with load_policy("fake", 7) as p:
        r1 = run(p, out=tmp_path, ep=0)
        r2 = run(p, dataset="hard-verify", out=tmp_path, ep=0)
        assert p.calls == {"load": 1, "reset": 2, "play": 2, "close": 0}
    assert p.calls["close"] == 1
    assert fakes.EVENTS.count("policy.load") == 1 and fakes.EVENTS.count("policy.close") == 1
    assert (r1.status, r2.status) == ("success", "success")
    assert p.episodes_run == 2
    rows = [json.loads(x) for x in (tmp_path / "rollouts/fake/ood/seed7/results.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["key"] == r1.key


def test_result_fields_and_explicit_zero_counters(tmp_path):
    p = load_policy("fake", 7, behavior="fail_status")
    res = run(p, out=tmp_path)
    d = read_json(Path(tmp_path) / "rollouts/fake/ood/seed7" / res.raw_dir / "result.json")
    for k in SPEC_FIELDS[:-1]:
        assert k in d
    for k in RESULT_COUNTERS + ("cap_hit", "infra", "infra_reason", "error", "error_kind", "server_seed", "model",
                                "policy_label", "raw_dir", "video", "timing", "recorder_verify"):
        assert k in d, k
    assert d["status"] == "fail" and d["task_success"] == 0 and d["decisions"] == 0
    assert all(isinstance(d[k], int) for k in RESULT_COUNTERS)
    assert d["exec_steps"] == 4 and d["steps"] == 4 and d["reset_calls"] == 2 and d["demo_frames"] == 2
    assert d["pp_sid_use_index"] == 0  # model-specific fields are flattened to the top level
    assert d["policy_seed"] == 7 and d["server_seed"] is None and d["recorder_verify"] == "PASS"
    assert E.EpisodeResult.from_dict(d).to_dict() == d


def test_success_result(tmp_path):
    p = load_policy("fake", 7)
    res = run(p, out=tmp_path)
    assert (res.status, res.task_success, res.error_kind, res.infra) == ("success", 1, None, False)
    assert res.decisions == 2 and res.extra["pp_sid_use_index"] == 1


# ── Fault handling ─────────────────────────────────────────────────────────────────


def test_play_exception_recorded_and_next_episode_runs(tmp_path):
    p = load_policy("fake", 7, behavior="raise")
    r1 = run(p, out=tmp_path, ep=0)
    assert (r1.status, r1.error_kind, r1.infra) == ("fail", "error", False)
    assert "ValueError" in r1.error
    assert fakes.EVENTS.count("env.close") == 1 and fakes.EVENTS.count("recorder.close") == 1  # finally still tears down
    p.behavior = "success"
    r2 = run(p, out=tmp_path, ep=1)
    assert r2.status == "success" and p.calls["load"] == 1


def test_build_failure_is_infra_and_skips_play(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "BUILDER_FACTORY", partial(fakes.FakeBuilder, fail_build=True))
    p = load_policy("fake", 7)
    res = run(p, out=tmp_path)
    assert (res.status, res.error_kind, res.infra, res.infra_reason) == ("fail", "error", True, "env_build")
    assert p.calls["play"] == 0 and "policy.play" not in fakes.EVENTS
    assert "recorder.close" in fakes.EVENTS  # recorder is still torn down


def test_step_cap_raised_becomes_timeout(tmp_path, monkeypatch):
    monkeypatch.setitem(E.DATASET_MAX_STEPS, "ood", 5)
    monkeypatch.setattr(E, "BUILDER_FACTORY", partial(fakes.FakeBuilder, finish_at=None))
    p = load_policy("fake", 7, behavior="stepcap")
    res = run(p, out=tmp_path)
    assert (res.status, res.cap_hit, res.exec_steps, res.error_kind) == ("timeout", True, 5, None)
    assert fakes.FakeBuilder.instances[0].envs[0].n == 5  # step 6 never reached the env


def test_step_cap_swallowed_by_model_still_timeout(tmp_path, monkeypatch):
    monkeypatch.setitem(E.DATASET_MAX_STEPS, "ood", 3)
    monkeypatch.setattr(E, "BUILDER_FACTORY", partial(fakes.FakeBuilder, finish_at=None))
    p = load_policy("fake", 7, behavior="swallow_cap")
    res = run(p, out=tmp_path)
    assert res.status == "timeout" and res.cap_hit and res.extra["client_status"] == "fail"


def test_hard_verify_has_no_strict_cap(tmp_path, monkeypatch):
    monkeypatch.setitem(E.DATASET_MAX_STEPS, "hard-verify", 3)
    p = load_policy("fake", 7)
    res = run(p, dataset="hard-verify", out=tmp_path)
    assert res.status == "success" and not res.cap_hit and res.exec_steps == 4


def test_error_status_merged_into_fail(tmp_path):
    p = load_policy("fake", 7, behavior="return_error")
    res = run(p, out=tmp_path)
    assert (res.status, res.error_kind, res.error, res.task_success) == ("fail", "error", "IK failed", 0)


def test_bad_play_output_is_error(tmp_path):
    p = load_policy("fake", 7, behavior="bad_output")
    res = run(p, out=tmp_path)
    assert res.status == "fail" and res.error_kind == "error" and "PlayContractError" in res.error


def test_identity_mismatch_writes_and_stops(tmp_path, capsys):
    p = load_policy("fake", 7)
    want = {"tier": "xhard2", "seed": 1, "candidate": 10, "spec_sha256": f"{0:064x}"}
    with pytest.raises(E.IdentityMismatch):
        run(p, out=tmp_path, expect=want)
    assert "RUN_BLOCKED reason=identity" in capsys.readouterr().out
    assert p.calls["reset"] == 0 and "builder.make_env" not in fakes.EVENTS
    rows = (tmp_path / "rollouts/fake/ood/seed7/results.jsonl").read_text().splitlines()
    d = json.loads(rows[0])
    assert d["error"].startswith("IDENTITY_MISMATCH") and d["run_blocked"] is True and d["status"] == "fail"


def test_identity_match_passes(tmp_path):
    p = load_policy("fake", 7)
    want = {"tier": "xhard0", "seed": 500, "candidate": None, "spec_sha256": None, "source_episode": 3}
    res = run(p, dataset="hard-verify", out=tmp_path, expect=want)
    assert res.status == "success"


def test_reset_refusal_propagates_without_result(tmp_path):
    from robomme_ood_eval.policy import ServerDead

    p = load_policy("fake", 7)

    def dead(spec):
        raise ServerDead("server has exited")
    p.reset = dead
    with pytest.raises(ServerDead):
        run(p, out=tmp_path)
    assert not (tmp_path / "rollouts/fake/ood/seed7/results.jsonl").exists()
    assert "builder.make_env" not in fakes.EVENTS


def test_budget_exhausted_records_then_raises(tmp_path):
    from robomme_ood_eval.session import ResetBudgetExhausted

    class Ledger:
        def __init__(self, cap):
            self.cap, self.n = cap, 0

        def claim(self, what):
            if self.n >= self.cap:
                raise ResetBudgetExhausted(f"{what} over budget")
            self.n += 1

    p = load_policy("fake", 7)
    with pytest.raises(ResetBudgetExhausted):
        run(p, out=tmp_path, ledger=Ledger(1))  # build gets a slot, reset is over budget
    d = json.loads((tmp_path / "rollouts/fake/ood/seed7/results.jsonl").read_text().splitlines()[0])
    assert d["budget_exhausted"] is True and d["infra"] is False and d["status"] == "fail" and d["reset_calls"] == 1


def test_watchdog_wall_timeout_writes_and_exits(tmp_path, monkeypatch, capsys):
    exits = []
    monkeypatch.setattr(E, "HARD_EXIT", exits.append)
    p = load_policy("fake", 7, behavior="sleep", sleep_s=1.5, with_server=True)
    p.episodes_run = 1  # not the first episode, so no 600 s extension
    res = run(p, out=tmp_path, wall_s=0.3)
    assert exits == [75]
    out = capsys.readouterr().out
    assert "INFRA_TIMEOUT phase=episode_wall" in out and "SERVER_LEFT pid=4242 port=18080" in out
    assert (res.status, res.infra, res.infra_reason, res.error_kind) == ("fail", True, "episode_wall", "error")
    rows = (tmp_path / "rollouts/fake/ood/seed7/results.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["error"].startswith("INFRA_TIMEOUT")


def test_first_episode_wall_extension():
    p = load_policy("fake", 7)
    assert E.wall_limit(p, True) == E.FALLBACK_WALL_S + 600
    p.model = "smvla"
    assert E.wall_limit(p, False) == 900
    p.model = "perceptual-framesamp-modul"
    assert E.wall_limit(p, False) == 1200
