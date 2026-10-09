"""GroundSG 整席上下文：GL 席位（``SeatRunner``）跑两局只建一次策略上下文（1003 评估计划 1.3，子任务 S3）。

由 ``test_groundsg_context.py`` 拆出（席位层 ``SeatRunner`` 只在 dev 侧）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import groundsg_fakes_dev as F

OFFICIAL_ENV = {"IMAGE_MAX_TOKEN_NUM": "256", "VIDEO_MAX_TOKEN_NUM": "64", "FPS_MAX_FRAMES": "10"}
QWEN_ENV = {"USE_HF": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


@pytest.fixture
def clean_env(monkeypatch):
    """被测代码直接写 os.environ；先经 monkeypatch 登记这些键，用例结束后恢复原值。"""
    for k in (*OFFICIAL_ENV, *QWEN_ENV):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


class _HybridBuilder:
    """真实 robomme_hard builder（hard-verify）只做身份解析；环境换成假环境（按解析出的 source_episode）。"""

    def __init__(self, task, dataset, max_steps, world):
        from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

        self.task, self.world = task, world
        self.real = BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="joint_angle", max_steps=max_steps)

    def resolve_identity(self, ep):
        return self.real.resolve_identity(ep)

    def make_env_for_episode(self, ep, max_steps=None):
        return self.world.new_env(self.task, int(self.real.resolve_identity(ep)["source_episode"]))


def test_policy_context_built_once_per_seat(tmp_path, clean_env):
    """拆仓后：GL 席位（``SeatRunner``，常驻 Policy）跑两局，真实 ``GroundSGPolicy.load`` 只调一次
    ``make_policy_context``（Qwen 引擎只构造一次），两局用同一上下文，席位收尾 ``close`` 一次（停服务端一次、释放上下文）。
    服务端只换成假进程（``load_mme_vla_server`` 替身：不起进程，记下种子元数据）；旧 ``policy_mod``／``make_policy_context
    (seat_info)`` 注入点已不存在，改在 ``load_policy`` 工厂处注入。"""
    from robomme_ood_eval import episode as E
    from robomme_ood_eval.models import framesamp_modul as fm
    from robomme_ood_eval.models import groundsg as gmod
    from robomme_ood_eval.session import NullRecorder

    from tests.pipeline.eval import eval_fakes_dev as EF

    ec = F.env_client()
    world = F.World(default=F.Plan(success_at=12))
    server, swift = F.FakeServer(), F.FakeSwift()
    calls = {"make": 0, "factory": 0, "ctx": [], "stops": 0}

    class _Srv:
        metadata = {"policy_seed": F.POLICY_SEED}

        def check(self):
            pass

        def stop(self, **_k):
            calls["stops"] += 1
            return "stopped"

        def left_line(self):
            return ""

    def fake_load_server(policy, model):
        policy.port = 18120
        policy.server = _Srv()
        return Path("/ck/groundsg")

    orig_make, orig_run = gmod.make_policy_context, gmod.run_episode

    def make(seat_info, **kw):
        calls["make"] += 1
        assert seat_info["policy_seed"] == F.POLICY_SEED and seat_info["groundsg_variant"] == F.QWENVL
        return orig_make(seat_info, **kw)

    def run_episode(session, identity, conn_info, recorder):
        calls["ctx"].append(id(conn_info["policy_context"]))
        return orig_run(session, identity, conn_info, recorder)

    clean_env.setattr(fm, "load_mme_vla_server", fake_load_server)
    clean_env.setattr(gmod, "make_policy_context", make)
    clean_env.setattr(gmod, "run_episode", run_episode)

    def factory(model, seed, **cfg):
        calls["factory"] += 1
        assert (model, seed, cfg) == ("groundsg", F.POLICY_SEED, {"groundsg_variant": F.QWENVL,
                                                                   "qwenvl_groundsg_adapter": F.ADAPTER})
        p = gmod.GroundSGPolicy(seed, **cfg, preflight=False, warmup=False, work_dir=str(tmp_path / "work"),
                                client_factory=lambda h, pt, ep: F.FakeClient(server), qwen_extra=swift.names)
        p.load()
        return p

    args = EF.seat_args(tmp_path / "out", "groundsg", seat="00", policy_seed=F.POLICY_SEED, reset_budget=10,
                        infra_retries=0, groundsg_variant=F.QWENVL, qwenvl_groundsg_adapter=F.ADAPTER)
    prev = E.BUILDER_FACTORY
    E.clear_builders()
    E.BUILDER_FACTORY = lambda task, ds, ms: _HybridBuilder(task, ds, ms, world)
    try:
        runner = ec.SeatRunner(args, policy_factory=factory, shared=None, hard_exit=lambda code: None,
                               episode_kwargs={"recorder_factory": lambda raw, meta: NullRecorder(), "render": False})
        rows = []
        for ep in (0, 1):
            ident = E.builder_for("PickXtimes", "hard-verify").resolve_identity(ep)
            rows.append({"dataset": "hard-verify", "task": "PickXtimes", "tier": ident["tier"], "seed": ident["seed"],
                         "candidate": None, "builder_episode": ep, "source_episode": ident["source_episode"],
                         "spec_sha256": None, "key": f"PickXtimes_xhard0_{ident['seed']}"})
        assert runner.run(rows) == 0
        runner.close()
    finally:
        E.BUILDER_FACTORY = prev
        E.clear_builders()
    assert calls["factory"] == 1 and calls["make"] == 1 and calls["stops"] == 1
    assert len(calls["ctx"]) == 2 and len(set(calls["ctx"])) == 1
    assert len(swift.engines) == 1
    assert runner.policy.ctx is None and runner.policy.evaluators == {}
    got = EF.read_jsonl(runner.results_path)
    assert [(g["status"], g["exec_steps"], g["dataset"]) for g in got] == [("success", 12, "hard-verify")] * 2
    for g in got:
        res = json.loads(Path(g["result"]).read_text())
        raw = Path(g["result"]).parent
        assert res["policy_variant"] == F.QWENVL and res["side"] == "new" and res["dataset"] == "hard-verify"
        assert res["trace_path"] == str(raw / "trace.jsonl") and (raw / "trace.jsonl").is_file()
        assert not (raw / "qwen-tmp").exists()
    assert [w.ep for w in world.envs] == [r["source_episode"] for r in rows]
