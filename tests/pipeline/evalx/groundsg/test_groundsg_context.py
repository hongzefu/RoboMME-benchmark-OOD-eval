"""GroundSG 官方定义摘取、变体隔离、互斥断言、整席上下文（1003 评估计划 1.3，子任务 S3）。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

import groundsg_fakes as F

OFFICIAL_ENV = {"IMAGE_MAX_TOKEN_NUM": "256", "VIDEO_MAX_TOKEN_NUM": "64", "FPS_MAX_FRAMES": "10"}
QWEN_ENV = {"USE_HF": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
OTHER_PREDICTORS = ("GeminiSubgoalPredictor", "MemERSubgoalPredictor", "NullSubgoalPredictor")
ALL_PREDICTORS = ("GeminiSubgoalPredictor", "QwenVLSubgoalPredictor", "MemERSubgoalPredictor", "OracleSubgoalPredictor",
                  "NullSubgoalPredictor")


@pytest.fixture
def clean_env(monkeypatch):
    """被测代码直接写 os.environ；先经 monkeypatch 登记这些键，用例结束后恢复原值。"""
    for k in (*OFFICIAL_ENV, *QWEN_ENV):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_extract_defs_takes_only_named_source(tmp_path):
    od = F.official_defs()
    src = tmp_path / "m.py"
    src.write_text("import does_not_exist_anywhere\nX = 3\nY: int = 4\nos.environ['S3_PROBE'] = '1'\n"
                   "def f(a):\n    return a + X\n@dataclasses.dataclass\nclass C:\n    v: int = 1\n", encoding="utf-8")
    ns = od.extract_defs(src, ["X", "f", "C"])
    assert ns["f"](1) == 4 and ns["C"]().v == 1 and "Y" not in ns
    assert "S3_PROBE" not in os.environ  # 模块级副作用与 import 行不执行
    assert ns["__source_sha256__"] == hashlib.sha256(src.read_bytes()).hexdigest()
    with pytest.raises(KeyError):
        od.extract_defs(src, ["X", "missing_name"])


def test_official_sources_hashes_recorded(clean_env):
    F.print_official_sha()
    od = F.official_defs()
    defs = od.load_groundsg(F.QWENVL, with_env_runner=True, env_runner_extra={"BenchmarkEnvBuilder": object},
                            ws_module=od.ws_shim(lambda h, p: None))
    want = F.official_sha256()
    assert defs["sha256"] == want
    for k, v in OFFICIAL_ENV.items():
        assert os.environ[k] == v


def test_oracle_variant_never_loads_qwen(tmp_path, clean_env):
    side = F.NewSide(F.ORACLE, 60, tmp_path, F.World())
    defs = side.ctx["defs"]
    assert defs["qwen"] is None and "subgoal_prediction/qwenvl/api.py" not in defs["sha256"]
    pred_names = set(defs["predictor"])
    assert "OracleSubgoalPredictor" in pred_names and "QwenVLSubgoalPredictor" not in pred_names
    assert "Qwen3VLModel" not in pred_names and not any(n in pred_names for n in OTHER_PREDICTORS)
    assert type(side.ctx["predictor"]).__name__ == "OracleSubgoalPredictor"
    assert side.swift.engines == [] and "swift" not in sys.modules and "google.generativeai" not in sys.modules
    for k in QWEN_ENV:
        assert k not in os.environ
    args = side.ctx["args"]
    assert (args.use_oracle, args.use_qwenvl, args.use_gemini, args.use_memer) == (True, False, False, False)
    assert args.subgoal_type == "grounded_subgoal" and args.max_steps == 60 and args.obs_horizon == 16


def test_qwenvl_variant_loads_only_qwen_classes(tmp_path, clean_env):
    side = F.NewSide(F.QWENVL, 60, tmp_path, F.World())
    pred_names = set(side.ctx["defs"]["predictor"])
    assert "QwenVLSubgoalPredictor" in pred_names and "OracleSubgoalPredictor" not in pred_names
    assert not any(n in pred_names for n in OTHER_PREDICTORS)
    assert type(side.ctx["predictor"]).__name__ == "QwenVLSubgoalPredictor"
    # 引擎参数取官方原文：底座、adapter、flash_attention_2，构造一次、不读权重
    assert side.swift.engines == [{"model_id_or_path": "Qwen/Qwen3-VL-4B-Instruct", "adapters": [F.ADAPTER],
                                   "attn_impl": "flash_attention_2"}]
    for k, v in {**OFFICIAL_ENV, **QWEN_ENV}.items():
        assert os.environ[k] == v
    res = side.run(F.identity())
    assert res["status"] == "success"
    assert {json.dumps(r["config"], sort_keys=True) for r in side.swift.requests} == {
        json.dumps({"max_tokens": 128, "temperature": 0}, sort_keys=True)}


def test_variants_are_mutually_exclusive(tmp_path, clean_env):
    od = F.official_defs()
    defs = od.load_groundsg(F.ORACLE, with_env_runner=False, ws_module=od.ws_shim(lambda h, p: None))
    Args = defs["Args"]
    for kw in ({"use_oracle": True, "use_qwenvl": True}, {"use_oracle": False, "use_qwenvl": False},
               {"use_oracle": True, "use_gemini": True}, {"use_qwenvl": True, "use_memer": True},
               {"use_oracle": True, "use_memer": True}, {"use_memer": True, "use_gemini": True},
               {"use_oracle": True, "use_qwenvl": True, "use_memer": True}):
        args = Args(subgoal_type="grounded_subgoal", model_seed=7, **kw)
        with pytest.raises(AssertionError):
            od.build_predictor(defs, args, tmp_path)
    # adapter 误配（接口冻结说明 2.3：QwenVL 只给 QwenVL adapter，MemER 只给 MemER adapter，Oracle 都不给）
    bad_pairs = [(F.QWENVL, None, None), (F.QWENVL, F.ADAPTER, F.MEMER_ADAPTER), (F.QWENVL, None, F.MEMER_ADAPTER),
                 (F.MEMER, None, None), (F.MEMER, F.ADAPTER, None), (F.MEMER, F.ADAPTER, F.MEMER_ADAPTER),
                 (F.ORACLE, F.ADAPTER, None), (F.ORACLE, None, F.MEMER_ADAPTER)]
    for variant, qa, ma in bad_pairs:
        with pytest.raises(ValueError):
            od.make_args(defs, variant=variant, host="h", port=1, max_steps=10, model_seed=7, adapter_path=qa,
                         memer_adapter_path=ma)
    # 模型种子必给：缺、负数、bool、非整数一律拒
    for seed in (None, -1, True, "x", 1.5):
        with pytest.raises(ValueError):
            od.make_args(defs, variant=F.ORACLE, host="h", port=1, max_steps=10, model_seed=seed)
    with pytest.raises(TypeError):
        od.make_args(defs, variant=F.ORACLE, host="h", port=1, max_steps=10)
    mc = F.groundsg_client()
    for bad in (None, "ground-sg-gemini"):
        with pytest.raises(ValueError):
            mc.make_policy_context(dict(F.seat_info(F.ORACLE, 60, tmp_path), groundsg_variant=bad))
    for bad_seed in (None, -3, "abc"):
        with pytest.raises(ValueError, match="RUN_BLOCKED reason=policy_seed"):
            mc.make_policy_context(dict(F.seat_info(F.ORACLE, 60, tmp_path), policy_seed=bad_seed))
    # MemER 变体缺 memer_adapter_path、或误带 QwenVL adapter：在加载任何模型之前就拒
    swift = F.FakeSwift()
    for extra in ({"memer_adapter_path": None}, {"qwenvl_groundSG_adapter_path": F.ADAPTER}):
        with pytest.raises(ValueError):
            mc.make_policy_context(dict(F.seat_info(F.MEMER, 60, tmp_path), **extra), qwen_extra=swift.names)
    assert swift.engines == []


def test_run_episode_rejects_mismatched_context(tmp_path, clean_env):
    side = F.NewSide(F.ORACLE, 60, tmp_path, F.World())
    ec = F.env_session()
    for conn in ({"policy_context": {}, "groundsg_variant": F.ORACLE, "max_steps": 60},
                 {"policy_context": side.ctx, "groundsg_variant": F.QWENVL, "max_steps": 60},
                 {"policy_context": side.ctx, "groundsg_variant": F.ORACLE, "max_steps": 1300}):
        with pytest.raises((RuntimeError, ValueError)):
            side.mc.run_episode(object(), F.identity(), conn, ec.NullRecorder())


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

    from tests.pipeline.eval import eval_fakes as EF

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
