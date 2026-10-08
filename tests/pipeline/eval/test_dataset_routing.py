"""数据集路由（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.2）：ood 与 hard-verify 两个数据集在清单、
席位客户端、汇总三处不串，默认 V9 行为不变；hard0 清单 192 行；hard-verify 结果行执行字段齐全；策略上下文与
conn_info 约定。

期望值一律手写：ood 为 ``--max-steps 1600 --strict-cap``，hard-verify 为 ``--max-steps 1300``、不带 ``--strict-cap``；
hard0 清单 16 任务 × 12 局 = 192。groundsg／pp 的策略模块由后续子任务提供，这里用替身。
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import eval_fakes as F

HARD0_TASK = "PickXtimes"


# ---------------------------------------------------------------- hard0 清单


def test_hard_verify_manifest_192_rows_paired_shards(tmp_path, capsys):
    """``--mode hard0 --pair-shards``：真实 builder（dataset="hard-verify"）枚举 16 任务 × 12 局 = 192 行，全为 xhard0，
    行键同 SHARD_ROW_KEYS、spec_sha256／candidate 为 null、key=<task>_xhard0_<seed>；每行经真实 builder 解析回同一身份，
    且通过客户端 hard-verify 身份核对；原侧与新侧共用同一份 shard-NN.json。"""
    em, ec = F.eval_manifest(), F.env_client()
    out = tmp_path / "hard0"
    capsys.readouterr()
    rc = em.main(["--mode", "hard0", "--out-dir", str(out), "--pair-shards"])
    lines = capsys.readouterr().out.splitlines()
    assert rc == 0
    (line,) = [x for x in lines if x.startswith("EVAL_SHARDS=")]
    v = F.verdict(lines, "EVAL_SHARDS")
    assert (v[""], v["mode"], v["total"], v["xhard0"], v["per_task"]) == ("PASS", "hard0", "192", "192", "12")
    assert any(x.startswith("PAIR_SHARDS sides=orig,new") for x in lines)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "hard0" and manifest["dataset"] == "hard-verify" and manifest["total"] == 192
    assert manifest["paired_sides"] == ["orig", "new"] and manifest["pair_shards"] is True
    assert len(manifest["cells"]) == 16 and set(manifest["cells"].values()) == {12}
    shards = sorted(out.glob("shard-*.json"))
    assert len(shards) == em.DEFAULT_SHARDS
    rows = [r for p in shards for r in json.loads(p.read_text(encoding="utf-8"))]
    assert len(rows) == len({r["key"] for r in rows}) == 192
    builders = {}
    for r in rows:
        assert set(r) == set(em.SHARD_ROW_KEYS)
        assert r["tier"] == "xhard0" and r["candidate"] is None and r["spec_sha256"] is None
        assert r["key"] == f"{r['task']}_xhard0_{r['seed']}"
        assert ec.validate_v8_identity(r, "hard-verify") is None
        b = builders.setdefault(r["task"], F.real_builder(r["task"], dataset="hard-verify"))
        assert ec.check_identity(b.resolve_identity(r["builder_episode"]), r, dataset="hard-verify") is None
    # 分片内按 (task, builder_episode) 正序（两侧调用序列相同）
    for p in shards:
        part = json.loads(p.read_text(encoding="utf-8"))
        assert part == sorted(part, key=lambda r: (r["task"], r["builder_episode"]))
    print(line)


def test_hard_verify_per_task_and_rejections():
    em = F.eval_manifest()
    manifest, parts = em.build_hard0(3, 4)
    assert manifest["total"] == 48 and sum(len(p) for p in parts) == 48
    assert {r["builder_episode"] for r in manifest["rows"]} == {0, 1, 2}
    for bad in (0, 13):
        with pytest.raises(em.ManifestError):
            em.build_hard0(bad, 4)

    class _Wrong:  # 解析出的不是 xhard0（把 ood 的 builder 当成 hard-verify 用）
        def __init__(self, task):
            self.real = F.real_builder(task)

        def get_episode_num(self):
            return 12

        def resolve_identity(self, ep):
            return dict(self.real.resolve_identity(ep), tier="xhard1", source_episode=None)

    with pytest.raises(em.ManifestError) as ei:
        em.build_hard0(1, 2, builder_factory=_Wrong)
    assert ei.value.stage == "hard0"


# ---------------------------------------------------------------- hard-verify 结果行


@pytest.mark.parametrize("policy", ("perceptual-framesamp-modul", "smvla"))
def test_hard_verify_result_row_has_exec_fields(tmp_path, monkeypatch, policy):
    """hard-verify（不带 --strict-cap）结果行必写 exec_steps、cap_hit、demo_frames、reset_calls，另带 dataset、
    policy_variant、strict_cap、max_steps＝effective_max_steps＝1300；录像目录 <key>.a<attempt>。"""
    ident = F.hard0_identity(HARD0_TASK, 0)
    world = F.World({(ident["task"], ident["builder_episode"]): [F.Plan(success_at=5)]})
    runner = F.make_runner(tmp_path, policy, F.policy_module(policy, monkeypatch, F.FakePolicyServer()), world,
                           dataset="hard-verify")
    assert F.run_rows(runner, [ident]) == 0
    (row,) = F.read_jsonl(tmp_path / "s00" / policy / "results.jsonl")
    for f in ("exec_steps", "cap_hit", "demo_frames", "reset_calls"):
        assert f in row and row[f] is not None, f
    assert row["exec_steps"] == 5 and row["cap_hit"] is False
    assert row["demo_frames"] == F.N_RESET_FRAMES - 1 and row["reset_calls"] == 2  # build 与 reset 各领一次
    assert row["dataset"] == "hard-verify" and row["strict_cap"] is False and row["policy_variant"] is None
    assert row["max_steps"] == row["effective_max_steps"] == 1300
    assert row["tier"] == "xhard0" and row["source_episode"] == ident["source_episode"]
    assert Path(row["rec_dir"]).name == f"{ident['key']}.a1"


def test_hard_verify_without_strict_cap_lets_official_loop_run_step_1301(tmp_path, monkeypatch):
    """hard-verify 不带 --strict-cap：环境侧不截断，perceptual-framesamp-modul 官方循环 count>max_steps 才停，真实执行第 1301 步，
    记 timeout、cap_hit=false（StepCapReached 只在 --strict-cap 时生效）。"""
    ident = F.hard0_identity(HARD0_TASK, 0)
    world = F.World({(ident["task"], ident["builder_episode"]): [F.Plan()]})
    runner = F.make_runner(tmp_path, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                           dataset="hard-verify")
    assert F.run_rows(runner, [ident]) == 0
    (row,) = F.read_jsonl(tmp_path / "s00" / "perceptual-framesamp-modul" / "results.jsonl")
    (env,) = world.envs
    assert env.n == 1301 and row["exec_steps"] == 1301
    assert row["status"] == "timeout" and row["cap_hit"] is False and row["infra"] is False


# ---------------------------------------------------------------- 两个数据集不串、默认不变


def test_dataset_routing(tmp_path, monkeypatch, capsys):
    ec, em = F.env_client(), F.eval_manifest()
    crossed = default_changed = 0
    h0 = F.hard0_identity(HARD0_TASK, 0)
    task, tier = F.v9_cells_sorted()[0]
    v9 = F.packaged_identity(task, tier, 0)

    # 1. 正向：各自的数据集 builder、步数上限与结果行
    for ds, ident, cap in (("hard-verify", h0, 1300), ("ood", v9, 1600)):
        world = F.World()
        runner = F.make_runner(tmp_path / ds, "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                               dataset=ds)
        assert F.run_rows(runner, [ident]) == 0
        (b,) = world.builders
        crossed += (b.dataset != ds) + (b.max_steps != cap)
        (row,) = F.read_jsonl(tmp_path / ds / "s00" / "perceptual-framesamp-modul" / "results.jsonl")
        crossed += (row["dataset"] != ds) + (row["max_steps"] != cap) + (row["status"] != "success")

    # 2. 反向串喂：席位客户端在 run_one 与 load_identities 两层都拦（运行阻塞 3），不建环境
    for ds, ident in (("ood", h0), ("hard-verify", v9)):
        world = F.World()
        runner = F.make_runner(tmp_path / f"x-{ds}", "perceptual-framesamp-modul", F.framesamp_modul_policy(monkeypatch, F.FakePolicyServer()), world,
                               dataset=ds)
        rc = F.run_rows(runner, [ident])
        crossed += not (rc == 3 and world.envs == [] and world.make_calls == [])
        p = tmp_path / f"shard-{ds}.json"
        p.write_text(json.dumps([ident]), encoding="utf-8")
        args = F.seat_args(tmp_path / f"o-{ds}", "perceptual-framesamp-modul", ledger=tmp_path / f"l-{ds}.jsonl", identities=str(p), dataset=ds)
        try:
            ec.load_identities(args)
            crossed += 1
        except SystemExit as e:
            crossed += e.code != 3

    # 3. 汇总：hard0 运行根按自己的数据集报 PASS；按 ood 报则串行被抓（dataset_crossed=1）
    manifest = F.write_manifest(tmp_path / "m" / "manifest.json", [h0], dataset="hard-verify")
    _, lines, rep = F.run_report(capsys, manifest, tmp_path / "hard-verify", ["perceptual-framesamp-modul"], tmp_path / "r0",
                                 "--dataset", "hard-verify", "--expect-total", "1")
    crossed += F.verdict(lines, "EVAL_COVERAGE")[""] != "PASS"
    crossed += F.verdict(lines, "EVAL_REPORT")[""] != "PASS"
    _, lines, rep = F.run_report(capsys, manifest, tmp_path / "hard-verify", ["perceptual-framesamp-modul"], tmp_path / "r1",
                                 "--dataset", "ood", "--expect-total", "1")
    crossed += rep["per_policy"]["perceptual-framesamp-modul"]["dataset_crossed"] != 1
    crossed += F.verdict(lines, "EVAL_REPORT")[""] != "FAIL"

    # 4. 默认不变：客户端两项必填无默认；清单默认 v9-new（缺 --exclude-evaluated 即参数错误）；汇总不带 --dataset
    #    仍出 V8 两行；EnvSession 默认 ood
    base = ["run", "--policy", "perceptual-framesamp-modul", "--identities", "x", "--cond", "c", "--seat", "s", "--port", "1", "--out", "o"]
    for missing in (["--max-steps", "1600"], ["--dataset", "ood"]):
        try:
            ec.build_parser().parse_args(base + missing)
            default_changed += 1
        except SystemExit as e:
            default_changed += e.code != 2
    try:
        em.main(["--out-dir", str(tmp_path / "d"), "--identities", "i", "--delivery", "d"])
        default_changed += 1
    except SystemExit as e:
        default_changed += e.code != 2
    m9 = F.write_manifest(tmp_path / "m9" / "manifest.json", [v9])
    _, lines, _ = F.run_report(capsys, m9, tmp_path / "ood", ["perceptual-framesamp-modul"], tmp_path / "r9", "--expect-total", "1")
    default_changed += not any(x.startswith("V8_EVAL_COVERAGE=PASS") for x in lines)
    default_changed += ec.EnvSession("T", 0).dataset != "ood"

    assert crossed == 0 and default_changed == 0
    print(f"DATASET_ROUTING=PASS crossed={crossed} default_changed={default_changed}")


# ---------------------------------------------------------------- 策略上下文与 conn_info（S3～S5 的接口约定）


def test_policy_context_built_once_and_conn_info(tmp_path):
    """groundsg 替身：make_policy_context 整席只调一次、每局拿到同一对象；conn_info 带 dataset、groundsg_variant、
    qwenvl_groundSG_adapter_path、trace_dir、policy_context；结果行 policy_variant 取 --groundsg-variant；结果行与录制器
    元数据都不含 policy_context；close() 调一次 close_policy_context。"""
    calls = {"make": [], "close": [], "conn": []}
    ctx = object()

    def make_policy_context(seat_info):
        calls["make"].append(dict(seat_info))
        return ctx

    def run_episode(session, identity, conn_info, recorder):
        calls["conn"].append(dict(conn_info))
        session.reset()
        for _ in range(3):
            *_, info = session.step([0.0] * 8)
        return {"status": info["status"], "steps": session.steps, "infra": False, "error": None}

    mod = types.SimpleNamespace(make_policy_context=make_policy_context, run_episode=run_episode,
                                close_policy_context=lambda c: calls["close"].append(c))
    a, b = F.hard0_identity(HARD0_TASK, 0), F.hard0_identity(HARD0_TASK, 1)
    world = F.World()
    trace_root = tmp_path / "trace"
    adapter = str(tmp_path / "adapter")
    runner = F.make_runner(tmp_path, "groundsg", mod, world, dataset="hard-verify",
                           policy_dir="groundsg-ground-sg-qwenvl", groundsg_variant="ground-sg-qwenvl",
                           qwenvl_groundsg_adapter=adapter, trace_root=str(trace_root))
    assert F.run_rows(runner, [a, b]) == 0
    runner.close()
    assert len(calls["make"]) == 1 and calls["close"] == [ctx]
    seat = calls["make"][0]
    assert (seat["dataset"], seat["max_steps"], seat["strict_cap"], seat["groundsg_variant"]) == \
        ("hard-verify", 1300, False, "ground-sg-qwenvl")
    assert seat["qwenvl_groundSG_adapter_path"] == adapter
    for ident, conn in zip((a, b), calls["conn"]):
        assert conn["policy_context"] is ctx
        assert conn["dataset"] == "hard-verify" and conn["max_steps"] == 1300 and conn["strict_cap"] is False
        assert conn["groundsg_variant"] == "ground-sg-qwenvl" and conn["qwenvl_groundSG_adapter_path"] == adapter
        assert conn["episode_tag"] == f"{ident['key']}.a1"
        assert conn["trace_dir"] == str(trace_root / f"{ident['key']}.a1")
    rows = F.read_jsonl(tmp_path / "s00" / "groundsg-ground-sg-qwenvl" / "results.jsonl")
    assert [r["policy_variant"] for r in rows] == ["ground-sg-qwenvl"] * 2
    assert all(r["policy"] == "groundsg" for r in rows)
    assert all("policy_context" not in r for r in rows)
    assert all("policy_context" not in rec.meta for rec in world.recorders)


def test_policy_context_defaults_to_shared_dict(tmp_path, monkeypatch):
    """策略模块没有 make_policy_context：上下文为整席共享的空 dict（策略可往里缓存）；未给 --trace-root 时 trace_dir 为 null。"""
    seen = []

    def run_episode(session, identity, conn_info, recorder):
        seen.append(conn_info)
        conn_info["policy_context"].setdefault("n", 0)
        conn_info["policy_context"]["n"] += 1
        session.reset()
        *_, info = session.step([0.0] * 8)
        return {"status": "fail", "steps": 1, "infra": False, "error": None}

    a, b = F.hard0_identity(HARD0_TASK, 0), F.hard0_identity(HARD0_TASK, 1)
    runner = F.make_runner(tmp_path, "pp", types.SimpleNamespace(run_episode=run_episode), F.World(),
                           dataset="hard-verify")
    assert F.run_rows(runner, [a, b]) == 0
    assert seen[0]["policy_context"] is seen[1]["policy_context"] and seen[1]["policy_context"]["n"] == 2
    assert seen[0]["trace_dir"] is None and seen[0]["groundsg_variant"] is None
