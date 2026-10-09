"""数据集路由（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.2）：ood 与 hard-verify 两个数据集在清单、
席位客户端、汇总三处不串，默认 V9 行为不变；hard0 清单 192 行；hard-verify 结果行执行字段齐全；策略上下文与
conn_info 约定。

期望值一律手写：ood 为 1800 步严格截断，hard-verify 为 1300 步、不截断（拆仓后步数上限只由数据集决定）；
hard0 清单 16 任务 × 12 局 = 192。席位用常驻假模型（``eval_fakes.fake_seat_policy``）。
"""
from __future__ import annotations

import json
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
        assert F.episode_mod().check_identity(b.resolve_identity(r["builder_episode"]), r, dataset="hard-verify") is None
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


# ---------------------------------------------------------------- hard-verify 结果行（新产物树 result.json）

LABEL = "perceptual-framesamp-modul"


def _result_of(stage: Path, dataset: str, ident: dict) -> dict:
    acc = json.loads((F.queue_dir(stage, LABEL) / "accepted" / f"{dataset}__{ident['key']}.json").read_text())
    return json.loads(Path(acc["result"]).read_text())


def test_hard_verify_result_has_exec_fields(tmp_path):
    """hard-verify（不截断）result.json 必写 exec_steps、cap_hit、demo_frames、reset_calls，另带 dataset、strict_cap、
    max_steps＝1300；局目录 raw/<Task>_ep<官方原号>_xhard0/。"""
    ident = F.hard0_identity(HARD0_TASK, 0)
    world = F.World({(ident["task"], ident["builder_episode"]): [F.Plan(success_at=5)]})
    stage = tmp_path / "stage"
    assert F.run_rows(F.make_runner(stage, LABEL, F.fake_seat_policy(), world), [ident]) == 0
    row = _result_of(stage, "hard-verify", ident)
    for f in ("exec_steps", "cap_hit", "demo_frames", "reset_calls"):
        assert f in row and row[f] is not None, f
    assert row["exec_steps"] == 5 and row["cap_hit"] is False
    assert row["demo_frames"] == F.N_RESET_FRAMES - 1 and row["reset_calls"] == 2  # build 与 reset 各领一次
    assert row["dataset"] == "hard-verify" and row["strict_cap"] is False and row["max_steps"] == 1300
    assert row["tier"] == "xhard0" and row["source_episode"] == ident["source_episode"]
    assert row["raw_dir"] == f"raw/{HARD0_TASK}_ep{ident['source_episode']}_xhard0"


def test_hard_verify_without_strict_cap_lets_official_loop_run_step_1301(tmp_path):
    """hard-verify 不截断：环境侧不拦，官方循环 count>max_steps 才停，真实执行第 1301 步，记 timeout、cap_hit=false。"""
    ident = F.hard0_identity(HARD0_TASK, 0)
    world = F.World({(ident["task"], ident["builder_episode"]): [F.Plan()]})
    stage = tmp_path / "stage"
    assert F.run_rows(F.make_runner(stage, LABEL, F.fake_seat_policy(), world), [ident]) == 0
    row = _result_of(stage, "hard-verify", ident)
    (env,) = world.envs
    assert env.n == 1301 and row["exec_steps"] == 1301
    assert row["status"] == "timeout" and row["cap_hit"] is False and row["infra"] is False


# ---------------------------------------------------------------- 两个数据集不串、默认不变


def test_dataset_routing(tmp_path, capsys):
    ec, E = F.env_client(), F.episode_mod()
    crossed = default_changed = 0
    h0 = F.hard0_identity(HARD0_TASK, 0)
    task, tier = F.v9_cells_sorted()[0]
    v9 = F.packaged_identity(task, tier, 0)

    # 1. 正向：同一个常驻 Policy 先后跑两个数据集，各自的 builder、步数上限与结果
    world = F.World()
    stage = tmp_path / "both"
    assert F.run_rows(F.make_runner(stage, LABEL, F.fake_seat_policy(), world), [h0, v9]) == 0
    got = {(b.dataset, b.max_steps) for b in world.builders}
    crossed += got != {("hard-verify", 1300), ("ood", 1800)}
    for ds, ident, cap in (("hard-verify", h0, 1300), ("ood", v9, 1800)):
        row = _result_of(stage, ds, ident)
        crossed += (row["dataset"] != ds) + (row["max_steps"] != cap) + (row["status"] != "success")

    # 2. 反向串喂：身份行标错数据集 → 清单层拦（RUN_BLOCKED reason=identities，退出 3）；直接喂给席位 → 身份核对拦
    #    （run_episode 的 IDENTITY_MISMATCH，退出 3），都不建环境
    for wrong_ds, ident in (("ood", h0), ("hard-verify", v9)):
        bad = dict(ident, dataset=wrong_ds)
        p = tmp_path / f"ids-{wrong_ds}.jsonl"
        p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        args = F.seat_args(tmp_path / f"o-{wrong_ds}", LABEL, identities=str(p))
        try:
            ec.load_identities(args)
            crossed += 1
        except ec.SeatStop as e:
            crossed += e.code != 3
        world = F.World()
        if wrong_ds == "ood":  # hard-verify 身份按 ood 解析：tier 对不上
            rc = F.run_rows(F.make_runner(tmp_path / f"x-{wrong_ds}", LABEL, F.fake_seat_policy(), world), [bad])
            crossed += not (rc == 3 and world.envs == [] and world.make_calls == [])

    # 3. 默认不变：步数上限只由数据集决定（hard-verify 1300、ood 1800 严格截断），席位不再接受 --max-steps；
    #    EnvSession 默认 ood
    default_changed += dict(E.DATASET_MAX_STEPS) != {"hard-verify": 1300, "ood": 1800}
    try:
        ec.build_parser().parse_args(["run", "--policy", "dummy", "--identities", "x", "--out", "o",
                                      "--max-steps", "1600"])
        default_changed += 1
    except SystemExit as e:
        default_changed += e.code != 2
    from robomme_ood_eval.session import EnvSession

    default_changed += EnvSession("T", 0).dataset != "ood"
    assert crossed == 0 and default_changed == 0
    print(f"DATASET_ROUTING=PASS crossed={crossed} default_changed={default_changed}")
