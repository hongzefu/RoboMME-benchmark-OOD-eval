"""C13 身份契约：评估步数上限的取法、执行身份行的结构核对（两个数据集）、三处 key 函数一致。

- 拆仓后步数上限只由数据集决定（已定口径第 3 条）：经真实 ``SeatRunner``（常驻假模型）核对 builder 构造参数、
  ``EpisodeSpec.max_steps``／``strict_cap``、``EnvSession.step_cap`` 与 ``result.json``；``make_env_for_episode`` 不逐局传步数。
  期望值手写：ood 为 1800 且严格截断，hard-verify 为 1300、不截断（step_cap 为空）。
- 身份核对（``episode.check_identity``）按数据集：ood 逐键严格核 tier／seed／candidate／spec_sha256；hard-verify 要求 xhard0、
  candidate 与 spec_sha256 为 null、source_episode 为整数且与 builder 解析结果相等；不符即运行阻塞（退出码 3）。
- 三处 key 函数（``seat.v8_key／key_of``、``eval_manifest.v8_key``、``eval_report.key_of``）对同一身份给出同一个键，
  且键按契约手写为 ``<task>_<tier>_<seed>``；同 seed 不同档不碰撞。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import eval_fakes as F

#: (数据集, 手写步数上限, 是否严格截断)
CASES = [("ood", 1800, True), ("hard-verify", 1300, False)]
LABEL = "perceptual-framesamp-modul"


def _ident_for(dataset: str) -> dict:
    if dataset == "hard-verify":
        return F.hard0_identity("PickXtimes", 0)
    task, tier = F.v9_cells_sorted()[0]
    return F.packaged_identity(task, tier, 0)


@pytest.mark.parametrize("dataset,cap,strict", CASES, ids=[c[0] for c in CASES])
def test_max_steps_reaches_builder_spec_and_session(tmp_path, dataset, cap, strict):
    ident = _ident_for(dataset)
    world = F.World()
    pol = F.fake_seat_policy()
    seen = {}
    real_play = pol.play

    def spy(session, spec, recorder):
        seen.update(step_cap=session.step_cap, spec=spec)
        return real_play(session, spec, recorder)

    pol.play = spy
    stage = tmp_path / "stage"
    assert F.run_rows(F.make_runner(stage, LABEL, pol, world), [ident]) == 0
    (b,) = world.builders
    assert (b.dataset, b.max_steps) == (dataset, cap)  # builder 按 (task, dataset) 建，构造参数即数据集步数
    assert world.make_calls[0][2] is None  # make_env_for_episode 不逐局传步数
    assert seen["step_cap"] == (cap if strict else None)
    assert seen["spec"].max_steps == cap and seen["spec"].strict_cap is strict and seen["spec"].dataset == dataset
    acc = json.loads((F.queue_dir(stage, LABEL) / "accepted" / f"{dataset}__{ident['key']}.json").read_text())
    rec = json.loads(Path(acc["result"]).read_text())
    assert rec["status"] == "success" and rec["max_steps"] == cap
    assert rec["dataset"] == dataset and rec["strict_cap"] is strict


def _blocked(tmp_path, bad: dict):
    world = F.World()
    stage = tmp_path / "stage"
    rc = F.run_rows(F.make_runner(stage, LABEL, F.fake_seat_policy(), world), [bad])
    assert rc == 3
    (row,) = F.seat_results(stage, LABEL)
    assert row["run_blocked"] is True and row["status"] == "error" and row["error"].startswith("IDENTITY_MISMATCH")
    assert row["void"] is True and not list((F.queue_dir(stage, LABEL) / "claims").glob("*.json"))
    assert world.make_calls == [] and world.envs == []


@pytest.mark.parametrize("field", ["seed", "spec_sha256", "tier", "candidate"])
def test_identity_mismatch_blocks_run(tmp_path, field):
    """ood：身份行任一项与 builder 真实解析不符 → RUN_BLOCKED（退出 3），不建环境、领取作废。"""
    task, tier = F.v9_cells_sorted()[0]
    bad = dict(F.packaged_identity(task, tier, 0))
    if field == "spec_sha256":
        bad[field] = "f" * 64
    elif field == "tier":
        bad["tier"] = next(t for _, t in F.v9_cells_sorted() if t != tier)
    else:
        bad[field] = bad[field] + 1
    bad["key"] = f"{bad['task']}_{bad['tier']}_{bad['seed']}"
    _blocked(tmp_path, bad)


@pytest.mark.parametrize("field", ["seed", "source_episode", "candidate", "spec_sha256"])
def test_hard_verify_identity_mismatch_blocks_run(tmp_path, field):
    """hard-verify：seed／source_episode 与 builder 不符，或 candidate／spec_sha256 非 null → RUN_BLOCKED、不建环境。"""
    bad = dict(F.hard0_identity("PickXtimes", 1))
    if field in ("seed", "source_episode"):
        bad[field] = bad[field] + 4
    elif field == "candidate":
        bad[field] = 0
    else:
        bad[field] = "a" * 64
    bad["key"] = f"{bad['task']}_{bad['tier']}_{bad['seed']}"
    _blocked(tmp_path, bad)


def test_validate_identity_row_structure():
    ec = F.env_client()
    task, tier = F.v9_cells_sorted()[0]
    good = F.packaged_identity(task, tier, 0)
    assert ec.validate_v8_identity(good, "ood") is None
    cases = {
        "缺字段": {k: v for k, v in good.items() if k != "spec_sha256"},
        "seed 为浮点": dict(good, seed=float(good["seed"])),
        "seed 为 bool": dict(good, seed=True),
        "candidate 为字符串": dict(good, candidate="3"),
        "source_episode 为浮点": dict(good, source_episode=1.0),
        "指纹长度不对": dict(good, spec_sha256="ab"),
        "key 与字段不符": dict(good, key=good["key"] + "x"),
    }
    for name, row in cases.items():
        assert ec.validate_v8_identity(row, "ood"), name
    # ood 的行喂给 hard-verify 模式：档位与指纹都不合 xhard0 口径
    assert ec.validate_v8_identity(good, "hard-verify")


def test_validate_hard0_identity_row_structure():
    ec = F.env_client()
    good = F.hard0_identity("PickXtimes", 0)
    assert ec.validate_v8_identity(good, "hard-verify") is None
    cases = {
        "档位不是 xhard0": dict(good, tier="xhard1", key=f"{good['task']}_xhard1_{good['seed']}"),
        "candidate 非 null": dict(good, candidate=0),
        "spec_sha256 非 null": dict(good, spec_sha256="a" * 64),
        "source_episode 为 null": dict(good, source_episode=None),
        "source_episode 为 bool": dict(good, source_episode=True),
        "key 与字段不符": dict(good, key=good["key"] + "x"),
    }
    for name, row in cases.items():
        assert ec.validate_v8_identity(row, "hard-verify"), name
    # xhard0 行喂给 ood 模式：没有规格指纹
    assert ec.validate_v8_identity(good, "ood")


def test_check_identity_requires_exact_int_candidate():
    """candidate 两侧须同为 null 或同一整数；True 与 1 不算相等。"""
    ec = F.episode_mod()
    task, tier = F.v9_cells_sorted()[0]
    want = F.packaged_identity(task, tier, 0)
    resolved = {"tier": tier, "seed": want["seed"], "candidate": want["candidate"], "spec_sha256": want["spec_sha256"]}
    assert ec.check_identity(resolved, want, dataset="ood") is None
    assert ec.check_identity(dict(resolved, candidate=None), want, dataset="ood")
    assert ec.check_identity(dict(resolved, candidate=1), dict(want, candidate=1), dataset="ood") is None
    assert ec.check_identity(dict(resolved, candidate=1), dict(want, candidate=True), dataset="ood")
    assert ec.check_identity(dict(resolved, candidate=None), dict(want, candidate=None), dataset="ood") is None
    assert ec.check_identity(resolved, dict(want, spec_sha256=None), dataset="ood")


def test_check_identity_hard0_mode():
    ec = F.episode_mod()
    want = F.hard0_identity("PickXtimes", 2)
    resolved = F.real_builder("PickXtimes", dataset="hard-verify").resolve_identity(want["builder_episode"])
    assert ec.check_identity(resolved, want, dataset="hard-verify") is None
    assert ec.check_identity(dict(resolved, source_episode=resolved["source_episode"] + 4), want, dataset="hard-verify")
    assert ec.check_identity(resolved, dict(want, source_episode=True), dataset="hard-verify")
    assert ec.check_identity(dict(resolved, spec_sha256="a" * 64), want, dataset="hard-verify")
    assert ec.check_identity(dict(resolved, tier="xhard1"), dict(want, tier="xhard1"), dataset="hard-verify")


def test_three_key_functions_agree():
    ec, em, er = F.env_client(), F.eval_manifest(), F.eval_report()
    rows = [F.packaged_identity(task, tier, 0) for task, tier in F.v9_cells_sorted()[:6]]
    rows.append(F.hard0_identity("PickXtimes", 0))
    for r in rows:
        want = f"{r['task']}_{r['tier']}_{r['seed']}"
        bare = {k: v for k, v in r.items() if k != "key"}
        assert ec.v8_key(bare) == em.v8_key(bare) == er.key_of(bare) == want
        assert ec.key_of(r) == er.key_of(r) == ec.key_of(bare) == want  # 行内带 key 时直接用它
    # 同 seed 不同档：三处都给出不同的键
    r = rows[0]
    other = dict(r, tier=next(t for _, t in F.v9_cells_sorted() if t != r["tier"]))
    other.pop("key")
    assert ec.v8_key(other) != ec.v8_key({k: v for k, v in r.items() if k != "key"})
    assert em.v8_key(other) == ec.v8_key(other) == er.key_of(other)


def test_manifest_join_writes_contract_key_without_step_cap():
    """清单第 3 步按契约写 key、不再写步数上限；清单行键与客户端身份行键同一份（两处常量同步）。"""
    em, hs = F.eval_manifest(), F.eval_manifest().load_hard_specs()
    ec = F.env_client()
    task, tier = F.v9_cells_sorted()[0]
    ident = F.packaged_identity(task, tier, 0)
    src = [{"task": task, "episode": ident["builder_episode"], "tier": tier, "seed": ident["seed"],
            "candidate": ident["candidate"], "source_episode": None, "round": None, "shard": None}]
    delivery = {"schema": em.DELIVERY_SCHEMA, "rows": [{"task": task, "tier": tier, "seed": ident["seed"],
                                                          "candidate": ident["candidate"],
                                                          "spec_sha256": ident["spec_sha256"]}]}
    (row,) = em.join_delivery(src, delivery, hs)
    assert row == {k: v for k, v in ident.items() if k != "dataset"}
    assert "effective_max_steps" not in row
    assert tuple(em.SHARD_ROW_KEYS) == tuple(ec.V8_IDENTITY_KEYS)
    assert ec.validate_v8_identity(row, "ood") is None
