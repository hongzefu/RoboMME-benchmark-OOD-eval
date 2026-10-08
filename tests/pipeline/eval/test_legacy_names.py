"""改名兼容（1006-rename-official-names-and-stage3-eval-plan.md 第二部分八.2「兼容」）：读历史时旧名映射成官方名，
写出与 CLI 只用官方名；别名表只在 ``scripts/eval-official/official_defs.py`` 放一份。

旧名一律取自别名表（``LEGACY_*``），测试里不另写旧名字面值（``OFFICIAL_NAMES`` 残留检查同样覆盖 tests/）。
纯 CPU、不加载权重、不联网。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._support.loaders import load_script


@pytest.fixture(scope="module")
def defs():
    return load_script("eval-official/official_defs.py")


def _old(defs, new: str) -> str:
    """官方策略标签对应的第一个旧标签（取自别名表）。"""
    return next(o for o, n in defs.LEGACY_POLICY_ALIASES.items() if n == new)


def _old_ds(defs, new: str) -> str:
    return next(o for o, n in defs.LEGACY_DATASET_ALIASES.items() if n == new)


def test_alias_tables_cover_every_rename(defs):
    assert set(defs.LEGACY_POLICY_ALIASES.values()) == {defs.POLICY_FRAMESAMP_MODUL, defs.POLICY_GROUNDSG}
    assert defs.LEGACY_DATASET_ALIASES == {_old_ds(defs, defs.DATASET_OOD): "ood",
                                           _old_ds(defs, defs.DATASET_HARD_VERIFY): "hard-verify"}
    assert set(defs.LEGACY_MODULE_ALIASES.values()) == {"framesamp_modul_client", "groundsg_client"}
    assert defs.LEGACY_CONFIG_KEY_ALIASES and set(defs.LEGACY_CONFIG_KEY_ALIASES.values()) == {"groundsg_variant"}


def test_canonical_functions(defs):
    fsm, gsg = defs.POLICY_FRAMESAMP_MODUL, defs.POLICY_GROUNDSG
    old_fsm, old_gsg = _old(defs, fsm), _old(defs, gsg)
    assert defs.canonical_policy(old_fsm) == fsm and defs.canonical_policy(old_gsg) == gsg
    assert defs.canonical_policy(f"{old_gsg}-ground-sg-oracle") == "groundsg-ground-sg-oracle"
    for keep in ("smvla", "pp", "astra", fsm, gsg, "groundsg-ground-sg-qwenvl", None, 3):
        assert defs.canonical_policy(keep) == keep
    old_hv = _old_ds(defs, "hard-verify")
    assert defs.canonical_dataset(old_hv) == "hard-verify" and defs.canonical_dataset("ood") == "ood"
    assert defs.canonical_route(f"{old_gsg}/ground-sg-oracle/orig") == "groundsg/ground-sg-oracle/orig"
    assert defs.canonical_route(f"{old_fsm}/new") == f"{fsm}/new"
    assert defs.canonical_route(f"{old_gsg}-ground-sg-oracle/{old_hv}/orig/K.a1") == \
        "groundsg-ground-sg-oracle/hard-verify/orig/K.a1"
    assert defs.canonical_route("pp/orig") == "pp/orig"
    row = {"policy": old_gsg, "dataset": old_hv, "route": f"{old_gsg}/x/new", "identity": {"dataset": old_hv}, "k": 1}
    out = defs.canonical_row(row)
    assert out == {"policy": gsg, "dataset": "hard-verify", "route": "groundsg/x/new",
                   "identity": {"dataset": "hard-verify"}, "k": 1}
    assert row["policy"] == old_gsg  # 不改入参
    assert defs.legacy_labels("groundsg-ground-sg-oracle") == [f"{old_gsg}-ground-sg-oracle"]
    assert defs.legacy_labels("smvla") == []


def test_env_client_reads_legacy_results_as_official(defs, tmp_path):
    env = load_script("eval-official/env_client.py")
    old_gsg, old_hv = _old(defs, defs.POLICY_GROUNDSG), _old_ds(defs, "hard-verify")
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps({"policy": old_gsg, "dataset": old_hv, "key": "K"}) + "\n{半行", encoding="utf-8")
    assert env.read_results(p) == [{"policy": "groundsg", "dataset": "hard-verify", "key": "K"}]
    assert "groundsg" in env.POLICIES and old_gsg not in env.POLICIES
    assert set(env.POLICY_MODULES) == set(env.POLICIES)
    for mod in env.POLICY_MODULES.values():
        assert (Path(env.__file__).parent / f"{mod}.py").is_file()


def test_budget_ledger_reads_legacy_route(defs, tmp_path):
    bl = load_script("eval-official/budget_ledger.py")
    old_fsm = _old(defs, defs.POLICY_FRAMESAMP_MODUL)
    path = tmp_path / "budget.jsonl"
    path.write_text(json.dumps({"kind": "retry", "route": f"{old_fsm}/orig", "key": "K", "interrupt": "infra"}) + "\n",
                    encoding="utf-8")
    led = bl.BudgetLedger(path)
    assert led._read()[0]["route"] == f"{defs.POLICY_FRAMESAMP_MODUL}/orig"


def test_eval_report_reads_legacy_stage_dirs_and_rejects_legacy_cli(defs, tmp_path):
    er = load_script("eval-official/eval_report.py")
    old_gsg, old_hv = _old(defs, defs.POLICY_GROUNDSG), _old_ds(defs, "hard-verify")
    d = tmp_path / "s00" / f"{old_gsg}-ground-sg-oracle"
    d.mkdir(parents=True)
    (d / "results.jsonl").write_text(json.dumps({"policy": old_gsg, "policy_variant": "ground-sg-oracle",
                                                 "dataset": old_hv, "key": "K"}) + "\n", encoding="utf-8")
    (d / f"{old_gsg}.ledger.jsonl").write_text(json.dumps({"kind": "accept", "policy": old_gsg, "key": "K"}) + "\n",
                                               encoding="utf-8")
    got = er.load_policy(tmp_path, "groundsg:ground-sg-oracle")
    assert [r["policy"] for r in got["results"]] == ["groundsg"] and got["results"][0]["dataset"] == "hard-verify"
    assert [r["policy"] for r in got["ledger"]] == ["groundsg"]
    names = er.layout_names("groundsg-ground-sg-oracle", "hard-verify")
    assert names[0] == ("groundsg-ground-sg-oracle", "hard-verify") and (f"{old_gsg}-ground-sg-oracle", old_hv) in names
    with pytest.raises(SystemExit):
        er.main(["--manifest", "m.json", "--stage", str(tmp_path), "--out", str(tmp_path / "o"),
                 "--policies", old_gsg])


def test_orig_tools_canonicalize_policy(defs, tmp_path):
    ora = load_script("eval-official/orig_observer/orig_results_adapter.py")
    old_fsm = _old(defs, defs.POLICY_FRAMESAMP_MODUL)
    res = ora.build([], tmp_path, policy=old_fsm)
    assert res["policy"] == defs.POLICY_FRAMESAMP_MODUL and res["route"] == f"{defs.POLICY_FRAMESAMP_MODUL}/orig"
    with pytest.raises(SystemExit):
        ora.main(["--policy", old_fsm, "--episode-log", str(tmp_path / "x"), "--rec-root", str(tmp_path),
                  "--out", str(tmp_path / "o.json")])


def test_official_media_check_rejects_legacy_cli(defs, tmp_path):
    omc = load_script("eval-official/official_media_check.py")
    with pytest.raises(SystemExit):
        omc.main(["--dataset", _old_ds(defs, "hard-verify"), "--ffmpeg", "/bin/true"])
    side = tmp_path / "trace.jsonl"
    side.write_text(json.dumps({"kind": "header", "route": f"{_old(defs, defs.POLICY_GROUNDSG)}/x/new",
                                "identity": {"dataset": _old_ds(defs, "ood")}}) + "\n", encoding="utf-8")
    row = omc.read_rows(side)[0]
    assert row["route"] == "groundsg/x/new" and row["identity"]["dataset"] == "ood"
