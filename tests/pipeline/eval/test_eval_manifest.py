"""C13 执行清单 ``eval_manifest.build_v9``：真实导出产物直接喂清单（xhard0 开关开／关两种），以及各步拒绝分支。

- 交付清单由包内真实规格的正式局（``hard_specs.delivered``）现场组装；身份清单由真实 ``export_eval_identities.main``
  （真实 builder 列出 ood 全部局）导出——不手写 800 行。
- 已评集合（V8 manifest）= 交付行里不属于新评规则 ``V9_NEW_RULE`` 的那部分；于是新评行恰为规则行，复用行恰为其余。
- F-2：xhard0 期望随开关；导出时与建清单时开关不一致必须在第 1 步被拒。
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

import eval_fakes as F
from tests._support.loaders import load_script


def _delivered_rows() -> list[dict]:
    return [dict(r) for r in _delivered_cached()]


@functools.lru_cache(maxsize=1)
def _delivered_cached() -> tuple[dict, ...]:
    hs = F.hard_specs()
    out = []
    for tier in hs.TIERS:
        path = hs.packaged_specs_path(tier)
        if not path.is_file():
            continue
        _, rows = hs.load_specs(path, check_fingerprint=False)
        out += [{"task": r["task"], "tier": r["tier"], "episode": r["episode"], "seed": r["seed"],
                 "candidate": r["candidate"], "spec_sha256": r["spec_sha256"]} for r in rows if hs.delivered(r)]
    return tuple(out)


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    em = F.eval_manifest()
    rows = _delivered_rows()
    delivery = _write(tmp_path / "in" / "delivery.json", {"schema": em.DELIVERY_SCHEMA, "rows": rows})
    v8 = [dict(task=r["task"], tier=r["tier"], seed=r["seed"], spec_sha256=r["spec_sha256"],
               key=f"{r['task']}_{r['tier']}_{r['seed']}") for r in rows if not em.in_new_rule(r)]
    v8_manifest = _write(tmp_path / "in" / "v8-manifest.json", {"schema": em.SCHEMA, "rows": v8})
    return {"rows": rows, "delivery": delivery, "v8_manifest": v8_manifest, "v8": v8}


def _switch(monkeypatch, *, export_on: bool, manifest_on: bool):
    monkeypatch.setattr(F.hard_specs(), "XHARD0_IN_TEST_HARD", export_on)
    monkeypatch.setattr(F.eval_manifest().load_hard_specs(), "XHARD0_IN_TEST_HARD", manifest_on)


def _export(tmp_path, delivery: Path, tag: str) -> tuple[Path, list[dict]]:
    exp = load_script("injection-dev/export_eval_identities.py")
    out = tmp_path / f"identities-{tag}.jsonl"
    rc = exp.main(["--out", str(out), "--official-out", str(tmp_path / f"official-{tag}.jsonl"),
                   "--delivery", str(delivery)])
    assert rc == 0
    return out, F.read_jsonl(out)


@pytest.mark.parametrize("on", [False, True], ids=["xhard0_off", "xhard0_on"])
def test_export_feeds_manifest(tmp_path, monkeypatch, inputs, on):
    em, hs = F.eval_manifest(), F.hard_specs()
    _switch(monkeypatch, export_on=on, manifest_on=on)
    ident_path, src = _export(tmp_path, inputs["delivery"], "x")
    n_x0 = len(hs.ALL_TASKS) * hs.XHARD0_PER_TASK if on else 0
    assert sum(r["tier"] == hs.XHARD0 for r in src) == n_x0
    assert len(src) == n_x0 + len(inputs["rows"])

    manifest, parts, reused_text = em.build_v9(ident_path, inputs["delivery"], em.DEFAULT_SHARDS,
                                               inputs["v8_manifest"])
    rule = [r for r in inputs["rows"] if em.in_new_rule(r)]
    assert manifest["xhard0_dropped"] == n_x0
    assert manifest["total"] == len(rule) and len(manifest["rows"]) == len(rule)
    assert {r["key"] for r in manifest["rows"]} == {f"{r['task']}_{r['tier']}_{r['seed']}" for r in rule}
    reused = json.loads(reused_text)
    assert reused["count"] == len(inputs["rows"]) - len(rule) == manifest["reused"]["count"]
    assert {r["v8_key"] for r in reused["rows"]} == {r["key"] for r in inputs["v8"]}
    # 分片两两不交、并集等于新评行
    keys = [r["key"] for p in parts for r in p]
    assert len(keys) == len(set(keys)) == len(rule) and len(parts) == em.DEFAULT_SHARDS
    # 每一行都是合法执行身份；builder_episode 由真实 builder 解析回同一身份（开关同口径）
    builders = {}
    for r in manifest["rows"]:
        assert F.env_client().validate_v8_identity(r, "ood") is None
        b = builders.setdefault(r["task"], F.real_builder(r["task"]))
        got = b.resolve_identity(r["builder_episode"])
        assert (got["tier"], got["seed"], got["candidate"], got["spec_sha256"]) == \
               (r["tier"], r["seed"], r["candidate"], r["spec_sha256"])


@pytest.mark.parametrize("export_on,manifest_on", [(True, False), (False, True)])
def test_switch_mismatch_rejected_at_source_step(tmp_path, monkeypatch, inputs, export_on, manifest_on):
    """F-2：导出与建清单的 xhard0 开关不一致，第 1 步按开关期望的 xhard0 行数判不符。"""
    em = F.eval_manifest()
    monkeypatch.setattr(F.hard_specs(), "XHARD0_IN_TEST_HARD", export_on)
    ident_path, _ = _export(tmp_path, inputs["delivery"], "y")
    monkeypatch.setattr(em.load_hard_specs(), "XHARD0_IN_TEST_HARD", manifest_on)
    with pytest.raises(em.ManifestError) as ei:
        em.build_v9(ident_path, inputs["delivery"], em.DEFAULT_SHARDS, inputs["v8_manifest"])
    assert ei.value.stage == "source"


# ---------------------------------------------------------------- 各步拒绝分支（小表，真实 hard_specs 的格表）


def _src_rows(rows):
    return [{"task": r["task"], "episode": i, "tier": r["tier"], "seed": r["seed"], "candidate": r["candidate"],
             "source_episode": None, "round": None, "shard": None} for i, r in enumerate(rows)]


def test_check_source_rejects_bad_rows(inputs):
    em = F.eval_manifest()
    hs = em.load_hard_specs()
    src = _src_rows(inputs["rows"])
    cells = dict(hs.V9_CELLS)
    assert em.check_source(src, hs, cells)["rows"] == len(src)
    bad_cases = {
        "少一行": src[:-1],
        "多一个键": [dict(src[0], extra=1)] + src[1:],
        "(task,episode) 重复": [src[0], dict(src[1], task=src[0]["task"], episode=src[0]["episode"])] + src[2:],
        "新值行 candidate 为空": [dict(src[0], candidate=None)] + src[1:],
        "新值行带 source_episode": [dict(src[0], source_episode=3)] + src[1:],
        "seed 为浮点": [dict(src[0], seed=float(src[0]["seed"]))] + src[1:],
    }
    for name, rows in bad_cases.items():
        with pytest.raises(em.ManifestError):
            em.check_source(rows, hs, cells)
        assert name


def test_join_delivery_missing_duplicate_extra(inputs):
    em = F.eval_manifest()
    hs = em.load_hard_specs()
    rows = inputs["rows"][:3]
    src = _src_rows(rows)
    deliv = {"schema": em.DELIVERY_SCHEMA, "rows": rows}
    assert len(em.join_delivery(src, deliv, hs)) == 3
    with pytest.raises(em.ManifestError, match="missing"):
        em.join_delivery(src, dict(deliv, rows=rows[:2]), hs)
    with pytest.raises(em.ManifestError) as ei:
        em.join_delivery(src, dict(deliv, rows=rows + [rows[0]]), hs)
    assert ei.value.counts["duplicate"] == 1
    with pytest.raises(em.ManifestError) as ei:
        em.join_delivery(src[:2], deliv, hs)
    assert ei.value.counts["extra"] == 1
    with pytest.raises(em.ManifestError, match="schema"):
        em.join_delivery(src, dict(deliv, schema="x"), hs)


def test_check_exec_rejects_sha_and_dup_and_xhard0(inputs):
    em = F.eval_manifest()
    hs = em.load_hard_specs()
    deliv = {"schema": em.DELIVERY_SCHEMA, "rows": inputs["rows"]}
    rows = em.join_delivery(_src_rows(inputs["rows"]), deliv, hs)
    cells = dict(hs.V9_CELLS)
    assert em.check_exec(rows, hs, cells)["total"] == len(rows)
    assert all("effective_max_steps" not in r for r in rows)  # 步数上限不进身份行
    for bad in ([dict(rows[0], spec_sha256="Z" * 64)] + rows[1:],
                [rows[0], dict(rows[1], key=rows[0]["key"])] + rows[2:],
                [dict(rows[0], tier=hs.XHARD0)] + rows[1:]):
        with pytest.raises(em.ManifestError):
            em.check_exec(bad, hs, cells)


def test_split_evaluated_requires_exact_rule(inputs):
    em = F.eval_manifest()
    hs = em.load_hard_specs()
    deliv = {"schema": em.DELIVERY_SCHEMA, "rows": inputs["rows"]}
    rows = em.join_delivery(_src_rows(inputs["rows"]), deliv, hs)
    index = em.load_v8_manifest(inputs["v8_manifest"])
    reused, new = em.split_evaluated(rows, index)
    assert all(em.in_new_rule(r) for r in new)
    assert {r["v8_key"] for r in reused} == {r["key"] for r in rows if not em.in_new_rule(r)}
    rule_row = next(r for r in rows if em.in_new_rule(r))
    with pytest.raises(em.ManifestError):  # 规则内的行也出现在 V8 里（被当成复用）
        em.split_evaluated(rows, {**index, em.quad(rule_row): dict(rule_row)})
    victim = next(iter(index))
    with pytest.raises(em.ManifestError):  # 规则外的行没命中 V8（本该复用）
        em.split_evaluated(rows, {k: v for k, v in index.items() if k != victim})


def test_load_v8_manifest_rejects_duplicates_and_schema(tmp_path, inputs):
    em = F.eval_manifest()
    rows = inputs["v8"]
    with pytest.raises(em.ManifestError):
        em.load_v8_manifest(_write(tmp_path / "d.json", {"schema": em.SCHEMA, "rows": rows + [rows[0]]}))
    with pytest.raises(em.ManifestError):
        em.load_v8_manifest(_write(tmp_path / "s.json", {"schema": "other", "rows": rows}))


def test_delivery_cells_must_match_registered_table(inputs):
    em = F.eval_manifest()
    hs = em.load_hard_specs()
    assert em.delivery_cells({"rows": inputs["rows"]}, hs) == dict(hs.V9_CELLS)
    with pytest.raises(em.ManifestError):
        em.delivery_cells({"rows": inputs["rows"][1:]}, hs)


def test_main_writes_and_reads_back(tmp_path, monkeypatch, capsys, inputs):
    em = F.eval_manifest()
    _switch(monkeypatch, export_on=False, manifest_on=False)
    ident_path, _ = _export(tmp_path, inputs["delivery"], "m")
    out = tmp_path / "manifest"
    capsys.readouterr()
    rc = em.main(["--identities", str(ident_path), "--delivery", str(inputs["delivery"]), "--out-dir", str(out),
                  "--exclude-evaluated", str(inputs["v8_manifest"])])
    lines = capsys.readouterr().out.splitlines()
    assert rc == 0
    v = F.verdict(lines, "V9_EVAL_SHARDS")
    rule = [r for r in inputs["rows"] if em.in_new_rule(r)]
    assert v[""] == "PASS" and v["total"] == str(len(rule)) and v["reused"] == str(len(inputs["rows"]) - len(rule))
    assert v["xhard0"] == "0" and v["missing"] == "0" and v["duplicate"] == "0"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert em.verify_outputs(out, manifest, em.DEFAULT_SHARDS, (out / em.REUSED_FILE).read_text(encoding="utf-8")) \
        == {"missing": 0, "extra": 0, "duplicate": 0, "roundtrip_mismatch": 0, "reused_mismatch": 0}
    # 篡改一个分片文件：读回核对必须报出
    p = out / "shard-00.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(doc[1:]), encoding="utf-8")
    chk = em.verify_outputs(out, manifest, em.DEFAULT_SHARDS, (out / em.REUSED_FILE).read_text(encoding="utf-8"))
    assert chk["missing"] == 1


def test_main_fail_line_and_no_outputs(tmp_path, capsys, inputs):
    em = F.eval_manifest()
    bad_ident = tmp_path / "bad.jsonl"
    bad_ident.write_text("", encoding="utf-8")
    out = tmp_path / "o"
    capsys.readouterr()
    rc = em.main(["--identities", str(bad_ident), "--delivery", str(inputs["delivery"]), "--out-dir", str(out),
                  "--exclude-evaluated", str(inputs["v8_manifest"])])
    assert rc == 1
    assert F.verdict(capsys.readouterr().out.splitlines(), "V9_EVAL_SHARDS")[""] == "FAIL"
    assert not (out / "manifest.json").exists()


def test_main_v9_full_mode(tmp_path, monkeypatch, capsys, inputs):
    """--mode v9-full：不剔除已评身份，800 局全量切片；不写 reused.json；判定行 EVAL_SHARDS（期望值按 V9 交付手写）。"""
    em = F.eval_manifest()
    _switch(monkeypatch, export_on=False, manifest_on=False)
    ident_path, _ = _export(tmp_path, inputs["delivery"], "full")
    out = tmp_path / "full"
    capsys.readouterr()
    rc = em.main(["--mode", "v9-full", "--identities", str(ident_path), "--delivery", str(inputs["delivery"]),
                  "--out-dir", str(out)])
    lines = capsys.readouterr().out.splitlines()
    assert rc == 0
    v = F.verdict(lines, "EVAL_SHARDS")
    assert v == {"": "PASS", "mode": "v9-full", "total": "800", "cells": "43", "missing": "0", "extra": "0",
                 "duplicate": "0", "xhard0": "0"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "v9-full" and manifest["total"] == 800 and len(manifest["rows"]) == 800
    assert not (out / em.REUSED_FILE).exists()
    shards = [json.loads((out / f"shard-{i:02d}.json").read_text(encoding="utf-8")) for i in range(em.DEFAULT_SHARDS)]
    keys = [r["key"] for p in shards for r in p]
    assert len(keys) == len(set(keys)) == 800
    assert all(set(r) == set(em.SHARD_ROW_KEYS) for p in shards for r in p)


def test_main_v9_full_rejects_shifted_builder_episodes(tmp_path, monkeypatch, capsys, inputs):
    """身份清单局号整体 +12（导出时 XHARD0_IN_TEST_HARD 开、评估时关）：写分片前即 FAIL stage=builder，不留产物。"""
    em = F.eval_manifest()
    _switch(monkeypatch, export_on=False, manifest_on=False)
    ident_path, _ = _export(tmp_path, inputs["delivery"], "shift")
    rows = [json.loads(x) for x in ident_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    shifted = tmp_path / "shifted.jsonl"
    shifted.write_text("".join(json.dumps({**r, "episode": r["episode"] + 12}) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "shift-out"
    capsys.readouterr()
    rc = em.main(["--mode", "v9-full", "--identities", str(shifted), "--delivery", str(inputs["delivery"]),
                  "--out-dir", str(out)])
    v = F.verdict(capsys.readouterr().out.splitlines(), "EVAL_SHARDS")
    assert rc == 1 and v[""] == "FAIL" and v["stage"] == "builder"
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("argv", [["--mode", "v9-full", "--out-dir", "o"],
                                  ["--out-dir", "o", "--identities", "i", "--delivery", "d"],
                                  ["--mode", "hard0", "--out-dir", "o", "--identities", "i"],
                                  ["--mode", "v9-full", "--out-dir", "o", "--identities", "i", "--delivery", "d",
                                   "--pair-shards"]],
                         ids=["full_missing_inputs", "default_v9_new_needs_exclude", "hard0_rejects_identities",
                              "pair_shards_only_hard0"])
def test_main_mode_argument_rules(argv):
    """默认模式仍是 v9-new（缺 --exclude-evaluated 即参数错误）；各模式的必填与互斥参数。"""
    with pytest.raises(SystemExit) as ei:
        F.eval_manifest().main(argv)
    assert ei.value.code == 2
