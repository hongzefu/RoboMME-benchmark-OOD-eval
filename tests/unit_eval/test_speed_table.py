"""手写结果夹具验证速度报表；不启动仿真、模型或外部服务。"""
from __future__ import annotations

import json

import pytest

from robomme_ood_eval import report
from robomme_ood_eval.timing import summarize_chunks
from tests._support.dev_loaders import load_script


@pytest.fixture
def er():
    return load_script("eval-official/eval_report.py")


def timing(values=(100, 200, 300, 40), gpu="A40", *, layers=None):
    chunks = []
    for i, value in enumerate(values):
        chunk = {"decision_wall_ms": value, "action_rtt_ms": value / 2,
                 "server_infer_ms": value / 4, "lang_ms": value / 4,
                 "phase": ("first", "second", "third")[i] if i < 3 else "steady"}
        if layers:
            planner, review, lang = layers[i]
            chunk.update(has_planner=planner, has_review=review, lang_ms=lang)
        chunks.append(chunk)
    summary = summarize_chunks(chunks)
    if layers:
        names = ("planner+review", "planner_only", "review_only", "neither", "no_lang")
        strata = {name: [] for name in names}
        for chunk in chunks:
            p, r = chunk["has_planner"], chunk["has_review"]
            name = "no_lang" if not chunk["lang_ms"] else "planner+review" if p and r else "planner_only" if p else "review_only" if r else "neither"
            strata[name].append(chunk)
        summary["by_planner_review"] = {name: summarize_chunks(cs) if cs else {"n": 0} for name, cs in strata.items()}
    return {"gpu_name": gpu, "chunks": chunks, "chunk_summary": summary,
            "conservation": {"kind": "serial", "expected": len(chunks), "checked": len(chunks),
                             "violations": 0, "missing_fields": 0}}


def row(n, *, model="perceptual-framesamp-modul", gpu="A40", task="A", tier="xhard1", status="success", **extra):
    return {"key": f"{task}_{tier}_{n}", "task": task, "tier": tier, "episode": n, "seed": n,
            "model": model, "policy_label": model, "status": status, "infra": False,
            "timing": {"policy": timing(gpu=gpu)}, **extra}


def by_policy(rows):
    out = {}
    for r in rows:
        out.setdefault(r["policy_label"], []).append(r)
    return out


def test_groups_exclusions_latest_sparse_and_cli(er, tmp_path, capsys):
    rows = [row(0), row(1, gpu="RTX 6000 Ada"), row(2, gpu=None),
            row(0, model="pp"), row(1, model="pp", gpu="RTX 6000 Ada"),
            row(3, infra=True), row(4, status="timeout"), row(5, model="pp")]
    rows[-1]["timing"]["policy"]["reconnected"] = True
    old = row(0)
    old["timing"]["policy"]["chunks"][3]["decision_wall_ms"] = 9999
    rows.insert(0, old)
    rows += [row(6, task="B", status="fail")]
    p = tmp_path / "results.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = er.speed_table_from_results([p])
    speed = rep["speed"]
    assert len(speed["rows"]) == 5
    assert speed["excluded_infra"] == speed["excluded_timeout"] == speed["excluded_reconnected"] == 1
    first = speed["rows"][0]
    assert first["decisions"] == 8 and first["wall_steady_mean_ms"] == 40
    assert "非满格" in first["mult"]
    assert {r["gpu_name"] for r in speed["rows"]} == {"A40", "RTX 6000 Ada", None}
    assert er.main(["--results", str(p), "--out", str(tmp_path / "out")]) == 0
    assert json.loads((tmp_path / "out/report.json").read_text())["speed"] == speed
    assert "SPEED_TABLE=PASS rows=5 decisions=24" in capsys.readouterr().out


def test_astra_layers_preserve_original_phases_and_empty_layers(er):
    r = row(0, model="astra")
    r["timing"]["policy"] = timing((100, 200, 300, 40, 60), layers=[
        (True, True, 1), (False, True, 1), (False, True, 1), (True, True, 1), (False, False, 0)])
    table = er.speed_table({"astra": [r]})
    assert len(table) == 3
    groups = {r["policy_label"]: r for r in table}
    assert groups["astra·规划+复核"]["wall_first_ms"] == 100
    assert groups["astra·规划+复核"]["wall_steady_mean_ms"] == 40
    assert groups["astra·仅复核"]["wall_first_ms"] is None
    assert groups["astra·无语言"]["wall_first_ms"] is None
    assert groups["astra·无语言"]["wall_steady_mean_ms"] == 60
    assert er.speed_line({"rows": table, "missing": 0}).startswith("SPEED_TABLE=PASS rows=3 decisions=5")
    del r["timing"]["policy"]["chunks"]
    table = er.speed_table({"astra": [r]})
    assert all(r["approx"] for r in table)
    assert er.speed_line({"rows": table, "missing": 0}).startswith("SPEED_TABLE=FAIL")


@pytest.mark.parametrize("missing", ("all", "checked", "expected", "missing_fields", "violations"))
def test_astra_parent_evidence_cannot_be_repaired_by_raw_layer_recalculation(er, tmp_path, missing):
    r = row(0, model="astra")
    r["timing"]["policy"] = timing((40,), layers=[(True, True, 1)])
    if missing == "all":
        r["timing"]["policy"]["conservation"] = {"kind": "serial"}
    else:
        del r["timing"]["policy"]["conservation"][missing]
    # 走真实序列化与读取，不只测内存里的 Python 字典。
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps(r) + "\n", encoding="utf-8")
    speed = er.speed_table_from_results([p])["speed"]
    assert len(speed["rows"]) == 1 and speed["rows"][0]["decisions"] == 1
    assert speed["rows"][0]["checked"] == speed["rows"][0]["expected"] == 1
    assert speed["rows"][0]["missing_fields"] == 0
    assert speed["conservation"]["missing_fields"] == 1
    assert speed["conservation"]["expected"] == 1
    if missing in ("all", "checked"):
        assert speed["conservation"]["checked"] == 0
    # 报表再次 JSON 往返后仍拒绝子层掩盖父组缺项。
    restored = json.loads(json.dumps(speed))
    line = er.speed_line(restored)
    assert line.startswith("SPEED_TABLE=FAIL rows=1 decisions=1")
    assert "expected=1 missing_fields=1" in line
    assert "layer_checked=1 layer_expected=1" in line


def test_complete_astra_parent_counts_not_duplicated_and_empty_exclusions_keep_formula(er):
    r = row(0, model="astra")
    r["timing"]["policy"] = timing((100, 200, 300, 40, 60), layers=[
        (True, True, 1), (True, False, 1), (False, True, 1), (False, False, 1), (False, False, 0)])
    speed = er._speed_section({"astra": [r]})
    assert len(speed["rows"]) == 5
    assert speed["conservation"] == {"violations": 0, "missing_fields": 0, "checked": 5, "expected": 5}
    assert sum(layer["decisions"] for layer in speed["rows"]) == 5
    assert er.speed_line(speed).startswith("SPEED_TABLE=PASS rows=5 decisions=5")
    r["status"] = "timeout"
    speed = er._speed_section({"astra": [r]})
    assert speed["rows"] == [] and speed["excluded_timeout"] == 1
    assert speed["conservation"] == {"violations": 0, "missing_fields": 0, "checked": 0, "expected": 0}
    assert er.speed_line(speed).startswith("SPEED_TABLE=PASS rows=0 decisions=0")


def test_full_and_sparse_multiplication(er):
    rows = [row(n, task=task) for task in ("A", "B") for n in range(3)]
    assert er.speed_table(by_policy(rows))[0]["mult"] == "2 任务 × 1 档 × 3 局 = 6"
    sparse = er.speed_table(by_policy(rows[:-1]))[0]["mult"]
    assert sparse.startswith("5 局") and "不齐" in sparse
    for r in rows:
        r["builder_episode"] = r.pop("episode")
        r["source_episode"] = None
    assert er.speed_table(by_policy(rows))[0]["mult"] == "2 任务 × 1 档 × 3 局 = 6"


def test_weighted_mean_exact_percentiles_and_summary_only(er):
    a, b = row(0), row(1)
    b["timing"]["policy"] = timing((100, 200, 300, 80, 120, 160))
    entry = er.speed_table(by_policy([a, b]))[0]
    assert entry["wall_steady_mean_ms"] == 100
    assert entry["wall_steady_p50_ms"] == 100
    assert entry["wall_steady_p95_ms"] == pytest.approx(154)
    assert entry["action_rtt_steady_mean_ms"] == 50
    assert entry["lang_steady_mean_ms"] == entry["server_infer_steady_mean_ms"] == 25
    assert not entry["approx"]
    del b["timing"]["policy"]["chunks"]
    entry = er.speed_table(by_policy([a, b]))[0]
    assert entry["approx"] and entry["wall_steady_mean_ms"] == 100


def test_old_data_missing_and_all_old_log(er, tmp_path):
    old = row(0, timing={})
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps(old) + "\n" + json.dumps(row(1)) + "\n")
    assert er.speed_of(old) is None
    speed = er.speed_table_from_results([p])["speed"]
    assert speed["missing"] == 1 and len(speed["rows"]) == 1
    assert report.summarize_rows([old])["speed"] is None


def test_violations_and_absent_conservation_fail_without_cli_exit_change(er, tmp_path, capsys):
    r = row(0)
    policy = r["timing"]["policy"]
    policy["conservation"]["violations"] = 1
    assert er.speed_line({"rows": er.speed_table(by_policy([r])), "missing": 0}).startswith("SPEED_TABLE=FAIL")
    for evidence in ({}, {"violations": 0}, {"violations": 0, "missing_fields": 0, "checked": 3, "expected": 4}):
        policy["conservation"] = evidence
        assert er.speed_line({"rows": er.speed_table(by_policy([r])), "missing": 0}).startswith("SPEED_TABLE=FAIL")
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps(r) + "\n")
    assert er.main(["--results", str(p), "--out", str(tmp_path / "out")]) == 0
    assert "SPEED_TABLE=FAIL" in capsys.readouterr().out


def minimal_report():
    return {"schema": "v8-eval-report/1", "partial": False, "generated_at": "now", "manifest": "m",
            "stage": "s", "videos": None, "cap": 1600, "expect_total": 0, "per_policy": {}, "policies": [],
            "coverage": {}, "report": {}, "count_mismatch_detail": [], "media_unexplained_detail": [],
            "exec_over_cap_detail": [], "cap_mismatch_detail": [], "progress": None}


def test_render_speed_milliseconds_notes_and_no_old_field(er):
    rep = minimal_report()
    rep["speed"] = er._speed_section(by_policy([row(0)]))
    md = er.render_md(rep)
    assert "## 推理速度" in md and "40 ms" in md and "chunk_rtt_ms" not in md
    assert "SEND_INTERVAL_S=20" in md and "合成值" in md
    assert er._fmt_ms(None) == "—"


def test_schema_upgrade_and_old_render(er, tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps(row(0)) + "\n")
    assert er.speed_table_from_results([p])["schema"] == "v8-eval-report/2"
    rep = minimal_report()
    assert "## 推理速度" not in er.render_md(rep)
    # 旧报告的现有判定接口继续可用。
    rep["coverage"] = {k: 0 for k in ("missing", "extra", "duplicate", "conflicting_terminal", "late_ignored", "error_final")}
    rep["coverage"]["pass"] = True
    rep["report"] = {"pass": True, "count_mismatch": 0, "media_unexplained": 0, "exec_over_cap": 0}
    assert all("=PASS" in line for line in er.lines_of(rep))


def test_log_speed_schema_groups_and_seven_models(er):
    models = ("perceptual-framesamp-modul", "perceptual-framesamp-modul:variant", "groundsg:a", "groundsg:b", "smvla", "pp", "astra")
    rows = [row(i, model=m, status="fail" if i % 2 else "success") for i, m in enumerate(models)]
    rows += [row(9, status="timeout"), row(10, gpu=None)]
    log = report.summarize_rows(rows)
    assert report.SCHEMA == log["schema"] == "robomme-ood-eval-log/2"
    assert len(log["speed"]["groups"]) == 8
    assert log["speed"]["excluded_timeout"] == 1
    table = er.speed_table(by_policy(rows))
    for group, entry in zip(log["speed"]["groups"], table):
        assert group["decision_wall_ms"]["steady_mean"] == entry["wall_steady_mean_ms"]
        assert group["checked"] == group["expected"] == entry["decisions"]
    empty = report.summarize_rows([])
    assert empty["speed"] is None
    print("SPEED_TABLE_TEST=PASS rows=8 cases=9")
    print(er.speed_line(er._speed_section(by_policy(rows))))
