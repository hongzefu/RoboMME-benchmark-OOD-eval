"""C13-SG-MODEL-REPORT：四模型评估总报告的覆盖、成功率（按任务、按档）、第二档并入与预算判定。期望值手写。"""
from __future__ import annotations

import json

from tests._support.dev_loaders import load_script

import sgx_report_fixtures as F


def M():
    return load_script("eval-official/model_eval_report.py")


def _v9(task, tier, seed, status, **kw):
    return {"task": task, "tier": tier, "seed": seed, "status": status, "canary": False, "infra": False, "late": False,
            "reset_calls": 2, "dataset": "ood", "max_steps": 1600, "strict_cap": True, **kw}


def _x0(task, ep, seed, status, **kw):
    return {**F.result_row(task=task, source_episode=ep, seed=seed, status=status), "reset_calls": 2, **kw}


def _sets(tmp_path):
    v9 = F.write_jsonl(tmp_path / "v9.jsonl", [
        _v9("A", "xhard1", 1, "success"), _v9("A", "xhard1", 2, "fail"), _v9("A", "xhard2", 3, "success"),
        _v9("B", "xhard1", 4, "timeout"),
        {**_v9("B", "xhard1", 5, "error"), "infra": True},  # 基础设施尝试：计入预算，不计覆盖
        _v9("B", "xhard1", 5, "success"),
    ])
    x0 = F.write_jsonl(tmp_path / "x0" / "shard-00.jsonl", [_x0("A", 3, 11, "success"), _x0("B", 3, 12, "fail")])
    return [
        {"policy": "groundsg", "variant": "ground-sg-oracle", "dataset": "ood", "side": "new", "site": "gl",
         "results": [str(v9)], "expect_total": 5},
        {"policy": "pp", "dataset": "hard-verify", "side": "orig", "results": [str(tmp_path / "x0" / "shard-*.jsonl")],
         "expect_total": 3},
    ]


def test_coverage_rates_by_task_and_tier(tmp_path):
    rep = M().build_report(_sets(tmp_path))
    a, b = rep["sets"]
    assert a["label"] == "groundsg-ground-sg-oracle/ood/new@gl"
    assert (a["covered"], a["missing"], a["success"], a["success_rate"], a["complete"]) == (5, 0, 3, 0.6, True)
    assert a["status"] == {"fail": 1, "success": 3, "timeout": 1}
    assert a["by_tier"] == {"xhard1": {"success": 2, "n": 4, "rate": 0.5}, "xhard2": {"success": 1, "n": 1, "rate": 1.0}}
    assert a["by_task"]["B"] == {"success": 1, "n": 2, "rate": 0.5}
    assert (a["attempts"], a["resets"]) == (6, 12)
    assert b["label"] == "pp/hard-verify/orig" and b["by_tier"] == {"xhard0": {"success": 1, "n": 2, "rate": 0.5}}
    assert (b["covered"], b["missing"], b["complete"]) == (2, 1, False)
    assert rep["verdict"] == "PARTIAL" and rep["incomplete"] == 1
    l1, l2 = M().verdict_lines(rep)
    assert l1 == "MODEL_EVAL_REPORT=PARTIAL sets=2 complete=1 incomplete=1 episodes=7 gate2=0"
    assert l2 == "EVAL_BUDGET=NA attempts=8 resets=16 source=results"


def test_duplicate_conflict_and_manifest(tmp_path):
    rows = [_v9("A", "xhard1", 1, "success"), _v9("A", "xhard1", 1, "fail"),
            _v9("A", "xhard1", 2, "fail"), _v9("A", "xhard1", 2, "fail"), _v9("Z", "xhard1", 9, "fail")]
    res = F.write_jsonl(tmp_path / "r.jsonl", rows)
    man = F.write_jsonl(tmp_path / "m.jsonl", [{"task": "A", "tier": "xhard1", "seed": s} for s in (1, 2, 3)])
    s = M().summarize_set({"policy": "perceptual-framesamp-modul", "dataset": "ood", "results": [str(res)], "manifest": str(man)})
    assert (s["conflicting"], s["duplicate"], s["missing"], s["extra"], s["complete"]) == (1, 1, 1, 1, False)


def test_gate2_merge_budget_from_ledger_and_cli(tmp_path, capsys):
    sets = _sets(tmp_path)
    sets[1]["expect_total"] = 2
    cfg = tmp_path / "sets.json"
    cfg.write_text(json.dumps({"sets": sets}), encoding="utf-8")
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"summary": {"policy": "groundsg-oracle", "verdict": "INFO", "compared": 192,
                                         "same_terminal": 180, "identical_trace": 150, "first_episode_identical": 3,
                                         "server_epochs": 4, "missing_orig": []}}), encoding="utf-8")
    led = F.write_jsonl(tmp_path / "led" / "seat-01.jsonl", [
        {"kind": "budget"}, {"kind": "attempt_start"}, {"kind": "reset_claim"}, {"kind": "reset_claim"},
        {"kind": "attempt_start"}, {"kind": "attempt_end"}, {"kind": "accept"}])
    out, md = tmp_path / "o" / "r.json", tmp_path / "o" / "r.md"
    rc = M().main(["--sets", str(cfg), "--gate2", str(g), "--ledger", str(tmp_path / "led" / "*.jsonl"),
                   "--max-attempts", "2", "--max-resets", "2", "--out-json", str(out), "--out-md", str(md)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert lines == ["MODEL_EVAL_REPORT=PASS sets=2 complete=2 incomplete=0 episodes=7 gate2=1",
                     "EVAL_BUDGET=PASS attempts=2<=2 resets=2<=2 source=ledger"]
    text = md.read_text(encoding="utf-8")
    assert "| groundsg-oracle | gl | INFO | 192 | 180 | 150 | 3/4 |" in text
    assert json.loads(out.read_text(encoding="utf-8"))["lines"] == lines
    rc = M().main(["--sets", str(cfg), "--max-attempts", "7", "--max-resets", "100"])
    assert rc == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "EVAL_BUDGET=FAIL attempts=8<=7 resets=16<=100 source=results"


def test_stage3_matrix_by_policy_seed(tmp_path, capsys):
    """第三阶段矩阵：多个组合按 policy_seed 汇总；每条最终行的 policy_seed 必须等于组合种子（缺字段也算不符）。"""
    ok = F.write_jsonl(tmp_path / "a.jsonl", [_v9("A", "xhard1", s, st, policy_seed=7)
                                              for s, st in ((1, "success"), (2, "fail"), (3, "timeout"))])
    memer = F.write_jsonl(tmp_path / "b.jsonl", [_x0("A", 3, 11, "success", policy_seed=7),
                                                 _x0("B", 3, 12, "error", policy_seed=7)])
    sets = [{"policy": "smvla", "dataset": "ood", "side": "new", "results": [str(ok)], "expect_total": 3},
            {"policy": "groundsg", "variant": "ground-sg-memer", "dataset": "hard-verify", "side": "new",
             "results": [str(memer)], "expect_total": 2}]
    cfg = tmp_path / "sets.json"
    cfg.write_text(json.dumps({"sets": sets}), encoding="utf-8")
    rc = M().main(["--sets", str(cfg), "--policy-seed", "7"])
    lines = capsys.readouterr().out.strip().splitlines()
    print(lines[-1])
    assert rc == 0 and lines[-1] == ("STAGE3_MATRIX=PASS policy_seed=7 combinations=2 unique_terminal=5 incomplete=0 "
                                     "seed_mismatch=0 duplicate_combination=0")
    # 一条行缺 policy_seed、一条是别的种子：seed_mismatch=2、FAIL、退出 1
    bad = F.write_jsonl(tmp_path / "a.jsonl", [_v9("A", "xhard1", 1, "success", policy_seed=7),
                                               _v9("A", "xhard1", 2, "fail"),
                                               _v9("A", "xhard1", 3, "timeout", policy_seed=42)])
    assert bad.exists()
    rc = M().main(["--sets", str(cfg), "--policy-seed", "7"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    print(line)
    assert rc == 1 and line.startswith("STAGE3_MATRIX=FAIL policy_seed=7 combinations=2 ") and "seed_mismatch=2" in line
    # 集合自带种子：两个种子各出一行；同一种子下组合标签重复计 duplicate_combination
    sets2 = [dict(sets[1], policy_seed=7), dict(sets[1], policy_seed=7), dict(sets[1], policy_seed=42)]
    rep = M().build_report(sets2)
    by = {m["policy_seed"]: m for m in rep["stage3"]}
    assert by[7]["duplicate_combination"] == 1 and by[7]["verdict"] == "FAIL"
    assert by[42]["seed_mismatch"] == 2 and by[42]["verdict"] == "FAIL"
    # 不给种子信息：不出矩阵行（旧用法输出不变）
    assert "stage3" not in M().build_report([sets[0]])
