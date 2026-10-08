"""C13 评估汇总 ``eval_report``（进程内调用真实 ``main``／``build_report``）：分母固定、唯一权威终态、各判定器的负例，
以及 V9 合并复用的 800 局总表逐格对 ``V9_CELLS``（F-3）。

运行根里的结果行与账本行都由本文件按生产格式手写（每条用例的期望直接写在用例里）；贯通生产者的用例见
``test_seat_runner_e2e.py``。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import eval_fakes as F

POL = "perceptual-framesamp-modul"


def _ident(i: int, task: str = "TaskA", tier: str = "xhard1") -> dict:
    seed = 7_000_000 + i
    return {"task": task, "tier": tier, "seed": seed, "candidate": i, "builder_episode": i, "source_episode": None,
            "spec_sha256": hashlib.sha256(f"{task}{tier}{seed}".encode()).hexdigest(),
            "key": f"{task}_{tier}_{seed}"}


Stage = F.Stage


def _report(tmp_path, capsys, idents, *extra, manifest_extra=None):
    manifest = F.write_manifest(tmp_path / "m" / "manifest.json", idents, **(manifest_extra or {}))
    return F.run_report(capsys, manifest, tmp_path / "stage", [POL], tmp_path / "out",
                        "--expect-total", str(len(idents)), *extra)


def test_clean_report_rates_and_outcome_keys(tmp_path, capsys):
    a, b, c = _ident(0), _ident(1), _ident(2, task="TaskB")
    st = Stage(tmp_path / "stage")
    st.accepted(a, "a1", "success")
    st.accepted(b, "b1", "timeout")
    st.accepted(c, "c1", "success")
    rc, lines, rep = _report(tmp_path, capsys, [a, b, c])
    assert rc == 0
    assert F.verdict(lines, "V8_EVAL_COVERAGE")[""] == "PASS" and F.verdict(lines, "V8_EVAL_REPORT")[""] == "PASS"
    pp = rep["per_policy"][POL]
    assert pp["outcomes"] == {"success": 2, "fail": 0, "timeout": 1, "error": 0, "missing": 0, "conflict": 0}
    assert pp["micro_success_rate"] == pytest.approx(2 / 3)
    assert pp["task_macro_success_rate"] == pytest.approx((1 / 2 + 1 / 1) / 2)  # TaskA 1/2、TaskB 1/1
    assert pp["cells"]["TaskA@xhard1"]["denominator"] == 2 and pp["cells"]["TaskB@xhard1"]["success"] == 1


def test_denominator_is_manifest_not_result_rows(tmp_path, capsys):
    """4 条结果行、3 个身份：分母 3；没有任何尝试的身份记 missing；infra 废弃尝试单列不入分。"""
    a, b, c = _ident(0), _ident(1), _ident(2)
    st = Stage(tmp_path / "stage")
    st.ledger("attempt_start", a, "a1", attempt_no=1)
    st.result(a, "a1", 1, "error", infra=True)
    st.ledger("attempt_end", a, "a1", attempt_no=1, status="error")
    st.accepted(a, "a2", "success", no=2)
    st.ledger("attempt_start", b, "b1", attempt_no=1)
    st.result(b, "b1", 1, "error", infra=True)
    st.accepted(b, "b2", "fail", no=2)
    rc, lines, rep = _report(tmp_path, capsys, [a, b, c])
    pp = rep["per_policy"][POL]
    assert pp["denominator"] == 3 and sum(pp["outcomes"].values()) == 3
    assert pp["outcomes"]["missing"] == 1 and len(pp["abandoned"]) == 2
    cov = F.verdict(lines, "V8_EVAL_COVERAGE")
    assert cov[""] == "FAIL" and cov["missing"] == "1"
    assert pp["cells"]["TaskA@xhard1"]["denominator"] == 3
    assert rc == 1


def test_partial_does_not_fail_on_missing(tmp_path, capsys):
    a, b = _ident(0), _ident(1)
    Stage(tmp_path / "stage").accepted(a, "a1", "success")
    rc, lines, rep = _report(tmp_path, capsys, [a, b], "--partial")
    cov = F.verdict(lines, "V8_EVAL_COVERAGE")
    assert cov[""] == "PASS" and cov["missing"] == "0"
    assert rep["per_policy"][POL]["outcomes"]["missing"] == 1  # 结局表照实记缺失
    assert rep["progress"]["policies"][POL]["done"] == 1


@pytest.mark.parametrize("case,reason", [("accept_infra", "accept_infra_error"),
                                         ("two_accepts", "multiple_accept"),
                                         ("rows_disagree", "attempt_rows_disagree")])
def test_conflicting_terminal(tmp_path, capsys, case, reason):
    a = _ident(0)
    st = Stage(tmp_path / "stage")
    if case == "accept_infra":
        st.result(a, "a1", 1, "error", infra=True)
        st.ledger("accept", a, "a1", status="error")
    elif case == "two_accepts":
        st.accepted(a, "a1", "success")
        st.accepted(a, "a2", "fail", no=2)
    else:
        st.accepted(a, "a1", "success")
        st.result(a, "a1", 1, "fail")
    rc, lines, rep = _report(tmp_path, capsys, [a])
    pp = rep["per_policy"][POL]
    assert [c["reason"] for c in pp["conflicts"]] == [reason]
    assert pp["outcomes"]["conflict"] == 1 and pp["outcomes"]["missing"] == 0
    cov = F.verdict(lines, "V8_EVAL_COVERAGE")
    assert cov[""] == "FAIL" and cov["conflicting_terminal"] == "1" and cov["missing"] == "0"


def test_accept_without_row_counts_as_conflict_only_when_final(tmp_path, capsys):
    a = _ident(0)
    Stage(tmp_path / "stage").ledger("accept", a, "ghost", status="success")
    _, lines, _ = _report(tmp_path, capsys, [a])
    assert F.verdict(lines, "V8_EVAL_COVERAGE")["conflicting_terminal"] == "1"
    _, lines, rep = _report(tmp_path, capsys, [a], "--partial")
    assert F.verdict(lines, "V8_EVAL_COVERAGE")["conflicting_terminal"] == "0"
    assert rep["per_policy"][POL]["accept_without_row"] == [{"key": a["key"], "attempt_id": "ghost"}]


def test_extra_and_duplicate(tmp_path, capsys):
    a, stray = _ident(0), _ident(9)
    st = Stage(tmp_path / "stage")
    st.accepted(a, "a1", "success")
    st.result(a, "a1", 1, "success")  # 同一 attempt_id 重复一行
    st.accepted(stray, "s1", "success")  # 清单之外的身份
    _, lines, rep = _report(tmp_path, capsys, [a])
    cov = F.verdict(lines, "V8_EVAL_COVERAGE")
    assert cov[""] == "FAIL" and cov["extra"] == "1" and cov["duplicate"] == "1"
    assert rep["per_policy"][POL]["denominator"] == 1


def test_late_terminal_ignored(tmp_path, capsys):
    a = _ident(0)
    st = Stage(tmp_path / "stage")
    st.accepted(a, "a1", "fail")
    st.ledger("attempt_start", a, "a2", attempt_no=2)
    st.result(a, "a2", 2, "success")
    st.ledger("attempt_end", a, "a2", attempt_no=2, status="success", late=True)
    _, lines, rep = _report(tmp_path, capsys, [a])
    assert rep["per_policy"][POL]["outcomes"]["fail"] == 1 and rep["per_policy"][POL]["outcomes"]["success"] == 0
    assert F.verdict(lines, "V8_EVAL_COVERAGE")["late_ignored"] == "1"
    assert F.verdict(lines, "V8_EVAL_COVERAGE")[""] == "PASS"


def test_canary_rows_not_counted(tmp_path, capsys):
    a = _ident(0)
    st = Stage(tmp_path / "stage")
    st.result(a, "c0", 1, "success", canary=True)
    _, lines, rep = _report(tmp_path, capsys, [a])
    assert rep["per_policy"][POL]["outcomes"]["missing"] == 1
    assert F.verdict(lines, "V8_EVAL_COVERAGE")["duplicate"] == "0"


@pytest.mark.parametrize("mutate,field", [
    (lambda st, a: st.accepted(a, "a1", "success", task_success=False), "task_success"),
    (lambda st, a: st.accepted(a, "a1", "success", exec_steps=None), "exec_steps"),
    (lambda st, a: st.accepted(a, "a1", "success", spec_sha256="0" * 64,
                               identity={"tier": a["tier"], "seed": a["seed"], "candidate": a["candidate"],
                                         "spec_sha256": "0" * 64}), "spec_sha256"),
    (lambda st, a: st.accepted(a, "a1", "success", seed=a["seed"] + 1,
                               identity={"tier": a["tier"], "seed": a["seed"] + 1, "candidate": a["candidate"],
                                         "spec_sha256": a["spec_sha256"]}), "seed"),
], ids=["task_success", "exec_steps", "spec", "seed"])
def test_count_mismatch_negatives(tmp_path, capsys, mutate, field):
    a = _ident(0)
    mutate(Stage(tmp_path / "stage"), a)
    _, lines, rep = _report(tmp_path, capsys, [a])
    rp = F.verdict(lines, "V8_EVAL_REPORT")
    assert rp[""] == "FAIL" and int(rp["count_mismatch"]) >= 1
    assert any(field in x for x in rep["count_mismatch_detail"])


def test_manifest_header_negatives(tmp_path, capsys):
    a, b = _ident(0), _ident(1)
    st = Stage(tmp_path / "stage")
    st.accepted(a, "a1", "success")
    st.accepted(b, "b1", "success")
    # expect_total 与清单行数不等
    manifest = F.write_manifest(tmp_path / "m" / "manifest.json", [a, b])
    _, lines, rep = F.run_report(capsys, manifest, tmp_path / "stage", [POL], tmp_path / "o1", "--expect-total", "3")
    assert F.verdict(lines, "V8_EVAL_REPORT")[""] == "FAIL"
    assert any("expect_total" in x for x in rep["count_mismatch_detail"])
    # 清单 cells 与 rows 逐格计数不一致
    _, lines, rep = _report(tmp_path, capsys, [a, b], manifest_extra={"cells": {"TaskA@xhard1": 3}})
    assert any("cells" in x for x in rep["count_mismatch_detail"])
    # 清单 total 与 rows 不等
    _, lines, rep = _report(tmp_path, capsys, [a, b], manifest_extra={"total": 5})
    assert any("total" in x for x in rep["count_mismatch_detail"])


def test_exec_over_cap(tmp_path, capsys):
    a = _ident(0)
    cap = 1600  # ood 启动约定的上限（手写）
    Stage(tmp_path / "stage").accepted(a, "a1", "timeout", exec_steps=cap + 1)
    _, lines, _ = _report(tmp_path, capsys, [a], "--cap", str(cap))
    rp = F.verdict(lines, "V8_EVAL_REPORT")
    assert rp[""] == "FAIL" and rp["exec_over_cap"] == "1"
    a2 = _ident(1)
    Stage(tmp_path / "stage2").accepted(a2, "b1", "timeout", exec_steps=cap)
    manifest = F.write_manifest(tmp_path / "m2" / "manifest.json", [a2])
    _, lines, _ = F.run_report(capsys, manifest, tmp_path / "stage2", [POL], tmp_path / "o2", "--expect-total", "1",
                               "--cap", str(cap))
    assert F.verdict(lines, "V8_EVAL_REPORT")["exec_over_cap"] == "0"  # 恰好等于上限不算越限


@pytest.mark.parametrize("status,kw,unexplained", [
    ("success", {}, 1),  # 已接受终态无录像
    ("error", {"error": "IK 失败"}, 0),  # 非 infra 错误终局，无录像但写明原因
    ("error", {"error": None}, 1),  # 非 infra 错误终局，无录像也无原因
])
def test_media_checks(tmp_path, capsys, status, kw, unexplained):
    a = _ident(0)
    Stage(tmp_path / "stage").accepted(a, "a1", status, media=False, **kw)
    _, lines, rep = _report(tmp_path, capsys, [a])
    assert F.verdict(lines, "V8_EVAL_REPORT")["media_unexplained"] == str(unexplained)


def test_recorder_verify_fail_is_unexplained(tmp_path, capsys):
    a = _ident(0)
    Stage(tmp_path / "stage").accepted(a, "a1", "success", recorder_verify="FAIL")
    _, lines, _ = _report(tmp_path, capsys, [a])
    assert F.verdict(lines, "V8_EVAL_REPORT")["media_unexplained"] == "1"


# ---------------------------------------------------------------- V9 合并复用：800 局总表逐格对 V9_CELLS（F-3）


def _packaged_identities() -> list[dict]:
    hs = F.hard_specs()
    out = []
    for tier in hs.TIERS:
        _, rows = hs.load_specs(hs.packaged_specs_path(tier), check_fingerprint=False)
        for r in rows:
            if hs.delivered(r):
                out.append({"task": r["task"], "tier": r["tier"], "seed": r["seed"], "candidate": r["candidate"],
                            "builder_episode": r["episode"], "source_episode": None, "spec_sha256": r["spec_sha256"],
                            "key": f"{r['task']}_{r['tier']}_{r['seed']}"})
    return out


def _v9_fixture(tmp_path, move_one: bool):
    """新评 = 新评规则内的身份；复用 = 其余。move_one=True 时把一条复用身份在 reused.json、V8 manifest、V8 结果行里
    一致地换到另一格（总数不变、各处自洽），只有逐格对 V9_CELLS 才能发现。"""
    em = F.eval_manifest()
    allrows = _packaged_identities()
    new = [r for r in allrows if em.in_new_rule(r)]
    reused = [dict(r) for r in allrows if not em.in_new_rule(r)]
    if move_one:
        victim = reused[0]
        other = next(t for t, tier in F.v9_cells_sorted() if tier == victim["tier"] and t != victim["task"])
        victim["task"] = other
        victim["key"] = f"{other}_{victim['tier']}_{victim['seed']}"
    v8_manifest = tmp_path / "v8" / "manifest.json"
    v8_manifest.parent.mkdir(parents=True)
    v8_manifest.write_text(json.dumps({"schema": em.SCHEMA, "rows": [
        {k: r[k] for k in ("task", "tier", "seed", "spec_sha256", "key")} for r in reused]}), encoding="utf-8")
    v8_sha = hashlib.sha256(v8_manifest.read_bytes()).hexdigest()
    rdoc = {"schema": em.REUSED_SCHEMA, "v8_manifest": str(v8_manifest), "v8_manifest_sha256": v8_sha,
            "count": len(reused), "rows": [{"task": r["task"], "tier": r["tier"], "seed": r["seed"],
                                            "spec_sha256": r["spec_sha256"], "v8_key": r["key"]} for r in reused]}
    mdir = tmp_path / "v9" / "manifest"
    mdir.mkdir(parents=True)
    rtext = json.dumps(rdoc, ensure_ascii=False)
    (mdir / "reused.json").write_text(rtext, encoding="utf-8")
    manifest = F.write_manifest(mdir / "manifest.json", new, reused={
        "path": "reused.json", "sha256": hashlib.sha256(rtext.encode()).hexdigest(), "count": len(reused),
        "v8_manifest": str(v8_manifest), "v8_manifest_sha256": v8_sha})
    for pol in ("smvla", "perceptual-framesamp-modul"):
        v8 = Stage(tmp_path / "v8" / "run", pol)
        for i, r in enumerate(reused):
            v8.accepted(r, f"v8-{i}", "success" if i % 3 else "fail", media=False)
        cur = Stage(tmp_path / "v9" / "run", pol)
        for i, r in enumerate(new):
            cur.accepted(r, f"v9-{i}", "success")
    return manifest, v8_manifest, len(new), len(reused)


@pytest.mark.parametrize("move_one", [False, True], ids=["exact", "one_moved"])
def test_v9_total_table_matches_v9_cells(tmp_path, capsys, move_one):
    er = F.eval_report()
    manifest, v8_manifest, n_new, n_reused = _v9_fixture(tmp_path, move_one)
    capsys.readouterr()
    rc = er.main(["--manifest", str(manifest), "--stage", str(tmp_path / "v9" / "run"), "--policies", "smvla,perceptual-framesamp-modul",
                  "--out", str(tmp_path / "out"), "--reuse", str(tmp_path / "v8"), "--reuse-manifest",
                  str(v8_manifest), "--expect-total", str(n_new), "--expect-reused", str(n_reused)])
    lines = capsys.readouterr().out.splitlines()
    rep = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    v9 = rep["v9"]
    want = {f"{t}@{tier}": n for (t, tier), n in F.hard_specs().V9_CELLS.items()}
    assert v9["total"] == sum(want.values()) and v9["new"] == n_new and v9["reused"] == n_reused
    rp = F.verdict(lines, "V9_EVAL_REPORT")
    for pol in ("smvla", "perceptual-framesamp-modul"):
        got = {c: v["denominator"] for c, v in v9["totals"][pol]["cells"].items()}
        assert (got == want) is (not move_one)
    if move_one:
        assert rp[""] == "FAIL"
        assert len(v9["count_mismatch_detail"]) == 2 * 2  # 两个模型 × 一格少一局、一格多一局
        assert all("总表格" in x for x in v9["count_mismatch_detail"])
    else:
        assert rp[""] == "PASS" and v9["count_mismatch_detail"] == []
        assert F.verdict(lines, "V9_EVAL_COVERAGE")[""] == "PASS"
        assert rc == 1  # 未给 --videos：视频行照实 FAIL，退出码非零
        assert F.verdict(lines, "V9_EVAL_VIDEOS")[""] == "FAIL"
        tot = v9["totals"]["perceptual-framesamp-modul"]["outcomes"]
        assert sum(tot.values()) == sum(want.values())
        assert tot["fail"] == sum(1 for i in range(n_reused) if i % 3 == 0)


def test_v9_reused_sha_mismatch_is_counted(tmp_path, capsys):
    er = F.eval_report()
    manifest, v8_manifest, n_new, n_reused = _v9_fixture(tmp_path, False)
    (manifest.parent / "reused.json").write_text("{}", encoding="utf-8")
    capsys.readouterr()
    er.main(["--manifest", str(manifest), "--stage", str(tmp_path / "v9" / "run"), "--policies", "perceptual-framesamp-modul",
             "--out", str(tmp_path / "out"), "--reuse", str(tmp_path / "v8"), "--reuse-manifest", str(v8_manifest),
             "--expect-total", str(n_new), "--expect-reused", str(n_reused)])
    lines = capsys.readouterr().out.splitlines()
    rp = F.verdict(lines, "V9_EVAL_REPORT")
    assert rp[""] == "FAIL" and int(rp["count_mismatch"]) >= 1


def test_reuse_args_must_come_together(tmp_path):
    er = F.eval_report()
    with pytest.raises(SystemExit):
        er.main(["--manifest", "m", "--stage", "s", "--out", str(tmp_path), "--reuse", "x"])


# ---------------------------------------------------------------- 带 --dataset：逐模型判定行（1003 评估计划 1.2）


def _hard0(i: int, task: str = "TaskA") -> dict:
    seed = 9_000_000 + i
    return {"task": task, "tier": "xhard0", "seed": seed, "candidate": None, "builder_episode": i,
            "source_episode": 3 + 4 * i, "spec_sha256": None, "key": f"{task}_xhard0_{seed}"}


def _lines_of(lines: list[str], name: str, policy: str) -> dict[str, str]:
    return F.verdict([x for x in lines if f" policy={policy} " in x + " "], name)


def _videos(root: Path, dirname: str, dataset: str, idents, names=("ep.mp4", "summary.json")):
    for ident in idents:
        d = root / dirname / dataset / "new" / f"{ident['key']}.a1"
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_text("x", encoding="utf-8")


def _ds_report(tmp_path, capsys, idents, dataset, policies, *extra, total=None):
    manifest = F.write_manifest(tmp_path / "m" / "manifest.json", idents, dataset=dataset)
    return F.run_report(capsys, manifest, tmp_path / "stage", policies, tmp_path / "out", "--dataset", dataset,
                        "--expect-total", str(len(idents) if total is None else total), *extra)


def test_dataset_hard0_lines_pass_without_spec_and_skip_cap(tmp_path, capsys, monkeypatch):
    """hard-verify：身份必备字段不含 spec_sha256；官方循环第 1301 步不算越限（exec_over_cap=skip）；视频 mp4 读得出帧即过。"""
    er = F.eval_report()
    monkeypatch.setattr(er, "count_media_frames", lambda path, videos: 5)
    a, b = _hard0(0), _hard0(1)
    st = Stage(tmp_path / "stage")
    st.accepted(a, "a1", "timeout", exec_steps=1301, dataset="hard-verify")
    st.accepted(b, "b1", "success", exec_steps=40, dataset="hard-verify")
    _videos(tmp_path / "videos", "perceptual-framesamp-modul", "hard-verify", [a, b])
    rc, lines, rep = _ds_report(tmp_path, capsys, [a, b], "hard-verify", ["perceptual-framesamp-modul"], "--videos", str(tmp_path / "videos"))
    cov, rp, vid = (_lines_of(lines, n, "perceptual-framesamp-modul") for n in ("EVAL_COVERAGE", "EVAL_REPORT", "EVAL_VIDEOS"))
    assert cov == {"": "PASS", "dataset": "hard-verify", "policy": "perceptual-framesamp-modul", "expected": "2", "missing": "0", "extra": "0",
                   "duplicate": "0", "conflicting_terminal": "0", "error_final": "0", "no_frame_error": "0",
                   "policy_seed": "NA"}
    assert rp[""] == "PASS" and rp["exec_over_cap"] == "skip" and rp["count_mismatch"] == "0"
    assert vid[""] == "PASS" and vid["expected"] == "2" and vid["videos"] == "2" and vid["decode_fail"] == "0"
    assert rc == 0 and rep["cap"] is None and rep["dataset"] == "hard-verify"


def test_dataset_hard0_requires_source_episode_and_decodable_video(tmp_path, capsys, monkeypatch):
    er = F.eval_report()
    monkeypatch.setattr(er, "count_media_frames", lambda path, videos: None)  # 解码失败
    a = _hard0(0)
    Stage(tmp_path / "stage").accepted(a, "a1", "success", dataset="hard-verify", source_episode=None)
    _videos(tmp_path / "videos", "perceptual-framesamp-modul", "hard-verify", [a])
    rc, lines, rep = _ds_report(tmp_path, capsys, [a], "hard-verify", ["perceptual-framesamp-modul"], "--videos", str(tmp_path / "videos"))
    rp, vid = _lines_of(lines, "EVAL_REPORT", "perceptual-framesamp-modul"), _lines_of(lines, "EVAL_VIDEOS", "perceptual-framesamp-modul")
    assert rp[""] == "FAIL" and any("source_episode" in x for x in rep["count_mismatch_detail"])
    assert vid[""] == "FAIL" and vid["decode_fail"] == "1" and rc == 1


def test_dataset_ood_keeps_cap_and_spec(tmp_path, capsys):
    a = _ident(0)
    Stage(tmp_path / "stage").accepted(a, "a1", "timeout", exec_steps=1601, dataset="ood", spec_sha256=None,
                                       identity={"tier": a["tier"], "seed": a["seed"], "candidate": a["candidate"]})
    _, lines, rep = _ds_report(tmp_path, capsys, [a], "ood", ["perceptual-framesamp-modul"])
    rp = _lines_of(lines, "EVAL_REPORT", "perceptual-framesamp-modul")
    assert rp[""] == "FAIL" and rp["exec_over_cap"] == "1"
    assert any("spec_sha256" in x for x in rep["count_mismatch_detail"])
    vid = _lines_of(lines, "EVAL_VIDEOS", "perceptual-framesamp-modul")  # 未给 --videos：照实 FAIL
    assert vid[""] == "FAIL" and vid["videos_root"] == "absent" and vid["missing"] == "1"


def test_dataset_crossed_rows_are_counted(tmp_path, capsys):
    """结果行的 dataset 与 --dataset 不符（两个数据集串了）→ dataset_crossed、count_mismatch、EVAL_REPORT=FAIL。"""
    a = _hard0(0)
    Stage(tmp_path / "stage").accepted(a, "a1", "success", dataset="ood")
    _, lines, rep = _ds_report(tmp_path, capsys, [a], "hard-verify", ["perceptual-framesamp-modul"])
    rp = _lines_of(lines, "EVAL_REPORT", "perceptual-framesamp-modul")
    assert rp[""] == "FAIL" and rp["dataset_crossed"] == "1"
    # 清单本身声明的数据集与 --dataset 不符也计入
    _, lines, rep = _ds_report(tmp_path, capsys, [a], "ood", ["perceptual-framesamp-modul"])
    assert any("manifest" in x for x in rep["count_mismatch_detail"])


def test_dataset_policy_variants_are_separate(tmp_path, capsys, monkeypatch):
    """--policies 接受 <policy>[:<variant>]：目录 sNN/<policy>-<variant>/，结果行按 policy_variant 过滤，两个变体互不串。"""
    er = F.eval_report()
    monkeypatch.setattr(er, "count_media_frames", lambda path, videos: 5)
    a, b = _hard0(0), _hard0(1)
    for var, status in (("ground-sg-oracle", "success"), ("ground-sg-qwenvl", "fail")):
        st = Stage(tmp_path / "stage", "groundsg", dirname=f"groundsg-{var}")
        for ident in (a, b):
            st.accepted(ident, f"{var}-{ident['key']}", status, dataset="hard-verify", policy_variant=var)
        _videos(tmp_path / "videos", f"groundsg-{var}", "hard-verify", [a, b])
    # 混进 oracle 目录的另一变体结果行被过滤掉，不算 extra／duplicate
    Stage(tmp_path / "stage", "groundsg", dirname="groundsg-ground-sg-oracle").result(
        a, "stray", 2, "fail", dataset="hard-verify", policy_variant="ground-sg-qwenvl")
    specs = ["groundsg:ground-sg-oracle", "groundsg:ground-sg-qwenvl"]
    rc, lines, rep = _ds_report(tmp_path, capsys, [a, b], "hard-verify", specs, "--videos", str(tmp_path / "videos"))
    assert rep["per_policy"]["groundsg:ground-sg-oracle"]["outcomes"]["success"] == 2
    assert rep["per_policy"]["groundsg:ground-sg-qwenvl"]["outcomes"]["fail"] == 2
    for spec in specs:
        for name in ("EVAL_COVERAGE", "EVAL_REPORT", "EVAL_VIDEOS"):
            assert _lines_of(lines, name, spec)[""] == "PASS", (spec, name)
    assert rc == 0


def test_dataset_partial_always_fails(tmp_path, capsys):
    a, b = _hard0(0), _hard0(1)
    Stage(tmp_path / "stage").accepted(a, "a1", "success", dataset="hard-verify")
    rc, lines, _ = _ds_report(tmp_path, capsys, [a, b], "hard-verify", ["perceptual-framesamp-modul"], "--partial")
    cov = _lines_of(lines, "EVAL_COVERAGE", "perceptual-framesamp-modul")
    assert cov[""] == "FAIL" and cov["partial"] == "1" and cov["missing"] == "1" and rc == 1


def test_dataset_requires_expect_total(tmp_path):
    er = F.eval_report()
    with pytest.raises(SystemExit):
        er.main(["--manifest", "m", "--stage", "s", "--out", str(tmp_path), "--dataset", "hard-verify"])


# ---------------------------------------------------------------- 1006 第三阶段：真实 1800 上限、模型种子、无帧 error 单列


def _ood_rows(tmp_path, cap_values, *, seed=7, trace_max=None, trace_extra=None, **kw):
    """ood 两个身份：结果行写 max_steps／effective_max_steps／effective_cap（取 cap_values），rec 目录写 trace header。"""
    idents = [_ident(0), _ident(1)]
    st = Stage(tmp_path / "stage")
    for i, ident in enumerate(idents):
        row = st.accepted(ident, f"a{i}", "success", dataset="ood", policy_seed=seed, **cap_values, **kw)
        if trace_max is not None:
            Path(row["rec_dir"], "trace.jsonl").write_text(
                json.dumps({"kind": "header", "max_steps": trace_max, **(trace_extra or {}),
                            "identity": {"key": ident["key"]}}) + "\n"
                + json.dumps({"kind": "end", "status": "success"}) + "\n", encoding="utf-8")
    return idents


def test_dataset_cap_1800_rejects_old_1600_rows(tmp_path, capsys):
    """--cap 1800：仍按 1600 截断的旧路线（结果行 max_steps=1600）不能误过；1800 的行通过；trace header 不符也拦。"""
    idents = _ood_rows(tmp_path, {"max_steps": 1600, "effective_max_steps": 1600})
    rc, lines, rep = _ds_report(tmp_path, capsys, idents, "ood", [POL], "--cap", "1800", "--policy-seed", "7")
    rp = _lines_of(lines, "EVAL_REPORT", POL)
    print(next(x for x in lines if x.startswith("EVAL_REPORT=")))
    assert rp[""] == "FAIL" and rp["cap"] == "1800" and rp["cap_mismatch"] == "2" and rp["exec_over_cap"] == "0"
    assert rp["policy_seed"] == "7" and rc == 1
    assert any("result.max_steps=1600" in p for x in rep["cap_mismatch_detail"] for p in x["problems"])

    ok = tmp_path / "ok"
    idents = _ood_rows(ok, {"max_steps": 1800, "effective_max_steps": 1800, "effective_cap": 1800}, trace_max=1800)
    _, lines, _ = _ds_report(ok, capsys, idents, "ood", [POL], "--cap", "1800", "--policy-seed", "7")
    rp = _lines_of(lines, "EVAL_REPORT", POL)
    print(next(x for x in lines if x.startswith("EVAL_REPORT=")))
    assert rp[""] == "PASS" and rp["cap_mismatch"] == "0" and rp["policy_seed"] == "7"

    bad_trace = tmp_path / "bad_trace"
    idents = _ood_rows(bad_trace, {"max_steps": 1800}, trace_max=1600)
    _, lines, rep = _ds_report(bad_trace, capsys, idents, "ood", [POL], "--cap", "1800")
    rp = _lines_of(lines, "EVAL_REPORT", POL)
    assert rp[""] == "FAIL" and rp["cap_mismatch"] == "2"
    assert any("trace.header.max_steps=1600" in p for x in rep["cap_mismatch_detail"] for p in x["problems"])

    absent = tmp_path / "absent"
    idents = _ood_rows(absent, {})
    _, lines, rep = _ds_report(absent, capsys, idents, "ood", [POL], "--cap", "1800")
    assert _lines_of(lines, "EVAL_REPORT", POL)["cap_mismatch"] == "2"  # 一个上限字段都没写也不算过


def test_dataset_cap_trace_header_effective_cap_over_theoretical_bound(tmp_path, capsys):
    """trace header 写了 effective_cap 时只核它：SimpleMemVLA header max_steps=1840（115×16 理论上界）＋
    effective_cap=1800 不算不符；只写 max_steps=1600（旧口径）仍拦；effective_cap=1600 也拦。"""
    ok = tmp_path / "smvla"
    idents = _ood_rows(ok, {"max_steps": 1800, "effective_max_steps": 1800, "effective_cap": 1800},
                       trace_max=1840, trace_extra={"effective_cap": 1800})
    _, lines, rep = _ds_report(ok, capsys, idents, "ood", [POL], "--cap", "1800", "--policy-seed", "7")
    rp = _lines_of(lines, "EVAL_REPORT", POL)
    assert rp[""] == "PASS" and rp["cap_mismatch"] == "0" and not rep["cap_mismatch_detail"]

    old = tmp_path / "old"
    idents = _ood_rows(old, {"max_steps": 1800}, trace_max=1600)
    _, lines, rep = _ds_report(old, capsys, idents, "ood", [POL], "--cap", "1800")
    assert _lines_of(lines, "EVAL_REPORT", POL)["cap_mismatch"] == "2"
    assert any("trace.header.max_steps=1600" in p for x in rep["cap_mismatch_detail"] for p in x["problems"])

    eff = tmp_path / "eff1600"
    idents = _ood_rows(eff, {"max_steps": 1800}, trace_max=1840, trace_extra={"effective_cap": 1600})
    _, lines, rep = _ds_report(eff, capsys, idents, "ood", [POL], "--cap", "1800")
    assert _lines_of(lines, "EVAL_REPORT", POL)["cap_mismatch"] == "2"
    probs = [p for x in rep["cap_mismatch_detail"] for p in x["problems"]]
    assert any("trace.header.effective_cap=1600" in p for p in probs)
    assert not any("trace.header.max_steps" in p for p in probs)


def test_dataset_policy_seed_grouping(tmp_path, capsys):
    """--policy-seed 7：结果行缺 policy_seed 或为别的种子计 count_mismatch；不给时混了两个种子计 policy_seed_mixed。"""
    idents = _ood_rows(tmp_path, {"max_steps": 1800}, seed=None)
    _, lines, rep = _ds_report(tmp_path, capsys, idents, "ood", [POL], "--cap", "1800", "--policy-seed", "7")
    rp = _lines_of(lines, "EVAL_REPORT", POL)
    assert rp[""] == "FAIL" and rp["count_mismatch"] == "2" and rp["policy_seed"] == "7"
    assert any("policy_seed=None != --policy-seed 7" in x for x in rep["count_mismatch_detail"])
    mixed = tmp_path / "mixed"
    a, b = _ident(0), _ident(1)
    st = Stage(mixed / "stage")
    st.accepted(a, "a0", "success", dataset="ood", policy_seed=7, max_steps=1800)
    st.accepted(b, "a1", "success", dataset="ood", policy_seed=42, max_steps=1800)
    _, lines, rep = _ds_report(mixed, capsys, [a, b], "ood", [POL], "--cap", "1800")
    rp = _lines_of(lines, "EVAL_REPORT", POL)
    assert rp[""] == "FAIL" and rp["policy_seed"] == "NA"
    assert any("policy_seed_mixed" in x for x in rep["count_mismatch_detail"])


def test_dataset_no_frame_error_reported_separately(tmp_path, capsys, monkeypatch):
    """无帧 error 例外：accepted = videos + no_frame_error，EVAL_COVERAGE 不因它 FAIL；有帧的错误终局没视频照常 missing。"""
    er = F.eval_report()
    monkeypatch.setattr(er, "count_media_frames", lambda path, videos: 5)
    a, b, c = _hard0(0), _hard0(1), _hard0(2)
    st = Stage(tmp_path / "stage")
    st.accepted(a, "a1", "success", dataset="hard-verify", policy_seed=7)
    st.accepted(b, "b1", "error", dataset="hard-verify", policy_seed=7, error="reset_failed", media=False)
    _videos(tmp_path / "videos", POL, "hard-verify", [a])
    nf = tmp_path / "videos" / POL / "hard-verify" / "new" / f"{b['key']}.a1"
    nf.mkdir(parents=True)
    (nf / "trace.jsonl").write_text(json.dumps({"kind": "header"}) + "\n"
                                    + json.dumps({"kind": "end", "status": "error", "no_frame": True,
                                                  "frames_recorded": 0}) + "\n", encoding="utf-8")
    rc, lines, _ = _ds_report(tmp_path, capsys, [a, b], "hard-verify", [POL], "--videos", str(tmp_path / "videos"),
                              "--policy-seed", "7")
    cov, vid = _lines_of(lines, "EVAL_COVERAGE", POL), _lines_of(lines, "EVAL_VIDEOS", POL)
    for x in lines:
        if x.startswith(("EVAL_COVERAGE=", "EVAL_VIDEOS=")):
            print(x)
    assert cov[""] == "PASS" and cov["error_final"] == "1" and cov["no_frame_error"] == "1"
    assert (vid[""], vid["accepted"], vid["videos"], vid["no_frame_error"]) == ("PASS", "2", "1", "1") and rc == 0

    # 有帧（frames_recorded=9）的错误终局没视频：不是无帧例外 → missing、两行 FAIL
    st.accepted(c, "c1", "error", dataset="hard-verify", policy_seed=7, error="policy_crash", media=False)
    wf = tmp_path / "videos" / POL / "hard-verify" / "new" / f"{c['key']}.a1"
    wf.mkdir(parents=True)
    (wf / "trace.jsonl").write_text(json.dumps({"kind": "header"}) + "\n"
                                    + json.dumps({"kind": "end", "status": "error", "frames_recorded": 9}) + "\n",
                                    encoding="utf-8")
    rc, lines, _ = _ds_report(tmp_path, capsys, [a, b, c], "hard-verify", [POL], "--videos", str(tmp_path / "videos"))
    cov, vid = _lines_of(lines, "EVAL_COVERAGE", POL), _lines_of(lines, "EVAL_VIDEOS", POL)
    assert cov[""] == "FAIL" and cov["error_final"] == "2" and cov["no_frame_error"] == "1"
    assert vid[""] == "FAIL" and vid["missing"] == "1" and vid["accepted"] == "2" and rc == 1
