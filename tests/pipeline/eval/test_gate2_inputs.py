"""S6 第二档对比工具：输入绑定、节点来源、补集合核对与默认格式逐字节不变（1005 计划第二部分一节 S6）。

夹具全部手写：假清单、假账本（``AttemptLedger`` 行格式）、两侧结果行与轨迹（轨迹由 ``sgx_report_fixtures`` 写出，
identity 带 ``attempt``）。期望值直接写在用例里。
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evalx" / "report"))
import sgx_report_fixtures as F  # noqa: E402

BASE_SHA = "b869a3df9e7406b8f5458697f22656165b9c50b3"
IDS = [("VideoUnmask", 3, 101), ("VideoUnmask", 7, 102), ("PickXtimes", 3, 201), ("PickXtimes", 7, 202)]
GL = "gl1512.arc-ts.umich.edu"


def g2():
    return F.g2()


def key(task, seed):
    return f"{task}_xhard0_{seed}"


def write_side(root: Path, side: str, *, ids=IDS, host=GL, mutate=None, statuses=None, attempt=1, extra_rows=(),
               skip=(), rows_name="results.jsonl", trace_root=None):
    """写一侧轨迹与结果行（带 key、attempt_id、host）；返回 (结果路径, 轨迹根, {身份: attempt_id})。"""
    troot = trace_root or root / side / "traces"
    rows, aids = [], {}
    for task, ep, seed in ids:
        if (task, ep, seed) in skip:
            continue
        st = (statuses or {}).get((task, ep, seed), "fail")
        F.write_episode(troot, task=task, source_episode=ep, seed=seed, status=st, attempt=attempt,
                        mutate=(mutate or {}).get((task, ep, seed)))
        aid = f"{side}-{task}-{seed}-a{attempt}"
        aids[(task, ep, seed)] = aid
        extra = {"key": key(task, seed), "attempt_id": aid, "tier": "xhard0"}
        if host is not None:
            extra["host"] = host
        rows.append(F.result_row(task=task, source_episode=ep, seed=seed, status=st, attempt=attempt, **extra))
    rows += list(extra_rows)
    return F.write_jsonl(root / side / rows_name, rows), troot, aids


def write_ledger(path: Path, accepts: dict) -> Path:
    """accepts：{key: (attempt_id, attempt_no)}；按 AttemptLedger 行格式写 attempt_start／attempt_end／accept。"""
    rows = [{"kind": "budget", "reset_budget": 100, "infra_retry_budget": 2}]
    for k, (aid, no) in accepts.items():
        rows += [{"kind": "attempt_start", "key": k, "attempt_id": aid, "attempt_no": no, "retry": False},
                 {"kind": "attempt_end", "key": k, "attempt_id": aid, "attempt_no": no, "status": "fail"},
                 {"kind": "accept", "key": k, "attempt_id": aid, "attempt_no": no, "accepted_attempt_id": aid,
                  "status": "fail"}]
    return F.write_jsonl(path, rows)


def write_manifest(path: Path, ids=IDS) -> Path:
    return F.write_jsonl(path, [{"task": t, "source_episode": e, "seed": s, "shard": 0} for t, e, s in ids])


def write_attempts(path: Path, ids=IDS, attempt=1, extra=()) -> Path:
    entries = [{"task": t, "source_episode": e, "seed": s, "attempt": attempt} for t, e, s in ids] + list(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"attempts": entries}), encoding="utf-8")
    return path


def setup(tmp_path, **kw):
    o = write_side(tmp_path, "orig", **kw.pop("orig", {}))
    n = write_side(tmp_path, "new", **kw.pop("new", {}))
    led = write_ledger(tmp_path / "new.ledger.jsonl", {key(t, s): (n[2][(t, e, s)], 1) for t, e, s in n[2]})
    return o, n, led, write_manifest(tmp_path / "manifest.jsonl"), write_attempts(tmp_path / "orig-attempts.json")


def run_cli(capsys, argv):
    rc = g2().main(argv)
    return rc, capsys.readouterr().out.strip().splitlines()


def base_args(o, n, policy="groundsg-oracle"):
    return ["--policy", policy, "--orig-results", str(o[0]), "--new-results", str(n[0]),
            "--orig-traces", str(o[1]), "--new-traces", str(n[1])]


def line_of(lines, prefix):
    hit = [x for x in lines if x.startswith(prefix)]
    assert len(hit) == 1, (prefix, lines)
    return hit[0]


# ── 对齐 ─────────────────────────────────────────────────────────────────────


def test_aligned_inputs_pass(tmp_path, capsys):
    o, n, led, man, att = setup(tmp_path)
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(man), "--new-ledger", str(led),
                                                    "--orig-attempts", str(att), "--expect-total", "4"])
    assert rc == 0
    assert line_of(lines, "GATE2_INPUTS=") == ("GATE2_INPUTS=PASS expected=4 missing=0 extra=0 unaccepted=0 "
                                               "ambiguous=0 trace_binding_mismatch=0")
    assert line_of(lines, "GATE2_PROVENANCE=") == "GATE2_PROVENANCE=PASS local_rows=0 unknown_rows=0"
    last = lines[-1]
    assert last.startswith("GATE2=INFO policy=groundsg-oracle compared=4 same_terminal=4 s2f=0 f2s=0 "
                           "sr_orig=0.0000 sr_new=0.0000 sr_diff_pp=0.00 mcnemar_p=1 ")
    assert "identical_trace=4" in last and "matrix=ss:0,sf:0,st:0,fs:0,ff:4,ft:0,ts:0,tf:0,tt:0" in last
    assert "reason=" not in last and last.endswith("expect_total=4")


# ── 缺对 ─────────────────────────────────────────────────────────────────────


def test_missing_pair_incomplete_without_manifest_invalid_with(tmp_path, capsys):
    o, n, led, man, att = setup(tmp_path, new={"skip": (IDS[3],)})
    rc, lines = run_cli(capsys, base_args(o, n) + ["--new-ledger", str(led), "--orig-attempts", str(att)])
    assert rc == 0 and not any(x.startswith("GATE2_INPUTS=") for x in lines)
    assert lines[-1].startswith("GATE2=INCOMPLETE policy=groundsg-oracle compared=3 ")
    assert " missing=1 " in lines[-1]
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(man), "--new-ledger", str(led),
                                                    "--orig-attempts", str(att)])
    assert line_of(lines, "GATE2_INPUTS=").startswith("GATE2_INPUTS=FAIL expected=4 missing=1 extra=0 unaccepted=0")
    assert lines[-1].startswith("GATE2=INVALID ") and lines[-1].endswith("reason=inputs")


# ── 未接受行 ─────────────────────────────────────────────────────────────────


def test_unaccepted_row_counted_and_not_compared(tmp_path, capsys):
    # 新侧 IDS[0] 另有一条第 2 次尝试的最终行（不是账本接受的那次）；IDS[1] 账本里根本没有 accept
    o = write_side(tmp_path, "orig")
    stray = F.result_row(task=IDS[0][0], source_episode=IDS[0][1], seed=IDS[0][2], status="success", attempt=2,
                         key=key(IDS[0][0], IDS[0][2]), attempt_id="stray", host=GL)
    n = write_side(tmp_path, "new", extra_rows=[stray])
    accepts = {key(t, s): (n[2][(t, e, s)], 1) for t, e, s in n[2] if (t, e, s) != IDS[1]}
    led = write_ledger(tmp_path / "l.jsonl", accepts)
    man = write_manifest(tmp_path / "m.jsonl")
    att = write_attempts(tmp_path / "a.json")
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(man), "--new-ledger", str(led),
                                                    "--orig-attempts", str(att)])
    inp = line_of(lines, "GATE2_INPUTS=")
    assert inp == ("GATE2_INPUTS=FAIL expected=4 missing=1 extra=0 unaccepted=2 ambiguous=0 "
                   "trace_binding_mismatch=0")
    # 被接受的 IDS[0] 用的是 attempt 1（fail），不是未接受的 success
    assert " compared=3 " in lines[-1] and " f2s=0 " in lines[-1] and lines[-1].startswith("GATE2=INVALID")


def test_infra_and_late_rows_are_not_unaccepted(tmp_path, capsys):
    o = write_side(tmp_path, "orig")
    infra = {**F.result_row(task=IDS[0][0], source_episode=IDS[0][1], seed=IDS[0][2], status="error", attempt=2,
                            key=key(IDS[0][0], IDS[0][2]), attempt_id="inf", host=GL), "infra": True}
    late = {**F.result_row(task=IDS[1][0], source_episode=IDS[1][1], seed=IDS[1][2], status="fail", attempt=2,
                           key=key(IDS[1][0], IDS[1][2]), attempt_id="late", host=GL), "late": True}
    n = write_side(tmp_path, "new", extra_rows=[infra, late])
    led = write_ledger(tmp_path / "l.jsonl", {key(t, s): (n[2][(t, e, s)], 1) for t, e, s in n[2]})
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(write_manifest(tmp_path / "m.jsonl")),
                                                    "--new-ledger", str(led),
                                                    "--orig-attempts", str(write_attempts(tmp_path / "a.json"))])
    assert line_of(lines, "GATE2_INPUTS=").startswith("GATE2_INPUTS=PASS")


def test_ambiguous_ledgers_and_orphan_attempts(tmp_path, capsys):
    o, n, led, man, _ = setup(tmp_path)
    # 第二份账本对 IDS[2] 给出另一个接受 id → ambiguous
    led2 = write_ledger(tmp_path / "l2.jsonl", {key(IDS[2][0], IDS[2][2]): ("other", 1)})
    # 原侧映射用 "task|source_episode" 字典格式，并夹一条孤儿（不进比较、不计未接受）
    amap = {f"{t}|{e}": 1 for t, e, _ in IDS}
    amap["BinFill|3"] = {"attempt": 1, "status": "orphan"}
    att = tmp_path / "orig-attempts.json"
    att.write_text(json.dumps(amap), encoding="utf-8")
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(man), "--new-ledger", str(led), str(led2),
                                                    "--orig-attempts", str(att)])
    assert line_of(lines, "GATE2_INPUTS=") == ("GATE2_INPUTS=FAIL expected=4 missing=0 extra=0 unaccepted=0 "
                                               "ambiguous=1 trace_binding_mismatch=0")
    g = g2()
    m = g.read_attempt_map(att)
    assert ("BinFill", 3, None) not in m and m[("VideoUnmask", 3, None)] == {1}


def test_orig_attempt_selects_mapped_attempt(tmp_path, capsys):
    # 原侧同一身份有两条终态行（attempt 1 fail、attempt 2 success），映射说 attempt 2 权威
    o = write_side(tmp_path, "orig")
    a2 = F.result_row(task=IDS[0][0], source_episode=IDS[0][1], seed=IDS[0][2], status="success", attempt=2,
                      key=key(IDS[0][0], IDS[0][2]), host=GL)
    F.write_episode(o[1], task=IDS[0][0], source_episode=IDS[0][1], seed=IDS[0][2], status="success", attempt=2)
    rows = [json.loads(x) for x in o[0].read_text(encoding="utf-8").splitlines()] + [a2]
    F.write_jsonl(o[0], rows)
    n = write_side(tmp_path, "new")
    led = write_ledger(tmp_path / "l.jsonl", {key(t, s): (n[2][(t, e, s)], 1) for t, e, s in n[2]})
    att = write_attempts(tmp_path / "a.json", ids=IDS[1:], extra=[{"task": IDS[0][0], "source_episode": IDS[0][1],
                                                                    "attempt": 2}])
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(write_manifest(tmp_path / "m.jsonl")),
                                                    "--new-ledger", str(led), "--orig-attempts", str(att)])
    # 旧尝试（attempt 1）由适配器裁定作废，不计未接受；比较用的是 attempt 2（success）
    assert line_of(lines, "GATE2_INPUTS=") == ("GATE2_INPUTS=PASS expected=4 missing=0 extra=0 unaccepted=0 "
                                               "ambiguous=0 trace_binding_mismatch=0")
    assert " s2f=1 f2s=0 sr_orig=0.2500 sr_new=0.0000 sr_diff_pp=-25.00 " in lines[-1]
    # 映射里没有的身份：其最终行计未接受
    att2 = write_attempts(tmp_path / "a2.json", ids=IDS[1:])
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(write_manifest(tmp_path / "m.jsonl")),
                                                    "--new-ledger", str(led), "--orig-attempts", str(att2)])
    assert line_of(lines, "GATE2_INPUTS=") == ("GATE2_INPUTS=FAIL expected=4 missing=1 extra=0 unaccepted=2 "
                                               "ambiguous=0 trace_binding_mismatch=0")


def test_extra_identity_not_in_manifest(tmp_path, capsys):
    o, n, led, _, att = setup(tmp_path)
    man = write_manifest(tmp_path / "m3.jsonl", ids=IDS[:3])
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(man), "--new-ledger", str(led),
                                                    "--orig-attempts", str(att)])
    assert line_of(lines, "GATE2_INPUTS=").startswith("GATE2_INPUTS=FAIL expected=3 missing=0 extra=1 ")


# ── trace 身份绑定 ───────────────────────────────────────────────────────────


def test_trace_identity_and_attempt_mismatch(tmp_path, capsys):
    o, n, led, man, att = setup(tmp_path)
    rows = [json.loads(x) for x in n[0].read_text(encoding="utf-8").splitlines()]
    # IDS[0] 的结果行指向 IDS[1] 的轨迹（身份不符）
    rows[0]["trace_path"] = str(n[1] / f"{key(IDS[1][0], IDS[1][2])}.a1" / "trace.jsonl")
    F.write_jsonl(n[0], rows)
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(man), "--new-ledger", str(led),
                                                    "--orig-attempts", str(att)])
    assert line_of(lines, "GATE2_INPUTS=").endswith("trace_binding_mismatch=1")
    assert lines[-1].startswith("GATE2=INVALID")
    g = g2()
    # attempt 不符：结果行说第 2 次，唯一一份轨迹 header 记第 1 次
    idx = g.TraceIndex(n[1])
    row = dict(rows[2], attempt=2)
    row.pop("trace_path", None)
    p, why = idx.lookup_bound(row)
    assert p is not None and why == "attempt"
    p, why = idx.lookup_bound(rows[2])
    assert why is None


def test_header_without_attempt_falls_back_to_dir_suffix(tmp_path):
    g = g2()
    t = F.tw()
    path = tmp_path / "tr" / "X_xhard0_5.a3" / "trace.jsonl"
    with t.TraceWriter(path, route="fixture", identity={"task": "X", "source_episode": 1, "seed": 5}, max_steps=10) as w:
        w.close(status="fail")
    idx = g.TraceIndex(tmp_path / "tr")
    assert idx.lookup_bound({"task": "X", "source_episode": 1, "seed": 5, "attempt": 3})[1] is None
    assert idx.lookup_bound({"task": "X", "source_episode": 1, "seed": 5, "attempt": 1})[1] == "attempt"


# ── 节点来源 ─────────────────────────────────────────────────────────────────


def test_non_gl_row_invalid_and_unknown_rows(tmp_path, capsys):
    o, n, led, man, att = setup(tmp_path)
    rows = [json.loads(x) for x in o[0].read_text(encoding="utf-8").splitlines()]
    rows[0]["host"] = "sled-vail"
    rows[1].pop("host")
    rows[2]["node"] = "gl3001"  # node 优先于 host
    rows[2]["host"] = "sled-aspen"
    F.write_jsonl(o[0], rows)
    rc, lines = run_cli(capsys, base_args(o, n) + ["--manifest", str(man), "--new-ledger", str(led),
                                                    "--orig-attempts", str(att)])
    assert line_of(lines, "GATE2_INPUTS=").startswith("GATE2_INPUTS=PASS")
    assert line_of(lines, "GATE2_PROVENANCE=") == "GATE2_PROVENANCE=FAIL local_rows=1 unknown_rows=1"
    assert lines[-1].startswith("GATE2=INVALID") and lines[-1].endswith("reason=cross_machine")
    g = g2()
    assert g.is_gl_node("gl1512.arc-ts.umich.edu") and g.is_gl_node("GL0001")
    assert not g.is_gl_node("glx123") and not g.is_gl_node("sled-vail") and not g.is_gl_node("gl12")


# ── 补集合 ───────────────────────────────────────────────────────────────────


def test_supplement_must_equal_manifest_difference(tmp_path, capsys):
    troot = tmp_path / "orig-traces"
    old = write_side(tmp_path / "old", "orig", ids=IDS[:1], trace_root=troot)
    supp = write_side(tmp_path / "supp", "orig", ids=IDS[1:], trace_root=troot)
    n = write_side(tmp_path, "new")
    led = write_ledger(tmp_path / "l.jsonl", {key(t, s): (n[2][(t, e, s)], 1) for t, e, s in n[2]})
    man = write_manifest(tmp_path / "m.jsonl")
    args = ["--policy", "groundsg-qwenvl", "--orig-results", str(old[0]), "--orig-supplement", str(supp[0]),
            "--new-results", str(n[0]), "--orig-traces", str(troot), "--new-traces", str(n[1]),
            "--manifest", str(man), "--new-ledger", str(led)]
    rc, lines = run_cli(capsys, args)
    assert line_of(lines, "GATE2_SUPPLEMENT=") == ("GATE2_SUPPLEMENT=PASS old=1 supplement=3 expected=3 missing=0 "
                                                   "extra=0 overlap=0")
    assert line_of(lines, "GATE2_INPUTS=").startswith("GATE2_INPUTS=PASS")
    assert lines[-1].startswith("GATE2=INFO policy=groundsg-qwenvl compared=4 ")
    # 条数相同但身份错一个：补跑里有旧原侧已有的 IDS[0]、少了 IDS[3] → FAIL（不凭条数）
    supp2 = write_side(tmp_path / "supp2", "orig", ids=[IDS[0], IDS[1], IDS[2]], trace_root=tmp_path / "t2")
    args2 = list(args)
    args2[args2.index(str(supp[0]))] = str(supp2[0])
    rc, lines = run_cli(capsys, args2)
    assert line_of(lines, "GATE2_SUPPLEMENT=") == ("GATE2_SUPPLEMENT=FAIL old=1 supplement=3 expected=3 missing=1 "
                                                   "extra=1 overlap=1")
    assert "reason=" in lines[-1] and "supplement" in lines[-1].split("reason=")[1]


# ── 默认格式与 BASE 逐字节相同 ───────────────────────────────────────────────


def _load_base(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("无 git，无法取 BASE 版本")
    repo = Path(__file__).resolve().parents[3]
    src = subprocess.run([git, "show", f"{BASE_SHA}:scripts/eval-official/gate2_compare.py"], cwd=repo,
                         capture_output=True, check=True).stdout
    d = tmp_path / "base_mod"
    d.mkdir()
    (d / "gate2_compare_base.py").write_bytes(src)
    sys.path.insert(0, str(repo / "scripts" / "eval-official"))
    try:
        spec = importlib.util.spec_from_file_location("_gate2_compare_base_s6", d / "gate2_compare_base.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


@pytest.mark.parametrize("extra", [[], ["--groundsg", "--site", "local", "--expect-total", "4"], ["--mode", "astra"]])
def test_default_format_byte_identical_to_base(tmp_path, capsys, extra):
    base = _load_base(tmp_path)
    o = write_side(tmp_path, "orig")
    n = write_side(tmp_path, "new", mutate={IDS[0]: {"action_bit": 2}, IDS[1]: {"text": 3}}, skip=(IDS[3],))
    outs = {}
    for tag, mod in (("base", base), ("cur", g2())):
        j, m = tmp_path / f"{tag}.json", tmp_path / f"{tag}.md"
        rc = mod.main(base_args(o, n) + extra + ["--out-json", str(j), "--out-md", str(m)])
        outs[tag] = (rc, capsys.readouterr().out, j.read_bytes(), m.read_bytes())
    assert outs["base"] == outs["cur"]
