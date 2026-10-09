"""C13-SG-GATE2：第二档原侧对新侧逐身份比较（计划第二部分 1.7）。

夹具覆盖：缺身份、重复、动作改 1 bit、画面哈希改变，以及状态／文本／停止步／请求分叉、Astra 模式、GroundSG 启动首局。
全部通过时 ``test_gate2_selftest`` 打印 ``GATE2_SELFTEST=PASS``。期望值手写。
"""
from __future__ import annotations

import json

import sgx_report_fixtures_dev as F

IDS = [("VideoUnmask", 3, 101), ("VideoUnmask", 7, 102), ("PickXtimes", 3, 201), ("PickXtimes", 7, 202)]


def _side(tmp_path, side, mutate=None, *, skip=(), extra_rows=(), statuses=None, rows_file=None, ids=IDS,
          trace_root=None, **row_extra):
    """写一侧全部身份的轨迹与结果行；返回 (结果 jsonl 路径, 轨迹根)。"""
    root = trace_root or tmp_path / side / "traces"
    rows = []
    for i, (task, ep, seed) in enumerate(ids):
        if (task, ep, seed) in skip:
            continue
        mut = (mutate or {}).get((task, ep, seed))
        st = (statuses or {}).get((task, ep, seed), "fail")
        F.write_episode(root, task=task, source_episode=ep, seed=seed, status=st, mutate=mut)
        n = (mut or {}).get("stop_at", 6)
        rows.append(F.result_row(task=task, source_episode=ep, seed=seed, status=st, exec_steps=n, **row_extra))
    rows += list(extra_rows)
    res = F.write_jsonl(rows_file or tmp_path / side / "results.jsonl", rows)
    return res, root


def _run(tmp_path, orig, new, **kw):
    g = F.g2()
    return g.compare(g.read_rows([orig[0]]), g.read_rows([new[0]]), orig[1], new[1], **kw)


def test_identical_sides(tmp_path):
    res = _run(tmp_path, _side(tmp_path, "orig"), _side(tmp_path, "new"), expect_total=4)
    s = res["summary"]
    assert s["verdict"] == "INFO" and s["compared"] == 4 and s["identical_trace"] == 4 and s["same_terminal"] == 4
    assert all(s[f"first_diverge_{k}"] == 0 for k in ("obs", "state", "text", "action", "stop", "request"))
    line = F.g2().verdict_line(res, "groundsg-oracle")
    assert line.startswith("GATE2=INFO policy=groundsg-oracle compared=4 same_terminal=4 identical_trace=4 "
                           "first_diverge_obs=0 first_diverge_state=0 first_diverge_text=0 first_diverge_action=0 missing=0")


def test_stale_node_local_trace_path_falls_back_to_index(tmp_path):
    # 原侧结果行记的是节点本地路径（发布后已删）：仍须按身份索引找到轨迹，不得判 missing_trace
    orig = _side(tmp_path, "orig", trace_path=str(tmp_path / "gone" / "trace.jsonl"))
    res = _run(tmp_path, orig, _side(tmp_path, "new"), expect_total=4)
    s = res["summary"]
    assert s["missing_trace"] == 0 and s["identical_trace"] == 4 and s["verdict"] == "INFO"


def test_action_one_bit_and_frame_hash(tmp_path):
    mut = {IDS[0]: {"action_bit": 4}, IDS[1]: {"front": 2}}
    res = _run(tmp_path, _side(tmp_path, "orig"), _side(tmp_path, "new", mut))
    s = res["summary"]
    assert s["verdict"] == "INFO" and s["identical_trace"] == 2
    assert s["first_diverge_action"] == 1 and s["first_diverge_obs"] == 1
    by = {(r["task"], r["source_episode"], r["seed"]): r for r in res["table"]}
    assert (by[IDS[0]]["diverge_step"], by[IDS[0]]["diverge_kind"], by[IDS[0]]["diverge_field"]) == (4, "action", "action")
    assert (by[IDS[1]]["diverge_step"], by[IDS[1]]["diverge_kind"], by[IDS[1]]["diverge_field"]) == (2, "obs", "front_sha256")
    assert by[IDS[0]]["same_terminal"] is True  # 终态相同而轨迹不同


def test_state_text_stop_request_kinds(tmp_path):
    mut = {IDS[0]: {"state": 5}, IDS[1]: {"text": 3}, IDS[2]: {"stop_at": 4}, IDS[3]: {"request": 1}}
    res = _run(tmp_path, _side(tmp_path, "orig"), _side(tmp_path, "new", mut))
    by = {(r["task"], r["source_episode"], r["seed"]): r for r in res["table"]}
    assert (by[IDS[0]]["diverge_step"], by[IDS[0]]["diverge_kind"]) == (5, "state")
    assert (by[IDS[1]]["diverge_step"], by[IDS[1]]["diverge_kind"]) == (3, "text")
    assert (by[IDS[2]]["diverge_step"], by[IDS[2]]["diverge_kind"], by[IDS[2]]["diverge_field"]) == (4, "stop", "terminated")
    # 第 2 个请求（索引 1）发生在第 3 步执行后
    assert (by[IDS[3]]["diverge_step"], by[IDS[3]]["diverge_kind"]) == (3, "request")
    s = res["summary"]
    assert (s["first_diverge_state"], s["first_diverge_text"], s["first_diverge_stop"], s["first_diverge_request"]) == (1, 1, 1, 1)
    assert s["identical_trace"] == 0


def test_missing_identity_is_incomplete(tmp_path):
    res = _run(tmp_path, _side(tmp_path, "orig"), _side(tmp_path, "new", skip=(IDS[3],)))
    s = res["summary"]
    assert s["verdict"] == "INCOMPLETE" and s["missing"] == 1 and s["compared"] == 3
    assert s["missing_new"] == [list(IDS[3])]
    assert F.g2().verdict_line(res, "pp").startswith("GATE2=INCOMPLETE policy=pp compared=3")


def test_duplicate_final_rows_incomplete_but_infra_rows_ignored(tmp_path):
    dup = F.result_row(task=IDS[0][0], source_episode=IDS[0][1], seed=IDS[0][2], status="success", attempt=2)
    infra = {**F.result_row(task=IDS[1][0], source_episode=IDS[1][1], seed=IDS[1][2], status="error", attempt=2),
             "infra": True}
    res = _run(tmp_path, _side(tmp_path, "orig"), _side(tmp_path, "new", extra_rows=[dup]))
    assert res["summary"]["verdict"] == "INCOMPLETE" and res["summary"]["duplicate"] == 1
    res2 = _run(tmp_path / "b", _side(tmp_path / "b", "orig"), _side(tmp_path / "b", "new", extra_rows=[infra]))
    assert res2["summary"]["verdict"] == "INFO" and res2["summary"]["duplicate"] == 0


def test_missing_trace_and_expect_total(tmp_path):
    orig = _side(tmp_path, "orig")
    new = _side(tmp_path, "new")
    victim = new[1] / f"{IDS[2][0]}_xhard0_{IDS[2][2]}.a1" / "trace.jsonl"
    victim.unlink()
    res = _run(tmp_path, orig, new)
    assert res["summary"]["verdict"] == "INCOMPLETE" and res["summary"]["missing_trace"] == 1
    res2 = _run(tmp_path, orig, orig, expect_total=192)
    assert res2["summary"]["verdict"] == "INCOMPLETE"  # 配上 4 个身份，与 192 不符


def test_terminal_mismatch_counted(tmp_path):
    res = _run(tmp_path, _side(tmp_path, "orig"), _side(tmp_path, "new", statuses={IDS[0]: "success"}))
    s = res["summary"]
    assert s["same_terminal"] == 3 and s["first_diverge_stop"] == 1


def test_astra_mode_compares_terminal_and_subtasks(tmp_path):
    g = F.g2()
    o = _side(tmp_path, "orig")
    n = _side(tmp_path, "new", mutate={IDS[1]: {"text": 4}})
    res = g.compare(g.read_rows([o[0]]), g.read_rows([n[0]]), o[1], n[1], mode="astra")
    s = res["summary"]
    assert s["verdict"] == "INFO" and s["compared"] == 4 and s["same_terminal"] == 4 and s["same_subtasks"] == 3
    assert "identical_trace" not in s
    row = next(r for r in res["table"] if r["source_episode"] == 7 and r["task"] == "VideoUnmask")
    assert row["orig_subtasks"] == ["subgoal0", "subgoal1"]
    assert row["new_subtasks"] == ["subgoal0", "subgoal1(edited)", "subgoal1"]
    assert row["orig_requests"] == {"infer": 2}
    line = g.verdict_line(res, "astra")
    assert line.startswith("GATE2=INFO policy=astra mode=astra compared=4 same_terminal=4 same_subtasks=3 missing=0")
    # 结果行自带 subtasks 时优先
    rows_o = g.read_rows([o[0]])
    rows_n = g.read_rows([n[0]])
    for r in rows_o + rows_n:
        r["subtasks"] = ["a", "b"]
    assert g.compare(rows_o, rows_n, None, None, mode="astra")["summary"]["same_subtasks"] == 4


def test_groundsg_first_episode_with_server_epoch(tmp_path):
    g = F.g2()
    # 新侧：一个分片、两次起服务（server_epoch 0、1），首局分别是 IDS[0]、IDS[2]；IDS[2] 的动作改 1 bit
    mut = {IDS[2]: {"action_bit": 1}}
    o = _side(tmp_path, "orig")
    n = _side(tmp_path, "new", mut)
    rows_o = g.read_rows([o[0]])
    rows_n = g.read_rows([n[0]])
    for rows in (rows_o, rows_n):
        for r in rows:
            r["server_epoch"] = 0 if r["task"] == "VideoUnmask" else 1
    res = g.compare(rows_o, rows_n, o[1], n[1], groundsg=True)
    s = res["summary"]
    assert (s["first_episode_identical"], s["server_epochs"], s["first_episode_aligned"]) == (1, 2, 2)
    assert "first_episode_identical=1/2 first_episode_aligned=2" in g.verdict_line(res, "groundsg-oracle")
    # 原侧只起了一次服务：IDS[2] 在原侧不是首局，不算对齐
    for r in rows_o:
        r["server_epoch"] = 0
    s2 = g.compare(rows_o, rows_n, o[1], n[1], groundsg=True)["summary"]
    assert (s2["first_episode_identical"], s2["server_epochs"], s2["first_episode_aligned"]) == (1, 2, 1)


def test_groundsg_without_epoch_uses_first_row_per_shard_file(tmp_path):
    g = F.g2()
    to, tn = tmp_path / "orig-traces", tmp_path / "new-traces"
    o1 = _side(tmp_path / "s1", "orig", ids=IDS[:2], trace_root=to)
    o2 = _side(tmp_path / "s2", "orig", ids=IDS[2:], trace_root=to)
    n1 = _side(tmp_path / "s1", "new", ids=IDS[:2], trace_root=tn)
    n2 = _side(tmp_path / "s2", "new", {IDS[2]: {"front": 1}}, ids=IDS[2:], trace_root=tn)
    rows_o = g.read_rows([o1[0], o2[0]])
    rows_n = g.read_rows([n1[0], n2[0]])
    res = g.compare(rows_o, rows_n, to, tn, groundsg=True)
    s = res["summary"]
    assert s["verdict"] == "INFO" and s["compared"] == 4
    assert (s["first_episode_identical"], s["server_epochs"]) == (1, 2)


def test_cli_site_local_and_outputs(tmp_path, capsys):
    o = _side(tmp_path, "orig")
    n = _side(tmp_path, "new", {IDS[0]: {"action_bit": 2}})
    out = tmp_path / "out" / "g.json"
    md = tmp_path / "out" / "g.md"
    rc = F.g2().main(["--policy", "groundsg-qwenvl", "--orig-results", str(o[0]), "--new-results", str(n[0]),
                      "--orig-traces", str(o[1]), "--new-traces", str(n[1]), "--site", "local", "--groundsg",
                      "--out-json", str(out), "--out-md", str(md)])
    assert rc == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("GATE2=INFO policy=groundsg-qwenvl compared=4") and line.endswith("site=local")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["line"] == line and len(data["table"]) == 4
    assert "| VideoUnmask | 3 | 101 | fail | fail | 1 | 0 | 2 | action |" in md.read_text(encoding="utf-8")


def test_gate2_selftest(tmp_path, capsys):
    """四类夹具合在一次运行里：缺身份、重复、动作改 1 bit、画面哈希改变。"""
    g = F.g2()
    dup = F.result_row(task=IDS[1][0], source_episode=IDS[1][1], seed=IDS[1][2], status="fail", attempt=2)
    o = _side(tmp_path, "orig")
    n = _side(tmp_path, "new", {IDS[0]: {"action_bit": 3}, IDS[2]: {"front": 5}}, skip=(IDS[3],), extra_rows=[dup])
    res = _run(tmp_path, o, n)
    s = res["summary"]
    checks = {
        "missing": s["missing"] == 1,
        "duplicate": s["duplicate"] == 1,
        "action_1bit": s["first_diverge_action"] == 1,
        "frame_hash": s["first_diverge_obs"] == 1,
        "verdict": s["verdict"] == "INCOMPLETE",
        "identical": s["identical_trace"] == 1,
    }
    assert all(checks.values()), checks
    with capsys.disabled():
        print(f"\nGATE2_SELFTEST=PASS cases={len(checks)} line={g.verdict_line(res, 'selftest')}")
