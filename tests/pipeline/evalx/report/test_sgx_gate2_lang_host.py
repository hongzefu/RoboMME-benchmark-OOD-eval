"""C13-SG-GATE2-HOST-LANG：比较器的来源主机模式与语言账本逐次比（1006 计划八.10 第 6 条、八.11；冻结说明第九节）。

两侧轨迹由 ``sgx_report_fixtures.write_episode`` 按生产格式写出，结果行带 ``host``；``language.jsonl`` 按冻结的
``LanguageLog`` 行格式手写在轨迹同目录。期望值（计数、片段）直接写在用例里。
"""
from __future__ import annotations

import json
from pathlib import Path

import sgx_report_fixtures_dev as F

IDS = [("VideoUnmask", 3, 101), ("PickXtimes", 7, 202)]


def _lang_rows(user: str = "goal: pick the cube", reply: str = "pick up the red cube", extra: bool = False):
    rows = [{"kind": "call_open", "call_id": "c0", "model": "subgoal_model", "step": 0, "retry": 0},
            {"kind": "message", "call_id": "c0", "message_index": 0, "dir": "in", "role": "system", "text": "SYS"},
            {"kind": "message", "call_id": "c0", "message_index": 1, "dir": "in", "role": "user", "text": user},
            {"kind": "message", "call_id": "c0", "message_index": 2, "dir": "out", "role": "assistant", "text": reply},
            {"kind": "call_close", "call_id": "c0", "status": "reply"},
            {"kind": "call_open", "call_id": "a0", "model": "action_model", "step": 0, "retry": 0},
            {"kind": "message", "call_id": "a0", "message_index": 0, "dir": "in", "role": "fields",
             "text": {"prompt": "pick", "grounded_subgoal": reply}},
            {"kind": "call_close", "call_id": "a0", "status": "reply", "server_final_text": "Task: pick;"}]
    if extra:
        rows += [{"kind": "call_open", "call_id": "a1", "model": "action_model", "step": 3, "retry": 0},
                 {"kind": "message", "call_id": "a1", "message_index": 0, "dir": "in", "role": "fields",
                  "text": {"prompt": "pick"}},
                 {"kind": "call_close", "call_id": "a1", "status": "reply"}]
    return rows


def _side(tmp_path: Path, side: str, *, host, lang=None):
    """写一侧轨迹、结果行（带 host）与可选的 language.jsonl；``lang``：{身份: 行列表} 或 None（不写）。"""
    root = tmp_path / side / "traces"
    rows = []
    for task, ep, seed in IDS:
        path = F.write_episode(root, task=task, source_episode=ep, seed=seed)
        if lang is not None and (task, ep, seed) in lang:
            (path.parent / "language.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lang[(task, ep, seed)]), encoding="utf-8")
        h = host(task) if callable(host) else host
        rows.append(F.result_row(task=task, source_episode=ep, seed=seed, **({"host": h} if h else {})))
    return F.write_jsonl(tmp_path / side / "results.jsonl", rows), root


def _cli(capsys, o, n, *extra):
    capsys.readouterr()
    rc = F.g2().main(["--policy", "groundsg-memer", "--orig-results", str(o[0]), "--new-results", str(n[0]),
                      "--orig-traces", str(o[1]), "--new-traces", str(n[1]), *extra])
    return rc, capsys.readouterr().out.strip().splitlines()


def _line(lines, prefix):
    (x,) = [y for y in lines if y.startswith(prefix)]
    return x


SAME = {i: _lang_rows() for i in IDS}


def test_expect_host_same_machine_passes_and_lang_identical(tmp_path, capsys):
    o = _side(tmp_path, "orig", host="sled-vail", lang=SAME)
    n = _side(tmp_path, "new", host="sled-vail.eecs.umich.edu", lang=SAME)  # 长主机名与短名等同
    rc, lines = _cli(capsys, o, n, "--expect-host", "sled-vail")
    print("\n".join(lines))
    assert rc == 0
    assert _line(lines, "GATE2_PROVENANCE=") == ("GATE2_PROVENANCE=PASS mode=host expect_host=sled-vail "
                                                 "foreign_rows=0 unknown_rows=0")
    last = lines[-1]
    assert last.startswith("GATE2=INFO policy=groundsg-memer compared=2 ") and " prompt_diff=0 reply_diff=0" in last
    assert not any(x.startswith("GATE2_LANG_FIRST_DIFF") for x in lines)


def test_expect_host_cross_machine_invalid(tmp_path, capsys):
    o = _side(tmp_path, "orig", host="sled-vail", lang=SAME)
    n = _side(tmp_path, "new", host=lambda t: "gl1512" if t == "VideoUnmask" else None, lang=SAME)
    rc, lines = _cli(capsys, o, n, "--expect-host", "sled-vail")
    print("\n".join(lines))
    assert _line(lines, "GATE2_PROVENANCE=") == ("GATE2_PROVENANCE=FAIL mode=host expect_host=sled-vail "
                                                 "foreign_rows=1 unknown_rows=1")
    assert lines[-1].startswith("GATE2=INVALID") and lines[-1].endswith("reason=cross_machine")
    # 不给 --expect-host：GL 模式原检查不变（本机行计 local_rows）
    rc, lines = _cli(capsys, o, n, "--manifest", str(F.write_jsonl(tmp_path / "m.jsonl", [
        {"task": t, "source_episode": e, "seed": s} for t, e, s in IDS])), "--orig-attempts",
        str(F.write_jsonl(tmp_path / "oa.jsonl", [{"task": t, "source_episode": e, "seed": s, "attempt": 1}
                                                  for t, e, s in IDS])))
    assert _line(lines, "GATE2_PROVENANCE=").startswith("GATE2_PROVENANCE=FAIL local_rows=2 unknown_rows=1")


def test_language_prompt_and_reply_diff_with_first_snippet(tmp_path, capsys):
    o = _side(tmp_path, "orig", host="sled-vail", lang=SAME)
    changed = {IDS[0]: _lang_rows(user="goal: pick the cube twice"),       # 输入文字不同
               IDS[1]: _lang_rows(reply="pick up the blue cube")}          # 回复不同（动作模型 fields 也随之不同）
    n = _side(tmp_path, "new", host="sled-vail", lang=changed)
    rc, lines = _cli(capsys, o, n, "--expect-host", "sled-vail")
    print("\n".join(lines))
    last = lines[-1]
    assert " prompt_diff=2 reply_diff=1" in last  # 第 1 局 user 1 条；第 2 局 fields 1 条（prompt）+ 回复 1 条
    first = _line(lines, "GATE2_LANG_FIRST_DIFF")
    assert "task=PickXtimes" in first and "call=0 message_index=2 dir=out role=assistant" in first
    assert '"pick up the red cube"' in first and '"pick up the blue cube"' in first


def test_language_extra_call_and_missing_files(tmp_path, capsys):
    o = _side(tmp_path, "orig", host="sled-vail", lang=SAME)
    n = _side(tmp_path, "new", host="sled-vail", lang={IDS[0]: _lang_rows(extra=True)})  # 第 2 局新侧缺账本
    rc, lines = _cli(capsys, o, n, "--expect-host", "sled-vail")
    print("\n".join(lines))
    assert " prompt_diff=1 reply_diff=0 lang_missing=1" in lines[-1]  # 多一次调用（只一侧有的消息）计 1
    o2 = _side(tmp_path / "x", "orig", host="sled-vail")
    n2 = _side(tmp_path / "x", "new", host="sled-vail")
    rc, lines = _cli(capsys, o2, n2, "--expect-host", "sled-vail")
    assert " prompt_diff=NA reply_diff=NA" in lines[-1] and "lang_missing" not in lines[-1]
