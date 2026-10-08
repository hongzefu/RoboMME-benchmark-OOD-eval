"""C13-SG-LANG-IO：语言账本验收 ``lang_io_check.py``（1006 计划八.11「验收」、接口冻结说明第五节）。

夹具按冻结的 ``LanguageLog`` 行格式手写 ``language.jsonl``（``call_open``／``message``／``call_close``／``reuse``），
``trace.jsonl`` 手写 demo、``response``、``step``（带 ``source_call_id``）行；帧哈希用可读的字符串代替。正例 PASS 后逐个
做反例：改 system、交换消息顺序、同一步多一次调用、缺一条回复、引用错 attempt，以及悬空调用、外壳最终文字为空、MemER
重问次数与归档日志不符、执行步无来源、缺账本。期望计数直接写在用例里。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from tests._support.loaders import load_script

SYS = "You are a helpful robot subgoal predictor."


def L():
    return load_script("eval-official/lang_io_check.py")


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def episode(root: Path, name: str = "PickXtimes_xhard1_7.a1", *, tag: str = "", system: str = SYS,
            server_text: str | None = "Task: pick;\nCurrent Subgoal: pick red;\nAction: ", route: str | None = None) -> Path:
    """一局：4 个执行步、2 次动作模型调用（第 0、2 步，各执行 2 步）、1 次子目标模型调用（第 0 步）。"""
    d = root / name
    d.mkdir(parents=True)
    trace = [{"kind": "header", "identity": {"key": name.split(".a")[0]}, "max_steps": 1800,
              **({"route": route} if route is not None else {})},
             {"kind": "demo", "frames": 2, "front_sha256": [f"df0{tag}", f"df1{tag}"],
              "wrist_sha256": [f"dw0{tag}", f"dw1{tag}"], "texts": ["pick"]},
             {"kind": "response", "step": 0}]
    for s in (1, 2, 3, 4):
        if s == 3:
            trace.append({"kind": "response", "step": 2})
        trace.append({"kind": "step", "step": s, "front_sha256": f"f{s}{tag}", "wrist_sha256": f"w{s}{tag}",
                      "source_call_id": "a0" if s <= 2 else "a1", "chunk_index": (s - 1) % 2})
    trace.append({"kind": "end", "status": "success", "exec_steps": 4})
    _write(d / "trace.jsonl", trace)
    lang = [
        {"kind": "call_open", "call_id": "c0", "model": "subgoal_model", "step": 0, "params": {"temperature": 0},
         "transport_attempt": 0, "retry": 0},
        {"kind": "message", "call_id": "c0", "message_index": 0, "dir": "in", "role": "system", "text": system},
        {"kind": "message", "call_id": "c0", "message_index": 1, "dir": "in", "role": "user", "text": "goal: pick",
         "images": [{"slot": 0, "ref": "current", "phase": "demo", "frame_idx": 1, "cam": "front",
                     "raw_sha256": f"df1{tag}"}]},
        {"kind": "message", "call_id": "c0", "message_index": 2, "dir": "out", "role": "assistant", "text": "pick red"},
        {"kind": "call_close", "call_id": "c0", "status": "reply", "parsed": "pick red"},
        {"kind": "call_open", "call_id": "a0", "model": "action_model", "step": 0, "retry": 0},
        {"kind": "message", "call_id": "a0", "message_index": 0, "dir": "in", "role": "fields",
         "text": {"prompt": "pick", "grounded_subgoal": "pick red"},
         "images": [{"slot": 0, "ref": "current", "phase": "demo", "frame_idx": 1, "cam": "wrist",
                     "raw_sha256": f"dw1{tag}"}]},
        {"kind": "call_close", "call_id": "a0", "status": "reply", "server_final_text": server_text},
        {"kind": "call_open", "call_id": "a1", "model": "action_model", "step": 2, "retry": 0},
        {"kind": "message", "call_id": "a1", "message_index": 0, "dir": "in", "role": "fields",
         "text": {"prompt": "pick", "grounded_subgoal": "pick red"},
         "images": [{"slot": 0, "ref": "current", "phase": "exec", "frame_idx": 2, "cam": "wrist",
                     "raw_sha256": f"w2{tag}", "sources": [{"raw_sha256": f"f2{tag}", "cam": "front"}]}]},
        {"kind": "call_close", "call_id": "a1", "status": "reply", "server_final_text": server_text},
    ]
    _write(d / "language.jsonl", lang)
    return d


def _lang(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "language.jsonl").read_text(encoding="utf-8").splitlines()]


def _run(capsys, *argv) -> tuple[int, dict, str]:
    capsys.readouterr()
    rc = L().main([str(a) for a in argv])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    head, *rest = line.split()
    return rc, {"": head.split("=", 1)[1], **dict(x.split("=", 1) for x in rest)}, line


def test_pass_two_episodes(tmp_path, capsys):
    episode(tmp_path, "A_xhard1_1.a1")
    episode(tmp_path, "A_xhard1_2.a1")
    rc, v, line = _run(capsys, "--root", tmp_path)
    print(line)
    assert rc == 0 and line == ("LANG_IO=PASS episodes=2 unresolved_steps=0 open_calls=0 image_ref_unresolved=0 "
                                "order_bad=0 extra_calls=0 reply_missing=0 retry_mismatch=0 server_text_empty=0 "
                                "system_inconsistent=0 language_missing=0 bad_rows=0")


def test_changed_system_fails(tmp_path, capsys):
    episode(tmp_path, "A_xhard1_1.a1")
    episode(tmp_path, "A_xhard1_2.a1")
    episode(tmp_path, "A_xhard1_3.a1", system=SYS + " (edited)")
    rc, v, line = _run(capsys, "--root", tmp_path)
    print(line)
    assert rc == 1 and v["system_inconsistent"] == "1"
    # 只有一局时与多数比不出来：给出期望哈希即可拦下
    import hashlib
    one = tmp_path / "one"
    episode(one, system=SYS + " (edited)")
    rc, v, _ = _run(capsys, "--root", one, "--expect-system", f"subgoal_model={hashlib.sha256(SYS.encode()).hexdigest()}")
    assert rc == 1 and v["system_inconsistent"] == "1"


def test_system_grouped_by_route_mixed_root(tmp_path, capsys):
    """MERGE-1（R7 审查 finding 2）：system 一致按（模型, trace header route）分组——QwenVL 与 MemER 两种路线混放在
    同一 --root、各自 system 不同，不误判；同一路线内改了 system 照样抓到。"""
    q, m = "groundsg/ground-sg-qwenvl/new", "groundsg/ground-sg-memer/new"
    memer_sys = "You are MemER, a memory-augmented subgoal predictor."
    episode(tmp_path, "A_xhard1_1.a1", route=q)
    episode(tmp_path, "A_xhard1_2.a1", route=q)
    episode(tmp_path, "A_xhard1_3.a1", route=m, system=memer_sys)
    rc, v, line = _run(capsys, "--root", tmp_path)
    print(line)
    assert rc == 0 and v["system_inconsistent"] == "0" and v["episodes"] == "3"
    # 同一 MemER 路线里再放两局、其一改了 system：只有它一条与组内多数不同
    episode(tmp_path, "A_xhard1_4.a1", route=m, system=memer_sys)
    episode(tmp_path, "A_xhard1_5.a1", route=m, system=memer_sys + " (edited)")
    rc, v, line = _run(capsys, "--root", tmp_path)
    print(line)
    assert rc == 1 and v["system_inconsistent"] == "1"


def test_swapped_message_order_fails(tmp_path, capsys):
    d = episode(tmp_path)
    rows = _lang(d)
    rows[1], rows[2] = rows[2], rows[1]  # user 先于 system 写入
    _write(d / "language.jsonl", rows)
    rc, v, line = _run(capsys, d)
    print(line)
    assert rc == 1 and int(v["order_bad"]) >= 1


def test_extra_call_in_same_step_fails(tmp_path, capsys):
    d = episode(tmp_path)
    rows = _lang(d) + [
        {"kind": "call_open", "call_id": "a9", "model": "action_model", "step": 0, "retry": 0},
        {"kind": "message", "call_id": "a9", "message_index": 0, "dir": "in", "role": "fields", "text": {"prompt": "pick"}},
        {"kind": "call_close", "call_id": "a9", "status": "reply", "server_final_text": "Task: pick;"}]
    _write(d / "language.jsonl", rows)
    rc, v, line = _run(capsys, d)
    print(line)
    assert rc == 1 and v["extra_calls"] == "1"


def test_missing_reply_and_dangling_call_fail(tmp_path, capsys):
    d = episode(tmp_path)
    rows = [r for r in _lang(d) if not (r["kind"] == "message" and r.get("dir") == "out")]
    _write(d / "language.jsonl", rows)
    rc, v, line = _run(capsys, d)
    print(line)
    assert rc == 1 and v["reply_missing"] == "1"
    d2 = episode(tmp_path / "b")
    rows = [r for r in _lang(d2) if not (r["kind"] == "call_close" and r["call_id"] == "a1")]
    _write(d2 / "language.jsonl", rows)
    rc, v, _ = _run(capsys, d2)
    assert rc == 1 and v["open_calls"] == "1"


def test_language_from_wrong_attempt_fails(tmp_path, capsys):
    a1 = episode(tmp_path, "A_xhard1_1.a1")
    a2 = episode(tmp_path / "other", "A_xhard1_1.a2", tag="-a2")  # 第 2 次尝试的画面不同
    shutil.copy(a2 / "language.jsonl", a1 / "language.jsonl")      # 引用错 attempt
    rc, v, line = _run(capsys, a1)
    print(line)
    assert rc == 1 and v["image_ref_unresolved"] == "4"


def test_empty_server_final_text_fails(tmp_path, capsys):
    d = episode(tmp_path)
    rows = _lang(d)
    rows[-1]["server_final_text"] = ""  # 该路线有外壳（a0 带了），a1 为空
    _write(d / "language.jsonl", rows)
    rc, v, line = _run(capsys, d)
    print(line)
    assert rc == 1 and v["server_text_empty"] == "1"
    d2 = episode(tmp_path / "noshell", server_text=None)  # 无外壳路线：不要求
    assert _run(capsys, d2)[0] == 0
    rc, v, _ = _run(capsys, d2, "--require-server-text")
    assert rc == 1 and v["server_text_empty"] == "2"


def test_memer_retry_count_against_archived_log(tmp_path, capsys):
    d = episode(tmp_path)
    rows = _lang(d)
    retry = [{"kind": "call_open", "call_id": "c1", "model": "subgoal_model", "step": 0, "retry": 1},
             {"kind": "message", "call_id": "c1", "message_index": 0, "dir": "in", "role": "system", "text": SYS},
             {"kind": "message", "call_id": "c1", "message_index": 1, "dir": "out", "role": "assistant", "text": "x"},
             {"kind": "call_close", "call_id": "c1", "status": "reply"}]
    _write(d / "language.jsonl", rows[:5] + retry + rows[5:])
    log = d / "memer" / "ep0_MemER_log.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"messages": ["q"]}) + "\n" + json.dumps({"response": "a"}) + "\n", encoding="utf-8")
    rc, v, line = _run(capsys, d)
    print(line)
    assert rc == 1 and v["retry_mismatch"] == "1"  # 日志只记了 1 次提问，账本是 2 次
    log.write_text((json.dumps({"messages": ["q"]}) + json.dumps({"response": "a"}) + "\n") * 2, encoding="utf-8")
    assert _run(capsys, d)[0] == 0
    bad = rows[:5] + [dict(retry[0], retry=2)] + retry[1:] + rows[5:]  # retry 跳号
    _write(d / "language.jsonl", bad)
    rc, v, _ = _run(capsys, d)
    assert rc == 1 and v["retry_mismatch"] == "1"


def test_unresolved_step_reuse_and_missing_language(tmp_path, capsys):
    d = episode(tmp_path)
    trace = [json.loads(x) for x in (d / "trace.jsonl").read_text().splitlines()]
    for r in trace:
        if r.get("kind") == "step" and r["step"] == 4:
            r.pop("source_call_id")
    _write(d / "trace.jsonl", trace)
    rc, v, _ = _run(capsys, d)
    assert rc == 1 and v["unresolved_steps"] == "1"
    _write(d / "language.jsonl", _lang(d) + [{"kind": "reuse", "step": 3, "reused_call_id": "a1",
                                              "reused_previous": True}])
    assert _run(capsys, d)[0] == 0  # 复用步可追溯
    for r in trace:
        if r.get("kind") == "step" and r["step"] == 1:
            r["source_call_id"] = "a1"  # 指向之后（第 2 步）才发生的调用
    _write(d / "trace.jsonl", trace)
    rc, v, _ = _run(capsys, d)
    assert rc == 1 and v["unresolved_steps"] == "1"
    (d / "language.jsonl").unlink()
    rc, v, line = _run(capsys, d)
    print(line)
    assert rc == 1 and v["language_missing"] == "1" and v["unresolved_steps"] == "4"


def test_exec_frame0_resolves_to_demo_last_frame(tmp_path, capsys):
    """执行帧号口径：``phase=exec, frame_idx=0`` 是 reset 后初始帧 = demo 段末帧（前视、腕部各一）；k≥1 对 step k。
    真实 smoke（MemER 第 0 步动作调用附腕部帧）曾因检查器只在 step 行里找而误报。改成不存在的哈希、或帧号错位
    （哈希虽在 trace 里但不是那一帧）仍 FAIL。"""
    d = episode(tmp_path)
    rows = _lang(d)
    a0 = next(r for r in rows if r.get("call_id") == "a0" and r["kind"] == "message")
    a0["images"] = [{"slot": 0, "ref": "current", "phase": "exec", "frame_idx": 0, "cam": "front", "raw_sha256": "df1"},
                    {"slot": 1, "ref": "wrist", "phase": "exec", "frame_idx": 0, "cam": "wrist", "raw_sha256": "dw1"}]
    _write(d / "language.jsonl", rows)
    rc, v, line = _run(capsys, d)
    assert rc == 0 and v["image_ref_unresolved"] == "0", line
    # 不存在的哈希
    a0["images"][1]["raw_sha256"] = "nope"
    _write(d / "language.jsonl", rows)
    rc, v, _ = _run(capsys, d)
    assert rc == 1 and v["image_ref_unresolved"] == "1"
    # 哈希在 trace 里、但帧号错位：exec 帧 0 填了 demo 首帧、demo 帧 1 填了 step 1 的画面
    a0["images"][1]["raw_sha256"] = "dw0"
    c0 = next(r for r in rows if r.get("call_id") == "c0" and r.get("role") == "user")
    c0["images"][0]["raw_sha256"] = "f1"
    _write(d / "language.jsonl", rows)
    rc, v, _ = _run(capsys, d)
    assert rc == 1 and v["image_ref_unresolved"] == "2"


def test_frame_idx_exact_for_sources_and_sheet_fallback(tmp_path, capsys):
    """拼图来源帧（对象，带自身 phase／frame_idx）按各自帧号精确比对；拼图本身 ``frame_idx=None`` 的退回集合查找
    （exec 集合含演示段末帧）。"""
    d = episode(tmp_path)
    rows = _lang(d)
    a1 = next(r for r in rows if r.get("call_id") == "a1" and r["kind"] == "message")
    sheet = {"slot": 1, "ref": "memory_sheet", "phase": "exec", "frame_idx": None, "cam": "front",
             "raw_sha256": "df1",
             "sources": [{"phase": "demo", "frame_idx": 0, "cam": "front", "raw_sha256": "df0"},
                         {"phase": "exec", "frame_idx": 0, "cam": "front", "raw_sha256": "df1"},
                         {"phase": "exec", "frame_idx": 2, "cam": "front", "raw_sha256": "f2"}]}
    a1["images"].append(sheet)
    _write(d / "language.jsonl", rows)
    rc, v, line = _run(capsys, d)
    assert rc == 0 and v["image_ref_unresolved"] == "0", line
    sheet["sources"][2]["frame_idx"] = 3  # f2 不是第 3 步的画面
    _write(d / "language.jsonl", rows)
    rc, v, _ = _run(capsys, d)
    assert rc == 1 and v["image_ref_unresolved"] == "1"
