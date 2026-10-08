"""C13-SG-TRACE-ARRAYS／C13-SG-LANGUAGE-LOG：第三阶段完整数值与语言账本接口（冻结说明第四、五节，R6）。

期望一律手写：数组按原始字节用 hashlib 直接算，行 schema 照冻结说明逐字段列出，不调用被测代码生成期望。
"""
from __future__ import annotations

import collections
import hashlib
import json
import threading

import numpy as np
import pytest

from tests._support.loaders import load_script


def _tw():
    return load_script("eval-official/trace_writer.py")


def _chk():
    return load_script("eval-official/trace_arrays_check.py")


def _img(k: int, cam: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), (k * 7 + cam * 3) % 251, dtype=np.uint8)


def _st(k: int) -> np.ndarray:
    return np.arange(8, dtype=np.float32) + np.float32(k)


def _act(k: int, dtype=np.float64) -> np.ndarray:
    return (np.arange(8) * 0.25 + k).astype(dtype)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _lines(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _episode(ep_dir, *, n=4, missing=(), dtypes=None, **tw_kw):
    """用 TraceWriter 写一局：n 步，``missing`` 里的步号为缺观测步；返回逐步动作。"""
    t = _tw()
    w = t.TraceWriter(ep_dir / "trace.jsonl", route="r6/new", identity={"task": "T", "key": "K"}, max_steps=10, **tw_kw)
    w.log_demo([_img(0)], [_img(0, 1)], [_st(0)], ["目标"])
    acts = []
    for s in range(1, n + 1):
        a = _act(s, (dtypes or {}).get(s, np.float64))
        acts.append(a)
        if s in missing:
            w.log_missing_step(step=s, action=a, reason="env_exception")
        else:
            w.log_step(step=s, front=_img(s), wrist=_img(s, 1), state=_st(s), action=a, subgoal=None,
                       terminated=False, truncated=False, status=None)
    w.close(status="fail", terminal_reason="fail")
    return acts


# ---------------------------------------------------------------- LanguageLog


LANG_KEYS = {
    "call_open": ["kind", "call_id", "model", "step", "params", "transport_attempt", "retry", "ts"],
    "message": ["kind", "call_id", "message_index", "dir", "role", "text", "images", "channel", "token_ids", "mask",
                "tokenizer", "truncated", "demo_video", "ts"],
    "call_close": ["kind", "call_id", "status", "parsed", "fallback", "server_final_text", "server_truncated", "ts"],
    "reuse": ["kind", "step", "reused_call_id", "reused_previous"],
}


def test_language_log_open_message_close_schema(tmp_path):
    t = _tw()
    p = tmp_path / "ep" / "language.jsonl"
    lang = t.LanguageLog(p)
    cid = lang.open_call("subgoal_model", 3, params={"temperature": 0.0}, retry=1)
    assert lang.message(cid, dir="in", role="system", text="你是规划器") == 0
    img = {"slot": 0, "ref": "current", "phase": "exec", "frame_idx": 3, "cam": "front", "raw_sha256": "ab",
           "sources": None, "transform": None, "encoded_sha256": None}
    assert lang.message(cid, dir="in", role="user", text="目标：叠方块", images=[img]) == 1
    assert lang.message(cid, dir="out", role="assistant", text="pick the cube") == 2
    lang.close_call(cid, status="reply", parsed="pick the cube", fallback="last_valid")
    lang.reuse(4, cid)
    cid2 = lang.open_call("action_model", 4)
    lang.message(cid2, dir="in", role="fields", text={"prompt": "叠方块"}, channel="task",
                 token_ids=np.array([1, 2, 3]), mask=[True, True, False], tokenizer="paligemma", truncated=False)
    lang.close_call(cid2, status="error")
    lang.close()
    rows = _lines(p)
    assert [r["kind"] for r in rows] == ["call_open", "message", "message", "message", "call_close", "reuse",
                                        "call_open", "message", "call_close"]
    for r in rows:
        assert list(r) == LANG_KEYS[r["kind"]], r["kind"]
    assert cid != cid2 and rows[0]["call_id"] == cid and rows[6]["call_id"] == cid2
    assert (rows[0]["model"], rows[0]["step"], rows[0]["params"], rows[0]["retry"], rows[0]["transport_attempt"]) == \
        ("subgoal_model", 3, {"temperature": 0.0}, 1, 0)
    assert [r["message_index"] for r in rows[1:4]] == [0, 1, 2] and rows[2]["images"] == [img]
    assert rows[1]["text"] == "你是规划器" and "你是规划器" in p.read_text(encoding="utf-8")  # ensure_ascii=False
    assert rows[4]["status"] == "reply" and rows[4]["fallback"] == "last_valid"
    assert rows[5] == {"kind": "reuse", "step": 4, "reused_call_id": cid, "reused_previous": True}
    assert rows[7]["token_ids"] == [1, 2, 3] and rows[7]["channel"] == "task" and rows[8]["status"] == "error"


def test_language_log_in_message_is_on_disk_before_send(tmp_path):
    """``dir=in`` 写入即 flush：模拟「发送时进程崩溃」——不关文件，另开句柄读到完整输入行。"""
    t = _tw()
    p = tmp_path / "language.jsonl"
    lang = t.LanguageLog(p)
    cid = lang.open_call("planner", 0)
    lang.message(cid, dir="in", role="user", text="CURRENT REQUEST …")
    rows = _lines(p)  # 未 close
    assert [r["kind"] for r in rows] == ["call_open", "message"] and rows[1]["text"] == "CURRENT REQUEST …"
    lang.close()


def test_language_log_close_cancels_open_calls_and_rejects_late_writes(tmp_path):
    t = _tw()
    p = tmp_path / "language.jsonl"
    lang = t.LanguageLog(p)
    a = lang.open_call("monitor", 1)
    b = lang.open_call("action_model", 1)
    lang.close_call(b, status="reply")
    lang.close()
    lang.close()  # 幂等
    rows = _lines(p)
    closes = [r for r in rows if r["kind"] == "call_close"]
    assert [(r["call_id"], r["status"]) for r in closes] == [(b, "reply"), (a, "cancelled")]
    with pytest.raises(RuntimeError):
        lang.open_call("monitor", 2)
    lang2 = t.LanguageLog(tmp_path / "l2.jsonl")
    c = lang2.open_call("monitor", 0)
    lang2.close_call(c, status="reply")
    with pytest.raises(ValueError):
        lang2.message(c, dir="in", role="user", text="迟到")  # 已关闭的调用
    with pytest.raises(ValueError):
        lang2.open_call("vla", 0)  # 未知 model
    with pytest.raises(ValueError):
        lang2.reuse(1, "c99999")  # 不存在的调用
    lang2.close()


# ---------------------------------------------------------------- merge_write_npz


def test_merge_write_npz_merges_consistent_keys(tmp_path):
    t = _tw()
    p = tmp_path / "arrays.npz"
    a0, a1, s0 = _act(1), _act(2), _st(1)
    t.merge_write_npz(p, {"exec_action__00000": a0, "model_action__00000": np.zeros((2, 8), np.float32)})
    t.merge_write_npz(p, {"exec_action__00000": a0.copy(), "exec_action__00001": a1, "exec_state__00000": s0})
    with np.load(p) as z:
        assert sorted(z.files) == ["exec_action__00000", "exec_action__00001", "exec_state__00000",
                                   "model_action__00000"]
        assert _sha(z["exec_action__00000"]) == _sha(a0) and z["exec_action__00001"].dtype == np.float64
        assert _sha(z["exec_state__00000"]) == _sha(s0)
    assert not (tmp_path / "arrays.npz.tmp").exists()
    t.merge_write_npz(tmp_path / "empty.npz", {})
    assert not (tmp_path / "empty.npz").exists()  # 空映射不建文件


@pytest.mark.parametrize("bad", ["dtype", "shape", "bytes"])
def test_merge_write_npz_conflict_raises_and_leaves_file_untouched(tmp_path, bad):
    t = _tw()
    p = tmp_path / "arrays.npz"
    a = _act(1)
    t.merge_write_npz(p, {"exec_action__00000": a})
    before = p.read_bytes()
    other = {"dtype": a.astype(np.float32), "shape": a.reshape(2, 4), "bytes": a + 1e-12}[bad]
    with pytest.raises(t.ArraysConflict):
        t.merge_write_npz(p, {"exec_action__00001": _act(2), "exec_action__00000": other})
    assert p.read_bytes() == before  # 冲突时一个字节都不写（新键也不落）


def test_merge_write_npz_is_atomic_when_write_fails(tmp_path, monkeypatch):
    t = _tw()
    p = tmp_path / "arrays.npz"
    t.merge_write_npz(p, {"exec_action__00000": _act(1)})
    before = p.read_bytes()

    def boom(fh, **kw):
        fh.write(b"PK\x03\x04half")
        raise OSError("磁盘写满")

    monkeypatch.setattr(t.np, "savez", boom)
    with pytest.raises(OSError):
        t.merge_write_npz(p, {"exec_action__00001": _act(2)})
    assert p.read_bytes() == before and not (tmp_path / "arrays.npz.tmp").exists()


# ---------------------------------------------------------------- TraceWriter 收集


def test_trace_writer_collects_actions_and_states_without_zero_fill(tmp_path):
    ep = tmp_path / "K.a1"
    acts = _episode(ep, n=4, missing=(3,))
    rows = _lines(ep / "trace.jsonl")
    end = rows[-1]
    assert end["arrays"] == {"path": "arrays.npz", "action_keys": 4, "state_keys": 3, "missing_state_steps": [3]}
    with np.load(ep / "arrays.npz") as z:
        assert sorted(z.files) == [f"exec_action__{i:05d}" for i in range(4)] + \
            [f"exec_state__{i:05d}" for i in (0, 1, 3)]  # 缺观测步 3 没有 exec_state__00002
        for i, a in enumerate(acts):
            assert z[f"exec_action__{i:05d}"].dtype == np.float64 and _sha(z[f"exec_action__{i:05d}"]) == _sha(a)
        assert _sha(z["exec_state__00003"]) == _sha(_st(4))
    steps = [r for r in rows if r["kind"] == "step"]
    assert all("source_call_id" not in r and "chunk_index" not in r for r in steps)
    assert "policy_seed" not in rows[0] and "effective_cap" not in rows[0] and "policy_seed" not in end


def test_trace_writer_seed_cap_and_call_links(tmp_path):
    t = _tw()
    w = t.TraceWriter(tmp_path / "trace.jsonl", route="r", identity={"policy_seed": 7}, max_steps=1800,
                      policy_seed=7, effective_cap=1800, collect_arrays=False)
    w.log_step(step=1, front=_img(1), wrist=_img(1, 1), state=_st(1), action=_act(1), subgoal=None,
               terminated=False, truncated=False, status=None, source_call_id="c00001", chunk_index=0)
    w.log_missing_step(step=2, action=_act(2), reason="obs_none", source_call_id="c00001", chunk_index=1)
    w.close(status="timeout", terminal_reason="timeout")
    rows = _lines(tmp_path / "trace.jsonl")
    assert rows[0]["policy_seed"] == 7 and rows[0]["effective_cap"] == 1800
    assert [(r["source_call_id"], r["chunk_index"]) for r in rows if r["kind"] == "step"] == [("c00001", 0),
                                                                                             ("c00001", 1)]
    assert rows[-1]["policy_seed"] == 7 and "arrays" not in rows[-1]
    assert not (tmp_path / "arrays.npz").exists()  # collect_arrays=False 不写


# ---------------------------------------------------------------- 录像器与 trace 同目录，先后任意


def _bare_recorder(out_dir):
    """真实 recorder.EpisodeRecorder 的 add_array／_write_arrays（不起 ffmpeg 写线程）。"""
    R = load_script("eval-official/recorder.py")
    rec = object.__new__(R.EpisodeRecorder)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.out_dir = out_dir
    rec._lock = threading.RLock()
    rec._arrays = []
    rec._array_counts = collections.defaultdict(int)
    rec._writer_error = None
    rec._writer = None
    rec._closed = False
    return rec


@pytest.mark.parametrize("order", ["recorder_first", "trace_first"])
def test_recorder_and_trace_same_dir_any_close_order(tmp_path, order):
    ep = tmp_path / "K.a1"
    rec = _bare_recorder(ep)
    for s in range(1, 4):  # 环境侧同一动作原值（与 EnvSession.step 的 np.array(action) 同）
        rec.add_array("exec_action", _act(s), step=s - 1)
        rec.add_array("joint_state", np.arange(7.0) + s, step=s - 1)
    if order == "recorder_first":
        rec._write_arrays()
        _episode(ep, n=3)
    else:
        _episode(ep, n=3)
        rec._write_arrays()
    with np.load(ep / "arrays.npz") as z:
        files = set(z.files)
    assert files == {f"exec_action__{i:05d}" for i in range(3)} | {f"exec_state__{i:05d}" for i in range(3)} | \
        {f"joint_state__{i:05d}" for i in range(3)}
    assert _lines(ep / "trace.jsonl")[-1]["arrays"].get("error") is None
    assert (ep / "arrays-index.jsonl").exists()


def test_recorder_conflicting_same_key_raises(tmp_path):
    ep = tmp_path / "K.a1"
    _episode(ep, n=2)
    before = (ep / "arrays.npz").read_bytes()
    rec = _bare_recorder(ep)
    rec.add_array("exec_action", _act(1).astype(np.float32))  # 同键不同 dtype
    with pytest.raises(ValueError) as ei:  # recorder 按路径另载 trace_writer，类对象不同，按类名核
        rec._write_arrays()
    assert type(ei.value).__name__ == "ArraysConflict" and (ep / "arrays.npz").read_bytes() == before


# ---------------------------------------------------------------- trace_arrays_check.py


def _run_check(capsys, *roots):
    rc = _chk().main([str(r) for r in roots])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    return rc, line


def _verdict(line: str) -> dict:
    head, *rest = line.split()
    return {"verdict": head.split("=")[1], **{k: int(v) for k, v in (x.split("=") for x in rest)}}


def test_check_positive_and_fixture_verdict_line(tmp_path, capsys):
    _episode(tmp_path / "run" / "K.a1", n=4, missing=(2,))
    _episode(tmp_path / "run" / "K2.a1", n=3)
    rc, line = _run_check(capsys, tmp_path / "run")
    print(line)
    assert rc == 0 and line.startswith("TRACE_ARRAYS=PASS episodes=2 attempted_steps_missing=0 observed_state_missing=0")
    v = _verdict(line)
    assert v["tampered"] == v["dtype_mixed"] == v["unreadable"] == v["summary_mismatch"] == 0


def test_check_negative_float32_float64_mixed(tmp_path, capsys):
    _episode(tmp_path / "K.a1", n=3, dtypes={2: np.float32})
    rc, line = _run_check(capsys, tmp_path)
    v = _verdict(line)
    assert rc == 1 and v["verdict"] == "FAIL" and v["dtype_mixed"] == 1


def test_check_negative_split_dir_arrays_only_in_recorder_dir(tmp_path, capsys):
    """分目录：trace 在 trace_dir，录像器在别处；只有录像器目录有数组时 trace 局判缺（共目录合并则 PASS）。"""
    tr, rd = tmp_path / "trace" / "K.a1", tmp_path / "rec" / "K.a1"
    _episode(tr, n=3)
    rec = _bare_recorder(rd)
    for s in range(1, 4):
        rec.add_array("exec_action", _act(s))
    rec._write_arrays()
    rc, line = _run_check(capsys, tmp_path / "trace")
    assert rc == 0 and _verdict(line)["verdict"] == "PASS"  # 分目录各写各的：trace 目录自有完整数组
    (tr / "arrays.npz").unlink()  # 反例：数组只在录像器目录
    rc, line = _run_check(capsys, tmp_path / "trace")
    v = _verdict(line)
    assert rc == 1 and v["attempted_steps_missing"] == 3 and v["observed_state_missing"] == 3 and v["unreadable"] == 1
    shared = tmp_path / "shared" / "K.a1"  # 正例：共目录合并
    rec2 = _bare_recorder(shared)
    for s in range(1, 4):
        rec2.add_array("exec_action", _act(s))
    _episode(shared, n=3)
    rec2._write_arrays()
    rc, line = _run_check(capsys, tmp_path / "shared")
    assert rc == 0 and _verdict(line)["verdict"] == "PASS"


def test_check_negative_observed_state_missing_and_zero_fill(tmp_path, capsys):
    ep = tmp_path / "K.a1"
    _episode(ep, n=3, missing=(2,))
    with np.load(ep / "arrays.npz") as z:
        d = {k: z[k] for k in z.files}
    d2 = {k: v for k, v in d.items() if k != "exec_state__00002"}  # 观测步 3 的状态被删
    np.savez(ep / "arrays.npz", **d2)
    rc, line = _run_check(capsys, tmp_path)
    assert rc == 1 and _verdict(line)["observed_state_missing"] == 1
    d3 = dict(d, exec_state__00001=np.zeros(8, np.float32))  # 给缺观测步 2 补零
    np.savez(ep / "arrays.npz", **d3)
    rc, line = _run_check(capsys, tmp_path)
    assert rc == 1 and _verdict(line)["tampered"] == 1


def test_check_negative_tampered_bytes(tmp_path, capsys):
    ep = tmp_path / "K.a1"
    _episode(ep, n=3)
    with np.load(ep / "arrays.npz") as z:
        d = {k: z[k] for k in z.files}
    a = d["exec_action__00001"].copy()
    a.view(np.uint8)[0] ^= 1  # 改 1 bit
    d["exec_action__00001"] = a
    np.savez(ep / "arrays.npz", **d)
    rc, line = _run_check(capsys, tmp_path)
    v = _verdict(line)
    assert rc == 1 and v["tampered"] == 1 and v["attempted_steps_missing"] == 0


def test_check_empty_root_fails(tmp_path, capsys):
    rc, line = _run_check(capsys, tmp_path)
    assert rc == 1 and line.startswith("TRACE_ARRAYS=FAIL episodes=0")
