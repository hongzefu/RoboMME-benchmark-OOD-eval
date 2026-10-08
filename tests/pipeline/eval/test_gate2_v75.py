"""S6 第二档对比工具：跨格式投影、统计、``--orig-format v75`` 与 ``--noise-runs``（1005 计划第二部分一节 S6）。

期望值手写：投影「执行相同记录格式不同」零误报、「文本不同且动作错」仍报动作差异、双 ``NOT_OBSERVED`` 不算相同；
统计的 s2f／f2s／成功率／3×3 矩阵／McNemar p 按手算；E0 读前 sha256 不符报错；E0／O1／O2 三对两方向翻转。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evalx" / "report"))
import sgx_report_fixtures as F  # noqa: E402

IDS = [("VideoUnmask", 3, 101), ("VideoUnmask", 7, 102), ("PickXtimes", 3, 201), ("PickXtimes", 7, 202)]
GL = "gl1512.arc-ts.umich.edu"


def g2():
    return F.g2()


# ── 跨格式投影 ───────────────────────────────────────────────────────────────


def _write_cross(path: Path, *, side: str, n: int = 6, text_at: int | None = None, action_at: int | None = None):
    """同一执行、两种记录格式：

    * orig：动作按 float64 交给环境（记录 dtype ``<f8``，原值写 ``arrays.npz``）、状态 float64 形状 (1, 8)、
      只记逻辑输入请求 ``logic``、动作块 float64、``terminated``／``truncated`` 为 ``NOT_OBSERVED``；
    * new：动作 float32、状态 float32 形状 (8,)、除 ``logic`` 外另有协议帧请求 ``ws_frame``、动作块 float32。"""
    t = F.tw()
    ident = {"task": "X", "source_episode": 1, "seed": 9, "tier": "xhard0", "attempt": 1}
    f64 = side == "orig"
    arrays = {}
    with t.TraceWriter(path, route=f"smvla/{side}", identity=ident, max_steps=100) as w:
        st = [F.state(-1).astype(np.float64).reshape(1, 8) if f64 else F.state(-1)]
        w.log_demo([F.frame(-1)], [F.frame(-1, 1)], st, ["演示"])
        for s in range(1, n + 1):
            if (s - 1) % 3 == 0:
                w.log_request("logic", t.canonical_bytes({"instr": "do", "step": s - 1}), step=s - 1)
                if not f64:
                    w.log_request("ws_frame", b"frame-%d" % s, step=s - 1)
                blk = np.stack([F.action(s - 1 + j) for j in range(3)])
                w.log_response(blk.astype(np.float64) if f64 else blk, step=s - 1)
            a = F.action(s)
            if action_at == s:
                a = F.flip_bit(a)
            a = a.astype(np.float64) if f64 else a
            if f64:
                arrays[f"exec_action__{s - 1:05d}"] = a
            sv = F.state(s).astype(np.float64).reshape(1, 8) if f64 else F.state(s)
            sg = f"子目标{(s - 1) // 3}" + ("（改）" if text_at == s else "")
            flag_t = t.NOT_OBSERVED if f64 else (s == n)
            flag_u = t.NOT_OBSERVED if f64 else False
            w.log_step(step=s, front=F.frame(s), wrist=F.frame(s, 1), state=sv, action=a, subgoal=sg,
                       terminated=flag_t, truncated=flag_u, status="fail" if s == n else "ongoing")
        w.close(status="fail", terminal_reason="env_terminated")
    if arrays:
        np.savez(path.parent / "arrays.npz", **arrays)
    return path


def _proj(tmp_path, orig_kw=None, new_kw=None):
    g = g2()
    po = _write_cross(tmp_path / "o" / "X_xhard0_9.a1" / "trace.jsonl", side="orig", **(orig_kw or {}))
    pn = _write_cross(tmp_path / "n" / "X_xhard0_9.a1" / "trace.jsonl", side="new", **(new_kw or {}))
    return g.project_traces(g.tw.read_trace(po), g.tw.read_trace(pn), g._load_arrays(po), g._load_arrays(pn))


def test_same_execution_different_record_format_zero_false_positive(tmp_path):
    pj = _proj(tmp_path)
    assert {d: pj[d]["diff"] for d in pj} == {d: 0 for d in ("action", "obs", "state", "logic", "text", "stop")}
    assert pj["action"]["same"] == 6 and pj["action"]["not_observed"] == 0  # float64 原值无损转 float32 后字节相同
    assert pj["state"]["same"] == 7  # 演示 1 帧 + 6 步，float64(1,8) 对 float32(8,) 按值相同
    assert pj["obs"]["same"] == 2 + 12
    assert pj["text"]["same"] == 1 + 6
    # 逻辑输入：logic 请求 2 次相同；ws_frame 只新侧有 → 2 项不可观察；动作块 dtype 不同且元素数 24>8 → 2 项不可观察
    assert (pj["logic"]["same"], pj["logic"]["not_observed"]) == (2, 4)
    # 停止：terminated／truncated 原侧 NOT_OBSERVED → 12 项不可观察；status 6 步 + end 1 项相同
    assert (pj["stop"]["same"], pj["stop"]["not_observed"]) == (7, 12)


def test_text_diff_and_wrong_action_both_reported(tmp_path):
    pj = _proj(tmp_path, new_kw={"text_at": 3, "action_at": 4})
    assert (pj["text"]["diff"], pj["text"]["first_diff_step"]) == (1, 3)
    assert (pj["action"]["diff"], pj["action"]["first_diff_step"]) == (1, 4)
    assert pj["obs"]["diff"] == pj["state"]["diff"] == 0


def test_action_value_lossy_conversion_is_diff(tmp_path):
    # 原侧 float64 原值不可无损转成 float32：即使 f32hex 相同也必须判不同
    g = g2()
    v = np.arange(8, dtype=np.float64) + 1e-12
    rec32 = g.tw.array_record(v.astype(np.float32))
    assert g._cmp_array(g.tw.array_record(v), rec32, v, None) == "diff"
    assert g._cmp_array(g.tw.array_record(v.astype(np.float32).astype(np.float64)), rec32,
                        v.astype(np.float32).astype(np.float64), None) == "same"


def test_double_not_observed_is_not_same(tmp_path):
    g = g2()
    t = F.tw()
    paths = []
    for side in ("a", "b"):
        p = tmp_path / side / "trace.jsonl"
        with t.TraceWriter(p, route="x", identity={"task": "X", "source_episode": 1, "seed": 1}, max_steps=9) as w:
            w.log_demo([], [])
            w.log_missing_step(step=1, action=np.zeros(8, np.float32), reason="no_obs")
            w.close(status="error")
        paths.append(p)
    pj = g.project_traces(t.read_trace(paths[0]), t.read_trace(paths[1]))
    assert pj["obs"]["same"] == 0 and pj["obs"]["not_observed"] == 2
    assert pj["state"]["same"] == 0 and pj["state"]["not_observed"] == 1
    # terminated／truncated 双 NOT_OBSERVED、缺观测步 status 双 None → 3 项不可观察；end.status 相同 1 项
    assert (pj["stop"]["not_observed"], pj["stop"]["same"]) == (3, 1) and pj["action"]["same"] == 1
    assert g._cmp_scalar(t.NOT_OBSERVED, t.NOT_OBSERVED) == "not_observed"


def test_compare_ext_projection_does_not_stop_at_first_diff(tmp_path):
    g = g2()
    rows = {}
    for side, mut in (("orig", {}), ("new", {IDS[0]: {"text": 2, "action_bit": 5, "front": 6}})):
        root = tmp_path / side
        rr = []
        for task, ep, seed in IDS:
            F.write_episode(root, task=task, source_episode=ep, seed=seed, mutate=mut.get((task, ep, seed)))
            rr.append(F.result_row(task=task, source_episode=ep, seed=seed, host=GL))
        rows[side] = rr
    res = g.compare_ext(rows["orig"], rows["new"], tmp_path / "orig", tmp_path / "new", expect_total=4)
    s = res["summary"]
    assert s["verdict"] == "INFO" and s["compared"] == 4 and s["identical_trace"] == 3
    assert s["episodes_diff"] == {"action": 1, "obs": 1, "state": 0, "logic": 0, "text": 1, "stop": 0}
    row = next(r for r in res["table"] if (r["task"], r["source_episode"]) == IDS[0][:2])
    assert row["diverge_kind"] == "text" and row["diverge_step"] == 2  # 旧的首个分叉字段照旧
    assert row["projection"]["action"]["first_diff_step"] == 5 and row["projection"]["obs"]["first_diff_step"] == 6
    lines = g.ext_lines(res, "pp")
    assert "GATE2_PROJ policy=pp action=23/1/0 " in lines[-2]  # 4 局 × 6 步 = 24 项动作
    assert lines[-1].startswith("GATE2=INFO policy=pp compared=4 same_terminal=4 s2f=0 f2s=0 ")


# ── 统计 ─────────────────────────────────────────────────────────────────────


def test_mcnemar_exact_hand_values():
    g = g2()
    assert g.mcnemar_exact(0, 0) == 1.0
    assert g.mcnemar_exact(1, 1) == 1.0
    assert g.mcnemar_exact(5, 0) == pytest.approx(0.0625)  # 2 × 1/32
    assert g.mcnemar_exact(10, 3) == pytest.approx(2 * 378 / 8192)  # C(13,0..3)=1+13+78+286
    assert g.mcnemar_exact(3, 10) == g.mcnemar_exact(10, 3)


def test_terminal_stats_matrix_and_rates():
    g = g2()
    pairs = [("success", "success"), ("success", "fail"), ("success", "timeout"), ("fail", "success"),
             ("timeout", "timeout"), ("fail", "fail"), ("error", "success"), ("timeout", "fail")]
    st = g.terminal_stats(pairs)
    assert (st["compared"], st["s2f"], st["f2s"], st["flips"]) == (8, 2, 2, 4)
    assert st["sr_orig"] == pytest.approx(3 / 8) and st["sr_new"] == pytest.approx(3 / 8)
    assert st["sr_diff_pp"] == pytest.approx(0.0)
    assert st["matrix"] == {"success": {"success": 1, "fail": 1, "timeout": 1},
                            "fail": {"success": 1, "fail": 1, "timeout": 0},
                            "timeout": {"success": 0, "fail": 1, "timeout": 1}}
    assert st["other_terminal"] == 1
    assert g._matrix_str(st["matrix"]) == "ss:1,sf:1,st:1,fs:1,ff:1,ft:0,ts:0,tf:1,tt:1"


# ── v75：E0 读前核 sha256、ORIG_RERUN_VS_E0、噪声三对 ──────────────────────────

E0 = ["success", "success", "fail", "timeout"]
O1 = ["success", "fail", "fail", "success"]
O2 = ["fail", "success", "success", "timeout"]
NEW = ["success", "fail", "success", "timeout"]


def _v75_rows(statuses, shard=0):
    return [{"task": t, "source_episode": e, "seed": s, "status": st, "task_success": st == "success",
             "steps": 100 + i, "error": None, "max_steps": 1300, "shard": shard, "video": f"/x/{t}_{e}.mp4"}
            for i, ((t, e, s), st) in enumerate(zip(IDS, statuses))]


def _v75_setup(tmp_path):
    rows = _v75_rows(E0)
    e0a = F.write_jsonl(tmp_path / "e0" / "episodes-shard00of02.jsonl", rows[:2])
    e0b = F.write_jsonl(tmp_path / "e0" / "episodes-shard01of02.jsonl", rows[2:])
    man = tmp_path / "input-manifest.json"
    man.write_text(json.dumps({"schema": "v75-input-manifest/1", "files": [
        {"path": str(p), "group": "smvla-episodes", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (e0a, e0b)]}), encoding="utf-8")
    new = F.write_jsonl(tmp_path / "rerun" / "results.jsonl",
                        [dict(F.result_row(task=t, source_episode=e, seed=s, status=st), host=GL)
                         for (t, e, s), st in zip(IDS, NEW)])
    o1 = F.write_jsonl(tmp_path / "O1-smvla-official.jsonl", _v75_rows(O1))
    F.write_jsonl(tmp_path / "O2" / "s0" / "episodes-shard00of01.jsonl", _v75_rows(O2))
    return e0a, e0b, man, new, o1, tmp_path / "O2"


def test_v75_rerun_vs_e0_and_noise_pairs(tmp_path, capsys):
    e0a, e0b, man, new, o1, o2 = _v75_setup(tmp_path)
    md = tmp_path / "out" / "g.md"
    rc = g2().main(["--policy", "smvla", "--orig-results", str(e0a), str(e0b), "--new-results", str(new),
                    "--orig-format", "v75", "--v75-input-manifest", str(man), "--noise-runs", str(o1), str(o2),
                    "--expect-total", "4", "--out-md", str(md)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert lines[0] == "GATE2_INPUT_SHA=PASS files=2"
    assert "GATE2_PROVENANCE=PASS local_rows=0 unknown_rows=0" in lines  # E0 历史文件不核来源，只核新侧
    assert "ORIG_RERUN_VS_E0=INFO policy=smvla compared=4 flips=2 s2f=1 f2s=1" in lines
    noise = [x for x in lines if x.startswith("GATE2_NOISE=")]
    assert noise == [
        "GATE2_NOISE=INFO policy=smvla pair=E0-O1 compared=4 s2f=1 f2s=1 flips=2 sr_a=0.5000 sr_b=0.5000 missing=0 "
        "note=descriptive_history",
        "GATE2_NOISE=INFO policy=smvla pair=E0-O2 compared=4 s2f=1 f2s=1 flips=2 sr_a=0.5000 sr_b=0.5000 missing=0 "
        "note=descriptive_history",
        "GATE2_NOISE=INFO policy=smvla pair=O1-O2 compared=4 s2f=2 f2s=2 flips=4 sr_a=0.5000 sr_b=0.5000 missing=0 "
        "note=descriptive_history"]
    last = lines[-1]
    assert last.startswith("GATE2=INFO policy=smvla compared=4 same_terminal=2 s2f=1 f2s=1 sr_orig=0.5000 "
                           "sr_new=0.5000 sr_diff_pp=0.00 mcnemar_p=1 not_observed=0 identical_trace=0 missing=0 ")
    assert "matrix=ss:1,sf:1,st:0,fs:1,ff:0,ft:0,ts:0,tf:0,tt:1" in last and "missing_trace=0" in last
    assert "相当" not in "\n".join(lines) and "| PickXtimes | 3 | 201 | fail | success | 0 |" in md.read_text("utf-8")


def test_v75_sha_mismatch_or_unlisted_file_errors(tmp_path, capsys):
    e0a, e0b, man, new, _, _ = _v75_setup(tmp_path)
    with e0b.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    rc = g2().main(["--policy", "smvla", "--orig-results", str(e0a), str(e0b), "--new-results", str(new),
                    "--orig-format", "v75", "--v75-input-manifest", str(man)])
    out = capsys.readouterr().out
    assert rc == 2 and "GATE2_INPUT_SHA=FAIL" in out and "reason=sha256" in out and "GATE2=ERROR" in out
    stray = F.write_jsonl(tmp_path / "other.jsonl", _v75_rows(E0))
    rc = g2().main(["--policy", "smvla", "--orig-results", str(stray), "--new-results", str(new),
                    "--orig-format", "v75", "--v75-input-manifest", str(man)])
    out = capsys.readouterr().out
    assert rc == 2 and "reason=not_in_manifest" in out


def test_noise_runs_require_v75(tmp_path, capsys):
    e0a, _, _, new, o1, o2 = _v75_setup(tmp_path)
    rc = g2().main(["--policy", "smvla", "--orig-results", str(e0a), "--new-results", str(new),
                    "--noise-runs", str(o1), str(o2)])
    assert rc == 2 and "noise_runs_need_orig_format_v75" in capsys.readouterr().out


REAL_MANIFEST = Path("/data/hongzefu/robomme_benchmark_MotionJEPANewTask/artifacts/v7.5eval/input-manifest.json")
REAL_E0 = Path("/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v7-eval/smvla-official-full")


@pytest.mark.skipif(not (REAL_MANIFEST.is_file() and REAL_E0.is_dir()), reason="无真实 E0 与 input-manifest（只读核对）")
def test_real_e0_smvla_files_match_input_manifest():
    """只读：真实 E0 SimpleMemVLA 10 个分片文件 sha256 与 v7.5eval 输入清单一致，读出 192 个身份。"""
    g = g2()
    files = sorted(REAL_E0.glob("episodes-shard*.jsonl"))
    checked = g.verify_v75_inputs(files, REAL_MANIFEST)
    assert len(checked) == len(files) == 10
    rows = g._v75_normalize(g.read_rows(files))
    assert len(g.final_rows(rows)[0]) == 192


# ── GroundSG：诊断列，192 局全比 ──────────────────────────────────────────────


def test_groundsg_diagnostic_column_compares_all(tmp_path):
    g = g2()
    rows = {}
    for side in ("orig", "new"):
        rr = []
        for task, ep, seed in IDS:
            F.write_episode(tmp_path / side, task=task, source_episode=ep, seed=seed)
            rr.append(dict(F.result_row(task=task, source_episode=ep, seed=seed, host=GL),
                           server_epoch=0 if task == "VideoUnmask" else 1))
        rows[side] = g.read_rows([F.write_jsonl(tmp_path / side / "r.jsonl", rr)])
    res = g.compare_ext(rows["orig"], rows["new"], tmp_path / "orig", tmp_path / "new", groundsg=True)
    s = res["summary"]
    assert s["compared"] == 4 and s["identical_trace"] == 4 and s["server_epoch_first"] == 2
    firsts = {(r["task"], r["source_episode"]): r["server_epoch_first"] for r in res["table"]}
    assert firsts == {("VideoUnmask", 3): True, ("VideoUnmask", 7): False, ("PickXtimes", 3): True,
                      ("PickXtimes", 7): False}
    assert "first_episode_identical" not in s
    assert "server_epoch_first=2" in g.ext_lines(res, "groundsg-oracle")[-1]


def test_v75_superseded_infra_error_rows_are_not_terminal():
    """原版 3 遍重评：前面的 reset 失败 error 行被同身份后面的正常终态取代时不计终态；没被取代的 error 行仍是终态。
    （2026-10-06 实测：O2 的 SimpleMemVLA 有 76 行 Vulkan reset 失败行，误计终态时 E0-O2 出 63 局假翻转。）"""
    m = g2()
    rows = [
        {"task": "A", "source_episode": 3, "seed": 1, "status": "error", "steps": 0, "_file": "f", "_line": 0},
        {"task": "B", "source_episode": 7, "seed": 2, "status": "error", "steps": 0, "_file": "f", "_line": 1},
        {"task": "A", "source_episode": 3, "seed": 1, "status": "success", "steps": 10, "_file": "f", "_line": 2},
    ]
    finals, _ = m.final_rows(m._v75_normalize(rows))
    assert {k: v["status"] for k, v in finals.items()} == {m.ident_of(rows[0]): "success", m.ident_of(rows[1]): "error"}
    assert rows[0]["infra"] is True and "infra" not in rows[1] and rows[2]["exec_steps"] == 10
