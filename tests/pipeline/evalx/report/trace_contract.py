"""第二阶段共享契约 C1～C11 的测试助手（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分〇节，S0）。

各路线子任务（S1、S3、S4、S5、S7）的测试对自己写出的局目录调用：

- ``assert_renderable(ep_dir)``：按契约逐条核对 ``trace.jsonl``（及 ``arrays.npz``）；有帧的局再交给重绘工具
  ``render_official_video.load_trace`` 读一遍，确认官方版式视频能从这份记录画出来。无帧 ``error`` 局
  （``end.no_frame=true``）只核契约，不调重绘器。
- ``assert_counts_consistent(ep_dir, result_row)``：C8 三分计数与结果行 ``exec_steps`` 对账。

契约正文在 ``scripts/eval-official/trace_writer.py`` 模块文档串；这里的判据只依据那份正文，不读被测路线的常量。
生产模块一律经 ``tests._support.loaders.load_script`` 按路径加载。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tests._support.loaders import load_script

TERMINALS = ("success", "fail", "timeout", "error")
NEW_ROUTES = re.compile(r"^(groundsg/[A-Za-z0-9_.-]+|pp|astra|smvla|perceptual-framesamp-modul)/(new|orig)$")
EP_DIR = re.compile(r"^(?P<key>.+)\.a(?P<attempt>\d+)$")
NOT_OBSERVED = "NOT_OBSERVED"


def _tw():
    return load_script("eval-official/trace_writer.py")


def _render():
    return load_script("eval-official/render_official_video.py")


def _fail(ep_dir: Path, msg: str) -> None:
    raise AssertionError(f"{ep_dir}: {msg}")


def contract_problems(ep_dir: str | Path) -> list[str]:
    """返回契约问题列表（空即合规）；``assert_renderable`` 的纯核对部分，便于反例测试逐条断言。"""
    ep_dir = Path(ep_dir)
    tpath = ep_dir / "trace.jsonl"
    if not tpath.is_file():
        return ["缺 trace.jsonl（C5）"]
    rows = _tw().read_trace(tpath)
    problems = list(_tw().validate_trace(rows))
    if problems:
        return problems
    header, end = rows[0], rows[-1]
    demo = next(r for r in rows if r["kind"] == "demo")
    steps = [r for r in rows if r["kind"] == "step"]

    route = header.get("route")
    if not isinstance(route, str) or not NEW_ROUTES.match(route):
        problems.append(f"C1 route 不合规：{route!r}")

    ident = header.get("identity") or {}
    for k in ("task", "tier", "seed", "dataset", "key", "attempt"):
        if ident.get(k) is None:
            problems.append(f"C6 identity 缺 {k}")
    if ident.get("source_episode") is None and ident.get("builder_episode") is None:
        problems.append("C6 identity 缺 source_episode／builder_episode")
    m = EP_DIR.match(ep_dir.name)
    if m and ident.get("key") is not None:
        if str(ident["key"]) != m["key"]:
            problems.append(f"C6 identity.key={ident['key']!r} 与目录名 {ep_dir.name!r} 不一致")
        if ident.get("attempt") is not None and int(ident["attempt"]) != int(m["attempt"]):
            problems.append(f"C6 identity.attempt={ident['attempt']} 与目录名 {ep_dir.name!r} 不一致")

    status, reason = end.get("status"), end.get("terminal_reason")
    if status not in TERMINALS or reason not in TERMINALS:
        problems.append(f"C3 终态不合规：status={status!r} terminal_reason={reason!r}")
    elif status != reason and (reason, status) != ("error", "timeout"):
        problems.append(f"C3 终态冲突：status={status!r} terminal_reason={reason!r}")
    no_frame = bool(end.get("no_frame"))
    if no_frame and status != "error":
        problems.append("C3 只有 error 局允许 no_frame")

    demo_frames = end.get("demo_frames")
    if type(demo_frames) is not int or demo_frames < 0:
        problems.append(f"C2 end.demo_frames 缺失或非法：{demo_frames!r}")
    elif not no_frame:
        if demo.get("frames") != demo_frames + 1:
            problems.append(f"C2 demo.frames={demo.get('frames')} != end.demo_frames+1={demo_frames + 1}")
        if len(demo.get("states") or []) != demo.get("frames"):
            problems.append("C2 演示状态数与帧数不等")
        texts = demo.get("texts")
        if not isinstance(texts, list) or len(texts) != 1 or not str(texts[0]).strip():
            problems.append("C2 demo.texts 必须是唯一非空任务目标")

    for st in steps:
        for k in ("terminated", "truncated"):
            if not isinstance(st.get(k), bool) and st.get(k) != NOT_OBSERVED:
                problems.append(f"C9 第 {st['step']} 步 {k} 只能是布尔或 NOT_OBSERVED")
        if not isinstance(st.get("subgoal"), (str, type(None))):
            problems.append(f"C7 第 {st['step']} 步 subgoal 必须为文本或 None")
        if st.get("action") is None:
            problems.append(f"C4 第 {st['step']} 步缺动作")
        if st.get("observed") is False:
            if not st.get("missing_reason"):
                problems.append(f"C8 第 {st['step']} 步缺观测但没写原因")
        elif not st.get("front_sha256") or not st.get("wrist_sha256") or st.get("state") is None:
            problems.append(f"C4 第 {st['step']} 步缺画面或状态（缺观测步须 observed=false）")

    observed = sum(1 for st in steps if st.get("observed") is not False)
    for k in ("steps_attempted", "steps_observed", "frames_recorded"):
        if type(end.get(k)) is not int:
            problems.append(f"C8 end 缺 {k}")
    if not any(p.startswith("C8 end 缺") for p in problems):
        omitted = int(end.get("omitted_timeout_frames", 0) or 0)
        if end["steps_attempted"] != len(steps) or end["steps_attempted"] != end.get("exec_steps"):
            problems.append(f"C8 steps_attempted={end['steps_attempted']} 与步行数 {len(steps)}／exec_steps 不符")
        if end["steps_observed"] != observed:
            problems.append(f"C8 steps_observed={end['steps_observed']} 与有效观测步 {observed} 不符")
        if not no_frame and type(demo_frames) is int:
            expect = demo_frames + 1 + end["steps_observed"] - omitted
            if end["frames_recorded"] != expect:
                problems.append(f"C8 frames_recorded={end['frames_recorded']} != {expect}")
        if no_frame and end["frames_recorded"] != 0:
            problems.append("C8 no_frame 局 frames_recorded 必须为 0")

    if "observer_hook_errors" in end and (type(end["observer_hook_errors"]) is not int or end["observer_hook_errors"] < 0):
        problems.append("C11 observer_hook_errors 必须为非负整数")

    arrays_path = ep_dir / "arrays.npz"
    non_f32 = [st for st in steps if st.get("action") and np.dtype(st["action"]["dtype"]) != np.dtype("<f4")]
    if non_f32 and not arrays_path.is_file():
        problems.append("C4 有非 float32 动作但缺 arrays.npz")
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as arr:
            keys = set(arr.files)
            for st in steps:
                key = f"exec_action__{st['step'] - 1:05d}"
                if key not in keys:
                    problems.append(f"C4 arrays.npz 缺 {key}")
                    continue
                a = arr[key]
                rec = st["action"]
                if a.dtype.str != rec["dtype"] or list(a.shape) != rec["shape"]:
                    problems.append(f"C4 {key} dtype/shape 与 trace 不符")
    return problems


def assert_renderable(ep_dir: str | Path) -> None:
    """契约核对 + 有帧局交重绘工具 ``load_trace`` 读一遍。"""
    ep_dir = Path(ep_dir)
    problems = contract_problems(ep_dir)
    if problems:
        _fail(ep_dir, "；".join(problems))
    end = _tw().read_trace(ep_dir / "trace.jsonl")[-1]
    if end.get("no_frame"):
        return
    try:
        _render().load_trace(ep_dir / "trace.jsonl")
    except Exception as e:  # 重绘器拒收即不可渲染
        _fail(ep_dir, f"重绘工具 load_trace 拒收：{type(e).__name__}: {e}")


def assert_counts_consistent(ep_dir: str | Path, result_row: dict) -> None:
    """C8：``end`` 三分计数自洽，且 ``steps_attempted`` 等于结果行 ``exec_steps``、终态与结果行一致。"""
    ep_dir = Path(ep_dir)
    rows = _tw().read_trace(ep_dir / "trace.jsonl")
    end = rows[-1]
    problems = [p for p in contract_problems(ep_dir) if p.startswith("C8")]
    if problems:
        _fail(ep_dir, "；".join(problems))
    if "exec_steps" in result_row and end["steps_attempted"] != int(result_row["exec_steps"]):
        _fail(ep_dir, f"steps_attempted={end['steps_attempted']} != 结果行 exec_steps={result_row['exec_steps']}")
    if "status" in result_row and result_row["status"] != end["status"]:
        _fail(ep_dir, f"结果行 status={result_row['status']!r} 与 trace end.status={end['status']!r} 不符")
