"""S7 报告类工具的手写夹具：按生产字段约定写轨迹与结果行（期望值直接写在各用例里，不读被测代码的常量）。

生产模块一律经 ``tests._support.loaders.load_script`` 按路径加载。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests._support.loaders import load_script


def tw():
    return load_script("eval-official/trace_writer.py")


def frame(step: int, cam: int = 0) -> np.ndarray:
    """确定性小画面（4×4×3 uint8），每步、每相机不同。"""
    return np.full((4, 4, 3), (step * 7 + cam * 3) % 251, dtype=np.uint8)


def state(step: int) -> np.ndarray:
    return np.arange(8, dtype=np.float32) + np.float32(step) * np.float32(0.01)


def action(step: int) -> np.ndarray:
    return np.arange(8, dtype=np.float32) * np.float32(0.5) + np.float32(step)


def flip_bit(a: np.ndarray, idx: int = 0) -> np.ndarray:
    """float32 数组第 idx 个元素的最低位翻转（动作改 1 bit）。"""
    b = np.array(a, dtype=np.float32, copy=True)
    v = b.view(np.uint32)
    v[idx] ^= np.uint32(1)
    return b


def write_episode(root: Path, *, task: str, source_episode: int, seed: int, n_steps: int = 6, attempt: int = 1,
                  status: str = "fail", subgoals: list[str] | None = None, mutate: dict | None = None,
                  route: str = "fixture") -> Path:
    """写一局 ``trace.jsonl``：演示 2 帧、每 3 步一次请求与动作块、逐步画面／状态／动作／子目标。

    ``mutate``：``{"action_bit": k}`` 第 k 步动作改 1 bit；``{"front": k}`` 第 k 步画面改 1 像素；
    ``{"state": k}``；``{"text": k}``；``{"request": k}`` 第 k 个请求字节改动；``{"stop_at": k}`` 第 k 步提前结束。"""
    m = mutate or {}
    t = tw()
    key = f"{task}_xhard0_{seed}"
    path = Path(root) / f"{key}.a{attempt}" / "trace.jsonl"
    ident = {"task": task, "source_episode": source_episode, "seed": seed, "tier": "xhard0", "dataset": "hard-verify",
             "attempt": attempt}
    n = int(m.get("stop_at", n_steps))
    with t.TraceWriter(path, route=route, identity=ident, max_steps=1300) as w:
        w.log_demo([frame(-2), frame(-1)], [frame(-2, 1), frame(-1, 1)], [state(-2), state(-1)], ["演示"])
        req_i = 0
        for s in range(1, n + 1):
            if (s - 1) % 3 == 0:
                payload = t.canonical_bytes({"step": s - 1, "img": frame(s - 1), "state": state(s - 1)})
                if m.get("request") == req_i:
                    payload = payload + b"!"
                w.log_request("infer", payload, step=s - 1)
                w.log_response(np.stack([action(s - 1 + j) for j in range(3)]), step=s - 1)
                if s > 1:
                    w.log_history(s - 3, s - 1)
                req_i += 1
            f = frame(s)
            if m.get("front") == s:
                f = f.copy()
                f[0, 0, 0] ^= 1
            a = action(s)
            if m.get("action_bit") == s:
                a = flip_bit(a)
            st = state(s)
            if m.get("state") == s:
                st = st.copy()
                st[3] += np.float32(1e-3)
            sg = (subgoals[min(s - 1, len(subgoals) - 1)] if subgoals else f"子目标{(s - 1) // 3}")
            if m.get("text") == s:
                sg = sg + "（改）"
            last = s == n
            w.log_step(step=s, front=f, wrist=frame(s, 1), state=st, action=a, subgoal=sg,
                       terminated=last, truncated=False, status=status if last else "ongoing")
        w.close(status=status, terminal_reason="env_terminated")
    return path


def result_row(*, task: str, source_episode: int, seed: int, status: str = "fail", attempt: int = 1,
               exec_steps: int = 6, **extra) -> dict:
    return {"task": task, "source_episode": source_episode, "seed": seed, "status": status, "attempt": attempt,
            "exec_steps": exec_steps, "canary": False, "infra": False, "late": False,
            "identity": {"tier": "xhard0", "seed": seed, "source_episode": source_episode},
            "dataset": "hard-verify", "max_steps": 1300, "effective_max_steps": 1300, "strict_cap": False, **extra}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path
