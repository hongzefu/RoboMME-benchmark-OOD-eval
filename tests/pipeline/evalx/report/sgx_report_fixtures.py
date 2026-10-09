"""Hand-written fixtures for the S7 report tools: write traces and result rows following production field conventions
(expected values live in each test case; constants of the code under test are not read).

Production modules are always loaded by path via ``tests._support.loaders.load_script``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests._support.loaders import load_script


def tw():
    return load_script("eval-official/trace_writer.py")


def frame(step: int, cam: int = 0) -> np.ndarray:
    """Small deterministic frame (4x4x3 uint8), different per step and per camera."""
    return np.full((4, 4, 3), (step * 7 + cam * 3) % 251, dtype=np.uint8)


def state(step: int) -> np.ndarray:
    return np.arange(8, dtype=np.float32) + np.float32(step) * np.float32(0.01)


def action(step: int) -> np.ndarray:
    return np.arange(8, dtype=np.float32) * np.float32(0.5) + np.float32(step)


def flip_bit(a: np.ndarray, idx: int = 0) -> np.ndarray:
    """Flip the lowest bit of element idx of a float32 array (a 1-bit action change)."""
    b = np.array(a, dtype=np.float32, copy=True)
    v = b.view(np.uint32)
    v[idx] ^= np.uint32(1)
    return b


def write_episode(root: Path, *, task: str, source_episode: int, seed: int, n_steps: int = 6, attempt: int = 1,
                  status: str = "fail", subgoals: list[str] | None = None, mutate: dict | None = None,
                  route: str = "fixture") -> Path:
    """Write one episode's ``trace.jsonl``: 2 demo frames, a request and action chunk every 3 steps, and per-step
    frames / state / action / subgoal.

    ``mutate``: ``{"action_bit": k}`` flips 1 bit of the action at step k; ``{"front": k}`` changes 1 pixel of the
    frame at step k; ``{"state": k}``; ``{"text": k}``; ``{"request": k}`` changes the bytes of request k;
    ``{"stop_at": k}`` ends early at step k."""
    m = mutate or {}
    t = tw()
    key = f"{task}_xhard0_{seed}"
    path = Path(root) / f"{key}.a{attempt}" / "trace.jsonl"
    ident = {"task": task, "source_episode": source_episode, "seed": seed, "tier": "xhard0", "dataset": "hard-verify",
             "attempt": attempt}
    n = int(m.get("stop_at", n_steps))
    with t.TraceWriter(path, route=route, identity=ident, max_steps=1300) as w:
        w.log_demo([frame(-2), frame(-1)], [frame(-2, 1), frame(-1, 1)], [state(-2), state(-1)], ["demo"])
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
            sg = (subgoals[min(s - 1, len(subgoals) - 1)] if subgoals else f"subgoal{(s - 1) // 3}")
            if m.get("text") == s:
                sg = sg + "(edited)"
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
