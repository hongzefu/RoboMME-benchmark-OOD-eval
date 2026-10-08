"""原侧一局的写出器：``trace.jsonl`` + 原始帧 + ``arrays.npz``，布局与 GroundSG 原侧 ``official_hard_runner.run_identity``
相同（计划第二部分一节 S7；契约 C1～C11）。

局目录 ``<root>/<key>.a<N>/``：
- ``trace.jsonl``：本仓库 ``trace_writer.TraceWriter``，``route`` 为 ``smvla/orig`` 或 ``perceptual-framesamp-modul/orig``（C1）；``identity`` 含
  ``task``、``tier``（xhard0）、``seed``、``source_episode``、``key``、``dataset``（hard-verify）、``attempt``（= N，C6）；
- ``frames/``：``groundsg_client.RawFrameWriter``（只读复用），reset 全部帧后逐步追加该步观测，缺观测步记 ``missing_steps``；
- ``arrays.npz``：``step_arrays.StepArrays``，每个执行步的实际动作原值（C4）。

计数三分（C8）写进 ``end``：``steps_attempted``（交给环境的步数）、``steps_observed``（返回有效观测的步数）、
``frames_recorded``（= ``demo_frames + 1 + steps_observed - omitted_timeout_frames``）；原侧不可见的 ``terminated``／
``truncated`` 一律写 ``NOT_OBSERVED``（C9）；``end.observer_hook_errors`` 为本局钩子异常数（C11）。

本类只被钩子调用；所有方法在钩子里由调用方 ``try`` 包住，异常交 ``HookErrors`` 计数，绝不外抛到原版逻辑。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_OBS_DIR = str(Path(__file__).resolve().parent)
if _OBS_DIR not in sys.path:
    sys.path.append(_OBS_DIR)
import _obs_common as C  # noqa: E402
import step_arrays as SA  # noqa: E402

groundsg = C.load_eval_module("groundsg_client")
tw = groundsg.trace_writer
NOT_OBSERVED = tw.NOT_OBSERVED


class OrigEpisode:
    def __init__(self, root: str | Path, *, route: str, task: str, source_episode: int, seed: int, max_steps: int,
                 errors: C.HookErrors, write_frames: bool = True) -> None:
        self.root = Path(root)
        self.key = C.identity_key(task, seed)
        self.dir, self.attempt = C.claim_episode_dir(self.root, self.key)
        self.name = self.dir.name
        self.errors = errors
        self.hook_errors = 0
        self.max_steps = int(max_steps)
        self.identity = {"task": task, "tier": C.XHARD0, "seed": int(seed), "source_episode": int(source_episode),
                         "key": self.key, "dataset": C.DATASET, "attempt": int(self.attempt)}
        self.trace = tw.TraceWriter(self.dir / "trace.jsonl", route=route, identity=self.identity,
                                    max_steps=self.max_steps)
        self.frames = groundsg.RawFrameWriter(self.dir / "frames") if write_frames else None
        self.arrays = SA.StepArrays()
        self.steps = 0
        self.observed = 0
        self.demo_frames: int | None = None
        self.subgoal: str | None = None
        self.history_from = 0
        self.encodings: set[str] = set()
        self.closed = False

    # ── 钩子异常：本局计数 + 全局计数 ────────────────────────────────────────
    def hook_error(self, where: str) -> None:
        self.hook_errors += 1
        self.errors.note(where, episode=self.name)

    # ── 演示段（C2）───────────────────────────────────────────────────────────
    def on_reset(self, fronts: list, wrists: list, states: list, text: str) -> None:
        fronts, wrists, states = list(fronts), list(wrists), list(states)
        self.trace.log_demo(fronts, wrists, states, [text])
        self.demo_frames = len(fronts) - 1
        if self.frames is not None:
            for f in fronts:
                self.frames.write("front", f)
            for w in wrists:
                self.frames.write("wrist", w)
            self.frames.meta.update(demo_frames=len(fronts) - 1, init_frames=1 if fronts else 0)

    # ── 请求与回复（C10）─────────────────────────────────────────────────────
    def on_request(self, name: str, payload: bytes) -> None:
        self.trace.log_request(name, payload, step=self.steps)

    def on_response(self, actions: Any) -> None:
        self.trace.log_response(actions, step=self.steps)

    def on_history(self, note: str) -> None:
        """历史缓冲边界（上次边界到当前步的闭区间），之后边界前移到当前步。"""
        self.trace.log_history(self.history_from, self.steps, note=note)
        self.history_from = self.steps

    # ── 执行步（C4、C8、C9）──────────────────────────────────────────────────
    def on_step(self, action: Any, front: Any, wrist: Any, state: Any, status: Any, **extra: Any) -> None:
        self.steps += 1
        self.observed += 1
        self.arrays.add(self.steps - 1, action)
        if self.frames is not None:
            self.frames.meta["exec_steps"] = self.steps
            self.frames.write("front", front)
            self.frames.write("wrist", wrist)
        self.trace.log_step(step=self.steps, front=front, wrist=wrist, state=state, action=action,
                            subgoal=self.subgoal, terminated=NOT_OBSERVED, truncated=NOT_OBSERVED, status=status,
                            **extra)

    def on_missing_step(self, action: Any, reason: str, **extra: Any) -> None:
        self.steps += 1
        self.arrays.add(self.steps - 1, action)
        if self.frames is not None:
            self.frames.meta["exec_steps"] = self.steps
            self.frames.meta["missing_steps"].append(self.steps)
        self.trace.log_missing_step(step=self.steps, action=action, reason=reason, subgoal=self.subgoal, **extra)

    # ── 收尾（C2、C3、C8、C11）───────────────────────────────────────────────
    def close(self, status: str, *, omitted_timeout_frames: int = 0, **extra: Any) -> dict:
        """写 ``arrays.npz``、关帧流、写 ``end``；返回本局摘要。``status`` 不在四终态内一律记 ``error``。"""
        if self.closed:
            return {}
        self.closed = True
        st = status if status in C.TERMINALS else "error"
        no_frame = self.demo_frames is None
        omitted = 0 if no_frame else int(omitted_timeout_frames)
        frames_recorded = 0 if no_frame else self.demo_frames + 1 + self.observed - omitted
        info: dict = {}
        try:
            self.arrays.save(self.dir / "arrays.npz")
        except Exception:  # noqa: BLE001
            self.hook_error("episode.close.arrays")
        try:
            if self.frames is not None:
                info["frames"] = self.frames.close()
        except Exception:  # noqa: BLE001
            self.hook_error("episode.close.frames")
        end = dict(side="orig", demo_frames=0 if no_frame else self.demo_frames, steps_attempted=self.steps,
                   steps_observed=self.observed, frames_recorded=frames_recorded,
                   omitted_timeout_frames=omitted, observer_hook_errors=self.hook_errors)
        if no_frame:
            end["no_frame"] = True
        if status != st:
            end["orig_status_raw"] = status
        end.update(extra)
        self.trace.close(status=st, terminal_reason=st, **end)
        info.update(episode=self.name, status=st, **{k: end[k] for k in (
            "steps_attempted", "steps_observed", "frames_recorded", "observer_hook_errors")})
        return info
