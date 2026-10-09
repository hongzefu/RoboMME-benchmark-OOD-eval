"""``EnvSession``: the environment session the outer loop hands to ``Policy.play``.

One instance per episode. ``episode.run_episode`` creates it, calls ``build()``, passes it to the model's ``play``
and finally calls ``close()`` itself:

* ``build()``: ``builder.make_env_for_episode(builder_episode)``; consumes one unit of reset budget;
* ``reset()``: returns ``env.reset()``'s ``(obs, info)`` unchanged; consumes one unit of reset budget; afterwards
  writes the demo-segment frames, states and events to the recorder;
* ``step(action)``: passes the action unchanged to ``env.step`` and records the array actually sent
  (``exec_action``) plus the returned current frame, state and terminal flags; with ``strict_cap`` the
  ``max_steps+1``-th call does not reach the environment, sets ``cap_hit`` and raises ``StepCapReached``;
* ``close()``: called only by the outer loop (the model side must never close the environment); idempotent.

Budget: ``ledger`` (any object with a ``claim(what)`` method that raises ``ResetBudgetExhausted``, or an exception
carrying a ``budget_exhausted`` attribute, when over budget) and ``claim_reset(what)`` (a persistent-ledger callback)
are each claimed once before every actual build / reset; if neither is given no budget is charged and only
``reset_calls`` is counted. Read-only properties: ``env``, ``info``, ``steps``, ``cap_hit``, ``reset_calls``,
``recorder``. Hooks: ``first_step_cb`` (fired once on the episode's first ``step`` call; ends first-inference
timing) and ``progress_cb(steps)`` (once every ``progress_every`` steps).
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

#: the two evaluation datasets
OOD = "ood"
HARD_VERIFY = "hard-verify"
DATASETS = (OOD, HARD_VERIFY)


class RecorderError(RuntimeError):
    """Recorder failure (including a full disk): classified as an infrastructure failure (infra=True), not an
    environment error."""


class StepCapReached(RuntimeError):
    """``strict_cap``: ``max_steps`` steps were executed without the episode ending; the ``max_steps+1``-th ``step``
    does not reach the environment (the outer loop records it as a timeout)."""


class ResetBudgetExhausted(RuntimeError):
    """The reset budget (build and reset count one each) is used up."""

    budget_exhausted = True


def sha(arr: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def state8(joint, gripper):
    """8-d state in the same form as the old official ``pack_state`` (7 joints + first gripper dim, float32)."""
    import numpy as np

    return np.concatenate([np.asarray(joint), np.asarray(gripper)[:1]], axis=0, dtype=np.float32)


def _scalar(x):
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return None


class _GuardedRecorder:
    """Recorder proxy: any exception raised by a recording call is converted to RecorderError (cause kept)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if not callable(fn):
            return fn

        def call(*a, **k):
            try:
                return fn(*a, **k)
            except RecorderError:
                raise
            except Exception as e:  # noqa: BLE001
                raise RecorderError(f"{name}: {type(e).__name__}: {e}") from e

        return call


class NullRecorder:
    """No-op recorder used when recording is off (same interface as ``record.recorder.EpisodeRecorder``)."""

    def set_phase(self, phase):
        pass

    def add_frames(self, stream, frames, *, tag=""):
        return []

    def add_array(self, name, arr, *, step=None):
        pass

    def add_event(self, event):
        pass

    def close(self, summary):
        return {"RECORDER_VERIFY": "SKIP"}


class EnvSession:
    """One episode's environment. ``builder`` is cached by the outer loop per ``(task, dataset)`` and passed in
    (matching the official "one builder per task")."""

    def __init__(self, task: str, builder_episode: int, *, dataset: str = OOD, max_steps: int | None = None,
                 recorder=None, builder=None, progress_cb: Callable[[int], None] | None = None,
                 progress_every: int = 16, step_cap: int | None = None, ledger: Any = None,
                 claim_reset: Callable[[str], None] | None = None, budget_claim: Callable[[str], None] | None = None,
                 first_step_cb: Callable[[], None] | None = None):
        if dataset not in DATASETS:
            raise ValueError(f"dataset={dataset!r} is not one of {DATASETS}")
        self.task = task
        self.dataset = dataset
        self.builder_episode = int(builder_episode)
        # max_steps is only used when building our own builder (constructor argument); never passed per episode
        # to make_env_for_episode
        self.max_steps = None if max_steps is None else int(max_steps)
        self.step_cap = None if step_cap is None else int(step_cap)
        self.ledger = ledger
        self.claim_reset = claim_reset
        self.budget_claim = budget_claim
        self.first_step_cb = first_step_cb
        self.progress_cb = progress_cb
        self.progress_every = max(1, int(progress_every))
        self.recorder = recorder if recorder is not None else NullRecorder()
        # internal recording always goes through the proxy; the original recorder is still exposed (the
        # SimpleMemVLA client checks ``session.recorder is recorder``)
        self._rec = _GuardedRecorder(self.recorder)
        self._builder = builder
        self._env = None
        self._info: Any = None
        self._steps = 0
        self._cap_hit = False
        self._reset_calls = 0
        self._closed = False
        self.budget_exhausted = False
        self.timing: dict[str, Any] = {}
        self._step_s: list[float] = []
        self._rec_s = 0.0
        self.task_goal = None

    # -- read-only properties ------------------------------------------------
    @property
    def env(self):
        return self._env

    @property
    def info(self):
        """The info returned by the latest reset / step (GroundSG Oracle reads ``grounded_subgoal_online`` from it
        every step)."""
        return self._info

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def cap_hit(self) -> bool:
        return self._cap_hit

    @property
    def reset_calls(self) -> int:
        return self._reset_calls

    @property
    def builder(self):
        if self._builder is None:
            from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

            if self.max_steps is None:
                raise ValueError("EnvSession needs max_steps to build its own builder (the step cap is set by the "
                                 "entry point per dataset)")
            t0 = time.perf_counter()
            self._builder = BenchmarkEnvBuilder(env_id=self.task, dataset=self.dataset, action_space="joint_angle",
                                                max_steps=self.max_steps)
            self.timing["builder_init_s"] = time.perf_counter() - t0
        return self._builder

    # -- budget ----------------------------------------------------------------
    def _claim(self, what: str) -> None:
        """Claim budget once before every actual build / reset; reset_calls is incremented only after a successful
        claim."""
        try:
            if self.ledger is not None:
                self.ledger.claim(what)
            if self.claim_reset is not None:
                self.claim_reset(what)
        except Exception as e:  # noqa: BLE001 budget exhausted (may come from another copy of the module, so match by attribute)
            if isinstance(e, ResetBudgetExhausted) or getattr(e, "budget_exhausted", False):
                self.budget_exhausted = True
            raise
        if self.budget_claim is not None:
            self.budget_claim(what)
        self._reset_calls += 1

    # -- environment -----------------------------------------------------------
    def build(self) -> None:
        """``make_env_for_episode(builder_episode)``: the step cap comes from the builder constructor argument
        ``max_steps`` (same as official)."""
        if self._env is not None:
            return
        if self._closed:
            raise RuntimeError("EnvSession is already closed; cannot build again")
        self._claim("build")
        self._rec.set_phase("reset")
        t0 = time.perf_counter()
        self._env = self.builder.make_env_for_episode(self.builder_episode)
        self.timing["env_build_s"] = time.perf_counter() - t0

    def reset(self):
        """Return ``env.reset()``'s ``(obs, info)`` unchanged; afterwards record the demo-segment frames and arrays
        (the recorder is still in the reset phase, so they are only queued)."""
        import numpy as np

        self.build()
        self._claim("reset")
        self._rec.set_phase("reset")
        t0 = time.perf_counter()
        obs, info = self._env.reset()
        self.timing["reset_s"] = time.perf_counter() - t0
        self._info = info
        t1 = time.perf_counter()
        goal = info.get("task_goal") if isinstance(info, dict) else None
        self.task_goal = goal[0] if isinstance(goal, list) else goal
        front = np.stack(obs["front_rgb_list"])
        wrist = np.stack(obs["wrist_rgb_list"])
        self._rec.add_frames("front", front, tag="reset")
        self._rec.add_frames("wrist", wrist, tag="reset")
        for key in ("joint_state_list", "gripper_state_list", "eef_state_list"):
            if key in obs and obs[key] is not None and len(obs[key]):
                self._rec.add_array(f"reset_{key[:-5]}", np.stack([np.asarray(x) for x in obs[key]]))
        s8 = [sha(state8(j, g)) for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])]
        self._rec.add_event({"kind": "env_reset", "task": self.task, "builder_episode": self.builder_episode,
                             "frames": int(front.shape[0]), "demo_frames": int(front.shape[0]) - 1,
                             "front": [sha(f) for f in front], "wrist": [sha(f) for f in wrist], "state8": s8,
                             "task_goal": self.task_goal,
                             "status": info.get("status") if isinstance(info, dict) else None,
                             "reset_s": self.timing["reset_s"]})
        self._rec_s += time.perf_counter() - t1
        self.timing["demo_frames"] = int(front.shape[0]) - 1
        self._rec.set_phase("run")
        return obs, info

    def step(self, action):
        """Pass the action unchanged to ``env.step``; record the array sent and the returned current frame, state
        and terminal flags."""
        import numpy as np

        if self.first_step_cb is not None:
            cb, self.first_step_cb = self.first_step_cb, None
            cb()
        if self.step_cap is not None and self._steps >= self.step_cap:
            self._cap_hit = True
            self._rec.add_event({"kind": "step_cap_reached", "step": self._steps, "cap": self.step_cap})
            raise StepCapReached(f"STEP_CAP exec_steps={self._steps} cap={self.step_cap}")
        t0 = time.perf_counter()
        a = np.array(action, copy=True)
        self._rec.add_array("exec_action", a, step=self._steps)
        self._rec.add_event({"kind": "env_step_action", "step": self._steps, "sha": sha(a), "dtype": a.dtype.str,
                             "shape": list(a.shape)})
        t1 = time.perf_counter()
        try:
            out = self._env.step(action)
        except Exception as e:
            self._step_s.append(time.perf_counter() - t1)
            self._rec.add_event({"kind": "env_step_exception", "step": self._steps,
                                 "error": f"{type(e).__name__}: {e}"[:800]})
            self._steps += 1
            raise
        t2 = time.perf_counter()
        self._step_s.append(t2 - t1)
        obs, reward, terminated, truncated, info = out
        self._info = info
        status = info.get("status") if isinstance(info, dict) else None
        ev: dict[str, Any] = {"kind": "env_step", "step": self._steps, "terminated": bool(terminated),
                              "truncated": bool(truncated), "status": status, "reward": _scalar(reward)}
        if obs is None:
            ev.update(front=None, wrist=None, obs_none=True)
        else:
            front = np.stack(obs["front_rgb_list"])
            wrist = np.stack(obs["wrist_rgb_list"])
            self._rec.add_frames("front", front, tag=f"step{self._steps}")
            self._rec.add_frames("wrist", wrist, tag=f"step{self._steps}")
            joint, grip = obs["joint_state_list"][-1], obs["gripper_state_list"][-1]
            self._rec.add_array("joint_state", np.asarray(joint), step=self._steps)
            self._rec.add_array("gripper_state", np.asarray(grip), step=self._steps)
            if obs.get("eef_state_list"):
                self._rec.add_array("eef_state", np.asarray(obs["eef_state_list"][-1]), step=self._steps)
            ev.update(front=[sha(f) for f in front], wrist=[sha(f) for f in wrist], state8=sha(state8(joint, grip)))
        self._rec.add_event(ev)
        self._steps += 1
        self._rec_s += (t1 - t0) + (time.perf_counter() - t2)
        if self.progress_cb is not None and self._steps % self.progress_every == 0:
            self.progress_cb(self._steps)
        return out

    def close(self) -> None:
        """Close the environment and summarize per-step timing; called only by the outer loop; idempotent."""
        import numpy as np

        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            t0 = time.perf_counter()
            try:
                self._env.close()
            finally:
                self.timing["env_close_s"] = time.perf_counter() - t0
                self._env = None
        if self._step_s:
            s = np.array(self._step_s)
            self.timing.update(step_n=int(s.size), step_total_s=float(s.sum()), step_mean_s=float(s.mean()),
                               step_p50_s=float(np.percentile(s, 50)), step_p95_s=float(np.percentile(s, 95)),
                               step_first_s=float(s[0]))
        self.timing["record_overhead_s"] = self._rec_s
