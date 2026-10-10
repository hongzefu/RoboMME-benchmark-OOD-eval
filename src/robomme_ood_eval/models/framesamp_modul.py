#!/usr/bin/env python3
"""MME-VLA client for the new interface.

Copies line by line the loop semantics of the old official client (MME-VLA ``ecf086c``
``examples/robomme/{utils.py,env_runner.py,eval.py}`` and the official historical-score client ``927c56d``
``eval.py::evaluate_manifest``) without importing the submodule's ``examples/``:

* a new websocket per episode (``MMEVLAWebsocketClientPolicy`` from ``openpi_client``), ``client.reset()`` at the
  start;
* all demo frames + the initial frame returned by reset go into the buffer, ``exec_start_idx = len(image_buffer) - 1``;
* empty action plan -> ``add_buffer(pack_buffer(front buffer, state buffer, exec_start_idx))`` -> ``infer`` -> take
  the first 16 -> clear the buffer; each executed step does ``count += 1``, and ``count > 1300`` means timeout
  (1301 steps), with this check coming before the terminal-status check;
* terminal mapping: ``success/fail/timeout`` unchanged, anything else (e.g. ``ongoing``/``unknown``) records
  ``error`` + ``success_flag=<value>``; any exception during the episode records ``error`` +
  ``<exception class>: <message>``, with steps taken from the last ``count`` before the exception.
  When ``env.step`` raises, ``EnvRunner.step`` returns ``(None,)*3, True, "error"`` and the subsequent
  ``add_observation`` calls ``.copy()`` on None and raises ``AttributeError`` -- that is how the old official code
  ended up with error, and it is kept unchanged here; when ``env.step`` does not raise but returns ``obs=None``, the
  old official code raises ``TypeError`` at ``obs["front_rgb_list"]`` outside the try without incrementing
  ``count``, which is also kept unchanged.

The pure-function layer (``pack_state``, ``pack_buffer``, ``EpisodeState``, ``pre_traj_from_reset``,
``EnvRunnerShim``, ``run_loop``, ``evaluate_one``) touches neither network nor simulation; unit tests assert message
sequences directly with synthetic observations.

Two extra subcommands (only for end-to-end checks): ``relay`` starts a per-message transparent websocket relay that
records the sha256 of every message; ``transport-check`` compares client events with the relay log and prints
``TRANSPORT=PASS frames=<n> mismatch=0``.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np

MAX_STEPS = 1300
OBS_HORIZON = 16
#: server wrapper reply audit key (same as trace_writer.AUDIT_KEY)
AUDIT_KEY = "_sgeval_audit"
NORMAL = ("success", "fail", "timeout")
INFRA_MARKERS = ("RecorderError", "svulkan2", "EXCLUSIVE", "Vulkan", "vk::", "out of memory", "RESOURCE_EXHAUSTED",
                 "CUDA_ERROR", "ConnectionClosed", "ConnectionRefused", "InvalidStatus", "Connection reset")


class ProtocolError(RuntimeError):
    """A synchronous server reply did not acknowledge the requested operation."""

    def __init__(self, flag: str, response: Any):
        details = (f"keys={sorted(response.keys())}" if isinstance(response, dict)
                   else f"reply_type={type(response).__name__}; expected dict")
        super().__init__(f"Server reply requires {flag}=True; {details}")


def sha(arr: Any) -> str:
    """sha256 of array bytes (C-contiguous); bytes/str are taken directly."""
    if isinstance(arr, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(arr)).hexdigest()
    if isinstance(arr, str):
        return hashlib.sha256(arr.encode("utf-8")).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# -- copied from examples/robomme/env_runner.py --------------------------------------


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space (copied from env_runner.pack_state)
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


def pre_traj_from_reset(obs: dict, info: dict) -> dict[str, Any]:
    """Copied from the post-reset part of ``EnvRunner.get_init_obs`` (the reset itself is done by EnvSession)."""
    if isinstance(info["task_goal"], list):
        task_goal = info["task_goal"][0]
    else:
        task_goal = info["task_goal"]
    images = obs["front_rgb_list"]
    wrist_images = obs["wrist_rgb_list"]
    states = [pack_state(joint_state, gripper_state) for joint_state, gripper_state in
              zip(obs["joint_state_list"], obs["gripper_state_list"])]
    return {"images": images, "wrist_images": wrist_images, "states": states, "task_goal": task_goal}


class EnvRunnerShim:
    """Copied from ``EnvRunner.step``: ``self.env.step`` is replaced by ``step_fn`` (EnvSession.step); everything
    else is identical line by line."""

    def __init__(self, step_fn: Callable[[Any], tuple]):
        self._step_fn = step_fn
        self.info: dict | None = None
        self.last_exception: BaseException | None = None

    def step(self, action: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], bool, str]:
        try:
            obs, _, terminated, truncated, self.info = self._step_fn(action)
        except Exception as e:
            if type(e).__name__ == "RecorderError":  # a recorder failure is not an environment error: re-raise, the episode records error + infra (new interface only)
                raise
            print(f"Error: {e}")
            self.last_exception = e
            return (None, None, None), True, "error"

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        joint_state = obs["joint_state_list"][-1]
        gripper_state = obs["gripper_state_list"][-1]
        state = pack_state(joint_state, gripper_state)

        outcome = self.info.get("status", "unknown")
        stop = terminated or truncated

        return (img, wrist_img, state), stop, outcome


# -- copied from examples/robomme/utils.py -------------------------------------------


def pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    image_output = np.stack(image_buffer, axis=0).astype(np.uint8)[:, None]
    state_output = np.stack(state_buffer, axis=0).astype(np.float32)
    return {
        "images": image_output,
        "state": state_output,
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }


class EpisodeState:
    def __init__(self):
        self.image_buffer = []
        self.wrist_image_buffer = []
        self.state_buffer = []
        self.action_plan = collections.deque()
        self.count = 0
        self.exec_start_idx = 0

    def add_observation(self, img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray):
        self.image_buffer.append(img.copy())
        self.wrist_image_buffer.append(wrist_img.copy())
        self.state_buffer.append(state.copy())

    def clear_buffers(self):
        self.image_buffer.clear()
        self.wrist_image_buffer.clear()
        self.state_buffer.clear()
        self.exec_start_idx = 0

    def get_current_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]


# -- copied from eval.py::EpisodeEvaluator (without subgoal prediction and video, both no-ops in official evaluation) --


class _Progress:
    """Records ``last_steps`` (the old official ``self.last_steps``) so the step count is available on exceptions."""

    def __init__(self):
        self.last_steps = 0
        self.decisions = 0


def run_loop(client, runner: EnvRunnerShim, reset_fn: Callable[[], dict], progress: _Progress, *,
             max_steps: int = MAX_STEPS, obs_horizon: int = OBS_HORIZON,
             on_decision: Callable[[int, np.ndarray], None] | None = None) -> str:
    """The ``eval_each_episode`` loop: returns success_flag (exceptions propagate and are mapped by
    ``evaluate_one``).

    ``reset_fn`` is called after ``client.reset()`` (same order as the old official "connect to the server and reset
    the policy, then reset the environment") and returns ``pre_traj``."""
    resp = client.reset()
    if not isinstance(resp, dict) or resp.get("reset_finished") is not True:
        raise ProtocolError("reset_finished", resp)

    epstate = EpisodeState()
    pre_traj = reset_fn()
    task_goal = pre_traj["task_goal"]
    epstate.image_buffer.extend(pre_traj["images"])
    epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
    epstate.state_buffer.extend(pre_traj["states"])
    epstate.exec_start_idx = len(epstate.image_buffer) - 1

    img, wrist_img, robot_state = epstate.get_current_obs()
    prompt = task_goal
    success_flag = "unknown"

    while True:
        if not epstate.action_plan:
            resp = client.add_buffer(pack_buffer(
                epstate.image_buffer,
                epstate.state_buffer,
                epstate.exec_start_idx,
            ))
            if not isinstance(resp, dict) or resp.get("add_buffer_finished") is not True:
                raise ProtocolError("add_buffer_finished", resp)
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": robot_state,
                "prompt": prompt,
            }
            actions = client.infer(element)["actions"]
            if on_decision is not None:
                on_decision(progress.decisions, actions)
            progress.decisions += 1
            action_chunk = actions[:obs_horizon]
            epstate.action_plan.extend(action_chunk)
            epstate.clear_buffers()

        action = epstate.action_plan.popleft()
        obs, stop_flag, success_flag = runner.step(action)
        epstate.count += 1

        progress.last_steps = epstate.count
        if epstate.count > max_steps:
            success_flag = "timeout"
            break

        img, wrist_img, robot_state = obs

        epstate.add_observation(img, wrist_img, robot_state)

        if stop_flag:
            break

    return success_flag


def classify_infra(error: str | None, runner: EnvRunnerShim | None) -> str | None:
    """Whether the error is an infrastructure failure (the same identity may be retried, counting toward the retry
    budget); only affects the ``infra`` flag, never status."""
    texts = [error or ""]
    if runner is not None and runner.last_exception is not None:
        texts.append(f"{type(runner.last_exception).__name__}: {runner.last_exception}")
    for text in texts:
        for marker in INFRA_MARKERS:
            if marker in text:
                return marker
    return None


def evaluate_one(client_factory: Callable[[], Any], step_fn: Callable[[Any], tuple], reset_fn: Callable[[], dict], *,
                 max_steps: int = MAX_STEPS, obs_horizon: int = OBS_HORIZON,
                 on_decision: Callable[[int, np.ndarray], None] | None = None) -> dict:
    """Copied from the per-episode terminal mapping of ``evaluate_manifest``. The client connection is inside the try
    too (as in the old official code)."""
    progress = _Progress()
    runner = EnvRunnerShim(step_fn)
    error = None
    error_exc = None
    client = None
    try:
        client = client_factory()
        success_flag = run_loop(client, runner, reset_fn, progress, max_steps=max_steps, obs_horizon=obs_horizon,
                                on_decision=on_decision)
    except Exception as e:  # noqa: BLE001 episode-level catch-all, same as the old official code
        print(f"Error evaluating episode: {e}")
        success_flag, error = "error", f"{type(e).__name__}: {e}"
        error_exc = e
    finally:
        if client is not None:
            try:
                client._ws.close()
            except Exception:  # noqa: BLE001
                pass
    status = success_flag if success_flag in NORMAL else "error"
    if status == "error" and error is None:
        error = f"success_flag={success_flag}"
    infra = ("server_protocol" if isinstance(error_exc, ProtocolError) else classify_infra(error, runner)) \
        if status == "error" else None
    env_exc = runner.last_exception
    return {"status": status, "task_success": status == "success", "steps": progress.last_steps, "error": error,
            "decisions": progress.decisions, "infra": infra is not None, "infra_reason": infra,
            "env_exception": None if env_exc is None else f"{type(env_exc).__name__}: {env_exc}"[:800]}


# -- websocket client with recording and timing ----------------------------------------


def payload_digest(obj: dict) -> dict:
    """Per-frame / per-array sha256 of an outgoing message (the client and the relay use the same function)."""
    out: dict[str, Any] = {}
    if obj.get("reset", False):
        out["kind"] = "reset"
    elif obj.get("add_buffer", False):
        out["kind"] = "add_buffer"
        imgs = np.asarray(obj["images"])
        out["frames"] = [sha(imgs[i, 0]) for i in range(imgs.shape[0])]
        out["states"] = [sha(np.asarray(obj["state"])[i]) for i in range(len(obj["state"]))]
        out["exec_start_idx"] = int(obj["exec_start_idx"])
        out["shape"] = list(imgs.shape)
    else:
        out["kind"] = "infer"
        out["image"] = sha(obj["observation/image"])
        out["wrist"] = sha(obj["observation/wrist_image"])
        out["state"] = sha(obj["observation/state"])
        out["prompt"] = obj.get("prompt")
    return out


def response_digest(obj: Any) -> dict:
    """Incoming message digest: infer replies record the sha256, dtype and shape of actions."""
    if not isinstance(obj, dict):
        return {"kind": "other"}
    if "actions" in obj:
        a = np.asarray(obj["actions"])
        return {"kind": "actions", "actions": sha(a), "dtype": a.dtype.str, "shape": list(a.shape),
                "infer_time_ms": float(obj.get("infer_time_ms", float("nan")))}
    if obj.get("reset_finished"):
        return {"kind": "reset_finished", "server_ms": float(obj.get("reset_time_ms", 0.0))}
    if obj.get("add_buffer_finished"):
        return {"kind": "add_buffer_finished", "server_ms": float(obj.get("add_buffer_time_ms", 0.0))}
    return {"kind": "metadata"}


def make_recording_client(host: str, port: int, recorder, timing: dict):
    """Subclass of ``MMEVLAWebsocketClientPolicy``: send/receive logic identical to the parent line by line
    (pack -> send -> recv -> error on str -> unpackb), only adding sha256 recording and timing on both sides.
    Connection parameters are the same as the parent's ``_wait_for_server``."""
    import websockets.sync.client
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy

    class RecordingClient(MMEVLAWebsocketClientPolicy):
        def __init__(self):
            self._seq = 0
            # optional raw-byte observer hook raw_hook(obj, sent_bytes, recv_bytes), called after each successful round trip (read-only, exceptions swallowed)
            self._raw_hook = None
            self._last_audit = None  # server wrapper audit block popped from the most recent reply
            super().__init__(host, port)

        def _wait_for_server(self):
            t0 = time.perf_counter()
            while True:
                try:
                    headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                    conn = websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None, additional_headers=headers,
                        ping_timeout=600, open_timeout=60, close_timeout=60
                    )
                    raw = conn.recv()
                    metadata = msgpack_numpy.unpackb(raw)
                    timing["connect_s"] = time.perf_counter() - t0
                    self._event({"kind": "ws_recv", "seq": -1, "msg": "metadata", "sha": sha(raw), "len": len(raw)})
                    return conn, metadata
                except ConnectionRefusedError:
                    time.sleep(5)

        def _event(self, ev: dict) -> None:
            if recorder is not None:
                recorder.add_event(ev)

        def _roundtrip(self, obj: dict) -> dict:
            t0 = time.perf_counter()
            data = self._packer.pack(obj)
            t1 = time.perf_counter()
            self._ws.send(data)
            response = self._ws.recv()
            t2 = time.perf_counter()
            if isinstance(response, str):
                self._event({"kind": "ws_recv", "seq": self._seq, "msg": "error_text", "sha": sha(response)})
                raise RuntimeError(f"Error in inference server:\n{response}")
            out = msgpack_numpy.unpackb(response)
            # pop the server wrapper audit key before the reply reaches the official loop (and thus the environment); keep it for the language ledger
            self._last_audit = out.pop(AUDIT_KEY, None) if isinstance(out, dict) else None
            t3 = time.perf_counter()
            dig = payload_digest(obj)
            rdig = response_digest(out)
            server_ms = rdig.get("server_ms", rdig.get("infer_time_ms", 0.0))
            self._event({"kind": "ws_send", "seq": self._seq, "sha": sha(data), "len": len(data), "payload": dig})
            self._event({"kind": "ws_recv", "seq": self._seq, "sha": sha(response), "len": len(response),
                         "payload": rdig, "pack_s": t1 - t0, "rtt_s": t2 - t1, "unpack_s": t3 - t2})
            per = timing.setdefault("per_msg", [])
            per.append({"seq": self._seq, "kind": dig["kind"], "pack_s": t1 - t0, "rtt_s": t2 - t1,
                        "unpack_s": t3 - t2, "server_ms": server_ms, "bytes": len(data)})
            self._seq += 1
            hook = self._raw_hook
            if hook is not None:
                try:
                    hook(obj, data, response)
                except Exception:  # noqa: BLE001 observer hooks must not affect send/receive
                    pass
            return out

        def infer(self, obs):  # noqa: D401
            return self._roundtrip(obs)

        def reset(self):
            return self._roundtrip({"reset": True})

        def add_buffer(self, buffer):
            return self._roundtrip(buffer)

    return RecordingClient()


def summarize_timing(timing: dict) -> dict:
    """Summarize per-message timing into first / 2nd, 3rd / steady-state mean / P95 (inference only counts infer
    messages)."""
    per = timing.pop("per_msg", [])
    out = {k: v for k, v in timing.items()}
    for kind in ("reset", "add_buffer", "infer"):
        rows = [r for r in per if r["kind"] == kind]
        if not rows:
            continue
        rtt = np.array([r["rtt_s"] for r in rows])
        srv = np.array([r["server_ms"] for r in rows]) / 1000.0
        pack = np.array([r["pack_s"] for r in rows])
        unpack = np.array([r["unpack_s"] for r in rows])
        d = {"n": len(rows), "rtt_first_s": float(rtt[0]), "rtt_mean_s": float(rtt.mean()),
             "rtt_p95_s": float(np.percentile(rtt, 95)), "server_mean_s": float(srv.mean()),
             "network_mean_s": float((rtt - srv).mean()), "pack_mean_s": float(pack.mean()),
             "unpack_mean_s": float(unpack.mean()), "bytes_mean": float(np.mean([r["bytes"] for r in rows]))}
        if kind == "infer":
            d["server_first_s"] = float(srv[0])
            d["server_2_s"] = float(srv[1]) if len(srv) > 1 else None
            d["server_3_s"] = float(srv[2]) if len(srv) > 2 else None
            steady = srv[3:] if len(srv) > 3 else srv
            d["server_steady_mean_s"] = float(steady.mean())
            d["server_steady_p95_s"] = float(np.percentile(steady, 95))
        out[kind] = d
    return out


#: legacy sibling module name -> module in the evaluation package (no longer loaded from each other by file path)
SIBLINGS = {"smvla_client": "robomme_ood_eval.models.smvla",
            "groundsg_client": "robomme_ood_eval.models.groundsg",
            "framesamp_modul_client": "robomme_ood_eval.models.framesamp_modul",
            "official_defs": "robomme_ood_eval.models._official_defs",
            "trace_writer": "robomme_ood_eval.record.trace_writer"}


def _load_sibling(name: str):
    """Return the evaluation-package module corresponding to a legacy sibling module (package import; reused if
    already imported)."""
    if name not in SIBLINGS:
        raise ModuleNotFoundError(f"unknown sibling module {name!r}; choices: {', '.join(SIBLINGS)}")
    return importlib.import_module(SIBLINGS[name])


class TracedClient:
    """Wraps ``reset`` / ``add_buffer`` / ``infer`` of the MME-VLA websocket client (or a stand-in), forwarding them
    unchanged and then recording requests and replies (C10).

    Requests record the sha256 of the raw msgpack frame bytes (the bytes actually sent, obtained via
    ``RecordingClient._raw_hook``; stand-ins without the hook fall back to ``canonical_bytes``); ``add_buffer`` also
    records a history-boundary line (the covered step range); ``infer`` replies record the full action chunk. The
    forwarded objects and return values are never modified, and a request that fails (raises) is not recorded."""

    def __init__(self, inner: Any, trace: Any):
        self._inner = inner
        self._trace = trace
        self._raw: tuple | None = None
        self._history_from = 0
        if hasattr(inner, "_raw_hook"):
            inner._raw_hook = self._on_raw

    def _on_raw(self, obj, data, response) -> None:
        self._raw = (data, response)

    def _record(self, name: str, obj: Any, out: Any) -> None:
        raw, self._raw = self._raw, None
        tr = self._trace
        tr.raw_request(name, obj, raw[0] if raw is not None else None)
        try:
            if name == "add_buffer":
                n = len(obj["images"]) if isinstance(obj, dict) and "images" in obj else 0
                tr.history(self._history_from, tr.steps,
                           note=f"add_buffer frames={n} exec_start_idx={obj.get('exec_start_idx')}")
                self._history_from = tr.steps
            elif name == "infer":
                tr.response(out["actions"])
        except Exception as e:  # noqa: BLE001
            tr._err(f"client.{name}", e)

    def reset(self):
        self._raw = None
        out = self._inner.reset()
        self._record("reset", {"reset": True}, out)
        return out

    def add_buffer(self, buffer):
        self._raw = None
        out = self._inner.add_buffer(buffer)
        self._record("add_buffer", buffer, out)
        return out

    def _image_refs(self, obs: Any) -> list | None:
        """References to the current front and wrist frames (``raw_sha256`` uses the same algorithm as the trace and
        can be found in the last demo frame or the previous step line)."""
        if not isinstance(obs, dict):
            return None
        tr = self._trace
        refs = [tr.image_ref(0, "current", "front", obs.get("observation/image")),
                tr.image_ref(1, "wrist", "wrist", obs.get("observation/wrist_image"))]
        return [r for r in refs if r is not None] or None

    def infer(self, obs):
        """Language ledger: one ``action_model`` call per inference step. Before sending write ``in``
        (``role=fields``: raw structured fields + references to the two current frames); after the reply pop
        ``_sgeval_audit`` (the real client already popped it in ``_roundtrip`` and stored it in ``_last_audit``),
        write per-channel tokenization messages, and ``close_call`` records ``server_final_text``; steps executed
        afterwards are linked to this call."""
        self._raw = None
        tr = self._trace
        cid = tr.lang_open("action_model")
        fields = {k: v for k, v in obs.items() if isinstance(v, (str, int, float, bool)) or v is None} \
            if isinstance(obs, dict) else None
        tr.lang_msg(cid, dir="in", role="fields", text=fields, images=self._image_refs(obs))
        if hasattr(self._inner, "_last_audit"):
            self._inner._last_audit = None
        try:
            out = self._inner.infer(obs)
        except BaseException:
            tr.lang_close(cid, status="error")
            raise
        audit = out.pop(AUDIT_KEY, None) if isinstance(out, dict) else None
        if audit is None:
            audit = getattr(self._inner, "_last_audit", None)
        final_text, final_trunc = tr.lang_audit(cid, audit)
        tr.lang_close(cid, status="reply", server_final_text=final_text, server_truncated=final_trunc)
        tr.begin_chunk(cid)
        self._record("infer", obs, out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def traced_step_fn(session: Any, trace: Any) -> Callable[[Any], tuple]:
    """Wrap ``session.step`` to get the full 5-tuple and record the per-step trace; the action object goes to the
    environment unchanged, and return values and exceptions pass through unchanged.

    With an observation: front and wrist images of the last frame and the ``pack_state`` state (same algorithm as
    ``EnvRunnerShim.step``), subgoal ``None`` (C7); ``obs is None``: step without observation; on an exception: add
    steps without observation according to the increase of the session's ``steps`` (i.e. the steps for which
    ``EnvRunnerShim`` returns ``(None,)*3``); ``StepCapReached`` never reaches the environment, is not counted and
    only sets ``cap_hit``."""

    def step(action):
        before = getattr(session, "steps", None)
        try:
            out = session.step(action)
        except Exception as e:
            trace.step_exception(action, e, before, getattr(session, "steps", None))
            raise
        try:
            obs, _r, terminated, truncated, info = out
            status = info.get("status", "unknown") if isinstance(info, dict) else None
            if obs is None:
                trace.missing(action, "obs_none")
            else:
                try:
                    img = obs["front_rgb_list"][-1]
                    wrist = obs["wrist_rgb_list"][-1]
                    state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
                except Exception as e:  # noqa: BLE001 unreadable observation: pass through as usual and let the official loop report it
                    trace._err("step.obs", e)
                    trace.missing(action, f"obs_unreadable: {type(e).__name__}: {e}")
                else:
                    trace.step(action, img, wrist, state, subgoal=None, terminated=terminated,
                               truncated=truncated, status=status)
        except Exception as e:  # noqa: BLE001
            trace._err("step", e)
        return out

    return step


def run_episode(session, identity: dict, conn_info: dict, recorder) -> dict:
    """Entry point called by the outer loop: one FrameSamp+Modulation episode. ``session`` is an already built
    EnvSession.

    With a trace location (``groundsg_client.trace_location``), writes ``trace.jsonl`` (route
    ``perceptual-framesamp-modul/new``): ``reset_fn`` is wrapped to record the demo (C2), ``session.step`` is wrapped
    to record per step (C4, C8), the client is wrapped to record requests and replies (C10), and finalization follows
    C2, C3, C8. Without a location none of the three wrappers is applied and behavior is unchanged."""
    timing: dict[str, Any] = {}
    max_steps = int(conn_info.get("max_steps", MAX_STEPS))
    sm = _load_sibling("smvla_client")  # the shared PolicyTrace
    trace = sm.PolicyTrace("perceptual-framesamp-modul/new", identity, conn_info, recorder, max_steps=max_steps,
                           recorder_has_actions=sm.recorder_writes_arrays(recorder) and
                           getattr(session, "recorder", None) is recorder,
                           omit_overflow_frame=True)

    def client_factory():
        client = make_recording_client(conn_info.get("host", "127.0.0.1"), int(conn_info["port"]), recorder, timing)
        return TracedClient(client, trace) if trace.enabled else client

    def reset_fn():
        obs, info = session.reset()
        pre = pre_traj_from_reset(obs, info)
        if trace.enabled:
            trace.demo(pre["images"], pre["wrist_images"], pre["states"], pre["task_goal"])
        return pre

    def on_decision(idx: int, actions: np.ndarray) -> None:
        if recorder is not None:
            recorder.add_array("model_action", np.array(actions, copy=True), step=idx)
            a = np.asarray(actions)
            recorder.add_event({"kind": "model_action", "decision": idx, "sha": sha(a),
                                "row_sha": [sha(a[i]) for i in range(a.shape[0])],
                                "dtype": np.asarray(actions).dtype.str, "shape": list(np.asarray(actions).shape),
                                "exec_n": min(OBS_HORIZON, len(actions)), "env_step": session.steps})

    step_fn = traced_step_fn(session, trace) if trace.enabled else session.step
    t0 = time.perf_counter()
    res = evaluate_one(client_factory, step_fn, reset_fn, max_steps=max_steps, on_decision=on_decision)
    timing["episode_s"] = time.perf_counter() - t0
    res["timing"] = summarize_timing(timing)
    if trace.enabled:
        trace.close(res["status"], cap_hit=bool(getattr(session, "cap_hit", False)), decisions=res["decisions"],
                    session_steps=getattr(session, "steps", None))
        res["trace_path"] = str(trace.path)
    return res


# -- relay and transport check (for end-to-end checks) --------------------------------


def cmd_relay(args) -> int:
    """Per-message transparent relay: the client connects to the listen port and the relay connects upstream; every
    message is forwarded unchanged (no compression, max_size=None), and the log records direction, sequence number,
    sha256, length and the per-frame digest after unpacking."""
    import asyncio

    import websockets
    import websockets.asyncio.client as wsc
    import websockets.asyncio.server as wss
    from openpi_client import msgpack_numpy

    log = open(args.log, "a", encoding="utf-8")
    conn_id = [0]

    def write(rec):
        log.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        log.flush()

    async def handler(client_ws):
        cid = conn_id[0]
        conn_id[0] += 1
        async with wsc.connect(f"ws://127.0.0.1:{args.upstream}", compression=None, max_size=None,
                               ping_timeout=600, open_timeout=60, close_timeout=60) as up:
            seq = {"c2s": 0, "s2c": -1}

            async def pump(src, dst, direction):
                try:
                    async for msg in src:
                        rec = {"conn": cid, "dir": direction, "seq": seq[direction], "sha": sha(msg),
                               "len": len(msg), "t": time.time()}
                        if not isinstance(msg, str):
                            try:
                                obj = msgpack_numpy.unpackb(msg)
                                rec["payload"] = (payload_digest(obj) if direction == "c2s" else response_digest(obj))
                            except Exception as e:  # noqa: BLE001
                                rec["decode_error"] = repr(e)
                        write(rec)
                        seq[direction] += 1
                        await dst.send(msg)
                except websockets.ConnectionClosed:
                    pass
                finally:
                    await dst.close()

            await asyncio.gather(pump(client_ws, up, "c2s"), pump(up, client_ws, "s2c"))

    async def main():
        async with wss.serve(handler, "127.0.0.1", args.listen, compression=None, max_size=None):
            print(f"RELAY_READY listen={args.listen} upstream={args.upstream}", flush=True)
            await asyncio.Future()

    asyncio.run(main())
    return 0


def transport_check(events: list[dict], relay: list[dict]) -> dict:
    """Client events vs relay log: same sha per message, same order; env frames / states == outgoing payload; reply
    actions == executed actions."""
    mismatch, frames, notes = 0, 0, []
    sends = [e for e in events if e.get("kind") == "ws_send"]
    recvs = [e for e in events if e.get("kind") == "ws_recv"]
    r_c2s = [r for r in relay if r["dir"] == "c2s"]
    r_s2c = [r for r in relay if r["dir"] == "s2c"]
    if len(sends) != len(r_c2s) or len(recvs) != len(r_s2c):
        mismatch += 1
        notes.append(f"count client send/recv={len(sends)}/{len(recvs)} relay={len(r_c2s)}/{len(r_s2c)}")
    for a, b in zip(sends, r_c2s):
        if a["sha"] != b["sha"] or a["len"] != b["len"] or a["payload"] != b.get("payload"):
            mismatch += 1
            notes.append(f"c2s seq={a['seq']}")
    for a, b in zip(recvs, r_s2c):
        if a["sha"] != b["sha"]:
            mismatch += 1
            notes.append(f"s2c seq={a['seq']}")
    # compare env frames with the outgoing payload frame by frame: add_buffer's frames must equal the sequence of front frames handed out by env since the previous decision
    env_front: list[str] = []
    env_wrist: list[str] = []
    env_state: list[str] = []
    goal = None
    exec_actions: list[str] = []
    model_rows: list[str] = []
    for e in events:
        k = e.get("kind")
        if k == "env_reset":
            env_front, env_wrist, env_state = list(e["front"]), list(e["wrist"]), list(e["state8"])
            goal = e.get("task_goal")
        elif k == "env_step" and e.get("front") is not None:
            env_front.append(e["front"][-1])
            env_wrist.append(e["wrist"][-1])
            env_state.append(e["state8"])
        elif k == "env_step_action":
            exec_actions.append(e["sha"])
        elif k == "model_action":
            model_rows.extend(e.get("row_sha", [])[: e.get("exec_n", OBS_HORIZON)])
        elif k == "ws_send":
            p = e["payload"]
            if p["kind"] == "add_buffer":
                frames += len(p["frames"])
                if p["frames"] != env_front or p["states"] != env_state:
                    mismatch += 1
                    notes.append(f"add_buffer seq={e['seq']} frames/state != env")
                env_front_last, env_wrist_last, env_state_last = env_front[-1], env_wrist[-1], env_state[-1]
                env_front, env_wrist, env_state = [], [], []
            elif p["kind"] == "infer":
                frames += 2
                if (p["image"], p["wrist"], p["state"]) != (env_front_last, env_wrist_last, env_state_last):
                    mismatch += 1
                    notes.append(f"infer seq={e['seq']} obs != env")
                if goal is not None and p["prompt"] != goal:
                    mismatch += 1
                    notes.append(f"infer seq={e['seq']} prompt")
    n = min(len(exec_actions), len(model_rows))
    bad_exec = sum(1 for i in range(n) if exec_actions[i] != model_rows[i])
    if bad_exec or len(exec_actions) > len(model_rows):
        mismatch += bad_exec + max(0, len(exec_actions) - len(model_rows))
        notes.append(f"exec_action != model rows: {bad_exec}")
    return {"mismatch": mismatch, "frames": frames, "messages": len(sends) + len(recvs),
            "exec_actions": len(exec_actions), "notes": notes[:20]}


def cmd_transport_check(args) -> int:
    events = [json.loads(line) for line in Path(args.events).read_text().splitlines() if line.strip()]
    relay = [json.loads(line) for line in Path(args.relay_log).read_text().splitlines() if line.strip()]
    if args.conn is not None:
        relay = [r for r in relay if r["conn"] == args.conn]
    res = transport_check(events, relay)
    ok = res["mismatch"] == 0 and res["frames"] > 0
    print(f"TRANSPORT={'PASS' if ok else 'FAIL'} frames={res['frames']} mismatch={res['mismatch']} "
          f"messages={res['messages']} exec_actions={res['exec_actions']}")
    for note in res["notes"]:
        print(f"  note: {note}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FrameSamp+Modulation client helper subcommands")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("relay")
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", type=int, required=True)
    p.add_argument("--log", required=True)
    p.set_defaults(func=cmd_relay)
    p = sub.add_parser("transport-check")
    p.add_argument("--events", required=True)
    p.add_argument("--relay-log", required=True)
    p.add_argument("--conn", type=int, default=None)
    p.set_defaults(func=cmd_transport_check)
    args = ap.parse_args(argv)
    return args.func(args)


# -- new interface: the four model-side methods --------------------------------------

from robomme_ood_eval import servers as _servers  # noqa: E402
from robomme_ood_eval.policy import Ready  # noqa: E402

#: synthetic image size for warm-up (both RoboMME cameras are 256x256) and frame count (steady state passes 16 frames
#: per add_buffer)
WARMUP_HW = (256, 256)
WARMUP_FRAMES = OBS_HORIZON


def warmup_server(host: str, port: int, *, frames: int = WARMUP_FRAMES, hw: tuple[int, int] = WARMUP_HW,
                  subgoal: str | None = None, prompt: str = "warm up", client_factory: Callable | None = None) -> dict:
    """Process-level warm-up (added for FrameSamp+Modulation and GroundSG): run synthetic observations through
    ``reset -> add_buffer -> infer`` so the server compiles the first input shape before the real episodes. The
    ``reset`` message at the start of a real episode makes the server rebuild the memory buffer and reset its random
    numbers (the official ``Policy.reset`` executes ``self._rng = jax.random.key(self._seed)``), so nothing carries
    over from warm-up. Returns the per-message timing summary and total time; warm-up only proves the first shape is
    compiled and does not claim zero compilation afterwards. ``client_factory`` is only for unit tests to inject a
    stand-in."""
    timing: dict[str, Any] = {}
    t0 = time.perf_counter()
    factory = client_factory or (lambda: make_recording_client(host, int(port), None, timing))
    client = factory()
    try:
        h, w = hw
        n = max(1, int(frames))
        imgs = [np.full((h, w, 3), (37 * i) % 256, np.uint8) for i in range(n)]
        states = [np.zeros(8, np.float32) for _ in range(n)]
        resp = client.reset()
        if not isinstance(resp, dict) or resp.get("reset_finished") is not True:
            raise ProtocolError("reset_finished", resp)
        resp = client.add_buffer(pack_buffer(imgs, states, n - 1))
        if not isinstance(resp, dict) or resp.get("add_buffer_finished") is not True:
            raise ProtocolError("add_buffer_finished", resp)
        element = {"observation/image": imgs[-1], "observation/wrist_image": imgs[-1],
                   "observation/state": states[-1], "prompt": prompt}
        if subgoal is not None:  # same as the official get_action_chunk: both subgoal keys get the same value
            element["simple_subgoal"] = subgoal
            element["grounded_subgoal"] = subgoal
        actions = np.asarray(client.infer(element)["actions"])
    finally:
        try:
            client._ws.close()
        except Exception:  # noqa: BLE001
            pass
    out = summarize_timing(timing)
    out.update(warmup_s=round(time.perf_counter() - t0, 3), frames=n, actions_shape=list(actions.shape))
    print(f"WARMUP=PASS port={port} frames={n} warmup_s={out['warmup_s']:.1f} actions={list(actions.shape)}",
          flush=True)
    return out


def mme_vla_server_spec(policy, ckpt: Any) -> tuple[list, dict, Path]:
    """Command, environment and cwd of the MME-VLA action server (shared by FrameSamp+Modulation and GroundSG),
    copied from the ``perceptual-framesamp-modul|groundsg`` branch of the legacy ``run_seat.sh::build_server_cmd``
    (the wrapper is now this package's ``servers/policy_server_wrap.py``; wrapper metadata goes to a separate
    ``server-wrap-metadata-<port>.json`` so it never overwrites ``ServerProcess``'s own
    ``server-metadata-<port>.json``). cfg: ``mme_vla_py`` (default ``third_party/mme-vla/.venv/bin/python``,
    overridable by env var ``MME_VLA_PY``), ``openpi_data_home``, ``xla_mem_fraction`` (default 0.75),
    ``compile_cache``, ``jax_cache_root`` (default ``<repo root>/artifacts/jax-cache``, with one subdirectory per GPU
    model), ``det``."""
    S = _servers
    cfg = policy.cfg
    sub = policy.root / "third_party" / "mme-vla"
    py = S.interpreter(cfg, "mme_vla_py", "MME_VLA_PY", sub / ".venv" / "bin" / "python")
    env = {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(cfg.get("xla_mem_fraction", S.DEFAULT_XLA_MEM_FRACTION)),
           "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=16384", "UV_LINK_MODE": "copy",
           "VIRTUAL_ENV": str(sub / ".venv")}
    if S.flag_on(cfg.get("compile_cache", False)):
        cache_root = Path(cfg.get("jax_cache_root") or policy.root / "artifacts" / "jax-cache")
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_root / S.gpu_slug(S.gpu_of(cfg)))
    if S.flag_on(cfg.get("det", False)):
        env["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
    env.update(OPENPI_DATA_HOME=str(cfg.get("openpi_data_home") or ""), PYTHONUNBUFFERED="1")
    argv = [str(py), str(S.SERVERS_DIR / "policy_server_wrap.py"), f"--sgeval-metadata-out={policy.wrap_meta}",
            f"--seed={int(policy.policy_seed)}", f"--port={int(policy.port)}", "policy:checkpoint",
            "--policy.config=mme_vla_suite", f"--policy.dir={ckpt}"]
    return argv, env, sub


def load_mme_vla_server(policy, model: str) -> Path:
    """Shared ``load()`` part of FrameSamp+Modulation / GroundSG: tokenizer gate -> MME-VLA preflight -> start (or
    attach) the server (ready = port listening + server log contains ``history_config='<yaml>'``) -> background
    ckpt fingerprint. Returns the ckpt used."""
    S = _servers
    cfg = policy.cfg
    ckpt = S.require_ckpt(cfg, model)
    policy._pick_port()
    argv, env, cwd = mme_vla_server_spec(policy, ckpt)
    if policy.preflight:
        S.tokenizer_gate(cfg.get("openpi_data_home"), cfg.get("tokenizer_sha256"))
        S.preflight_mme_vla(policy.root, model, ckpt, Path(argv[0]))
    srv = policy._launch(argv, env, cwd, [Ready.port()], ckpt)
    policy.load_info["server_config"] = S.check_server_log(srv, S.YAML_EXPECT[model],
                                                           grace_s=float(cfg.get("server_config_grace_s", 30.0)))
    policy._fingerprint(ckpt)
    return ckpt


class FrameSampModulPolicy(_servers.ServedPolicy):
    """FrameSamp+Modulation (``perceptual-framesamp-modul``).

    * ``load``: ``load_mme_vla_server`` (cwd ``third_party/mme-vla``, ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.75``) + the
      added warm-up ``warmup_server`` (``cfg["warmup"]``, on by default);
    * ``reset(spec)``: sends no message and never touches the environment -- only probes liveness and records this
      episode's identity (the client buffer and action plan are recreated per episode in ``run_episode``);
    * ``play``: calls this module's ``run_episode`` unchanged: a new websocket per episode, sending the server
      ``reset`` first and then ``session.reset()`` (the old order);
    * ``close``: stop the server process group (base class).

    Required cfg: ``ckpt`` (no default), ``openpi_data_home``, ``tokenizer_sha256`` (when preflight is on);
    optional ``mme_vla_py``, ``compile_cache``, ``det``, ``xla_mem_fraction``, ``warmup``, ``warmup_frames``."""

    model = "perceptual-framesamp-modul"
    requires_ckpt = True

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        self.ckpt: Path | None = None
        self.current_spec = None

    def load(self) -> None:
        self.ckpt = load_mme_vla_server(self, self.model)
        if _servers.flag_on(self.cfg.get("warmup", True)):
            self.load_info["warmup"] = warmup_server(self.host, int(self.port),
                                                     frames=int(self.cfg.get("warmup_frames", WARMUP_FRAMES)))

    def reset(self, spec) -> None:
        super().reset(spec)
        self.current_spec = spec

    def play(self, session, spec, recorder) -> dict:
        res = run_episode(session, spec.identity(), self._conn_info(spec), recorder)
        return self._finish_play(res)


if __name__ == "__main__":
    sys.exit(main())
