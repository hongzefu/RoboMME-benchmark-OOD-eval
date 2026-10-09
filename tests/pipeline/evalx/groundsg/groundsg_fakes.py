"""Shared test doubles for GroundSG (S3) tests: fake env, two builders, fake MME-VLA server and connection,
a swift stand-in that loads no weights, and drivers for both sides.

Design rules:
- Official sources are read-only and referenced via ``SGEVAL_THIRD_PARTY``; when unset, the ``third_party`` of the
  current checkout is used (submodule dirs are empty in a worktree, so point it at the main checkout explicitly).
  Missing official sources fail the test; they are never skipped.
- Production modules are always loaded by path via ``tests._support.loaders.load_script``; no doubles are injected
  into sys.modules.
- The fake env derives the next frame deterministically from "executed action bytes + step number", and the fake
  server derives each action chunk deterministically from "fingerprints of all requests received since this
  episode's reset": any differing request or action on either side changes all later frames, requests and actions
  (differences are observable, equalities are non-trivial).
- The fake server's fingerprint function is hand-written here and does not call the code under test
  (``official_defs.canonical_bytes``).
- swift stand-in: ``PtEngine`` construction only records its arguments and reads no weights; ``infer`` replies are
  generated deterministically from the request text and attached image bytes (MemER requests, whose system prompt
  contains ``keyframe_positions``, get JSON; a ``script`` can also give the exact reply text per call).
- Stage 3 (1006 plan 8.3): all three variants carry ``policy_seed`` (default 7); MemER uses the same two-side wiring.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tests._support.loaders import REPO, load_script

#: Fake frame shape H x W: width matches the real camera's 256 so the official text-overlay video has a fixed-height
#: text area as in real runs (narrow frames put each word on its own line and make frame height vary with the text,
#: and the official mimsave fails with "All images in a movie should have same size"); height is 8 to save memory
H, W = 8, 256
N_RESET_FRAMES = 3  # 2 demo frames + 1 initial frame
CHUNK_ROWS = 20  # action rows per fake-server reply (official code executes only the first 16)
EXEC_HORIZON = 16  # official Args.obs_horizon (hand-written)
THIRD_PARTY_ENV = "SGEVAL_THIRD_PARTY"
ORACLE = "ground-sg-oracle"
QWENVL = "ground-sg-qwenvl"
MEMER = "ground-sg-memer"
VARIANTS = (ORACLE, QWENVL, MEMER)
DATASET = "hard-verify"
POLICY_SEED = 7  # real runs in this version only pass 7 (plan 0.1)


def official_dir() -> Path:
    # When unset, use the current checkout's third_party (the daily gate in the main checkout takes this path); in a
    # worktree the submodules are empty, so point it at the main checkout explicitly, or the assert below fails (no skip).
    tp = os.environ.get(THIRD_PARTY_ENV) or str(REPO / "third_party")
    d = Path(tp) / "mme-vla" / "examples" / "robomme"
    assert (d / "eval.py").is_file(), f"official sources not found at {d}"
    return d


def official_sha256() -> dict[str, str]:
    d = official_dir()
    rels = ["eval.py", "env_runner.py", "utils.py", "subgoal_predictor.py", "subgoal_prediction/qwenvl/api.py"]
    return {r: hashlib.sha256((d / r).read_bytes()).hexdigest() for r in rels}


def print_official_sha() -> None:
    for rel, s in official_sha256().items():
        print(f"OFFICIAL_SOURCE {rel} sha256={s}")


# ---------------------------------------------------------------- production modules


def env_session():
    """After the repo split, ``EnvSession`` / ``NullRecorder`` / ``StepCapReached`` live in ``robomme_ood_eval.session``."""
    from robomme_ood_eval import session

    return session


def groundsg_client():
    return load_script("eval-official/groundsg_client.py")


def official_defs():
    return load_script("eval-official/official_defs.py")


# ---------------------------------------------------------------- fake env


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frame(v: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[...] = int(v) % 256
    img[0, 0] = [(int(v) * 7) % 256, (int(v) * 13) % 256, 1]
    return img


def obs_of(vals: list[int]) -> dict:
    return {
        "front_rgb_list": [frame(v) for v in vals],
        "wrist_rgb_list": [frame(v + 1) for v in vals],
        "joint_state_list": [np.full(7, v / 10.0, dtype=np.float64) for v in vals],
        "gripper_state_list": [np.array([v / 100.0, v / 100.0], dtype=np.float64) for v in vals],
        "eef_state_list": [np.zeros(6, dtype=np.float64) for _ in vals],
    }


def subgoal_at(n: int) -> str:
    """Oracle grounded subgoal: changes every 25 steps (has coordinates, so the official overlay draws a point)."""
    k = n // 25
    return f"pick up the cube at <{(k * 37) % 250}, {(k * 91) % 250}>"


def goal_of(task: str, ep: int) -> str:
    return f"goal {task} {ep}"


@dataclasses.dataclass
class Plan:
    """Fake env behavior: at step n report success / fail, terminate without a status (official result is unknown),
    or raise; if all are None, never terminate."""

    success_at: int | None = None
    fail_at: int | None = None
    unknown_at: int | None = None
    raise_at: int | None = None


class FakeEnv:
    def __init__(self, task: str, ep: int, plan: Plan):
        self.task, self.ep, self.plan = task, int(ep), plan
        self.n = 0
        self.actions: list[np.ndarray] = []
        self.resets = 0
        self.closed = False
        self.difficulty = "hard"
        self.unwrapped = self

    def _info(self, status: str | None) -> dict:
        info = {"grounded_subgoal_online": subgoal_at(self.n), "simple_subgoal_online": f"simple {self.n // 25}"}
        if status is not None:
            info["status"] = status
        return info

    def reset(self):
        self.resets += 1
        base = (self.ep * 7) % 150
        info = self._info("ongoing")
        info["task_goal"] = [goal_of(self.task, self.ep), "alt"]
        return obs_of([base + i for i in range(N_RESET_FRAMES)]), info

    def step(self, action):
        a = np.array(action, copy=True)
        self.actions.append(a)
        self.n += 1
        if self.plan.raise_at == self.n:
            raise RuntimeError(f"fake env step failure at {self.n}")
        v = int(sha(np.ascontiguousarray(a).tobytes() + str(self.n).encode())[:6], 16) % 250
        status: str | None = "ongoing"
        terminated = False
        if self.plan.success_at == self.n:
            status, terminated = "success", True
        elif self.plan.fail_at == self.n:
            status, terminated = "fail", True
        elif self.plan.unknown_at == self.n:
            status, terminated = None, True
        return obs_of([v]), 0.0, terminated, False, self._info(status)

    def close(self):
        self.closed = True


class World:
    """A set of fake envs: assigns a Plan per source_episode and records every fake env and the official builder's
    constructor arguments."""

    def __init__(self, plans: dict[int, Plan] | None = None, default: Plan | None = None):
        self.plans = dict(plans or {})
        self.default = default or Plan(success_at=40)
        self.envs: list[FakeEnv] = []
        self.builder_kwargs: list[dict] = []

    def new_env(self, task: str, source_episode: int) -> FakeEnv:
        env = FakeEnv(task, source_episode, self.plans.get(int(source_episode), self.default))
        self.envs.append(env)
        return env

    def official_builder_cls(self):
        """Stand-in for the official ``env_runner.py`` ``BenchmarkEnvBuilder`` (records constructor args by official
        keyword)."""
        world = self

        class FakeOfficialBuilder:
            def __init__(self, env_id, dataset, action_space, gui_render, max_steps):
                self.env_id = env_id
                world.builder_kwargs.append(dict(env_id=env_id, dataset=dataset, action_space=action_space,
                                                 gui_render=gui_render, max_steps=max_steps))

            def get_episode_num(self):
                return 50

            def make_env_for_episode(self, episode_id):
                return world.new_env(self.env_id, int(episode_id))

        return FakeOfficialBuilder


class NewSideBuilder:
    """Builder stand-in for the new-side ``EnvSession``: the builder_episode -> source_episode map is given by the
    caller (hand-written)."""

    def __init__(self, task: str, world: World, ep_map: dict[int, int]):
        self.task, self.world, self.ep_map = task, world, dict(ep_map)

    def make_env_for_episode(self, ep, max_steps=None):
        return self.world.new_env(self.task, self.ep_map[int(ep)])


# ---------------------------------------------------------------- fake MME-VLA server


def _arr_fp(a: Any) -> bytes:
    a = np.ascontiguousarray(np.asarray(a))
    return f"{a.dtype.str}{a.shape}".encode() + a.tobytes()


def request_fp(obj: dict) -> tuple[str, str]:
    """(Hand-written) request fingerprint: returns (kind, sha256)."""
    if obj.get("reset"):
        return "reset", sha(b"reset")
    if obj.get("add_buffer"):
        return "add_buffer", sha(_arr_fp(obj["images"]) + _arr_fp(obj["state"]) + str(obj["exec_start_idx"]).encode())
    parts = [_arr_fp(obj["observation/image"]), _arr_fp(obj["observation/wrist_image"]),
             _arr_fp(obj["observation/state"]), str(obj.get("prompt")).encode(),
             str(obj.get("grounded_subgoal")).encode(), str(obj.get("simple_subgoal")).encode()]
    return "infer", sha(b"|".join(parts))


class FakeServer:
    """Action chunks are derived deterministically from all request fingerprints since this episode's reset;
    records every request and every chunk sent."""

    def __init__(self):
        self.log: list[tuple[str, str, dict]] = []
        self.chunks: list[np.ndarray] = []
        self._h = hashlib.sha256()

    def handle(self, obj: dict) -> dict:
        kind, fp = request_fp(obj)
        self.log.append((kind, fp, {k: obj[k] for k in obj if k in ("prompt", "grounded_subgoal", "simple_subgoal",
                                                                      "exec_start_idx")}))
        if kind == "reset":
            self._h = hashlib.sha256()
            return {"reset_finished": True}
        self._h.update(fp.encode())
        if kind == "add_buffer":
            self.log[-1][2]["n_frames"] = int(np.asarray(obj["images"]).shape[0])
            return {"add_buffer_finished": True}
        seed = int(self._h.hexdigest()[:8], 16)
        acts = np.random.default_rng(seed).standard_normal((CHUNK_ROWS, 8)).astype(np.float32)
        self.chunks.append(acts.copy())
        return {"actions": acts}


class FakeClient:
    """In-process stand-in for ``MMEVLAWebsocketClientPolicy`` (interface: reset / add_buffer / infer)."""

    def __init__(self, server: FakeServer):
        self.server = server
        self._ws = None

    def reset(self):
        return self.server.handle({"reset": True})

    def add_buffer(self, buf):
        return self.server.handle(buf)

    def infer(self, obs):
        return self.server.handle(obs)


# ---------------------------------------------------------------- swift stand-in (loads no weights)


def memer_reply(h: int, n_images: int) -> str:
    """(Hand-written) MemER stand-in reply: valid JSON; when there is more than 1 input image and h%3==0, picks image
    1+h%n as the keyframe, otherwise an empty list."""
    pos = [1 + h % n_images] if n_images > 1 and h % 3 == 0 else []
    return json.dumps({"current_subtask": f"pick up the cube at <|box_start|>({h % 1000},{(h // 1000) % 1000})<|box_end|>",
                       "keyframe_positions": pos})


class FakeSwift:
    """Stand-ins for the three ``swift.llm`` names. ``PtEngine`` construction only records args; ``infer`` replies
    are generated deterministically from the request.

    ``script``: list of exact reply texts, one per call (falls back to deterministic generation once used up);
    like real swift, ``InferRequest`` puts ``messages`` / ``images`` / ``videos`` / ``objects`` on same-named
    attributes (and also keeps ``kw`` as is)."""

    def __init__(self, script: list[str] | None = None):
        self.engines: list[dict] = []
        self.requests: list[dict] = []
        self.script = list(script or [])
        outer = self

        class InferRequest:
            def __init__(self, **kw):
                self.kw = kw
                self.messages = kw.get("messages")
                self.images = list(kw.get("images") or [])
                self.videos = list(kw.get("videos") or [])
                self.objects = kw.get("objects") or {}

        class RequestConfig:
            def __init__(self, **kw):
                self.kw = kw

        class PtEngine:
            def __init__(self, model_id_or_path, adapters, attn_impl):
                outer.engines.append(dict(model_id_or_path=model_id_or_path, adapters=list(adapters),
                                          attn_impl=attn_impl))

            def infer(self, reqs, request_config=None):
                (req,) = reqs
                kw = req.kw
                imgs = [Path(x).read_bytes() for x in kw["images"]]
                img = imgs[0]
                text = json.dumps(kw["messages"], sort_keys=True)
                rec = {"messages": kw["messages"], "image_sha": sha(img), "has_video": "videos" in kw,
                       "objects": kw.get("objects"), "config": dict(request_config.kw),
                       "image_shas": [sha(b) for b in imgs]}
                outer.requests.append(rec)
                h = int(sha(text.encode() + b"".join(imgs))[:6], 16)
                if outer.script:
                    content = outer.script.pop(0)
                elif "keyframe_positions" in kw["messages"][0]["content"]:
                    content = memer_reply(h, len(imgs))
                else:
                    content = f"pick up the cube at <|box_start|>({h % 1000},{(h // 1000) % 1000})<|box_end|>"
                msg = type("M", (), {"content": content})()
                choice = type("C", (), {"message": msg})()
                return [type("R", (), {"choices": [choice]})()]

        self.names = {"PtEngine": PtEngine, "InferRequest": InferRequest, "RequestConfig": RequestConfig}


# ---------------------------------------------------------------- two-side drivers


ADAPTER = "/fake/qwenvl/grounded_subgoal/checkpoint-1200"
MEMER_ADAPTER = "/fake/memer/grounded_subgoal/checkpoint-1300"


def adapters_of(variant: str) -> dict:
    """Adapter keywords for a variant (hand-written pairing: QwenVL gets only the QwenVL one, MemER only the MemER
    one, Oracle gets neither)."""
    return {"qwenvl_groundSG_adapter_path": ADAPTER if variant == QWENVL else None,
            "memer_adapter_path": MEMER_ADAPTER if variant == MEMER else None}


def identity(task: str = "PickXtimes", source_episode: int = 3, builder_episode: int = 0, seed: int = 510300) -> dict:
    return {"task": task, "tier": "xhard0", "seed": seed, "candidate": None, "builder_episode": builder_episode,
            "source_episode": source_episode, "spec_sha256": None, "key": f"{task}_xhard0_{seed}"}


def seat_info(variant: str, max_steps: int, tmp: Path, port: int = 18120, policy_seed: int = POLICY_SEED) -> dict:
    return {"policy": "groundsg", "seat": "00", "host": "127.0.0.1", "port": port, "dataset": DATASET,
            "max_steps": max_steps, "strict_cap": False, "groundsg_variant": variant, **adapters_of(variant),
            "policy_seed": policy_seed, "trace_root": str(tmp / "trace"), "out": str(tmp)}


class NewSide:
    """New side: real ``EnvSession`` + ``groundsg_client`` (fake env, fake server, swift stand-in)."""

    def __init__(self, variant: str, max_steps: int, tmp: Path, world: World, *, strict_cap: bool = False,
                 port: int = 18120, real_client: bool = False, policy_seed: int = POLICY_SEED,
                 swift: FakeSwift | None = None, server: FakeServer | None = None):
        """With ``real_client=True``, use the production default client factory (a real websocket client connecting
        to the loopback fake server on ``port``)."""
        self.variant, self.max_steps, self.tmp, self.world, self.strict_cap = variant, max_steps, tmp, world, strict_cap
        self.port = port
        self.server = server or FakeServer()
        self.swift = swift or FakeSwift()
        self.mc = groundsg_client()
        factory = None if real_client else (lambda h, p, ep: FakeClient(self.server))
        self.ctx = self.mc.make_policy_context(seat_info(variant, max_steps, tmp, port, policy_seed),
                                               client_factory=factory, qwen_extra=self.swift.names)
        self.sessions: list[Any] = []

    def run(self, ident: dict, *, attempt: int = 1) -> dict:
        ec = env_session()
        sess = ec.EnvSession(ident["task"], ident["builder_episode"], max_steps=self.max_steps,
                             builder=NewSideBuilder(ident["task"], self.world,
                                                    {ident["builder_episode"]: ident["source_episode"]}),
                             step_cap=self.max_steps if self.strict_cap else None, dataset=DATASET)
        sess.build()
        tag = f"{ident['key']}.a{attempt}"
        conn = {"host": "127.0.0.1", "port": self.port, "max_steps": self.max_steps, "policy": "groundsg", "seat": "00",
                "dataset": DATASET, "strict_cap": self.strict_cap, "groundsg_variant": self.variant,
                **adapters_of(self.variant),
                "trace_root": str(self.tmp / "trace"), "trace_dir": str(self.tmp / "trace" / tag),
                "episode_tag": tag, "rec_dir": str(self.tmp / "rec" / tag), "policy_context": self.ctx}
        res = self.mc.run_episode(sess, ident, conn, ec.NullRecorder())
        sess.close()
        self.sessions.append(sess)
        res["_session_steps"] = sess.steps
        res["_cap_hit"] = sess.cap_hit
        return res


def read_trace(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


#: Link fields on step rows when the language ledger is on (call_id is each side's own ledger number and is not
#: compared item by item across sides)
LANG_LINK_FIELDS = ("source_call_id", "chunk_index")


def trace_parts(rows: list[dict]) -> dict:
    """Split a trace by kind; drop fields that always differ between sides (header.route, end.side, and the
    language-ledger link ids on step rows)."""
    out: dict[str, list] = {"request": [], "response": [], "step": [], "demo": [], "history": [], "end": []}
    for r in rows:
        k = r["kind"]
        if k in out:
            r = dict(r)
            r.pop("side", None)
            for f in LANG_LINK_FIELDS:
                r.pop(f, None)
            out[k].append(r)
    return out


_BASE_TRACE_PARTS = trace_parts  # used by legacy_trace_parts (still the original after monkeypatch swaps trace_parts)

#: Fields added only on new-side end rows in stage 2 S1 (original side R1 does not write them)
NEW_ONLY_END = ("steps_attempted", "steps_observed", "frames_recorded", "omitted_timeout_frames", "no_frame",
                "official_source", "official_videos")


def legacy_trace_parts(rows: list[dict]) -> dict:
    """Map new-side contract fields back to the BASE convention, then split by kind (original-side rows unchanged;
    1005 plan S1).

    Only strips record fields the new side added, as a reversible mapping: on end rows the official raw return value
    ``success_flag`` goes back to ``terminal_reason`` and ``NEW_ONLY_END`` is dropped; missing-observation steps
    (``observed=false``) go back to the old form. ``status``, actions, frames and requests are never touched, so
    differences are still reported. For item-by-item two-side comparison, swap it in locally with
    ``monkeypatch.setattr(F, "trace_parts", F.legacy_trace_parts)``."""
    conv = []
    for r in rows:
        r = dict(r)
        if r.get("kind") == "end" and "success_flag" in r:
            r["terminal_reason"] = r.pop("success_flag")
            for k in NEW_ONLY_END:
                r.pop(k, None)
        if r.get("kind") == "step" and r.get("observed") is False:
            assert r.pop("missing_reason")
            r.pop("observed")
            r.update(terminated=False, truncated=False, status="error")
        conv.append(r)
    return _BASE_TRACE_PARTS(conv)


# ---------------------------------------------------------------- loopback websocket fake server (slow tests)


class LoopbackServer:
    """Websocket fake server on ``127.0.0.1``: sends metadata on handshake, then msgpack-decodes each message, hands
    it to ``FakeServer`` and replies."""

    def __init__(self, server: FakeServer):
        import threading

        import websockets.sync.server as wss
        from openpi_client import msgpack_numpy

        self.fake = server
        packer = msgpack_numpy.Packer()

        def handler(ws):
            ws.send(packer.pack({"server": "fake"}))
            try:
                for msg in ws:
                    ws.send(packer.pack(self.fake.handle(msgpack_numpy.unpackb(msg))))
            except Exception:  # noqa: BLE001 client disconnected
                pass

        self._srv = wss.serve(handler, "127.0.0.1", 0, compression=None, max_size=None)
        self.port = self._srv.socket.getsockname()[1]
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()

    def close(self):
        self._srv.shutdown()
        self._t.join(timeout=10)
