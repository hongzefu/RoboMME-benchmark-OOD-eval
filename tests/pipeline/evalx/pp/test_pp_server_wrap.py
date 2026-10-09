"""Equivalence and subgoal semantics of the PonderPounce server wrapper ``pp_server_wrap.py`` (S5) (daily gate, CPU).

The wrapper subclasses ``ponderpounce.eval.robomme_server.PonderPounceRoboMMEServer`` (submodule ``723df357``, file
sha256 pinned). This test drives one episode each through the original class and the wrapper using the **real parent
methods** (``on_episode_start``/``on_observation``/``_fire_s2``/``_visible_cognition``/``_fire_s1``/``_dispense``/
``_hold`` verbatim), replacing only the two heavy models with CPU stubs:

- System 2: the module-global ``SoftS2SessionContext`` is replaced by a scripted stub (cognition tensor and subgoal
  text by fire index); the parent ``_fire_s2`` timing, ``ready_at_ns`` and ``cognitions`` truncation run as usual;
- System 1: ``_s1`` becomes a deterministic stub (``noise_spec``, ``null_cognition``, ``predict_action``); the parent
  still draws noise from this episode's ``noise_rng``.

Per step it compares actions (dtype/shape/bytes), RNG state, ``cursor``, chunk contents, fire counts and timing; apart
from the wrapper's added ``subgoal`` and ``_sgeval_audit`` keys everything must match, printing
``PP_SERVER_ACTION_EQ=PASS``. Stage 3 additionally checks: with ``SGEVAL_AUDIT=0`` replies carry no audit key and are
otherwise identical to the enabled run; when enabled, the System 2 generation blocks in the audit key match the stub's
real ``fire`` results one to one (none extra, none missing), printing ``PP_AUDIT_OBS_EQ=PASS``. The subgoals returned by
the wrapper are compared step by step against expectations derived independently from the timing formula (``None``
before the first subgoal is visible, the old subgoal before ``ready_at_ns``, unchanged after the chunk is exhausted).

The parent depends on vla-eval, torch and transformers; the main checkout's ``.venv`` lacks vla-eval, so the
equivalence part runs this file's ``--child`` branch in a subprocess with the main checkout's client-env interpreter
(``envs/client-env/.venv``, with vla-eval 0.7.0 and torch, CPU). PonderPounce sources are referenced read-only from the
main checkout's ``third_party/PonderPounce`` (override with ``SGEVAL_PP_PYTHON``/``SGEVAL_THIRD_PARTY``). A missing
interpreter or source fails the test, no skip. The subprocess inherits this process's ``PYTHONPATH`` (so the resource
guard's sitecustomize applies too).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

try:  # the subprocess (client-env interpreter) has no pytest; the slow marker is only needed on the pytest side
    import pytest
except ModuleNotFoundError:  # pragma: no cover - subprocess
    class _NoMark:
        def __getattr__(self, name):
            return lambda f: f

    pytest = type("pytest", (), {"mark": _NoMark()})

REPO = Path(__file__).resolve().parents[4]
WRAP = REPO / "src" / "robomme_ood_eval" / "servers" / "pp_server_wrap.py"  # after the repo split the wrapper lives in the eval package
PP_GITLINK = "723df35762bb641e1d520e4fa9359b98644adc21"
PARENT_REL = "ponderpounce/eval/robomme_server.py"
PARENT_SHA256 = "0664c0abc1598c68f75a8f2eaf2889062a34e39d4e1d75269512907a8f1cb135"
CLIENT_ENV_PY = "envs/client-env/.venv/bin/python"  # after the repo split the client extension env is envs/client-env/.venv

# Timing (ms) and the System 2 stub's subgoal scripts; dt = 1000/20 = 50 ms
DT_MS = 50
CHUNK = 5
N_STEPS = 200
SCHEDULES = [
    # default timing: Ponder/Pounce 1000 ms each, compute delay 400 ms
    {"name": "p1000_s1000_d400", "s2_ms": 1000, "s1_ms": 1000, "delay_ms": 400,
     "subgoals": ["", "", "pick up the cube at [612, 247]", "", "put it at [10, 990]", "", "", "press at [500, 500]"]},
    # unequal timing, delay longer than the period: several invisible cognitions at once, the newest visible one may
    # have an empty subgoal
    {"name": "p500_s1000_d1300", "s2_ms": 500, "s1_ms": 1000, "delay_ms": 1300,
     "subgoals": ["", "open the drawer", "", "", "", "grab at [100, 900]", "", "", "", "", "", "lift"]},
]


# -- Subprocess: drive the real parent methods in an interpreter with vla-eval/torch ----------------


def _child_main(mode: str) -> None:  # pragma: no cover - runs in the subprocess
    if mode == "s2input":
        return _child_s2input()
    import asyncio
    import importlib.util
    import inspect
    import re
    import types

    import numpy as np
    import torch

    import ponderpounce.eval.robomme_server as rs

    spec = importlib.util.spec_from_file_location("pp_server_wrap", os.environ["SGEVAL_PP_WRAP"])
    wrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrap)
    assert wrap.PonderPounceRoboMMEServer is rs.PonderPounceRoboMMEServer
    ns = rs.NS_PER_MS

    class StubS2Context:
        """Stub of ``SoftS2SessionContext``: the n-th ``fire`` yields ``subgoals[n % L]`` and a deterministic cognition."""

        subgoals: list = []

        def __init__(self, *, session, system2, device, dtype, max_new_tokens):
            self.n = 0

        def fire(self, pils):
            img = float(np.asarray(pils[0], dtype=np.float32).mean()) / 255.0
            text = self.subgoals[self.n % len(self.subgoals)]
            cog = torch.full((2, 4), float(self.n), dtype=torch.float32) + img
            self.n += 1
            return rs.S2ContextResult(subgoal_tokens=None, subgoal_text=text, cognition=cog, kind="stub",
                                      gate_score=None, n_input_frames=1)

    class StubS1:
        """Stub of ``LocalSystem1``: the action chunk depends only on the inputs and the noise drawn by the parent."""

        action_chunk_size, action_dim = CHUNK, 8
        noise_spec = (CHUNK, 8)
        null_cognition = torch.zeros(2, 4)

        def predict_action(self, **kw):
            out = kw["noise"][0] * 0.05 + kw["cognition"].mean() * 0.1 + kw["images"].mean() * 0.01
            out = out + kw["age_ms"][0] * 1e-4
            if kw.get("proprio") is not None:
                out = out + kw["proprio"][0].sum() * 1e-3
            return out.unsqueeze(0)

    def s1_transform(pil):
        return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def build(cls, sch):
        srv = cls.__new__(cls)
        super(rs.PonderPounceRoboMMEServer, srv).__init__()
        srv._device, srv._dtype = torch.device("cpu"), torch.float32
        srv._dt_ns = int(round(1e9 / (1000.0 / DT_MS)))
        srv._s2_period_ns, srv._s1_period_ns = sch["s2_ms"] * ns, sch["s1_ms"] * ns
        srv._wait_first_cognition = True
        srv._camera_keys = ("agentview", "wrist")
        srv._seed = 0
        srv._episodes, srv._episode_counts = {}, {}
        srv._s2_delay_ns = int(sch["delay_ms"] * ns)
        srv._model, srv._norm_stats = None, None
        srv._s1 = StubS1()
        srv._s2 = types.SimpleNamespace(num_cognition_tokens=2)
        srv._s2_processor, srv._s2_max_new_tokens, srv._subgoal_grounded = None, 40, False
        srv._s1_transform = s1_transform
        srv._adapter = rs.ObsAdapter(camera_keys=srv._camera_keys, proprio_dim=8, max_demo_frames=0,
                                     norm_stats=None, demo_fps=0.0, env_fps=1000.0 / DT_MS)
        srv._s1_tokenizer, srv._s1_max_token_len, srv._s1_num_camera_slots = None, 0, 0
        return srv

    # every attribute set by the parent __init__ must be set by the stub (fails here first when upstream adds fields)
    init_src = inspect.getsource(rs.PonderPounceRoboMMEServer.__init__)
    init_attrs = sorted(set(re.findall(r"self\.(_\w+)\s*(?::[^=\n]+)?=", init_src)))

    class Ctx:
        def __init__(self, sid):
            self.session_id = sid
            self.sent = []

        async def send_action(self, a):
            self.sent.append(a)

    def obs_at(t):
        rng = np.random.default_rng([7, t])
        o = {"images": {"agentview": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
                        "wrist": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)},
             "task_description": "pick up the cube", "states": rng.normal(size=8).astype(np.float32)}
        if t == 0:
            o["video_history"] = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(3)]
            o["episode_restart"] = True
        return o

    def arr_rec(a):
        a = np.ascontiguousarray(np.asarray(a))
        return [a.dtype.str, list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]

    async def run(cls, sch):
        StubS2Context.subgoals = sch["subgoals"]
        srv = build(cls, sch)
        missing = [a for a in init_attrs if not hasattr(srv, a)]
        sid = "T|0|0"
        ctx = Ctx(sid)
        await srv.on_episode_start({"task": {"name": "T", "env_id": "T", "episode_idx": 0},
                                    "recording": {"sid": sid, "eid": sid, "eval_id": "", "db_path": ""}}, ctx)
        steps = []
        for t in range(N_STEPS):
            await srv.on_observation(obs_at(t), ctx)
            ep = srv._episodes[sid]
            a = ctx.sent[-1]
            steps.append({
                "keys": sorted(a), "actions": arr_rec(a["actions"]), "subgoal": a.get("subgoal", "<absent>"),
                "audit": a.get("_sgeval_audit", "<absent>"),
                "tick": ep.tick, "n_s1": ep.n_s1_fires, "n_s2": ep.n_s2_fires, "s1_started": ep.s1_started,
                "s1_next": ep.s1_next_fire_ns, "s2_next": ep.s2_next_fire_ns, "n_cogs": len(ep.cognitions),
                "cursor": None if ep.chunk is None else ep.chunk.cursor,
                "chunk": None if ep.chunk is None else arr_rec(ep.chunk.actions.numpy()),
                "rng": hashlib.sha256(ep.noise_rng.get_state().numpy().tobytes()).hexdigest(),
                "active_subgoal": ep.active_subgoal})
        await srv.on_episode_end({}, ctx)
        return steps, missing

    def compare(a_steps, b_steps):
        bad = 0
        for x, y in zip(a_steps, b_steps):
            xs = {k: v for k, v in x.items() if k not in ("subgoal", "keys", "audit")}
            ys = {k: v for k, v in y.items() if k not in ("subgoal", "keys", "audit")}
            keys_ok = sorted(set(y["keys"]) - {"subgoal", "_sgeval_audit"}) == x["keys"] and "subgoal" in y["keys"]
            bad += int(xs != ys or not keys_ok)
        return bad + abs(len(a_steps) - len(b_steps))

    rs.SoftS2SessionContext = StubS2Context
    out = {"parent_file": rs.__file__, "init_attrs": init_attrs, "schedules": []}
    for sch in SCHEDULES:
        base_steps, miss = asyncio.run(run(rs.PonderPounceRoboMMEServer, sch))
        wrap_steps, _ = asyncio.run(run(wrap.SubgoalReportingServer, sch))
        rec = {"name": sch["name"], "mismatch": compare(base_steps, wrap_steps), "missing_attrs": miss,
               "base_has_subgoal_key": any("subgoal" in s["keys"] for s in base_steps),
               "subgoals": [s["subgoal"] for s in wrap_steps], "cursor": [s["cursor"] for s in wrap_steps],
               "n_s1": [s["n_s1"] for s in wrap_steps], "n_s2": [s["n_s2"] for s in wrap_steps],
               "s1_started": [s["s1_started"] for s in wrap_steps], "rng_final": wrap_steps[-1]["rng"]}
        # Observation switch control: with SGEVAL_AUDIT=0 the wrapper adds no audit key; everything else (actions, RNG,
        # cursor, counts, subgoals) matches the enabled run item by item
        prev = os.environ.get("SGEVAL_AUDIT")
        os.environ["SGEVAL_AUDIT"] = "0"
        try:
            off_steps, _ = asyncio.run(run(wrap.SubgoalReportingServer, sch))
        finally:
            if prev is None:
                os.environ.pop("SGEVAL_AUDIT", None)
            else:
                os.environ["SGEVAL_AUDIT"] = prev
        strip = lambda st: [{k: v for k, v in x.items() if k not in ("audit", "keys")} for x in st]  # noqa: E731
        rec["audit_off_has_key"] = any("_sgeval_audit" in x["keys"] for x in off_steps)
        rec["audit_on_all_keyed"] = all("_sgeval_audit" in x["keys"] for x in wrap_steps)
        rec["audit_on_off_mismatch"] = sum(int(a_ != b_) for a_, b_ in zip(strip(off_steps), strip(wrap_steps))) \
            + abs(len(off_steps) - len(wrap_steps))
        # pp_generation: a list with one block per fire, None when no fire (MERGE-1 aligns with the keys the client
        # language log reads)
        fires = [f for x in wrap_steps for f in (x["audit"]["pp_generation"] or [])]
        rec["audit_gen_keys_ok"] = all(f["subgoal_raw"] == f["subgoal_text"] and f["reasoning"] == f["reasoning_text"]
                                       and f["params"]["fire_index"] == f["fire_index"] for f in fires)
        rec["audit_text_reconstructed"] = all(c.get("text_reconstructed") is True for x in wrap_steps
                                              for c in x["audit"]["channels"])
        rec["audit_fire_subgoals"] = [f["subgoal_text"] for f in fires]
        rec["audit_fire_index"] = [f["fire_index"] for f in fires]
        rec["audit_kinds"] = sorted({f["kind"] for f in fires})
        if mode == "mutant":  # comparator self-check: a "bad wrapper" that draws one extra random number must be caught
            class RngMutant(wrap.SubgoalReportingServer):
                def _fire_s1(self, ep, obs, now):
                    torch.randn(1, generator=ep.noise_rng)
                    super()._fire_s1(ep, obs, now)

            class CursorMutant(wrap.SubgoalReportingServer):
                def _dispense(self, ep, obs):
                    out_ = super()._dispense(ep, obs)
                    if ep.chunk is not None and ep.chunk.cursor < CHUNK:
                        ep.chunk.cursor += 1
                    return out_

            rec["mutant_rng"] = compare(base_steps, asyncio.run(run(RngMutant, sch))[0])
            rec["mutant_cursor"] = compare(base_steps, asyncio.run(run(CursorMutant, sch))[0])
        out["schedules"].append(rec)
    print("CHILD_RESULT " + json.dumps(out), flush=True)


# -- FIX-3: real upstream SoftS2SessionContext + CPU fake System 2 / fake tokenizer, checks S2 input decoding --

S2IN_STEPS = 120
S2IN_SPECIAL = ["<|im_start|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|fim_pad|>", "<|fim_prefix|>"]
S2IN_SCRIPTS = ["pick up the cube at [612, 247]", None, ("think", "put it at [10, 990]"), "ROLLBACK"]


def _child_s2input() -> None:  # pragma: no cover - runs in the subprocess
    """Real parent ``on_observation``/``_fire_s2`` with the **real upstream** ``SoftS2SessionContext`` (fire/_append/
    _restore verbatim) as context; only the System 2 VLM forward becomes a CPU fake model (hidden state = deterministic
    function of the tokens seen in context) and the tokenizer a per-character fake. The original class, wrapper with
    audit on, and wrapper with audit off each run one episode; actions/RNG/timing/fake-model call counts and final
    context tokens are compared per step. The ``input_text`` in the wrapper's generation blocks is compared per fire
    against the expectation decoded independently from the context tokens the fake model recorded itself."""
    import asyncio
    import contextlib
    import importlib.util
    import re
    import types

    import numpy as np
    import torch

    import ponderpounce.eval.robomme_server as rs
    from ponderpounce.inference import append_context as ac

    spec = importlib.util.spec_from_file_location("pp_server_wrap", os.environ["SGEVAL_PP_WRAP"])
    wrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrap)
    assert rs.SoftS2SessionContext is ac.SoftS2SessionContext
    ns = rs.NS_PER_MS
    sp = {t: i + 1 for i, t in enumerate(S2IN_SPECIAL)}
    inv = {v: k for k, v in sp.items()}
    TRIG, COG, VS, PAD, VE = sp["<|fim_prefix|>"], sp["<|fim_pad|>"], sp["<|vision_start|>"], sp["<|image_pad|>"], \
        sp["<|vision_end|>"]
    VOCAB = 1200

    class FakeTok:
        unk_token_id = None

        def convert_tokens_to_ids(self, t):
            return sp.get(t)

        def convert_ids_to_tokens(self, i):
            return inv.get(int(i))

        def __call__(self, text, add_special_tokens=False):
            ids, i = [], 0
            while i < len(text):
                for t, v in sp.items():
                    if text.startswith(t, i):
                        ids.append(v)
                        i += len(t)
                        break
                else:
                    ids.append(1000 + ord(text[i]))
                    i += 1
            return types.SimpleNamespace(input_ids=ids)

        def decode(self, ids, skip_special_tokens=False):
            return "".join(inv[i] if i in inv else chr(i - 1000) for i in map(int, ids))

    def chars(s):
        return [1000 + ord(c) for c in s]

    class FakeImageProcessor:
        merge_size = 2

        def __call__(self, images, return_tensors="pt"):
            px = torch.tensor([[float(np.asarray(im, dtype=np.float32).mean())] * 3 for im in images for _ in range(4)])
            return {"pixel_values": px, "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images))}

    class FakeProcessor:
        def __init__(self):
            self.tokenizer, self.image_processor = FakeTok(), FakeImageProcessor()

    class Layer:
        keys = None
        max_cache_len = 10 ** 6

        def __init__(self):
            self.cumulative_length = torch.zeros((), dtype=torch.long)

    class Cache:
        def __init__(self):
            self.layers, self.buf = [Layer()], []

    vcfg = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))

    class Inner:
        config = vcfg

        @staticmethod
        def get_vision_position_ids(start, grid, t, merge, device=None):
            n = int(grid.prod().item()) // merge ** 2
            return torch.arange(n, dtype=torch.long).view(1, -1).expand(3, -1) + int(start)

    class FakeS2:
        """Fake System 2: the forward writes tokens into cache.buf (truncated at cumulative_length, consistent with a
        real rollback); the hidden state encodes ``[number of vision segments, tokens after the latest vision
        segment]``; lm_head picks a script by observation index: transition / nontransition / reasoning+transition /
        never terminates (triggers the max_new_tokens rollback)."""

        num_cognition_tokens = 2
        effective_context_cap = 10 ** 6

        def __init__(self, current_obs_only):
            self.current_obs_only = current_obs_only
            self._subgoal_trigger_id, self._cognition_token_id = TRIG, COG
            self.n_forward = self.n_lm_head = 0
            self.decision_ctx = []  # full context tokens after each observation is appended (k==0)
            self.last_cache = None
            outer = self

            class VLM:
                config = vcfg
                model = Inner()

                @staticmethod
                def lm_head(h):
                    return outer._lm_head(h)

            self.vlm = VLM()

        def _new_static_cache(self):
            self.last_cache = Cache()
            self.last_cache.n_fwd = 0
            return self.last_cache

        def _build_inputs_embeds(self, *, input_ids, cognition_mask, pixel_values, image_grid_thw, s2_total_queries):
            return input_ids.to(torch.float32).unsqueeze(-1)

        def _isolation_stash(self, *a, **k):
            return contextlib.nullcontext()

        def _run_cached_language_model(self, *, position_ids, attention_mask, inputs_embeds, past_key_values,
                                       compile_ok):
            self.n_forward += 1
            c = past_key_values
            del c.buf[int(c.layers[0].cumulative_length.item()):]
            ids = [int(x) for x in inputs_embeds[0, :, 0].tolist()]
            c.buf.extend(ids)
            c.layers[0].cumulative_length.fill_(len(c.buf))
            n_vend = sum(1 for x in c.buf if x == VE)
            last_ve = max([i for i, x in enumerate(c.buf) if x == VE], default=-1)
            k = len(c.buf) - 1 - last_ve
            feat = torch.tensor([float(n_vend), float(k)], dtype=torch.float32)
            c.n_fwd += 1
            if k == 0 and c.n_fwd > 1:  # the first forward is the prefix appended by reset() (ending in demo images), not an observation decision point
                self.decision_ctx.append(list(c.buf))
            return types.SimpleNamespace(last_hidden_state=feat.view(1, 1, 2).expand(1, len(ids), 2).clone())

        def _lm_head(self, h):
            self.n_lm_head += 1
            n_vend, k = int(h[0, 0].item()), int(h[0, 1].item())
            script = S2IN_SCRIPTS[(n_vend // 2) % len(S2IN_SCRIPTS)]
            if script is None:
                seq = [COG]
            elif script == "ROLLBACK":
                seq = [TRIG] + chars("x" * 200)
            elif isinstance(script, tuple):
                seq = [TRIG] + chars(script[0]) + [TRIG] + chars(script[1]) + [COG]
            else:
                seq = [TRIG] + chars(script) + [COG]
            logits = torch.zeros(1, VOCAB)
            logits[0, seq[k] if k < len(seq) else COG] = 10.0
            return logits

    class StubS1:
        action_chunk_size, action_dim = CHUNK, 8
        noise_spec = (CHUNK, 8)
        null_cognition = torch.zeros(2, 2)

        def predict_action(self, **kw):
            out = kw["noise"][0] * 0.05 + kw["cognition"].mean() * 0.1 + kw["images"].mean() * 0.01
            return (out + kw["age_ms"][0] * 1e-4 + kw["proprio"][0].sum() * 1e-3).unsqueeze(0)

    def s1_transform(pil):
        return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def build(cls, current_obs_only, s2_ms=250, s1_ms=500, delay_ms=300):
        srv = cls.__new__(cls)
        super(rs.PonderPounceRoboMMEServer, srv).__init__()
        srv._device, srv._dtype = torch.device("cpu"), torch.float32
        srv._dt_ns = int(round(1e9 / (1000.0 / DT_MS)))
        srv._s2_period_ns, srv._s1_period_ns = s2_ms * ns, s1_ms * ns
        srv._wait_first_cognition = True
        srv._camera_keys = ("agentview", "wrist")
        srv._seed = 0
        srv._episodes, srv._episode_counts = {}, {}
        srv._s2_delay_ns = int(delay_ms * ns)
        srv._model, srv._norm_stats = None, None
        srv._s1 = StubS1()
        srv._s2 = FakeS2(current_obs_only)
        srv._s2_processor, srv._s2_max_new_tokens, srv._subgoal_grounded = FakeProcessor(), 40, False
        srv._s1_transform = s1_transform
        srv._adapter = rs.ObsAdapter(camera_keys=srv._camera_keys, proprio_dim=8, max_demo_frames=0,
                                     norm_stats=None, demo_fps=0.0, env_fps=1000.0 / DT_MS)
        srv._s1_tokenizer, srv._s1_max_token_len, srv._s1_num_camera_slots = None, 0, 0
        return srv

    def obs_at(t):
        rng = np.random.default_rng([11, t])
        o = {"images": {"agentview": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
                        "wrist": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)},
             "task_description": "pick up the cube", "states": rng.normal(size=8).astype(np.float32)}
        if t == 0:
            o["video_history"] = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(3)]
            o["episode_restart"] = True
        return o

    def img_sha(a):
        a = np.ascontiguousarray(np.asarray(a))
        h = hashlib.sha256()
        h.update(f"{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
        return h.hexdigest()

    def arr_rec(a):
        a = np.ascontiguousarray(np.asarray(a))
        return [a.dtype.str, list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]

    class Ctx:
        def __init__(self, sid):
            self.session_id, self.sent = sid, []

        async def send_action(self, a):
            self.sent.append(a)

    async def run(cls, current_obs_only):
        srv = build(cls, current_obs_only)
        sid = "T|0|0"
        ctx = Ctx(sid)
        await srv.on_episode_start({"task": {"name": "T", "env_id": "T", "episode_idx": 0},
                                    "recording": {"sid": sid, "eid": sid, "eval_id": "", "db_path": ""}}, ctx)
        steps, fires = [], []
        for t in range(S2IN_STEPS):
            await srv.on_observation(obs_at(t), ctx)
            ep = srv._episodes[sid]
            a = ctx.sent[-1]
            for b in ((a.get("_sgeval_audit") or {}).get("pp_generation") or []):
                fires.append(dict(b, _t=t))
            steps.append({"actions": arr_rec(a["actions"]), "tick": ep.tick, "n_s1": ep.n_s1_fires,
                          "n_s2": ep.n_s2_fires, "s1_next": ep.s1_next_fire_ns, "s2_next": ep.s2_next_fire_ns,
                          "cursor": None if ep.chunk is None else ep.chunk.cursor,
                          "chunk": None if ep.chunk is None else arr_rec(ep.chunk.actions.numpy()),
                          "rng": hashlib.sha256(ep.noise_rng.get_state().numpy().tobytes()).hexdigest(),
                          "active_subgoal": ep.active_subgoal, "subgoal": a.get("subgoal", "<absent>")})
        ctx_tokens = list(srv._s2.last_cache.buf) if srv._s2.last_cache is not None else None
        s2 = srv._s2
        await srv.on_episode_end({}, ctx)
        return {"steps": steps, "fires": fires, "n_forward": s2.n_forward, "n_lm_head": s2.n_lm_head,
                "ctx_tokens": ctx_tokens, "decision_ctx": s2.decision_ctx}

    def oracle_decode(ids):
        out, i = [], 0
        while i < len(ids):
            if ids[i] == VS and i + 1 < len(ids) and ids[i + 1] == PAD:
                j = i + 1
                while j < len(ids) and ids[j] == PAD:
                    j += 1
                out.append("<image>")
                i = j + 1 if j < len(ids) and ids[j] == VE else j
                continue
            out.append(inv[ids[i]] if ids[i] in inv else chr(ids[i] - 1000))
            i += 1
        return "".join(out)

    class ExtraHeadMutant(wrap.SubgoalReportingServer):
        """Bad wrapper: calls lm_head once more after every S2 (one extra model call) -- the comparator must catch it."""

        def _fire_s2(self, ep, obs, now):
            super()._fire_s2(ep, obs, now)
            self._s2.vlm.lm_head(torch.zeros(1, 2))

    def strip(r):
        return [{k: v for k, v in s.items() if k != "subgoal"} for s in r["steps"]]

    def mismatch(a, b):
        n = sum(int(x != y) for x, y in zip(strip(a), strip(b))) + abs(len(a["steps"]) - len(b["steps"]))
        n += int(a["n_forward"] != b["n_forward"]) + int(a["n_lm_head"] != b["n_lm_head"])
        n += int(a["ctx_tokens"] != b["ctx_tokens"])
        return n

    out = {"variants": []}
    for coo in (False, True):
        base = asyncio.run(run(rs.PonderPounceRoboMMEServer, coo))
        on = asyncio.run(run(wrap.SubgoalReportingServer, coo))
        prev = os.environ.get("SGEVAL_AUDIT")
        os.environ["SGEVAL_AUDIT"] = "0"
        try:
            off = asyncio.run(run(wrap.SubgoalReportingServer, coo))
        finally:
            if prev is None:
                os.environ.pop("SGEVAL_AUDIT", None)
            else:
                os.environ["SGEVAL_AUDIT"] = prev
        mut = asyncio.run(run(ExtraHeadMutant, coo))
        dctx = on["decision_ctx"]
        prefix_len = len(dctx[0]) - 2 * (4 + 2)  # first decision point = prefix + one observation segment (2 images x (4 pads + start/end markers))
        checks = []
        for j, f in enumerate(on["fires"]):
            if j >= len(dctx):
                checks.append({"j": j, "ok": False, "why": "not enough decision points"})
                continue
            want_base = 0 if j == 0 else (prefix_len if coo else len(dctx[j - 1]))
            want = oracle_decode(dctx[j][want_base:])
            got = re.sub(r"<image:\d+>", "<image>", f.get("input_text") or "")
            o = obs_at(f["_t"])
            imgs = f.get("input_images") or []
            want_imgs = ([{"source": "demo", "demo_pos": i, "pixel_sha256": img_sha(v)}
                          for i, v in enumerate(o.get("video_history", []))] if j == 0 else []) + \
                [{"source": "obs", "cam_key": c, "cam_slot": s, "pixel_sha256": img_sha(o["images"][c])}
                 for s, c in enumerate(("agentview", "wrist"))]
            got_imgs = [{k: v for k, v in d.items() if k not in ("index", "n_tokens")} for d in imgs]
            checks.append({
                "j": j, "text_ok": got == want, "count_ok": f.get("input_token_count") == len(dctx[j]) - want_base,
                "base_ok": f.get("input_base_len") == want_base, "imgs_ok": got_imgs == want_imgs,
                "placeholders": [int(x) for x in re.findall(r"<image:(\d+)>", f.get("input_text") or "")],
                "flag": f.get("input_decoded_from_tokens"), "restored": f.get("input_context_restored"),
                "kind": f.get("kind"), "committed": f.get("committed"), "rolled_back": f.get("rolled_back"),
                "subgoal": f.get("subgoal_text"), "input_text": f.get("input_text"),
                "segments": [s["origin"] for s in f.get("input_segments") or []]})
        out["variants"].append({
            "current_obs_only": coo, "n_fires": len(on["fires"]), "n_s2": on["steps"][-1]["n_s2"],
            "on_vs_base": mismatch(base, on), "on_vs_off": mismatch(on, off), "mutant_vs_on": mismatch(on, mut),
            "off_has_audit": bool(off["fires"]), "checks": checks,
            "subgoal_eq": [s["subgoal"] for s in on["steps"]] == [s["subgoal"] for s in off["steps"]]})
    print("CHILD_RESULT " + json.dumps(out), flush=True)


# -- pytest side ------------------------------------------------------------


def _main_checkout() -> Path:
    out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                         capture_output=True, text=True, timeout=30)
    return Path(out.stdout.strip()).parent if out.returncode == 0 and out.stdout.strip() else REPO


def _pp_root() -> Path:
    tp = os.environ.get("SGEVAL_THIRD_PARTY")
    root = Path(tp) / "PonderPounce" if tp else _main_checkout() / "third_party" / "PonderPounce"
    assert (root / PARENT_REL).is_file(), f"PonderPounce sources not found (set SGEVAL_THIRD_PARTY): {root}"
    return root


def _pp_python() -> str:
    py = os.environ.get("SGEVAL_PP_PYTHON") or str(_main_checkout() / CLIENT_ENV_PY)
    assert Path(py).exists(), f"interpreter with vla-eval/torch not found (set SGEVAL_PP_PYTHON): {py}"
    return py


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_pp_root()), env.get("PYTHONPATH", "")) if p)
    env["SGEVAL_PP_WRAP"] = str(WRAP)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


_CACHE: dict = {}


def _child(mode: str) -> dict:
    if mode not in _CACHE:
        out = subprocess.run([_pp_python(), str(Path(__file__).resolve()), "--child", mode], cwd=REPO,
                             env=_child_env(), capture_output=True, text=True, timeout=240)
        lines = [ln for ln in out.stdout.splitlines() if ln.startswith("CHILD_RESULT ")]
        assert out.returncode == 0 and lines, f"subprocess failed rc={out.returncode}\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
        _CACHE[mode] = json.loads(lines[-1][len("CHILD_RESULT "):])
    return _CACHE[mode]


def _expected_subgoals(sch: dict, n_steps: int) -> tuple[list, list]:
    """Independently derive, from the parent timing formula, the subgoal each step should return (without reading
    the code under test). Returns ``(per-step expectations, S1 fire step numbers)``.

    Ponder's j-th fire happens at ``dt + j*P2`` and becomes visible after ``+ delay``; Pounce fires every ``P1``
    starting from the first control step after the first visible one. On a fire it takes the newest subgoal that is
    "visible and non-empty" (otherwise keeps the previous one, None at episode start); between fires it returns the
    same value, and None before starting."""
    dt, p2, p1, delay = DT_MS, sch["s2_ms"], sch["s1_ms"], sch["delay_ms"]
    subs = sch["subgoals"]
    fire_t = lambda j: dt + j * p2  # noqa: E731
    expected, fires = [], []
    s1_next = None
    current = None
    for k in range(1, n_steps + 1):
        now = k * dt
        if s1_next is None and fire_t(0) + delay <= now:
            s1_next = now
        if s1_next is not None and now >= s1_next:
            vis = [j for j in range(0, (now - dt) // p2 + 1) if fire_t(j) + delay <= now and subs[j % len(subs)]]
            if vis:
                current = subs[max(vis) % len(subs)]
            fires.append(k)
            s1_next += p1
        expected.append(current if fires else None)
    return expected, fires


def test_parent_class_is_pinned():
    src = (_pp_root() / PARENT_REL).read_bytes()
    out = subprocess.run(["git", "ls-tree", "HEAD", "third_party/PonderPounce"], cwd=REPO, capture_output=True,
                         text=True, timeout=30)
    assert PP_GITLINK in out.stdout, out.stdout
    assert hashlib.sha256(src).hexdigest() == PARENT_SHA256
    text = src.decode()
    for frag in ("def _fire_s1(self, ep: Episode, obs: Observation, now: int) -> None:",
                 "def _dispense(self, ep: Episode, obs: Observation) -> Action:",
                 "def _visible_cognition(ep: Episode, now: int) -> Cognition | None:",
                 "ep.chunk = Chunk(actions=chunk, raw_state=raw_state)", "await ctx.send_action(self._dispense(ep, obs))"):
        assert frag in text, frag


def test_wrapper_actions_rng_cursor_counts_identical_to_parent():
    res = _child("eq")
    assert Path(res["parent_file"]).resolve() == (_pp_root() / PARENT_REL).resolve()
    assert "_s1" in res["init_attrs"] and "_adapter" in res["init_attrs"]
    total = 0
    for rec in res["schedules"]:
        assert rec["missing_attrs"] == [], rec["missing_attrs"]
        assert rec["base_has_subgoal_key"] is False
        assert rec["mismatch"] == 0, rec["name"]
        assert rec["n_s1"][-1] >= 5 and rec["n_s2"][-1] >= 9  # not an empty run: both timings fired several times
        total += len(rec["subgoals"])
    print(f"PP_SERVER_ACTION_EQ=PASS schedules={len(res['schedules'])} steps={total} mismatch=0 "
          f"parent_sha256={PARENT_SHA256[:12]}")


def test_wrapper_subgoal_semantics_match_schedule():
    res = _child("eq")
    by_name = {r["name"]: r for r in res["schedules"]}
    for sch in SCHEDULES:
        rec = by_name[sch["name"]]
        got = rec["subgoals"]
        want, fires = _expected_subgoals(sch, N_STEPS)
        assert got == want, (sch["name"], [(i + 1, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w][:5])
        # S1 fire steps agree with the parent counter (the expectation derivation itself is not off)
        n_s1 = rec["n_s1"]
        assert [i + 1 for i in range(N_STEPS) if n_s1[i] != (n_s1[i - 1] if i else 0)] == fires
        # None before the first subgoal: both before starting (hold) and on fires where "the first visible cognition has
        # an empty subgoal" return None
        first_fire = fires[0]
        assert all(g is None for g in got[:first_fire])
        assert got[first_fire - 1] is None and rec["s1_started"][first_fire - 1] is True
        # after the chunk is exhausted (cursor == CHUNK) the subgoal is unchanged: equal to the value at that chunk's fire step
        exhausted = [i for i, c in enumerate(rec["cursor"]) if c == CHUNK]
        assert exhausted
        for i in exhausted:
            k = max(f for f in fires if f <= i + 1)
            assert got[i] == got[k - 1]
        # old subgoal before ready_at_ns: when Ponder has produced a new subgoal that is not yet visible, the earlier one is
        # still returned
        subs = sch["subgoals"]
        stale = 0
        for i, g in enumerate(got):
            now = (i + 1) * DT_MS
            produced = [j for j in range(0, (now - DT_MS) // sch["s2_ms"] + 1) if subs[j % len(subs)]]
            if produced and g is not None and subs[max(produced) % len(subs)] != g:
                stale += 1
        assert stale > 0, sch["name"]
    # the second timing specifically covers "at fire time the newest visible cognition has an empty subgoal, so the
    # earlier non-empty subgoal is kept"
    sch_b = SCHEDULES[1]
    _, fires_b = _expected_subgoals(sch_b, N_STEPS)
    newest_visible_empty = 0
    for k in fires_b:
        now = k * DT_MS
        vis = [j for j in range(0, (now - DT_MS) // sch_b["s2_ms"] + 1) if DT_MS + j * sch_b["s2_ms"] + sch_b["delay_ms"] <= now]
        if vis and not sch_b["subgoals"][max(vis) % len(sch_b["subgoals"])] and by_name[sch_b["name"]]["subgoals"][k - 1]:
            newest_visible_empty += 1
    assert newest_visible_empty > 0


def test_comparator_detects_rng_and_cursor_mutants():
    res = _child("mutant")
    for rec in res["schedules"]:
        assert rec["mismatch"] == 0
        assert rec["mutant_rng"] > 0 and rec["mutant_cursor"] > 0, rec["name"]


def test_entrypoint_accepts_same_args_as_original_server():
    """Launched by absolute path with cwd in the third-party dir: jsonargparse recognizes the same ``--args.*`` as the
    original server (only --help is checked, no weights loaded)."""
    root = _pp_root()
    out = subprocess.run([_pp_python(), str(WRAP), "--help"], cwd=root, env=_child_env(), capture_output=True,
                         text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-3000:]
    for flag in ("--args.checkpoint_path", "--args.seed", "--args.s1_period_ms", "--port"):
        assert flag in out.stdout, flag
    assert "SubgoalReportingServer" in out.stdout


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--child":
    _child_main(sys.argv[2])


def test_audit_switch_and_generation_blocks_observe_only():
    """Stage 3 (interface freeze notes, section 5): with SGEVAL_AUDIT=0 there is no audit key and everything else
    matches the enabled run; when enabled every reply carries the audit key, and the System 2 generation blocks in the
    audit map one to one onto the stub's real fires (the j-th fire's subgoal = script entry j % L, indices contiguous,
    total equal to the parent's Ponder count) -- the wrapper neither infers more nor records less."""
    res = _child("eq")
    for sch in SCHEDULES:
        rec = next(r for r in res["schedules"] if r["name"] == sch["name"])
        assert rec["audit_off_has_key"] is False and rec["audit_on_all_keyed"] is True
        assert rec["audit_on_off_mismatch"] == 0, sch["name"]
        n_s2 = rec["n_s2"][-1]
        subs = sch["subgoals"]
        assert rec["audit_fire_index"] == list(range(n_s2))
        assert rec["audit_fire_subgoals"] == [subs[j % len(subs)] for j in range(n_s2)]
        assert rec["audit_kinds"] == ["stub"]
        assert rec["audit_gen_keys_ok"] is True and rec["audit_text_reconstructed"] is True
    print(f"PP_AUDIT_OBS_EQ=PASS schedules={len(SCHEDULES)} audit_on_off_mismatch=0")


@pytest.mark.slow  # keeps the core short tests fast (main session 2026-10-07): runs under -m ""
def test_s2_input_decoded_from_real_context_tokens_observe_only():
    """FIX-3 (plan 8.11, PonderPounce row: "full S2 context including fed-back history, incremental image/text
    segments, attached image refs"): real upstream ``SoftS2SessionContext`` + fake System 2. Each fire's
    ``input_text`` equals what the test decodes independently from the context tokens recorded by the fake model:
    "after the previous fire's observation segment -> this observation segment" (append mode; for current_obs_only
    it is this observation segment after the prefix, flagged ``input_context_restored``). The first fire includes the
    task prefix and 3 demo images, later ones the fed-back previous subgoal and the cognition placeholder; image
    sources, cameras and pixel hashes match one by one. OBS_EQ: between wrapper audit on/off and the original class,
    actions, RNG, timing, fake-model forward and lm_head counts and final context tokens are identical; a bad wrapper
    calling lm_head once more must be caught."""
    res = _child("s2input")
    fires = nonempty = mism = 0
    for v in res["variants"]:
        assert v["off_has_audit"] is False and v["subgoal_eq"] is True
        assert v["on_vs_base"] == 0 and v["on_vs_off"] == 0, v["current_obs_only"]
        assert v["mutant_vs_on"] > 0
        assert v["n_fires"] == v["n_s2"] >= 20
        for c in v["checks"]:
            assert c["text_ok"] and c["count_ok"] and c["base_ok"] and c["imgs_ok"], c
            assert c["flag"] is True and c["placeholders"] == list(range(len(c["placeholders"])))
        first = v["checks"][0]
        assert first["input_text"].startswith("<|im_start|>user\npick up the cube<image:0><image:1><image:2>")
        assert first["segments"] == ["prefix", "obs"] and first["restored"] is False
        if v["current_obs_only"]:
            assert all(c["restored"] is True and c["segments"] == ["obs"] for c in v["checks"][1:])
        else:
            kinds = {(c["kind"], c["committed"], c["rolled_back"]) for c in v["checks"]}
            assert ("transition", True, False) in kinds and ("nontransition", False, False) in kinds
            assert ("nontransition", False, True) in kinds  # generation hit the cap without terminating -> rollback
            for prev, cur in zip(v["checks"], v["checks"][1:]):
                assert cur["restored"] is False
                if prev["committed"]:  # feedback: the generation block committed by the previous fire (incl. raw subgoal) appears in this input
                    assert cur["segments"][0] == "generated" and prev["subgoal"] in cur["input_text"]
                    assert cur["input_text"].startswith("<|fim_prefix|>")
                else:  # not committed (incl. rollback): only the cognition placeholder + this observation; rolled-back tokens are
                    # not in the input
                    assert cur["segments"] == ["cog", "obs"] and "xxx" not in cur["input_text"]
                assert cur["input_text"].endswith("<image:0><image:1>")
        fires += len(v["checks"])
        nonempty += sum(1 for c in v["checks"] if c["input_text"])
        mism += v["on_vs_base"] + v["on_vs_off"]
    print(f"PP_S2_INPUT=PASS fires={fires} input_text_nonempty={nonempty} obs_eq_mismatch={mism}")
