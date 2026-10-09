#!/usr/bin/env python3
"""PonderPounce server wrapper: attaches the model's own subgoal to every ACTION reply.

The upstream ``ponderpounce.eval.robomme_server.PonderPounceRoboMMEServer`` (submodule pinned at ``723df357``) only
writes the subgoal text produced by System 2 to the server log and does not send it back in the protocol. This
wrapper subclasses the original class (the original file is untouched) and overrides only two methods:

- ``_fire_s1``: before calling the parent, take from this episode's ``ep.cognitions`` the latest entry that is
  "visible by ``now`` (``ready_at_ns <= now``) with a non-empty subgoal" and update this episode's "latest visible
  non-empty subgoal" (keep the old value when there is no new one; ``None`` at the start of the episode); after the
  parent produces ``ep.chunk``, attach this value to the chunk. With the default cadence (Ponder / Pounce 1000 ms
  each) exactly one cognition becomes visible between two Pounce triggers, equivalent to looking only at
  ``_visible_cognition(ep, now)``; with unequal cadences no subgoal that became visible between two triggers is
  missed.
- ``_dispense``: the parent's result is kept unchanged, with ``"subgoal": <subgoal on the current chunk>`` added;
  ``None`` when there is no chunk (arm hold). After the chunk is exhausted the parent repeats the last action and the
  subgoal stays the same.

The wrapper never reads or writes random number generators and never changes ``cursor`` / trigger counts / actions;
it only adds one private attribute (``_sgeval_subgoal``) to each ``Episode`` and ``Chunk`` instance. Equivalence is
pinned by ``tests/pipeline/evalx/pp/test_pp_server_wrap.py`` (``PP_SERVER_ACTION_EQ``).

Audit additions:

- Every ACTION reply additionally carries ``_sgeval_audit``: ``channels`` is the prompt System 1 actually used in
  this episode (raw ``Task: ...;\nAction: `` text, token ids and mask computed by the parent, tokenizer name,
  whether filled / truncated; ``text_reconstructed=true``: the text was rebuilt from the same task_text),
  ``server_final_text`` equals that text, and ``pp_generation`` is a list with one block per System 2 ``fire``
  since the previous reply (reasoning, raw subgoal text and tokens, kind, gate score, whether committed, whether
  rolled back, context length increase; key names match the client's language ledger, see ``generation_blocks``;
  None when there was no fire). System 2 observation hooks the ``fire`` / ``_restore`` of this episode's context
  instance (calling the original methods and returning unchanged); the parent only creates the context at the first
  Ponder, so during the parent's ``_fire_s2`` the wrapper temporarily replaces the module's context class with a
  factory that "attaches observation right after construction", and restores it immediately afterwards. The wrapper
  only observes real results: no extra inference, no extra random draws, actions unchanged. With the environment
  variable ``SGEVAL_AUDIT=0`` nothing is observed and no audit key is added (``OBS_EQ`` comparison).
- S2 input: a read-only hook on the context instance's ``_append`` mirrors the token ids entering the context; each
  ``fire`` decodes "the segments added since the previous fire input + this observation segment" with the S2
  tokenizer into ``input_text`` (vision segments become ``<image:k>``), with image descriptions in
  ``input_images``; see ``_S2Watch`` and ``generation_blocks``.
- The wrapper's own argument ``--sgeval-metadata-out <path>``: stripped from argv at startup (the rest goes
  unchanged to vla_eval); writes server metadata ``server-metadata-<port>.json`` (``policy_seed`` is
  ``--args.seed``, ``argv``, ``pid``, ``port``).

Startup (arguments exactly as for the original server, cwd in the third-party PonderPounce directory, script by
absolute path)::

    python /abs/path/src/robomme_ood_eval/servers/pp_server_wrap.py --args.seed 0 --args.checkpoint_path ... --port 8000

``python -m`` puts cwd first on ``sys.path`` whereas running a script by path puts the script directory there; to
match the original command's import environment, the entry point first removes this directory from ``sys.path``,
puts cwd first, and only then imports ``ponderpounce``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

#: key added to replies (read by the client's ``pp_client.TracedConnection.act``)
SUBGOAL_KEY = "subgoal"
#: private attribute name attached to Episode / Chunk instances
_ATTR = "_sgeval_subgoal"


def _prepare_sys_path() -> None:
    """Match the ``sys.path`` of ``python -m ponderpounce.eval.robomme_server``: cwd first, this directory not on the
    path."""
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != here]
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


if __name__ == "__main__":  # pragma: no cover - only adjusted when started as a script
    _prepare_sys_path()

from ponderpounce.eval.robomme_server import PonderPounceRoboMMEServer  # noqa: E402

#: server reply audit key; not added when SGEVAL_AUDIT=0
AUDIT_KEY = "_sgeval_audit"
ENV_AUDIT = "SGEVAL_AUDIT"
#: the wrapper's own argument (stripped from argv at startup; the rest goes unchanged to vla_eval's run_server)
METADATA_FLAG = "--sgeval-metadata-out"
#: private attribute on Episode instances: System 2 generation blocks accumulated before this reply (cleared after
#: each reply)
_GEN_ATTR = "_sgeval_generations"


def audit_enabled() -> bool:
    """``SGEVAL_AUDIT`` is on by default; with ``0`` the wrapper observes nothing and adds no audit key (``OBS_EQ``
    comparison)."""
    return os.environ.get(ENV_AUDIT, "1") != "0"


def latest_visible_subgoal(cognitions: Any, now: int, previous: str | None) -> str | None:
    """Text of the latest entry in ``cognitions`` with ``ready_at_ns <= now`` and a non-empty ``subgoal``; otherwise
    returns ``previous``."""
    for cog in reversed(list(cognitions)):
        if cog.ready_at_ns <= now and cog.subgoal:
            return str(cog.subgoal)
    return previous


def _tolist(x: Any) -> list | None:
    if x is None:
        return None
    try:
        return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]
    except Exception:  # noqa: BLE001 observation failures do not affect the server
        return None


def _pixel_sha256(img: Any) -> str | None:
    """Pixel hash of one frame fed to S2 (PIL or array), same algorithm as ``trace_writer.image_sha256``
    (``dtype.str|shape|`` + contiguous bytes); an RGB PIL image converted back to ``uint8 (H, W, 3)`` is
    byte-identical to the raw frame the client sent, which the client uses to check frame numbers. Returns None on
    failure."""
    try:
        import numpy as np

        arr = np.ascontiguousarray(np.asarray(img))
        h = hashlib.sha256()
        h.update(f"{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001 observation failures do not affect the server
        return None


class _S2Watch:
    """Read-only observation of the System 2 context (``SoftS2SessionContext``): wraps the instance's ``fire`` and
    ``_restore``, calls them unchanged and returns unchanged, recording this real generation result (reasoning,
    subgoal, tokens, gate score), the context length increase and the number of rollbacks.

    S2 input (full S2 context including fed-back history, incremental image/text segments and image references): the
    upstream ``ponderpounce/inference/append_context.py::SoftS2SessionContext`` keeps only the KV cache
    (``self.cache``) and the accumulated cognition mask / query ids, **not the context token ids**; every token that
    enters the context goes through the instance method ``_append(segment)`` (``_Segment.input_ids`` /
    ``mm_token_type_ids`` are CPU tensors), and the task prefix + demo images (``self._prefix``) are already appended
    in ``reset()`` during construction. So input observation hooks the instance's ``_append``: the original method
    is called as usual and returns unchanged, and on success this segment's token ids are recorded read-only into a
    mirror (truncated at the append start ``_context_len``, which reflects ``_restore`` rollbacks / ``reset``); the
    prefix appended during construction is added to the mirror read-only from ``ctx._prefix``. The input of each
    ``fire`` = all mirror segments from "after the previous fire input" up to "this observation segment" (the first
    one includes the task prefix and demo images; later ones include the generated tokens committed by the previous
    fire, i.e. the fed-back previous subgoal, cognition placeholder tokens, and this observation image; rolled-back
    generated segments are excluded), decoded to text with S2's own tokenizer (``ctx.compiler.tokenizer``, i.e.
    ``session.s2_processor.tokenizer``) ``decode(skip_special_tokens=False)``, with vision token segments
    (``<|vision_start|>`` + ``<|image_pad|>`` x n + ``<|vision_end|>``) replaced by ``<image:k>`` placeholders and no
    pixels decoded. Read-only: never calls the model, draws no random numbers, never changes the context, no extra
    forward passes; when ``ctx`` has no ``_append`` / ``compiler.tokenizer`` (test stubs) input is not observed and
    generation blocks are byte-identical to before."""

    def __init__(self, ctx: Any, sink: list, cam_keys: Any = None):
        self.ctx, self.sink = ctx, sink
        self.restores = 0
        self.fire_index = 0
        self.cam_keys = tuple(cam_keys) if cam_keys else None
        self.input_errors = 0
        self._segs: list[dict] = []      # context mirror: each segment {start, end, ids, mm, origin, images}
        self._consumed = 0               # context length already reported as some fire's input
        self._await_obs = False          # this fire's observation segment has not been appended yet
        self._fire_pils: Any = None
        self._restored = False
        self._pending: dict | None = None
        orig_fire, orig_restore = ctx.fire, getattr(ctx, "_restore", None)
        compiler = getattr(ctx, "compiler", None)
        self._tok = getattr(compiler, "tokenizer", None)
        self._vs = getattr(compiler, "vision_start_id", None)
        self._ve = getattr(compiler, "vision_end_id", None)
        orig_append = getattr(ctx, "_append", None)
        self.observe_input = callable(orig_append) and callable(getattr(self._tok, "decode", None))
        if self.observe_input:
            try:
                self._seed_prefix()
            except Exception:  # noqa: BLE001 observation failures do not affect the server
                self.input_errors += 1

        def fire(*a, **k):
            before = getattr(ctx, "_context_len", None)
            prefix_restore = getattr(ctx, "_prefix_snapshot", None) is not None
            self.restores = 0
            self._await_obs, self._pending, self._restored = True, None, False
            self._fire_pils = a[0] if a else k.get("obs_pils")
            try:
                result = orig_fire(*a, **k)
            finally:
                self._await_obs, self._fire_pils = False, None
            pending, self._pending = self._pending, None
            after = getattr(ctx, "_context_len", None)
            rollbacks = max(0, self.restores - (1 if prefix_restore else 0))
            sg_tokens = getattr(result, "subgoal_tokens", None)
            self.sink.append({
                "fire_index": self.fire_index,
                "subgoal_text": getattr(result, "subgoal_text", None),
                "reasoning_text": getattr(result, "reasoning_text", None),
                "subgoal_tokens": _tolist(sg_tokens),
                "kind": getattr(result, "kind", None),
                "gate_score": None if getattr(result, "gate_score", None) is None else float(result.gate_score),
                "n_input_frames": getattr(result, "n_input_frames", None),
                "committed": sg_tokens is not None,
                "rolled_back": rollbacks > 0,
                "context_len_before": before,
                "context_len_after": after,
                "context_delta": (after - before) if isinstance(before, int) and isinstance(after, int) else None,
                **(pending or {}),
            })
            self.fire_index += 1
            return result

        ctx.fire = fire
        if callable(orig_restore):
            def restore(*a, **k):
                self.restores += 1
                return orig_restore(*a, **k)
            ctx._restore = restore
        if self.observe_input:
            def append(segment, *a, **k):
                start = getattr(ctx, "_context_len", None)
                out = orig_append(segment, *a, **k)
                try:
                    self._on_append(segment, start)
                except Exception:  # noqa: BLE001 observation failures do not affect the server
                    self.input_errors += 1
                return out
            ctx._append = append

    # -- S2 input observation (read-only: only reads the segment's CPU token ids, never touches the cache or calls the model) --
    def _demo_images(self) -> list:
        demo = getattr(getattr(self.ctx, "session", None), "demo_images", None) or []
        return [{"source": "demo", "demo_pos": i, "pixel_sha256": _pixel_sha256(im)} for i, im in enumerate(demo)]

    def _obs_images(self) -> list:
        pils = list(self._fire_pils or [])
        keys = self.cam_keys or ()
        return [{"source": "obs", "cam_key": keys[j] if j < len(keys) else None, "cam_slot": j,
                 "pixel_sha256": _pixel_sha256(im)} for j, im in enumerate(pils)]

    def _seed_prefix(self) -> None:
        """Add the task prefix + demo images (``ctx._prefix``) appended by ``reset()`` during construction to the
        mirror (read-only)."""
        prefix = getattr(self.ctx, "_prefix", None)
        clen = getattr(self.ctx, "_context_len", None)
        ids = _tolist(getattr(prefix, "input_ids", None))
        if not ids or not isinstance(clen, int) or clen < len(ids):
            return
        self._segs.append({"start": 0, "end": len(ids), "ids": ids,
                           "mm": _tolist(getattr(prefix, "mm_token_type_ids", None)), "origin": "prefix",
                           "images": self._demo_images()})

    def _on_append(self, segment: Any, start: Any) -> None:
        if not isinstance(start, int):
            return
        ids = _tolist(getattr(segment, "input_ids", None)) or []
        # mirror segments at or after the append start were discarded by _restore / reset (segments are appended whole and never straddle the start)
        self._segs = [s for s in self._segs if s["start"] < start]
        if self._consumed > start:  # the context went back before the reported position (current_obs_only returns to the prefix every time, or reset)
            self._consumed, self._restored = start, True
        if segment is getattr(self.ctx, "_prefix", None):
            origin, images = "prefix", self._demo_images()
        elif self._await_obs:
            origin, images = "obs", self._obs_images()
        else:
            cm = _tolist(getattr(segment, "cognition_mask", None)) or []
            origin, images = ("cog" if any(cm) else "generated"), []
        end = start + len(ids)
        if ids:
            self._segs.append({"start": start, "end": end, "ids": ids,
                               "mm": _tolist(getattr(segment, "mm_token_type_ids", None)), "origin": origin,
                               "images": images})
        if origin == "obs":
            self._await_obs = False
            segs = [s for s in self._segs if s["start"] >= self._consumed and s["end"] <= end]
            self._pending = self._decode_input(segs, base=self._consumed)
            self._consumed = end

    def _decode_input(self, segs: list, *, base: int) -> dict:
        """Mirror segments -> text (S2 tokenizer decode, vision segments become ``<image:k>``) + per-image description
        (source, camera, pixel hash)."""
        tok = self._tok
        parts: list[str] = []
        buf: list[int] = []
        images: list[dict] = []

        def flush() -> None:
            if buf:
                parts.append(str(tok.decode(list(buf), skip_special_tokens=False)))
                buf.clear()

        for s in segs:
            ids = s["ids"]
            mm = s["mm"] if s["mm"] is not None and len(s["mm"]) == len(ids) else [0] * len(ids)
            pending_imgs = list(s["images"])
            i = 0
            while i < len(ids):
                if mm[i] == 1:  # vision token segment of one image (<|image_pad|> x n)
                    j = i
                    while j < len(ids) and mm[j] == 1:
                        j += 1
                    if buf and self._vs is not None and buf[-1] == self._vs:
                        buf.pop()
                    flush()
                    k = len(images)
                    parts.append(f"<image:{k}>")
                    d = dict(pending_imgs.pop(0)) if pending_imgs else {"source": s["origin"]}
                    d.update(index=k, n_tokens=j - i)
                    images.append(d)
                    if j < len(ids) and self._ve is not None and ids[j] == self._ve:
                        j += 1
                    i = j
                    continue
                buf.append(int(ids[i]))
                i += 1
        flush()
        return {"input_text": "".join(parts), "input_token_count": sum(len(s["ids"]) for s in segs),
                "input_images": images, "input_decoded_from_tokens": True, "input_base_len": base,
                "input_context_restored": self._restored,
                "input_segments": [{"origin": s["origin"], "tokens": len(s["ids"])} for s in segs]}


class SubgoalReportingServer(PonderPounceRoboMMEServer):
    """Original class + ``subgoal`` (and audit key) attached to ACTION replies; actions, random numbers, cursor and
    trigger cadence are identical to the original class item by item."""

    def _fire_s2(self, ep, obs, now: int) -> None:
        if not audit_enabled():
            return super()._fire_s2(ep, obs, now)
        sink = getattr(ep, _GEN_ATTR, None)
        if sink is None:
            sink = []
            setattr(ep, _GEN_ATTR, sink)
        if ep.s2_context is not None:
            return super()._fire_s2(ep, obs, now)
        # first Ponder of this episode: the parent creates the context in _fire_s2 and fires immediately. Temporarily
        # replace the module's context class with a factory that "attaches observation right after construction"
        # (constructor arguments forwarded unchanged, returns an instance of the original class) and restore it right
        # after the parent returns; nothing extra is constructed or called.
        import ponderpounce.eval.robomme_server as rs

        orig_cls = rs.SoftS2SessionContext

        def factory(*a, **k):
            ctx = orig_cls(*a, **k)
            _S2Watch(ctx, sink, cam_keys=getattr(self, "_camera_keys", None))
            return ctx

        rs.SoftS2SessionContext = factory
        try:
            return super()._fire_s2(ep, obs, now)
        finally:
            rs.SoftS2SessionContext = orig_cls

    def _fire_s1(self, ep, obs, now: int) -> None:
        current = latest_visible_subgoal(ep.cognitions, now, getattr(ep, _ATTR, None))
        setattr(ep, _ATTR, current)
        super()._fire_s1(ep, obs, now)
        if ep.chunk is not None:
            setattr(ep.chunk, _ATTR, current)

    def _dispense(self, ep, obs):
        action = super()._dispense(ep, obs)
        subgoal = getattr(ep.chunk, _ATTR, None) if ep.chunk is not None else None
        out = dict(action)
        out[SUBGOAL_KEY] = subgoal
        if audit_enabled():
            out[AUDIT_KEY] = self._audit_block(ep)
        return out

    def _audit_block(self, ep) -> dict:
        """S1 prompt (token ids / mask actually computed by the parent on this episode's session) + the full S2
        generation blocks since the previous reply."""
        s = ep.session
        ids, mask = getattr(s, "s1_prompt_ids", None), getattr(s, "s1_prompt_mask", None)
        text = None
        if ids is not None:
            try:
                from ponderpounce.pi05.prompt import build_pi0_prompt

                text = build_pi0_prompt(getattr(s, "task_text", None) or "", None)
            except Exception:  # noqa: BLE001 observation failures do not affect the server
                text = None
        mask_l = _tolist(mask)
        tok = getattr(self, "_s1_tokenizer", None)
        channels = [] if ids is None else [{
            "channel": "task", "text": text, "token_ids": _tolist(ids), "mask": mask_l,
            "tokenizer": getattr(tok, "name_or_path", None) or (type(tok).__name__ if tok is not None else None),
            # right-padded to max_token_len with truncation=True: an all-ones mask means filled (possibly truncated)
            "truncated": None if mask_l is None else bool(mask_l) and all(mask_l),
            # text is rebuilt with build_pi0_prompt from the same task_text (only token_ids / mask are truly captured)
            "text_reconstructed": True}]
        gens = getattr(ep, _GEN_ATTR, None)
        fires = list(gens) if gens else []
        if gens:
            gens.clear()
        return {"channels": channels, "server_final_text": text,
                "pp_generation": generation_blocks(fires, task_text=getattr(s, "task_text", None),
                                                   active_subgoal=getattr(ep, "active_subgoal", None),
                                                   n_s2_fires=getattr(ep, "n_s2_fires", None),
                                                   n_s1_fires=getattr(ep, "n_s1_fires", None))}


def generation_blocks(fires: list, *, task_text: Any = None, active_subgoal: Any = None, n_s2_fires: Any = None,
                      n_s1_fires: Any = None) -> list | None:
    """``pp_generation``: one block per System 2 ``fire`` since the previous reply (a list; ``None`` when there was no
    fire, in which case the client opens no call).

    Each block keeps the original fields recorded by ``_S2Watch`` and adds the keys read by the client's
    ``pp_client.TracedConnection._log_generation``: ``subgoal_raw`` (= ``subgoal_text``, the raw ``at [x, y]``
    text), ``reasoning`` (= ``reasoning_text``), ``kind`` / ``committed`` (original fields), ``params`` (fire index,
    gate score, number of input frames, context length increase, whether rolled back). ``text`` is not provided
    (upstream only keeps the split reasoning and subgoal, not the raw decoded block; the client falls back to
    ``subgoal_raw`` when ``text`` is missing); ``context`` / ``prompt`` / ``images`` are not provided. S2 input: when
    the context instance is observable each block also carries ``input_text`` (tokens added to the context before
    this fire, decoded with the S2 tokenizer, vision segments as ``<image:k>``), ``input_token_count``,
    ``input_images`` (the image for the k-th placeholder: ``source=demo`` with ``demo_pos``, ``source=obs`` with
    ``cam_key`` / ``cam_slot``, all with ``pixel_sha256``), ``input_decoded_from_tokens=True``, ``input_base_len``
    (the context length this was appended after), ``input_context_restored`` (the context first went back before
    the reported position, e.g. ``current_obs_only`` returning to the prefix every time), ``input_segments`` (origin
    of each segment prefix / generated / cog / obs and its token count); these keys are absent when not observable.
    ``s2_task_text`` and the reply-time ``active_subgoal`` / ``n_s2_fires`` / ``n_s1_fires`` are attached to every
    block."""
    if not fires:
        return None
    out = []
    for f in fires:
        b = dict(f)
        b.update(subgoal_raw=f.get("subgoal_text"), reasoning=f.get("reasoning_text"),
                 params={k: f.get(k) for k in ("fire_index", "gate_score", "n_input_frames", "context_len_before",
                                               "context_len_after", "context_delta", "rolled_back")},
                 s2_task_text=task_text, active_subgoal=active_subgoal, n_s2_fires=n_s2_fires, n_s1_fires=n_s1_fires)
        out.append(b)
    return out


def split_wrapper_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """Strip the wrapper's own ``--sgeval-metadata-out[=]<path>`` from argv; returns (path, remaining arguments)."""
    meta, rest, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith(METADATA_FLAG + "="):
            meta = a.split("=", 1)[1]
        elif a == METADATA_FLAG and i + 1 < len(argv):
            meta = argv[i + 1]
            i += 1
        else:
            rest.append(a)
        i += 1
    return meta, rest


def _flag_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def write_metadata(path: str, argv_full: list[str], argv_rest: list[str]) -> dict:
    """Server metadata ``server-metadata-<port>.json``: policy_seed (i.e. --args.seed), argv, pid, port, audit."""
    seed = _flag_value(argv_rest, "--args.seed")
    port = _flag_value(argv_rest, "--port")
    doc = {"policy": "pp", "policy_seed": int(seed) if seed is not None and seed.lstrip("-").isdigit() else None,
           "argv": list(argv_full), "pid": os.getpid(), "port": int(port) if port and port.isdigit() else port,
           "audit": audit_enabled(), "wrapper": "pp_server_wrap.py"}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return doc


def main() -> None:
    from vla_eval.model_servers.serve import run_server

    full = list(sys.argv)
    meta, rest = split_wrapper_args(full[1:])
    if meta:
        write_metadata(meta, full, rest)
    sys.argv = [full[0], *rest]
    run_server(SubgoalReportingServer)


if __name__ == "__main__":  # pragma: no cover
    main()
