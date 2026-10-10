#!/usr/bin/env python3
"""MME-VLA action server wrapper (shared by the three GroundSG variants and FrameSamp+Modulation).

Wraps the pinned third-party ``third_party/mme-vla/scripts/serve_policy.py`` (**the third-party source is not
modified**):

1. Arguments: ``--sgeval-metadata-out <path>`` and ``--sgeval-serve-root <dir>`` are stripped from argv; everything else
   goes unchanged to the third-party ``Args`` (tyro). ``--seed=<policy_seed>`` is the third-party ``Args.seed``
   (passed explicitly by the server command builder, no longer hard-coded to 7).
2. Server metadata: after the third-party ``create_policy`` returns (weights loaded), write
   ``server-metadata-<port>.json``: ``policy_seed``, ``argv``, ``pid``, ``port``, ``audit`` and the sha256 of the
   third-party ``serve_policy.py``, so the client can look up the result row's ``server_seed``.
3. Reply audit key ``_sgeval_audit`` (environment variable ``SGEVAL_AUDIT``, on by default): the return value of
   the third-party ``create_policy`` is wrapped in a proxy whose ``infer`` calls the real policy as usual and returns
   its result unchanged, only adding one key to the result dict::

       {"channels": [{"channel": "task"|"symbolic", "text", "token_ids", "mask", "tokenizer", "truncated"}, ...],
        "server_final_text": <raw symbolic-channel text, else the task-channel text>, "pp_generation": null,
        "server_timing": {"infer_ms": <whole blocking infer duration>, "gpu": <first device name or null>}}

   Channels come from the real calls to the third-party ``PaligemmaTokenizer.tokenize`` during this ``infer``: the
   call without a subgoal is ``task`` (``tokenized_prompt``), the one with a subgoal is ``symbolic``
   (``symbolic_tokenized_prompt``). ``text`` is the full string the third party actually passed to sentencepiece
   ``encode`` (``Task: ...;\\nCurrent Subgoal: ...;\\nAction: `` etc., several ``encode`` calls concatenated in
   order), ``token_ids`` / ``mask`` are the real return values of ``tokenize`` (including padding), and
   ``truncated`` is whether the raw ``encode`` length exceeded ``max_len``. How it observes: only while an infer is
   in progress, ``tokenize`` temporarily swaps the instance's sentencepiece processor for a proxy that forwards
   ``encode`` unchanged and records arguments and return values, and swaps it back right after; no extra
   tokenization, no extra inference, no random number generator is touched, and actions are unchanged. With
   ``SGEVAL_AUDIT=0`` there is neither proxy nor patch, and replies are byte-identical to the third party's
   (``OBS_EQ`` comparison).
4. The import environment matches ``python scripts/serve_policy.py``: cwd is ``third_party/mme-vla`` (the launcher
   starts this wrapper by absolute path), ``sys.path[0]`` becomes ``<mme-vla>/scripts`` and this directory is
   removed; logging matches the third-party ``__main__`` (``basicConfig(INFO, force=True)``), so the
   ``history_config='...'`` log line checked by the launcher still appears. If the third-party script is not found
    under cwd, fall back to ``$SGEVAL_THIRD_PARTY/mme-vla``. An explicit serving root takes precedence and must
    contain the serving script.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
import threading
import time
from functools import cache
from pathlib import Path
from typing import Any

#: server reply audit key
AUDIT_KEY = "_sgeval_audit"
ENV_AUDIT = "SGEVAL_AUDIT"
#: the wrapper's own argument
METADATA_FLAG = "--sgeval-metadata-out"
SERVE_ROOT_FLAG = "--sgeval-serve-root"
#: path of the pinned third-party script relative to the mme-vla root
SERVE_REL = "scripts/serve_policy.py"

_HERE = str(Path(__file__).resolve().parent)
_ACTIVE = threading.local()  # channel collection of the infer in progress (non-empty only during infer)


def audit_enabled() -> bool:
    """``SGEVAL_AUDIT`` is on by default; with ``0`` the wrapper does not patch, does not proxy and adds no audit
    key."""
    return os.environ.get(ENV_AUDIT, "1") != "0"


def split_wrapper_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """Strip ``--sgeval-metadata-out[=]<path>`` from argv; returns (path, remaining arguments)."""
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


def split_serve_root(argv: list[str]) -> tuple[str | None, list[str]]:
    """Strip the optional serving root without changing the metadata parser contract."""
    root, rest, i = None, [], 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith(SERVE_ROOT_FLAG + "="):
            root = arg.split("=", 1)[1]
        elif arg == SERVE_ROOT_FLAG:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError(f"{SERVE_ROOT_FLAG} requires a directory")
            root = argv[i + 1]
            i += 1
        else:
            rest.append(arg)
        i += 1
    return root, rest


def mme_vla_root(serve_root: str | None = None) -> Path:
    """Validate an explicit serving root, otherwise use cwd then ``$SGEVAL_THIRD_PARTY/mme-vla``."""
    if serve_root is not None:
        root = Path(serve_root)
        if not root.is_dir() or not (root / SERVE_REL).is_file():
            raise FileNotFoundError(f"third-party {SERVE_REL} not found in explicit root {root}")
        return root
    cwd = Path.cwd()
    if (cwd / SERVE_REL).is_file():
        return cwd
    tp = os.environ.get("SGEVAL_THIRD_PARTY")
    if tp and (Path(tp) / "mme-vla" / SERVE_REL).is_file():
        return Path(tp) / "mme-vla"
    raise FileNotFoundError(f"third-party {SERVE_REL} not found (cwd={cwd}, SGEVAL_THIRD_PARTY={tp})")


def prepare_sys_path(root: Path) -> None:
    """Match ``python scripts/serve_policy.py``: ``sys.path[0]`` is ``<root>/scripts`` and this directory is not on
    the path."""
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != _HERE]
    scripts = str(root / "scripts")
    if scripts in sys.path:
        sys.path.remove(scripts)
    sys.path.insert(0, scripts)


def load_serve_policy(root: Path):
    """Load the pinned third-party script by file path (module name ``serve_policy``; ``__name__`` is not
    ``__main__``, so it does not start a server on its own)."""
    path = root / SERVE_REL
    spec = importlib.util.spec_from_file_location("serve_policy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serve_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ints(x: Any) -> list[int]:
    return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]


def _bools(x: Any) -> list[bool]:
    return [bool(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]


class _EncodeSpy:
    """Transparent proxy of the sentencepiece processor: ``encode`` is forwarded unchanged and (input text, returned
    ids) recorded; other attributes go straight to the original object."""

    def __init__(self, real: Any, log: list):
        self._real, self._log = real, log

    def encode(self, *a, **k):
        ids = self._real.encode(*a, **k)
        text = a[0] if a else k.get("input", k.get("text"))
        self._log.append((text, ids))
        return ids

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_tokenizer_spy(tokenizer_cls: Any) -> None:
    """Install observation on the third-party ``PaligemmaTokenizer.tokenize`` (once per class): with no infer in
    progress it passes straight through; otherwise it temporarily swaps the instance's sentencepiece processor for
    ``_EncodeSpy``, calls the original method, returns unchanged, and records one channel."""
    if getattr(tokenizer_cls, "_sgeval_spy", False):
        return
    orig = tokenizer_cls.tokenize

    def tokenize(self, *a, **k):
        sink = getattr(_ACTIVE, "sink", None)
        if sink is None:
            return orig(self, *a, **k)
        subgoal = k["subgoal"] if "subgoal" in k else (a[2] if len(a) > 2 else None)
        real = self._tokenizer
        log: list = []
        self._tokenizer = _EncodeSpy(real, log)
        try:
            tokens, mask = orig(self, *a, **k)
        finally:
            self._tokenizer = real
        raw_len = sum(len(ids) for _, ids in log)
        max_len = getattr(self, "_max_len", None)
        sink.append({"channel": "symbolic" if subgoal is not None else "task",
                     "text": "".join(str(t) for t, _ in log),
                     "token_ids": _ints(tokens), "mask": _bools(mask),
                     "tokenizer": f"paligemma_tokenizer.model max_len={max_len}",
                     "truncated": bool(max_len is not None and raw_len > int(max_len))})
        return tokens, mask

    tokenizer_cls.tokenize = tokenize
    tokenizer_cls._sgeval_spy = True


def build_audit(channels: list[dict]) -> dict:
    sym = [c for c in channels if c["channel"] == "symbolic"]
    task = [c for c in channels if c["channel"] == "task"]
    final = (sym or task or [{}])[-1].get("text")
    return {"channels": list(channels), "server_final_text": final, "pp_generation": None}


@cache
def _gpu_name() -> str | None:
    """Cache the first JAX device name; never synchronize a device for timing."""
    try:
        import jax

        return jax.devices()[0].device_kind
    except Exception:  # noqa: BLE001 optional diagnostic must not affect inference
        return None


class AuditedPolicy:
    """Proxy of the third-party policy: ``infer`` calls the real policy and returns its result unchanged, only adding
    the audit key; ``reset`` / ``add_buffer`` / ``metadata`` etc. go straight to the real policy."""

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def infer(self, obs: dict) -> dict:
        sink: list = []
        _ACTIVE.sink = sink
        try:
            t0 = time.perf_counter()
            out = self._inner.infer(obs)
            ms = (time.perf_counter() - t0) * 1000.0
        finally:
            _ACTIVE.sink = None
        out = dict(out)
        out[AUDIT_KEY] = build_audit(sink)
        out[AUDIT_KEY]["server_timing"] = {"infer_ms": ms, "gpu": _gpu_name()}
        return out


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def write_metadata(path: str, *, args: Any, argv_full: list[str], serve_file: Path | None) -> dict:
    doc = {"policy_seed": int(getattr(args, "seed")), "argv": list(argv_full), "pid": os.getpid(),
           "port": getattr(args, "port", None), "audit": audit_enabled(), "wrapper": "policy_server_wrap.py",
           "serve_policy": str(serve_file) if serve_file else None,
           "serve_policy_sha256": _sha256(serve_file) if serve_file else None}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return doc


def _default_tokenizer_cls():
    from mme_vla_suite.training.config import PaligemmaTokenizer

    return PaligemmaTokenizer


def run(sp: Any, args: Any, *, meta_path: str | None, argv_full: list[str], tokenizer_cls: Any = None,
        serve_file: Path | None = None) -> Any:
    """Start the server via the third-party ``sp.main(args)``; meanwhile ``sp.create_policy`` is replaced by
    "original function + write metadata + (with audit on) wrap in proxy" and restored afterwards. Returns what
    ``sp.main`` returns (a real server never returns)."""
    audit = audit_enabled()
    orig_create = sp.create_policy

    def create_policy(a):
        policy = orig_create(a)
        if meta_path:
            write_metadata(meta_path, args=a, argv_full=argv_full, serve_file=serve_file)
        if not audit:
            return policy
        install_tokenizer_spy(tokenizer_cls if tokenizer_cls is not None else _default_tokenizer_cls())
        return AuditedPolicy(policy)

    sp.create_policy = create_policy
    try:
        return sp.main(args)
    finally:
        sp.create_policy = orig_create


def main(argv: list[str] | None = None) -> Any:
    full = list(sys.argv if argv is None else argv)
    serve_root, rest = split_serve_root(full[1:])
    meta, rest = split_wrapper_args(rest)
    root = mme_vla_root(serve_root)
    prepare_sys_path(root)
    sp = load_serve_policy(root)
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)  # same as the third-party __main__: set up logging before parsing args
    args = tyro.cli(sp.Args, args=rest)
    return run(sp, args, meta_path=meta, argv_full=full, serve_file=root / SERVE_REL)


if __name__ == "__main__":  # pragma: no cover - the real server only starts on a GPU machine
    main()
