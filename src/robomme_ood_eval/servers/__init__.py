"""Shared helpers for the four regular models (FrameSamp+Modulation, GroundSG, SimpleMemVLA, PonderPounce) that
start a policy server from the client side.

``policy_server_wrap.py``, ``smvla_server.py`` and ``pp_server_wrap.py`` in this package are run **by file path** with
each model's own venv interpreter (not through this ``__init__``); this module is only used by the Policy subclasses
in the client process. Its contents come from the legacy ``run_seat.sh`` launcher:

* paths and defaults: evaluation repo root ``repo_root``, per-server interpreter ``interpreter``, checkpoint check
  ``require_ckpt`` (no default checkpoint; ``--ckpt`` is mandatory), default port bases (``DEFAULT_PORT_BASE``, as in
  the legacy ``run_policy``: ``18000 + 10 x policy index``);
* ``CleanServerProcess``: on top of ``ServerProcess``, first drops variables that affect determinism / compile caches
  (the legacy ``CLEAN_ENV``), then adds what the model needs;
* pre-launch gates (Python versions of the legacy shell functions of the same names; failures raise
  ``PreflightError`` whose text starts with ``RUN_BLOCKED reason=``): ``tokenizer_gate``, ``preflight_mme_vla``,
  ``preflight_pp``, ``variant_pairing``;
* post-launch checks: ``check_server_log`` (the MME-VLA server log must contain ``history_config='<yaml>'``) and
  ``check_wrap_metadata`` (``policy_seed`` in the metadata written by the server wrapper must equal this run's seed);
* ``ckpt_fingerprint``: per-file sha256 combined into one fingerprint (legacy ``ckpt_fingerprint``; runs in a
  background thread and writes a single line of text);
* ``ServedPolicy``: common base class of the four Policy subclasses (pick a port, start or attach the server, build
  the legacy ``conn_info``, normalize ``PlayOutput``); ``SessionNoClose``: session proxy handed to the legacy client
  loops that blocks a model-side ``session.close()`` (only the outer loop closes the environment).

Metadata written by the server wrappers always goes to ``<server_dir>/server-wrap-metadata-<port>.json``, separate
from ``ServerProcess``'s own ``server-metadata-<port>.json`` (used for attach and ``--stop-server``), so neither
overwrites the other.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from robomme_ood_eval.policy import Policy, ServerMismatch, ServerProcess, pick_port

#: this directory (where the three server wrapper scripts live)
SERVERS_DIR = Path(__file__).resolve().parent
#: evaluation repo root (this file lives in src/robomme_ood_eval/servers/)
REPO = Path(__file__).resolve().parents[3]
#: variables always removed before starting a server (CLEAN_ENV of the legacy run_seat.sh)
CLEAN_ENV = ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
             "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "CUBLAS_WORKSPACE_CONFIG")
#: commit the mme-vla submodule must be pinned to (legacy MME_VLA_COMMIT)
MME_VLA_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
#: tokenizer path relative to OPENPI_DATA_HOME (legacy TOKENIZER_REL)
TOKENIZER_REL = "big_vision/paligemma_tokenizer.model"
#: history_config that must appear in each model's server log (legacy *_YAML_EXPECT)
YAML_EXPECT = {"perceptual-framesamp-modul": "perceptual-framesamp-modul.yaml",
               "groundsg": "symbolic-grounded-subgoal.yaml"}
#: default port bases (legacy run_policy: 18000 + 10 x policy index; smvla=0, perceptual-framesamp-modul=1,
#: groundsg=2, pp=3)
DEFAULT_PORT_BASE = {"smvla": 18000, "perceptual-framesamp-modul": 18010, "groundsg": 18020, "pp": 18030}
#: legacy default: server readiness timeout (seconds)
DEFAULT_READY_TIMEOUT_S = 1200.0
#: default XLA memory fraction for the MME-VLA server (legacy SEAT_XLA_MEM_FRACTION)
DEFAULT_XLA_MEM_FRACTION = "0.75"
#: the three GroundSG variants (same as models/_official_defs.VARIANTS; not imported here to avoid pulling in the
#: official extraction logic)
GROUNDSG_VARIANTS = ("ground-sg-oracle", "ground-sg-qwenvl", "ground-sg-memer")


def require_ckpt(cfg: dict, model: str) -> Path:
    """Return the checkpoint path from ``cfg["ckpt"]``; there is no default, so a missing value raises ``ValueError``."""
    ckpt = cfg.get("ckpt")
    if not ckpt:
        raise ValueError(f"--ckpt is required for model {model}")
    return Path(ckpt)


class PreflightError(RuntimeError):
    """A pre-launch gate failed (text starts with ``RUN_BLOCKED reason=``); raised from ``load()``, it stops the
    whole batch."""


# -- paths and parameters ----------------------------------------------------------


def repo_root(cfg: dict) -> Path:
    """Evaluation repo root: ``cfg["repo_root"]`` -> env var ``ROBOMME_EVAL_ROOT`` -> the repo containing this
    package."""
    v = cfg.get("repo_root") or os.environ.get("ROBOMME_EVAL_ROOT")
    return Path(v).resolve() if v else REPO


def interpreter(cfg: dict, key: str, env_name: str, default: Path) -> Path:
    """Server interpreter: ``cfg[key]`` -> env var ``env_name`` -> default path (same lookup as the legacy
    run_seat.sh)."""
    v = cfg.get(key) or os.environ.get(env_name)
    return Path(v) if v else Path(default)


def server_dir(cfg: dict) -> Path:
    """Directory for server metadata and server logs: ``cfg["server_dir"]`` -> ``<work_dir>/servers`` -> a per-user
    directory under the system temp dir."""
    if cfg.get("server_dir"):
        return Path(cfg["server_dir"])
    if cfg.get("work_dir"):
        return Path(cfg["work_dir"]) / "servers"
    return Path(tempfile.gettempdir()) / f"robomme-eval-servers-{os.getuid()}"


def gpu_of(cfg: dict) -> str | None:
    """GPU for the server: the first of ``cfg["gpus"]`` (``--gpus 0,1``) -> ``cfg["gpu"]`` -> None (inherit the
    caller's ``CUDA_VISIBLE_DEVICES``, e.g. as set by srun on a cluster)."""
    gpus = cfg.get("gpus")
    if isinstance(gpus, (list, tuple)) and gpus:
        return str(gpus[0])
    if isinstance(gpus, (int, str)) and str(gpus).strip():
        return str(gpus).split(",")[0].strip()
    g = cfg.get("gpu")
    return None if g is None else str(g)


def choose_port(cfg: dict, model: str, metadata_dir: Path) -> int:
    """Server port: an explicit ``cfg["port"]`` is used as is; otherwise start from ``cfg["port_base"]`` (default
    ``DEFAULT_PORT_BASE``). If a metadata file already exists for that port (a server left behind when a previous
    watchdog exited), keep it and let ``ServerProcess.start`` attach / refuse; otherwise use ``pick_port`` to find a
    port where both it and +1 are free."""
    if cfg.get("port") is not None:
        return int(cfg["port"])
    base = int(cfg.get("port_base") or DEFAULT_PORT_BASE[model])
    if (Path(metadata_dir) / f"server-metadata-{base}.json").is_file():
        return base
    return pick_port(base)


def flag_on(v: Any) -> bool:
    """Switch argument (``on`` / ``1`` / ``true`` / ``yes`` or a truthy value)."""
    if isinstance(v, str):
        return v.strip().lower() in ("on", "1", "true", "yes")
    return bool(v)


def gpu_slug(gpu: str | None) -> str:
    """GPU model name as a directory name (legacy gpu_slug); ``unknown`` when ``nvidia-smi`` is unavailable."""
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"] + (["-i", str(gpu)] if gpu is not None else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (out[0].strip().replace(" ", "_").replace("/", "_") if out else "") or "unknown"


# -- server process ------------------------------------------------------------------


class CleanServerProcess(ServerProcess):
    """``ServerProcess`` whose child environment first drops ``CLEAN_ENV`` (legacy ``env -u ...``) and then adds
    the model's own variables."""

    unset_vars = CLEAN_ENV

    def _child_env(self) -> dict:
        env = super()._child_env()
        for k in self.unset_vars:
            if k not in self.env:
                env.pop(k, None)
        return env


def check_server_log(srv: Any, expect_yaml: str, *, grace_s: float = 30.0, poll_s: float = 0.5) -> str:
    """Once the MME-VLA server is ready, check its log contains ``history_config='<expect_yaml>'`` (the legacy
    start_server SERVER_CONFIG check). The config line is printed while loading weights, before the port starts
    listening, so only ``grace_s`` seconds of slack are allowed; if missing, stop the server and raise
    ``PreflightError``."""
    needle = f"history_config='{expect_yaml}'"
    deadline = time.monotonic() + float(grace_s)
    while True:
        try:
            text = Path(srv.log_path).read_text(errors="replace")
        except OSError:
            text = ""
        if needle in text:
            line = f"SERVER_CONFIG=PASS name={srv.name} history_config={expect_yaml} seed={srv.policy_seed}"
            print(line, flush=True)
            return line
        if time.monotonic() >= deadline:
            srv.stop()
            raise PreflightError(f"RUN_BLOCKED reason=server_config (server log lacks {needle}) log={srv.log_path}")
        time.sleep(poll_s)


def check_wrap_metadata(path: Path, policy_seed: int) -> dict | None:
    """``policy_seed`` in the metadata written by the server wrapper (``--sgeval-metadata-out`` or
    ``--metadata_out``) must equal this run's seed. A missing file (attached to an existing server, or wrapper
    disabled) returns None; a mismatch raises ``ServerMismatch``."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ServerMismatch(f"cannot read server wrapper metadata: {p}: {e}") from e
    if doc.get("policy_seed") != int(policy_seed):
        print(f"RUN_BLOCKED reason=server_mismatch detail=wrap_metadata policy_seed={doc.get('policy_seed')!r} "
              f"want={int(policy_seed)} metadata={p}", flush=True)
        raise ServerMismatch(f"server wrapper metadata policy_seed={doc.get('policy_seed')!r} does not match this run's "
                             f"{int(policy_seed)}: {p}")
    return doc


# -- pre-launch gates ----------------------------------------------------------------


def tokenizer_gate(openpi_home: Any, expected_sha: Any) -> str:
    """Before starting the MME-VLA server, verify the sha256 of the tokenizer under ``OPENPI_DATA_HOME`` (never
    downloads a substitute on the spot)."""
    if not openpi_home or not expected_sha:
        raise PreflightError("RUN_BLOCKED reason=tokenizer_sha detail=missing_args (openpi_data_home and "
                             "tokenizer_sha256 are required)")
    f = Path(openpi_home) / TOKENIZER_REL
    if not f.is_file():
        raise PreflightError(f"RUN_BLOCKED reason=tokenizer_sha detail=file_missing file={f}")
    actual = file_sha256(f)
    if actual.lower() != str(expected_sha).lower():
        raise PreflightError(f"RUN_BLOCKED reason=tokenizer_sha expected={str(expected_sha).lower()} actual={actual} "
                             f"file={f}")
    line = f"TOKENIZER_SHA=PASS sha256={actual} file={f} openpi_data_home={openpi_home}"
    print(line, flush=True)
    return line


def _git(sub: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(sub), *args], capture_output=True, text=True, timeout=60)


def preflight_mme_vla(root: Path, model: str, ckpt: Path, py: Path, *, commit: str = MME_VLA_COMMIT) -> str:
    """Legacy ``preflight_mme_vla``: submodule commit pinned and clean, nested submodule empty, ``history_config.txt``
    next to the ckpt equals the expected yaml, the yaml exists, the ckpt has ``params`` / ``assets``, the interpreter
    is executable, and the server wrapper exists."""
    sub = Path(root) / "third_party" / "mme-vla"
    expect = YAML_EXPECT[model]
    head = _git(sub, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise PreflightError(f"RUN_BLOCKED reason=mme_vla_commit head={head or 'none'} expect={commit}")
    if _git(sub, "diff", "--quiet", "HEAD", "--", "src", "scripts", "packages", "pyproject.toml",
            "uv.lock").returncode != 0:
        raise PreflightError("RUN_BLOCKED reason=mme_vla_dirty")
    nested = sub / "third_party" / "robomme_benchmark"
    if nested.is_dir() and any(nested.iterdir()):
        raise PreflightError("RUN_BLOCKED reason=nested_submodule_not_empty")
    ckpt = Path(ckpt)
    try:
        hc = (ckpt / ".." / "history_config.txt").read_text(encoding="utf-8").strip()
    except OSError:
        hc = ""
    if hc != expect:
        raise PreflightError(f"RUN_BLOCKED reason=history_config value='{hc}' expect='{expect}' policy={model}")
    if not (sub / "src" / "mme_vla_suite" / "models" / "config" / "robomme" / hc).is_file():
        raise PreflightError(f"RUN_BLOCKED reason=yaml_missing {hc}")
    if not ((ckpt / "params").is_dir() and (ckpt / "assets").is_dir()):
        raise PreflightError(f"RUN_BLOCKED reason=ckpt_layout {ckpt}")
    if not (Path(py).is_file() and os.access(py, os.X_OK)):
        raise PreflightError(f"RUN_BLOCKED reason=mme_vla_venv_missing {py}")
    wrap = SERVERS_DIR / "policy_server_wrap.py"
    if not wrap.is_file():
        raise PreflightError(f"RUN_BLOCKED reason=policy_server_wrap_missing path={wrap}")
    line = f"MME_VLA_PREFLIGHT=PASS policy={model} commit={head} history_config={hc} ckpt={ckpt} py={py}"
    print(line, flush=True)
    return line


def preflight_pp(root: Path, ckpt: Any, py: Path, *, server_wrap: bool, seed: int) -> str:
    """Legacy ``preflight_pp``: submodule directory present, interpreter executable, ckpt directory present with
    ``norm_stats.json``, and the wrapper present when it is enabled."""
    sub = Path(root) / "third_party" / "PonderPounce"
    if not sub.is_dir():
        raise PreflightError(f"RUN_BLOCKED reason=pp_submodule_missing dir={sub}")
    if not (Path(py).is_file() and os.access(py, os.X_OK)):
        raise PreflightError(f"RUN_BLOCKED reason=pp_venv_missing {py}")
    if not ckpt or not Path(ckpt).is_dir():
        raise PreflightError(f"RUN_BLOCKED reason=pp_ckpt_missing ckpt={ckpt or 'unset'}")
    if not (Path(ckpt) / "norm_stats.json").is_file():
        raise PreflightError(f"RUN_BLOCKED reason=pp_ckpt_layout ckpt={ckpt} (missing norm_stats.json)")
    wrap = SERVERS_DIR / "pp_server_wrap.py"
    if server_wrap and not wrap.is_file():
        raise PreflightError(f"RUN_BLOCKED reason=pp_server_wrap_missing path={wrap}")
    line = (f"PP_PREFLIGHT=PASS ckpt={ckpt} py={py} seed={seed} hf_home={os.environ.get('HF_HOME', 'unset')} "
            f"server_wrap={int(bool(server_wrap))}")
    print(line, flush=True)
    return line


def variant_pairing(variant: Any, qwenvl_adapter: Any, memer_adapter: Any) -> str:
    """Legacy ``variant_pairing`` (groundsg only): pairing of the variant with the QwenVL / MemER adapters."""
    why = None
    if variant == "ground-sg-oracle":
        if qwenvl_adapter:
            why = "adapter_without_qwenvl"
        elif memer_adapter:
            why = "memer_adapter_without_memer"
    elif variant == "ground-sg-qwenvl":
        if not qwenvl_adapter:
            why = "qwenvl_needs_adapter"
        elif memer_adapter:
            why = "memer_adapter_without_memer"
        elif not Path(qwenvl_adapter).is_dir():
            why = f"adapter_missing path={qwenvl_adapter}"
    elif variant == "ground-sg-memer":
        if not memer_adapter:
            why = "memer_needs_adapter"
        elif qwenvl_adapter:
            why = "adapter_without_qwenvl"
        elif not Path(memer_adapter).is_dir():
            why = f"memer_adapter_missing path={memer_adapter}"
    elif not variant:
        why = "groundsg_needs_variant"
    else:
        why = f"unknown_variant variant={variant}"
    if why:
        raise PreflightError(f"RUN_BLOCKED reason=variant_pairing variant={variant or 'none'} detail={why}")
    line = (f"VARIANT_PAIRING=PASS variant={variant} adapter={qwenvl_adapter or 'none'} "
            f"memer_adapter={memer_adapter or 'none'}")
    print(line, flush=True)
    return line


# -- ckpt fingerprint ----------------------------------------------------------------


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_fingerprint(root: Any) -> str:
    """Legacy ``ckpt_fingerprint``: combine per-file "relative path \\t bytes \\t sha256" lines into one sha256 and
    return a one-line verdict text."""
    t0 = time.time()
    root = Path(root).resolve()
    h = hashlib.sha256()
    n = b = 0
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        size = p.stat().st_size
        h.update(f"{p.relative_to(root)}\t{size}\t{file_sha256(p)}\n".encode())
        n += 1
        b += size
    return f"CKPT_FINGERPRINT dir={root} sha256={h.hexdigest()} files={n} bytes={b} secs={time.time() - t0:.1f}"


def start_ckpt_fingerprint(root: Any, out: Path) -> threading.Thread | None:
    """Compute the ckpt fingerprint in a background thread and write it to ``out`` (like the legacy run_seat.sh,
    this does not block server start; failures only write the reason)."""
    if not root or not Path(root).is_dir():
        return None

    def work():
        try:
            line = ckpt_fingerprint(root)
        except Exception as e:  # noqa: BLE001 a fingerprint failure must not affect evaluation
            line = f"CKPT_FINGERPRINT=ERROR dir={root} {type(e).__name__}: {e}"
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(line + "\n", encoding="utf-8")
        except OSError:
            pass

    t = threading.Thread(target=work, name="ckpt-fingerprint", daemon=True)
    t.start()
    return t


# -- common base class of the four regular model Policies ---------------------------


class ServedPolicy(Policy):
    """Policy base class with one local server (shared by the four regular models; Astra does not use it).

    In ``load()`` a subclass builds its command and calls ``_launch`` (start or attach the server); in ``play()`` it
    uses ``_conn_info(spec)`` to build the legacy ``conn_info`` (server address, step cap, identity, trace location),
    passes it to the model module's existing ``run_episode``, and normalizes the result into ``PlayOutput`` via
    ``_finish_play``. Stopping the server is handled uniformly by the base ``Policy.close()``.

    Common cfg: ``repo_root``, ``server_dir`` / ``work_dir``, ``port`` / ``port_base``, ``gpus`` / ``gpu``,
    ``ready_timeout_s`` (default 1200), ``preflight`` (default on; off in unit tests), ``ckpt_fingerprint`` (default
    on, background thread)."""

    model = "served"
    host = "127.0.0.1"

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        if self.policy_seed < 0:
            raise PreflightError(f"RUN_BLOCKED reason=policy_seed detail=policy_seed must be a non-negative integer "
                                 f"(got {policy_seed!r})")
        self.root = repo_root(self.cfg)
        self.metadata_dir = server_dir(self.cfg)
        self.port: int | None = None
        self.wrap_meta: Path | None = None
        self.load_info: dict[str, Any] = {}
        self._load_reported = False

    # server startup
    @property
    def preflight(self) -> bool:
        return flag_on(self.cfg.get("preflight", True))

    def _pick_port(self) -> int:
        self.port = choose_port(self.cfg, self.model, self.metadata_dir)
        self.wrap_meta = self.metadata_dir / f"server-wrap-metadata-{self.port}.json"
        return self.port

    def _launch(self, argv: list, env: dict, cwd: Path, ready, ckpt: Any) -> ServerProcess:
        """Create a ``CleanServerProcess`` and ``start()`` it (metadata for this port and a live process -> attach /
        refuse; otherwise start fresh and wait until ready)."""
        srv = CleanServerProcess(argv, env, cwd, gpu_of(self.cfg), ready, port=int(self.port),
                                 metadata_dir=self.metadata_dir, policy_seed=self.policy_seed,
                                 ckpt=None if ckpt is None else str(ckpt), name=str(self.label),
                                 ready_timeout_s=float(self.cfg.get("ready_timeout_s", DEFAULT_READY_TIMEOUT_S)))
        if self.wrap_meta is not None and not srv.metadata_path.is_file():
            self.wrap_meta.unlink(missing_ok=True)  # fresh start: delete the previous wrapper metadata so a stale seed is never read
        self.server = srv
        t0 = time.monotonic()
        srv.start()
        self.load_info.update(server_ready_s=round(time.monotonic() - t0, 3), server_attached=bool(srv.attached),
                              port=int(self.port))
        if self.wrap_meta is not None:
            check_wrap_metadata(self.wrap_meta, self.policy_seed)
        return srv

    def _fingerprint(self, ckpt: Any) -> None:
        if flag_on(self.cfg.get("ckpt_fingerprint", True)):
            out = self.metadata_dir / f"ckpt-fingerprint-{self.label}.txt"
            start_ckpt_fingerprint(ckpt, out)
            self.load_info["ckpt_fingerprint_file"] = str(out)

    # per episode
    def _conn_info(self, spec) -> dict:
        """The ``conn_info`` the legacy ``env_client`` passed to the model's ``run_episode``, now built from the
        ``EpisodeSpec`` and instance attributes.

        The trace always goes to ``trace.jsonl`` in this episode's raw directory (``trace_path``); ``episode_tag`` is
        ``<key>.a<attempt>``."""
        return {"host": self.host, "port": int(self.port), "max_steps": int(spec.max_steps), "policy": self.model,
                "dataset": spec.dataset, "strict_cap": bool(spec.strict_cap), "policy_seed": int(self.policy_seed),
                "effective_cap": int(spec.max_steps) if spec.strict_cap else None,
                "trace_path": str(Path(spec.out_dir) / "trace.jsonl"), "trace_dir": None,
                "episode_tag": f"{spec.key}.a{int(spec.attempt)}", "attempt": int(spec.attempt),
                "rec_dir": str(spec.out_dir)}

    def _finish_play(self, res: dict) -> dict:
        """Model ``run_episode`` result -> ``PlayOutput``: ``task_success`` becomes 0/1 and required fields are
        filled in; the first episode in the process also carries ``policy_load`` in ``timing`` (server readiness and
        warm-up time)."""
        out = dict(res)
        status = out.get("status", "error")
        out["status"] = status
        out["task_success"] = int(status == "success")
        out["steps"] = int(out.get("steps") or 0)
        out.setdefault("error", None)
        out["infra"] = bool(out.get("infra"))
        out.setdefault("infra_reason", None)
        if not self._load_reported:
            timing = dict(out.get("timing") or {})
            timing["policy_load"] = dict(self.load_info)
            out["timing"] = timing
            self._load_reported = True
        return out


class SessionNoClose:
    """Session proxy handed to the legacy client loops: ``close()`` does not close the environment (only the outer
    loop does); every other attribute and method is forwarded unchanged.

    SimpleMemVLA copies the old official ``SimEnvService.reset``, which calls ``env.close()`` after a failed reset;
    under the new interface the environment belongs to the outer loop, so that call is blocked."""

    def __init__(self, session: Any):
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "close_calls", 0)

    def close(self) -> None:
        object.__setattr__(self, "close_calls", self.close_calls + 1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


__all__ = ["SERVERS_DIR", "REPO", "CLEAN_ENV", "MME_VLA_COMMIT", "TOKENIZER_REL", "YAML_EXPECT", "require_ckpt",
           "DEFAULT_PORT_BASE", "DEFAULT_READY_TIMEOUT_S", "GROUNDSG_VARIANTS", "PreflightError", "CleanServerProcess",
           "repo_root", "interpreter", "server_dir", "gpu_of", "choose_port", "flag_on", "gpu_slug",
           "check_server_log", "check_wrap_metadata", "tokenizer_gate", "preflight_mme_vla", "preflight_pp",
           "variant_pairing", "ckpt_fingerprint", "start_ckpt_fingerprint", "file_sha256", "ServedPolicy",
           "SessionNoClose"]
