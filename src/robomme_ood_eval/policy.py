"""The ``Policy`` base class for the four model-side methods, the outer factory ``load_policy``, and server
subprocess management ``ServerProcess``.

Call frequency: ``load`` and ``close`` once each per process lifetime, ``reset`` and ``play`` once each per episode;
``run_episode`` orchestrates a single episode and never calls ``load()``. The base class keeps a call counter
``calls`` for each of the four methods (overridden methods are counted too; ``super()`` calls are not counted twice),
which the ``EVAL_EPISODE`` verdict line uses to check loads / resets / closes.

``ServerProcess`` is extracted from the legacy ``run_seat.sh`` functions ``build_server_cmd`` / ``start_server`` /
``stop_server`` / ``kill_group`` / ``pick_port`` / ``server_ready``: start a process group with ``setsid``, write
``server-metadata-<port>.json``, three readiness probes, a four-item check in ``attach()``, ``stop()`` that sends
TERM to the process group, waits 60 s and then KILLs, and an optional GPU-memory-release check;
``stop_by_metadata(path)`` lets ``scripts/evaluate.py --stop-server`` stop a server left behind after a watchdog exit
using only its metadata.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

#: proxy variables always removed when starting a server (cluster compute nodes may set an HTTP proxy, which
#: rejects local loopback connections with 403)
PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
#: default readiness wait (seconds)
DEFAULT_READY_TIMEOUT_S = 1800.0
#: seconds stop() waits for the process group to exit after TERM before sending KILL
STOP_GRACE_S = 60.0
METHODS = ("load", "reset", "play", "close")


class ServerDead(RuntimeError):
    """The server process is dead (found by the ``reset()`` liveness probe or while waiting for readiness). No
    automatic restart: stop and hand over to the user."""


class AstraStop(RuntimeError):
    """Astra signalled stop or the cost guard refused a request: stop the whole batch and keep the ledger."""


class ServerMismatch(RuntimeError):
    """The server reached by ``attach()`` does not match this run's configuration
    (``RUN_BLOCKED reason=server_mismatch``)."""


class ServerNotReady(RuntimeError):
    """Timed out waiting for readiness."""


# -- Policy base class ---------------------------------------------------------------


def _counted(name: str, fn):
    """Wrap with a call counter: per instance and method only the outermost call counts (a subclass's
    ``super().reset(spec)`` is not counted again)."""

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        depth = self.__dict__.setdefault("_call_depth", {})
        calls = self.__dict__.setdefault("calls", {m: 0 for m in METHODS})
        if depth.get(name, 0) == 0:
            calls[name] += 1
        depth[name] = depth.get(name, 0) + 1
        try:
            return fn(self, *a, **k)
        finally:
            depth[name] -= 1

    wrapper._policy_counted = True
    return wrapper


class Policy:
    """Common base class of the five models + dummy. Subclasses override ``load`` / ``reset`` / ``play`` / ``close``.

    ``reset(spec)`` contract (pinned by unit tests):
    **send no message to the server, never touch the environment**; only clear client state, prepare session
    identifiers from the episode identity, probe server health once (raise ``ServerDead`` if dead), and apply
    pre-episode refusals (e.g. Astra's episode registration and STOP check, raising ``AstraStop``). The server's reset / start_episode message is sent by ``play`` on a fresh
    connection for each episode.

    ``play(session, spec, recorder) -> dict`` returns a ``PlayOutput``: always ``status`` (success / fail / timeout /
    error), ``task_success`` (0/1), ``steps``, ``error``, ``infra``, ``infra_reason``; optionally ``decisions``,
    ``timing`` and model-specific fields. The environment is touched only via ``session.reset()`` /
    ``session.step()``; ``session.close()`` must not be called.

    Attributes: ``model`` (registered name), ``label`` (model directory name in the output tree, defaults to
    ``model``; GroundSG may include its variant), ``policy_seed``, ``cfg`` (model-specific arguments received by
    ``load_policy``), ``server`` (``ServerProcess`` or None), ``episode_wall_s`` (default per-episode wall clock;
    None means ``episode.DEFAULT_WALL_S``), ``calls`` (call counts of the four methods), ``episodes_run`` (episodes run
    in this process, maintained by the outer loop, used to relax the wall clock for the first episode).
    """

    model: str = "base"
    episode_wall_s: float | None = None
    #: True when the model needs ``--ckpt`` (no default checkpoint); mirrored by ``models.MODELS_REQUIRING_CKPT``
    requires_ckpt: bool = False

    def __new__(cls, *a, **k):
        obj = super().__new__(cls)
        obj.calls = {m: 0 for m in METHODS}
        obj._call_depth = {}
        obj.episodes_run = 0
        return obj

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for name in METHODS:
            fn = cls.__dict__.get(name)
            if fn is not None and not getattr(fn, "_policy_counted", False):
                setattr(cls, name, _counted(name, fn))

    def __init__(self, policy_seed: int, **cfg: Any):
        self.policy_seed = int(policy_seed)
        self.cfg = dict(cfg)
        self.server: ServerProcess | None = None
        self.closed = False
        if not hasattr(self, "label") or self.label is None:
            self.label = type(self).model

    label: str | None = None

    # the four methods (base-class defaults)
    def load(self) -> None:
        """Once per process: start the server (or attach to a running one), load the client-side model, compile
        caches, warm up. The base class does nothing."""

    def reset(self, spec) -> None:
        """First statement of every episode: the base class only probes server health once (when there is a
        server)."""
        self.check_server()

    def play(self, session, spec, recorder) -> dict:
        raise NotImplementedError(f"{type(self).__name__}.play is not implemented")

    def close(self) -> None:
        """Once per process: stop the server process group. Idempotent."""
        if self.closed:
            return
        self.closed = True
        for srv in self.servers():
            srv.stop()

    # helpers
    def servers(self) -> list["ServerProcess"]:
        """Servers held by this Policy (used to print ``SERVER_LEFT`` before a watchdog exit); override when a
        subclass has several."""
        return [self.server] if self.server is not None else []

    def check_server(self) -> None:
        for srv in self.servers():
            srv.check()

    def server_seed(self) -> int | None:
        """``server_seed`` for the result row: ``policy_seed`` from the server metadata; None without a server (never
        fabricated)."""
        for srv in self.servers():
            v = (srv.metadata or {}).get("policy_seed")
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return None

    def __enter__(self) -> "Policy":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


for _m in METHODS:
    setattr(Policy, _m, _counted(_m, Policy.__dict__[_m]))


def load_policy(model: str, policy_seed: int, **cfg: Any) -> Policy:
    """Construct the subclass from the ``models`` registry and call ``load()`` (once per process). Usable as a
    context manager: leaving the ``with`` block calls ``close()``.

    If ``load()`` raises, ``close()`` is called first (stopping any server already started) and the exception is
    re-raised unchanged."""
    from robomme_ood_eval import models

    cls = models.resolve(model)
    policy = cls(policy_seed=policy_seed, **cfg)
    policy.model = model
    if getattr(policy, "label", None) in (None, "base"):
        policy.label = model
    try:
        policy.load()
    except BaseException:
        policy.close()
        raise
    return policy


# -- ports and readiness probes ------------------------------------------------------


def port_busy(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a process is already listening on the port (same meaning as the legacy ``run_seat.sh::port_busy``:
    busy if a connection succeeds)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, int(port)))
            return True
        except OSError:
            return False


def pick_port(base: int | None = None, *, span: int = 2, tries: int = 5) -> int:
    """Pick a server port: with ``base``, start there and add ``span`` each time, requiring both the port and +1
    (relay) to be free, for at most ``tries`` attempts; with ``base`` None, ask the OS for an ephemeral free port.
    Raises RuntimeError if everything fails."""
    if base is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
    p = int(base)
    for _ in range(int(tries)):
        if not port_busy(p) and not port_busy(p + 1):
            return p
        p += int(span)
    raise RuntimeError(f"starting from {base}, all {tries} tries found no free port")


def http_health(port: int, path: str = "/health", host: str = "127.0.0.1", timeout: float = 2.0) -> bool:
    """Healthy if ``GET http://host:port/path`` returns 200 (bypassing proxies)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{host}:{int(port)}{path}", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@dataclasses.dataclass(frozen=True)
class Ready:
    """One readiness criterion: ``port`` (the port is listening), ``health`` (``GET value`` returns 200; value
    defaults to ``/health``), ``log`` (the server log contains the string value). ``ServerProcess`` requires all
    criteria to hold at once."""

    kind: str
    value: str | None = None

    def __post_init__(self):
        if self.kind not in ("port", "health", "log"):
            raise ValueError(f"Ready.kind must be port / health / log: {self.kind!r}")
        if self.kind == "log" and not self.value:
            raise ValueError("Ready('log') requires the string to match")

    @classmethod
    def port(cls) -> "Ready":
        return cls("port")

    @classmethod
    def health(cls, path: str = "/health") -> "Ready":
        return cls("health", path)

    @classmethod
    def log(cls, text: str) -> "Ready":
        return cls("log", text)


# -- process utilities ---------------------------------------------------------------


def pid_alive(pid: int | None) -> bool:
    """The process exists and is not a zombie."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return None
    return [x.decode(errors="replace") for x in raw.split(b"\0") if x]


def _group_alive(pgid: int) -> bool:
    """The process group still has a non-zombie process (zombies have exited and only await reaping, so they do
    not count as alive)."""
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    proc = Path("/proc")
    if not proc.is_dir():  # pragma: no cover non-Linux
        return True
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[2] == str(int(pgid)) and fields[0] != "Z":
            return True
    return False


def kill_group(pgid: int, *, grace_s: float = STOP_GRACE_S, child: subprocess.Popen | None = None,
               poll_s: float = 0.2) -> str:
    """TERM the process group, wait ``grace_s`` seconds, KILL if still alive; returns ``term`` / ``kill`` / ``gone``."""
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        if child is not None:
            child.poll()
        return "gone"
    deadline = time.monotonic() + float(grace_s)
    how = "term"
    while True:
        if child is not None:
            child.poll()  # reap our own child so a zombie does not keep the check true forever
        leader_alive = pid_alive(pgid) if child is None else child.returncode is None
        if not leader_alive and not _group_alive(pgid):
            break
        if time.monotonic() >= deadline:
            how = "kill"
            try:
                os.killpg(int(pgid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            if child is not None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            break
        time.sleep(poll_s)
    return how


def wait_gpu_free(pid: int, *, timeout_s: float = 30.0, poll_s: float = 1.0) -> bool:
    """Poll the ``nvidia-smi`` compute-process list until ``pid`` is gone (GPU memory released); returns True when
    nvidia-smi is unavailable."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return True
        if str(int(pid)) not in {x.strip() for x in out.splitlines()}:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


# ── ServerProcess ────────────────────────────────────────────────────────────


class ServerProcess:
    """A server subprocess (running in the model's own venv).

    ``argv``: full command (copied from the matching branch of the legacy ``run_seat.sh::build_server_cmd``);
    ``env``: variables added / overridden on top of the current environment (proxy variables are always removed);
    ``cwd``; ``gpu``: sets ``CUDA_VISIBLE_DEVICES`` (None leaves it unchanged); ``ready``: list of ``Ready`` criteria
    (ready only when all hold); ``port``: server port (used for the metadata file name and the port probe);
    ``metadata_dir``: directory for ``server-metadata-<port>.json``; ``policy_seed``, ``ckpt``: written to the
    metadata and checked one by one by ``attach()``.
    """

    def __init__(self, argv: list[str], env: dict | None = None, cwd: str | Path | None = None,
                 gpu: str | int | None = None, ready: Iterable[Ready] | Ready | None = None, *, port: int,
                 metadata_dir: str | Path, policy_seed: int | None = None, ckpt: str | None = None,
                 log_path: str | Path | None = None, name: str = "server",
                 ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S, ready_poll_s: float = 0.5):
        self.argv = [str(x) for x in argv]
        self.env = dict(env or {})
        self.cwd = None if cwd is None else str(cwd)
        self.gpu = None if gpu is None else str(gpu)
        if ready is None:
            ready = [Ready.port()]
        elif isinstance(ready, Ready):
            ready = [ready]
        self.ready = list(ready)
        self.port = int(port)
        self.metadata_dir = Path(metadata_dir)
        self.policy_seed = policy_seed
        self.ckpt = None if ckpt is None else str(ckpt)
        self.name = name
        self.log_path = Path(log_path) if log_path else self.metadata_dir / f"{name}-{self.port}.log"
        self.ready_timeout_s = float(ready_timeout_s)
        self.ready_poll_s = float(ready_poll_s)
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.metadata: dict | None = None
        self.attached = False

    @property
    def metadata_path(self) -> Path:
        return self.metadata_dir / f"server-metadata-{self.port}.json"

    # startup
    def _child_env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
        env.update(self.env)
        for k in PROXY_VARS:
            env.pop(k, None)
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        return env

    def start(self) -> "ServerProcess":
        """If metadata for this port exists and the process is alive -> ``attach()`` (refuse on mismatch);
        otherwise start fresh with setsid, write the metadata and wait until ready."""
        if self.metadata_path.is_file():
            if self.attach():
                return self
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen(self.argv, env=self._child_env(), cwd=self.cwd, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        self.pid = self.proc.pid
        self.metadata = {"pid": self.pid, "pgid": self.pid, "port": self.port, "argv": self.argv,
                         "policy_seed": self.policy_seed, "ckpt": self.ckpt, "cwd": self.cwd, "gpu": self.gpu,
                         "log": str(self.log_path), "name": self.name, "host": socket.gethostname(),
                         "started": time.time()}
        tmp = self.metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.metadata, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.metadata_path)
        print(f"SERVER_START name={self.name} port={self.port} pid={self.pid} log={self.log_path}", flush=True)
        self.wait_ready()
        return self

    def _ready_now(self) -> bool:
        for r in self.ready:
            if r.kind == "port" and not port_busy(self.port):
                return False
            if r.kind == "health" and not http_health(self.port, r.value or "/health"):
                return False
            if r.kind == "log":
                try:
                    text = self.log_path.read_text(errors="replace")
                except OSError:
                    return False
                if r.value not in text:
                    return False
        return True

    def log_tail(self, n: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def wait_ready(self, timeout_s: float | None = None) -> float:
        """Poll the readiness criteria; if the server dies meanwhile raise ``ServerDead`` (with the log tail); on
        timeout stop the server and raise ``ServerNotReady``."""
        limit = self.ready_timeout_s if timeout_s is None else float(timeout_s)
        t0 = time.monotonic()
        while True:
            if not self.alive():
                tail = self.log_tail()
                print(f"SERVER_DIED_BEFORE_READY name={self.name} port={self.port}\n{tail}", flush=True)
                raise ServerDead(f"{self.name} server exited before becoming ready port={self.port}: {tail[-800:]}")
            if self._ready_now():
                dt = time.monotonic() - t0
                print(f"SERVER_READY name={self.name} port={self.port} ready_s={dt:.1f}", flush=True)
                return dt
            if time.monotonic() - t0 > limit:
                print(f"SERVER_READY_TIMEOUT name={self.name} port={self.port} limit_s={limit:.0f}", flush=True)
                self.stop()
                raise ServerNotReady(f"{self.name} readiness wait exceeded {limit:.0f} s")
            time.sleep(self.ready_poll_s)

    # attach
    def attach(self) -> bool:
        """Read ``server-metadata-<port>.json``: if the process is gone return False (the caller starts a new one);
        if alive, check four items -- the metadata's ``policy_seed``, ``argv``, ``port`` and ``ckpt`` match this run,
        and ``/proc/<pid>/cmdline`` is exactly that argv. Any mismatch prints ``RUN_BLOCKED reason=server_mismatch``
        and raises ``ServerMismatch`` (refuse)."""
        try:
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        pid = meta.get("pid")
        if not isinstance(pid, int) or not pid_alive(pid):
            return False
        bad = []
        for key, want in (("policy_seed", self.policy_seed), ("argv", self.argv), ("port", self.port),
                          ("ckpt", self.ckpt)):
            if meta.get(key) != want:
                bad.append(f"{key} metadata={meta.get(key)!r} want={want!r}")
        cmd = proc_cmdline(pid)
        if cmd != meta.get("argv"):
            bad.append(f"cmdline pid={pid} actual={cmd!r} does not match metadata argv")
        if bad:
            detail = "; ".join(bad)
            print(f"RUN_BLOCKED reason=server_mismatch port={self.port} metadata={self.metadata_path} detail={detail}",
                  flush=True)
            raise ServerMismatch(detail)
        self.pid, self.metadata, self.attached = pid, meta, True
        print(f"SERVER_ATTACH name={self.name} port={self.port} pid={pid}", flush=True)
        if not self._ready_now():
            self.wait_ready()
        return True

    # state and stopping
    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return pid_alive(self.pid)

    def check(self) -> None:
        """Liveness probe: raise ``ServerDead`` if dead."""
        if not self.alive():
            raise ServerDead(f"{self.name} server has exited port={self.port} pid={self.pid}: {self.log_tail()[-800:]}")

    def stop(self, *, grace_s: float = STOP_GRACE_S, check_gpu: bool = True) -> str:
        """TERM the process group, wait ``grace_s`` seconds, then KILL; with ``check_gpu`` poll nvidia-smi until GPU
        memory is released; delete the metadata."""
        if self.pid is None:
            return "none"
        pgid = int((self.metadata or {}).get("pgid") or self.pid)
        how = kill_group(pgid, grace_s=grace_s, child=self.proc)
        gpu_ok = wait_gpu_free(self.pid) if check_gpu else None
        print(f"SERVER_STOPPED name={self.name} port={self.port} pid={self.pid} how={how} gpu_free={gpu_ok}",
              flush=True)
        try:
            self.metadata_path.unlink()
        except OSError:
            pass
        self.pid = None
        self.proc = None
        return how

    def left_line(self) -> str:
        """The line printed right before a hard watchdog exit."""
        return (f"SERVER_LEFT pid={self.pid} port={self.port} metadata={self.metadata_path} "
                f"stop=\"scripts/evaluate.py --stop-server {self.metadata_path}\"")

    @staticmethod
    def stop_by_metadata(path: str | Path, *, grace_s: float = STOP_GRACE_S, check_gpu: bool = True) -> str:
        """Stop a server using only its metadata file: stop it only if the process is alive and
        ``/proc/<pid>/cmdline`` matches the metadata argv (guards against pid reuse), then delete the metadata.
        Returns ``term`` / ``kill`` / ``gone`` / ``mismatch``."""
        path = Path(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        pid = meta.get("pid")
        if not isinstance(pid, int) or not pid_alive(pid):
            print(f"SERVER_STOP_BY_METADATA result=gone pid={pid} metadata={path}", flush=True)
            path.unlink(missing_ok=True)
            return "gone"
        if proc_cmdline(pid) != meta.get("argv"):
            print(f"SERVER_STOP_BY_METADATA result=mismatch pid={pid} metadata={path} (cmdline does not match "
                  f"metadata; left untouched)",
                  flush=True)
            return "mismatch"
        how = kill_group(int(meta.get("pgid") or pid), grace_s=grace_s)
        if check_gpu:
            wait_gpu_free(pid)
        path.unlink(missing_ok=True)
        print(f"SERVER_STOP_BY_METADATA result={how} pid={pid} metadata={path}", flush=True)
        return how


def import_attr(module: str, attr: str):
    """Lazily fetch an object by module path and attribute name (used by the registry)."""
    return getattr(importlib.import_module(module), attr)


__all__ = ["Policy", "load_policy", "ServerProcess", "Ready", "ServerDead", "AstraStop", "ServerMismatch",
           "ServerNotReady", "pick_port", "port_busy", "http_health", "kill_group", "pid_alive", "proc_cmdline",
           "wait_gpu_free", "METHODS"]

if sys.version_info < (3, 10):  # pragma: no cover
    raise RuntimeError("robomme_ood_eval.policy requires Python >= 3.10")
