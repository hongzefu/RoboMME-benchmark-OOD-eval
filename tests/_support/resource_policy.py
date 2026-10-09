"""Test resource guard (a pytest plugin loaded before collection via ``-p tests._support.resource_policy`` in pyproject addopts).

Two modes:
- Default mode (daily gate): refuses real simulation scene construction, GPU initialization, model weight loading
  and non-loopback networking; reset/step on CPU fakes is unaffected. Subprocesses inherit the same mode via sitecustomize.
- Simulation mode (``--allow-sim-reset``): no guard is installed, and only an explicit selection of ``tests/sim`` is allowed.

A violation raises ``ResourcePolicyError`` on the spot and is also written to an event ledger; even if a test swallows
the exception, a non-empty ledger at session end fails the whole run. A verdict line is printed at the end:
``TEST_RESOURCE=PASS|FAIL native_reset=<n> gpu_init=<n> weights=<n> network=<n> violations=<n>``.

Three suite-discipline checks are also enforced (any failure fails the run): empty collection, any xfail (this suite
registers no xfail), and any skip whose reason does not start with a "not verified" prefix (see NOT_VERIFIED_PREFIXES).
The guard is a test-execution constraint and does not claim to stop arbitrary circumvention.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

ENV_MODE = "ROBOMME_TEST_RESOURCE_POLICY"
ENV_LEDGER = "ROBOMME_TEST_RESOURCE_LEDGER"
REPO = Path(__file__).resolve().parents[2]
SITE_DIR = Path(__file__).resolve().parent / "sitecustomize_dir"
SIM_DIR = REPO / "tests" / "sim"

KINDS = ("native_reset", "gpu_init", "weights", "network")
# Accepted skip-reason prefixes: the English one, plus the legacy Chinese one (written as escapes) still used by
# tests that are not part of the public tree.
NOT_VERIFIED_PREFIXES = ("Not verified", "\u672a\u9a8c\u8bc1")


class ResourcePolicyError(RuntimeError):
    """A forbidden resource was touched under the daily gate."""


# ---------------------------------------------------------------- ledger


def _ledger_path() -> Path | None:
    p = os.environ.get(ENV_LEDGER)
    return Path(p) if p else None


def record(kind: str, detail: str) -> None:
    """Record one violation (parent and child processes append to the same jsonl)."""
    path = _ledger_path()
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "detail": detail, "pid": os.getpid()}, ensure_ascii=False) + "\n")


def violate(kind: str, detail: str):
    record(kind, detail)
    raise ResourcePolicyError(f"resource guard refused {kind}: {detail}")


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- patches


def _patch_mani_skill(mod) -> None:
    base = getattr(mod, "BaseEnv", None)
    if base is None or getattr(base, "_resource_policy_patched", False):
        return

    def _blocked_init(self, *a, **k):  # noqa: ANN001
        violate("native_reset", f"{type(self).__name__}.__init__ would build a real SAPIEN scene")

    base.__init__ = _blocked_init
    base._resource_policy_patched = True


def _patch_torch_cuda(mod) -> None:
    if getattr(mod, "_resource_policy_patched", False):
        return

    def _blocked(*a, **k):
        violate("gpu_init", "torch.cuda initialization")

    for name in ("init", "_lazy_init"):
        if hasattr(mod, name):
            setattr(mod, name, _blocked)
    mod.is_available = lambda: False
    mod._resource_policy_patched = True


def _patch_torch(mod) -> None:
    if getattr(mod, "_resource_policy_patched", False):
        return

    def _blocked_load(*a, **k):
        violate("weights", "torch.load reading weights")

    mod.load = _blocked_load
    mod._resource_policy_patched = True


def _patch_safetensors(mod) -> None:
    def _blocked(*a, **k):
        violate("weights", f"{mod.__name__} reading weights")

    for name in ("load_file", "safe_open", "load"):
        if hasattr(mod, name):
            setattr(mod, name, _blocked)


def _patch_sapien(mod) -> None:
    """sapien itself is a C extension whose constructors cannot be replaced reliably; block the scene entry point and
    swap the material and render system in the render submodule for refusing factories (a real RenderMaterial starts
    a render context on CPU and was observed to segfault; the offline-world fixture replaces and restores them within
    its own context)."""
    for name in ("Scene",):
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            def _blocked(self, *a, _n=name, **k):  # noqa: ANN001
                violate("native_reset", f"sapien.{_n} construction")

            cls.__init__ = _blocked
        except (TypeError, AttributeError):
            pass
    render = getattr(mod, "render", None)
    if render is None or getattr(render, "_resource_policy_patched", False):
        return
    for name in ("RenderMaterial", "RenderSystem"):
        if getattr(render, name, None) is None:
            continue

        def _blocked_factory(*a, _n=name, **k):
            violate("native_reset", f"sapien.render.{_n} construction")

        try:
            setattr(render, name, _blocked_factory)
        except (TypeError, AttributeError):
            pass
    try:
        render._resource_policy_patched = True
    except (TypeError, AttributeError):
        pass


PATCHES = {
    "mani_skill.envs.sapien_env": _patch_mani_skill,
    "torch.cuda": _patch_torch_cuda,
    "torch": _patch_torch,
    "safetensors.torch": _patch_safetensors,
    "safetensors": _patch_safetensors,
    "sapien": _patch_sapien,
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        PATCHES[self.name](module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = getattr(finder, "find_spec", lambda *a: None)(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchLoader(spec.loader, name)
                return spec
        return None


_orig_connect = socket.socket.connect
_orig_create_connection = socket.create_connection
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def _host_of(address) -> str | None:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return None  # unix sockets etc.


def _guarded_connect(self, address):
    host = _host_of(address)
    if host is not None and host not in LOOPBACK and not host.startswith("127."):
        violate("network", f"connect {host}")
    return _orig_connect(self, address)


def _guarded_create_connection(address, *a, **k):
    host = _host_of(address)
    if host is not None and host not in LOOPBACK and not host.startswith("127."):
        violate("network", f"create_connection {host}")
    return _orig_create_connection(address, *a, **k)


_installed = False


def install() -> None:
    """Install the guard (idempotent). Already imported modules are patched now; others are patched on import."""
    global _installed
    if _installed:
        return
    _installed = True
    for name, fn in PATCHES.items():
        if name in sys.modules:
            fn(sys.modules[name])
    sys.meta_path.insert(0, _PatchFinder())
    socket.socket.connect = _guarded_connect
    socket.create_connection = _guarded_create_connection
    # Hide all render and CUDA devices so the GPU stays unreachable even if a patch is bypassed.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------- pytest hooks


def pytest_addoption(parser):
    parser.addoption(
        "--allow-sim-reset",
        action="store_true",
        default=False,
        help="Simulation mode: no resource guard, only tests/sim may run (59 + 1 = 60 resets per run; standing authorization per plan Q8)",
    )


def _args_touch_sim(args) -> bool:
    for a in args:
        p = Path(str(a).split("::")[0])
        try:
            p = p.resolve()
        except OSError:
            continue
        if p == SIM_DIR or SIM_DIR in p.parents:
            return True
    return False


def pytest_configure(config):
    import pytest

    config._rp_skips = []
    config._rp_xfails = []
    sim = config.getoption("--allow-sim-reset")
    touches_sim = _args_touch_sim(config.args)
    if touches_sim and not sim:
        raise pytest.UsageError("tests/sim may only run with --allow-sim-reset (it performs real simulation resets)")
    if sim and not touches_sim:
        raise pytest.UsageError("--allow-sim-reset may only be used together with an explicit selection of tests/sim")
    config._rp_mode = "sim" if sim else "cpu"
    if sim:
        return
    fd, ledger = tempfile.mkstemp(prefix="resource-ledger-", suffix=".jsonl")
    os.close(fd)
    config._rp_ledger = Path(ledger)
    os.environ[ENV_MODE] = "cpu"
    os.environ[ENV_LEDGER] = ledger
    # Subprocesses inherit the guard via sitecustomize.
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(x for x in (str(SITE_DIR), old) if x)
    install()


def pytest_ignore_collect(collection_path, config):
    p = Path(str(collection_path)).resolve()
    if (p == SIM_DIR or SIM_DIR in p.parents) and config._rp_mode != "sim":
        return True
    return None


def pytest_runtest_logreport(report):
    if report.skipped and hasattr(report, "wasxfail"):
        _XFAILS.append(report.nodeid)
    elif report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reason = reason.removeprefix("Skipped: ")
        _SKIPS.append((report.nodeid, reason))
    elif report.passed and hasattr(report, "wasxfail"):
        _XFAILS.append(report.nodeid)


_SKIPS: list[tuple[str, str]] = []
_XFAILS: list[str] = []
_PROBLEMS: list[str] = []


def pytest_sessionfinish(session, exitstatus):
    cfg = session.config
    if getattr(cfg, "_rp_mode", "cpu") == "sim":
        return
    rows = read_ledger(cfg._rp_ledger)
    counts = {k: sum(1 for r in rows if r["kind"] == k) for k in KINDS}
    bad_skips = [(n, r) for n, r in _SKIPS if not r.startswith(NOT_VERIFIED_PREFIXES)]
    if rows:
        _PROBLEMS.append(f"resource guard ledger has {len(rows)} violation(s)")
    if bad_skips:
        _PROBLEMS.append(f"{len(bad_skips)} skip reason(s) do not start with 'Not verified': {bad_skips[:5]}")
    if _XFAILS:
        _PROBLEMS.append(f"unregistered xfail(s): {_XFAILS[:5]}")
    if session.testscollected == 0 and not cfg.option.collectonly:
        _PROBLEMS.append("empty collection")
    cfg._rp_counts = counts
    cfg._rp_violations = len(rows)
    if _PROBLEMS and session.exitstatus == 0:
        session.exitstatus = 1
    try:
        cfg._rp_ledger.unlink()
    except OSError:
        pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if getattr(config, "_rp_mode", "cpu") == "sim":
        terminalreporter.write_line("TEST_RESOURCE=SKIP mode=sim")
        return
    counts = getattr(config, "_rp_counts", {k: 0 for k in KINDS})
    for p in _PROBLEMS:
        terminalreporter.write_line(f"RESOURCE_POLICY_PROBLEM {p}")
    not_verified = [n for n, r in _SKIPS if r.startswith(NOT_VERIFIED_PREFIXES)]
    ok = not _PROBLEMS
    terminalreporter.write_line(
        f"TEST_RESOURCE={'PASS' if ok else 'FAIL'} "
        + " ".join(f"{k}={counts[k]}" for k in KINDS)
        + f" violations={getattr(config, '_rp_violations', 0)} not_verified={len(not_verified)}"
    )
