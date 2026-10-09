"""Load script modules by file path (scripts/ is not a package; production code also loads modules by path). Legacy-repo paths are mapped through LEGACY_PATHS."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
_CACHE: dict[str, object] = {}

# Repo-split mapping (1008 split plan, part two, section 2): keys are paths relative to the legacy repo's scripts/,
# values are new paths relative to this repo's root. Only entries that landed in the eval package src/ are listed;
# any other rel resolves against this repo's scripts/.
LEGACY_PATHS: dict[str, str] = {
    "eval-official/astra_cost_guard.py": "src/robomme_ood_eval/servers/astra_cost_guard.py",
    "eval-official/astra_hard_runner.py": "src/robomme_ood_eval/models/astra.py",
    "eval-official/framesamp_modul_client.py": "src/robomme_ood_eval/models/framesamp_modul.py",
    "eval-official/groundsg_client.py": "src/robomme_ood_eval/models/groundsg.py",
    "eval-official/official_defs.py": "src/robomme_ood_eval/models/_official_defs.py",
    "eval-official/policy_server_wrap.py": "src/robomme_ood_eval/servers/policy_server_wrap.py",
    "eval-official/pp_client.py": "src/robomme_ood_eval/models/pp.py",
    "eval-official/pp_server_wrap.py": "src/robomme_ood_eval/servers/pp_server_wrap.py",
    "eval-official/recorder.py": "src/robomme_ood_eval/record/recorder.py",
    "eval-official/smvla_client.py": "src/robomme_ood_eval/models/smvla.py",
    "eval-official/smvla_server.py": "src/robomme_ood_eval/servers/smvla_server.py",
    "eval-official/trace_writer.py": "src/robomme_ood_eval/record/trace_writer.py",
}


def script_path(rel: str) -> Path:
    """``rel`` is a path relative to the legacy repo's scripts/ (e.g. ``eval-official/recorder.py``) or to this repo's scripts/."""
    if rel in LEGACY_PATHS:
        p = REPO / LEGACY_PATHS[rel]
    else:
        p = SCRIPTS / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_script(rel: str, *, fresh: bool = False):
    """Load a script module. The same path reuses one module object by default; ``fresh=True`` executes an independent copy.

    The module name is derived from the relative path (``_script_parity_noise_gate``), and the script's directory is
    temporarily prepended to sys.path so that ``import <sibling module>`` inside the script keeps working.
    """
    return load_path(rel, script_path(rel), fresh=fresh)


def load_path(rel: str, path: Path, *, fresh: bool = False):
    """Load from an already resolved ``path`` (``rel`` only derives the module name); shares the module cache with ``load_script``."""
    key = str(path)
    if not fresh and key in _CACHE:
        return _CACHE[key]
    name = "_script_" + rel.removesuffix(".py").replace("/", "_").replace("-", "_")
    if fresh:
        name += f"_{len(_CACHE)}_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    d = str(path.parent)
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    try:
        spec.loader.exec_module(mod)
    finally:
        if added:
            try:
                sys.path.remove(d)
            except ValueError:
                pass
    if not fresh:
        _CACHE[key] = mod
    return mod
