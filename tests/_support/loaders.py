"""按文件路径加载脚本模块（scripts/ 不是包，生产代码也按路径互相加载）；旧仓路径经 LEGACY_PATHS 映射。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
_CACHE: dict[str, object] = {}

# 拆仓映射表（1008 拆分方案第二部分 §二）：键是旧仓 scripts/ 下的相对路径，值是本仓相对仓根的新路径。
# 本表只收落在评估包 src/ 里的条目；表外的 rel 按本仓 scripts/ 解析。
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
    """rel 是旧仓 scripts/ 下的相对路径（如 ``eval-official/recorder.py``），或本仓 scripts/ 下的相对路径。"""
    if rel in LEGACY_PATHS:
        p = REPO / LEGACY_PATHS[rel]
    else:
        p = SCRIPTS / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_script(rel: str, *, fresh: bool = False):
    """加载脚本模块；同一路径默认复用同一模块对象，``fresh=True`` 时重新执行一份独立副本。

    模块名由相对路径派生（``_script_parity_noise_gate``），脚本目录临时加到 sys.path 头部，
    以便脚本里 ``import <同目录模块>`` 的写法照常工作。
    """
    return load_path(rel, script_path(rel), fresh=fresh)


def load_path(rel: str, path: Path, *, fresh: bool = False):
    """按已解析的 ``path`` 加载（``rel`` 只用来派生模块名）；与 ``load_script`` 共用同一份模块缓存。"""
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
