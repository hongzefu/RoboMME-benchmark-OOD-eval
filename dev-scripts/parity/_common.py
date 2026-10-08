"""dev-scripts/parity 各模块共用的路径设置（``dev-scripts`` 带连字符、不是包；入口按路径直跑）。

只做两件事：

* 把本目录放进 ``sys.path``，脚本之间按同目录模块名 ``import``（如 ``import hard_parity``）；
* 给出评估仓根 ``REPO_ROOT``、benchmark 子模块根 ``bench_root()`` 与 ``hard_specs`` 的轻量加载。

**不**把子模块 ``src`` 插到 ``sys.path`` 头部：``robomme``／``robomme_hard`` 一律经 venv 的 editable 安装或调用方的
``PYTHONPATH`` 解析（对拍 smoke 的 H_old 侧靠 ``PYTHONPATH`` 指向旧代码树，这里插入会把它遮住）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: 评估仓根（dev-scripts/parity → dev-scripts → 仓根）
REPO_ROOT = HERE.parents[1]
#: benchmark 子模块根（钉 40 位 sha 的 submodule；src/ 下是 robomme 与 robomme_hard）
SUBMODULE_ROOT = REPO_ROOT / "third_party" / "robomme_benchmark"
#: 官方编排代码（vendor，四文件 + SOURCE.json，字节不动）
OFFICIAL_ROOT = HERE / "official"
#: 对拍配置（拆仓前的对拍配置五份）
CONFIGS = HERE / "configs"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def bench_root() -> Path:
    """benchmark 源码树根（其下 ``src/robomme``、``src/robomme_hard``）。

    子模块已检出时就是 ``third_party/robomme_benchmark``；子模块目录为空（如 git worktree 里）时退回当前解释器
    能找到的 ``robomme_hard`` 所在源码树（只查找、不导入，不触发仿真依赖）。两处都没有即报错。"""
    if (SUBMODULE_ROOT / "src" / "robomme_hard" / "__init__.py").is_file():
        return SUBMODULE_ROOT
    spec = importlib.util.find_spec("robomme_hard")
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parents[2]
    raise FileNotFoundError(f"找不到 benchmark 源码树：子模块 {SUBMODULE_ROOT} 未检出，当前解释器也找不到 robomme_hard")


def hard_specs_file() -> Path:
    """``robomme_hard/env_record_wrapper/hard_specs.py`` 的文件路径（轻量按文件加载用）。"""
    return bench_root() / "src" / "robomme_hard" / "env_record_wrapper" / "hard_specs.py"


def sibling(name: str):
    """按模块名导入本目录的另一个脚本模块（函数体内的延迟导入用：调用时本目录可能已不在 ``sys.path``，例如被
    测试加载器按路径加载后路径已还原），导入前确保本目录在 ``sys.path`` 里。"""
    import importlib

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return importlib.import_module(name)


_HARD_SPECS_LIGHT = None


def hard_specs_light():
    """``hard_specs`` 是纯函数模块，但经 ``robomme_hard.env_record_wrapper`` 包导入会连带导入仿真。已导入过包时直接复用
    包内模块；否则按文件路径单独加载一份（不经包 ``__init__``，不导入 mani_skill／sapien）。"""
    global _HARD_SPECS_LIGHT
    loaded = sys.modules.get("robomme_hard.env_record_wrapper.hard_specs")
    if loaded is not None:
        return loaded
    if _HARD_SPECS_LIGHT is None:
        spec = importlib.util.spec_from_file_location("_hard_specs_light", hard_specs_file())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HARD_SPECS_LIGHT = module
    return _HARD_SPECS_LIGHT
