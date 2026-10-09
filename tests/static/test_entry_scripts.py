"""L0：评估仓入口与目录布局（C18 入口白名单、生产代码不依赖 tests、dev-scripts 不走 ``scripts.`` 包式导入）。

拆仓后（1008 拆分方案第二部分 §二）评估仓的入口约定：

- ``scripts/`` 只有 ``evaluate.py`` 一个文件（本机入口；benchmark 的五个入口在子模块 ``third_party/robomme_benchmark``
  里，由 benchmark 仓自己的同名测试守护）；
- ``dev-scripts/`` 顶层只有 ``gl``、``orig``、``checks``、``media``、``parity``、``site`` 六个子目录，没有文件；
- 生产代码（``src/``、``scripts/``、``dev-scripts/``）不 import ``tests``（AST 收集 import 语句）；
- ``dev-scripts/`` 下的脚本不按包导入 ``scripts``（``from scripts.…``／``import scripts.…``）：``scripts/`` 不是包，
  脚本之间一律按路径加载或经评估包导入。

拆仓前本文件另有 ``evaluation_hard.py`` 与上游 ``evaluation.py`` 的差异形态、``run_example.EPISODE_LIMITS`` 与官方元数据
一致两组用例，它们守护的是 benchmark 仓内容，已随 benchmark 仓的同名测试迁走，本仓不再测。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._support.loaders import REPO

SCRIPTS = REPO / "scripts"
DEV_SCRIPTS = REPO / "dev-scripts"
ENTRY_SET = {"evaluate.py"}
DEV_SUBDIRS = {"gl", "orig", "checks", "media", "parity", "site", "release"}
PRODUCTION_DIRS = (REPO / "src", REPO / "scripts", REPO / "dev-scripts")


def _listing(d: Path) -> tuple[set[str], set[str]]:
    """目录顶层的 (文件名集合, 子目录名集合)；忽略 ``__pycache__``。"""
    files, dirs = set(), set()
    for p in d.iterdir():
        if p.name == "__pycache__":
            continue
        (dirs if p.is_dir() else files).add(p.name)
    return files, dirs


# ---------------------------------------------------------------- 入口清单与目录布局


def test_scripts_top_level_has_only_evaluate():
    files, dirs = _listing(SCRIPTS)
    assert files == ENTRY_SET and dirs == set(), (files, dirs)


def test_dev_scripts_has_only_six_subdirs_and_no_top_level_files():
    files, dirs = _listing(DEV_SCRIPTS)
    assert files == set(), files
    assert dirs == DEV_SUBDIRS, dirs


# ---------------------------------------------------------------- 生产代码不 import tests、不把 scripts 当包


def _imports_of(source: str, pkg: str) -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name == pkg or a.name.startswith(pkg + ".")]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == pkg or node.module.startswith(pkg + "."):
                hits.append(node.module)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            arg = node.args[0].value
            if name in ("import_module", "__import__") and isinstance(arg, str) \
                    and (arg == pkg or arg.startswith(pkg + ".")):
                hits.append(arg)
    return hits


def imports_of_tests(source: str) -> list[str]:
    """源码中指向 ``tests`` 包的导入（import／from-import／importlib.import_module／__import__ 字面值）。"""
    return _imports_of(source, "tests")


def imports_of_scripts_pkg(source: str) -> list[str]:
    """源码中把 ``scripts`` 当包导入的语句（``from scripts.x import y``、``import scripts.x``、``from scripts import x``、
    ``import_module('scripts.x')``）。"""
    return _imports_of(source, "scripts")


def _py_under(*dirs: Path) -> list[Path]:
    out = []
    for d in dirs:
        out += [p for p in d.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


def test_production_code_does_not_import_tests():
    files = _py_under(*PRODUCTION_DIRS)
    assert len(files) > 50, len(files)
    bad = {str(p.relative_to(REPO)): h for p in files if (h := imports_of_tests(p.read_text(encoding="utf-8")))}
    assert bad == {}


def test_dev_scripts_do_not_import_scripts_as_package():
    files = _py_under(DEV_SCRIPTS)
    assert files
    bad = {str(p.relative_to(REPO)): h for p in files
           if (h := imports_of_scripts_pkg(p.read_text(encoding="utf-8")))}
    assert bad == {}


@pytest.mark.parametrize("src", [
    "import tests\n",
    "import tests._support.loaders as L\n",
    "from tests._support import loaders\n",
    "from tests import conftest\n",
    "import importlib\nimportlib.import_module('tests._support.loaders')\n",
    "__import__('tests')\n",
])
def test_imports_of_tests_catches(src):
    assert imports_of_tests(src)


@pytest.mark.parametrize("src", [
    "import testscenario\n",
    "from .tests import x\n",
    "from robomme import tests_helper\n",
    "s = 'import tests'\n",
])
def test_imports_of_tests_ignores_non_tests(src):
    assert imports_of_tests(src) == []


@pytest.mark.parametrize("src", [
    "from scripts.parity import hard_parity\n",
    "import scripts.evaluate\n",
    "from scripts import evaluate\n",
    "import importlib\nimportlib.import_module('scripts.parity._common')\n",
])
def test_imports_of_scripts_pkg_catches(src):
    assert imports_of_scripts_pkg(src)


@pytest.mark.parametrize("src", [
    "import scriptsy\n",
    "from .scripts import x\n",
    "p = REPO / 'scripts' / 'evaluate.py'\n",
    "s = 'from scripts.x import y'\n",
])
def test_imports_of_scripts_pkg_ignores_paths_and_strings(src):
    assert imports_of_scripts_pkg(src) == []
