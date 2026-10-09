"""C17 写入方与读取方的 schema 贯通：``semantic_diff.build`` 产出的 ``semantic.json`` schema，必须是浏览器检查器
``site_browser_check.py`` 认可的 schema（检查器发现 ``/api/semantic`` 不是它要的 schema 会记 problem 并判 FAIL）。

两边都从生产对象取值，测试里不写 schema 字面值：写入方取真实 ``semantic_diff.build`` 的输出；读取方跑真实
``site_browser_check.main``。检查器顶层导入 Playwright（日常门禁环境没有装，也不该为测试装），所以在**子进程**里
给它一个最小 Playwright 替身包（只放在子进程的 PYTHONPATH，测试进程不加载、不登记 sys.modules）：
``request.get`` 按路径返回目录／逐段／语义三份 JSON，其余页面操作一律抛错——检查器把中断记进 problems、照打判定行。
读取方的结论只看它打印的 ``# /api/semantic 不是 …`` 这条 problem 是否出现。

负例：把同一份产物的 schema 改掉一个字符，同一子进程流程必须报出这条 problem——证明正例的「没报」不是检查器
根本没走到 schema 判定。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests._support.dev_loaders import REPO, load_script, script_path

SEM = load_script("injection-dev/site/semantic_diff.py")
CHECKER = script_path("injection-dev/site/site_browser_check.py")
PROBLEM_PREFIX = "# /api/semantic 不是"

#: 最小 Playwright 替身：sync_playwright() → chromium.launch → new_context → new_page；
#: page.request.get(url).json() 按 url 末段取 FAKE_PW_PAYLOADS 里的 JSON，其余属性访问一律抛错
FAKE_SYNC_API = '''
import json, os


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Request:
    def get(self, url):
        payloads = json.loads(os.environ["FAKE_PW_PAYLOADS"])
        return _Resp(payloads[url.rstrip("/").rsplit("/", 1)[-1]])


class _Page:
    request = _Request()

    def on(self, *_a, **_k):
        return None

    def __getattr__(self, name):
        raise RuntimeError(f"替身页面不支持 {name}")


class _Context:
    def new_page(self):
        return _Page()


class _Browser:
    def new_context(self, **_k):
        return _Context()

    def close(self):
        return None


class _Chromium:
    def launch(self, **_k):
        return _Browser()


class _PW:
    chromium = _Chromium()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sync_playwright():
    return _PW()
'''


def _fake_playwright(root: Path) -> Path:
    pkg = root / "fakepw" / "playwright"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sync_api.py").write_text(FAKE_SYNC_API, encoding="utf-8")
    return pkg.parent


def _run_checker(tmp_path: Path, semantic: dict) -> str:
    """子进程里跑真实 site_browser_check.main，返回其标准输出。"""
    fake = _fake_playwright(tmp_path)
    payloads = {"catalog": {"tasks": []}, "subgoals": {"schema": None, "episodes": {}, "oracle": {}},
                "semantic": semantic}
    env = dict(os.environ)
    env["FAKE_PW_PAYLOADS"] = json.dumps(payloads)
    env["PYTHONPATH"] = os.pathsep.join([str(fake), *filter(None, [env.get("PYTHONPATH")])])
    proc = subprocess.run([sys.executable, str(CHECKER), "--base", "http://127.0.0.1:1", "--shots", str(tmp_path / "shots")],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert "V8_SITE=" in proc.stdout, proc.stdout + proc.stderr  # 检查器走完并照打判定行
    # 逐段数据的 schema 判定排在语义判定之后：它被报出，证明检查器确实走过了语义 schema 那一步
    assert "# subgoals.json 缺失或为空" in proc.stdout, proc.stdout
    return proc.stdout


def test_semantic_schema_written_is_accepted_by_browser_check(tmp_path):
    produced, _ = SEM.build({"episodes": {}})
    out = _run_checker(tmp_path, produced)
    assert PROBLEM_PREFIX not in out, out
    assert "V8_SITE=FAIL" in out  # 替身页面中断 + 空目录，整场照样 FAIL；本用例只关心 schema 那一条没报


def test_semantic_schema_mismatch_is_reported_by_browser_check(tmp_path):
    produced, _ = SEM.build({"episodes": {}})
    wrong = dict(produced, schema=produced["schema"] + "x")
    out = _run_checker(tmp_path, wrong)
    assert PROBLEM_PREFIX in out, out
