"""S7 报告类工具手写夹具的 dev 侧：在公开版 ``sgx_report_fixtures`` 之上补 gate2 对拍工具 ``g2()``（``dev-scripts/checks/``）。

公开版的全部名字经 ``from sgx_report_fixtures import *`` 原样重新导出，dev 测试 ``import sgx_report_fixtures_dev as F`` 后用法不变。
"""
from __future__ import annotations

if __package__:  # 以 tests.pipeline.evalx.report.sgx_report_fixtures_dev 导入时与包内的公开版配对
    from .sgx_report_fixtures import *  # noqa: F401,F403
else:  # 测试目录在 sys.path 上、按顶层名导入时与顶层的公开版配对
    from sgx_report_fixtures import *  # noqa: F401,F403
from tests._support.dev_loaders import load_script


def g2():
    return load_script("eval-official/gate2_compare.py")
