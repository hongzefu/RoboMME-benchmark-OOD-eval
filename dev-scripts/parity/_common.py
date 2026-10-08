"""injection-dev 各模块共用的路径设置（目录名带连字符，不能作包导入；入口按路径直跑，本模块负责 sys.path）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for extra in (HERE, REPO_ROOT / "scripts" / "parity", REPO_ROOT / "scripts", REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
