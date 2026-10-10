"""旧仓封存模块的最小保留部分：只留生成链路回写与 MoveCube 同方式递补要用的两个纯函数。

原 ``_freeze.py``（抽签结果封存成 ``hard-specs/4``、候选数默认表、逐方式配额等）随规格抽签退役，不搬进评估仓；
这里两个函数从旧仓 ``fd0017d6`` 的生成链路封存模块 ``_freeze.py`` 逐字摘出，语义不变：

* :func:`_movecube_way`：规格里录像局用的 MoveCube 运动方式（最后一次 ``_initialize_episode`` 的 ``way_idx``）；
* :func:`write_jsonl_exclusive`：排他落盘（目标已存在即失败，不用会静默覆盖的 ``os.replace``）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import _common  # noqa: F401  路径设置

hard_specs = _common.hard_specs_light()
SpecsError = hard_specs.SpecsError


def _movecube_way(spec: dict[str, Any]) -> int | None:
    """录像局用的是最后一次 _initialize_episode 的 way_idx（构造期 reset 是 initializations.0，正式 reset 是最大序号）。"""
    inits = spec.get("initializations") if isinstance(spec, dict) else None
    if not isinstance(inits, dict) or not inits:
        return None
    last = max(inits, key=lambda k: int(k))
    way = inits[last].get("way_idx") if isinstance(inits[last], dict) else None
    return int(way) if way is not None else None


def write_jsonl_exclusive(path: Path, records: list[dict[str, Any]]) -> None:
    """已存在拒绝覆盖；同目录临时文件 + ``os.link``（目标已存在即原子失败）。"""
    path = Path(path)
    if path.exists():
        raise SpecsError(f"{path} 已存在，禁止覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".hardspecs-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(hard_specs.canonical_json(record) + "\n")
        os.chmod(name, 0o644)
        os.link(name, path)
    finally:
        os.unlink(name)
