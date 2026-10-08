#!/usr/bin/env python3
"""对拍入口（0927 计划第一部分 §5.4、§6.1，第二部分 §1.3 / §3.2 / §3.4；评估仓 dev-scripts/parity）：generate / publish /
compare / anchor / import-delivery / binding / export-xhard0-manifest / smoke。

benchmark 包取自子模块 ``third_party/robomme_benchmark``。侧：O 官方（vendor 编排 d53f21a7 + 子模块 ``robomme``，官方
``_worker``）；P 修改前（锚点登记的缓存）；H 修改后（子模块 ``robomme_hard``，``--force-mirror`` +
``ROBOMME_ENV_PACKAGE=robomme_hard``）；H2 第二次生成。判定是「行为一致」（用户 U-1）：身份、setup、结构、任务成功全等
才 PASS；动作／状态／图像／帧数四项差异按同目录 ``configs/hard-parity-tolerances.json`` 的阈值判（U-19）；sha 相等数等
只作参考。

原三档（``--tier native``）的 144 局清单 ``subset_manifest.json`` 随原三档对拍退役，未搬入评估仓（旧仓 git 历史
``6e70c0bf`` 可取回）；xhard0 按数据集名取局：H 侧镜像 worker 的 gym.make 实参取自构建器 ``dataset="hard-verify"``。

子命令::

    # 生成（GL 节点；--stage 给 NFS 暂存目录时每局 sha256 后搬到暂存并删本地副本）
    uv run --frozen --no-sync python dev-scripts/parity/hard_parity.py generate --side O --tier xhard0 \
        --manifest dev-scripts/parity/configs/xhard0/xhard0_manifest.json --src-root third_party/robomme_benchmark \
        --workers 16 --gpu 0 --out /tmp/hs/O-xhard0 --stage <NFS>/hs-stage/O-xhard0
    # 比对（sled-vail，读 /data 上的拉回目录）
    uv run --no-sync python dev-scripts/parity/hard_parity.py compare --pair O:H --tier xhard0 \
        --manifest dev-scripts/parity/configs/xhard0/xhard0_manifest.json
    # 拆仓验收：1 任务 × {hard-verify, ood} × 2 局 × 2 侧 = 8 次生成后按容差比（PARITY_GEN_SMOKE）
    uv run --no-sync python dev-scripts/parity/hard_parity.py smoke --task VideoUnmask \
        --hard-verify-episodes 0:2 --ood-episodes 0:2 --hard-verify-sides O,H --ood-sides H_old,H_new \
        --h-old-src <旧仓 SRC 只读快照>/src --workers 2 --out <目录>

``generate`` 默认断言 GPU 型号为 A40；``--dev-smoke`` 放行本机 Ada（开发冒烟，``NATIVE_SMOKE``）。

v9 二次生成对拍（1002 方案 §2.3 ``PARITY_H_H2 tier=v9``）：``--identities`` 给 assemble 的 800 行清单（只取
``source == "v9-new"``）或任意身份 jsonl；默认目录 ``artifacts/newtask-v9/parity/{h5,compare}``，``--cells`` 缺省
``v9full``（V9_CELLS）。v8 二次生成对拍（``--tier v8``）与 v6 四档回归（``--tier xhard``）已于 1003 维护计划细则 2.3 删除。
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import _common  # noqa: E402  同目录：评估仓根、子模块根、hard_specs 轻量加载

HERE = _common.HERE
ROOT = _common.REPO_ROOT
RUNNER = HERE / "train_split_runner.py"
GENERATE_H5 = HERE / "generate_h5.py"
VENDOR = _common.OFFICIAL_ROOT
TOLERANCES = _common.CONFIGS / "hard-parity-tolerances.json"
# v8 起默认目录（1001 方案第一部分 §2.3）；compare／publish／binding 另可用 --h5-root／--compare-root 显式指定
LOCAL_H5_ROOT = ROOT / "artifacts" / "newtask-v8" / "parity" / "h5"
COMPARE_ROOT = ROOT / "artifacts" / "newtask-v8" / "parity" / "compare"
#: v9（1002 方案 §2.3「PARITY_H_H2 tier=v9」）：--tier v9 的默认目录另起 newtask-v9，不与 v8 混放
V9_LOCAL_H5_ROOT = ROOT / "artifacts" / "newtask-v9" / "parity" / "h5"
V9_COMPARE_ROOT = ROOT / "artifacts" / "newtask-v9" / "parity" / "compare"
ANCHORS = ROOT / "docs" / "validation" / "parity-anchors.json"
#: xhard0 清单 v8 不变（xhard0 即官方 hard，§2.5「不动」）
XHARD0_MANIFEST = _common.CONFIGS / "xhard0" / "xhard0_manifest.json"
BUCKET = "HongzeFu/robomme-hard-parity"
HF = ["uvx", "--from", "huggingface_hub==1.8.0", "--with", "click", "hf"]
#: H2＝v8 正式局的第二次生成（gen2），与 H（gen1）比对 PARITY_H_H2
SIDES = ("O", "P", "H", "H2")
PAIRS = ("O:P", "P:H", "O:H", "H:H2")
#: native＝原三档 144；xhard0＝官方 test 的 hard 192；v9＝v9 正式局 800（43 格，V9_CELLS）——对拍只比新生成的
#: 80 局（compare --identities）。v6 四档 ``xhard`` 与 v8 正式局 ``v8`` 已删（1003 维护计划细则 2.3）
TIERS = ("native", "xhard0", "v9")
#: 走 /4 规格根 + 交付清单（v8-delivery/1）的档：分母核对与五终态、只生成 H2
DELIVERY_TIERS = ("v9",)
TIER_NAMES = ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5")
XHARD0_PER_TASK = 12
XHARD0_EPISODES = tuple(range(3, 48, 4))
#: v9 形状＝V9_CELLS 按档求和 272／272／92／144／20
SHAPES = {"native": "16x3x3", "xhard0": "16x1x12", "v9": "cells43:272+272+92+144+20"}
#: v8 交付清单（S2-B ``_rollout`` 聚合步写）的 schema 与必须显式写出（含零值）的计数键（§2.2 第 7 条）
V8_DELIVERY_SCHEMA = "v8-delivery/1"
DELIVERY_COUNT_KEYS = ("exec_over_cap", "backfills", "infra_retries", "failed")
#: v8 二次生成对拍逐身份五个互斥终态（§2.2 第 11 条），合计必须等于 compared
TERMINAL_STATES = ("byte_equal", "noise", "h2_fail", "flipped", "structural")
# 容差自定规则（用户 U-22，第二部分 §3.4）：最大值 × 1.5，带下界与合理性上界
TOL_RULES = {
    "action_max": {"floor": 0.005, "ceiling": 0.05},
    "state_max": {"floor": 0.005, "ceiling": 0.05},
    "image_mad": {"floor": 1.0, "ceiling": 10.0},
    "frames_max": {"floor": 5, "ceiling": 200},
}
TOL_KEYS = {"action_max": "action_max_rad", "state_max": "state_max", "image_mad": "image_mad",
            "frames_max": "frames_max"}


class ParityError(RuntimeError):
    """前置、清单、上传或比对失败；一律停止、不重试。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def gpu_facts() -> dict[str, str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    names = [line.split(",") for line in out.stdout.strip().splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if names and visible and visible.isdigit() and int(visible) < len(names):
        names = [names[int(visible)]]
    name, driver = (names[0][0].strip(), names[0][1].strip()) if names else ("unknown", "unknown")
    return {"gpu_model": name, "driver": driver}


def versions() -> dict[str, str]:
    code = ("import json,importlib.metadata as m,sys;"
            "out={'python':sys.version.split()[0]};"
            "[out.__setitem__(p,m.version(p)) for p in ('mani_skill','sapien','torch') if __import__('importlib').util.find_spec(p.replace('-','_'))];"
            "import torch;out['cuda']=str(torch.version.cuda);print(json.dumps(out))")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": proc.stderr[-300:]}


# ── 身份 ───────────────────────────────────────────────────────────────


def native_rows(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text())
    rows = payload["rows"]
    if len(rows) != int(payload["rows_total"]):
        raise ParityError("清单行数与 rows_total 不符")
    return [{"task": r["task"], "tier": r["difficulty"], "episode": int(r["episode"]), "seed": int(r["seed"])} for r in rows]


def hard_specs_light():
    """``hard_specs`` 是纯函数模块，但经 ``robomme_hard.env_record_wrapper`` 包导入会连带导入仿真（R9 精神：本进程
    不导入 robomme_hard 包）。已导入过包时直接复用包内模块；否则按文件路径单独加载子模块里的那一份
    （``_common.hard_specs_light``，不经包 ``__init__``）。"""
    return _common.hard_specs_light()


# ── v8 交付清单读取适配（S2-B 产出，键名以 §2.2 第 7 条为准；不确定的键名集中在这里兜底）──────────────


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def read_delivery(path: Path) -> dict[str, Any]:
    """读 gen1 的 ``delivery.json``，归一成 ``{schema, base, rows, counts, cells, raw}``。

    假设的字段（S2-B 契约，v8 方案第二部分 §2.2 第 7 条）：

    * 顶层 ``schema``（v8 为 ``v8-delivery/1``）；
    * 全局计数 ``exec_over_cap``／``backfills``／``infra_retries``／``failed``：放在 ``counts``（或 ``totals``）字典里，
      或直接放在顶层；缺失的键在 ``counts`` 里记为 ``None``（调用方据此判「没有显式写零」）；``infra_retries``
      只由 S2-B 聚合行与 delivery.json 给出（/4 规格行里没有），守卫只核其显式存在；
    * 逐格 ``cells``：``{"<task>/<tier>": {...}}`` 或 ``[{task, tier, ...}]``，原样保留在 ``cells`` 里（键统一成
      ``"<task>/<tier>"``）；
    * 逐局 ``rows``（或 ``delivered``）：``task``、``tier``（或 ``difficulty``）、``episode``、``seed``、``candidate``、
      h5 路径 ``path``（相对 delivery.json 所在目录；或 ``h5_path``／``h5`` 绝对路径）、``h5_sha256``（或 ``sha256``）、
      ``exec_steps``、``frames``、``env_module``。
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        payload = {"rows": payload}
    rows = []
    for r in payload.get("rows") or payload.get("delivered") or []:
        rows.append({"task": r["task"], "tier": r.get("tier", r.get("difficulty")), "episode": int(r["episode"]),
                     "seed": int(r["seed"]), "candidate": r.get("candidate", r.get("episode")),
                     "path": _first(r, "path", "h5_path", "h5"), "h5_sha256": _first(r, "h5_sha256", "sha256"),
                     "exec_steps": r.get("exec_steps"), "frames": r.get("frames"), "env_module": r.get("env_module"),
                     "recovery_mode": r.get("recovery_mode")})
    source = payload.get("counts") or payload.get("totals") or payload
    counts = {key: source.get(key) for key in DELIVERY_COUNT_KEYS}
    cells: dict[str, Any] = {}
    raw_cells = payload.get("cells") or {}
    if isinstance(raw_cells, dict):
        cells = {str(k): v for k, v in raw_cells.items()}
    else:
        cells = {f"{c['task']}/{c.get('tier', c.get('difficulty'))}": c for c in raw_cells}
    return {"schema": payload.get("schema"), "base": path.parent, "rows": rows, "counts": counts, "cells": cells,
            "raw": payload}


def delivery_h5(delivery: dict[str, Any], row: dict[str, Any]) -> Path | None:
    """交付行的 h5 绝对路径：相对路径按 delivery.json 所在目录解析（v7 同口径）。"""
    if not row.get("path"):
        return None
    path = Path(row["path"])
    return path if path.is_absolute() else delivery["base"] / path


def v8_rows(delivery: Path) -> list[dict[str, Any]]:
    """v8 正式局：gen1 的 ``delivery.json``（经 :func:`read_delivery` 适配）。重复行原样保留，由 compare 计数。"""
    return [{"task": r["task"], "tier": r["tier"], "episode": r["episode"], "seed": r["seed"],
             "candidate": r["candidate"], "h5_sha256": r["h5_sha256"]} for r in read_delivery(delivery)["rows"]]


def rows_for(tier: str, manifest: Path) -> list[dict[str, Any]]:
    if tier in ("native", "xhard0"):
        return native_rows(manifest)
    if tier in DELIVERY_TIERS:
        return v8_rows(manifest)
    raise ParityError(f"未知档 {tier!r}（只支持 {TIERS}）")


#: v9 具名格表（与同目录 ``_rollout.py`` 的 ``V9_SMOKE_CELLS``／``V9_SHARD_TASKS`` 同值，测试逐项核对；
#: 本模块只用标准库、不导入 robomme_hard 包，故不 import _rollout）
V9_SMOKE_CELLS = {("MoveCube", "xhard4"): 1, ("InsertPeg", "xhard4"): 1}
V9_SHARD_TASKS = {"shard1": ("MoveCube",)}
#: ``--cells`` 具名格表；``full`` 另跟随当前 ``hard_specs.EXPECTED_CELLS``（v9 阶段 3b 切换后即 V9）。
#: V8 专用的 ``v8full``（V8 1070 局表）与 ``smoke``（V8 2b 冒烟 7 格）已于维护计划阶段 1b（W4）删除。
CELL_NAMES_V9 = ("v9full", "v9", "v9smoke", *(f"v9{name}" for name in V9_SHARD_TASKS))


def _fits(cells: dict[tuple[str, str], int], table: dict[tuple[str, str], int]) -> bool:
    return bool(cells) and all(key in table and isinstance(n, int) and not isinstance(n, bool) and 0 < n <= table[key]
                               for key, n in cells.items())


def table_version(table: dict[tuple[str, str], int], hs=None) -> str:
    """完整交付格表 → 版本名（``hard_specs.CELL_TABLES`` 的键，现只有 ``v9``）；按值比较。"""
    hs = hs or hard_specs_light()
    for name, known in hs.CELL_TABLES.items():
        if known == table:
            return name
    raise ParityError(f"未登记的交付格表（{len(table)} 格、合计 {sum(table.values())}）")


def expected_version(hs=None) -> str:
    """当前包内 ``EXPECTED_CELLS`` 的版本（v9 阶段 3b 切换前 v8、切换后 v9）。"""
    hs = hs or hard_specs_light()
    return table_version(hs.EXPECTED_CELLS, hs)


def cells_version(cells: dict[tuple[str, str], int], hs=None) -> str:
    """（子）格表 → 版本：与某张完整表逐格相等即该版本；否则能被 V9_CELLS 覆盖判 v9；都不覆盖时取当前 EXPECTED 版本。
    （原先先判 V8 1070 局表覆盖为 v8；V8 表已于维护计划阶段 1b 删除。）"""
    hs = hs or hard_specs_light()
    for name, table in hs.CELL_TABLES.items():
        if cells == table:
            return name
    for name in ("v9",):
        if _fits(cells, hs.CELL_TABLES[name]):
            return name
    return expected_version(hs)


def parse_cells_versioned(spec: str | None, hs=None) -> tuple[dict[tuple[str, str], int], str]:
    """``--cells`` → ``(格表, 版本)``。名字：

    * ``full``：当前 ``hard_specs.EXPECTED_CELLS``（v9 阶段 3b 切换后 = V9_CELLS 800）；
    * ``v9full``／``v9``：V9_CELLS（43 格 800）；``v9smoke``：MoveCube／InsertPeg 的 xhard4 各 1 局；
      ``v9shard1``：V9 MoveCube 一片（局数取 V9_CELLS）；
    * JSON（文件路径或内联文本）：``{"<task>/<tier>": n}``、``{"<task>@<tier>": n}``、``{task: {tier: n}}``、
      ``[[task, tier, n], ...]``、``[{"task", "tier", "n"|"count"}, ...]`` 或 ``{"cells": [...]}``；整张子表必须能被
      V9_CELLS 覆盖（键在表内、局数为不超过表值的正整数），版本见 :func:`cells_version`。
    V8 专用的 ``v8full``／``smoke`` 已删除。"""
    hs = hs or hard_specs_light()
    spec = spec or "full"
    if spec == "full":
        return dict(hs.EXPECTED_CELLS), expected_version(hs)
    if spec in ("v9full", "v9"):
        return dict(hs.V9_CELLS), "v9"
    if spec == "v9smoke":
        return dict(V9_SMOKE_CELLS), "v9"
    if spec.startswith("v9") and spec[2:] in V9_SHARD_TASKS:
        return {k: n for k, n in hs.V9_CELLS.items() if k[0] in V9_SHARD_TASKS[spec[2:]]}, "v9"
    text = Path(spec).read_text() if Path(spec).is_file() else spec
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ParityError(f"--cells 须为 full／{'／'.join(CELL_NAMES_V9)} 或格表 JSON：{spec!r}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("cells"), list):
        payload = payload["cells"]
    cells: dict[tuple[str, str], int] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                for tier, n in value.items():
                    cells[(key, tier)] = n
            else:
                sep = "@" if "@" in str(key) else "/"
                task, _, tier = str(key).partition(sep)
                cells[(task, tier)] = value
    else:
        for item in payload:
            if isinstance(item, dict):
                cells[(item["task"], item["tier"])] = item.get("n", item.get("count"))
            else:
                task, tier, n = item
                cells[(task, tier)] = n
    if not _fits(cells, hs.CELL_TABLES["v9"]):
        bad = {k: v for k, v in cells.items()
               if k not in hs.V9_CELLS or not isinstance(v, int) or isinstance(v, bool)
               or not 0 < v <= hs.V9_CELLS[k]}
        raise ParityError("格表非法（须为 V9_CELLS 的非空子集、局数为不超过该表的正整数）："
                          f"{bad or '空'}")
    return cells, cells_version(cells, hs)


def parse_cells(spec: str | None, hs=None) -> dict[tuple[str, str], int]:
    """格表参数（只要格表）；名字与 JSON 形态见 :func:`parse_cells_versioned`。"""
    return parse_cells_versioned(spec, hs)[0]


def root_cell_table(specs_root: str | Path, hs=None) -> tuple[str, dict[tuple[str, str], int]] | None:
    """/4 规格根（``<根>/xhard{1..5}/specs.jsonl``）→ ``(版本, 完整交付格表)``：汇总各档 header 的
    ``delivery_per_cell`` 成格表，按 :func:`cells_version` 判版本（完整 V9 根恰等于 V9_CELLS；局部根如 MoveCube 50
    的分片根判 v9）。没有任何 /4 文件（v7 根、空根）返回 None；首行读不出的文件跳过。"""
    hs = hs or hard_specs_light()
    cells: dict[tuple[str, str], int] = {}
    for tier in hs.TIERS:
        path = Path(specs_root) / tier / "specs.jsonl"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as stream:
                header = json.loads(stream.readline())
        except (OSError, ValueError):
            continue
        if not isinstance(header, dict) or header.get("schema") != hs.SCHEMA:
            continue
        quota = header.get("delivery_per_cell")
        if isinstance(quota, dict):
            for task, n in quota.items():
                cells[(task, header.get("difficulty"))] = n
    if not cells:
        return None
    version = cells_version(cells, hs)
    return version, hs.CELL_TABLES[version]


def load_specs_any(path: str | Path, hs=None, *, check_fingerprint: bool = False):
    """单文件读取（同 ``_rollout.load_specs_any``）：/4 文件的配额上限格表按 header 自带的逐任务配额推出
    （``hard_specs.header_cell_table``：V9 文件落 V9_CELLS）；/2、/3 文件已不再支持。"""
    hs = hs or hard_specs_light()
    with Path(path).open(encoding="utf-8") as stream:
        first = stream.readline()
    try:
        table = hs.header_cell_table(json.loads(first))
    except ValueError:
        table = None  # 首行坏了：交给 load_specs 报具体错
    return hs.load_specs(path, expected_cells=table, check_fingerprint=check_fingerprint)


def read_identity_subset(path: Path) -> set[tuple[str, str, int]]:
    """``--identities``：只比／只生成／只登记给定身份子集（v9 的新 80 局）。接受

    * jsonl：每行 ``{task, tier|difficulty, seed}``；
    * json：交付清单（``rows``／``delivered``）或身份列表。行带 ``source`` 字段（``v9_subset_specs.py assemble`` 的
      800 行清单，``v8-reuse|v9-new``）时只取 ``source == "v9-new"`` 的行；不带时取全部行。
    空子集即报错（空分母不得判 PASS）。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        items = [json.loads(t) for t in text.splitlines() if t.strip()]
    else:
        payload = json.loads(text)
        items = (payload.get("rows") or payload.get("delivered") or []) if isinstance(payload, dict) else payload
    if any(isinstance(r, dict) and "source" in r for r in items):
        items = [r for r in items if r.get("source") == "v9-new"]
    subset = {(r["task"], r.get("tier", r.get("difficulty")), int(r["seed"])) for r in items}
    if not subset:
        raise ParityError(f"--identities 子集为空：{path}")
    return subset


def default_h5_root(tier: str) -> Path:
    """``--h5-root`` 缺省：v9 → ``artifacts/newtask-v9/parity/h5``，其余 → v8 目录（行为不变）。"""
    return V9_LOCAL_H5_ROOT if tier == "v9" else LOCAL_H5_ROOT


def default_compare_root(tier: str) -> Path:
    return V9_COMPARE_ROOT if tier == "v9" else COMPARE_ROOT


def frozen_delivery(specs_root: Path, cells: dict[tuple[str, str], int],
                    cell_table: dict[tuple[str, str], int] | None = None) -> list[dict[str, Any]]:
    """冻结交付集：/4 规格根里 ``delivered``（selected 且 rollout ok）的行；先过 ``load_specs_root`` 全部校验。
    ``cell_table``：作配额上限的完整格表（v9 传 V9_CELLS）；缺省由 ``hard_specs.resolve_cell_table`` 推出。"""
    hs = hard_specs_light()
    try:
        loaded = hs.load_specs_root(specs_root, cells, cell_table=cell_table, check_fingerprint=False)
    except Exception as exc:  # noqa: BLE001 冻结根本身不合法即停（第⑤类之上的前置错误）
        raise ParityError(f"冻结规格根校验失败：{type(exc).__name__}: {exc}") from exc
    return [{"task": r["task"], "tier": tier, "episode": int(r["episode"]), "seed": int(r["seed"]),
             "h5_sha256": (r.get("rollout") or {}).get("h5_sha256")}
            for tier, (_, rows) in loaded.items() for r in rows if hs.delivered(r)]


def ident(row: dict[str, Any]) -> tuple[str, str, int]:
    return (row["task"], row["tier"], int(row["seed"]))


# ── generate ────────────────────────────────────────────────────────────


class Mover(threading.Thread):
    """盯各轮 ``results.partial.jsonl``：新完成的局算 sha256，写 identities 行；给了暂存目录就复制过去、核 sha、删本地副本。

    ``expect_ref``（``generate --expect-ref``，对拍细则 3.4「边生成边判定」）：算出 sha 后立刻查噪声基线参照
    （``noise_gate.py gen-regress build-ref`` 的产物）里这一局的期望，在同一行 identities 写 ``verdict``：

    - ``match``：与基线逐字节相同（确定性失败局为失败且占位 sha 相同）→ 删本地 h5／mp4，不调 ``_ship``；
    - ``jitter_info``：已知抖动局 → 照常 ``_ship``；
    - ``flip``：期望不符（含参照里没有这一局）→ 照常 ``_ship``，追加一行 ``flips.jsonl``（``{task, tier, seed}``，可直接作
      ``generate --identities``），打印 ``EPISODE_FLIP <tier>/<task>/<seed>``。

    判定函数与 ``gen-regress check`` 共用 ``noise_gate.expect_verdict``。不给 ``expect_ref`` 时代码路径与原先相同。"""

    def __init__(self, out: Path, stage: Path | None, side: str, tier: str, runner_meta: dict[str, Any],
                 expect_ref: Path | None = None):
        super().__init__(daemon=True)
        self.out, self.stage, self.side, self.tier, self.meta = out, stage, side, tier, runner_meta
        self.seen: set[tuple[str, int, str]] = set()
        self.stop_flag = threading.Event()
        self.errors: list[str] = []
        self.identities = out / "identities.jsonl"
        self.flips = out / "flips.jsonl"
        self.expect: dict[tuple[str, int], dict[str, Any]] | None = None
        self._verdict = None
        if expect_ref is not None:
            noise_gate = self._load_noise_gate()
            self.expect = noise_gate.ref_index(noise_gate.load_ref(expect_ref))
            self._verdict = noise_gate.expect_verdict

    @staticmethod
    def _load_noise_gate():
        """按文件路径加载同目录的 ``noise_gate.py``（模块顶层只依赖标准库）。"""
        import importlib.util

        name = "_hard_parity_noise_gate"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, HERE / "noise_gate.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def _judge(self, line: dict[str, Any]) -> str:
        """对参照判定一局并写进 line（``verdict``、``ref_class``）。"""
        entry = self.expect.get((line["task"], int(line["seed"])))
        ok = bool(line["success"]) and not line.get("h5_error")
        verdict = self._verdict(entry, ok, line["sha256"])
        line["verdict"] = verdict
        line["ref_class"] = entry.get("class") if entry else None
        if entry is None:
            line["verdict_note"] = "not_in_ref"
        return verdict

    def _drop_media(self, wdir: Path) -> None:
        """``match`` 局：本机与 HF 上已有同字节，节点 /tmp 的 h5／mp4 直接删、不复制。"""
        for path in sorted((p for p in wdir.rglob("*") if p.is_file() and p.suffix in (".h5", ".mp4")), reverse=True):
            path.unlink()

    def _record_flip(self, line: dict[str, Any]) -> None:
        entry = self.expect.get((line["task"], int(line["seed"])))
        tier = entry["tier"] if entry else line["tier"]
        with self.flips.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"task": line["task"], "tier": tier, "seed": int(line["seed"])},
                                    ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"EPISODE_FLIP {tier}/{line['task']}/{line['seed']}", flush=True)

    def _worker_dir(self, record: dict[str, Any]) -> Path:
        tier = str(record.get("difficulty"))
        nested = self.out / "episodes" / tier / f"{record['task']}_episode_{record['episode']}"
        return nested if nested.exists() else self.out / "episodes" / f"{record['task']}_episode_{record['episode']}"

    def handle(self, record: dict[str, Any]) -> None:
        key = (record["task"], int(record["episode"]), str(record.get("difficulty")))
        if key in self.seen:
            return
        self.seen.add(key)
        wdir = self._worker_dir(record)
        h5s = sorted((wdir / "hdf5_files").glob("*.h5")) if wdir.exists() else []
        line = {"side": self.side, "tier": str(record.get("difficulty")), "task": record["task"],
                "episode": int(record["episode"]), "seed": int(record["seed"]), "success": bool(record.get("ok")),
                "error_type": record.get("error_type"), "env_package": record.get("env_package"),
                "env_module": record.get("env_module"), "wrapper_modules": record.get("wrapper_modules"),
                "worker": self.meta.get("worker"), "robomme_module": self.meta.get("robomme_module"),
                "recovery_mode": record.get("recovery_mode"), "builder_route": record.get("builder_route"),
                "builder_episode": record.get("builder_episode"), "builder_tier": record.get("builder_tier"),
                "path": None, "bytes": None, "sha256": None, "frames": None, "media": []}
        if h5s:
            h5 = h5s[0]
            line.update(sha256=sha256_file(h5), bytes=h5.stat().st_size)
            try:
                import h5py

                with h5py.File(h5, "r") as handle:
                    episode = handle[list(handle.keys())[0]]
                    line["frames"] = sum(1 for k in episode if k.startswith("timestep_"))
            except Exception as exc:  # noqa: BLE001 打不开也如实记
                line["h5_error"] = f"{type(exc).__name__}: {exc}"
            rel = wdir.relative_to(self.out)
            line["path"] = str(rel / "hdf5_files" / h5.name)
            line["media"] = [str(p.relative_to(self.out)) for p in sorted(wdir.rglob("*.mp4"))]
            verdict = self._judge(line) if self.expect is not None else None
            if verdict == "match":
                self._drop_media(wdir)
            elif self.stage is not None:
                self._ship(wdir, rel)
        elif self.expect is not None:
            self._judge(line)
        if self.expect is not None and line.get("verdict") == "flip":
            self._record_flip(line)
        with self.identities.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"EPISODE_DONE side={self.side} {line['tier']}/{line['task']}/{line['episode']} "
              f"success={int(line['success'])} sha={str(line['sha256'])[:12]}"
              + (f" verdict={line['verdict']}" if "verdict" in line else ""), flush=True)

    def _ship(self, wdir: Path, rel: Path) -> None:
        dest = self.stage / rel
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(p for p in wdir.rglob("*") if p.is_file()):
            target = dest / path.relative_to(wdir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if path.suffix == ".h5" and sha256_file(target) != sha256_file(path):
                self.errors.append(f"暂存副本 sha 不符：{target}")
                return
        # 完成标记：拉取端只拉带 SHIPPED 的局，避免拉到复制了一半的文件
        shipped = {str(p.relative_to(wdir)): sha256_file(p) for p in sorted(wdir.rglob("*"))
                   if p.is_file() and p.suffix in (".h5", ".mp4")}
        (dest / "SHIPPED").write_text(json.dumps(shipped, ensure_ascii=False, sort_keys=True) + "\n")
        for path in sorted((p for p in wdir.rglob("*") if p.is_file() and p.suffix in (".h5", ".mp4")), reverse=True):
            path.unlink()  # 节点 /tmp 只留小文件（sidecar），大文件搬走即删

    def scan(self) -> None:
        for partial in sorted(self.out.rglob("results.partial.jsonl")):
            for text in partial.read_text(encoding="utf-8").splitlines():
                if text.strip():
                    try:
                        self.handle(json.loads(text))
                    except Exception as exc:  # noqa: BLE001
                        self.errors.append(f"{type(exc).__name__}: {exc}")

    def run(self) -> None:
        while not self.stop_flag.is_set():
            self.scan()
            self.stop_flag.wait(5)
        self.scan()


def xhard0_subset_rows(rows: list[dict[str, Any]], subset: set[tuple[str, str, int]]) -> list[dict[str, Any]]:
    """``generate --tier xhard0 --identities``：从完整 xhard0 清单行（tier 为官方 ``hard``）里按 (task, seed) 筛出子集。

    子集身份的 tier 须为 ``xhard0``；有清单外身份即报错。只筛要生成的 jobs，传给运行器的 ``--xhard0-manifest``
    仍是完整 192 行清单（运行器把清单与官方 test 元数据 hard 子集双向核对，不能传子集）。"""
    bad_tier = sorted(s for s in subset if s[1] != "xhard0")
    if bad_tier:
        raise ParityError(f"--tier xhard0 的 --identities 身份 tier 须为 xhard0：{bad_tier[:3]}")
    keys = {(task, int(seed)) for task, _tier, seed in subset}
    stray = sorted(keys - {(r["task"], int(r["seed"])) for r in rows})
    if stray:
        raise ParityError(f"--identities 含 xhard0 清单之外的身份 {len(stray)} 个：{stray[:3]}")
    return [r for r in rows if (r["task"], int(r["seed"])) in keys]


def run_with_mover(command: list[str], env: dict[str, str], out: Path, side: str, tier: str, meta: dict[str, Any], *,
                   stage: Path | None = None, expect_ref: Path | None = None):
    """起生成子进程（输出进 ``<out>/generate.log``），同时用 :class:`Mover` 逐局写 ``<out>/identities.jsonl``。
    返回 ``(子进程结果, identities 行, mover)``。``generate`` 与 ``smoke`` 共用。"""
    mover = Mover(out, stage, side, tier, meta, expect_ref=expect_ref)
    mover.start()
    with (out / "generate.log").open("a", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    runner_json = out / "_runner" / "results.json"
    if runner_json.exists():
        payload = json.loads(runner_json.read_text())
        meta.update(robomme_module=payload.get("robomme_module"), worker=payload.get("worker"))
    mover.stop_flag.set()
    mover.join()
    lines = [json.loads(t) for t in (out / "identities.jsonl").read_text().splitlines() if t.strip()] \
        if (out / "identities.jsonl").exists() else []
    return proc, lines, mover


def cmd_generate(args) -> int:
    if args.expect_ref is not None:
        # 参照校验（schema／顶层 sha／partial 拒收）提前到建输出目录、写 launch 之前：参照坏时不留任何文件。
        # Mover 构造时仍按原逻辑再读一次参照（语义不变）
        try:
            Mover._load_noise_gate().load_ref(args.expect_ref)
        except Exception as exc:  # noqa: BLE001 GateError 与读文件错误一律转成前置失败
            raise ParityError(f"--expect-ref 参照不可用：{type(exc).__name__}: {exc}") from exc
    facts = gpu_facts()
    if not args.dev_smoke and "A40" not in facts["gpu_model"]:
        raise ParityError(f"正式 generate 只许在 A40 上跑：当前 {facts}")
    rows = rows_for(args.tier, args.manifest)
    if args.tier in DELIVERY_TIERS and args.side not in ("H2",):
        raise ParityError(f"{args.tier} 经 hard_parity 只生成 H2（gen2）；gen1 由 generate_h5 --mode continue 出，再 import-delivery 登记为 H")
    if args.identities is not None:
        if args.tier == "xhard0":
            rows = xhard0_subset_rows(rows, read_identity_subset(args.identities))
        elif args.tier not in DELIVERY_TIERS:
            raise ParityError("--identities 只用于 --tier v9／xhard0")
        else:
            subset = read_identity_subset(args.identities)
            stray = sorted(subset - {ident(r) for r in rows})
            if stray:
                raise ParityError(f"--identities 含交付清单之外的身份 {len(stray)} 个：{stray[:3]}")
            rows = [r for r in rows if ident(r) in subset]
    if args.tier == "xhard0" and args.side not in ("O", "H"):
        raise ParityError("xhard0 只生成 O、H 两侧（首次，无 P）")
    if args.smoke:
        rows = rows[: args.smoke]
    out = args.out
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ParityError(f"{out} 已存在且非空；续跑用 --resume")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    meta: dict[str, Any] = {}
    if args.tier in ("native", "xhard0"):
        jobs = [{"task": r["task"], "episode": r["episode"], "seed": r["seed"], "difficulty": r["tier"],
                 "worker_dir": str(out / "episodes" / f"{r['task']}_episode_{r['episode']}")} for r in rows]
        runner_dir = out / "_runner"
        runner_dir.mkdir(exist_ok=True)
        (runner_dir / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1))
        command = [sys.executable, str(RUNNER), "--official-root", str(VENDOR), "--src-root", str(args.src_root),
                   "--jobs-json", str(runner_dir / "jobs.json"), "--results-json", str(runner_dir / "results.json"),
                   "--workers", str(args.workers), "--gpu", str(args.gpu)]
        if args.tier == "xhard0":
            # xhard0 身份来自官方 test 元数据 hard 子集（train_metadata 会判不符），两侧同一复核（v7 §1.5）
            command += ["--identity-source", "test_metadata", "--xhard0-manifest", str(args.manifest)]
        if args.side == "H":
            command.append("--force-mirror")
            env["ROBOMME_ENV_PACKAGE"] = "robomme_hard"
            meta["worker"] = "train_split_worker.run_one"
            if args.tier == "xhard0":
                # H 侧 gym.make 实参取自 robomme_hard 的 builder（dataset="hard-verify"，局 0～11 即 xhard0）（R9）
                command += ["--builder-route", "hard-verify"]
                meta["builder_route"] = "hard-verify"
        else:
            env["ROBOMME_ENV_PACKAGE"] = "robomme"
            meta["worker"] = "official._worker"
        if args.resume:
            command.append("--resume")
    else:
        identities = out / "_identities.jsonl"
        identities.write_text("".join(json.dumps({"task": r["task"], "tier": r["tier"], "seed": r["seed"]}) + "\n"
                                      for r in rows))
        command = [sys.executable, str(GENERATE_H5), "--mode", "replay", "--identities", str(identities),
                   "--output", str(out), "--workers", str(args.workers), "--gpu", str(args.gpu),
                   "--pkg", "robomme_hard", "--src-root", str(args.src_root)]
        if args.tier in DELIVERY_TIERS:
            if args.specs_root is None:
                raise ParityError(f"--tier {args.tier} 须给 --specs-root（{args.tier} 五档 /4 规格根，gen2 按 gen1 交付清单重放）")
            command += ["--specs", str(args.specs_root)]
        if args.dev_smoke:
            command.append("--dev-smoke")
        if args.resume:
            command.append("--resume")
        meta["worker"] = "train_split_worker.run_one"
    launch = {"schema": "hard-parity-launch/1", "side": args.side, "tier": args.tier, "rows": len(rows),
              "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
              "identities": str(args.identities) if args.identities else None,
              "src_root": str(args.src_root), "src_commit": subprocess.run(
                  ["git", "-C", str(args.src_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "orchestration": str(VENDOR), "workers": args.workers, "gpu": args.gpu, **facts,
              "versions": versions(), "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
              "started_at": now(), "command": command, "env_package": env.get("ROBOMME_ENV_PACKAGE", "robomme_hard"),
              "stage": str(args.stage) if args.stage else None, "dev_smoke": bool(args.dev_smoke)}
    (out / f"launch-{int(time.time())}.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2))
    print(f"GENERATE_START side={args.side} tier={args.tier} rows={len(rows)} gpu={facts['gpu_model']} "
          f"driver={facts['driver']} workers={args.workers}", flush=True)
    proc, lines, mover = run_with_mover(command, env, out, args.side, args.tier, meta, stage=args.stage,
                                        expect_ref=args.expect_ref)
    got = {ident(r) for r in lines}
    want = {ident(r) for r in rows}
    ok_count = sum(r["success"] for r in lines)
    status = "PASS" if got == want and proc.returncode in (0, 1) and not mover.errors else "FAIL"
    label = "NATIVE_SMOKE" if args.dev_smoke else ("SHARD_SMOKE" if args.smoke else "GENERATE")
    print(f"{label}={status if not args.smoke or ok_count == len(rows) else 'FAIL'} side={args.side} tier={args.tier} "
          f"rows={len(rows)} recorded={len(got & want)} success={ok_count} runner_exit={proc.returncode} "
          f"gpu={facts['gpu_model']} mover_errors={len(mover.errors)}", flush=True)
    for error in mover.errors[:10]:
        print(f"# {error}")
    return 0 if status == "PASS" else 1


# ── publish ─────────────────────────────────────────────────────────────


def side_prefix(side: str, tier: str, manifest: dict[str, Any]) -> str:
    return f"{manifest['prefix']}/{tier}"


def cmd_publish(args) -> int:
    local = args.local or (args.h5_root / f"{args.side}-{args.tier}")
    ident_path = local / "identities.jsonl"
    if not ident_path.exists():
        raise ParityError(f"{ident_path} 不存在")
    lines = [json.loads(t) for t in ident_path.read_text().splitlines() if t.strip()]
    sums = []
    for line in lines:
        if line.get("path"):
            path = local / line["path"]
            actual = sha256_file(path)
            if actual != line["sha256"]:
                raise ParityError(f"本地副本 sha 与 identities 不符：{path}")
            sums.append(f"{actual}  {line['path']}")
    (local / "SHA256SUMS").write_text("\n".join(sorted(sums)) + "\n")
    launches = sorted(local.glob("launch-*.json"))
    first = json.loads(launches[0].read_text()) if launches else {}
    manifest = {
        "schema": "hard-parity-side/1", "side": args.side, "tier": args.tier, "prefix": args.prefix,
        "src_commit": first.get("src_commit"), "orchestration_commit": "d53f21a7947d2d8daf6e3e8bad9f59b4f89a77fa",
        "tag": "pre-hard-split" if args.side == "P" else None, "code_baseline": args.code_baseline,
        "gpu_model": first.get("gpu_model", args.gpu_model), "driver": first.get("driver", args.driver),
        "versions": first.get("versions"), "job_ids": sorted({json.loads(p.read_text()).get("slurm_job") for p in launches} - {None}),
        "nodes": sorted({json.loads(p.read_text()).get("host") for p in launches} - {None}),
        "workers": first.get("workers", args.workers), "manifest": first.get("manifest"),
        "subset_manifest_sha256": first.get("manifest_sha256"), "objects": len(sums), "generated_at": first.get("started_at"),
        "published_at": now(), "notes": args.note,
    }
    (local / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    remote = f"hf://buckets/{BUCKET}/{args.prefix}/{args.tier}"
    listing = subprocess.run([*HF, "buckets", "list", f"{BUCKET}/{args.prefix}/{args.tier}", "-R"],
                             capture_output=True, text=True)
    existing = [l for l in listing.stdout.splitlines()
                if l.strip() and l.strip() != "(empty)" and not l.startswith(("ID", "---", "Installed ", "Resolved "))]
    if existing and not args.resume_upload:
        raise ParityError(f"bucket 目录已存在（只增不改，R13）：{remote}；重传须请示并另起目录名")
    include = ["--include", "identities.jsonl", "--include", "SHA256SUMS", "--include", "manifest.json",
               "--include", "launch-*.json", "--include", "*.h5"]
    if args.dry_run:
        print(f"PUBLISH_DRY_RUN remote={remote} objects={len(sums)}")
        return 0
    # 只给 --include 即白名单；再加 --exclude "*" 会把全部排除（1.8.0 实测 uploads=0）
    subprocess.run([*HF, "buckets", "sync", str(local), remote, *include], check=True)
    mismatch = 0
    with tempfile.TemporaryDirectory(dir=args.readback_tmp) as tmp:
        for entry in sums:
            sha, rel = entry.split("  ", 1)
            target = Path(tmp) / "obj"
            subprocess.run([*HF, "buckets", "cp", f"{remote}/{rel}", str(target)], check=True, capture_output=True)
            if not target.is_file():  # 远端没有该对象时 cp 只警告不报错
                mismatch += 1
                continue
            mismatch += int(sha256_file(target) != sha)
            target.unlink()
    listing = subprocess.run([*HF, "buckets", "list", f"{BUCKET}/{args.prefix}/{args.tier}", "-R"],
                             capture_output=True, text=True).stdout
    (local / "bucket-list.txt").write_text(listing)
    status = "PASS" if mismatch == 0 else "FAIL"
    print(f"BUCKET_SYNC={status} side={args.side} tier={args.tier} objects={len(sums)} "
          f"readback_sha_equal={len(sums) - mismatch} mismatch={mismatch} remote={remote}", flush=True)
    return 0 if status == "PASS" else 1


# ── compare ─────────────────────────────────────────────────────────────


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "tolist"):
        return _decode(value.tolist())
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def read_setup(episode) -> dict[str, Any]:
    setup = episode["setup"]
    keys = ("seed", "difficulty", "task_goal", "available_multi_choices", "front_camera_intrinsic",
            "wrist_camera_intrinsic")
    return {k: _decode(setup[k][()]) for k in keys if k in setup}


def schema_of(episode) -> list[str]:
    import h5py

    items: set[str] = set()

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            head, _, rest = name.partition("/")
            group = "timestep" if head.startswith("timestep_") else head
            items.add(f"{group}/{rest}|{obj.dtype}|{obj.shape}")
    episode.visititems(visit)
    return sorted(items)


def side_lines(side_dir: Path) -> list[dict[str, Any]]:
    """读一侧 identities.jsonl；运行中逐局写的行拿不到 runner 结束后才有的 robomme_module／worker，
    从该侧 _runner/results.json（runner 探针原值）补齐空缺，不覆盖已有值。"""
    lines = [json.loads(t) for t in (side_dir / "identities.jsonl").read_text().splitlines() if t.strip()]
    runner = side_dir / "_runner" / "results.json"
    if runner.is_file():
        payload = json.loads(runner.read_text())
        for line in lines:
            if line.get("robomme_module") is None:
                line["robomme_module"] = payload.get("robomme_module")
            if line.get("worker") is None and payload.get("worker"):
                line["worker"] = payload.get("worker")
    else:
        # bucket 拉回的一侧不含 _runner/results.json：按 launch 记录补 robomme_module。runner 探针本就断言
        # robomme 解析到 <src_root>/src/robomme，故该值与探针原值相同（v7 实测 O-native 82 行有值者全部等于此路径）
        launches = sorted(side_dir.glob("launch-*.json"))
        launch = json.loads(launches[0].read_text()) if launches else {}
        if launch.get("env_package") == "robomme" and launch.get("src_root"):
            derived = f"{launch['src_root']}/src/robomme/__init__.py"
            for line in lines:
                if line.get("robomme_module") is None and line.get("worker") == "official._worker":
                    line["robomme_module"] = derived
                    line["robomme_module_source"] = "launch"
    return lines


def self_check(side_dir: Path, rows: list[dict[str, Any]],
               subset: set[tuple[str, str, int]] | None = None) -> tuple[dict[tuple, dict], list[str]]:
    """一侧 identities 自检：无重复、身份集合 = 清单。``subset``（--identities）时该侧只看子集内的行。"""
    lines = side_lines(side_dir)
    if subset is not None:
        lines = [line for line in lines if ident(line) in subset]
    problems = []
    by_id: dict[tuple, dict] = {}
    for line in lines:
        key = ident(line)
        if key in by_id:
            problems.append(f"重复身份 {key}")
        by_id[key] = line
    want = {ident(r) for r in rows}
    if set(by_id) != want:
        problems.append(f"身份集合不符：缺 {len(want - set(by_id))} 多 {len(set(by_id) - want)}")
    return by_id, problems


def _ts(episode) -> list[str]:
    return sorted((k for k in episode if k.startswith("timestep_")), key=lambda k: int(k.split("_")[1]))


def pair_metrics(left: str, right: str) -> dict[str, Any]:
    """一对 h5：setup／结构／四项容差指标／首个分叉步。在进程池里跑。"""
    import h5py
    import numpy as np

    out: dict[str, Any] = {"left": left, "right": right}
    try:
        with h5py.File(left, "r") as lf, h5py.File(right, "r") as rf:
            le, re_ = lf[list(lf.keys())[0]], rf[list(rf.keys())[0]]
            out["setup_left"], out["setup_right"] = read_setup(le), read_setup(re_)
            out["setup_equal"] = out["setup_left"] == out["setup_right"]
            out["schema_equal"] = schema_of(le) == schema_of(re_)
            lt, rt = _ts(le), _ts(re_)
            out["contiguous"] = [int(k.split("_")[1]) for k in lt] == list(range(len(lt))) and \
                [int(k.split("_")[1]) for k in rt] == list(range(len(rt)))
            out["frames_left"], out["frames_right"] = len(lt), len(rt)
            n = min(len(lt), len(rt))
            action = state = 0.0
            mads = []
            diverge = None
            for i in range(n):
                a, b = le[lt[i]], re_[rt[i]]
                ja, jb = a["action/joint_action"][()], b["action/joint_action"][()]
                d_action = float(np.max(np.abs(ja.astype(np.float64) - jb.astype(np.float64)))) if ja.size else 0.0
                d_state = max(float(np.max(np.abs(a[f"obs/{k}"][()].astype(np.float64) - b[f"obs/{k}"][()].astype(np.float64))))
                              for k in ("joint_state", "gripper_state"))
                action, state = max(action, d_action), max(state, d_state)
                frame = [float(np.mean(np.abs(a[f"obs/{c}"][()].astype(np.int16) - b[f"obs/{c}"][()].astype(np.int16))))
                         for c in ("front_rgb", "wrist_rgb")]
                mads.append(sum(frame) / 2)
                if diverge is None and (d_action > 0 or d_state > 0 or frame[0] > 0 or frame[1] > 0):
                    diverge = i
            out.update(action_max=action, state_max=state, image_mad=float(np.mean(mads)) if mads else 0.0,
                       frames_diff=abs(len(lt) - len(rt)), first_divergence=diverge, common=n, error=None)
    except Exception as exc:  # noqa: BLE001 打不开即判定层不等
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def load_tolerances() -> dict[str, float]:
    if not TOLERANCES.is_file():
        raise ParityError(f"容差文件不存在，拒跑（R21）：{TOLERANCES}；先跑 compare --pair O:P --calibrate")
    payload = json.loads(TOLERANCES.read_text())
    return {k: float(payload[v]) for k, v in TOL_KEYS.items()}


def calibrate(pairs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    import numpy as np

    raw_p95, raw_max, tol, ceiling_hit = {}, {}, {}, []
    for key, field in (("action_max", "action_max"), ("state_max", "state_max"), ("image_mad", "image_mad"),
                       ("frames_max", "frames_diff")):
        values = [p[field] for p in pairs if p.get("error") is None]
        raw_p95[key] = float(np.percentile(values, 95)) if values else 0.0
        raw_max[key] = float(max(values)) if values else 0.0
        rule = TOL_RULES[key]
        value = raw_max[key] * 1.5
        value = math.ceil(value) if key == "frames_max" else value
        tol[key] = max(value, rule["floor"])
        if raw_max[key] > rule["ceiling"]:
            ceiling_hit.append(key)
    payload = {TOL_KEYS[k]: tol[k] for k in tol}
    payload.update(calibrated_from="O:P native", n=len(pairs), raw_p95=raw_p95, raw_max=raw_max,
                   rule="最大值 × 1.5（帧数向上取整），下界 0.005/0.005/1.0/5，合理性上界 0.05/0.05/10/200（U-22）",
                   calibrated_at=now())
    return not ceiling_hit, {"payload": payload, "ceiling_hit": ceiling_hit}


def _anchor_registry(*, create: bool = False) -> dict[str, Any]:
    """锚点登记表 ``docs/validation/parity-anchors.json``。文件缺失时响亮失败（原先静默返回空锚点，会把「登记表丢了」
    误报成「锚点未登记」）；只有 ``anchor register`` 首次登记（``create=True``）允许从空表起步。"""
    if ANCHORS.is_file():
        return json.loads(ANCHORS.read_text())
    if create:
        return {"schema": "parity-anchors/1", "anchors": {}}
    raise ParityError(f"锚点登记表不存在：{ANCHORS}")


def _git_tag_commit(tag: str) -> str | None:
    proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or None


def _identities_digest(side_dir: Path) -> tuple[str, int, list[dict[str, Any]]]:
    lines = side_lines(side_dir)
    body = "\n".join(sorted(f"{l.get('sha256')}  {l.get('path')}" for l in lines))
    return hashlib.sha256(body.encode()).hexdigest(), len(lines), lines


def anchor_check(tag: str, h5_root: Path, tiers: list[str] | None = None, *, rehash: bool = True) -> tuple[bool, str, dict]:
    """PARITY_ANCHOR：tag 解析出的 commit == 登记 commit；每段逐局 sha 重算 == identities 与登记摘要。"""
    entry = _anchor_registry()["anchors"].get(tag)
    if entry is None:
        return False, f"PARITY_ANCHOR=FAIL tag={tag} reason=未登记", {}
    problems, cached, sha_bad = [], [], 0
    resolved = _git_tag_commit(tag)
    if resolved is None or not resolved.startswith(entry["commit"]):
        problems.append(f"tag 指向 {resolved}，登记 {entry['commit']}")
    for tier, seg in entry["segments"].items():
        if tiers and tier not in tiers:
            continue
        side_dir = h5_root / seg["dir"]
        if not (side_dir / "identities.jsonl").is_file():
            problems.append(f"{tier} 段缺本机拉回目录 {side_dir}")
            continue
        digest_now, n, lines = _identities_digest(side_dir)
        if digest_now != seg["identities_sha256"] or n != seg["rows"]:
            problems.append(f"{tier} 段 identities 摘要／行数与登记不符")
        if rehash:
            for line in lines:
                if line.get("path"):
                    sha_bad += int(sha256_file(side_dir / line["path"]) != line["sha256"])
        cached.append(str(n))
    ok = not problems and sha_bad == 0
    line = (f"PARITY_ANCHOR={'PASS' if ok else 'FAIL'} tag={tag} commit={entry['commit'][:12]} "
            f"cached={'+'.join(cached)} sha_bad={sha_bad} gpu={entry.get('gpu_model')} driver={entry.get('driver')}"
            + ("" if ok else f" problems={problems[:3]}"))
    return ok, line, entry


def cmd_anchor(args) -> int:
    if args.action == "check":
        ok, line, _ = anchor_check(args.tag, args.h5_root)
        print(line, flush=True)
        return 0 if ok else 1
    registry = _anchor_registry(create=True)
    if args.tag in registry["anchors"] and not args.replace:
        raise ParityError(f"锚点 {args.tag} 已登记；锚点不移动（R5），要换锚点就另打新 tag")
    segments, gpus = {}, set()
    for item in args.segments.split(","):
        tier, name = item.split(":", 1)
        side_dir = args.h5_root / name
        digest_now, n, lines = _identities_digest(side_dir)
        launches = sorted(side_dir.glob("launch-*.json"))
        first = json.loads(launches[0].read_text()) if launches else {}
        manifest = json.loads((side_dir / "manifest.json").read_text()) if (side_dir / "manifest.json").is_file() else {}
        gpu = (first.get("gpu_model") or manifest.get("gpu_model"), first.get("driver") or manifest.get("driver"))
        gpus.add(gpu)
        segments[tier] = {
            "dir": name, "bucket_prefix": args.bucket_prefix.get(tier) if isinstance(args.bucket_prefix, dict) else None,
            "rows": n, "identities_sha256": digest_now, "gpu_model": gpu[0], "driver": gpu[1],
            "generated_at_commit": first.get("src_commit") or manifest.get("src_commit"),
            "source_side": args.source_side,
            "worker": sorted({l.get("worker") for l in lines} - {None}),
            "env_module_prefix": sorted({str(l.get("env_module") or "").split(".")[0] for l in lines} - {""}),
        }
    if len(gpus) != 1:
        raise ParityError(f"锚点各段 GPU／驱动不一致：{gpus}（D-13）")
    gpu_model, driver = gpus.pop()
    registry["anchors"][args.tag] = {"commit": args.commit, "gpu_model": gpu_model, "driver": driver,
                                     "segments": segments, "equivalence": args.equivalence, "registered_at": now()}
    ANCHORS.parent.mkdir(parents=True, exist_ok=True)
    ANCHORS.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    ok, line, _ = anchor_check(args.tag, args.h5_root)
    print(line, flush=True)
    return 0 if ok else 1


def cmd_import_delivery(args) -> int:
    """gen1 → H 侧：把 delivery.json 与逐局 h5 登记成 ``<h5-root>/H-<tier>/identities.jsonl``（只读 symlink，sha 重算核对）。
    交付行缺 h5 路径、缺 ``h5_sha256`` 或文件不存在都记为 sha_mismatch（不静默跳过）。
    ``--identities``（v9：新 80 局）只登记该子集；子集里有交付清单之外的身份即报错。"""
    delivery = read_delivery(args.delivery)
    if delivery["schema"] != V8_DELIVERY_SCHEMA:
        raise ParityError(f"--tier {args.tier} 只接受 {V8_DELIVERY_SCHEMA}：{delivery['schema']!r}")
    rows = delivery["rows"]
    if args.identities is not None:
        subset = read_identity_subset(args.identities)
        stray = sorted(subset - {ident(r) for r in rows})
        if stray:
            raise ParityError(f"--identities 含交付清单之外的身份 {len(stray)} 个：{stray[:3]}")
        rows = [r for r in rows if ident(r) in subset]
    h5_root = args.h5_root if args.h5_root is not None else default_h5_root(args.tier)
    out = h5_root / f"H-{args.tier}"
    if out.exists():
        raise ParityError(f"{out} 已存在")
    out.mkdir(parents=True)
    lines, bad = [], 0
    for r in rows:
        h5 = delivery_h5(delivery, r)
        if h5 is None or not h5.is_file() or not r.get("h5_sha256"):
            bad += 1
            continue
        src = h5.resolve()
        rel = Path("episodes") / r["tier"] / f"{r['task']}_episode_{r['episode']}" / "hdf5_files" / src.name
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).symlink_to(src)
        actual = sha256_file(src)
        bad += int(actual != r["h5_sha256"])
        lines.append({"side": "H", "tier": r["tier"], "task": r["task"], "episode": int(r["episode"]),
                      "seed": int(r["seed"]), "candidate": r.get("candidate"), "success": True, "error_type": None,
                      "env_package": "robomme_hard", "env_module": r.get("env_module"), "wrapper_modules": None,
                      "worker": "train_split_worker.run_one", "robomme_module": None, "recovery_mode": r.get("recovery_mode"),
                      "path": str(rel), "bytes": src.stat().st_size, "sha256": actual, "frames": r.get("frames"),
                      "exec_steps": r.get("exec_steps"), "media": [], "source": str(args.delivery)})
    (out / "identities.jsonl").write_text("".join(json.dumps(l, ensure_ascii=False, sort_keys=True) + "\n" for l in lines))
    for launch in sorted(delivery["base"].glob("launch-*.json")):
        shutil.copy2(launch, out / launch.name)
    status = "PASS" if bad == 0 and lines else "FAIL"
    print(f"IMPORT_DELIVERY={status} tier={args.tier} rows={len(lines)} sha_mismatch={bad} out={out}", flush=True)
    return 0 if status == "PASS" else 1


def _classify_over(record: dict[str, Any]) -> str:
    """B3（用户 2026-09-29 定「按首个分叉步判」）：setup／结构／成功与否都相等且 first_divergence>0 → noise；否则 fail。"""
    fd = record.get("first_divergence")
    ok_layers = bool(record.get("setup_equal")) and bool(record.get("schema_equal")) and record.get("success_same", False)
    return "noise" if ok_layers and fd is not None and int(fd) > 0 else "fail"


def compare_cells(args) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """``--tier v9`` 的 ``(格表, 作配额上限的完整格表)``：``--cells`` 缺省取 ``v9full``
    （不随 EXPECTED_CELLS 漂移）；给了格表时必须能被该档的完整表覆盖，否则报错（档与格表不得混用）。"""
    hs = hard_specs_light()
    table = hs.CELL_TABLES[args.tier]
    cells, _ = parse_cells_versioned(args.cells or f"{args.tier}full", hs)
    if not _fits(cells, table):
        raise ParityError(f"--cells 与 --tier {args.tier} 不符：格表须为 {args.tier.upper()}_CELLS 的子集"
                          f"（{args.tier} 档请用 {args.tier}full 或该表的子表）")
    return cells, table


def _v8_denominator(args, rows: list[dict[str, Any]], left_lines: list[dict[str, Any]],
                    right_lines: list[dict[str, Any]], subset: set[tuple[str, str, int]] | None = None) -> dict[str, Any]:
    """v8／v9 分母核对（v8 §2.2 第 11 条）：冻结交付集 F（/4 规格根里 delivered 的行）= delivery.json 的行 D = H 侧 = H2 侧。

    * ``missing``：F 中至少缺席 D／H／H2 之一的身份数；
    * ``extra``：出现在 D∪H∪H2 却不在 F 里的身份数；
    * ``duplicate``：四方各自的重复出现次数之和（同一身份在一方出现 k 次记 k−1）；
    * 比对全集 ``universe`` = F∪D∪H∪H2，``compared`` = 其大小，空集即 FAIL。

    ``subset``（``--identities``，v9 只比新 80 局）：四方都先限到该子集，子集本身作第五方（子集里不在冻结交付集的
    身份同时计 missing 与 extra）；``expected`` 改为子集大小。子集外的行数只记进 ``outside_subset``（报告用）。
    """
    cells, table = compare_cells(args)
    frozen = frozen_delivery(args.specs_root, cells, table)
    sources = {"frozen": [ident(r) for r in frozen], "delivery": [ident(r) for r in rows],
               "left": [ident(line) for line in left_lines], "right": [ident(line) for line in right_lines]}
    outside = {}
    if subset is not None:
        outside = {name: sum(key not in subset for key in items) for name, items in sources.items()}
        sources = {name: [key for key in items if key in subset] for name, items in sources.items()}
        sources["identities"] = sorted(subset)
    sets = {name: set(items) for name, items in sources.items()}
    counters = {name: collections.Counter(items) for name, items in sources.items()}
    dup_keys = {key for counter in counters.values() for key, c in counter.items() if c > 1}
    duplicate = sum(c - 1 for counter in counters.values() for c in counter.values() if c > 1)
    universe = set().union(*sets.values())
    in_all = set.intersection(*sets.values())
    frozen_kept = [r for r in frozen if subset is None or ident(r) in subset]
    return {"cells": cells, "expected": len(subset) if subset is not None else sum(cells.values()),
            "frozen_n": len(frozen_kept),
            "frozen_cells": len({key[:2] for key in sets["frozen"]}), "universe": universe, "in_all": in_all,
            "dup_keys": dup_keys, "duplicate": duplicate, "missing": len(sets["frozen"] - in_all)
            + (len(sets["identities"] - sets["frozen"]) if subset is not None else 0),
            "extra": len(universe - sets["frozen"]), "sizes": {name: len(items) for name, items in sources.items()},
            "outside_subset": outside, "subset": len(subset) if subset is not None else None,
            "frozen_sha": {ident(r): r.get("h5_sha256") for r in frozen_kept},
            "delivery_sha": {ident(r): r.get("h5_sha256") for r in rows}}


def classify_terminal(key: tuple, a: dict[str, Any] | None, b: dict[str, Any] | None, m: dict[str, Any],
                      denom: dict[str, Any], binding_ok: bool, recovery_bad: bool) -> str:
    """逐身份五个互斥终态（§2.2 第 11 条），每个身份恰好落一类：

    ⑤ ``structural``：身份不在四方交集、有重复、包归属（来源）不符、gen1 的 h5 sha 与交付清单／冻结规格不符、
      两侧都没产出、h5 打不开、setup 或结构（schema、帧号连续）不等；
    ③ ``h2_fail``：gen1（左）成功而二次生成（右）失败；④ ``flipped``：gen1 失败而二次生成成功；
    ① ``byte_equal``：两侧 h5 sha 相同；② ``noise``：身份／setup／schema／成败相同，只 sha 与帧数不同（首个分叉步另记）。
    """
    if key not in denom["in_all"] or key in denom["dup_keys"] or not binding_ok or recovery_bad:
        return "structural"
    success_a = bool(a and a.get("success") and a.get("path"))
    success_b = bool(b and b.get("success") and b.get("path"))
    if success_a:
        sha_a = a.get("sha256")
        for source in (denom["delivery_sha"].get(key), denom["frozen_sha"].get(key)):
            if source is not None and source != sha_a:
                return "structural"
    if success_a and not success_b:
        return "h2_fail"
    if success_b and not success_a:
        return "flipped"
    if not success_a and not success_b:
        return "structural"
    if m.get("error") is not None or not m.get("setup_equal") or not m.get("schema_equal") or not m.get("contiguous"):
        return "structural"
    return "byte_equal" if a.get("sha256") is not None and a.get("sha256") == b.get("sha256") else "noise"


def cmd_compare(args) -> int:
    left_side, right_side = args.pair.split(":")
    if args.calibrate and args.pair != "O:P":
        raise ParityError("--calibrate 只允许 O:P（R21）")
    if args.tier in DELIVERY_TIERS and args.specs_root is None:
        raise ParityError(f"--tier {args.tier} 须给 --specs-root（冻结交付集取自 /4 规格根的 delivered 行，用于分母核对）")
    if args.identities is not None and args.tier not in DELIVERY_TIERS:
        raise ParityError("--identities 只用于 --tier v9")
    h5_root = args.h5_root if args.h5_root is not None else default_h5_root(args.tier)
    compare_root = args.compare_root if args.compare_root is not None else default_compare_root(args.tier)
    rows = rows_for(args.tier, args.manifest)
    subset = read_identity_subset(args.identities) if args.identities is not None else None
    manifest_rows_all = len(rows)
    if subset is not None:
        rows = [r for r in rows if ident(r) in subset]
    anchor_entry = None
    dirs = {}
    for side in (left_side, right_side):
        explicit = args.left if side == left_side else args.right
        if explicit is not None:
            dirs[side] = explicit
        elif side == "P":
            if not args.p_anchor:
                raise ParityError("P 侧须给 --p-anchor <tag>（固定锚点，D-11）")
            ok, line, anchor_entry = anchor_check(args.p_anchor, h5_root, [args.tier])
            print(line, flush=True)
            if not ok:
                return 1
            if args.tier not in anchor_entry["segments"]:
                raise ParityError(f"锚点 {args.p_anchor} 没有 {args.tier} 段")
            dirs[side] = h5_root / anchor_entry["segments"][args.tier]["dir"]
        else:
            dirs[side] = h5_root / f"{side}-{args.tier}"
    left_dir, right_dir = dirs[left_side], dirs[right_side]
    out = compare_root / f"{left_side}{right_side}-{args.tier}"
    if args.run_name:
        out = out / args.run_name
    if out.exists() and not args.overwrite_compare:
        raise ParityError(f"{out} 已存在；另起子目录用 --run-name")
    out.mkdir(parents=True, exist_ok=True)
    left, left_problems = self_check(left_dir, rows, subset)
    right, right_problems = self_check(right_dir, rows, subset)
    for label, problems in ((left_side, left_problems), (right_side, right_problems)):
        print(f"SIDE_SELF_CHECK={'PASS' if not problems else 'FAIL'} side={label} tier={args.tier} "
              f"problems={len(problems)}" + (f" detail={problems[:3]}" if problems else ""), flush=True)
    # 分母：调用方清单本身不得为空、不得有重复身份（空清单时各项计数都是 0 = 0，旧口径会误判 PASS）
    manifest_ids = [ident(r) for r in rows]
    manifest_dup = len(manifest_ids) - len(set(manifest_ids))
    denom = None
    if args.tier in DELIVERY_TIERS:
        denom = _v8_denominator(args, rows, side_lines(left_dir), side_lines(right_dir), subset)
        keys = sorted(denom["universe"])
    else:
        keys = sorted(set(manifest_ids))
    jobs = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for key in keys:
            a, b = left.get(key), right.get(key)
            if a and b and a.get("path") and b.get("path"):
                jobs[key] = pool.submit(pair_metrics, str(left_dir / a["path"]), str(right_dir / b["path"]))
        metrics = {key: future.result() for key, future in jobs.items()}
    tol = None if args.calibrate else load_tolerances()
    counts = {k: 0 for k in ("identity_equal", "setup_equal", "schema_equal", "success_equal", "both_success",
                             "both_fail", "sha_equal", "frames_equal", "binding_ok", "recovery_mismatch")}
    terminal = {state: 0 for state in TERMINAL_STATES}
    noise_divergence: list[int] = []
    tol_hits, pair_rows, both_fail_rows = [], [], []
    for key in keys:
        a, b, m = left.get(key), right.get(key), metrics.get(key, {})
        record = {"task": key[0], "tier": key[1], "seed": key[2], "left": a, "right": b,
                  **{k: v for k, v in m.items() if k not in ("left", "right")}}
        present = a is not None and b is not None and (denom is None or key in denom["in_all"])
        counts["identity_equal"] += int(present)
        success_a = bool(a and a["success"] and a.get("path"))
        success_b = bool(b and b["success"] and b.get("path"))
        record["success_same"] = present and success_a == success_b
        counts["success_equal"] += int(record["success_same"])
        counts["both_success"] += int(success_a and success_b)
        if present and not success_a and not success_b:
            # 两侧同样没产出（如官方 test 的 hard seed 两侧同因规划失败）：单列，不当作不一致（v7 §1.5）
            counts["both_fail"] += 1
            both_fail_rows.append({"key": key, "left": a.get("error_type"), "right": b.get("error_type")})
        rec_a, rec_b = (a or {}).get("recovery_mode"), (b or {}).get("recovery_mode")
        recovery_bad = present and rec_a is not None and rec_b is not None and rec_a != rec_b
        counts["recovery_mismatch"] += int(recovery_bad)
        counts["setup_equal"] += int(bool(m.get("setup_equal")) and m.get("error") is None)
        counts["schema_equal"] += int(bool(m.get("schema_equal")) and bool(m.get("contiguous")))
        counts["sha_equal"] += int(present and a.get("sha256") is not None and a.get("sha256") == b.get("sha256"))
        counts["frames_equal"] += int(m.get("frames_diff") == 0)
        binding_ok = present and all(_binding_ok(side, line, anchor_entry, args.tier)
                                     for side, line in ((left_side, a), (right_side, b)))
        counts["binding_ok"] += int(binding_ok)
        record["binding_ok"] = binding_ok
        if denom is not None:
            state = classify_terminal(key, a, b, m, denom, binding_ok, recovery_bad)
            terminal[state] += 1
            record["terminal"] = state
            if state == "noise" and m.get("first_divergence") is not None:
                noise_divergence.append(int(m["first_divergence"]))
        if tol and m.get("error") is None and m:
            over = {k: m[f] for k, f in (("action_max", "action_max"), ("state_max", "state_max"),
                                          ("image_mad", "image_mad"), ("frames_max", "frames_diff")) if m[f] > tol[k]}
            if over:
                record["tol_over"] = over
                record["tol_class"] = _classify_over(record)
                tol_hits.append({"key": key, "over": over, "class": record["tol_class"],
                                 "first_divergence": m.get("first_divergence")})
        pair_rows.append(record)
    with (out / "h5_pairs.jsonl").open("w", encoding="utf-8") as stream:
        for record in pair_rows:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    n = len(keys)
    # 实际比较数：两侧 h5 都可读（pair_metrics 无错）的身份；任一侧 h5 读不出计 unreadable，不计入 compared
    compared = sum(1 for m in metrics.values() if m.get("error") is None)
    unreadable = sum(1 for m in metrics.values() if m.get("error") is not None)
    # 判定：分母非空、实际比较数 > 0 且等于身份数；两侧同失败（both_fail）单列、不计入通过——同失败只会让结论
    # 落到 FAIL（另打 PARITY_BOTH_FAIL=REVIEW 交人工看），不能 PASS
    judged = n > 0 and compared > 0 and compared == n and unreadable == 0 and counts["both_fail"] == 0 \
        and manifest_dup == 0 \
        and all(counts[k] == n for k in ("identity_equal", "success_equal", "binding_ok")) \
        and counts["setup_equal"] == counts["both_success"] == n and counts["schema_equal"] == n \
        and counts["recovery_mismatch"] == 0 and not left_problems and not right_problems
    if denom is not None:
        if sum(terminal.values()) != n:  # 互斥终态合计必须等于 compared（实现自检，不应发生）
            raise ParityError(f"五终态合计 {sum(terminal.values())} ≠ compared {n}")
        judged = judged and denom["missing"] == denom["extra"] == denom["duplicate"] == 0 \
            and n == denom["expected"] == denom["frozen_n"] and terminal["byte_equal"] + terminal["noise"] == n
    ok_metrics = [m for m in metrics.values() if m.get("error") is None]
    worst = {k: max((m[f] for m in ok_metrics), default=0.0) for k, f in (
        ("action_max", "action_max"), ("state_max", "state_max"), ("image_mad", "image_mad"), ("frames_max", "frames_diff"))}
    divergence = [m["first_divergence"] for m in ok_metrics if m.get("first_divergence") is not None]
    shape = SHAPES[args.tier]
    base = (f"tier={args.tier} shape={shape} keys={n} compared={compared} unreadable={unreadable} "
            f"identity_equal={counts['identity_equal']} "
            f"setup_equal={counts['setup_equal']} schema_equal={counts['schema_equal']} "
            f"success_equal={counts['success_equal']} both_success={counts['both_success']} both_fail={counts['both_fail']}")
    name = f"PARITY_{left_side}_{right_side}"
    if args.calibrate:
        within, result = calibrate(ok_metrics)
        payload = result["payload"]
        payload["tol_file_note"] = "由 hard_parity.py compare --pair O:P --calibrate 写入；改阈值须改本文件并写进留档（R21）"
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        TOLERANCES.parent.mkdir(parents=True, exist_ok=True)
        TOLERANCES.write_text(text)
        tol_sha = hashlib.sha256(text.encode()).hexdigest()
        print(f"PARITY_TOL_CALIB={'PASS' if within else 'FAIL'} pair=O:P n={len(ok_metrics)} "
              f"action_p95={payload['raw_p95']['action_max']:.3g} action_max={payload['raw_max']['action_max']:.3g} "
              f"state_max={payload['raw_max']['state_max']:.3g} image_mad_max={payload['raw_max']['image_mad']:.3g} "
              f"frames_max={payload['raw_max']['frames_max']:.3g} tol_file_sha={tol_sha[:12]}"
              + ("" if within else f" ceiling_hit={result['ceiling_hit']}（停止类）"), flush=True)
        tol = load_tolerances()
    noise = sum(hit["class"] == "noise" for hit in tol_hits)
    fail_over = len(tol_hits) - noise
    # B3 硬线（用户 2026-09-29）：任一 pair 超容差局数 > 该 pair 局数 5%，不论分类一律停
    hard_line = len(tol_hits) > 0.05 * n
    verdict = "PASS" if judged and fail_over == 0 and not hard_line else "FAIL"
    tol_text = " ".join(f"{k}={worst[k]:.3g}/{tol[k]:.3g}" for k in ("action_max", "state_max", "image_mad", "frames_max"))
    tail = (f"hard_line_5pct={'HIT' if hard_line else 'ok'} {tol_text} sha_equal={counts['sha_equal']} "
            f"frames_equal={counts['frames_equal']} binding_ok={counts['binding_ok']} "
            f"recovery_mismatch={counts['recovery_mismatch']}")
    if denom is None:
        print(f"{name}={verdict} {base} tol_over={fail_over} noise={noise} over_total={len(tol_hits)} {tail}"
              + ("" if n and not manifest_dup else f" manifest_rows={len(rows)} manifest_duplicate={manifest_dup}"),
              flush=True)
    else:
        # v8：分母与五终态在前（§3 判定行形状）；容差分类里的 noise 改名 tol_noise，免得与终态 noise 同名
        print(f"{name}={verdict} tier={args.tier} keys={n} compared={compared} unreadable={unreadable} "
              f"cells={denom['frozen_cells']} missing={denom['missing']} "
              f"extra={denom['extra']} duplicate={denom['duplicate']} identity_equal={counts['identity_equal']} "
              + " ".join(f"{state}={terminal[state]}" for state in TERMINAL_STATES)
              + f" expected={denom['expected']} frozen={denom['sizes']['frozen']} delivery={denom['sizes']['delivery']} "
              f"left_rows={denom['sizes']['left']} right_rows={denom['sizes']['right']} shape={shape} "
              f"setup_equal={counts['setup_equal']} schema_equal={counts['schema_equal']} "
              f"success_equal={counts['success_equal']} both_success={counts['both_success']} both_fail={counts['both_fail']} "
              f"noise_first_divergence_min={min(noise_divergence) if noise_divergence else None} "
              f"tol_over={fail_over} tol_noise={noise} over_total={len(tol_hits)} {tail}"
              + ("" if subset is None else f" identities={len(subset)} manifest_rows_all={manifest_rows_all}"),
              flush=True)
    if both_fail_rows:
        print(f"PARITY_BOTH_FAIL=REVIEW pair={args.pair} tier={args.tier} n={len(both_fail_rows)} "
              f"detail={both_fail_rows[:5]}", flush=True)
    print(f"PARITY_REFERENCE=INFO pair={args.pair} tier={args.tier} first_divergence_n={len(divergence)} "
          f"first_divergence_median={sorted(divergence)[len(divergence) // 2] if divergence else None} "
          f"first_divergence_min={min(divergence) if divergence else None}", flush=True)
    summary = {"verdict": verdict, "keys": n, "compared": compared, "unreadable": unreadable, "counts": counts,
               "worst": worst, "tolerances": tol, "tol_hits": tol_hits,
               "noise": noise, "tol_over": fail_over, "hard_line_5pct": hard_line, "both_fail": both_fail_rows,
               "left_dir": str(left_dir), "right_dir": str(right_dir), "p_anchor": args.p_anchor,
               "left_problems": left_problems, "right_problems": right_problems,
               "manifest_rows": len(rows), "manifest_duplicate": manifest_dup}
    if denom is not None:
        summary["terminal"] = terminal
        summary["denominator"] = {k: denom[k] for k in ("expected", "frozen_n", "frozen_cells", "missing", "extra",
                                                         "duplicate", "sizes", "subset", "outside_subset")}
        summary["identities"] = str(args.identities) if args.identities else None
        summary["terminal_by_identity"] = {"/".join(map(str, r_key)): rec["terminal"]
                                           for r_key, rec in zip(keys, pair_rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    return 0 if verdict == "PASS" else 1


def _binding_ok(side: str, line: dict[str, Any], anchor: dict[str, Any] | None = None, tier: str | None = None) -> bool:
    """ENV_PACKAGE_BINDING：H／H2 侧环境类模块必须属于 robomme_hard；O 侧走官方 _worker、robomme 来自其 src_root；
    P 侧按锚点登记表记下的生成来源判（锚点产物当年是作为 H 生成的，v7 §7.7）。"""
    if line is None:
        return False
    if side in ("H", "H2"):
        return str(line.get("env_module") or "").startswith("robomme_hard.")
    if side == "P" and anchor is not None and tier in anchor.get("segments", {}):
        seg = anchor["segments"][tier]
        worker_ok = not seg.get("worker") or line.get("worker") in seg["worker"]
        module = str(line.get("env_module") or "")
        module_ok = not seg.get("env_module_prefix") or (module.split(".")[0] in seg["env_module_prefix"])
        return worker_ok and module_ok
    module = str(line.get("robomme_module") or "")
    return line.get("worker") == "official._worker" and "/robomme/" in module and "robomme_hard" not in module


def cmd_binding(args) -> int:
    counts = {}
    mismatch = 0
    anchor = _anchor_registry()["anchors"].get(args.p_anchor) if args.p_anchor else None
    for side in ("O", "P", "H"):
        side_dir = args.h5_root / (anchor["segments"]["native"]["dir"] if side == "P" and anchor else f"{side}-native")
        lines = side_lines(side_dir)
        bad = sum(not _binding_ok(side, line, anchor, "native") for line in lines)
        mismatch += bad
        counts[side] = "robomme_hard" if side == "H" or (side == "P" and anchor) else "robomme"
    status = "PASS" if mismatch == 0 else "FAIL"
    print(f"ENV_PACKAGE_BINDING={status} sides=3 O={counts['O']} P={counts['P']} H={counts['H']} mismatch={mismatch}")
    return 0 if status == "PASS" else 1


def official_recovery_mode(episode: int) -> str:
    """官方 ``_worker`` 的失败恢复模式按 episode 号定（≤2 z、≤5 xy、其余关；v7 方案第一部分 §2）。"""
    return "z" if episode <= 2 else ("xy" if episode <= 5 else "off")


def _all_tasks() -> tuple[str, ...]:
    """16 任务规范序：``hard_specs.ALL_TASKS``（按文件轻量加载，不导入 robomme_hard 包）。"""
    return tuple(hard_specs_light().ALL_TASKS)


def xhard0_records(src_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """从官方 test 元数据取每任务 ``difficulty=="hard"`` 的记录（只读 ``src/robomme/``），按任务规范序、原 episode 升序。"""
    meta_root = src_root / "src" / "robomme" / "env_metadata" / "test"
    rows, sources = [], {}
    for task in _all_tasks():
        path = meta_root / f"record_dataset_{task}_metadata.json"
        sources[str(path.relative_to(src_root))] = sha256_file(path)
        payload = json.loads(path.read_text())
        hard = sorted((r for r in payload["records"] if r.get("difficulty") == "hard"), key=lambda r: int(r["episode"]))
        for r in hard:
            if r.get("task", task) != task:
                raise ParityError(f"{path} 记录任务名不符：{r}")
            rows.append({"task": task, "episode": int(r["episode"]), "seed": int(r["seed"]), "difficulty": "hard",
                         "recovery_mode": official_recovery_mode(int(r["episode"]))})
    return rows, sources


def check_xhard0(rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None,
                 builder_rows: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """XHARD0_IDENTITY：16 任务 × 1 档 × 12 局；原 episode 恰为 3,7,…,47；seed 任务内唯一；
    与清单、（给了时）与 robomme_hard builder 的 xhard0 条目逐条相等。"""
    problems = []
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    if len(by_task) != 16:
        problems.append(f"任务数 {len(by_task)} ≠ 16")
    for task, items in by_task.items():
        if tuple(r["episode"] for r in items) != XHARD0_EPISODES:
            problems.append(f"{task} 原 episode 号 {[r['episode'] for r in items]}")
        if len({r["seed"] for r in items}) != len(items):
            problems.append(f"{task} seed 重复")
    want = {(r["task"], r["episode"], r["seed"]) for r in rows}
    missing = extra = 0
    for label, other in (("清单", manifest["rows"] if manifest else None), ("builder", builder_rows)):
        if other is None:
            continue
        got = {(r["task"], int(r["episode"]), int(r["seed"])) for r in other}
        missing, extra = missing + len(want - got), extra + len(got - want)
        if want != got:
            problems.append(f"{label}与元数据不符：缺 {len(want - got)} 多 {len(got - want)}")
    ok = not problems and len(rows) == 16 * XHARD0_PER_TASK
    line = (f"XHARD0_IDENTITY={'PASS' if ok else 'FAIL'} shape=16x1x12 identities={len(rows)} "
            f"missing={missing} extra={extra}" + ("" if ok else f" problems={problems[:3]}"))
    return ok, line


def _builder_xhard0_rows() -> list[dict[str, Any]]:
    """在子进程里读 robomme_hard builder（``dataset="hard-verify"``，局 0～11 即 xhard0）的条目（本进程不导入
    robomme_hard，R9 精神）。"""
    code = ("import json;from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder as B;"
            f"tasks={list(_all_tasks())!r};out=[]\n"
            "for t in tasks:\n"
            " b=B(t,dataset='hard-verify')\n"
            " for ep in range(12):\n"
            "  i=b.resolve_identity(ep);assert i['tier']=='xhard0',i;assert b._hard_env_kwargs(ep)=={'seed':i['seed'],'difficulty':'hard'}\n"
            "  out.append({'task':t,'episode':i['source_episode'],'seed':i['seed']})\n"
            "print(json.dumps(out))")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise ParityError("读 builder xhard0 条目失败：\n" + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def cmd_export_xhard0_manifest(args) -> int:
    rows, sources = xhard0_records(args.src_root if args.src_root is not None else _common.bench_root())
    manifest = {
        "schema": "train-parity-manifest/1", "kind": "xhard0",
        "source_repo": "https://github.com/RoboMME/robomme_benchmark.git",
        "source_ref": "1fadc0ec", "source_dataset": "test",
        "selection_rule": "每任务官方 test 元数据 difficulty==\"hard\" 的全部记录，原 episode 升序（seed 照抄元数据，不套公式）",
        "per_cell": XHARD0_PER_TASK, "source_files": sources,
        "records_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "tasks": list(dict.fromkeys(r["task"] for r in rows)),
        "rows_total": len(rows),
        "recovery_config_counts": {m: sum(r["recovery_mode"] == m for r in rows) for m in ("z", "xy", "off")},
        "rows": rows,
    }
    ok, line = check_xhard0(rows)
    if ok:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
        ok, line = check_xhard0(rows, json.loads(args.out.read_text()), _builder_xhard0_rows())
    print(line, flush=True)
    return 0 if ok else 1


# ── smoke：拆仓验收的生成链路对拍（PARITY_GEN_SMOKE）────────────────────────────

#: 生成次数硬上限：1 任务 × 2 数据集 × 2 局 × 2 侧 = 8（项目 P3：超过即须另行授权，本子命令一律拒跑）
SMOKE_MAX_GENERATIONS = 8
#: 两个数据集各自的两侧：hard-verify 用官方 ``_worker``（O）对本仓子模块（H）；ood 用拆分前旧代码树（H_old）对本仓子模块（H_new）
SMOKE_SIDES = {"hard-verify": ("O", "H"), "ood": ("H_old", "H_new")}
#: H_old 侧解释器：拆分前旧仓的 venv（旧侧本来就是旧代码，用它自己的依赖）
OLD_PYTHON = Path("/data/hongzefu/robomme_benchmark_MotionJEPANewTask/.venv/bin/python")

_SMOKE_RESOLVE = r'''
import json, sys
dataset, task, eps = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
import robomme_hard
from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder
builder = BenchmarkEnvBuilder(task, dataset=dataset)
rows = []
for ep in eps:
    identity = dict(builder.resolve_identity(ep))
    identity["builder_episode"] = ep
    identity["task"] = task
    rows.append(identity)
print("RESOLVE_JSON " + json.dumps({"robomme_hard_file": robomme_hard.__file__, "episodes": builder.get_episode_num(),
                                    "rows": rows}, default=str))
'''


def parse_episode_range(text: str) -> list[int]:
    """``a:b`` → builder 局号半开区间 ``[a, b)``；须 ``0 ≤ a < b``。"""
    head, sep, tail = str(text).partition(":")
    try:
        lo, hi = int(head), int(tail)
    except ValueError as exc:
        raise ParityError(f"局号区间须写成 a:b（半开区间）：{text!r}") from exc
    if not sep or lo < 0 or hi <= lo:
        raise ParityError(f"局号区间须为 0 ≤ a < b：{text!r}")
    return list(range(lo, hi))


def smoke_generations(tasks: list[str], episodes: dict[str, list[int]], sides: dict[str, tuple[str, ...]]) -> int:
    """smoke 的生成次数 = 任务数 × Σ（各数据集局数 × 该数据集侧数）。"""
    return len(tasks) * sum(len(episodes[ds]) * len(sides[ds]) for ds in episodes)


def _parse_sides(text: str, dataset: str) -> tuple[str, ...]:
    sides = tuple(s.strip() for s in str(text).split(",") if s.strip())
    if sides != SMOKE_SIDES[dataset]:
        raise ParityError(f"--{dataset}-sides 只接受 {','.join(SMOKE_SIDES[dataset])}（收到 {text!r}）")
    return sides


def resolve_builder_identities(python: str | Path, src: Path, dataset: str, task: str,
                               episodes: list[int]) -> list[dict[str, Any]]:
    """在独立进程里用 ``src`` 下的 ``robomme_hard`` 构建器把 builder 局号解析成身份（只构建 builder，不建环境、不 reset）；
    断言该进程导入的 ``robomme_hard`` 位于 ``src`` 下。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    proc = subprocess.run([str(python), "-c", _SMOKE_RESOLVE, dataset, task, json.dumps(episodes)],
                          capture_output=True, text=True, env=env, cwd=str(src))
    line = next((t for t in reversed(proc.stdout.splitlines()) if t.startswith("RESOLVE_JSON ")), None)
    if proc.returncode != 0 or line is None:
        raise ParityError(f"{dataset}/{task} builder 解析失败（{python}）：{proc.stderr[-1500:]}")
    payload = json.loads(line[len("RESOLVE_JSON "):])
    where = Path(payload["robomme_hard_file"]).resolve()
    if not str(where).startswith(str(Path(src).resolve()) + os.sep):
        raise ParityError(f"{dataset} 解析进程导入的 robomme_hard 不在 {src} 下：{where}")
    return payload["rows"]


def _round_modules(out: Path) -> set[str]:
    """generate_h5 replay 各轮 runner 探针记下的 ``robomme`` 模块路径（``_rounds/*/results.json``）。"""
    found = set()
    for path in out.glob("_rounds/*/results.json"):
        module = json.loads(path.read_text()).get("robomme_module")
        if module:
            found.add(str(module))
    return found


def smoke_compare(dataset: str, sides: tuple[str, str], want: list[dict[str, Any]], dirs: dict[str, Path],
                  tol: dict[str, float], module_roots: dict[str, Path] | None = None) -> dict[str, Any]:
    """一个数据集的两侧逐身份比对（键 (task, seed)）。

    * 两侧都无 h5（同失败）→ ``both_fail``，单列、不计入通过；
    * 只有一侧有 h5 → ``one_side``；两侧都有但读不出（``pair_metrics`` 报错）→ ``unreadable``；
    * 两侧可读 → ``compared``；setup／结构／帧连续不等计 ``struct_bad``；动作／状态／图像／帧数任一项超
      ``hard-parity-tolerances.json`` 计 ``tol_over``（4 局的 smoke 上 5% 硬线即任一超限）；
    * 绑定：O 侧须为官方 ``_worker``、robomme 不来自 robomme_hard；其余侧环境类模块须属于 robomme_hard；
      ``module_roots`` 给出的侧另核 runner 探针记下的 robomme 路径在该源码树下。"""
    lines = {side: {(l["task"], int(l["seed"])): l for l in side_lines(dirs[side])} for side in sides}
    counts = {k: 0 for k in ("identities", "compared", "both_fail", "one_side", "unreadable", "struct_bad",
                             "tol_over", "binding_bad", "missing")}
    records = []
    for row in want:
        key = (row["task"], int(row["seed"]))
        counts["identities"] += 1
        a, b = (lines[side].get(key) for side in sides)
        record: dict[str, Any] = {"dataset": dataset, "task": key[0], "seed": key[1],
                                  "builder_episode": row["builder_episode"], "tier": row.get("tier"),
                                  "left": a, "right": b}
        if a is None or b is None:
            counts["missing"] += 1
            record["state"] = "missing"
            records.append(record)
            continue
        for side, line in zip(sides, (a, b)):
            check = "O" if side == "O" else "H"
            if not _binding_ok(check, line):
                counts["binding_bad"] += 1
        ok_a, ok_b = bool(a.get("success") and a.get("path")), bool(b.get("success") and b.get("path"))
        if not ok_a and not ok_b:
            counts["both_fail"] += 1
            record["state"] = "both_fail"
        elif ok_a != ok_b:
            counts["one_side"] += 1
            record["state"] = "one_side"
        else:
            m = pair_metrics(str(dirs[sides[0]] / a["path"]), str(dirs[sides[1]] / b["path"]))
            record.update({k: v for k, v in m.items() if k not in ("left", "right")})
            if m.get("error") is not None:
                counts["unreadable"] += 1
                record["state"] = "unreadable"
            else:
                counts["compared"] += 1
                record["state"] = "compared"
                if not (m.get("setup_equal") and m.get("schema_equal") and m.get("contiguous")):
                    counts["struct_bad"] += 1
                over = {k: m[f] for k, f in (("action_max", "action_max"), ("state_max", "state_max"),
                                              ("image_mad", "image_mad"), ("frames_max", "frames_diff")) if m[f] > tol[k]}
                if over:
                    counts["tol_over"] += 1
                    record["tol_over"] = over
        records.append(record)
    for side, root in (module_roots or {}).items():
        modules = _round_modules(dirs[side])
        prefix = str(Path(root).resolve()) + os.sep
        if not modules or any(not str(Path(m).resolve()).startswith(prefix) for m in modules):
            counts["binding_bad"] += 1
    return {"dataset": dataset, "sides": list(sides), "counts": counts, "records": records}


def cmd_smoke(args) -> int:
    """PARITY_GEN_SMOKE（拆仓验收第二行，行为一致层级）：1 任务 × {hard-verify, ood} × 2 局 × 2 侧 = 8 次生成。

    * hard-verify 局（builder 局号，局 0、1 对应官方原 episode 3、7）：O＝官方 ``_worker``（vendor 编排 + 子模块
      ``robomme``，``ROBOMME_ENV_PACKAGE=robomme``）对 H＝镜像 worker + 子模块 ``robomme_hard``（gym.make 实参取自
      builder ``dataset="hard-verify"``）；两侧都经 ``train_split_runner.py``，身份与官方 test 元数据 hard 子集双向核对；
    * ood 局：H_old＝``--h-old-src`` 指向的拆分前旧代码树（旧仓 venv 解释器，``PYTHONPATH`` 指向它，进程内断言
      ``robomme_hard.__file__`` 在其下）对 H_new＝本仓子模块；两侧都经 ``generate_h5.py --mode replay`` 按各自包内规格重放；
    * 身份一律以 builder 的解析为准（独立进程构建 builder，不建环境）；ood 两侧解析出的身份（档、seed、spec_sha256）
      须逐局相同，否则不生成；
    * 之后按 ``configs/hard-parity-tolerances.json`` 比。PASS 须：身份数 = 期望、两侧都有可读 h5 且实际比较数 =
      身份数、``both_fail``（单列，不计入通过）= 0、超容差 = 0，结构与包绑定全对。"""
    task = args.task
    hs = hard_specs_light()
    if task not in hs.ALL_TASKS:
        raise ParityError(f"未知任务 {task!r}")
    episodes = {"hard-verify": parse_episode_range(args.hard_verify_episodes),
                "ood": parse_episode_range(args.ood_episodes)}
    sides = {"hard-verify": _parse_sides(args.hard_verify_sides, "hard-verify"),
             "ood": _parse_sides(args.ood_sides, "ood")}
    planned = smoke_generations([task], episodes, sides)
    if planned > SMOKE_MAX_GENERATIONS:
        raise ParityError(f"smoke 生成次数 {planned} 超过硬上限 {SMOKE_MAX_GENERATIONS}"
                          "（1 任务 × 2 数据集 × 2 局 × 2 侧）；更大规模须另行授权，不经本子命令")
    h_old_src = Path(args.h_old_src).resolve()
    if not (h_old_src / "robomme_hard" / "__init__.py").is_file():
        raise ParityError(f"--h-old-src 须为含 robomme_hard 的 src 目录：{h_old_src}")
    old_python = Path(args.h_old_python)
    if not old_python.is_file():
        raise ParityError(f"H_old 侧解释器不存在：{old_python}")
    tol = load_tolerances()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ParityError(f"{out} 已存在且非空；smoke 不续跑，另起目录")
    bench = _common.bench_root()
    new_src = bench / "src"

    # ① 身份：以 builder 的解析为准
    hv_rows = resolve_builder_identities(sys.executable, new_src, "hard-verify", task, episodes["hard-verify"])
    ood_rows = resolve_builder_identities(sys.executable, new_src, "ood", task, episodes["ood"])
    old_rows = resolve_builder_identities(old_python, h_old_src, "ood", task, episodes["ood"])
    keys = ("task", "tier", "seed", "spec_sha256")
    if [{k: r.get(k) for k in keys} for r in ood_rows] != [{k: r.get(k) for k in keys} for r in old_rows]:
        raise ParityError(f"ood 局 {args.ood_episodes} 新旧两棵树解析出的身份不同：新 {ood_rows}，旧 {old_rows}")
    if any(r.get("tier") != "xhard0" or r.get("source_episode") is None for r in hv_rows):
        raise ParityError(f"hard-verify 局应全为 xhard0 且带官方原 episode：{hv_rows}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "identities.json").write_text(json.dumps({"hard-verify": hv_rows, "ood": ood_rows, "ood_old": old_rows},
                                                    ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"SMOKE_IDENTITIES task={task} hard_verify={[(r['builder_episode'], r['source_episode'], r['seed']) for r in hv_rows]} "
          f"ood={[(r['builder_episode'], r['tier'], r['seed']) for r in ood_rows]} generations={planned}", flush=True)

    # ② 生成：hard-verify 两侧经 generate（tier xhard0，子集身份），ood 两侧经 generate_h5 replay
    dirs: dict[str, dict[str, Path]] = {"hard-verify": {}, "ood": {}}
    gen_rc: dict[str, int] = {}
    subset = out / "hard-verify-identities.jsonl"
    subset.write_text("".join(json.dumps({"task": r["task"], "tier": "xhard0", "seed": int(r["seed"])}) + "\n"
                              for r in hv_rows))
    for side in sides["hard-verify"]:
        side_out = out / "hard-verify" / side
        dirs["hard-verify"][side] = side_out
        gen_args = argparse.Namespace(side=side, tier="xhard0", manifest=XHARD0_MANIFEST, src_root=bench,
                                      workers=args.workers, gpu=args.gpu, out=side_out, stage=None, smoke=0,
                                      dev_smoke=True, specs_root=None, identities=subset, resume=False,
                                      expect_ref=None)
        gen_rc[f"hard-verify/{side}"] = cmd_generate(gen_args)
    ood_ids = out / "ood-identities.jsonl"
    ood_ids.write_text("".join(json.dumps({"task": r["task"], "tier": r["tier"], "seed": int(r["seed"])}) + "\n"
                               for r in ood_rows))
    roots = {"H_old": (old_python, h_old_src), "H_new": (Path(sys.executable), new_src)}
    for side in sides["ood"]:
        python, src = roots[side]
        side_out = out / "ood" / side
        side_out.mkdir(parents=True)
        dirs["ood"][side] = side_out
        command = [str(python), str(GENERATE_H5), "--mode", "replay", "--identities", str(ood_ids),
                   "--output", str(side_out), "--workers", str(args.workers), "--gpu", str(args.gpu),
                   "--pkg", "robomme_hard", "--src-root", str(src.parent), "--expect-src", str(src), "--dev-smoke"]
        env = dict(os.environ)
        env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(src))
        print(f"GENERATE_START side={side} dataset=ood rows={len(ood_rows)} python={python} src={src}", flush=True)
        proc, _lines, mover = run_with_mover(command, env, side_out, side, "ood",
                                             {"worker": "train_split_worker.run_one"})
        gen_rc[f"ood/{side}"] = 0 if proc.returncode == 0 and not mover.errors else 1
        print(f"SMOKE_SIDE dataset=ood side={side} exit={proc.returncode} mover_errors={len(mover.errors)}", flush=True)

    # ③ 比对
    results = [smoke_compare("hard-verify", sides["hard-verify"], hv_rows, dirs["hard-verify"], tol),
               smoke_compare("ood", sides["ood"], ood_rows, dirs["ood"], tol,
                             module_roots={"H_old": h_old_src, "H_new": new_src})]
    total = {k: sum(r["counts"][k] for r in results) for k in results[0]["counts"]}
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            for record in result["records"]:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    expected = len(hv_rows) + len(ood_rows)
    ok = (total["identities"] == expected and total["compared"] == expected and total["compared"] > 0
          and total["both_fail"] == 0 and total["tol_over"] == 0 and total["struct_bad"] == 0
          and total["binding_bad"] == 0 and total["one_side"] == 0 and total["unreadable"] == 0
          and total["missing"] == 0)
    for result in results:
        c = result["counts"]
        print(f"PARITY_GEN_SMOKE_DATASET=INFO dataset={result['dataset']} pair={':'.join(result['sides'])} "
              + " ".join(f"{k}={v}" for k, v in c.items()), flush=True)
    print(f"PARITY_GEN_SMOKE_DETAIL=INFO generations={planned} side_exit={gen_rc} struct_bad={total['struct_bad']} "
          f"binding_bad={total['binding_bad']} one_side={total['one_side']} unreadable={total['unreadable']} "
          f"missing={total['missing']}", flush=True)
    (out / "summary.json").write_text(json.dumps({"ok": ok, "task": task, "episodes": episodes, "generations": planned,
                                                  "side_exit": gen_rc, "totals": total, "tolerances": tol,
                                                  "datasets": [{k: v for k, v in r.items() if k != "records"}
                                                               for r in results]},
                                                 ensure_ascii=False, indent=2) + "\n")
    print(f"PARITY_GEN_SMOKE={'PASS' if ok else 'FAIL'} identities={total['identities']} sides=2 "
          f"compared={total['compared']} both_fail={total['both_fail']} tol_over={total['tol_over']}", flush=True)
    return 0 if ok else 1


#: --identities 的帮助文字（generate／compare／import-delivery 共用）
IDENTITIES_HELP = ("身份子集（只取这些身份）：jsonl（每行 task、tier、seed）或交付清单 json；行带 source 字段"
                   "（v9 assemble 的 800 行清单）时只取 source=v9-new 的行（v9 新 80 局）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--side", choices=SIDES, required=True)
    gen.add_argument("--tier", choices=TIERS, required=True)
    gen.add_argument("--manifest", type=Path, required=True)
    gen.add_argument("--src-root", type=Path, required=True)
    gen.add_argument("--workers", type=int, default=16)
    gen.add_argument("--gpu", default="0")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--stage", type=Path, default=None, help="NFS 暂存目录；每局 sha 后复制过去并删本地大文件")
    gen.add_argument("--smoke", type=int, default=0, help="只跑前 N 局（SHARD_SMOKE）")
    gen.add_argument("--dev-smoke", action="store_true", help="放行非 A40（本机开发冒烟 NATIVE_SMOKE）")
    gen.add_argument("--specs-root", type=Path, default=None,
                     help="--tier v9：五档 /4 规格根（含 xhard{1..5}/specs.jsonl，如 artifacts/newtask-v9/specs-root）")
    gen.add_argument("--identities", type=Path, default=None,
                     help="--tier v9：只重放该" + IDENTITIES_HELP + "；子集须 ⊆ --manifest。--tier xhard0："
                          "身份 tier 须为 xhard0，只生成子集，运行器仍拿完整 --manifest 核对")
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--expect-ref", type=Path, default=None,
                     help="噪声基线参照文件（noise_gate.py gen-regress build-ref 产物，如 dev-scripts/parity/configs/noise-ref-20261003.json）："
                          "每局算出 sha 后当场判 match／jitter_info／flip 写进 identities 的 verdict；match 删本地大文件不复制，"
                          "flip 照常复制并追加 flips.jsonl、打印 EPISODE_FLIP。不给时行为不变")
    gen.set_defaults(func=cmd_generate)
    pub = sub.add_parser("publish")
    pub.add_argument("--side", choices=SIDES, required=True)
    pub.add_argument("--tier", choices=TIERS, required=True)
    pub.add_argument("--prefix", required=True, help="bucket 侧目录，如 O-1fadc0e-a40、H-<sha7>-a40")
    pub.add_argument("--local", type=Path, default=None)
    pub.add_argument("--h5-root", type=Path, default=LOCAL_H5_ROOT)
    pub.add_argument("--code-baseline", default=None)
    pub.add_argument("--gpu-model", default=None)
    pub.add_argument("--driver", default=None)
    pub.add_argument("--workers", type=int, default=16)
    pub.add_argument("--note", default=None)
    pub.add_argument("--readback-tmp", default=str(ROOT / "artifacts" / "hard-split"))
    pub.add_argument("--resume-upload", action="store_true", help="目录已存在时只补缺对象（上传中断后续传）")
    pub.add_argument("--dry-run", action="store_true")
    pub.set_defaults(func=cmd_publish)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--pair", choices=PAIRS, required=True)
    cmp_.add_argument("--tier", choices=TIERS, required=True)
    cmp_.add_argument("--manifest", type=Path, required=True,
                      help="身份清单；--tier v9 时为交付清单（v8-delivery/1 格式，assemble 的 800 行清单）")
    cmp_.add_argument("--specs-root", type=Path, default=None,
                      help="--tier v9 必填：冻结 /4 规格根，其 delivered 行即冻结交付集（分母核对：冻结集 = delivery.json = H = H2）")
    cmp_.add_argument("--cells", default=None,
                      help="--tier v9：格表，缺省 v9full（V9_CELLS 43 格 800）；"
                           "另可 full（当前 EXPECTED_CELLS）｜v9smoke｜v9shard1｜JSON 文件或内联 JSON（须为 V9_CELLS 的子表）")
    cmp_.add_argument("--identities", type=Path, default=None,
                      help="--tier v9：只比" + IDENTITIES_HELP + "；分母与 expected 改为子集大小（判定行尾加 identities=）")
    cmp_.add_argument("--left", type=Path, default=None)
    cmp_.add_argument("--right", type=Path, default=None)
    cmp_.add_argument("--p-anchor", default=None, help="P 侧锚点 tag（从 docs/validation/parity-anchors.json 取目录并先核验）")
    cmp_.add_argument("--h5-root", type=Path, default=None,
                      help="两侧目录的上级（<h5-root>/<side>-<tier>）；缺省 v9 → artifacts/newtask-v9/parity/h5，其余 → newtask-v8")
    cmp_.add_argument("--compare-root", type=Path, default=None,
                      help="比对输出上级；缺省 v9 → artifacts/newtask-v9/parity/compare，其余 → newtask-v8")
    cmp_.add_argument("--run-name", default=None, help="比对目录已存在时另起子目录")
    cmp_.add_argument("--workers", type=int, default=16)
    cmp_.add_argument("--calibrate", action="store_true")
    cmp_.add_argument("--overwrite-compare", action="store_true")
    cmp_.set_defaults(func=cmd_compare)
    anc = sub.add_parser("anchor", help="parity 固定锚点：register 登记 P 缓存、check 输出 PARITY_ANCHOR（D-11～D-13）")
    anc.add_argument("action", choices=("register", "check"))
    anc.add_argument("--tag", required=True)
    anc.add_argument("--commit", default=None, help="register：tag 指向的完整 commit sha")
    anc.add_argument("--segments", default=None, help="register：档:目录名,…（目录相对 --h5-root），如 native:P-native")
    anc.add_argument("--source-side", default="H", help="register：产物当年作为哪一侧生成（parity-anchor-v6 两段都是 H）")
    anc.add_argument("--equivalence", default=None, help="register：等价核验判定行原文（tag commit ≠ 生成 commit 时）")
    anc.add_argument("--bucket-prefix", type=json.loads, default=None, help="register：{档: bucket 段名} JSON")
    anc.add_argument("--replace", action="store_true", help="仅用于修正登记错误；不得用来移动锚点")
    anc.add_argument("--h5-root", type=Path, default=LOCAL_H5_ROOT)
    anc.set_defaults(func=cmd_anchor)
    dlv = sub.add_parser("import-delivery", help="gen1 的 delivery.json 与逐局 h5 登记成 H 侧 identities.jsonl")
    dlv.add_argument("--delivery", type=Path, required=True)
    dlv.add_argument("--tier", default="v9", choices=DELIVERY_TIERS)
    dlv.add_argument("--identities", type=Path, default=None, help="只登记该" + IDENTITIES_HELP)
    dlv.add_argument("--h5-root", type=Path, default=None,
                     help="缺省 artifacts/newtask-v9/parity/h5（写 <h5-root>/H-<tier>）")
    dlv.set_defaults(func=cmd_import_delivery)
    bind = sub.add_parser("binding", help="ENV_PACKAGE_BINDING：三侧 native identities 的包归属")
    bind.add_argument("--h5-root", type=Path, default=LOCAL_H5_ROOT)
    bind.add_argument("--p-anchor", default=None)
    bind.set_defaults(func=cmd_binding)
    x0 = sub.add_parser("export-xhard0-manifest", help="从官方 test 元数据导出 xhard0 的 16×1×12 清单（XHARD0_IDENTITY）")
    x0.add_argument("--src-root", type=Path, default=None,
                    help="只读其 src/robomme/env_metadata/test/（缺省 benchmark 子模块根）")
    x0.add_argument("--out", type=Path, default=XHARD0_MANIFEST)
    x0.set_defaults(func=cmd_export_xhard0_manifest)
    sm = sub.add_parser("smoke", help="拆仓验收：1 任务 × {hard-verify, ood} × 2 局 × 2 侧 = 8 次生成后按容差比（PARITY_GEN_SMOKE）",
                        description=cmd_smoke.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sm.add_argument("--task", required=True, help="任务名（两个数据集都有的任务，如 VideoUnmask）")
    sm.add_argument("--hard-verify-episodes", default="0:2",
                    help="hard-verify 的 builder 局号半开区间 a:b（局 0、1 对应官方原 episode 3、7，以 builder 解析为准）")
    sm.add_argument("--ood-episodes", default="0:2", help="ood 的 builder 局号半开区间 a:b")
    sm.add_argument("--hard-verify-sides", default="O,H", help="hard-verify 两侧（固定 O,H：官方 _worker 对本仓子模块）")
    sm.add_argument("--ood-sides", default="H_old,H_new", help="ood 两侧（固定 H_old,H_new：拆分前旧代码树对本仓子模块）")
    sm.add_argument("--h-old-src", type=Path, required=True,
                    help="拆分前旧仓 SRC 只读快照的 src 目录（git worktree add --detach <快照> <SRC> 后的 <快照>/src）")
    sm.add_argument("--h-old-python", type=Path, default=OLD_PYTHON, help="H_old 侧解释器（缺省旧仓 .venv）")
    sm.add_argument("--workers", type=int, default=2)
    sm.add_argument("--gpu", default="0")
    sm.add_argument("--out", type=Path, required=True, help="输出目录（须不存在或为空）")
    sm.set_defaults(func=cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("manifest", "src_root", "out", "stage", "local", "left", "right", "h5_root", "compare_root",
                 "specs_root", "delivery", "h_old_src"):
        value = getattr(args, name, None)
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, name, (Path.cwd() / value).resolve())
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
