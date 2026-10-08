#!/usr/bin/env python3
"""逐局站点目录（v8 方案第一部分 §2.2 站点行、第二部分 §2.2 第 9 条、§2.8 第 16 条；V9 复用）。

目录形态 ``tasks[].tiers[tier].episodes[]``，页面 ``site.html``。维护计划 W2 起删除 V8 单次评估运行分支
（``--eval-run``／``--xhard0-eval``）：评估位只走 V9 复用口径（``--eval-reuse``／``--reused``／``--eval-new``），不给评估
来源时全部记「未评估」。数据来源：

- **新值局**（xhard1～5，43 格 1070 局）：身份与配置取 v8 规格根（``<specs-root>/<tier>/specs.jsonl``，
  ``hard-specs/4``，经 ``hard_specs.load_specs_root`` 校验），h5 与执行步取生成产物 ``delivery.json``
  （schema ``v8-delivery/1``）。生成视频在 h5 所在 episode 目录的 ``videos/`` 下（与 v7 gen1 同一约定），
  或在 ``--gen-videos`` 给出的目录下按 ``<tier>/<task>_episode_<episode>/videos`` 查找；
  取文件名不以 ``FAILED``／``success_NO_OBJECT`` 开头、且含 ``_seed<seed>_`` 的那一个 mp4，必须恰好 1 个。
  ``delivery.json`` 口径（S2-B）：顶层 ``schema``、``rows``、``counts``、``cells``；每行 ``task``、``tier``、``episode``、
  ``candidate``、``seed``、``exec_steps``、``frames``、``h5``（绝对路径，优先）、``path``（相对 delivery.json 目录）、
  ``h5_sha256``、``env_module``、可选 ``video``（绝对路径）。缺 ``exec_steps``／``frames`` 即 FAIL。
- **xhard0**（16 任务 × 12 局）：复用 v7 已渲染的 ``site-media/xhard0-gen``（``manifest-{H,O}.jsonl``，0 次渲染），
  新入口 H／旧入口 O 两行，与 v7 相同。
- **配置**：每局从规格行 ``spec.objects``／``spec.actions`` 按 ``DIMS`` 抽取表 1 的维度值（不写死数值），
  逐格汇总取值集合；``TABLE1`` 是表 1 的参照值，只用来核对（``config_mismatch``），不进页面。
- **身份清单**：``eval-identities-<n>.jsonl``（S2-B ``export_eval_identities.py`` 产出），总数一律由格表推出
  （``sum(格表) + 16 × 12``：V8 ``1070 + 192 = 1262``、V9 ``800 + 192 = 992``），非 xhard0 身份必须与交付集逐一相等。
- **格表**（v9 方案 §2.1 S1-E）：``--cells v8|v9`` 显式选择完整交付格表（缺省 ``v8``，即 V8 行为不变；不读
  ``EXPECTED_CELLS``，所以阶段 3b 切换前后结果相同）；``--cells-json`` 给子表时优先于 ``--cells``。``--cells v9`` 且未显式
  给 ``--specs-root``／``--delivery``／``--identities`` 时取 V9 缺省路径（``V9_DEFAULT_SOURCES``）。
- **V9 评估复用**（v9 方案第一部分 §1 第 7 条、第二部分 §2.4.2 第 8 步、§2.5 R-3）：``--eval-reuse <V8 site-eval 目录>``
  ``--reused <reused.json>`` ``[--eval-new <V9 评估运行目录>]`` 三者一起用（评估侧策略 ``smvla``／``perceptual-framesamp-modul`` → 页面
  策略 ID ``simplememvla``／FrameSamp+Modulation 的历史数据键，见 ``EVAL_POLICY``）：
  - 复用集合**只认** ``reused.json``（S1-F ``eval_manifest.py --exclude-evaluated`` 产出，schema ``v9-eval-reused/1``）；
    V8 ``site-eval/catalog.json`` 没有 ``spec_sha256``，不单独当复用依据。逐行核：``reused.json`` 的 ``v8_manifest``
    文件 sha256 等于 ``v8_manifest_sha256``，该行 ``v8_key`` 在 V8 manifest 里存在且四元组 (task, tier, seed,
    ``spec_sha256``) 逐键相等，再与本次规格行的四元组逐键相等；然后用 ``v8_key`` 在 V8 site-eval 里定位记录（其
    (task, tier, seed) 也必须相等），两模型评估位（结局、步数、上限、视频）原样复制，视频按 V8 ``media-private.json``
    取绝对路径重新登记。任一处对不上 → 该局评估置空并计数（``eval_empty``、``reuse_sha_mismatch``、
    ``reuse_identity_mismatch``、``reuse_missing``），不回退到 (task, tier, seed) 猜。
  - 不在复用集合里的新值局取 ``--eval-new``：目录结构与 V8 评估运行相同（``load_eval_run`` 同一套账本 ``accept`` 口径、
    ``site-media/manifest.jsonl``、``report/report.json``），结果行的 ``spec_sha256`` 必须等于规格行，否则置空
    （``new_sha_mismatch``）；逐格「新评部分」成败数与新评 ``report.json`` 的 ``per_policy.<p>.cells`` 核对。
  - xhard0 评估（V8 阶段 3′ 两路线）按 (task, seed) 原样取自 V8 site-eval（规格、视频都没变，``eval_x0_reused``）。
  - 有置空局（``eval_empty > 0``）时判定 FAIL、不写产物（``--allow-eval-empty`` 时照写，供排查）。
  - 每局写 ``eval_origin``（``reused``／``new``／``empty``），目录 ``eval.mode = "v9-reuse"``、``eval.reuse`` 记三类计数，
    供浏览器检查器出 ``V9_SITE`` 行；页面模板与 V8 完全相同（不改 ``site.html``／``site_app.py``）。

输出 ``catalog.json``（schema ``v8-site-catalog/1``）与 ``media-private.json``（媒体 ID → 绝对路径白名单），
都以 ``open("x")`` 写入、拒绝覆盖。末行打印
``V8_SITE_CATALOG=PASS|FAIL identities=<n> expected=<n> gen_v8=<n> gen_xhard0_new=<n> gen_xhard0_old=<n>
gen_failed=<n> eval_filled=<n> eval_media=<n> eval_unevaluated=<n> flip=<n> rate_mismatch=<n> config_mismatch=0
media=<n> problems=<n>``；V9 复用模式另在行尾追加 ``eval_reused=<n> eval_new=<n> eval_empty=<n> eval_x0_reused=<n>
reuse_sha_mismatch=<n> reuse_identity_mismatch=<n> reuse_missing=<n> new_sha_mismatch=<n>``。

    # V9（阶段 4c）：800 + 192 = 992 局，720 复用 V8 评估 + 80 新评
    uv run --no-sync python dev-scripts/site/site_catalog.py --cells v9 \\
      --specs-root artifacts/newtask-v9/specs-root --delivery artifacts/newtask-v9/delivery/delivery.local.json \\
      --identities artifacts/v9-evaluation/inputs/eval-identities-992.jsonl \\
      --xhard0-gen artifacts/newtask-v7/site-media/xhard0-gen \\
      --eval-reuse artifacts/newtask-v8/site-eval --reused artifacts/v9-evaluation/<run_name>/manifest/reused.json \\
      --eval-new artifacts/v9-evaluation/<run_name> --out artifacts/newtask-v9/site

本模块只用标准库（``DIMS``／``TABLE1``／路径解析供 ``subgoal_lengths.py`` 与浏览器检查器复用）；
``hard_specs`` 按文件路径加载（它只依赖标准库），不导入 ``robomme_hard`` 包。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # dev-scripts/site/<本文件> → 仓根
ART8 = REPO_ROOT / "artifacts/newtask-v8"
#: benchmark 子模块的源码根（``robomme``／``robomme_hard`` 两个包）
BENCHMARK_SRC = REPO_ROOT / "third_party/robomme_benchmark/src"
#: 旧名别名表（原评估目录的 ``official_defs.py``，现为评估包内模块）
OFFICIAL_DEFS_PATH = REPO_ROOT / "src/robomme_hard_eval/models/_official_defs.py"
ART7 = REPO_ROOT / "artifacts/newtask-v7"
TIERS = ("xhard0", "xhard1", "xhard2", "xhard3", "xhard4", "xhard5")
NEW_TIERS = TIERS[1:]
#: 页面策略 ID 与展示名；FrameSamp+Modulation 的页面 ID 是已发布 catalog.json 里的历史数据键，为读 V8 站点目录原样保留
POLICIES = (("simplememvla", "SimpleMemVLA"), ("mmevla", "FrameSamp+Modulation"))  # 历史数据键（页面 ID）
#: 评估侧策略名（官方名）→ 页面策略 ID
EVAL_POLICY = {"smvla": "simplememvla", "perceptual-framesamp-modul": "mmevla"}  # 历史数据键（页面 ID）


def official_defs():
    """``src/robomme_hard_eval/models/_official_defs.py``（旧名别名表的唯一来源；已加载则复用同一模块）。"""
    mod = sys.modules.get("official_defs")
    if mod is None:
        spec = importlib.util.spec_from_file_location("official_defs", OFFICIAL_DEFS_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["official_defs"] = mod
        spec.loader.exec_module(mod)
    return mod
FINAL = ("success", "fail", "timeout")
XHARD0_PER_TASK = 12
DELIVERY_SCHEMA = "v8-delivery/1"
NAMES = {
    "BinFill": "分类投放", "PickXtimes": "重复抓取", "SwingXtimes": "重复摆动",
    "PickHighlight": "高亮目标抓取", "VideoUnmask": "视频遮挡记忆", "ButtonUnmask": "按钮遮挡记忆",
    "VideoUnmaskSwap": "视频遮挡交换", "ButtonUnmaskSwap": "按钮遮挡交换", "VideoRepick": "视频重复抓取",
    "PatternLock": "图案解锁", "RouteStick": "路线引导", "VideoPlaceButton": "视频放置与按钮",
    "VideoPlaceOrder": "视频放置顺序", "MoveCube": "移动方块", "InsertPeg": "插入插销", "StopCube": "停止方块",
}
DEFAULT_SOURCES = {
    "specs_root": ART8 / "specs-root",
    "delivery": ART8 / "gen1/delivery.json",
    "identities": ART8 / "eval-identities-1262.jsonl",
    "xhard0_gen": ART7 / "site-media/xhard0-gen",
    "gen_videos": None,
    "path_base": REPO_ROOT,
    "cells_json": None,
    "eval_reuse": None,    # V9：V8 站点目录 artifacts/newtask-v8/site-eval（catalog.json + media-private.json）
    "reused": None,        # V9：S1-F 产出的 reused.json（v9-eval-reused/1），复用集合的唯一依据
    "eval_new": None,      # V9：新 80 局的评估运行目录（结构同 V8 评估运行）
}
#: ``--cells v9`` 时未显式给出的输入取这些缺省（v9 方案 §2.2 第 5、7 条，§2.4.2 第 7 步）
V9_DEFAULT_SOURCES = {
    "specs_root": REPO_ROOT / "artifacts/newtask-v9/specs-root",
    "delivery": REPO_ROOT / "artifacts/newtask-v9/delivery/delivery.local.json",
    "identities": REPO_ROOT / "artifacts/v9-evaluation/inputs/eval-identities-992.jsonl",
}
#: V8 的 1070 局表（``v8``）已于维护计划阶段 1b（W4）随 V8 1070 局表删除，只剩 v9
CELL_VERSIONS = ("v9",)
REUSED_SCHEMA = "v9-eval-reused/1"
SITE_CATALOG_SCHEMA = "v8-site-catalog/1"
#: V9 复用模式的逐局计数键（判定行尾追加，零值也写）
V9_COUNT_KEYS = ("eval_reused", "eval_new", "eval_empty", "eval_x0_reused", "reuse_sha_mismatch",
                 "reuse_identity_mismatch", "reuse_missing", "new_sha_mismatch")


# ── 表 1 维度：从规格行抽值（不写死数值）──────────────────────────────────

def _objects(spec: dict) -> dict:
    return spec.get("objects") or {}


def _actions(spec: dict) -> dict:
    return spec.get("actions") or {}


def _swap_count(spec: dict):
    obj = _objects(spec)
    if obj.get("n_swaps") is not None:
        return obj["n_swaps"]
    pairs = _actions(spec).get("swap_pairs")
    return len(pairs) if pairs is not None else None


#: 任务 → [(维度名, 从 spec 取值的函数)]；维度名与表 1 的「维度」列一一对应（复合列拆成多行）。
DIMS = {
    "PickXtimes": [("抓放次数", lambda s: _objects(s).get("num_repeats")),
                   ("干扰块", lambda s: (_objects(s).get("distractor_count") or {}).get("actual"))],
    "SwingXtimes": [("摆动轮数", lambda s: _objects(s).get("num_repeats")),
                    ("干扰块", lambda s: (_objects(s).get("distractor_count") or {}).get("actual"))],
    "StopCube": [("停止序号 stop_time", lambda s: _actions(s).get("stop_time")),
                 ("方块速度 move_interval", lambda s: _actions(s).get("move_interval"))],
    "BinFill": [("投入块数", lambda s: sum(_objects(s)["target_numbers"]) if _objects(s).get("target_numbers") else None)],
    "VideoUnmask": [("抓取数", lambda s: _objects(s).get("n_picks")),
                    ("干扰容器", lambda s: (_objects(s).get("distractors") or {}).get("placed")),
                    ("干扰方块", lambda s: (_objects(s).get("distractors") or {}).get("cube_count"))],
    "VideoUnmaskSwap": [("换位次数", _swap_count), ("抓取数", lambda s: _objects(s).get("n_picks")),
                        ("外圈干扰", lambda s: (_objects(s).get("distractors") or {}).get("placed"))],
    "VideoPlaceButton": [("放置次数", lambda s: _actions(s).get("target_placement_count"))],
    "VideoPlaceOrder": [("访问总次数", lambda s: _actions(s).get("target_placement_count"))],
    "PickHighlight": [("抓取数", lambda s: _objects(s).get("highlight_count")),
                      ("总方块数", lambda s: _objects(s).get("n_cubes"))],
    "VideoRepick": [("方块数", lambda s: (_objects(s).get("cube_count") or {}).get("actual")),
                    ("换位", lambda s: _objects(s).get("n_swaps")),
                    ("重抓", lambda s: _objects(s).get("num_repeats"))],
    "RouteStick": [("路线段数", lambda s: _objects(s).get("L"))],
    "PatternLock": [("图案节点数", lambda s: len(_actions(s)["path_nodes"]) if _actions(s).get("path_nodes") else None)],
    "MoveCube": [],
    "InsertPeg": [],
}
DIMS["ButtonUnmask"] = DIMS["VideoUnmask"]
DIMS["ButtonUnmaskSwap"] = DIMS["VideoUnmaskSwap"]

#: 表 1 参照值（v8 方案第一部分 §1）：任务 → 维度 → 档 → 定值或 [lo, hi] 闭区间。只用于核对，不进页面。
TABLE1 = {
    "PickXtimes": {"抓放次数": {"xhard1": 6, "xhard2": 7, "xhard3": 8},
                   "干扰块": {"xhard1": 1, "xhard2": 2, "xhard3": 3}},
    "SwingXtimes": {"摆动轮数": {"xhard1": 4, "xhard2": 5, "xhard3": 6, "xhard4": 7, "xhard5": 8},
                    "干扰块": {"xhard1": 1, "xhard2": 2, "xhard3": 3, "xhard4": 4, "xhard5": 4}},
    "StopCube": {"停止序号 stop_time": {"xhard1": 6, "xhard2": 7, "xhard3": 8, "xhard4": 9, "xhard5": 10},
                 "方块速度 move_interval": {t: 60 for t in NEW_TIERS}},
    "BinFill": {"投入块数": {"xhard1": 6, "xhard2": 7}},
    "VideoUnmask": {"抓取数": {"xhard1": 2, "xhard2": 3, "xhard3": 3, "xhard4": 3},
                    "干扰容器": {"xhard1": 4, "xhard2": 4, "xhard3": 8, "xhard4": 12},
                    "干扰方块": {"xhard1": 2, "xhard2": 2, "xhard3": 4, "xhard4": 6}},
    "VideoUnmaskSwap": {"换位次数": {"xhard1": 5, "xhard2": 7}, "抓取数": {"xhard1": 2, "xhard2": 3},
                        "外圈干扰": {"xhard1": 2, "xhard2": 4}},
    "ButtonUnmaskSwap": {"换位次数": {"xhard1": 3, "xhard2": 5}, "抓取数": {"xhard1": 2, "xhard2": 3},
                         "外圈干扰": {"xhard1": 2, "xhard2": 4}},
    "VideoPlaceButton": {"放置次数": {"xhard1": 3, "xhard2": 4}},
    "VideoPlaceOrder": {"访问总次数": {"xhard1": 5, "xhard2": 6}},
    "PickHighlight": {"抓取数": {"xhard1": 4, "xhard2": 5}, "总方块数": {"xhard1": 7, "xhard2": 8}},
    "VideoRepick": {"方块数": {"xhard1": 4, "xhard2": 5}, "换位": {"xhard1": 4, "xhard2": 6},
                    "重抓": {"xhard1": 2, "xhard2": 3}},
    "RouteStick": {"路线段数": {"xhard1": [8, 10], "xhard2": [11, 13], "xhard3": [14, 16]}},
    "PatternLock": {"图案节点数": {"xhard1": [9, 12], "xhard2": [13, 15], "xhard3": [16, 18]}},
    "MoveCube": {}, "InsertPeg": {},
}
TABLE1["ButtonUnmask"] = TABLE1["VideoUnmask"]
NO_DIM_TEXT = "官方 xhard4 配置（本任务没有档位梯度，表 1「不动」）"
XHARD0_TEXT = "官方 hard（xhard0 不动），各局配置随机；逐局页列出能从本局 subgoal 数出的值"


def table1_ok(task: str, dim: str, tier: str, value) -> bool:
    """值是否符合表 1：定值相等，区间落在 [lo, hi] 内。"""
    want = TABLE1.get(task, {}).get(dim, {}).get(tier)
    if want is None or value is None:
        return False
    if isinstance(want, list):
        return want[0] <= value <= want[1]
    return value == want


def spec_config(task: str, spec: dict) -> dict:
    """一局的维度值 {维度名: 值}（取不到为 None）。"""
    out = {}
    for dim, fn in DIMS[task]:
        try:
            out[dim] = fn(spec)
        except (KeyError, TypeError):
            out[dim] = None
    return out


# ── 路径与通用读写 ────────────────────────────────────────────────────


class Media:
    """媒体白名单：ID = sha256(键)[:24]，值为解析后的绝对路径。"""

    def __init__(self):
        self.paths: dict[str, str] = {}

    def add(self, key: str, path: Path) -> str:
        path = Path(path).resolve(strict=True)
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"不是 mp4：{path}")
        ident = hashlib.sha256(key.encode()).hexdigest()[:24]
        if self.paths.setdefault(ident, str(path)) != str(path):
            raise ValueError(f"媒体 ID 冲突：{key}")
        return ident


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(path: str | Path, *bases: Path) -> Path:
    """绝对路径原样；相对路径依次相对各 base 查找，取第一个存在的（都不存在时返回第一个 base 下的拼接）。"""
    p = Path(path)
    if p.is_absolute():
        return p
    for base in bases:
        if base is not None and (Path(base) / p).exists():
            return Path(base) / p
    return Path(bases[0]) / p


def load_delivery(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != DELIVERY_SCHEMA:
        raise ValueError(f"delivery.json schema 应为 {DELIVERY_SCHEMA}，实为 {data.get('schema')!r}")
    return data


def delivery_h5(row: dict, delivery_path: Path, path_base: Path) -> Path:
    """逐局 h5：按 ``h5``（S2-B 口径为绝对路径，优先）→ ``h5_path`` → ``path``（相对 delivery.json 目录）取第一个；相对路径先相对 delivery.json 所在目录，再相对 path_base。"""
    raw = row.get("h5") or row.get("h5_path") or row.get("path")
    if not raw:
        raise ValueError(f"delivery 行缺 h5 路径：{row.get('task')}/{row.get('tier')}/{row.get('seed')}")
    return resolve(raw, Path(delivery_path).resolve().parent, path_base)


def pick_gen_video(h5: Path, row: dict, gen_videos: Path | None, delivery_path: Path | None = None,
                   path_base: Path | None = None) -> Path:
    """生成视频：先看 ``video``／``mp4`` 字段（相对路径与 h5 同口径：先相对 delivery.json 所在目录、再相对
    ``path_base``），再看 h5 同 episode 的 ``videos/``，再看 ``--gen-videos``。"""
    seed = int(row["seed"])
    if row.get("video") or row.get("mp4"):
        bases = [b for b in (Path(delivery_path).resolve().parent if delivery_path else None, path_base) if b is not None]
        return resolve(row.get("video") or row.get("mp4"), *(bases or [REPO_ROOT]))
    dirs = [h5.parent.parent / "videos"]
    if gen_videos is not None:
        dirs.append(Path(gen_videos) / row["tier"] / f"{row['task']}_episode_{row['episode']}" / "videos")
    for videos in dirs:
        if not videos.is_dir():
            continue
        hits = [p for p in sorted(videos.glob("*.mp4"))
                if not p.name.startswith(("FAILED", "success_NO_OBJECT")) and f"_seed{seed}_" in p.name]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise ValueError(f"{videos} 下符合条件的 mp4 有 {len(hits)} 个（应为 1）")
    raise ValueError(f"找不到生成视频：{row['task']}/{row['tier']}/seed {seed}（查过 {[str(d) for d in dirs]}）")


def robomme_hard_root() -> Path:
    """``robomme_hard`` 包目录：优先本仓子模块 ``third_party/robomme_benchmark/src``；子模块未检出（如干净 worktree）时
    按 ``robomme_hard`` 包的查找位置定位（``find_spec`` 只查位置、不执行包的 ``__init__``，不触发 torch／sapien）。"""
    path = BENCHMARK_SRC / "robomme_hard"
    if (path / "__init__.py").is_file():
        return path
    found = importlib.util.find_spec("robomme_hard")
    for loc in (found.submodule_search_locations or []) if found else []:
        if (Path(loc) / "__init__.py").is_file():
            return Path(loc)
    raise FileNotFoundError(f"找不到 robomme_hard 包：{path} 不存在，sys.path 上也定位不到")


def hard_specs_path() -> Path:
    """``hard_specs.py`` 的位置（``robomme_hard/env_record_wrapper/hard_specs.py``）。"""
    return robomme_hard_root() / "env_record_wrapper/hard_specs.py"


def load_hard_specs():
    """按文件路径加载 ``hard_specs``（只依赖标准库），不触发 ``robomme_hard`` 包导入（torch／sapien）。"""
    name = "_v8_site_hard_specs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, hard_specs_path())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def full_cells(version: str = "v9") -> dict[tuple[str, str], int]:
    """完整交付格表：``v9`` → ``V9_CELLS``（43 格 800）。不读 ``EXPECTED_CELLS``。"""
    H = load_hard_specs()
    if version not in CELL_VERSIONS:
        raise ValueError(f"--cells 只接受 {CELL_VERSIONS}：{version!r}")
    return dict(H.V9_CELLS)


def load_cells(cells_json, version: str = "v9") -> dict[tuple[str, str], int]:
    """交付格表：``--cells-json`` 给 {"Task/tier": n} 子表（冒烟／合成夹具）时用它；否则按 ``--cells v9`` 取完整表。"""
    if cells_json is None:
        return full_cells(version or "v9")
    raw = json.loads(Path(cells_json).read_text(encoding="utf-8"))
    return {tuple(key.split("/")): int(n) for key, n in raw.items()}


def expected_identities(cells: dict[tuple[str, str], int], n_tasks: int = len(NAMES)) -> int:
    """格表推出的身份总数：新值局 + 16 任务 × 12 局 xhard0（V8 完整根 1070 + 192 = 1262，V9 800 + 192 = 992）。"""
    return sum(cells.values()) + n_tasks * XHARD0_PER_TASK


# ── 评估来源 ──────────────────────────────────────────────────────────


def load_eval_run(run: Path) -> tuple[dict, dict, dict]:
    """评估运行（V9 新评 ``--eval-new``，目录结构同 V8 双模型评估运行）：返回 (按身份的权威结果行, 按 (策略, key) 的转码 mp4, report.json)。

    权威结果 = 账本 ``accept`` 行的 ``accepted_attempt_id`` 所指结果行；身份键 (tier, task, seed)，值 {页面策略 ID: 行}。"""
    run = Path(run)
    out: dict[tuple, dict] = defaultdict(dict)
    for seat in sorted(glob.glob(str(run / "nfs-records/run/s[0-9]*"))):
        for pol, pid in EVAL_POLICY.items():
            # 历史运行根的目录与账本文件用旧标签：官方名优先，旧名只读兼容
            name = next((n for n in [pol] + official_defs().legacy_labels(pol) if (Path(seat) / n).is_dir()), pol)
            d = Path(seat) / name
            if not (d / "results.jsonl").exists():
                continue
            rows = {r["attempt_id"]: r for r in jsonl(d / "results.jsonl") if r.get("attempt_id")}
            for led in jsonl(d / f"{name}.ledger.jsonl"):
                if led.get("kind") != "accept":
                    continue
                row = rows.get(led["accepted_attempt_id"])
                if row is None:
                    raise ValueError(f"accept 指向的结果行不存在：{d} {led['accepted_attempt_id']}")
                key = (row["tier"], row["task"], int(row["seed"]))
                if pid in out[key]:
                    raise ValueError(f"同一身份两条 accept：{pol} {key}")
                out[key][pid] = row
    media = {}
    manifest = run / "site-media/manifest.jsonl"
    if manifest.exists():
        for r in jsonl(manifest):
            if r.get("mp4_frames") == r.get("frames") and not r.get("error"):
                media[(EVAL_POLICY[official_defs().canonical_policy(r["policy"])], r["key"])] = r["mp4"]
    report = json.loads((run / "report/report.json").read_text(encoding="utf-8"))
    return dict(out), media, report


# ── V9 评估复用（--eval-reuse / --reused / --eval-new）────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _quad(row: dict) -> tuple:
    """复用键四元组 (task, tier, seed, spec_sha256)。"""
    return (row.get("task"), row.get("tier"), int(row["seed"]) if row.get("seed") is not None else None,
            row.get("spec_sha256"))


def load_reused(path: Path, path_base: Path) -> tuple[dict, dict]:
    """读 ``reused.json``（``v9-eval-reused/1``）并对 V8 manifest 逐行核四元组。

    返回 ``(by_ident, meta)``：``by_ident[(tier, task, seed)] = {"row": 行, "invalid": None | 原因}``；``invalid``
    非空的行不复用（置空）。文件级不符（schema、count、重复身份、V8 manifest 不可读或 sha256 不符）直接抛错。"""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != REUSED_SCHEMA:
        raise ValueError(f"reused.json schema 应为 {REUSED_SCHEMA}，实为 {doc.get('schema') if isinstance(doc, dict) else doc!r}")
    rows = doc.get("rows")
    if not isinstance(rows, list) or doc.get("count") != len(rows):
        raise ValueError(f"reused.json count={doc.get('count')!r} 与 rows 行数 {None if not isinstance(rows, list) else len(rows)} 不符")
    manifest_path = resolve(doc.get("v8_manifest") or "", path.resolve().parent, path_base)
    if not doc.get("v8_manifest") or not manifest_path.is_file():
        raise ValueError(f"reused.json 的 v8_manifest 不可读：{doc.get('v8_manifest')!r}")
    got = sha256_file(manifest_path)
    if got != doc.get("v8_manifest_sha256"):
        raise ValueError(f"V8 manifest sha256 不符：{got} ≠ reused.json 记录的 {doc.get('v8_manifest_sha256')}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_key = {}
    for r in manifest.get("rows") or []:
        if r.get("key") in by_key:
            raise ValueError(f"V8 manifest key 重复：{r.get('key')}")
        by_key[r.get("key")] = r
    out: dict[tuple, dict] = {}
    seen_keys = set()
    for row in rows:
        missing = [k for k in ("task", "tier", "seed", "spec_sha256", "v8_key") if row.get(k) is None]
        if missing:
            raise ValueError(f"reused.json 行缺 {missing}：{row}")
        ident = (row["tier"], row["task"], int(row["seed"]))
        if ident in out or row["v8_key"] in seen_keys:
            raise ValueError(f"reused.json 身份或 v8_key 重复：{ident} {row['v8_key']}")
        seen_keys.add(row["v8_key"])
        invalid = None
        mrow = by_key.get(row["v8_key"])
        if not _is_sha(row["spec_sha256"]):
            invalid = "sha"
        elif mrow is None:
            invalid = "missing"
        elif _quad(mrow)[:3] != _quad(row)[:3]:
            invalid = "identity"
        elif mrow.get("spec_sha256") != row["spec_sha256"]:
            invalid = "sha"
        out[ident] = {"row": row, "invalid": invalid}
    meta = {"path": str(path), "sha256": sha256_file(path), "count": len(rows), "v8_manifest": str(manifest_path),
            "v8_manifest_sha256": got}
    return out, meta


def load_site_eval(site: Path) -> tuple[dict, dict, dict, dict]:
    """V8 站点目录（``catalog.json`` + ``media-private.json``）→ ``(新值局 {key: (task, tier, ep)},
    xhard0 {(task, seed): ep}, 媒体 {ID: 绝对路径}, catalog["eval"])``；key = ``<task>_<tier>_<seed>``（与 V8 manifest
    的 ``key`` 同形态）。"""
    site = Path(site)
    cat = json.loads((site / "catalog.json").read_text(encoding="utf-8"))
    if cat.get("schema") != SITE_CATALOG_SCHEMA or (cat.get("eval") or {}).get("status") != "evaluated":
        raise ValueError(f"{site} 不是已评估的 V8 站点目录（schema={cat.get('schema')!r} "
                         f"eval.status={(cat.get('eval') or {}).get('status')!r}）")
    media = json.loads((site / "media-private.json").read_text(encoding="utf-8"))
    new_eps: dict[str, tuple] = {}
    x0: dict[tuple, dict] = {}
    for t in cat["tasks"]:
        for tier, cell in t["tiers"].items():
            for ep in cell["episodes"]:
                if tier == "xhard0":
                    x0[(t["id"], int(ep["seed"]))] = ep
                else:
                    key = f"{t['id']}_{tier}_{int(ep['seed'])}"
                    if key in new_eps:
                        raise ValueError(f"V8 站点目录身份重复：{key}")
                    new_eps[key] = (t["id"], tier, ep)
    return new_eps, x0, media, cat.get("eval") or {}


def copy_eval_items(src_items: dict, media: "Media", v8_media: dict, key_fmt: str) -> dict:
    """逐策略复制评估位；视频 ID 按 V8 白名单取绝对路径，以本站键 ``key_fmt.format(p=策略)`` 重新登记（文件必须
    存在）。键与 V8 建站同形态，所以同一视频的媒体 ID 与 V8 站点相同。"""
    out = {}
    for p, item in src_items.items():
        item = dict(item)
        if item.get("media"):
            path = v8_media.get(item["media"])
            if path is None:
                raise ValueError(f"V8 媒体 ID 不在 media-private.json：{item['media']}")
            item["media"] = media.add(key_fmt.format(p=p), Path(path))
        out[p] = item
    return out


def fill_v9_eval(ep: dict, key: tuple, spec: dict | None, reused: dict, v8_eps: dict, v8_media: dict, v8_eval: dict,
                 ev_rows: dict, ev_media: dict, src: dict, media: "Media", counts: Counter, problems: list[str],
                 path_base: Path) -> str | None:
    """一局新值局的 V9 评估位：复用（reused.json 四元组全对上）→ 新评（spec_sha256 对上）→ 否则置空并计数。

    返回置空原因（未置空返回 None）。"""
    tier, task, seed = key
    sha = (spec or {}).get("spec_sha256")
    ep["eval"]["new"] = {}
    origin, note = "empty", None
    hit = reused.get(key)
    if hit is not None:
        row = hit["row"]
        if ev_rows.get(key):
            problems.append(f"同一身份既在复用集合又有新评结果 {key}（以复用为准）")
        if hit["invalid"] == "sha" or row["spec_sha256"] != sha:
            counts["reuse_sha_mismatch"] += 1
            note = f"spec_sha256 不符（规格 {str(sha)[:12]}，reused.json／V8 manifest {str(row['spec_sha256'])[:12]}）"
        elif hit["invalid"] == "identity":
            counts["reuse_identity_mismatch"] += 1
            note = f"v8_key {row['v8_key']} 在 V8 manifest 里身份不同"
        elif hit["invalid"] == "missing" or row["v8_key"] not in v8_eps:
            counts["reuse_missing"] += 1
            note = f"v8_key {row['v8_key']} 在 V8 manifest 或 V8 站点目录里不存在"
        else:
            o_task, o_tier, old_ep = v8_eps[row["v8_key"]]
            items = (old_ep.get("eval") or {}).get("new") or {}
            if (o_task, o_tier, int(old_ep["seed"])) != (task, tier, seed):
                counts["reuse_identity_mismatch"] += 1
                note = f"V8 站点记录 {row['v8_key']} 身份为 {(o_task, o_tier, old_ep['seed'])}"
            elif not all(p in items and items[p].get("status") in FINAL for p, _ in POLICIES):
                counts["reuse_missing"] += 1
                note = f"V8 站点记录 {row['v8_key']} 缺两模型终态"
            else:
                try:
                    ep["eval"]["new"] = copy_eval_items(items, media, v8_media, f"eval/new/{{p}}/{tier}/{task}/{seed}")
                    origin = "reused"
                    ep["eval_source"] = (f"V8 双模型评估 {v8_eval.get('run')}（V9 复用：四元组 task/tier/seed/spec_sha256 "
                                         f"与 V8 逐字节相同，执行段 1600 步严格截断）")
                except (OSError, ValueError) as exc:
                    counts["reuse_missing"] += 1
                    note = f"V8 评估视频不可用：{exc}"
    else:
        rows = ev_rows.get(key) or {}
        if rows:
            if not all(p in rows for p, _ in POLICIES):
                note = f"新评缺策略 {[p for p, _ in POLICIES if p not in rows]}"
            elif any(r.get("spec_sha256") != sha for r in rows.values()):
                counts["new_sha_mismatch"] += 1
                note = "新评结果行 spec_sha256 与规格行不符"
            else:
                items = {}
                for p, _ in POLICIES:
                    row = rows[p]
                    item = {"status": row["status"], "steps": row.get("exec_steps"),
                            "max_steps": row.get("effective_max_steps") or row.get("max_steps")}
                    mp4 = ev_media.get((p, row["key"]))
                    if mp4 is None:
                        problems.append(f"{p} 缺新评视频 {row['key']}")
                    else:
                        try:
                            item["media"] = media.add(f"eval/new/{p}/{tier}/{task}/{seed}", resolve(mp4, path_base))
                        except (OSError, ValueError) as exc:
                            problems.append(f"{p} 新评视频不可用 {row['key']}：{exc}")
                    items[p] = item
                ep["eval"]["new"] = items
                origin = "new"
                ep["eval_source"] = f"V9 双模型评估 {Path(src['eval_new']).name}（执行段 1600 步严格截断）"
        else:
            note = "不在 reused.json 复用集合里，也没有新评结果"
    ep["eval_origin"] = origin
    if origin == "empty":
        counts["eval_empty"] += 1
        counts["eval_unevaluated"] += len(POLICIES)
        ep["eval_note"] = note
    else:
        counts[f"eval_{origin}"] += 1
        counts["eval_filled"] += len(ep["eval"]["new"])
        counts["eval_media"] += sum(1 for it in ep["eval"]["new"].values() if it.get("media"))
    return note if origin == "empty" else None


def _fill_config(entry: dict, task: str, tier: str, eps: list[dict]) -> None:
    """格级配置：xhard0 / 无梯度任务给说明文字，其余按 DIMS 汇总逐局取值集合与分布。"""
    if tier == "xhard0":
        entry["config"] = [{"dim": None, "values": [], "text": XHARD0_TEXT}]
    elif not DIMS[task]:
        entry["config"] = [{"dim": None, "values": [], "text": NO_DIM_TEXT}]
    else:
        entry["config"] = []
        for dim, _ in DIMS[task]:
            dist = Counter(ep["config"][dim] for ep in eps if "config" in ep)
            entry["config"].append({"dim": dim, "values": sorted(v for v in dist if v is not None),
                                    "dist": {str(k): n for k, n in sorted(dist.items(), key=lambda x: str(x[0]))}})


# ── 构建 ────────────────────────────────────────────────────────────


def build_catalog(src: dict) -> tuple[dict, dict, dict]:
    H = load_hard_specs()

    media = Media()
    problems: list[str] = []
    counts = Counter()
    path_base = Path(src.get("path_base") or REPO_ROOT)
    cells_want = load_cells(src.get("cells_json"), src.get("cells") or "v9")
    expected = expected_identities(cells_want)

    # 规格（/4，load_specs_root 校验格表、selected 行数、跨档 seed 不交）
    specs = H.load_specs_root(src["specs_root"], cells_want, check_fingerprint=False)
    spec_rows: dict[tuple, dict] = {}
    exec_cap: dict[str, int] = {}
    for tier, (header, rows) in specs.items():
        exec_cap[tier] = header["exec_cap"]
        for row in rows:
            if row["selected"]:
                spec_rows[(tier, row["task"], int(row["seed"]))] = row

    # 生成交付
    delivery_path = Path(src["delivery"])
    delivery = load_delivery(delivery_path)
    gen = {}
    for row in delivery["rows"]:
        key = (row["tier"], row["task"], int(row["seed"]))
        if key in gen:
            problems.append(f"delivery 身份重复 {key}")
        gen[key] = row
    if set(gen) != set(spec_rows):
        problems.append(f"delivery 与规格 selected 行不一致：缺 {sorted(set(spec_rows) - set(gen))[:5]}，"
                        f"多 {sorted(set(gen) - set(spec_rows))[:5]}")

    # xhard0 生成（v7 已渲染，0 次渲染）
    xgen = {side: {(r["task"], int(r["seed"])): r for r in jsonl(Path(src["xhard0_gen"]) / f"manifest-{side}.jsonl")}
            for side in ("H", "O")}

    # 评估来源
    # 不给评估来源（合成夹具、单测）时保持「全部未评估」：eval 为空、eval_status = unevaluated
    v9 = bool(src.get("eval_reuse") or src.get("reused") or src.get("eval_new"))
    if v9 and not (src.get("eval_reuse") and src.get("reused")):
        raise ValueError("V9 复用模式须同时给 --eval-reuse 与 --reused（复用集合只认 reused.json）")
    has_eval = v9
    if v9:
        reused, reused_meta = load_reused(src["reused"], path_base)
        v8_eps, v8_x0, v8_media, v8_eval = load_site_eval(src["eval_reuse"])
        ev_rows, ev_media, ev_report = load_eval_run(src["eval_new"]) if src.get("eval_new") else ({}, {}, {})
        for k in V9_COUNT_KEYS:
            counts[k] += 0

    # 身份清单
    idents = jsonl(src["identities"])
    by_key = {(r["tier"], r["task"], int(r["seed"])): r for r in idents}
    if len(by_key) != len(idents):
        problems.append("身份清单 (tier, task, seed) 不唯一")
    new_keys = {k for k in by_key if k[0] != "xhard0"}
    if new_keys != set(gen):
        problems.append(f"身份清单新值局与 delivery 不一致：缺 {sorted(set(gen) - new_keys)[:5]}，"
                        f"多 {sorted(new_keys - set(gen))[:5]}")
    x0_per_task = Counter(k[1] for k in by_key if k[0] == "xhard0")
    if x0_per_task != Counter({task: XHARD0_PER_TASK for task in NAMES}):
        problems.append(f"xhard0 身份应为 16 任务 × {XHARD0_PER_TASK}：{dict(x0_per_task)}")

    cells: dict[tuple, list[dict]] = defaultdict(list)
    empties: list[str] = []
    order = lambda k: (TIERS.index(k[0]), k[1], by_key[k].get("candidate") if by_key[k].get("candidate") is not None
                       else -1, by_key[k].get("source_episode") or 0, by_key[k].get("episode") or 0)
    for key in sorted(by_key, key=order):
        tier, task, seed = key
        if tier not in TIERS or task not in NAMES:
            problems.append(f"未知档位或任务 {key}")
            continue
        ident = by_key[key]
        ep = {"seed": seed, "eval_episode": ident.get("episode"), "round": ident.get("round"),
              "shard": ident.get("shard"), "candidate": ident.get("candidate"),
              "source_episode": ident.get("source_episode"),
              "gen": {}, "eval": {"new": {}} if has_eval else {}}
        if not has_eval:
            ep["eval_status"] = "unevaluated"
            counts["eval_unevaluated"] += 1
        if tier == "xhard0":
            for side, entry in (("H", "new"), ("O", "old")):
                row = xgen[side].get((task, seed))
                if row is None:
                    problems.append(f"xhard0 {side} 缺生成记录 {task}/{seed}")
                    continue
                item = {"frames": row["frames"], "demo_frames": row["demo_frames"], "h5_sha12": row["h5_sha256"][:12]}
                if row.get("generation_failed"):
                    item["status"] = "generation_failed"
                    counts["gen_failed"] += 1
                else:
                    try:
                        item["media"] = media.add(f"gen/{entry}/{tier}/{task}/{seed}", resolve(row["mp4"], path_base))
                        counts[f"gen_{entry}_xhard0"] += 1
                    except (OSError, ValueError) as exc:
                        problems.append(f"xhard0 {side} 视频不可用 {task}/{seed}：{exc}")
                ep["gen"][entry] = item
        if tier == "xhard0" and v9:
            # xhard0 规格与视频 V9 未动：按 (task, seed) 原样取 V8 站点目录的两路线评估位
            old_ep = v8_x0.get((task, seed))
            if old_ep is None or not all(p in (old_ep.get("eval") or {}).get(e, {}) for p, _ in POLICIES
                                         for e in ("new", "old")):
                problems.append(f"V8 站点目录缺 xhard0 评估 {task}/{seed}")
                counts["eval_unevaluated"] += len(POLICIES)
            else:
                try:
                    for entry in ("new", "old"):
                        ep["eval"][entry] = copy_eval_items(old_ep["eval"][entry], media, v8_media,
                                                            f"eval/{entry}/{{p}}/{tier}/{task}/{seed}")
                        counts["eval_filled"] += len(ep["eval"][entry])
                        counts["eval_media"] += sum(1 for it in ep["eval"][entry].values() if it.get("media"))
                    ep["flip"] = dict(old_ep.get("flip") or {})
                    counts["flip"] += sum(bool(v) for v in ep["flip"].values())
                    ep["eval_source"] = old_ep.get("eval_source")
                    counts["eval_x0_reused"] += 1
                except (OSError, ValueError) as exc:
                    problems.append(f"xhard0 评估复用失败 {task}/{seed}：{exc}")
        if tier != "xhard0":
            row = gen.get(key)
            spec = spec_rows.get(key)
            if row is None or spec is None:
                problems.append(f"缺交付或规格行 {key}")
            else:
                ep["episode"] = row.get("episode")
                ep["exec_cap"] = exec_cap.get(tier)
                try:
                    h5 = delivery_h5(row, delivery_path, path_base)
                    missing_keys = [k for k in ("exec_steps", "frames") if row.get(k) is None]
                    if missing_keys:
                        raise ValueError(f"delivery 行缺 {missing_keys}：{key}")
                    video = pick_gen_video(h5, row, src.get("gen_videos"), delivery_path, path_base)
                    item = {"frames": row.get("frames"), "exec_steps": row.get("exec_steps"),
                            "media": media.add(f"gen/new/{tier}/{task}/{seed}", video)}
                    if row.get("frames") is not None and row.get("exec_steps") is not None:
                        item["demo_frames"] = row["frames"] - row["exec_steps"]
                    ep["gen"]["new"] = item
                    counts["gen_v8"] += 1
                except (OSError, ValueError) as exc:
                    problems.append(str(exc))
                cfg = spec_config(task, spec["spec"])
                ep["config"] = cfg
                for dim, value in cfg.items():
                    if not table1_ok(task, dim, tier, value):
                        counts["config_mismatch"] += 1
                        problems.append(f"配置与表 1 不符 {task}/{tier}/seed {seed}：{dim}={value!r}")
        if tier != "xhard0" and v9:
            note = fill_v9_eval(ep, key, spec, reused, v8_eps, v8_media, v8_eval, ev_rows, ev_media, src, media,
                                counts, problems, path_base)
            if note is not None:
                empties.append(f"评估置空 {task}/{tier}/seed {seed}：{note}")
        cells[(task, tier)].append(ep)

    if v9:
        stray = sorted(set(reused) - {k for k in by_key if k[0] != "xhard0"})
        if stray:
            problems.append(f"reused.json 有 {len(stray)} 个身份不在本次身份清单里：{stray[:5]}")
        stray = sorted(set(ev_rows) - {k for k in by_key if k[0] != "xhard0"})
        if stray:
            problems.append(f"新评结果有 {len(stray)} 个身份不在本次身份清单里：{stray[:5]}")

    tasks = []
    for task in NAMES:
        tiers = {}
        for tier in TIERS:
            eps = cells.get((task, tier))
            if not eps:
                continue
            for i, ep in enumerate(eps, 1):
                ep["idx"] = i
            if not has_eval:
                tiers[tier] = {"episodes": eps, "rates": {}, "eval_status": "unevaluated"}
                _fill_config(tiers[tier], task, tier, eps)
                continue
            rates = {"new": {p: dict(Counter(ep["eval"]["new"][p]["status"] for ep in eps if p in ep["eval"]["new"]))
                             for p, _ in POLICIES}}
            if tier == "xhard0":
                rates["old"] = {p: dict(Counter(ep["eval"]["old"][p]["status"] for ep in eps if p in ep["eval"].get("old", {})))
                                for p, _ in POLICIES}
            else:
                # 复用部分的数字是 V8 当时跑的（V8 report 按 V8 格局数统计，不再逐格对账）；新评部分与新评 report.json 对账
                for pol, p in EVAL_POLICY.items():
                    got = dict(Counter(ep["eval"]["new"][p]["status"] for ep in eps
                                       if ep.get("eval_origin") == "new" and p in ep["eval"]["new"]))
                    per_pol = ev_report.get("per_policy") or {}
                    pol_rep = next((per_pol[n] for n in [pol] + official_defs().legacy_labels(pol) if n in per_pol), {})
                    want = (pol_rep or {}).get("cells", {}).get(f"{task}@{tier}", {})
                    want = {k: want.get(k, 0) for k in FINAL if want.get(k)}
                    got = {k: v for k, v in got.items() if v}
                    if want != got:
                        counts["rate_mismatch"] += 1
                        problems.append(f"{p} {task}/{tier} 新评成败数与新评 report.json 不符：{got} vs {want}")
            entry = {"episodes": eps, "rates": rates}
            _fill_config(entry, task, tier, eps)
            tiers[tier] = entry
        tasks.append({"id": task, "name": NAMES[task], "tiers": tiers})

    if v9:
        n_new_value = sum(1 for k in by_key if k[0] != "xhard0")
        eval_head = {
            "status": "evaluated", "mode": "v9-reuse", "run": Path(src["eval_new"]).name if src.get("eval_new") else None,
            "reuse": {"run": v8_eval.get("run"), "site_eval": str(Path(src["eval_reuse"]).resolve()),
                      "reused_json": reused_meta["path"], "reused_sha256": reused_meta["sha256"],
                      "reused_rows": reused_meta["count"], "v8_manifest_sha256": reused_meta["v8_manifest_sha256"],
                      "counts": {k: counts[k] for k in V9_COUNT_KEYS}},
            "summary": {p: {"success": sum(1 for t in tasks for tier, cell in t["tiers"].items() if tier != "xhard0"
                                           for ep in cell["episodes"]
                                           if (ep["eval"].get("new", {}).get(p) or {}).get("status") == "success"),
                            "denominator": n_new_value} for p, _ in POLICIES},
        }
    catalog = {
        "schema": SITE_CATALOG_SCHEMA,
        "tiers": list(TIERS),
        "policies": [{"id": p, "label": label} for p, label in POLICIES],
        "eval": eval_head if v9 else {"status": "unevaluated",
                                      "reason": "未提供评估来源（--eval-reuse／--reused），评估位显示「未评估」。"},
        "tasks": tasks,
        "notes": {
            "eval": "xhard1～5：V8 双模型评估（2026-10-02，十张 A40，SimpleMemVLA 官方权重与 MME-VLA perceptual-framesamp-modul/79999，"
                    "执行段 1600 步严格截断计 timeout），每身份取唯一权威终态；评估视频为录像器 FFV1 无损录像展开重复帧后转成的 H.264"
                    "（左前视、右腕部，左上角 DEMO／EXEC），无损原片留在本机存档。xhard0：V8 阶段 3′ 两路线评估（上限 1300），"
                    "新入口 = hard 路线、旧入口 = 官方路线；SimpleMemVLA 当时未录视频，MME-VLA 有视频。",
            "config": "xhard1～5 的任务配置逐局取自 v8 冻结规格（hard-specs/4）的规格行，格内汇总为取值集合；"
                      "RouteStick／PatternLock 是区间，列出实际取值分布。",
            "xhard0_gen": "xhard0 生成视频复用 v7 由 h5 离线合成的版本（左前视、右腕部，左上角 DEMO／EXEC），版式与录像器视频不同；"
                          "新入口 h5 与旧入口逐字节相同，两行生成视频因此相同。",
            "gen_failed": "VideoPlaceOrder 有 2 局官方原版在该 seed 上即生成失败（DatasetGenerationError），两入口相同，无生成视频。",
        },
    }
    if v9:
        catalog["notes"]["eval"] = (
            "xhard1～5：与 V8 逐字节相同的局（按 task／tier／seed／spec_sha256 四元组对齐，复用集合以 reused.json 为准）"
            f"直接复用 V8 双模型评估 {v8_eval.get('run')} 的结果，数字是 V8 当时跑的；其余新生成的局另做两模型评估"
            f"{'（' + Path(src['eval_new']).name + '）' if src.get('eval_new') else ''}，口径与 V8 相同（SimpleMemVLA 官方权重与 "
            "MME-VLA perceptual-framesamp-modul/79999，执行段 1600 步严格截断计 timeout）。评估视频为录像器 FFV1 无损录像"
            "展开重复帧后转成的 H.264（左前视、右腕部，左上角 DEMO／EXEC）。xhard0：沿用 V8 阶段 3′ 两路线评估（上限 1300），"
            "新入口 = hard 路线、旧入口 = 官方路线；SimpleMemVLA 当时未录视频，MME-VLA 有视频。")
        catalog["notes"]["config"] = ("xhard1～5 的任务配置逐局取自 V9 规格（hard-specs/4）的规格行，格内汇总为取值集合；"
                                      "RouteStick／PatternLock 是区间，列出实际取值分布。")
    stats = {"identities": len(by_key), "expected": expected, "counts": dict(counts),
             "problems": problems, "media": len(media.paths), "v9": v9, "empties": empties}
    return catalog, media.paths, stats


def verdict_line(stats: dict, ok: bool) -> str:
    c = stats["counts"]
    return (f"V8_SITE_CATALOG={'PASS' if ok else 'FAIL'} identities={stats['identities']} expected={stats['expected']} "
            f"gen_v8={c.get('gen_v8', 0)} gen_xhard0_new={c.get('gen_new_xhard0', 0)} "
            f"gen_xhard0_old={c.get('gen_old_xhard0', 0)} gen_failed={c.get('gen_failed', 0)} "
            f"eval_filled={c.get('eval_filled', 0)} eval_media={c.get('eval_media', 0)} "
            f"eval_unevaluated={c.get('eval_unevaluated', 0)} flip={c.get('flip', 0)} rate_mismatch={c.get('rate_mismatch', 0)} "
            f"config_mismatch={c.get('config_mismatch', 0)} media={stats['media']} problems={len(stats['problems'])}"
            + ("".join(f" {k}={c.get(k, 0)}" for k in V9_COUNT_KEYS) if stats.get("v9") else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", choices=CELL_VERSIONS, default="v9",
                    help="完整交付格表：v9（V9_CELLS，缺省且唯一；V8 的 1070 局表已删除）；--cells-json 给子表时优先")
    ap.add_argument("--allow-eval-empty", action="store_true",
                    help="V9 复用模式下有评估置空局时仍写产物（判定仍为 FAIL，供排查）")
    for name, default in DEFAULT_SOURCES.items():
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, type=Path,
                        help=f"缺省 {default}" + (f"；--cells v9 时缺省 {V9_DEFAULT_SOURCES[name]}"
                                                  if name in V9_DEFAULT_SOURCES else ""))
    args = ap.parse_args(argv)
    src = {}
    for name, default in DEFAULT_SOURCES.items():
        value = getattr(args, name)
        if value is None:
            value = V9_DEFAULT_SOURCES.get(name, default) if args.cells == "v9" else default
        src[name] = value
    src["cells"] = args.cells
    try:
        catalog, media, stats = build_catalog(src)
    except Exception as exc:  # 规格校验、文件缺失等：照实 FAIL，不写产物
        print(f"# {type(exc).__name__}: {exc}", flush=True)
        print("V8_SITE_CATALOG=FAIL identities=0 expected=0 gen_v8=0 gen_xhard0_new=0 gen_xhard0_old=0 gen_failed=0 "
              "eval_filled=0 eval_media=0 eval_unevaluated=0 flip=0 rate_mismatch=0 config_mismatch=0 media=0 problems=1",
              flush=True)
        return 1
    for problem in stats["problems"][:40]:
        print(f"# {problem}", flush=True)
    for line in stats["empties"][:40]:
        print(f"# {line}", flush=True)
    empty = stats["counts"].get("eval_empty", 0)
    ok = not stats["problems"] and stats["identities"] == stats["expected"] and not empty
    write = ok or (bool(empty) and args.allow_eval_empty and not stats["problems"]
                   and stats["identities"] == stats["expected"])
    targets = [args.out / "catalog.json", args.out / "media-private.json"]
    exists = [str(t) for t in targets if t.exists()]
    if write and exists:  # 写入前确认两个目标都不存在，不留半份产物
        print(f"# 目标文件已存在，拒绝覆盖：{exists}", flush=True)
        stats["problems"].append("目标文件已存在")
        ok = write = False
    if stats["identities"] != stats["expected"]:
        print(f"# 身份总数 {stats['identities']} ≠ 格表推出的 {stats['expected']}", flush=True)
    if empty:
        print(f"# 评估置空 {empty} 局" + ("（--allow-eval-empty：照写产物，判定 FAIL）" if write else ""), flush=True)
    if write:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "catalog.json").open("x", encoding="utf-8") as handle:
            json.dump(catalog, handle, ensure_ascii=False, separators=(",", ":"))
        with (args.out / "media-private.json").open("x", encoding="utf-8") as handle:
            json.dump(media, handle, ensure_ascii=False, indent=0)
    print(verdict_line(stats, ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
