#!/usr/bin/env python3
"""robomme_hard 回归工具（现行口径 V9：1002-newtask-v9-movecube-region-800-plan.md 第二部分 §2.3；各闸门沿用
v8 的 1001-newtask-v8-xhard-gradient-plan.md 第一部分 §2.3、§3，第二部分 §2.3 闸门总表；xhard0 reset 层对拍源自
v7 的 0928 方案第二部分 §1.5）。

子命令：

* ``delivery-set``（纯 CPU，v8）：读 /4 规格根，逐格交付数与格表**相等** → ``V8_DELIVERY_SET``；同任务跨档 seed
  两两不交 → ``V8_SEED_DISJOINT``；header 布局规则 independent、行 ``layout_parent`` 全空且同任务跨档位置指纹相同的
  对数为 0 → ``V8_LAYOUT_INDEPENDENT``。
* ``tier-values``（纯 CPU，v8）：14 个有取值维度的任务逐格逐局取值等于表 1（RouteStick／PatternLock 落在区间内，
  另打印逐格长度直方图）→ ``V8_TIER_VALUES``。
* ``step-headroom``（纯 CPU）：v8／v9 交付 h5 非演示步全部 ≤ 1600、抽样阶段超限过滤数、xhard0 按 1300 单独查 →
  ``V8_STEP_CAP``；只接受 ``v8-delivery/1`` 交付清单。
* ``reset-replay``（GPU）：每格 candidate 最小的正式局经评估链 ``make_env_for_episode`` + reset，``spec_binding()``
  零差；/4 规格（43 格）→ ``V8_RESET_REPLAY``；非 /4 规格根直接拒收。
* ``eval-smoke``（GPU，1 任务 × 1 档 × 1 局）：合作者入口可用；xhard0 局须为导出模式；每任务局数按交付格表推出
  （v8：PickXtimes／SwingXtimes／StopCube 62、MoveCube／InsertPeg 32、其余 92）→ ``HARD_EVAL_SMOKE``。
* ``xhard0-reset-parity``（GPU）：官方 robomme 与 robomme_hard 两进程各 reset 192 局，确定性层逐位比 →
  ``XHARD0_RESET_PARITY``；演示层只报告 ``XHARD0_DEMO_DIFF=INFO``。
* ``env-digest``（GPU，v7.5eval）：身份逐层摘要 + 原始数组 + 测速 → ``ENV_DIGEST_DONE``、``ENV_SPEED=INFO``。
* ``env-digest-compare``（纯 CPU，v7.5eval）：两格逐层对拍，只报告 → ``ENV_DIGEST_PARITY``。

v9（1002-newtask-v9-movecube-region-800-plan.md 第二部分 §2.3）：判定行前缀按格表版本输出——``delivery-set`` 按
``--cells``（``v9full``／``v9``／``v9smoke``／``v9shard1``，或阶段 3b 切换 ``EXPECTED_CELLS`` 后的 ``full``）、
``reset-replay`` 按规格根 header 的逐任务配额、``step-headroom`` 按交付清单（行带 ``source`` 或只能被 V9_CELLS 覆盖）
推出 V9 时打 ``V9_DELIVERY_SET``／``V9_RESET_REPLAY``／``V9_STEP_CAP`` 等，否则保持 ``V8_*``；新增：

* ``movecube-layout``（纯 CPU，v9）：MoveCube xhard4 交付局两段 × 方块／goal／抓杆点共 300 点用源码
  ``MoveCube._in_region_u`` 判在规格 region 的 U 内、数落在旧 V8 区域外的点 → ``V9_MOVECUBE_LAYOUT``；按
  ``_freeze._movecube_way`` 数三种运动方式 = ``V9_MOVECUBE_QUOTA_BY_WAY`` → ``V9_MOVECUBE_WAYS``。

旧的 ``s4-subset`` 与 S4 映射表随 v6 规格一起退役；v7 专用的 ``layout-shared``、``prefix-geometry``、
``xhard0-eval-parity``、``step-headroom --v7`` 与 reset-replay／eval-smoke 的 v7 口径于 1003 维护计划细则 2.3
删除（git 历史可取回）。
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "src", REPO):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))



def _hard_specs():
    from robomme_hard.env_record_wrapper import hard_specs

    return hard_specs


def _xhard0_prefix(hs) -> int:
    """builder 每任务前置的 xhard0 局数：开关 ``XHARD0_IN_TEST_HARD`` 开为 12、关为 0（旧 hard_specs 无开关时按 12）。"""
    return getattr(hs, "xhard0_prefix", lambda: hs.XHARD0_PER_TASK)()


def _xhard0_in_test_hard(hs) -> bool:
    return bool(getattr(hs, "XHARD0_IN_TEST_HARD", True))


def _require_xhard0_in_test_hard(hs) -> None:
    """逻辑上必须有 ood 里 xhard0 的子命令：开关关闭时明确报错，不许静默错位。"""
    if not _xhard0_in_test_hard(hs):
        raise SystemExit("xhard0 已退出 ood：需设 ROBOMME_HARD_XHARD0_IN_TEST_HARD=1 再跑本子命令")


# ── 记录点路径（静态收集 SpecRecorder.record 的第一个参数）─────────────────────


def record_path_patterns(task: str, root: Path = REPO / "src" / "robomme_hard" / "robomme_env",
                         method: str = "record") -> set[str]:
    """该任务环境文件与共用 utils 里 ``*.record(<路径>, …)`` 的路径正则（按任务分开，不同任务同名路径可能一记一注）；
    f-string 的占位符只匹配一个路径段（``[^.]+``）。"""
    import re

    patterns: set[str] = set()
    for path in [root / f"{task}.py", *sorted((root / "utils").glob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == method
                    and node.args):
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                patterns.add(re.escape(arg.value))
            elif isinstance(arg, ast.JoinedStr):
                parts = [re.escape(str(v.value)) if isinstance(v, ast.Constant) else "[^.]+" for v in arg.values]
                pattern = "".join(parts)
                if not pattern.startswith("[^.]+"):  # 整条路径都是变量（如 f"{spec_prefix}.placed"）无法静态归类，不收
                    patterns.add(pattern)
    return patterns


def is_record_path(key: str, patterns: set[str]) -> bool:
    import re

    return any(re.fullmatch(p, key) or re.match(p + r"\.", key) for p in patterns)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def _leaf_diff(a: Any, b: Any) -> float:
    from robomme_hard.env_record_wrapper.hard_specs import _max_abs_diff

    return _max_abs_diff(a, b)


def compare_specs(s4_spec: dict[str, Any], hard_spec: dict[str, Any], record_prefixes: set[str],
                  value_patterns: set[str] = frozenset()) -> dict[str, Any]:
    """逐叶比对（忽略 provenance 与 identity 两个说明性分支）；返回回注点差、记录点差与最大差。"""
    skip = ("provenance", "identity")
    a = {k: v for k, v in _flatten(s4_spec).items() if k.split(".")[0] not in skip}
    b = {k: v for k, v in _flatten(hard_spec).items() if k.split(".")[0] not in skip}
    injected, within, max_abs, paths = 0, 0, 0.0, []
    tol = _hard_specs().RECORDED_FLOAT_TOL
    for key in sorted(set(a) | set(b)):
        if key in a and key in b and a[key] == b[key]:
            continue
        diff = _leaf_diff(a.get(key), b.get(key)) if key in a and key in b else math.inf
        is_record = is_record_path(key, record_prefixes) and not is_record_path(key, value_patterns)
        if is_record and diff <= tol:
            within += 1
            max_abs = max(max_abs, diff)
        else:
            injected += 1
            paths.append(key)
    return {"injected": injected, "within": within, "max_abs": max_abs, "paths": paths}


def _hp():
    """同目录的 hard_parity（纯标准库导入；v8 交付清单适配、格表解析、hard_specs 轻量加载都在那里）。"""
    from scripts.parity import hard_parity

    return hard_parity


def _hs_light():
    """不经 robomme_hard 包 __init__（会连带导入仿真）的 hard_specs；纯 CPU 守卫与规格读取用它。"""
    return _hp().hard_specs_light()


#: xhard0 的执行步上限（与官方 scripts/evaluation.py 的默认步数相同）。按档的步数查表已从 hard_specs 删除，
#: 评估入口按数据集传 max_steps；本工具在 dataset="ood" 下混跑 xhard0 与新值档，故在模块内按档取值。
XHARD0_STEP_CAP = 1300


def _cap_for(tier: str) -> int:
    """对拍工具内部的逐局步数上限：xhard0 取 ``XHARD0_STEP_CAP``（1300），新值档取 ``hard_specs.EXEC_CAP``（1600）。

    逐局传给 ``make_env_for_episode(max_steps=…)``：不传就退回 builder 的单一值，混档下会错配其中一类局。"""
    return XHARD0_STEP_CAP if tier == "xhard0" else _hs_light().EXEC_CAP


def specs_tiers(specs_root: str | None = None) -> tuple[tuple[str, ...], bool]:
    """规格根（显式 > ``ROBOMME_HARD_SPECS_ROOT`` > 包内）的档序与是否 /4：``TIERS`` 下任一存在的档文件 header
    为 ``hard-specs/4`` 即按 ``TIERS`` 读五档（v8 方案第二部分 §2.2 第 9 条「delivery_index 按 TIERS 读」；
    只含 xhard5 等部分档的局部根也判 v8）。档序恒为 ``TIERS``（原 V8_TIERS 已并入）。首行读不出的文件跳过。"""
    hs = _hs_light()
    root = hs.specs_root(specs_root)
    v8 = False
    for tier in hs.TIERS:
        path = root / tier / "specs.jsonl"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as stream:
                v8 = json.loads(stream.readline()).get("schema") == hs.SCHEMA
        except (OSError, ValueError, AttributeError):
            continue
        if v8:
            break
    return hs.TIERS, v8


def delivery_index(specs_root: str | None = None) -> dict[tuple[str, str, int], dict[str, Any]]:
    """(task, tier, seed) → {row, builder_episode}；builder 号与 hard_builder 相同：xhard0 前置局数（开关开 12、关 0）在前，
    之后档序主序、档内 candidate 升序（v7 方案第二部分 §7.2）。v8 规格按 ``TIERS`` 读五档。"""
    hs = _hs_light()
    tiers, v8 = specs_tiers(specs_root)
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    offsets: dict[str, int] = collections.Counter({task: _xhard0_prefix(hs) for task in hs.ALL_TASKS})
    for tier in tiers:
        path = hs.specs_root(specs_root) / tier / "specs.jsonl"
        if v8 and not path.is_file():
            continue  # v8 局部根（冒烟／分片）只含部分档；完整性由 reset-replay 的 43 格判据与 delivery-set 负责
        # 配额上限格表按 header 推出（V9 文件 → V9_CELLS）
        _, rows = _hp().load_specs_any(path, hs)
        for task in hs.ALL_TASKS:
            chosen = sorted((r for r in rows if r["task"] == task and hs.delivered(r)), key=lambda r: r["candidate"])
            for row in chosen:
                index[(task, tier, int(row["seed"]))] = {"row": row, "builder_episode": offsets[task]}
                offsets[task] += 1
    return index


# ── reset-replay / eval-smoke（GPU）────────────────────────────────────────


def _replay_targets(specs_root: str | None) -> list[dict[str, Any]]:
    """每格 candidate 最小的正式局 → builder episode 号（v8／v9 43 格），带该行 ``spec_sha256``。"""
    tiers, _ = specs_tiers(specs_root)
    first: dict[tuple[str, str], dict[str, Any]] = {}
    for (task, tier, seed), hit in delivery_index(specs_root).items():
        key = (task, tier)
        if key not in first or hit["row"]["candidate"] < first[key]["candidate"]:
            first[key] = {"task": task, "tier": tier, "seed": seed, "candidate": int(hit["row"]["candidate"]),
                          "spec_sha256": hit["row"].get("spec_sha256"), "builder_episode": hit["builder_episode"]}
    return [first[k] for k in sorted(first, key=lambda k: (tiers.index(k[1]), k[0]))]


def replay_key(record: dict[str, Any]) -> tuple:
    """reset-replay 续跑与取数的键：(task, tier, seed, spec_sha256)。v9 MoveCube 与 v8 同 seed、布局不同（1002 方案
    审计 9），只按 (task, tier, seed) 会把 v8 旧记录当成已完成；旧记录没有 ``spec_sha256`` 字段即永不命中、重跑。"""
    return (record.get("task"), record.get("tier"), record.get("seed"), record.get("spec_sha256"))


def _replay_out_path(out: str) -> Path:
    """``--out`` 必须是 jsonl 文件路径：给目录（已存在的目录、或以 / 结尾）即响亮报错，不在目录里猜文件名。"""
    path = Path(out)
    if str(out).endswith(("/", "\\")) or path.is_dir():
        raise SystemExit(f"reset-replay 的 --out 必须是 jsonl 文件路径，收到目录：{out}"
                         "（如 artifacts/newtask-v9/gates/reset-replay.jsonl）")
    return path


def specs_version(specs_root: str | None) -> tuple[str, dict[tuple[str, str], int] | None]:
    """规格根（显式 > ``ROBOMME_HARD_SPECS_ROOT`` > 包内）的版本与完整交付格表：/4 根按 header 的逐任务配额推出
    （``hard_parity.root_cell_table``：完整 V9 根恰等于 V9_CELLS → v9）。非 /4 根（v7 口径已删）直接报错。"""
    hs = _hs_light()
    _, v8 = specs_tiers(specs_root)
    if not v8:
        raise SystemExit(f"规格根不是 {hs.SCHEMA}（v7 口径已于 1003 维护计划删除）：{hs.specs_root(specs_root)}")
    found = _hp().root_cell_table(hs.specs_root(specs_root), hs)
    return found if found is not None else (_hp().expected_version(hs), hs.EXPECTED_CELLS)


def replay_verdict(targets: list[dict[str, Any]], rows: list[dict[str, Any]], *, version: str,
                   table: dict[tuple[str, str], int] | None, limited: bool) -> tuple[bool, str]:
    """reset-replay 判定行（纯函数，便于夹具测试）：每个目标恰有一条记录、无错误、injected_mismatch／layout_drift／
    unused／layout_hit_bad 全 0、全部 replay；/4 规格（v8／v9）另要求 ``spec_bound``（绑定规格的 spec_sha256 与目标行
    相同的局数）= 目标数，未截断时目标数 = 完整格表格数（43）。"""
    good = [r for r in rows if r.get("ok")]
    injected = sum(r["binding"]["injected_mismatch"] for r in good)
    drift = sum(r["binding"].get("layout_drift", 0) for r in good)
    unused = sum(r["binding"]["unused"] for r in good)
    hit_bad = sum(1 for r in good if r["binding"].get("layered")
                  and r["binding"].get("layout_hit") != r["binding"].get("layout_paths_expected"))
    replay = sum(1 for r in good if r["binding"].get("mode") == "replay")
    spec_bound = sum(1 for r in good if r.get("spec_sha256") is not None
                     and r["binding"].get("spec_sha256") == r.get("spec_sha256"))
    errors = len(rows) - len(good)
    v4 = version in ("v8", "v9")
    # /4 未截断时每格 1 局必须恰为完整格表的格数（43），缺档文件不得静默少测
    cells_ok = not v4 or limited or (table is not None and len(targets) == len(table))
    ok = len(rows) == len(targets) and errors == 0 and injected == drift == unused == hit_bad == 0 \
        and replay == len(targets) and cells_ok and (not v4 or spec_bound == len(targets))
    shape = f"cells{len(targets)}"
    line = (f"{version.upper()}_RESET_REPLAY={'PASS' if ok else 'FAIL'} shape={shape} resets={len(rows)} replay={replay} "
            f"injected_mismatch={injected} layout_drift={drift} spec_bound={spec_bound} unused={unused} "
            f"layout_hit_bad={hit_bad} errors={errors}")
    return ok, line


def cmd_reset_replay(args) -> int:
    """V9_RESET_REPLAY（v9 /4 规格，格表按 header 推出为 V9_CELLS）／V8_RESET_REPLAY（v8 /4）：
    每格 1 局经评估链 make_env_for_episode + reset，spec_binding 须 injected_mismatch==0、layout_drift==0、unused==0，
    绑定规格的 spec_sha256 等于目标行（spec_bound），派生局 layout_hit == layout_paths_hit 条数。

    ``--out`` 必填且必须是文件（v9 用独立文件 ``artifacts/newtask-v9/gates/reset-replay.jsonl``）；续跑键
    (task, tier, seed, spec_sha256)，同 seed 不同规格的旧记录不算已完成、也不进本轮判定。"""
    import os

    out_path = _replay_out_path(args.out)
    if args.specs_root:
        os.environ[_hard_specs().SPECS_ROOT_ENV] = str(Path(args.specs_root).resolve())
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder, spec_binding

    targets = _replay_targets(args.specs_root)
    if args.limit:
        targets = targets[: args.limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {replay_key(r) for r in map(json.loads, out_path.read_text().splitlines()) if r.get("ok") is not None}
    builders: dict[str, Any] = {}
    for target in targets:
        key = replay_key(target)
        if key in done:
            continue
        started = time.time()
        record = dict(target)
        env = None
        try:
            builder = builders.setdefault(target["task"], BenchmarkEnvBuilder(target["task"], dataset="ood"))
            identity = builder.resolve_identity(target["builder_episode"])
            assert identity["seed"] == target["seed"] and identity["tier"] == target["tier"], identity
            assert identity.get("spec_sha256") == target["spec_sha256"], (identity, target["spec_sha256"])
            env = builder.make_env_for_episode(target["builder_episode"], max_steps=_cap_for(target["tier"]))
            obs, info = env.reset()
            record.update({"binding": spec_binding(env), "identity": identity, "ok": True,
                           "demo_frames": len(obs.get("front_rgb_list", [])) - 1 if isinstance(obs, dict) else None})
        except Exception as exc:  # noqa: BLE001 如实记录
            record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:800]})
        finally:
            if env is not None:
                env.close()
        record["wall_s"] = round(time.time() - started, 1)
        with out_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"RESET_REPLAY {key[:3]} ok={record['ok']} binding={record.get('binding')}", flush=True)
    want = {replay_key(t) for t in targets}
    rows = [r for r in map(json.loads, out_path.read_text().splitlines()) if replay_key(r) in want]
    version, table = specs_version(args.specs_root)
    ok, line = replay_verdict(targets, rows, version=version, table=table, limited=bool(args.limit))
    print(line, flush=True)
    return 0 if ok else 1


def expected_episodes(task: str, hs, cells: dict[tuple[str, str], int] | None = None) -> int:
    """builder 每任务局数＝xhard0 前置局数（开关开 12、关 0）+ 交付格表在该任务的局数之和（不写死）。

    按 ``cells``（缺省当前 ``EXPECTED_CELLS``；V9_CELLS：每任务 50 + xhard0 前置局数）。v7 规格的 92／32 口径与
    V8 1070 局表的口径已删。"""
    table = cells if cells is not None else hs.EXPECTED_CELLS
    return _xhard0_prefix(hs) + sum(n for (name, _), n in table.items() if name == task)


def cmd_eval_smoke(args) -> int:
    """合作者入口：与 scripts/evaluation_hard.py 同样的构建，只跑 1 任务 × 1 档 × 1 局；逐局 max_steps 按 :func:`_cap_for`
    取（xhard0 1300、新值档 1600）。
    xhard0 局须为导出模式（原生 hard 分支）、回注局须为 replay 且 injected_mismatch==0。"""
    import os

    import numpy as np

    if args.specs_root:
        os.environ[_hard_specs().SPECS_ROOT_ENV] = str(Path(args.specs_root).resolve())
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder, spec_binding

    hs = _hard_specs()
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset="ood", action_space="joint_angle", max_steps=1300)
    num = builder.get_episode_num()
    seed, tier = builder.resolve_episode(args.episode)
    env = builder.make_env_for_episode(args.episode, max_steps=_cap_for(tier))
    obs, info = env.reset()
    binding = spec_binding(env)
    base = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0], dtype=np.float32)
    steps, status = 0, "unknown"
    while True:
        obs, reward, terminated, truncated, info = env.step(base)
        steps += 1
        if info is not None and info.get("status") == "error":
            status = "error"
            break
        if terminated or truncated:
            status = info.get("status", "unknown")
            break
        if steps >= args.max_policy_steps:
            status = "smoke_cut"
            break
    env.close()
    if tier == hs.XHARD0:
        mode_ok = binding.get("mode") == "export" and binding.get("spec_kind") == "native-parity/1"
    else:
        mode_ok = binding.get("mode") == "replay"
    # 局数按 builder 实际读的规格根推格表（v9 根 → V9_CELLS）
    expected = expected_episodes(args.task, hs, specs_version(args.specs_root)[1])
    ok = mode_ok and binding.get("injected_mismatch") == 0 and status != "error" and num == expected
    print(f"HARD_EVAL_SMOKE={'PASS' if ok else 'FAIL'} task={args.task} episode={args.episode} tier={tier} seed={seed} "
          f"episodes={num} max_steps={_cap_for(tier)} mode={binding.get('mode')} status={status} steps={steps} "
          f"injected_mismatch={binding.get('injected_mismatch')} goal={str(info.get('task_goal'))[:60] if info else None}")
    return 0 if ok else 1


# ── xhard0：reset 层对拍（判定）──────────────────────────────────────────────


_PROBE = r'''
import json, sys, importlib
side, task, src, eps, gpu = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), sys.argv[5]
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
sys.path = [p for p in sys.path if not os.path.isfile(os.path.join(p, "robomme", "__init__.py"))]
sys.path.insert(0, os.path.join(src, "src"))
import numpy as np, gymnasium as gym, hashlib
calls = []
_orig = gym.make
def _make(env_id, **kw):
    calls.append({"env_id": env_id, **{k: v for k, v in kw.items() if k not in ("native_episode_spec", "sampling_config")},
                  "has_sampling_config": "sampling_config" in kw, "has_native_episode_spec": "native_episode_spec" in kw})
    return _orig(env_id, **kw)
gym.make = _make
if side == "official":
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    assert "robomme_hard" not in sys.modules
    builder = BenchmarkEnvBuilder(task, dataset="test")
else:
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(task, dataset="ood")
def digest(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: digest(v) for k, v in sorted(x.items())}
    arr = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(arr.tobytes() + str(arr.dtype).encode() + str(arr.shape).encode()).hexdigest()
out = []
for ep in eps:
    calls.clear()
    seed, diff = builder.resolve_episode(ep)
    env = builder.make_env_for_episode(ep, max_steps=1300, include_available_multi_choices=True)
    kw = dict(calls[-1]); make_kw = {k: v for k, v in kw.items()}
    base = _orig(kw["env_id"], **{k: v for k, v in kw.items() if k not in ("env_id", "has_sampling_config", "has_native_episode_spec")})
    base.reset()  # 与评估链一致：seed 已随 gym.make 传入，reset 不另传（用户 2026-09-29 批准每局多 1 次底层 reset）
    pre = digest(base.unwrapped.get_state_dict())
    base.close()
    chain, e = [], env
    while hasattr(e, "env"):
        chain.append(type(e).__name__); e = e.env
    chain.append(type(e.unwrapped).__name__)
    obs, info = env.reset()
    goal = info.get("task_goal"); choices = info.get("available_multi_choices")
    demo = obs.get("front_rgb_list", []) if isinstance(obs, dict) else []
    out.append({"episode": ep, "seed": seed, "difficulty": diff, "make_kwargs": make_kw, "wrapper_chain": chain,
                "pre_demo_state": pre, "task_goal": [str(g) for g in (goal if isinstance(goal, (list, tuple)) else [goal])],
                "choices": json.loads(json.dumps(choices, default=str)) if choices is not None else None,
                "demo_frames": max(len(demo) - 1, 0), "demo_digest": digest(np.stack([np.asarray(f) for f in demo])) if len(demo) else None,
                "post_state": digest(env.unwrapped.get_state_dict())})
    env.close()
print("PROBE_JSON " + json.dumps(out))
'''


def _run_probe(side: str, task: str, src: Path, eps: list[int], gpu: str) -> list[dict[str, Any]]:
    import subprocess

    proc = subprocess.run([sys.executable, "-c", _PROBE, side, task, str(src), json.dumps(eps), gpu],
                          capture_output=True, text=True, cwd=str(REPO))
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("PROBE_JSON ")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError(f"{side}/{task} 探针失败：{proc.stderr[-1500:]}")
    return json.loads(line[len("PROBE_JSON "):])


def cmd_xhard0_reset_parity(args) -> int:
    """XHARD0_RESET_PARITY（D-16）：同卡两进程——官方侧只导入 robomme（dataset="test"，原 episode 号），
    robomme_hard 侧 dataset="ood" episode 0～11。确定性层逐位比：gym.make 实参（除 hard 侧不应有的键）、
    包装链类名序列、演示回放前的底层状态（另起底层环境 reset 取 get_state_dict）、seed、task_goal、多选项；
    演示层（演示帧数、演示帧、演示后状态）只报告。开关 ``XHARD0_IN_TEST_HARD`` 关闭时 builder 无 xhard0，直接报错。"""
    hs = _hard_specs()
    _require_xhard0_in_test_hard(hs)
    manifest = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    tasks = [t for t in hs.ALL_TASKS if not args.tasks or t in args.tasks.split(",")]
    for task in tasks:
        episodes = [r["episode"] for r in manifest["rows"] if r["task"] == task]
        official = _run_probe("official", task, Path(args.src_root), episodes, args.gpu)
        hard = _run_probe("hard", task, REPO, list(range(len(episodes))), args.gpu)
        for o, h in zip(official, hard):
            fields = {
                "seed": o["seed"] == h["seed"],
                "difficulty": o["difficulty"] == "hard" and h["difficulty"] == "xhard0",
                "make_kwargs": o["make_kwargs"] == h["make_kwargs"],
                "wrapper_chain": o["wrapper_chain"] == h["wrapper_chain"],
                "pre_demo_state": o["pre_demo_state"] == h["pre_demo_state"],
                "task_goal": o["task_goal"] == h["task_goal"],
                "choices": o["choices"] == h["choices"],
            }
            bad = [k for k, v in fields.items() if not v]
            # 演示前状态确有实体改名（键集合不同）、且每类 actor 的状态值集合相等 ⇒ 只是命名不同（如 robomme_hard BUS
            # 的 F3 左右按钮改名），场景逐位相同：单列 name_only，不计 det_diff（v7 方案 D-16 的实现细节，写进留档）。
            # 键集合相同而取值不同（如两个同形 actor 互换位姿）是真差异，与 env-digest-compare 的判法一致；
            # 唯一例外是 XHARD0_DECLARED_RENAMES 里逐名声明的对调，须按映射改名后逐键相等
            name_only = bad == ["pre_demo_state"] and _pre_state_name_only(
                o["pre_demo_state"], h["pre_demo_state"], XHARD0_DECLARED_RENAMES.get(task))
            rows.append({"task": task, "source_episode": o["episode"], "hard_episode": h["episode"],
                         "det_bad": [] if name_only else bad, "name_only": name_only,
                         "demo_frames": [o["demo_frames"], h["demo_frames"]],
                         "demo_equal": o["demo_digest"] == h["demo_digest"], "post_equal": o["post_state"] == h["post_state"],
                         "pre_state": [o["pre_demo_state"], h["pre_demo_state"]]})
        mine = [r for r in rows if r["task"] == task]
        print(f"XHARD0_RESET_TASK {task} compared={len(mine)} det_bad={sum(bool(r['det_bad']) for r in mine)} "
              f"name_only={sum(r['name_only'] for r in mine)}", flush=True)
    if args.merge_with:
        rows += [r for r in map(json.loads, Path(args.merge_with).read_text().splitlines()) if r["task"] not in tasks]
    (out / args.report_name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return _xhard0_reset_verdict(rows, hs.XHARD0_PER_TASK, n_tasks=len(hs.ALL_TASKS))


def _name_agnostic(state: Any) -> Any:
    """状态摘要按类（actors／articulations…）取值的有序多重集，忽略实体名字。"""
    if isinstance(state, dict) and all(isinstance(v, dict) for v in state.values()):
        return {section: sorted(json.dumps(v, sort_keys=True) for v in items.values()) for section, items in state.items()}
    return state


#: 已声明的实体改名：任务 → 分区 → {官方侧名: robomme_hard 侧名}。ButtonUnmaskSwap 的 F3 把左右按钮名对调
#: （官方 buttons[0]（y=-0.1）名 button_left，robomme_hard 名 button_right），两侧键集合相同、取值互换，
#: 只能靠逐名声明认定为改名；未声明的同键互换一律计真差异
XHARD0_DECLARED_RENAMES: dict[str, dict[str, dict[str, str]]] = {
    "ButtonUnmaskSwap": {"articulations": {"button_left": "button_right", "button_right": "button_left"}},
}


def _check_renames_bijective(renames: dict[str, dict[str, dict[str, str]]]) -> None:
    """每个分区的改名映射必须是该分区实体名上的置换（双射、值集合 = 键集合），否则抛 ValueError。"""
    for task, sections in renames.items():
        for section, mapping in sections.items():
            values = list(mapping.values())
            if len(set(values)) != len(values) or set(values) != set(mapping):
                raise ValueError(f"XHARD0_DECLARED_RENAMES[{task!r}][{section!r}] 不是置换：{mapping}")


# 加载即校验：非置换映射（如两个名改成同一个名、只改一边）会把两个实体并成一个或凭空造名，改名后「逐键相等」失去意义
_check_renames_bijective(XHARD0_DECLARED_RENAMES)


def _apply_renames(state: Any, renames: dict[str, dict[str, str]] | None) -> Any:
    """按声明映射给官方侧状态摘要的实体改名（只动映射里点名的分区与实体）。"""
    if not renames or not (isinstance(state, dict) and all(isinstance(v, dict) for v in state.values())):
        return state
    return {section: {renames.get(section, {}).get(name, name): value for name, value in items.items()}
            for section, items in state.items()}


def _state_names(state: Any) -> Any:
    """状态摘要的键集合（分区 → 实体名集合）；非「分区 → 实体 → 值」形态时原样返回。"""
    if isinstance(state, dict) and all(isinstance(v, dict) for v in state.values()):
        return {section: sorted(items) for section, items in state.items()}
    return state


def _pre_state_name_only(a: Any, b: Any, renames: dict[str, dict[str, str]] | None = None) -> bool:
    """两侧演示前状态「只是实体改名」，满足其一即可：
    ① 给了声明映射 ``renames`` 且官方侧 ``a`` 按映射改名后与 ``b`` 逐键相等；
    ② 键集合确实不同（存在改名），且去名后各分区取值的有序多重集相同。
    键集合相同且无声明映射可解释时，任何取值差异都是真差异（同形 actor 互换位姿不得被当成改名）。"""
    if renames and _apply_renames(a, renames) == b:
        return True
    return _state_names(a) != _state_names(b) and _name_agnostic(a) == _name_agnostic(b)


def _xhard0_reset_verdict(rows: list[dict[str, Any]], per_task: int, n_tasks: int | None = None) -> int:
    """期望局数 = 任务清单长度 × 每任务局数；任务清单取 hard_specs.ALL_TASKS（调用方未给时现取），不写死 16。"""
    if n_tasks is None:
        n_tasks = len(_hard_specs().ALL_TASKS)
    det = [r for r in rows if r["det_bad"]]
    first = f"{det[0]['task']}/ep{det[0]['source_episode']}:{det[0]['det_bad']}" if det else "-"
    name_only = [r for r in rows if r.get("name_only")]
    ok = not det and len(rows) == n_tasks * per_task
    print(f"XHARD0_RESET_PARITY={'PASS' if ok else 'FAIL'} shape={n_tasks}x1x{per_task} compared={len(rows)} "
          f"det_diff={len(det)} "
          f"name_only={len(name_only)} first_det_diff={first}"
          + (f" name_only_tasks={sorted({r['task'] for r in name_only})}" if name_only else ""))
    print(f"XHARD0_DEMO_DIFF=INFO frames_equal={sum(r['demo_frames'][0] == r['demo_frames'][1] for r in rows)} "
          f"max_frame_diff={max((abs(r['demo_frames'][0] - r['demo_frames'][1]) for r in rows), default=0)} "
          f"demo_equal={sum(r['demo_equal'] for r in rows)} post_equal={sum(r['post_equal'] for r in rows)}")
    return 0 if ok else 1


# ── V8 守卫（1001 方案第一部分 §3 验收表、第二部分 §2.3 闸门总表）：只读、纯 CPU ───────────────────────

#: 位置类子树（布局比较取材）：spec 里记录摆放位置、初始化位姿、路径节点、外环容器的分支
POSITION_SUBTREES = ("layout", "initializations", "actions.path_nodes", "actions.nodes", "objects.distractors.bins")
#: 整数路径序列里唯一参与判定的一条：PatternLock 的图案节点（5×5 网格），按前缀判
PATH_NODES = "actions.path_nodes"
#: 两条路径共同前缀至少这么长才算照抄（v8 PatternLock 最短 9 节点；独立抽样首 9 节点全同的概率约 25⁻¹×5⁻⁸≈1e-7）
PATH_NODES_MIN_PREFIX = 9


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _position_leaves(tree: Any, prefix: str) -> dict[str, Any]:
    """位置类叶子：数值列表（坐标、位姿、路径节点序列）与浮点标量；整数／布尔计数、字符串、颜色名等非位置叶子丢弃。
    列表的列表逐元素展开（``<路径>.<下标>``），所以前缀派生的低档与高档在同名叶子上可直接比较。"""
    out: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(_position_leaves(value, f"{prefix}.{key}"))
    elif isinstance(tree, list) and tree and all(_is_number(v) for v in tree):
        out[prefix] = tree
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            out.update(_position_leaves(value, f"{prefix}.{index}"))
    elif isinstance(tree, float):
        out[prefix] = tree
    return out


def position_leaves(spec: dict[str, Any]) -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for path in POSITION_SUBTREES:
        node: Any = spec
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is not None:
            leaves.update(_position_leaves(node, path))
    return leaves


def _has_float(value: Any) -> bool:
    return isinstance(value, float) or (isinstance(value, list) and any(isinstance(v, float) for v in value))


def layout_overlap(items: list[tuple[str, dict[str, Any], bool]]) -> dict[str, Any]:
    """同一任务的跨档位置照抄检测（逐叶子，v8 方案第一部分 §3「布局独立」）。

    ``items`` = 该任务全部规格行 ``(tier, row, 是否交付)``（含未交付行，用来识别恒定叶子）。口径：

    1. 每行取 :func:`position_leaves`；
    2. 剔除恒定叶子：在该任务全部规格行里出现 ≥ 2 次且取值处处相同的路径（如 ``layout.cube_min_center_dist``、
       固定的 ``layout.goal_xy``、``layout.demo.region.*``）——它们是配置常量，不是抽出来的位置；
    3. 剩下的叶子只保留**浮点**叶子（浮点标量或含浮点的列表），外加 PatternLock 的 ``actions.path_nodes``；
       其余整数列表（如 RouteStick 的 ``actions.nodes``，取值只有几个格点）不参与判定，免得独立抽样偶然相同；
    4. 对每一对「不同档的交付行」：两行共有的浮点叶子中任一**逐位相等**，或两条 ``path_nodes`` 有长度
       ≥ :data:`PATH_NODES_MIN_PREFIX` 的共同前缀（短者是长者前缀），即记一对 ``layout_equal_pairs``。
       理由：独立抽样的浮点坐标逐位相等的概率≈0，而 v7 式派生（低档照抄高档母布局或其前缀）必然在同名叶子上逐位相等。

    返回 ``{"pairs", "no_position", "constant", "detail"}``；跨档交付行剔除后没有任何可比叶子记 ``no_position``。
    """
    leaves = [(tier, row, delivered, position_leaves(row.get("spec") or {})) for tier, row, delivered in items]
    seen: dict[str, set[str]] = collections.defaultdict(set)
    count: collections.Counter = collections.Counter()
    for _, _, _, row_leaves in leaves:
        for path, value in row_leaves.items():
            seen[path].add(json.dumps(value, sort_keys=True))
            count[path] += 1
    constant = {path for path, values in seen.items() if count[path] >= 2 and len(values) == 1}
    comparable = []
    for tier, row, delivered, row_leaves in leaves:
        if not delivered:
            continue
        floats = {p: json.dumps(v) for p, v in row_leaves.items() if p not in constant and _has_float(v)}
        nodes = row_leaves.get(PATH_NODES) if PATH_NODES not in constant else None
        comparable.append((tier, row, floats, nodes if isinstance(nodes, list) else None))
    tiers = {tier for tier, *_ in comparable}
    no_position = sum(1 for _, _, floats, nodes in comparable if not floats and nodes is None) if len(tiers) > 1 else 0
    pairs, detail = 0, []
    for i, (tier_a, row_a, floats_a, nodes_a) in enumerate(comparable):
        for tier_b, row_b, floats_b, nodes_b in comparable[i + 1:]:
            if tier_a == tier_b:
                continue
            hit = next((p for p in floats_a.keys() & floats_b.keys() if floats_a[p] == floats_b[p]), None)
            if hit is None and nodes_a is not None and nodes_b is not None:
                n = min(len(nodes_a), len(nodes_b))
                if n >= PATH_NODES_MIN_PREFIX and nodes_a[:n] == nodes_b[:n]:
                    hit = PATH_NODES
            if hit is not None:
                pairs += 1
                if len(detail) < 6:
                    detail.append(f"{row_a['task']}:{tier_a}/{row_a['candidate']}={tier_b}/{row_b['candidate']}@{hit}")
    return {"pairs": pairs, "no_position": no_position, "constant": sorted(constant), "detail": detail}


def _read_v8_root(specs_root: str | Path, cells: dict[tuple[str, str], int]) -> tuple[dict[str, tuple], list[str], list[str]]:
    """格表涉及各档的 /4 文件原样读出（不校验，校验另走 ``load_specs_root``），使校验失败时计数仍可产出。
    返回 ``(files, missing, bad)``：缺文件记 ``missing``；空文件、坏 JSON、首行不是 header 记 ``bad``（不抛异常）。"""
    hs = _hs_light()
    files: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    missing: list[str] = []
    bad: list[str] = []
    for tier in hs.TIERS:
        if not any(t == tier for _, t in cells):
            continue
        path = Path(specs_root) / tier / "specs.jsonl"
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            records = hs.read_jsonl(path)
            if not records or records[0].get("record") != "header":
                raise ValueError("空文件或首行不是 header")
        except Exception as exc:  # noqa: BLE001 坏文件计数，不中断判定行
            bad.append(f"{path}: {type(exc).__name__}: {exc}"[:300])
            continue
        files[tier] = (records[0], records[1:])
    return files, missing, bad


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n")


def _delivery_sources(path: str | Path) -> tuple[set[tuple[str, str, int]], dict[str, int], int]:
    """交付清单（v8-delivery/1；v9 为 ``v9_subset_specs.py assemble`` 的 800 行清单）→ (身份集合, 来源计数, 坏行数)。
    来源计数只在行带 ``source`` 字段时给出 ``{"v9-new": n, "v8-reuse": n}``（没有该字段返回空字典，判定行不打
    new／reused）；``source`` 不是这两个值之一、或只有部分行带该字段，计坏行。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = (payload.get("rows") or payload.get("delivered") or []) if isinstance(payload, dict) else payload
    ids: set[tuple[str, str, int]] = set()
    bad = valid = 0
    sources: collections.Counter = collections.Counter()
    with_source = sum(isinstance(r, dict) and "source" in r for r in items)
    for r in items:
        try:
            ids.add((r["task"], r.get("tier", r.get("difficulty")), int(r["seed"])))
            valid += 1
        except (KeyError, TypeError, ValueError, AttributeError):
            bad += 1
            continue
        if with_source:
            if r.get("source") in ("v9-new", "v8-reuse"):
                sources[r["source"]] += 1
            else:
                bad += 1
    counts = {"v9-new": sources["v9-new"], "v8-reuse": sources["v8-reuse"]} if with_source else {}
    return ids, counts, bad + (valid - len(ids))  # 重复身份也计坏行


def cmd_delivery_set(args) -> int:
    """{V8|V9}_DELIVERY_SET／_SEED_DISJOINT／_LAYOUT_INDEPENDENT（只读）。判定行前缀按格表版本：``--cells`` 解析为
    V9 表（``v9full``／``v9``／``v9smoke``／``v9shard1``，或 3b 切换后的 ``full``，或能被 V9_CELLS 覆盖的 JSON）
    打 ``V9_*``；校验的配额上限格表随之取 V9_CELLS（V8 1070 局表已于维护计划阶段 1b 删除）。

    * ``--delivery``（v9 阶段 3b 必给 assemble 的 800 行清单）：清单身份 (task, tier, seed) 必须与规格根的交付行逐一
      相同（对称差计 ``delivery_mismatch``），行带 ``source``（``v8-reuse|v9-new``）时判定行在 total 后打
      ``new=<v9-new 行数> reused=<v8-reuse 行数>``（两者之和须等于 total）；重复身份、缺键、``source`` 值不合法或
      只有部分行带 ``source`` 计 ``delivery_bad_rows``；不给 ``--delivery`` 则不打这两项。

    * 交付形态：先过 ``load_specs_root(root, cells)`` 全部校验（失败计 ``load_errors``），再逐格数 ``delivered``
      （selected 且 rollout ok），与格表**相等**比较（不是 ≤）；格表外有交付行、selected 而未成功、交付行执行步超
      ``EXEC_CAP`` 都判 FAIL。另报 /4 结果段里的 ``failed``／``exec_over_cap``／``backfills``（全部显式写零）。
    * seed 按档隔离：同任务不同档的 seed 集合（全部规格行，不只交付行）两两求交，``shared`` 为交集元素总数。
    * 布局独立：header ``layout_rule == {"mode": "independent"}``（``mode_bad``）、行 ``layout_parent`` 全空
      （``parent_non_null``），**并且**同任务跨档交付行逐叶子比较（:func:`layout_overlap`：剔除恒定叶子后任一浮点
      位置叶子逐位相等，或 PatternLock 路径有 ≥ 9 节点的共同前缀）的对数 ``layout_equal_pairs`` 为 0；只查前两项是
      同义反复（冻结器自己写的标志）。跨档任务的交付行剔除后没有任何可比叶子记 ``no_position``，同样判 FAIL。
    * 健壮性：缺档文件计 ``missing_files``；空文件、坏 JSON、缺键的行与 ``load_specs_root`` 抛出的异常（如某格备用候选
      耗尽、selected 少于期望）一律计入 ``load_errors``，照常逐格计数并打印三行判定，不崩溃。
    * 计数口径：``failed``／``exec_over_cap``／``backfills`` 取自 /4 行的结果段；``infra_retries`` 不落在规格行里，
      只由 S2-B ``_rollout`` 聚合步的 ``V8_DELIVERY_SET`` 行与 ``delivery.json`` 给出，本命令不报。
    """
    hp, hs = _hp(), _hs_light()
    cells, version = hp.parse_cells_versioned(args.cells, hs)
    prefix = version.upper()
    load_error = None
    try:
        hs.load_specs_root(args.specs_root, cells, cell_table=hs.CELL_TABLES[version], check_fingerprint=False)
    except Exception as exc:  # noqa: BLE001 校验失败如实计入判定，不中断计数
        load_error = f"{type(exc).__name__}: {exc}"
    files, missing_files, bad_files = _read_v8_root(args.specs_root, cells)
    bad_rows: list[str] = []
    delivered: collections.Counter = collections.Counter()
    counts = {"failed": 0, "exec_over_cap": 0, "backfills": 0, "selected_not_ok": 0, "delivered_over_cap": 0}
    seeds: dict[str, dict[str, set[int]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    parent_non_null = 0
    by_task: dict[str, list[tuple[str, dict[str, Any], bool]]] = collections.defaultdict(list)
    delivered_ids: list[tuple[str, str, int]] = []
    for tier, (_, rows) in files.items():
        for row in rows:
            try:
                seed = int(row["seed"])
                task = row["task"]
                rollout = row.get("rollout") or {}
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                bad_rows.append(f"{tier}:{type(exc).__name__}: {exc}"[:200])
                continue
            seeds[task][tier].add(seed)
            by_task[task].append((tier, row, hs.delivered(row)))
            parent_non_null += int(row.get("layout_parent") is not None)
            if rollout.get("status") == "failed":
                counts["failed"] += 1
                counts["exec_over_cap"] += int(rollout.get("error_type") == "exec_over_cap")
            if hs.delivered(row):
                delivered[(row["task"], tier)] += 1
                delivered_ids.append((row["task"], tier, seed))
                counts["backfills"] += int(not row.get("initial_selected"))
                exec_steps = rollout.get("exec_steps")
                counts["delivered_over_cap"] += int(_is_number(exec_steps) and exec_steps > hs.EXEC_CAP)
            elif row.get("selected"):
                counts["selected_not_ok"] += 1
    cell_mismatch = [f"{t}/{tier}:{delivered.get((t, tier), 0)}/{n}" for (t, tier), n in sorted(cells.items())
                     if delivered.get((t, tier), 0) != n]
    extra_cells = sorted(f"{t}/{tier}" for (t, tier) in delivered if (t, tier) not in cells)
    total = sum(delivered.values())
    load_errors = int(load_error is not None) + len(bad_files) + len(bad_rows)
    manifest = None
    if args.delivery:
        try:
            ids, sources, bad = _delivery_sources(args.delivery)
            mismatch = len(ids ^ set(delivered_ids))
            manifest = {"rows": len(ids), "sources": sources, "bad_rows": bad, "mismatch": mismatch}
        except (OSError, ValueError) as exc:
            manifest = {"rows": 0, "sources": {}, "bad_rows": 1, "mismatch": total, "error": f"{type(exc).__name__}: {exc}"}
    manifest_ok = manifest is None or (manifest["bad_rows"] == 0 and manifest["mismatch"] == 0
                                       and (not manifest["sources"] or sum(manifest["sources"].values()) == total))
    ok_set = (load_errors == 0 and not missing_files and not cell_mismatch and not extra_cells
              and counts["selected_not_ok"] == 0 and counts["delivered_over_cap"] == 0
              and total == sum(cells.values()) > 0 and manifest_ok)
    source_text = (f"new={manifest['sources']['v9-new']} reused={manifest['sources']['v8-reuse']} "
                   if manifest is not None and manifest["sources"] else "")
    manifest_text = ("" if manifest is None else
                     f" delivery_rows={manifest['rows']} delivery_mismatch={manifest['mismatch']} "
                     f"delivery_bad_rows={manifest['bad_rows']}")
    set_line = (f"{prefix}_DELIVERY_SET={'PASS' if ok_set else 'FAIL'} tasks={len({t for t, _ in delivered})} "
                f"cells={len(delivered)} total={total} {source_text}expected_cells={len(cells)} "
                f"expected_total={sum(cells.values())} "
                f"cell_mismatch={len(cell_mismatch)} extra_cells={len(extra_cells)} "
                f"selected_not_ok={counts['selected_not_ok']} delivered_over_cap={counts['delivered_over_cap']} "
                f"failed={counts['failed']} exec_over_cap={counts['exec_over_cap']} backfills={counts['backfills']} "
                f"missing_files={len(missing_files)} load_errors={load_errors}{manifest_text}"
                + ("" if ok_set else f" detail={(cell_mismatch + extra_cells + missing_files + bad_files + bad_rows)[:6]}"
                   f" load_error={load_error!r}"))
    tier_pairs = shared = 0
    shared_detail: list[str] = []
    for task in sorted(seeds):
        names = sorted(seeds[task], key=hs.TIERS.index)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                tier_pairs += 1
                common = seeds[task][a] & seeds[task][b]
                shared += len(common)
                if common:
                    shared_detail.append(f"{task}:{a}&{b}={len(common)}")
    ok_seed = shared == 0 and bool(seeds) and not missing_files and not bad_files and not bad_rows
    seed_line = (f"{prefix}_SEED_DISJOINT={'PASS' if ok_seed else 'FAIL'} tasks={len(seeds)} tier_pairs={tier_pairs} "
                 f"shared={shared}" + ("" if ok_seed else f" detail={shared_detail[:6]}"))
    mode_bad = sum(header.get("layout_rule") != hs.LAYOUT_RULE for header, _ in files.values())
    layout_equal_pairs = no_position = 0
    equal_detail: list[str] = []
    per_task_pairs: dict[str, int] = {}
    for task in sorted(by_task):
        result = layout_overlap(by_task[task])
        per_task_pairs[task] = result["pairs"]
        layout_equal_pairs += result["pairs"]
        no_position += result["no_position"]
        equal_detail += result["detail"]
    ok_layout = (bool(files) and not missing_files and not bad_files and not bad_rows and mode_bad == 0 and parent_non_null == 0
                 and layout_equal_pairs == 0 and no_position == 0)
    layout_line = (f"{prefix}_LAYOUT_INDEPENDENT={'PASS' if ok_layout else 'FAIL'} files={len(files)} delivered={total} "
                   f"parent_non_null={parent_non_null} layout_equal_pairs={layout_equal_pairs} mode_bad={mode_bad} "
                   f"no_position={no_position} load_errors={len(bad_files) + len(bad_rows)}"
                   + ("" if ok_layout else f" detail={equal_detail[:6]}"))
    for line in (set_line, seed_line, layout_line):
        print(line, flush=True)
    _write_report(args.out, {
        "cells": {f"{t}/{tier}": {"expected": n, "delivered": delivered.get((t, tier), 0)}
                  for (t, tier), n in sorted(cells.items())},
        "version": version, "delivery_manifest": manifest,
        "delivery_set": {"verdict": "PASS" if ok_set else "FAIL", "tasks": len({t for t, _ in delivered}),
                         "cells": len(delivered), "total": total, "expected_total": sum(cells.values()),
                         "cell_mismatch": len(cell_mismatch), "extra_cells": len(extra_cells),
                         "missing_files": len(missing_files), "load_errors": int(load_error is not None), **counts,
                         "detail": {"cell_mismatch": cell_mismatch, "extra_cells": extra_cells,
                                    "missing_files": missing_files, "load_error": load_error,
                                    "bad_files": bad_files, "bad_rows": bad_rows}},
        "seed_disjoint": {"verdict": "PASS" if ok_seed else "FAIL", "tasks": len(seeds), "tier_pairs": tier_pairs,
                          "shared": shared, "detail": shared_detail},
        "layout_independent": {"verdict": "PASS" if ok_layout else "FAIL", "files": len(files), "delivered": total,
                               "parent_non_null": parent_non_null, "layout_equal_pairs": layout_equal_pairs,
                               "mode_bad": mode_bad, "no_position": no_position, "per_task_pairs": per_task_pairs,
                               "detail": equal_detail},
        "lines": [set_line, seed_line, layout_line]})
    return 0 if ok_set and ok_seed and ok_layout else 1


def _tier_table(tiers: tuple[str, ...], **dims: tuple) -> dict[str, dict[str, Any]]:
    return {tier: {dim: values[i] for dim, values in dims.items()} for i, tier in enumerate(tiers)}


_X123 = ("xhard1", "xhard2", "xhard3")
_X1234 = ("xhard1", "xhard2", "xhard3", "xhard4")
_X12345 = ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5")
_X12 = ("xhard1", "xhard2")
#: v8 表 1（只含交付格）：{task: {tier: {维度: 定值 或 (lo, hi) 闭区间}}}；14 任务 41 格（MoveCube、InsertPeg 不计取值）
V8_TIER_TABLE: dict[str, dict[str, dict[str, Any]]] = {
    "PickXtimes": _tier_table(_X123, times=(6, 7, 8), distractors=(1, 2, 3)),
    "SwingXtimes": _tier_table(_X12345, rounds=(4, 5, 6, 7, 8), distractors=(1, 2, 3, 4, 4)),
    "StopCube": _tier_table(_X12345, stop_time=(6, 7, 8, 9, 10), move_interval=(60, 60, 60, 60, 60)),
    "VideoUnmask": _tier_table(_X1234, pick=(2, 3, 3, 3), distractor_bins=(4, 4, 8, 12), distractor_cubes=(2, 2, 4, 6)),
    "ButtonUnmask": _tier_table(_X1234, pick=(2, 3, 3, 3), distractor_bins=(4, 4, 8, 12), distractor_cubes=(2, 2, 4, 6)),
    "BinFill": _tier_table(_X12, put_in=(6, 7)),
    "VideoUnmaskSwap": _tier_table(_X12, swap=(5, 7), pick=(2, 3), outer=(2, 4)),
    "ButtonUnmaskSwap": _tier_table(_X12, swap=(3, 5), pick=(2, 3), outer=(2, 4)),
    "VideoPlaceButton": _tier_table(_X12, placements=(3, 4)),
    "VideoPlaceOrder": _tier_table(_X12, visits=(5, 6)),
    "PickHighlight": _tier_table(_X12, pick=(4, 5), total=(7, 8)),
    "VideoRepick": _tier_table(_X12, cubes=(4, 5), swap=(4, 6), repick=(2, 3)),
    "RouteStick": _tier_table(_X123, segments=((8, 10), (11, 13), (14, 16))),
    "PatternLock": _tier_table(_X123, nodes=((9, 12), (13, 15), (16, 18))),
}
#: 区间任务：生成后逐格报告长度直方图（PatternLock 区间内比例不保证均匀，只报告）
RANGE_DIMS = {"RouteStick": "segments", "PatternLock": "nodes"}


def _actual_int(value: Any, where: str) -> int:
    """取实际整数：``{actual|placed: n}`` 取实际值；否则须为非负整数。"""
    if isinstance(value, dict):
        value = value.get("actual", value.get("placed"))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{where} 不是非负整数：{value!r}")
    return value


def tier_dims(task: str, spec: dict[str, Any]) -> dict[str, int]:
    """从 /4 行 ``spec``（reset 后封存的 EpisodeSpec）读本局的实际取值，不读配置默认值。字段（v7 规格样例核实）：
    PickXtimes／SwingXtimes ``objects.num_repeats``、``objects.distractor_count.actual``；StopCube
    ``actions.stop_time``、``actions.move_interval``；VideoUnmask／ButtonUnmask ``objects.n_picks``、
    ``objects.distractors.placed``（干扰容器）、``objects.distractors.cube_count``（干扰方块）；两个 Swap 任务
    ``objects.n_swaps``、``objects.n_picks``、``objects.distractors.placed``（外圈干扰）；BinFill
    ``sum(objects.target_numbers)``；VideoPlaceButton ``actions.target_placement_count``；VideoPlaceOrder
    ``sum(objects.visit_counts_by_object)``（缺失时取 ``actions.target_placement_count``）；PickHighlight
    ``objects.highlight_count``、``objects.n_cubes_spawned``；VideoRepick ``objects.cube_count.actual``、
    ``objects.n_swaps``、``objects.num_repeats``；RouteStick ``objects.L``；PatternLock ``len(actions.path_nodes)``。"""
    objects, actions = spec.get("objects") or {}, spec.get("actions") or {}
    where = f"{task}.spec"
    if task in ("PickXtimes", "SwingXtimes"):
        return {"times" if task == "PickXtimes" else "rounds": _actual_int(objects["num_repeats"], f"{where}.objects.num_repeats"),
                "distractors": _actual_int(objects["distractor_count"], f"{where}.objects.distractor_count")}
    if task == "StopCube":
        return {"stop_time": _actual_int(actions["stop_time"], f"{where}.actions.stop_time"),
                "move_interval": _actual_int(actions["move_interval"], f"{where}.actions.move_interval")}
    if task in ("VideoUnmask", "ButtonUnmask"):
        dist = objects["distractors"]
        return {"pick": _actual_int(objects["n_picks"], f"{where}.objects.n_picks"),
                "distractor_bins": _actual_int(dist["placed"], f"{where}.objects.distractors.placed"),
                "distractor_cubes": _actual_int(dist["cube_count"], f"{where}.objects.distractors.cube_count")}
    if task in ("VideoUnmaskSwap", "ButtonUnmaskSwap"):
        return {"swap": _actual_int(objects["n_swaps"], f"{where}.objects.n_swaps"),
                "pick": _actual_int(objects["n_picks"], f"{where}.objects.n_picks"),
                "outer": _actual_int(objects["distractors"]["placed"], f"{where}.objects.distractors.placed")}
    if task == "BinFill":
        return {"put_in": sum(_actual_int(v, f"{where}.objects.target_numbers") for v in objects["target_numbers"])}
    if task == "VideoPlaceButton":
        return {"placements": _actual_int(actions["target_placement_count"], f"{where}.actions.target_placement_count")}
    if task == "VideoPlaceOrder":
        visits = objects.get("visit_counts_by_object")
        if isinstance(visits, list):
            return {"visits": sum(_actual_int(v, f"{where}.objects.visit_counts_by_object") for v in visits)}
        return {"visits": _actual_int(actions["target_placement_count"], f"{where}.actions.target_placement_count")}
    if task == "PickHighlight":
        return {"pick": _actual_int(objects["highlight_count"], f"{where}.objects.highlight_count"),
                "total": _actual_int(objects["n_cubes_spawned"], f"{where}.objects.n_cubes_spawned")}
    if task == "VideoRepick":
        return {"cubes": _actual_int(objects["cube_count"], f"{where}.objects.cube_count"),
                "swap": _actual_int(objects["n_swaps"], f"{where}.objects.n_swaps"),
                "repick": _actual_int(objects["num_repeats"], f"{where}.objects.num_repeats")}
    if task == "RouteStick":
        return {"segments": _actual_int(objects["L"], f"{where}.objects.L")}
    if task == "PatternLock":
        nodes = actions["path_nodes"]
        if not isinstance(nodes, list):
            raise ValueError(f"{where}.actions.path_nodes 不是列表")
        return {"nodes": len(nodes)}
    raise KeyError(f"表 1 不含任务 {task}")


def _value_ok(got: Any, want: Any) -> bool:
    if isinstance(want, tuple):
        return isinstance(got, int) and want[0] <= got <= want[1]
    return got == want


def cmd_tier_values(args) -> int:
    """V8_TIER_VALUES（只读，新写）：格表里有取值维度的格（14 任务；MoveCube、InsertPeg 不计）逐格逐局读实际取值
    （:func:`tier_dims`）与表 1（:data:`V8_TIER_TABLE`）比：定值逐档相等，RouteStick／PatternLock 落在区间内，并逐格
    打印长度直方图 ``V8_TIER_LENGTH_HIST=INFO``。``mismatches`` = 取值不符的局数 + 读取失败的局数 + 无行的格数。
    默认取交付行（selected 且 rollout ok）；``--selected`` 取 selected 行（生成前核对抽签结果）。"""
    hp, hs = _hp(), _hs_light()
    cells = hp.parse_cells(args.cells, hs)
    valued = {key: n for key, n in cells.items() if key[0] in V8_TIER_TABLE}
    files, missing_files, bad_files = _read_v8_root(args.specs_root, valued)
    missing_files = missing_files + bad_files  # 坏文件与缺文件同样计入 missing_files，判定行照常打印
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for tier, (_, rows) in files.items():
        for row in rows:
            if (row.get("task"), tier) in valued and (row.get("selected") if args.selected else hs.delivered(row)):
                by_cell[(row["task"], tier)].append(row)
    mismatches, checked = 0, 0
    detail: list[str] = []
    histograms: dict[str, dict[str, int]] = {}
    per_cell: dict[str, Any] = {}
    for (task, tier) in sorted(valued, key=lambda k: (hs.ALL_TASKS.index(k[0]), hs.TIERS.index(k[1]))):
        want = V8_TIER_TABLE[task][tier]
        rows = sorted(by_cell.get((task, tier), []), key=lambda r: int(r["candidate"]))
        hist: collections.Counter = collections.Counter()
        bad_rows = 0
        if not rows:
            mismatches += 1
            detail.append(f"{task}/{tier}:无行")
        for row in rows:
            checked += 1
            try:
                got = tier_dims(task, row.get("spec") or {})
            except (KeyError, TypeError, ValueError) as exc:
                mismatches += 1
                bad_rows += 1
                detail.append(f"{task}/{tier}/{row['candidate']}:读取失败 {exc}")
                continue
            wrong = {dim: (got.get(dim), value) for dim, value in want.items() if not _value_ok(got.get(dim), value)}
            if wrong:
                mismatches += 1
                bad_rows += 1
                detail.append(f"{task}/{tier}/{row['candidate']}:{wrong}")
            if task in RANGE_DIMS and isinstance(got.get(RANGE_DIMS[task]), int):
                hist[got[RANGE_DIMS[task]]] += 1
        per_cell[f"{task}/{tier}"] = {"rows": len(rows), "mismatches": bad_rows + int(not rows), "want": want}
        if task in RANGE_DIMS:
            lo, hi = want[RANGE_DIMS[task]]
            full = {str(v): hist.get(v, 0) for v in range(lo, hi + 1)}
            out_of_range = sum(c for v, c in hist.items() if not lo <= v <= hi)
            histograms[f"{task}/{tier}"] = {**full, **({"out_of_range": out_of_range} if out_of_range else {})}
            print(f"V8_TIER_LENGTH_HIST=INFO task={task} tier={tier} dim={RANGE_DIMS[task]} range=[{lo},{hi}] "
                  f"n={len(rows)} hist={json.dumps(full, separators=(',', ':'))} out_of_range={out_of_range}", flush=True)
    ok = mismatches == 0 and bool(valued) and not missing_files
    line = (f"V8_TIER_VALUES={'PASS' if ok else 'FAIL'} tasks={len({t for t, _ in valued})} cells={len(valued)} "
            f"mismatches={mismatches} rows={checked} missing_files={len(missing_files)} "
            f"source={'selected' if args.selected else 'delivered'}"
            + ("" if ok else f" detail={(missing_files + detail)[:6]}"))
    print(line, flush=True)
    _write_report(args.out, {"verdict": "PASS" if ok else "FAIL", "tasks": len({t for t, _ in valued}),
                             "cells": len(valued), "mismatches": mismatches, "rows": checked,
                             "missing_files": missing_files, "per_cell": per_cell, "histograms": histograms,
                             "detail": detail, "line": line})
    return 0 if ok else 1


def _h5_steps(path: Path) -> tuple[int, int]:
    """(演示帧数, 执行步数)：执行步 = ``timestep_*`` 个数 − ``info/is_video_demo`` 为真的帧数（v8 方案术语）。"""
    import h5py

    with h5py.File(path, "r") as handle:
        episode = handle[list(handle.keys())[0]]
        steps = [k for k in episode if k.startswith("timestep_")]
        demo = sum(bool(episode[k]["info/is_video_demo"][()]) for k in steps)
    return demo, len(steps) - demo


def pool_over_cap(paths: list[str | Path]) -> set[tuple]:
    """:func:`pool_scan` 的身份集合（不含读取错误）。"""
    return pool_scan(paths)[0]


def pool_scan(paths: list[str | Path]) -> tuple[set[tuple], list[str]]:
    """抽样阶段因执行步超限（``error_type == "exec_over_cap"``）被丢弃并递补的候选身份集合。

    ``--pool`` 可给多个：目录（递归读 ``*.jsonl`` 与 ``results.json``——覆盖 /4 规格行的 ``rollout.error_type``、
    各轮 ``results.partial.jsonl`` 的逐局结果）或单个 json／jsonl 文件。身份按 (task, tier, seed) 去重（没有 seed 的
    记录用候选号），同一候选在规格行与轮次结果里各出现一次只算一个。记录不带档名时取文件所在路径里最近的档目录名
    （``<档>/specs.jsonl``）。空文件、坏 JSON 不抛异常，逐文件计入返回的错误清单（调用方记 ``pool_load_errors``）。"""
    found: set[tuple] = set()
    errors: list[str] = []
    tiers = set(_hs_light().TIERS)

    def take(record: Any, tier_hint: str | None = None) -> None:
        if not isinstance(record, dict):
            return
        rollout = record.get("rollout") if isinstance(record.get("rollout"), dict) else {}
        if (record.get("error_type") or rollout.get("error_type")) != "exec_over_cap":
            return
        tier = record.get("tier", record.get("difficulty", tier_hint))
        seed = record.get("seed")
        found.add((record.get("task"), tier,
                   int(seed) if seed is not None else f"c{record.get('candidate', record.get('episode'))}"))

    for base in map(Path, paths):
        if not base.exists():
            errors.append(f"{base}: 不存在")
            continue
        files = (sorted(base.rglob("*.jsonl")) + sorted(base.rglob("results.json"))) if base.is_dir() else [base]
        for file in files:
            hint = next((part for part in reversed(file.parent.parts) if part in tiers), None)
            try:
                if file.suffix == ".jsonl":
                    for text in file.read_text(encoding="utf-8").splitlines():
                        if text.strip():
                            take(json.loads(text), hint)
                else:
                    payload = json.loads(file.read_text(encoding="utf-8"))
                    items = (payload.get("results") or payload.get("rows") or []) if isinstance(payload, dict) else payload
                    for record in items:
                        take(record, hint)
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"{file}: {type(exc).__name__}: {exc}"[:300])
    return found, errors


def _xhard0_h5s(source: str | Path) -> tuple[list[Path], int]:
    """xhard0 h5 来源（v8 交付清单不含 xhard0）：hard_parity 一侧目录（读其 ``identities.jsonl``）、identities
    jsonl 或带 ``rows`` 的 json；只取档为 xhard0／hard 且成功产出 h5 的行，路径相对文件所在目录。返回 (路径, 行数)。"""
    source = Path(source)
    file = source / "identities.jsonl" if source.is_dir() else source
    if file.suffix == ".jsonl":
        items = [json.loads(t) for t in file.read_text(encoding="utf-8").splitlines() if t.strip()]
    else:
        payload = json.loads(file.read_text(encoding="utf-8"))
        items = payload.get("rows", []) if isinstance(payload, dict) else payload
    rows = [r for r in items if r.get("tier", r.get("difficulty")) in (None, "xhard0", "hard")]
    paths = [Path(r["path"]) if Path(r["path"]).is_absolute() else file.parent / r["path"]
             for r in rows if r.get("path") and r.get("success", True)]
    return paths, len(rows)


def delivery_version(delivery: dict[str, Any]) -> str:
    """交付清单（``read_delivery`` 归一后）→ 版本：行带 ``source`` 字段（v9 assemble 的 800 行清单）即 v9；否则按逐格
    行数 ``hard_parity.cells_version``（登记的完整格表只剩 V9_CELLS，能被它覆盖的子表一律 → v9；V8 的 1070 局表已于
    维护计划阶段 1b 删除，原先判为 v8 的 V8 子表现在同样判 v9）；没有行时取当前 EXPECTED 版本。"""
    hp, hs = _hp(), _hs_light()
    raw = delivery.get("raw") or {}
    raw_rows = (raw.get("rows") or raw.get("delivered") or []) if isinstance(raw, dict) else []
    if any(isinstance(r, dict) and "source" in r for r in raw_rows):
        return "v9"
    cells = collections.Counter((r["task"], r["tier"]) for r in delivery["rows"])
    return hp.cells_version(dict(cells), hs) if cells else hp.expected_version(hs)


def _step_headroom_v8(args, delivery: dict[str, Any]) -> int:
    """{V8|V9}_STEP_CAP（判定行前缀按 :func:`delivery_version`）（1001 方案第一部分 §3「步数上限」）：交付 h5 的非演示步全部 ≤ ``EXEC_CAP``（1600）；
    ``filtered`` = 候选池里抽样阶段因 exec_over_cap 被丢弃并递补的候选数（与 delivery.json 的 ``exec_over_cap``
    不一致即 FAIL）；xhard0 按模块常量 ``XHARD0_STEP_CAP``（1300）单独查。交付行自带 ``exec_steps`` 时与 h5 实测比对。

    正式闸门（阶段 3）必须带 ``--xhard0``；``--skip-xhard0`` 只供局部核对，判定行改打 ``V8_STEP_CAP=INFO …
    xhard0_max=skipped``，永不出 PASS。delivery.json 的四个全局计数键（``exec_over_cap``／``backfills``／
    ``infra_retries``／``failed``）必须显式写出（含零），缺任一计 ``count_keys_absent`` 判 FAIL；其中
    ``infra_retries`` 只由 S2-B 聚合行与 delivery.json 给出（规格行里没有），本命令只核其存在。候选池里的空文件、
    坏 JSON 计 ``pool_load_errors`` 判 FAIL，判定行照常打印。"""
    hs = _hs_light()
    cap, x0cap = hs.EXEC_CAP, XHARD0_STEP_CAP
    if not args.pool:
        raise SystemExit("v8 step-headroom 须给 --pool <候选池目录或结果文件>（计 filtered=）")
    if not args.xhard0 and not args.skip_xhard0:
        raise SystemExit("v8 step-headroom 须给 --xhard0 <xhard0 h5 来源>，或显式 --skip-xhard0（判定行写 xhard0_max=skipped）")
    rows = delivery["rows"]
    per_cell: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    missing_h5, exec_mismatch = 0, 0
    over: list[str] = []
    for row in rows:
        path = _hp().delivery_h5(delivery, row)
        try:
            demo, exec_steps = _h5_steps(path)
        except Exception:  # noqa: BLE001 缺文件、打不开都计 missing_h5
            missing_h5 += 1
            continue
        per_cell[f"{row['task']}/{row['tier']}"].append((demo, exec_steps))
        exec_mismatch += int(row.get("exec_steps") is not None and int(row["exec_steps"]) != exec_steps)
        if exec_steps > cap:
            over.append(f"{row['task']}/{row['tier']}/{row['episode']}:{exec_steps}")
    max_exec = max((e for values in per_cell.values() for _, e in values), default=0)
    pool_found, pool_errors = pool_scan(args.pool)
    filtered = len(pool_found)
    absent = [key for key in hs_count_keys() if delivery["counts"].get(key) is None]
    reported = delivery["counts"].get("exec_over_cap")
    filtered_mismatch = int(reported is None or int(reported) != filtered)
    x0_max: int | str = "skipped"
    x0_over = x0_missing = x0_rows = 0
    if args.xhard0:
        try:
            x0_paths, x0_rows = _xhard0_h5s(args.xhard0)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            x0_paths, x0_rows = [], 0
            x0_missing += 1
            pool_errors.append(f"xhard0 来源读取失败：{type(exc).__name__}: {exc}"[:300])
        values = []
        for path in x0_paths:
            try:
                values.append(_h5_steps(path)[1])
            except Exception:  # noqa: BLE001
                x0_missing += 1
        x0_max = max(values, default=0)
        x0_over = sum(v > x0cap for v in values)
        x0_ok = bool(values) and x0_over == 0 and x0_missing == 0
    else:
        x0_ok = True
    ok = (bool(rows) and not over and missing_h5 == 0 and exec_mismatch == 0 and filtered_mismatch == 0
          and not absent and x0_ok and not pool_errors and delivery["schema"] == _hp().V8_DELIVERY_SCHEMA)
    # --skip-xhard0 不是完整闸门：最多 INFO，不出 PASS
    label = ("INFO" if not args.xhard0 else "PASS") if ok else "FAIL"
    report = {"cells": len(per_cell), "rows": len(rows), "max_exec": max_exec, "cap": cap, "over": over,
              "filtered": filtered, "filtered_delivery": reported, "count_keys_absent": absent,
              "missing_h5": missing_h5, "exec_steps_mismatch": exec_mismatch, "xhard0_max": x0_max,
              "pool_load_errors": pool_errors, "verdict": None,
              "xhard0_cap": x0cap, "xhard0_over": x0_over, "xhard0_missing": x0_missing,
              "per_cell_max": {k: max(e for _, e in v) for k, v in per_cell.items()},
              "per_cell_mean": {k: {"demo": round(sum(a for a, _ in v) / len(v)), "exec": round(sum(b for _, b in v) / len(v)),
                                    "total": round(sum(a + b for a, b in v) / len(v)), "n": len(v)}
                                for k, v in per_cell.items()}}
    line = (f"{delivery_version(delivery).upper()}_STEP_CAP={label} max={max_exec} cap={cap} over={len(over)} filtered={filtered} "
            f"xhard0_max={x0_max} xhard0_cap={x0cap} rows={len(rows)} cells={len(per_cell)} missing_h5={missing_h5} "
            f"exec_steps_mismatch={exec_mismatch} filtered_delivery={'absent' if reported is None else reported} "
            f"filtered_mismatch={filtered_mismatch} count_keys_absent={len(absent)} xhard0_rows={x0_rows} "
            f"xhard0_over={x0_over} xhard0_missing={x0_missing} pool_load_errors={len(pool_errors)}"
            + ("" if ok else f" detail={(over + absent + pool_errors)[:6]}"))
    report["line"] = line
    report["verdict"] = label
    _write_report(args.out, report)
    print(line, flush=True)
    return 0 if ok else 1


def hs_count_keys() -> tuple[str, ...]:
    """delivery.json 必须显式写出（含零值）的全局计数键（§2.2 第 7 条）。"""
    return _hp().DELIVERY_COUNT_KEYS


def cmd_step_headroom(args) -> int:
    """只接受 ``v8-delivery/1`` 交付清单（v8／v9）→ ``V8_STEP_CAP``／``V9_STEP_CAP``；v7 旧判据已删。"""
    hp = _hp()
    delivery = hp.read_delivery(Path(args.delivery))
    schema = delivery["schema"]
    if schema != hp.V8_DELIVERY_SCHEMA:
        raise SystemExit(f"未知交付清单 schema {schema!r}（须为 {hp.V8_DELIVERY_SCHEMA}；v7 旧判据已删）")
    return _step_headroom_v8(args, delivery)


# ── v7.5eval 环境检测（env-digest / env-digest-compare）───────────────────────
# 0929-v7.5eval-restructure-plan.md §3.2 第 1 步第 1 项、2.1、§4 测速、口径 9。
# 每个身份逐层存摘要（rows.jsonl）与原始数组（每身份一个 npz，np.savez_compressed 无损），对拍时摘要不等再读原始数组算差值。

#: 层的固定顺序（first_diff 取这个顺序里第一个不等的层）
ENV_DIGEST_LAYERS = ("identity", "pre_demo_state", "demo_frames", "reset_obs", "post_demo_state",
                     "step_frames", "step_obs", "step_state", "step_status")
#: 状态类层：键是 ``<分区>/<实体名>``，可做「仅改名」判定
_ENV_STATE_LAYERS = ("pre_demo_state", "post_demo_state", "step_state")
#: 数值观测层：摘要不等时算最大绝对差（单列 obs_max_abs）
_ENV_NUMERIC_OBS_LAYERS = ("reset_obs", "step_obs")
#: 两个摇杆任务动作只有 7 维
_STICK_TASKS = ("PatternLock", "RouteStick")
#: 未采到的计时段一律写这个字符串，不填 0
UNCOLLECTED = "uncollected"
#: 计时字段 → 实际包到的调用（写进每行 timing_notes，便于核对口径）
ENV_TIMING_NOTES = {
    "proc.import_numpy_s": "import numpy",
    "proc.import_torch_s": "import torch",
    "proc.import_sapien_s": "import sapien",
    "proc.import_mani_skill_s": "import mani_skill + mani_skill.envs",
    "proc.import_robomme_hard_s": "import robomme_hard.env_record_wrapper（连带注册 16 个任务类）",
    "proc.spawn_to_worker_s": "父进程 Popen 前 time.time() → 子进程进入 worker 函数（解释器启动 + 本文件导入）",
    "proc.first_vulkan_s": "进程内第一次 BaseEnv._setup_scene（含首个 sapien.render.RenderSystem 即首次 Vulkan 设备创建；"
                           "另含 physx 场景构造，无法单独拆出）",
    "make_env_s": "BenchmarkEnvBuilder.make_env_for_episode 整体",
    "gym_make_s": "make_env_for_episode 内 gymnasium.make（任务类 __init__，含 BaseEnv 构造期 reset(reconfigure=True)）",
    "wrapper_chain_s": "make_env_s − gym_make_s（DemonstrationWrapper / FailAwareWrapper 等包装链构造）",
    "eval_reset_s": "评估实例 env.reset()（FailAwareWrapper 起整条链）",
    "inner_reset_s": "eval reset 期间 mani_skill BaseEnv.reset 最外层调用累计",
    "initialize_episode_s": "eval reset 期间任务类 _initialize_episode 累计",
    "demo_s": "eval_reset_s − inner_reset_s（演示轨迹生成 + 初始一步）",
    "demo_s_per_frame": "demo_s / 演示帧数",
    "choices_s": "reset 后直接调 get_vqa_options 取多选项（评估实例不带 include_available_multi_choices）",
    "step_s": "每步 env.step() 墙钟（列表）",
    "step_physics_s": "每步内 BaseEnv._step_action 累计（物理步 + 控制器）",
    "step_get_obs_s": "每步内 BaseEnv.get_obs 累计（含传感器渲染）",
    "close_s": "env.close()",
    "probe_make_s": "演示前状态探针：另建底层环境 gymnasium.make（原始实参）",
    "probe_reset_s": "演示前状态探针：底层环境 reset()",
    "probe_close_s": "演示前状态探针：底层环境 close()",
    "class_timers": "按阶段（probe/make/reset/step/close）累计的类级计时：16 任务类 × {_load_agent,_load_scene,"
                    "_initialize_episode}（口径 9 批准清单）+ BaseEnv.{_setup_scene,_setup_sensors,_load_lighting,"
                    "_reconfigure,reset,_step_action,get_obs}；只计时、不改参数与返回值、只在本进程内",
}


def _env_digest(x: Any) -> str:
    """与 ``_PROBE.digest`` 同一口径：数组字节 + dtype + shape 的 sha256。"""
    import hashlib

    import numpy as np

    arr = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(arr.tobytes() + str(arr.dtype).encode() + str(arr.shape).encode()).hexdigest()


def _env_json_digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _env_layer(keys: dict[str, str]) -> dict[str, Any]:
    """一层的摘要：逐键摘要 + 整层摘要（逐键摘要字典的 json sha256）。"""
    return {"digest": _env_json_digest(keys), "keys": dict(sorted(keys.items()))}


def _env_np(x: Any):
    """torch 张量 / 列表 → numpy（不改 dtype）。"""
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _env_flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    """嵌套状态字典（get_state_dict）→ ``{"actors/<名>": ndarray, ...}``。"""
    out: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in sorted(tree.items()):
            out.update(_env_flatten(value, f"{prefix}/{key}" if prefix else str(key)))
    else:
        out[prefix] = _env_np(tree)
    return out


def _env_identity_key(row: dict[str, Any]) -> str:
    return f"{row['task']}/ep{row['source_episode']}/seed{row['seed']}/b{row['builder_episode']}"


def _env_rng_probe() -> dict[str, str]:
    """全局随机状态 sha：torch CPU 生成器、numpy 全局 RandomState、python random（不碰 CUDA 生成器，免得提前初始化 CUDA）。"""
    import hashlib
    import random

    import numpy as np
    import torch

    kind, keys, pos, has_gauss, cached = np.random.get_state()
    np_bytes = np.asarray(keys).tobytes() + f"{kind}|{pos}|{has_gauss}|{cached!r}".encode()
    return {"torch": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
            "numpy": hashlib.sha256(np_bytes).hexdigest(),
            "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}


def _env_rng_save():
    import random

    import numpy as np
    import torch

    return torch.random.get_rng_state(), np.random.get_state(), random.getstate()


def _env_rng_restore(saved) -> None:
    import random

    import numpy as np
    import torch

    torch.random.set_rng_state(saved[0])
    np.random.set_state(saved[1])
    random.setstate(saved[2])


class _EnvTimerHub:
    """类级计时：按当前阶段累计秒数与调用次数；同一标签递归调用只计最外层。"""

    def __init__(self) -> None:
        self.phase = "init"
        self.buckets: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
        self.counts: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self.first: dict[str, float] = {}
        self.depth: dict[str, int] = collections.defaultdict(int)
        self.wrapped: list[str] = []
        self.missing: list[str] = []

    def wrap(self, cls: type, name: str, label: str) -> None:
        import functools

        orig = cls.__dict__.get(name)
        if orig is None:
            self.missing.append(label)
            return
        hub = self

        @functools.wraps(orig)
        def timed(*a, **k):
            hub.depth[label] += 1
            started = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                hub.depth[label] -= 1
                if hub.depth[label] == 0:
                    dt = time.perf_counter() - started
                    hub.buckets[hub.phase][label] += dt
                    hub.counts[hub.phase][label] += 1
                    hub.first.setdefault(label, dt)

        setattr(cls, name, timed)
        self.wrapped.append(label)

    def total(self, phase: str, label: str) -> float | str:
        return round(self.buckets[phase][label], 6) if self.counts[phase].get(label) else UNCOLLECTED

    def snapshot(self, phase: str) -> dict[str, float]:
        return dict(self.buckets[phase])

    def take(self, phase: str) -> dict[str, Any]:
        return {"s": {k: round(v, 6) for k, v in sorted(self.buckets[phase].items())},
                "n": dict(sorted(self.counts[phase].items()))}


def _env_install_timers(tasks) -> _EnvTimerHub:
    """口径 9：16 任务类各自的 _load_agent/_load_scene/_initialize_episode（48 个）+ BaseEnv 几个方法；只在本进程内。"""
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.registration import REGISTERED_ENVS

    hub = _EnvTimerHub()
    for task in tasks:
        cls = REGISTERED_ENVS[task].cls
        for name in ("_load_agent", "_load_scene", "_initialize_episode"):
            hub.wrap(cls, name, f"{task}.{name}")
    for name in ("_setup_scene", "_setup_sensors", "_load_lighting", "_reconfigure", "reset", "_step_action", "get_obs"):
        hub.wrap(BaseEnv, name, f"BaseEnv.{name}")
    return hub


def _env_fs_type(path: str) -> dict[str, str]:
    """路径所在挂载点与文件系统类型（区分 NVMe / NFS 介质）。"""
    best = ("", "?", "?")
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            dev, mnt, fstype = line.split()[:3]
            if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best[0]):
                best = (mnt, fstype, dev)
    except OSError:
        pass
    return {"path": path, "mount": best[0], "fstype": best[1], "device": best[2]}


def _env_host_info() -> dict[str, Any]:
    """主机、GPU（按 CUDA_VISIBLE_DEVICES 对到 nvidia-smi 一次性查询，不采样）、CPU、affinity、包版本、代码与 venv 介质。"""
    import importlib.metadata as md
    import os
    import platform
    import socket
    import subprocess

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu: dict[str, Any] = {"cuda_visible_devices": visible}
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout
        cards = [dict(zip(("index", "uuid", "name", "driver"), (c.strip() for c in l.split(",")))) for l in text.splitlines() if l.strip()]
        want = (visible or "0").split(",")[0].strip()
        hit = next((c for c in cards if want in (c["index"], c["uuid"])), None)
        gpu.update(hit or {"name": UNCOLLECTED, "uuid": UNCOLLECTED, "driver": UNCOLLECTED})
    except (OSError, subprocess.SubprocessError):
        gpu.update({"name": UNCOLLECTED, "uuid": UNCOLLECTED, "driver": UNCOLLECTED})
    cpu = UNCOLLECTED
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    versions = {}
    for pkg in ("torch", "sapien", "mani_skill", "numpy", "gymnasium"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = UNCOLLECTED
    return {"host": socket.gethostname(), "gpu": gpu, "cpu_model": cpu, "affinity_cores": len(os.sched_getaffinity(0)),
            "python": platform.python_version(), "versions": versions,
            "code_medium": _env_fs_type(str(REPO)), "venv_medium": _env_fs_type(sys.prefix)}


def _env_stack(values: list, name: str, arrays: dict[str, Any], keys: dict[str, str]) -> None:
    """逐元素转 numpy 后 stack，存进 arrays 并记摘要；参差不齐时退化为逐元素摘要的 json 摘要（不进 npz）。"""
    import numpy as np

    try:
        arr = np.stack([_env_np(v) for v in values]) if len(values) else np.zeros((0,))
    except (ValueError, TypeError):
        keys[name] = _env_json_digest([_env_digest(_env_np(v)) if v is not None else None for v in values])
        return
    arrays[name] = arr
    keys[name] = _env_digest(arr)


def _env_digest_one(ident: dict[str, Any], builder, hub: _EnvTimerHub, fixed_steps: int, npz_path: Path) -> dict[str, Any]:
    """单个身份：演示前状态探针 → make_env → reset → 取多选项 → 固定动作 N 步 → close；逐层摘要 + 原始数组。"""
    import gymnasium as gym
    import numpy as np

    from robomme_hard.robomme_env.utils.vqa_options import get_vqa_options

    task = ident["task"]
    arrays: dict[str, Any] = {}
    layers: dict[str, dict[str, str]] = {name: {} for name in ENV_DIGEST_LAYERS}
    timing: dict[str, Any] = {}
    rng: dict[str, Any] = {}
    resolved = builder.resolve_identity(int(ident["builder_episode"]))
    if int(resolved["seed"]) != int(ident["seed"]) or int(resolved.get("source_episode", -1)) != int(ident["source_episode"]):
        raise RuntimeError(f"身份不符：输入 {ident}，builder {resolved}")
    seed, tier = builder.resolve_episode(int(ident["builder_episode"]))

    calls: list[tuple[str, dict[str, Any]]] = []
    orig_make = gym.make

    def make_spy(env_id, **kw):
        started = time.perf_counter()
        try:
            return orig_make(env_id, **kw)
        finally:
            calls.append((env_id, kw))
            timing["gym_make_s"] = round(time.perf_counter() - started, 6)

    # ① 与 _PROBE 同序：先经 builder 建评估实例（顺带截获 gym.make 实参），再用同一实参另建底层环境 reset 取演示前状态，
    #    最后才 reset 评估实例。探针前后保存/恢复全局随机状态，使探针对评估实例的全局随机流不可见。
    rng["before_make"] = _env_rng_probe()
    hub.phase = "make"
    gym.make = make_spy
    started = time.perf_counter()
    try:
        env = builder.make_env_for_episode(int(ident["builder_episode"]), max_steps=_cap_for(tier))
    finally:
        gym.make = orig_make
    timing["make_env_s"] = round(time.perf_counter() - started, 6)
    timing.setdefault("gym_make_s", UNCOLLECTED)
    timing["wrapper_chain_s"] = (round(timing["make_env_s"] - timing["gym_make_s"], 6)
                                 if timing["gym_make_s"] != UNCOLLECTED else UNCOLLECTED)
    rng["after_make"] = _env_rng_probe()
    env_id, make_kw = calls[-1]

    saved = _env_rng_save()
    hub.phase = "probe"
    started = time.perf_counter()
    base = orig_make(env_id, **make_kw)
    timing["probe_make_s"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    base.reset()  # 与 _PROBE 一致：seed 已随 gym.make 传入，reset 不另传（用户 2026-09-29 批准每局多 1 次底层 reset）
    timing["probe_reset_s"] = round(time.perf_counter() - started, 6)
    pre = _env_flatten(base.unwrapped.get_state_dict())
    started = time.perf_counter()
    base.close()
    timing["probe_close_s"] = round(time.perf_counter() - started, 6)
    del base
    _env_rng_restore(saved)
    for key, value in pre.items():
        arrays[f"pre_demo_state/{key}"] = value
        layers["pre_demo_state"][key] = _env_digest(value)

    chain, e = [], env
    while hasattr(e, "env"):
        chain.append(type(e).__name__)
        e = e.env
    chain.append(type(e.unwrapped).__name__)
    big = {k: _env_json_digest(v) for k, v in make_kw.items() if k in ("native_episode_spec", "sampling_config")}
    small = {k: v for k, v in make_kw.items() if k not in big}

    # ② 评估实例 reset（演示生成在内）
    rng["before_reset"] = _env_rng_probe()
    hub.phase = "reset"
    started = time.perf_counter()
    obs, info = env.reset()
    timing["eval_reset_s"] = round(time.perf_counter() - started, 6)
    rng["after_reset"] = _env_rng_probe()
    timing["inner_reset_s"] = hub.total("reset", "BaseEnv.reset")
    timing["inner_reset_calls"] = hub.counts["reset"].get("BaseEnv.reset", 0)
    timing["initialize_episode_s"] = hub.total("reset", f"{task}._initialize_episode")
    timing["demo_s"] = (round(timing["eval_reset_s"] - timing["inner_reset_s"], 6)
                        if timing["inner_reset_s"] != UNCOLLECTED else UNCOLLECTED)
    post = _env_flatten(env.unwrapped.get_state_dict())
    for key, value in post.items():
        arrays[f"post_demo_state/{key}"] = value
        layers["post_demo_state"][key] = _env_digest(value)
    obs = obs if isinstance(obs, dict) else {}
    front, wrist = list(obs.get("front_rgb_list", [])), list(obs.get("wrist_rgb_list", []))
    demo_frames = max(len(front) - 1, 0)
    timing["demo_frames"] = demo_frames
    timing["demo_s_per_frame"] = (round(timing["demo_s"] / demo_frames, 6)
                                  if demo_frames and timing["demo_s"] != UNCOLLECTED else UNCOLLECTED)
    demo_arrays: dict[str, Any] = {}
    reset_arrays: dict[str, Any] = {}
    for stream, frames in (("front", front), ("wrist", wrist)):
        # 演示帧 = 列表去掉最后一个元素；最后一个元素是初始帧，归 reset_obs 层
        _env_stack(frames[:-1], stream, demo_arrays, layers["demo_frames"])
        if frames:
            reset_arrays[f"{stream}_init"] = _env_np(frames[-1])
            layers["reset_obs"][f"{stream}_init"] = _env_digest(reset_arrays[f"{stream}_init"])
    for key, values in sorted(obs.items()):
        if key not in ("front_rgb_list", "wrist_rgb_list"):
            _env_stack(list(values), key, reset_arrays, layers["reset_obs"])
    arrays.update({f"demo_frames/{k}": v for k, v in demo_arrays.items()})
    arrays.update({f"reset_obs/{k}": v for k, v in reset_arrays.items()})

    # ③ 多选项：评估实例不开 include_available_multi_choices（与评估一致），reset 后直接调同一函数取一次
    demo_wrapper = env
    while not hasattr(demo_wrapper, "include_available_multi_choices") and hasattr(demo_wrapper, "env"):
        demo_wrapper = demo_wrapper.env
    hub.phase = "choices"
    started = time.perf_counter()
    try:
        raw = get_vqa_options(demo_wrapper, None, {"obj": None, "name": None, "seg_id": None}, task)
        choices = [{"label": o.get("label"), "action": o.get("action", "Unknown"), "need_parameter": bool(o.get("available"))}
                   for o in raw]
    except Exception as exc:  # noqa: BLE001 如实记录
        # 只记异常类型名：消息里可能带对象地址，进摘要会造成假差异；完整消息另存 choices_error_message（不进摘要）
        choices = {"error": type(exc).__name__}
        timing["choices_error_message"] = f"{exc}"[:300]
    timing["choices_s"] = round(time.perf_counter() - started, 6)
    rng["after_choices"] = _env_rng_probe()
    goal = info.get("task_goal") if isinstance(info, dict) else None
    identity_fields = {
        "seed": int(seed), "tier": tier, "resolve_identity": resolved, "make_env_id": env_id,
        "make_kwargs": json.loads(json.dumps(small, default=str)), "make_kwargs_big_sha256": big,
        "wrapper_chain": chain, "task_goal": [str(g) for g in (goal if isinstance(goal, (list, tuple)) else [goal])],
        "available_multi_choices": json.loads(json.dumps(choices, default=str)),
    }
    layers["identity"] = {k: _env_json_digest(v) for k, v in identity_fields.items()}

    # ④ 固定动作 N 步：动作 = reset 返回的最后一个关节状态（7 维）+ 夹爪 1.0（摇杆任务只有 7 维），与 eval-smoke 同为「原地保持」
    joint = np.asarray(_env_np(obs["joint_state_list"][-1]), dtype=np.float64).reshape(-1)[:7]
    action = joint if task in _STICK_TASKS else np.concatenate([joint, [1.0]])
    step_rows: dict[str, list] = collections.defaultdict(list)
    statuses, step_s, physics_s, get_obs_s = [], [], [], []
    hub.phase = "step"
    for _ in range(fixed_steps):
        before = hub.snapshot("step")
        started = time.perf_counter()
        obs_t, reward, terminated, truncated, info_t = env.step(action.copy())
        step_s.append(round(time.perf_counter() - started, 6))
        after = hub.snapshot("step")
        physics_s.append(round(after.get("BaseEnv._step_action", 0.0) - before.get("BaseEnv._step_action", 0.0), 6))
        get_obs_s.append(round(after.get("BaseEnv.get_obs", 0.0) - before.get("BaseEnv.get_obs", 0.0), 6))
        status = (info_t or {}).get("status")
        statuses.append({"status": status, "terminated": bool(_env_np(terminated).any()), "truncated": bool(_env_np(truncated).any()),
                         "n_elems": len((obs_t or {}).get("front_rgb_list", []) or []),
                         "error": (info_t or {}).get("error_message")})
        if status == "error" or not isinstance(obs_t, dict):
            break
        for key, values in obs_t.items():
            step_rows[key].append(values[-1] if len(values) else None)
        step_rows["__reward"].append(_env_np(reward))
        for key, value in _env_flatten(env.unwrapped.get_state_dict()).items():
            step_rows[f"__state/{key}"].append(value)
        if statuses[-1]["terminated"] or statuses[-1]["truncated"]:
            break
    for key, values in sorted(step_rows.items()):
        if key in ("front_rgb_list", "wrist_rgb_list"):
            name, layer = key.split("_")[0], "step_frames"
        elif key.startswith("__state/"):
            name, layer = key[len("__state/"):], "step_state"
        else:
            name, layer = key.lstrip("_"), "step_obs"
        tmp: dict[str, Any] = {}
        _env_stack(values, name, tmp, layers[layer])
        for k, v in tmp.items():
            arrays[f"{layer}/{k}"] = v
    layers["step_status"] = {"statuses": _env_json_digest(statuses), "action": _env_digest(action)}
    arrays["step_status/action"] = action
    timing.update({"step_s": step_s, "step_physics_s": physics_s if hub.wrapped else UNCOLLECTED,
                   "step_get_obs_s": get_obs_s if hub.wrapped else UNCOLLECTED})

    hub.phase = "close"
    started = time.perf_counter()
    env.close()
    timing["close_s"] = round(time.perf_counter() - started, 6)
    timing["class_timers"] = {phase: hub.take(phase) for phase in ("probe", "make", "reset", "choices", "step", "close")}
    for phase in ("probe", "make", "reset", "choices", "step", "close"):
        hub.buckets.pop(phase, None)
        hub.counts.pop(phase, None)
    hub.phase = "idle"

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(npz_path)
    rng_consumed = {
        "torch_rng_consumed_reset": rng["before_reset"]["torch"] != rng["after_reset"]["torch"],
        "np_rng_consumed_reset": rng["before_reset"]["numpy"] != rng["after_reset"]["numpy"],
        "py_rng_consumed_reset": rng["before_reset"]["python"] != rng["after_reset"]["python"],
        "torch_rng_consumed_make": rng["before_make"]["torch"] != rng["after_make"]["torch"],
        "np_rng_consumed_make": rng["before_make"]["numpy"] != rng["after_make"]["numpy"],
        "torch_rng_consumed_choices": rng["after_reset"]["torch"] != rng["after_choices"]["torch"],
        "np_rng_consumed_choices": rng["after_reset"]["numpy"] != rng["after_choices"]["numpy"],
    }
    return {"layers": {name: _env_layer(keys) for name, keys in layers.items()}, "identity_fields": identity_fields,
            "rng": rng, **rng_consumed, "torch_rng_consumed": rng_consumed["torch_rng_consumed_reset"],
            "np_rng_consumed": rng_consumed["np_rng_consumed_reset"], "timing": timing,
            "steps_done": len(statuses), "step_statuses": statuses, "fixed_action": action.tolist(),
            "npz": str(npz_path.relative_to(npz_path.parents[1])), "npz_bytes": npz_path.stat().st_size}


def cmd_env_digest_worker(args) -> int:
    """子进程：分记 import 耗时 → 装计时包装 → 逐身份跑 _env_digest_one，每身份一行追加到 rows.jsonl。"""
    import os

    entered = time.time()
    proc: dict[str, Any] = {}
    spawn = os.environ.get("V75_ENV_DIGEST_SPAWN_T")
    proc["spawn_to_worker_s"] = round(entered - float(spawn), 6) if spawn else UNCOLLECTED
    for label, mods in (("numpy", ("numpy",)), ("torch", ("torch",)), ("sapien", ("sapien",)),
                        ("mani_skill", ("mani_skill", "mani_skill.envs")),
                        ("robomme_hard", ("robomme_hard.env_record_wrapper",))):
        started = time.perf_counter()
        for mod in mods:
            __import__(mod)
        proc[f"import_{label}_s"] = round(time.perf_counter() - started, 6)
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    hs = _hard_specs()
    hub = _env_install_timers(hs.ALL_TASKS)
    proc["timers_wrapped"] = len(hub.wrapped)
    proc["timers_missing"] = hub.missing
    host = _env_host_info()
    idents = json.loads(Path(args.batch).read_text())
    cell_dir = Path(args.out) / args.cell
    rows_path = cell_dir / "rows.jsonl"
    builders: dict[str, Any] = {}
    for position, ident in enumerate(idents):
        started = time.time()
        row = {k: ident[k] for k in ("task", "source_episode", "seed", "builder_episode")}
        row.update({"cell": args.cell, "mode": args.mode, "order_index": ident["_order_index"], "proc_index": args.proc_index,
                    "position_in_proc": position, "pid": os.getpid(), "host": host, "timing_notes": ENV_TIMING_NOTES,
                    "include_available_multi_choices": False, "fixed_steps": args.fixed_steps,
                    "resume_generation": args.resume_generation, "resumed": args.resume_generation > 0,
                    "proc_init": proc if position == 0 else {"note": "见本进程 position_in_proc=0 的行"}, "error": None})
        try:
            builder = builders.setdefault(ident["task"], BenchmarkEnvBuilder(
                env_id=ident["task"], dataset="ood", action_space="joint_angle", max_steps=1300))
            npz = cell_dir / "arrays" / f"{ident['task']}-ep{ident['source_episode']}-b{ident['builder_episode']}.npz"
            row.update(_env_digest_one(ident, builder, hub, args.fixed_steps, npz))
        except Exception as exc:  # noqa: BLE001 如实记录，续跑时重做
            import traceback

            row["error"] = f"{type(exc).__name__}: {exc}"[:800]
            row["traceback"] = traceback.format_exc()[-3000:]
            hub.phase = "idle"
        if position == 0:
            # 第一个身份出错也照记（只要 _setup_scene 被调到过）
            first = hub.first.get("BaseEnv._setup_scene")
            proc["first_vulkan_s"] = round(first, 6) if first is not None else UNCOLLECTED
        row["wall_s"] = round(time.time() - started, 3)
        with rows_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        print(f"ENV_DIGEST_ROW cell={args.cell} id={_env_identity_key(row)} wall_s={row['wall_s']} "
              f"npz_mib={row.get('npz_bytes', 0) / 2**20:.1f} steps={row.get('steps_done')} "
              f"torch_rng_consumed={row.get('torch_rng_consumed')} np_rng_consumed={row.get('np_rng_consumed')} "
              f"error={row['error']}", flush=True)
    return 0


def _env_rows(cell_dir: Path) -> dict[str, dict[str, Any]]:
    """rows.jsonl → {身份键: 最后一行无错的行}（有错的行不算完成）。"""
    out: dict[str, dict[str, Any]] = {}
    path = cell_dir / "rows.jsonl"
    if not path.exists():
        return out
    for text in path.read_text().splitlines():
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue  # 半行（进程被杀）
        if row.get("error") is None and "layers" in row:
            out[_env_identity_key(row)] = row
    return out


def _env_resume_generation(cell_dir: Path) -> int:
    """本次运行的续跑代数：rows.jsonl 无任何行 → 0；否则 = 已有行里最大 resume_generation + 1（旧行缺字段按 0）。"""
    path = cell_dir / "rows.jsonl"
    gens = []
    if path.exists():
        for text in path.read_text().splitlines():
            try:
                gens.append(int(json.loads(text).get("resume_generation") or 0))
            except (json.JSONDecodeError, ValueError, AttributeError):
                continue
    return max(gens) + 1 if gens else 0


def _env_p50(values: list) -> str:
    nums = sorted(v for v in values if isinstance(v, (int, float)))
    if not nums:
        return UNCOLLECTED
    return f"{nums[len(nums) // 2]:.3f}"


def _env_speed_line(cell: str, rows: list[dict[str, Any]]) -> str:
    t = [r["timing"] for r in rows]
    firsts = [r["proc_init"] for r in rows if r.get("position_in_proc") == 0 and isinstance(r.get("proc_init"), dict)]
    host = rows[0]["host"] if rows else {}
    fields = {
        "rows": len(rows), "host": host.get("host", UNCOLLECTED),
        "gpu": str((host.get("gpu") or {}).get("name", UNCOLLECTED)).replace(" ", "_"),
        "cores": host.get("affinity_cores", UNCOLLECTED),
        "procs": len(firsts),
        "import_torch_s_p50": _env_p50([p.get("import_torch_s") for p in firsts]),
        "import_sapien_s_p50": _env_p50([p.get("import_sapien_s") for p in firsts]),
        "import_mani_skill_s_p50": _env_p50([p.get("import_mani_skill_s") for p in firsts]),
        "import_robomme_hard_s_p50": _env_p50([p.get("import_robomme_hard_s") for p in firsts]),
        "first_vulkan_s_p50": _env_p50([p.get("first_vulkan_s") for p in firsts]),
        "make_env_s_p50": _env_p50([x.get("make_env_s") for x in t]),
        "gym_make_s_p50": _env_p50([x.get("gym_make_s") for x in t]),
        "eval_reset_s_p50": _env_p50([x.get("eval_reset_s") for x in t]),
        "inner_reset_s_p50": _env_p50([x.get("inner_reset_s") for x in t]),
        "demo_s_p50": _env_p50([x.get("demo_s") for x in t]),
        "demo_s_per_frame_p50": _env_p50([x.get("demo_s_per_frame") for x in t]),
        "step_s_p50": _env_p50([s for x in t for s in (x.get("step_s") or [])]),
        "step_physics_s_p50": _env_p50([s for x in t if isinstance(x.get("step_physics_s"), list) for s in x["step_physics_s"]]),
        "step_get_obs_s_p50": _env_p50([s for x in t if isinstance(x.get("step_get_obs_s"), list) for s in x["step_get_obs_s"]]),
        "close_s_p50": _env_p50([x.get("close_s") for x in t]),
        "wall_s_p50": _env_p50([r.get("wall_s") for r in rows]),
    }
    return f"ENV_SPEED=INFO cell={cell} " + " ".join(f"{k}={v}" for k, v in fields.items())


def cmd_env_digest(args) -> int:
    """ENV_DIGEST_DONE：按身份清单逐个建环境、固定动作走 N 步，逐层存摘要与原始数组（不接策略）。
    默认每任务一个新进程（正序）；--resident 全部放进一个常驻进程；--reverse 倒序；已在 rows.jsonl 的身份跳过（续跑）。
    ⚠ 演示前场景状态取自另建的底层环境 reset 后的状态（_PROBE 的做法），不是评估实例本身的瞬间状态。"""
    import os
    import subprocess

    idents = json.loads(Path(args.identities).read_text())
    for index, ident in enumerate(idents):
        ident["_order_index"] = index
    if args.reverse:
        idents = idents[::-1]
    if args.limit:
        idents = idents[: args.limit]
    cell_dir = Path(args.out) / args.cell
    (cell_dir / "batches").mkdir(parents=True, exist_ok=True)
    done = _env_rows(cell_dir)
    todo = [i for i in idents if _env_identity_key(i) not in done]
    generation = _env_resume_generation(cell_dir)
    if args.resident and generation > 0 and not args.allow_resume_resident:
        # 常驻条件的含义是「全部身份在同一进程里连续跑」；续跑会拆成多个进程，条件被静默改变
        print(f"ENV_DIGEST_RESUME_REFUSED cell={args.cell} mode=resident resume_generation={generation} "
              f"done={len(idents) - len(todo)} todo={len(todo)}（换新 --cell 重跑，或显式加 --allow-resume-resident）",
              flush=True)
        return 2
    batches: list[list[dict[str, Any]]] = []
    if args.resident:
        batches = [todo] if todo else []
    else:
        for ident in todo:
            if batches and batches[-1][0]["task"] == ident["task"]:
                batches[-1].append(ident)
            else:
                batches.append([ident])
    mode = "resident" if args.resident else "per-task"
    mode += "-reverse" if args.reverse else "-forward"
    print(f"ENV_DIGEST_PLAN cell={args.cell} identities={len(idents)} done={len(idents) - len(todo)} "
          f"resume_generation={generation} resumed={generation > 0} "
          f"todo={len(todo)} procs={len(batches)} mode={mode}", flush=True)
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    failures = 0
    for proc_index, batch in enumerate(batches):
        batch_path = cell_dir / "batches" / f"proc{proc_index:03d}-{os.getpid()}.json"
        batch_path.write_text(json.dumps(batch, ensure_ascii=False))
        env["V75_ENV_DIGEST_SPAWN_T"] = repr(time.time())
        code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "env-digest-worker", "--cell", args.cell,
                               "--out", args.out, "--batch", str(batch_path), "--fixed-steps", str(args.fixed_steps),
                               "--proc-index", str(proc_index), "--mode", mode,
                               "--resume-generation", str(generation)], env=env, cwd=str(REPO)).returncode
        if code != 0:
            failures += 1
            print(f"ENV_DIGEST_PROC_FAIL cell={args.cell} proc={proc_index} code={code} tasks={sorted({b['task'] for b in batch})}",
                  flush=True)
    done = _env_rows(cell_dir)
    rows = [done[k] for k in (_env_identity_key(i) for i in idents) if k in done]
    print(_env_speed_line(args.cell, rows), flush=True)
    print(f"ENV_DIGEST_DONE cell={args.cell} rows={len(rows)}", flush=True)
    return 0 if len(rows) == len(idents) and not failures else 1


def _env_name_agnostic(keys: dict[str, str]) -> dict[str, list[str]]:
    """与 _name_agnostic 同一思路：按分区（键的第一段）取逐键摘要的有序多重集，忽略实体名。"""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for key, value in keys.items():
        out[key.split("/")[0]].append(value)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _env_load_npz(cell_dir: Path, row: dict[str, Any]):
    import numpy as np

    path = cell_dir / row["npz"]
    return np.load(path) if path.exists() else None


def _env_is_image(arr) -> bool:
    """uint8 且形如 (H,W,3) 或 (N,H,W,3) 的数组按图像处理（含 reset_obs 的 front_init/wrist_init）。"""
    return arr.dtype == "uint8" and arr.ndim in (3, 4) and arr.shape[-1] == 3


def _env_image_diff(x, y) -> dict[str, Any] | None:
    """逐帧 MAD（每帧在 H、W、通道上取平均绝对差，0–255 刻度）：最大值、均值、第一个不等帧、不等帧数；
    帧数不同只比公共前缀并记两边帧数；单帧尺寸不同则不可测（返回 None）。"""
    import numpy as np

    if x.ndim == 3:
        x, y = x[None], y[None]
    if x.shape[1:] != y.shape[1:]:
        return None
    n = min(len(x), len(y))
    out: dict[str, Any] = {"frames": [int(len(x)), int(len(y))]}
    if n == 0:
        out.update({"per_frame_mad_max": None, "per_frame_mad_mean": None, "first_diff_frame": None, "diff_frames": 0})
        return out
    mad = np.abs(x[:n].astype(np.int16) - y[:n].astype(np.int16)).mean(axis=(1, 2, 3))
    bad = np.nonzero(mad > 0)[0]
    out.update({"per_frame_mad_max": float(mad.max()), "per_frame_mad_mean": float(mad.mean()),
                "first_diff_frame": int(bad[0]) if len(bad) else None, "diff_frames": int(len(bad))})
    return out


def _env_compare_row(a: dict[str, Any], b: dict[str, Any], dir_a: Path, dir_b: Path) -> dict[str, Any]:
    """单个身份逐层比：相等 / 仅改名 / 键集合差 / 数值差（状态最大绝对差、图像逐帧 MAD）。
    量不出来的（npz 缺失、形状或 dtype 不符、参差回退未存数组、只有键集合差）一律记 None 并计入 unmeasured，不填 0。"""
    import numpy as np

    detail: dict[str, Any] = {"id": _env_identity_key(a), "layers": {}, "npz_missing": [], "unmeasured": 0}
    za = zb = None
    loaded = False
    first_diff = None
    for layer in ENV_DIGEST_LAYERS:
        la, lb = a["layers"].get(layer), b["layers"].get(layer)
        if la is None or lb is None:
            detail["layers"][layer] = {"equal": False, "missing": "a" if la is None else "b"}
            first_diff = first_diff or layer
            continue
        if la["digest"] == lb["digest"]:
            detail["layers"][layer] = {"equal": True}
            continue
        ka, kb = la["keys"], lb["keys"]
        info: dict[str, Any] = {"equal": False}
        only_a, only_b = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
        if only_a or only_b:
            info.update({"only_a": only_a[:20], "only_b": only_b[:20], "key_set_diff": True})
            # 只有键集合不同时才可能「仅改名」；键集合相同而值不同（如两个同形 actor 互换位姿）是真差异
            if layer in _ENV_STATE_LAYERS and _env_name_agnostic(ka) == _env_name_agnostic(kb):
                info["name_only"] = True
                detail["layers"][layer] = info
                continue
        diff_keys = sorted(k for k in set(ka) & set(kb) if ka[k] != kb[k])
        info["diff_keys"] = diff_keys[:20]
        info["diff_key_count"] = len(diff_keys)
        first_diff = first_diff or layer
        info["max_abs"] = None
        info["image"] = None
        if layer in ("identity", "step_status"):
            detail["layers"][layer] = info
            continue
        if not diff_keys:
            # 只有键集合差：公共键全相等，没有可量的数值
            detail["unmeasured"] += 1
            detail["layers"][layer] = info
            continue
        if not loaded:
            za, zb, loaded = _env_load_npz(dir_a, a), _env_load_npz(dir_b, b), True
            detail["npz_missing"] = [side for side, z in (("a", za), ("b", zb)) if z is None]
        if za is None or zb is None:
            detail["unmeasured"] += len(diff_keys)
            detail["layers"][layer] = info
            continue
        worst: float | None = None
        images: dict[str, Any] = {}
        unmeasured: list[str] = []
        for key in diff_keys:
            name = f"{layer}/{key}"
            if name not in za.files or name not in zb.files:
                unmeasured.append(key)  # 参差回退只存了摘要
                continue
            x, y = za[name], zb[name]
            if _env_is_image(x) and _env_is_image(y):
                got = _env_image_diff(x, y)
                if got is None:
                    unmeasured.append(key)
                else:
                    images[key] = got
                continue
            if x.shape != y.shape or x.dtype.kind not in "biuf" or y.dtype.kind not in "biuf":
                unmeasured.append(key)
                continue
            value = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max(initial=0.0))
            worst = value if worst is None else max(worst, value)
        info["max_abs"] = worst
        if images:
            maxes = [v["per_frame_mad_max"] for v in images.values() if v["per_frame_mad_max"] is not None]
            means = [v["per_frame_mad_mean"] for v in images.values() if v["per_frame_mad_mean"] is not None]
            info["image"] = {"per_frame_mad_max": max(maxes) if maxes else None,
                             "per_frame_mad_mean": max(means) if means else None, "keys": images}
        if unmeasured:
            info["unmeasured_keys"] = unmeasured[:20]
            detail["unmeasured"] += len(unmeasured)
        detail["layers"][layer] = info
    detail["first_diff"] = first_diff
    return detail


def _env_fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def cmd_env_digest_compare(args) -> int:
    """ENV_DIGEST_PARITY（只报告）：两格按身份对齐逐层比，打一行汇总并写 json 明细。
    state_max_abs／obs_max_abs／image_mad 在没有任何可量差异时为 n/a（全同时也是 n/a：摘要相等不再读数组）。"""
    dir_a, dir_b = Path(args.a), Path(args.b)
    rows_a, rows_b = _env_rows(dir_a), _env_rows(dir_b)
    common = [k for k in rows_a if k in rows_b]
    common.sort(key=lambda k: rows_a[k].get("order_index", 0))
    details = [_env_compare_row(rows_a[k], rows_b[k], dir_a, dir_b) for k in common]
    layer_equal = {layer: sum(d["layers"][layer]["equal"] for d in details) for layer in ENV_DIGEST_LAYERS}
    name_only = {layer: sum(bool(d["layers"][layer].get("name_only")) for d in details) for layer in ENV_DIGEST_LAYERS}
    key_set = sum(any(v.get("key_set_diff") for v in d["layers"].values()) for d in details)
    firsts = [d["first_diff"] for d in details if d["first_diff"]]
    first = min(firsts, key=ENV_DIGEST_LAYERS.index) if firsts else "-"
    first_id = next((d["id"] for d in details if d["first_diff"] == first), "-")

    def pick(layers, field):
        vals = [v.get(field) for d in details for layer, v in d["layers"].items() if layer in layers]
        vals = [x for x in vals if x is not None]
        return max(vals) if vals else None

    state = pick(_ENV_STATE_LAYERS, "max_abs")
    obs = pick(_ENV_NUMERIC_OBS_LAYERS, "max_abs")
    imgs = [v["image"] for d in details for v in d["layers"].values() if v.get("image")]
    mad_max = max((i["per_frame_mad_max"] for i in imgs if i["per_frame_mad_max"] is not None), default=None)
    mad_mean = max((i["per_frame_mad_mean"] for i in imgs if i["per_frame_mad_mean"] is not None), default=None)
    frame_hits = [(d["id"], layer, key, v["first_diff_frame"], v["diff_frames"]) for d in details
                  for layer, lv in d["layers"].items() if lv.get("image") for key, v in lv["image"]["keys"].items()]
    first_frame = next((f"{layer}/{key}@{idx}" for _i, layer, key, idx, _n in frame_hits if idx is not None), "-")
    diff_frames = sum(n for *_x, n in frame_hits)
    npz_missing = sum(bool(d["npz_missing"]) for d in details)
    unmeasured = sum(d["unmeasured"] for d in details)
    missing_a = sorted(set(rows_b) - set(rows_a))
    missing_b = sorted(set(rows_a) - set(rows_b))
    resumed = {"a": sum(bool(r.get("resumed")) for r in rows_a.values()),
               "b": sum(bool(r.get("resumed")) for r in rows_b.values())}
    summary = {
        "pair": f"{dir_a.name}:{dir_b.name}", "a": str(dir_a), "b": str(dir_b), "compared": len(details),
        "rows_a": len(rows_a), "rows_b": len(rows_b), "missing_in_a": missing_a, "missing_in_b": missing_b,
        "layer_equal": layer_equal, "name_only": name_only, "key_set_diff": key_set,
        "first_diff": first, "first_diff_id": first_id, "identities_with_diff": len(firsts),
        "state_max_abs": state, "obs_max_abs": obs,
        "image_mad": mad_max, "image_mad_unit": None if mad_max is None else mad_max / 255.0,
        "image_mad_mean": mad_mean, "image_first_diff_frame": first_frame, "image_diff_frames": diff_frames,
        "npz_missing": npz_missing, "unmeasured": unmeasured, "resumed_rows": resumed,
        "note": "演示前场景状态取自另建底层环境 reset 后的状态（_PROBE 做法），非评估实例瞬间状态；"
                "image_mad = 各身份各图像键逐帧 MAD（0–255 刻度，每帧在 H、W、通道上平均）的最大值，image_mad_unit 为 ÷255，"
                "image_mad_mean = 各图像键逐帧 MAD 均值的最大值；量不出来记 None（行里 n/a）并计入 unmeasured",
        "details": details,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1) + "\n")
    print(f"ENV_DIGEST_PARITY pair={summary['pair']} compared={len(details)} "
          f"layer_equal={','.join(f'{k}:{v}' for k, v in layer_equal.items())} first_diff={first} "
          f"state_max_abs={_env_fmt(state)} image_mad={_env_fmt(mad_max)} "
          f"image_mad_unit={_env_fmt(None if mad_max is None else mad_max / 255.0)} image_mad_mean={_env_fmt(mad_mean)} "
          f"image_first_diff_frame={first_frame} image_diff_frames={diff_frames} obs_max_abs={_env_fmt(obs)} "
          f"name_only={sum(name_only.values())} key_set_diff={key_set} npz_missing={npz_missing} unmeasured={unmeasured} "
          f"rows_a={len(rows_a)} rows_b={len(rows_b)} missing_in_a={len(missing_a)} missing_in_b={len(missing_b)} "
          f"resumed_a={resumed['a']} resumed_b={resumed['b']} first_diff_id={first_id}", flush=True)
    return 0


# ── v9：MoveCube xhard4 新区域落点与运动方式（1002 方案 §2.3 V9_MOVECUBE_LAYOUT／V9_MOVECUBE_WAYS）──────


#: v8 MoveCube xhard4 区域（V6 圆环版，S1-A 改值前）：只用来数「落在旧区域之外」的点（outside_old）
V8_MOVECUBE_REGION = {"r_in": 0.12, "r_out": 0.20, "base_dist": [0.35, 0.76]}
#: v9 区域三个数（1002 方案第一部分 §1 已定口径 1）；规格 ``layout.<seg>.region`` 与源码 ``config_xhard4`` 都须等于它
V9_MOVECUBE_REGION = {"r_in": 0.24, "r_out": 0.42, "base_dist": [0.31, 0.80]}
MOVECUBE_SEGMENTS = ("demo", "execution")
MOVECUBE_POINTS = ("cube", "goal", "grasp")


def _movecube_source():
    """源码判定函数（不重写规则）：``MoveCube._in_region_u`` 判点在 U 内、``_xhard4_region`` 把规格里的 region 字典
    校验并整理成判定用结构、``_peg_root_xy``／``_peg_geometry``／``_peg_axis_extent`` 由杆根与朝向推抓取点
    （杆尾 = 杆根 − length·u）；``_freeze._movecube_way`` 取运动方式、``V9_MOVECUBE_QUOTA_BY_WAY`` 取期望配额。
    只导入、不建环境、不 reset。"""
    injection = REPO / "scripts" / "injection-dev"
    if str(injection) not in sys.path:
        sys.path.insert(0, str(injection))
    import importlib  # noqa: PLC0415

    import _freeze  # noqa: PLC0415

    # 包 __init__ 把同名类导出到 robomme_hard.robomme_env.MoveCube 属性上，按模块路径取模块本身
    module = importlib.import_module("robomme_hard.robomme_env.MoveCube")
    return module, _freeze


def movecube_points(spec: dict[str, Any], seg: str, module) -> dict[str, Any]:
    """一段（demo／execution）的三个落点与该段规格 region（源码 ``_xhard4_region`` 整理后）。

    规格字段（``_load_scene_xhard4_region`` 写入）：``layout.<seg>.cube_pose = [x, y, yaw]``、``goal_xy = [x, y]``、
    ``peg_offsets = [base_y, 杆根 x, 杆根 y]``、``peg_yaw``、``region``（决策字典 + ``peg_axis_extent_m`` 等记录项）。
    杆长取 ``2 × peg_axis_extent_m[1]``（``_peg_axis_extent(length) = (−1.5·length, 0.5·length)``），并以源码函数
    反算复核，不符即抛 ValueError（计坏行）。"""
    import numpy as np

    layout = spec["layout"][seg]
    region_cfg = layout["region"]
    region = module.MoveCube._xhard4_region(None, {"xhard4": {"region": region_cfg}}, f"{seg}_layout")
    extent = tuple(float(v) for v in region_cfg["peg_axis_extent_m"])
    length = 2.0 * extent[1]
    if not np.allclose(module._peg_axis_extent(length), extent, rtol=0.0, atol=1e-12):
        raise ValueError(f"{seg}.peg_axis_extent_m {extent} 与 _peg_axis_extent({length}) 不符")
    base_y, root_x, root_y = (float(v) for v in layout["peg_offsets"])
    root = module._peg_root_xy(base_y, root_x, root_y)
    grasp, _, _ = module._peg_geometry(root, float(layout["peg_yaw"]), length, extent)
    points = {"cube": np.asarray(layout["cube_pose"][:2], dtype=np.float64),
              "goal": np.asarray(layout["goal_xy"][:2], dtype=np.float64),
              "grasp": np.asarray(grasp, dtype=np.float64)}
    return {"region_cfg": region_cfg, "region": region, "points": points}


def _region_with(module, base_cfg: dict[str, Any], override: dict[str, Any], seg: str) -> dict[str, Any]:
    return module.MoveCube._xhard4_region(None, {"xhard4": {"region": {**base_cfg, **override}}}, f"{seg}_layout")


def movecube_layout_check(header: dict[str, Any], rows: list[dict[str, Any]], *, module, freeze,
                          expected_episodes: int) -> dict[str, Any]:
    """纯函数核心（夹具可直接调）：逐局两段三点 → in_region／outside_old／region_mismatch／坏行，按运动方式计数。"""
    source_cfg = module.MoveCube.config_xhard4["region"]
    in_region = outside_old = points = region_mismatch = 0
    bad_rows: list[str] = []
    out_detail: list[str] = []
    mismatch_detail: list[str] = []
    ways: collections.Counter = collections.Counter()
    for row in rows:
        label = f"{row.get('task')}/{row.get('tier')}/c{row.get('candidate')}/seed{row.get('seed')}"
        try:
            segs = {seg: movecube_points(row["spec"], seg, module) for seg in MOVECUBE_SEGMENTS}
        except Exception as exc:  # noqa: BLE001 缺字段、类型不对、源码校验拒绝 → 坏行，照常打判定行
            bad_rows.append(f"{label}: {type(exc).__name__}: {exc}"[:300])
            continue
        ways[freeze._movecube_way(row["spec"])] += 1
        for seg, item in segs.items():
            cfg = item["region_cfg"]
            diff = sorted(k for k in source_cfg if cfg.get(k) != source_cfg[k])
            diff += sorted(k for k, v in V9_MOVECUBE_REGION.items() if cfg.get(k) != v)
            if diff:
                region_mismatch += 1
                mismatch_detail.append(f"{label}/{seg}:{sorted(set(diff))}")
            old = _region_with(module, cfg, V8_MOVECUBE_REGION, seg)
            for name in MOVECUBE_POINTS:
                xy = item["points"][name]
                points += 1
                why = module._in_region_u(xy, item["region"])
                if why is None:
                    in_region += 1
                elif len(out_detail) < 6:
                    out_detail.append(f"{label}/{seg}/{name}:{why}")
                outside_old += int(module._in_region_u(xy, old) is not None)
    expected_ways = {int(k): int(v) for k, v in freeze.V9_MOVECUBE_QUOTA_BY_WAY.items()}
    way_keys = sorted(expected_ways)
    got_ways = {k: ways.get(k, 0) for k in way_keys}
    unknown_ways = sum(n for k, n in ways.items() if k not in expected_ways)
    episodes = len(rows)
    want_points = expected_episodes * len(MOVECUBE_SEGMENTS) * len(MOVECUBE_POINTS)
    ok_layout = (episodes == expected_episodes and not bad_rows and points == want_points
                 and in_region == points and outside_old >= 1 and region_mismatch == 0)
    ok_ways = not bad_rows and episodes == expected_episodes and got_ways == expected_ways and unknown_ways == 0
    return {"episodes": episodes, "expected_episodes": expected_episodes, "points": points, "in_region": in_region,
            "outside_old": outside_old, "region_mismatch": region_mismatch, "bad_rows": bad_rows,
            "out_detail": out_detail, "mismatch_detail": mismatch_detail[:6], "ok_layout": ok_layout,
            "ways": got_ways, "expected_ways": expected_ways, "unknown_ways": unknown_ways, "ok_ways": ok_ways,
            "way_order": way_keys, "header_quota": (header.get("delivery_per_cell") or {}).get("MoveCube")}


def cmd_movecube_layout(args) -> int:
    """V9_MOVECUBE_LAYOUT／V9_MOVECUBE_WAYS（纯 CPU，只读规格，不建环境、不 reset）。

    取数：``--specs-root`` 为规格根（读 ``<根>/<--tier>/specs.jsonl``，缺省档 xhard4）或单个 specs.jsonl；文件经
    header 推出的格表校验（``hard_parity.load_specs_any``）。取 MoveCube 行：给 ``--delivery`` 时取 ``selected`` 且身份
    (task, tier, seed) 在该交付清单里的行，不给时取交付行（selected 且 rollout ok）。

    判定：每局 demo、execution 两段 × 方块中心（``cube_pose[:2]``）、goal 中心（``goal_xy``）、抓杆点（杆尾 =
    杆根 − length·u）三点，用源码 ``MoveCube._in_region_u`` 对该段规格 ``layout.<seg>.region`` 判在 U 内；同一函数对
    旧 V8 区域（r_in 0.12、r_out 0.20、base_dist [0.35, 0.76]，其余键同规格）判落在旧 U 之外的点数 ``outside_old``；
    规格 region 与源码 ``config_xhard4["region"]`` 及 V9 三个数（0.24／0.42／[0.31, 0.80]）不符的段数
    ``region_mismatch``。PASS 须：局数 = ``--expected-episodes``（缺省 V9_CELLS 的 MoveCube/xhard4 = 50）、
    ``in_region`` = 点数（50 × 2 × 3 = 300）、``outside_old`` ≥ 1、``region_mismatch`` = 0、无坏行。
    运动方式按 ``_freeze._movecube_way``（最后一次 _initialize_episode 的 way_idx）计数，须逐方式等于
    ``_freeze.V9_MOVECUBE_QUOTA_BY_WAY``（17／17／16）。"""
    hp, hs = _hp(), _hs_light()
    module, freeze = _movecube_source()
    source = Path(args.specs_root)
    path = source if source.is_file() else source / args.tier / "specs.jsonl"
    load_error = None
    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    try:
        header, all_rows = hp.load_specs_any(path, hs)
        if args.delivery:
            ids, _, _ = _delivery_sources(args.delivery)
            rows = [r for r in all_rows if r["task"] == "MoveCube" and r.get("selected")
                    and (r["task"], r["tier"], int(r["seed"])) in ids]
        else:
            rows = [r for r in all_rows if r["task"] == "MoveCube" and hs.delivered(r)]
    except Exception as exc:  # noqa: BLE001 读不出 / 校验失败如实计入判定
        load_error = f"{type(exc).__name__}: {exc}"[:400]
    expected = args.expected_episodes if args.expected_episodes else hs.V9_CELLS[("MoveCube", "xhard4")]
    result = movecube_layout_check(header, rows, module=module, freeze=freeze, expected_episodes=expected)
    ok_layout = result["ok_layout"] and load_error is None
    ok_ways = result["ok_ways"] and load_error is None
    order = result["way_order"]
    layout_line = (f"V9_MOVECUBE_LAYOUT={'PASS' if ok_layout else 'FAIL'} episodes={result['episodes']} "
                   f"in_region={result['in_region']} outside_old={result['outside_old']} points={result['points']} "
                   f"expected_episodes={expected} region_mismatch={result['region_mismatch']} "
                   f"bad_rows={len(result['bad_rows'])} load_errors={int(load_error is not None)}"
                   + ("" if ok_layout else f" detail={(result['out_detail'] + result['mismatch_detail'] + result['bad_rows'])[:6]}"
                      f" load_error={load_error!r}"))
    ways_line = (f"V9_MOVECUBE_WAYS={'PASS' if ok_ways else 'FAIL'} "
                 f"ways={'/'.join(str(result['ways'][k]) for k in order)} "
                 f"expected={'/'.join(str(result['expected_ways'][k]) for k in order)} "
                 f"way_idx={'/'.join(map(str, order))} unknown={result['unknown_ways']} episodes={result['episodes']}")
    for line in (layout_line, ways_line):
        print(line, flush=True)
    _write_report(args.out, {"specs": str(path), "delivery": args.delivery, "load_error": load_error,
                             **{k: v for k, v in result.items() if k not in ("ok_layout", "ok_ways")},
                             "lines": [layout_line, ways_line]})
    return 0 if ok_layout and ok_ways else 1


_CELLS_HELP = ("格表（判定行前缀随格表版本 V9_）：full（默认＝当前 hard_specs.EXPECTED_CELLS，即 V9 的 43 格 800）"
               "｜v9full 或 v9（V9_CELLS 43 格 800）｜v9smoke（MoveCube／InsertPeg xhard4 各 1 局）"
               "｜v9shard1（V9 MoveCube 一片）｜JSON 文件路径或内联 JSON（分片子集，形如 {\"PickXtimes/xhard1\": 17}、"
               "{\"MoveCube@xhard4\": 50} 或 [[task, tier, n], ...]；须能被 V9_CELLS 覆盖）。"
               "V8 专用的 v8full／smoke 已删除")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ds = sub.add_parser("delivery-set", help="v8 交付形态／seed 按档隔离／布局独立（只读，纯 CPU）",
                        description=cmd_delivery_set.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ds.add_argument("--specs-root", required=True, help="v8／v9 /4 规格根（<根>/xhard{1..5}/specs.jsonl）")
    ds.add_argument("--cells", default="full", help=_CELLS_HELP)
    ds.add_argument("--delivery", default=None,
                    help="可选（v9 阶段 3b 必给）：交付清单 json（v9_subset_specs.py assemble 的 800 行 delivery.local.json）；"
                         "身份须与规格根交付行逐一相同，行带 source（v8-reuse|v9-new）时判定行打 new=／reused=")
    ds.add_argument("--out", default=None, help="可选：把计数与明细写成 JSON")
    ds.set_defaults(func=cmd_delivery_set)
    tv = sub.add_parser("tier-values", help="v8 档位取值逐格等于表 1（只读，纯 CPU）",
                        description=cmd_tier_values.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_argument("--specs-root", required=True, help="v8 /4 规格根")
    tv.add_argument("--cells", default="full", help=_CELLS_HELP)
    tv.add_argument("--selected", action="store_true", help="取 selected 行（生成前核对抽签），默认取交付行")
    tv.add_argument("--out", default=None, help="可选：逐格明细与直方图写成 JSON")
    tv.set_defaults(func=cmd_tier_values)
    rr = sub.add_parser("reset-replay", help="每格 1 局经评估链 reset 回注（GPU；V8／V9_RESET_REPLAY 按规格根推）",
                        description=cmd_reset_replay.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rr.add_argument("--out", required=True,
                    help="必填：结果 jsonl 文件路径（不得是目录；v9 用 artifacts/newtask-v9/gates/reset-replay.jsonl）；"
                         "续跑键 (task, tier, seed, spec_sha256)")
    rr.add_argument("--limit", type=int, default=0)
    rr.add_argument("--specs-root", default=None,
                    help="规格根（换包前经 ROBOMME_HARD_SPECS_ROOT 读）；/4 根按 header 配额推格表：V9 → V9_RESET_REPLAY、"
                         "V8 → V8_RESET_REPLAY；非 /4 根拒收")
    rr.set_defaults(func=cmd_reset_replay)
    ev = sub.add_parser("eval-smoke")
    ev.add_argument("--task", default="BinFill")
    ev.add_argument("--episode", type=int, default=0)
    ev.add_argument("--max-policy-steps", type=int, default=50)
    ev.add_argument("--specs-root", default=None)
    ev.set_defaults(func=cmd_eval_smoke)
    x0 = sub.add_parser("xhard0-reset-parity")
    x0.add_argument("--src-root", required=True, help="官方 1fadc0ec worktree（只导入其 robomme）")
    x0.add_argument("--manifest", default=str(REPO / "scripts" / "configs" / "xhard0" / "xhard0_manifest.json"))
    x0.add_argument("--gpu", default="0")
    x0.add_argument("--tasks", default=None)
    x0.add_argument("--out", required=True)
    x0.add_argument("--merge-with", default=None, help="与上一轮 jsonl 合并：本轮重跑的任务整段替换，其余沿用")
    x0.add_argument("--report-name", default="xhard0-reset-parity.jsonl")
    x0.set_defaults(func=cmd_xhard0_reset_parity)
    mc = sub.add_parser("movecube-layout", help="v9：MoveCube xhard4 两段三物体落点在新区域 U 内、运动方式 17/17/16"
                                                "（V9_MOVECUBE_LAYOUT／V9_MOVECUBE_WAYS；纯 CPU，不 reset）",
                        description=cmd_movecube_layout.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_argument("--specs-root", required=True,
                    help="规格根（读 <根>/<--tier>/specs.jsonl，如 MoveCube 生成片的 <片>/specs 或 artifacts/newtask-v9/specs-root）"
                         "或单个 specs.jsonl")
    mc.add_argument("--tier", default="xhard4", help="规格根下的档目录名（缺省 xhard4）")
    mc.add_argument("--delivery", default=None,
                    help="可选：交付清单 json（delivery.json 或 800 行 delivery.local.json）；给了只取 selected 且身份在清单里的行，"
                         "不给取交付行（selected 且 rollout ok）")
    mc.add_argument("--expected-episodes", type=int, default=0,
                    help="期望局数（缺省 V9_CELLS 的 MoveCube/xhard4 = 50；点数 = 局数 × 2 段 × 3 物体）")
    mc.add_argument("--out", default=None, help="可选：计数与明细写成 JSON")
    mc.set_defaults(func=cmd_movecube_layout)
    sh = sub.add_parser("step-headroom", help="v8／v9：交付 h5 非演示步 ≤ 1600、超限过滤数、xhard0 ≤ 1300"
                                              "（V8_STEP_CAP；清单行带 source 或只能被 V9_CELLS 覆盖时 V9_STEP_CAP）",
                        description=_step_headroom_v8.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sh.add_argument("--delivery", required=True,
                    help="gen1 的 delivery.json 或 v9 assemble 的 800 行清单（v8-delivery/1 走 v8 判据，判定行前缀按清单"
                         "推版本；行里 path 相对其所在目录）")
    sh.add_argument("--pool", nargs="+", default=None,
                    help="v8 必填：候选池（生成输出目录、/4 规格根或 results 文件，可多个），数 exec_over_cap 候选")
    sh.add_argument("--xhard0", default=None,
                    help="v8：xhard0 h5 来源（hard_parity 一侧目录或 identities.jsonl），按 1300 单独查；正式闸门必须带")
    sh.add_argument("--skip-xhard0", action="store_true",
                    help="v8：显式不查 xhard0（只供局部核对；判定行打 V8_STEP_CAP=INFO … xhard0_max=skipped，不出 PASS）")
    sh.add_argument("--out", default=None)
    sh.set_defaults(func=cmd_step_headroom)
    ed = sub.add_parser("env-digest", help="v7.5eval 环境检测：逐层摘要 + 原始数组 + 测速（GPU）")
    ed.add_argument("--cell", required=True, help="条件名（输出子目录）")
    ed.add_argument("--identities", required=True, help="身份清单 json：[{task, source_episode, seed, builder_episode, ...}]")
    ed.add_argument("--out", required=True)
    ed.add_argument("--resident", action="store_true", help="全部身份放进一个常驻进程（默认每任务一个新进程）")
    ed.add_argument("--reverse", action="store_true", help="身份倒序")
    ed.add_argument("--fixed-steps", type=int, default=30)
    ed.add_argument("--gpu", default=None, help="子进程的 CUDA_VISIBLE_DEVICES（不给则继承）")
    ed.add_argument("--limit", type=int, default=0)
    ed.add_argument("--allow-resume-resident", action="store_true",
                    help="--resident 下允许续跑（续跑会把一个常驻进程拆成多个，改变条件；默认拒绝）")
    ed.set_defaults(func=cmd_env_digest)
    ew = sub.add_parser("env-digest-worker", help=argparse.SUPPRESS)
    ew.add_argument("--cell", required=True)
    ew.add_argument("--out", required=True)
    ew.add_argument("--batch", required=True)
    ew.add_argument("--fixed-steps", type=int, default=30)
    ew.add_argument("--proc-index", type=int, default=0)
    ew.add_argument("--mode", default="per-task-forward")
    ew.add_argument("--resume-generation", type=int, default=0)
    ew.set_defaults(func=cmd_env_digest_worker)
    ec = sub.add_parser("env-digest-compare", help="两格环境检测结果逐层对拍（纯 CPU，只报告）")
    ec.add_argument("--a", required=True, help="<out>/<cell> 目录")
    ec.add_argument("--b", required=True)
    ec.add_argument("--out", required=True, help="json 明细")
    ec.set_defaults(func=cmd_env_digest_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
