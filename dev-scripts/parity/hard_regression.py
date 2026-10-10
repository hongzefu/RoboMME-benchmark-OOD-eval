#!/usr/bin/env python3
"""robomme_ood 回归工具（评估仓 dev-scripts/parity；benchmark 包取自子模块 third_party/robomme_benchmark）。

现行口径 V9（1002-newtask-v9-movecube-region-800-plan.md 第二部分 §2.3）；xhard0 reset 层对拍源自 v7 的 0928 方案
第二部分 §1.5。按数据集名取局：``hard-verify`` 只有 xhard0（每任务 12 局，局 0～11 对应官方原 episode 3,7,…,47），
``ood`` 只有新值五档（V9 每任务 50 局）。

子命令：

* ``tier-values``（纯 CPU，v8）：14 个有取值维度的任务逐格逐局取值等于表 1（RouteStick／PatternLock 落在区间内，
  另打印逐格长度直方图）→ ``V8_TIER_VALUES``；规格根缺省读子模块包内。
* ``eval-smoke``（GPU，1 任务 × 1 数据集 × 1 局）：评估链可用；xhard0 局须为导出模式；每任务局数按数据集推出
  （hard-verify 12、ood 按交付格表）→ ``HARD_EVAL_SMOKE``。
* ``xhard0-reset-parity``（GPU）：官方 robomme（``dataset="test"``）与 robomme_ood（``dataset="hard-verify"``）
  两进程各 reset 所选局（``--tasks``、``--episodes a:b``），确定性层逐位比 → ``XHARD0_RESET_PARITY``（分母按所选
  身份计）；演示层只报告 ``XHARD0_DEMO_DIFF=INFO``。
* ``env-digest``（GPU，v7.5eval）：xhard0 身份逐层摘要 + 原始数组 + 测速 → ``ENV_DIGEST_DONE``、``ENV_SPEED=INFO``。
* ``env-digest-compare``（纯 CPU，v7.5eval）：两格逐层对拍，只报告 → ``ENV_DIGEST_PARITY``。

拆仓时删去的生成阶段子命令（git 历史可取回，旧仓 tag 见拆分方案）：``delivery-set``、``reset-replay``、
``step-headroom``、``movecube-layout``，以及经 ``ROBOMME_HARD_SPECS_ROOT`` 换规格根的 ``--specs-root`` 参数
（benchmark 子模块已不认外部规格根）。更早的 ``s4-subset``、``layout-shared``、``prefix-geometry``、
``xhard0-eval-parity``、``step-headroom --v7`` 与 v7 口径于 1003 维护计划细则 2.3 删除。
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import _common  # noqa: E402  同目录：评估仓根、子模块根、hard_specs 轻量加载

REPO = _common.REPO_ROOT
#: 官方侧 xhard0 身份清单（16 任务 × 12 局，官方原 episode 号与 seed）
XHARD0_MANIFEST = _common.CONFIGS / "xhard0" / "xhard0_manifest.json"


def _hard_specs():
    from robomme_ood.env_record_wrapper import hard_specs

    return hard_specs


# ── 记录点路径（静态收集 SpecRecorder.record 的第一个参数）─────────────────────


def record_path_patterns(task: str, root: Path | None = None, method: str = "record") -> set[str]:
    """该任务环境文件与共用 utils 里 ``*.record(<路径>, …)`` 的路径正则（按任务分开，不同任务同名路径可能一记一注）；
    f-string 的占位符只匹配一个路径段（``[^.]+``）。``root`` 缺省为子模块的 ``robomme_ood/robomme_env``。"""
    import re

    if root is None:
        root = _common.bench_root() / "src" / "robomme_ood" / "robomme_env"

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
    from robomme_ood.env_record_wrapper.hard_specs import _max_abs_diff

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
    return _common.sibling("hard_parity")


def _hs_light():
    """不经 robomme_ood 包 __init__（会连带导入仿真）的 hard_specs；纯 CPU 守卫与规格读取用它。"""
    return _hp().hard_specs_light()


#: xhard0 的执行步上限（与官方 scripts/evaluation.py 的默认步数相同）。按档的步数查表已从 hard_specs 删除，
#: 评估入口按数据集传 max_steps；本工具按局的档取值（hard-verify 全为 xhard0，ood 全为新值档）。
XHARD0_STEP_CAP = 1300


def _cap_for(tier: str) -> int:
    """对拍工具内部的逐局步数上限：xhard0 取 ``XHARD0_STEP_CAP``（1300），新值档取 ``hard_specs.EXEC_CAP``（1600）。

    逐局传给 ``make_env_for_episode(max_steps=…)``：不传就退回 builder 的单一值，混档下会错配其中一类局。"""
    return XHARD0_STEP_CAP if tier == "xhard0" else _hs_light().EXEC_CAP


def specs_tiers(specs_root: str | None = None) -> tuple[tuple[str, ...], bool]:
    """规格根（显式给出，否则包内）的档序与是否 /4：``TIERS`` 下任一存在的档文件 header
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
    """(task, tier, seed) → {row, builder_episode}；builder 号与 ``dataset="ood"`` 的 hard_builder 相同（ood 不含 xhard0）：
    档序主序、档内 candidate 升序，从 0 起编（v7 方案第二部分 §7.2）。v8 规格按 ``TIERS`` 读五档。"""
    hs = _hs_light()
    tiers, v8 = specs_tiers(specs_root)
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    offsets: dict[str, int] = collections.Counter({task: 0 for task in hs.ALL_TASKS})
    for tier in tiers:
        path = hs.specs_root(specs_root) / tier / "specs.jsonl"
        if v8 and not path.is_file():
            continue  # v8 局部根（冒烟／分片）只含部分档；完整性由生成侧交付守卫负责
        # 配额上限格表按 header 推出（V9 文件 → V9_CELLS）
        _, rows = _hp().load_specs_any(path, hs)
        for task in hs.ALL_TASKS:
            chosen = sorted((r for r in rows if r["task"] == task and hs.delivered(r)), key=lambda r: r["candidate"])
            for row in chosen:
                index[(task, tier, int(row["seed"]))] = {"row": row, "builder_episode": offsets[task]}
                offsets[task] += 1
    return index


# ── eval-smoke（GPU）────────────────────────────────────────────────────


def expected_episodes(task: str, hs, dataset: str = "ood", cells: dict[tuple[str, str], int] | None = None) -> int:
    """builder 每任务局数（按数据集名，不写死）：``hard-verify`` 只有 xhard0，即 ``XHARD0_PER_TASK``（12）；
    ``ood`` 只有新值五档，即交付格表在该任务的局数之和（缺省当前 ``EXPECTED_CELLS``；V9_CELLS 每任务 50）。"""
    if dataset == "hard-verify":
        return int(hs.XHARD0_PER_TASK)
    if dataset != "ood":
        raise ValueError(f"未知数据集 {dataset!r}（只认 hard-verify／ood）")
    table = cells if cells is not None else hs.EXPECTED_CELLS
    return sum(n for (name, _), n in table.items() if name == task)


def cmd_eval_smoke(args) -> int:
    """合作者入口：与评估入口同样的构建，只跑 1 任务 × 1 数据集 × 1 局；逐局 max_steps 按 :func:`_cap_for`
    取（xhard0 1300、新值档 1600）。``--dataset`` 按数据集名取局（hard-verify 全为 xhard0，ood 全为新值档）。
    xhard0 局须为导出模式（原生 hard 分支）、回注局须为 replay 且 injected_mismatch==0。"""
    import numpy as np

    from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder, spec_binding

    hs = _hard_specs()
    builder = BenchmarkEnvBuilder(env_id=args.task, dataset=args.dataset, action_space="joint_angle", max_steps=1300)
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
    expected = expected_episodes(args.task, hs, args.dataset)
    ok = mode_ok and binding.get("injected_mismatch") == 0 and status != "error" and num == expected
    print(f"HARD_EVAL_SMOKE={'PASS' if ok else 'FAIL'} task={args.task} dataset={args.dataset} episode={args.episode} "
          f"tier={tier} seed={seed} "
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
    assert "robomme_ood" not in sys.modules
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    assert "robomme_ood" not in sys.modules
    builder = BenchmarkEnvBuilder(task, dataset="test")
else:
    from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(task, dataset="hard-verify")
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
    source_episode = builder.resolve_identity(ep).get("source_episode") if side != "official" else ep
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
    out.append({"episode": ep, "source_episode": source_episode, "seed": seed, "difficulty": diff,
                "make_kwargs": make_kw, "wrapper_chain": chain,
                "pre_demo_state": pre, "task_goal": [str(g) for g in (goal if isinstance(goal, (list, tuple)) else [goal])],
                "choices": json.loads(json.dumps(choices, default=str)) if choices is not None else None,
                "demo_frames": max(len(demo) - 1, 0), "demo_digest": digest(np.stack([np.asarray(f) for f in demo])) if len(demo) else None,
                "post_state": digest(env.unwrapped.get_state_dict())})
    env.close()
if side == "official":
    assert "robomme_ood" not in sys.modules, "官方侧进程不得导入 robomme_ood"
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


def parse_episodes(text: str | None, total: int) -> list[int]:
    """``--episodes a:b`` → hard-verify 的 builder 局号半开区间 ``[a, b)``；不给即全部 ``0..total-1``。
    须 ``0 ≤ a < b ≤ total``（越界即报错，不静默截断）。"""
    if not text:
        return list(range(total))
    head, sep, tail = str(text).partition(":")
    try:
        lo, hi = int(head), int(tail)
    except ValueError as exc:
        raise SystemExit(f"--episodes 须写成 a:b（半开区间）：{text!r}") from exc
    if not sep or lo < 0 or hi <= lo or hi > total:
        raise SystemExit(f"--episodes 须满足 0 ≤ a < b ≤ {total}：{text!r}")
    return list(range(lo, hi))


def cmd_xhard0_reset_parity(args) -> int:
    """XHARD0_RESET_PARITY（D-16）：同卡两进程——官方侧只导入 ``--src-root``（缺省 benchmark 子模块）的 robomme
    （``dataset="test"``，官方原 episode 号；进程内断言未导入 robomme_ood），robomme_ood 侧 ``dataset="hard-verify"``
    取 ``--episodes``（builder 局号半开区间，缺省 0～11；局 0、1 对应官方原 episode 3、7）。

    先跑 hard 侧，官方侧的原 episode 号以 builder 解析出的 ``source_episode`` 为准，并与 ``--manifest`` 第 i 行逐局
    核对。确定性层逐位比：官方原 episode 号、seed、gym.make 实参（除 hard 侧不应有的键）、包装链类名序列、演示回放前
    的底层状态（另起底层环境 reset 取 get_state_dict）、task_goal、多选项；演示层（演示帧数、演示帧、演示后状态）
    只报告。分母按所选身份计：任务数 × 所选局数（判定行 ``shape=<任务数>x1x<局数>``）。"""
    hs = _hs_light()
    manifest = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    unknown = sorted(set(wanted or ()) - set(hs.ALL_TASKS))
    if unknown:
        raise SystemExit(f"--tasks 含未知任务：{unknown}")
    tasks = [t for t in hs.ALL_TASKS if wanted is None or t in wanted]
    episodes = parse_episodes(args.episodes, int(hs.XHARD0_PER_TASK))
    src_root = Path(args.src_root) if args.src_root else _common.bench_root()
    for task in tasks:
        source_eps = sorted(int(r["episode"]) for r in manifest["rows"] if r["task"] == task)
        if len(source_eps) != int(hs.XHARD0_PER_TASK):
            raise SystemExit(f"{task} 在 xhard0 清单里有 {len(source_eps)} 局，应为 {hs.XHARD0_PER_TASK}")
        hard = _run_probe("hard", task, _common.bench_root(), episodes, args.gpu)
        official_eps = [int(h["source_episode"]) if h.get("source_episode") is not None else -1 for h in hard]
        official = _run_probe("official", task, src_root, official_eps, args.gpu)
        for index, (o, h) in enumerate(zip(official, hard)):
            fields = {
                "source_episode": o["episode"] == h.get("source_episode") == source_eps[episodes[index]],
                "seed": o["seed"] == h["seed"],
                "difficulty": o["difficulty"] == "hard" and h["difficulty"] == "xhard0",
                "make_kwargs": o["make_kwargs"] == h["make_kwargs"],
                "wrapper_chain": o["wrapper_chain"] == h["wrapper_chain"],
                "pre_demo_state": o["pre_demo_state"] == h["pre_demo_state"],
                "task_goal": o["task_goal"] == h["task_goal"],
                "choices": o["choices"] == h["choices"],
            }
            bad = [k for k, v in fields.items() if not v]
            # 演示前状态确有实体改名（键集合不同）、且每类 actor 的状态值集合相等 ⇒ 只是命名不同（如 robomme_ood BUS
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
        if len(official) != len(episodes) or len(hard) != len(episodes):
            print(f"# {task} 探针局数不符：官方 {len(official)}、hard {len(hard)}、应为 {len(episodes)}", flush=True)
        mine = [r for r in rows if r["task"] == task]
        print(f"XHARD0_RESET_TASK {task} compared={len(mine)} det_bad={sum(bool(r['det_bad']) for r in mine)} "
              f"name_only={sum(r['name_only'] for r in mine)}", flush=True)
    merged_tasks: set[str] = set()
    if args.merge_with:
        previous = [r for r in map(json.loads, Path(args.merge_with).read_text().splitlines()) if r["task"] not in tasks]
        merged_tasks = {r["task"] for r in previous}
        rows += previous
    (out / args.report_name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return _xhard0_reset_verdict(rows, len(episodes), n_tasks=len(tasks) + len(merged_tasks))


def _name_agnostic(state: Any) -> Any:
    """状态摘要按类（actors／articulations…）取值的有序多重集，忽略实体名字。"""
    if isinstance(state, dict) and all(isinstance(v, dict) for v in state.values()):
        return {section: sorted(json.dumps(v, sort_keys=True) for v in items.values()) for section, items in state.items()}
    return state


#: 已声明的实体改名：任务 → 分区 → {官方侧名: robomme_ood 侧名}。ButtonUnmaskSwap 的 F3 把左右按钮名对调
#: （官方 buttons[0]（y=-0.1）名 button_left，robomme_ood 名 button_right），两侧键集合相同、取值互换，
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
    """期望局数 = 任务数 × 每任务所选局数（调用方按所选身份给出；``n_tasks`` 未给时取 ``hard_specs.ALL_TASKS`` 的长度），
    不写死 16 × 12。"""
    if n_tasks is None:
        n_tasks = len(_hs_light().ALL_TASKS)
    det = [r for r in rows if r["det_bad"]]
    first = f"{det[0]['task']}/ep{det[0]['source_episode']}:{det[0]['det_bad']}" if det else "-"
    name_only = [r for r in rows if r.get("name_only")]
    ok = not det and len(rows) > 0 and len(rows) == n_tasks * per_task
    print(f"XHARD0_RESET_PARITY={'PASS' if ok else 'FAIL'} shape={n_tasks}x1x{per_task} compared={len(rows)} "
          f"det_diff={len(det)} "
          f"name_only={len(name_only)} first_det_diff={first}"
          + (f" name_only_tasks={sorted({r['task'] for r in name_only})}" if name_only else ""))
    print(f"XHARD0_DEMO_DIFF=INFO frames_equal={sum(r['demo_frames'][0] == r['demo_frames'][1] for r in rows)} "
          f"max_frame_diff={max((abs(r['demo_frames'][0] - r['demo_frames'][1]) for r in rows), default=0)} "
          f"demo_equal={sum(r['demo_equal'] for r in rows)} post_equal={sum(r['post_equal'] for r in rows)}")
    return 0 if ok else 1


# ── V8 守卫（1001 方案第一部分 §3 验收表、第二部分 §2.3 闸门总表）：只读、纯 CPU ───────────────────────


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
    files, missing_files, bad_files = _read_v8_root(args.specs_root or hs.PACKAGED_SPECS_ROOT, valued)
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
    "proc.import_robomme_ood_s": "import robomme_ood.env_record_wrapper（连带注册 16 个任务类）",
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

    from robomme_ood.robomme_env.utils.vqa_options import get_vqa_options

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
                        ("robomme_ood", ("robomme_ood.env_record_wrapper",))):
        started = time.perf_counter()
        for mod in mods:
            __import__(mod)
        proc[f"import_{label}_s"] = round(time.perf_counter() - started, 6)
    from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder

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
            # 身份全为 xhard0（官方 test 的 hard 子集）：按数据集名取局，builder_episode 即 hard-verify 的局号 0～11
            builder = builders.setdefault(ident["task"], BenchmarkEnvBuilder(
                env_id=ident["task"], dataset="hard-verify", action_space="joint_angle", max_steps=1300))
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
        "import_robomme_ood_s_p50": _env_p50([p.get("import_robomme_ood_s") for p in firsts]),
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


_CELLS_HELP = ("格表（判定行前缀随格表版本 V9_）：full（默认＝当前 hard_specs.EXPECTED_CELLS，即 V9 的 43 格 800）"
               "｜v9full 或 v9（V9_CELLS 43 格 800）｜v9smoke（MoveCube／InsertPeg xhard4 各 1 局）"
               "｜v9shard1（V9 MoveCube 一片）｜JSON 文件路径或内联 JSON（分片子集，形如 {\"PickXtimes/xhard1\": 17}、"
               "{\"MoveCube@xhard4\": 50} 或 [[task, tier, n], ...]；须能被 V9_CELLS 覆盖）。"
               "V8 专用的 v8full／smoke 已删除")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tv = sub.add_parser("tier-values", help="v8 档位取值逐格等于表 1（只读，纯 CPU）",
                        description=cmd_tier_values.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_argument("--specs-root", default=None, help="/4 规格根（缺省子模块包内 env_metadata/ood）")
    tv.add_argument("--cells", default="full", help=_CELLS_HELP)
    tv.add_argument("--selected", action="store_true", help="取 selected 行（生成前核对抽签），默认取交付行")
    tv.add_argument("--out", default=None, help="可选：逐格明细与直方图写成 JSON")
    tv.set_defaults(func=cmd_tier_values)
    ev = sub.add_parser("eval-smoke", help="1 任务 × 1 数据集 × 1 局经评估链跑通（GPU；HARD_EVAL_SMOKE）",
                        description=cmd_eval_smoke.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ev.add_argument("--task", default="BinFill")
    ev.add_argument("--dataset", default="ood", choices=("hard-verify", "ood"), help="按数据集名取局（缺省 ood）")
    ev.add_argument("--episode", type=int, default=0)
    ev.add_argument("--max-policy-steps", type=int, default=50)
    ev.set_defaults(func=cmd_eval_smoke)
    x0 = sub.add_parser("xhard0-reset-parity", help="官方 robomme 与 robomme_ood（hard-verify）两进程 reset 确定性层逐位比"
                                                    "（GPU；XHARD0_RESET_PARITY）",
                        description=cmd_xhard0_reset_parity.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    x0.add_argument("--src-root", default=None,
                    help="官方侧源码树根（只导入其 src/robomme；缺省 benchmark 子模块，其 robomme 与官方 1fadc0ec 逐字节相同）")
    x0.add_argument("--manifest", default=str(XHARD0_MANIFEST),
                    help="xhard0 清单（官方原 episode 号与 seed；缺省 dev-scripts/parity/configs/xhard0/xhard0_manifest.json）")
    x0.add_argument("--episodes", default=None,
                    help="hard-verify 的 builder 局号半开区间 a:b（缺省 0:12 全部；局 0、1 对应官方原 episode 3、7）")
    x0.add_argument("--gpu", default="0")
    x0.add_argument("--tasks", default=None, help="逗号分隔的任务子集（缺省 16 任务）")
    x0.add_argument("--out", required=True)
    x0.add_argument("--merge-with", default=None, help="与上一轮 jsonl 合并：本轮重跑的任务整段替换，其余沿用")
    x0.add_argument("--report-name", default="xhard0-reset-parity.jsonl")
    x0.set_defaults(func=cmd_xhard0_reset_parity)
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
