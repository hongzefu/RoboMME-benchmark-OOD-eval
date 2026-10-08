"""固定检查集：V9 每格 3 局（43 格 × 3 = 129）与 xhard0 每任务 3 局（16 任务 × 1 档 × 3 = 48）
（1003 噪声基线计划第一部分「固定检查集」、第二部分 §一、§二 S1 行）。噪声基线与以后每次回归都用这同一批局。

规模来源：用户 2026-10-03 原话「全面减少现在的规模把V9xhard0每个任务每个难度都只有3集」（原为 V9 每任务 12 局
= 192、xhard0 全部 192）。

一、V9 检查集（G9，确定、可重算，只读包内规格，不读 ``artifacts/``）：

1. 数据源：benchmark 子模块包内 V9 规格 ``src/robomme_hard/env_metadata/ood/xhard{1..5}/specs.jsonl``；交付行＝
   ``hard_specs.delivered(row)`` 为真的行（``selected`` 且 ``rollout.status == "ok"``）。交付行与 builder episode 号
   直接复用 ``hard_regression.delivery_index``（与 ``hard_builder._ood_entries`` 同一排序：档序主序、档内
   ``candidate`` 升序；ood 不含 xhard0），本模块不另造一套。
2. 每个交付格（任务, 档）取交付行里 ``candidate`` 升序最小的 ``PER_CELL = 3`` 个；格数按数据算（V9 为 43 格）。
3. ``builder_episode`` 是该身份在 ``BenchmarkEnvBuilder(env_id=task, dataset="ood")`` 里的局号（0～49；按数据集名取局，
   ood 只有新值五档）。

行内排序：(16 任务规范序, tier, candidate)。规范序取 ``hard_specs.ALL_TASKS``（原 ``seed_layout.ALL_TASKS`` 与之逐字
相同，随拆仓不再搬入，一律以 ``hard_specs.ALL_TASKS`` 为准）。

冻结文件 ``dev-scripts/parity/configs/gate-set-v9-129.json``：``{"schema", "source", "rule", "rows", "sha256"}``，
``sha256`` 是剔掉该键后 canonical JSON（``sort_keys=True, ensure_ascii=False, separators=(",", ":")``）的 sha256。

二、xhard0 检查集（直接采用 v7.5 评估已用过的小样本 ``artifacts/v7.5eval/identities-small48.json``，每任务 3 局）：

``build_xhard0_set`` 把源文件规范化为行 ``{task, tier:"xhard0", seed, source_episode, builder_episode}``（丢弃
``e0_*`` 排序字段——``env_client`` 旧路线只读 task／source_episode／seed／builder_episode），按 (任务规范序,
source_episode) 排序。``builder_episode`` 沿用源文件，＝(source_episode-3)//4，恰是该身份在
``BenchmarkEnvBuilder(env_id=task, dataset="hard-verify")`` 里的局号（0～11）。
冻结为 ``dev-scripts/parity/configs/gate-set-xhard0-48.json``（``schema: gate-set-xhard0/1``，``source`` 记源文件名与 sha256，
``rows``、顶层 ``sha256`` 同上 canonical JSON 规则）。冻结后不再依赖 ``artifacts/``。

用法（仓库根）::

    C=dev-scripts/parity/configs
    uv run --no-sync python dev-scripts/parity/gate_set.py build --out $C/gate-set-v9-129.json
    uv run --no-sync python dev-scripts/parity/gate_set.py check --path $C/gate-set-v9-129.json
    uv run --no-sync python dev-scripts/parity/gate_set.py build-xhard0 --source artifacts/v7.5eval/identities-small48.json \\
        --out $C/gate-set-xhard0-48.json
    uv run --no-sync python dev-scripts/parity/gate_set.py check-xhard0 --path $C/gate-set-xhard0-48.json
    uv run --no-sync python dev-scripts/parity/gate_set.py export --set v9 --kind generate --out <dir>/g9.identities.jsonl
    uv run --no-sync python dev-scripts/parity/gate_set.py export --set xhard0 --kind legacy --out <dir>/x0.identities.json

``check`` 末行判定 ``GATE_SET=PASS cells=43 per_cell=3 total=129 in_delivery=129 sha=<前12位>``；``check-xhard0`` 末行
``GATE_SET_XHARD0=PASS tasks=16 per_task=3 total=48 sha=<前12位>``（不符为 FAIL）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import _common  # noqa: E402  同目录：评估仓根、配置目录、hard_specs 轻量加载

REPO = _common.REPO_ROOT

SCHEMA = "gate-set-v9/2"
#: 每个交付格取 3 局——用户 2026-10-03 原话「全面减少现在的规模把V9xhard0每个任务每个难度都只有3集」
PER_CELL = 3
N_TASKS = 16
DEFAULT_PATH = _common.CONFIGS / "gate-set-v9-129.json"
ROW_KEYS = ("task", "tier", "candidate", "seed", "spec_sha256", "builder_episode")
#: 与冻结配置 gate-set-v9-129.json 的 rule 字段逐字节相同。拆仓时随 xhard0 前置开关删除改写（编号本身不变：
#: 原「开关为 0 时的 test-hard 编号」就是现在 dataset="ood" 的局号），并按 canonical JSON 重签该文件与噪声参照
RULE = ("V9 包内规格交付行（hard_specs.delivered），每个交付格（任务, 档）取 candidate 升序最小的 3 个；"
        "builder_episode 按 dataset=ood（不含 xhard0）的局号")

XHARD0_SCHEMA = "gate-set-xhard0/1"
XHARD0_TIER = "xhard0"
#: xhard0 每任务 3 局（同上用户原话）
XHARD0_PER_TASK = 3
XHARD0_DEFAULT_PATH = _common.CONFIGS / "gate-set-xhard0-48.json"
XHARD0_ROW_KEYS = ("task", "tier", "seed", "source_episode", "builder_episode")
XHARD0_RULE = ("直接采用 v7.5 评估小样本 identities-small48.json（每任务 3 局）；行规范化为 task／tier=xhard0／seed／"
               "source_episode／builder_episode（开关为 1 时的 xhard0 前置编号），按 (任务规范序, source_episode) 排序")


class GateSetError(RuntimeError):
    """检查集构建、读取或核对失败。"""


# ── 依赖（复用 hard_parity／hard_regression 的轻量入口，不导入仿真）─────────────────


def _hs():
    return _common.hard_specs_light()


def _delivery_index(specs_root: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    return _common.sibling("hard_regression").delivery_index(specs_root)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def payload_sha256(obj: dict[str, Any]) -> str:
    """剔掉顶层 ``sha256`` 键后 canonical JSON 的 sha256。"""
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# ── V9 选局 ─────────────────────────────────────────────────────────────────


def _specs_root(hs, specs_root: str | Path | None) -> str:
    """缺省固定为子模块包内规格根（冻结清单只认包内 V9 规格）。"""
    return str(Path(specs_root).resolve() if specs_root is not None else hs.PACKAGED_SPECS_ROOT)


def build_gate_set(specs_root: str | Path | None = None) -> list[dict[str, Any]]:
    """按模块文档的规则算出每格 3 局的行 ``{task, tier, candidate, seed, spec_sha256, builder_episode}``，
    按 (任务规范序, tier, candidate) 排序。某格交付局数不足 3 即报错。"""
    hs = _hs()
    index = _delivery_index(_specs_root(hs, specs_root))
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (task, tier, seed), hit in index.items():
        if not hs.delivered(hit["row"]):
            raise GateSetError(f"delivery_index 返回了非交付行：{task}@{tier} seed={seed}")
        cells.setdefault((task, tier), []).append(hit)
    rows: list[dict[str, Any]] = []
    for (task, tier), hits in cells.items():
        if len(hits) < PER_CELL:
            raise GateSetError(f"{task}@{tier} 交付局数 {len(hits)} < {PER_CELL}")
        for hit in sorted(hits, key=lambda h: int(h["row"]["candidate"]))[:PER_CELL]:
            row = hit["row"]
            rows.append({
                "task": task,
                "tier": tier,
                "candidate": int(row["candidate"]),
                "seed": int(row["seed"]),
                "spec_sha256": row["spec_sha256"],
                "builder_episode": int(hit["builder_episode"]),
            })
    order = {task: i for i, task in enumerate(hs.ALL_TASKS)}
    rows.sort(key=lambda r: (order[r["task"]], r["tier"], r["candidate"]))
    return rows


def allocation_table(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """{task: {tier: 局数}}，报告与测试用。"""
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = table.setdefault(row["task"], {})
        cell[row["tier"]] = cell.get(row["tier"], 0) + 1
    return table


def source_headers(specs_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """各档 specs header 的 ``delivery_sha256`` 与 ``identity_sha256``（有则记），写进冻结文件的 ``source``。"""
    hs = _hs()
    root = Path(_specs_root(hs, specs_root))
    out: dict[str, dict[str, Any]] = {}
    for tier in hs.TIERS:
        path = root / tier / "specs.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            header = json.loads(stream.readline())
        out[tier] = {key: header[key] for key in ("delivery_sha256", "identity_sha256") if key in header}
    return out


def build_payload(specs_root: str | Path | None = None) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "schema": SCHEMA,
        "source": source_headers(specs_root),
        "rule": RULE,
        "rows": build_gate_set(specs_root),
    }
    obj["sha256"] = payload_sha256(obj)
    return obj


def dumps_payload(obj: dict[str, Any]) -> str:
    """冻结文件的确定字节：sort_keys、缩进 1、末尾换行（``check`` 逐字节比它）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1) + "\n"


def _load_frozen(path: str | Path, schema: str, row_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("schema") != schema:
        raise GateSetError(f"{path}：schema 应为 {schema}")
    want = obj.get("sha256")
    got = payload_sha256(obj)
    if want != got:
        raise GateSetError(f"{path}：sha256 不符（文件 {want}，重算 {got}）")
    rows = obj.get("rows")
    if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != set(row_keys) for r in rows):
        raise GateSetError(f"{path}：rows 须为键恰为 {row_keys} 的对象列表")
    return rows


def load_gate_set(path: str | Path = DEFAULT_PATH) -> list[dict[str, Any]]:
    """读 V9 冻结 json，校验 schema、行键与顶层 ``sha256``；不一致抛 ``GateSetError``。返回行列表。

    文件 schema 为 ``gate-set-xhard0/1`` 时转交 :func:`load_xhard0_set`——``noise_gate.load_identities`` 对任何
    ``gate-set*`` schema 都调本函数，这样 xhard0 冻结文件也能直接当身份清单用。"""
    try:
        schema = json.loads(Path(path).read_text(encoding="utf-8")).get("schema")
    except AttributeError:
        schema = None
    if schema == XHARD0_SCHEMA:
        return load_xhard0_set(path)
    return _load_frozen(path, SCHEMA, ROW_KEYS)


# ── xhard0 小样本 ────────────────────────────────────────────────────────────


def build_xhard0_set(source_path: str | Path) -> list[dict[str, Any]]:
    """从 v7.5 小样本 json（每行 builder_episode／e0_*／seed／source_episode／task）规范化出 xhard0 检查集行。"""
    hs = _hs()
    items = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise GateSetError(f"{source_path}：应为身份列表")
    rows = [{"task": str(r["task"]), "tier": XHARD0_TIER, "seed": int(r["seed"]),
             "source_episode": int(r["source_episode"]), "builder_episode": int(r["builder_episode"])}
            for r in items]
    order = {task: i for i, task in enumerate(hs.ALL_TASKS)}
    unknown = sorted({r["task"] for r in rows} - set(order))
    if unknown:
        raise GateSetError(f"{source_path}：未知任务 {unknown}")
    rows.sort(key=lambda r: (order[r["task"]], r["source_episode"]))
    _validate_xhard0_rows(rows, str(source_path))
    return rows


def _validate_xhard0_rows(rows: list[dict[str, Any]], where: str) -> None:
    """16 任务 × 3 局、身份不重复、builder_episode ＝ (source_episode-3)//4 且 source_episode ≡ 3 (mod 4)。"""
    table: dict[str, int] = {}
    for r in rows:
        table[r["task"]] = table.get(r["task"], 0) + 1
        if r["tier"] != XHARD0_TIER:
            raise GateSetError(f"{where}：tier 应为 {XHARD0_TIER}：{r}")
        se = int(r["source_episode"])
        if se % 4 != 3 or int(r["builder_episode"]) != (se - 3) // 4:
            raise GateSetError(f"{where}：builder_episode 与 source_episode 不自洽：{r}")
    if len(table) != N_TASKS or set(table.values()) != {XHARD0_PER_TASK}:
        raise GateSetError(f"{where}：规模应为 {N_TASKS} 任务 × {XHARD0_PER_TASK} 局，实为 {table}")
    if len({(r["task"], r["seed"]) for r in rows}) != len(rows):
        raise GateSetError(f"{where}：身份重复")


def build_xhard0_payload(source_path: str | Path) -> dict[str, Any]:
    source_path = Path(source_path)
    obj: dict[str, Any] = {
        "schema": XHARD0_SCHEMA,
        "source": {"file": source_path.name,
                   "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
        "rule": XHARD0_RULE,
        "rows": build_xhard0_set(source_path),
    }
    obj["sha256"] = payload_sha256(obj)
    return obj


def load_xhard0_set(path: str | Path = XHARD0_DEFAULT_PATH) -> list[dict[str, Any]]:
    """读 xhard0 冻结 json：校验 schema、行键、顶层 ``sha256`` 与 16 任务 × 3 局。"""
    rows = _load_frozen(path, XHARD0_SCHEMA, XHARD0_ROW_KEYS)
    _validate_xhard0_rows(rows, str(path))
    return rows


# ── 导出给下游入口 ──────────────────────────────────────────────────────────


def to_generate_identities(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """``hard_parity.py generate／compare／import-delivery --identities`` 的身份：``[{task, tier, seed}]``
    （xhard0 行的 tier 为 ``xhard0``，供 ``generate --tier xhard0 --identities``）。

    ``hard_parity.read_identity_subset`` 按文件后缀读：``.jsonl`` 每行一个 ``{task, tier|difficulty, seed}``；
    ``.json`` 为列表或带 ``rows``／``delivered`` 的对象（行带 ``source`` 时只取 ``v9-new``，本函数不写 ``source``）。
    写文件用 ``write_generate_identities``（jsonl）。"""
    return [{"task": r["task"], "tier": r["tier"], "seed": int(r["seed"])} for r in rows]


def to_env_digest_identities(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """``hard_regression.py env-digest --identities`` 的身份：json 列表 ``[{task, source_episode, seed, builder_episode}]``。

    ``_env_digest_one`` 用 ``builder.resolve_identity(builder_episode)`` 核对 seed 与 ``source_episode``；新值档的
    resolve_identity 不带 ``source_episode``，按 -1 比较，所以 V9 行固定写 -1；xhard0 行带真实 ``source_episode``。
    写文件用 ``write_env_digest_identities``。"""
    return [{"task": r["task"], "source_episode": int(r["source_episode"]) if "source_episode" in r else -1,
             "seed": int(r["seed"]), "builder_episode": int(r["builder_episode"])} for r in rows]


def to_legacy_identities(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """``scripts/eval-official/env_client.py`` 旧路线（非 ``--v8``）的 ``--identities``：与 small48 同格式的列表
    ``[{task, source_episode, seed, builder_episode}]``。旧路线只读这四个字段（``check_identity`` 核 tier=xhard0、
    seed、source_episode；``order_identities`` 按 task、source_episode 排），``e0_*`` 排序字段不需要，故不输出。
    只接受 xhard0 行。"""
    out = []
    for r in rows:
        if r.get("tier") != XHARD0_TIER or "source_episode" not in r:
            raise GateSetError(f"legacy 身份只适用 xhard0 行：{r}")
        out.append({"task": r["task"], "source_episode": int(r["source_episode"]), "seed": int(r["seed"]),
                    "builder_episode": int(r["builder_episode"])})
    return out


def to_eval_keys(rows: Iterable[dict[str, Any]]) -> set[str]:
    """V8 评估链路身份键 ``<task>_<tier>_<seed>``（与 ``scripts/eval-official/env_client.py::v8_key`` 同式）。"""
    return {f"{r['task']}_{r['tier']}_{int(r['seed'])}" for r in rows}


def write_generate_identities(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """写 jsonl（每行一个身份），供 ``hard_parity.py --identities``；后缀必须是 ``.jsonl``。"""
    path = Path(path)
    if path.suffix != ".jsonl":
        raise GateSetError(f"generate 身份文件须以 .jsonl 结尾（read_identity_subset 按后缀判格式）：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in to_generate_identities(rows)),
                    encoding="utf-8")
    return path


def _write_json_list(items: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def write_env_digest_identities(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """写 json 列表，供 ``hard_regression.py env-digest --identities``（它用 ``json.loads`` 读整个文件）。"""
    return _write_json_list(to_env_digest_identities(rows), path)


def write_legacy_identities(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """写 json 列表，供 ``env_client.py --identities``（旧路线）。"""
    return _write_json_list(to_legacy_identities(rows), path)


# ── 核对 ────────────────────────────────────────────────────────────────────


def check(path: str | Path, specs_root: str | Path | None = None) -> tuple[bool, str]:
    """重算清单，与冻结文件逐字节比；另核每格恰 3 局、每行属交付集。返回 (是否通过, 判定行)。"""
    reasons: list[str] = []
    path = Path(path)
    fresh = build_payload(specs_root)
    rows = fresh["rows"]
    # F-7：冻结文件读不到、为空或解析失败一律 FAIL（此前零字节文件会跳过全部文件侧核对而判 PASS）
    readable = True
    try:
        frozen_bytes = path.read_bytes()
    except OSError as exc:
        frozen_bytes = b""
        readable = False
        reasons.append(f"读不到冻结文件：{exc}")
    if readable and not frozen_bytes.strip():
        reasons.append("冻结文件为空")
    elif frozen_bytes:
        if frozen_bytes != dumps_payload(fresh).encode("utf-8"):
            reasons.append("冻结文件与重算结果字节不同")
        try:
            load_gate_set(path)
        except (GateSetError, ValueError, AttributeError, KeyError, TypeError) as exc:
            reasons.append(f"冻结文件解析失败：{type(exc).__name__}: {exc}")
    hs = _hs()
    index = _delivery_index(_specs_root(hs, specs_root))
    delivered_cells = {(t, tier) for (t, tier, _s) in index}
    table = allocation_table(rows)
    per_cell = sorted({n for cell in table.values() for n in cell.values()})
    n_cells = sum(len(cell) for cell in table.values())
    in_delivery = sum(
        1 for r in rows
        if (hit := index.get((r["task"], r["tier"], r["seed"]))) is not None
        and hs.delivered(hit["row"]) and int(hit["row"]["candidate"]) == r["candidate"]
        and hit["row"]["spec_sha256"] == r["spec_sha256"] and hit["builder_episode"] == r["builder_episode"])
    if n_cells != len(delivered_cells) or per_cell != [PER_CELL] or len(rows) != n_cells * PER_CELL:
        reasons.append(f"规模不符：cells={n_cells}（交付格 {len(delivered_cells)}） per_cell={per_cell} total={len(rows)}")
    if in_delivery != len(rows):
        reasons.append(f"有 {len(rows) - in_delivery} 行不在交付集")
    ok = not reasons
    per = per_cell[0] if len(per_cell) == 1 else "/".join(map(str, per_cell))
    line = (f"GATE_SET={'PASS' if ok else 'FAIL'} cells={n_cells} per_cell={per} total={len(rows)} "
            f"in_delivery={in_delivery} sha={fresh['sha256'][:12]}")
    for reason in reasons:
        print(f"GATE_SET_REASON {reason}", flush=True)
    return ok, line


def check_xhard0(path: str | Path, source: str | Path | None = None) -> tuple[bool, str]:
    """校验 xhard0 冻结文件（schema、sha256、16 × 3、自洽）；给 ``source`` 时另核源文件 sha 并重算逐字节比。"""
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    sha = "-"
    try:
        rows = load_xhard0_set(path)
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        sha = obj["sha256"]
    except (GateSetError, ValueError, OSError, KeyError) as exc:
        reasons.append(str(exc))
    if source is not None and not reasons:
        fresh = build_xhard0_payload(source)
        if dumps_payload(fresh).encode("utf-8") != Path(path).read_bytes():
            reasons.append("冻结文件与按源文件重算结果字节不同")
    table: dict[str, int] = {}
    for r in rows:
        table[r["task"]] = table.get(r["task"], 0) + 1
    per = sorted(set(table.values()))
    per_s = per[0] if len(per) == 1 else "/".join(map(str, per))
    ok = not reasons
    line = (f"GATE_SET_XHARD0={'PASS' if ok else 'FAIL'} tasks={len(table)} per_task={per_s} total={len(rows)} "
            f"sha={sha[:12]}")
    for reason in reasons:
        print(f"GATE_SET_XHARD0_REASON {reason}", flush=True)
    return ok, line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="固定检查集（V9 43 格 × 3 = 129；xhard0 16 任务 × 1 档 × 3 = 48）")
    parser.add_argument("--specs-root", default=None, help="规格根（缺省为包内 V9 规格；冻结清单只认包内）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build", help="写 V9 冻结 json")
    p_build.add_argument("--out", type=Path, default=DEFAULT_PATH)
    p_check = sub.add_parser("check", help="重算 V9 清单并与冻结文件逐字节比较")
    p_check.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p_bx = sub.add_parser("build-xhard0", help="从 v7.5 小样本 json 写 xhard0 冻结 json")
    p_bx.add_argument("--source", type=Path, required=True)
    p_bx.add_argument("--out", type=Path, default=XHARD0_DEFAULT_PATH)
    p_cx = sub.add_parser("check-xhard0", help="校验 xhard0 冻结 json（给 --source 时另按源文件重算比较）")
    p_cx.add_argument("--path", type=Path, default=XHARD0_DEFAULT_PATH)
    p_cx.add_argument("--source", type=Path, default=None)
    p_export = sub.add_parser("export", help="从冻结文件导出下游身份清单")
    p_export.add_argument("--set", dest="which", choices=("v9", "xhard0"), default="v9")
    p_export.add_argument("--kind", choices=("generate", "env-digest", "legacy"), required=True)
    p_export.add_argument("--path", type=Path, default=None, help="冻结清单（读前校验 sha256；缺省按 --set 取）")
    p_export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        obj = build_payload(args.specs_root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps_payload(obj), encoding="utf-8")
        table = allocation_table(obj["rows"])
        for task, cell in table.items():
            print(f"GATE_SET_ALLOC {task} " + " ".join(f"{t}={n}" for t, n in sorted(cell.items())), flush=True)
        print(f"GATE_SET_BUILT path={args.out} total={len(obj['rows'])} sha={obj['sha256'][:12]}", flush=True)
        return 0
    if args.cmd == "check":
        ok, line = check(args.path, args.specs_root)
        print(line, flush=True)
        return 0 if ok else 1
    if args.cmd == "build-xhard0":
        obj = build_xhard0_payload(args.source)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps_payload(obj), encoding="utf-8")
        print(f"GATE_SET_XHARD0_BUILT path={args.out} total={len(obj['rows'])} sha={obj['sha256'][:12]} "
              f"source_sha={obj['source']['sha256'][:12]}", flush=True)
        return 0
    if args.cmd == "check-xhard0":
        ok, line = check_xhard0(args.path, args.source)
        print(line, flush=True)
        return 0 if ok else 1
    if args.which == "v9":
        rows = load_gate_set(args.path or DEFAULT_PATH)
    else:
        rows = load_xhard0_set(args.path or XHARD0_DEFAULT_PATH)
    if args.kind == "generate":
        out = write_generate_identities(rows, args.out)
    elif args.kind == "env-digest":
        out = write_env_digest_identities(rows, args.out)
    else:
        out = write_legacy_identities(rows, args.out)
    print(f"GATE_SET_EXPORT set={args.which} kind={args.kind} out={out} n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
