#!/usr/bin/env python3
"""生成噪声的比较器与逐局回归闸门（1003 噪声基线计划第一部分第二、三节；1003 代码测试维护计划第二部分
「细则 2.5 噪声工具瘦身」「对拍细则 3.2～3.4」「G 块」）。

纯 CPU；只用标准库与 h5py。噪声工具只留生成这一条线（用户口径：以后只测生成噪声），评估抽取、reset 摘要、
统计线与冻结／核验子命令已删（细则 2.5；``run-fresh`` 只认评估抽取格式，随 ``eval-extract`` 一并删——
``noise_run_gl.sh`` 的 ``RUN_FRESH`` 行由 ``noise_run.py::check_out_root`` 打出，与本文件无关）。

子命令（仓库根运行；每个都打印具名判定行）::

    noise_gate.py gen-compare --ref <侧> --new <侧> --identities <清单> --out <jsonl>
    noise_gate.py gen-regress build-ref --runs v9=<a 遍根>,<b 遍根> --runs xhard0=<a 遍根>,<b 遍根> \\
        --identities-v9 <冻结清单> --identities-xhard0 <冻结清单> --records <比对记录目录> --out <参照 json>
    noise_gate.py gen-regress check --ref <参照 json> --set {v9,xhard0} --new <新跑根> --out <jsonl> \\
        [--rerun-identities-out <jsonl>] [--rerun-new <改后第二次根> --rerun-old <旧代码根>] [--local-ref-root <基线 gen 根>]
    noise_gate.py selftest

「侧」（``gen-compare`` 的 ``--ref``／``--new``）可以是：
- 交付清单 json（``schema`` 以 ``v8-delivery`` 开头，读 ``rows`` 的 ``h5``／``h5_sha256``，交付行一律视为成功）；
- 生成输出根目录（``hard_parity.py generate`` 的 ``identities.jsonl``，``path`` 相对该目录；xhard0 的 H 侧与官方 O 侧
  bucket 目录同格式）；
- 只有 ``results.jsonl`` 的 ``generate_h5.py`` 输出根（行 ``{"kind": "result", "record": {...}}``）。
身份按 (task, seed) 对齐；``tier`` 为 ``xhard0`` 的身份接受侧行写成官方难度名 ``hard``。

逐帧结构的具体口径（对计划 3.1「逐帧键集合、dtype、shape 全同」的细化，主会话裁定）：真实 V9 二次生成里
``action/waypoint_action`` 每帧都在，但无待执行路点时录像器写 float32 的 NaN 占位、有路点时写 float64 实值，两种签名落在
哪些帧随轨迹而变（V9 新生成 7 个分叉局里有 3 局在分叉后出现这种差异）。只有显式白名单 ``POLYMORPHIC_KEYS`` 里的字段
放行这种差异：每帧签名须属于登记的允许集合，分叉后两侧签名不同按内容差异计（第 0 帧仍须全同）；其余字段两侧必须每帧
签名恒定且相同，某一帧缺字段即结构不同。

``gen-regress``（对拍细则 3.2～3.4）：

- ``build-ref`` 从本机噪声基线四遍（每个集合 a、b 两遍）重算每局 h5 的 sha256，与各遍 ``identities.jsonl`` 记录、git 里的
  ``gen-compare`` 比对记录交叉核对一致后，把每局归入 ``stable``（两遍都成功且 sha 相同）、``known_fail``（两遍都失败且失败
  占位 h5 sha 相同）、``jitter``（其余）三类，写参照文件（顶层 ``sha256`` 为剔除自身后 canonical JSON 的 sha256）。
  末行 ``NOISE_REF=PASS|FAIL``；FAIL 时不写文件。``--limit N`` 只取每个集合前 N 个身份作冒烟、``--trust-recorded-sha``
  不重算 h5 sha，两者的产物都标 ``partial``，
  ``check`` 与 ``hard_parity.py generate --expect-ref`` 拒收 partial 参照。
- ``check`` 默认只读新跑各遍 ``identities.jsonl`` 里 ``Mover`` 边生成边写的 sha 与 ``verdict``（不重读 h5），逐局对参照期望：
  ``match``／``jitter_info``／``flip``；翻转先记 ``flip_pending``，给 ``--local-ref-root`` 时对已回传的翻转局用
  ``compare_h5`` 对照基线 a 遍 h5，细分为 ``structural``／``diverge``／``gen_fail``。首跑有翻转（≤ 10）即
  ``GEN_REGRESS=NEED_RERUN``，并按细则 3.4 写重跑清单（不足 4 局用陪跑局补足、行内 ``filler=true``）；给
  ``--rerun-new``／``--rerun-old`` 时按细则 3.2 四格定性（噪声／回归／环境变了／每次都不同），打印最终判定行，
  另写逐局报告 ``<out>.episodes.md``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "src", REPO):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

GEN_SCHEMA = "noise-gen-compare/1"

#: 生成五类（计划第一部分 3.1），互斥且完备
GEN_CLASSES = ("byte_equal", "diverge", "gen_fail", "structural", "unknown")
#: 求解器走完任务列表仍未成功的记录类型（计入「生成失败」）；其余失败类型一律「原因不明」
SOLVER_FAIL_TYPES = ("DatasetGenerationError", "PlannerExhausted")
#: 多态字段白名单：{h5 时间步组内的数据集路径: 允许的签名集合 {(dtype, shape)}}。只有这里登记的字段允许各帧签名不同。
#: action/waypoint_action：录像器（robomme_hard.env_record_wrapper.RecordWrapper）无待执行路点时写 float32 的 NaN 占位、
#: 有路点时写 float64 实值，分叉后各帧占位状态随轨迹变化。来源：2026-10-03 只读扫描 V9 交付 h5 与 V9 二次生成
#: （artifacts/newtask-v9/parity/h2-nfs）共 23 个非空 h5（12 局两侧，去掉 1 个空文件）的全部时间步：多签名字段只有这一个，
#: 签名恰为 float32 (7,)（276 帧，全部 NaN）与 float64 (7,)（11728 帧）。主会话裁定以显式白名单替代自动识别。
POLYMORPHIC_KEYS: dict[str, frozenset] = {
    "action/waypoint_action": frozenset({("float32", (7,)), ("float64", (7,))}),
}


class GateError(RuntimeError):
    """输入不合契约、参照文件不自洽等，拒跑。"""


# ── 通用小工具 ──────────────────────────────────────────────────────────────


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def payload_sha256(obj: dict[str, Any]) -> str:
    """剔掉顶层 ``sha256`` 键后 canonical JSON 的 sha256（与 ``gate_set.payload_sha256`` 同口径）。"""
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        if text.strip():
            out.append(json.loads(text))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


def line_count(path: str | Path) -> int:
    return sum(1 for t in Path(path).read_text(encoding="utf-8").splitlines() if t.strip())


def id_str(task: str, tier: str, seed: int) -> str:
    return f"{task}|{tier}|{int(seed)}"


def tier_compatible(want: str, got: Any) -> bool:
    """身份档位与侧行档位是否相容：xhard0 接受官方难度名 hard；未写档位的侧行不判冲突。"""
    if got is None:
        return True
    got = str(got)
    return got == want or (want == "xhard0" and got == "hard")


# ── 身份清单 ────────────────────────────────────────────────────────────────


def load_identities(path: str | Path) -> list[dict[str, Any]]:
    """读 G9（``gate_set`` 冻结 json）、D0（``identities-full192.json`` 列表，缺 tier 记 xhard0）或 jsonl 身份清单。
    统一为 ``{id, task, tier, seed, spec_sha256?, source_episode?, candidate?}``；(task, seed) 重复即拒。"""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = read_jsonl(path)
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and str(obj.get("schema", "")).startswith("gate-set"):
            from scripts.parity import gate_set

            raw = gate_set.load_gate_set(path)
        elif isinstance(obj, dict) and isinstance(obj.get("rows"), list):
            raw = obj["rows"]
        elif isinstance(obj, list):
            raw = obj
        else:
            raise GateError(f"{path}：认不出身份清单格式")
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in raw:
        task, seed = str(row["task"]), int(row["seed"])
        tier = str(row.get("tier") or row.get("difficulty") or "xhard0")
        if (task, seed) in seen:
            raise GateError(f"{path}：身份重复 {task} seed={seed}")
        seen.add((task, seed))
        item = {"id": id_str(task, tier, seed), "task": task, "tier": tier, "seed": seed}
        for key in ("spec_sha256", "source_episode", "candidate", "builder_episode"):
            if row.get(key) is not None:
                item[key] = row[key]
        out.append(item)
    return out


# ── 生成：侧加载器 ──────────────────────────────────────────────────────────


def _side_entry(h5: Any, sha: Any, ok: bool, error_type: Any = None, error: Any = None, tier: Any = None,
                **extra: Any) -> dict[str, Any]:
    return {"h5": str(h5) if h5 else None, "sha256": sha, "status": "ok" if ok else "fail",
            "error_type": error_type, "error": error, "tier": tier, **extra}


def load_side(spec: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """把交付清单／生成输出根／O-bucket 目录统一映射为 ``(task, seed) -> {h5, sha256, status, error_type, error, tier}``。
    同一身份出现多行（续跑重复）时记 ``duplicate=True``，比较时归「原因不明」。"""
    spec = Path(spec)
    out: dict[tuple[str, int], dict[str, Any]] = {}

    def put(key: tuple[str, int], entry: dict[str, Any]) -> None:
        if key in out:
            entry["duplicate"] = True
        out[key] = entry

    if spec.is_file():
        obj = json.loads(spec.read_text(encoding="utf-8"))
        if not (isinstance(obj, dict) and str(obj.get("schema", "")).startswith("v8-delivery")):
            raise GateError(f"{spec}：文件侧只认 v8-delivery 交付清单")
        for row in obj.get("rows", []):
            h5 = row.get("h5")
            if h5 and not Path(h5).is_absolute():
                h5 = str((spec.parent / h5).resolve())
            put((str(row["task"]), int(row["seed"])),
                _side_entry(h5, row.get("h5_sha256"), True, tier=row.get("tier"), source="delivery"))
        return out
    if not spec.is_dir():
        raise GateError(f"侧不存在：{spec}")
    identities = spec / "identities.jsonl"
    results = spec / "results.jsonl"
    if identities.is_file():
        for line in read_jsonl(identities):
            rel = line.get("path")
            ok = bool(line.get("success")) and not line.get("h5_error")
            error = line.get("h5_error") or line.get("error")
            put((str(line["task"]), int(line["seed"])),
                _side_entry(spec / rel if rel else None, line.get("sha256"), ok, line.get("error_type"), error,
                            tier=line.get("tier"), source="identities"))
        return out
    if results.is_file():
        for line in read_jsonl(results):
            rec = line.get("record", line)
            if line.get("kind") not in (None, "result"):
                continue
            h5 = rec.get("h5")
            put((str(rec["task"]), int(rec["seed"])),
                _side_entry(h5 if rec.get("ok") else None, rec.get("h5_sha256"), bool(rec.get("ok")),
                            rec.get("error_type"), rec.get("error"), tier=rec.get("tier"), source="results"))
        # results.jsonl 里同一身份的多轮记录（基础设施重试）按最后一行算，不当重复
        for entry in out.values():
            entry.pop("duplicate", None)
        return out
    raise GateError(f"{spec}：既无 identities.jsonl 也无 results.jsonl")


# ── 生成：h5 逐帧比较器 ─────────────────────────────────────────────────────


def _is_ts(name: str) -> bool:
    return name.startswith("timestep_") and name[len("timestep_"):].isdigit()


def _attrs(obj) -> dict[str, Any]:
    out = {}
    for k in sorted(obj.attrs.keys()):
        v = obj.attrs[k]
        out[k] = v.tolist() if hasattr(v, "tolist") else v
    return out


def _value_key(value: Any) -> Any:
    """数据集值的可比形态：数值数组按字节，字符串／对象按 tolist。"""
    import numpy as np

    arr = np.asarray(value)
    if arr.dtype.kind in "OSUV":
        return ("obj", repr(arr.tolist()))
    return ("bin", arr.dtype.str, arr.shape, arr.tobytes())


def _tree(group) -> tuple[dict[str, tuple[str, tuple]], dict[str, Any]]:
    """一个组下全部数据集 {相对路径: (dtype, shape)} 与全部组／数据集属性 {路径: attrs}。"""
    import h5py

    datasets: dict[str, tuple[str, tuple]] = {}
    attrs: dict[str, Any] = {"": _attrs(group)}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = (str(obj.dtype), tuple(obj.shape))
        a = _attrs(obj)
        if a:
            attrs[name] = a
    group.visititems(visit)
    return datasets, attrs


def _frames(episode) -> dict[int, str]:
    return {int(k[len("timestep_"):]): k for k in episode.keys() if _is_ts(k)}


def _max_abs(a: Any, b: Any) -> float:
    import numpy as np

    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or x.dtype.kind not in "biuf" or y.dtype.kind not in "biuf" or x.size == 0:
        return 0.0
    return float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))


#: 参考量（只写报告、不参与判定）
REF_ACTION = ("action/joint_action",)
REF_STATE = ("obs/joint_state", "obs/gripper_state")
REF_IMAGE = ("obs/front_rgb", "obs/wrist_rgb")


def compare_h5(left: str | Path, right: str | Path) -> dict[str, Any]:
    """两份成功生成的 h5 内容比较（字节已知不同）。返回 ``{verdict, reason, first_divergence, ...参考量}``，
    verdict ∈ {diverge, structural, unknown}。规则见计划第一部分 3.1。"""
    import h5py

    out: dict[str, Any] = {"first_divergence": None, "action_max": 0.0, "state_max": 0.0, "image_max": 0.0,
                           "frames_ref": None, "frames_new": None, "frames_diff": None}
    try:
        with h5py.File(left, "r") as lf, h5py.File(right, "r") as rf:
            if _attrs(lf) != _attrs(rf):
                return {**out, "verdict": "structural", "reason": "file_attrs"}
            if sorted(lf.keys()) != sorted(rf.keys()):
                return {**out, "verdict": "structural", "reason": "top_keys",
                        "detail": [sorted(lf.keys()), sorted(rf.keys())]}
            tops = sorted(lf.keys())
            first: int | None = None
            for top in tops:
                le, re_ = lf[top], rf[top]
                if not isinstance(le, h5py.Group) or not isinstance(re_, h5py.Group):
                    if _value_key(le[()]) != _value_key(re_[()]):
                        return {**out, "verdict": "structural", "reason": f"top_dataset:{top}"}
                    continue
                # 非时间步内容：键集合、dtype、shape、值、属性全同
                if _attrs(le) != _attrs(re_):
                    return {**out, "verdict": "structural", "reason": f"{top}:attrs"}
                lnon = sorted(k for k in le.keys() if not _is_ts(k))
                rnon = sorted(k for k in re_.keys() if not _is_ts(k))
                if lnon != rnon:
                    return {**out, "verdict": "structural", "reason": f"{top}:non_timestep_keys",
                            "detail": [lnon, rnon]}
                for name in lnon:
                    lo, ro = le[name], re_[name]
                    if isinstance(lo, h5py.Group) != isinstance(ro, h5py.Group):
                        return {**out, "verdict": "structural", "reason": f"{top}/{name}:kind"}
                    if isinstance(lo, h5py.Group):
                        ld, la = _tree(lo)
                        rd, ra = _tree(ro)
                        if ld != rd or la != ra:
                            return {**out, "verdict": "structural", "reason": f"{top}/{name}:schema"}
                        for ds in ld:
                            if _value_key(lo[ds][()]) != _value_key(ro[ds][()]):
                                return {**out, "verdict": "structural", "reason": f"{top}/{name}/{ds}:value"}
                    else:
                        if (lo.dtype, lo.shape) != (ro.dtype, ro.shape) or _attrs(lo) != _attrs(ro) \
                                or _value_key(lo[()]) != _value_key(ro[()]):
                            return {**out, "verdict": "structural", "reason": f"{top}/{name}:value"}
                # 时间步：逐帧对应（不做帧号归一）
                lfr, rfr = _frames(le), _frames(re_)
                nl, nr = len(lfr), len(rfr)
                out.update(frames_ref=nl, frames_new=nr, frames_diff=abs(nl - nr))
                if sorted(lfr) != list(range(nl)) or sorted(rfr) != list(range(nr)):
                    return {**out, "verdict": "structural", "reason": f"{top}:frames_not_contiguous"}
                if nl == 0 or nr == 0:
                    return {**out, "verdict": "structural", "reason": f"{top}:no_frame0"}
                # 逐帧结构（第一遍只读元数据）。每个数据集在每帧的「签名」= 不存在，或 (dtype, shape)。
                # 白名单多态字段（POLYMORPHIC_KEYS）：两侧每一帧的签名都必须属于登记的允许集合，集合外（含不存在）即结构
                # 不同；分叉后两侧同帧签名不同按内容差异计（第 0 帧仍须全同）。非白名单字段两侧都必须每帧签名恒定且相同——
                # 只在一侧随帧变化（如某一帧缺字段）、两侧恒定签名不同、或任一侧出现过另一侧从未出现的字段，一律结构不同。
                lsch = [_tree(le[lfr[i]]) for i in range(nl)]
                rsch = [_tree(re_[rfr[i]]) for i in range(nr)]
                lall = set().union(*(set(d) for d, _a in lsch))
                rall = set().union(*(set(d) for d, _a in rsch))
                if lall != rall:
                    return {**out, "verdict": "structural", "reason": f"{top}:dataset_universe",
                            "detail": sorted(lall ^ rall)[:10]}
                variable: set[str] = set()
                for key in sorted(lall):
                    lsig = [d.get(key) for d, _a in lsch]
                    rsig = [d.get(key) for d, _a in rsch]
                    if key in POLYMORPHIC_KEYS:
                        allowed = POLYMORPHIC_KEYS[key]
                        for side, sig in (("ref", lsig), ("new", rsig)):
                            bad = next((i for i, x in enumerate(sig) if x not in allowed), None)
                            if bad is not None:
                                return {**out, "verdict": "structural",
                                        "reason": f"{top}:frame{bad}:polymorphic_signature_{side}",
                                        "detail": [key, str(sig[bad])]}
                        variable.add(key)
                        continue
                    lvar, rvar = len(set(lsig)) > 1, len(set(rsig)) > 1
                    if lvar or rvar:
                        side, sig, other = ("ref", lsig, rsig[0]) if lvar else ("new", rsig, lsig[0])
                        frame = next(i for i, x in enumerate(sig) if x != other)
                        return {**out, "verdict": "structural", "reason": f"{top}:frame{frame}:schema_{side}",
                                "detail": [key, str(sig[frame]), str(other)]}
                    if lsig[0] != rsig[0]:
                        return {**out, "verdict": "structural", "reason": f"{top}:signature",
                                "detail": [key, str(lsig[0]), str(rsig[0])]}
                out["variable_keys"] = sorted(variable)
                common = min(nl, nr)
                local_first: int | None = None
                out["variable_signature_frames"] = 0
                for i in range(common):
                    lg, rg = le[lfr[i]], re_[rfr[i]]
                    ld, la = lsch[i]
                    rd, ra = rsch[i]
                    shared = {k for k in set(ld) & set(rd) if ld[k] == rd[k]}
                    sig_diff = sorted(k for k in set(ld) | set(rd) if ld.get(k) != rd.get(k))
                    if sig_diff:
                        if i == 0:
                            return {**out, "verdict": "structural", "reason": f"{top}:frame0:schema",
                                    "detail": sig_diff[:10]}
                        out["variable_signature_frames"] += 1  # 只可能是白名单多态字段（其余字段上面已判恒定且相同）
                    if local_first is None:
                        same = not sig_diff and la == ra and all(
                            _value_key(lg[ds][()]) == _value_key(rg[ds][()]) for ds in ld)
                        if not same:
                            if i == 0:
                                return {**out, "verdict": "structural", "reason": f"{top}:frame0_content"}
                            local_first = i
                    # 参考量：只算公共帧
                    for ds in REF_ACTION:
                        if ds in shared:
                            out["action_max"] = max(out["action_max"], _max_abs(lg[ds][()], rg[ds][()]))
                    for ds in REF_STATE:
                        if ds in shared:
                            out["state_max"] = max(out["state_max"], _max_abs(lg[ds][()], rg[ds][()]))
                    if local_first is not None:
                        for ds in REF_IMAGE:
                            if ds in shared:
                                out["image_max"] = max(out["image_max"], _max_abs(lg[ds][()], rg[ds][()]))
                if local_first is None and nl != nr:
                    local_first = common  # 公共帧全同、只差帧数：分叉在第一个多出来的帧
                if local_first is not None:
                    first = local_first if first is None else min(first, local_first)
            if first is None:
                return {**out, "verdict": "unknown", "reason": "content_equal_bytes_differ"}
            out["first_divergence"] = first
            return {**out, "verdict": "diverge", "reason": None}
    except Exception as exc:  # noqa: BLE001 打不开或读错一律原因不明
        return {**out, "verdict": "unknown", "reason": f"open_error:{type(exc).__name__}: {exc}"[:300]}


def h5_problem(path: str | Path) -> str | None:
    """一份「成功」h5 是否合法：零字节、打不开、没有任何顶层键都返回原因，合法返回 None。"""
    import h5py

    path = Path(path)
    if path.stat().st_size == 0:
        return "empty"
    try:
        with h5py.File(path, "r") as handle:
            if not list(handle.keys()):
                return "no_keys"
    except Exception as exc:  # noqa: BLE001 打不开一律原因不明
        return f"open_error:{type(exc).__name__}"
    return None


def classify_pair(ident: dict[str, Any], ref: dict[str, Any] | None, new: dict[str, Any] | None,
                  *, rehash: bool = True) -> dict[str, Any]:
    """一局归入互斥五类之一（计划第一部分 3.1），返回逐局报告行。"""
    row: dict[str, Any] = {"kind": "pair", "id": ident["id"], "task": ident["task"], "tier": ident["tier"],
                           "seed": ident["seed"], "class": None, "reason": None, "fail_side": None,
                           "first_divergence": None, "action_max": None, "state_max": None, "image_max": None,
                           "frames_ref": None, "frames_new": None, "frames_diff": None,
                           "ref_h5": ref.get("h5") if ref else None, "new_h5": new.get("h5") if new else None,
                           "ref_sha256": None, "new_sha256": None}

    def done(cls: str, reason: str | None = None, **kw: Any) -> dict[str, Any]:
        row.update(kw)
        row["class"], row["reason"] = cls, reason
        return row

    if ref is None or new is None:
        return done("unknown", f"missing_{'ref' if ref is None else 'new'}")
    for side, entry in (("ref", ref), ("new", new)):
        if entry.get("duplicate"):
            return done("unknown", f"duplicate_{side}")
        if not tier_compatible(ident["tier"], entry.get("tier")):
            return done("unknown", f"tier_mismatch_{side}:{entry.get('tier')}")
    fails = {side for side, entry in (("ref", ref), ("new", new)) if entry["status"] != "ok"}
    if fails:
        bad = [side for side in sorted(fails)
               if (ref if side == "ref" else new).get("error_type") not in SOLVER_FAIL_TYPES]
        if bad:
            entry = ref if bad[0] == "ref" else new
            return done("unknown", f"infra_{bad[0]}:{entry.get('error_type')}:{str(entry.get('error'))[:120]}")
        side = "both" if len(fails) == 2 else next(iter(fails))
        return done("gen_fail", (ref if side == "ref" else new).get("error_type"), fail_side=side)
    for side, entry in (("ref", ref), ("new", new)):
        if not entry.get("h5") or not Path(entry["h5"]).is_file():
            return done("unknown", f"missing_file_{side}")
    # F-5：先判两侧都是合法 h5（非空、能打开、至少一个顶层键），再判字节关系——此前两个零字节文件会判 byte_equal
    for side, entry in (("ref", ref), ("new", new)):
        problem = h5_problem(entry["h5"])
        if problem:
            return done("unknown", f"invalid_h5_{side}:{problem}")
    shas = {}
    for side, entry in (("ref", ref), ("new", new)):
        recorded = entry.get("sha256")
        if rehash or not recorded:
            actual = sha256_file(entry["h5"])
            if recorded and recorded != actual:
                return done("unknown", f"sha_recorded_mismatch_{side}", **{f"{side}_sha256": actual})
            shas[side] = actual
        else:
            shas[side] = recorded
    row["ref_sha256"], row["new_sha256"] = shas["ref"], shas["new"]
    if shas["ref"] == shas["new"]:
        return done("byte_equal")
    got = compare_h5(ref["h5"], new["h5"])
    extra = {k: got.get(k) for k in ("first_divergence", "action_max", "state_max", "image_max", "frames_ref", "variable_signature_frames", "variable_keys",
                                     "frames_new", "frames_diff")}
    if got.get("detail") is not None:
        extra["detail"] = got["detail"]
    return done(got["verdict"], got.get("reason"), **extra)


def gen_compare(ref_spec: str | Path, new_spec: str | Path, identities: list[dict[str, Any]],
                *, rehash: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ref, new = load_side(ref_spec), load_side(new_spec)
    rows = [classify_pair(i, ref.get((i["task"], i["seed"])), new.get((i["task"], i["seed"])), rehash=rehash)
            for i in identities]
    counts = {c: sum(r["class"] == c for r in rows) for c in GEN_CLASSES}
    return rows, counts


def read_gen_pairs(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = read_jsonl(path)
    meta = next((r for r in lines if r.get("kind") == "meta"), None)
    if meta is None or meta.get("schema") != GEN_SCHEMA:
        raise GateError(f"{path}：不是 gen-compare 输出（缺 meta 或 schema 不符）")
    rows = [r for r in lines if r.get("kind") == "pair"]
    for r in rows:
        if r.get("class") not in GEN_CLASSES:
            raise GateError(f"{path}：{r.get('id')} 类别非法 {r.get('class')}")
    if len({r["id"] for r in rows}) != len(rows):
        raise GateError(f"{path}：身份重复")
    return meta, rows


def gen_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {c: sum(r["class"] == c for r in rows) for c in GEN_CLASSES}
    counts["diff"] = counts["diverge"] + counts["gen_fail"]
    counts["n"] = len(rows)
    return counts


# ── gen-regress：逐局期望的参照文件与回归判定（对拍细则 3.2～3.4）────────────


REF_SCHEMA = "noise-ref/1"
REGRESS_SCHEMA = "gen-regress/1"
#: 参照里每局的三类（细则 3.2 表）
REF_CLASSES = ("stable", "known_fail", "jitter")
#: 噪声基线代码锚点与异地副本（细则 3.0）
REF_ANCHOR = "f8f76fba"
REF_HF_BUCKET = "HongzeFu/robomme-hard-v9-noise-baseline"
#: 集合名（与 hard_parity generate 的身份口径一致）
REF_SETS = ("v9", "xhard0")
#: Mover 边生成边写的逐局判定
VERDICTS = ("match", "jitter_info", "flip")
#: 确认为噪声的翻转上限（细则 3.3 第 2 条）
NOISE_MAX = {"v9": 2, "xhard0": 1}
#: 翻转局超过此数不进第二次跑，直接交用户（细则 3.4 预算）
FLIP_RERUN_MAX = 10
#: 第二次跑每席并发的 worker 数；翻转局不足时用陪跑局补足到此数（细则 3.4）
RERUN_MIN = 4
#: 跑法前提（细则 3.3 第 4 条）
PRECOND_WORKERS = 4
PRECOND_GPU = "A40"
#: 翻转局四格定性（细则 3.2 表）
FINAL_CLASSES = ("noise", "regression", "env_changed", "unstable")
FINAL_NAMES = {"noise": "噪声", "regression": "回归", "env_changed": "环境变了", "unstable": "每次都不同"}
LAUNCH_SCHEMA = "hard-parity-launch/1"


def _line_ok(line: dict[str, Any]) -> bool:
    """一行 identities 是否算「成功」：生成成功且 h5 能打开（Mover 打不开时写 ``h5_error``）。"""
    return bool(line.get("success")) and not line.get("h5_error")


def _short(sha: Any) -> str:
    return str(sha)[:12] if sha else "-"


def read_run(root: str | Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[tuple[str, int]]]:
    """读一遍的 ``identities.jsonl``：返回 ``(task, seed) -> 行`` 与重复身份列表（重复的保留最后一行）。"""
    path = Path(root) / "identities.jsonl"
    if not path.is_file():
        raise GateError(f"{root}：缺 identities.jsonl")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    dup: list[tuple[str, int]] = []
    for line in read_jsonl(path):
        key = (str(line["task"]), int(line["seed"]))
        if key in out:
            dup.append(key)
        out[key] = line
    return out, dup


def run_launch(root: str | Path) -> dict[str, Any] | None:
    """一遍的 ``hard_parity generate`` 启动记录（``launch-*.json`` 里 schema 为 ``hard-parity-launch/1`` 的最新一份）。"""
    best = None
    for path in sorted(Path(root).glob("launch-*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("schema") == LAUNCH_SCHEMA:
            best = doc
    return best


def _launch_facts(doc: dict[str, Any] | None) -> dict[str, Any]:
    keys = ("gpu_model", "driver", "workers", "host", "slurm_job", "src_commit", "side", "tier", "rows")
    return {k: (doc or {}).get(k) for k in keys}


def _rel_hint(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


JITTER_OBSERVED_KEYS = ("set", "task", "tier", "seed", "shas_seen", "reason", "decision")


def _observed_jitter(obj: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """``jitter_observed``：经用户确认、在基线之后才观察到的环境敏感局（对拍细则 3.3「噪声」格与环境变了的裁决）。
    逐条核对字段齐全、指向参照里同集合同档的已有一局、不重复；返回 ``(task, seed) -> 登记条目``。"""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for item in obj.get("jitter_observed") or []:
        if not isinstance(item, dict) or any(k not in item for k in JITTER_OBSERVED_KEYS):
            raise GateError(f"jitter_observed 条目缺字段（须含 {JITTER_OBSERVED_KEYS}）：{item}")
        key = (item["task"], int(item["seed"]))
        if key in out:
            raise GateError(f"jitter_observed 身份重复：{key}")
        block = (obj.get("sets") or {}).get(item["set"])
        hit = [e for e in (block or {}).get("episodes", []) if (e["task"], int(e["seed"])) == key]
        if len(hit) != 1 or hit[0].get("tier") != item["tier"]:
            raise GateError(f"jitter_observed 条目 {key} 在集合 {item['set']} 里找不到同档的局")
        if not isinstance(item["shas_seen"], list) or not item["shas_seen"]:
            raise GateError(f"jitter_observed 条目 {key} 的 shas_seen 须为非空列表")
        out[key] = item
    return out


def ref_index(obj: dict[str, Any], set_name: str | None = None) -> dict[tuple[str, int], dict[str, Any]]:
    """参照文件的逐局索引 ``(task, seed) -> 局条目``；不给 ``set_name`` 时合并全部集合（Mover 用）。

    ``jitter_observed`` 登记过的局按 ``jitter`` 期望判定（不参与判定，只报告），基线归类保留在
    ``baseline_class``；逐局基线数据本身不改。"""
    observed = _observed_jitter(obj)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    names = [set_name] if set_name else sorted(obj["sets"])
    for name in names:
        for entry in obj["sets"][name]["episodes"]:
            key = (entry["task"], int(entry["seed"]))
            if key in out:
                raise GateError(f"参照里身份重复：{key}")
            out[key] = dict(entry, set=name)
            if key in observed and observed[key]["set"] == name:
                out[key].update(baseline_class=entry["class"], **{"class": "jitter"}, jitter_observed=True,
                                shas_seen=list(observed[key]["shas_seen"]))
    return out


def mark_jitter(obj: dict[str, Any], *, set_name: str, task: str, seed: int, shas_seen: list[str],
                reason: str, decision: str) -> dict[str, Any]:
    """在 ``jitter_observed`` 末尾追加一局并重算顶层 sha256；逐局基线数据（``sets``）一个字节不动。
    该局须在参照里、且尚未登记；返回新参照对象（不写盘）。"""
    if (task, int(seed)) in _observed_jitter(obj):
        raise GateError(f"{task}|{seed} 已登记在 jitter_observed")
    hit = [e for e in obj["sets"].get(set_name, {}).get("episodes", []) if (e["task"], int(e["seed"])) == (task, int(seed))]
    if len(hit) != 1:
        raise GateError(f"参照集合 {set_name} 里没有 {task}|{seed}")
    new = json.loads(json.dumps(obj))
    new["jitter_observed"] = [*(obj.get("jitter_observed") or []),
                              {"set": set_name, "task": task, "tier": hit[0]["tier"], "seed": int(seed),
                               "shas_seen": list(shas_seen), "reason": reason, "decision": decision}]
    new["sha256"] = payload_sha256(new)
    _observed_jitter(new)
    return new


def load_ref(path: str | Path, *, allow_partial: bool = False) -> dict[str, Any]:
    """读参照文件：schema、顶层 sha256（剔除自身后 canonical JSON）、每局类别合法；partial 参照默认拒收。"""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"{path}：参照文件读不了或不是 JSON：{exc}") from exc
    if not isinstance(obj, dict) or obj.get("schema") != REF_SCHEMA:
        raise GateError(f"{path}：schema 应为 {REF_SCHEMA}")
    if obj.get("sha256") != payload_sha256(obj):
        raise GateError(f"{path}：顶层 sha256 与内容不符（文件 {obj.get('sha256')}，重算 {payload_sha256(obj)}）")
    if obj.get("partial") and not allow_partial:
        raise GateError(f"{path}：partial 参照（build-ref --limit 冒烟产物）不得用于判定")
    for name, block in (obj.get("sets") or {}).items():
        for entry in block.get("episodes", []):
            if entry.get("class") not in REF_CLASSES:
                raise GateError(f"{path}：{name} {entry.get('id')} 类别非法 {entry.get('class')}")
    ref_index(obj)  # 身份唯一
    return obj


def expect_verdict(entry: dict[str, Any] | None, ok: bool, sha: Any) -> str:
    """一局新跑结果对参照期望的判定（``Mover`` 与 ``check`` 共用同一函数）：

    - ``stable``：成功且 sha ∈ 基线两遍 sha → ``match``；
    - ``known_fail``：失败且失败占位 h5 的 sha ∈ 基线 sha → ``match``；
    - ``jitter``：不参与判定 → ``jitter_info``；
    - 其余（含参照里没有这一局、稳定局生成失败）→ ``flip``。"""
    if entry is None:
        return "flip"
    cls = entry.get("class")
    if cls == "jitter":
        return "jitter_info"
    shas = set(entry.get("shas") or [])
    if cls == "stable" and ok and sha in shas:
        return "match"
    if cls == "known_fail" and not ok and sha in shas:
        return "match"
    return "flip"


# ── build-ref ──


def _rehash_run(set_name: str, root: Path, ids: list[dict[str, Any]], *, full: bool, rehash: bool,
                reasons: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    """核一遍：每个身份恰一行、档位相容、h5 在且重算 sha 与 identities 记录相同；返回 (task, seed) -> {ok, sha, path}。"""
    lines, dup = read_run(root)
    for key in dup:
        reasons.append(f"{set_name}/{root.name}：身份重复 {key}")
    want = {(i["task"], i["seed"]) for i in ids}
    if full:
        extra = sorted(set(lines) - want)
        if extra:
            reasons.append(f"{set_name}/{root.name}：identities.jsonl 有清单外身份 {len(extra)} 个，如 {extra[:3]}")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for ident in ids:
        key = (ident["task"], ident["seed"])
        line = lines.get(key)
        if line is None:
            reasons.append(f"{set_name}/{root.name}：缺身份 {ident['id']}")
            continue
        if not tier_compatible(ident["tier"], line.get("tier")):
            reasons.append(f"{set_name}/{root.name}：{ident['id']} 档位不符 {line.get('tier')}")
        rel = line.get("path")
        sha = None
        if rel:
            path = root / rel
            if not path.is_file():
                reasons.append(f"{set_name}/{root.name}：{ident['id']} 缺 h5 {rel}")
            else:
                sha = sha256_file(path) if rehash else line.get("sha256")
                if line.get("sha256") and line["sha256"] != sha:
                    reasons.append(f"{set_name}/{root.name}：{ident['id']} 重算 sha {_short(sha)} 与记录 "
                                   f"{_short(line['sha256'])} 不符")
        elif _line_ok(line):
            reasons.append(f"{set_name}/{root.name}：{ident['id']} 成功却无 h5")
        out[key] = {"ok": _line_ok(line), "sha": sha, "path": rel}
    return out


def _classify_ref(a: dict[str, Any], b: dict[str, Any]) -> str:
    if a["ok"] and b["ok"] and a["sha"] and a["sha"] == b["sha"]:
        return "stable"
    if not a["ok"] and not b["ok"] and a["sha"] and a["sha"] == b["sha"]:
        return "known_fail"
    return "jitter"


def _cross_check_records(records: Path, sets: dict[str, dict[str, Any]], reasons: list[str]
                         ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """交叉核对 git 里的 ``gen-compare`` 比对记录：记录里有 ``ref_sha256``／``new_sha256`` 的必须与重算值相同；
    两侧都是同一集合的基线两遍时，``byte_equal`` 必须两遍 sha 相同（反向亦然）；有 a-b 记录覆盖每一局。
    返回 (记录文件清单, 每集合核对行数)。"""
    root_map: dict[str, tuple[str, int]] = {}
    for name, block in sets.items():
        for k, root in enumerate(block["_roots"]):
            root_map[str(root.resolve())] = (name, k)
    files = sorted(records.glob("*.jsonl")) if records.is_dir() else []
    if not files:
        reasons.append(f"比对记录目录无 jsonl：{records}")
    listing: list[dict[str, Any]] = []
    checked: Counter = Counter()
    ab_cover: dict[str, set] = {name: set() for name in sets}
    for path in files:
        meta, rows = read_gen_pairs(path)
        sides = {side: root_map.get(str(Path(meta[side]).resolve())) for side in ("ref", "new")}
        listing.append({"file": path.name, "sha256": sha256_file(path), "rows": len(rows),
                        "ref": Path(meta["ref"]).name, "new": Path(meta["new"]).name})
        mapped = {side: m for side, m in sides.items() if m is not None}
        if not mapped:
            continue
        for row in rows:
            for side, (name, k) in mapped.items():
                ep = sets[name]["_eps"].get((row["task"], int(row["seed"])))
                if ep is None:
                    continue  # 不在本集合（或 --limit 之外）
                checked[name] += 1
                run = ep["_runs"][k]
                rec_sha = row.get(f"{side}_sha256")
                if rec_sha is not None and rec_sha != run["sha"]:
                    reasons.append(f"记录 {path.name} {row['id']} {side}_sha256={_short(rec_sha)} 与重算 "
                                   f"{_short(run['sha'])} 不符")
                if row["class"] in ("byte_equal", "diverge", "structural", "gen_fail"):
                    rec_ok = not (row["class"] == "gen_fail" and row.get("fail_side") in (side, "both"))
                    if rec_ok != run["ok"]:
                        reasons.append(f"记录 {path.name} {row['id']} {side} 侧成败 {rec_ok} 与 identities {run['ok']} 不符")
            if len(mapped) == 2 and mapped["ref"][0] == mapped["new"][0] and mapped["ref"][1] != mapped["new"][1]:
                name = mapped["ref"][0]
                ep = sets[name]["_eps"].get((row["task"], int(row["seed"])))
                if ep is None:
                    continue
                ab_cover[name].add((row["task"], int(row["seed"])))
                same = ep["_runs"][0]["sha"] == ep["_runs"][1]["sha"] and ep["_runs"][0]["sha"] is not None
                both_ok = ep["_runs"][0]["ok"] and ep["_runs"][1]["ok"]
                if row["class"] == "byte_equal" and not same:
                    reasons.append(f"记录 {path.name} {row['id']} 判 byte_equal，但两遍重算 sha 不同")
                if same and both_ok and row["class"] != "byte_equal":
                    reasons.append(f"记录 {path.name} {row['id']} 判 {row['class']}，但两遍重算 sha 相同")
    for name, block in sets.items():
        lost = sorted(set(block["_eps"]) - ab_cover[name])
        if lost:
            reasons.append(f"{name}：{len(lost)} 局没有 a-b 两遍比对记录覆盖，如 {lost[:3]}")
    return listing, dict(checked)


def build_ref(runs: dict[str, tuple[Path, Path]], identities: dict[str, Path], records: Path, *,
              limit: int | None = None, rehash: bool = True) -> tuple[dict[str, Any] | None, bool, str, list[str]]:
    """生成参照对象。返回 (对象或 None, 是否通过, 判定行, 原因)。"""
    reasons: list[str] = []
    if set(runs) != set(identities):
        raise GateError(f"--runs 集合 {sorted(runs)} 与 --identities-* 集合 {sorted(identities)} 不一致")
    sets: dict[str, dict[str, Any]] = {}
    for name in sorted(runs):
        if name not in REF_SETS:
            raise GateError(f"未知集合 {name}（只认 {REF_SETS}）")
        roots = tuple(Path(r) for r in runs[name])
        if len(roots) != 2 or roots[0].resolve() == roots[1].resolve():
            raise GateError(f"{name}：--runs 须给两个不同的遍根")
        ids = load_identities(identities[name])
        full = limit is None
        if not full:
            ids = ids[:limit]
        per_run = [_rehash_run(name, root, ids, full=full, rehash=rehash, reasons=reasons) for root in roots]
        eps: dict[tuple[str, int], dict[str, Any]] = {}
        for ident in ids:
            key = (ident["task"], ident["seed"])
            if key not in per_run[0] or key not in per_run[1]:
                continue
            a, b = per_run[0][key], per_run[1][key]
            shas = list(dict.fromkeys(s for s in (a["sha"], b["sha"]) if s))
            eps[key] = {"id": ident["id"], "task": ident["task"], "tier": ident["tier"], "seed": ident["seed"],
                        "class": _classify_ref(a, b), "shas": shas, "ok_a": a["ok"], "ok_b": b["ok"],
                        "h5_a": f"{roots[0].name}/{a['path']}" if a["path"] else None,
                        "h5_b": f"{roots[1].name}/{b['path']}" if b["path"] else None,
                        "_runs": (a, b)}
        launches = [_launch_facts(run_launch(r)) for r in roots]
        sets[name] = {"_roots": roots, "_eps": eps, "_ids": ids, "_launches": launches,
                      "_identities": Path(identities[name])}
    listing, checked = _cross_check_records(records, sets, reasons)
    counts_txt = []
    obj: dict[str, Any] = {"schema": REF_SCHEMA, "anchor": REF_ANCHOR, "hf_bucket": REF_HF_BUCKET,
                           "partial": limit is not None or not rehash, "limit": limit, "rehash": rehash, "sets": {},
                           "records": {"dir": _rel_hint(records), "files": listing},
                           "jitter_observed": [],
                           "rule": ("stable：两遍都成功且 h5 sha 相同；known_fail：两遍都失败且失败占位 h5 sha 相同；"
                                    "jitter：其余。新跑期望见 noise_gate.expect_verdict")}
    for name, block in sets.items():
        episodes = [{k: v for k, v in block["_eps"][(i["task"], i["seed"])].items() if not k.startswith("_")}
                    for i in block["_ids"] if (i["task"], i["seed"]) in block["_eps"]]
        counts = {c: sum(e["class"] == c for e in episodes) for c in REF_CLASSES}
        counts_txt.append(f"{name}=" + ",".join(f"{c}:{counts[c]}" for c in REF_CLASSES))
        obj["sets"][name] = {
            "runs": [{"name": r.name, "root": _rel_hint(r), **lf} for r, lf in zip(block["_roots"], block["_launches"])],
            "identities": {"file": _rel_hint(block["_identities"]), "sha256": sha256_file(block["_identities"])},
            "n": len(episodes), "counts": counts, "records_checked": checked.get(name, 0), "episodes": episodes}
    obj["sha256"] = payload_sha256(obj)
    ok = not reasons
    line = (f"NOISE_REF={'PASS' if ok else 'FAIL'} sets={len(sets)} " + " ".join(counts_txt)
            + f" records={len(listing)} records_checked={sum(checked.values())} mismatches={len(reasons)}"
            + f" partial={int(obj['partial'])} sha={obj['sha256'][:12]}")
    return (obj if ok else None), ok, line, reasons


# ── check ──


def _precondition(root: Path, label: str, invalid: list[str]) -> dict[str, Any]:
    facts = _launch_facts(run_launch(root))
    if not run_launch(root):
        invalid.append(f"{label}：缺 hard_parity 启动记录 launch-*.json")
        return facts
    if facts["workers"] != PRECOND_WORKERS:
        invalid.append(f"{label}：workers={facts['workers']}（须为 {PRECOND_WORKERS}）")
    if PRECOND_GPU not in str(facts["gpu_model"]):
        invalid.append(f"{label}：GPU={facts['gpu_model']}（须为 {PRECOND_GPU}）")
    return facts


def _observe(line: dict[str, Any] | None, dup: bool) -> dict[str, Any]:
    """把一行 identities 归成观察值 {ok, sha, unknown}；unknown 非空即「原因不明」。"""
    if line is None:
        return {"ok": None, "sha": None, "unknown": "missing", "error_type": None, "verdict_recorded": None}
    obs = {"ok": _line_ok(line), "sha": line.get("sha256"), "unknown": None, "error_type": line.get("error_type"),
           "verdict_recorded": line.get("verdict"), "path": line.get("path")}
    if dup:
        obs["unknown"] = "duplicate"
    elif line.get("success") and line.get("h5_error"):
        obs["unknown"] = f"open_error:{str(line['h5_error'])[:80]}"
    elif line.get("success") and not line.get("sha256"):
        obs["unknown"] = "missing_file"
    elif not line.get("success") and line.get("error_type") not in SOLVER_FAIL_TYPES:
        obs["unknown"] = f"infra:{line.get('error_type')}"
    return obs


def _same(x: dict[str, Any] | None, y: dict[str, Any] | None) -> bool | None:
    if not x or not y or x.get("ok") is None or y.get("ok") is None:
        return None
    return x["ok"] == y["ok"] and x["sha"] == y["sha"]


def _jitter_note(entry: dict[str, Any], obs: dict[str, Any]) -> str:
    if not obs["ok"]:
        return "fail"
    shas = entry.get("shas") or []
    if entry.get("ok_a") and shas and obs["sha"] == shas[0]:
        return "same_as_a"
    if entry.get("ok_b") and obs["sha"] in shas:
        return "same_as_b"
    if obs["sha"] in (entry.get("shas_seen") or []):
        return "same_as_observed"
    return "other_trajectory"


def _subgoal_at(h5: Path, frame: int | None) -> str | None:
    if frame is None:
        return None
    try:
        import h5py

        with h5py.File(h5, "r") as f:
            ep = f[sorted(f.keys())[0]]
            g = ep.get(f"timestep_{frame}")
            if g is None or "info/simple_subgoal" not in g:
                return None
            v = g["info/simple_subgoal"][()]
            return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
    except Exception:  # noqa: BLE001 只作报告
        return None


def _local_detail(entry: dict[str, Any], obs: dict[str, Any], new_root: Path, local_ref_root: Path
                  ) -> dict[str, Any]:
    """本机细分一局翻转：新跑失败 → gen_fail；新跑 h5 已回传 → compare_h5 对照基线 a 遍（a 遍失败时取 b 遍）。"""
    if not obs["ok"]:
        return {"sub": "gen_fail", "sub_reason": obs.get("error_type")}
    rel = obs.get("path")
    new_h5 = new_root / rel if rel else None
    if new_h5 is None or not new_h5.is_file():
        return {"sub": "flip_pending", "sub_reason": "new_h5_not_local"}
    if sha256_file(new_h5) != obs["sha"]:
        return {"sub": "unknown", "sub_reason": "sha_recorded_mismatch_new"}
    ref_rel = entry.get("h5_a") if entry.get("ok_a") else (entry.get("h5_b") if entry.get("ok_b") else None)
    if ref_rel is None:
        return {"sub": "diverge", "sub_reason": "ref_known_fail_now_ok"}
    ref_h5 = local_ref_root / ref_rel
    if not ref_h5.is_file():
        return {"sub": "flip_pending", "sub_reason": "ref_h5_not_local"}
    got = compare_h5(ref_h5, new_h5)
    first = got.get("first_divergence")
    return {"sub": got["verdict"], "sub_reason": got.get("reason"), "first_divergence": first,
            "subgoal": _subgoal_at(ref_h5, first), "frames_ref": got.get("frames_ref"),
            "frames_new": got.get("frames_new")}


def pick_fillers(episodes: list[dict[str, Any]], rows: dict[tuple[str, int], dict[str, Any]],
                 flips: list[tuple[str, int]], need: int) -> list[tuple[str, int]]:
    """陪跑局（细则 3.4）：同集合里首跑已与基线逐字节相同的稳定局，按档位与翻转局档位的距离（同档优先、再相邻档）、
    再按身份顺序挑 ``need`` 个。"""
    if need <= 0:
        return []
    tiers = list(dict.fromkeys(e["tier"] for e in episodes))
    flip_tiers = {tiers.index(rows[k]["tier"]) for k in flips}
    cands = []
    for order, e in enumerate(episodes):
        key = (e["task"], int(e["seed"]))
        r = rows.get(key)
        if key in flips or r is None or e["class"] != "stable" or r.get("verdict") != "match":
            continue
        dist = min(abs(tiers.index(e["tier"]) - t) for t in flip_tiers) if flip_tiers else 0
        cands.append((dist, order, key))
    return [key for _d, _o, key in sorted(cands)[:need]]


def _final_class(entry: dict[str, Any], first: dict[str, Any], new2: dict[str, Any], old: dict[str, Any]) -> str:
    """翻转局四格定性（细则 3.2 表）。"""
    if expect_verdict(entry, bool(new2["ok"]), new2["sha"]) == "match":
        return "noise"
    if _same(new2, first):
        return "regression" if expect_verdict(entry, bool(old["ok"]), old["sha"]) == "match" else "env_changed"
    return "unstable"


def regress_check(ref: dict[str, Any], set_name: str, new_root: str | Path, *,
                  rerun_new: str | Path | None = None, rerun_old: str | Path | None = None,
                  local_ref_root: str | Path | None = None) -> dict[str, Any]:
    """返回 {verdict, line, rows, rerun, counts, invalid, reasons, meta}。"""
    if set_name not in ref["sets"]:
        raise GateError(f"参照里没有集合 {set_name}")
    if (rerun_new is None) != (rerun_old is None):
        raise GateError("--rerun-new 与 --rerun-old 须同时给")
    new_root = Path(new_root)
    block = ref["sets"][set_name]
    index = ref_index(ref, set_name)
    # 逐局期望一律取 ref_index 的条目（含 jitter_observed 登记后的 jitter 期望），不直接读基线归类。
    episodes = [index[(x["task"], int(x["seed"]))] for x in block["episodes"]]
    invalid: list[str] = []
    reasons: list[str] = []
    facts = _precondition(new_root, "首跑", invalid)
    drivers = {r.get("driver") for r in block.get("runs", []) if r.get("driver")}
    driver_note = None
    if facts.get("driver") and drivers and facts["driver"] not in drivers:
        driver_note = f"驱动 {facts['driver']} 与基线 {sorted(drivers)} 不同（只报告）"
    lines, dup = read_run(new_root)
    dupset = set(dup)
    extra = sorted(set(lines) - set(index))
    if extra:
        invalid.append(f"首跑有参照外身份 {len(extra)} 个，如 {extra[:3]}")
    missing = [k for k in index if k not in lines]
    if missing:
        invalid.append(f"首跑缺 {len(missing)} 局结果，如 {missing[:3]}")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    lref = Path(local_ref_root) if local_ref_root else None
    for e in episodes:
        key = (e["task"], int(e["seed"]))
        obs = _observe(lines.get(key), key in dupset)
        row: dict[str, Any] = {"kind": "episode", "id": e["id"], "task": e["task"], "tier": e["tier"],
                               "seed": e["seed"], "ref_class": e["class"], "first": obs, "verdict": None,
                               **({"baseline_class": e["baseline_class"]} if e.get("jitter_observed") else {}),
                               "category": None, "filler": False}
        if obs["unknown"] == "missing":
            row["category"] = "missing"
        elif obs["unknown"]:
            row["category"] = "unknown"
            row["reason"] = obs["unknown"]
        else:
            verdict = expect_verdict(e, bool(obs["ok"]), obs["sha"])
            row["verdict"] = verdict
            late_jitter = bool(e.get("jitter_observed")) and verdict == "jitter_info"
            if obs["verdict_recorded"] is not None and obs["verdict_recorded"] != verdict and not late_jitter:
                row["category"] = "unknown"
                row["reason"] = f"verdict_recorded={obs['verdict_recorded']}!={verdict}"
            elif verdict == "match":
                row["category"] = "match"
            elif verdict == "jitter_info":
                row["category"] = "jitter_info"
                row["jitter_note"] = _jitter_note(e, obs)
            else:
                row["category"] = "flip_pending"
                if lref is not None:
                    detail = _local_detail(e, obs, new_root, lref)
                    row.update({k: v for k, v in detail.items() if k != "sub"})
                    row["category"] = detail["sub"]
        rows[key] = row
    flips = [k for k, r in rows.items() if r["category"] in ("flip_pending", "diverge", "gen_fail")]
    n_struct = sum(r["category"] == "structural" for r in rows.values())
    n_unknown = sum(r["category"] == "unknown" for r in rows.values())
    for k, r in rows.items():
        if r["category"] in ("structural", "unknown"):
            reasons.append(f"{r['id']} {r['category']} {r.get('reason') or r.get('sub_reason')}")
    fillers = pick_fillers(episodes, rows, flips, RERUN_MIN - len(flips)) if 0 < len(flips) <= FLIP_RERUN_MAX else []
    for key in fillers:
        rows[key]["filler"] = True
    rerun = [{"task": rows[k]["task"], "tier": rows[k]["tier"], "seed": rows[k]["seed"],
              "filler": rows[k]["filler"], "ref_class": rows[k]["ref_class"]} for k in [*flips, *fillers]]
    final_counts = dict.fromkeys(FINAL_CLASSES, 0)
    rerun_facts = {}
    if rerun_new is not None:
        rn, ro = Path(rerun_new), Path(rerun_old)
        rerun_facts = {"new": _precondition(rn, "改后第二次", invalid), "old": _precondition(ro, "旧代码", invalid)}
        # 同一节点只能在三份启动记录都写了 host 时核对；缺 host 即无法核对，判 INVALID，不静默放过
        host_of = {"首跑": facts.get("host"), "改后第二次": rerun_facts["new"].get("host"),
                   "旧代码": rerun_facts["old"].get("host")}
        lacking = [label for label, h in host_of.items() if not h]
        if lacking:
            invalid.append(f"missing_host：{'、'.join(lacking)} 启动记录缺 host，无法核对第二次跑与首跑同一节点")
        elif len(set(host_of.values())) != 1:
            invalid.append(f"第二次跑须与首跑同一节点：{sorted(set(host_of.values()))}")
        new2_lines, new2_dup = read_run(rn)
        old_lines, old_dup = read_run(ro)
        for key in [*flips, *fillers]:
            r = rows[key]
            n2 = _observe(new2_lines.get(key), key in set(new2_dup))
            od = _observe(old_lines.get(key), key in set(old_dup))
            r["new2"], r["old"] = n2, od
            if n2["unknown"] == "missing" or od["unknown"] == "missing":
                invalid.append(f"第二次跑缺 {r['id']}")
                continue
            if n2["unknown"] or od["unknown"]:
                if not r["filler"]:
                    r["category"] = "unknown"
                    r["reason"] = f"rerun:{n2['unknown'] or od['unknown']}"
                    reasons.append(f"{r['id']} unknown {r['reason']}")
                continue
            entry = index[key]
            if r["filler"]:
                r["filler_new2_match"] = expect_verdict(entry, bool(n2["ok"]), n2["sha"]) == "match"
                r["filler_old_match"] = expect_verdict(entry, bool(od["ok"]), od["sha"]) == "match"
                continue
            r["final"] = _final_class(entry, r["first"], n2, od)
            final_counts[r["final"]] += 1
        n_unknown = sum(r["category"] == "unknown" for r in rows.values())
    counts = Counter(r["category"] for r in rows.values())
    n_flip = len(flips)
    if invalid:
        verdict = "INVALID"
    elif n_unknown or n_struct:
        verdict = "FAIL"
    elif n_flip == 0:
        verdict = "PASS"
    elif n_flip > FLIP_RERUN_MAX:
        verdict = "FAIL"
        reasons.append(f"翻转 {n_flip} 局超过 {FLIP_RERUN_MAX}，不进第二次跑，交用户")
    elif rerun_new is None:
        verdict = "NEED_RERUN"
    else:
        bad = final_counts["regression"] + final_counts["env_changed"] + final_counts["unstable"]
        over = final_counts["noise"] > NOISE_MAX[set_name]
        if over:
            reasons.append(f"确认为噪声的翻转 {final_counts['noise']} 局超过上限 {NOISE_MAX[set_name]}，逐局明细交用户")
        verdict = "FAIL" if bad or over else "PASS"
    head = f"GEN_REGRESS={verdict}"
    if verdict == "NEED_RERUN":
        head += f" flip={n_flip}"
    line = (f"{head} set={set_name} n={len(episodes)} match={counts.get('match', 0)} "
            f"jitter={counts.get('jitter_info', 0)} flip={n_flip} structural={n_struct} unknown={n_unknown} "
            f"missing={counts.get('missing', 0)} invalid={len(invalid)}")
    if fillers or rerun_new is not None:
        line += f" rerun_rows={len(rerun)} filler={len(fillers)}"
    if rerun_new is not None:
        line += " " + " ".join(f"{c}={final_counts[c]}" for c in FINAL_CLASSES)
        line += f" noise_max={NOISE_MAX[set_name]}"
    meta = {"kind": "meta", "schema": REGRESS_SCHEMA, "set": set_name, "ref_sha256": ref["sha256"],
            "new": str(new_root.resolve()), "rerun_new": str(Path(rerun_new).resolve()) if rerun_new else None,
            "rerun_old": str(Path(rerun_old).resolve()) if rerun_old else None,
            "local_ref_root": str(lref.resolve()) if lref else None, "launch": facts, "rerun_launch": rerun_facts,
            "driver_note": driver_note, "counts": dict(counts), "final_counts": final_counts, "invalid": invalid,
            "reasons": reasons, "line": line}
    return {"verdict": verdict, "line": line, "rows": rows, "rerun": rerun, "meta": meta, "invalid": invalid,
            "reasons": reasons, "driver_note": driver_note}


def _md_obs(obs: dict[str, Any] | None) -> str:
    if not obs or obs.get("ok") is None:
        return "—"
    return f"`{_short(obs['sha'])}` {'成功' if obs['ok'] else '失败'}"


def _md_same(x: Any, y: Any) -> str:
    s = _same(x, y)
    return "—" if s is None else ("同" if s else "异")


def episodes_markdown(result: dict[str, Any], ref_path: str) -> str:
    """逐局报告（细则 3.2「逐局报告」）：只列出问题的局（含陪跑）与已知抖动局。"""
    meta = result["meta"]
    out = [f"# gen-regress 逐局报告：{meta['set']}", "",
           f"- 判定行：`{result['line']}`",
           f"- 参照：`{ref_path}`（sha256 `{meta['ref_sha256'][:12]}`）",
           f"- 首跑：`{meta['new']}`；改后第二次：`{meta['rerun_new'] or '—'}`；旧代码：`{meta['rerun_old'] or '—'}`"]
    if result.get("driver_note"):
        out.append(f"- {result['driver_note']}")
    for item in result["invalid"]:
        out.append(f"- 本遍无效：{item}")
    out += ["", "| 任务 | 档 | seed | 参照类 | 陪跑 | 第一次 | 改后第二次 | 旧代码 | 一=二 | 一=旧 | 二=旧 | 分叉步 | 子目标 | 定性 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    shown = [r for r in result["rows"].values()
             if r["category"] not in ("match", "missing") or r["filler"]]
    for r in shown:
        if r["filler"]:
            verdict = "陪跑（只报告）" + ("" if "filler_new2_match" not in r else
                                       f"：改后{'同' if r['filler_new2_match'] else '异'}基线、"
                                       f"旧码{'同' if r['filler_old_match'] else '异'}基线")
        elif r["category"] == "jitter_info":
            verdict = f"已知抖动（只报告：{r.get('jitter_note')}）"
        elif r.get("final"):
            verdict = FINAL_NAMES[r["final"]]
        else:
            verdict = r["category"] + (f"（{r.get('reason') or r.get('sub_reason')}）"
                                       if r.get("reason") or r.get("sub_reason") else "")
        out.append(f"| {r['task']} | {r['tier']} | {r['seed']} | {r['ref_class']} | {'是' if r['filler'] else ''} | "
                   f"{_md_obs(r['first'])} | {_md_obs(r.get('new2'))} | {_md_obs(r.get('old'))} | "
                   f"{_md_same(r['first'], r.get('new2'))} | {_md_same(r['first'], r.get('old'))} | "
                   f"{_md_same(r.get('new2'), r.get('old'))} | {r.get('first_divergence', '—') if r.get('first_divergence') is not None else '—'} | "
                   f"{r.get('subgoal') or '—'} | {verdict} |")
    if not shown:
        out.append("| （无出问题的局，也无已知抖动局） |||||||||||||| ")
    return "\n".join(out) + "\n"


# ── selftest：内存夹具 ──────────────────────────────────────────────────────


def _fx_h5(path: Path, *, frames: int = 4, seed: int = 1, setup_seed: int | None = None, drop: tuple | None = None,
           diverge_at: int | None = None, frame0_delta: bool = False, waypoint_frames: Iterable[int] = (),
           waypoint_nan_frames: Iterable[int] = (), waypoint_int_frames: Iterable[int] = ()) -> Path:
    """小型 h5：一个 ``episode_<seed>`` 组，``setup`` 组 + ``timestep_<i>`` 组（动作／观测／信息三类数据集）。
    ``drop=(帧号, 数据集)`` 删一个字段；``diverge_at`` 起动作与关节状态加偏移；``frame0_delta`` 改第 0 帧图像；
    ``waypoint_frames`` 里的帧写 float64 实值的 ``action/waypoint_action``，``waypoint_nan_frames`` 里的帧写 float32 的 NaN
    占位（仿录像器：有待执行路点写实值、没有写占位，两种签名的帧随轨迹而变）；``waypoint_int_frames`` 写白名单外的 int64
    签名（反例用）。"""
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ep = f.create_group(f"episode_{seed}")
        st = ep.create_group("setup")
        st["seed"] = np.int64(setup_seed if setup_seed is not None else seed)
        st["difficulty"] = "xhard1"
        st["front_camera_intrinsic"] = np.eye(3, dtype=np.float32)
        for i in range(frames):
            g = ep.create_group(f"timestep_{i}")
            off = 0.5 if diverge_at is not None and i >= diverge_at else 0.0
            g["action/joint_action"] = np.full(8, 0.1 * i + off, dtype=np.float64)
            g["obs/joint_state"] = np.full(7, 0.2 * i + off, dtype=np.float32)
            g["obs/gripper_state"] = np.zeros(2, dtype=np.float32)
            img = np.full((4, 4, 3), i % 255, dtype=np.uint8)
            if frame0_delta and i == 0:
                img[0, 0, 0] = 9
            g["obs/front_rgb"] = img
            g["obs/front_depth"] = np.zeros((4, 4, 1), dtype=np.int16)
            g["info/is_completed"] = np.bool_(i == frames - 1)
            if i in set(waypoint_frames):
                g["action/waypoint_action"] = np.full(7, 0.3, dtype=np.float64)
            elif i in set(waypoint_nan_frames):
                g["action/waypoint_action"] = np.full(7, np.nan, dtype=np.float32)
            elif i in set(waypoint_int_frames):
                g["action/waypoint_action"] = np.zeros(7, dtype=np.int64)
            if drop is not None and drop[0] == i:
                del g[drop[1]]
    return path


def _fx_side(root: Path, entries: dict[tuple[str, int], dict[str, Any]]) -> Path:
    """生成输出根夹具：写 identities.jsonl（``path`` 相对根）。"""
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for (task, seed), e in entries.items():
        line = {"task": task, "tier": e.get("tier", "xhard1"), "seed": seed, "success": e.get("ok", True),
                "error_type": e.get("error_type"), "path": None, "sha256": None}
        if e.get("h5"):
            line["path"] = os.path.relpath(e["h5"], root)  # 可指向根外（反例夹具复用同代码侧的文件）
            line["sha256"] = sha256_file(e["h5"])
        lines.append(line)
    write_jsonl(root / "identities.jsonl", lines)
    return root


def _fx_idents(n: int, tier: str = "xhard1", task: str = "T") -> list[dict[str, Any]]:
    return [{"id": id_str(task, tier, s), "task": task, "tier": tier, "seed": s} for s in range(n)]


def _fx_run(root: Path, items: list[dict[str, Any]], *, workers: int = 4, gpu: str = "NVIDIA A40",
            host: str = "n1", driver: str = "595.71.05") -> Path:
    """gen-regress 用的「一遍」夹具：identities.jsonl（每项 {task, tier, seed, ok, sha?, h5?, error_type?, h5_error?,
    verdict?}）与 hard_parity 启动记录。给 ``h5`` 时 ``path`` 相对根、sha 按文件算。"""
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for it in items:
        line = {"task": it["task"], "tier": it.get("tier", "xhard1"), "seed": it["seed"], "success": it.get("ok", True),
                "error_type": it.get("error_type", None if it.get("ok", True) else "DatasetGenerationError"),
                "path": None, "sha256": it.get("sha")}
        if it.get("h5"):
            line["path"] = os.path.relpath(it["h5"], root)
            line["sha256"] = sha256_file(it["h5"])
        elif it.get("sha"):
            line["path"] = f"episodes/{it['task']}_{it['seed']}/hdf5_files/x.h5"
        for k in ("h5_error", "verdict"):
            if k in it:
                line[k] = it[k]
        lines.append(line)
    write_jsonl(root / "identities.jsonl", lines)
    (root / "launch-1.json").write_text(json.dumps({"schema": LAUNCH_SCHEMA, "workers": workers, "gpu_model": gpu,
                                                    "driver": driver, "host": host}), encoding="utf-8")
    return root


def _fx_placeholder(path: Path) -> Path:
    """失败局的占位文件（仿录像器失败时留下的 800 字节小文件：每次失败字节相同）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"FAILED-PLACEHOLDER\n" * 8)
    return path


def _fx_baseline(tmp: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """六局的小基线：seed 0～3 稳定，seed 4 确定性失败，seed 5 抖动（a 成功、b 失败）；返回 (参照对象, 夹具信息)。"""
    ids = [{"task": "T", "tier": "xhard1" if s < 3 else "xhard2", "seed": s} for s in range(6)]
    ident_path = write_jsonl(tmp / "ids.jsonl", ids)
    runs = {}
    for name in ("a", "b"):
        items = []
        for i in ids:
            s = i["seed"]
            if s == 4 or (s == 5 and name == "b"):
                items.append(dict(i, ok=False, h5=_fx_placeholder(tmp / name / f"e{s}" / "hdf5_files" / "x.h5")))
            else:
                items.append(dict(i, h5=_fx_h5(tmp / name / f"e{s}" / "hdf5_files" / "x.h5", seed=s)))
        runs[name] = _fx_run(tmp / name, items)
    rec = tmp / "records"
    rows, counts = gen_compare(runs["a"], runs["b"], load_identities(ident_path))
    write_jsonl(rec / "ab.jsonl", [{"kind": "meta", "schema": GEN_SCHEMA, "ref": str(runs["a"]), "new": str(runs["b"]),
                                    "n": len(rows), "counts": counts}, *rows])
    obj, ok, line, why = build_ref({"v9": (runs["a"], runs["b"])}, {"v9": ident_path}, rec)
    info = {"ids": ids, "ident_path": ident_path, "runs": runs, "records": rec, "line": line, "ok": ok, "why": why}
    return obj, info


def _fx_new(tmp: Path, name: str, ref: dict[str, Any], overrides: dict[int, dict[str, Any]] | None = None,
            only: Iterable[int] | None = None, **launch: Any) -> Path:
    """按参照造一遍新跑：默认每局与基线 a 遍相同（稳定局成功同 sha，确定性失败局失败同 sha）；``overrides`` 按 seed 改。"""
    items = []
    for e in ref["sets"]["v9"]["episodes"]:
        if only is not None and e["seed"] not in set(only):
            continue
        base = {"task": e["task"], "tier": e["tier"], "seed": e["seed"], "ok": e["ok_a"], "sha": e["shas"][0]}
        if not e["ok_a"]:
            base["h5_error"] = "IndexError: list index out of range"
        base.update((overrides or {}).get(e["seed"], {}))
        items.append(base)
    return _fx_run(tmp / name, items, **launch)


def selftest(verbose: bool = True) -> tuple[bool, str, list[tuple[str, bool]]]:
    """同代码一对须通过（gen-compare 无结构不同；gen-regress 全部 match 判 PASS）；反例须逐个被抓到。"""
    results: list[tuple[str, bool, str]] = []  # (名字, 是否符合预期, 说明)

    def expect(name: str, want_pass: bool, got_pass: bool, note: str = "") -> None:
        results.append((name, want_pass == got_pass, note))
        if verbose:
            print(f"# selftest {name}: 预期={'PASS' if want_pass else 'FAIL'} 实得={'PASS' if got_pass else 'FAIL'} {note}")

    def no_struct(rows: list[dict[str, Any]]) -> bool:
        return not any(r["class"] in ("structural", "unknown") for r in rows)

    with tempfile.TemporaryDirectory(prefix="noise-gate-selftest-") as tmpd:
        tmp = Path(tmpd)
        # ── gen-compare：同代码一对（含全等局、一个正常分叉局）──
        ids = _fx_idents(6)
        ref_e, same_e = {}, {}
        for i in ids:
            s = i["seed"]
            # seed 5 是正常分叉局：第 2 帧起分叉，白名单字段 waypoint_action 的 NaN 占位帧也随之不同（不得判结构不同）
            ref_e[("T", s)] = {"h5": _fx_h5(tmp / "ref" / f"e{s}.h5", seed=s, waypoint_frames=(1, 2) if s == 5 else (),
                                            waypoint_nan_frames=(0, 3) if s == 5 else ())}
            same_e[("T", s)] = {"h5": _fx_h5(tmp / "same" / f"e{s}.h5", seed=s, diverge_at=2 if s == 5 else None,
                                             waypoint_frames=(1, 3) if s == 5 else (),
                                             waypoint_nan_frames=(0, 2) if s == 5 else ())}
        ref_root = _fx_side(tmp / "ref", ref_e)
        same_root = _fx_side(tmp / "same", same_e)
        rows, counts = gen_compare(ref_root, same_root, ids)
        expect("same_code_gen", True, no_struct(rows) and counts["byte_equal"] == 5 and counts["diverge"] == 1,
               str(counts))
        eq_rows = [r for r in rows if r["class"] == "byte_equal"]
        expect("全等局不被分叉步规则拒绝", True, bool(eq_rows) and all(r["first_divergence"] is None for r in eq_rows))
        for name, sub, entry in (
                ("初帧缺一个字段", "bad1", (0, {"seed": 0, "drop": (0, "obs/front_depth")})),
                ("中间帧缺一个字段", "bad2", (1, {"seed": 1, "drop": (2, "obs/front_depth")})),
                ("setup 改一个值", "bad3", (2, {"seed": 2, "setup_seed": 999})),
                ("白名单字段集合外签名", "bad4", (5, {"seed": 5, "diverge_at": 2, "waypoint_frames": (1,),
                                                    "waypoint_nan_frames": (0, 2), "waypoint_int_frames": (3,)}))):
            bad_e = dict(same_e)
            seed, kw = entry
            bad_e[("T", seed)] = {"h5": _fx_h5(tmp / sub / f"e{seed}.h5", **kw)}
            rows_b, _c = gen_compare(ref_root, _fx_side(tmp / sub, bad_e), ids)
            expect(name, False, no_struct(rows_b), str(_c))
        # F-5：两侧都是零字节「成功」文件，不得判 byte_equal
        e0 = {"h5": str(tmp / "empty" / "a.h5"), "sha256": None, "status": "ok", "error_type": None, "error": None,
              "tier": "xhard1"}
        Path(e0["h5"]).parent.mkdir(parents=True, exist_ok=True)
        Path(e0["h5"]).write_bytes(b"")
        r = classify_pair(ids[0], e0, dict(e0))
        expect("两侧空文件（F-5）", False, r["class"] == "byte_equal", f"class={r['class']} reason={r['reason']}")

        # ── gen-regress build-ref ──
        bdir = tmp / "base"
        ref, info = _fx_baseline(bdir)
        cnt = (ref or {}).get("sets", {}).get("v9", {}).get("counts")
        expect("same_code_build_ref", True,
               info["ok"] and cnt == {"stable": 4, "known_fail": 1, "jitter": 1}, info["line"])
        if ref is None:
            for why in info["why"][:5]:
                print(f"# selftest build_ref: {why}")
            return False, "GATE_SELFTEST=FAIL same_code=FAIL cases_fail=0/0 reason=build_ref", \
                [(n, ok) for n, ok, _ in results]
        # 反例：比对记录把一局写成 byte_equal 但两遍 sha 不同（篡改记录里的 new_sha256）
        bad_rec = tmp / "bad-records"
        meta, prow = read_gen_pairs(info["records"] / "ab.jsonl")
        prow = [dict(p) for p in prow]
        prow[0]["new_sha256"] = "0" * 64
        write_jsonl(bad_rec / "ab.jsonl", [meta, *prow])
        _o, ok, line, _w = build_ref({"v9": (info["runs"]["a"], info["runs"]["b"])}, {"v9": info["ident_path"]}, bad_rec)
        expect("比对记录 sha 与重算不符", False, ok, line)
        # 反例：改参照里一局的类别而不重算 sha256
        rp = tmp / "ref-good.json"
        rp.write_text(json.dumps(ref), encoding="utf-8")
        tampered = json.loads(rp.read_text(encoding="utf-8"))
        tampered["sets"]["v9"]["episodes"][0]["class"] = "jitter"
        tp = tmp / "ref-bad.json"
        tp.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            load_ref(tp)
            got = True
        except GateError:
            got = False
        expect("改参照不改 sha256", False, got)
        ref = load_ref(rp)

        # ── gen-regress check ──
        res = regress_check(ref, "v9", _fx_new(tmp, "n-same", ref))
        expect("same_code_regress", True, res["verdict"] == "PASS", res["line"])
        # 抖动局任意结果都只报告
        res = regress_check(ref, "v9", _fx_new(tmp, "n-jit", ref, {5: {"ok": False, "sha": "f" * 64}}))
        expect("same_code_jitter_only_info", True, res["verdict"] == "PASS", res["line"])
        flip = {0: {"sha": "1" * 64}}
        res = regress_check(ref, "v9", _fx_new(tmp, "n-flip", ref, flip))
        fillers = [x for x in res["rerun"] if x["filler"]]
        expect("稳定局翻转须重跑", False, res["verdict"] == "PASS",
               f"{res['line']} rerun={len(res['rerun'])} filler={len(fillers)}")
        expect("same_code_filler_fill_to_4", True, len(res["rerun"]) == 4 and len(fillers) == 3
               and [x["seed"] for x in fillers] == [1, 2, 3])
        # 四格定性
        only = [0, 1, 2, 3]
        cases = {"noise": ({0: {}}, {0: {"sha": "2" * 64}}, "PASS"),
                 "regression": ({0: {"sha": "1" * 64}}, {0: {}}, "FAIL"),
                 "env_changed": ({0: {"sha": "1" * 64}}, {0: {"sha": "3" * 64}}, "FAIL"),
                 "unstable": ({0: {"sha": "4" * 64}}, {0: {}}, "FAIL")}
        for name, (n2, od, want) in cases.items():
            res = regress_check(ref, "v9", tmp / "n-flip",
                                rerun_new=_fx_new(tmp, f"r2-{name}", ref, n2, only=only),
                                rerun_old=_fx_new(tmp, f"ro-{name}", ref, od, only=only))
            row = res["rows"][("T", 0)]
            good = res["verdict"] == want and row.get("final") == name
            label = f"四格定性 {FINAL_NAMES[name]}"
            if want == "PASS":
                expect(f"same_code_{label}", True, good, res["line"])
            else:
                expect(label, False, not good, res["line"])
        # 噪声超上限：三局翻转第二次都回到基线（v9 上限 2）
        many = {0: {"sha": "1" * 64}, 1: {"sha": "1" * 64}, 2: {"sha": "1" * 64}}
        res = regress_check(ref, "v9", _fx_new(tmp, "n-3flip", ref, many),
                            rerun_new=_fx_new(tmp, "r2-3", ref, only=[0, 1, 2, 3]),
                            rerun_old=_fx_new(tmp, "ro-3", ref, only=[0, 1, 2, 3]))
        expect("噪声翻转超上限", False, res["verdict"] == "PASS", res["line"])
        # 确定性失败局变成功、稳定局生成失败，都算翻转
        res = regress_check(ref, "v9", _fx_new(tmp, "n-kf", ref, {4: {"ok": True, "sha": "5" * 64, "h5_error": None}}))
        expect("确定性失败局变成功", False, res["verdict"] == "PASS", res["line"])
        res = regress_check(ref, "v9", _fx_new(tmp, "n-gf", ref, {1: {"ok": False, "sha": ref["sets"]["v9"]["episodes"][4]["shas"][0]}}))
        expect("稳定局生成失败", False, res["verdict"] == "PASS", res["line"])
        # 原因不明：成功却打不开；Mover 写的 verdict 与重算不符
        res = regress_check(ref, "v9", _fx_new(tmp, "n-open", ref, {2: {"h5_error": "OSError: x"}}))
        expect("成功局打不开", False, res["verdict"] != "FAIL", res["line"])
        res = regress_check(ref, "v9", _fx_new(tmp, "n-vd", ref, {3: {"verdict": "flip"}}))
        expect("Mover verdict 与重算不符", False, res["verdict"] != "FAIL", res["line"])
        # 跑法前提：workers 不是 4、缺局
        res = regress_check(ref, "v9", _fx_new(tmp, "n-w1", ref, workers=1))
        expect("workers=1 判本遍无效", False, res["verdict"] != "INVALID", res["line"])
        res = regress_check(ref, "v9", _fx_new(tmp, "n-miss", ref, only=[0, 1, 2, 3, 4]))
        expect("缺局判本遍无效", False, res["verdict"] != "INVALID", res["line"])
        # 本机细分：翻转局 h5 已回传时用 compare_h5 对照基线 a 遍
        lroot = bdir
        h_div = _fx_h5(tmp / "n-loc" / "e0" / "hdf5_files" / "x.h5", seed=0, diverge_at=2)
        h_str = _fx_h5(tmp / "n-loc" / "e1" / "hdf5_files" / "x.h5", seed=1, drop=(1, "obs/front_depth"))
        res = regress_check(ref, "v9", _fx_new(tmp, "n-loc", ref, {0: {"h5": h_div}, 1: {"h5": h_str}}),
                            local_ref_root=lroot)
        r0, r1 = res["rows"][("T", 0)], res["rows"][("T", 1)]
        expect("same_code_local_diverge", True, r0["category"] == "diverge" and r0.get("first_divergence") == 2,
               f"{r0['category']} {r0.get('first_divergence')}")
        expect("本机细分结构不同", False, not (r1["category"] == "structural" and res["verdict"] == "FAIL"), res["line"])

    same_ok = all(ok for name, ok, _ in results if name.startswith("same_code") or name.startswith("全等局"))
    cases_ = [(name, ok) for name, ok, _ in results if not (name.startswith("same_code") or name.startswith("全等局"))]
    passed = same_ok and all(ok for _n, ok in cases_)
    good = sum(ok for _n, ok in cases_)
    line = (f"GATE_SELFTEST={'PASS' if passed else 'FAIL'} same_code={'PASS' if same_ok else 'FAIL'} "
            f"cases_fail={good}/{len(cases_)}")
    return passed, line, [(n, ok) for n, ok, _ in results]


# ── CLI ─────────────────────────────────────────────────────────────────────


def cmd_gen_compare(args) -> int:
    identities = load_identities(args.identities)
    rows, counts = gen_compare(args.ref, args.new, identities, rehash=not args.trust_recorded_sha)
    meta = {"kind": "meta", "schema": GEN_SCHEMA, "ref": str(Path(args.ref).resolve()),
            "new": str(Path(args.new).resolve()), "identities": str(Path(args.identities).resolve()),
            "n": len(rows), "rehash": not args.trust_recorded_sha, "counts": counts}
    write_jsonl(args.out, [meta, *rows])
    for r in rows:
        if r["class"] not in ("byte_equal", "diverge"):
            print(f"# {r['class']} {r['id']} reason={r['reason']} fail_side={r['fail_side']}")
    print(f"GEN_PAIR=INFO ref={Path(args.ref).name} new={Path(args.new).name} n={len(rows)} "
          + " ".join(f"{c}={counts[c]}" for c in GEN_CLASSES), flush=True)
    return 0


def _parse_runs(items: list[str]) -> dict[str, tuple[Path, Path]]:
    out: dict[str, tuple[Path, Path]] = {}
    for item in items:
        name, sep, value = item.partition("=")
        parts = [p for p in value.split(",") if p]
        if not sep or len(parts) != 2:
            raise GateError(f"--runs 须为 <集合>=<a 遍根>,<b 遍根>：{item}")
        if name in out:
            raise GateError(f"--runs 集合重复：{name}")
        out[name] = (Path(parts[0]), Path(parts[1]))
    return out


def cmd_build_ref(args) -> int:
    runs = _parse_runs(args.runs)
    identities = {name: Path(p) for name, p in (("v9", args.identities_v9), ("xhard0", args.identities_xhard0))
                  if p is not None}
    out = Path(args.out)
    if out.exists():
        raise GateError(f"{out} 已存在：参照文件生成后只读（红线 R2），不覆盖")
    obj, ok, line, reasons = build_ref(runs, identities, Path(args.records), limit=args.limit,
                                       rehash=not args.trust_recorded_sha)
    for r in reasons[:50]:
        print(f"# {r}")
    if ok and obj is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for name, block in obj["sets"].items():
            for e in block["episodes"]:
                if e["class"] != "stable":
                    print(f"# {name} {e['class']} {e['id']} ok_a={e['ok_a']} ok_b={e['ok_b']} "
                          f"shas={[s[:12] for s in e['shas']]}")
    print(line, flush=True)
    return 0 if ok else 1


def cmd_check(args) -> int:
    ref = load_ref(args.ref)
    res = regress_check(ref, args.set, args.new, rerun_new=args.rerun_new, rerun_old=args.rerun_old,
                        local_ref_root=args.local_ref_root)
    out = Path(args.out)
    write_jsonl(out, [res["meta"], *res["rows"].values()])
    md = out.with_name(out.name + ".episodes.md") if out.suffix != ".jsonl" else \
        out.with_name(out.name[: -len(".jsonl")] + ".episodes.md")
    md.write_text(episodes_markdown(res, str(args.ref)), encoding="utf-8")
    if res["verdict"] == "NEED_RERUN" and args.rerun_identities_out:
        write_jsonl(args.rerun_identities_out, res["rerun"])
        print(f"# 重跑清单 {args.rerun_identities_out}：{len(res['rerun'])} 行（陪跑 "
              f"{sum(r['filler'] for r in res['rerun'])}）", flush=True)
    for item in res["invalid"]:
        print(f"# 本遍无效：{item}")
    for item in res["reasons"][:30]:
        print(f"# {item}")
    if res.get("driver_note"):
        print(f"# {res['driver_note']}")
    for r in res["rows"].values():
        if r["category"] not in ("match",) or r["filler"]:
            tag = "FILLER" if r["filler"] else r["category"]
            print(f"# {tag} {r['id']} first={_short(r['first'].get('sha'))} "
                  f"final={r.get('final') or '-'} sub={r.get('sub_reason') or r.get('reason') or '-'}")
    print(f"# 逐局报告 {md}", flush=True)
    print(res["line"], flush=True)
    return {"PASS": 0, "NEED_RERUN": 3, "INVALID": 4}.get(res["verdict"], 1)


def cmd_mark_jitter(args) -> int:
    ref_path = Path(args.ref)
    obj = load_ref(ref_path)
    new = mark_jitter(obj, set_name=args.set, task=args.task, seed=args.seed, shas_seen=args.sha,
                      reason=args.reason, decision=args.decision)
    ref_path.write_text(json.dumps(new, sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    load_ref(ref_path)
    sets_same = canonical_json(new["sets"]) == canonical_json(obj["sets"])
    ok = sets_same
    print(f"NOISE_REF_MARK={'PASS' if ok else 'FAIL'} set={args.set} id={args.task}|{args.seed} "
          f"observed={len(new['jitter_observed'])} sets_unchanged={int(sets_same)} "
          f"sha={obj['sha256'][:12]}->{new['sha256'][:12]}", flush=True)
    return 0 if ok else 1


def cmd_selftest(args) -> int:
    ok, line, _cases = selftest(verbose=True)
    print(line, flush=True)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="生成噪声比较器与逐局回归闸门")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gen-compare", help="两侧生成 h5 逐局五类归类")
    p.add_argument("--ref", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--identities", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trust-recorded-sha", action="store_true", help="不重算文件 sha256，直接用侧记录的值（快、弱）")
    p.set_defaults(func=cmd_gen_compare)

    gr = sub.add_parser("gen-regress", help="逐局期望的参照文件与回归判定（对拍细则 3.2～3.4）")
    grs = gr.add_subparsers(dest="regress_cmd", required=True)
    p = grs.add_parser("build-ref", help="从噪声基线四遍重算 sha、交叉核对比对记录、写参照文件（NOISE_REF）")
    p.add_argument("--runs", action="append", required=True, metavar="集合=a 遍根,b 遍根",
                   help="可重复；集合为 v9 或 xhard0")
    p.add_argument("--identities-v9", default=None, help="V9 身份清单（如 scripts/configs/gate-set-v9-129.json）")
    p.add_argument("--identities-xhard0", default=None, help="xhard0 身份清单（如 scripts/configs/gate-set-xhard0-48.json）")
    p.add_argument("--records", required=True, help="gen-compare 比对记录目录（docs/validation/.../records/compare）")
    p.add_argument("--out", required=True, help="参照文件（已存在即拒，不覆盖）")
    p.add_argument("--limit", type=int, default=None, help="冒烟：每个集合只取前 N 个身份，产物标 partial、不得用于判定")
    p.add_argument("--trust-recorded-sha", action="store_true", help="不重算 h5 sha256（只用于冒烟，产物标 partial）")
    p.set_defaults(func=cmd_build_ref)
    p = grs.add_parser("check", help="新跑对参照逐局判定（GEN_REGRESS）")
    p.add_argument("--ref", required=True)
    p.add_argument("--set", choices=REF_SETS, required=True)
    p.add_argument("--new", required=True, help="首跑输出根（含 identities.jsonl 与 launch-*.json）")
    p.add_argument("--out", required=True, help="逐局 jsonl；同目录另写 <out 去掉 .jsonl>.episodes.md")
    p.add_argument("--rerun-identities-out", default=None, help="NEED_RERUN 时写重跑清单（含陪跑局，filler=true）")
    p.add_argument("--rerun-new", default=None, help="改后代码第二次跑的输出根")
    p.add_argument("--rerun-old", default=None, help="旧代码（f8f76fba）同一节点那一次的输出根")
    p.add_argument("--local-ref-root", default=None,
                   help="本机噪声基线 gen 根（artifacts/noise-baseline/gen）：对已回传的翻转局细分类别")
    p.set_defaults(func=cmd_check)
    p = grs.add_parser("mark-jitter",
                       help="把一局登记为环境敏感局（写参照的 jitter_observed，须用户确认；逐局基线数据不动）")
    p.add_argument("--ref", required=True)
    p.add_argument("--set", choices=REF_SETS, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sha", action="append", required=True, help="基线之后观察到的 h5 sha256（可重复）")
    p.add_argument("--reason", required=True, help="登记原因（证据摘要）")
    p.add_argument("--decision", required=True, help="用户裁决原话")
    p.set_defaults(func=cmd_mark_jitter)

    p = sub.add_parser("selftest", help="自检：同代码须通过、反例须逐个被抓到（GATE_SELFTEST）")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except GateError as exc:
        print(f"# 拒跑：{exc}", flush=True)
        name = {"selftest": "GATE_SELFTEST"}.get(args.cmd)
        if args.cmd == "gen-regress":
            name = "NOISE_REF" if args.regress_cmd == "build-ref" else "GEN_REGRESS"
        if name:
            print(f"{name}=FAIL reason=refused", flush=True)
        return 2


if __name__ == "__main__":
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    sys.exit(main())
