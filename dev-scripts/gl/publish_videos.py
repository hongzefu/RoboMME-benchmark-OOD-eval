#!/usr/bin/env python3
"""把账本接受的那一次 attempt 的官方版式视频发布到上游 mme-vla 布局（1006 计划第一部分二「视频布局」、第二部分八.7；
接口冻结说明第六节）。纯 CPU、只用标准库；局目录与 ``official/`` 原位不动，只新增发布目录与索引。

布局（与上游 ``examples/robomme/eval.py`` 的差别只有三处：无 ``ckpt<id>`` 层、``ep<N>`` 后不带 attempt、末尾用 tier）::

    <out-root>/<model_id>/seed<policy_seed>/[oracle|qwenvl|memer/]videos/<task>_ep<N>_<success|fail|timeout>_<task_goal>_<tier>.mp4
    <out-root>/<model_id>/seed<policy_seed>/[oracle|qwenvl|memer/]videos/index.tsv

用法::

    python scripts/eval-official/publish_videos.py --run-root <运行根> [--run-root …] --model-id groundsg \\
        --variant ground-sg-qwenvl --policy-seed 7 --dataset ood --side new [--out-root <run 根>] \\
        [--manifest <清单>] [--expect-total 86] [--mode link|copy] [--verify] [--out-json <报告>]

输入（每个 ``--run-root`` 递归找；隐藏目录与 ``videos/`` 发布目录不进）：

- 结果行 ``results.jsonl``：按 ``policy``（官方名，须等于 ``--model-id``）、``policy_variant``（给了 ``--variant`` 时）、
  ``dataset``（须等于 ``--dataset``）过滤；
- 账本 ``*.ledger.jsonl``：每个账本内每身份第一条 ``accept`` 行的 ``accepted_attempt_id`` 为权威（同
  ``env_client.AttemptLedger``）；多个账本给出不同接受计 ``ambiguous``。``--accept-from results`` 时（原侧没有
  AttemptLedger）改取每身份唯一的最终结果行（``canary``／``infra``／``late`` 不算），多于一条计 ``ambiguous``；
- 局目录 ``<key>.a<attempt_no>/``（内含 ``trace.jsonl``）：同名出现多份计 ``ambiguous``。

逐个被接受的身份：

- 终态只允许三态：strict-cap 命中（结果行或 trace 末行 ``cap_hit``）或 ``status=timeout`` 一律 ``timeout``；
  success／fail 照写；``status=error``：无帧（trace 末行 ``no_frame`` 为真或 ``frames_recorded == 0``，或结果行
  ``no_frame``）且 ``official/`` 无 mp4 的计 ``no_frame_error``，只进索引（``dst_name`` 为空）、不出视频；有帧或有视频的
  error 局计 ``error_named``、不发布（本版不允许 ``error`` 命名）；
- ``ep<N>``：ood 用 ``builder_episode``，hard-verify 用 ``source_episode``（trace header identity 优先，其次结果行）；
- 视频：``official/`` 下恰 1 个 mp4；sidecar（``provenance.json``，没有时 ``render.json``）必须在，记录的视频
  sha256 必须等于实际文件；``task_goal`` 取 sidecar，其次 trace 演示文本；
- 模型种子：结果行、trace header／identity、sidecar 里写了的 ``policy_seed`` 都必须等于 ``--policy-seed``，一处都没写
  计 ``seed_unproven``（不把历史缺字段补成已证种子）；
- 文件名超 255 字节沿用 ``render_official_video.safe_filename`` 截断加哈希。

发布：``--mode link``（默认，硬链接；跨文件系统时退回拷贝并在报告记 ``link_fallback``）或 ``copy``，先写临时名再
``os.replace``。目的文件已存在且 sha256 相同 → 幂等跳过（``reused``）；sha256 不同 → ``conflict``，不覆盖。
``index.tsv`` 列：``model_id policy_seed dataset side key accepted_attempt_id episode_id terminal src_rel src_sha256
dst_name``（首行为列名）；与已有索引按 ``(dataset, side, key)`` 合并，同键内容不同计 ``conflict``，原子重写。

``--verify``：只核不发布——期望文件都在且 sha256 相符（``missing_published``／``sha_mismatch``）、索引行与期望逐行一致
（``index_mismatch``）、``videos/`` 下没有期望之外的 mp4（``extra_files``）、没有任何文件名带 ``_error_``（``error_named``）。

判定行（末行）::

    VIDEO_LAYOUT=PASS|FAIL model=<m> seed=<s> videos=<n> no_frame_error=<e> error_named=<k> accepted=<n+e>
        published=<n> reused=<n> problems=<n> mode=publish|verify dir=<videos 目录>

``accepted = videos + no_frame_error``（1006 计划八.10 第 9 条用户裁决：无帧 error 保留例外、报告单列）。``problems``
为下列计数之和，>0 即 FAIL：missing、ambiguous、accept_without_row、conflict、name_collision、sha_mismatch、
provenance_missing、missing_video、multi_video、seed_mismatch、seed_unproven、episode_id_missing、task_goal_missing、
terminal_conflict、expect_total_mismatch、missing_published、index_mismatch、extra_files（逐项在 ``--out-json``）。
``error_named`` > 0 同样 FAIL。PASS 退出 0，否则 1。
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

HERE = Path(__file__).resolve().parent
DIR_RE = re.compile(r"^(?P<key>.+)\.a(?P<n>[1-9]\d*)$")
SIDECARS = ("provenance.json", "render.json")
VARIANT_DIRS = {"ground-sg-oracle": "oracle", "ground-sg-qwenvl": "qwenvl", "ground-sg-memer": "memer"}
VARIANT_MODEL = "groundsg"
INDEX_COLUMNS = ("model_id", "policy_seed", "dataset", "side", "key", "accepted_attempt_id", "episode_id", "terminal",
                 "src_rel", "src_sha256", "dst_name")
EPISODE_FIELD = {"ood": "builder_episode", "hard-verify": "source_episode"}
PROBLEM_KEYS = ("missing", "ambiguous", "accept_without_row", "conflict", "name_collision", "sha_mismatch",
                "provenance_missing", "missing_video", "multi_video", "seed_mismatch", "seed_unproven",
                "episode_id_missing", "task_goal_missing", "terminal_conflict", "expect_total_mismatch",
                "missing_published", "index_mismatch", "extra_files")


def _load(name: str, filename):
    """按路径加载模块（``filename`` 为相对本目录的文件名或绝对路径）；已加载则复用。"""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return mod


def official_defs():
    """评估包 ``robomme_hard_eval/models/_official_defs.py``（旧名别名表的唯一来源；模块名 ``official_defs``，已加载则复用）。"""
    mod = sys.modules.get("official_defs")
    if mod is not None and hasattr(mod, "canonical_row"):
        return mod
    sys.modules.pop("official_defs", None)
    return _load("official_defs", HERE.parents[1] / "src" / "robomme_hard_eval" / "models" / "_official_defs.py")


def safe_filename(full_name: str) -> str:
    """与重绘器同一实现（超 255 字节截断加 ``__<sha16>.mp4``）。"""
    src = str(HERE.parents[1] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from robomme_hard_eval.record.official_render import safe_filename as _safe

    return _safe(full_name)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """JSONL（半行跳过）；历史行的旧标签映射成官方名。"""
    canon = official_defs().canonical_row
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(canon(r))
    return out


def read_rows(path: Path) -> list[dict]:
    """清单：JSON 数组／``{"rows": [...]}`` 或 JSONL。"""
    text = Path(path).read_text(encoding="utf-8")
    s = text.lstrip()
    if s.startswith("[") or s.startswith("{"):
        try:
            data = json.loads(s)
        except ValueError:
            data = None
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            return [r for r in (data.get("rows") or data.get("identities") or []) if isinstance(r, dict)]
    return read_jsonl(path)


def key_of(row: dict) -> str:
    if row.get("key"):
        return str(row["key"])
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    tier = row.get("tier", ident.get("tier"))
    return f"{row.get('task', ident.get('task'))}_{tier}_{row.get('seed', ident.get('seed'))}"


def is_final(row: dict) -> bool:
    return not (row.get("canary") or row.get("infra") or row.get("late"))


# ── 扫描运行根 ───────────────────────────────────────────────────────────────


def scan_root(root: Path) -> dict:
    """``{"results": [...], "ledgers": [...], "episodes": [(局目录, 运行根)]}``；隐藏目录与 ``videos/`` 不进。"""
    out = {"results": [], "ledgers": [], "episodes": []}
    for cur, dirs, files in os.walk(root):
        cur_p = Path(cur)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "videos")
        if DIR_RE.match(cur_p.name) and "trace.jsonl" in files:
            out["episodes"].append(cur_p)
            dirs[:] = []
            continue
        for f in sorted(files):
            if f == "results.jsonl":
                out["results"].append(cur_p / f)
            elif f.endswith(".ledger.jsonl"):
                out["ledgers"].append(cur_p / f)
    return out


def trace_edges(d: Path) -> tuple[dict, dict, dict]:
    """(header, demo, end)；读不到返回空字典。"""
    try:
        rows = [json.loads(x) for x in (d / "trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        return {}, {}, {}
    rows = [r for r in rows if isinstance(r, dict)]
    head = rows[0] if rows and rows[0].get("kind") == "header" else {}
    end = rows[-1] if rows and rows[-1].get("kind") == "end" else {}
    demo = next((r for r in rows if r.get("kind") == "demo"), {})
    return head, demo, end


def load_sidecar(off: Path) -> tuple[str | None, dict | None]:
    for name in SIDECARS:
        p = off / name
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return name, None
            return name, data if isinstance(data, dict) else None
    return None, None


def sidecar_video_sha(side: dict):
    v = side.get("video") if isinstance(side.get("video"), dict) else {}
    o = side.get("output_fingerprint") if isinstance(side.get("output_fingerprint"), dict) else {}
    return side.get("video_sha256") or v.get("sha256") or o.get("sha256")


# ── 主体 ─────────────────────────────────────────────────────────────────────


def accepted_from_ledgers(ledgers: list[Path], model_id: str) -> tuple[dict, set]:
    """``{key: (accepted_attempt_id, attempt_no|None)}`` 与冲突 key 集合（每账本第一条 accept 为权威）。"""
    out: dict = {}
    conflict: set = set()
    for path in ledgers:
        rows = [r for r in read_jsonl(path) if r.get("policy") in (None, model_id)]
        start_no = {r.get("attempt_id"): r.get("attempt_no") for r in rows if r.get("kind") == "attempt_start"}
        per: dict = {}
        for r in rows:
            if r.get("kind") != "accept" or not r.get("key"):
                continue
            aid = r.get("accepted_attempt_id") or r.get("attempt_id")
            no = r.get("attempt_no") if r.get("attempt_id") == aid and r.get("attempt_no") is not None else start_no.get(aid)
            per.setdefault(str(r["key"]), (aid, no))
        for k, v in per.items():
            if k in out and out[k] != v:
                conflict.add(k)
            out.setdefault(k, v)
    return out, conflict


def videos_dir(out_root: Path, model_id: str, policy_seed: int, variant: str | None) -> Path:
    d = Path(out_root) / model_id / f"seed{int(policy_seed)}"
    if variant is not None:
        d = d / VARIANT_DIRS[variant]
    return d / "videos"


def name_terminal(row: dict, end: dict) -> tuple[str | None, str | None]:
    """(命名终态, 冲突说明)。strict-cap 命中或 timeout → timeout；success／fail 照写；其余返回 error。"""
    rs, es = row.get("status"), end.get("status")
    if row.get("cap_hit") is True or end.get("cap_hit") is True or rs == "timeout" or es == "timeout":
        return "timeout", None
    if rs in ("success", "fail"):
        if es is not None and es != rs:
            return rs, f"result.status={rs} trace.end.status={es}"
        return rs, None
    if rs is None and es in ("success", "fail"):
        return es, None
    return "error", None


def plan_episode(k: str, row: dict, ep: Path, run_root: Path, args, counts: dict, problems: list) -> dict | None:
    """一个被接受身份的发布计划；返回索引行（含内部键 ``_src``）或 None（已计入问题）。"""
    def bad(kind: str, detail: str) -> None:
        counts[kind] += 1
        problems.append({"key": k, "kind": kind, "detail": detail, "dir": str(ep)})

    head, demo, end = trace_edges(ep)
    ident = head.get("identity") if isinstance(head.get("identity"), dict) else {}
    row_ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    off = ep / "official"
    mp4s = sorted(p for p in off.glob("*.mp4") if not p.name.startswith(".")) if off.is_dir() else []
    side_name, side = load_sidecar(off)
    # 模型种子：写了的都要等于 --policy-seed，一处都没写记 seed_unproven
    seeds = {"result": row.get("policy_seed"), "trace.header": head.get("policy_seed"),
             "trace.identity": ident.get("policy_seed"), "sidecar": (side or {}).get("policy_seed")}
    seen = {s: v for s, v in seeds.items() if v is not None}
    if not seen:
        bad("seed_unproven", "结果行、trace、sidecar 都没有 policy_seed")
        return None  # 种子未证实的局不进 seed<s>/ 目录
    if any(v != args.policy_seed for v in seen.values()):
        bad("seed_mismatch", json.dumps(seen, ensure_ascii=False))
        return None
    terminal, conflict = name_terminal(row, end)
    if conflict:
        bad("terminal_conflict", conflict)
        return None
    field = EPISODE_FIELD[args.dataset]
    epn = ident.get(field, row_ident.get(field, row.get(field)))
    entry = {"model_id": args.model_id, "policy_seed": str(args.policy_seed), "dataset": args.dataset,
             "side": args.side, "key": k, "accepted_attempt_id": str(row.get("attempt_id") or ""),
             "episode_id": "" if epn is None else str(epn), "terminal": terminal, "src_rel": "",
             "src_sha256": "", "dst_name": ""}
    if terminal == "error":
        no_frame = (row.get("no_frame") is True or end.get("no_frame") is True or end.get("frames_recorded") == 0)
        if no_frame and not mp4s:
            entry["_no_frame"] = True
            return entry
        bad_kind = "error_named"
        counts[bad_kind] += 1
        problems.append({"key": k, "kind": bad_kind, "dir": str(ep),
                         "detail": f"error 局有帧或有视频（mp4={len(mp4s)} frames_recorded={end.get('frames_recorded')}），"
                                   "本版不允许 error 命名、不发布"})
        return None
    if epn is None:
        bad("episode_id_missing", f"缺 {field}")
        return None
    if not mp4s:
        bad("missing_video", "official/ 下没有 mp4")
        return None
    if len(mp4s) > 1:
        bad("multi_video", ",".join(p.name for p in mp4s))
        return None
    src = mp4s[0]
    sha = sha256_file(src)
    if side_name is None or side is None:
        bad("provenance_missing", f"official/ 缺可读 sidecar（{side_name}）")
        return None
    if sidecar_video_sha(side) != sha:
        bad("sha_mismatch", f"sidecar={sidecar_video_sha(side)} actual={sha}")
        return None
    texts = demo.get("texts") if isinstance(demo.get("texts"), list) else []
    goal = side.get("task_goal") or (texts[0] if texts and isinstance(texts[0], str) else None) or row.get("task_goal")
    if not goal:
        bad("task_goal_missing", "sidecar、trace 演示文本、结果行都没有 task_goal")
        return None
    task = ident.get("task", row.get("task"))
    tier = ident.get("tier", row.get("tier", row_ident.get("tier")))
    full = f"{task}_ep{epn}_{terminal}_{goal}_{tier}.mp4"
    try:
        rel = src.relative_to(run_root).as_posix()
    except ValueError:
        rel = str(src)
    entry.update(src_rel=rel, src_sha256=sha, dst_name=safe_filename(full), _src=src, _full_name=full)
    return entry


def read_index(path: Path) -> tuple[dict, list[str]]:
    """``{(dataset, side, key): 行}`` 与坏行说明。"""
    out: dict = {}
    bad: list[str] = []
    if not path.is_file():
        return out, bad
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != INDEX_COLUMNS:
        return out, ["index_header"]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) != len(INDEX_COLUMNS):
            bad.append(ln[:120])
            continue
        r = dict(zip(INDEX_COLUMNS, parts))
        out[(r["dataset"], r["side"], r["key"])] = r
    return out, bad


def write_index(path: Path, rows: dict) -> None:
    tmp = path.with_name(f".index-{uuid.uuid4().hex}.tsv")
    body = ["\t".join(INDEX_COLUMNS)] + ["\t".join(r[c] for c in INDEX_COLUMNS) for _, r in sorted(rows.items())]
    tmp.write_text("\n".join(body) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def place(src: Path, dst: Path, mode: str) -> bool:
    """临时名写入后 ``os.replace``；返回是否发生了硬链接退回拷贝。"""
    tmp = dst.with_name(f".publish-{uuid.uuid4().hex}.mp4")
    fallback = False
    try:
        if mode == "link":
            try:
                os.link(src, tmp)
            except OSError as e:
                if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
                    raise
                shutil.copyfile(src, tmp)
                fallback = True
        else:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return fallback


def run(args) -> dict:
    defs = official_defs()
    counts = {k: 0 for k in PROBLEM_KEYS}
    counts.update(videos=0, no_frame_error=0, error_named=0, published=0, reused=0, link_fallback=0)
    problems: list[dict] = []
    roots = [Path(r).resolve() for r in args.run_root]
    out_root = Path(args.out_root).resolve() if args.out_root else roots[0]
    vdir = videos_dir(out_root, args.model_id, args.policy_seed, args.variant)
    scans = {r: scan_root(r) for r in roots}

    def wanted(r: dict) -> bool:
        if r.get("policy") not in (None, args.model_id):
            return False
        if args.variant is not None and r.get("policy_variant") not in (None, args.variant):
            return False
        return defs.canonical_dataset(r.get("dataset")) == args.dataset

    rows = [r for sc in scans.values() for p in sc["results"] for r in read_jsonl(p) if wanted(r) and not r.get("canary")]
    by_aid: dict = {}
    for r in rows:
        if r.get("attempt_id"):
            by_aid.setdefault(str(r["attempt_id"]), []).append(r)
    keys = {key_of(r) for r in rows}
    accepts: dict = {}
    conflict: set = set()
    if args.accept_from == "ledger":
        acc, conflict = accepted_from_ledgers([p for sc in scans.values() for p in sc["ledgers"]], args.model_id)
        accepts = {k: v for k, v in acc.items() if k in keys}
        conflict &= keys  # 别的模型／数据集的账本冲突不归本次发布
    else:
        finals: dict = {}
        for r in rows:
            if is_final(r):
                finals.setdefault(key_of(r), []).append(r)
        for k, lst in finals.items():
            if len(lst) > 1:
                conflict.add(k)
            r = lst[0]
            accepts[k] = (r.get("attempt_id"), r.get("attempt", r.get("attempt_no")))
    eps: dict = {}
    for root, sc in scans.items():
        for d in sc["episodes"]:
            m = DIR_RE.match(d.name)
            eps.setdefault((m.group("key"), int(m.group("n"))), []).append((d, root))
    expected_keys = None
    if args.manifest:
        expected_keys = {key_of(r) for r in read_rows(Path(args.manifest))}
        for k in sorted(expected_keys - set(accepts) - conflict):
            counts["missing"] += 1
            problems.append({"key": k, "kind": "missing", "detail": "清单身份没有被接受的 attempt"})
    entries: dict = {}
    for k in sorted(set(accepts) | conflict):
        if expected_keys is not None and k not in expected_keys:
            continue
        if k in conflict:
            counts["ambiguous"] += 1
            problems.append({"key": k, "kind": "ambiguous", "detail": "账本／结果给出多个接受尝试"})
            continue
        aid, no = accepts[k]
        lst = by_aid.get(str(aid), []) if aid is not None else []
        row = lst[-1] if lst else None
        if row is None and args.accept_from == "results":
            row = next((r for r in rows if key_of(r) == k and is_final(r)), None)
        if row is None:
            counts["accept_without_row"] += 1
            problems.append({"key": k, "kind": "accept_without_row", "detail": f"accepted_attempt_id={aid}"})
            continue
        if no is None:
            no = row.get("attempt_no", row.get("attempt"))
        cands = eps.get((k, int(no)), []) if no is not None else []
        if not cands:
            counts["missing"] += 1
            problems.append({"key": k, "kind": "missing", "detail": f"局目录 {k}.a{no} 不存在"})
            continue
        if len(cands) > 1:
            counts["ambiguous"] += 1
            problems.append({"key": k, "kind": "ambiguous", "detail": ",".join(str(d) for d, _ in cands)})
            continue
        ep, root = cands[0]
        e = plan_episode(k, row, ep, root, args, counts, problems)
        if e is not None:
            entries[(args.dataset, args.side, k)] = e
    # 同名不同身份
    by_name: dict = {}
    for ik, e in entries.items():
        if e["dst_name"]:
            by_name.setdefault(e["dst_name"], []).append(ik)
    for name, iks in by_name.items():
        if len(iks) > 1:
            counts["name_collision"] += 1
            problems.append({"kind": "name_collision", "detail": name, "keys": [x[2] for x in iks]})
            for ik in iks:
                entries.pop(ik, None)
    counts["no_frame_error"] = sum(1 for e in entries.values() if e.get("_no_frame"))
    counts["videos"] = sum(1 for e in entries.values() if e["dst_name"])
    accepted_n = counts["videos"] + counts["no_frame_error"]
    if args.expect_total is not None and accepted_n != args.expect_total:
        counts["expect_total_mismatch"] += 1
        problems.append({"kind": "expect_total_mismatch", "detail": f"accepted={accepted_n} expect={args.expect_total}"})
    index_path = vdir / "index.tsv"
    old_index, bad_lines = read_index(index_path)
    public = {ik: {c: e[c] for c in INDEX_COLUMNS} for ik, e in entries.items()}
    if args.verify:
        if not vdir.is_dir():
            counts["missing_published"] += len([e for e in entries.values() if e["dst_name"]])
        for ik, e in entries.items():
            if not e["dst_name"]:
                continue
            dst = vdir / e["dst_name"]
            if not dst.is_file():
                counts["missing_published"] += 1
                problems.append({"key": ik[2], "kind": "missing_published", "detail": str(dst)})
            elif sha256_file(dst) != e["src_sha256"]:
                counts["sha_mismatch"] += 1
                problems.append({"key": ik[2], "kind": "sha_mismatch", "detail": f"published {dst.name}"})
        mine = {ik: r for ik, r in old_index.items() if ik[0] == args.dataset and ik[1] == args.side}
        if bad_lines or mine != public:
            diff = sorted(set(mine) ^ set(public)) + sorted(ik for ik in set(mine) & set(public) if mine[ik] != public[ik])
            counts["index_mismatch"] += max(1, len(diff)) if (bad_lines or diff) else 0
            problems.append({"kind": "index_mismatch", "detail": bad_lines[:3] + [list(x) for x in diff[:10]]})
        if vdir.is_dir():
            indexed = {r["dst_name"] for r in old_index.values() if r["dst_name"]}
            for p in sorted(vdir.glob("*.mp4")):
                if "_error_" in p.name:
                    counts["error_named"] += 1
                    problems.append({"kind": "error_named", "detail": p.name})
                if p.name not in indexed:
                    counts["extra_files"] += 1
                    problems.append({"kind": "extra_files", "detail": p.name})
    else:
        merged = dict(old_index)
        if bad_lines:
            counts["conflict"] += 1
            problems.append({"kind": "conflict", "detail": f"已有 index.tsv 不合格：{bad_lines[:3]}"})
        for ik, e in sorted(entries.items()):
            old = old_index.get(ik)
            if old is not None and old != public[ik]:
                counts["conflict"] += 1
                problems.append({"key": ik[2], "kind": "conflict", "detail": f"索引已有不同记录：{old}"})
                continue
            if e["dst_name"]:
                dst = vdir / e["dst_name"]
                if dst.exists():
                    if sha256_file(dst) == e["src_sha256"]:
                        counts["reused"] += 1
                    else:
                        counts["conflict"] += 1
                        problems.append({"key": ik[2], "kind": "conflict", "detail": f"同名异 sha：{dst.name}"})
                        continue
                else:
                    vdir.mkdir(parents=True, exist_ok=True)
                    counts["link_fallback"] += place(e["_src"], dst, args.mode)
                    if sha256_file(dst) != e["src_sha256"]:
                        counts["sha_mismatch"] += 1
                        problems.append({"key": ik[2], "kind": "sha_mismatch", "detail": f"发布后 {dst.name}"})
                        continue
                    counts["published"] += 1
            merged[ik] = public[ik]
        if entries or old_index:
            vdir.mkdir(parents=True, exist_ok=True)
            write_index(index_path, merged)
    n_prob = sum(counts[k] for k in PROBLEM_KEYS)
    verdict = "PASS" if n_prob == 0 and counts["error_named"] == 0 else "FAIL"
    line = (f"VIDEO_LAYOUT={verdict} model={args.model_id} seed={args.policy_seed} videos={counts['videos']} "
            f"no_frame_error={counts['no_frame_error']} error_named={counts['error_named']} accepted={accepted_n} "
            f"published={counts['published']} reused={counts['reused']} problems={n_prob} "
            f"mode={'verify' if args.verify else 'publish'} dir={vdir}")
    return {"verdict": verdict, "line": line, "counts": counts, "problems": problems, "videos_dir": str(vdir),
            "entries": [{**public[ik], "full_name": e.get("_full_name")} for ik, e in sorted(entries.items())]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", action="append", required=True, help="运行根（可重复；递归找结果、账本与局目录）")
    ap.add_argument("--out-root", default=None, help="发布根（缺省第一个 --run-root）")
    ap.add_argument("--model-id", required=True, help="策略标签官方名：perceptual-framesamp-modul／smvla／pp／groundsg／astra")
    ap.add_argument("--variant", default=None, choices=sorted(VARIANT_DIRS), help="GroundSG 变体（多一层子目标目录）")
    ap.add_argument("--policy-seed", type=int, required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(EPISODE_FIELD))
    ap.add_argument("--side", required=True, choices=["new", "orig"])
    ap.add_argument("--accept-from", choices=["ledger", "results"], default="ledger",
                    help="接受尝试的来源：账本 accept 行（默认）或每身份唯一最终结果行（原侧无 AttemptLedger 时）")
    ap.add_argument("--manifest", default=None, help="身份清单：每个身份都必须有被接受的尝试，清单外的不发布")
    ap.add_argument("--expect-total", type=int, default=None, help="accepted（= videos + no_frame_error）必须等于它")
    ap.add_argument("--mode", choices=["link", "copy"], default="link")
    ap.add_argument("--verify", action="store_true", help="只核不发布")
    ap.add_argument("--out-json", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    defs = official_defs()
    if defs.canonical_policy(args.model_id) != args.model_id:
        ap.error(f"--model-id 只接受官方名（收到 {args.model_id}）")
    if (args.model_id == VARIANT_MODEL) != (args.variant is not None):
        ap.error("groundsg 必须给 --variant，其他模型不得给")
    if args.policy_seed < 0:
        ap.error("--policy-seed 必须为非负整数")
    res = run(args)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n",
                                       encoding="utf-8")
    for p in res["problems"][:20]:
        print(f"VIDEO_LAYOUT_PROBLEM {json.dumps(p, ensure_ascii=False, default=str)}", flush=True)
    print(res["line"], flush=True)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
