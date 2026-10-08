#!/usr/bin/env python3
"""官方版式视频验收（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「S2b」，审计第 9 条）。

只读：不改、不删任何局目录；只用标准库与 ffmpeg（完整解码数帧）。

两种用法：

1. 全量验收（主会话在每批收尾跑）::

     official_media_check.py --manifest <冻结清单> [--manifest …] --ledger <账本> [--ledger …]
         --root <发布根> [--root …] --dataset {ood,hard-verify} [--route <路线>] [--out official-media.jsonl]

   - 身份从冻结清单枚举（``eval_manifest.py`` 的分片 JSON 数组或 JSONL，每行一个身份，取 ``key``，缺省按
     ``<task>_<tier>_<seed>`` 现算），不从目录反推；
   - 账本按 ``env_client.AttemptLedger`` 的口径读：每个账本文件内每身份第一条 ``accept`` 行的
     ``accepted_attempt_id`` 为权威，尝试号取该行 ``attempt_no``（缺失时按同 ``attempt_id`` 的 ``attempt_start`` 行补）；
     多个账本对同一身份给出不同的接受尝试计 ``ambiguous``；
   - 局目录 = 某个 ``--root`` 下的 ``<key>.a<尝试号>/``；同名出现在多个根、或另有 ``.dupN`` 副本，计 ``ambiguous``；
   - 逐局核 ``official/`` 下恰 1 个 mp4、完整解码帧数等于 ``trace.jsonl`` 末行 ``frames_recorded``（C8）、
     provenance（``official/provenance.json``，没有时取重绘器的 ``official/render.json``）的 identity、route、
     dataset、attempt 与该局 trace 及账本一致、记录的视频 sha256 等于实际文件；
   - ``end.status == "error"`` 且无帧（``end.no_frame`` 为真或 ``frames_recorded == 0``）、``official/`` 无 mp4 的局
     计 ``no_frame_error``，不计 ``fail``；
   - 输出两行判定与逐身份 ``official-media.jsonl``（缺省写在第一个 ``--root`` 下）::

       OFFICIAL_MEDIA_INPUTS=PASS missing=0 extra=0 ambiguous=0 identity_mismatch=0 attempt_mismatch=0 provenance_missing=0
       OFFICIAL_MEDIA=PASS total=<n> skip=0 fail=0 no_frame_error=<n> videos=<n> accepted=<n> [policy_seed=<s>]

     ``missing``：清单身份无接受尝试或接受尝试的局目录不存在；``extra``：账本接受了、或发布根里有清单外的身份；
     ``skip``：无法定位局目录（missing／ambiguous）的身份数；``fail``：定位到但核验不过的身份数。
     两行都 PASS 退出 0，否则 1。

     无帧 error 口径（1006 计划八.10 第 9 条用户裁决「保留例外、报告单列」，三个检查器统一）：
     ``accepted = videos + no_frame_error``——``videos`` 为核验通过、有视频的身份数，``no_frame_error`` 单列，不设上限、
     不计 ``fail``；86 个身份完整覆盖不等于 86 个视频。
   - 1006 第三阶段：视频文件名带 ``_error_`` 而该局 trace 终态不是 error（strict-cap 命中 ``end.cap_hit`` 或
     ``status=timeout`` 应命名 ``timeout``）计 ``terminal_name`` 失败；``--policy-seed <s>`` 时 trace header／identity
     与 sidecar 记录的 ``policy_seed`` 必须等于它（trace 缺字段即失败，不把历史缺字段补成已证种子）。

2. 单局核验（``seat_media_lib.sh::render_official_dir`` 调用）::

     official_media_check.py --verify-dir <局目录> [--ffmpeg <路径>]

   规则同上，但没有清单与账本可比；trace 尚无 ``frames_recorded``（S4／S5 合入前的旧路线）时以 provenance 记录的
   帧数为期望（打印 ``frames_source=sidecar``）。打印 ``OFFICIAL_VERIFY=PASS|NO_FRAME|FAIL dir= …``，
   PASS／NO_FRAME 退出 0，FAIL 退出 1。

provenance 里视频 sha256 的键按以下顺序取：``video_sha256``、``video.sha256``、``output_fingerprint.sha256``
（重绘器 ``render.json``）；帧数取 ``frames_recorded``，其次 ``frames``；dataset／attempt 取顶层同名键，其次
``identity`` 内同名键。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

DIR_RE = re.compile(r"^(?P<key>.+)\.a(?P<n>[1-9]\d*)(?:\.dup(?P<dup>\d+))?$")
SIDECARS = ("provenance.json", "render.json")
IDENTITY_KEYS = ("task", "tier", "seed", "source_episode", "builder_episode", "key")


def ffmpeg_exe() -> str | None:
    """顺序与 seat_media_lib.sh 的 transcode_episode_dir／seat_media_ffmpeg 相同。"""
    for c in (os.environ.get("SGEVAL_FFMPEG"), os.environ.get("V75_FFMPEG"), "/usr/bin/ffmpeg", shutil.which("ffmpeg")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def decoded_frames(ff: str, path: Path) -> int:
    """完整解码数帧；解码出错返回 -1。"""
    p = subprocess.run([ff, "-nostdin", "-v", "error", "-nostats", "-i", str(path), "-map", "0:v:0", "-f", "null", "-",
                        "-progress", "pipe:1"], capture_output=True, text=True)
    fr = [x.split("=", 1)[1].strip() for x in p.stdout.splitlines() if x.startswith("frame=")]
    if p.returncode != 0 or p.stderr.strip() or not fr or not fr[-1].isdigit():
        return -1
    return int(fr[-1])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def official_defs():
    """同目录 ``official_defs.py``（旧名别名表的唯一来源；已加载则复用同一模块）。"""
    import importlib.util

    mod = sys.modules.get("official_defs")
    if mod is None:
        spec = importlib.util.spec_from_file_location("official_defs", Path(__file__).resolve().parent / "official_defs.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["official_defs"] = mod
        spec.loader.exec_module(mod)
    return mod


def read_rows(path: Path) -> list[dict]:
    """JSON 数组或 JSONL；半行跳过。历史行的旧策略标签／数据集名／路线映射成官方名。"""
    canon = official_defs().canonical_row
    text = Path(path).read_text(encoding="utf-8")
    s = text.lstrip()
    if s.startswith("["):
        data = json.loads(s)
        return [canon(r) for r in data if isinstance(r, dict)]
    rows = []
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict):
            rows.append(canon(r))
    return rows


def key_of(row: dict) -> str:
    if row.get("key"):
        return str(row["key"])
    return f"{row['task']}_{row['tier']}_{int(row['seed'])}"


def _get(d: dict, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def sidecar_video_sha(side: dict):
    return side.get("video_sha256") or _get(side, "video", "sha256") or _get(side, "output_fingerprint", "sha256")


def sidecar_frames(side: dict):
    """sidecar 记录的帧数。重绘器 render.json 写整数；GroundSG 官方原生视频的 provenance.json（S1）把 ``frames``
    写成字典（``decoded``／``expected``／``frames_recorded``／``basis`` 等），此时取 ``decoded``，并要求三者一致，
    不一致返回字符串说明，由调用方计入失败原因。"""
    v = side.get("frames_recorded")
    if v is not None:
        return v
    f = side.get("frames")
    if isinstance(f, dict):
        vals = {k: f.get(k) for k in ("decoded", "expected", "frames_recorded") if f.get(k) is not None}
        if not vals:
            return "frames_dict_without_counts"
        if len(set(vals.values())) != 1:
            return f"frames_dict_inconsistent:{vals}"
        return next(iter(vals.values()))
    return f


def sidecar_field(side: dict, name: str):
    v = side.get(name)
    return v if v is not None else _get(side, "identity", name)


def load_sidecar(off: Path) -> tuple[str | None, dict | None]:
    for name in SIDECARS:
        p = off / name
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return name, None
            return name, official_defs().canonical_row(data) if isinstance(data, dict) else None
    return None, None


def verify_dir(ep: Path, ff: str, *, strict_frames: bool, expected_attempt: int | None = None,
               expected_dataset: str | None = None, expected_route: str | None = None,
               expected_identity: dict | None = None, expected_policy_seed: int | None = None) -> dict:
    """单局核验。返回 ``{"status": pass|fail|no_frame_error, "reasons": [...], ...}``；reasons 的前缀决定计数类别：
    ``identity:``、``attempt:``、``provenance_missing``，其余只计 fail。"""
    ep = Path(ep)
    res: dict = {"dir": ep.name, "path": str(ep), "reasons": []}
    reasons: list[str] = res["reasons"]
    trace = ep / "trace.jsonl"
    try:
        rows = read_rows(trace)
    except (OSError, ValueError) as e:
        reasons.append(f"trace_unreadable:{type(e).__name__}")
        res["status"] = "fail"
        return res
    if not rows or rows[0].get("kind") != "header" or rows[-1].get("kind") != "end":
        reasons.append("trace_incomplete")
        res["status"] = "fail"
        return res
    header, end = rows[0], rows[-1]
    ident = header.get("identity") if isinstance(header.get("identity"), dict) else {}
    route = header.get("route")
    res.update(key=ident.get("key"), route=route, trace_attempt=ident.get("attempt"), end_status=end.get("status"))
    # 目录名与 trace 身份（C6）
    m = DIR_RE.match(ep.name)
    if not m or m.group("dup"):
        reasons.append(f"identity:dir_name={ep.name}")
    else:
        if ident.get("key") != m.group("key"):
            reasons.append(f"identity:dir_key={m.group('key')} trace_key={ident.get('key')}")
        if ident.get("attempt") != int(m.group("n")):
            reasons.append(f"attempt:dir={m.group('n')} trace={ident.get('attempt')}")
    if expected_identity is not None:
        for k in IDENTITY_KEYS:
            if k in expected_identity and k in ident and expected_identity[k] != ident[k]:
                reasons.append(f"identity:{k} manifest={expected_identity[k]} trace={ident[k]}")
        if expected_identity.get("key") and ident.get("key") != expected_identity["key"]:
            if not any(r.startswith("identity:key ") for r in reasons):
                reasons.append(f"identity:key manifest={expected_identity['key']} trace={ident.get('key')}")
    if expected_attempt is not None and ident.get("attempt") != expected_attempt:
        reasons.append(f"attempt:ledger={expected_attempt} trace={ident.get('attempt')}")
    if expected_dataset is not None and ident.get("dataset") != expected_dataset:
        reasons.append(f"identity:dataset expect={expected_dataset} trace={ident.get('dataset')}")
    if expected_route is not None and route != expected_route:
        reasons.append(f"route expect={expected_route} trace={route}")
    trace_seed = header.get("policy_seed", ident.get("policy_seed"))
    res["policy_seed"] = trace_seed
    if expected_policy_seed is not None and trace_seed != expected_policy_seed:
        reasons.append(f"policy_seed:expect={expected_policy_seed} trace={trace_seed}")
    # 官方视频
    off = ep / "official"
    mp4s = sorted(p for p in off.glob("*.mp4") if not p.name.startswith(".")) if off.is_dir() else []
    res["official_mp4s"] = [p.name for p in mp4s]
    no_frame = end.get("status") == "error" and (end.get("no_frame") is True or end.get("frames_recorded") == 0)
    if no_frame:
        if mp4s:
            reasons.append(f"no_frame_but_video:{len(mp4s)}")
        res["status"] = "fail" if reasons else "no_frame_error"
        return res
    if len(mp4s) != 1:
        reasons.append(f"official_count={len(mp4s)}")
        res["status"] = "fail"
        return res
    video = mp4s[0]
    # 1006：strict-cap 命中或 status=timeout 的局命名 timeout；文件名带 _error_ 的旧口径不再放行
    named = "timeout" if (end.get("cap_hit") is True or end.get("status") == "timeout") else end.get("status")
    res["named_terminal"] = named
    if "_error_" in video.name and named != "error":
        reasons.append(f"terminal_name:error_named status={end.get('status')} cap_hit={end.get('cap_hit')}")
    side_name, side = load_sidecar(off)
    res["sidecar"] = side_name
    expected = end.get("frames_recorded")
    res["frames_source"] = "trace"
    if not (isinstance(expected, int) and not isinstance(expected, bool)):
        if strict_frames:
            reasons.append("frames_recorded_missing")
            expected = None
        else:
            expected = sidecar_frames(side) if side else None
            res["frames_source"] = "sidecar"
            if not isinstance(expected, int) or isinstance(expected, bool):
                reasons.append("frames_expected_missing")
                expected = None
    n = decoded_frames(ff, video)
    actual_sha = sha256_file(video)
    res.update(video=video.name, frames_expected=expected, frames_decoded=n, video_sha256=actual_sha)
    if n < 0:
        reasons.append("undecodable")
    elif expected is not None and n != expected:
        reasons.append(f"frames_mismatch decoded={n} expect={expected}")
    if side_name is None:
        reasons.append("provenance_missing")
    elif side is None:
        reasons.append(f"provenance_unreadable:{side_name}")
    else:
        if side.get("identity") != ident:
            reasons.append("provenance_identity_mismatch")
        if side.get("route") != route:
            reasons.append(f"provenance_route={side.get('route')} trace={route}")
        if sidecar_field(side, "dataset") != ident.get("dataset"):
            reasons.append(f"provenance_dataset={sidecar_field(side, 'dataset')} trace={ident.get('dataset')}")
        if sidecar_field(side, "attempt") != ident.get("attempt"):
            reasons.append(f"provenance_attempt={sidecar_field(side, 'attempt')} trace={ident.get('attempt')}")
        if expected_attempt is not None and sidecar_field(side, "attempt") != expected_attempt:
            reasons.append(f"attempt:provenance={sidecar_field(side, 'attempt')} ledger={expected_attempt}")
        if sidecar_video_sha(side) != actual_sha:
            reasons.append("provenance_video_sha_mismatch")
        sf = sidecar_frames(side)
        if sf is not None and n >= 0 and sf != n:
            reasons.append(f"provenance_frames={sf} decoded={n}")
        side_seed = side.get("policy_seed")
        if side_seed is not None and trace_seed is not None and side_seed != trace_seed:
            reasons.append(f"policy_seed:provenance={side_seed} trace={trace_seed}")
    res["status"] = "fail" if reasons else "pass"
    return res


def load_accepts(ledgers: list[Path]) -> tuple[dict[str, tuple[str, int | None]], set[str]]:
    """``{key: (accepted_attempt_id, attempt_no)}`` 与有冲突的 key 集合。"""
    out: dict[str, tuple[str, int | None]] = {}
    conflict: set[str] = set()
    for path in ledgers:
        rows = read_rows(path)
        start_no = {r.get("attempt_id"): r.get("attempt_no") for r in rows if r.get("kind") == "attempt_start"}
        per: dict[str, tuple[str, int | None]] = {}
        for r in rows:
            if r.get("kind") != "accept" or not r.get("key"):
                continue
            aid = r.get("accepted_attempt_id")
            no = r.get("attempt_no") if r.get("attempt_id") == aid and r.get("attempt_no") is not None else start_no.get(aid)
            per.setdefault(str(r["key"]), (aid, no))  # 与 AttemptLedger 相同：第一条 accept 为权威
        for k, v in per.items():
            if k in out and out[k] != v:
                conflict.add(k)
            out.setdefault(k, v)
    return out, conflict


def index_roots(roots: list[Path]) -> dict[tuple[str, int], list[Path]]:
    idx: dict[tuple[str, int], list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            m = DIR_RE.match(p.name)
            if m:
                idx.setdefault((m.group("key"), int(m.group("n"))), []).append(p)
    return idx


def check_all(manifests: list[Path], ledgers: list[Path], roots: list[Path], dataset: str, route: str | None,
              ff: str, policy_seed: int | None = None) -> tuple[dict, dict, list[dict]]:
    counts = dict(missing=0, extra=0, ambiguous=0, identity_mismatch=0, attempt_mismatch=0, provenance_missing=0)
    media = dict(total=0, skip=0, fail=0, no_frame_error=0, passed=0)
    out: list[dict] = []
    ident_rows: dict[str, dict] = {}
    dup_keys: set[str] = set()
    for mf in manifests:
        for r in read_rows(mf):
            k = key_of(r)
            if k in ident_rows:
                dup_keys.add(k)
            ident_rows.setdefault(k, r)
    accepts, conflict = load_accepts(ledgers)
    idx = index_roots(roots)
    for k in sorted(ident_rows):
        row = ident_rows[k]
        media["total"] += 1
        rec = {"key": k, "status": None, "reasons": []}
        acc = accepts.get(k)
        if k in dup_keys or k in conflict:
            counts["ambiguous"] += 1
            rec.update(status="skip", reasons=["manifest_duplicate" if k in dup_keys else "ledger_conflict"])
        elif acc is None:
            counts["missing"] += 1
            rec.update(status="skip", reasons=["no_accepted_attempt"])
        elif not isinstance(acc[1], int):
            counts["attempt_mismatch"] += 1
            rec.update(status="skip", accepted_attempt_id=acc[0], reasons=["attempt_no_unknown"])
        else:
            rec.update(accepted_attempt_id=acc[0], attempt=acc[1])
            cands = idx.get((k, acc[1]), [])
            if not cands:
                counts["missing"] += 1
                rec.update(status="skip", reasons=[f"dir_missing:{k}.a{acc[1]}"])
            elif len(cands) > 1:
                counts["ambiguous"] += 1
                rec.update(status="skip", reasons=["ambiguous_dirs:" + ",".join(str(p) for p in cands)])
            else:
                res = verify_dir(cands[0], ff, strict_frames=True, expected_attempt=acc[1], expected_dataset=dataset,
                                 expected_route=route, expected_identity=row, expected_policy_seed=policy_seed)
                rec.update(res, key=k)
                rs = res["reasons"]
                if any(r.startswith("identity:") for r in rs):
                    counts["identity_mismatch"] += 1
                if any(r.startswith("attempt:") for r in rs):
                    counts["attempt_mismatch"] += 1
                if "provenance_missing" in rs:
                    counts["provenance_missing"] += 1
        if rec["status"] == "skip":
            media["skip"] += 1
        elif rec["status"] == "fail":
            media["fail"] += 1
        elif rec["status"] == "no_frame_error":
            media["no_frame_error"] += 1
        else:
            media["passed"] += 1
        out.append(rec)
    # 清单外：账本接受的身份、发布根里的身份目录
    extra_keys = {k for k in accepts if k not in ident_rows} | {k for (k, _n) in idx if k not in ident_rows}
    for k in sorted(extra_keys):
        counts["extra"] += 1
        out.append({"key": k, "status": "extra", "reasons": ["not_in_manifest"],
                    "accepted": k in accepts, "dirs": sorted(str(p) for (kk, _n), ps in idx.items() if kk == k for p in ps)})
    return counts, media, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-dir", type=Path)
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--ledger", type=Path, action="append", default=[])
    ap.add_argument("--root", type=Path, action="append", default=[])
    ap.add_argument("--dataset")
    ap.add_argument("--route")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ffmpeg")
    ap.add_argument("--policy-seed", type=int, default=None,
                    help="模型种子：每局 trace 与 sidecar 的 policy_seed 必须等于它（判定行带 policy_seed=）")
    args = ap.parse_args(argv)
    defs = official_defs()
    if defs.canonical_dataset(args.dataset) != args.dataset or defs.canonical_route(args.route) != args.route:
        ap.error(f"--dataset／--route 只接受官方名（收到 dataset={args.dataset} route={args.route}）")
    ff = args.ffmpeg or ffmpeg_exe()
    if not ff:
        print("OFFICIAL_MEDIA=FAIL reason=no_ffmpeg", flush=True)
        return 2
    if args.verify_dir is not None:
        res = verify_dir(args.verify_dir, ff, strict_frames=False, expected_policy_seed=args.policy_seed)
        tag = {"pass": "PASS", "no_frame_error": "NO_FRAME"}.get(res["status"], "FAIL")
        print(f"OFFICIAL_VERIFY={tag} dir={res['dir']} frames={res.get('frames_decoded')} "
              f"expect={res.get('frames_expected')} frames_source={res.get('frames_source')} "
              f"sidecar={res.get('sidecar')} reasons={json.dumps(res['reasons'], ensure_ascii=False)}", flush=True)
        return 0 if tag in ("PASS", "NO_FRAME") else 1
    if not args.manifest or not args.ledger or not args.root or not args.dataset:
        ap.error("全量验收须给 --manifest、--ledger、--root、--dataset（各可重复）")
    counts, media, rows = check_all(args.manifest, args.ledger, args.root, args.dataset, args.route, ff,
                                    policy_seed=args.policy_seed)
    out = args.out or (args.root[0] / "official-media.jsonl")
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, out)
    inputs_ok = all(v == 0 for v in counts.values())
    media_ok = inputs_ok and media["skip"] == 0 and media["fail"] == 0
    print(f"OFFICIAL_MEDIA_INPUTS={'PASS' if inputs_ok else 'FAIL'} " + " ".join(f"{k}={v}" for k, v in counts.items()),
          flush=True)
    print(f"OFFICIAL_MEDIA={'PASS' if media_ok else 'FAIL'} total={media['total']} skip={media['skip']} "
          f"fail={media['fail']} no_frame_error={media['no_frame_error']} videos={media['passed']} "
          f"accepted={media['passed'] + media['no_frame_error']}"
          + (f" policy_seed={args.policy_seed}" if args.policy_seed is not None else "") + f" report={out}", flush=True)
    return 0 if media_ok else 1


if __name__ == "__main__":
    sys.exit(main())
