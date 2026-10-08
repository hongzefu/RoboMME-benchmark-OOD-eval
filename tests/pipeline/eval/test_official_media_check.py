"""S2b 官方版式视频验收 ``scripts/eval-official/official_media_check.py``（1005 计划第二部分一节「S2b」，审计第 9 条）。

夹具全部现做：局目录按 ``run_eval_gl.sh`` 发布布局 ``<root>/<key>.a<N>/``，``trace.jsonl`` 按共享契约 C6／C8 写
header 与 end（``frames_recorded``），``official/`` 里放 ffmpeg 现做的微型 mp4 与重绘器 ``render.json``（或 GroundSG
的 ``provenance.json``），账本按 ``env_client.AttemptLedger`` 的行格式写 ``attempt_start``／``accept``。

覆盖：全通过（含无帧 error 局计 ``no_frame_error`` 不计 ``fail``）；缺视频、缺目录、重复目录、两个视频、帧数不符、
调换两局视频（只换 mp4／连 provenance 一起换）、换成普通录像、错 attempt、账本接受的尝试目录不存在、等数量替换身份、
provenance 缺失、无帧局却有视频，均 FAIL；``--verify-dir`` 单局模式的三种退出与帧数回退。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts" / "eval-official" / "official_media_check.py"
DATASET = "hard-verify"
ROUTE = "pp/new"

if shutil.which("ffmpeg") is None:  # pragma: no cover
    pytest.skip("未验证：缺 ffmpeg", allow_module_level=True)


def _mp4(path: Path, frames: int, color: str = "red") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={color}:s=32x16:r=30", "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ident(k: int, attempt: int = 1) -> dict:
    seed = 510300 + k
    return {"task": "PickXtimes", "tier": "xhard0", "seed": seed, "source_episode": k, "builder_episode": k,
            "dataset": DATASET, "key": f"PickXtimes_xhard0_{seed}", "attempt": attempt}


def _manifest_row(k: int) -> dict:
    i = _ident(k)
    return {"task": i["task"], "tier": i["tier"], "seed": i["seed"], "candidate": None, "builder_episode": k,
            "source_episode": k, "spec_sha256": None, "key": i["key"]}


def _write_trace(d: Path, ident: dict, *, status="success", frames_recorded=4, no_frame=None):
    end = {"kind": "end", "status": status, "terminal_reason": status, "exec_steps": 2, "demo_frames": 1}
    if frames_recorded is not None:
        end["frames_recorded"] = frames_recorded
    if no_frame is not None:
        end["no_frame"] = no_frame
    rows = [{"kind": "header", "schema": "sgeval-trace/1", "route": ROUTE, "identity": ident, "max_steps": 1300}, end]
    (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _episode(root: Path, k: int, *, attempt: int = 1, frames: int = 4, color: str = "red", sidecar="render.json",
             trace_frames: int | None = None, status="success", ident: dict | None = None) -> Path:
    ident = ident or _ident(k, attempt)
    d = root / f"{ident['key']}.a{attempt}"
    d.mkdir(parents=True)
    _write_trace(d, ident, status=status, frames_recorded=frames if trace_frames is None else trace_frames)
    v = _mp4(d / "official" / f"official-rerender__{ident['task']}_ep{k}a{attempt}_{status}_goal_xhard0.mp4", frames, color)
    if sidecar == "render.json":
        side = {"schema": "official-rerender/1", "identity": ident, "route": ROUTE, "frames": frames,
                "output_fingerprint": {"sha256": _sha(v), "size": v.stat().st_size}}
    elif sidecar == "provenance.json":
        side = {"identity": ident, "route": ROUTE, "dataset": DATASET, "attempt": attempt, "frames_recorded": frames,
                "video_sha256": _sha(v), "official_source": "official"}
    else:
        side = None
    if side is not None:
        (d / "official" / sidecar).write_text(json.dumps(side), encoding="utf-8")
    return d


def _no_frame_episode(root: Path, k: int) -> Path:
    ident = _ident(k)
    d = root / f"{ident['key']}.a1"
    d.mkdir(parents=True)
    _write_trace(d, ident, status="error", frames_recorded=0, no_frame=True)
    (d / "official").mkdir()
    (d / "official" / "render.json").write_text(json.dumps({"render_status": "no_frame", "reason": "no_frame"}))
    return d


def _ledger(path: Path, accepts: dict[str, int]) -> Path:
    rows = []
    for key, n in accepts.items():
        for a in range(1, n + 1):
            rows.append({"kind": "attempt_start", "key": key, "attempt_id": f"{key}-{a}", "attempt_no": a, "retry": a > 1})
            rows.append({"kind": "attempt_end", "key": key, "attempt_id": f"{key}-{a}", "attempt_no": a,
                         "status": "success" if a == n else "error", "infra": a != n})
        rows.append({"kind": "accept", "key": key, "attempt_id": f"{key}-{n}", "attempt_no": n,
                     "accepted_attempt_id": f"{key}-{n}", "status": "success"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(*args) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _verdicts(out: str) -> dict[str, dict[str, str]]:
    res = {}
    for line in out.splitlines():
        if line.startswith(("OFFICIAL_MEDIA_INPUTS=", "OFFICIAL_MEDIA=")):
            head, *kv = line.split()
            name, val = head.split("=", 1)
            res[name] = {"": val, **dict(x.split("=", 1) for x in kv if "=" in x)}
    return res


@pytest.fixture
def world(tmp_path):
    """3 个正常局（render.json ×2、provenance.json ×1）+ 1 个无帧 error 局；第 2 局账本里是第 2 次尝试被接受。"""
    root = tmp_path / "media" / "pp" / DATASET / "new"
    root.mkdir(parents=True)
    eps = {0: _episode(root, 0, color="red"),
           1: _episode(root, 1, attempt=2, color="blue", sidecar="provenance.json"),
           2: _episode(root, 2, color="green"),
           3: _no_frame_episode(root, 3)}
    # 第 1 局的第 1 次尝试是 infra 错误（未接受），目录存在但不应被验收
    (root / f"{_ident(1)['key']}.a1").mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([_manifest_row(k) for k in range(4)]), encoding="utf-8")
    ledger = _ledger(tmp_path / "pp.ledger.jsonl", {_ident(k)["key"]: (2 if k == 1 else 1) for k in range(4)})
    return {"root": root, "eps": eps, "manifest": manifest, "ledger": ledger, "tmp": tmp_path}


def _check(w, *extra, manifest=None, ledger=None):
    rc, out = _run("--manifest", manifest or w["manifest"], "--ledger", ledger or w["ledger"], "--root", w["root"],
                   "--dataset", DATASET, "--route", ROUTE, *extra)
    return rc, out, _verdicts(out)


def test_all_pass_and_no_frame_error_not_fail(world):
    rc, out, v = _check(world)
    assert rc == 0, out
    assert v["OFFICIAL_MEDIA_INPUTS"] == {"": "PASS", "missing": "0", "extra": "0", "ambiguous": "0",
                                          "identity_mismatch": "0", "attempt_mismatch": "0", "provenance_missing": "0"}
    m = v["OFFICIAL_MEDIA"]
    assert (m[""], m["total"], m["skip"], m["fail"], m["no_frame_error"]) == ("PASS", "4", "0", "0", "1")
    rows = [json.loads(x) for x in (world["root"] / "official-media.jsonl").read_text().splitlines()]
    by = {r["key"]: r for r in rows}
    assert by[_ident(3)["key"]]["status"] == "no_frame_error"
    assert by[_ident(1)["key"]]["dir"] == f"{_ident(1)['key']}.a2" and by[_ident(1)["key"]]["attempt"] == 2
    assert all(by[_ident(k)["key"]]["frames_decoded"] == 4 for k in (0, 1, 2))


def _mutate_and_expect(world, mutate, *, inputs=None, fail=None, skip=None):
    mutate(world)
    rc, out, v = _check(world)
    assert rc == 1, out
    assert v["OFFICIAL_MEDIA"][""] == "FAIL", out
    for k, want in (inputs or {}).items():
        assert v["OFFICIAL_MEDIA_INPUTS"][k] == str(want), out
    if inputs:
        assert v["OFFICIAL_MEDIA_INPUTS"][""] == "FAIL"
    if fail is not None:
        assert v["OFFICIAL_MEDIA"]["fail"] == str(fail), out
    if skip is not None:
        assert v["OFFICIAL_MEDIA"]["skip"] == str(skip), out
    assert v["OFFICIAL_MEDIA"]["no_frame_error"] in ("0", "1")
    return out


def _off(w, k):
    return w["eps"][k] / "official"


def _video(w, k):
    (v,) = [p for p in _off(w, k).glob("*.mp4")]
    return v


def test_missing_video_fails(world):
    _mutate_and_expect(world, lambda w: _video(w, 0).unlink(), fail=1)


def test_missing_dir_counts_missing(world):
    _mutate_and_expect(world, lambda w: shutil.rmtree(w["eps"][2]), inputs={"missing": 1}, skip=1)


def test_duplicate_published_dir_is_ambiguous(world):
    def dup(w):
        shutil.copytree(w["eps"][0], w["root"] / (w["eps"][0].name + ".dup1"))
    _mutate_and_expect(world, dup, inputs={"ambiguous": 1}, skip=1)


def test_two_videos_fail(world):
    _mutate_and_expect(world, lambda w: shutil.copy2(_video(w, 0), _off(w, 0) / "second.mp4"), fail=1)


def test_frame_count_mismatch_fails(world):
    def short(w):
        v = _video(w, 2)
        _mp4(v, 3, "green")  # 同名重写成 3 帧；trace frames_recorded 仍为 4
    out = _mutate_and_expect(world, short, fail=1)
    assert "frames_mismatch" in (world["root"] / "official-media.jsonl").read_text()


def test_swapped_videos_fail(world):
    def swap(w):
        a, b = _video(w, 0), _video(w, 2)
        tmp = a.with_name("tmp.bin")
        a.rename(tmp)
        shutil.move(b, a.parent / b.name)
        shutil.move(tmp, b.parent / a.name)
    _mutate_and_expect(world, swap, fail=2)


def test_swapped_videos_with_provenance_fail(world):
    def swap(w):
        a, b = _off(w, 0), _off(w, 2)
        tmp = a.with_name("official.tmp")
        a.rename(tmp)
        b.rename(a)
        tmp.rename(b)
    out = _mutate_and_expect(world, swap, fail=2)
    assert "provenance_identity_mismatch" in (world["root"] / "official-media.jsonl").read_text()


def test_plain_recording_instead_of_official_fails(world):
    def plain(w):
        v = _video(w, 0)
        plain = _mp4(w["tmp"] / "episode.mp4", 4, "white")  # 普通录像：帧数恰好相同也不行
        shutil.copy2(plain, v)
    _mutate_and_expect(world, plain, fail=1)
    assert "provenance_video_sha_mismatch" in (world["root"] / "official-media.jsonl").read_text()


def test_wrong_attempt_fails(world):
    def wrong(w):
        d = w["eps"][2]
        ident = _ident(2, attempt=2)
        _write_trace(d, ident)  # .a1 目录里的 trace 自称第 2 次尝试；账本接受的是第 1 次
        side = json.loads((d / "official" / "render.json").read_text())
        side["identity"] = ident
        (d / "official" / "render.json").write_text(json.dumps(side))
    _mutate_and_expect(world, wrong, inputs={"attempt_mismatch": 1}, fail=1)


def test_ledger_accepts_attempt_without_dir(world):
    led = _ledger(world["tmp"] / "other.ledger.jsonl", {_ident(k)["key"]: (2 if k in (1, 2) else 1) for k in range(4)})
    rc, out, v = _check(world, ledger=led)
    assert rc == 1 and v["OFFICIAL_MEDIA_INPUTS"]["missing"] == "1", out


def test_equal_count_identity_substitution_fails(world):
    rows = [_manifest_row(k) for k in (0, 1, 2)] + [_manifest_row(9)]  # 第 3 局被换成清单外的同数量身份
    m = world["tmp"] / "swapped-manifest.json"
    m.write_text(json.dumps(rows), encoding="utf-8")
    rc, out, v = _check(world, manifest=m)
    assert rc == 1, out
    assert v["OFFICIAL_MEDIA_INPUTS"]["missing"] == "1" and v["OFFICIAL_MEDIA_INPUTS"]["extra"] == "1", out
    assert v["OFFICIAL_MEDIA"]["total"] == "4"


def test_provenance_missing(world):
    _mutate_and_expect(world, lambda w: (_off(w, 0) / "render.json").unlink(), inputs={"provenance_missing": 1}, fail=1)


def test_no_frame_episode_with_video_fails(world):
    _mutate_and_expect(world, lambda w: _mp4(_off(w, 3) / "x.mp4", 2), fail=1)


def test_conflicting_ledgers_ambiguous(world):
    led2 = _ledger(world["tmp"] / "seat2.ledger.jsonl", {_ident(0)["key"]: 2})
    rc, out = _run("--manifest", world["manifest"], "--ledger", world["ledger"], "--ledger", led2,
                   "--root", world["root"], "--dataset", DATASET)
    v = _verdicts(out)
    assert rc == 1 and v["OFFICIAL_MEDIA_INPUTS"]["ambiguous"] == "1", out


def test_verify_dir_modes(world):
    rc, out = _run("--verify-dir", world["eps"][0])
    assert rc == 0 and "OFFICIAL_VERIFY=PASS" in out and "frames_source=trace" in out, out
    rc, out = _run("--verify-dir", world["eps"][3])
    assert rc == 0 and "OFFICIAL_VERIFY=NO_FRAME" in out, out
    # trace 尚无 frames_recorded：单局模式以 provenance 帧数为期望；全量模式必须拒绝
    d = world["eps"][2]
    _write_trace(d, _ident(2), frames_recorded=None)
    rc, out = _run("--verify-dir", d)
    assert rc == 0 and "frames_source=sidecar" in out, out
    rc, out, v = _check(world)
    assert rc == 1 and v["OFFICIAL_MEDIA"]["fail"] == "1", out
    # 截断的视频不能算可解码
    v0 = _video(world, 0)
    v0.write_bytes(v0.read_bytes()[: v0.stat().st_size // 2])
    rc, out = _run("--verify-dir", world["eps"][0])
    assert rc == 1 and "OFFICIAL_VERIFY=FAIL" in out, out


def test_sidecar_frames_accepts_groundsg_provenance_dict():
    """GroundSG 官方原生视频的 provenance.json（S1）把 frames 写成字典：三值一致取 decoded，不一致或缺计数给失败说明。
    （2026-10-06 批次 2 本机 smoke 实测：字典被当整数比较，原生视频被误判不合格后改走重绘。）"""
    from tests._support.loaders import load_script

    m = load_script("eval-official/official_media_check.py")
    assert m.sidecar_frames({"frames": 392}) == 392
    assert m.sidecar_frames({"frames_recorded": 7, "frames": 9}) == 7
    ok = {"frames": {"basis": "x", "decoded": 392, "expected": 392, "frames_recorded": 392, "demo_frames": 66}}
    assert m.sidecar_frames(ok) == 392
    bad = {"frames": {"decoded": 391, "expected": 392, "frames_recorded": 392}}
    assert isinstance(m.sidecar_frames(bad), str) and "inconsistent" in m.sidecar_frames(bad)
    assert m.sidecar_frames({"frames": {"basis": "x"}}) == "frames_dict_without_counts"


# ── 1006 第三阶段：accepted = videos + no_frame_error、strict-cap 命名、模型种子 ─────────────────


def test_accepted_equals_videos_plus_no_frame(world):
    rc, out, v = _check(world)
    print(next(x for x in out.splitlines() if x.startswith("OFFICIAL_MEDIA=")))
    m = v["OFFICIAL_MEDIA"]
    assert rc == 0 and (m["videos"], m["no_frame_error"], m["accepted"], m["total"]) == ("3", "1", "4", "4"), out


def test_all_86_identities_no_frame_reported_separately(tmp_path):
    """86 个身份全是无帧 error：按用户裁决保留例外、单列，videos=0、no_frame_error=86、accepted=86，不计 fail。"""
    root = tmp_path / "media"
    root.mkdir()
    for k in range(86):
        _no_frame_episode(root, k)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([_manifest_row(k) for k in range(86)]), encoding="utf-8")
    ledger = _ledger(tmp_path / "pp.ledger.jsonl", {_ident(k)["key"]: 1 for k in range(86)})
    rc, out = _run("--manifest", manifest, "--ledger", ledger, "--root", root, "--dataset", DATASET, "--route", ROUTE)
    line = next(x for x in out.splitlines() if x.startswith("OFFICIAL_MEDIA="))
    print(line)
    m = _verdicts(out)["OFFICIAL_MEDIA"]
    assert rc == 0 and (m[""], m["total"], m["fail"], m["videos"], m["no_frame_error"], m["accepted"]) == \
        ("PASS", "86", "0", "0", "86", "86"), out


def test_strict_cap_error_name_rejected(world):
    """strict-cap 命中（end.cap_hit）或 status=timeout 的局视频名带 _error_：旧口径，计 terminal_name 失败。"""
    def old_name(w):
        d = w["eps"][0]
        ident = _ident(0)
        rows = [{"kind": "header", "schema": "sgeval-trace/1", "route": ROUTE, "identity": ident, "max_steps": 1800},
                {"kind": "end", "status": "timeout", "terminal_reason": "error", "cap_hit": True, "exec_steps": 2,
                 "demo_frames": 1, "frames_recorded": 4}]
        (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        v = _video(w, 0)
        v.rename(v.with_name(f"official-rerender__PickXtimes_ep0a1_error_goal_xhard0.mp4"))
    out = _mutate_and_expect(world, old_name, fail=1)
    rows = [json.loads(x) for x in (world["root"] / "official-media.jsonl").read_text().splitlines()]
    bad = next(r for r in rows if r["key"] == _ident(0)["key"])
    assert any(r.startswith("terminal_name:error_named") for r in bad["reasons"]), out


def test_policy_seed_checked(world):
    rc, out, v = _check(world, "--policy-seed", "7")
    # 4 个局（含无帧 error 局：trace header 同样要有种子）都缺 policy_seed → 全部 fail
    assert rc == 1 and v["OFFICIAL_MEDIA"]["fail"] == "4" and v["OFFICIAL_MEDIA"]["policy_seed"] == "7", out
    for k in (0, 1, 2, 3):  # trace 补上种子 7 后通过；sidecar 写成 42 则拒
        d = world["eps"][k]
        rows = [json.loads(x) for x in (d / "trace.jsonl").read_text().splitlines()]
        rows[0]["policy_seed"] = 7
        (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rc, out, v = _check(world, "--policy-seed", "7")
    assert rc == 0 and v["OFFICIAL_MEDIA"][""] == "PASS", out
    side = _off(world, 0) / "render.json"
    s = json.loads(side.read_text())
    s["policy_seed"] = 42
    side.write_text(json.dumps(s))
    rc, out, v = _check(world, "--policy-seed", "7")
    assert rc == 1 and v["OFFICIAL_MEDIA"]["fail"] == "1", out
