"""C15 边生成边判定的贯通：``hard_parity.Mover.handle`` → ``noise_run.py ship --finalize`` → ``hard_pull.py``。

真实生产方法、真实文件：节点输出根里放微型 h5／mp4，``Mover.handle`` 当场算 sha、对噪声基线参照判 verdict；
``match`` 删节点大文件不复制，``jitter_info``／``flip`` 复制到暂存并写 ``SHIPPED``，``flip`` 追加 ``flips.jsonl``
并打印 ``EPISODE_FLIP``；``ship --finalize`` 认 ``verdict=match`` 计入 matched；``hard_pull`` 带／不带 ``--identities``、
空清单、旧格式段（行无 verdict）。不给 ``--expect-ref`` 时 Mover 的行与搬运与原先相同。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

import parity_fixtures as F

LAYOUT = [("PickXtimes", "xhard1", 1, "stable"), ("BinFill", "xhard2", 2, "jitter"),
          ("PickXtimes", "xhard1", 3, "stable"), ("VideoPlaceOrder", "xhard2", 4, "known_fail")]
OUTSIDE = ("StopCube", "xhard3", 5)  # 参照里没有的身份


@pytest.fixture(scope="module")
def mods():
    return F.noise_gate(), F.hard_parity(), F.noise_run(), F.hard_pull()


@pytest.fixture()
def ref(tmp_path, mods):
    ng = mods[0]
    b = F.build_baseline(tmp_path / "base", {"v9": LAYOUT})
    assert ng.main(b["build_argv"]) == 0
    return b


def node_episode(out: Path, task: str, tier: str, ep: int, h5_bytes: bytes) -> Path:
    wdir = out / "episodes" / tier / f"{task}_episode_{ep}"
    (wdir / "hdf5_files").mkdir(parents=True, exist_ok=True)
    (wdir / "hdf5_files" / f"{task}_ep{ep}.h5").write_bytes(h5_bytes)
    (wdir / "videos").mkdir(exist_ok=True)
    (wdir / "videos" / f"{task}_ep{ep}.mp4").write_bytes(b"fake-mp4-" + task.encode() + bytes([ep]))
    (wdir / "sidecar.json").write_text("{}", encoding="utf-8")
    return wdir


def record(task, tier, ep, seed, ok=True, error_type=None):
    return {"task": task, "episode": ep, "seed": seed, "difficulty": tier, "ok": ok, "error_type": error_type}


def run_mover(hp, ref_b, out: Path, stage: Path | None, *, expect: bool) -> list[dict]:
    """五局：1 稳定复现、2 抖动、3 稳定翻转（字节不同）、4 确定性失败复现、5 参照外身份。"""
    h5 = ref_b["h5"]
    a_root = ref_b["runs"]["v9"][0]
    other = F.write_h5(out.parent / "other.h5", seed=3, joint_offset_from=1).read_bytes()
    placeholder = (a_root / "episodes/xhard2/VideoPlaceOrder_episode_4/hdf5_files/VideoPlaceOrder_seed4.h5").read_bytes()
    node_episode(out, "PickXtimes", "xhard1", 1, h5[("v9", 1)].read_bytes())
    node_episode(out, "BinFill", "xhard2", 2, F.write_h5(out.parent / "j.h5", seed=2, frames=5).read_bytes())
    node_episode(out, "PickXtimes", "xhard1", 3, other)
    node_episode(out, "VideoPlaceOrder", "xhard2", 4, placeholder)
    node_episode(out, *OUTSIDE[:2], 5, F.write_h5(out.parent / "o.h5", seed=5).read_bytes())
    mover = hp.Mover(out, stage, "H2", "v9", {"worker": "w"}, expect_ref=ref_b["ref"] if expect else None)
    for rec in (record("PickXtimes", "xhard1", 1, 1), record("BinFill", "xhard2", 2, 2),
                record("PickXtimes", "xhard1", 3, 3),
                record("VideoPlaceOrder", "xhard2", 4, 4, ok=False, error_type="DatasetGenerationError"),
                record(*OUTSIDE[:2], 5, 5)):
        mover.handle(rec)
        mover.handle(rec)  # 同一局重复出现在 partial 里只处理一次
    assert mover.errors == []
    return [json.loads(t) for t in (out / "identities.jsonl").read_text(encoding="utf-8").splitlines()]


def media(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".h5", ".mp4"))


def test_Mover_expect_ref_逐局判定与搬运(mods, ref, tmp_path, capsys):
    _ng, hp, _nr, _hpull = mods
    out, stage = tmp_path / "node", tmp_path / "stage" / "seg1"
    lines = run_mover(hp, ref, out, stage, expect=True)
    by = {x["seed"]: x for x in lines}
    assert len(lines) == 5
    assert {s: by[s]["verdict"] for s in by} == {1: "match", 2: "jitter_info", 3: "flip", 4: "match", 5: "flip"}
    assert {s: by[s]["ref_class"] for s in by} == {1: "stable", 2: "jitter", 3: "stable", 4: "known_fail", 5: None}
    assert by[5]["verdict_note"] == "not_in_ref" and "verdict_note" not in by[3]
    assert by[4]["success"] is False and by[4]["h5_error"]  # 占位文件打不开，失败局如实记
    # match：节点大文件已删、暂存里没有这局；其余三局有 SHIPPED 且 h5 sha 与 identities 相同
    assert media(out) == []
    for s in (1, 4):
        assert not (stage / Path(by[s]["path"]).parent.parent).exists()
    for s in (2, 3, 5):
        ep = stage / Path(by[s]["path"]).parent.parent
        shipped = json.loads((ep / "SHIPPED").read_text(encoding="utf-8"))
        assert shipped[str(Path(by[s]["path"]).relative_to(Path(by[s]["path"]).parent.parent))] == by[s]["sha256"]
        assert F.sha256(stage / by[s]["path"]) == by[s]["sha256"]
    flips = [json.loads(t) for t in (out / "flips.jsonl").read_text(encoding="utf-8").splitlines()]
    assert flips == [{"task": "PickXtimes", "tier": "xhard1", "seed": 3},
                     {"task": OUTSIDE[0], "tier": OUTSIDE[1], "seed": 5}]
    text = capsys.readouterr().out
    assert text.count("EPISODE_FLIP ") == 2 and "EPISODE_FLIP xhard1/PickXtimes/3" in text
    assert text.count("EPISODE_DONE ") == 5 and "verdict=match" in text


def test_Mover_不给expect_ref时行为不变(mods, ref, tmp_path, capsys):
    _ng, hp, _nr, _hpull = mods
    out, stage = tmp_path / "node", tmp_path / "stage" / "seg1"
    lines = run_mover(hp, ref, out, stage, expect=False)
    assert all("verdict" not in x and "ref_class" not in x for x in lines)
    assert not (out / "flips.jsonl").exists() and media(out) == []
    assert sum(1 for _ in stage.rglob("SHIPPED")) == 5  # 每局都复制（含确定性失败局的占位文件）
    text = capsys.readouterr().out
    assert "EPISODE_FLIP" not in text and "verdict=" not in text


def ship(nr, src, stage, capsys, *extra) -> tuple[int, str]:
    rc = nr.main(["ship", "--src", str(src), "--stage", str(stage), *extra])
    return rc, capsys.readouterr().out.strip().splitlines()[-1]


def test_ship_finalize_认match局为按设计不复制(mods, ref, tmp_path, capsys):
    _ng, hp, nr, _hpull = mods
    out, stage = tmp_path / "node", tmp_path / "stage" / "seg1"
    run_mover(hp, ref, out, stage, expect=True)
    capsys.readouterr()
    rc, line = ship(nr, out, stage, capsys, "--finalize")
    assert rc == 0, line
    assert line == ("NOISE_SHIP=PASS episodes=5 identities=5 shipped=3 matched=2 pulled=0 sha_bad=0 missing=0 "
                    "node_leftover_media=0 finalized=1")
    assert (stage / "SEGMENT_DONE").is_file() and (stage / "identities.jsonl").is_file()
    assert (stage / "flips.jsonl").is_file()


def test_ship_负例_非match局缺SHIPPED_节点残留(mods, ref, tmp_path, capsys):
    _ng, hp, nr, _hpull = mods
    out, stage = tmp_path / "node", tmp_path / "stage" / "seg1"
    lines = run_mover(hp, ref, out, stage, expect=True)
    capsys.readouterr()
    flip = next(x for x in lines if x["seed"] == 3)
    (stage / Path(flip["path"]).parent.parent / "SHIPPED").unlink()
    rc, line = ship(nr, out, stage, capsys)
    assert rc == 1 and " missing=1 " in line and line.startswith("NOISE_SHIP=FAIL")
    lines_m = [x for x in lines if x["verdict"] == "match"]
    leftover = out / Path(lines_m[0]["path"])
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"x")
    (stage / Path(flip["path"]).parent.parent / "SHIPPED").write_text(json.dumps({}), encoding="utf-8")
    rc, line = ship(nr, out, stage, capsys)
    assert rc == 1 and "node_leftover_media=1" in line
    assert not (stage / "SEGMENT_DONE").exists()


def pull(hpull, monkeypatch, capsys, stage_root, dest, identities=None) -> tuple[int, list[str]]:
    argv = ["hard_pull.py", "--stage", str(stage_root), "--dest", str(dest), "--segments", "seg1",
            "--interval", "0"]
    if identities is not None:
        argv += ["--identities", str(identities)]
    monkeypatch.setattr(sys, "argv", argv)
    rc = hpull.main()
    return rc, capsys.readouterr().out.strip().splitlines()


@pytest.fixture()
def segment(mods, ref, tmp_path, capsys):
    """跑完 Mover（带 --expect-ref）并 ship --finalize 的一段。"""
    _ng, hp, nr, _hpull = mods
    out, stage = tmp_path / "node", tmp_path / "stage" / "seg1"
    lines = run_mover(hp, ref, out, stage, expect=True)
    capsys.readouterr()
    assert nr.main(["ship", "--src", str(out), "--stage", str(stage), "--finalize"]) == 0
    capsys.readouterr()
    return {"out": out, "stage": stage, "lines": {x["seed"]: x for x in lines}}


def test_pull_只拉翻转清单内的局(mods, segment, tmp_path, monkeypatch, capsys):
    hpull = mods[3]
    dest = tmp_path / "dest"
    rc, text = pull(hpull, monkeypatch, capsys, segment["stage"].parent, dest, segment["out"] / "flips.jsonl")
    assert rc == 0, text
    seg_line = next(t for t in text if t.startswith("PULL_SEGMENT="))
    assert seg_line == ("PULL_SEGMENT=PASS segment=seg1 identities=5 h5=5 sha_bad=0 nfs_leftover_media=0 "
                        "selected=2 absent=0 not_shipped=0 not_selected=1 matched=2")
    assert text[-1].startswith("PULL_DONE=PASS")
    lines = segment["lines"]
    for s in (3, 5):
        assert F.sha256(dest / "seg1" / lines[s]["path"]) == lines[s]["sha256"]
        assert not (segment["stage"] / lines[s]["path"]).exists()  # 拉回核对后删 NFS 副本
    assert not (dest / "seg1" / lines[2]["path"]).exists()        # 抖动局不在清单：不拉不删
    assert (segment["stage"] / lines[2]["path"]).exists()
    for name in ("identities.jsonl", "flips.jsonl", "SEGMENT_DONE"):
        assert (dest / "seg1" / name).is_file()


def test_pull_空清单只拉报告(mods, segment, tmp_path, monkeypatch, capsys):
    hpull = mods[3]
    empty = F.write_jsonl(tmp_path / "empty.jsonl", [])
    dest = tmp_path / "dest"
    rc, text = pull(hpull, monkeypatch, capsys, segment["stage"].parent, dest, empty)
    assert rc == 0
    assert any(t.startswith("PULL_SEGMENT=PASS ") and "selected=0" in t and "matched=2" in t for t in text)
    assert media(dest) == [] and (dest / "seg1" / "identities.jsonl").is_file()


def test_pull_清单内但未复制的局只计数(mods, segment, tmp_path, monkeypatch, capsys):
    """同一份清单跨首跑与重跑段使用：清单里 match 局（未复制）计 not_shipped，段里没有的身份计 absent。"""
    hpull = mods[3]
    lst = F.write_jsonl(tmp_path / "ids.jsonl", [{"task": "PickXtimes", "seed": 1}, {"task": "Nope", "seed": 9}])
    rc, text = pull(hpull, monkeypatch, capsys, segment["stage"].parent, tmp_path / "dest", lst)
    assert rc == 0
    assert any("selected=0 absent=1 not_shipped=1 not_selected=3" in t for t in text)


def test_pull_不给清单拉全部已复制局(mods, segment, tmp_path, monkeypatch, capsys):
    hpull = mods[3]
    dest = tmp_path / "dest"
    rc, text = pull(hpull, monkeypatch, capsys, segment["stage"].parent, dest)
    assert rc == 0
    seg_line = next(t for t in text if t.startswith("PULL_SEGMENT="))
    assert seg_line == ("PULL_SEGMENT=PASS segment=seg1 identities=5 h5=5 sha_bad=0 nfs_leftover_media=0 matched=2")
    assert sum(1 for t in text if t.startswith("PULLED ")) == 3
    assert media(segment["stage"]) == []


def test_pull_旧格式段行为不变(mods, ref, tmp_path, monkeypatch, capsys):
    _ng, hp, nr, hpull = mods
    out, stage = tmp_path / "node", tmp_path / "stage" / "seg1"
    run_mover(hp, ref, out, stage, expect=False)
    assert nr.main(["ship", "--src", str(out), "--stage", str(stage), "--finalize"]) == 0
    capsys.readouterr()
    rc, text = pull(hpull, monkeypatch, capsys, stage.parent, tmp_path / "dest")
    seg_line = next(t for t in text if t.startswith("PULL_SEGMENT="))
    assert rc == 0 and seg_line == "PULL_SEGMENT=PASS segment=seg1 identities=5 h5=5 sha_bad=0 nfs_leftover_media=0"


def test_pull_拉回字节被改即FAIL(mods, segment, tmp_path, monkeypatch, capsys):
    hpull = mods[3]
    lines = segment["lines"]
    shipped_h5 = segment["stage"] / lines[3]["path"]
    shipped_h5.write_bytes(b"corrupted")
    rc, text = pull(hpull, monkeypatch, capsys, segment["stage"].parent, tmp_path / "dest",
                    segment["out"] / "flips.jsonl")
    assert rc == 1 and any(t.startswith("PULL_SEGMENT=FAIL") for t in text)
    assert any(t.startswith("PULLED seg1/") and "ok=0" in t for t in text)


# ── generate --expect-ref 的前置校验（不起运行器、不查 GPU）─────────────────────────────


@pytest.mark.parametrize("kind", ["partial", "tampered", "missing"])
def test_generate_expect_ref参照不可用_建目录前拒跑(mods, ref, tmp_path, kind):
    ng, hp, _nr, _hpull = mods
    bad = tmp_path / "bad-ref.json"
    if kind == "partial":
        shutil.copyfile(ref["ref"], bad)  # 拷贝正式参照后改成 partial 并按契约重签
        obj = json.loads(bad.read_text(encoding="utf-8"))
        obj["partial"] = True
        body = {k: v for k, v in obj.items() if k != "sha256"}
        obj["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                                  separators=(",", ":")).encode()).hexdigest()
        bad.write_text(json.dumps(obj), encoding="utf-8")
    elif kind == "tampered":
        obj = json.loads(ref["ref"].read_text(encoding="utf-8"))
        obj["sets"]["v9"]["episodes"][0]["class"] = "known_fail"
        bad.write_text(json.dumps(obj), encoding="utf-8")
    out = tmp_path / "gen-out"
    args = hp.build_parser().parse_args(["generate", "--side", "H2", "--tier", "v9", "--manifest", str(tmp_path / "m"),
                                         "--src-root", str(tmp_path), "--out", str(out), "--expect-ref", str(bad)])
    with pytest.raises(hp.ParityError, match="--expect-ref"):
        args.func(args)
    assert not out.exists()
