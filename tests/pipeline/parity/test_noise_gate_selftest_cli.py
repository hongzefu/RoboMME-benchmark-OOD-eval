"""C15.15 ``noise_gate.py`` 的内置自检（``selftest`` 子命令，GATE_SELFTEST）、CLI 拒跑口径，以及 ``compare_h5`` 的
其余结构分支（顶层数据集、非时间步键、组／数据集种类、只差帧数、打不开）。

自检是闸门本身的一部分：同代码一对必须 PASS、每个反例必须被抓到。这里除了跑通它，还把被测模块里的一个关键函数
临时换坏（测试进程内对 scripts/ 的 patch），要求自检随之 FAIL——证明它不是恒真。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import parity_fixtures as F


@pytest.fixture(scope="module")
def ng():
    return F.noise_gate()


def fields(line: str) -> dict[str, str]:
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


# ── selftest ─────────────────────────────────────────────────────────────


def test_selftest_cli_passes_every_case(ng, capsys):
    assert ng.main(["selftest"]) == 0
    out = capsys.readouterr().out.splitlines()
    f = fields(out[-1])
    assert f["GATE_SELFTEST"] == "PASS" and f["same_code"] == "PASS"
    good, total = map(int, f["cases_fail"].split("/"))
    assert good == total and total > 0
    case_lines = [t for t in out if t.startswith("# selftest ")]
    assert len(case_lines) > total  # 同代码类 + 反例类都逐条打印
    assert all("预期=PASS 实得=PASS" in t or "预期=FAIL 实得=FAIL" in t for t in case_lines)
    for name in ("改参照不改 sha256", "稳定局翻转须重跑", "噪声翻转超上限", "缺局判本遍无效", "两侧空文件（F-5）"):
        assert any(f"# selftest {name}:" in t for t in case_lines), name


def test_selftest_detects_broken_ref_check(ng, monkeypatch, capsys):
    """把 load_ref 换成不核顶层 sha256 的版本：「改参照不改 sha256」这一反例不再被抓到 → GATE_SELFTEST=FAIL、退出 1。"""
    real = ng.load_ref
    monkeypatch.setattr(ng, "load_ref", lambda path, **kw: json.loads(Path(path).read_text(encoding="utf-8"))
                        if "ref-bad" in str(path) else real(path, **kw))
    assert ng.main(["selftest"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert fields(out[-1])["GATE_SELFTEST"] == "FAIL"
    assert any(t.startswith("# selftest 改参照不改 sha256: 预期=FAIL 实得=PASS") for t in out)


def test_selftest_detects_broken_same_code_path(ng, monkeypatch):
    """把 gen_compare 的分类换成一律 structural：同代码一对不再通过 → same_code=FAIL。"""
    real = ng.classify_pair

    def broken(ident, ref, new, *a, **kw):
        r = real(ident, ref, new, *a, **kw)
        return dict(r, **{"class": "structural"})

    monkeypatch.setattr(ng, "classify_pair", broken)
    ok, line, _cases = ng.selftest(verbose=False)
    assert not ok and fields(line)["same_code"] == "FAIL"


# ── CLI 拒跑口径 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("runs", [["v9=only-one"], ["v9"], ["v9=a,b", "v9=c,d"]])
def test_build_ref_bad_runs_refused(ng, tmp_path, capsys, runs):
    argv = ["gen-regress", "build-ref", "--records", str(tmp_path), "--out", str(tmp_path / "ref.json")]
    for r in runs:
        argv += ["--runs", r]
    assert ng.main(argv) == 2
    out = capsys.readouterr().out
    assert "# 拒跑：--runs" in out and out.strip().endswith("NOISE_REF=FAIL reason=refused")


def test_build_ref_refuses_existing_out(ng, tmp_path, capsys):
    out = tmp_path / "ref.json"
    out.write_text("{}")
    assert ng.main(["gen-regress", "build-ref", "--runs", "v9=a,b", "--records", str(tmp_path), "--out", str(out)]) == 2
    assert "参照文件生成后只读" in capsys.readouterr().out and out.read_text() == "{}"


def test_check_with_bad_ref_refused(ng, tmp_path, capsys):
    bad = tmp_path / "ref.json"
    bad.write_text(json.dumps({"schema": "noise-ref/0"}))
    assert ng.main(["gen-regress", "check", "--ref", str(bad), "--set", "v9", "--new", str(tmp_path),
                    "--out", str(tmp_path / "o.jsonl")]) == 2
    assert capsys.readouterr().out.strip().endswith("GEN_REGRESS=FAIL reason=refused")


# ── compare_h5 的其余结构分支 ─────────────────────────────────────────────────


def _pair(tmp: Path, **right_kw) -> tuple[Path, Path]:
    return F.write_h5(tmp / "l.h5", seed=3, frames=4), F.write_h5(tmp / "r.h5", seed=3, **{"frames": 4, **right_kw})


def _edit(path: Path, fn) -> None:
    with h5py.File(path, "a") as f:
        fn(f)


def test_compare_h5_only_frame_count_differs_diverges_at_first_extra(ng, tmp_path):
    left, right = _pair(tmp_path, frames=6)
    # 夹具把「最后一帧」标 is_completed；右侧第 3 帧不是最后一帧，先当作分叉在第 3 帧（对照）
    assert ng.compare_h5(left, right)["first_divergence"] == 3

    def same_flag(f):
        del f["episode_3/timestep_3/info/is_completed"]
        f["episode_3/timestep_3/info/is_completed"] = np.bool_(True)

    _edit(right, same_flag)  # 公共 4 帧逐值相同后，分叉落在第一个多出来的帧
    r = ng.compare_h5(left, right)
    assert r["verdict"] == "diverge" and r["first_divergence"] == 4
    assert (r["frames_ref"], r["frames_new"], r["frames_diff"]) == (4, 6, 2)


@pytest.mark.parametrize("case,reason", [
    ("top_dataset", "top_dataset:meta"),
    ("non_ts_keys", "episode_3:non_timestep_keys"),
    ("kind", "episode_3/extra:kind"),
    ("ds_value", "episode_3/note:value"),
    ("group_value", "episode_3/setup/seed:value"),
])
def test_compare_h5_structural_branches(ng, tmp_path, case, reason):
    left, right = _pair(tmp_path)
    if case == "top_dataset":
        _edit(left, lambda f: f.create_dataset("meta", data=np.arange(3)))
        _edit(right, lambda f: f.create_dataset("meta", data=np.arange(3) + 1))
    elif case == "non_ts_keys":
        _edit(right, lambda f: f["episode_3"].create_group("extra"))
    elif case == "kind":
        _edit(left, lambda f: f["episode_3"].create_group("extra"))
        _edit(right, lambda f: f["episode_3"].create_dataset("extra", data=1))
    elif case == "ds_value":
        _edit(left, lambda f: f["episode_3"].create_dataset("note", data=1))
        _edit(right, lambda f: f["episode_3"].create_dataset("note", data=2))
    else:
        def bump(f):
            del f["episode_3/setup/seed"]
            f["episode_3/setup/seed"] = np.int64(4)
        _edit(right, bump)
    r = ng.compare_h5(left, right)
    assert (r["verdict"], r["reason"]) == ("structural", reason)


def test_compare_h5_equal_top_dataset_and_unreadable(ng, tmp_path):
    left, right = _pair(tmp_path, joint_offset_from=2)
    for p in (left, right):
        _edit(p, lambda f: f.create_dataset("meta", data=np.arange(3)))
    r = ng.compare_h5(left, right)
    assert r["verdict"] == "diverge" and r["first_divergence"] == 2  # 顶层数据集相同即跳过，照常逐帧比
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not an h5 file" * 10)
    r = ng.compare_h5(left, bad)
    assert r["verdict"] == "unknown" and r["reason"].startswith("open_error:")


def test_compare_h5_no_frames_is_structural(ng, tmp_path):
    def strip(f):
        for k in [k for k in f["episode_3"] if k.startswith("timestep_")]:
            del f["episode_3"][k]

    left, right = _pair(tmp_path)
    _edit(left, strip)
    _edit(right, strip)
    r = ng.compare_h5(left, right)
    assert (r["verdict"], r["reason"]) == ("structural", "episode_3:no_frame0")
