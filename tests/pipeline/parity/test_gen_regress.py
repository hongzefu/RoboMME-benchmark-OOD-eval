"""C15 逐局回归闸门 ``noise_gate.py gen-regress``（对拍细则 3.2～3.4、G 块）。

- ``build-ref``：从小基线（a、b 两遍 + 真实 gen-compare 比对记录）生成参照；三类归属、canonical sha、
  交叉核对规则的每条反例、partial 拒收、输出已存在拒写。
- ``check``：首跑逐局期望 match／jitter_info／flip；INVALID 前提；原因不明与结构不同直接 FAIL；
  NEED_RERUN 的陪跑局挑选；四格定性（噪声／回归／环境变了／每次都不同）与噪声上限；翻转上限；逐局报告。
- 真实参照 ``scripts/configs/noise-ref-20261003.json`` 只读核对：canonical sha 自洽、三类计数自洽、
  身份集合与 gate-set 冻结清单相同（计数不写字面值，红线 R8）。

期望全部由用例里的布局表手写得出；首跑／重跑的「新跑」只写 identities 行（check 不重读 h5），
细分翻转类别（--local-ref-root）的用例才写真实微型 h5。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import parity_fixtures as F

KNOWN_FAIL_TASK = "VideoPlaceOrder"


def canonical_sha(obj: dict) -> str:
    """契约（细则 3.2）：顶层 sha256 = 剔掉该键后 canonical JSON（sort_keys、紧凑分隔、不转义非 ASCII）的 sha256。"""
    body = {k: v for k, v in obj.items() if k != "sha256"}
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def v9_layout(n_big: int) -> list[tuple[str, str, int, str]]:
    """档序 xhard1 → xhard2 → xhard3 → xhard4；xhard2 里放一个抖动局与一个确定性失败局。"""
    return ([("PickXtimes", "xhard1", s, "stable") for s in (10, 11, 12)]
            + [("BinFill", "xhard2", 20, "stable"), ("BinFill", "xhard2", 21, "stable"),
               ("BinFill", "xhard2", 22, "jitter"), ("BinFill", "xhard2", 23, "known_fail")]
            + [("StopCube", "xhard3", s, "stable") for s in (30, 31, 32)]
            + [("MoveCube", "xhard4", 40 + i, "stable") for i in range(n_big)])


def x0_layout(n_stable: int) -> list[tuple[str, str, int, str]]:
    return ([(KNOWN_FAIL_TASK, "xhard0", 610701, "known_fail")]
            + [("PickXtimes", "xhard0", 510300 + 100 * i, "stable") for i in range(n_stable)])


@pytest.fixture(scope="module")
def ng():
    return F.noise_gate()


@pytest.fixture(scope="module")
def base(tmp_path_factory, ng):
    """模块级小基线：v9 与 xhard0 两个集合（规模按闸门常量留足余量，不写字面值）。"""
    root = tmp_path_factory.mktemp("noise-base")
    layouts = {"v9": v9_layout(ng.FLIP_RERUN_MAX + 2), "xhard0": x0_layout(max(ng.NOISE_MAX.values()) + 3)}
    b = F.build_baseline(root, layouts)
    assert ng.main(b["build_argv"]) == 0
    return b


def valid_launch(ng) -> dict:
    return {"workers": ng.PRECOND_WORKERS, "gpu": f"NVIDIA {ng.PRECOND_GPU}"}


def run_check(ng, base, tmp_path, set_name, first_lines, *, rerun=None, local=False, first_launch=None,
              host="gl1001"):
    """first_lines → 首跑根；rerun=(new2 行, old 行, {new2/old 的 host 覆盖}) → 两个重跑根。"""
    kw = dict(valid_launch(ng), **(first_launch or {}))
    first = F.write_run(tmp_path / f"first-{set_name}", first_lines, host=host, **{k: v for k, v in kw.items()
                                                                                    if k != "launch"},
                        launch=kw.get("launch", True))
    args = {}
    if rerun is not None:
        new2_lines, old_lines, hosts = rerun
        args["rerun_new"] = F.write_run(tmp_path / f"new2-{set_name}", new2_lines, host=hosts.get("new2", host),
                                        **valid_launch(ng))
        args["rerun_old"] = F.write_run(tmp_path / f"old-{set_name}", old_lines, host=hosts.get("old", host),
                                        **valid_launch(ng))
    if local:
        args["local_ref_root"] = base["gen_root"]
    return ng.regress_check(ng.load_ref(base["ref"]), set_name, first, **args)


def layout_of(base, set_name):
    return base["layouts"][set_name]


def seeds_of(base, set_name, kind):
    return [s for _t, _tier, s, k in layout_of(base, set_name) if k == kind]


def entry(base, set_name, seed):
    return next(e for e in F.ref_episodes(base["ref"], set_name) if e["seed"] == seed)


def rerun_lines(base, set_name, seeds, override: dict[int, dict] | None = None):
    """重跑根的行：只含给定身份；默认复现基线（与 first_run_lines 同口径），override 按 seed 改。"""
    return F.first_run_lines(base["ref"], set_name, overrides=override or {}, only=seeds)


FAKE = {k: (c * 64) for k, c in (("x", "e"), ("y", "d"), ("z", "c"))}


# ══ expect_verdict：逐局期望的真值表（Mover 与 check 共用的唯一判定函数）══════════════════════


@pytest.mark.parametrize("cls,ok,sha,want", [
    ("stable", True, "s1", "match"),
    ("stable", True, "s9", "flip"),          # 成功但字节不同
    ("stable", False, "s1", "flip"),         # 稳定局失败（即使 sha 恰好相同）
    ("known_fail", False, "s1", "match"),    # 仍失败且占位字节相同
    ("known_fail", False, "s9", "flip"),     # 失败但占位字节不同
    ("known_fail", True, "s1", "flip"),      # 确定性失败局变成功
    ("jitter", True, "s9", "jitter_info"),
    ("jitter", False, None, "jitter_info"),
])
def test_expect_verdict真值表(ng, cls, ok, sha, want):
    assert ng.expect_verdict({"class": cls, "shas": ["s1", "s2"]}, ok, sha) == want


def test_expect_verdict_参照外身份判翻转(ng):
    assert ng.expect_verdict(None, True, "s1") == "flip"


# ══ build-ref ══════════════════════════════════════════════════════════════════════════


SMALL = [("PickXtimes", "xhard1", 1, "stable"), ("PickXtimes", "xhard1", 2, "stable"),
         ("BinFill", "xhard2", 3, "jitter"), ("BinFill", "xhard2", 4, "known_fail"),
         ("StopCube", "xhard2", 5, "stable")]


@pytest.fixture()
def small(tmp_path):
    return F.build_baseline(tmp_path, {"v9": SMALL})


def test_build_ref_三类归属与参照自洽(ng, small, capsys):
    assert ng.main(small["build_argv"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].startswith("NOISE_REF=PASS sets=1 v9=stable:3,known_fail:1,jitter:1 ")
    obj = json.loads(small["ref"].read_text(encoding="utf-8"))
    assert obj["sha256"] == canonical_sha(obj)
    assert obj["partial"] is False and obj["rehash"] is True and obj["limit"] is None
    block = obj["sets"]["v9"]
    assert block["counts"] == {"stable": 3, "known_fail": 1, "jitter": 1} and block["n"] == len(SMALL)
    assert block["identities"]["sha256"] == F.sha256(small["ids"]["v9"])
    a_root, b_root = small["runs"]["v9"]
    by_seed = {e["seed"]: e for e in block["episodes"]}
    assert [e["seed"] for e in block["episodes"]] == [s for *_x, s, _k in SMALL]  # 保持清单顺序
    for task, tier, seed, kind in SMALL:
        e = by_seed[seed]
        assert (e["task"], e["tier"], e["class"], e["id"]) == (task, tier, kind, f"{task}|{tier}|{seed}")
        rel = f"episodes/{tier}/{task}_episode_{seed}/hdf5_files/{task}_seed{seed}.h5"
        sha_a, sha_b = F.sha256(a_root / rel), F.sha256(b_root / rel)
        assert e["shas"] == list(dict.fromkeys([sha_a, sha_b]))
        assert (e["ok_a"], e["ok_b"]) == {"stable": (True, True), "known_fail": (False, False),
                                          "jitter": (True, False)}[kind]
        assert e["h5_a"] == f"{a_root.name}/{rel}" and e["h5_b"] == f"{b_root.name}/{rel}"
    assert by_seed[4]["shas"] == [hashlib.sha256(F.PLACEHOLDER).hexdigest()]
    assert ng.load_ref(small["ref"])["sha256"] == obj["sha256"]


def test_build_ref_输出已存在拒写(ng, small, capsys):
    small["ref"].write_text("占位\n", encoding="utf-8")
    assert ng.main(small["build_argv"]) == 2
    assert capsys.readouterr().out.strip().splitlines()[-1] == "NOISE_REF=FAIL reason=refused"
    assert small["ref"].read_text(encoding="utf-8") == "占位\n"


@pytest.mark.parametrize("extra", [["--limit", "2"], ["--trust-recorded-sha"]])
def test_build_ref_partial产物_判定方一律拒收(ng, small, tmp_path, capsys, extra):
    assert ng.main(small["build_argv"] + extra) == 0
    obj = json.loads(small["ref"].read_text(encoding="utf-8"))
    assert obj["partial"] is True and capsys.readouterr().out.strip().splitlines()[-1].startswith("NOISE_REF=PASS")
    with pytest.raises(ng.GateError, match="partial"):
        ng.load_ref(small["ref"])
    assert ng.load_ref(small["ref"], allow_partial=True)["partial"] is True
    first = F.write_run(tmp_path / "first", F.first_run_lines(small["ref"], "v9"), **valid_launch(ng))
    rc = ng.main(["gen-regress", "check", "--ref", str(small["ref"]), "--set", "v9", "--new", str(first),
                  "--out", str(tmp_path / "r.jsonl")])
    assert rc == 2 and capsys.readouterr().out.strip().splitlines()[-1] == "GEN_REGRESS=FAIL reason=refused"


def _rewrite_jsonl(path: Path, fn) -> None:
    rows = [json.loads(t) for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
    rows = fn(rows)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _record(small) -> Path:
    return next(small["ref"].parent.joinpath("records").glob("*.jsonl"))


def _set_field(seed, **kw):
    def fn(rows):
        for r in rows:
            if r.get("seed") == seed:
                r.update(kw)
        return rows
    return fn


@pytest.mark.parametrize("name,mutate,needle", [
    ("记录 sha 与重算不符", lambda s: _rewrite_jsonl(_record(s), _set_field(1, ref_sha256="0" * 64)), "ref_sha256="),
    ("记录判不同但两遍 sha 相同", lambda s: _rewrite_jsonl(_record(s), _set_field(2, **{"class": "diverge"})),
     "两遍重算 sha 相同"),
    ("记录判 byte_equal 但两遍 sha 不同",
     lambda s: _rewrite_jsonl(_record(s), _set_field(3, **{"class": "byte_equal", "fail_side": None})),
     "两遍重算 sha 不同"),
    ("记录漏了一局", lambda s: _rewrite_jsonl(_record(s), lambda rows: [r for r in rows if r.get("seed") != 5]),
     "没有 a-b 两遍比对记录覆盖"),
    ("identities 记录 sha 与文件不符",
     lambda s: _rewrite_jsonl(s["runs"]["v9"][1] / "identities.jsonl", _set_field(1, sha256="f" * 64)), "不符"),
    ("一遍里身份重复",
     lambda s: _rewrite_jsonl(s["runs"]["v9"][0] / "identities.jsonl", lambda rows: rows + [rows[0]]), "身份重复"),
    ("一遍里缺一局",
     lambda s: _rewrite_jsonl(s["runs"]["v9"][0] / "identities.jsonl", lambda rows: rows[1:]), "缺身份"),
    ("一遍里有清单外身份",
     lambda s: _rewrite_jsonl(s["runs"]["v9"][0] / "identities.jsonl",
                              lambda rows: rows + [dict(rows[0], seed=999)]), "清单外身份"),
    ("成功却无 h5",
     lambda s: _rewrite_jsonl(s["runs"]["v9"][0] / "identities.jsonl", _set_field(1, path=None)), "成功却无 h5"),
])
def test_build_ref_交叉核对反例_FAIL且不写文件(ng, small, capsys, name, mutate, needle):
    mutate(small)
    assert ng.main(small["build_argv"]) == 1, name
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1].startswith("NOISE_REF=FAIL "), name
    assert needle in out, (name, out)
    assert not small["ref"].exists()


def test_build_ref_两遍同根或集合不一致即拒(ng, small, capsys):
    a_root, _b = small["runs"]["v9"]
    argv = [x if not x.startswith("v9=") else f"v9={a_root},{a_root}" for x in small["build_argv"]]
    assert ng.main(argv) == 2
    argv = small["build_argv"] + ["--runs", f"xhard0={a_root},{small['runs']['v9'][1]}"]
    assert ng.main(argv) == 2
    out = capsys.readouterr().out
    assert out.count("NOISE_REF=FAIL reason=refused") == 2 and not small["ref"].exists()


def _resign(path: Path, fn) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fn(obj)
    obj["sha256"] = canonical_sha(obj)
    path.write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def test_参照篡改_不重签与重签后语义非法都拒(ng, small):
    assert ng.main(small["build_argv"]) == 0
    raw = small["ref"].read_text(encoding="utf-8")
    obj = json.loads(raw)
    obj["sets"]["v9"]["episodes"][0]["shas"] = ["0" * 64]
    small["ref"].write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ng.GateError, match="sha256"):
        ng.load_ref(small["ref"])
    small["ref"].write_text(raw, encoding="utf-8")
    _resign(small["ref"], lambda o: o["sets"]["v9"]["episodes"][0].update({"class": "stable?"}))
    with pytest.raises(ng.GateError, match="类别非法"):
        ng.load_ref(small["ref"])
    small["ref"].write_text(raw, encoding="utf-8")
    _resign(small["ref"], lambda o: o["sets"].update(xhard0={"episodes": [o["sets"]["v9"]["episodes"][0]]}))
    with pytest.raises(ng.GateError, match="身份重复"):
        ng.load_ref(small["ref"])
    small["ref"].write_text(raw, encoding="utf-8")
    _resign(small["ref"], lambda o: o.update(schema="noise-ref/0"))
    with pytest.raises(ng.GateError, match="schema"):
        ng.load_ref(small["ref"])
    small["ref"].write_text("{", encoding="utf-8")
    with pytest.raises(ng.GateError):
        ng.load_ref(small["ref"])


# ══ check：首跑 ══════════════════════════════════════════════════════════════════════════


def test_首跑全部复现基线_PASS(ng, base, tmp_path):
    for set_name in ("v9", "xhard0"):
        res = run_check(ng, base, tmp_path, set_name, F.first_run_lines(base["ref"], set_name))
        n = len(layout_of(base, set_name))
        jit = len(seeds_of(base, set_name, "jitter"))
        assert res["verdict"] == "PASS", res["line"]
        assert res["line"] == (f"GEN_REGRESS=PASS set={set_name} n={n} match={n - jit} jitter={jit} flip=0 "
                               "structural=0 unknown=0 missing=0 invalid=0")
        assert res["rerun"] == [] and res["invalid"] == [] and res["reasons"] == []


@pytest.mark.parametrize("override,note", [
    ({"ok": False, "sha": FAKE["x"], "error_type": "DatasetGenerationError"}, "fail"),
    ({"ok": True, "sha": FAKE["x"]}, "other_trajectory"),
    ({}, "same_as_a"),
])
def test_已知抖动局只报告不判定(ng, base, tmp_path, override, note):
    (seed,) = seeds_of(base, "v9", "jitter")
    over = {}
    if override:
        over = {seed: {"success": override["ok"], "sha256": override["sha"], "error_type": override.get("error_type")}}
    res = run_check(ng, base, tmp_path, "v9", F.first_run_lines(base["ref"], "v9", overrides=over))
    row = next(r for r in res["rows"].values() if r["seed"] == seed)
    assert res["verdict"] == "PASS" and row["category"] == "jitter_info" and row["jitter_note"] == note


@pytest.mark.parametrize("over", [
    {"success": True, "h5_error": None, "error_type": None},       # 确定性失败局变成功
    {"sha256": "a" * 64},                                         # 仍失败但占位字节不同
])
def test_确定性失败局不符期望即翻转(ng, base, tmp_path, over):
    (seed,) = seeds_of(base, "xhard0", "known_fail")
    res = run_check(ng, base, tmp_path, "xhard0", F.first_run_lines(base["ref"], "xhard0", overrides={seed: over}))
    assert res["verdict"] == "NEED_RERUN" and res["line"].startswith("GEN_REGRESS=NEED_RERUN flip=1 ")
    assert next(r for r in res["rows"].values() if r["seed"] == seed)["category"] == "flip_pending"


def test_稳定局失败或字节不同即翻转_并按同档优先挑陪跑局(ng, base, tmp_path):
    over = {20: {"success": False, "error_type": "DatasetGenerationError", "sha256": FAKE["x"]}}
    res = run_check(ng, base, tmp_path, "v9", F.first_run_lines(base["ref"], "v9", overrides=over))
    assert res["verdict"] == "NEED_RERUN"
    # 手写期望：同档（xhard2）的稳定且首跑 match 的 21 → 相邻档按身份顺序 10、11、12、30、31、32 ……；
    # 抖动局 22、确定性失败局 23 不当陪跑
    order = [21, 10, 11, 12, 30, 31, 32] + seeds_of(base, "v9", "stable")[8:]
    need = ng.RERUN_MIN - 1
    assert [r["seed"] for r in res["rerun"]] == [20] + order[:need]
    assert [r["filler"] for r in res["rerun"]] == [False] + [True] * need
    assert all(set(r) == {"task", "tier", "seed", "filler", "ref_class"} for r in res["rerun"])
    assert f"rerun_rows={1 + need} filler={need}" in res["line"]


def test_陪跑局只取首跑已与基线相同的稳定局(ng, base, tmp_path):
    # 20 翻转；21 也翻转 → 21 不能当陪跑；同档再无候选，转相邻档
    over = {20: {"sha256": FAKE["x"]}, 21: {"sha256": FAKE["y"]}}
    res = run_check(ng, base, tmp_path, "v9", F.first_run_lines(base["ref"], "v9", overrides=over))
    need = max(ng.RERUN_MIN - 2, 0)
    assert [r["seed"] for r in res["rerun"] if r["filler"]] == [10, 11, 12, 30, 31, 32][:need]
    assert [r["seed"] for r in res["rerun"] if not r["filler"]] == [20, 21]


def test_交换两个身份的产物_总数不变也判翻转(ng, base, tmp_path):
    """M10：两局的 sha 互换（局数、成功数都不变），逐局期望必须各自翻转。"""
    sa, sb = entry(base, "v9", 10)["shas"][0], entry(base, "v9", 11)["shas"][0]
    over = {10: {"sha256": sb}, 11: {"sha256": sa}}
    res = run_check(ng, base, tmp_path, "v9", F.first_run_lines(base["ref"], "v9", overrides=over))
    assert res["verdict"] == "NEED_RERUN" and " flip=2 " in res["line"]


def test_翻转数等于上限仍进第二次跑_超过上限直接FAIL(ng, base, tmp_path):
    big = [s for _t, tier, s, k in layout_of(base, "v9") if tier == "xhard4"]
    at = {s: {"sha256": FAKE["x"]} for s in big[: ng.FLIP_RERUN_MAX]}
    res = run_check(ng, base, tmp_path / "at", "v9", F.first_run_lines(base["ref"], "v9", overrides=at))
    assert res["verdict"] == "NEED_RERUN" and f" flip={ng.FLIP_RERUN_MAX} " in res["line"]
    over = {s: {"sha256": FAKE["x"]} for s in big[: ng.FLIP_RERUN_MAX + 1]}
    res = run_check(ng, base, tmp_path / "over", "v9", F.first_run_lines(base["ref"], "v9", overrides=over))
    assert res["verdict"] == "FAIL" and res["line"].startswith("GEN_REGRESS=FAIL ")
    assert not any(r["filler"] for r in res["rows"].values())
    assert any("超过" in x for x in res["reasons"])


@pytest.mark.parametrize("kind", ["gpu", "workers", "no_launch", "missing", "extra"])
def test_跑法前提不满足_本遍无效INVALID(ng, base, tmp_path, kind):
    lines = F.first_run_lines(base["ref"], "v9")
    launch = {}
    if kind == "gpu":
        launch = {"gpu": "NVIDIA RTX 6000 Ada Generation"}
    elif kind == "workers":
        launch = {"workers": ng.PRECOND_WORKERS + 1}
    elif kind == "no_launch":
        launch = {"launch": False}
    elif kind == "missing":
        lines = lines[1:]
    else:
        lines = lines + [F.id_line("PickXtimes", "xhard1", 999, sha=FAKE["x"])]
    res = run_check(ng, base, tmp_path, "v9", lines, first_launch=launch)
    assert res["verdict"] == "INVALID" and res["line"].startswith("GEN_REGRESS=INVALID ")
    assert res["invalid"], kind


@pytest.mark.parametrize("kind", ["duplicate", "infra", "no_sha", "h5_error", "verdict_recorded"])
def test_原因不明一局即FAIL_不进第二次跑(ng, base, tmp_path, kind):
    lines = F.first_run_lines(base["ref"], "v9")
    i = next(k for k, x in enumerate(lines) if x["seed"] == 30)
    if kind == "duplicate":
        lines = lines + [dict(lines[i])]
    elif kind == "infra":
        lines[i].update(success=False, error_type="TimeoutError")
    elif kind == "no_sha":
        lines[i].update(sha256=None)
    elif kind == "h5_error":
        lines[i].update(h5_error="OSError: truncated")
    else:
        lines[i].update(verdict="flip")  # Mover 当场写的判定与 check 重算不一致
    res = run_check(ng, base, tmp_path, "v9", lines)
    assert res["verdict"] == "FAIL", res["line"]
    assert " unknown=1 " in res["line"] and " structural=0 " in res["line"]
    assert next(r for r in res["rows"].values() if r["seed"] == 30)["category"] == "unknown"


def test_遗漏身份用重复行补齐_既缺局又重复(ng, base, tmp_path):
    """M15c：去掉一局、用另一局的重复行把行数补回原数。"""
    lines = F.first_run_lines(base["ref"], "v9")
    j = next(k for k, x in enumerate(lines) if x["seed"] == 31)
    lines = [x for x in lines if x["seed"] != 32] + [dict(lines[j])]
    assert len(lines) == len(layout_of(base, "v9"))
    res = run_check(ng, base, tmp_path, "v9", lines)
    assert res["verdict"] == "INVALID"
    assert next(r for r in res["rows"].values() if r["seed"] == 31)["category"] == "unknown"
    assert next(r for r in res["rows"].values() if r["seed"] == 32)["category"] == "missing"


# ══ check：第二次跑四格定性 ════════════════════════════════════════════════════════════════


def _rerun_case(ng, base, tmp_path, set_name, flips: dict[int, tuple], *, hosts=None, new2_extra=None,
                old_extra=None, drop_new2=()):
    """flips[seed] = (首跑覆盖, 改后第二次覆盖, 旧代码覆盖)；覆盖为 {} 表示复现基线。"""
    first = F.first_run_lines(base["ref"], set_name, overrides={s: v[0] for s, v in flips.items()})
    res1 = run_check(ng, base, tmp_path / "probe", set_name, first)
    assert res1["verdict"] == "NEED_RERUN", res1["line"]
    seeds = [r["seed"] for r in res1["rerun"]]
    new2 = rerun_lines(base, set_name, [s for s in seeds if s not in drop_new2],
                       {**{s: v[1] for s, v in flips.items()}, **(new2_extra or {})})
    old = rerun_lines(base, set_name, seeds, {**{s: v[2] for s, v in flips.items()}, **(old_extra or {})})
    return run_check(ng, base, tmp_path / "final", set_name, first, rerun=(new2, old, hosts or {})), res1


X = {"sha256": FAKE["x"]}
Y = {"sha256": FAKE["y"]}
Z = {"sha256": FAKE["z"]}


@pytest.mark.parametrize("cell,first,new2,old,want_verdict", [
    ("noise", X, {}, X, "PASS"),            # 改后第二次回到基线 → 噪声（旧代码不看）
    ("regression", X, X, {}, "FAIL"),       # 改后两次相同、旧代码回到基线 → 回归
    ("env_changed", X, X, Y, "FAIL"),       # 改后两次相同、旧代码也不同于基线 → 环境变了
    ("unstable", X, Y, {}, "FAIL"),         # 改后第二次与第一次也不同 → 每次都不同
])
def test_四格定性(ng, base, tmp_path, cell, first, new2, old, want_verdict):
    res, _ = _rerun_case(ng, base, tmp_path, "v9", {20: (first, new2, old)})
    row = next(r for r in res["rows"].values() if r["seed"] == 20)
    assert row["final"] == cell
    assert res["verdict"] == want_verdict, res["line"]
    counts = {c: int(c == cell) for c in ng.FINAL_CLASSES}
    assert " ".join(f"{c}={counts[c]}" for c in ng.FINAL_CLASSES) in res["line"]
    md = ng.episodes_markdown(res, "ref.json")
    assert f"| {ng.FINAL_NAMES[cell]} |" in md


def test_确定性失败局翻转后第二次回到同一失败_判噪声(ng, base, tmp_path):
    (seed,) = seeds_of(base, "xhard0", "known_fail")
    ok = {"success": True, "h5_error": None, "error_type": None, "sha256": FAKE["x"]}
    res, _ = _rerun_case(ng, base, tmp_path, "xhard0", {seed: (ok, {}, {})})
    assert next(r for r in res["rows"].values() if r["seed"] == seed)["final"] == "noise"
    assert res["verdict"] == "PASS"


@pytest.mark.parametrize("set_name", ["v9", "xhard0"])
def test_确认为噪声的翻转_上限处PASS_超一局FAIL(ng, base, tmp_path, set_name):
    stable = seeds_of(base, set_name, "stable")
    cap = ng.NOISE_MAX[set_name]
    res, _ = _rerun_case(ng, base, tmp_path / "at", set_name, {s: (X, {}, X) for s in stable[:cap]})
    assert res["verdict"] == "PASS" and f"noise={cap} " in res["line"], res["line"]
    res, _ = _rerun_case(ng, base, tmp_path / "over", set_name, {s: (X, {}, X) for s in stable[:cap + 1]})
    assert res["verdict"] == "FAIL" and f"noise={cap + 1} " in res["line"]
    assert any("上限" in x for x in res["reasons"])


def test_陪跑局不参与判定_只报告(ng, base, tmp_path):
    """首跑只 1 局翻转（噪声）；陪跑局在两次重跑里都走了别的轨迹，仍 PASS，四格计数只含翻转局。"""
    res1 = run_check(ng, base, tmp_path / "p", "v9", F.first_run_lines(base["ref"], "v9", overrides={20: X}))
    fillers = [r["seed"] for r in res1["rerun"] if r["filler"]]
    assert fillers, "需要至少一个陪跑局才能验证"
    bad = {s: Z for s in fillers}
    res, _ = _rerun_case(ng, base, tmp_path, "v9", {20: (X, {}, X)}, new2_extra=bad, old_extra=bad)
    assert res["verdict"] == "PASS", res["line"]
    assert sum(1 for r in res["rows"].values() if r.get("final")) == 1
    for s in fillers:
        row = next(r for r in res["rows"].values() if r["seed"] == s)
        assert row["filler"] and "final" not in row
        assert row["filler_new2_match"] is False and row["filler_old_match"] is False
    md = ng.episodes_markdown(res, "ref.json")
    assert md.count("陪跑（只报告）：改后异基线、旧码异基线") == len(fillers)


def test_陪跑局重跑原因不明不影响判定_翻转局原因不明FAIL(ng, base, tmp_path):
    res1 = run_check(ng, base, tmp_path / "p", "v9", F.first_run_lines(base["ref"], "v9", overrides={20: X}))
    filler = next(r["seed"] for r in res1["rerun"] if r["filler"])
    infra = {"success": False, "error_type": "TimeoutError"}
    res, _ = _rerun_case(ng, base, tmp_path / "f", "v9", {20: (X, {}, X)}, new2_extra={filler: infra})
    assert res["verdict"] == "PASS", res["line"]
    res, _ = _rerun_case(ng, base, tmp_path / "x", "v9", {20: (X, infra, X)})
    assert res["verdict"] == "FAIL" and " unknown=1 " in res["line"]


def test_第二次跑换节点或缺局_本遍无效(ng, base, tmp_path):
    res, _ = _rerun_case(ng, base, tmp_path / "h", "v9", {20: (X, {}, X)}, hosts={"old": "gl9999"})
    assert res["verdict"] == "INVALID" and any("同一节点" in x for x in res["invalid"])
    res, _ = _rerun_case(ng, base, tmp_path / "m", "v9", {20: (X, {}, X)}, drop_new2=(20,))
    assert res["verdict"] == "INVALID" and any("第二次跑缺" in x for x in res["invalid"])
    with pytest.raises(ng.GateError):
        ng.regress_check(ng.load_ref(base["ref"]), "v9", tmp_path / "h" / "final" / "first-v9",
                         rerun_new=tmp_path / "h" / "final" / "new2-v9")


@pytest.mark.parametrize("who", ["first", "new2", "old"])
def test_任一启动记录缺host_本遍无效_不静默放过(ng, base, tmp_path, who):
    """「第二次跑与首跑同一节点」只能在三份启动记录都写了 host 时核对；任一份缺 host 即判 INVALID
    （原因写 missing_host），不得把无法核对当成通过。"""
    first = F.first_run_lines(base["ref"], "v9", overrides={20: X})
    res1 = run_check(ng, base, tmp_path / "probe", "v9", first)
    assert res1["verdict"] == "NEED_RERUN", res1["line"]
    seeds = [r["seed"] for r in res1["rerun"]]
    new2 = rerun_lines(base, "v9", seeds, {20: {}})
    old = rerun_lines(base, "v9", seeds, {20: X})
    hosts = {who: None} if who != "first" else {"new2": "gl1001", "old": "gl1001"}
    res = run_check(ng, base, tmp_path / "final", "v9", first, rerun=(new2, old, hosts),
                    host=None if who == "first" else "gl1001")
    assert res["verdict"] == "INVALID", res["line"]
    assert sum("missing_host" in x for x in res["invalid"]) == 1, res["invalid"]


def test_缺workers或缺GPU字段_本遍无效(ng, base, tmp_path):
    """跑法前提的 workers／gpu 字段缺失（null）同样判 INVALID，不静默放过。"""
    lines = F.first_run_lines(base["ref"], "v9")
    res = run_check(ng, base, tmp_path / "w", "v9", lines, first_launch={"workers": None})
    assert res["verdict"] == "INVALID" and any("workers=None" in x for x in res["invalid"])
    res = run_check(ng, base, tmp_path / "g", "v9", lines, first_launch={"gpu": None})
    assert res["verdict"] == "INVALID" and any("GPU=None" in x for x in res["invalid"])


def test_逐局报告列齐_只列问题局陪跑局与抖动局(ng, base, tmp_path):
    (jseed,) = seeds_of(base, "v9", "jitter")
    res, _ = _rerun_case(ng, base, tmp_path, "v9", {20: (X, X, {})})
    md = ng.episodes_markdown(res, "ref.json")
    header = next(t for t in md.splitlines() if t.startswith("| 任务 |"))
    assert [c.strip() for c in header.strip("|").split("|")] == [
        "任务", "档", "seed", "参照类", "陪跑", "第一次", "改后第二次", "旧代码", "一=二", "一=旧", "二=旧", "分叉步",
        "子目标", "定性"]
    body = [t for t in md.splitlines() if t.startswith("| ") and not t.startswith("| 任务") and "---" not in t]
    seeds = sorted(int(t.split("|")[3]) for t in body)
    fillers = [r["seed"] for r in res["rerun"] if r["filler"]]
    assert seeds == sorted([20, jseed, *fillers])
    row20 = next(t for t in body if t.split("|")[3].strip() == "20")
    cells = [c.strip() for c in row20.strip("|").split("|")]
    assert cells[5].startswith(f"`{FAKE['x'][:12]}` 成功") and cells[8] == "同" and cells[9] == "异" and cells[10] == "异"
    assert cells[13] == ng.FINAL_NAMES["regression"]


# ══ check：本机细分翻转类别（--local-ref-root）══════════════════════════════════════════════


def test_本机细分_分叉步与子目标_结构不同直接FAIL(ng, base, tmp_path):
    new_root = tmp_path / "first-v9"
    rel_d, rel_s = "ep/div/x.h5", "ep/str/x.h5"
    F.write_h5(new_root / rel_d, seed=20, joint_offset_from=2)
    F.write_h5(new_root / rel_s, seed=21, drop=(1, "obs/joint_state"))
    over = {20: {"path": rel_d, "sha256": F.sha256(new_root / rel_d)},
            21: {"path": rel_s, "sha256": F.sha256(new_root / rel_s)},
            30: {"success": False, "error_type": "PlannerExhausted", "sha256": None, "path": None}}
    res = run_check(ng, base, tmp_path, "v9", F.first_run_lines(base["ref"], "v9", overrides=over), local=True)
    rows = {r["seed"]: r for r in res["rows"].values()}
    assert rows[20]["category"] == "diverge" and rows[20]["first_divergence"] == 2
    assert rows[20]["subgoal"] == "sg2"  # 子目标取自基线 a 遍 h5 的分叉帧
    assert rows[21]["category"] == "structural" and rows[30]["category"] == "gen_fail"
    assert res["verdict"] == "FAIL" and " structural=1 " in res["line"]


def test_本机细分_翻转局h5未回传仍记待定(ng, base, tmp_path):
    res = run_check(ng, base, tmp_path, "v9", F.first_run_lines(base["ref"], "v9", overrides={20: X}), local=True)
    row = next(r for r in res["rows"].values() if r["seed"] == 20)
    assert row["category"] == "flip_pending" and row["sub_reason"] == "new_h5_not_local"
    assert res["verdict"] == "NEED_RERUN"


# ══ check：CLI（退出码、逐局 jsonl、重跑清单）══════════════════════════════════════════════


def test_cli_check_退出码与产物(ng, base, tmp_path, capsys):
    first = F.write_run(tmp_path / "first", F.first_run_lines(base["ref"], "v9", overrides={20: X}),
                        **valid_launch(ng))
    out, rerun = tmp_path / "rg.jsonl", tmp_path / "rerun.jsonl"
    rc = ng.main(["gen-regress", "check", "--ref", str(base["ref"]), "--set", "v9", "--new", str(first),
                  "--out", str(out), "--rerun-identities-out", str(rerun)])
    text = capsys.readouterr().out.strip().splitlines()
    assert rc == 3 and text[-1].startswith("GEN_REGRESS=NEED_RERUN flip=1 set=v9 ")
    rows = [json.loads(t) for t in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "meta" and rows[0]["schema"] == "gen-regress/1" and rows[0]["line"] == text[-1]
    assert len(rows) == 1 + len(layout_of(base, "v9"))
    assert (tmp_path / "rg.episodes.md").is_file()
    plan = [json.loads(t) for t in rerun.read_text(encoding="utf-8").splitlines()]
    assert plan[0]["seed"] == 20 and not plan[0]["filler"] and all(p["filler"] for p in plan[1:])
    # 重跑清单可直接作 hard_parity generate --identities（每行 task、tier、seed）
    assert F.hard_parity().read_identity_subset(rerun) == {(p["task"], p["tier"], p["seed"]) for p in plan}
    bad = F.write_run(tmp_path / "bad", F.first_run_lines(base["ref"], "v9"), workers=ng.PRECOND_WORKERS + 1,
                      gpu=valid_launch(ng)["gpu"])
    assert ng.main(["gen-regress", "check", "--ref", str(base["ref"]), "--set", "v9", "--new", str(bad),
                    "--out", str(tmp_path / "b.jsonl")]) == 4
    ok = F.write_run(tmp_path / "ok", F.first_run_lines(base["ref"], "v9"), **valid_launch(ng))
    assert ng.main(["gen-regress", "check", "--ref", str(base["ref"]), "--set", "v9", "--new", str(ok),
                    "--out", str(tmp_path / "o.jsonl")]) == 0
    capsys.readouterr()


# ══ 真实参照文件（只读）══════════════════════════════════════════════════════════════════


def test_真实参照文件_自洽且身份集合等于冻结检查集(ng):
    path = F.REPO / "scripts" / "configs" / "noise-ref-20261003.json"
    raw = path.read_bytes()
    ref = ng.load_ref(path)  # schema、canonical sha、类别合法、身份唯一、非 partial
    assert ref["sha256"] == canonical_sha(json.loads(raw))
    assert ref["partial"] is False and ref["rehash"] is True
    assert set(ref["sets"]) == set(ng.REF_SETS)
    for name, block in ref["sets"].items():
        ident_file = F.REPO / block["identities"]["file"]
        assert F.sha256(ident_file) == block["identities"]["sha256"]
        gate = ng.load_identities(ident_file)
        assert [(e["task"], e["tier"], e["seed"]) for e in block["episodes"]] == \
            [(g["task"], g["tier"], g["seed"]) for g in gate]
        counted = {c: sum(e["class"] == c for e in block["episodes"]) for c in ng.REF_CLASSES}
        assert block["counts"] == counted and sum(counted.values()) == block["n"] == len(gate)
        for e in block["episodes"]:
            # 每类的 sha 个数与成败：稳定局与确定性失败局两遍同字节（1 个 sha），抖动局两遍至少一边成功
            if e["class"] == "stable":
                assert e["ok_a"] and e["ok_b"] and len(e["shas"]) == 1
            elif e["class"] == "known_fail":
                assert not e["ok_a"] and not e["ok_b"] and len(e["shas"]) == 1
            else:
                assert e["ok_a"] != e["ok_b"] or len(e["shas"]) == 2
        for run in block["runs"]:
            assert run["workers"] == ng.PRECOND_WORKERS and ng.PRECOND_GPU in run["gpu_model"]
    assert path.read_bytes() == raw  # 只读


# ══ jitter_observed：用户确认的环境敏感局登记（mark-jitter）══════════════════════════════


def _marked_copy(ng, base, tmp_path, seed, shas_seen):
    """把模块基线参照复制一份，经 CLI 把 v9 的一局登记为环境敏感局；返回 (原参照对象, 新参照路径)。"""
    ref = tmp_path / "ref-marked.json"
    ref.write_text(base["ref"].read_text(encoding="utf-8"), encoding="utf-8")
    before = ng.load_ref(ref)
    e = entry(base, "v9", seed)
    argv = ["gen-regress", "mark-jitter", "--ref", str(ref), "--set", "v9", "--task", e["task"],
            "--seed", str(seed), "--reason", "测试：同节点新旧代码同为新 sha", "--decision", "测试裁决"]
    for s in shas_seen:
        argv += ["--sha", s]
    assert ng.main(argv) == 0
    return before, ref


def test_登记环境敏感局_基线逐局数据不动_翻转改判只报告(ng, base, tmp_path, capsys):
    seed = seeds_of(base, "v9", "stable")[-1]
    flip_line = {seed: {"sha256": FAKE["y"]}}
    first = F.write_run(tmp_path / "first", F.first_run_lines(base["ref"], "v9", overrides=flip_line),
                        **valid_launch(ng))
    # 登记前：稳定局字节不同 → 翻转、要第二次跑
    assert ng.regress_check(ng.load_ref(base["ref"]), "v9", first)["verdict"] == "NEED_RERUN"

    before, ref = _marked_copy(ng, base, tmp_path, seed, [FAKE["y"]])
    after = ng.load_ref(ref)
    out = capsys.readouterr().out
    assert f"NOISE_REF_MARK=PASS set=v9 id={entry(base, 'v9', seed)['task']}|{seed} observed=1 sets_unchanged=1" in out
    assert after["sets"] == before["sets"]                       # 逐局基线数据一个字节不动
    assert after["sha256"] != before["sha256"] and after["sha256"] == canonical_sha(after)
    idx = ng.ref_index(after, "v9")[(entry(base, "v9", seed)["task"], seed)]
    assert (idx["class"], idx["baseline_class"], idx["jitter_observed"]) == ("jitter", "stable", True)

    res = ng.regress_check(after, "v9", first)
    row = next(r for r in res["rows"].values() if r["seed"] == seed)
    n, jit = len(layout_of(base, "v9")), len(seeds_of(base, "v9", "jitter")) + 1
    assert res["line"] == (f"GEN_REGRESS=PASS set=v9 n={n} match={n - jit} jitter={jit} flip=0 "
                           "structural=0 unknown=0 missing=0 invalid=0")
    assert (row["category"], row["jitter_note"], row["baseline_class"]) == ("jitter_info", "same_as_observed", "stable")

    # 登记后又出现第三种字节：照样只报告，但标成另一条轨迹
    other = F.write_run(tmp_path / "other", F.first_run_lines(base["ref"], "v9", overrides={seed: {"sha256": FAKE["z"]}}),
                        **valid_launch(ng))
    row = next(r for r in ng.regress_check(after, "v9", other)["rows"].values() if r["seed"] == seed)
    assert (row["category"], row["jitter_note"]) == ("jitter_info", "other_trajectory")


def test_登记前已在生成时记为翻转的局_事后登记不判原因不明(ng, base, tmp_path):
    """生成时 Mover 按旧参照把这一局记成 flip；登记后重判，记录与现判不同只因登记在后，归抖动而非 unknown。"""
    seed = seeds_of(base, "v9", "stable")[-1]
    _before, ref = _marked_copy(ng, base, tmp_path, seed, [FAKE["y"]])
    lines = F.first_run_lines(base["ref"], "v9", overrides={seed: {"sha256": FAKE["y"], "verdict": "flip"}})
    res = ng.regress_check(ng.load_ref(ref), "v9", F.write_run(tmp_path / "first", lines, **valid_launch(ng)))
    row = next(r for r in res["rows"].values() if r["seed"] == seed)
    assert res["verdict"] == "PASS" and row["category"] == "jitter_info"
    # 反例：未登记的稳定局，记录 flip 却现判 match → 仍是原因不明
    other = seeds_of(base, "v9", "stable")[0]
    lines = F.first_run_lines(base["ref"], "v9", overrides={other: {"verdict": "flip"}})
    res = ng.regress_check(ng.load_ref(ref), "v9", F.write_run(tmp_path / "first2", lines, **valid_launch(ng)))
    assert res["verdict"] == "FAIL" and next(r for r in res["rows"].values() if r["seed"] == other)["category"] == "unknown"


def test_登记反例_重复登记与参照外身份都拒(ng, base, tmp_path):
    seed = seeds_of(base, "v9", "stable")[-1]
    _before, ref = _marked_copy(ng, base, tmp_path, seed, [FAKE["y"]])
    obj = ng.load_ref(ref)
    with pytest.raises(ng.GateError, match="已登记"):
        ng.mark_jitter(obj, set_name="v9", task=entry(base, "v9", seed)["task"], seed=seed, shas_seen=[FAKE["z"]],
                       reason="r", decision="d")
    other = seeds_of(base, "v9", "stable")[0]          # 未登记、但不在 xhard0 集合里
    with pytest.raises(ng.GateError, match="没有"):
        ng.mark_jitter(obj, set_name="xhard0", task=entry(base, "v9", other)["task"], seed=other,
                       shas_seen=[FAKE["z"]], reason="r", decision="d")


@pytest.mark.parametrize("mutate,needle", [
    (lambda item: item.pop("decision"), "缺字段"),
    (lambda item: item.update(tier="xhard9"), "找不到同档"),
    (lambda item: item.update(seed=item["seed"] + 987654), "找不到同档"),
    (lambda item: item.update(shas_seen=[]), "非空列表"),
])
def test_登记条目被改坏_重签后仍拒(ng, base, tmp_path, mutate, needle):
    seed = seeds_of(base, "v9", "stable")[-1]
    _before, ref = _marked_copy(ng, base, tmp_path, seed, [FAKE["y"]])
    _resign(ref, lambda o: mutate(o["jitter_observed"][0]))
    with pytest.raises(ng.GateError, match=needle):
        ng.load_ref(ref)
