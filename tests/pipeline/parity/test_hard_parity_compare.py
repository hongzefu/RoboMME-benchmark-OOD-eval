"""C15 ``hard_parity.py compare`` 的分母与终态：经真实 CLI 入口 ``hard_parity.main([...])`` 跑三条路径。

- v9（``--tier v9``，H:H2）：tmp 下的微型 /4 规格根（拷贝包内最小的 xhard5 档 ``specs.jsonl``，只读源文件）
  作冻结交付集 F；手写交付清单 D（v8-delivery/1）、两侧输出根 H／H2（微型 h5，H2 为 H 的逐字节拷贝）。
  分母 ``_v8_denominator``：F = D = H = H2（子集口径再加 identities 第五方），missing／extra／duplicate 与判定行。
- native、xhard0（O:H）：无分母核对，按清单身份逐局比；空清单、重复行不得判 PASS。

期望由用例独立得出：冻结交付集的身份直接按「selected 且 rollout.status == ok」从规格文件读，计数手算。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import parity_fixtures as F

TIER = "xhard5"
SRC_SPECS = F.REPO / "src" / "robomme_hard" / "env_metadata" / "ood" / TIER / "specs.jsonl"
OUTSIDE = ("StopCube", TIER, 99_999_999)


@pytest.fixture(scope="module")
def hp():
    return F.hard_parity()


def delivered_rows() -> tuple[list[dict], dict[str, int]]:
    lines = [json.loads(t) for t in SRC_SPECS.read_text(encoding="utf-8").splitlines() if t.strip()]
    header, rows = lines[0], lines[1:]
    got = [{"task": r["task"], "tier": TIER, "episode": int(r["episode"]), "seed": int(r["seed"]),
            "sha": r["rollout"]["h5_sha256"]}
           for r in rows if r.get("selected") and (r.get("rollout") or {}).get("status") == "ok"]
    return got, dict(header["delivery_per_cell"])


def side_line(side: str, r: dict, path: str, sha: str) -> dict:
    line = {"side": side, "tier": r["tier"], "task": r["task"], "episode": r["episode"], "seed": r["seed"],
            "success": True, "path": path, "sha256": sha, "recovery_mode": None}
    if side in ("H", "H2"):
        line["env_module"] = f"robomme_hard.robomme_env.{r['task']}"
    else:
        line.update(worker="official._worker", robomme_module="/official/src/robomme/__init__.py")
    return line


def write_side(root: Path, side: str, rows: list[dict], h5_bytes: dict, sha_of) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in rows:
        rel = f"episodes/{r['task']}_episode_{r['episode']}/hdf5_files/x.h5"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(h5_bytes[(r["task"], r["seed"])])
        lines.append(side_line(side, r, rel, sha_of(r)))
    F.write_jsonl(root / "identities.jsonl", lines)
    return root


@pytest.fixture()
def v9(tmp_path):
    rows, quota = delivered_rows()
    specs_root = tmp_path / "specs"
    (specs_root / TIER).mkdir(parents=True)
    shutil.copyfile(SRC_SPECS, specs_root / TIER / "specs.jsonl")
    cells = {f"{task}/{TIER}": n for task, n in quota.items()}
    h5 = {}
    for r in rows + [dict(task=OUTSIDE[0], tier=TIER, episode=999, seed=OUTSIDE[2], sha="0" * 64)]:
        h5[(r["task"], r["seed"])] = F.write_h5(tmp_path / "h5src" / f"{r['seed']}.h5", seed=r["seed"] % 997,
                                                frames=3).read_bytes()
    return {"tmp": tmp_path, "rows": rows, "cells": cells, "specs_root": specs_root, "h5": h5, "n": len(rows)}


def build(w, *, delivery=None, left=None, right=None, identities=None, delivery_sha=None, tag="r") -> list[str]:
    """各方默认 = 冻结交付集；传入的列表覆盖该方。返回 compare 的 argv。"""
    rows = w["rows"]
    root = w["tmp"] / tag
    d_rows = delivery if delivery is not None else rows
    manifest = root / "delivery.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema": "v8-delivery/1", "rows": [
        {"task": r["task"], "tier": r["tier"], "episode": r["episode"], "seed": r["seed"], "candidate": r["episode"],
         "h5_sha256": (delivery_sha or {}).get(r["seed"], r["sha"])} for r in d_rows]}), encoding="utf-8")
    left_dir = write_side(root / "H", "H", left if left is not None else rows, w["h5"], lambda r: r["sha"])
    right_dir = write_side(root / "H2", "H2", right if right is not None else rows, w["h5"], lambda r: r["sha"])
    argv = ["compare", "--pair", "H:H2", "--tier", "v9", "--manifest", str(manifest),
            "--specs-root", str(w["specs_root"]), "--cells", json.dumps(w["cells"]), "--left", str(left_dir),
            "--right", str(right_dir), "--compare-root", str(root / "cmp"), "--workers", "2"]
    if identities is not None:
        path = F.write_jsonl(root / "ids.jsonl", [{"task": r["task"], "tier": r["tier"], "seed": r["seed"]}
                                                  for r in identities])
        argv += ["--identities", str(path)]
    return argv


def run(hp, capsys, argv) -> tuple[int, str, dict]:
    rc = hp.main(argv)
    out = capsys.readouterr().out.splitlines()
    line = next(t for t in out if t.startswith("PARITY_"))
    root = Path(argv[argv.index("--compare-root") + 1])
    summary = json.loads(next(root.glob("*/summary.json")).read_text(encoding="utf-8"))
    return rc, line, summary


def fields(line: str) -> dict[str, str]:
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


# ── v9：全集与子集口径 ─────────────────────────────────────────────────────────────


def test_v9_四方一致_PASS_分母与五终态(hp, v9, capsys):
    n = v9["n"]
    rc, line, summary = run(hp, capsys, build(v9))
    f = fields(line)
    assert rc == 0 and line.startswith("PARITY_H_H2=PASS tier=v9 ")
    assert {k: f[k] for k in ("compared", "cells", "missing", "extra", "duplicate", "identity_equal", "byte_equal",
                              "noise", "h2_fail", "flipped", "structural", "expected", "frozen", "delivery",
                              "left_rows", "right_rows")} == {
        "compared": str(n), "cells": str(len(v9["cells"])), "missing": "0", "extra": "0", "duplicate": "0",
        "identity_equal": str(n), "byte_equal": str(n), "noise": "0", "h2_fail": "0", "flipped": "0",
        "structural": "0", "expected": str(n), "frozen": str(n), "delivery": str(n), "left_rows": str(n),
        "right_rows": str(n)}
    assert summary["denominator"]["frozen_n"] == n and summary["terminal"]["byte_equal"] == n


def test_v9_子集口径_只比清单内身份(hp, v9, capsys):
    k = 3
    rc, line, summary = run(hp, capsys, build(v9, identities=v9["rows"][:k]))
    f = fields(line)
    assert rc == 0 and line.startswith("PARITY_H_H2=PASS ")
    assert (f["compared"], f["expected"], f["identities"], f["manifest_rows_all"]) == (str(k), str(k), str(k),
                                                                                          str(v9["n"]))
    # 清单外的行只记进 outside_subset（第五方 identities 少了它们 = 合法子集，不是缺局）；交付清单在进分母前
    # 已按子集筛过，所以它的 outside 为 0
    assert summary["denominator"]["outside_subset"] == {"frozen": v9["n"] - k, "delivery": 0,
                                                        "left": v9["n"] - k, "right": v9["n"] - k}


def test_v9_空子集清单即拒(hp, v9):
    argv = build(v9)
    empty = F.write_jsonl(v9["tmp"] / "empty.jsonl", [])
    with pytest.raises(hp.ParityError, match="子集为空"):
        hp.main(argv + ["--identities", str(empty)])


def test_v9_空交付清单不得因0等于0判PASS(hp, v9, capsys):
    rc, line, _ = run(hp, capsys, build(v9, delivery=[], left=[], right=[]))
    f = fields(line)
    assert rc == 1 and line.startswith("PARITY_H_H2=FAIL ")
    assert f["missing"] == str(v9["n"]) and f["delivery"] == "0"


@pytest.mark.parametrize("party", ["delivery", "left", "right"])
def test_v9_交付清单或任一侧缺一局(hp, v9, capsys, party):
    short = v9["rows"][1:]
    rc, line, _ = run(hp, capsys, build(v9, **{party: short}))
    f = fields(line)
    assert rc == 1 and line.startswith("PARITY_H_H2=FAIL ")
    assert (f["missing"], f["extra"], f["duplicate"]) == ("1", "0", "0")
    assert f[{"delivery": "delivery", "left": "left_rows", "right": "right_rows"}[party]] == str(v9["n"] - 1)


def test_v9_冻结集缺一局_子集清单含清单外身份(hp, v9, capsys):
    """F 缺一：其余四方（含 identities）都多出一个冻结集里没有的身份 → 同时计 missing 与 extra。"""
    out = dict(task=OUTSIDE[0], tier=TIER, episode=999, seed=OUTSIDE[2], sha="0" * 64)
    rows = v9["rows"][:2] + [out]
    rc, line, _ = run(hp, capsys, build(v9, delivery=v9["rows"] + [out], left=v9["rows"] + [out],
                                        right=v9["rows"] + [out], identities=rows))
    f = fields(line)
    assert rc == 1 and (f["missing"], f["extra"]) == ("1", "1") and f["structural"] == "1"


def test_v9_全集口径下一侧多出清单外身份(hp, v9, capsys):
    out = dict(task=OUTSIDE[0], tier=TIER, episode=999, seed=OUTSIDE[2], sha="0" * 64)
    rc, line, _ = run(hp, capsys, build(v9, left=v9["rows"] + [out]))
    f = fields(line)
    assert rc == 1 and (f["missing"], f["extra"], f["compared"]) == ("0", "1", str(v9["n"] + 1))


@pytest.mark.parametrize("party", ["delivery", "left"])
def test_v9_重复行(hp, v9, capsys, party):
    dup = v9["rows"] + [v9["rows"][0]]
    rc, line, _ = run(hp, capsys, build(v9, **{party: dup}))
    f = fields(line)
    assert rc == 1 and f["duplicate"] == "1" and f["structural"] == "1"


def test_v9_gen1的sha与交付清单不符判结构不同(hp, v9, capsys):
    seed = v9["rows"][0]["seed"]
    rc, line, summary = run(hp, capsys, build(v9, delivery_sha={seed: "1" * 64}))
    f = fields(line)
    assert rc == 1 and (f["missing"], f["extra"], f["duplicate"]) == ("0", "0", "0")
    assert f["structural"] == "1" and f["byte_equal"] == str(v9["n"] - 1)
    bad = [k for k, v in summary["terminal_by_identity"].items() if v == "structural"]
    assert bad == [f"{v9['rows'][0]['task']}/{TIER}/{seed}"]


# ── native／xhard0：按清单身份逐局比 ─────────────────────────────────────────────────


def native_world(tmp: Path, tier_name: str, difficulty: str, *, manifest_rows=None, left_rows=None, right_rows=None):
    rows = [{"task": "PickXtimes", "tier": difficulty, "episode": e, "seed": 5000 + e} for e in range(3)]
    h5 = {(r["task"], r["seed"]): F.write_h5(tmp / "h5src" / f"{r['seed']}.h5", seed=r["seed"] % 997,
                                             frames=3).read_bytes() for r in rows}
    m_rows = rows if manifest_rows is None else manifest_rows
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"rows_total": len(m_rows), "rows": [
        {"task": r["task"], "difficulty": r["tier"], "episode": r["episode"], "seed": r["seed"]} for r in m_rows]}))
    left = write_side(tmp / "O", "O", rows if left_rows is None else left_rows, h5, lambda r: f"{r['seed']:064d}")
    right = write_side(tmp / "H", "H", rows if right_rows is None else right_rows, h5, lambda r: f"{r['seed']:064d}")
    return ["compare", "--pair", "O:H", "--tier", tier_name, "--manifest", str(manifest), "--left", str(left),
            "--right", str(right), "--compare-root", str(tmp / "cmp"), "--workers", "2"], rows


@pytest.mark.parametrize("tier_name,difficulty", [("native", "easy"), ("xhard0", "hard")])
def test_native与xhard0_两侧一致PASS(hp, tmp_path, capsys, tier_name, difficulty):
    argv, rows = native_world(tmp_path, tier_name, difficulty)
    rc, line, summary = run(hp, capsys, argv)
    f = fields(line)
    assert rc == 0 and line.startswith(f"PARITY_O_H=PASS tier={tier_name} ")
    assert (f["compared"], f["identity_equal"], f["both_success"], f["binding_ok"]) == ("3", "3", "3", "3")
    assert "manifest_rows" not in f and "denominator" not in summary


@pytest.mark.parametrize("case", ["empty", "duplicate", "left_missing"])
def test_native负例_空清单_重复行_一侧缺局(hp, tmp_path, capsys, case):
    base = [{"task": "PickXtimes", "tier": "easy", "episode": e, "seed": 5000 + e} for e in range(3)]
    # 空清单时两侧也空：各项计数都是 0 = 0，只靠「compared > 0」挡住
    kw = {"empty": {"manifest_rows": [], "left_rows": [], "right_rows": []}, "duplicate": {"manifest_rows": base + [base[0]]},
          "left_missing": {"left_rows": base[1:]}}[case]
    argv, _ = native_world(tmp_path, "native", "easy", **kw)
    rc, line, _ = run(hp, capsys, argv)
    f = fields(line)
    assert rc == 1 and line.startswith("PARITY_O_H=FAIL ")
    if case == "empty":
        assert f["compared"] == "0" and f["manifest_rows"] == "0"
    elif case == "duplicate":
        assert f["manifest_duplicate"] == "1"
    else:
        assert f["identity_equal"] == "2"


def test_identities只用于v9(hp, tmp_path):
    argv, _ = native_world(tmp_path, "native", "easy")
    ids = F.write_jsonl(tmp_path / "ids.jsonl", [{"task": "PickXtimes", "tier": "easy", "seed": 5000}])
    with pytest.raises(hp.ParityError, match="--identities"):
        hp.main(argv + ["--identities", str(ids)])
