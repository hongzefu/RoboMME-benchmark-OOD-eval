"""拆仓验收 ``hard_parity.py smoke``（PARITY_GEN_SMOKE）：参数闸门、生成次数硬上限、两侧逐身份比对与判定行。

不起任何仿真：builder 解析（``resolve_builder_identities``）、hard-verify 两侧的 ``cmd_generate``、ood 两侧的
``run_with_mover`` 一律换成测试进程内的替身，替身按命令行在各侧输出目录写微型 h5 与 ``identities.jsonl``；
比对（``smoke_compare`` → 真实 ``pair_metrics``）、容差读取（真实 ``configs/hard-parity-tolerances.json``）与判定行
全部走真实代码。期望由本文件手写的输入推出。
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import parity_fixtures as F

TASK = "VideoUnmask"


@pytest.fixture(scope="module")
def hp():
    return F.hard_parity()


def fields(line: str) -> dict[str, str]:
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


# ── 纯函数 ──────────────────────────────────────────────────────────────


def test_局号区间是半开区间_非法即拒(hp):
    assert hp.parse_episode_range("0:2") == [0, 1]
    assert hp.parse_episode_range("3:4") == [3]
    for bad in ("2:2", "3:1", "-1:1", "0", "a:b", ""):
        with pytest.raises(hp.ParityError):
            hp.parse_episode_range(bad)


def test_生成次数按乘式计_硬上限是1x2x2x2(hp):
    sides = {"hard-verify": ("O", "H"), "ood": ("H_old", "H_new")}
    assert hp.smoke_generations([TASK], {"hard-verify": [0, 1], "ood": [0, 1]}, sides) == 1 * 2 * 2 * 2
    assert hp.SMOKE_MAX_GENERATIONS == 1 * 2 * 2 * 2
    assert hp.smoke_generations([TASK], {"hard-verify": [0, 1, 2], "ood": [0, 1]}, sides) == 10


# ── 替身世界 ────────────────────────────────────────────────────────────


def _hv_rows():
    return [{"task": TASK, "tier": "xhard0", "seed": 900 + k, "source_episode": 3 + 4 * k, "builder_episode": k,
             "spec_sha256": None} for k in range(2)]


def _ood_rows():
    return [{"task": TASK, "tier": "xhard1", "seed": 1700 + k, "candidate": k, "builder_episode": k,
             "spec_sha256": f"spec{k}"} for k in range(2)]


class World:
    """记录全部替身调用；``outcome[(侧, seed)]`` 控制该局：``ok``（缺省，h5 与基准同字节）、``fail``（无 h5）、
    ``state``（关节状态整体偏 0.5，超容差）、``badmod``（环境类模块不属于 robomme_hard）。"""

    def __init__(self, tmp: Path, outcome: dict | None = None, old_rows=None):
        self.tmp, self.outcome = tmp, dict(outcome or {})
        self.old_rows = old_rows
        self.resolve_calls: list[tuple] = []
        self.gen_calls: list = []
        self.mover_calls: list = []

    def resolve(self, python, src, dataset, task, episodes):
        self.resolve_calls.append((str(python), str(src), dataset, task, list(episodes)))
        if dataset == "hard-verify":
            return [r for r in _hv_rows() if r["builder_episode"] in episodes]
        if self.old_rows is not None and "old" in str(src):
            return self.old_rows
        return [r for r in _ood_rows() if r["builder_episode"] in episodes]

    def _write_side(self, out: Path, side: str, rows, *, official: bool, src: str | None = None):
        out.mkdir(parents=True, exist_ok=True)
        lines = []
        for r in rows:
            kind = self.outcome.get((side, r["seed"]), "ok")
            rel = f"episodes/{r['tier']}/{TASK}_episode_{r['seed']}/hdf5_files/x.h5"
            module = "robomme.robomme_env." + TASK if official else "robomme_hard.robomme_env." + TASK
            if kind == "badmod":
                module = "robomme.robomme_env." + TASK
            line = {"task": TASK, "tier": "hard" if official else r["tier"], "seed": r["seed"],
                    "success": kind != "fail", "path": None if kind == "fail" else rel,
                    "worker": "official._worker" if official else "train_split_worker.run_one",
                    "env_module": module, "robomme_module": "/bench/src/robomme/__init__.py"}
            if kind != "fail":
                F.write_h5(out / rel, seed=r["seed"], state_offset_from=0 if kind == "state" else None)
            lines.append(line)
        F.write_jsonl(out / "identities.jsonl", lines)
        if src is not None:
            rnd = out / "_rounds" / "xhard1_round_00"
            rnd.mkdir(parents=True)
            (rnd / "results.json").write_text(json.dumps({"robomme_module": f"{src}/robomme/__init__.py"}))
        return lines

    def generate(self, args):
        self.gen_calls.append(args)
        subset = [json.loads(t) for t in Path(args.identities).read_text().splitlines() if t.strip()]
        rows = [r for r in _hv_rows() if r["seed"] in {s["seed"] for s in subset}]
        self._write_side(Path(args.out), args.side, rows, official=args.side == "O")
        return 0

    def mover(self, command, env, out, side, tier, meta, **_kw):
        self.mover_calls.append((list(command), dict(env), side))
        ids = [json.loads(t) for t in Path(command[command.index("--identities") + 1]).read_text().splitlines()]
        rows = [r for r in _ood_rows() if r["seed"] in {i["seed"] for i in ids}]
        src = command[command.index("--expect-src") + 1]
        lines = self._write_side(Path(out), side, rows, official=False, src=src)
        return types.SimpleNamespace(returncode=0), lines, types.SimpleNamespace(errors=[])

    def install(self, hp, monkeypatch):
        monkeypatch.setattr(hp, "resolve_builder_identities", self.resolve)
        monkeypatch.setattr(hp, "cmd_generate", self.generate)
        monkeypatch.setattr(hp, "run_with_mover", self.mover)
        return self


def _old_src(tmp: Path) -> Path:
    src = tmp / "old-snapshot" / "src"
    (src / "robomme_hard").mkdir(parents=True, exist_ok=True)
    (src / "robomme_hard" / "__init__.py").write_text("")
    return src


def _argv(tmp: Path, **over) -> list[str]:
    opts = {"--task": TASK, "--hard-verify-episodes": "0:2", "--ood-episodes": "0:2", "--hard-verify-sides": "O,H",
            "--ood-sides": "H_old,H_new", "--h-old-src": str(_old_src(tmp)), "--h-old-python": sys.executable,
            "--workers": "2", "--out": str(tmp / "out")}
    opts.update(over)
    argv = ["smoke"]
    for k, v in opts.items():
        argv += [k, v]
    return argv


def _last(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


# ── 端到端（替身生成 + 真实比对）──────────────────────────────────────────


def test_四身份两侧同字节_PASS_且生成恰好8次(hp, tmp_path, monkeypatch, capsys):
    world = World(tmp_path).install(hp, monkeypatch)
    assert hp.main(_argv(tmp_path)) == 0
    line = _last(capsys)
    assert line == "PARITY_GEN_SMOKE=PASS identities=4 sides=2 compared=4 both_fail=0 tol_over=0"
    # hard-verify 两侧各 2 局、ood 两侧各 2 局：1 任务 × 2 数据集 × 2 局 × 2 侧 = 8
    assert [a.side for a in world.gen_calls] == ["O", "H"]
    assert all(a.tier == "xhard0" and a.workers == 2 and a.dev_smoke for a in world.gen_calls)
    assert [side for _c, _e, side in world.mover_calls] == ["H_old", "H_new"]
    produced = sum(len((tmp_path / "out" / ds / side / "identities.jsonl").read_text().splitlines())
                   for ds, sides in (("hard-verify", ("O", "H")), ("ood", ("H_old", "H_new"))) for side in sides)
    assert produced == 8
    # H_old：旧仓解释器、PYTHONPATH 指向 --h-old-src、generate_h5 断言 robomme_hard 在其下
    cmd, env, _ = world.mover_calls[0]
    old_src = str((tmp_path / "old-snapshot" / "src").resolve())
    assert cmd[0] == sys.executable and env["PYTHONPATH"] == old_src
    assert cmd[cmd.index("--expect-src") + 1] == old_src and cmd[cmd.index("--src-root") + 1] == str(Path(old_src).parent)
    # 身份以 builder 解析为准：新树两个数据集 + 旧树 ood 各解析一次
    assert [(c[1] == old_src, c[2]) for c in world.resolve_calls] == [(False, "hard-verify"), (False, "ood"), (True, "ood")]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["ok"] is True and summary["generations"] == 8


def test_两侧同失败单列且不计入通过(hp, tmp_path, monkeypatch, capsys):
    World(tmp_path, {("O", 900): "fail", ("H", 900): "fail"}).install(hp, monkeypatch)
    assert hp.main(_argv(tmp_path)) == 1
    assert _last(capsys) == "PARITY_GEN_SMOKE=FAIL identities=4 sides=2 compared=3 both_fail=1 tol_over=0"


def test_只有一侧有h5即FAIL(hp, tmp_path, monkeypatch, capsys):
    World(tmp_path, {("H_new", 1701): "fail"}).install(hp, monkeypatch)
    assert hp.main(_argv(tmp_path)) == 1
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == ("PARITY_GEN_SMOKE=FAIL identities=4 sides=2 compared=3 both_fail=0 "
                                            "tol_over=0")
    assert "one_side=1" in out


def test_超容差计入tol_over(hp, tmp_path, monkeypatch, capsys):
    World(tmp_path, {("H", 901): "state"}).install(hp, monkeypatch)
    assert hp.main(_argv(tmp_path)) == 1
    assert _last(capsys) == "PARITY_GEN_SMOKE=FAIL identities=4 sides=2 compared=4 both_fail=0 tol_over=1"


def test_环境包绑定不对即FAIL(hp, tmp_path, monkeypatch, capsys):
    World(tmp_path, {("H_new", 1700): "badmod"}).install(hp, monkeypatch)
    assert hp.main(_argv(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "binding_bad=1" in out and out.strip().splitlines()[-1].startswith("PARITY_GEN_SMOKE=FAIL ")


def test_新旧两棵树解析出的ood身份不同即不生成(hp, tmp_path, monkeypatch):
    other = [dict(r, seed=r["seed"] + 1) for r in _ood_rows()]
    world = World(tmp_path, old_rows=other).install(hp, monkeypatch)
    with pytest.raises(hp.ParityError, match="身份不同"):
        hp.main(_argv(tmp_path))
    assert world.gen_calls == [] and world.mover_calls == []


# ── 参数闸门（都在任何生成之前）─────────────────────────────────────────


@pytest.mark.parametrize("over,needle", [
    ({"--hard-verify-episodes": "0:3"}, "硬上限"),
    ({"--ood-sides": "H_new,H_old"}, "只接受"),
    ({"--hard-verify-sides": "O"}, "只接受"),
    ({"--task": "NoSuchTask"}, "未知任务"),
])
def test_参数闸门在生成之前拒跑(hp, tmp_path, monkeypatch, over, needle):
    world = World(tmp_path).install(hp, monkeypatch)
    with pytest.raises(hp.ParityError, match=needle):
        hp.main(_argv(tmp_path, **over))
    assert world.resolve_calls == [] and world.gen_calls == [] and world.mover_calls == []


def test_旧代码树不含robomme_hard或输出目录非空即拒(hp, tmp_path, monkeypatch):
    World(tmp_path).install(hp, monkeypatch)
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(hp.ParityError, match="--h-old-src"):
        hp.main(_argv(tmp_path, **{"--h-old-src": str(bare)}))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("")
    with pytest.raises(hp.ParityError, match="非空"):
        hp.main(_argv(tmp_path))


def test_builder解析进程的robomme_hard不在指定源码树即拒(hp, tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    payload = {"robomme_hard_file": "/elsewhere/robomme_hard/__init__.py", "episodes": 12, "rows": []}
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="RESOLVE_JSON " + json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(hp, "subprocess", types.SimpleNamespace(run=fake_run))
    with pytest.raises(hp.ParityError, match="不在"):
        hp.resolve_builder_identities(sys.executable, src, "ood", TASK, [0, 1])
    cmd, kw = calls[0]
    assert kw["env"]["PYTHONPATH"] == str(src) and cmd[-3:] == ["ood", TASK, "[0, 1]"]
    payload["robomme_hard_file"] = str(src / "robomme_hard" / "__init__.py")
    payload["rows"] = [{"task": TASK, "tier": "xhard1", "seed": 1, "builder_episode": 0}]
    assert hp.resolve_builder_identities(sys.executable, src, "ood", TASK, [0]) == payload["rows"]
