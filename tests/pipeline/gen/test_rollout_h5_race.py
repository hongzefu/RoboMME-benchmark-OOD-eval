"""run_batch 回读 h5 与搬运线程的竞态（2026-10-04 GL 对拍首跑实测：Mover 在 run_batch 列出 h5 之后、
打开之前按 match 删掉了文件，h5_facts 抛 FileNotFoundError 使整批生成以 runner_exit=1 退出）。

契约：文件在列目录与打开之间消失，与「列目录时已不在」同等处理——该局 h5 记 None、不回读事实、整批不崩；
文件仍在时照常回读事实。

前两条钉住 h5_facts 的两种基本行为；后两条经真实 ``run_batch`` 贯通：``subprocess.run`` 由
``gen_world.FakeRunner`` 冒充，``_rollout.h5_facts`` 换成「先删掉该文件、再调真函数」，复现列目录与打开之间的竞态。
反向用例在隔离副本里撤回 12.403 的保护（源码文本替换后另执行一份模块，不改主检出、不登记 sys.modules），
同一剧本必须整批抛 FileNotFoundError——证明正向用例确实依赖那处保护。"""
from __future__ import annotations

import importlib.util
import subprocess
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from gen_world import R, FakeRunner, freeze_file
from tests._support.dev_loaders import load_script, script_path

S = "StopCube"
#: 12.403 加的保护：回读时文件已不在则 h5 记 None（隔离副本里把它换回保护前的一行）
GUARDED = """            try:
                record.update(h5_facts(h5[0]))
            except FileNotFoundError:"""


@pytest.fixture
def rollout():
    return load_script("parity/_rollout.py")


def _write_h5(path: Path, steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ep = f.create_group("episode_0")
        for i in range(steps):
            g = ep.create_group(f"timestep_{i}")
            g.create_dataset("info/is_video_demo", data=np.bool_(i == 0))


def test_h5_facts_missing_file_raises(tmp_path, rollout):
    with pytest.raises(FileNotFoundError):
        rollout.h5_facts(tmp_path / "gone.h5")


def test_h5_facts_reads_exec_steps(tmp_path, rollout):
    p = tmp_path / "a.h5"
    _write_h5(p, steps=5)
    facts = rollout.h5_facts(p)
    assert facts["frames"] == 5 and facts["exec_steps"] == 4 and facts["bytes"] == p.stat().st_size


# ── 经真实 run_batch 的竞态贯通 ─────────────────────────────────────


@pytest.fixture
def batch(tmp_path):
    """xhard1 一批三局：StopCube 两局 + SwingXtimes 一局（经真实 _freeze 封存的小规格）。"""
    header, rows = freeze_file(tmp_path / "frozen" / "xhard1" / "specs.jsonl", "xhard1", {S: (2, 2), "SwingXtimes": (1, 1)})
    return header, [dict(r) for r in rows if r["selected"]]


def _racing_h5_facts(module, victim: str, deleted: list[str]):
    """``victim`` 局的 h5 在被打开前删掉（模拟 Mover 抢先搬走），再调同一模块的真 ``h5_facts``。"""
    real = module.h5_facts

    def facts(path):
        if Path(path).name.startswith(victim):
            Path(path).unlink()
            deleted.append(str(path))
        return real(path)

    return facts


def _run(module, batch, out: Path):
    header, rows = batch
    return module.run_batch(rows, header, out, 0, src_root=Path("/src-root"), workers=1, gpu="0", pkg="robomme_ood")


def test_run_batch_survives_h5_removed_between_list_and_open(monkeypatch, batch, tmp_path):
    header, rows = batch
    ep = next(r["episode"] for r in rows if r["task"] == S)
    victim = f"{S}_ep{ep}_"
    FakeRunner().install(monkeypatch)
    deleted: list[str] = []
    monkeypatch.setattr(R, "h5_facts", _racing_h5_facts(R, victim, deleted))
    recs = _run(R, batch, tmp_path / "out")
    assert len(deleted) == 1  # 竞态确实发生在那一局
    assert len(recs) == 3
    by = {(r["task"], r["episode"]): r for r in recs}
    lost = by[(S, ep)]
    assert lost["h5"] is None and "h5_sha256" not in lost and "exec_steps" not in lost
    others = [r for key, r in by.items() if key != (S, ep)]
    assert len(others) == 2
    for rec in others:
        h5 = Path(rec["h5"])
        assert rec["ok"] and h5.is_file()
        # FakeRunner 默认每局 5 帧、1 帧演示 → 执行步 4（替身剧本手写的值）
        assert (rec["frames"], rec["exec_steps"], rec["bytes"]) == (5, 4, h5.stat().st_size)


def _unguarded_copy(tmp_path: Path):
    """隔离副本：把 12.403 的 try/except 换回保护前的单行回读，另执行一份模块（不改主检出、不登记 sys.modules）。"""
    src = script_path("parity/_rollout.py").read_text(encoding="utf-8")
    assert src.count(GUARDED) == 1, "12.403 保护的源码形态变了，反向用例需同步"
    lines = src.split(GUARDED, 1)
    tail = lines[1].split("\n", 4)  # 去掉 except 体的三行注释与赋值
    assert "record[\"h5\"] = None" in "\n".join(tail[:4])
    patched = lines[0] + "            record.update(h5_facts(h5[0]))\n" + tail[4]
    path = tmp_path / "_rollout_unguarded.py"
    path.write_text(patched, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_t7_rollout_unguarded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # _common／robomme_ood 已在进程里，按模块缓存解析
    return module


def test_run_batch_without_12403_guard_crashes_whole_batch(monkeypatch, batch, tmp_path):
    """反向：撤回保护后同一剧本整批抛 FileNotFoundError（即 GL 首跑的 runner_exit=1）。"""
    header, rows = batch
    ep = next(r["episode"] for r in rows if r["task"] == S)
    victim = f"{S}_ep{ep}_"
    module = _unguarded_copy(tmp_path)
    fake = FakeRunner()
    monkeypatch.setattr(module, "subprocess", types.SimpleNamespace(run=fake.run, STDOUT=subprocess.STDOUT))
    deleted: list[str] = []
    monkeypatch.setattr(module, "h5_facts", _racing_h5_facts(module, victim, deleted))
    with pytest.raises(FileNotFoundError):
        _run(module, batch, tmp_path / "out")
    assert len(deleted) == 1
