"""GroundSG 三个变体的 policy_seed 两侧一致（判定行 POLICY_SEEDS；1006-rename-official-names-and-stage3-eval-plan.md
第二部分八.3）：新侧 ``groundsg_client`` 与原侧 ``official_hard_runner`` 都把种子传到 ``Args.model_seed`` 并按该值
``seed_everything``。

由 ``test_groundsg_memer.py`` 拆出（原侧 ``official_hard_runner`` 只在 dev 侧）。
"""
from __future__ import annotations

import random

import numpy as np
import pytest

import groundsg_fakes_dev as F


def od():
    return F.official_defs()


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("IMAGE_MAX_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "FPS_MAX_FRAMES", "USE_HF", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_policy_seed_reaches_args_and_seed_everything(tmp_path, clean_env, monkeypatch, seed):
    """0／7／42 传到 Args.model_seed；QwenVL／MemER 构造预测器前按该值 seed_everything，Oracle 不调；两侧都一样。"""
    calls = []
    for mod in {id(m): m for m in (F.groundsg_client().official_defs, F.official_hard_runner().official_defs)}.values():
        real = mod.seed_everything

        def spy(s, _real=real):
            calls.append(s)
            return _real(s)

        monkeypatch.setattr(mod, "seed_everything", spy)
    for variant in F.VARIANTS:
        calls.clear()
        n = F.NewSide(variant, 60, tmp_path / variant / "n", F.World(), policy_seed=seed)
        o = F.OrigSide(variant, 60, tmp_path / variant / "o", F.World(), policy_seed=seed)
        assert n.ctx["args"].model_seed == o.ctx["args"].model_seed == seed
        assert n.ctx["policy_seed"] == o.ctx["policy_seed"] == seed
        assert calls == ([] if variant == F.ORACLE else [seed, seed]), (variant, calls)
    # seed_everything 真实效果：random、numpy、torch 三处都按该值重置（手写对照：同种子的新生成器）
    res = od().seed_everything(seed)
    a, b = random.random(), np.random.random()
    assert a == random.Random(seed).random() and b == np.random.RandomState(seed).random_sample()
    assert res == {"seed": seed, "random": True, "numpy": True, "torch": True}
    import torch

    t = torch.rand(1).item()
    torch.manual_seed(seed)
    assert t == torch.rand(1).item()
    print(f"POLICY_SEEDS=PASS route=groundsg seed={seed} variants={len(F.VARIANTS)}")
