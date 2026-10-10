"""阶段身份筛选的纯 CPU 契约验证。"""
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'dev-scripts/gl'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('select_stage_identities', TOOLS / 'select_stage_identities.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def identity(task='VideoUnmask', tier='xhard1', ep=0, seed=None):
    seed = ep if seed is None else seed
    return dict(dataset='ood', task=task, episode=ep, builder_episode=ep, tier=tier, seed=seed,
                candidate=ep, source_episode=None, spec_sha256='a' * 64, key=f'{task}_{tier}_{seed}')


@pytest.mark.parametrize('count', [2, 3, 4, 5])
def test_rotation_and_complement(count):
    rows = [identity(tier=f'xhard{tier + 1}', ep=tier * 4 + n)
            for tier in range(count) for n in range(4)]
    a = selector.select_rows(rows, 'A')
    expected = {tier * 4 + n for n in range(4) for tier in range(count)
                if n * count + tier < 5}
    assert {r['episode'] for r in a} == expected
    b = selector.select_rows(rows, 'B', minus=a)
    assert len(a) == 5 and len(a) + len(b) == len(rows)
    assert not {r['key'] for r in a} & {r['key'] for r in b}
    assert selector.select_rows(rows, 'C') == rows == selector.select_rows(rows, 'D')


def test_exclusion_and_invalid():
    rows = [identity(ep=n) for n in range(6)] + [identity('MoveCube'), identity('InsertPeg')]
    assert len(selector.select_rows(rows, 'C')) == 6
    for bad in ([rows[0], rows[0]], [{**rows[0], 'dataset': 'hard-verify'}],
                [{k: v for k, v in rows[0].items() if k != 'candidate'}]):
        with pytest.raises(ValueError):
            selector.select_rows(bad, 'C')
    with pytest.raises(ValueError):
        selector.select_rows(rows, 'B', minus=rows[:4])
