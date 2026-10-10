"""真实序列化布局下核对 accepted、追加结果和阶段白名单。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dev-scripts/gl'))
import stage_report as report


def identity(ep):
    return dict(dataset='ood', task='VideoUnmask', episode=ep, builder_episode=ep, tier='xhard1', seed=ep,
                candidate=ep, source_episode=None, spec_sha256='a' * 64, key=f'VideoUnmask_xhard1_{ep}')


def write_episode(root, model, ident, *, success=0, wall=10, attempt=1):
    path = root / 'rollouts' / model / 'ood' / 'seed7' / ident['task'] / str(ident['episode']) / 'result.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in ident.items() if k != 'builder_episode'}
    row.update(model=model, policy_label=model, policy_seed=7, attempt=attempt,
               status='success' if success else 'fail', task_success=success,
               infra=False, run_blocked=False, budget_exhausted=False, t_start=100, t_end=100 + wall)
    path.write_text(json.dumps(row))
    index = root / 'rollouts' / model / 'ood' / 'seed7' / 'results.jsonl'
    with index.open('a') as f:
        f.write(json.dumps(row) + '\n')
    accepted = root / 'queue' / model / 'seed7' / 'accepted' / f"ood__{ident['key']}.json"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_text(json.dumps(dict(dataset='ood', key=ident['key'], attempt=attempt,
                                       status=row['status'], result=str(path))))
    return path, index, accepted


def test_six_models_and_stage_isolation(tmp_path):
    a, b = identity(0), identity(1)
    for model in report.MODELS:
        write_episode(tmp_path, model, a, success=1, wall=10)
        write_episode(tmp_path, model, b, wall=1000)
    result = report.build_report(tmp_path, [a], 'A')
    assert result['ok'] and result['episode_wall_sum_s'] == 60
    assert len(result['coverage']) == 6
    assert all(r['episode_success_rate'] == 1 for r in result['success'])
    assert all(r['wall_p95_s'] == 10 for r in result['timing'])
    assert 'STAGE_TIMING=PASS' in report.markdown(result)


@pytest.mark.parametrize('damage', ['missing', 'unknown', 'duplicate', 'attempt', 'index_mismatch'])
def test_fail_closed(tmp_path, damage):
    ident = identity(0)
    path, index, marker = write_episode(tmp_path, 'smvla', ident)
    row = json.loads(path.read_text())
    if damage == 'missing':
        marker.unlink()
    elif damage == 'unknown':
        del row['task_success']
        path.write_text(json.dumps(row))
    elif damage == 'duplicate':
        with index.open('a') as f:
            f.write(json.dumps(row) + '\n')
    elif damage == 'attempt':
        accepted = json.loads(marker.read_text())
        accepted['attempt'] = 2
        marker.write_text(json.dumps(accepted))
    else:
        row['t_end'] = 111
        index.write_text(json.dumps(row) + '\n')
    result = report.build_report(tmp_path, [ident], 'A', models=('smvla',))
    assert not result['ok'] and result['coverage'][0]['accepted'] == 0
    assert result['verdicts'][-1].startswith('STAGE_TIMING=FAIL')


def test_macro_mean_and_p95():
    rows = [dict(report_model='smvla', tier='xhard1', task=task, task_success=success,
                 t_start=1, t_end=1 + wall)
            for task, success, wall in [('A', 1, 1), ('A', 0, 2), ('A', 0, 3), ('B', 1, 100)]]
    rates = report.success_table(rows)[0]
    assert rates['episode_success_rate'] == .5
    assert rates['task_macro_success_rate'] == pytest.approx(2 / 3)
    assert report.timing_table(rows)[0]['wall_p95_s'] == 100


def test_sacct_steps(tmp_path):
    path = tmp_path / 'sacct.txt'
    path.write_text('JobIDRaw|Start|End|ElapsedRaw|AllocTRES\n'
                    '123|2026-10-10T01:00:00|2026-10-10T01:02:00|120|gres/gpu=1\n'
                    '123.0|2026-10-10T01:00:00|2026-10-10T01:01:00|60|cpu=1,gres/gpu=1\n')
    assert report.read_sacct(path)[0]['allocated_gpu_s'] == 60
    path.write_text('JobIDRaw|Start\n123.0|Unknown\n')
    with pytest.raises(ValueError):
        report.read_sacct(path)
