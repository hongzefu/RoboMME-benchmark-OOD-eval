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
    row.update(model='groundsg' if model.startswith('groundsg-') else model, policy_label=model, policy_seed=7, attempt=attempt,
               status='success' if success else 'fail', task_success=success,
               infra=False, run_blocked=False, budget_exhausted=False, t_start=100, t_end=100 + wall)
    row.update(max_steps=1800, strict_cap=True, exec_steps=10)
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


@pytest.mark.parametrize('models', [(), ('smvla',)])
def test_incomplete_models_never_stage_pass(tmp_path, models):
    ident = identity(0)
    write_episode(tmp_path, 'smvla', ident)
    result = report.build_report(tmp_path, [ident], 'A', models=models)
    assert not result['ok']
    assert result['verdicts'][-1].startswith('STAGE_TIMING=FAIL')


@pytest.mark.parametrize('document', ['result', 'marker', 'index'])
@pytest.mark.parametrize('bad', [None, [], 5])
def test_non_object_fail_closed(tmp_path, document, bad):
    ident = identity(0)
    path, index, marker = write_episode(tmp_path, 'smvla', ident)
    {'result': path, 'marker': marker, 'index': index}[document].write_text(json.dumps(bad) + '\n')
    result = report.build_report(tmp_path, [ident], 'A', models=('smvla',))
    assert not result['ok'] and result['errors']


def test_model_tampering_and_error_exception(tmp_path):
    ident = identity(0)
    paths = []
    for model in report.MODELS:
        paths.append(write_episode(tmp_path, model, ident))
    path, index, marker = paths[0]
    row = json.loads(path.read_text())
    row['model'] = 'astra'
    path.write_text(json.dumps(row))
    index.write_text(json.dumps(row) + '\n')
    assert not report.build_report(tmp_path, [ident], 'A')['ok']
    row.update(model=report.MODELS[0], status='error', task_success=0, error='初始化失败，无帧')
    accepted = json.loads(marker.read_text())
    accepted['status'] = 'error'
    marker.write_text(json.dumps(accepted))
    path.write_text(json.dumps(row))
    index.write_text(json.dumps(row) + '\n')
    result = report.build_report(tmp_path, [ident], 'A')
    assert result['ok'] and result['coverage'][0]['no_frame_error'] == 1
    (path.parent / 'trace.jsonl').write_text(json.dumps({'frames_recorded': 1}) + '\n')
    assert not report.build_report(tmp_path, [ident], 'A')['ok']


@pytest.mark.parametrize('damage', ['max_steps', 'exec_steps', 'strict_cap', 'trace'])
def test_stage_cap_guard(tmp_path, damage):
    ident = identity(0)
    for model in report.MODELS:
        path, index, _ = write_episode(tmp_path, model, ident)
        row = json.loads(path.read_text())
        if damage == 'trace':
            (path.parent / 'trace.jsonl').write_text(json.dumps({'kind': 'header', 'effective_cap': 1801}) + '\n' +
                                                    json.dumps({'kind': 'end'}) + '\n')
        else:
            row[damage] = {'max_steps': 1799, 'exec_steps': 1801, 'strict_cap': False}[damage]
            path.write_text(json.dumps(row))
            index.write_text(json.dumps(row) + '\n')
    result = report.build_report(tmp_path, [ident], 'A')
    assert not result['ok']
    assert any(line.startswith('EVAL_REPORT=FAIL') for line in result['verdicts'])


def test_effective_cap_precedence_and_media_disabled(tmp_path):
    ident = identity(0)
    for model in report.MODELS:
        path, _, _ = write_episode(tmp_path, model, ident)
        (path.parent / 'trace.jsonl').write_text(json.dumps({'kind': 'header', 'max_steps': 1840, 'effective_cap': 1800}) + '\n' +
                                                json.dumps({'kind': 'end'}) + '\n')
    result = report.build_report(tmp_path, [ident], 'A')
    assert result['ok']
    assert any(line.startswith('OFFICIAL_MEDIA=UNVERIFIED') for line in result['verdicts'])


def recovery_fixture(root):
    checker = report.helper('fixture_media', 'media/official_media_check.py')
    ident = identity(0)
    raw = root / 'rollouts/pp/ood/seed7/raw/VideoUnmask_ep0_xhard1'
    raw.mkdir(parents=True)
    row = {k: v for k, v in ident.items() if k != 'builder_episode'}
    row.update(model='pp', policy_seed=7, attempt=1, status='success', task_success=1)
    (raw / 'result.json').write_text(json.dumps(row))
    trace_ident = {**ident, 'attempt': 1, 'policy_seed': 7}
    (raw / 'trace.jsonl').write_text(json.dumps({'kind': 'header', 'identity': trace_ident, 'policy_seed': 7}) + '\n' +
                                    json.dumps({'kind': 'end', 'status': 'success', 'frames_recorded': 2}) + '\n')
    for name in ('arrays.npz', 'front.mkv', 'wrist.mkv', 'frames-front.jsonl', 'frames-wrist.jsonl', 'meta.json'):
        (raw / name).write_bytes(b'fixture')
    rec = raw / 'recovered-video'
    rec.mkdir()
    video = rec / 'VideoUnmask_ep0_success_goal_xhard1.mp4'
    video.write_bytes(b'video')
    fp = lambda p: {'size': p.stat().st_size, 'sha256': checker.sha256_file(p)}
    render = {'identity': trace_ident, 'status': 'success', 'policy_seed': 7, 'frames_recorded': 2,
              'source_trace': fp(raw / 'trace.jsonl'), 'source_arrays': fp(raw / 'arrays.npz'),
              'source_media': {'streams': {name: fp(raw / name) for name in ('front.mkv', 'wrist.mkv')},
                               'index': {name: fp(raw / name) for name in ('frames-front.jsonl', 'frames-wrist.jsonl', 'meta.json')}},
              'out_rel': video.name, 'output_fingerprint': fp(video)}
    proof = {'model': 'pp', 'task_success': 1, 'result_sha256': checker.sha256_file(raw / 'result.json'),
             'render': render, 'video_sha256': checker.sha256_file(video), 'video_bytes': video.stat().st_size,
             'original_files': {str(Path('/old/copy') / p.relative_to(root)): {'bytes': p.stat().st_size,
                                                                          'sha256': checker.sha256_file(p)}
                                for p in raw.iterdir() if p.is_file()}}
    index = root / 'rollouts/pp/ood/seed7/results.jsonl'
    index.write_text(json.dumps(row) + '\n')
    proof['original_files'][str(Path('/old/copy') / index.relative_to(root))] = {'bytes': index.stat().st_size,
                                                                           'sha256': checker.sha256_file(index)}
    (rec / 'render.json').write_text(json.dumps(render))
    (rec / 'recovery.json').write_text(json.dumps(proof))
    return raw, row, checker, index


@pytest.mark.parametrize('damage', ['none', 'result', 'video', 'trace', 'flag', 'history'])
def test_recovery_bound_inputs_and_appended_history(tmp_path, monkeypatch, damage):
    raw, row, checker, index = recovery_fixture(tmp_path)
    monkeypatch.setattr(checker, 'decoded_frames', lambda ff, path: 2)
    with index.open('a') as stream:
        stream.write(json.dumps({'stage': 'B'}) + '\n')
    if damage in ('result', 'trace'):
        (raw / f'{damage}.json' if damage == 'result' else raw / 'trace.jsonl').write_bytes(b'tampered')
    elif damage == 'video':
        next((raw / 'recovered-video').glob('*.mp4')).write_bytes(b'tampered')
    elif damage == 'flag':
        (raw / 'recovered-video/recovery.json').write_text(json.dumps({'original_unchanged': True}))
    elif damage == 'history':
        index.write_text('tampered\n')
    if damage == 'none':
        assert report.verify_recovery(raw, tmp_path, row, 'fake', checker)['status'] == 'pass'
    else:
        with pytest.raises((ValueError, KeyError)):
            report.verify_recovery(raw, tmp_path, row, 'fake', checker)


@pytest.mark.parametrize('media_status', ['pass', 'fail'])
def test_standard_media_white_list(tmp_path, monkeypatch, media_status):
    from types import SimpleNamespace
    ident = identity(0)
    other = identity(1)
    for model in report.MODELS:
        write_episode(tmp_path, model, ident)
        write_episode(tmp_path, model, other)
    calls = []
    def verify(raw, run_dir, ff, **kwargs):
        calls.append((raw, run_dir, kwargs))
        return {'status': media_status, 'reasons': []}
    original = report.helper
    monkeypatch.setattr(report, 'helper', lambda name, path: SimpleNamespace(verify_new_episode=verify)
                        if path.startswith('media/') else original(name, path))
    result = report.build_report(tmp_path, [ident], 'A', check_media=True, ffmpeg='fixture')
    assert len(calls) == 6
    assert result['ok'] == (media_status == 'pass')
    assert all(c[2] == {'dataset': 'ood', 'seed': 7, 'expected_policy_seed': 7} for c in calls)


@pytest.mark.parametrize('damage', ['identity', 'attempt', 'policy_seed', 'end', 'decoded'])
def test_recovery_semantic_binding(tmp_path, monkeypatch, damage):
    raw, row, checker, _ = recovery_fixture(tmp_path)
    monkeypatch.setattr(checker, 'decoded_frames', lambda ff, path: 1 if damage == 'decoded' else 2)
    if damage != 'decoded':
        rows = checker.read_rows(raw / 'trace.jsonl')
        if damage == 'end':
            rows[-1]['status'] = 'fail'
        else:
            key = {'identity': 'task', 'attempt': 'attempt', 'policy_seed': 'policy_seed'}[damage]
            rows[0]['identity'][key] = 'wrong' if key == 'task' else 2
        (raw / 'trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        proof = report.object_json(raw / 'recovered-video/recovery.json')
        render = report.object_json(raw / 'recovered-video/render.json')
        render['identity'] = rows[0]['identity']
        render['source_trace'] = {'size': (raw / 'trace.jsonl').stat().st_size,
                                  'sha256': checker.sha256_file(raw / 'trace.jsonl')}
        proof['render'] = render
        for stored in proof['original_files']:
            if stored.endswith('/trace.jsonl'):
                proof['original_files'][stored] = {'bytes': (raw / 'trace.jsonl').stat().st_size,
                                                   'sha256': checker.sha256_file(raw / 'trace.jsonl')}
        (raw / 'recovered-video/render.json').write_text(json.dumps(render))
        (raw / 'recovered-video/recovery.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        report.verify_recovery(raw, tmp_path, row, 'fake', checker)
