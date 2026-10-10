"""接续控制器以伪命令验证序列化及失败封闭，不启动仿真。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('continuation', Path(__file__).parents[1] / 'dev-scripts/gl/continue_full_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, command=None):
    module.ROOT = tmp_path
    identities = tmp_path / 'identities/stage-A.jsonl'
    identities.parent.mkdir()
    identity = {'key': 'Task_xhard1_1', 'dataset': 'ood', 'task': 'Task',
        'episode': 0, 'builder_episode': 0, 'tier': 'xhard1', 'seed': 1, 'candidate': 0,
        'source_episode': None, 'spec_sha256': 'a' * 64}
    for s in 'ABCD':
        (identities.parent / f'stage-{s}.jsonl').write_text(json.dumps(identity) + '\n')
    c = {'controller_root': str(tmp_path / 'controller'), 'carrier': '/carrier',
         'report_code': '/report', 'python': '/python', 'ffmpeg': '/ffmpeg', 'report_timeout_s': 3600,
         'models': dict(module.MODELS), 'priority': ['smvla', 'pp', 'oracle', 'perceptual-framesamp-modul', 'qwen', 'memer'],
         'stages': {s: {'identities': str(identities.parent / f'stage-{s}.jsonl'),
                       'out': str(tmp_path / f'seed{module.SEEDS[s]}'), 'seed': module.SEEDS[s],
                       'budget': str(tmp_path / f'budget/stage-{s}.jsonl'), 'caps': module.CAPS[s]}
                    for s in 'ABCD'},
         'seats': [{'job': str(i), 'slot': i} for i in range(4)]}
    c['carrier'] = str(tmp_path / 'run-model-r2.sh')
    return module.Controller(c, command or (lambda *a, **k: SimpleNamespace(stdout='RUNNING', returncode=0)))


def accept(controller, stage='A', status='fail', model='smvla'):
    out = Path(controller.c['stages'][stage]['out'])
    label = module.MODELS[model]
    raw = out / 'rollouts' / label / 'result.json'
    raw.parent.mkdir(parents=True)
    row = json.loads(Path(controller.c['stages'][stage]['identities']).read_text())
    row.update(attempt=1, status=status, policy_seed=7, policy_label=label,
               model='groundsg' if label.startswith('groundsg-') else label,
               policy_variant=label.removeprefix('groundsg-'), task_success=int(status == 'success'),
               infra=False, run_blocked=False, budget_exhausted=False)
    raw.write_text(json.dumps(row))
    marker = out / f'queue/{label}/seed7/accepted/ood__Task_xhard1_1.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({**row, 'result': f'rollouts/{label}/result.json'}))
    return raw


def test_normal_failure_accepted_zero_persisted(tmp_path):
    c = fixture(tmp_path)
    accept(c)
    assert c.remaining('A', 'smvla') == []
    c.save()
    loaded = module.read(c.path)
    assert loaded['missing'] == loaded['media_failed'] == loaded['report_failed'] == 0


def test_missing_result_rejected(tmp_path):
    c = fixture(tmp_path)
    accept(c).unlink()
    with pytest.raises(FileNotFoundError):
        c.remaining('A', 'smvla')


def test_report_roundtrip_advances(tmp_path):
    def command(argv, **kw):
        if '--json' in argv:
            Path(argv[argv.index('--json') + 1]).write_text(json.dumps({
                'stage': 'A', 'ok': True, 'errors': [], 'verdicts': [name + '=PASS' for name in
                    ('EVAL_COVERAGE', 'STAGE_MODELS', 'EVAL_REPORT', 'STAGE_TIMING',
                     'OFFICIAL_MEDIA_INPUTS', 'OFFICIAL_MEDIA')]}))
        return SimpleNamespace(stdout='', returncode=0)
    c = fixture(tmp_path, lambda *a, **kw: SimpleNamespace(stdout='', returncode=1))
    for model in module.MODELS:
        accept(c, model=model)
    output = c.root / 'report-A.json'
    report = json.loads((Path(__file__).parents[1] / 'docs/validation/full-eval-20261010/records/smoke-report-strict-media.json').read_text())
    for row in report['media']:
        row['key'] = 'Task_xhard1_1'
    output.write_text(json.dumps(report))
    log = c.root / 'report.log'
    log.write_text('EXIT_CODE=0\n')
    c.state['report_task'] = {'session': 'report', 'log': str(log), 'started': 0}
    c.tick()
    assert module.read(c.path)['stage'] == 'B'


def test_report_crash_detected(tmp_path):
    def command(argv, **kw):
        raise RuntimeError('报告崩溃')
    c = fixture(tmp_path, command)
    for model in module.MODELS:
        accept(c, model=model)
    with pytest.raises(RuntimeError, match='报告崩溃'):
        c.tick()
    assert c.state['stage'] == 'A'


def test_missing_report_fields_rejected(tmp_path):
    def command(argv, **kw):
        Path(argv[argv.index('--json') + 1]).write_text('{}')
    c = fixture(tmp_path, lambda *a, **kw: SimpleNamespace(stdout='', returncode=1))
    log = c.root / 'report.log'
    log.write_text('EXIT_CODE=0\n')
    (c.root / 'report-A.json').write_text('{}')
    c.state['report_task'] = {'session': 'report', 'log': str(log), 'started': 0}
    with pytest.raises(KeyError):
        c.report('A')


def test_live_session_adopted_without_relaunch(tmp_path):
    calls = []
    def command(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(stdout='RUNNING', returncode=0)
    c = fixture(tmp_path, command)
    for seat in c.state['seats']:
        seat.update(session=f"fe-controller-A-smvla-{seat['slot']}-1", model='smvla',
                    log=str(c.root / 'unused'), stage='A')
    c.tick()
    assert not any('new-session' in x for x in calls)


@pytest.mark.parametrize('state', ['COMPLETED', 'CANCELLED', ''])
def test_only_timeout_can_replace(tmp_path, state):
    def command(argv, **kw):
        return SimpleNamespace(stdout='' if argv[0] == 'squeue' else state, returncode=0)
    c = fixture(tmp_path, command)
    with pytest.raises(ValueError):
        c.replace_expired(c.state['seats'][0])


def test_uncertain_submission_never_repeated(tmp_path):
    calls = []
    def command(argv, **kw):
        calls.append(argv[0])
        return SimpleNamespace(stdout={'squeue': '', 'sacct': 'TIMEOUT|', 'sbatch': 'unknown'}[argv[0]], returncode=0)
    c = fixture(tmp_path, command)
    with pytest.raises(ValueError):
        c.replace_expired(c.state['seats'][0])
    with pytest.raises(ValueError):
        c.replace_expired(c.state['seats'][0])
    assert calls.count('sbatch') == 1


def test_four_seat_quota(tmp_path):
    c = fixture(tmp_path)
    c.c['seats'].append({'job': '5', 'slot': 4})
    with pytest.raises(ValueError, match='四席'):
        module.Controller(c.c)


def test_environment_guard_finite_recovery(tmp_path):
    calls = []
    def command(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(stdout='RUNNING', returncode=1 if 'has-session' in argv else 0)
    c = fixture(tmp_path, command)
    log = tmp_path / 'worker.log'
    log.write_text('RUN_BLOCKED reason=env_build policy=smvla seat=a key=Task_xhard1_1 attempt=2 settled=1\nEXIT_CODE=3\n')
    seat = c.state['seats'][0]
    seat.update(session='fe-controller-A-smvla-0-1', model='smvla', log=str(log), stage='A',
                seat_name='a', onlykey='Task_xhard1_1', recoveries={'Task_xhard1_1': 18})
    ledger = tmp_path / 'seed7/seats/smvla/seed7/a/smvla.ledger.jsonl'
    ledger.parent.mkdir(parents=True)
    ledger.write_text('\n'.join(json.dumps(r) for r in [
        {'kind': 'attempt_start', 'key': 'Task_xhard1_1', 'attempt_no': 2, 'attempt_id': 'aid', 'budget_rid': 'rid'},
        {'kind': 'attempt_end', 'attempt_id': 'aid', 'infra': True, 'infra_reason': 'env_build',
         'status': 'error', 'budget_exhausted': False}]))
    budget = tmp_path / 'budget/stage-A.jsonl'
    budget.parent.mkdir()
    budget.write_text(json.dumps({'kind': 'commit', 'rid': 'rid', 'infra': True, 'status': 'error'}))
    c.tick()
    assert seat['recoveries']['Task_xhard1_1'] == 19
    assert seat['onlykey'] == 'Task_xhard1_1'
    seat['seat_name'] = 'a'
    Path(seat['log']).write_text(log.read_text())
    with pytest.raises(ValueError, match='上限'):
        c.tick()


def test_missing_exit_refuses_relaunch(tmp_path):
    c = fixture(tmp_path, lambda argv, **k: SimpleNamespace(stdout='RUNNING', returncode=1 if 'has-session' in argv else 0))
    log = tmp_path / 'worker.log'
    log.write_text('')
    c.state['seats'][0].update(session='fe-controller-A-smvla-0-1', model='smvla', log=str(log), stage='A')
    with pytest.raises(ValueError, match='退出码'):
        c.tick()


def test_expired_nonzero_replaces_once_waits_then_only_remaining(tmp_path):
    calls, ready, sessions = [], [False], set()
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'tmux':
            if 'new-session' in argv:
                sessions.add(argv[argv.index('-s') + 1])
            return SimpleNamespace(stdout='', returncode=(0 if argv[-1].removeprefix('=') in sessions else 1)
                                   if 'has-session' in argv else 0)
        if argv[0] == 'squeue':
            job = argv[argv.index('-j') + 1]
            value = '' if job == '0' else ('RUNNING' if job != '100' or ready[0] else 'PENDING')
            return SimpleNamespace(stdout=value, returncode=0)
        if argv[0] == 'sacct':
            return SimpleNamespace(stdout='TIMEOUT|\n', returncode=0)
        if argv[0] == 'sbatch':
            return SimpleNamespace(stdout='100', returncode=0)
        raise AssertionError(argv)
    c = fixture(tmp_path, command)
    accept(c)
    identity = json.loads(Path(c.c['stages']['A']['identities']).read_text())
    identity.update(key='Task_xhard1_2', seed=2, episode=1, builder_episode=1, candidate=1)
    with Path(c.c['stages']['A']['identities']).open('a') as f:
        f.write(json.dumps(identity) + '\n')
    log = c.root / 'expired.log'
    log.write_text('EXIT_CODE=143\n')
    c.state['seats'][0].update(session='fe-controller-A-smvla-0-1', model='smvla',
                              log=str(log), stage='A', onlykey='Task_xhard1_2')
    c.tick()
    assert c.state['seats'][0]['job'] == '100'
    assert c.state['expired_jobs'] == ['0']
    assert not c.state['seats'][0].get('session')
    ready[0] = True
    c.tick()
    assert all(s['onlykey'] == 'Task_xhard1_2' for s in c.state['seats'] if s.get('session') and s['model'] == 'smvla')
    assert sum(x[0] == 'sbatch' for x in calls) == 1
    assert len(c.state['seats']) == 4
    launch = [x for x in calls if 'new-session' in x][-1][-1]
    assert 'expired-jobs.txt' in launch
    assert (c.root / 'expired-jobs.txt').read_text().strip() == '0'


def test_whitespace_sacct_never_replaces(tmp_path):
    def command(argv, **kwargs):
        assert argv[0] != 'sbatch'
        return SimpleNamespace(stdout='\n   \n', returncode=0)
    c = fixture(tmp_path, command)
    with pytest.raises(ValueError, match='TIMEOUT'):
        c.replace_expired(c.state['seats'][0])


def test_report_renews_all_expired_then_waits(tmp_path):
    count = [100]
    def command(argv, **kwargs):
        if argv[0] == 'squeue':
            return SimpleNamespace(stdout='', returncode=0)
        if argv[0] == 'sacct':
            return SimpleNamespace(stdout='TIMEOUT|', returncode=0)
        if argv[0] == 'sbatch':
            count[0] += 1
            return SimpleNamespace(stdout=str(count[0]), returncode=0)
        raise AssertionError(argv)
    c = fixture(tmp_path, command)
    assert c.report('A') is False
    assert len(c.state['expired_jobs']) == 4
    assert len({s['job'] for s in c.state['seats']}) == 4
    assert not c.state.get('report_task')


@pytest.mark.parametrize('field', ['seed', 'identities', 'caps', 'budget'])
def test_stage_configuration_cannot_change(tmp_path, field):
    c = fixture(tmp_path)
    c.c['stages']['A'][field] = 99 if field == 'seed' else ([] if field == 'caps' else '/wrong')
    with pytest.raises(ValueError):
        module.Controller(c.c)


def test_config_digest_rejects_changed_priority(tmp_path):
    c = fixture(tmp_path)
    c.save()
    c.c['priority'].reverse()
    with pytest.raises(ValueError, match='指纹'):
        module.Controller(c.c)


def test_unknown_session_rejected(tmp_path):
    c = fixture(tmp_path)
    c.state['seats'][0].update(session='users-session', log=str(c.root / 'x'), model='smvla', stage='A')
    with pytest.raises(ValueError, match='清单'):
        c.validate_session(c.state['seats'][0])


def test_environment_guard_missing_shared_commit_rejected(tmp_path):
    c = fixture(tmp_path)
    seat = c.state['seats'][0]
    seat.update(stage='A', model='smvla', onlykey='Task_xhard1_1', seat_name='a')
    ledger = tmp_path / 'seed7/seats/smvla/seed7/a/smvla.ledger.jsonl'
    ledger.parent.mkdir(parents=True)
    ledger.write_text('\n'.join(json.dumps(r) for r in [
        {'kind': 'attempt_start', 'key': 'Task_xhard1_1', 'attempt_no': 2, 'attempt_id': 'aid', 'budget_rid': 'rid'},
        {'kind': 'attempt_end', 'attempt_id': 'aid', 'infra': True, 'infra_reason': 'env_build',
         'status': 'error', 'budget_exhausted': False}]))
    budget = tmp_path / 'budget/stage-A.jsonl'
    budget.parent.mkdir()
    budget.write_text('')
    with pytest.raises(ValueError, match='共享预算'):
        c.env_build_settled(seat, 'RUN_BLOCKED reason=env_build policy=smvla seat=a key=Task_xhard1_1 attempt=2 settled=1')


def real_report_fixture():
    # 直接读取已提交的真实六模型报告结构；不复制实测进新增源码。
    doc = json.loads((Path(__file__).parents[1] / 'docs/validation/full-eval-20261010/records/smoke-report-strict-media.json').read_text())
    for row in doc['media']:
        row['key'] = 'Task_xhard1_1'
    return doc


def test_report_expiry_renew_launch_serialize_read_advance(tmp_path):
    running, count, calls = [False], [100], []
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'squeue':
            job = int(argv[argv.index('-j') + 1])
            return SimpleNamespace(stdout='' if job < 100 else ('RUNNING' if running[0] else 'PENDING'), returncode=0)
        if argv[0] == 'sacct':
            return SimpleNamespace(stdout='TIMEOUT|', returncode=0)
        if argv[0] == 'sbatch':
            count[0] += 1
            return SimpleNamespace(stdout=str(count[0]), returncode=0)
        if argv[0] == 'tmux':
            if 'new-session' in argv:
                assert 'srun' in argv[-1] and '--gpu_cmode=shared' in argv[-1] and '--check-media' in argv[-1]
                (tmp_path / 'controller/report-A.json').write_text(json.dumps(real_report_fixture()))
                (tmp_path / 'controller/fe-controller-report-A.log').write_text('EXIT_CODE=0\n')
            return SimpleNamespace(stdout='', returncode=1 if 'has-session' in argv else 0)
        raise AssertionError(argv)
    c = fixture(tmp_path, command)
    for model in module.MODELS:
        accept(c, model=model)
    c.tick()
    assert c.state['stage'] == 'A' and count[0] == 104
    running[0] = True
    c.tick()
    assert c.state['report_task']['session'] == 'fe-controller-report-A'
    c.tick()
    assert module.read(c.path)['stage'] == 'B'
    report = module.read(c.root / 'report-A.json')
    assert all(row['missing'] == row['invalid'] == row['no_frame_error'] == 0 for row in report['coverage'])
    assert sum(x[0] == 'sbatch' for x in calls) == 4


@pytest.mark.parametrize('mutation', ['missing_model', 'duplicate_media', 'missing_zero', 'wrong_seed', 'fake_pass'])
def test_true_report_contract_rejects_corruption(tmp_path, mutation):
    c = fixture(tmp_path, lambda *a, **k: SimpleNamespace(stdout='', returncode=1))
    doc = real_report_fixture()
    if mutation == 'missing_model':
        doc['coverage'].pop()
    elif mutation == 'duplicate_media':
        doc['media'][-1] = doc['media'][0]
    elif mutation == 'missing_zero':
        del doc['coverage'][0]['missing']
    elif mutation == 'wrong_seed':
        doc['policy_seed'] = 42
    else:
        doc['verdicts'][0] = 'EVAL_COVERAGE=UNKNOWN=PASS'
    (c.root / 'report-A.json').write_text(json.dumps(doc))
    log = c.root / 'report.log'
    log.write_text('EXIT_CODE=0\n')
    c.state['report_task'] = {'session': 'fe-controller-report-A', 'log': str(log), 'started': 0}
    with pytest.raises((ValueError, KeyError)):
        c.report('A')


def test_invalid_job_id_requires_sacct_timeout(tmp_path):
    calls = []
    def command(argv, **kwargs):
        calls.append(argv[0])
        if argv[0] == 'squeue':
            assert kwargs['check'] is False
            return SimpleNamespace(stdout='', stderr='squeue: error: Invalid job id specified\n', returncode=1)
        if argv[0] == 'sacct':
            return SimpleNamespace(stdout='TIMEOUT|', returncode=0)
        return SimpleNamespace(stdout='100', returncode=0)
    c = fixture(tmp_path, command)
    assert c.replace_expired(c.state['seats'][0])
    assert calls == ['squeue', 'sacct', 'sbatch']


def test_other_squeue_failure_stops_without_submission(tmp_path):
    def command(argv, **kwargs):
        assert argv[0] == 'squeue'
        return SimpleNamespace(stdout='', stderr='controller communication failure', returncode=1)
    c = fixture(tmp_path, command)
    with pytest.raises(ValueError, match='查询失败'):
        c.replace_expired(c.state['seats'][0])
