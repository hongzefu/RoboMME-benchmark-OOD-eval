"""接续控制器以伪命令验证序列化及失败封闭，不启动仿真。"""
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('continuation', Path(__file__).parents[1] / 'dev-scripts/gl/continue_full_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, command=None):
    module.ROOT = tmp_path
    module.REPORT_CODE = tmp_path / 'runtime-code-r3/dev-scripts/gl/stage_report.py'
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
    c['carrier'] = str(tmp_path / 'run-model-r3.sh')
    c['report_code'] = str(module.REPORT_CODE)
    return module.Controller(c, command or (lambda *a, **k: SimpleNamespace(stdout='RUNNING', returncode=0)))


def production_env_settlement(controller, *, stage='A', attempt=2, seat_name='a'):
    """使用生产 _end、AttemptLedger 与 BudgetLedger 生成真实落盘契约，不创建环境。"""
    alias = 'controller_test_production_seat'
    if alias not in sys.modules:
        seat_spec = importlib.util.spec_from_file_location(alias, Path(__file__).parents[1] / 'dev-scripts/gl/seat.py')
        seat_module = importlib.util.module_from_spec(seat_spec)
        sys.modules[alias] = seat_module
        seat_spec.loader.exec_module(seat_module)
    seat_module = sys.modules[alias]
    stage_spec = controller.c['stages'][stage]
    identity = json.loads(Path(stage_spec['identities']).read_text())
    seed = stage_spec['seed']
    raw = Path(stage_spec['out']) / 'rollouts/smvla/raw/result.json'
    raw.parent.mkdir(parents=True, exist_ok=True)
    result = {**identity, 'attempt': attempt, 'policy_seed': seed, 'policy_label': 'smvla',
              'model': 'smvla', 'status': 'fail', 'task_success': 0, 'infra': True,
              'infra_reason': 'env_build', 'budget_exhausted': False, 'run_blocked': False}
    raw.write_text(json.dumps(result))
    caps = module.CAPS[stage]
    shared = seat_module.load_sibling('budget_ledger').BudgetLedger(stage_spec['budget'],
        trajectory_cap=caps[0], reset_cap=caps[1], shared_infra_cap=caps[2],
        expired_cap=caps[3], planned_first_tries=caps[4])
    route = f'smvla/seed{seed}/new'
    token = f'{route}|{identity["key"]}|a{attempt}'
    rid = shared.reserve(resets=2, route=route, key=identity['key'], token=token,
                         dataset='ood', attempt_no=attempt)
    ledger_path = Path(stage_spec['out']) / f'seats/smvla/seed{seed}/{seat_name}/smvla.ledger.jsonl'
    ledger = seat_module.AttemptLedger(ledger_path, seat=seat_name, policy='smvla', shared=shared, route=route)
    aid = f'aid-{stage}-{attempt}'
    ledger.attempt_start(key=identity['key'], attempt_id=aid, attempt_no=attempt,
                         retry=True, result_path=str(raw), budget_rid=rid)
    runner = seat_module.SeatRunner.__new__(seat_module.SeatRunner)
    runner.ledger, runner._lock = ledger, threading.Lock()
    runner.args, runner.seat = SimpleNamespace(policy='smvla'), seat_name
    runner.results_path = raw.parent / 'results.jsonl'
    runner.queue = SimpleNamespace(end_claim=lambda *a, **k: None)
    claim = seat_module.Claim(identity, attempt, raw.parent / 'claim.json', token, rid, True)
    runner._end(claim, aid, {**result, 'result': str(raw)}, void=False)
    # _end 真正追加并序列化共享 commit 与本地 error；以下控制器再读文件。
    return raw, ledger_path, Path(stage_spec['budget'])


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
                seat_name='a', onlykey='Task_xhard1_1', recoveries={'smvla|7|Task_xhard1_1': 18})
    production_env_settlement(c)
    c.tick()
    assert seat['recoveries']['smvla|7|Task_xhard1_1'] == 19
    assert seat['onlykey'] == 'Task_xhard1_1'
    seat['seat_name'] = 'a'
    production_env_settlement(c, attempt=20)
    Path(seat['log']).write_text(log.read_text().replace('attempt=2 ', 'attempt=20 '))
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
    _, _, budget = production_env_settlement(c)
    rows = [json.loads(line) for line in budget.read_text().splitlines()]
    budget.write_text('\n'.join(json.dumps(r) for r in rows if r['kind'] != 'commit'))
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


def test_production_env_status_fail_shared_and_error_local_roundtrip(tmp_path):
    c = fixture(tmp_path)
    raw, ledger, budget = production_env_settlement(c)
    result = module.read(raw)
    local = [json.loads(x) for x in ledger.read_text().splitlines()]
    shared = [json.loads(x) for x in budget.read_text().splitlines()]
    assert result['status'] == 'fail'
    assert local[-1]['kind'] == 'attempt_end' and local[-1]['status'] == 'error'
    assert shared[-1]['kind'] == 'commit' and shared[-1]['status'] == 'fail'
    seat = c.state['seats'][0]
    seat.update(stage='A', model='smvla', onlykey='Task_xhard1_1', seat_name='a')
    c.env_build_settled(seat, 'RUN_BLOCKED reason=env_build policy=smvla seat=a key=Task_xhard1_1 attempt=2 settled=1')


@pytest.mark.parametrize('field,value', [('status', 'error'), ('infra_reason', 'other'),
                                        ('attempt', 3), ('policy_seed', 0)])
def test_production_env_original_result_mismatch_rejected(tmp_path, field, value):
    c = fixture(tmp_path)
    raw, _, _ = production_env_settlement(c)
    result = module.read(raw)
    result[field] = value
    raw.write_text(json.dumps(result))
    seat = c.state['seats'][0]
    seat.update(stage='A', model='smvla', onlykey='Task_xhard1_1', seat_name='a')
    with pytest.raises(ValueError):
        c.env_build_settled(seat, 'RUN_BLOCKED reason=env_build policy=smvla seat=a key=Task_xhard1_1 attempt=2 settled=1')


def test_same_key_new_policy_seed_does_not_inherit_retry_limit(tmp_path):
    c = fixture(tmp_path, lambda argv, **k: SimpleNamespace(stdout='RUNNING', returncode=1 if 'has-session' in argv else 0))
    c.state['stage'] = 'C'
    production_env_settlement(c, stage='C')
    log = c.root / 'seed0-env.log'
    log.write_text('RUN_BLOCKED reason=env_build policy=smvla seat=a key=Task_xhard1_1 attempt=2 settled=1\nEXIT_CODE=3\n')
    seat = c.state['seats'][0]
    seat.update(stage='C', model='smvla', onlykey='Task_xhard1_1', seat_name='a',
                session='fe-controller-C-smvla-0-1', log=str(log),
                recoveries={'smvla|7|Task_xhard1_1': 20})
    c.tick()
    assert seat['recoveries']['smvla|7|Task_xhard1_1'] == 20
    assert seat['recoveries']['smvla|0|Task_xhard1_1'] == 1
    assert seat['onlykey'] == 'Task_xhard1_1'


def test_r2_carrier_is_rejected(tmp_path):
    c = fixture(tmp_path)
    c.c['carrier'] = str(tmp_path / 'run-model-r2.sh')
    with pytest.raises(ValueError, match='固定路径'):
        module.Controller(c.c)


def test_report_entry_must_be_r3(tmp_path):
    c = fixture(tmp_path)
    c.c['report_code'] = '/runtime-code-r2/dev-scripts/gl/stage_report.py'
    with pytest.raises(ValueError, match='报告入口'):
        module.Controller(c.c)


def legacy_fixture(tmp_path, *, attempts=2, max_attempts=2, final=False, stage='A', model='pp'):
    """真实队列、尝试账本和共享账本落盘；只伪造命令执行，不导入环境。"""
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='RUNNING', returncode=1 if 'has-session' in argv else 0)
    c = fixture(tmp_path, command)
    c.state['stage'] = stage
    seat_spec = importlib.util.spec_from_file_location('legacy_fixture_seat',
        Path(__file__).parents[1] / 'dev-scripts/gl/seat.py')
    sm = importlib.util.module_from_spec(seat_spec)
    sys.modules[seat_spec.name] = sm
    seat_spec.loader.exec_module(sm)
    stage_spec = c.c['stages'][stage]
    identity = module.read(stage_spec['identities'])
    queue_ident = sm.identity_for_queue(identity, 'ood')
    label = module.MODELS[model]
    policy = 'groundsg' if label.startswith('groundsg-') else label
    route = f"{label}/seed{stage_spec['seed']}/new"
    seat_name = 'fe-A-pp-01'
    caps = stage_spec['caps']
    shared = sm.load_sibling('budget_ledger').BudgetLedger(stage_spec['budget'],
        trajectory_cap=caps[0], reset_cap=caps[1], shared_infra_cap=caps[2],
        expired_cap=caps[3], planned_first_tries=caps[4])
    queue = sm.DynamicQueue(Path(stage_spec['out']) / f"queue/{label}/seed{stage_spec['seed']}",
                            seat=seat_name, wall_s=60, max_attempts=max_attempts)
    ledger_path = Path(stage_spec['out']) / f"seats/{label}/seed{stage_spec['seed']}/{seat_name}/{label}.ledger.jsonl"
    ledger = sm.AttemptLedger(ledger_path, seat=seat_name, policy=policy, shared=shared, route=route)
    runner = sm.SeatRunner.__new__(sm.SeatRunner)
    runner.ledger, runner.queue, runner._lock = ledger, queue, threading.Lock()
    runner.args, runner.seat = SimpleNamespace(policy=policy), seat_name
    runner.results_path = ledger_path.parent / f'{label}.results.jsonl'
    raw = Path(stage_spec['out']) / f'rollouts/{label}/result.json'
    raw.parent.mkdir(parents=True)
    for n in range(1, attempts + 1):
        token = f'{route}|{identity["key"]}|a{n}'
        if n > 1:
            assert shared.claim_retry(route=route, key=identity['key'], interrupt='infra',
                                      token=token, max_attempts=20)
        rid = shared.reserve(resets=2, route=route, key=identity['key'], token=token,
                             dataset='ood', attempt_no=n, kind_of_try='first' if n == 1 else 'recovery')
        claim_path = queue.create_claim(queue_ident, n, token=token, rid=rid, route=route)
        aid = f'aid-{n}'
        ledger.attempt_start(key=identity['key'], attempt_id=aid, attempt_no=n, retry=n > 1,
                             identity=queue_ident, result_path=str(raw), budget_rid=rid,
                             claim=str(claim_path), token=token)
        result = {**{k: v for k, v in identity.items() if k != 'builder_episode'},
                  'attempt': n, 'policy_seed': stage_spec['seed'], 'policy_label': label,
                  'model': policy, 'status': 'fail', 'task_success': 0, 'infra': not final,
                  'infra_reason': 'pp_server_error', 'budget_exhausted': False, 'run_blocked': False}
        raw.write_text(json.dumps(result))
        runner._end(sm.Claim(queue_ident, n, claim_path, token, rid, n > 1), aid,
                    {**result, 'result': str(raw)}, void=False)
    log = c.root / 'legacy.log'
    missing, accepted = (0, 1) if final else (1, 0)
    log.write_text(f'RUN_PLAN policy={policy} label={label} seat={seat_name} total=1 claimable=1 '
                   f'max_attempts={max_attempts} route={route} reset_left=inf\n'
                   f'RUN_SUMMARY policy={policy} label={label} seat={seat_name} total=1 '
                   f'accepted={accepted} running_elsewhere=0 missing={missing} episodes_run={attempts}\n'
                   + (f'RUN_INCOMPLETE policy={policy} seat={seat_name} total=1 missing=1 '
                      f'first=ood:{identity["key"]}\n' if missing else '')
                   + f'EXIT_CODE={6 if missing else 0}\n')
    seat = c.state['seats'][0]
    seat.update(session=f'fe-controller-{stage}-{model}-0-1', model=model, stage=stage,
                log=str(log), seat_name=seat_name, onlykey=None)
    return c, seat, raw, ledger_path, shared.path, queue, calls


@pytest.mark.parametrize('stage,model', [('A', 'pp'), ('B', 'memer'), ('C', 'qwen'), ('D', 'pp')])
def test_legacy_max2_serialized_settlement_resumes_r3(tmp_path, stage, model):
    c, seat, _, _, budget, queue, calls = legacy_fixture(tmp_path, stage=stage, model=model)
    original_budget = budget.read_bytes()
    c.tick()
    events = [json.loads(line) for line in (c.root / 'events.jsonl').read_text().splitlines()]
    resumed = next(e for e in events if e['kind'] == 'legacy_retry_resume')
    assert resumed['attempts'] == {'Task_xhard1_1': 2} and resumed['max_attempts'] == 20
    assert 'run-model-r3.sh' in next(x[-1] for x in calls if 'new-session' in x)
    assert budget.read_bytes() == original_budget
    assert module.read(c.path)['stage'] == stage
    assert len(list(queue.claims.glob('*.json'))) == 2


@pytest.mark.parametrize('attempts,max_attempts', [(2, 20), (20, 2), (1, 2)])
def test_legacy_new20_or_exhausted_attempts_refuse(tmp_path, attempts, max_attempts):
    c, seat, _, _, _, _, calls = legacy_fixture(tmp_path, attempts=attempts, max_attempts=max_attempts)
    with pytest.raises(ValueError):
        c.tick()
    assert not any('new-session' in argv for argv in calls)
    assert seat['session']


def test_legacy_normal_failure_accepted_is_never_reclaimed(tmp_path):
    c, seat, _, _, _, queue, calls = legacy_fixture(tmp_path, attempts=1, final=True)
    assert c.remaining('A', 'pp') == []
    c.tick()
    assert not any('new-session' in argv and 'fe-controller-A-pp-' in str(argv) for argv in calls)
    assert module.read(queue.accepted_dir / 'ood__Task_xhard1_1.json')['status'] == 'fail'
    assert len(list(queue.claims.glob('*.json'))) == 1
    assert queue.claimable(module.read(c.c['stages']['A']['identities'])) is False


def test_legacy_attempt19_remains_allowed_without_spending_budget(tmp_path):
    c, seat, _, _, budget, _, _ = legacy_fixture(tmp_path, attempts=19)
    before = budget.read_bytes()
    assert c.legacy_retry_settled(seat, Path(seat['log']).read_text(), ['Task_xhard1_1']) == {'Task_xhard1_1': 19}
    assert budget.read_bytes() == before


@pytest.mark.parametrize('cap', ['reset', 'trajectory', 'reserved_first', 'identity_retry'])
def test_legacy_each_shared_budget_guard_refuses(tmp_path, cap):
    c, seat, _, _, budget, _, _ = legacy_fixture(tmp_path)
    rows = [json.loads(line) for line in budget.read_text().splitlines()]
    if cap == 'reset':
        next(r for r in rows if r['kind'] == 'reserve')['resets'] = module.CAPS['A'][1]
    elif cap in ('trajectory', 'reserved_first'):
        extra = module.CAPS['A'][0] - 2 if cap == 'trajectory' else module.CAPS['A'][0] - module.CAPS['A'][4]
        rows += [{'kind': 'reserve', 'rid': f'extra-{i}', 'resets': 0} for i in range(extra)]
    else:
        rows += [{'kind': 'retry_claim', 'route': 'pp/seed7/new', 'key': 'Task_xhard1_1', 'interrupt': 'infra'}
                 for _ in range(18)]
    budget.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    with pytest.raises(ValueError, match='预算耗尽'):
        c.legacy_retry_settled(seat, Path(seat['log']).read_text(), ['Task_xhard1_1'])


@pytest.mark.parametrize('mutation', ['fake_identity', 'no_local_end', 'no_shared_commit', 'live_claim',
    'normal_result', 'lost_accept', 'budget', 'bad_summary', 'running_elsewhere', 'missing_log', 'unknown_exit',
    'bad_reserve', 'budget_config', 'open_reserve', 'no_shared_retry', 'missing_prior_claim'])
def test_legacy_inconsistent_or_unsettled_evidence_refuses(tmp_path, mutation):
    c, seat, raw, local, budget, queue, calls = legacy_fixture(tmp_path)
    if mutation in ('fake_identity', 'normal_result'):
        doc = module.read(raw)
        doc['seed' if mutation == 'fake_identity' else 'infra'] = 999 if mutation == 'fake_identity' else False
        raw.write_text(json.dumps(doc))
    elif mutation in ('no_local_end', 'lost_accept'):
        rows = [json.loads(line) for line in local.read_text().splitlines()]
        if mutation == 'no_local_end':
            rows = [r for r in rows if r.get('attempt_id') != 'aid-2' or r['kind'] != 'attempt_end']
        else:
            rows.append({'kind': 'accept', 'key': 'Task_xhard1_1'})
        local.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    elif mutation == 'missing_prior_claim':
        (queue.claims / 'ood__Task_xhard1_1.a1.json').unlink()
    elif mutation == 'live_claim':
        path = queue.claims / 'ood__Task_xhard1_1.a2.json'
        doc = module.read(path)
        doc['ended'] = False
        path.write_text(json.dumps(doc))
    elif mutation in ('no_shared_commit', 'budget', 'bad_reserve', 'budget_config', 'open_reserve', 'no_shared_retry'):
        rows = [json.loads(line) for line in budget.read_text().splitlines()]
        if mutation == 'no_shared_commit':
            rows = [r for r in rows if r['kind'] != 'commit']
        elif mutation == 'no_shared_retry':
            rows = [r for r in rows if r['kind'] != 'retry_claim']
        elif mutation == 'budget':
            rows += [{'kind': 'retry_claim', 'route': 'other', 'key': str(i), 'interrupt': 'infra'}
                     for i in range(module.CAPS['A'][2])]
        elif mutation == 'bad_reserve':
            next(r for r in reversed(rows) if r['kind'] == 'reserve')['key'] = 'wrong'
        elif mutation == 'budget_config':
            rows[0]['trajectory_cap'] += 1
        else:
            rows.append({'kind': 'reserve', 'rid': 'open', 'route': 'pp/seed7/new', 'key': 'Task_xhard1_1'})
        budget.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    else:
        path = Path(seat['log'])
        text = path.read_text()
        before, after = {'bad_summary': ('accepted=0', 'accepted=1'),
                         'running_elsewhere': ('running_elsewhere=0', 'running_elsewhere=1'),
                         'missing_log': ('RUN_INCOMPLETE', 'UNKNOWN'),
                         'unknown_exit': ('EXIT_CODE=6', 'EXIT_CODE=75')}[mutation]
        path.write_text(text.replace(before, after))
    with pytest.raises(ValueError):
        c.tick()
    assert not any('new-session' in argv for argv in calls)
    assert not (c.root / 'events.jsonl').exists()
