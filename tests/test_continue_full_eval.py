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
    identities = tmp_path / 'identities.jsonl'
    identities.write_text(json.dumps({'key': 'task', 'dataset': 'ood', 'task': 'Task',
        'episode': 0, 'tier': 'xhard1', 'seed': 1, 'candidate': 0,
        'source_episode': None, 'spec_sha256': 'sha'}) + '\n')
    c = {'controller_root': str(tmp_path / 'controller'), 'carrier': '/carrier',
         'report_code': '/report', 'python': '/python', 'ffmpeg': '/ffmpeg', 'report_timeout_s': 3600,
         'models': {'smvla': 'smvla'}, 'priority': ['smvla'],
         'stages': {s: {'identities': str(identities), 'out': str(tmp_path / s), 'seed': 7}
                    for s in 'ABCD'},
         'seats': [{'job': str(i), 'slot': i} for i in range(4)]}
    return module.Controller(c, command or (lambda *a, **k: SimpleNamespace(stdout='RUNNING', returncode=0)))


def accept(controller, stage='A', status='fail'):
    out = Path(controller.c['stages'][stage]['out'])
    raw = out / 'rollouts' / 'result.json'
    raw.parent.mkdir(parents=True)
    row = json.loads(Path(controller.c['stages'][stage]['identities']).read_text())
    row.update(attempt=1, status=status, policy_seed=7, policy_label='smvla', model='smvla',
               infra=False, run_blocked=False, budget_exhausted=False)
    raw.write_text(json.dumps(row))
    marker = out / 'queue/smvla/seed7/accepted/ood__task.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({**row, 'result': 'rollouts/result.json'}))
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
    accept(c)
    output = c.root / 'report-A.json'
    command(['--json', str(output)])
    log = c.root / 'report.log'
    log.write_text('EXIT_CODE=0\n')
    c.state['report_task'] = {'session': 'report', 'log': str(log), 'started': 0}
    c.tick()
    assert module.read(c.path)['stage'] == 'B'


def test_report_crash_detected(tmp_path):
    def command(argv, **kw):
        raise RuntimeError('报告崩溃')
    c = fixture(tmp_path, command)
    accept(c)
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
    c.state['seats'][0].update(session='existing', model='smvla', log='/unused')
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
    log.write_text('RUN_BLOCKED reason=env_build policy=smvla seat=a key=task attempt=2 settled=1\nEXIT_CODE=3\n')
    seat = c.state['seats'][0]
    seat.update(session='old', model='smvla', log=str(log), onlykey='task', recoveries={'task': 18})
    c.tick()
    assert seat['recoveries']['task'] == 19
    assert seat['onlykey'] == 'task'
    Path(seat['log']).write_text(log.read_text())
    with pytest.raises(ValueError, match='上限'):
        c.tick()


def test_missing_exit_refuses_relaunch(tmp_path):
    c = fixture(tmp_path, lambda *a, **k: SimpleNamespace(stdout='', returncode=1))
    log = tmp_path / 'worker.log'
    log.write_text('')
    c.state['seats'][0].update(session='old', model='smvla', log=str(log))
    with pytest.raises(ValueError, match='退出码'):
        c.tick()
