"""由主会话明确配置的四席接续；失败封闭，不提供代理自动唤醒。"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_stage_identities import read_rows
from budget_ledger import BudgetLedger, BudgetState

ROOT = Path('/nfs/turbo/coe-chaijy-unreplicated/hongzefu/artifacts/full-eval-20261010')
REPORT_CODE = Path('/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/full-eval-20261010/runtime-code-r3/dev-scripts/gl/stage_report.py')
MODELS = {'perceptual-framesamp-modul': 'perceptual-framesamp-modul', 'smvla': 'smvla',
          'pp': 'pp', 'oracle': 'groundsg-ground-sg-oracle',
          'qwen': 'groundsg-ground-sg-qwenvl', 'memer': 'groundsg-ground-sg-memer'}
SEEDS = dict(A=7, B=7, C=0, D=42)
CAPS = {'A': (4680, 9360, 420, 420, 426), 'B': (41580, 83160, 3780, 3780, 3780),
        'C': (46200, 92400, 4200, 4200, 4200), 'D': (46200, 92400, 4200, 4200, 4200)}


def read(path):
    return json.loads(Path(path).read_text())


def run(argv, check=True):
    return subprocess.run(argv, text=True, capture_output=True, check=check, timeout=300)


class Controller:
    def __init__(self, config, command=run):
        self.c, self.command = config, command
        self.root = Path(config['controller_root'])
        if self.root != ROOT / 'controller' or Path(config['carrier']) != ROOT / 'run-model-r3.sh':
            raise ValueError('控制器或载体不属于本轮固定路径')
        if Path(config['report_code']) != REPORT_CODE:
            raise ValueError('报告入口不属于冻结的 r3 副本')
        if config['models'] != MODELS or set(config['priority']) != set(MODELS) or len(config['priority']) != 6:
            raise ValueError('模型必须为已批准六项且优先队列无重复')
        if set(config['stages']) != set('ABCD'):
            raise ValueError('阶段必须恰为 A/B/C/D')
        for stage, spec in config['stages'].items():
            if (type(spec['seed']) is not int or spec['seed'] != SEEDS[stage]
                    or Path(spec['identities']) != ROOT / 'identities' / f'stage-{stage}.jsonl'
                    or Path(spec['out']) != ROOT / f'seed{SEEDS[stage]}'
                    or Path(spec['budget']) != ROOT / 'budget' / f'stage-{stage}.jsonl'
                    or tuple(spec['caps']) != CAPS[stage]):
                raise ValueError('阶段身份、种子、产物或预算与固定载体不符')
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'state.json'
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.state = read(self.path) if self.path.exists() else {
            'config_sha256': digest,
            'stage': 'A', 'seats': json.loads(json.dumps(config['seats'])), 'serial': 0, 'done': False,
            'expired_jobs': [],
            'missing': 0, 'media_failed': 0, 'report_failed': 0}
        if self.state['config_sha256'] != digest:
            raise ValueError('配置指纹改变，禁止接手')
        if len(self.state['seats']) != 4:
            raise ValueError('仅允许主会话指定的四席')
        if {s['slot'] for s in self.state['seats']} != set(range(4)):
            raise ValueError('席位编号必须互异且为 0～3')
        if len({str(s['job']) for s in self.state['seats']}) != 4:
            raise ValueError('四席作业编号重复')
        for seat in self.state['seats']:
            if not str(seat['job']).isdigit():
                raise ValueError('非法作业编号')
            if seat.get('session'):
                self.validate_session(seat)

    def validate_session(self, seat):
        name = seat['session']
        initial = {s.get('session'): s for s in self.c['seats'] if s.get('session')}
        generated = re.fullmatch(r'fe-controller-[ABCD]-(?:' + '|'.join(MODELS) + r')-[0-3]-\d+', name)
        if name not in initial and not generated:
            raise ValueError('会话不在主会话接手清单中')
        if name in initial and (seat['log'] != initial[name]['log'] or seat['model'] != initial[name]['model']):
            raise ValueError('接手会话与原清单不符')
        if not Path(seat['log']).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('会话日志不属于本轮路径')
        if seat['model'] not in MODELS or seat['stage'] not in SEEDS:
            raise ValueError('会话模型或阶段不符')

    def job_state(self, seat):
        active = self.active_state(seat)
        if active:
            return active
        states = [x.strip().split('|')[0] for x in self.command(
            ['sacct', '-n', '-X', '-j', str(seat['job']), '-o', 'State', '-P']).stdout.splitlines() if x.strip()]
        if states and all(x == 'TIMEOUT' for x in states):
            return 'EXPIRED'
        raise ValueError('占位作业消失但未确认 TIMEOUT')

    def active_state(self, seat):
        result = self.command(['squeue', '-h', '-j', str(seat['job']), '-o', '%T'], check=False)
        if result.returncode != 0:
            if (result.returncode == 1 and not result.stdout.strip()
                    and re.fullmatch(r'(?:squeue: error: )?Invalid job id specified\s*', getattr(result, 'stderr', ''))):
                return ''
            raise ValueError('squeue 查询失败，禁止推断到期')
        value = result.stdout.strip()
        if '\n' in value:
            raise ValueError('squeue 返回多个作业状态')
        return value

    def env_build_settled(self, seat, log):
        key = seat['onlykey']
        matches = re.findall(r'RUN_BLOCKED reason=env_build .*key=' + re.escape(key)
                             + r' attempt=(\d+) settled=1', log)
        if not matches:
            raise ValueError('退出码 3 缺少环境创建结算日志')
        attempt = int(matches[-1])
        if attempt >= 20:
            raise ValueError('身份达到二十次尝试上限')
        stage = seat['stage']
        spec = self.c['stages'][stage]
        label = MODELS[seat['model']]
        ledger_path = Path(spec['out']) / 'seats' / label / f"seed{spec['seed']}" / seat['seat_name'] / f'{label}.ledger.jsonl'
        records = [json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()]
        starts = [r for r in records if r.get('kind') == 'attempt_start' and r.get('key') == key]
        if not starts or starts[-1]['attempt_no'] != attempt:
            raise ValueError('环境故障不是该身份最后一次尝试')
        start = starts[-1]
        ident = next((r for r in read_rows(spec['identities']) if r['key'] == key), None)
        if ident is None:
            raise ValueError('环境故障身份不属于阶段白名单')
        stored = Path(start['result_path'])
        if 'rollouts' not in stored.parts:
            raise ValueError('环境故障结果路径缺少 rollouts')
        result_path = Path(spec['out']) / Path(*stored.parts[stored.parts.index('rollouts'):])
        if not result_path.resolve().is_relative_to(Path(spec['out']).resolve()):
            raise ValueError('环境故障结果路径越界')
        result = read(result_path)
        for field in ('dataset', 'task', 'episode', 'tier', 'seed', 'candidate',
                      'source_episode', 'spec_sha256', 'key'):
            if result[field] != ident[field] or type(result[field]) is not type(ident[field]):
                raise ValueError('环境故障结果身份不符：' + field)
        if (type(result['attempt']) is not int or result['attempt'] != attempt
                or result['policy_seed'] != spec['seed'] or result['policy_label'] != label
                or result['model'] != 'smvla' or result['infra'] is not True
                or result['infra_reason'] != 'env_build' or result['budget_exhausted'] is not False
                or result['status'] not in ('fail', 'error') or type(result['task_success']) is not int
                or result['task_success'] != 0):
            raise ValueError('环境故障原结果或尝试不符')
        ends = [r for r in records if r.get('kind') == 'attempt_end' and r.get('attempt_id') == start['attempt_id']]
        if not ends or not (ends[-1]['infra'] is True and ends[-1]['infra_reason'] == 'env_build'
                            and ends[-1]['status'] == 'error' and ends[-1]['attempt_no'] == attempt
                            and ends[-1]['key'] == key and ends[-1]['budget_exhausted'] is False):
            raise ValueError('环境故障本地账本未结算')
        budget = [json.loads(x) for x in Path(spec['budget']).read_text().splitlines() if x.strip()]
        commits = [r for r in budget if r.get('kind') == 'commit' and r.get('rid') == start['budget_rid']]
        if not commits or not (commits[-1]['infra'] is True and commits[-1]['status'] == result['status']):
            raise ValueError('环境故障共享预算未结算')

    def legacy_retry_settled(self, seat, log, missing):
        """只核实旧两次上限退出；不领取额度，不修复证据，重试仍由原队列原子领取。"""
        spec = self.c['stages'][seat['stage']]
        label = MODELS[seat['model']]
        policy = 'groundsg' if label.startswith('groundsg-') else label
        head = 'groundsg/' + label.removeprefix('groundsg-') if policy == 'groundsg' else policy
        route = f"{head}/seed{spec['seed']}/new"
        rows = read_rows(spec['identities'])
        if (seat['stage'] != self.state['stage'] or seat['model'] == 'smvla'
                or seat.get('onlykey') or not missing or 'RUN_BLOCKED ' in log):
            raise ValueError('旧上限接续范围或缺失身份不符')

        def fields(name):
            lines = re.findall(r'^' + name + r' (.+)$', log, re.M)
            if len(lines) != 1:
                raise ValueError('旧上限日志缺失或重复：' + name)
            pairs = [word.split('=', 1) for word in lines[0].split()]
            if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
                raise ValueError('旧上限日志字段损坏：' + name)
            result = dict(pairs)
            if (result['policy'] != policy or result['seat'] != seat['seat_name']
                    or result['total'] != str(len(rows))):
                raise ValueError('旧上限日志身份不符：' + name)
            return result

        plan, summary, incomplete = (fields(name) for name in ('RUN_PLAN', 'RUN_SUMMARY', 'RUN_INCOMPLETE'))
        prefix = incomplete['first'].split(',')
        if (plan['label'] != label or plan['route'] != route or plan['max_attempts'] != '2'
                or summary['label'] != label or summary['running_elsewhere'] != '0'
                or summary['accepted'] != str(len(rows) - len(missing))
                or summary['missing'] != str(len(missing)) or incomplete['missing'] != str(len(missing))
                or len(prefix) != min(5, len(missing)) or len(set(prefix)) != len(prefix)
                or not set(prefix) <= {'ood:' + key for key in missing}):
            raise ValueError('旧上限日志与实际缺失集合不符')
        if not log.index('RUN_PLAN ') < log.index('RUN_SUMMARY ') < log.index('RUN_INCOMPLETE '):
            raise ValueError('旧上限日志顺序不符')

        def records(path):
            return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

        caps = spec['caps']
        shared = BudgetLedger(spec['budget'], trajectory_cap=caps[0], reset_cap=caps[1],
                              shared_infra_cap=caps[2], expired_cap=caps[3], planned_first_tries=caps[4])
        # 只读已有账本；不能调用 state() 在缺 config 时补写证据。
        with Path(str(spec['budget']) + '.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            budget = BudgetState(records(spec['budget']))
        if budget.bad_rows or shared._config_diff(budget):
            raise ValueError('旧上限共享预算损坏或配置不符')
        queue = Path(spec['out']) / 'queue' / label / f"seed{spec['seed']}"
        attempts = {}
        for ident in rows:
            key = ident['key']
            if key not in missing:
                continue
            claims = sorted((int(p.name.rsplit('.a', 1)[1].removesuffix('.json')), p, read(p))
                            for p in (queue / 'claims').glob('ood__' + key + '.a*.json'))
            if (not claims or [n for n, _, _ in claims] != list(range(1, claims[-1][0] + 1))
                    or any(doc.get('ended') is not True or doc.get('key') != key
                           or doc.get('dataset') != 'ood' or type(doc.get('attempt')) is not int
                           or doc['attempt'] != n for n, _, doc in claims)):
                raise ValueError('旧上限领取缺失或尚未收尾')
            attempt, claim_path, claim = claims[-1]
            token = f'{route}|{key}|a{attempt}'
            if (not 2 <= attempt < 20 or type(claim['attempt']) is not int or claim['attempt'] != attempt
                    or claim['dataset'] != 'ood' or claim['key'] != key or claim['route'] != route
                    or claim['token'] != token or claim['end_status'] != 'infra'
                    or claim['final'] is not False or claim['accepted'] is not False
                    or not re.fullmatch(r'[A-Za-z0-9._-]+', claim['seat'])):
                raise ValueError('旧上限领取身份、终态或二十次上限不符')
            local = records(Path(spec['out']) / 'seats' / label / f"seed{spec['seed']}"
                            / claim['seat'] / f'{label}.ledger.jsonl')
            starts = [r for r in local if r.get('kind') == 'attempt_start' and r.get('key') == key]
            if not starts:
                raise ValueError('旧上限本地尝试缺失')
            start = starts[-1]
            if (start['attempt_no'] != attempt or start['budget_rid'] != claim['rid']
                    or start['token'] != token or start['route'] != route
                    or start['seat'] != claim['seat'] or start['policy'] != policy
                    or Path(start['claim']) != claim_path
                    or start['identity'] != {k: v for k, v in ident.items() if k != 'episode'}):
                raise ValueError('旧上限本地尝试与领取不符')
            stored = Path(start['result_path'])
            if not stored.resolve().is_relative_to((Path(spec['out']) / 'rollouts').resolve()):
                raise ValueError('旧上限原结果路径越界')
            result = read(stored)
            if any(result[k] != ident[k] or type(result[k]) is not type(ident[k])
                   for k in ident if k != 'builder_episode'):
                raise ValueError('旧上限原结果完整身份不符')
            if (type(result['attempt']) is not int or result['attempt'] != attempt
                    or result['policy_seed'] != spec['seed'] or result['policy_label'] != label
                    or result['model'] != policy or result['infra'] is not True
                    or not isinstance(result['infra_reason'], str) or not result['infra_reason']
                    or result['budget_exhausted'] is not False or result['run_blocked'] is not False
                    or result['status'] not in ('fail', 'error')
                    or type(result['task_success']) is not int or result['task_success'] != 0):
                raise ValueError('旧上限缺失身份不是完整基础设施结果')
            ends = [r for r in local if r.get('kind') == 'attempt_end' and r.get('attempt_id') == start['attempt_id']]
            if (len(ends) != 1 or any(r.get('kind') == 'accept' and r.get('key') == key for r in local)
                    or not (ends[0]['key'] == key and ends[0]['attempt_no'] == attempt
                            and ends[0]['status'] == 'error' and ends[0]['infra'] is True
                            and ends[0]['infra_reason'] == result['infra_reason']
                            and ends[0]['budget_exhausted'] is False and not ends[0].get('run_blocked'))):
                raise ValueError('旧上限本地结束未结算或已有正常终态')
            reserve = budget.reserves.get(claim['rid'], {})
            commit = budget.commits.get(claim['rid'], {})
            retries = [r for r in budget.retries if r.get('token') == token]
            if (reserve.get('route') != route or reserve.get('key') != key or reserve.get('token') != token
                    or reserve.get('attempt_no') != attempt or reserve.get('dataset') != 'ood'
                    or reserve.get('kind_of_try') != 'recovery' or len(retries) != 1
                    or retries[0].get('route') != route or retries[0].get('key') != key
                    or retries[0].get('interrupt') not in ('infra', 'expired')
                    or claim['rid'] in budget.releases or commit.get('infra') is not True
                    or commit.get('status') != result['status']
                    or any(r.get('route') == route and r.get('key') == key and rid not in budget.commits
                           and rid not in budget.releases for rid, r in budget.reserves.items())):
                raise ValueError('旧上限共享预约未结算或身份不符')
            if budget.retries_for(route, key) >= 19:
                raise ValueError('旧上限身份重试预算耗尽')
            attempts[key] = attempt
        needed = len(missing)
        pending = max(0, caps[4] - budget.first_started)
        if (budget.trajectories + needed + pending > caps[0] or budget.resets + 2 * needed > caps[1]
                or budget.retries_of('infra') + needed > caps[2]
                or max(budget.recovery_used, len(budget.retries)) + needed > caps[0] - caps[4]):
            raise ValueError('旧上限接续共享预算耗尽')
        return attempts

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w') as f:
            f.write(json.dumps(self.state, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)
        expired_tmp = self.root / 'expired-jobs.tmp'
        expired_tmp.write_text('\n'.join(self.state['expired_jobs']) + '\n')
        expired_tmp.replace(self.root / 'expired-jobs.txt')

    def event(self, kind, **fields):
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.time(), 'kind': kind, **fields}) + '\n')
        self.save()

    def alive(self, name):
        r = self.command(['tmux', 'has-session', '-t', '=' + name], check=False)
        if r.returncode not in (0, 1):
            raise ValueError('tmux 查询失败')
        return r.returncode == 0

    def remaining(self, stage, model):
        spec = self.c['stages'][stage]
        rows = read_rows(spec['identities'])
        queue = Path(spec['out']) / 'queue' / self.c['models'][model] / f"seed{spec['seed']}"
        missing = []
        for row in rows:
            marker = queue / 'accepted' / ('ood__' + row['key'] + '.json')
            if not marker.exists():
                missing.append(row['key'])
                continue
            accepted = read(marker)
            stored = Path(accepted['result'])
            if 'rollouts' not in stored.parts:
                raise ValueError('结果缺少 rollouts 路径')
            result = Path(spec['out']) / Path(*stored.parts[stored.parts.index('rollouts'):])
            if not result.resolve().is_relative_to(Path(spec['out']).resolve()):
                raise ValueError('accepted 结果越界')
            data = read(result)
            for field in ('dataset', 'task', 'episode', 'tier', 'seed', 'candidate',
                          'source_episode', 'spec_sha256', 'key'):
                if data[field] != row[field] or type(data[field]) is not type(row[field]):
                    raise ValueError('完整身份不符：' + field)
            label = self.c['models'][model]
            expected = 'groundsg' if label.startswith('groundsg-') else label
            if (data['policy_seed'] != spec['seed'] or data['policy_label'] != label
                    or data['model'] != expected or data['infra'] is not False
                    or data['run_blocked'] is not False or data['budget_exhausted'] is not False):
                raise ValueError('模型或基础设施终态不符')
            if label.startswith('groundsg-') and data['policy_variant'] != label.removeprefix('groundsg-'):
                raise ValueError('GroundSG 变体不符')
            if (accepted['dataset'] != 'ood' or accepted['key'] != row['key'] or data['key'] != row['key']
                    or type(data['attempt']) is not int or not 1 <= data['attempt'] <= 20
                    or type(data['task_success']) is not int or data['task_success'] != int(data['status'] == 'success')
                    or accepted['attempt'] != data['attempt']
                    or accepted['status'] != data['status']
                    or data['status'] not in ('success', 'fail', 'timeout', 'error')):
                raise ValueError('accepted 与结果契约不符')
        return missing

    def replace_expired(self, seat):
        job = str(seat['job'])
        if self.job_state(seat) != 'EXPIRED':
            return False
        # 提交意图先持久化；回执不明或崩溃后绝不重复提交。
        if seat.get('submission_pending'):
            raise ValueError('上次提交状态不明，交主会话核查')
        seat['submission_pending'] = True
        self.event('hold_submit_intent', old_job=job)
        r = self.command(['sbatch', '--parsable', '--account=chaijy2', '--partition=spgpu',
                          '--nodes=1', '--ntasks-per-node=1', '--gres=gpu:1', '--time=48:00:00',
                          '--cpus-per-task=4', '--mem=64G', '--wrap=sleep infinity'])
        value = r.stdout.strip()
        if not re.fullmatch(r'[0-9]+(?:;[\w.-]+)?', value):
            raise ValueError('sbatch 回执不明，禁止重试')
        seat['job'] = value.split(';')[0]
        if sum(str(s['job']) == seat['job'] for s in self.state['seats']) != 1:
            raise ValueError('新作业回执与已登记席位重复')
        seat['submission_pending'] = False
        if job not in self.state['expired_jobs']:
            self.state['expired_jobs'].append(job)
        self.event('hold_replaced', old_job=job, job=seat['job'])
        return True

    def launch(self, seat, stage, model, key=None):
        self.state['serial'] += 1
        name = f"fe-controller-{stage}-{model}-{seat['slot']}-{self.state['serial']}"
        log = self.root / (name + '.log')
        argv = ['bash', self.c['carrier'], stage, model, str(seat['job']),
                f"controller-{seat['slot']}-{self.state['serial']}", 'formal', str(seat['slot'])]
        if key:
            argv.append(key)
        body = 'set -o pipefail; PYTHONUNBUFFERED=1 SGEVAL_EXPIRED_JOBS=' + shlex.quote(str(self.root / 'expired-jobs.txt')) + ' ' + shlex.join(argv)
        body += ' 2>&1 | tee ' + shlex.quote(str(log)) + '; echo "EXIT_CODE=$?" >> ' + shlex.quote(str(log))
        # 起跑意图先保存；重启只接手，不重复创建。
        seat.update(session=name, log=str(log), model=model, stage=stage, onlykey=key,
                    seat_name=argv[5], pending=True)
        self.event('launch_intent', argv=argv, session=name, log=str(log))
        self.command(['tmux', 'new-session', '-d', '-s', name, 'bash -c ' + shlex.quote(body)])
        seat['pending'] = False
        self.event('launched', session=name)

    def report(self, stage):
        spec = self.c['stages'][stage]
        output = self.root / ('report-' + stage + '.json')
        task = self.state.get('report_task')
        if task:
            if self.alive(task['session']):
                if time.time() - task['started'] > self.c['report_timeout_s']:
                    raise ValueError('报告超时；保留进程交主会话处理')
                return False
            codes = re.findall(r'^EXIT_CODE=(\d+)$', Path(task['log']).read_text(), re.M)
            if not codes or codes[-1] != '0':
                raise ValueError('报告进程异常或缺少退出码')
        else:
            if output.exists():
                raise ValueError('报告已存在但阶段未结算，交主会话核查')
            seat = next((s for s in self.state['seats'] if self.active_state(s) == 'RUNNING'), None)
            if not seat:
                for candidate in self.state['seats']:
                    if self.job_state(candidate) == 'EXPIRED':
                        self.replace_expired(candidate)
                self.event('report_wait_hold', stage=stage)
                return False
            name = 'fe-controller-report-' + stage
            log = self.root / (name + '.log')
            argv = ['srun', '--jobid=' + str(seat['job']), '--overlap', '--exact', '--ntasks=1',
                    '--cpus-per-task=4', '--gpu_cmode=shared', self.c['python'], self.c['report_code'],
                    '--stage', stage, '--out-root', spec['out'], '--identities', spec['identities'],
                    '--json', str(output), '--md', str(output.with_suffix('.md')),
                    '--check-media', '--ffmpeg', self.c['ffmpeg']]
            body = 'set -o pipefail; PYTHONUNBUFFERED=1 ' + shlex.join(argv)
            body += ' 2>&1 | tee ' + shlex.quote(str(log)) + '; echo "EXIT_CODE=$?" >> ' + shlex.quote(str(log))
            self.state['report_task'] = {'session': name, 'log': str(log), 'started': time.time(), 'job': seat['job']}
            self.event('report_launch_intent', argv=argv)
            self.command(['tmux', 'new-session', '-d', '-s', name, 'bash -c ' + shlex.quote(body)])
            return False
        report = read(output)
        identities = read_rows(spec['identities'])
        labels = set(MODELS.values())
        if report['policy_seed'] != spec['seed']:
            raise ValueError('报告模型种子不符')
        for field in ('coverage', 'cap'):
            if len(report[field]) != 6 or {r['model'] for r in report[field]} != labels:
                raise ValueError('报告模型集合不完整：' + field)
        for row in report['coverage']:
            if (any(type(row[k]) is not int for k in ('expected', 'accepted', 'missing', 'invalid', 'no_frame_error'))
                    or row['expected'] != len(identities) or row['accepted'] != len(identities)
                    or row['missing'] != 0 or row['invalid'] != 0
                    or type(row['no_frame_error']) is not int):
                raise ValueError('报告覆盖计数不完整')
        if any(r['cap_mismatch'] != 0 or r['exec_over_cap'] != 0 for r in report['cap']):
            raise ValueError('报告步数判据失败')
        expected_media = {(label, r['key']) for label in labels for r in identities}
        actual_media = [(r['model'], r['key']) for r in report['media']]
        if (len(actual_media) != len(expected_media) or set(actual_media) != expected_media
                or any(r['status'] not in ('pass', 'no_frame_error') for r in report['media'])):
            raise ValueError('报告媒体身份或完整性不符')
        required = {'EVAL_COVERAGE', 'STAGE_MODELS', 'EVAL_REPORT', 'STAGE_TIMING',
                    'OFFICIAL_MEDIA_INPUTS', 'OFFICIAL_MEDIA'}
        names = {line.split('=', 1)[0] for line in report['verdicts']}
        if (report['stage'] != stage or report['ok'] is not True or report['errors'] != []
                or not required <= names
                or not report['verdicts'] or any('=FAIL' in x for x in report['verdicts'])
                or not all(re.match(r'^[A-Z_]+=PASS(?: |$)', x) for x in report['verdicts'])):
            raise ValueError('阶段报告契约失败')
        for line in report['verdicts']:
            print(line, flush=True)
        self.event('stage_done', stage=stage, report=str(output))
        self.state.pop('report_task')
        return True

    def tick(self):
        stage = self.state['stage']
        remaining = {m: self.remaining(stage, m) for m in self.c['models']}
        busy = set()
        for seat in self.state['seats']:
            if seat.get('session'):
                self.validate_session(seat)
                if self.alive(seat['session']):
                    busy.add(seat['model'])
                    continue
                # 最后一局可能在本轮快照与死活检查之间刚结算；核证和派发须重新读接受标记。
                remaining[seat['model']] = self.remaining(stage, seat['model'])
                log = Path(seat['log']).read_text()
                codes = re.findall(r'^EXIT_CODE=(\d+)$', log, re.M)
                if self.job_state(seat) == 'EXPIRED':
                    self.event('worker_expired', session=seat['session'], job=seat['job'],
                               exit_code=int(codes[-1]) if codes else None)
                    seat.pop('session')
                    if seat['model'] == 'smvla' and seat.get('onlykey') in remaining['smvla']:
                        seat['resume_key'] = seat['onlykey']
                    self.replace_expired(seat)
                    continue
                if not codes:
                    raise ValueError('席位退出但缺少退出码，停止受影响调度')
                code = int(codes[-1])
                if code == 3 and seat['model'] == 'smvla':
                    if not seat.get('onlykey'):
                        raise ValueError('SMVLA 环境失败缺少独立身份')
                    self.env_build_settled(seat, log)
                    seat['resume_key'] = seat['onlykey']
                    scope = f"{seat['model']}|{self.c['stages'][seat['stage']]['seed']}|{seat['onlykey']}"
                    # 仅诊断计数；是否可恢复由真实 attempt_no < 20 决定，不新增预算。
                    seat.setdefault('recoveries', {})[scope] = seat.get('recoveries', {}).get(scope, 0) + 1
                elif code == 6:
                    attempts = self.legacy_retry_settled(seat, log, remaining[seat['model']])
                    self.event('legacy_retry_resume', slot=seat['slot'], model=seat['model'],
                               stage=seat['stage'], session=seat['session'], attempts=attempts,
                               max_attempts=20)
                elif code != 0:
                    raise ValueError(f'席位失败退出码 {code}，交主会话恢复')
                seat.pop('session')
                self.event('worker_done', code=code, slot=seat['slot'])
        if not any(remaining.values()) and not busy:
            if not self.report(stage):
                return
            if stage == 'D':
                self.state['done'] = True
            else:
                self.state['stage'] = chr(ord(stage) + 1)
            self.save()
            return
        for seat in self.state['seats']:
            if seat.get('session'):
                continue
            if seat.get('submission_pending'):
                raise ValueError('占位提交待核查')
            active = self.active_state(seat)
            if active != 'RUNNING':
                if not active:
                    self.replace_expired(seat)
                continue
            available = [m for m in self.c['priority'] if remaining[m] and m not in busy]
            if not available:
                # 仅 SMVLA 按身份领取；其它模型依赖已有共享队列的原子领取。
                available = [m for m in self.c['priority'] if remaining[m] and m != 'smvla']
            if not available:
                continue
            model = available[0]
            if seat.get('resume_key'):
                if seat['resume_key'] not in remaining['smvla']:
                    seat.pop('resume_key')
                elif 'smvla' in busy:
                    continue
                else:
                    model = 'smvla'
            key = (seat.get('resume_key') or remaining[model][0]) if model == 'smvla' else None
            self.launch(seat, stage, model, key)
            seat.pop('resume_key', None)
            busy.add(model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    args = p.parse_args()
    controller = None
    try:
        config = read(args.config)
        root = Path(config['controller_root'])
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'controller.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            controller = Controller(config)
            while not controller.state['done']:
                controller.tick()
                time.sleep(15)
        print('CONTROLLER=PASS stages=4', flush=True)
        return 0
    except Exception as exc:
        if controller:
            controller.state['report_failed'] += 1
            controller.event('stopped', error=str(exc))
        print(f'CONTROLLER=FAIL error={exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
