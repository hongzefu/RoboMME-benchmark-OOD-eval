"""由主会话明确配置的四席接续；失败封闭，不提供代理自动唤醒。"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def run(argv, check=True):
    return subprocess.run(argv, text=True, capture_output=True, check=check, timeout=300)


class Controller:
    def __init__(self, config, command=run):
        self.c, self.command = config, command
        self.root = Path(config['controller_root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'state.json'
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.state = read(self.path) if self.path.exists() else {
            'config_sha256': digest,
            'stage': 'A', 'seats': config['seats'], 'serial': 0, 'done': False,
            'missing': 0, 'media_failed': 0, 'report_failed': 0}
        if self.state['config_sha256'] != digest:
            raise ValueError('配置指纹改变，禁止接手')
        if len(self.state['seats']) != 4:
            raise ValueError('仅允许主会话指定的四席')
        if {s['slot'] for s in self.state['seats']} != set(range(4)):
            raise ValueError('席位编号必须互异且为 0～3')

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w') as f:
            f.write(json.dumps(self.state, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)

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
        rows = [json.loads(x) for x in Path(spec['identities']).read_text().splitlines() if x.strip()]
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
            if (accepted['key'] != row['key'] or data['key'] != row['key']
                    or accepted['attempt'] != data['attempt']
                    or accepted['status'] != data['status']
                    or data['status'] not in ('success', 'fail', 'timeout', 'error')):
                raise ValueError('accepted 与结果契约不符')
        return missing

    def replace_expired(self, seat):
        job = str(seat['job'])
        active = self.command(['squeue', '-h', '-j', job, '-o', '%T']).stdout.strip()
        if active:
            return False
        states = self.command(['sacct', '-n', '-X', '-j', job, '-o', 'State', '-P']).stdout.splitlines()
        if not states or any(x.strip().split('|')[0] != 'TIMEOUT' for x in states if x.strip()):
            raise ValueError('作业未确认到期，不申请接替')
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
        seat['submission_pending'] = False
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
        body = 'set -o pipefail; PYTHONUNBUFFERED=1 ' + shlex.join(argv)
        body += ' 2>&1 | tee ' + shlex.quote(str(log)) + '; echo "EXIT_CODE=$?" >> ' + shlex.quote(str(log))
        # 起跑意图先保存；重启只接手，不重复创建。
        seat.update(session=name, log=str(log), model=model, stage=stage, onlykey=key, pending=True)
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
            seat = next((s for s in self.state['seats'] if self.command(
                ['squeue', '-h', '-j', str(s['job']), '-o', '%T']).stdout.strip() == 'RUNNING'), None)
            if not seat:
                raise ValueError('没有可执行报告的 RUNNING 占位作业')
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
        required = {'EVAL_COVERAGE', 'STAGE_MODELS', 'EVAL_REPORT', 'STAGE_TIMING',
                    'OFFICIAL_MEDIA_INPUTS', 'OFFICIAL_MEDIA'}
        names = {line.split('=', 1)[0] for line in report['verdicts']}
        if (report['stage'] != stage or report['ok'] is not True or report['errors'] != []
                or not required <= names
                or not report['verdicts'] or any('=FAIL' in x for x in report['verdicts'])
                or not all('=PASS' in x for x in report['verdicts'])):
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
                if self.alive(seat['session']):
                    busy.add(seat['model'])
                    continue
                log = Path(seat['log']).read_text()
                codes = re.findall(r'^EXIT_CODE=(\d+)$', log, re.M)
                if not codes:
                    raise ValueError('席位退出但缺少退出码，停止受影响调度')
                code = int(codes[-1])
                if code == 3 and seat['model'] == 'smvla':
                    if not re.search(r'RUN_BLOCKED reason=env_build .*key=' + re.escape(str(seat.get('onlykey'))) + r' attempt=\d+ settled=1', log):
                        raise ValueError('退出码 3 缺少环境创建故障证据')
                    if not seat.get('onlykey'):
                        raise ValueError('SMVLA 环境失败缺少独立身份')
                    key = seat['onlykey']
                    count = seat.setdefault('recoveries', {}).get(key, 0) + 1
                    seat['recoveries'][key] = count
                    if count >= 20:
                        raise ValueError('SMVLA 恢复达到上限')
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
            active = self.command(['squeue', '-h', '-j', str(seat['job']), '-o', '%T']).stdout.strip()
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
            key = remaining[model][0] if model == 'smvla' else None
            self.launch(seat, stage, model, key)
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
