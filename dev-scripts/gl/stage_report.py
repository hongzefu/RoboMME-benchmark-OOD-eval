"""以阶段身份白名单及共享队列 accepted 为权威生成成绩、墙钟报告。"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from select_stage_identities import read_rows

MODELS = ('perceptual-framesamp-modul', 'smvla', 'pp', 'groundsg-ground-sg-oracle',
          'groundsg-ground-sg-qwenvl', 'groundsg-ground-sg-memer')


def load_results(out_root):
    """读取原始追加行，保留重试，权威筛选由 build_report 完成。"""
    rows = []
    for path in sorted(Path(out_root).glob('rollouts/*/ood/seed*/results.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'结果追加行须为对象：{path}')
                rows.append(row)
    return rows


def timing_table(rows):
    """P95 使用最近秩；总秒数是独占一 GPU 席位的单局墙钟代理。"""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['report_model'], row['tier'])].append(row['t_end'] - row['t_start'])
    return [{'model': model, 'tier': tier, 'episodes': len(values),
             'wall_mean_s': statistics.mean(values),
             'wall_p95_s': sorted(values)[math.ceil(len(values) * .95) - 1],
             'episode_wall_sum_s': sum(values), 'gpu_seconds_proxy': sum(values)}
            for (model, tier), values in sorted(groups.items())]


def success_table(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['report_model'], row['tier'])].append(row)
    table = []
    for (model, tier), values in sorted(groups.items()):
        tasks = defaultdict(list)
        for row in values:
            tasks[row['task']].append(row['task_success'])
        successes = sum(row['task_success'] for row in values)
        table.append({'model': model, 'tier': tier, 'episodes': len(values), 'successes': successes,
                      'episode_success_rate': successes / len(values),
                      'task_macro_success_rate': statistics.mean(statistics.mean(v) for v in tasks.values())})
    return table


def result_path(root, stored):
    """允许 GL 产物搬回本机；只映射 rollouts 下的相对尾部。"""
    if not isinstance(stored, str):
        raise ValueError('accepted 缺少 result 路径')
    path = Path(stored)
    parts = path.parts
    if 'rollouts' not in parts:
        raise ValueError('result 路径不属于 rollouts')
    relative = Path(*parts[parts.index('rollouts'):])
    candidate = Path(root) / relative
    if '..' in relative.parts or not candidate.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('result 路径越界')
    return candidate


def build_report(out_root, identities, stage, *, models=MODELS):
    """不接受无 accepted 的结果；缺项、身份冲突、非法终态均明确失败。"""
    from select_stage_identities import validate_rows
    validate_rows(identities)
    if stage not in ('A', 'B', 'C', 'D'):
        raise ValueError('非法阶段')
    root = Path(out_root)
    seed = {'A': 7, 'B': 7, 'C': 0, 'D': 42}[stage]
    try:
        raw_rows = load_results(root)
    except (ValueError, OSError) as exc:
        return {'stage': stage, 'policy_seed': seed, 'ok': False, 'coverage': [], 'errors': [str(exc)],
                'success': [], 'timing': [], 'episode_wall_sum_s': 0,
                'timing_scope': '结果解析失败；未验证',
                'verdicts': [f'EVAL_COVERAGE=FAIL stage={stage} malformed=1',
                             f'STAGE_TIMING=FAIL stage={stage} models={len(models)} episodes=0 wall_h=0']}
    accepted_rows, errors, coverage = [], [], []
    for model in models:
        missing, invalid, no_frame_error = 0, 0, 0
        for ident in identities:
            marker = root / 'queue' / model / f'seed{seed}' / 'accepted' / f"ood__{ident['key']}.json"
            if not marker.exists():
                missing += 1
                continue
            try:
                accepted = json.loads(marker.read_text())
                if not isinstance(accepted, dict):
                    raise ValueError('accepted 文档须为对象')
                if accepted['dataset'] != 'ood' or accepted['key'] != ident['key']:
                    raise ValueError('accepted 身份不符')
                path = result_path(root, accepted['result'])
                row = json.loads(path.read_text())
                if not isinstance(row, dict):
                    raise ValueError('result 文档须为对象')
                if any(row[k] != ident[k] for k in ('dataset', 'task', 'episode', 'tier', 'seed',
                                                     'candidate', 'source_episode', 'spec_sha256', 'key')):
                    raise ValueError('结果身份不符')
                if row['policy_seed'] != seed or row['policy_label'] != model:
                    raise ValueError('模型标签或模型种子不符')
                expected_model = 'groundsg' if model.startswith('groundsg-') else model
                if row['model'] != expected_model:
                    raise ValueError('model 与 policy_label 不符')
                if type(row['attempt']) is not int or row['attempt'] != accepted['attempt']:
                    raise ValueError('accepted 尝试号不符')
                if (row['status'] not in ('success', 'fail', 'timeout', 'error') or accepted['status'] != row['status']
                        or type(row['task_success']) is not int or row['task_success'] not in (0, 1)
                        or row['task_success'] != int(row['status'] == 'success')
                        or row['infra'] is not False or row['run_blocked'] is not False
                        or row['budget_exhausted'] is not False):
                    raise ValueError('非法或基础设施终态')
                if row['status'] == 'error':
                    if not (row.get('error') or row.get('infra_reason')):
                        raise ValueError('错误终局缺少原因')
                    if any(path.parent.rglob('*.mp4')) or any(path.parent.rglob('*.mkv')):
                        raise ValueError('有视频错误终局不属于无帧例外')
                    trace = path.parent / 'trace.jsonl'
                    if row.get('no_frame') is not True and trace.exists():
                        trace_rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
                        end = trace_rows[-1] if trace_rows else {}
                        if not isinstance(end, dict) or not (end.get('no_frame') is True or end.get('frames_recorded') == 0):
                            raise ValueError('有帧错误终局不属于无帧例外')
                    # 与旧报告一致：无 trace 时须写明原因，按无帧例外单列。
                if any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) for k in ('t_start', 't_end')) \
                        or row['t_start'] <= 0 or row['t_end'] < row['t_start']:
                    raise ValueError('墙钟字段非法')
                matches = [r for r in raw_rows if r.get('policy_label') == model and r.get('policy_seed') == seed
                           and r.get('dataset') == 'ood' and r.get('key') == ident['key']
                           and r.get('attempt') == accepted['attempt']]
                if len(matches) != 1 or matches[0] != row:
                    raise ValueError('results.jsonl 缺失、重复或与 result.json 不一致')
                accepted_rows.append({**row, 'report_model': model})
                no_frame_error += int(row['status'] == 'error')
            except (KeyError, ValueError, OSError, TypeError) as exc:
                invalid += 1
                errors.append(f'{model}/{ident["key"]}: {exc}')
        coverage.append({'model': model, 'expected': len(identities), 'missing': missing,
                         'invalid': invalid, 'accepted': len(identities) - missing - invalid,
                         'no_frame_error': no_frame_error})
    models_complete = len(models) == len(MODELS) and set(models) == set(MODELS)
    ok = models_complete and all(c['missing'] == c['invalid'] == 0 for c in coverage)
    wall = sum(r['t_end'] - r['t_start'] for r in accepted_rows)
    lines = [f'EVAL_COVERAGE={"PASS" if c["missing"] == c["invalid"] == 0 else "FAIL"} '
             f'stage={stage} model={c["model"]} expected={c["expected"]} accepted={c["accepted"]} '
             f'missing={c["missing"]} invalid={c["invalid"]} no_frame_error={c["no_frame_error"]}' for c in coverage]
    lines.append(f'STAGE_MODELS={"PASS" if models_complete else "FAIL"} expected=6 actual={len(models)}')
    lines.append(f'STAGE_TIMING={"PASS" if ok else "FAIL"} stage={stage} models={len(models)} '
                 f'episodes={len(accepted_rows)} wall_h={wall / 3600:.6f}')
    return {'stage': stage, 'policy_seed': seed, 'ok': ok, 'coverage': coverage, 'errors': errors,
            'success': success_table(accepted_rows), 'timing': timing_table(accepted_rows),
            'episode_wall_sum_s': wall, 'timing_scope': '仅阶段 accepted 局；GPU 秒为单局墙钟之和代理，不含加载及基础设施失败',
            'verdicts': lines}


def markdown(report):
    lines = [f'# 阶段 {report["stage"]} 报告', '', report['timing_scope'], '',
             '|模型|档位|局数|成功数|局级成功率|任务宏平均|均值秒|P95秒|GPU秒代理|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    timing = {(r['model'], r['tier']): r for r in report['timing']}
    for row in report['success']:
        t = timing[(row['model'], row['tier'])]
        lines.append(f'|{row["model"]}|{row["tier"]}|{row["episodes"]}|{row["successes"]}|'
                     f'{row["episode_success_rate"]:.6f}|{row["task_macro_success_rate"]:.6f}|'
                     f'{t["wall_mean_s"]:.3f}|{t["wall_p95_s"]:.3f}|{t["gpu_seconds_proxy"]:.3f}|')
    lines.extend(['', '```', *report['verdicts'], '```', '', *report['errors']])
    if 'seat_steps' in report:
        lines.extend(['', '## 席位整程', '', '席位可跨阶段；下表不计入阶段单局墙钟。', '',
                      '|作业步|开始|结束|整程秒|分配GPU秒|', '|---|---|---|---:|---:|'])
        for step in report['seat_steps']:
            lines.append(f'|{step["job_id"]}|{step["start"]}|{step["end"]}|{step["wall_s"]}|{step["allocated_gpu_s"]}|')
    return '\n'.join(lines) + '\n'


def read_sacct(path):
    """读取 sacct -P 的 JobIDRaw,Start,End,ElapsedRaw,AllocTRES；只计数字工作步。"""
    lines = [line.split('|') for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError('sacct 为空')
    header = lines[0]
    required = {'JobIDRaw', 'Start', 'End', 'ElapsedRaw', 'AllocTRES'}
    if not required <= set(header):
        raise ValueError('sacct 缺少 JobIDRaw,Start,End,ElapsedRaw,AllocTRES 表头')
    steps, seen = [], set()
    for values in lines[1:]:
        if len(values) != len(header):
            raise ValueError('sacct 字段数量不符')
        row = dict(zip(header, values))
        job = row['JobIDRaw']
        if '.' not in job or not job.rsplit('.', 1)[1].isdigit():
            continue
        if job in seen:
            raise ValueError('sacct 重复工作步')
        seen.add(job)
        duration = (datetime.fromisoformat(row['End']) - datetime.fromisoformat(row['Start'])).total_seconds()
        elapsed = int(row['ElapsedRaw'])
        resources = dict(item.split('=', 1) for item in row['AllocTRES'].split(',') if '=' in item)
        gpu = int(resources['gres/gpu'])
        if duration < 0 or elapsed < 0 or gpu < 1:
            raise ValueError('sacct 非法时长或 GPU 数量')
        steps.append({'job_id': job, 'start': row['Start'], 'end': row['End'],
                      'wall_s': duration, 'elapsed_s': elapsed, 'allocated_gpu_s': elapsed * gpu})
    if not steps:
        raise ValueError('sacct 没有数字工作步')
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', required=True, type=Path)
    p.add_argument('--identities', required=True, type=Path)
    p.add_argument('--stage', required=True, choices=list('ABCD'))
    p.add_argument('--md', required=True, type=Path)
    p.add_argument('--json', type=Path)
    p.add_argument('--sacct', type=Path)
    args = p.parse_args()
    try:
        report = build_report(args.out_root, read_rows(args.identities), args.stage)
        if args.sacct:
            report['seat_steps'] = read_sacct(args.sacct)
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(report))
        output = args.json or args.md.with_suffix('.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('\n'.join(report['verdicts']))
        return 0 if report['ok'] else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'STAGE_TIMING=FAIL stage={args.stage} error={exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
