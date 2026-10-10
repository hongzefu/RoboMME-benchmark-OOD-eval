"""以阶段身份白名单及共享队列 accepted 为权威生成成绩、墙钟报告。"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import importlib.util
import hashlib
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from select_stage_identities import read_rows

MODELS = ('perceptual-framesamp-modul', 'smvla', 'pp', 'groundsg-ground-sg-oracle',
          'groundsg-ground-sg-qwenvl', 'groundsg-ground-sg-memer')


def helper(name, relative):
    """只从同仓固定路径读取既有判据，禁止替换实现。"""
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def object_json(path):
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f'文档须为对象：{path}')
    return doc


def relocated_path(root, stored):
    """搬运路径只允许 rollouts／queue 的固定尾部映射。"""
    parts = Path(stored).parts
    anchor = next((p for p in ('rollouts', 'queue') if p in parts), None)
    if anchor is None:
        raise ValueError('原始文件路径缺少合法根')
    tail = Path(*parts[parts.index(anchor):])
    target = Path(root) / tail
    if '..' in tail.parts or not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('原始文件路径越界')
    return target


def bind_trace(raw, row, checker):
    """按 episode._classify 归一化终态；完整绑定真实 trace 的身份契约。"""
    meta = object_json(raw / 'meta.json')
    recorded = meta.get('identity')
    if not isinstance(recorded, dict):
        raise ValueError('录制器 meta.identity 缺失')
    for key in ('dataset', 'task', 'tier', 'seed', 'candidate', 'spec_sha256', 'source_episode',
                'key', 'attempt', 'policy_seed', 'episode', 'max_steps', 'strict_cap'):
        if key not in recorded or recorded[key] != row[key] or type(recorded[key]) is not type(row[key]):
            raise ValueError(f'录制器 meta.identity 不符：{key}')
    if type(recorded.get('builder_episode')) is not int or recorded['builder_episode'] != row['episode']:
        raise ValueError('录制器 meta.identity builder 局号不符')
    trace = checker.read_rows(raw / 'trace.jsonl')
    if not trace or not all(isinstance(v, dict) for v in trace) or trace[0].get('kind') != 'header' or trace[-1].get('kind') != 'end':
        raise ValueError('trace 不完整')
    header, end = trace[0], trace[-1]
    identity = header.get('identity')
    if not isinstance(identity, dict):
        raise ValueError('trace identity 缺失')
    for key in ('dataset', 'task', 'tier', 'seed', 'key', 'source_episode', 'attempt', 'policy_seed'):
        if key not in identity or identity[key] != row[key] or (key in ('attempt', 'policy_seed', 'seed') and type(identity[key]) is not int):
            raise ValueError(f'trace 身份不符：{key}')
    if identity.get('builder_episode') != row['episode'] or type(identity.get('builder_episode')) is not int:
        raise ValueError('trace builder 局号不符')
    for key in ('candidate', 'spec_sha256'):
        # 现有 trace 格式未写两字段；若写了则必须逐键核对，结果自身已与阶段清单绑定。
        if key in identity and identity[key] != row[key]:
            raise ValueError(f'trace 身份不符：{key}')
    for doc in (header, end):
        if 'policy_seed' not in doc or type(doc['policy_seed']) is not int or doc['policy_seed'] != row['policy_seed']:
            raise ValueError('trace header/end 模型种子不符或缺失')
    status = end.get('status')
    if status not in ('success', 'fail', 'timeout', 'error'):
        raise ValueError('trace 终态未知')
    normalized = 'timeout' if end.get('cap_hit') is True else ('fail' if status == 'error' else status)
    if row['status'] != normalized:
        # 已登记无帧错误终局是历史例外，仍不算成功。
        no_frame = end.get('no_frame') is True or end.get('frames_recorded') == 0
        if not (row['status'] == status == 'error' and no_frame and not end.get('cap_hit')):
            raise ValueError('trace 归一化终态不符')
    return header, end, normalized


def verify_recovery(raw, root, row, ff, checker):
    """恢复证据须绑定原输入字节、身份与实际全解码视频，标志位不构成证据。"""
    recovery = raw / 'recovered-video'
    proof = object_json(recovery / 'recovery.json')
    render = object_json(recovery / 'render.json')
    if proof['render'] != render or proof['model'] != row['model'] or proof['task_success'] != row['task_success']:
        raise ValueError('恢复模型、成功字段或渲染清单不符')
    if proof['result_sha256'] != checker.sha256_file(raw / 'result.json'):
        raise ValueError('恢复 result 指纹不符')
    originals = proof['original_files']
    if not isinstance(originals, dict) or not originals:
        raise ValueError('恢复原文件证据缺失')
    verified = set()
    for stored, fingerprint in originals.items():
        path = relocated_path(root, stored)
        if not isinstance(fingerprint, dict) or type(fingerprint.get('bytes')) is not int:
            raise ValueError('原始文件指纹非法')
        size = fingerprint['bytes']
        if path.name == 'results.jsonl':
            # 追加日志只冻结恢复时的历史前缀，允许 B 后续合法追加。
            with path.open('rb') as stream:
                content = stream.read(size)
            if len(content) != size or not content.endswith(b'\n') or hashlib.sha256(content).hexdigest() != fingerprint['sha256']:
                raise ValueError('恢复追加日志历史前缀不符')
        elif path.stat().st_size != size or checker.sha256_file(path) != fingerprint['sha256']:
            raise ValueError(f'恢复原始文件指纹不符：{path.name}')
        verified.add(path.resolve())
    required = [raw / name for name in ('result.json', 'trace.jsonl', 'arrays.npz', 'front.mkv', 'wrist.mkv',
                                       'frames-front.jsonl', 'frames-wrist.jsonl', 'meta.json')]
    required.append(raw.parent.parent / 'results.jsonl')
    if not all(path.resolve() in verified for path in required):
        raise ValueError('恢复原文件清单缺少核心输入')
    header, end, normalized = bind_trace(raw, row, checker)
    identity = header['identity']
    if render['identity'] != identity:
        raise ValueError('恢复 trace 与 sidecar 身份不符')
    for key in ('dataset', 'task', 'tier', 'seed', 'key', 'source_episode', 'attempt', 'policy_seed'):
        if identity.get(key) != row[key]:
            raise ValueError(f'恢复身份不符：{key}')
    if identity.get('builder_episode') != row['episode'] or header.get('policy_seed', identity.get('policy_seed')) != row['policy_seed']:
        raise ValueError('恢复 builder 局号或模型种子不符')
    for key in ('candidate', 'spec_sha256'):
        if key in render['identity'] and render['identity'][key] != row[key]:
            raise ValueError(f'恢复渲染身份不符：{key}')
    if render['status'] != end['status'] or render.get('terminal_reason') != normalized or render['policy_seed'] != row['policy_seed']:
        raise ValueError('恢复 trace 终态或渲染种子不符')
    for key, name in (('source_trace', 'trace.jsonl'), ('source_arrays', 'arrays.npz')):
        fp = render[key]
        source = raw / name
        if fp['size'] != source.stat().st_size or fp['sha256'] != checker.sha256_file(source):
            raise ValueError('渲染来源指纹不符')
    for group in ('streams', 'index'):
        for name, fp in render['source_media'][group].items():
            source = raw / name
            if source.resolve() not in verified or fp['size'] != source.stat().st_size or fp['sha256'] != checker.sha256_file(source):
                raise ValueError('渲染帧来源指纹不符')
    videos = list(recovery.glob('*.mp4'))
    if len(videos) != 1:
        raise ValueError('恢复视频数量不符')
    video = videos[0]
    prefix = f"{row['task']}_ep{row['episode']}_{row['status']}_"
    if not video.name.startswith(prefix) or not video.name.endswith(f"_{row['tier']}.mp4") or video.name != render['out_rel']:
        raise ValueError('恢复视频终态命名不符')
    fingerprint = render['output_fingerprint']
    sha = checker.sha256_file(video)
    if sha != proof['video_sha256'] or sha != fingerprint['sha256'] or video.stat().st_size != proof['video_bytes'] or video.stat().st_size != fingerprint['size']:
        raise ValueError('恢复视频指纹不符')
    frames = checker.decoded_frames(ff, video)
    expected = end.get('frames_recorded', checker.sidecar_frames(render))
    if type(expected) is not int or frames <= 0 or frames != expected or frames != checker.sidecar_frames(render):
        raise ValueError('恢复视频全解码帧数不符')
    return {'status': 'pass', 'recovered': True, 'path': str(raw), 'frames_decoded': frames}


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


def build_report(out_root, identities, stage, *, models=MODELS, check_media=False, ffmpeg=None):
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
    cap_checker = helper('stage_cap_checker', 'gl/eval_report.py')
    media_checker = helper('stage_media_checker', 'media/official_media_check.py') if check_media else None
    media_records, cap_records = [], []
    for model in models:
        missing, invalid, no_frame_error, cap_mismatch, exec_over_cap = 0, 0, 0, 0, 0
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
                cap_issues = cap_checker.cap_problems(row, 1800, {'path': str(path.parent)})
                if row.get('strict_cap') is not True:
                    cap_issues.append('strict_cap!=True')
                cap_mismatch += int(bool(cap_issues))
                over = type(row.get('exec_steps')) is not int or row['exec_steps'] > 1800 or row['exec_steps'] < 0
                exec_over_cap += int(over)
                if cap_issues or over:
                    errors.append(f'{model}/{ident["key"]}: cap={cap_issues} exec_over_cap={int(over)}')
                if check_media:
                    try:
                        if not ffmpeg:
                            raise ValueError('媒体验收缺少 ffmpeg')
                        raw = path.parent
                        if (raw / 'recovered-video' / 'recovery.json').exists():
                            media = verify_recovery(raw, root, row, ffmpeg, media_checker)
                        else:
                            bind_trace(raw, row, media_checker)
                            media = media_checker.verify_new_episode(raw, raw.parent.parent, ffmpeg, dataset='ood',
                                                                     seed=seed, expected_policy_seed=seed)
                        if media['status'] not in ('pass', 'no_frame_error'):
                            raise ValueError(str(media.get('reasons', '媒体验收失败')))
                    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
                        media = {'status': 'fail', 'error': str(exc)}
                        errors.append(f'{model}/{ident["key"]}: media={exc}')
                    media_records.append({'model': model, 'key': ident['key'], **media})
                accepted_rows.append({**row, 'report_model': model})
                no_frame_error += int(row['status'] == 'error')
            except (KeyError, ValueError, OSError, TypeError) as exc:
                invalid += 1
                errors.append(f'{model}/{ident["key"]}: {exc}')
        coverage.append({'model': model, 'expected': len(identities), 'missing': missing,
                         'invalid': invalid, 'accepted': len(identities) - missing - invalid,
                         'no_frame_error': no_frame_error})
        cap_records.append({'model': model, 'cap_mismatch': cap_mismatch, 'exec_over_cap': exec_over_cap})
    models_complete = len(models) == len(MODELS) and set(models) == set(MODELS)
    ok = models_complete and all(c['missing'] == c['invalid'] == 0 for c in coverage)
    ok = ok and all(c['cap_mismatch'] == c['exec_over_cap'] == 0 for c in cap_records)
    media_ok = len(media_records) == len(identities) * len(models) and all(r['status'] in ('pass', 'no_frame_error') for r in media_records)
    if check_media:
        ok = ok and media_ok
    wall = sum(r['t_end'] - r['t_start'] for r in accepted_rows)
    lines = [f'EVAL_COVERAGE={"PASS" if c["missing"] == c["invalid"] == 0 else "FAIL"} '
             f'stage={stage} model={c["model"]} expected={c["expected"]} accepted={c["accepted"]} '
             f'missing={c["missing"]} invalid={c["invalid"]} no_frame_error={c["no_frame_error"]}' for c in coverage]
    lines.append(f'STAGE_MODELS={"PASS" if models_complete else "FAIL"} expected=6 actual={len(models)}')
    for cap, cov in zip(cap_records, coverage):
        passed = cap['cap_mismatch'] == cap['exec_over_cap'] == cov['missing'] == cov['invalid'] == 0
        lines.append(f'EVAL_REPORT={"PASS" if passed else "FAIL"} stage={stage} model={cap["model"]} '
                     f'cap_mismatch={cap["cap_mismatch"]} exec_over_cap={cap["exec_over_cap"]}')
    for name in ('OFFICIAL_MEDIA_INPUTS', 'OFFICIAL_MEDIA'):
        lines.append(f'{name}={"PASS" if media_ok else "FAIL"} stage={stage} total={len(media_records)} '
                     f'fail={sum(r["status"] == "fail" for r in media_records)}') if check_media else lines.append(f'{name}=UNVERIFIED stage={stage} enabled=0')
    lines.append(f'STAGE_TIMING={"PASS" if ok else "FAIL"} stage={stage} models={len(models)} '
                 f'episodes={len(accepted_rows)} wall_h={wall / 3600:.6f}')
    return {'stage': stage, 'policy_seed': seed, 'ok': ok, 'coverage': coverage, 'errors': errors,
            'success': success_table(accepted_rows), 'timing': timing_table(accepted_rows),
            'episode_wall_sum_s': wall, 'timing_scope': '仅阶段 accepted 局；GPU 秒为单局墙钟之和代理，不含加载及基础设施失败',
            'verdicts': lines, 'cap': cap_records, 'media': media_records}


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
    p.add_argument('--check-media', action='store_true')
    p.add_argument('--ffmpeg')
    args = p.parse_args()
    try:
        report = build_report(args.out_root, read_rows(args.identities), args.stage,
                              check_media=args.check_media, ffmpeg=args.ffmpeg)
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
