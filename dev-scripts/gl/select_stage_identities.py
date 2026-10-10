"""只读身份清单后按阶段筛选；不导入环境、不触发 reset。"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

FIELDS = {'dataset', 'task', 'episode', 'builder_episode', 'tier', 'seed', 'candidate',
          'source_episode', 'spec_sha256', 'key'}
TIERS = ('xhard1', 'xhard2', 'xhard3', 'xhard4', 'xhard5')


def validate_rows(rows):
    """拒绝缺字段、非法身份和重复身份；原行保持不变。"""
    seen, episodes = set(), set()
    if not rows:
        raise ValueError('身份清单为空')
    for row in rows:
        if set(row) != FIELDS:
            raise ValueError('身份字段不完整或存在未知字段')
        integers = ('episode', 'builder_episode', 'seed', 'candidate')
        if any(type(row[k]) is not int or row[k] < 0 for k in integers):
            raise ValueError('局号、种子、候选须为非负整数')
        if (row['dataset'] != 'ood' or row['tier'] not in TIERS
                or not isinstance(row['task'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', row['task'])
                or row['episode'] != row['builder_episode'] or row['source_episode'] is not None
                or not isinstance(row['spec_sha256'], str)
                or not re.fullmatch(r'[0-9a-f]{64}', row['spec_sha256'])
                or row['key'] != f"{row['task']}_{row['tier']}_{row['seed']}"):
            raise ValueError('非法 ood 身份')
        key = (row['dataset'], row['key'])
        ep = (row['task'], row['episode'])
        if key in seen or ep in episodes:
            raise ValueError('重复身份或 builder 局号')
        seen.add(key)
        episodes.add(ep)
    return rows


def read_rows(path):
    return validate_rows([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()])


def select_rows(rows, stage, per_task=5, *, minus=None, exclude_tasks=('MoveCube', 'InsertPeg')):
    """A 按规范档位轮转；B 精确差集；C/D 全量。"""
    validate_rows(rows)
    rows = [r for r in rows if r['task'] not in exclude_tasks]
    if stage not in ('A', 'B', 'C', 'D') or per_task < 1:
        raise ValueError('阶段或每任务局数非法')
    if stage == 'B':
        if minus is None:
            raise ValueError('B 必须提供 A 清单')
        validate_rows(minus)
        full = {r['key']: r for r in rows}
        if any(r['key'] not in full or r != full[r['key']] for r in minus):
            raise ValueError('减项不是全量清单的精确子集')
        expected = select_rows(rows, 'A', per_task, exclude_tasks=())
        if {r['key'] for r in minus} != {r['key'] for r in expected}:
            raise ValueError('减项不等于规范 A 清单')
        removed = {r['key'] for r in minus}
        return [r for r in rows if r['key'] not in removed]
    if minus is not None:
        raise ValueError('只有 B 接受减项')
    if stage != 'A':
        return rows
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cells[row['task']][row['tier']].append(row)
    chosen = set()
    for task, tiers in cells.items():
        for values in tiers.values():
            values.sort(key=lambda r: r['builder_episode'])
        ordered = [r for i in range(max(map(len, tiers.values()))) for tier in TIERS
                   if i < len(tiers.get(tier, ())) for r in [tiers[tier][i]]]
        if len(ordered) < per_task:
            raise ValueError(f'{task} 不足 {per_task} 局')
        chosen.update(r['key'] for r in ordered[:per_task])
    return [r for r in rows if r['key'] in chosen]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identities', required=True, type=Path)
    p.add_argument('--stage', required=True, choices=list('ABCD'))
    p.add_argument('--per-task', type=int, default=5)
    p.add_argument('--exclude-tasks', default='MoveCube,InsertPeg')
    p.add_argument('--minus', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    try:
        rows = select_rows(read_rows(args.identities), args.stage, args.per_task,
                           minus=read_rows(args.minus) if args.minus else None,
                           exclude_tasks=args.exclude_tasks.split(','))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        seed = {'A': 7, 'B': 7, 'C': 0, 'D': 42}[args.stage]
        print(f'STAGE_IDENTITIES=PASS stage={args.stage} seed={seed} hard_verify=0 ood={len(rows)} '
              f'tasks={len({r["task"] for r in rows})} tiers_covered={len({(r["task"], r["tier"]) for r in rows})} out={args.out}')
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'STAGE_IDENTITIES=FAIL stage={args.stage} error={exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
