#!/usr/bin/env python3
"""v8 各档总表页的浏览器检查（v8 方案第一部分 §3「各档总表页」）：不播放视频、不修改服务端。

由 V7 各档总表页检查器（已删除，git 历史可取回）改写：

- 数据一律来自 ``/api/subgoals``（``subgoal_lengths.py`` 从 v8 规格与真实 h5 统计）与 ``/api/catalog``；
  ``subgoals.json`` 缺失时服务端返回 ``{}``，本检查器据此判 FAIL，不允许页面静默缺内容；
- 成功率格（2026-10-02 起评估接入）：按目录每格 ``rates.new``（xhard0 另有 ``rates.old``）推算「百分比」与「成功 / 局数」，与页面逐格一致；执行步上限（原任务页表格行，已撤下，不再核对）等于 ``subgoals.json`` 的 ``max_steps``
  （规格 ``exec_cap``／xhard0 1300）；
- 配置格逐维与表 1 一致（``site_catalog.TABLE1``），不对照任何计划表格；去掉 v7 的 PatternLock 定值断言；
- 可选 ``--delivery``：逐格执行步均值／最小／最大与 ``delivery.json`` 的 ``exec_steps`` 再核一遍（与交付 h5 一致）。

逐格覆盖目录里全部 (任务, 档)（完整根 43 新值格 + 16 xhard0 格 = 59）。只核「各档总表」页（任务页对比表已按用户要求撤下，只断言其隐藏）；截图写 ``--shots``。
末行打印 ``V8_ORACLE_BROWSER=PASS|FAIL cells=<n> missing=<n>``（另附 mismatch、page_errors）。**判定行按前缀匹配**：
``V8_ORACLE_BROWSER=PASS cells=59 missing=0`` 之后可能追加键。任何中断都记入 problems 并照打判定行（FAIL）。
``--expect-cells`` 缺省由目录推出：目录新值格等于完整格表（``V9_CELLS``）时为
``len(格表) + 16``（= 59），子表时为目录新值格数 + 16（xhard0 每任务一格）。

**端口**：``--port`` 给出时替换 ``--base`` 里的端口（主机不变；V9 独立站缺省 8082，V8 正式站 8081 不动）。

**V9 站点**（v9 方案第一部分 §5「站点」行，目录 ``eval.mode == "v9-reuse"`` 时自动启用）：逐局按 ``eval_origin`` 数
「复用／新评／置空」——复用、新评两类须两模型都有终态，否则算置空；末行另打印
``V9_SITE=PASS|FAIL cells=<n> missing=<n> eval_reused=<n> eval_new=<n> eval_empty=<n> port=<port>``（在
``V8_ORACLE_BROWSER`` 行之后）。PASS 要求总表检查通过、``eval_empty == 0``、复用 + 新评 = 新值局总数、计数与目录
``eval.reuse.counts`` 一致，给了 ``--expect-reused``／``--expect-new`` 时还要相等。V8 目录（无 ``eval.mode``）只打印原判定行。

    uv run --no-project --with playwright python scripts/injection-dev/site/oracle_browser_check.py \\
      --base http://127.0.0.1:8081 --shots artifacts/newtask-v8/site-checks/oracle \\
      --delivery artifacts/newtask-v8/gen1/delivery.json

    # V9（阶段 4c）
    uv run --no-project --with playwright python scripts/injection-dev/site/oracle_browser_check.py \\
      --port 8082 --shots artifacts/newtask-v9/site-checks/oracle \\
      --delivery artifacts/newtask-v9/delivery/delivery.local.json --expect-reused 720 --expect-new 80
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# playwright 只在 main() 里导入：纯函数（V9 计数、端口替换）可在没有 playwright 的环境里单测

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("site_catalog", HERE / "site_catalog.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

METRICS = ["config", "total", "demo", "exec", "policy-simplememvla", "policy-mmevla", "n"]  # 历史数据键：FrameSamp+Modulation 的页面 ID（site_catalog.EVAL_POLICY）
TASK_METRICS = METRICS + ["old-policy-simplememvla", "old-policy-mmevla", "max_steps"]  # 历史数据键：FrameSamp+Modulation 的页面 ID（site_catalog.EVAL_POLICY）

CHECK_JS = """({cat, oracle, metrics, scope}) => {
    const root = document.querySelector(scope), bad = [], missing = [];
    const nums = s => (s.match(/\\d+(?:\\.\\d+)?/g) || []).map(Number);
    const tasks = scope === '#matrix' ? cat.tasks.filter(t => root.querySelector(`td[data-task="${t.id}"]`)) : cat.tasks;
    let checked = 0;
    for (const task of tasks) {
        for (const tier of cat.tiers) {
            const c = task.tiers[tier], gt = (oracle[task.id] || {})[tier];
            if (c && !gt) { missing.push(task.id + '/' + tier + ' subgoals 无该格'); continue; }
            if (c) checked++;
            for (const metric of metrics) {
                const key = `${task.id}/${tier}/${metric}`;
                const cells = root.querySelectorAll(`td[data-task="${task.id}"][data-tier="${tier}"][data-metric="${metric}"]`);
                if (cells.length !== 1) { missing.push(key + ' 单元格数量 ' + cells.length); continue; }
                const cell = cells[0], text = cell.textContent.trim();
                if (!c) { if (text !== '无该档位') bad.push(key + ' 缺档标记不符'); continue; }
                if (metric === 'config') {
                    const dims = [...cell.querySelectorAll('[data-dim]')].map(d => [d.dataset.dim, JSON.parse(d.dataset.values)]);
                    bad.push(...(window.__cfgCheck(task.id, tier, dims, text) || []).map(x => key + ' ' + x));
                    continue;
                }
                if (metric.includes('policy-')) {
                    const old = metric.startsWith('old-');
                    if (old && tier !== 'xhard0') { if (text !== '—') bad.push(key + ' 旧入口适用范围不符'); }
                    else {
                        const pid = metric.replace(/^(old-)?policy-/, ''), counts = ((c.rates || {})[old ? 'old' : 'new'] || {})[pid];
                        const n = c.episodes.length, s = (counts || {}).success || 0;
                        if (!counts || !Object.keys(counts).length) bad.push(key + ' 成功率格缺评估数据');
                        else if (!text.includes(Math.round(100 * s / n) + '%') || !text.includes(s + ' / ' + n + ' 局成功'))
                            bad.push(key + ' 成功率不符：' + text + ' vs ' + s + '/' + n);
                    }
                    continue;
                }
                const value = cell.querySelector('.oracle-value')?.textContent || '', range = cell.querySelector('.oracle-range')?.textContent || '';
                if (metric === 'n' || metric === 'max_steps') {
                    if (gt[metric] == null ? text !== '—' : JSON.stringify(nums(value)) !== JSON.stringify([gt[metric]])) bad.push(key + ' 数值不符：' + text);
                } else {
                    const p = metric === 'exec' ? '' : metric + '_';
                    if (JSON.stringify(nums(value)) !== JSON.stringify([gt[p + 'mean']]) || JSON.stringify(nums(range)) !== JSON.stringify([gt[p + 'min'], gt[p + 'max']])) bad.push(key + ' 长度不符');
                }
            }
        }
    }
    return {bad, missing, checked};
}"""


def install_cfg_check(page, table1: dict) -> None:
    """把表 1 参照值注入页面，供 CHECK_JS 逐维比较（定值相等、区间落在内、维度集合相同）。"""
    page.evaluate("""([table1, noDim, x0]) => {
        window.__cfgCheck = (task, tier, dims, text) => {
            const want = table1[task] || {}, bad = [];
            if (tier === 'xhard0') { if (!text.includes(x0.slice(0, 8))) bad.push('xhard0 配置说明缺失'); return bad; }
            const names = Object.keys(want).filter(d => want[d][tier] !== undefined);
            if (!names.length) { if (!text.includes(noDim.slice(0, 8))) bad.push('无梯度说明缺失'); return bad; }
            if (JSON.stringify(dims.map(d => d[0]).sort()) !== JSON.stringify(names.slice().sort())) bad.push('配置维度不符');
            for (const [dim, values] of dims) {
                const w = (want[dim] || {})[tier];
                const okv = v => Array.isArray(w) ? v >= w[0] && v <= w[1] : v === w;
                if (!values.length || !values.every(okv)) bad.push(`配置 ${dim}=${JSON.stringify(values)} 与表 1 不符`);
            }
            return bad;
        };
    }""", [table1, C.NO_DIM_TEXT, C.XHARD0_TEXT])


def delivery_exec(path: Path) -> dict:
    """``delivery.json`` 逐格执行步统计（与 subgoal_lengths 同口径）。"""
    data = C.load_delivery(path)
    by_cell: dict = defaultdict(list)
    for row in data["rows"]:
        by_cell[(row["task"], row["tier"])].append(row["exec_steps"])
    return {k: {"mean": round(statistics.mean(v), 1), "min": min(v), "max": max(v), "n": len(v)} for k, v in by_cell.items()}


def with_port(base: str, port: int | None) -> str:
    """``--port`` 给出时替换 base 的端口（主机、协议不变）；返回去掉末尾斜杠的 base。"""
    base = base.rstrip("/")
    if port is None:
        return base
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, f"{parts.hostname}:{int(port)}", parts.path, "", "")).rstrip("/")


def base_port(base: str) -> int | None:
    parts = urlsplit(base)
    return parts.port or {"http": 80, "https": 443}.get(parts.scheme)


def v9_eval_counts(catalog: dict) -> dict | None:
    """V9 目录（``eval.mode == "v9-reuse"``）逐局数「复用／新评／置空」；V8 目录返回 None。

    只数新值局（xhard1～5）；``eval_origin`` 为 ``reused``／``new`` 且两模型都有终态才算数，其余（含缺字段）算置空。"""
    head = catalog.get("eval") or {}
    if head.get("mode") != "v9-reuse":
        return None
    n = Counter()
    for task in catalog.get("tasks", []):
        for tier, cell in task.get("tiers", {}).items():
            if tier == "xhard0":
                continue
            for ep in cell.get("episodes", []):
                n["episodes"] += 1
                origin = ep.get("eval_origin")
                items = (ep.get("eval") or {}).get("new") or {}
                done = all((items.get(p) or {}).get("status") in C.FINAL for p, _ in C.POLICIES)
                n[f"eval_{origin}" if origin in ("reused", "new") and done else "eval_empty"] += 1
    return {k: n[k] for k in ("episodes", "eval_reused", "eval_new", "eval_empty")}


def v9_problems(counts: dict, catalog: dict, expect_reused: int | None = None,
                expect_new: int | None = None) -> list[str]:
    """V9 计数的判定：无置空、复用 + 新评 = 新值局总数、与目录记录的计数一致、与期望值一致（给了才比）。"""
    out = []
    if counts["eval_empty"]:
        out.append(f"评估置空 {counts['eval_empty']} 局")
    if counts["eval_reused"] + counts["eval_new"] != counts["episodes"]:
        out.append(f"复用 {counts['eval_reused']} + 新评 {counts['eval_new']} ≠ 新值局 {counts['episodes']}")
    recorded = ((catalog.get("eval") or {}).get("reuse") or {}).get("counts") or {}
    for key in ("eval_reused", "eval_new", "eval_empty"):
        if recorded.get(key) != counts[key]:
            out.append(f"{key} 页面目录数 {counts[key]} 与目录 eval.reuse.counts 记录 {recorded.get(key)} 不符")
    for key, want in (("eval_reused", expect_reused), ("eval_new", expect_new)):
        if want is not None and counts[key] != want:
            out.append(f"{key}={counts[key]} ≠ 期望 {want}")
    return out


def v9_site_line(ok: bool, cells: int, missing: int, counts: dict, port: int | None) -> str:
    return (f"V9_SITE={'PASS' if ok else 'FAIL'} cells={cells} missing={missing} eval_reused={counts['eval_reused']} "
            f"eval_new={counts['eval_new']} eval_empty={counts['eval_empty']} port={port}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--port", type=int, default=None, help="替换 --base 的端口（V9 独立站，如 8082）")
    ap.add_argument("--expect-reused", type=int, default=None, help="V9：期望复用局数（如 720）")
    ap.add_argument("--expect-new", type=int, default=None, help="V9：期望新评局数（如 80）")
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--delivery", type=Path, help="可选：再与 delivery.json 的 exec_steps 逐格核对")
    ap.add_argument("--expect-cells", type=int, default=None,
                    help="期望格数；缺省由目录推出（完整根 len(格表)+16 = 59，子表为目录新值格数 + 16）")
    ap.add_argument("--chrome", help="Chromium 可执行文件；缺省用 Playwright 自带")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright  # noqa: PLC0415（解析参数之后再导入：--help 不需要 playwright）

    args.shots.mkdir(parents=True, exist_ok=True)
    base = with_port(args.base, args.port)
    problems: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    cells = 0
    catalog: dict = {"tasks": []}
    table1 = {task: {dim: dict(v) for dim, v in dims.items()} for task, dims in C.TABLE1.items()}
    with sync_playwright() as pw:
        launch = {"headless": True, "args": ["--disable-gpu"]}
        if args.chrome:
            launch["executable_path"] = args.chrome
        browser = pw.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            catalog = page.request.get(base + "/api/catalog").json()
            sg = page.request.get(base + "/api/subgoals").json()
            if not isinstance(sg, dict) or sg.get("schema") != "v8-subgoals/1" or not sg.get("oracle"):
                raise RuntimeError("subgoals.json 缺失或为空（/api/subgoals 不是 v8-subgoals/1）")
            want = [(t["id"], tier) for t in catalog["tasks"] for tier in t["tiers"]]
            cells = len(want)
            for task, tier in want:
                if tier not in sg["oracle"].get(task, {}):
                    missing.append(f"{task}/{tier} oracle 缺格")
            if args.delivery:
                for (task, tier), st in delivery_exec(args.delivery).items():
                    gt = sg["oracle"].get(task, {}).get(tier)
                    if gt is None or [gt["mean"], gt["min"], gt["max"], gt["n"]] != [st["mean"], st["min"], st["max"], st["n"]]:
                        problems.append(f"{task}/{tier} 执行步统计与 delivery.json 不符：{gt} vs {st}")
            page.goto(base + "/#view=oracle", wait_until="domcontentloaded")
            page.wait_for_selector("#oracle-section .oracle-table")
            install_cfg_check(page, table1)
            heads = [h.split("\n")[0].strip() for h in page.locator("#oracle-section .oracle-table thead th").all_inner_texts()]
            if heads != ["任务", "对比项", *catalog["tiers"]]:
                problems.append(f"总表表头不符：{heads}")
            result = page.evaluate(CHECK_JS, {"cat": catalog, "oracle": sg["oracle"], "metrics": METRICS,
                                              "scope": "#oracle-section"})
            problems += result["bad"]
            missing += result["missing"]
            if result["checked"] != cells:
                problems.append(f"总表核对格数 {result['checked']} != {cells}")
            for metric in METRICS:
                toggle = page.locator(f'#oracle-section input[data-oracle-metric="{metric}"]')
                if not toggle.is_checked():
                    problems.append(f"默认未勾选 {metric}")
                toggle.uncheck()
                if page.locator(f'#oracle-section td[data-metric="{metric}"]:visible').count():
                    problems.append(f"取消后仍显示 {metric}")
                toggle.check()
            for width in (390, 1024, 1440):
                page.set_viewport_size({"width": width, "height": 1000})
                page.wait_for_timeout(300)
                if page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"):
                    problems.append(f"{width}px 页面横向溢出")
                page.screenshot(path=str(args.shots / f"oracle-{width}.png"))
            page.set_viewport_size({"width": 1440, "height": 1000})
            # 任务页不再显示各档对比表（用户 2026-10-01「每个页面的这个表格不要再显示了。太占位置」）：
            # 逐任务确认 #matrix 是隐藏的空容器；配置、长度与评估位只在上面的「各档总表」核对
            for task in catalog["tasks"]:
                tier = next(iter(task["tiers"]))
                page.evaluate("h => { location.hash = h; }", f"#task={task['id']}&tier={tier}&ep=1")
                page.wait_for_function("() => !document.getElementById('task-section').hidden")
                if not page.evaluate("() => { const m = document.getElementById('matrix'); return m && m.hidden && !m.children.length; }"):
                    problems.append(f"任务页 {task['id']} 的对比表未撤下")
            page.screenshot(path=str(args.shots / "task-1440.png"))
        except Exception as exc:
            problems.append(f"检查中断：{type(exc).__name__}: {exc}")
        finally:
            browser.close()
    expect = args.expect_cells
    if expect is None:
        try:
            H = C.load_hard_specs()
            new_cells = {(t["id"], tier) for t in catalog["tasks"] for tier in t["tiers"] if tier != "xhard0"}
            tables = [table for table in (H.V9_CELLS,) if new_cells == set(table)]
            expect = (len(tables[0]) if tables else len(new_cells)) + len(C.NAMES)
        except Exception as exc:
            problems.append(f"无法推出期望格数：{type(exc).__name__}: {exc}")
            expect = -1
    if cells != expect:
        problems.append(f"格数 {cells} != 期望 {expect}")
    problems += ["页面脚本错误：" + e for e in errors]
    report = {"cells": cells, "missing": missing, "problems": problems, "page_errors": errors}
    (args.shots / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for line in (missing + problems)[:60]:
        print(f"# {line}")
    ok = not problems and not missing
    print(f"V8_ORACLE_BROWSER={'PASS' if ok else 'FAIL'} cells={cells} missing={len(missing)} "
          f"mismatch={len(problems)} page_errors={len(errors)}", flush=True)
    counts = v9_eval_counts(catalog)
    if counts is None:
        return 0 if ok else 1
    v9_bad = v9_problems(counts, catalog, args.expect_reused, args.expect_new)
    for line in v9_bad:
        print(f"# V9 {line}")
    report["v9"] = {"counts": counts, "problems": v9_bad, "port": base_port(base)}
    (args.shots / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = ok and not v9_bad
    print(v9_site_line(ok, cells, len(missing), counts, base_port(base)), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
