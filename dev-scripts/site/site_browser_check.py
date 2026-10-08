#!/usr/bin/env python3
"""v8 站点的浏览器交互检查（v8 方案第一部分 §3「站点与 v7 布局一致」）：Playwright + headless Chromium，连测试实例。

由 V7 站点浏览器检查器（已删除，git 历史可取回）改写。2026-10-02 起评估结果接入（用户「把所有的结果放在8081端口」）：xhard1～5 为
V8 双模型评估、xhard0 为阶段 3′ 两路线评估，评估断言由「全部未评估」改为逐局与目录一致。逐格（目录里全部 (任务, 档)，
完整根 59 格）按 hash 直达第 1 局，核对：

- **布局**：v7 页面的 15 个 DOM 区块都在（``SECTIONS``）；
- **评估数据**（目录层）：每局两策略都有终态（xhard0 新旧入口各两策略），新值局两策略都有评估视频、xhard0 MME-VLA
  两入口都有评估视频，缺一记 ``eval_missing``；
- **评估位**（页面层）：评估栏没有「未评估」占位（``eval_placeholders`` 必须为 0）；每栏徽标与目录终态一致、局号点与
  目录终态一致（``eval_mismatch``）；
- **成败筛选**：两者都成／分歧／两者都未成／翻转按钮计数与目录推算一致，「未评估」为 0；实点三个成败筛选确认可见局号
  与计数一致（不一致计入 ``eval_mismatch``）；
- **评估媒体**：每格第 1 局有评估视频的栏实际播放（``currentTime > 0.2``，计 ``eval_played``）；
- **语义调整**（``/api/semantic``，``semantic_diff.py``）：任务页语义面板已撤下（只断言隐藏），第 1 局 goal／subgoal 的
  「语义调整」标签数与逐局标记一致；「语义调整合集」页的卡片恰为 ``semantic.json`` 中有调整的格，每张都有调整前／后对照表
  （不一致计 ``semantic_mismatch``）；
- **逐段数据**：``/api/subgoals`` 必须是 ``v8-subgoals/1``（缺失或空对象即 FAIL），每局（含 xhard0 旧入口）都有逐段记录，
  第 1 局页面的逐段表行数、task goal 条数与数据一致（``subgoal_missing``）；
- **任务页对比表已撤下**：用户 2026-10-01 要求任务页不再显示各档对比总表，``#matrix`` 保留为隐藏的空容器（区块数不变）；
  同一张表只在「各档总表」页显示，由 ``oracle_browser_check.py`` 逐格核对；
- **配置**：逐局配置（``li[data-dim]``）逐维与表 1 一致（``site_catalog.TABLE1``，定值相等、区间落在内），
  维度集合与表 1 相同（``config_mismatch``）；
- **xhard5 与生成视频**：SwingXtimes／StopCube 的 xhard5 页签可用，每格第 1 局的生成视频元数据可读并实际播放
  （``currentTime > 0.2``）；「同步播放」让 xhard0 本局全部视频（两段生成＋MME-VLA 两入口评估）前进；
- 390px 宽度下无横向溢出；页面无脚本错误。

截图写到 ``--shots``。末行打印
``V8_SITE=PASS|FAIL sections=15 eval_placeholders=0 eval_missing=0 eval_mismatch=0 eval_played=<n> subgoal_missing=0
config_mismatch=0``（另附 cells、played、page_errors）。接续脚本只认 ``V8_SITE=PASS sections=15`` 前缀。任何中断（超时、页面崩溃）都记入 problems 并照打判定行（FAIL）。
实点筛选、xhard5 页签、同步播放、移动端几段的锚点按目录里实际存在的格选取（首选格缺了换同类格），
一类都没有则跳过并计入 problems，所以子表目录不含 xhard5 时会判 FAIL 并写明原因。

    uv run --no-project --with playwright python scripts/injection-dev/site/site_browser_check.py \\
      --base http://127.0.0.1:8081 --shots artifacts/newtask-v8/site-checks
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("site_catalog", HERE / "site_catalog.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

SECTIONS = ("sidebar", "task-search", "task-nav", "status-panel", "outlier-section", "oracle-section", "task-section",
            "matrix", "tier-tabs", "filters", "legend", "rerun-panel", "chips", "episode", "notes")
FINAL = ("success", "fail", "timeout", "error")
POLICY_IDS = ("simplememvla", "mmevla")  # 历史数据键：FrameSamp+Modulation 的页面 ID（site_catalog.EVAL_POLICY）


def ev_status(ep: dict, entry: str, p: str) -> str:
    return ((ep.get("eval") or {}).get(entry) or {}).get(p, {}).get("status") or "unevaluated"


def want_filters(eps: list[dict]) -> dict:
    """与页面 epMatches 同口径：两策略新入口都有终态时才参与成败筛选。"""
    out = {"both": 0, "split": 0, "none": 0, "flip": 0, "uneval": 0}
    for ep in eps:
        sts = [ev_status(ep, "new", p) for p in POLICY_IDS]
        done = all(s in FINAL for s in sts)
        a, b = (s == "success" for s in sts)
        out["both"] += done and a and b
        out["split"] += done and a != b
        out["none"] += done and not a and not b
        out["flip"] += bool(ep.get("flip")) and any(ep["flip"].values())
        out["uneval"] += any(s not in FINAL for s in sts)
    return out


def wait_played(page, selector: str, timeout_ms: int = 15000) -> bool:
    try:
        page.wait_for_function(
            """sel => { const v = document.querySelector(sel); return v && v.currentTime > 0.2; }""",
            arg=selector, timeout=timeout_ms)
        return True
    except Exception:
        return False


def check_values(task: str, tier: str, dim: str, values) -> bool:
    return bool(values) and all(C.table1_ok(task, dim, tier, v) for v in values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--chrome", help="Chromium 可执行文件；缺省用 Playwright 自带")
    args = ap.parse_args()
    args.shots.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/")
    problems: list[str] = []
    page_errors: list[str] = []
    n = {"semantic_mismatch": 0, "sections": 0, "cells": 0, "eval_placeholders": 0, "eval_missing": 0, "eval_mismatch": 0, "eval_played": 0,
         "subgoal_missing": 0, "config_mismatch": 0, "played": 0, "videos_meta": 0}
    with sync_playwright() as p:
        launch = {"headless": True, "args": ["--disable-gpu", "--autoplay-policy=no-user-gesture-required"]}
        if args.chrome:
            launch["executable_path"] = args.chrome
        browser = p.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1500, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        try:
            catalog = page.request.get(f"{base}/api/catalog").json()
            sg = page.request.get(f"{base}/api/subgoals").json()
            sem = page.request.get(f"{base}/api/semantic").json()
            if not isinstance(sem, dict) or sem.get("schema") != "v8-semantic/1":
                problems.append("/api/semantic 不是 v8-semantic/1")
                sem = {"tasks": {}, "episodes": {}}
            for t in catalog["tasks"]:
                for tier, cell in t["tiers"].items():
                    for ep in cell["episodes"]:
                        for entry in (("new", "old") if tier == "xhard0" else ("new",)):
                            for p in POLICY_IDS:
                                r = ((ep.get("eval") or {}).get(entry) or {}).get(p)
                                need_media = tier != "xhard0" or p == "mmevla"  # 历史数据键：FrameSamp+Modulation 的页面 ID（site_catalog.EVAL_POLICY）
                                if r is None or r.get("status") not in FINAL or (need_media and not r.get("media")):
                                    n["eval_missing"] += 1
            if n["eval_missing"]:
                problems.append(f"目录里缺评估终态或视频 {n['eval_missing']} 处")
            if not isinstance(sg, dict) or sg.get("schema") != "v8-subgoals/1" or not sg.get("episodes"):
                problems.append("subgoals.json 缺失或为空（/api/subgoals 不是 v8-subgoals/1）")
                sg = {"episodes": {}, "oracle": {}}
            # 数据层：每局逐段记录齐全（xhard0 两个入口）
            for t in catalog["tasks"]:
                for tier, cell in t["tiers"].items():
                    for ep in cell["episodes"]:
                        rec = sg["episodes"].get(t["id"], {}).get(tier, {}).get(str(ep["idx"]))
                        sides = ("new", "old") if tier == "xhard0" else ("new",)
                        if rec is None or any(s not in rec for s in sides) or "goal" not in rec:
                            n["subgoal_missing"] += 1
            if n["subgoal_missing"]:
                problems.append(f"逐段数据缺 {n['subgoal_missing']} 局")

            page.goto(f"{base}/", wait_until="domcontentloaded")
            page.wait_for_selector("#task-nav .task-link")
            page.wait_for_selector("#tier-tabs button")  # 等首个任务页渲染完（侧栏 oracle-link 先于任务页出现）
            present = page.evaluate("ids => ids.filter(id => document.getElementById(id))", list(SECTIONS))
            n["sections"] = len(present)
            if n["sections"] != len(SECTIONS):
                problems.append(f"缺少区块：{sorted(set(SECTIONS) - set(present))}")
            # 任务页不再显示各档对比总表（用户 2026-10-01）：#matrix 保留为空并隐藏；该表只在「各档总表」页，
            # 由 oracle_browser_check.py 逐格核对配置、长度与评估位
            if not page.evaluate("() => { const m = document.getElementById('matrix'); return m && m.hidden && !m.children.length; }"):
                problems.append("任务页 #matrix 未隐藏或仍有内容")

            for task in catalog["tasks"]:
                for tier, cell in task["tiers"].items():
                    n["cells"] += 1
                    key = f"{task['id']}/{tier}"
                    entries = 2 if tier == "xhard0" else 1
                    page.evaluate("h => { location.hash = h; }", f"#task={task['id']}&tier={tier}&ep=1")
                    try:
                        page.wait_for_function(
                            "([t, tier, n]) => document.querySelector('#episode h3')?.textContent.startsWith(t + ' · ' + tier + ' · 第 1 局')"
                            " && document.querySelectorAll('#episode .col').length === n",
                            arg=[task["id"], tier, 3 * entries], timeout=10000)
                    except Exception:
                        problems.append(f"{key} 栏数或标题不符")
                        continue
                    cnt = page.locator("#episode .eval-placeholder").count()
                    n["eval_placeholders"] += cnt
                    if cnt:
                        problems.append(f"{key} 出现「未评估」占位 {cnt} 个")
                    ep1 = cell["episodes"][0]
                    if tier != "xhard0":
                        want_c = sem["tasks"].get(task["id"], {}).get("tiers", {}).get(tier)
                        # 任务页面板已撤下（用户「不要写在这里了」），只断言隐藏；调整前后对照在合集页核对
                        got_c = page.evaluate("() => { const b = document.getElementById('sem-panel'); return b && b.hidden && !b.children.length ? 'hidden' : 'shown'; }")
                        want_e = sem["episodes"].get(task["id"], {}).get(tier, {}).get(str(ep1["idx"]), {})
                        tags = page.evaluate("() => [document.querySelectorAll('#episode .goal li.sem-changed').length,"
                                             " document.querySelectorAll('#episode .sg-table tr.sem-changed').length]")
                        if want_c is None or got_c != "hidden" \
                                or tags != [sum(want_e.get("goal", [])), sum(want_e.get("sub", []))]:
                            n["semantic_mismatch"] += 1
                            problems.append(f"{key} 语义面板 {got_c} 或标签 {tags} 与 semantic.json 不符")
                    badges = page.evaluate("() => [...document.querySelectorAll('#episode .col.is-eval')]"
                                           ".map(c => [c.dataset.policy, (c.querySelector('.badge')?.className || '').replace('badge s-', '')])")
                    want_badges = [[p, ev_status(ep1, entry, p)] for entry in (("new", "old") if tier == "xhard0" else ("new",))
                                   for p in POLICY_IDS]
                    if badges != want_badges:
                        n["eval_mismatch"] += 1
                        problems.append(f"{key} 第 1 局评估徽标 {badges} != 目录 {want_badges}")
                    chips = page.locator("#chips .chip").count()
                    if chips != len(cell["episodes"]):
                        problems.append(f"{key} 局号 {chips} != {len(cell['episodes'])}")
                    dots = page.evaluate("() => [...document.querySelectorAll('#chips .chip')].map(c => [...c.querySelectorAll('.dot')]"
                                         ".map(d => d.className.replace('dot s-', '')))")
                    if dots != [[ev_status(ep, "new", p) for p in POLICY_IDS] for ep in cell["episodes"]]:
                        n["eval_mismatch"] += 1
                        problems.append(f"{key} 局号点与目录终态不一致")
                    counts = page.evaluate("() => Object.fromEntries([...document.querySelectorAll('#filters .filter')]"
                                           ".map(b => [b.dataset.filter, Number(b.textContent.trim().split(/\\s+/).pop())]))")
                    want = want_filters(cell["episodes"])
                    for f, v in want.items():
                        if f == "flip" and tier != "xhard0":
                            continue
                        if counts.get(f) != v:
                            n["eval_mismatch"] += 1
                            problems.append(f"{key} 筛选「{f}」计数 {counts.get(f)} != {v}")
                    # 评估视频：第 1 局每个有视频的评估栏实际播放
                    for i, p_has in enumerate(page.evaluate("() => [...document.querySelectorAll('#episode .col.is-eval')]"
                                                            ".map(c => !!c.querySelector('video'))")):
                        if not p_has:
                            continue
                        page.evaluate("i => { const v = document.querySelectorAll('#episode .col.is-eval')[i].querySelector('video');"
                                      " v.play(); }", i)
                        ok_play = page.evaluate("""i => new Promise(res => { const v = document.querySelectorAll('#episode .col.is-eval')[i]
                                                  .querySelector('video'); const t0 = Date.now();
                                                  (function tick(){ if (v.currentTime > 0.2) res(true); else if (Date.now() - t0 > 15000) res(false);
                                                  else setTimeout(tick, 200); })(); })""", i)
                        if ok_play:
                            n["eval_played"] += 1
                        else:
                            problems.append(f"{key} 第 {i + 1} 个评估栏视频未播放")
                        page.evaluate("() => document.querySelectorAll('video').forEach(v => v.pause())")
                    # 逐段
                    rec = sg["episodes"].get(task["id"], {}).get(tier, {}).get("1")
                    if rec is not None:
                        rows = page.locator("#episode .sg-ep .sg-table tr").count()
                        segs = len(rec.get("new", []))
                        goals = page.locator("#episode .goal ol li").count()
                        if (segs and rows != segs + 1) or goals != max(1, len(rec.get("goal", []))):
                            n["subgoal_missing"] += 1
                            problems.append(f"{key} 第 1 局逐段表 {rows - 1}/{segs} 行或 goal {goals} 条不符")
                    # 配置（逐局；各档总表的配置格由 oracle_browser_check.py 核对）
                    if tier != "xhard0":
                        want_dims = set(C.TABLE1.get(task["id"], {}))
                        lis = page.evaluate("() => [...document.querySelectorAll('#episode .cfg-sem li[data-dim]')]"
                                            ".map(d => [d.dataset.dim, JSON.parse(d.dataset.value)])")
                        if {d for d, _ in lis} != want_dims:
                            n["config_mismatch"] += 1
                            problems.append(f"{key} 配置维度不符：{sorted(d for d, _ in lis)} vs 表 1 {sorted(want_dims)}")
                        for dim, value in lis:
                            if not check_values(task["id"], tier, dim, [value]):
                                n["config_mismatch"] += 1
                                problems.append(f"{key} 第 1 局配置 {dim}={value} 与表 1 不符")
                    # 生成视频
                    if cell["episodes"][0]["gen"].get("new", {}).get("media"):
                        try:
                            page.wait_for_function(
                                "() => [...document.querySelectorAll('#episode video')].every(v => v.readyState >= 1 && !v.error)",
                                timeout=15000)
                            n["videos_meta"] += page.locator("#episode video").count()
                        except Exception:
                            problems.append(f"{key} 视频元数据未就绪")
                        page.evaluate("() => document.querySelector('#episode .col video').play()")
                        if wait_played(page, "#episode .col video"):
                            n["played"] += 1
                        else:
                            problems.append(f"{key} 生成视频未播放")
                        page.evaluate("() => document.querySelectorAll('video').forEach(v => v.pause())")
                    if tier == "xhard5" and task["id"] == "SwingXtimes":
                        page.wait_for_timeout(300)
                        page.screenshot(path=str(args.shots / "xhard5-swingxtimes.png"), full_page=True)

            def goto(task: str, tier: str, ep: int = 1) -> None:
                page.evaluate("h => { location.hash = h; }", f"#task={task}&tier={tier}&ep={ep}")
                page.wait_for_function("p => document.querySelector('#episode h3')?.textContent.startsWith(p)",
                                       arg=f"{task} · {tier} · 第 {ep} 局", timeout=10000)

            # 锚点按目录里实际存在的格选取：先取首选格，缺了换同类的第一格；一类都没有则跳过并计入 problems
            cells_all = [(t["id"], tier, cell) for t in catalog["tasks"] for tier, cell in t["tiers"].items()]

            def anchor(prefer: tuple[str, str], ok) -> tuple[str, str, dict] | None:
                hits = [c for c in cells_all if ok(c)]
                return next((c for c in hits if c[:2] == prefer), hits[0] if hits else None)

            def section(name: str, fn) -> None:
                try:
                    fn()
                except Exception as exc:  # 单段失败只记 problems，后续段照跑，判定行照打
                    problems.append(f"{name} 中断：{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")

            def filters_section():
                a = anchor(("BinFill", "xhard1"), lambda c: c[1] != "xhard0") or anchor(("BinFill", "xhard0"), lambda c: True)
                if a is None:
                    problems.append("目录无任何格，跳过筛选实点")
                    return
                goto(a[0], a[1])
                want = want_filters(a[2]["episodes"])
                for filt in ("both", "split", "none"):
                    page.click(f'#filters .filter[data-filter="{filt}"]')
                    visible = page.locator("#chips .chip:visible").count()
                    if visible != want[filt]:
                        n["eval_mismatch"] += 1
                        problems.append(f"点「{filt}」后可见局号 {visible} != {want[filt]}")
                page.click('#filters .filter[data-filter="all"]')
                page.screenshot(path=str(args.shots / f"filters-{a[0]}-{a[1]}.png"), full_page=True)

            def xhard5_section():
                a = anchor(("StopCube", "xhard5"), lambda c: c[1] == "xhard5")
                if a is None:
                    problems.append("目录无 xhard5 格，跳过 xhard5 页签检查")
                    return
                goto(a[0], a[1])
                if page.locator('#tier-tabs .tier-tab:has-text("xhard5"):not([disabled])').count() != 1:
                    problems.append(f"{a[0]} 的 xhard5 页签不可用")
                page.screenshot(path=str(args.shots / f"xhard5-{a[0]}.png"), full_page=True)

            def sync_section():
                def both_media(c):
                    return c[1] == "xhard0" and any(all(g.get("media") for g in ep["gen"].values()) and len(ep["gen"]) == 2
                                                    for ep in c[2]["episodes"])
                a = anchor(("BinFill", "xhard0"), both_media)
                if a is None:
                    problems.append("目录无两入口都有生成视频的 xhard0 局，跳过同步播放")
                    return
                ep = next(e for e in a[2]["episodes"] if len(e["gen"]) == 2 and all(g.get("media") for g in e["gen"].values()))
                goto(a[0], a[1], ep["idx"])
                # 本局视频 = 两入口生成视频 + 有视频的评估栏（xhard0 为 MME-VLA 两入口）
                want_videos = 2 + sum(1 for entry in ("new", "old") for r in (ep["eval"].get(entry) or {}).values() if r.get("media"))
                page.wait_for_function("n => document.querySelectorAll('#episode video').length === n"
                                       " && [...document.querySelectorAll('#episode video')].every(v => v.readyState >= 1)",
                                       arg=want_videos, timeout=20000)
                page.click("#sync-play")
                try:
                    page.wait_for_function("() => [...document.querySelectorAll('#episode video')].every(v => v.currentTime > 0.2)",
                                           timeout=20000)
                except Exception:
                    problems.append("同步播放后并非全部视频前进")
                page.evaluate("() => document.querySelectorAll('video').forEach(v => v.pause())")
                page.screenshot(path=str(args.shots / f"xhard0-{a[0]}.png"), full_page=True)

            def semantic_section():
                want = sorted((t, tier) for t, info in sem["tasks"].items() for tier, c in info["tiers"].items() if c["changed"])
                page.evaluate("location.hash='#view=semantic'")
                page.wait_for_selector("#semantic-section .sem-coll", timeout=10000)
                cards = page.evaluate("() => [...document.querySelectorAll('#semantic-section .sem-coll > .sem-panel')]"
                                      ".map(c => [c.dataset.task, c.dataset.tier, c.querySelectorAll('table.sem-ba').length,"
                                      " c.querySelectorAll('th').length])")
                got = sorted((c[0], c[1]) for c in cards)
                if got != want:
                    n["semantic_mismatch"] += 1
                    problems.append(f"语义调整合集格 {len(got)} != {len(want)}")
                thin = [c[:2] for c in cards if c[2] < 2 or c[3] < 4]
                if thin:
                    n["semantic_mismatch"] += 1
                    problems.append(f"合集里缺调整前后对照表：{thin[:5]}")
                page.screenshot(path=str(args.shots / "semantic.png"), full_page=True)

            def oracle_section():
                page.evaluate("location.hash='#view=oracle'")
                try:
                    page.wait_for_selector("#oracle-section .oracle-table", timeout=10000)
                    left = page.locator("#oracle-section td[data-metric^='policy-'] .eval-uneval").count()
                    if left:
                        problems.append(f"各档总表仍有 {left} 个成功率格显示「未评估」")
                except Exception:
                    problems.append("各档总表未显示")
                page.screenshot(path=str(args.shots / "oracle.png"), full_page=False)

            def mobile_section():
                a = anchor(("VideoPlaceOrder", "xhard0"), lambda c: c[1] == "xhard0") or anchor(("BinFill", "xhard1"), lambda c: True)
                if a is None:
                    problems.append("目录无任何格，跳过移动端检查")
                    return
                mobile = context.new_page()
                mobile.set_viewport_size({"width": 390, "height": 900})
                mobile.goto(f"{base}/#task={a[0]}&tier={a[1]}&ep=1", wait_until="domcontentloaded")
                mobile.wait_for_selector("#episode .col", timeout=15000)
                overflow = mobile.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
                if overflow > 0:
                    problems.append(f"390px 横向溢出 {overflow}px")
                mobile.screenshot(path=str(args.shots / "mobile.png"), full_page=True)

            for name, fn in (("成败筛选实点", filters_section), ("xhard5 页签", xhard5_section), ("同步播放", sync_section),
                             ("各档总表", oracle_section), ("语义调整合集", semantic_section), ("移动端", mobile_section)):
                section(name, fn)
        except Exception as exc:  # 任何中断都记入 problems，判定行照打
            problems.append(f"检查中断：{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        finally:
            browser.close()

    problems += [f"页面报错：{e}" for e in page_errors]
    for problem in problems[:60]:
        print(f"# {problem}")
    ok = not problems and n["sections"] == len(SECTIONS)
    (args.shots / "browser-result.json").write_text(
        json.dumps({**n, "problems": problems}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"V8_SITE={'PASS' if ok else 'FAIL'} sections={n['sections']} eval_placeholders={n['eval_placeholders']} "
          f"eval_missing={n['eval_missing']} eval_mismatch={n['eval_mismatch']} eval_played={n['eval_played']} "
          f"subgoal_missing={n['subgoal_missing']} config_mismatch={n['config_mismatch']} "
          f"cells={n['cells']} played={n['played']} semantic_mismatch={n['semantic_mismatch']} page_errors={len(page_errors)}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
