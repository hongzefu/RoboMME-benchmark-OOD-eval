#!/usr/bin/env python3
"""对官方版式浏览站点执行真实 Chromium 播放、筛选、跳转与响应式检查。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote, urlsplit


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_catalog(catalog: dict) -> dict:
    """浏览器取回公开目录后独立核对身份全集、终态与媒体 ID。"""
    require(catalog.get("schema") == "official-overlay-site/1", "目录版本不符")
    episodes = catalog["episodes"]
    counts = Counter(ep["task"] for ep in episodes)
    keys = [(ep["task"], ep["tier"], ep["seed"]) for ep in episodes]
    ids = [ep["id"] for ep in episodes]
    require(len(episodes) == len(set(keys)) == len(set(ids)) == 800, "目录不是 800 个唯一身份")
    require(len(counts) == 16 and set(counts.values()) == {50}, "目录不是 16 任务 × 每任务 50 局")
    require(set(counts) == {task["id"] for task in catalog["tasks"]}, "任务索引与身份不同")
    require(all(task["episodes"] == counts[task["id"]] for task in catalog["tasks"]), "任务索引数量不符")
    require(all(task["name"] == task["id"] for task in catalog["tasks"]) and
            all(ep["task_name"] == ep["task"] for ep in episodes), "Task 名称不是英文标识符")
    require(catalog.get("counts") == {"episodes": 800, "tasks": 16, "per_task": 50, "media": 1600}, "目录计数不符")
    media = [ep[key] for ep in episodes for key in ("official_media", "original_media")]
    require(len(set(media)) == 1600, "媒体 ID 重复")
    for ep in episodes:
        require(ep["status"] in ("success", "fail", "timeout"), "出现未知终态")
        for key in ("official_media", "original_media"):
            require(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", ep[key]) is not None, "媒体 ID 不符服务器契约")
    require(not re.search(r"/(?:data|nfs|home|tmp)/", json.dumps(catalog, ensure_ascii=False)), "公开目录暴露本机路径")
    return {"episodes": 800, "tasks": 16, "per_task": dict(counts), "statuses": dict(Counter(ep["status"] for ep in episodes))}


def authority_rates(path: Path, *, baseline: bool) -> tuple[dict, dict]:
    """独立从权威结果重算成功率，不导入目录构建器或复用其计算结果。"""
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    require(len(rows) == (192 if baseline else 800), "成功率来源局数不符")
    seen, counts, successes = set(), Counter(), Counter()
    for row in rows:
        key = row["key"]
        require(key not in seen and key == f"{row['task']}_{row['tier']}_{row['seed']}", "成功率来源身份重复或不符")
        seen.add(key)
        require(row["policy"] == "groundsg" and row["policy_variant"] == "ground-sg-oracle" and
                row["side"] == "new" and row["host"] == "sled-vail" and row["infra"] is False,
                "成功率来源不是本机 Oracle 新侧完成结果")
        require(row["dataset"] == ("hard-verify" if baseline else "ood") and
                (row["tier"] == "xhard0" if baseline else row["tier"] in ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5")),
                "成功率来源数据集或难度不符")
        require(row["status"] in ("success", "fail", "timeout") and type(row["task_success"]) is bool and
                row["task_success"] == (row["status"] == "success"), "成功率来源成功字段冲突")
        counts[row["task"]] += 1
        successes[row["task"]] += row["task_success"]
    require(len(counts) == 16 and set(counts.values()) == {12 if baseline else 50}, "成功率来源 Task 覆盖不符")
    return ({task: {"success": successes[task], "total": counts[task]} for task in counts},
            {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload), "episodes": len(rows)})


def check_rates(page, catalog: dict, v9: dict, xhard0: dict) -> dict:
    require(set(v9) == set(xhard0), "新旧成功率 Task 集合不同")
    rows = catalog["success_rates"]
    require(len(rows) == 16 and {row["task"] for row in rows} == set(v9), "目录成功率表 Task 不符")
    require(page.locator("#success-rates tbody tr").count() == 16, "成功率表不是仅 16 行")
    require(page.locator("#success-rates thead th").all_inner_texts() == ["Task", "Xhard", "原版hard"],
            "成功率表不止 Task/Xhard/原版hard 三列")
    for row in rows:
        task = row["task"]
        tr = page.locator(f'#success-rates tbody tr[data-task="{task}"]')
        require(tr.locator("td").first.inner_text() == task, "成功率表 Task 不是英文标识符")
        for column, expected in (("v9", v9[task]), ("xhard0", xhard0[task])):
            require(row[column] == expected, "目录成功率与独立权威重算不同")
            td = tr.locator(f'td[data-rate="{column}"]')
            require(td.get_attribute("data-success") == str(expected["success"]) and
                    td.get_attribute("data-total") == str(expected["total"]), "页面成功率分子分母不同")
            require(td.evaluate("node=>node.childNodes[0].textContent") == f"{100 * expected['success'] / expected['total']:.1f}%",
                    "页面百分比或一位小数不符")
            require(td.locator(".fraction").inner_text() == f"{expected['success']} / {expected['total']}", "页面成功分数不符")
    overall = page.locator("#success-rates tfoot tr")
    require(overall.count() == 1 and overall.locator("th").inner_text() == "总体", "成功率表缺少唯一总体行")
    totals = {}
    for column, authority in (("v9", v9), ("xhard0", xhard0)):
        success = sum(row["success"] for row in authority.values())
        total = sum(row["total"] for row in authority.values())
        td = overall.locator(f'td[data-rate="{column}"]')
        require(td.get_attribute("data-success") == str(success) and td.get_attribute("data-total") == str(total),
                "总体成功率分子分母与权威结果不符")
        percent = f"{100 * success / total:.1f}%"
        require(td.evaluate("node=>node.childNodes[0].textContent") == percent, "总体成功率百分比不符")
        require(td.locator(".fraction").inner_text() == f"{success} / {total}", "总体成功率分数不符")
        totals[column] = {"success": success, "total": total, "percent": percent}
    return {"rows": 16, "overall_rows": 1, "columns": 3, "overall": totals,
            "v9_success": totals["v9"]["success"], "xhard0_success": totals["xhard0"]["success"]}


def check_range(request, base: str, media_id: str) -> dict:
    response = request.get(base + "/media/" + quote(media_id), headers={"Range": "bytes=0-1023"})
    require(response.status == 206, f"媒体 Range 状态码 {response.status}")
    require(re.fullmatch(r"bytes 0-1023/\d+", response.headers.get("content-range", "")) is not None,
            "媒体缺少正确 Content-Range")
    require(response.headers.get("accept-ranges") == "bytes" and len(response.body()) == 1024,
            "媒体范围长度或 Accept-Ranges 不符")
    return {"status": response.status, "content_range": response.headers["content-range"], "bytes": len(response.body())}


def wait_player(page, episode: dict, version: str = "official") -> None:
    page.wait_for_function(
        "([id, version]) => {const p=document.getElementById('player');"
        "return p.dataset.episode===id && p.dataset.version===version && p.readyState>=2 && p.duration>0 && !p.error;}",
        arg=[episode["id"], version], timeout=45_000,
    )
    require(page.locator("#current-identity").get_attribute("data-episode") == episode["id"], "播放器身份不同")


def play_pause_seek(page, episode: dict, *, version: str = "official") -> dict:
    """真实播放推进时间，再暂停并 seek；不将 metadata 加载当成播放成功。"""
    wait_player(page, episode, version)
    page.locator("#player").evaluate("async video => {video.currentTime=0;await video.play();}")
    page.wait_for_function("document.getElementById('player').currentTime > .5", timeout=20_000)
    played = page.locator("#player").evaluate("video => ({time:video.currentTime,paused:video.paused,duration:video.duration})")
    require(not played["paused"] and played["time"] > .5, "视频没有真实播放")
    expected_frames = episode["frames"] + (episode["omitted_timeout_frames"] if version == "original" else 0)
    require(abs(played["duration"] - expected_frames / 30) < .1, "浏览器视频时长与目录帧数不符")
    page.locator("#player").evaluate("video => video.pause()")
    paused_time = page.locator("#player").evaluate("video => video.currentTime")
    page.wait_for_timeout(200)
    require(page.locator("#player").evaluate("video => video.paused && Math.abs(video.currentTime - " + str(paused_time) + ") < .05"),
            "视频暂停未生效")
    target = max(.1, min(played["duration"] * .55, played["duration"] - .1))
    page.locator("#player").evaluate("(video,target) => {video.currentTime=target;}", target)
    page.wait_for_function(
        "target => {const p=document.getElementById('player');return !p.seeking && Math.abs(p.currentTime-target)<.15 && p.readyState>=2;}",
        arg=target, timeout=30_000,
    )
    result = page.locator("#player").evaluate("video => ({time:video.currentTime,duration:video.duration,paused:video.paused,error:video.error?.code||0})")
    require(result["time"] > 0 and result["paused"] and result["error"] == 0, "视频 seek 失败")
    return {"episode": episode["id"], "version": version, "played_time": played["time"], "seek_time": result["time"],
            "duration": result["duration"], "pause": True}


def chromium_path(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    cache = Path.home() / ".cache/ms-playwright"
    candidates = list(cache.glob("chromium-*/chrome-linux64/chrome"))
    if candidates:
        return str(max(candidates, key=lambda path: int(path.parents[1].name.split("-")[-1])))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--chrome", help="覆盖已安装的 Chromium 可执行文件")
    parser.add_argument("--xhard0-results", type=Path, required=True)
    parser.add_argument("--v9-results", type=Path, required=True)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    base = args.base.rstrip("/")
    args.shots.mkdir(parents=True, exist_ok=True)
    errors, network, failures, checks = [], [], [], {}
    with sync_playwright() as pw:
        browser = None
        try:
            launch = {"headless": True, "args": ["--disable-gpu"]}
            if executable := chromium_path(args.chrome):
                launch["executable_path"] = executable
            browser = pw.chromium.launch(**launch)
            page = browser.new_page(viewport={"width": 1440, "height": 1120})
            page.on("pageerror", lambda error: errors.append({"kind": "pageerror", "message": str(error)}))
            page.on("console", lambda message: errors.append({"kind": "console", "message": message.text}) if message.type == "error" else None)
            page.on("response", lambda response: network.append({"status": response.status, "url": response.url})
                    if response.status >= 400 and urlsplit(response.url).netloc == urlsplit(base).netloc else None)
            response = page.request.get(base + "/api/catalog")
            require(response.status == 200, "目录请求失败")
            catalog = response.json()
            checks["catalog"] = check_catalog(catalog)
            v9_rates, v9_proof = authority_rates(args.v9_results, baseline=False)
            xhard0_rates, xhard0_proof = authority_rates(args.xhard0_results, baseline=True)
            checks["rate_sources"] = {"v9": v9_proof, "xhard0": xhard0_proof}
            non_demo = next(ep for ep in catalog["episodes"] if ep["demo_frames"] == 0)
            demo = next(ep for ep in catalog["episodes"] if ep["demo_frames"] > 0 and ep["task"] == "VideoPlaceButton")
            checks["range"] = {
                "official": check_range(page.request, base, non_demo["official_media"]),
                "original": check_range(page.request, base, non_demo["original_media"]),
            }
            page.goto(base + "/#episode=" + quote(non_demo["id"]), wait_until="domcontentloaded")
            page.wait_for_function("window.__officialSiteReady === true", timeout=20_000)
            checks["success_rates"] = check_rates(page, catalog, v9_rates, xhard0_rates)
            require("不代表 Great Lakes 正式评估成绩" in page.locator("#scope-notice").inner_text(), "缺少本机复刻说明")
            require(page.locator("video").count() == 1 and page.locator("#episodes button").count() == 10, "视频未分页或加载多个播放器")
            require(page.locator("#filter-task").count() == 0, "仍保留全部 Task 混合目录筛选")
            task_buttons = page.locator("#task-tabs button")
            require(task_buttons.count() == 16, "Task 栏目数量不是 16")
            require(task_buttons.locator("span").all_inner_texts() == [row["id"] for row in catalog["tasks"]],
                    "Task 栏目名称不是英文 Task 标识符")
            for task in catalog["tasks"]:
                page.locator(f'#task-tabs button[data-task="{task["id"]}"]').click()
                require(page.locator("#workspace").get_attribute("data-task") == task["id"], "Task 栏目未切换")
                require(page.locator("#filtered-count").inner_text() == "50", "Task 栏目不是自己的 50 局")
                visible = page.locator("#episodes button").evaluate_all("nodes=>nodes.map(node=>node.dataset.episode)")
                expected_task_ids = [ep["id"] for ep in catalog["episodes"] if ep["task"] == task["id"]]
                require(visible == expected_task_ids[:10], "Task 栏目混入其他 Task 视频")
                require(page.locator("#current-task").inner_text() == task["id"], "播放器 Task 不是英文名称")
            page.locator(f'#task-tabs button[data-task="{non_demo["task"]}"]').click()
            checks["task_sections"] = {"tasks": 16, "per_task": 50, "independent_lists": True}
            checks["non_demo_playback"] = play_pause_seek(page, non_demo)
            page.screenshot(path=str(args.shots / "wide-official.png"), full_page=True)

            require(page.locator("#filtered-count").inner_text() == "50", "按任务筛选数量不符")
            tier = non_demo["tier"]
            page.locator("#filter-tier").select_option(tier)
            page.locator("#filter-status").select_option(non_demo["status"])
            expected = [ep for ep in catalog["episodes"] if ep["task"] == non_demo["task"] and ep["tier"] == tier and ep["status"] == non_demo["status"]]
            require(int(page.locator("#filtered-count").inner_text()) == len(expected), "联合筛选数量不符")
            shown = page.locator("#episodes button").evaluate_all("nodes=>nodes.map(node=>node.dataset.episode)")
            require(shown == [ep["id"] for ep in expected[:10]], "联合筛选列表身份不符")
            checks["filters"] = {"task_count": 50, "combined_count": len(expected), "visible": len(shown)}
            check_rates(page, catalog, v9_rates, xhard0_rates)

            for key in ("tier", "status"):
                page.locator("#filter-" + key).select_option("")
            task_rows = [ep for ep in catalog["episodes"] if ep["task"] == non_demo["task"]]
            page.locator("#episodes button").first.click()
            wait_player(page, task_rows[0])
            page.locator("#episode-next").click()
            wait_player(page, task_rows[1])
            page.locator("#episode-prev").click()
            wait_player(page, task_rows[0])
            page.locator("#page-next").click()
            shown = page.locator("#episodes button").evaluate_all("nodes=>nodes.map(node=>node.dataset.episode)")
            require(shown == [ep["id"] for ep in task_rows[10:20]], "下一页身份不符")
            page.locator("#episodes button").first.click()
            wait_player(page, task_rows[10])
            checks["navigation"] = {"next_previous": True, "pagination": True, "selected": task_rows[10]["id"]}

            page.locator("#view-original").click()
            checks["original_playback"] = play_pause_seek(page, task_rows[10], version="original")
            require(page.locator("#view-original").get_attribute("aria-pressed") == "true", "原视频按钮状态不符")
            page.locator("#view-official").click()
            wait_player(page, task_rows[10])
            checks["version_switch"] = True

            page.locator('#task-tabs button[data-task="VideoPlaceButton"]').click()
            require(page.locator("#filtered-count").inner_text() == "50", "演示 Task 栏目局数不同")
            checks["demo_playback"] = play_pause_seek(page, demo)
            page.locator("#player").evaluate("video => {video.currentTime=.2;}")
            page.wait_for_function("() => {const p=document.getElementById('player');return !p.seeking&&p.readyState>=2;}")
            page.screenshot(path=str(args.shots / "demo-official.png"), full_page=True)
            widths = []
            for width in (390, 768):
                page.set_viewport_size({"width": width, "height": 980})
                page.wait_for_timeout(100)
                require(not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"), f"{width}px 页面横向溢出")
                require(page.locator("#player").is_visible(), "窄屏播放器不可见")
                page.screenshot(path=str(args.shots / f"narrow-{width}.png"), full_page=True)
                widths.append(width)
            checks["responsive"] = {"widths": widths, "horizontal_overflow": False}
            checks["rates_after_filters"] = check_rates(page, catalog, v9_rates, xhard0_rates)
            require(not page.locator("#player-error").is_visible() and page.locator("#player").evaluate("video=>!video.error"), "播放器媒体错误")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}：{exc}")
        finally:
            if browser:
                browser.close()
    ok = not failures and not errors and not network
    report = {"schema": "official-overlay-browser/1", "status": "PASS" if ok else "FAIL", "base": base,
              "checks": checks, "failures": failures, "page_errors": errors, "http_errors": network,
              "screenshots": sorted(path.name for path in args.shots.glob("*.png"))}
    (args.shots / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for message in failures + [f"页面错误：{error}" for error in errors] + [f"请求错误：{error}" for error in network]:
        print(message, flush=True)
    print(f"OFFICIAL_BROWSER={'PASS' if ok else 'FAIL'} episodes=800 checks={len(checks)} page_errors={len(errors)} http_errors={len(network)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
