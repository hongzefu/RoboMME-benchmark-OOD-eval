"""C17 页面与服务端的接口面：``site.html`` 引用的路由与字段 ⊆ 服务端实际提供的；微型媒体转码（``eval_transcode.py``）。

页面里的路由与字段名用正则从 ``site.html`` 的脚本里抽出（只作清单），断言对象是真实服务对真实产物的响应：
先用真实 ``site_catalog``／``subgoal_lengths``／``semantic_diff`` 在合成小子表上建站，再起真实服务逐个请求。
真实浏览器交互（筛选、导航、错误态、播放）按计划 Q14 不做，契约清单登记「未验证」。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from site_world import CAT, H, HTML, catalog_inputs, request, serve
from tests._support.loaders import load_script

SG = load_script("injection-dev/site/subgoal_lengths.py")
SEM = load_script("injection-dev/site/semantic_diff.py")
SCRIPT = HTML.read_text(encoding="utf-8")

#: 页面只在存在性判断之后才读的字段（读法见 site.html：``ep.rerun &&``／``if (ep.rerun)``、``ep.flip &&``、
#: ``ep.eval_source ||``、``ep.config || {}``、``ep.round == null``、``!S.cat.rerun11`` 隐藏面板）；目录可以不给。
GUARDED = {"cat": {"rerun11"}, "ep": {"rerun", "flip", "eval_source", "config", "round"}}


def _fields(prefix: str) -> set[str]:
    return set(re.findall(rf"\b{re.escape(prefix)}\.([A-Za-z_][A-Za-z0-9_]*)", SCRIPT))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("assets")
    cells = {("StopCube", "xhard1"): H.V9_CELLS[("StopCube", "xhard1")]}
    src = catalog_inputs(root / "in", cells)
    site = root / "in" / "site"
    argv = ["--out", str(site), "--cells-json", str(src["cells_json"])]
    for key in ("specs_root", "delivery", "identities", "xhard0_gen", "path_base"):
        argv += [f"--{key.replace('_', '-')}", str(src[key])]
    assert CAT.main(argv) == 0
    assert SG.main(["--site-dir", str(site), "--specs-root", str(src["specs_root"]), "--delivery", str(src["delivery"]),
                    "--xhard0-gen", str(src["xhard0_gen"]), "--path-base", str(src["path_base"]), "--workers", "1"]) == 0
    assert SEM.main(["--site-dir", str(site)]) == 0
    return {"site": site, "root": root / "in"}


@pytest.fixture(scope="module")
def server(built):
    with serve(built["site"], built["root"]) as srv:
        yield srv


def test_every_fetched_api_route_is_served(server, built):
    routes = set(re.findall(r"fetch\('(/[^']*)'", SCRIPT))
    assert routes == {"/api/catalog", "/api/subgoals", "/api/semantic"}  # 清单本身（页面改了就要同步更新测试）
    for route in routes:
        status, headers, body = request(server, "GET", route)
        assert status == 200 and headers["content-type"].startswith("application/json"), route
        assert json.loads(body), route  # 非空文档


def test_media_route_template_resolves_catalog_media(server, built):
    templates = set(re.findall(r"`(/media/)\$\{[a-z]+\}", SCRIPT))
    assert templates == {"/media/"}
    cat = json.loads(request(server, "GET", "/api/catalog")[2])
    ids = {ep["gen"][e]["media"] for t in cat["tasks"] for c in t["tiers"].values() for ep in c["episodes"]
           for e in ep["gen"] if ep["gen"][e].get("media")}
    assert ids
    for media_id in sorted(ids)[:5]:
        status, headers, _ = request(server, "GET", f"/media/{media_id}", {"Range": "bytes=0-9"})
        assert status == 206 and headers["content-type"] == "video/mp4"


def test_catalog_fields_read_by_page_are_served(server):
    cat = json.loads(request(server, "GET", "/api/catalog")[2])
    need = _fields("S.cat") - GUARDED["cat"]
    assert need and need <= set(cat), need - set(cat)
    eps = [ep for t in cat["tasks"] for c in t["tiers"].values() for ep in c["episodes"]]
    have = set().union(*(set(ep) for ep in eps))
    need_ep = _fields("ep") - GUARDED["ep"]
    assert need_ep <= have, need_ep - have
    # 每局都必须有的字段（页面无保护地直接读）
    for key in ("seed", "idx", "gen", "eval"):
        assert key in need_ep and all(key in ep for ep in eps), key


def test_subgoal_and_semantic_fields_read_by_page_are_served(server):
    sg = json.loads(request(server, "GET", "/api/subgoals")[2])
    sem = json.loads(request(server, "GET", "/api/semantic")[2])
    assert _fields("S.sg") <= set(sg), _fields("S.sg") - set(sg)
    assert _fields("S.sem") <= set(sem), _fields("S.sem") - set(sem)


def test_guarded_list_is_not_a_blanket_exemption():
    # 豁免清单里的每个字段都确实出现在页面里，且页面对它有存在性判断（防止把必需字段塞进豁免）
    for name in GUARDED["ep"]:
        assert re.search(rf"ep\.{name}\s*(&&|\|\||==\s*null|\?)|if \(ep\.{name}\)|!!ep\.{name}", SCRIPT), name
    assert "!S.cat.rerun11" in SCRIPT


# ── 微型媒体转码（真实 ffmpeg，慢）─────────────────────────────────────


def _need_ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("未验证：缺 ffmpeg")


def _mkv(path: Path, frames: list[np.ndarray]) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x256",
                           "-r", "30", "-i", "-", "-c:v", "ffv1", str(path)],
                          input=b"".join(f.tobytes() for f in frames), capture_output=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.slow
def test_transcode_expands_duplicate_frames(tmp_path, monkeypatch):
    _need_ffmpeg()
    T = load_script("injection-dev/site/eval_transcode.py")
    monkeypatch.setattr(T, "REPO_ROOT", tmp_path)  # render_one 要求输出在仓库根之下；指到临时根
    rec = tmp_path / "rec"
    rec.mkdir()
    red, blue = np.zeros((256, 256, 3), np.uint8), np.zeros((256, 256, 3), np.uint8)
    red[..., 0], blue[..., 2] = 200, 200
    _mkv(rec / "front.mkv", [red, blue])     # 编码 2 帧
    _mkv(rec / "wrist.mkv", [blue])          # 编码 1 帧
    tags = ["reset", "reset", "step0", "step1"]
    front_enc, wrist_enc = [0, 0, 1, 1], [0, 0, 0, 0]  # 4 个原始帧：去重后各只编码了 1～2 帧
    (rec / "frames-front.jsonl").write_text("".join(json.dumps({"idx": i, "enc": e, "tag": t}) + "\n"
                                                    for i, (e, t) in enumerate(zip(front_enc, tags))))
    (rec / "frames-wrist.jsonl").write_text("".join(json.dumps({"idx": i, "enc": e, "tag": t}) + "\n"
                                                    for i, (e, t) in enumerate(zip(wrist_enc, tags))))
    row = {"path": str(rec), "policy": "perceptual-framesamp-modul", "tier": "xhard1", "task": "StopCube", "key": "k", "status": "success"}
    res = T.render_one((row, str(tmp_path / "out")))
    assert "error" not in res, res
    assert (res["frames"], res["mp4_frames"], res["demo_frames"]) == (4, 4, 2)
    mp4 = tmp_path / "out" / "perceptual-framesamp-modul" / "xhard1" / "StopCube" / "k.mp4"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 256, 512, 3).astype(int)
    # 左半前视：前两帧红、后两帧蓝；右半腕部恒蓝（H.264 有损，看主色通道）
    assert [int(np.argmax(f[128, 128])) for f in frames] == [0, 0, 2, 2]
    assert all(int(np.argmax(f[128, 384])) == 2 for f in frames)
    # 越界的 enc 记逐局错误，不抛出
    (rec / "frames-front.jsonl").write_text(json.dumps({"idx": 0, "enc": 5, "tag": "reset"}) + "\n")
    (rec / "frames-wrist.jsonl").write_text(json.dumps({"idx": 0, "enc": 0, "tag": "reset"}) + "\n")
    bad = T.render_one((dict(row, key="bad"), str(tmp_path / "out")))
    assert bad["mp4_frames"] == -1 and "IndexError" in bad["error"]
