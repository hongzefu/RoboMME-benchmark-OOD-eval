"""C17 站点 HTTP：真实 ``site_server``（经 ``site_app.create_server``）在进程内线程里服务合成站点，``http.client`` 发原始请求。

守：固定路由、媒体白名单（只认启动时登记的 ID）、单段 Range（206 字节逐位、416 拒绝多段与越界）、HEAD 不带正文、
不以 URL 拼文件路径（越界与编码绕过一律 404／400）、逐级 ``O_NOFOLLOW`` 打开（启动后把目录换成符号链接即 404）、
白名单构造期的根内／mp4／ID 格式校验（M20）。期望字节直接从磁盘文件切片得出。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from site_world import HTML, MP4, SERVER, make_site, request, serve


@pytest.fixture
def site(tmp_path):
    info = make_site(tmp_path / "root")
    with serve(info["site"], info["root"]) as server:
        info["server"] = server
        info["bytes"] = {k: Path(v).read_bytes() for k, v in info["media"].items()}
        yield info


def test_fixed_routes(site):
    srv = site["server"]
    status, headers, body = request(srv, "GET", "/")
    assert status == 200 and body == HTML.read_bytes() and headers["content-type"].startswith("text/html")
    assert request(srv, "GET", "/index.html")[2] == body
    status, headers, body = request(srv, "GET", "/api/catalog")
    assert status == 200 and json.loads(body) == {"schema": "t7", "tasks": []}
    assert headers["x-content-type-options"] == "nosniff"
    # 缺文件的两个可选接口返回空表，页面照常显示
    assert request(srv, "GET", "/api/subgoals")[:1] == (200,) and request(srv, "GET", "/api/subgoals")[2] == b"{}"
    assert request(srv, "GET", "/api/semantic")[2] == b"{}"
    (site["site"] / "subgoals.json").write_text('{"schema": "x"}')
    assert request(srv, "GET", "/api/subgoals")[2] == b'{"schema": "x"}'


def test_media_full_and_head(site):
    srv, data = site["server"], site["bytes"]["id1"]
    status, headers, body = request(srv, "GET", "/media/id1")
    assert status == 200 and body == data and headers["accept-ranges"] == "bytes"
    assert headers["content-length"] == str(len(data)) and "content-range" not in headers
    status, headers, body = request(srv, "HEAD", "/media/id1", {"Range": "bytes=10-19"})
    assert status == 206 and body == b"" and headers["content-length"] == "10"
    assert headers["content-range"] == f"bytes 10-19/{len(data)}"


@pytest.mark.parametrize("rng, start, stop", [
    ("bytes=2-5", 2, 6),           # 闭区间两端都含
    ("bytes=2000-", 2000, None),   # 开放尾
    ("bytes=-3", -3, None),        # 后缀
    ("bytes=2040-99999", 2040, None),  # 右端越界截到末尾
    ("bytes=0-0", 0, 1),
])
def test_single_range_bytes_exact(site, rng, start, stop):
    data = site["bytes"]["id0"]
    status, headers, body = request(site["server"], "GET", "/media/id0", {"Range": rng})
    want = data[start:stop]
    assert status == 206 and body == want and int(headers["content-length"]) == len(want)
    first = start % len(data)
    assert headers["content-range"] == f"bytes {first}-{first + len(want) - 1}/{len(data)}"


@pytest.mark.parametrize("rng", ["bytes=5-2", f"bytes={len(MP4)}-", "bytes=-0", "bytes=-", "items=0-1",
                                 "bytes=0-1,4-5", ["bytes=0-1", "bytes=2-3"]])
def test_unsatisfiable_ranges_are_416(site, rng):
    status, headers, body = request(site["server"], "GET", "/media/id0", {"Range": rng})
    assert status == 416 and body == b"" and headers["content-range"] == f"bytes */{len(MP4)}"


def test_byte_range_table():
    # 判定器本身：正例与负例（与 HTTP 层分开钉死）
    assert SERVER.byte_range(None, 10) == (200, 0, 10)
    assert SERVER.byte_range("bytes=3-", 10) == (206, 3, 7)
    assert SERVER.byte_range("bytes=-4", 10) == (206, 6, 4)
    assert SERVER.byte_range("bytes=-40", 10) == (206, 0, 10)
    for bad in ("bytes=10-", "bytes=4-3", "bytes=-0", "bytes=0-1,3-4", " bytes = 1-2"):
        with pytest.raises(ValueError):
            SERVER.byte_range(bad, 10)
    with pytest.raises(ValueError):
        SERVER.byte_range("bytes=0-0", 0)  # 空文件任何 Range 都不可满足


@pytest.mark.parametrize("path", [
    "/media/../site/catalog.json", "/media/..%2Fsite%2Fcatalog.json", "/media/%2e%2e/media-private.json",
    "/media/id0/", "/media/id0.mp4", "/media//id0", "/media/ID0", "/media/" + "a" * 129, "/site/catalog.json",
    "/media-private.json", "/poster/../media/v0.mp4", "/poster/id9", "/api/catalog/../../media/v0.mp4",
])
def test_paths_outside_whitelist_are_404(site, path):
    status, _, body = request(site["server"], "GET", path)
    assert status == 404
    assert MP4[:64] not in body and b"tmp" not in body  # 不回显文件内容或路径


def test_malformed_percent_encoding_is_400(site):
    assert request(site["server"], "GET", "/media/%ff%fe")[0] == 400


def test_poster_only_for_whitelisted_ids(site):
    status, headers, body = request(site["server"], "GET", "/poster/id1")
    assert status == 200 and body == b"\xff\xd8poster\x01" and headers["content-type"] == "image/jpeg"
    (site["site"] / "posters" / "id0.jpg").unlink()
    assert request(site["server"], "GET", "/poster/id0")[0] == 404


def test_symlink_swapped_in_after_start_is_not_followed(site, tmp_path):
    # 启动后把媒体所在目录换成指向别处（同名文件、不同内容）的符号链接：逐级 O_NOFOLLOW 打开 → 404
    media_dir = site["root"] / "media"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "v0.mp4").write_bytes(b"SECRET" * 10)
    os.rename(media_dir, tmp_path / "moved")
    os.symlink(elsewhere, media_dir)
    status, _, body = request(site["server"], "GET", "/media/id0")
    assert status == 404 and b"SECRET" not in body


def test_file_replaced_by_symlink_is_not_followed(site, tmp_path):
    target = tmp_path / "secret.mp4"
    target.write_bytes(b"SECRET" * 10)
    leaf = Path(site["media"]["id1"])
    leaf.unlink()
    os.symlink(target, leaf)
    status, _, body = request(site["server"], "GET", "/media/id1")
    assert status == 404 and b"SECRET" not in body


# ── 白名单构造期校验 ─────────────────────────────────────────────────


def _whitelist(site_dir: Path, mapping: dict):
    (site_dir / "media-private.json").write_text(json.dumps(mapping), encoding="utf-8")


@pytest.mark.parametrize("case, exc, needle", [
    ("outside", ValueError, "不是产物根内的MP4文件"),
    ("symlink_out", ValueError, "不是产物根内的MP4文件"),
    ("not_mp4", ValueError, "不是产物根内的MP4文件"),
    ("relative", ValueError, "必须是绝对路径"),
    ("bad_id", ValueError, "媒体ID格式不符"),
    ("missing", FileNotFoundError, "nope.mp4"),
])
def test_whitelist_construction_rejects(tmp_path, case, exc, needle):
    info = make_site(tmp_path / "root")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(MP4)
    mapping = dict(info["media"])
    if case == "outside":
        mapping["x"] = str(outside)
    elif case == "symlink_out":
        link = info["root"] / "media" / "link.mp4"
        os.symlink(outside, link)  # 根内的符号链接指向根外：按解析后的真实路径判，拒绝
        mapping["x"] = str(link)
    elif case == "not_mp4":
        txt = info["root"] / "media" / "a.txt"
        txt.write_text("x")
        mapping["x"] = str(txt)
    elif case == "relative":
        mapping["x"] = "media/v0.mp4"
    elif case == "bad_id":
        mapping["../x"] = mapping["id0"]
    elif case == "missing":
        mapping["x"] = str(info["root"] / "media" / "nope.mp4")
    _whitelist(info["site"], mapping)
    with pytest.raises(exc, match=needle):
        SERVER.create_server("127.0.0.1", 0, site_dir=info["site"], media_root=info["root"], html_path=HTML)


def test_site_dir_must_be_inside_media_root(tmp_path):
    info = make_site(tmp_path / "root")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="站点目录"):
        SERVER.create_server("127.0.0.1", 0, site_dir=info["site"], media_root=other, html_path=HTML)


def test_whitelist_positive_counterpart(tmp_path):
    info = make_site(tmp_path / "root", n_media=3)
    server = SERVER.create_server("127.0.0.1", 0, site_dir=info["site"], media_root=info["root"], html_path=HTML)
    try:
        assert set(server.files.media) == {"id0", "id1", "id2"}
        assert all(not p.is_absolute() for p in server.files.media.values())
    finally:
        server.server_close()
