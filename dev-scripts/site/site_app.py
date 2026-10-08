#!/usr/bin/env python3
"""v8 逐局站点服务（v8 方案第一部分 §2.2 站点行）：与 v7 站点布局逐一一致，评估板块原位保留、内容置空。

服务端完全复用同目录 ``site_server.py`` 的 ``create_server``（媒体白名单、单段 Range、``/api/catalog``、
``/api/subgoals``），显式传页面 ``site.html``；目录由 ``site_catalog.py`` 生成，逐段数据由
``subgoal_lengths.py`` 生成。``--media-root`` 是白名单根（站点目录与全部 mp4 必须在它之下），缺省为仓库
``artifacts/``；合成目录检查时传合成根。

    uv run --no-sync python dev-scripts/site/site_app.py --port 8080 --site-dir artifacts/newtask-v8/site
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("site_server", HERE / "site_server.py")
site_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site_server)

DEFAULT_SITE_DIR = site_server.REPO_ROOT / "artifacts/newtask-v8/site"
DEFAULT_MEDIA_ROOT = site_server.REPO_ROOT / "artifacts"
HTML_PATH = HERE / "site.html"


def create_server(host: str, port: int, site_dir: Path, media_root: Path = DEFAULT_MEDIA_ROOT):
    return site_server.create_server(host, port, site_dir=site_dir, media_root=media_root, html_path=HTML_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    args = parser.parse_args()
    with create_server(args.host, args.port, args.site_dir, args.media_root) as server:
        print(f"V8_SITE_READY host={args.host} port={server.server_port} videos={len(server.files.media)}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
