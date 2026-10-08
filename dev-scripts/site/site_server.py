"""V6 视频浏览服务：只开放固定首页、公开目录和白名单视频，支持单段Range播放。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]  # dev-scripts/site/<本文件> → 仓根
DEFAULT_SITE_DIR = REPO_ROOT / 'artifacts/newtask-v6/site'
MEDIA_ID = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')


def byte_range(header: str | None, size: int) -> tuple[int, int, int]:
    """返回状态码、起始位置和长度；拒绝多段、越界、空后缀与畸形Range。"""
    if header is None:
        return 200, 0, size
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if match is None or size == 0 or not any(match.groups()):
        raise ValueError('无法满足字节范围')
    left, right = match.groups()
    if not left:
        suffix = int(right)
        if suffix <= 0:
            raise ValueError('后缀范围必须大于零')
        start, end = max(0, size - suffix), size - 1
    else:
        start = int(left)
        end = min(int(right), size - 1) if right else size - 1
        if start >= size or end < start:
            raise ValueError('字节范围越界')
    return 206, start, end - start + 1


class SiteFiles:
    """启动时固定媒体白名单；路径只能落在已批准的产物根内。"""
    def __init__(self, site_dir: Path, media_root: Path, html_path: Path):
        self.media_root = media_root.resolve(strict=True)
        self.site_dir = site_dir.resolve(strict=True)
        if not self.site_dir.is_relative_to(self.media_root):
            raise ValueError('站点目录必须位于产物根内')
        self.html_path = html_path.resolve(strict=True)
        self.catalog_path = self.site_dir / 'catalog.json'
        if not self.catalog_path.resolve(strict=True).is_relative_to(self.media_root):
            raise ValueError('公开目录路径超出产物根')
        private = self.site_dir / 'media-private.json'
        if not private.resolve(strict=True).is_relative_to(self.media_root):
            raise ValueError('媒体白名单路径超出产物根')
        mapping = json.loads(private.read_text(encoding='utf-8'))
        if not isinstance(mapping, dict):
            raise ValueError('媒体白名单必须是ID到绝对路径的对象')
        self.media: dict[str, Path] = {}
        for identity, filename in mapping.items():
            if not isinstance(identity, str) or MEDIA_ID.fullmatch(identity) is None:
                raise ValueError('媒体ID格式不符')
            if not isinstance(filename, str) or not Path(filename).is_absolute():
                raise ValueError('媒体路径必须是绝对路径')
            path = Path(filename).resolve(strict=True)
            if not path.is_relative_to(self.media_root) or path.suffix.lower() != '.mp4' or not path.is_file():
                raise ValueError('媒体路径不是产物根内的MP4文件')
            self.media[identity] = path.relative_to(self.media_root)

    def open_media(self, identity: str):
        """逐级openat且禁止跟随符号链接，避免路径替换绕过白名单边界。"""
        return self._open_relative(self.media[identity])

    def open_poster(self, identity: str):
        """预览图只接受已有媒体ID，固定读取站点下posters中的同名JPEG。"""
        if identity not in self.media or MEDIA_ID.fullmatch(identity) is None:
            raise KeyError(identity)
        relative = self.site_dir.relative_to(self.media_root) / 'posters' / f'{identity}.jpg'
        return self._open_relative(relative)

    def _open_relative(self, relative: Path):
        directory = os.open(self.media_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in relative.parts[:-1]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        finally:
            os.close(directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise OSError('媒体不是普通文件')
        return os.fdopen(descriptor, 'rb')


class MediaServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, files: SiteFiles):
        self.files = files
        super().__init__(address, MediaHandler)


class MediaHandler(BaseHTTPRequestHandler):
    server_version = 'V6Media'
    sys_version = ''
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        self._serve(head=False)

    def do_HEAD(self):
        self._serve(head=True)

    def _headers(self, status: int, content_type: str, length: int, **extra):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('X-Content-Type-Options', 'nosniff')
        for name, value in extra.items():
            self.send_header(name.replace('_', '-'), str(value))
        self.end_headers()

    def _message(self, status: int, message: str, head: bool):
        payload = (message + '\n').encode('utf-8')
        self._headers(status, 'text/plain; charset=utf-8', len(payload), Cache_Control='no-store')
        if not head:
            self.wfile.write(payload)

    def _serve(self, head: bool):
        try:
            # 不以任何URL部分构造文件路径；只做固定路由和白名单ID查找。
            path = unquote(urlsplit(self.path).path, errors='strict')
            files = self.server.files
            if path in ('/', '/index.html'):
                payload = files.html_path.read_bytes()
                self._headers(200, 'text/html; charset=utf-8', len(payload), Cache_Control='no-cache')
                if not head:
                    self.wfile.write(payload)
                return
            if path == '/api/gtlen':
                # 真值步数统计（历史 V6 统计脚本 v6_gt_lengths.py 生成，该脚本已不在本仓）；缺文件时返回空表，页面照常显示
                gt = files.html_path.with_name('v6_gt_lengths.json')
                payload = gt.read_bytes() if gt.exists() else b'{"cells":{},"media":{}}'
                self._headers(200, 'application/json; charset=utf-8', len(payload), Cache_Control='no-cache')
                if not head:
                    self.wfile.write(payload)
                return
            if path == '/api/subgoals':
                # 逐段 subgoal 帧数（dev-scripts/site/subgoal_lengths.py 生成）；缺文件时返回空表，页面照常显示
                sg = files.site_dir / 'subgoals.json'
                payload = sg.read_bytes() if sg.exists() else b'{}'
                self._headers(200, 'application/json; charset=utf-8', len(payload), Cache_Control='no-cache')
                if not head:
                    self.wfile.write(payload)
                return
            if path == '/api/semantic':
                # xhard1～5 相对 xhard0 的 goal／subgoal 语义调整（dev-scripts/site/semantic_diff.py 生成）；缺文件时返回空表
                sem = files.site_dir / 'semantic.json'
                payload = sem.read_bytes() if sem.exists() else b'{}'
                self._headers(200, 'application/json; charset=utf-8', len(payload), Cache_Control='no-cache')
                if not head:
                    self.wfile.write(payload)
                return
            if path == '/api/catalog':
                payload = files.catalog_path.read_bytes()
                self._headers(200, 'application/json; charset=utf-8', len(payload), Cache_Control='no-cache')
                if not head:
                    self.wfile.write(payload)
                return
            if path.startswith('/poster/'):
                identity = path[len('/poster/'):]
                if MEDIA_ID.fullmatch(identity) is None or identity not in files.media:
                    self._message(404, '未找到预览图', head)
                    return
                try:
                    stream = files.open_poster(identity)
                except (OSError, KeyError):
                    self._message(404, '未找到预览图', head)
                    return
                with stream:
                    size = os.fstat(stream.fileno()).st_size
                    self._headers(200, 'image/jpeg', size, Cache_Control='private, max-age=3600')
                    if not head:
                        self.wfile.write(stream.read())
                return
            prefix = '/media/'
            identity = path[len(prefix):] if path.startswith(prefix) else ''
            if MEDIA_ID.fullmatch(identity) is None or identity not in files.media:
                self._message(404, '未找到资源', head)
                return
            try:
                stream = files.open_media(identity)
            except (OSError, KeyError):
                self._message(404, '未找到视频', head)
                return
            with stream:
                size = os.fstat(stream.fileno()).st_size
                ranges = self.headers.get_all('Range', [])
                try:
                    if len(ranges) > 1:
                        raise ValueError('不支持多段范围')
                    code, start, length = byte_range(ranges[0] if ranges else None, size)
                except ValueError:
                    self._headers(416, 'video/mp4', 0, Content_Range=f'bytes */{size}', Accept_Ranges='bytes')
                    return
                headers = {'Accept_Ranges': 'bytes', 'Cache_Control': 'private, max-age=0'}
                if code == 206:
                    headers['Content_Range'] = f'bytes {start}-{start + length - 1}/{size}'
                self._headers(code, 'video/mp4', length, **headers)
                if head:
                    return
                stream.seek(start)
                remaining = length
                while remaining:
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        self.close_connection = True
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # 浏览器拖动进度条会主动取消旧请求，不将正常取消当成服务故障。
            return
        except (UnicodeError, ValueError):
            self._message(400, '请求格式不正确', head)
        except OSError:
            self._message(503, '资源暂时不可用', head)


def create_server(host='0.0.0.0', port=8060, *, site_dir=DEFAULT_SITE_DIR,
                  media_root=REPO_ROOT / 'artifacts', html_path=Path(__file__).with_suffix('.html')):
    files = SiteFiles(Path(site_dir), Path(media_root), Path(html_path))
    return MediaServer((host, port), files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8060)
    parser.add_argument('--site-dir', type=Path, default=DEFAULT_SITE_DIR)
    args = parser.parse_args()
    with create_server(args.host, args.port, site_dir=args.site_dir) as server:
        print(f'V6_SITE_READY host={args.host} port={server.server_port} videos={len(server.files.media)}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
