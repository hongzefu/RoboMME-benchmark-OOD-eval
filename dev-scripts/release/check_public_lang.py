"""闸门一：公开面零中文（口径⑥）。

扫清单（或 ``--paths``）里每个文本文件，逐行找 CJK 统一表意文字（含扩展 A、兼容区）、CJK 符号与标点、
康熙部首与 CJK 部首补充、全角与半角形式。二进制文件（含 NUL 字节或 UTF-8 解码失败）跳过并计数。
末行 ``PUBLIC_LANG=PASS|FAIL files=<n> cjk_hits=<m> binary_skipped=<k>``；FAIL 时退出码 1。

用法：``uv run --no-project python dev-scripts/release/check_public_lang.py [--paths P ...] [--manifest M] [--root DIR]``
``--root`` 指定读文件的根目录（缺省仓库根；同步脚本拿它扫 main 的临时检出）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _manifest as M  # noqa: E402

CJK = re.compile("[⺀-⿟　-〿぀-ヿ㐀-䶿一-鿿豈-﫿︰-﹏＀-￯]")


def scan(root: Path, paths: list[str], show: int = 50) -> tuple[int, int, int]:
    hits = binary = 0
    for rel in paths:
        data = (root / rel).read_bytes()
        if b"\0" in data:
            binary += 1
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            binary += 1
            continue
        for no, line in enumerate(text.splitlines(), 1):
            if CJK.search(line):
                hits += 1
                if hits <= show:
                    print(f"{rel}:{no}: {line.strip()[:160]}")
    return len(paths), hits, binary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="*")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--root", type=Path, default=M.REPO)
    ap.add_argument("--show", type=int, default=50, help="最多打印多少条命中行")
    a = ap.parse_args(argv)
    paths = M.resolve_paths(a.paths, a.manifest)
    n, hits, binary = scan(a.root, paths, a.show)
    ok = hits == 0
    print(f"PUBLIC_LANG={'PASS' if ok else 'FAIL'} files={n} cjk_hits={hits} binary_skipped={binary}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
