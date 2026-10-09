"""闸门二：公开面不暴露内部路径、不回指 dev 代码。

逐行扫清单（或 ``--paths``）里的文本文件，命中下列任一模式即记一次：
本机盘 ``/data/hongzefu``、NFS ``/nfs/turbo``、集群账户 ``chaijy``、集群名 ``greatlakes``、主机名前缀 ``sled-``、
``dev-scripts`` / ``dev_scripts``、个人邮箱前缀 ``hongzefu@``。大小写不敏感。
末行 ``PUBLIC_PATHS=PASS|FAIL files=<n> hits=<m>``；FAIL 时退出码 1。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _manifest as M  # noqa: E402

PATTERNS = [r"/data/hongzefu", r"/nfs/turbo", r"chaijy", r"greatlakes", r"sled-", r"dev[-_]scripts", r"hongzefu@"]
RX = re.compile("|".join(PATTERNS), re.IGNORECASE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="*")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--root", type=Path, default=M.REPO)
    ap.add_argument("--show", type=int, default=50)
    a = ap.parse_args(argv)
    paths = M.resolve_paths(a.paths, a.manifest)
    hits = 0
    for rel in paths:
        data = (a.root / rel).read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for no, line in enumerate(text.splitlines(), 1):
            m = RX.search(line)
            if m:
                hits += 1
                if hits <= a.show:
                    print(f"{rel}:{no}: [{m.group(0)}] {line.strip()[:160]}")
    ok = hits == 0
    print(f"PUBLIC_PATHS={'PASS' if ok else 'FAIL'} files={len(paths)} hits={hits}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
