"""公开清单的公共读取与文件枚举（只在 dev）。

``public-manifest.txt`` 一行一个跟踪文件路径（相对仓库根），不许通配；``#`` 开头的行与空行忽略。
三道闸门脚本（``check_public_lang.py``、``check_public_paths.py``、``check_manifest.py``）与 ``sync_to_main.sh``
都从这里取清单，保证「main 的全部内容」只有这一处定义。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "public-manifest.txt"


def read_manifest(path: Path | None = None) -> list[str]:
    """读清单，返回去重保序的路径列表；遇到通配符或绝对路径直接报错。"""
    p = Path(path) if path else MANIFEST
    out: list[str] = []
    seen: set[str] = set()
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(c in line for c in "*?[]") or line.startswith("/") or ".." in Path(line).parts:
            raise SystemExit(f"{p}:{i}: 清单行不许通配、绝对路径或 ..：{line!r}")
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def tracked_files(root: Path = REPO, ref: str | None = None) -> list[str]:
    """``ref`` 为空时列工作树索引里的普通文件（排除 gitlink）；否则列该提交树里的普通文件。"""
    if ref:
        cmd = ["git", "-C", str(root), "ls-tree", "-r", "--full-tree", ref]
        rows = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()
        return [r.split("\t", 1)[1] for r in rows if not r.startswith("160000")]
    rows = subprocess.run(["git", "-C", str(root), "ls-files", "-s"], check=True, capture_output=True,
                          text=True).stdout.splitlines()
    return [r.split("\t", 1)[1] for r in rows if not r.startswith("160000")]


def gitlinks(root: Path = REPO, ref: str = "HEAD") -> dict[str, str]:
    """返回 {子模块路径: 钉住的 sha}。"""
    rows = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--full-tree", ref], check=True,
                          capture_output=True, text=True).stdout.splitlines()
    out = {}
    for r in rows:
        meta, path = r.split("\t", 1)
        mode, _typ, sha = meta.split()
        if mode == "160000":
            out[path] = sha
    return out


def resolve_paths(args_paths: list[str] | None, manifest: Path | None) -> list[str]:
    """命令行 ``--paths`` 优先（可给目录，展开为其下跟踪文件）；否则用清单。"""
    if not args_paths:
        return read_manifest(manifest)
    tracked = tracked_files()
    out: list[str] = []
    for a in args_paths:
        a = a.rstrip("/")
        hits = [t for t in tracked if t == a or t.startswith(a + "/")]
        if not hits:
            raise SystemExit(f"--paths 里的 {a!r} 没有对应的跟踪文件")
        out.extend(h for h in hits if h not in out)
    return out
