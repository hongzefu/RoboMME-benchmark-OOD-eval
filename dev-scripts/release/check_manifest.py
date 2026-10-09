"""闸门三：main 恰等于清单；附带子模块可取性检查（只在 dev）。

两种用法：

1. 清单检查（缺省）：``check_manifest.py [--ref REF] [--tree DIR] [--collect] [--expect-collected N]``
   * 清单每行都必须是 ``--ref``（缺省 HEAD）里的普通跟踪文件 → ``missing``；
   * ``--tree DIR``：DIR 是 main 的检出，其跟踪文件必须恰好等于清单 → ``extra``／``missing``，
     其 5 个 gitlink 必须与 ``--ref`` 相同 → ``gitlinks``；
   * ``--collect``：用 ``git archive`` 把清单内文件（取 ``--ref`` 的已提交字节）导到临时目录，
     ``third_party`` 以 symlink 指向本仓子模块，在临时目录里 ``pytest --collect-only``，
     统计收集到的测试文件数 → ``collected``；导入失败即 FAIL。证明公开测试不依赖任何 dev 文件。
   末行 ``PUBLIC_MANIFEST=PASS|FAIL listed=<n> missing=<m> extra=<e> collected=<c|NA> gitlinks=<g>``。

2. 子模块检查：``check_manifest.py --submodules``：对每个 gitlink 用 ``gh repo view --json visibility`` 与
   ``gh api repos/<o>/<r>/commits/<sha>`` 核对仓库公开、钉住的提交可取。
   末行 ``SUBMODULE_PUBLIC=PASS|FAIL repos=<n> public=<p> reachable=<r>``。
"""
from __future__ import annotations

import argparse
import configparser
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _manifest as M  # noqa: E402

REQUIRED = ["readme.md", "LICENSE", "pyproject.toml", "uv.lock", ".python-version", ".gitignore", ".gitmodules",
            "scripts/evaluate.py"]


def submodule_urls(root: Path = M.REPO) -> dict[str, str]:
    cp = configparser.ConfigParser()
    cp.read(root / ".gitmodules")
    return {cp[s]["path"]: cp[s]["url"] for s in cp.sections()}


def check_submodules() -> int:
    links = M.gitlinks()
    urls = submodule_urls()
    public = reachable = 0
    for path, sha in sorted(links.items()):
        url = urls.get(path, "")
        m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
        if not m:
            print(f"{path}: 无法从 url 解析 owner/repo：{url!r}")
            continue
        slug = f"{m.group(1)}/{m.group(2)}"
        vis = subprocess.run(["gh", "repo", "view", slug, "--json", "visibility", "-q", ".visibility"],
                             capture_output=True, text=True)
        v = vis.stdout.strip()
        got = subprocess.run(["gh", "api", f"repos/{slug}/commits/{sha}", "-q", ".sha"], capture_output=True, text=True)
        ok_sha = got.returncode == 0 and got.stdout.strip() == sha
        public += v == "PUBLIC"
        reachable += ok_sha
        print(f"{path}: {slug} visibility={v or vis.stderr.strip()[:80]} sha={sha[:12]} reachable={int(ok_sha)}")
    n = len(links)
    ok = n > 0 and public == n and reachable == n
    print(f"SUBMODULE_PUBLIC={'PASS' if ok else 'FAIL'} repos={n} public={public} reachable={reachable}")
    return 0 if ok else 1


def collect(manifest: list[str], ref: str) -> tuple[int | None, str]:
    """把清单导到临时目录后做 pytest 收集；返回（测试文件数，失败说明）。"""
    with tempfile.TemporaryDirectory(prefix="public-collect-") as td:
        tmp = Path(td)
        arc = subprocess.run(["git", "-C", str(M.REPO), "archive", "--format=tar", ref, "--", *manifest],
                             check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
        (tmp / "third_party").mkdir(exist_ok=True)
        for path in M.gitlinks(ref=ref):
            dst = tmp / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(M.REPO / path)
        env = dict(os.environ, PYTHONPATH=str(tmp / "src"))
        py = M.REPO / ".venv" / "bin" / "python"
        probe = subprocess.run([str(py), "-c", "import robomme_ood_eval as m; print(m.__file__)"], cwd=tmp, env=env,
                               capture_output=True, text=True)
        if not probe.stdout.strip().startswith(str(tmp)):
            return None, f"robomme_ood_eval 没有从临时目录导入：{probe.stdout.strip()} {probe.stderr.strip()[-300:]}"
        r = subprocess.run([str(py), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=tmp, env=env,
                           capture_output=True, text=True)
        files = {line.split("::", 1)[0] for line in r.stdout.splitlines() if "::" in line}
        if r.returncode != 0:
            return None, "pytest 收集失败：\n" + (r.stdout + r.stderr)[-3000:]
        return len(files), ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--tree", type=Path, help="main 的检出目录（比对跟踪文件与 gitlink）")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--expect-collected", type=int)
    ap.add_argument("--submodules", action="store_true")
    a = ap.parse_args(argv)
    if a.submodules:
        return check_submodules()

    manifest = M.read_manifest(a.manifest)
    tracked = set(M.tracked_files(ref=a.ref))
    missing = [p for p in manifest if p not in tracked]
    missing += [p for p in REQUIRED if p not in manifest and p not in missing]
    for p in missing:
        print(f"MISSING {p}")
    extra: list[str] = []
    gl_ok = True
    links = M.gitlinks(ref=a.ref)
    if a.tree:
        tree_files = set(M.tracked_files(root=a.tree))
        extra = sorted(tree_files - set(manifest))
        for p in extra:
            print(f"EXTRA {p}")
        for p in sorted(set(manifest) - tree_files):
            if p not in missing:
                missing.append(p)
                print(f"MISSING_IN_TREE {p}")
        tree_links = dict(
            (r.split("\t", 1)[1], r.split()[1])
            for r in subprocess.run(["git", "-C", str(a.tree), "ls-files", "-s"], check=True, capture_output=True,
                                    text=True).stdout.splitlines() if r.startswith("160000"))
        gl_ok = tree_links == links
        if not gl_ok:
            print(f"GITLINK_MISMATCH tree={tree_links} ref={links}")
    collected: int | None = None
    coll_ok = True
    if a.collect:
        collected, why = collect(manifest, a.ref)
        if collected is None:
            coll_ok = False
            print(why)
        elif a.expect_collected is not None and collected != a.expect_collected:
            coll_ok = False
            print(f"COLLECTED_MISMATCH got={collected} expect={a.expect_collected}")
    ok = not missing and not extra and gl_ok and coll_ok
    c = "NA" if collected is None else collected
    print(f"PUBLIC_MANIFEST={'PASS' if ok else 'FAIL'} listed={len(manifest)} missing={len(missing)} extra={len(extra)} "
          f"collected={c} gitlinks={len(links) if gl_ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
