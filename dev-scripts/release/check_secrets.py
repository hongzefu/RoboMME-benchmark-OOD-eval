"""凭据扫描：dev 带全部历史公开前，扫所有 ref 的全部提交补丁（口径③、红线 R0-9）。

扫 ``git log --all -p`` 的新增与删除行，命中下列任一模式即记一次：HF token ``hf_``、GitHub ``ghp_``／``github_pat_``、
独立的 ``sk-``（前面不是字母数字，避免把 ``task-``、``disk-`` 之类误报）、AWS ``AKIA``、私钥头、Slack token。
白名单只放两条已知假值。命中（白名单外）即 FAIL，交用户裁决，不自行判误报、不改写历史。
末行 ``PUBLIC_SECRETS=PASS|FAIL commits=<n> hits=<m> allowlisted=<k>``。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATTERNS = {
    "hf": r"\bhf_[A-Za-z0-9]{30,}",
    "ghp": r"\bghp_[A-Za-z0-9]{36}",
    "github_pat": r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    "sk": r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{16,}",
    "aws": r"\bAKIA[0-9A-Z]{16}\b",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "slack": r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
}
ALLOW = {"sk-test-placeholder-not-a-real-key", "sk-test-sentinel-not-a-real-key"}
RX = re.compile("|".join(f"(?P<{k}>{v})" for k, v in PATTERNS.items()))


def main() -> int:
    commits = subprocess.run(["git", "-C", str(REPO), "rev-list", "--all"], check=True, capture_output=True,
                             text=True).stdout.split()
    log = subprocess.run(["git", "-C", str(REPO), "log", "--all", "-p", "--no-color", "--format=COMMIT %H"],
                         check=True, capture_output=True).stdout.decode("utf-8", errors="replace")
    hits = allowed = 0
    cur = ""
    for line in log.splitlines():
        if line.startswith("COMMIT "):
            cur = line[7:19]
            continue
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        for m in RX.finditer(line):
            if m.group(0) in ALLOW:
                allowed += 1
                continue
            hits += 1
            print(f"HIT commit={cur} kind={m.lastgroup} text={m.group(0)[:12]}…")
    ok = hits == 0
    print(f"PUBLIC_SECRETS={'PASS' if ok else 'FAIL'} commits={len(commits)} hits={hits} allowlisted={allowed}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
