#!/usr/bin/env bash
# dev → main 单向同步（只在 dev；只在用户说「同步到 main」时由主会话在 sled-vail 上跑）。
# main 的全部内容 = dev 上 dev-scripts/release/public-manifest.txt 列出的文件 + 5 个 gitlink，逐字节取自 dev 的已提交版本。
#
# 用法：
#   bash dev-scripts/release/sync_to_main.sh --message "<英文摘要>"   # 正式同步并快进推送 main
#   bash dev-scripts/release/sync_to_main.sh --dry-run                 # 跑到第 6 步（算出差异）为止，不提交不推送
# 末行：SYNC_MAIN=PASS|NOOP|FAIL dev=<sha> main=<sha> files=<n> changed=<m>
#
# 八步：①前置（在 dev、工作区 clean、HEAD == origin/dev）②三道闸门 ③清单内测试短测
#       ④取 main 临时检出并核对最新提交带 Dev-Source: 尾行 ⑤复制与逐路径暂存（git rm / git add 喂清单，禁 add -A）
#       ⑥无差异则 NOOP ⑦提交（尾行 Dev-Source: <dev 完整 sha>）并快进推送 ⑧清理临时检出
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REL="$REPO/dev-scripts/release"
MANIFEST="$REL/public-manifest.txt"
PY="$REPO/.venv/bin/python"
DRY=0
MSG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --message) MSG="$2"; shift 2 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
fail() { echo "$1" >&2; echo "SYNC_MAIN=FAIL dev=${DEV_SHA:-NA} main=${MAIN_SHA:-NA} files=0 changed=0 reason=$2"; exit 1; }
if [[ $DRY -eq 0 && -z "$MSG" ]]; then
  fail "正式同步必须给 --message（英文摘要，写进 main 的提交 subject）" no_message
fi
if LC_ALL=C grep -qP '[^\x00-\x7F]' <<<"$MSG"; then
  fail "--message 必须是纯 ASCII 英文" message_not_ascii
fi

cd "$REPO"
# ① 前置
BR="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BR" == "dev" ]] || fail "当前分支是 $BR，不是 dev" not_on_dev
DIRTY="$(git status --porcelain --ignore-submodules=dirty -- . ':!docs/subagent-stats')"
[[ -z "$DIRTY" ]] || fail "工作区不 clean：
$DIRTY" dirty
git fetch -q origin
DEV_SHA="$(git rev-parse HEAD)"
[[ "$DEV_SHA" == "$(git rev-parse origin/dev)" ]] || fail "HEAD 与 origin/dev 不一致，先 push" dev_not_pushed
MAIN_SHA="$(git rev-parse origin/main)"

# ② 闸门
"$PY" "$REL/check_public_lang.py" --show 20 | tail -n 21 || fail "PUBLIC_LANG 未通过" lang
"$PY" "$REL/check_public_paths.py" --show 20 | tail -n 21 || fail "PUBLIC_PATHS 未通过" paths
"$PY" "$REL/check_manifest.py" --collect | tail -n 21 || fail "PUBLIC_MANIFEST 未通过" manifest

# ③ 清单内测试短测
mapfile -t TESTS < <(grep -E '^tests/.*/?test_[^/]*\.py$|^tests/test_[^/]*\.py$' "$MANIFEST")
"$PY" -m pytest -m 'not slow' -q -p no:cacheprovider "${TESTS[@]}" | tail -n 3 || fail "清单内测试未全过" tests

# ④ 取 main 临时检出
WT="$(mktemp -d "${TMPDIR:-/tmp}/main-sync.XXXXXX")"
cleanup() { git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$WT"; }
trap cleanup EXIT
rmdir "$WT"
git worktree add -q --detach "$WT" origin/main
git -C "$WT" log -1 --format=%B | grep -q '^Dev-Source: ' || fail "origin/main 最新提交没有 Dev-Source: 尾行，main 可能被别处写过" main_not_synced

# ⑤ 复制与逐路径暂存
LIST="$(mktemp "${TMPDIR:-/tmp}/manifest.XXXXXX")"
grep -vE '^\s*(#|$)' "$MANIFEST" | sed 's/[[:space:]]*$//' > "$LIST"
DEL="$(mktemp "${TMPDIR:-/tmp}/delete.XXXXXX")"
git -C "$WT" ls-files -s | awk '$1!="160000"{sub(/^[^\t]*\t/,""); print}' | sort > "$DEL.all"
sort "$LIST" | comm -23 "$DEL.all" - > "$DEL"
if [[ -s "$DEL" ]]; then
  git -C "$WT" rm -q --pathspec-from-file="$DEL"
fi
git archive --format=tar "$DEV_SHA" -- $(cat "$LIST") | tar -x -C "$WT"
git -C "$WT" add --pathspec-from-file="$LIST"
while read -r mode sha _stage path; do
  [[ "$mode" == "160000" ]] || continue
  git -C "$WT" update-index --add --cacheinfo "160000,$sha,$path"
done < <(git ls-tree -r "$DEV_SHA" | awk '$1=="160000"{print $1, $3, 0, $4}')
FILES="$(wc -l < "$LIST")"
CHANGED="$(git -C "$WT" diff --cached --name-only | wc -l)"
rm -f "$LIST" "$DEL" "$DEL.all"

# ⑥ 无差异
if git -C "$WT" diff --cached --quiet; then
  echo "SYNC_MAIN=NOOP dev=$DEV_SHA main=$MAIN_SHA files=$FILES changed=0"
  exit 0
fi
git -C "$WT" diff --cached --stat | tail -n 15
if [[ $DRY -eq 1 ]]; then
  echo "SYNC_MAIN=DRYRUN dev=$DEV_SHA main=$MAIN_SHA files=$FILES changed=$CHANGED"
  exit 0
fi

# ⑦ 提交与快进推送
git -C "$WT" commit -q -m "Sync from dev @${DEV_SHA:0:7}: $MSG" -m "Dev-Source: $DEV_SHA"
git -C "$WT" push -q origin HEAD:main || fail "推送 main 被拒（非快进或认证失败），原样交用户" push_rejected
NEW_MAIN="$(git -C "$WT" rev-parse HEAD)"
git fetch -q origin
# ⑧ 清理由 trap 完成
echo "SYNC_MAIN=PASS dev=$DEV_SHA main=$NEW_MAIN files=$FILES changed=$CHANGED"
