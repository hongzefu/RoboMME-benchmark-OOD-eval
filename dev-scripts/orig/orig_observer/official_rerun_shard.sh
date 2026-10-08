#!/usr/bin/env bash
# 原侧重跑单片执行器（v7.5eval 方案 2.3 编排器，S7 改为本轮第二档批次 3a 用）：在 GL 的 srun 步骤内（或本机给定 GPU）
# 按历史顺序先跑 SimpleMemVLA 片、再跑 FrameSamp+Modulation 片，均用原版代码 + 只读观测器（run_orig_smvla.sh、run_orig_framesamp_modul.sh）。
# 用法：ORIG_STAGE_ROOT=<本轮 stage 根> bash official_rerun_shard.sh <run 名> <分片 0-9> [--run-idx K] [--port-base P] [--dry-run]
# - run 名只许 [A-Za-z0-9._-]；--run-idx（缺省 0）决定端口号段，同节点并发的不同 run 须给不同值。
# - 清单：/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v7-eval/eval-official-xhard0-192.jsonl（只读；FrameSamp+Modulation 用 SHARD=i，
#   SMVLA 用 i/10）。仅当 run 名以 smoke 结尾时允许用环境变量 MANIFEST 覆盖（冒烟用小清单；分片口径不变）。
# - 逐局结果：$ORIG_STAGE_ROOT/<run>/{smvla,perceptual-framesamp-modul}/s<i>/（新目录）。
# - 观测记录与原版 mp4：先写 NODE_TMP=/tmp/orig-<run>-s<i>-<作业号>，每个策略跑完 rsync -a 到
#   $ORIG_STAGE_ROOT/<run>/stage/<policy>/s<i>/，核对（文件清单 + 大小 + 全部 trace.jsonl 的 sha256）后删节点副本。
# - SMVLA 照抄原版 gl_run_official_xhard0.sh 的内层循环：最多 3 次、每次 --resume、退出码 0 或 2 即停；第 2、3 次带 ORIG_RESUME=1。
# - 起跑前、结束后都用 E0 sha 清单（缺省 $NFS/v75eval/e0-sha.txt，可由 ORIG_E0_SHA 指定；只读）核对本片 E0 文件与清单的 sha256 未变（R7）。
# - V75_ENCODE_CPUS = 本进程 CPU 亲和集合的最后一核（FrameSamp+Modulation 代理记账用）。
# 输出：OFFICIAL_SHARD_DONE run= shard= smvla_rows= framesamp_modul_rows= smvla_rc= framesamp_modul_rc= staged=yes|no，最后 EXIT_CODE=。
set -uo pipefail
OBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NFS=/nfs/turbo/coe-chaijy-unreplicated/hongzefu
RUN="${1:-}"; SHARD="${2:-}"; shift 2 2>/dev/null || true
PORT_BASE=19200; DRY=0; RUN_IDX=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-base) PORT_BASE="$2"; shift 2;;
    --run-idx) RUN_IDX="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "错误: 未知参数 $1"; echo "EXIT_CODE=2"; exit 2;;
  esac
done
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "错误: run 名只许 [A-Za-z0-9._-]：$RUN"; echo "EXIT_CODE=2"; exit 2; }
[[ "$RUN_IDX" =~ ^[0-9]+$ ]] || { echo "错误: --run-idx 须为非负整数：$RUN_IDX"; echo "EXIT_CODE=2"; exit 2; }
: "${ORIG_STAGE_ROOT:?必须给本轮 stage 根 ORIG_STAGE_ROOT}"
ORIG_STAGE_ROOT="$(readlink -m "$ORIG_STAGE_ROOT")"; export ORIG_STAGE_ROOT
[[ "$SHARD" =~ ^[0-9]$ ]] || { echo "错误: 分片须为 0-9：$SHARD"; echo "EXIT_CODE=2"; exit 2; }
E0_MANIFEST="$NFS/v7-eval/eval-official-xhard0-192.jsonl"
if [[ "$RUN" == *smoke && -n "${MANIFEST:-}" ]]; then MANIFEST="$(readlink -f "$MANIFEST")"; else MANIFEST="$E0_MANIFEST"; fi
# 端口：每个 run 占 100 个号段，每片 4 个（server、代理），同节点两遍并发也不撞
PORT=$((PORT_BASE + RUN_IDX * 100 + SHARD * 4))
EP_ROOT="$ORIG_STAGE_ROOT/$RUN"
STAGE_ROOT="$ORIG_STAGE_ROOT/$RUN/stage"
NODE_TMP="/tmp/orig-$RUN-s$SHARD-${SLURM_JOB_ID:-local}"
SMVLA_OUT="$EP_ROOT/smvla/s$SHARD"; FRAMESAMP_MODUL_SAVE="$EP_ROOT/perceptual-framesamp-modul/s$SHARD"
E0_SHA="${ORIG_E0_SHA:-$NFS/v75eval/e0-sha.txt}"  # 原文件随 10-03 清理删除时由主会话以 artifacts/v7.5eval/input-manifest.json 重建并经此变量指定
V75_ENCODE_CPUS="$(python3 -c 'import os; print(max(os.sched_getaffinity(0)))')"
export V75_ENCODE_CPUS

run() {  # 执行或（--dry-run）只打印命令
  if [[ "$DRY" = 1 ]]; then printf 'DRY_RUN:'; printf ' %q' "$@"; printf '\n'; return 0; fi
  "$@"
}

e0_check() {  # 本片相关的 E0 sha256 行：清单 + FrameSamp+Modulation 第 i 片 + SMVLA 第 i 片
  local tag="$1" pat
  pat="eval-official-xhard0-192\.jsonl$|mme-official-full-s$SHARD/|episodes-shard$(printf %02d "$SHARD")of10\.jsonl$"  # 历史目录名：E0 sha256 文件（v7.5eval 留档）原样匹配
  local n; n=$(grep -cE "$pat" "$E0_SHA")
  if [[ "$n" != 3 ]] || ! grep -E "$pat" "$E0_SHA" | sha256sum -c --quiet -; then
    echo "E0_FREEZE_CHECK=FAIL when=$tag lines=$n"; return 1
  fi
  echo "E0_FREEZE_CHECK=PASS when=$tag lines=$n"
}

stage() {  # stage <policy> <节点目录>：rsync 到 NFS 中转、核对、删节点副本
  local pol="$1" src="$2"
  local dst="$STAGE_ROOT/$pol/s$SHARD"
  if [[ "$DRY" = 1 ]]; then run rsync -a "$src/" "$dst/"; run rm -rf "$src"; return 0; fi
  [[ -d "$src" ]] || { echo "STAGE_FAIL policy=$pol 节点目录不存在 $src"; return 1; }
  mkdir -p "$dst" && rsync -a "$src/" "$dst/" || { echo "STAGE_FAIL policy=$pol rsync"; return 1; }
  local a b
  a=$(cd "$src" && find . -type f -printf '%P %s\n' | LC_ALL=C sort | sha256sum)
  b=$(cd "$dst" && find . -type f -printf '%P %s\n' | LC_ALL=C sort | sha256sum)
  [[ "$a" == "$b" ]] || { echo "STAGE_FAIL policy=$pol 文件清单或大小不符"; return 1; }
  a=$(cd "$src" && find . -name trace.jsonl -type f -print0 | LC_ALL=C sort -z | xargs -0r sha256sum)
  b=$(cd "$dst" && find . -name trace.jsonl -type f -print0 | LC_ALL=C sort -z | xargs -0r sha256sum)
  [[ "$a" == "$b" ]] || { echo "STAGE_FAIL policy=$pol trace.jsonl sha256 不符"; return 1; }
  echo "STAGED policy=$pol files=$(cd "$dst" && find . -type f | wc -l) bytes=$(du -sb "$dst" | cut -f1) dst=$dst"
  rm -rf "$src"
}

final_rows() {  # 逐局 jsonl 里去重后的 (task, source_episode) 数
  python3 -c "
import json,sys
k=set()
for p in sys.argv[1:]:
    try:
        for l in open(p):
            if l.strip():
                r=json.loads(l); k.add((r['task'],int(r['source_episode'])))
    except FileNotFoundError: pass
print(len(k))" "$@"
}

echo "=== OFFICIAL_RERUN_SHARD run=$RUN shard=$SHARD host=$(hostname) job=${SLURM_JOB_ID:-none} port=$PORT node_tmp=$NODE_TMP manifest=$MANIFEST encode_cpus=$V75_ENCODE_CPUS affinity=$(python3 -c 'import os; print(sorted(os.sched_getaffinity(0)))') dry=$DRY start=$(date -Is) ==="
e0_check before || { echo "EXIT_CODE=1"; exit 1; }
if [[ "$DRY" != 1 ]]; then
  for d in "$SMVLA_OUT" "$FRAMESAMP_MODUL_SAVE" "$STAGE_ROOT/smvla/s$SHARD" "$STAGE_ROOT/perceptual-framesamp-modul/s$SHARD" "$NODE_TMP"; do
    [[ -e "$d" ]] && { echo "错误: 目录已存在（原侧重跑一律新目录）: $d"; echo "EXIT_CODE=1"; exit 1; }
  done
  mkdir -p "$NODE_TMP/smvla" "$NODE_TMP/perceptual-framesamp-modul" "$EP_ROOT/smvla" "$EP_ROOT/perceptual-framesamp-modul"
fi

# ---- SimpleMemVLA 片（历史顺序：先 SMVLA）
# 分片总数取自清单（旧 load_episode_manifest 要求 n 恰为清单分片数；E0 清单为 10，R1 小样本清单为 1）
SMVLA_N="$(python3 -c 'import json,sys; print(len({json.loads(l)["shard"] for l in open(sys.argv[1]) if l.strip()}))' "$MANIFEST")"
SMVLA_RC=1
for attempt in 1 2 3; do
  echo "XHARD0_PASS $attempt policy=smvla"
  resume=0; [[ "$attempt" -gt 1 ]] && resume=1
  run env MANIFEST="$MANIFEST" SHARD="$SHARD/$SMVLA_N" OUTDIR="$SMVLA_OUT" REC_ROOT="$NODE_TMP/smvla/rec" \
    VIDEO_DIR="$NODE_TMP/smvla/videos" NODE_TMP="$NODE_TMP" ORIG_RESUME="$resume" \
    bash "$OBS_DIR/run_orig_smvla.sh"
  SMVLA_RC=$?
  [[ "$SMVLA_RC" = 0 || "$SMVLA_RC" = 2 ]] && break
done
echo "SMVLA_SHARD_END rc=$SMVLA_RC"
stage smvla "$NODE_TMP/smvla"; ST1=$?

# ---- FrameSamp+Modulation 片
run env MANIFEST="$MANIFEST" SHARD="$SHARD" PORT="$PORT" RUN_TAG="orig-$RUN-s$SHARD" SAVE_ROOT="$FRAMESAMP_MODUL_SAVE" \
  REC_ROOT="$NODE_TMP/perceptual-framesamp-modul/rec" VIDEO_DIR="$NODE_TMP/perceptual-framesamp-modul/videos" NODE_TMP="$NODE_TMP" \
  bash "$OBS_DIR/run_orig_framesamp_modul.sh"
FRAMESAMP_MODUL_RC=$?
echo "FRAMESAMP_MODUL_SHARD_END rc=$FRAMESAMP_MODUL_RC"
stage perceptual-framesamp-modul "$NODE_TMP/perceptual-framesamp-modul"; ST2=$?
[[ "$DRY" != 1 ]] && rmdir "$NODE_TMP" 2>/dev/null

e0_check after; E0_AFTER=$?
SROWS=$(final_rows "$SMVLA_OUT"/episodes-shard*.jsonl)
MROWS=$(final_rows "$FRAMESAMP_MODUL_SAVE/mmevla-official-xhard0/ckpt79999/seed7/episodes.jsonl")  # 历史目录名：原版 eval.py 按 policy_name 落盘（同 run_orig_framesamp_modul.sh）
STAGED=no; [[ "$ST1" = 0 && "$ST2" = 0 ]] && STAGED=yes
echo "OFFICIAL_SHARD_DONE run=$RUN shard=$SHARD smvla_rows=$SROWS framesamp_modul_rows=$MROWS smvla_rc=$SMVLA_RC framesamp_modul_rc=$FRAMESAMP_MODUL_RC staged=$STAGED"
RC=0
[[ "$SMVLA_RC" != 0 ]] && RC=$SMVLA_RC
[[ "$RC" = 0 && "$FRAMESAMP_MODUL_RC" != 0 ]] && RC=$FRAMESAMP_MODUL_RC
[[ "$RC" = 0 && "$STAGED" != yes ]] && RC=7
[[ "$RC" = 0 && "$E0_AFTER" != 0 ]] && RC=8
echo "EXIT_CODE=$RC"
exit "$RC"
