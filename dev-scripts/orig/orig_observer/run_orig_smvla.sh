#!/usr/bin/env bash
# SimpleMemVLA 原侧（原版分支 official-xhard0-0929／4e0c04f + 外挂只读观测器）单片启动器：环境与参数逐项照抄原版
#   SimpleMemVLA@4e0c04f:scripts/run_official_xhard0.sh
# 差别（计划第二部分一节 S7，均须写进留档）：
#   1. 用 orig_observer/smvla_wrap.py 代替 ``-m robomme_sim.eval_success``（同进程执行同一模块，外挂只读钩子，逐局写
#      <REC_ROOT>/<key>.a<N>/{trace.jsonl,frames/,arrays.npz}，route smvla/orig）；PYTHONPATH = 原版工作树 : 本目录 : 原 PYTHONPATH。
#   2. 起跑断言（R1）：REPO（缺省主会话 git worktree add 检出的 /nfs/.../SimpleMemVLA-official-xhard0）HEAD =
#      4e0c04fdcbbd2fe90921bb20bf40bd105816d501 且 git status --porcelain 为空，不符即 RUN_BLOCKED（NFS 主仓库工作区在
#      别的分支，不得直接用）。解释器 <主仓库>/.venv-robomme/bin/python，权重 <主仓库>/checkpoints/simplememvla_robomme。
#   3. OUTDIR／REC_ROOT 必须是新目录且落在本轮 stage 根 ORIG_STAGE_ROOT（必给）或 NODE_TMP 下，历史目录一律拒绝（R7）；
#      ORIG_RESUME=1 只许续用标记 .orig-created 中清单 sha256 与分片都一致的目录（REC_ROOT 续用时局目录号接着编）；
#      VIDEO_DIR=none 关闭录像（同历史）。
#   4. 预算（R6、R11）：起跑前对本片未完成的每局经 orig_budget.py 调 S8 budget_ledger.py（先 release 上一遍未结 rid，
#      再 reserve --resets 6 --route smvla/orig --key <key>；原版 SimEnvService 内部 reset 重试不关：build+reset 各 1 次
#      × 最多 3 次尝试），任一失败即 RUN_BLOCKED reason=budget；跑完对有终态的局 commit。rid 记在 OUTDIR/budget-rids.json。
#      账本路径：BUDGET_LEDGER_ARGS="--ledger <路径>"（顶层参数，放在子命令之前）或环境变量 SGEVAL_BUDGET_LEDGER。
#   5. 始终带 --resume（与历史一致：原版 gl_run_official_xhard0.sh 每遍都传 --resume）；3 遍重试由编排器负责。
#   6. 收尾由 observer_status.py 写 observer-status.json 与末行 OBSERVER_COMPLETE=…（无代理：conns=0 mismatch=0
#      report=ok），并写 orig-attempts.json；EXIT_CODE= 保持原语义（= 原版进程退出码）。
#   7. 可选 NODE_TMP：OUTDIR／REC_ROOT 默认落在节点本地盘，跑完由调用方搬回。
# 用法：MANIFEST=<清单.jsonl> SHARD=<i/n> ORIG_STAGE_ROOT=<本轮 stage 根> OUTDIR=<新目录> REC_ROOT=<新目录> \
#       [VIDEO_DIR=<目录>] [NODE_TMP=<目录>] [SMVLA_ORIG_REPO=<原版工作树>] [ORIG_RESUME=1] [BUDGET_LEDGER_ARGS=...] \
#       bash dev-scripts/orig/orig_observer/run_orig_smvla.sh [额外参数，透传给 eval_success]
set -euo pipefail
OBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_REPO="$(cd "$OBS_DIR/../../.." && pwd)"
# shellcheck source=orig_observer_lib.sh
source "$OBS_DIR/orig_observer_lib.sh"
ORIG_COMMIT=4e0c04fdcbbd2fe90921bb20bf40bd105816d501
REPO="${SMVLA_ORIG_REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA-official-xhard0}"
cd "$REPO" || { echo "RUN_BLOCKED reason=orig_repo_missing repo=$REPO"; echo "EXIT_CODE=1"; exit 1; }
orig_pin_check "$REPO" "$ORIG_COMMIT" smvla || { echo "EXIT_CODE=1"; exit 1; }
if [ "${ORIG_OBSERVER_SELFTEST:-0}" = 1 ]; then echo "ORIG_SELFTEST_GUARD=PASS"; echo "EXIT_CODE=0"; exit 0; fi
# 本检出（git worktree）不带 venv 与权重：默认借主仓库的
MAIN_REPO="${MAIN_REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA}"
PYBIN="${PYBIN:-$MAIN_REPO/.venv-robomme/bin/python}"
CHECKPOINT="$(readlink -f "${CHECKPOINT:-$MAIN_REPO/checkpoints/simplememvla_robomme}")"
: "${MANIFEST:?}" "${SHARD:?}"
if [ -n "${NODE_TMP:-}" ]; then
  mkdir -p "$NODE_TMP"
  OUTDIR="${OUTDIR:-$NODE_TMP/out}"
  REC_ROOT="${REC_ROOT:-$NODE_TMP/rec}"
fi
: "${OUTDIR:?必须显式给新目录 OUTDIR（或 NODE_TMP）}" "${REC_ROOT:?必须显式给新目录 REC_ROOT（或 NODE_TMP）}"
OUTDIR="$(readlink -m "$OUTDIR")"; REC_ROOT="$(readlink -m "$REC_ROOT")"; MANIFEST="$(readlink -f "$MANIFEST")"
VIDEO_DIR="${VIDEO_DIR:-$OUTDIR/videos}"
# 与原版 gl_run_official_xhard0.sh 一致：VIDEO_DIR=none 关闭录像
if [ "$VIDEO_DIR" = none ]; then VIDEO_DIR=""; else VIDEO_DIR="$(readlink -m "$VIDEO_DIR")"; fi
i="${SHARD%/*}"; n="${SHARD#*/}"
EPISODE_LOG="$OUTDIR/episodes-shard$(printf %02d "$i")of$(printf %02d "$n").jsonl"
NFS=/nfs/turbo/coe-chaijy-unreplicated/hongzefu
: "${ORIG_STAGE_ROOT:?必须给本轮 stage 根 ORIG_STAGE_ROOT}"
ORIG_STAGE_ROOT="$(readlink -m "$ORIG_STAGE_ROOT")"
FORBID=("$NFS/v7-eval/" "$NFS/v7-eval-stage/" "$NFS/v75eval/" "$NFS/sgeval-20261004/" "$NFS/eval-out/"
  "$BENCH_REPO/artifacts/v7.5eval/" "/data/hongzefu/robomme_benchmark_MotionJEPANewTask/artifacts/v7.5eval/"
  "$REPO/" "$MAIN_REPO/")
ALLOW=("$ORIG_STAGE_ROOT/")
[ -n "${NODE_TMP:-}" ] && ALLOW+=("$(readlink -m "$NODE_TMP")/")
for d in "$ORIG_STAGE_ROOT" "$OUTDIR" "$REC_ROOT" ${VIDEO_DIR:+"$VIDEO_DIR"}; do
  for f in "${FORBID[@]}"; do
    case "$d/" in "$f"*) echo "错误: 输出目录落在历史／原版目录下: $d"; echo "EXIT_CODE=1"; exit 1;; esac
  done
  [ "$d" = "$ORIG_STAGE_ROOT" ] && continue
  ok=0; for a in "${ALLOW[@]}"; do case "$d/" in "$a"*) ok=1;; esac; done
  [ "$ok" = 1 ] || { echo "错误: 输出目录不在白名单根下（${ALLOW[*]}）: $d"; echo "EXIT_CODE=1"; exit 1; }
done
MARK="manifest_sha256=$(sha256sum "$MANIFEST" | cut -c1-64) shard=$SHARD"
if [ "${ORIG_RESUME:-0}" = 1 ]; then
  if [ ! -f "$OUTDIR/.orig-created" ] || [ "$(cat "$OUTDIR/.orig-created")" != "$MARK" ]; then
    echo "错误: ORIG_RESUME=1 但 $OUTDIR/.orig-created 缺失或与本次清单／分片不符"; echo "EXIT_CODE=1"; exit 1
  fi
  echo "ORIG_RESUME: 续用本脚本建的 $OUTDIR（$MARK）"
else
  if [ -d "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -name 'episodes*.jsonl' -print -quit 2>/dev/null)" ]; then
    echo "错误: OUTDIR 已有 episodes*.jsonl（必须新目录）: $OUTDIR"; echo "EXIT_CODE=1"; exit 1
  fi
  if [ -d "$REC_ROOT" ] && [ -n "$(ls -A "$REC_ROOT" 2>/dev/null)" ]; then
    echo "错误: REC_ROOT 非空（必须新目录）: $REC_ROOT"; echo "EXIT_CODE=1"; exit 1
  fi
fi
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for cand in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$cand" ] && { export VK_ICD_FILENAMES="$cand"; break; }
  done
fi
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export MPLBACKEND=Agg MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
export PYTHONPATH="$REPO:$OBS_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
# 以下沿用原版 eval_robomme.sh 的环境设定
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/torchinductor}"
export REC_ROOT
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$OUTDIR" "$REC_ROOT"
[ -f "$OUTDIR/.orig-created" ] || echo "$MARK" > "$OUTDIR/.orig-created"
unset __EGL_VENDOR_LIBRARY_DIRS SAPIEN_DISABLE_RAY_TRACING ROBOMME_GPU_RASTER XLA_FLAGS || true
REC_ROOT="$REC_ROOT/preflight" "$PYBIN" "$OBS_DIR/smvla_wrap.py" --orig-preflight \
  || { echo "错误: 观测器包装自检失败"; echo "EXIT_CODE=1"; exit 1; }
PENDING=$(orig_pending_count "$MANIFEST" "$i" "$EPISODE_LOG") || { echo "RUN_BLOCKED reason=pending_count_failed"; echo "EXIT_CODE=1"; exit 1; }
orig_budget_prepare --manifest "$MANIFEST" --shard "$i" --episode-log "$EPISODE_LOG" --state "$OUTDIR/budget-rids.json" \
  --resets 6 --route smvla/orig || { echo "EXIT_CODE=1"; exit 1; }
echo "XHARD0_PREFLIGHT host=$(hostname) repo=$REPO head=$(git -C "$REPO" rev-parse HEAD) shard=$SHARD pending=$PENDING manifest=$MANIFEST ckpt=$CHECKPOINT ckpt_config_sha256=$(sha256sum "$CHECKPOINT/config.json" | cut -c1-64) log=$EPISODE_LOG video_dir=${VIDEO_DIR:-<关闭>} observer=$OBS_DIR rec_root=$REC_ROOT py=$PYBIN"
nvidia-smi --query-gpu=index,name,compute_mode --format=csv,noheader || true
set +e
"$PYBIN" "$OBS_DIR/smvla_wrap.py" \
  --pretrained_checkpoint "$CHECKPOINT" \
  --dataset_split test \
  --group_size 1 \
  --execute_horizon "${EXECUTE_HORIZON:-16}" \
  --max_steps "${MAX_STEPS:-1300}" \
  --num_denoising_steps "${NUM_DENOISING_STEPS:-10}" \
  --eval_temperature "${EVAL_TEMPERATURE:-1.0}" \
  --max_subtask_tokens "${MAX_SUBTASK_TOKENS:-64}" \
  --compute_dtype "${COMPUTE_DTYPE:-bfloat16}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --num_gpus 1 \
  --episode_manifest "$MANIFEST" --shard "$SHARD" --episode_log "$EPISODE_LOG" \
  ${VIDEO_DIR:+--video_dir "$VIDEO_DIR"} \
  --resume "$@"
RC=$?
# 预算结算：已有终态行的局 commit 其 rid（未完成局的 rid 留到下一遍起跑前 release）；不改 RC
orig_budget_settle --manifest "$MANIFEST" --shard "$i" --episode-log "$EPISODE_LOG" --state "$OUTDIR/budget-rids.json"
# 观测器完整性判定：只打印 OBSERVER_COMPLETE 与写 observer-status.json／orig-attempts.json，不改 RC
python3 "$OBS_DIR/observer_status.py" --rec-root "$REC_ROOT" --policy smvla --episode-log "$EPISODE_LOG" \
  --out "$REC_ROOT/observer-status.json"
[ -n "${NODE_TMP:-}" ] && echo "NODE_TMP_STAGED outdir=$OUTDIR rec_root=$REC_ROOT（由调用方搬回并核对 sha256）"
echo "EXIT_CODE=$RC"
exit "$RC"
