#!/usr/bin/env bash
# 噪声基线 GL 单遍运行包装（1003-noise-baseline-plan.md 第一部分 3.3，第二部分 §一、§二 S3、§五）。
#
# 每一遍生成都经本脚本启动，保证「真是新跑」、预算累计不超、资产与环境指纹落盘（噪声工具只留生成这一条线，
# --kind 只认 gen；评估、环境摘要两类已删，1003 代码测试维护计划「细则 2.5」）：
#   1. 导出 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1；
#   2. noise_run.py preflight：输出根不存在或为空、资产与 tokenizer 全量 sha256 对锁、预算账本登记（fsync）、
#      环境指纹与来源报告；不过即写 EXIT_CODE= 尾行退出，不执行真实命令、不调 finish；
#   3. 执行 -- 之后的真实命令，输出 tee 进日志；
#   4. noise_run.py finish：账本追加 finish 行、来源报告追加 finished_at／exit_code／实际数；
#   5. 最后一行写 EXIT_CODE=<n>（真实命令非零取其值，否则取 finish 的返回码），作为 Monitor 统一完成信号。
#
# 用法：
#   bash dev-scripts/parity/noise_run_gl.sh --pass <名> --kind gen --out-root <dir> --log <日志> \
#     --provenance-out <json> --budget-ledger <jsonl> --budget-caps <json> --attempts N --resets N --retries N \
#     [--assets-lock J --asset-dir 名=路径 ...] [--tokenizer P --tokenizer-sha256 H] [--policy-repo 名=路径 ...] \
#     [--max-steps N] [--server-args S] [--actual-attempts N|unknown] [--actual-resets N|unknown] \
#     [--actual-retries N|unknown] -- <真实命令...>
# --actual-* 只给 finish（真实命令跑完后才知道的实际数，事先给不出时缺省 unknown，由事后统计补核）；
# 其余非本脚本自有的参数原样转给 preflight。
# 环境变量：PY 指定 Python 解释器（默认 <仓库>/.venv/bin/python）；NOISE_RUN_NVIDIA_SMI 可覆盖 nvidia-smi。
# 退出码：真实命令的退出码；真实命令为 0 时取 finish 的返回码（实际数超登记为 6）；preflight 不过取其返回码
#   （输出根非空 4、预算 5、资产 7、参数 2）；被 TERM／INT／HUP 中断 143／130／129。
# 不含任何本机或 NFS 绝对路径默认值。
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
NOISE_PY="$HERE/noise_run.py"

PASS_NAME="" ; KIND="" ; OUT_ROOT="" ; LOG="" ; PROV=""
ACT_ATTEMPTS="unknown" ; ACT_RESETS="unknown" ; ACT_RETRIES="unknown"
PRE_ARGS=() ; CMD=()

usage() { sed -n '2,26p' "${BASH_SOURCE[0]}" >&2; }
die2() { echo "NOISE_RUN=FAIL reason=bad_args detail=$1" >&2; echo "EXIT_CODE=2"; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pass) PASS_NAME="$2"; shift 2;;
    --kind) KIND="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --log) LOG="$2"; shift 2;;
    --provenance-out) PROV="$2"; shift 2;;
    --actual-attempts) ACT_ATTEMPTS="$2"; shift 2;;
    --actual-resets) ACT_RESETS="$2"; shift 2;;
    --actual-retries) ACT_RETRIES="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    --) shift; CMD=("$@"); break;;
    *)
      [[ $# -ge 2 ]] || die2 "参数 $1 缺值"
      PRE_ARGS+=("$1" "$2"); shift 2;;
  esac
done
[[ -n "$PASS_NAME" && -n "$KIND" && -n "$OUT_ROOT" && -n "$LOG" && -n "$PROV" ]] \
  || die2 "缺少必需参数（--pass --kind --out-root --log --provenance-out）"
[[ "$PASS_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die2 "--pass 只许字母数字 . _ -"
[[ "$KIND" == gen ]] || die2 "--kind 只认 gen（评估 eval 与环境摘要 digest 两类已删）"
[[ ${#CMD[@]} -gt 0 ]] || die2 "缺少 -- 之后的真实命令"
[[ -x "$PY" ]] || die2 "解释器不可执行：$PY（用环境变量 PY 指定）"

mkdir -p "$(dirname "$LOG")"
log() { echo "$*" | tee -a "$LOG"; }

REAL_CMD="$(printf '%q ' "${CMD[@]}")"
log "NOISE_RUN_START pass=$PASS_NAME kind=$KIND out_root=$OUT_ROOT host=$(hostname) job=${SLURM_JOB_ID:-none} $(date -Is)"

# ---------------- 1. preflight ----------------
set +e
"$PY" "$NOISE_PY" preflight --pass "$PASS_NAME" --kind "$KIND" --out-root "$OUT_ROOT" \
  --provenance-out "$PROV" --real-command "$REAL_CMD" ${PRE_ARGS[@]+"${PRE_ARGS[@]}"} 2>&1 | tee -a "$LOG"
prc=${PIPESTATUS[0]}
set -e
if (( prc != 0 )); then
  log "NOISE_RUN_DONE pass=$PASS_NAME stage=preflight preflight_rc=$prc $(date -Is)"
  log "EXIT_CODE=$prc"
  exit "$prc"
fi

# ---------------- 2. 真实命令 ----------------
# 子 shell 内管道：真实命令输出 tee 进日志，tee 忽略 TERM／INT／HUP 以免收尾行写不进日志；子 shell 以真实命令的退出码退出。
INTERRUPTED=0
CMD_SUB=""
on_sig() {  # $1 = 中断退出码
  INTERRUPTED="$1"
  if [[ -n "$CMD_SUB" ]]; then
    pkill -TERM -P "$CMD_SUB" 2>/dev/null || true
  fi
}
trap 'on_sig 143' TERM
trap 'on_sig 130' INT
trap 'on_sig 129' HUP

set +e
(
  set -o pipefail
  "${CMD[@]}" 2>&1 | (trap '' TERM INT HUP; exec tee -a "$LOG")
  exit "${PIPESTATUS[0]}"
) &
CMD_SUB=$!
wait "$CMD_SUB"; rc=$?
while kill -0 "$CMD_SUB" 2>/dev/null; do wait "$CMD_SUB"; rc=$?; done
set -e
trap '' TERM INT HUP  # 收尾期间不再被打断
(( INTERRUPTED != 0 )) && rc=$INTERRUPTED

# ---------------- 3. finish ----------------
set +e
"$PY" "$NOISE_PY" finish --pass "$PASS_NAME" --provenance "$PROV" --exit-code "$rc" \
  --actual-attempts "$ACT_ATTEMPTS" --actual-resets "$ACT_RESETS" --actual-retries "$ACT_RETRIES" 2>&1 | tee -a "$LOG"
frc=${PIPESTATUS[0]}
set -e

final=$rc
(( final == 0 )) && final=$frc
log "NOISE_RUN_DONE pass=$PASS_NAME cmd_rc=$rc finish_rc=$frc interrupted=$INTERRUPTED $(date -Is)"
log "EXIT_CODE=$final"
exit "$final"
