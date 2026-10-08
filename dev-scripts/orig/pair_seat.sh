#!/usr/bin/env bash
# 第二档一席的串行链（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.6、第五节「第二档」；子任务 S6）：
# 同一分片、同一张卡，先原侧（run_official_hard.sh）后新侧（run_eval_gl.sh --dataset hard-verify --max-steps 1300）。
# 前一侧的服务完全退出、显存释放后才起后一侧：按该侧记下的 setsid 进程组（<stage>/sNN/orig/.v8-pgids、
# <stage>/sNN/.v8-pgids）逐个 kill -0 核实已退出且不在 nvidia-smi 计算进程列表里（≤--release-wait 秒，缺省 120），
# 并用 port_busy 核对本策略端口段已空；超时打印 RUN_BLOCKED reason=gpu_not_released 并停链（CHAIN_STOP）。
# 守卫复用 run_seat.sh（source）：port_busy、kill -0、step_cap_pairing；两侧各自的 NO_PROGRESS 无进展检测与首局
# 600 s 放宽在 run_official_hard.sh／run_seat.sh 内（同一套函数）。
#
# 衔接：原侧 rc 为 0、3（该路线预检不过，只停该路线）、6（身份未齐）、7（同步 FAIL）时照常跑新侧；原侧 rc 为 4／5
#   （基础设施或额度用尽）或被信号终止（>128）时打印 CHAIN_STOP 并不跑新侧（剩余分片由主会话重分）。第二档差异不停链。
# 用法：
#   bash pair_seat.sh --run-name R --seat NN --repo <执行副本> --stage <NFS 运行根> --shard <shard-NN.json> \
#     --policy {groundsg,pp} [--groundsg-variant V] [--qwenvl-groundsg-adapter D] [--memer-adapter D] \
#     [--groundsg-ckpt D --openpi-data-home D --tokenizer-sha256 H] [--pp-ckpt D] \
#     --policy-seed N --budget-ledger P --trajectory-cap N --shared-infra-cap N --expired-cap N --planned-first-tries N \
#     --reset-budget N --infra-retry-budget N --orig-infra-retry-budget N \
#     [--cond C] [--media-root D] [--local-root D] [--limit N] [--episode-wall S] [--sync-interval S] [--release-wait S] [--gpu N]
#   数据集与步数在本脚本里固定为 hard-verify／1300（不带 --strict-cap），起跑先过 step_cap_pairing。
#   第三阶段（接口冻结说明 2.2／2.3／2.4）：--policy-seed、五个预算参数与 --memer-adapter 本脚本不核对、不取默认值，
#   给了就原样转发给两侧（原侧 run_official_hard.sh、新侧 run_eval_gl.sh）；缺失时由两侧各自 RUN_BLOCKED（退出 3）。
# 判定行：PAIR_SEAT_DONE seat=NN policy=<label> orig_rc=… new_rc=… rc=… outcome=pass|fail|aborted；末行 EXIT_CODE=。
# 退出码：两侧都 0 为 0；任一侧为 4／5／信号时取它（显存未释放记 4）；否则取首个非零的一侧 rc；中断 130/143（HUP 129）。
set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_seat.sh
source "$HERE/run_seat.sh"

RUN_NAME="" ; SEAT="" ; STAGE="" ; SHARD="" ; POLICY="" ; MEDIA_ROOT="" ; LOCAL_ROOT="" ; COND="SGEVAL"
NEW_RESET_BUDGET="" ; NEW_INFRA_BUDGET="" ; ORIG_INFRA_BUDGET="" ; P_LIMIT=0 ; P_WALL="" ; P_SYNC="" ; RELEASE_WAIT=120
CHILD_PID="" ; ORIG_RC="" ; NEW_RC="" ; PAIR_DONE=0
P_POLICY_SEED="" ; P_MEMER_ADAPTER="" ; P_BUDGET=()  # 第三阶段：只转发（P_ 前缀，不与 run_seat.sh 的同名全局混用）

pair_die2() {
  echo "$1" >&2
  echo "PAIR_SEAT_DONE seat=${SEAT:-?} policy=${POLICY:-?} outcome=fail rc=2 reason=bad_args"
  echo "EXIT_CODE=2"
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2;;
    --seat) SEAT="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --stage) STAGE="$2"; shift 2;;
    --shard) SHARD="$2"; shift 2;;
    --policy) POLICY="$2"; shift 2;;
    --groundsg-variant) GROUNDSG_VARIANT="$2"; shift 2;;
    --qwenvl-groundsg-adapter) QWENVL_ADAPTER="$2"; shift 2;;
    --memer-adapter) P_MEMER_ADAPTER="$2"; shift 2;;
    --policy-seed) P_POLICY_SEED="$2"; shift 2;;
    --budget-ledger|--trajectory-cap|--shared-infra-cap|--expired-cap|--planned-first-tries) P_BUDGET+=("$1" "$2"); shift 2;;
    --groundsg-ckpt) GROUNDSG_CKPT="$2"; shift 2;;
    --pp-ckpt) PP_CKPT="$2"; shift 2;;
    --openpi-data-home) OPENPI_HOME="$2"; shift 2;;
    --tokenizer-sha256) TOKENIZER_SHA="$2"; shift 2;;
    --reset-budget) NEW_RESET_BUDGET="$2"; shift 2;;
    --infra-retry-budget) NEW_INFRA_BUDGET="$2"; shift 2;;
    --orig-infra-retry-budget) ORIG_INFRA_BUDGET="$2"; shift 2;;
    --cond) COND="$2"; shift 2;;
    --media-root) MEDIA_ROOT="$2"; shift 2;;
    --local-root) LOCAL_ROOT="$2"; shift 2;;
    --limit) P_LIMIT="$2"; shift 2;;
    --episode-wall) P_WALL="$2"; shift 2;;
    --sync-interval) P_SYNC="$2"; shift 2;;
    --release-wait) RELEASE_WAIT="$2"; shift 2;;
    --gpu) P_GPU="$2"; shift 2;;
    *) pair_die2 "未知参数 $1";;
  esac
done
[[ -n "$RUN_NAME" && -n "$SEAT" && -n "$REPO" && -n "$STAGE" && -n "$SHARD" && -n "$POLICY" ]] \
  || pair_die2 "缺少必需参数（--run-name --seat --repo --stage --shard --policy）"
[[ "$SEAT" =~ ^[0-9]{2}$ ]] || pair_die2 "--seat 须为两位席号 NN"
[[ "$POLICY" == "groundsg" || "$POLICY" == "pp" ]] || pair_die2 "--policy 只许 groundsg、pp"
[[ "$NEW_RESET_BUDGET" =~ ^[0-9]+$ && "$NEW_INFRA_BUDGET" =~ ^[0-9]+$ && "$ORIG_INFRA_BUDGET" =~ ^[0-9]+$ ]] \
  || pair_die2 "--reset-budget、--infra-retry-budget、--orig-infra-retry-budget 必填且为非负整数"
[[ "$RELEASE_WAIT" =~ ^[0-9]+$ ]] || pair_die2 "--release-wait 须为非负整数"
SEAT_IDX=$((10#$SEAT))
DATASET="hard-verify" ; MAX_STEPS=1300 ; STRICT_CAP=0
LABEL="$(pol_label "$POLICY")"
SEAT_STAGE="$STAGE/s$SEAT"
mkdir -p "$SEAT_STAGE" || { echo "RUN_BLOCKED reason=mkdir stage=$SEAT_STAGE"; echo "EXIT_CODE=3"; exit 3; }

common=(--run-name "$RUN_NAME" --seat "$SEAT" --repo "$REPO" --stage "$STAGE" --shard "$SHARD"
        --dataset "$DATASET" --max-steps "$MAX_STEPS")
[[ -n "$MEDIA_ROOT" ]] && common+=(--media-root "$MEDIA_ROOT")
[[ -n "$LOCAL_ROOT" ]] && common+=(--local-root "$LOCAL_ROOT")
[[ "$P_LIMIT" != "0" ]] && common+=(--limit "$P_LIMIT")
[[ -n "$P_WALL" ]] && common+=(--episode-wall "$P_WALL")
[[ -n "$P_SYNC" ]] && common+=(--sync-interval "$P_SYNC")
# 卡号缺省 0（GL 占位 job 内只见一张卡）；本机多卡并行时显式给物理卡号，两侧同卡
common+=(--gpu "${P_GPU:-0}")
# 第三阶段：模型种子与预算五参数两侧共用（同一共享账本）；有值才加
[[ -n "$P_POLICY_SEED" ]] && common+=(--policy-seed "$P_POLICY_SEED")
common+=("${P_BUDGET[@]}")
model=()
if [[ "$POLICY" == "groundsg" ]]; then
  model=(--groundsg-variant "$GROUNDSG_VARIANT" --groundsg-ckpt "$GROUNDSG_CKPT" --openpi-data-home "$OPENPI_HOME" --tokenizer-sha256 "$TOKENIZER_SHA")
  [[ -n "$QWENVL_ADAPTER" ]] && model+=(--qwenvl-groundsg-adapter "$QWENVL_ADAPTER")
else
  model=(--pp-ckpt "$PP_CKPT")
fi
# MemER adapter 原样转发（pp 误给时由两侧 variant_pairing 拒跑，不在这里吞掉）
[[ -n "$P_MEMER_ADAPTER" ]] && model+=(--memer-adapter "$P_MEMER_ADAPTER")
ORIG_CMD=(bash "$HERE/run_official_hard.sh" "${common[@]}" --policy "$POLICY" "${model[@]}" --infra-retry-budget "$ORIG_INFRA_BUDGET")
NEW_CMD=(bash "$HERE/run_eval_gl.sh" "${common[@]}" --policies "$POLICY" "${model[@]}" --cond "$COND"
         --reset-budget "$NEW_RESET_BUDGET" --infra-retry-budget "$NEW_INFRA_BUDGET")

wait_side_released() {  # $1 = 该侧的 .v8-pgids；进程组全退出且不在计算进程列表、端口段空闲才返回 0
  local f="$1" t0 role pid alive base p busy
  local -a pids=()
  [[ -f "$f" ]] && while read -r role pid; do [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid"); done < "$f"
  base=$((18000 + 100 * SEAT_IDX + 10 * $(pol_index "$POLICY")))
  t0=$(ts)
  while true; do
    alive=""
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null || \
         nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$pid"; then
        alive+=" $pid"
      fi
    done
    busy=""
    for p in $(seq "$base" $((base + 9))); do port_busy "$p" && busy+=" $p"; done
    if [[ -z "$alive" && -z "$busy" ]]; then
      echo "SIDE_RELEASED seat=$SEAT pgids=${#pids[@]} waited_s=$(( $(ts) - t0 )) ports_free=$base-$((base + 9))"
      return 0
    fi
    if (( $(ts) - t0 > RELEASE_WAIT )); then
      echo "RUN_BLOCKED reason=gpu_not_released seat=$SEAT alive=${alive:- none} busy_ports=${busy:- none}"
      return 1
    fi
    sleep 1 & wait $!
  done
}

run_side() {  # $1 = 侧名；其余为命令。前台等待（信号由 trap 转发给子进程）
  local side="$1" rc; shift
  echo "PAIR_SIDE_START seat=$SEAT side=$side policy=$LABEL $(date -Is)"
  "$@" &
  CHILD_PID=$!
  while true; do
    wait "$CHILD_PID"; rc=$?
    kill -0 "$CHILD_PID" 2>/dev/null || break
  done
  CHILD_PID=""
  echo "PAIR_SIDE_DONE seat=$SEAT side=$side policy=$LABEL rc=$rc $(date -Is)"
  return "$rc"
}

pair_done() {  # $1 = outcome；$2 = rc
  (( PAIR_DONE == 1 )) && return
  PAIR_DONE=1
  echo "PAIR_SEAT_DONE seat=$SEAT policy=$LABEL orig_rc=${ORIG_RC:-na} new_rc=${NEW_RC:-na} rc=$2 outcome=$1 $(date -Is)"
  echo "EXIT_CODE=$2"
  exit "$2"
}

pair_on_signal() {
  local r=143
  case "$1" in INT) r=130;; HUP) r=129;; esac
  trap '' TERM INT HUP
  echo "PAIR_SEAT_SIGNAL sig=$1 seat=$SEAT $(date -Is)"
  if [[ -n "$CHILD_PID" ]]; then
    kill -TERM "$CHILD_PID" 2>/dev/null
    wait "$CHILD_PID" 2>/dev/null
  fi
  pair_done aborted "$r"
}
trap 'pair_on_signal TERM' TERM
trap 'pair_on_signal INT' INT
trap 'pair_on_signal HUP' HUP
trap '' PIPE

exec > >(trap '' TERM INT HUP PIPE; exec tee -p -a "$SEAT_STAGE/pair_seat-s$SEAT-$LABEL.log") 2>&1
echo "PAIR_SEAT_START run_name=$RUN_NAME seat=$SEAT policy=$LABEL shard=$SHARD stage=$STAGE host=$(hostname) $(date -Is)"
step_cap_pairing || pair_done fail 3

run_side orig "${ORIG_CMD[@]}"; ORIG_RC=$?
if (( ORIG_RC == 4 || ORIG_RC == 5 || ORIG_RC > 128 )); then
  echo "CHAIN_STOP seat=$SEAT policy=$LABEL side=orig rc=$ORIG_RC（基础设施／额度／信号，不跑新侧）"
  pair_done fail "$ORIG_RC"
fi
if ! wait_side_released "$SEAT_STAGE/orig/.v8-pgids"; then
  echo "CHAIN_STOP seat=$SEAT policy=$LABEL reason=gpu_not_released"
  pair_done fail 4
fi
run_side new "${NEW_CMD[@]}"; NEW_RC=$?
wait_side_released "$SEAT_STAGE/.v8-pgids" || { (( NEW_RC == 0 )) && NEW_RC=4; }
if (( NEW_RC == 4 || NEW_RC == 5 || NEW_RC > 128 )); then
  echo "CHAIN_STOP seat=$SEAT policy=$LABEL side=new rc=$NEW_RC"
fi
# 基础设施类（4／5／信号）优先上报，供外层链判断是否 CHAIN_STOP；否则取首个非零
rc=0
for r in "$ORIG_RC" "$NEW_RC"; do
  if (( r == 4 || r == 5 || r > 128 )); then rc=$r; break; fi
done
if (( rc == 0 )); then
  (( ORIG_RC != 0 )) && rc=$ORIG_RC
  (( rc == 0 && NEW_RC != 0 )) && rc=$NEW_RC
fi
if (( rc == 0 )); then pair_done pass 0; else pair_done fail "$rc"; fi
