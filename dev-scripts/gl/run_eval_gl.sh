#!/usr/bin/env bash
# GL 评估席位外壳（1008 拆分方案第二部分 §二 dev-scripts/gl/ 行、runbook S5）：在占位 job 内由
#   srun --jobid=<占位> --overlap --exact --ntasks=1 --gpu_cmode=shared bash dev-scripts/gl/run_eval_gl.sh …
# 调用，逐个策略起 dev-scripts/gl/seat.py run（常驻 Policy + 共享动态队列），并吸收旧 run_seat.sh 剩下的两件事：
#   1. 客户端重起：客户端以 75（单局墙钟／阶段期限，已记录）或被无进展看门狗杀掉退出时，按 --client-restarts 重起
#      （本轮缺省 0：不重起，停下交用户）；
#   2. 无进展看门狗：每 --poll-s 秒看一次本策略的 progress.json（席位 seats/<标签>/seed<n>/<席位>/progress.json 与
#      rollouts/<模型>/<数据集>/seed<n>/progress.json，取最新修改时间），超过 --noprog-s 秒没有任何更新即先 TERM（客户端
#      收到 TERM 会经 policy.close() 停服务端）、等 --term-grace-s 再 KILL，计一次无进展（rc=124）。
# 服务端由 Python 的 ServerProcess 起停（本脚本不起服务端）；客户端非零退出后，按客户端日志里的 SERVER_LEFT 行与
# SERVER_START 行定位 server-metadata-<port>.json，用 scripts/evaluate.py --stop-server 停掉残留服务端（只停元数据里
# cmdline 一致的进程组）。
#
# 用法（runbook S5 的写法，其余参数有缺省）：
#   bash dev-scripts/gl/run_eval_gl.sh --policies <模型>[,<模型>…] [--groundsg-variant ground-sg-oracle] \
#     --policy-seed 7 --identities <身份清单 jsonl> --budget-ledger <NFS 上本轮账本> --run <run_name> \
#     [--out <产物根，缺省 <repo>/artifacts/<run_name>/gl>] [--seat <席位名>] [--gpus 0] \
#     [--trajectory-cap 26 --reset-cap 60 --shared-infra-cap 0 --expired-cap 0 --planned-first-tries 26] \
#     [--infra-retries 0] [--client-restarts 0] [--noprog-s 2700] [--poll-s 30] [--term-grace-s 90] \
#     [--qwenvl-groundsg-adapter D] [--memer-adapter D] [其余 --<名> <值> 原样透传给 seat.py（模型参数）]
# ckpt（1009 拆分计划：评估包不再内置缺省 ckpt）：透传参数里没有 --ckpt（或 --cfg ckpt=…）时，按策略从
# dev-scripts/gl/ckpt_paths.sh 取缺省路径（`bash ckpt_paths.sh <模型>`，可用 FRAMESAMP_MODUL_CKPT／GROUNDSG_CKPT／
# SMVLA_CKPT 覆盖）显式传 --ckpt，并打印 GL_CKPT 行；这三个模型取不到路径即 RUN_INPUTS=FAIL reason=ckpt_missing_<模型>。
# 没有缺省的模型（pp 等）不补，仍须调用方自己给 --ckpt。
# 预算缺省值即拆分方案 §五预算表（身份执行 26、计量 60 两个硬上限，基础设施重试 0）；共享账本 config 行会与之比对，
# 本机 Astra 等其他入口必须给同一组上限。
# 解释器：BENCH_PY（缺省 <repo>/.venv/bin/python）；起跑前打印 RUN_INPUTS 行，robomme_ood_eval 不在 <repo>/src 下、
# robomme_ood 导入失败、身份清单不存在即 RUN_BLOCKED reason=run_inputs（退出 3）。
# 末尾三行：每个策略 GL_POLICY_DONE policy= rc= restarts= noprog=，GL_SEAT_DONE outcome=pass|fail|aborted rc=，
# EXIT_CODE=<rc>。退出码：0 全部策略 rc=0；参数错 2；输入不全／运行阻塞 3；其余取首个非零策略 rc（124 = 无进展）；
# 中断 130/143（HUP 129）。不嵌入任何 JobID，不含 /data 默认路径。
set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

POLICIES="" ; GROUNDSG_VARIANT="" ; POLICY_SEED="" ; IDENTITIES="" ; BUDGET_LEDGER="" ; RUN_NAME="" ; OUT="" ; SEAT=""
GPUS="0" ; TRAJECTORY_CAP=26 ; RESET_CAP=60 ; SHARED_INFRA_CAP=0 ; EXPIRED_CAP=0 ; PLANNED_FIRST_TRIES=26
INFRA_RETRIES=0 ; CLIENT_RESTARTS=0 ; NOPROG_S=2700 ; POLL_S=30 ; TERM_GRACE_S=90
PASS=()

usage() { sed -n '2,/^# 中断 130/p' "${BASH_SOURCE[0]}" >&2; }
die2() {
  echo "$1" >&2
  echo "GL_SEAT_DONE outcome=fail rc=2 reason=bad_args"
  echo "EXIT_CODE=2"
  exit 2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policies) POLICIES="$2"; shift 2;;
    --groundsg-variant) GROUNDSG_VARIANT="$2"; shift 2;;
    --policy-seed) POLICY_SEED="$2"; shift 2;;
    --identities) IDENTITIES="$2"; shift 2;;
    --budget-ledger) BUDGET_LEDGER="$2"; shift 2;;
    --run) RUN_NAME="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --seat) SEAT="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --trajectory-cap) TRAJECTORY_CAP="$2"; shift 2;;
    --reset-cap) RESET_CAP="$2"; shift 2;;
    --shared-infra-cap) SHARED_INFRA_CAP="$2"; shift 2;;
    --expired-cap) EXPIRED_CAP="$2"; shift 2;;
    --planned-first-tries) PLANNED_FIRST_TRIES="$2"; shift 2;;
    --infra-retries) INFRA_RETRIES="$2"; shift 2;;
    --client-restarts) CLIENT_RESTARTS="$2"; shift 2;;
    --noprog-s) NOPROG_S="$2"; shift 2;;
    --poll-s) POLL_S="$2"; shift 2;;
    --term-grace-s) TERM_GRACE_S="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    --*=*) PASS+=("$1"); shift;;
    --*) if [[ $# -ge 2 && "$2" != --* ]]; then PASS+=("$1" "$2"); shift 2; else PASS+=("$1"); shift; fi;;
    *) usage; die2 "无法解析的参数 $1";;
  esac
done
[[ -n "$POLICIES" && -n "$POLICY_SEED" && -n "$IDENTITIES" && -n "$BUDGET_LEDGER" && -n "$RUN_NAME" ]] \
  || die2 "缺少必需参数（--policies --policy-seed --identities --budget-ledger --run）"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die2 "--run 只许字母数字 . _ -"
for _v in "$POLICY_SEED" "$TRAJECTORY_CAP" "$RESET_CAP" "$SHARED_INFRA_CAP" "$EXPIRED_CAP" "$PLANNED_FIRST_TRIES" \
          "$INFRA_RETRIES" "$CLIENT_RESTARTS" "$NOPROG_S" "$POLL_S" "$TERM_GRACE_S"; do
  [[ "$_v" =~ ^[0-9]+$ ]] || die2 "种子、预算、重起次数与秒数须为非负整数（得到 $_v）"
done
(( POLL_S >= 1 )) || die2 "--poll-s 至少 1"
IFS=',' read -r -a POLS <<< "$POLICIES"
for pol in "${POLS[@]}"; do
  [[ "$pol" =~ ^[a-z0-9-]+$ ]] || die2 "未知策略名 $pol"
  if [[ "$pol" == "groundsg" && -z "$GROUNDSG_VARIANT" ]]; then die2 "--policies groundsg 须给 --groundsg-variant"; fi
done
[[ -n "$OUT" ]] || OUT="$REPO/artifacts/$RUN_NAME/gl"
BENCH_PY="${BENCH_PY:-$REPO/.venv/bin/python}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
for f in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  [[ -f "$f" ]] && { export VK_ICD_FILENAMES="$f"; break; }
done

ts_iso() { date -Is; }
CLIENT_PID="" ; FINALIZED=0 ; overall=0 ; CUR_LOG=""

label_of() { if [[ "$1" == "groundsg" ]]; then echo "groundsg-$GROUNDSG_VARIANT"; else echo "$1"; fi; }

newest_progress() {  # $1 = 标签；打印本策略各 progress.json 的最新修改时间（epoch 秒，取整），没有打印 0
  local lab="$1" t
  t="$( { find "$OUT/seats/$lab" -name progress.json -printf '%T@\n' 2>/dev/null
          find "$OUT/rollouts" -path "*/seed$POLICY_SEED/progress.json" -printf '%T@\n' 2>/dev/null; } \
        | sort -n | tail -n 1)"
  echo "${t%%.*}"
}

stop_left_servers() {  # $1 = 客户端日志：按 SERVER_LEFT／SERVER_START 行定位元数据，停残留服务端
  local log="$1" meta
  [[ -f "$log" ]] || return 0
  while IFS= read -r meta; do
    [[ -n "$meta" && -f "$meta" ]] || continue
    echo "GL_STOP_SERVER metadata=$meta"
    ( cd "$REPO" && "$BENCH_PY" scripts/evaluate.py --stop-server "$meta" ) || echo "GL_STOP_SERVER_FAIL metadata=$meta"
  done < <("$BENCH_PY" - "$log" <<'PY'
import os, re, sys
seen = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.search(r"SERVER_LEFT .*?metadata=(\S+)", line)
    if m:
        seen.append(m.group(1))
        continue
    m = re.search(r"SERVER_START .*?port=(\d+) .*?log=(\S+)", line)
    if m:
        seen.append(os.path.join(os.path.dirname(m.group(2)), f"server-metadata-{m.group(1)}.json"))
for p in dict.fromkeys(seen):
    print(p)
PY
)
}

kill_client() {  # TERM 整个客户端进程组，等 TERM_GRACE_S，再 KILL
  [[ -n "$CLIENT_PID" ]] || return 0
  kill -0 "$CLIENT_PID" 2>/dev/null || return 0
  kill -TERM -- "-$CLIENT_PID" 2>/dev/null || kill -TERM "$CLIENT_PID" 2>/dev/null
  local i
  for i in $(seq 1 $((TERM_GRACE_S * 2))); do kill -0 "$CLIENT_PID" 2>/dev/null || break; sleep 0.5; done
  if kill -0 "$CLIENT_PID" 2>/dev/null; then
    echo "GL_CLIENT_KILL pid=$CLIENT_PID（${TERM_GRACE_S}s 未退出）"
    kill -KILL -- "-$CLIENT_PID" 2>/dev/null || kill -KILL "$CLIENT_PID" 2>/dev/null
  fi
}

finalize() {  # $1 = outcome；$2 = rc
  local outcome="$1" rc="$2"
  (( FINALIZED == 1 )) && return
  FINALIZED=1
  trap '' TERM INT HUP
  kill_client
  [[ -n "$CUR_LOG" ]] && stop_left_servers "$CUR_LOG"
  echo "GL_SEAT_DONE outcome=$outcome rc=$rc run=$RUN_NAME policies=$POLICIES host=$(hostname) $(ts_iso)"
  echo "EXIT_CODE=$rc"
  exit "$rc"
}

on_signal() {
  local rc=143
  case "$1" in INT) rc=130;; HUP) rc=129;; esac
  echo "GL_SEAT_SIGNAL sig=$1 $(ts_iso)"
  finalize aborted "$rc"
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
trap '' PIPE

if ! mkdir -p "$OUT/logs"; then
  echo "RUN_BLOCKED reason=mkdir out=$OUT"
  finalize fail 3
fi
GL_LOG="$OUT/logs/run_eval_gl-$(hostname)-$$.log"
exec > >(trap '' TERM INT HUP PIPE; exec tee -p -a "$GL_LOG") 2>&1

# ---------------- 起跑输入核对（RUN_INPUTS） ----------------
why=()
[[ -x "$BENCH_PY" ]] || why+=("bench_py_missing")
[[ -f "$IDENTITIES" ]] || why+=("identities_missing")
[[ -f "$REPO/dev-scripts/gl/seat.py" ]] || why+=("seat_missing")
pkg="$( cd "$REPO" && "$BENCH_PY" -c 'import robomme_ood_eval, robomme_ood; print(robomme_ood_eval.__file__, robomme_ood.__file__)' 2>/dev/null | tail -n 1)"
eval_file="${pkg%% *}" ; hard_file="${pkg##* }"
[[ "$eval_file" == "$REPO/src/robomme_ood_eval/__init__.py" ]] || why+=("robomme_ood_eval_not_in_repo")
[[ "$hard_file" == */robomme_ood/__init__.py ]] || why+=("robomme_ood_import")
# ckpt：调用方已给（--ckpt V、--ckpt=V、--cfg ckpt=V）则原样透传；否则按策略取 ckpt_paths.sh 的缺省路径
PASS_HAS_CKPT=0
for ((_i = 0; _i < ${#PASS[@]}; _i++)); do
  case "${PASS[$_i]}" in
    --ckpt|--ckpt=*) PASS_HAS_CKPT=1;;
    --cfg) [[ "${PASS[$((_i + 1))]:-}" == ckpt=* ]] && PASS_HAS_CKPT=1;;
    --cfg=ckpt=*) PASS_HAS_CKPT=1;;
  esac
done
declare -A POL_CKPT=()
if (( PASS_HAS_CKPT == 0 )); then
  for pol in "${POLS[@]}"; do
    case "$pol" in
      perceptual-framesamp-modul|groundsg|smvla)
        POL_CKPT[$pol]="$(bash "$HERE/ckpt_paths.sh" "$pol" 2>/dev/null)"
        [[ -n "${POL_CKPT[$pol]}" ]] || why+=("ckpt_missing_$pol");;
    esac
  done
fi
CPUS="$("$BENCH_PY" -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0)))))' 2>/dev/null)"
verdict=PASS; (( ${#why[@]} == 0 )) || verdict=FAIL
echo "RUN_INPUTS=$verdict repo=$REPO bench_py=$BENCH_PY robomme_ood_eval=${eval_file:-IMPORT_FAIL} \
robomme_ood=${hard_file:-IMPORT_FAIL} identities=$IDENTITIES out=$OUT host=$(hostname) cpus=${CPUS:-?} \
slurm_job=${SLURM_JOB_ID:-none} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}${why:+ reason=$(IFS=,; echo "${why[*]}")}"
if [[ "$verdict" != PASS ]]; then
  echo "RUN_BLOCKED reason=run_inputs（见上一行 RUN_INPUTS=FAIL 的 reason）"
  finalize fail 3
fi
echo "GL_SEAT_START run=$RUN_NAME policies=$POLICIES variant=${GROUNDSG_VARIANT:-none} policy_seed=$POLICY_SEED \
budget_ledger=$BUDGET_LEDGER caps=$TRAJECTORY_CAP/$RESET_CAP/$SHARED_INFRA_CAP/$EXPIRED_CAP/$PLANNED_FIRST_TRIES \
infra_retries=$INFRA_RETRIES client_restarts=$CLIENT_RESTARTS noprog_s=$NOPROG_S poll_s=$POLL_S gpus=$GPUS \
seat=${SEAT:-auto} passthrough=${PASS[*]:-none} $(ts_iso)"
for pol in "${POLS[@]}"; do
  if (( PASS_HAS_CKPT == 1 )); then
    echo "GL_CKPT policy=$pol source=passthrough"
  elif [[ -n "${POL_CKPT[$pol]:-}" ]]; then
    echo "GL_CKPT policy=$pol source=ckpt_paths.sh ckpt=${POL_CKPT[$pol]}"
  else
    echo "GL_CKPT policy=$pol source=none（该模型无缺省 ckpt，须调用方给 --ckpt）"
  fi
done

# ---------------- 主流程 ----------------
for pol in "${POLS[@]}"; do
  lab="$(label_of "$pol")"
  restarts=0 ; noprog=0 ; rc=0
  while true; do
    args=(run --policy "$pol" --policy-seed "$POLICY_SEED" --identities "$IDENTITIES" --out "$OUT"
          --budget-ledger "$BUDGET_LEDGER" --trajectory-cap "$TRAJECTORY_CAP" --reset-cap "$RESET_CAP"
          --shared-infra-cap "$SHARED_INFRA_CAP" --expired-cap "$EXPIRED_CAP"
          --planned-first-tries "$PLANNED_FIRST_TRIES" --infra-retries "$INFRA_RETRIES" --gpus "$GPUS")
    [[ "$pol" == "groundsg" ]] && args+=(--groundsg-variant "$GROUNDSG_VARIANT")
    [[ -n "$SEAT" ]] && args+=(--seat "$SEAT")
    [[ -n "${POL_CKPT[$pol]:-}" ]] && args+=(--ckpt "${POL_CKPT[$pol]}")
    args+=("${PASS[@]}")
    CUR_LOG="$OUT/logs/client-$lab-$(hostname)-$$-r$restarts.log"
    echo "GL_CLIENT_START policy=$lab restart=$restarts log=$CUR_LOG $(ts_iso)"
    ( cd "$REPO" && exec setsid "$BENCH_PY" dev-scripts/gl/seat.py "${args[@]}" ) \
      > >(trap '' TERM INT HUP PIPE; exec tee -p -a "$CUR_LOG") 2>&1 &
    CLIENT_PID=$!
    start_t=$(date +%s)
    killed=0
    while kill -0 "$CLIENT_PID" 2>/dev/null; do
      sleep "$POLL_S" & wait $! 2>/dev/null
      kill -0 "$CLIENT_PID" 2>/dev/null || break
      last="$(newest_progress "$lab")"; (( last > start_t )) || last=$start_t
      idle=$(( $(date +%s) - last ))
      if (( idle > NOPROG_S )); then
        echo "GL_NOPROGRESS policy=$lab idle_s=$idle limit_s=$NOPROG_S pid=$CLIENT_PID $(ts_iso)"
        kill_client
        killed=1
        break
      fi
    done
    wait "$CLIENT_PID" 2>/dev/null; rc=$?
    CLIENT_PID=""
    if (( killed == 1 )); then rc=124; noprog=$((noprog + 1)); fi
    echo "GL_CLIENT_EXIT policy=$lab rc=$rc restart=$restarts $(ts_iso)"
    if (( rc != 0 )); then stop_left_servers "$CUR_LOG"; fi
    if (( rc == 75 || rc == 124 )) && (( restarts < CLIENT_RESTARTS )); then
      restarts=$((restarts + 1))
      echo "GL_CLIENT_RESTART policy=$lab restart=$restarts/$CLIENT_RESTARTS reason=$([[ $rc == 124 ]] && echo noprogress || echo watchdog_exit)"
      continue
    fi
    break
  done
  echo "GL_POLICY_DONE policy=$lab rc=$rc restarts=$restarts noprog=$noprog $(ts_iso)"
  (( rc != 0 && overall == 0 )) && overall=$rc
done

if (( overall == 0 )); then finalize pass 0; else finalize fail "$overall"; fi
