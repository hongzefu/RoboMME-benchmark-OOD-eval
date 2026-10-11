#!/usr/bin/env bash
# 五模型 OOD 最简四卡方案的只读巡检（计划第一部分 §十四）。用法：bash dev-scripts/gl/simple/simple_check.sh
# 只读：一次 ssh（经 cm2 主连接）查 hostname、作业步、作业到期时间、登录节点 tmux；本机直读 NFS 挂载上的日志与 result.json。
# 不改任何文件、不提交/取消 job、不起/杀任何进程。
# 每分片一行 SHARD，五行模型计数，一行 infra 计数，每分片一行外推；末行 CHECK=OK|RESTART gN..|BLOCKED gN..|DONE|OVERRUN gN..
# 环境变量覆盖（换 job 时用）：SIMPLE_JOBS="g1:<id> g2:<id> g3:<id> g4:<id>"、SIMPLE_ROOT=<NFS 运行根>、
#   SIMPLE_STALE_S（无作业步时判停更的秒数，默认 600）、SIMPLE_ALIVE_STALE_S（有作业步但日志久不更新的告警秒数，默认 3600，
#   须大于脚本无进展看门 2700 s + 30 s）、SIMPLE_SEC_<模型>（外推用单局秒数，默认取计划 §八 的 02 轮实测／估算）。
# restarts.log 由巡检代理在每次重起时追加一行，行内必须含 `group=gN `（本脚本按它计重起次数，≥3 次不再判 RESTART）。
set -uo pipefail
N="${SIMPLE_ROOT:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02}"
JOBS="${SIMPLE_JOBS:-g1:63664391 g2:63664392 g3:63664393 g4:63664404}"
STALE_S="${SIMPLE_STALE_S:-600}"; ALIVE_STALE_S="${SIMPLE_ALIVE_STALE_S:-3600}"
SSH=(ssh -o "ControlPath=/home/hongzefu/.ssh/cm2-%r@%h:%p" -o ControlMaster=no -o BatchMode=yes greatlakes)
LABELS=(groundsg-ground-sg-oracle perceptual-framesamp-modul pp smvla groundsg-ground-sg-qwenvl)
declare -A SEC=([groundsg-ground-sg-oracle]=${SIMPLE_SEC_oracle:-130} [perceptual-framesamp-modul]=${SIMPLE_SEC_framesamp:-115}
  [pp]=${SIMPLE_SEC_pp:-286} [smvla]=${SIMPLE_SEC_smvla:-335} [groundsg-ground-sg-qwenvl]=${SIMPLE_SEC_qwenvl:-581})
declare -A TASKS=([g1]="PickHighlight SwingXtimes StopCube VideoUnmaskSwap" [g2]="BinFill VideoUnmask ButtonUnmaskSwap VideoRepick"
  [g3]="MoveCube PickXtimes RouteStick VideoPlaceButton" [g4]="InsertPeg ButtonUnmask PatternLock VideoPlaceOrder")  # 与 run_shard.sh 的 case 表一致
IDS=$(for p in $JOBS; do printf '%s,' "${p#*:}"; done); IDS=${IDS%,}
REMOTE=$("${SSH[@]}" "hostname; echo '##STEPS'; squeue -j $IDS --steps -o '%i %j' --noheader 2>&1; echo '##END'; squeue -j $IDS -o '%i %e' --noheader 2>&1; echo '##TMUX'; tmux ls -F '#{session_name}' 2>/dev/null; true")
SRC=$?; HOST=$(sed -n 1p <<<"$REMOTE")
echo "HOST=$HOST ssh_rc=$SRC"
[[ $SRC -eq 0 && "$HOST" == gl-login1* ]] || { echo "CHECK=BLOCKED host (ssh_rc=$SRC host=${HOST:-none}，须经 cm2 落在 gl-login1)"; exit 1; }
section() { awk -v s="##$1" '/^##/{on=($0==s); next} on' <<<"$REMOTE"; }
NOW=$(date +%s); declare -a RESTART=() BLOCKED=() OVERRUN=(); NDONE=0
for p in $JOBS; do
  g=${p%%:*}; job=${p#*:}; L=$N/simple/logs/shard-$g.log
  step=无; grep -q "^$job\.[0-9]* ood-simple-$g\$" <<<"$(section STEPS)" && step=有
  tmx=无; grep -qx "ev-ood-simple-$g" <<<"$(section TMUX)" && tmx=有
  end=$(awk -v j="$job" '$1==j{print $2}' <<<"$(section END)"); end_s=$([[ -n "$end" ]] && date -d "$end" +%s 2>/dev/null || echo "")
  if [[ -f "$L" ]]; then age=$(( NOW - $(stat -c %Y "$L") )); tail1=$(tail -n 1 "$L")
    run=$(awk '/^SHARD_START /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$L")   # 最近一次起跑之后的日志
    last=$(grep -E '^(EXIT_CODE=|SHARD_START|SHARD_DONE|SHARD_SIGNAL|MODEL_START|MODEL_DONE|RUN_BLOCKED|NOPROG|REATTEMPT)' <<<"$run" | tail -n 1 | cut -d' ' -f1-4 | tr ' ' '_')
  else age=-1; tail1=""; run=""; last=无日志; fi
  sexit=$(grep -o 'SRUN_EXIT=[0-9]*' "$N/simple/logs/srun-$g.log" 2>/dev/null | tail -n 1); sexit=${sexit#SRUN_EXIT=}
  reat=0; [[ -f "$N/simple/logs/reattempts-$g.log" ]] && reat=$(wc -l < "$N/simple/logs/reattempts-$g.log")
  rest=$(grep -cE "group=$g( |$)" "$N/simple/logs/restarts.log" 2>/dev/null); rest=${rest:-0}
  echo "SHARD $g step=$step tmux=$tmx last=$last log_age_s=$age srun_exit=${sexit:--} reattempts=$reat restarts=$rest job=$job end=${end:-gone}"
  # 判定（§十四 决策表）
  if [[ "$tail1" == "EXIT_CODE=0" && $step == 无 ]]; then NDONE=$((NDONE + 1)); continue; fi
  if grep -qE '^RUN_BLOCKED reason=(infra_cascade|reattempt_cap)' <<<"$run" || [[ "$tail1" =~ ^EXIT_CODE=(2|3|4)$ ]]; then BLOCKED+=("$g"); continue; fi
  if [[ $step == 无 ]]; then
    if [[ "$tail1" =~ ^EXIT_CODE=(1|143)$ || $age -lt 0 || $age -ge $STALE_S ]]; then
      if (( rest < 3 )); then RESTART+=("$g"); else BLOCKED+=("$g(restarts=$rest)"); fi
    else echo "NOTE $g 无作业步但日志 ${age}s 内有更新且未见退出码，等下一次巡检（防双开）"; fi
    continue
  fi
  (( age > ALIVE_STALE_S )) && { BLOCKED+=("$g(stale=${age}s)"); continue; }
  # 外推：本分片各模型剩余局数 × 单局秒数，串行相加
  rem=0; for lab in "${LABELS[@]}"; do d=0
    for t in ${TASKS[$g]}; do d=$(( d + $(find "$N/gl/rollouts/$lab/ood/seed7/raw" -maxdepth 2 -path "*/${t}_ep*/result.json" 2>/dev/null | wc -l) )); done
    rem=$(( rem + (200 - d) * ${SEC[$lab]} )); done
  eta=$(( NOW + rem )); echo "ETA $g remaining_h=$(( rem / 3600 )) eta=$(date -d "@$eta" -Is) job_end=${end:-gone}"
  [[ -n "$end_s" ]] && (( eta > end_s - 4 * 3600 )) && OVERRUN+=("$g")
done
infra=0; for lab in "${LABELS[@]}"; do R=$N/gl/rollouts/$lab/ood/seed7/raw
  c=$(find "$R" -maxdepth 2 -name result.json 2>/dev/null | wc -l)
  i=$(find "$R" -maxdepth 2 -name result.json -exec grep -l '"infra": true' {} + 2>/dev/null | wc -l); infra=$(( infra + i ))
  echo "MODEL $lab results=$c/800 infra=$i"; done
echo "INFRA_TOTAL=$infra"
if (( NDONE == 4 )); then echo "CHECK=DONE"
elif (( ${#BLOCKED[@]} )); then echo "CHECK=BLOCKED ${BLOCKED[*]}${RESTART:+ also_restart=${RESTART[*]}}"
elif (( ${#RESTART[@]} )); then echo "CHECK=RESTART ${RESTART[*]}${OVERRUN:+ also_overrun=${OVERRUN[*]}}"
elif (( ${#OVERRUN[@]} )); then echo "CHECK=OVERRUN ${OVERRUN[*]}"
else echo "CHECK=OK done=$NDONE"; fi
