#!/bin/bash
# 同CPU步骤只运行一次监督器，结束后继续占位，避免Slurm回收仍存活的控制器。
set -uo pipefail
ROOT=${1:?必须提供运行根}; shift
[[ "$ROOT" = /* && -d "$ROOT/control" && ! -L "$ROOT" && ! -L "$ROOT/control" ]] || exit 2
(( $# > 0 )) || exit 2
trap 'echo "SUPERVISOR_HOLD_MANUAL_TERM pid=$$"; exit 0' TERM INT
"$@" &
SUPERVISOR_PID=$!
wait "$SUPERVISOR_PID"
SUPERVISOR_RC=$?
record="$ROOT/control/supervisor-hold-$$.json"
printf '{"component":"supervisor_hold","wrapper_pid":%s,"supervisor_pid":%s,"supervisor_rc":%s,"time":%s,"hold":true,"restarts":0,"child_signals":0,"gpu_cancellations":0}\n' \
  "$$" "$SUPERVISOR_PID" "$SUPERVISOR_RC" "$(date +%s)" > "$record.tmp"
mv "$record.tmp" "$record"
echo "SUPERVISOR_EXIT_CODE=$SUPERVISOR_RC HOLD_PID=$$"
# 不重启监督器，不向其遗留控制器、守卫或其它作业传递信号。
while :; do sleep 60 & wait "$!"; done
