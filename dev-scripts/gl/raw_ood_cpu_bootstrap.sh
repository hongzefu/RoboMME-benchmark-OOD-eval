#!/usr/bin/env bash
# CPU持续占位：监督器只起一次；失败、正常结束和准备超时都不取消GPU。
set -uo pipefail
ROOT=${1:?运行根}; RUN=${2:?运行名}; CPU=${3:-${SLURM_JOB_ID:?}}; DEADLINE=${4:?准备截止秒}
: "${BENCH_PY:?必须提供uv管理解释器}"
export PYTHONUNBUFFERED=1
HOLD_PID=""; PHASE=prepare; LAUNCH=()
if [[ -d "$ROOT/control" ]]; then exec > >(tee -a "$ROOT/control/bootstrap-$CPU.log") 2>&1; fi
finish() {
  local rc=$?
  trap - EXIT
  [[ -z "$HOLD_PID" ]] || kill "$HOLD_PID" 2>/dev/null || true
  echo "CPU_BOOTSTRAP_EXIT rc=$rc gpu_action=none notification_established=0"
  echo "EXIT_CODE=$rc"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
prepare_and_run() {
  [[ "$CPU" =~ ^[0-9]+$ && "$DEADLINE" =~ ^[0-9]+$ && "$RUN" =~ ^[A-Za-z0-9._-]+$ && -d "$ROOT/control" && ! -L "$ROOT" ]] || return 2
  (( ${SLURM_RESTART_COUNT:-0} == 0 )) || return 3
  while [[ ! -f "$ROOT/control/launch.ready" ]]; do
    [[ ! -f "$ROOT/control/STOP" ]] || return 4
    (( $(date +%s) < DEADLINE )) || return 5
    sleep 5
  done
  PHASE=ready_validate
  mapfile -t LAUNCH < <("$BENCH_PY" -I -S -u - "$ROOT" "$RUN" "$CPU" <<'PY'
import hashlib,json,pathlib,re,sys
root,run,cpu=sys.argv[1:]
r=json.loads((pathlib.Path(root)/'control/launch.ready').read_text())
assert r['root']==root and r['run']==run and str(r['controller_job_id'])==cpu
assert re.fullmatch('[0-9a-f]{40}',r['exec_commit'])
c=pathlib.Path(r['config_path']); assert c.is_relative_to(pathlib.Path(root))
assert hashlib.sha256(c.read_bytes()).hexdigest()==r['config_sha256']
j=json.loads((pathlib.Path(root)/'control/jobs.json').read_text())
assert j['run']==run and str(j['controller_job_id'])==cpu
ids=j['gpu_job_ids']; assert len(ids)==4 and len(set(map(str,ids)))==4
assert all(str(i).isdigit() and str(i)!=cpu for i in ids)
for k in ('python','supervisor','config_path','exec_commit','config_sha256'): print(r[k])
PY
)
  [[ ${#LAUNCH[@]} == 5 && -x "${LAUNCH[0]}" && -f "${LAUNCH[1]}" ]] || return 6
  PHASE=supervisor_exit
  "${LAUNCH[0]}" "${LAUNCH[1]}" run --config "${LAUNCH[2]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
}
prepare_and_run; OUTCOME=$?
echo "CPU_SUPERVISOR_OUTCOME phase=$PHASE supervisor_exit_code=$OUTCOME cpu_hold=1 gpu_action=none notification_established=0"
if [[ -d "$ROOT/control" && "$CPU" =~ ^[0-9]+$ && "$RUN" =~ ^[A-Za-z0-9._-]+$ ]]; then
  outcome="$ROOT/control/cpu-bootstrap-outcome.json"
  printf '{"run":"%s","run_name":"%s","controller_job_id":"%s","phase":"%s","supervisor_exit_code":%s,"cpu_hold":true,"gpu_action":"none","notification_established":false,"time":%s}\n' \
    "$RUN" "$RUN" "$CPU" "$PHASE" "$OUTCOME" "$(date +%s)" > "$outcome.$$.tmp"
  mv "$outcome.$$.tmp" "$outcome"
  if [[ ${#LAUNCH[@]} == 5 && ! -f "$ROOT/control/intentional-supervisor-stop.json" ]]; then
    intent="$ROOT/control/intentional-supervisor-stop.json"
    printf '{"run":"%s","run_name":"%s","controller_job_id":"%s","exec_commit":"%s","config_sha256":"%s","supervisor_exit_code":%s,"reason":"supervisor_exit_cpu_hold","cpu_hold":true,"gpu_action":"none","notification_established":false,"time":%s}\n' \
      "$RUN" "$RUN" "$CPU" "${LAUNCH[3]}" "${LAUNCH[4]}" "$OUTCOME" "$(date +%s)" > "$intent.$$.tmp"
    mv "$intent.$$.tmp" "$intent"
  fi
fi
# 不重试监督器；allocation仅由120h期限或用户明确终止释放。
while :; do sleep 3600 & HOLD_PID=$!; wait "$HOLD_PID" || true; done
