#!/usr/bin/env bash
# 本轮GPU独立守卫：CPU消失不能被尚未发布ready掩盖。
set -euo pipefail
ROOT=${1:?运行根}; RUN=${2:?运行名}; CPU=${3:?CPU作业}; DEADLINE=${4:?准备截止秒}
SELF=${SLURM_JOB_ID:?}; : "${BENCH_PY:?必须提供uv管理解释器}"
[[ "$CPU" =~ ^[0-9]+$ && "$SELF" =~ ^[0-9]+$ && "$DEADLINE" =~ ^[0-9]+$ ]] || exit 2
exec > >(tee -a "$ROOT/control/guard-$SELF.log") 2>&1
normal_complete() {
  [[ -f "$ROOT/control/gpu_work_complete.json" ]] || return 1
  "$BENCH_PY" - "$ROOT" "$RUN" "$CPU" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'control'; r=json.loads((p/'launch.ready').read_text()); d=json.loads((p/'gpu_work_complete.json').read_text())
assert r['root']==sys.argv[1] and r['run']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
assert d['run']==r['run'] and str(d['controller_job_id'])==sys.argv[3]
assert d['config_sha256']==r['config_sha256'] and d['exec_commit']==r['exec_commit']
PY
}
cleanup() {
  local rc=$?
  trap - EXIT
  if normal_complete; then rc=0; fi
  if (( rc != 0 )); then
    printf 'guard_failure gpu=%s cpu=%s rc=%s\n' "$SELF" "$CPU" "$rc" > "$ROOT/control/STOP"
    # 文件锁由进程持有，死亡自动释放，幸存守卫可接续；自身最后取消。
    exec 9>"$ROOT/control/cleanup.lock"
    if flock -w 5 9; then
      if [[ -f "$ROOT/control/gpu-job-ids.txt" ]]; then
        while IFS= read -r id; do [[ "$id" =~ ^[0-9]+$ && "$id" != "$CPU" && "$id" != "$SELF" ]] && scancel "$id" || true; done < "$ROOT/control/gpu-job-ids.txt"
      fi
      scancel "$SELF" || true
    fi
  fi
  echo "EXIT_CODE=$rc"; exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
(( ${SLURM_RESTART_COUNT:-0} == 0 )) || exit 3
QUERY_FAIL=0
while :; do
  if [[ -f "$ROOT/control/gpu_work_complete.json" ]]; then
    normal_complete && exit 0
    exit 4
  fi
  [[ ! -f "$ROOT/control/STOP" ]] || exit 5
  if STATE=$(squeue -h -j "$CPU" -o '%T' 2>/dev/null); then
    QUERY_FAIL=0
    [[ "$STATE" == PENDING || "$STATE" == CONFIGURING || "$STATE" == RUNNING ]] || exit 6
  else
    (( QUERY_FAIL += 10 )); (( QUERY_FAIL < 30 )) || exit 7
  fi
  if [[ -f "$ROOT/control/launch.ready" ]]; then
    "$BENCH_PY" - "$ROOT" "$RUN" "$CPU" <<'PY' || exit 8
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1])/'control'; r=json.loads((p/'launch.ready').read_text())
assert r['run']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
h=p/'cpu-heartbeat.json'
if h.exists():
 d=json.loads(h.read_text()); assert d['run']==r['run'] and str(d['controller_job_id'])==str(r['controller_job_id']) and d['config_sha256']==r['config_sha256']; assert time.time()-d['t']<120
else: assert time.time()-(p/'launch.ready').stat().st_mtime<120
PY
  else
    (( $(date +%s) < DEADLINE )) || exit 9
  fi
  sleep 10
done
