#!/usr/bin/env bash
# 本轮GPU独立守卫：CPU消失不能被尚未发布ready掩盖。
set -euo pipefail
ROOT=${1:?运行根}; RUN=${2:?运行名}; CPU=${3:?CPU作业}; DEADLINE=${4:?准备截止秒}
SELF=${SLURM_JOB_ID:?}; : "${BENCH_PY:?必须提供uv管理解释器}"
[[ "$CPU" =~ ^[0-9]+$ && "$SELF" =~ ^[0-9]+$ && "$DEADLINE" =~ ^[0-9]+$ ]] || exit 2
exec > >(tee -a "$ROOT/control/guard-$SELF.log") 2>&1
normal_complete() {
  [[ -f "$ROOT/control/gpu_work_complete.json" ]] || return 1
  timeout --kill-after=.2s 3s "$BENCH_PY" - "$ROOT" "$RUN" "$CPU" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'control'; r=json.loads((p/'launch.ready').read_text()); d=json.loads((p/'gpu_work_complete.json').read_text())
assert r['root']==sys.argv[1] and r['run']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
assert d['run']==r['run'] and str(d['controller_job_id'])==sys.argv[3]
assert d['config_sha256']==r['config_sha256'] and d['exec_commit']==r['exec_commit']
PY
}
cleanup() {
  local rc=$?
  local cleanup_start=$SECONDS
  trap - EXIT
  if normal_complete; then rc=0; fi
  if (( rc != 0 )); then
    printf 'guard_failure gpu=%s cpu=%s rc=%s\n' "$SELF" "$CPU" "$rc" > "$ROOT/control/STOP"
    # 文件锁由进程持有，死亡自动释放，幸存守卫可接续；自身最后取消。
    exec 9>"$ROOT/control/cleanup.lock"
    while (( SECONDS - cleanup_start < 29 )); do
      if flock -w 1 9; then
        if [[ -f "$ROOT/control/gpu-job-ids.txt" ]]; then
          while IFS= read -r id; do
            [[ "$id" =~ ^[0-9]+$ && "$id" != "$CPU" && "$id" != "$SELF" ]] && bounded_cancel "$id" "$cleanup_start" || true
          done < "$ROOT/control/gpu-job-ids.txt"
        fi
        bounded_cancel "$SELF" "$cleanup_start" || true
        break
      fi
    done
  fi
  echo "EXIT_CODE=$rc"; exit "$rc"
}
bounded_cancel() {
  local id=$1 started=$2 remaining rc
  remaining=$(( 29 - SECONDS + started ))
  if (( remaining <= 0 )); then echo "GUARD_CLEANUP_INCOMPLETE job=$id"; return 1; fi
  (( remaining <= 5 )) || remaining=5
  if timeout --kill-after=.2s "${remaining}s" scancel "$id"; then rc=0; else rc=$?; fi
  echo "GUARD_CANCEL job=$id rc=$rc"
  return "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
(( ${SLURM_RESTART_COUNT:-0} == 0 )) || exit 3
QUERY_FAIL_SINCE=-1
while :; do
  (( QUERY_FAIL_SINCE < 0 || SECONDS - QUERY_FAIL_SINCE < 30 )) || exit 7
  if [[ -f "$ROOT/control/gpu_work_complete.json" ]]; then
    normal_complete && exit 0
    exit 4
  fi
  [[ ! -f "$ROOT/control/STOP" ]] || exit 5
  QUERY_START=$SECONDS
  if STATE=$(timeout --kill-after=.2s 5s squeue -h -j "$CPU" -o '%T' 2>/dev/null); then
    QUERY_FAIL_SINCE=-1
    [[ "$STATE" == PENDING || "$STATE" == CONFIGURING || "$STATE" == RUNNING ]] || exit 6
  else
    (( QUERY_FAIL_SINCE >= 0 )) || QUERY_FAIL_SINCE=$QUERY_START
    (( SECONDS - QUERY_FAIL_SINCE < 30 )) || exit 7
  fi
  if [[ -f "$ROOT/control/launch.ready" ]]; then
    timeout --kill-after=.2s 5s "$BENCH_PY" - "$ROOT" "$RUN" "$CPU" "$SELF" "$$" <<'PY' || exit 8
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1])/'control'; r=json.loads((p/'launch.ready').read_text())
assert r['run']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
h=p/'cpu-heartbeat.json'
if h.exists():
 d=json.loads(h.read_text()); assert d['run']==r['run'] and str(d['controller_job_id'])==str(r['controller_job_id']) and d['config_sha256']==r['config_sha256']; assert time.time()-d['t']<120
else: assert time.time()-(p/'launch.ready').stat().st_mtime<120
# 新固定源码守卫的真实就绪回执；原冻结快照不会写此文件。
target=p/('guardian-ready-'+sys.argv[4]+'.json')
if not target.exists():
 doc={k:r[k] for k in ('run','exec_commit','config_sha256','controller_job_id')}
 doc.update(run_name=r['run'],gpu_job_id=sys.argv[4],step_id=os.environ.get('SLURM_STEP_ID'),pid=int(sys.argv[5]),time=time.time())
 tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(doc)); os.replace(tmp,target)
PY
  else
    (( $(date +%s) < DEADLINE )) || exit 9
  fi
  sleep 10
done
