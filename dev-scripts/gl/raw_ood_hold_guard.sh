#!/usr/bin/env bash
# 本轮GPU独立守卫：CPU消失不能被尚未发布ready掩盖。
set -euo pipefail
ROOT=${1:?运行根}; RUN=${2:?运行名}; CPU=${3:?CPU作业}; DEADLINE=${4:?准备截止秒}
READY=${5:-$ROOT/control/launch.ready}
SELF=${SLURM_JOB_ID:?}; : "${BENCH_PY:?必须提供uv管理解释器}"
HOLD_PID=""
FAILURE_PHASE=initialization
[[ "$CPU" =~ ^[0-9]+$ && "$SELF" =~ ^[0-9]+$ && "$DEADLINE" =~ ^[0-9]+$ ]] || exit 2
exec > >(tee -a "$ROOT/control/guard-$SELF.log") 2>&1
normal_complete() {
  [[ -f "$ROOT/control/gpu_work_complete.json" ]] || return 1
  timeout --kill-after=.2s 3s "$BENCH_PY" -I -S -u - "$ROOT" "$RUN" "$CPU" "$READY" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'control'; ready=pathlib.Path(sys.argv[4])
assert ready.is_absolute() and not ready.is_symlink() and ready.resolve().is_relative_to(p.resolve())
assert '..' not in ready.relative_to(p).parts and not p.is_symlink() and not p.parent.is_symlink()
assert all(not node.is_symlink() for node in ready.parents if node.is_relative_to(p))
r=json.loads(ready.read_text()); d=json.loads((p/'gpu_work_complete.json').read_text())
assert r['root']==sys.argv[1] and r['run']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
assert d['run']==r['run'] and str(d['controller_job_id'])==sys.argv[3]
assert d['config_sha256']==r['config_sha256'] and d['exec_commit']==r['exec_commit']
PY
}
cleanup() {
  local rc=$?
  trap - EXIT
  [[ -z "$HOLD_PID" ]] || kill "$HOLD_PID" 2>/dev/null || true
  if (( rc == 143 )) && normal_complete; then rc=0; fi
  if (( rc != 0 )) && [[ ! -f "$ROOT/control/guard-failure-$SELF.json" ]]; then
    failure="$ROOT/control/guard-failure-$SELF.json"
    printf '{"run":"%s","controller_job_id":"%s","gpu_job_id":"%s","component":"guard","exit_code":%s,"payload_rc":null,"phase":"%s","time":%s}\n' \
      "$RUN" "$CPU" "$SELF" "$rc" "$FAILURE_PHASE" "$(date +%s)" > "$failure.$$.tmp"
    mv "$failure.$$.tmp" "$failure"
  fi
  echo "GUARD_SELF_EXIT rc=$rc gpu_action=none notification_established=0"
  echo "EXIT_CODE=$rc"; exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
(( ${SLURM_RESTART_COUNT:-0} == 0 )) || exit 3
if [[ -z "${SLURM_STEP_ID:-}" || "$SLURM_STEP_ID" == batch ]]; then
  echo "GUARD_BATCH_MISUSE cpu_monitor=not_started allocation_hold=1 gpu_action=none notification_established=0"
  # GPU batch应只sleep；误用也保持allocation，不因监视器退出结束工作负载。
  while :; do sleep 3600 & HOLD_PID=$!; wait "$HOLD_PID" || true; done
fi
QUERY_FAIL_SINCE=-1
while :; do
  FAILURE_PHASE=slurm_cpu_query
  (( QUERY_FAIL_SINCE < 0 || SECONDS - QUERY_FAIL_SINCE < 30 )) || exit 7
  if [[ -f "$ROOT/control/gpu_work_complete.json" ]]; then
    normal_complete && exit 0
    exit 4
  fi
  [[ ! -f "$ROOT/control/STOP" ]] || exit 5
  QUERY_START=$SECONDS
  if [[ -f "$ROOT/control/intentional-supervisor-stop.json" ]]; then
    echo "GUARD_INTENT_PENDING_VALIDATE gpu=$SELF"
  elif STATE=$(timeout --kill-after=.2s 5s squeue -h -j "$CPU" -o '%T' 2>/dev/null); then
    QUERY_FAIL_SINCE=-1
    FAILURE_PHASE=slurm_cpu_state
    [[ "$STATE" == PENDING || "$STATE" == CONFIGURING || "$STATE" == RUNNING ]] || exit 6
  else
    (( QUERY_FAIL_SINCE >= 0 )) || QUERY_FAIL_SINCE=$QUERY_START
    (( SECONDS - QUERY_FAIL_SINCE < 30 )) || exit 7
  fi
  if [[ -f "$READY" ]]; then
    FAILURE_PHASE=ready_heartbeat_payload
    echo "GUARD_CHECK_BEGIN gpu=$SELF time=$(date +%s) startup_deadline_s=30 read_deadline_s=5"
    if timeout --kill-after=.2s 30s "$BENCH_PY" -I -S -u - "$ROOT" "$RUN" "$CPU" "$SELF" "$$" "$READY" <<'PY'
import sys
print('GUARD_PHASE gpu='+sys.argv[4]+' phase=python_entered',flush=True)
print('GUARD_PHASE gpu='+sys.argv[4]+' phase=startup_standard_library',flush=True)
import os,time,signal
phase='stdlib_import'
def expired(*_): raise TimeoutError('guard读取验证超过5秒 phase='+phase)
signal.signal(signal.SIGALRM,expired); signal.alarm(5)
try:
 print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 import json,pathlib
 p=pathlib.Path(sys.argv[1])/'control'
 ready=pathlib.Path(sys.argv[6])
 assert ready.is_absolute() and not ready.is_symlink() and ready.resolve().is_relative_to(p.resolve())
 assert '..' not in ready.relative_to(p).parts and not p.is_symlink() and not p.parent.is_symlink()
 assert all(not node.is_symlink() for node in ready.parents if node.is_relative_to(p))
 phase='ready_read'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 r=json.loads(ready.read_text())
 phase='ready_validate'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 assert r['root']==sys.argv[1] and r['run']==sys.argv[2] and r['run_name']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
 intent=p/'intentional-supervisor-stop.json'
 if intent.exists():
  phase='intentional_supervisor_stop'; stop=json.loads(intent.read_text())
  for key in ('run','exec_commit','config_sha256','controller_job_id'): assert str(stop[key])==str(r[key])
  print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
  accepted=p/('guardian-intent-accepted-'+sys.argv[4]+'.json')
  tmp=accepted.with_name(accepted.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(stop)); os.replace(tmp,accepted)
  raise SystemExit(0)
 phase='heartbeat_read'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 h=p/'cpu-heartbeat.json'
 if h.exists():
  d=json.loads(h.read_text()); phase='heartbeat_validate'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
  assert all(str(d[k])==str(r[k]) for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'))
  age=time.time()-float(d['t']); print('GUARD_HEARTBEAT gpu='+sys.argv[4]+' age_s='+str(age),flush=True)
  assert -30<=age<120
 else:
  phase='heartbeat_missing_grace'; assert time.time()-ready.stat().st_mtime<120
 phase='ready_publish'
 target=p/('guardian-ready-'+sys.argv[4]+'.json')
 if not target.exists():
  doc={k:r[k] for k in ('run','exec_commit','config_sha256','controller_job_id')}
  doc.update(run_name=r['run'],gpu_job_id=sys.argv[4],step_id=os.environ.get('SLURM_STEP_ID'),pid=int(sys.argv[5]),time=time.time())
  tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(doc)); os.replace(tmp,target)
 print('GUARD_PHASE gpu='+sys.argv[4]+' phase=validated',flush=True)
except Exception as e:
 signal.alarm(0)
 print('GUARD_PAYLOAD_FAILURE gpu='+sys.argv[4]+' phase='+phase+' error_kind='+type(e).__name__+' message='+str(e),flush=True)
 if 'json' in locals() and 'pathlib' in locals():
  p=pathlib.Path(sys.argv[1])/'control'; target=p/('guard-failure-'+sys.argv[4]+'.json')
  doc=dict(run=sys.argv[2],controller_job_id=sys.argv[3],gpu_job_id=sys.argv[4],component='guard',exit_code=8,payload_rc=1,
           phase=phase,error_kind=type(e).__name__,message=str(e),time=time.time(),pid=int(sys.argv[5]))
  if 'r' in locals():
   doc.update({k:r[k] for k in ('exec_commit','config_sha256') if k in r})
  tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(doc)); os.replace(tmp,target)
 raise
finally: signal.alarm(0)
PY
    then
      echo "GUARD_CHECK_END gpu=$SELF payload_rc=0 time=$(date +%s)"
    else
      payload_rc=$?
      echo "GUARD_CHECK_END gpu=$SELF payload_rc=$payload_rc time=$(date +%s)"
      # 先保存原始失败；仅退出此guard，不停止任何GPU工作负载。
      failure="$ROOT/control/guard-failure-$SELF.json"
      if [[ ! -f "$failure" ]]; then
        printf '{"run":"%s","controller_job_id":"%s","gpu_job_id":"%s","component":"guard","exit_code":8,"payload_rc":%s,"phase":"payload_unreported","time":%s}\n' \
          "$RUN" "$CPU" "$SELF" "$payload_rc" "$(date +%s)" > "$failure.$$.tmp"
        mv "$failure.$$.tmp" "$failure"
      fi
      exit 8
    fi
    if [[ -f "$ROOT/control/guardian-intent-accepted-$SELF.json" ]]; then exit 0; fi
  else
    FAILURE_PHASE=prepare_deadline
    (( $(date +%s) < DEADLINE )) || exit 9
  fi
  sleep 10
done
