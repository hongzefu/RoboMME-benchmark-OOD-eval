#!/usr/bin/env bash
# 本轮GPU独立守卫：CPU消失不能被尚未发布ready掩盖。
set -euo pipefail
ROOT=${1:?运行根}; RUN=${2:?运行名}; CPU=${3:?CPU作业}; DEADLINE=${4:?准备截止秒}
READY=${5:-$ROOT/control/launch.ready}
SELF=${SLURM_JOB_ID:?}; : "${BENCH_PY:?必须提供uv管理解释器}"
HOLD_PID=""
CACHE_B64=""; RETRY_SLEEP=10
FAILURE_PHASE=initialization
CONTEXT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ "$CPU" =~ ^[0-9]+$ && "$SELF" =~ ^[0-9]+$ && "$DEADLINE" =~ ^[0-9]+$ ]] || exit 2
exec > >(tee -a "$ROOT/control/guard-$SELF.log") 2>&1
normal_complete() {
  [[ -f "$ROOT/control/gpu_work_complete.json" ]] || return 1
  timeout --kill-after=.2s 30s "$BENCH_PY" -I -S -u - "$ROOT" "$RUN" "$CPU" "$READY" "$CONTEXT_DIR" <<'PY'
import sys,signal
def expired(*_): raise TimeoutError('完成标记读取超过5秒')
signal.signal(signal.SIGALRM,expired); signal.alarm(5)
import json,pathlib,hashlib
sys.path.insert(0,sys.argv[5])
from raw_ood_context import validate_terminal_marker
p=pathlib.Path(sys.argv[1])/'control'; ready=pathlib.Path(sys.argv[4])
assert ready.is_absolute() and not ready.is_symlink() and ready.resolve().is_relative_to(p.resolve())
assert '..' not in ready.relative_to(p).parts and not p.is_symlink() and not p.parent.is_symlink()
assert all(not node.is_symlink() for node in ready.parents if node.is_relative_to(p))
r=json.loads(ready.read_text()); d=json.loads((p/'gpu_work_complete.json').read_text())
assert r['root']==sys.argv[1] and r['run']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
cfg=pathlib.Path(r['config_path'])
assert cfg.is_absolute() and cfg.is_relative_to(p.parent) and not cfg.is_symlink()
assert all(not node.is_symlink() for node in cfg.parents if node.is_relative_to(p.parent))
raw=cfg.read_bytes(); c=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==r['config_sha256']
c['config_sha256']=r['config_sha256']
assert all(str(c[k])==str(r[k]) for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'))
assert c['root']==sys.argv[1]
validate_terminal_marker(d,c)
PY
}
cleanup() {
  local rc=$?
  trap - EXIT
  [[ -z "$HOLD_PID" ]] || kill "$HOLD_PID" 2>/dev/null || true
  if (( rc == 143 )) && normal_complete; then rc=0; fi
  if (( rc != 0 )); then
    failure="$ROOT/control/guard-failure-$SELF.json"
    [[ ! -f "$failure" ]] || failure="$ROOT/control/guard-failure-$SELF.current-$$-$(date +%s%N).json"
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
UNVERIFIED_QUERY_SINCE=""
while :; do
  FAILURE_PHASE=slurm_cpu_query
  if [[ -f "$ROOT/control/gpu_work_complete.json" ]]; then
    normal_complete && exit 0
    exit 4
  fi
  [[ ! -f "$ROOT/control/STOP" ]] || echo "GUARD_PAUSE_DRAINING gpu=$SELF gpu_action=none"
  if [[ -f "$ROOT/control/intentional-supervisor-stop.json" ]]; then
    echo "GUARD_INTENT_PENDING_VALIDATE gpu=$SELF"
  elif STATE=$(timeout --kill-after=.2s 5s squeue -h -j "$CPU" -o '%T' 2>/dev/null); then
    UNVERIFIED_QUERY_SINCE=""
    FAILURE_PHASE=slurm_cpu_state
    [[ "$STATE" == PENDING || "$STATE" == CONFIGURING || "$STATE" == RUNNING ]] || exit 6
  else
    echo "GUARD_CPU_QUERY state=UNKNOWN gpu=$SELF heartbeat_authority=payload"
    # 已验证心跳的真实t/loop_t决定失效，不用另一个RPC失败期限误停健康守卫。
    if [[ -z "$CACHE_B64" ]]; then
      [[ -n "$UNVERIFIED_QUERY_SINCE" ]] || UNVERIFIED_QUERY_SINCE=$(date +%s)
      (( $(date +%s) - UNVERIFIED_QUERY_SINCE < 120 )) || exit 7
    fi
  fi
  if [[ -f "$READY" || -n "$CACHE_B64" ]]; then
    FAILURE_PHASE=ready_heartbeat_payload
    echo "GUARD_CHECK_BEGIN gpu=$SELF time=$(date +%s) startup_deadline_s=30 read_deadline_s=5"
    RETRY_SLEEP=10
    if payload_output=$(timeout --kill-after=.2s 30s "$BENCH_PY" -I -S -u - "$ROOT" "$RUN" "$CPU" "$SELF" "$$" "$READY" "$CACHE_B64" <<'PY'
import sys
print('GUARD_PHASE gpu='+sys.argv[4]+' phase=python_entered',flush=True)
print('GUARD_PHASE gpu='+sys.argv[4]+' phase=startup_standard_library',flush=True)
import os,time,signal
phase='stdlib_import'
def expired(*_): raise TimeoutError('guard读取验证超过5秒 phase='+phase)
signal.signal(signal.SIGALRM,expired); signal.alarm(5)
try:
 print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 import json,pathlib,base64,math
 cached=None
 if sys.argv[7]:
  cached=json.loads(base64.b64decode(sys.argv[7],validate=True))
  assert cached['root']==sys.argv[1] and cached['ready_path']==sys.argv[6] and cached['gpu_job_id']==sys.argv[4] and cached['guard_pid']==int(sys.argv[5])
  assert cached['ready']['run']==sys.argv[2] and str(cached['ready']['controller_job_id'])==sys.argv[3]
  assert math.isfinite(float(cached['verified_t']))
 p=pathlib.Path(sys.argv[1])/'control'
 ready=pathlib.Path(sys.argv[6])
 assert ready.is_absolute() and not ready.is_symlink() and ready.resolve().is_relative_to(p.resolve())
 assert '..' not in ready.relative_to(p).parts and not p.is_symlink() and not p.parent.is_symlink()
 assert all(not node.is_symlink() for node in ready.parents if node.is_relative_to(p))
 phase='ready_read'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 r=json.loads(ready.read_text())
 phase='ready_validate'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 assert r['root']==sys.argv[1] and r['run']==sys.argv[2] and r['run_name']==sys.argv[2] and str(r['controller_job_id'])==sys.argv[3]
 if cached is not None: assert all(cached['ready'][k]==r[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'))
 intent=p/'intentional-supervisor-stop.json'
 paused=p/'paused.json'
 if (p/'STOP').exists() and paused.exists():
  import hashlib
  pointer=json.loads(paused.read_text()); proof_path=pathlib.Path(pointer['proof_path'])
  assert proof_path.is_relative_to(p) and not proof_path.is_symlink()
  raw=proof_path.read_bytes(); proof=json.loads(raw)
  assert hashlib.sha256(raw).hexdigest()==pointer['proof_sha256']
  assert all(proof[k]==r[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'))
  assert proof['expected']==4000 and proof['accepted']+proof['remaining']==4000 and proof['unsettled']==0 and proof['drained'] is True
  raise SystemExit(12)
 monitor='supervisor'
 if intent.exists():
  phase='intentional_supervisor_stop'; stop=json.loads(intent.read_text())
  for key in ('run','exec_commit','config_sha256','controller_job_id'): assert str(stop[key])==str(r[key])
  print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
  accepted_stop=stop
  monitor='controller'
  # 监督器退出不结束健康守卫；改由控制器真实主循环进展维持监视。
 phase='heartbeat_read'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
 h=p/('controller-heartbeat.json' if monitor=='controller' else 'cpu-heartbeat.json')
 if h.exists():
  d=json.loads(h.read_text()); phase='heartbeat_validate'; print('GUARD_PHASE gpu='+sys.argv[4]+' phase='+phase,flush=True)
  assert all(str(d[k])==str(r[k]) for k in ('run','run_name','exec_commit','config_sha256','controller_job_id'))
  age=time.time()-float(d['t']); print('GUARD_HEARTBEAT gpu='+sys.argv[4]+' age_s='+str(age),flush=True)
  assert -30<=age<120
  assert -30<=time.time()-float(d['loop_t'])<120
  if monitor=='controller': assert type(d['pid']) is int and d['pid']>0 and type(d['start_ticks']) is int
  cached=dict(root=sys.argv[1],ready_path=sys.argv[6],gpu_job_id=sys.argv[4],guard_pid=int(sys.argv[5]),ready=r,verified_t=float(d['t']),loop_t=float(d['loop_t']),monitor=monitor)
  print('GUARD_CACHE_B64='+base64.b64encode(json.dumps(cached).encode()).decode(),flush=True)
 else:
  if monitor=='controller': raise ValueError('退出意图后没有控制器心跳')
  phase='heartbeat_missing_grace'; assert time.time()-ready.stat().st_mtime<120
 phase='ready_publish'
 if 'accepted_stop' in locals():
  accepted=p/('guardian-intent-accepted-'+sys.argv[4]+'.json')
  tmp=accepted.with_name(accepted.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(accepted_stop)); os.replace(tmp,accepted)
 target=p/('guardian-ready-'+sys.argv[4]+'.json')
 if not target.exists():
  doc={k:r[k] for k in ('run','exec_commit','config_sha256','controller_job_id')}
  doc.update(run_name=r['run'],gpu_job_id=sys.argv[4],step_id=os.environ.get('SLURM_STEP_ID'),pid=int(sys.argv[5]),time=time.time())
  tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(doc)); os.replace(tmp,target)
 if cached is not None:
  target=p/('guardian-heartbeat-'+sys.argv[4]+'.json')
  doc={k:r[k] for k in ('run','run_name','exec_commit','config_sha256','controller_job_id')}
  doc.update(gpu_job_id=sys.argv[4],step_id=os.environ.get('SLURM_STEP_ID'),pid=int(sys.argv[5]),t=time.time(),monitor_t=time.time(),verified_t=cached['verified_t'],verified_loop_t=cached['loop_t'],monitor=monitor)
  tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(doc)); os.replace(tmp,target)
 print('GUARD_PHASE gpu='+sys.argv[4]+' phase=validated',flush=True)
except Exception as e:
 signal.alarm(0)
 fs_timeout=isinstance(e,TimeoutError) and phase in ('ready_read','heartbeat_read','intentional_supervisor_stop','ready_publish')
 transient_io=isinstance(e,OSError) and e.errno in (5,121,116)
 if (transient_io or fs_timeout) and 'cached' in locals() and cached is not None:
  remaining=min(cached['verified_t']+120,cached['loop_t']+120)-time.time()
  if remaining>0:
   print('GUARD_NFS_RETRY errno='+str(e.errno)+' expires='+str(cached['verified_t']+120)+' phase='+phase,flush=True)
   print('GUARD_RETRY_SLEEP='+format(min(10,remaining),'.3f'),flush=True)
   raise SystemExit(11)
 print('GUARD_PAYLOAD_FAILURE gpu='+sys.argv[4]+' phase='+phase+' error_kind='+type(e).__name__+' message='+str(e),flush=True)
 if 'json' in locals() and 'pathlib' in locals():
  p=pathlib.Path(sys.argv[1])/'control'; target=p/('guard-failure-'+sys.argv[4]+'.json')
  if target.exists(): target=p/('guard-failure-'+sys.argv[4]+'.current-'+str(os.getpid())+'-'+str(time.time_ns())+'.json')
  doc=dict(run=sys.argv[2],controller_job_id=sys.argv[3],gpu_job_id=sys.argv[4],component='guard',exit_code=8,payload_rc=1,
           phase=phase,error_kind=type(e).__name__,message=str(e),time=time.time(),pid=int(sys.argv[5]))
  if 'r' in locals():
   doc.update({k:r[k] for k in ('exec_commit','config_sha256') if k in r})
  tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp'); tmp.write_text(json.dumps(doc)); os.replace(tmp,target)
 raise
finally: signal.alarm(0)
PY
)
    then
      printf '%s\n' "$payload_output"
      echo "GUARD_CHECK_END gpu=$SELF payload_rc=0 time=$(date +%s)"
    else
      payload_rc=$?
      printf '%s\n' "$payload_output"
      echo "GUARD_CHECK_END gpu=$SELF payload_rc=$payload_rc time=$(date +%s)"
      # 仅本次payload完成身份校验后的专用状态可结束守卫；旧回执只是证据。
      if (( payload_rc == 12 )); then exit 0; fi
      if (( payload_rc == 11 )); then
        while IFS= read -r line; do
          [[ "$line" != GUARD_CACHE_B64=* ]] || CACHE_B64=${line#GUARD_CACHE_B64=}
          [[ "$line" != GUARD_RETRY_SLEEP=* ]] || RETRY_SLEEP=${line#GUARD_RETRY_SLEEP=}
        done <<< "$payload_output"
        [[ "$RETRY_SLEEP" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 8
        sleep "$RETRY_SLEEP"; continue
      fi
      # 先保存原始失败；仅退出此guard，不停止任何GPU工作负载。
      failure="$ROOT/control/guard-failure-$SELF.json"
      if [[ ! -f "$failure" ]]; then
        printf '{"run":"%s","controller_job_id":"%s","gpu_job_id":"%s","component":"guard","exit_code":8,"payload_rc":%s,"phase":"payload_unreported","time":%s}\n' \
          "$RUN" "$CPU" "$SELF" "$payload_rc" "$(date +%s)" > "$failure.$$.tmp"
        mv "$failure.$$.tmp" "$failure"
      fi
      exit 8
    fi
    while IFS= read -r line; do
      [[ "$line" != GUARD_CACHE_B64=* ]] || CACHE_B64=${line#GUARD_CACHE_B64=}
    done <<< "$payload_output"
  else
    FAILURE_PHASE=prepare_deadline
    (( $(date +%s) < DEADLINE )) || exit 9
  fi
  sleep 10
done
