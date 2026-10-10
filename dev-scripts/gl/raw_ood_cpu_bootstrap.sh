#!/usr/bin/env bash
# 本轮不可变启动快照：等待配置，准备超时或异常仅取消登记GPU。
set -euo pipefail
ROOT=${1:?运行根}; RUN=${2:?运行名}; CPU=${3:-${SLURM_JOB_ID:?}}; DEADLINE=${4:?准备截止秒}
: "${BENCH_PY:?必须提供uv管理解释器}"
export PYTHONUNBUFFERED=1
[[ "$CPU" =~ ^[0-9]+$ && "$DEADLINE" =~ ^[0-9]+$ && -d "$ROOT/control" && ! -L "$ROOT" ]] || exit 2
exec > >(tee -a "$ROOT/control/bootstrap-$CPU.log") 2>&1
cleanup() {
  local rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    printf 'bootstrap_failure cpu=%s rc=%s\n' "$CPU" "$rc" > "$ROOT/control/STOP"
    if [[ -f "$ROOT/control/gpu-job-ids.txt" ]]; then
      while IFS= read -r id; do [[ "$id" =~ ^[0-9]+$ && "$id" != "$CPU" ]] && scancel "$id" || true; done < "$ROOT/control/gpu-job-ids.txt"
    fi
  fi
  echo "EXIT_CODE=$rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
(( ${SLURM_RESTART_COUNT:-0} == 0 )) || exit 3
while [[ ! -f "$ROOT/control/launch.ready" ]]; do
  [[ ! -f "$ROOT/control/STOP" ]] || exit 4
  (( $(date +%s) < DEADLINE )) || exit 5
  sleep 5
done
# 校验后只执行ready中确定的解释器、入口、配置；不加载模型或提交作业。
mapfile -t LAUNCH < <("$BENCH_PY" - "$ROOT" "$RUN" "$CPU" <<'PY'
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
for k in ('python','supervisor','config_path'): print(r[k])
PY
)
[[ ${#LAUNCH[@]} == 3 && -x "${LAUNCH[0]}" && -f "${LAUNCH[1]}" ]] || exit 6
# 保留EXIT trap；监督器收到TERM后独立清理，bootstrap负责最终退出记录。
"${LAUNCH[0]}" "${LAUNCH[1]}" run --config "${LAUNCH[2]}"
