#!/usr/bin/env bash
# run_shard.sh 控制流夹具（不触发仿真、不跑模型、不连集群）。用法：bash dev-scripts/gl/simple/shard_fixture.sh
# 在 mktemp 临时目录里放桩 evaluate.py，用 SHARD_ROOT/SHARD_CODE/SHARD_MAIN_PY/SHARD_POLL_S/SHARD_NOPROG_S
# 覆盖让同目录的 run_shard.sh（与仓库提交的逐字节相同）跑 7 个场景，逐个断言：
#   退出码、日志末行 EXIT_CODE=、reattempts 登记行数、RUN_BLOCKED／NOPROG／SHARD_SIGNAL 行。
# 注意：run_shard.sh 里各模型解释器是 NFS 上的绝对路径，桩只用标准库，本机挂载 NFS 时可直接执行；只读 NFS。
# 末行 SHARD_FIXTURE=PASS cases=7；跑完删除临时目录。耗时约 1～2 分钟（noprog 场景含脚本内固定的 30 s 等待）。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SH=$HERE/run_shard.sh
STUB=$HERE/shard_fixture_stub_evaluate.py
TMP=$(mktemp -d "${TMPDIR:-/tmp}/shard-fixture.XXXXXX") || { echo "SHARD_FIXTURE=FAIL reason=mktemp"; exit 1; }
trap 'rm -rf "$TMP"' EXIT
# run_shard.sh 每次尝试用 mktemp 建一个 T0 时间戳文件；TERM 路径（finish 143）不删它，故让它落在本临时目录里一并清掉
mkdir -p "$TMP/t0"; export TMPDIR="$TMP/t0"
mkdir -p "$TMP/code/scripts"
cp "$STUB" "$TMP/code/scripts/evaluate.py"
PY=$(command -v python3)
PASS=0; FAIL=0

# 公共断言：$1=场景 $2=实际退出码 $3=期望退出码 $4=期望登记行数 $5=日志必须含的正则（空=不检查） $6=日志不得含的正则（空=不检查）
check() {
  local sc=$1 rc=$2 expect=$3 nreg=$4 must=$5 mustnot=$6 L=$TMP/root-$1/simple/logs/shard-g1.log RL=$TMP/root-$1/simple/logs/reattempts-g1.log
  local got_reg=0 why=""
  [[ -f "$RL" ]] && got_reg=$(wc -l < "$RL")
  [[ $rc -eq $expect ]] || why+=" rc=$rc"
  [[ "$(tail -n 1 "$L" 2>/dev/null)" == "EXIT_CODE=$expect" ]] || why+=" tail=$(tail -n 1 "$L" 2>/dev/null | tr ' ' '_')"
  [[ $got_reg -eq $nreg ]] || why+=" reattempts=$got_reg"
  [[ -z "$must" ]] || grep -qE "$must" "$L" || why+=" missing=/$must/"
  [[ -z "$mustnot" ]] || ! grep -qE "$mustnot" "$L" || why+=" unexpected=/$mustnot/"
  if [[ -z "$why" ]]; then PASS=$((PASS + 1)); echo "FIXTURE_CASE=PASS scenario=$sc rc=$rc reattempts=$got_reg"
  else FAIL=$((FAIL + 1)); echo "FIXTURE_CASE=FAIL scenario=$sc expect_rc=$expect expect_reattempts=$nreg why:$why"; tail -n 8 "$L" 2>/dev/null | sed 's/^/    /'; fi
}

# 普通场景：前台跑完整个分片
run_case() {  # $1=场景 $2=期望退出码 $3=期望登记行数 $4=必含正则 $5=不得含正则 $6..=额外环境变量
  local sc=$1 expect=$2 nreg=$3 must=$4 mustnot=$5 R=$TMP/root-$1; shift 5
  mkdir -p "$R/gl/rollouts"
  env "$@" FIXTURE_SCENARIO="$sc" FIXTURE_CNT="$R/cnt" SHARD_ROOT="$R" SHARD_CODE="$TMP/code" \
    SHARD_MAIN_PY="$PY" SHARD_POLL_S=1 bash "$SH" g1 >/dev/null 2>&1
  check "$sc" $? "$expect" "$nreg" "$must" "$mustnot"
}

run_case ok       0 0 'SHARD_DONE group=g1 overall_rc=0' 'RUN_BLOCKED|NOPROG'
run_case wall75   0 1 'MODEL_DONE group=g1 label=groundsg-ground-sg-oracle try=1 rc=75' 'RUN_BLOCKED|NOPROG'
run_case envbuild 3 0 'RUN_BLOCKED reason=infra_cascade group=g1 label=pp env_build=[1-9]' 'NOPROG'
run_case infra2   3 0 'RUN_BLOCKED reason=infra_cascade group=g1 label=pp env_build=0 infra_total=([2-9]|[1-9][0-9])' 'NOPROG'
run_case cap      4 4 'RUN_BLOCKED reason=reattempt_cap group=g1 total=4 cap=3' 'infra_cascade|NOPROG'
run_case noprog   0 0 'NOPROG group=g1 label=pp idle_s=[0-9]+ limit_s=3 try=1' 'RUN_BLOCKED' SHARD_NOPROG_S=3

# term 场景：后台起分片，等桩在 pp 模型里进入 sleep 后对 bash 进程发 TERM
R=$TMP/root-term; PF=$R/stub.pid; mkdir -p "$R/gl/rollouts"
FIXTURE_SCENARIO=term FIXTURE_CNT="$R/cnt" FIXTURE_PIDFILE="$PF" SHARD_ROOT="$R" SHARD_CODE="$TMP/code" \
  SHARD_MAIN_PY="$PY" SHARD_POLL_S=1 bash "$SH" g1 >/dev/null 2>&1 &
BPID=$!
for _ in $(seq 1 300); do [[ -s "$PF" ]] && break; sleep 0.1; done
SPID=$(cat "$PF" 2>/dev/null || echo 0)
sleep 1.5; kill -TERM "$BPID"; wait "$BPID"; TRC=$?
check term "$TRC" 143 0 'SHARD_SIGNAL sig=TERM' 'RUN_BLOCKED|NOPROG'
if [[ "$SPID" == 0 ]] || kill -0 "$SPID" 2>/dev/null; then
  PASS=$((PASS - 1)); FAIL=$((FAIL + 1)); echo "FIXTURE_CASE=FAIL scenario=term why: stub_pid=$SPID 收到 TERM 后仍存活或未启动"
  kill -KILL "$SPID" 2>/dev/null
fi

TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 && $TOTAL -eq 7 ]]; then echo "SHARD_FIXTURE=PASS cases=$TOTAL"; exit 0; fi
echo "SHARD_FIXTURE=FAIL cases=$TOTAL passed=$PASS failed=$FAIL"; exit 1
