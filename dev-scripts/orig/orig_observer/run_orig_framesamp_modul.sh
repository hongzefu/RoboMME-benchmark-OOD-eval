#!/usr/bin/env bash
# FrameSamp+Modulation 原侧（原版分支 official-xhard0-0929／927c56d + 外挂只读观测器）单片启动器：逻辑照抄原版
#   robomme_policy_learning@927c56d:scripts/gl_eval_official_xhard0.sh
# （同样的环境变量、守卫、server 命令、3 遍重评、无进展看门狗），只有以下差别（计划第二部分一节 S7，均须写进留档）：
#   1. server 仍监听 PORT；透明代理 framesamp_modul_proxy.py 监听 PORT+1 转发到 PORT；客户端连 PORT+1。
#   2. 客户端经 framesamp_modul_client_wrap.py 同进程 runpy 运行原版 eval.py（外挂只读钩子，逐局写 <REC_ROOT>/<key>.a<N>/
#      {trace.jsonl,frames/,arrays.npz}，route perceptual-framesamp-modul/orig）；PYTHONPATH 在原版子模块 src 之后追加本目录（原版 robomme
#      仍优先解析），起跑前核对 robomme.__file__ 在原版子模块 src 下且 robomme_hard 未导入。
#   3. 起跑断言（R1）：原版工作树 HEAD = 927c56d9879762c1afa0217d79a591093c9ffbe9、git status --porcelain 为空、子模块
#      third_party/robomme_benchmark = 856bc3a189d4172f3f47dbee4424d585f8d78db3；不符即 RUN_BLOCKED。
#   4. 解释器：客户端 ORIG_FRAMESAMP_MODUL_CLIENT_PY（缺省本轮执行副本的 client-env venv），服务端 MME_VLA_PY（缺省
#      <本仓库>/third_party/mme-vla/.venv/bin/python，与 run_seat.sh 同一条 serve_policy.py 命令）；server 代码相对官方
#      ecf086c 零 diff 守卫照旧。注意：原版启动器用 ``uv run --frozen --no-sync``，本脚本直接用 MME_VLA_PY 解释器，
#      **不经 uv 校验 venv 与 uv.lock 是否一致**（只靠上述零 diff 守卫核 uv.lock 文件本身未改）；起跑行打印
#      server_uv_frozen=no 与 uv.lock 的 sha256 供留档。
#   5. 输出：SAVE_ROOT／REC_ROOT 必须是新目录且落在本轮 stage 根 ORIG_STAGE_ROOT（必给）或 NODE_TMP 下；历史目录
#      （v7-eval、v7-eval-stage、v75eval、sgeval-20261004、eval-out、artifacts/v7.5eval、原版工作树）一律拒绝（R7）；
#      ORIG_RESUME=1 只许续用标记 .orig-created 中清单 sha256 与分片都一致的目录。
#   6. 预算（R6）：每遍评估起跑前对本片未完成的每局经 orig_budget.py 调 S8 budget_ledger.py（先 release 上一遍未结 rid，
#      再 reserve --resets 2 --route perceptual-framesamp-modul/orig --key <key>），失败即 RUN_BLOCKED；每遍结束对有终态的局 commit。rid 记在
#      SAVE_ROOT/budget-rids.json；账本路径 BUDGET_LEDGER_ARGS="--ledger <路径>"（顶层参数）或 SGEVAL_BUDGET_LEDGER。
#   7. 看门狗循环里另查 server／代理进程：死亡即打印 SERVER_DIED／PROXY_DIED、杀 eval、退出码 5／6。
#   8. 代理收尾（审计第 13 条）：kill -TERM 后等进程确实退出（≤PROXY_STOP_TIMEOUT=900 s，超时 kill -9 并记
#      proxy_force_killed），PID 在确认退出前不清空；代理写 proxy-<pid>.done（日志封口）后才跑 transparency_check.py
#      （--manifest/--episodes），再由 observer_status.py 写 observer-status.json，末行
#      OBSERVER_COMPLETE=PASS|FAIL episodes= conns= mismatch= hook_errors= report=<ok|missing|crashed>；
#      评估 EXIT_CODE= 保持原语义（观测器失败不改成绩与退出码）。
#   9. 可选 NODE_TMP：SAVE_ROOT／REC_ROOT 默认落在节点本地盘，跑完由调用方搬回（本脚本不搬）。
#  10. server 启动环境显式去掉 XLA_FLAGS 与 JAX 编译缓存变量（与历史一致：只设 XLA_PYTHON_CLIENT_MEM_FRACTION=0.75）。
# 用法（环境变量）：
#   MANIFEST=<清单 jsonl> SHARD=<i> PORT=<端口> RUN_TAG=<标签> ORIG_STAGE_ROOT=<本轮 stage 根> SAVE_ROOT=<新目录> \
#   REC_ROOT=<新目录> [NODE_TMP=<节点本地目录>] [VIDEO_DIR=<目录>] [ONLY_TASKS=a,b] [ORIG_FRAMESAMP_MODUL_CLIENT_PY=<python>] \
#   [MME_VLA_PY=<python>] [SERVER_REPO=<目录>] [FRAMESAMP_MODUL_ORIG_REPO=<原版工作树>] [ORIG_RESUME=1] [BUDGET_LEDGER_ARGS=...] \
#   [V75_ENCODE_CPUS=<代理记账编码核，默认 CPU 亲和集合最后一核>] bash scripts/eval-official/orig_observer/run_orig_framesamp_modul.sh
set -uo pipefail
OBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_REPO="$(cd "$OBS_DIR/../../.." && pwd)"
# shellcheck source=orig_observer_lib.sh
source "$OBS_DIR/orig_observer_lib.sh"
ORIG_COMMIT=927c56d9879762c1afa0217d79a591093c9ffbe9
ORIG_SUBMODULE=third_party/robomme_benchmark
ORIG_SUBMODULE_COMMIT=856bc3a189d4172f3f47dbee4424d585f8d78db3
REPO="${FRAMESAMP_MODUL_ORIG_REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning-official-xhard0}"
cd "$REPO" || { echo "RUN_BLOCKED reason=orig_repo_missing repo=$REPO"; echo "EXIT_CODE=1"; exit 1; }
orig_pin_check "$REPO" "$ORIG_COMMIT" perceptual-framesamp-modul "$ORIG_SUBMODULE" "$ORIG_SUBMODULE_COMMIT" || { echo "EXIT_CODE=1"; exit 1; }
if [[ "${ORIG_OBSERVER_SELFTEST:-0}" == 1 ]]; then echo "ORIG_SELFTEST_GUARD=PASS"; echo "EXIT_CODE=0"; exit 0; fi
: "${MANIFEST:?}" "${SHARD:?}" "${PORT:?}" "${RUN_TAG:?}"
PROXY_PORT=$((PORT + 1))
CKPT=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/eval-out/mmevla-ckpt/perceptual-framesamp-modul/79999  # 历史目录名（NFS 真实路径）
if [[ -n "${NODE_TMP:-}" ]]; then
  mkdir -p "$NODE_TMP" && [[ -w "$NODE_TMP" ]] || { echo "错误: NODE_TMP 不可写: $NODE_TMP"; echo "EXIT_CODE=1"; exit 1; }
  SAVE_ROOT="${SAVE_ROOT:-$NODE_TMP/save}"
  REC_ROOT="${REC_ROOT:-$NODE_TMP/rec}"
fi
: "${SAVE_ROOT:?必须显式给新目录 SAVE_ROOT（或 NODE_TMP）}" "${REC_ROOT:?必须显式给新目录 REC_ROOT（或 NODE_TMP）}"
SAVE_ROOT="$(readlink -m "$SAVE_ROOT")"; REC_ROOT="$(readlink -m "$REC_ROOT")"; MANIFEST="$(readlink -f "$MANIFEST")"
ORIG_POLICY_NAME="mmevla-official-xhard0"  # 历史目录名：原版 eval.py 按 policy_name 落盘，与 v7.5eval 留档同名
SAVE_DIR="$SAVE_ROOT/$ORIG_POLICY_NAME/ckpt79999/seed7"
EPISODE_LOG="$SAVE_DIR/episodes.jsonl"
VIDEO_DIR="$(readlink -m "${VIDEO_DIR:-$SAVE_ROOT/videos-xhard0}")"
# 代理记账编码核：默认取本进程 CPU 亲和集合的最后一个核；必须是亲和集合的子集（录制器初始化时还会再校验）
V75_ENCODE_CPUS="${V75_ENCODE_CPUS:-$(python3 -c 'import os; print(max(os.sched_getaffinity(0)))')}"
python3 -c "
import os,sys
w=set()
for p in sys.argv[1].split(','):
    a,_,b=p.partition('-'); w.update(range(int(a),int(b or a)+1))
sys.exit(0 if w and w<=os.sched_getaffinity(0) else 1)" "$V75_ENCODE_CPUS" \
  || { echo "错误: V75_ENCODE_CPUS=$V75_ENCODE_CPUS 不在 CPU 亲和集合内"; echo "EXIT_CODE=1"; exit 1; }
export V75_ENCODE_CPUS
CLIENT_PY="${ORIG_FRAMESAMP_MODUL_CLIENT_PY:-$BENCH_REPO/artifacts/sg-evaluation/venvs/client-env/bin/python}"
OFFICIAL_SRC="$REPO/$ORIG_SUBMODULE/src"
SERVER_REPO="${SERVER_REPO:-$BENCH_REPO/third_party/mme-vla}"
MME_VLA_PY="${MME_VLA_PY:-$SERVER_REPO/.venv/bin/python}"
# 新目录守卫（R7）：输出只许落在本轮 stage 根或 NODE_TMP 下，历史目录一律拒绝；
# 已有逐局结果即拒绝（原版 done_keys 会整片静默跳过且 EXIT_CODE=0）
NFS=/nfs/turbo/coe-chaijy-unreplicated/hongzefu
: "${ORIG_STAGE_ROOT:?必须给本轮 stage 根 ORIG_STAGE_ROOT}"
ORIG_STAGE_ROOT="$(readlink -m "$ORIG_STAGE_ROOT")"
FORBID=("$NFS/v7-eval/" "$NFS/v7-eval-stage/" "$NFS/v75eval/" "$NFS/sgeval-20261004/" "$NFS/eval-out/"
  "$BENCH_REPO/artifacts/v7.5eval/" "/data/hongzefu/robomme_benchmark_MotionJEPANewTask/artifacts/v7.5eval/" "$REPO/")
ALLOW=("$ORIG_STAGE_ROOT/")
[[ -n "${NODE_TMP:-}" ]] && ALLOW+=("$(readlink -m "$NODE_TMP")/")
for d in "$ORIG_STAGE_ROOT" "$SAVE_ROOT" "$REC_ROOT" "$VIDEO_DIR"; do
  for f in "${FORBID[@]}"; do
    [[ "$d/" == "$f"* ]] && { echo "错误: 输出目录落在历史／原版目录下: $d"; echo "EXIT_CODE=1"; exit 1; }
  done
  [[ "$d" == "$ORIG_STAGE_ROOT" ]] && continue
  ok=0; for a in "${ALLOW[@]}"; do [[ "$d/" == "$a"* ]] && ok=1; done
  [[ "$ok" = 1 ]] || { echo "错误: 输出目录不在白名单根下（${ALLOW[*]}）: $d"; echo "EXIT_CODE=1"; exit 1; }
done
MARK="manifest_sha256=$(sha256sum "$MANIFEST" | cut -c1-64) shard=$SHARD"
if [[ "${ORIG_RESUME:-0}" = 1 ]]; then
  [[ -f "$SAVE_ROOT/.orig-created" && "$(cat "$SAVE_ROOT/.orig-created")" == "$MARK" ]] \
    || { echo "错误: ORIG_RESUME=1 但 $SAVE_ROOT/.orig-created 缺失或与本次清单／分片不符"; echo "EXIT_CODE=1"; exit 1; }
  echo "ORIG_RESUME: 续用本脚本建的 $SAVE_ROOT（$MARK）"
else
  if [[ -d "$SAVE_ROOT" ]] && [[ -n "$(find "$SAVE_ROOT" -name 'episodes*.jsonl' -print -quit 2>/dev/null)" ]]; then
    echo "错误: SAVE_ROOT 已有 episodes*.jsonl（必须新目录）: $SAVE_ROOT"; echo "EXIT_CODE=1"; exit 1
  fi
  if [[ -d "$REC_ROOT" ]] && [[ -n "$(ls -A "$REC_ROOT" 2>/dev/null)" ]]; then
    echo "错误: REC_ROOT 非空（必须新目录）: $REC_ROOT"; echo "EXIT_CODE=1"; exit 1
  fi
fi
mkdir -p "$SAVE_ROOT" "$REC_ROOT"
[[ -f "$SAVE_ROOT/.orig-created" ]] || echo "$MARK" > "$SAVE_ROOT/.orig-created"
SERVER_LOG="$SAVE_ROOT/server.log"
PROXY_LOG="$SAVE_ROOT/proxy.log"
[[ -f "$MANIFEST" ]] || { echo "错误: 清单不存在: $MANIFEST"; echo "EXIT_CODE=1"; exit 1; }
[[ -d "$CKPT/params" ]] || { echo "错误: checkpoint 缺 params: $CKPT"; echo "EXIT_CODE=1"; exit 1; }
[[ -x "$MME_VLA_PY" ]] || { echo "RUN_BLOCKED reason=mme_vla_venv_missing $MME_VLA_PY"; echo "EXIT_CODE=1"; exit 1; }
git -C "$SERVER_REPO" diff --quiet ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b -- src scripts/serve_policy.py pyproject.toml uv.lock packages \
  || { echo "错误: $SERVER_REPO 的 server 代码与官方 ecf086c 不同"; echo "EXIT_CODE=1"; exit 1; }
[[ -x "$CLIENT_PY" ]] || { echo "RUN_BLOCKED reason=client_venv_missing $CLIENT_PY"; echo "EXIT_CODE=1"; exit 1; }
[[ -d "$OFFICIAL_SRC/robomme" ]] || { echo "错误: 原版子模块未初始化: $OFFICIAL_SRC"; echo "EXIT_CODE=1"; exit 1; }
SHARD_N=$(python3 -c "
import json,sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and int(json.loads(l)['shard'])==int(sys.argv[2])))" "$MANIFEST" "$SHARD")
[[ "$SHARD_N" -gt 0 ]] || { echo "错误: 清单里 shard=$SHARD 没有行"; echo "EXIT_CODE=1"; exit 1; }
# 客户端公共环境：去掉代理变量（GL 节点代理拒绝 websocket，HTTP 403）、原版子模块 src 置顶、本目录追加在后
CLIENT_ENV=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY
  NO_PROXY=127.0.0.1,localhost PYTHONUNBUFFERED=1 GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384
  PYTHONPATH="$OFFICIAL_SRC:$OBS_DIR" REC_ROOT="$REC_ROOT" V75_DATA_ROOT="${V75_DATA_ROOT:-$REC_ROOT}"
  V75_ENCODE_CPUS="$V75_ENCODE_CPUS")
# 起跑前核对（同原版启动器）：robomme 必须解析到本工作树原版子模块，且导入 env_runner 后 robomme_hard 不在 sys.modules
( cd examples/robomme && "${CLIENT_ENV[@]}" "$CLIENT_PY" -c "
import sys, robomme, env_runner
assert robomme.__file__.startswith(sys.argv[1]), robomme.__file__
assert 'robomme_hard' not in sys.modules
print('ORIG_ROBOMME', robomme.__file__)" "$OFFICIAL_SRC" ) || { echo "RUN_BLOCKED reason=robomme_not_orig 客户端未解析到原版 robomme 或导入了 robomme_hard"; echo "EXIT_CODE=1"; exit 1; }
# 包装器自检：钩子装得上、原版 robomme 仍优先（写进 REC_ROOT/preflight，不占逐局目录）
( cd examples/robomme && "${CLIENT_ENV[@]}" REC_ROOT="$REC_ROOT/preflight" "$CLIENT_PY" "$OBS_DIR/framesamp_modul_client_wrap.py" --orig-preflight ) \
  || { echo "错误: 观测器包装自检失败"; echo "EXIT_CODE=1"; exit 1; }
echo "=== FRAMESAMP_MODUL_ORIG_XHARD0_OBSERVED host=$(hostname) job=${SLURM_JOB_ID:-none} HEAD=$(git -C "$REPO" rev-parse HEAD) server_repo=$SERVER_REPO@$(git -C "$SERVER_REPO" rev-parse --short HEAD) server_py=$MME_VLA_PY server_uv_frozen=no server_uv_lock_sha256=$(sha256sum "$SERVER_REPO/uv.lock" 2>/dev/null | cut -c1-64) client_py=$CLIENT_PY submodule=$(git -C "$REPO/$ORIG_SUBMODULE" rev-parse HEAD) observer=$OBS_DIR shard=$SHARD shard_n=$SHARD_N port=$PORT proxy_port=$PROXY_PORT ckpt=$CKPT save_root=$SAVE_ROOT rec_root=$REC_ROOT start=$(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
# 端口占用守卫（server 与代理两个端口）
for p in "$PORT" "$PROXY_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null; then
    exec 3>&-
    echo "错误: 端口 ${p} 起跑前已被占用"; echo "EXIT_CODE=1"; exit 1
  fi
done
# 显存上限 0.75 只改 server 启动环境；不开确定性标志、不开编译缓存（与历史一致）
( cd "$SERVER_REPO" && exec env -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS \
  -u JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -u JAX_ENABLE_COMPILATION_CACHE \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 PYTHONUNBUFFERED=1 \
  "$MME_VLA_PY" scripts/serve_policy.py --seed=7 --port="$PORT" \
    policy:checkpoint --policy.dir="$CKPT" --policy.config=mme_vla_suite ) >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
PROXY_PID=""
PROXY_FORCE_KILLED=0
EVAL_PID=""
cleanup() {
  [[ -n "$EVAL_PID" ]] && kill "$EVAL_PID" 2>/dev/null
  orig_stop_proxy
  if kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID"; sleep 2; kill -9 "$SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
for _ in $(seq 1 600); do   # server 就绪上限 20 分钟；server 先死立即退出
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "错误: server 提前退出"; tail -30 "$SERVER_LOG"; echo "EXIT_CODE=1"; exit 1; fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then exec 3>&-; break; fi
  sleep 2
done
grep -m1 -E "Restoring checkpoint|checkpoint" "$SERVER_LOG" | sed 's/^/SERVER_CKPT_LINE /'
echo "server 端口就绪 $(date +%T)"
# 透明代理（客户端 venv：websockets 15 + openpi_client）
( cd examples/robomme && exec "${CLIENT_ENV[@]}" "$CLIENT_PY" "$OBS_DIR/framesamp_modul_proxy.py" \
    --listen "$PROXY_PORT" --upstream "$PORT" --log-dir "$REC_ROOT/proxy" ) >> "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 150); do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then echo "错误: 代理提前退出"; tail -30 "$PROXY_LOG"; echo "EXIT_CODE=1"; exit 1; fi
  grep -q PROXY_READY "$PROXY_LOG" 2>/dev/null && break
  sleep 2
done
grep -q PROXY_READY "$PROXY_LOG" || { echo "错误: 代理 300 s 未就绪"; echo "EXIT_CODE=1"; exit 1; }
echo "代理就绪 $(grep -m1 PROXY_READY "$PROXY_LOG") $(date +%T)"
RC=0
unresolved=0
died=""
for pass in 1 2 3; do
  errors=$(grep -c '"status": "error"' "$EPISODE_LOG" 2>/dev/null); errors=${errors:-0}
  if [[ "$errors" -gt "$SHARD_N" ]]; then echo "RETRY_CAP_HIT shard=$SHARD errors=$errors"; RC=4; break; fi
  pending=$(orig_pending_count "$MANIFEST" "$SHARD" "$EPISODE_LOG" "${ONLY_TASKS:-}") || pending=""
  [[ "$pending" =~ ^[0-9]+$ ]] || { echo "RUN_BLOCKED reason=pending_count_failed"; RC=1; break; }
  [[ "$pending" = 0 ]] && { echo "EVAL_PASS $pass 本片已无未完成局"; break; }
  orig_budget_prepare --manifest "$MANIFEST" --shard "$SHARD" --episode-log "$EPISODE_LOG" --state "$SAVE_ROOT/budget-rids.json" \
    --resets 2 --route perceptual-framesamp-modul/orig ${ONLY_TASKS:+--only-tasks "$ONLY_TASKS"} || { RC=1; break; }
  echo "EVAL_PASS $pass errors_so_far=$errors pending=$pending $(date -Is)"
  ( cd examples/robomme && "${CLIENT_ENV[@]}" \
      "$CLIENT_PY" "$OBS_DIR/framesamp_modul_client_wrap.py" eval.py --args.host=127.0.0.1 --args.port="$PROXY_PORT" \
      --args.model_seed=7 --args.model_ckpt_id=79999 \
      --args.policy_name="$ORIG_POLICY_NAME" --args.episode_manifest="$MANIFEST" --args.shard="$SHARD" \
      --args.video_dir="$VIDEO_DIR" --args.save_dir="$SAVE_ROOT" ${ONLY_TASKS:+--args.only_tasks="$ONLY_TASKS"} ) &
  EVAL_PID=$!
  started=$(date +%s)
  stalled=0
  while kill -0 "$EVAL_PID" 2>/dev/null; do   # 无进展看门狗：首局放宽 10 分钟，之后 30 分钟
    sleep 30
    # server／代理进程守卫：任一死亡即杀 eval、停止重评，以不同退出码报告
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then died=server; echo "SERVER_DIED pass=$pass $(date -Is)"; tail -30 "$SERVER_LOG"; fi
    if [[ -z "$died" ]] && ! kill -0 "$PROXY_PID" 2>/dev/null; then died=proxy; echo "PROXY_DIED pass=$pass $(date -Is)"; tail -30 "$PROXY_LOG"; fi
    if [[ -n "$died" ]]; then pkill -TERM -P "$EVAL_PID" 2>/dev/null; kill "$EVAL_PID" 2>/dev/null; break; fi
    last=$(stat -c %Y "$EPISODE_LOG" 2>/dev/null || echo "$started")
    [[ "$last" -lt "$started" ]] && last=$started
    limit=1800; [[ ! -f "$EPISODE_LOG" ]] && limit=2400
    if (( $(date +%s) - last > limit )); then
      echo "EVAL_STALL pass=$pass idle_s=$(( $(date +%s) - last )) 杀掉重起（记基础设施 error）"
      pkill -TERM -P "$EVAL_PID" 2>/dev/null; kill "$EVAL_PID" 2>/dev/null; stalled=1; break
    fi
  done
  wait "$EVAL_PID"; RC=$?; EVAL_PID=""
  # 预算结算：有终态的局 commit（不改 RC）
  orig_budget_settle --manifest "$MANIFEST" --shard "$SHARD" --episode-log "$EPISODE_LOG" --state "$SAVE_ROOT/budget-rids.json" \
    ${ONLY_TASKS:+--only-tasks "$ONLY_TASKS"}
  if [[ "$died" = server ]]; then RC=5; break; fi
  if [[ "$died" = proxy ]]; then RC=6; break; fi
  [[ "$stalled" = 1 ]] && { RC=124; continue; }
  unresolved=$(python3 -c "
import json,sys
last={}
for l in open(sys.argv[1]):
    if l.strip():
        r=json.loads(l); last[(r['task'],r['source_episode'])]=r['status']
print(sum(v=='error' for v in last.values()))" "$EPISODE_LOG" 2>/dev/null || echo 0)
  echo "EVAL_PASS_END $pass rc=$RC unresolved_errors=$unresolved"
  [[ "$unresolved" = 0 ]] && break
done
# 退出码：0 全部终态；1 预算／起跑阻断；3 有未解决 error；4 重试上限；5 server 死亡；6 代理死亡；124 最后一遍无进展
# 有未解决的 error 身份时不得以 0 退出
[[ "$RC" = 0 && "${unresolved:-0}" != 0 ]] && RC=3
# 先停代理（等进程确实退出）→ 等日志封口 → 透明性对账 → 观测器完整性判定；以上都不改 RC
orig_stop_proxy
orig_wait_seal "$REC_ROOT/proxy" "${PROXY_STOPPED_PID:-none}"
orig_finalize_framesamp_modul "$CLIENT_PY" "$OBS_DIR" "$REC_ROOT" "$MANIFEST" "$SHARD" "$EPISODE_LOG" "${ONLY_TASKS:-}"
[[ -n "${NODE_TMP:-}" ]] && echo "NODE_TMP_STAGED save_root=$SAVE_ROOT rec_root=$REC_ROOT（由调用方搬回并核对 sha256）"
echo "EVAL_RC=$RC end=$(date -Is)"
echo "EXIT_CODE=$RC"
exit "$RC"
