#!/usr/bin/env bash
# 五模型 OOD 评估：单卡分片脚本（最简方案）。用法：bash run_shard.sh <g1|g2|g3|g4>
# 一张卡 = 一个任务组；依次跑五个模型，每个模型只跑本组任务；已有 result.json 的局由 scripts/evaluate.py 自动跳过。
# 模型参数与环境变量逐字取自 control/run-config-resume-03.json（exec_commit 891b462），由生成脚本机械写出，禁止手改路径。
# 退出码：0 全部模型 rc=0；1 某模型三次尝试仍非零；2 参数错；3 输出根不是实体目录或 infra 级联（env_build 1 局／其它 infra 2 局）；4 重试登记超封顶；5 无进展 2700 s 且三次仍非零按 1 计；143 收到 TERM。
set -uo pipefail
ROOT="${SHARD_ROOT:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02}"   # SHARD_ROOT 只供本机夹具演练覆盖
CODE="${SHARD_CODE:-$ROOT/runtime-code-resume-03}"
OUT=$ROOT/gl
G="${1:?用法: run_shard.sh <g1|g2|g3|g4>}"
case "$G" in
  g1) TASKS=PickHighlight,SwingXtimes,StopCube,VideoUnmaskSwap; PORT=22400;;
  g2) TASKS=BinFill,VideoUnmask,ButtonUnmaskSwap,VideoRepick;    PORT=22500;;
  g3) TASKS=MoveCube,PickXtimes,RouteStick,VideoPlaceButton;     PORT=22600;;
  g4) TASKS=InsertPeg,ButtonUnmask,PatternLock,VideoPlaceOrder;  PORT=22700;;
  *) echo "未知分组 $G"; echo "EXIT_CODE=2"; exit 2;;
esac
WORK=$ROOT/simple/$G/work; LOG=$ROOT/simple/logs/shard-$G.log; RLOG=$ROOT/simple/logs/reattempts-$G.log; HALF=$ROOT/simple/halfdone/$G
[[ -d "$OUT" && ! -L "$OUT" ]] || { echo "RUN_BLOCKED reason=out_not_dir out=$OUT"; echo "EXIT_CODE=3"; exit 3; }   # 第 14 条：输出根必须是实体目录
mkdir -p "$WORK/servers" "$ROOT/simple/logs" "$HALF" || { echo "EXIT_CODE=3"; exit 3; }
exec > >(exec tee -a "$LOG") 2>&1; TEE_PID=$!
export HF_HOME=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/hf-cache
export HF_HUB_OFFLINE=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENPI_DATA_HOME=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/openpi-data
export PYTHONPATH=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code-resume-03/src:/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code-resume-03/third_party/robomme_benchmark/src:/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code-resume-03/third_party/PonderPounce:/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code-resume-03/third_party/mme-vla/src:/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code-resume-03/third_party/mme-vla/packages/openpi-client/src
export PYTHONUNBUFFERED=1
export ROBOMME_EVAL_ROOT=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code-resume-03
export SGEVAL_AUDIT=1
export SGEVAL_THIRD_PARTY=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code-resume-03/third_party
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR=/home/hongzefu/.cache/uv
export XDG_CACHE_HOME=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/cache
export PYTHONUNBUFFERED=1 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
for f in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do [[ -f "$f" ]] && { export VK_ICD_FILENAMES="$f"; break; }; done
MAIN_PY="${SHARD_MAIN_PY:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python}"
NOPROG_S="${SHARD_NOPROG_S:-2700}"; POLL_S="${SHARD_POLL_S:-60}"
CHILD=""
stop_left() {  # 停掉上一模型留下的服务端（墙钟 75 硬退出不会关服务端）
  local m; for m in "$WORK"/servers/server-metadata-*.json; do [[ -f "$m" ]] || continue
    echo "STOP_LEFT_SERVER metadata=$m"; ( cd "$CODE" && "$MAIN_PY" scripts/evaluate.py --stop-server "$m" ) || echo "STOP_LEFT_SERVER_FAIL metadata=$m"; done; }
finish() {  # $1=退出码：统一收尾，保证 EXIT_CODE= 尾行落盘后再退出
  stop_left; echo "SHARD_DONE group=$G overall_rc=$1 $(date -Is)"; echo "EXIT_CODE=$1"; exec >/dev/null 2>&1; wait "$TEE_PID" 2>/dev/null; exit "$1"; }
on_term() { echo "SHARD_SIGNAL sig=TERM $(date -Is)"; [[ -n "$CHILD" ]] && { kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; }; finish 143; }
trap on_term TERM INT HUP
task_dirs() {  # $1=标签：本分片任务在该模型下的全部局目录（glob 无匹配时静默）
  local t d; for t in ${TASKS//,/ }; do for d in "$OUT/rollouts/$1/ood/seed7/raw/${t}_ep"*; do [[ -d "$d" ]] && echo "$d"; done; done; }
register_halfdone() {  # $1=标签：有目录但无 result.json 的局 = 上一次尝试被中断；移存到 $HALF 保留证据并逐条登记（移走后原路径不再存在，不会重复登记）；返回累计数
  local d n; while read -r d; do [[ -n "$d" && ! -e "$d/result.json" ]] || continue
    n="$HALF/$1/$(date -u +%Y%m%dT%H%M%SZ)-$(basename "$d")"; mkdir -p "$(dirname "$n")"; mv -- "$d" "$n" 2>/dev/null || continue
    echo "$(date -Is) group=$G label=$1 halfdone=$d moved_to=$n" >> "$RLOG"; done < <(task_dirs "$1")
  wc -l < "$RLOG" 2>/dev/null || echo 0; }
cap_check() {  # $1=累计数：超过封顶即整个分片以 4 退出
  (( $1 > 3 )) && { echo "RUN_BLOCKED reason=reattempt_cap group=$G total=$1 cap=3 $(date -Is)"; finish 4; }; return 0; }
infra_hits() {  # $1=标签 $2=起始 epoch $3=正则：本分片任务、本次尝试新写的 result.json 里 infra_reason 命中 $3 的个数（不用管道退出码判定，避免 pipefail 下 SIGPIPE 漏报）
  local list; list=$(task_dirs "$1" | while read -r d; do [[ -f "$d/result.json" && "$d/result.json" -nt "$T0FILE" ]] && grep -lE "\"infra_reason\": *\"($3)\"" "$d/result.json" 2>/dev/null; done); [[ -z "$list" ]] && echo 0 || echo "$list" | wc -l; }
newest_mtime() {  # $1=标签：本分片任务目录下最新文件的 epoch（无文件给 0）
  local m; m=$(task_dirs "$1" | while read -r d; do find "$d" -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null; done | sort -n | tail -n 1); echo "${m%%.*}"; }
run_model() {  # $1=标签 $2=解释器 $3..=模型参数；每模型最多 3 次尝试；每次尝试前登记并移存半截目录（口径 8，封顶 3 局／分片）
  local label="$1" py="$2" rc=1 try t0 total n_env n_inf last idle; shift 2
  for try in 1 2 3; do
    stop_left
    total=$(register_halfdone "$label"); echo "REATTEMPT group=$G label=$label try=$try total=$total cap=3"; cap_check "$total"
    t0=$(date +%s); T0FILE=$(mktemp); touch -d "@$t0" "$T0FILE"
    echo "MODEL_START group=$G label=$label try=$try host=$(hostname) job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} cuda=${CUDA_VISIBLE_DEVICES:-unset} $(date -Is)"
    ( cd "$CODE" && exec "$py" scripts/evaluate.py --dataset ood --seed 7 --tasks "$TASKS" --out "$OUT" --no-render \
        --port-base "$PORT" --work-dir "$WORK" --cfg "server_dir=$WORK/servers" "$@" ) & CHILD=$!
    while kill -0 "$CHILD" 2>/dev/null; do
      sleep "$POLL_S" & wait $! 2>/dev/null
      n_env=$(infra_hits "$label" "$t0" 'env_build'); n_inf=$(infra_hits "$label" "$t0" '[a-z_]+')
      if (( n_env >= 1 || n_inf >= 2 )); then
        echo "RUN_BLOCKED reason=infra_cascade group=$G label=$label env_build=$n_env infra_total=$n_inf $(date -Is)"
        kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; CHILD=""; rm -f "$T0FILE"; finish 3
      fi
      last=$(newest_mtime "$label"); (( last > t0 )) || last=$t0; idle=$(( $(date +%s) - last ))
      if (( idle > NOPROG_S )); then
        echo "NOPROG group=$G label=$label idle_s=$idle limit_s=$NOPROG_S try=$try $(date -Is)"
        kill -TERM "$CHILD" 2>/dev/null; sleep 30; kill -KILL "$CHILD" 2>/dev/null; break
      fi
    done
    wait "$CHILD" 2>/dev/null; rc=$?; CHILD=""; rm -f "$T0FILE"
    echo "MODEL_DONE group=$G label=$label try=$try rc=$rc $(date -Is)"
    stop_left
    [[ $rc -eq 0 ]] && break
  done
  total=$(register_halfdone "$label"); echo "REATTEMPT group=$G label=$label try=final total=$total cap=3"; cap_check "$total"   # 最后一次中断也要登记
  return $rc; }
echo "SHARD_START group=$G tasks=$TASKS port=$PORT code=$CODE out=$OUT $(date -Is)"
for lab in groundsg-ground-sg-oracle perceptual-framesamp-modul pp smvla groundsg-ground-sg-qwenvl; do total=$(register_halfdone "$lab"); done; echo "REATTEMPT group=$G label=all try=start total=$total cap=3"; cap_check "$total"
overall=0
run_model groundsg-ground-sg-oracle /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/envs/client-env/.venv/bin/python --model groundsg --ckpt /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/ckpt/mme/symbolic-grounded-subgoal/79999 --cfg mme_vla_py=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-sgeval/third_party/mme-vla/.venv/bin/python --cfg openpi_data_home=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/openpi-data --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 --cfg jax_cache_root=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/jax-cache --groundsg-variant ground-sg-oracle --cfg groundsg_keep_official=false || overall=$?
run_model perceptual-framesamp-modul /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python --model perceptual-framesamp-modul --ckpt /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/ckpt/mme/perceptual-framesamp-modul/79999 --cfg mme_vla_py=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-sgeval/third_party/mme-vla/.venv/bin/python --cfg openpi_data_home=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/openpi-data --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 --cfg jax_cache_root=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/jax-cache || overall=$?
run_model pp /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/envs/client-env/.venv/bin/python --model pp --ckpt /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/ckpt/pp/ponderpounce-9b-robomme --cfg pp_py=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-sgeval/third_party/PonderPounce/.venv/bin/python --cfg openpi_data_home=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/openpi-data --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 --cfg jax_cache_root=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/jax-cache || overall=$?
run_model smvla /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python --model smvla --ckpt /nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/checkpoints/simplememvla_robomme --cfg smvla_py=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-sgeval2-b34/artifacts/v8-two/venvs/smvla-env/bin/python --cfg openpi_data_home=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/openpi-data --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 --cfg jax_cache_root=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/jax-cache || overall=$?
run_model groundsg-ground-sg-qwenvl /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/venvs/client-env/bin/python --model groundsg --ckpt /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/ckpt/mme/symbolic-grounded-subgoal/79999 --cfg mme_vla_py=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-sgeval/third_party/mme-vla/.venv/bin/python --cfg openpi_data_home=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/openpi-data --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 --cfg jax_cache_root=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/jax-cache --groundsg-variant ground-sg-qwenvl --qwenvl-groundsg-adapter /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/ckpt/qwenvl-groundsg/checkpoint-1200 --cfg xla_mem_fraction=0.65 --cfg groundsg_keep_official=false || overall=$?
(( overall == 0 )) || overall=1
finish "$overall"
