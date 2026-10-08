#!/usr/bin/env bash
# 第二档原侧席位启动器（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.6、1.7；子任务 S6）。
#
# 一个席位（一张 GPU）上：起模型服务（与新侧同一条命令：groundsg 走 serve_policy.py --seed=7、pp 走
# ponderpounce.eval.robomme_server --args.seed 0，就绪判定与 kill -0 存活检查都复用 run_seat.sh 的 start_server）→
# 跑官方原版驱动（groundsg：official_hard_runner.py --variant …；pp：pp_official_runner.py）→ 每局 rgb24 原始帧就地转码
# 为 episode.mp4（帧数核对一致才删原始帧）并原子发布到 NFS 输出键 <media-root>/<label>/hard-verify/orig/<key>.a<attempt>/
# → trap 收尾（收驱动与服务进程组、全量同步、判定行）。
#
# 守卫一律复用 run_seat.sh（source）：pick_port／port_busy 起前探端口、start_server 等就绪时 kill -0 查服务存活、
# noprog_limit／idle_s 做 NO_PROGRESS 无进展检测（结果文件与本轮起点的 mtime；首局含 600 s 放宽）、step_cap_pairing
# 与 variant_pairing 配对核对、note_epoch／epoch_annotate 记服务启动边界、transcode_episode_dir／publish_dir 转码发布。
#
# 原侧驱动没有账本：本脚本按驱动写的 results.jsonl 分轮调度——第 1 轮跑全部身份（--attempt 1）；之后只把「仍无非
# infra 结果行」的身份按「已用尝试数 + 1」分组重跑（每身份至多 2 次尝试）。重试（第 2 次尝试）每局占一次
# --infra-retry-budget，额度用完的身份不再跑，收尾计 missing。每一轮开跑前都先重启服务（第 1 轮除外）：PonderPounce
# 的固定 sid 在一个服务进程内只用一次（n=0），基础设施重试一律「先重启服务，再重发同一 sid」；驱动自己的 reconnect
# 只用于同一局内的断线。驱动退出 2（服务不可达）、服务中途死亡、NO_PROGRESS（杀驱动与服务）都按基础设施处理：
# 服务重启最多 2 次、无进展最多 1 次，超出即 INFRA_EXHAUSTED（rc 4）；驱动退出 3（分片或导入断言）即 RUN_BLOCKED。
# 本局目录存在而驱动没写结果行（被收掉或崩溃）时，在 results.jsonl 补一条 infra 错误行（launcher_synthetic=true），
# 保证每个已发布目录都有结果行指向它。
#
# 目录：驱动的 --out 在节点本地 <local-root>/orig/<label>/（每局目录与原始帧、results.jsonl、server-epochs.tsv）；
#   NFS 状态 <stage>/sNN/orig/<label>/（results.jsonl 与 results.epochs.jsonl 副本、server-epochs.tsv、runner.log、
#   服务日志、本脚本日志 official-sNN.log）；setsid 进程组记在 <stage>/sNN/orig/.v8-pgids（pair_seat.sh 据此核实
#   服务已退出、显存已释放）。续跑（如 48 小时到期后换节点）时节点上没有 results.jsonl 就先从 NFS 副本恢复。
# server_epoch：同 run_seat.sh——每次服务就绪在 server-epochs.tsv 记「epoch、此刻 results.jsonl 行数」，收尾与每次周期
#   同步时把结果行补上 server_epoch 写成 <stage>/sNN/orig/<label>/results.epochs.jsonl（gate2_compare.py 读这份）。
# 视频帧数口径：PonderPounce 原侧 = video_history 帧数 + 1 个初始帧 + 已执行步数 − 缺画面步数（frames.json 的 count）；
#   GroundSG 原侧 = 外围委托记录的帧数（演示帧 + 已执行步数，以驱动写的 frames.json 为准）。转码前后帧数相等才算 ok。
#
# 第三阶段（1006-rename-official-names-and-stage3-eval-plan.md 八.3、八.10 第 3、8 条；接口冻结说明 2.2～2.4、三）：
#   --policy-seed <n> 必填（缺失或非非负整数 RUN_BLOCKED reason=policy_seed、退出 3）；预算五参数
#   --budget-ledger --trajectory-cap --shared-infra-cap --expired-cap --planned-first-tries 必填（缺任一
#   RUN_BLOCKED reason=budget_args、退出 3，不回落 budget_ledger.py 常量默认值），groundsg 原样转发给
#   official_hard_runner.py（驱动经共享账本 reserve／claim_reset／claim_retry／commit／release，route
#   groundsg/<variant>/seed<n>/orig）；驱动退出 5（轨迹预算不足）即 RUN_BLOCKED reason=budget、本席退出 5。
#   变体放行 ground-sg-memer（--memer-adapter <dir>，与 --qwenvl-groundsg-adapter 互斥；配对核对由 run_seat.sh 的
#   variant_pairing 负责）。服务端的模型种子由 run_seat.sh 的 build_server_cmd 读 POLICY_SEED（R3）。
#
# 用法：
#   bash run_official_hard.sh --run-name R --seat NN --repo <执行副本> --stage <NFS 运行根> --shard <shard-NN.json> \
#     --policy {groundsg,pp} --dataset hard-verify --max-steps 1300 --infra-retry-budget N --policy-seed N \
#     --budget-ledger L --trajectory-cap N --shared-infra-cap N --expired-cap N --planned-first-tries N \
#     [--groundsg-variant {ground-sg-oracle,ground-sg-qwenvl,ground-sg-memer}] \
#     [--qwenvl-groundsg-adapter D | --memer-adapter D] \
#     [--groundsg-ckpt D --openpi-data-home D --tokenizer-sha256 H] [--pp-ckpt D] \
#     [--gpu 0] [--cpus 0-3] [--media-root D] [--local-root D] [--limit N] [--episode-wall S] [--sync-interval S]
# 判定行：OFFICIAL_SEAT_DONE seat=NN policy=<label> outcome=pass|fail|aborted rc=<rc> …；SEAT_REC_SYNC=PASS|FAIL …
#   transcoded=<n> frame_mismatch=<n> transcode_fail=<n>；末行 EXIT_CODE=<rc>。
# 退出码：0 全部身份有非 infra 结果行且同步 PASS；2 参数错误；3 RUN_BLOCKED（配对、预检、导入断言、policy_seed、
#   budget_args）；4 基础设施用尽；5 轨迹预算不足；6 仍有身份无终态（重试额度或 2 次尝试用尽）；7 只有同步 FAIL；
#   中断 130/143（HUP 129）。
# 解释器：驱动用 SGEVAL_CLIENT_PY（客户端扩展环境），服务用 MME_VLA_PY／PP_PY（同 run_seat.sh）；本脚本内小工具用 BENCH_PY。
set -uo pipefail
export PYTHONUNBUFFERED=1

_ENV_SGEVAL_CLIENT_PY="${SGEVAL_CLIENT_PY:-}" ; _ENV_PP_PY="${PP_PY:-}" ; _ENV_BENCH_PY="${BENCH_PY:-}" ; _ENV_MME_VLA_PY="${MME_VLA_PY:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_seat.sh
source "$HERE/run_seat.sh"

RUN_NAME="" ; SEAT="" ; STAGE="" ; MEDIA_ROOT="" ; SHARD="" ; POLICY="" ; LOCAL_ROOT="" ; SYNC_INTERVAL=120
GPU=0 ; CPUS="" ; LIMIT=0 ; INFRA_RETRY_BUDGET="" ; DATASET="" ; MAX_STEPS="" ; STRICT_CAP=0
RUNNER_PID="" ; FINALIZED=0 ; RUN_OUT="" ; ORIG_STATE="" ; PUB_ROOT="" ; LABEL="" ; PORT=""
# 第三阶段：模型种子、MemER adapter 与预算五参数（全部必填项在 parse_official_args 里核对）
POLICY_SEED="${POLICY_SEED:-}" ; MEMER_ADAPTER="${MEMER_ADAPTER:-}" ; BUDGET_LEDGER="" ; TRAJECTORY_CAP=""
SHARED_INFRA_CAP="" ; EXPIRED_CAP="" ; PLANNED_FIRST_TRIES=""

official_die2() {
  echo "$1" >&2
  echo "OFFICIAL_SEAT_DONE seat=${SEAT:-?} policy=${POLICY:-?} outcome=fail rc=2 reason=bad_args"
  echo "EXIT_CODE=2"
  exit 2
}

official_block3() {  # $1 = reason；$2 = 说明。必填项缺失（policy_seed／budget_args）：RUN_BLOCKED、退出 3
  echo "RUN_BLOCKED reason=$1 $2"
  echo "OFFICIAL_SEAT_DONE seat=${SEAT:-?} policy=${POLICY:-?} outcome=fail rc=3 reason=$1"
  echo "EXIT_CODE=3"
  exit 3
}

check_stage3_args() {  # 第三阶段必填项：policy_seed 与预算五参数（不回落任何默认值）
  [[ "$POLICY_SEED" =~ ^[0-9]+$ ]] || official_block3 policy_seed "value=${POLICY_SEED:-unset}（--policy-seed 必填，非负整数）"
  local miss=() n v
  for n in BUDGET_LEDGER TRAJECTORY_CAP SHARED_INFRA_CAP EXPIRED_CAP PLANNED_FIRST_TRIES; do
    v="${!n}"
    if [[ -z "$v" ]]; then miss+=("$n")
    elif [[ "$n" != BUDGET_LEDGER && ! "$v" =~ ^[0-9]+$ ]]; then miss+=("$n=$v")
    fi
  done
  (( ${#miss[@]} == 0 )) || official_block3 budget_args "missing=${miss[*]}"
  return 0
}

parse_official_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-name) RUN_NAME="$2"; shift 2;;
      --seat) SEAT="$2"; shift 2;;
      --repo) REPO="$2"; shift 2;;
      --stage) STAGE="$2"; shift 2;;
      --media-root) MEDIA_ROOT="$2"; shift 2;;
      --shard) SHARD="$2"; shift 2;;
      --policy) POLICY="$2"; shift 2;;
      --dataset) DATASET="$2"; shift 2;;
      --max-steps) MAX_STEPS="$2"; shift 2;;
      --strict-cap) STRICT_CAP=1; shift;;
      --groundsg-variant) GROUNDSG_VARIANT="$2"; shift 2;;
      --qwenvl-groundsg-adapter) QWENVL_ADAPTER="$2"; shift 2;;
      --memer-adapter) MEMER_ADAPTER="$2"; shift 2;;
      --policy-seed) POLICY_SEED="$2"; shift 2;;
      --budget-ledger) BUDGET_LEDGER="$2"; shift 2;;
      --trajectory-cap) TRAJECTORY_CAP="$2"; shift 2;;
      --shared-infra-cap) SHARED_INFRA_CAP="$2"; shift 2;;
      --expired-cap) EXPIRED_CAP="$2"; shift 2;;
      --planned-first-tries) PLANNED_FIRST_TRIES="$2"; shift 2;;
      --groundsg-ckpt) GROUNDSG_CKPT="$2"; shift 2;;
      --pp-ckpt) PP_CKPT="$2"; shift 2;;
      --openpi-data-home) OPENPI_HOME="$2"; shift 2;;
      --tokenizer-sha256) TOKENIZER_SHA="$2"; shift 2;;
      --infra-retry-budget) INFRA_RETRY_BUDGET="$2"; shift 2;;
      --gpu) GPU="$2"; shift 2;;
      --cpus) CPUS="$2"; shift 2;;
      --limit) LIMIT="$2"; shift 2;;
      --episode-wall) WALL_ALL="$2"; shift 2;;
      --noprog-s) NOPROG_S="$2"; shift 2;;
      --sync-interval) SYNC_INTERVAL="$2"; shift 2;;
      --local-root) LOCAL_ROOT="$2"; shift 2;;
      *) official_die2 "未知参数 $1";;
    esac
  done
  [[ -n "$RUN_NAME" && -n "$SEAT" && -n "$REPO" && -n "$STAGE" && -n "$SHARD" && -n "$POLICY" ]] \
    || official_die2 "缺少必需参数（--run-name --seat --repo --stage --shard --policy）"
  [[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || official_die2 "--run-name 只许字母数字 . _ -"
  [[ "$SEAT" =~ ^[0-9]{2}$ ]] || official_die2 "--seat 须为两位席号 NN（如 03）"
  [[ "$POLICY" == "groundsg" || "$POLICY" == "pp" ]] || official_die2 "--policy 只许 groundsg、pp（原侧）"
  [[ "$INFRA_RETRY_BUDGET" =~ ^[0-9]+$ ]] || official_die2 "--infra-retry-budget 必填且为非负整数"
  [[ "$LIMIT" =~ ^[0-9]+$ && "$SYNC_INTERVAL" =~ ^[0-9]+$ ]] || official_die2 "--limit／--sync-interval 须为非负整数"
  [[ -z "$MAX_STEPS" || "$MAX_STEPS" =~ ^[0-9]+$ ]] || official_die2 "--max-steps 须为非负整数"
  [[ -z "$WALL_ALL" || "$WALL_ALL" =~ ^[0-9]+$ ]] || official_die2 "--episode-wall 须为非负整数秒"
  check_stage3_args
  if [[ "$POLICY" == "groundsg" ]]; then
    [[ -n "$OPENPI_HOME" && -n "$TOKENIZER_SHA" ]] || official_die2 "跑 groundsg 须给 --openpi-data-home --tokenizer-sha256"
  else
    [[ -n "$PP_CKPT" ]] || official_die2 "跑 pp 须给 --pp-ckpt"
  fi
  REPO="$(cd "$REPO" && pwd)" || official_die2 "--repo 不存在"
  SEAT_IDX=$((10#$SEAT))
  LABEL="$(pol_label "$POLICY")"
  [[ -n "$LOCAL_ROOT" ]] || LOCAL_ROOT="/tmp/$RUN_NAME-s$SEAT"
  [[ -n "$MEDIA_ROOT" ]] || MEDIA_ROOT="$STAGE/media"
  OUT="$STAGE/s$SEAT/orig"           # .v8-pgids 落这里（note_pgid）
  ORIG_STATE="$OUT/$LABEL"
  RUN_OUT="$LOCAL_ROOT/orig/$LABEL"
  PUB_ROOT="$MEDIA_ROOT/$LABEL/${DATASET:-unset}/orig"
  [[ -n "$CPUS" ]] && TASKSET=(taskset -c "$CPUS")
  # 解释器：调用方覆盖优先，其余按 --repo 取缺省
  BENCH_PY="${_ENV_BENCH_PY:-$REPO/.venv/bin/python}"
  MME_VLA_PY="${_ENV_MME_VLA_PY:-$REPO/third_party/mme-vla/.venv/bin/python}"
  SGEVAL_CLIENT_PY="${_ENV_SGEVAL_CLIENT_PY:-$REPO/artifacts/sg-evaluation/venvs/client-env/bin/python}"
  PP_PY="${_ENV_PP_PY:-$REPO/third_party/PonderPounce/.venv/bin/python}"
  return 0
}

runner_script_of() { if [[ "$1" == "groundsg" ]]; then echo official_hard_runner.py; else echo pp_official_runner.py; fi; }

build_runner_cmd() {  # $1 = 尝试号；$2 = 逗号分隔 key；$3 = 端口 → 设 RUN_ENV、RUN_ARGV（不启动）
  RUN_ENV=(PYTHONUNBUFFERED=1)
  [[ "$POLICY" == "groundsg" ]] && RUN_ENV+=(USE_HF=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}")
  RUN_ARGV=("$SGEVAL_CLIENT_PY" "dev-scripts/orig/$(runner_script_of "$POLICY")" --shard "$SHARD" --out "$RUN_OUT"
            --host 127.0.0.1 --port "$3" --max-steps "$MAX_STEPS" --attempt "$1" --only "$2")
  if [[ "$POLICY" == "groundsg" ]]; then
    RUN_ARGV+=(--variant "$GROUNDSG_VARIANT")
    [[ -n "$QWENVL_ADAPTER" ]] && RUN_ARGV+=(--qwenvl-groundsg-adapter "$QWENVL_ADAPTER")
    [[ -n "${MEMER_ADAPTER:-}" ]] && RUN_ARGV+=(--memer-adapter "$MEMER_ADAPTER")
    # 第三阶段：模型种子与预算五参数原样转发（有值才加；必填核对在 parse_official_args）
    [[ -n "${POLICY_SEED:-}" ]] && RUN_ARGV+=(--policy-seed "$POLICY_SEED")
    [[ -n "${BUDGET_LEDGER:-}" ]] && RUN_ARGV+=(--budget-ledger "$BUDGET_LEDGER")
    [[ -n "${TRAJECTORY_CAP:-}" ]] && RUN_ARGV+=(--trajectory-cap "$TRAJECTORY_CAP")
    [[ -n "${SHARED_INFRA_CAP:-}" ]] && RUN_ARGV+=(--shared-infra-cap "$SHARED_INFRA_CAP")
    [[ -n "${EXPIRED_CAP:-}" ]] && RUN_ARGV+=(--expired-cap "$EXPIRED_CAP")
    [[ -n "${PLANNED_FIRST_TRIES:-}" ]] && RUN_ARGV+=(--planned-first-tries "$PLANNED_FIRST_TRIES")
  fi
  return 0
}

start_runner() {  # $1 = 尝试号；$2 = 逗号分隔 key
  build_runner_cmd "$1" "$2" "$PORT"
  ( cd "$REPO" && exec setsid env "${NOPROXY_ENV[@]}" "${RUN_ENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "${TASKSET[@]}" \
      "${RUN_ARGV[@]}" ) >>"$ORIG_STATE/runner.log" 2>&1 &
  RUNNER_PID=$!
  note_pgid runner "$RUNNER_PID"
  echo "ORIG_RUNNER_START policy=$LABEL attempt=$1 n=$(tr ',' '\n' <<<"$2" | grep -c .) pid=$RUNNER_PID port=$PORT max_steps=$MAX_STEPS"
}

# 读驱动的 results.jsonl 与节点上的每局目录。$1=plan：补 synthetic 行后给出下一轮（attempt=0 表示没有可跑的）；
# $1=mark：只补 synthetic 行并报告 missing。打印一行 ORIG_PLAN …。
plan_round() {
  "$(tool_py)" - "$1" "$SHARD" "$RUN_OUT" "$INFRA_RETRY_BUDGET" "$LIMIT" "$POLICY" "${GROUNDSG_VARIANT:-}" "${DATASET:-}" <<'PY'
import json, os, re, sys
mode, shard, out, budget, limit, policy, variant, dataset = sys.argv[1:9]
budget, limit = int(budget), int(limit)
rows = json.load(open(shard, encoding="utf-8"))
if limit:
    rows = rows[:limit]
keys = [r["key"] for r in rows]
by_key = {r["key"]: r for r in rows}
res_path = os.path.join(out, "results.jsonl")
res = []
if os.path.exists(res_path):
    for line in open(res_path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("key"):
            res.append(r)
have = {(r["key"], int(r.get("attempt") or 1)) for r in res}
pat = re.compile(r"^(.+)\.a(\d+)$")
dirs = set()
if os.path.isdir(out):
    for name in os.listdir(out):
        m = pat.match(name)
        if m and m.group(1) in by_key and os.path.isdir(os.path.join(out, name)):
            dirs.add((m.group(1), int(m.group(2))))
synth = sorted(dirs - have)
if synth:
    with open(res_path, "a", encoding="utf-8") as fh:
        for k, a in synth:
            row = dict(by_key[k])
            row.update(side="orig", policy=policy, dataset=dataset or None, attempt=a, status="error", task_success=False,
                       infra=True, infra_reason="launcher_no_result_row", launcher_synthetic=True,
                       error="LAUNCHER: 本局目录存在但驱动未写结果行（进程被收掉或崩溃）",
                       ep_dir=os.path.join(out, f"{k}.a{a}"))
            if policy == "groundsg" and variant:
                row["policy_variant"] = variant
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            res.append(row)
            print(f"ORIG_SYNTH_INFRA key={k} attempt={a}")
        fh.flush()
        os.fsync(fh.fileno())
    have |= set(synth)
seen: dict[str, int] = {}
for k, a in have:
    seen[k] = max(seen.get(k, 0), a)
final = {r["key"] for r in res if not r.get("infra")} & set(keys)
used = sum(1 for k, a in have if a >= 2 and k in by_key)
left = max(0, budget - used)
todo = [k for k in keys if k not in final and seen.get(k, 0) < 2]
fresh = [k for k in todo if seen.get(k, 0) == 0]
retry = [k for k in todo if seen.get(k, 0) == 1]
attempt, group, skipped = 0, [], []
if mode == "plan":
    if fresh:
        attempt, group = 1, fresh
    elif retry:
        group, skipped = retry[:left], retry[left:]
        attempt = 2 if group else 0
else:
    skipped = retry if left == 0 else []
missing = [k for k in keys if k not in final]
print(f"ORIG_PLAN attempt={attempt} n={len(group)} total={len(keys)} final={len(final)} missing={len(missing)} "
      f"retry_used={used} retry_left={left} retry_skipped={len(skipped)} keys={','.join(group)}")
PY
}

plan_field() { sed -n "s/.*ORIG_PLAN.* $1=\\([^ ]*\\).*/\\1/p" <<<"$2" | tail -n 1; }

stop_runner() { [[ -n "$RUNNER_PID" ]] && kill_group "$RUNNER_PID"; RUNNER_PID=""; }

copy_state() {  # 节点上的 results.jsonl／server-epochs.tsv 复制到 NFS 状态目录并补 server_epoch
  local f
  for f in results.jsonl server-epochs.tsv; do
    [[ -f "$RUN_OUT/$f" ]] || continue
    cp -f "$RUN_OUT/$f" "$ORIG_STATE/.$f.tmp" && mv -f "$ORIG_STATE/.$f.tmp" "$ORIG_STATE/$f"
  done
  epoch_annotate "$ORIG_STATE/results.jsonl" "$ORIG_STATE/server-epochs.tsv" "$ORIG_STATE/results.epochs.jsonl" >/dev/null
}

done_names() {  # 已有结果行的每局目录名 <key>.a<attempt>
  [[ -f "$RUN_OUT/results.jsonl" ]] || return 0
  "$(tool_py)" - "$RUN_OUT/results.jsonl" <<'PY'
import json, sys
names = set()
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(r, dict) and r.get("key"):
        names.add(f"{r['key']}.a{int(r.get('attempt') or 1)}")
for n in sorted(names):
    print(n)
PY
}

sync_orig() {  # $1 = periodic|final。periodic 只处理已有结果行的目录；final 处理节点上剩余全部每局目录
  local mode="$1" name fail=0 left n bytes ok mism tcf
  copy_state
  local -a names=()
  if [[ "$mode" == "periodic" ]]; then
    mapfile -t names < <(done_names)
  else
    for d in "$RUN_OUT"/*.a*/; do [[ -d "$d" ]] && names+=("$(basename "$d")"); done
  fi
  for name in "${names[@]}"; do
    [[ -n "$name" && -d "$RUN_OUT/$name" ]] || continue
    if finish_episode_dir "$RUN_OUT/$name" "" "$PUB_ROOT" "$name"; then
      echo "REC_SYNCED policy=$LABEL side=orig dir=$name mode=$mode"
    else
      fail=$((fail + 1)); echo "REC_SYNC_FAIL policy=$LABEL side=orig dir=$name mode=$mode"
    fi
  done
  [[ "$mode" == "final" ]] || return $(( fail > 0 ))
  left=0
  for d in "$RUN_OUT"/*.a*/; do [[ -d "$d" ]] && left=$((left + 1)); done
  read -r n bytes < <(awk '{n += $1; b += $2} END {printf "%d %d\n", n, b}' "$TALLY" 2>/dev/null || echo "0 0")
  ok="$(tc_count ok)"; mism="$(tc_count frame_mismatch)"; tcf="$(tc_count fail)"
  if (( fail == 0 && left == 0 && mism == 0 && tcf == 0 )); then
    echo "SEAT_REC_SYNC=PASS n=${n:-0} bytes=${bytes:-0} left=0 seat=$SEAT side=orig transcoded=$ok frame_mismatch=0 transcode_fail=0"
    return 0
  fi
  echo "SEAT_REC_SYNC=FAIL n=${n:-0} bytes=${bytes:-0} left=$left seat=$SEAT side=orig failed=$fail transcoded=$ok frame_mismatch=$mism transcode_fail=$tcf"
  return 1
}

orig_loop() {  # 分轮调度；返回 0 完成（含 missing 由调用方判）/3 阻塞/4 基础设施用尽
  local round=0 server_restarts=0 noprog_restarts=0 runner_fails=0 plan attempt keys rc idle limit first rows0 rows
  local max_rounds=$((3 + MAX_CLIENT_RESTARTS + MAX_SERVER_RESTARTS)) last_sync
  while true; do
    plan="$(plan_round plan)"; echo "$plan"
    attempt="$(plan_field attempt "$plan")"; keys="$(plan_field keys "$plan")"
    [[ "$attempt" =~ ^[0-9]+$ ]] || { echo "INFRA plan_failed"; return 4; }
    (( attempt == 0 )) && return 0
    round=$((round + 1))
    (( round > max_rounds )) && { echo "INFRA_EXHAUSTED policy=$LABEL side=orig rounds=$round"; return 4; }
    if (( round > 1 )) || [[ -z "$SERVER_PID" ]]; then
      # 基础设施重试：先重启服务（新进程内固定 sid 重新从 n=0 开始），再重发同一身份
      stop_server
      restart_server "$POLICY" "$PORT" "$ORIG_STATE/server-$PORT-$(ts).log" "$RUN_OUT" || return $?
    fi
    touch "$RUN_OUT/.round-start"
    rows0=$(wc -l < "$RUN_OUT/results.jsonl" 2>/dev/null || echo 0)
    start_runner "$attempt" "$keys"
    first=1 ; rc=0 ; last_sync=$(ts)
    while true; do
      if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        wait "$RUNNER_PID"; rc=$?; RUNNER_PID=""
        echo "ORIG_RUNNER_EXIT policy=$LABEL attempt=$attempt rc=$rc"
        break
      fi
      if [[ -n "$SERVER_PID" ]] && ! server_alive; then
        echo "SERVER_DIED policy=$POLICY side=orig pid=$SERVER_PID"; SERVER_PID=""
        stop_runner; rc=-1
        server_restarts=$((server_restarts + 1))
        (( server_restarts > MAX_SERVER_RESTARTS )) && { echo "INFRA_EXHAUSTED policy=$LABEL side=orig server_restarts=$server_restarts"; return 4; }
        break
      fi
      rows=$(wc -l < "$RUN_OUT/results.jsonl" 2>/dev/null || echo 0)
      (( rows > rows0 )) && first=0  # 本轮第一局写完后不再含首局放宽
      limit="$(noprog_limit "$POLICY" "$first")"
      idle="$(idle_s "$RUN_OUT/results.jsonl" "$RUN_OUT/.round-start")"
      if (( idle > limit )); then
        echo "NO_PROGRESS policy=$LABEL side=orig idle_s=$idle limit_s=$limit restarts=$noprog_restarts"
        stop_runner; stop_server; rc=-1
        (( noprog_restarts >= 1 )) && { echo "INFRA_EXHAUSTED policy=$LABEL side=orig no_progress_twice"; return 4; }
        noprog_restarts=$((noprog_restarts + 1))
        break
      fi
      if (( $(ts) - last_sync >= SYNC_INTERVAL )); then sync_orig periodic; last_sync=$(ts); fi
      sleep "$SEAT_POLL_S" & wait $!
    done
    case "$rc" in
      0|-1) ;;
      2) server_restarts=$((server_restarts + 1))
         echo "ORIG_SERVER_UNREACHABLE policy=$LABEL restarts=$server_restarts"
         (( server_restarts > MAX_SERVER_RESTARTS )) && { echo "INFRA_EXHAUSTED policy=$LABEL side=orig server_restarts=$server_restarts"; return 4; };;
      3) echo "RUN_BLOCKED reason=official_runner policy=$LABEL（驱动退出 3：分片、导入断言或账本配置）"; return 3;;
      5) echo "RUN_BLOCKED reason=budget policy=$LABEL（驱动退出 5：共享账本轨迹额度不足）"; return 5;;
      *) runner_fails=$((runner_fails + 1))
         (( runner_fails > MAX_CLIENT_RESTARTS )) && { echo "INFRA_EXHAUSTED policy=$LABEL side=orig runner_fails=$runner_fails"; return 4; };;
    esac
    sync_orig periodic
  done
}

official_finalize() {  # $1 = outcome；$2 = rc
  local outcome="$1" rc="$2" plan missing
  (( FINALIZED == 1 )) && return
  FINALIZED=1
  trap '' TERM INT HUP
  stop_runner
  stop_server
  if [[ -n "$RUN_OUT" && -d "$RUN_OUT" && -f "$SHARD" ]]; then
    plan="$(plan_round mark)"; echo "$plan"
    missing="$(plan_field missing "$plan")"
    if (( rc == 0 )) && [[ "$missing" =~ ^[0-9]+$ ]] && (( missing > 0 )); then
      echo "RUN_INCOMPLETE side=orig policy=$LABEL seat=$SEAT dataset=$DATASET missing=$missing"
      rc=6; outcome="fail"
    fi
    if ! sync_orig final; then
      [[ "$outcome" == "pass" ]] && outcome="fail"
      (( rc == 0 )) && rc=7
    fi
  fi
  echo "OFFICIAL_SEAT_DONE seat=$SEAT policy=$LABEL outcome=$outcome rc=$rc dataset=${DATASET:-unset} run_name=$RUN_NAME host=$(hostname) $(date -Is)"
  echo "EXIT_CODE=$rc"
  exit "$rc"
}

official_on_signal() {
  local r=143
  case "$1" in INT) r=130;; HUP) r=129;; esac
  echo "OFFICIAL_SEAT_SIGNAL sig=$1 seat=$SEAT $(date -Is)"
  official_finalize aborted "$r"
}

official_entry() {
  parse_official_args "$@"
  if ! mkdir -p "$ORIG_STATE" "$RUN_OUT"; then
    echo "RUN_BLOCKED reason=mkdir state=$ORIG_STATE local=$RUN_OUT"
    echo "OFFICIAL_SEAT_DONE seat=$SEAT policy=$LABEL outcome=fail rc=3 reason=mkdir"
    echo "EXIT_CODE=3"
    exit 3
  fi
  TALLY="$LOCAL_ROOT/orig-sync-tally.txt"
  TC_TALLY="$LOCAL_ROOT/orig-transcode-tally.txt"
  : > "$TALLY"; : > "$TC_TALLY"
  trap 'official_on_signal TERM' TERM
  trap 'official_on_signal INT' INT
  trap 'official_on_signal HUP' HUP
  trap '' PIPE
  exec > >(trap '' TERM INT HUP PIPE; exec tee -p -a "$ORIG_STATE/official-s$SEAT.log") 2>&1
  : > "$OUT/.v8-pgids"
  echo "OFFICIAL_SEAT_START run_name=$RUN_NAME seat=$SEAT idx=$SEAT_IDX gpu=$GPU cpus=${CPUS:-all} policy=$POLICY label=$LABEL \
dataset=${DATASET:-unset} max_steps=${MAX_STEPS:-unset} strict_cap=$STRICT_CAP variant=${GROUNDSG_VARIANT:-none} \
policy_seed=$POLICY_SEED memer_adapter=${MEMER_ADAPTER:-none} budget_ledger=$BUDGET_LEDGER trajectory_cap=$TRAJECTORY_CAP \
shared_infra_cap=$SHARED_INFRA_CAP expired_cap=$EXPIRED_CAP planned_first_tries=$PLANNED_FIRST_TRIES \
runner=$(runner_script_of "$POLICY") repo=$REPO git=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) state=$ORIG_STATE \
local=$RUN_OUT publish=$PUB_ROOT infra_retry_budget=$INFRA_RETRY_BUDGET limit=$LIMIT client_py_ext=$SGEVAL_CLIENT_PY \
pp_py=$PP_PY no_proxy=127.0.0.1,localhost hf_home=${HF_HOME:-unset} host=$(hostname) $(date -Is)"
  step_cap_pairing || official_finalize fail 3
  [[ "$DATASET" == "hard-verify" ]] || { echo "RUN_BLOCKED reason=orig_dataset dataset=$DATASET（原侧只跑 hard-verify）"; official_finalize fail 3; }
  variant_pairing "$POLICY" || official_finalize fail 3
  case "$POLICY" in
    groundsg) tokenizer_gate || official_finalize fail 3; preflight_mme_vla groundsg || official_finalize fail 3;;
    pp) preflight_pp || official_finalize fail 3;;
  esac
  [[ -x "$SGEVAL_CLIENT_PY" ]] || { echo "RUN_BLOCKED reason=client_py_missing py=$SGEVAL_CLIENT_PY"; official_finalize fail 3; }
  [[ -f "$SHARD" ]] || { echo "RUN_BLOCKED reason=shard_missing $SHARD"; official_finalize fail 3; }
  [[ -f "$REPO/dev-scripts/orig/$(runner_script_of "$POLICY")" ]] \
    || { echo "RUN_BLOCKED reason=runner_missing $(runner_script_of "$POLICY")"; official_finalize fail 3; }
  local imp
  imp="$( cd "$REPO" && env "${NOPROXY_ENV[@]}" "$SGEVAL_CLIENT_PY" "dev-scripts/orig/$(runner_script_of "$POLICY")" --check-imports 2>&1 )" \
    || { echo "$imp" | tail -n 5; echo "RUN_BLOCKED reason=official_imports"; official_finalize fail 3; }
  echo "$imp" | tail -n 1
  # 续跑：节点上没有结果文件时从 NFS 副本恢复（含 server-epochs.tsv）
  if [[ ! -f "$RUN_OUT/results.jsonl" && -f "$ORIG_STATE/results.jsonl" ]]; then
    cp -f "$ORIG_STATE/results.jsonl" "$RUN_OUT/results.jsonl"
    [[ -f "$ORIG_STATE/server-epochs.tsv" ]] && cp -f "$ORIG_STATE/server-epochs.tsv" "$RUN_OUT/server-epochs.tsv"
    echo "ORIG_RESUME results=$(wc -l < "$RUN_OUT/results.jsonl")"
  fi
  local base pidx rc
  pidx="$(pol_index "$POLICY")"
  base=$((18000 + 100 * SEAT_IDX + 10 * pidx))
  PORT="$(pick_port "$base")" || { echo "INFRA port_exhausted base=$base"; official_finalize fail 4; }
  ckpt_fingerprint "$(ckpt_of "$POLICY")" > "$ORIG_STATE/ckpt-fingerprint.txt" 2>&1 &
  local fp_pid=$!
  start_server "$POLICY" "$PORT" "$ORIG_STATE/server-$PORT-$(ts).log" "$RUN_OUT"; rc=$?
  wait "$fp_pid"; cat "$ORIG_STATE/ckpt-fingerprint.txt"
  if (( rc != 0 )); then official_finalize fail "$( (( rc == 3 )) && echo 3 || echo 4 )"; fi
  orig_loop; rc=$?
  if (( rc == 0 )); then official_finalize pass 0; else official_finalize fail "$rc"; fi
}

# 直接执行才进入口；被 source 时只提供函数（供测试核对驱动命令）
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  official_entry "$@"
fi
