#!/usr/bin/env bash
# 评估新侧：GL 计算节点内跑一个席位（V8 起用，1001-v8-post-evaluation-gl-plan.md §2／§5、契约 C3；
# 1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.6 扩到 groundsg／pp 与两个数据集、就地转码）。
#
# 已在占位 job 的 srun 步骤里（由主会话在 GL 登录节点 tmux 内发起），本脚本：
#   1. 解释器（调用方环境变量优先，未设才取缺省；1005 计划审计第 1 条）：BENCH_PY 缺省 <repo>/.venv/bin/python、
#      MME_VLA_PY 缺省 <repo>/third_party/mme-vla/.venv/bin/python、SMVLA_PY 缺省
#      <repo>/artifacts/v8-two/venvs/smvla-env/bin/python；groundsg／pp 客户端用 SGEVAL_CLIENT_PY（缺省
#      <repo>/artifacts/sg-evaluation/venvs/client-env/bin/python），pp 服务用 PP_PY（缺省
#      <repo>/third_party/PonderPounce/.venv/bin/python）。起跑打印五个解释器、BENCH_PY 下 robomme_hard.__file__、
#      各解释器 venv 的 editable 安装指向（RUN_INPUT_PY 行），写 RUN_INPUTS=PASS|FAIL 行；本席策略要用的解释器
#      不可执行、robomme_hard 导入失败、官方重绘器不支持 --source、pp 外壳缺失任一即 FAIL →
#      RUN_BLOCKED reason=run_inputs。导出 SGEVAL_OFFICIAL_RENDER=1（每局并入轨迹后、转码前出官方版式视频，失败保留
#      原始帧照常发布，见 run_seat.sh::finish_episode_dir）与 SGEVAL_PP_SERVER_WRAP=1（pp 服务走
#      pp_server_wrap.py）；两者调用方显式设了别的值时照用。VK_ICD_FILENAMES 按候选 ICD 文件逐个探测；
#      --cpus 取本进程 sched_getaffinity；--gpu 缺省 0（GL 占位 job 内只见本席一张卡；本机多卡并行时显式给物理卡号）；端口按 run_seat.sh 既有规则（seat-idx = 席号 NN）。
#   2. 起跑打印并核对 --dataset 与 --max-steps 的配对（hard-verify↔1300 且不带 --strict-cap，ood↔1800 且必须带
#      --strict-cap），不符 RUN_BLOCKED reason=step_cap_pairing；groundsg 的变体配对（含 ground-sg-memer↔--memer-adapter）
#      同 run_seat.sh。第三阶段（接口冻结说明 2.2／2.4）：--policy-seed 必填（缺失 RUN_BLOCKED reason=policy_seed）；
#      --budget-ledger --trajectory-cap --shared-infra-cap --expired-cap --planned-first-tries 必填（缺任一
#      RUN_BLOCKED reason=budget_args，不回落 budget_ledger.py 常量默认值），连同种子原样转发给 run_seat.sh →
#      env_client.py run；--reset-budget 改为可选（不给只计量）。三项执行期限 --context-deadline-s
#      --first-infer-deadline-s --media-deadline-s 可选、原样转发。
#   3. 按策略顺序（同卡绝不同时驻留）各调一次 run_seat.sh：持久状态（results.jsonl、<label>.ledger.jsonl、
#      results.epochs.jsonl、server-epochs.tsv、progress.json、client.log、seat 日志）直接写 NFS <stage>/sNN/<label>/
#      （label 为策略名，groundsg 为 groundsg-<variant>；不同数据集请用不同 --stage，run_seat.sh 见到别的数据集的结果行
#      即 RUN_BLOCKED reason=dataset_crossed）；录像写节点本地 <local-root>/rec/<label>/，轨迹写
#      <local-root>/trace/<label>/（默认 local-root=/tmp/<R>-sNN）。一律带 --never-degrade，不接受 --no-record。
#   4. 后台同步循环每 --sync-interval 秒（默认 120）处理「该 label results.jsonl 已有行的 rec_dir 指向它、且目录内
#      summary.json 已写出」的每局目录：并入同名轨迹目录（trace.jsonl 等，不含 qwen-tmp）→ 就地转码为 episode.mp4
#      （前视与腕部左右拼接；libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart，30 fps，参数同
#      scripts/injection-dev/site/eval_transcode.py）→ 完整解码数帧、与录像器写出的帧数一致才删节点原始帧（*.mkv、
#      .spool/）→ 原子发布到 NFS 输出键 <media-root>/<label>/<dataset>/new/<key>.a<attempt>/（media-root 缺省
#      <stage>/media）：先 rsync 到 <…>/new/.incoming/<目录名>/，逐文件 sha256 一致后 mv（目标已存在则落
#      <目录名>.dupN，不覆盖），再删节点副本。转码不成功的目录原始帧照样发布（不丢数据），计入 frame_mismatch／
#      transcode_fail。
#   5. 正常、失败、中断（trap TERM/INT/HUP）三路收尾：先回收子进程（等待 ≤25 s），再对节点上剩余的全部录像与轨迹
#      目录做全量处理，打印 SEAT_REC_SYNC=PASS|FAIL n= bytes= left= seat= transcoded= frame_mismatch= transcode_fail=
#      （n／bytes 为周期 + 收尾累计，left 为节点残留目录数；PASS 要求 left、frame_mismatch、transcode_fail 都为 0），
#      官方重绘另打一行 OFFICIAL_RENDER_TALLY ok= fail=（不进 SEAT_REC_SYNC 判定，失败局原始帧已随目录发布），
#      再打印 V8_SEAT_DONE seat=NN outcome=pass|fail|aborted rc=，末行 EXIT_CODE=<rc>。参数错误与目录创建失败也写
#      V8_SEAT_DONE outcome=fail。
#   信号：slurmstepd 会把 TERM/INT 发给 step 内全部进程，所以所有日志 tee 都忽略 TERM/INT/HUP/PIPE 并以 -p 运行，
#   主脚本忽略 PIPE，保证收尾三行在 tee 存活时写出（日志文件里一定有）。
#
# 用法（一席一行）：
#   bash <repo>/scripts/eval-official/run_eval_gl.sh --run-name R --seat NN --repo <NFS 执行副本> --stage <NFS 运行根> \
#     --shard <shard-NN.json> --dataset {ood,hard-verify} --max-steps N [--strict-cap] \
#     --policies smvla,perceptual-framesamp-modul,groundsg,pp [--groundsg-variant V] [--qwenvl-groundsg-adapter D] \
#     [--framesamp-modul-ckpt D] [--groundsg-ckpt D] [--smvla-ckpt D] [--pp-ckpt D] [--openpi-data-home D --tokenizer-sha256 H] \
#     --policy-seed N --budget-ledger P --trajectory-cap N --shared-infra-cap N --expired-cap N --planned-first-tries N \
#     [--memer-adapter D] [--reset-budget N] --infra-retry-budget N [--cond C] [--media-root D] [--limit N] \
#     [--context-deadline-s S] [--first-infer-deadline-s S] [--media-deadline-s S] \
#     [--episode-wall S] [--episode-wall-smvla S] [--episode-wall-framesamp-modul S] [--sync-interval S] [--local-root DIR] [--gpu N]
# 退出码：0 全部策略 rc=0 且录像同步 PASS；中断 130/143（HUP 129）；其余失败取首个非零策略 rc（录像同步 FAIL 且策略全 0
#   时为 7）；参数错误 2；执行环境缺失（解释器、shard 等）与配对核对不过 3。
# 不嵌入任何 JobID，不含 /data 默认路径。
set -uo pipefail
export PYTHONUNBUFFERED=1

# 调用方给的解释器覆盖先记下（source run_seat.sh 会按它自己的位置填缺省值；参数解析后再按 "${X:-缺省}" 取用）
_ENV_SGEVAL_CLIENT_PY="${SGEVAL_CLIENT_PY:-}" ; _ENV_PP_PY="${PP_PY:-}"
_ENV_BENCH_PY="${BENCH_PY:-}" ; _ENV_MME_VLA_PY="${MME_VLA_PY:-}" ; _ENV_SMVLA_PY="${SMVLA_PY:-}"
# 复用 run_seat.sh 的配对核对、转码与原子发布（只定义函数，不运行）
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_seat.sh
source "$HERE/run_seat.sh"

RUN_NAME="" ; SEAT="" ; REPO="" ; STAGE="" ; SHARD="" ; POLICIES="smvla,perceptual-framesamp-modul"
FRAMESAMP_MODUL_CKPT="" ; GROUNDSG_CKPT="" ; SMVLA_CKPT="" ; PP_CKPT="" ; OPENPI_HOME="" ; TOKENIZER_SHA="" ; RESET_BUDGET="" ; INFRA_RETRY_BUDGET=""
GPU="0" ; LIMIT="0" ; WALL_SMVLA="" ; WALL_FRAMESAMP_MODUL="" ; WALL_ALL="" ; SYNC_INTERVAL=120 ; LOCAL_ROOT="" ; COND="V8" ; MEDIA_ROOT=""
DATASET="" ; MAX_STEPS="" ; STRICT_CAP=0 ; GROUNDSG_VARIANT="" ; QWENVL_ADAPTER="" ; MEMER_ADAPTER=""
POLICY_SEED="" ; BUDGET_LEDGER="" ; TRAJECTORY_CAP="" ; SHARED_INFRA_CAP="" ; EXPIRED_CAP="" ; PLANNED_FIRST_TRIES=""
CONTEXT_DEADLINE="" ; FIRST_INFER_DEADLINE="" ; MEDIA_DEADLINE=""

usage() { sed -n '2,/^# 不嵌入任何 JobID/p' "${BASH_SOURCE[0]}" >&2; }
die2() {  # 参数错误：也写收尾两行，便于外层 Monitor 统一判定
  echo "$1" >&2
  echo "V8_SEAT_DONE seat=${SEAT:-?} outcome=fail rc=2 reason=bad_args"
  echo "EXIT_CODE=2"
  exit 2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2;;
    --seat) SEAT="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --stage) STAGE="$2"; shift 2;;
    --media-root) MEDIA_ROOT="$2"; shift 2;;
    --shard) SHARD="$2"; shift 2;;
    --policies) POLICIES="$2"; shift 2;;
    --cond) COND="$2"; shift 2;;
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
    --context-deadline-s) CONTEXT_DEADLINE="$2"; shift 2;;
    --first-infer-deadline-s) FIRST_INFER_DEADLINE="$2"; shift 2;;
    --media-deadline-s) MEDIA_DEADLINE="$2"; shift 2;;
    --framesamp-modul-ckpt) FRAMESAMP_MODUL_CKPT="$2"; shift 2;;
    --groundsg-ckpt) GROUNDSG_CKPT="$2"; shift 2;;
    --smvla-ckpt) SMVLA_CKPT="$2"; shift 2;;
    --pp-ckpt) PP_CKPT="$2"; shift 2;;
    --openpi-data-home) OPENPI_HOME="$2"; shift 2;;
    --tokenizer-sha256) TOKENIZER_SHA="$2"; shift 2;;
    --reset-budget) RESET_BUDGET="$2"; shift 2;;
    --infra-retry-budget) INFRA_RETRY_BUDGET="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --episode-wall) WALL_ALL="$2"; shift 2;;
    --episode-wall-smvla) WALL_SMVLA="$2"; shift 2;;
    --episode-wall-framesamp-modul) WALL_FRAMESAMP_MODUL="$2"; shift 2;;
    --sync-interval) SYNC_INTERVAL="$2"; shift 2;;
    --local-root) LOCAL_ROOT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --no-record) die2 "run_eval_gl.sh 禁止 --no-record（全部视频保留）";;
    -h|--help) usage; exit 0;;
    *) usage; die2 "未知参数 $1";;
  esac
done
[[ -n "$RUN_NAME" && -n "$SEAT" && -n "$REPO" && -n "$STAGE" && -n "$SHARD" ]] \
  || die2 "缺少必需参数（--run-name --seat --repo --stage --shard）"
# --dataset／--max-steps 缺失或配错不在这里拦：起跑后由 step_cap_pairing 打印 RUN_BLOCKED reason=step_cap_pairing
[[ -z "$MAX_STEPS" || "$MAX_STEPS" =~ ^[0-9]+$ ]] || die2 "--max-steps 须为非负整数"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die2 "--run-name 只许字母数字 . _ -"
[[ "$COND" =~ ^[A-Za-z0-9._-]+$ ]] || die2 "--cond 只许字母数字 . _ -"
[[ "$SEAT" =~ ^[0-9]{2}$ ]] || die2 "--seat 须为两位席号 NN（如 03）"
[[ "$INFRA_RETRY_BUDGET" =~ ^[0-9]+$ ]] || die2 "--infra-retry-budget 必填且为非负整数"
[[ -z "$RESET_BUDGET" || "$RESET_BUDGET" =~ ^[0-9]+$ ]] || die2 "--reset-budget 须为非负整数（可不给：只计量）"
for _d in "$CONTEXT_DEADLINE" "$FIRST_INFER_DEADLINE" "$MEDIA_DEADLINE"; do
  [[ -z "$_d" || "$_d" =~ ^[0-9]+$ ]] || die2 "--*-deadline-s 须为非负整数秒"
done
[[ "$LIMIT" =~ ^[0-9]+$ && "$SYNC_INTERVAL" =~ ^[0-9]+$ ]] || die2 "--limit／--sync-interval 须为非负整数"
IFS=',' read -r -a POLS <<< "$POLICIES"
for pol in "${POLS[@]}"; do
  case "$pol" in
    smvla) [[ -n "$SMVLA_CKPT" ]] || die2 "跑 smvla 须给 --smvla-ckpt";;
    perceptual-framesamp-modul) [[ -n "$FRAMESAMP_MODUL_CKPT" && -n "$OPENPI_HOME" && -n "$TOKENIZER_SHA" ]] \
           || die2 "跑 perceptual-framesamp-modul 须给 --framesamp-modul-ckpt --openpi-data-home --tokenizer-sha256";;
    groundsg) [[ -n "$GROUNDSG_CKPT" && -n "$OPENPI_HOME" && -n "$TOKENIZER_SHA" ]] \
           || die2 "跑 groundsg 须给 --groundsg-ckpt --openpi-data-home --tokenizer-sha256";;
    pp) [[ -n "$PP_CKPT" ]] || die2 "跑 pp 须给 --pp-ckpt";;
    *) die2 "未知策略 $pol（只许 smvla,perceptual-framesamp-modul,groundsg,pp）";;
  esac
done
SEAT_IDX=$((10#$SEAT))
[[ -n "$LOCAL_ROOT" ]] || LOCAL_ROOT="/tmp/$RUN_NAME-s$SEAT"
[[ -n "$MEDIA_ROOT" ]] || MEDIA_ROOT="$STAGE/media"
REC_LOCAL="$LOCAL_ROOT/rec"
TRACE_LOCAL="$LOCAL_ROOT/trace"
SEAT_STAGE="$STAGE/s$SEAT"
LABELS=()
for pol in "${POLS[@]}"; do LABELS+=("$(pol_label "$pol")"); done

export BENCH_PY="${_ENV_BENCH_PY:-$REPO/.venv/bin/python}"
export MME_VLA_PY="${_ENV_MME_VLA_PY:-$REPO/third_party/mme-vla/.venv/bin/python}"
export SMVLA_PY="${_ENV_SMVLA_PY:-$REPO/artifacts/v8-two/venvs/smvla-env/bin/python}"
export SGEVAL_CLIENT_PY="${_ENV_SGEVAL_CLIENT_PY:-$REPO/artifacts/sg-evaluation/venvs/client-env/bin/python}"
export PP_PY="${_ENV_PP_PY:-$REPO/third_party/PonderPounce/.venv/bin/python}"
# 预算账本转绝对路径（run_seat.sh 会在别的目录下起客户端）
[[ -n "$BUDGET_LEDGER" ]] && BUDGET_LEDGER="$(mkdir -p "$(dirname "$BUDGET_LEDGER")" && cd "$(dirname "$BUDGET_LEDGER")" && pwd)/$(basename "$BUDGET_LEDGER")"
# 第二阶段开关（run_seat.sh 文档串）：官方重绘、pp 服务外壳；调用方显式给了别的值时照用
export SGEVAL_OFFICIAL_RENDER="${SGEVAL_OFFICIAL_RENDER:-1}"
export SGEVAL_PP_SERVER_WRAP="${SGEVAL_PP_SERVER_WRAP:-1}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
for f in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  [[ -f "$f" ]] && { export VK_ICD_FILENAMES="$f"; break; }
done
# 录像降级只看此盘空间；--never-degrade 下恒无损，此处指向节点本地录像根以免探测不存在的路径
export V75_DATA_ROOT="$LOCAL_ROOT"

ts_iso() { date -Is; }
RUN_SEAT_PID="" ; SYNC_PID="" ; ABORTED=0 ; ABORT_RC=0 ; FINALIZED=0

# ---------------- 录像同步 ----------------
# 列出 $1（results.jsonl）中 rec_dir 的目录名（basename），每行一个
rec_names_in_results() {
  [[ -f "$1" ]] || return 0
  "$BENCH_PY" - "$1" <<'PY'
import json, os, sys
names = set()
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = r.get("rec_dir") if isinstance(r, dict) else None
        if d:
            names.add(os.path.basename(str(d).rstrip("/")))
for n in sorted(names):
    print(n)
PY
}

pub_root() { echo "$MEDIA_ROOT/$1/$DATASET/new"; }  # $1 = label

or_count() {  # $1 = ok|fail；从 OR_TALLY 计数（官方重绘结果）
  [[ -n "${OR_TALLY:-}" && -f "$OR_TALLY" ]] || { echo 0; return; }
  grep -cx "$1" "$OR_TALLY" || true
}

# ---------------- 起跑输入核对（RUN_INPUTS） ----------------
editable_of() {  # $1 = 解释器；打印该 venv 里 editable 安装的「包名=源路径」，分号分隔；没有打印 none
  "$1" - <<'PY' 2>/dev/null || echo "probe_failed"
import json
try:
    import importlib.metadata as md
except ImportError:  # pragma: no cover
    print("no_importlib_metadata")
    raise SystemExit(0)
out = []
for d in md.distributions():
    try:
        t = d.read_text("direct_url.json")
        j = json.loads(t) if t else {}
    except Exception:
        continue
    if (j.get("dir_info") or {}).get("editable"):
        out.append(f"{d.metadata['Name']}={str(j.get('url', '')).replace('file://', '', 1)}")
print(";".join(sorted(set(out))) or "none")
PY
}

run_inputs_check() {  # 打印五个解释器与 editable 指向、robomme_hard 位置，写 RUN_INPUTS= 行；FAIL 返回 1
  local why=() rh role py ed need=" bench " renderer="off" wrap="off" pol
  for pol in "${POLS[@]}"; do
    case "$pol" in
      smvla) need+=" smvla ";;
      perceptual-framesamp-modul) need+=" mme-vla policy-wrap ";;
      groundsg) need+=" mme-vla policy-wrap client ";;
      pp) need+=" client pp ";;
    esac
  done
  for role in bench mme-vla smvla client pp; do
    case "$role" in
      bench) py="$BENCH_PY";; mme-vla) py="$MME_VLA_PY";; smvla) py="$SMVLA_PY";; client) py="$SGEVAL_CLIENT_PY";; pp) py="$PP_PY";;
    esac
    if [[ -x "$py" ]]; then ed="$(editable_of "$py" | tail -n 1)"; else ed="missing"; fi
    echo "RUN_INPUT_PY role=$role py=$py needed=$([[ "$need" == *" $role "* ]] && echo 1 || echo 0) editable=$ed"
    [[ "$need" == *" $role "* && ! -x "$py" ]] && why+=("${role}_py_missing")
  done
  rh="$("$BENCH_PY" -c "import robomme_hard,sys;print(robomme_hard.__file__)" 2>/dev/null | tail -n 1)"
  [[ "$rh" == */robomme_hard/__init__.py ]] || { why+=("robomme_hard_import"); rh="IMPORT_FAIL"; }
  if [[ "$SGEVAL_OFFICIAL_RENDER" == "1" ]]; then
    renderer="$REPO/scripts/eval-official/render_official_video.py"
    if [[ ! -f "$renderer" ]]; then why+=("renderer_missing"); renderer="missing"
    elif ! "$BENCH_PY" "$renderer" --help 2>/dev/null | grep -q -- '--source'; then why+=("renderer_no_source_raw"); renderer="no_source_raw"
    else renderer="source_raw"; fi
    [[ -f "$REPO/scripts/eval-official/official_media_check.py" ]] || why+=("media_check_missing")
  fi
  if [[ "$need" == *" policy-wrap "* && ! -f "$REPO/scripts/eval-official/policy_server_wrap.py" ]]; then
    why+=("policy_server_wrap_missing")
  fi
  if [[ "$SGEVAL_PP_SERVER_WRAP" == "1" && "$need" == *" pp "* ]]; then
    wrap="$REPO/scripts/eval-official/pp_server_wrap.py"
    if [[ -f "$wrap" ]]; then wrap="present"; else why+=("pp_server_wrap_missing"); wrap="missing"; fi
  fi
  local verdict=PASS; (( ${#why[@]} == 0 )) || verdict=FAIL
  echo "RUN_INPUTS=$verdict bench_py=$BENCH_PY mme_vla_py=$MME_VLA_PY smvla_py=$SMVLA_PY client_py=$SGEVAL_CLIENT_PY pp_py=$PP_PY \
robomme_hard=$rh official_render=$SGEVAL_OFFICIAL_RENDER renderer=$renderer pp_server_wrap=$SGEVAL_PP_SERVER_WRAP \
pp_wrap_file=$wrap policies=$POLICIES${why:+ reason=$(IFS=,; echo "${why[*]}")}"
  [[ "$verdict" == PASS ]]
}

left_on_node() {
  local lab left=0 d
  for lab in "${LABELS[@]}"; do
    for d in "$REC_LOCAL/$lab" "$TRACE_LOCAL/$lab"; do
      [[ -d "$d" ]] && left=$((left + $(find "$d" -mindepth 1 -maxdepth 1 -type d | wc -l)))
    done
  done
  echo "$left"
}

# $1 = periodic|final。periodic 只处理「结果行已指向且 summary.json 已写出」的目录；final 处理节点上剩余全部录像与
# 轨迹目录。返回 0（本轮全部成功且 final 时节点无残留、无帧数不符）/1。final 时打印 SEAT_REC_SYNC 判定行。
sync_recordings() {
  local mode="$1" lab name fail=0 left=0 n bytes ok mism tcf
  for lab in "${LABELS[@]}"; do
    local -A names=()
    if [[ "$mode" == "periodic" ]]; then
      [[ -d "$REC_LOCAL/$lab" ]] || continue
      while IFS= read -r name; do
        [[ -n "$name" && -f "$REC_LOCAL/$lab/$name/summary.json" ]] && names["$name"]=1
      done < <(rec_names_in_results "$SEAT_STAGE/$lab/results.jsonl")
    else
      for d in "$REC_LOCAL/$lab"/*/ "$TRACE_LOCAL/$lab"/*/; do
        [[ -d "$d" ]] && names["$(basename "$d")"]=1
      done
    fi
    for name in "${!names[@]}"; do
      [[ "$mode" == "periodic" && "$SYNC_STOP" == 1 ]] && break  # 收尾已开始：做完手上这一局就停，余下交给 final
      if finish_episode_dir "$REC_LOCAL/$lab/$name" "$TRACE_LOCAL/$lab/$name" "$(pub_root "$lab")" "$name"; then
        echo "REC_SYNCED policy=$lab dir=$name mode=$mode"
      else
        fail=$((fail + 1)); echo "REC_SYNC_FAIL policy=$lab dir=$name mode=$mode"
      fi
    done
    unset names
  done
  if [[ "$mode" == "final" ]]; then
    left="$(left_on_node)"
    read -r n bytes < <(awk '{n += $1; b += $2} END {printf "%d %d\n", n, b}' "$TALLY" 2>/dev/null || echo "0 0")
    ok="$(tc_count ok)"; mism="$(tc_count frame_mismatch)"; tcf="$(tc_count fail)"
    if [[ "$SGEVAL_OFFICIAL_RENDER" == "1" ]]; then
      echo "OFFICIAL_RENDER_TALLY ok=$(or_count ok) fail=$(or_count fail) seat=$SEAT（失败局原始帧已随目录发布，不进 SEAT_REC_SYNC 判定）"
    fi
    if (( fail == 0 && left == 0 && mism == 0 && tcf == 0 )); then
      echo "SEAT_REC_SYNC=PASS n=${n:-0} bytes=${bytes:-0} left=0 seat=$SEAT transcoded=$ok frame_mismatch=0 transcode_fail=0"
    else
      echo "SEAT_REC_SYNC=FAIL n=${n:-0} bytes=${bytes:-0} left=$left seat=$SEAT failed=$fail transcoded=$ok frame_mismatch=$mism transcode_fail=$tcf"
    fi
    (( fail == 0 && left == 0 && mism == 0 && tcf == 0 ))
    return
  fi
  (( fail == 0 ))
}

SYNC_STOP=0
sync_loop() {  # 后台：周期同步；收到 TERM 时做完手上这一局再退出（不在转码／发布半途被打断），主流程随后做全量同步
  local sp=""
  trap 'SYNC_STOP=1; [[ -n "$sp" ]] && kill "$sp" 2>/dev/null' TERM INT
  while (( SYNC_STOP == 0 )); do
    sleep "$SYNC_INTERVAL" & sp=$!
    wait "$sp"; sp=""
    (( SYNC_STOP == 0 )) || break
    sync_recordings periodic
  done
  exit 0
}

stop_sync_loop() {
  [[ -n "$SYNC_PID" ]] || return 0
  kill -TERM "$SYNC_PID" 2>/dev/null
  wait "$SYNC_PID" 2>/dev/null
  SYNC_PID=""
}

# ---------------- 子进程回收 ----------------
# run_seat.sh 异常死亡（如被 KILL）时，其 setsid 起的 server／客户端进程组可能残留：按 .v8-pgids 回收，
# 只杀命令行确属本评估入口的进程组（防 PID 复用误杀）。
reap_orphans() {  # $1 = TERM 后宽限秒数；$2 = KILL 后等进程组退出并核显存释放的上限秒数
  local f="$SEAT_STAGE/.v8-pgids" grace="${1:-10}" post="${2:-30}" role pid cmd i
  [[ -f "$f" ]] || return 0
  while read -r role pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -0 "$pid" 2>/dev/null || continue
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    case "$cmd" in
      *env_client.py*|*smvla_server.py*|*serve_policy.py*|*policy_server_wrap.py*|*framesamp_modul_client.py*|*ponderpounce.eval.robomme_server*|*pp_server_wrap.py*)
        echo "REAP_ORPHAN role=$role pgid=$pid"
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        for i in $(seq 1 $((grace * 2))); do kill -0 -- "-$pid" 2>/dev/null || break; sleep 0.5; done
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        for i in $(seq 1 $((post * 2))); do
          if ! kill -0 -- "-$pid" 2>/dev/null && \
             ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$pid"; then
            break
          fi
          sleep 0.5
        done
        if kill -0 -- "-$pid" 2>/dev/null; then echo "REAP_ORPHAN_STUCK role=$role pgid=$pid"
        else echo "REAP_ORPHAN_GONE role=$role pgid=$pid"; fi;;
    esac
  done < "$f"
}

stop_run_seat() {  # 转发 TERM 给 run_seat.sh（它会收掉 server／客户端），最多等 25 s，超时连同其子进程 KILL
  [[ -n "$RUN_SEAT_PID" ]] || return 0
  if kill -0 "$RUN_SEAT_PID" 2>/dev/null; then
    kill -TERM "$RUN_SEAT_PID" 2>/dev/null
    local i; for i in $(seq 1 50); do kill -0 "$RUN_SEAT_PID" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$RUN_SEAT_PID" 2>/dev/null; then
      echo "RUN_SEAT_KILL pid=$RUN_SEAT_PID（25 s 未退出）"
      pkill -KILL -P "$RUN_SEAT_PID" 2>/dev/null
      kill -KILL "$RUN_SEAT_PID" 2>/dev/null
    fi
  fi
  wait "$RUN_SEAT_PID" 2>/dev/null
  RUN_SEAT_PID=""
}

finalize() {  # $1 = outcome；$2 = rc
  local outcome="$1" rc="$2"
  (( FINALIZED == 1 )) && return
  FINALIZED=1
  trap '' TERM INT HUP  # 收尾期间不再被打断
  stop_sync_loop
  stop_run_seat
  reap_orphans 3 10  # 收尾路径压短：TERM 宽限 3 s、KILL 后 ≤10 s
  if ! sync_recordings final; then
    [[ "$outcome" == "pass" ]] && outcome="fail"
    (( rc == 0 )) && rc=7
  fi
  echo "V8_SEAT_DONE seat=$SEAT outcome=$outcome rc=$rc run_name=$RUN_NAME dataset=$DATASET host=$(hostname) $(ts_iso)"
  echo "EXIT_CODE=$rc"
  exit "$rc"
}

on_signal() {  # $1 = 信号名
  ABORTED=1
  case "$1" in INT) ABORT_RC=130;; HUP) ABORT_RC=129;; *) ABORT_RC=143;; esac
  echo "V8_SEAT_SIGNAL sig=$1 seat=$SEAT $(ts_iso)"
  finalize aborted "$ABORT_RC"
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
trap '' PIPE  # 输出管道断了也不能让收尾被 SIGPIPE 杀掉

# ---------------- 主流程 ----------------
if ! mkdir -p "$SEAT_STAGE" "$REC_LOCAL" "$TRACE_LOCAL" "$MEDIA_ROOT"; then
  echo "RUN_BLOCKED reason=mkdir stage=$SEAT_STAGE local=$LOCAL_ROOT media=$MEDIA_ROOT"
  echo "V8_SEAT_DONE seat=$SEAT outcome=fail rc=3 reason=mkdir"
  echo "EXIT_CODE=3"
  exit 3
fi
TALLY="$LOCAL_ROOT/rec-sync-tally.txt"    # 周期 + 收尾同步的累计计数（每成功一个目录一行「1 <字节数>」）
TC_TALLY="$LOCAL_ROOT/transcode-tally.txt"  # 每局转码结果（ok／frame_mismatch／fail／none／…）一行
OR_TALLY="$LOCAL_ROOT/official-render-tally.txt"  # 每局官方重绘结果（ok／fail）一行；run_seat.sh::finish_episode_dir 写
export OR_TALLY
: > "$TALLY"
: > "$TC_TALLY"
: > "$OR_TALLY"
SEAT_LOG="$SEAT_STAGE/run_eval_gl-s$SEAT.log"
# tee 忽略 TERM/INT/HUP/PIPE 并以 -p 运行：slurmstepd 发给全组的信号不能先杀掉日志管道
exec > >(trap '' TERM INT HUP PIPE; exec tee -p -a "$SEAT_LOG") 2>&1

CPUS="$("$BENCH_PY" -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0)))))' 2>/dev/null)"
echo "V8_SEAT_START run_name=$RUN_NAME seat=$SEAT idx=$SEAT_IDX host=$(hostname) cpus=${CPUS:-?} repo=$REPO \
stage=$SEAT_STAGE media=$MEDIA_ROOT local=$LOCAL_ROOT shard=$SHARD policies=$POLICIES labels=${LABELS[*]} \
dataset=$DATASET max_steps=$MAX_STEPS strict_cap=$STRICT_CAP cond=$COND reset_budget=${RESET_BUDGET:-meter_only} \
policy_seed=${POLICY_SEED:-unset} budget_ledger=${BUDGET_LEDGER:-unset} caps=${TRAJECTORY_CAP:-?}/${SHARED_INFRA_CAP:-?}/${EXPIRED_CAP:-?}/${PLANNED_FIRST_TRIES:-?} \
infra_retry_budget=$INFRA_RETRY_BUDGET limit=$LIMIT sync_interval=$SYNC_INTERVAL vk_icd=${VK_ICD_FILENAMES:-unset} \
no_proxy=$NO_PROXY client_py_ext=$SGEVAL_CLIENT_PY pp_py=$PP_PY hf_home=${HF_HOME:-unset} $(ts_iso)"

step_cap_pairing || finalize fail 3
variant_pairing "$POLICIES" || finalize fail 3
policy_seed_check || finalize fail 3
budget_args_check || finalize fail 3
blocked=""
[[ -x "$BENCH_PY" ]] || blocked="bench_py_missing $BENCH_PY"
[[ -z "$blocked" && -n "$CPUS" ]] || blocked="${blocked:-cpus_unknown}"
[[ -z "$blocked" && -f "$SHARD" ]] || blocked="${blocked:-shard_missing $SHARD}"
[[ -z "$blocked" && -f "$REPO/scripts/eval-official/run_seat.sh" ]] || blocked="${blocked:-run_seat_missing}"
for pol in "${POLS[@]}"; do
  case "$pol" in
    groundsg|pp) [[ -n "$blocked" || -x "$SGEVAL_CLIENT_PY" ]] || blocked="client_py_missing $SGEVAL_CLIENT_PY";;
  esac
done
if [[ -n "$blocked" ]]; then
  echo "RUN_BLOCKED reason=$blocked"
  finalize fail 3
fi
if ! run_inputs_check; then
  echo "RUN_BLOCKED reason=run_inputs（见上一行 RUN_INPUTS=FAIL 的 reason）"
  finalize fail 3
fi

sync_loop &
SYNC_PID=$!

overall=0
for i in "${!POLS[@]}"; do
  pol="${POLS[$i]}"
  lab="${LABELS[$i]}"
  mkdir -p "$SEAT_STAGE/$lab" "$REC_LOCAL/$lab" "$TRACE_LOCAL/$lab"
  : > "$SEAT_STAGE/.v8-pgids"
  args=(--seat "$SEAT" --seat-idx "$SEAT_IDX" --gpu "$GPU" --cpus "$CPUS" --cond "$COND" --out "$SEAT_STAGE"
        --policies "$pol" --identities "$SHARD" --limit "$LIMIT" --never-degrade
        --dataset "$DATASET" --max-steps "$MAX_STEPS"
        --ledger-dir "$SEAT_STAGE/$lab" --infra-retry-budget "$INFRA_RETRY_BUDGET"
        --policy-seed "$POLICY_SEED" --budget-ledger "$BUDGET_LEDGER" --trajectory-cap "$TRAJECTORY_CAP"
        --shared-infra-cap "$SHARED_INFRA_CAP" --expired-cap "$EXPIRED_CAP" --planned-first-tries "$PLANNED_FIRST_TRIES"
        --rec-root "$REC_LOCAL" --trace-root "$TRACE_LOCAL")
  [[ -n "$RESET_BUDGET" ]] && args+=(--reset-budget "$RESET_BUDGET")
  [[ -n "$CONTEXT_DEADLINE" ]] && args+=(--context-deadline-s "$CONTEXT_DEADLINE")
  [[ -n "$FIRST_INFER_DEADLINE" ]] && args+=(--first-infer-deadline-s "$FIRST_INFER_DEADLINE")
  [[ -n "$MEDIA_DEADLINE" ]] && args+=(--media-deadline-s "$MEDIA_DEADLINE")
  (( STRICT_CAP == 1 )) && args+=(--strict-cap)
  [[ -n "$WALL_ALL" ]] && args+=(--episode-wall "$WALL_ALL")
  [[ -n "$WALL_SMVLA" ]] && args+=(--episode-wall-smvla "$WALL_SMVLA")
  [[ -n "$WALL_FRAMESAMP_MODUL" ]] && args+=(--episode-wall-framesamp-modul "$WALL_FRAMESAMP_MODUL")
  case "$pol" in
    perceptual-framesamp-modul) args+=(--framesamp-modul-ckpt "$FRAMESAMP_MODUL_CKPT" --openpi-data-home "$OPENPI_HOME" --tokenizer-sha256 "$TOKENIZER_SHA");;
    groundsg) args+=(--groundsg-ckpt "$GROUNDSG_CKPT" --openpi-data-home "$OPENPI_HOME" --tokenizer-sha256 "$TOKENIZER_SHA"
                  --groundsg-variant "$GROUNDSG_VARIANT")
           [[ -n "$QWENVL_ADAPTER" ]] && args+=(--qwenvl-groundsg-adapter "$QWENVL_ADAPTER")
           [[ -n "$MEMER_ADAPTER" ]] && args+=(--memer-adapter "$MEMER_ADAPTER");;
    pp) args+=(--pp-ckpt "$PP_CKPT");;
    *) args+=(--smvla-ckpt "$SMVLA_CKPT");;
  esac
  echo "V8_POLICY_START seat=$SEAT policy=$lab dataset=$DATASET $(ts_iso)"
  bash "$REPO/scripts/eval-official/run_seat.sh" "${args[@]}" \
    > >(trap '' TERM INT HUP PIPE; exec tee -p -a "$SEAT_STAGE/$lab/seat-$SEAT.log") 2>&1 &
  RUN_SEAT_PID=$!
  while true; do
    wait "$RUN_SEAT_PID"; rc=$?
    kill -0 "$RUN_SEAT_PID" 2>/dev/null || break
  done
  RUN_SEAT_PID=""
  echo "V8_POLICY_DONE seat=$SEAT policy=$lab rc=$rc $(ts_iso)"
  if (( rc > 128 )); then
    echo "V8_SUPERVISOR_DIED seat=$SEAT policy=$lab rc=$rc（run_seat.sh 被信号终止，回收残留子进程）"
    reap_orphans 10 30  # KILL 后等进程组退出并核显存释放（≤30 s）再起下一个策略
  fi
  (( rc != 0 && overall == 0 )) && overall=$rc
done

if (( overall == 0 )); then finalize pass 0; else finalize fail "$overall"; fi
