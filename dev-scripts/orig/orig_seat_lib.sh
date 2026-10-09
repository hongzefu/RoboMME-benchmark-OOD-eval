#!/usr/bin/env bash
# 原侧对照的席位函数库（拆仓后由旧仓评估目录的席位执行器 run_seat.sh改名迁来，只作库 `source`，不再直接执行）。
#
# 1008 拆分方案：新侧席位已改为 dev-scripts/gl/seat.py（常驻 Policy + 动态队列，服务端由 Python ServerProcess 起停），
# 旧席位执行器不迁入；但原侧对照（dev-scripts/orig/run_official_hard.sh、pair_seat.sh）照旧复用这里的服务端起停、
# 端口挑选、配对核对、无进展检测与媒体收尾函数，语义不改。拆仓只改路径：服务外壳在 src/robomme_ood_eval/servers/、
# 媒体函数库在 dev-scripts/gl/seat_media_lib.sh；新侧客户端构造（build_client_cmd）已停用，调用即 RUN_BLOCKED。
# 以下为原文件说明（路径已按拆仓更新）。
#
# 评估席位执行器（V9／hard-verify 评估经 run_eval_gl.sh 调用；最初为 v7.5eval 编写，0929-v7.5eval-restructure-plan.md §5；
# 1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.6 扩到四个策略与两个数据集）。
#
# 一个席位（一张 GPU）上按策略顺序逐个：起 server → 等就绪 → 起常驻客户端（env_client.py run）→ 看门狗 → 收 server；
# 同一张卡上两个策略的 server 绝不同时驻留。
#
# 用法：
#   run_seat.sh --seat 甲 --seat-idx 1 --gpu 0 --cond N --out <dir> --identities <shard-NN.json>
#               --dataset {ood,hard-verify} --max-steps N [--strict-cap]
#               --ledger-dir <dir> [--reset-budget N] --infra-retry-budget N --policy-seed N
#               --budget-ledger <path> --trajectory-cap N --shared-infra-cap N --expired-cap N --planned-first-tries N
#               [--policies smvla,perceptual-framesamp-modul,groundsg,pp]
#               [--groundsg-variant {ground-sg-oracle,ground-sg-qwenvl,ground-sg-memer}]
#               [--qwenvl-groundsg-adapter <dir>] [--memer-adapter <dir>]
#               [--context-deadline-s S] [--first-infer-deadline-s S] [--media-deadline-s S]
#               [--framesamp-modul-ckpt <run>/79999] [--groundsg-ckpt <run>/79999]
#               [--pp-ckpt <dir>] [--smvla-ckpt <dir>] [--rec-root <dir>] [--trace-root <dir>]
#               [--openpi-data-home <dir> --tokenizer-sha256 <hex>]（跑 perceptual-framesamp-modul／groundsg 时这两项必填）
#               [--cpus 0-3] [--order forward|reverse|shuffle] [--compile-cache on|off] [--det on|off]
#               [--relay on|off] [--limit N] [--never-degrade] [--noprog-s S]
#               [--episode-wall S] [--episode-wall-smvla S] [--episode-wall-framesamp-modul S]
#   --v8 已删除（env_client.py 不再接受）。
#
# 数据集与步数（1006 计划第三阶段；接口冻结说明 2.1）：起跑打印并核对 --dataset 与 --max-steps 的配对——hard-verify ↔
#   1300 且不带 --strict-cap，ood ↔ 1800 且必须带 --strict-cap（第 1801 次 step 在进入环境前被拒）；不符打印
#   RUN_BLOCKED reason=step_cap_pairing、退出 3。这是启动参数的一致性检查，不是按档查表。groundsg 必须给
#   --groundsg-variant；ground-sg-qwenvl 必须且只能给已存在的 --qwenvl-groundsg-adapter，ground-sg-memer 必须且只能给
#   已存在的 --memer-adapter，oracle 两者都不许给；变体或 adapter 配错（含给了变体却不跑 groundsg）打印
#   RUN_BLOCKED reason=variant_pairing。
# 模型种子（接口冻结说明 2.2）：--policy-seed 必填（非负整数，缺失 RUN_BLOCKED reason=policy_seed、退出 3），转发给
#   客户端（seat_info、结果行）与服务：perceptual-framesamp-modul／groundsg（含 MemER，动作服务命令相同）改走外壳
#   src/robomme_ood_eval/servers/policy_server_wrap.py --seed=<n>（包锁定三方 serve_policy.py，不改三方源码），smvla
#   smvla_server.py serve --policy-seed <n>，pp --args.seed <n>；服务写 <out>/<label>/server-metadata-<端口>.json
#   （含 policy_seed、argv），客户端经 --server-metadata 反查结果行 server_seed。作为库被 source 时种子取全局
#   POLICY_SEED（缺省继承同名环境变量），为空则 build_server_cmd 打印 RUN_BLOCKED reason=policy_seed、返回 3。
# 预算（接口冻结说明 2.4）：--budget-ledger --trajectory-cap --shared-infra-cap --expired-cap --planned-first-tries 必填、
#   原样转发给 env_client.py，缺任一 RUN_BLOCKED reason=budget_args、退出 3（不回落 budget_ledger.py 常量默认值）；
#   --reset-budget 改为可选，不给时客户端只计量、不因 reset 额度停机。
# 策略与目录：策略号 smvla=0、perceptual-framesamp-modul=1、groundsg=2、pp=3；输出与账本目录名（label）为策略名，groundsg 为
#   groundsg-<variant>：结果 <out>/<label>/results.jsonl，账本 <ledger-dir>/<label>.ledger.jsonl，录像
#   <rec-root>/<label>/<key>.a<attempt>/，轨迹 <trace-root>/<label>/<key>.a<attempt>/。
#   结果目录里已有别的数据集的结果行时打印 RUN_BLOCKED reason=dataset_crossed（不同数据集须用不同 --out）。
# 服务：perceptual-framesamp-modul 与 groundsg 同一条 serve_policy.py 命令（--seed=7），核对 checkpoint 父目录 history_config.txt 与 server
#   日志：perceptual-framesamp-modul 取 perceptual-framesamp-modul.yaml，groundsg 取 symbolic-grounded-subgoal.yaml；两者启动前都过 tokenizer
#   sha256 闸门（不符 RUN_BLOCKED reason=tokenizer_sha，相符 TOKENIZER_SHA=PASS）。pp：third_party/PonderPounce 下
#   "$PP_PY" -m ponderpounce.eval.robomme_server --args.checkpoint_path <pp-ckpt> --args.seed 0 --args.device cuda:0
#   --port <端口>，就绪看 GET /health 返回 200（只证明权重已加载；首局另有 600 s 放宽）；HF_HUB_OFFLINE／
#   TRANSFORMERS_OFFLINE 缺省为 1，HF_HOME 取调用方环境。
# 解释器：BENCH_PY（smvla／perceptual-framesamp-modul 的客户端与本脚本内的小工具）、MME_VLA_PY、PP_PY 可用环境变量覆盖（有默认值）；groundsg
#   与 pp 的客户端用客户端扩展环境 SGEVAL_CLIENT_PY（缺省 <repo>/artifacts/sg-evaluation/venvs/client-env/bin/python，
#   GL 由环境变量覆盖）；SMVLA_PY 无默认值，跑 smvla 时必须显式传入。缓存根 V75_JAX_CACHE_ROOT；perceptual-framesamp-modul／groundsg 的
#   XLA_PYTHON_CLIENT_MEM_FRACTION 取 SEAT_XLA_MEM_FRACTION（缺省 0.75，QwenVL 变体由预检定值）。
#
# 端口：18000 + 100 × 席号 + 10 × 策略号，FrameSamp+Modulation 录制中继 +1；起前 /dev/tcp 探测，被占依次 +2，最多 5 次。
# 超时：server 就绪 1200 s（等待中 kill -0 查 server 存活）；客户端第一局另放宽 600 s（首次推理编译）；
#       单局墙钟 smvla 900 s、perceptual-framesamp-modul 1200 s、groundsg／pp 1800 s（--episode-wall 统一覆盖，--episode-wall-smvla／-framesamp-modul
#       优先）→ 基础设施超时（客户端退出 75，本脚本重起客户端）；
#       无进展：读 progress.json 的 phase／identity／step（env_client.py 只在这些量变化时更新，不做定时心跳）；
#       这组值超过「单局墙钟 + 600 + 600」秒（不小于 --noprog-s，缺省 1200）不变 → 打印 NO_PROGRESS，杀掉 server
#       与客户端重起一次（不再看 client.log 的 mtime：日志刷新不算进展）；server 中途死亡 → 重起（最多 2 次）。
#       客户端／服务／无进展三项重启计数持久化到 <ledger-dir>/<label>.recovery.json（client_restarts、server_restarts、
#       noprog_restarts），续跑沿用、不清零。
# 客户端退出码：0 完成；3 阻塞、5 reset 额度耗尽、6 RUN_INCOMPLETE（账本读回仍有身份无权威终态——不是服务故障）
#       一律不重启 server 与客户端、照实收尾；75 等其余非 0 重起客户端（最多 8 次）。
# server_epoch（gate2_compare.py 的服务启动边界）：每次 server 就绪（含重起）在 <out>/<label>/server-epochs.tsv 追加
#   一行「epoch<TAB>此刻 results.jsonl 的行数<TAB>端口<TAB>pid<TAB>时间」（epoch 接着文件里最大值加 1，跨多次运行
#   连续），并打印 SERVER_EPOCH。客户端只在 server 新起之后启动、server 重起前客户端一律先被收掉，所以「行号 ≥ 记录
#   行数」的结果行都属于这次启动。每个策略收尾时据此把 results.jsonl 逐行补上整数字段 server_epoch，原子写成
#   <out>/<label>/results.epochs.jsonl（不改 env_client.py 写的 results.jsonl）；gate2_compare.py --groundsg 读这份。
# 日志：<out>/seat-<seat>.log，末行 EXIT_CODE=<rc>。setsid 起的进程组记在 <out>/.v8-pgids（文件名沿用），供外层
#   run_eval_gl.sh 在本脚本异常死亡后回收。
# 轮询间隔：看门狗主循环读环境变量 SEAT_POLL_S（默认 10 s），等 server 就绪读 SEAT_READY_POLL_S（默认 2 s）；
#       只供测试缩短等待。
#
# 作为库：`source run_seat.sh` 只定义变量缺省值与函数、不解析参数也不运行（run_eval_gl.sh、run_official_hard.sh、
#   pair_seat.sh 由此复用 port_busy／pick_port、start_server（含 kill -0 存活检查与就绪判定）、stop_server、
#   noprog_limit／idle_s（NO_PROGRESS 无进展检测与首局 600 s 放宽）、step_cap_pairing、variant_pairing、
#   note_epoch／epoch_annotate、transcode_episode_dir、publish_dir，不另写一套）。
#   媒体函数（transcode_episode_dir 的实现、render_official_dir）在同目录 seat_media_lib.sh，本脚本加载时一并 source。
#
# 第二阶段开关（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分 S2b；缺省关闭、行为同 BASE）：
#   SGEVAL_OFFICIAL_RENDER=1  finish_episode_dir 在并入轨迹之后、转码之前调 render_official_dir 出官方版式视频
#                             （<局目录>/official/）；失败则本局转码带 --keep-raw（原始帧随目录发布）、写
#                             official-render.failed、照常发布成绩；结果按行记进 OR_TALLY（若设）。
#   SGEVAL_PP_SERVER_WRAP=1   pp 服务改以绝对路径起 src/robomme_ood_eval/servers/pp_server_wrap.py（回包带 subgoal），
#                             参数与 ponderpounce.eval.robomme_server 相同；外壳缺失则 RUN_BLOCKED reason=pp_server_wrap_missing。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=seat_media_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../gl" && pwd)/seat_media_lib.sh" \
  || { echo "RUN_BLOCKED reason=seat_media_lib_missing dir=$(dirname "${BASH_SOURCE[0]}")" >&2; return 3 2>/dev/null || exit 3; }
BENCH_PY="${BENCH_PY:-$REPO/.venv/bin/python}"
MME_VLA_PY="${MME_VLA_PY:-$REPO/third_party/mme-vla/.venv/bin/python}"
SMVLA_PY="${SMVLA_PY:-}"  # 无默认值：跑 smvla 时必须显式传入
SGEVAL_CLIENT_PY="${SGEVAL_CLIENT_PY:-$REPO/artifacts/sg-evaluation/venvs/client-env/bin/python}"
PP_PY="${PP_PY:-$REPO/third_party/PonderPounce/.venv/bin/python}"
MME_VLA_COMMIT="ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
FRAMESAMP_MODUL_YAML_EXPECT="perceptual-framesamp-modul.yaml"     # perceptual-framesamp-modul
GROUNDSG_YAML_EXPECT="symbolic-grounded-subgoal.yaml"    # groundsg（GroundSG 两个变体共用）
SEAT_XLA_MEM_FRACTION="${SEAT_XLA_MEM_FRACTION:-0.75}"

SEAT="" ; SEAT_IDX="" ; GPU="" ; COND="" ; OUT="" ; IDENTS="" ; POLICIES="smvla,perceptual-framesamp-modul" ; CPUS=""
ORDER="forward" ; COMPILE_CACHE="off" ; DET="off" ; RELAY="off" ; LIMIT="0"
NEVER_DEGRADE="" ; WALL_SMVLA="" ; WALL_FRAMESAMP_MODUL="" ; WALL_ALL=""
LEDGER_DIR="" ; RESET_BUDGET="" ; INFRA_RETRY_BUDGET="" ; REC_ROOT="" ; TRACE_ROOT="" ; OPENPI_HOME="" ; TOKENIZER_SHA=""
DATASET="" ; MAX_STEPS="" ; STRICT_CAP=0 ; GROUNDSG_VARIANT="" ; QWENVL_ADAPTER="" ; MEMER_ADAPTER=""
# 第三阶段：模型种子（作为库被 source 时继承同名环境变量，供原侧驱动复用 build_server_cmd）、预算参数、执行期限
POLICY_SEED="${POLICY_SEED:-}"
BUDGET_LEDGER="" ; TRAJECTORY_CAP="" ; SHARED_INFRA_CAP="" ; EXPIRED_CAP="" ; PLANNED_FIRST_TRIES=""
CONTEXT_DEADLINE="" ; FIRST_INFER_DEADLINE="" ; MEDIA_DEADLINE=""
SERVER_PORT_CUR=""  # 当前服务端口（客户端据此找服务元数据）
PROG_SIG="" ; PROG_SINCE=0 ; PROG_FLOOR=0 ; IDLE=0  # 无进展检测状态（progress_idle）
CLIENT_RESTARTS=0 ; SERVER_RESTARTS=0 ; NOPROG_RESTARTS=0  # 持久化到 <ledger-dir>/<label>.recovery.json
TOKENIZER_REL="big_vision/paligemma_tokenizer.model"
FRAMESAMP_MODUL_CKPT="${FRAMESAMP_MODUL_CKPT:-/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999}"
GROUNDSG_CKPT="${GROUNDSG_CKPT:-/data/hongzefu/robomme_policy_learning-vqa-test/runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999}"
SMVLA_CKPT="${SMVLA_CKPT:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/checkpoints/simplememvla_robomme}"
PP_CKPT="${PP_CKPT:-}"
READY_TIMEOUT=1200 ; FIRST_EXTRA=600 ; NOPROG_S=1200 ; MAX_CLIENT_RESTARTS=8 ; MAX_SERVER_RESTARTS=2
SEAT_POLL_S="${SEAT_POLL_S:-10}" ; SEAT_READY_POLL_S="${SEAT_READY_POLL_S:-2}"  # 轮询间隔（秒）
TOOL_PY="${TOOL_PY:-}"  # 本脚本内小工具（报告、epoch、转码）用的解释器；空则取 BENCH_PY

TASKSET=()
# server 分支都先清掉会影响确定性／编译缓存的变量，再只设本席位要的（-u 须在赋值之前）
CLEAN_ENV=(-u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES
           -u JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS -u CUBLAS_WORKSPACE_CONFIG)
# 去代理变量、只连 127.0.0.1（GL 计算节点有 HTTP 代理，websockets 15 会读代理变量）
NOPROXY_ENV=(-u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY
             NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost)

SERVER_PID="" ; CLIENT_PID="" ; RELAY_PID="" ; FRESH_SERVER=0 ; SERVER_EPOCH=0
SRV_DIR="" ; SRV_ENV=() ; SRV_ARGV=() ; CLI_ENV=() ; CLI_ARGV=()

ts() { date +%s; }
tool_py() { echo "${TOOL_PY:-$BENCH_PY}"; }
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

http_health() {  # $1 = 端口；GET /health 返回 200 即真（不走代理）
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --noproxy '*' -m 5 -o /dev/null "http://127.0.0.1:$1/health" 2>/dev/null
  else
    "$(tool_py)" - "$1" <<'PY' 2>/dev/null
import sys, urllib.request
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    sys.exit(0 if op.open(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).status == 200 else 1)
except Exception:
    sys.exit(1)
PY
  fi
}

pick_port() {  # $1 = 基础端口；server 端口与其 +1（中继）都空闲才用
  local p="$1" i
  for i in 1 2 3 4 5; do
    if ! port_busy "$p" && ! port_busy "$((p + 1))"; then echo "$p"; return 0; fi
    p=$((p + 2))
  done
  return 1
}

kill_group() {  # $1 = 进程组首进程 pid；先 TERM 再 KILL
  local pid="$1" i
  [[ -z "$pid" ]] && return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

server_alive() { [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; }

stop_server() {
  [[ -n "$RELAY_PID" ]] && kill_group "$RELAY_PID"; RELAY_PID=""
  if [[ -n "$SERVER_PID" ]]; then
    kill_group "$SERVER_PID"
    # 核实显存已释放：该 pid 不应再出现在计算进程列表里
    local i
    for i in $(seq 1 30); do
      nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$SERVER_PID" || break
      sleep 1
    done
    echo "SERVER_STOPPED pid=$SERVER_PID"
  fi
  SERVER_PID=""
}

cleanup() {
  [[ -n "$CLIENT_PID" ]] && kill_group "$CLIENT_PID"; CLIENT_PID=""
  stop_server
}

gpu_slug() { nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" 2>/dev/null | head -1 | tr ' /' '__'; }

# ---------------------------------------------------------------- 策略、目录、配对核对

pol_index() {  # 策略号：smvla=0、perceptual-framesamp-modul=1、groundsg=2、pp=3；未知返回 1
  case "$1" in smvla) echo 0;; perceptual-framesamp-modul) echo 1;; groundsg) echo 2;; pp) echo 3;; *) return 1;; esac
}

pol_label() {  # 输出与账本目录名：groundsg 为 groundsg-<variant>，其余为策略名
  if [[ "$1" == "groundsg" ]]; then echo "groundsg-${GROUNDSG_VARIANT:-novariant}"; else echo "$1"; fi
}

ckpt_of() {
  case "$1" in perceptual-framesamp-modul) echo "$FRAMESAMP_MODUL_CKPT";; groundsg) echo "$GROUNDSG_CKPT";; pp) echo "$PP_CKPT";; *) echo "$SMVLA_CKPT";; esac
}

yaml_expect_of() { if [[ "$1" == "groundsg" ]]; then echo "$GROUNDSG_YAML_EXPECT"; else echo "$FRAMESAMP_MODUL_YAML_EXPECT"; fi; }

client_py_of() {  # groundsg／pp 的客户端用客户端扩展环境，其余用 BENCH_PY
  case "$1" in groundsg|pp) echo "$SGEVAL_CLIENT_PY";; *) echo "$BENCH_PY";; esac
}

step_cap_pairing() {  # 全局 DATASET／MAX_STEPS／STRICT_CAP 的配对核对；打印核对行，不符返回 3
  local want_steps want_strict
  case "$DATASET" in
    hard-verify) want_steps=1300; want_strict=0;;
    ood) want_steps=1800; want_strict=1;;
    *) echo "RUN_BLOCKED reason=step_cap_pairing dataset=${DATASET:-unset} max_steps=${MAX_STEPS:-unset} strict_cap=$STRICT_CAP detail=unknown_dataset"
       return 3;;
  esac
  if [[ "$MAX_STEPS" != "$want_steps" || "$STRICT_CAP" != "$want_strict" ]]; then
    echo "RUN_BLOCKED reason=step_cap_pairing dataset=$DATASET max_steps=${MAX_STEPS:-unset} strict_cap=$STRICT_CAP expect_max_steps=$want_steps expect_strict_cap=$want_strict"
    return 3
  fi
  echo "STEP_CAP_PAIRING=PASS dataset=$DATASET max_steps=$MAX_STEPS strict_cap=$STRICT_CAP"
}

variant_pairing() {  # $1 = 逗号分隔策略；groundsg 变体与 QwenVL／MemER adapter 的配对核对，不符返回 3
  local pols=",$1," why="" memer="${MEMER_ADAPTER:-}"
  if [[ "$pols" == *",groundsg,"* ]]; then
    case "$GROUNDSG_VARIANT" in
      ground-sg-oracle)
        if [[ -n "$QWENVL_ADAPTER" ]]; then why="adapter_without_qwenvl"
        elif [[ -n "$memer" ]]; then why="memer_adapter_without_memer"; fi;;
      ground-sg-qwenvl)
        if [[ -z "$QWENVL_ADAPTER" ]]; then why="qwenvl_needs_adapter"
        elif [[ -n "$memer" ]]; then why="memer_adapter_without_memer"
        elif [[ ! -d "$QWENVL_ADAPTER" ]]; then why="adapter_missing path=$QWENVL_ADAPTER"; fi;;
      ground-sg-memer)
        if [[ -z "$memer" ]]; then why="memer_needs_adapter"
        elif [[ -n "$QWENVL_ADAPTER" ]]; then why="adapter_without_qwenvl"
        elif [[ ! -d "$memer" ]]; then why="memer_adapter_missing path=$memer"; fi;;
      "") why="groundsg_needs_variant";;
      *) why="unknown_variant variant=$GROUNDSG_VARIANT";;
    esac
  elif [[ -n "$GROUNDSG_VARIANT$QWENVL_ADAPTER$memer" ]]; then
    why="variant_without_groundsg"
  fi
  if [[ -n "$why" ]]; then
    echo "RUN_BLOCKED reason=variant_pairing policies=$1 variant=${GROUNDSG_VARIANT:-none} detail=$why"; return 3
  fi
  echo "VARIANT_PAIRING=PASS policies=$1 variant=${GROUNDSG_VARIANT:-none} adapter=${QWENVL_ADAPTER:-none} memer_adapter=${memer:-none}"
}

policy_seed_check() {  # --policy-seed 必填且为非负整数；不符打印 RUN_BLOCKED reason=policy_seed、返回 3
  if [[ ! "${POLICY_SEED:-}" =~ ^[0-9]+$ ]]; then
    echo "RUN_BLOCKED reason=policy_seed detail=--policy-seed 必填且为非负整数（现为 '${POLICY_SEED:-}'，不回落任何旧默认值）"
    return 3
  fi
  echo "POLICY_SEED_CHECK=PASS policy_seed=$POLICY_SEED"
}

budget_args_check() {  # 五个预算参数必填（非负整数＋账本路径）；缺任一打印 RUN_BLOCKED reason=budget_args、返回 3
  local miss=() pair name
  [[ -n "$BUDGET_LEDGER" ]] || miss+=(--budget-ledger)
  for pair in "TRAJECTORY_CAP:--trajectory-cap" "SHARED_INFRA_CAP:--shared-infra-cap" "EXPIRED_CAP:--expired-cap" \
              "PLANNED_FIRST_TRIES:--planned-first-tries"; do
    name="${pair%%:*}"
    [[ "${!name}" =~ ^[0-9]+$ ]] || miss+=("${pair#*:}")
  done
  if (( ${#miss[@]} )); then
    echo "RUN_BLOCKED reason=budget_args detail=必须给 ${miss[*]}（非负整数；不回落 budget_ledger.py 常量默认值）"
    return 3
  fi
  echo "BUDGET_ARGS=PASS budget_ledger=$BUDGET_LEDGER trajectory_cap=$TRAJECTORY_CAP shared_infra_cap=$SHARED_INFRA_CAP expired_cap=$EXPIRED_CAP planned_first_tries=$PLANNED_FIRST_TRIES reset_budget=${RESET_BUDGET:-meter_only}"
}

dataset_crossed() {  # $1 = results.jsonl；已有别的数据集的结果行即返回 0（打印 RUN_BLOCKED）
  [[ -f "$1" ]] || return 1
  "$(tool_py)" - "$1" "$DATASET" <<'PY'
import json, sys
bad = set()
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(r, dict) and r.get("dataset") not in (None, sys.argv[2]):
        bad.add(str(r.get("dataset")))
if bad:
    print(f"RUN_BLOCKED reason=dataset_crossed results={sys.argv[1]} dataset={sys.argv[2]} found={','.join(sorted(bad))}")
    sys.exit(0)
sys.exit(1)
PY
}

ckpt_fingerprint() {  # $1 = 目录；逐文件 sha256 汇成一个指纹
  "$(tool_py)" - "$1" <<'PY'
import hashlib, sys, time
from pathlib import Path
t0 = time.time(); root = Path(sys.argv[1]).resolve(); h = hashlib.sha256(); n = b = 0
for p in sorted(x for x in root.rglob("*") if x.is_file()):
    fh = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            fh.update(chunk)
    h.update(f"{p.relative_to(root)}\t{p.stat().st_size}\t{fh.hexdigest()}\n".encode()); n += 1; b += p.stat().st_size
print(f"CKPT_FINGERPRINT dir={root} sha256={h.hexdigest()} files={n} bytes={b} secs={time.time()-t0:.1f}")
PY
}

preflight_mme_vla() {  # $1 = perceptual-framesamp-modul|groundsg
  local pol="${1:-perceptual-framesamp-modul}" sub="$REPO/third_party/mme-vla" head hc ck expect
  ck="$(ckpt_of "$pol")"; expect="$(yaml_expect_of "$pol")"
  head="$(git -C "$sub" rev-parse HEAD 2>/dev/null)"
  [[ "$head" == "$MME_VLA_COMMIT" ]] || { echo "RUN_BLOCKED reason=mme_vla_commit head=$head"; return 1; }
  git -C "$sub" diff --quiet HEAD -- src scripts packages pyproject.toml uv.lock || { echo "RUN_BLOCKED reason=mme_vla_dirty"; return 1; }
  if [[ -n "$(ls -A "$sub/third_party/robomme_benchmark" 2>/dev/null)" ]]; then
    echo "RUN_BLOCKED reason=nested_submodule_not_empty"; return 1
  fi
  hc="$(cat "$ck/../history_config.txt" 2>/dev/null)"
  [[ "$hc" == "$expect" ]] || { echo "RUN_BLOCKED reason=history_config value='$hc' expect='$expect' policy=$pol"; return 1; }
  [[ -f "$sub/src/mme_vla_suite/models/config/robomme/$hc" ]] || { echo "RUN_BLOCKED reason=yaml_missing $hc"; return 1; }
  [[ -d "$ck/params" && -d "$ck/assets" ]] || { echo "RUN_BLOCKED reason=ckpt_layout $ck"; return 1; }
  [[ -x "$MME_VLA_PY" ]] || { echo "RUN_BLOCKED reason=mme_vla_venv_missing $MME_VLA_PY"; return 1; }
  [[ -f "$(policy_server_wrap_path)" ]] || { echo "RUN_BLOCKED reason=policy_server_wrap_missing path=$(policy_server_wrap_path)"; return 1; }
  echo "MME_VLA_PREFLIGHT=PASS policy=$pol commit=$head history_config=$hc ckpt=$ck py=$MME_VLA_PY"
}

preflight_pp() {
  [[ -d "$REPO/third_party/PonderPounce" ]] || { echo "RUN_BLOCKED reason=pp_submodule_missing dir=$REPO/third_party/PonderPounce"; return 1; }
  [[ -x "$PP_PY" ]] || { echo "RUN_BLOCKED reason=pp_venv_missing $PP_PY"; return 1; }
  [[ -n "$PP_CKPT" && -d "$PP_CKPT" ]] || { echo "RUN_BLOCKED reason=pp_ckpt_missing ckpt=${PP_CKPT:-unset}"; return 1; }
  [[ -f "$PP_CKPT/norm_stats.json" ]] || { echo "RUN_BLOCKED reason=pp_ckpt_layout ckpt=$PP_CKPT（缺 norm_stats.json）"; return 1; }
  if [[ "${SGEVAL_PP_SERVER_WRAP:-0}" == "1" && ! -f "$(pp_server_wrap_path)" ]]; then
    echo "RUN_BLOCKED reason=pp_server_wrap_missing path=$(pp_server_wrap_path)"; return 1
  fi
  echo "PP_PREFLIGHT=PASS ckpt=$PP_CKPT py=$PP_PY seed=${POLICY_SEED:-unset} hf_home=${HF_HOME:-unset} hf_hub_offline=${HF_HUB_OFFLINE:-1} server_wrap=${SGEVAL_PP_SERVER_WRAP:-0}"
}

pp_server_wrap_path() {  # pp 服务外壳的绝对路径（服务 cwd 在第三方目录，必须用绝对路径起）
  echo "$REPO/src/robomme_ood_eval/servers/pp_server_wrap.py"
}

policy_server_wrap_path() {  # MME-VLA 动作服务外壳（perceptual-framesamp-modul／groundsg 共用）的绝对路径
  echo "$REPO/src/robomme_ood_eval/servers/policy_server_wrap.py"
}

server_metadata_path() {  # $1 = 策略；$2 = 端口 → 服务元数据路径（SRV_META_DIR 可覆盖目录；缺省 <out>/<label>）
  echo "${SRV_META_DIR:-${OUT:-$PWD}/$(pol_label "$1")}/server-metadata-$2.json"
}

tokenizer_gate() {  # perceptual-framesamp-modul／groundsg server 启动前核 OPENPI_DATA_HOME 下 tokenizer 的 sha256（不现场下载顶替）
  local f actual
  if [[ -z "$OPENPI_HOME" || -z "$TOKENIZER_SHA" ]]; then
    echo "RUN_BLOCKED reason=tokenizer_sha detail=missing_args（跑 perceptual-framesamp-modul／groundsg 须给 --openpi-data-home 与 --tokenizer-sha256）"
    return 1
  fi
  f="$OPENPI_HOME/$TOKENIZER_REL"
  if [[ ! -f "$f" ]]; then
    echo "RUN_BLOCKED reason=tokenizer_sha detail=file_missing file=$f"; return 1
  fi
  actual="$(sha256sum "$f" | awk '{print $1}')"
  if [[ "${actual,,}" != "${TOKENIZER_SHA,,}" ]]; then
    echo "RUN_BLOCKED reason=tokenizer_sha expected=${TOKENIZER_SHA,,} actual=$actual file=$f"; return 1
  fi
  echo "TOKENIZER_SHA=PASS sha256=$actual file=$f openpi_data_home=$OPENPI_HOME"
}

note_pgid() {  # 记下 setsid 起的进程组，供外层编排在本脚本异常死亡后回收（$1 = 角色，$2 = pid）
  [[ -n "$OUT" ]] && echo "$1 $2" >> "$OUT/.v8-pgids" 2>/dev/null
  return 0
}

# ---------------------------------------------------------------- server_epoch

note_epoch() {  # $1 = results.jsonl；$2 = epochs.tsv；$3 = 端口。追加一行并设 SERVER_EPOCH
  local res="$1" tsv="$2" port="$3" last=0 n=0
  [[ -f "$tsv" ]] && last="$(awk -F'\t' '$1 ~ /^[0-9]+$/ && $1 > e {e = $1} END {print e + 0}' "$tsv")"
  [[ -f "$res" ]] && n="$(wc -l < "$res" | tr -d ' ')"
  SERVER_EPOCH=$((last + 1))
  printf '%s\t%s\t%s\t%s\t%s\n' "$SERVER_EPOCH" "$n" "$port" "${SERVER_PID:-}" "$(date -Is)" >> "$tsv"
  echo "SERVER_EPOCH epoch=$SERVER_EPOCH results_line=$n port=$port pid=${SERVER_PID:-}"
}

epoch_annotate() {  # $1 = results.jsonl；$2 = epochs.tsv；$3 = 输出（逐行补 server_epoch，原子替换）
  [[ -f "$1" ]] || return 0
  "$(tool_py)" - "$1" "$2" "$3" <<'PY'
import json, os, sys
res, tsv, out = sys.argv[1:4]
eps = []
if os.path.exists(tsv):
    for line in open(tsv, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0].isdigit() and p[1].isdigit():
            eps.append((int(p[1]), int(p[0])))
eps.sort()
with open(res, "rb") as fh:
    lines = fh.read().split(b"\n")
rows = []
for i, raw in enumerate(lines):
    if not raw.strip():
        continue
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if not isinstance(r, dict):
        continue
    ep = 0  # 早于任何 epoch 记录的行（旧运行）记 0
    for off, e in eps:
        if off <= i:
            ep = e
        else:
            break
    r["server_epoch"] = ep
    rows.append(r)
tmp = out + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
os.replace(tmp, out)
print(f"SERVER_EPOCH_ANNOTATE rows={len(rows)} epochs={len(eps)} out={out}")
PY
}

# ---------------------------------------------------------------- server

build_server_cmd() {  # $1 = 策略；$2 = 端口 → 设 SRV_DIR、SRV_ENV、SRV_ARGV（不启动）；POLICY_SEED 为空返回 3
  local pol="$1" port="$2" meta
  SRV_ENV=() ; SRV_ARGV=()
  if [[ ! "${POLICY_SEED:-}" =~ ^[0-9]+$ ]]; then
    echo "RUN_BLOCKED reason=policy_seed detail=build_server_cmd 需要 POLICY_SEED（现为 '${POLICY_SEED:-}'）"
    return 3
  fi
  meta="$(server_metadata_path "$pol" "$port")"
  case "$pol" in
    perceptual-framesamp-modul|groundsg)
      SRV_DIR="$REPO/third_party/mme-vla"
      SRV_ENV=(XLA_PYTHON_CLIENT_MEM_FRACTION="$SEAT_XLA_MEM_FRACTION" GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384
               UV_LINK_MODE=copy VIRTUAL_ENV="$REPO/third_party/mme-vla/.venv")
      if [[ "$COMPILE_CACHE" == "on" ]]; then
        SRV_ENV+=(JAX_COMPILATION_CACHE_DIR="${V75_JAX_CACHE_ROOT:-$REPO/artifacts/v7.5eval/jax-cache}/$(gpu_slug)")
      fi
      [[ "$DET" == "on" ]] && SRV_ENV+=(XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0")
      # tokenizer 缓存根固定（server 在 env 清理后启动，须显式写进 server 环境）
      SRV_ENV+=(OPENPI_DATA_HOME="$OPENPI_HOME" PYTHONUNBUFFERED=1)
      # 外壳包锁定三方 scripts/serve_policy.py（cwd 仍在 third_party/mme-vla），--sgeval-* 由外壳自用、其余原样交给 tyro
      SRV_ARGV=("$MME_VLA_PY" "$(policy_server_wrap_path)" --sgeval-metadata-out="$meta" --seed="$POLICY_SEED"
                --port="$port" policy:checkpoint --policy.config=mme_vla_suite --policy.dir="$(ckpt_of "$pol")");;
    pp)
      SRV_DIR="$REPO/third_party/PonderPounce"
      SRV_ENV=(PYTHONUNBUFFERED=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}")
      [[ -n "${HF_HOME:-}" ]] && SRV_ENV+=(HF_HOME="$HF_HOME")
      if [[ "${SGEVAL_PP_SERVER_WRAP:-0}" == "1" ]]; then  # 外壳子类化原服务（回包加 subgoal 与审计键、写服务元数据）
        SRV_ARGV=("$PP_PY" "$(pp_server_wrap_path)" --sgeval-metadata-out="$meta")
      else
        SRV_ARGV=("$PP_PY" -m ponderpounce.eval.robomme_server)
      fi
      SRV_ARGV+=(--args.checkpoint_path "$PP_CKPT" --args.seed "$POLICY_SEED" --args.device cuda:0 --port "$port");;
    *)
      SRV_DIR="$REPO"
      # 确定性模式：--det + CUBLAS_WORKSPACE_CONFIG=:4096:8
      local detarg=()
      [[ "$DET" == "on" ]] && detarg=(--det) && SRV_ENV=(CUBLAS_WORKSPACE_CONFIG=:4096:8)
      SRV_ENV+=(PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg)
      SRV_ARGV=("$SMVLA_PY" src/robomme_ood_eval/servers/smvla_server.py serve --port "$port" --ckpt "$SMVLA_CKPT"
                --warmup "${detarg[@]}" --policy-seed "$POLICY_SEED" --metadata_out "$meta");;
  esac
  return 0
}

server_ready() {  # $1 = 策略；$2 = 端口。pp 看 /health 200，其余看端口在听
  if [[ "$1" == "pp" ]]; then http_health "$2"; else port_busy "$2"; fi
}

start_server() {  # $1 = 策略；$2 = 端口；$3 = 日志；$4 = 结果目录（server_epoch 记在这里，缺省 <out>/<label>）
  local pol="$1" port="$2" slog="$3" rdir="${4:-$OUT/$(pol_label "$1")}"
  build_server_cmd "$pol" "$port" || return 3
  SERVER_PORT_CUR="$port"
  ( cd "$SRV_DIR" && exec setsid env "${CLEAN_ENV[@]}" "${NOPROXY_ENV[@]}" "${SRV_ENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" \
      "${TASKSET[@]}" "${SRV_ARGV[@]}" ) >"$slog" 2>&1 &
  SERVER_PID=$!
  note_pgid server "$SERVER_PID"
  local t0; t0=$(ts)
  echo "SERVER_START policy=$pol port=$port pid=$SERVER_PID log=$slog"
  while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "SERVER_DIED_BEFORE_READY policy=$pol port=$port"; tail -n 20 "$slog"; SERVER_PID=""; return 1
    fi
    if server_ready "$pol" "$port"; then break; fi
    if (( $(ts) - t0 > READY_TIMEOUT )); then
      echo "SERVER_READY_TIMEOUT policy=$pol port=$port limit_s=$READY_TIMEOUT"; stop_server; return 1
    fi
    sleep "$SEAT_READY_POLL_S" & wait $!
  done
  echo "SERVER_READY policy=$pol port=$port ready_s=$(( $(ts) - t0 ))"
  FRESH_SERVER=1  # 下一个客户端的第一局享受首次推理放宽
  if [[ "$pol" == "perceptual-framesamp-modul" || "$pol" == "groundsg" ]]; then
    local expect; expect="$(yaml_expect_of "$pol")"
    if grep -q "history_config='$expect'" "$slog"; then
      echo "SERVER_CONFIG=PASS policy=$pol history_config=$expect seed=$POLICY_SEED wrap=policy_server_wrap"
    else
      echo "RUN_BLOCKED reason=server_config（日志里没有 history_config='$expect'）"; stop_server; return 3
    fi
    if [[ "$pol" == "perceptual-framesamp-modul" && "$RELAY" == "on" ]]; then
      ( exec setsid env "${NOPROXY_ENV[@]}" PYTHONUNBUFFERED=1 "$BENCH_PY" "$REPO/src/robomme_ood_eval/models/framesamp_modul.py" relay \
          --listen "$((port + 1))" --upstream "$port" --log "$rdir/relay-$port.jsonl" ) >"$rdir/relay-$port.log" 2>&1 &
      RELAY_PID=$!
      local i; for i in $(seq 1 60); do port_busy "$((port + 1))" && break; sleep 0.5; done
      echo "RELAY_READY port=$((port + 1)) pid=$RELAY_PID"
    fi
  elif [[ "$pol" == "pp" ]]; then
    echo "SERVER_CONFIG=INFO policy=pp health=200 seed=$POLICY_SEED ckpt=$PP_CKPT server_wrap=${SGEVAL_PP_SERVER_WRAP:-0}"
  fi
  mkdir -p "$rdir"
  note_epoch "$rdir/results.jsonl" "$rdir/server-epochs.tsv" "$port"
  return 0
}

wall_of() {  # 单局墙钟：--episode-wall-smvla／-framesamp-modul 优先，其次 --episode-wall，再次缺省 smvla 900／perceptual-framesamp-modul 1200／其余 1800
  case "$1" in
    perceptual-framesamp-modul) if [[ -n "$WALL_FRAMESAMP_MODUL" ]]; then echo "$WALL_FRAMESAMP_MODUL"; return; fi;;
    smvla) if [[ -n "$WALL_SMVLA" ]]; then echo "$WALL_SMVLA"; return; fi;;
  esac
  if [[ -n "$WALL_ALL" ]]; then echo "$WALL_ALL"; return; fi
  case "$1" in perceptual-framesamp-modul) echo 1200;; smvla) echo 900;; *) echo 1800;; esac
}

noprog_limit() {  # 无进展阈值：$1 = 策略；$2 = 1（缺省）含首局放宽 FIRST_EXTRA，0 不含；不小于 NOPROG_S
  local n=$(( $(wall_of "$1") + 600 ))
  (( ${2:-1} == 1 )) && n=$((n + FIRST_EXTRA))
  (( NOPROG_S > n )) && n=$NOPROG_S
  echo "$n"
}

progress_sig() {  # $1 = progress.json → 打印「phase|identity|attempt_no|step|episodes_done<TAB>t」；读不到打印空
  [[ -f "$1" ]] || return 0
  "$(tool_py)" - "$1" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("unreadable\t0")
    raise SystemExit(0)
sig = "|".join(str(d.get(k)) for k in ("phase", "identity", "attempt_no", "step", "episodes_done"))
t = d.get("t")
print(f"{sig}\t{int(t) if isinstance(t, (int, float)) else 0}")
PY
}

progress_idle() {  # $1 = progress.json；设全局 IDLE = 具名进展（phase／identity／step）最后一次变化距今的秒数
  # 只有签名变化才算进展：客户端只在阶段、身份、步数变化时写 progress.json；同一签名被重写（刷新 t）不重新计时。
  # 起点不早于 PROG_FLOOR（客户端每次（重）起时设为当时）。
  local line sig t now
  now=$(ts)
  line="$(progress_sig "$1")"
  sig="${line%%$'\t'*}"; t="${line##*$'\t'}"
  [[ "$t" =~ ^[0-9]+$ ]] || t=0
  if [[ -n "$sig" && "$sig" != "$PROG_SIG" ]]; then
    PROG_SIG="$sig"
    PROG_SINCE=$(( t > 0 ? t : now ))
  fi
  (( PROG_SINCE < PROG_FLOOR )) && PROG_SINCE=$PROG_FLOOR
  (( PROG_SINCE == 0 )) && PROG_SINCE=$now
  IDLE=$(( now - PROG_SINCE ))
  (( IDLE < 0 )) && IDLE=0
  return 0
}

idle_s() {  # 参数：若干文件。唯一参数是 progress.json 时按具名进展计（progress_idle）；否则取最新 mtime 距今的秒数，
  # 都不存在返回 0（视为刚开始；原侧驱动看 results.jsonl 等文件 mtime，沿用此口径）
  if [[ $# -eq 1 && "$(basename "$1")" == progress.json ]]; then
    progress_idle "$1"; echo "$IDLE"; return
  fi
  local f m last=0
  for f in "$@"; do
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    (( m > last )) && last=$m
  done
  (( last == 0 )) && { echo 0; return; }
  echo $(( $(ts) - last ))
}

recovery_path() { echo "$LEDGER_DIR/$1.recovery.json"; }  # $1 = label

recovery_load() {  # $1 = label；读回三项重启计数（文件不存在为 0），续跑沿用、不清零
  local f txt k v
  f="$(recovery_path "$1")"
  CLIENT_RESTARTS=0 ; SERVER_RESTARTS=0 ; NOPROG_RESTARTS=0
  if [[ -f "$f" ]]; then
    txt="$(cat "$f")"
    for k in client_restarts server_restarts noprog_restarts; do
      v=0; [[ "$txt" =~ \"$k\":\ *([0-9]+) ]] && v="${BASH_REMATCH[1]}"
      case "$k" in client_restarts) CLIENT_RESTARTS=$v;; server_restarts) SERVER_RESTARTS=$v;; *) NOPROG_RESTARTS=$v;; esac
    done
  fi
  echo "RECOVERY_STATE label=$1 client_restarts=$CLIENT_RESTARTS server_restarts=$SERVER_RESTARTS noprog_restarts=$NOPROG_RESTARTS file=$f"
}

recovery_save() {  # $1 = label；原子写三项重启计数
  local f; f="$(recovery_path "$1")"
  printf '{"client_restarts": %d, "server_restarts": %d, "noprog_restarts": %d}\n' \
    "$CLIENT_RESTARTS" "$SERVER_RESTARTS" "$NOPROG_RESTARTS" > "$f.tmp" && mv -f -- "$f.tmp" "$f"
}

restart_server() {  # 起 server 失败时保留 RUN_BLOCKED 的 3，其余记基础设施 4
  local rc
  start_server "$@"; rc=$?
  (( rc == 0 )) && return 0
  stop_server
  (( rc == 3 )) && return 3
  return 4
}

# ---------------------------------------------------------------- 转码与原子发布（run_eval_gl.sh／run_official_hard.sh 共用）

transcode_episode_dir() {  # [--keep-raw] <局目录>；实现在 seat_media_lib.sh（_seat_media_transcode），此处保留同名入口
  _seat_media_transcode "$@"  # 供 test_eval_wiring 的共享函数守卫与既有 source run_seat.sh 的脚本沿用
}

# 原子发布一个目录：rsync 到 <root>/.incoming/<name>/（已存在即重试，加 -c）→ 逐文件 sha256 → 一致才 mv 成
# <root>/<name>（已存在则 <name>.dupN，不覆盖）→ 删源目录；成功且设了 TALLY 时在计数文件追加一行「1 <字节数>」。
publish_dir() {  # $1 = 源目录；$2 = 目的根；$3 = 目录名
  local src="$1" root="$2" name="$3" inc f rel a b bytes=0 bad=0 copt=() target k
  inc="$root/.incoming/$name"
  [[ -d "$inc" ]] && copt=(-c)
  mkdir -p "$inc" || { echo "REC_SYNC_MKDIR_FAIL dir=$inc"; return 1; }
  rsync -rt --delete "${copt[@]}" "$src/" "$inc/" || { echo "REC_SYNC_RSYNC_FAIL src=$src"; return 1; }
  while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    a="$(sha256sum "$f" | awk '{print $1}')"
    b="$(sha256sum "$inc/$rel" 2>/dev/null | awk '{print $1}')"
    if [[ -z "$a" || "$a" != "$b" ]]; then
      echo "REC_SYNC_SHA_MISMATCH file=$rel src_sha=$a dst_sha=${b:-missing}"; bad=1
    else
      bytes=$((bytes + $(stat -c %s "$f")))
    fi
  done < <(find "$src" -type f -print0)
  (( bad == 0 )) || return 1
  target="$root/$name"; k=0
  while [[ -e "$target" ]]; do k=$((k + 1)); target="$root/$name.dup$k"; done
  mv -T -- "$inc" "$target" || { echo "REC_SYNC_MV_FAIL from=$inc to=$target"; return 1; }
  (( k > 0 )) && echo "REC_SYNC_DUP dir=$name published_as=$(basename "$target")"
  rm -rf -- "$src" || return 1
  [[ -n "${TALLY:-}" ]] && echo "1 $bytes" >> "$TALLY"
  return 0
}

# 一局收尾：并入轨迹目录（不含 qwen-tmp）→ [SGEVAL_OFFICIAL_RENDER=1 时官方重绘] → 就地转码 → 原子发布。转码结果
# 按行记进 TC_TALLY（ok／frame_mismatch／fail／其他）；转码不成功时原始帧照样发布（不丢数据），由 SEAT_REC_SYNC 的
# frame_mismatch／transcode_fail 判 FAIL。官方重绘失败不影响转码与发布：本局转码带 --keep-raw（原始帧保留随目录发布）、
# 写 official-render.failed，结果（ok／fail）记进 OR_TALLY（若设）；重入时重绘成功会删掉旧的 official-render.failed。
finish_episode_dir() {  # $1 = 本局录像目录（可不存在）；$2 = 本局轨迹目录（可不存在）；$3 = 发布根；$4 = 目录名
  local src="$1" tsrc="$2" root="$3" name="$4" out rc res keep=()
  if [[ -d "$tsrc" ]]; then
    mkdir -p "$src"
    rsync -a --exclude=qwen-tmp "$tsrc/" "$src/" && rm -rf -- "$tsrc"
  fi
  [[ -d "$src" ]] || return 0
  if [[ "${SGEVAL_OFFICIAL_RENDER:-0}" == "1" ]]; then
    out="$(render_official_dir "$src")"; rc=$?
    [[ -n "$out" ]] && echo "$out"
    if (( rc != 0 )); then
      keep=(--keep-raw)
      printf 'dir=%s\nrc=%s\ntime=%s\n%s\n' "$name" "$rc" "$(date -Is)" "$(grep '^OFFICIAL_RENDER=FAIL' <<<"$out" | tail -n 1)" \
        > "$src/official-render.failed"
      echo "OFFICIAL_RENDER_KEEP_RAW dir=$name rc=$rc（原始帧保留、照常转码发布）"
      [[ -n "${OR_TALLY:-}" ]] && echo fail >> "$OR_TALLY"
    else
      rm -f -- "$src/official-render.failed"
      [[ -n "${OR_TALLY:-}" ]] && echo ok >> "$OR_TALLY"
    fi
  fi
  out="$(transcode_episode_dir "${keep[@]}" "$src")"; rc=$?
  [[ -n "$out" ]] && echo "$out"
  res="$(sed -n 's/.* result=\([a-z_]*\).*/\1/p' <<<"$out" | tail -n 1)"
  [[ -n "${TC_TALLY:-}" ]] && echo "${res:-fail}" >> "$TC_TALLY"
  (( rc == 0 )) || echo "REC_TRANSCODE_KEEP_RAW dir=$name rc=$rc（原始帧随目录发布）"
  publish_dir "$src" "$root" "$name"
}

tc_count() {  # $1 = 结果名；从 TC_TALLY 计数
  [[ -n "${TC_TALLY:-}" && -f "$TC_TALLY" ]] || { echo 0; return; }
  grep -cx "$1" "$TC_TALLY" || true
}

# ---------------------------------------------------------------- 报告与客户端

write_report() {  # $1 = 策略；$2 = 退出状态；写 <out>/<label>/seat-report.json、results.epochs.jsonl 并打印 SEAT_DONE
  local label dir; label="$(pol_label "$1")"; dir="$OUT/$label"
  epoch_annotate "$dir/results.jsonl" "$dir/server-epochs.tsv" "$dir/results.epochs.jsonl"
  "$(tool_py)" - "$dir" "$label" "$COND" "$SEAT" "$2" <<'PY'
import json, sys
from pathlib import Path
out, pol, cond, seat, rc = sys.argv[1:6]
rows = []
p = Path(out) / "results.jsonl"
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
main = [r for r in rows if not r.get("canary")]
last = {}
for r in main:
    # V8 结果行带 key（task_tier_seed）；旧行仍按 (task, seed)
    last[r["key"] if r.get("v8") and r.get("key") else (r["task"], int(r["seed"]))] = r
final = list(last.values())
rep = {"policy": pol, "cond": cond, "seat": seat, "loop_exit_status": int(rc), "records": len(rows),
       "canary": sum(1 for r in rows if r.get("canary")), "done": sum(1 for r in final if not r.get("infra")),
       "errors": sum(1 for r in final if r.get("status") == "error" and not r.get("infra")),
       "infra": sum(1 for r in main if r.get("infra")), "run_blocked": sum(1 for r in rows if r.get("run_blocked")),
       "queue_check": "NA", "queue": None}
for k in ("success", "fail", "timeout"):
    rep[k] = sum(1 for r in final if r.get("status") == k and not r.get("infra"))
# 队列模式已删除：queue／queue_check 键恒为 None／NA，只为保持 seat-report.json 与 SEAT_DONE 行格式不变
(Path(out) / "seat-report.json").write_text(json.dumps(rep, sort_keys=True, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"SEAT_DONE policy={pol} cond={cond} seat={seat} done={rep['done']} errors={rep['errors']} infra={rep['infra']} "
      f"queue_check={rep['queue_check']} loop_exit_status={rc}")
sys.exit(0)
PY
}

build_client_cmd() {  # $1 = 策略；$2 = 客户端连接端口；$3 = 单局墙钟；$4 = 首局放宽 → 设 CLI_ENV、CLI_ARGV（不启动）
  local pol="$1" cport="$2" wall="$3" extra="$4" label
  echo "RUN_BLOCKED reason=legacy_client（新侧客户端已改为 dev-scripts/gl/seat.py，经 dev-scripts/gl/run_eval_gl.sh 调用）"
  return 3
  label="$(pol_label "$pol")"
  CLI_ENV=()
  # smvla 旧官方环境与推理同进程、OMP_NUM_THREADS=1；新接口环境侧照设
  [[ "$pol" == "smvla" ]] && CLI_ENV=(OMP_NUM_THREADS=1)
  # QwenVL 预测器（ms-swift）默认走 ModelScope：一律 HF 且离线（HF_HOME 取调用方环境）
  [[ "$pol" == "groundsg" ]] && CLI_ENV=(USE_HF=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}")
  CLI_ARGV=("$(client_py_of "$pol")" legacy-env-client-removed run --policy "$pol"
            --identities "$IDENTS" --order "$ORDER" --dataset "$DATASET" --max-steps "$MAX_STEPS"
            --cond "$COND" --seat "$SEAT" --port "$cport" --out "$OUT/$label"
            --episode-wall-s "$wall" --first-extra-s "$extra" --limit "$LIMIT"
            --ledger "$LEDGER_DIR/$label.ledger.jsonl" --infra-retry-budget "$INFRA_RETRY_BUDGET"
            --policy-seed "$POLICY_SEED"
            --budget-ledger "$BUDGET_LEDGER" --trajectory-cap "$TRAJECTORY_CAP" --shared-infra-cap "$SHARED_INFRA_CAP"
            --expired-cap "$EXPIRED_CAP" --planned-first-tries "$PLANNED_FIRST_TRIES")
  [[ -n "$RESET_BUDGET" ]] && CLI_ARGV+=(--reset-budget "$RESET_BUDGET")
  [[ -n "$SERVER_PORT_CUR" ]] && CLI_ARGV+=(--server-metadata "$(server_metadata_path "$pol" "$SERVER_PORT_CUR")")
  [[ -n "$CONTEXT_DEADLINE" ]] && CLI_ARGV+=(--context-deadline-s "$CONTEXT_DEADLINE")
  [[ -n "$FIRST_INFER_DEADLINE" ]] && CLI_ARGV+=(--first-infer-deadline-s "$FIRST_INFER_DEADLINE")
  [[ -n "$MEDIA_DEADLINE" ]] && CLI_ARGV+=(--media-deadline-s "$MEDIA_DEADLINE")
  (( STRICT_CAP == 1 )) && CLI_ARGV+=(--strict-cap)
  if [[ "$pol" == "groundsg" ]]; then
    CLI_ARGV+=(--groundsg-variant "$GROUNDSG_VARIANT")
    [[ -n "$QWENVL_ADAPTER" ]] && CLI_ARGV+=(--qwenvl-groundsg-adapter "$QWENVL_ADAPTER")
    [[ -n "$MEMER_ADAPTER" ]] && CLI_ARGV+=(--memer-adapter "$MEMER_ADAPTER")
  fi
  [[ -n "$NEVER_DEGRADE" ]] && CLI_ARGV+=(--never-degrade)
  [[ -n "$REC_ROOT" ]] && CLI_ARGV+=(--rec-root "$REC_ROOT/$label")
  [[ -n "$TRACE_ROOT" ]] && CLI_ARGV+=(--trace-root "$TRACE_ROOT/$label")
  return 0
}

start_client() {  # $1 = 策略；$2 = 客户端连接端口
  local pol="$1" cport="$2" wall extra=0 dir
  dir="$OUT/$(pol_label "$pol")"
  wall="$(wall_of "$pol")"
  # 首次推理放宽只给 server 新（重）起后的第一个客户端；客户端单独重起（server 已热）不放宽
  (( FRESH_SERVER == 1 )) && extra="$FIRST_EXTRA"
  FRESH_SERVER=0
  PROG_SIG="" ; PROG_FLOOR=$(ts)  # 新客户端：无进展计时从此刻起算
  build_client_cmd "$pol" "$cport" "$wall" "$extra"
  ( cd "$REPO" && exec setsid env "${NOPROXY_ENV[@]}" "${CLI_ENV[@]}" GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384 \
      PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" V75_DATA_ROOT="${V75_DATA_ROOT:-/data}" "${TASKSET[@]}" \
      "${CLI_ARGV[@]}" ) >>"$dir/client.log" 2>&1 &
  CLIENT_PID=$!
  note_pgid client "$CLIENT_PID"
  echo "CLIENT_START policy=$pol port=$cport pid=$CLIENT_PID wall_s=$wall first_extra_s=$extra dataset=$DATASET max_steps=$MAX_STEPS strict_cap=$STRICT_CAP"
}

run_policy() {  # $1 = 策略；$2 = 策略号
  local pol="$1" pidx="$2" base port cport slog rc label dir cpy
  label="$(pol_label "$pol")"; dir="$OUT/$label"
  mkdir -p "$dir"
  [[ -n "$SERVER_PID" ]] && { echo "RUN_BLOCKED reason=previous_server_alive pid=$SERVER_PID"; return 3; }
  case "$pol" in
    perceptual-framesamp-modul|groundsg) tokenizer_gate || return 3; preflight_mme_vla "$pol" || return 3;;
    pp) preflight_pp || return 3;;
  esac
  [[ "$pol" == "smvla" && -z "$SMVLA_PY" ]] && { echo "RUN_BLOCKED reason=smvla_py_unset（SMVLA_PY 必须显式传入）"; return 3; }
  [[ "$pol" == "smvla" && ! -x "$SMVLA_PY" ]] && { echo "RUN_BLOCKED reason=smvla_venv_missing $SMVLA_PY"; return 3; }
  cpy="$(client_py_of "$pol")"
  [[ -x "$cpy" ]] || { echo "RUN_BLOCKED reason=client_py_missing policy=$pol py=$cpy"; return 3; }
  dataset_crossed "$dir/results.jsonl" && return 3
  base=$((18000 + 100 * SEAT_IDX + 10 * pidx))
  port="$(pick_port "$base")" || { echo "INFRA port_exhausted base=$base"; return 4; }
  ckpt_fingerprint "$(ckpt_of "$pol")" > "$dir/ckpt-fingerprint.txt" 2>&1 &
  local fp_pid=$!
  slog="$dir/server-$port-$(ts).log"
  start_server "$pol" "$port" "$slog" "$dir"; rc=$?
  wait "$fp_pid"; cat "$dir/ckpt-fingerprint.txt"
  if (( rc != 0 )); then write_report "$pol" "$rc"; return "$rc"; fi
  cport="$port"; [[ "$pol" == "perceptual-framesamp-modul" && "$RELAY" == "on" ]] && cport=$((port + 1))

  rc=0
  policy_loop "$pol" "$port" "$cport" || rc=$?
  stop_server
  write_report "$pol" "$rc"
  echo "POLICY_DONE policy=$label seat=$SEAT rc=$rc"
  return "$rc"
}

policy_loop() {  # $1 策略 $2 server 端口 $3 客户端端口；返回 0 完成 / 3 阻塞 / 4 基础设施用尽 / 5 额度耗尽 / 6 账本未齐
  local pol="$1" port="$2" cport="$3" rc slog dir idle label
  label="$(pol_label "$pol")"; dir="$OUT/$label"
  # 无进展阈值须大于「单局墙钟 + 首次放宽 + 启动余量」，单局卡死先由客户端墙钟计时器以 75 退出并回收
  local noprog; noprog="$(noprog_limit "$pol" 1)"
  recovery_load "$label"  # 重启计数按运行组持久化：续跑沿用，不清零
  start_client "$pol" "$cport"
  while true; do
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
      wait "$CLIENT_PID"; rc=$?; CLIENT_PID=""
      echo "CLIENT_EXIT policy=$pol rc=$rc"
      if (( rc == 0 )); then break; fi
      if (( rc == 3 )); then echo "RUN_BLOCKED reason=client policy=$pol"; return 3; fi
      # reset 额度耗尽（客户端退出 5）不重启，照实收尾
      if (( rc == 5 )); then echo "RESET_BUDGET_EXHAUSTED policy=$pol seat=$SEAT（客户端退出 5，不重启）"; return 5; fi
      # 账本读回仍有身份无权威终态（客户端退出 6）：不是基础设施故障，不重启 server 与客户端
      if (( rc == 6 )); then echo "RUN_INCOMPLETE_SEAT policy=$pol seat=$SEAT（客户端退出 6，不重启服务与客户端）"; return 6; fi
      CLIENT_RESTARTS=$((CLIENT_RESTARTS + 1)); recovery_save "$label"
      if (( CLIENT_RESTARTS > MAX_CLIENT_RESTARTS )); then echo "INFRA_EXHAUSTED policy=$pol client_restarts=$CLIENT_RESTARTS"; return 4; fi
      start_client "$pol" "$cport"
      continue
    fi
    if [[ -n "$SERVER_PID" ]] && ! server_alive; then
      echo "SERVER_DIED policy=$pol pid=$SERVER_PID"; SERVER_PID=""
      kill_group "$CLIENT_PID"; CLIENT_PID=""
      SERVER_RESTARTS=$((SERVER_RESTARTS + 1)); recovery_save "$label"
      if (( SERVER_RESTARTS > MAX_SERVER_RESTARTS )); then echo "INFRA_EXHAUSTED policy=$pol server_restarts=$SERVER_RESTARTS"; return 4; fi
      stop_server
      slog="$dir/server-$port-$(ts).log"
      restart_server "$pol" "$port" "$slog" "$dir" || return $?
      start_client "$pol" "$cport"
      continue
    fi
    progress_idle "$dir/progress.json"; idle=$IDLE  # 具名进展（phase／identity／step），不看 client.log
    if (( idle > noprog )); then
      echo "NO_PROGRESS policy=$pol idle_s=$idle limit_s=$noprog restarts=$NOPROG_RESTARTS sig=${PROG_SIG:-none}"
      kill_group "$CLIENT_PID"; CLIENT_PID=""
      if (( NOPROG_RESTARTS >= 1 )); then echo "INFRA_EXHAUSTED policy=$pol no_progress_twice"; return 4; fi
      NOPROG_RESTARTS=$((NOPROG_RESTARTS + 1)); recovery_save "$label"
      stop_server
      slog="$dir/server-$port-$(ts).log"
      restart_server "$pol" "$port" "$slog" "$dir" || return $?
      start_client "$pol" "$cport"
      continue
    fi
    sleep "$SEAT_POLL_S" & wait $!  # 后台 sleep + wait：收到 SIGTERM 时 trap 立即生效
  done
  return 0
}

# ---------------------------------------------------------------- 入口

parse_seat_args() {
  POLICY_SEED=""  # 直接执行时种子只认 --policy-seed，不继承环境变量
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seat) SEAT="$2"; shift 2;;
      --seat-idx) SEAT_IDX="$2"; shift 2;;
      --gpu) GPU="$2"; shift 2;;
      --cond) COND="$2"; shift 2;;
      --out) OUT="$2"; shift 2;;
      --identities) IDENTS="$2"; shift 2;;
      --policies) POLICIES="$2"; shift 2;;
      --cpus) CPUS="$2"; shift 2;;
      --order) ORDER="$2"; shift 2;;
      --framesamp-modul-ckpt) FRAMESAMP_MODUL_CKPT="$2"; shift 2;;
      --groundsg-ckpt) GROUNDSG_CKPT="$2"; shift 2;;
      --smvla-ckpt) SMVLA_CKPT="$2"; shift 2;;
      --pp-ckpt) PP_CKPT="$2"; shift 2;;
      --compile-cache) COMPILE_CACHE="$2"; shift 2;;
      --det) DET="$2"; shift 2;;
      --relay) RELAY="$2"; shift 2;;
      --limit) LIMIT="$2"; shift 2;;
      --noprog-s) NOPROG_S="$2"; shift 2;;
      --never-degrade) NEVER_DEGRADE="--never-degrade"; shift;;
      --episode-wall) WALL_ALL="$2"; shift 2;;
      --episode-wall-smvla) WALL_SMVLA="$2"; shift 2;;
      --episode-wall-framesamp-modul) WALL_FRAMESAMP_MODUL="$2"; shift 2;;
      --ledger-dir) LEDGER_DIR="$2"; shift 2;;
      --reset-budget) RESET_BUDGET="$2"; shift 2;;
      --infra-retry-budget) INFRA_RETRY_BUDGET="$2"; shift 2;;
      --rec-root) REC_ROOT="$2"; shift 2;;
      --trace-root) TRACE_ROOT="$2"; shift 2;;
      --openpi-data-home) OPENPI_HOME="$2"; shift 2;;
      --tokenizer-sha256) TOKENIZER_SHA="$2"; shift 2;;
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
      *) echo "未知参数 $1" >&2; exit 2;;
    esac
  done
  if [[ -z "$SEAT" || -z "$SEAT_IDX" || -z "$GPU" || -z "$COND" || -z "$OUT" ]]; then
    echo "缺少必需参数（--seat --seat-idx --gpu --cond --out）" >&2; exit 2
  fi
  if [[ -z "$IDENTS" ]]; then
    echo "缺少必需参数 --identities" >&2; exit 2
  fi
  local _w
  for _w in "$WALL_SMVLA" "$WALL_FRAMESAMP_MODUL" "$WALL_ALL"; do
    [[ -z "$_w" || "$_w" =~ ^[0-9]+$ ]] || { echo "--episode-wall*／须为非负整数秒" >&2; exit 2; }
  done
  [[ -z "$MAX_STEPS" || "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "--max-steps 须为非负整数" >&2; exit 2; }
  [[ -n "$LEDGER_DIR" ]] || { echo "必须给 --ledger-dir（env_client.py 两个数据集都要求账本）" >&2; exit 2; }
  [[ -z "$RESET_BUDGET" || "$RESET_BUDGET" =~ ^[0-9]+$ ]] || { echo "--reset-budget 须为非负整数（可不给：只计量）" >&2; exit 2; }
  local _d
  for _d in "$CONTEXT_DEADLINE" "$FIRST_INFER_DEADLINE" "$MEDIA_DEADLINE"; do
    [[ -z "$_d" || "$_d" =~ ^[0-9]+$ ]] || { echo "--*-deadline-s 须为非负整数秒" >&2; exit 2; }
  done
  [[ "$INFRA_RETRY_BUDGET" =~ ^[0-9]+$ ]] || { echo "必须给 --infra-retry-budget <非负整数>" >&2; exit 2; }
  mkdir -p "$OUT"
  OUT="$(cd "$OUT" && pwd)"  # 转绝对路径（server 子进程会 cd 进子模块）
  LEDGER_DIR="$(mkdir -p "$LEDGER_DIR" && cd "$LEDGER_DIR" && pwd)"
  [[ -n "$REC_ROOT" ]] && REC_ROOT="$(mkdir -p "$REC_ROOT" && cd "$REC_ROOT" && pwd)"
  [[ -n "$TRACE_ROOT" ]] && TRACE_ROOT="$(mkdir -p "$TRACE_ROOT" && cd "$TRACE_ROOT" && pwd)"
  [[ -n "$BUDGET_LEDGER" ]] && BUDGET_LEDGER="$(mkdir -p "$(dirname "$BUDGET_LEDGER")" && cd "$(dirname "$BUDGET_LEDGER")" && pwd)/$(basename "$BUDGET_LEDGER")"
  [[ -n "$OPENPI_HOME" && -d "$OPENPI_HOME" ]] && OPENPI_HOME="$(cd "$OPENPI_HOME" && pwd)"
  IDENTS="$(cd "$(dirname "$IDENTS")" && pwd)/$(basename "$IDENTS")"
  [[ -n "$CPUS" ]] && TASKSET=(taskset -c "$CPUS")
  return 0
}

main() {
  trap cleanup EXIT
  trap 'exit 143' TERM INT
  echo "SEAT_START seat=$SEAT idx=$SEAT_IDX gpu=$GPU cpus=${CPUS:-all} cond=$COND policies=$POLICIES host=$(hostname) \
git=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) compile_cache=$COMPILE_CACHE det=$DET relay=$RELAY \
dataset=${DATASET:-unset} max_steps=${MAX_STEPS:-unset} strict_cap=$STRICT_CAP variant=${GROUNDSG_VARIANT:-none} \
policy_seed=${POLICY_SEED:-unset}"
  step_cap_pairing || return 3
  variant_pairing "$POLICIES" || return 3
  policy_seed_check || return 3
  budget_args_check || return 3
  echo "SEAT_CONFIG ledger_dir=$LEDGER_DIR reset_budget=${RESET_BUDGET:-meter_only} infra_retry_budget=$INFRA_RETRY_BUDGET \
rec_root=${REC_ROOT:-<out>/<label>/rec} trace_root=${TRACE_ROOT:-none} wall_smvla=$(wall_of smvla) wall_framesamp_modul=$(wall_of perceptual-framesamp-modul) \
wall_groundsg=$(wall_of groundsg) wall_pp=$(wall_of pp) never_degrade=${NEVER_DEGRADE:+1} no_proxy=127.0.0.1,localhost \
client_py_ext=$SGEVAL_CLIENT_PY"
  local overall=0 pol pidx rc
  IFS=',' read -r -a POLS <<< "$POLICIES"
  for pol in "${POLS[@]}"; do
    pidx="$(pol_index "$pol")" || { echo "未知策略 $pol"; return 2; }
    run_policy "$pol" "$pidx"; rc=$?
    (( rc != 0 )) && overall=$rc
  done
  echo "全部完成 seat=$SEAT rc=$overall"
  return "$overall"
}

seat_entry() {
  parse_seat_args "$@"
  LOG="$OUT/seat-${SEAT}.log"
  # main 放后台、外层转发 TERM/INT：信号能到达 main 的 trap（cleanup 杀掉 setsid 起的 server／客户端进程组）
  # tee 忽略 TERM/INT/HUP/PIPE：slurmstepd 把信号发给 step 内全部进程时，日志管道须活到收尾行写完
  exec > >(trap '' TERM INT HUP PIPE; exec tee -p -a "$LOG") 2>&1
  main &
  MAIN_PID=$!
  trap 'kill -TERM "$MAIN_PID" 2>/dev/null' TERM INT
  local rc
  while true; do
    wait "$MAIN_PID"; rc=$?
    kill -0 "$MAIN_PID" 2>/dev/null || break
  done
  echo "EXIT_CODE=$rc"
  exit "$rc"
}

# 直接执行才进入口；被 source 时只提供变量缺省值与函数
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "orig_seat_lib.sh 只作库 source（原侧对照用）；新侧席位请用 dev-scripts/gl/run_eval_gl.sh" >&2
  exit 2
fi
