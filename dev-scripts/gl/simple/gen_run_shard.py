"""由 run-config JSON 机械生成单卡分片脚本 run_shard.sh（1011 最简四卡方案，计划第二部分 §二）。

用法：
  uv run --no-sync python dev-scripts/gl/simple/gen_run_shard.py \
      --config <root>/control/run-config-resume-03.json --out <输出路径>

做的事：
  1. 读 run-config，断言五个模型的 env 块除 BENCH_PY 外逐字相同（否则一份环境块不能共用）；
  2. 每个模型的 args 去掉含 {work_dir} 占位的参数对（--work-dir 与 server_dir 由分片脚本自己给）；
  3. 给 groundsg 两个变体追加 --cfg groundsg_keep_official=false（--no-render 下与 seat.py 隐式行为一致）；
  4. 按 oracle → framesamp → pp → smvla → qwenvl 顺序写出 run_model 行；
  5. 末行打印 SHARD_SCRIPT=PASS sha256=<输出文件 sha256>。
模板正文是计划 §二 的全文，改模板等于改计划，须先改计划再改这里。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

# 模型顺序：前两个从未全量跑过，放最前面最早暴露环境问题；qwenvl 最慢且已完成最多，放最后（计划 §四）。
ORDER = ("oracle", "framesamp", "pp", "smvla", "qwenvl")
GROUNDSG_EXTRA = ("--cfg", "groundsg_keep_official=false")

# 四个任务组（按 export_eval_identities.py::TASK_SECONDS × 50 局贪心四分，计划 §四）；端口沿用 02 轮席位槽。
# case 表是模板正文的一部分（逐字取自计划 §二，含其原有对齐空格），生成时只做解析核对：
# 四组两两不相交、并集恰为 16 任务、端口与 run-config 的 seats[].port_base 逐个相同。
CASES = """  g1) TASKS=PickHighlight,SwingXtimes,StopCube,VideoUnmaskSwap; PORT=22400;;
  g2) TASKS=BinFill,VideoUnmask,ButtonUnmaskSwap,VideoRepick;    PORT=22500;;
  g3) TASKS=MoveCube,PickXtimes,RouteStick,VideoPlaceButton;     PORT=22600;;
  g4) TASKS=InsertPeg,ButtonUnmask,PatternLock,VideoPlaceOrder;  PORT=22700;;"""
CASE_RE = re.compile(r"^  (g[1-4])\) TASKS=([A-Za-z,]+); +PORT=(\d+);;$")

HEAD = """#!/usr/bin/env bash
# 五模型 OOD 评估：单卡分片脚本（最简方案）。用法：bash run_shard.sh <g1|g2|g3|g4>
# 一张卡 = 一个任务组；依次跑五个模型，每个模型只跑本组任务；已有 result.json 的局由 scripts/evaluate.py 自动跳过。
# 模型参数与环境变量逐字取自 control/@@CONFIG_NAME@@（exec_commit @@EXEC_SHORT@@），由生成脚本机械写出，禁止手改路径。
# 退出码：0 全部模型 rc=0；1 某模型三次尝试仍非零；2 参数错；3 输出根不是实体目录或 infra 级联（env_build 1 局／其它 infra 2 局）；4 重试登记超封顶；5 无进展 2700 s 且三次仍非零按 1 计；143 收到 TERM。
set -uo pipefail
ROOT="${SHARD_ROOT:-@@ROOT@@}"   # SHARD_ROOT 只供本机夹具演练覆盖
CODE="${SHARD_CODE:-$ROOT/@@CODE_REL@@}"
OUT=$ROOT/gl
G="${1:?用法: run_shard.sh <g1|g2|g3|g4>}"
case "$G" in
@@CASES@@
  *) echo "未知分组 $G"; echo "EXIT_CODE=2"; exit 2;;
esac
WORK=$ROOT/simple/$G/work; LOG=$ROOT/simple/logs/shard-$G.log; RLOG=$ROOT/simple/logs/reattempts-$G.log; HALF=$ROOT/simple/halfdone/$G
[[ -d "$OUT" && ! -L "$OUT" ]] || { echo "RUN_BLOCKED reason=out_not_dir out=$OUT"; echo "EXIT_CODE=3"; exit 3; }   # 第 14 条：输出根必须是实体目录
mkdir -p "$WORK/servers" "$ROOT/simple/logs" "$HALF" || { echo "EXIT_CODE=3"; exit 3; }
exec > >(exec tee -a "$LOG") 2>&1; TEE_PID=$!
@@EXPORTS@@
export PYTHONUNBUFFERED=1 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
for f in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do [[ -f "$f" ]] && { export VK_ICD_FILENAMES="$f"; break; }; done
MAIN_PY="${SHARD_MAIN_PY:-@@MAIN_PY@@}"
"""

BODY = """NOPROG_S="${SHARD_NOPROG_S:-2700}"; POLL_S="${SHARD_POLL_S:-60}"
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
  local list; list=$(task_dirs "$1" | while read -r d; do [[ -f "$d/result.json" && "$d/result.json" -nt "$T0FILE" ]] && grep -lE "\\"infra_reason\\": *\\"($3)\\"" "$d/result.json" 2>/dev/null; done); [[ -z "$list" ]] && echo 0 || echo "$list" | wc -l; }
newest_mtime() {  # $1=标签：本分片任务目录下最新文件的 epoch（无文件给 0）
  local m; m=$(task_dirs "$1" | while read -r d; do find "$d" -maxdepth 1 -type f -printf '%T@\\n' 2>/dev/null; done | sort -n | tail -n 1); echo "${m%%.*}"; }
run_model() {  # $1=标签 $2=解释器 $3..=模型参数；每模型最多 3 次尝试；每次尝试前登记并移存半截目录（口径 8，封顶 3 局／分片）
  local label="$1" py="$2" rc=1 try t0 total n_env n_inf last idle; shift 2
  for try in 1 2 3; do
    stop_left
    total=$(register_halfdone "$label"); echo "REATTEMPT group=$G label=$label try=$try total=$total cap=3"; cap_check "$total"
    t0=$(date +%s); T0FILE=$(mktemp); touch -d "@$t0" "$T0FILE"
    echo "MODEL_START group=$G label=$label try=$try host=$(hostname) job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} cuda=${CUDA_VISIBLE_DEVICES:-unset} $(date -Is)"
    ( cd "$CODE" && exec "$py" scripts/evaluate.py --dataset ood --seed 7 --tasks "$TASKS" --out "$OUT" --no-render \\
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
for lab in @@LABELS@@; do total=$(register_halfdone "$lab"); done; echo "REATTEMPT group=$G label=all try=start total=$total cap=3"; cap_check "$total"
overall=0
@@RUN_LINES@@
(( overall == 0 )) || overall=1
finish "$overall"
"""


#: ckpt 磁盘真实路径里的旧目录段；含它的 run_model 行在行尾追加下面的注释（bash 语义不变）
LEGACY_DIR_SEG = "/mme/"  # 历史目录名：ckpt 磁盘真实路径
LEGACY_DIR_NOTE = " # 历史目录名：ckpt 磁盘真实路径"


def die(msg: str) -> None:
    print(f"SHARD_SCRIPT=FAIL reason={msg}")
    sys.exit(1)


def strip_work_dir(args: list[str]) -> list[str]:
    """去掉值里含 {work_dir} 占位的参数对（--work-dir {work_dir}、--cfg server_dir={work_dir}/servers）。"""
    out, i = [], 0
    while i < len(args):
        if i + 1 < len(args) and args[i].startswith("--") and "{work_dir}" in args[i + 1]:
            i += 2
            continue
        if "{work_dir}" in args[i]:
            die(f"孤立的 work_dir 占位 {args[i]!r}")
        out.append(args[i])
        i += 1
    return out


def render(cfg: dict, config_name: str, expected_tasks: set[str] | None) -> str:
    models = cfg["models"]
    if set(models) != set(ORDER):
        die(f"模型集合不符 {sorted(models)}")
    envs = {k: {e: v for e, v in m["env"].items() if e != "BENCH_PY"} for k, m in models.items()}
    ref = envs[ORDER[0]]
    for k in ORDER:
        if envs[k] != ref or list(envs[k]) != list(ref):
            die(f"env 块除 BENCH_PY 外与 {ORDER[0]} 不同：{k}")
    root = cfg["root"].rstrip("/")
    if cfg["gl_root"].rstrip("/") != f"{root}/gl":
        die("gl_root 不是 root/gl")
    eval_root = ref["ROBOMME_EVAL_ROOT"]
    if not eval_root.startswith(root + "/"):
        die("ROBOMME_EVAL_ROOT 不在 root 下")
    code_rel = eval_root[len(root) + 1:]
    ports = [s["port_base"] for s in cfg["seats"]]
    groups = []
    for line in CASES.splitlines():
        mt = CASE_RE.match(line)
        if not mt:
            die(f"case 表行格式不符 {line!r}")
        groups.append((mt.group(1), mt.group(2).split(","), int(mt.group(3))))
    if [g for g, _, _ in groups] != ["g1", "g2", "g3", "g4"]:
        die("case 表分组名必须依次为 g1..g4")
    if [p for _, _, p in groups] != ports:
        die(f"case 表端口 {[p for _, _, p in groups]} 与 seats {ports} 不符")
    all_tasks = [t for _, ts, _ in groups for t in ts]
    if len(all_tasks) != 16 or len(set(all_tasks)) != 16:
        die("四组任务必须两两不相交且并集为 16")
    if expected_tasks is not None and set(all_tasks) != expected_tasks:
        die(f"case 表任务集合与身份清单不符：{sorted(set(all_tasks) ^ expected_tasks)}")

    def safe(s: str) -> str:
        if shlex.quote(s) != s:
            die(f"参数需要引号，不允许机械拼接：{s!r}")
        return s

    exports = "\n".join(f"export {k}={safe(v)}" for k, v in ref.items())

    lines = []
    for k in ORDER:
        m = models[k]
        args = strip_work_dir(list(m["args"]))
        if m["policy"] == "groundsg":
            args += list(GROUNDSG_EXTRA)
        argv = [m["label"], m["env"]["BENCH_PY"], "--model", m["policy"], *args]
        line = "run_model " + " ".join(safe(a) for a in argv) + " || overall=$?"
        if any(LEGACY_DIR_SEG in a for a in argv):
            # 行尾 bash 注释不改语义，只为让官方名残留检查（tests/static/test_official_names.py）按「历史目录名」放行
            line += LEGACY_DIR_NOTE
        lines.append(line)

    text = HEAD + BODY
    for key, val in (
        ("CONFIG_NAME", config_name),
        ("EXEC_SHORT", cfg["exec_commit"][:7]),
        ("ROOT", safe(root)),
        ("CODE_REL", safe(code_rel)),
        ("CASES", CASES),
        ("EXPORTS", exports),
        ("MAIN_PY", safe(cfg["python"])),
        ("LABELS", " ".join(safe(models[k]["label"]) for k in ORDER)),
        ("RUN_LINES", "\n".join(lines)),
    ):
        marker = f"@@{key}@@"
        if text.count(marker) != 1:
            die(f"模板占位 {marker} 出现 {text.count(marker)} 次")
        text = text.replace(marker, val)
    if "@@" in text:
        die("模板残留占位")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="由 run-config JSON 生成 run_shard.sh")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    # 若 run-config 同目录有身份清单（control/identities.jsonl），顺带核对 case 表任务集合与它相同。
    ident = a.config.parent / "identities.jsonl"
    expected = None
    if ident.is_file():
        expected = {json.loads(x)["task"] for x in ident.read_text().splitlines() if x.strip()}
    text = render(cfg, a.config.name, expected)
    data = text.encode("utf-8")
    a.out.write_bytes(data)
    print(f"SHARD_SCRIPT=PASS sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
