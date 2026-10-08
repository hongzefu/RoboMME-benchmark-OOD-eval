# shellcheck shell=bash
# 原侧观测器启动器共用函数（只含函数，由 run_orig_smvla.sh／run_orig_framesamp_modul.sh source；计划第二部分一节 S7）。
#
# - orig_pin_check <工作树> <期望提交> <标签> [<子模块路径> <子模块期望提交>]
#     起跑断言原版工作树 HEAD 等于期望提交、``git status --porcelain`` 为空（R1）、子模块提交相符；打印
#     ``ORIG_BRANCH_PIN=PASS|FAIL label= repo= head= expect= dirty=<n> [submodule=]``，不符打印
#     ``RUN_BLOCKED reason=orig_commit_mismatch|orig_worktree_dirty|orig_submodule_mismatch`` 并返回 1。
#     自测：仅当 ORIG_OBSERVER_SELFTEST=1 时 ORIG_PIN_SELFTEST_EXPECT 才可替换期望提交（启动器在自测模式下过了守卫即
#     打印 ``ORIG_SELFTEST_GUARD=PASS`` 退出，绝不进入评估），生产调用不认这个变量。
# - orig_pending_count <清单> <分片号> <逐局日志> [only_tasks]
#     本片尚无终态（success／fail／timeout）行的身份数（与原版续评跳过规则一致），打印一个整数。
# - orig_budget_prepare --manifest M --shard I --episode-log L --state S --resets N --route R [--only-tasks a,b]
#     每遍评估起跑前：本片未完成身份先 release 上一遍未结的 rid，再逐局 ``budget_ledger.py [顶层参数] reserve
#     --resets N --route R --key K``（orig_budget.py；顶层参数 BUDGET_LEDGER_ARGS 在子命令之前，缺省由账本读
#     SGEVAL_BUDGET_LEDGER）；失败打印 ``RUN_BLOCKED reason=budget`` 并返回 5。
# - orig_budget_settle --manifest M --shard I --episode-log L --state S [--only-tasks a,b]
#     每遍评估结束后：已有终态行的身份 commit 其 rid。
# - orig_stop_proxy
#     ``kill -TERM $PROXY_PID`` 后等到进程确实退出（上限 PROXY_STOP_TIMEOUT，缺省 900 s），超时 ``kill -9`` 并置
#     PROXY_FORCE_KILLED=1、打印 ``proxy_force_killed pid=``；确认退出前不清空 PROXY_PID（退出后记 PROXY_STOPPED_PID）。
# - orig_wait_seal <代理日志目录> <pid>
#     等 ``proxy-<pid>.done`` 出现（上限 SEAL_TIMEOUT，缺省 60 s）；置 SEALED=yes|timeout（被 kill -9 的直接 no）。
# - orig_finalize_framesamp_modul <python> <obs_dir> <rec_root> <manifest> <shard> <episodes.jsonl> [only_tasks]
#     日志封口后才跑 transparency_check.py（--manifest/--episodes），再跑 observer_status.py 写 observer-status.json
#     并打印 ``OBSERVER_COMPLETE=…``；全程不读写调用方的 RC（评估 EXIT_CODE 语义不变）。

orig_pin_check() {
  local repo="$1" expect="$2" label="$3" sub="${4:-}" sub_expect="${5:-}"
  if [[ "${ORIG_OBSERVER_SELFTEST:-0}" == 1 && -n "${ORIG_PIN_SELFTEST_EXPECT:-}" ]]; then
    expect="$ORIG_PIN_SELFTEST_EXPECT"
  fi
  if [[ "${ORIG_OBSERVER_SELFTEST:-0}" == 1 && -n "${ORIG_PIN_SELFTEST_SUB_EXPECT:-}" ]]; then
    sub_expect="$ORIG_PIN_SELFTEST_SUB_EXPECT"
  fi
  local head dirty sub_head="" st
  head=$(git -C "$repo" rev-parse HEAD 2>/dev/null) || head="<none>"
  st=$(git -C "$repo" status --porcelain 2>/dev/null) || st="<git status 失败>"
  dirty=$(printf '%s' "$st" | grep -c . || true)
  if [[ -n "$sub" ]]; then
    sub_head=$(git -C "$repo/$sub" rev-parse HEAD 2>/dev/null) || sub_head="<none>"
  fi
  local verdict=PASS reason=""
  if [[ "$head" != "$expect" ]]; then verdict=FAIL; reason=orig_commit_mismatch
  elif [[ "$dirty" != 0 ]]; then verdict=FAIL; reason=orig_worktree_dirty
  elif [[ -n "$sub" && "$sub_head" != "$sub_expect" ]]; then verdict=FAIL; reason=orig_submodule_mismatch
  fi
  echo "ORIG_BRANCH_PIN=$verdict label=$label repo=$repo head=$head expect=$expect dirty=$dirty${sub:+ submodule=$sub@$sub_head expect_submodule=$sub_expect}"
  if [[ "$verdict" != PASS ]]; then
    [[ "$dirty" != 0 ]] && printf '%s\n' "$st" | head -20 | sed 's/^/ORIG_DIRTY /'
    echo "RUN_BLOCKED reason=$reason label=$label repo=$repo"
    return 1
  fi
  return 0
}

orig_pending_count() {
  python3 - "$@" <<'PY'
import json, os, sys
manifest, shard, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] else None
done = set()
if os.path.exists(log):
    for l in open(log, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("status") in ("success", "fail", "timeout"):
                done.add((r["task"], int(r["source_episode"])))
n = 0
for l in open(manifest, encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    if int(r["shard"]) != shard or (only is not None and r["task"] not in only):
        continue
    n += (r["task"], int(r["source_episode"])) not in done
print(n)
PY
}

orig_budget_prepare() {
  python3 "$(dirname "${BASH_SOURCE[0]}")/orig_budget.py" prepare "$@"
}

orig_budget_settle() {
  python3 "$(dirname "${BASH_SOURCE[0]}")/orig_budget.py" settle "$@"
}

orig_stop_proxy() {
  PROXY_FORCE_KILLED="${PROXY_FORCE_KILLED:-0}"
  [[ -n "${PROXY_PID:-}" ]] || return 0
  local pid="$PROXY_PID" waited=0 limit="${PROXY_STOP_TIMEOUT:-900}"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && (( waited < limit )); do
      sleep 1; waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
      PROXY_FORCE_KILLED=1
      echo "proxy_force_killed pid=$pid waited_s=$waited"
      while kill -0 "$pid" 2>/dev/null; do sleep 1; done
    fi
  fi
  wait "$pid" 2>/dev/null || true
  PROXY_STOPPED_PID="$pid"
  PROXY_PID=""
  echo "PROXY_EXITED pid=$pid waited_s=$waited force_killed=$PROXY_FORCE_KILLED"
}

orig_wait_seal() {
  local dir="$1" pid="$2" waited=0 limit="${SEAL_TIMEOUT:-60}"
  if [[ "${PROXY_FORCE_KILLED:-0}" == 1 ]]; then SEALED=no; echo "PROXY_SEAL=no pid=$pid（被 kill -9）"; return 0; fi
  while [[ ! -f "$dir/proxy-$pid.done" ]] && (( waited < limit )); do sleep 1; waited=$((waited + 1)); done
  if [[ -f "$dir/proxy-$pid.done" ]]; then SEALED=yes; else SEALED=timeout; fi
  echo "PROXY_SEAL=$SEALED pid=$pid waited_s=$waited"
}

orig_finalize_framesamp_modul() {
  local py="$1" obs="$2" rec="$3" manifest="$4" shard="$5" eplog="$6" only="${7:-}"
  local crc=0
  rm -f "$rec/transparency.json"  # 续跑时不读上一遍的旧报告
  if [[ "${SEALED:-no}" == yes ]]; then
    "$py" "$obs/transparency_check.py" --rec-root "$rec" --out "$rec/transparency.json" \
      --manifest "$manifest" --shard "$shard" --episodes "$rec" ${only:+--only-tasks "$only"}
    crc=$?
  else
    echo "TRANSPARENCY_SKIPPED sealed=${SEALED:-no}（日志未封口，不对账）"
    crc=""
  fi
  python3 "$obs/observer_status.py" --rec-root "$rec" --policy perceptual-framesamp-modul --report "$rec/transparency.json" \
    ${crc:+--checker-rc "$crc"} --sealed "${SEALED:-no}" --proxy-force-killed "${PROXY_FORCE_KILLED:-0}" \
    --episode-log "$eplog" --out "$rec/observer-status.json"
}
