"""EVAL_WIRING：配对核对、服务命令与守卫在场（拆仓后布局：新侧席位 ``dev-scripts/gl/seat.py``、原侧对照
``dev-scripts/orig/``）。

不起任何进程（除 bash 本身）：

- 原侧席位函数库 ``dev-scripts/orig/orig_seat_lib.sh``（被 source 时只定义函数；新侧客户端构造 ``build_client_cmd`` 已停用，
  调用即 ``RUN_BLOCKED reason=legacy_client``）：10 条服务路线（smvla、perceptual-framesamp-modul 走 ood；groundsg 三个
  变体与 pp 各走 hard-verify 与 ood）的服务 argv——perceptual-framesamp-modul／groundsg 走外壳
  ``src/robomme_ood_eval/servers/policy_server_wrap.py --seed=<种子>``、``--policy.dir`` 与 history_config 期望，pp 的
  ``-m ponderpounce.eval.robomme_server --args.seed <种子>``，smvla 的 ``src/robomme_ood_eval/servers/smvla_server.py
  --policy-seed <种子>``，均带服务元数据路径；端口策略号。
- 原侧 3 条路线（groundsg 两个变体、pp，hard-verify）：驱动脚本 ``dev-scripts/orig/<驱动>``、``--max-steps 1300``、
  ``--attempt``／``--only``、``--variant``、adapter 只在 QwenVL 变体。
- 配对核对拦得住错配：数据集与步数（含缺失、未知数据集、strict-cap 配错）、groundsg 变体与 adapter（含给了变体却不跑
  groundsg、adapter 目录不存在）——bash 侧（原侧函数库）与 Python 侧（``seat.entry_blockers``）各核一遍。
- 守卫在场：``run_official_hard.sh``、``pair_seat.sh`` 都 source ``orig_seat_lib.sh`` 且不另定义共享函数；
  ``run_eval_gl.sh`` 不再 source 旧函数库、不再起服务端，只调 ``seat.py``，带无进展看门狗与 ``EXIT_CODE=``。

通过时打印 ``EVAL_WIRING=PASS routes=<n> dataset_mismatch=0 variant_mismatch=0``。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

import eval_fakes as F

EO = F.REPO / "dev-scripts" / "orig"
GL = F.REPO / "dev-scripts" / "gl"
SERVERS = F.REPO / "src" / "robomme_ood_eval" / "servers"

if shutil.which("bash") is None:  # pragma: no cover
    pytest.skip("未验证：缺 bash", allow_module_level=True)

NEW_ROUTES = [  # (策略, 变体, 数据集)
    ("smvla", "", "ood"), ("perceptual-framesamp-modul", "", "ood"),
    ("groundsg", "ground-sg-oracle", "hard-verify"), ("groundsg", "ground-sg-qwenvl", "hard-verify"), ("pp", "", "hard-verify"),
    ("groundsg", "ground-sg-oracle", "ood"), ("groundsg", "ground-sg-qwenvl", "ood"), ("pp", "", "ood"),
    ("groundsg", "ground-sg-memer", "hard-verify"), ("groundsg", "ground-sg-memer", "ood"),
]
ORIG_ROUTES = [("groundsg", "ground-sg-oracle"), ("groundsg", "ground-sg-qwenvl"), ("pp", "")]
CAP = {"ood": ("1800", "1"), "hard-verify": ("1300", "0")}  # 手写的启动约定（第三阶段 ood↔1800）
SEED = "42"  # 非旧默认值（7／0）的种子：服务与客户端都必须照传
POL_IDX = {"smvla": 0, "perceptual-framesamp-modul": 1, "groundsg": 2, "pp": 3}

LIB_NEW = r'''
set -u
source "$EO/orig_seat_lib.sh"
OUT=/o; IDENTS=/s.json; SEAT=T; SEAT_IDX=5; GPU=0; COND=C; LEDGER_DIR=/l; RESET_BUDGET=7; INFRA_RETRY_BUDGET=3
REC_ROOT=/r; TRACE_ROOT=/t; NEVER_DEGRADE=--never-degrade; LIMIT=0
FRAMESAMP_MODUL_CKPT=/ck/perceptual-framesamp-modul; GROUNDSG_CKPT=/ck/sg; PP_CKPT=/ck/pp; SGEVAL_CLIENT_PY=/py/client; BENCH_PY=/py/bench
MME_VLA_PY=/py/perceptual-framesamp-modul; PP_PY=/py/pp; SMVLA_PY=/py/smvla; OPENPI_HOME=/openpi
DATASET="$W_DATASET"; MAX_STEPS="$W_MAX"; STRICT_CAP="$W_STRICT"; GROUNDSG_VARIANT="$W_VARIANT"; QWENVL_ADAPTER="$W_ADAPTER"
MEMER_ADAPTER="${W_MEMER:-}"; POLICY_SEED="$W_SEED"; BUDGET_LEDGER=/b/budget.jsonl; TRAJECTORY_CAP=870; SHARED_INFRA_CAP=50
EXPIRED_CAP=50; PLANNED_FIRST_TRIES=821
step_cap_pairing; echo "PAIRING_RC=$?"
variant_pairing "$W_POLS"; echo "VARIANT_RC=$?"
if [[ -n "${W_BUILD:-}" ]]; then
  echo "LABEL=$(pol_label "$W_POLS")"
  echo "PIDX=$(pol_index "$W_POLS")"
  echo "YAML=$(yaml_expect_of "$W_POLS")"
  build_client_cmd "$W_POLS" 18123 900 600; echo "CLIENT_RC=$?"
  build_server_cmd "$W_POLS" 18123
  echo "SRVDIR=$SRV_DIR"
  for a in "${SRV_ARGV[@]}"; do printf 'SRV %s\n' "$a"; done
fi
'''

LIB_ORIG = r'''
set -u
source "$EO/run_official_hard.sh"
POLICY="$W_POLS"; GROUNDSG_VARIANT="$W_VARIANT"; QWENVL_ADAPTER="$W_ADAPTER"; SGEVAL_CLIENT_PY=/py/client
SHARD=/s.json; RUN_OUT=/run; MAX_STEPS=1300; DATASET=hard-verify
variant_pairing "$POLICY"; echo "VARIANT_RC=$?"
build_runner_cmd 2 k1,k2 18555
for a in "${RUN_ARGV[@]}"; do printf 'RUN %s\n' "$a"; done
for a in "${RUN_ENV[@]}"; do printf 'RUNENV %s\n' "$a"; done
'''


def _bash(script: str, **env) -> dict:
    e = dict(os.environ, EO=str(EO), W_POLS="smvla", W_DATASET="ood", W_MAX="1800", W_STRICT="1",
             W_VARIANT="", W_ADAPTER="", W_MEMER="", W_SEED=SEED)
    e.pop("POLICY_SEED", None)
    e.update({k: str(v) for k, v in env.items()})
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=60)
    out: dict = {"raw": p.stdout + p.stderr, "rc": p.returncode}
    for line in p.stdout.splitlines():
        for tag in ("CLI", "CLIENV", "SRV", "RUN", "RUNENV"):
            if line.startswith(tag + " "):
                out.setdefault(tag, []).append(line[len(tag) + 1:])
        m = re.match(r"^([A-Z_]+)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _opt(argv: list[str], name: str):
    return argv[argv.index(name) + 1] if name in argv else None


def _check_new_route(pol, variant, dataset, adapter_dir) -> tuple[int, int, list[str]]:
    """原侧函数库对一条路线的配对核对与服务命令。返回 (数据集类不符数, 变体类不符数, 说明)。"""
    max_steps, strict = CAP[dataset]
    adapter = str(adapter_dir) if variant == "ground-sg-qwenvl" else ""
    memer = str(adapter_dir) if variant == "ground-sg-memer" else ""
    r = _bash(LIB_NEW, W_POLS=pol, W_DATASET=dataset, W_MAX=max_steps, W_STRICT=strict, W_VARIANT=variant,
              W_ADAPTER=adapter, W_MEMER=memer, W_BUILD=1)
    ds_bad, var_bad, why = 0, 0, []
    srv = r.get("SRV", [])
    label = f"groundsg-{variant}" if pol == "groundsg" else pol
    if r.get("PAIRING_RC") != "0" or r.get("VARIANT_RC") != "0":
        why.append(f"核对未通过 {r['raw'][-400:]}")
        return 1, 1, why
    if int(r["PIDX"]) != POL_IDX[pol] or r["LABEL"] != label:
        ds_bad += 1; why.append("策略号／标签")
    if r.get("CLIENT_RC") != "3" or "RUN_BLOCKED reason=legacy_client" not in r["raw"]:
        var_bad += 1; why.append("旧客户端构造未停用")
    if pol in ("perceptual-framesamp-modul", "groundsg"):
        want_dir = "/ck/sg" if pol == "groundsg" else "/ck/perceptual-framesamp-modul"
        want_yaml = "symbolic-grounded-subgoal.yaml" if pol == "groundsg" else "perceptual-framesamp-modul.yaml"
        if srv[:2] != ["/py/perceptual-framesamp-modul", str(SERVERS / "policy_server_wrap.py")] \
                or f"--policy.dir={want_dir}" not in srv or f"--seed={SEED}" not in srv or "--port=18123" not in srv \
                or f"--sgeval-metadata-out=/o/{label}/server-metadata-18123.json" not in srv or r["YAML"] != want_yaml \
                or any(a.startswith("--seed=") and a != f"--seed={SEED}" for a in srv):
            var_bad += 1; why.append(f"perceptual-framesamp-modul 服务 {srv}")
    elif pol == "pp":
        if srv[:3] != ["/py/pp", "-m", "ponderpounce.eval.robomme_server"] or _opt(srv, "--args.seed") != SEED \
                or _opt(srv, "--args.checkpoint_path") != "/ck/pp" or _opt(srv, "--args.device") != "cuda:0" \
                or _opt(srv, "--port") != "18123" or not r["SRVDIR"].endswith("/third_party/PonderPounce"):
            var_bad += 1; why.append(f"pp 服务 {srv}")
    else:
        if srv[:2] != ["/py/smvla", "src/robomme_ood_eval/servers/smvla_server.py"] or _opt(srv, "--policy-seed") != SEED \
                or _opt(srv, "--metadata_out") != "/o/smvla/server-metadata-18123.json":
            var_bad += 1; why.append(f"smvla 服务 {srv}")
    return ds_bad, var_bad, why


def _check_orig_route(pol, variant, adapter_dir) -> tuple[int, list[str]]:
    adapter = str(adapter_dir) if variant == "ground-sg-qwenvl" else ""
    r = _bash(LIB_ORIG, W_POLS=pol, W_VARIANT=variant, W_ADAPTER=adapter)
    run, why, bad = r.get("RUN", []), [], 0
    script = "official_hard_runner.py" if pol == "groundsg" else "pp_official_runner.py"
    if r.get("VARIANT_RC") != "0" or run[:2] != ["/py/client", f"dev-scripts/orig/{script}"]:
        bad += 1; why.append(f"驱动 {run} {r['raw'][-300:]}")
    if _opt(run, "--max-steps") != "1300" or _opt(run, "--attempt") != "2" or _opt(run, "--only") != "k1,k2" \
            or _opt(run, "--port") != "18555" or _opt(run, "--shard") != "/s.json" or _opt(run, "--out") != "/run":
        bad += 1; why.append(f"驱动参数 {run}")
    if (_opt(run, "--variant") or "") != (variant if pol == "groundsg" else ""):
        bad += 1; why.append("variant")
    if (_opt(run, "--qwenvl-groundsg-adapter") or "") != adapter:
        bad += 1; why.append("adapter")
    if (pol == "groundsg") != ("USE_HF=1" in r.get("RUNENV", [])):
        bad += 1; why.append("USE_HF")
    return bad, why


BAD_CAPS = [("hard-verify", "1800", "0"), ("hard-verify", "1300", "1"), ("ood", "1300", "1"),
            ("ood", "1800", "0"), ("ood", "1600", "1"), ("", "1800", "1"), ("ood", "", "1"), ("bogus", "1800", "1")]


def _bad_variants(adapter_dir) -> list[tuple[str, str, str, str]]:
    """(策略, 变体, QwenVL adapter, MemER adapter)。"""
    a = str(adapter_dir)
    return [("groundsg", "", "", ""), ("groundsg", "ground-sg-qwenvl", "", ""),
            ("groundsg", "ground-sg-qwenvl", "/nonexistent/adapter", ""),
            ("groundsg", "ground-sg-oracle", a, ""), ("groundsg", "bogus", "", ""), ("pp", "ground-sg-oracle", "", ""),
            ("smvla,perceptual-framesamp-modul", "", a, ""),
            ("groundsg", "ground-sg-memer", "", ""), ("groundsg", "ground-sg-memer", "", "/nonexistent/memer"),
            ("groundsg", "ground-sg-memer", a, a), ("groundsg", "ground-sg-qwenvl", a, a),
            ("groundsg", "ground-sg-oracle", "", a), ("pp", "", "", a)]


GUARD_USES = {
    "run_official_hard.sh": ("pick_port", "server_alive", "noprog_limit", "idle_s", "NO_PROGRESS", "start_server",
                             "step_cap_pairing", "variant_pairing", "trap ", "EXIT_CODE=", "OFFICIAL_SEAT_DONE"),
    "pair_seat.sh": ("port_busy", "kill -0", "step_cap_pairing", "trap ", "EXIT_CODE=", "PAIR_SEAT_DONE",
                     "../gl/run_eval_gl.sh"),
}
GL_USES = ("dev-scripts/gl/seat.py", "GL_NOPROGRESS", "--client-restarts", "SERVER_LEFT", "--stop-server", "EXIT_CODE=",
           "GL_SEAT_DONE", "RUN_INPUTS=")
SHARED_FUNCS = ("port_busy", "pick_port", "noprog_limit", "idle_s", "start_server", "kill_group", "publish_dir",
                "transcode_episode_dir")


def _guard_problems() -> list[str]:
    probs = []
    for name, uses in GUARD_USES.items():
        src = (EO / name).read_text(encoding="utf-8")
        if not re.search(r'^source "\$[^"]*/orig_seat_lib\.sh"', src, re.M):
            probs.append(f"{name} 未 source orig_seat_lib.sh")
        for u in uses:
            if u not in src:
                probs.append(f"{name} 缺 {u}")
        for fn in SHARED_FUNCS:
            if re.search(rf"^\s*{fn}\(\)\s*\{{", src, re.M):
                probs.append(f"{name} 另定义了 {fn}()")
    lib = (EO / "orig_seat_lib.sh").read_text(encoding="utf-8")
    for fn in SHARED_FUNCS + ("step_cap_pairing", "variant_pairing", "server_alive", "note_epoch", "epoch_annotate"):
        if not re.search(rf"^{fn}\(\)\s*\{{", lib, re.M):
            probs.append(f"orig_seat_lib.sh 缺 {fn}()")
    if 'if [[ "${BASH_SOURCE[0]}" == "$0" ]]' not in lib:
        probs.append("orig_seat_lib.sh 被 source 时会运行")
    gl = (GL / "run_eval_gl.sh").read_text(encoding="utf-8")
    code = re.sub(r"#.*", "", gl)
    for u in GL_USES:
        if u not in gl:
            probs.append(f"run_eval_gl.sh 缺 {u}")
    if re.search(r"^\s*source ", code, re.M):
        probs.append("run_eval_gl.sh 不应再 source 旧函数库")
    if "start_server" in code or "serve_policy" in code:
        probs.append("run_eval_gl.sh 不应在 shell 里起服务端")
    return probs


def _seat_variant_blocked(pols: str, variant: str, adp: str, memer: str) -> bool:
    """Python 侧（seat.entry_blockers）同一组错配也必须拦：多策略逗号列表在席位层不存在（一个席位一个模型），按第一个算。"""
    ec = F.env_client()
    args = F.seat_args(F.REPO / "unused", pols.split(",")[0], groundsg_variant=variant or None,
                       qwenvl_groundsg_adapter=adp or None, memer_adapter=memer or None, budget_ledger="/b.jsonl",
                       trajectory_cap=26, reset_cap=60, shared_infra_cap=0, expired_cap=0, planned_first_tries=26)
    blk = ec.entry_blockers(args)
    return blk is not None and blk[0] == "variant_pairing"


def test_eval_wiring(tmp_path):
    adapter = tmp_path / "qwenvl-adapter"
    adapter.mkdir()
    routes = ds_mismatch = var_mismatch = 0
    problems: list[str] = []
    for pol, variant, dataset in NEW_ROUTES:
        d, v, why = _check_new_route(pol, variant, dataset, adapter)
        routes += 1
        ds_mismatch += d
        var_mismatch += v
        problems += [f"new {pol}/{variant}/{dataset}: {w}" for w in why]
    for pol, variant in ORIG_ROUTES:
        bad, why = _check_orig_route(pol, variant, adapter)
        routes += 1
        var_mismatch += bad
        problems += [f"orig {pol}/{variant}: {w}" for w in why]
    # 错配必须被拦（RUN_BLOCKED + 返回 3）；没拦住的计入 mismatch
    for dataset, max_steps, strict in BAD_CAPS:
        r = _bash(LIB_NEW, W_DATASET=dataset, W_MAX=max_steps, W_STRICT=strict)
        if r.get("PAIRING_RC") != "3" or "RUN_BLOCKED reason=step_cap_pairing" not in r["raw"]:
            ds_mismatch += 1
            problems.append(f"配对未拦住 {dataset}/{max_steps}/{strict}")
    for pols, variant, adp, memer in _bad_variants(adapter):
        r = _bash(LIB_NEW, W_POLS=pols, W_VARIANT=variant, W_ADAPTER=adp, W_MEMER=memer)
        if r.get("VARIANT_RC") != "3" or "RUN_BLOCKED reason=variant_pairing" not in r["raw"]:
            var_mismatch += 1
            problems.append(f"变体未拦住 {pols}/{variant}/{adp}/{memer}")
        if pols != "smvla,perceptual-framesamp-modul" and not _seat_variant_blocked(pols, variant, adp, memer):
            var_mismatch += 1
            problems.append(f"seat.py 变体未拦住 {pols}/{variant}/{adp}/{memer}")
    problems += _guard_problems()
    assert problems == [], "\n".join(problems)
    assert routes == len(NEW_ROUTES) + len(ORIG_ROUTES) == 13
    assert ds_mismatch == 0 and var_mismatch == 0
    print(f"EVAL_WIRING=PASS routes={routes} dataset_mismatch={ds_mismatch} variant_mismatch={var_mismatch}")
