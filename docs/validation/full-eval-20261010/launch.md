# 分阶段全量 OOD 评估：启动记录

## 目的与用户决定

1. 用户原话：「/data/hongzefu/RoboMME-benchmark-OOD-eval/1010-full-eval-staged-plan.html 开工」。授权执行该计划的四阶段连续评估，不设阶段放行。
2. 用户对主检出 SimpleMemVLA 的脏状态回复：「这些应该是noise」。只读逐字节核对确认 16 个文件内容全部等于锁定提交，差异来自 LFS clean 过滤器。主检出不修改、不清理。
3. 用户询问到期重试预算含义后追加：「把现在的尝试上线提高十倍，不要有任何的阻塞，一路跑到底。」以此覆盖旧预算；正式身份不增，正常任务失败不重跑，源码身份与媒体正确性守卫保留。

## 版本与代码状态

实施起点为 `ee04d65a578c058be2659e456a35da6ebe341021`（1.65），不是计划编写时的 1.62。两个工具在独立分支 `codex/full-eval-s0-tools`、工作树 `artifacts/worktrees/full-eval-s0-tools` 实现；审查与合并提交从实际历史接续编号。起跑提交在全部工具与本档案提交后补记；正式执行副本必须与该提交及五个 gitlink 一致。

旧 NFS 执行副本存在权限模式噪声与未跟踪的 `src/robomme_hard_eval/`，不执行计划旧模板的 `rsync --delete`。本轮另建 `artifacts/full-eval-20261010/runtime-code`，只从固定提交及各子模块锁定提交恢复源码，在本机准备后 rsync 至 NFS 对应本轮目录；不在 GL 进行 git 操作、不复制原主检出的脏文件、不改模型源码。独立副本的 LFS 过滤器仅取消上述明确文本文件的过滤，源码字节仍需逐文件核对。

## 数据与种子

只评估 `ood`，实际执行 cap 为 1800 步 strict；排除 MoveCube、InsertPeg。任务为 BinFill、StopCube、PickXtimes、SwingXtimes、ButtonUnmask、VideoUnmask、VideoUnmaskSwap、ButtonUnmaskSwap、PickHighlight、VideoRepick、VideoPlaceButton、VideoPlaceOrder、PatternLock、RouteStick。

每模型每种子为 7 任务 × 2 档 × 25 局 + 2 任务 × 4 档（13/13/12/12 局）+ 3 任务 × 3 档（17/17/16 局）+ 2 任务 × 5 档 × 10 局 = 700 局；六模型 × 三种子 = 12600 个正式身份。模型种子顺序 7 → 0 → 42，不替换身份行内的环境种子。

A：六模型 × 14 任务 × 每任务跨档轮转 5 局 = 420 局；B：六模型 × 14 任务 × 剩余 45 局 = 3780 局；C、D 各六模型 × 上述 700 局 = 4200 局。六变体各 1 个最小 smoke 另计入 A 账本，不增加正式清单。MemER 的同队列 smoke 用两个席位竞争同一身份，禁止重复生成；并发多身份领取另由不触发仿真的夹具核验。

## 预算

| 阶段 | 轨迹硬上限 | reset 硬上限 | infra 重试上限 | 到期重试上限 | 计划首试 |
|---|---:|---:|---:|---:|---:|
| A（含 smoke） | 4680 | 9360 | 420 | 420 | 426 |
| B | 41580 | 83160 | 3780 | 3780 | 3780 |
| C | 46200 | 92400 | 4200 | 4200 | 4200 |
| D | 46200 | 92400 | 4200 | 4200 | 4200 |

每尝试预占 build 与 reset 两次。恢复合计仍受总尝试上限减计划首试约束；每身份最多首试加一次基础设施／到期重试，正常失败不重试。Astra 不启动，局数与费用均为零。阶段账本独立且不重置，已有完成身份复用。

## 环境、资源与路径

本机 sled-vail，2 × RTX 6000 Ada，只作准备、单测与核验；所有模型评估在 GL 的 A40 上运行。四个用户既有占位作业 63431430、63431431、63431432、63431433 均已核实 RUNNING，分别位于 gl1525、gl1512、gl1512、gl1513。它们的原时长为五天，实际到期为 2026-10-12 13:18:52～13:23:28 EDT；本轮复用，不提前申请。后续接替按计划申请四个 48 小时作业，不换账户或分区。

本机产物根：`/data/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/full-eval-20261010/`。NFS 产物根按计划为 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/artifacts/full-eval-20261010/`；执行代码放 NFS 仓库的 `artifacts/full-eval-20261010/runtime-code/`。种子 7 的 A、B 共用 `seed7/` 输出与队列，B 使用补集清单；种子 0、42 分开目录。阶段报表按身份白名单筛选 accepted 终态，不能把滚动进入 B 的结果混入 A。

模型服务端解释器、权重与适配器沿用 [上轮启动档案](../infer-timing-20261009/launch.md) 的「GL 六变体」及「QwenVL／MemER 首次加载失败与 r1 展开命令」。QwenVL／MemER 客户端使用已验证兼容解释器 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/venvs/client-env/bin/python`，不安装新依赖。MemER 显式 `--wall-s 7200 --noprog-s 7500`；其余模型沿用默认单局时限。

## 执行顺序与起跑证据

工具定向测试与独立审查 → 合并、核心短测 → 固定运行副本与身份筛选 → 提交启动记录 → GL 六变体最小 smoke → 正式 A/B/C/D 按策略滚动接续。阶段结束报告产物、成功率与实测耗时，不等待用户回复；失败只停止受影响部分，保留已完成结果。

只读身份导出命令（不调用 build/reset）：

```bash
UV_CACHE_DIR=/home/hongzefu/.cache/uv PYTHONPATH=/data/hongzefu/RoboMME-benchmark-OOD-eval/src:/data/hongzefu/RoboMME-benchmark-OOD-eval/third_party/robomme_benchmark/src uv run --no-sync python dev-scripts/gl/export_eval_identities.py --dataset ood --tasks BinFill,StopCube,PickXtimes,SwingXtimes,ButtonUnmask,VideoUnmask,VideoUnmaskSwap,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,PatternLock,RouteStick --out artifacts/full-eval-20261010/identities/all.jsonl
```

退出码 0，原文判定：`EVAL_IDENTITY_EXPORT=PASS datasets=ood episodes=700 hard_verify=0 ood=700 xhard0=0 tasks=14 cell_mismatch=0 bad=0 dup_keys=0 count_mismatch=0 out=artifacts/full-eval-20261010/identities/all.jsonl`。导出时包内规格报告历史 hard_fingerprint 与当前源码不符的既有警告；这是来源边界，不把身份 PASS 解释为历史生成源码逐位一致。

正式启动命令、起跑 commit、精确 tmux 名与首局证据在启动时补记；当前尚未启动仿真。后台存活、阶段接续、通知与代理唤醒分别验证，未建立机制不宣称无人值守已完成。

### 固定启动参数的还原

以下各变量均为本轮确定落点；解释器不安装依赖，只复用已有 uv 管理环境。运行载体在产物根 `run-model.sh`，仅承载下列参数展开，不进档案复制脚本。命令展开同时写入席位日志。

```bash
FE_N=/nfs/turbo/coe-chaijy-unreplicated/hongzefu
FE_REPO=$FE_N/RoboMME-benchmark-OOD-eval
FE_CODE=$FE_REPO/artifacts/full-eval-20261010/runtime-code
FE_ROOT=$FE_N/artifacts/full-eval-20261010
export ROBOMME_EVAL_ROOT=$FE_CODE SGEVAL_THIRD_PARTY=$FE_CODE/third_party
export PYTHONPATH=$FE_CODE/src:$FE_CODE/third_party/robomme_benchmark/src:$FE_CODE/third_party/mme-vla/packages/openpi-client/src
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$FE_N/hf-cache
export OPENPI_DATA_HOME=$FE_N/sg-eval/openpi-data UV_CACHE_DIR=/home/hongzefu/.cache/uv XDG_CACHE_HOME=$FE_ROOT/cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# 清除登录上下文遗留的 SLURM_*、CUDA_VISIBLE_DEVICES、SGEVAL_AUDIT、MESA_VK_DEVICE_SELECT、SAPIEN_RENDER_DEVICE。
# BENCH_PY 与模型专属参数按下表取值；A阶段的展开示例：
cd "$FE_CODE"
srun --jobid=63431430 --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared \
 bash dev-scripts/gl/run_eval_gl.sh --policies groundsg --groundsg-variant ground-sg-memer \
 --policy-seed 7 --identities "$FE_ROOT/identities/stage-A.jsonl" \
 --budget-ledger "$FE_ROOT/budget/stage-A.jsonl" --run fe-A-seed7 --out "$FE_ROOT/seed7" \
 --seat fe-A-memer-01 --gpus 0 --trajectory-cap 4680 --reset-cap 9360 \
 --shared-infra-cap 420 --expired-cap 420 --planned-first-tries 426 --infra-retries 1 --client-restarts 0 \
 --noprog-s 7500 --wall-s 7200 --ckpt "$FE_N/sg-eval/ckpt/mme/symbolic-grounded-subgoal/79999" \
 --port-base 19750 --work-dir "$FE_ROOT/seed7/work/memer-fe-A-memer-01" \
 --cfg mme_vla_py=$FE_N/robomme_benchmark-sgeval/third_party/mme-vla/.venv/bin/python \
 --cfg xla_mem_fraction=0.65 --memer-adapter "$FE_N/sg-eval/ckpt/memer/grounded_subgoal/checkpoint-1300" \
 --cfg openpi_data_home=$FE_N/sg-eval/openpi-data \
 --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 \
 --cfg jax_cache_root=$FE_ROOT/jax-cache --cfg server_dir="$FE_ROOT/seed7/servers/memer-fe-A-memer-01"
```

| 模型参数标签 | policies／variant | BENCH_PY（相对 NFS） | 模型专属参数 | 基础端口 |
|---|---|---|---|---:|
| smvla | smvla | 仓库 `.venv/bin/python` | `smvla_py=robomme_benchmark-sgeval2-b34/artifacts/v8-two/venvs/smvla-env/bin/python`；ckpt `SimpleMemVLA/checkpoints/simplememvla_robomme` | 19700 |
| perceptual-framesamp-modul | 同标签 | 仓库 `.venv/bin/python` | mme_vla_py 同上示例；ckpt `sgeval-20261006-02/ckpt/mme/perceptual-framesamp-modul/79999` | 19710 |
| oracle | groundsg／ground-sg-oracle | 仓库 `envs/client-env/.venv/bin/python` | mme_vla_py 与 base ckpt 同示例；无 adapter | 19720 |
| pp | pp | 仓库 `envs/client-env/.venv/bin/python` | `pp_py=robomme_benchmark-sgeval/third_party/PonderPounce/.venv/bin/python`；ckpt `sg-eval/ckpt/pp/ponderpounce-9b-robomme` | 19730 |
| qwen | groundsg／ground-sg-qwenvl | `sg-eval/venvs/client-env/bin/python` | mme_vla_py 与 base ckpt 同示例；`xla_mem_fraction=0.65`；`--qwenvl-groundsg-adapter sg-eval/ckpt/qwenvl-groundsg/checkpoint-1200` | 19740 |
| memer | groundsg／ground-sg-memer | `sg-eval/venvs/client-env/bin/python` | 同示例 | 19750 |

端口按席位槽号 0～3 加 `100 × 槽号`，避免同节点的两个作业端口冲突。所有模型均传共同的 tokenizer、openpi-data、jax-cache、独立 work-dir/server-dir。smoke 换 `identities/smoke.jsonl`、输出 `smoke/`、`--infra-retries 0`，仍使用 A 账本。正式各阶段只替换清单、种子、输出目录、阶段预算与 run/seat；每个阶段都用新的席位名，避免跨账本的悬空记录误结算。

## 本轮资源清单

既有占位 JobID：63431430、63431431、63431432、63431433。尚未提交接替作业、尚未创建运行 tmux；清理不得涉及清单外资源。

## 准备验证结果

工具首次固定提交审查失败，原代码代理修复后 `PRE_MERGE_REVIEW=PASS`。整合核心短测命令 `UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q` 在 186.20 秒结束，退出 1：1489 项通过、6 个条件跳过、101 项排除，两个失败都来自新增工具没有登记测试契约；资源守卫 `TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=6`。

补齐两个工具的 `C13-STAGED-EVAL-TOOLS` 契约后，登记改动独立固定提交审查通过，未增加豁免或放宽验收。主检出复验命令 `UV_CACHE_DIR=/home/hongzefu/.cache/uv uv run --no-sync python -m pytest tests/static/test_inventory.py tests/test_select_stage_identities.py tests/test_stage_report.py -q`，退出 0，36 项通过，4.38 秒；`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。其余核心短测已在前次整合运行通过；没有声称登记后重跑了全部套件。`git diff --check`、运行载体 `bash -n`、HTML 两个一级标题检查通过。

本机固定运行副本及五个子模块已核实状态为空；mme-vla 的内嵌 benchmark 目录为空。起跑前将固定到本次准备完成的提交，复制至新的 NFS 目录，并核对复制后 HEAD 与字节身份。

## 实际起跑记录（第一批 smoke）

评估代码固定在 `a02a4dad039b326530068977f22a04f4a4a287eb`（1.67）；后续本机留档提交不改变独立执行副本。rsync 退出 0，603156625 字节、约 86 秒。复制后本机对 NFS 路径核对根与五个子模块，六项状态均为空；`EXEC_COPY=PASS head=a02a4dad039b326530068977f22a04f4a4a287eb dirty=0`，原始证据在 [执行副本核验](records/execution-copy.json)。没有在 GL 进行 git 操作。

以下四个 detached tmux 已在 gl-login3 启动，均为本轮清单所有；日期为 2026-10-10，时间按 EDT。运行载体调用格式为 `bash <NFS产物根>/run-model.sh A <模型> <JobID> <席位> smoke <槽号>`，完整展开参数见上节与各日志首行。

| 模型 | JobID／节点 | tmux | 席位／槽号 | 启动时间 | 日志（相对 NFS 产物根） |
|---|---|---|---|---|---|
| MemER | 63431430／gl1525 | fe-smoke-memer-63431430 | fe-smoke-memer-01／0 | 00:41:49 | logs/fe-smoke-memer-63431430.log |
| QwenVL | 63431431／gl1512 | fe-smoke-qwen-63431431 | fe-smoke-qwen-01／1 | 00:42:02 | logs/fe-smoke-qwen-63431431.log |
| SimpleMemVLA | 63431432／gl1512 | fe-smoke-smvla-63431432 | fe-smoke-smvla-01／2 | 00:42:16 | logs/fe-smoke-smvla-63431432.log |
| PonderPounce | 63431433／gl1513 | fe-smoke-pp-63431433 | fe-smoke-pp-01／3 | 00:42:21 | logs/fe-smoke-pp-63431433.log |

四条命令各为 1 任务（VideoUnmask）× 1 档（xhard1）× 1 局，不重试。FrameSamp、Oracle 在后两条对应作业完成并核验媒体后启动；合计仍是六模型 × 同一任务档位 × 1 局 = 6 次冒烟轨迹，预占 12 次 reset。当前阶段为服务加载／起跑核验，不能据启动回执称 smoke 已通过。MemER 已见 `RUN_INPUTS=PASS`、`CLIENT_READY ... git=a02a4dad039b dirty=False`、`VARIANT_PAIRING=PASS`、`TOKENIZER_SHA=PASS`、`MME_VLA_PREFLIGHT=PASS`；首局与服务就绪另核对。

主会话保持活动，运行代理流式观察各自日志，主会话负责接续、账本和最终核验。当前未建立后续自动唤醒，不因 tmux 存活宣称无人值守已验证。
