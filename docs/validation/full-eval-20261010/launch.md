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

## 媒体修正与第二批 smoke

首批四模型均完成，原结果统一缺 ffprobe；未重跑模型。GL 系统模块的 AV1 计数探针失败后，补充固定 7.0.2 静态工具到 NFS 产物根 `media-tools/`，来源校验与完整指纹见 result.md 及 records/media-assets.json。新增命令载体 `run-model-r1.sh`（原载体保留），只增加 `ROBOMME_FFPROBE=<NFS产物根>/media-tools/ffmpeg-7.0.2-amd64-static/ffprobe` 和长度／首尾资产锁；清单必须匹配固定 canonical SHA256 `c4801fc13264422f9c512ef5be8cbf21c4ab32ca9c19f67aff1f38a2c365081d`。不改正在使用的 a02a4da 执行源码。

| 类型 | JobID | tmux | 起跑时间 EDT | 命令载体与参数 |
|---|---|---|---|---|
| FrameSamp smoke | 63431432 | fe-smoke-framesamp-63431432 | 00:52:43 | `run-model-r1.sh A perceptual-framesamp-modul 63431432 fe-smoke-framesamp-01 smoke 2` |
| Oracle smoke | 63431433 | fe-smoke-oracle-63431433 | 00:52:50 | `run-model-r1.sh A oracle 63431433 fe-smoke-oracle-01 smoke 3` |
| 原四局媒体补全 | 63431431 | fe-smoke-media-63431431 | 00:54:26 | 下面的固定 CPU 命令 |

所有外层命令均沿用 `set -o pipefail`、`tee logs/<tmux>.log`、`EXIT_CODE`；三会话已自然结束，不执行强杀。媒体恢复只读取已结束的四局，完整命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
 srun --jobid=63431431 --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared \
 /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python \
 /nfs/turbo/coe-chaijy-unreplicated/hongzefu/artifacts/full-eval-20261010/recover-smoke-media.py
```

恢复载体对四模型依次调用固定库的 `render_episode`，参数为 `official_root=CODE, ffmpeg=<media-tools>/ffmpeg-7.0.2-amd64-static/ffmpeg, source='raw', overwrite=False, max_memory_mib=6144, output=<raw>/recovered-video/<safe_filename>, sidecar=<raw>/recovered-video/render.json, episode_id=0, terminal=<原status>`；每局执行前后核对原 raw 全部文件、results.jsonl 与 accepted 标记的 SHA256／字节数相同，独立写 recovery.json，原 result.json 不回写。完整输出归档于 records/media-recovery-summary.json。

汇总验收在 GL 63431431 上用同一主解释器运行 `check-smoke.py`，完整解码六模型的原始两路与官方视频、核 identity/cap/accepted/result、核恢复证明，再执行固定 `dev-scripts/checks/trace_arrays_check.py <NFS产物根>/smoke/rollouts --json <NFS产物根>/smoke/trace-arrays-report.json`；退出 0，SMOKE 与 TRACE_ARRAYS 通过，账本 open=0。六个 smoke 身份与 A 相同，逐字节拷贝126文件至seed7输出，复用其accepted，不重复执行或新建账本。

## 独立进程恢复与连续接续

SimpleMemVLA 原 A 会话因连续环境构建失败精确停止，占位 63431432 保留。固定干净 r2 执行副本提交 `43e4dd0e214cae4a16efafe04c6ee3a4541f9c14`，评估 `src/` 和模型 gitlink 与原 a02 相同，仅私有席位／预算工具变化。每身份总上限 20，原阶段预算不变。02:33:23 EDT 起的 `fe-A-smvla-r2-check-63431432` 按原 A 清单仅续跑 BinFill／xhard1 两身份 `16400100`、`16400200` 的第三次，均正常失败接受后自然退出 0；03:00:18 起的 `fe-A-smvla-r2-next-63431432` 仅续跑原 A 的 BinFill／xhard2 身份 `18400100` 第三次。日志均为 NFS 根 `logs/<会话名>.log`，预算与两个原 a2 raw 副本保留。

连续调度器从全新 r3 冻结副本启动，旧 a02 与 r2 执行副本保持。参数由本轮产物 `controller-config.json` 明确给出：原四 JobID／slot、三个仍在运行的 A 会话和上述 r2 单身份会话，六模型优先顺序 `smvla,pp,oracle,perceptual-framesamp-modul,memer,qwen`，A/B 种子 7、C 种子 0、D 种子 42，原身份清单、原阶段账本与十倍硬上限。载体 `run-model-r3.sh` 仅改变执行副本路径，沿 r2 所有模型解释器、ckpt、adapter、端口分段、20 次上限与环境创建失败护栏。每个 SMVLA 身份独立加载／关闭，其他模型按阶段常驻。

主会话的启动命令为 `PYTHONUNBUFFERED=1 <NFS主uv解释器> <r3>/dev-scripts/gl/continue_full_eval.py --config <NFS产物根>/controller-config.json`，2026-10-10 03:18:11 EDT 在登录端独立 tmux `fe-full-eval-controller-20261010` 中按 pipefail／tee／EXIT_CODE 三件套启动，日志 `<NFS产物根>/logs/fe-full-eval-controller-20261010.log`；状态与事件在 `<NFS产物根>/controller/`。r3 冻结提交 `0c6f8a3a552876020611d4d654a41adb3accba65`，主会话核对根与五子模块均干净；NFS 上控制器 39 项夹具通过、0.62 秒、资源全零。配置 canonical 指纹 `12d997b55e7a92c2f9f04ef5c1e837b8b31f32fe2e076f5e225ae8f29bc0b721`。

实际接手保持 MemER／QwenVL／PP 三会话运行，r2 单身份正常退出后自动启动 `fe-controller-A-smvla-2-1`，只领取下一个未完成 A 身份 `BinFill_xhard2_18400200`；`RUN_INPUTS=PASS`、`CLIENT_READY ... git=0c6f8a3a5528 dirty=False`、`RUN_PLAN ... max_attempts=20` 均有回执。之后已跨多个任务自动接续。零尝试空转会话仅跳过已接受身份，没有重复模拟；不能把会话数当局数。程序只接手清单会话，新增 worker 命名 `fe-controller-<阶段>-<模型>-<slot>-<序号>`；报告会话 `fe-controller-report-<阶段>`，完整实际清单由 `controller/events.jsonl` 逐项记录。到期接替只在原作业明确 TIMEOUT 后提交 `chaijy2/spgpu/gpu:1/cpu4/64G/48h`，四席上限不变；提交不明停止核查、不重复提交。

## 正式阶段 A 起跑

2026-10-10 EDT，四席使用 `run-model-r1.sh A <模型> <job> <seat> formal <slot>`，代码仍固定 a02a4da；70 行 A 身份清单中每模型已完成1个 smoke 身份，`RUN_PLAN total=70 claimable=69 max_attempts=2`，不会重复执行已接受身份。其余 FrameSamp／Oracle 在短模型 A 清空后使用同一既有作业依次运行，空闲席再并入长模型队列。

| 模型 | JobID | tmux／日志基名 | seat／slot | 启动时间 |
|---|---|---|---|---|
| MemER | 63431430 | fe-A-memer-63431430 | fe-A-memer-01／0 | 01:06:05 |
| QwenVL | 63431431 | fe-A-qwen-63431431 | fe-A-qwen-01／1 | 01:06:15 |
| SimpleMemVLA | 63431432 | fe-A-smvla-63431432 | fe-A-smvla-01／2 | 01:06:17 |
| PonderPounce | 63431433 | fe-A-pp-63431433 | fe-A-pp-01／3 | 01:06:28 |

日志均在 NFS 产物根 `logs/<tmux>.log`；四项 `MEDIA_ASSETS=PASS level=cheap assets=2 mismatches=0`、`RUN_INPUTS=PASS`、`CLIENT_READY ... git=a02a4dad039b dirty=False` 已核对。MemER／QwenVL／PP 已见 `SERVER_READY ready_s=26.5`，SimpleMemVLA 正在加载；首个新 accepted 与阶段终态继续流式观察，不据启动状态宣称阶段通过。阶段A为六模型 ×（七任务两档3/2局 + 两任务四档2/1/1/1局 + 三任务三档2/2/1局 + 两任务五档每档1局）=420个身份，已复用6，余414。

## 旧两次上限控制器修订的实际接手

2026-10-10 06:42:12 EDT，新控制器从 clean 提交 `dc63643fc0945e59872e4a2c50e67de42921e9b9` 的独立副本 `runtime-controller-r4` 接手。该副本只运行标准库控制器与同目录预算模块，不初始化模型子模块；工作负载、模型源码、报告入口和 `run-model-r3.sh` 仍使用原冻结 r3 副本，三个健康老 worker 不重启。配置指纹继续为 `12d997b55e7a92c2f9f04ef5c1e837b8b31f32fe2e076f5e225ae8f29bc0b721`，原状态的 A 阶段、四席作业与 serial 32 保留。

精确停止本轮旧监督器 `fe-full-eval-controller-20261010` 前后，会话差集恰好为它一个；`fe-A-memer-63431430`、`fe-A-qwen-63431431`、`fe-A-pp-63431433`、`fe-controller-A-smvla-2-32` 均保留。旧监督器是主动停止，没有自然 `EXIT_CODE`，不当作全量成功。新会话 `fe-full-eval-controller-r4-20261010` 已核 `tmux has-session` 退出 0，日志在 NFS `logs/fe-full-eval-controller-r4-20261010.log`；[接手回执](records/controller-r4-start.json) 给出 `CONTROLLER_TAKEOVER=PASS workers_preserved=4 config_unchanged=1`。

实际启动使用产物根临时载体 `controller-r4-start.sh`，先拒绝已存在日志，再执行如下完整主体；载体不进 git，不另存留档脚本，源码由上述提交还原。

```bash
set -o pipefail
controller_log=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/artifacts/full-eval-20261010/logs/fe-full-eval-controller-r4-20261010.log
PYTHONUNBUFFERED=1 /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/full-eval-20261010/runtime-controller-r4/dev-scripts/gl/continue_full_eval.py \
  --config /nfs/turbo/coe-chaijy-unreplicated/hongzefu/artifacts/full-eval-20261010/controller-config.json \
  2>&1 | tee "$controller_log"
controller_exit=$?
printf 'EXIT_CODE=%s\n' "$controller_exit" | tee -a "$controller_log"
exit "$controller_exit"
```

主会话流式读取新日志与原 `controller/events.jsonl`。软件测试与实际接手均已验；PP 旧席位尚未结束，真实 `legacy_retry_resume` 事件及其第三次尝试仍待观察，完整 A／B／C／D 结果未验收。
