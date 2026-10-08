# sg-eval-gl-20261004-01 起跑档案（launch）

四模型（GroundSG-Oracle、GroundSG-QwenVL、PonderPounce、Astra）接入与三档闸门评估。权威计划：仓库根
[`1003-oracle-subgoal-groundsg-eval-plan.md`](../../../1003-oracle-subgoal-groundsg-eval-plan.md)。本文件按正本第 12 条两段式的第一段写
（版本与代码状态、启动与配置还原），跑完另写 `result.md` 与 `records/`。

## ① 目的与授权

- 用户开工令（2026-10-04）：「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/1003-oracle-subgoal-groundsg-eval-plan.md 开始实现 有问题问用户 但不要阻塞」。
- 实施中追加：「同意加子模块」「如果安全分类器把你必要的权限都拦截了,你需要立刻问用户让用户授权」「push同意」「检查S2合并是不是卡死了」
  「继续说中文你现在还在说英文不对」。
- 计划与预算此前已获批（计划文首用户原话 1～34 条）。Astra 局数逐项审批：本机预检 2 局（失败可加到 6 局）、GL 第二档 32 局、V9 连通 1 局；
  费用合计上限 30 美元。

## ② 版本与代码状态

| 项 | 值 |
|---|---|
| 工作副本 | `/data/hongzefu/robomme_benchmark_MotionJEPANewTask`，分支 `newtaskRelease-taskV9` |
| 开工 BASE | `3ec9b7fd5cd0821fe6baf5a741cdf87623505b6e`（12.431） |
| 接线合入后、GL 执行副本冻结提交 | `63b2896695fd91868642d0fd0f7c0d6afa330cd3`（12.450） |
| GL 执行副本 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-sgeval`（`git clone --no-hardlinks` 自本机，检出上行提交） |
| 本机第一档旧码快照 | `artifacts/sg-evaluation/sg-eval-gl-20261004-01/old-wt`（`git worktree add --detach … 3ec9b7fd`，跑完删除） |
| `src/robomme` | 相对开工 BASE 零 diff；录像器 `RecordWrapper.py` 零 diff；`scripts/*.py` 仍为 4 个入口 |

接线七块（计划执行模式：worktree 隔离的 opus 写入型子代理、`sub/` 前缀提交、`--no-ff` 合并、合并前只读审查与合并后门禁）：

| 块 | 子代理提交 | 合并提交 | PRE_MERGE_REVIEW |
|---|---|---|---|
| S1 test-hard0 与删查表 | `1d3cc143` | 12.434 `c7d48e1e` | PASS files=15 |
| S2 客户端／清单／报告 | `46c2ccf4` | 12.437 `147c4d62` | PASS files=14 |
| S3 GroundSG | `77fec450` | 12.445 `52c1a0ca` | PASS files=9 |
| S4 PonderPounce | `71443592`、`18f17419` | 12.439 `947a2007` | PASS files=6 |
| S5 Astra | `cc03e064` | 12.441 `d1138b6d` | PASS files=9 |
| S6 启动脚本与转码 | `1fa445dd` | 12.447 `4e2aa0fe` | PASS files=10 |
| S7 轨迹／对比／报告／探针 | `d15c88d9` | 12.443 `adc5ad4e` | PASS files=14 |

合并后主会话修补：12.442、12.446（astra／groundsg 测试缺省取当前检出 third_party）、12.449（`run_eval_gl.sh`／`pair_seat.sh` 增 `--gpu`）、
12.450（`video_check` 不计空的 `.incoming`）。日常门禁（12.448）`2862 passed`，slow（eval＋evalx）`67 passed`。

## ③ 第三方锁定与资产

| 项 | 锁定 |
|---|---|
| `third_party/mme-vla` | `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` |
| `third_party/PonderPounce` | `723df35762bb641e1d520e4fa9359b98644adc21`（https://github.com/worv-ai/ponderpounce） |
| `third_party/Astra-on-RoboMME` | `4c3fd6a8667e7a219e547fe1a533a4b9726fc6db`；嵌套 `third_party/robomme_benchmark` `856bc3a189d4172f3f47dbee4424d585f8d78db3`（R10 豁免，只供 Astra 原侧） |
| vla-eval | PyPI 0.7.0（PonderPounce `uv.lock`） |
| PonderPounce 权重 | `worv-ai/ponderpounce-9b-robomme@5691e05614e8e6581d6358959e60561acfa73534` → `$N/sg-eval/ckpt/pp/ponderpounce-9b-robomme`（`PP_FILES=PASS`；逐文件 sha256 `inputs/pp-weights.sha256`） |
| Qwen3.5-9B 分词器与配置（无权重） | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` → `$N/hf-cache/hub`（`refs/main` 已补写） |
| PaliGemma 分词器与配置（无权重） | `google/paligemma-3b-pt-224@35e4f46485b4d07967e7e9935bc3786aad50687c` → `$N/hf-cache/hub`（gated，可读） |
| Astra 监视器 | `bingaochen/Astra-on-RoboMME-Monitor@18ac28316a8b22c758945546c95f3910fc5dd27b`（Astra `weights.json` 所钉；`ASTRA_MONITOR_SHA256SUMS=PASS`） |
| 本机拷贝的资产 | GroundSG `symbolic-grounded-subgoal/79999` 与 `history_config.txt`、QwenVL 适配器 `checkpoint-1200`、Qwen3-VL-4B@`ebb281ec70b05090aa6165b016eac8ec08e71b17`、openpi 分词器（sha256 `8986bb4f…`）→ NFS：`ASSETS=PASS files=89 mismatch=0 scope=local-copies` |

`$N` = `/nfs/turbo/coe-chaijy-unreplicated/hongzefu`。`ASSETS=PASS` 只保证字节相同，不保证数值逐位可对拍。

## ④ 环境

| 环境 | 位置 | 判定 |
|---|---|---|
| 主环境 | 本机 `.venv`（不动）；GL 副本 `.venv`（`uv sync --frozen --group eval-client`） | — |
| 客户端扩展（本机） | `artifacts/sg-evaluation/venvs/client-env`（`scripts/eval-official/client-env/uv.lock`） | torch 2.9.1+cu128、sapien 3.0.2、`FLASH_ATTN_KERNEL=PASS maxdiff=0` |
| 客户端扩展（GL） | `$N/sg-eval/venvs/client-env`（不装 robomme、openpi-client，运行时由 GL 副本 PYTHONPATH 提供） | `GL_CLIENT_ENV=PASS torch=2.9.1+cu128 flash_attn=2.8.3 sapien=3.0.2` |
| PonderPounce（本机，做法 A） | `artifacts/sg-evaluation/pp-local-env`（官方 lock 导出、只换 `torch 2.14.0+cu126`） | `PP_LOCAL_ENV=PASS torch=2.14.0+cu126 cuda=12.6 transformers=5.13.1` |
| PonderPounce（GL） | GL 副本 `third_party/PonderPounce/.venv`（`uv sync --frozen`，CUDA 13 构建） | GL smoke 判定 |
| MME 服务 | `third_party/mme-vla/.venv`（本机已有；GL 副本另建） | — |

## ⑤ 本机预检（计划第二部分四）

`PREFLIGHT_SUMMARY=PASS routes=13 failed=0 stopped_routes=none step_cap_ok=1 video_ok=1 site=sled-vail gpu=RTX6000Ada`

| 路线 | rss 峰值 GB | 显存峰值 MB | 单局 s | 终态 |
|---|---|---|---|---|
| GroundSG-Oracle 原侧 xhard0 | 13.2 | 35296 | 48.7 | success |
| GroundSG-Oracle 新侧 xhard0 | — | 35297 | 46.2 | success |
| GroundSG-Oracle 新侧 V9 | — | 35301 | 46.4 | fail |
| GroundSG-QwenVL 原侧 xhard0 | 15.3 | （无效） | 355.9 | fail |
| GroundSG-QwenVL 新侧 xhard0 | 15.0 | 44180 | 194.5 | fail |
| GroundSG-QwenVL 新侧 V9 | 16.3 | （无效） | 214.5 | success |
| PonderPounce 原侧 xhard0 | 28.3 | 27366 | 62.1 | success（exec 262） |
| PonderPounce 新侧 xhard0 | 29.8 | 27366 | 63.4 | success（exec 262） |
| PonderPounce 新侧 V9 | 26.9 | 27366 | 83.6 | success |
| Astra 原侧 xhard0 | — | — | 49.1 | success（291 步，规划 3 次） |
| Astra 新侧 xhard0 | — | — | 49.4 | success（276 步，规划 3 次） |

- 步数到顶（`cap_probe.py`，VideoUnmask，不加载模型）：
  `STEP_CAP=PASS dataset=test-hard0 max_steps=1300 builder_cap=1302 loop=mme exec_steps=1301 terminal_reason=loop_count source=run`；
  `… loop=range exec_steps=1300 terminal_reason=loop_exit source=prefix`；
  `STEP_CAP=PASS dataset=test-hard max_steps=1600 builder_cap=1602 loop=strict exec_steps=1600 terminal_reason=strict_cap source=run`。
- 步数接线：xhard0 路线结果行 `max_steps=effective_max_steps=1300`、strict cap 关；V9 路线 1600、`strict_cap=True`。
- 视频：9 条非 Astra 路线 `VIDEO_SAVED=PASS … frame_mismatch=0 key_mismatch=0`（`--frames-rule demo+exec --frame-offset 1`，即帧数 = demo_frames + 1 + exec_steps）；
  每局 mp4 约 0.4～0.9 MB；Astra 两侧 `rollout.mp4` 约 0.52 MB。
- Astra：`ASTRA_NESTED_ROBOMME=PASS files=102 ours=102 set_diff=0 diff=0`；费用 6 次请求 0.5149 美元（单次输入约 8 千 token，远低于 272K 加价线）；
  单价配置 `inputs/astra-prices.json`（官方：input 10、cached 1、cache write 12.5、output 50 美元／百万 token；非缓存输入保守按 12.5 计）；
  账本 `astra-local/astra-cost-ledger.json`，GL 运行接续累计。
- 生成 smoke（新码，`--workers 4 --dev-smoke`）：V9 PickXtimes xhard1 seed 16100000、xhard0 PickXtimes seed 510300 各 1 局，`rc=0`。

预检中的意外与重跑（计入「修复后重跑」额度，每路线 ≤2）：卡号被启动器覆盖致两会话同卡、PonderPounce 三条 OOM（12.449 修复后重跑各 1 局）；
主会话在 `run_eval_gl.sh` 运行中改文件致 QwenVL 新侧 xhard0 收尾 syntax error 与孤儿同步循环（按精确 PID 停掉后重跑 1 局）；
Oracle 原侧为显存复测重跑 1 局；Astra 原侧第一次因清单路径写错在任何请求之前退出（未消耗局数）。

## ⑥ 执行方决定（计划写明可自定的事项）

1. GL 上 QwenVL 变体 `SEAT_XLA_MEM_FRACTION=0.65`（A40 约 46 GB：0.75 时预估 43.4 GB 余量过小）；Oracle、PonderPounce 保持 0.75；GL smoke 复核。
2. 不需两卡 job：三模型显存峰值都在单张 A40 内；单卡席位 4 CPU／64 GB 够用（rss 峰值 ×1.25 最大 37.3 GB）。
3. PonderPounce 加速内核不装（两侧、全部席位一致）：不装时单局 62～84 s 已足够；预检未另跑「装内核」对照局。
4. 本机第一档与第二档同时开跑：0 号卡先第一档、后第二档 Oracle→PonderPounce；1 号卡第二档 QwenVL（单局最长，最先开始）。

## ⑦ 启动与配置还原（本机，已起跑）

一次性脚本（跑完逐字归档到 `records/*.sh.txt`）：
- 预检：`artifacts/sg-evaluation/sg-eval-gl-20261004-01/preflight/preflight_card.sh <物理卡> <路线…>`
- 本机第一档：`…/local-g1/local_gate1.sh <物理卡>`（旧码 `old-wt` 与新码各生成 V9 129 局 + xhard0 48 局，`noise_gate.py gen-compare` 逐局互比）
- 本机第二档：`…/local-g2/local_gate2.sh <物理卡> <oracle|qwenvl|pp…>`（`pair_seat.sh`，分片为 hard0 192 局全量，`--reset-budget 202 --infra-retry-budget 10 --orig-infra-retry-budget 10`）
- Astra 本机冒烟：`…/astra-local/astra_local.sh 1`（新侧）、`…/astra-local/astra_local_orig.sh 1`（原侧）

本轮 tmux 会话清单（只按此清单精确清理）：
`sgev-assets`、`sgev-download`、`sgev-flashattn`、`sgev-ppenv`、`sgev-ppenv2`、`sgev-glenv`、`sgev-glenv2`、`sgev-glenv3`、`sgev-local-cap-0`、`sgev-local-cap-1`、
`sgev-local-pre-0`、`sgev-local-pre-1`、`sgev-local-pre-1b`、`sgev-local-pre-1c`、`sgev-local-astra`、`sgev-local-astra-cost`、`sgev-local-astra-orig`、
`sgev-local-g1g2-0`、`sgev-local-g2-1`、`sgev-glcopy`（多数已自行结束）。

## ⑧ GL 占位 job

清单 `$N/gl-hold-logs/hold-jobs-sgeval-20261004.txt`：单卡 `sgev-hold-00～07` = 63188711、63188712、63188713、63188714、63188715、63188716、63188719、63188720；
两卡 `sgev-astra-hold` = 63188721。截至本文件写入全部 `PENDING (AssocGrpGRES)`。排到卡即唤醒：主会话后台轮询（宿主任务，110 分钟续挂一次）。

## ⑨ 预算（计划第二部分六，已获批）

合计轨迹 5673、reset 11925；Astra 金额上限 30 美元。已用：预检非 Astra 9 局 + 重跑 5 局、步数到顶 2 局、Astra 2 局（0.51 美元）、生成 smoke 2 局。

## ⑩ 运行记录与收尾状态（2026-10-05 22:17 用户令中止）

用户原话：「现在立刻收尾。停止本机器和gl上所有任务 但是job不要关。」「Turbo上的资源先不动。只收尾不搬运。」结果见同目录 `result.md`。

### GL 席位（登录节点 gl-login3 的 tmux；`$R=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261004`）

| tmux 会话 | 占位 job | 内容 | 结局 |
|---|---|---|---|
| `sgev-seat-00` | 63188711 | Oracle：第一档两遍 → 第二档 → 第三档 | 第一档、第二档完成；第三档首局 `IDENTITY_MISMATCH`（分片局号错位 12）`CHAIN_STOP` |
| `sgev-seat-03` | 63188711 | QwenVL 席位 03（首次） | GLIBC_2.30／2.32 ImportError，未出结果（目录改名 `gate2-qwenvl.glibc-fail`） |
| `sgev-seat-00b` | 63188711 | Oracle 第三档（重出分片） | 800/800 完成，`SEAT_CHAIN_DONE` 13:33 |
| `sgev-seat-01` | 63188712 | PonderPounce 第二档分片 0 → 第三档分片 0 | 完成；第三档 rc=6（1 局上下文超限 error） |
| `sgev-after-00b` | 63188711 | 等 00b 结束 → 第二档 Oracle 12 局 SwingXtimes 补跑（`oracle_g2_swing.sh`）→ 席位 03 QwenVL 第二档分片 00 → 第三档分片 00 | 补跑 12/12；席位 03 第二档 37+2 error，第三档 9/160 时中止 |
| `sgev-seat-04b` | 63188712 | QwenVL 席位 04（v2）第二档分片 01 → 第三档分片 01 | 22:09 起跑，约 7 分钟后中止 |
| `sgev-seat-02`、`sgev-seat-04`、`sgev-seat-05`、`sgev-seat-06`、`sgev-seat-07` | 63188713～16、63188719 | 预挂的 v2 启动器（等 job RUNNING） | job 未排到卡；`sgev-seat-04` 改挂到 63188712 前关闭，其余在收尾时关闭 |

22:17 收尾：上表在用会话全部 `tmux kill-session -t '=<名>'`；工作步骤（`63188711.3` 等）随 srun 退出，`squeue -s` 只剩各 job 的 batch／extern；**9 个占位 job 全部保留**（63188711、63188712 RUNNING 空占；63188713～16、63188719、63188720、63188721 PENDING）。

### 本机会话

| tmux 会话 | 内容 | 结局 |
|---|---|---|
| `sgev-lg3-card0` | 本机复刻第三档：Oracle 800 → PonderPounce 分片 00 → 01（`local_gate3.sh 0`） | Oracle 800 完成；PonderPounce 184/400 时中止 |
| `sgev-lg3-card1` | 本机复刻第三档：QwenVL 分片 00～04（`local_gate3.sh 1`） | 137/160 时中止 |

本机第二档 v2 续跑与 SwingXtimes 补跑会话此前已自行结束。收尾后两张卡显存 7 MB／6 MB，无残留评估进程（PID 2961357 为 10-03 遗留，未动）。

### 产物位置（未搬运）

- GL：`$R/{gate1,gate2-*,gate3-*,media,logs,inputs,scripts}`，约 11.9 GB，原地保留（续跑依赖这些 stage 目录）。
- 本机：`artifacts/sg-evaluation/sg-eval-gl-20261004-01/{preflight,local-g1,local-g2,local-g3,gl-g2,astra-local}`，182 GB（其中 `local-g1` 171 GB 为第一档中间 h5）。
- 一次性脚本逐字归档：`records/scripts/`。

## ⑪ 本机第三档 Oracle 官方版式离线重绘与视频站点（2026-10-05）

用户原话：「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/1005-eval-video-official-overlay-plan.md实现第一阶段的转码转完之后host在一个网站上。host完网站做Playeright测试。」本轮只复用已有评估文件，零 reset、零轨迹生成、零 GPU；本机复刻成绩沿用已有结果，重绘不构成重新评估。

运行代码锚点 `6a23af541b4f136e754eeb06e8fd1548700c4b23`（12.459），运行目录 `artifacts/worktrees/official-overlay-run/`，detached HEAD，起跑前 `git status --porcelain` 为空。主检出的 `third_party/SimpleMemVLA` 有既有在途修改，完全排除；运行副本不读取该子模块。官方类只读来自主检出 `third_party/mme-vla/examples/robomme/utils.py`，gitlink 锁定 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，逐局清单记录实际官方代码、工具和 FFmpeg 指纹。uv 使用主仓库既有 `.venv`，只运行 `--no-sync`，不安装或变更项目依赖。

媒体全集是下表的乘式之和，即 `16 任务 × 每任务合计 50 局 = 800 局`；第三档指评估阶段，不是单一难度：

| 任务组 | 任务与难度乘式 | 局数 |
|---|---|---|
| BinFill、ButtonUnmaskSwap、PickHighlight、VideoPlaceButton、VideoPlaceOrder、VideoRepick、VideoUnmaskSwap | 7 任务 × 2 档（xhard1/2）× 25 局 | 350 |
| ButtonUnmask、VideoUnmask | 2 任务 ×（2 档 xhard1/2 × 13 局 + 2 档 xhard3/4 × 12 局） | 100 |
| PatternLock、PickXtimes、RouteStick | 3 任务 ×（2 档 xhard1/2 × 17 局 + 1 档 xhard3 × 16 局） | 150 |
| StopCube、SwingXtimes | 2 任务 × 5 档（xhard1～5）× 10 局 | 100 |
| InsertPeg、MoveCube | 2 任务 × 1 档（xhard4）× 50 局 | 100 |

真实冒烟先执行 `1 任务（VideoPlaceButton）× 1 档（xhard2）× 1 局`，该局属于上述全集，正式全量会核验并复用派生结果，避免覆盖。输入 `local-g3/media/mmesg-ground-sg-oracle/test-hard/new/` 下 `episode.mp4`、`trace.jsonl`、`arrays.npz`；输出仍在对应每局 `official/`，来源记录 `official/render.json`。站点和运行日志集中在 `artifacts/sg-evaluation/sg-eval-gl-20261004-01/official-overlay/`。

从运行副本执行，环境覆盖为 `UV_CACHE_DIR=/home/hongzefu/.cache/uv`、`UV_PROJECT_ENVIRONMENT=/data/hongzefu/robomme_benchmark_MotionJEPANewTask/.venv`、`PYTHONUNBUFFERED=1`、`CUDA_VISIBLE_DEVICES=''`；全量额外钉 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。完整重绘命令：

```bash
uv run --no-sync python scripts/eval-official/render_official_video.py \
  --root /data/hongzefu/robomme_benchmark_MotionJEPANewTask/artifacts/sg-evaluation/sg-eval-gl-20261004-01/local-g3/media/mmesg-ground-sg-oracle/test-hard/new \
  --official-root /data/hongzefu/robomme_benchmark_MotionJEPANewTask \
  --label mmesg-ground-sg-oracle --jobs 16 --threads 2 --max-memory-mib 6144
```

长任务放独立 tmux `ovl-rerender-g3`，`set -o pipefail`、`tee` 写 `official-overlay/logs/rerender.log`，完成写 `EXIT_CODE=`。主会话保持活动，使用行缓冲日志监听接续；不是后台自动唤醒。站点使用本轮独立会话 `ovl-site-g3-8083`，端口起跑前探测，用户请求托管，因此收尾保留该会话及固定代码运行副本。既有 `site-v9-8082` 等会话不处理。

目录构建、服务与浏览器完整入口：

```bash
R=/data/hongzefu/robomme_benchmark_MotionJEPANewTask
O="$R/artifacts/sg-evaluation/sg-eval-gl-20261004-01/official-overlay"
export UV_CACHE_DIR=/home/hongzefu/.cache/uv CUDA_VISIBLE_DEVICES=''
uv run --no-project python scripts/injection-dev/site/official_overlay_site.py \
  --root "$R/artifacts/sg-evaluation/sg-eval-gl-20261004-01/local-g3/media/mmesg-ground-sg-oracle/test-hard/new" \
  --site-dir "$O/site" --expect-episodes 800
uv run --no-project python scripts/injection-dev/site/official_overlay_site.py \
  --serve --site-dir "$O/site" --media-root "$R/artifacts" --host 0.0.0.0 --port 8083
uv run --no-project --with playwright python scripts/injection-dev/site/official_overlay_browser_check.py \
  --base http://sled-vail.eecs.umich.edu:8083 --shots "$O/checks"
```

全量逐局检查来源、原动作和输出帧数/帧率/尺寸；网站构建重核 trace、NPZ 与重绘视频全量 sha256。Playwright 核目录全部身份，实际播放覆盖视频演示与非演示代表局及原视频，不能把它描述成 800 局均经浏览器逐帧播放。截图、报告和判定行在完成后写入 `result.md`；本节是起跑时记录，不预写运行通过。

### 网站追加：逐 Task 成功率、英文 Task 栏目

用户在首版上线后纠正为只显示逐 Task 成功率，V9 与上一版 xhard0 各给百分比和成功／总；Task 内不拆，视频按 Task 分栏目，所有任务名用英文。上述首次运行命令对应 `6a23af54`；追加版构建与浏览器都增加下列两个参数：

```bash
--v9-results "$R/artifacts/sg-evaluation/sg-eval-gl-20261004-01/local-g3/stage-oracle-00/s90/mmesg-ground-sg-oracle/results.jsonl" \
--xhard0-results "$R/artifacts/sg-evaluation/sg-eval-gl-20261004-01/local-g2/gate2-oracle/new-merged.results.jsonl"
```

两列均用本机 Oracle 新接口接受的终态结果；xhard0 为 `16 任务 × 1 档 xhard0 × 12 局 = 192 局`，不重跑评估。公开仅 16 行计数，完整来源及 sha256 写 `site-tasks/success-private-proof.json`。新版站点目录改为 `official-overlay/site-tasks/`，截图与报告改为 `official-overlay/checks-tasks/`；旧目录与首版报告保留。只重启本轮自己的 `ovl-site-g3-8083`，删前删后核对会话差集恰好一个；新服务从主检出读取新增版页面，源码在浏览器通过后原样提交，原重绘锚点不变。最终保留该站点会话，临时工作副本清理由主会话负责。

用户最后将可见列名明确为 `Xhard`、`原版hard`，仅修改 HTML 和浏览器断言；最终检查改用 `official-overlay/checks-labels/`，原 `checks-tasks/` 保留。此时服务器逐请求读取主检出的 HTML，表头更新无需再启动进程；实际两份结果文件和成功局数不变。

最终网站源码锚点 `22a90ce9c33d01b729013d6f7d21457b87d5ceef`；重绘仍固定 `6a23af541b4f136e754eeb06e8fd1548700c4b23`。两者用完整哈希还原，避免同时存在的计划提交与实施提交编号重合时选错来源。
