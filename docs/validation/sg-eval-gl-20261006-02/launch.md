# sg-eval-gl-20261006-02 起跑档案（launch）

第二阶段：六条新接口路线直出官方版式视频 + 统一逐步记录，GL A40 上跑第一、二、三档。权威计划：
[`docs/plans/1005-eval-video-phase2-all-models-rerun-plan.md`](../../plans/1005-eval-video-phase2-all-models-rerun-plan.md)。
本文件按正本第 12 条两段式的第一段写（版本与代码状态、启动与配置还原），随批次推进逐节追加；跑完另写 `result.md` 与 `records/`。

## ① 目的与授权

- 用户开工令（2026-10-06 02:2x EDT）：「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/docs/plans/1005-eval-video-phase2-all-models-rerun-plan.md 开始执行 有问题半小时内问用户 尽可能不要阻塞 一口气做完再让用户判断」。
- 此前已批（计划文首与六节）：开工后无人值守跑到计划结束、不设截止、≤10 卡、占位 job 自管；预算轨迹 6366（硬）、reset 计量 14143、账本软上限约 141430（「Reset额度可以。提高到现在的十倍左右不用作为硬边界。」）；Astra 本机 smoke ≤2 局、5 美元硬上限（「唯一需要注意的就是astra还是只跑两次。」）；集群操作预授权。

## ② 版本与代码状态

| 项 | 值 |
|---|---|
| 工作副本 | `/data/hongzefu/robomme_benchmark_MotionJEPANewTask`，分支 `newtaskRelease-taskV9` |
| 开工时 HEAD | `3f5237797d644211cf17ed9245c8b96f42bcd7a0`（12.474） |
| BASE（S0 提交后） | `b869a3df9e7406b8f5458697f22656165b9c50b3`（12.475） |
| 回放工具 | `ba1bbb06`（12.476，`client_replay_eq.py`） |
| 第一档执行副本 | `$N/robomme_benchmark-noise` 检出 `b869a3df`（clean；`uv.lock` 与此前 `f8f76fba` 零 diff，借用其 `.venv`） |
| 原侧补跑执行副本 | `$N/robomme_benchmark-sgeval` HEAD `a299fcdcd8d8664a07bff758011d7ea7ca899f55`（上一轮冻结，不同步、不改；链脚本起跑断言 HEAD 与 clean） |
| SimpleMemVLA 原版工作树 | `$N/SimpleMemVLA-official-xhard0`（`git -C $N/SimpleMemVLA worktree add --detach … 4e0c04f`），clean |
| MME 原版克隆 | `$N/robomme_policy_learning-official-xhard0`，分支 `official-xhard0-0929` = `927c56d9879762c1afa0217d79a591093c9ffbe9`；子模块 `third_party/robomme_benchmark` = `856bc3a189d4172f3f47dbee4424d585f8d78db3`（本地克隆源 `/data/hongzefu/robomme_benchmark_MotionJEPA`，仅在该克隆里覆盖子模块 URL），其 `src/robomme` 与本仓库 `diff -rq` 为空 |

`$N` = `/nfs/turbo/coe-chaijy-unreplicated/hongzefu`；本轮 NFS stage `R2=$N/sgeval-20261006-02`（`scripts/`、`logs/`、`inputs/`、各档目录）。

## ③ 占位 job 与席位

| JobID | 名称 | 节点 | 本轮用途 |
|---|---|---|---|
| 63188711 | sgev-hold-00 | gl1525 | 批次 1 第一档（V9 → xhard0 串行） |
| 63188712 | sgev-hold-01 | gl1512 | 批次 1 PonderPounce 原侧 shard-01（96 局，席 11） |
| 63188713 | sgev-hold-02 | gl1513 | 批次 1 QwenVL 原侧 shard-01（38 局，席 21） |
| 63188714 | sgev-hold-03 | gl1518 | 批次 1 QwenVL 原侧 shard-02（38 局，席 22） |
| 63188715 | sgev-hold-04 | gl1503 | 批次 1 QwenVL 原侧 shard-03（39 局，席 23） |
| 63188716 | sgev-hold-05 | gl1527 | 批次 1 QwenVL 原侧 shard-04（38 局，席 24） |
| 63188719 | sgev-hold-06 | gl1525 | 空闲（留给批次 2） |
| 63188720、63325534、63325535 | sgev-hold-07～09 | PENDING（AssocGrpGRES） | 排到后用于后续批次 |

清单文件 `$N/gl-hold-logs/hold-jobs-sgeval-20261004.txt`；收尾按清单逐个 `scancel`。

## ④ 批次 1（2026-10-06 02:36 EDT 起跑）

登录节点 `gl-login3` tmux（本轮会话清单，清理只按此清单）：`p2-g1`、`p2-pp-s11`、`p2-qw-s21`、`p2-qw-s22`、`p2-qw-s23`、`p2-qw-s24`。

```bash
R=$N/sgeval-20261006-02; I=$N/sgeval-20261004/inputs
tmux new-session -d -s p2-g1 "bash $R/scripts/launch_seat.sh 63188711 gate1 $R/scripts/chain_gate1.sh"
tmux new-session -d -s p2-pp-s11 "bash $R/scripts/launch_seat.sh 63188712 orig-pp-s11 $R/scripts/chain_orig.sh --seat 11 --model pp --shard $I/pp/gate2/shard-01.json"
# 席 21～24 ↔ 63188713～16 ↔ QwenVL shard-01～04
tmux new-session -d -s p2-qw-s2$i "bash $R/scripts/launch_seat.sh <job> orig-qwenvl-s2$i $R/scripts/chain_orig.sh --seat 2$i --model qwenvl --shard $I/qwenvl/gate2/shard-0$i.json"
```

- 第一档：`$R/scripts/gate1_pass.sh`（照抄上一轮 `sgeval-20261004/scripts/gate1_pass.sh`，执行副本换成 `robomme_benchmark-noise@b869a3df`）——`hard_parity.py generate --expect-ref scripts/configs/noise-ref-20261003.json`，V9 `--side H2 --tier v9`（43 格 × 3 局 = 129）、xhard0 `--side H --tier xhard0`（16 任务 × 3 局 = 48），各 4 worker；`noise_run.py ship --finalize` 回 `$R/gate1/g1-new-{v9,x0}`；`noise_gate.py gen-regress check` 出 `GEN_REGRESS=`。
- 原侧补跑：`$R/scripts/chain_orig.sh` → `run_official_hard.sh --dataset test-hard0 --max-steps 1300 --gpu 0 --infra-retry-budget 3`（QwenVL 另 `--episode-wall 3600`、`SEAT_XLA_MEM_FRACTION=0.65`），stage `$R/gate2-orig-{qwenvl,pp}`。环境变量照抄上一轮 `seat_chain_v2.sh`。
- 补集合核对（起跑前）：上一轮 QwenVL 原侧非 infra 结果 39 局恰为 shard-00（与 shard-01～04 交集 0）；PonderPounce 96 局恰为 shard-00（与 shard-01 交集 0）。本轮补跑集合 = QwenVL 38+38+39+38 = 153、PonderPounce 96。

## ⑤ 代码改动（子代理分工）

S0（主会话，12.475）；S2a、S2b、S8、S1、S6、S7、S4、S5、S3 九个 opus 写入型子代理于 BASE 同时派出，按 S2a → S2b → S8 → S1 → S6 → S7 → S4 → S5 → S3 逐个合并（合并记录随合并追加到本节）。

### 合并记录

| 块 | 子代理提交（原样保留） | 合并提交 | PRE_MERGE_REVIEW | POST_MERGE_REVIEW |
|---|---|---|---|---|
| S2a 重绘工具 | `1420724e` | 12.478 `72698ee8`（+12.479 回放工具契约登记修补） | PASS files=3 findings=2 | 首次 FAIL（`TEST_INVENTORY` 未登记 12.476 的 `client_replay_eq.py`）→ 12.479 修补后 PASS 2896 passed |
| S2b 席位函数库与视频验收 | `21444f9d` | 12.480（+12.481 登记） | PASS files=6 findings=2 | PASS 2912 passed |
| S8 预算账本 | `345606ec` | 12.482（+12.483 登记） | PASS files=3 findings=2 | PASS 2934 passed |
| S1 GroundSG 官方视频 | `b1fd664e`、`b1325f81` | 12.484 | PASS files=4 findings=2（首交两个 slow 用例失败，主会话扩大可写集合续改） | PASS 2953 passed |
| S6 第二档对比 | `4d59d41a` | 12.485 | PASS files=3 findings=2 | PASS 2979 passed |
| S7 原侧观测器 | `91edf600`、`283761cf`、`189fac53` | 12.486（+12.487 登记） | 第一轮 PASS findings=8（含 1 高 2 中，交回续改）；第二轮 PASS findings=0 | PASS 3001 passed |
| S4 SimpleMemVLA／MME 新侧记录 | `7858efa5` | 12.489 | PASS files=3 findings=3 | PASS 3016 passed |
| S5 PonderPounce 子目标 | `e6965b13`、`40032a33` | 12.490（+12.491 登记） | PASS files=4 findings=2 | PASS 3035 passed |
| S3 Astra | `58ebcc4b`、`2354590a` | 12.492 | 第一轮 FAIL findings=3（缺 `--dataset`、`--keep-raw` 位置错）；第二轮 PASS findings=0 | PASS 3055 passed（Astra 测试走真实脚本 `ASTRA_MEDIA_MODE=real`） |

合并后审查每次都含：核心短测、`ls -1 scripts/*.py` 四入口、录像器 `RecordWrapper.py` 零 diff、`ORIG_FILES_UNCHANGED=PASS`（原侧文件、`third_party`、`src/robomme` 相对 BASE 零 diff）、`UPSTREAM_GUARD=PASS`。九个 worktree 与分支已按删前删后清单清理。

主会话集成修补：12.488 `official_media_check.sidecar_frames` 接受 S1 provenance 的字典帧数；12.493 `official_rerun_shard.sh` 的 E0 sha 清单可由 `ORIG_E0_SHA` 指定；12.494 新增 `orig-mme-client-env` 子项目。

## ⑥ 各批起跑前三件事与冻结提交

| 批次 | 冻结提交 | 执行副本 | RUN_INPUTS | CLIENT_REPLAY_EQ（base=b869a3df） | 本机 smoke |
|---|---|---|---|---|---|
| 2 GroundSG 新侧 | `0bfcc076` | `$N/robomme_benchmark-sgeval2-b2` | PASS deps=S0,S2a,S2b,S8,S1 | mmesg PASS（自检三类改动全抓到） | Oracle、QwenVL 各 1 局（hard0 VideoUnmask_xhard0_560300）：OFFICIAL_MEDIA=PASS total=1；QwenVL `OFFICIAL_RENDER=KEPT`（官方原生视频） |
| 3a SimpleMemVLA／MME 原侧 | `80a402cc` | `$N/robomme_benchmark-sgeval2-b34` | PASS deps=S0,S2a,S2b,S8,S7 | （原侧不经新客户端） | SimpleMemVLA 原侧 PickXtimes_xhard0_510300：3 次同为 success 818 步、OBSERVER_COMPLETE=PASS；MME 原侧：fail 688 步、`OBSERVER_TRANSPARENT=PASS messages=175 mismatch=0`、OBSERVER_COMPLETE=PASS |
| 3b SimpleMemVLA／MME 新侧 | `80a402cc` | 同上 | PASS deps=S0,S2a,S2b,S8,S4 | smvla、mme PASS | smvla/new 331 帧、mme/new 171 帧，OFFICIAL_MEDIA=PASS |
| 4 PonderPounce 新侧 | `80a402cc` | 同上 | PASS deps=S0,S2a,S2b,S8,S5 | pp PASS | pp/new 329 帧，OFFICIAL_MEDIA=PASS；子目标为模型输出（`at [486, 329]` → `at <84, 124>`） |
| 5 Astra 新侧 | `22a04e0c`（本机主检出） | 本机 | S3 已合入 | 不覆盖（S3 零外联夹具测试） | 见下 |

执行副本：`git clone --no-hardlinks` 本机仓库 → 检出冻结提交 → 初始化所需子模块 → 主 `.venv` 用共享解释器 `uv-python/cpython-3.11.14` 建、`uv sync --frozen --group eval-client`；b34 另建 `artifacts/v8-two/venvs/smvla-env`（`scripts/eval-official/smvla-env` 的 `uv sync --frozen`，共享解释器 3.10.19），建好后删掉子模块里的 `simplememvla.egg-info`，子模块与顶层均 clean。三方服务 venv：`MME_PY`、`PP_PY` 指向上一轮执行副本 `robomme_benchmark-sgeval`（同为锁定提交），`SGEVAL_CLIENT_PY=$N/sg-eval/venvs/client-env`。

起跑前发现并处理的环境问题（均未改评估代码语义）：
1. MME 权重 `eval-out/mmevla-ckpt/perceptual-framesamp-modul/79999` 是符号链接，`run_seat.sh` 读 `$MME_CKPT/../history_config.txt` 落空（`RUN_BLOCKED reason=history_config`，v8/v9two 留档同坑）。在 `$R2/ckpt/mme/perceptual-framesamp-modul/` 建实体目录：`history_config.txt` + `79999/`（`cp -al`），`ASSET mme-gl files=18 locked=18 missing=0 mismatch=0`。
2. `$NFS/v75eval/e0-sha.txt` 已随 10-03 清理删除；从 `artifacts/v7.5eval/input-manifest.json` 取 manifest、MME `episodes.jsonl`、SMVLA 逐局文件三组（剔除 `server.log`）重建 `$R2/inputs/e0-sha.txt`（31 行，每片恰 3 行，`sha256sum -c` 全部通过），经 `ORIG_E0_SHA` 传入。
3. MME 原版 `eval.py` 无条件导入 `google.generativeai`，且原版环境自带 `openpi_client`：新建 `$R2/venvs/orig-mme-client`（12.494 子项目，`--no-install-package robomme openpi-client`），再把 MME 原版克隆的 `packages/openpi-client`（与 ecf086c 逐字节相同）`--no-deps` 装入：`ORIG_MME_CLIENT_ENV=PASS genai 0.8.6 torch 2.9.1+cu128 sapien 3.0.2`、`ORIG_MME_OPENPI=PASS`。

## ⑦ GL 任务队列（2026-10-06 03:45 EDT 起）

`$R2/queue/{pending,running,done,failed}`：每个分片一份任务文件，worker `scripts/seat_worker.sh` 在占位 job 的 srun 步骤内按文件名顺序原子领取（`mv`），日志 `logs/tasks/<任务>.log`，`queue/STOP` 存在即退出。任务脚本：新侧 `task_new.sh`（批次 2）、`task_new_b34.sh`（批次 3b／4）、原侧 `task_orig.sh`（批次 3a，`official_rerun_shard.sh p2-orig <片> --run-idx 1`）。所有任务开 S8 共享预算账本 `SGEVAL_BUDGET_LEDGER=$R2/budget-ledger.jsonl`。

| 批次 | 档 | 分片 × 局数 | 席号 | stage |
|---|---|---|---|---|
| 2 Oracle | 第二档 | 4 × 48 | 30–33 | `$R2/gate2-new-oracle` |
| 2 QwenVL | 第二档 | 8 × 24 | 50–57 | `$R2/gate2-new-qwenvl` |
| 2 Oracle | 第三档 | 8 × 100 | 40–47 | `$R2/gate3-new-oracle` |
| 2 QwenVL | 第三档 | 16 × 50 | 60–75 | `$R2/gate3-new-qwenvl` |
| 3a 原侧 | 第二档 | 10 片（各 SimpleMemVLA + MME） | — | `$R2/gate2-orig-rerun/p2-orig` |
| 3b SimpleMemVLA、MME | 第二档 | 各 4 × 48 | 30–33 | `$R2/gate2-new-{smvla,mme}` |
| 3b SimpleMemVLA、MME | 第三档 | 各 8 × 100 | 40–47 | `$R2/gate3-new-{smvla,mme}` |
| 4 PonderPounce | 第二档 | 4 × 48 | 30–33 | `$R2/gate2-new-pp` |
| 4 PonderPounce | 第三档 | 8 × 100 | 40–47 | `$R2/gate3-new-pp` |

QwenVL 任务一律 `--episode-wall 3600`、`SEAT_XLA_MEM_FRACTION=0.65`；第二档 `--dataset test-hard0 --max-steps 1300`，第三档 `--dataset test-hard --max-steps 1600 --strict-cap`。

worker 会话（gl-login3 tmux）：`p2-w711`、`p2-w719`（即时）；`p2-w712`～`p2-w716`（各自等批次 1 该席会话结束后接上）；`p2-w720`、`p2-w534`、`p2-w535`（等占位 job RUNNING）。本机会话：`ovl2-smoke-*`（已结束）、`ovl2-astra-guard`、`ovl2-astra-try1`。

## ⑧ 批次 1 结果（第一档已完成）

- V9：`GEN_REGRESS=NEED_RERUN flip=1 set=v9 n=129 match=126 jitter=2 flip=1`（InsertPeg xhard4 seed 23300100）→ 定性第二次跑（改后 `robomme_benchmark-noise@b869a3df` 与旧码 `robomme_benchmark-noise-old@f8f76fba` 各 4 局：1 翻转 + 3 陪跑）→ 最终 `GEN_REGRESS=PASS set=v9 n=129 match=126 jitter=2 flip=1 … noise=1 regression=0 env_changed=0 unstable=0 noise_max=2`。
- xhard0：`GEN_REGRESS=PASS set=xhard0 n=48 match=48 jitter=0 flip=0`。
- PonderPounce 原侧补跑 shard-01（96 局）：`OFFICIAL_SEAT_DONE seat=11 policy=pp outcome=pass rc=0`。
- QwenVL 原侧补跑 shard-01～04 进行中。
