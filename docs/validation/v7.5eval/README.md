# v7.5eval 实施留档（新接口 · 官方重跑两遍 · 正式跑法 vs 官方）

> 方案：[`docs/plans/0929-v7.5eval-restructure-plan.md`](../../plans/0929-v7.5eval-restructure-plan.md)（下称「方案」），本目录是它的实测留档（方案第二部分 §6）。代码锚点：benchmark `newtaskRelease-v5`，工具提交链 12.266 `d393f59c` → 12.275 `198965ae`（见下表）；策略仓库零改动：MME `hongzefu/robomme_policy_learning_MotionJEPA` `eval-official-v1` = `ecf086c3`，SimpleMemVLA fork `eval-official-v1` = `c564c17d`，两者均无新提交。实跑一律从 detached worktree 钉提交起跑（方案 R6），本机在 `artifacts/v7.5eval/wt/<提交>`，GL 在 NFS `v75eval/wt/<提交>`。运行环境：环境 A（sled-vail，2 × RTX 6000 Ada）＋ GL spgpu A40 八席 ＋ standard 编排席一席。
>
> 写法：高层导读，判定行一律内联原文（取自 `artifacts/v7.5eval/summary/verdicts.txt`，由 `scripts/eval-official/step6_summary.py --final` 于 2026-09-30 08:10:46 EDT 生成，0 条 PENDING／BLOCKED，汇总脚本的终版与本留档同一提交落地）；逐局记录与视频在 `artifacts/v7.5eval/`（不进 git），本目录不复述其内容。GL 侧（NFS `v75eval/`）的状态、报告、日志、队列与跳过标记已归档到 `artifacts/v7.5eval/nfs-archive/`（849M，`rsync -rc` 核对 0 差异）后删除 NFS 原件；车道辅助脚本的最终版保存在 `scripts/eval-official/v75-lanes/{gl,local}/`。

## ① 一句话结论与指标速览

**SimpleMemVLA：正式跑法（新接口、A40、多席共享 NFS 认领队列按打乱顺序消费）192 局与官方历史成绩、官方重跑一、官方重跑二逐局终态与步数全部相同（0 翻转），官方重跑本身三对也全 0 翻转 → `RELATIVE_ACCEPT … inside=n/a note=官方重跑无翻转 prod_flips=none`。MME：正式跑法对官方重跑一 7 局成功→失败、9 局失败→成功，成功率 +1.04 个百分点（95% 置信区间 [−3.125, 5.208]）；两个方向的翻转数都在官方自己两两重跑的最大值以内（7 ≤ 10、9 ≤ 9），但成功率差点估计不在官方三对的区间 [−1.04, −0.52] 内 → `RELATIVE_ACCEPT … inside=no`，交用户判断。** 环境侧：同 GPU 型号下换卡、换机器、常驻倒序全部逐字节一致，换 GPU 型号（RTX 6000 ↔ A40）则 44/48 身份有差；旧环境栈（sapien 3.0.3＋官方 `robomme`）与新环境栈（sapien 3.0.2＋`robomme_hard` xhard0）在同卡上 reset 逐字节相同。

| 项 | 判定（原文摘要，全文见各分文档） | 文档 |
|---|---|---|
| 第 0 步冻结 | `ASSETS=PASS assets=45 mismatches=0`、`E0_FREEZE=PASS files=440`、`IDENTITY_FREEZE=PASS small=48 full=192` | [preflight.md](preflight.md) |
| 2.1 环境 | 同型号 4 对 `first_diff=-`；`pair=C1:C5 … identities_with_diff=44 … image_mad=25.8414` | [env-parity.md](env-parity.md) |
| 旧／新环境栈 | `ENV_STACK=INFO src=O1 … cell=C5 … frames_sha_equal=17646/17646 … state_max_abs=0`（两策略、C5／C7／C1 共 6 行全等） | [env-parity.md](env-parity.md) |
| 2.2 录制器 | `OBSERVER_SMOKE=INFO identity=PickXtimes/510300 file_equal=yes frames_equal=yes frames=816/816 … steps=815/815` | [official-rerun.md](official-rerun.md) |
| 2.3 官方噪声带 | SMVLA 三对全 0 翻转；MME `历史:重跑一 s2f=4 f2s=3`、`历史:重跑二 s2f=9 f2s=7`、`重跑一:重跑二 s2f=10 f2s=9`（各 192 局） | [official-rerun.md](official-rerun.md) |
| 第 3 步跑通 | `STEP3=INFO policy=smvla … status=success steps=818 … sha_mismatch=0`；`TRANSPORT=PASS frames=993 mismatch=0` | [eval-conditions.md](eval-conditions.md) |
| 第 4 步策略 | SMVLA 关态 16/16 逐位一致（P1/P3/P5/P7）；MME 关态重启不一致 `max_abs≤0.0096`（< `action_max=0.0413`），开态 16/16 一致但减速 46.8～52.2% → 两策略 `default=off`；`IFACE_OPEN` 两策略 `payload_equal=yes exec_equal=yes` | [policy-replay.md](policy-replay.md) |
| 5.1 换条件 | SMVLA 同型号各对（含 E5:E6）0 翻转、`R1:E1` 新旧接口同卡 0 翻转；MME 各对 2～3 翻转／方向 | [eval-conditions.md](eval-conditions.md) |
| 5.2 正式跑法 | `QUEUE_CLAIM=PASS` × 4；`PROD_MERGE=INFO policy=mme from_N=91 from_N2=100 from_N3=1 … conflicts=0`；`CANARY` 两策略各 8/8；`KILLTEST` 两策略 `server_killed=yes server_restarted=yes … errors=0 … queue_check=PASS` | [prod-vs-official.md](prod-vs-official.md) |
| 主比较 | SMVLA `ref=重跑一 … s2f=0 f2s=0 … ci=[0,0]`；MME `ref=重跑一 … s2f=7 f2s=9 … sr_diff_pp=1.0417 ci=[-3.125,5.208]` | [prod-vs-official.md](prod-vs-official.md) |
| 相对标准 | `RELATIVE_ACCEPT=INFO policy=smvla inside=n/a … note=官方重跑无翻转 prod_flips=none`；`RELATIVE_ACCEPT=INFO policy=mme inside=no s2f=7 f2s=9 max_off_s2f=10 max_off_f2s=9 sr_diff_pp=1.0417 off_sr_range=[-1.0417,-0.5208] …`；两策略 `EXTRA_SAMPLE … trigger=no` | [summary.md](summary.md) |
| 事故 | 6 起，最重的是 GL `--gpu_cmode=shared` 步骤结束把卡重置为独占，造成 347 次 0 步基础设施失败、0 局真实结果丢失 | [incidents.md](incidents.md) |
| 预算 | `BUDGET_TOTAL=INFO trajectory_attempts=2151 cap=2271 over=no incident_infra_attempts=347 all_attempts=2498 attempts_over_items=5.2_金丝雀 retry_over_items=2.2_录制器验证,2.3_官方重跑两遍,3.1_跑通,5.1_换条件评估,5.2_正式跑法,5.2_金丝雀,其他_杀server测试 shared_retries=6 shared_retry_cap=2` | [summary.md](summary.md) |

## ② 用户指令原话（本轮，按时间，编号供各文档引用）

方案制定期的原话 1～33 逐字写在方案文首，各文档记作「方案原话 N」；下面是开跑后本轮的原话，记作「本轮原话 N」。

1. 「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/0929-v7.5eval-restructure-plan.md 开始实现 有问题在半小时内问用户」
2. 「之后尽可能一口气跑完跑到底。」
3. 「如果有可以并行改代码的内容可以用subagent来改但是subagent必须是Opus」
4. 「但是你改完了代码你要自己check一下或者用别的agentchcheck。」
5. 「还是通过用SubAgent的形式来保证你的上下文窗口少压缩几次」
6. 「全部跑完要多久」
7. 「同意合并 尽可能并行」（回应：P2 并入 P1、P6 并入 P5）
8. 「你一边改一边启动一个subagent。评估这个并行的情况是否还能并行如果可以的话就自己决策自己改但是该测的都要测完我要睡觉了不要再来问我」

半小时提问窗口（本轮原话 1）：全部问题可由方案与代码解决，未提问。

## ③ 提交链与席位

| 编号 | 提交 | 内容 | 被哪些实跑钉住 |
|---|---|---|---|
| 12.266 | `d393f59c` | 两子模块 ＋ 依赖组 `eval-client` | — |
| 12.267 | `2176a0e3` | `hard_regression.py env-digest`／`env-digest-compare`、`compare.py`；第 0 步冻结 | 2.1 六格 |
| 12.268 | `1112197c` | 新接口、认领队列、`run_seat.sh`、编排器与看门狗 | — |
| 12.269 | `4c19c39a` | `recorder.py`、`official_observer/`、`official_rerun_shard.sh` | 2.2、2.3（O1／O2） |
| 12.270 | `2abf227d` | 片级脚本 SMVLA 分片数取自清单 | 2.4（R1）、第 3 步 |
| 12.271 | `30257b46` | 第 4 步 `policy_replay.py`／`run_policy_replay.sh`；SMVLA `--det` | 第 4 步 P1／P3／P5／P7 |
| 12.272 | `a81c4f6b` | `run_seat.sh --client-per-task` | 5.1 E1～E7、5.2 N／N2／N3 |
| 12.273 | `bac285d8` | 回放支持官方 MME 多局代理根目录 | 输入 B-mme、新旧接口开环 |
| 12.274 | `f724f07e` | `step6_summary.py` | — |
| 12.275 | `198965ae` | 汇总按事故后数据布局更新 | 本留档的判定行 |

席位（GL `chaijy2`／`spgpu`，每席 1 × A40、4 CPU、48G、48 h，清单 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/gl-hold-logs/hold-jobs-envdet-20260929.txt`）：甲 62608429（gl1513）、丙 62608430（gl1513）、乙 62608440（gl1525）、丁 62608595（gl1527）、新1 62618838（gl1512）、新2 62618839（gl1528）、新3 62618840（gl1528）、新4 62618841（gl1528）；编排席 62612889（gl3009，standard，1 CPU／4G，无 GPU）。新增四席首次提交 62618785／86／88／90 全落 gl1525（与乙同节点），未跑任何工作即取消，按 `--exclude=gl1525,gl1513,gl1527` 重交。本机：sled-vail 卡 0（`taskset -c 0-3`）、卡 1（`taskset -c 4-7`）。

## ④ 分文档索引

| 文档 | 覆盖方案步骤 |
|---|---|
| [preflight.md](preflight.md) | 第 0 步：资产锁、E0 冻结、身份冻结、环境栈差异、旧代码可用性 |
| [env-parity.md](env-parity.md) | 2.1 环境检测六格五对；第 6 步旧／新环境栈；环境测速 |
| [official-rerun.md](official-rerun.md) | 2.2 录制器验证；2.3 官方重跑两遍与噪声带；2.4 本机旧官方小样本 |
| [policy-replay.md](policy-replay.md) | 第 4 步策略回放、确定性标志、编译缓存、新旧接口开环 |
| [eval-conditions.md](eval-conditions.md) | 第 3 步跑通与传输核对；5.1 换条件评估六格 |
| [prod-vs-official.md](prod-vs-official.md) | 5.2 正式跑法（队列、金丝雀、杀 server）；主比较与参照比较 |
| [incidents.md](incidents.md) | 6 起计划外事件：时间线、根因、影响（乘式）、处置、证据、教训 |
| [summary.md](summary.md) | 第 6 步结论：相对标准、噪声带、测到／没测到、偏离方案、预算 |

## ⑤ 编排器与会话总表

- **GL 编排器实例**（登录节点 gl-login3 的 tmux 内，经 `srun --jobid=62612889 --overlap` 跑在编排席；计划与状态归档在 `artifacts/v7.5eval/nfs-archive/state/<实例>{-plan.json,/}`，启动器 `scripts/eval-official/v75-lanes/gl/orch_<实例>.sh`）：`official`（2.2＋2.3）、`main`（第 4 步 GL、5.1 GL、5.2、金丝雀）、`o2x`（改派 O2）、`mmeprod`（MME 首轮）、`retry`、`final`（事故后补跑）、`final2`（E7 MME 重跑）、`final3`（E6 SMVLA 重跑）、`final4`（O2 SMVLA 续跑，空操作）、`final5`（N3）、`final6`（O2 片 0、6 改派）、`final7`（MME 杀 server 测试 K 与新2 金丝雀）、`final8`（金丝雀补跑）。登录节点另有 2.1 的 `v75-env-yi`、`v75-env-ding` 与保持步骤 `v75-keep-*`、`v75-keep2-*`（见 [incidents.md](incidents.md) 事故 6）；这些登录节点 tmux 会话全部自行结束，收尾时已无残留。
- **本机 tmux**：收尾时按精确名删除 `v75-aggregate`、`v75-aggregate2`～`v75-aggregate12`、`v75-mover`；此前 `v75-env-card0`、`v75-env-card1`、`v75-r1-card0`、`v75-replay-card1`、`v75-step3-card1`、`v75-chain0`、`v75-chain1`、`v75-bmme`、`v75-assets` 已自行结束。启动脚本在 `scripts/eval-official/v75-lanes/local/`。
- **收尾**（方案第 7 步）：约 08:1x EDT 按清单逐个 `scancel` 9 个 JobID（62608429 62608430 62608440 62608595 62618838 62618839 62618840 62618841 62612889），未用 `scancel -u`；NFS `v75eval/` 小文件归档到 `artifacts/v7.5eval/nfs-archive/`（849M，`rsync -rc` 差异 0）后删除 NFS 上的 worktree、venv、权重副本与中转目录；本机临时 worktree 与 `dev-*` 草稿目录删除。

## ⑥ 归档与证据位置

- 汇总：`artifacts/v7.5eval/summary/`（`summary.json`、`verdicts.txt`、`detail/*.json`、`merged/*.jsonl`、`fragments/*.md`），终值日志 `artifacts/v7.5eval/logs/step6-final.log`。重跑命令：`uv run --no-sync python scripts/eval-official/step6_summary.py --final`，只写该目录。
- 逐局结果与录制：`artifacts/v7.5eval/{env,official-rec,newiface,replay}/`（失败尝试 `official-rec/E6/smvla/s-yi-vulkanfail-0544`、`…/E7/mme/s-ding-vulkanfail-0540` 留证）；GL 侧状态、报告、日志、队列：`artifacts/v7.5eval/nfs-archive/{state,reports,logs,queue,skip,replay}/`。
- 本目录只有 Markdown，不含 `.sh`／`.yaml`／数据文件。
