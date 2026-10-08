# sg-eval-gl-20261006-03 结果（result.md）

> 起跑记录见同目录 `launch.md`（①～⑫ 节，含运行中的全部变更与用户裁决原话）。计划：根目录 `1006-rename-official-names-and-stage3-eval-plan.md`。

## ① 一句话结论与指标速览

五模型 OOD 第三档（V9 新值档，1800 步 strict、模型 seed 7、每格 2 局）430 局全部完成并通过全部验收；本机 MemER hard-verify 原侧 vs 新侧对拍（按用户裁决缩减为每侧 `16 任务 × 3 局 = 48`）完成，差异报告 `GATE2=INFO`。

| 模型 | 局数 | 成功 | 局级成功率 | 任务宏平均 |
|---|---|---|---|---|
| SimpleMemVLA | 86 | 19 | 22.1% | 19.1% |
| PonderPounce（r2 重跑） | 86 | 15 | 17.4% | 14.5% |
| GroundSG+QwenVL | 86 | 13 | 15.1% | 14.6% |
| MemER | 86 | 5 | 5.8% | 6.3% |
| FrameSamp+Modulation | 86 | 3 | 3.5% | 4.7% |

PonderPounce 一行取 FIX-3（补记 S2 输入全文）后的 4 卡重跑 `ood-new-pp-seed7-r2`：86 局终态与执行步数与首跑逐局相同，成绩不变（见 `launch.md` ⑬）。每个任务难度格 n=2，成绩只看大致排序，不作细分比较；未跑 seed 0／42，不报多种子均值。与旧 1600 步成绩条件不同（步数上限、SimpleMemVLA／PonderPounce 模型种子由旧实际 0 改为 7），不宣称等价或无回归。GroundSG+Oracle 本轮未跑，沿用 1004 轮 1600 步成绩；3-tier Astra 只改代码未实跑。

总闸门：
- `STAGE3_MATRIX=PASS policy_seed=7 combinations=5 unique_terminal=430 incomplete=0 seed_mismatch=0 duplicate_combination=0`（PP 换 r2 后重算，结果同首算）
- `MODEL_EVAL_REPORT=PASS sets=5 complete=5 incomplete=0 episodes=430 gate2=1`
- `BUDGET_ENFORCEMENT=PASS trajectories=530/870 resets=1254/141430 astra=0/2 shared_infra=1/50`（`BUDGET_DETAIL reserves=530 committed=528 released=0 open=2 expired=0/50 first_started=529/821 recovery=1/49`；`open=2` 为 MemER 换片时被 TERM 打断的两次预约，未提交、按已用计）。PP 重跑沿用首跑幂等 token、账本未新增 reserve，实际轨迹消耗按 530 + 88 ≈ 618 计（< 870）；reset 1078 → 1254 为重跑如实计量

## ② 版本与代码状态

冻结执行提交 `0816c0a9fa5c2ca37b47bba07dd4c36334f2fb79`（GL 执行副本 `robomme_benchmark-sgeval3`）；PonderPounce r2 重跑用 `40c2642e`（12.535，含 FIX-3／FIX-4）。运行期间主检出只合入两处与运行无关的修补：12.529 `eval_report.py` 上限核对（FIX-2，验收工具）、文档提交。本机对拍以主检出运行（代码与 `0816c0a9` 相同）。

本轮代码链：12.519 改名（R1）→ 12.520 文档与接口冻结 → 12.522 第三阶段功能（R6／R2／R3／R7／R5 + MERGE-1）→ 12.524 FIX-1／R4／D2 修复（MERGE-2）→ 12.529 FIX-2 → 12.533 FIX-4（分片 lease id 含数据集）→ 12.534 FIX-3（PP S2 输入全文进语言账本）。每块合并前审查与合并后审查见各合并提交 body。

## ③～⑥ 启动、数据、超参、硬件

见 `launch.md` ②～⑥。补充实际用时：GL 五模型 OOD 2026-10-06 21:25 起至 2026-10-07 08:47 止（约 11.4 小时，4 × A40）；本机对拍 2026-10-06 21:23 起、22:00 缩减重启、2026-10-07 06:03 止（2 × RTX 6000 Ada）。

## ⑦ 运行过程行为

| 组 | 分片 | 用时与异常 |
|---|---|---|
| FrameSamp+Modulation | 2 × 43 | 每片约 29 min，零错误零重试 |
| PonderPounce | 2 × 43；r2 重跑 4 × 21～22 | 首跑第一片 1 局 S2 上下文超限按基础设施重试（见下）；r2 四片 10:29～11:13（约 44 min），同一局同样重试 1 次，其余零重试 |
| SimpleMemVLA | 2 × 43 | 每片约 1.6 h，零错误零重试 |
| GroundSG+QwenVL | 2 × 43 | 每片约 3.5～4.5 h，零错误零重试 |
| MemER | 2 片 → 换 4 片 | 1800 步超时局 46～56 min（113 次子目标提问，A40 约 24～30 s／次）；02:13 由 2 片换 4 片（`launch.md` ⑪），08:47 完成 |

单列（用户裁决「沿用现状当基础设施重试」）：PonderPounce S2 上下文 16384 token 上限在 1800 步下被触及 **1 例**——`VideoPlaceOrder_xhard1_17100000` 第 1 次尝试第 1661 步 `S2 context length 16439 exceeds cap 16384`，按基础设施错误重试，第 2 次 1057 步 `fail` 被接受；全部 430 局中仅此 1 例。成绩含此次重跑。r2 重跑在同一局、同一步、同一长度上再次触发，重试结果也相同（1057 步 fail），说明该超限可确定复现。

运行中事件与处置（详见 `launch.md` ⑧～⑫）：
1. MemER 第一片分片锁撞名（lease 名 = 路线 + 分片文件名，路线不含数据集，与本机对拍同名 `shard-00.json`）→ 逐字节副本改名重排，零预算消耗。缺陷：lease 名应含数据集，待后续修。
2. SimpleMemVLA `EVAL_REPORT` 误报 `cap_mismatch=86`（trace header `max_steps=1840` 为理论动作上界）→ 用户裁决按计划修检查器（FIX-2，12.529），重跑 PASS。
3. MemER 换片时被 TERM 打断的两局（`ButtonUnmask_xhard2_18800100`、`ButtonUnmask_xhard4_22800000`）在新席重跑，媒体根下与半局目录同名，发布工具按不覆盖规则把完整局落为 `.a1.dup1`。验收以 `reports/memer/accepted-view/`（`cp -al` 硬链接视图，只含 86 个被接受的完整局）为根；原局目录未动。
4. 本机对拍首局实测约 16 min／局，按原口径 50～65 小时 → 用户裁决缩减为每任务 3 局（`launch.md` ⑧）。

## ⑧ 验收（逐组，全部 PASS）

每组判定行（`reports/<组>/`，MemER 以已接受视图为根）：

| 判定 | FrameSamp+Modulation | SimpleMemVLA | PonderPounce | GroundSG+QwenVL | MemER |
|---|---|---|---|---|---|
| `EVAL_COVERAGE` expected=86 missing=0 | PASS | PASS | PASS | PASS | PASS |
| `EVAL_REPORT` cap=1800 exec_over_cap=0 cap_mismatch=0 | PASS | PASS（FIX-2 后） | PASS | PASS | PASS |
| `EVAL_VIDEOS` videos=86 decode_fail=0 | PASS | PASS | PASS | PASS | PASS |
| `OFFICIAL_MEDIA` total=86 fail=0 no_frame_error=0 | PASS | PASS | PASS | PASS | PASS |
| `TRACE_ARRAYS` attempted_steps_missing=0 observed_state_missing=0 | PASS（86） | PASS（86） | PASS（87，含重试第 1 次） | PASS（86） | PASS（86） |
| `LANG_IO` 中 PP 的 S2 输入（`subgoal_model` in） | — | — | r2：87／87 局含 | — | — |
| `LANG_IO` unresolved=0 open_calls=0 image_ref_unresolved=0 server_text_empty=0（`--require-server-text`） | PASS | PASS | PASS | PASS | PASS |
| `VIDEO_LAYOUT` published=86 error_named=0 | PASS | PASS | PASS | PASS | PASS |

PonderPounce 列为 r2 重跑的验收（`reports/pp-r2/`，首跑验收 `reports/pp/` 亦全 PASS）。

视频按上游 mme-vla 布局发布于 `$N/sgeval-20261006-03/publish/<模型 ID>/seed7/[qwenvl|memer/]videos/`（硬链接，附 `index.tsv`）；PP r2 发布于 `$N/sgeval-20261006-03/publish-pp-r2/pp/seed7/videos/`。

**视频站**：http://sled-vail.eecs.umich.edu:8084/ （tmux `stage3-site-8084`，日志 `artifacts/sg-evaluation/sg-eval-gl-20261006-03/logs/site-8084.log`）。按用户要求只放视频与成功率、不放语言记录，版式沿用 8083 Oracle 站；按用户「把ground truth的也放入」「GroundSG+Oracle和生成的ground truth data也都放入」「按模型与 Task 浏览视频 改为按task 同时显示所有模型 可以勾选不显示」，页面按 Task 逐局并排显示生成真值（V9 交付的专家轨迹录像，按 task／tier／seed 对齐）、GroundSG+Oracle（1004 轮本机 1600 步、同 86 局身份、`spec_sha256` 86／86 一致，条件不同、灰色仅作参考，86 局成功 38）与五个模型，每路可勾选隐藏（浏览器本地记忆）、官方版式／原始画面统一切换、同步播放。`scripts/injection-dev/site/stage3_eval_site.py` 由五组发布清单建目录（逐局核对发布硬链接 inode、轨迹终态与身份），五模型视频从 NFS 复制进本机 `artifacts/sg-evaluation/sg-eval-gl-20261006-03/site-media/`（1.5 GB），站点目录 `…/site-stage3/`、媒体根 `artifacts/`：`STAGE3_SITE=PASS models=6 episodes=86 media=1118 smvla=19/86 pp=15/86 groundsg-qwenvl=13/86 groundsg-memer=5/86 framesamp=3/86 groundsg-oracle=38/86`；`MEDIA_FETCH=PASS total=1118 bad=0`；浏览器检查 `STAGE3_SITE_BROWSER=PASS videos_loaded=42 failed=0 page_errors=0 hide=True persist=True phone_scroll_width=390`。

## ⑨ 本机 MemER hard-verify 对拍

- `GATE2_INPUTS=PASS expected=48 missing=0 extra=0 unaccepted=0 ambiguous=0 trace_binding_mismatch=0`
- `GATE2_PROVENANCE=PASS mode=host expect_host=sled-vail foreign_rows=0 unknown_rows=0`
- `GATE2=INFO policy=groundsg-memer compared=48 same_terminal=41 s2f=0 f2s=4 sr_orig=0.2083 sr_new=0.2917 sr_diff_pp=8.33 mcnemar_p=0.125 identical_trace=0 matrix=ss:10,sf:0,st:0,fs:4,ff:26,ft:2,ts:0,tf:1,tt:5 prompt_diff=3078 reply_diff=1067 site=local`
- 两侧 48 局全部在运行中分叉（`identical_trace=0`）；首处语言差异为子目标模型回复坐标 `(203,629)` vs `(203,625)`（BinFill 源局 11 第 10 次提问）：GPU 推理非逐位可复现所致，终态差异不能归为接口行为不一致。差异报告，不证明等价；成功率差 8.3 个百分点、McNemar p=0.125，统计上不显著。
- 新侧 `OFFICIAL_MEDIA=PASS total=48 videos=48`、`TRACE_ARRAYS=PASS`、`LANG_IO=PASS`、`VIDEO_LAYOUT=PASS`；原侧 `TRACE_ARRAYS=PASS`、`LANG_IO=PASS`、`VIDEO_LAYOUT=PASS`（原侧无尝试账本，`--accept-from results`）。两侧均完整记录服务端分词通道与 `server_final_text`。
- 报告 `artifacts/sg-evaluation/sg-eval-gl-20261006-03/compare/report/gate2-memer.{json,md}`（副本 `records/reports/gate2-memer.md`）。

## ⑩ 用户决策记录（本轮新增，原话）

- 「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/1006-rename-official-names-and-stage3-eval-plan.md 开始实现 有问题问用户但不要阻塞」
- 「你在本机跑MMER。是并行两个job来跑吗本集有两块卡」
- 「先把Greatlakes跑完再说」
- 「本机的对拍缩减规模。先给我方案」→ 选「每任务 3 局（推荐）」
- PonderPounce S2 上下文超限 → 选「沿用现状当基础设施重试」
- SimpleMemVLA 上限误报 → 选「按计划修检查器（推荐）」
- 「PonderPounce需要补」→ 选「重跑 PonderPounce 86 局」
- 「共享账本的分片锁名不区分数据集，这次靠改分片文件名绕开了。要不要另外修？这个也要跑」
- 「ponderponce能否4卡完全并行」
- 「GreatLakes释放然后重新排队这次query5天的。」
- 「把ground truth的也放入」「GroundSG+Oracle和生成的ground truth data也都放入」「按模型与 Task 浏览视频 改为按task 同时显示所有模型 可以勾选不显示」
- 「http://sled-vail.eecs.umich.edu:8070/ 参考这个做一个你跑完的网页」「只放视频就可以 成功率也要 language的记录就先不放了」「说错了 是这个http://sled-vail.eecs.umich.edu:8083/#task=BinFill&episode=BinFill_xhard1_16400000.a1」

## ⑪ 结论、遗留与下一步

- 计划第三段全部判定项完成：五模型 OOD 430 局、`STAGE3_MATRIX=PASS`、`BUDGET_ENFORCEMENT=PASS`；MemER hard-verify 对拍按缩减规模完成。
- 遗留（待用户定）：
  1. ~~PonderPounce S2 输入全文未进语言账本~~ → FIX-3 已修，r2 重跑验证（`launch.md` ⑬）。
  2. ~~共享账本分片 lease 名不含数据集~~ → FIX-4 已修。
  3. PP 重跑沿用幂等 token 未新增 reserve：账本对「有意整组重跑」不扣额度，下次重跑应换 token 前缀（如 `RUN_PREFIX` 进 token）——本轮未改代码，只在此登记。
  4. ~~GL 4 个占位 job 空转~~ → 2026-10-07 13:18 按用户「GreatLakes释放然后重新排队这次query5天的。」释放 63188714／15／16／19，改提 4 个 5 天占位 job `sgev-hold-10`～`13`（63431430～63431433；提交时 3 个 RUNNING、63431433 `PENDING (AssocGrpGRES)`），清单 `gl-hold-logs/hold-jobs-sgeval-20261004.txt`。
  5. ~~本机遗留 pytest 假服务进程 PID 2961357~~ → 2026-10-07 按用户「旧 pytest 进程（PID 2961357） 清理」TERM 结束（孤儿进程，`serve_policy.py --port=23310`，checkpoint 指向 pytest 临时目录，已运行 3 天 19 小时，无子进程）；端口 23310 已释放。

## ⑫ 归档文件清单

- `records/scripts/`：GL 运行根脚本（`gl-scripts_*.txt`：task_ood、launch_seat、seat_worker、enqueue、check_smoke、accept_group、watch_all、build_exec、memer_rebalance）与本机脚本（`local_*.txt`：MemER smoke 三份、run_side／run_side48、清单生成 build_ood86／build_hv192／build_hv48）。
- `records/reports/`：五组 `<组>/report.md`、`pp-r2/{report.md,accept.txt}`、`stage3-report.md`、`stage3-sets.json`（均为 PP 换 r2 后的版本）、`gate2-memer.md`。
- `records/scripts/` 另含 `gl-scripts_accept_group_r2.sh.txt`、`gl-scripts_watch_r2.sh.txt`。
- 产物（不进 git）：`$N/sgeval-20261006-03/`（stage、media、publish、reports、queue、logs）；本机 `artifacts/sg-evaluation/sg-eval-gl-20261006-03/`（smoke、compare、inputs）。
