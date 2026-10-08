# sg-eval-gl-20261006-02 结果（result）

> 两段式第二段。第一档、第二档、批次 5（Astra smoke）已完成；第三档按用户 2026-10-06 13:1x EDT「停止现在第三档的evaluation」「立刻收尾」中止，只有部分结果（见第四节）。判定行原文见 [`records/verdicts.md`](records/verdicts.md)，逐局差异表见 `records/diff/`。

## ① 一句话结论

六条新接口路线都已能直出官方版式视频：第二档五个模型共 5 × 192 = 960 局 `OFFICIAL_MEDIA=PASS fail=0`（GroundSG 两变体保留官方 `RolloutRecorder` 原生视频，其余由官方原类重绘），Astra 本机 1 局 smoke 通过（0.30 美元）。第二档「原侧 vs 新侧」：SimpleMemVLA 192 局在可共同观察的五个维度上逐步逐位一致；PonderPounce 终态 192/192 相同、执行动作逐步全一致；GroundSG Oracle、QwenVL 与 MME 的终态差异都在各自的服务端数值噪声范围内，成功率差不显著（McNemar p ≥ 0.63）。第一档生成回归 V9、xhard0 均 `GEN_REGRESS=PASS`。第三档按用户指令中止：GroundSG Oracle 已定终态 542/800（成功 270）、QwenVL 163/800（成功 15），PonderPounce、SimpleMemVLA、MME 未开始；已跑部分的账本与媒体保留在 NFS，可续跑。

## ② 第一档：生成回归（177 局）

V9 43 格 × 3 局 = 129、xhard0 16 任务 × 3 局 = 48，GL A40，执行副本 `robomme_benchmark-noise@b869a3df`，对照参照 `scripts/configs/noise-ref-20261003.json`。

| 集合 | 首跑 | 定性重跑 | 最终 |
|---|---|---|---|
| V9 | match 126、jitter 2、flip 1（InsertPeg xhard4 seed 23300100）→ `NEED_RERUN` | 1 翻转局 + 3 陪跑局，改后代码与旧代码 `f8f76fba` 各 4 局 | `GEN_REGRESS=PASS … noise=1 regression=0 env_changed=0 unstable=0` |
| xhard0 | match 48 | — | `GEN_REGRESS=PASS` |

结论：本阶段代码改动（只动评估侧 `scripts/eval-official/`）对生成无回归；V9 那 1 局翻转判为噪声（旧代码同样翻转）。

## ③ 第二档：原侧 vs 新侧（五模型各 16 任务 × 1 档 × 12 局 = 192）

| 模型 | 原侧来源 | 成功率 原侧 / 新侧 | 终态相同 | 翻转 s→f / f→s | McNemar p | 逐步比较 | 参照 |
|---|---|---|---|---|---|---|---|
| GroundSG Oracle | 上一轮 GL 192 局（官方 `eval_each_episode` 原文） | 75.0% / 75.0% | 188/192 | 2 / 2 | 1 | 144 局全字段一致；48 局在第 0 步动作块分叉 | 分叉 48 局全部来自同一个新侧服务进程（gl1527 上 s32 那片；同节点 s33 片 48 局全一致） |
| GroundSG QwenVL | 上一轮 39 局 + 本轮补跑 153 局（`GATE2_SUPPLEMENT=PASS`） | 22.9% / 24.0% | 177/192 | 3 / 5 | 0.73 | 86 局全字段一致 | 预测器与 JAX 服务都有数值非确定性 |
| PonderPounce | 上一轮 96 局 + 本轮补跑 96 局（`GATE2_SUPPLEMENT=PASS`） | 41.7% / 41.7% | 192/192 | 0 / 0 | 1 | 执行动作 192 局逐步全一致 | 画面、状态各差恰 1 处／局（新侧按 C2 多记初始帧）；子目标新侧改记模型自己的输出 |
| SimpleMemVLA | 本轮重跑原版 `4e0c04f` + 只读观测器 | 73.4% / 73.4% | 192/192 | 0 / 0 | 1 | 动作、画面、状态、逻辑输入、子目标五维 192 局逐步全一致 | 原侧重跑 vs E0 0 翻转；E0／O1／O2 两两 0 翻转 |
| MME | 本轮重跑原版 `927c56d` + 只读观测器 + 透明代理 | 25.5% / 27.1% | 174/192 | 7 / 10 | 0.63 | 59 局动作逐步一致 | 官方噪声带：E0-O1 7、E0-O2 16、O1-O2 19；本轮原侧重跑 vs E0 15 |

口径说明：
- `identical_trace` 要求所有维度相同，原侧不可观察的 `terminated`／`truncated` 记 `NOT_OBSERVED`，按用户裁决「只比共同可观察字段」不算相同，所以 SimpleMemVLA、MME、PonderPounce 的 `identical_trace` 为 0 并不表示行为差异，应看分维度投影（`GATE2_PROJ`）。
- GroundSG、MME 的逐步分叉都在第 0 步动作块、输入哈希相同，与计划八节订正一致：属服务进程启动时 XLA 自动调优／JIT 的数值非确定性，按噪声报告，不是新接口改了行为。
- 第二档是差异报告，不证明等价；以上均为 GL A40 同环境（`GATE2_PROVENANCE=PASS local_rows=0`）。

## ③′ 第二档新侧覆盖与官方视频

五模型各 `EVAL_COVERAGE=PASS expected=192 missing=0`、`EVAL_REPORT=PASS media_unexplained=0`、`EVAL_VIDEOS=PASS videos=192`、`OFFICIAL_MEDIA_INPUTS=PASS`、`OFFICIAL_MEDIA=PASS total=192 fail=0`。

## ④ 第三档（中止，只有部分结果）

用户 2026-10-06 13:1x EDT：「停止现在第三档的evaluation」，随即「立刻收尾」「只保留4快卡 最新的job 其他资源全部释放 ！先做这个」。

| 模型 | 分母 | 已定终态 | 成功 | 失败 | timeout | 缺失 | 已定终态内成功率 |
|---|---|---|---|---|---|---|---|
| GroundSG Oracle | 800 | 542 | 270 | 133 | 139 | 258 | 49.8%（270/542） |
| GroundSG QwenVL | 800 | 163 | 15 | 115 | 33 | 637 | 9.2%（15/163） |
| PonderPounce、SimpleMemVLA、MME | 800 | 0 | — | — | — | 800 | 未开始 |

- 判定行（`--partial`）：Oracle `EVAL_COVERAGE=FAIL … missing=258 partial=1`、`EVAL_REPORT … count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=0`、`EVAL_VIDEOS … videos=541 decode_fail=0`；QwenVL `missing=637`、`videos=160 decode_fail=0`。已定终态的局官方视频与普通视频都在 `$R2/media/` 下。
- 已定终态的局来自按 `rows[i::k]` 等距切出的分片中先跑完的那部分，不是随机抽样，成功率不能当作全量估计。逐任务表见 `records/diff/gate3-partial-{oracle,qwenvl}.md`。
- 续跑方法：`queue/held-gate3/` 里是 36 个未领取任务与 7 个被中断任务（`interrupted__` 前缀），删除 `queue/STOP` 后放回 `queue/pending/` 并起 worker；`run_eval_gl.sh` 按各席账本续跑，被中断的局按基础设施中断重评。

## ⑤ 批次 5：Astra 本机 smoke

1 局（VideoUnmask，test-hard0 本地局号 0 = 官方 episode 3，1300 步）：success，282 步；官方视频重绘 349 帧、转码 349 帧、`OFFICIAL_MEDIA=PASS total=1`；费用守卫 `--cap 5`，3 次请求共 0.3043 美元。第 2 局额度未使用（用户「唯一需要注意的就是astra还是只跑两次」）。

## ⑥ 计划外事件与处置

1. 12.476 回放工具、12.494 环境子项目在 `git add` 之前跑的清单测试看不到新文件，合并后核心短测才暴露 `TEST_INVENTORY=FAIL`；由 12.479、12.496 补登记。
2. S1↔S2b 接口：GroundSG 原生视频的 provenance 帧数是字典，验收把它当整数比，本机 smoke 中原生视频被误判后改走重绘；12.488 修，QwenVL smoke 证实 `OFFICIAL_RENDER=KEPT`。
3. 原侧 E0 sha 清单 `$NFS/v75eval/e0-sha.txt` 已在 10-03 清理中删除：12.493 加 `ORIG_E0_SHA`，由 `artifacts/v7.5eval/input-manifest.json` 重建（剔除 `server.log`，每片恰 3 行）。
4. MME 权重路径是符号链接，`history_config.txt` 解析落空：在运行根建实体目录（`cp -al`），`ASSET mme-gl 18/18`。
5. MME 原侧客户端环境：原版 `eval.py` 无条件导入 `google.generativeai` 且依赖环境自带 `openpi_client`：新建 `orig-mme-client` 环境（12.494）并装入原版 `packages/openpi-client`；GL 计算节点 glibc 2.28 不认 `cryptography 50.0.2` 的 manylinux_2_34 轮子，换同版本 manylinux_2_28 轮子后 GL 上整链导入通过。
6. GL 系统 `python3` 为 3.6.8，原侧编排脚本里裸 `python3` 调 S7 新辅助脚本失败：任务脚本 `task_orig2.sh` 把执行副本 `.venv/bin`（3.11）放 PATH 最前。受影响的第 0 片：SimpleMemVLA 段在修复版任务中完成，MME 段单独补跑（`task_orig_mme.sh`），失败目录归档在 `gate2-orig-rerun/_failed-{py36,glibc}-s0/`，泄漏的 20 条预算预约已 release。
7. GL 上 imageio 自带 ffmpeg 目录没有 `ffprobe`，非 GroundSG 路线席位内重绘全部失败：建 `tools/ff/`（ffmpeg 为同一二进制的包装脚本、ffprobe 取 conda-forge 7.1.1），后续任务改用 `task_new*_v2.sh`；pp 第二档 s30 那片 48 局在本机重入补绘 `BACKFILL_RENDER ok=48 fail=0`。
8. 对比工具读 v7.5eval 历史逐局行时把被重评取代的 Vulkan reset 失败行当终态，SimpleMemVLA E0-O2 一度显示 63 局假翻转；12.496 修后与 v7.5eval 留档一致（两两 0 翻转）。
9. PonderPounce 原侧结果行（`pp_official_runner.py`，R1 不能改）与 SimpleMemVLA、MME 原版逐局行不带节点字段：合表副本按各片任务日志 `host=` 行补注 `node` 与 `node_source`；SimpleMemVLA 原版行不带 `attempt`，按 S7 适配器映射补注。补注副本在 `$R2/gate2-report/annot/`，原文件不动。

10. 第三档中止（用户指令）：先把 36 个未领取任务移到 `queue/held-gate3/` 并写 `queue/STOP`，再按精确步骤号 `scancel` 7 个 worker 步骤（63188711.6、63188712.4、63188713.1、63188714.1、63188715.1、63188716.1、63188719.0；全部 `EXIT_CODE=137` 退出，`squeue -s` 核对无残留步骤），7 个运行中任务移到 `held-gate3/interrupted__*`；登录节点上 3 个等待 PENDING job 的启动会话（p2-w534、p2-w535、p2-w720）按名精确 kill，删后 `tmux ls` 为空。
11. 资源释放（用户「只保留4快卡 最新的job 其他资源全部释放」）：`scancel` 63188711、63188712、63188713（RUNNING）与 63188720、63325534、63325535（PENDING）；保留剩余时长最长的 4 个 RUNNING 占位 job：63188714（gl1518）、63188715（gl1503）、63188716（gl1527）、63188719（gl1525），均空闲无工作步骤。已记入 `$N/gl-hold-logs/hold-jobs-sgeval-20261004.txt`。

## ⑦ 预算（中止时）

`BUDGET_ENFORCEMENT=PASS trajectories=2064/6366 resets=4908/141430 astra=0/2 shared_infra=0/50`（`BUDGET_DETAIL reserves=2084 committed=2056 released=20 open=8`；open 8 为中断时在跑的局）。账本只记经 S8 账本的 GL 任务与本机 smoke；Astra 本机 smoke 费用由独立守卫记账：1 局、0.3043 美元（5 美元硬上限）。第一档生成 177 + 定性重跑 8 条轨迹走生成侧预算参数，不在评估账本内。

## ⑧ 下一步（交用户决定）

- 是否续跑第三档（五模型共需约 3200 局，按 7 卡估约一天；现只保留 4 卡）。
- 是否删除中间产物：`$R2/gate2-report/traces/*` 只是符号链接；NFS 上 `$R2/media/` 体量未统计（席位内转码核对通过即删原始无损帧；pp 第二档 s30 那 48 局因重绘失败带 `--keep-raw`，原始帧约 5.3 GB 仍在）。
