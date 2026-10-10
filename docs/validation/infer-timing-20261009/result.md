# 推理测速改造与验证：实测结果

七变体的单身份正式结果、六 GL 媒体恢复及最终速度表已经完成；用户取消本机 PP 后续，PP-on 加载失败与 Astra-on 下载中断原样保留。用户要求的 GL 八工作负载与本机 Astra 三尝试总耗时已核实，见第六节；报表通过不扩写成全部闸门或真实开关数值一致性通过。

## 一 一句话结论与指标速览

按动作块保存计时的实现已经合入；六个 GL 变体在 A40 上、Astra 在本机 Ada 上，各执行 VideoUnmask 的 **1 任务 × 1 档（xhard0）× 1 正式局**。SimpleMemVLA、Oracle、PonderPounce、MemER、Astra 的独立 `task_success=1/1`，FrameSamp、QwenVL 为正常失败 `0/1`，不为成功重试。SimpleMemVLA 的启动元数据为 `git_dirty=True`，单列探索性；其正常完成不构成全部正式局均从 clean HEAD 起跑的证明。

以下仅为本次单身份的描述性计时，单位毫秒。稳态按原聚合器排除前三块；单局样本不代表跨任务性能。A40 与本机 RTX 6000 Ada 分组，不能混算。

| 变体 | 原终态 | task_success | 实际步 | 动作块 | 首块 wall | 稳态 wall 均值 | 稳态 wall P95 | 稳态语言均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleMemVLA（探索性） | success | 1/1 | 262 | 17 | 2300.886 | 1909.781 | 1962.305 | 1704.020 |
| FrameSamp+Modulation | fail | 0/1 | 104 | 7 | 27951.985 | 252.512 | 261.228 | 0.000 |
| GroundSG Oracle | success | 1/1 | 301 | 19 | 323.265 | 169.543 | 173.672 | 0.000 |
| PonderPounce | success | 1/1 | 259 | 13 | 11049.631 | 1670.330 | 4642.007 | 1148.412 |
| GroundSG QwenVL | fail | 0/1 | 289 | 19 | 12910.518 | 11287.536 | 11536.660 | 11120.223 |
| GroundSG MemER | success | 1/1 | 326 | 21 | 15493.448 | 31216.715 | 33726.906 | 31047.383 |

Astra 正式在 **RTX 6000 Ada** 上成功，实际／结果步数 274，reset 2，动作块 18；规划 3 次、监视 17 次、复核 0 次。整体首推理 `server_infer_ms=11552.263` 含首次编译；排除前三块后的服务端纯推理均值为 **81.597933 毫秒**，客户端动作 RTT 均值 **82.719 毫秒**。以下两层来自同一正式局，不是两个模型：

| Astra 层（Ada） | 全部块 | 稳态块 | 稳态 wall 均值（毫秒） | 稳态语言均值（毫秒） | 稳态动作 RTT（毫秒） | 稳态纯 server infer（毫秒） |
| --- | --- | --- | --- | --- | --- | --- |
| 仅规划（有 planner、无 review） | 3 | 2 | 11021.3505 | 10916.9225 | 93.817 | 92.57 |
| 仅监视（无 planner／review） | 15 | 13 | 485.028462 | 399.860615 | 81.011615 | 79.909923 |

“仅规划”是报表层名，其中两个块也含监视；17 次监视调用不等于 15 个监视层块。规划层 wall 含规划器的 `SEND_INTERVAL=20` 等候与客户端路径，不能读作纯模型推理时延；规划稳态只有两个样本，不扩大为跨任务性能结论。最终表为七变体／八行（六 GL 行加 Astra 两层），114 个块；原文为 `SPEED_TABLE=PASS rows=8 decisions=114 violations=0 missing=0 checked=114 expected=114 missing_fields=0 layer_violations=0 layer_missing_fields=0 layer_checked=114 layer_expected=114`，见 [最终报表](records/report-final.json)。

六个结果共 96 个块，原结果的守恒证据分别为 `checked=expected`、`violations=0`、`missing_fields=0`。主会话六 GL 报表的原文判定为 `SPEED_TABLE=PASS rows=6 decisions=96 violations=0 missing=0 checked=96 expected=96 missing_fields=0 layer_violations=0 layer_missing_fields=0 layer_checked=96 layer_expected=96`（[报告快照](records/report-gl-six.json)）。FrameSamp、Oracle 的 `overhead_pct_p50` 分别为 39.4813%、25.9426%，高于计划 20% 的观察线，保留实测、不放宽判据。

这里的 overhead 是墙钟拆分中未被语言段／动作 RTT 解释的**残差比例**，包括原有 pack_buffer、预处理、客户端记录等路径；它不是新增审计或计时的额外性能成本，也不能单凭该值判新增计时实现失败。真实本机开关各只有首块、无稳态样本，不能从两独立运行耗时差异推出因果。

## 二 版本与代码状态

GL 固定源码为 `7e9e6e5288b683aadca4358b482f3789748e29c0`；FrameSamp、Oracle、PonderPounce、QwenVL-r1、MemER-r1 的实际进程记录均为 `git_dirty=False`。旧 SimpleMemVLA 同提交却为 `git_dirty=True`，因此保留探索性标注。

本机 SimpleMemVLA 审计开的原启动为 `dad678adc1ef096e5eccfd4d03a652e5aec357f2`，审计关为 `7e9e6e5288b683aadca4358b482f3789748e29c0`；原 FrameSamp 审计开同为 `7e9e6e5`。上述三份短冒烟保持原样，不重写或重跑。私有控制停止修复随后合入 `959e624c852a3bdbe4a47d3f580cfb9714f5805f`（1.61），本机后续以该 clean 锚点运行；模型代码与第三方输入保持原固定字节。

本目录留档基线为 `959e624c852a3bdbe4a47d3f580cfb9714f5805f`，记录写入发生在独立 `codex/infer-timing-results` 工作树。留档代理只读已有数据并核验视频，没有运行模型、仿真、GPU、API、Slurm 或依赖安装。

## 三 启动与配置还原

完整原定二十一组启动命令、权重、身份、费用入口与逐次用户决策保留在 [launch.md](launch.md)。该文末追加实际固定代码副本、语言 r1 恢复、停止修复和媒体后处理的实际口径；历史命令不覆盖。

所有模型侧配置与入口可从启动提交还原，例如 `git show 7e9e6e5288b683aadca4358b482f3789748e29c0:src/robomme_ood_eval/models/groundsg.py`，私有停止入口为 `git show 959e624c852a3bdbe4a47d3f580cfb9714f5805f:dev-scripts/checks/infer_timing_smoke.py`。QwenVL／MemER 的恢复只复用已有兼容环境并显式覆盖新源码 `PYTHONPATH`，不改依赖、模型配置或注意力实现；新输出名分别为 `infer-timing-gl-qwen-r1`、`infer-timing-gl-memer-r1`，`--infra-retries=0`、`--client-restarts=0`。

## 四 数据集与划分口径

统一身份为 `hard-verify / VideoUnmask / xhard0 / VideoUnmask_xhard0_560300`，`builder_episode=0`、`source_episode=3`、`seed=560300`。环境身份 seed 与策略 seed 分开：GL `policy_seed=7`，本机 `policy_seed=0`。身份单行文件指纹为 `48ced718820688e271b60e5d8679c05c621888428d78a47625ead5fe72d8a1c7`。

正式分母是六个 GL 变体各 **1 任务 × 1 档 × 1 局**，及已结束的 Astra **1 任务 × 1 档 × 1 正式局**；本机审计开／关各 **1 任务 × 1 档 × 1 次短冒烟**属于链路验证，不进入正式稳态速度统计。

用户追加「收尾后报告Great Lakes的所有任务和Astra本地的任务的耗时。」总耗时分母另定为 **GL 八个 workload step（六个最终完整局，加 QwenVL／MemER 原加载失败各一次）＋本机 Astra 三次尝试（on 中断、off 短、正式）**。它与七变体正式速度表的局数分母不同；既有占位 sleep、batch／extern、历史步骤、CPU 诊断和后续单独媒体恢复均不计入下表 workload 秒数。

## 五 关键参数与预算

本仓无训练超参变更。正式身份的 `max_steps=1300` 保留；短冒烟取得首合法动作块后只执行一步，首块前最多十六步。修复后的私有入口在有效非终态返回上标记 `truncated=True / status=timeout / single_chunk_smoke`，不设置正式步数上界的 `cap_hit`，不把任何 `AttributeError` 当通过。

原计划为七变体本机审计开／关短冒烟共十四次，六 GL 正式与 Astra 正式共七次，即二十一次计划尝试。最新用户取消 PP 本机后续，最终执行为：六 GL 变体各 **1 任务 × 1 档 × 1 正式局**，五个已做本机开关的非 Astra 变体各 **1 任务 × 1 档 × 2 次短冒烟**，PP-on **1 任务 × 1 档 × 1 次加载失败尝试**，Astra **1 任务 × 1 档 × 3 次尝试（on 中断，off／正式各一次）**，合计二十次尝试。PP-off 用户取消未启动，PP-on 与 Astra-on 各零真实 reset，已完成的两次 Astra 各 build/reset，两类总数分别为二十尝试、三十六次真实 reset。

累计硬限仍为 **27 次轨迹尝试、54 次 reset、Astra 3 次、5 美元**；未返还 PP-on 或 Astra-on 的尝试额度，不另建账本。计划第四稿原来的三美元文字作为历史保留，实际执行以用户后续明确五美元裁决为准。

阶段预算先为 17／32／0，Astra-on 中断后为 18 次预约／32 次真实 reset／1 Astra 预约。主会话依据独立零 build/reset 证据对 rid `d430cd2af124468182bac0626cc29195` 补记结算 `resets=0,status=interrupted,exit=143`，不是 release：原尝试仍占额度，该 rid 不留未结算残留；中断时费用账本零 API／内部 episodes 0／零美元。QwenVL／MemER 首次 context_load 失败在轨迹预约前，各零轨迹、零 reset。

off 与正式结束后的最终共享账本为 **20/27 次轨迹尝试、36/54 次真实 reset、Astra 3/3、基础设施恢复 0/6**；预约与结算各 20、release/open 均 0。原 `planned_first_tries=21` 保留，实际 first_started=20，用户取消 PP-off 后不为补满计划制造第 21 次。Astra 内部费用守卫登记 off／formal 两个 episode，on 在 preload 中断不登记；共享尝试账本仍把 on／off／formal 三次完整计入。

API 用量累计 **4 次请求、输入 32012／输出 68 token、cached 7972、reasoning 0、未知 usage 0**。按原费用配置（每百万 token 非缓存输入保守按 12.5 美元、缓存输入 1 美元、输出 50 美元）折算，账本合计 **0.311872 美元**，其中 off 一次 **0.1006875**、正式三次 **0.2111845**；这是配置单价的账本估算，不是提供商结算账单。原 5 美元上限不变，完整用量与逐请求值见 [费用账本](records/astra-cost-ledger.json)、[共享尝试账本](records/shared-budget-final.jsonl)。

原命令、退出码与逐字输出见 [budget-final.txt](records/budget-final.txt)。最终两行原文：

`BUDGET_DETAIL ledger=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/infer-timing-20261009/budget-ledger.jsonl reserves=20 committed=20 released=0 open=0 expired=0/0 reset_cap_exceeded=0 first_started=20/21 recovery=0/6 config=present`

`BUDGET_ENFORCEMENT=PASS trajectories=20/27 resets=36/54 astra=3/3 shared_infra=0/6`，`EXIT_CODE=0`。

## 六 硬件与耗时

GL 为 NVIDIA A40，持有作业与节点分别为 `63431430/gl1525`、`63431431/gl1512`、`63431432/gl1512`、`63431433/gl1513`；均为用户原有席位，本轮未申请或取消。GL 原产物位于 NFS，本机只通过 rsync 把六个已完成输出搬入主仓 `artifacts/`。

本机为两张 RTX 6000 Ada。非 Astra 沿原命令使用物理 GPU 1；Astra 双卡分工与费用权限保持用户授权口径。本稿不从 GPU 利用率或单步吞吐推断瓶颈。媒体补渲染为纯 CPU 后处理，其耗时不并入推理 wall。

### GL 所有本轮工作负载的启动至退出

以下全部为 **2026-10-09，America/Detroit（EDT，UTC−4）**。时间来自 sacct 的 workload step Start／End，耗时为 `ElapsedRaw` 整秒，包含该 step 内加载、预热、环境、当时媒体录制与清理。step 定位同时核对原命令、作业、节点、PID 与独立日志终点；没有把旧 Oracle `.0` 或 PP `.0/.1` 当成本轮。完整原行与正反排除证据保存在 [A](records/runner-A-record.json)、[B](records/runner-B-record.json)、[C](records/runner-C-record.json)、[D](records/runner-D-record.json)。

| GL 负载 | step | 启动 | 退出 | 整程秒／分秒 | Slurm 退出 | 任务字段 |
| --- | --- | --- | --- | --- | --- | --- |
| SimpleMemVLA（探索性） | 63431430.3 | 21:58:59 | 22:04:13 | 314／5分14秒 | 0:0 | 1/1 |
| QwenVL 初次加载失败 | 63431430.4 | 22:17:01 | 22:19:20 | 139／2分19秒 | 1:0 | 未评估，null |
| QwenVL-r1 | 63431430.5 | 22:34:38 | 22:40:20 | 342／5分42秒 | 0:0 | 0/1，正常失败 |
| FrameSamp | 63431431.2 | 22:12:07 | 22:15:52 | 225／3分45秒 | 0:0 | 0/1，正常失败 |
| MemER 初次加载失败 | 63431431.3 | 22:16:29 | 22:19:08 | 159／2分39秒 | 1:0 | 未评估，null |
| MemER-r1 | 63431431.4 | 22:34:33 | 22:48:21 | 828／13分48秒 | 0:0 | 1/1 |
| Oracle | 63431432.1 | 22:12:14 | 22:14:59 | 165／2分45秒 | 0:0 | 1/1 |
| PonderPounce | 63431433.2 | 22:12:15 | 22:14:31 | 136／2分16秒 | 0:0 | 1/1 |

八个 step **执行时间累计 2308 秒＝38分28秒**，其中六个最终完整负载累计 **2010 秒＝33分30秒**、两次加载失败累计 298 秒；由于并行，不能把相加值当用户等待时间。最早 21:58:59 到最后 22:48:21 的日历跨度为 **2962 秒＝49分22秒**，包含阶段间空档。上述量不包括原多日占位作业的 sleep，也不说明 GPU 利用率吃满。

### 本机 Astra 三次尝试的整程与内层区间

同日同一 EDT 时区，以本次 tee 新建日志的文件 birth 至最终 `EXIT_CODE` 写入后的 mtime 为已记录整程，包含加载、预热、环境、媒体、服务清理；不包含 tmux 创建至 tee 打开的未记录小段。原记录 `started_at` 比日志 birth 滞后约 0.124～0.127 秒，两者原值均保存在 [local-runner-record.json](records/local-runner-record.json)。`finished_at` 是稍后收集证据的观察时刻，不用于退出；没有单独 `policy.load()` 总时长字段，不用 SERVER_READY 拼造它。

| Astra 尝试 | 日志 birth | 退出 mtime | 整程秒 | 模型内部秒 | 外层 episode wall 秒 | result t_end−t_start 秒（含媒体） | 退出／结果 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| on 预加载中断 | 23:00:55.291791 | 23:04:13.112607 | 197.820816 | 未进入 | 未进入 | 无结果 | 143／未评估 |
| off 一步短冒烟 | 23:20:39.626729 | 23:22:03.321656 | 83.694927 | 15.827913 | 17.542 | 22.767046 | 0／控制 timeout，task_success=0 |
| 正式成功 | 23:23:12.416596 | 23:26:14.855438 | 182.438841 | 62.042169 | 63.901 | 121.266155 | 0／success，task_success=1 |

三个执行窗口累计 **463.954584 秒＝7分43.955秒**；首次 on 到正式退出全跨度 **1519.563647 秒＝25分19.564秒**，含资产准备／审批／等待，两者分开。模型内部 `astra_seconds`、外层 episode wall、含媒体的结果窗口、整程日志窗口是不同区间，不互相替代或相加；正式 **62.042169 秒不是整程 182.438841 秒**。编码累计 CPU 秒可并行，也不与墙钟直接相加。

## 七 运行过程行为（本仓无训练）

最终实现阶段的核心短测日志 [s5-core.summary.log](records/s5-core.summary.log) 原文为：`1444 passed, 6 skipped, 101 deselected, 22 warnings in 184.50s (0:03:04)`；退出码 0。资源守卫原文为 `TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=6`，六项跳过保留为未验证，不改写成全量全部已验。

私有停止修复在主会话整合后 CPU 定向测试为二十二项通过、0.17 秒；提交 `959e624c852a3bdbe4a47d3f580cfb9714f5805f` 保留该验证与独立合并前静态审查／合并后树复核通过的过程。真实 FrameSamp、SimpleMemVLA 驱动配合 CPU 假环境／假模型在审计开／关四种组合中均只执行一步、推理一次，`result.steps=exec_steps=trace.steps=1`、`task_success=0`；错误、空观测与自然终态保持原路径，Astra 正式入口不安装停止包装。具名行不从文字摘要另行伪造。

## 八 正式评估与本机短冒烟

六 GL 正式原结果保存在 `records/gl-*-result.json`；第一节的表是这些字段的高层导读，完整 chunks 与计时段不在正文复述。全部原结果的 `recorder_verify=PASS`，但 `video=null` 且 `video_error` 保留缺少 ffprobe 的原文。补渲染关联另存各 `gl-*-recovery.json`，不回写原结果。

六份恢复日志原文各有 `OFFICIAL_VIDEO=PASS`、`RAW_STREAMS=PASS`、`ARRAYS_TRACE=PASS`、`MEDIA_RECOVERY=PASS`、`ORIGINAL_BYTES=PASS` 与 `EXIT_CODE=0`，逐文件见 [来源清单](records/source-manifest.json)。恢复核对原始文件共 134 项（45 + 29 + 60），全部零变更；搬运文件共 206 项（99 + 35 + 72），两端 sha256／字节数及路径集合全等，NFS 源保留。原始 AV1 是有损帧，恢复记录明确 `mode=skipped-lossy-av1`，不宣称解码帧与编码前数组逐像素相等。

本机已完成的 FrameSamp-off、Oracle-on/off、QwenVL-on/off、MemER-on/off 有一步截断标记且 `error=null`；原 FrameSamp-on 为 `exec_steps=1、steps=2、AttributeError: None.copy`，原 SimpleMemVLA-on 的 `exec_steps=1、steps=0` 同属停止控制缺陷，原三份短测如实保留。PP-on 为启动基础设施失败（CUDA 驱动不兼容 `found version12080`），退出码 3，零实际 reset／步／块，`task_success=null`；不是正常任务失败，也不是任务成功 0/1。

Astra-on 在 959e624 起跑，ModelScope 路径绕过 HF 离线旗标，开始约 8.88 GB 基座下载；不是 HF 下载自行忽略离线配置。运行者只向本次 client 408888、guard 409172、VLA 409192 发送 TERM；`it-astra-on` 退出码 143。中断时 API `calls=[]`、费用 0 美元，真实 reset／动作块均为零，无任务结果，partial cache 保留。用户要求「astra跑完收尾」，只执行 off／正式各 **1 任务 × 1 档 × 1 次**；用户另明确「PP 本机不跑了」，PP-off 取消未启动，PP-on 恢复不执行，GL PonderPounce 正式成功不变。

Astra-off 正常结束：`EXIT_CODE=0`、`error=null`、实际／结果步数均 1、块数 1、reset 2、`task_success=0`、`recorder_verify=PASS`。首规划器 3564.368 毫秒，其中预算账本检查 9.859 毫秒，review 0；动作块 wall 15129.436 毫秒、动作 RTT 11549.222 毫秒，审计关 `server_infer_ms/gpu=null` 原样保存，不补值。守恒 `checked=expected=1、missing_fields=violations=0`；off 时累计一次请求、0.1006875 美元。VLA GPU1／monitor GPU0，guard 443615 与 VLA 443618 已退出。

唯一正式局 `it-astra-formal`（pane 447880）以 959e624 clean 源码／启动器 SHA3455 执行完毕，`EXIT_CODE=0,status=success,task_success=1,error=null`，274 实际步／18 块／两 reset。父组守恒 `checked=expected=18、missing_fields=violations=0`；语言累计 31317.59 毫秒，规划账本检查三次合计 13.737045 毫秒。原始 AV1 4:4:4 两路验证通过：front 解码 246、wrist 281、时间戳通过，dropped／编码错误／decode_mismatch 均 0。guard 448045 与 VLA 448047 已退出。

正式外层官方视频实际存在于主仓 `artifacts/infer-timing-formal-astra/rollouts/astra/hard-verify/seed0/videos/VideoUnmask_ep3_success_watch the video carefully, then pick up the container hiding the green cube, finally pick up another container hiding the red cube_xhard0.mp4`；留档代理只读系统 ffprobe 核验为 **341 帧、512×528、AV1／yuv420p／30fps**，文件 733711 字节、SHA256 `834133310c1003d4148c0e2e87fa294df78c7679d3d09d7508c75114699a66a9`。off 视频同格式 68 帧。判定原文 `ASTRA_OFFICIAL_VIDEO_READBACK=PASS videos=2 off_frames=68 formal_frames=341 codec=av1 pix_fmt=yuv420p fps=30` 与两份完整路径／指纹位于 [视频只读核验](records/astra-video-probes.json)，未重新渲染或复制视频。

六媒体完成判定原文：

- `MEDIA_RECOVERY=PASS model=smvla official_video=1 raw_streams=2 arrays_trace=1 originals=15 reset=0 trajectory=0 api=0 task_success=1`（[gl-smvla-media.summary.log](records/gl-smvla-media.summary.log)）。
- `MEDIA_RECOVERY=PASS model=framesamp official_video=1 raw_streams=2 arrays_trace=1 originals=15 reset=0 trajectory=0 api=0 task_success=0`（[gl-framesamp-media.summary.log](records/gl-framesamp-media.summary.log)）。
- `MEDIA_RECOVERY=PASS model=oracle official_video=1 raw_streams=2 arrays_trace=1 originals=29 reset=0 trajectory=0 api=0 task_success=1`（[gl-oracle-media.summary.log](records/gl-oracle-media.summary.log)）。
- `MEDIA_RECOVERY=PASS model=pp official_video=1 raw_streams=2 arrays_trace=1 originals=15 reset=0 trajectory=0 api=0 task_success=1`（[gl-pp-media.summary.log](records/gl-pp-media.summary.log)）。
- `MEDIA_RECOVERY=PASS model=qwen originals=30 task_success=0 status=fail reset=0 trajectory=0 api=0`（[gl-qwen-r1-media.summary.log](records/gl-qwen-r1-media.summary.log)）。
- `MEDIA_RECOVERY=PASS model=memer originals=30 task_success=1 status=success reset=0 trajectory=0 api=0`（[gl-memer-r1-media.summary.log](records/gl-memer-r1-media.summary.log)）。

Oracle 搬运判定原文为 `TRANSFER=PASS model=oracle files=35 bytes=4048009 mismatch=0 symlinks=0`；另外两组 transfer 原记录为 `pass=true`，各文件 manifest 均 `match=true`，不另编一条不存在的命名判定行。

## 九 用户决策记录

1. 「/data/hongzefu/RoboMME-benchmark-OOD-eval/docs/plans/1009-infer-timing-plan.html 开始实现 有问题问用户 但尽可能不要阻塞」。
2. 「gl」「gl已经占用的卡可以直接用」。
3. 「批准：最多 54 次 reset、27 次轨迹尝试」。
4. 「本局的冒烟是什么意思？是要跑完吗？不用跑完。」
5. Astra 先定「Astra 只做一次短冒烟和一次正式，共 2 次」，随后改为「astra同意做3次 放宽到5美元」。本稿保留顺序，以后者作为本轮执行上限；不改写历史第四稿的三美元口径。
6. 最新收尾裁决：「PP 本机不跑了」「astra跑完收尾」。取消本机 PP-on 恢复及 off，不创建新 PP 环境；Astra 保留已扣的 on 中断尝试，只执行既有 off／正式两个剩余尝试。
7. 追加耗时要求：「收尾后报告Great Lakes的所有任务和Astra本地的任务的耗时。」第六节按实际工作负载与日志区间给全表，不把 hold 睡眠或局内秒当整程。

## 十 计划外事件与处置

语言变体首次加载缺少 flash-attn，QwenVL／MemER 均尚未开始轨迹，原失败输出与日志保留。独立诊断证据原文为 `LANG_CLIENT_ROOT_CAUSE=PASS missing_flash_attn=1 failed_models=2 episodes_started=0`。恢复使用已有兼容环境、固定新源码根、全新 r1 输出与零自动重试，不降级注意力实现。

本机 PP 原 NFS 环境为 torch 2.14.0+cu130／transformers 5.13.1，而本机 driver 570.211.01 的 API 为 12080；已有本机客户端虽为 cu128，却使用 transformers 4.57.3，不具 Qwen3.5，不能直接替换完整 PP 环境。用户随后取消本机 PP，环境不建、依赖不安装、原失败尝试不退还；已准备的一次恢复入口提交 `32ab713` 仅留专属分支，不合入主分支或执行。

Astra 根因链为 `AstraPolicy.load → runner.Monitor → PtEngine → safe_snapshot_download → get_hub(None) → use_hf_hub(USE_HF 默认 0) → ModelScope master`；HF_HUB_OFFLINE 和 TRANSFORMERS_OFFLINE 不约束这条 ModelScope 路径。原 HF 缓存中一直存在同锚 `ebb281ec70b05090aa6165b016eac8ec08e71b17` 的完整基座，曾用于 split 成功运行。资产准备按授权把十四文件／8,887,292,732 字节复制为本轮 `artifacts/infer-timing-20261009/astra-monitor-base` 实体目录，零下载、零指纹或字节数不符、十四来源均匹配，旧权重缓存和 ModelScope partial 原样保留。完整实测见 [资产证据](records/astra-assets-evidence.json)。

判定原文：`ASTRA_BASE_COPY=PASS files=14 bytes=8887292732 missing=0 fingerprint_mismatches=0 symlinks=0 download_bytes=0`；`ASTRA_LOCAL_PATH_BRANCH=PASS local_paths=2 download_callbacks=0 download_context=0 torch_import=0 model_initialization=0 source_patch=0`。本机启动器仅新增 `--cfg astra_monitor_base=/data/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/infer-timing-20261009/astra-monitor-base`，SHA256 更新为 `3455d14e65ecccefb1f767c21809d56e4d587e7586d6dab797ceca40c8d57551`，预算、费用、GPU、密钥逻辑及其余六模型命令不变。资产身份保证复用旧 split 同一公开 revision，不扩大为训练原基座同源；Astra 实际 GPU 链路另由本次 off 与正式终局独立证明，不用本地分支夹具替代真实运行。

停止控制原先通过第二次 step 抛异常，FrameSamp 包装器返回空观测后产生属性错误，SimpleMemVLA 未交付 consumed。只改私有入口返回明确控制截断；旧三份短结果不覆盖、不复跑。正式 GL 命令与媒体渲染库仍固定在原 7e9e6e5。

**真实独立审计开／关对比另有数值差异，不能被 CPU OBS_EQ 通过抹掉。** SimpleMemVLA 首 `model_action 30×8 float32` 与 `exec_action 8 float64` 逐位相同，首请求 hash 相同。FrameSamp 首 `20×8 float64` 的 160 项有 116 项不同，最大绝对差 `0.00158659747314438`；执行动作 8 项有 7 项不同，最大差 `0.0004883030233977514`。Oracle 仅保存执行动作 8 项，其中 4 项不同，最大差 `0.00020051194312420417`。QwenVL 执行动作 8 项有 6 项不同，最大差以实际 NPZ 读回为 `0.0009763676757810202`；此前文字回执为 `0.0009763676754810202`，约差 3e-13，原文字与读回值并存注明校准。Oracle／QwenVL 完整 `20×8` 动作没有保存，摘要不同不等于逐数核验。

后三者输入 hash、seed、checkpoint 与 QwenVL 子目标相同；两次独立运行差异的因果尚未验证。既有 CPU OBS_EQ 是假输入／真实包装器的审计关字节对照，验证范围与上述真实独立实验不同。只读 NumPy 复核已经复现四对请求指纹与上述全部差异，原文判定 `REAL_AUDIT_READBACK=PASS pairs=4 input_infer_hash_equal=4 numeric_differences_preserved=3 new_model_calls=0`，每源文件 SHA256／字节数、完整请求 hash 与数组 hash 位于 [真实独立对比](records/real-audit-comparison.json)。保留零容差反例，不放宽阈值、不追加实验、不宣称真实数值逐位不变；不再扩展到未具备配对的 PP／Astra 或额外 MemER 验证。

## 十一 结论与下一步

七变体各 **1 任务 × 1 档 × 1 正式局**已结束，最终报表七变体／八行／114 个计时块，累计账本、媒体与用户要求的十一行整程耗时均有源证据；SimpleMemVLA 探索性、两项较高墙钟残差、真实审计开／关差异、本机 PP 用户取消及 Astra-on 中断继续保留，不宣称全部闸门通过。所有本轮运行／媒体 tmux 与精确 worker 已结束，原用户三个 site 会话和四个既有 GL hold 保留，没有取消它们；留档代理不扩大运行或预算。

## 十二 归档文件清单

- [launch.md](launch.md)：版本、完整命令、参数、用户授权、资源与精确会话清单。
- [source-manifest.json](records/source-manifest.json)：每份快照的主源绝对路径、sha256、字节数。
- `records/gl-*-result.json`：六份原始正式结果，含原 video 缺失与全部动作块计时。
- `records/gl-*-process.json`：实际启动提交、dirty 标志、依赖与硬件元数据，保留 SimpleMemVLA 探索性来源。
- [report-gl-six.json](records/report-gl-six.json)：主会话已产出的六 GL 速度表及原文具名判定；不是未完成的七变体最终表。
- `records/gl-*-recovery.json`、`records/gl-*-media.summary.log`：独立媒体恢复关联与原文判定行。
- `records/transfer-{three,oracle,language}-record.json`：206 个文件的两端指纹与搬运证据。
- [astra-assets-evidence.json](records/astra-assets-evidence.json)：原同锚基座、十四文件复制、零下载、离线路径分支及唯一 cfg 改动的实测快照；不包含权重或密钥。
- [real-audit-comparison.json](records/real-audit-comparison.json)：四模型独立开关对比的输入与数组指纹、零容差差异、原始来源及 Qwen 文本回执校准。
- `records/s*-core.summary.log`、[trace-schema.summary.log](records/trace-schema.summary.log)：各阶段核心短测与轨迹契约原文。
- `records/astra-{off,formal}-result.json`、对应 `results.jsonl` 与 `summary.json`：本机已结束短／正式两次实测；Astra-on 没有任务结果，保留 interrupted 日志。
- `records/astra-*.summary.log`：[on](records/astra-on.summary.log)／[off](records/astra-off.summary.log)／[正式](records/astra-formal.summary.log)日志，保留嵌在 tqdm 行中的退出码 143／0／0。
- `records/astra-cost-{ledger,state,reservations}.json`、[shared-budget-final.jsonl](records/shared-budget-final.jsonl)：同一费用／共享预算账本的最终累计与登记。
- [report-final.json](records/report-final.json)、[astra-video-probes.json](records/astra-video-probes.json)：七变体分层速度报表与两份本机已生成官方视频的只读核验。
- `records/runner-{A,B,C,D}-record.json`、[local-runner-record.json](records/local-runner-record.json)：最新完整工作负载映射、sacct 原行与 Astra 日志 birth／mtime、原观察时间及内层区间。
- [budget-final.txt](records/budget-final.txt)：主会话实际预算报告命令、完整 stdout、退出码 0。

不归档配置、启动脚本、权重、视频、原始大数组或重复源码。
