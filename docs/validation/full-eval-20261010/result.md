# 分阶段全量 OOD 评估：结果

S0 最小冒烟与媒体补全已通过，正式 A～D 尚未完成。以下数字仅为单任务冒烟，不代表十四任务成绩。

## 首批单局结果

四模型各 1 任务（VideoUnmask）× 1 档（xhard1）× 1 局，模型种子 7；每局 1 次轨迹尝试、2 次 reset，没有重试。独立任务成功字段如下；MemER 的正常失败保留，不挑成功回合重跑。

| 模型 | task_success | 实际步数 | 执行段墙钟秒 | result 起止墙钟秒 | recorder_verify |
|---|---:|---:|---:|---:|---|
| PonderPounce | 1 | 276 | 49.190 | 61.774 | PASS |
| SimpleMemVLA | 1 | 269 | 43.866 | 50.116 | PASS |
| GroundSG+QwenVL | 1 | 440 | 377.088 | 384.586 | PASS |
| GroundSG+MemER | 0 | 257 | 450.924 | 457.569 | PASS |
| FrameSamp+Modulation | 0 | 106 | 35.604 | 76.422 | PASS |
| GroundSG+Oracle | 0 | 205 | 34.593 | 106.062 | PASS |

执行段来自 `timing.episode_wall_s`，result 起止来自 `t_end-t_start`；不含此前 Policy.load。首批官方网页视频失败后单独补全，后处理耗时不混入这些原始数字。原四份 `video=null` 与 `video_error=ffprobe not found` 保留。FrameSamp／Oracle 冒烟在修正后的媒体环境中运行，官方视频字段正常。合计 3 成功、3 正常失败；完整 smoke 流程通过不等于任务全成功。

## 计划外事件：GL 媒体工具缺失

GL 默认 PATH 没有 ffprobe。系统模块 `ffmpeg/7.1.0` 的 native AV1 解码器读 PP 原始视频时失败，`nb_read_frames=N/A`；命令返回 0 也不能判有效帧数。未把该探针算作通过、未降级 codec 或重跑模型。

从 [固定发布来源](https://johnvansickle.com/ffmpeg/) 获取 7.0.2 静态工具，压缩包 41888096 字节；发布方 MD5 `7fa72b652e19bf84c9461e332ea1cdf3` 匹配，归档 SHA256 为 `abda8d77ce8309141f83ab8edf0596834087c52467f6badf376a6a2a4c87cf67`。对应上游 n7.0.2 源码锚点 `e3a61e91030696348b56361bdf80ea358aef4a19`，两个实际二进制的完整指纹与自校验清单在 [媒体资产记录](records/media-assets.json)。后续运行载体读取该清单，并与固定清单指纹 `c4801fc13264422f9c512ef5be8cbf21c4ab32ca9c19f67aff1f38a2c365081d` 核对；每次起跑做 cheap 首尾／长度检查，首次恢复做 full SHA256，边界随清单声明。

在既有作业 63431433 中运行静态 ffprobe，对 PP front.mkv 得到 `codec_name=av1 pix_fmt=yuv444p nb_read_frames=248`，退出 0。没有改变模型环境、评估源码或原始结果。四局视频恢复由 63431431 的 CPU 后处理执行，tmux `fe-smoke-media-63431431`，2026-10-10 00:54:26 EDT 起；日志 `logs/fe-smoke-media-63431431.log`。只读取已有原始帧、动作和 trace，输出每局 `recovered-video/`，逐文件核对原始数据不变，reset／轨迹／模型请求均为零。

## 复用与预算

smoke 的六个身份是 A 清单的子集；完整验收后将相同字节的已接受结果与媒体复用进 seed7 正式队列，不重跑六局。共享账本的幂等 token 是 `route|key|attempt`，不含输出目录；直接在另一个目录重新运行这六局会冲突，不能用新账本绕过。因此原计划的 426 首试保留上限不变，A 唯一首试实际为 420，正式身份不变。

初次账本核对：`BUDGET_DETAIL reserves=6 committed=4 released=0 open=2 expired=0/420 first_started=6/426 recovery=0/4254 config=present`；`BUDGET_ENFORCEMENT=PASS trajectories=6/4680 resets=12/9360 astra=0/2 shared_infra=0/420`。open=2 是两条仍在执行的 smoke，不能作为阶段收尾通过；Astra 实际未启动、费用为零。

## 六模型完整冒烟验收

媒体恢复：`MEDIA_RECOVERY_SUMMARY=PASS models=4 reset=0 trajectory=0`、退出 0，原文件共 62 个的字节数与 SHA256 全部保持不变；四个视频帧数分别 343／336／507／324。FrameSamp、Oracle 正常生成的视频为 173／272 帧。全部视频为 AV1/yuv420p/30fps，原始 front/wrist 为 AV1/yuv444p，完整解码帧数逐流与 recorder 计数吻合。

主会话在 GL 既有 63431431 中运行汇总检查，退出 0：`SMOKE=PASS models=6 results=6 videos=6`；`TRACE_ARRAYS=PASS episodes=6 attempted_steps_missing=0 observed_state_missing=0 tampered=0 dtype_mixed=0 unreadable=0 summary_mismatch=0 extra_action_keys=0`。原始输出在 [冒烟检查日志](records/smoke-check.summary.log)，逐模型视频和源结果指纹在 [冒烟报告](records/smoke-report.json)，数组逐局证据在 [数组检查](records/trace-arrays-report.json)，恢复证据在 [媒体恢复汇总](records/media-recovery-summary.json)。规模仍为六模型 × 1 任务（VideoUnmask）× 1 档（xhard1）× 1 局 = 6 次轨迹、12 次 reset，零重试。

阶段 A 复用：只拷贝已结束 smoke 的 `rollouts/` 与 `queue/` 至 `seed7/`，源文件保留。`SMOKE_REUSE=PASS models=6 accepted=6 files=126 reset=0 trajectory=0`，126 个文件逐字节 SHA256 相同；阶段报表读取复制后的六个权威 accepted 与 result/追加行，全部一致。该次只核 1 局身份，不把 `STAGE_TIMING=PASS ... episodes=6` 解释为完整 A 的 420 局通过。

收尾账本：`BUDGET_DETAIL reserves=6 committed=6 released=0 open=0 expired=0/420 first_started=6/426 recovery=0/4254 config=present`；`BUDGET_ENFORCEMENT=PASS trajectories=6/4680 resets=12/9360 astra=0/2 shared_infra=0/420`。六个已接受身份属于 A；之后六模型各补 69 局，其余尝试继续使用同一本 A 账本。

## 当前状态与下一步

第一批已完成产物按 accepted 白名单搬回本机，不同步活动 `rollouts/` 整层，也不复制未完成的临时视频。84 个已接受身份共 1353 文件、796827998 字节，完整 raw、对应官方视频及恢复证明、accepted 标记逐文件 SHA256 与字节数一致；各模型 `results.jsonl` 按稳定历史前缀复制。执行入口 `UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 280s uv run --no-sync python artifacts/full-eval-20261010/copy-completed.py`，退出 0：`COMPLETED_COPY=PASS accepted=84 files=1353 reset=0 trajectory=0`。落点 `artifacts/full-eval-20261010/seed7/`，源暂保留给阶段媒体全验；该复制不作为完整 A 通过。详细清单在 [第一批复制证据](records/completed-copy-first.json)。

控制器 03:18:11 EDT 从干净 r3 提交 `0c6f8a3a552876020611d4d654a41adb3accba65` 实际接手；NFS 控制器 39 项夹具再次通过、0.62 秒、资源全零。实际原 A 的单身份 r2 正常退出后，控制器自动创建下一个 r3 单身份、确认源码与二十次上限，再连续跨任务推进。04:10 左右 A 已接受数为 PP 38／70、QwenVL 22／70、MemER 5／70、SimpleMemVLA 随下一完成达到 11／70，FrameSamp／Oracle 各 1／70 复用局；这是进度快照，完整阶段仍未完成。每模型的 70 是十四任务 × 跨档轮转各五局；六模型合计 420 个 A 身份。会话序号含零尝试跳过，不作为 native 局数。

连续四席调度器经固定提交独立审查通过，并在生产结算代码生成的真实 JSON 格式夹具中核对环境创建失败恢复。完整核心短测 `timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q` 为 1615 项通过、6 项跳过、101 项慢测排除，186.44 秒、退出 0；`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=6`。GreatLakes 既有 63431432 上显式绑定候选 r3 评估与 benchmark 源码的真实 NFS 夹具，20 项到期半局正负例通过、11.61 秒，资源守卫全零；未运行仿真。先前首次 NFS 夹具未显式绑定评估源码，随后按正确路径重测，本结论引用第二次结果。

调度范围仍是六变体 A／B／C／D，四个原占位作业及其到期后 48 小时接替；每阶段同一本预算、每身份至多 20 次，正常任务失败不重跑。接手精确会话与日志，不重复启动已有 worker；只在 Slurm 明确 TIMEOUT 后申请对应席位，提交状态不明拒绝重复。阶段报告在占位作业中独立运行，真实报告的六模型、种子、身份数、cap、全解码媒体和零计数字段全部核验后才接续。程序接续与代理自动唤醒是不同能力；本轮宿主未提供可注册的任务自动唤醒接口，主会话仍保持活动处理事件。

第一原失败身份 `BinFill_xhard1_16400100` 第三次在独立进程中正常完成：`task_success=0 status=fail exec_steps=1650 infra=False`，已接受且不重跑；加载 238.569 秒、执行 297.204 秒、结果起止 724.025 秒，视频错误为空、录像器通过。主会话只读全解码得到 `RESUME_MEDIA=PASS model=smvla identities=1 reset=0 trajectory=0`，1651 帧与 trace 一致；这是 1 任务（BinFill）× 1 档（xhard1）× 1 局的恢复验证，不代表剩余身份完成。原失败 a2 四文件 SHA256 保持，旧模型追加结果日志 96 行、208658 字节前缀保持 SHA256 `f9d324db0f152d53d9f3f0127f33e59245fd80ebb740a3109e7c310e96910cc2`。

私有归档另补真实到期半局的恢复路径：缺失 `result.json` 时，仅领取、旧本地账本、共享重试与预约记录、Slurm 到期证据和完整元数据全部相符才允许独占复制，记录 `incomplete_expired` 与旧账本整体／逐证据行 SHA256；不把半局接受或计分。trace 已有正常 success／fail／timeout 终态则拒绝重跑，保留后处理恢复窗口。账本路线含模型种子，模型 trace 路线无种子，两者按固定模型规范精确绑定，拒绝原侧／其他模型／缺路线。独立审查通过；主树归档、预算及契约定向 88 项测试通过、11.01 秒，资源守卫全部为零。新代码不写入仍在运行的 r2 执行副本。

阶段报表补齐严格步数上限与真实媒体入口，媒体未开启时明确标为未验证；普通视频调用原验收函数，恢复视频绑定原结果、全部原始文件、trace／meta 身份、渲染清单与全解码帧数。真实六局读回发现恢复证明使用 GroundSG 完整策略标签，而结果使用基础模型名；修正为同时核对外层标签、基础模型与变体，不修改任何原结果。最终六模型 × 1 任务（VideoUnmask）× 1 档（xhard1）× 1 局 = 6 局读回退出 0：六条 `EVAL_REPORT=PASS ... cap_mismatch=0 exec_over_cap=0`；`OFFICIAL_MEDIA_INPUTS=PASS stage=A total=6 fail=0`；`OFFICIAL_MEDIA=PASS stage=A total=6 fail=0`。这是冒烟白名单验收，不是完整 A 成绩。报表及契约定向 83 项通过、4.60 秒，独立审查通过；新增检查零 reset、零轨迹请求。

SimpleMemVLA 正式 A 首个新身份正常截断，已接受且不重跑；随后 10 任务 × 各任务不同难度格（31 格，逐格见失败记录）× 合计 47 身份 × 每身份 2 次尝试 = 94 次环境创建失败，均为 Vulkan `ErrorInitializationFailed`、零执行步。该席位新增 95 次轨迹尝试、计量 96 次 reset。完整逐格乘式、结果与账本见 [失败快照](records/smvla-env-build-failure/summary.json)。原始账本保留，不把基础设施故障当任务失败成绩。已精确停止本轮 `fe-A-smvla-63431432`，另三个正式席位继续，四个占位作业均保留。

用户追加原话：「把现在的尝试上线提高十倍，不要有任何的阻塞，一路跑到底。」因此每身份总尝试由 2 增至 20，`--infra-retries 19`；各阶段已提高十倍的总预算不再重建，正常已接受失败与超时不重跑。新增私有席位护栏在环境创建失败时先保存结果、结算预算再退出；不修改评估 `src/` 或模型子模块。SimpleMemVLA 后续用每局独立进程隔离状态，原服务端每局重置种子与缓存的语义保持，额外加载耗时单列；具体 Vulkan 根因仍未证实。

真实 NFS 不支持该次 `renameat2(RENAME_NOREPLACE)`，返回 EINVAL；目录即使指定 0700 也继承共享 ACL，不能把权限当隔离保证。最终方案使用独占容器与文件创建、流式复制、逐文件读回 SHA256、源清单再次核验，失败时保留原目录与半归档证据。GL 既有 63431432 中实际调用待部署函数的夹具退出 0：`ARCHIVE_FS=PASS actual_function=1 copy_verified=1 source_preserved=1 conflict_rejected=1 reset=0 trajectory=0`，见 [真实共享盘验证](records/smvla-env-build-failure/archive-filesystem-report-r2.json)。同身份锁约定保证唯一写者，共享 ACL 并非权限沙箱。

私有恢复逻辑定向 67 项测试通过；整合前核心短测 1507 项通过、6 项跳过、101 项慢测排除，187.86 秒、退出 0，资源守卫未触发 reset、GPU、权重或网络。最终复制修订再次定向验证，并独立固定提交审查通过。下一步在新冻结执行副本恢复原失败身份，保留三个健康席位当前执行版本。

六变体 smoke 完整媒体验收通过，自动续行 A；某模型 A 清单完成即按计划滚动进入 B，阶段完成只报告不等待放行。之后种子 0、42，同一四席动态队列；到期后才申请接替。正式成绩、阶段耗时、三种子汇总与最终具名验收仍未验证。

## 旧两次上限席位的接续修订

PonderPounce 的 1 任务（VideoPlaceOrder）× 1 档（xhard1）× 1 身份 × 2 次尝试，均在执行 1660 步后遇到同一服务端上下文错误：长度 16439 超过上限 16384，`infra_reason=pp_server_error`。这沿原计划既有裁决记基础设施失败，未接受、未计入正常失败成绩；模型上下文、评估步数与失败分类均未改。两次结果还报告视频估计内存 6343 MiB 超过默认 6144 MiB，不能据此宣称媒体通过。第二次原结果逐字节快照见 [原结果](records/pp-context-overflow/result-a2.json)：`PP_FAILURE_SNAPSHOT=PASS attempt=2 bytes=64231 sha256=34f9662f242c70968c3f8e09e10d4dd9f90514626e659df6c1c2c9933283f13e`；未将已被旧入口覆盖的首次 raw 目录宣称为完整保存。

三个健康旧席位仍使用起跑时的两次上限。私有控制器仅对其 `EXIT_INCOMPLETE=6` 新增核证：旧 `RUN_PLAN max_attempts=2`、实际缺失集合、原领取与结果、本地结束及共享预算预约／重试／结算全部一致，最后尝试小于 20 且预算充足，才按原 r3 入口与同一账本接续已批准的二十次上限。真正二十次耗尽、正常终态漏接受、活领取、未知退出码及未结算仍拒绝，不放宽最终验收。

独立静态审查发现首稿把 GroundSG 队列标签误当预算路线；已修为生产 `policy_route` 的 `groundsg/<variant>/seed<n>/new`，夹具直接调用该生产函数。保留两个子提交 `3f2eb070d52db10f20ea0b9224709ce068890a1b`、`cf3c3962214652f5b24814ba00199b1b4fdf74af`；新固定提交审查 `STATIC_REVIEW=PASS findings=0`。生产队列、结束与预算序列化夹具的 68 项定向测试通过、0.58 秒，资源守卫全零。实际控制器接手与下一次真实重试另记，当前仍未完成阶段 A。

主树整合后的核心短测 `UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q` 退出 0，1644 项通过、6 项跳过、101 项慢测排除，187.69 秒；`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=6`。这是控制器与核心软件验证，不是新增仿真，也不是阶段成绩验收。

### 首次真实退出与剩余清单竞态

PP 旧席位于 07:46:44 EDT 结束，`RUN_SUMMARY total=70 accepted=69 running_elsewhere=0 missing=1 episodes_run=70`，`RUN_INCOMPLETE first=ood:VideoPlaceOrder_xhard1_17100000`，退出 6。控制器随后记录「旧上限日志与实际缺失集合不符」并退出 1；没有建立 PP 后继。其它活动工作进程继续，原日志及 `report_failed=1` 保留，不能把这次接续写成通过。

主会话重新只读核对当前缺失集合与同一原日志、领取、本地结果和共享预算，守卫通过且账本字节不变：`LEGACY_EVIDENCE_NOW=PASS attempts=2 missing=1 budget_unchanged=1`，见 [真实失败与重核证据](records/controller-r4-failure.json)。源码先读剩余清单、后检查进程结束，推断其间最后一局被接受造成快照过期；停止瞬间的旧集合没有被输出，未宣称观察到其具体内容。使用生产 `_end` 在两步之间接受正常失败局的夹具可复现该问题。

修订只在确认 worker 已结束后，再刷新该模型剩余清单，随后执行原严格核证与调度；不修改日志数字、结果、预算、模型或验收条件。子提交 `3d0a5e0b4319f8599f00845772bd20347bb0a096` 独立静态审查通过；PP 只接续原基础设施身份，SMVLA 已完成身份不再被空启动，真实失败计数保留。主树 `uv run --no-sync python -m pytest -q tests/test_continue_full_eval.py tests/pipeline/eval/test_budget_ledger.py` 退出 0，95 项通过、5.92 秒；`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。新监督器接手与真实第三次重试仍另记。

07:54:06 EDT 新 r5 监督器已实际接手；随后真实 `legacy_retry_resume` 核原 PP 两次失败，原队列只领取缺失身份第三次，`RUN_PLAN claimable=1 max_attempts=20`，原第二次的 12 文件归档结果哈希保持。`LEGACY_RETRY_RESUME=PASS claimable=1 attempt=3 max_attempts=20`，见 [恢复回执](records/controller-r5-start.json)。MemER／QwenVL 原 worker 保留，SimpleMemVLA 同步进入下一未接受身份，四席恢复执行；PP 第三次成绩尚未产生，不能将接续通过当作该身份完成。

## 实际二十次耗尽与局墙钟退出

PP 的 1 任务（VideoPlaceOrder）× 1 档（xhard1）× 1 身份 × 20 次尝试 = 20 次轨迹、每次两次 reset 共 40 次 reset，全部为同一上下文错误，最后仍 `exec_steps=1660 task_success=0 infra=True`。10:19:10 EDT 恢复席位结束：`RUN_SUMMARY total=70 accepted=69 missing=1 episodes_run=18`，两条同值 `EXIT_CODE=6`；其中本席位执行 a3～a20，a1／a2 来自老席位。共享账本此身份 20 次预约、20 次基础设施失败结算、无未结算项，未产生 a21。原结果快照见 [第二十次](records/pp-context-overflow/result-a20.json)。此前旧入口覆盖了首次 raw；新私有入口完整保留 a2～a19，最终 a20 留在当前 raw，不宣称首次 raw 全量仍在。

锁定 checkpoint `sg-eval/ckpt/pp/ponderpounce-9b-robomme/config.json` 的 `s2.max_context_tokens=16384`。上游 `SoftS2SessionContext._append` 在新上下文超过 `effective_context_cap` 时拒绝追加，`model/loading.py::_apply_inference_opts` 强制静态 KV cache 长度不超过模型上限；单调缓存参数不能解决。改变上下文配置会改变后续推理可见历史，原计划口径 8 则把此错误裁定为基础设施重试，两者均未擅改。

Qwen 的 1 任务（VideoPlaceOrder）× 1 档（xhard2）× 1 身份 × 1 次尝试，在 1056 执行步达到 1800 秒局墙钟限制，于 10:07:14 EDT 退出 75。原 `result` 是 `status=fail infra_reason=episode_wall`，本地账本是 `status=error infra_reason=watchdog_exit`，领取为 `infra_timeout/75` 且没有 `final/accepted` 字段；三者按生产 `SeatRunner._hard_exit` 契约结算完整，不能当成正常 1800 步截断。该路线原 A 身份已接受 59／70，待恢复 11，其中此身份重试、另 10 首次执行；本轮新增 59 次尝试、118 次 reset，另含复用冒烟的一次预约与两次 reset。原 [超时结果](records/qwen-wall-timeout/result-a1.json) 保留。

私有控制器新增严格结算恢复与阻塞隔离：Qwen 只在真实 1800 秒退出证据、身份、领取、本地／共享结算及预算都通过且尝试小于 20 时恢复，同一 r3 入口和原时间限制；PP 只隔离本轮 A 的上述二十次耗尽身份，保存原缺失、不接受、不评分，空席可执行其它已批准 A 模型。阶段 A 的完整报告与进入 B 仍要求原全部身份通过，未放宽闸门。独立固定提交审查通过；主树控制器与预算定向 130 项通过、14.05 秒、退出 0，`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。主会话对两个真实 helper 只读重核均通过，预算字节不变，见 [核证快照](records/settled-failures-before-r6.json)。

清单刷新夹具的通过不代表 NFS 上所有空跳过已消除：实跑 `fe-controller-A-smvla-2-47` 再次选到刚接受的 key，但 `RUN_PLAN claimable=0`、`RUN_SUMMARY episodes_run=0` 后退出，未新增 reset 或轨迹，随后进入新身份。具体读可见性原因未证，不宣称刷新已彻底解决此现象。
