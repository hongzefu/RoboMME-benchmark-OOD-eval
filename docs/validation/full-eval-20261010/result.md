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

六变体 smoke 完整媒体验收通过，自动续行 A；某模型 A 清单完成即按计划滚动进入 B，阶段完成只报告不等待放行。之后种子 0、42，同一四席动态队列；到期后才申请接替。正式成绩、阶段耗时、三种子汇总与最终具名验收仍未验证。
