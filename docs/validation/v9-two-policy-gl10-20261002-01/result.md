# V9 双模型评估运行档案：result

计划：`docs/plans/1002-newtask-v9-movecube-region-800-plan.md`；起跑档案见同目录 `launch.md`。评估执行副本冻结在 `820ca142`（12.333）。汇总全文 `records/report.md`。

## 一、结论

2026-10-02 17:55 起跑、18:48 十席全部结束：V9 新生成的 80 局（MoveCube 1 任务 × 1 档 × 50 + InsertPeg 1 任务 × 1 档 × 30）两模型各取得唯一终态，无 error、无 infra 重试；执行步严格截断 1600。与 V8 逐字节相同的 720 局复用 V8 评估终态（按 (task, tier, seed, spec_sha256) 四元组、`reused.json` 为唯一复用集合），合成 800 局总表。

### 表 1：新 80 局（本轮实跑）

| 模型 | 格 | 局数 | 成功 | 失败 | timeout | 成功率 |
|---|---|---|---|---|---|---|
| SimpleMemVLA | MoveCube@xhard4 | 50 | 4 | 4 | 42 | 8% |
| SimpleMemVLA | InsertPeg@xhard4（新 30） | 30 | 7 | 9 | 14 | 23% |
| MME-VLA | MoveCube@xhard4 | 50 | 8 | 24 | 18 | 16% |
| MME-VLA | InsertPeg@xhard4（新 30） | 30 | 0 | 12 | 18 | 0% |
| **合计** | | 160 | SimpleMemVLA 11／80，MME-VLA 8／80 | | | |

### 表 2：800 局总表（720 复用 V8 + 80 新评）

| 模型 | 局数 | 成功 | 失败 | timeout | 成功率 | V8 对照（1070 局） |
|---|---|---|---|---|---|---|
| SimpleMemVLA | 800 | 178 | 471 | 151 | **22.3%** | 26.0% |
| MME-VLA | 800 | 39 | 678 | 83 | **4.9%** | 6.0% |

注：720 局复用的是 V8 于 2026-10-02 04:26～11:23 跑出的终态，新 80 局是 17:55～18:48 跑的；权重、tokenizer、客户端代码都钉在同一锁值，但两部分不是同一时刻的采样（MME-VLA 同入口重跑本就有翻转，V8 xhard0 两路线 18 局翻转，计划 R-7）。V8 对照的分母是 1070 局（每任务 50～80 局），与 V9 每任务 50 局的分母不同，只作量级参考。MoveCube 的 50 局区域与 V8 的 20 局不同，其成功率变化不能只归因于策略。

### 表 3：按任务（800 局）

| 任务 | 新／复用 | SimpleMemVLA | MME-VLA |
|---|---|---|---|
| BinFill | 复用 50 | 4/50（8%） | 0/50（0%） |
| ButtonUnmask | 复用 50 | 29/50（58%） | 1/50（2%） |
| ButtonUnmaskSwap | 复用 50 | 9/50（18%） | 2/50（4%） |
| InsertPeg | 复用 20 + 新 30 | 16/50（32%） | 0/50（0%） |
| MoveCube | 新 50 | 4/50（8%） | 8/50（16%） |
| PatternLock | 复用 50 | 12/50（24%） | 0/50（0%） |
| PickHighlight | 复用 50 | 0/50（0%） | 0/50（0%） |
| PickXtimes | 复用 50 | 0/50（0%） | 0/50（0%） |
| RouteStick | 复用 50 | 13/50（26%） | 0/50（0%） |
| StopCube | 复用 50 | 0/50（0%） | 0/50（0%） |
| SwingXtimes | 复用 50 | 12/50（24%） | 0/50（0%） |
| VideoPlaceButton | 复用 50 | 15/50（30%） | 12/50（24%） |
| VideoPlaceOrder | 复用 50 | 10/50（20%） | 10/50（20%） |
| VideoRepick | 复用 50 | 18/50（36%） | 2/50（4%） |
| VideoUnmask | 复用 50 | 33/50（66%） | 0/50（0%） |
| VideoUnmaskSwap | 复用 50 | 3/50（6%） | 4/50（8%） |

| 档 | 局数 | SimpleMemVLA | MME-VLA |
|---|---|---|---|
| xhard1 | 272 | 82（30.1%） | 18（6.6%） |
| xhard2 | 272 | 45（16.5%） | 13（4.8%） |
| xhard3 | 92 | 20（21.7%） | 0（0.0%） |
| xhard4 | 144 | 31（21.5%） | 8（5.6%） |
| xhard5 | 20 | 0（0.0%） | 0（0.0%） |

## 二、判定行

| 判定 | 原文 |
|---|---|
| 输入 | `EVAL_IDENTITY_EXPORT=PASS version=v9 episodes=992 expected=992 xhard0=192 cells=43 cell_mismatch=0 delivery_mismatch=0` |
| 清单 | `V9_EVAL_SHARDS=PASS shards=10 total=80 cells=2 reused=720 reused_sha256=a476b25b7927 missing=0 extra=0 duplicate=0 xhard0=0` |
| 资产 | `V9_EVAL_ASSETS=PASS assets=2`；`TOKENIZER_SHA=PASS` |
| 冒烟 | `V9_EVAL_SMOKE=PASS infra_errors=0 identity_errors=0 exec_steps_max<=1600` |
| 覆盖 | `V9_EVAL_COVERAGE=PASS policies=2 expected=80 missing=0 extra=0 duplicate=0 conflicting_terminal=0 error_final=0` |
| 报告 | `V9_EVAL_REPORT=PASS total=800 new=80 reused=720 count_mismatch=0 media_unexplained=0` |
| 视频 | `V9_EVAL_VIDEOS=PASS policies=2 expected=160 videos=160 missing=0 decode_fail=0 sha_mismatch=0 moved_record_absent=0` |
| 转码 | `V8_EVAL_TRANSCODE=PASS episodes=160 frame_mismatch=0 stream_len_mismatch=0 failed=0`（脚本行名沿用 V8，即计划的 `V9_EVAL_TRANSCODE=PASS expected=160 mp4=160 failed=0`） |
| 席位 | 十席 `V8_SEAT_DONE outcome=pass rc=0`，每席两模型各 `done=8 errors=0 infra=0`，`SEAT_REC_SYNC=PASS n=16 left=0` |

## 三、执行过程

- 十席于阶段 3 期间提交；17:55 起跑时 00～07 RUNNING、08／09 `AssocGrpCpuLimit` 排队，生成席释放后数分钟内全部 RUNNING，未触发切片接力。SimpleMemVLA 每席 8 局约 25～30 分钟，随后 MME-VLA 约 5～15 分钟；最晚席 07 于 18:48 结束。
- 视频：本机 `eval_video_mover.py --mode v8` 常驻搬运 160 段（`moved=160 bytes=30513503429 sha_mismatch=0`），stop-file 后退出，`--once` 对账 `pending=0`；NFS 运行根的结果、账本、日志与冒烟录像 rsync 到本机 `artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/nfs-records/`。
- 转码首跑在报告生成前启动，找不到 `report/video-index.jsonl` 失败（日志 `logs/transcode-try1.log`），报告生成后重跑 PASS。
- 预算：两模型各 8 局 × 10 席，每局 `reset_calls=2`（build + reset）：reset 2 模型 × 80 局 × 2 = 320，rollout 160；冒烟 reset 4、rollout 2；infra 重试 0。均在计划 §2.4.3 上限内。
- 释放：报告三行 PASS、视频搬完后按清单 `scancel` 十席（63117915、63117927～63117935，rc 均 0）。

## 四、归档与产物

- `records/report.md`：汇总全文（逐格、逐席）。
- 本机产物（不进 git）：`artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/{manifest,nfs-records,videos,report,site-media}`、`artifacts/v9-evaluation/inputs/`、`artifacts/v9-evaluation/logs/`。
- 站点：V9 独立站 `artifacts/newtask-v9/site/`，端口 8082（tmux `site-v9-8082`），`V9_SITE=PASS cells=59 missing=0 eval_reused=720 eval_new=80 eval_empty=0 port=8082`，见 `docs/validation/newtask-v9/result.md`。
