# 评估汇总（dataset=ood）

- 生成时间：2026-10-07T08:50:42-0400
- manifest：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-03/inputs/manifests/ood86/manifest.json`；运行根：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-03/ood-new-memer-seed7`；本机视频根：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-03/reports/memer/vroot`
- 口径：每身份唯一权威终态取账本 `accept` 行的 `accepted_attempt_id`；迟到终态与废弃尝试单列、不入分数；分母固定为 manifest 身份数（86／模型）；执行步上限 1800，超过即判 FAIL。

```
EVAL_COVERAGE=PASS dataset=ood policy=groundsg:ground-sg-memer expected=86 missing=0 extra=0 duplicate=0 conflicting_terminal=0 error_final=0 no_frame_error=0 policy_seed=7
EVAL_REPORT=PASS dataset=ood policy=groundsg:ground-sg-memer count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=0 cap=1800 cap_mismatch=0 policy_seed=7
EVAL_VIDEOS=PASS dataset=ood policy=groundsg:ground-sg-memer expected=86 accepted=86 videos=86 no_frame_error=0 missing=0 decode_fail=0 policy_seed=7
```

## 总表

| 模型 | 分母 | 已定终态 | 成功 | 失败 | timeout | error | 缺失 | 冲突 | 全局微平均 | 任务宏平均 | 迟到 | 废弃尝试 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| groundsg:ground-sg-memer | 86 | 86 | 5 | 72 | 9 | 0 | 0 | 0 | 5.8% | 6.2% | 0 | 0 |

## 任务×档成功率（成功／分母；括号内 timeout／error／缺失）

| 格 | groundsg:ground-sg-memer |
|---|---:|
| BinFill@xhard1 | 0/2 0.0% (2/0/0) |
| BinFill@xhard2 | 0/2 0.0% (0/0/0) |
| ButtonUnmask@xhard1 | 1/2 50.0% (0/0/0) |
| ButtonUnmask@xhard2 | 1/2 50.0% (1/0/0) |
| ButtonUnmask@xhard3 | 0/2 0.0% (0/0/0) |
| ButtonUnmask@xhard4 | 0/2 0.0% (0/0/0) |
| ButtonUnmaskSwap@xhard1 | 0/2 0.0% (2/0/0) |
| ButtonUnmaskSwap@xhard2 | 0/2 0.0% (1/0/0) |
| InsertPeg@xhard4 | 0/2 0.0% (0/0/0) |
| MoveCube@xhard4 | 1/2 50.0% (1/0/0) |
| PatternLock@xhard1 | 0/2 0.0% (0/0/0) |
| PatternLock@xhard2 | 0/2 0.0% (0/0/0) |
| PatternLock@xhard3 | 0/2 0.0% (0/0/0) |
| PickHighlight@xhard1 | 0/2 0.0% (0/0/0) |
| PickHighlight@xhard2 | 0/2 0.0% (0/0/0) |
| PickXtimes@xhard1 | 0/2 0.0% (0/0/0) |
| PickXtimes@xhard2 | 0/2 0.0% (0/0/0) |
| PickXtimes@xhard3 | 0/2 0.0% (0/0/0) |
| RouteStick@xhard1 | 0/2 0.0% (0/0/0) |
| RouteStick@xhard2 | 0/2 0.0% (0/0/0) |
| RouteStick@xhard3 | 0/2 0.0% (0/0/0) |
| StopCube@xhard1 | 0/2 0.0% (0/0/0) |
| StopCube@xhard2 | 0/2 0.0% (0/0/0) |
| StopCube@xhard3 | 0/2 0.0% (0/0/0) |
| StopCube@xhard4 | 0/2 0.0% (0/0/0) |
| StopCube@xhard5 | 0/2 0.0% (0/0/0) |
| SwingXtimes@xhard1 | 0/2 0.0% (0/0/0) |
| SwingXtimes@xhard2 | 0/2 0.0% (0/0/0) |
| SwingXtimes@xhard3 | 0/2 0.0% (0/0/0) |
| SwingXtimes@xhard4 | 0/2 0.0% (0/0/0) |
| SwingXtimes@xhard5 | 0/2 0.0% (0/0/0) |
| VideoPlaceButton@xhard1 | 0/2 0.0% (0/0/0) |
| VideoPlaceButton@xhard2 | 0/2 0.0% (0/0/0) |
| VideoPlaceOrder@xhard1 | 0/2 0.0% (0/0/0) |
| VideoPlaceOrder@xhard2 | 0/2 0.0% (0/0/0) |
| VideoRepick@xhard1 | 0/2 0.0% (0/0/0) |
| VideoRepick@xhard2 | 0/2 0.0% (0/0/0) |
| VideoUnmask@xhard1 | 1/2 50.0% (0/0/0) |
| VideoUnmask@xhard2 | 0/2 0.0% (1/0/0) |
| VideoUnmask@xhard3 | 1/2 50.0% (0/0/0) |
| VideoUnmask@xhard4 | 0/2 0.0% (1/0/0) |
| VideoUnmaskSwap@xhard1 | 0/2 0.0% (0/0/0) |
| VideoUnmaskSwap@xhard2 | 0/2 0.0% (0/0/0) |

## 按任务（跨档）与按档（跨任务）

| 任务 | groundsg:ground-sg-memer |
|---|---:|
| BinFill | 0/4 0.0% |
| ButtonUnmask | 2/8 25.0% |
| ButtonUnmaskSwap | 0/4 0.0% |
| InsertPeg | 0/2 0.0% |
| MoveCube | 1/2 50.0% |
| PatternLock | 0/6 0.0% |
| PickHighlight | 0/4 0.0% |
| PickXtimes | 0/6 0.0% |
| RouteStick | 0/6 0.0% |
| StopCube | 0/10 0.0% |
| SwingXtimes | 0/10 0.0% |
| VideoPlaceButton | 0/4 0.0% |
| VideoPlaceOrder | 0/4 0.0% |
| VideoRepick | 0/4 0.0% |
| VideoUnmask | 2/8 25.0% |
| VideoUnmaskSwap | 0/4 0.0% |

| 档 | groundsg:ground-sg-memer |
|---|---:|
| xhard1 | 2/28 7.1% |
| xhard2 | 1/28 3.6% |
| xhard3 | 1/14 7.1% |
| xhard4 | 1/12 8.3% |
| xhard5 | 0/4 0.0% |

## 预算（逐席）

**groundsg:ground-sg-memer**：reset_claim 总数 176，尝试 88，infra 重试 0，额度提升 0 次。

| 席 | reset_claim | reset 额度 | 尝试 | infra 重试 | infra 重试额度 | infra 错误尝试 | 额度耗尽 | 运行阻塞 | 额度提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s50 | 12 | None | 6 | 0 | 1 | 0 | 0 | 0 | — |
| s51 | 8 | None | 4 | 0 | 1 | 0 | 0 | 0 | — |
| s53 | 40 | None | 20 | 0 | 1 | 0 | 0 | 0 | — |
| s54 | 40 | None | 20 | 0 | 1 | 0 | 0 | 0 | — |
| s55 | 38 | None | 19 | 0 | 1 | 0 | 0 | 0 | — |
| s56 | 38 | None | 19 | 0 | 1 | 0 | 0 | 0 | — |

## 迟到终态与废弃尝试（不入分数）

- 无

## 异常明细

- 无

视频索引见同目录 `video-index.jsonl`（每次尝试一行，含终态、迟到、错误尝试）。
