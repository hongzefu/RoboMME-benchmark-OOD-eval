# 评估汇总（dataset=test-hard）（中途进度）

- 生成时间：2026-10-06T13:21:08-0400
- manifest：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/inputs/manifests/qwenvl-gate3.json`；运行根：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/gate3-new-qwenvl`；本机视频根：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/media`
- 口径：每身份唯一权威终态取账本 `accept` 行的 `accepted_attempt_id`；迟到终态与废弃尝试单列、不入分数；分母固定为 manifest 身份数（800／模型）；执行步上限 1600，超过即判 FAIL。中途进度下判定行一律 FAIL 并标 partial=1。

```
EVAL_COVERAGE=FAIL dataset=test-hard policy=mmesg:ground-sg-qwenvl expected=800 missing=637 extra=0 duplicate=0 conflicting_terminal=0 error_final=0 partial=1
EVAL_REPORT=FAIL dataset=test-hard policy=mmesg:ground-sg-qwenvl count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=0 partial=1
EVAL_VIDEOS=FAIL dataset=test-hard policy=mmesg:ground-sg-qwenvl expected=800 videos=160 missing=640 decode_fail=0 partial=1
```

## 总表

| 模型 | 分母 | 已定终态 | 成功 | 失败 | timeout | error | 缺失 | 冲突 | 全局微平均 | 任务宏平均 | 迟到 | 废弃尝试 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmesg:ground-sg-qwenvl | 800 | 163 | 15 | 115 | 33 | 0 | 637 | 0 | 1.9% | 1.9% | 0 | 0 |

## 任务×档成功率（成功／分母；括号内 timeout／error／缺失）

| 格 | mmesg:ground-sg-qwenvl |
|---|---:|
| BinFill@xhard1 | 1/25 4.0% (11/0/8) |
| BinFill@xhard2 | 0/25 0.0% (0/0/23) |
| ButtonUnmask@xhard1 | 0/13 0.0% (0/0/13) |
| ButtonUnmask@xhard2 | 0/13 0.0% (0/0/13) |
| ButtonUnmask@xhard3 | 0/12 0.0% (4/0/6) |
| ButtonUnmask@xhard4 | 0/12 0.0% (2/0/4) |
| ButtonUnmaskSwap@xhard1 | 0/25 0.0% (0/0/20) |
| ButtonUnmaskSwap@xhard2 | 0/25 0.0% (0/0/25) |
| InsertPeg@xhard4 | 1/50 2.0% (3/0/32) |
| MoveCube@xhard4 | 3/50 6.0% (1/0/44) |
| PatternLock@xhard1 | 0/17 0.0% (0/0/5) |
| PatternLock@xhard2 | 0/17 0.0% (0/0/17) |
| PatternLock@xhard3 | 0/16 0.0% (0/0/16) |
| PickHighlight@xhard1 | 0/25 0.0% (1/0/22) |
| PickHighlight@xhard2 | 0/25 0.0% (3/0/11) |
| PickXtimes@xhard1 | 0/17 0.0% (0/0/17) |
| PickXtimes@xhard2 | 0/17 0.0% (0/0/17) |
| PickXtimes@xhard3 | 0/16 0.0% (0/0/16) |
| RouteStick@xhard1 | 0/17 0.0% (0/0/6) |
| RouteStick@xhard2 | 0/17 0.0% (1/0/12) |
| RouteStick@xhard3 | 0/16 0.0% (0/0/16) |
| StopCube@xhard1 | 0/10 0.0% (0/0/10) |
| StopCube@xhard2 | 0/10 0.0% (0/0/10) |
| StopCube@xhard3 | 0/10 0.0% (0/0/10) |
| StopCube@xhard4 | 0/10 0.0% (0/0/4) |
| StopCube@xhard5 | 0/10 0.0% (0/0/4) |
| SwingXtimes@xhard1 | 0/10 0.0% (0/0/7) |
| SwingXtimes@xhard2 | 0/10 0.0% (0/0/10) |
| SwingXtimes@xhard3 | 0/10 0.0% (0/0/10) |
| SwingXtimes@xhard4 | 0/10 0.0% (0/0/10) |
| SwingXtimes@xhard5 | 0/10 0.0% (0/0/10) |
| VideoPlaceButton@xhard1 | 2/25 8.0% (1/0/16) |
| VideoPlaceButton@xhard2 | 1/25 4.0% (0/0/19) |
| VideoPlaceOrder@xhard1 | 0/25 0.0% (0/0/25) |
| VideoPlaceOrder@xhard2 | 1/25 4.0% (0/0/21) |
| VideoRepick@xhard1 | 2/25 8.0% (0/0/16) |
| VideoRepick@xhard2 | 0/25 0.0% (0/0/25) |
| VideoUnmask@xhard1 | 0/13 0.0% (0/0/13) |
| VideoUnmask@xhard2 | 2/13 15.4% (1/0/10) |
| VideoUnmask@xhard3 | 2/12 16.7% (3/0/5) |
| VideoUnmask@xhard4 | 0/12 0.0% (2/0/9) |
| VideoUnmaskSwap@xhard1 | 0/25 0.0% (0/0/25) |
| VideoUnmaskSwap@xhard2 | 0/25 0.0% (0/0/25) |

## 按任务（跨档）与按档（跨任务）

| 任务 | mmesg:ground-sg-qwenvl |
|---|---:|
| BinFill | 1/50 2.0% |
| ButtonUnmask | 0/50 0.0% |
| ButtonUnmaskSwap | 0/50 0.0% |
| InsertPeg | 1/50 2.0% |
| MoveCube | 3/50 6.0% |
| PatternLock | 0/50 0.0% |
| PickHighlight | 0/50 0.0% |
| PickXtimes | 0/50 0.0% |
| RouteStick | 0/50 0.0% |
| StopCube | 0/50 0.0% |
| SwingXtimes | 0/50 0.0% |
| VideoPlaceButton | 3/50 6.0% |
| VideoPlaceOrder | 1/50 2.0% |
| VideoRepick | 2/50 4.0% |
| VideoUnmask | 4/50 8.0% |
| VideoUnmaskSwap | 0/50 0.0% |

| 档 | mmesg:ground-sg-qwenvl |
|---|---:|
| xhard1 | 5/272 1.8% |
| xhard2 | 4/272 1.5% |
| xhard3 | 2/92 2.2% |
| xhard4 | 4/144 2.8% |
| xhard5 | 0/20 0.0% |

## 预算（逐席）

**mmesg:ground-sg-qwenvl**：reset_claim 总数 338，尝试 169，infra 重试 0，额度提升 0 次。

| 席 | reset_claim | reset 额度 | 尝试 | infra 重试 | infra 重试额度 | infra 错误尝试 | 额度耗尽 | 运行阻塞 | 额度提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s60 | 58 | 120 | 29 | 0 | 3 | 0 | 0 | 0 | — |
| s61 | 58 | 120 | 29 | 0 | 3 | 0 | 0 | 0 | — |
| s62 | 72 | 120 | 36 | 0 | 3 | 0 | 0 | 0 | — |
| s63 | 66 | 120 | 33 | 0 | 3 | 0 | 0 | 0 | — |
| s64 | 54 | 120 | 27 | 0 | 3 | 0 | 0 | 0 | — |
| s65 | 30 | 120 | 15 | 0 | 3 | 0 | 0 | 0 | — |

## 迟到终态与废弃尝试（不入分数）

- 无

## 异常明细

- 视频 mmesg:ground-sg-qwenvl `VideoPlaceOrder_xhard2_19101800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard2_18600900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16400300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmask_xhard3_20800700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23301700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `MoveCube_xhard4_23405700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard1_17202300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard1_17600300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard4_22200300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard1_17001300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceOrder_xhard2_19101900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard2_18601000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16400400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmask_xhard3_20800800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23301900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `MoveCube_xhard4_23405900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard1_17202400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard1_17600400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard4_22200400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard1_17001400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceOrder_xhard2_19102000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard2_18601100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceOrder_xhard2_19102300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard3_20600100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16400900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmask_xhard4_22800000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23302500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `MoveCube_xhard4_23406600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19200300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard1_17600800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard4_22200800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard1_17001800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceOrder_xhard2_19102402`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard3_20600200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16401000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmask_xhard4_22800100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23302800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `MoveCube_xhard4_23407000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19200400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard1_17600900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard4_22200900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard1_17001902`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceOrder_xhard2_19102502`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard3_20600300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19200800`：missing /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/media/mmesg-ground-sg-qwenvl/test-hard/new/PickHighlight_xhard2_19200800.a1
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard1_17601300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard5_24200300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard1_17002300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16900301`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard3_20600700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16401500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmask_xhard4_22800600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23303700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard1_17500400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19200900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard1_17601400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard5_24200400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard1_17002400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16900401`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard3_20600800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmask_xhard4_22801000`：missing /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/media/mmesg-ground-sg-qwenvl/test-hard/new/ButtonUnmask_xhard4_22801000.a1
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23304300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard1_17500800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19201300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19600100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard5_24200800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19000400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16900800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22600000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16402200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmask_xhard4_22801100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23304400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard1_17500900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19201400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19600200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `StopCube_xhard5_24200900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19000500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16900900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22600100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19600500`：missing /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/media/mmesg-ground-sg-qwenvl/test-hard/new/RouteStick_xhard2_19600500.a1
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard1_16300200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19000800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16901200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22600400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16402900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16700302`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23304900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard1_17501300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19201800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19600600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard1_16300300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19000900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16901300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22600500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard1_16403000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16700402`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23305000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard1_17501400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19201900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19600700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard1_16300400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16901400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22600600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19202100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19600900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard1_16300600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16901601`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22600800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18400700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16700701`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23305600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19202200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19601000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard1_16300700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16901701`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22600900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18400900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16700800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23305700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19202300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19601100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard1_16300800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16901801`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22601000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18401000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16700900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23305800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickHighlight_xhard2_19202400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19601200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard1_16300900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16901900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmask_xhard4_22601100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18401100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16701002`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23305900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickXtimes_xhard1_16100000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19601300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard2_18300000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16902001`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmaskSwap_xhard1_16500000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18401300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16701101`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23306000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickXtimes_xhard1_16100100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19601400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard2_18300100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001703`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16902100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmaskSwap_xhard1_16500100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18401400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16701202`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23306200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickXtimes_xhard1_16100200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19601500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard2_18300200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16902201`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmaskSwap_xhard1_16500201`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18401600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16701302`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23306300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickXtimes_xhard1_16100300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard2_19601600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard2_18300300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19001900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16902302`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmaskSwap_xhard1_16500300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18401700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16701400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23306500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickXtimes_xhard1_16100400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard3_21600000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard2_18300400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19002000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard1_16902400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmaskSwap_xhard1_16500400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `BinFill_xhard2_18401900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `ButtonUnmaskSwap_xhard1_16701500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `InsertPeg_xhard4_23306700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PatternLock_xhard2_19500800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `PickXtimes_xhard1_16100500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `RouteStick_xhard3_21600100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `SwingXtimes_xhard2_18300500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoPlaceButton_xhard2_19002100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoRepick_xhard2_18900000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-qwenvl `VideoUnmaskSwap_xhard1_16500500`：no_accepted_terminal 

## 中途进度

- mmesg:ground-sg-qwenvl：已完成 163/800，已完成局平均墙钟 0h05m
  - 未观测格（尚无完成局，用该模型全局平均估算）21 个：ButtonUnmask@xhard1, ButtonUnmask@xhard2, ButtonUnmaskSwap@xhard2, PatternLock@xhard2, PatternLock@xhard3, PickXtimes@xhard1, PickXtimes@xhard2, PickXtimes@xhard3, RouteStick@xhard3, StopCube@xhard1, StopCube@xhard2, StopCube@xhard3, SwingXtimes@xhard2, SwingXtimes@xhard3, SwingXtimes@xhard4, SwingXtimes@xhard5, VideoPlaceOrder@xhard1, VideoRepick@xhard2, VideoUnmask@xhard1, VideoUnmaskSwap@xhard1, VideoUnmaskSwap@xhard2

预计剩余（最慢席位）：**5h56m**（席 74）

| 席 | 预计剩余 | mmesg:ground-sg-qwenvl 剩余局 |
|---|---:|---:|
| 60 | 2h39m | 22 |
| 61 | 2h39m | 22 |
| 62 | 1h30m | 15 |
| 63 | 1h45m | 18 |
| 64 | 2h20m | 24 |
| 65 | 3h23m | 36 |
| 66 | 4h22m | 50 |
| 67 | 4h22m | 50 |
| 68 | 4h21m | 50 |
| 69 | 4h21m | 50 |
| 70 | 3h53m | 50 |
| 71 | 3h53m | 50 |
| 72 | 4h36m | 50 |
| 73 | 4h36m | 50 |
| 74 | 5h56m | 50 |
| 75 | 5h56m | 50 |

视频索引见同目录 `video-index.jsonl`（每次尝试一行，含终态、迟到、错误尝试）。
