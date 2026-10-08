# 评估汇总（dataset=test-hard）（中途进度）

- 生成时间：2026-10-06T13:17:31-0400
- manifest：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/inputs/manifests/oracle-gate3.json`；运行根：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/gate3-new-oracle`；本机视频根：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/media`
- 口径：每身份唯一权威终态取账本 `accept` 行的 `accepted_attempt_id`；迟到终态与废弃尝试单列、不入分数；分母固定为 manifest 身份数（800／模型）；执行步上限 1600，超过即判 FAIL。中途进度下判定行一律 FAIL 并标 partial=1。

```
EVAL_COVERAGE=FAIL dataset=test-hard policy=mmesg:ground-sg-oracle expected=800 missing=258 extra=0 duplicate=0 conflicting_terminal=0 error_final=0 partial=1
EVAL_REPORT=FAIL dataset=test-hard policy=mmesg:ground-sg-oracle count_mismatch=0 dataset_crossed=0 media_unexplained=0 exec_over_cap=0 partial=1
EVAL_VIDEOS=FAIL dataset=test-hard policy=mmesg:ground-sg-oracle expected=800 videos=541 missing=259 decode_fail=0 partial=1
```

## 总表

| 模型 | 分母 | 已定终态 | 成功 | 失败 | timeout | error | 缺失 | 冲突 | 全局微平均 | 任务宏平均 | 迟到 | 废弃尝试 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmesg:ground-sg-oracle | 800 | 542 | 270 | 133 | 139 | 0 | 258 | 0 | 33.8% | 33.8% | 0 | 0 |

## 任务×档成功率（成功／分母；括号内 timeout／error／缺失）

| 格 | mmesg:ground-sg-oracle |
|---|---:|
| BinFill@xhard1 | 7/25 28.0% (11/0/6) |
| BinFill@xhard2 | 6/25 24.0% (13/0/6) |
| ButtonUnmask@xhard1 | 3/13 23.1% (2/0/3) |
| ButtonUnmask@xhard2 | 1/13 7.7% (3/0/3) |
| ButtonUnmask@xhard3 | 1/12 8.3% (1/0/4) |
| ButtonUnmask@xhard4 | 0/12 0.0% (4/0/2) |
| ButtonUnmaskSwap@xhard1 | 11/25 44.0% (4/0/6) |
| ButtonUnmaskSwap@xhard2 | 9/25 36.0% (4/0/6) |
| InsertPeg@xhard4 | 5/50 10.0% (31/0/14) |
| MoveCube@xhard4 | 14/50 28.0% (23/0/12) |
| PatternLock@xhard1 | 10/17 58.8% (0/0/4) |
| PatternLock@xhard2 | 6/17 35.3% (0/0/4) |
| PatternLock@xhard3 | 2/16 12.5% (0/0/4) |
| PickHighlight@xhard1 | 0/25 0.0% (19/0/6) |
| PickHighlight@xhard2 | 0/25 0.0% (16/0/8) |
| PickXtimes@xhard1 | 9/17 52.9% (1/0/7) |
| PickXtimes@xhard2 | 10/17 58.8% (0/0/7) |
| PickXtimes@xhard3 | 10/16 62.5% (0/0/6) |
| RouteStick@xhard1 | 3/17 17.6% (0/0/6) |
| RouteStick@xhard2 | 5/17 29.4% (0/0/6) |
| RouteStick@xhard3 | 1/16 6.2% (0/0/6) |
| StopCube@xhard1 | 2/10 20.0% (0/0/3) |
| StopCube@xhard2 | 2/10 20.0% (0/0/4) |
| StopCube@xhard3 | 2/10 20.0% (0/0/5) |
| StopCube@xhard4 | 0/10 0.0% (0/0/3) |
| StopCube@xhard5 | 1/10 10.0% (0/0/3) |
| SwingXtimes@xhard1 | 6/10 60.0% (0/0/4) |
| SwingXtimes@xhard2 | 5/10 50.0% (0/0/5) |
| SwingXtimes@xhard3 | 6/10 60.0% (1/0/3) |
| SwingXtimes@xhard4 | 6/10 60.0% (0/0/3) |
| SwingXtimes@xhard5 | 6/10 60.0% (0/0/4) |
| VideoPlaceButton@xhard1 | 13/25 52.0% (0/0/10) |
| VideoPlaceButton@xhard2 | 13/25 52.0% (0/0/10) |
| VideoPlaceOrder@xhard1 | 13/25 52.0% (0/0/9) |
| VideoPlaceOrder@xhard2 | 15/25 60.0% (0/0/9) |
| VideoRepick@xhard1 | 16/25 64.0% (0/0/9) |
| VideoRepick@xhard2 | 16/25 64.0% (0/0/9) |
| VideoUnmask@xhard1 | 4/13 30.8% (1/0/6) |
| VideoUnmask@xhard2 | 5/13 38.5% (1/0/4) |
| VideoUnmask@xhard3 | 1/12 8.3% (1/0/5) |
| VideoUnmask@xhard4 | 0/12 0.0% (3/0/4) |
| VideoUnmaskSwap@xhard1 | 15/25 60.0% (0/0/10) |
| VideoUnmaskSwap@xhard2 | 10/25 40.0% (0/0/10) |

## 按任务（跨档）与按档（跨任务）

| 任务 | mmesg:ground-sg-oracle |
|---|---:|
| BinFill | 13/50 26.0% |
| ButtonUnmask | 5/50 10.0% |
| ButtonUnmaskSwap | 20/50 40.0% |
| InsertPeg | 5/50 10.0% |
| MoveCube | 14/50 28.0% |
| PatternLock | 18/50 36.0% |
| PickHighlight | 0/50 0.0% |
| PickXtimes | 29/50 58.0% |
| RouteStick | 9/50 18.0% |
| StopCube | 7/50 14.0% |
| SwingXtimes | 29/50 58.0% |
| VideoPlaceButton | 26/50 52.0% |
| VideoPlaceOrder | 28/50 56.0% |
| VideoRepick | 32/50 64.0% |
| VideoUnmask | 10/50 20.0% |
| VideoUnmaskSwap | 25/50 50.0% |

| 档 | mmesg:ground-sg-oracle |
|---|---:|
| xhard1 | 112/272 41.2% |
| xhard2 | 103/272 37.9% |
| xhard3 | 23/92 25.0% |
| xhard4 | 25/144 17.4% |
| xhard5 | 7/20 35.0% |

## 预算（逐席）

**mmesg:ground-sg-oracle**：reset_claim 总数 1086，尝试 543，infra 重试 0，额度提升 0 次。

| 席 | reset_claim | reset 额度 | 尝试 | infra 重试 | infra 重试额度 | infra 错误尝试 | 额度耗尽 | 运行阻塞 | 额度提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s40 | 200 | 220 | 100 | 0 | 3 | 0 | 0 | 0 | — |
| s41 | 200 | 220 | 100 | 0 | 3 | 0 | 0 | 0 | — |
| s42 | 200 | 220 | 100 | 0 | 3 | 0 | 0 | 0 | — |
| s43 | 200 | 220 | 100 | 0 | 3 | 0 | 0 | 0 | — |
| s44 | 200 | 220 | 100 | 0 | 3 | 0 | 0 | 0 | — |
| s45 | 86 | 220 | 43 | 0 | 3 | 0 | 0 | 0 | — |

## 迟到终态与废弃尝试（不入分数）

- 无

## 异常明细

- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard2_19200800`：missing /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261006-02/media/mmesg-ground-sg-oracle/test-hard/new/PickHighlight_xhard2_19200800.a1
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard2_19201600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard2_19202400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard1_16100700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard1_16101500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard2_18100600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard2_18101400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard3_20100500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard3_20101300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard1_17600500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard1_17601300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard2_19600400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard2_19601200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard3_21600300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard3_21601100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard1_16200300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard2_18200100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard2_18200900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard3_20200700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard4_22200500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard5_24200300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard1_16300100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard1_16300900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard2_18300700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard3_20300500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard4_22300300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard5_24300100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard5_24300900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard1_17000700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard1_17001501`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard1_17002300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard2_19000701`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard2_19001500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard2_19002300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard1_17100500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard1_17101300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard1_17102200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard2_19100401`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard2_19101201`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard2_19102100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard1_16900301`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard1_16901101`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard1_16901900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard2_18900200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard2_18901000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard2_18901800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard1_16600100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard1_16600900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard2_18600400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard2_18601200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard3_20600700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard4_22600300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard4_22601100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard1_16500700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard1_16501500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard1_16502300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard2_18500600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard2_18501400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard2_18502200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard1_16400700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard1_16401500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard1_16402700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard2_18401100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard2_18402400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard2_18403400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard1_16800400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard1_16801200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard2_18800700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard3_20800200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard3_20801000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard4_22800600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard1_16700201`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard1_16701002`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard1_16701800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard2_18700100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard2_18700900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard2_18701701`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23300000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23301200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23302200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23303700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23304800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23305900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23307300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23400600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23401500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23402400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23403200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23404800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23406100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard1_17500400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard1_17501200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard2_19500300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard2_19501100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard3_21500200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard3_21501000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard1_17200200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard1_17201000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard1_17201800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard2_19200100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard2_19200900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard2_19201700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard1_16100000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard1_16100800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard1_16101600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard2_18100700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard2_18101500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard3_20100600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickXtimes_xhard3_20101400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard1_17600600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard1_17601400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard2_19600500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard2_19601300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard3_21600400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `RouteStick_xhard3_21601200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard1_16200400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard2_18200200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard3_20200000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard3_20200800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard4_22200600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `StopCube_xhard5_24200400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard1_16300200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard2_18300000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard2_18300800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard3_20300600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard4_22300400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `SwingXtimes_xhard5_24300200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard1_17000000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard1_17000800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard1_17001600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard1_17002400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard2_19000800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard2_19001600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceButton_xhard2_19002401`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard1_17100608`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard1_17101501`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard1_17102300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard2_19100502`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard2_19101305`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoPlaceOrder_xhard2_19102200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard1_16900401`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard1_16901200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard1_16902001`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard2_18900300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard2_18901100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoRepick_xhard2_18901901`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard1_16600200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard1_16601000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard2_18600500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard3_20600000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard3_20600800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmask_xhard4_22600400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard1_16500000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard1_16500800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard1_16501600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard1_16502400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard2_18500700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard2_18501500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `VideoUnmaskSwap_xhard2_18502300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard1_16400800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard1_16401700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard1_16402900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard2_18401300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard2_18402500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `BinFill_xhard2_18403500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard1_16800500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard2_18800000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard2_18800800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard3_20800300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard3_20801100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmask_xhard4_22800700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard1_16700302`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard1_16701101`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard1_16701900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard2_18700201`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard2_18701000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `ButtonUnmaskSwap_xhard2_18701800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23300100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23301300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23302300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23303800`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23304900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23306000`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `InsertPeg_xhard4_23307400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23400700`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23401600`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23402500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23403500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23404900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `MoveCube_xhard4_23406500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard1_17500500`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard1_17501300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard2_19500400`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard2_19501200`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard3_21500300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PatternLock_xhard3_21501100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard1_17200300`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard1_17201100`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard1_17201900`：no_accepted_terminal 
- 视频 mmesg:ground-sg-oracle `PickHighlight_xhard2_19200200`：no_accepted_terminal 

## 中途进度

- mmesg:ground-sg-oracle：已完成 542/800，已完成局平均墙钟 0h00m
  - 未观测格（尚无完成局，用该模型全局平均估算）0 个：无

预计剩余（最慢席位）：**0h56m**（席 46）

| 席 | 预计剩余 | mmesg:ground-sg-oracle 剩余局 |
|---|---:|---:|
| 40 | 0h00m | 0 |
| 41 | 0h00m | 0 |
| 42 | 0h00m | 0 |
| 43 | 0h00m | 0 |
| 44 | 0h00m | 0 |
| 45 | 0h27m | 58 |
| 46 | 0h56m | 100 |
| 47 | 0h56m | 100 |

视频索引见同目录 `video-index.jsonl`（每次尝试一行，含终态、迟到、错误尝试）。
