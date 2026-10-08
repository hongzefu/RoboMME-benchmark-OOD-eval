# V9 双模型评估汇总

- 新评 80 局（本次运行）+ 复用 720 局（V8 评估，终态取 V8 账本 `accepted_attempt_id`） = 总表 800 局。复用集合唯一依据：`artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/manifest/reused.json`（sha256 `a476b25b792799f3e6d484be74462bed8bb95e330fea004956522f8dcb6d9b66`）；V8 运行根 `artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01/nfs-records/run`；V8 manifest `artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01/manifest/manifest.json`。
- 注意（R-7）：复用的 720 局是 V8 2026-10-02 跑的结果，新评 80 局是之后跑的；权重、tokenizer、客户端钉在同一锁值，但 MME-VLA 同入口重跑本有翻转，总表新旧两部分不是同一时刻的采样。

```
V9_EVAL_COVERAGE=PASS policies=2 expected=80 missing=0 extra=0 duplicate=0 conflicting_terminal=0 error_final=0
V9_EVAL_REPORT=PASS total=800 new=80 reused=720 count_mismatch=0 media_unexplained=0
V9_EVAL_VIDEOS=PASS policies=2 expected=160 videos=160 missing=0 decode_fail=0 sha_mismatch=0 moved_record_absent=0
```

## 800 局总表

| 模型 | 分母 | 成功 | 失败 | timeout | error | 缺失 | 冲突 | 全局微平均 | 任务宏平均 | 新评成功 | 复用成功 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smvla | 800 | 178 | 471 | 151 | 0 | 0 | 0 | 22.2% | 22.2% | 11/80 | 167/720 |
| mme | 800 | 39 | 678 | 83 | 0 | 0 | 0 | 4.9% | 4.9% | 8/80 | 31/720 |

## 按任务（跨档，800 局）

| 分组 | smvla | mme |
|---|---:|---:|
| BinFill | 4/50 8.0% | 0/50 0.0% |
| ButtonUnmask | 29/50 58.0% | 1/50 2.0% |
| ButtonUnmaskSwap | 9/50 18.0% | 2/50 4.0% |
| InsertPeg | 16/50 32.0% | 0/50 0.0% |
| MoveCube | 4/50 8.0% | 8/50 16.0% |
| PatternLock | 12/50 24.0% | 0/50 0.0% |
| PickHighlight | 0/50 0.0% | 0/50 0.0% |
| PickXtimes | 0/50 0.0% | 0/50 0.0% |
| RouteStick | 13/50 26.0% | 0/50 0.0% |
| StopCube | 0/50 0.0% | 0/50 0.0% |
| SwingXtimes | 12/50 24.0% | 0/50 0.0% |
| VideoPlaceButton | 15/50 30.0% | 12/50 24.0% |
| VideoPlaceOrder | 10/50 20.0% | 10/50 20.0% |
| VideoRepick | 18/50 36.0% | 2/50 4.0% |
| VideoUnmask | 33/50 66.0% | 0/50 0.0% |
| VideoUnmaskSwap | 3/50 6.0% | 4/50 8.0% |

## 按档（跨任务，800 局）

| 分组 | smvla | mme |
|---|---:|---:|
| xhard1 | 82/272 30.1% | 18/272 6.6% |
| xhard2 | 45/272 16.5% | 13/272 4.8% |
| xhard3 | 20/92 21.7% | 0/92 0.0% |
| xhard4 | 31/144 21.5% | 8/144 5.6% |
| xhard5 | 0/20 0.0% | 0/20 0.0% |

## 按格（800 局）

| 分组 | smvla | mme |
|---|---:|---:|
| BinFill@xhard1 | 4/25 16.0% | 0/25 0.0% |
| BinFill@xhard2 | 0/25 0.0% | 0/25 0.0% |
| ButtonUnmask@xhard1 | 9/13 69.2% | 1/13 7.7% |
| ButtonUnmask@xhard2 | 9/13 69.2% | 0/13 0.0% |
| ButtonUnmask@xhard3 | 6/12 50.0% | 0/12 0.0% |
| ButtonUnmask@xhard4 | 5/12 41.7% | 0/12 0.0% |
| ButtonUnmaskSwap@xhard1 | 9/25 36.0% | 2/25 8.0% |
| ButtonUnmaskSwap@xhard2 | 0/25 0.0% | 0/25 0.0% |
| InsertPeg@xhard4 | 16/50 32.0% | 0/50 0.0% |
| MoveCube@xhard4 | 4/50 8.0% | 8/50 16.0% |
| PatternLock@xhard1 | 7/17 41.2% | 0/17 0.0% |
| PatternLock@xhard2 | 2/17 11.8% | 0/17 0.0% |
| PatternLock@xhard3 | 3/16 18.8% | 0/16 0.0% |
| PickHighlight@xhard1 | 0/25 0.0% | 0/25 0.0% |
| PickHighlight@xhard2 | 0/25 0.0% | 0/25 0.0% |
| PickXtimes@xhard1 | 0/17 0.0% | 0/17 0.0% |
| PickXtimes@xhard2 | 0/17 0.0% | 0/17 0.0% |
| PickXtimes@xhard3 | 0/16 0.0% | 0/16 0.0% |
| RouteStick@xhard1 | 8/17 47.1% | 0/17 0.0% |
| RouteStick@xhard2 | 3/17 17.6% | 0/17 0.0% |
| RouteStick@xhard3 | 2/16 12.5% | 0/16 0.0% |
| StopCube@xhard1 | 0/10 0.0% | 0/10 0.0% |
| StopCube@xhard2 | 0/10 0.0% | 0/10 0.0% |
| StopCube@xhard3 | 0/10 0.0% | 0/10 0.0% |
| StopCube@xhard4 | 0/10 0.0% | 0/10 0.0% |
| StopCube@xhard5 | 0/10 0.0% | 0/10 0.0% |
| SwingXtimes@xhard1 | 9/10 90.0% | 0/10 0.0% |
| SwingXtimes@xhard2 | 1/10 10.0% | 0/10 0.0% |
| SwingXtimes@xhard3 | 2/10 20.0% | 0/10 0.0% |
| SwingXtimes@xhard4 | 0/10 0.0% | 0/10 0.0% |
| SwingXtimes@xhard5 | 0/10 0.0% | 0/10 0.0% |
| VideoPlaceButton@xhard1 | 9/25 36.0% | 6/25 24.0% |
| VideoPlaceButton@xhard2 | 6/25 24.0% | 6/25 24.0% |
| VideoPlaceOrder@xhard1 | 7/25 28.0% | 5/25 20.0% |
| VideoPlaceOrder@xhard2 | 3/25 12.0% | 5/25 20.0% |
| VideoRepick@xhard1 | 8/25 32.0% | 2/25 8.0% |
| VideoRepick@xhard2 | 10/25 40.0% | 0/25 0.0% |
| VideoUnmask@xhard1 | 10/13 76.9% | 0/13 0.0% |
| VideoUnmask@xhard2 | 10/13 76.9% | 0/13 0.0% |
| VideoUnmask@xhard3 | 7/12 58.3% | 0/12 0.0% |
| VideoUnmask@xhard4 | 6/12 50.0% | 0/12 0.0% |
| VideoUnmaskSwap@xhard1 | 2/25 8.0% | 2/25 8.0% |
| VideoUnmaskSwap@xhard2 | 1/25 4.0% | 2/25 8.0% |

## 新评视频核对

期望 160，通过 160，缺失 0，解码失败 0，sha 不符 0，错误终局无录像（写明原因）0，moved.jsonl 无记录 0。

---

以下为新评身份分表（V8 口径，分母为新评行数）。

## 新评身份分表（V8 口径）

- 生成时间：2026-10-02T18:54:58-0400
- manifest：`artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/manifest/manifest.json`；运行根：`artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/nfs-records/run`；本机视频根：`artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/videos`
- 口径：每身份唯一权威终态取账本 `accept` 行的 `accepted_attempt_id`；迟到终态与废弃尝试单列、不入分数；分母固定为 manifest 身份数（80／模型）；执行步上限 1600，超过即判 FAIL。

```
V8_EVAL_COVERAGE=PASS policies=2 missing=0 extra=0 duplicate=0 conflicting_terminal=0 late_ignored=0 error_final=0
V8_EVAL_REPORT=PASS count_mismatch=0 media_unexplained=0 exec_over_cap=0
```

## 总表

| 模型 | 分母 | 已定终态 | 成功 | 失败 | timeout | error | 缺失 | 冲突 | 全局微平均 | 任务宏平均 | 迟到 | 废弃尝试 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smvla | 80 | 80 | 11 | 13 | 56 | 0 | 0 | 0 | 13.8% | 15.7% | 0 | 0 |
| mme | 80 | 80 | 8 | 36 | 36 | 0 | 0 | 0 | 10.0% | 8.0% | 0 | 0 |

## 任务×档成功率（成功／分母；括号内 timeout／error／缺失）

| 格 | smvla | mme |
|---|---:|---:|
| InsertPeg@xhard4 | 7/30 23.3% (14/0/0) | 0/30 0.0% (18/0/0) |
| MoveCube@xhard4 | 4/50 8.0% (42/0/0) | 8/50 16.0% (18/0/0) |

## 按任务（跨档）与按档（跨任务）

| 任务 | smvla | mme |
|---|---:|---:|
| InsertPeg | 7/30 23.3% | 0/30 0.0% |
| MoveCube | 4/50 8.0% | 8/50 16.0% |

| 档 | smvla | mme |
|---|---:|---:|
| xhard4 | 11/80 13.8% | 8/80 10.0% |

## 预算（逐席）

**smvla**：reset_claim 总数 160，尝试 80，infra 重试 0，额度提升 0 次。

| 席 | reset_claim | reset 额度 | 尝试 | infra 重试 | infra 重试额度 | infra 错误尝试 | 额度耗尽 | 运行阻塞 | 额度提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s00 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s01 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s02 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s03 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s04 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s05 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s06 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s07 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s08 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s09 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |

**mme**：reset_claim 总数 160，尝试 80，infra 重试 0，额度提升 0 次。

| 席 | reset_claim | reset 额度 | 尝试 | infra 重试 | infra 重试额度 | infra 错误尝试 | 额度耗尽 | 运行阻塞 | 额度提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s00 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s01 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s02 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s03 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s04 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s05 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s06 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s07 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s08 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |
| s09 | 16 | 64 | 8 | 0 | 1 | 0 | 0 | 0 | — |

## 迟到终态与废弃尝试（不入分数）

- 无

## 异常明细

- 无

视频索引见同目录 `video-index.jsonl`（每次尝试一行，含终态、迟到、错误尝试）。
