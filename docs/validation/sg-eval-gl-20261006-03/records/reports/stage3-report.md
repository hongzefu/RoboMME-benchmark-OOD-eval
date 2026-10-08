# 四模型评估总报告

## 覆盖与成功率

| 集合 | 模型种子 | 期望 | 覆盖 | 缺失 | 多余 | 重复 | 冲突 | 成功 | 成功率 | 终态分布 |
|---|---|---|---|---|---|---|---|---|---|---|
| perceptual-framesamp-modul/ood/new@gl | 7 | 86 | 86 | 0 | 0 | 0 | 0 | 3 | 0.0349 | fail:79，success:3，timeout:4 |
| smvla/ood/new@gl | 7 | 86 | 86 | 0 | 0 | 0 | 0 | 19 | 0.2209 | fail:57，success:19，timeout:10 |
| pp/ood/new@gl | 7 | 86 | 86 | 0 | 0 | 0 | 0 | 15 | 0.1744 | fail:59，success:15，timeout:12 |
| groundsg-ground-sg-qwenvl/ood/new@gl | 7 | 86 | 86 | 0 | 0 | 0 | 0 | 13 | 0.1512 | fail:62，success:13，timeout:11 |
| groundsg-ground-sg-memer/ood/new@gl | 7 | 86 | 86 | 0 | 0 | 0 | 0 | 5 | 0.0581 | fail:72，success:5，timeout:9 |

### perceptual-framesamp-modul/ood/new@gl：按档

| 档 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| xhard1 | 2 | 28 | 0.0714 |
| xhard2 | 1 | 28 | 0.0357 |
| xhard3 | 0 | 14 | 0.0 |
| xhard4 | 0 | 12 | 0.0 |
| xhard5 | 0 | 4 | 0.0 |

### perceptual-framesamp-modul/ood/new@gl：按任务

| 任务 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| BinFill | 0 | 4 | 0.0 |
| ButtonUnmask | 0 | 8 | 0.0 |
| ButtonUnmaskSwap | 0 | 4 | 0.0 |
| InsertPeg | 0 | 2 | 0.0 |
| MoveCube | 0 | 2 | 0.0 |
| PatternLock | 0 | 6 | 0.0 |
| PickHighlight | 0 | 4 | 0.0 |
| PickXtimes | 0 | 6 | 0.0 |
| RouteStick | 0 | 6 | 0.0 |
| StopCube | 0 | 10 | 0.0 |
| SwingXtimes | 0 | 10 | 0.0 |
| VideoPlaceButton | 1 | 4 | 0.25 |
| VideoPlaceOrder | 0 | 4 | 0.0 |
| VideoRepick | 0 | 4 | 0.0 |
| VideoUnmask | 0 | 8 | 0.0 |
| VideoUnmaskSwap | 2 | 4 | 0.5 |

### smvla/ood/new@gl：按档

| 档 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| xhard1 | 8 | 28 | 0.2857 |
| xhard2 | 5 | 28 | 0.1786 |
| xhard3 | 4 | 14 | 0.2857 |
| xhard4 | 2 | 12 | 0.1667 |
| xhard5 | 0 | 4 | 0.0 |

### smvla/ood/new@gl：按任务

| 任务 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| BinFill | 0 | 4 | 0.0 |
| ButtonUnmask | 3 | 8 | 0.375 |
| ButtonUnmaskSwap | 0 | 4 | 0.0 |
| InsertPeg | 1 | 2 | 0.5 |
| MoveCube | 0 | 2 | 0.0 |
| PatternLock | 1 | 6 | 0.1667 |
| PickHighlight | 0 | 4 | 0.0 |
| PickXtimes | 0 | 6 | 0.0 |
| RouteStick | 2 | 6 | 0.3333 |
| StopCube | 0 | 10 | 0.0 |
| SwingXtimes | 3 | 10 | 0.3 |
| VideoPlaceButton | 1 | 4 | 0.25 |
| VideoPlaceOrder | 0 | 4 | 0.0 |
| VideoRepick | 1 | 4 | 0.25 |
| VideoUnmask | 7 | 8 | 0.875 |
| VideoUnmaskSwap | 0 | 4 | 0.0 |

### pp/ood/new@gl：按档

| 档 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| xhard1 | 7 | 28 | 0.25 |
| xhard2 | 5 | 28 | 0.1786 |
| xhard3 | 2 | 14 | 0.1429 |
| xhard4 | 1 | 12 | 0.0833 |
| xhard5 | 0 | 4 | 0.0 |

### pp/ood/new@gl：按任务

| 任务 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| BinFill | 0 | 4 | 0.0 |
| ButtonUnmask | 1 | 8 | 0.125 |
| ButtonUnmaskSwap | 0 | 4 | 0.0 |
| InsertPeg | 0 | 2 | 0.0 |
| MoveCube | 0 | 2 | 0.0 |
| PatternLock | 0 | 6 | 0.0 |
| PickHighlight | 0 | 4 | 0.0 |
| PickXtimes | 3 | 6 | 0.5 |
| RouteStick | 0 | 6 | 0.0 |
| StopCube | 0 | 10 | 0.0 |
| SwingXtimes | 2 | 10 | 0.2 |
| VideoPlaceButton | 1 | 4 | 0.25 |
| VideoPlaceOrder | 1 | 4 | 0.25 |
| VideoRepick | 1 | 4 | 0.25 |
| VideoUnmask | 6 | 8 | 0.75 |
| VideoUnmaskSwap | 0 | 4 | 0.0 |

### groundsg-ground-sg-qwenvl/ood/new@gl：按档

| 档 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| xhard1 | 7 | 28 | 0.25 |
| xhard2 | 4 | 28 | 0.1429 |
| xhard3 | 2 | 14 | 0.1429 |
| xhard4 | 0 | 12 | 0.0 |
| xhard5 | 0 | 4 | 0.0 |

### groundsg-ground-sg-qwenvl/ood/new@gl：按任务

| 任务 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| BinFill | 0 | 4 | 0.0 |
| ButtonUnmask | 0 | 8 | 0.0 |
| ButtonUnmaskSwap | 0 | 4 | 0.0 |
| InsertPeg | 0 | 2 | 0.0 |
| MoveCube | 0 | 2 | 0.0 |
| PatternLock | 0 | 6 | 0.0 |
| PickHighlight | 0 | 4 | 0.0 |
| PickXtimes | 5 | 6 | 0.8333 |
| RouteStick | 0 | 6 | 0.0 |
| StopCube | 0 | 10 | 0.0 |
| SwingXtimes | 0 | 10 | 0.0 |
| VideoPlaceButton | 1 | 4 | 0.25 |
| VideoPlaceOrder | 1 | 4 | 0.25 |
| VideoRepick | 2 | 4 | 0.5 |
| VideoUnmask | 4 | 8 | 0.5 |
| VideoUnmaskSwap | 0 | 4 | 0.0 |

### groundsg-ground-sg-memer/ood/new@gl：按档

| 档 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| xhard1 | 2 | 28 | 0.0714 |
| xhard2 | 1 | 28 | 0.0357 |
| xhard3 | 1 | 14 | 0.0714 |
| xhard4 | 1 | 12 | 0.0833 |
| xhard5 | 0 | 4 | 0.0 |

### groundsg-ground-sg-memer/ood/new@gl：按任务

| 任务 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| BinFill | 0 | 4 | 0.0 |
| ButtonUnmask | 2 | 8 | 0.25 |
| ButtonUnmaskSwap | 0 | 4 | 0.0 |
| InsertPeg | 0 | 2 | 0.0 |
| MoveCube | 1 | 2 | 0.5 |
| PatternLock | 0 | 6 | 0.0 |
| PickHighlight | 0 | 4 | 0.0 |
| PickXtimes | 0 | 6 | 0.0 |
| RouteStick | 0 | 6 | 0.0 |
| StopCube | 0 | 10 | 0.0 |
| SwingXtimes | 0 | 10 | 0.0 |
| VideoPlaceButton | 0 | 4 | 0.0 |
| VideoPlaceOrder | 0 | 4 | 0.0 |
| VideoRepick | 0 | 4 | 0.0 |
| VideoUnmask | 2 | 8 | 0.25 |
| VideoUnmaskSwap | 0 | 4 | 0.0 |

## 第二档差异

| 模型 | 地点 | 判定 | 配对 | 同终态 | 逐项相同 | 启动首局相同 |
|---|---|---|---|---|---|---|
| groundsg-memer | local | INFO | 48 | 41 | 0 | — |

## 预算

- 来源：results；尝试 431（上限 None），reset 862（上限 None），判定 NA。


```
MODEL_EVAL_REPORT=PASS sets=5 complete=5 incomplete=0 episodes=430 gate2=1
EVAL_BUDGET=NA attempts=431 resets=862 source=results
STAGE3_MATRIX=PASS policy_seed=7 combinations=5 unique_terminal=430 incomplete=0 seed_mismatch=0 duplicate_combination=0
```
