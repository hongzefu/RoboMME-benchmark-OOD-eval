# split-accept-20261008-01：拆仓实施与验收（launch）

按旧仓 `hongzefu/robomme_benchmark_MotionJEPA`（分支 `newtaskRelease-taskV9`）的 `1008-split-benchmark-eval-repos-plan.html`（第四稿）实施。
开工令（2026-10-08，用户原话）：「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/1008-split-benchmark-eval-repos-plan.html开始实施有问题问用户但是尽可能不要阻塞。」

## 一、锚点

| 项 | 值 |
|---|---|
| 运行环境 | 环境 A：sled-vail，2 × RTX 6000 Ada，NFS 与 `~/.ssh/config` 均在 |
| 源码锚 `SRC` | `fd0017d6a06b7a225ebd026e188b2c8dba10b951`（旧仓开工时 HEAD）；`git diff --stat 9c78c076 SRC -- src scripts tests pyproject.toml .gitmodules third_party` 为空 |
| 官方锚点 | `RoboMME/robomme_benchmark` main `016ac1c4ef3df2b88488abc19db08f3de83647b5`；`src/robomme`（102 文件）、`challenge_interface`、三个官方入口、`Dockerfile`、`assets`、`LICENSE` 与旧仓逐字节相同；`doc/` 差 `Wechat.jpg` 与 `submission/ponderpounce.md`；旧守卫锚 `1fadc0ec` 是 `016ac1c4` 的祖先，两者只差这两个 doc 文件 |
| benchmark 仓 | `/data/hongzefu/RoboMME-benchmark-OOD` → `github.com/hongzefu/RoboMME-benchmark-OOD`（2026-10-08 先以私有建仓；用户回复「同意，改公开并 fork 两个仓」后经 `gh api -X PATCH … visibility=public` 改为公开） |
| eval 仓 | `/data/hongzefu/RoboMME-benchmark-OOD-eval` → `github.com/hongzefu/RoboMME-benchmark-OOD-eval`（私有） |
| 子模块 | robomme_benchmark（随候选版本升级）、mme-vla `ecf086c3`、SimpleMemVLA `c564c17d`、PonderPounce `723df357`、Astra-on-RoboMME `4c3fd6a8`（后两个已 fork 为 `hongzefu/PonderPounce`、`hongzefu/Astra-on-RoboMME`，锁定 sha 均可从 fork 取到） |

## 二、GL 占位 job（本轮不新提交，用已有四个）

| JobID | 名称 | 节点 | 用途（S5） |
|---|---|---|---|
| 63431430 | sgev-hold-10 | gl1525 | FrameSamp+Modulation |
| 63431431 | sgev-hold-11 | gl1512 | GroundSG Oracle |
| 63431432 | sgev-hold-12 | gl1512 | SimpleMemVLA |
| 63431433 | sgev-hold-13 | gl1513 | PonderPounce |

2026-10-08 17:4x 核对：四个都 RUNNING，剩约 3 天 19 小时，只有 batch／extern 步骤。

## 三、基线（S0，旧仓 SRC，同口径 `timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q`）

`3255 passed, 4 skipped, 683 deselected`，267 s，`TEST_RESOURCE=PASS`。按归属：

| 去向 | 目录 | passed |
|---|---|---|
| benchmark 仓 | contract 219、pipeline/challenge 80（+4 skip）、pipeline/recording 151、static 125、unit/common 12、unit/hard 857、unit/robomme 575、unit/wrappers 109 | 2128 |
| eval 仓 | pipeline/eval 328、pipeline/evalx 334、pipeline/gen 112、pipeline/parity 287、pipeline/site 66 | 1127 |

## 四、本轮 tmux 会话清单

| 会话名 | 用途 | 状态 |
|---|---|---|
| split-sync | 三个子环境 venv 构建 | 已结束（`EXIT_CODE=0`，会话随命令退出） |

## 五、预算账本

按方案 §五：身份执行 26 次硬上限、计量额度 60 硬上限、显式 reset 下限 30；Astra 2 局、5 美元硬上限；基础设施重试 0 次。实际消耗随阶段记在 result.md。
