# 计划外事件与处置（6 起）

> 按时间排列，时间一律 EDT（2026-09-30）。每起写：时间线、根因、影响（局数写乘式）、处置、证据路径、教训。汇总口径：0 步、环境未建成的基础设施失败 `infra=True` 永不计为策略结果，计入各项重试额度并如实标超额（12.275）。用户原话：本轮原话 2（一口气跑完）、7、8（自行决策、该测的都要测完、不再问用户）——全部处置均在已批准的方案范围内由主会话自行决定，未向用户请示。

## 总账

| # | 事件 | 丢失的真实局 | 0 步基础设施失败 | 需要重跑的格 |
|---|---|---:|---:|---|
| 1 | 后台脚本 `cd` 失败后在主仓库 `uv sync`，重建主 `.venv` | 0（2.1 半格作废重跑） | — | C1（6 个身份 × 2 次 reset = 12 次 reset 作废，另 1 个身份卡住） |
| 2 | 回放输入 B-mme 把整片代理根目录当单连接 | 0 | — | 无（0 局工具） |
| 3 | 运行中替换 NFS 上的 `step.sh` | 0 | — | 无（报告补写） |
| 4 | 暂存核对把 NFS ACL／组差异当内容差；MME 权重符号链接 | 0 | — | 无（补暂存） |
| 5 | `restage.sh` 通配符误删正在写的输出目录 | 0（该步当时 0 局） | — | E5 MME 整格 |
| 6 | GL `--gpu_cmode=shared` 步骤结束把卡重置为独占 | 0（全部补跑） | 347 | E6 SMVLA、E7 MME、5.2 MME 101 个身份、O2 片 1、2、3、4、7、8 |

```text
BUDGET=INFO item=incident_infra_attempts attempts=0 unique=0 incident=347 retries=347 cap=n/a retry_cap=见各项 over=no retry_over=n/a est=0 步、环境未建成（GPU 计算模式事故等），不计轨迹尝试、已计入各项重试；其中事故留档目录内 148
```

## 事故 1：主仓库 `.venv` 被重建（01:42）

- **时间线**：01:42:39 为 GL 建 worktree 的后台脚本里 `git worktree add` 失败，随后 `cd` 失败；`set -e` 被管道吞掉，脚本继续在主仓库执行 `uv sync`，按 NFS 上的 `cpython-3.11.14` 重建了主 `.venv` 并删去 dev extra（pytest 等）。本机卡 0 环境检测车道（tmux `v75-env-card0`，C1 首遍）恰在重装中读文件，卡在资产下载提示。
- **根因**：脚本没有逐步校验返回值，`cd` 失败不退出；`uv sync` 用了相对当前目录的工程。
- **影响**：C1 已完成 6 个身份（6 × 2 = 12 次 reset）作废，外加卡住的 1 个身份；2.1 重试比子额度 6 多 1 次身份尝试（`BUDGET … item=2.1_环境检测 attempts=289 … retries=1`），仍在总上限内。其他车道不受影响。
- **处置**：按清单停掉 `v75-env-card0`（`tmux ls` 前后核对只少该会话）；用原解释器 `~/.local/share/uv/python/cpython-3.11` ＋ `--group eval-client --extra dev` 恢复，包清单与 `.claude/worktrees/v7/.venv` 对照只多 `eval-client` 与 dev 应有项，mani_skill 资产一致；删除 C1 半格后重跑。GL worktree 改用严格脚本（先 `fetch`、每步校验）重建。
- **证据**：`artifacts/v7.5eval/logs/env-card0.aborted-venv-rebuild.log`；12.268 commit body「意外」段；`/home/hongzefu/.claude/jobs/6f127313/tmp/gl_wt*.log`。
- **教训**：后台脚本一律 `set -euo pipefail` 且每个 `cd` 带 `|| exit`；任何 `uv sync` 都用绝对路径指定工程与 `UV_PROJECT_ENVIRONMENT`，不依赖当前目录。

## 事故 2：B-mme 回放输入重建失败

- **时间线与根因**：第 4 步输入 B（重跑一片 9 首局 BinFill ep31 seed 543100，MME 失败 575 步）从官方录制重建时，`--proxy-rec` 传入的是整片代理根目录（预检 ＋ 19 局共 20 条连接），`policy_replay.py build-inputs` 按单连接处理，找不到 `s2c.actions`，首次推理后中止：`BUILD_INPUTS=FAIL policy=mme kind=official messages=3 infers=1 basis=official_client_send_sha mismatch=70 first_mismatch=3 exec_equal=False`。
- **影响**：0 局；MME 回放车道（`replay-local2.sh`、`replay-gl.sh` 都等待 `B-mme/meta.json`）顺延。
- **处置**：12.273 `bac285d8`：在代理根目录下按「本局客户端发送 sha 序列逐条全等」挑选连接（跳过预检、他局、截断前缀），显式单连接须序列全等否则报错；新增多局合成回归测试。重建后 `BUILD_INPUTS=PASS policy=mme kind=official messages=73 infers=36 mismatch=0 exec_equal=True`（匹配 `conn-3401045-0000`，执行 575/575 行）。
- **证据**：12.273 commit body；`artifacts/v7.5eval/logs/build-bmme.log`；`artifacts/v7.5eval/replay/inputs/B-mme/`。
- **教训**：录制格式的「一局一连接」假设要在工具里断言，而不是隐含依赖。

## 事故 3：运行中替换 `step.sh`（02:55）

- **时间线**：02:55 为加入改派跳过逻辑，用 `mv` 替换了 NFS 上的 `scripts/eval-official/v75-lanes/gl/step.sh`。正在 GL 上执行旧 `step.sh` 的步骤在命令结束后回到脚本读下一行时读到失效句柄，没有写报告 JSON：O1-s7（05:30:23 `report_missing`）、O1-s8（05:35:16）、E5-smvla（05:36:29）、E7-smvla（05:38:09）、N-smvla 新1～新4。
- **根因**：bash 边执行边从文件读脚本；NFS 上 `mv` 替换使旧 inode 在其他节点失效。编排器只认报告 JSON（方案 P4 设计），缺报告即判 `report_missing`。
- **影响**：0 局。产物完整：O1-s7、O1-s8 两策略各 19 行、rc=0、已暂存，E0 冻结检查起止都 PASS；E5／E7 SMVLA 的产物留在节点 `/tmp`。
- **处置**：按日志补写报告（O1-s7、O1-s8）；E5／E7 SMVLA 由 `retry` 编排器在乙、丁上 `restage` 补暂存（`restage-ding-E7smvla`：`ok=1 rc=0`）；搬运器随后接手。
- **证据**：`artifacts/v7.5eval/nfs-archive/state/official/O1-s7.fail`、`O1-s8.fail`（`"reason": "report_missing"`）；`artifacts/v7.5eval/nfs-archive/state/{official,main}/events.log`；`artifacts/v7.5eval/nfs-archive/reports/O1-s7.json`（补写）。
- **教训**：运行中的脚本不原地替换；新版本另起文件名（如 `step2.sh`），新计划指向新文件。

## 事故 4：暂存核对误报与 MME 权重符号链接

- **根因与影响**：`seat_run.sh` 暂存核对原用 `rsync -a --itemize`，把 NFS 的 ACL 与组权限差异当作内容差异，报 `SEAT_STAGE_FAIL n=492`、步骤 rc=7（产物其实完整，留在节点 `/tmp`）；另外 `run_seat.sh` 以 `$MME_CKPT/..` 读 `history_config`，GL 上权重是符号链接，父目录解析错。0 局损失。
- **处置**：核对改为 `rsync -rt` 复制 ＋ `rsync -rc --dry-run` 只比内容；复制真实权重目录到 `v75eval/ckpt/mme`（18 个文件 sha256 与资产锁一致），`seat_run.sh` 改用该路径；受影响目录补暂存。
- **证据**：`scripts/eval-official/v75-lanes/gl/seat_run.sh` 文件头注释；`artifacts/v7.5eval/nfs-archive/state/main/events.log`（`E5-mme-s-yi … reason=exit … rc=7`）。
- **教训**：跨文件系统核对只比内容（`-c`），不比元数据；权重路径一律解析为真实目录后再做任何 `..` 运算。

## 事故 5：`restage.sh` 误删正在写的输出目录

- **时间线与根因**：补暂存脚本 `restage.sh` 早期版本用通配符扫描 `/tmp/v75-*`，把乙上正在写的 E5 MME 输出目录当作「已结束待暂存」处理并删除。当时该步 0 局。
- **影响**：0 局丢失；E5 MME 需整格重跑（2 策略中的 1 策略 × 16 任务 × 1 档 × 3 局 = 48 局重来）。
- **处置**：`restage.sh` 改为只处理显式点名的 `<cond>:<policy>`，且先以 `pgrep -u "$USER" -f "[r]un_seat.sh .*--cond $COND .*--out $d"` 确认对应步骤已结束（`RESTAGE_BUSY` 即跳过）；E5 MME 在 `final` 编排器中整格重跑，06:54:12 完成。
- **证据**：`scripts/eval-official/v75-lanes/gl/restage.sh`（现行显式点名版，文件头写明原因）；`artifacts/v7.5eval/nfs-archive/state/final/events.log`（`E5-mme-s-yi … STEP_DONE`）。
- **教训**：清理类脚本永远不用通配符匹配可能在用的目录；删除前核对属主进程。

## 事故 6：GPU 计算模式被重置为独占（05:15～05:52，最重）

### 时间线

- 05:15 起：同一席位上并存多个 `srun --overlap` 作业步（长跑的评估步骤 ＋ 短的查询、补暂存 srun ＋ 编排器设计的「在 srun 里等待」步骤）。每当带 `--gpu_cmode=shared` 的某个作业步结束，Slurm 就把这张卡重置为 `Exclusive_Process`。已查明的触发点：05:15 新1～新4 上的补暂存步骤（`mmeprod` 的 `restage-new1`～`restage-new4`）、05:39 丁上的补暂存步骤（`retry` 的 `restage-ding-E7smvla`）、约 05:44 主会话在全部 8 个 GPU 席位上各发一条带 `--gpu_cmode=shared` 的查询 srun——它们结束时把 8 张卡全部重置为独占，是 05:44～05:51 集中损坏的直接原因。
- 05:38～05:46：丁上 E7 MME 跑完第 1 局后，后续每局建环境都失败（Vulkan `createDeviceUnique`），8 分钟就「跑完」48 局——比预期约 28 分钟快太多，由此被发现。乙上 E6 SMVLA 同时出现同样错误。
- 05:47～05:52：`o2x` 编排器的新1～新4（O2 片 3、4、7、8）MME 段 rc=1；`mmeprod` 的新1～新4 MME 首轮 rc=137；保持步骤启动时取消了点名的等待型步骤（`KEEPER cancelled 62608440.8 …`、`62608429.1`、`62608429.7`、`62608430.2`、`62608430.8`、`62608595.6`、`62608595.14`），甲、丙的 5.2 步骤与丁的 O2 片 1、2 随之 rc=137。
- 05:51:36～05:52:18：甲、丙、乙、丁的保持步骤（`srun --overlap --gpu_cmode=shared … sleep infinity`）起跑，卡恢复共享（`keeper-*.log` 显示甲、丙、丁各起过两次）；登录节点 tmux 会话 `v75-keep-*`、`v75-keep2-*`，08:1x 占位 job 按清单取消时随之自行结束。
- 05:55 起：`final`、`final2`～`final6` 编排器补跑（等待一律放编排席本地步骤，`where=local`）。

### 根因与实验

- Slurm 的 SPANK 插件 `gpu_cmode`：带 `--gpu_cmode=shared` 的作业步启动时把卡设为 `Default`，**结束时把卡重置回集群默认的 `exclusive`**；不带该参数的作业步不改模式（主会话实测结论，见下）。日志原文：

  ```text
  [2026-09-30T05:49:37.358] error: SPANK:gpu_cmode: error resetting compute mode to default 'exclusive' on GPU(s): 0
  [2026-09-30T05:51:36.813] error: SPANK:gpu_cmode: can't find nvidia-smi in PATH.
  ```

  （前者见 `artifacts/v7.5eval/nfs-archive/state/mmeprod/logs/N-mme-s-new1.log`，后者见 `artifacts/v7.5eval/nfs-archive/state/retry/logs/E5-mme-retry-s-yi.log`、`artifacts/v7.5eval/nfs-archive/state/main/logs/N-mme-s-jia.log`。）
- 卡处于 `Exclusive_Process` 时一张卡只允许一个 CUDA context，而本链路的环境进程需要两个（torch 的 CUDA context ＋ svulkan2 为 CUDA-Vulkan 互操作开的那个），于是新建环境报 `RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed`；已经建好环境、正在跑的进程不受影响——这就是 E7 MME「第 1 局真实、之后全错」的原因。
- 旁证：10:22Z（06:22 EDT）核查时两张卡都是 `Default`；中途看到的独占是瞬态——丙的 O1-s6 在 06:21:55 结束（结束时重置），紧接着启动的 O2-s9 又设回共享。
- **对照实验**（主会话，gl1527＝丁，约 05:47）：t0 卡为 `Exclusive_Process` → 起带 `--gpu_cmode=shared` 的作业步 A 并保持运行：`Default` → A 运行期间起一个不带 `--gpu_cmode` 的作业步 B 并让它结束：仍为 `Default` → 结束 A：`Exclusive_Process`。结论：带该参数的作业步开始时设 `Default`、结束时无论同卡是否还有其他作业步都重置为 `Exclusive_Process`；不带该参数的作业步不改模式。实验的逐条命令原文未落文件，只有主会话当时的终端记录，这里按主会话转述记录，属探索性证据（`AGENTS.md` 第 17 条）。

### 影响（全部为 0 步基础设施失败，另有 8 局真实结果留在事故目录、不进比较）

```text
INCIDENT=INFO dir=s-yi-vulkanfail-0544 cond=E6 policy=smvla records=76 real=7 infra=69 incident_0step=69 hist=env_build|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:69 compared=excluded
INCIDENT=INFO dir=s-ding-vulkanfail-0540 cond=E7 policy=mme records=80 real=1 infra=79 incident_0step=79 hist=env_build|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:79 compared=excluded
INCIDENT=INFO queue=prod policy=mme infra_terminal=101 seats=None,new1,new2,new4 zero_step=101 incident=100 replaced=101 unreplaced=0 hist=env_build|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:100;n/a|认领后进程崩溃，无结果:1
OFFICIAL_SHARD_DONE run=O2 shard=3 smvla_rows=19 mme_rows=6 smvla_rc=0 mme_rc=1 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=4 smvla_rows=19 mme_rows=7 smvla_rc=0 mme_rc=1 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=7 smvla_rows=19 mme_rows=14 smvla_rc=0 mme_rc=1 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=8 smvla_rows=19 mme_rows=12 smvla_rc=0 mme_rc=1 staged=yes
```

| 受影响项 | 乘式 | 0 步失败 | 真实局 | 处置 |
|---|---|---:|---:|---|
| 5.1 E6 SMVLA（乙） | 1 策略 × 16 任务 × 1 档 × 3 局 = 48 | 69 | 7 | 整格重跑（`final3`，07:00～07:55，48/48、0 错误、0 基础设施失败） |
| 5.1 E7 MME（丁） | 1 × 16 × 1 × 3 = 48 | 79 | 1 | 整格重跑（`final2`，07:00:47 完成） |
| 5.2 MME 原队列 | 新1 27 ＋ 新2 29 ＋ 新4 48 次；最终 100 个身份以 infra 终态结束，另 1 个认领后崩溃 | 104 | — | 补跑队列 `prod2`（100）＋ `prod3`（1），合并 0 冲突 |
| 5.2 SMVLA 原队列 | — | 2 | — | 新1 2 次连接拒绝来自杀 server 测试（不是本事故，汇总脚本同样按 0 步基础设施失败计），队列内重试 |
| 2.3 O2 片 3、4、7、8 | SMVLA 4 片 × 19 局 = 76 | 76 | — | 首遍全错，第二遍 `--resume` 补齐 |
| 2.3 O2 片 3、4、7、8 MME | 4 片 × 19 局 = 76，事故时只跑了 6／7／14／12 局 | — | 39 | 按历史 `done_keys` 机制续跑（`official_mme_resume.sh`） |
| 2.3 O2 片 1、2（丁） | 20 ＋ 19 局 | — | — | 首次起跑被取消，在新1、新2 整片重跑（06:47:53、06:49:19 完成） |
| 2.3 O1 片 0、6 SMVLA | 6 ＋ 11 局 | 17 | — | 旧启动器自带 `--resume` 重试补齐 |

合计 0 步基础设施失败 = 2.3 的 93（O1 17 ＋ O2 76）＋ 5.1 的 148（69 ＋ 79）＋ 5.2 的 106（MME 104 ＋ SMVLA 2）= 347，与上方 `BUDGET … incident=347` 一致（部分数据阶段曾算作 349，终值汇总把金丝雀局的基础设施失败单列到 `CANARY_INFRA`——丁 SMVLA 与新4 MME 各 1 次 Vulkan 建环境失败、新4 MME 1 次录制目录非空，前两次原先计在 5.2 的 0 步失败里，故少 2）。

```text
CANARY_INFRA=INFO policy=smvla n=1 items=ding:N:PatternLock/650700:env_build
CANARY_INFRA=INFO policy=mme n=3 items=new4:N:PatternLock/650700:env_build;new4:N:PatternLock/650700:recorder;new1:K:PatternLock/650700:ConnectionClosed
```事实清单写的「O2 片 3／4／7／8 MME 仅 6～7 局」与日志不一致：片 3、4 为 6、7 局，片 7、8 为 14、12 局（`OFFICIAL_SHARD_DONE` 原文如上），以日志为准。

### 处置

1. **止损**：在每个受影响席位起一个永不结束的 `--gpu_cmode=shared` 保持步骤（`scripts/eval-official/v75-lanes/gl/keeper.sh`：先 `scancel` 点名的等待型作业步，再 `exec srun --jobid=$J --overlap --ntasks=1 --cpus-per-task=1 --gpu_cmode=shared --job-name=v75-keeper sleep infinity`），只要它不结束，卡就停在共享模式；登录节点 tmux `v75-keep-*`、`v75-keep2-*`，收尾随占位 job 一并取消。
2. **清除所有等待型作业步**，此后所有「等某步结束再开始」的等待都放在编排席本地（计划里 `L-wait*` 车道，`where=local`），不再占用 GPU 席位的作业步。
3. **留证**：失败尝试改名 `*-vulkanfail-<时刻>`（目录与报告同名），汇总排除并单列 `INCIDENT` 行（12.275）。
4. **补跑**：见上表；O2 片 1、2、3、4、7、8（其后又加片 0、6）改派到新1～新4 与丁，靠 `artifacts/v7.5eval/nfs-archive/skip/O2-s<n>.skip` 让原编排器跳过（本轮原话 7「尽可能并行」、原话 8「自己决策自己改」）。
5. **汇总脚本**：12.275 把 0 步基础设施失败计入各项重试并加 `retry_over` 标记，不再藏在「未超」里。

### 证据

`artifacts/v7.5eval/summary/verdicts.txt`（`INFRA`、`INCIDENT`、`BUDGET` 行）；`artifacts/v7.5eval/nfs-archive/reports/E6-smvla-s-yi.vulkanfail-0544.json`、`E7-mme-s-ding.vulkanfail-0540.json`；`artifacts/v7.5eval/nfs-archive/logs/keeper-{jia,yi,bing,ding}.log`；`artifacts/v7.5eval/nfs-archive/state/{main,mmeprod,o2x,retry}/events.log`（05:47～05:52 的 `STEP_FAIL`）；`artifacts/v7.5eval/nfs-archive/skip/`；12.275 commit body。

### 教训

- GL 上 `--gpu_cmode=shared` 是「作业步级」开关：**任何带它的作业步结束都会把整张卡重置为独占**，同卡上还在跑的其他作业步随后新建的 CUDA／Vulkan 上下文全部失败。同一席位上只允许一个长期存活的 shared 作业步兜底；查询、补暂存这类辅助 `srun` 不带 `--gpu_cmode`，或改在登录节点做。
- 不设计「在 srun 里等待」的作业步；等待一律放在编排器本地。
- Monitor 过滤词要覆盖 `createDeviceUnique` 与 `SPANK:gpu_cmode`；「跑得比预期快很多」本身就是故障信号（E7 MME 8 分钟跑完 48 局）。
- 已写入主会话记忆（`greatlakes-gpu-cmode-shared`）；是否同步进 `greatlakes.md` 与正本：需用户决定，本轮未改规则文件。

## 其他计划外事件（无局数影响）

- 新增 4 席首次提交 62618785／86／88／90 全落 gl1525（与 A40-乙同节点），未跑任何工作即 `scancel`，按 `--exclude=gl1525,gl1513,gl1527` 重交为 62618838（gl1512）、62618839／40／41（gl1528）；清单 `hold-jobs-envdet-20260929.txt` 已更新（12.267 commit body）。
- SimpleMemVLA 确定性开态起不来（`cumsum` 无确定性实现），按规则判关，见 [policy-replay.md](policy-replay.md)。
- 12.272 曾决定「O2 保持在甲、丙」以免节点效应混入噪声带，04:56 起为缩短关键路径改派到其他席位（本轮原话 7、8 授权），节点效应因此进入官方噪声带，见 [summary.md](summary.md) §5。
