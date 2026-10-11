# 五模型 OOD 原始录像恢复轮：启动记录

## 目的与授权

本轮名为 `ood-five-raw-seed7-20261010-02`，完整方案为 [恢复计划](../../plans/1010-five-models-ood-restart-plan.html)。用户原话按时间顺序保留：

1. 「监督器只要复发他自己的停止，不要对于GPU任务进行停止，CPU持续占位。」
2. 「批准完整恢复方案并启动」
3. 「独立邮件通知，先验证投递链路」

新轮五模型各 800 次首试。每模型为：7 任务 × 2 档 × 25 局 + 2 任务 × 4 档（13／13／12／12 局）+ 3 任务 × 3 档（17／17／16 局）+ 2 任务 × 5 档 × 10 局 + 2 任务 × 1 档 × 50 局 = 800 局。五模型合计 4000 新轨迹／8000 新 reset 计量；五模型 × 1 任务 VideoUnmask × 1 档 xhard1 × 1 首局冒烟计在此分母中。额外重试、递补、抽样为零。

旧轮 PP × 1 任务 VideoUnmask × 1 档 xhard1 × 1 首试已消耗 1 轨迹／2 reset 计量，原账本保留，不退还、不补写成功。两轮累计硬上限为 4001／8002。新轮以原字节副本绑定旧账本 SHA256 `fc294d2d4dfe5d3ad00b1cd93afe7d1e3e47989ba457b0241c0695b1b1131d74`，启动及最终再次核对；完整旧事实见旧轮档案。

## 资源与运行位置

CPU `63664373`：`chaijy2/standard`，1 CPU／4G／120 小时，节点 `gl3010`，开始 `2026-10-10 17:52:59`，结束 `2026-10-15 17:52:59 America/Detroit`。四 GPU：`63664391`、`63664392`、`63664393`、`63664404`，均 `chaijy2/spgpu`、1 GPU／1 CPU／24G／120 小时。前三在 `gl1505` 运行，第四提交后等待排队；每份只提交一次、零重提。实际启动时点另留 Slurm 原始回执。

GPU batch 为独立 `sleep infinity`，生命周期守卫只作为额外作业步。CPU 使用已冻结的持续占位 bootstrap，SHA256 `342402d8a8742eeaafabfa3e0e8e8d37945eeb0be0ff71c687e27a7641f560a6`；监督器运行一次，任何退出码之后均持续占位。异常不主动取消 GPU，不终止其已起工作负载；正常全部完成后仍释放 GPU，CPU 继续占位。没有使用提前两分钟终止 batch 的 Slurm 信号选项。

本机根 `/data/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02`；NFS 根 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02`。搬运会话名固定 `ev-ood-five-raw-mover-20261010-02`，启动成功另记回执。准备截止 epoch `1791755700`；发布 `launch.ready` 前 CPU 只等待，不启动策略。

## 通知与前置验证

收件人为 `hongzefu@umich.edu`。在同一 CPU 作业内执行独立邮件探针，先核工具、提交退出码，再核实际收到；发送工具退出零只代表提交，不能代替投递。计算节点已实查 `/usr/sbin/sendmail`、`/usr/bin/mailx` 存在，查询退出零。当前连接的 Gmail 账户不是目标邮箱，因此不以另一邮箱的搜索结果宣称目标已收到。

## 版本、配置与启动还原

工程基线为 `753a8121c64641f3ef54bba37844ced9d3a56c3d`，邮件与跨轮预算由独立实现代理补齐后固定审查、真实 CPU 短测、合并推送，再制作 clean 提交快照。正式执行提交、完整 JSON 配置 SHA256、五 gitlink、实际解释器来源、邮件探针及启动命令将在真正起跑前追加；尚未完成的门禁不能记为通过。

正式执行提交已固定为 `656c591f32238703c54967d0f550220cf0cec074`（1.107），本机／NFS 均使用本轮根下独立 `runtime-code/`，不随主仓文档提交变化。完整配置原字节 SHA256 为 `5cc29c377a607186a6a21b27791b02eec480b2fd7bd2f390ff64043a07f4c9b3`，保存 `records/run-config.json`；所有席位停止新领取的期限保守取 CPU EndTime 前两小时。五 gitlink 与旧轮完全一致，不初始化嵌套 benchmark，不改共享环境 `.pth`。

复制核验 `EXEC_COPY=PASS commit=656c591f32238703c54967d0f550220cf0cec074 gitlinks=5 files=2185`、`INPUTS_LOCK=PASS files=2185 gitlinks=5 identities=800 tokenizer=1 client_origins=3 server_origins=5 gpu_initializations=0`，两者退出 0，原始回执在 `records/`。仅新快照 Git 元数据采用已验证的 `core.filemode=false` 与 SimpleMemVLA 的 LFS cat 过滤配置，逐文件源码字节仍强核。第一次输入检查在 rsync 未结束时过早运行，缺文件退出 1；没有启动模型，保留 `records/check-inputs.log`，同步明确退出零后再核得到上述通过行，不覆盖失败记录。

CPU 内累计预算预检退出 0：`WORK_BUDGET=PASS`，prior 为 1 轨迹／2 reset，current 为 0／0，combined 为 1／2，上限为 4001／8002；完整计数与 SHA 在 `records/work-budget-preflight.log`。邮件探针一次提交退出 0：`NOTIFICATION state=submitted event=probe rc=0 delivery_verified=0`。这只证明计算节点 MTA 提交，实际目标邮箱收到仍待用户确认。探针回执、唯一主题、CPU／版本／配置身份均保存在 `records/notification-probe.json`。

实际探针命令（一次执行，不重发）：

```bash
timeout 60s ssh greatlakes 'srun --jobid=63664373 --overlap --exact --ntasks=1 --cpus-per-task=1 --gres=none /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/runtime-code/dev-scripts/gl/raw_ood_supervisor.py notification-probe --config /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/control/run-config.json'
```

B3 固定提交 `93f1475e37339dae90fae3306674593a87c7174a` 的独立静态审查：`RESTART_CONTROL_REVIEW=PASS`，审查原文在 `records/control-review.txt`。主会话整合后命令如下，退出 0，`118 passed, 2 deselected, 23 warnings in 35.12s`；两个未改的长守卫复现此前已通过，本次不重跑。`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`，完整原始输出在 `records/main-target-tests.log`。

```bash
UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 120s uv run --no-sync python -m pytest tests/unit/test_raw_ood_controller.py tests/unit/test_raw_ood_supervisor.py tests/unit/test_raw_ood_seat_drain.py tests/unit/test_raw_ood_mover.py tests/pipeline/eval/test_seat_runner_e2e.py -q -m 'not slow' -k 'not hanging_query and not old_five'
```

## 交接判据

必须先完成 `EXEC_COPY`、`INPUTS_LOCK`、`WORK_BUDGET`、`MAIL_DELIVERY`；随后五个正式首局通过执行与原始媒体核验，再观察首个正式分片真实合法动作一步。仅在 `RAW_SMOKE`、`FIRST_DISPATCH`、`AGENT_HANDOFF` 有原始证据后退出当前代理。任务成功字段独立报告，正常执行但任务失败不重试。

实际邮件确认后，用以下确定命令启动本机搬运并发布就绪。当前仅准备命令，尚未执行；发布器强核同版本／同 CPU 的用户实收确认、无错误／停止标记、有效搬运心跳与主仓 clean 状态，不把等待超时当确认。CPU bootstrap 随后校验就绪文件原始配置 SHA 并在自身 allocation 启动固定监督器。

```bash
tmux new-session -d -s ev-ood-five-raw-mover-20261010-02 'bash /data/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/mover-run.sh'
UV_CACHE_DIR=/home/hongzefu/.cache/uv uv run --no-sync python artifacts/ood-five-raw-seed7-20261010-02/publish_ready.py
```

## 实际接续的版本与入口

用户确认已收到探针后，原启动已执行；前三首局完成，首次控制器接续及随后人工监督失败均保留在result与records。用户追加：「可以再次启动这个CPU的任务。你可以随意地启动CPU的任务，不需要每次都问用户。」以及录像裁决：「只交付原始录像，允许内部临时 MP4」。前述准备命令仅还原首次启动，禁止重复发布旧ready或重起旧搬运。

后续worker快照为 `runtime-code-resume-01/ @8c3c84575e2a07727931c698b6ca17bedf495bac`，新增配置 `control/run-config-resume-01.json` 的SHA为 `bbe580d4948cd6e07b4ded82f633b0786b0d1d245c9dfeaa01b7de9b09e0f7f2`；其 `legacy_source` 绑定三个已完成首局的656／5cc，身份、预算、权重、解释器与原GPU作业不变。原冻结快照与配置禁止覆盖。

私有编排从已推送 `ae44271248931eb7e1aa9eb9e7ab93dd171b3db7` 导出五个控制文件至 `orchestration-recovery-02/`，逐文件SHA在运行根 `control/orchestration-lock-recovery-02.json`。新搬运会话为 `ev-ood-five-raw-mover-resume-20261010-02`；新CPU启动会话为 `ev-ood-five-raw-supervisor-resume-02-20261010-02`，日志为本机运行根 `control/supervisor-resume-02.log`。这两项是当前会话的精确清单，三个既有网站会话不属于本轮。

确定CPU恢复命令如下；外部detached tmux持有SSH/srun，wrapper仅执行一次监督器并在退出后继续hold，禁止自动重启监督器。手动停止CPU步骤仍会触发Slurm清理其子进程。

```bash
srun --jobid=63664373 --overlap --exact --ntasks=1 --cpus-per-task=1 --gres=none \
  bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/orchestration-recovery-02/raw_ood_supervisor_hold.sh \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02 \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/orchestration-recovery-02/raw_ood_supervisor.py run \
  --config /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/control/run-config-resume-01.json \
  --resume-completed-smokes --orchestration-commit ae44271248931eb7e1aa9eb9e7ab93dd171b3db7
```

CPU只读预检、实际新步骤／PID及交接判定回写result；这里只固定已提交源码和真实参数，不把命令准备视为运行成功。

## 最终真实接管入口

第二次恢复曾因排队作业查询超时自停，原三个formal继续；03／04／05均仅预检拒绝，未启动监督器。最后实际接管使用六文件 `orchestration-recovery-06/ @e39c6d4a407bb81f49f6f293a0be86b4f9a03cf3`，锁在 `records/orchestration-lock-recovery-06.json`，worker仍8c3／配置bbe。CPU只读 `existing-dispatches-preflight` 已通过，运行入口为 `--resume-existing-dispatches`，取代旧的三首局恢复flag。

本轮仍需保留的精确会话：`ev-ood-five-raw-mover-resume-20261010-02`、`ev-ood-five-raw-supervisor-resume-02-20261010-02`（旧CPUstep12保持原三srun所在cgroup）及 `ev-ood-five-raw-supervisor-resume-06-20261010-02`（新监督／控制器）。主连接也不能为了清理被关闭。实际一次启动命令：

```bash
tmux new-session -d -s ev-ood-five-raw-supervisor-resume-06-20261010-02 'bash /data/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/supervisor-resume-06-run.sh'
```

该载体在63664373同CPU作业内运行 `raw_ood_supervisor_hold.sh <NFS运行根> <NFS主解释器> <NFS运行根>/orchestration-recovery-06/raw_ood_supervisor.py run --config <NFS运行根>/control/run-config-resume-01.json --resume-existing-dispatches --orchestration-commit e39c6d4a407bb81f49f6f293a0be86b4f9a03cf3`。新监督器734024／控制器734113，原3worker PID不改。真实首步和心跳已取得；最后Slurm刷新仍待SSH可新开会话，详见result的具名判定。未建立代理自动唤醒。

## 五轮审查后的最终接续10

上述06为历史载体。审查后07正式恢复运行的完整worker快照为 `33bef4f485ca95b6c43017369220eec7eb7fd668`，配置 `control/run-config-resume-02.json` SHA `6eb5cdcdf6c0de32df2323f28231b9b25278377be8cac677dce87f186737028e`；原5cc／bbe配置及60已尝试身份保留复用。源清单 `control/source-contexts-resume-02.json` 锁两个旧配置原字节，独立暂停证明 `control/pause-snapshot-resume-02.json` 锁原STOP及三条排空分片。来源字段不重标，现有模型进程不重启。

控制器后来因共享盘errno121自行退出。第五轮审查结束后按用户限额完成第六次最终修复，编排锚 `65a852bafecba5454529e40c90d93d8168dc27d8`，七文件闭包位于 `orchestration-recovery-10/`，清单 `control/orchestration-lock-recovery-10.json`。该束仅执行CPU监督和控制，模型与搬运器仍用完整33bef4f快照；不从这个七文件束执行真实搬运。CPU实际在途预检退出0：`EXISTING_DISPATCHES_PREFLIGHT=PASS accepted=78 active=4 drained=3 published=65 remaining=3922`，确认旧三条暂停历史和当前四条RUNNING可接管，未新增尝试。

资源始终为CPU63664373及GPU63664391／63664392／63664393／63664404，各120小时获批配置不变；新控制器接管原步骤63664391.10／63664392.8／63664393.7／63664404.1，不新起模型，不重新申请资源。实际10启动命令（变量仅为书写缩写，均给出绝对值）：

```bash
TASK_NFS_ROOT=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02
TASK_CODE="$TASK_NFS_ROOT/orchestration-recovery-10"
ssh -o ControlPath=/home/hongzefu/.ssh/cm2-%r@%h:%p -o ControlMaster=no -o BatchMode=yes greatlakes \
  srun --jobid=63664373 --overlap --exact --ntasks=1 --cpus-per-task=1 --gres=none \
  bash "$TASK_CODE/raw_ood_supervisor_hold.sh" "$TASK_NFS_ROOT" \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python \
  "$TASK_CODE/raw_ood_supervisor.py" run \
  --config "$TASK_NFS_ROOT/control/run-config-resume-02.json" \
  --resume-existing-dispatches --orchestration-commit 65a852bafecba5454529e40c90d93d8168dc27d8
```

外部持有该命令的detached tmux为 `ev-ood-five-raw-supervisor-resume-10-20261010-02`，日志本机运行根 `control/supervisor-resume-10.log`；独立搬运为 `ev-ood-five-raw-mover-resume-02-20261010-02`，日志 `control/mover-resume-02.log`。监督异常只停止监督器自身，GPU不中止，CPU wrapper持续占位，不自动重启。通知失败／完成按每个监督实例独立留原字节，原邮件探针与用户实收证明不修改。

本轮新增但未清理的准确会话另有 `ev-ood-five-raw-supervisor-resume-07-20261010-02`、`ev-ood-five-raw-supervisor-resume-08-20261010-02`、`ev-ood-five-raw-supervisor-resume-09-20261010-02`；它们仅保留各自CPU wrapper。前述02／06及用户站点会话未动，不关闭SSH master，不使用全局清理命令。完整进展及验收见result；代理自动唤醒未建立。

## 十月十一日完整对抗修复后的接续十四

用户要求修复后再做完整子代理对抗审查，覆盖成功、失败、超时和正常接续；继续原作业，不停止 GPU，CPU 持续占位。三路最终审查与五百 CPU 用例锚定 `891b462726bf018c439a2fce9657aa85199d9f77`；实际 GL 心跳协议的最后修复仅更新 CPU 编排器，锚为 `75b722f0016f9dbb22e10440d5d6466ba190d865`。完整过程、拒绝反例和动态边界见 [本轮复核记录](1011-adversarial-recovery.md)，不将代码测试写成全量模型评估完成。

模型与真实搬运继续用完整 `runtime-code-resume-03/ @891b462726bf018c439a2fce9657aa85199d9f77`；两端逐文件字节核验二千二百八十五项、五个 gitlink，身份每模型八百，实际 Python 包来源核验三种客户端解释器和五个服务器。NFS 权限差异仅在新副本的 Git 元数据中校正，六仓状态为空，模型源码和权重不改。新配置 `control/run-config-resume-03.json` SHA 为 `cecdacdd07ef98958632efe91b2f84234ec73b5c4b4e5d2d15d968fee78cda02`；新来源清单锁旧 5cc、bbe、6eb 三份原配置。旧源码和配置均不覆盖，已尝试的超时身份不重跑。

CPU 编排七文件闭包为 `orchestration-recovery-14/`，锁为 `control/orchestration-lock-recovery-14.json`，其中明确 worker 仍为 891、配置仍为 cecd；不从这个束执行模型或真实搬运。接续十四真实预检退出零，识别四个在途步骤 `63664391.10`、`63664392.8`、`63664393.14`、`63664404.8`，没有重新启动四个模型进程。实际 CPU 步骤 `63664373.37`，监督器 `1931600`、控制器 `1931859`。资源仍为原 CPU 63664373 和四 GPU 63664391／63664392／63664393／63664404，各一百二十小时，不新增 job。

启动命令由本机 detached 会话 `ev-ood-five-raw-supervisor-resume-14-20261010-02` 持有，日志本机本轮 `control/supervisor-resume-14.log`：

```bash
ssh -o ControlPath=/home/hongzefu/.ssh/cm2-%r@%h:%p -o ControlMaster=no -o BatchMode=yes greatlakes \
  srun --jobid=63664373 --overlap --exact --ntasks=1 --cpus-per-task=1 --gres=none \
  bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/orchestration-recovery-14/raw_ood_supervisor_hold.sh \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02 \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/.venv/bin/python \
  /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/orchestration-recovery-14/raw_ood_supervisor.py run \
  --config /nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02/control/run-config-resume-03.json \
  --resume-existing-dispatches --orchestration-commit 75b722f0016f9dbb22e10440d5d6466ba190d865
```

唯一实际搬运器会话为 `ev-ood-five-raw-mover-resume-03-20261010-02`，使用本机完整 891 的 `dev-scripts/gl/eval_video_mover.py --layout raw-ood --stage <本轮NFS根>/gl --dest <本轮本机根>/gl --manifest <本轮NFS根>/control/run-config-resume-03.json --interval 30 --min-free-gib 50`，心跳、错误和 STOP 槽继续原运行根。旧 mover02 只按清单精确删除该会话，删前后差集恰一项，原 PID 已消失；没有第二个搬运器并发。

这次另登记的会话 `ev-ood-five-raw-supervisor-resume-11-20261010-02`、`ev-ood-five-raw-supervisor-resume-12-20261010-02`、`ev-ood-five-raw-supervisor-resume-13-20261010-02` 保留各自 CPU hold。独立验证会话 `ev-ood-failure-final-validation-1011` 已自然结束。准备和加载失败原证据分别在 `control/resume-11-startup-failure/`、`resume-12-load-failure/`、`resume-13-heartbeat-failure/`；模型加载前拒绝的一次重载没有任何预约或 reset，原目录已原字节移存。未清理用户站点或其他会话，未关闭 SSH master。GPU／CPU 不自动取消；失败通知沿已实收探针确认的 sendmail 通道，新失败的发送零退出仅记提交成功。

## 简化续跑（2026-10-11，最简四卡方案）

按根目录计划 [`1011-ood-simple-four-gpu-plan.html`](../../../1011-ood-simple-four-gpu-plan.html) 执行。用户原话（2026-10-11）：「codex 把问题搞得太复杂了」「我只是我还是想eval这些内容 规模不变 给出一个最简单的方式我需要分配在4个GPU上」「同意授权 写入计划」「起跑的时候不提交备用job」「NFS 大文件清理是跑完单独确认」「设计一个起跑的opus的监控它需要做最小化的监控每过几个小时去唤醒他一次」；开工令「开始 …1011-ood-simple-four-gpu-plan.html 一路修改修改完了开始长期eval」；排空期间「同意stop文件」「同意清理你来做」。

### 口径（编号同计划第一部分 §二）

1. 规模不变：5 模型 × 16 任务 × 50 局 = 4000 首试，每局 2 次 reset 计量 = 8000；旧轮 PP 1 局另计，原上限 4001／8002。已完成的局原样复用。
2. 参数逐字取自 `control/run-config-resume-03.json`（SHA256 `cecdacdd…cda02`），执行副本仍为 NFS `runtime-code-resume-03/`（exec_commit `891b462726bf018c439a2fce9657aa85199d9f77`）。
3. 只交付原始录像：`--no-render` + `--cfg groundsg_keep_official=false`。
4. 正常失败不重跑，基础设施失败零重试；已写 result.json 的 infra 局续跑自动跳过。
5. 占位 job 不停；6. 不含 Astra、MemER；7. PP 上下文溢出由汇总脚本重分类为模型失败 0，并给出重分类前后两组数字。
8. 中断重跑授权：只覆盖被基础设施中断、没有 result.json 的局；每分片封顶 3 局，四分片合计 12，累计上限 4013 轨迹／8026 reset。

### 旧编排排空（S1）

起跑前实测（15:55 EDT）：旧控制器早已崩溃，四个 GPU job 只剩一个 worker 步 `63664404.8`（qwenvl-03），其余三卡空占。15:56:39 建 `control/STOP`（用户同意）；该 worker 跑完在跑局 `ButtonUnmaskSwap_xhard2_18701903`（status=timeout，infra=0）后 15:57:41 以 `GL_SEAT_DONE outcome=fail rc=6` 自行退出；搬运器 `ev-ood-five-raw-mover-resume-03-20261010-02` 见 STOP 以 6 自退（`MOVER_PAUSED=PASS moved=361`），会话自然消失。本机 10 个旧监督器会话 `ev-ood-five-raw-supervisor-resume-{02,06,07,08,09,10,11,12,13,14}-20261010-02` 逐个按精确名结束，每次删前删后 diff 只少目标；用户站点会话与 `claude` 未动。CPU job 63664373 残留 bash 步 `.12 .20 .24 .25 .26 .29 .31 .33 .34 .37` 经用户同意按步号逐个 `scancel`，只剩 batch/extern。五个 job 均未取消。排空时 NFS 已完成：qwenvl 278、pp 100、smvla 23、oracle 1、framesamp 1，无半截目录。

`OLD_DRAINED=PASS gpu_steps=0 inflight_done=1 cpu_steps=0`

### 分片与启动命令

| 分片 | 任务 | 端口基 | GPU job | tmux（gl-login1） |
|---|---|---|---|---|
| g1 | PickHighlight, SwingXtimes, StopCube, VideoUnmaskSwap | 22400 | 63664391 | `ev-ood-simple-g1` |
| g2 | BinFill, VideoUnmask, ButtonUnmaskSwap, VideoRepick | 22500 | 63664392 | `ev-ood-simple-g2` |
| g3 | MoveCube, PickXtimes, RouteStick, VideoPlaceButton | 22600 | 63664393 | `ev-ood-simple-g3` |
| g4 | InsertPeg, ButtonUnmask, PatternLock, VideoPlaceOrder | 22700 | 63664404 | `ev-ood-simple-g4` |

分片脚本 `dev-scripts/gl/simple/run_shard.sh`（sha256 `f674e7837abb482ee56d3d58211ff71cccf45dded1bca0d74f27514010426ce2`；计划正文记的 `0daffdf…` 是 1.140 加行尾「历史目录名」注释前的版本）复制到 NFS `simple/run_shard.sh`。每个分片的启动命令（gN／job 按上表替换），在 gl-login1 执行：

```bash
tmux new-session -d -s ev-ood-simple-g1 "srun --jobid=63664391 --job-name=ood-simple-g1 --overlap --exact --ntasks=1 --cpus-per-task=1 --gpu_cmode=shared bash $N/simple/run_shard.sh g1 > $N/simple/logs/srun-g1.log 2>&1; echo SRUN_EXIT=\$? >> $N/simple/logs/srun-g1.log"
```

其中 `$N=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-02`。日志：`$N/simple/logs/shard-gN.log`、`srun-gN.log`、`reattempts-gN.log`、`restarts.log`；半截目录移存 `$N/simple/halfdone/gN/`。g1 先起即冒烟，首个 infra=false 新结果出现后再起 g2～g4。

### 回执（起跑后追加）
