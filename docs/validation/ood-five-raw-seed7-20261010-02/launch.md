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
