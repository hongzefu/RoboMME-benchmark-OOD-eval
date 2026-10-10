# 五模型全任务 OOD 原始录像：准备与交接状态

本轮真实启动后在五模型冒烟完成前因生命周期守卫退出8而停止：完整结果0、首正式分片未启动。四GPU已自动取消、CPU已退出1；原始行为按当时已批准方案记录，不改写成后续用户要求的“只停止监督器、CPU持续占位”。

## 工程与静态审查

搬运子任务两次提交保留，固定 `30c03dc88c9826cbab50e3da0dbcd6e5c384292a` 的独立静态结论为 `RAW_MOVER_STATIC_REVIEW=PASS`。控制／监督子任务四次提交保留，固定 `4f3cc20b9513294674e17ecb5f9f31b7d3dd23c1` 的独立静态结论为 `CONTROL_REVIEW=PASS`；二者基线均为 `2d2cc25e195294ebd9f980f1bbc864e894fb7dfe`，静态审查未执行脚本。

已修复独立审查发现的真实缺口：录像收尾基础设施失败、双路共同缺帧、跨模型同键错配、已落地副本删除恢复；专属心跳锁初始化位置、心跳临时文件并发写、正常步数上限下计时为空、参数连字符／下划线别名绕过固定路由。旧快照查询可挂起的限制经用户一次裁决，通过同作业内追加独立守卫补强，未改已提交快照字节、未重提资源。

## 主会话真实 CPU 验证

```bash
UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 150s uv run --no-sync python -m pytest tests/unit/test_raw_ood_controller.py tests/unit/test_raw_ood_supervisor.py tests/unit/test_raw_ood_seat_drain.py tests/unit/test_raw_ood_mover.py -q
```

退出 0，`43 passed in 55.22s`；具名判定原文：`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。日志位于 `artifacts/ood-five-raw-seed7-20261010-01/control/targeted-tests.log`，末行 `EXIT_CODE=0`。包含真实 AV1 编码与全解码、生产格式 JSON 往返、正常任务失败、挂死 Slurm 查询的超时、原始守卫正常释放、同作业多步号、真实席位构造及并发心跳。

```bash
UV_CACHE_DIR=/home/hongzefu/.cache/uv uv run --no-sync python dev-scripts/gl/raw_ood_supervisor.py self-test --root artifacts/ood-five-raw-seed7-20261010-01/tests/main-detached-fixture
```

退出 0，`DETACHED_FLOW=PASS success=1 global_stop=1 cpu_lost=1 mover_lost=1 parent_exit=1 simulated_slurm=1`。夹具确实起独立进程并经 JSON 写后读回；其中两条预期 `RAW_SUPERVISOR=FAIL` 分别为控制器崩溃和退出 7，属于被验证的失败分支。Slurm 由测试替身提供，此结果不等于集群真实取消已验证。

## 全量核心短测

实跑仓库规定的 `timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q`，退出 0：`1733 passed, 6 skipped, 110 deselected, 23 warnings in 252.13s`，用时 4 分 12 秒，未超过 5 分钟。资源判定 `TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=6`。六项跳过按“未验证”保留，不称全部用例均已执行；110 项慢测未运行。日志位于本轮 `control/core-tests.log`，末行 `EXIT_CODE=0`。

另由已选 QwenVL 解释器真实导入 `flash_attn` 与 `torch`，退出 0：`QWEN_DEPENDENCIES=PASS flash_attn=2.8.3 torch=2.9.1 swift=3.11.1 transformers=4.57.3 gpu_initialized=False`。此检查不加载权重、不创建环境，日志位于 `control/qwen-dependencies.log`。

## 真实运行与未验证项

执行代码固定为 `3b8d6be938002d00bf1b40e9b6a771844bdf4a15`，本机及 NFS 独立快照不跟随开发仓文档更新。`EXEC_COPY=PASS commit=3b8d6be938002d00bf1b40e9b6a771844bdf4a15 gitlinks=5 files=2157`；NFS 逐文件字节、固定提交、身份 SHA、tokenizer 和真实包来源复核退出 0：`INPUTS_LOCK=PASS files=2157 gitlinks=5 identities=800 tokenizer=1 client_origins=3 server_origins=5 gpu_initializations=0`。配置原字节 SHA 为 `a289e5f3e19c6b6f4d86411155ccf99ea3f393b34b9042025a5f9d4a77c06e2f`，恢复配置用本目录 `records/run-config.json`。

第一次 NFS 前置核验没有放行：复制字节已相等，但文件呈现为可执行模式，Git 状态检查报脏。仅在各新快照配置 `core.filemode=false`，继续保留逐文件 SHA 和固定提交检查；复核通过日志与失败日志都留在 records。SimpleMemVLA 的非指针文本触发 LFS 过滤伪差异同样经 Git blob 比较后只调整新副本 Git 配置，模型源码未变。

资源登记为 `GL_ALLOCATION=PASS gpu_jobs=4 cpu_jobs=1 hours_each=120 resubmits=0`，原始 `scontrol` 记录和五份提交行在本轮 `control/allocation.txt`。这只证明资源申请被接纳，不证明五模型已经运行。

计划冒烟为 5 模型 × 1 任务 VideoUnmask × 1 档 xhard1 × 1 局，包含于 5 模型 × 800 身份 = 4000 次首试。CPU实际夹具通过后于本轮17:10左右启动CPU监督器、三席guard与三模型加载，第四席仍PENDING。五模型完整冒烟和首次正式分配未达到；`RAW_SMOKE=FAIL completed=0 expected=5`、`FIRST_DISPATCH=FAIL shard_started=0`、`AGENT_HANDOFF=FAIL reason=guard_exit`，不把服务器ready当作合法动作已经执行。

## 实际故障与预算

原始 `sacct` 中 `63661005.0` 为 `FAILED ExitCode=8:0`，对应新guard中解释器启动／读取／验证的唯一退出8分支；`guard-63661005.log` 给出四次 `GUARD_CANCEL rc=0` 与 `EXIT_CODE=8`。其他guard的退出141／Broken pipe发生在同一秒取消后，属于次生失败。CPU `failure.json` 捕获的是次生141，不能以该文件替代首因追踪。

原脚本没有保留内部 `payload_rc` 和阶段，所以目前只能定位上述分支，不能在解释器启动超时、NFS读取、JSON／身份断言、120秒心跳之间确定唯一原因。CPU最后心跳距取消约0.27秒，但缺少历史读回版本，不以该单点排除曾读到陈旧心跳。后续修复将标准库启动与5秒读取验证分离、保存第一错误阶段和原始返回码；此修复尚未获新资源实跑验证。

实际消耗：1模型PP × 1任务VideoUnmask × 1档xhard1 × 1次首试 = 1轨迹预约；build／reset各1次计量，共2。原账本报告：`BUDGET_ENFORCEMENT=PASS trajectories=1/4000 resets=2/8000 astra=0/2 shared_infra=0/0`，`committed=0 open=1`。部分轨迹没有任何step或end记录，完整 `result.json` 为0；该中断不作正常任务成功率分母，也不伪造 `task_success=0` 的完整结果。原账本保留SHA256 `fc294d2d4dfe5d3ad00b1cd93afe7d1e3e47989ba457b0241c0695b1b1131d74`，不退回或重新初始化额度。

CPU `63660978` 于2026-10-10 17:13:00 America/Detroit退出 `FAILED 1:0`；四GPU于17:12:54取消，其中第四席从未运行，`squeue` 已无本轮活动作业。清理回执为四次取消退出0、最终active为空；`STOPPED_RESOURCES=PASS gpu_terminal=4 cpu_failed=1 active=0`。本机搬运读取全局STOP后退出1，唯一tmux会话已不存在，`CONTROL_MIRROR_EXIT_CODE=0`。原始部分gl文件复制到本机 `artifacts/ood-five-raw-seed7-20261010-01/stopped-gl/`，NFS原件保留，不能称完整RAW_DELIVERY通过。

## 用户追加决策

1. 初始指令：「/data/hongzefu/RoboMME-benchmark-OOD-eval/docs/plans/1010-five-models-ood-raw-only-plan.html 开始实现 有问题问用户」。
2. 守卫查询挂起的补强选择：「采用，同一批作业内追加独立守卫」。
3. 故障后新指令：「监督器主复发他自己的停止，不要对于GPU任务进行停止，CPU持续占位。」
4. 再次强调：「监督器只要复发他自己的停止，不要对于GPU任务进行停止，CPU持续占位。」新实现按“只退出监督器，不主动终止controller/GPU，CPU持续hold”执行；已取消的旧JobID不可恢复，不自行申请新资源。

## 新口径的工程验证（不改旧运行事实）

搬运侧固定 `e98b0e9eafdab1c955451cf1ea293ca638fb2cbc` 静态通过；控制侧固定 `2b75b977d9c978cd6ee892d7f19b9e95d0edd4ea` 的 `CONTROL_STOP_REVIEW=PASS`。监督器异常只记录首因与自身停机，不写全局STOP、不杀控制器／guard、不取消GPU；控制器异常不主动终止已启动GPU step；CPUbootstrap一次执行监督器并在其退出0或1后继续hold，不自动重起。GPU batch的新提交模板使用独立 `sleep infinity`，guard只是额外srun step，退出自身不结束allocation；未申请实际新作业。

主会话集成定向退出0：`58 passed in 82.09s`，`TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0`。真实独立进程夹具退出0：`STOP_ISOLATION=PASS supervisor_exit=1 controller_alive=1 guard_self_exit=1 gpu_cancel=0 global_stop=0 parent_exit=1 notification_established=0 simulated_slurm=1`，确实保留其他测试子PID继续存活，不用单纯mock的“未调用kill”替代存活证明。

新全量命令在统计 `1748 passed, 6 skipped, 110 deselected in 278.20s` 后，进程收尾被280秒截止，真实退出码124，不能判门禁通过。按仓库“超时选核心最小子集”规则分两份真实运行：本轮58项退出0；其余 `timeout 280s uv run --no-sync python -m pytest -m 'not slow' --ignore=tests/unit -q` 退出0，`1690 passed, 6 skipped, 110 deselected in 196.31s`。两份合计覆盖完整非慢测的1748个通过用例，6项仍未验证；`CORE_SPLIT=PASS scope_tests=58 remainder_tests=1690 exits=0,0 native_reset=0 gpu_init=0 violations=0 not_verified=6`。不放宽280秒上限、不把超时日志改写为成功；三份日志均保留。

守卫新标准库检查把解释器启动与5秒读取分离，保留120秒心跳判据，并记录第一错误phase／payload返回码；新正常收官保留初始清理回执，另读最终GPU状态，非空或未知不判通过。此工程验证不等于本轮真实故障的具体内部原因已证实。CPU持续占位时原Slurm FAIL邮件不触发，通知字段明确为未建立；独立通知渠道与新一轮资源／累计4001轨迹、8002reset仍待用户裁定。

## 原始记录清洗说明

1.103 归档的 Slurm 文本末尾带一个空行，差异检查未正确阻断该次提交；本次仅去掉归档末尾空行，重新检查通过，原始 `artifacts/control/allocation.txt` 字节保留。工程测试结果与冻结执行源码不受此文档修正影响。
