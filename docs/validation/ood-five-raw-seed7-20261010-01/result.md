# 五模型全任务 OOD 原始录像：准备与交接状态

本档案当前记录工程准备，尚未把真实五模型冒烟、首次正式分片或完整评估写成通过。后续权威运行终态由冻结程序在本轮 `artifacts` 下生成，不能从作业存活推断。

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

资源登记为 `GL_ALLOCATION=PASS gpu_jobs=4 cpu_jobs=1 hours_each=120 resubmits=0`，原始 `scontrol` 记录和五份提交行在本轮 `control/allocation.txt`。这只证明资源申请被接纳，不证明五模型已经运行。

真实冒烟为 5 模型 × 1 任务 VideoUnmask × 1 档 xhard1 × 1 局，包含于 5 模型 × 800 身份 = 4000 次首试；目前尚未启动。`RAW_SMOKE`、`FIRST_DISPATCH`、`AGENT_HANDOFF`、`RUN_BUDGET`、`OOD_RESULTS`、`RAW_DELIVERY` 与 `RUN_CLEANUP` 均待真实运行证据，禁止提前填入通过。

## 用户追加决策

1. 初始指令：「/data/hongzefu/RoboMME-benchmark-OOD-eval/docs/plans/1010-five-models-ood-raw-only-plan.html 开始实现 有问题问用户」。
2. 守卫查询挂起的补强选择：「采用，同一批作业内追加独立守卫」。
