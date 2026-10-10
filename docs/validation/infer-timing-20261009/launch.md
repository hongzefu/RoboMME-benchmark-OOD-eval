# 推理测速改造与验证：启动记录

## 用户决策与范围

1. 用户原话：「/data/hongzefu/RoboMME-benchmark-OOD-eval/docs/plans/1009-infer-timing-plan.html 开始实现 有问题问用户 但尽可能不要阻塞」。实施范围为该计划第四稿的 S6、S1、S2、S2b、S3、S4、S5 与集成验证。
2. 用户补充：「gl」「gl已经占用的卡可以直接用」。复用既有 GL 占位作业，不申请新席位，不在结束时取消这些既有占位作业。

## 版本与代码状态

- 实施起点：`210183625a5c073eb9d3557f6792d0cb912eb5f3`，主检出 `dev`，提交编号从 `1.49` 接续。
- 工作区既有改动：`third_party/SimpleMemVLA` 内 16 个描述文件有未提交修改。本轮保留、绕开，不暂存、不回滚、不将其混入固定提交审查；正式评估使用干净的固定版本副本。
- 五个第三方 gitlink、受保护目录、`pyproject.toml` 与 `uv.lock` 冻结；公开清单中的源码按项目例外使用英文。
- 环境：`sled-vail`，2 张 RTX 6000 Ada，共享盘与 SSH 配置存在，`micromamba` 与 `uv` 可用。Codex 子代理并发配置为 16。

## 运行位置与资源归属

- 本机工作副本：`/data/hongzefu/RoboMME-benchmark-OOD-eval`。
- GL 副本：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval`，仅本机同步，GL 侧不做 git 操作。
- 用户授权复用的既有占位作业：`63431430`（gl1525）、`63431431`（gl1512）、`63431432`（gl1512）、`63431433`（gl1513）。本会话未提交这些作业，禁止在收尾取消。
- 本轮 tmux 会话：尚未启动；启动前逐个登记完整名称、日志与命令。

## 配置与验证口径

- 权威延迟为每次 `ActionChunk` 更新的 `decision_wall_ms`，包含子目标、规划器、监视器及上传等待；服务端与语言段分别记录。
- 本机链路冒烟：7 变体 × 1 任务（VideoUnmask）× 1 档 × 2 局（审计开、关各一局）= 14 局。
- GL 正式：6 GPU 变体 × 1 任务（VideoUnmask）× 1 档（xhard0）× 1 局 = 6 局；Astra 正式：1 变体 × 1 任务 × 1 档 × 1 局 = 1 局，在本机运行。
- Astra 三局共享 3 美元上限，沿用同一费用账本；不通过新建账本绕过额度。
- 基础设施重试按计划至多一次，正常任务失败不重试。完整 reset 与轨迹预算须在起跑前盘点，包括预热。
- 短测入口：`UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q`；正式命令、权重、启动提交与退出状态在实际起跑前补齐。

## 留档边界

日志与临时工作树在本仓 `artifacts/infer-timing-20261009/`、`artifacts/worktrees/`；长期证据在本目录 `records/`。不归档权重、配置副本或独立启动脚本。验收失败保留原始证据，不放宽判据。
