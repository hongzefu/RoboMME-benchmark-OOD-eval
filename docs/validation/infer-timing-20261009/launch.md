# 推理测速改造与验证：启动记录

## 用户决策与范围

1. 用户原话：「/data/hongzefu/RoboMME-benchmark-OOD-eval/docs/plans/1009-infer-timing-plan.html 开始实现 有问题问用户 但尽可能不要阻塞」。实施范围为该计划第四稿的 S6、S1、S2、S2b、S3、S4、S5 与集成验证。
2. 用户补充：「gl」「gl已经占用的卡可以直接用」。复用既有 GL 占位作业，不申请新席位，不在结束时取消这些既有占位作业。
3. 完整预算问题答复：「批准：最多 54 次 reset、27 次轨迹尝试」。原定 21 次尝试各计构建与 reset 两次，42 次；六个 GL 正式身份各最多一次基础设施重试，额外最多 12 次，总上限 54 次。正常任务失败不重试，Astra 不追加局。
4. 用户纠偏：「本局的冒烟是什么意思？是要跑完吗？不用跑完。」本机审计开／关冒烟各取得一个合法动作块、执行一步后停止；只验证链路，不用于正式稳态速度结论。GL 正式每变体一局的范围保持。
5. Astra 守卫局数裁决先为「Astra 只做一次短冒烟和一次正式，共 2 次」，随后用户更新为「astra同意做3次 放宽到5美元」。以后者为准：两次短冒烟与一次正式共三次、共享同一 5 美元账本；增加显式局数参数，本轮传 3，默认仍为 2，不重置旧账本额度。

## 版本与代码状态

- 实施起点：`210183625a5c073eb9d3557f6792d0cb912eb5f3`，主检出 `dev`，提交编号从 `1.49` 接续。
- 工作区既有改动：`third_party/SimpleMemVLA` 内 16 个描述文件有未提交修改。本轮保留、绕开，不暂存、不回滚、不将其混入固定提交审查；正式评估使用干净的固定版本副本。
- 五个第三方 gitlink、受保护目录、`pyproject.toml` 与 `uv.lock` 冻结；公开清单中的源码按项目例外使用英文。
- 环境：`sled-vail`，2 张 RTX 6000 Ada，共享盘与 SSH 配置存在，`micromamba` 与 `uv` 可用。Codex 子代理并发配置为 16。
- 后续核验澄清：SimpleMemVLA 的 16 个描述文件实际与其锁定提交原始 blob 逐字节一致，所谓在途 diff 是 LFS clean 过滤器将普通文本转为指针造成。主检出仍原样保留。另建本轮干净运行工作树，只在该工作树子模块的 `.git/info/attributes` 对这 16 个明确文本路径取消过滤器；源码字节、`.gitattributes` 和五个 gitlink 全部不改。判定：`SUBMODULE_SOURCE_BYTES=PASS files=16 original_matches_git=1 snapshot_matches_git=1 cause=lfs_clean_filter`。

## 运行位置与资源归属

- 本机工作副本：`/data/hongzefu/RoboMME-benchmark-OOD-eval`。
- GL 副本：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval`，仅本机同步，GL 侧不做 git 操作。
- 用户授权复用的既有占位作业：`63431430`（gl1525）、`63431431`（gl1512）、`63431432`（gl1512）、`63431433`（gl1513）。本会话未提交这些作业，禁止在收尾取消。
- 本轮 tmux 会话：尚未启动；启动前逐个登记完整名称、日志与命令。
- 本机运行副本：`artifacts/worktrees/infer-timing-runtime`，五个子模块均从本机既有对象按 gitlink 初始化，不递归初始化；正式起跑前快进到最终实现提交并复核 clean HEAD。本机非 Astra 使用 `CUDA_VISIBLE_DEVICES=1` 且省略 `--gpus`，使客户端与服务子进程共同继承物理 GPU 1；Astra 使用原有双卡 `--gpus 1,0`。GPU 0 的既有 738 MiB 进程不清理、不干预。

## 配置与验证口径

- 权威延迟为每次 `ActionChunk` 更新的 `decision_wall_ms`，包含子目标、规划器、监视器及上传等待；服务端与语言段分别记录。
- 本机链路冒烟：7 变体 × 1 任务（VideoUnmask）× 1 档 × 2 次（审计开、关各一次）= 14 次短尝试；每次只取一个动作块并执行一步，不跑完整局。
- GL 正式：6 GPU 变体 × 1 任务（VideoUnmask）× 1 档（xhard0）× 1 局 = 6 局；Astra 正式：1 变体 × 1 任务 × 1 档 × 1 局 = 1 局，在本机运行。
- Astra 三次尝试共享 5 美元上限，使用本轮唯一费用账本；不通过分别建账本绕过累计额度。旧任务账本保持原样。
- 六个 GL 正式身份各最多一次基础设施重试；正常任务失败不重试。累计硬上限为 54 次 reset、27 次轨迹尝试。模型缓冲 reset 与真实环境 reset 分别核对，不增加预热环境局。
- 短测入口：`UV_CACHE_DIR=/home/hongzefu/.cache/uv timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q`；正式命令、权重、启动提交与退出状态在实际起跑前补齐。
- 已导出单行身份：`VideoUnmask_xhard0_560300`，`builder_episode=0`、`source_episode=3`、`seed=560300`，两个空身份字段为 null；`identities.jsonl` 的 sha256 为 `48ced718820688e271b60e5d8679c05c621888428d78a47625ead5fe72d8a1c7`。此 seed 是环境身份；本机 policy_seed=0，GL policy_seed=7，不混淆二者。
- 短冒烟停止控制在本轮临时入口 `artifacts/infer-timing-20261009/smoke_once.py`：保留 `spec.max_steps=1300` 的正式配对，只在评估仓外层会话拦截额外执行；首次合法动作块之后最多执行一步，审计关路径最多一步。`SINGLE_CHUNK_STOP_FIXTURE=PASS audit_on=1 audit_off=1 hold_prefix=1 extra_env_step=0`。正式 Astra 通过同入口的 `--it-formal` 仅登记累计预算，不施加短冒烟停止。

## 留档边界

日志与临时工作树在本仓 `artifacts/infer-timing-20261009/`、`artifacts/worktrees/`；长期证据在本目录 `records/`。不归档权重、配置副本或独立启动脚本。验收失败保留原始证据，不放宽判据。
