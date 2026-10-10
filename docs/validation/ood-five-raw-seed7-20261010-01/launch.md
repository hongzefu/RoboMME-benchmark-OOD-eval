# 五模型全任务 OOD 原始录像：启动记录

## 目的与用户授权

用户原话：「/data/hongzefu/RoboMME-benchmark-OOD-eval/docs/plans/1010-five-models-ood-raw-only-plan.html 开始实现 有问题问用户」。本轮按该完整计划执行，五个真实模型，仅 OOD，策略种子 7，不制作官方视频；原始 AV1 双视角、动作、轨迹与结果异步搬回本机。

每模型规模：7 任务 × 2 档 × 25 局 + 2 任务 × 4 档（13／13／12／12 局）+ 3 任务 × 3 档（17／17／16 局）+ 2 任务 × 5 档 × 10 局 + 2 任务 × 1 档 × 50 局 = 800 局；五模型共 4000 次轨迹首试。冒烟为 5 模型 × 1 任务 VideoUnmask × 1 档 xhard1 × 1 局，包含在正式分母内。reset 计量最多 8000，基础设施重试、递补和客户端重启均为零；正常任务失败保留独立成功字段。

## 版本与准备状态

实施起点 `2d2cc25e195294ebd9f980f1bbc864e894fb7dfe`，`dev` 工作区干净；此提交仅为规划锚点，尚非正式执行版本。正式运行前补记通过验证的冻结提交、配置和解释器指向，未完成冻结前不发布 `launch.ready`。

环境核实：`sled-vail`，两张 RTX 6000 Ada，NFS、`/data/hongzefu` 与 SSH 配置存在；Great Lakes 主连接可用。当前五个 gitlink 与计划一致：benchmark `adf19363d4347af77f3fb6746c3a6b685991aba0`；mme-vla `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`；SimpleMemVLA `c564c17d276d7294200122b286c21901a3bfb99f`；PonderPounce `723df35762bb641e1d520e4fa9359b98644adc21`；Astra `4c3fd6a8667e7a219e547fe1a533a4b9726fc6db`（不调用）。

## 身份与预算

准备命令（仅导出包内身份，不创建环境或轨迹）：

```bash
UV_CACHE_DIR=/home/hongzefu/.cache/uv uv run --no-sync python dev-scripts/gl/export_eval_identities.py --dataset ood --episodes 0:50 --out artifacts/ood-five-raw-seed7-20261010-01/control/identities.jsonl
```

退出码 0，原始判定：`EVAL_IDENTITY_EXPORT=PASS datasets=ood episodes=800 hard_verify=0 ood=800 xhard0=0 tasks=16 cell_mismatch=0 bad=0 dup_keys=0 count_mismatch=0`。各档既有 `hard_fingerprint` 警告如实保留；未修改规格或受保护环境源码。

## 资源、通知与运行位置

本轮一次申请四个 GPU 作业，各 `chaijy2/spgpu`、1 GPU／1 CPU／24G／120 小时；一个 `chaijy2/standard` 纯 CPU 监督作业，1 CPU／4G／120 小时。CPU 收件人为 `hongzefu@umich.edu`，通知类型 `BEGIN,FAIL,END,TIME_LIMIT`。CPU 为 `63660978`，四 GPU 为 `63661000`、`63661005`、`63661007`、`63661014`；五次提交均退出 0，无重提。`scontrol show job` 已核时限 `5-00:00:00`、`Requeue=0` 与资源规格，原始输出留在 `artifacts/ood-five-raw-seed7-20261010-01/control/allocation.txt`。准备截止为 epoch `1791750249`，等待就绪期间不执行模型。

启动快照由独立控制实现工作树制作，经 `bash -n` 与 `git diff --check` 检查后以 rsync 同步；CPU 脚本 SHA256 `cbbae1d037e7abc6da693ee673d241481291797d1ea834fc95fcc9970fcfb1db`，GPU 守卫 SHA256 `81d1f448d260a170f2d5c67e44a59e1f011b08a7aca1f069c64765321ef3a10a`。提交后快照字节冻结；正式代码以最终 `launch.ready` 指定的验证提交与配置哈希为准。外部邮件是否实际投递尚未核实。

本机根：`/data/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-01`；NFS 根：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/ood-five-raw-seed7-20261010-01`。本轮尚未启动任何 tmux 会话。

## 实施中补强与环境核对

用户就守卫查询可能挂起的故障窗口明确答复：「采用，同一批作业内追加独立守卫」。保留已提交启动快照，正式 CPU 监督器在同四个 GPU 占位作业内追加纯 CPU 生命周期守卫，对 Slurm 查询和取消加超时；不申请新作业，不新增任何轨迹或 reset 尝试。工作负载步号按该席专属进度中的真实 Slurm 作业／步号确认，不能因同作业内存在守卫步而误判。

三个先启动的 GPU 席位在同节点 `gl1505`，本轮显式传入空的 `--gpus` 参数，让服务端保留 `srun` 分配的 `CUDA_VISIBLE_DEVICES`，不把可见卡强制改成物理卡 0。端口按四席分别为 22400／22500／22600／22700，工作目录和进度路径按片独占。

QwenVL 的官方源码固定使用 `flash_attention_2`；本仓 NFS 客户端环境缺少编译模块，既有实跑载体使用 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sg-eval/venvs/client-env/bin/python`。本轮沿用这一现成解释器：目录核对为 `flash_attn=2.8.3`、`torch=2.9.1`、`ms-swift=3.11.1`、`transformers=4.57.3`、`peft=0.18.1`，与客户端项目关键依赖版本一致；真实导入 `flash_attn` 与 `torch` 已退出 0，`gpu_initialized=False`。未安装依赖、未切换注意力实现；冻结源码通过显式 `PYTHONPATH` 和实际导入路径断言绑定。

## 交接验收

工程验证、冻结版本、五正式首局冒烟通过后，继续观察 CPU 首次分配正式全量分片；必须同时核验领取、合法动作实际执行一步、专属进度、CPU 与搬运心跳及错误零。仅在 `FIRST_DISPATCH=PASS` 与 `AGENT_HANDOFF=PASS` 有真实证据后结束会话。后台作业存活不能代替这些证据。
