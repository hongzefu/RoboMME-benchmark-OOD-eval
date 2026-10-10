# SimpleMemVLA 常驻与逐局启动的最小诊断

用户要求在 GreatLakes 实测是否需要每局重启，选择「常驻与逐局重启做诊断对照」，随后要求「不要交叉复核了」「尽快收尾，用一个最简单的测试方法来测试」，并明确「同意开始」。本轮只运行以下已确认两组，不恢复全量。用户同时要求：出现创建失败不得用逐局重启持续绕过，必须定位后问用户；本轮逐局启动仅为已批准的两局诊断对照。

## 两组与预算

1 任务（VideoRepick）× 1 档（xhard1）× 2 局 × 2 种运行方式 = 最多 4 次轨迹尝试、8 次 reset。两身份固定为 `VideoRepick_xhard1_16900101`（episode 1）、`VideoRepick_xhard1_16900200`（episode 2），模型种子 7、1800 步上限。常驻组一次加载，同一客户端和服务端连续两局；逐局组两次独立客户端和服务端，依次执行相同两局。

| 组别 | 原占位 JobID／节点 | 客户端 seat | 端口 | 组内硬上限 |
|---|---|---|---|---|
| 常驻 | 63431432／gl1512 | persistent-two | 20800 | 轨迹 2、reset 4、重试 0 |
| 逐局启动 | 63431433／gl1513 | restart-one、restart-two | 20900 | 共享轨迹 2、reset 4、重试 0 |

两组分开输出与预算；两组上限合计固定为 4／8，不重建或扩额。逐局组两个客户端共享同一组预算和输出队列，两局身份不同；常驻与逐局组的账本分开以免相同输入 token 发生跨组碰撞。自动客户端重启和基础设施重试均关闭；创建失败立即退出，主会话停止剩余测试、保留证据并定位，不自动切换运行方式、补跑或交叉复核。正常任务失败与正常步数截断不重跑。

## 版本、位置与判据

所有模型／benchmark 源码来自已核 clean 的冻结副本 `runtime-code-r3 @ 0c6f8a3a552876020611d4d654a41adb3accba65`。SimpleMemVLA gitlink `c564c17d276d7294200122b286c21901a3bfb99f`、benchmark gitlink `a5efb992e769b7011281a1d31739b76d8868602a`，权重、解释器及渲染设置沿原已验证配置，零模型／受保护源码修改。四个旧占位 job 保留，本轮只复用其中两个，各 1 GPU／4 CPU／64G，不提交新作业。

本机产物 `artifacts/smvla-restart-ab-20261010-01`，GL 产物 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/smvla-restart-ab-20261010-01`，均已确认新建实体目录。各组输出为其下 `output/<组别>`、预算 `budget-<组别>.jsonl`。唯一 tmux 会话分别为 `ev-smvla-ab-persistent-20261010-01`、`ev-smvla-ab-restart-20261010-01`，日志 `persistent.log`、`restart.log`。

验收读取真实 PID／server metadata、加载次数、两个身份的环境创建结果、`infra` 与各预算，任务成功单独报告。常驻两局都完成且同 PID、一次 load：`SMVLA_PERSISTENT=PASS episodes=2 loads=1`；逐局组两局都完成且不同 PID、两次 load：`SMVLA_RESTART=PASS episodes=2 loads=2`。任一创建失败记录 FAIL 并定位。两组不同节点、且只各两局，结果为最小诊断，不足以证明长期必要性或单独归因于重启方式。

## 启动与还原

运行载体在 artifacts 中，不归档独立 bash／yaml。环境显式绑定冻结 src、benchmark、openpi-client 的 PYTHONPATH，既有主 uv 解释器与 SMVLA 子解释器；清除旧 SLURM／CUDA／MESA／SAPIEN 覆盖，保持 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、HF 离线、原缓存和 ffprobe。核心命令如下；`diag_arm` 取 persistent／restart，`diag_job` 取上述作业，`diag_identity` 常驻为 identities.jsonl，逐局两次分别 identity-1.jsonl／identity-2.jsonl。

```bash
srun --jobid="$diag_job" --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared \
 bash dev-scripts/gl/run_eval_gl.sh --policies smvla --policy-seed 7 \
 --identities "$diag_root/$diag_identity" --budget-ledger "$diag_root/budget-$diag_arm.jsonl" \
 --run "smvla-restart-ab-20261010-01-$diag_arm" --out "$diag_root/output/$diag_arm" --seat "$diag_seat" --gpus 0 \
 --trajectory-cap 2 --reset-cap 4 --shared-infra-cap 0 --expired-cap 0 --planned-first-tries 2 \
 --infra-retries 0 --client-restarts 0 --noprog-s 2700 --stop-on-env-build-error \
 --ckpt "$diag_nfs/SimpleMemVLA/checkpoints/simplememvla_robomme" --port-base "$diag_port" \
 --work-dir "$diag_root/work/$diag_seat" --cfg openpi_data_home=$diag_nfs/sg-eval/openpi-data \
 --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 \
 --cfg jax_cache_root=$diag_nfs/artifacts/full-eval-20261010/jax-cache --cfg server_dir="$diag_root/servers/$diag_seat" \
 --cfg smvla_py=$diag_nfs/robomme_benchmark-sgeval2-b34/artifacts/v8-two/venvs/smvla-env/bin/python
```

两组 detached tmux、PYTHONUNBUFFERED／pipefail／tee／EXIT_CODE；主会话负责监听与精确清理自己的进程，不取消占位 job。预计约 15～25 分钟；没有额外探针或大规模生成。原 `smvla-persistent-20261010-1311` 仅为未起跑草稿，不属于本轮执行资源。
