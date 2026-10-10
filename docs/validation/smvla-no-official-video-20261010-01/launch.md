# SimpleMemVLA 跳过官方视频的两局实测：启动记录

## 目的与授权

用户要求：「做一个最简单的 Greatx的实测，就是你试一下不生成官方视频的制作，然后大概需要多久。就实测一下这个时间，然后给我一个重新评估。」完整最小方案呈现后用户：「同意开始。」用户另问原始帧是否有损，已明确当前两路原始画面为 AV1 4:4:4、CRF 24，不保证原始像素逐字节一致；动作与轨迹数值不经视频压缩。

本轮仅 1 任务 VideoRepick × 1 档 xhard1 × 2 局（builder episode 1、2），policy_seed=7、1800 步 strict。沿用此前常驻诊断的两个身份，最多 2 次轨迹、4 次 reset，基础设施／到期／客户端重试均为 0；首次单局正常结束后才顺序进入第二局。正常任务失败保留 0，不挑成功重跑。环境创建失败立即停止受影响任务、留证定位并问用户，禁止逐局重启绕过。

## 版本与环境

本机 sled-vail，执行 Great Lakes 既有占位 job 63431432，gl1512，4 CPU／1 GPU；四个占位作业均保留，不新提交。源码使用此前两组诊断的独立 clean 冻结副本 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/full-eval-20261010/runtime-code-r3`，HEAD `0c6f8a3a552876020611d4d654a41adb3accba65`。本机对应冻结副本核实 porcelain 为空；GL 运行前由导入指向守卫核验源路径。模型、环境、权重、种子与录像参数不改，只增加已有 `--no-render`。

权重 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/checkpoints/simplememvla_robomme`。客户端为 GL 评估仓 `.venv/bin/python`，服务端为共享盘 `robomme_benchmark-sgeval2-b34/artifacts/v8-two/venvs/smvla-env/bin/python`；沿用原 tokenizer SHA256、openpi-data、jax 编译缓存与离线 HF 缓存。模型加载一次，客户端和服务端跨局常驻。

## 启动与配置还原

本机产物 `artifacts/smvla-no-official-video-20261010-01`，GL 产物 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/RoboMME-benchmark-OOD-eval/artifacts/smvla-no-official-video-20261010-01`。输出 `output/`、预算 `budget.jsonl`、日志 `run.log`、端口 20820、席位 `no-official-two`。唯一 tmux 会话清单：`ev-smvla-no-official-20261010-01`。启动载体只在 artifacts，不归档 bash/yaml 到文档。

完整工作负载参数：

```bash
srun --jobid=63431432 --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared \
 bash dev-scripts/gl/run_eval_gl.sh --policies smvla --policy-seed 7 \
 --identities "$diag_root/identities.jsonl" --budget-ledger "$diag_root/budget.jsonl" \
 --run smvla-no-official-video-20261010-01 --out "$diag_root/output" --seat no-official-two --gpus 0 \
 --trajectory-cap 2 --reset-cap 4 --shared-infra-cap 0 --expired-cap 0 --planned-first-tries 2 \
 --infra-retries 0 --client-restarts 0 --noprog-s 2700 --stop-on-env-build-error --no-render \
 --ckpt "$diag_nfs/SimpleMemVLA/checkpoints/simplememvla_robomme" --port-base 20820 \
 --work-dir "$diag_root/work/no-official-two" \
 --cfg openpi_data_home=$diag_nfs/sg-eval/openpi-data \
 --cfg tokenizer_sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 \
 --cfg jax_cache_root=$diag_nfs/artifacts/full-eval-20261010/jax-cache \
 --cfg server_dir="$diag_root/servers/no-official-two" \
 --cfg smvla_py=$diag_nfs/robomme_benchmark-sgeval2-b34/artifacts/v8-two/venvs/smvla-env/bin/python
```

其中 `diag_nfs=/nfs/turbo/coe-chaijy-unreplicated/hongzefu`，`diag_root=$diag_nfs/RoboMME-benchmark-OOD-eval/artifacts/smvla-no-official-video-20261010-01`。`BENCH_PY` 指主 venv，`ROBOMME_EVAL_ROOT` 与 `PYTHONPATH` 指冻结副本，OMP／MKL=1、离线模型缓存沿用旧诊断。完整载体路径在产物根 `run.sh`，外壳 `monitor-run.sh` 使用 pipefail、PYTHONUNBUFFERED、tee 和 EXIT_CODE。预计约 10 分钟，最大等待 30 分钟；失败不重试。

## 验收与计时

读取实际两局 `result.json`，报告任务成功、步数、块数、稳态客户端决策耗时、`t_end-t_start`、`recorder.finalize_s` 与加载时间。验原始双视图录像、动作数组和轨迹存在、recorder PASS、官方视频不存在、同一客户端／服务端与一次 load、2 次轨迹及4次 reset且零重试。

对比 `docs/validation/smvla-restart-ab-20261010-01/records/persistent-ep{1,2}.json`。只以相同身份的历史常驻局作时间参考；两局不代表完整 OOD，若步数或成绩不同则明确不是严格相同轨迹对照。四卡全量估算保留其它五模型的历史样本，只用本轮检验跳过官方视频后的墙钟口径，不把两个身份外推成各任务新实测。
