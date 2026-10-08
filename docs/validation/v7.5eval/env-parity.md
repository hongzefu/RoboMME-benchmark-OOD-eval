# 2.1 环境本身稳不稳定 · 第 6 步旧环境栈 vs 新环境栈 · 环境测速

> 对应方案 §3.2「2.1」「第 6 步 旧环境栈 vs 新环境栈」、§4 测速。启动提交：12.267 `2176a0e3`（本机 worktree `artifacts/v7.5eval/wt/2176a0e3`，GL worktree `v75eval/wt/2176a0e3`）；旧／新环境栈比较由 12.275 `198965ae` 的 `step6_summary.py` 离线计算，0 额外局。用户原话：本轮原话 1、2、8；方案原话 2、5、7、9。

## 1. 结论

- **同 GPU 型号内环境逐字节可复现**：本机卡 0 首遍对常驻倒序（C1:C2）、对本机卡 1（C1:C3），A40-乙首遍对同卡复跑（C5:C6）、对 A40-丁（C5:C7），九层中身份、演示前状态、演示帧、reset 观测、固定动作 30 步的帧与观测全部 48/48 相等；状态层的差只来自 actor 名字与键集合（`name_only=21 key_set_diff=12`，扣除后 `identities_with_diff=0`、`first_diff=-`）。进程常驻＋倒序（C2）不改变任何一层——但这是「常驻＋倒序」的组合，不能单独归因（方案 §3.1 归因规则）。
- **换 GPU 型号（RTX 6000 Ada ↔ A40）环境即不再逐位一致**：C1:C5 首差在演示前场景状态，44/48 个身份有差，演示帧 23/48 相等，逐帧图像平均绝对差最大 25.84/255（0.101）；固定动作 30 步后终态 30/48 相同。
- **旧环境栈与新环境栈在同一张卡上 reset 逐字节相同**：官方重跑一（A40，sapien 3.0.3＋官方 `robomme`）录下的 reset 帧与 C5、C7（A40，sapien 3.0.2＋`robomme_hard` xhard0）逐身份比，17646/17646 帧 sha256 相同、状态最大绝对差 0；本机旧官方小样本 R1（RTX）对 C1 同样 17572/17572 相同。因此「正式跑法 vs 官方」的差距中，环境栈（环境代码、sapien 版本、环境侧 torch 版本）在 reset 层贡献为 0；闭环层仍无法把它与接口差拆开（方案盲区第 2 条）。
- **随机流**：`gym.make` 消耗 numpy 全局随机数，`reset` 不消耗；torch 全局随机数两者都不消耗（12.267 冒烟实测）。SimpleMemVLA 推理唯一随机源是 `dit_action_head` 的 `torch.randn`，所以新接口「每局 reset 时 `torch.manual_seed(0); np.random.seed(0)`」与旧官方「建环境前重设种子」等价（12.268 核实，方案已定口径 5）。

## 2. 判定行原文

```text
ENV_DIGEST_PARITY=INFO pair=C1:C2 compared=48 layer_equal=demo_frames:48,identity:48,post_demo_state:39,pre_demo_state:48,reset_obs:48,step_frames:48,step_obs:48,step_state:36,step_status:48 first_diff=- state_max_abs=n/a image_mad=n/a image_mad_unit=n/a obs_max_abs=n/a identities_with_diff=0 name_only=21 key_set_diff=12 missing_in_a=0 missing_in_b=0 unmeasured=0 source=reused partial=no
ENV_DIGEST_PARITY=INFO pair=C1:C3 compared=48 layer_equal=demo_frames:48,identity:48,post_demo_state:39,pre_demo_state:48,reset_obs:48,step_frames:48,step_obs:48,step_state:36,step_status:48 first_diff=- state_max_abs=n/a image_mad=n/a image_mad_unit=n/a obs_max_abs=n/a identities_with_diff=0 name_only=21 key_set_diff=12 missing_in_a=0 missing_in_b=0 unmeasured=0 source=reused partial=no
ENV_DIGEST_PARITY=INFO pair=C1:C5 compared=48 layer_equal=demo_frames:23,identity:48,post_demo_state:4,pre_demo_state:19,reset_obs:25,step_frames:32,step_obs:30,step_state:4,step_status:30 first_diff=pre_demo_state state_max_abs=209.9051 image_mad=25.8414 image_mad_unit=0.1013 obs_max_abs=0.0022 identities_with_diff=44 name_only=0 key_set_diff=12 missing_in_a=0 missing_in_b=0 unmeasured=6 source=reused partial=no
ENV_DIGEST_PARITY=INFO pair=C5:C6 compared=48 layer_equal=demo_frames:48,identity:48,post_demo_state:39,pre_demo_state:48,reset_obs:48,step_frames:48,step_obs:48,step_state:36,step_status:48 first_diff=- state_max_abs=n/a image_mad=n/a image_mad_unit=n/a obs_max_abs=n/a identities_with_diff=0 name_only=21 key_set_diff=12 missing_in_a=0 missing_in_b=0 unmeasured=0 source=reused partial=no
ENV_DIGEST_PARITY=INFO pair=C5:C7 compared=48 layer_equal=demo_frames:48,identity:48,post_demo_state:39,pre_demo_state:48,reset_obs:48,step_frames:48,step_obs:48,step_state:36,step_status:48 first_diff=- state_max_abs=n/a image_mad=n/a image_mad_unit=n/a obs_max_abs=n/a identities_with_diff=0 name_only=21 key_set_diff=12 missing_in_a=0 missing_in_b=0 unmeasured=0 source=reused partial=no
ENV_STACK=INFO src=O1 policy=smvla cell=C5 compared=48 errors=0 demo_count_equal=48/48 frames_sha_equal=17646/17646 image_mad_max=0 image_mad_mean=0 init_mad_max=0 state_max_abs=0 unmeasured=0 partial=no note=官方reset帧=演示帧+初始帧;MAD为0-255刻度
ENV_STACK=INFO src=O1 policy=mme cell=C5 compared=48 errors=0 demo_count_equal=48/48 frames_sha_equal=17646/17646 image_mad_max=0 image_mad_mean=0 init_mad_max=0 state_max_abs=0 unmeasured=0 partial=no note=官方reset帧=演示帧+初始帧;MAD为0-255刻度
ENV_STACK=INFO src=O1 policy=smvla cell=C7 compared=48 errors=0 demo_count_equal=48/48 frames_sha_equal=17646/17646 image_mad_max=0 image_mad_mean=0 init_mad_max=0 state_max_abs=0 unmeasured=0 partial=no note=官方reset帧=演示帧+初始帧;MAD为0-255刻度
ENV_STACK=INFO src=O1 policy=mme cell=C7 compared=48 errors=0 demo_count_equal=48/48 frames_sha_equal=17646/17646 image_mad_max=0 image_mad_mean=0 init_mad_max=0 state_max_abs=0 unmeasured=0 partial=no note=官方reset帧=演示帧+初始帧;MAD为0-255刻度
ENV_STACK=INFO src=R1 policy=smvla cell=C1 compared=48 errors=0 demo_count_equal=48/48 frames_sha_equal=17572/17572 image_mad_max=0 image_mad_mean=0 init_mad_max=0 state_max_abs=0 unmeasured=0 partial=no note=官方reset帧=演示帧+初始帧;MAD为0-255刻度
ENV_STACK=INFO src=R1 policy=mme cell=C1 compared=48 errors=0 demo_count_equal=48/48 frames_sha_equal=17572/17572 image_mad_max=0 image_mad_mean=0 init_mad_max=0 state_max_abs=0 unmeasured=0 partial=no note=官方reset帧=演示帧+初始帧;MAD为0-255刻度
ENV_SPEED=INFO cell=C1 rows=48 wall_s_p50=4.7075 wall_s_p95=15.5403 make_env_s_p50=1.2384 make_env_s_p95=1.8082 gym_make_s_p50=1.2296 gym_make_s_p95=1.799 eval_reset_s_p50=0.8839 eval_reset_s_p95=4.9311 inner_reset_s_p50=0.0052 inner_reset_s_p95=0.0097 demo_s_p50=0.8796 demo_s_p95=4.9259 demo_s_per_frame_p50=0.0077 demo_s_per_frame_p95=0.0215 close_s_p50=0.1202 close_s_p95=0.1337
ENV_SPEED=INFO cell=C2 rows=48 wall_s_p50=4.2605 wall_s_p95=14.8611 make_env_s_p50=1.1983 make_env_s_p95=1.3341 gym_make_s_p50=1.189 gym_make_s_p95=1.3247 eval_reset_s_p50=0.8807 eval_reset_s_p95=4.794 inner_reset_s_p50=0.0052 inner_reset_s_p95=0.0087 demo_s_p50=0.8764 demo_s_p95=4.7893 demo_s_per_frame_p50=0.0076 demo_s_per_frame_p95=0.0212 close_s_p50=0.1227 close_s_p95=0.127
ENV_SPEED=INFO cell=C3 rows=48 wall_s_p50=4.7445 wall_s_p95=15.5453 make_env_s_p50=1.2673 make_env_s_p95=1.8235 gym_make_s_p50=1.2583 gym_make_s_p95=1.8143 eval_reset_s_p50=0.8857 eval_reset_s_p95=4.8303 inner_reset_s_p50=0.0052 inner_reset_s_p95=0.0086 demo_s_p50=0.8814 demo_s_p95=4.8257 demo_s_per_frame_p50=0.0076 demo_s_per_frame_p95=0.0215 close_s_p50=0.1203 close_s_p95=0.1269
ENV_SPEED=INFO cell=C5 rows=48 wall_s_p50=6.3445 wall_s_p95=21.2967 make_env_s_p50=1.2331 make_env_s_p95=2.3859 gym_make_s_p50=1.2211 gym_make_s_p95=2.3743 eval_reset_s_p50=1.2423 eval_reset_s_p95=6.3343 inner_reset_s_p50=0.0065 inner_reset_s_p95=0.0127 demo_s_p50=1.2366 demo_s_p95=6.3286 demo_s_per_frame_p50=0.0106 demo_s_per_frame_p95=0.03 close_s_p50=0.1129 close_s_p95=0.1519
ENV_SPEED=INFO cell=C6 rows=48 wall_s_p50=6.107 wall_s_p95=20.3798 make_env_s_p50=1.2312 make_env_s_p95=2.0736 gym_make_s_p50=1.2202 gym_make_s_p95=2.0621 eval_reset_s_p50=1.2389 eval_reset_s_p95=6.3837 inner_reset_s_p50=0.0065 inner_reset_s_p95=0.0128 demo_s_p50=1.2336 demo_s_p95=6.3779 demo_s_per_frame_p50=0.0106 demo_s_per_frame_p95=0.0297 close_s_p50=0.1152 close_s_p95=0.1496
ENV_SPEED=INFO cell=C7 rows=48 wall_s_p50=6.287 wall_s_p95=20.8757 make_env_s_p50=1.2677 make_env_s_p95=2.0641 gym_make_s_p50=1.2542 gym_make_s_p95=2.0525 eval_reset_s_p50=1.2428 eval_reset_s_p95=6.5906 inner_reset_s_p50=0.008 inner_reset_s_p95=0.0209 demo_s_p50=1.2373 demo_s_p95=6.5839 demo_s_per_frame_p50=0.011 demo_s_per_frame_p95=0.0301 close_s_p50=0.1179 close_s_p95=0.1539
```

明细：`artifacts/v7.5eval/env/parity-C1-C2.json`、`parity-C1-C3.json`、`parity-C1-C5.json`、`parity-C5-C6.json`、`parity-C5-C7.json`；旧／新环境栈 `artifacts/v7.5eval/summary/detail/env-stack-{O1-C5,O1-C7,R1-C1}-{mme,smvla}.json`。`source=reused` 表示第 6 步汇总直接读 2.1 当时产出的对比 JSON、未重算。

## 3. 测速读法

- 环境生成（`make_env`，其中 `gym_make` 几乎占全部）本机约 1.2 s、A40 约 1.23～1.27 s，两型号相近；内层 `BaseEnv.reset` 只有 5～8 ms。
- 评估 reset 的耗时几乎全是演示生成（`demo_s ≈ eval_reset_s`）：本机中位 0.88 s、A40 中位 1.24 s，P95 本机约 4.8～4.9 s、A40 约 6.3～6.6 s；折合每帧本机 7.6 ms、A40 10.6～11.0 ms（A40 慢约 40%）。
- 同卡复跑（C5→C6）与换机器（C5→C7）的测速差在 5% 以内；gl1525（乙）节点上只有本任务一个席位（方案 §1.3 选址理由）。

## 4. 预算与重试（P5 乘式）

计划 6 条件 × 16 任务 × 1 档 × 3 局 = 288 次身份尝试（每次 2 次 reset：评估 reset ＋ 演示前另建环境的底层 reset，288 × 2 = 576 次 reset）。实际 289：C1 首次起跑被主会话事故打断（[incidents.md](incidents.md) 事故 1），已完成的 6 个身份（12 次 reset）作废重跑，外加卡住的 1 个身份，2.1 基础设施重试比子额度 6 多 1 次身份尝试，仍在总上限内。

```text
BUDGET=INFO item=2.1_环境检测 attempts=289 unique=288 incident=0 retries=1 cap=288 retry_cap=6 over=no retry_over=no est=含开发冒烟1行(估)
```

## 5. 命令原文与会话

```bash
# 本机卡 0：C1 首遍 → C2 常驻倒序（artifacts/v7.5eval/lanes/env-card0.sh，tmux v75-env-card0）
taskset -c 0-3 $PY scripts/parity/hard_regression.py env-digest --cell C1 --identities $ID --out $OUT --gpu 0
taskset -c 0-3 $PY scripts/parity/hard_regression.py env-digest --cell C2 --identities $ID --out $OUT --gpu 0 --resident --reverse
$PY scripts/parity/hard_regression.py env-digest-compare --a $OUT/C1 --b $OUT/C2 --out $OUT/parity-C1-C2.json
# 本机卡 1：C3（artifacts/v7.5eval/lanes/env-card1.sh，与卡 1 其他 GPU 任务经 flock 串行）
flock …/gpu1.lock taskset -c 4-7 $PY scripts/parity/hard_regression.py env-digest --cell C3 --identities …/identities-small48.json --out $OUT --gpu 1
# GL 登录节点 gl-login3 tmux v75-env-yi（乙 C5→C6 串行）、v75-env-ding（丁 C7）：v75eval/lanes/env-gl-login.sh <yi|ding>
srun --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared --jobid=62608440 bash $N/lanes/env-gl.sh C5   # 随后 C6
srun --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared --jobid=62608595 bash $N/lanes/env-gl.sh C7
# env-gl.sh 内：$WT/.venv/bin/python scripts/parity/hard_regression.py env-digest --cell <C5|C6|C7> --identities $N/identities-small48.json --out $N/env
```

其中 `$PY`＝主仓库 `.venv/bin/python`，`$ID`＝`artifacts/v7.5eval/identities-small48.json`，`$OUT`＝`artifacts/v7.5eval/env`，`$N`＝`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v75eval`。日志：`artifacts/v7.5eval/logs/env-card0.log`（事故前半格另存 `env-card0.aborted-venv-rebuild.log`）、`env-card1.log`，GL `artifacts/v7.5eval/nfs-archive/logs/env-yi.log`、`env-ding.log`。

## 6. 边界

- 「演示前场景状态」取自另建的一个底层环境 reset 后的状态（`hard_regression.py::_PROBE` 做法），不是评估实例本身的瞬间状态（方案盲区）。
- C1:C5 的 `unmeasured=6`：6 处无法逐帧计算图像差（两边帧数不等时逐帧 MAD 记 `None`、不填 0，12.275 口径），`image_mad` 只统计可测部分。
- 方案风险 K8 的 VideoPlaceOrder 610701、611101（已知官方生成失败 seed）在同型号四对中同样逐层相等，未单独出现差异。
