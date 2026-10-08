# 第 3 步新接口跑通 · 5.1 新接口换条件还一致吗

> 对应方案 §3.2「第 3 步」「5.1」。启动提交：第 3 步为 12.270 `2abf227d`（本机 worktree `artifacts/v7.5eval/wt/2abf227d`，`scripts/eval-official/run_seat.sh`）；5.1 六格为 12.272 `a81c4f6b`（本机 `artifacts/v7.5eval/wt/a81c4f6b`，GL `v75eval/wt/a81c4f6b`，新增 `--client-per-task`）。确定性标志两策略均为关（[policy-replay.md](policy-replay.md)）。用户原话：本轮原话 2、7、8；方案原话 6、17、22。

## 1. 结论

- **第 3 步跑通**：本机卡 1 真席位脚本每策略 1 局，SimpleMemVLA 成功 818 步、106 条消息 sha 零错漏；MME 失败 775 步（同身份 MME 历次 892 成功／688 失败／768 成功／775 失败，同卡重跑本就会翻，只记录不调参）。
- **SimpleMemVLA 在同 GPU 型号内闭环逐局可复现，新旧接口也逐局相同**：本机卡 0 首遍对常驻倒序（E1:E2）、对本机卡 1（E1:E3）、A40-乙首遍对同卡复跑（E5:E6）、对 A40-丁（E5:E7）全部 0 翻转、步数全部相等；**同卡新旧接口闭环 R1:E1 0 翻转、步数 46/46 相等**；A40 上官方重跑一对新接口 O1:E5 0 翻转、45/45 相等；E5、E6、E7 对官方历史成绩 0 翻转。跨 GPU 型号（E1:E5）则 2 局成功→失败、1 局失败→成功，与环境层 C1:C5 的差一致。
- **MME 在任何两格之间都翻 2～3 局／方向**，包括同卡复跑（E5:E6 3／3）、常驻倒序（E1:E2 2／0）、同卡新旧接口（R1:E1 2／3），与第 4 步「跨 server 启动动作即有小差」一致；48 局粒度 1 局 ≈ 2.08 个百分点，成功率差在 −6.25～＋2.08 个百分点之间，McNemar 均不显著（p ≥ 0.25）。
- E6 SimpleMemVLA 首次尝试受 GPU 计算模式事故影响（7 局真实 ＋ 69 次 0 步 Vulkan 错误，目录 `artifacts/v7.5eval/official-rec/E6/smvla/s-yi-vulkanfail-0544` 留证、不进比较，见 [incidents.md](incidents.md) 事故 6），07:00～07:55 在乙上整格重跑 48/48、0 错误、0 基础设施失败：A40 同卡复跑 E5:E6 0 翻转、步数 45/45 相等，E6 对官方历史成绩 0 翻转。E5 MME、E7 MME 同样整格重跑 48/48（E7 失败尝试留证为 `s-ding-vulkanfail-0540`）。

## 2. 判定行原文

第 3 步：

```text
STEP3=INFO policy=smvla identity=PickXtimes/510300 status=success steps=818 error=none recorder_verify=PASS messages=106 frames_sent=819 sha_mismatch=0 episode_wall_s=66.2986 gpu=NVIDIA_RTX_6000_Ada_Generation git=2abf227d
STEP3=INFO policy=mme identity=PickXtimes/510300 status=fail steps=775 error=none recorder_verify=PASS messages=n/a frames_sent=n/a sha_mismatch=n/a episode_wall_s=78.4188 gpu=NVIDIA_RTX_6000_Ada_Generation git=2abf227d
TRANSPORT=PASS frames=993 mismatch=0 messages=227 exec_actions=892
```

`TRANSPORT` 行来源冲突，保留并注明：12.268 commit body 把它记在开发期 MME 冒烟（`EPISODE_DONE status=success steps=892`，`exec_actions=892` 与之吻合）；主会话事实清单把它列在「第 3 步」下。第 3 步真席位那局 MME 的 `STEP3` 行 `messages=n/a sha_mismatch=n/a`，即该局本身没有单独的传输核对行。

5.1（`artifacts/v7.5eval/summary/verdicts.txt`）：

```text
EVAL_PARITY=INFO pair=E1:E2 policy=smvla compared=48 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 lost_err=0 new_timeout=0 lost_timeout=0 sr_a=75 sr_b=75 sr_diff_pp=0 ci=[0,0] mcnemar_p=1 steps_equal=46/46 steps_diff_p50=0 steps_diff_p95=0 first_div=diverged:0/48,min:n/a,p50:n/a,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>36s0f0t0e;f>0s10f0t0e;t>0s0f2t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E1:E3 policy=smvla compared=48 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 lost_err=0 new_timeout=0 lost_timeout=0 sr_a=75 sr_b=75 sr_diff_pp=0 ci=[0,0] mcnemar_p=1 steps_equal=46/46 steps_diff_p50=0 steps_diff_p95=0 first_div=diverged:0/48,min:n/a,p50:n/a,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>36s0f0t0e;f>0s10f0t0e;t>0s0f2t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E1:E5 policy=smvla compared=48 missing_a=0 missing_b=0 s2f=2 f2s=1 new_err=0 lost_err=0 new_timeout=2 lost_timeout=1 sr_a=75 sr_b=72.9167 sr_diff_pp=-2.0833 ci=[-10.417,4.167] mcnemar_p=1 steps_equal=22/44 steps_diff_p50=0 steps_diff_p95=26.7 first_div=diverged:48/48,min:0,p50:0,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>34s2f0t0e;f>0s8f2t0e;t>1s0f1t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E5:E6 policy=smvla compared=48 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 lost_err=0 new_timeout=0 lost_timeout=0 sr_a=72.9167 sr_b=72.9167 sr_diff_pp=0 ci=[0,0] mcnemar_p=1 steps_equal=45/45 steps_diff_p50=0 steps_diff_p95=0 first_div=diverged:0/48,min:n/a,p50:n/a,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>35s0f0t0e;f>0s10f0t0e;t>0s0f3t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E5:E7 policy=smvla compared=48 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 lost_err=0 new_timeout=0 lost_timeout=0 sr_a=72.9167 sr_b=72.9167 sr_diff_pp=0 ci=[0,0] mcnemar_p=1 steps_equal=45/45 steps_diff_p50=0 steps_diff_p95=0 first_div=diverged:0/48,min:n/a,p50:n/a,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>35s0f0t0e;f>0s10f0t0e;t>0s0f3t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=R1:E1 policy=smvla compared=48 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 lost_err=0 new_timeout=0 lost_timeout=0 sr_a=75 sr_b=75 sr_diff_pp=0 ci=[0,0] mcnemar_p=1 steps_equal=46/46 steps_diff_p50=0 steps_diff_p95=0 first_div=n/a pp_per_ep=2.0833 matrix=s>36s0f0t0e;f>0s10f0t0e;t>0s0f2t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=O1:E5 policy=smvla compared=48 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 lost_err=0 new_timeout=0 lost_timeout=0 sr_a=72.9167 sr_b=72.9167 sr_diff_pp=0 ci=[0,0] mcnemar_p=1 steps_equal=45/45 steps_diff_p50=0 steps_diff_p95=0 first_div=n/a pp_per_ep=2.0833 matrix=s>35s0f0t0e;f>0s10f0t0e;t>0s0f3t0e;e>0s0f0t0e
EVAL_VS_E0=INFO cond=E1 policy=smvla compared=48 missing_b=0 s2f=1 f2s=2 new_err=0 new_timeout=1 success=35>36 steps_equal=22/44 matrix=s>34s0f1t0e;f>2s8f0t0e;t>0s2f1t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E2 policy=smvla compared=48 missing_b=0 s2f=1 f2s=2 new_err=0 new_timeout=1 success=35>36 steps_equal=22/44 matrix=s>34s0f1t0e;f>2s8f0t0e;t>0s2f1t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E3 policy=smvla compared=48 missing_b=0 s2f=1 f2s=2 new_err=0 new_timeout=1 success=35>36 steps_equal=22/44 matrix=s>34s0f1t0e;f>2s8f0t0e;t>0s2f1t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E5 policy=smvla compared=48 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 success=35>35 steps_equal=45/45 matrix=s>35s0f0t0e;f>0s10f0t0e;t>0s0f3t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E6 policy=smvla compared=48 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 success=35>35 steps_equal=45/45 matrix=s>35s0f0t0e;f>0s10f0t0e;t>0s0f3t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E7 policy=smvla compared=48 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 success=35>35 steps_equal=45/45 matrix=s>35s0f0t0e;f>0s10f0t0e;t>0s0f3t0e;e>0s0f0t0e partial=no
EVAL_PARITY=INFO pair=E1:E2 policy=mme compared=48 missing_a=0 missing_b=0 s2f=2 f2s=0 new_err=0 lost_err=0 new_timeout=0 lost_timeout=0 sr_a=27.0833 sr_b=22.9167 sr_diff_pp=-4.1667 ci=[-10.417,0] mcnemar_p=0.5 steps_equal=20/46 steps_diff_p50=0 steps_diff_p95=143.5 first_div=diverged:48/48,min:0,p50:0,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>11s2f0t0e;f>0s33f0t0e;t>0s0f2t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E1:E3 policy=mme compared=48 missing_a=0 missing_b=0 s2f=3 f2s=0 new_err=0 lost_err=0 new_timeout=1 lost_timeout=0 sr_a=27.0833 sr_b=20.8333 sr_diff_pp=-6.25 ci=[-14.583,0] mcnemar_p=0.25 steps_equal=17/45 steps_diff_p50=0 steps_diff_p95=145.4 first_div=diverged:48/48,min:0,p50:0,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>10s2f1t0e;f>0s33f0t0e;t>0s0f2t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E1:E5 policy=mme compared=48 missing_a=0 missing_b=0 s2f=3 f2s=1 new_err=0 lost_err=0 new_timeout=3 lost_timeout=0 sr_a=27.0833 sr_b=22.9167 sr_diff_pp=-4.1667 ci=[-12.5,4.167] mcnemar_p=0.625 steps_equal=13/43 steps_diff_p50=0 steps_diff_p95=150.4 first_div=diverged:48/48,min:0,p50:0,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>10s1f2t0e;f>1s31f1t0e;t>0s0f2t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E5:E6 policy=mme compared=48 missing_a=0 missing_b=0 s2f=3 f2s=3 new_err=0 lost_err=0 new_timeout=1 lost_timeout=2 sr_a=22.9167 sr_b=22.9167 sr_diff_pp=0 ci=[-10.417,10.417] mcnemar_p=1 steps_equal=10/42 steps_diff_p50=0 steps_diff_p95=143.75 first_div=diverged:48/48,min:0,p50:0,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>8s2f1t0e;f>2s30f0t0e;t>1s1f3t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=E5:E7 policy=mme compared=48 missing_a=0 missing_b=0 s2f=2 f2s=2 new_err=0 lost_err=0 new_timeout=1 lost_timeout=0 sr_a=22.9167 sr_b=22.9167 sr_diff_pp=0 ci=[-8.333,8.333] mcnemar_p=1 steps_equal=14/42 steps_diff_p50=0 steps_diff_p95=239.25 first_div=diverged:48/48,min:0,p50:0,missing:0,unit:env_step pp_per_ep=2.0833 matrix=s>9s1f1t0e;f>2s30f0t0e;t>0s0f5t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=R1:E1 policy=mme compared=48 missing_a=0 missing_b=0 s2f=2 f2s=3 new_err=0 lost_err=0 new_timeout=0 lost_timeout=3 sr_a=25 sr_b=27.0833 sr_diff_pp=2.0833 ci=[-6.25,10.417] mcnemar_p=1 steps_equal=14/43 steps_diff_p50=0 steps_diff_p95=105.8 first_div=n/a pp_per_ep=2.0833 matrix=s>10s2f0t0e;f>1s30f0t0e;t>2s1f2t0e;e>0s0f0t0e
EVAL_PARITY=INFO pair=O1:E5 policy=mme compared=48 missing_a=0 missing_b=0 s2f=2 f2s=3 new_err=0 lost_err=0 new_timeout=1 lost_timeout=1 sr_a=20.8333 sr_b=22.9167 sr_diff_pp=2.0833 ci=[-6.25,10.417] mcnemar_p=1 steps_equal=15/42 steps_diff_p50=0 steps_diff_p95=174.6 first_div=n/a pp_per_ep=2.0833 matrix=s>8s2f0t0e;f>2s30f1t0e;t>1s0f4t0e;e>0s0f0t0e
EVAL_VS_E0=INFO cond=E1 policy=mme compared=48 missing_b=0 s2f=2 f2s=6 new_err=0 new_timeout=0 success=9>13 steps_equal=18/44 matrix=s>7s2f0t0e;f>4s31f0t0e;t>2s0f2t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E2 policy=mme compared=48 missing_b=0 s2f=3 f2s=5 new_err=0 new_timeout=0 success=9>11 steps_equal=17/44 matrix=s>6s3f0t0e;f>3s32f0t0e;t>2s0f2t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E3 policy=mme compared=48 missing_b=0 s2f=3 f2s=4 new_err=0 new_timeout=0 success=9>10 steps_equal=15/44 matrix=s>6s3f0t0e;f>3s32f0t0e;t>1s0f3t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E5 policy=mme compared=48 missing_b=0 s2f=2 f2s=4 new_err=0 new_timeout=2 success=9>11 steps_equal=18/42 matrix=s>7s2f0t0e;f>3s30f2t0e;t>1s0f3t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E6 policy=mme compared=48 missing_b=0 s2f=1 f2s=3 new_err=0 new_timeout=0 success=9>11 steps_equal=19/44 matrix=s>8s1f0t0e;f>3s32f0t0e;t>0s0f4t0e;e>0s0f0t0e partial=no
EVAL_VS_E0=INFO cond=E7 policy=mme compared=48 missing_b=0 s2f=1 f2s=3 new_err=0 new_timeout=2 success=9>11 steps_equal=15/42 matrix=s>8s1f0t0e;f>3s30f2t0e;t>0s0f4t0e;e>0s0f0t0e partial=no
```

读法：`matrix=s>…` 是逐身份终态转移矩阵（行为 a 侧终态，列为 b 侧终态 s／f／t／e）；`first_div` 是首个执行动作分叉的环境步（`diverged:48/48,min:0` 表示每局第 0 步就有动作差，MME 跨进程启动的预期表现）；`EVAL_VS_E0` 只报转移、不报区间（48 局粒度 2.08 个百分点）。E5 MME、E6 SMVLA、E7 MME 的数据来自事故后的整格重跑（分别于 06:54、07:55、07:00:47 完成）；首次失败的 E6 SMVLA、E7 MME 目录改名 `s-yi-vulkanfail-0544`、`s-ding-vulkanfail-0540` 留证、不进比较。判定行取自 `step6_summary.py --final`（08:10:46 EDT）。明细：`artifacts/v7.5eval/summary/detail/eval-parity-*.json`、`E*-vs-E0-*.json`；逐局 `artifacts/v7.5eval/summary/merged/E*-{mme,smvla}.jsonl`。

## 3. 六格怎么跑的

| 条件 | 位置 | 跑法 | 编排 |
|---|---|---|---|
| E1 本机卡 0 首遍 | 本机卡 0，`taskset -c 0-3` | server 常驻、**每任务起新客户端进程**（`--client-per-task`） | 本机 `v75-chain0`（R1 → P1 → E1 → E2） |
| E2 本机卡 0 常驻倒序 | 同上 | 一个常驻客户端、倒序（`--order reverse`） | 同上 |
| E3 本机卡 1 | 本机卡 1，`taskset -c 4-7` | 同 E1 | 本机 `v75-chain1`（P3 → 开环 → E3） |
| E5 A40-乙首遍 | 乙 gl1525 | 同 E1 | GL `main` → `final`（E5 MME 补跑） |
| E6 A40-乙复跑 | 乙，另起进程 | 同 E1 | GL `main`（MME）→ `final3`（SMVLA 整格重跑，07:00～07:55） |
| E7 A40-丁 | 丁 gl1527 | 同 E1 | GL `main`（SMVLA）→ `final2`（MME 重跑） |

每格每策略 16 任务 × 1 档 × 3 局 = 48 局，每格先 SimpleMemVLA 后 MME（同卡两策略 server 不同时驻留）；所有格接在各自卡的第 4 步回放之后。

## 4. 测速

```text
EVAL_SPEED=INFO cond=E1 policy=smvla seat=L0 rows=48 episode_wall_s_p50=29.8676 episode_wall_s_p95=88.5674 env.env_build_s_p50=1.2143 env.env_build_s_p95=1.3419 env.reset_s_p50=0.9134 env.reset_s_p95=4.8867 env.step_mean_s_p50=0.0075 env.step_mean_s_p95=0.0131 recorder.encode_cpu_s_p50=4.7165 recorder.encode_cpu_s_p95=13.0336 recorder.finalize_s_p50=3.886 recorder.finalize_s_p95=9.9413
EVAL_SPEED=INFO cond=E1 policy=mme seat=L0 rows=48 episode_wall_s_p50=15.9337 episode_wall_s_p95=39.3868 env.env_build_s_p50=1.2213 env.env_build_s_p95=1.3694 env.reset_s_p50=0.8825 env.reset_s_p95=4.7881 env.step_mean_s_p50=0.0072 env.step_mean_s_p95=0.0127 policy.infer.server_steady_mean_s_p50=0.0615 policy.infer.server_steady_mean_s_p95=0.0621 policy.infer.rtt_mean_s_p50=0.0919 policy.infer.rtt_mean_s_p95=0.095 recorder.encode_cpu_s_p50=3.6665 recorder.encode_cpu_s_p95=8.8658 recorder.finalize_s_p50=3.1285 recorder.finalize_s_p95=7.9966
EVAL_SPEED=INFO cond=E5 policy=smvla seat=yi rows=48 episode_wall_s_p50=51.5124 episode_wall_s_p95=182.4677 env.env_build_s_p50=1.6366 env.env_build_s_p95=2.1679 env.reset_s_p50=1.2475 env.reset_s_p95=6.3199 env.step_mean_s_p50=0.0094 env.step_mean_s_p95=0.0192 recorder.encode_cpu_s_p50=0.707 recorder.encode_cpu_s_p95=0.7555 recorder.finalize_s_p50=0.688 recorder.finalize_s_p95=0.8285
EVAL_SPEED=INFO cond=E5 policy=mme seat=yi rows=48 episode_wall_s_p50=23.948 episode_wall_s_p95=66.2306 env.env_build_s_p50=1.6 env.env_build_s_p95=2.0909 env.reset_s_p50=1.2463 env.reset_s_p95=6.3762 env.step_mean_s_p50=0.0091 env.step_mean_s_p95=0.0195 policy.infer.server_steady_mean_s_p50=0.1096 policy.infer.server_steady_mean_s_p95=0.1098 policy.infer.rtt_mean_s_p50=0.1426 policy.infer.rtt_mean_s_p95=0.1457 recorder.encode_cpu_s_p50=0.6965 recorder.encode_cpu_s_p95=0.7419 recorder.finalize_s_p50=0.6605 recorder.finalize_s_p95=0.8042
```

E2、E3、E6、E7 各行同形，见 `verdicts.txt`（E6 SMVLA 重跑：`episode_wall_s_p50=51.5061`，与 E5 的 51.5124 几乎相同）。读法：MME 稳态推理本机约 61 ms、A40 约 105～110 ms（往返 90 ms 对 143 ms）；A40 上单局中位 SimpleMemVLA 约 52 s、MME 约 24～26 s，均快于方案 §4 的历史参考（约 68 s、约 35 s）。GL 各格录制编码 CPU 中位约 0.7 s／局，本机约 3.7～4.8 s／局，差异原因本轮未核实，列为未解项（不影响逐局结果，录制无损核对全部通过）。

## 5. 预算（P5 乘式）

第 3 步：2 策略 × 1 任务 × 1 档 × 1 局 = 2；5.1：2 策略 × 6 条件 × 16 任务 × 1 档 × 3 局 = 576。

```text
BUDGET=INFO item=3.1_跑通 attempts=2 unique=2 incident=0 retries=0 cap=2 retry_cap=共用2 over=no retry_over=yes est=中断未写结果的尝试不计(估)
BUDGET=INFO item=5.1_换条件评估 attempts=584 unique=576 incident=148 retries=156 cap=576 retry_cap=12 over=no retry_over=yes est=含事故留档目录真实局 8、infra 148
```

`unique=576` 即计划数全齐；`attempts=584` = 576 ＋ 事故留档目录里的 8 局真实结果（E6 SMVLA 7 局 ＋ E7 MME 1 局，计入尝试但不进比较）；0 步基础设施失败 148 次（E6 SMVLA 69 ＋ E7 MME 79），超出子额度 12（如实标超额）。3.1 的 `retry_over=yes` 是共用额度 2 被其他项用超所致，本项自身 0 重试。

## 6. 命令原文与会话

```bash
# 第 3 步（artifacts/v7.5eval/lanes/step3-card1.sh；与卡 1 其他 GPU 任务经 flock 串行）
flock …/gpu1.lock bash scripts/eval-official/run_seat.sh --seat L1 --seat-idx 9 --gpu 1 --cpus 4-7 \
  --cond S3 --out $R/artifacts/v7.5eval/newiface/step3 --identities $R/artifacts/v7.5eval/identities-step3-1.json \
  --policies smvla,mme --limit 1 \
  --mme-ckpt /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999 \
  --smvla-ckpt $R/artifacts/v7.5eval/ckpt/simplememvla_robomme
# 5.1 本机格（artifacts/v7.5eval/lanes/eval-local.sh <cond> <seat> <idx> <gpu> <cpus> <per-task|reverse>）
bash scripts/eval-official/run_seat.sh --seat L0 --seat-idx 10 --gpu 0 --cpus 0-3 --cond E1 \
  --out $R/artifacts/v7.5eval/newiface/E1 --identities $R/artifacts/v7.5eval/identities-small48.json \
  --policies $P --det off --client-per-task --mme-ckpt … --smvla-ckpt …
# 5.1 GL 格（编排器步骤；seat_run.sh 先写节点 /tmp，结束后 rsync -rt 暂存并以 rsync -rc 核对）
bash $N/lanes/step.sh $N/reports/E5-smvla-s-yi.json env V75_WT=a81c4f6b bash $N/lanes/seat_run.sh E5 yi 2 smvla \
  $N/replay/P5/smvla/report.json --identities $N/identities-small48.json --client-per-task
```

`$R`＝主仓库根，`$N`＝NFS `v75eval`，`$P`＝`smvla`／`mme`，`--det` 由 `det_of.sh` 从回放报告读出（全部 `off`）。编排器：`main`（E5～E7 首次）、`retry`（E5 MME 首次补跑，被事故保持步骤清理时取消）、`final`（E5 MME 整格重跑）、`final2`（E7 MME 整格重跑）、`final3`（E6 SMVLA 整格重跑，07:55 完成）。
