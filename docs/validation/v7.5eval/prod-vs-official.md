# 5.2 正式跑法全量 · 主比较（正式跑法 vs 官方）

> 对应方案 §3.2「5.2」、§5「以后在 A40 上怎么并行」、§3.1 最终相对标准。启动提交：12.272 `a81c4f6b`（GL worktree `v75eval/wt/a81c4f6b`；`run_seat.sh` 队列模式、`claim_queue.py` 认领队列）；汇总 `step6_summary.py --final`（2026-09-30 08:10:46 EDT，0 条 PENDING／BLOCKED）。确定性标志两策略均为关。车道辅助脚本在 `scripts/eval-official/v75-lanes/`，GL 侧状态、报告与日志归档在 `artifacts/v7.5eval/nfs-archive/`。用户原话：本轮原话 2、7、8；方案原话 10、22、24、25、26。

## 1. 结论

- **SimpleMemVLA**：正式跑法 192 局（多席按打乱顺序认领同一条 NFS 队列、server 常驻、每席连续跑多个任务，新1 上故意杀一次 server）与官方历史成绩、官方重跑一、官方重跑二**逐局终态全同、步数 192/192 全同**，0 翻转，成功 141 局（73.4%）。
- **MME**：正式跑法成功 51/192；官方历史成绩 50、重跑一 49、重跑二 48。对重跑一（主比较）：7 局成功→失败、9 局失败→成功，成功率差 +1.04 个百分点，配对 bootstrap 95% 置信区间 [−3.125, 5.208]，McNemar p=0.80，步数相等 82/192；对历史 6／7（+0.52 个百分点），对重跑二 4／7（+1.56 个百分点）。
- **最终相对标准**：SimpleMemVLA `inside=n/a note=官方重跑无翻转 prod_flips=none`——官方三对全 0 翻转，按规则不套区间，正式跑法的翻转逐局列出为「无」；MME **`inside=no`**：两个方向的翻转数都在官方最大值以内（7 ≤ 10、9 ≤ 9），**只因成功率差点估计 +1.04 不在官方三对的区间 [−1.04, −0.52] 内**。两策略追加样本均不触发（`trigger=no`）。解读见 [summary.md](summary.md) §2。
- **队列与续跑**：四条正式队列核对全部 `QUEUE_CLAIM=PASS dup=0 missing=0`；两策略杀 server 测试都通过（SimpleMemVLA 在正式队列上、MME 在 3 个身份的专用测试队列上：server 被杀 → 自动重起 → 认领回收 → 续跑完成、0 错误）；MME 的 101 个被 GPU 计算模式事故污染的身份（100 个 0 步 Vulkan 错误 ＋ 1 个认领后进程崩溃）经补跑队列 `prod2`（100）与 `prod3`（1）全部替换，合并 0 冲突。
- **金丝雀**（PatternLock seed 650700，官方历史成绩里 SimpleMemVLA 成功且步数最短的 xhard0 身份，两策略共用）：两策略各 8 席、8/8 与正式跑法、重跑一、历史成绩终态相同，`seats_missing_real=none`。

## 2. 判定行原文

```text
QUEUE_CLAIM=PASS dup=0 missing=0 requeued=4 done=192 total=192 late=0 open_claims=0 retries_used=4 infra_exhausted=0 queue=prod policy=smvla infra_terminal=0
PROD_MERGE=INFO policy=smvla from_N=192 from_N2=0 from_N3=0 poisoned_replaced=0 still_missing=0 conflicts=0 poisoned_unreplaced=0
PROD_VS_OFFICIAL=INFO ref=重跑一 policy=smvla compared=192 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 sr_diff_pp=0 ci=[0,0] mcnemar_p=1
PROD_VS_OFFICIAL=INFO ref=历史 policy=smvla compared=192 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 sr_diff_pp=0 ci=[0,0] mcnemar_p=1
PROD_VS_OFFICIAL=INFO ref=重跑二 policy=smvla compared=192 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 sr_diff_pp=0 ci=[0,0] mcnemar_p=1
RELATIVE_ACCEPT=INFO policy=smvla inside=n/a s2f=0 f2s=0 max_off_s2f=0 max_off_f2s=0 sr_diff_pp=0 off_sr_range=[0,0] ci=[0,0] off_ci=历史:重跑一[0,0];历史:重跑二[0,0];重跑一:重跑二[0,0] extra_sample_trigger=no note=官方重跑无翻转 prod_flips=none
EXTRA_SAMPLE=INFO policy=smvla trigger=no main_ci_half_width_pp=0 official_max_ci_half_width_pp=0 rule=主比较CI半宽>2×官方3对最大半宽
CANARY=INFO policy=smvla n=8 seats=bing,ding,jia,new1,new2,new3,new4,yi from=bing:N,ding:C,jia:N,new1:N,new2:N,new3:N,new4:N,yi:N status_eq_N=8/8 steps_eq_N=8/8 status_eq_O1=8/8 status_eq_E0=8/8 seats_missing_real=none
QUEUE_CLAIM=PASS dup=0 missing=0 requeued=10 done=192 total=192 late=0 open_claims=0 retries_used=8 infra_exhausted=101 queue=prod policy=mme infra_terminal=101
QUEUE_CLAIM=PASS dup=0 missing=0 requeued=0 done=100 total=100 late=0 open_claims=0 retries_used=0 infra_exhausted=0 queue=prod2 policy=mme infra_terminal=0
QUEUE_CLAIM=PASS dup=0 missing=0 requeued=0 done=1 total=1 late=0 open_claims=0 retries_used=0 infra_exhausted=0 queue=prod3 policy=mme infra_terminal=0
PROD_MERGE=INFO policy=mme from_N=91 from_N2=100 from_N3=1 poisoned_replaced=101 still_missing=0 conflicts=0 poisoned_unreplaced=0
PROD_VS_OFFICIAL=INFO ref=重跑一 policy=mme compared=192 missing_a=0 missing_b=0 s2f=7 f2s=9 new_err=0 new_timeout=1 sr_diff_pp=1.0417 ci=[-3.125,5.208] mcnemar_p=0.8036
PROD_VS_OFFICIAL=INFO ref=历史 policy=mme compared=192 missing_a=0 missing_b=0 s2f=6 f2s=7 new_err=0 new_timeout=1 sr_diff_pp=0.5208 ci=[-3.125,4.167] mcnemar_p=1
PROD_VS_OFFICIAL=INFO ref=重跑二 policy=mme compared=192 missing_a=0 missing_b=0 s2f=4 f2s=7 new_err=0 new_timeout=0 sr_diff_pp=1.5625 ci=[-1.562,5.208] mcnemar_p=0.5488
RELATIVE_ACCEPT=INFO policy=mme inside=no s2f=7 f2s=9 max_off_s2f=10 max_off_f2s=9 sr_diff_pp=1.0417 off_sr_range=[-1.0417,-0.5208] ci=[-3.125,5.208] off_ci=历史:重跑一[-3.125,2.083];历史:重跑二[-5.208,3.125];重跑一:重跑二[-5.208,4.167] extra_sample_trigger=no
EXTRA_SAMPLE=INFO policy=mme trigger=no main_ci_half_width_pp=4.1667 official_max_ci_half_width_pp=4.6875 rule=主比较CI半宽>2×官方3对最大半宽
CANARY=INFO policy=mme n=8 seats=bing,ding,jia,new1,new2,new3,new4,yi from=bing:N,ding:N2,jia:N,new1:C,new2:C,new3:N2,new4:N2,yi:N status_eq_N=8/8 steps_eq_N=8/8 status_eq_O1=8/8 status_eq_E0=8/8 seats_missing_real=none
KILLTEST=INFO policy=mme seat=new1 cond=K server_killed=yes server_restarted=yes requeued=1 done=3/3 errors=0 seat_infra=0 queue_check=PASS log=K-mme-s-new1.log
KILLTEST=INFO policy=smvla seat=new1 cond=N server_killed=yes server_restarted=yes requeued=4 done=46 errors=0 seat_infra=3 queue_check=PASS log=N-smvla-s-new1.log
INFRA=INFO cond=N policy=smvla seat=new1 n=3 zero_step=2 incident=2 identities=1 sigs=connectionrefused:2;websocket:1 hist=connection:ConnectionRefusedError|ConnectionRefusedError:[Errno_111]_Connection_refused:2;connection:ConnectionClosedError|websockets.exceptions.ConnectionClosedError:no_close_frame_r:1
INFRA=INFO cond=N policy=mme seat=None n=1 zero_step=1 incident=0 identities=1 sigs=none:1 hist=n/a|认领后进程崩溃，无结果:1
INFRA=INFO cond=N policy=mme seat=new1 n=27 zero_step=27 incident=27 identities=27 sigs=createdeviceunique:27 hist=env_build|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:27
INFRA=INFO cond=N policy=mme seat=new2 n=29 zero_step=29 incident=29 identities=29 sigs=createdeviceunique:29 hist=env_build|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:29
INFRA=INFO cond=N policy=mme seat=new4 n=48 zero_step=48 incident=48 identities=44 sigs=createdeviceunique:48 hist=env_build|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:48
INFRA=INFO cond=N2 policy=mme seat=all n=0
INFRA=INFO cond=N3 policy=mme seat=all n=0
INCIDENT=INFO queue=prod policy=mme infra_terminal=101 seats=None,new1,new2,new4 zero_step=101 incident=100 replaced=101 unreplaced=0 hist=env_build|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:100;n/a|认领后进程崩溃，无结果:1
```

明细：`artifacts/v7.5eval/summary/detail/prod-vs-official-{mme,smvla}.json`（含每一局的翻转、两侧来源文件行号）、`relative-accept-*.json`、`canary-*.json`；合并后逐局 `artifacts/v7.5eval/summary/merged/NP-{mme,smvla}.jsonl`。

## 3. 读法

- **条件标签**：`N` 是原正式跑法队列 `queue/prod`；`N2` 是 100 个被污染 MME 身份的补跑队列 `queue/prod2`；`N3` 是那 1 个「认领后进程崩溃」身份（InsertPeg seed 630300）的单身份队列 `queue/prod3`；`C` 是事后给缺真实金丝雀的席位补跑金丝雀（丁 SMVLA、新1 MME、新2 MME）；`K` 是 MME 杀 server 测试（新1，3 个身份的专用队列，`identities-killtest3.json`，不是正式结果）。汇总按身份合并 N、N2、N3，只取真实终态，冲突即 FAIL（`PROD_MERGE`）；五者的运行口径（worktree `a81c4f6b`、`seat_run.sh`、确定性关）相同。
- **谁实际消费了队列**：SimpleMemVLA 的 192 局全部由新1～新4 四席消费（`EVAL_SPEED … rows=46／47／47／52`），04:07:19 起、约 05:08:52 队列清空（约 62 min，含四席各自 server 加载）；乙、丁、甲、丙加入时队列已空，只跑了金丝雀。MME 真实结果由丁、新3、新4 三席产生（N 50＋21＋20 = 91，N2 39＋32＋29 = 100，N3 新3 1），新1、新2 的 N 步骤只产出事故 0 步行；MME 从 05:15:22 入队到 06:40:13 最后 1 个身份完成约 85 min，中间含事故处置与补跑。甲、丙的 N 步骤是 03:14 就起跑、在 srun 里等待的步骤，05:51 事故处置时被取消，07:09～07:17 重起时队列已空、只跑了金丝雀。方案「8 席凑满」的并发形态因此没有实测到，实际同时消费队列的最多 4 席。
- **SimpleMemVLA 的 0 翻转意味着什么**：正式跑法与官方评估在接口（新 websocket 宿主 vs 旧进程内评估）、环境代码（`robomme_hard` xhard0 vs 官方 `robomme`）、sapien 3.0.2 vs 3.0.3、环境侧 torch 2.9.1 vs 2.4.1、调度（队列乱序、常驻进程、杀 server 续跑）同时不同，而三份官方结果的 192 局终态与步数全同——在 A40 上这些差异对 SimpleMemVLA 的闭环结果贡献为 0。前提是每局重设种子（方案已定口径 5）与同型号 GPU。
- **MME 的差距为什么不能归因**：MME 跨 server 启动动作就有小差（[policy-replay.md](policy-replay.md)），闭环放大后同卡重跑也翻；主比较的 7／9 翻转与官方重跑两两的 3～10 翻转同源，无法拆出接口本身的贡献。

## 4. 测速（每席）

```text
EVAL_SPEED=INFO cond=N policy=smvla seat=new1 rows=46 episode_wall_s_p50=45.5689 episode_wall_s_p95=183.3834 env.env_build_s_p50=1.7505 env.env_build_s_p95=2.0146 env.reset_s_p50=2.0084 env.reset_s_p95=8.1512 env.step_mean_s_p50=0.0106 env.step_mean_s_p95=0.0262 recorder.encode_cpu_s_p50=0.8365 recorder.encode_cpu_s_p95=0.8915 recorder.finalize_s_p50=0.7535 recorder.finalize_s_p95=0.9185
EVAL_SPEED=INFO cond=N policy=mme seat=ding rows=50 episode_wall_s_p50=23.1363 episode_wall_s_p95=55.1512 env.env_build_s_p50=1.6745 env.env_build_s_p95=1.8525 env.reset_s_p50=1.8863 env.reset_s_p95=7.7736 env.step_mean_s_p50=0.0096 env.step_mean_s_p95=0.019 policy.infer.server_steady_mean_s_p50=0.1096 policy.infer.server_steady_mean_s_p95=0.1101 policy.infer.rtt_mean_s_p50=0.144 policy.infer.rtt_mean_s_p95=0.1478 recorder.encode_cpu_s_p50=0.6985 recorder.encode_cpu_s_p95=0.7452 recorder.finalize_s_p50=0.6585 recorder.finalize_s_p95=0.7755
```

其余各席（SMVLA 新2～新4；MME 新3、新4，N2 丁／新3／新4，N3 新3）同形，见 `verdicts.txt`。SimpleMemVLA 单局中位 45.6～55.4 s，MME 21.0～28.9 s；队列墙钟见 §3。

## 5. 预算（P5 乘式）

计划：5.2 为 2 策略 × 16 任务 × 1 档 × 12 局 = 384；金丝雀 8 席 × 2 策略 × 1 局 = 16。

```text
BUDGET=INFO item=5.2_正式跑法 attempts=386 unique=384 incident=106 retries=108 cap=384 retry_cap=8 over=no retry_over=yes est=含补跑队列 N2、N3
BUDGET=INFO item=5.2_金丝雀 attempts=22 unique=16 incident=0 retries=6 cap=16 retry_cap=共用2 over=yes retry_over=yes est=含 C 金丝雀补跑与 K 测试的金丝雀；重复席位计为重试
BUDGET=INFO item=其他_杀server测试 attempts=3 unique=3 incident=0 retries=0 cap=3 retry_cap=共用2 over=no retry_over=yes est=K：MME 杀 server 测试的队列局（不是正式结果）
```

- 5.2 轨迹尝试 386 = 去重身份 384 ＋ 2 次非事故重试；0 步基础设施失败 106 次（MME 新1 27 ＋ 新2 29 ＋ 新4 48，SMVLA 新1 杀 server 测试的连接拒绝 2），重试合计 108 次，超出子额度 8（如实标超额）。
- **金丝雀 22 次，超出计划 16 次**（`over=yes`）：同一席位在 N、N2、C、K 各起一次 server 时都会先跑 1 局金丝雀，重复的 6 次计为重试。
- 杀 server 测试 K：1 策略 × 3 身份 × 1 局 = 3，方案预算表未单列这一项（原计划在正式队列上做），实际另建 3 身份专用队列，须用户追认（见 [summary.md](summary.md) §6）。

## 6. 命令原文与会话

```bash
# 主编排器（登录节点 tmux 内；编排席 62612889）：scripts/eval-official/v75-lanes/gl/orch_main.sh
srun --jobid=62612889 --overlap --ntasks=1 --cpus-per-task=1 $PY $N/wt/30257b46/scripts/eval-official/orchestrate.py \
  --plan $N/state/main-plan.json --state $N/state/main --workdir $N
# 每席每策略一个步骤（例：新1 MME；--queue 指向 prod／prod2／prod3）
bash $N/lanes/step.sh $N/reports/N-mme-s-new1.json env V75_WT=a81c4f6b bash $N/lanes/seat_run.sh N new1 5 mme \
  $N/replay/P5/mme/report.json --queue $N/queue/prod --canary PatternLock:7:650700
bash $N/lanes/step.sh $N/reports/N2-mme-s-ding.json env V75_WT=a81c4f6b bash $N/lanes/seat_run.sh N2 ding 4 mme \
  $N/replay/P5/mme/report.json --queue $N/queue/prod2 --canary PatternLock:7:650700
bash $N/lanes/step.sh $N/reports/N3-mme-s-new3.json env V75_WT=a81c4f6b bash $N/lanes/seat_run.sh N3 new3 7 mme \
  $N/replay/P5/mme/report.json --queue $N/queue/prod3
# MME 杀 server 测试（final7，新1）：150 s 后杀掉 serve_policy 进程
bash -c "( sleep 150; pkill -9 -f '[s]erve_policy.py --seed=7' && echo SERVER_KILLED_FOR_TEST ) & exec bash $N/lanes/step.sh $N/reports/K-mme-s-new1.json env V75_WT=a81c4f6b bash $N/lanes/seat_run.sh K new1 …"
# 金丝雀补跑（final7／final8）
bash $N/lanes/step.sh $N/reports/C-smvla-s-ding.json env V75_WT=a81c4f6b bash $N/lanes/seat_run.sh C ding 4 smvla $N/replay/P5/smvla/report.json …
```

`$N`＝`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v75eval`（运行时路径，小文件已归档到 `artifacts/v7.5eval/nfs-archive/`，`$N/lanes/` 即 `scripts/eval-official/v75-lanes/gl/` 的最终版）。K、C 两条命令里 `…` 处的其余参数以归档的计划文件 `artifacts/v7.5eval/nfs-archive/state/final7-plan.json`、`final8-plan.json` 原文为准。编排器实例：`main`（N 首轮，八席车道）、`mmeprod`（新1～新4 的 MME 首轮，先 `restage` 再入队）、`final`（N2：丁、新3、新4）、`final5`（N3：新3）、`final7`（K：新1 07:35:02～07:41:29；C：新2 MME 07:29:32～07:32:00）、`final8`（C：丁 SMVLA、新1 MME，08:01:43～08:06:16）。SimpleMemVLA 杀 server 日志 `artifacts/v7.5eval/nfs-archive/state/main/logs/N-smvla-s-new1.log`（`SERVER_KILLED_FOR_TEST` → `SERVER_DIED policy=smvla` → `SERVER_READY … ready_s=236` → `CLIENT_EXIT policy=smvla rc=0`）；MME 杀 server 日志 `artifacts/v7.5eval/nfs-archive/state/final7/logs/K-mme-s-new1.log`（见 `KILLTEST` 行）。
