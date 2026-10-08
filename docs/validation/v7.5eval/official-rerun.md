# 2.2 录制器验证 · 2.3 官方自己重跑两遍（官方噪声带）· 2.4 本机旧官方小样本

> 对应方案 §3.2「2.2」「2.3」「2.4」、§2.5 旧官方录制器、§3.1 官方噪声带。启动提交：2.2 与 2.3 为 12.269 `4c19c39a`（GL worktree `v75eval/wt/4c19c39a`，片级脚本 `scripts/eval-official/official_observer/official_rerun_shard.sh`）；2.4 为 12.270 `2abf227d`（本机 worktree `artifacts/v7.5eval/wt/2abf227d`）。旧官方代码只读：MME `robomme_policy_learning-official-xhard0`（`927c56d`），SimpleMemVLA `SimpleMemVLA-official-xhard0`（`4e0c04f`）。车道辅助脚本保存在 `scripts/eval-official/v75-lanes/`，GL 侧状态、报告与日志归档在 `artifacts/v7.5eval/nfs-archive/`（NFS 原件已删）。用户原话：本轮原话 2、7、8；方案原话 4、14、23（d4a、官方噪声带）、25。

## 1. 结论

- **录制器不改结果**：SimpleMemVLA 同一身份（PickXtimes ep3 seed 510300）在 A40 上「原版启动器、不带录制器」与「官方重跑一、带录制器」两局，官方 mp4 逐字节相同、816/816 帧相同、终态与步数（成功 815 步）相同，也与官方历史成绩相同。MME 只做结构性证明：透明代理在 GL 上 22 次核对全部 `mismatch=0`；逐局视频 807 次解码核对全部 `decode_mismatch=0 dropped=0 reordered=0`，全程无存储降级（`level=0`）。
- **官方噪声带（终值，两遍各 2 策略 × 16 任务 × 1 档 × 12 局 = 384，全齐）**：SimpleMemVLA 官方三对（各 192 局）**全部 0 翻转**，成功率差 0——旧官方评估在 A40 上对 SimpleMemVLA 逐局可复现。MME 官方自己重跑就会翻：历史:重跑一 4 局成功→失败、3 局失败→成功（−0.52 个百分点）；历史:重跑二 9／7（−1.04 个百分点）；重跑一:重跑二 10／9（−0.52 个百分点）；三对 McNemar 均不显著（p ≥ 0.80）。三份的成功局数：历史 50、重跑一 49、重跑二 48。
- **本机旧官方小样本（RTX）对官方历史成绩（A40）**：SimpleMemVLA 1 局成功→失败、2 局失败→成功；MME 1／4。这是换 GPU 型号的差，与 [env-parity.md](env-parity.md) 的 C1:C5 结论一致。

## 2. 判定行原文

2.2：

```text
OBSERVER_SMOKE=INFO identity=PickXtimes/510300 file_equal=yes frames_equal=yes frames=816/816 first_diff_frame=n/a status_equal=yes steps_equal=yes steps=815/815
OBSERVER_TRANSPARENT=PASS messages=195 mismatch=0 conns=1 unmatched_conns=0 proxy_only_conns=0
```

第二行是本机卡 1 冒烟（12.269 commit body）；GL 上官方重跑各片 MME 段共 22 行 `OBSERVER_TRANSPARENT=PASS … mismatch=0 … unmatched_conns=0 proxy_only_conns=0`（`messages` 467～2305、`conns` 5～20，见 `artifacts/v7.5eval/nfs-archive/state/{official,o2x,final,final4,final6}/logs/O*.log`），另 807 行 `RECORDER_VERIFY=PASS … decode_mismatch=0 dropped=0 reordered=0 … level=0`。

2.3 与 2.4（`step6_summary.py --final`，2026-09-30 08:10:46 EDT，`artifacts/v7.5eval/summary/verdicts.txt`）：

```text
E0_SELF=INFO policy=smvla compared=192 missing=0 s2f=0 f2s=0 sr_pct=73.4375 success=141 files=10 lines=192 superseded=0 sanity=ok
OFFICIAL_RERUN=INFO run=O1 policy=smvla have=192/192
OFFICIAL_RERUN=INFO run=O2 policy=smvla have=192/192
OFFICIAL_NOISE=INFO pair=历史:重跑一 policy=smvla compared=192 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 sr_diff_pp=0 ci=[0,0] mcnemar_p=1
OFFICIAL_NOISE=INFO pair=历史:重跑二 policy=smvla compared=192 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 sr_diff_pp=0 ci=[0,0] mcnemar_p=1
OFFICIAL_NOISE=INFO pair=重跑一:重跑二 policy=smvla compared=192 missing_a=0 missing_b=0 s2f=0 f2s=0 new_err=0 new_timeout=0 sr_diff_pp=0 ci=[0,0] mcnemar_p=1
OFFICIAL_VS_E0=INFO cond=R1 policy=smvla compared=48 missing_b=0 s2f=1 f2s=2 new_err=0 new_timeout=1 success=35>36 steps_equal=22/44 matrix=s>34s0f1t0e;f>2s8f0t0e;t>0s2f1t0e;e>0s0f0t0e partial=no
E0_SELF=INFO policy=mme compared=192 missing=0 s2f=0 f2s=0 sr_pct=26.0417 success=50 files=10 lines=192 superseded=0 sanity=ok
OFFICIAL_RERUN=INFO run=O1 policy=mme have=192/192
OFFICIAL_RERUN=INFO run=O2 policy=mme have=192/192
OFFICIAL_NOISE=INFO pair=历史:重跑一 policy=mme compared=192 missing_a=0 missing_b=0 s2f=4 f2s=3 new_err=0 new_timeout=1 sr_diff_pp=-0.5208 ci=[-3.125,2.083] mcnemar_p=1
OFFICIAL_NOISE=INFO pair=历史:重跑二 policy=mme compared=192 missing_a=0 missing_b=0 s2f=9 f2s=7 new_err=0 new_timeout=1 sr_diff_pp=-1.0417 ci=[-5.208,3.125] mcnemar_p=0.8036
OFFICIAL_NOISE=INFO pair=重跑一:重跑二 policy=mme compared=192 missing_a=0 missing_b=0 s2f=10 f2s=9 new_err=0 new_timeout=1 sr_diff_pp=-0.5208 ci=[-5.208,4.167] mcnemar_p=1
OFFICIAL_VS_E0=INFO cond=R1 policy=mme compared=48 missing_b=0 s2f=1 f2s=4 new_err=0 new_timeout=2 success=9>12 steps_equal=13/42 matrix=s>8s1f0t0e;f>3s30f2t0e;t>1s0f3t0e;e>0s0f0t0e partial=no
```

明细：`artifacts/v7.5eval/summary/detail/official-noise-{mme,smvla}.json`（逐局翻转与两侧来源行号）、`R1-vs-E0-{mme,smvla}.json`、`observer-smoke.json`。

## 3. 怎么跑的

- **分片与顺序**：照官方历史成绩的 10 片（每片 19～20 局）、片内顺序（MME `xhard0_manifest.py::load_rows`，SMVLA 按 (task, source_episode) 排序），每片一个新进程、先 SimpleMemVLA 后 MME。按历史片墙钟均衡：甲跑片 5、1、3、7、0（SMVLA 历史合计 6,576 s），丙跑片 9、2、4、8、6（6,209 s）。重跑一全部按此分配跑完；重跑二改派后的实际位置见下。
- **与历史一致的运行条件**：4 CPU、A40、驱动 595.71.05、`max_steps=1300`、`obs_horizon=16`、MME server 只设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`（不开确定性标志与编译缓存）、每局录视频；输出一律新目录，起跑前断言无 `episodes*.jsonl`，起止各核 E0 sha256（`E0_FREEZE_CHECK=PASS when=before|after`）。
- **与历史不一致、须写进结论的**：
  1. 节点：历史分布在 gl1512／gl1527／gl1525／gl1506／gl1518；重跑一全部在 gl1513（甲、丙同节点并发，方案风险 K2）；**重跑二改派后分布在三个节点**——gl1513（甲片 5、丙片 9）、gl1512（新1：片 1、3、6）、gl1528（新2：片 0、2、4；新3：片 7；新4：片 8）；片 1、2 首次在 gl1527（丁）起跑后被取消。三份的节点组成互不相同，节点效应混入官方噪声带。
  2. 视频先写节点 `/tmp`。
  3. 加了录制器。
  4. MME 旧启动器写死 `PYTHONPATH`，另写包装启动器 `mme_client_wrap.py`；SMVLA 以真名加载 `robomme_sim.eval_success` 后调 `main()`（与 `-m` 仅 `__name__` 不同）。
- **事故中的分片**（[incidents.md](incidents.md) 事故 6）：重跑一片 0（6 局）、片 6（11 局）的 SMVLA 首遍出现 0 步 Vulkan 错误，由旧启动器自带的 `--resume` 重试补齐。重跑二片 3、4、7、8 的 SMVLA 首遍 19 局全错、旧启动器内部第二遍 `--resume` 当场补齐（其后编排器 `final4` 的 SMVLA 续跑步骤启动时 `pending=0`，是空操作）；MME 段事故时只跑了 6、7、14、12 局，用历史 `done_keys` 机制续跑到每片 19 局。

```text
INFRA=INFO cond=O1 policy=smvla seat=s0 n=6 zero_step=6 incident=6 identities=6 sigs=createdeviceunique:6 hist=n/a|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:6
INFRA=INFO cond=O1 policy=smvla seat=s6 n=11 zero_step=11 incident=11 identities=11 sigs=createdeviceunique:11 hist=n/a|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:11
INFRA=INFO cond=O2 policy=smvla seat=s3 n=19 zero_step=19 incident=19 identities=19 sigs=createdeviceunique:19 hist=n/a|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:19
INFRA=INFO cond=O2 policy=smvla seat=s4 n=19 zero_step=19 incident=19 identities=19 sigs=createdeviceunique:19 hist=n/a|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:19
INFRA=INFO cond=O2 policy=smvla seat=s7 n=19 zero_step=19 incident=19 identities=19 sigs=createdeviceunique:19 hist=n/a|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:19
INFRA=INFO cond=O2 policy=smvla seat=s8 n=19 zero_step=19 incident=19 identities=19 sigs=createdeviceunique:19 hist=n/a|RuntimeError:vk::PhysicalDevice::createDeviceUnique:_ErrorIn:19
INFRA=INFO cond=O1 policy=mme seat=all n=0
INFRA=INFO cond=O2 policy=mme seat=all n=0
OFFICIAL_SHARD_DONE run=O2 shard=3 smvla_rows=19 mme_rows=6 smvla_rc=0 mme_rc=1 staged=yes
MME_RESUME_DONE run=O2 shard=3 final_rows=19 rc=0 staged=yes
MME_RESUME_DONE run=O2 shard=4 final_rows=19 rc=0 staged=yes
MME_RESUME_DONE run=O2 shard=7 final_rows=19 rc=0 staged=yes
MME_RESUME_DONE run=O2 shard=8 final_rows=19 rc=0 staged=yes
SMVLA_RESUME_DONE run=O2 shard=3 final_rows=19 rc=0 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=0 smvla_rows=20 mme_rows=20 smvla_rc=0 mme_rc=0 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=1 smvla_rows=20 mme_rows=20 smvla_rc=0 mme_rc=0 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=2 smvla_rows=19 mme_rows=19 smvla_rc=0 mme_rc=0 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=5 smvla_rows=19 mme_rows=19 smvla_rc=0 mme_rc=0 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=6 smvla_rows=19 mme_rows=19 smvla_rc=0 mme_rc=0 staged=yes
OFFICIAL_SHARD_DONE run=O2 shard=9 smvla_rows=19 mme_rows=19 smvla_rc=0 mme_rc=0 staged=yes
```

0 步行 `infra=True`，永不计为策略结果；汇总按身份取真实终态（`OFFICIAL_RERUN … have=192/192`）。片 4、7、8 的 `OFFICIAL_SHARD_DONE`／`SMVLA_RESUME_DONE` 同形（SMVLA 19 行；MME 首遍 7、14、12 行），见 `artifacts/v7.5eval/nfs-archive/state/*/logs/O2*.log`。

## 4. 测速

```text
EVAL_SPEED=INFO cond=E0 policy=smvla seat=all rows=192 elapsed_s_p50=47.25 elapsed_s_p95=192.27
EVAL_SPEED=INFO cond=O1 policy=smvla seat=all rows=192 elapsed_s_p50=56 elapsed_s_p95=215.525
EVAL_SPEED=INFO cond=O2 policy=smvla seat=all rows=192 elapsed_s_p50=54.4 elapsed_s_p95=210.83
EVAL_SPEED=INFO cond=R1 policy=smvla seat=all rows=48 elapsed_s_p50=35.5 elapsed_s_p95=104.94
```

两遍重跑 SMVLA 单局中位比历史慢约 15%～19%（录制开销 ＋ 同节点并发），在方案预估的 10～30% 录制开销内；MME 旧日志无逐局耗时字段，只有局数。片墙钟：重跑一各片 2,466～3,499 s，重跑二片 5、9 为 3,405 s、2,830 s（`artifacts/v7.5eval/nfs-archive/state/official/events.log` 的 `elapsed_s`）。录制体积约 105～129 MB／局（FFV1，约 76 KB／帧）。

## 5. 预算（P5 乘式）

计划：2.2 为 1 策略 × 1 身份 × 1 局 = 1；2.3 为 2 策略 × 2 遍 × 16 任务 × 1 档 × 12 局 = 768；2.4 为 2 策略 × 1 条件 × 16 × 1 × 3 = 96。

```text
BUDGET=INFO item=2.2_录制器验证 attempts=1 unique=1 incident=0 retries=0 cap=1 retry_cap=共用2 over=no retry_over=yes est=none
BUDGET=INFO item=2.3_官方重跑两遍 attempts=768 unique=768 incident=93 retries=93 cap=768 retry_cap=12 over=no retry_over=yes est=按逐局文件行数;启动器内部整遍重评未落行的不计(估)
BUDGET=INFO item=2.4_本机旧官方小样本 attempts=96 unique=96 incident=0 retries=0 cap=96 retry_cap=4 over=no retry_over=no est=同上(估)
```

2.3 轨迹尝试恰为计划数 768；0 步基础设施重试 93 次（重跑一 17 ＋ 重跑二 76），超出子额度 12，全部来自 GPU 计算模式事故（如实标超额）。2.2 的 `retry_over=yes` 是共用额度 2 被其他项用超所致（`BUDGET_TOTAL … shared_retries=6 shared_retry_cap=2`），本项自身 0 重试。2.4 的 4 次重试额度已被开发期冒烟用满（MME 2 局、SMVLA 2 局，12.269）。

## 6. 命令原文与会话

```bash
# 编排器（登录节点 tmux 内；编排席 62612889）：scripts/eval-official/v75-lanes/gl/orch_official.sh
srun --jobid=62612889 --overlap --ntasks=1 --cpus-per-task=1 $PY $N/wt/4c19c39a/scripts/eval-official/orchestrate.py \
  --plan $N/state/official-plan.json --state $N/state/official --workdir $N
# 每片（编排器派发到甲／丙；step.sh 写报告 JSON）
cd $N/wt/4c19c39a && bash $N/lanes/step.sh $N/reports/O1-s5.json \
  bash $N/wt/4c19c39a/scripts/eval-official/official_observer/official_rerun_shard.sh O1 5 --port-base 20000
# 2.2 不带录制器 1 局（丙）：v75-lanes/gl/norec_smvla.sh，内部调用原版启动器
MANIFEST=$N/official/manifest-norec-1row.jsonl SHARD=0/1 OUTDIR=$N/official/norec/smvla VIDEO_DIR=$N/official/norec/smvla-videos \
  bash $O/scripts/run_official_xhard0.sh --resume
# 重跑二续跑（final／final4 编排器）
bash $N/lanes/step.sh $N/reports/O2-s3-mmeresume.json bash $N/lanes/official_mme_resume.sh O2 3 27100
# 2.4 本机卡 0（v75-lanes/local/r1-card0.sh，tmux v75-r1-card0）
MANIFEST=$N/official/manifest-small48.jsonl taskset -c 0-3 bash scripts/eval-official/official_observer/official_rerun_shard.sh R1smoke 0 --port-base 19000
```

`$N`＝`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v75eval`（运行时路径；小文件已归档到 `artifacts/v7.5eval/nfs-archive/`，`$N/lanes/` 下的脚本即 `scripts/eval-official/v75-lanes/gl/` 的最终版），`$O`＝`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA-official-xhard0`，`$PY`＝NFS 上钉死的 `cpython-3.11.14`。编排器实例：`official`（甲、丙两车道；重跑二片 5 于 07:08:42、片 9 于 07:09:04 完成）、`o2x`（04:56 起改派片 1、2、3、4、7、8）、`final`／`final4`（事故后续跑与片 1、2 整片重跑）、`final6`（06:52 起把片 0、6 改派到新2、新1，07:29:16、07:34:49 完成）。改派靠 `skip/O2-s<n>.skip` 让 `official` 编排器里的原步骤直接跳过（`STEP_SKIPPED_REASSIGNED`，`STEP_RC=3`），skip 文件原文「改派到其他 A40 席位以缩短关键路径（用户授权「尽可能并行」）」「改派到空闲的新席位以缩短关键路径（用户授权「尽可能并行」）」（本轮原话 7、8）。
