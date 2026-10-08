# 第 6 步结论（相对标准 · 官方噪声带 · 测到与没测到 · 偏离方案 · 预算）

> 对应方案 §3.1「最终相对标准」、§3.2「第 6 步」、§3.4 预算、盲区清单。汇总命令 `uv run --no-sync python scripts/eval-official/step6_summary.py --final`（2026-09-30 08:10:46 EDT 生成，rc=0，0 条 PENDING／BLOCKED；日志 `artifacts/v7.5eval/logs/step6-final.log`，输出 `artifacts/v7.5eval/summary/`）。汇总脚本终版与本留档同一提交落地。用户原话：本轮原话 1～8；方案原话 19、20、22、23、26、28。

## 1. 一句话结论（按策略）

- **SimpleMemVLA**：在 A40 上，新接口正式跑法与旧官方评估**逐局相同**——192 局终态与步数和官方历史成绩、官方重跑一、官方重跑二全部相同（0 翻转），官方重跑本身三对也全 0 翻转。按方案规则属「官方重跑无翻转」：不套区间、不放宽，正式跑法的翻转逐局列出为「无」（`prod_flips=none`）。
- **MME**：正式跑法对官方重跑一 7 局成功→失败、9 局失败→成功，成功率 +1.04 个百分点（95% 置信区间 [−3.125, 5.208]，McNemar p=0.80）。翻转规模落在官方自己两两重跑之内（官方最大 10／9），但成功率差点估计 +1.04 不在官方三对的区间 [−1.04, −0.52] 内，按跑前写死的相对标准判 **`inside=no`**。结论交用户判断，读法见 §2。
- **环境与策略结论**：同 GPU 型号内环境逐字节可复现、换型号不可；旧／新环境栈 reset 逐字节相同；SimpleMemVLA 关态同卡逐位可复现；MME 同 server 逐位、跨 server 启动不逐位（最大动作差 0.0096 < 0.0413），确定性开态逐位但减速约五成；新旧接口在同一输入下模型输入字节与执行动作全等。

## 2. 最终相对标准（`RELATIVE_ACCEPT`）

规则（方案 §3.1，跑前写死）：对每个策略，官方噪声带 = 历史、重跑一、重跑二两两配对的 3 对；主比较 = 正式跑法 vs 重跑一；`inside=yes` 当且仅当主比较每个方向的翻转数 ≤ 官方 3 对中该方向的最大翻转数，**且**主比较成功率差的点估计落在官方 3 对成功率差的最小值与最大值之间；官方 3 对全 0 翻转时写「官方重跑无翻转」，不套区间。追加样本触发条件：主比较置信区间半宽 > 官方 3 对最大半宽的 2 倍。

```text
RELATIVE_ACCEPT=INFO policy=smvla inside=n/a s2f=0 f2s=0 max_off_s2f=0 max_off_f2s=0 sr_diff_pp=0 off_sr_range=[0,0] ci=[0,0] off_ci=历史:重跑一[0,0];历史:重跑二[0,0];重跑一:重跑二[0,0] extra_sample_trigger=no note=官方重跑无翻转 prod_flips=none
EXTRA_SAMPLE=INFO policy=smvla trigger=no main_ci_half_width_pp=0 official_max_ci_half_width_pp=0 rule=主比较CI半宽>2×官方3对最大半宽
RELATIVE_ACCEPT=INFO policy=mme inside=no s2f=7 f2s=9 max_off_s2f=10 max_off_f2s=9 sr_diff_pp=1.0417 off_sr_range=[-1.0417,-0.5208] ci=[-3.125,5.208] off_ci=历史:重跑一[-3.125,2.083];历史:重跑二[-5.208,3.125];重跑一:重跑二[-5.208,4.167] extra_sample_trigger=no
EXTRA_SAMPLE=INFO policy=mme trigger=no main_ci_half_width_pp=4.1667 official_max_ci_half_width_pp=4.6875 rule=主比较CI半宽>2×官方3对最大半宽
```

MME 三个条件逐项：

| 量 | 历史:重跑一 | 历史:重跑二 | 重跑一:重跑二 | 官方 3 对 | 主比较 | 是否满足 |
|---|---:|---:|---:|---|---:|---|
| 成功→失败 | 4 | 9 | 10 | 最大 10 | 7 | 7 ≤ 10，满足 |
| 失败→成功 | 3 | 7 | 9 | 最大 9 | 9 | 9 ≤ 9，满足 |
| 成功率差（百分点） | −0.5208 | −1.0417 | −0.5208 | 区间 [−1.0417, −0.5208] | +1.0417 | 不在区间内，不满足 |

读法（交用户判断，不替用户改判据）：

1. `inside=no` 只由第三条造成。官方三对的成功率差恰好都为负（历史 50、重跑一 49、重跑二 48，三次官方运行成功数依次少 1 局），区间只有 1 局宽（0.52 个百分点）且整体在 0 以下；正式跑法 51 局，比三份官方都多，点估计因此必然落在区间外。
2. 翻转数两条都满足，说明逐局层面的不一致程度与官方自己重跑相当；主比较置信区间 [−3.125, 5.208] 覆盖 0，McNemar p=0.80，与官方三对（p=1、0.80、1）同样不显著。
3. 追加样本不触发：主比较半宽 4.1667 个百分点，官方 3 对最大半宽 4.6875，远小于 2 倍阈值。
4. MME 的噪声源是 server 每次进程启动的 JAX 编译／自动调优（[policy-replay.md](policy-replay.md)），与接口无关；所以本判定回答的是「正式跑法这一次的成功数是否落在官方三次的成功数范围里」，不是「新接口是否改变了 MME 的行为」。

## 3. 官方噪声带（官方自己重跑一遍会有的差距）

- **SimpleMemVLA**：旧官方评估在 A40 上逐局可复现（3 对全 0 翻转，各 192 局）；A40 与本机 RTX 6000 之间才会翻（R1 对历史 1／2）。噪声来源在 GPU 型号，不在重跑。
- **MME**：官方重跑本身就翻，历史:重跑一 4／3、历史:重跑二 9／7、重跑一:重跑二 10／9（各 192 局）；成功数历史 50、重跑一 49、重跑二 48。确定性标志能消掉跨进程差异但慢约五成，按规则未开，与历史运行条件一致。
- **噪声带的条件边界**：重跑一全部在 gl1513（甲、丙同节点并发），重跑二改派后分布在 gl1513、gl1512、gl1528 三个节点（§5），历史成绩分布在 gl1512／gl1527／gl1525／gl1506／gl1518；三份的节点组成互不相同，节点效应混在噪声带里。加了录制器的只有两遍重跑（录制器对 SMVLA 无扰动已实证，对 MME 只有结构性证明）。

判定行原文见 [official-rerun.md](official-rerun.md) §2。

## 4. 测到了什么、没测到什么

**测到（有判定行）**：

1. 环境：同型号换卡、换机器、常驻倒序逐字节一致；换型号 44/48 身份有差（[env-parity.md](env-parity.md)）。
2. 旧／新环境栈：reset 层逐字节相同（A40 两格、RTX 一格）。
3. 录制器：SMVLA 无扰动（帧、终态、步数全同）；MME 代理 22 次透明核对全部 `mismatch=0`。
4. 策略开环：两策略四条件的同 server、重启、常驻顺序、MME 编译缓存三态；确定性标志开关的逐位性与减速；随机状态恢复。
5. 新旧接口：开环（同一输入，打包字节与执行动作全等）与同卡闭环（R1:E1，SMVLA 0 翻转、MME 2／3）。
6. 换条件闭环：5.1 五对 ＋ 对历史 ＋ 对重跑一，六格全齐（E5 MME、E6 SMVLA、E7 MME 为事故后整格重跑）。
7. 正式跑法：多席队列乱序、常驻、两策略杀 server 续跑（`KILLTEST` 两行 `errors=0 queue_check=PASS`）、8 席金丝雀两策略各 8/8；主比较与两个参照比较。
8. 测速：环境各段（`ENV_SPEED`）、每条件每席（`EVAL_SPEED`）。

**没测到或只测到一部分（如实列出）**：

1. 方案设想的「8 席同时消费队列」没有实测到：SimpleMemVLA 的 192 局实际全部由新1～新4 四席消费，MME 真实结果由丁、新3、新4 三席产生；乙、丁（SMVLA）、甲、丙加入时队列已空，只跑了金丝雀（[prod-vs-official.md](prod-vs-official.md) §3）。
2. MME 杀 server 测试在 3 个身份的专用队列 K 上做，不在正式队列上（正式队列已完成）；SimpleMemVLA 的杀 server 测试在正式队列上。
3. SimpleMemVLA 确定性开态：`cumsum` 无确定性实现，0 次比较。
4. 闭环层无法把「接口 / 环境代码 / sapien 版本 / 环境侧 torch 版本」四项差拆开，只在 reset 层证明环境栈贡献为 0（方案盲区第 2 条）。
5. MME 旧客户端代码以 `927c56d` 代表 E0 运行代码，仍属推断（方案盲区第 1 条）。
6. reset 次数硬上限（方案 §3.4「以实测 × 1.1 为硬上限写进留档」）没有单列判定行；各项只按轨迹尝试计（`BUDGET` 行），reset 次数按每局 1 次评估 reset 估算（2.1 每局 2 次，288 × 2 = 576）。
7. GPU 计算模式对照实验只有主会话终端记录，未落文件，属探索性证据（[incidents.md](incidents.md) 事故 6）。
8. 方案未覆盖项照旧未覆盖：xhard1～4、48 h 长时间运行、未用过的 A40 节点与驱动。

## 5. 与方案的偏离

| # | 偏离 | 依据 / 原因 | 对结论的影响 |
|---|---|---|---|
| 1 | 第 4 步 P2 并入 P1（卡 0 常驻倒序改由 ABA／BAB 模式覆盖），P6 并入 P5（乙同卡复跑改由 restart 模式覆盖），只跑四个回放条件 | 本轮原话 7「同意合并 尽可能并行」 | 无：合并后的模式覆盖了原条件要问的问题 |
| 2 | 官方重跑二改派：12.272 原定「O2 保持在甲、丙」，04:56 起片 1、2 改派丁，片 3、4、7、8 改派新1～新4；事故后片 1、2 在新1、新2 整片重跑，片 3、4、7、8 续跑；06:52 起片 0、6 改派到新2、新1（`final6`）；只有片 5、9 留在甲、丙 | 缩短关键路径；本轮原话 7、8；跳过标记 `skip/O2-s<n>.skip` | 重跑二节点组成与重跑一不同，节点效应进入官方噪声带（§3） |
| 3 | 5.2 条件标签拆成 `N`、`N2`、`N3`，另有金丝雀补跑 `C`、MME 杀 server 测试 `K` | GPU 计算模式事故污染 MME 101 个身份，补跑队列 `prod2`（100）、`prod3`（1）；缺真实金丝雀的席位补跑 | 无：同口径，按身份合并 0 冲突；K 不计入正式结果 |
| 4 | 整格重跑：E5 MME（事故 5 误删目录）、E6 SMVLA、E7 MME（事故 6）；O2 片 1、2 整片重跑，片 3、4、7、8 的 MME 续跑 | 基础设施故障可重试原身份（方案 §3.4） | 失败尝试目录 `*-vulkanfail-*` 留证、不进比较 |
| 5 | 5.2 实际同时消费队列的最多 4 席（见 §4 第 1 条） | 事故 6 中甲、丙的等待型步骤被取消，重起时队列已空 | 未测 8 席并发；逐局结果不受影响 |
| 6 | MME 杀 server 测试改在专用 3 身份队列上做；金丝雀 22 次（计划 16） | 正式队列已完成；同一席位每次起 server 都先跑金丝雀 | 续跑路径两策略都有实证；金丝雀超计划 6 次 |
| 7 | 第 0 步合并判定行 `PREFLIGHT=PASS seats=5 data_free_gib=… slack=…` 未生成；`/data` 开跑时剩余 4.2 TiB（01:10 EDT `df -h /data`；02:50 EDT 搬运器首行 `DATA_FREE 4244G`），开跑后 Slack 测试私信发送成功（频道 `D09233N44SF`，01:5x EDT）；两项由主会话在会话记录中核实补记 | 各项核对分散执行 | 无数值影响；全程无存储降级 |
| 8 | 预算超额，见 §6 | 事故 6；本轮原话 8「不要再来问我」使超额未事前请示 | **须用户事后追认** |

## 6. 预算（P3 一次性批准，P5 乘式，如实计超额）

```text
BUDGET=INFO item=2.1_环境检测 attempts=289 unique=288 incident=0 retries=1 cap=288 retry_cap=6 over=no retry_over=no est=含开发冒烟1行(估)
BUDGET=INFO item=2.2_录制器验证 attempts=1 unique=1 incident=0 retries=0 cap=1 retry_cap=共用2 over=no retry_over=yes est=none
BUDGET=INFO item=2.3_官方重跑两遍 attempts=768 unique=768 incident=93 retries=93 cap=768 retry_cap=12 over=no retry_over=yes est=按逐局文件行数;启动器内部整遍重评未落行的不计(估)
BUDGET=INFO item=2.4_本机旧官方小样本 attempts=96 unique=96 incident=0 retries=0 cap=96 retry_cap=4 over=no retry_over=no est=同上(估)
BUDGET=INFO item=3.1_跑通 attempts=2 unique=2 incident=0 retries=0 cap=2 retry_cap=共用2 over=no retry_over=yes est=中断未写结果的尝试不计(估)
BUDGET=INFO item=5.1_换条件评估 attempts=584 unique=576 incident=148 retries=156 cap=576 retry_cap=12 over=no retry_over=yes est=含事故留档目录真实局 8、infra 148
BUDGET=INFO item=5.2_正式跑法 attempts=386 unique=384 incident=106 retries=108 cap=384 retry_cap=8 over=no retry_over=yes est=含补跑队列 N2、N3
BUDGET=INFO item=5.2_金丝雀 attempts=22 unique=16 incident=0 retries=6 cap=16 retry_cap=共用2 over=yes retry_over=yes est=含 C 金丝雀补跑与 K 测试的金丝雀；重复席位计为重试
BUDGET=INFO item=其他_杀server测试 attempts=3 unique=3 incident=0 retries=0 cap=3 retry_cap=共用2 over=no retry_over=yes est=K：MME 杀 server 测试的队列局（不是正式结果）
BUDGET=INFO item=incident_infra_attempts attempts=0 unique=0 incident=347 retries=347 cap=n/a retry_cap=见各项 over=no retry_over=n/a est=0 步、环境未建成（GPU 计算模式事故等），不计轨迹尝试、已计入各项重试；其中事故留档目录内 148
BUDGET_TOTAL=INFO trajectory_attempts=2151 cap=2271 over=no incident_infra_attempts=347 all_attempts=2498 attempts_over_items=5.2_金丝雀 retry_over_items=2.2_录制器验证,2.3_官方重跑两遍,3.1_跑通,5.1_换条件评估,5.2_正式跑法,5.2_金丝雀,其他_杀server测试 shared_retries=6 shared_retry_cap=2
```

计划乘式：2.1 为 6 条件 × 16 任务 × 1 档 × 3 局 = 288；2.2 为 1 × 1 × 1 = 1；2.3 为 2 策略 × 2 遍 × 16 × 1 × 12 = 768；2.4 为 2 × 1 × 16 × 1 × 3 = 96；3.1 为 2 × 1 × 1 × 1 = 2；5.1 为 2 × 6 × 16 × 1 × 3 = 576；5.2 为 2 × 16 × 1 × 12 = 384；金丝雀 8 席 × 2 策略 × 1 = 16；小计 2131，加基础设施重试上限 44（2.1 6、2.3 12、2.4 4、5.1 12、5.2 8、其他 2）与同卡迁移预留 96，总上限 2271。

实际：

- **轨迹尝试 2151，在总上限 2271 内**；除金丝雀外各项尝试数均未超计划数（2.3 恰 768、5.1 去重 576、5.2 去重 384）。
- **金丝雀 22 次，超计划 16 次共 6 次**（`attempts_over_items=5.2_金丝雀`）；另有方案未列的 MME 杀 server 测试 K：1 策略 × 3 身份 × 1 局 = 3。
- **基础设施重试超出子额度**：2.3 为 93 次（子额度 12）、5.1 为 156 次（子额度 12）、5.2 为 108 次（子额度 8）；共用额度 2 被金丝雀重复的 6 次用超（`shared_retries=6 shared_retry_cap=2`），所以 2.2、3.1、杀 server 测试虽自身 0 重试也标 `retry_over=yes`。0 步失败共 347 次，几乎全部来自 GPU 计算模式事故（[incidents.md](incidents.md) 事故 6）。
- **把 0 步失败也算进去，全部尝试 2498 次，超出总上限 2271 共 227 次。** 方案 §3.4 写的是「某项额度用完，该格标 `INFRA_EXHAUSTED`，其余照跑」，实际没有停格而是继续补跑到身份齐全（0 步失败没有建成环境），这一点偏离方案，须用户事后追认。

## 7. 用户待决事项

1. 两策略的相对标准结论（SMVLA「官方重跑无翻转、正式跑法 0 翻转」；MME `inside=no`，只因成功率差点估计）是否接受，见 §2。
2. 预算超额（§6）的事后追认。
3. GPU 计算模式教训（带 `--gpu_cmode=shared` 的作业步结束即把整卡重置为独占）是否写入 `greatlakes.md` 与规则正本；本轮未改规则文件。
