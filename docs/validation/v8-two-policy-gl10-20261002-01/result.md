# V8 双模型评估运行档案：result

计划 `1001-v8-post-evaluation-gl-plan.md`；起跑档案见同目录 `launch.md`。评估代码 `6039a7af`（12.312，E-A／E-B／E-C 全部合入；GL 执行副本运行时为 `35b1e0b0`，E-C 只含汇总与搬运脚本、在本机运行，评估链路两版零 diff）。

## 一、结论

2026-10-02 04:26 起跑、11:23 全部结束：两模型各 1070 个 V8 新值档身份（xhard1～xhard5，43 格，不含 xhard0）全部取得唯一权威终态，无 error、无 infra 重试、无额度提升；执行步严格截断 1600，越限 0。

| 模型 | 分母 | 成功 | 失败 | timeout（第 1600 步未成功） | error | 全局微平均 | 任务宏平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleMemVLA | 1070 | 278 | 643 | 149 | 0 | **26.0%** | 28.1% |
| MME-VLA（perceptual-framesamp-modul/79999） | 1070 | 64 | 938 | 68 | 0 | **6.0%** | 8.0% |

按档（跨任务；各档任务构成不同，档间不可直接比难度）：

| 档 | 局数 | SimpleMemVLA | MME-VLA |
|---|---:|---:|---:|
| xhard1 | 411 | 131（31.9%） | 30（7.3%） |
| xhard2 | 411 | 73（17.8%） | 21（5.1%） |
| xhard3 | 128 | 34（26.6%） | 0（0.0%） |
| xhard4 | 100 | 40（40.0%） | 13（13.0%） |
| xhard5 | 20 | 0（0.0%） | 0（0.0%） |

按任务（跨档）：

| 任务 | 局数 | SimpleMemVLA | MME-VLA |
|---|---:|---:|---:|
| VideoUnmask | 80 | 54（67.5%） | 0（0.0%） |
| ButtonUnmask | 80 | 49（61.3%） | 1（1.2%） |
| MoveCube | 20 | 16（80.0%） | 13（65.0%） |
| InsertPeg | 20 | 9（45.0%） | 0（0.0%） |
| VideoRepick | 80 | 29（36.2%） | 5（6.2%） |
| VideoPlaceButton | 80 | 25（31.2%） | 22（27.5%） |
| PatternLock | 80 | 21（26.2%） | 0（0.0%） |
| RouteStick | 80 | 21（26.2%） | 0（0.0%） |
| SwingXtimes | 50 | 12（24.0%） | 0（0.0%） |
| ButtonUnmaskSwap | 80 | 16（20.0%） | 3（3.8%） |
| VideoPlaceOrder | 80 | 15（18.8%） | 15（18.8%） |
| BinFill | 80 | 7（8.8%） | 0（0.0%） |
| VideoUnmaskSwap | 80 | 4（5.0%） | 5（6.2%） |
| PickHighlight | 80 | 0（0.0%） | 0（0.0%） |
| PickXtimes | 50 | 0（0.0%） | 0（0.0%） |
| StopCube | 50 | 0（0.0%） | 0（0.0%） |

43 格逐格成功率、timeout 数与逐席预算见 `artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01/report/report.md`（同目录 `report.json`、`video-index.jsonl`）。

与历史对照（仅供参考，协议不同不逐局对齐）：V7 两策略在 xhard1～4（上限 1600～3800、不截断）SimpleMemVLA 32.7%／18.1%／12.7%／14.1%、MME-VLA 4.2%／6.9%／4.6%／7.2%（`docs/validation/newtask-v7/README.md`）；V8 xhard0 两路线 SimpleMemVLA 141/192、MME-VLA 48～50/192（`docs/validation/newtask-v8/result.md`）。MME-VLA 在新值档上明显低于 SimpleMemVLA，与 V7 同一量级。

## 二、验收判定行

| 判定 | 结果 |
|---|---|
| V8_PREREQUISITE | V8 阶段 1～4 全部留档（`docs/validation/newtask-v8/result.md` ①～⑫，12.303 收尾）——本轮以该档案为前提，未单独实现判定脚本 |
| V8_EVAL_INPUTS | 权重按 v7.5eval 资产锁逐文件核对（MME 18/18、SimpleMemVLA 锁定 9/9，另 24 个 HF 下载簿记文件）；tokenizer GCS md5 一致后锁 sha256 `8986bb4f…8fc6`，每次 MME server 启动前 `TOKENIZER_SHA=PASS`；`identity_count=1070`（未作为单一脚本判定行输出，证据见 launch.md §三） |
| V8_EVAL_SHARDS | `V8_EVAL_SHARDS=PASS shards=10 missing=0 extra=0 duplicate=0 total=1070 cells=43 xhard0=0`；运行中 07～09 重分为 10～16（§三），`RESPLIT_OK disjoint union=321` |
| V8_STEP_CAP_CONTRACT | `V8_STEP_CAP_CONTRACT=PASS cap=1600 over=0 smvla_blocks_v8=102 smvla_blocks_legacy=84`（E-A 合入测试） |
| V8_EVAL_SMOKE | `V8_EVAL_SMOKE=PASS infra_errors=0 identity_errors=0 demo_frames_min=66 exec_steps_max<=1600 smvla_hard_bound=102`（smoke2，见 launch.md §六） |
| V8_EVAL_COVERAGE | `V8_EVAL_COVERAGE=PASS policies=2 missing=0 extra=0 duplicate=0 conflicting_terminal=0 late_ignored=0 error_final=0` |
| V8_EVAL_REPORT | `V8_EVAL_REPORT=PASS count_mismatch=0 media_unexplained=0 exec_over_cap=0` |
| V8_EVAL_VIDEOS | `V8_EVAL_VIDEOS=PASS policies=2 expected=2140 videos=2140 missing=0 decode_fail=0 sha_mismatch=0 error_attempt_videos=0 stage_left=0 orphan_videos=0 error_final_no_video=0` |
| V8_EVAL_CLEANUP | `V8_EVAL_CLEANUP=DONE outcome=pass jobs_released=10 videos_kept=2146`（7 个跑满席位收尾后释放、3 个排队席位取消；正式 2140＋冒烟 6） |

## 三、运行过程

- 席位：v8ev-hold-00～09（63071528～37）03:19 提交；00～06 约 03:20 RUNNING，07～09 因 spgpu 253 个作业排队始终未起。
- 04:26 00～06 起跑分片 00～06（每片 107 身份，先 SimpleMemVLA 后 MME-VLA）；SimpleMemVLA 推理每动作块 1.7～2.0 s、GPU 100%，跑满 1600 步的局约 200～280 s；MME-VLA 平均约 40 s／局。
- 08:00 把未起的分片 07～09（321 身份）按 TASK_SECONDS 贪心重分为 7 个子分片 10～16（45～46 身份，`inputs/resplit-07-09.json`），派发器在 00～06 各自跑完后接力（08:32～09:22），排队的 63071535～37 于 09:23 取消。infra 重试额度调整为 s10～s12 各 1、s13～s16 各 0，每模型合计仍为 10。
- 各子分片收尾 `V8_SEAT_DONE outcome=pass`、`SEAT_REC_SYNC=PASS left=0` 后自动释放其作业（10:28～11:23）。
- 全部 14 个分片（00～06、10～16）× 2 模型：`V8_SEAT_DONE outcome=pass`，无 RUN_BLOCKED／RESET_BUDGET_EXHAUSTED／SERVER_DIED／INFRA_TIMEOUT。
- 预算：两模型各 reset_claim 2140（每局恰 2 次：build＋reset），尝试 1070，infra 重试 0，额度提升 0；冒烟另 6 次尝试（smoke 2 次零执行步＋smoke2 4 次）。

## 四、视频

- 2140 个正式身份每局都有录像（成功、失败、timeout 一视同仁），本机 `artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01/videos/<策略>/<tier>/<task>/<key>.a1/`（`front.mkv`、`wrist.mkv` FFV1 无损＋逐帧 sha256、逐步数组、事件），共 254 GB；`moved.jsonl` 2140 行（逐文件 sha256），`decode-cache.jsonl` 为 4280 个 mkv 的 ffprobe 帧数。
- 搬运：节点 /tmp → 每 120 s 原子发布到 NFS 运行根（`SEAT_REC_SYNC=PASS`，14 片均 `left=0`）→ 本机常驻 `eval_video_mover.py --mode v8`（09:52 起，约 65 MB/s，逐目录 rsync 到本机临时目录、逐文件 sha256 一致后改名落地、再删 NFS 源）→ 评估结束后 `--once` 全量对账（上行判定）。
- 冒烟录像单独存 `videos-smoke/{smoke,smoke2}/`（2＋4 局，208 MB，各 `V8_EVAL_VIDEOS=PASS`），不计入正式分母。

## 五、收尾与清理

- 占位作业：63071535～37 于 09:23 取消（排队未起）；63071529／30／28／33／32／34／31 于各自子分片收尾通过后释放（10:28～11:23）。清单 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/gl-hold-logs/hold-jobs-v8ev-20261002.txt` 逐条记录；`squeue -u $USER` 为空，GL 登录节点 `ev-v8-*` tmux 已全部结束。
- NFS：运行根 `v8two-out/v8-two-policy-gl10-20261002-01`（结果行、账本、日志、输入、硬链接权重目录、tokenizer 副本）在非录像记录拷回本机 `nfs-records/`（16 MB，`rsync -c` 复核差异 0）后删除；GL 脚本存档到本机 `gl-scripts/` 后删除 NFS 副本。保留 NFS 执行副本 `robomme_benchmark-v8two`（含三套 venv，可复跑；是否删除待用户定）。
- 本机：子代理 worktree `agent-a764153e…`（E-A）、`agent-a78979bd…`（E-B）、`agent-a43b876b…`（E-C）已 `git worktree remove` 并 `git branch -d`；本机保留 `artifacts/v8-evaluation/`（输入、清单、正式汇总 `report/`、视频、NFS 记录、日志）。
- 正式汇总全文另存 `records/report.md`（43 格逐格、逐席预算）。

## 六、遗留与说明

1. 评估中「非 infra 错误」为最终结局（主会话裁定，用户同意 E-C 第三轮时一并确认「无录像时写明原因即可」）；本轮实际 0 例。
2. 冒烟 1 的两局错误被记为非 infra（`ModuleNotFoundError: openpi_client`），分类口径上环境缺包应属基础设施，本轮未改分类逻辑，仅留档。
3. PickXtimes、StopCube、PickHighlight 三任务两模型全部 0 成功；PickXtimes／StopCube 无 timeout（环境提前判失败），PickHighlight 多为 1600 步 timeout。未做额外诊断。
4. 录像体量约 125 MB／局（FFV1 无损双相机＋逐步数组），计划 §6 的 5 GB 估计偏小两个数量级（launch.md §七 4）。

## 七、站点（8081）

用户 2026-10-02：「把所有的结果放在8081端口」「注意你给我链接要是 http://sled-vail.eecs.umich.edu:8081/ 你现在给的是错误的」。

- 站点：**http://sled-vail.eecs.umich.edu:8081/**（tmux `site-v8-8081`，`v8_site.py --site-dir artifacts/newtask-v8/site-eval`，日志 `artifacts/newtask-v8/logs/site-v8-8081-eval.log`，`V8_SITE_READY videos=3974`）；原无评估的站点目录 `artifacts/newtask-v8/site/` 保留不动，beta 站 8080 不动。
- 评估视频转码：`scripts/injection-dev/site/v8_eval_transcode.py`——录像器 FFV1 无损 mkv 浏览器不能播，且同流重复帧只编一份，须按 `frames-*.jsonl` 的 `idx→enc` 展开回逐帧原图，再左右拼成 512×256（左上角 DEMO／EXEC）以 libx264 crf23 输出 mp4；`V8_EVAL_TRANSCODE=PASS episodes=2140 frame_mismatch=0 stream_len_mismatch=0 failed=0`，产物 `artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01/site-media/`（2.8 GB，manifest 逐局帧数与 sha256）；无损原片不动。
- 目录：`v8_site_catalog.py --eval-run artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01 --xhard0-eval artifacts/newtask-v8/xhard0-eval --out artifacts/newtask-v8/site-eval`（另拷原站 `subgoals.json`）→ `V8_SITE_CATALOG=PASS identities=1262 expected=1262 gen_v8=1070 gen_xhard0_new=190 gen_xhard0_old=190 gen_failed=4 eval_filled=2908 eval_media=2524 eval_unevaluated=0 flip=18 rate_mismatch=0 config_mismatch=0 media=3974 problems=0`。xhard1～5 每身份取账本 accept 权威结果、逐格成败数与 `report.json` 核对一致；xhard0 取阶段 3′ 两路线（新入口 = hard 路线、旧入口 = 官方路线），MME-VLA 两入口 384 个视频，SimpleMemVLA 当时未录视频、评估位注明「只有结局与步数」；两路线翻转 18 处均为 MME-VLA。
- 浏览器检查（Playwright，`--base http://sled-vail.eecs.umich.edu:8081`，截图 `artifacts/newtask-v8/site-eval-checks/`）：`V8_SITE=PASS sections=15 eval_placeholders=0 eval_missing=0 eval_mismatch=0 eval_played=118 subgoal_missing=0 config_mismatch=0 cells=59 played=59 page_errors=0`；`V8_ORACLE_BROWSER=PASS cells=59 missing=0 mismatch=0 page_errors=0`。截图已目视：局号点颜色、筛选计数、评估徽标与步数、评估视频（EXEC／DEMO 标签）均正常。
- 规则：「给用户的网页链接一律写完整域名」已写入 AgentMetaRules 正本第 23 条（`361e987`）并回流 benchmark／policy／mjepa 三仓库；`docs/validation/newtask-v8/result.md` 中的短主机名链接一并改为完整域名。

### 7.1 goal／subgoal 语义调整标注（用户 2026-10-02「Go和SubGo有哪些语义上的调整」）

- 新增 `scripts/injection-dev/site/v8_semantic_diff.py`：读站点 `subgoals.json` 的逐局 task goal 措辞与 subgoal 同类模板，把序数、颜色、数量、次数换成占位符后逐格与 xhard0 比句式；写 `artifacts/newtask-v8/site-eval/semantic.json`，站点新增 `/api/semantic`。`V8_SEMANTIC=PASS tasks=16 cells=43 changed=17 param_only=26 note_missing=0`。
- 有语义调整的 17 格：PickHighlight xhard1／2（goal 重写为逐个抓取并补结尾按钮，subgoal 去掉颜色提示、结尾加按钮）；VideoUnmask、ButtonUnmask xhard2～4 与 VideoUnmaskSwap、ButtonUnmaskSwap xhard2（goal 由 2 个目标变 3 个，多出「next pick up another container…」分句）；ButtonUnmaskSwap xhard1／2（新增 subgoal「wait for the containers to finish swapping」）；VideoRepick xhard1／2（goal 不再用「again」、写明次数，新增等待段 static）；VideoPlaceButton xhard1／2（goal 只剩「where it was 〈序数〉 placed before／after the button was pressed」句式）；VideoPlaceButton、VideoPlaceOrder xhard1／2（subgoal「drop the cube onto table」改为「put the cube back to its original position」）。其余 26 格仅参数变化。
- 页面：任务页当前档显示「与 xhard0 相比的 goal／subgoal 语义」面板（中文说明、新增／不再出现的句式与类型、仅重复次数变化）；逐局 task goal 与 subgoal 表中新句式／新类型带「语义调整」标签；各档总表新增一行「goal／subgoal 语义（相对 xhard0）」。
- 检查：`V8_SITE=PASS … semantic_mismatch=0 page_errors=0`、`V8_ORACLE_BROWSER=PASS cells=59 missing=0 mismatch=0`；截图已目视。

### 7.2 语义调整前后对照与合集（用户 2026-10-02「网站上有语义调整的需要把语义调整前后的都写上。然后也要做一个语义调整的合集」「不要写在这里了」）

- `v8_semantic_diff.py` 对有调整的格另给调整前后数据：`goal_before`／`goal_after`（xhard0 与该档全部 goal 句式各附原文例句，标保留／去掉／新增）、`sub_pairs`（不再出现与新增的 subgoal 类型按文字相似度配对）、`example`（xhard0 第 1 局与该档一局的 task goal 与整条 subgoal 序列，新句式／新类型标出）。
- 站点侧栏新增「语义调整合集」（`#view=semantic`）：17 格逐格三张「调整前 · xhard0 ｜ 调整后 · 本档」对照表（goal 句式、subgoal 类型、示例局全序列），卡片标题链接到该格任务页。任务页上的语义面板按用户要求撤下（容器隐藏）；逐局 goal／subgoal 的「语义调整」小标签保留。
- 检查：`V8_SITE=PASS … semantic_mismatch=0 page_errors=0`（合集卡片恰为 17 格且每张都有对照表、任务页面板隐藏）；合集页截图已目视。
