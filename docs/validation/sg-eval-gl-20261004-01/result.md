# sg-eval-gl-20261004-01 四模型评估：result（收尾时点，部分完成）

计划：`1003-oracle-subgoal-groundsg-eval-plan.md`；起跑档案见同目录 `launch.md`。GL 执行副本冻结在 `a299fcdc`（12.453）；本机主检出运行时 `4b8b4636`（与冻结提交的运行路径零差异，见 `LG3_RUNTIME_DIFF`）。

**本轮在 2026-10-05 22:17 按用户指令中止**（原话：「现在立刻收尾。停止本机器和gl上所有任务 但是job不要关。」「Turbo上的资源先不动。只收尾不搬运。」）。本文件只记录到中止时点为止已经产出的结果；未跑完的部分逐项列在第七节。NFS（Turbo）上的结果与视频原地保留，没有搬回 `/data`，也没有删除；GL 占位 job 全部保留未取消。

逐表原文：`records/tables.md`（各档总表与按任务表）、`records/verdicts.md`（全部判定行原文）、`records/diff/`（逐局差异表与对照表）、`records/video-index.tsv`（视频目录索引）、`records/scripts/`（全部一次性脚本逐字归档）。

## 一、结论

1. **接线正确性（第二档）**：新接口与原版接口在 Oracle、PonderPounce 上等价。
   - PonderPounce：本机 192/192、GL 分片 0 96/96 逐步完全一致。
   - Oracle：GL 192 局终态 192/192 相同、逐步一致 180（其余 12 局是单独起服务的补跑局，服务端随机数不对齐，只比终态）；本机终态 189/192。
   - QwenVL：GL 分片 00 两侧都有终态的 37 局逐步完全一致；本机终态 176/192、逐步 141/192（分叉都在动作数值末位，未专门定性）。
2. **生成无回归（第一档）**：GL `GEN_REGRESS=PASS`（V9 129 局 match 127、jitter 2；xhard0 48/48）；本机旧码对新码两侧都生成成功的局全部逐字节相同。
3. **V9 第三档已有正式成绩**：Oracle 800 局 **52.9%**（423/800）；PonderPounce 分片 0（400 局）**16.8%**（67/400）。QwenVL 只跑了 9 局即中止，没有成绩。
4. **跨机器复刻（第三档 Oracle，不计正式成绩）**：本机 RTX 6000 Ada 复刻 800 局 51.75%，与 GL A40 的 52.9% 无显著差异（McNemar p=0.42，终态一致 84.4%）。

## 二、第一档（生成对拍）

| 站点 | 集合 | 判定 |
|---|---|---|
| GL（gl1525，A40） | V9 129 局 | `GEN_REGRESS=PASS`：match 127、jitter 2（MoveCube 23400200、BinFill 16400000）、flip/structural/missing 0 |
| GL | xhard0 48 局 | `GEN_REGRESS=PASS`：match 48 |
| 本机（旧码对新码） | V9 129 局 | byte_equal 128、gen_fail 1（BinFill xhard1 16400000，新码失败） |
| 本机 | xhard0 48 局 | byte_equal 47、gen_fail 1（VideoPlaceOrder xhard0 611101，两侧都失败） |

逐局定性：`records/diff/gate1-gl-v9-episodes.md`、`records/diff/gate1-gl-xhard0-episodes.md`。

## 三、第二档（xhard0，原版接口对新接口）

### 表 1：判定行摘要

| 站点 | 模型 | 比较局数 | 终态相同 | 逐步一致 | 判定 |
|---|---|---|---|---|---|
| 本机 | Oracle | 192 | 189 | 101 | INFO |
| 本机 | PonderPounce | 192 | 192 | 192 | INFO |
| 本机 | QwenVL | 192 | 176 | 141 | INFO |
| GL | Oracle | 192 | 192 | 180 | INFO |
| GL | PonderPounce 分片 0 | 96 | 96 | 96 | INFO |
| GL | QwenVL 分片 00 | 37（另 2 局新侧 error） | 37 | 37 | INCOMPLETE |

读法：GroundSG 服务端随机数跨局累积，只有每次起服务后的第一局逐步判别有效；服务重启（续跑、补跑）之后的局只比终态。

### 表 2：成功率

| 站点 | 模型 | 原版 | 新接口 |
|---|---|---|---|
| 本机 | Oracle | 140/192（72.9%） | 139/192（72.4%，含 12 局补跑） |
| 本机 | QwenVL | 47/192（24.5%） | 48/192（25.0%，含 5 局补跑） |
| 本机 | PonderPounce | 82/192（42.7%） | 82/192（42.7%） |
| GL | Oracle | 144/192（75.0%） | 144/192（75.0%，含 12 局补跑） |
| GL | PonderPounce 分片 0 | 39/96（40.6%） | 39/96（40.6%） |
| GL | QwenVL 分片 00 | 9/39 | 9/37 有终态（另 2 局 error） |

按任务表见 `records/tables.md`；逐局差异表见 `records/diff/gate2-*.md`。

### 补跑与合表

- SwingXtimes 叠字视频文件名超 255 字节（12.453 修复前）致新侧 error：本机 Oracle 12 局、本机 QwenVL 5 局、GL Oracle 12 局，都用修复后的代码补跑（GL 用 `a299fcdc`），补跑行按 key 替换原 error 行，合表文件在 `artifacts/sg-evaluation/sg-eval-gl-20261004-01/{local-g2,gl-g2}/gate2-*/new-merged.results.jsonl`，原结果文件未改。补跑结果：本机 Oracle 12/12 成功，本机 QwenVL 4 fail + 1 timeout，GL Oracle 12/12 成功。
- GL QwenVL 分片 00 新侧 InsertPeg_xhard0_631501、VideoPlaceOrder_xhard0_610701 两局各撞 30 分钟单局墙钟上限 2 次（截断在第 1120～1264 步，A40 上 QwenVL 每步约 1.5 s），infra 重试额度 3 耗尽记 error，未补跑。

## 四、第三档（V9 test-hard，1600 步 strict cap）

### 表 3：GL 正式结果

| 模型 | 计划局数 | 已有终态 | 成功 | 失败 | timeout | error | 成功率 |
|---|---|---|---|---|---|---|---|
| Oracle | 800 | 800 | 423 | 191 | 186 | 0 | **52.9%** |
| PonderPounce | 800 | 400（分片 0） | 67 | 260 | 72 | 1 | 16.8%（仅分片 0） |
| QwenVL | 800 | 9（分片 00，中断） | 0 | 3 | 6 | 0 | — |
| Astra | 1（连通） | 0 | | | | | 未开跑 |

PonderPounce 那 1 局 error：VideoPlaceOrder_xhard2_19100901，第 1541 步 `Ponder context overflow: S2 context length 16453 exceeds cap 16384`，3 次重试都在同一步失败，属模型自身上下文上限，不是基础设施故障。

### 表 4：按任务（成功／已跑）

| 任务 | GL Oracle | GL PonderPounce 分片 0 | 本机复刻 Oracle |
|---|---|---|---|
| BinFill | 19/50 | 1/25 | 20/50 |
| ButtonUnmask | 13/50 | 4/25 | 13/50 |
| ButtonUnmaskSwap | 21/50 | 3/25 | 20/50 |
| InsertPeg | 9/50 | 1/25 | 11/50 |
| MoveCube | 19/50 | 4/25 | 18/50 |
| PatternLock | 30/50 | 0/25 | 30/50 |
| PickHighlight | 0/50 | 0/25 | 0/50 |
| PickXtimes | 48/50 | 20/25 | 49/50 |
| RouteStick | 10/50 | 0/25 | 10/50 |
| StopCube | 10/50 | 0/25 | 13/50 |
| SwingXtimes | 44/50 | 5/25 | 38/50 |
| VideoPlaceButton | 42/50 | 5/25 | 40/50 |
| VideoPlaceOrder | 44/50 | 7/25 | 44/50 |
| VideoRepick | 50/50 | 10/25 | 49/50 |
| VideoUnmask | 17/50 | 7/25 | 16/50 |
| VideoUnmaskSwap | 47/50 | 0/25 | 43/50 |

中断分片（本机 PonderPounce 184 局、本机 QwenVL 137 局、GL QwenVL 9 局）的按任务表见 `records/tables.md`。

## 五、本机结果（不参与成绩标注）

- 预检：`PREFLIGHT_SUMMARY=PASS routes=13`；显存峰值 Oracle 35.3 GB、QwenVL 44.2 GB（预分配 0.75）、PonderPounce 27.4 GB；`STEP_CAP=PASS` ×3（xhard0 1300、V9 1600）；`VIDEO_SAVED=PASS` ×9。
- 第一档、第二档见上两节的「本机」行。
- Astra 本机预检两侧各 1 局（VideoUnmask），都成功，6 次规划调用共 **0.5149 美元**（每次未命中缓存约 0.10 美元）。

### 本机复刻 GL 第三档（2026-10-05 用户批准；不计正式成绩）

分片文件与 GL 逐字节相同（`LG3_SHARD_SHA=PASS n=8`），参数照抄 GL 席位链，2 张 RTX 6000 Ada。

| 模型 | 本机已跑 | 对齐局数 | GL 成功率 | 本机成功率 | 终态一致 | 只 GL 成功／只本机成功 | McNemar p |
|---|---|---|---|---|---|---|---|
| Oracle | 800/800 | 800 | 52.88% | 51.75% | 84.4% | 54／45 | 0.42 |
| PonderPounce 分片 00 | 184/400（中断） | 184 | 10.9% | 13.6% | 82.6% | 3／8 | 0.23 |
| QwenVL 分片 00 | 137/160（中断） | 9 | 0% | 0% | 77.8% | 0／0 | 1 |

逐任务对照：`records/diff/gate3-compare-*.md`。

## 六、运行中的偏离与事故（均已处置或记录）

1. SwingXtimes 叠字视频文件名超 255 字节致 Broken pipe：12.453 修复 episode_id 为短编号；受影响的第二档局全部补跑（见第三节）。GL 第三档本就跑在修复后的 `a299fcdc` 上，SwingXtimes 50 局无 error，无需补跑。
2. GL 第三档 Oracle 分片局号整体错位 12：首局即被身份检查拦下（`IDENTITY_MISMATCH`），重出分片后重起；旧分片保留为 `*.bad-epoffset`；12.455 在清单工具里加了逐行核对。
3. GL QwenVL 客户端环境与 glibc 2.28 不兼容：重建 GL client-env，flash-attn 以 sysroot 2.17 编译；修好后席位 03 正常运行。
4. 本机第二档 Oracle 新侧 reset 额度按每局 1 次给错（应为 2N+20）：v2 续跑修正。
5. 预检时在运行中修改 `run_eval_gl.sh` 致一条路线 syntax error：在重跑额度内重跑；此后不改运行中的脚本。
6. 主会话监听两次漏报致 GL 卡空转数小时：加每分钟兜底巡检。
7. GL 上 QwenVL 单局墙钟上限 30 分钟不够（见第三节）：已备好 `seat_chain_v2.sh`／`launch_seat_v2.sh`（上限 3600 s，步数上限不变），因取消步骤 `63188711.3` 的操作被安全分类器拦下、等待用户授权期间本轮中止，v2 只在席位 04 起跑了约 7 分钟。
8. 第二档 `validate_trace` 行序约定与 GroundSG 实际轨迹不符（12.454 放宽校验方）；`gate2_compare` 遇失效节点本地路径退回身份索引（12.452）。

## 七、未完成清单（中止时点）

| 项目 | 状态 |
|---|---|
| GL 第二档 PonderPounce 分片 1（96 局） | 未开跑 |
| GL 第二档 QwenVL 分片 01～04（约 153 局） | 分片 01 原版侧起跑约 7 分钟后中止，其余未开跑 |
| GL 第二档 QwenVL 分片 00 新侧 2 局 error | 未补跑 |
| GL 第二档 Astra（32 局，已批） | 未开跑（两卡 job 一直排队） |
| GL 第三档 PonderPounce 分片 1（400 局） | 未开跑 |
| GL 第三档 QwenVL（800 局） | 分片 00 跑到 9/160 中止，其余未开跑 |
| GL 第三档 Astra 连通 1 局（已批） | 未开跑 |
| 本机复刻 PonderPounce（800 局） | 分片 00 跑到 184/400 中止 |
| 本机复刻 QwenVL（800 局） | 分片 00 跑到 137/160 中止 |

续跑方式：同一 `--stage` 重跑即续跑（按 results 与账本只补无终态的身份）；QwenVL 席位改用 `launch_seat_v2.sh`。Astra 已用 2 局、0.5149 美元。

## 八、用户决策项

1. **9 个 GL 占位 job 是否释放**：63188711、63188712 为 RUNNING 空占，63188713～16、63188719、63188720、63188721 为 PENDING；按用户指令未取消。释放用清单 `scancel`，不要 `scancel -u`。
2. **Turbo（NFS）上的产物**：用户令「先不动」。约 11.9 GB（`media` 5.6 GB、`gate2-qwenvl` 4.3 GB、`gate2-oracle` 1.1 GB 等）在 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sgeval-20261004/`，是否搬回 `/data`、何时清理待定。若要续跑，必须保留（续跑读这些 stage 目录）。
3. **PonderPounce 上下文超限那 1 局如何计分**：建议按失败计（策略到第 1541 步仍未成功）；另一选项是单列「模型容量超限」、不进分母。
4. **GL QwenVL 第二档那 2 局**：若不补跑，GL QwenVL 分片 00 的 GATE2 保持 INCOMPLETE；补跑需 v2（单局上限 1 小时）。
5. **本机第二档 QwenVL 起服务后第一局也逐步分叉**（0/2），分叉在动作数值末位、子目标文本相同；GL 同模型 37/37 逐步一致。若需定性，需在同机同卡对同一局重跑两次比较。
6. **生成疑点 BinFill xhard1 16400000**：新码在本机与 A40 上都生成失败、本机旧码成功；闸门判抖动非回归。若要定性需新旧码同机对照。
7. **残留进程 PID 2961357**（`serve_policy.py --port=23310`，10-03 测试遗留，不占显卡）：未动，是否清理待定。
8. **中间产物**：本机第一档 h5（`local-g1`，实测 171 GB；整个本机产物目录 182 GB）等按「收尾只保留最终产物」应删除；本轮按「只收尾不搬运」未删，待用户确认。

## 九、本机 Oracle 存量官方版式重绘与视频站点（第一阶段完成，2026-10-05）

**已完成本机 `16 任务 × 每任务合计 50 局 = 800 局` 的官方版式重绘与网站托管。** 各难度实际乘式见 `launch.md` 第⑪节，不把第三阶段误称为单一难度档。网站：[http://sled-vail.eecs.umich.edu:8083/](http://sled-vail.eecs.umich.edu:8083/)。转码代码锚点 `6a23af541b4f136e754eeb06e8fd1548700c4b23`，干净 detached 运行副本；网站分Task与命名版锚点 `22a90ce9c33d01b729013d6f7d21457b87d5ceef`，后续总体追加见本节末。本轮没有新增 reset、轨迹生成、GPU 或集群任务，第二阶段模型链路未实施。

### 用户指令与界面口径

1. 「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/1005-eval-video-official-overlay-plan.md实现第一阶段的转码转完之后host在一个网站上。host完网站做Playeright测试。」
2. 「你还需要在最上方给我一个成功率的表格，不仅要有现在的V9版本，还需要有上一版X号的0的成功画面。」随后纠正为「你还需要在最上方给我一个成功率的表格，不仅要有现在的V9版本，还需要有上一版X号的0的成功率」。最新含义只增加成功率对照，不增加旧版视频生成。
3. 「我只要分task的成功率。」
4. 「Task内部就不用再拆了。然后成功率要有百分比和分数。」
5. 「然后你的网页按照任务来分视频的栏目，不要堆砌在一个矩目录的列表里，并且所有的任务名字都用中文，啊，都用英文。」最后一句明确采用英文 Task 名。

6. 「V9／xhard0 这里改名为X哈尔德和原版哈尔德。」随后纠正为「V9／xhard0 这里改名为Xhard和原版hard」。仅改可见名称，真实来源与统计不变。

顶部表格保留 16 个 Task 行、三列 `Task / Xhard / 原版hard`，按用户最新追加另有表尾总体行；每格同时显示一位小数百分比与成功局数／总局数，没有任务内分档。下方 16 个英文 Task 各有独立栏目，当前栏目只分页展示本 Task 的 50 局；支持官方版式／原始画面切换、筛选、上下局、进度拖动。成功率表不随当前视频筛选变化。

### 实测重绘结果与范围

`1 任务（VideoPlaceButton）× 1 档（xhard2）× 1 局` 冒烟：`OFFICIAL_RENDER=PASS dir=VideoPlaceButton_xhard2_19002401.a1 frames=1245 demo=1082 steps=162 omitted=0 size=512x528 status=rendered terminal_reason=fail end_status=fail`，退出 0。Frame 0、1082、1083、1244 抽帧目视确认演示红框、初始 Action:None、执行 Action/State/Subgoal 与黄色坐标点；这局的任务失败仍保持 fail，不重跑挑成功。

全量 2026-10-05 23:18:26～23:23:51 EDT，约 325 秒；tmux `ovl-rerender-g3`，16 个 CPU worker、每个 2 核、每局内存上限 6144 MiB。`OFFICIAL_RENDER_SUMMARY=PASS total=800 ok=800 fail=0 reused=1`；`EXIT_CODE=0`。1 个复用项即冒烟局，复用前后输出哈希不变。

共 848318 帧：演示 246698、初始 800、执行 600820；全部 30 fps。尺寸分别 `512×512` 311 局、`512×528` 265 局、`512×560` 224 局，来自官方 Task Goal 换行高度及 imageio 默认块对齐。原视频合计 1289673040 字节，官方重绘输出 2161212582 字节；原 `episode.mp4`、trace、NPZ 均保持来源指纹不变。17 个超长名按 UTF-8 边界截短加摘要，完整官方名保留在每局 `official/render.json`。

所有原 float64 动作逐步从 NPZ 恢复且与 trace 哈希相等；若先降为 float32，600820 步中的 883 个标量会改变四位小数显示，当前均避免。194 个严格步数上限超时局 `status=timeout`、`terminal_reason=error`，全部保留 1600 执行帧，省略帧总数 0；网站用前者分组，官方文件名用后者。

### 逐 Task 成功率来源

两列均取 **sled-vail 本机 GroundSG+Oracle 新接口** 的接受终态结果：V9 800 局（每 Task 50）；xhard0 `16 任务 × 1 档 xhard0 × 12 局 = 192 局`（已含接受的 SwingXtimes 补跑）。不混用 GL 正式 423/800 或原版接口 140/192。仅作各任务实测成功比例展示，不据此声明新旧版本数值等价或因果差异。

- V9：`local-g3/stage-oracle-00/s90/mmesg-ground-sg-oracle/results.jsonl`，sha256 `95ef1ad8aea6551d2bbd5bb09a8d31cfef5f97deef1cc56fad35ddcaade01920`，3410265 字节。
- xhard0：`local-g2/gate2-oracle/new-merged.results.jsonl`，sha256 `a78dc420afc928a6cc393e3256ec837290e4511ca5f618477f4e282c88e09502`，788850 字节。

全部身份唯一、每任务分母齐全、infra=false，成功字段与 `status=success` 一致。例如 BinFill 为 V9 `40.0%（20/50）`、xhard0 `66.7%（8/12）`；PickHighlight 两侧都为 0，未筛掉失败局。来源指纹在站点私有记录，公开目录不暴露本机路径。

### 验收与证据边界

独立只读核验 4 个 CPU worker、35.333 秒，退出 0；逐一核 2400 个输入指纹、800 个输出指纹、全部 600820 原动作及 800 份视频元信息。`OFFICIAL_IDENTITY_FIELDS=PASS episodes=800 fields=7 mismatches=0`、`OFFICIAL_FRAME_TOTAL=PASS source_frames=848318 frames=848318 exec_steps=600820`、`OFFICIAL_ACTION_HASH=PASS actions=600820 mismatches=0`、`OFFICIAL_VIDEO_METADATA=PASS videos=800 fps=30 full_decode=0`、`OFFICIAL_INDEPENDENT_VERIFY=PASS`。重绘器已经逐局完整解码和数帧，独立核验没有再次完整解码。

最新浏览器命令见 `launch.md` 的追加参数：`official_overlay_browser_check.py --base http://sled-vail.eecs.umich.edu:8083 --shots <本轮>/checks-labels --v9-results <V9结果> --xhard0-results <xhard0结果>`。`OFFICIAL_BROWSER=PASS episodes=800 checks=13 page_errors=0 http_errors=0`，退出 0；浏览器独立重算全部 32 格比率，逐 16 栏目核无混 Task，并断言最终表头 `Task / Xhard / 原版hard`；实际播放演示／非演示代表局及原视频，验证暂停、seek、筛选、分页、版本切换与 Range 206。宽屏、390 和 768 像素截图均无横向溢出，并经目视复核。真实浏览器播放为代表局检查，不声称全部 800 局在浏览器逐帧播放。

代码验证：官方逐帧对照及真实 FFmpeg 端到端 `16 passed in 1.58s`；核心短测 `2882 passed, 4 skipped, 630 deselected in 159.82s`；网站更新定向短测 `66 passed, 2 deselected in 5.18s`。三次资源守卫均 `TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0`；核心短测另有 `not_verified=4`，不将四个跳过项称为实测。`UPSTREAM_GUARD=PASS`，官方受保护包、三条官方入口和第二阶段评估代码无改动。

重绘保证对同一组已解码画面及核验原数组调用官方原类；画面来自既有 H.264 视频，不能宣称与官方原始相机像素逐位一致。

### 保留与复现

站点保留 tmux `ovl-site-g3-8083`，当前目录 `official-overlay/site-tasks/`，日志 `official-overlay/logs/site-tasks.log`；既有其他会话未处理。完整重绘日志、首次和最终浏览器报告、四张最终截图及来源指纹均在本轮产物根；Git 中归档为 `records/rerender-summary.jsonl`、`records/official-overlay/{render-summary,independent-verify,browser-initial,browser,success-sources}.json` 和四张 PNG。重绘 mp4 不进 Git。

复核命令：从转码锚点按 `launch.md` 的重绘 CLI 再跑，会检查来源、完整输出和原动作后复用；按最终网站源码及两份已钉住指纹的结果文件构建新目录并运行 Playwright，可以还原当前表格与交互验收。后续第二阶段改造、重跑预算和 Turbo 重绘仍按未定清单另行处理。

### 总体成功率追加（2026-10-05）

用户原话：「你这里还需要一个总合的成功率啊。」表尾新增唯一总体行，直接累加各 Task 成功数和总数后计算：Xhard `51.8%（414/800）`，原版hard `72.4%（139/192）`，不平均已四舍五入的百分比。旧分Task与命名版锚点 `22a90ce9` 保留；本次总体版仅增 HTML 的 `drawRates` 汇总及浏览器独立断言，没有改动媒体或源结果。

复现命令仍为 `official_overlay_browser_check.py` 和相同两份权威输入，只把 `--shots` 改为 `artifacts/sg-evaluation/sg-eval-gl-20261004-01/official-overlay/checks-overall`。`OFFICIAL_BROWSER=PASS episodes=800 checks=13 page_errors=0 http_errors=0`，退出 0；34 格全部核验，总体在筛选及窄屏后不变，390/768 像素无横向溢出。`OVERALL_CODE_REVIEW=PASS`，宽屏截图经目视确认。报告及截图为 `records/official-overlay/browser-overall.json`、`overall-wide.png`，完整四张截图仍在产物根。
