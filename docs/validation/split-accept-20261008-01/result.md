# split-accept-20261008-01：拆仓实施与验收（result）

## ① 一句话结论与判定速览

现仓库已按 1008 拆分方案第四稿一分为二：benchmark 仓 `hongzefu/RoboMME-benchmark-OOD`（公开）在 `v1.0-ood`＝`a5efb99` 锁死并 `gh repo archive`；评估仓 `hongzefu/RoboMME-benchmark-OOD-eval`（私有）以 submodule 钉住该提交，外层 3 函数 + 模型侧 4 方法、AV1 产物、GL 动态队列与对拍工具全部到位；§五三条实跑验收全部 PASS。

| 阶段 | 判定行 |
|---|---|
| S1（benchmark `a5efb99`） | `BENCH_GATES=PASS gates=11`：`BENCH_DELTA=PASS added_roots=2 added_files=6 modified=3 deleted=0 unexpected=0`、`BENCH_UPSTREAM=PASS files=0 clean=1`、`BENCH_UNCHANGED=PASS files=64 diffs=0`、`BENCH_SPECS_SHA=PASS files=5`、`BENCH_MANIFEST=PASS self_sig=1 dangling=0`、`BENCH_DATASETS=PASS rejected=4 hard_verify=192 ood=800 default=ood`、`BENCH_ENTRY_DIFF=PASS changes=4 unexpected=0`、`BENCH_NO_GEN=PASS ast_hits=0 dirs_absent=3`、`BENCH_SCRIPTS=PASS files=5`、`BENCH_PACKAGE=PASS passed=4`、`BENCH_REGISTRY=PASS passed=4` |
| S1（候选 `d22fe20`） | 上列 11 行 + `BENCH_SMOKE=PASS identities=2 resets=2`（`a5efb99` 只多 3 个规则文件提交，未重跑 smoke） |
| S1 测试 | benchmark 日常门禁 2073 passed（`TEST_RESOURCE=PASS`）；slow 553 passed；sim 冒烟 `tests/robomme_hard/sim --allow-sim-reset` 60 passed（用户追加授权）；`TEST_CONTRACTS entries=119 missing=0` |
| S2 | `EVAL_IMPORT`：主 `.venv` 与三个子 venv 的 `robomme`／`robomme_hard` 指 submodule、`robomme_hard_eval` 指本仓 `src/`；`EVAL_GITLINKS`：五个 gitlink 等于锁定值；`EVAL_LOCK_SAME`：主锁与三子锁版本逐包不变（只去掉 benchmark server 组 4 包与旧根项目带来的 pebble）；`EVAL_EPISODE=PASS identities=2 files_ok=2 decoded=6/6 loads_per_process=1 resets=2`；`AV1_BENCH=PASS episodes=3 decode_ok=3`（PSNR 最小 42.3～44.1 dB） |
| S3（评估仓 `32b7845`） | 全量门禁 1223 passed、6 skipped（非 slow）、101 passed（slow）；`TEST_INVENTORY=PASS unclassified=0 stale=0 exempt=13`；`TEST_CONTRACTS=PASS entries=115 verified=112 conditional=3 missing=0`；`EVAL_PATHS=PASS legacy_hits=0 exempt_legacy_map_keys=83`；`EVAL_ENTRIES=PASS scripts=1 bench_scripts=5`；`PARITY_IMPORT=PASS legacy_hits=0 files=12 light_ok=1`；`CLIENT_REPLAY_EQ=PASS routes=5 cases=13 tamper_detected=1` |
| S4 | `EXPERT_DEMOS=PASS episodes=16 files=32 frames_match=16`（用户改口径：每任务 1 局） |
| S5 | `XHARD0_RESET_PARITY=PASS shape=1x1x2 compared=2 det_diff=0 name_only=0`；`PARITY_GEN_SMOKE=PASS identities=4 sides=2 compared=4 both_fail=0 tol_over=0`；`EVAL_SMOKE_GL=PASS policies=4 identities=2 total=8 decoded=8 loads_per_policy=1`；`EVAL_SMOKE_ASTRA=PASS identities=2 decoded=6/6 loads=1 usd=0.61<=5` |
| S6 | `BENCH_LOCKED=PASS tag=v1.0-ood archived=1 push_rejected=1 gitlink_eq=1` |

## ② 版本与代码状态

- 源码锚 `SRC=fd0017d6`（旧仓开工时 HEAD；旧仓其后被另一会话推进到 `16a7fc19`，本轮一律锚定 SRC）。官方锚点 `016ac1c4`。
- benchmark 仓：`1.0`～`1.6` 本会话（搬入、测试改接、H1～H3 合并、植入执行器），`1.7`～`1.9` 另一会话的规则正本同步；锁死 `v1.0-ood`＝`a5efb99`。
- 评估仓：S5 冻结 `1beda8c`（benchmark gitlink `d22fe20`）；S5 后 `1.15`（修 `find_ffprobe`）、`1.16`（gitlink 升 `a5efb99`）。版本号 `1.12`～`1.14` 与另一会话的规则同步提交撞号（两边各有一份），此后从 `1.15` 接续。
- 子模块：mme-vla `ecf086c3`、SimpleMemVLA `c564c17d`、PonderPounce `723df357`（`hongzefu/PonderPounce` fork）、Astra-on-RoboMME `4c3fd6a8`（`hongzefu/Astra-on-RoboMME` fork）。

## ③ 子代理分工与审查

| 块 | 合并 | 审查 |
|---|---|---|
| H1 robomme_hard 裁剪 | benchmark `1.3` | PASS（一轮） |
| H2 入口与说明 | benchmark `1.4` | PASS（一轮） |
| H3 测试改接 | benchmark `1.5` | 一轮 FAIL（误删包内行级取值断言）→ 续改 → PASS |
| E1 外层三函数与 AV1 记录 | 评估 `1.4` | PASS |
| E6 站点 | 评估 `1.5` | PASS |
| E5 对拍工具 | 评估 `1.6` | 一轮 FAIL（误删比较器测试、compare 未落实不可读与同失败不计入）→ 续改 → PASS |
| E3 Astra | 评估 `1.8` | PASS |
| E2 四个普通模型 | 评估 `1.9` | PASS（7 条不阻塞备注） |
| E4 GL 席位／原侧／检查／渲染 | 评估 `1.10` | 一轮 FAIL（动态队列零测试、契约悬空、心跳竞态、回收不结算、meta 不可读误删）→ 续改 → PASS |
| MERGE-1 整合 | 评估 `1.12` | PASS |

超过 15 分钟的子代理：E1 26 min、H3 22 min、E5 27 min、E2 24 min、E3 23 min、E4 71 min、MERGE-1 62 min（SubagentStop hook 已记）。

## ④ 预算实耗（P5 乘式）

| 项 | 乘式 | 身份执行 | 说明 |
|---|---|---|---|
| BENCH_SMOKE | 1 任务（MoveCube）× 2 数据集 × 1 局 | 2 | 本机卡 0 |
| EVAL_EPISODE | 1 任务（VideoUnmask）× 2 数据集 × 1 局 + 补跑 1 | 3 | ood 局首跑被主会话外层 `timeout 900` 在网站视频渲染阶段杀掉，用户批准补跑 1 局（上限 26→27） |
| PARITY_GEN_SMOKE | 1 任务 × 2 数据集 × 2 局 × 2 侧 | 8 次生成 | 本机卡 0 |
| XHARD0_RESET_PARITY | 1 任务 × 1 档 × 2 局 × 2 侧 | 4 | 本机卡 1，每侧探针内 2 次 reset |
| EVAL_SMOKE_GL | 4 策略 × 1 任务 × 2 数据集 × 1 局 | 8 | 共享账本 reserve 8、commit 8、reset_claim 16 |
| EVAL_SMOKE_ASTRA | 1 策略 × 1 任务 × 2 数据集 × 1 局 | 2 | 费用 0.61 美元（上限 5） |
| 合计 | — | 27 | 等于用户追加后的上限 27 |
| 表外（用户追加授权） | benchmark sim 冒烟 59 + 1 | 60 次 reset | 「都同意，补跑那局，sim 冒烟也跑」 |

GL 首跑被拦（RUN_INPUTS／mme_vla_dirty）与 Astra 首跑 load 失败均发生在领局与登记之前，账本零消耗。

## ⑤ S5 任务结果（如实记录，不为挑成功局重跑）

| 模型 | hard-verify 局 0（原号 3） | ood 局 0（xhard1） |
|---|---|---|
| FrameSamp+Modulation（GL） | fail，104 步 | fail，108 步 |
| GroundSG Oracle（GL） | success，301 步 | fail，205 步 |
| SimpleMemVLA（GL） | success，262 步 | success，269 步 |
| PonderPounce（GL） | success，259 步 | success，276 步 |
| Astra（本机） | success，290 步 | success，327 步 |

## ⑥ 用户决策记录（原话）

1. 「同意，改公开并 fork 两个仓」——建公开仓被自动模式分类器拦下后的授权。
2. 「都同意，补跑那局，sim 冒烟也跑」。
3. 「你现在为什么要进行全量的渲染」「不需要渲染所有的专家演示渲染。我只需要每一个任务渲染一局就可以了」「以后任何这种全量的数据集的重新改动都要交用户审批 不可以自己决定。」
4. 「这个子代理怎么跑这么久检查一下」（MERGE-1）。
5. EVAL_SMOKE_GL 裁决：「修代码后在 GL 原地补渲（推荐）」；锁死版本：「锁在 a5efb99（推荐）」。

## ⑦ 计划外事件与处置

- 建公开仓被分类器拦下 → 先建私有仓推送，获用户授权后改公开并 fork。
- 另一会话在 20:20～20:25 向两仓各推 3 个规则同步提交（新增 `.claude/settings.json`），造成评估仓版本号撞号、benchmark HEAD 前移 → 用户裁决锁在 `a5efb99`，`.claude/settings.json` 并入 BENCH_DELTA 白名单。
- `tests/pipeline/eval` 内测试互借夹具，原分配表整目录划给 E2 不可行 → 按测试实际加载的脚本逐文件划给 E2／E4，并派整合子代理 MERGE-1。
- GL：client-env 未装评估包（RUN_INPUTS 不加 `src`）→ 启动脚本 `gl-seat2.sh` 补 `PYTHONPATH`；NFS ACL 让子模块显示可执行位改动 → NFS 副本 `core.fileMode false`；GL 节点无 ffprobe → 评估仓 1.15 修 `find_ffprobe`，GL 占位 job 内补渲 8 局（`GL_RERENDER=PASS episodes=8 decoded=8`）。
- 本机 Astra：新建 client-env 漏装 flash-attn（按 env 约定不进锁）→ 从 uv 缓存装 `flash_attn-2.8.3-cp311` wheel，`FLASH_ATTN_KERNEL=PASS maxdiff=0.0`。
- `scripts/evaluate.py` 无 `--budget-ledger`：本机 Astra 未进 GL 共享账本，局数与费用由 Astra 自有费用账本硬限（2 局、5 美元）。

## ⑧ 后续待办（不在本轮范围）

1. AV1 4:4:4 编码反压仿真（dummy 两局 `queue_wait_s` 226／279 s）：正式大批评估前把原始帧编码改为后台队列，不降级参数。
2. `dev-scripts/gl/run_eval_gl.sh` 的 RUN_INPUTS 检查应带 `PYTHONPATH=$REPO/src`（client-env 席位）；`scripts/evaluate.py` 补 `--budget-ledger` 以并入共享账本。
3. MERGE-1 留下：`dev-scripts/gl/eval_report.py` 仍读旧 stage 布局；`dev-scripts/orig/{run_official_hard.sh,orig_seat_lib.sh}` 的 `SGEVAL_CLIENT_PY` 缺省仍是旧位置；`official_hard_runner.py` 的 `BENCH_SRC` 不认 `SGEVAL_THIRD_PARTY`；`tests/mutation/plugins/mut_inproc.py` 的 `SUPPORTED` 有死块。
4. benchmark 仓 `src/robomme_hard/robomme_env/BinFill.py` 与 `utils/sampling_config.py` 各有一条指向旧脚本路径的注释（锁死前按「其余文件字节不变」保留）。
5. 停下的全量专家演示目录 `artifacts/split-accept-20261008-01/expert_demos/ood/`（128 局，含被打断的半成品）待用户决定是否删除。

## ⑨ 归档文件清单

`launch.md`（锚点、JobID、tmux 清单、GL 副本与环境、S3／S4 判定行）、本文件。原始日志与产物在本机 `artifacts/split-accept-20261008-01/`（`logs/`、`eval-episode/`、`parity-gen/`、`xhard0-reset/`、`astra/`、`expert_demos_ep0/`、`gl-rerender/`、`s1/`、`s3/`）与 NFS `RoboMME-benchmark-OOD-eval/artifacts/split-accept-20261008-01/`（`gl/`、`budget-ledger.jsonl`、`gl-identities.jsonl`、启动与补渲脚本；已整份 rsync 回本机 `artifacts/split-accept-20261008-01/gl-nfs/`）。

## ⑩ S7 收尾

- 规则同步：`sync_rules.py check --repo benchmark-ood`／`--repo eval-ood` → `SYNC_SUMMARY=PASS pass=3 fail=0 missing=0`（正本 `29a4050`）。
- 旧仓：`docs/plans/1008-split-benchmark-eval-repos-plan.html`（`git mv` 自根目录，12.562），归档 tag `archive-newtask-v9-20261008` → `5f4f2f9`（已推送）。
- worktree：benchmark 仓 H1～H3、评估仓 E1～E6 与 MERGE-1、旧仓只读快照 `split-old-src` 共 11 个已 `git worktree remove`，`sub/*` 分支 `git branch -d`；删前删后清单差集恰为这 11 个，旧仓原有 4 个 worktree 未动。
- tmux：本轮 `split-*` 会话均已随命令结束，`tmux ls` 无会话，无需清理。
- GL 占位 job `63431430～63431433`：用户 2026-10-08 选「保留给后续工作」，不取消。
- `CLEANUP=PASS worktrees=11 tmux=0 jobs=0（4 个占位 job 按用户决定保留）`
