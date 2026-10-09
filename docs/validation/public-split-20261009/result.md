# public-split-20261009：公开 main 与 dev 分支拆分（result）

计划：[`docs/plans/1009-public-main-dev-branch-split-plan.html`](../../plans/1009-public-main-dev-branch-split-plan.html)（第六稿，1.21 入库）。本档记 S1～S6 的实施结果与判定行。

## ① 一句话结论与判定速览

仓库已拆成两条分支：`dev`（全部内容，唯一开发分支，中文为主）与公开面 `main`（79 个文件 + 5 个子模块指针，全英文、零 CJK，ckpt 必须手动传入）；`main` 由 dev 上的 `dev-scripts/release/sync_to_main.sh` 按 `public-manifest.txt` 单向同步，两边同名文件逐字节相同；S1～S5 全部判定行 PASS。

| 判定项 | 判定行 | 在哪、何时 |
|---|---|---|
| 改名干净 | `RENAME_CLEAN=PASS hits=0 pkg_in_repo=1` | dev 主检出，S2-0（1.22）；排除 `docs/validation/` 与 `docs/plans/` |
| main 无中文 | `PUBLIC_LANG=PASS files=79 cjk_hits=0 binary_skipped=0` | dev 主检出，S2 收尾（1.38）；每次同步复跑 |
| main 无私有路径 | `PUBLIC_PATHS=PASS files=79 hits=0` | 同上 |
| main 恰等于清单 | `PUBLIC_MANIFEST=PASS listed=79 missing=0 extra=0 collected=28 gitlinks=5` | dev 主检出 `--collect`（1.38）；S3 孤儿检出 `--tree` 再核 `extra=0` |
| 全历史无凭据 | `PUBLIC_SECRETS=PASS commits=67 hits=0 allowlisted=15` | S3 前置（1.38 后，全部 ref） |
| 子模块可取 | `SUBMODULE_PUBLIC=PASS repos=5 public=5 reachable=5` | S3 前置 |
| ckpt 无默认 | `CKPT_NO_DEFAULT=PASS exit=2` | src、scripts 零 `DEFAULT_CKPTS`；`evaluate.py --model smvla` 不带 `--ckpt` 退出码 2 |
| 翻译不改行为 | `TRANSLATE_EQ=PASS passed=1242 base=1242` | 每次合并后全量短测均 `1242 passed, 6 skipped, 101 deselected`（BASE1 1228 + A1 新增 14） |
| main 能跑 | `PUBLIC_SMOKE=PASS model=dummy episodes=1 exit=0 tests_passed=254` | 见 ⑤ |
| 同步可用 | `SYNC_MAIN=PASS dev=f786143 main=0bca124 files=79 changed=1`、`SYNC_MAIN=NOOP dev=f786143 main=0bca124 files=79 changed=0` | S5；其后 1.41 再同步 `main=3482d7b`，public-gate success |
| 公开与保护 | `PUBLIC_SECRETS=PASS commits=74 hits=0 allowlisted=18`、`SUBMODULE_PUBLIC=PASS repos=5 public=5 reachable=5`；visibility=public、default=main；main 保护 force_push=false deletions=false enforce_admins=true pr_required=false；dev protected=false | S6 |

## ② 版本与锚点

- `OLD_MAIN=f1a0c3652eb4585fa93172bc8c2e85913cccc0b3`（1.21 计划入库时点）；留档 tag `private-main-20261009` 指向它并已推送。
- S3 孤儿提交 `7ad6fb4`「Initial public release」，`Dev-Source: 398aa1b`（1.39）。最初构造的 `8a76cc5`（`Dev-Source: 8863e1a`）因 S4 预演发现问题，推送前改为 `7ad6fb4`，从未推送。
- S3 强推：`git push --force-with-lease=main:f1a0c36… origin main-new:main` → `+ f1a0c36...7ad6fb4 main-new -> main (forced update)`。第一次执行被本机自动模式分类器拦截，用户原话「授权执行 S3 的 --force-with-lease 强推」后执行；推送前 `ls-remote` 确认远端 main 仍为 OLD_MAIN。
- S5 后 main：`7ad6fb4` → `0bca124`（README 子模块初始化）→ `3482d7b`（`__pycache__` 测试修复），两次均为 `sync_to_main.sh` 快进推送。
- dev：1.21～1.41 本轮提交，子模块 gitlink 未动。

## ③ 改名记录

`robomme_hard_eval` → `robomme_ood_eval`，发行名 `robomme-hard-eval` → `robomme-ood-eval`（1.22，90 个文件 ±486 行，不留兼容层）。`docs/validation/legacy-names.md` 已加一行对照。本机 `.venv` 已 `uv sync --extra dev`；GL 的 NFS 副本 venv 仍是旧名，下次 GL 评估前 rsync 后在 sled-vail 对其 `UV_LINK_MODE=copy uv sync --extra dev`。

## ④ 子代理分工、合并与审查

| 块 | 合并 | 第一次审查 | 第二次审查（合并后全量短测） |
|---|---|---|---|
| A1 ckpt 接口 + evaluate.py 翻译 | 1.24 | PASS（4 条非阻断） | 1242 passed |
| D dev 侧显式传 ckpt | 1.25 + 1.26 | PASS（3 条非阻断） | 首次 2 failed（`ckpt_paths.sh` 未登记契约总表）→ 1.26 前向修补 → 1242 passed |
| T0 测试拆分 | 1.27 | PASS（2 条非阻断） | 1242 passed |
| A2 src 全部翻译 | 1.32 | PASS（2 条 minor） | 1242 passed |
| T1 根／_support／unit_eval／static | 1.34 | PASS | 1242 passed |
| T4 groundsg／report | 1.35 | PASS | 1242 passed |
| T2 pipeline/eval | 1.36 | PASS | 1242 passed |
| T3 astra／pp | 1.37 | PASS | 1242 passed |

主会话自做：1.22 改名、1.23／1.28 发布工具与覆盖项、1.29 README 与声明文件注释、1.30／1.31 清单与 public-gate、1.33 入口文档串、1.38 dev-gate、1.39～1.41 S4/S5 发现的修补。合并顺序与计划的偏离：D 先于 T0、T4 先于 T2/T3（审查先完成、文件不相交）。超过 15 分钟的子代理：T0 约 21 min、A2 约 38 min（hook 已记）。

## ⑤ S4 验收（干净 clone）

- `git clone --branch main https://github.com/hongzefu/RoboMME-benchmark-OOD-eval.git` + `git submodule update --init`：5 个 gitlink 与 dev 相同，两个嵌套 benchmark 未初始化。
- `UV_LINK_MODE=copy uv sync --extra dev`；`robomme_ood_eval.__file__` 在 clone 内 `src/`。
- `pytest -m 'not slow'` → `254 passed, 5 skipped, 18 deselected`，`TEST_RESOURCE=PASS violations=0 not_verified=5`（5 个为缺 `envs/client-env` 的 Not verified 跳过）；全部跟踪文件 CJK 扫描 0 行。
- dummy 冒烟（1 任务 × 1 档 × 1 局 = 2 次 reset；另有一次被 290 s 超时截断的同规模尝试，合计 4 次，在 P3 单 worker 10 次阈值内）：在 S4 预演 clone（src 与公开 main 逐字节相同，只差一个测试文件）的 tmux 会话 `ps4-dummy` 中运行，`EPISODE_DONE … status=timeout exec_steps=1301`，`EVAL_LOG … episodes=1 counted=1 success=0 infra=0`，`EXIT_CODE=0`；产物 `results.jsonl`、`log.json`、`progress.json`、官方版式视频、raw 目录齐全。

## ⑥ 计划外事件与处置

1. **gh 凭据缺 `workflow` scope**：推送 `.github/workflows/*.yml` 被拒；用户执行 `gh auth refresh -s workflow` 后补入（1.31、1.38）。
2. **D 合并后 `test_inventory` 失败**：新文件未登记契约总表；1.26 前向修补（合并提交当时未推送）。
3. **MemER 实现指纹变化**：A2 翻译了 `_official_defs.MEMER_COMPAT_SOURCE` 补丁源文本里的注释，`MEMER_COMPAT_SHA256` 由 `105f05113a1e5f0b36ce5e4a7d050aff661694dbb770c4f88c5dbd35dd4e3814` 变为 `c6285247a0964aa819fcdcacf3cfc48b289e116318820b61d43abd338cd094ce`；审查以 AST 与去注释逐行比较确认补丁代码不变。与旧结果（如 `sg-eval-gl-20261006-03`）对比 MemER 指纹时按此对应。
4. **资源守卫 skip 前缀**：公开测试不能含中文，T1 把 `resource_policy.py` 的判定改为接受 `("Not verified", "未验证")`，公开侧统一英文前缀。
5. **S4 预演发现 `test_pp_server_wrap` 在无 `envs/client-env` 时硬失败**：1.39 改为 Not verified 跳过（本机 dev 行为不变）。
6. **README 原写 `git clone --recurse-submodules`**：会填满 `mme-vla`／`Astra-on-RoboMME` 内嵌的 benchmark，违反 MME-VLA 预检（`RUN_BLOCKED reason=nested_submodule_not_empty`）；1.40 改为只初始化第一层，并作为 S5 第一次同步。
7. **public-gate 首跑失败**：`test_scripts_dir_has_only_evaluate` 把 CI 运行时生成的 `scripts/__pycache__` 算进目录；1.41 修复后再同步。
8. 与计划数字的出入：公开清单 79 个文件（计划估约 120）、公开测试收集 28 个文件（计划 27，多出 A1 的 `test_ckpt_required.py`）、进 main 的 tests 文件 42 个（含 `entry_bootstrap.py`，计划原判留 dev，因公开测试子进程运行它而改判）。

## ⑦ 用户决策记录

计划「已定口径」①～⑮逐条见计划文件第一部分「一」，本轮新增：

1. 「/data/hongzefu/RoboMME-benchmark-OOD-eval/1009-public-main-dev-branch-split-plan.html 开始实施 有问题问用户 但是不要阻塞」（开工令）。
2. 「gh auth 已经 refresh 好了」。
3. 「授权执行 S3 的 --force-with-lease 强推」。
4. MemER 指纹变化：主会话接受新值并在本档记新旧对应，已告知用户，用户未提异议。
5. 「main开保护 改public dev不要动」；AskUserQuestion 答复「照计划，dev 一起公开」。

## ⑧ S6 执行

- 先尝试给 main 开保护：私有仓库在免费账户下返回 403（「Upgrade to GitHub Pro or make this repository public」），顺序改为先公开、后保护。
- 公开前用 AskUserQuestion 确认「GitHub 无法只公开 main，dev 会一起公开」，用户选「照计划，dev 一起公开」。
- 重跑 `PUBLIC_SECRETS=PASS commits=74 hits=0 allowlisted=18`、`SUBMODULE_PUBLIC=PASS repos=5 public=5 reachable=5`。
- 本机 gh 2.45 不认 `--accept-visibility-change-consequences`，改用 `gh api -X PATCH repos/hongzefu/RoboMME-benchmark-OOD-eval -f visibility=public` → `public private=false default=main`。
- main 分支保护（`PUT …/branches/main/protection`）：禁强推、禁删除、enforce_admins、不要求 PR、无必需检查；dev 按用户「dev不要动」不加保护。
- 计划原写 dev 同样禁强推与删除，按用户指令偏离。

## ⑨ 当前状态与下一步

- 仓库公开，默认分支 main = `3482d7b`，public-gate 与 dev-gate 均为 success。
- 以后改公开面：只在 dev 上改，用户说「同步到 main」才跑 `sync_to_main.sh --message "<英文摘要>"`。
- 下次 GL 评估前：rsync 后对 NFS 副本 `UV_LINK_MODE=copy uv sync --extra dev`（包名已改）。
