# dev-scripts/parity/：对拍工具（与官方比、新旧代码树比、生成噪声回归）

评估仓的对拍设施。benchmark 包（`robomme` 与 `robomme_hard`）一律取自子模块 `third_party/robomme_benchmark/src`（钉 40 位 sha），本目录不复制其代码。`dev-scripts` 带连字符、不是包：各脚本把本目录放进 `sys.path`（`_common.py`）后按同目录模块名互相 `import`；`hard_specs` 在不该导入仿真的进程里按文件路径轻量加载（`_common.hard_specs_light`，不经 `robomme_hard` 包 `__init__`）。

数据集按名取局：`hard-verify` 只有 xhard0（官方 test 的 hard 子集，每任务 12 局，builder 局 0～11 对应官方原 episode 3, 7, …, 47）；`ood` 只有新值五档（V9 每任务 50 局）。

| 文件 | 干什么 |
|---|---|
| `_common.py` | 路径设置：本目录进 `sys.path`；`REPO_ROOT`（评估仓根）、`bench_root()`（子模块根；子模块未检出时退回当前解释器能找到的 `robomme_hard` 源码树，只查找不导入）、`CONFIGS`、`OFFICIAL_ROOT`、`hard_specs_light()` |
| `official/` | 官方生成编排 `d53f21a7:scripts/data-generation/` 四文件逐字节 vendor（`generate_dataset.py` 依赖另三个）；`SOURCE.json` 记 url／commit／tree／逐文件 sha256。不得修改（红线 R2） |
| `configs/` | 对拍配置五份：`hard-parity-tolerances.json`（容差，R21）、`xhard0/xhard0_manifest.json`（xhard0 192 局身份）、`gate-set-v9-129.json`、`gate-set-xhard0-48.json`（固定检查集）、`noise-ref-20261003.json`（噪声基线逐局参照） |
| `train_split_runner.py` | 隔离运行器：用 vendor 的官方编排调官方 `_worker`（`--src-root` 指环境源码树；官方侧给子模块根）或镜像 worker（`--force-mirror`／`--sampling-config`／`--episode-specs`）。`--identity-source test_metadata` 把 jobs 与官方 test 元数据 hard 子集、`--xhard0-manifest` 双向核对；`--builder-route hard-verify`（H 侧）让镜像 worker 的 gym.make 实参取自构建器 `dataset="hard-verify"`；逐局追加 `results.partial.jsonl` 并 fsync，`--resume` 续跑。16 任务规范序按文件读 `<src-root>` 的 `hard_specs.ALL_TASKS` |
| `train_split_worker.py` | 镜像 worker：官方 `_worker` 的最小镜像，只多传 `sampling_config`／`native_episode_spec`；环境包由 `ROBOMME_ENV_PACKAGE`（`robomme`／`robomme_hard`）决定，结果记 `env_module`／`wrapper_modules` |
| `generate_h5.py`、`_rollout.py`、`_freeze_min.py` | 生成链路：`continue`／`split`／`aggregate`（gen1 交付）与 `replay`（对拍专用，按身份清单只读重放，缺省读包内规格）；`--src-root` 缺省子模块根，`--expect-src` 断言本进程的 `robomme_hard` 位于给定 src 下。`_freeze_min.py` 只留 `_movecube_way`、`write_jsonl_exclusive` 两个纯函数（规格抽签封存随拆仓退役） |
| `hard_parity.py` | 对拍入口：`generate`（档 `native`／`xhard0`／`v9`；A40 断言、每局 sha 后搬到 NFS 暂存并写 `SHIPPED`；`--identities` 筛身份子集，`--expect-ref` 边生成边对噪声基线判定）、`publish`、`compare`（判定层 + 容差层 + 参考层 → `PARITY_*`；`both_fail` 单列）、`anchor register／check`（登记表 `docs/validation/parity-anchors.json` 缺失即报错）、`export-xhard0-manifest`、`import-delivery`、`binding`，以及拆仓验收的 **`smoke`**（见下） |
| `hard_pull.py` | sled-vail 上逐局拉取：NFS 暂存 → `/data` → 核 sha → 删 NFS 副本（只拉带 `SHIPPED` 的局） |
| `hard_regression.py` | `tier-values`（→ `V8_TIER_VALUES`）、`eval-smoke`（`--dataset hard-verify|ood` → `HARD_EVAL_SMOKE`）、**`xhard0-reset-parity`**（见下）、`env-digest`／`env-digest-worker`／`env-digest-compare`。生成阶段子命令 `delivery-set`、`reset-replay`、`step-headroom`、`movecube-layout` 与 `--specs-root` 换规格根随拆仓删除 |
| `gate_set.py` | 生成噪声基线的固定检查集：V9 每个交付格取 candidate 最小 3 局（43 格 × 3 = 129）、xhard0 16 任务 × 1 档 × 3 局 = 48；`check`／`check-xhard0` → `GATE_SET`／`GATE_SET_XHARD0` |
| `noise_run.py`、`noise_run_gl.sh` | GL 单遍运行包装（只留生成线）：`preflight`、`finish`、`ship`；壳脚本最后一行 `EXIT_CODE=` |
| `noise_gate.py` | 生成噪声比较器与逐局回归闸门：`gen-compare`、`gen-regress build-ref／check`、`selftest` |

## 拆仓验收的两条对拍

```bash
# hard-verify 在新仓与官方逐位一致：VideoUnmask 1 任务 × hard-verify 局 0、1（官方原号 3、7）= 2 局
uv run --no-sync python dev-scripts/parity/hard_regression.py xhard0-reset-parity \
  --tasks VideoUnmask --episodes 0:2 --out <输出目录>          # XHARD0_RESET_PARITY=PASS shape=1x1x2 compared=2
# 生成链路行为一致：1 任务 × {hard-verify, ood} × 2 局 × 2 侧 = 8 次生成
uv run --no-sync python dev-scripts/parity/hard_parity.py smoke \
  --task VideoUnmask --hard-verify-episodes 0:2 --ood-episodes 0:2 \
  --hard-verify-sides O,H --ood-sides H_old,H_new --h-old-src <旧仓 SRC 只读快照>/src \
  --workers 2 --out <输出目录>   # PARITY_GEN_SMOKE=PASS identities=4 sides=2 compared=4 both_fail=0 tol_over=0
```

- `xhard0-reset-parity`：官方侧在独立进程只导入 `--src-root`（缺省子模块）的 `robomme`（`dataset="test"`，进程内断言未导入 `robomme_hard`），hard 侧 `dataset="hard-verify"`；官方原 episode 号以 hard 侧 builder 解析出的 `source_episode` 为准并与清单核对；分母 = 任务数 × 所选局数。
- `smoke`：hard-verify 两侧 O（官方 `_worker` + 子模块 `robomme`）对 H（镜像 worker + 子模块 `robomme_hard`）；ood 两侧 H_old（旧仓 venv 解释器，`PYTHONPATH` 指向 `--h-old-src`，进程内断言 `robomme_hard.__file__` 在其下）对 H_new（子模块）。身份以 builder 解析为准，ood 新旧两棵树解析出的身份须相同。生成次数硬上限 8，超出即拒跑。PASS 须两侧都有可读 h5、实际比较数等于身份数、`both_fail`（单列，不计入通过）为 0、超容差为 0、结构与包绑定全对。

## 对拍判定（行为一致，用户 U-1）

- **每侧自检**：身份唯一、h5 `setup` 的 seed／difficulty 与身份一致、`timestep_*` 连续、数据集可读。
- **判定层（全等才 PASS）**：身份集合、`setup`（seed、difficulty、task_goal、多选项、相机内参）、结构（数据集名／dtype／shape 集合，不比帧数）、任务成功相等且双侧各自成功（`both_success`，同失败不算相等）、包归属（`ENV_PACKAGE_BINDING`）。
- **容差层（U-19）**：共同前缀上的 `action_max`、`state_max`、`image_mad`、`frames_max`。阈值只从 `configs/hard-parity-tolerances.json` 读（R21），由 `compare --pair O:P --calibrate` 按「O:P 最大值 × 1.5，下界 0.005/0.005/1.0/5，合理性上界 0.05/0.05/10/200」写入。
- **参考层（INFO）**：`sha_equal`、帧数相等数、首个分叉时间步分布。

判定不做字节级：mplib RRT 的墙钟预算让多 worker 并发下同身份轨迹分叉。

## 拆仓时没有搬入的

`upstream_guard.py`（由 benchmark 仓的 `BENCH_UPSTREAM` 闸门取代）、`train_split_{parity,audit,comparison,config}.py`、`comparator_fixtures.py`、原三档对拍配置目录，以及规格抽签封存一族（`_draw`、`_extract`、整份 `_freeze`、`append_candidates`、`freeze_specs`、`seed_layout`、`v9_movecube_region_fig`、`v9_subset_specs`）。原文在旧仓归档 tag 的 git 历史里。
