# scripts/parity/：与官方比

拆包后（[0927 计划](../../docs/plans/0927-robomme-hard-layered-plan.md)）本目录只做「与官方比」。新值档的生产链路在 [`../injection-dev/`](../injection-dev/)，新值环境源码在 [`src/robomme_hard/`](../../src/robomme_hard/README.md)。

| 文件 | 干什么 |
|---|---|
| `official/` | 官方生成编排 `d53f21a7:scripts/data-generation/` 四文件逐字节 vendor（`generate_dataset.py` 依赖另三个）；`SOURCE.json` 记 url／commit／tree／逐文件 sha256。不得修改（红线 R2） |
| `upstream_guard.py` | G1 守卫：`build` 生成 `src/robomme_hard/UPSTREAM.json`；`check` 输出 `UPSTREAM_BYTES`（`src/robomme` 与官方 `1fadc0ec` 逐字节）、`VENDOR_SAME`、`SHIMS`、`ABS_IMPORT`、`BORROWED_DEPS`（借用 shim 目标在官方源码上的传递闭包不含任何被复制的模块），`--require-upstream`、`--net`；`manifest-md` 出复制／借用／子类／新增逐文件表。纯 CPU、秒级 |
| `train_split_runner.py` | 隔离运行器：用 vendor 的官方编排调官方 `_worker`（A 路：`--src-root` 指官方源码树；P 侧：指 tag `pre-hard-split` 的 worktree）或镜像 worker（`--force-mirror`／`--sampling-config`／`--episode-specs`）。`--official-root` 默认 `official/`，此时 `--src-root` 必填；`--metadata-root` 显式把官方 train 元数据根（原默认 `configs/newtask-v3/official_train` 已于 12.214 删除，须显式传入或从提交 `6e70c0bf` 取回；核 `records_sha256`）传给官方 `read_train_metadata`；逐局追加 `results.partial.jsonl` 并 fsync，`--resume` 续跑 |
| `train_split_worker.py` | 镜像 worker：官方 `_worker` 的最小镜像，只多传 `sampling_config`／`native_episode_spec`；环境包由 `ROBOMME_ENV_PACKAGE`（`robomme`／`robomme_hard`）决定，结果记 `env_module`／`wrapper_modules` |
| `train_split_config.py` | 从环境源码提取 `sampling_config`（`extract --pkg robomme_hard --release newtask-v7`，v7 起为默认；缺省落点 `artifacts/hard-split/`） |
| `train_split_parity.py`、`train_split_comparison.py`、`train_split_audit.py`、`comparator_fixtures.py` | S0 原始 train 五路逐位对拍设施（`compare_h5_pair` 等），见 git 历史里 V3 的说明 |
| `hard_parity.py` | 四侧对拍入口（O 官方 / P 修改前 / H 修改后 / H2 第二次生成；档 `native`／`xhard0`／`v9`，v6 四档 `xhard` 与 v8 正式局 `v8` 已于 1003 维护计划删除；`compare` 先核对分母）：`generate`（A40 断言、每局 sha 后搬到 NFS 暂存并写 `SHIPPED`；`--identities` 筛身份子集，`--expect-ref` 对噪声基线参照边生成边判定、翻转局写 `flips.jsonl`）、`publish`（上传 bucket 并逐对象读回核 sha → `BUCKET_SYNC`；`BUCKET` 常量仍指向已删的旧 bucket，归 HF 发布方案处理）、`compare`（判定层 + 容差层 + 参考层 → `PARITY_*`；`--p-anchor` 按登记的锚点取 P 侧并先核验；两侧都失败单列 `both_fail`；`--calibrate` 只许 O:P，按 U-22 规则写 `configs/hard-parity-tolerances.json`）、`anchor register／check`（锚点登记表 `docs/validation/parity-anchors.json` → `PARITY_ANCHOR`）、`export-xhard0-manifest`（从官方 test 元数据导出 xhard0 192 局清单 `configs/xhard0/xhard0_manifest.json`）、`import-delivery`（gen1 交付清单转对拍清单，`--tier v9`）、`binding`（`ENV_PACKAGE_BINDING`） |
| `hard_pull.py` | sled-vail 上逐局拉取：NFS 暂存 → `/data` → 核 sha → 删 NFS 副本（只拉带 `SHIPPED` 的局）；`--identities` 只拉给定身份（翻转重跑清单，允许空清单），`--expect-ref` 写出的 `match` 局按设计不复制 |
| `hard_regression.py` | `delivery-set`／`tier-values`（/4 交付形态、seed 按档隔离、布局独立与档位取值 → `V9_DELIVERY_SET`／`V9_SEED_DISJOINT`／`V9_LAYOUT_INDEPENDENT`／`V8_TIER_VALUES`，前缀按格表版本）、`reset-replay`（读包内 /4 规格，43 格各 1 次回注 reset 经 builder 评估链 → `V9_RESET_REPLAY`；非 /4 规格根直接拒收）、`eval-smoke`（合作者入口冒烟 → `HARD_EVAL_SMOKE`）、`xhard0-reset-parity`（xhard0 O:H 的 reset 层对拍，逐局另起子进程取演示前状态，`name_only` 单列；`--manifest` 缺省 `configs/xhard0/xhard0_manifest.json`）、`step-headroom`（交付 h5 非演示步 ≤ 1600、超限过滤数、xhard0 ≤ 1300 → `V9_STEP_CAP`）、`movecube-layout`（→ `V9_MOVECUBE_LAYOUT`／`V9_MOVECUBE_WAYS`）、`env-digest`／`env-digest-worker`／`env-digest-compare`（逐层摘要与对拍）。v7 专用的 `layout-shared`、`prefix-geometry`、`xhard0-eval-parity`、`step-headroom --v7` 已于 1003 维护计划删除 |
| `gate_set.py` | 生成噪声基线的固定检查集：V9 每个交付格取 candidate 最小 3 局（43 格 × 3 = 129，冻结 `scripts/configs/gate-set-v9-129.json`）、xhard0 小样本 16 × 3 = 48（冻结 `gate-set-xhard0-48.json`）；`check`／`check-xhard0` 输出 `GATE_SET`／`GATE_SET_XHARD0`，`export` 导出 generate／env-digest／legacy 身份清单。`hard_parity.py generate --identities` 自 12.363 起也接受 xhard0 子集（只筛作业，运行器仍拿完整 192 行清单） |
| `noise_run.py`、`noise_run_gl.sh` | GL 单遍运行包装（只留生成线，`--kind` 只认 `gen`）：`preflight`（输出根必须不存在或为空 → `RUN_FRESH=FAIL`；flock + fsync 追加式预算账本按 kind 与总量核上限；资产与 tokenizer 全量 sha256 对资产锁；环境指纹与来源报告）、`finish`、`ship`（核对 `SHIPPED` 逐局搬运，`--finalize` 收尾）；壳脚本最后一行 `EXIT_CODE=`。评估分片 `eval-shard` 已于 1003 维护计划删除 |
| `noise_gate.py` | 生成噪声比较器与逐局回归闸门（只留生成线）：`gen-compare`（h5 全数据集逐帧比较，互斥五类 byte_equal／diverge／gen_fail／structural／unknown；多态字段白名单 `POLYMORPHIC_KEYS`）、`gen-regress build-ref`（从噪声基线四遍写逐局参照 `configs/noise-ref-20261003.json` → `NOISE_REF`）、`gen-regress check`（新跑对参照逐局判定、翻转重跑确认 → `GEN_REGRESS`）、`selftest`（`GATE_SELFTEST`）。评估抽取、reset 摘要、统计线与冻结／核验子命令（`eval-extract`、`run-fresh`、`baseline-*`、`freeze`／`verify`、`check-*`）已于 1003 维护计划删除；口径见 `docs/1003-noise-baseline.md` |
| `results/` | 历史日志（被 ignore，按计划不删） |

## 对拍判定（行为一致，用户 U-1）

- **每侧自检**：身份唯一、h5 `setup` 的 seed／difficulty 与身份一致、`timestep_*` 连续、数据集可读。
- **判定层（全等才 PASS）**：身份集合、`setup`（seed、difficulty、task_goal、多选项、相机内参）、结构（数据集名／dtype／shape 集合，不比帧数）、任务成功相等且双侧各自成功（`both_success`，同失败不算相等）、包归属（`ENV_PACKAGE_BINDING`）。
- **容差层（U-19）**：共同前缀（两侧按时间步对齐的公共长度）上的 `action_max`（`joint_action` 最大绝对差，rad）、`state_max`（`joint_state`／`gripper_state`）、`image_mad`（前视与腕视 RGB 逐像素平均绝对差的每身份均值，取最大）、`frames_max`（帧数差）。阈值只从 `configs/hard-parity-tolerances.json` 读（R21），由 `compare --pair O:P --calibrate` 按「O:P 最大值 × 1.5，下界 0.005/0.005/1.0/5，合理性上界 0.05/0.05/10/200」写入。
- **参考层（INFO）**：`sha_equal`、帧数相等数、首个分叉时间步分布。

判定不做字节级：mplib RRT 的墙钟预算让 16 worker 并发下同身份轨迹分叉（见 `docs/validation/newtask-v3/20260921-worker-nondeterminism.md`）。

## 历史

2026-09-22 起本目录由 `scripts/test-vs-original/` 改名而来。拆包前的「vs 原版发布集容差校验」全文（`tolerance.json` 三档、16x3 身份与实测结果）以及 V4/V5 的 `v4_specs`／`v4_rollout`／`v5_generation` 说明已移出，原文见 `git show 7a6cee35:scripts/parity/README.md`。
