# V8 双模型评估运行档案：launch（起跑时写定）

计划：`docs/plans/1001-v8-post-evaluation-gl-plan.md`（12.304～12.308）。本文件记录首次实际启动前的代码、权重、规格、资源与命令；结果见同目录 `result.md`。

## 一、用户授权口径（2026-10-02）

- 「/data/hongzefu/robomme_benchmark_MotionJEPANewTask/1001-v8-post-evaluation-gl-plan.md 继续做evaluation」
- 「批准所有的gl job」：十席 A40 占位作业（计划第二部分 §4 一次确认清单第 ③ 项）。
- 「批准所有的reset额度」：reset 预授权上限按计划 §4 放到十倍 43280。
- 「尽可能一口气坐下去直到做到早上十二点为止尽可能不要停尽可能给我最完整的结果」

此前已定（计划引言 ⑧～⑩）：执行段 1600 步严格截断计 timeout、第 1600 步成功记成功、tokenizer 以 GCS 校验和为可信源、中午未完成继续跑。

## 二、代码与环境

| 项 | 值 |
|---|---|
| 评估代码提交 | `35b1e0b0`（12.310；E-A 合入 `527297f5`、E-B 合入 `35b1e0b0`；E-C 未合入，见 §七） |
| GL 执行副本 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-v8two`（本机 `git clone --no-hardlinks`，`origin` 指回本机；工作区 clean） |
| 子模块 | `third_party/mme-vla` = `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，`third_party/SimpleMemVLA` = `c564c17d276d7294200122b286c21901a3bfb99f`（`git -c protocol.file.allow=always submodule update --init`，嵌套 `robomme_benchmark` 保持空） |
| bench venv | `<副本>/.venv`，`uv venv --python <NFS uv-python 3.11.14>` + `uv sync --frozen --group eval-client`（首次漏装 `eval-client` 组，冒烟 1 因 `openpi_client` 缺失失败，见 §六） |
| MME venv | `<副本>/third_party/mme-vla/.venv`（3.11.14，`uv sync --frozen`） |
| SimpleMemVLA venv | `<副本>/artifacts/v8-two/venvs/smvla-env`（3.10.19，`scripts/eval-official/smvla-env` 的 `uv sync --frozen`） |
| 环境脚本 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v8two-scripts/setup_clone.sh`，日志 `artifacts/v8-evaluation/logs/setup_clone.log` |

## 三、输入与资产锁

- 身份源：`artifacts/v8-evaluation/inputs/eval-identities-1262.jsonl`（`export_eval_identities.py`，`EVAL_IDENTITY_EXPORT=PASS episodes=1262 xhard0=192 cells=43 cell_mismatch=0`）。
- 执行清单：`v8_manifest.py --shards 10` → `artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01/manifest/`，`manifest.json` sha256 `8b5810b344e25c9cce21cbfcb88c5a65e20e6613582401de9bf6addeb3671408`；`V8_EVAL_SHARDS=PASS shards=10 missing=0 extra=0 duplicate=0 total=1070 cells=43 xhard0=0`，十片各 107 局。副本在 NFS `<运行根>/inputs/`。
- MME 权重：可信期望值取 `artifacts/v7.5eval/assets-lock.json` 的 `mme-gl`（NFS `eval-out/mmevla-ckpt/perceptual-framesamp-modul/79999`，18 文件逐个 sha256）→ 现算 `missing=0 extra=0 mismatch=0`。该路径是符号链接，`run_seat.sh` 以 `$MME_CKPT/..` 读 `history_config.txt` 会落到链接目标父目录（无此文件，冒烟 1 `RUN_BLOCKED reason=history_config`），故在 `<运行根>/ckpt/mme/perceptual-framesamp-modul/` 建真实目录：`history_config.txt`（`perceptual-framesamp-modul.yaml`）+ `79999/` 以 `cp -al` 硬链接（18 文件与链接目标同 inode）。
- SimpleMemVLA 权重：`assets-lock.json` 的 `smvla`（NFS `SimpleMemVLA/checkpoints/simplememvla_robomme`），锁定 9 文件 `missing=0 mismatch=0`；另有 24 个锁外文件全是 HuggingFace 下载簿记（`.cache/huggingface/**`、`.DS_Store`，2026-09 下载时生成），不参与加载。
- tokenizer：GCS 官方对象 `gs://big_vision/paligemma_tokenizer.model`（generation 1711547605575873）元数据 `md5Hash=FCCtyYVnIKVZ6KhyhLGV4g==`、`size=4264023`（存 `artifacts/v8-evaluation/inputs/gcs-paligemma_tokenizer.meta.json`），本机缓存现算 md5 与大小一致 → 锁值 sha256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`；`OPENPI_DATA_HOME=<运行根>/openpi-data`，server 启动前 `TOKENIZER_SHA=PASS`。

## 四、席位（清单 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/gl-hold-logs/hold-jobs-v8ev-20261002.txt`）

v8ev-hold-00～09 = 63071528～63071537，各 1 A40／4 CPU／48 GiB／48 h，`chaijy2/spgpu`，`--gpu_cmode=shared`，`--wrap='sleep infinity'`。03:19 提交，00～06 约 03:20 RUNNING，07～09 排队（Priority）。

## 五、命令

运行根 `R=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v8two-out/v8-two-policy-gl10-20261002-01`。

- 单席包装 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v8two-scripts/seat.sh <子目录> <NN> <shard json> <JobID> <reset额度> <infra额度>`：清 `SLURM_*` 后
  `srun --jobid=<JobID> --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared bash <副本>/scripts/eval-official/run_v8_gl.sh --run-name v8-two-policy-gl10-20261002-01 --seat NN --repo <副本> --stage $R/<子目录> --shard <json> --policies smvla,mme --mme-ckpt $R/ckpt/mme/perceptual-framesamp-modul/79999 --smvla-ckpt <NFS>/SimpleMemVLA/checkpoints/simplememvla_robomme --openpi-data-home $R/openpi-data --tokenizer-sha256 8986bb4f…8fc6 --reset-budget <n> --infra-retry-budget <n>`，输出 tee 到 `$R/<子目录>/logs/seat-NN.log`。
- 正式：`launch_official.sh 0..9`，GL 登录节点 tmux `ev-v8-v8-two-policy-gl10-20261002-01-sNN`，作业未 RUNNING 时会话内每 60 s 查一次；子目录 `run`，`--shard $R/inputs/shard-NN.json`，reset 额度 2140、infra 重试额度 1（每账本）。
- 进度：`v8two-scripts/progress.py`（只读汇总各席 results.jsonl 与日志异常行）。

## 六、冒烟（E2，子目录 `smoke`、`smoke2`，不计入正式分母）

身份：VideoUnmask_xhard1_16600000（有演示）、SwingXtimes_xhard5_24300001（最高档），两模型各 1 局，席位 00，reset 额度 80、infra 1。

- `smoke`（04:10）：失败。SimpleMemVLA 两局 `ModuleNotFoundError: openpi_client`（bench venv 漏装 `eval-client` 组），连接 server 前即退出、执行 0 步；MME `RUN_BLOCKED reason=history_config value=''`（符号链接，见 §三）。修复后重跑。此两局超出冒烟回合预算 4（共 6 次尝试，其中 2 次零执行步），如实记录。
- `smoke2`（04:16～04:26）：`V8_EVAL_SMOKE=PASS infra_errors=0 identity_errors=0 demo_frames_min=66 exec_steps_max<=1600 smvla_hard_bound=102`。逐局：smvla VideoUnmask success 270 步 38 s、SwingXtimes fail 527 步 76 s；mme VideoUnmask fail 108 步 92 s、SwingXtimes fail 456 步 31 s；四局 `reset_calls=2`（build+reset，与预算「每回合 2 次」一致）、`recorder_verify=PASS`、`TOKENIZER_SHA=PASS`、`MME_PREFLIGHT=PASS`、`SERVER_CONFIG=PASS seed=7`、`SEAT_REC_SYNC=PASS n=4 bytes=215006377 left=0`。

## 七、与计划的出入

1. 运行名改为 `v8-two-policy-gl10-20261002-01`（计划建议名带 20261001，实际起跑日为 10-02）。
2. GL 工作副本新建 `robomme_benchmark-v8two`，未用计划写的 `robomme_benchmark-newtask-gl`（后者停在 `newtaskRelease-v5`、无 mme-vla 子模块，非本轮分支）。
3. 计划写的 MME 候选路径 `v75eval/ckpt/...` 已不存在，改用 `eval-out/mmevla-ckpt/...`（V8 阶段 3′ 所用、v7.5eval 资产锁 `mme-gl` 条目）并经硬链接真实目录引用。
4. 视频体量：录像器为 FFV1 无损双相机 + 逐步数组，冒烟 4 局 215 MB（约 54 MB／局），计划 §6「约 5 GB」是按 v7 MP4 估的，本轮预计约 100～250 GB；本机 /data 余 3.5 TB。
5. E-C（汇总与视频搬运）两轮合并前审查仍 FAIL（常驻搬运在 stop-file 后遇持续失败目录不退出；非 infra 错误终局无录像时视频闸门与报告口径不一致），按计划执行模式规则交用户裁决，未合入；正式评估不依赖 E-C，视频在 NFS 运行根暂存、不搬不删。
6. 非 infra 错误（如 reset 失败、环境自报 error）由主会话裁定为该身份最终结局（写 accept、计入 error 列、占分母），计划原文只写「错误单列」。
