# V9 双模型评估运行档案：launch（起跑时写定）

计划：`docs/plans/1002-newtask-v9-movecube-region-800-plan.md` 第一部分 §1 第 7、8 条与第二部分 §2.2 第 7 条、§2.4.2 第 7 步。本轮只评 V9 新生成的 80 局（MoveCube xhard4 50 + InsertPeg xhard4 30），与 V8 逐字节相同的 720 局复用 V8 双模型评估（`docs/validation/v8-two-policy-gl10-20261002-01/`）。结果见同目录 `result.md`。

## 一、用户授权口径（2026-10-02）

- 「1002-newtask-v9-movecube-region-800-plan.md 开始实施 有问题问用户但不要阻塞 应该可以执行到底 所有预算job都批准」
- 计划 §2.8：「生成完了和V8一样要做eval两个model都需要」「J0B还是和之前一样生成的时候用四个。Evaluate的时候用十个。」「2同意各类预算在上浮十倍的基础上。都可以自行决策不用来问我」
- 评估口径与 V8 完全一致：SimpleMemVLA `group_size=1`、MME-VLA `perceptual-framesamp-modul/79999`、执行段 1600 步严格截断计 timeout、第 1600 步成功记成功、tokenizer 以 GCS 校验和为可信源、权重按 v7.5eval 资产锁核对。

## 二、代码与环境

| 项 | 值 |
|---|---|
| 评估代码提交 | `820ca142`（12.333，V9 换包提交；评估链路代码 S1-F 合入于 12.329） |
| GL 执行副本 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_benchmark-v9two`（本机 `git clone --no-hardlinks`，`origin` 指回本机；建于 `b462e358`，评估前 fetch + `merge --ff-only` 到 `820ca142`；`seat.sh` 开头核 HEAD 与工作区干净） |
| 子模块 | `third_party/mme-vla` = `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，`third_party/SimpleMemVLA` = `c564c17d276d7294200122b286c21901a3bfb99f`（与 V8 评估相同） |
| bench venv | `<副本>/.venv`，NFS uv-python 3.11.14，`uv sync --frozen --group eval-client` |
| MME venv | `<副本>/third_party/mme-vla/.venv`（3.11.14，`uv sync --frozen`） |
| SimpleMemVLA venv | `<副本>/artifacts/v8-two/venvs/smvla-env`（3.10.19，`scripts/eval-official/smvla-env` 的 `uv sync --frozen`；路径由 `run_v8_gl.sh` 写死） |
| 建法脚本与日志 | 本机 job 临时脚本 `v9_evsetup.sh`（照 V8 launch.md §二复原，V8 的 `v8two-scripts/setup_clone.sh` 已随 NFS 清理删除），日志 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/gl-setup-logs/v9-evsetup.log`，EXIT_CODE=0 |

## 三、输入与资产锁

- 身份源：`artifacts/v9-evaluation/inputs/eval-identities-992.jsonl`（`export_eval_identities.py --specs-root artifacts/newtask-v9/specs-root --delivery artifacts/newtask-v9/delivery/delivery.local.json --official-out artifacts/v9-evaluation/inputs/eval-official-xhard0-192.jsonl`），`EVAL_IDENTITY_EXPORT=PASS version=v9 episodes=992 expected=992 xhard0=192 cells=43 cell_mismatch=0 delivery_mismatch=0`。
- 执行清单：`v8_manifest.py --identities … --delivery … --exclude-evaluated artifacts/v8-evaluation/v8-two-policy-gl10-20261002-01/manifest/manifest.json --shards 10 --out-dir artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/manifest`，`V9_EVAL_SHARDS=PASS shards=10 total=80 cells=2 reused=720 reused_sha256=a476b25b7927 missing=0 extra=0 duplicate=0 xhard0=0`，十片各 8 局；复用集合 `reused.json` sha256 `a476b25b792799f3e6d484be74462bed8bb95e330fea004956522f8dcb6d9b66`（count 720，V8 manifest sha256 `8b5810b344e25c9cce21cbfcb88c5a65e20e6613582401de9bf6addeb3671408`）。清单副本在 NFS `<运行根>/inputs/`，逐文件 `cmp` 一致（`INPUTS_COPY=PASS files=12`）。
- MME 权重：`<运行根>/ckpt/mme/perceptual-framesamp-modul/` 真实目录，`history_config.txt`（`perceptual-framesamp-modul.yaml`）+ `79999/` 以 `cp -al` 硬链接自 `eval-out/mmevla-ckpt/...` 的链接目标。按 `artifacts/v7.5eval/assets-lock.json` 逐文件 sha256：`ASSET mme-gl files=18 locked=18 missing=0 extra=0 mismatch=0`。
- SimpleMemVLA 权重：`SimpleMemVLA/checkpoints/simplememvla_robomme`，`ASSET smvla locked=9 missing=0 mismatch=0`，另 24 个锁外 HuggingFace 簿记文件（与 V8 相同，不参与加载）。判定 `V9_EVAL_ASSETS=PASS assets=2`（日志 `artifacts/v9-evaluation/logs/verify_assets.log`；首跑脚本键名未去 `./` 前缀误报 missing，修正后重跑）。
- tokenizer：本机 `artifacts/v8-evaluation/openpi-data/big_vision/paligemma_tokenizer.model` 拷到 `<运行根>/openpi-data/`，`TOKENIZER_SHA=PASS sha256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。

## 四、席位（清单 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/gl-hold-logs/hold-jobs-v9ev-20261002.txt`）

v9ev-hold-00～09 = 63117915、63117927、63117928、63117929、63117930、63117931、63117932、63117933、63117934、63117935，各 1 A40／4 CPU／48 GiB／48 h，`chaijy2/spgpu`，`--gpu_cmode=shared`，`--wrap='sleep infinity'`。阶段 3 期间（16:59）提前提交；17:55 正式起跑时 00～07 RUNNING，08、09 因 `AssocGrpCpuLimit` 排队。

## 五、命令

运行根 `R=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v9two-out/v9-two-policy-gl10-20261002-01`，脚本目录 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/v9two-scripts/`。

- 单席 `seat.sh <子目录> <NN> <shard json> <JobID> <reset额度> <infra额度>`：核副本 HEAD = `820ca142` 且干净，清 `SLURM_*` 后
  `srun --jobid=<JobID> --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared bash <副本>/scripts/eval-official/run_v8_gl.sh --run-name v9-two-policy-gl10-20261002-01 --seat NN --repo <副本> --stage $R/<子目录> --shard <json> --policies smvla,mme --mme-ckpt $R/ckpt/mme/perceptual-framesamp-modul/79999 --smvla-ckpt <NFS>/SimpleMemVLA/checkpoints/simplememvla_robomme --openpi-data-home $R/openpi-data --tokenizer-sha256 8986bb4f…8fc6 --reset-budget <n> --infra-retry-budget <n>`，输出 tee 到 `$R/<子目录>/logs/seat-NN.log`，尾行 `EXIT_CODE=`。
- `launch_seat.sh`：GL 登录节点 tmux 内，作业未 RUNNING 时每 60 s 查一次，RUNNING 后调 `seat.sh`。
- 正式：GL 登录节点 tmux `ev-v9-v9-two-policy-gl10-20261002-01-s00`～`s09`，子目录 `run`，`--shard $R/inputs/shard-NN.json`，reset 额度 64（8 局 × 2 模型 × 2 次的 2 倍）、infra 重试 1。
- 视频搬运：本机 tmux `v9-vmove`，`eval_video_mover.py --mode v8 --stage $R/run --dest artifacts/v9-evaluation/v9-two-policy-gl10-20261002-01/videos --interval 120 --stop-file <...>/vmove.stop`，日志 `artifacts/v9-evaluation/logs/video_mover.log`；评估结束后 `--once` 全量对账。

## 六、冒烟（子目录 `smoke`，不计入正式分母）

身份 MoveCube_xhard4_23400000（V9 新区域），两模型各 1 局，席位 00（63117915），reset 额度 8、infra 1，GL tmux `ev-v9-smoke-s00`（17:4x～17:55）。

- SimpleMemVLA：`timeout`，`exec_steps=1600`（严格截断生效），`reset_calls=2`。
- MME-VLA：`fail`，`exec_steps=187`，`reset_calls=2`；`TOKENIZER_SHA=PASS`、`MME_PREFLIGHT=PASS commit=ecf086c3`、`SERVER_CONFIG=PASS seed=7`。
- `SEAT_REC_SYNC=PASS n=2 bytes=349977073 left=0`，`V8_SEAT_DONE seat=00 outcome=pass rc=0`；两局 `spec_sha256` 与清单一致。
- 判定：`V9_EVAL_SMOKE=PASS infra_errors=0 identity_errors=0 exec_steps_max<=1600`。

## 七、本轮 tmux 会话清单

| 会话名 | 位置 | 用途 |
|---|---|---|
| `v9-evsetup` | sled-vail | 建 GL 执行副本与三套 venv（已结束） |
| `v9-evroot`、`v9-assets` | sled-vail | 运行根、tokenizer、资产锁核对（已结束） |
| `ev-v9-smoke-s00` | GL 登录节点 | 冒烟（已结束） |
| `ev-v9-v9-two-policy-gl10-20261002-01-s00`～`s09` | GL 登录节点 | 十席正式评估 |
| `v9-vmove` | sled-vail | 视频搬运常驻 |

## 八、与计划的出入

1. 计划 §2.4.2 第 7 步写的 `v8_report.py --run` 与实际接口不符，出报告按 `--manifest／--stage／--videos／--out` 加 `--reuse／--reuse-manifest`（S1-F 交回报告）。
2. 评估席在阶段 3 期间提前提交排队（计划写 4b 前提交），以减轻 V8 时后三席排不上的风险；未增加席位数。
3. V8 的 `v8two-scripts/`（`setup_clone.sh`、`seat.sh`、`launch_official.sh`、`progress.py`）与 NFS 运行根已在 V8 收尾时删除，V9 按 V8 launch.md 的记载重写 `seat.sh`／`launch_seat.sh` 与资产核对脚本。
