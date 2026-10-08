# 第 0 步：开跑前核实（资产锁 · 官方历史成绩冻结 · 身份冻结 · 环境栈差异）

> 对应方案第一部分 §3.2「第 0 步」与判定行总表 `ASSETS`／`E0_FREEZE`／`IDENTITY_FREEZE`／`PREFLIGHT`（运行阻塞类）。启动提交：12.267 `2176a0e3`（冻结工具 `scripts/eval-official/compare.py` 与第 0 步判定同提交落地）；资产哈希脚本 `artifacts/v7.5eval/preflight/hash.sh`、`hash_assets.sh`。用户原话：本轮原话 1、2；方案原话 23（d1a）、26。

## 1. 结论

资产锁、官方历史成绩冻结、身份冻结三项运行阻塞判定全部 PASS，开跑前后 E0 文件未被改动（每个官方分片起止各核一次）。方案要求的合并判定行 `PREFLIGHT=PASS seats=5 data_free_gib=<n> slack=<ok|off>` **没有单独生成**，席位与空间核对分散在下文，属偏离，记入 [summary.md](summary.md)。

## 2. 判定行原文

```text
ASSETS=PASS assets=45 mismatches=0
E0_FREEZE=PASS files=440
IDENTITY_FREEZE=PASS small=48 full=192 shuffle_sha256=bd15cc49c8aaa5b69f2414bd4115cd5b0424f82fba55f65b8b64e59c4444e56a
E0_SELF=INFO policy=smvla compared=192 missing=0 s2f=0 f2s=0 sr_pct=73.4375 success=141 files=10 lines=192 superseded=0 sanity=ok
E0_SELF=INFO policy=mme compared=192 missing=0 s2f=0 f2s=0 sr_pct=26.0417 success=50 files=10 lines=192 superseded=0 sanity=ok
E0_FREEZE_CHECK=PASS when=before lines=3
E0_FREEZE_CHECK=PASS when=after lines=3
MME_PREFLIGHT=PASS commit=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b history_config=perceptual-framesamp-modul.yaml ckpt=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999 py=/data/hongzefu/robomme_benchmark_MotionJEPANewTask/third_party/mme-vla/.venv/bin/python
```

前三行取自 12.267 commit body（当时终端输出，未落单独文件）；`E0_SELF` 取自 `artifacts/v7.5eval/summary/verdicts.txt`；`E0_FREEZE_CHECK` 由 `official_rerun_shard.sh` 在每个官方分片起止对 NFS `v75eval/e0-sha.txt`（清单 ＋ 20 个 E0 逐片文件）做 `sha256 -c`，上面两行取自 `artifacts/v7.5eval/nfs-archive/state/official/logs/O1-s5.log`，其余分片日志同形；`MME_PREFLIGHT` 是 `run_seat.sh` 每次起 MME server 前的断言，本行取自本机第 3 步日志 `artifacts/v7.5eval/logs/step3-card1.log`。

## 3. 各项核实内容

1. **资产锁**（`artifacts/v7.5eval/assets-lock.json`）：MME 权重本机与 GL 各 18 个文件逐一 sha256 相同（`artifacts/v7.5eval/logs/assets.log` 末行 `MME_LOCAL_VS_GL=EQUAL`、`EXIT_CODE=0`），SimpleMemVLA 9 个文件；HF 40 位 revision：MME 本机 `c0f565dd…`、SimpleMemVLA `6bf72043…`，GL 副本「revision=未解析」，以本地 sha256 为锁。GL 上 MME 权重是符号链接，`run_seat.sh` 以 `$MME_CKPT/..` 读 `history_config` 会解析到错误父目录，于是复制真实目录到 `v75eval/ckpt/mme`，18 个文件 sha256 与资产锁一致（见 [incidents.md](incidents.md) 事故 4）。
2. **官方历史成绩冻结**（`artifacts/v7.5eval/input-manifest.json`）：440 个文件、567,010,858 字节；`compare.py freeze --check` 复核 `changed=0`。E0 自比两策略 0 翻转；数据分布 MME 成功／失败／超时 = 50／131／11，SimpleMemVLA 141／36／15，与方案 §1.2 一致。
3. **身份冻结**：小样本 `artifacts/v7.5eval/identities-small48.json`（16 任务 × 1 档 × 3 局 = 48，按 `xhard0_manifest.json` 每任务原 episode 升序取前 3 个）；全量 `identities-full192.json`（16 任务 × 1 档 × 12 局 = 192，即官方历史成绩身份）；正式跑法队列打乱种子 `shuffle_seed=20260930`，结果 `queue-order-full192.json`，sha256 见上方判定行。
4. **官方源码对应**：E0 MME 客户端实际用的官方 `robomme` `856bc3a1` 与锚点 `1fadc0ec` 的 `src/robomme` 逐字节相同（`diff -rq` rc=0）；SimpleMemVLA 内置的 `robomme_sim/robomme` 与之相比只少一个杂散文件「vqa_options copy.py」（12.266 核实）。`scripts/parity/upstream_guard.py check --require-upstream` → `UPSTREAM_GUARD=PASS`。
5. **新旧环境栈差异**（方案第 0 步要求记录，第 6 步实测见 [env-parity.md](env-parity.md)）：旧 MME 客户端 venv 与旧 SimpleMemVLA venv 均为 sapien 3.0.3（后者环境侧 torch 2.4.1），环境源码为官方 `robomme`；新接口为 benchmark `.venv`（Python 3.11.14、sapien 3.0.2、torch 2.9.1+cu128、numpy 1.26.4）＋ `robomme_hard` xhard0。12.266 的 `uv sync --group eval-client --inexact` 只新增 `dm-tree`、`msgpack`、`websockets`、`tree`、`svgwrite`、`openpi-client`，上述三个核心包版本不变。
6. **旧官方代码可用**：MME 旧客户端 worktree `927c56d` 代表 E0 运行时代码属推断（方案盲区第 1 条，未变）；两套旧 venv 均能 import 旧客户端，`OBSERVER_PREFLIGHT=PASS`（MME 与 SMVLA 各一行，见 `artifacts/v7.5eval/logs/r1-card0.log`）。
7. **席位与空间**：五个既有席位与新增四席的节点、剩余时长见 [README.md](README.md) ③；`/data` 开跑时剩余 4.2 TiB（主会话 01:10 EDT `df -h /data` 输出 `/dev/nvme1n1p1 14T 9.0T 4.2T 69%`；02:50 EDT 搬运器首行 `DATA_FREE 4244G`）；Slack 测试私信 01:5x EDT 发送成功（频道 `D09233N44SF`），但全程未触发存储降级：GL 与本机录制共 807 行 `RECORDER_VERIFY=PASS … level=0`，没有任何 `STORAGE_DEGRADE` 行。Slack 测试私信在规划期 2026-09-30 已通（方案第二部分 §3）；开跑后主会话按方案重发了测试私信，结果未落文件，记为未验证。

## 4. 命令原文

```bash
# 资产哈希（本机 MME、GL MME、SimpleMemVLA；日志 artifacts/v7.5eval/logs/assets.log）
bash artifacts/v7.5eval/preflight/hash_assets.sh
# 冻结与身份（提交 2176a0e3 的 compare.py）
uv run --no-sync python scripts/eval-official/compare.py assets …
uv run --no-sync python scripts/eval-official/compare.py freeze …          # 之后 freeze --check
uv run --no-sync python scripts/eval-official/compare.py identities …
```

`compare.py` 各子命令当时的完整参数没有留原文（只在子代理会话里执行），这里只能给子命令名；输出文件路径与判定行以上文为准，复核用 `compare.py freeze --check`（12.267 实测 `changed=0`）。

## 5. 会话

第 0 步全部在本机前台或短后台完成，未开 tmux 会话。
