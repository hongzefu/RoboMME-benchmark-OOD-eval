# 第 4 步：策略本身稳不稳定（开环回放 · 确定性标志 · 编译缓存 · 新旧接口开环）

> 对应方案 §3.2「第 4 步」与确定性标志默认值规则（跑前写死：开态同卡逐位一致、且单次推理耗时增加不超过 10% 才默认开）。启动提交：12.271 `30257b46`（`scripts/eval-official/policy_replay.py`、`run_policy_replay.sh`；本机 worktree `artifacts/v7.5eval/wt/30257b46`，GL `v75eval/wt/30257b46`）；输入 B-mme 重建与新旧接口开环为 12.273 `bac285d8`。0 局、不建环境、不占 reset。用户原话：本轮原话 7（P2 并入 P1、P6 并入 P5）、8；方案原话 16、17、20、23。

## 1. 结论

- **SimpleMemVLA 在确定性标志关的状态下已逐位可复现**：四个条件（本机卡 0、本机卡 1、A40-乙、A40-丁）× 三种模式（同一 server 重放、每次重启 server、A→B→A／B→A→B 常驻顺序）共 16 次比较全部逐位相同，`max_abs=0`。开态（`torch.use_deterministic_algorithms(True)`）**起不来**：SimpleMemVLA 的 Qwen3.5 层里有一个 `cumsum` 算子，PyTorch 没有它的确定性实现，server 在就绪前退出（`SERVER_DIED_BEFORE_READY policy=smvla`）。按跑前写死的规则判 `default=off`；不改用 `warn_only` 去凑「开态」，因为那会放过不确定算子，回答不了规则要问的问题。
- **MME 关态：同一 server 内逐位可复现，跨 server 启动不可复现**：同一 server 重放 4/4 逐位相同；每次重启 server 后 0～2/5 逐位相同，自第 0 次推理起就有差，最大绝对差 0.0034～0.0096，全部低于 v6 对拍容差线 `action_max=0.0413`（`above_action_max=0`）。编译缓存三态（关、冷、重启后命中）1/3 逐位相同，说明差异来自每次进程启动的编译／自动调优，缓存命中也消不掉。
- **MME 开态（`XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`）16/16 逐位一致，但稳态推理减速 46.8%～52.2%**（本机约 82 → 125 ms，A40 约 141 → 208 ms），超过 10% → 按规则 `default=off`。第 5 步两策略都按关态起跑，与官方历史成绩的运行条件一致。
- **随机状态恢复**：每次预热后核对 MME `reset()`、SimpleMemVLA 重设种子，`rng_restored_all=yes`（MME 38 次、SMVLA 18 次运行全部恢复）。
- **新旧接口开环**：同一份官方录制（重跑一片 9 首局 BinFill ep31 seed 543100）分别走旧官方打包与新接口打包，两策略模型输入字节全等、执行动作全等（SMVLA 630/630 步、MME 575/575 步）。

## 2. 判定行原文

```text
POLICY_REPLAY=INFO cond=P1 policy=smvla det=off mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=smvla det=off mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=smvla det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P1 policy=smvla det_bitwise=no slowdown_pct=n/a default=off recheck=same rng_restored_all=yes gpu_busy_events=21
POLICY_REPLAY=INFO cond=P1 policy=mme det=off mode=ABA n=7 bitwise=2/7 max_abs=0.004 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=mme det=off mode=cache n=3 bitwise=1/3 max_abs=0.0037 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=mme det=off mode=restart n=5 bitwise=2/5 max_abs=0.0039 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=mme det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=mme det=on mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=mme det=on mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P1 policy=mme det=on mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P1 policy=mme det_bitwise=yes slowdown_pct=46.8304 default=off recheck=same rng_restored_all=yes gpu_busy_events=44
POLICY_REPLAY=INFO cond=P3 policy=smvla det=off mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=smvla det=off mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=smvla det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P3 policy=smvla det_bitwise=no slowdown_pct=n/a default=off recheck=same rng_restored_all=yes gpu_busy_events=0
POLICY_REPLAY=INFO cond=P3 policy=mme det=off mode=ABA n=7 bitwise=2/7 max_abs=0.0057 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=mme det=off mode=cache n=3 bitwise=1/3 max_abs=0.0057 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=mme det=off mode=restart n=5 bitwise=0/5 max_abs=0.0057 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=mme det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=mme det=on mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=mme det=on mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P3 policy=mme det=on mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P3 policy=mme det_bitwise=yes slowdown_pct=52.1708 default=off recheck=same rng_restored_all=yes gpu_busy_events=0
POLICY_REPLAY=INFO cond=P5 policy=smvla det=off mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=smvla det=off mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=smvla det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P5 policy=smvla det_bitwise=no slowdown_pct=n/a default=off recheck=same rng_restored_all=yes gpu_busy_events=0
POLICY_REPLAY=INFO cond=P5 policy=mme det=off mode=ABA n=7 bitwise=2/7 max_abs=0.0091 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=mme det=off mode=cache n=3 bitwise=1/3 max_abs=0.0045 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=mme det=off mode=restart n=5 bitwise=0/5 max_abs=0.0096 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=mme det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=mme det=on mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=mme det=on mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P5 policy=mme det=on mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P5 policy=mme det_bitwise=yes slowdown_pct=47.5348 default=off recheck=same rng_restored_all=yes gpu_busy_events=0
POLICY_REPLAY=INFO cond=P7 policy=smvla det=off mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=smvla det=off mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=smvla det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P7 policy=smvla det_bitwise=no slowdown_pct=n/a default=off recheck=same rng_restored_all=yes gpu_busy_events=0
POLICY_REPLAY=INFO cond=P7 policy=mme det=off mode=ABA n=7 bitwise=2/7 max_abs=0.0055 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=mme det=off mode=cache n=3 bitwise=1/3 max_abs=0.0034 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=mme det=off mode=restart n=5 bitwise=0/5 max_abs=0.005 first_diff_step_min=0 above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=mme det=off mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=mme det=on mode=ABA n=7 bitwise=7/7 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=mme det=on mode=restart n=5 bitwise=5/5 max_abs=0 first_diff_step_min=n/a above_action_max=0
POLICY_REPLAY=INFO cond=P7 policy=mme det=on mode=same n=4 bitwise=4/4 max_abs=0 first_diff_step_min=n/a above_action_max=0
DET_RULE=INFO cond=P7 policy=mme det_bitwise=yes slowdown_pct=47.9266 default=off recheck=same rng_restored_all=yes gpu_busy_events=0
IFACE_OPEN=INFO policy=mme file=o1-mme.json payload_equal=yes payload_mismatch=0 wire_mismatch=0 action_diff=0(by_payload) exec_equal=yes
IFACE_OPEN=INFO policy=smvla file=o1-smvla.json payload_equal=yes payload_mismatch=0 wire_mismatch=n/a action_diff=0(by_payload) exec_equal=yes
BUILD_INPUTS=PASS policy=mme kind=official messages=73 infers=36 mismatch=0 exec_equal=True
RNG_RESTORE=INFO policy=mme all=yes runs=38 smvla_warmup_all=None gpu_busy_events=0
RNG_RESTORE=INFO policy=smvla all=yes runs=18 smvla_warmup_all=True gpu_busy_events=0
SERVER_DIED_BEFORE_READY policy=smvla name=same
```

`BUILD_INPUTS` 取自 12.273 commit body；`RNG_RESTORE`、`SERVER_DIED_BEFORE_READY` 取自本机回放日志（`artifacts/v7.5eval/logs/chain-card0.log`、`chain-card1.log`、`replay-P3-smvla.log`）；其余取自 `verdicts.txt`。逐次比较的报告：`artifacts/v7.5eval/replay/<P1|P3>/<策略>/report.json`，GL 的 P5、P7 在 `artifacts/v7.5eval/nfs-archive/replay/<P5|P7>/<策略>/report.json`；新旧接口开环 `artifacts/v7.5eval/replay/iface/o1-{mme,smvla}.json`。

## 3. 读法与设计要点

- **条件**：P1 本机卡 0、P3 本机卡 1、P5 A40-乙、P7 A40-丁。方案原有的 P2（卡 0 常驻倒序）并入 P1 的 ABA／BAB 模式，P6（乙同卡复跑）并入 P5 的 restart 模式（本轮原话 7「同意合并 尽可能并行」），所以只有四个条件。
- **模式与次数**：`same` = 同一 server 重放；`restart` = 每次重启 server 重放；`ABA` = A→B→A、B→A→B 与各自独立启动比；`cache` 只对 MME：编译缓存关、冷、重启后命中三态（热态以缓存目录文件数不变且日志命中核实）。判定行里的 `n` 是该模式下的两两比较次数（两个输入合计），每次比较的构成见对应 `report.json`。
- **输入**：A = 第 3 步那一局（PickXtimes ep3 seed 510300，新接口录制）；B = 官方重跑一片 9 首局 BinFill ep31 seed 543100（官方无损录制，经新接口客户端函数重新打包；MME 该局失败 575 步）。B-mme 首次重建把整片代理根目录（预检 ＋ 19 局共 20 条连接）当单连接处理而失败（`BUILD_INPUTS=FAIL … mismatch=70`），12.273 改为按「本局客户端发送 sha 序列逐条全等」挑连接后通过（[incidents.md](incidents.md) 事故 2）。
- **测速可信度**：P1 的 `gpu_busy_events=21／44` 表示回放期间卡 0 上检测到同卡的其他 GPU 进程（`WARN_GPU_BUSY`，不阻塞，来源未逐个归因）；P1 的减速百分比因此只作参考，结论以 P3／P5／P7（`gpu_busy_events=0`）为准，三者均 47.5%～52.2%。
- **为什么 MME 同一 server 能逐位、跨 server 不能**：JAX 在进程启动时编译并自动调优，选到的核函数实现可能不同；进程内不再重选，所以同一 server 重放逐位相同。确定性标志关掉自动调优并强制确定性算子后逐位一致，代价是约五成推理时间。这解释了 MME 闭环同卡重跑也会翻（12.248 实测 11 局翻 5 局；本轮第 3 步同身份历次 892 成功／688 失败／768 成功／775 失败）。

## 4. 命令原文与会话

```bash
# 本机（artifacts/v7.5eval/lanes/replay-local2.sh <gpu> <cpus> <cond...>）：卡 0 接力 v75-chain0 内跑 P1；卡 1 v75-replay-card1 跑 P3 SMVLA，v75-chain1 接跑 P3 MME
bash scripts/eval-official/run_policy_replay.sh $COND $GPU $P $IN/A-$P $IN/B-$P $R/artifacts/v7.5eval/replay/$COND/$P
# GL（main 编排器派发到乙、丁）：v75eval/lanes/replay-gl.sh <P5|P7>
bash $N/lanes/step.sh $N/reports/P5-smvla.json env POLICIES=smvla bash $N/lanes/replay-gl.sh P5
# 输入 B-mme（artifacts/v7.5eval/lanes/build-bmme.sh；12.273 修复后重建）
$R/.venv/bin/python scripts/eval-official/policy_replay.py build-inputs --policy mme --rec $D/rec/BinFill_31_543100 --proxy-rec $D/rec/proxy --kind official --out $R/artifacts/v7.5eval/replay/inputs/B-mme.tmp
# 新旧接口开环（artifacts/v7.5eval/lanes/iface-open-o1.sh，worktree bac285d8）
$R/.venv/bin/python scripts/eval-official/policy_replay.py iface-open --policy smvla --rec-official $S/rec/BinFill_31_543100 \
  --old-src /nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA-official-xhard0/robomme_sim/robomme_env.py --out $R/artifacts/v7.5eval/replay/iface/o1-smvla.json
$R/.venv/bin/python scripts/eval-official/policy_replay.py iface-open --policy mme --rec-official $M/rec/BinFill_31_543100 --proxy-rec $M/rec/proxy \
  --old-src /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning-official-xhard0/examples/robomme --out $R/artifacts/v7.5eval/replay/iface/o1-mme.json
```

`$R`＝主仓库根，`$IN`＝`$R/artifacts/v7.5eval/replay/inputs`，`$N`＝NFS `v75eval`，`$S`／`$M`＝重跑一片 9 的 SMVLA／MME 录制根（`lanes/o1root.sh` 解析）。第 5 步各格的 `--det` 由 `scripts/eval-official/v75-lanes/gl/det_of.sh` 读对应条件 `report.json` 的 `DET_RULE` 得出，全部为 `off`。

## 5. 未覆盖

- 方案要求 SMVLA「确定性开」一遍：因算子缺实现无法运行，开态比较数为 0（如实记录，不是遗漏）。
- 回放只用了 A、B 两个输入局；动作差的分布只代表这两局。
