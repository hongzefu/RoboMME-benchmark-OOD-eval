# 旧名与官方名对照（2026-10-06 起）

用户 2026-10-06 裁决「照官方；改活代码 + 现行文档」。活代码与现行文档自 R1 改名（`sub/R1:` 三个提交，合入见 `git log --grep 'sub/R1'`）起只用官方名；`docs/validation/`、`docs/plans/` 下的历史留档、指向磁盘／NFS 真实目录的字符串、已发布站点数据键保持原样，读这些历史材料时按本表对照。读历史逐局行、账本、站点目录的工具在读入时自动映射，唯一一份别名表在 `scripts/eval-official/official_defs.py` 的 `LEGACY_NAMES` 段（`LEGACY_POLICY_ALIASES`、`LEGACY_DATASET_ALIASES`、`LEGACY_MODULE_ALIASES`、`LEGACY_CONFIG_KEY_ALIASES`）；CLI 只接受新名。

## 模型与路线

| 旧名（历史留档里的写法） | 官方展示名 | 新的代码／参数 ID | Python 标识符／文件名 |
|---|---|---|---|
| `mme`、`mmevla`、「MME」 | FrameSamp+Modulation | 策略标签 `perceptual-framesamp-modul`；route `perceptual-framesamp-modul/{new,orig}` | `framesamp_modul`（`framesamp_modul_client.py`） |
| `mmesg`、`mmesg-<variant>` | GroundSG+Oracle、GroundSG+QwenVL | 策略标签 `groundsg`；输出 label `groundsg-<variant>`；route `groundsg/<variant>/{new,orig}`；变体值 `ground-sg-oracle`／`ground-sg-qwenvl` 不变 | `groundsg`（`groundsg_client.py`） |
| （无） | MemER | 变体 `ground-sg-memer`（第三阶段新增） | 复用 `groundsg` 装配 |
| Astra | 3-tier Astra | `astra` 不变（文件名、标签、账本键、route 都不动） | `astra` |
| MME-VLA（家族名） | 不变，官方名 | `third_party/mme-vla`、`MMEVLAWebsocketClientPolicy`、`mme_vla_suite` 不动 | — |

## 数据集接口

| 旧名 | 新名 | 步数配对 |
|---|---|---|
| `test-hard0`（第二阶段，官方 hard 12 局，代码里 xhard0） | `hard-verify` | 1300（不变） |
| `test-hard`（第三阶段 V9） | `ood` | 改名时保持 1600；第三阶段功能合入后为 1800 |

目录 `src/robomme_hard/env_metadata/test-hard/` 已改为 `env_metadata/ood/`（五个 `specs.jsonl` 字节不变，`tests/contract/packaged_specs.sha256` 钉值不变）。环境变量 `ROBOMME_HARD_XHARD0_IN_TEST_HARD` 与常量 `XHARD0_IN_TEST_HARD` 保持原名（与冻结配置 `scripts/configs/gate-set-v9-129.json` 互相引用）。

## 文件、参数与环境变量

| 旧 | 新 |
|---|---|
| `scripts/eval-official/mme_client.py` | `framesamp_modul_client.py` |
| `scripts/eval-official/mmesg_client.py` | `groundsg_client.py` |
| `scripts/eval-official/orig-mme-client-env/` | `orig-framesamp-modul-client-env/` |
| `orig_observer/{mme_client_wrap,mme_proxy}.py` | `orig_observer/{framesamp_modul_client_wrap,framesamp_modul_proxy}.py` |
| `orig_observer/run_orig_mme.sh` | `orig_observer/run_orig_framesamp_modul.sh` |
| `tests/pipeline/eval/test_mme_transport.py` | `test_framesamp_modul_transport.py` |
| `--mme-variant` | `--groundsg-variant` |
| `--mme-ckpt` ／ `--mmesg-ckpt` | `--framesamp-modul-ckpt` ／ `--groundsg-ckpt` |
| `--episode-wall-mme` | `--episode-wall-framesamp-modul` |
| `MME_PY` ／ `MME_COMMIT` | `MME_VLA_PY` ／ `MME_VLA_COMMIT`（指 MME-VLA 子模块，用家族名） |
| `MME_CKPT` ／ `MMESG_CKPT` ／ `MME_VARIANT` | `FRAMESAMP_MODUL_CKPT` ／ `GROUNDSG_CKPT` ／ `GROUNDSG_VARIANT` |
| `ORIG_MME_CLIENT_PY` | `ORIG_FRAMESAMP_MODUL_CLIENT_PY` |
| 结果行／配置键 `mme_variant` | `groundsg_variant` |
| `cap_probe.py --loop mme` | `--loop mme-vla`（测的是两模型共用的官方 `eval.py` 循环） |
| 日志标记 `MMEVLA_ORIG_XHARD0_OBSERVED` | `FRAMESAMP_MODUL_ORIG_XHARD0_OBSERVED` |
| 评估包 `robomme_hard_eval`（发行名 `robomme-hard-eval`，目录 `src/robomme_hard_eval/`） | `robomme_ood_eval`（`robomme-ood-eval`，`src/robomme_ood_eval/`；2026-10-09 版本 1.22 改名，不留兼容层） |

## 保留原样的历史名（代码里以「历史目录名」或「历史数据键」行标注）

- 真实目录：NFS／本机 `sg-eval/ckpt/mme/…`、`mmevla-ckpt/…/79999`、`mmevla-testhard*`、`mmevla-official-xhard0`、E0 校验文件里的 `mme-official-full-s<i>/`、`/data/…/official-mme-vla/perceptual-framesamp-modul/79999`。
- 已发布数据键：V8 站点 `catalog.json` 的页面 ID `mmevla`（页面展示名已改为 FrameSamp+Modulation）；`scripts/parity/gate_set.py::RULE` 与冻结配置逐字节相同，其中的 `test-hard` 字样保留。
