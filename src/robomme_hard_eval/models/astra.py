"""3-tier Astra 新侧驱动：把 Astra-on-RoboMME 的单局循环 ``runner.episode`` 接到本仓库的 ``BenchmarkEnvBuilder`` 上。

计划：``1003-oracle-subgoal-groundsg-eval-plan.md`` 第二部分 1.5（S5）。原侧是原样运行 Astra 的
``examples/champ/run.sh``；本文件是新侧，不改 Astra 任何源码，只做三件事：

1. ``sys.path`` 加 Astra 的 ``examples/champ`` 与本仓库 ``src``，``import robomme_hard.robomme_env``，
   以 ``BenchmarkEnvBuilder(task, dataset=<hard-verify|ood>, action_space="joint_angle",
   gui_render=False, max_steps=<启动命令给的值>)`` 建环境；
2. 构造 Astra 的 ``Monitor``、``Planner``、``ResponsesClient`` 与 websocket 客户端后，直接调用
   ``runner.episode(...)``（不经它的 ``main()``：其中按 ``metadata_index`` 取 50 局的断言对本仓库 builder 不成立）；
3. 逐项复刻 ``main()`` 的六项行为（见 ``run_cases`` 文档串），并在环境、VLA、规划、监视四处以委托方式
   调用 ``trace_writer`` 记每局 ``trace.jsonl``。

输出目录与原侧相同：``<根>/group_<0|1>/<RUN>/{results,planner_calls}``；``results/<task>/ep<NNN>/`` 下是
Astra 自己写的 ``identity.json``、``decisions.jsonl``、``actions.npy``、``result.json``、``rollout.mp4``、
``monitor_inputs/``；本驱动在其下另建 ``<key>.a1/``（轨迹、原动作、无损录像，见文末「第二阶段」）并给
``result.json`` 追加映射字段。``group_*`` 这一层必须保留：
``main()`` 每局前查 ``<output>/../../STOP.json``，``ResponsesClient._send`` 每次发送前查路径上名为
``group_0``／``group_1`` 的目录里的 ``STOP.json``——费用守卫 ``astra_cost_guard.py`` 就往那里写。

子命令：

- ``prepare``：生成新侧的局清单（``{"dataset", "cases": [{task, episode, tier, seed[, source_episode]}]}``），
  ``episode`` 是本仓库 builder 的本地局号；身份字段取自 builder，供 ``run`` 起跑前逐局核对。
- ``check``：不加载模型、不联网：核对局清单与身份、数据集与步数的配对、``validate_checkpoints``、端口可绑定，
  并打印 ``robomme_hard.__file__``、``robomme.__file__``。``run_astra.sh`` 在起 VLA 前调用。
- ``run``：正式运行（``run_astra.sh`` 调用）。
- ``summarize``：汇总 ``results`` 为 ``summary.json``（Astra 自带的 ``summarize.py`` 只认 test／val）。

密钥只从环境变量 ``OPENAI_API_KEY`` 读，交给 ``ResponsesClient``；本文件不读密钥文件、不打印密钥。

第二阶段（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节 S3；共享契约见 ``trace_writer.py`` 文档串）：

- 轨迹按 C1～C8：``route="astra/new"``；identity 补 ``builder_episode``（本地局号）、``key``（``<task>_<tier>_<seed>``，
  与 ``env_client.v8_key`` 同式）、``attempt=1``；演示段记全部 reset 帧（含初始帧），收尾写 ``demo_frames``、
  ``steps_attempted``／``steps_observed``／``frames_recorded``；结束原因只取 success／fail／timeout／error
  （不再写 ``env_terminated``／``loop_exit``／``exception``）。
- 目录：Astra 自己的 ``<RUN>/results/<task>/ep<NNN>/`` 照旧，其下新建 ``<key>.a1/``：``trace.jsonl``、
  ``arrays.npz``（每个执行步的原动作 ``exec_action__%05d``）、``media/``（``recorder.EpisodeRecorder`` 的目录，
  首次 reset 时新建且必须为空，不用 ``overwrite``）、收尾时由 ``run_astra.sh`` 生成的 ``official/``。录制器收尾
  核对通过（``RECORDER_VERIFY=PASS``）后，把两路无损流 ``{front,wrist}.mkv`` 与 ``frames-{front,wrist}.jsonl``
  从 ``media/`` 移到 ``<key>.a1/``，使轨迹、原动作、原始帧同在一个目录（C5），重绘与转码直接对 ``<key>.a1/`` 做；
  ``media/`` 留录制器的 ``meta.json``、``events.jsonl``、``summary.json`` 与它自己的 ``arrays.npz``。
  ``result.json`` 追加 ``key``、``attempt``、``episode_dir``、``media_dir``、``exec_steps``（= steps_attempted）等映射。
- 费用硬上限：``run`` 必须给 ``--guard-state``（``astra_cost_guard.py --state`` 的文件）；规划客户端用本文件的
  ``GuardedResponsesClient``（子类化第三方 ``ResponsesClient``，第三方文件零改动），每次真正发送前同步读守卫状态、
  原子预留单次最坏费用，守卫失联（心跳超过 10 秒）、STOP、预留失败任一即拒发；等待间隔结束后再检一次。
  局数硬上限 2：局清单超过 2 局起跑前拒绝，且每局开跑前在守卫预留文件里跨 RUN 登记。
- 子命令 ``media-inputs``：为 ``run_astra.sh`` 收尾时调用的 ``official_media_check.py`` 写本局的身份清单行与
  账本行（Astra 没有 ``env_client`` 账本，``accepted_attempt_id`` 由 key 与尝试号确定）。

第三阶段（1006-rename-official-names-and-stage3-eval-plan.md 第二部分一、八.3、八.11；接口见
``docs/plans/1006-stage3-interface-freeze.md`` 2.1、2.2、四、五节，R5）：

- 1800 步：``ood ↔ 1800``、``strict_cap=1``；``hard-verify ↔ 1300`` 非 strict 不变。strict 时 ``TracedEnv.step`` 在
  计数与动作追加**之前**检查：已执行 ``effective_cap`` 步再调用即不进真实 ``env.step``、置 ``cap_hit``、抛
  ``StepCapReached``（与 ``env_client.EnvSession`` 同义），trace 不多落一行，收尾按 ``timeout``。Astra 自己的循环本来就
  只走 ``max_steps`` 步，此守卫是入口自己的硬上限。``recorder_meta``、trace header（若 ``TraceWriter`` 支持）／end 行与
  ``result.json`` 记实际 ``strict_cap``／``effective_cap``／``cap_hit``。
- 模型 seed：``check``／``run`` 必须给 ``--policy-seed <非负整数>``，缺失即 ``RUN_BLOCKED reason=policy_seed``（退出 3）；
  ``run_astra.sh`` 把它作为 VLA 服务的 ``--seed``（替代上游固定 42）。云端 planner／monitor 无 seed 接口：只在 trace、
  ``result.json``、来源清单里记 ``policy_seed`` 与 ``cloud_seed=null``，不伪造。
- 语言账本（``trace_writer.LanguageLog``，R6 提供；不存在时不记、不报错）：每局 ``<trace 同目录>/language.jsonl``。
  planner（含第二按钮复审）每次请求一个调用：``prompt.txt`` 全文与附图引用在交给发送方**之前**落盘，回复原文、
  ``parsed``（校验后子目标或复审布尔）、``fallback=continue_last``；传输重试（429）每次另开一个调用并记
  ``transport_attempt``。monitor 每次一个调用：system、user（10 张图引用）、回复 ``true/false``，也在推理前落盘。
  VLA 每个推理步一个 ``action_model`` 调用（``prompt``／``grounded_subgoal`` 字段原文），执行步 ``log_step`` 带
  ``source_call_id``／``chunk_index``。``prompts/*.md`` 与 ``index.json`` 的 sha256 写进局目录 ``provenance.json``。
- ``arrays.npz`` 改走 ``trace_writer.merge_write_npz``（R6 提供；不存在时保持旧的 ``np.savez``）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REPO_SRC = REPO_ROOT / "src"

#: 新侧只接受这两个数据集；与启动命令的 ``--max-steps`` 做配对一致性检查（不是按档查表，数值由启动命令给）
DATASET_STEP_PAIRING = {"hard-verify": 1300, "ood": 1800}
#: 冻结说明 2.1：ood 为 strict（第 1801 次 step 在进入真实环境前被拒），hard-verify 不变（截断靠底层环境）
DATASET_STRICT_CAP = {"hard-verify": False, "ood": True}
#: 上游 ``api_client.ResponsesClient`` 请求体里固定的输出上限（规划与复审共用；只用于语言账本 params）
PLANNER_MAX_OUTPUT_TOKENS = 2048
#: 上游 ``runner.Monitor`` 的 ``RequestConfig(max_tokens=8, temperature=0)``（只用于语言账本 params）
MONITOR_PARAMS = {"temperature": 0, "max_tokens": 8}
#: 来源清单 provenance.json 的文件名（局目录 ``<key>.a1/`` 下）
PROVENANCE_FILE = "provenance.json"
#: ``main()`` 第⑤项：错误信息以这三个前缀开头的局，单次即停整个分片
PLANNER_STOP_PREFIXES = ("Planner API", "Pilot planner-call", "Planner bridge failed")
#: ``main()`` 第④项：非规划类 error 连续累计到此数即停
INFRA_ERROR_LIMIT = 3
#: ``ResponsesClient._send`` 只认这两个目录名下的 STOP.json
GROUP_DIR_NAMES = ("group_0", "group_1")
#: 启动时打印 sha256 的 Astra 源文件（留档用）
ASTRA_SOURCE_FILES = ("runner.py", "core.py", "api_client.py", "input_contract.py", "release_utils.py",
                      "train_entry.py", "weights.json")
#: C1 新侧路线名
ROUTE = "astra/new"
#: Astra 每局只跑一次，尝试号恒为 1（C6）
ATTEMPT = 1
#: C3 终态
TERMINALS = ("success", "fail", "timeout", "error")
#: 录制器收尾通过后从 media/ 移到局目录的原始帧文件（C5）
RAW_MEDIA_FILES = ("front.mkv", "wrist.mkv", "frames-front.jsonl", "frames-wrist.jsonl")
#: 守卫心跳超时（秒）：超过即视为守卫失联、拒发（与 astra_cost_guard.HEARTBEAT_TIMEOUT_S 相同，由测试核对）
GUARD_HEARTBEAT_TIMEOUT_S = 10.0
#: 第三方 ResponsesClient 两次发送之间的固定间隔（秒；上游 _send 里的 20）
SEND_INTERVAL_S = 20


class AstraStop(RuntimeError):
    """``main()`` 里那几处 ``raise RuntimeError`` 的同义异常；``reason`` 供测试与日志判读。"""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class StepCapReached(RuntimeError):
    """strict cap：已执行 ``effective_cap`` 步后再调用 ``step``，不进入真实环境（与 ``env_client.StepCapReached`` 同义）。
    继承 ``RuntimeError``：Astra ``runner.episode`` 的 ``except Exception`` 接住它，本驱动收尾时改记 ``timeout``。"""


# ── 路径与上游模块 ─────────────────────────────────────────────────────────

def astra_root(explicit: str | None = None) -> Path:
    """Astra 子模块根：命令行 > ``SGEVAL_THIRD_PARTY``（worktree 测试只读引用主检出）> 本仓库 ``third_party``。"""
    if explicit:
        return Path(explicit).resolve()
    third = os.environ.get("SGEVAL_THIRD_PARTY")
    base = Path(third) if third else REPO_ROOT / "third_party"
    return (base / "Astra-on-RoboMME").resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(root: Path) -> SimpleNamespace:
    """``sys.path`` 加 Astra ``examples/champ`` 与本仓库 ``src``，导入 Astra 的四个模块并返回。"""
    champ = root / "examples" / "champ"
    if not (champ / "runner.py").is_file():
        raise FileNotFoundError(f"找不到 Astra 源码 {champ}/runner.py（子模块未初始化？）")
    for entry in (str(REPO_SRC), str(champ), str(HERE)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    import api_client  # noqa: PLC0415  Astra 的直连 Responses API 客户端
    import core  # noqa: PLC0415
    import release_utils  # noqa: PLC0415
    import runner as astra_runner  # noqa: PLC0415  Astra 的单局循环（模块级 episode 函数）
    return SimpleNamespace(root=root, champ=champ, runner=astra_runner, core=core,
                           release_utils=release_utils, api_client=api_client)


def source_digests(champ: Path) -> dict[str, str]:
    return {name: file_sha256(champ / name) for name in ASTRA_SOURCE_FILES if (champ / name).is_file()}


def assert_env_sources() -> dict[str, str]:
    """环境源必须是本仓库 ``src``：``robomme`` 与 ``robomme_hard`` 都不得来自 Astra 嵌套的官方副本。"""
    import robomme  # noqa: PLC0415
    import robomme_hard  # noqa: PLC0415
    files = {"robomme_hard": str(Path(robomme_hard.__file__).resolve()),
             "robomme": str(Path(robomme.__file__).resolve())}
    for name, path in files.items():
        if not path.startswith(str(REPO_SRC.resolve()) + os.sep):
            raise RuntimeError(f"{name} 不是来自本仓库 src：{path}")
    print(f"ASTRA_ENV_SOURCE robomme_hard={files['robomme_hard']} robomme={files['robomme']}", flush=True)
    return files


def default_builder_cls():
    import robomme_hard.robomme_env  # noqa: F401,PLC0415  注册环境（与官方 main() 的 import robomme.robomme_env 对应）
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder  # noqa: PLC0415
    return BenchmarkEnvBuilder


def make_builder(builder_cls, task: str, dataset: str, max_steps: int):
    return builder_cls(task, dataset=dataset, action_space="joint_angle", gui_render=False, max_steps=max_steps)


# ── 局清单 ────────────────────────────────────────────────────────────────

def check_pairing(dataset: str, max_steps: int) -> None:
    expected = DATASET_STEP_PAIRING.get(dataset)
    if expected is None:
        raise ValueError(f"RUN_BLOCKED reason=dataset dataset={dataset!r}（只接受 {sorted(DATASET_STEP_PAIRING)}）")
    if int(max_steps) != expected:
        raise ValueError(f"RUN_BLOCKED reason=step_cap_pairing dataset={dataset} max_steps={max_steps}（应为 {expected}）")


def check_policy_seed(value) -> int:
    """冻结说明 2.2：``--policy-seed`` 必填、非负整数；缺失或非法即 ``RUN_BLOCKED reason=policy_seed``（不回落旧默认 42）。"""
    if value is None or isinstance(value, bool):
        raise ValueError("RUN_BLOCKED reason=policy_seed 必须显式给 --policy-seed <非负整数>（不回落任何旧默认值）")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.strip().isdigit():
        seed = int(value.strip())
    else:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed policy_seed={value!r} 须为非负整数")
    if seed < 0:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed policy_seed={value!r} 须为非负整数")
    return seed


def validate_cases(document: dict, tasks: list[str]) -> list[dict]:
    """新侧局清单：``dataset`` 为 hard-verify／ood；``episode`` 为本地局号；身份字段必备以供逐局核对。"""
    dataset = document.get("dataset")
    if dataset not in DATASET_STEP_PAIRING:
        raise ValueError(f"局清单 dataset 应为 {sorted(DATASET_STEP_PAIRING)}，实为 {dataset!r}")
    cases = document.get("cases") or []
    if not cases:
        raise ValueError("局清单为空")
    seen = set()
    for case in cases:
        task, episode = case.get("task"), case.get("episode")
        if task not in tasks or type(episode) is not int or episode < 0:
            raise ValueError(f"未知任务或本地局号非法：{case}")
        if (task, episode) in seen:
            raise ValueError(f"重复身份：{case}")
        seen.add((task, episode))
        need = ("tier", "seed", "source_episode") if dataset == "hard-verify" else ("tier", "seed")
        missing = [key for key in need if key not in case]
        if missing:
            raise ValueError(f"局清单缺身份字段 {missing}：{case}")
    return cases


def verify_identity(builder, case: dict) -> dict:
    """起跑前逐局核对：builder 解析出的身份与局清单一致（hard-verify 局号 0 ↔ 官方 test episode 3 即由此保证）。"""
    episode = int(case["episode"])
    if not 0 <= episode < builder.get_episode_num():
        raise ValueError(f"{case['task']} 本地局号 {episode} 超出 {builder.dataset} 的 {builder.get_episode_num()} 局")
    identity = builder.resolve_identity(episode)
    for key in ("tier", "seed", "source_episode"):
        if key in case and identity.get(key) != case[key]:
            raise ValueError(f"身份不符 {case['task']} ep{episode} {key}: builder={identity.get(key)!r} 清单={case[key]!r}")
    return identity


def prepare_cases(builder_cls, dataset: str, tasks: list[str], *, source_episodes: list[int] | None = None,
                  tier: str | None = None, index: int = 0) -> dict:
    """hard-verify：按官方 test 局号（与原侧 ``prepare_cases.py --episodes`` 同义）找本地局号；
    ood：取每任务 ``tier`` 档的第 ``index`` 局（V9 连通局为 VideoUnmask xhard1 第 0 局）。"""
    cases = []
    for task in tasks:
        builder = make_builder(builder_cls, task, dataset, DATASET_STEP_PAIRING[dataset])
        identities = [builder.resolve_identity(ep) for ep in range(builder.get_episode_num())]
        if dataset == "hard-verify":
            for source in source_episodes or []:
                hits = [i for i in identities if i.get("source_episode") == source]
                if len(hits) != 1:
                    raise ValueError(f"{task} 在 hard-verify 里找不到官方 test episode {source}")
                hit = hits[0]
                cases.append({"task": task, "episode": hit["episode"], "tier": hit["tier"], "seed": hit["seed"],
                              "source_episode": hit["source_episode"]})
        else:
            in_tier = [i for i in identities if i["tier"] == tier]
            if index >= len(in_tier):
                raise ValueError(f"{task} 在 ood 的 {tier} 档只有 {len(in_tier)} 局")
            hit = in_tier[index]
            cases.append({"task": task, "episode": hit["episode"], "tier": hit["tier"], "seed": hit["seed"],
                          "candidate": hit.get("candidate"), "spec_sha256": hit.get("spec_sha256")})
    return {"dataset": dataset, "cases": cases}


# ── 轨迹委托（trace_writer） ───────────────────────────────────────────────

def _canonical(obj: Any) -> bytes:
    """规范化字节：数组一律换成 ``array_record``（dtype、shape、sha256），其余按 JSON 排序键序列化。"""
    import numpy as np  # noqa: PLC0415
    import trace_writer as tw  # noqa: PLC0415

    def conv(value):
        if isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")):
            rec = tw.array_record(value)
            return {"dtype": rec["dtype"], "shape": rec["shape"], "sha256": rec["sha256"]}
        if isinstance(value, dict):
            return {str(k): conv(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [conv(v) for v in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    return json.dumps(conv(obj), sort_keys=True, ensure_ascii=False).encode()


class TraceContext:
    """一局的共享状态：当前步号与子目标（VLA 请求里的 ``grounded_subgoal``）、C8 三分计数、原动作、录制器。

    ``open_recorder``：无参可调用，首次 reset 时由 ``TracedEnv`` 调用一次建录制器（``run_one`` 注入）；
    为 ``None`` 时不录（只供只测轨迹的单元测试）。

    第三阶段：``effective_cap``／``strict_cap``（strict 时 ``TracedEnv.step`` 在计数前拒第 cap+1 步，置 ``cap_hit``）；
    ``lang``：``trace_writer.LanguageLog`` 实例或 ``None``（不记语言账本）；``source_call_id``／``chunk_index``：
    当前动作块来自哪次 ``action_model`` 调用、下一步是块内第几个动作；``action_params``／``monitor_params``：
    语言账本 ``open_call`` 的 ``params``。"""

    def __init__(self, writer, open_recorder: Callable | None = None, *, effective_cap: int | None = None,
                 strict_cap: bool = False, lang=None, action_params: dict | None = None,
                 monitor_params: dict | None = None) -> None:
        self.writer = writer
        self.t = 0
        self.subgoal: str | None = None
        self.attempted = 0  # C8 交给环境的步数（含异常步）
        self.observed = 0  # C8 返回有效观测的步数
        self.demo_frames: int | None = None  # 演示帧数（不含初始帧）；None = 还没 reset
        self.actions: list = []  # 每个执行步实际交给环境的动作（原 dtype／shape／bytes）
        self.recorder = None
        self._open_recorder = open_recorder
        self.effective_cap = None if effective_cap is None else int(effective_cap)
        self.strict_cap = bool(strict_cap)
        self.cap_hit = False
        self.lang = lang
        self.source_call_id: str | None = None
        self.chunk_index = 0
        self.action_params = dict(action_params or {})
        self.monitor_params = dict(monitor_params or {})
        self._sha_cache: dict = {}

    def ensure_recorder(self):
        if self.recorder is None and self._open_recorder is not None:
            self.recorder = self._open_recorder()
        return self.recorder

    def frame_sha(self, phase: str, idx: int, frames) -> str | None:
        """Astra ``frames``／``demo`` 列表第 ``idx`` 帧的原像素 sha256（与 trace 的 ``front_sha256`` 同算法），按帧缓存。"""
        import trace_writer as tw  # noqa: PLC0415
        key = (phase, int(idx))
        if key not in self._sha_cache:
            self._sha_cache[key] = tw.image_sha256(frames[idx]) if 0 <= idx < len(frames) else None
        return self._sha_cache[key]

    def step_link(self) -> dict:
        """执行步关联（冻结说明五）：开了语言账本时给 ``log_step`` 的 ``source_call_id``／``chunk_index``，否则空
        （不传时 step 行键集合与旧格式逐字节相同）。每取一次块内序号加一。"""
        if self.lang is None:
            return {}
        link = {"source_call_id": self.source_call_id, "chunk_index": self.chunk_index}
        self.chunk_index += 1
        return link


def _pack_state(obs, i: int = -1):
    import numpy as np  # noqa: PLC0415
    return np.concatenate([np.asarray(obs["joint_state_list"][i]),
                           np.asarray(obs["gripper_state_list"][i])[:1]]).astype(np.float32)


def _goal_text(goal) -> str | None:
    """与 Astra ``runner.episode`` 取任务目标同式：列表取第一个。"""
    if isinstance(goal, list):
        return goal[0] if goal else None
    return goal


class TracedEnv:
    """包住 builder 给的环境：reset 记演示段（C2），step 记执行后的画面、状态、动作与终止标志（C4、C8）。

    录制器只用 ``add_frames``／``add_array``：演示段全部帧（含初始帧）与每个有效观测步的最后一帧各进一次，
    故两路流的帧数 = ``frames_recorded`` = 演示帧数 + 1 + 有效观测步数。

    strict cap（ood）：守卫在计数与动作追加**之前**——已执行 ``effective_cap`` 步再调用即不进真实环境、不记录器、
    不落 trace 行，置 ``cap_hit`` 并抛 ``StepCapReached``。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def reset(self, *a, **k):
        import numpy as np  # noqa: PLC0415
        obs, info = self._inner.reset(*a, **k)
        ctx = self._ctx
        rec = ctx.ensure_recorder()
        fronts = [np.asarray(x, dtype=np.uint8) for x in obs.get("front_rgb_list", [])]
        wrists = [np.asarray(x, dtype=np.uint8) for x in obs.get("wrist_rgb_list", [])]
        n = min(len(obs.get("joint_state_list", [])), len(obs.get("gripper_state_list", [])))
        states = [_pack_state(obs, i) for i in range(n)]
        if rec is not None and fronts:
            rec.add_frames("front", np.stack(fronts), tag="reset")
            rec.add_frames("wrist", np.stack(wrists), tag="reset")
            for key in ("joint_state_list", "gripper_state_list"):
                if obs.get(key) is not None and len(obs[key]):
                    rec.add_array(f"reset_{key[:-5]}", np.stack([np.asarray(x) for x in obs[key]]))
        goal = _goal_text(info.get("task_goal"))
        ctx.writer.log_demo(fronts, wrists, states, [goal] if goal is not None else [])
        ctx.demo_frames = max(len(fronts) - 1, 0)
        return obs, info

    def step(self, action):
        import numpy as np  # noqa: PLC0415
        ctx = self._ctx
        if ctx.strict_cap and ctx.effective_cap is not None and ctx.attempted >= ctx.effective_cap:
            ctx.cap_hit = True
            rec = ctx.recorder
            if rec is not None and hasattr(rec, "add_event"):
                rec.add_event({"kind": "step_cap_reached", "step": ctx.attempted, "cap": ctx.effective_cap})
            print(f"ASTRA_STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap} rejected_step={ctx.attempted + 1}",
                  flush=True)
            raise StepCapReached(f"STEP_CAP exec_steps={ctx.attempted} cap={ctx.effective_cap}")
        rec = ctx.ensure_recorder()
        ctx.attempted += 1
        ctx.t = n = ctx.attempted
        a = np.array(action, copy=True)
        ctx.actions.append(a)
        link = ctx.step_link()
        if rec is not None:
            rec.add_array("exec_action", a, step=n - 1)
        try:
            out = self._inner.step(action)
        except BaseException as exc:
            ctx.writer.log_missing_step(step=n, action=a, reason=f"env_step_exception:{type(exc).__name__}",
                                        subgoal=ctx.subgoal, **link)
            raise
        obs, reward, terminated, truncated, info = out
        status = info.get("status") if isinstance(info, dict) else None
        if obs is None or not obs.get("front_rgb_list") or not obs.get("wrist_rgb_list"):
            ctx.writer.log_missing_step(step=n, action=a, reason="obs_none", subgoal=ctx.subgoal, **link)
            return out
        front = np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8)
        wrist = np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8)
        if rec is not None:
            rec.add_frames("front", front, tag=f"step{n}")
            rec.add_frames("wrist", wrist, tag=f"step{n}")
            rec.add_array("joint_state", np.asarray(obs["joint_state_list"][-1]), step=n - 1)
            rec.add_array("gripper_state", np.asarray(obs["gripper_state_list"][-1]), step=n - 1)
        ctx.observed += 1
        ctx.writer.log_step(step=n, front=front, wrist=wrist, state=_pack_state(obs), action=a, subgoal=ctx.subgoal,
                            terminated=terminated, truncated=truncated, status=status, **link)
        return out

    def close(self):
        return self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TracedBuilder:
    """只暴露 ``runner.episode`` 用到的两个方法；``make_env_for_episode`` 不传步数，取构造时的 ``max_steps``。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def make_env_for_episode(self, episode):
        return TracedEnv(self._inner.make_env_for_episode(episode), self._ctx)

    def resolve_episode(self, episode):
        return self._inner.resolve_episode(episode)


# ── 语言账本（冻结说明五；LanguageLog 由 R6 的 trace_writer 提供） ─────────────

#: 附图变换描述：无损 PNG（规划／监视的单帧附件）、``core.sheets`` 拼图（JPEG q92）、websocket 原数组（VLA）
PNG_TRANSFORM = {"resize": None, "crop": None, "layout": None, "encode": "png"}
SHEET_TRANSFORM = {"resize": None, "crop": None, "layout": "core.sheets 4x4 grid, 256x280 cell, frame label",
                   "encode": "jpeg q92"}
RAW_TRANSFORM = {"resize": None, "crop": None, "layout": None, "encode": "msgpack_numpy"}
#: Astra 拼图每页帧数（``core.sheets``）
SHEET_PAGE = 16


def language_log_cls():
    """可选探测：``trace_writer.LanguageLog``（R6）；不存在时返回 ``None``（不记语言账本、不报错）。"""
    import trace_writer as tw  # noqa: PLC0415
    return getattr(tw, "LanguageLog", None)


def _image_ref(slot: int, ref: str, phase: str, frame_idx, cam: str, raw_sha: str | None, *, transform: dict,
               sources: list | None = None, encoded: Path | None = None) -> dict:
    """附图引用（不存图片本身）：单帧的 ``sources`` 为它自己；拼图给有序来源帧列表、``frame_idx=None``。"""
    if sources is None:
        sources = [{"phase": phase, "frame_idx": frame_idx, "cam": cam, "raw_sha256": raw_sha}]
    enc = file_sha256(encoded) if encoded is not None and Path(encoded).is_file() else None
    return {"slot": slot, "ref": ref, "phase": phase, "frame_idx": frame_idx, "cam": cam, "raw_sha256": raw_sha,
            "sources": sources, "transform": dict(transform), "encoded_sha256": enc}


def _sheet_sources(ctx: TraceContext, phase: str, indices: list, frames) -> list:
    return [{"phase": phase, "frame_idx": int(i), "cam": "front", "raw_sha256": ctx.frame_sha(phase, int(i), frames)}
            for i in indices]


def planner_images(ctx: TraceContext, out: Path, request: dict, frames, demo, *, wrist=None,
                   command_start: int | None = None) -> list:
    """按 ``request.json`` 的 ``images`` 顺序给规划／复审请求的附图引用。

    exec 帧号 = Astra ``frames`` 下标（0 = reset 后初始帧，即 trace demo 段末帧；k = 第 k 步执行后，即 trace step k）；
    demo 帧号 = 演示段下标。拼图页 p 覆盖下标列表的第 ``16p`` 至 ``16p+15`` 个。"""
    now = len(frames) - 1
    memory = [int(i) for i in request.get("memory_frame_ids") or []]
    demo_idx = list(range(len(demo or [])))
    images = []
    for slot, name in enumerate(request.get("images") or []):
        path = out / name
        if name == "current.png":
            images.append(_image_ref(slot, "current", "exec", now, "front", ctx.frame_sha("exec", now, frames),
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name == "execution_start.png":
            images.append(_image_ref(slot, "keyframe", "exec", 0, "front", ctx.frame_sha("exec", 0, frames),
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name == "command_start.png" and command_start is not None:
            images.append(_image_ref(slot, "command_start", "exec", int(command_start), "front",
                                     ctx.frame_sha("exec", int(command_start), frames), transform=PNG_TRANSFORM,
                                     encoded=path))
        elif name == "current_wrist.png":
            import trace_writer as tw  # noqa: PLC0415
            images.append(_image_ref(slot, "wrist", "exec", now, "wrist",
                                     tw.image_sha256(wrist) if wrist is not None else None,
                                     transform=PNG_TRANSFORM, encoded=path))
        elif name.startswith(("demo_", "memory_")):
            phase, ref = ("demo", "demo_sheet") if name.startswith("demo_") else ("exec", "memory_sheet")
            page = int(Path(name).stem.split("_")[1])
            pool = demo_idx if phase == "demo" else memory
            src_frames = demo if phase == "demo" else frames
            idx = pool[page * SHEET_PAGE:(page + 1) * SHEET_PAGE]
            images.append(_image_ref(slot, ref, phase, None, "front", None, transform=SHEET_TRANSFORM,
                                     sources=_sheet_sources(ctx, phase, idx, src_frames), encoded=path))
        else:  # 未知附件：只记文件哈希，不猜来源
            images.append(_image_ref(slot, None, "exec", None, "front", None, transform=PNG_TRANSFORM, sources=[],
                                     encoded=path))
    return images


class PlannerLanguageCall:
    """一次规划（或第二按钮复审）请求的语言账本记录器。

    ``wrap(responder)`` 返回给 Astra ``Planner`` 用的发送方：Planner 写完 ``prompt.txt``／``request.json``／附图后调用它，
    它**先**开调用、写 ``in`` 消息（prompt 全文 + 附图引用）落盘，再交给真实发送方；发送方若支持
    ``transport_hook``（本文件的 ``GuardedResponsesClient``），每次传输重试（attempt ≥ 1）把上一个调用记 ``error``、
    另开一个 ``transport_attempt=k`` 的调用并重写 ``in`` 消息。``finish`` 在 Planner 返回或抛错后写回复原文与收尾。"""

    def __init__(self, ctx: TraceContext, image_fn: Callable[[Path, dict], list], kind: str) -> None:
        self.ctx = ctx
        self.image_fn = image_fn
        self.kind = kind
        self.call_id: str | None = None
        self.out: Path | None = None
        self.prompt: str | None = None
        self.images: list | None = None
        self.params: dict | None = None
        self.step = ctx.t

    def _open(self, transport_attempt: int) -> None:
        lang = self.ctx.lang
        self.call_id = lang.open_call("planner", self.step, params=self.params, transport_attempt=transport_attempt)
        lang.message(self.call_id, dir="in", role="user", text=self.prompt, images=self.images)

    def wrap(self, inner: Callable) -> Callable:
        def responder(out):
            out = Path(out)
            self.out = out
            request = json.loads((out / "request.json").read_text())
            self.prompt = (out / "prompt.txt").read_text()
            self.images = self.image_fn(out, request)
            self.params = {"temperature": None, "max_tokens": PLANNER_MAX_OUTPUT_TOKENS,
                           "model_id": request.get("model"), "adapter_sha": None, "effort": request.get("effort"),
                           "kind": request.get("kind", self.kind), "request_id": out.name, "cloud_seed": None,
                           "policy_seed": self.ctx.action_params.get("policy_seed")}
            self._open(0)  # 发送前落盘
            has_hook = hasattr(inner, "transport_hook")
            if has_hook:
                previous = inner.transport_hook
                inner.transport_hook = self.transport
            try:
                return inner(out)
            finally:
                if has_hook:
                    inner.transport_hook = previous
        return responder

    def transport(self, out, attempt: int) -> None:
        """``GuardedResponsesClient._send`` 每次传输尝试前调用；attempt 0 即 ``wrap`` 已开的那个调用。"""
        if int(attempt) == 0 or self.call_id is None:
            return
        self.ctx.lang.close_call(self.call_id, status="error")
        self._open(int(attempt))

    def finish(self, *, parsed=None, fallback=None, failed: bool = False) -> None:
        if self.call_id is None:
            return
        lang = self.ctx.lang
        response = None
        if self.out is not None and (self.out / "response.json").is_file():
            try:
                response = json.loads((self.out / "response.json").read_text())
            except ValueError:
                response = None
        replied = bool(response) and response.get("status") == "ok"
        if replied:
            lang.message(self.call_id, dir="out", role="assistant", text=response.get("text"))
            if failed and fallback is None:  # 有回复但不合约定（如复审不是 true/false）
                fallback = "model_response_error"
        lang.close_call(self.call_id, status="reply" if replied else "error", parsed=parsed, fallback=fallback)
        self.call_id = None


def _monitor_images(ctx: TraceContext, ids, frames, command_start: int, wrist) -> list:
    """监视器 10 张图（``input_contract.build_input`` 顺序）：最近 8 帧、本条命令起点帧、当前腕部帧。"""
    import trace_writer as tw  # noqa: PLC0415
    images = [_image_ref(slot, "recent", "exec", int(fid), "front", ctx.frame_sha("exec", int(fid), frames),
                         transform=PNG_TRANSFORM) for slot, fid in enumerate(ids or [])]
    images.append(_image_ref(len(images), "command_start", "exec", int(command_start), "front",
                             ctx.frame_sha("exec", int(command_start), frames), transform=PNG_TRANSFORM))
    images.append(_image_ref(len(images), "wrist", "exec", len(frames) - 1, "wrist",
                             tw.image_sha256(wrist) if wrist is not None else None, transform=PNG_TRANSFORM))
    return images


class TracedClient:
    """包住 VLA websocket 客户端：记每次 ``infer`` 的规范化请求与完整动作块；开了语言账本时每次 ``infer`` 记一个
    ``action_model`` 调用（``in``：``prompt``／``grounded_subgoal``／``simple_subgoal`` 字段原文与两张当前帧引用，
    发送前落盘；Astra 的 VLA 服务无审计回包，``server_final_text=None``）。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def reset(self):
        self._ctx.writer.log_request("vla_reset", b"", step=self._ctx.t)
        return self._inner.reset()

    def _open_lang(self, element) -> str | None:
        ctx = self._ctx
        if ctx.lang is None:
            return None
        import trace_writer as tw  # noqa: PLC0415
        call_id = ctx.lang.open_call("action_model", ctx.t, params=dict(ctx.action_params))
        fields = {k: element.get(k) for k in ("prompt", "grounded_subgoal", "simple_subgoal") if k in element}
        images = [_image_ref(0, "current", "exec", ctx.t, "front", tw.image_sha256(element.get("observation/image")),
                             transform=RAW_TRANSFORM),
                  _image_ref(1, "wrist", "exec", ctx.t, "wrist",
                             tw.image_sha256(element.get("observation/wrist_image")), transform=RAW_TRANSFORM)]
        ctx.lang.message(call_id, dir="in", role="fields", text=fields, images=images)
        return call_id

    def infer(self, element):
        ctx = self._ctx
        ctx.subgoal = element.get("grounded_subgoal")
        call_id = self._open_lang(element)
        ctx.source_call_id, ctx.chunk_index = call_id, 0
        ctx.writer.log_request("vla_infer", _canonical(element), step=ctx.t)
        try:
            out = self._inner.infer(element)
        except BaseException:
            if call_id is not None:
                ctx.lang.close_call(call_id, status="error")
            raise
        if call_id is not None:
            ctx.lang.close_call(call_id, status="reply", server_final_text=None, server_truncated=None)
        ctx.writer.log_response(out.get("actions"), step=ctx.t)
        return out


def _spool_payload(out: Path) -> bytes:
    """规划请求的规范化字节：``request.json``（去掉 id／created）+ ``prompt.txt`` + 各附图字节的 sha256。"""
    request = json.loads((out / "request.json").read_text())
    images = {name: file_sha256(out / name) for name in request.get("images", []) if (out / name).is_file()}
    request = {k: v for k, v in request.items() if k not in ("id", "created")}
    prompt = (out / "prompt.txt").read_text() if (out / "prompt.txt").is_file() else ""
    return json.dumps({"request": request, "prompt": prompt, "images": images}, sort_keys=True,
                      ensure_ascii=False).encode()


class TracedPlanner:
    """包住 Astra ``Planner``：每次规划／复审请求按 spool 目录内容记一行 request；开了语言账本时临时把
    ``Planner.responder`` 换成 ``PlannerLanguageCall.wrap`` 的发送方（调用结束即还原）。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def _log(self, name: str, rid: str) -> None:
        out = Path(self._inner.spool) / rid
        if (out / "request.json").is_file():
            self._ctx.writer.log_request(name, _spool_payload(out), step=self._ctx.t)

    def _call(self, method: str, image_fn: Callable, parse: Callable, a, k):
        ctx = self._ctx
        fn = getattr(self._inner, method)
        responder = getattr(self._inner, "responder", None)
        if ctx.lang is None or responder is None:
            return fn(*a, **k)
        call = PlannerLanguageCall(ctx, image_fn, method)
        self._inner.responder = call.wrap(responder)
        try:
            result = fn(*a, **k)
        except BaseException:
            call.finish(failed=True)
            raise
        finally:
            self._inner.responder = responder
        parsed, fallback = parse(result)
        call.finish(parsed=parsed, fallback=fallback)
        return result

    def predict(self, task, goal, frames, demo, memory, completed, issued, episode, t):
        a = (task, goal, frames, demo, memory, completed, issued, episode, t)
        result = self._call(
            "predict", lambda out, req: planner_images(self._ctx, out, req, frames, demo),
            lambda r: (r[0], None if r[0] is not None else "continue_last"), a, {})
        self._log("planner", result[1])
        return result

    def review_second_button(self, goal, frames, wrist, subgoal, command_start, completed, issued, episode, t):
        a = (goal, frames, wrist, subgoal, command_start, completed, issued, episode, t)
        result = self._call(
            "review_second_button",
            lambda out, req: planner_images(self._ctx, out, req, frames, [], wrist=wrist, command_start=command_start),
            lambda r: (r[0], None), a, {})
        self._log("planner_review", result[1])
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TracedMonitor:
    """包住监视器：记 ``input.json``（去掉图片路径，换成各图 sha256）。开了语言账本时每次 ``predict`` 记一个
    ``monitor`` 调用：推理**之前**按上游 ``input_contract.from_observations`` 落 system、user（10 张图引用），
    之后写回复原文（``response.json`` 的 ``text``）与 ``parsed`` 布尔。"""

    def __init__(self, inner, ctx: TraceContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def _open_lang(self, task, goal, subgoal, frames, command_start, wrist) -> str | None:
        ctx = self._ctx
        if ctx.lang is None:
            return None
        try:
            import input_contract  # noqa: PLC0415  Astra 的监视器输入契约（与 Monitor.predict 第一行同一函数）
            sample, ids = input_contract.from_observations(task, goal, subgoal, frames, command_start, wrist)
            system, user = sample["messages"][0]["content"], sample["messages"][1]["content"]
        except Exception:  # noqa: BLE001 构造失败时上游 predict 也会同样失败；仍先落一条可读的输入
            ids, system = None, None
            user = json.dumps({"task": task, "goal": goal, "subgoal": subgoal, "command_start": command_start},
                              ensure_ascii=False)
        call_id = ctx.lang.open_call("monitor", ctx.t, params=dict(ctx.monitor_params))
        ctx.lang.message(call_id, dir="in", role="system", text=system)
        images = _monitor_images(ctx, ids, frames, command_start, wrist) if ids is not None else []
        ctx.lang.message(call_id, dir="in", role="user", text=user, images=images)
        return call_id

    def _close_lang(self, call_id: str, out: Path, pred, failed: bool) -> None:
        lang = self._ctx.lang
        text = None
        response = Path(out) / "response.json"
        if response.is_file():
            try:
                text = json.loads(response.read_text()).get("text")
            except ValueError:
                text = None
        if text is not None:
            lang.message(call_id, dir="out", role="assistant", text=text)
        if failed:
            lang.close_call(call_id, status="reply" if text is not None else "error", parsed=None,
                            fallback="model_response_error" if text is not None else None)
        else:
            lang.close_call(call_id, status="reply", parsed=pred)

    def predict(self, task, goal, subgoal, frames, command_start, wrist, out):
        call_id = self._open_lang(task, goal, subgoal, frames, command_start, wrist)
        try:
            result = self._inner.predict(task, goal, subgoal, frames, command_start, wrist, out)
        except BaseException:
            if call_id is not None:
                self._close_lang(call_id, out, None, failed=True)
            raise
        if call_id is not None:
            self._close_lang(call_id, out, result[0], failed=False)
        sample_path = Path(out) / "input.json"
        if sample_path.is_file():
            sample = json.loads(sample_path.read_text())
            sample["images"] = [file_sha256(Path(p)) if Path(p).is_file() else None for p in sample.get("images", [])]
            payload = json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()
        else:
            payload = _canonical({"task": task, "goal": goal, "subgoal": subgoal, "command_start": command_start})
        self._ctx.writer.log_request("monitor", payload, step=self._ctx.t)
        return result


# ── 运行（复刻 main() 六项） ───────────────────────────────────────────────

def terminal_of(result: dict | None) -> str:
    """C3：Astra 结果的 ``status`` 已是 success／fail／timeout／error（``runner.episode`` 把其他环境状态归为 error）；
    循环走满 ``max_steps`` 与环境报 timeout 都是 ``timeout``；其余（含驱动异常、无结果）一律 ``error``。"""
    status = (result or {}).get("status")
    return status if status in TERMINALS else "error"


def episode_key(task: str, identity: dict) -> str:
    """C6 局目录 key：``<task>_<tier>_<seed>``（与 ``env_client.v8_key`` 同式）。"""
    return f"{task}_{identity['tier']}_{int(identity['seed'])}"


def recorder_meta(args, task: str, ep: int, identity: dict, key: str, media_dir: Path) -> dict:
    """录制器 meta：字段取 ``env_client.SeatRunner.base_record`` + ``run_one`` 同款，``never_degrade=True``（始终无损）。"""
    import uuid  # noqa: PLC0415
    ident_full = {"tier": identity.get("tier"), "seed": int(identity["seed"]), "candidate": identity.get("candidate"),
                  "spec_sha256": identity.get("spec_sha256"), "builder_episode": int(ep),
                  "source_episode": identity.get("source_episode")}
    return {"v8": True, "key": key, "task": task, "tier": identity.get("tier"), "seed": int(identity["seed"]),
            "candidate": identity.get("candidate"), "spec_sha256": identity.get("spec_sha256"),
            "source_episode": identity.get("source_episode"), "identity": ident_full, "builder_episode": int(ep),
            "dataset": args.dataset, "policy": "astra", "policy_variant": "astra", "route": ROUTE,
            "strict_cap": strict_cap_of(args.dataset), "cond": None, "seat": None, "host": socket.gethostname(),
            "attempt": ATTEMPT, "attempt_no": ATTEMPT, "canary": False, "gpu_name": None, "gpu_uuid": None,
            "git_commit": None, "git_dirty": None, "max_steps": int(args.max_steps),
            "effective_max_steps": int(args.max_steps), "effective_cap": int(args.max_steps),
            "policy_seed": getattr(args, "policy_seed", None), "cloud_seed": None,
            "rec_dir": str(media_dir), "attempt_id": uuid.uuid4().hex, "resolved_identity": dict(identity),
            "env": None, "never_degrade": True, "baseline": False}


def strict_cap_of(dataset: str) -> bool:
    """冻结说明 2.1：ood 为 strict，hard-verify 不是；未知数据集按非 strict（配对检查早已拒绝）。"""
    return bool(DATASET_STRICT_CAP.get(dataset, False))


def default_recorder_factory(media_dir: Path, meta: dict):
    import recorder as recorder_mod  # noqa: PLC0415  本目录的 recorder.py（bootstrap 已把 HERE 放进 sys.path）
    return recorder_mod.EpisodeRecorder(media_dir, meta)


def _open_media(factory: Callable, media_dir: Path, meta: dict):
    """``media/`` 必须是新建的空目录（不覆盖任何已有证据，不传 ``overwrite=True``）。"""
    media_dir.mkdir(parents=False, exist_ok=False)
    return factory(media_dir, meta)


def stage_raw_media(media_dir: Path, ep_dir: Path) -> list[str]:
    """录制器核对通过后把原始帧移到局目录（C5）；目标已存在即拒绝（不覆盖）。返回移动的文件名。"""
    moved = []
    for name in RAW_MEDIA_FILES:
        src, dst = media_dir / name, ep_dir / name
        if not src.is_file():
            continue
        if dst.exists():
            raise FileExistsError(f"局目录已有 {dst}，拒绝覆盖")
        os.replace(src, dst)
        moved.append(name)
    return moved


def write_exec_actions(path: Path, actions: list) -> None:
    """C4：每个执行步的原动作写 ``exec_action__%05d``（0 起步序号）；一旦写就每步都有键。

    冻结说明四.4：``arrays.npz`` 唯一允许的写法是 ``trace_writer.merge_write_npz``（与 ``TraceWriter.close`` 收集的同键
    逐项核对后合并、原子替换）；该函数不存在（R6 未合入）时保持旧的 ``np.savez``。"""
    import numpy as np  # noqa: PLC0415
    import trace_writer as tw  # noqa: PLC0415
    if not actions:
        return
    mapping = {f"exec_action__{i:05d}": a for i, a in enumerate(actions)}
    merge = getattr(tw, "merge_write_npz", None)
    if merge is not None:
        merge(Path(path), mapping)
    else:
        np.savez(path, **mapping)


def prompt_digests(champ: Path) -> dict[str, str]:
    """``prompts/*.md`` 与 ``prompts/index.json`` 的 sha256（键为相对 ``examples/champ`` 的路径）。"""
    prompts = Path(champ) / "prompts"
    files = sorted(prompts.glob("*.md")) + ([prompts / "index.json"] if (prompts / "index.json").is_file() else [])
    return {str(p.relative_to(champ)): file_sha256(p) for p in files}


def write_provenance(path: Path, *, astra, args, key: str, strict_cap: bool, language: bool) -> dict:
    """局目录来源清单：Astra 源码与 prompts 的 sha256、模型 seed（云端无 seed 接口，``cloud_seed=null``）、实际 cap。"""
    doc = {"route": ROUTE, "key": key, "attempt": ATTEMPT, "dataset": args.dataset,
           "policy_seed": getattr(args, "policy_seed", None), "server_seed": getattr(args, "policy_seed", None),
           "cloud_seed": None, "cloud_seed_note": "planner/monitor cloud API has no seed interface; not fabricated",
           "effective_cap": int(args.max_steps), "strict_cap": bool(strict_cap),
           "astra_root": str(getattr(astra, "root", "")), "astra_sources": source_digests(astra.champ),
           "prompts": prompt_digests(astra.champ), "language_log": "language.jsonl" if language else None}
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return doc


def stop_file(output: Path) -> Path:
    """``main()`` 第③项查的位置：``<output>/../../STOP.json``（即 ``group_*/STOP.json``）。"""
    return Path(output).parent.parent / "STOP.json"


def check_layout(output: Path, spool: Path) -> None:
    """``results`` 与 ``planner_calls`` 同属一个 RUN 目录，RUN 的上一层必须叫 group_0／group_1。"""
    output, spool = Path(output).resolve(), Path(spool).resolve()
    if output.parent != spool.parent:
        raise ValueError(f"RUN_BLOCKED reason=layout --output 与 --spool 须在同一 RUN 目录下：{output} {spool}")
    if output.parent.parent.name not in GROUP_DIR_NAMES:
        raise ValueError(f"RUN_BLOCKED reason=layout RUN 目录的上一层须为 {GROUP_DIR_NAMES} 之一（STOP.json 停机口），"
                         f"实为 {output.parent.parent}")


def run_cases(args, deps: SimpleNamespace) -> dict:
    """复刻 Astra ``main()``（2026-10-04 按上游源码核实的六项）：

    ① ``validate_checkpoints``：VLA 与监视器文件齐全、监视器哈希与 ``weights.json`` 相同；
    ② ``--output`` 与 ``--spool`` 都必须不存在，``mkdir(exist_ok=False)``；
    ③ 每局开跑前查 ``<output>/../../STOP.json``，存在即停；
    ④ ``status=="error"`` 且不是规划输出不合模板（``core.is_planner_failure``）的局计入错误数，
       其余结局清零（与上游同为「连续」口径），累计 3 次停整个分片；
    ⑤ 错误信息以 ``Planner API``／``Pilot planner-call``／``Planner bridge failed`` 开头的局，单次即停；
    ⑥ 全部局正常走完写 ``PILOT_FINISHED.json``。

    ``deps``：``astra``（bootstrap 结果）、``builder_cls``、``make_client()``、``make_monitor(base, adapter)``、
    ``make_responder()``；可选 ``recorder_factory(media_dir, meta)``（缺省用真实 ``recorder.EpisodeRecorder``）与
    ``episode_gate``（``CostGate``：局数硬上限，正式运行由 ``default_deps`` 给）。测试注入替身，正式运行用 ``default_deps``。
    """
    astra = deps.astra
    output, spool = Path(args.output), Path(args.spool)
    check_layout(output, spool)
    document = json.loads(Path(args.cases).read_text())
    cases = validate_cases(document, astra.core.TASKS)
    args.dataset = document["dataset"]
    check_pairing(args.dataset, args.max_steps)
    args.policy_seed = check_policy_seed(getattr(args, "policy_seed", None))
    gate = getattr(deps, "episode_gate", None)
    if gate is not None:
        check_episode_cap(cases)
    astra.release_utils.validate_checkpoints(args.vla_checkpoint, args.monitor_adapter)  # ①
    responder = deps.make_responder()
    for directory in (output, spool):  # ②
        if directory.exists():
            raise ValueError("Use new output and spool directories; existing evidence is preserved")
    output.mkdir(parents=True, exist_ok=False)
    spool.mkdir(parents=True, exist_ok=False)

    tasks = list(dict.fromkeys(c["task"] for c in cases))
    builders, identities = {}, {}
    for task in tasks:  # 付费请求之前先把全部身份核完
        builders[task] = make_builder(deps.builder_cls, task, args.dataset, args.max_steps)
        for case in (c for c in cases if c["task"] == task):
            identities[(task, case["episode"])] = verify_identity(builders[task], case)

    client = deps.make_client()
    monitor = deps.make_monitor(args.monitor_base, args.monitor_adapter)
    planner = astra.core.Planner(str(spool), responder=responder)
    results, errors = [], 0
    for task in tasks:
        episodes = sorted(c["episode"] for c in cases if c["task"] == task)
        for ep in episodes:
            if stop_file(output).exists():  # ③
                print(f"ASTRA_STOP reason=host_stop task={task} episode={ep}", flush=True)
                raise AstraStop("host_stop", "Host requested stop before next episode")
            rp = output / task / f"ep{ep:03d}" / "result.json"
            if rp.exists():
                continue
            if rp.parent.exists():
                raise RuntimeError("Incomplete prior episode; use a new output directory to preserve its evidence")
            if gate is not None:  # 局数硬上限（R6）：跨 RUN 登记，第 3 局在建环境与任何请求之前拒绝
                gate.register_episode(f"{args.dataset}:{task}:{ep}")
            result = run_one(args, task, ep, identities[(task, ep)], builders[task], monitor, planner, client, astra,
                             recorder_factory=getattr(deps, "recorder_factory", None))
            results.append(result)
            failed = result["status"] == "error" and not astra.core.is_planner_failure(result)
            errors = errors + 1 if failed else 0  # ④
            if result["status"] == "error" and result.get("error", "").startswith(PLANNER_STOP_PREFIXES):  # ⑤
                print(f"ASTRA_STOP reason=planner_error task={task} episode={ep}", flush=True)
                raise AstraStop("planner_error", "Stopping pilot after planner service or budget error")
            if errors >= INFRA_ERROR_LIMIT:
                print(f"ASTRA_STOP reason=infra_errors task={task} episode={ep} errors={errors}", flush=True)
                raise AstraStop("infra_errors", "Three consecutive infrastructure/protocol errors; stopping shard")
    from core import atomic_json  # noqa: PLC0415
    finished = {"dataset": args.dataset, "tasks": tasks, "finished_at": time.time()}
    atomic_json(output / "PILOT_FINISHED.json", finished)  # ⑥
    print(f"ASTRA_PILOT_FINISHED dataset={args.dataset} episodes={len(results)}", flush=True)
    return {"results": results, "finished": finished}


def run_one(args, task, ep, identity, builder, monitor, planner, client, astra, *,
            recorder_factory: Callable | None = None) -> dict:
    """一局：建 ``<key>.a1/`` 与 trace，经委托包装调 Astra 的 ``runner.episode``；``finally`` 里统一收尾：
    录制器 ``close(summary)`` → 原始帧移入局目录 → 写 ``arrays.npz`` → trace ``close``（C2、C3、C8）→
    ``result.json`` 追加映射。驱动异常（``runner.episode`` 自己不接的 ``BaseException``）照样收尾后再抛。"""
    import trace_writer as tw  # noqa: PLC0415
    ep_dir = Path(args.output) / task / f"ep{ep:03d}"
    key = episode_key(task, identity)
    tag = f"{key}.a{ATTEMPT}"
    a_dir = ep_dir / tag
    media_dir = a_dir / "media"
    a_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = Path(args.trace_root) / task / f"ep{ep:03d}" / tag if getattr(args, "trace_root", None) else a_dir
    policy_seed = getattr(args, "policy_seed", None)
    strict = strict_cap_of(args.dataset)
    trace_identity = {"task": task, "dataset": args.dataset, **identity, "builder_episode": int(ep), "key": key,
                      "attempt": ATTEMPT, "policy_seed": policy_seed}
    trace_dir.mkdir(parents=True, exist_ok=True)
    header_extra = _supported_kwargs(tw.TraceWriter, policy_seed=policy_seed, effective_cap=int(args.max_steps),
                                     strict_cap=strict)
    writer = tw.TraceWriter(trace_dir / "trace.jsonl", route=ROUTE, identity=trace_identity, max_steps=args.max_steps,
                            **header_extra)
    lang_cls = language_log_cls()
    lang = lang_cls(trace_dir / "language.jsonl") if lang_cls is not None else None
    write_provenance(a_dir / PROVENANCE_FILE, astra=astra, args=args, key=key, strict_cap=strict,
                     language=lang is not None)
    meta = recorder_meta(args, task, ep, identity, key, media_dir)
    factory = recorder_factory or default_recorder_factory
    ctx = TraceContext(writer, open_recorder=lambda: _open_media(factory, media_dir, meta),
                       effective_cap=int(args.max_steps), strict_cap=strict, lang=lang,
                       action_params={"temperature": None, "max_tokens": None, "model_id": "mme_vla_suite",
                                      "adapter_sha": None, "checkpoint": str(args.vla_checkpoint),
                                      "policy_seed": policy_seed},
                       monitor_params={**MONITOR_PARAMS, "model_id": getattr(args, "monitor_base", None),
                                       "adapter": str(args.monitor_adapter), "adapter_sha": None})
    result: dict | None = None
    driver_error: BaseException | None = None
    try:
        result = astra.runner.episode(args, task, ep, TracedBuilder(builder, ctx), TracedMonitor(monitor, ctx),
                                      TracedPlanner(planner, ctx), TracedClient(client, ctx))
        return result
    except BaseException as exc:
        driver_error = exc
        raise
    finally:
        _finish_one(args, ep_dir, a_dir, trace_dir, media_dir, key, ctx, writer, result, driver_error)


def _supported_kwargs(fn: Callable, **candidates) -> dict:
    """可选探测：只把 ``fn`` 签名里显式声明的关键字传过去（R6 的 ``TraceWriter`` 新增 header 字段前后都能跑）。"""
    import inspect  # noqa: PLC0415
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in candidates.items() if k in params}


def _finish_one(args, ep_dir: Path, a_dir: Path, trace_dir: Path, media_dir: Path, key: str, ctx: TraceContext,
                writer, result: dict | None, driver_error: BaseException | None) -> None:
    terminal = terminal_of(result) if driver_error is None else "error"
    if ctx.cap_hit and driver_error is None:  # strict cap 拒第 cap+1 步：Astra 记的 error 改为 timeout（冻结说明 2.1）
        terminal = "timeout"
    has_demo = ctx.demo_frames is not None
    no_frame = not has_demo
    demo_frames = ctx.demo_frames if has_demo else 0
    frames_recorded = demo_frames + 1 + ctx.observed if has_demo else 0
    summary = {"status": terminal, "steps": (result or {}).get("steps"), "exec_steps": ctx.attempted,
               "steps_observed": ctx.observed, "frames_recorded": frames_recorded, "route": ROUTE, "key": key}
    rec_verify, rec_error, staged = None, None, []
    if ctx.recorder is not None:
        try:
            rsum = ctx.recorder.close(summary)
            rec_verify = (rsum or {}).get("RECORDER_VERIFY")
            if rec_verify == "PASS":
                staged = stage_raw_media(media_dir, a_dir)
        except Exception as exc:  # noqa: BLE001 录制收尾失败只记原因，原始块留在 media/，成绩照常收尾
            rec_verify, rec_error = "ERROR", f"{type(exc).__name__}: {exc}"[:800]
    write_exec_actions(trace_dir / "arrays.npz", ctx.actions)
    if ctx.lang is not None:
        ctx.lang.close()  # 仍未关闭的调用补 cancelled
    extra = {"policy_seed": getattr(args, "policy_seed", None), "cloud_seed": None, "strict_cap": ctx.strict_cap,
             "effective_cap": ctx.effective_cap, "cap_hit": ctx.cap_hit,
             "language": "language.jsonl" if ctx.lang is not None else None, "provenance": PROVENANCE_FILE}
    if result is not None:
        extra.update({k: result.get(k) for k in ("planner_calls", "monitor_calls", "review_calls", "error")})
    if driver_error is not None:
        extra["driver_exception"] = f"{type(driver_error).__name__}: {driver_error}"[:800]
    if no_frame and terminal != "error":  # 理论上不会发生：没 reset 却正常结束；按 error 记，避免假终态
        terminal = "error"
    writer.close(status=terminal, terminal_reason=terminal, demo_frames=demo_frames, steps_attempted=ctx.attempted,
                 steps_observed=ctx.observed, frames_recorded=frames_recorded, omitted_timeout_frames=0,
                 no_frame=no_frame, recorder_verify=rec_verify, recorder_error=rec_error, raw_media_staged=staged,
                 **extra)
    if result is not None:
        rel = lambda p: os.path.relpath(p, ep_dir)  # noqa: E731
        if ctx.cap_hit and driver_error is None:
            result["status"] = "timeout"  # run_cases 第④项不把 strict cap 计为基础设施错误
        result.update(route=ROUTE, key=key, attempt=ATTEMPT, episode_dir=a_dir.name, media_dir=rel(media_dir),
                      trace=rel(trace_dir / "trace.jsonl"), exec_steps=ctx.attempted, steps_observed=ctx.observed,
                      frames_recorded=frames_recorded, demo_frames=demo_frames, terminal_reason=terminal,
                      recorder_verify=rec_verify, policy_seed=getattr(args, "policy_seed", None), cloud_seed=None,
                      strict_cap=ctx.strict_cap, effective_cap=ctx.effective_cap, cap_hit=ctx.cap_hit)
        if (ep_dir / "result.json").is_file():
            from core import atomic_json  # noqa: PLC0415  Astra 的原子写（与它写 result.json 同一函数）
            atomic_json(ep_dir / "result.json", result)


# ── 费用硬上限：守卫状态同步预留 + 子类化 ResponsesClient ─────────────────────

_GUARD_MOD = None


def guard_module():
    """按路径加载同目录 ``astra_cost_guard.py``（预留协议、锁、文件名的唯一来源）。"""
    global _GUARD_MOD
    if _GUARD_MOD is None:
        import importlib.util  # noqa: PLC0415
        spec = importlib.util.spec_from_file_location("astra_cost_guard_for_runner", HERE / "astra_cost_guard.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GUARD_MOD = mod
    return _GUARD_MOD


class GuardRefused(RuntimeError):
    """发送前拒发（守卫失联、守卫已停、预留失败、局数到顶）。消息以 ``Planner API`` 开头：经 Planner 包成
    ``Planner bridge failed`` 后同样触发 ``run_cases`` 第⑤项单次即停。"""


def _image_tokens_upper(width: int, height: int) -> int:
    """单张 ``detail=high`` 图片输入 token 的保守上界：取两种公开计价口径的较大者。

    ① 512 切块：先缩到 2048×2048 内、再把短边缩到 768，``85 + 170 × 块数``；
    ② 32 像素小块：小块数（上限 1536）× 2.5（各型号乘数的最大值取整上浮）。"""
    import math  # noqa: PLC0415
    w, h = max(int(width), 1), max(int(height), 1)
    scale = min(1.0, 2048 / max(w, h))
    w1, h1 = w * scale, h * scale
    scale2 = min(1.0, 768 / min(w1, h1))
    w2, h2 = w1 * scale2, h1 * scale2
    tiles = math.ceil(w2 / 512) * math.ceil(h2 / 512)
    patches = min(math.ceil(w / 32) * math.ceil(h / 32), 1536)
    return max(85 + 170 * tiles, math.ceil(patches * 2.5))


def _image_size(data_url: str) -> tuple[int, int]:
    """从 ``data:<mime>;base64,<...>`` 解出图片尺寸；解不出按 2048×2048（最坏）计。"""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=False)
        from PIL import Image  # noqa: PLC0415
        with Image.open(io.BytesIO(raw)) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return 2048, 2048


def estimate_request(payload: dict) -> dict:
    """按本次实际请求估输入 token 上界：文本按 UTF-8 字节数（每 token 至少 1 字节），图片按 ``_image_tokens_upper``，
    另加 64 的格式开销；输出按请求自带的 ``max_output_tokens``。"""
    text_bytes, images = 0, []
    for item in payload.get("input") or []:
        for part in item.get("content") or []:
            if part.get("type") == "input_text":
                text_bytes += len(str(part.get("text", "")).encode("utf-8"))
            elif part.get("type") == "input_image":
                images.append(_image_tokens_upper(*_image_size(str(part.get("image_url", "")))))
    return {"input_tokens": text_bytes + sum(images) + 64, "text_bytes": text_bytes, "images": len(images),
            "image_tokens": sum(images), "max_output_tokens": int(payload.get("max_output_tokens") or 0)}


class CostGate:
    """runner 一侧的守卫协议：发送前同步读守卫状态、原子预留单次最坏费用；登记局数。

    状态文件由 ``astra_cost_guard.py --state`` 每轮写；预留文件与锁由守卫模块定名。任何读不到、过期（心跳超过
    ``GUARD_HEARTBEAT_TIMEOUT_S``）、守卫已退出或已停的情况都拒发——宁停不发。"""

    def __init__(self, state_path: str | Path, *, clock: Callable[[], float] = time.time,
                 heartbeat_timeout: float = GUARD_HEARTBEAT_TIMEOUT_S) -> None:
        self.state_path = Path(state_path)
        self.clock = clock
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.guard = guard_module()

    def read_state(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError) as exc:
            raise GuardRefused(f"Planner API cost guard unavailable: state unreadable ({type(exc).__name__})") from None
        if state.get("schema") != self.guard.STATE_SCHEMA:
            raise GuardRefused(f"Planner API cost guard unavailable: schema {state.get('schema')!r}")
        age = self.clock() - float(state.get("heartbeat") or 0)
        if state.get("exited"):
            raise GuardRefused("Planner API cost guard unavailable: guard exited")
        if not -1.0 <= age <= self.heartbeat_timeout:
            raise GuardRefused(f"Planner API cost guard unavailable: heartbeat age {age:.1f}s > {self.heartbeat_timeout:g}s")
        if state.get("stop"):
            raise GuardRefused(f"Planner API cost guard stopped: {state.get('reason')}")
        return state

    def cap(self, state: dict) -> float:
        """生效上限 = min(守卫的 --cap, 本轮硬上限)；守卫状态被改大也不放宽。"""
        return min(float(state.get("cap") or 0.0), float(self.guard.HARD_CAP_USD))

    def worst_usd(self, state: dict, estimate: dict) -> float:
        prices = state["prices"]
        out_tokens = max(int(estimate["max_output_tokens"]), int(state.get("max_output_tokens") or 0))
        usd = (estimate["input_tokens"] * float(prices["input"]) + out_tokens * float(prices["output"])) / 1e6
        if prices.get("reasoning_billed_separately"):
            usd += out_tokens * float(prices["output"]) / 1e6
        return usd

    def reserve(self, rid: str, payload: dict) -> dict:
        """锁内：读状态 → 已计 + 未结预留 + 本次最坏 > 上限即拒；否则记预留（同一 rid 重试不重复计，取较大者）。"""
        estimate = estimate_request(payload)
        g = self.guard
        with g.locked(self.state_path):
            state = self.read_state()
            doc = g.load_reservations(self.state_path)
            worst = self.worst_usd(state, estimate)
            existing = doc["reservations"].get(rid)
            others = g.outstanding_reservations(
                {"reservations": {k: v for k, v in doc["reservations"].items() if k != rid}}, state.get("counted") or [])
            committed = float(state.get("committed_usd") or 0.0)
            amount = max(worst, float(existing["usd"])) if existing and existing.get("state") != "released" else worst
            cap = self.cap(state)
            if committed + others + amount > cap:
                raise GuardRefused(f"Planner API cost reservation refused: committed={committed:.4f} "
                                   f"reserved={others:.4f} worst={amount:.4f} cap={cap:g}")
            doc["reservations"][rid] = {"usd": amount, "state": "reserved", "time": self.clock(), **estimate}
            g.save_reservations(self.state_path, doc)
        return {"usd": amount, **estimate}

    def mark(self, rid: str, state_name: str) -> None:
        g = self.guard
        with g.locked(self.state_path):
            doc = g.load_reservations(self.state_path)
            if rid in doc["reservations"]:
                doc["reservations"][rid]["state"] = state_name
                doc["reservations"][rid][f"{state_name}_at"] = self.clock()
                g.save_reservations(self.state_path, doc)

    def register_episode(self, episode_id: str) -> int:
        """局数硬上限：同一预留文件里跨 RUN 登记；已登记的同一局不重复计；第 3 局拒绝。"""
        g = self.guard
        with g.locked(self.state_path):
            self.read_state()
            doc = g.load_reservations(self.state_path)
            if episode_id not in doc["episodes"]:
                if len(doc["episodes"]) >= g.ASTRA_MAX_EPISODES:
                    print(f"ASTRA_STOP reason=episode_cap episode={episode_id} used={len(doc['episodes'])}", flush=True)
                    raise AstraStop("episode_cap", f"Astra episode cap {g.ASTRA_MAX_EPISODES} reached")
                doc["episodes"].append(episode_id)
                g.save_reservations(self.state_path, doc)
            return len(doc["episodes"])


def check_episode_cap(cases: list) -> None:
    limit = guard_module().ASTRA_MAX_EPISODES
    if len(cases) > limit:
        raise ValueError(f"RUN_BLOCKED reason=astra_episode_cap cases={len(cases)} cap={limit}（R6）")


_GUARDED_CLASSES: dict = {}


def guarded_client_class(api_client):
    """返回第三方 ``api_client.ResponsesClient`` 的子类（按基类缓存）；第三方文件零改动。

    覆写 ``_send``：每一次真正 ``urlopen`` 之前依次 ① 查 ``group_*/STOP.json``；② 同步读守卫状态并原子预留
    本次最坏费用（同一请求的 429 重试复用同一份预留）；③ 等待第三方的 20 秒发送间隔；④ 等待后再查一次
    STOP 与守卫状态。任一不通过即写 ``guard_refused.json``、释放预留并抛错（第三方 ``__call__`` 把它记成
    ``status=error`` 的 ``response.json``）。其余（HTTP 错误记录、429 有界退避、脱敏）与第三方 ``_send`` 逐项同式。"""
    base = api_client.ResponsesClient
    if base in _GUARDED_CLASSES:
        return _GUARDED_CLASSES[base]

    class GuardedResponsesClient(base):
        #: 语言账本的传输尝试回调 ``(out, attempt)``（``PlannerLanguageCall.transport``）；``None`` 时不调用。
        #: 只记录、不改变发送与费用守卫的任何判断。
        transport_hook = None

        def __init__(self, key, gate: CostGate, *, sleep: Callable[[float], None] = time.sleep,
                     monotonic: Callable[[], float] = time.monotonic) -> None:
            super().__init__(key)
            self.gate = gate
            self._sleep = sleep
            self._monotonic = monotonic

        @staticmethod
        def _stop_requested(out: Path) -> bool:
            return any(p.name in GROUP_DIR_NAMES and (p / "STOP.json").exists() for p in out.parents)

        def _refuse(self, out: Path, rid: str, message: str):
            api_client.atomic_json(out / guard_module().REFUSED_MARKER, {"time": time.time(), "reason": message})
            try:
                self.gate.mark(rid, "released")
            except Exception:  # noqa: BLE001 守卫已失联时释放也可能失败；拒发本身不受影响
                pass
            print(f"ASTRA_GUARD_REFUSED request={rid} reason={json.dumps(message, ensure_ascii=False)}", flush=True)
            raise RuntimeError(message)

        def _gate_before_send(self, request, out: Path, rid: str) -> None:
            if self._stop_requested(out):
                self._refuse(out, rid, "Host requested stop; no new API request")
            try:
                self.gate.reserve(rid, json.loads(request.data))
            except GuardRefused as exc:
                self._refuse(out, rid, str(exc))
            self._sleep(max(0, self.next_request_at - self._monotonic()))
            if self._stop_requested(out):  # 等待间隔里 STOP 可能已到
                self._refuse(out, rid, "Host requested stop; no new API request")
            try:
                self.gate.read_state()
            except GuardRefused as exc:
                self._refuse(out, rid, str(exc))

        def _send(self, request, out):
            import random  # noqa: PLC0415
            import urllib.error  # noqa: PLC0415
            import urllib.request  # noqa: PLC0415
            from email.utils import parsedate_to_datetime  # noqa: PLC0415
            out = Path(out)
            rid = out.name
            for attempt in range(9):
                if self.transport_hook is not None:  # 语言账本：每次传输尝试的输入在过闸与发送之前落盘
                    self.transport_hook(out, attempt)
                self._gate_before_send(request, out, rid)
                self.next_request_at = self._monotonic() + SEND_INTERVAL_S
                self.gate.mark(rid, "sent")
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        return json.loads(response.read()), response.headers.get("x-request-id")
                except urllib.error.HTTPError as error:
                    body = error.read().decode("utf-8", errors="replace")
                    try:
                        detail = json.loads(body).get("error", {})
                    except ValueError:
                        detail = {}
                    code = detail.get("code") if isinstance(detail, dict) else None
                    kind = detail.get("type") if isinstance(detail, dict) else None
                    headers = {k: v for k, v in error.headers.items()
                               if k.lower() == "retry-after" or k.lower() == "x-request-id"
                               or k.lower().startswith("x-ratelimit-")}
                    api_client.atomic_json(out / f"http_error_{attempt:02d}.json",
                                           json.loads(self.scrub(json.dumps({"status": error.code, "body": body,
                                                                             "headers": headers, "time": time.time()}))))
                    retryable = error.code == 429 and (code in ("rate_limit_exceeded", "slow_down")
                                                       or kind == "rate_limit_error")
                    if not retryable or attempt == 8:
                        raise RuntimeError(self.scrub(f"HTTP {error.code}: {body}")) from None
                    delay = min(120, 10 * 2 ** attempt)
                    retry_after = error.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            try:
                                delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                            except (TypeError, ValueError, OverflowError):
                                pass
                    self._sleep(delay + random.uniform(0, 3))

    _GUARDED_CLASSES[base] = GuardedResponsesClient
    return GuardedResponsesClient


def default_deps(astra: SimpleNamespace, port: int, guard_state: str | Path | None = None) -> SimpleNamespace:
    """正式运行的依赖。``guard_state`` 必给：规划客户端换成 ``GuardedResponsesClient``，并启用局数硬上限。"""
    if guard_state is None:
        raise ValueError("RUN_BLOCKED reason=cost_guard 正式运行必须给 --guard-state（先起 astra_cost_guard.py --cap 5）")
    gate = CostGate(guard_state)

    def make_client():
        from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy  # noqa: PLC0415
        import openpi_client  # noqa: PLC0415
        print(f"ASTRA_VLA_CLIENT openpi_client={openpi_client.__file__} port={port}", flush=True)
        return MMEVLAWebsocketClientPolicy("127.0.0.1", port)

    def make_responder():
        # 与 main() 的 ResponsesClient(os.environ.get('OPENAI_API_KEY','')) 同一密钥来源；不支持 --key-file（R5）
        return guarded_client_class(astra.api_client)(os.environ.get("OPENAI_API_KEY", ""), gate)

    return SimpleNamespace(astra=astra, builder_cls=default_builder_cls(), make_client=make_client,
                           make_monitor=lambda base, adapter: astra.runner.Monitor(base, adapter),
                           make_responder=make_responder, recorder_factory=default_recorder_factory,
                           episode_gate=gate)


# ── 汇总 ──────────────────────────────────────────────────────────────────

def summarize(cases_path: Path, results_root: Path) -> dict:
    document = json.loads(Path(cases_path).read_text())
    counts = dict(success=0, fail=0, timeout=0, error=0, pending=0)
    rows = []
    for case in document["cases"]:
        path = Path(results_root) / case["task"] / f"ep{case['episode']:03d}" / "result.json"
        result = json.loads(path.read_text()) if path.is_file() else {**case, "dataset": document["dataset"], "status": "pending"}
        if result.get("dataset") != document["dataset"]:
            raise ValueError(f"结果 dataset 与局清单不符：{path}")
        counts[result["status"] if result["status"] in counts else "error"] += 1
        rows.append(result)
    n = len(document["cases"])
    return dict(dataset=document["dataset"], expected=n, counts=counts,
                complete=counts["error"] == counts["pending"] == 0,
                success_rate=counts["success"] / n if n else 0.0, cases=rows)


# ── 命令行 ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="生成新侧局清单")
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_STEP_PAIRING))
    p.add_argument("--tasks", nargs="+", default=None, help="默认 Astra core.TASKS 全部 16 个")
    p.add_argument("--source-episodes", nargs="+", type=int, default=None, help="hard-verify：官方 test 局号（如 3）")
    p.add_argument("--tier", default=None, help="ood：档名（如 xhard1）")
    p.add_argument("--index", type=int, default=0, help="ood：该档内第几局（从 0 计）")
    p.add_argument("--output", required=True)
    p.add_argument("--astra-root", default=None)

    def common(q):
        q.add_argument("--cases", required=True)
        q.add_argument("--vla-checkpoint", required=True)
        q.add_argument("--monitor-adapter", required=True)
        q.add_argument("--max-steps", type=int, required=True, help="步数只来自启动命令：hard-verify 1300，ood 1800")
        q.add_argument("--policy-seed", default=None,
                       help="模型 seed（必填，非负整数；VLA 服务 --seed 同值；云端 planner／monitor 无 seed 接口，只记录）")
        q.add_argument("--port", type=int, default=18762)
        q.add_argument("--astra-root", default=None)

    c = sub.add_parser("check", help="起跑前核对（不加载模型、不联网）")
    common(c)
    c.add_argument("--guard-state", default=None, help="给出时核对费用守卫心跳新鲜、上限不超过 5 美元")

    r = sub.add_parser("run", help="正式运行")
    common(r)
    r.add_argument("--max-planner-calls", type=int, default=24)
    r.add_argument("--monitor-base", default=None, help="默认 Astra runner.BASE")
    r.add_argument("--output", required=True)
    r.add_argument("--spool", required=True)
    r.add_argument("--trace-root", default=None,
                   help="默认写在 results/<task>/ep<NNN>/<key>.a1/trace.jsonl（给出时为 <根>/<task>/ep<NNN>/<key>.a1/）")
    r.add_argument("--guard-state", required=True, help="astra_cost_guard.py --state 的状态文件（发送前同步预留）")

    s = sub.add_parser("summarize")
    s.add_argument("--cases", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--output", required=True)

    m = sub.add_parser("media-inputs", help="为 official_media_check.py 写本局身份清单行与账本行")
    m.add_argument("--episode-dir", required=True, help="<RUN>/results/<task>/ep<NNN>/<key>.a<N>")
    m.add_argument("--manifest", required=True)
    m.add_argument("--ledger", required=True)
    return parser


def media_inputs(episode_dir: Path) -> tuple[dict, list[dict]]:
    """从 trace header 取身份：清单行（身份字段，``official_media_check.py`` 取 ``key``）与账本行。

    账本按 ``env_client.AttemptLedger`` 口径写两行：``attempt_start``（``attempt_id``、``attempt_no``）与 ``accept``
    （``accepted_attempt_id == attempt_id``、``attempt_no`` = 目录名 ``.a<N>``）；Astra 没有 env_client 账本，
    ``attempt_id`` 由 key 与尝试号确定。"""
    import re  # noqa: PLC0415
    episode_dir = Path(episode_dir).resolve()
    header = json.loads((episode_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ident = header["identity"]
    m = re.fullmatch(r"(?P<key>.+)\.a(?P<n>[1-9]\d*)", episode_dir.name)
    if not m or m["key"] != ident.get("key") or int(m["n"]) != int(ident.get("attempt", -1)):
        raise ValueError(f"局目录名 {episode_dir.name} 与 trace identity key／attempt 不符")
    fields = ("task", "tier", "seed", "candidate", "spec_sha256", "builder_episode", "source_episode", "dataset", "key")
    manifest = {k: ident.get(k) for k in fields}
    manifest["route"] = header["route"]
    n = int(m["n"])
    attempt_id = f"astra-{ident['key']}-a{n}"
    common = {"key": ident["key"], "task": ident["task"], "tier": ident.get("tier"), "dataset": ident["dataset"],
              "route": header["route"], "attempt_id": attempt_id, "attempt_no": n, "episode_dir": str(episode_dir)}
    ledger = [{"kind": "attempt_start", **common, "builder_episode": ident.get("builder_episode"), "retry": False},
              {"kind": "accept", **common, "accepted_attempt_id": attempt_id, "status": end_status(episode_dir)}]
    return manifest, ledger


def end_status(episode_dir: Path) -> str | None:
    lines = [ln for ln in (Path(episode_dir) / "trace.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    return json.loads(lines[-1]).get("status") if lines else None


def _print_sources(astra: SimpleNamespace) -> None:
    digests = source_digests(astra.champ)
    print("ASTRA_SOURCE root=" + str(astra.root) + " " + " ".join(f"{k}={v}" for k, v in digests.items()), flush=True)


def cmd_check(args) -> int:
    astra = bootstrap(astra_root(args.astra_root))
    _print_sources(astra)
    assert_env_sources()
    document = json.loads(Path(args.cases).read_text())
    cases = validate_cases(document, astra.core.TASKS)
    check_pairing(document["dataset"], args.max_steps)
    args.policy_seed = check_policy_seed(args.policy_seed)
    check_episode_cap(cases)
    if args.guard_state:
        gate = CostGate(args.guard_state)
        state = gate.read_state()
        print(f"ASTRA_GUARD=PASS cap={gate.cap(state):g} committed={float(state.get('committed_usd') or 0):.4f} "
              f"heartbeat_age={time.time() - float(state['heartbeat']):.1f}s", flush=True)
    astra.release_utils.validate_checkpoints(args.vla_checkpoint, args.monitor_adapter)
    builder_cls = default_builder_cls()
    for task in dict.fromkeys(c["task"] for c in cases):
        builder = make_builder(builder_cls, task, document["dataset"], args.max_steps)
        for case in (c for c in cases if c["task"] == task):
            verify_identity(builder, case)
    with socket.socket() as sock:  # 端口探测：与 run.sh 相同的 bind 检查
        sock.bind(("0.0.0.0", int(args.port)))
    print(f"ASTRA_CHECK=PASS dataset={document['dataset']} max_steps={args.max_steps} "
          f"strict_cap={int(strict_cap_of(document['dataset']))} policy_seed={args.policy_seed} cases={len(cases)} "
          f"port={args.port}", flush=True)
    return 0


def main(argv: list[str] | None = None, deps_factory: Callable | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "summarize":
        summary = summarize(Path(args.cases), Path(args.results))
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"ASTRA_SUMMARY dataset={summary['dataset']} expected={summary['expected']} "
              + " ".join(f"{k}={v}" for k, v in summary["counts"].items()), flush=True)
        return 0
    if args.cmd == "media-inputs":
        manifest, ledger = media_inputs(Path(args.episode_dir))
        for path, rows in ((args.manifest, [manifest]), (args.ledger, ledger)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
        acc = ledger[-1]
        # 末行只给 run_astra.sh 取 dataset 用：dataset=<值> 恒在行尾
        print(f"ASTRA_MEDIA_INPUTS key={acc['key']} attempt_no={acc['attempt_no']} dir={acc['episode_dir']} "
              f"dataset={acc['dataset']}", flush=True)
        return 0
    if args.cmd in ("check", "run"):  # 冻结说明 2.2：在加载任何上游模块、建任何目录之前拒绝
        try:
            args.policy_seed = check_policy_seed(args.policy_seed)
        except ValueError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 3
    if args.cmd == "check":
        return cmd_check(args)
    astra = bootstrap(astra_root(args.astra_root))
    if args.cmd == "prepare":
        out = Path(args.output)
        if out.exists():
            raise SystemExit(f"{out} 已存在；局清单不覆盖")
        document = prepare_cases(default_builder_cls(), args.dataset, args.tasks or list(astra.core.TASKS),
                                 source_episodes=args.source_episodes, tier=args.tier, index=args.index)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"ASTRA_CASES dataset={document['dataset']} cases={len(document['cases'])} output={out}", flush=True)
        return 0
    _print_sources(astra)
    assert_env_sources()
    if args.monitor_base is None:
        args.monitor_base = astra.runner.BASE
    print(f"ASTRA_POLICY_SEED policy_seed={args.policy_seed} server_seed={args.policy_seed} cloud_seed=null", flush=True)
    deps = deps_factory(astra, args.port) if deps_factory else default_deps(astra, args.port, args.guard_state)
    run_cases(args, deps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
