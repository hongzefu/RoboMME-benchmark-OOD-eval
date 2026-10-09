#!/usr/bin/env python3
"""从官方源码按名摘取定义（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.3）。

官方 MME-VLA 客户端 ``third_party/mme-vla/examples/robomme/eval.py`` 经 ``subgoal_predictor.py`` 无条件导入 gemini
（``google.generativeai``）与 memer 模块，评估环境里都没装，所以 GroundSG 的新侧（``groundsg_client.py``）与原侧
（``official_hard_runner.py``）都**不整模块 import** 官方文件，而是用 ``extract_defs`` 从源文件里取出指定顶层
函数、类、单目标赋值的**原文**并执行；模块其余部分（import 行、模块级副作用）不执行，依赖由调用方经 ``extra``
注入。返回的命名空间就是这些定义的 globals，之后往里补名字（如延迟导入的 ``PtEngine``）对已取出的函数同样生效。

官方模块头部有三项环境设置（``eval.py`` 与 ``subgoal_prediction/qwenvl/api.py`` 顶部），摘取时不会被带走，由
``apply_official_env`` 负责；QwenVL 的离线运行约束由 ``apply_qwen_runtime_env`` 负责。

``load_groundsg`` 是两侧共用的装配：按变体只取需要的类（Oracle 不读 qwenvl/api.py、不导入 swift；QwenVL 只取
``QwenVLSubgoalPredictor`` 与 ``Qwen3VLModel``，不取 Gemini／MemER；MemER 只取 ``MemERSubgoalPredictor`` 与
``qwenvl/api_memer.py::Qwen3VLModelMemER``，不取 Gemini／QwenVL）。

MemER 兼容层（1006-rename-official-names-and-stage3-eval-plan.md 第二部分八.3「MemER 兼容层」；用户 2026-10-06
「同意兼容层」）：摘出 ``Qwen3VLModelMemER`` 原文后在 **AST 上**打补丁（只动摘出来的副本，``third_party`` 与 gitlink
不动）——官方的 ``merge_key_frame_paths``／``_get_current_execution_frame_paths``／``update_history_subgoals``／
``call`` 原样改名为 ``_official_<名>``（``official_method_name``）留在类里，再拼入 ``MEMER_COMPAT_SOURCE`` 里的同名方法：

1. ``merge_key_frame_paths``：记忆为空直接返回，非空调官方原函数（逐字节同）；
2. ``call``：每次合法解析把换算后交给动作模型的子目标存进 ``self.subgoals``；解析失败最多重问两次（共三次），第二、三
   次 user prompt 末尾追加 ``MEMER_RETRY_NOTE``、``RequestConfig(max_tokens=128, temperature=0.7)``，请求与回复带
   ``retry=<n>`` 追加进 ``ep*_MemER_log.jsonl``；三次都坏有上一次合法子目标即沿用（``fallback=last_valid``），否则抛
   ``MemERResponseError``（客户端记 ``error_kind=model_response_error``，不重跑）；
3. ``_get_current_execution_frame_paths``：从末帧起隔一张取一张、数到第 1 张之前即停（不足 15 张有几张取几张），
   1 张或 ≥15 张时调官方原函数；
4. ``update_history_subgoals``：原子校验——先在临时副本上核 JSON 结构、非空字符串子任务、整数关键帧位置（拒 bool、
   拒 0 与负数、拒超范围）、坐标换算与候选记忆合并，全部通过才一次提交；坏回复不改关键帧、历史与执行帧。

补丁源文本的 sha256 即实现指纹 ``MEMER_COMPAT_SHA256``（写进判定行、结果行与媒体 provenance）。提问模板、system
prompt、``prepare_infer_request``、关键帧挑选与（非空时）合并规则一字不改。

官方源码位置：环境变量 ``SGEVAL_THIRD_PARTY``（指向某个检出的 ``third_party``；worktree 里子模块目录为空时测试
用它只读引用主检出），否则取本仓库 ``third_party``。
"""
from __future__ import annotations

import ast
import collections
import copy
import dataclasses
import hashlib
import json
import os
import pprint
import random
import re
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

#: 评估仓根（本文件在 src/robomme_ood_eval/models/ 下）
REPO = Path(__file__).resolve().parents[3]
#: 三个变体（接口冻结说明 2.3；env_client.GROUNDSG_VARIANTS 由 R3 同步放行 MemER）
VARIANT_ORACLE = "ground-sg-oracle"
VARIANT_QWENVL = "ground-sg-qwenvl"
VARIANT_MEMER = "ground-sg-memer"
VARIANTS = (VARIANT_ORACLE, VARIANT_QWENVL, VARIANT_MEMER)
_SEQ = 0
#: 官方模块头部的三项环境设置（eval.py 与 qwenvl/api.py 顶部原样）
OFFICIAL_ENV = {"IMAGE_MAX_TOKEN_NUM": "256", "VIDEO_MAX_TOKEN_NUM": "64", "FPS_MAX_FRAMES": "10"}
#: QwenVL 运行约束（ms-swift 默认走 ModelScope；本计划一律离线走 HF 缓存）
QWEN_RUNTIME_ENV = {"USE_HF": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# ── 官方名与旧名兼容（1006-rename-official-names-and-stage3-eval-plan.md 第二部分八.2；全仓别名表只此一份）──
# 写出一律用官方名，CLI 只接受官方名；只有读历史逐局行、预算账本、trace 头、v7.5eval 留档时经 canonical_* 映射。
#: 策略标签（官方名）：FrameSamp+Modulation 与 GroundSG
POLICY_FRAMESAMP_MODUL = "perceptual-framesamp-modul"
POLICY_GROUNDSG = "groundsg"
#: 数据集接口（官方名）：第三阶段 OOD（原 V9）与第二阶段 hard-verify（官方 hard 12 局）
DATASET_OOD = "ood"
DATASET_HARD_VERIFY = "hard-verify"
# >>> LEGACY_NAMES（OFFICIAL_NAMES 残留检查只豁免本段）
#: 旧策略标签 → 官方标签（v7.5eval／第二阶段留档里的 policy、route 首段、目录名）
LEGACY_POLICY_ALIASES = {"mme": POLICY_FRAMESAMP_MODUL, "mmevla": POLICY_FRAMESAMP_MODUL, "mmesg": POLICY_GROUNDSG}
#: 旧数据集名 → 官方数据集名
LEGACY_DATASET_ALIASES = {"test-hard": DATASET_OOD, "test-hard0": DATASET_HARD_VERIFY}
#: 改名前检出（如回放闸门的 base 侧）的客户端模块名与配置键 → 官方名（client_replay_eq.py 驱动旧检出时用）
LEGACY_MODULE_ALIASES = {"mme_client": "framesamp_modul_client", "mmesg_client": "groundsg_client"}
LEGACY_CONFIG_KEY_ALIASES = {"mme_variant": "groundsg_variant"}
# <<< LEGACY_NAMES


def canonical_policy(name: Any) -> Any:
    """策略标签：旧名映射到官方名；带变体的标签（旧 GroundSG 前缀 + ``-<variant>``）同样换前缀；其余原样返回。"""
    if not isinstance(name, str):
        return name
    if name in LEGACY_POLICY_ALIASES:
        return LEGACY_POLICY_ALIASES[name]
    for old, new in LEGACY_POLICY_ALIASES.items():
        if new == POLICY_GROUNDSG and name.startswith(old + "-"):
            return new + name[len(old):]
    return name


def canonical_dataset(name: Any) -> Any:
    """数据集名：旧名映射到官方名，其余原样返回。"""
    return LEGACY_DATASET_ALIASES.get(name, name) if isinstance(name, str) else name


def canonical_route(route: Any) -> Any:
    """路线／媒体键这类以 ``/`` 分段的串：逐段按策略标签与数据集名映射（如旧 GroundSG 路线
    ``<旧名>/<variant>/orig`` → ``groundsg/<variant>/orig``）。"""
    if not isinstance(route, str) or not route:
        return route
    return "/".join(canonical_dataset(canonical_policy(seg)) for seg in route.split("/"))


#: canonical_row 映射的字段
_POLICY_FIELDS = ("policy", "label", "policy_label")
_DATASET_FIELDS = ("dataset",)
_ROUTE_FIELDS = ("route",)


def canonical_row(row: Any) -> Any:
    """逐局行／账本行／trace 头：``policy``／``label``、``dataset``、``route`` 按旧名映射，``identity`` 内的
    ``dataset`` 一并映射；返回新字典，不改入参。非字典原样返回。"""
    if not isinstance(row, dict):
        return row
    out = dict(row)
    for k in _POLICY_FIELDS:
        if k in out:
            out[k] = canonical_policy(out[k])
    for k in _DATASET_FIELDS:
        if k in out:
            out[k] = canonical_dataset(out[k])
    for k in _ROUTE_FIELDS:
        if k in out:
            out[k] = canonical_route(out[k])
    if isinstance(out.get("identity"), dict) and "dataset" in out["identity"]:
        out["identity"] = {**out["identity"], "dataset": canonical_dataset(out["identity"]["dataset"])}
    return out


def legacy_labels(label: str) -> list[str]:
    """官方标签对应的旧标签（读历史目录用；不含官方标签自身），如 ``groundsg-<variant>`` → 旧 GroundSG 前缀版本。"""
    out = []
    for old, new in LEGACY_POLICY_ALIASES.items():
        if label == new:
            out.append(old)
        elif new == POLICY_GROUNDSG and label.startswith(new + "-"):
            out.append(old + label[len(new):])
    return out


def third_party_root() -> Path:
    """官方第三方源码根：``SGEVAL_THIRD_PARTY`` 优先，否则本仓库 ``third_party``。"""
    env = os.environ.get("SGEVAL_THIRD_PARTY")
    return Path(env).resolve() if env else REPO / "third_party"


def official_robomme_dir() -> Path:
    """官方 ``examples/robomme`` 目录；缺 ``eval.py`` 即报错（不静默跳过）。"""
    d = third_party_root() / "mme-vla" / "examples" / "robomme"
    if not (d / "eval.py").is_file():
        raise FileNotFoundError(f"官方 MME-VLA 源码不在：{d}/eval.py（子模块未初始化？可设 SGEVAL_THIRD_PARTY）")
    return d


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_defs(path: str | Path, names: list[str], extra: dict | None = None, *,
                 transform: dict | None = None) -> dict:
    """从源文件用 ast 取出指定顶层函数／类／单目标赋值的原文并执行，返回命名空间。

    * 只取 ``names`` 里列出的顶层 ``def``／``async def``／``class``（含装饰器）与单目标赋值 ``X = ...``；
      同名多次定义按源码顺序全部取出（与整模块执行时最后一个生效相同）；
    * 名字找不到抛 ``KeyError``；
    * 不执行模块其余部分（import 行、模块级副作用），依赖由 ``extra`` 注入（基础名 ``np``、``Any`` 等已预置）；
    * 命名空间记 ``__source_path__``、``__source_sha256__``（整文件字节的 sha256）；
    * ``transform``：``{名字: f(ast 节点) -> ast 节点}``，在执行前对取出的该定义做 AST 级改写（只用于 MemER 兼容层，
      ``__source_sha256__`` 仍是官方原文件的 sha256，改写内容另记指纹）。
    """
    path = Path(path)
    raw = path.read_bytes()
    src = raw.decode("utf-8")
    tree = ast.parse(src, filename=str(path))
    body = []
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                body.append(node)
                found.add(node.targets[0].id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            if node.target.id in names:
                body.append(node)
                found.add(node.target.id)
    missing = sorted(set(names) - found)
    if missing:
        raise KeyError(f"{path} 里找不到 {missing}")
    if transform:
        for i, node in enumerate(body):
            name = node.name if hasattr(node, "name") else None
            if name in transform:
                body[i] = transform[name](node)
    # 命名空间挂在一个独立的模块对象上并登记进 sys.modules（dataclass 等按 __module__ 回查模块）；
    # 模块名带 _official_ 前缀与序号，不与真实模块（eval、utils 等）同名
    global _SEQ
    _SEQ += 1
    mod = types.ModuleType(f"_official_{path.stem}_{_SEQ}")
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod
    ns: dict[str, Any] = mod.__dict__
    ns.update({
        "__builtins__": __builtins__,
        "np": np, "Any": Any, "Optional": Optional, "Tuple": Tuple, "List": List, "Path": Path,
        "os": os, "re": re, "json": json, "time": time, "shutil": shutil, "collections": collections,
        "dataclasses": dataclasses, "pprint": pprint,
    })
    ns.update(extra or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), ns)  # noqa: S102 只执行官方定义原文
    ns["__source_path__"] = str(path)
    ns["__source_sha256__"] = hashlib.sha256(raw).hexdigest()
    return ns


def apply_official_env() -> dict:
    """官方 eval.py／qwenvl/api.py 头部的三项环境设置（原样覆盖写）。返回写入的键值。"""
    for k, v in OFFICIAL_ENV.items():
        os.environ[k] = v
    return dict(OFFICIAL_ENV)


def apply_qwen_runtime_env() -> dict:
    """QwenVL 运行约束：``USE_HF=1``、``HF_HUB_OFFLINE=1``、``TRANSFORMERS_OFFLINE=1``（导入 swift 之前调用）。"""
    for k, v in QWEN_RUNTIME_ENV.items():
        os.environ[k] = v
    return dict(QWEN_RUNTIME_ENV)


def import_swift_names() -> dict:
    """真实的 ``swift.llm`` 三个名字（只在 QwenVL／MemER 变体构造预测器时调用；先设运行约束再导入）。"""
    apply_qwen_runtime_env()
    apply_official_env()
    from swift.llm import InferRequest, PtEngine, RequestConfig

    return {"PtEngine": PtEngine, "InferRequest": InferRequest, "RequestConfig": RequestConfig}


def seed_everything(seed: int) -> dict:
    """模型种子（接口冻结说明 2.2）：``random``、``numpy``、``torch``（可导入时）三处一起设。

    QwenVL／MemER 预测器构造前调用（``build_predictor``）；只调 ``torch.manual_seed``（它对 CUDA 是惰性登记，不在
    这里初始化 GPU）。返回实际设过的随机源，供结果与测试核对。"""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"policy_seed={seed!r} 须为非负整数")
    random.seed(seed)
    np.random.seed(seed)
    out = {"seed": int(seed), "random": True, "numpy": True, "torch": False}
    try:
        import torch
    except ImportError:
        return out
    torch.manual_seed(seed)
    out["torch"] = True
    return out


def check_policy_seed(value: Any) -> int:
    """``policy_seed`` 取值核对：非负整数（拒 bool、拒 None、拒负数）；字符串形式的十进制整数也收。"""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r}（必填，非负整数）")
    try:
        n = int(str(value).strip()) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r}（必填，非负整数）") from None
    if isinstance(value, float) and value != n:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r}（必填，非负整数）")
    if n < 0:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r}（必填，非负整数）")
    return n


#: 各官方文件要摘取的名字
UTILS_NAMES = ["TASK_WITH_VIDEO_DEMO", "TASK_NAME_LIST", "SUBGOAL_TYPES", "pack_buffer", "check_args", "EpisodeState",
               "RolloutRecorder"]
ENV_RUNNER_NAMES = ["pack_state", "EnvRunner"]
EVAL_NAMES = ["Args", "EpisodeEvaluator"]
PREDICTOR_NAMES = {
    VARIANT_ORACLE: ["SubgoalPredictorBase", "OracleSubgoalPredictor", "build_subgoal_predictor"],
    VARIANT_QWENVL: ["SubgoalPredictorBase", "QwenVLSubgoalPredictor", "build_subgoal_predictor"],
    VARIANT_MEMER: ["SubgoalPredictorBase", "MemERSubgoalPredictor", "build_subgoal_predictor"],
}
#: 各变体的子目标模型源文件（相对 examples/robomme）与类名；Oracle 无
SUBGOAL_MODEL_SOURCES = {
    VARIANT_QWENVL: ("subgoal_prediction/qwenvl/api.py", "Qwen3VLModel"),
    VARIANT_MEMER: ("subgoal_prediction/qwenvl/api_memer.py", "Qwen3VLModelMemER"),
}


# ── MemER 兼容层（八.3「MemER 兼容层」改法 1～4） ───────────────────────────────


class MemERResponseError(RuntimeError):
    """MemER 三次提问的回复都不合法、且本局此前没有任何合法子目标（情形 D）：客户端记
    ``status=error, terminal_reason=error, error_kind=model_response_error``，不伪造子目标、不重跑。"""

    error_kind = "model_response_error"


#: 第二、三次重问在 user prompt 末尾追加的提醒句（用户 2026-10-06「2的A和b都用」）
MEMER_RETRY_NOTE = "Your previous reply was not valid JSON. Reply with the JSON object only."
#: 提问总次数（首问 + 两次重问）与重问的采样温度
MEMER_MAX_TRIES = 3
MEMER_RETRY_TEMPERATURE = 0.7
#: 兼容层改动的官方方法（原文改名为 ``_official_<名>`` 保留在类里）
MEMER_PATCHED_METHODS = ("merge_key_frame_paths", "_get_current_execution_frame_paths", "update_history_subgoals",
                         "call")

#: 补丁源文本：拼进摘出来的 ``Qwen3VLModelMemER`` 类体；其 UTF-8 字节的 sha256 即实现指纹 ``MEMER_COMPAT_SHA256``。
#: 依赖名（``json``、``re``、``copy``、``RequestConfig``、``InferRequest``、``MemERResponseError``、``MEMER_*``）由
#: ``load_groundsg`` 注入摘取命名空间。改这段文字 = 换指纹，须同步计划与成绩表注明。
MEMER_COMPAT_SOURCE = '''\
def merge_key_frame_paths(self, dist: int = 8):
    # 兼容 1：关键帧记忆为空时直接返回（官方原文 cur = [nums[0]] 越界）；非空时调官方原函数，逐字节相同
    if not self.key_frame_paths:
        return
    return self._official_merge_key_frame_paths(dist)


def _get_current_execution_frame_paths(self) -> list:
    # 兼容 3：从末帧起隔一张取一张、最多 8 张，数到第 1 张之前即停；1 张或 >=15 张时调官方原函数（逐字相同）
    n = len(self.execution_frame_paths)
    if n == 1 or n >= 15:
        return self._official_get_current_execution_frame_paths()
    paths = []
    idx = n - 1
    while idx >= 0 and len(paths) < 8:
        paths.insert(0, self.execution_frame_paths[idx])
        idx -= 2
    return paths


def _memer_validate(self, subgoal: str):
    # 兼容 4：原子校验——全部在临时对象上做完，不改 self 的任何状态；任一项不合法即抛异常
    response = json.loads(subgoal)
    if not isinstance(response, dict):
        raise ValueError("reply is not a JSON object")
    current_subtask = response["current_subtask"]
    keyframe_positions = response["keyframe_positions"]
    if not isinstance(current_subtask, str) or not current_subtask.strip():
        raise ValueError(f"current_subtask must be a non-empty string: {current_subtask!r}")
    if not isinstance(keyframe_positions, list):
        raise ValueError(f"keyframe_positions must be a list: {keyframe_positions!r}")
    n_frames = len(self.current_execution_frame_paths)
    for key_id in keyframe_positions:
        if type(key_id) is not int or key_id < 1 or key_id > n_frames:
            raise ValueError(f"keyframe position {key_id!r} out of range 1..{n_frames}")
    vla_subgoal = self._parse_box_patterns(current_subtask, replacement="scaled_coords", return_bbox=False)
    if not isinstance(vla_subgoal, str) or not vla_subgoal.strip():
        raise ValueError(f"converted subgoal is empty: {vla_subgoal!r}")
    candidate = dict(self.key_frame_paths)
    for key_id in keyframe_positions:
        path_str = self.current_execution_frame_paths[key_id - 1]
        int_idx = int(re.search(r"step_(\\d+)_image.png", path_str).group(1))
        candidate[int_idx] = path_str
    saved = self.key_frame_paths
    try:
        self.key_frame_paths = candidate
        self.merge_key_frame_paths()
        merged = self.key_frame_paths
    finally:
        self.key_frame_paths = saved
    return current_subtask, vla_subgoal, merged, list(keyframe_positions)


def update_history_subgoals(self, subgoal: str):
    # 兼容 4：校验全过才一次提交关键帧记忆（返回值与官方相同：原始 current_subtask）
    current_subtask, _vla, merged, _pos = self._memer_validate(subgoal)
    self.key_frame_paths = merged
    return current_subtask


def _memer_request_fields(self, infer_request):
    # 首问请求的字段副本（发送前取，重问在其上只改 user prompt 末尾）
    fields = {"messages": copy.deepcopy(list(infer_request.messages)), "images": list(infer_request.images)}
    videos = getattr(infer_request, "videos", None)
    if videos:
        fields["videos"] = list(videos)
    return fields


def _memer_log(self, row):
    with open(self.save_json_path, "a") as f:
        json.dump(row, f)
        f.write("\\n")


def call(self) -> str:
    # 兼容 2：合法子目标存进 self.subgoals；坏回复最多重问两次（追加提醒句 + temperature=0.7）；三次都坏沿用上一次
    # 合法子目标，没有则抛 MemERResponseError。首问的请求与回复日志行与官方逐字相同，重问行带 retry
    infer_request = self.prepare_infer_request()
    base_fields = self._memer_request_fields(infer_request)
    self._memer_fallback = None
    self._memer_errors = []
    self._memer_last_positions = None
    for retry in range(MEMER_MAX_TRIES):
        self._memer_retry = retry
        if retry == 0:
            request = infer_request
            config = RequestConfig(max_tokens=128, temperature=0)
        else:
            fields = copy.deepcopy(base_fields)
            for message in fields["messages"]:
                if message.get("role") == "user":
                    message["content"] = message["content"] + "\\n" + MEMER_RETRY_NOTE
            self._memer_log({**fields, "retry": retry})
            request = InferRequest(**fields)
            config = RequestConfig(max_tokens=128, temperature=MEMER_RETRY_TEMPERATURE)
        response = self.engine.infer([request], request_config=config)
        response = response[0].choices[0].message.content
        print("Response: ", response)
        self._memer_log({"response": response} if retry == 0 else {"response": response, "retry": retry})
        try:
            _subtask, vla_subgoal, merged, positions = self._memer_validate(response)
        except Exception as e:
            print(f"Error updating history subgoals: {e}")
            self._memer_errors.append(f"{type(e).__name__}: {e}")
            continue
        self.key_frame_paths = merged
        self.subgoals.append(vla_subgoal)
        self._memer_last_positions = positions
        return vla_subgoal
    if self.subgoals:
        self._memer_fallback = "last_valid"
        self._memer_log({"fallback_used": 1, "fallback": "last_valid", "subgoal": self.subgoals[-1],
                         "errors": self._memer_errors})
        return self.subgoals[-1]
    self._memer_fallback = "model_response_error"
    self._memer_log({"fallback_used": 0, "fallback": "model_response_error", "errors": self._memer_errors})
    raise MemERResponseError(f"MemER replies invalid {MEMER_MAX_TRIES} times and no previous valid subgoal: "
                             + " | ".join(self._memer_errors)[:600])
'''
MEMER_COMPAT_SHA256 = hashlib.sha256(MEMER_COMPAT_SOURCE.encode("utf-8")).hexdigest()


def official_method_name(name: str) -> str:
    """被补官方方法改名后的名字：``call`` → ``_official_call``，``_get_x`` → ``_official_get_x``。"""
    return "_official" + ("" if name.startswith("_") else "_") + name


def patch_memer_class(cls_node: ast.ClassDef) -> ast.ClassDef:
    """AST 补丁：官方 ``MEMER_PATCHED_METHODS`` 改名为 ``_official_<名>``（函数体一字不动），再在类体末尾拼入
    ``MEMER_COMPAT_SOURCE`` 的方法。官方类里缺任一被补方法即 ``KeyError``（上游变了，不静默套用）。"""
    have = {n.name for n in cls_node.body if isinstance(n, ast.FunctionDef)}
    missing = sorted(set(MEMER_PATCHED_METHODS) - have)
    if missing:
        raise KeyError(f"Qwen3VLModelMemER 缺 {missing}，兼容层不适用")
    for n in cls_node.body:
        if isinstance(n, ast.FunctionDef) and n.name in MEMER_PATCHED_METHODS:
            n.name = official_method_name(n.name)
    compat = ast.parse(MEMER_COMPAT_SOURCE, filename="<MEMER_COMPAT_SOURCE>")
    cls_node.body.extend(compat.body)
    return cls_node


def memer_compat_extra(swift_names: dict | None) -> dict:
    """兼容层方法用到的名字（注入摘取命名空间）。"""
    extra = {"copy": copy, "MemERResponseError": MemERResponseError, "MEMER_RETRY_NOTE": MEMER_RETRY_NOTE,
             "MEMER_MAX_TRIES": MEMER_MAX_TRIES, "MEMER_RETRY_TEMPERATURE": MEMER_RETRY_TEMPERATURE,
             "MEMER_COMPAT_SHA256": MEMER_COMPAT_SHA256}
    extra.update(swift_names or {})
    return extra


def load_memer_model(d: str | Path | None = None, *, swift_names: dict | None = None, compat: bool = True) -> dict:
    """摘 ``subgoal_prediction/qwenvl/api_memer.py::Qwen3VLModelMemER``；``compat=True``（默认、两侧唯一用法）套
    兼容层，``compat=False`` 只供测试拿官方原文做回归对照。落实官方模块头部的三项环境变量（与 ``api.py`` 相同）。"""
    d = Path(d) if d is not None else official_robomme_dir()
    apply_official_env()
    import imageio

    transform = {"Qwen3VLModelMemER": patch_memer_class} if compat else None
    ns = extract_defs(d / "subgoal_prediction" / "qwenvl" / "api_memer.py", ["Qwen3VLModelMemER"],
                      {"imageio": imageio, **memer_compat_extra(swift_names)}, transform=transform)
    ns["__memer_compat_sha256__"] = MEMER_COMPAT_SHA256 if compat else None
    return ns


def load_groundsg(variant: str, *, env_runner_extra: dict | None = None, ws_module: Any = None,
                  qwen_extra: dict | None = None, with_env_runner: bool = True) -> dict:
    """两侧共用的官方定义装配，返回 ``{"utils","env_runner","predictor","qwen","eval","sha256","EnvRunner",...}``。

    * ``env_runner_extra``：给 ``env_runner.py`` 的依赖（原侧传真实 ``BenchmarkEnvBuilder``；测试传替身）。
      ``with_env_runner=False`` 时不取 ``EnvRunner``（新侧不用它，只取 ``pack_state``），子目标预测器与评估器
      里的 ``EnvRunner`` 注解以占位类代替（注解在定义时求值，不影响行为）。
    * ``ws_module``：注入为 ``eval.py`` 的 ``_websocket_client_policy``（须有 ``MMEVLAWebsocketClientPolicy``）；
      ``None`` 时用真实 ``openpi_client.websocket_client_policy``。
    * ``qwen_extra``：QwenVL／MemER 变体时注入子目标模型源文件的 ``PtEngine``／``InferRequest``／``RequestConfig``；
      ``None`` 时不注入，构造预测器前须由调用方补（``import_swift_names``）。Oracle 变体不读两个 qwenvl 源文件。
    * MemER：``qwen`` 键是套了兼容层的 ``Qwen3VLModelMemER`` 命名空间，``memer_compat_sha256`` 为实现指纹；
      ``sha256`` 记官方 ``api_memer.py`` 整文件 sha256。
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant={variant!r} 不是 {VARIANTS} 之一")
    d = official_robomme_dir()
    apply_official_env()
    import cv2
    import imageio

    utils = extract_defs(d / "utils.py", UTILS_NAMES, {"cv2": cv2, "imageio": imageio})
    er_names = ENV_RUNNER_NAMES if with_env_runner else ["pack_state"]
    er_extra = {"TASK_NAME_LIST": utils["TASK_NAME_LIST"]}
    er_extra.update(env_runner_extra or {})
    env_runner = extract_defs(d / "env_runner.py", er_names, er_extra)
    runner_cls = env_runner.get("EnvRunner") or type("EnvRunner", (), {"__doc__": "注解占位（新侧不用官方 EnvRunner）"})
    qwen = None
    pred_extra: dict[str, Any] = {"EnvRunner": runner_cls, "EpisodeState": utils["EpisodeState"],
                                  "SUBGOAL_TYPES": utils["SUBGOAL_TYPES"],
                                  "TASK_WITH_VIDEO_DEMO": utils["TASK_WITH_VIDEO_DEMO"]}
    if variant == VARIANT_QWENVL:
        qwen = extract_defs(d / "subgoal_prediction" / "qwenvl" / "api.py", ["Qwen3VLModel"],
                            {"imageio": imageio, **(qwen_extra or {})})
        pred_extra["Qwen3VLModel"] = qwen["Qwen3VLModel"]
    elif variant == VARIANT_MEMER:
        qwen = load_memer_model(d, swift_names=qwen_extra, compat=True)
        pred_extra["Qwen3VLModelMemER"] = qwen["Qwen3VLModelMemER"]
    predictor = extract_defs(d / "subgoal_predictor.py", PREDICTOR_NAMES[variant], pred_extra)
    if ws_module is None:
        from openpi_client import websocket_client_policy as ws_module  # noqa: N813 与官方同名
    ev = extract_defs(d / "eval.py", EVAL_NAMES, {
        "_websocket_client_policy": ws_module, "pack_buffer": utils["pack_buffer"], "check_args": utils["check_args"],
        "TASK_NAME_LIST": utils["TASK_NAME_LIST"], "TASK_WITH_VIDEO_DEMO": utils["TASK_WITH_VIDEO_DEMO"],
        "SUBGOAL_TYPES": utils["SUBGOAL_TYPES"], "EpisodeState": utils["EpisodeState"],
        "RolloutRecorder": utils["RolloutRecorder"], "EnvRunner": runner_cls,
        "build_subgoal_predictor": predictor["build_subgoal_predictor"],
        "SubgoalPredictorBase": predictor["SubgoalPredictorBase"],
    })
    sha = {"utils.py": utils["__source_sha256__"], "env_runner.py": env_runner["__source_sha256__"],
           "subgoal_predictor.py": predictor["__source_sha256__"], "eval.py": ev["__source_sha256__"]}
    if qwen is not None:
        sha[SUBGOAL_MODEL_SOURCES[variant][0]] = qwen["__source_sha256__"]
    return {"variant": variant, "dir": str(d), "utils": utils, "env_runner": env_runner, "predictor": predictor,
            "qwen": qwen, "eval": ev, "sha256": sha, "EnvRunner": env_runner.get("EnvRunner"),
            "pack_state": env_runner["pack_state"], "Args": ev["Args"], "EpisodeEvaluator": ev["EpisodeEvaluator"],
            "memer_compat_sha256": MEMER_COMPAT_SHA256 if variant == VARIANT_MEMER else None}


def make_args(defs: dict, *, variant: str, host: str, port: int, max_steps: int, model_seed: Any,
              adapter_path: str | None = None, memer_adapter_path: str | None = None,
              save_dir: str = "runs/evaluation") -> Any:
    """按变体构造官方 ``Args``：``subgoal_type="grounded_subgoal"``，``use_oracle``／``use_qwenvl``／``use_memer``
    恰一个为真（构造后断言；官方 ``build_subgoal_predictor`` 多开时静默取高优先级，这里不允许），``model_seed`` 必给
    且显式写进 ``Args.model_seed``（不沿用官方默认 42），再过官方 ``check_args``。

    adapter 配对（接口冻结说明 2.3）：QwenVL 必须且只能给 ``adapter_path``，MemER 必须且只能给
    ``memer_adapter_path``，Oracle 两个都不许给；配错抛 ``ValueError``。"""
    if variant not in VARIANTS:
        raise ValueError(f"variant={variant!r} 不是 {VARIANTS} 之一")
    seed = check_policy_seed(model_seed)
    want_q, want_m = variant == VARIANT_QWENVL, variant == VARIANT_MEMER
    if bool(adapter_path) != want_q:
        raise ValueError(f"{variant}：qwenvl_groundSG_adapter_path 仅且必须与 ground-sg-qwenvl 同用（给了 {adapter_path!r}）")
    if bool(memer_adapter_path) != want_m:
        raise ValueError(f"{variant}：memer_adapter_path 仅且必须与 ground-sg-memer 同用（给了 {memer_adapter_path!r}）")
    kw: dict[str, Any] = dict(host=host, port=int(port), max_steps=int(max_steps), save_dir=save_dir,
                              subgoal_type="grounded_subgoal", use_oracle=variant == VARIANT_ORACLE,
                              use_qwenvl=want_q, use_memer=want_m, model_seed=seed)
    if want_q:
        kw["qwenvl_groundSG_adapter_path"] = str(adapter_path)
    if want_m:
        kw["memer_adapter_path"] = str(memer_adapter_path)
    args = defs["Args"](**kw)
    assert_one_predictor(args)
    defs["utils"]["check_args"](args)
    return args


def assert_one_predictor(args: Any) -> None:
    """``use_oracle``／``use_qwenvl``／``use_memer`` 恰有一个为真，且 ``use_gemini`` 为假。"""
    flags = (bool(args.use_oracle), bool(args.use_qwenvl), bool(getattr(args, "use_memer", False)))
    if sum(flags) != 1 or getattr(args, "use_gemini", False):
        raise AssertionError(f"use_oracle={args.use_oracle} use_qwenvl={args.use_qwenvl} "
                             f"use_memer={getattr(args, 'use_memer', None)} use_gemini={getattr(args, 'use_gemini', None)}"
                             "：必须恰有 oracle／qwenvl／memer 之一")


def build_predictor(defs: dict, args: Any, save_dir: str | Path) -> Any:
    """官方 ``build_subgoal_predictor``（构造前再断言一次互斥）。QwenVL／MemER 先设离线运行约束、按 ``Args.model_seed``
    调 ``seed_everything``，未注入 swift 名字时此处导入真实 swift；``attn_impl='flash_attention_2'`` 等引擎参数一律取
    官方原文，不改。"""
    assert_one_predictor(args)
    if args.use_qwenvl or getattr(args, "use_memer", False):
        apply_qwen_runtime_env()
        if "PtEngine" not in defs["qwen"]:
            defs["qwen"].update(import_swift_names())
        seed_everything(check_policy_seed(args.model_seed))
    return defs["predictor"]["build_subgoal_predictor"](args, Path(save_dir))


def canonical_bytes(obj: Any) -> bytes:
    """请求的规范化字节（两侧同一函数）：dict 按键排序；数组记 dtype、shape 与 C 连续字节；字符串 UTF-8。"""
    out = bytearray()

    def put(x: Any) -> None:
        if isinstance(x, dict):
            out.extend(b"{")
            for k in sorted(x, key=str):
                put(str(k))
                out.extend(b":")
                put(x[k])
                out.extend(b",")
            out.extend(b"}")
        elif isinstance(x, (list, tuple)):
            out.extend(b"[")
            for v in x:
                put(v)
                out.extend(b",")
            out.extend(b"]")
        elif isinstance(x, (bytes, bytearray)):
            out.extend(b"b%d:" % len(x) + bytes(x))
        elif isinstance(x, str):
            b = x.encode("utf-8")
            out.extend(b"s%d:" % len(b) + b)
        elif x is None or isinstance(x, (bool, int, float)):
            out.extend(("p:" + repr(x)).encode())
        else:
            a = np.ascontiguousarray(np.asarray(x))
            out.extend(f"a:{a.dtype.str}|{a.shape}|".encode() + a.tobytes())

    put(obj)
    return bytes(out)


def ws_shim(factory) -> Any:
    """``eval.py`` 里 ``_websocket_client_policy`` 的替身模块：``MMEVLAWebsocketClientPolicy(host, port)`` → ``factory``。"""
    return types.SimpleNamespace(MMEVLAWebsocketClientPolicy=factory)
