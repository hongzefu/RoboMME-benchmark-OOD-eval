#!/usr/bin/env python3
"""MME-VLA 动作服务外壳（GroundSG 三个变体与 FrameSamp+Modulation 共用；1006 计划第二部分一节 R3 行、八.11、八.12
第 5／7 条；接口冻结说明 2.2、五节「服务外壳回包审计键」）。

包住锁定的三方 ``third_party/mme-vla/scripts/serve_policy.py``（**不改三方源码**）：

1. 参数：外壳自用 ``--sgeval-metadata-out <path>``（启动时从 argv 摘掉），其余原样交给三方 ``Args``（tyro），
   ``--seed=<policy_seed>`` 即三方 ``Args.seed``（第三阶段由 ``run_seat.sh::build_server_cmd`` 显式给出，不再写死 7）。
2. 服务元数据：三方 ``create_policy`` 返回后（权重已加载）写 ``server-metadata-<port>.json``：``policy_seed``、
   ``argv``、``pid``、``port``、``audit``、三方 ``serve_policy.py`` 的 sha256，供客户端反查结果行 ``server_seed``。
3. 回包审计键 ``_sgeval_audit``（环境变量 ``SGEVAL_AUDIT`` 缺省开）：三方 ``create_policy`` 的返回值包一层代理，
   ``infer`` 照常调用真实策略、原样返回其结果，只在结果 dict 上多加一个键::

       {"channels": [{"channel": "task"|"symbolic", "text", "token_ids", "mask", "tokenizer", "truncated"}, ...],
        "server_final_text": <symbolic 通道原文，没有则 task 通道原文>, "pp_generation": null}

   通道取自本次 ``infer`` 里三方 ``PaligemmaTokenizer.tokenize`` 的真实调用：``tokenize`` 不带 subgoal 的那次是
   ``task``（对应 ``tokenized_prompt``），带 subgoal 的那次是 ``symbolic``（对应 ``symbolic_tokenized_prompt``）。
   ``text`` 是三方实际交给 sentencepiece ``encode`` 的完整字符串（``Task: …;\\nCurrent Subgoal: …;\\nAction: `` 等，
   多次 ``encode`` 依序拼接），``token_ids``／``mask`` 是 ``tokenize`` 的真实返回（含补齐），``truncated`` 为
   ``encode`` 原始长度是否超过 ``max_len``。观察方式：``tokenize`` 只在「当前有正在进行的 infer」时，把实例的
   sentencepiece 处理器临时换成一个原样转调 ``encode`` 并记下参数与返回的代理，返回后立即换回；不另行分词、不多推理、
   不碰任何随机数发生器、不改动作。``SGEVAL_AUDIT=0`` 时既不包代理、也不打补丁，回包与三方逐字节相同
   （``OBS_EQ`` 对照）。
4. 导入环境与 ``python scripts/serve_policy.py`` 对齐：cwd 为 ``third_party/mme-vla``（``run_seat.sh`` 以绝对路径起本
   外壳），``sys.path[0]`` 换成 ``<mme-vla>/scripts``、本目录移出；日志同三方 ``__main__``（``basicConfig(INFO,
   force=True)``），所以 ``run_seat.sh`` 核对的 ``history_config='…'`` 日志行照常出现。cwd 下找不到三方脚本时退回
   ``$SGEVAL_THIRD_PARTY/mme-vla``。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

#: 服务回包审计键（接口冻结说明五节）
AUDIT_KEY = "_sgeval_audit"
ENV_AUDIT = "SGEVAL_AUDIT"
#: 外壳自用参数
METADATA_FLAG = "--sgeval-metadata-out"
#: 锁定三方脚本在 mme-vla 根下的相对路径
SERVE_REL = "scripts/serve_policy.py"

_HERE = str(Path(__file__).resolve().parent)
_ACTIVE = threading.local()  # 正在进行的 infer 的通道收集表（只在 infer 期间非空）


def audit_enabled() -> bool:
    """``SGEVAL_AUDIT`` 缺省开；为 ``0`` 时外壳不打补丁、不包代理、不加审计键。"""
    return os.environ.get(ENV_AUDIT, "1") != "0"


def split_wrapper_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """从 argv 摘掉 ``--sgeval-metadata-out[=]<path>``，返回 (路径, 其余参数)。"""
    meta, rest, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith(METADATA_FLAG + "="):
            meta = a.split("=", 1)[1]
        elif a == METADATA_FLAG and i + 1 < len(argv):
            meta = argv[i + 1]
            i += 1
        else:
            rest.append(a)
        i += 1
    return meta, rest


def mme_vla_root() -> Path:
    """三方 mme-vla 根：cwd 下有锁定脚本即 cwd，否则 ``$SGEVAL_THIRD_PARTY/mme-vla``。"""
    cwd = Path.cwd()
    if (cwd / SERVE_REL).is_file():
        return cwd
    tp = os.environ.get("SGEVAL_THIRD_PARTY")
    if tp and (Path(tp) / "mme-vla" / SERVE_REL).is_file():
        return Path(tp) / "mme-vla"
    raise FileNotFoundError(f"找不到三方 {SERVE_REL}（cwd={cwd}，SGEVAL_THIRD_PARTY={tp}）")


def prepare_sys_path(root: Path) -> None:
    """与 ``python scripts/serve_policy.py`` 对齐：``sys.path[0]`` 为 ``<root>/scripts``，本目录不在路径里。"""
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != _HERE]
    scripts = str(root / "scripts")
    if scripts in sys.path:
        sys.path.remove(scripts)
    sys.path.insert(0, scripts)


def load_serve_policy(root: Path):
    """按文件路径加载锁定的三方脚本（模块名 ``serve_policy``，``__name__`` 不是 ``__main__``，不会自行起服务）。"""
    path = root / SERVE_REL
    spec = importlib.util.spec_from_file_location("serve_policy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serve_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ints(x: Any) -> list[int]:
    return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]


def _bools(x: Any) -> list[bool]:
    return [bool(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]


class _EncodeSpy:
    """sentencepiece 处理器的透明代理：``encode`` 原样转调并记下 (输入文字, 返回 id)；其余属性直接转给原对象。"""

    def __init__(self, real: Any, log: list):
        self._real, self._log = real, log

    def encode(self, *a, **k):
        ids = self._real.encode(*a, **k)
        text = a[0] if a else k.get("input", k.get("text"))
        self._log.append((text, ids))
        return ids

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_tokenizer_spy(tokenizer_cls: Any) -> None:
    """给三方 ``PaligemmaTokenizer.tokenize`` 套观察（每个类只套一次）：没有正在进行的 infer 时原样直通；有时把实例
    的 sentencepiece 处理器临时换成 ``_EncodeSpy``，调用原方法、原样返回，记下一条通道。"""
    if getattr(tokenizer_cls, "_sgeval_spy", False):
        return
    orig = tokenizer_cls.tokenize

    def tokenize(self, *a, **k):
        sink = getattr(_ACTIVE, "sink", None)
        if sink is None:
            return orig(self, *a, **k)
        subgoal = k["subgoal"] if "subgoal" in k else (a[2] if len(a) > 2 else None)
        real = self._tokenizer
        log: list = []
        self._tokenizer = _EncodeSpy(real, log)
        try:
            tokens, mask = orig(self, *a, **k)
        finally:
            self._tokenizer = real
        raw_len = sum(len(ids) for _, ids in log)
        max_len = getattr(self, "_max_len", None)
        sink.append({"channel": "symbolic" if subgoal is not None else "task",
                     "text": "".join(str(t) for t, _ in log),
                     "token_ids": _ints(tokens), "mask": _bools(mask),
                     "tokenizer": f"paligemma_tokenizer.model max_len={max_len}",
                     "truncated": bool(max_len is not None and raw_len > int(max_len))})
        return tokens, mask

    tokenizer_cls.tokenize = tokenize
    tokenizer_cls._sgeval_spy = True


def build_audit(channels: list[dict]) -> dict:
    sym = [c for c in channels if c["channel"] == "symbolic"]
    task = [c for c in channels if c["channel"] == "task"]
    final = (sym or task or [{}])[-1].get("text")
    return {"channels": list(channels), "server_final_text": final, "pp_generation": None}


class AuditedPolicy:
    """三方策略的代理：``infer`` 调真实策略并原样返回结果，只多加审计键；``reset``／``add_buffer``／``metadata`` 等
    一律直接转给真实策略。"""

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def infer(self, obs: dict) -> dict:
        sink: list = []
        _ACTIVE.sink = sink
        try:
            out = self._inner.infer(obs)
        finally:
            _ACTIVE.sink = None
        out = dict(out)
        out[AUDIT_KEY] = build_audit(sink)
        return out


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def write_metadata(path: str, *, args: Any, argv_full: list[str], serve_file: Path | None) -> dict:
    doc = {"policy_seed": int(getattr(args, "seed")), "argv": list(argv_full), "pid": os.getpid(),
           "port": getattr(args, "port", None), "audit": audit_enabled(), "wrapper": "policy_server_wrap.py",
           "serve_policy": str(serve_file) if serve_file else None,
           "serve_policy_sha256": _sha256(serve_file) if serve_file else None}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return doc


def _default_tokenizer_cls():
    from mme_vla_suite.training.config import PaligemmaTokenizer

    return PaligemmaTokenizer


def run(sp: Any, args: Any, *, meta_path: str | None, argv_full: list[str], tokenizer_cls: Any = None,
        serve_file: Path | None = None) -> Any:
    """以三方 ``sp.main(args)`` 起服务；期间把 ``sp.create_policy`` 换成「原函数 + 写元数据 + （审计开时）包代理」，
    返回后还原。返回 ``sp.main`` 的返回值（真实服务永不返回）。"""
    audit = audit_enabled()
    orig_create = sp.create_policy

    def create_policy(a):
        policy = orig_create(a)
        if meta_path:
            write_metadata(meta_path, args=a, argv_full=argv_full, serve_file=serve_file)
        if not audit:
            return policy
        install_tokenizer_spy(tokenizer_cls if tokenizer_cls is not None else _default_tokenizer_cls())
        return AuditedPolicy(policy)

    sp.create_policy = create_policy
    try:
        return sp.main(args)
    finally:
        sp.create_policy = orig_create


def main(argv: list[str] | None = None) -> Any:
    full = list(sys.argv if argv is None else argv)
    meta, rest = split_wrapper_args(full[1:])
    root = mme_vla_root()
    prepare_sys_path(root)
    sp = load_serve_policy(root)
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)  # 同三方 __main__：先设日志再解析参数
    args = tyro.cli(sp.Args, args=rest)
    return run(sp, args, meta_path=meta, argv_full=full, serve_file=root / SERVE_REL)


if __name__ == "__main__":  # pragma: no cover - 真实服务只在 GPU 席位上起
    main()
