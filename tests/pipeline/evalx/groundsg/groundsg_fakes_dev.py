"""GroundSG（S3）测试的 dev 侧替身：在公开版 ``groundsg_fakes`` 之上补依赖 dev-scripts 的部分。

``env_client``（席位层 ``dev-scripts/gl/seat.py``）、原侧 ``official_hard_runner``（``dev-scripts/orig/``）、原侧驱动
``OrigSide`` 与两侧逐项比较 ``diffs``。公开版的全部名字经 ``from groundsg_fakes import *`` 原样重新导出，
dev 测试 ``import groundsg_fakes_dev as F`` 后用法不变；``monkeypatch.setattr(F, "trace_parts", ...)`` 换的是本模块的
``trace_parts``，``diffs`` 在本模块里按全局名取用，替换照常生效。
"""
from __future__ import annotations

from pathlib import Path

if __package__:  # 以 tests.pipeline.evalx.groundsg.groundsg_fakes_dev 导入时与包内的公开版配对
    from .groundsg_fakes import *  # noqa: F401,F403
    from .groundsg_fakes import (ADAPTER, MEMER, MEMER_ADAPTER, POLICY_SEED, QWENVL, FakeClient, FakeServer, FakeSwift,
                                 World, read_trace, trace_parts)
else:  # 测试目录在 sys.path 上、按顶层名导入时与顶层的公开版配对
    from groundsg_fakes import *  # noqa: F401,F403
    from groundsg_fakes import (ADAPTER, MEMER, MEMER_ADAPTER, POLICY_SEED, QWENVL, FakeClient, FakeServer, FakeSwift,
                                World, read_trace, trace_parts)
from tests._support.dev_loaders import load_script

# ---------------------------------------------------------------- 生产模块（dev-scripts）


def env_client():
    return load_script("eval-official/env_client.py")


def official_hard_runner():
    return load_script("eval-official/official_hard_runner.py")


# ---------------------------------------------------------------- 原侧驱动


class OrigSide:
    """原侧：``official_hard_runner``（官方 ``EnvRunner`` 摘取原文 + 假 builder、假服务、swift 替身）。"""

    def __init__(self, variant: str, max_steps: int, tmp: Path, world: World, *, port: int = 18120,
                 real_client: bool = False, policy_seed: int = POLICY_SEED, swift: FakeSwift | None = None,
                 server: FakeServer | None = None):
        self.variant, self.max_steps, self.tmp, self.world = variant, max_steps, tmp, world
        self.server = server or FakeServer()
        self.swift = swift or FakeSwift()
        self.ohr = official_hard_runner()
        factory = None if real_client else (lambda h, p, ep: FakeClient(self.server))
        self.ctx = self.ohr.make_context(variant, host="127.0.0.1", port=port, max_steps=max_steps,
                                         policy_seed=policy_seed, adapter=ADAPTER if variant == QWENVL else None,
                                         memer_adapter=MEMER_ADAPTER if variant == MEMER else None,
                                         builder_cls=world.official_builder_cls(), scratch_root=tmp,
                                         client_factory=factory, qwen_extra=self.swift.names)

    def run(self, ident: dict, *, attempt: int = 1) -> dict:
        return self.ohr.run_identity(self.ctx, ident, out=self.tmp, attempt=attempt)


# ---------------------------------------------------------------- 两侧逐项比较


def seq_diff(a: list, b: list) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def diffs(new_side, orig_side) -> dict:
    (new, wn, rn), (orig, wo, ro) = new_side, orig_side
    tn = trace_parts(read_trace(rn["trace_path"]))
    to = trace_parts(read_trace(ro["trace_path"]))
    payload = (seq_diff([(r["name"], r["sha256"], r["step"]) for r in tn["request"]],
                        [(r["name"], r["sha256"], r["step"]) for r in to["request"]])
               + seq_diff([x[:2] for x in new.server.log], [x[:2] for x in orig.server.log])
               + seq_diff(tn["response"], to["response"]) + seq_diff(tn["demo"], to["demo"])
               + seq_diff(tn["history"], to["history"]) + seq_diff(new.swift.requests, orig.swift.requests))
    acts_n = [a.tobytes() for e in wn.envs for a in e.actions]
    acts_o = [a.tobytes() for e in wo.envs for a in e.actions]
    exec_ = seq_diff(acts_n, acts_o) + seq_diff(tn["step"], to["step"])
    term = int((rn["status"], rn["steps"], rn["error"], rn["success_flag"])
               != (ro["status"], ro["exec_steps"], ro["error"], ro["success_flag"])) + seq_diff(tn["end"], to["end"])
    if new.variant == MEMER:  # MemER 的子目标模型请求另比附图逐张字节（关键帧 + 最近帧）
        payload += seq_diff([r["image_shas"] for r in new.swift.requests], [r["image_shas"] for r in orig.swift.requests])
    return {"payload": payload, "exec": exec_, "terminal": term, "n_req": len(tn["request"]), "n_exec": len(acts_n)}


__all__ = [n for n in dir() if not n.startswith("_")]
