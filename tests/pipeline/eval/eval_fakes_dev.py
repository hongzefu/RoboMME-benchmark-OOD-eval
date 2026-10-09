"""评估流水线测试的 dev 侧替身：在公开版 ``eval_fakes`` 之上补依赖 dev-scripts 的部分。

``env_client``（席位层 ``dev-scripts/gl/seat.py``）、``eval_report``、``eval_manifest`` 三个生产模块，以及建在它们之上的
席位客户端（常驻 Policy 替身、``seat_args``／``make_runner``／``run_rows``）、清单与报告、手写运行根 ``Stage``。
公开版的全部名字经 ``from eval_fakes import *`` 原样重新导出，dev 测试 ``import eval_fakes_dev as F`` 后用法不变。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

if __package__:  # 以 tests.pipeline.eval.eval_fakes_dev 导入时与包内的公开版配对
    from .eval_fakes import *  # noqa: F401,F403
    from .eval_fakes import FakePolicyServer, FakeRecorder, HybridBuilder, World, episode_mod, read_jsonl, sha_bytes
else:  # 测试目录在 sys.path 上、按顶层名导入时与顶层的公开版配对
    from eval_fakes import *  # noqa: F401,F403
    from eval_fakes import FakePolicyServer, FakeRecorder, HybridBuilder, World, episode_mod, read_jsonl, sha_bytes
from tests._support.dev_loaders import load_script


# ---------------------------------------------------------------- 生产模块（dev-scripts）


def env_client():
    return load_script("eval-official/env_client.py")


def eval_report():
    return load_script("eval-official/eval_report.py")


def eval_manifest():
    return load_script("eval-official/eval_manifest.py")


# ---------------------------------------------------------------- 席位客户端（拆仓后：常驻 Policy + 动态队列）
#
# 新 ``seat.py`` 的 ``SeatRunner`` 不再接收策略模块，而是经 ``policy_factory`` 拿一个常驻 ``Policy``，逐局调评估包的
# ``episode.run_episode``。这里给两种替身：
# - ``FakeSeatPolicy``：新接口的假模型，动作由 ``FakePolicyServer`` 按「本局 reset 后收到的帧 + 状态 + 指令」确定性生成；
#   环境 step 抛含 ``svulkan2`` 的异常、或推理连接断开，记基础设施错误（infra=True）；其余异常记普通错误；
# - ``ModulePolicy``：把旧的模块级客户端（``run_episode(session, identity, conn_info, recorder)``）包成 Policy，
#   旧用例的 ``framesamp_modul_policy``／``smvla_policy`` 照旧能喂给 ``make_runner``。
# builder 经 ``episode.BUILDER_FACTORY`` 换成 ``HybridBuilder``（真实身份解析 + CPU 假环境），``run_rows`` 收尾复原。

#: 假模型每次推理执行的动作行数
EXEC_ROWS = 5


def _policy_base():
    from robomme_ood_eval.policy import Policy

    return Policy


def _make_policy_classes():
    Policy = _policy_base()

    class FakeSeatPolicy(Policy):
        model = "fake-seat"

        def __init__(self, policy_seed: int = 7, server: "FakePolicyServer | None" = None, **cfg):
            super().__init__(policy_seed, **cfg)
            self.server = None  # 无真实服务端进程（ServerProcess）
            self.fake = server or FakePolicyServer()
            self.specs: list = []

        def reset(self, spec) -> None:
            super().reset(spec)
            self.specs.append(spec)
            self.fake.reset(spec.key)

        def play(self, session, spec, recorder) -> dict:
            obs, info = session.reset()
            goal = info.get("task_goal") if isinstance(info, dict) else None
            prompt = str(goal[0] if isinstance(goal, list) else goal)
            self.fake.observe([sha_bytes(f) for f in obs["front_rgb_list"]])
            steps = 0
            try:
                while True:
                    state = np.concatenate([obs["joint_state_list"][-1], obs["gripper_state_list"][-1][:1]])
                    chunk = self.fake.actions(state, prompt)
                    for row in chunk[:EXEC_ROWS]:
                        if not spec.strict_cap and steps > spec.max_steps:  # 官方循环 count > max_steps 才停
                            return {"status": "timeout", "task_success": 0, "steps": steps, "error": None,
                                    "infra": False, "infra_reason": None, "decisions": steps}
                        obs, _r, terminated, truncated, info = session.step(row)
                        steps += 1
                        self.fake.observe([sha_bytes(obs["front_rgb_list"][-1])])
                        if terminated or truncated:
                            st = info.get("status")
                            status = st if st in ("success", "fail") else "timeout"
                            return {"status": status, "task_success": int(status == "success"), "steps": steps,
                                    "error": None, "infra": False, "infra_reason": None, "decisions": steps}
            except Exception as e:  # noqa: BLE001
                from websockets.exceptions import ConnectionClosed

                from robomme_ood_eval.session import StepCapReached

                if isinstance(e, StepCapReached):
                    raise
                infra = isinstance(e, ConnectionClosed) or "svulkan2" in str(e)
                return {"status": "error", "task_success": 0, "steps": steps, "error": f"{type(e).__name__}: {e}",
                        "infra": infra, "infra_reason": "env_exception" if infra else None}

    class ModulePolicy(Policy):
        model = "module"

        def __init__(self, policy_seed: int = 7, mod=None, policy_name: str = "module", **cfg):
            super().__init__(policy_seed, **cfg)
            self.mod, self.policy_name = mod, policy_name
            self.ctx = None
            # 模块可带 conn_extra（如 GroundSG 的变体与 adapter），并入每局 conn_info 与 make_policy_context 的入参
            self.conn_extra = dict(getattr(mod, "conn_extra", None) or {})

        def play(self, session, spec, recorder) -> dict:
            make = getattr(self.mod, "make_policy_context", None)
            if self.ctx is None:
                self.ctx = make({"policy": self.policy_name, "max_steps": spec.max_steps, "host": "127.0.0.1",
                                 "port": 1, "policy_seed": self.policy_seed, "dataset": spec.dataset,
                                 **self.conn_extra}) if callable(make) else {}
            ident = dict(spec.identity())
            # 与 servers.ServedPolicy._conn_info 同口径：effective_cap 只在 strict 时给；轨迹落在本局 raw 目录
            conn_info = {"host": "127.0.0.1", "port": 1, "max_steps": spec.max_steps, "policy": self.policy_name,
                         "dataset": spec.dataset, "strict_cap": spec.strict_cap, "policy_seed": self.policy_seed,
                         "effective_cap": spec.max_steps if spec.strict_cap else None,
                         "trace_dir": spec.out_dir, "episode_tag": Path(spec.out_dir).name, "rec_dir": spec.out_dir,
                         "policy_context": self.ctx, **self.conn_extra}
            # 旧 SeatRunner 的口径：模块的 run_episode 接受 max_steps／reset_retries 关键字时显式传本局上限与 0 次 reset 重试
            # （SimpleMemVLA 的 hard_bound 由 max_steps 算）；只读 conn_info 的模块不传
            kw = {k: v for k, v in (("max_steps", spec.max_steps), ("reset_retries", 0))
                  if _accepts(self.mod.run_episode, k)}
            res = dict(self.mod.run_episode(session, ident, conn_info, recorder, **kw))
            res.setdefault("error", None)
            res.setdefault("infra", False)
            res.setdefault("infra_reason", None)
            res["task_success"] = int(bool(res.get("task_success")))
            res.setdefault("steps", 0)
            return res

    return FakeSeatPolicy, ModulePolicy


def _accepts(fn, name: str) -> bool:
    import inspect

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def fake_seat_policy(server: "FakePolicyServer | None" = None, seed: int = 7):
    return _make_policy_classes()[0](seed, server=server)


def as_policy(policy_obj, policy_name: str, seed: int = 7):
    """``make_runner`` 的策略参数：Policy 实例原样返回；旧模块（或带 ``run_episode`` 的命名空间）包成 ``ModulePolicy``。"""
    if isinstance(policy_obj, _policy_base()):
        return policy_obj
    return _make_policy_classes()[1](seed, mod=policy_obj, policy_name=policy_name)


def seat_args(out: Path, policy: str, *, seat: str = "s00", reset_budget: int | None = 100, infra_retries: int = 1,
              policy_seed: int = 7, dataset: str | None = None, budget_ledger=None, **kw) -> argparse.Namespace:
    """新 ``seat.py run`` 的参数（按其 parser 的缺省值补齐）；``out`` 为产物根。测试缺省给 1 次基础设施重试（每身份至多 2
    次尝试），以便覆盖重试路径；生产缺省为 0。"""
    ec = env_client()
    argv = ["run", "--policy", "dummy", "--identities", "x", "--out", str(out)]
    args, _ = ec.build_parser().parse_known_args(argv)
    args.policy = policy
    args.extra_cfg = {}
    args.seat = seat
    args.reset_budget = reset_budget
    args.infra_retries = infra_retries
    args.policy_seed = policy_seed
    args.dataset = dataset
    args.budget_ledger = budget_ledger if isinstance(budget_ledger, (str, Path)) else None
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def make_runner(stage: Path, policy: str, policy_obj, world: World, *, seat_dir: str = "s00",
                budget_ledger=None, recorder_factory=None, now=None, hard_exit=None, **kw):
    """新席位：产物根 ``<stage>``（其下 rollouts/、queue/、seats/）；builder 换成 ``HybridBuilder``，录制器换成
    ``FakeRecorder``，不出网站视频。``budget_ledger``：路径或已打开的账本对象（共享模式），None 即关闭。"""
    ec, E = env_client(), episode_mod()
    prev = E.BUILDER_FACTORY
    E.clear_builders()
    E.BUILDER_FACTORY = lambda task, ds, ms: HybridBuilder(task, ms, world, ds)
    args = seat_args(Path(stage), policy, seat=seat_dir, budget_ledger=budget_ledger, **kw)
    pol = as_policy(policy_obj, policy, seed=args.policy_seed)
    shared = budget_ledger if budget_ledger is not None else None
    runner = ec.SeatRunner(args, policy_factory=lambda model, seed, **cfg: pol,
                           episode_kwargs={"recorder_factory": recorder_factory
                                           or (lambda raw, meta: FakeRecorder(raw, meta, world)), "render": False},
                           hard_exit=hard_exit or (lambda code: None), shared=shared,
                           **({"now": now} if now is not None else {}))
    runner._fake_restore = prev
    runner.fake_policy = pol
    return runner


def run_rows(runner, rows: list[dict]) -> int:
    """返回生产退出码：``SeatRunner.run`` 的返回值（0／3／5／6）；收尾复原 ``BUILDER_FACTORY``。"""
    E = episode_mod()
    try:
        return int(runner.run(rows))
    finally:
        E.BUILDER_FACTORY = getattr(runner, "_fake_restore", None)
        E.clear_builders()


def seat_dir(stage: Path, label: str, seat: str = "s00", seed: int = 7) -> Path:
    return Path(stage) / "seats" / label / f"seed{seed}" / seat


def seat_ledger(stage: Path, label: str, seat: str = "s00", seed: int = 7) -> list[dict]:
    return read_jsonl(seat_dir(stage, label, seat, seed) / f"{label}.ledger.jsonl")


def seat_results(stage: Path, label: str, seat: str = "s00", seed: int = 7) -> list[dict]:
    return read_jsonl(seat_dir(stage, label, seat, seed) / "seat-results.jsonl")


def queue_dir(stage: Path, label: str, seed: int = 7) -> Path:
    return Path(stage) / "queue" / label / f"seed{seed}"


# ---------------------------------------------------------------- 清单与报告


def write_manifest(path: Path, rows: list[dict], *, shard: str = "00", **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = Counter(f"{r['task']}@{r['tier']}" for r in rows)
    doc = {"schema": eval_manifest().SCHEMA, "total": len(rows), "cells": dict(cells),
           "rows": [dict(r, shard=shard) for r in rows], **extra}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def run_report(capsys, manifest: Path, stage: Path, policies: list[str], out: Path, *extra: str) -> tuple[int, list[str], dict]:
    """进程内调用真实 ``eval_report.main``；返回 (退出码, 判定行, report.json)。"""
    er = eval_report()
    capsys.readouterr()
    rc = er.main(["--manifest", str(manifest), "--stage", str(stage), "--policies", ",".join(policies),
                  "--out", str(out), *extra])
    lines = [x for x in capsys.readouterr().out.splitlines() if "=" in x.split(" ")[0]]
    return rc, lines, json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))


def verdict(lines: list[str], name: str) -> dict[str, str]:
    """把 ``NAME=PASS k=v ...`` 解析成 {"": "PASS", k: v}。"""
    for line in lines:
        head, *rest = line.split()
        if head.startswith(name + "="):
            d = {"": head.split("=", 1)[1]}
            for kv in rest:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = v
            return d
    raise AssertionError(f"没有判定行 {name}：{lines}")


# ---------------------------------------------------------------- 手写运行根


class Stage:
    """按生产布局手写一个席位的结果行、账本行与录像目录。"""

    def __init__(self, root: Path, policy: str = "perceptual-framesamp-modul", seat: str = "s00", dirname: str | None = None):
        self.root, self.policy = Path(root), policy
        self.dir = self.root / seat / (dirname or policy)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, row: dict):
        with open(self.dir / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def result(self, ident: dict, aid: str, no: int, status: str, *, infra: bool = False, media: bool = True,
               **kw) -> dict:
        row = {"v8": True, "key": ident["key"], "task": ident["task"], "tier": ident["tier"], "seed": ident["seed"],
               "candidate": ident["candidate"], "spec_sha256": ident["spec_sha256"],
               "source_episode": ident.get("source_episode"),
               "identity": {k: ident[k] for k in ("tier", "seed", "candidate", "spec_sha256")},
               "policy": self.policy, "attempt_id": aid, "attempt_no": no, "status": status,
               "task_success": status == "success", "infra": infra, "exec_steps": 5,
               "rec_dir": str(self.dir / "rec" / f"{ident['key']}.a{no}"), "recorder_verify": "PASS"}
        row.update(kw)
        self._append("results.jsonl", row)
        if media:
            d = self.dir / "rec" / f"{ident['key']}.a{no}"
            d.mkdir(parents=True, exist_ok=True)
            for f in ("front.mkv", "wrist.mkv", "summary.json"):
                (d / f).write_text("x", encoding="utf-8")
        return row

    def ledger(self, kind: str, ident_or_key, aid: str, **kw):
        key = ident_or_key if isinstance(ident_or_key, str) else ident_or_key["key"]
        row = {"kind": kind, "key": key, "attempt_id": aid, "policy": self.policy, **kw}
        if kind == "accept":
            row.setdefault("accepted_attempt_id", aid)
        self._append(f"{self.policy}.ledger.jsonl", row)

    def accepted(self, ident: dict, aid: str, status: str, no: int = 1, **kw) -> dict:
        self.ledger("attempt_start", ident, aid, attempt_no=no)
        row = self.result(ident, aid, no, status, **kw)
        self.ledger("attempt_end", ident, aid, attempt_no=no, status=status)
        self.ledger("accept", ident, aid, status=status)
        return row


__all__ = [n for n in dir() if not n.startswith("_")] + ["REPO"]
