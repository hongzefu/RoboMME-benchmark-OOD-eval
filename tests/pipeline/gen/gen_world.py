"""生成链路测试的公共世界：唯一一份 FakeRunner、CPU 抽签替身、微型 h5 与经真实 ``_freeze.freeze`` 封存的小规格根。

计划原写 ``tests/_support/gen_world.py``；``_support`` 归主会话，本块按分配表放在 ``tests/pipeline/gen/``。

设计口径：
- 生产模块一律经 ``generate_h5.py``／``freeze_specs.py`` 的模块属性取得（它们按普通模块名导入 ``_rollout``、
  ``_freeze``、``_draw``），测试与生产拿到的是同一个模块对象，打补丁时不会出现两份副本。
- 抽签替身只替换 ``_draw.draw_task`` 的 ``draw_one`` 注入点（生产预留的单测入口），抽签循环、合并、封存、签名、
  落盘全部走真实代码。
- ``FakeRunner`` 冒充 ``train_split_runner.py`` 的命令行契约：读 ``--jobs-json``，按剧本在局目录写真实 h5／mp4／
  ``spec_replay.json``，逐局追加 ``results.partial.jsonl``，最后按模式写或不写 ``results.json``；作为
  ``subprocess.run`` 的替身装进 ``_rollout``，``run_batch`` 与 ``run_continue`` 的解析、记账全部是真实代码。
- 剧本里的「期望」由测试手写（哪局失败、递补到谁），不从被测函数推出。
"""
from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tests._support.loaders import load_script

# ── 生产模块（同一对象）──────────────────────────────────────────────

GH = load_script("injection-dev/generate_h5.py")
FS = load_script("injection-dev/freeze_specs.py")
R = GH._rollout
F = FS._freeze
D = FS._draw
H = R.hard_specs
assert FS._rollout is R, "freeze_specs 与 generate_h5 须共用同一个 _rollout 模块对象"

SPEC_KIND = "native-newvalue/2"


# ── 抽签替身 ────────────────────────────────────────────────────────


def fake_spec(task: str, tier: str, episode: int, attempt: int, way: int | None = None) -> dict[str, Any]:
    """一条形如 SpecRecorder.to_dict() 的规格：构造期 initializations.0 与正式 reset 的最大序号；
    ``way`` 只写在最大序号那次（_freeze._movecube_way 的口径），构造期故意写另一个值以检出取错序号。"""
    spec: dict[str, Any] = {"spec_kind": SPEC_KIND,
                            "identity": {"task": task, "difficulty": tier, "episode": episode, "attempt": attempt},
                            "objects": {"probe": episode * 10 + attempt}, "actions": {}}
    if way is not None:
        spec["initializations"] = {"0": {"way_idx": (way + 1) % 3}, "1": {"way_idx": way}}
    return spec


class FakeDraw:
    """``draw_one(task, seed, episode, sampling)`` 替身：``fail[(task, episode)]`` 给出该局前若干次 attempt 失败；
    ``ways[task]`` 是 ``episode → way`` 函数（MoveCube 用）。逐次调用记在 ``calls``。"""

    def __init__(self, tier: str, fail: dict[tuple[str, int], int] | None = None, ways: dict | None = None,
                 seed_rule: dict | None = None):
        self.tier = tier
        self.fail = dict(fail or {})
        self.ways = dict(ways or {})
        self.seed_rule = seed_rule
        self.calls: list[tuple[str, int, int, int]] = []
        self._attempt: dict[tuple[str, int], int] = {}

    def __call__(self, task, seed, episode, sampling):
        attempt = self._attempt.get((task, episode), 0)
        self._attempt[(task, episode)] = attempt + 1
        self.calls.append((task, seed, episode, attempt))
        if attempt < self.fail.get((task, episode), 0):
            return False, None, "ResetRejected", "替身：布局不可行"
        way = self.ways[task](episode) if task in self.ways else None
        return True, fake_spec(task, self.tier, episode, attempt, way), None, None


def header_parts(tier: str, tasks: list[str], drafts: list[dict], run_id: str = "t7") -> dict[str, Any]:
    return {"difficulty": tier, "tasks": list(tasks), "seed_rule": H.seed_rule_for(tier, "v8"),
            "sampling_config": {t: {"decision": {"t7": True}, "native": {"task": t}} for t in tasks},
            "recovery_rule": {"rule": "t7 替身"}, "identity_source": "formula", "run_id": run_id,
            "draw_stats": D.draw_stats(drafts), "provenance": {"env_package": "robomme_hard", "note": "t7"}}


def draw(tier: str, plan: dict[str, tuple[int, int]], *, fail=None, ways=None, max_resets: int = 1000):
    """``plan = {task: (配额, 候选数)}`` → 经真实 ``_draw.draw_task`` 循环抽出的草稿行与替身。"""
    fake = FakeDraw(tier, fail, ways)
    drafts: list[dict] = []
    for task, (_quota, cands) in plan.items():
        drafts += D.draw_task(task, {}, cands, max_resets, fake, tier, H.seed_rule_for(tier, "v8"))
    return drafts, fake


def freeze_file(path: Path, tier: str, plan: dict[str, tuple[int, int]], *, fail=None, ways=None,
                select=None, run_id: str = "t7") -> tuple[dict, list[dict]]:
    """抽签 → 真实 ``_freeze.freeze`` → 真实 ``write_jsonl_exclusive`` 落一份 /4 规格文件。"""
    drafts, _ = draw(tier, plan, fail=fail, ways=ways)
    tasks = list(plan)
    if select is None:
        select = {t: tuple(range(plan[t][0])) for t in tasks}
    header, rows = F.freeze(drafts, header_parts(tier, tasks, drafts, run_id), select,
                            {t: plan[t][1] for t in tasks}, schema=H.SCHEMA)
    F.write_jsonl_exclusive(Path(path), [header, *rows])
    return header, rows


def freeze_root(root: Path, cells: dict[tuple[str, str], tuple[int, int]], **kw) -> Path:
    """``cells = {(task, tier): (配额, 候选数)}`` → ``<root>/<tier>/specs.jsonl``（每档一份）。"""
    by_tier: dict[str, dict[str, tuple[int, int]]] = {}
    for (task, tier), qc in cells.items():
        by_tier.setdefault(tier, {})[task] = qc
    for tier, plan in by_tier.items():
        ordered = {t: plan[t] for t in H.ALL_TASKS if t in plan}
        freeze_file(Path(root) / tier / "specs.jsonl", tier, ordered, **kw)
    return Path(root)


def read_rows(path: Path) -> tuple[dict, list[dict]]:
    records = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return records[0], records[1:]


def state(path: Path) -> dict[tuple[str, int], tuple[bool, bool, str | None]]:
    """规格文件 → ``{(task, candidate): (selected, tried, rollout.status)}``（断言用的紧凑视图）。"""
    _, rows = read_rows(path)
    return {(r["task"], r["candidate"]): (r["selected"], r["tried"], (r["rollout"] or {}).get("status")) for r in rows}


# ── 微型 h5 ────────────────────────────────────────────────────────


def write_h5(path: Path, frames: int, demo: int, subgoals: list[str] | None = None, goal: list[str] | None = None) -> Path:
    """录像器同形态的最小 h5：``episode_0/timestep_<k>/info/{is_video_demo,simple_subgoal}``，前 ``demo`` 帧为演示。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("episode_0")
        for k in range(frames):
            info = g.create_group(f"timestep_{k}/info")
            info.create_dataset("is_video_demo", data=np.bool_(k < demo))
            text = subgoals[k] if subgoals else "move"
            info.create_dataset("simple_subgoal", data=text.encode())
        if goal is not None:
            setup = g.create_group("setup")
            setup.create_dataset("task_goal", data=[x.encode() for x in goal])
            setup.create_dataset("difficulty", data=b"t7")
    return path


# ── FakeRunner ─────────────────────────────────────────────────────


def install_run(monkeypatch, run) -> None:
    """把 ``_rollout`` 模块里的 ``subprocess`` 换成只带 ``run``／``STDOUT`` 的替身命名空间。"""
    monkeypatch.setattr(R, "subprocess", types.SimpleNamespace(run=run, STDOUT=subprocess.STDOUT))


class Crash(Exception):
    """替身模拟驱动进程在一轮中途被杀（run_continue 之外的中断）。"""


OUTCOME_KINDS = ("ok", "fail", "infra", "vulkan", "none")


class FakeRunner:
    """``train_split_runner.py`` 的命令行替身（装为 ``_rollout.subprocess.run``）。

    ``script[(task, episode)]`` 是该身份逐次执行的结局列表，每项为 ``"ok"``／``("ok", 帧数, 演示帧)``／``"fail"``
    （任务失败）／``"infra"``（BrokenProcessPool）／``"vulkan"``（错误文本含 svulkan2）／``"none"``（不出结果）；
    列表用尽后按 ``"ok"``。``output``：``"json"`` 写 ``results.json``（同时逐局写 partial）；``"partial"`` 只写 partial；
    ``"nothing"`` 什么都不写。``crash_on_call``：第 n 次（从 1 计）调用在写完前 ``crash_after_jobs`` 局的 partial 后
    抛 ``Crash``，模拟驱动中途被杀。``executions`` 逐局记 ``(task, episode, 结局)``——每条就是一次真实预算消耗。"""

    def __init__(self, script: dict | None = None, *, output: str = "json", frames: int = 5, demo: int = 1,
                 crash_on_call: int | None = None, crash_after_jobs: int = 0):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.output = output
        self.frames, self.demo = frames, demo
        self.crash_on_call, self.crash_after_jobs = crash_on_call, crash_after_jobs
        self.calls: list[list[str]] = []
        self.executions: list[tuple[str, int, str]] = []
        self.envs: list[dict] = []

    def install(self, monkeypatch) -> "FakeRunner":
        """只替换 ``_rollout`` 看到的 ``subprocess``（不动全局 subprocess 模块）。"""
        install_run(monkeypatch, self.run)
        return self

    def _next(self, key):
        seq = self.script.get(key)
        return seq.pop(0) if seq else "ok"

    def run(self, argv, *, text=None, stdout=None, stderr=None, env=None, **_kw):
        self.calls.append(list(argv))
        self.envs.append(dict(env or {}))
        arg = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if str(argv[i]).startswith("--")}
        jobs = json.loads(Path(arg["--jobs-json"]).read_text(encoding="utf-8"))
        results_json = Path(arg["--results-json"])
        partial = results_json.with_name("results.partial.jsonl")
        done: dict[tuple[str, int], dict] = {}
        if "--resume" in argv and partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    done[(rec["task"], int(rec["episode"]))] = rec
        results = list(done.values())
        crash_now = self.crash_on_call is not None and len(self.calls) == self.crash_on_call
        for n, job in enumerate(jobs):
            key = (job["task"], int(job["episode"]))
            if key in done:
                continue
            if crash_now and n >= self.crash_after_jobs:
                raise Crash(f"替身：第 {len(self.calls)} 次调用在第 {n} 局前中断")
            outcome = self._next(key)
            kind = outcome[0] if isinstance(outcome, tuple) else outcome
            assert kind in OUTCOME_KINDS, outcome
            self.executions.append((key[0], key[1], kind))
            if kind == "none":
                continue
            rec = self._execute(job, outcome)
            results.append(rec)
            with partial.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        if stdout is not None:
            stdout.write(f"RUNNER_DONE jobs={len(results)}\n")
        if self.output == "json":
            results_json.write_text(json.dumps({"schema": "train-parity-runner-results/1", "results": results}),
                                    encoding="utf-8")
        elif self.output == "nothing" and partial.exists():
            partial.unlink()
        ok = all(r.get("ok") for r in results)
        return types.SimpleNamespace(returncode=0 if ok else 1, args=argv)

    def _execute(self, job: dict, outcome) -> dict:
        kind = outcome[0] if isinstance(outcome, tuple) else outcome
        base = {"task": job["task"], "episode": int(job["episode"]), "seed": int(job["seed"]),
                "difficulty": job["difficulty"], "env_package": "robomme_hard",
                "env_module": f"robomme_hard.robomme_env.{job['task']}",
                "wrapper_modules": {"RobommeRecordWrapper": "robomme_hard.env_record_wrapper.RecordWrapper"}}
        wdir = Path(job["worker_dir"])
        if kind == "fail":
            return {**base, "ok": False, "error_type": "TaskFailed", "error": "替身：任务未完成"}
        if kind == "infra":
            return {**base, "ok": False, "error_type": "BrokenProcessPool", "error": "替身：进程池崩溃"}
        if kind == "vulkan":
            return {**base, "ok": False, "error_type": "RuntimeError", "error": "替身：svulkan2 device lost"}
        frames, demo = (outcome[1], outcome[2]) if isinstance(outcome, tuple) else (self.frames, self.demo)
        name = f"{job['task']}_ep{job['episode']}_seed{job['seed']}"
        write_h5(wdir / "hdf5_files" / f"{name}.h5", frames, demo)
        (wdir / "videos").mkdir(parents=True, exist_ok=True)
        (wdir / "videos" / f"success_{job['task']}_seed{job['seed']}_x.mp4").write_bytes(b"\x00t7mp4")
        (wdir / "spec_replay.json").write_text(json.dumps(
            {"mismatches": [], "unused": [], "value_points": 3, "layout_hit": 1}), encoding="utf-8")
        return {**base, "ok": True, "error_type": None, "error": None}


def continue_kwargs(tmp: Path) -> dict[str, Any]:
    """run_continue／run_continue_v8 的公共参数（真实 run_batch，runner 由 FakeRunner 冒充）。"""
    return {"src_root": Path(tmp), "workers": 1, "gpu": "0", "pkg": "robomme_hard", "code_baseline": "t7-baseline"}
