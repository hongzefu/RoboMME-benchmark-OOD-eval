"""3-tier Astra 独立入口的第三阶段功能（R5）：1800 步 strict cap、模型 seed、语言账本、数组合并写。

依据：``1006-rename-official-names-and-stage3-eval-plan.md`` 第二部分一（Astra 行）、八.3「1800 步的真实执行」、
八.11（Astra 行）；接口 ``docs/plans/1006-stage3-interface-freeze.md`` 2.1、2.2、四、五节。全部零外联、零费用、零 GPU：
环境、VLA、规划、监视都是替身；启动器用例的 VLA／仿真解释器是只记参数的桩。

判定行（由对应用例打印）：
- ``EVAL_CAP=PASS route=astra max_steps=1800 rejected_step=1801``
- ``POLICY_SEEDS=PASS route=astra seeds=0,7,42 server_seed_ok=3 trace_ok=3 cloud_seed=null``
- ``ASTRA_LANG_IO=PASS planner=<n> monitor=<n> action=<n> unresolved_steps=0 open_calls=0 image_ref_unresolved=0``

语言账本：``trace_writer.LanguageLog``（R6）在时用真实实现，不在时用 ``astra_fakes.SpecLanguageLog``（按冻结签名写的替身）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_fakes import (FakeEnv, FakeMonitor, FakeResponder, FakeVLA, NetCounter, astra_session, ensure_language_log,
                         make_args, make_deps, read_language, recording_builder_cls, template_text, third_party,
                         write_cases)
from tests.pipeline.evalx.report import trace_contract as tc

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "eval-official" / "run_astra.sh"


class _NullWriter:
    def append_data(self, frame):
        pass

    def close(self):
        pass


def _trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _a_dir(output: Path, task: str, ep: int) -> Path:
    dirs = [p for p in (Path(output) / task / f"ep{ep:03d}").iterdir() if p.is_dir() and ".a" in p.name]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _ctx(mod, tmp_path: Path, *, cap: int, strict: bool, lang=None):
    import trace_writer as tw
    writer = tw.TraceWriter(tmp_path / "trace.jsonl", route="astra/new",
                            identity={"task": "BinFill", "dataset": "ood", "key": "k", "attempt": 1}, max_steps=cap)
    return mod.TraceContext(writer, effective_cap=cap, strict_cap=strict, lang=lang), writer


# ── 1800 步 strict cap ───────────────────────────────────────────────────

def test_ood_strict_cap_step_1800_runs_and_1801_never_reaches_env(tmp_path):
    """ood（strict，cap 1800）：第 1800 步照常进真实环境并落 trace；第 1801 次 ``step`` 在计数与动作追加之前被拒，
    真实环境步数仍 1800、trace 不多一行、``cap_hit=True``、抛 ``StepCapReached``。"""
    with astra_session() as (mod, _astra):
        assert mod.DATASET_STEP_PAIRING == {"hard-verify": 1300, "ood": 1800}
        assert mod.DATASET_STRICT_CAP == {"hard-verify": False, "ood": True}
        ctx, writer = _ctx(mod, tmp_path, cap=1800, strict=True)
        inner = FakeEnv(terminal_step=None)
        env = mod.TracedEnv(inner, ctx)
        env.reset()
        action = np.zeros(8, dtype=np.float32)
        for _ in range(1800):
            env.step(action)
        rows_before = (tmp_path / "trace.jsonl").read_text()
        assert inner.steps_taken == 1800 and ctx.attempted == 1800 and not ctx.cap_hit
        with pytest.raises(mod.StepCapReached, match="STEP_CAP exec_steps=1800 cap=1800"):
            env.step(action)
        assert inner.steps_taken == 1800, "第 1801 步不得进入真实环境"
        assert ctx.attempted == 1800 and len(ctx.actions) == 1800 and ctx.cap_hit is True
        assert (tmp_path / "trace.jsonl").read_text() == rows_before, "被拒的第 1801 步不得多落一行"
        writer.close(status="timeout", terminal_reason="timeout")
    steps = [r for r in _trace(tmp_path / "trace.jsonl") if r["kind"] == "step"]
    assert [r["step"] for r in steps][-1] == 1800 and len(steps) == 1800


def test_hard_verify_is_not_strict(tmp_path):
    """hard-verify（cap 1300）保持非 strict：第 1301 次 step 照常交给底层环境（截断靠环境自己）；meta 记 strict_cap=False。"""
    with astra_session() as (mod, _astra):
        ctx, writer = _ctx(mod, tmp_path, cap=1300, strict=mod.strict_cap_of("hard-verify"))
        inner = FakeEnv(terminal_step=None)
        env = mod.TracedEnv(inner, ctx)
        env.reset()
        for _ in range(1301):
            env.step(np.zeros(8, dtype=np.float32))
        writer.close(status="timeout", terminal_reason="timeout")
        assert inner.steps_taken == 1301 and ctx.cap_hit is False
        args = SimpleNamespace(dataset="hard-verify", max_steps=1300, policy_seed=7)
        ident = {"tier": "xhard0", "seed": 1, "source_episode": 3}
        meta = mod.recorder_meta(args, "BinFill", 0, ident, "BinFill_xhard0_1", tmp_path / "m")
        assert meta["strict_cap"] is False and meta["effective_cap"] == 1300 and meta["policy_seed"] == 7
        args.dataset, args.max_steps = "ood", 1800
        meta = mod.recorder_meta(args, "BinFill", 0, ident, "BinFill_xhard0_1", tmp_path / "m")
        assert meta["strict_cap"] is True and meta["effective_cap"] == 1800 and meta["cloud_seed"] is None


def test_strict_cap_overrun_finishes_as_timeout(tmp_path, monkeypatch):
    """端到端：让 Astra 自己的循环以为上限是 1900（模拟循环越界），入口守卫在第 1801 步拒绝；本局按 timeout 收尾、
    ``cap_hit=True``、不计基础设施错误（照常写 PILOT_FINISHED），trace／arrays 恰 1800 步。"""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        envs = []

        def plan(builder, ep):
            envs.append(FakeEnv(terminal_step=None))
            return envs[-1]

        cls = recording_builder_cls(plan)
        doc = mod.prepare_cases(cls, "ood", ["VideoUnmask"], tier="xhard1", index=0)
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1800, policy_seed=7)
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        real_episode = astra.runner.episode

        def overrun(run_args, *rest):
            return real_episode(SimpleNamespace(**{**vars(run_args), "max_steps": 1900}), *rest)

        monkeypatch.setattr(astra.runner, "episode", overrun)
        deps = make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=FakeResponder(astra.champ),
                         check_calls=[])
        out = mod.run_cases(args, deps)
    (result,) = out["results"]
    assert envs[0].steps_taken == 1800
    assert result["status"] == "timeout" and result["cap_hit"] is True and result["exec_steps"] == 1800
    assert result["strict_cap"] is True and result["effective_cap"] == 1800
    assert (Path(args.output) / "PILOT_FINISHED.json").is_file()
    a_dir = _a_dir(Path(args.output), "VideoUnmask", doc["cases"][0]["episode"])
    rows = _trace(a_dir / "trace.jsonl")
    end = rows[-1]
    assert end["status"] == end["terminal_reason"] == "timeout" and end["cap_hit"] is True
    assert end["steps_attempted"] == end["exec_steps"] == 1800
    assert sum(1 for r in rows if r["kind"] == "step") == 1800
    with np.load(a_dir / "arrays.npz") as arr:
        acts = sorted(k for k in arr.files if k.startswith("exec_action__"))
        assert acts[-1] == "exec_action__01799" and len(acts) == 1800  # 第 1801 步被拒，不进数组
        assert sorted(k for k in arr.files if k.startswith("exec_state__"))[-1] == "exec_state__01799"
        assert len(arr.files) == 2 * 1800
    assert end["arrays"]["action_keys"] == end["arrays"]["state_keys"] == 1800 and "error" not in end["arrays"]
    tc.assert_renderable(a_dir)
    tc.assert_counts_consistent(a_dir, {"exec_steps": result["exec_steps"], "status": result["status"]})
    saved = json.loads((a_dir.parent / "result.json").read_text())
    assert saved["status"] == "timeout" and saved["cap_hit"] is True
    assert net.calls == 0
    print("EVAL_CAP=PASS route=astra max_steps=1800 rejected_step=1801")


# ── 模型 seed ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, "", "abc", "-1", "7.5", True])
def test_policy_seed_missing_or_bad_blocks(value):
    with astra_session() as (mod, _astra):
        with pytest.raises(ValueError, match="RUN_BLOCKED reason=policy_seed"):
            mod.check_policy_seed(value)
        assert [mod.check_policy_seed(v) for v in (0, "7", 42)] == [0, 7, 42]


def test_policy_seed_required_before_side_effects(tmp_path, capsys):
    """runner 两层：CLI ``check``／``run`` 缺 ``--policy-seed`` 退出 3；``run_cases`` 缺 seed 在建任何目录之前拒绝。"""
    with astra_session() as (mod, astra):
        cases = write_cases(tmp_path / "cases.json", {"dataset": "hard-verify", "cases": []})
        base = ["--cases", str(cases), "--vla-checkpoint", "v", "--monitor-adapter", "m", "--max-steps", "1300"]
        assert mod.main(["check", *base]) == 3
        assert "RUN_BLOCKED reason=policy_seed" in capsys.readouterr().err
        run_dir = tmp_path / "group_0" / "run"
        assert mod.main(["run", *base, "--output", str(run_dir / "results"), "--spool", str(run_dir / "planner_calls"),
                         "--guard-state", "g"]) == 3
        assert "RUN_BLOCKED reason=policy_seed" in capsys.readouterr().err
        assert not run_dir.exists()
        cls = recording_builder_cls()
        doc = mod.prepare_cases(cls, "hard-verify", ["BinFill"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "c2.json", doc), max_steps=1300, policy_seed=None)
        deps = make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=FakeResponder(astra.champ),
                         check_calls=[])
        with pytest.raises(ValueError, match="RUN_BLOCKED reason=policy_seed"):
            mod.run_cases(args, deps)
    assert not Path(args.output).exists() and not Path(args.spool).exists()


STUB_SIM = r"""#!/usr/bin/env bash
# 假 SIM_PYTHON：runner 调用只记参数；端口等待直接通过；其余（路径解析、包清单、服务元数据）交真解释器
for a in "$@"; do
  if [[ "$a" == *astra_hard_runner.py ]]; then printf '%s\n' "$*" >> "$STUB_LOG"; exit 0; fi
done
if [[ "${1:-}" == "-" && "${2:-}" =~ ^[0-9]+$ ]]; then cat >/dev/null; exit 0; fi
exec "$REAL_PY" "$@"
"""
STUB_VLA = r"""#!/usr/bin/env bash
# 假 VLA_PYTHON：包清单（stdin 脚本）忽略；服务启动只记 argv 后退出
if [[ "${1:-}" == "-" ]]; then cat >/dev/null; exit 0; fi
printf '%s\n' "$*" > "$VLA_LOG"
"""
STUB_LIB = "render_official_dir() { :; }\ntranscode_episode_dir() { :; }\n"


def _launch(tmp_path: Path, seed_args: list[str]) -> tuple[subprocess.CompletedProcess, Path, Path, Path]:
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    sim, vla, lib = stubs / "sim.sh", stubs / "vla.sh", stubs / "lib.sh"
    sim.write_text(STUB_SIM)
    vla.write_text(STUB_VLA)
    lib.write_text(STUB_LIB)
    sim.chmod(0o755)
    vla.chmod(0o755)
    astra_root = tmp_path / "astra-root"
    astra_root.mkdir(exist_ok=True)
    cases = write_cases(tmp_path / "cases.json", {"dataset": "ood", "cases": []})
    run = tmp_path / "group_0" / "run"
    log, vla_log = tmp_path / "runner-argv.log", tmp_path / "vla-argv.log"
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.update(VLA_PYTHON=str(vla), SIM_PYTHON=str(sim), REAL_PY=sys.executable, STUB_LOG=str(log),
               VLA_LOG=str(vla_log), VLA_CHECKPOINT=str(tmp_path / "ckpt"), MONITOR_BASE=str(tmp_path / "base"),
               MONITOR_ADAPTER=str(tmp_path / "adapter"), MAX_STEPS="1800",
               OPENAI_API_KEY="sk-test-placeholder-not-a-real-key", ASTRA_GUARD_STATE=str(tmp_path / "g.json"),
               ASTRA_ROOT=str(astra_root), SEAT_MEDIA_LIB=str(lib), VLA_GPU="0", MONITOR_GPU="1", PORT="18999")
    proc = subprocess.run(["bash", str(SCRIPT), *seed_args, str(cases), str(run)], capture_output=True, text=True,
                          env=env, timeout=120)
    return proc, run, log, vla_log


@pytest.mark.parametrize("seed_args", [[], ["--policy-seed", ""], ["--policy-seed", "abc"], ["--policy-seed=-1"]])
def test_launcher_policy_seed_missing_or_bad_exits_3(tmp_path, seed_args):
    """``run_astra.sh`` 缺 ``--policy-seed`` 或非法：退出 3、``RUN_BLOCKED reason=policy_seed``，不建 RUN、不起服务。"""
    proc, run, log, vla_log = _launch(tmp_path, seed_args)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "RUN_BLOCKED reason=policy_seed" in proc.stderr
    assert not run.exists() and not log.exists() and not vla_log.exists()


def test_policy_seeds_service_command_and_trace(tmp_path, monkeypatch):
    """seed 0／7／42：① 启动器把它作为 VLA 服务 ``--seed``（不再是 42 常量），check 与 run 都转发 ``--policy-seed``，
    服务元数据反查一致；② runner 把它记进 trace identity／end、``result.json``、provenance 与录制器 meta，
    ``cloud_seed`` 恒为 null（云端无 seed 接口，不伪造）。"""
    net = NetCounter().install(monkeypatch)
    server_ok = trace_ok = 0
    for seed in (0, 7, 42):
        sub = tmp_path / f"s{seed}"
        sub.mkdir()
        proc, run, log, vla_log = _launch(sub, ["--policy-seed", str(seed)])
        assert proc.returncode == 0, proc.stdout + proc.stderr
        argv = vla_log.read_text().split()
        assert argv[0] == "scripts/serve_policy.py" and f"--seed={seed}" in argv
        assert not any(a == "--seed=42" for a in argv) or seed == 42
        runner_calls = log.read_text().splitlines()
        check = [c for c in runner_calls if " check " in f" {c} "]
        run_call = [c for c in runner_calls if " run " in f" {c} "]
        assert len(check) == 1 and len(run_call) == 1
        for call in (check[0], run_call[0]):
            assert f"--policy-seed {seed}" in call
        meta = json.loads((run / "server-metadata-18999.json").read_text())
        assert meta["policy_seed"] == seed and meta["cloud_seed"] is None and f"--seed={seed}" in meta["argv"]
        assert meta["argv"][1:] == argv, "元数据 argv（首项为解释器）与实际服务命令逐项相同"
        assert f"policy_seed={seed} cloud_seed=null" in (run / "run-inputs.txt").read_text()
        server_ok += 1

        with astra_session() as (mod, astra):
            cls = recording_builder_cls(lambda b, ep: FakeEnv(terminal_step=20))
            doc = mod.prepare_cases(cls, "hard-verify", ["BinFill"], source_episodes=[3])
            args = make_args(sub, write_cases(sub / "c.json", doc), max_steps=1300, policy_seed=seed)
            monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
            deps = make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=FakeResponder(astra.champ),
                             check_calls=[])
            (result,) = mod.run_cases(args, deps)["results"]
        a_dir = _a_dir(Path(args.output), "BinFill", 0)
        rows = _trace(a_dir / "trace.jsonl")
        assert rows[0]["identity"]["policy_seed"] == seed
        if "policy_seed" in rows[0]:  # R6 的 TraceWriter 支持 header 字段时
            assert rows[0]["policy_seed"] == seed
        assert rows[-1]["policy_seed"] == seed and rows[-1]["cloud_seed"] is None
        assert rows[-1]["strict_cap"] is False and rows[-1]["effective_cap"] == 1300
        assert result["policy_seed"] == seed and result["cloud_seed"] is None
        saved = json.loads((a_dir.parent / "result.json").read_text())
        assert saved["policy_seed"] == seed and saved["cloud_seed"] is None
        prov = json.loads((a_dir / "provenance.json").read_text())
        assert prov["policy_seed"] == prov["server_seed"] == seed and prov["cloud_seed"] is None
        rmeta = json.loads((a_dir / "media" / "meta.json").read_text())
        assert rmeta["policy_seed"] == seed and rmeta["strict_cap"] is False
        trace_ok += 1
    assert net.calls == 0
    print(f"POLICY_SEEDS=PASS route=astra seeds=0,7,42 server_seed_ok={server_ok} trace_ok={trace_ok} cloud_seed=null")


# ── 语言账本 ─────────────────────────────────────────────────────────────

def _calls(rows: list[dict]) -> dict:
    """按 call_id 聚合：open、messages、close。"""
    calls: dict = {}
    for r in rows:
        if r["kind"] == "call_open":
            calls[r["call_id"]] = {"open": r, "messages": [], "close": None}
        elif r["kind"] == "message":
            calls[r["call_id"]]["messages"].append(r)
        elif r["kind"] == "call_close":
            calls[r["call_id"]]["close"] = r
    return calls


def _known_shas(rows: list[dict]) -> tuple[dict, set]:
    """trace 里可解析的画面 sha：exec 帧 k（0 = demo 末帧，k = step k）与 demo 帧 i，外加所有 wrist sha。"""
    demo = next(r for r in rows if r["kind"] == "demo")
    exec_front = {0: demo["front_sha256"][-1]}
    for r in rows:
        if r["kind"] == "step":
            exec_front[r["step"]] = r["front_sha256"]
    wrists = set(demo["wrist_sha256"]) | {r["wrist_sha256"] for r in rows if r["kind"] == "step"}
    demo_front = dict(enumerate(demo["front_sha256"][:-1]))
    return {"exec": exec_front, "demo": demo_front}, wrists


def _unresolved_images(images: list, fronts: dict, wrists: set) -> int:
    bad = 0
    for img in images:
        for src in img["sources"]:
            if src.get("cam") == "wrist":
                ok = src.get("raw_sha256") in wrists
            else:
                ok = fronts[src["phase"]].get(src["frame_idx"]) == src.get("raw_sha256")
            bad += 0 if ok and src.get("raw_sha256") else 1
    return bad


def test_language_log_planner_monitor_action(tmp_path, monkeypatch):
    """两局（VideoUnmask 有演示拼图、ButtonUnmask 有执行记忆拼图）：planner／monitor／action_model 三类调用齐全、
    输入原文与 spool／上游输入契约逐字相同、附图引用全部能在 trace 里解析、每个执行步可追溯到 action_model 调用、
    第二次规划回不合模板的原文时记 ``fallback=continue_last``；来源清单含 prompts 的 sha256。"""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        kind = ensure_language_log(monkeypatch)
        import input_contract
        tasks = ["VideoUnmask", "ButtonUnmask"]
        cls = recording_builder_cls(lambda b, ep: FakeEnv(terminal_step=60))
        doc = mod.prepare_cases(cls, "hard-verify", tasks, source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        texts = [template_text(astra.champ, "VideoUnmask"), "I cannot comply.",
                 template_text(astra.champ, "ButtonUnmask"), "I cannot comply."]
        monitor = FakeMonitor(predictions=[False, True, False, True])
        deps = make_deps(astra, cls, monitor=monitor, vla=FakeVLA(),
                         responder=FakeResponder(astra.champ, texts=texts), check_calls=[])
        out = mod.run_cases(args, deps)
        normal = input_contract.NORMAL
    assert [r["status"] for r in out["results"]] == ["success", "success"]
    totals = {"planner": 0, "monitor": 0, "action_model": 0}
    unresolved_steps = open_calls = image_bad = 0
    spool = Path(args.spool)
    for task, result in zip(tasks, out["results"]):
        a_dir = _a_dir(Path(args.output), task, 0)
        rows = _trace(a_dir / "trace.jsonl")
        lang_rows = read_language(a_dir / "language.jsonl")
        calls = _calls(lang_rows)
        fronts, wrists = _known_shas(rows)
        by_model: dict = {}
        for cid, c in calls.items():
            by_model.setdefault(c["open"]["model"], []).append(c)
            open_calls += c["close"] is None
            ins = [m for m in c["messages"] if m["dir"] == "in"]
            assert ins, f"调用 {cid} 没有输入"
            # 输入必须早于收尾落盘
            pos_in = lang_rows.index(ins[-1])
            assert c["close"] is None or pos_in < lang_rows.index(c["close"])
            for m in ins:
                image_bad += _unresolved_images(m.get("images") or [], fronts, wrists)
        # planner：2 次（首次规划 + monitor 触发重规划，后者回非模板 → continue_last）
        planner = by_model["planner"]
        assert len(planner) == result["planner_calls"] == 2
        first, second = planner
        (msg_in,) = [m for m in first["messages"] if m["dir"] == "in"]
        rid = first["open"]["params"]["request_id"]
        assert msg_in["role"] == "user" and msg_in["text"] == (spool / rid / "prompt.txt").read_text()
        req = json.loads((spool / rid / "request.json").read_text())
        assert [i["slot"] for i in msg_in["images"]] == list(range(len(req["images"])))
        assert msg_in["images"][0]["ref"] == "current" and msg_in["images"][1]["ref"] == "keyframe"
        assert first["open"]["params"]["cloud_seed"] is None and first["open"]["transport_attempt"] == 0
        assert first["close"]["status"] == "reply" and first["close"]["parsed"] == texts[0 if task == tasks[0] else 2]
        assert first["close"]["fallback"] is None
        assert [m["text"] for m in second["messages"] if m["dir"] == "out"] == ["I cannot comply."]
        assert second["close"]["parsed"] is None and second["close"]["fallback"] == "continue_last"
        refs = {i["ref"] for c in planner for m in c["messages"] for i in (m.get("images") or [])}
        assert ("demo_sheet" in refs) == (task == "VideoUnmask") and ("memory_sheet" in refs) == (task == "ButtonUnmask")
        # monitor：每次 system + user（10 张图）+ 回复 true/false
        monitors = by_model["monitor"]
        assert len(monitors) == result["monitor_calls"] and monitors
        for c in monitors:
            sys_m, user_m = [m for m in c["messages"] if m["dir"] == "in"]
            assert sys_m["role"] == "system" and sys_m["text"] == normal and sys_m["message_index"] == 0
            assert user_m["role"] == "user" and len(user_m["images"]) == 10
            assert [i["ref"] for i in user_m["images"]] == ["recent"] * 8 + ["command_start", "wrist"]
            assert "Current grounded subgoal:" in user_m["text"]
            (reply,) = [m for m in c["messages"] if m["dir"] == "out"]
            assert reply["text"] in ("true", "false") and c["close"]["parsed"] == (reply["text"] == "true")
        # action_model：每个推理步一个调用，字段原文
        actions = by_model["action_model"]
        assert len(actions) == sum(1 for r in rows if r["kind"] == "request" and r["name"] == "vla_infer")
        goal = next(r for r in rows if r["kind"] == "demo")["texts"][0]
        for c in actions:
            (m,) = c["messages"]
            assert m["role"] == "fields" and m["text"]["prompt"] == goal and m["text"]["grounded_subgoal"]
            assert c["close"]["status"] == "reply" and c["close"]["server_final_text"] is None
        action_ids = {c["open"]["call_id"] for c in actions}
        for r in rows:
            if r["kind"] == "step":
                ok = r.get("source_call_id") in action_ids and 0 <= r.get("chunk_index", -1) < 16
                unresolved_steps += 0 if ok else 1
        for k in totals:
            totals[k] += len(by_model.get(k, []))
        # 来源清单：prompts/*.md 与 index.json 的 sha256
        prov = json.loads((a_dir / "provenance.json").read_text())
        champ = third_party() / "examples" / "champ"
        assert prov["prompts"]["prompts/index.json"] == mod.file_sha256(champ / "prompts" / "index.json")
        assert prov["prompts"][f"prompts/{task}.md"] == mod.file_sha256(champ / "prompts" / f"{task}.md")
        assert prov["language_log"] == "language.jsonl" and rows[-1]["language"] == "language.jsonl"
        tc.assert_renderable(a_dir)
    assert unresolved_steps == 0 and open_calls == 0 and image_bad == 0
    assert net.calls == 0
    print(f"ASTRA_LANG_IO=PASS planner={totals['planner']} monitor={totals['monitor']} action={totals['action_model']} "
          f"unresolved_steps={unresolved_steps} open_calls={open_calls} image_ref_unresolved={image_bad} log={kind}")


def test_language_input_persisted_before_send_failure(tmp_path, monkeypatch):
    """假 planner 在「发送时」抛异常（不写 response.json）：prompt 全文与附图引用已先落盘，调用以 ``status=error`` 收尾；
    分片照上游第⑤项停（Planner API 前缀），没有任何外联。"""
    net = NetCounter().install(monkeypatch)
    with astra_session() as (mod, astra):
        ensure_language_log(monkeypatch)
        cls = recording_builder_cls(lambda b, ep: FakeEnv(terminal_step=40))
        doc = mod.prepare_cases(cls, "hard-verify", ["BinFill"], source_episodes=[3])
        args = make_args(tmp_path, write_cases(tmp_path / "cases.json", doc), max_steps=1300)
        monkeypatch.setattr(astra.runner.imageio, "get_writer", lambda *a, **k: _NullWriter())
        responder = FakeResponder(astra.champ, raise_on_send=RuntimeError("Planner API fake transport failure"))
        deps = make_deps(astra, cls, monitor=FakeMonitor(), vla=FakeVLA(), responder=responder, check_calls=[])
        with pytest.raises(mod.AstraStop) as stop:
            mod.run_cases(args, deps)
        assert stop.value.reason == "planner_error"
    a_dir = _a_dir(Path(args.output), "BinFill", 0)
    calls = _calls(read_language(a_dir / "language.jsonl"))
    (call,) = [c for c in calls.values() if c["open"]["model"] == "planner"]
    (msg_in,) = call["messages"]
    rid = call["open"]["params"]["request_id"]
    assert msg_in["dir"] == "in" and msg_in["text"] == (Path(args.spool) / rid / "prompt.txt").read_text()
    assert msg_in["images"] and not (Path(args.spool) / rid / "response.json").exists()
    assert call["close"]["status"] == "error"
    assert net.calls == 0


def test_transport_retry_opens_new_call_with_attempt(tmp_path, monkeypatch):
    """真实 ``GuardedResponsesClient``（零外联替身）遇 429 后重试成功：语言账本两个 planner 调用，
    ``transport_attempt`` 0（error）与 1（reply），输入原文相同；费用守卫行为不变（urlopen 恰 2 次、同一份预留）。"""
    import urllib.request
    from test_astra_wiring import Clock, FakeUrlopen, GuardFixture, _client, _http_429, _resp
    fake = FakeUrlopen([_http_429(), {"input_tokens": 10}])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fx = GuardFixture(tmp_path)
    fx.round()
    clock = Clock()
    with astra_session() as (mod, astra):
        ensure_language_log(monkeypatch)
        import trace_writer as tw
        client, _gate = _client(mod, astra, fx, clock)
        lang = tw.LanguageLog(tmp_path / "lang" / "language.jsonl")
        ctx = mod.TraceContext(None, lang=lang)
        call = mod.PlannerLanguageCall(ctx, lambda out, req: [], "predict")
        out = fx.request(1000)
        call.wrap(client)(out)
        call.finish(parsed="x", fallback=None)
        lang.close()
        assert client.transport_hook is None, "调用结束后回调还原"
    assert fake.calls == 2 and _resp(out)["status"] == "ok"
    assert list(fx.reservations()) == [out.name]
    calls = list(_calls(read_language(tmp_path / "lang" / "language.jsonl")).values())
    assert [c["open"]["transport_attempt"] for c in calls] == [0, 1]
    assert [c["close"]["status"] for c in calls] == ["error", "reply"]
    ins = [[m["text"] for m in c["messages"] if m["dir"] == "in"] for c in calls]
    assert ins[0] == ins[1] == [(out / "prompt.txt").read_text()]
    assert [m["text"] for m in calls[1]["messages"] if m["dir"] == "out"] == ["ok"]


# ── arrays.npz 合并写 ─────────────────────────────────────────────────────

def test_write_exec_actions_uses_merge_write_npz_when_present(tmp_path, monkeypatch):
    """``trace_writer.merge_write_npz`` 存在时 ``write_exec_actions`` 只经它写（键 ``exec_action__%05d``）；
    不存在时保持旧的 ``np.savez``。"""
    with astra_session() as (mod, _astra):
        import trace_writer as tw
        actions = [np.full(8, i, dtype=np.float32) for i in range(3)]
        seen = []
        real = getattr(tw, "merge_write_npz", None)

        def spy(path, mapping):
            seen.append((Path(path), sorted(mapping)))
            if real is not None:
                return real(path, mapping)
            np.savez(path, **mapping)

        monkeypatch.setattr(tw, "merge_write_npz", spy, raising=False)
        (tmp_path / "a").mkdir()
        mod.write_exec_actions(tmp_path / "a" / "arrays.npz", actions)
        assert seen == [(tmp_path / "a" / "arrays.npz", [f"exec_action__{i:05d}" for i in range(3)])]
        monkeypatch.delattr(tw, "merge_write_npz", raising=False)
        (tmp_path / "b").mkdir()
        mod.write_exec_actions(tmp_path / "b" / "arrays.npz", actions)
        with np.load(tmp_path / "b" / "arrays.npz") as arr:
            assert sorted(arr.files) == [f"exec_action__{i:05d}" for i in range(3)]
            assert arr["exec_action__00002"].dtype == np.float32
