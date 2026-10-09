"""GroundSG 原侧驱动 ``official_hard_runner.py``（1003 评估计划 1.3、1.7，子任务 S3；判定行 OFFICIAL_HARD_SOURCE）。

导入断言在子进程里跑（本测试进程已导入 robomme_hard，不能代表原侧进程）；分片循环、原始帧格式、服务不可达在进程内
用官方 ``EnvRunner`` 原文 + 官方 builder 替身跑，不建真实仿真。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

import groundsg_fakes as F
from tests._support.loaders import REPO, script_path

RUNNER = script_path("eval-official/official_hard_runner.py")
#: 第三阶段必填项（接口冻结说明 2.2、2.4；本轮取值 870／50／50／821）
BUDGET_ARGV = ["--budget-ledger", "/nonexistent/ledger.jsonl", "--trajectory-cap", "870", "--shared-infra-cap", "50",
               "--expired-cap", "50", "--planned-first-tries", "821"]
STAGE3_ARGS = ["--policy-seed", "7", *BUDGET_ARGV]


def _run(*argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run([sys.executable, str(RUNNER), *argv], capture_output=True, text=True, env=env, timeout=240,
                          cwd=str(REPO))


def _line(text: str, prefix: str) -> str:
    lines = [x for x in text.splitlines() if x.startswith(prefix)]
    assert lines, text[-3000:]
    return lines[-1]


def test_official_hard_source_imports_only_official_robomme():
    F.print_official_sha()
    p = _run("--check-imports")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    line = _line(p.stdout, "OFFICIAL_IMPORTS=")
    kv = dict(x.split("=", 1) for x in line.split())
    assert kv["OFFICIAL_IMPORTS"] == "PASS" and kv["robomme_hard_imported"] == "0"
    assert kv["official_modules_imported"] == "0" and kv["variants"] == "3"
    # 拆仓后 robomme 只在 benchmark 子模块里（评估仓没有 src/robomme 副本）；原侧驱动只往 sys.path 加官方目录与子模块 src
    bench_src = REPO / "third_party" / "robomme_benchmark" / "src"
    assert Path(kv["robomme"]).resolve() == (bench_src / "robomme" / "__init__.py").resolve()
    assert kv["sys_path_added"].split(",") == [str(F.official_dir().resolve()), str(bench_src)]
    defs = [x for x in p.stdout.splitlines() if x.startswith("OFFICIAL_DEFS ")]
    want = F.official_sha256()
    assert len(defs) == 3 and all(f"eval.py={want['eval.py']}" in d for d in defs)
    assert "predictors=OracleSubgoalPredictor " in defs[0] and "predictors=QwenVLSubgoalPredictor " in defs[1]
    assert "predictors=MemERSubgoalPredictor " in defs[2]
    print(f"OFFICIAL_HARD_SOURCE=PASS robomme_hard_imported={kv['robomme_hard_imported']}")


def test_cli_rejects_bad_input_with_exit_3(tmp_path):
    good = [F.identity()]
    shard = tmp_path / "shard-00.json"
    shard.write_text(json.dumps(good), encoding="utf-8")
    bad_tier = tmp_path / "shard-bad.json"
    bad_tier.write_text(json.dumps([dict(F.identity(), tier="xhard1")]), encoding="utf-8")
    (tmp_path / "out" / f"{good[0]['key']}.a1").mkdir(parents=True)
    base = ["--out", str(tmp_path / "out"), "--port", "1", *STAGE3_ARGS]
    cases = {
        "missing_variant": (["--shard", str(shard), *base], "reason=args"),
        "qwen_no_adapter": (["--shard", str(shard), *base, "--variant", F.QWENVL], "reason=args"),
        "adapter_on_oracle": (["--shard", str(shard), *base, "--variant", F.ORACLE,
                               "--qwenvl-groundsg-adapter", "/x"], "reason=args"),
        "bad_tier": (["--shard", str(bad_tier), *base, "--variant", F.ORACLE], "reason=shard"),
        "ep_dir_exists": (["--shard", str(shard), *base, "--variant", F.ORACLE], "reason=ep_dir_exists"),
    }
    for name, (argv, want) in cases.items():
        p = _run(*argv)
        assert p.returncode == 3, (name, p.stdout[-2000:], p.stderr[-2000:])
        assert want in _line(p.stdout, "GROUNDSG_ORIG_BLOCKED"), name


def test_raw_frames_format_and_missing_steps(tmp_path):
    """原始帧：reset 全部帧 + 每步观测；环境抛异常的那步不出帧、记入 missing_steps；字节数 = 帧数 × H × W × 3。"""
    orig = F.OrigSide(F.ORACLE, 60, tmp_path, F.World(plans={3: F.Plan(raise_at=5), 7: F.Plan(success_at=9)}))
    # 第三阶段：局目录多官方视频目录 official/（keep_official=True）
    r_err = orig.run(F.identity())
    r_ok = orig.run(F.identity(source_episode=7, builder_episode=1, seed=510700))
    for r, exec_steps, missing in ((r_err, 5, [5]), (r_ok, 9, [])):
        fdir = Path(r["frames_dir"])
        assert fdir == tmp_path / f"{r['key']}.a1" / "frames"
        fj = json.loads((fdir / "frames.json").read_text())
        n = F.N_RESET_FRAMES + exec_steps - len(missing)
        assert fj == {"pix_fmt": "rgb24", "order": "reset_all_then_per_step_last", "demo_frames": 2, "init_frames": 1,
                      "exec_steps": exec_steps, "missing_steps": missing,
                      "streams": {s: {"width": F.W, "height": F.H, "count": n} for s in ("front", "wrist")}}
        assert r["video_frames"] == {"front": n, "wrist": n}
        for s in ("front", "wrist"):
            assert (fdir / f"{s}.rgb24").stat().st_size == n * F.H * F.W * 3
        # 第三阶段（冻结说明四.3／五）：局目录另有 TraceWriter 收集的 arrays.npz 与语言账本 language.jsonl
        assert sorted(p.name for p in fdir.parent.iterdir()) == \
            ["arrays.npz", "frames", "language.jsonl", "official", "trace.jsonl"]
        end = [json.loads(x) for x in (fdir.parent / "trace.jsonl").read_text().splitlines()][-1]
        with np.load(fdir.parent / "arrays.npz") as arr:
            assert sorted(k for k in arr.files if k.startswith("exec_action__")) == \
                [f"exec_action__{i:05d}" for i in range(exec_steps)]
            assert sorted(k for k in arr.files if k.startswith("exec_state__")) == \
                [f"exec_state__{i - 1:05d}" for i in range(1, exec_steps + 1) if i not in missing]
        assert end["arrays"]["missing_state_steps"] == missing and "error" not in end["arrays"]
    # 首帧字节 = 假环境 reset 第一帧（手算）
    raw = np.frombuffer((Path(r_ok["frames_dir"]) / "front.rgb24").read_bytes(), np.uint8).reshape(-1, F.H, F.W, 3)
    assert raw[0].tobytes() == F.frame((7 * 7) % 150).tobytes()
    assert r_err["status"] == "error" and r_ok["status"] == "success"
    # 官方 builder 构造参数：dataset="test"、joint_angle、max_steps 透传；每任务一个 EnvRunner（builder 只建一次）
    assert orig.world.builder_kwargs == [dict(env_id="PickXtimes", dataset="test", action_space="joint_angle",
                                              gui_render=False, max_steps=60)]


def test_server_unreachable_aborts_shard(tmp_path, monkeypatch):
    """服务不可达：默认客户端工厂探测失败 → 本局记 error（基础设施）并停止整个分片。"""
    ohr = F.official_hard_runner()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # 端口已释放、无人监听
    monkeypatch.setattr(ohr, "PROBE_TIMEOUT_S", 0.5)
    world = F.World(default=F.Plan(success_at=5))
    ctx = ohr.make_context(F.ORACLE, host="127.0.0.1", port=port, max_steps=60, policy_seed=F.POLICY_SEED,
                           builder_cls=world.official_builder_cls(), scratch_root=tmp_path)
    rows = [F.identity(), F.identity(source_episode=7, builder_episode=1, seed=510700)]
    summary = ohr.run_shard(ctx, rows, out=tmp_path)
    assert summary["aborted"] is True and summary["episodes"] == 1
    got = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(got) == 1 and got[0]["status"] == "error" and got[0]["infra"] is True
    assert got[0]["infra_reason"] == "groundsg_unreachable" and got[0]["exception"] == "ServerUnreachable"
