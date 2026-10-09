"""C16 接续部分：``site_build.py --site-only --cells v9`` 的步骤编排、指纹复用、失败停止、锁与幂等。

真实调用 ``site_build.main``（进程内），各步命令经 ``--cmd`` 换成假命令（真子进程、零依赖），假命令把自己的
执行记进计数文件——「不重跑」以计数文件为准，而不是看脚本打印的 PASS。中断用例由假检查器给父进程发 SIGTERM，
走生产的信号处理与进度续行。
"""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import socket
from pathlib import Path

import pytest

from tests._support.dev_loaders import load_script

SB = load_script("injection-dev/site_build.py")
H = SB.load_catalog_module().load_hard_specs()

FAKE = r'''
import argparse, os, signal, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--name"); ap.add_argument("--count")
ap.add_argument("--line", action="append", default=[]); ap.add_argument("--write", action="append", default=[])
ap.add_argument("--exit", type=int, default=0); ap.add_argument("--serve", action="store_true")
ap.add_argument("--term-parent-if")
a, rest = ap.parse_known_args()
with open(a.count, "a") as fh:
    fh.write(a.name + " " + " ".join(rest) + "\n")
for path in a.write:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(a.name + "\n")
if a.term_parent_if and os.path.exists(a.term_parent_if):
    with open(a.term_parent_if + ".pid", "w") as fh:
        fh.write(str(os.getpid()))
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(30)
if a.serve:
    print("V8_SITE_READY host=127.0.0.1 port=4321 videos=0", flush=True)
    time.sleep(60)
for line in a.line:
    print(line, flush=True)
sys.exit(a.exit)
'''

PASS_LINES = {"catalog": "V8_SITE_CATALOG=PASS identities=1", "subgoals": "V8_SUBGOALS=PASS h5=1",
              "site_check": "V8_SITE=PASS sections=1", "oracle_check": "V8_ORACLE_BROWSER=PASS cells=1"}


@pytest.fixture(autouse=True)
def keep_signal_handlers():
    """site_build.main 收尾会把 TERM／HUP／INT 设成忽略；每个用例结束复原测试进程原来的处理器。"""
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


class World:
    def __init__(self, root: Path):
        self.root = root
        self.fake = root / "fake_step.py"
        self.fake.write_text(FAKE)
        self.count = root / "count.log"
        self.flag = root / "term.flag"
        self.delivery = root / "delivery.json"
        self.delivery.write_text(json.dumps({"schema": "v8-delivery/1", "rows": [{"task": "StopCube"}] * 3}))
        self.identities = root / "ids.jsonl"
        self.identities.write_text('{"task": "StopCube"}\n{"task": "StopCube"}\n')

    def cmd(self, step, *extra, line=None, exit_code=0):
        argv = ["{python}", str(self.fake), "--name", step, "--count", str(self.count)]
        if step == "catalog":
            argv += ["--write", "{site_dir}/catalog.json", "--write", "{site_dir}/media-private.json"]
        if step == "subgoals":
            argv += ["--write", "{site_dir}/subgoals.json"]
        if step == "serve":
            argv += ["--serve"]
        else:
            argv += ["--line", line if line is not None else PASS_LINES[step], "--exit", str(exit_code)]
        return f"{step}={shlex.join([*argv, *extra])}"

    def argv(self, work="work", site="site", cmds=None, extra=()):
        cmds = dict(cmds or {})
        base = {s: self.cmd(s) for s in ("catalog", "subgoals", "serve", "site_check")}
        base["oracle_check"] = self.cmd("oracle_check", "{expect_cells}", "--term-parent-if", str(self.flag))
        base.update(cmds)
        out = ["--site-only", "--cells", "v9", "--delivery", str(self.delivery), "--specs-root", str(self.root / "specs"),
               "--identities", str(self.identities), "--work-dir", str(self.root / work),
               "--site-dir", str(self.root / site), "--port", "0", "--xhard0-gen", str(self.root / "x0"),
               "--path-base", str(self.root), "--media-root", str(self.root), *extra]
        for value in base.values():
            out += ["--cmd", value]
        return out

    def runs(self):
        if not self.count.exists():
            return []
        return [l.split()[0] for l in self.count.read_text().splitlines()]


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _events(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def test_site_only_pipeline_runs_steps_in_order(world, capsys):
    rc = SB.main(world.argv())
    out = capsys.readouterr().out
    assert rc == 0
    assert world.runs() == ["catalog", "subgoals", "serve", "site_check", "oracle_check"]
    oracle_echo = world.count.read_text().splitlines()[-1].split()
    assert oracle_echo[1] == str(len(H.V9_CELLS) + len(H.ALL_TASKS))  # 43 个新值格 + 16 个 xhard0 格
    report = json.loads((world.root / "work" / "report.json").read_text())
    assert report["schema"] == SB.REPORT_SCHEMA and report["verdict"] == "PASS" and report["exit_code"] == 0
    assert [s["step"] for s in report["steps"]] == list(SB.STEPS)
    assert all(s["status"] == "PASS" and not s["reused"] for s in report["steps"])
    assert report["counts"] == {"steps_done": 7, "steps_failed": 0, "steps_reused": 0, "delivered": 3}
    final = [l for l in out.splitlines() if l.startswith("V8_CONTINUE=")][-1]
    assert final.startswith("V8_CONTINUE=PASS step=done ") and "site_only=1" in final and "reused=0" in final
    assert _events(world.root / "work" / "events.log")[-1] == final
    serve = next(s for s in report["steps"] if s["step"] == "serve")
    with pytest.raises(ProcessLookupError):  # 服务进程组已收掉
        os.killpg(serve["server_pid"], 0)


def test_repeated_call_reuses_report_without_rerun(world, capsys):
    assert SB.main(world.argv()) == 0
    report = (world.root / "work" / "report.json").read_bytes()
    before = world.runs()
    capsys.readouterr()
    assert SB.main(world.argv()) == 0
    out = capsys.readouterr().out
    assert world.runs() == before
    assert (world.root / "work" / "report.json").read_bytes() == report
    assert "V8_CONTINUE=PASS step=done" in out and "reused=1" in out


def test_changed_inputs_refuse_reuse(world, capsys):
    assert SB.main(world.argv()) == 0
    report = (world.root / "work" / "report.json").read_bytes()
    before = world.runs()
    assert SB.main(world.argv(extra=("--workers", "3"))) == 2
    assert "report_identity_mismatch:workers" in capsys.readouterr().out
    assert world.runs() == before and (world.root / "work" / "report.json").read_bytes() == report


@pytest.mark.parametrize("line, exit_code, reason", [
    ("V8_SITE_CATALOG=FAIL identities=0", 0, "verdict:V8_SITE_CATALOG"),
    ("V8_SITE_CATALOG=PASS identities=1", 3, "exit_3"),
    ("V8_SITE_CATALOGUE=PASS", 0, "line_missing:V8_SITE_CATALOG"),
])
def test_failed_step_stops_pipeline(world, line, exit_code, reason):
    rc = SB.main(world.argv(cmds={"catalog": world.cmd("catalog", line=line, exit_code=exit_code)}))
    assert rc == 1
    assert world.runs() == ["catalog"]
    report = json.loads((world.root / "work" / "report.json").read_text())
    assert (report["verdict"], report["final_step"], report["reason"], report["site_built"]) == \
        ("FAIL", "catalog", reason, False)
    assert any(l.startswith("P4_NOTIFY=FAIL step=catalog") for l in _events(world.root / "work" / "events.log"))


def test_nonempty_site_dir_without_record_fails(world):
    (world.root / "site").mkdir()
    (world.root / "site" / "old.txt").write_text("别人的产物")
    assert SB.main(world.argv()) == 1
    assert world.runs() == [] and (world.root / "site" / "old.txt").read_text() == "别人的产物"


def test_second_instance_on_same_work_dir_is_refused(world, capsys):
    work = world.root / "work"
    work.mkdir()
    with open(work / ".lock", "a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert SB.main(world.argv()) == 2
    assert "V8_CONTINUE_BUSY" in capsys.readouterr().out
    assert world.runs() == [] and not (work / "report.json").exists()


def test_sigterm_mid_check_resumes_only_unfinished_steps(world, capsys):
    world.flag.write_text("1")
    assert SB.main(world.argv()) == 130
    # 发出 SIGTERM 后仍在 sleep 的假检查器（独立进程组，组号 = 其 pid）已被收掉
    oracle_pid = int(Path(str(world.flag) + ".pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(oracle_pid, 0)
    work = world.root / "work"
    assert not (work / "report.json").exists()
    progress = json.loads((work / "progress.json").read_text())
    assert {k: v["status"] for k, v in progress["steps"].items() if k in ("catalog", "subgoals")} == \
        {"catalog": "PASS", "subgoals": "PASS"}
    assert any("V8_CONTINUE_ABORT" in l for l in _events(work / "events.log"))
    world.flag.unlink()
    capsys.readouterr()
    assert SB.main(world.argv()) == 0
    runs = world.runs()
    assert runs.count("catalog") == 1 and runs.count("subgoals") == 1  # 已 PASS 且产物 sha 未变：复用
    assert runs.count("oracle_check") == 2 and runs.count("serve") == 2
    report = json.loads((work / "report.json").read_text())
    reused = {s["step"]: s["reused"] for s in report["steps"]}
    assert reused["catalog"] and reused["subgoals"] and not reused["oracle_check"]


def test_tampered_catalog_is_refused_on_resume(world):
    world.flag.write_text("1")
    assert SB.main(world.argv()) == 130
    world.flag.unlink()
    (world.root / "site" / "catalog.json").write_text("被改过\n")  # 产物 sha 与进度记录不符
    # 站点目录非空且 catalog 不可复用 → 拒绝（不覆盖），而不是静默沿用被改过的产物
    assert SB.main(world.argv()) == 1
    assert world.runs().count("catalog") == 1


def test_port_busy_fails_serve(world):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        argv = world.argv()
        argv[argv.index("--port") + 1] = str(port)
        assert SB.main(argv) == 1
    report = json.loads((world.root / "work" / "report.json").read_text())
    assert report["final_step"] == "serve" and report["reason"] == f"port_busy:127.0.0.1:{port}"
    assert "serve" not in world.runs()


def test_v9_eval_mode_requires_v9_site_line(world, tmp_path):
    reused = tmp_path / "reused.json"
    reused.write_text(json.dumps({"count": 5}))
    extra = ("--eval-reuse", str(tmp_path / "v8site"), "--reused", str(reused))
    oracle = world.cmd("oracle_check", "@v9_expect_args")
    assert SB.main(world.argv(cmds={"oracle_check": oracle}, extra=extra)) == 1
    report = json.loads((world.root / "work" / "report.json").read_text())
    assert report["reason"] == "line_missing:V9_SITE"
    echo = world.count.read_text().splitlines()[-1].split()[1:]
    assert echo == ["--expect-reused", "5", "--expect-new", str(sum(H.V9_CELLS.values()) - 5)]
    ok = world.cmd("oracle_check", "--line", "V9_SITE=PASS cells=59")
    assert SB.main(world.argv(work="w2", site="s2", cmds={"oracle_check": ok}, extra=extra)) == 0


def test_v9_eval_mode_preflight(world, capsys):
    assert SB.main(world.argv(extra=("--eval-reuse", str(world.root)))) == 2
    assert "v9_eval_needs_eval_reuse_and_reused" in capsys.readouterr().out
    assert world.runs() == []


def test_removed_v8_cells_rejected(world):
    with pytest.raises(SystemExit, match="已删除"):
        SB.parse_cells_arg("full", {})
    with pytest.raises(SystemExit, match="Task/tier"):
        bad = world.root / "c.json"
        bad.write_text('{"StopCube@xhard1": 1}')
        SB.parse_cells_arg(str(bad), {})
