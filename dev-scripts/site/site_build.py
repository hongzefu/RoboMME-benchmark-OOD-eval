#!/usr/bin/env python3
"""V9 建站脚本：身份清单 → 目录 → 子目标帧数 → 起服务 → 浏览器检查（v9 方案 §2.4.2 第 8 步、§2.4.4 阶段 4c）。

维护计划 W2 起只保留「只建站」路径：原 V8 gen1 专用的等报告（``wait_report``）、报告核验（``report``）、三个守卫
（``delivery_set``／``tier_values``／``step_headroom``）、心跳文件与配套看门狗脚本、3b 前的身份清单合成
都已删除；交付与守卫由生成侧（``generate_h5.py`` 聚合与 ``hard_regression.py``）各自验收，本脚本不再重复。
``--site-only`` 仍接受（与历史命令兼容），行为恒为只建站，收尾行带 ``site_only=1``。

步骤（按序；``--cmd STEP=…`` 可替换任一步的命令，测试用假命令替换浏览器检查器等）：

1. ``identities``：``--identities`` 给出的现成身份清单（必填，如 4b 导出的 ``eval-identities-992.jsonl``）。
2. ``catalog``：``site/site_catalog.py`` → ``V8_SITE_CATALOG=PASS``。
3. ``subgoals``：``site/subgoal_lengths.py`` → ``V8_SUBGOALS=PASS``。
4. ``serve``：起服务前探端口占用（``--port 0`` 由系统分配）；``site/site_app.py`` 打出 ``V8_SITE_READY`` 视为就绪。
5. ``site_check``／``oracle_check``：两个浏览器检查器 → ``V8_SITE=PASS``、``V8_ORACLE_BROWSER=PASS``。
6. ``stop_serve``：finally 中收掉服务进程组（SIGTERM → 5 秒 → SIGKILL）；SIGTERM／SIGHUP／SIGINT 同样走 finally。

**格表**（v9 方案 §2.1 S1-E）：``--cells v9``（缺省，``V9_CELLS``）｜``{"Task/tier": n}`` JSON 文件。
期望格数由格表推出；格表一律写 ``<work-dir>/cells.json`` 传给 catalog（``--cells-json``）。V8 专用的
``full``／``v8``（V8 1070 局表）已于维护计划阶段 1b（W4）删除。

**V9 建站**：``--site-dir`` 指向 V9 独立目录（如 ``artifacts/newtask-v9/site``，已有本轮产物时按 progress 指纹复用，
非空且无可复用记录即 FAIL，不覆盖）；``--eval-reuse <V8 site-eval 目录>`` ``--reused <reused.json>``
``--eval-new <V9 评估运行目录>`` 原样透传给 catalog（复用 720 + 新评 80），``--port`` 是服务端口并透传给总表检查器，
检查器在 V8 行之外再出 ``V9_SITE=PASS cells=59 missing=0 eval_reused=… eval_new=… eval_empty=0 port=…``（期望复用数取
``reused.json`` 的 ``count``，期望新评数 = 格表合计 − 复用数），该行也是本步必需判定行。

**产物**（全部在 ``--work-dir``）：``progress.json``（已完成步骤，供崩溃后续行）、``logs/<step>.log``、``report.json``
（schema ``v8-continue-report/1``：每步判定行原文、退出码、耗时；以 ``os.link`` 独占写入，从不覆盖）。
事件日志（``--event-log``，缺省 ``<work-dir>/events.log``，只追加）逐步写 ``V8_CONTINUE_STEP step=<名> status=<PASS|FAIL> …``，
失败写 ``P4_NOTIFY=FAIL step=<名> reason=…``，收尾写 ``V8_CONTINUE=PASS|FAIL step=<名> …``（stdout 同样打印）。

**重复调用**：``report.json`` 已存在 → 先核 schema、输入指纹与完整性，全部相符才复用：不重跑任何步骤、不覆盖任何文件，
重打一行带 ``reused=1`` 的 ``V8_CONTINUE``，退出码与原报告一致；指纹不符 → 拒绝（退出码 2），换新 ``--work-dir``。
没有报告但有 ``progress.json``（上次中途崩溃）→ 只跑未完成步骤（已 PASS 的步骤核其产物 sha256 后复用）。
同一 ``--work-dir`` 用 ``flock`` 互斥，第二个并发实例直接退出（防重复派发）。

退出码：0 = PASS，1 = FAIL，2 = 参数／报告身份冲突，130 = 被信号中断（不写 report.json，可续行）。

V9 阶段 4c（评估完成后只建站）::

    uv run --no-sync python dev-scripts/site/site_build.py --site-only --cells v9 \\
      --delivery artifacts/newtask-v9/delivery/delivery.local.json --specs-root artifacts/newtask-v9/specs-root \\
      --identities artifacts/v9-evaluation/inputs/eval-identities-992.jsonl \\
      --work-dir artifacts/newtask-v9/continue-site --site-dir artifacts/newtask-v9/site --port 8082 \\
      --eval-reuse artifacts/newtask-v8/site-eval --reused artifacts/v9-evaluation/<run_name>/manifest/reused.json \\
      --eval-new artifacts/v9-evaluation/<run_name>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE = Path(__file__).resolve().parent  # 本文件与站点脚本同在 dev-scripts/site
DEFAULT_XHARD0_GEN = REPO_ROOT / "artifacts/newtask-v7/site-media/xhard0-gen"

REPORT_SCHEMA = "v8-continue-report/1"
PROGRESS_SCHEMA = "v8-continue-progress/1"
#: 报告与进度的计数键（全部显式写零）
COUNT_KEYS = ("steps_done", "steps_failed", "steps_reused", "delivered")
STEPS = ("identities", "catalog", "subgoals", "serve", "site_check", "oracle_check", "stop_serve")
#: 每步必须出现的判定行名（取最后一次出现）
EXPECT = {
    "catalog": ("V8_SITE_CATALOG",),
    "subgoals": ("V8_SUBGOALS",),
    "site_check": ("V8_SITE",),
    "oracle_check": ("V8_ORACLE_BROWSER",),
}
READY_RE = re.compile(r"^V8_SITE_READY\b.*\bport=(\d+)")
#: 缺省命令模板：``{名}`` 逐 token 替换；``@cells_json_args``／``@eval_args``／``@v9_expect_args`` 展开为多个 token
DEFAULT_CMDS = {
    "catalog": ["{python}", "{site}/site_catalog.py", "--specs-root", "{specs_root}", "--delivery", "{delivery}",
                "--identities", "{identities}", "--xhard0-gen", "{xhard0_gen}", "--path-base", "{path_base}",
                "@cells_json_args", "@eval_args", "--out", "{site_dir}"],
    "subgoals": ["{python}", "{site}/subgoal_lengths.py", "--site-dir", "{site_dir}", "--specs-root", "{specs_root}",
                 "--delivery", "{delivery}", "--xhard0-gen", "{xhard0_gen}", "--path-base", "{path_base}",
                 "--workers", "{workers}"],
    "serve": ["{python}", "-u", "{site}/site_app.py", "--host", "{host}", "--port", "{port}", "--site-dir", "{site_dir}",
              "--media-root", "{media_root}"],
    "site_check": ["uv", "run", "--no-project", "--with", "playwright", "python", "{site}/site_browser_check.py",
                   "--base", "{base}", "--shots", "{shots}/site"],
    "oracle_check": ["uv", "run", "--no-project", "--with", "playwright", "python", "{site}/oracle_browser_check.py",
                     "--base", "{base}", "--port", "{port_actual}", "--shots", "{shots}/oracle", "--delivery", "{delivery}",
                     "--expect-cells", "{expect_cells}", "@v9_expect_args"],
}
DEFAULT_TIMEOUTS = {"catalog": 1800, "subgoals": 3600, "serve": 120, "site_check": 1800, "oracle_check": 1800}


class Interrupted(Exception):
    def __init__(self, signum: int):
        super().__init__(f"signal {signum}")
        self.signum = signum


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def real(path: str | Path | None) -> str | None:
    return None if path is None else os.path.realpath(os.path.abspath(str(path)))


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_json_exclusive(path: Path, payload: Any) -> None:
    """独占写入：目标已存在即抛 FileExistsError（从不覆盖）。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_line(path: Path, line: str) -> None:
    """事件日志只追加；一次 ``os.write``（O_APPEND）写整行，并发追加不交错。"""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line.rstrip("\n") + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def verdict_lines(lines: list[str], names: tuple[str, ...]) -> dict[str, str | None]:
    """每个判定行名取最后一次出现的原文（``NAME=PASS|FAIL|INFO`` 开头，``V8_SITE`` 不会误配 ``V8_SITE_READY``）。"""
    out: dict[str, str | None] = {name: None for name in names}
    for raw in lines:
        line = raw.strip()
        for name in names:
            if re.match(rf"^{re.escape(name)}=(PASS|FAIL|INFO)\b", line):
                out[name] = line
    return out


def verdict_of(line: str) -> str:
    """判定行 ``NAME=PASS …`` 的结论（PASS／FAIL／INFO）。"""
    return line.split("=", 1)[1].split()[0] if "=" in line else ""


def load_catalog_module():
    spec = importlib.util.spec_from_file_location("site_catalog_c", SITE / "site_catalog.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["site_catalog_c"] = module
    spec.loader.exec_module(module)
    return module


def parse_cells_arg(spec: str, v9: dict[tuple[str, str], int] | None = None) -> dict[tuple[str, str], int]:
    """``v9``（V9 完整格表）或 JSON 文件（``{"Task/tier": n}``，与 catalog ``--cells-json`` 同形态）。
    V8 专用的 ``full``／``v8`` 已删除，给出即报错。"""
    if spec in ("full", "v8"):
        raise SystemExit(f"--cells {spec} 只服务 V8（1070 局表），已删除；请用 v9 或 JSON 文件")
    if spec == "v9":
        if v9 is None:
            raise SystemExit("--cells v9 需要 V9_CELLS")
        return dict(v9)
    raw = json.loads(Path(spec).read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], int] = {}
    for key, n in raw.items():
        task, sep, tier = str(key).partition("/")
        if not sep:
            raise SystemExit(f"--cells JSON 键应为 Task/tier：{key!r}")
        cells[(task, tier)] = int(n)
    return cells


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.work = Path(args.work_dir)
        self.event_log = Path(args.event_log) if args.event_log else self.work / "events.log"
        self.report_path = self.work / "report.json"
        self.progress_path = self.work / "progress.json"
        self.logs = self.work / "logs"
        self.counts = {key: 0 for key in COUNT_KEYS}
        self.steps: dict[str, dict] = {}
        self.step = "init"
        self.status = "running"
        self.child: subprocess.Popen | None = None
        self.server: subprocess.Popen | None = None
        self.base: str | None = None
        self.prev_steps: dict[str, dict] = {}
        self.vars: dict[str, Any] = {}

    # ── 事件、进度 ─────────────────────────────────────────────

    def event(self, line: str) -> None:
        print(line, flush=True)
        append_line(self.event_log, line)

    def save_progress(self) -> None:
        write_json_atomic(self.progress_path, {"schema": PROGRESS_SCHEMA, "fingerprint": self.fingerprint,
                                               "steps": self.steps, "counts": self.counts, "ts": now_iso()})

    def record(self, step: str, status: str, *, rc: int | None = None, lines: list[str] | None = None,
               elapsed: float = 0.0, reason: str | None = None, log: Path | None = None, reused: bool = False,
               extra: dict | None = None) -> dict:
        entry = {"step": step, "status": status, "rc": rc, "lines": lines or [], "elapsed_s": round(elapsed, 3),
                 "reason": reason, "log": str(log) if log else None, "reused": reused, "ended": now_iso(),
                 **(extra or {})}
        self.steps[step] = entry
        if reused:
            self.counts["steps_reused"] += 1
        if status == "PASS":
            self.counts["steps_done"] += 1
        elif status == "FAIL":
            self.counts["steps_failed"] += 1
        self.status = status.lower()
        self.save_progress()
        self.event(f"V8_CONTINUE_STEP step={step} status={status} rc={rc} elapsed_s={entry['elapsed_s']:.1f}"
                   + (f" reused=1" if reused else "") + (f" reason={reason}" if reason else ""))
        return entry

    # ── 子进程 ────────────────────────────────────────────────────

    def command(self, step: str) -> list[str]:
        template = self.vars["cmd_overrides"].get(step, DEFAULT_CMDS[step])
        out: list[str] = []
        for token in template:
            if token == "@cells_json_args":
                out += (["--cells-json", self.vars["cells_json"]] if self.vars["cells_json"] else [])
            elif token == "@eval_args":
                out += self.vars["eval_args"]
            elif token == "@v9_expect_args":
                out += self.vars["v9_expect_args"]
            else:
                out.append(token.format(**{k: v for k, v in self.vars.items() if isinstance(v, (str, int))}))
        return out

    def run_child(self, step: str, argv: list[str], timeout: float) -> tuple[int | None, list[str], str | None, Path]:
        """跑一步子进程（独立进程组），逐行落 ``logs/<step>.log`` 并回显。"""
        log = self.logs / f"{step}.log"
        lines: list[str] = []
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(f"# {now_iso()} argv={shlex.join(argv)}\n")
            handle.flush()
            self.child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                                          env=env, start_new_session=True)
            fd = self.child.stdout.fileno()
            sel = selectors.DefaultSelector()
            sel.register(fd, selectors.EVENT_READ)
            buf, deadline, reason = b"", time.time() + timeout, None

            def take(chunk: bytes) -> None:
                nonlocal buf
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    text = raw.decode("utf-8", "replace")
                    lines.append(text)
                    handle.write(text + "\n")
                    handle.flush()
                    print(f"[{step}] {text}", flush=True)

            try:
                while True:
                    # 至多 1 秒醒一次查子进程是否已退出
                    events = sel.select(timeout=min(1.0, max(0.05, deadline - time.time())))
                    if events:
                        chunk = os.read(fd, 65536)
                        if not chunk:
                            break
                        take(chunk)
                    else:
                        if self.child.poll() is not None:
                            # 子进程已退出、管道却被遗留的孙进程占着：先非阻塞读尽管道里剩下的数据再退出，不丢判定行
                            os.set_blocking(fd, False)
                            while True:
                                try:
                                    chunk = os.read(fd, 65536)
                                except BlockingIOError:
                                    break
                                if not chunk:
                                    break
                                take(chunk)
                            break
                    if time.time() > deadline:
                        reason = f"timeout_{int(timeout)}s"
                        break
            finally:
                sel.close()
            if buf:
                text = buf.decode("utf-8", "replace")
                lines.append(text)
                handle.write(text + "\n")
            # 无论组长怎么结束（正常退出、超时、遗留孙进程），都对整组收尾：孙进程不成孤儿
            rc = kill_group(self.child) if reason else self.child.wait()
            if reason is None:
                kill_group(self.child)
            self.child.stdout.close()
            handle.write(f"# {now_iso()} EXIT_CODE={rc}\n")
        self.child = None
        return rc, lines, reason, log

    def run_step(self, step: str) -> dict:
        self.step, self.status = step, "running"
        started = time.time()
        rc, lines, reason, log = self.run_child(step, self.command(step), self.vars["timeouts"][step])
        found = verdict_lines(lines, self.vars.get("expect", EXPECT)[step])
        kept = [line for line in found.values() if line]
        status = "PASS"
        if reason:
            status = "FAIL"
        elif rc != 0:
            status, reason = "FAIL", f"exit_{rc}"
        missing = [name for name, line in found.items() if line is None]
        if status == "PASS" and missing:
            status, reason = "FAIL", "line_missing:" + ",".join(missing)
        not_pass = [name for name, line in found.items() if line and verdict_of(line) != "PASS"]
        if status == "PASS" and not_pass:
            status, reason = "FAIL", "verdict:" + ",".join(not_pass)
        return self.record(step, status, rc=rc, lines=kept, elapsed=time.time() - started, reason=reason, log=log)

    # ── 各步 ──────────────────────────────────────────────────────

    def identities(self) -> dict:
        """只认 ``--identities`` 给出的现成清单（3b 前的合成路径已删）。"""
        self.step, self.status = "identities", "running"
        started = time.time()
        path = Path(self.args.identities)
        if not path.is_file():
            return self.record("identities", "FAIL", elapsed=time.time() - started, reason=f"missing:{path}")
        self.vars["identities"] = str(path)
        n = sum(1 for t in path.read_text(encoding="utf-8").splitlines() if t.strip())
        return self.record("identities", "PASS", elapsed=time.time() - started,
                           lines=[f"IDENTITIES source=given rows={n} path={path}"],
                           extra={"sha256": sha256_file(path), "source": "given"})

    def serve(self) -> dict:
        self.step, self.status = "serve", "running"
        started = time.time()
        host, port = self.args.host, int(self.args.port)
        if port and port_busy(host, port):
            return self.record("serve", "FAIL", elapsed=time.time() - started, reason=f"port_busy:{host}:{port}")
        log = self.logs / "serve.log"
        handle = open(log, "a", encoding="utf-8")
        argv = self.command("serve")
        handle.write(f"# {now_iso()} argv={shlex.join(argv)}\n")
        handle.flush()
        offset = handle.tell()  # 续行时日志里还有上一次服务的就绪行，只认本次启动之后写入的内容
        # 服务输出直接落文件（不走管道），请求日志再多也不会因管道写满而卡住服务
        self.server = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                                       env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
        handle.close()
        deadline = started + self.vars["timeouts"]["serve"]
        while True:
            with open(log, "rb") as reader:
                reader.seek(offset)
                text = reader.read().decode("utf-8", "replace")
            ready =next((l for l in text.splitlines() if READY_RE.match(l)), None)
            if ready:
                actual = int(READY_RE.match(ready).group(1))
                self.base = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{actual}"
                self.vars["base"], self.vars["port_actual"] = self.base, actual
                return self.record("serve", "PASS", rc=None, lines=[ready], elapsed=time.time() - started, log=log,
                                   extra={"server_pid": self.server.pid, "base": self.base})
            if self.server.poll() is not None:
                return self.record("serve", "FAIL", rc=self.server.returncode, elapsed=time.time() - started, log=log,
                                   reason="server_exited", lines=text.splitlines()[-3:])
            if time.time() > deadline:
                return self.record("serve", "FAIL", elapsed=time.time() - started, log=log, reason="ready_timeout")
            time.sleep(0.1)

    def stop_serve(self) -> None:
        if self.server is None:
            return
        started = time.time()
        rc = kill_group(self.server)
        self.server = None
        if "serve" in self.steps and self.steps["serve"].get("status") == "PASS":
            prev = self.step
            self.step = "stop_serve"
            self.record("stop_serve", "PASS", rc=rc, elapsed=time.time() - started)
            self.step = prev

    # ── 续行与复用 ────────────────────────────────────────────────

    def reusable(self, step: str) -> bool:
        prev = self.prev_steps.get(step)
        if not prev or prev.get("status") != "PASS":
            return False
        if step == "catalog":
            paths = [Path(self.args.site_dir) / "catalog.json", Path(self.args.site_dir) / "media-private.json"]
        elif step == "subgoals":
            paths = [Path(self.args.site_dir) / "subgoals.json"]
        elif step == "identities":
            if prev.get("path_used") is None or not Path(prev["path_used"]).is_file():
                return False
            return sha256_file(Path(prev["path_used"])) == prev.get("sha256")
        else:
            return True
        return all(p.is_file() for p in paths) and prev.get("sha256s") == [sha256_file(p) for p in paths]

    def reuse(self, step: str) -> dict:
        prev = dict(self.prev_steps[step])
        self.step = step
        if step == "identities":
            self.vars["identities"] = prev["path_used"]
        extra = {k: v for k, v in prev.items() if k not in ("step", "status", "rc", "lines", "elapsed_s", "reason",
                                                             "log", "reused", "ended")}
        return self.record(step, prev["status"], rc=prev.get("rc"), lines=prev.get("lines"), elapsed=0.0,
                           reason=prev.get("reason"), reused=True,
                           log=Path(prev["log"]) if prev.get("log") else None, extra=extra)

    # ── 主流程 ────────────────────────────────────────────────────

    def finish(self, verdict: str, step: str, reason: str | None, site_built: bool, reused: bool = False) -> int:
        final = (f"V8_CONTINUE={verdict} step={step} steps={len(self.steps)} site={'built' if site_built else 'not_built'} "
                 f"delivered={self.counts['delivered']} reused={int(reused)} site_only=1"
                 + (f" reason={reason}" if reason else "") + f" report={self.report_path}")
        report = {"schema": REPORT_SCHEMA, "fingerprint": self.fingerprint, "verdict": verdict, "final_step": step,
                  "reason": reason, "site_built": site_built, "site_dir": real(self.args.site_dir),
                  "counts": dict(self.counts), "steps": [self.steps[s] for s in STEPS if s in self.steps],
                  "final_line": final, "ended": now_iso(), "exit_code": EXIT_CODES[verdict]}
        self.step, self.status = step, "done"
        write_json_exclusive(self.report_path, report)
        if verdict != "PASS":
            self.event(f"P4_NOTIFY={verdict} step={step} reason={reason} report={self.report_path}")
        self.event(final)
        return EXIT_CODES[verdict]

    def count_delivered(self) -> None:
        """交付行数照实记进计数（交付与守卫由生成侧验收，本脚本只读行数）。"""
        try:
            rows = json.loads(Path(self.args.delivery).read_text(encoding="utf-8")).get("rows")
            self.counts["delivered"] = len(rows) if isinstance(rows, list) else 0
        except (OSError, ValueError, AttributeError):
            self.counts["delivered"] = 0

    def run(self) -> int:
        self.count_delivered()
        site = Path(self.args.site_dir)
        if not self.reusable("catalog") and site.exists() and any(site.iterdir()):
            self.step = "catalog"
            entry = self.record("catalog", "FAIL", reason=f"site_dir_not_empty:{site}")
            return self.finish("FAIL", "catalog", entry["reason"], False)
        for name in ("identities", "catalog", "subgoals"):
            if self.reusable(name):
                entry = self.reuse(name)
            elif name == "identities":
                entry = self.identities()
                entry["path_used"] = self.vars.get("identities")
                self.save_progress()
            else:
                entry = self.run_step(name)
                if entry["status"] == "PASS":
                    paths = ([site / "catalog.json", site / "media-private.json"] if name == "catalog"
                             else [site / "subgoals.json"])
                    entry["sha256s"] = [sha256_file(p) for p in paths]
                    self.save_progress()
            if entry["status"] == "FAIL":
                return self.finish("FAIL", name, entry["reason"], False)
        if self.reusable("site_check") and self.reusable("oracle_check"):
            for name in ("serve", "site_check", "oracle_check", "stop_serve"):
                if name in self.prev_steps:
                    self.reuse(name)
        else:
            Path(self.vars["shots"]).mkdir(parents=True, exist_ok=True)
            try:
                entry = self.serve()
                if entry["status"] == "FAIL":
                    return self.finish("FAIL", "serve", entry["reason"], False)
                for name in ("site_check", "oracle_check"):
                    entry = self.run_step(name)
                    if entry["status"] == "FAIL":
                        self.stop_serve()
                        return self.finish("FAIL", name, entry["reason"], False)
            finally:
                self.stop_serve()
        return self.finish("PASS", "done", None, True)


EXIT_CODES = {"PASS": 0, "FAIL": 1}


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_group(proc: subprocess.Popen, grace: float = 5.0) -> int | None:
    """收掉子进程所在进程组（子进程以 ``start_new_session`` 启动，组号 = 其 pid；只按这个精确组号发信号）。

    组长已退出也照样对整组发信号：``uv run`` 下的 playwright 驱动／chromium 等孙进程留在同一组里，组长退出后
    不收就成孤儿。SIGTERM → 等组空（``killpg(pgid, 0)``）至 ``grace`` 秒 → SIGKILL → 再等至 5 秒。"""
    pgid = proc.pid
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        if proc.poll() is not None and not _group_alive(pgid):
            break
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + wait
        while time.time() < deadline:
            proc.poll()  # 回收组长，免得僵尸让组看起来还活着
            if proc.returncode is not None and not _group_alive(pgid):
                break
            time.sleep(0.05)
    try:
        return proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        return proc.poll()


def port_busy(host: str, port: int) -> bool:
    target = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((target, port), timeout=1.0):
            return True
    except OSError:
        return False


def cells_help() -> str:
    """``--cells`` 帮助文字：格数与局数由格表推出（不写死）。"""
    try:
        H = load_catalog_module().load_hard_specs()
        n_x0 = len(H.ALL_TASKS) * 12
        desc = [f"{name}（{len(t)} 格 {sum(t.values())} 局 + xhard0 {n_x0} = {sum(t.values()) + n_x0}）"
                for name, t in (("v9", H.V9_CELLS),)]
    except Exception:  # 帮助文字不因格表加载失败而报错
        desc = ["v9（V9_CELLS）"]
    return "｜".join(desc) + "｜{\"Task/tier\": n} JSON 文件；缺省 v9（V8 专用的 full／v8 已删除）"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delivery", required=True, help="交付清单 delivery.json（V9 为 assemble 写出的清单；只读行数并透传 catalog）")
    ap.add_argument("--specs-root", required=True, help="/4 规格根（透传 catalog／subgoals）")
    ap.add_argument("--cells", default="v9", help=cells_help())
    ap.add_argument("--work-dir", required=True, help="进度、日志、报告目录（同一次建站复用同一目录）")
    ap.add_argument("--site-dir", required=True,
                    help="站点输出目录（V8 artifacts/newtask-v8/site、V9 artifacts/newtask-v9/site；首次须不存在或为空，"
                         "已有本轮产物时按指纹复用）")
    ap.add_argument("--site-only", action="store_true",
                    help="只建站（与历史命令兼容；W2 起恒为只建站，可省略）")
    ap.add_argument("--eval-reuse", default=None, help="透传 catalog：V8 站点目录（artifacts/newtask-v8/site-eval）")
    ap.add_argument("--reused", default=None, help="透传 catalog：S1-F 产出的 reused.json（复用集合唯一依据）")
    ap.add_argument("--eval-new", default=None, help="透传 catalog：V9 新评运行目录（结构同 V8 评估运行）")
    ap.add_argument("--event-log", default=None, help="事件日志（缺省 <work-dir>/events.log）")
    ap.add_argument("--identities", required=True, help="现成身份清单（如 eval-identities-992.jsonl）")
    ap.add_argument("--xhard0-gen", default=str(DEFAULT_XHARD0_GEN), help="v7 已渲染的 xhard0 生成视频目录")
    ap.add_argument("--path-base", default=str(REPO_ROOT), help="catalog／subgoals 的 --path-base")
    ap.add_argument("--media-root", default=str(REPO_ROOT / "artifacts"), help="站点服务媒体白名单根")
    ap.add_argument("--workers", type=int, default=16, help="subgoals 的 --workers")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090,
                    help="站点服务端口（0 = 系统分配；V9 独立站用 8082 等空闲端口）；占用即 FAIL；实际端口透传给总表检查器")
    ap.add_argument("--shots", default=None, help="浏览器截图目录（缺省 <work-dir>/shots）")
    ap.add_argument("--step-timeout", action="append", default=[], metavar="STEP=SECONDS",
                    help=f"覆盖单步超时（缺省 {DEFAULT_TIMEOUTS}）")
    ap.add_argument("--cmd", action="append", default=[], metavar="STEP=COMMAND",
                    help="替换某步命令（shlex 切分；可用 {python} {delivery} {base} {shots} 等占位与 @eval_args 等展开）")
    return ap


def fingerprint_of(args: argparse.Namespace, cells: dict, overrides: dict) -> dict:
    delivery = Path(args.delivery)
    return {
        "delivery": real(delivery), "delivery_sha256": sha256_file(delivery) if delivery.is_file() else None,
        "specs_root": real(args.specs_root), "cells": {f"{t}/{tier}": n for (t, tier), n in sorted(cells.items())},
        "site_dir": real(args.site_dir), "xhard0_gen": real(args.xhard0_gen), "identities": real(args.identities),
        "cmd_overrides": {k: v for k, v in sorted(overrides.items())}, "path_base": real(args.path_base),
        "media_root": real(args.media_root), "workers": int(args.workers), "host": args.host, "port": int(args.port),
        "site_only": True,
        **({"eval_reuse": real(args.eval_reuse), "reused": real(args.reused),
            "reused_sha256": sha256_file(Path(args.reused)) if args.reused and Path(args.reused).is_file() else None,
            "eval_new": real(args.eval_new)} if (args.eval_reuse or args.reused or args.eval_new) else {}),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    lock_handle = open(work / ".lock", "a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"V8_CONTINUE_BUSY work_dir={work} reason=another_instance_running", flush=True)
        return 2
    C = load_catalog_module()
    H = C.load_hard_specs()
    cells = parse_cells_arg(args.cells, dict(H.V9_CELLS))
    # V9 透传参数（复用 + 新评）：参数不全或 reused.json 不可读在 refuse 定义后拒绝
    v9_eval = bool(args.eval_reuse or args.reused or args.eval_new)
    preflight = None
    if v9_eval and not (args.eval_reuse and args.reused):
        preflight = "v9_eval_needs_eval_reuse_and_reused"
    eval_args: list[str] = []
    v9_expect_args: list[str] = []
    if v9_eval and preflight is None:
        eval_args = ["--eval-reuse", str(args.eval_reuse), "--reused", str(args.reused)]
        eval_args += ["--eval-new", str(args.eval_new)] if args.eval_new else []
        try:
            n_reused = int(json.loads(Path(args.reused).read_text(encoding="utf-8"))["count"])
            v9_expect_args = ["--expect-reused", str(n_reused), "--expect-new", str(sum(cells.values()) - n_reused)]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            preflight = f"reused_unreadable:{type(exc).__name__}"
    overrides: dict[str, list[str]] = {}
    for item in args.cmd:
        step, sep, cmd = item.partition("=")
        if not sep or step not in DEFAULT_CMDS:
            raise SystemExit(f"--cmd 须为 STEP=COMMAND，STEP ∈ {sorted(DEFAULT_CMDS)}：{item!r}")
        overrides[step] = shlex.split(cmd)
    timeouts = dict(DEFAULT_TIMEOUTS)
    for item in args.step_timeout:
        step, sep, sec = item.partition("=")
        if not sep or step not in timeouts:
            raise SystemExit(f"--step-timeout 须为 STEP=SECONDS，STEP ∈ {sorted(timeouts)}：{item!r}")
        timeouts[step] = float(sec)
    runner = Runner(args)
    def refuse(step: str) -> int:
        """拒绝执行（参数／报告身份冲突）：退出码 2。"""
        runner.step, runner.status = step, "aborted"
        return 2

    if preflight is not None:
        print(f"V8_CONTINUE=FAIL step=preflight reason={preflight}", flush=True)
        return refuse("preflight")
    # 原「V8 完整表时传 full、不写 cells.json」分支随 V8 表删除；格表一律写 cells.json（--cells v9 时与删除前相同）
    cells_json = work / "cells.json"
    text = json.dumps({f"{t}/{tier}": n for (t, tier), n in sorted(cells.items())}, ensure_ascii=False)
    if not cells_json.exists():
        cells_json.write_text(text, encoding="utf-8")
    elif cells_json.read_text(encoding="utf-8") != text:
        print(f"V8_CONTINUE=FAIL step=preflight reason=cells_json_differs:{cells_json}", flush=True)
        return refuse("preflight")
    runner.fingerprint = fingerprint_of(args, cells, {k: shlex.join(v) for k, v in overrides.items()})
    runner.vars = {
        "python": sys.executable, "site": str(SITE), "repo": str(REPO_ROOT),
        "delivery": str(args.delivery), "specs_root": str(args.specs_root),
        "cells": str(cells_json), "cells_json": str(cells_json),
        "cells_table": cells, "xhard0_gen": str(args.xhard0_gen), "path_base": str(args.path_base), "site_dir": str(args.site_dir),
        "media_root": str(args.media_root), "workers": str(args.workers), "host": args.host, "port": str(args.port),
        "shots": str(args.shots or work / "shots"), "work_dir": str(work), "base": "",
        "expect_cells": str(len(cells) + len(H.ALL_TASKS)), "identities": str(args.identities),
        "cmd_overrides": overrides, "timeouts": timeouts, "eval_args": eval_args, "v9_expect_args": v9_expect_args,
    }
    if v9_eval:  # V9 复用模式：总表检查器另出 V9_SITE 行，也是该步必需判定行
        runner.vars["expect"] = dict(EXPECT, oracle_check=EXPECT["oracle_check"] + ("V9_SITE",))

    # 同一完成事件重复到达：报告已在 → 核身份、来源、完整性后复用，不重跑、不覆盖
    if runner.report_path.exists():
        try:
            old = json.loads(runner.report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            runner.event(f"V8_CONTINUE=FAIL step=resume reason=report_unreadable:{type(exc).__name__} "
                         f"reused=0 report={runner.report_path}")
            return refuse("resume")
        steps_ok = (isinstance(old, dict) and isinstance(old.get("steps"), list) and bool(old["steps"])
                    and all(isinstance(s, dict) and s.get("step") in STEPS and s.get("status") for s in old["steps"]))
        complete = (steps_ok and old.get("schema") == REPORT_SCHEMA
                    and str(old.get("final_line", "")).startswith(f"V8_CONTINUE={old.get('verdict')} ")
                    and old.get("verdict") in EXIT_CODES and old.get("exit_code") == EXIT_CODES[old["verdict"]]
                    and (old.get("final_step") == "done"
                         or old.get("final_step") in {s["step"] for s in old["steps"]})
                    and isinstance(old.get("counts"), dict)
                    and all(isinstance(old["counts"].get(k), int) for k in COUNT_KEYS))
        if not complete:
            runner.event(f"V8_CONTINUE=FAIL step=resume reason=report_incomplete reused=0 report={runner.report_path}")
            return refuse("resume")
        if old.get("fingerprint") != runner.fingerprint:
            diff = sorted(k for k in set(old.get("fingerprint", {})) | set(runner.fingerprint)
                          if old.get("fingerprint", {}).get(k) != runner.fingerprint.get(k))
            runner.event(f"V8_CONTINUE=FAIL step=resume reason=report_identity_mismatch:{','.join(diff)} reused=0 "
                         f"report={runner.report_path}")
            return refuse("resume")
        runner.step, runner.status = str(old.get("final_step")), "done"
        runner.counts = {key: old["counts"][key] for key in COUNT_KEYS}
        runner.event(old["final_line"].replace(" reused=0", " reused=1"))
        return int(old["exit_code"])

    if runner.progress_path.exists():
        try:
            prog = json.loads(runner.progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prog = {}
        if prog.get("schema") == PROGRESS_SCHEMA and prog.get("fingerprint") == runner.fingerprint:
            runner.prev_steps = dict(prog.get("steps") or {})
            print(f"# 续行：上次已完成 {sorted(k for k, v in runner.prev_steps.items() if v.get('status') == 'PASS')}",
                  flush=True)
        else:
            runner.event(f"V8_CONTINUE=FAIL step=resume reason=progress_identity_mismatch reused=0 "
                         f"report={runner.report_path}")
            return refuse("resume")

    def on_signal(signum, _frame):
        raise Interrupted(signum)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, on_signal)
    runner.event(f"V8_CONTINUE_START pid={os.getpid()} work_dir={work} delivery={args.delivery} "
                 f"resume={int(bool(runner.prev_steps))}")
    try:
        return runner.run()
    except Interrupted as exc:
        runner.status = "aborted"
        runner.save_progress()
        runner.event(f"V8_CONTINUE_ABORT step={runner.step} signal={exc.signum} report=absent（可续行）")
        return 130
    finally:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if runner.child is not None:
            kill_group(runner.child)
        if runner.server is not None:
            kill_group(runner.server)
        lock_handle.close()


if __name__ == "__main__":
    sys.exit(main())
