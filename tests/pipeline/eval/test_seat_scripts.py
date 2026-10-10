"""C16 席位脚本（慢）：真 bash 运行拆仓后的 ``dev-scripts/gl/run_eval_gl.sh``、原侧 ``dev-scripts/orig/run_official_hard.sh``／
``pair_seat.sh`` 与媒体函数库 ``dev-scripts/gl/seat_media_lib.sh``；服务端、新侧席位客户端（``seat.py``）与原侧驱动换成假引擎
（``seat_fake_engine.py``），``nvidia-smi`` 换成空输出的替身，不碰 GPU；转码用真 ffmpeg（16×16 微型帧）。

覆盖：

- ``run_eval_gl.sh``（新）：runbook 写法的参数原样落到 ``seat.py run``（策略、GroundSG 变体、种子、身份清单、共享账本与
  §五缺省上限 26／60／0／0／26、基础设施重试 0、``--gpus 0``、产物根缺省 ``<repo>/artifacts/<run>/gl``、透传的模型参数）；
  ``RUN_INPUTS``（``robomme_ood_eval`` 必须来自执行副本 ``src``）；客户端以 75 退出时按 ``SERVER_LEFT`` 行停残留服务端、
  重起次数缺省 0 即停、给 1 次则重起；无进展看门狗（``progress.json`` 超时不动即 TERM、rc=124）；TERM 收尾只写一次；参数错误；
- 原侧函数库 ``orig_seat_lib.sh`` 不能直接执行；``seat_media_lib.sh`` 的完整收尾链（旧 FFV1 原始帧：重绘在并入轨迹之后、
  转码之前，转码后删原始帧；新 AV1 原始帧：``permanent_raw`` 不转码不删）、``render_official_dir`` 的 KEPT／重绘／不支持
  ``--source`` 的干净失败；
- ``run_official_hard.sh`` 的分轮重试（先重启服务再重发、重试额度）与原侧转码发布；``pair_seat.sh`` 先原侧后新侧、新侧走
  新的 ``run_eval_gl.sh``（分片作身份清单、``--dataset hard-verify``、种子与预算参数转发）。

旧 ``run_seat.sh`` 不迁入（职责已进 ``ServerProcess``、``seat.py``、``episode.py``），针对它的用例随之删除。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import eval_fakes_dev as F

pytestmark = pytest.mark.slow

DS = F.REPO / "dev-scripts"
ENGINE = Path(__file__).resolve().parent / "seat_fake_engine.py"
SCRIPTS = ("run_eval_gl.sh", "run_official_hard.sh", "pair_seat.sh", "orig_seat_lib.sh", "seat.py")


def _sync_fixture(tmp_path, mode):
    root = tmp_path / "sync-fixture"
    rel = root / "dev-scripts/release"
    rel.mkdir(parents=True)
    (root / "artifacts").mkdir()
    script = rel / "sync_to_main.sh"
    script.write_bytes((DS / "release/sync_to_main.sh").read_bytes())
    (rel / "public-manifest.txt").write_text("readme.md\ntests/test_one.py\n")
    (root / "readme.md").write_text("fixture")
    (root / "tests").mkdir()
    (root / "tests/test_one.py").write_text("fixture")
    events = root / "events"
    bin_dir = root / "fake-bin"
    bin_dir.mkdir()

    def executable(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    executable(bin_dir / "git", r'''
echo "git $*" >> "$EVENTS"
if [[ ${1:-} == -C ]]; then shift 2; fi
case "$1" in
 rev-parse)
  case "$2" in --abbrev-ref) echo dev;; origin/main) echo mainsha;; *) echo devsha;; esac;;
 status|fetch|update-index|rm|add) exit 0;;
 worktree) if [[ $2 == add ]]; then mkdir -p "$5"; fi;;
 log) echo 'Dev-Source: previous';;
 archive) tar -c -C "$FIXTURE_ROOT" readme.md tests/test_one.py;;
 ls-files|ls-tree) exit 0;;
 diff)
  if [[ $* == *--name-only* ]]; then echo readme.md
  elif [[ $* == *--stat* ]]; then echo fixture
  elif [[ $* == *devsha* && $MODE == bytes ]]; then exit 1
  else exit 0; fi;;
 commit|push) echo FORBIDDEN >> "$EVENTS"; exit 99;;
 *) exit 98;;
esac
''')
    executable(root / ".venv/bin/python", r'''
echo "python $*" >> "$EVENTS"
case "$1" in
 *check_public_lang.py) echo PUBLIC_LANG=PASS;;
 *check_public_paths.py) echo PUBLIC_PATHS=PASS;;
 *check_manifest.py)
  if [[ $* == *--submodules* ]]; then
   [[ $MODE != submodules ]] || exit 1
   echo 'SUBMODULE_PUBLIC=PASS repos=5 public=5 reachable=5'
  elif [[ $* == *--tree* ]]; then
   [[ $MODE != tree ]] || exit 1
   echo 'PUBLIC_MANIFEST=PASS missing=0 extra=0'
  else
   [[ $MODE != collect ]] || exit 1
   echo 'PUBLIC_MANIFEST=PASS missing=0 extra=0 collected=1'
  fi;;
 -m)
  [[ $MODE != tests ]] || exit 1
  echo 'TEST_RESOURCE=PASS native_reset=0 gpu_init=0 weights=0 network=0 violations=0 not_verified=0';;
 -c) [[ $MODE != completeness ]] || exit 1; echo PUBLIC_TEST_COMPLETENESS=PASS;;
 *) exit 97;;
esac
''')
    executable(bin_dir / "timeout", r'''
shift 3
if [[ $MODE == timeout ]]; then exit 124; fi
exec "$@"
''')
    if mode == "tee":
        executable(bin_dir / "tee", 'exit 23\n')
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               FIXTURE_ROOT=str(root), EVENTS=str(events), MODE=mode)
    result = subprocess.run(["bash", str(script), "--dry-run"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=15)
    return result, events.read_text()


@pytest.mark.parametrize("mode", ["tree", "bytes", "submodules", "collect", "tests", "completeness", "timeout", "tee"])
def test_sync_main_rejects_invalid_candidate_and_failed_gate(tmp_path, mode):
    result, events = _sync_fixture(tmp_path, mode)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "SYNC_MAIN=NOOP" not in result.stdout and "SYNC_MAIN=DRYRUN" not in result.stdout
    assert "FORBIDDEN" not in events


def test_sync_main_noop_requires_all_gates(tmp_path):
    result, events = _sync_fixture(tmp_path, "ok")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SYNC_MAIN=NOOP" in result.stdout and "PUBLIC_BYTES=PASS" in result.stdout
    for gate in ("check_public_lang.py", "check_public_paths.py", "--collect", "--submodules", "--tree", "--junitxml", "devsha -- readme.md"):
        assert gate in events, (gate, events)
    assert "FORBIDDEN" not in events

if shutil.which("setsid") is None or shutil.which("rsync") is None or shutil.which("ffmpeg") is None:  # pragma: no cover
    pytest.skip("未验证：缺 setsid、rsync 或 ffmpeg", allow_module_level=True)


def _bench_src() -> str:
    import importlib.util

    own = F.REPO / "third_party" / "robomme_benchmark" / "src"
    if (own / "robomme_ood").is_dir():
        return str(own)
    spec = importlib.util.find_spec("robomme_ood")
    return str(Path(next(iter(spec.submodule_search_locations))).parent)


def _port_free(p: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p))
        except OSError:
            return False
    return True


def _free_seat_idx() -> int:
    """找一个席号，使其四个策略的基础端口段（各 +0～+9）都空闲。"""
    for idx in range(97, 40, -1):
        base = 18000 + 100 * idx
        if all(_port_free(base + d) for d in range(40)):
            return idx
    raise RuntimeError("找不到空闲端口段")


def _killpg(pid: int, sig=signal.SIGKILL) -> None:
    """按进程组发信号；绝不对 pytest 自己所在的进程组发（那时退回只发给该进程）。"""
    try:
        pg = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        if pg != os.getpgid(0):
            os.killpg(pg, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _run(cmd, env, timeout=90):
    """bash 本身起在新会话里；超时或异常时按 pgid 杀整组。"""
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except BaseException:
        _killpg(p.pid)
        p.communicate()
        raise
    return p.returncode, out


def _popen(cmd, env):
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def _recorded_pids(tmp: Path) -> list[int]:
    """本用例记下的全部 pid：假引擎每次启动的事件行，加脚本写的 ``.v8-pgids``（setsid 进程组首进程）。"""
    pids = {int(r["pid"]) for r in F.read_jsonl(tmp / "fake.jsonl") if r.get("event") == "start"}
    for f in tmp.rglob(".v8-pgids"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                pids.add(int(parts[1]))
    return sorted(pids)


def _is_ours(pid: int) -> bool:
    """防 PID 复用误杀：只认命令行确属假引擎、假解释器包装 ``fakepy``（exec 之前的瞬间）或本测试起的席位脚本的进程。"""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return any(x in cmd for x in ("seat_fake_engine", "fakepy") + SCRIPTS)


def _reap(tmp: Path, grace: float = 3.0) -> list[int]:
    """对仍存活的已记录进程按进程组 SIGTERM，等至多 grace 秒，仍在的 SIGKILL 并再等至多 5 秒。返回动过手的 pid。"""
    live = [pid for pid in _recorded_pids(tmp) if _alive(pid) and _is_ours(pid)]
    for pid in live:
        _killpg(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < grace and any(_alive(p) for p in live):
        time.sleep(0.05)
    for pid in live:
        if _alive(pid):
            _killpg(pid, signal.SIGKILL)
    t0 = time.time()
    while time.time() - t0 < 5 and any(_alive(p) for p in live):
        time.sleep(0.05)
    return live


def _events(rig, role=None, event=None):
    rows = F.read_jsonl(rig["tmp"] / "fake.jsonl")
    return [r for r in rows if (role is None or r["role"] == role) and (event is None or r["event"] == event)]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # 僵尸进程也算已结束
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _wait_line(path: Path, needle: str, timeout: float = 30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists() and needle in path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(0.1)
    raise AssertionError(f"{timeout}s 内日志 {path} 没有出现 {needle!r}")


def _wait_events(rig, event: str, count: int, timeout: float = 30.0) -> list[dict]:
    """轮询假引擎事件账本，直到 ``event`` 事件数 ≥ ``count``（有限超时）。日志里出现 CLIENT_START 只说明席位脚本
    已发起客户端，不保证客户端进程已写下自己的 start 事件；按事件数等才不会在两者之间的窗口里提前动手。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = _events(rig, None, event)
        if len(rows) >= count:
            return rows
        time.sleep(0.05)
    raise AssertionError(f"{timeout}s 内 {event} 事件不足 {count} 条：{_events(rig, None, event)}")


def _mp4_frames(p: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True)
    return int(out.stdout.strip())


def _tsv_epochs(p: Path) -> list[list[str]]:
    return [line.split("\t") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hard0_rows(n: int) -> list[dict]:
    """hard-verify 执行身份行（手写；假引擎不解析 builder）。"""
    out = []
    for k in range(n):
        seed = 510300 + k
        out.append({"task": "PickXtimes", "tier": "xhard0", "seed": seed, "candidate": None, "builder_episode": k,
                    "source_episode": k, "spec_sha256": None, "key": f"PickXtimes_xhard0_{seed}"})
    return out


@pytest.fixture
def rig(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fakepy = bin_dir / "fakepy"
    fakepy.write_text(f"""#!/usr/bin/env bash
case "$1" in
  -|-c) exec "{sys.executable}" "$@";;
  -m) if [[ "$2" == ponderpounce.eval.robomme_server ]]; then shift 2; exec "{sys.executable}" "{ENGINE}" server "$@"; fi
      exec "{sys.executable}" "$@";;
  *pp_server_wrap.py) echo "$1" >> "{tmp_path}/pp-wrap-argv0.txt"; shift; exec "{sys.executable}" "{ENGINE}" server "$@";;
  *smvla_server.py|*serve_policy.py|*policy_server_wrap.py) shift; exec "{sys.executable}" "{ENGINE}" server "$@";;
  *seat.py) shift; exec "{sys.executable}" "{ENGINE}" seat "$@";;
  *pp_official_runner.py|*official_hard_runner.py) shift; exec "{sys.executable}" "{ENGINE}" runner "$@";;
  *) exec "{sys.executable}" "$@";;
esac
""", encoding="utf-8")
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for p in (fakepy, bin_dir / "nvidia-smi"):
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    pp_ckpt = tmp_path / "pp-ckpt"
    pp_ckpt.mkdir()
    (pp_ckpt / "norm_stats.json").write_text("{}", encoding="utf-8")
    hard0 = _hard0_rows(2)
    shard0 = tmp_path / "shard0-00.json"
    shard0.write_text(json.dumps(hard0), encoding="utf-8")
    ids = tmp_path / "ids.jsonl"
    ids.write_text("".join(json.dumps(dict(r, dataset="hard-verify")) + "\n" for r in hard0), encoding="utf-8")
    env = dict(os.environ)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", BENCH_PY=str(fakepy), SMVLA_PY=str(fakepy),
               SGEVAL_CLIENT_PY=str(fakepy), PP_PY=str(fakepy),
               SEAT_POLL_S="0.2", SEAT_READY_POLL_S="0.1", FAKE_LOG=str(tmp_path / "fake.jsonl"),
               FAKE_STATE=str(tmp_path / "client-count"), FAKE_SERVER_MODE="ok", FAKE_CLIENT_CODES="0",
               FAKE_STOP_LOG=str(tmp_path / "stop-server.log"), POLICY_SEED="7")
    yield {"tmp": tmp_path, "fakepy": fakepy, "env": env, "ids": ids, "pp_ckpt": pp_ckpt, "shard0": shard0,
           "hard0": hard0, "idx": _free_seat_idx()}
    _reap(tmp_path)
    left = [pid for pid in _recorded_pids(tmp_path) if _alive(pid) and _is_ours(pid)]
    assert left == [], f"teardown 后仍有存活进程 {left}"


BUDGET_CAPS = ("--trajectory-cap", "870", "--shared-infra-cap", "50", "--expired-cap", "50",
               "--planned-first-tries", "821")



# ---------------------------------------------------------------- 媒体函数库与原侧函数库


STUB_RENDERER = r'''#!/usr/bin/env python3
"""测试桩：按 S2a 约定的 CLI（<局目录> --source raw --jobs 1 --ffmpeg F --out-subdir official）读原始帧出视频。

只为驱动席位收尾链：帧数取 frames-front.jsonl 行数，出一个同帧数的灰色 mp4 与 render.json（identity／route／
frames／output_fingerprint）；FAKE_RENDER_FAIL=1、原始帧已不在或无 trace 时失败；每次调用记一行 FAKE_RENDER_LOG。"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("episodes", nargs="*", type=Path)
ap.add_argument("--source", choices=["auto", "mp4", "raw"], default="auto")  # @@SOURCE@@
ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--ffmpeg", default="ffmpeg")
ap.add_argument("--out-subdir", default="official")
a = ap.parse_args()
ok = True
for d in a.episodes:
    ev = {"dir": d.name, "source": getattr(a, "source", None), "jobs": a.jobs, "trace": (d / "trace.jsonl").is_file(),
          "episode_mp4": (d / "episode.mp4").exists(), "raw": (d / "front.mkv").exists(), "t": time.time()}
    if os.environ.get("FAKE_RENDER_LOG"):
        with open(os.environ["FAKE_RENDER_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")
    if os.environ.get("FAKE_RENDER_FAIL") == "1" or not ev["raw"] or not ev["trace"]:
        print(f"OFFICIAL_RENDER=FAIL dir={d.name} reason=stub_fail")
        ok = False
        continue
    rows = [json.loads(x) for x in (d / "trace.jsonl").read_text().splitlines() if x.strip()]
    n = sum(1 for x in (d / "frames-front.jsonl").read_text().splitlines() if x.strip())
    out = d / a.out_subdir
    out.mkdir(exist_ok=True)
    ident = rows[0].get("identity") or {}
    tmp = out / ".render-tmp.mp4"
    subprocess.run([a.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=32x16:r=30",
                    "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", str(tmp)], check=True)
    v = out / f"official-rerender__{ident.get('key', d.name)}.mp4"
    os.replace(tmp, v)
    sha = hashlib.sha256(v.read_bytes()).hexdigest()
    (out / "render.json").write_text(json.dumps({"identity": ident, "route": rows[0].get("route"), "frames": n,
                                                 "output_fingerprint": {"sha256": sha}}))
    print(f"OFFICIAL_RENDER=PASS dir={d.name} frames={n} source_kind=raw-new")
sys.exit(0 if ok else 1)
'''
STUB_RENDERER_NO_SOURCE = STUB_RENDERER.replace(
    'ap.add_argument("--source", choices=["auto", "mp4", "raw"], default="auto")  # @@SOURCE@@', "")
KEY0 = "PickXtimes_xhard0_510300"
STOP_STUB = """import json, os, sys
with open(os.environ["FAKE_STOP_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
print("SERVER_STOP_BY_METADATA result=gone (stub)")
"""


def _media_repo(repo: Path, renderer: str = STUB_RENDERER) -> Path:
    """执行副本最小形态（拆仓后布局）：真实席位脚本、原侧函数库、媒体函数库与验收工具（含其读的别名表）、桩重绘器、
    假的 scripts/evaluate.py（只记 --stop-server 调用）与评估包占位。"""
    for sub, names in (("orig", ("run_official_hard.sh", "orig_seat_lib.sh", "pair_seat.sh")),
                       ("gl", ("run_eval_gl.sh", "seat_media_lib.sh", "budget_ledger.py")),
                       ("media", ("official_media_check.py",))):
        d = repo / "dev-scripts" / sub
        d.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(DS / sub / name, d / name)
    (repo / "dev-scripts" / "media" / "render_official_video.py").write_text(renderer, encoding="utf-8")
    pkg = repo / "src" / "robomme_ood_eval"
    (pkg / "models").mkdir(parents=True, exist_ok=True)
    (pkg / "servers").mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(F.REPO / "src" / "robomme_ood_eval" / "models" / "_official_defs.py", pkg / "models" / "_official_defs.py")
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "evaluate.py").write_text(STOP_STUB, encoding="utf-8")
    return repo


def _px(v: int) -> bytes:
    return bytes([v % 256, (v * 7) % 256, (v * 13) % 256]) * (16 * 16)


def _raw_episode(d: Path) -> None:
    """录像器真实格式：front/wrist.mkv（FFV1 16×16，3 帧编码）+ frames-<stream>.jsonl 4 行（含一帧重复）+ summary.json。"""
    d.mkdir(parents=True, exist_ok=True)
    for i, stream in enumerate(("front", "wrist")):
        raw = b"".join(_px(v) for v in (10 + i, 50 + i, 90 + i))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", "16x16", "-r", "30", "-i", "-", "-c:v", "ffv1", str(d / f"{stream}.mkv")], input=raw, check=True)
        (d / f"frames-{stream}.jsonl").write_text(
            "".join(json.dumps({"idx": idx, "enc": enc, "sha256": f"{stream}-{enc}"}) + "\n"
                    for idx, enc in enumerate((0, 1, 1, 2))), encoding="utf-8")
    (d / "summary.json").write_text('{"RECORDER_VERIFY": "PASS"}', encoding="utf-8")
    (d / "meta.json").write_text('{"codec": "ffv1", "level": 0}', encoding="utf-8")  # 旧录制器：无 raw_codec 字段


def _trace_dir(td: Path, key: str = KEY0, attempt: int = 1) -> None:
    """契约 trace（C6 身份、C8 frames_recorded = 演示 1 + 初始 1 + 观测 2 = 4）。"""
    td.mkdir(parents=True, exist_ok=True)
    ident = {"task": "PickXtimes", "tier": "xhard0", "seed": 510300, "source_episode": 0, "builder_episode": 0,
             "dataset": "hard-verify", "key": key, "attempt": attempt}
    rows = [{"kind": "header", "schema": "sgeval-trace/1", "route": "pp/new", "identity": ident, "max_steps": 1300},
            {"kind": "end", "status": "success", "terminal_reason": "success", "exec_steps": 2, "demo_frames": 1,
             "steps_attempted": 2, "steps_observed": 2, "frames_recorded": 4}]
    (td / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _raw_fingerprint(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.iterdir()) if p.name.endswith(".mkv") or p.name.startswith("frames-")}


def _finish(repo: Path, src: Path, tsrc: Path | str, root: Path, name: str, **env_extra) -> tuple[int, str]:
    """``source orig_seat_lib.sh`` 后调真实 ``finish_episode_dir``（并入轨迹 → 官方重绘 → 转码 → 原子发布）。"""
    env = dict(os.environ, TOOL_PY=sys.executable, SGEVAL_OFFICIAL_RENDER="1", **env_extra)
    script = 'source "$1/dev-scripts/orig/orig_seat_lib.sh"; finish_episode_dir "$2" "$3" "$4" "$5"; echo "FINISH_RC=$?"'
    p = subprocess.run(["bash", "-c", script, "_", str(repo), str(src), str(tsrc), str(root), name], env=env,
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def _render_log(p: Path) -> list[dict]:
    return F.read_jsonl(p) if p.exists() else []


def test_finish_chain_renders_official_after_trace_merge_before_transcode(tmp_path):
    repo = _media_repo(tmp_path / "repo")
    src, tsrc, root = tmp_path / "rec" / f"{KEY0}.a1", tmp_path / "trace" / f"{KEY0}.a1", tmp_path / "pub"
    _raw_episode(src)
    _trace_dir(tsrc)
    log = tmp_path / "render.jsonl"
    rc, out = _finish(repo, src, tsrc, root, src.name, FAKE_RENDER_LOG=str(log))
    assert rc == 0 and "FINISH_RC=0" in out, out
    (ev,) = _render_log(log)
    # 调用顺序：轨迹已并入、原始帧仍在、尚未转码；按约定 CLI 调用
    assert ev["trace"] and ev["raw"] and not ev["episode_mp4"] and ev["source"] == "raw" and ev["jobs"] == 1
    lines = out.splitlines()
    i_render = next(i for i, x in enumerate(lines) if x.startswith("OFFICIAL_RENDER=PASS"))
    i_verify = next(i for i, x in enumerate(lines) if x.startswith("OFFICIAL_VERIFY=PASS"))
    i_tc = next(i for i, x in enumerate(lines) if x.startswith("REC_TRANSCODE ") and "result=ok" in x)
    assert i_render < i_verify < i_tc
    d = root / src.name
    assert sorted(p.name for p in d.glob("*.mp4")) == ["episode.mp4"]
    (off,) = list((d / "official").glob("*.mp4"))
    assert _mp4_frames(off) == 4 and _mp4_frames(d / "episode.mp4") == 4
    assert not list(d.glob("*.mkv")) and not (d / "official-render.failed").exists()
    assert not src.exists() and not tsrc.exists()
    # 发布后的目录过全量验收（冻结清单 + 账本 + 发布根）
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"task": "PickXtimes", "tier": "xhard0", "seed": 510300, "key": KEY0}]))
    ledger = tmp_path / "pp.ledger.jsonl"
    ledger.write_text(json.dumps({"kind": "accept", "key": KEY0, "attempt_id": "x", "attempt_no": 1,
                                  "accepted_attempt_id": "x", "status": "success"}) + "\n")
    p = subprocess.run([sys.executable, str(DS / "media" / "official_media_check.py"), "--manifest", str(manifest), "--ledger",
                        str(ledger), "--root", str(root), "--dataset", "hard-verify", "--route", "pp/new"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "OFFICIAL_MEDIA=PASS total=1 skip=0 fail=0 no_frame_error=0" in p.stdout


def test_finish_chain_render_failure_keeps_raw_then_reentry_recovers(tmp_path):
    repo = _media_repo(tmp_path / "repo")
    src, tsrc, root = tmp_path / "rec" / f"{KEY0}.a1", tmp_path / "trace" / f"{KEY0}.a1", tmp_path / "pub"
    _raw_episode(src)
    _trace_dir(tsrc)
    before = _raw_fingerprint(src)
    rc, out = _finish(repo, src, tsrc, root, src.name, FAKE_RENDER_FAIL="1")
    assert rc == 0 and "FINISH_RC=0" in out, out  # 重绘失败照常发布
    assert "OFFICIAL_RENDER=FAIL" in out and f"OFFICIAL_RENDER_KEEP_RAW dir={src.name}" in out
    d = root / src.name
    assert _raw_fingerprint(d) == before  # 原始帧逐文件指纹不变、随目录发布
    assert (d / "official-render.failed").is_file() and "OFFICIAL_RENDER=FAIL" in (d / "official-render.failed").read_text()
    assert sorted(p.name for p in d.glob("*.mp4")) == ["episode.mp4"] and _mp4_frames(d / "episode.mp4") == 4
    tc = json.loads((d / "transcode.json").read_text())
    assert tc["result"] == "ok" and tc["raw_kept"] is True
    assert not (d / "official").exists() or not list((d / "official").glob("*.mp4"))
    # 重入：把发布出的目录拿回节点再走一遍收尾（重绘器恢复正常）
    src2 = tmp_path / "rec2" / src.name
    shutil.copytree(d, src2)
    rc, out = _finish(repo, src2, "", tmp_path / "pub2", src.name)
    assert rc == 0 and "FINISH_RC=0" in out, out
    d2 = tmp_path / "pub2" / src.name
    assert len(list((d2 / "official").glob("*.mp4"))) == 1
    assert sorted(p.name for p in d2.glob("*.mp4")) == ["episode.mp4"] and _mp4_frames(d2 / "episode.mp4") == 4
    assert not (d2 / "official-render.failed").exists() and not list(d2.glob("*.mkv"))
    assert "raw_kept" not in json.loads((d2 / "transcode.json").read_text())


def test_render_official_dir_kept_only_after_full_verify(tmp_path):
    repo = _media_repo(tmp_path / "repo")
    d = tmp_path / "ep" / f"{KEY0}.a1"
    _raw_episode(d)
    _trace_dir(d)
    log = tmp_path / "render.jsonl"
    env = dict(os.environ, TOOL_PY=sys.executable, FAKE_RENDER_LOG=str(log))
    lib = repo / "dev-scripts" / "gl" / "seat_media_lib.sh"
    call = ["bash", "-c", 'source "$1"; render_official_dir "$2"; echo "RC=$?"', "_", str(lib), str(d)]
    out = subprocess.run(call, env=env, capture_output=True, text=True).stdout
    assert "RC=0" in out and len(_render_log(log)) == 1, out
    # 已有官方视频且核验通过：KEPT，不再调用重绘器
    out = subprocess.run(call, env=env, capture_output=True, text=True).stdout
    assert "OFFICIAL_RENDER=KEPT" in out and "RC=0" in out and len(_render_log(log)) == 1, out
    # 截断官方视频：不得 KEPT；旧目录移到隐藏留证目录，重绘后恰一个可解码视频
    (v,) = list((d / "official").glob("*.mp4"))
    v.write_bytes(v.read_bytes()[: v.stat().st_size // 2])
    out = subprocess.run(call, env=env, capture_output=True, text=True).stdout
    assert "OFFICIAL_RENDER=KEPT" not in out and "OFFICIAL_RENDER_REJECTED" in out and "RC=0" in out, out
    assert len(_render_log(log)) == 2
    (v2,) = list((d / "official").glob("*.mp4"))
    assert _mp4_frames(v2) == 4
    assert len(list(d.glob(".official-rejected-*"))) == 1


def test_render_official_dir_without_source_flag_fails_cleanly(tmp_path):
    repo = _media_repo(tmp_path / "repo", renderer=STUB_RENDERER_NO_SOURCE)
    d = tmp_path / "ep" / f"{KEY0}.a1"
    _raw_episode(d)
    _trace_dir(d)
    lib = repo / "dev-scripts" / "gl" / "seat_media_lib.sh"
    p = subprocess.run(["bash", "-c", 'source "$1"; render_official_dir "$2"; echo "RC=$?"', "_", str(lib), str(d)],
                       env=dict(os.environ, TOOL_PY=sys.executable), capture_output=True, text=True)
    assert "reason=renderer_no_source_raw" in p.stdout and "RC=0" not in p.stdout, p.stdout
    assert not (d / "official").exists()




def test_transcode_keeps_permanent_av1_raw(tmp_path):
    """拆仓后录制器的原始帧是永久 AV1（meta.json 记 raw_codec=av1-yuv444p）：不进旧的「转码后删原始帧」分支。"""
    repo = _media_repo(tmp_path / "repo")
    d = tmp_path / "ep" / f"{KEY0}.a1"
    _raw_episode(d)
    (d / "meta.json").write_text(json.dumps({"raw_codec": "av1-yuv444p"}), encoding="utf-8")
    before = _raw_fingerprint(d)
    lib = repo / "dev-scripts" / "gl" / "seat_media_lib.sh"
    p = subprocess.run(["bash", "-c", 'source "$1"; transcode_episode_dir "$2"; echo "RC=$?"', "_", str(lib), str(d)],
                       env=dict(os.environ, TOOL_PY=sys.executable), capture_output=True, text=True)
    assert "result=permanent_raw" in p.stdout and "RC=0" in p.stdout, p.stdout + p.stderr
    assert _raw_fingerprint(d) == before and not (d / "episode.mp4").exists()


@pytest.mark.parametrize("meta", [None, "{坏 json", '{"raw_codec": "h265"}', '{"raw_codec": "ffv1"}'],
                         ids=["missing", "unreadable", "unknown_codec", "explicit_ffv1"])
def test_transcode_deletes_raw_only_for_explicit_ffv1(tmp_path, meta):
    """kind=new：meta.json 缺失或读不出 → refused_unreadable_meta、原始帧不动；未知编码 → permanent_raw 不动；只有明确
    FFV1 才转码并删原始帧。"""
    repo = _media_repo(tmp_path / "repo")
    d = tmp_path / "ep" / f"{KEY0}.a1"
    _raw_episode(d)
    if meta is None:
        (d / "meta.json").unlink()
    else:
        (d / "meta.json").write_text(meta, encoding="utf-8")
    before = _raw_fingerprint(d)
    lib = repo / "dev-scripts" / "gl" / "seat_media_lib.sh"
    p = subprocess.run(["bash", "-c", 'source "$1"; transcode_episode_dir "$2"; echo "RC=$?"', "_", str(lib), str(d)],
                       env=dict(os.environ, TOOL_PY=sys.executable), capture_output=True, text=True)
    assert "RC=0" in p.stdout, p.stdout + p.stderr
    if meta == '{"raw_codec": "ffv1"}':
        assert "result=ok" in p.stdout and not list(d.glob("*.mkv")) and _mp4_frames(d / "episode.mp4") == 4
    else:
        want = "permanent_raw" if meta == '{"raw_codec": "h265"}' else "refused_unreadable_meta"
        assert f"result={want}" in p.stdout, p.stdout
        assert _raw_fingerprint(d) == before and not (d / "episode.mp4").exists()


def test_orig_seat_lib_refuses_direct_execution(tmp_path):
    p = subprocess.run(["bash", str(DS / "orig" / "orig_seat_lib.sh")], capture_output=True, text=True)
    assert p.returncode == 2 and "只作库 source" in p.stderr


# ---------------------------------------------------------------- run_eval_gl.sh（拆仓后）


@pytest.fixture
def gl_repo(rig):
    """执行副本最小形态：真实脚本、假解释器、空的 PonderPounce 子模块目录、原侧驱动与服务外壳占位文件；评估包从副本
    ``src`` 导入（RUN_INPUTS 核对它），``robomme_ood`` 从 benchmark 子模块导入。"""
    repo = rig["tmp"] / "repo"
    _media_repo(repo)
    for rel in ("dev-scripts/orig/pp_official_runner.py", "dev-scripts/orig/official_hard_runner.py",
                "dev-scripts/gl/seat.py", "src/robomme_ood_eval/servers/pp_server_wrap.py",
                "src/robomme_ood_eval/servers/policy_server_wrap.py"):
        (repo / rel).write_text("# 占位：由假解释器分派到假引擎\n", encoding="utf-8")
    (repo / "third_party" / "PonderPounce").mkdir(parents=True)
    for rel in (".venv/bin/python",):
        (repo / rel).parent.mkdir(parents=True)
        (repo / rel).symlink_to(rig["fakepy"])
    rig["env"]["PYTHONPATH"] = f"{repo / 'src'}:{_bench_src()}"
    yield repo
    _reap(rig["tmp"])


def _gl_cmd(rig, repo, *extra, policies="pp", out=None):
    cmd = ["bash", str(repo / "dev-scripts" / "gl" / "run_eval_gl.sh"), "--policies", policies, "--policy-seed", "7",
           "--identities", str(rig["ids"]), "--budget-ledger", str(rig["tmp"] / "budget" / "budget-ledger.jsonl"),
           "--run", "R", "--poll-s", "1", "--term-grace-s", "5", "--seat", "S1"]
    if out is not None:
        cmd += ["--out", str(out)]
    return cmd + list(extra)


def _seat_argv(rig) -> list[list[str]]:
    return [r["argv"] for r in _events(rig, "seat", "start")]


def _opt(argv: list[str], name: str):
    return argv[argv.index(name) + 1] if name in argv else None


def test_gl_runbook_args_reach_seat(rig, gl_repo):
    """runbook 写法：参数原样落到 seat.py run；预算缺省为 §五口径；产物根缺省 <repo>/artifacts/<run>/gl；透传参数原样。"""
    rc, out = _run(_gl_cmd(rig, gl_repo, "--groundsg-variant", "ground-sg-oracle", "--ckpt", "/ck/x",
                           policies="groundsg"), rig["env"])
    assert rc == 0, out
    ri = F.verdict(out.splitlines(), "RUN_INPUTS")
    assert ri[""] == "PASS" and ri["robomme_ood_eval"] == str(gl_repo / "src/robomme_ood_eval/__init__.py"), out
    (argv,) = _seat_argv(rig)
    assert argv[0] == "run" and _opt(argv, "--policy") == "groundsg"
    assert _opt(argv, "--groundsg-variant") == "ground-sg-oracle" and _opt(argv, "--policy-seed") == "7"
    assert _opt(argv, "--identities") == str(rig["ids"]) and _opt(argv, "--out") == str(gl_repo / "artifacts/R/gl")
    assert _opt(argv, "--budget-ledger") == str(rig["tmp"] / "budget" / "budget-ledger.jsonl")
    assert [_opt(argv, k) for k in ("--trajectory-cap", "--reset-cap", "--shared-infra-cap", "--expired-cap",
                                    "--planned-first-tries", "--infra-retries", "--gpus", "--seat", "--ckpt")] == \
        ["26", "60", "0", "0", "26", "0", "0", "S1", "/ck/x"]
    assert "GL_POLICY_DONE policy=groundsg-ground-sg-oracle rc=0 restarts=0 noprog=0" in out
    assert "GL_SEAT_DONE outcome=pass rc=0" in out and out.rstrip().splitlines()[-1] == "EXIT_CODE=0"


def test_gl_watchdog_exit_stops_left_server_without_restart(rig, gl_repo):
    """客户端以 75 退出（看门狗只杀客户端）：按 SERVER_LEFT 停残留服务端；重起次数缺省 0 → 不重起、整席 rc=75。"""
    rig["env"]["FAKE_CLIENT_CODES"] = "75,0"
    rc, out = _run(_gl_cmd(rig, gl_repo), rig["env"])
    assert rc == 75, out
    assert len(_seat_argv(rig)) == 1 and "GL_CLIENT_RESTART" not in out
    stops = F.read_jsonl(rig["tmp"] / "stop-server.log")
    assert stops and all(s[0] == "--stop-server" and s[1].endswith("server-metadata-18999.json") for s in stops)
    assert "GL_SEAT_DONE outcome=fail rc=75" in out


def test_gl_client_restart_when_allowed(rig, gl_repo):
    rig["env"]["FAKE_CLIENT_CODES"] = "75,0"
    rc, out = _run(_gl_cmd(rig, gl_repo, "--client-restarts", "1"), rig["env"])
    assert rc == 0, out
    assert len(_seat_argv(rig)) == 2 and "GL_CLIENT_RESTART policy=pp restart=1/1 reason=watchdog_exit" in out
    assert "GL_POLICY_DONE policy=pp rc=0 restarts=1 noprog=0" in out


def test_gl_noprogress_watchdog_kills_hung_client(rig, gl_repo):
    """客户端卡死（进度不再更新）：超过 --noprog-s 即 TERM、计 rc=124；重起 0 次即停。"""
    rig["env"]["FAKE_CLIENT_CODES"] = "-1"
    rc, out = _run(_gl_cmd(rig, gl_repo, "--noprog-s", "2"), rig["env"], timeout=60)
    assert rc == 124, out
    assert "GL_NOPROGRESS policy=pp" in out and "GL_POLICY_DONE policy=pp rc=124 restarts=0 noprog=1" in out
    assert [r["event"] for r in _events(rig, "seat")] == ["start", "term"]


def test_gl_term_finalizes_once(rig, gl_repo):
    rig["env"]["FAKE_CLIENT_CODES"] = "-1"
    p = _popen(_gl_cmd(rig, gl_repo), rig["env"])
    try:
        _wait_events(rig, "start", 1)
        time.sleep(0.5)
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            _killpg(p.pid)
            p.wait()
    assert p.returncode == 143
    assert out.count("GL_SEAT_DONE") == 1 and "GL_SEAT_DONE outcome=aborted rc=143" in out
    assert out.rstrip().splitlines()[-1] == "EXIT_CODE=143"
    for r in _events(rig, None, "start"):
        assert not _alive(r["pid"]), r


@pytest.mark.parametrize("drop", ["--policies", "--policy-seed", "--identities", "--budget-ledger", "--run"])
def test_gl_required_args_exit_2(rig, gl_repo, drop):
    cmd = _gl_cmd(rig, gl_repo)
    i = cmd.index(drop)
    del cmd[i:i + 2]
    rc, out = _run(cmd, rig["env"])
    assert rc == 2 and "GL_SEAT_DONE outcome=fail rc=2 reason=bad_args" in out
    assert _events(rig) == []


def test_gl_groundsg_without_variant_exit_2(rig, gl_repo):
    rc, out = _run(_gl_cmd(rig, gl_repo, policies="groundsg"), rig["env"])
    assert rc == 2 and "--groundsg-variant" in out


def test_gl_run_inputs_block_when_package_not_in_repo(rig, gl_repo):
    """评估包导入到执行副本之外（editable 指向别处）：RUN_INPUTS=FAIL、RUN_BLOCKED、不起客户端。"""
    rig["env"]["PYTHONPATH"] = f"{F.REPO / 'src'}:{_bench_src()}"
    rc, out = _run(_gl_cmd(rig, gl_repo), rig["env"])
    assert rc == 3, out
    ri = F.verdict(out.splitlines(), "RUN_INPUTS")
    assert ri[""] == "FAIL" and "robomme_ood_eval_not_in_repo" in ri["reason"]
    assert "RUN_BLOCKED reason=run_inputs" in out and _events(rig) == []


# ---------------------------------------------------------------- run_official_hard.sh／pair_seat.sh


def _official_cmd(rig, repo, stage, seat, *extra, budget=1):
    # 第三阶段：run_official_hard.sh 对 pp 原侧同样要求 --policy-seed 与五个预算参数（不转发给 pp 驱动，见 F）
    return ["bash", str(repo / "dev-scripts" / "orig" / "run_official_hard.sh"), "--run-name", "R", "--seat", seat,
            "--repo", str(repo), "--stage", str(stage), "--shard", str(rig["shard0"]), "--policy", "pp",
            "--pp-ckpt", str(rig["pp_ckpt"]), "--dataset", "hard-verify", "--max-steps", "1300",
            "--infra-retry-budget", str(budget), "--sync-interval", "1", "--local-root", str(rig["tmp"] / "local"),
            "--policy-seed", "7", "--budget-ledger", str(rig["tmp"] / "budget" / "budget-ledger.jsonl"), *BUDGET_CAPS,
            *extra]


def test_official_pp_retries_after_server_restart_and_publishes(rig, gl_repo):
    seat = f"{rig['idx']:02d}"
    stage = rig["tmp"] / "stage"
    rig["env"]["FAKE_RUNNER_INFRA_ONCE"] = "1"
    rc, out = _run(_official_cmd(rig, gl_repo, stage, seat), rig["env"])
    assert rc == 0, out
    assert f"OFFICIAL_SEAT_DONE seat={seat} policy=pp outcome=pass rc=0" in out
    assert out.rstrip().splitlines()[-1] == "EXIT_CODE=0"
    k1, k2 = (r["key"] for r in rig["hard0"])
    runs = _events(rig, "runner", "start")
    assert [(r["attempt"], r["only"]) for r in runs] == [(1, [k1, k2]), (2, [k2])]
    assert all(r["max_steps"] == "1300" and r["variant"] is None for r in runs)
    servers = _events(rig, "server", "start")
    assert len(servers) == 2  # 重试前先重启服务
    assert _events(rig, "server", "term")[0]["t"] < runs[1]["t"]  # 旧服务先收掉，再重发同一身份
    rows = F.read_jsonl(stage / f"s{seat}" / "orig" / "pp" / "results.epochs.jsonl")
    assert [(r["key"], r["attempt"], r["infra"], r["server_epoch"]) for r in rows] == \
        [(k1, 1, False, 1), (k2, 1, True, 1), (k2, 2, False, 2)]
    sync = F.verdict(out.splitlines(), "SEAT_REC_SYNC")
    assert sync[""] == "PASS" and sync["transcoded"] == "3" and sync["frame_mismatch"] == "0"
    pub = stage / "media" / "pp" / "hard-verify" / "orig"
    for name in (f"{k1}.a1", f"{k2}.a1", f"{k2}.a2"):
        d = pub / name
        assert _mp4_frames(d / "episode.mp4") == 3
        assert not list((d / "frames").glob("*.rgb24")) and (d / "frames" / "frames.json").is_file()
        assert (d / "trace.jsonl").is_file()


def test_official_retry_budget_zero_leaves_missing(rig, gl_repo):
    seat = f"{rig['idx']:02d}"
    rig["env"]["FAKE_RUNNER_INFRA_ONCE"] = "1"
    rc, out = _run(_official_cmd(rig, gl_repo, rig["tmp"] / "stage", seat, budget=0), rig["env"])
    assert rc == 6, out
    assert "RUN_INCOMPLETE side=orig policy=pp" in out and "missing=1" in out
    assert len(_events(rig, "runner", "start")) == 1


def test_pair_seat_runs_orig_then_new_on_released_gpu(rig, gl_repo):
    """先原侧后新侧，两侧都 rc=0；新侧走拆仓后的 run_eval_gl.sh → seat.py：分片作身份清单、hard-verify、种子与预算参数
    原样转发（reset 计量上限缺省 141430，与原侧账本常量同一 config）。"""
    seat = f"{rig['idx']:02d}"
    stage = rig["tmp"] / "stage"
    ledger = rig["tmp"] / "budget" / "budget-ledger.jsonl"
    cmd = ["bash", str(gl_repo / "dev-scripts" / "orig" / "pair_seat.sh"), "--run-name", "R", "--seat", seat,
           "--repo", str(gl_repo), "--stage", str(stage), "--shard", str(rig["shard0"]), "--policy", "pp",
           "--pp-ckpt", str(rig["pp_ckpt"]), "--reset-budget", "10", "--infra-retry-budget", "0",
           "--orig-infra-retry-budget", "1", "--sync-interval", "1", "--local-root", str(rig["tmp"] / "local"),
           "--policy-seed", "7", "--budget-ledger", str(ledger), *BUDGET_CAPS]
    rc, out = _run(cmd, rig["env"], timeout=120)
    assert rc == 0, out
    assert f"PAIR_SEAT_DONE seat={seat} policy=pp orig_rc=0 new_rc=0 rc=0 outcome=pass" in out
    runner, seats = _events(rig, "runner", "start"), _events(rig, "seat", "start")
    assert len(runner) == 1 and len(seats) == 1 and runner[0]["t"] < seats[0]["t"]
    argv = seats[0]["argv"]
    assert _opt(argv, "--policy") == "pp" and _opt(argv, "--identities") == str(rig["shard0"])
    assert _opt(argv, "--dataset") == "hard-verify" and _opt(argv, "--policy-seed") == "7"
    assert _opt(argv, "--budget-ledger") == str(ledger) and _opt(argv, "--reset-cap") == "141430"
    assert [_opt(argv, k) for k in BUDGET_CAPS[::2]] == list(BUDGET_CAPS[1::2])
    assert _opt(argv, "--out") == str(stage / f"new-s{seat}") and _opt(argv, "--infra-retries") == "0"
    pub = stage / "media" / "pp" / "hard-verify" / "orig"
    assert sorted(p.name for p in pub.iterdir() if not p.name.startswith(".")) == \
        sorted(f"{r['key']}.a1" for r in rig["hard0"])


def test_teardown_reaps_orphans_when_production_cleanup_is_bypassed(rig, gl_repo):
    """故意让生产清理失效：客户端常驻，测试侧直接 SIGKILL 掉 run_eval_gl.sh 整个进程组（trap 来不及跑）。setsid 起的
    客户端成为孤儿仍存活；兜底函数必须把它收干净。"""
    rig["env"]["FAKE_CLIENT_CODES"] = "-1"
    p = _popen(_gl_cmd(rig, gl_repo), rig["env"])
    try:
        _wait_events(rig, "start", 1)
        _killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=10)
    finally:
        if p.poll() is None:
            _killpg(p.pid)
            p.wait()
    orphans = [r["pid"] for r in _events(rig, None, "start")]
    assert len(orphans) == 1 and all(_alive(x) for x in orphans)
    reaped = _reap(rig["tmp"])
    assert set(reaped) >= set(orphans)
    assert [x for x in orphans if _alive(x)] == []
