"""席位脚本测试（``test_seat_scripts.py``）的假引擎：由假解释器按脚本名分派，充当策略 server、常驻客户端或原侧驱动。

- ``server``（代替 ``smvla_server.py serve``、``serve_policy.py``、``python -m ponderpounce.eval.robomme_server``）：在
  ``--port`` 上监听回环端口，每个连接回一行 ``HTTP/1.0 200 OK``（PonderPounce 的 ``GET /health`` 就绪判定用它，端口探测
  连上即断也无妨）；``FAKE_SERVER_MODE``：``ok``（正常，收 TERM 即退）／``die_before_ready``（立即退出 1）／``die_after``
  （监听 ``FAKE_SERVER_LIFE`` 秒后退出 1）／``ignore_term``（忽略 TERM，只能被 KILL）。启动事件记完整 argv。
- ``client``（代替 ``env_client.py run``）：第 n 次启动按 ``FAKE_CLIENT_CODES`` 的第 n 项（超出取最后一项）行事：
  非负整数 = 写进度后以该码退出（0 时为身份清单每行写一条结果行和录像目录）；``-1`` = 一直睡（收 TERM 即退）。
  录像目录按录像器的真实格式写：``front.mkv``／``wrist.mkv`` 为 ffmpeg 现做的 16×16 FFV1（3 帧编码），
  ``frames-<stream>.jsonl`` 4 行（idx 0..3 → enc 0,1,1,2，含一帧重复），``summary.json``；给了 ``--trace-root`` 时在
  ``<trace-root>/<key>.a1/trace.jsonl`` 写一行轨迹。启动事件记数据集、步数、strict-cap、变体等透传参数。
- ``seat``（代替拆仓后的 ``dev-scripts/gl/seat.py run``）：第 n 次启动按 ``FAKE_CLIENT_CODES`` 的第 n 项行事：非负整数 =
  写本席位 ``progress.json`` 后以该码退出（75 时先打印 ``SERVER_START``／``SERVER_LEFT`` 两行并写服务元数据，模拟看门狗
  只杀客户端、服务端留下）；``-1`` = 写一次进度后一直睡、不再更新进度（收 TERM 即退，模拟卡死）。启动事件记完整 argv。
- ``runner``（代替 ``pp_official_runner.py``／``official_hard_runner.py``）：``--check-imports`` 打印
  ``OFFICIAL_IMPORTS=PASS``；否则对 ``--only`` 的每个 key 写 ``<out>/<key>.a<attempt>/``（``frames/{front,wrist}.rgb24``
  各 3 帧 16×16、``frames/frames.json``、``trace.jsonl``）与 ``<out>/results.jsonl`` 一行。``FAKE_RUNNER_INFRA_ONCE=1``
  时第 1 次尝试的最后一个 key 记基础设施错误（目录照写），``FAKE_RUNNER_EXIT`` 为退出码（缺省 0）。
每次启动、收信号都往 ``FAKE_LOG`` 追加一行 JSON（含 pid），测试据此核对。
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROLE, ARGS = sys.argv[1], sys.argv[2:]
LOG = os.environ["FAKE_LOG"]
SIDE = 16


def emit(**kw):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": ROLE, "pid": os.getpid(), "t": time.time(), **kw}, ensure_ascii=False) + "\n")


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


def on_term(*_):
    emit(event="term")
    sys.exit(0)


def frame(v: int) -> bytes:
    return bytes([v % 256, (v * 7) % 256, (v * 13) % 256]) * (SIDE * SIDE)


def server() -> None:
    mode = os.environ.get("FAKE_SERVER_MODE", "ok")
    port = int(arg("--port"))
    emit(event="start", port=port, mode=mode, argv=ARGS)
    if mode == "die_before_ready":
        sys.exit(1)
    signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore_term" else on_term)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(64)
    s.settimeout(0.1)
    t_end = time.time() + float(os.environ.get("FAKE_SERVER_LIFE", "0")) if mode == "die_after" else None
    while True:
        if t_end is not None and time.time() > t_end:
            emit(event="died")
            os._exit(1)
        try:
            c, _ = s.accept()
        except (socket.timeout, OSError):
            continue
        try:
            c.settimeout(0.2)
            try:
                c.recv(1024)
            except OSError:
                pass
            c.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\n\r\n")
        except OSError:
            pass
        finally:
            c.close()


def write_mkv(path: Path, values: list[int]) -> None:
    raw = b"".join(frame(v) for v in values)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{SIDE}x{SIDE}", "-r", "30", "-i", "-", "-c:v", "ffv1", str(path)], input=raw, check=True)


def write_recording(d: Path, key: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, stream in enumerate(("front", "wrist")):
        write_mkv(d / f"{stream}.mkv", [10 + i, 50 + i, 90 + i])
        with open(d / f"frames-{stream}.jsonl", "w", encoding="utf-8") as fh:
            for idx, enc in enumerate((0, 1, 1, 2)):
                fh.write(json.dumps({"idx": idx, "sha256": f"{stream}-{enc}", "tag": "reset" if idx < 2 else f"step{idx - 1}",
                                     "enc": enc}) + "\n")
    (d / "summary.json").write_bytes(b'{"RECORDER_VERIFY": "PASS"}')


def client() -> None:
    state = Path(os.environ["FAKE_STATE"])
    n = int(state.read_text()) if state.exists() else 0
    state.write_text(str(n + 1))
    codes = [int(x) for x in os.environ.get("FAKE_CLIENT_CODES", "0").split(",")]
    code = codes[min(n, len(codes) - 1)]
    out = Path(arg("--out"))
    out.mkdir(parents=True, exist_ok=True)
    emit(event="start", n=n, code=code, first_extra_s=arg("--first-extra-s"), wall_s=arg("--episode-wall-s"),
         port=arg("--port"), policy=arg("--policy"), dataset=arg("--dataset"), max_steps=arg("--max-steps"),
         strict_cap="--strict-cap" in ARGS, groundsg_variant=arg("--groundsg-variant"),
         adapter=arg("--qwenvl-groundsg-adapter"), trace_root=arg("--trace-root"), v8="--v8" in ARGS,
         ledger=arg("--ledger"), rec_root=arg("--rec-root"), out=str(out), argv=ARGS)
    signal.signal(signal.SIGTERM, on_term)
    (out / "progress.json").write_text(json.dumps({"t": time.time()}), encoding="utf-8")
    if code == -1:
        while True:
            time.sleep(0.1)
    if code == 0:
        rec_root = Path(arg("--rec-root") or out / "rec")
        trace_root = arg("--trace-root")
        for row in json.loads(Path(arg("--identities")).read_text(encoding="utf-8")):
            key = row["key"]
            d = rec_root / f"{key}.a1"
            write_recording(d, key)
            if trace_root:
                td = Path(trace_root) / f"{key}.a1"
                td.mkdir(parents=True, exist_ok=True)
                (td / "trace.jsonl").write_text(json.dumps({"kind": "header", "identity": {"key": key}}) + "\n",
                                                encoding="utf-8")
            with open(out / "results.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"v8": True, "key": key, "task": row["task"], "tier": row.get("tier"),
                                     "seed": row["seed"], "source_episode": row.get("source_episode"),
                                     "dataset": arg("--dataset"), "policy": arg("--policy"),
                                     "policy_variant": arg("--groundsg-variant"), "attempt": 1, "attempt_no": 1,
                                     "status": "success", "infra": False, "demo_frames": 1, "exec_steps": 3,
                                     "rec_dir": str(d)}) + "\n")
    sys.exit(code)


def seat() -> None:
    state = Path(os.environ["FAKE_STATE"])
    n = int(state.read_text()) if state.exists() else 0
    state.write_text(str(n + 1))
    codes = [int(x) for x in os.environ.get("FAKE_CLIENT_CODES", "0").split(",")]
    code = codes[min(n, len(codes) - 1)]
    out = Path(arg("--out"))
    policy = arg("--policy")
    label = f"groundsg-{arg('--groundsg-variant')}" if policy == "groundsg" else policy
    sd = out / "seats" / label / f"seed{arg('--policy-seed')}" / (arg("--seat") or "auto")
    sd.mkdir(parents=True, exist_ok=True)
    emit(event="start", n=n, code=code, policy=policy, argv=ARGS)
    signal.signal(signal.SIGTERM, on_term)
    (sd / "progress.json").write_text(json.dumps({"phase": "context_load", "t": time.time()}), encoding="utf-8")
    print(f"CLIENT_READY policy={policy} seat={sd.name}", flush=True)
    if code == -1:
        while True:
            time.sleep(0.1)
    if code == 75:
        meta = sd / "server-metadata-18999.json"
        meta.write_text(json.dumps({"pid": 999999, "argv": ["fake-server"], "port": 18999}), encoding="utf-8")
        print(f"SERVER_START name=fake port=18999 pid=999999 log={sd / 'fake-18999.log'}", flush=True)
        print(f"INFRA_TIMEOUT phase=episode_wall key=K limit_s=1", flush=True)
        print(f"SERVER_LEFT pid=999999 port=18999 metadata={meta} stop=\"scripts/evaluate.py --stop-server {meta}\"",
              flush=True)
    else:
        print(f"全部完成 policy={policy} seat={sd.name} episodes=0 rc={code}", flush=True)
    sys.exit(code)


def runner() -> None:
    if "--check-imports" in ARGS:
        print("OFFICIAL_IMPORTS=PASS robomme=fake robomme_ood_imported=0", flush=True)
        sys.exit(0)
    out = Path(arg("--out"))
    out.mkdir(parents=True, exist_ok=True)
    attempt = int(arg("--attempt", "1"))
    only = [k for k in (arg("--only") or "").split(",") if k]
    emit(event="start", attempt=attempt, only=only, port=arg("--port"), max_steps=arg("--max-steps"),
         variant=arg("--variant"), adapter=arg("--qwenvl-groundsg-adapter"), shard=arg("--shard"), argv=ARGS)
    signal.signal(signal.SIGTERM, on_term)
    rows = {r["key"]: r for r in json.loads(Path(arg("--shard")).read_text(encoding="utf-8"))}
    infra_key = only[-1] if os.environ.get("FAKE_RUNNER_INFRA_ONCE") == "1" and attempt == 1 and only else None
    for key in only:
        ep = out / f"{key}.a{attempt}"
        (ep / "frames").mkdir(parents=True, exist_ok=False)
        for i, s in enumerate(("front", "wrist")):
            (ep / "frames" / f"{s}.rgb24").write_bytes(b"".join(frame(20 * j + i) for j in range(3)))
        (ep / "frames" / "frames.json").write_text(json.dumps(
            {"pix_fmt": "rgb24", "streams": {s: {"width": SIDE, "height": SIDE, "count": 3} for s in ("front", "wrist")},
             "demo_frames": 1, "init_frames": 1, "exec_steps": 1, "missing_steps": []}), encoding="utf-8")
        (ep / "trace.jsonl").write_text(json.dumps({"kind": "header", "identity": {"key": key}}) + "\n", encoding="utf-8")
        infra = key == infra_key
        row = dict(rows[key])
        row.update(side="orig", policy="groundsg" if arg("--variant") else "pp", policy_variant=arg("--variant"),
                   dataset="hard-verify", attempt=attempt, status="error" if infra else "success", infra=infra,
                   exec_steps=1, demo_frames=1, ep_dir=str(ep))
        with open(out / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    sys.exit(int(os.environ.get("FAKE_RUNNER_EXIT", "0")))


if __name__ == "__main__":
    {"server": server, "client": client, "seat": seat, "runner": runner}[ROLE]()
