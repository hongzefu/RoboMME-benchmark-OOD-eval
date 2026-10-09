"""C13-15 ``framesamp_modul_client`` 的真实 websocket 客户端（``make_recording_client``）、透明中继（``relay`` 子命令）与
传输核对（``transport_check`` / ``transport-check`` 子命令），以及计时汇总与收发摘要。

全部在 127.0.0.1 回环上跑：假策略 server 用 ``openpi_client.msgpack_numpy`` 收发（与真实 FrameSamp+Modulation server 同编码），
中继就是生产的 ``cmd_relay``。期望（帧数、消息数、执行动作数、各 kind 的计数）都由本文件的假环境与假 server 手算，
不调用被测函数生成。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import socket
import threading
import time

import numpy as np
import pytest

import eval_fakes as F

pytest.importorskip("openpi_client", reason="未验证：openpi_client 未安装，无法起真实 FrameSamp+Modulation websocket 客户端")
pytest.importorskip("websockets", reason="未验证：websockets 未安装")

ACTION_ROWS = 50  # 假 server 每次推理回的动作行数（多于执行段 16）


# ---------------------------------------------------------------- 回环上的假 server 与中继


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_port(port: int, t: float = 10.0) -> None:
    end = time.time() + t
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError(port)


class _FakeServer:
    """FrameSamp+Modulation 协议假 server：先发 metadata，再按 reset／add_buffer／infer 回包。``mode="error_text"`` 时 infer 回字符串。"""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.port = _free_port()
        self.received: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import websockets.asyncio.server as wss
        from openpi_client import msgpack_numpy

        async def handler(ws):
            packer = msgpack_numpy.Packer()
            await ws.send(packer.pack({"meta": 1}))
            k = 0
            async for msg in ws:
                obs = msgpack_numpy.unpackb(msg)
                if obs.get("reset"):
                    self.received.append("reset")
                    await ws.send(packer.pack({"reset_finished": True, "reset_time_ms": 4.0}))
                elif obs.get("add_buffer"):
                    self.received.append("add_buffer")
                    await ws.send(packer.pack({"add_buffer_finished": True, "add_buffer_time_ms": 2.0}))
                else:
                    self.received.append("infer")
                    if self.mode == "error_text":
                        await ws.send("Traceback: 假 server 推理崩溃")
                        continue
                    k += 1
                    a = np.random.default_rng(k).normal(size=(ACTION_ROWS, 8)).astype(np.float32)
                    await ws.send(packer.pack({"actions": a, "infer_time_ms": 8.0}))

        async def main():
            async with wss.serve(handler, "127.0.0.1", self.port, compression=None, max_size=None):
                while not self._stop.is_set():
                    await asyncio.sleep(0.02)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()

    def __enter__(self):
        self._thread.start()
        _wait_port(self.port)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(5)


class _Relay:
    """在线程里跑生产的 ``cmd_relay``；``asyncio.run`` 临时换成可从外部取消的版本，用完即停。"""

    def __init__(self, mc, monkeypatch, upstream: int, log_path):
        self.mc = mc
        self.port = _free_port()
        self.args = argparse.Namespace(listen=self.port, upstream=upstream, log=str(log_path))
        self.loop = None
        self.task = None
        self.ret = None
        self.ready = threading.Event()
        orig_new_loop = asyncio.new_event_loop

        def run(coro):
            loop = orig_new_loop()
            self.loop = loop
            self.task = loop.create_task(coro)
            self.ready.set()
            try:
                return loop.run_until_complete(self.task)
            finally:
                loop.close()

        monkeypatch.setattr(asyncio, "run", run)
        self._thread = threading.Thread(target=self._target, daemon=True)

    def _target(self):
        try:
            self.ret = self.mc.cmd_relay(self.args)
        except asyncio.CancelledError:
            self.ret = "cancelled"

    def __enter__(self):
        self._thread.start()
        assert self.ready.wait(5)
        _wait_port(self.port)
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.task.cancel)
        self._thread.join(5)
        assert not self._thread.is_alive()


class _Builder:
    def __init__(self, plan):
        self.plan = plan
        self.env = None

    def make_env_for_episode(self, ep, max_steps=None):
        self.env = F.FakeEnv("T", ep, self.plan)
        return self.env


def _episode(tmp_path, monkeypatch, plan, *, mode="ok", via_relay=True):
    """一局：假 server（＋生产中继）＋真实 RecordingClient ＋ EnvSession(假 builder)。返回 (结果, 录制器, 中继日志, server)。"""
    mc, ec = F.framesamp_modul_client(), F.env_session()
    log = tmp_path / "relay.jsonl"
    rec = F.FakeRecorder(tmp_path / "rec", {})
    b = _Builder(plan)
    sess = ec.EnvSession("T", 2, recorder=rec, builder=b)
    with _FakeServer(mode) as srv:
        if via_relay:
            with _Relay(mc, monkeypatch, srv.port, log) as relay:
                res = mc.run_episode(sess, {"task": "T"}, {"host": "127.0.0.1", "port": relay.port}, rec)
                # 中继在转发前写日志；等最后一条 s2c 落盘
                end = time.time() + 5
                while time.time() < end and not _relay_done(log, srv):
                    time.sleep(0.02)
        else:
            res = mc.run_episode(sess, {"task": "T"}, {"host": "127.0.0.1", "port": srv.port}, rec)
    relay_rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return res, rec, relay_rows, srv, b


def _relay_done(log, srv) -> bool:
    if not log.exists():
        return False
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    # c2s = 服务器收到的消息数；s2c = 1 条 metadata + 每条请求一条回复
    return sum(r["dir"] == "c2s" for r in rows) == len(srv.received) and \
        sum(r["dir"] == "s2c" for r in rows) == len(srv.received) + 1


# ---------------------------------------------------------------- 正例：真实客户端 + 中继 + 传输核对


def test_real_client_through_relay_transport_check_passes(tmp_path, monkeypatch, capsys):
    """执行段 h（取 framesamp_modul_client.OBS_HORIZON）、success_at=2h+8：决策 3 次（1～h、h+1～2h、2h+1～2h+8），
    手算帧数 = reset 帧 + h + h（add_buffer）+ 3×2（infer 两路图）。"""
    mc = F.framesamp_modul_client()
    h = mc.OBS_HORIZON
    steps = 2 * h + 8
    frames = F.N_RESET_FRAMES + 2 * h + 3 * 2
    res, rec, relay, srv, b = _episode(tmp_path, monkeypatch, F.Plan(success_at=steps))
    assert ACTION_ROWS > h  # server 回的行多于执行段，「只执行前 h 行」才可观测
    assert res["status"] == "success" and res["steps"] == steps and res["decisions"] == 3 and res["error"] is None
    assert srv.received == ["reset"] + ["add_buffer", "infer"] * 3  # 策略 reset 每局只一次
    # 执行动作 = server 回的前 h 行（server 用 rng(k) 生成，独立重算）
    want = np.concatenate([np.random.default_rng(k).normal(size=(ACTION_ROWS, 8)).astype(np.float32)[:h]
                           for k in (1, 2, 3)])[:steps]
    assert np.array_equal(np.stack(b.env.actions), want)
    # 计时汇总：reset 1 条、add_buffer 3 条、infer 3 条；server 侧耗时取回包里的毫秒字段
    t = res["timing"]
    assert t["reset"]["n"] == 1 and t["add_buffer"]["n"] == 3 and t["infer"]["n"] == 3
    assert t["infer"]["server_first_s"] == pytest.approx(0.008) and t["reset"]["server_mean_s"] == pytest.approx(0.004)
    assert t["add_buffer"]["server_mean_s"] == pytest.approx(0.002)
    assert "per_msg" not in t and t["connect_s"] >= 0 and t["episode_s"] > 0
    # 录制：每次决策一份 model_action 数组与事件；中继每方向一条日志
    assert rec.arrays["model_action"] == 3
    mas = [e for e in rec.events if e["kind"] == "model_action"]
    assert [e["decision"] for e in mas] == [0, 1, 2] and [e["exec_n"] for e in mas] == [h, h, h]
    assert all(e["shape"] == [ACTION_ROWS, 8] for e in mas)
    assert [r["seq"] for r in relay if r["dir"] == "c2s"] == list(range(7))
    assert [r["seq"] for r in relay if r["dir"] == "s2c"] == list(range(-1, 7))  # -1 是 metadata
    assert [r["payload"]["kind"] for r in relay if r["dir"] == "c2s"] == ["reset"] + ["add_buffer", "infer"] * 3
    assert [r["payload"]["kind"] for r in relay if r["dir"] == "s2c"][1:4] == \
        ["reset_finished", "add_buffer_finished", "actions"]
    assert relay[0]["payload"] == {"kind": "metadata"}  # 首条 s2c 是 metadata

    out = mc.transport_check(rec.events, relay)
    assert out["mismatch"] == 0 and out["frames"] == frames and out["exec_actions"] == steps, out
    assert out["messages"] == 7 + 8  # 7 次发送 + 7 次回复 + 1 次 metadata 接收

    # 子命令：落 events／relay 日志，--conn 0 判 PASS
    ev_path = tmp_path / "events.jsonl"
    ev_path.write_text("\n".join(json.dumps(e) for e in rec.events) + "\n", encoding="utf-8")
    log = tmp_path / "relay.jsonl"
    capsys.readouterr()
    assert mc.main(["transport-check", "--events", str(ev_path), "--relay-log", str(log), "--conn", "0"]) == 0
    assert capsys.readouterr().out.startswith(f"TRANSPORT=PASS frames={frames} mismatch=0 messages=15 exec_actions={steps}")
    # 负例：--conn 1 过滤后中继为空 → 条数不符 FAIL、退出码 1
    assert mc.main(["transport-check", "--events", str(ev_path), "--relay-log", str(log), "--conn", "1"]) == 1
    out_text = capsys.readouterr().out
    assert out_text.startswith("TRANSPORT=FAIL ") and "note: count client send/recv=7/8 relay=0/0" in out_text


def _one_run(tmp_path, monkeypatch):
    return _episode(tmp_path, monkeypatch, F.Plan(success_at=20))


@pytest.mark.parametrize("tamper", ["c2s_sha", "s2c_sha", "drop_s2c", "exec_action", "prompt", "add_buffer_frame",
                                    "infer_image", "extra_exec"])
def test_transport_check_detects_each_tamper(tmp_path, monkeypatch, tamper):
    """每种篡改都必须被测出（mismatch≥1）；原样输入 mismatch=0（同一局作对照）。"""
    mc = F.framesamp_modul_client()
    res, rec, relay, _, _ = _one_run(tmp_path, monkeypatch)
    assert res["status"] == "success"
    events = [json.loads(json.dumps(e)) for e in rec.events]
    assert mc.transport_check(events, relay)["mismatch"] == 0
    bad_r = [dict(r) for r in relay]
    if tamper == "c2s_sha":
        i = next(i for i, r in enumerate(bad_r) if r["dir"] == "c2s" and r["seq"] == 1)
        bad_r[i]["sha"] = "0" * 64
        want_note = "c2s seq=1"
    elif tamper == "s2c_sha":
        i = next(i for i, r in enumerate(bad_r) if r["dir"] == "s2c" and r["seq"] == 2)
        bad_r[i]["sha"] = "0" * 64
        want_note = "s2c seq=2"
    elif tamper == "drop_s2c":
        bad_r = [r for r in bad_r if not (r["dir"] == "s2c" and r["seq"] == 0)]
        want_note = "count client send/recv="
    elif tamper == "exec_action":
        e = next(e for e in events if e["kind"] == "env_step_action")
        e["sha"] = "f" * 64
        want_note = "exec_action != model rows: 1"
    elif tamper == "prompt":
        e = next(e for e in events if e["kind"] == "ws_send" and e["payload"]["kind"] == "infer")
        e["payload"]["prompt"] = "别的指令"
        want_note = f"infer seq={e['seq']} prompt"
    elif tamper == "add_buffer_frame":
        e = next(e for e in events if e["kind"] == "env_reset")
        e["front"][0] = "0" * 64
        want_note = "add_buffer seq=1 frames/state != env"
    elif tamper == "infer_image":
        e = next(e for e in events if e["kind"] == "ws_send" and e["payload"]["kind"] == "infer")
        e["payload"]["image"] = "0" * 64
        want_note = f"infer seq={e['seq']} obs != env"
    else:  # 执行动作比模型给的行还多一条
        e = next(e for e in events if e["kind"] == "env_step_action")
        events.append(dict(e))
        want_note = "exec_action != model rows: "
    out = mc.transport_check(events, bad_r)
    assert out["mismatch"] >= 1 and any(n.startswith(want_note) for n in out["notes"]), out


def test_server_error_text_becomes_error_not_infra(tmp_path, monkeypatch):
    """server 回字符串（推理异常）→ RecordingClient 抛 RuntimeError，整局 error、非基础设施，并记一条 error_text 事件。"""
    res, rec, _, srv, b = _episode(tmp_path, monkeypatch, F.Plan(success_at=5), mode="error_text", via_relay=False)
    assert res["status"] == "error" and res["infra"] is False and res["steps"] == 0
    assert res["error"].startswith("RuntimeError: Error in inference server:\nTraceback: 假 server 推理崩溃")
    assert srv.received == ["reset", "add_buffer", "infer"] and b.env.n == 0
    assert [e["msg"] for e in rec.events if e["kind"] == "ws_recv" and "msg" in e] == ["metadata", "error_text"]
    # infer 没有成功往返，不进计时；reset、add_buffer 各一次完整往返
    assert "infer" not in res["timing"] and res["timing"]["reset"]["n"] == 1 and res["timing"]["add_buffer"]["n"] == 1


def test_relay_logs_decode_error_for_non_msgpack(tmp_path, monkeypatch):
    """中继对无法解包的二进制消息记 decode_error，仍原样转发（server 端收到同样字节）。"""
    import websockets.sync.client as wsc

    mc = F.framesamp_modul_client()
    log = tmp_path / "relay.jsonl"

    got: list[bytes] = []
    stop = threading.Event()
    port = _free_port()

    def echo_server():
        import websockets.asyncio.server as wss

        async def handler(ws):
            async for msg in ws:
                got.append(msg)
                await ws.send(msg)

        async def main():
            async with wss.serve(handler, "127.0.0.1", port):
                while not stop.is_set():
                    await asyncio.sleep(0.02)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(main())
        loop.close()

    th = threading.Thread(target=echo_server, daemon=True)
    th.start()
    _wait_port(port)
    try:
        with _Relay(mc, monkeypatch, port, log) as relay:
            with wsc.connect(f"ws://127.0.0.1:{relay.port}") as c:
                c.send(b"\xc1not-msgpack")
                assert c.recv() == b"\xc1not-msgpack"
                c.send("纯文本")
                assert c.recv() == "纯文本"
            end = time.time() + 5
            while time.time() < end and len(log.read_text().splitlines()) < 4:
                time.sleep(0.02)
    finally:
        stop.set()
        th.join(5)
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert got == [b"\xc1not-msgpack", "纯文本"]
    bin_rows = [r for r in rows if r["len"] == len(b"\xc1not-msgpack")]
    assert {r["dir"] for r in bin_rows} == {"c2s", "s2c"} and all("decode_error" in r for r in bin_rows)
    txt = [r for r in rows if r["len"] == len("纯文本")]
    assert txt and all("decode_error" not in r and "payload" not in r for r in txt)  # 文本不解包
    assert {r["conn"] for r in rows} == {0}


# ---------------------------------------------------------------- 纯函数：摘要与计时


def test_payload_and_response_digest_kinds():
    mc = F.framesamp_modul_client()
    imgs = np.stack([F.frame(1), F.frame(2)])[:, None]
    states = np.array([[0.5] * 8, [0.25] * 8], dtype=np.float32)
    d = mc.payload_digest({"images": imgs, "state": states, "add_buffer": True, "exec_start_idx": 1})
    assert d == {"kind": "add_buffer", "frames": [F.sha_bytes(F.frame(1)), F.sha_bytes(F.frame(2))],
                 "states": [F.sha_bytes(states[0]), F.sha_bytes(states[1])], "exec_start_idx": 1,
                 "shape": [2, 1, F.HW, F.HW, 3]}
    assert mc.payload_digest({"reset": True}) == {"kind": "reset"}
    inf = mc.payload_digest({"observation/image": F.frame(3), "observation/wrist_image": F.frame(4),
                             "observation/state": states[0], "prompt": "p"})
    assert inf == {"kind": "infer", "image": F.sha_bytes(F.frame(3)), "wrist": F.sha_bytes(F.frame(4)),
                   "state": F.sha_bytes(states[0]), "prompt": "p"}
    # 回包摘要：非 dict → other；actions 记 dtype/shape；无 infer_time_ms 记 nan
    assert mc.response_digest([1, 2]) == {"kind": "other"}
    a = np.zeros((3, 8), dtype=np.float32)
    r = mc.response_digest({"actions": a})
    assert r["kind"] == "actions" and r["dtype"] == "<f4" and r["shape"] == [3, 8] and math.isnan(r["infer_time_ms"])
    assert r["actions"] == F.sha_bytes(a)
    assert mc.response_digest({"reset_finished": True}) == {"kind": "reset_finished", "server_ms": 0.0}
    assert mc.response_digest({"add_buffer_finished": True, "add_buffer_time_ms": 3}) == \
        {"kind": "add_buffer_finished", "server_ms": 3.0}
    assert mc.response_digest({"reset_finished": False, "x": 1}) == {"kind": "metadata"}


def test_sha_accepts_bytes_str_and_arrays():
    import hashlib

    mc = F.framesamp_modul_client()
    assert mc.sha(b"ab") == hashlib.sha256(b"ab").hexdigest()
    assert mc.sha("中") == hashlib.sha256("中".encode("utf-8")).hexdigest()
    # 非 C 连续数组按连续拷贝取字节
    a = np.arange(12, dtype=np.int16).reshape(3, 4)[:, ::2]
    assert mc.sha(a) == hashlib.sha256(np.array([[0, 2], [4, 6], [8, 10]], dtype=np.int16).tobytes()).hexdigest()


def test_summarize_timing_hand_computed():
    """6 条 infer：首次 1.0 s、第 2/3 次 0.5/0.4 s、稳态（第 4 条起）0.1/0.2/0.3 → 均值 0.2；只有 1 条的 kind 不给第 2/3 次。"""
    mc = F.framesamp_modul_client()
    srv_ms = [1000.0, 500.0, 400.0, 100.0, 200.0, 300.0]
    per = [{"seq": i, "kind": "infer", "pack_s": 0.01, "rtt_s": s / 1000 + 0.05, "unpack_s": 0.02,
            "server_ms": s, "bytes": 100} for i, s in enumerate(srv_ms)]
    per.append({"seq": 9, "kind": "reset", "pack_s": 0.0, "rtt_s": 0.3, "unpack_s": 0.0, "server_ms": 100.0,
                "bytes": 10})
    out = mc.summarize_timing({"per_msg": per, "connect_s": 1.5})
    assert out["connect_s"] == 1.5 and "add_buffer" not in out
    inf = out["infer"]
    assert inf["n"] == 6 and inf["server_first_s"] == 1.0 and inf["server_2_s"] == 0.5 and inf["server_3_s"] == 0.4
    assert inf["server_steady_mean_s"] == pytest.approx(0.2)
    assert inf["network_mean_s"] == pytest.approx(0.05) and inf["bytes_mean"] == 100.0
    assert inf["rtt_first_s"] == pytest.approx(1.05)
    rs = out["reset"]
    assert rs["n"] == 1 and rs["network_mean_s"] == pytest.approx(0.2) and "server_2_s" not in rs
    # 只有 1 条 infer：第 2/3 次为 None，稳态退化为全体
    one = mc.summarize_timing({"per_msg": per[:1]})
    assert one["infer"]["server_2_s"] is None and one["infer"]["server_3_s"] is None
    assert one["infer"]["server_steady_mean_s"] == pytest.approx(1.0)
