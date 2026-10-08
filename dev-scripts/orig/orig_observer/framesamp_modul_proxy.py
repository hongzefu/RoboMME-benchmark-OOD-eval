"""FrameSamp+Modulation 原侧客户端与 policy server 之间的透明 websocket 代理（v7.5eval 方案 §2.5；S7 只改输出与日志封口）。

- 监听 127.0.0.1:<listen>，每个客户端连接新开一条到 127.0.0.1:<upstream> 的上游连接（官方客户端每局新连接）。
- 双向逐条转发，保持帧类型（二进制／文本）与顺序，包括 server 连上后先发的 metadata 帧；不开压缩、
  ``max_size=None``。
- 保活（websocket ping/pong 是逐跳的控制帧，库无法转发）：两跳各自照抄原来那一端的设置——
  面向客户端一侧照抄官方 server（websockets 15 ``serve`` 默认 ``ping_interval=20, ping_timeout=20``，
  ``close_timeout=10``），面向 server 一侧照抄官方客户端 ``WebsocketClientPolicy._wait_for_server``
  （``ping_interval=20``（默认）、``ping_timeout=600``、``open_timeout=60``、``close_timeout=60``）。
  这样 server 推理阻塞时的保活超时仍在「代理↔server」这一跳按历史口径触发，触发后代理把同样的
  关闭码与原因转给客户端；客户端看到的是 ConnectionClosedError（历史上是客户端自身的保活超时，
  同为该异常类型、同样记 error），此差别写进留档。
- 上游连不上时在握手阶段拒绝客户端（HTTP 503），客户端握手即失败，不会先连上再收到 1011。
  （历史上 server 不在时客户端得到 ConnectionRefusedError 并无限重试；这里得到 InvalidStatus，同样无法建立会话。）
- 转发优先：先 ``send`` 给对端，再把同一个消息对象交给后台线程记账，记账永不改动数据、失败只记日志。
- 记账：``<log_dir>/proxy-<pid>.jsonl`` 逐条 {conn, dir(c2s|s2c), idx, type, len, sha256, t}；
  msgpack 解码后的数值数组（动作、状态等）写进 ``<log_dir>/conn-<pid>-<NNNN>/``（EpisodeRecorder 目录，
  只有 arrays 与 events），图像只记逐帧 sha256（图像本身由客户端钩子录）。

- 日志封口（S7，审计第 13 条）：收到 SIGTERM／SIGINT 后停止监听、等记账线程把积压全部写完并关闭日志，**之后**才写
  ``<log_dir>/proxy-<pid>.done``（``{"pid","t","sealed":true,"accountant_alive":false}``）；记账线程 600 s 内没收完则
  不写 ``.done``，启动器据此判封口超时。透明性对账只在看到 ``.done`` 之后才跑。

用法：python framesamp_modul_proxy.py --listen 19001 --upstream 19000 --log-dir <REC_ROOT>/proxy
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import http
import itertools
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _obs_common as C  # noqa: E402

import websockets  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

try:
    from openpi_client.msgpack_numpy import unpackb as _unpackb
except Exception:  # 与 openpi_client.msgpack_numpy.unpack_array 逐行等价的兜底
    import msgpack

    def _unpack_array(obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    def _unpackb(b):
        return msgpack.unpackb(b, object_hook=_unpack_array)

_UNSENDABLE_CLOSE = {1005, 1006, 1015}


def _flatten(obj, prefix=""):
    """把解码后的嵌套 dict 摊平成 (键路径, 值)。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = k.decode(errors="replace") if isinstance(k, bytes) else str(k)
            yield from _flatten(v, f"{prefix}{ks}/" if isinstance(v, dict) else f"{prefix}{ks}")
    else:
        yield prefix.rstrip("/"), obj


ACCT_MAX_BYTES = int(os.environ.get("V75_PROXY_ACCT_MAX_BYTES", str(2 * 2**30)))  # 记账积压上限，超过即让转发协程等待


class Accountant:
    """后台记账线程：逐条写日志、解码 msgpack、按连接写 EpisodeRecorder 目录。积压有上限（字节）。"""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log = C.JsonlLog(log_dir / f"proxy-{os.getpid()}.jsonl")
        self.R = C.load_recorder()
        self._q: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._recs: dict[int, object] = {}
        self._stop = False
        self.queued_bytes = 0
        self._t = threading.Thread(target=self._loop, name="v75-proxy-accountant", daemon=True)
        self._t.start()

    def submit(self, item: tuple) -> None:
        with self._cv:
            self._q.append(item)
            if item[0] == "msg":
                self.queued_bytes += len(item[-1])
            self._cv.notify()

    async def submit_async(self, item: tuple) -> None:
        """转发协程用：积压超过上限时先让出事件循环等待记账线程追上（只影响时序，不改数据）。"""
        while self.queued_bytes > ACCT_MAX_BYTES and self._t.is_alive():
            await asyncio.sleep(0.01)
        self.submit(item)

    def stop(self) -> bool:
        """停止并等记账线程收完；返回记账线程是否已退出（日志已关闭）。"""
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(timeout=600)
        return not self._t.is_alive()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if not self._q and self._stop:
                    break
                item = self._q.popleft()
                if item[0] == "msg":
                    self.queued_bytes -= len(item[-1])
            try:
                self._handle(item)
            except Exception as e:  # 记账失败只记一行，不影响转发
                self.log.write({"kind": "account_error", "error": f"{type(e).__name__}: {e}", "item": repr(item[:3])})
        for conn in list(self._recs):
            self._close_conn(conn, {"reason": "proxy_stop"})
        self.log.close()

    def _rec(self, conn: int):
        r = self._recs.get(conn)
        if r is None:
            r = self.R.EpisodeRecorder(self.log_dir / f"conn-{os.getpid()}-{conn:04d}",
                                       {"kind": "proxy_conn", "conn": conn, "pid": os.getpid(), "never_degrade": True})
            r.set_phase("run")
            self._recs[conn] = r
        return r

    def _close_conn(self, conn: int, info: dict) -> None:
        r = self._recs.pop(conn, None)
        if r is not None:
            res = r.close(info)
            self.log.write({"kind": "conn_recorder", "conn": conn, "verify": res["RECORDER_VERIFY"]})

    def _handle(self, item: tuple) -> None:
        kind = item[0]
        if kind == "open":
            _, conn, t, peer = item
            self.log.write({"kind": "open", "conn": conn, "t": t, "peer": peer})
            self._rec(conn)
            return
        if kind == "close":
            _, conn, t, info = item
            self.log.write({"kind": "close", "conn": conn, "t": t, **info})
            self._close_conn(conn, info)
            return
        _, conn, direction, idx, t, msg = item
        ftype, n, sha = C.payload_sha(msg)
        self.log.write({"kind": "msg", "conn": conn, "dir": direction, "idx": idx, "type": ftype, "len": n,
                        "sha256": sha, "t": t})
        rec = self._rec(conn)
        ev = {"kind": "msg", "dir": direction, "idx": idx, "type": ftype, "len": n, "sha256": sha}
        if ftype == "binary":
            try:
                obj = _unpackb(msg)
            except Exception as e:
                ev["decode_error"] = f"{type(e).__name__}: {e}"
                rec.add_event(ev)
                return
            fields = {}
            for key, val in _flatten(obj):
                if isinstance(val, np.ndarray):
                    if val.dtype == np.uint8 and val.ndim >= 3:
                        frames = val.reshape((-1,) + val.shape[-3:])
                        fields[key] = {"image_shape": list(val.shape),
                                       "frame_sha256": [self.R.frame_sha256(f) for f in frames]}
                    else:
                        rec.add_array(f"{direction}.{key}", val, step=idx)
                        fields[key] = {"array": list(val.shape), "dtype": val.dtype.str,
                                       "sha256": self.R.array_sha256(val)}
                elif isinstance(val, (bytes, bytearray)):
                    fields[key] = {"bytes": len(val)}
                elif isinstance(val, np.generic):
                    fields[key] = val.item()
                else:
                    fields[key] = val if isinstance(val, (str, int, float, bool, type(None))) else repr(val)
            ev["fields"] = fields
        else:
            ev["text_head"] = msg[:500]
        rec.add_event(ev)


class Proxy:
    def __init__(self, listen: int, upstream: int, acct: Accountant):
        self.listen = listen
        self.upstream = upstream
        self.acct = acct
        self._conn_ids = itertools.count()
        self._pending: dict[int, tuple] = {}  # id(客户端连接) → (conn 序号, 上游连接)

    async def process_request(self, connection, request):
        """握手阶段：/healthz 照官方 server 应答；其余先连上游，连不上即 503 拒绝握手。"""
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        conn = next(self._conn_ids)
        headers = None
        auth = request.headers.get("Authorization")
        if auth:
            headers = {"Authorization": auth}
        try:
            # 照抄官方客户端 WebsocketClientPolicy._wait_for_server 的连接参数（ping_interval 用默认 20）
            up = await connect(f"ws://127.0.0.1:{self.upstream}", compression=None, max_size=None,
                               additional_headers=headers, ping_timeout=600, open_timeout=60, close_timeout=60)
        except Exception as e:
            self.acct.submit(("close", conn, time.time(), {"upstream_error": f"{type(e).__name__}: {e}",
                                                            "rejected": 503}))
            return connection.respond(http.HTTPStatus.SERVICE_UNAVAILABLE, "v75 proxy: upstream unavailable\n")
        self._pending[id(connection)] = (conn, up)
        return None

    async def handler(self, client_ws):
        conn, up = self._pending.pop(id(client_ws))
        self.acct.submit(("open", conn, time.time(), repr(client_ws.remote_address)))
        counters = {"c2s": 0, "s2c": 0}

        async def pump(src, dst, direction):
            async for msg in src:  # msg 为 bytes（二进制帧）或 str（文本帧），原样转发
                await dst.send(msg)
                await self.acct.submit_async(("msg", conn, direction, counters[direction], time.time(), msg))
                counters[direction] += 1

        t_c2s = asyncio.create_task(pump(client_ws, up, "c2s"))
        t_s2c = asyncio.create_task(pump(up, client_ws, "s2c"))
        done, pending = await asyncio.wait({t_c2s, t_s2c}, return_when=asyncio.FIRST_COMPLETED)
        info = {}
        for t in done:
            exc = t.exception()
            if exc is not None and not isinstance(exc, websockets.ConnectionClosed):
                info["pump_error"] = f"{type(exc).__name__}: {exc}"
        # 一端结束（含上游保活超时 1011 "keepalive ping timeout"）：把关闭码与原因原样传给另一端
        src, dst = (client_ws, up) if t_c2s in done else (up, client_ws)
        code = src.close_code if src.close_code is not None else 1000
        reason = src.close_reason or ""
        if code in _UNSENDABLE_CLOSE:
            code, reason = 1011, f"v75 proxy: peer closed abnormally ({src.close_code})"
        try:
            await dst.close(code=code, reason=reason)
        except Exception:
            pass
        for t in pending:
            try:
                await asyncio.wait_for(t, timeout=60)
            except Exception:
                t.cancel()
        info.update(c2s=counters["c2s"], s2c=counters["s2c"], client_close=client_ws.close_code,
                    client_close_reason=client_ws.close_reason, upstream_close=up.close_code,
                    upstream_close_reason=up.close_reason,
                    first_closed="client" if t_c2s in done else "upstream")
        self.acct.submit(("close", conn, time.time(), info))

    async def run(self, stop: asyncio.Event):
        # 面向客户端一侧照抄官方 server：compression=None、max_size=None，保活等其余参数用 websockets 默认
        async with serve(self.handler, "127.0.0.1", self.listen, compression=None, max_size=None,
                         process_request=self.process_request):
            print(f"PROXY_READY listen={self.listen} upstream={self.upstream} pid={os.getpid()}", flush=True)
            await stop.wait()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream", type=int, required=True)
    ap.add_argument("--log-dir", required=True)
    a = ap.parse_args(argv)
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    acct = Accountant(log_dir)

    async def _amain():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, stop.set)
        await Proxy(a.listen, a.upstream, acct).run(stop)

    try:
        asyncio.run(_amain())
    finally:
        sealed = acct.stop()
        if sealed:
            write_done(log_dir)
        print(f"PROXY_STOPPED sealed={'yes' if sealed else 'no'} pid={os.getpid()}", flush=True)
    return 0


def write_done(log_dir: Path) -> Path:
    """日志封口标记 ``proxy-<pid>.done``（先写临时名再原子改名）。"""
    p = Path(log_dir) / f"proxy-{os.getpid()}.done"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(C.dumps({"pid": os.getpid(), "t": time.time(), "sealed": True, "accountant_alive": False}) + "\n",
                   encoding="utf-8")
    tmp.replace(p)
    return p


if __name__ == "__main__":
    sys.exit(main())
