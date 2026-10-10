"""FrameSamp+Modulation 原侧（``robomme_policy_learning`` 原版分支 ``official-xhard0-0929``／``927c56d``）客户端的包装启动器：同进程 runpy
运行原版 ``examples/robomme/eval.py``，外挂只读钩子，写出与 GroundSG 原侧同布局的逐局
``<key>.a<N>/{trace.jsonl,frames/,arrays.npz}``（route ``perceptual-framesamp-modul/orig``）。

用法（cwd = 原版工作树 examples/robomme，REC_ROOT 必设）：
    python framesamp_modul_client_wrap.py eval.py --args.host=127.0.0.1 --args.port=<代理端口> ...（参数原样透传给 eval.py）
    python framesamp_modul_client_wrap.py --orig-preflight   # 只装钩子并核对导入，不跑评估

钩子（计划第二部分一节 S7「钩子铁律」：只复制主机端 numpy、不调随机函数、不做 GPU 运算、不改参数与返回值；
钩子内异常只记 ``OBSERVER_HOOK_ERROR`` 并累加 ``observer_hook_errors``，绝不外抛）：
- ``EnvRunner.get_init_obs``：reset 返回的全部演示帧 + 初始帧（前视、腕部）、8 维状态、task_goal → ``demo`` 行（C2）。
- ``EnvRunner.step``：调用前复制实际交给 env 的动作（原 dtype／shape），调用后记返回的 (img, wrist, state)、
  ``stop``、``status``；原版吞掉环境异常返回 ``(None, None, None)`` 的一步记 ``observed=false`` 与原因（C8）；
  ``terminated``／``truncated`` 原版不向调用方返回，写 ``NOT_OBSERVED``（C9）；FrameSamp+Modulation 无子目标功能，``subgoal`` 全程
  ``None``（C7）。
- ``MMEVLAWebsocketClientPolicy.reset／add_buffer／infer``：标记当前请求名，``ClientConnection.send`` 在标记期间
  发出的原始字节暂存；往返成功后按新侧 ``framesamp_modul_client.TracedClient`` 同口径记 ``request``（名即方法名，载荷为原始
  msgpack 字节，C10「同协议比原始哈希」）、``add_buffer`` 的历史边界行、``infer`` 的完整动作块 ``response``；调用
  抛异常时不记该请求（与新侧相同）。
- ``ClientConnection.send/recv``：另把每条发出／收到消息的帧类型、长度、sha256 写 ``REC_ROOT/client-transport-<pid>.jsonl``
  （``episode`` 字段为局目录名），供 transparency_check.py 与代理日志逐条对账、并核对清单身份都有连接。
- 局边界：``EpisodeEvaluator.eval_each_episode``（在 eval.py 调 ``tyro.cli`` 时从 ``__main__`` 取类挂上）；终态与
  ``evaluate_manifest`` 写 ``episodes.jsonl`` 同口径：返回值属 success／fail／timeout 原样，其余（``unknown``）与
  异常记 ``error``；官方超时（第 ``max_steps+1`` 步先 break 不录像）记 ``omitted_timeout_frames=1``。

钩子都在对应模块首次导入完成后才挂（PostImportHooks），不提前导入任何原版模块，导入顺序与直接运行一致。
"""
from __future__ import annotations

import os
import runpy
import sys
import threading
import time
from pathlib import Path

import numpy as np

_OBS_DIR = str(Path(__file__).resolve().parent)
if _OBS_DIR not in sys.path:
    sys.path.append(_OBS_DIR)
import _obs_common as C  # noqa: E402
import orig_episode as OE  # noqa: E402

ROUTE = "perceptual-framesamp-modul/orig"
ERR = C.HookErrors()
_lock = threading.RLock()
_state: dict = {"ep": None, "ep_name": None, "conn_seq": 0, "log": None, "root": None, "req": None}


def _ep_error(ep, where: str) -> None:
    if ep is not None:
        ep.hook_error(where)
    else:
        ERR.note(where)


def _log():
    if _state["log"] is None:
        _state["log"] = C.JsonlLog(_state["root"] / f"client-transport-{os.getpid()}.jsonl")
    return _state["log"]


def _host_flag(x):
    """主机端布尔：Python／numpy 标量直接取；CPU 张量取 item；GPU 张量不碰（返回 None）。"""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if getattr(x, "is_cuda", False):
        return None
    if isinstance(x, np.ndarray) and x.size == 1:
        return bool(x.reshape(-1)[0])
    if hasattr(x, "item") and getattr(getattr(x, "device", None), "type", "cpu") == "cpu":
        return bool(x.item())
    return None


# ------------------------------------------------------------------ EnvRunner 钩子

def _patch_env_runner(mod) -> None:
    cls = mod.EnvRunner
    orig_init, orig_step = cls.get_init_obs, cls.step

    def get_init_obs(self):
        out = orig_init(self)
        ep = _state["ep"]
        try:
            if ep is not None:
                ep.on_reset([np.asarray(x) for x in out["images"]], [np.asarray(x) for x in out["wrist_images"]],
                            [np.asarray(x) for x in out["states"]], out["task_goal"])
        except Exception:  # noqa: BLE001
            _ep_error(ep, "get_init_obs.post")
        return out

    def step(self, action):
        ep = _state["ep"]
        a = None
        try:
            if ep is not None:
                a = np.array(action, copy=True)
        except Exception:  # noqa: BLE001
            _ep_error(ep, "step.pre")
        out = orig_step(self, action)
        try:
            if ep is not None:
                if a is None:
                    a = np.zeros(0, np.float32)
                (img, wrist, state), stop, status = out
                st = status if isinstance(status, str) else f"<{type(status).__name__}>"
                flag = _host_flag(stop)
                if img is None:
                    ep.on_missing_step(a, reason=f"env_step_exception status={st}", stop=flag, orig_status=st)
                else:
                    ep.on_step(a, np.asarray(img), np.asarray(wrist), np.asarray(state), st, stop=flag)
        except Exception:  # noqa: BLE001
            _ep_error(ep, "step.post")
        return out

    cls.get_init_obs, cls.step = get_init_obs, step
    cls._orig_observer_hooked = True


# ------------------------------------------------------------------ 策略客户端钩子（请求名与动作块）

def record_call(ep, name: str, obj, out, raw: bytes | None) -> None:
    """与新侧 ``framesamp_modul_client.TracedClient._record`` 同口径（C10）：往返成功后才记；``request`` 载荷为实际发出的原始
    msgpack 字节（拿不到时退回 ``canonical_bytes``，并在 ``end`` 记 ``request_encoding``）；``add_buffer`` 另记历史边界
    ``history(<上次边界>, <当前步>, note="add_buffer frames=<n> exec_start_idx=<…>")``；``infer`` 记完整动作块。"""
    if raw is not None:
        payload = raw
        ep.encodings.add("msgpack")
    else:
        obj_ = {"reset": True} if name == "reset" else obj
        payload = OE.tw.canonical_bytes(obj_)
        ep.encodings.add("canonical")
    ep.on_request(name, payload)
    if name == "add_buffer":
        n = len(obj["images"]) if isinstance(obj, dict) and "images" in obj else 0
        ep.on_history(f"add_buffer frames={n} exec_start_idx={obj.get('exec_start_idx') if isinstance(obj, dict) else None}")
    elif name == "infer":
        ep.on_response(np.asarray(out["actions"]))


def _patch_policy_client(mod) -> None:
    cls = getattr(mod, "MMEVLAWebsocketClientPolicy", None)
    if cls is None:
        ERR.note("policy_client.missing_class")
        return

    def wrap(name, orig):
        def method(self, *args, **kwargs):
            prev, prev_raw = _state["req"], _state.get("req_raw")
            _state["req"], _state["req_raw"] = name, None
            try:
                out = orig(self, *args, **kwargs)
                raw = _state["req_raw"]
            finally:
                _state["req"], _state["req_raw"] = prev, prev_raw
            ep = _state["ep"]
            try:
                if ep is not None:
                    record_call(ep, name, args[0] if args else kwargs.get("buffer", kwargs.get("obs")), out, raw)
            except Exception:  # noqa: BLE001
                _ep_error(ep, f"policy.{name}.post")
            return out

        method.__name__ = getattr(orig, "__name__", name)
        method.__doc__ = getattr(orig, "__doc__", None)
        return method

    for name in ("reset", "add_buffer", "infer"):
        if hasattr(cls, name):
            setattr(cls, name, wrap(name, getattr(cls, name)))
    cls._orig_observer_hooked = True


# ------------------------------------------------------------------ websocket 钩子

def _conn_id(ws) -> int:
    cid = getattr(ws, "_orig_obs_conn", None)
    if cid is None:
        with _lock:
            cid = _state["conn_seq"]
            _state["conn_seq"] += 1
        ws._orig_obs_conn = cid
        ws._orig_obs_idx = {"send": 0, "recv": 0}
    return cid


def _note_msg(ws, direction: str, msg) -> None:
    ep = _state["ep"]
    try:
        cid = _conn_id(ws)
        idx = ws._orig_obs_idx[direction]
        ws._orig_obs_idx[direction] += 1
        ftype, n, sha = C.payload_sha(msg)
        _log().write({"pid": os.getpid(), "conn": cid, "dir": direction, "idx": idx, "type": ftype, "len": n,
                      "sha256": sha, "t": time.time(), "episode": _state["ep_name"]})
        if direction == "send" and _state["req"] is not None and _state.get("req_raw") is None:
            _state["req_raw"] = C.payload_bytes(msg)  # 本次请求实际发出的原始字节，往返成功后由 record_call 记
    except Exception:  # noqa: BLE001
        _ep_error(ep, f"ws.{direction}")


def _patch_ws(mod) -> None:
    cls = mod.ClientConnection
    orig_send, orig_recv = cls.send, cls.recv

    def send(self, message, *args, **kwargs):
        r = orig_send(self, message, *args, **kwargs)
        _note_msg(self, "send", message)  # 先发出再记账：sha 计算与 server 计算重叠
        return r

    def recv(self, *args, **kwargs):
        msg = orig_recv(self, *args, **kwargs)
        _note_msg(self, "recv", msg)
        return msg

    cls.send, cls.recv = send, recv
    cls._orig_observer_hooked = True


# ------------------------------------------------------------------ 局边界

def episode_status(ret, exc) -> str:
    """与 ``evaluate_manifest`` 同口径：异常 → error；返回值属 success／fail／timeout 原样，其余 → error。"""
    if exc is not None:
        return "error"
    return ret if ret in C.FINAL_STATUSES else "error"


def _patch_evaluator(main_mod, seeds: dict) -> None:
    cls = main_mod.EpisodeEvaluator
    orig = cls.eval_each_episode

    def eval_each_episode(self, env_runner, *args, **kwargs):
        ep = None
        max_steps = 1300
        try:
            task, src = env_runner.env_id, int(env_runner.episode_id)
            max_steps = int(getattr(getattr(self, "args", None), "max_steps", 1300))
            if (task, src) not in seeds:
                raise KeyError(f"清单里没有 ({task}, {src}) 的 seed，无法定局目录")
            ep = OE.OrigEpisode(_state["root"], route=ROUTE, task=task, source_episode=src, seed=seeds[(task, src)],
                                max_steps=max_steps, errors=ERR)
        except Exception:  # noqa: BLE001
            ERR.note("episode.open")
        with _lock:
            _state.update(ep=ep, ep_name=None if ep is None else ep.name)
        ret, exc = None, None
        try:
            ret = orig(self, env_runner, *args, **kwargs)
            return ret
        except BaseException as e:
            exc = f"{type(e).__name__}: {e}"[:800]
            raise
        finally:
            with _lock:
                _state.update(ep=None, ep_name=None)
            if ep is not None:
                try:
                    status = episode_status(ret, exc)
                    omitted = int(status == "timeout" and ep.steps == max_steps + 1)
                    enc = sorted(ep.encodings)
                    info = ep.close(status, omitted_timeout_frames=omitted,
                                    request_encoding=enc[0] if len(enc) == 1 else ("mixed" if enc else None),
                                    success_flag=ret if isinstance(ret, str) else None, exception=exc,
                                    steps_official=getattr(self, "last_steps", None))
                    print(f"OBSERVER_EPISODE route={ROUTE} episode={info.get('episode')} status={info.get('status')} "
                          f"steps_attempted={info.get('steps_attempted')} steps_observed={info.get('steps_observed')} "
                          f"frames_recorded={info.get('frames_recorded')} hook_errors={info.get('observer_hook_errors')}",
                          flush=True)
                    C.JsonlLog(_state["root"] / "observer-index.jsonl").write(info)
                except Exception:  # noqa: BLE001
                    ERR.note("episode.close", getattr(ep, "name", None))

    cls.eval_each_episode = eval_each_episode
    cls._orig_observer_hooked = True


def _patch_tyro(mod, seeds: dict) -> None:
    orig_cli = mod.cli

    def cli(*args, **kwargs):
        main_mod = sys.modules.get("__main__")
        try:
            if main_mod is not None and hasattr(main_mod, "EpisodeEvaluator"):
                _patch_evaluator(main_mod, seeds)
            else:
                ERR.note("tyro.cli 找不到 __main__.EpisodeEvaluator")
        except Exception:  # noqa: BLE001
            ERR.note("tyro.cli")
        return orig_cli(*args, **kwargs)

    mod.cli = cli


def install(argv: list[str]) -> None:
    _state["root"] = C.rec_root()
    ERR.root = _state["root"]
    seeds = C.manifest_seeds(C.parse_flag(argv, ("--args.episode_manifest", "--args.episode-manifest")))
    C.PostImportHooks({
        "env_runner": _patch_env_runner,
        "websockets.sync.client": _patch_ws,
        "openpi_client.websocket_client_policy": _patch_policy_client,
        "tyro": lambda m: _patch_tyro(m, seeds),
    }).install()


def assert_official_namespace() -> None:
    bad = [m for m in sys.modules if any(m == p or m.startswith(p + ".")
                                        for p in ("robomme_hard", "robomme_ood"))]
    assert not bad, f"原侧不得导入 OOD 包：{bad}"


def main() -> None:
    assert_official_namespace()
    argv = sys.argv[1:]
    if argv[:1] == ["--orig-preflight"]:
        install([])
        sys.path[0] = os.getcwd()
        import robomme  # noqa: F401
        import env_runner
        import websockets.sync.client as wsc
        from openpi_client import websocket_client_policy as wcp

        assert getattr(env_runner.EnvRunner, "_orig_observer_hooked", False), "EnvRunner 未挂钩"
        assert getattr(wsc.ClientConnection, "_orig_observer_hooked", False), "ClientConnection 未挂钩"
        assert getattr(wcp.MMEVLAWebsocketClientPolicy, "_orig_observer_hooked", False), "策略客户端未挂钩"
        assert_official_namespace()
        print(f"OBSERVER_PREFLIGHT=PASS route={ROUTE} robomme={robomme.__file__} trace_writer={OE.tw.__file__} "
              f"groundsg_client={OE.groundsg.__file__}", flush=True)
        return
    if not argv or not argv[0].endswith(".py"):
        raise SystemExit("用法：framesamp_modul_client_wrap.py eval.py [eval.py 参数...]")
    script = os.path.abspath(argv[0])
    install(argv[1:])
    # 与 ``python eval.py`` 一致：sys.path[0] 为脚本所在目录，sys.argv 为 [eval.py, 参数...]
    sys.path[0] = os.path.dirname(script)
    sys.argv = [argv[0]] + argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
