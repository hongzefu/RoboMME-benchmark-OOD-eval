"""SimpleMemVLA 原侧（原版分支 ``official-xhard0-0929``／``4e0c04f``）的包装启动器：同进程执行
``robomme_sim.eval_success``，外挂只读钩子，写出与 GroundSG 原侧同布局的逐局 ``<key>.a<N>/{trace.jsonl,frames/,arrays.npz}``。

用法（cwd = 原版工作树根，REC_ROOT 必设；参数与 ``python -m robomme_sim.eval_success`` 完全相同）：
    python smvla_wrap.py --pretrained_checkpoint ... --episode_manifest M --shard 0/10 --episode_log L --resume
    python smvla_wrap.py --orig-preflight

钩子（计划第二部分一节 S7「钩子铁律」：只复制主机端 numpy、不调随机函数、不做 GPU 运算、不改参数与返回值；
钩子内异常只记 ``OBSERVER_HOOK_ERROR`` 并累加 ``observer_hook_errors``，绝不外抛）：
- ``InProcSimPool.reset``（主线程，``fut.result`` 之后）：reset 返回的全部演示帧 + 初始帧（前视、腕部）、8 维状态、
  instruction → ``trace`` 的 ``demo`` 行（C2）与原始帧。
- ``InProcSimPool.step``（主线程，``fut.result`` 之后）：按返回的 ``consumed`` 截取动作块，逐行换算成实际交给环境的
  float64×8 动作（``step_arrays.smvla_exec_rows``，与 ``SimEnvService.step``／``RoboMMESimEnv.step_one`` 逐字相同），
  每行一个 ``step``：画面与状态取该步返回的最后一帧；块内最后一步写返回的 ``status``，其余步原侧不可见写
  ``NOT_OBSERVED``；``terminated``／``truncated`` 原侧不可见写 ``NOT_OBSERVED``（C9）；环境报错那一步无观测，
  记 ``observed=false`` 与原因（C8）；池返回 ``error``（线程异常）时实际执行行数不可知，不记步，只在 ``end`` 记
  ``pool_errors``（与原版逐局日志 ``steps`` 不计该块一致）。
- ``BatchedEvalPolicy.generate_batch``：调用前记 ``request``（名 ``infer``，载荷为逻辑输入
  ``canonical_bytes({"instruction","state"(float32),"frames":[[前视 sha, 腕部 sha], …]})``，与新侧 ``smvla_client``
  ``logical_request`` 逐字节同构，见 ``logical_input``，C10），调用后记完整动作块 ``response`` 与子任务
  文本（作为之后各步的 ``subgoal``）。GPU 张量一律不碰。
- 局边界：包 ``run_group``（``evaluate_manifest`` 的逐行循环每行调一次，``specs`` 给出 task 与 episode）；终态取
  ``details[0]["status"]``（与 ``evaluate_manifest`` 写逐局日志的口径相同：缺失或非四终态记 ``error``，``run_group``
  抛异常记 ``error``）。

⚠ 与 ``-m`` 直接运行的唯一差别：为了在 ``main()`` 之前把 ``run_group`` 换成包装，本启动器用
importlib 以模块本名 ``robomme_sim.eval_success`` 执行模块体（此时 ``if __name__ == "__main__"`` 不触发），
替换活模块命名空间里的 ``run_group`` 后再调用同一个 ``main()``（见 ``run_observed``）。模块源码、参数解析、种子设置位置都不变。
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

_OBS_DIR = str(Path(__file__).resolve().parent)
if _OBS_DIR not in sys.path:
    sys.path.append(_OBS_DIR)
import _obs_common as C  # noqa: E402
import orig_episode as OE  # noqa: E402
import step_arrays as SA  # noqa: E402

tw = OE.tw
NOT_OBSERVED = OE.NOT_OBSERVED
ROUTE = "smvla/orig"

ERR = C.HookErrors()
_lock = threading.RLock()
_state: dict = {"ep": None, "root": None}


def _fresh_episode_state(ep) -> None:
    _state.update(ep=ep, instruction=None, last_state=None, pending_frames=[], decisions=0,
                  pool_errors=[])


def _ep_error(ep, where: str) -> None:
    if ep is not None:
        ep.hook_error(where)
    else:
        ERR.note(where)


# ------------------------------------------------------------------ 环境钩子（主线程）

def _record_reset(ep, r) -> None:
    if not (isinstance(r, dict) and r.get("ok")):
        _state["reset_reason"] = r.get("reason") if isinstance(r, dict) else repr(r)
        return
    fronts = [np.asarray(f["front"]) for f in r["frames"]]
    wrists = [np.asarray(f["wrist"]) for f in r["frames"]]
    states = [np.asarray(s) for s in r["states"]]
    ep.on_reset(fronts, wrists, states, r["instruction"])
    _state["instruction"] = r["instruction"]
    _state["last_state"] = np.asarray(r["states"][-1], dtype=np.float32).copy()
    _state["pending_frames"] = [[tw.image_sha256(f), tw.image_sha256(w)] for f, w in zip(fronts, wrists)]


def _record_step(ep, chunk, r) -> None:
    if not isinstance(r, dict) or "error" in r:
        _state["pool_errors"].append(str(r.get("error") if isinstance(r, dict) else r)[:500])
        return
    consumed = int(r.get("consumed", 0) or 0)
    rows = SA.smvla_exec_rows(chunk, consumed)
    err = r.get("error_message")
    n_obs = max(0, consumed - 1) if err is not None else consumed
    frames = list(r.get("frames") or [])
    states = list(r.get("states") or [])
    per = len(frames) // n_obs if n_obs else 0
    aligned = n_obs == 0 or (per >= 1 and len(frames) == per * n_obs and len(states) == per * n_obs)
    if not aligned:
        ep.hook_error(f"pool.step.frame_count consumed={consumed} frames={len(frames)} states={len(states)}")
    for j in range(consumed):
        last = j == consumed - 1
        if j < n_obs and aligned:
            k = (j + 1) * per - 1
            extra = {"done": bool(r.get("done"))} if last else {}
            ep.on_step(rows[j], np.asarray(frames[k]["front"]), np.asarray(frames[k]["wrist"]), np.asarray(states[k]),
                       r.get("status") if last else NOT_OBSERVED, **extra)
        elif j < n_obs:
            ep.on_missing_step(rows[j], reason="observer_frame_count_mismatch")
        else:
            ep.on_missing_step(rows[j], reason=f"env_step_error: {err}"[:500])
    _state["pending_frames"].extend([tw.image_sha256(np.asarray(f["front"])), tw.image_sha256(np.asarray(f["wrist"]))]
                                    for f in frames)
    if states:
        _state["last_state"] = np.asarray(states[-1], dtype=np.float32).copy()


def _patch_pool(mod) -> None:
    """包 InProcSimPool.reset/step：在主线程里、fut.result 返回之后，从返回的 dict 只读记录；
    不进环境线程，不占 step_timeout／reset_timeout 的计时。"""
    cls = mod.InProcSimPool
    orig_reset, orig_step = cls.reset, cls.step

    def reset(self, specs):
        out = orig_reset(self, specs)
        ep = _state["ep"]
        try:
            if ep is not None:
                if len(specs) != 1:
                    ep.hook_error(f"pool.reset.group_size={len(specs)}（清单模式应为 1，只记录槽位 0）")
                _record_reset(ep, out[0])
        except Exception:  # noqa: BLE001
            _ep_error(ep, "pool.reset.post")
        return out

    def step(self, action_chunks, active):
        out = orig_step(self, action_chunks, active)
        ep = _state["ep"]
        try:
            if ep is not None:
                for i in active:
                    if out[i] is None:  # alive=False 的槽位未派发
                        continue
                    if i != 0:
                        ep.hook_error(f"pool.step.slot={i}（只记录槽位 0）")
                        continue
                    _record_step(ep, action_chunks[i], out[i])
        except Exception:  # noqa: BLE001
            _ep_error(ep, "pool.step.post")
        return out

    cls.reset, cls.step = reset, step
    cls._orig_observer_hooked = True


# ------------------------------------------------------------------ 策略钩子

def logical_input(instruction, state, pending_frames) -> dict:
    """C10 逻辑输入，与新侧 ``smvla_client`` 的 ``logical_request`` 逐字节同构：
    ``{"instruction": 指令, "state": float32 状态副本, "frames": [[前视 sha, 腕部 sha], …]}``，``frames`` 为自上次决策以来
    策略缓冲 observe 的全部帧（首次决策含 reset 返回的全部演示帧 + 初始帧），哈希为 ``trace_writer.image_sha256``；
    ``state`` 与原版 ``run_group`` 的 ``cur_state = np.asarray(states[-1], dtype=np.float32)`` 同值。"""
    return {"instruction": instruction, "state": np.array(state, copy=True), "frames": pending_frames}


def _patch_policy(mod) -> None:
    cls = mod.BatchedEvalPolicy
    orig = cls.generate_batch

    def generate_batch(self, processed_list, state_norm_list):
        ep = _state["ep"]
        try:
            if ep is not None:
                ep.on_request("infer", tw.canonical_bytes(logical_input(
                    _state["instruction"], _state["last_state"], _state["pending_frames"])))
                _state["pending_frames"] = []
        except Exception:  # noqa: BLE001
            _ep_error(ep, "generate_batch.pre")
        out = orig(self, processed_list, state_norm_list)
        try:
            if ep is not None:
                acts, sub = out[0]
                ep.on_response(np.asarray(acts))
                ep.subgoal = sub if (sub is None or isinstance(sub, str)) else str(sub)
                with _lock:
                    _state["decisions"] += 1
        except Exception:  # noqa: BLE001
            _ep_error(ep, "generate_batch.post")
        return out

    cls.generate_batch = generate_batch
    cls._orig_observer_hooked = True


# ------------------------------------------------------------------ 局边界

def episode_status(details, ret, exc) -> str:
    """与 ``evaluate_manifest`` 写逐局日志同口径的终态：异常 → error；``details[0]["status"]`` 缺失或非四终态 → error。
    没有 ``details``（非清单模式）时按 ``run_group`` 返回值：True → success、False → fail、None → error。"""
    if exc is not None:
        return "error"
    if details:
        s = details[0].get("status") or "error"
        return s if s in C.TERMINALS else "error"
    if isinstance(ret, list) and ret:
        return {True: "success", False: "fail"}.get(ret[0], "error")
    return "error"


def _wrap_run_group(orig, seeds: dict):
    def run_group(args, pool, batched, buffer_factory, normalize_state, task, specs, *a, **kw):
        ep = None
        try:
            src = int(specs[0]["episode"])
            if (task, src) not in seeds:
                raise KeyError(f"清单里没有 ({task}, {src}) 的 seed，无法定局目录")
            ep = OE.OrigEpisode(_state["root"], route=ROUTE, task=task, source_episode=src, seed=seeds[(task, src)],
                                max_steps=int(getattr(args, "max_steps", 1300)), errors=ERR)
            _fresh_episode_state(ep)
        except Exception:  # noqa: BLE001
            ERR.note("episode.open")
            _fresh_episode_state(None)
        ret, exc = None, None
        try:
            ret = orig(args, pool, batched, buffer_factory, normalize_state, task, specs, *a, **kw)
            return ret
        except BaseException as e:
            exc = f"{type(e).__name__}: {e}"[:800]
            raise
        finally:
            _state["ep"] = None
            if ep is not None:
                try:
                    status = episode_status(kw.get("details"), ret, exc)
                    info = ep.close(status, decisions=_state["decisions"], pool_errors=list(_state["pool_errors"]),
                                    exception=exc, reset_reason=_state.get("reset_reason"))
                    _state.pop("reset_reason", None)
                    print(f"OBSERVER_EPISODE route={ROUTE} episode={info.get('episode')} status={info.get('status')} "
                          f"steps_attempted={info.get('steps_attempted')} steps_observed={info.get('steps_observed')} "
                          f"frames_recorded={info.get('frames_recorded')} hook_errors={info.get('observer_hook_errors')}",
                          flush=True)
                    C.JsonlLog(_state["root"] / "observer-index.jsonl").write(info)
                except Exception:  # noqa: BLE001
                    ERR.note("episode.close", getattr(ep, "name", None))

    return run_group


def install(argv: list[str]) -> dict:
    _state["root"] = C.rec_root()
    ERR.root = _state["root"]
    seeds = C.manifest_seeds(C.parse_flag(argv, ("--episode_manifest",)))
    C.PostImportHooks({"robomme_sim.inproc_pool": _patch_pool, "robomme_sim.batched_policy": _patch_policy}).install()
    return seeds


def main() -> None:
    argv = sys.argv[1:]
    # 与 ``python -m`` 一致：sys.path[0] 为当前目录
    sys.path[0] = os.getcwd()
    if argv[:1] == ["--orig-preflight"]:
        install([])
        import robomme_sim.batched_policy as bp
        import robomme_sim.inproc_pool as ip

        assert getattr(ip.InProcSimPool, "_orig_observer_hooked", False), "InProcSimPool 未挂钩"
        assert getattr(bp.BatchedEvalPolicy, "_orig_observer_hooked", False), "BatchedEvalPolicy 未挂钩"
        assert "robomme_hard" not in sys.modules, "原侧不得导入 robomme_hard"
        print(f"OBSERVER_PREFLIGHT=PASS route={ROUTE} robomme_sim={ip.__file__} trace_writer={tw.__file__} "
              f"groundsg_client={OE.groundsg.__file__}", flush=True)
        return
    seeds = install(argv)
    run_observed("robomme_sim.eval_success", argv, seeds)


def run_observed(modname: str, argv: list[str], seeds: dict) -> None:
    """执行模块体（__name__ 不是 "__main__"，故不自动调 main），替换活模块命名空间里的 run_group 后调 main()。

    注意不能用 runpy.run_module 的返回值打补丁：它返回的是模块全局变量的副本，函数仍引用原命名空间。
    原版 ``main()`` 清单模式以 ``os._exit(code)`` 结束进程，退出码不经本函数；逐局 trace 已在每局收尾时落盘。
    """
    import importlib.util

    spec = importlib.util.find_spec(modname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    sys.argv = [spec.origin] + argv
    spec.loader.exec_module(mod)
    mod.run_group = _wrap_run_group(mod.run_group, seeds)
    mod.main()


if __name__ == "__main__":
    main()
