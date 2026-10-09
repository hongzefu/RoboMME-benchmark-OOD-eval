"""SimpleMemVLA 推理 server（v7.5eval 方案 §2.3；新接口的策略侧）。

运行在独立子项目 venv（envs/smvla-env/ 的 pyproject.toml 与 uv.lock 锁定；Python 3.10、torch 2.4.1+cu121），
PYTHONPATH 指子模块 third_party/SimpleMemVLA（c564c17）根目录；环境不在本进程（口径 3）。

策略构建直接 import 上游 ``robomme_sim.eval_success.build_policy``：该模块顶层只 import
``robomme_sim.batched_policy`` / ``inproc_pool`` / ``robomme_env``，后两者顶层不 import sapien /
mani_skill（只调 prepare_sapien_runtime 调整 LD_LIBRARY_PATH），与旧官方进程执行的 import 链相同。
build_policy 的参数用上游 ``parse_args`` 按旧官方命令行解析得到（与 E0 的 Namespace 同值）。

协议（msgpack + openpi_client.msgpack_numpy，websockets compression=None、max_size=None）：
- 连接建立后 server 先发 metadata（权重路径、config.json sha256、版本、固定参数、预热结果）。
- ``{"reset": {"episode_key": str}}`` → 重设种子并新建缓冲：
  ``torch.manual_seed(0); np.random.seed(0)``，再 ``buffer = buffer_factory(); buffer.reset()``。
  旧官方在 evaluate_manifest 里于 run_group（即环境 reset）之前、同一进程内设种子；本 server 在
  局 reset 消息时设种子，时间点在本局第一次推理之前（口径 5）。策略采样只用 CUDA 生成器
  （dit_action_head.sample 的 torch.randn(device=cuda)），旧进程里环境 reset/step 走 CPU 仿真；
  推理路径（BatchedEvalPolicy.generate_batch、DiT action_head.sample、RoboMMEPolicy、
  Qwen3VL 图像/视频 processor）只用 torch 生成器：子任务解码为 argmax，唯一随机源是
  dit_action_head.py 的 torch.randn(device=cuda)；不用 numpy 全局随机数与 python random。
  env-digest 实测：benchmark 环境的 gym.make 消耗 numpy 全局随机数，env.reset 两者都不消耗，
  torch 全局随机数两者都不消耗——旧官方首推时 torch 状态即干净的 seed(0)，与本处等价。
- ``{"observe": {"frames": [{"front": uint8 HxWx3, "wrist": ...}, ...]}}`` → 逐帧
  ``buffer.observe(to_full(fr))``；回包带逐帧 sha256 供双向核对。
- ``{"infer": {"instruction": str, "state": float32[8]}}`` →
  ``processed = buffer._prepare_inputs(instruction)``；
  ``batched.generate_batch([processed], [state_norm(state)])`` → 回
  ``{"actions": actions_unnorm[:16], "actions_full": actions_unnorm, "subtask", "infer_ms", ...}``。
- 出错时回 ``{"error": traceback}`` 后关闭连接（与 openpi server 发 traceback 的做法一致）。

第三阶段（1006 计划；接口冻结说明 2.2、五节「服务外壳回包审计键」）：

- ``serve --policy-seed <n>`` 必填，``reseed(n)`` 替代常量 ``EPISODE_SEED=0``，生命周期不变（加载后一次、每局
  ``new_episode`` 前一次）；元数据（``--metadata_out``，即 ``server-metadata-<port>.json``）加 ``policy_seed``、
  ``argv``、``pid``、``port``。
- ``infer`` 回包加审计键 ``_sgeval_audit``：``{"channels": [{"channel": "task", "text": <模板化完整 prompt>,
  "token_ids": null, "mask": null, "tokenizer": <名>, "truncated": false}], "server_final_text": <同 text>,
  "pp_generation": null}``。完整 prompt 取自上游 ``processor.apply_chat_template`` 本次（或本局最近一次，提示缓存命中时
  上游不重新套模板）真实返回的文字：在 buffer 的 processor 实例上挂只读观察（调用原方法、原样返回），不多推理、不碰
  随机数、不改动作。环境变量 ``SGEVAL_AUDIT=0`` 时既不挂观察也不加键（``OBS_EQ`` 对照）。
"""

from __future__ import annotations

import os
import sys

# 本目录有 queue.py 等与标准库同名的模块；作为脚本运行时 sys.path[0] 即本目录，会遮蔽标准库
# （torch.fx 的 ``from queue import Queue`` 即会失败），故先把本目录移出 sys.path。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

# 旧官方 run_official_xhard0.sh 的运行口径；仅作为脚本运行时设定（单测加载本模块时不改测试进程的环境变量）。
_FIXED_ENV = {
    "OMP_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUTF8": "1",
}
# 确定性模式（第 4 步 --det）：cuBLAS 在建句柄时读取该变量，必须在 import torch 之前设定。
DET_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if __name__ == "__main__":  # 作为脚本运行：在 import numpy / torch 之前设定（OpenBLAS 线程数在加载时读取）
    for _k, _v in _FIXED_ENV.items():
        os.environ[_k] = _v
    if "--det" in sys.argv[1:]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DET_CUBLAS_WORKSPACE_CONFIG

import argparse
import asyncio
import hashlib
import json
import logging
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

#: 评估仓根（本文件在 src/robomme_hard_eval/servers/ 下）
REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "third_party" / "SimpleMemVLA"
OPENPI_CLIENT_SRC = REPO_ROOT / "third_party" / "mme-vla" / "packages" / "openpi-client" / "src"
DEFAULT_CKPT = REPO_ROOT / "artifacts" / "v7.5eval" / "ckpt" / "simplememvla_robomme"

# 旧官方 E0 命令行（O/scripts/run_official_xhard0.sh）里与策略有关的参数，逐项照抄；
# --episode_manifest/--shard/--episode_log/--resume/--video_dir 只影响旧客户端循环，不进 build_policy。
EXECUTE_HORIZON = 16
MAX_STEPS = 1300
OFFICIAL_ARGV = [
    "--dataset_split", "test",
    "--group_size", "1",
    "--execute_horizon", str(EXECUTE_HORIZON),
    "--max_steps", str(MAX_STEPS),
    "--num_denoising_steps", "10",
    "--eval_temperature", "1.0",
    "--max_subtask_tokens", "64",
    "--compute_dtype", "bfloat16",
    "--attn_implementation", "sdpa",
    "--num_gpus", "1",
]
#: 旧常量（第三阶段起只作单测里 object.__new__ 构造的 host 的回退值；生产由 --policy-seed 显式给出）
EPISODE_SEED = 0
#: 服务回包审计键（接口冻结说明五节）；SGEVAL_AUDIT=0 时不加
AUDIT_KEY = "_sgeval_audit"
ENV_AUDIT = "SGEVAL_AUDIT"


def audit_enabled() -> bool:
    """``SGEVAL_AUDIT`` 缺省开；为 ``0`` 时外壳完全不加审计键、不挂观察。"""
    return os.environ.get(ENV_AUDIT, "1") != "0"

log = logging.getLogger("smvla_server")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(arr: np.ndarray) -> str:
    """数值数组指纹：字节 + dtype + shape（与 recorder.array_sha256 同一定义）。"""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(a.tobytes())
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    return h.hexdigest()


def frame_sha(frame: np.ndarray) -> str:
    """单帧指纹：C 连续字节的 sha256（与 recorder.frame_sha256 同一定义）。"""
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _setup_paths() -> None:
    for p in (str(SUBMODULE_ROOT), str(OPENPI_CLIENT_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def official_args(ckpt: str) -> argparse.Namespace:
    """用上游 parse_args 按旧官方命令行解析出 Namespace（默认值与 E0 完全相同）。"""
    from robomme_sim import eval_success

    saved = sys.argv
    try:
        sys.argv = ["eval_success", "--pretrained_checkpoint", str(ckpt), *OFFICIAL_ARGV]
        return eval_success.parse_args()
    finally:
        sys.argv = saved


def make_closures(buffer_factory, normalize_state, batched):
    """逐行照抄上游 c564c17 robomme_sim/eval_success.py::run_group 第 180–190 行的两个局部闭包。

    与上游原文的逐行一致可用 ast 从上游文件取出该段比对（原单测已随旧测试删除）。
    """
    import torch

    image_keys = list(buffer_factory().image_keys)
    cam_map = {k.split(".")[-1]: k for k in image_keys}

    def to_full(frame):
        return {cam_map.get(k, k): np.asarray(v, dtype=np.uint8) for k, v in frame.items()}

    def state_norm(state):
        if normalize_state is None:
            return None
        s = normalize_state.normalize(torch.from_numpy(np.asarray(state, dtype=np.float32)))
        return s.unsqueeze(0).to(batched.device)

    return to_full, state_norm


def rng_digest() -> dict:
    """当前 torch CPU / CUDA 与 numpy 全局随机状态的 sha256（核对重设种子与预热后恢复）。"""
    import torch

    out = {"torch_cpu": sha256_bytes(torch.get_rng_state().numpy().tobytes())}
    if torch.cuda.is_available():
        out["torch_cuda"] = sha256_bytes(torch.cuda.get_rng_state().numpy().tobytes())
    st = np.random.get_state()
    out["numpy"] = sha256_bytes(st[1].tobytes() + str(st[2:]).encode())
    return out


def reseed(seed: int = EPISODE_SEED) -> None:
    """口径 5：每局第一次推理前重设种子（旧官方 evaluate_manifest 的两行）；第三阶段种子由 --policy-seed 给出。"""
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


class SMVLAPolicyHost:
    """持有模型；每条连接一个 Episode 状态（buffer）。"""

    def __init__(self, ckpt: str, *, policy_seed: int):
        import torch
        import transformers

        self.policy_seed = int(policy_seed)
        _setup_paths()
        t0 = time.monotonic()
        from robomme_sim.eval_success import build_policy

        self.ckpt = str(Path(ckpt).resolve())
        self.args = official_args(self.ckpt)
        self.batched, self.buffer_factory, self.normalize_state = build_policy(self.args)
        self.to_full, self.state_norm = make_closures(self.buffer_factory, self.normalize_state, self.batched)
        self.load_s = time.monotonic() - t0
        # 参照随机状态：加载完成后、任何预热 / 推理之前做一次纯净重设种子并取摘要；
        # 之后每局 reset（new_episode）后的摘要都必须与之相同。
        reseed(self.policy_seed)
        self.rng_ref = rng_digest()
        ck = Path(self.ckpt)
        self.metadata = {
            "policy": "smvla",
            "ckpt": self.ckpt,
            "ckpt_config_sha256": sha256_file(ck / "config.json"),
            "ckpt_stats_sha256": sha256_file(ck / "stats.json") if (ck / "stats.json").exists() else None,
            "submodule_commit": _git_head(SUBMODULE_ROOT),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
            },
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "fixed_env": {k: os.environ.get(k) for k in _FIXED_ENV},
            "args": {k: v for k, v in sorted(vars(self.args).items()) if isinstance(v, (int, float, str, bool, type(None)))},
            "execute_horizon": EXECUTE_HORIZON,
            "episode_seed": self.policy_seed,
            "policy_seed": self.policy_seed,
            "load_s": round(self.load_s, 3),
            "warmup": None,
            "rng_ref": self.rng_ref,
        }

    # ---- 单局操作 ----
    def seed(self) -> int:
        """本服务的模型种子（object.__new__ 构造的单测 host 没有该属性时回退旧常量）。"""
        return int(getattr(self, "policy_seed", EPISODE_SEED))

    def new_episode(self):
        reseed(self.seed())
        buf = self.buffer_factory()
        buf.reset()
        return buf

    def observe(self, buf, frames: list[dict]) -> list[dict]:
        shas = []
        for fr in frames:
            fr = {k: np.array(v, copy=True) for k, v in fr.items()}  # 解包得到的只读视图换成普通数组，值不变
            shas.append({k: frame_sha(v) for k, v in sorted(fr.items())})
            buf.observe(self.to_full(fr))
        return shas

    def infer(self, buf, instruction: str, state) -> dict:
        import torch

        state = np.array(state, copy=True)
        audit = audit_enabled()
        if audit:
            self._watch_prompt(buf)
        t0 = time.monotonic()
        processed = buf._prepare_inputs(instruction)
        decisions = self.batched.generate_batch([processed], [self.state_norm(state)])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_ms = (time.monotonic() - t0) * 1000.0
        actions_unnorm, subtask = decisions[0]
        reply = {
            "actions": actions_unnorm[:EXECUTE_HORIZON],
            "actions_full": actions_unnorm,
            "subtask": str(subtask),
            "infer_ms": infer_ms,
            "recv_state_sha": array_sha(state),
            "recv_instruction_sha": sha256_bytes(instruction.encode("utf-8")),
        }
        if audit:
            reply[AUDIT_KEY] = self._audit_block(buf)
        return reply

    # ---- 回包审计（只观察上游真实结果） ----
    def _watch_prompt(self, buf) -> None:
        """在 buffer 的 processor 实例上挂一次只读观察：调用上游原 ``apply_chat_template``、原样返回，顺手记下返回的
        模板化文字。processor 缺失或已挂过则不动。"""
        proc = getattr(buf, "processor", None)
        if proc is None or getattr(proc, "_sgeval_prompt_watch", False):
            return
        orig = proc.apply_chat_template
        host = self

        def watched(*a, **k):
            out = orig(*a, **k)
            text = out if isinstance(out, str) else (out[0] if isinstance(out, (list, tuple)) and out else None)
            if isinstance(text, str):
                host._sgeval_prompt = text
            return out

        proc.apply_chat_template = watched
        proc._sgeval_prompt_watch = True

    def _audit_block(self, buf) -> dict:
        proc = getattr(buf, "processor", None)
        tok = getattr(proc, "tokenizer", None)
        text = getattr(self, "_sgeval_prompt", None)
        name = getattr(tok, "name_or_path", None) or (type(tok).__name__ if tok is not None else None)
        return {"channels": [{"channel": "task", "text": text, "token_ids": None, "mask": None, "tokenizer": name,
                              "truncated": False if text is not None else None}],
                "server_final_text": text, "pp_generation": None}

    def warmup(self) -> dict:
        """加载后跑一次假推理（全零图 + 零状态）；随后走真实的每局重设路径（new_episode），
        核对其随机状态摘要等于加载后纯净重设的参照摘要 rng_ref（预热确实消耗了随机数时才有区分力）。"""
        t0 = time.monotonic()
        buf = self.new_episode()
        z = np.zeros((256, 256, 3), dtype=np.uint8)
        self.observe(buf, [{"front": z, "wrist": z}])
        out = self.infer(buf, "warmup", np.zeros(8, dtype=np.float32))
        wall = time.monotonic() - t0
        after_infer = rng_digest()
        self.new_episode()
        after_reset = rng_digest()
        res = {"warmup_s": round(wall, 3), "infer_ms": round(out["infer_ms"], 3),
               "rng_consumed_by_warmup": after_infer != self.rng_ref,
               "rng_restored": after_reset == self.rng_ref, "rng": after_reset}
        self.metadata["warmup"] = res
        return res


async def _handler(host: SMVLAPolicyHost, websocket):
    from openpi_client import msgpack_numpy
    import websockets

    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack(host.metadata))
    buf = None
    episode_key = None
    n_infer = 0
    while True:
        try:
            raw = await websocket.recv()
        except websockets.ConnectionClosed:
            log.info("连接关闭 episode=%s infer=%d", episode_key, n_infer)
            return
        try:
            msg = msgpack_numpy.unpackb(raw)
            req_sha = sha256_bytes(raw)
            if "reset" in msg:
                t0 = time.monotonic()
                episode_key = str(msg["reset"].get("episode_key"))
                buf = host.new_episode()
                n_infer = 0
                reply = {"reset_finished": True, "episode_key": episode_key,
                         "reset_time_ms": (time.monotonic() - t0) * 1000.0, "rng": rng_digest()}
                reply["rng_matches_ref"] = reply["rng"] == host.rng_ref
            elif "observe" in msg:
                if buf is None:
                    raise RuntimeError("observe 之前未 reset")
                t0 = time.monotonic()
                shas = host.observe(buf, list(msg["observe"]["frames"]))
                reply = {"observe_finished": True, "n": len(shas), "frame_sha": shas,
                         "observe_time_ms": (time.monotonic() - t0) * 1000.0}
            elif "infer" in msg:
                if buf is None:
                    raise RuntimeError("infer 之前未 reset")
                p = msg["infer"]
                reply = host.infer(buf, str(p["instruction"]), p["state"])
                reply["decision"] = n_infer
                n_infer += 1
            else:
                raise ValueError(f"未知消息键 {sorted(msg)}")
            reply["req_sha"] = req_sha
            await websocket.send(packer.pack(reply))
        except websockets.ConnectionClosed:
            return
        except Exception:
            tb = traceback.format_exc()
            log.error("处理消息出错：\n%s", tb)
            try:
                await websocket.send(packer.pack({"error": tb}))
                await websocket.close()
            except Exception:
                pass
            return


async def _serve(host: SMVLAPolicyHost, bind: str, port: int) -> None:
    import websockets.asyncio.server as _server

    async def handler(ws):
        await _handler(host, ws)

    async with _server.serve(handler, bind, port, compression=None, max_size=None,
                             ping_interval=None, ping_timeout=None) as server:
        print(f"SMVLA_SERVER_READY host={bind} port={port} load_s={host.load_s:.1f}", flush=True)
        await server.serve_forever()


def enable_det() -> dict:
    """确定性模式（默认关）：torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8。
    环境变量须已在 import torch 之前设好（见文件头），这里只核对，不在 torch 已加载后补设。"""
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DET_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(f"--det 需要在 import torch 前设 CUBLAS_WORKSPACE_CONFIG={DET_CUBLAS_WORKSPACE_CONFIG}，"
                           f"当前为 {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
    torch.use_deterministic_algorithms(True)
    return {"det": True, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def cmd_serve(a) -> int:
    t0 = time.monotonic()
    det_info = enable_det() if a.det else {"det": False, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    print(f"SMVLA_DET det={'on' if det_info['det'] else 'off'} cublas={det_info['cublas_workspace_config']}", flush=True)
    host = SMVLAPolicyHost(a.ckpt, policy_seed=a.policy_seed)
    host.metadata.update(det_info)
    host.metadata.update(policy_seed=int(a.policy_seed), argv=list(sys.argv), pid=os.getpid(), port=a.port,
                         audit=audit_enabled())
    print(f"SMVLA_LOAD load_s={host.load_s:.1f} config_sha={host.metadata['ckpt_config_sha256'][:12]} "
          f"torch={host.metadata['versions']['torch']} gpu={host.metadata['gpu_name']}", flush=True)
    if a.expect_config_sha and host.metadata["ckpt_config_sha256"] != a.expect_config_sha:
        print(f"SMVLA_CONFIG_SHA=FAIL got={host.metadata['ckpt_config_sha256']} want={a.expect_config_sha}", flush=True)
        return 2
    if a.warmup:
        w = host.warmup()
        print(f"SMVLA_WARMUP warmup_s={w['warmup_s']} infer_ms={w['infer_ms']:.1f} "
              f"rng_consumed={w['rng_consumed_by_warmup']} rng_restored={w['rng_restored']}",
              flush=True)
    if a.metadata_out:
        Path(a.metadata_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.metadata_out).write_text(json.dumps(host.metadata, sort_keys=True, ensure_ascii=False, indent=1))
    host.metadata["startup_s"] = round(time.monotonic() - t0, 3)
    asyncio.run(_serve(host, a.host, a.port))
    return 0


def main(argv=None) -> int:
    bad = {k: os.environ.get(k) for k, v in _FIXED_ENV.items() if os.environ.get(k) != v}
    if bad:
        raise RuntimeError(f"固定环境变量未生效：{bad}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="加载权重并起 websocket server")
    s.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--warmup", action="store_true", help="加载后跑一次假推理再重设种子")
    s.add_argument("--expect_config_sha", default=None, help="config.json 期望 sha256，不符即退出 2")
    s.add_argument("--metadata_out", default=None, help="服务元数据 JSON（server-metadata-<port>.json，含 policy_seed）")
    s.add_argument("--policy-seed", type=int, required=True,
                   help="模型种子（必填）：reseed(n) 替代旧常量 0，加载后一次、每局 new_episode 前一次")
    s.add_argument("--det", action="store_true",
                   help="确定性模式：torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8（默认关）")
    s.set_defaults(func=cmd_serve)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
