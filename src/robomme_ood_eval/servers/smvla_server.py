"""SimpleMemVLA inference server (the policy side of the new interface).

Runs in a separate sub-project venv (locked by the pyproject.toml and uv.lock in envs/smvla-env/; Python 3.10,
torch 2.4.1+cu121), with PYTHONPATH pointing at the root of the submodule third_party/SimpleMemVLA (c564c17); the
environment is not in this process.

The policy is built by importing the upstream ``robomme_sim.eval_success.build_policy`` directly: at top level that
module only imports ``robomme_sim.batched_policy`` / ``inproc_pool`` / ``robomme_env``, and the latter two do not
import sapien / mani_skill at top level (they only call prepare_sapien_runtime to adjust LD_LIBRARY_PATH), which is
the same import chain the old official process executed. build_policy's arguments are obtained by parsing the old
official command line with the upstream ``parse_args`` (same values as the original official Namespace).

Protocol (msgpack + openpi_client.msgpack_numpy, websockets compression=None, max_size=None):
- After the connection is established the server first sends metadata (weight path, config.json sha256, versions,
  fixed parameters, warm-up result).
- ``{"reset": {"episode_key": str}}`` -> reseed and create a new buffer:
  ``torch.manual_seed(0); np.random.seed(0)``, then ``buffer = buffer_factory(); buffer.reset()``.
  The old official code seeds in evaluate_manifest before run_group (i.e. the environment reset), in the same
  process; this server seeds on the episode reset message, before the episode's first inference. Policy sampling
  only uses the CUDA generator (torch.randn(device=cuda) in dit_action_head.sample), while in the old process the
  environment reset/step ran a CPU simulation; the inference path (BatchedEvalPolicy.generate_batch, DiT
  action_head.sample, RoboMMEPolicy, Qwen3VL image/video processor) only uses torch generators: subtask decoding is
  argmax, and the only random source is torch.randn(device=cuda) in dit_action_head.py; neither numpy global random
  numbers nor python random are used. Measured with an environment digest: the benchmark environment's gym.make
  consumes numpy global random numbers, env.reset consumes neither, and torch global random numbers are consumed by
  neither -- so at the old official first inference the torch state was a clean seed(0), equivalent to this.
- ``{"observe": {"frames": [{"front": uint8 HxWx3, "wrist": ...}, ...]}}`` -> per frame
  ``buffer.observe(to_full(fr))``; the reply carries per-frame sha256 for two-way checking.
- ``{"infer": {"instruction": str, "state": float32[8]}}`` ->
  ``processed = buffer._prepare_inputs(instruction)``;
  ``batched.generate_batch([processed], [state_norm(state)])`` -> reply
  ``{"actions": actions_unnorm[:16], "actions_full": actions_unnorm, "subtask", "infer_ms", ...}``.
- On error reply ``{"error": traceback}`` and then close the connection (same as the openpi server sending a
  traceback).

Seed and audit additions:

- ``serve --policy-seed <n>`` is required; ``reseed(n)`` replaces the constant ``EPISODE_SEED=0`` with the same
  lifecycle (once after loading, once before each episode's ``new_episode``); the metadata (``--metadata_out``, i.e.
  ``server-metadata-<port>.json``) adds ``policy_seed``, ``argv``, ``pid``, ``port``.
- ``infer`` replies add the audit key ``_sgeval_audit``: ``{"channels": [{"channel": "task", "text": <full templated
  prompt>, "token_ids": null, "mask": null, "tokenizer": <name>, "truncated": false}], "server_final_text": <same
  text>, "pp_generation": null}``. The full prompt is the text actually returned by the upstream
  ``processor.apply_chat_template`` this time (or most recently in this episode; on a prompt-cache hit upstream does
  not re-apply the template): a read-only observer is attached to the buffer's processor instance (calls the
  original method, returns unchanged), with no extra inference, no random numbers touched and actions unchanged.
  With the environment variable ``SGEVAL_AUDIT=0`` neither the observer nor the key is added (``OBS_EQ``
  comparison).
"""

from __future__ import annotations

import os
import sys

# this directory has modules named like the standard library (e.g. queue.py); when run as a script sys.path[0] is
# this directory and would shadow the standard library (torch.fx's ``from queue import Queue`` would fail), so remove
# this directory from sys.path first.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

# runtime settings of the old official launcher; applied only when run as a script (loading this module in unit tests
# does not change the test process's environment variables).
_FIXED_ENV = {
    "OMP_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUTF8": "1",
}
# deterministic mode (--det): cuBLAS reads this variable when creating its handle, so it must be set before importing torch.
DET_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if __name__ == "__main__":  # run as a script: set before importing numpy / torch (OpenBLAS reads its thread count at load time)
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

#: evaluation repo root (this file lives in src/robomme_ood_eval/servers/)
REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "third_party" / "SimpleMemVLA"
OPENPI_CLIENT_SRC = REPO_ROOT / "third_party" / "mme-vla" / "packages" / "openpi-client" / "src"

# policy-related arguments of the old official command line, copied item by item;
# --episode_manifest/--shard/--episode_log/--resume/--video_dir only affect the old client loop and do not reach
# build_policy.
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
#: legacy constant (now only the fallback for hosts built via object.__new__ in unit tests; production passes
#: --policy-seed explicitly)
EPISODE_SEED = 0
#: server reply audit key; not added when SGEVAL_AUDIT=0
AUDIT_KEY = "_sgeval_audit"
ENV_AUDIT = "SGEVAL_AUDIT"


def audit_enabled() -> bool:
    """``SGEVAL_AUDIT`` is on by default; with ``0`` the wrapper adds no audit key and attaches no observer."""
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
    """Numeric array fingerprint: bytes + dtype + shape (same definition as recorder.array_sha256)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(a.tobytes())
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    return h.hexdigest()


def frame_sha(frame: np.ndarray) -> str:
    """Single-frame fingerprint: sha256 of the C-contiguous bytes (same definition as recorder.frame_sha256)."""
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _setup_paths() -> None:
    for p in (str(SUBMODULE_ROOT), str(OPENPI_CLIENT_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def official_args(ckpt: str) -> argparse.Namespace:
    """Parse a Namespace from the old official command line with the upstream parse_args (defaults identical to the
    original official run)."""
    from robomme_sim import eval_success

    saved = sys.argv
    try:
        sys.argv = ["eval_success", "--pretrained_checkpoint", str(ckpt), *OFFICIAL_ARGV]
        return eval_success.parse_args()
    finally:
        sys.argv = saved


def make_closures(buffer_factory, normalize_state, batched):
    """Line-by-line copy of the two local closures in upstream c564c17 robomme_sim/eval_success.py::run_group
    (lines 180-190).

    Line-by-line identity with the upstream text can be checked by extracting that block from the upstream file with
    ast.
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
    """sha256 of the current torch CPU / CUDA and numpy global random states (to check reseeding and restoration after
    warm-up)."""
    import torch

    out = {"torch_cpu": sha256_bytes(torch.get_rng_state().numpy().tobytes())}
    if torch.cuda.is_available():
        out["torch_cuda"] = sha256_bytes(torch.cuda.get_rng_state().numpy().tobytes())
    st = np.random.get_state()
    out["numpy"] = sha256_bytes(st[1].tobytes() + str(st[2:]).encode())
    return out


def reseed(seed: int = EPISODE_SEED) -> None:
    """Reseed before each episode's first inference (the two lines of the old official evaluate_manifest); the seed
    comes from --policy-seed."""
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
    """Holds the model; one Episode state (buffer) per connection."""

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
        # reference random state: after loading and before any warm-up / inference, do one clean reseed and take its
        # digest; the digest after every later episode reset (new_episode) must equal it.
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

    # ---- per-episode operations ----
    def seed(self) -> int:
        """This server's model seed (unit-test hosts built via object.__new__ lack the attribute and fall back to the
        legacy constant)."""
        return int(getattr(self, "policy_seed", EPISODE_SEED))

    def new_episode(self):
        reseed(self.seed())
        buf = self.buffer_factory()
        buf.reset()
        return buf

    def observe(self, buf, frames: list[dict]) -> list[dict]:
        shas = []
        for fr in frames:
            fr = {k: np.array(v, copy=True) for k, v in fr.items()}  # replace read-only unpacked views with plain arrays; values unchanged
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

    # ---- reply audit (only observes real upstream results) ----
    def _watch_prompt(self, buf) -> None:
        """Attach a read-only observer once on the buffer's processor instance: call the upstream original
        ``apply_chat_template``, return unchanged, and record the returned templated text. Does nothing if the
        processor is missing or already observed."""
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
        """Run one fake inference after loading (all-zero images + zero state), then go through the real per-episode
        reset path (new_episode) and check that its random-state digest equals the reference digest rng_ref from the
        clean reseed after loading (only discriminating when warm-up actually consumed random numbers)."""
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
            log.info("connection closed episode=%s infer=%d", episode_key, n_infer)
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
                    raise RuntimeError("observe before reset")
                t0 = time.monotonic()
                shas = host.observe(buf, list(msg["observe"]["frames"]))
                reply = {"observe_finished": True, "n": len(shas), "frame_sha": shas,
                         "observe_time_ms": (time.monotonic() - t0) * 1000.0}
            elif "infer" in msg:
                if buf is None:
                    raise RuntimeError("infer before reset")
                p = msg["infer"]
                reply = host.infer(buf, str(p["instruction"]), p["state"])
                reply["decision"] = n_infer
                n_infer += 1
            else:
                raise ValueError(f"unknown message keys {sorted(msg)}")
            reply["req_sha"] = req_sha
            await websocket.send(packer.pack(reply))
        except websockets.ConnectionClosed:
            return
        except Exception:
            tb = traceback.format_exc()
            log.error("error while handling message:\n%s", tb)
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
    """Deterministic mode (off by default): torch.use_deterministic_algorithms(True) +
    CUBLAS_WORKSPACE_CONFIG=:4096:8. The environment variable must already be set before importing torch (see the file
    header); this only checks it and never sets it after torch is loaded."""
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DET_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(f"--det requires CUBLAS_WORKSPACE_CONFIG={DET_CUBLAS_WORKSPACE_CONFIG} set before importing "
                           f"torch; currently {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
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
        raise RuntimeError(f"fixed environment variables not in effect: {bad}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="load weights and start the websocket server")
    s.add_argument("--ckpt", required=True, help="checkpoint directory (required; there is no default)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--warmup", action="store_true", help="run one fake inference after loading, then reseed")
    s.add_argument("--expect_config_sha", default=None, help="expected config.json sha256; exit 2 on mismatch")
    s.add_argument("--metadata_out", default=None, help="server metadata JSON (server-metadata-<port>.json, with "
                                                        "policy_seed)")
    s.add_argument("--policy-seed", type=int, required=True,
                   help="model seed (required): reseed(n) replaces the legacy constant 0, once after loading and once "
                        "before each episode's new_episode")
    s.add_argument("--det", action="store_true",
                   help="deterministic mode: torch.use_deterministic_algorithms(True) + "
                        "CUBLAS_WORKSPACE_CONFIG=:4096:8 (off by default)")
    s.set_defaults(func=cmd_serve)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
