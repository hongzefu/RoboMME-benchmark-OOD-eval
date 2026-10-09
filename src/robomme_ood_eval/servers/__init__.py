"""四个普通模型（FrameSamp+Modulation、GroundSG、SimpleMemVLA、PonderPounce）在客户端侧起服务端的共用件。

本包里的 ``policy_server_wrap.py``、``smvla_server.py``、``pp_server_wrap.py`` 由各模型自己 venv 的解释器**按文件路径**
运行（不经过本 ``__init__``）；本文件只给客户端进程里的 Policy 子类用，内容抽自旧 ``run_seat.sh``：

* 路径与缺省值：评估仓根 ``repo_root``、各服务端解释器 ``interpreter``、缺省 ckpt（``DEFAULT_CKPTS``，照旧
  ``run_seat.sh`` 顶部）、缺省端口基数（``DEFAULT_PORT_BASE``，照旧 ``run_policy`` 的 ``18000 + 10 × 策略号``）；
* ``CleanServerProcess``：在 ``ServerProcess`` 之上先去掉影响确定性／编译缓存的变量（旧 ``CLEAN_ENV``），再加本模型要的；
* 起服务前的闸门（旧同名 shell 函数的 Python 版，失败抛 ``PreflightError``，文字以 ``RUN_BLOCKED reason=`` 开头）：
  ``tokenizer_gate``、``preflight_mme_vla``、``preflight_pp``、``variant_pairing``；
* 起服务后的核对：``check_server_log``（MME-VLA 服务日志须含 ``history_config='<yaml>'``）、``check_wrap_metadata``
  （服务外壳自写的元数据里 ``policy_seed`` 须等于本次种子）；
* ``ckpt_fingerprint``：逐文件 sha256 汇成一个指纹（旧 ``ckpt_fingerprint``，后台线程跑、只写一行文本）；
* ``ServedPolicy``：四个 Policy 子类的共同基类（选端口、起或 attach 服务端、拼旧 ``conn_info``、整理 ``PlayOutput``）；
  ``SessionNoClose``：交给旧客户端循环的会话代理，挡掉模型侧的 ``session.close()``（环境只由外层关）。

服务外壳自写的元数据一律落 ``<server_dir>/server-wrap-metadata-<port>.json``，与 ``ServerProcess`` 自己的
``server-metadata-<port>.json``（attach 与 ``--stop-server`` 用）分开，互不覆盖。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from robomme_ood_eval.policy import Policy, ServerMismatch, ServerProcess, pick_port

#: 本目录（三个服务外壳脚本所在处）
SERVERS_DIR = Path(__file__).resolve().parent
#: 评估仓根（本文件在 src/robomme_ood_eval/servers/ 下）
REPO = Path(__file__).resolve().parents[3]
#: 起服务端前一律去掉的变量（旧 run_seat.sh 的 CLEAN_ENV）
CLEAN_ENV = ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
             "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "CUBLAS_WORKSPACE_CONFIG")
#: mme-vla 子模块须钉在的提交（旧 MME_VLA_COMMIT）
MME_VLA_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
#: tokenizer 在 OPENPI_DATA_HOME 下的相对路径（旧 TOKENIZER_REL）
TOKENIZER_REL = "big_vision/paligemma_tokenizer.model"
#: 各模型服务日志里必须出现的 history_config（旧 *_YAML_EXPECT）
YAML_EXPECT = {"perceptual-framesamp-modul": "perceptual-framesamp-modul.yaml",
               "groundsg": "symbolic-grounded-subgoal.yaml"}
#: 缺省 ckpt（照旧 run_seat.sh 顶部；PonderPounce 没有缺省，必须显式给）
DEFAULT_CKPTS = {
    "perceptual-framesamp-modul": "/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/"
                                  "official-mme-vla/perceptual-framesamp-modul/79999",
    "groundsg": "/data/hongzefu/robomme_policy_learning-vqa-test/runs/ckpts/mme_vla_suite/"
                "symbolic-grounded-subgoal/79999",
    "smvla": "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/checkpoints/simplememvla_robomme",
    "pp": None,
}
#: 缺省端口基数（旧 run_policy：18000 + 10 × 策略号，smvla=0、perceptual-framesamp-modul=1、groundsg=2、pp=3）
DEFAULT_PORT_BASE = {"smvla": 18000, "perceptual-framesamp-modul": 18010, "groundsg": 18020, "pp": 18030}
#: 旧缺省：服务端就绪等待上限（秒）
DEFAULT_READY_TIMEOUT_S = 1200.0
#: MME-VLA 服务端 XLA 显存占比缺省（旧 SEAT_XLA_MEM_FRACTION）
DEFAULT_XLA_MEM_FRACTION = "0.75"
#: GroundSG 三个变体（与 models/_official_defs.VARIANTS 相同，这里不导入以免拖进官方摘取逻辑）
GROUNDSG_VARIANTS = ("ground-sg-oracle", "ground-sg-qwenvl", "ground-sg-memer")


class PreflightError(RuntimeError):
    """起服务前的闸门不过（文字以 ``RUN_BLOCKED reason=`` 开头）；``load()`` 抛出即整批停。"""


# ── 路径与参数 ───────────────────────────────────────────────────────────────


def repo_root(cfg: dict) -> Path:
    """评估仓根：``cfg["repo_root"]`` → 环境变量 ``ROBOMME_EVAL_ROOT`` → 本包所在仓。"""
    v = cfg.get("repo_root") or os.environ.get("ROBOMME_EVAL_ROOT")
    return Path(v).resolve() if v else REPO


def interpreter(cfg: dict, key: str, env_name: str, default: Path) -> Path:
    """服务端解释器：``cfg[key]`` → 环境变量 ``env_name`` → 缺省路径（与旧 run_seat.sh 取法相同）。"""
    v = cfg.get(key) or os.environ.get(env_name)
    return Path(v) if v else Path(default)


def server_dir(cfg: dict) -> Path:
    """服务元数据与服务日志目录：``cfg["server_dir"]`` → ``<work_dir>/servers`` → 系统临时目录下按用户分的目录。"""
    if cfg.get("server_dir"):
        return Path(cfg["server_dir"])
    if cfg.get("work_dir"):
        return Path(cfg["work_dir"]) / "servers"
    return Path(tempfile.gettempdir()) / f"robomme-eval-servers-{os.getuid()}"


def gpu_of(cfg: dict) -> str | None:
    """服务端所在 GPU：``cfg["gpus"]`` 的第一张（``--gpus 0,1``）→ ``cfg["gpu"]`` → None（沿用调用方的
    ``CUDA_VISIBLE_DEVICES``，GL 席位由 srun 设好）。"""
    gpus = cfg.get("gpus")
    if isinstance(gpus, (list, tuple)) and gpus:
        return str(gpus[0])
    if isinstance(gpus, (int, str)) and str(gpus).strip():
        return str(gpus).split(",")[0].strip()
    g = cfg.get("gpu")
    return None if g is None else str(g)


def choose_port(cfg: dict, model: str, metadata_dir: Path) -> int:
    """服务端口：``cfg["port"]`` 显式给出即用它；否则从 ``cfg["port_base"]``（缺省 ``DEFAULT_PORT_BASE``）起——
    该端口已有元数据文件（上一轮看门狗退出留下的服务端）就沿用它，交给 ``ServerProcess.start`` 去 attach／拒接；
    否则按 ``pick_port`` 找本端口与 +1 都空闲的端口。"""
    if cfg.get("port") is not None:
        return int(cfg["port"])
    base = int(cfg.get("port_base") or DEFAULT_PORT_BASE[model])
    if (Path(metadata_dir) / f"server-metadata-{base}.json").is_file():
        return base
    return pick_port(base)


def flag_on(v: Any) -> bool:
    """开关参数（``on``／``1``／``true``／``yes`` 或真值）。"""
    if isinstance(v, str):
        return v.strip().lower() in ("on", "1", "true", "yes")
    return bool(v)


def gpu_slug(gpu: str | None) -> str:
    """GPU 型号做目录名（旧 gpu_slug）：``nvidia-smi`` 不可用时为 ``unknown``。"""
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"] + (["-i", str(gpu)] if gpu is not None else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (out[0].strip().replace(" ", "_").replace("/", "_") if out else "") or "unknown"


# ── 服务端进程 ───────────────────────────────────────────────────────────────


class CleanServerProcess(ServerProcess):
    """``ServerProcess``：子进程环境先去掉 ``CLEAN_ENV``（旧 ``env -u …``），再叠加本模型要的变量。"""

    unset_vars = CLEAN_ENV

    def _child_env(self) -> dict:
        env = super()._child_env()
        for k in self.unset_vars:
            if k not in self.env:
                env.pop(k, None)
        return env


def check_server_log(srv: Any, expect_yaml: str, *, grace_s: float = 30.0, poll_s: float = 0.5) -> str:
    """MME-VLA 服务就绪后核日志含 ``history_config='<expect_yaml>'``（旧 start_server 的 SERVER_CONFIG 核对）。
    配置日志在加载权重时打印、早于端口监听，这里只给 ``grace_s`` 秒容错；不含即停服务并抛 ``PreflightError``。"""
    needle = f"history_config='{expect_yaml}'"
    deadline = time.monotonic() + float(grace_s)
    while True:
        try:
            text = Path(srv.log_path).read_text(errors="replace")
        except OSError:
            text = ""
        if needle in text:
            line = f"SERVER_CONFIG=PASS name={srv.name} history_config={expect_yaml} seed={srv.policy_seed}"
            print(line, flush=True)
            return line
        if time.monotonic() >= deadline:
            srv.stop()
            raise PreflightError(f"RUN_BLOCKED reason=server_config（服务日志里没有 {needle}）log={srv.log_path}")
        time.sleep(poll_s)


def check_wrap_metadata(path: Path, policy_seed: int) -> dict | None:
    """服务外壳自写的元数据（``--sgeval-metadata-out`` 或 ``--metadata_out``）里 ``policy_seed`` 必须等于本次种子；
    文件不存在（attach 到旧服务、或外壳关）返回 None，不符抛 ``ServerMismatch``。"""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ServerMismatch(f"服务外壳元数据读不出：{p}：{e}") from e
    if doc.get("policy_seed") != int(policy_seed):
        print(f"RUN_BLOCKED reason=server_mismatch detail=wrap_metadata policy_seed={doc.get('policy_seed')!r} "
              f"want={int(policy_seed)} metadata={p}", flush=True)
        raise ServerMismatch(f"服务外壳元数据 policy_seed={doc.get('policy_seed')!r} 与本次 {int(policy_seed)} 不符：{p}")
    return doc


# ── 起服务前的闸门 ───────────────────────────────────────────────────────────


def tokenizer_gate(openpi_home: Any, expected_sha: Any) -> str:
    """MME-VLA 服务起之前核 ``OPENPI_DATA_HOME`` 下 tokenizer 的 sha256（不现场下载顶替）。"""
    if not openpi_home or not expected_sha:
        raise PreflightError("RUN_BLOCKED reason=tokenizer_sha detail=missing_args（须给 openpi_data_home 与 "
                             "tokenizer_sha256）")
    f = Path(openpi_home) / TOKENIZER_REL
    if not f.is_file():
        raise PreflightError(f"RUN_BLOCKED reason=tokenizer_sha detail=file_missing file={f}")
    actual = file_sha256(f)
    if actual.lower() != str(expected_sha).lower():
        raise PreflightError(f"RUN_BLOCKED reason=tokenizer_sha expected={str(expected_sha).lower()} actual={actual} "
                             f"file={f}")
    line = f"TOKENIZER_SHA=PASS sha256={actual} file={f} openpi_data_home={openpi_home}"
    print(line, flush=True)
    return line


def _git(sub: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(sub), *args], capture_output=True, text=True, timeout=60)


def preflight_mme_vla(root: Path, model: str, ckpt: Path, py: Path, *, commit: str = MME_VLA_COMMIT) -> str:
    """旧 ``preflight_mme_vla``：子模块提交与干净、嵌套子模块为空、ckpt 旁 ``history_config.txt`` 等于期望 yaml、
    yaml 存在、ckpt 有 ``params``／``assets``、解释器可执行、服务外壳存在。"""
    sub = Path(root) / "third_party" / "mme-vla"
    expect = YAML_EXPECT[model]
    head = _git(sub, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise PreflightError(f"RUN_BLOCKED reason=mme_vla_commit head={head or 'none'} expect={commit}")
    if _git(sub, "diff", "--quiet", "HEAD", "--", "src", "scripts", "packages", "pyproject.toml",
            "uv.lock").returncode != 0:
        raise PreflightError("RUN_BLOCKED reason=mme_vla_dirty")
    nested = sub / "third_party" / "robomme_benchmark"
    if nested.is_dir() and any(nested.iterdir()):
        raise PreflightError("RUN_BLOCKED reason=nested_submodule_not_empty")
    ckpt = Path(ckpt)
    try:
        hc = (ckpt / ".." / "history_config.txt").read_text(encoding="utf-8").strip()
    except OSError:
        hc = ""
    if hc != expect:
        raise PreflightError(f"RUN_BLOCKED reason=history_config value='{hc}' expect='{expect}' policy={model}")
    if not (sub / "src" / "mme_vla_suite" / "models" / "config" / "robomme" / hc).is_file():
        raise PreflightError(f"RUN_BLOCKED reason=yaml_missing {hc}")
    if not ((ckpt / "params").is_dir() and (ckpt / "assets").is_dir()):
        raise PreflightError(f"RUN_BLOCKED reason=ckpt_layout {ckpt}")
    if not (Path(py).is_file() and os.access(py, os.X_OK)):
        raise PreflightError(f"RUN_BLOCKED reason=mme_vla_venv_missing {py}")
    wrap = SERVERS_DIR / "policy_server_wrap.py"
    if not wrap.is_file():
        raise PreflightError(f"RUN_BLOCKED reason=policy_server_wrap_missing path={wrap}")
    line = f"MME_VLA_PREFLIGHT=PASS policy={model} commit={head} history_config={hc} ckpt={ckpt} py={py}"
    print(line, flush=True)
    return line


def preflight_pp(root: Path, ckpt: Any, py: Path, *, server_wrap: bool, seed: int) -> str:
    """旧 ``preflight_pp``：子模块目录在、解释器可执行、ckpt 目录在且有 ``norm_stats.json``、外壳打开时外壳在。"""
    sub = Path(root) / "third_party" / "PonderPounce"
    if not sub.is_dir():
        raise PreflightError(f"RUN_BLOCKED reason=pp_submodule_missing dir={sub}")
    if not (Path(py).is_file() and os.access(py, os.X_OK)):
        raise PreflightError(f"RUN_BLOCKED reason=pp_venv_missing {py}")
    if not ckpt or not Path(ckpt).is_dir():
        raise PreflightError(f"RUN_BLOCKED reason=pp_ckpt_missing ckpt={ckpt or 'unset'}")
    if not (Path(ckpt) / "norm_stats.json").is_file():
        raise PreflightError(f"RUN_BLOCKED reason=pp_ckpt_layout ckpt={ckpt}（缺 norm_stats.json）")
    wrap = SERVERS_DIR / "pp_server_wrap.py"
    if server_wrap and not wrap.is_file():
        raise PreflightError(f"RUN_BLOCKED reason=pp_server_wrap_missing path={wrap}")
    line = (f"PP_PREFLIGHT=PASS ckpt={ckpt} py={py} seed={seed} hf_home={os.environ.get('HF_HOME', 'unset')} "
            f"server_wrap={int(bool(server_wrap))}")
    print(line, flush=True)
    return line


def variant_pairing(variant: Any, qwenvl_adapter: Any, memer_adapter: Any) -> str:
    """旧 ``variant_pairing``（只看 groundsg）：变体与 QwenVL／MemER adapter 的配对。"""
    why = None
    if variant == "ground-sg-oracle":
        if qwenvl_adapter:
            why = "adapter_without_qwenvl"
        elif memer_adapter:
            why = "memer_adapter_without_memer"
    elif variant == "ground-sg-qwenvl":
        if not qwenvl_adapter:
            why = "qwenvl_needs_adapter"
        elif memer_adapter:
            why = "memer_adapter_without_memer"
        elif not Path(qwenvl_adapter).is_dir():
            why = f"adapter_missing path={qwenvl_adapter}"
    elif variant == "ground-sg-memer":
        if not memer_adapter:
            why = "memer_needs_adapter"
        elif qwenvl_adapter:
            why = "adapter_without_qwenvl"
        elif not Path(memer_adapter).is_dir():
            why = f"memer_adapter_missing path={memer_adapter}"
    elif not variant:
        why = "groundsg_needs_variant"
    else:
        why = f"unknown_variant variant={variant}"
    if why:
        raise PreflightError(f"RUN_BLOCKED reason=variant_pairing variant={variant or 'none'} detail={why}")
    line = (f"VARIANT_PAIRING=PASS variant={variant} adapter={qwenvl_adapter or 'none'} "
            f"memer_adapter={memer_adapter or 'none'}")
    print(line, flush=True)
    return line


# ── ckpt 指纹 ────────────────────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_fingerprint(root: Any) -> str:
    """旧 ``ckpt_fingerprint``：逐文件「相对路径 \\t 字节数 \\t sha256」汇成一个 sha256，返回一行判定文本。"""
    t0 = time.time()
    root = Path(root).resolve()
    h = hashlib.sha256()
    n = b = 0
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        size = p.stat().st_size
        h.update(f"{p.relative_to(root)}\t{size}\t{file_sha256(p)}\n".encode())
        n += 1
        b += size
    return f"CKPT_FINGERPRINT dir={root} sha256={h.hexdigest()} files={n} bytes={b} secs={time.time() - t0:.1f}"


def start_ckpt_fingerprint(root: Any, out: Path) -> threading.Thread | None:
    """后台线程算 ckpt 指纹并写到 ``out``（与旧 run_seat.sh 一样不阻塞起服务；失败只写原因）。"""
    if not root or not Path(root).is_dir():
        return None

    def work():
        try:
            line = ckpt_fingerprint(root)
        except Exception as e:  # noqa: BLE001 指纹失败不影响评估
            line = f"CKPT_FINGERPRINT=ERROR dir={root} {type(e).__name__}: {e}"
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(line + "\n", encoding="utf-8")
        except OSError:
            pass

    t = threading.Thread(target=work, name="ckpt-fingerprint", daemon=True)
    t.start()
    return t


# ── 四个普通模型 Policy 的共同基类 ─────────────────────────────────────────


class ServedPolicy(Policy):
    """带一个本机服务端的 Policy 基类（四个普通模型共用；Astra 不用）。

    子类在 ``load()`` 里拼好命令后调 ``_launch``（起或 attach 服务端），在 ``play()`` 里用 ``_conn_info(spec)`` 拼出
    旧 ``conn_info``（服务地址、步数上限、身份、轨迹落点）交给模型模块原有的 ``run_episode``，再经 ``_finish_play``
    整理成 ``PlayOutput``。服务端的停止由基类 ``Policy.close()`` 统一做。

    公共 cfg：``repo_root``、``server_dir``／``work_dir``、``port``／``port_base``、``gpus``／``gpu``、
    ``ready_timeout_s``（缺省 1200）、``preflight``（缺省开；单测关）、``ckpt_fingerprint``（缺省开，后台线程）。"""

    model = "served"
    host = "127.0.0.1"

    def __init__(self, policy_seed: int, **cfg: Any):
        super().__init__(policy_seed, **cfg)
        if self.policy_seed < 0:
            raise PreflightError(f"RUN_BLOCKED reason=policy_seed detail=policy_seed 须为非负整数（现为 {policy_seed!r}）")
        self.root = repo_root(self.cfg)
        self.metadata_dir = server_dir(self.cfg)
        self.port: int | None = None
        self.wrap_meta: Path | None = None
        self.load_info: dict[str, Any] = {}
        self._load_reported = False

    # 起服务端
    @property
    def preflight(self) -> bool:
        return flag_on(self.cfg.get("preflight", True))

    def _pick_port(self) -> int:
        self.port = choose_port(self.cfg, self.model, self.metadata_dir)
        self.wrap_meta = self.metadata_dir / f"server-wrap-metadata-{self.port}.json"
        return self.port

    def _launch(self, argv: list, env: dict, cwd: Path, ready, ckpt: Any) -> ServerProcess:
        """建 ``CleanServerProcess`` 并 ``start()``（有本端口元数据且进程在 → attach／拒接；否则新起、等就绪）。"""
        srv = CleanServerProcess(argv, env, cwd, gpu_of(self.cfg), ready, port=int(self.port),
                                 metadata_dir=self.metadata_dir, policy_seed=self.policy_seed,
                                 ckpt=None if ckpt is None else str(ckpt), name=str(self.label),
                                 ready_timeout_s=float(self.cfg.get("ready_timeout_s", DEFAULT_READY_TIMEOUT_S)))
        if self.wrap_meta is not None and not srv.metadata_path.is_file():
            self.wrap_meta.unlink(missing_ok=True)  # 新起：先删上一轮外壳元数据，免得读到旧种子
        self.server = srv
        t0 = time.monotonic()
        srv.start()
        self.load_info.update(server_ready_s=round(time.monotonic() - t0, 3), server_attached=bool(srv.attached),
                              port=int(self.port))
        if self.wrap_meta is not None:
            check_wrap_metadata(self.wrap_meta, self.policy_seed)
        return srv

    def _fingerprint(self, ckpt: Any) -> None:
        if flag_on(self.cfg.get("ckpt_fingerprint", True)):
            out = self.metadata_dir / f"ckpt-fingerprint-{self.label}.txt"
            start_ckpt_fingerprint(ckpt, out)
            self.load_info["ckpt_fingerprint_file"] = str(out)

    # 每局
    def _conn_info(self, spec) -> dict:
        """旧 ``env_client`` 交给模型 ``run_episode`` 的 ``conn_info``，改由 ``EpisodeSpec`` 与实例属性拼出。

        轨迹落点固定为本局 raw 目录下的 ``trace.jsonl``（``trace_path``）；``episode_tag`` 为 ``<key>.a<attempt>``。"""
        return {"host": self.host, "port": int(self.port), "max_steps": int(spec.max_steps), "policy": self.model,
                "dataset": spec.dataset, "strict_cap": bool(spec.strict_cap), "policy_seed": int(self.policy_seed),
                "effective_cap": int(spec.max_steps) if spec.strict_cap else None,
                "trace_path": str(Path(spec.out_dir) / "trace.jsonl"), "trace_dir": None,
                "episode_tag": f"{spec.key}.a{int(spec.attempt)}", "attempt": int(spec.attempt),
                "rec_dir": str(spec.out_dir)}

    def _finish_play(self, res: dict) -> dict:
        """模型 ``run_episode`` 的返回 → ``PlayOutput``：``task_success`` 改 0/1、必有字段补齐；进程内第一局的
        ``timing`` 另带 ``policy_load``（服务端就绪、预热耗时）。"""
        out = dict(res)
        status = out.get("status", "error")
        out["status"] = status
        out["task_success"] = int(status == "success")
        out["steps"] = int(out.get("steps") or 0)
        out.setdefault("error", None)
        out["infra"] = bool(out.get("infra"))
        out.setdefault("infra_reason", None)
        if not self._load_reported:
            timing = dict(out.get("timing") or {})
            timing["policy_load"] = dict(self.load_info)
            out["timing"] = timing
            self._load_reported = True
        return out


class SessionNoClose:
    """交给旧客户端循环的会话代理：``close()`` 不关环境（环境只由外层关，S1），其余属性与方法原样转发。

    SimpleMemVLA 照抄旧官方 ``SimEnvService.reset`` 在 reset 失败后 ``env.close()``；新接口下环境归外层管，故挡掉。"""

    def __init__(self, session: Any):
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "close_calls", 0)

    def close(self) -> None:
        object.__setattr__(self, "close_calls", self.close_calls + 1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


__all__ = ["SERVERS_DIR", "REPO", "CLEAN_ENV", "MME_VLA_COMMIT", "TOKENIZER_REL", "YAML_EXPECT", "DEFAULT_CKPTS",
           "DEFAULT_PORT_BASE", "DEFAULT_READY_TIMEOUT_S", "GROUNDSG_VARIANTS", "PreflightError", "CleanServerProcess",
           "repo_root", "interpreter", "server_dir", "gpu_of", "choose_port", "flag_on", "gpu_slug",
           "check_server_log", "check_wrap_metadata", "tokenizer_gate", "preflight_mme_vla", "preflight_pp",
           "variant_pairing", "ckpt_fingerprint", "start_ckpt_fingerprint", "file_sha256", "ServedPolicy",
           "SessionNoClose"]
