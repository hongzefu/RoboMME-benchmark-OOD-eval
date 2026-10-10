"""站点测试的公共世界：合成站点目录（目录 + 媒体白名单 + mp4）、进程内线程服务、由包内真实 V9 规格切出的小规格根。

- 服务：真实 ``site_app.create_server``（即 ``site_server.create_server`` + 真实 ``site.html``），绑 ``127.0.0.1``、
  端口 0，跑在守护线程里，用 ``http.client`` 发原始请求（可带任意 Range 与未规范化路径）。
- 规格：从 benchmark 子模块 ``robomme_ood/env_metadata/ood/<tier>/specs.jsonl`` 读真实行，只保留要测的任务，用生产的
  ``identity_sha256``／``delivery_sha256``／``digest`` 重签头部——行内容（seed、规格、选中与 rollout）原样保留，
  站点目录的配置抽值因此对的是真实交付数据。
"""
from __future__ import annotations

import contextlib
import copy
import http.client
import json
import threading
from pathlib import Path

import h5py
import numpy as np

from tests._support.dev_loaders import load_script

APP = load_script("injection-dev/site/site_app.py")
SERVER = APP.site_server
CAT = load_script("injection-dev/site/site_catalog.py")
H = CAT.load_hard_specs()
PACKAGED = CAT.robomme_ood_root() / "env_metadata/ood"
HTML = Path(APP.HTML_PATH)

MP4 = bytes(range(256)) * 8  # 2048 字节，逐字节可辨


# ── 合成站点 + 线程服务 ───────────────────────────────────────────────


def make_site(root: Path, n_media: int = 2) -> dict:
    """``root`` 即白名单根：``root/site/{catalog,media-private}.json``、``root/media/v<i>.mp4``、``root/site/posters``。"""
    site = root / "site"
    media_dir = root / "media"
    site.mkdir(parents=True)
    media_dir.mkdir()
    (site / "posters").mkdir()
    mapping = {}
    for i in range(n_media):
        path = media_dir / f"v{i}.mp4"
        path.write_bytes(MP4[i:] + MP4[:i])
        mapping[f"id{i}"] = str(path)
        (site / "posters" / f"id{i}.jpg").write_bytes(b"\xff\xd8poster" + bytes([i]))
    (site / "media-private.json").write_text(json.dumps(mapping), encoding="utf-8")
    (site / "catalog.json").write_text(json.dumps({"schema": "t7", "tasks": []}), encoding="utf-8")
    return {"root": root, "site": site, "media": mapping}


@contextlib.contextmanager
def serve(site: Path, media_root: Path):
    server = APP.create_server("127.0.0.1", 0, site, media_root)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def request(server, method: str, path: str, headers: dict | None = None):
    """原始 HTTP 请求（路径不经规范化）；返回 ``(status, headers, body)``。"""
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        conn.putrequest(method, path, skip_accept_encoding=True)
        for name, value in (headers or {}).items():
            if isinstance(value, list):
                for v in value:
                    conn.putheader(name, v)
            else:
                conn.putheader(name, value)
        conn.endheaders()
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body
    finally:
        conn.close()


def write_h5(path: Path, subgoals: list[str], demo: int, goal: list[str] | None = None) -> Path:
    """录像器同形态的最小 h5：逐帧 ``info/simple_subgoal`` 与 ``info/is_video_demo``（前 ``demo`` 帧为演示）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("episode_0")
        for k, text in enumerate(subgoals):
            info = g.create_group(f"timestep_{k}/info")
            info.create_dataset("is_video_demo", data=np.bool_(k < demo))
            info.create_dataset("simple_subgoal", data=text.encode())
        if goal is not None:
            setup = g.create_group("setup")
            setup.create_dataset("task_goal", data=[x.encode() for x in goal])
            setup.create_dataset("difficulty", data=b"t7")
    return path


#: 合成局的逐帧 subgoal：7 帧、前 2 帧演示 → 执行 5 步；段为 a×3、b×2、a×2（a 第二次出现记「后续」）
SUBGOALS = ["pick up the red cube"] * 3 + ["press the button"] * 2 + ["pick up the blue cube"] * 2
DEMO = 2


# ── 由包内真实规格切小规格根 ──────────────────────────────────────────


def packaged(tier: str) -> tuple[dict, list[dict]]:
    lines = (PACKAGED / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]), [json.loads(x) for x in lines[1:] if x.strip()]


def resign(header: dict, rows: list[dict]) -> dict:
    header = copy.deepcopy(header)
    header["sampling_config_sha256"] = H.digest(header["sampling_config"])
    header["identity_sha256"] = H.identity_sha256(header, rows)
    header["delivery_sha256"] = H.delivery_sha256(rows)
    return header


def subset_root(root: Path, cells: dict[tuple[str, str], int], edit=None) -> dict[str, list[dict]]:
    """``cells`` 的格必须取包内该格的全部配额（行原样）。``edit(rows)`` 可就地改行（造负例），之后重签。
    返回 ``{tier: rows}``。"""
    out = {}
    for tier in [t for t in H.TIERS if any(tt == t for _, tt in cells)]:
        header, rows = packaged(tier)
        tasks = [t for t in header["tasks"] if (t, tier) in cells]
        for t in tasks:
            assert header["delivery_per_cell"][t] == cells[(t, tier)], "子表只取整格"
        rows = [r for r in rows if r["task"] in tasks]
        if edit is not None:
            edit(tier, rows)
            for r in rows:
                r["spec_sha256"] = H.spec_sha256(r["spec"])
        sub = copy.deepcopy(header)
        sub["tasks"] = tasks
        for key in ("per_env", "select_rule", "delivery_per_cell"):
            sub[key] = {t: header[key][t] for t in tasks}
        sub["sampling_config"] = {t: header["sampling_config"][t] for t in tasks}
        sub = resign(sub, rows)
        path = Path(root) / tier / "specs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(H.canonical_json(r) + "\n" for r in [sub, *rows]), encoding="utf-8")
        out[tier] = rows
    return out


def catalog_inputs(root: Path, cells: dict[tuple[str, str], int], *, edit=None) -> dict:
    """规格根 + delivery.json + 身份清单 + xhard0 两侧清单 + 共享 mp4／h5，返回 ``build_catalog`` 的 ``src``。"""
    root = Path(root)
    by_tier = subset_root(root / "specs", cells, edit)
    video = root / "gen.mp4"
    video.write_bytes(MP4)
    h5 = write_h5(root / "gen.h5", SUBGOALS, DEMO, goal=["pick up the cubes"])
    frames, demo = len(SUBGOALS), DEMO
    delivery, idents = [], []
    for tier, rows in by_tier.items():
        for r in rows:
            if H.delivered(r):
                delivery.append({"task": r["task"], "tier": tier, "episode": r["episode"], "candidate": r["candidate"],
                                 "seed": r["seed"], "exec_steps": frames - demo, "frames": frames, "h5": str(h5),
                                 "video": str(video)})
                idents.append({"task": r["task"], "tier": tier, "seed": r["seed"], "candidate": r["candidate"],
                               "episode": len(idents)})
    x0 = []
    for task in H.ALL_TASKS:
        for k in range(H.XHARD0_PER_TASK):
            seed = 900_000 + H.ALL_TASKS.index(task) * 100 + k
            idents.append({"task": task, "tier": "xhard0", "seed": seed, "source_episode": k})
            x0.append({"task": task, "seed": seed, "frames": frames, "demo_frames": demo, "h5_sha256": "ab" * 32,
                       "mp4": str(video), "h5": str(h5)})
    (root / "x0").mkdir(exist_ok=True)
    for side in ("H", "O"):
        (root / "x0" / f"manifest-{side}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in x0))
    (root / "delivery.json").write_text(json.dumps({"schema": CAT.DELIVERY_SCHEMA, "rows": delivery}))
    (root / "ids.jsonl").write_text("".join(json.dumps(r) + "\n" for r in idents))
    cells_json = root / "cells.json"
    cells_json.write_text(json.dumps({f"{t}/{tier}": n for (t, tier), n in cells.items()}))
    return {"specs_root": root / "specs", "delivery": root / "delivery.json", "identities": root / "ids.jsonl",
            "xhard0_gen": root / "x0", "gen_videos": None, "path_base": root, "cells_json": cells_json,
            "cells": "v9", "eval_reuse": None, "reused": None, "eval_new": None}
