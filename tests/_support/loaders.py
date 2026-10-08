"""按文件路径加载脚本模块（scripts/、dev-scripts/ 不是包，生产代码也按路径互相加载）；旧仓路径经 LEGACY_PATHS 映射。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
_CACHE: dict[str, object] = {}

# 拆仓映射表（1008 拆分方案第二部分 §二）：键是旧仓 scripts/ 下的相对路径，值是本仓相对仓根的新路径。
# 测试里约 90 处 load_script("eval-official/…") 不逐个改，统一经这张表落到新位置；表外的 rel 先按本仓 scripts/
# 解析，再按 dev-scripts/ 解析。旧仓 run_seat.sh、run_astra.sh 不迁（职责已进 ServerProcess、seat.py 与 models/astra.py）。
LEGACY_PATHS: dict[str, str] = {
    "eval-official/astra_cost_guard.py": "src/robomme_hard_eval/servers/astra_cost_guard.py",
    "eval-official/astra_hard_runner.py": "src/robomme_hard_eval/models/astra.py",
    "eval-official/budget_ledger.py": "dev-scripts/gl/budget_ledger.py",
    "eval-official/cap_probe.py": "dev-scripts/checks/cap_probe.py",
    "eval-official/client_replay_eq.py": "dev-scripts/checks/client_replay_eq.py",
    "eval-official/env_client.py": "dev-scripts/gl/seat.py",
    "eval-official/eval_manifest.py": "dev-scripts/gl/eval_manifest.py",
    "eval-official/eval_report.py": "dev-scripts/gl/eval_report.py",
    "eval-official/framesamp_modul_client.py": "src/robomme_hard_eval/models/framesamp_modul.py",
    "eval-official/gate2_compare.py": "dev-scripts/checks/gate2_compare.py",
    "eval-official/groundsg_client.py": "src/robomme_hard_eval/models/groundsg.py",
    "eval-official/lang_io_check.py": "dev-scripts/checks/lang_io_check.py",
    "eval-official/model_eval_report.py": "dev-scripts/checks/model_eval_report.py",
    "eval-official/official_defs.py": "src/robomme_hard_eval/models/_official_defs.py",
    "eval-official/official_hard_runner.py": "dev-scripts/orig/official_hard_runner.py",
    "eval-official/official_media_check.py": "dev-scripts/media/official_media_check.py",
    "eval-official/orig_observer/_obs_common.py": "dev-scripts/orig/orig_observer/_obs_common.py",
    "eval-official/orig_observer/framesamp_modul_client_wrap.py": "dev-scripts/orig/orig_observer/framesamp_modul_client_wrap.py",
    "eval-official/orig_observer/framesamp_modul_proxy.py": "dev-scripts/orig/orig_observer/framesamp_modul_proxy.py",
    "eval-official/orig_observer/observer_status.py": "dev-scripts/orig/orig_observer/observer_status.py",
    "eval-official/orig_observer/official_rerun_shard.sh": "dev-scripts/orig/orig_observer/official_rerun_shard.sh",
    "eval-official/orig_observer/orig_budget.py": "dev-scripts/orig/orig_observer/orig_budget.py",
    "eval-official/orig_observer/orig_episode.py": "dev-scripts/orig/orig_observer/orig_episode.py",
    "eval-official/orig_observer/orig_observer_lib.sh": "dev-scripts/orig/orig_observer/orig_observer_lib.sh",
    "eval-official/orig_observer/orig_results_adapter.py": "dev-scripts/orig/orig_observer/orig_results_adapter.py",
    "eval-official/orig_observer/run_orig_framesamp_modul.sh": "dev-scripts/orig/orig_observer/run_orig_framesamp_modul.sh",
    "eval-official/orig_observer/run_orig_smvla.sh": "dev-scripts/orig/orig_observer/run_orig_smvla.sh",
    "eval-official/orig_observer/smvla_wrap.py": "dev-scripts/orig/orig_observer/smvla_wrap.py",
    "eval-official/orig_observer/step_arrays.py": "dev-scripts/orig/orig_observer/step_arrays.py",
    "eval-official/orig_observer/transparency_check.py": "dev-scripts/orig/orig_observer/transparency_check.py",
    "eval-official/pair_seat.sh": "dev-scripts/orig/pair_seat.sh",
    "eval-official/policy_server_wrap.py": "src/robomme_hard_eval/servers/policy_server_wrap.py",
    "eval-official/pp_client.py": "src/robomme_hard_eval/models/pp.py",
    "eval-official/pp_official_runner.py": "dev-scripts/orig/pp_official_runner.py",
    "eval-official/pp_server_wrap.py": "src/robomme_hard_eval/servers/pp_server_wrap.py",
    "eval-official/publish_videos.py": "dev-scripts/gl/publish_videos.py",
    "eval-official/recorder.py": "src/robomme_hard_eval/record/recorder.py",
    "eval-official/render_official_video.py": "dev-scripts/media/render_official_video.py",
    "eval-official/resource_probe.py": "dev-scripts/checks/resource_probe.py",
    "eval-official/run_eval_gl.sh": "dev-scripts/gl/run_eval_gl.sh",
    "eval-official/run_official_hard.sh": "dev-scripts/orig/run_official_hard.sh",
    "eval-official/seat_media_lib.sh": "dev-scripts/gl/seat_media_lib.sh",
    "eval-official/smvla_client.py": "src/robomme_hard_eval/models/smvla.py",
    "eval-official/smvla_server.py": "src/robomme_hard_eval/servers/smvla_server.py",
    "eval-official/trace_arrays_check.py": "dev-scripts/checks/trace_arrays_check.py",
    "eval-official/trace_writer.py": "src/robomme_hard_eval/record/trace_writer.py",
    "eval-official/video_check.py": "dev-scripts/checks/video_check.py",
    "injection-dev/_common.py": "dev-scripts/parity/_common.py",
    "injection-dev/_rollout.py": "dev-scripts/parity/_rollout.py",
    "injection-dev/eval_video_mover.py": "dev-scripts/gl/eval_video_mover.py",
    "injection-dev/export_eval_identities.py": "dev-scripts/gl/export_eval_identities.py",
    "injection-dev/generate_h5.py": "dev-scripts/parity/generate_h5.py",
    "injection-dev/site/eval_transcode.py": "dev-scripts/site/eval_transcode.py",
    "injection-dev/site/official_overlay.html": "dev-scripts/site/official_overlay.html",
    "injection-dev/site/official_overlay_browser_check.py": "dev-scripts/site/official_overlay_browser_check.py",
    "injection-dev/site/official_overlay_site.py": "dev-scripts/site/official_overlay_site.py",
    "injection-dev/site/oracle_browser_check.py": "dev-scripts/site/oracle_browser_check.py",
    "injection-dev/site/render_xhard0.py": "dev-scripts/site/render_xhard0.py",
    "injection-dev/site/semantic_diff.py": "dev-scripts/site/semantic_diff.py",
    "injection-dev/site/site.html": "dev-scripts/site/site.html",
    "injection-dev/site/site_app.py": "dev-scripts/site/site_app.py",
    "injection-dev/site/site_browser_check.py": "dev-scripts/site/site_browser_check.py",
    "injection-dev/site/site_catalog.py": "dev-scripts/site/site_catalog.py",
    "injection-dev/site/site_server.py": "dev-scripts/site/site_server.py",
    "injection-dev/site/stage3_eval_site.html": "dev-scripts/site/stage3_eval_site.html",
    "injection-dev/site/stage3_eval_site.py": "dev-scripts/site/stage3_eval_site.py",
    "injection-dev/site/subgoal_lengths.py": "dev-scripts/site/subgoal_lengths.py",
    "injection-dev/site_build.py": "dev-scripts/site/site_build.py",
    "parity/README.md": "dev-scripts/parity/README.md",
    "parity/gate_set.py": "dev-scripts/parity/gate_set.py",
    "parity/hard_parity.py": "dev-scripts/parity/hard_parity.py",
    "parity/hard_pull.py": "dev-scripts/parity/hard_pull.py",
    "parity/hard_regression.py": "dev-scripts/parity/hard_regression.py",
    "parity/noise_gate.py": "dev-scripts/parity/noise_gate.py",
    "parity/noise_run.py": "dev-scripts/parity/noise_run.py",
    "parity/noise_run_gl.sh": "dev-scripts/parity/noise_run_gl.sh",
    "parity/official/SOURCE.json": "dev-scripts/parity/official/SOURCE.json",
    "parity/official/scripts/data-generation/compare_joint_actions.py": "dev-scripts/parity/official/scripts/data-generation/compare_joint_actions.py",
    "parity/official/scripts/data-generation/generate_dataset.py": "dev-scripts/parity/official/scripts/data-generation/generate_dataset.py",
    "parity/official/scripts/data-generation/validate_generated_dataset_contract.py": "dev-scripts/parity/official/scripts/data-generation/validate_generated_dataset_contract.py",
    "parity/official/scripts/data-generation/write_generation_report.py": "dev-scripts/parity/official/scripts/data-generation/write_generation_report.py",
    "parity/train_split_runner.py": "dev-scripts/parity/train_split_runner.py",
    "parity/train_split_worker.py": "dev-scripts/parity/train_split_worker.py",
}


def script_path(rel: str) -> Path:
    """rel 是旧仓 scripts/ 下的相对路径（如 ``eval-official/recorder.py``），或本仓 scripts/、dev-scripts/ 下的相对路径。"""
    if rel in LEGACY_PATHS:
        p = REPO / LEGACY_PATHS[rel]
    elif (SCRIPTS / rel).is_file():
        p = SCRIPTS / rel
    else:
        p = REPO / "dev-scripts" / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_script(rel: str, *, fresh: bool = False):
    """加载脚本模块；同一路径默认复用同一模块对象，``fresh=True`` 时重新执行一份独立副本。

    模块名由相对路径派生（``_script_parity_noise_gate``），脚本目录临时加到 sys.path 头部，
    以便脚本里 ``import <同目录模块>`` 的写法照常工作。
    """
    path = script_path(rel)
    key = str(path)
    if not fresh and key in _CACHE:
        return _CACHE[key]
    name = "_script_" + rel.removesuffix(".py").replace("/", "_").replace("-", "_")
    if fresh:
        name += f"_{len(_CACHE)}_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    d = str(path.parent)
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    try:
        spec.loader.exec_module(mod)
    finally:
        if added:
            try:
                sys.path.remove(d)
            except ValueError:
                pass
    if not fresh:
        _CACHE[key] = mod
    return mod
