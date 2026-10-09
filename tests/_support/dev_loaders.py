"""dev 分支的按路径加载器：在公开版 ``loaders`` 之上补 dev-scripts/ 的拆仓映射与表外回退。

解析顺序与拆分前的 ``loaders.script_path`` 完全相同：先查合并后的 LEGACY_PATHS，再按本仓 scripts/，最后按 dev-scripts/。
模块缓存与公开版共用（``loaders.load_path``），同一路径无论经哪个加载器都得到同一个模块对象。
"""
from __future__ import annotations

from pathlib import Path

from tests._support.loaders import LEGACY_PATHS as PUBLIC_LEGACY_PATHS
from tests._support.loaders import REPO, SCRIPTS, load_path

DEV_SCRIPTS = REPO / "dev-scripts"

# 拆仓映射表中落在 dev-scripts/ 的条目（测试里约 90 处 load_script("eval-official/…") 不逐个改，统一经这张表落到新位置）。
# 旧仓 run_seat.sh、run_astra.sh 不迁（职责已进 ServerProcess、seat.py 与 models/astra.py）。
DEV_LEGACY_PATHS: dict[str, str] = {
    "eval-official/budget_ledger.py": "dev-scripts/gl/budget_ledger.py",
    "eval-official/cap_probe.py": "dev-scripts/checks/cap_probe.py",
    "eval-official/client_replay_eq.py": "dev-scripts/checks/client_replay_eq.py",
    "eval-official/env_client.py": "dev-scripts/gl/seat.py",
    "eval-official/eval_manifest.py": "dev-scripts/gl/eval_manifest.py",
    "eval-official/eval_report.py": "dev-scripts/gl/eval_report.py",
    "eval-official/gate2_compare.py": "dev-scripts/checks/gate2_compare.py",
    "eval-official/lang_io_check.py": "dev-scripts/checks/lang_io_check.py",
    "eval-official/model_eval_report.py": "dev-scripts/checks/model_eval_report.py",
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
    "eval-official/pp_official_runner.py": "dev-scripts/orig/pp_official_runner.py",
    "eval-official/publish_videos.py": "dev-scripts/gl/publish_videos.py",
    "eval-official/render_official_video.py": "dev-scripts/media/render_official_video.py",
    "eval-official/resource_probe.py": "dev-scripts/checks/resource_probe.py",
    "eval-official/run_eval_gl.sh": "dev-scripts/gl/run_eval_gl.sh",
    "eval-official/run_official_hard.sh": "dev-scripts/orig/run_official_hard.sh",
    "eval-official/seat_media_lib.sh": "dev-scripts/gl/seat_media_lib.sh",
    "eval-official/trace_arrays_check.py": "dev-scripts/checks/trace_arrays_check.py",
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

LEGACY_PATHS: dict[str, str] = {**PUBLIC_LEGACY_PATHS, **DEV_LEGACY_PATHS}


def script_path(rel: str) -> Path:
    """rel 是旧仓 scripts/ 下的相对路径（如 ``eval-official/recorder.py``），或本仓 scripts/、dev-scripts/ 下的相对路径。"""
    if rel in LEGACY_PATHS:
        p = REPO / LEGACY_PATHS[rel]
    elif (SCRIPTS / rel).is_file():
        p = SCRIPTS / rel
    else:
        p = DEV_SCRIPTS / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_script(rel: str, *, fresh: bool = False):
    """同公开版 ``loaders.load_script``，但按本模块的 ``script_path`` 解析（含 dev-scripts/）。"""
    return load_path(rel, script_path(rel), fresh=fresh)
