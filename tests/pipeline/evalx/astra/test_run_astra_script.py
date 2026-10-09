"""The old launcher ``run_astra.sh`` is not migrated (its duties moved into ``AstraPolicy.load``/``close`` and
``ServerProcess``); this file turns the conventions it used to pin into equivalent assertions on the new entry point
``scripts/evaluate.py --model astra --gpus 0,1 --astra-ledger ... --astra-cap-usd 5``:

- the entry point's Astra args are passed through verbatim as ``load_policy(**cfg)`` keys, from which
  ``AstraPolicy`` resolves the two GPUs, the ledger and the cap;
- items copied from upstream ``examples/champ/run.sh``: the VLA launch command (only ``--seed=42`` is replaced by the
  model seed), each env var, the first three PYTHONPATH entries (the fourth, the env source, becomes the eval repo's
  ``third_party/robomme_benchmark/src``), ``env -u OPENAI_API_KEY``, the two GPUs must differ, default port 18762;
- the old launcher's refusals: same GPU twice, ``group_*`` layout, missing key, cost cap above 5 USD -- all refused
  before any server starts, without reading a key file.
"""
from __future__ import annotations

import inspect
import re

import pytest

from astra_fakes import FakeServer, Harness, astra_session, third_party
from tests._support.loaders import load_script


def _evaluate():
    return load_script("evaluate.py")


ARGV = ["--model", "astra", "--dataset", "hard-verify", "--seed", "7", "--out", "OUT", "--gpus", "0,1",
        "--ckpt", "/ckpt/symbolic-grounded-subgoal/79999", "--astra-ledger", "/cost/ledger.json",
        "--astra-cap-usd", "5", "--astra-prices", "/cost/prices.json", "--astra-monitor-adapter", "/ckpt/monitor"]


def test_evaluate_cli_astra_args_map_to_policy_cfg():
    ev = _evaluate()
    args, rest = ev.build_parser().parse_known_args(ARGV)
    cfg = ev.make_cfg(args, ev.parse_extra(rest))
    assert args.model == "astra" and args.seed == 7
    assert cfg == {"ckpt": "/ckpt/symbolic-grounded-subgoal/79999", "gpus": [0, 1], "astra_ledger": "/cost/ledger.json",
                   "astra_cap_usd": 5, "astra_prices": "/cost/prices.json", "astra_monitor_adapter": "/ckpt/monitor"}
    with astra_session() as (mod, _astra):
        p = mod.AstraPolicy(7, **cfg)
        p._parse_cfg()
        assert (p.vla_gpu, p.monitor_gpu, p.cap_usd) == (0, 1, 5.0)
        assert str(p.ledger) == "/cost/ledger.json" and str(p.state_path) == "/cost/ledger.state.json"
        assert str(p.group_dir) == "/cost/group_0" and p.port_base == mod.DEFAULT_PORT_BASE == 18762
        assert p.vla_argv(18762)[1:] == ["scripts/serve_policy.py", "--port=18762", "--seed=7", "policy:checkpoint",
                                         "--policy.config=mme_vla_suite",
                                         "--policy.dir=/ckpt/symbolic-grounded-subgoal/79999"]


def test_vla_command_and_env_copied_from_upstream_run_sh():
    """Copied from upstream run.sh: the VLA launch command only swaps the seed; env vars match one by one; the first
    three PYTHONPATH entries are identical, the fourth becomes the eval repo's env source."""
    upstream = (third_party() / "examples" / "champ" / "run.sh").read_text()
    up_cmd = '--port="$PORT" --seed=42 policy:checkpoint --policy.config=mme_vla_suite'
    assert up_cmd in upstream and 'env -u OPENAI_API_KEY CUDA_VISIBLE_DEVICES="$VLA_GPU"' in upstream
    assert "PORT=${PORT:-18762}" in upstream and "Use distinct GPUs for VLA and monitor" in upstream
    with astra_session() as (mod, _astra):
        p = mod.AstraPolicy(7, gpus=[0, 1], astra_ledger="/c/l.json", astra_prices="/c/p.json", ckpt="/ck",
                            astra_monitor_adapter="/m")
        p._parse_cfg()
        argv = p.vla_argv(18762)
        env = p.vla_env()
        assert argv[1:] == ["scripts/serve_policy.py", "--port=18762", "--seed=7", "policy:checkpoint",
                            "--policy.config=mme_vla_suite", "--policy.dir=/ck"]
        assert mod.VLAServerProcess.DROP_ENV == ("OPENAI_API_KEY",)
        exports = dict(kv.split("=", 1) for line in re.findall(r"^export ((?:[A-Z_]+=[^\s\"$]+ ?)+)$", upstream, re.M)
                       for kv in line.split())
        assert exports, "no fixed exports parsed from upstream run.sh"
        for k, v in exports.items():
            assert env[k] == v, f"{k}: upstream {v} != new {env.get(k)}"
        up_pp = re.search(r'export PYTHONPATH="([^"]+)"', upstream).group(1).split(":")
        new_pp = env["PYTHONPATH"].split(":")
        root = str(third_party())
        assert [x.replace("$REPO", root) for x in up_pp[:3]] == new_pp[:3]
        assert up_pp[3] == "$REPO/third_party/robomme_benchmark/src" and new_pp[3] == str(mod.benchmark_src())
        src = inspect.getsource(mod)
        assert "openai.key" not in src and "from_key_file" not in src, "the key is read only from env vars"


@pytest.mark.parametrize("cfg,match", [
    ({"gpus": [0, 0]}, "astra_gpus"),
    ({"astra_group_dir": "elsewhere"}, "reason=layout"),
    ({"astra_cap_usd": 6}, "ASTRA_COST_BLOCKED"),
    ({"_no_key": True}, "OPENAI_API_KEY"),
])
def test_old_launcher_refusals_now_in_load(tmp_path, monkeypatch, cfg, match):
    """The old launcher's four refusals (same GPU, layout, cost cap, missing key) are now done by ``load`` before any
    server starts, without leaking the key."""
    with astra_session() as (mod, astra):
        h = Harness(tmp_path, monkeypatch, mod, astra)
        if cfg.pop("_no_key", False):
            monkeypatch.setenv("OPENAI_API_KEY", "")
        for k, v in cfg.items():
            h.cfg[k] = str(tmp_path / v) if k == "astra_group_dir" else v
        with pytest.raises(ValueError, match=match) as info:
            h.load()
    assert "sk-test" not in str(info.value)
    assert FakeServer.instances == []
