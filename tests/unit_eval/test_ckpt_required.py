"""``--ckpt`` has no default: models that need weights must be given ``--ckpt`` on every run (CPU only).

Covers the CLI argument check in ``scripts/evaluate.py`` (exit code 2 before anything is loaded), the
``servers.require_ckpt`` helper, and the agreement between the static ``models.MODELS_REQUIRING_CKPT`` set and each
Policy subclass's ``requires_ckpt`` attribute."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robomme_ood_eval import policy as P
from robomme_ood_eval import servers as S
from robomme_ood_eval.models import MODELS_REQUIRING_CKPT, REGISTRY, resolve

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "evaluate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_ckpt_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Reached(Exception):
    """Raised by the stubbed ``load_policy`` to prove argument checking passed."""


def _stub_load_policy(monkeypatch) -> list:
    seen: list = []

    def fake(model, seed, **cfg):
        seen.append((model, seed, cfg))
        raise _Reached
    monkeypatch.setattr(P, "load_policy", fake)
    return seen


@pytest.mark.parametrize("model", sorted(MODELS_REQUIRING_CKPT))
def test_missing_ckpt_is_argument_error(model, tmp_path, monkeypatch, capsys):
    seen = _stub_load_policy(monkeypatch)
    mod = load_script()
    with pytest.raises(SystemExit) as ei:
        mod.main(["--model", model, "--dataset", "ood", "--seed", "0", "--out", str(tmp_path)])
    assert ei.value.code == 2
    assert f"--ckpt is required for model {model}" in capsys.readouterr().err
    assert seen == []


def test_ckpt_given_passes_argument_check(tmp_path, monkeypatch):
    seen = _stub_load_policy(monkeypatch)
    mod = load_script()
    with pytest.raises(_Reached):
        mod.main(["--model", "smvla", "--dataset", "ood", "--seed", "0", "--out", str(tmp_path),
                  "--ckpt", str(tmp_path / "ck")])
    assert seen == [("smvla", 0, {"ckpt": str(tmp_path / "ck")})]


def test_dummy_without_ckpt_passes_argument_check(tmp_path, monkeypatch):
    seen = _stub_load_policy(monkeypatch)
    mod = load_script()
    with pytest.raises(_Reached):
        mod.main(["--model", "dummy", "--dataset", "ood", "--seed", "0", "--out", str(tmp_path)])
    assert seen == [("dummy", 0, {})]


def test_require_ckpt():
    with pytest.raises(ValueError, match="--ckpt is required for model smvla"):
        S.require_ckpt({}, "smvla")
    with pytest.raises(ValueError, match="--ckpt is required for model groundsg"):
        S.require_ckpt({"ckpt": None}, "groundsg")
    assert S.require_ckpt({"ckpt": "/x/y"}, "smvla") == Path("/x/y")


def test_no_default_ckpt_table():
    assert not hasattr(S, "DEFAULT_CKPTS")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_requires_ckpt_matches_static_set(name):
    cls = resolve(name)
    assert cls.requires_ckpt is (name in MODELS_REQUIRING_CKPT)
