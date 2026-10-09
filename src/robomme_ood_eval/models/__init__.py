"""Model registry: name string -> (module path, Policy subclass name), imported lazily.

All six models are registered here once; adding a model only means adding its Policy subclass at the end of
its own module.
"""
from __future__ import annotations

import importlib

#: registered name -> (module path, class name); the name is the value of ``scripts/evaluate.py --model`` and the
#: ``<model>`` directory name in the output tree
REGISTRY: dict[str, tuple[str, str]] = {
    "dummy": ("robomme_ood_eval.models.dummy", "DummyPolicy"),
    "perceptual-framesamp-modul": ("robomme_ood_eval.models.framesamp_modul", "FrameSampModulPolicy"),
    "groundsg": ("robomme_ood_eval.models.groundsg", "GroundSGPolicy"),
    "smvla": ("robomme_ood_eval.models.smvla", "SmvlaPolicy"),
    "pp": ("robomme_ood_eval.models.pp", "PPPolicy"),
    "astra": ("robomme_ood_eval.models.astra", "AstraPolicy"),
}
MODELS = tuple(REGISTRY)
#: Models that need ``--ckpt`` (no default checkpoint). Kept static so the CLI can check it without importing
#: every model module; must match each Policy subclass's ``requires_ckpt`` (pinned by a unit test).
MODELS_REQUIRING_CKPT: frozenset[str] = frozenset({"perceptual-framesamp-modul", "groundsg", "smvla", "pp"})


def resolve(name: str) -> type:
    """Return the Policy subclass for a registered name (the module is imported only now). Unknown names raise
    ``KeyError``; a module without the class raises ``ImportError``."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; choices: {', '.join(MODELS)}")
    mod_name, cls_name = REGISTRY[name]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ImportError(f"{mod_name} has no {cls_name} (the Policy subclass for model {name!r} is not implemented yet)")
    return cls
