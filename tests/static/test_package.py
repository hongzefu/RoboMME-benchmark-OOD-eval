"""L0 (slow): packaging and provenance of the eval package ``robomme-ood-eval`` (C18-WHEEL).

After the repo split this repo only packages the eval package ``src/robomme_ood_eval``
(``[tool.hatch.build.targets.wheel] packages``); the ``robomme`` / ``robomme_hard`` wheels are guarded by the
benchmark repo's own tests of the same name.

Build the wheel -> create a fresh uv env in tmp -> non-editable install (``--no-deps``; dependencies are borrowed
read-only from the current interpreter's site-packages via ``PYTHONPATH``, where editable-install ``.pth`` files are not
processed, so ``robomme_ood_eval`` can only come from the tmp env) -> import from a cwd outside the repo, and check:

- the eval package modules actually live in the tmp env's site-packages, and the tmp env has no ``.pth`` pointing back
  at the repo;
- the file set in the wheel equals the git-tracked file set of ``src/robomme_ood_eval``, and after install it is
  byte-identical to the source tree (including the ``models``, ``record`` and ``servers`` subpackages);
- dist-info Name / Version match ``pyproject.toml`` and the install is not editable.

Network is restricted: build, env creation and install all use ``--offline``; if installation is impossible, call
``pytest.skip("Not verified: <reason>")`` instead of recording a PASS.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path

import pytest

from tests._support.loaders import REPO
from tests._support.resource_policy import SITE_DIR

pytestmark = pytest.mark.slow

PACKAGES = ("robomme_ood_eval",)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=240, **kw)


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("Not verified: uv is not on PATH")
    return uv


def _tracked_sources() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *(f"src/{p}" for p in PACKAGES)],
                         cwd=REPO, check=True, capture_output=True).stdout
    return sorted(x.decode() for x in out.split(b"\0") if x)


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    uv = _uv()
    tmp = tmp_path_factory.mktemp("wheel")
    dist, venv, cwd = tmp / "dist", tmp / "venv", tmp / "outside"
    cwd.mkdir()
    env = dict(os.environ)
    # Build and install must not be redirected to the main .venv by the current project env vars.
    for k in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "PYTHONPATH"):
        env.pop(k, None)
    r = _run([uv, "build", "--wheel", "--offline", "--out-dir", str(dist)], cwd=REPO, env=env)
    if r.returncode != 0:
        pytest.skip(f"Not verified: offline wheel build failed: {r.stderr.strip()[-300:]}")
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels
    r = _run([uv, "venv", "--offline", "--python", sys.executable, str(venv)], cwd=cwd, env=env)
    if r.returncode != 0:
        pytest.skip(f"Not verified: offline tmp env creation failed: {r.stderr.strip()[-300:]}")
    py = venv / "bin" / "python"
    r = _run([uv, "pip", "install", "--offline", "--no-deps", "--python", str(py), str(wheels[0])], cwd=cwd, env=env)
    if r.returncode != 0:
        pytest.skip(f"Not verified: offline wheel install failed: {r.stderr.strip()[-300:]}")
    site = Path(_run([str(py), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                     cwd=cwd, env={"PATH": env.get("PATH", "")}).stdout.strip())
    return {"wheel": wheels[0], "py": py, "site": site, "cwd": cwd, "env": env}


def test_wheel_contents_equal_tracked_sources(installed):
    with zipfile.ZipFile(installed["wheel"]) as z:
        names = {n for n in z.namelist() if ".dist-info/" not in n}
    assert names == {rel.removeprefix("src/") for rel in _tracked_sources()}


def test_installed_files_byte_identical_to_source(installed):
    site = installed["site"]
    bad = [rel for rel in _tracked_sources()
           if (site / rel.removeprefix("src/")).read_bytes() != (REPO / rel).read_bytes()]
    assert bad == []
    # All subpackages are present (not just the top-level __init__).
    for sub in ("models", "record", "servers"):
        assert (site / "robomme_ood_eval" / sub / "__init__.py").is_file(), sub
    assert (site / "robomme_ood_eval" / "episode.py").is_file() and (site / "robomme_ood_eval" / "policy.py").is_file()


def test_dist_info_metadata_and_not_editable(installed):
    site = installed["site"]
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    infos = list(site.glob("*.dist-info"))
    mine = [d for d in infos if d.name.startswith(f"{project['name'].replace('-', '_')}-")]  # the dist-info name normalizes - to _
    assert len(mine) == 1, infos
    meta = (mine[0] / "METADATA").read_text(encoding="utf-8").splitlines()
    assert f"Name: {project['name']}" in meta
    assert f"Version: {project['version']}" in meta
    direct = mine[0] / "direct_url.json"
    if direct.exists():
        assert not json.loads(direct.read_text()).get("dir_info", {}).get("editable", False)
    # No .pth in the tmp env points back at the repo.
    for pth in site.glob("*.pth"):
        assert str(REPO) not in pth.read_text(encoding="utf-8"), pth


def test_import_resolves_to_tmp_env_outside_repo(installed):
    deps = sysconfig.get_paths()["purelib"]  # borrow dependencies only; editable .pth files there are not processed via PYTHONPATH
    env = dict(installed["env"])
    env["PYTHONPATH"] = os.pathsep.join([str(SITE_DIR), deps])
    code = ("import json, robomme_ood_eval, robomme_ood_eval.episode as ep, robomme_ood_eval.policy as pol\n"
            "print(json.dumps({'robomme_ood_eval': robomme_ood_eval.__file__, 'episode': ep.__file__,"
            " 'policy': pol.__file__}))\n")
    r = _run([str(installed["py"]), "-c", code], cwd=installed["cwd"], env=env)
    if r.returncode != 0 and "No module named" in r.stderr \
            and "robomme_ood_eval" not in r.stderr.split("No module named")[-1]:
        pytest.skip(f"Not verified: import with borrowed dependencies failed: {r.stderr.strip()[-300:]}")
    assert r.returncode == 0, r.stderr[-2000:]
    files = json.loads(r.stdout.strip().splitlines()[-1])
    site = installed["site"].resolve()
    for name, f in files.items():
        p = Path(f).resolve()
        assert site in p.parents, (name, p)
        assert REPO.resolve() not in p.parents, (name, p)
