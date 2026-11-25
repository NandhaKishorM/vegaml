"""Continuous learning left vegaml in 0.20 for the noncommercial vegaml-learn package.

vegaml must stay usable without it, must never pull it in, and must say how to get it when asked."""

import pathlib
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

# Run as if vegaml-learn were not installed, whether or not this environment has it.
WITHOUT = "import sys; sys.modules['vegaml_learn'] = None\n"


def _run(code):
    return subprocess.run([sys.executable, "-c", WITHOUT + code], capture_output=True, text=True,
                          timeout=300, cwd=str(ROOT))


def test_vegaml_imports_and_star_imports_without_vegaml_learn():
    out = _run("import vegaml\nfrom vegaml import *\nprint('Learner' in vegaml.__all__)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_asking_for_the_learner_names_the_package_and_its_licence():
    for code in ("import vegaml.learn", "from vegaml import Learner", "import vegaml.learn.proxy"):
        out = _run(code)
        assert out.returncode != 0, code
        assert "pip install vegaml-learn" in out.stderr, (code, out.stderr)
        assert "noncommercial" in out.stderr, (code, out.stderr)


def test_no_extra_brings_in_noncommercial_code():
    extras = PYPROJECT["project"]["optional-dependencies"]
    for name, reqs in extras.items():
        assert not any(r.replace("_", "-").startswith("vegaml-learn") for r in reqs), name
    assert not {"learn", "proxy", "claude"} & set(extras)
    assert PYPROJECT["project"]["license"] == "Apache-2.0"


def test_only_the_shim_ships_under_vegaml_learn():
    shipped = sorted(p.name for p in (ROOT / "vegaml" / "learn").glob("*.py"))
    assert shipped == ["__init__.py", "__main__.py"]
