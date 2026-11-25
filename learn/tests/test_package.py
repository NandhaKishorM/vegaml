"""vegaml-learn as a package: its licence, its version, and the old vegaml.learn paths resolving to it."""

import pathlib
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import vegaml_learn

HERE = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((HERE / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_licence_is_polyform_noncommercial_with_its_required_notice():
    text = (HERE / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Required Notice: Copyright 2026 Nandakishor M")
    assert "# PolyForm Noncommercial License 1.0.0" in text
    assert PYPROJECT["project"]["license"] == "PolyForm-Noncommercial-1.0.0"
    assert PYPROJECT["project"]["license-files"] == ["LICENSE"]


def test_the_version_matches_pyproject():
    assert vegaml_learn.__version__ == PYPROJECT["project"]["version"]


def test_old_import_paths_are_the_same_objects():
    import vegaml
    import vegaml.learn
    import vegaml.learn.learner
    import vegaml.learn.records
    import vegaml_learn.learner
    import vegaml_learn.records

    assert vegaml.Learner is vegaml_learn.Learner
    assert vegaml.learn.Learner is vegaml_learn.Learner
    assert vegaml.learn.Escalator is vegaml_learn.Escalator
    # One module object, not a second copy: module-level state and isinstance checks stay shared.
    assert vegaml.learn.learner is vegaml_learn.learner
    assert vegaml.learn.records is vegaml_learn.records
    from vegaml.learn.learner import Store

    assert Store is vegaml_learn.learner.Store
    assert sorted(vegaml.learn.__all__) == sorted(vegaml_learn.__all__)


def test_python_dash_m_vegaml_learn_still_runs():
    for module in ("vegaml.learn", "vegaml_learn"):
        out = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True,
                             timeout=120)
        assert out.returncode == 0, out.stderr
        assert "serve" in out.stdout
