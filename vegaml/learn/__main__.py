"""`python -m vegaml.learn` runs `python -m vegaml_learn` (pip install vegaml-learn)."""

import runpy

import vegaml.learn  # noqa: F401  (says how to install vegaml-learn when it is missing)

runpy.run_module("vegaml_learn", run_name="__main__", alter_sys=True)
