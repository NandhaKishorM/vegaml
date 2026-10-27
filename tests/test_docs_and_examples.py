"""Docs and examples cannot rot quietly: files exist, examples compile, extras are real."""
from __future__ import annotations

import pathlib
import py_compile
import re
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

PAGES = ["index.md", "questions.md", "cli.md", "http-api.md", "mcp.md",
         "structured.md", "router.md", "guard.md", "calibration.md", "escalation.md", "cpu.md",
         "evals.md", "langchain.md", "onnx.md", "mlx.md", "docker.md"]


@pytest.mark.parametrize("page", PAGES)
def test_every_documented_page_exists(page):
    assert (ROOT / "docs" / page).is_file()


def test_the_index_links_every_page_that_exists():
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    on_disk = {p.name for p in (ROOT / "docs").glob("*.md")} - {"index.md"}
    linked = set(re.findall(r"\]\((\w[\w-]*\.md)\)", index))
    assert on_disk == linked, "docs/ and the index table drifted"


@pytest.mark.parametrize("script", sorted((ROOT / "examples").glob("*.py")),
                         ids=lambda p: p.name)
def test_every_example_compiles(script):
    py_compile.compile(str(script), doraise=True)


def test_every_extra_named_in_docs_is_declared():
    declared = set(PYPROJECT["project"]["optional-dependencies"])
    named = set()
    for page in (ROOT / "docs").glob("*.md"):
        named |= set(re.findall(r"vegaml\[(\w+)\]", page.read_text(encoding="utf-8")))
    for script in (ROOT / "examples").glob("*.py"):
        named |= set(re.findall(r"vegaml\[(\w+)\]", script.read_text(encoding="utf-8")))
    assert named <= declared, "docs name extras pyproject does not declare: %s" % (named - declared)


def test_every_console_script_named_in_docs_is_declared():
    scripts = set(PYPROJECT["project"]["scripts"])
    text = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "docs").glob("*.md"))
    for name in re.findall(r"`(vegaml[\w-]*)`", text):
        if name.startswith("vegaml-"):
            assert name in scripts, "docs name %r, pyproject does not" % name
