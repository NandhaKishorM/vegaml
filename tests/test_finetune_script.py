"""The fine-tuning script: guards for the paths that failed quietly or crashed confusingly.

Everything here is checkable without a GPU or a checkpoint. The defects pinned:

* `n < batch` ran zero training iterations and then divided by zero on the log line. Reachable from
  the documented smoke test: a 96-row slice over 2 ranks is 48 rows per rank, and stage 2 runs with
  `--batch 64`.
* The positive forward pass was computed twice per step with identical arguments.
* The scheduler was sized from `n // batch`, which is not the number of iterations actually run.
* `fit_thermal` and `conformal_thresholds` skip silently below 30 rows per type, so an adapter with
  no calibration and no conformal table could still be written with `gate_trained: True`.
* The seven adapter target paths were typed out a second time, away from the package's canonical
  list.
* `np.load` on an `.npz` holds the zip open until closed.
* The dataset's integer labels index `QUESTION["criteria"]` positionally, with nothing asserting the
  two orders agree.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "notebooks" / "vega_finetune.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def _stage_node(name):
    return next(n for n in ast.walk(ast.parse(SOURCE))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _stage(name):
    return ast.unparse(_stage_node(name))


# ------------------------------------------------------------------ the crash
def test_training_refuses_a_split_smaller_than_one_batch():
    """`range(0, n - batch + 1, batch)` is empty when n < batch, so `seen` stayed 0 and the
    per-epoch log line divided by it."""
    body = _stage("stage_train")
    assert "n < args.batch" in body, "an undersized split must be refused, not divided by"
    assert "raise SystemExit" in body, "and refused with an explanation"


def test_the_log_line_still_dives_by_seen_so_the_guard_is_where_it_looks():
    """The guard is before the loop; the division is inside it. Both must be present."""
    body = _stage("stage_train")
    assert "run / seen" in body and "hits / seen" in body
    guard_at = body.index("n < args.batch")
    loop_at = body.index("for s in range(0, n - args.batch + 1, args.batch)")
    assert guard_at < loop_at, "the guard has to come before the loop that would skip every batch"


# ------------------------------------------------------------------ the wasted work
def test_the_positive_forward_pass_is_not_computed_twice():
    """The first call already produced out_pos, and the gate step then recomputed it identically,
    roughly doubling the cost of every training step.

    Checked on the AST, not on unparsed text: `ast.unparse` writes a tuple target as
    `(out_pos, logits) = ...` on 3.10 and `out_pos, logits = ...` from 3.11, so a text match
    could not tell a regression from an interpreter difference.
    """
    stage = _stage_node("stage_train")
    positive = [n for n in ast.walk(stage)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "engine_logits"
                and ast.unparse(n) == "engine_logits(engine, feats, opts, mask, qt_pos[:len(b)])"]
    assert len(positive) == 1, "the identical call must appear once"

    bound = [n for n in ast.walk(stage)
             if isinstance(n, ast.Assign) and n.value is positive[0]]
    assert len(bound) == 1 and isinstance(bound[0].targets[0], ast.Tuple), \
        "the positive pass must be unpacked"
    names = [getattr(e, "id", None) for e in bound[0].targets[0].elts]
    assert names == ["out_pos", "logits"], "and its result must be what the gate step reads"

    reads = [n for n in ast.walk(stage)
             if isinstance(n, ast.Name) and n.id == "out_pos" and isinstance(n.ctx, ast.Load)]
    assert reads, "the gate step must read out_pos rather than recompute it"


def test_the_scheduler_is_sized_to_the_iterations_actually_run():
    """floor((n - batch) / batch) + 1 equals n // batch only when batch divides n. Scheduling against
    n // batch cut the anneal tail off whenever it did not, leaving the final LR above the minimum."""
    body = _stage("stage_train")
    assert "iters = max(1, (n - args.batch) // args.batch + 1)" in body
    assert "steps = iters * args.epochs" in body
    assert "n // args.batch) * args.epochs" not in body


# ------------------------------------------------------------------ the silent no-calibration adapter
def test_a_run_that_could_not_be_calibrated_stops_instead_of_writing_the_adapter():
    """Both fitters skip below 30 rows per type, leaving theta all zeros (tau = 1) and no conformal
    table. The adapter was then written with gate_trained: True and produced prediction_set: None."""
    body = _stage("stage_train")
    assert "len(yv) < 30" in body, "an uncalibratable validation split must stop the run"
    assert "conf.get('choice')" in body, "an empty conformal table must stop the run"
    assert "np.any(theta)" in body, "an all-zero theta must stop the run"
    # and each guard must come before the adapter is written
    assert body.index("len(yv) < 30") < body.index("save_file(")


def test_the_metadata_records_how_many_examples_the_calibration_came_from():
    """So a reader can tell at a glance whether valid_accuracy rests on enough data."""
    assert "'valid_examples': int(len(yv))" in _stage("stage_train")


# ------------------------------------------------------------------ drift from the package
def test_the_adapter_targets_come_from_the_package():
    """The script typed the seven paths out a second time. If the package's list changed, the script
    would train one set of projections and write metadata describing another."""
    body = _stage("stage_train")
    assert "ADAPTER_TARGETS" in body, "import the canonical list rather than repeating it"
    from vegaml._common import ADAPTER_TARGETS
    for path in ADAPTER_TARGETS:
        assert path not in body, "%s is typed out again instead of imported" % path


def test_the_targets_written_to_the_metadata_are_the_ones_that_were_trained():
    body = _stage("stage_train")
    assert "'targets': list(ADAPTER_TARGETS)" in body


# ------------------------------------------------------------------ resource handling
def test_every_cached_shard_is_closed():
    """np.load on an .npz holds the zip open until it is closed; leaving them open keeps every
    shard locked, which on Windows means the cache directory cannot be deleted."""
    assert "with np.load(p) as z:" in SOURCE, "read each shard inside a with"
    assert "d = [np.load(p) for p in parts]" not in SOURCE


def test_every_json_read_in_the_script_is_closed():
    """json.load(open(...)) leaves the handle to refcounting; on CPython that happens to close it,
    but it is not a guarantee the script should be relying on."""
    tree = ast.parse(SOURCE)
    json_loads = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "load" and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == "json"]
    for call in json_loads:
        assert not (call.args and isinstance(call.args[0], ast.Call)
                    and isinstance(call.args[0].func, ast.Name)
                    and call.args[0].func.id == "open"), \
            "line %d: json.load(open(...)) leaves the handle unclosed" % call.lineno
    assert json_loads, "expected at least one json.load to check"


# ------------------------------------------------------------------ the label-order hazard
def test_the_dataset_label_order_is_asserted_against_the_question_criteria():
    """The dataset's integer labels index the criteria positionally. A mismatch trains every example
    against the wrong label and reports itself as a success."""
    assert 'assert list(QUESTION["criteria"]) == EMOTIONS' in SOURCE


def test_the_criteria_and_the_dataset_label_order_actually_agree():
    """Evaluated on the real literals, so this is a check and not a text match."""
    env = {}
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("EMOTIONS", "QUESTION"):
            env[node.targets[0].id] = ast.literal_eval(node.value)
    assert list(env["QUESTION"]["criteria"]) == env["EMOTIONS"], \
        "dair-ai/emotion label order does not match the question's criteria order"
    assert len(env["QUESTION"]["criteria"]) == 6, "the dataset has six emotions"


# ------------------------------------------------------------------ the notebook follows
def test_the_notebook_embeds_this_script_byte_for_byte():
    """The notebook writes this file to disk on Kaggle, so any of the above that is not in the
    notebook has not actually been fixed for a notebook user."""
    import json

    nb = json.loads((ROOT / "notebooks" / "vega_finetune_2xT4_kaggle.ipynb").read_text(encoding="utf-8"))
    written = [c for c in nb["cells"] if c["cell_type"] == "code"
               and "".join(c["source"]).startswith("%%writefile vega_finetune.py")]
    assert len(written) == 1, "expected exactly one %%writefile cell"
    assert "".join(written[0]["source"]).split("\n", 1)[1] == SOURCE, \
        "notebook and vega_finetune.py have diverged -- rerun notebooks/build_finetune_nb.py"


def test_the_documented_row_count_matches_the_defaults():
    """The docstring said 18k; the defaults sum to 24k, which is what build_finetune_nb.py says.

    The defaults are read out of the AST rather than matched as text, so this stays true if the
    numbers are changed.
    """
    total = 0
    for node in ast.walk(ast.parse(SOURCE)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            if node.args and isinstance(node.args[0], ast.Constant):
                flag = node.args[0].value
                if flag in ("--n-train", "--n-valid", "--n-test", "--n-neg"):
                    for kw in node.keywords:
                        if kw.arg == "default":
                            total += ast.literal_eval(kw.value)
    assert total == 24000, "the defaults sum to %d rows" % total
    assert "reading ~24k" in SOURCE, "the docstring must state that same total"
