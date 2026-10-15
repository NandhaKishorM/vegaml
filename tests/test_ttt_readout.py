"""The TTT readout: crashes that named nothing, and a cross-validation that could flatter itself.

Three groups of defects, all on the path `mode="ttt"` and the head-selecting half of
`mode="auto"`.

1. Crashes with messages that identified nothing. A head asked for labels it was never fitted on
   raised `max() arg is an empty sequence`; `fit` raised a bare `KeyError('zzz')`; `predict` on an
   unfitted head raised `AttributeError` on `self.mu`.

2. Cross-validation cut folds from a single random permutation, so a class with few examples could
   be absent from a training fold entirely. The head for that fold was then fitted on a label space
   it could not predict, and the resulting accuracy was still reported as if it measured the head.

3. Every head answer was keyed `"choice"`, whatever the question type. `mode="auto"` is documented
   as reproducing the engine's answer shape, but a boolean question answered by a head had no
   `"boolean"` field, so a caller reading a mixed result needed two code paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from vegaml import TTTHead, Vega


class _Engine:
    """Stands in for a loaded checkpoint: the mode switch's surface, none of the weights."""

    cfg = {"max_len": 73728}
    model_dir = "/nowhere"
    backbone = "dummy"

    def decide(self, state, questions, alpha="0.1", views=1, **kw):
        """A stand-in that answers in the engine's real shape, including per-type keys."""
        out = {}
        for qid, q in questions.items():
            t = q["type"]
            if t == "choice":
                labels = list(q["criteria"])
            elif t == "score":
                labels = list(range(len(q["criteria"])))
            else:
                labels = [False, True]
            probs = {str(labels[i]): 1.0 / len(labels) for i in range(len(labels))}
            a = {"probs": probs, "confidence": max(probs.values())}
            if t == "choice":
                a["choice"] = labels[0]
            elif t == "score":
                a["score"] = 0
                a["expected_score"] = 0.0
            else:
                a["boolean"] = 0.5
            out[qid] = a
        return {"answers": out}


def _separable(n=40, d=48, classes=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(classes), d)) * 3.0
    X, y = [], []
    for i, c in enumerate(classes):
        X.append(centres[i] + rng.normal(scale=0.35, size=(n, d)))
        y += [c] * n
    return np.vstack(X), y


def _vega_with_head(mode="ttt", labels=("a", "b"), cv=0.95):
    v = Vega(_Engine(), mode=mode)
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(20, 16)) + 4, rng.normal(size=(20, 16)) - 4])
    v.heads["q"] = (TTTHead(list(labels)).fit(X, [labels[0]] * 20 + [labels[1]] * 20), cv)
    v.features = lambda state, q: X[0]
    return v


BOOLEAN = {"type": "boolean", "instructions": "does it hold?"}
SCORE3 = {"type": "score", "instructions": "how bad?", "criteria": ["low", "mid", "high"]}


# ------------------------------------------------------------------ 1. crashes that named nothing
def test_a_head_asked_for_labels_it_never_saw_names_both_sides():
    """It raised `max() arg is an empty sequence`, naming neither the head nor the question."""
    v = _vega_with_head("ttt", labels=("a", "b"))
    with pytest.raises(RuntimeError) as excinfo:
        v.decide("s", {"q": BOOLEAN})
    message = str(excinfo.value)
    assert "q" in message, "the message must name the question"
    assert "a" in message and "b" in message, "and what the head was fitted on"
    assert "true" in message, "and what the question actually offers"


def test_fit_names_the_offending_label_and_the_head_label_set():
    X, _ = _separable()
    with pytest.raises(ValueError) as excinfo:
        TTTHead(["a", "b"]).fit(X, ["a"] * 20 + ["b"] * 19 + ["zzz"])
    message = str(excinfo.value)
    assert "zzz" in message, "the offending label"
    assert "a" in message and "b" in message, "and the label set it belongs to"


def test_predicting_with_an_unfitted_head_says_so():
    with pytest.raises(RuntimeError, match="not been fitted"):
        TTTHead(["a", "b"]).predict(np.zeros(4))


def test_predicting_with_an_allowed_set_that_shares_nothing_says_so():
    X, y = _separable(classes=("a", "b"))
    head = TTTHead(["a", "b"]).fit(X, y)
    with pytest.raises(ValueError) as excinfo:
        head.predict(X[0], allowed=["x", "y"])
    message = str(excinfo.value)
    assert "no overlap" in message and "x" in message


def test_fit_reports_a_label_outside_the_question_options_instead_of_raising():
    """`fit` has a report per question, so this belongs in the report, not in a traceback."""
    v = Vega(_Engine())
    v.features = lambda state, q: np.random.default_rng(0).normal(size=16)
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    report = v.fit([("s%d" % i, {"q": "zzz"}) for i in range(12)], qs)
    assert report["q"]["fitted"] is False
    assert "zzz" in report["q"]["reason"] and "a" in report["q"]["reason"]
    assert "q" not in v.heads


def test_fit_reports_a_single_class_question_rather_than_fitting_nothing():
    v = Vega(_Engine())
    v.features = lambda state, q: np.random.default_rng(0).normal(size=16)
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    report = v.fit([("s%d" % i, {"q": "a"}) for i in range(12)], qs)
    assert report["q"]["fitted"] is False
    assert "same label" in report["q"]["reason"]
    assert "q" not in v.heads


# ------------------------------------------------------------------ 2. cross-validation
def test_every_example_is_still_scored_exactly_once_on_imbalanced_data():
    """Folds used to come from one permutation, so a rare class could miss a training fold."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 16))
    y = ["a"] * 55 + ["b"] * 5
    head = TTTHead(["a", "b"])
    acc = head.cross_val_accuracy(X, y, folds=5)
    assert acc is not None
    assert head.cross_val_n == 60, "every example must still be scored exactly once"


def test_a_class_too_rare_to_stratify_is_excluded_from_the_estimate():
    """5 folds needs 5 members per class to place one in each. Fewer and it cannot be stratified."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(62, 8))
    y = ["a"] * 40 + ["b"] * 20 + ["c"] * 2
    head = TTTHead(["a", "b", "c"])
    acc = head.cross_val_accuracy(X, y, folds=5)
    assert acc is not None
    assert head.cross_val_n == 60, "the two rare examples are excluded; the other 60 are scored once"


def test_cross_val_returns_none_when_no_class_can_fill_the_folds():
    """Returning a number here would be a head scoring itself on data it was fitted on."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8))
    y = ["a"] * 29 + ["b"]
    head = TTTHead(["a", "b"])
    assert head.cross_val_accuracy(X, y, folds=5) is None


def test_cross_val_n_is_readable_before_any_estimate_has_been_run():
    head = TTTHead(["a", "b"])
    assert head.cross_val_n == 0, "reading it must never be an AttributeError"


def test_cross_val_is_deterministic():
    """A fixed seed, so the same examples always produce the same reported accuracy."""
    X, y = _separable()
    first = TTTHead(["a", "b", "c"]).cross_val_accuracy(X, y)
    second = TTTHead(["a", "b", "c"]).cross_val_accuracy(X, y)
    assert first == second


def test_cross_val_still_high_on_separable_data_and_low_on_noise():
    X, y = _separable(classes=("a", "b", "c"))
    assert TTTHead(["a", "b", "c"]).cross_val_accuracy(X, y) > 0.9
    Xn = np.random.default_rng(1).normal(size=(80, 48))
    yn = [["a", "b"][i % 2] for i in range(80)]
    assert TTTHead(["a", "b"]).cross_val_accuracy(Xn, yn) < 0.75, \
        "labels independent of features: the number that tells a caller the head learned nothing"


@pytest.mark.parametrize("n_adapters", [1, 2, 3])
def test_cross_val_survives_a_class_with_one_example(n_adapters):
    """The degenerate case that used to be reachable: a class that cannot fill any fold."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    y = ["a"] * (19 - n_adapters) + ["b"] * n_adapters + ["rare"]
    head = TTTHead(["a", "b", "rare"])
    acc = head.cross_val_accuracy(X, y, folds=3)
    assert acc is None or 0.0 <= acc <= 1.0, "never a crash and never a nonsense number"


# ------------------------------------------------------------------ 3. the answer shape
def test_a_boolean_question_answered_by_a_head_carries_its_boolean_field():
    """The engine keys a boolean answer "boolean"; auto is documented as reproducing that shape."""
    v = _vega_with_head("ttt", labels=("false", "true"))
    ans = v.decide("s", {"q": BOOLEAN})["answers"]["q"]
    assert "boolean" in ans
    assert set(ans["probs"]) == {"false", "true"}
    assert ans["boolean"] == pytest.approx(ans["probs"]["true"])


def test_a_score_question_answered_by_a_head_carries_score_and_expected_score():
    v = _vega_with_head("ttt", labels=("0", "1", "2"))
    ans = v.decide("s", {"q": SCORE3})["answers"]["q"]
    assert ans["score"] in (0, 1, 2)
    assert "expected_score" in ans
    assert 0.0 <= ans["expected_score"] <= 2.0


def test_a_choice_question_still_carries_choice():
    v = _vega_with_head("ttt", labels=("a", "b"))
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    ans = v.decide("s", qs)["answers"]["q"]
    assert ans["choice"] in ("a", "b")


def test_auto_answers_carry_the_same_keys_regardless_of_which_readout_produced_them():
    """The whole point of "auto is the engine, flags and all": the shape must not change."""
    v = _vega_with_head("auto", labels=("a", "b"))
    fitted = v.decide("s", {"q": {"type": "choice", "instructions": "which?",
                                  "criteria": {"a": "", "b": ""}}})["answers"]["q"]
    assert fitted["readout"] == "ttt"
    assert "choice" in fitted and "confidence" in fitted

    unfitted = v.decide("s", {"other": {"type": "boolean", "instructions": "?"}})["answers"]["other"]
    assert unfitted["readout"] == "engine"
    # The engine answer carries the calibration fields the head answer deliberately does not, which
    # is the documented difference between the two readouts -- but the answer-type key matches.
    assert "boolean" in unfitted and "choice" in fitted


def test_auto_marks_every_answer_with_the_readout_that_produced_it():
    v = _vega_with_head("auto", labels=("a", "b"))
    questions = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}},
                 "other": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "c": ""}}}
    out = v.decide("s", questions)
    assert out["answers"]["q"]["readout"] == "ttt"
    assert out["answers"]["other"]["readout"] == "engine"
    assert [k for k in out["answers"]] == ["q", "other"], "asked order is kept"


def test_a_head_answer_is_marked_ttt_even_in_ttt_mode():
    v = _vega_with_head("ttt", labels=("a", "b"))
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    assert v.decide("s", qs)["answers"]["q"]["readout"] == "ttt"


class _ScriptedEngine(_Engine):
    def __init__(self, answers):
        self.answers = answers

    def decide(self, state, questions, alpha="0.1", views=1, **kw):
        out = super().decide(state, questions, alpha=alpha, views=views, **kw)
        for qid, q in questions.items():
            key = state if isinstance(state, str) and state in self.answers else q["instructions"]
            if key not in self.answers:
                continue
            want = self.answers[key]
            t = q["type"]
            if t == "choice":
                labels = list(q["criteria"])
                rest = 0.15 / max(1, len(labels) - 1)
                out["answers"][qid]["choice"] = want
                out["answers"][qid]["probs"] = {str(k): (0.85 if k == want else rest) for k in labels}
            elif t == "score":
                k = len(q["criteria"])
                out["answers"][qid]["score"] = int(want)
                out["answers"][qid]["probs"] = {str(i): (0.85 if i == int(want) else 0.15 / max(1, k - 1))
                                                for i in range(k)}
            else:
                p_true = 0.9 if want in (True, "true", "True") else 0.1
                out["answers"][qid]["boolean"] = p_true
                out["answers"][qid]["probs"] = {"False": 1.0 - p_true, "True": p_true}
        return out


def test_auto_keeps_the_engine_when_a_head_beats_chance_but_loses_to_the_engine():
    classes = ("a", "b", "c", "d")
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {k: "" for k in classes}}}
    X_sep, y = _separable(n=10, d=24, classes=classes, seed=4)
    rng = np.random.default_rng(4)
    X = X_sep.copy()
    for idx in range(0, 40, 4):
        X[idx] = rng.normal(size=24) * 5.0
    feats = {"s%d" % i: X[i] for i in range(len(y))}
    engine_answers = {"s%d" % i: (y[i] if i % 10 != 0 else classes[(classes.index(y[i]) + 1) % 4])
                      for i in range(len(y))}
    v = Vega(_ScriptedEngine(engine_answers))
    v.features = lambda state, q: feats[state]
    examples = [("s%d" % i, {"q": y[i]}) for i in range(len(y))]

    report = v.fit(examples, qs)
    assert report["q"]["fitted"] is True, "beats chance, so the head still exists for mode='ttt'"
    assert "q" in v.heads
    assert report["q"]["cv_accuracy"] > 0.25
    assert v.decide("s0", qs, mode="ttt")["answers"]["q"]["readout"] == "ttt"
    assert v.decide("s0", qs)["answers"]["q"]["readout"] == "engine"
    assert report["q"]["used_by_auto"] is False
    assert report["q"]["engine_accuracy"] == pytest.approx(0.9)
    assert report["q"]["cv_accuracy"] < report["q"]["engine_accuracy"]


def test_auto_uses_the_head_when_it_beats_both_chance_and_the_engine():
    classes = ("a", "b", "c", "d")
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {k: "" for k in classes}}}
    X, y = _separable(n=10, d=24, classes=classes, seed=5)
    feats = {"s%d" % i: X[i] for i in range(len(y))}
    engine_answers = {"s%d" % i: (y[i] if i % 2 == 0 else classes[(classes.index(y[i]) + 1) % 4])
                      for i in range(len(y))}
    v = Vega(_ScriptedEngine(engine_answers))
    v.features = lambda state, q: feats[state]
    examples = [("s%d" % i, {"q": y[i]}) for i in range(len(y))]

    report = v.fit(examples, qs)
    assert report["q"]["fitted"] is True
    assert report["q"]["engine_accuracy"] == pytest.approx(0.5)
    assert report["q"]["cv_accuracy"] > report["q"]["engine_accuracy"]
    assert report["q"]["used_by_auto"] is True
    assert v.decide("s0", qs)["answers"]["q"]["readout"] == "ttt"


def test_auto_keeps_the_engine_on_a_tie_and_scores_boolean_and_per_example_questions():
    X, y = _separable(n=10, d=16, classes=("false", "true"), seed=0)
    feats = {"u%d" % i: X[i] for i in range(len(y))}
    engine_answers = {"u%d" % i: y[i] for i in range(len(y))}
    v = Vega(_ScriptedEngine(engine_answers))
    v.features = lambda state, q: feats[q["instructions"]]
    examples = [({}, {"q": y[i]}, {"q": dict(BOOLEAN, instructions="u%d" % i)})
                for i in range(len(y))]

    report = v.fit(examples, {"q": BOOLEAN})
    assert report["q"]["fitted"] is True
    assert report["q"]["cv_accuracy"] == pytest.approx(1.0)
    assert report["q"]["engine_accuracy"] == pytest.approx(1.0)
    assert report["q"]["used_by_auto"] is False
    assert v.decide({}, {"q": dict(BOOLEAN, instructions="u0")})["answers"]["q"]["readout"] == "engine"


def test_engine_accuracy_is_scored_on_the_same_stratified_rows_as_cross_val():
    classes = ("a", "b", "rare")
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {k: "" for k in classes}}}
    X_ab, y_ab = _separable(n=10, d=24, classes=("a", "b"), seed=7)
    rng = np.random.default_rng(7)
    X_ab[0] = rng.normal(size=24) * 5.0
    X_ab[10] = rng.normal(size=24) * 5.0
    X = np.vstack([X_ab, rng.normal(size=(4, 24))])
    y = list(y_ab) + ["rare"] * 4
    feats = {"s%d" % i: X[i] for i in range(len(y))}
    engine_answers = {"s%d" % i: (y[i] if i < 20 else "a") for i in range(len(y))}
    v = Vega(_ScriptedEngine(engine_answers))
    v.features = lambda state, q: feats[state]
    examples = [("s%d" % i, {"q": y[i]}) for i in range(len(y))]

    report = v.fit(examples, qs)
    assert report["q"]["engine_accuracy"] == pytest.approx(1.0)
    assert report["q"]["cv_accuracy"] < report["q"]["engine_accuracy"]
    assert report["q"]["used_by_auto"] is False
    assert v.decide("s0", qs)["answers"]["q"]["readout"] == "engine"


def test_refit_evicts_stale_heads_and_falls_back_when_options_change():
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    X, y = _separable(n=10, d=16, classes=("a", "b"), seed=8)
    feats = {"s%d" % i: X[i] for i in range(len(y))}
    v = Vega(_Engine())
    v.features = lambda state, q: feats[state]
    examples = [("s%d" % i, {"q": y[i]}) for i in range(len(y))]

    assert v.fit(examples, qs)["q"]["used_by_auto"] is True
    assert "q" in v.heads
    other_q = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "c": ""}}}
    assert v.decide("s0", other_q)["answers"]["q"]["readout"] == "engine"

    assert v.fit([("s0", {"q": "a"})], qs)["q"]["fitted"] is False
    assert "q" not in v.heads

    v.fit(examples, qs)
    assert v.fit([("s%d" % i, {"q": "zzz"}) for i in range(10)], qs)["q"]["fitted"] is False
    assert "q" not in v.heads

    v.fit(examples, qs)
    assert v.fit([("s%d" % i, {"q": "a"}) for i in range(10)], qs)["q"]["fitted"] is False
    assert "q" not in v.heads


def test_auto_end_to_end_through_the_real_physics_engine(tmp_path):
    import json

    import torch
    from safetensors.torch import save_file

    from vegaml._api import Vega as RealEngine
    from vegaml._common import VegaEngine

    cfg = {
        "backbone_name": "dummy", "d_model": 12, "feature_layers": [0, 1], "option_layers": [1],
        "latent_dim": 8, "world_dim": 8, "proj_hidden": 16, "n_steps": 4, "dt_max": 0.5,
        "settle_eps": 1e-3, "dropout": 0.0, "feature_dropout": 0.0, "ssm_state": 4,
        "max_len": 512, "opt_max_tokens": 32,
        "calibration": {
            "theta": np.zeros((3, 5)).tolist(),
            "conformal": {"choice": {"0.1": 0.5}, "boolean": {"0.1": 0.5}, "score": {"0.1": 0.5}},
            "abstain_threshold": {"0.1": 0.3}, "default_risk": "0.1", "unbound_threshold": 100.0,
        },
    }
    (tmp_path / "vega_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    torch.manual_seed(0)
    ve = VegaEngine(12, 2, 1, cfg).eval()
    save_file(ve.state_dict(), str(tmp_path / "engine.safetensors"))

    found = {}
    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        f_t = torch.randn((1, 2, 3, 12), generator=g)
        o_t = torch.randn((1, 3, 1, 12), generator=g)
        logits = ve(f_t, o_t, torch.ones((1, 3), dtype=torch.bool),
                    torch.zeros(1, dtype=torch.long), halt=True, return_traj=False)["logits"]
        found.setdefault(["a", "b", "c"][int(logits[0].argmax())], (f_t[0], o_t[0]))
        if len(found) == 3:
            break

    class _Tok:
        bos_token_id = 1

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 250 + 2 for c in str(text)] or [2]}

    class _Perception:
        feature_layers = [0, 1]

        def __init__(self, state_feats, opt_vecs):
            self.tok = _Tok()
            self.state_feats = state_feats
            self.opt_vecs = opt_vecs

        def pooled(self, items, prefix_min=4096):
            feats = torch.zeros((len(items), 2, 3, 12), dtype=torch.float32)
            opts = []
            for b, it in enumerate(items):
                key = tuple(it["ids"])
                feats[b] = self.state_feats(key, b)
                opts.append(self.opt_vecs(key, len(it["spans"]["options"])))
            return feats, torch.cat(opts, dim=0)

    q = {"q": {"type": "choice", "instructions": "pick",
               "criteria": {"a": "first", "b": "second", "c": "third"}}}
    states = ["row-%02d-%s" % (i, "abc"[i // 10]) for i in range(30)]
    gold = ["abc"[i // 10] for i in range(30)]
    key_to_idx = {}
    probe_perc = _Perception(lambda k, b: torch.zeros((2, 3, 12)),
                             lambda k, n: torch.zeros((n, 1, 12)))
    probe_v = Vega(RealEngine(str(tmp_path), device="cpu", perception=probe_perc))
    for idx, s in enumerate(states):
        probe_perc.state_feats = (lambda k, b, idx=idx:
                                  (key_to_idx.setdefault(k, idx), torch.zeros((2, 3, 12)))[1])
        probe_v.features(s, q["q"])

    rng = np.random.default_rng(11)
    noisy = {0, 4, 8, 10, 14, 18, 20, 24, 28}
    st_1 = {i: (torch.tensor(rng.standard_normal((2, 3, 12)) * 4.0, dtype=torch.float32)
                if i in noisy else found[gold[i]][0]
                + torch.tensor(rng.standard_normal((2, 3, 12)) * 0.05, dtype=torch.float32))
            for i in range(30)}
    perc1 = _Perception(
        lambda k, b: st_1[key_to_idx[k]],
        lambda k, n: found[gold[key_to_idx[k]] if key_to_idx[k] not in {0, 10, 20}
                           else ("b" if gold[key_to_idx[k]] == "a" else "a")][1])
    v1 = Vega(RealEngine(str(tmp_path), device="cpu", perception=perc1))
    examples = [(states[i], {"q": gold[i]}) for i in range(30)]
    rep1 = v1.fit(examples, q)
    ans1 = v1.decide(states[1], q)["answers"]["q"]
    assert rep1["q"]["fitted"] is True and rep1["q"]["used_by_auto"] is False
    assert 0.333 < rep1["q"]["cv_accuracy"] < rep1["q"]["engine_accuracy"]
    assert ans1["readout"] == "engine" and "prediction_set" in ans1 and "settle_steps" in ans1
    assert v1.decide(states[1], q, mode="ttt")["answers"]["q"]["readout"] == "ttt"

    st_2 = {i: found[gold[i]][0]
            + torch.tensor(rng.standard_normal((2, 3, 12)) * 0.05, dtype=torch.float32)
            for i in range(30)}
    perc2 = _Perception(lambda k, b: st_2[key_to_idx[k]], lambda k, n: found["a"][1])
    v2 = Vega(RealEngine(str(tmp_path), device="cpu", perception=perc2))
    rep2 = v2.fit(examples, q)
    assert rep2["q"]["used_by_auto"] is True
    assert v2.decide(states[1], q)["answers"]["q"]["readout"] == "ttt"

