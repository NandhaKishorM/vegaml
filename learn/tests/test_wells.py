"""The wells learner: one well per answer in the energy landscape, a drop-in for the TTT head."""

import numpy as np
import pytest

from vegaml.ttt import TTTHead
from vegaml_learn import learner as LN
from vegaml_learn.wells import WellsHead, _resolution


def blobs(n, d=64, k=3, spread=0.6, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, d)) * 2
    y = rng.integers(0, k, n)
    X = centres[y] + spread * rng.standard_normal((n, d))
    return X.astype(np.float32), [("a", "b", "c", "d")[i] for i in y]


def test_it_learns_what_the_head_learns():
    X2, y2 = blobs(500, seed=0)  # 300 to learn from, 200 held out
    w = WellsHead(["a", "b", "c"]).fit(X2[:300], y2[:300])
    t = TTTHead(["a", "b", "c"]).fit(X2[:300], y2[:300])

    def right(h):
        return np.mean(
            [max(p := h.predict(x), key=p.get) == g for x, g in zip(X2[300:], y2[300:], strict=True)]
        )

    assert right(w) > 0.95
    assert abs(right(w) - right(t)) < 0.05


def test_an_answer_is_an_occupancy_over_the_options_offered():
    X, y = blobs(200)
    h = WellsHead(["a", "b", "c"]).fit(X, y)
    p = h.predict(X[0])
    assert set(p) == {"a", "b", "c"} and abs(sum(p.values()) - 1) < 1e-6
    q = h.predict(X[0], allowed=["a", "b"])
    assert set(q) == {"a", "b"} and abs(sum(q.values()) - 1) < 1e-6
    with pytest.raises(ValueError, match="no overlap"):
        h.predict(X[0], allowed=["z"])


def test_an_answer_with_no_example_has_no_well():
    X, y = blobs(200)  # only a, b, c ever appear
    h = WellsHead(["a", "b", "c", "d"]).fit(X, y)
    assert all(h.predict(x)["d"] == 0.0 for x in X[:20])


def test_it_refuses_a_label_it_was_not_given_and_an_answer_before_it_has_learned():
    X, y = blobs(50)
    with pytest.raises(ValueError, match="not in this head's label set"):
        WellsHead(["a", "b"]).fit(X, y)
    with pytest.raises(RuntimeError, match="not been fitted"):
        WellsHead(["a", "b", "c"]).predict(X[0])


def test_the_same_examples_give_the_same_wells():
    X, y = blobs(900, d=128)  # past 400 rows the directions come from a randomised range finder
    a = WellsHead(["a", "b", "c"]).fit(X, y)
    b = WellsHead(["a", "b", "c"]).fit(X, y)
    assert np.allclose(a.A, b.A) and np.allclose(a.b, b.b)
    assert a.predict(X[3]) == b.predict(X[3])


def test_resolution_grows_with_the_examples_and_never_past_them():
    assert _resolution(5, 15360) == 4
    assert _resolution(100, 15360) == 40
    assert _resolution(10_000, 15360) == 400
    assert _resolution(1_000_000, 15360) == 1024
    assert _resolution(1_000_000, 300) == 300


def test_cross_validation_scores_every_example_with_wells_that_never_saw_it():
    X, y = blobs(120)
    h = WellsHead(["a", "b", "c"]).fit(X, y)
    acc = h.cross_val_accuracy(X, y)
    assert acc is not None and acc > 0.9
    assert h.cross_val_n == 120 and sorted(h.cross_val_rows) == list(range(120))
    assert set(h.cross_val_preds) == set(range(120))
    assert WellsHead(["a", "b", "c"]).cross_val_accuracy(X[:8], y[:8]) is None  # too little to say anything


def test_the_learner_uses_the_head_by_default_and_wells_when_asked(monkeypatch):
    X, y = blobs(60)
    q = {"type": "choice", "instructions": "which", "criteria": {"a": "a", "b": "b", "c": "c"}}
    for asked in (None, "ttt"):
        if asked is None:
            monkeypatch.delenv("VEGA_LEARN_HEAD", raising=False)
        else:
            monkeypatch.setenv("VEGA_LEARN_HEAD", asked)
        h = LN.fit_head(q, X, y, [""] * len(y))
        assert isinstance(h["head"], TTTHead) and h["kind"] == "ttt"
    for asked in ("wells", "landscape", " Landscape "):
        monkeypatch.setenv("VEGA_LEARN_HEAD", asked)
        h = LN.fit_head(q, X, y, [""] * len(y))
        assert isinstance(h["head"], WellsHead) and h["kind"] == "wells" and h["usable"], asked
    monkeypatch.setenv("VEGA_LEARN_HEAD", "valleys")
    with pytest.raises(ValueError, match="ttt or wells"):
        LN.fit_head(q, X, y, [""] * len(y))


def test_a_temperature_takes_false_certainty_out_and_changes_no_answer():
    rng = np.random.default_rng(3)  # hard: two weak directions of signal among 400 of noise, 150 examples
    y = rng.integers(0, 2, 450)
    X = rng.standard_normal((450, 400)).astype(np.float32)
    X[:, :2] += 0.9 * (2 * y[:, None] - 1)
    lab = [("a", "b")[i] for i in y]
    h = WellsHead(["a", "b"]).fit(X[:150], lab[:150])
    cold = WellsHead(["a", "b"], temperature=1.0).fit(X[:150], lab[:150])

    def sure_and_right(m):
        P = [m.predict(x) for x in X[150:]]
        conf = np.mean([max(p.values()) for p in P])
        acc = np.mean([max(p, key=p.get) == g for p, g in zip(P, lab[150:], strict=True)])
        return conf, acc

    (c1, a1), (c0, a0) = sure_and_right(h), sure_and_right(cold)
    assert a1 == a0  # a temperature never changes which well is deepest
    assert h.T > 1.0 and c1 < c0  # and it takes the false certainty out


def test_perfect_held_out_answers_still_leave_room_for_doubt():
    X, y = blobs(300, spread=0.2)  # easy: every held-out example is answered right
    h = WellsHead(["a", "b", "c"]).fit(X, y)
    conf = [max(h.predict(x).values()) for x in X]
    assert h.T > 1.0
    assert max(conf) < 1.0 and len({round(c, 6) for c in conf}) > 50  # still a number that ranks answers


def test_the_status_says_which_learner_fitted_a_question(monkeypatch, tmp_path):
    import test_learn as TL

    monkeypatch.setenv("VEGA_LEARN_HEAD", "wells")
    learner = TL.make(tmp_path)[0]
    st = learner.status()["questions"]
    fitted = {v["learner"] for v in st.values() if v["readout"] == "head"}
    assert fitted == {"wells"}
