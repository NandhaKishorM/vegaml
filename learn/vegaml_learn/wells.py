"""Landscape learning (wells): learning by reshaping the energy landscape, as a drop-in for the TTT head.

Vega's engine answers by letting a particle settle into the well of one option. The TTT head learns from
labelled examples by fitting a separate logistic layer on the same frozen features. This learns the same thing
as wells: every option gets a well (its centre is the mean of its examples), all wells share one metric (how
far examples spread around their own centre), and the answer is the Boltzmann occupancy of the wells,

    E_k(z) = (z - c_k)' M (z - c_k) / 2 - log a_k,        p(k | z) = exp(-E_k / T) / sum_j exp(-E_j / T).

Nothing is iterated: a fit is a few sums and one matrix inverse, so it costs milliseconds where the logistic
head costs seconds, and it grows no slower as examples accumulate.

How it compares, measured on three public sales datasets with the vega 4B (ten questions, held-out
conversations): level with the TTT head on average at every size from five examples to thirty thousand, and
ten or more times faster to fit. It is the faster learner, not a more accurate one.

The landscape is drawn in coordinates of the examples themselves: the features are standardised and projected
on their top principal directions, each scaled to unit variance. Its resolution grows with the data (about
4 * sqrt(n) directions for n examples, at most 1024): a landscape with more detail than it has examples to
support it memorises them.

The occupancy has a temperature. Wells fitted in many directions are far too sure of themselves (nearly every
answer above 0.99), and the learner decides what vega may answer alone from that confidence. So each fit also
fits one temperature, on examples the wells did not see (three folds), by held-out likelihood, with a sliver
of doubt that held-out answers being all right cannot remove.
"""

from __future__ import annotations

import numpy as np


def _resolution(n, d, cap=1024):
    """How many directions n examples can support."""
    return int(max(2, min(n - 1, d, cap, round(4 * np.sqrt(n)))))


def _directions(Z, k, seed=0):
    """The top k principal directions of centred rows Z and their standard deviations, by a randomised range
    finder (exact enough here, and it does not pay for a full decomposition of a wide matrix)."""
    n, d = Z.shape
    if n <= 400 or k >= min(n, d) - 1:
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        return Vt[:k].T, S[:k] / np.sqrt(max(n - 1, 1))
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(Z @ rng.standard_normal((d, min(k + 32, min(n, d)))).astype(np.float32))
    for _ in range(2):
        Q, _ = np.linalg.qr(Z @ (Z.T @ Q))
    _, S, Vt = np.linalg.svd(Q.T @ Z, full_matrices=False)
    return Vt[:k].T, S[:k] / np.sqrt(max(n - 1, 1))


class WellsHead:
    """One well per answer over frozen features. Same interface as :class:`vegaml.ttt.TTTHead`."""

    BASIS_ROWS = (
        4000  # the directions are found on at most this many examples; every example then shapes the wells
    )
    TEMPERATURES = tuple(float(t) for t in np.geomspace(1.0, 256.0, 25))

    def __init__(self, labels, cap=1024, shrink=0.3, temperature=None):
        self.labels = list(labels)
        self.cap, self.shrink = cap, shrink
        self.temperature = temperature  # None: fitted on held-out examples at fit time
        self.ready = False
        self.cross_val_n = 0
        self.cross_val_rows = []
        self.cross_val_preds = {}
        self.engine_accuracy = None

    def fit(self, X, y_labels):
        X = np.asarray(X, dtype=np.float32)
        idx = {name: i for i, name in enumerate(self.labels)}
        strs = [str(v) for v in y_labels]
        unknown = sorted({s for s in strs if s not in idx})
        if unknown:
            raise ValueError(
                "label(s) %s are not in this head's label set %s. Fit the head with "
                "label_ids(question) so the two agree."
                % (", ".join(repr(u) for u in unknown[:5]), self.labels)
            )
        y = np.array([idx[s] for s in strs])
        self._settle(X, y)
        self.T = float(self.temperature) if self.temperature else self._held_out_temperature(X, y)
        self.ready = True
        return self

    def _held_out_temperature(self, X, y):
        """The temperature that makes wells least surprised by examples they did not see."""
        n = len(y)
        if n < 12:
            return 4.0  # too few to hold any out: stay modest
        fold = np.random.default_rng(0).permutation(n) % 3
        scores, gold = [], []
        for f in range(3):
            tr, te = fold != f, fold == f
            if len(set(y[tr].tolist())) < 2:
                continue
            h = WellsHead(self.labels, self.cap, self.shrink, temperature=1.0)
            h._settle(X[tr], y[tr])
            col = {c: j for j, c in enumerate(h.present)}
            seen = np.array([c in col for c in y[te]])
            if not seen.any():
                continue
            S = (h._place(X[te][seen]) @ h.A.T + h.b).astype(np.float64)
            scores.append(S - S.max(1, keepdims=True))
            gold.append(np.array([col[c] for c in y[te][seen]]))
        if not scores:
            return 4.0
        # With m held-out answers all right, the wells still cannot claim to be right more often than
        # (m + 1) / (m + 2). So a sliver of each held-out example (1 / (m + 2)) is counted as if the runner-up
        # had been the answer: perfect held-out scores then cool the wells instead of leaving them at
        # certainty, and a confidence stays a number that can rank answers.
        m = sum(len(g) for g in gold)
        doubt = 1.0 / (m + 2)
        runner = []
        for S, g in zip(scores, gold, strict=True):
            R = S.copy()
            R[np.arange(len(g)), g] = -np.inf
            runner.append(R.argmax(1))
        best, best_t = None, 4.0
        for t in self.TEMPERATURES:
            nll = 0.0
            for S, g, r in zip(scores, gold, runner, strict=True):
                Z = S / t
                logz = np.log(np.exp(Z).sum(1))
                rows = np.arange(len(g))
                nll += float(((1 - doubt) * (logz - Z[rows, g]) + doubt * (logz - Z[rows, r])).sum())
            if best is None or nll < best:
                best, best_t = nll, t
        return best_t

    def _settle(self, X, y):
        n = len(y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        if n > self.BASIS_ROWS:  # a fixed subsample: the same data always gives the same wells
            pick = np.sort(np.random.default_rng(0).choice(n, self.BASIS_ROWS, replace=False))
            Zb = (X[pick] - self.mu) / self.sd
        else:
            Zb = (X - self.mu) / self.sd
        k = _resolution(n, X.shape[1], self.cap)
        self.P, s = _directions(Zb, min(k, len(Zb) - 1))
        self.s = s.astype(np.float32) + 1e-6
        k = self.P.shape[1]
        Z = self._place(X).astype(np.float64)
        present = sorted(set(y.tolist()))
        counts = np.array([(y == c).sum() for c in present], dtype=np.float64)
        centres = np.stack([Z[y == c].mean(0) for c in present])
        within = Z.T @ Z - (centres * counts[:, None]).T @ centres
        within = within / (n - len(present)) if n > len(present) else np.eye(k)
        within = (1 - self.shrink) * within + self.shrink * (np.trace(within) / k) * np.eye(
            k
        )  # toward a round well
        M = np.linalg.inv(within)
        self.A = (centres @ M).astype(np.float32)
        self.b = (
            -0.5 * np.einsum("ij,jk,ik->i", centres, M, centres) + np.log(counts / counts.sum())
        ).astype(np.float32)
        self.present = present  # an answer with no example yet has no well, and gets no occupancy

    def _place(self, X):
        return (((np.asarray(X, dtype=np.float32) - self.mu) / self.sd) @ self.P) / self.s

    def predict(self, x, allowed=None):
        if not self.ready:
            raise RuntimeError("this WellsHead has not been fitted yet; call fit(X, y) first")
        s = (self._place(np.asarray(x, dtype=np.float32).reshape(1, -1)) @ self.A.T + self.b)[0].astype(
            np.float64
        )
        s = s / self.T
        e = np.exp(s - s.max())
        occ = e / e.sum()
        out = {name: 0.0 for name in self.labels}
        for c, p in zip(self.present, occ, strict=True):
            out[self.labels[c]] = float(p)
        if allowed is not None:  # restrict to the options this case actually offers
            keep = set(allowed)
            out = {k: v for k, v in out.items() if k in keep}
            if not out:
                raise ValueError(
                    "no overlap between the labels this head knows (%s) and the options "
                    "this question offers (%s). Refit with fit(examples, questions) for "
                    "the question as it is now written." % (self.labels, sorted(keep))
                )
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()}

    def cross_val_accuracy(self, X, y_labels, folds=5):
        """Every example scored by wells that never saw it. Same folds, and the same refusals, as the TTT
        head: stratified by label, classes too rare to appear in every fold are left out, None when there is
        too little data to say anything."""
        X = np.asarray(X, dtype=np.float32)
        y = [str(v) for v in y_labels]
        self.cross_val_n = 0
        self.cross_val_rows = []
        self.cross_val_preds = {}
        if len(y) < 2 * folds:
            return None
        by_class = {}
        for i, lab in enumerate(y):
            by_class.setdefault(lab, []).append(i)
        keep = sorted(lab for lab, members in by_class.items() if len(members) >= folds)
        if len(keep) < 2:
            return None
        rows, rows_y = [], []
        for lab in keep:
            for i in by_class[lab]:
                rows.append(i)
                rows_y.append(lab)
        rows = np.asarray(rows, dtype=int)
        where = np.empty(len(y), dtype=int)
        where[rows] = np.arange(len(rows))
        rng = np.random.default_rng(0)
        fold_of = np.empty(len(rows), dtype=int)
        for lab in keep:
            members = np.asarray(by_class[lab], dtype=int)
            positions = where[members]
            for rank, member in enumerate(rng.permutation(len(members))):
                fold_of[positions[member]] = rank % folds
        hit, scored = 0, 0
        scored_rows = []
        for f in range(folds):
            te = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            if len(tr) == 0 or len({rows_y[i] for i in tr}) < 2:
                continue
            h = WellsHead(self.labels, self.cap, self.shrink, temperature=1.0).fit(
                X[rows[tr]], [rows_y[i] for i in tr]
            )
            S = h._place(X[rows[te]]) @ h.A.T + h.b  # the answer is the deepest well whatever the temperature
            for i, j in zip(te, S.argmax(1), strict=True):
                best = self.labels[h.present[int(j)]]
                hit += int(best == rows_y[i])
                scored_rows.append(int(rows[i]))
                self.cross_val_preds[int(rows[i])] = best
            scored += len(te)
        self.cross_val_n = scored
        self.cross_val_rows = scored_rows
        return hit / scored if scored else None
