"""TTT: a cross-validated linear head on Vega's frozen perception features.

Vega's engine already computes pooled features from the frozen backbone. TTT reads those same features
and decides with a small logistic head instead of the physics readout. The head costs one matmul
(~0.2 ms, measured 169 us on a T4), so a TTT decision is the same price as an engine decision.

What it needs that the engine does not: labelled examples for the question you are asking. The head is
fitted per label set, which is why it cannot be a drop-in for an arbitrary one-off question.
"""
from __future__ import annotations

import numpy as np


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _fit(Z, y, C, l2, iters=400, lr=1.0):
    W = np.zeros((Z.shape[1], C), np.float32)
    b = np.zeros(C, np.float32)
    Y = np.zeros((len(y), C), np.float32)
    Y[np.arange(len(y)), y] = 1
    for _ in range(iters):
        P = _softmax(Z @ W + b)
        G = ((P - Y) / len(y)).astype(np.float32)
        W -= lr * (Z.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


class TTTHead:
    """PCA + multinomial logistic head over frozen features."""

    def __init__(self, labels, k=128, l2=1e-2):
        self.labels = list(labels)
        self.k, self.l2 = k, l2
        self.ready = False
        # Set by cross_val_accuracy: how many examples the last estimate actually scored, which is
        # fewer than len(X) when a class was too rare to stratify. Defined from the start so that
        # reading it is never an AttributeError.
        self.cross_val_n = 0
        self.cross_val_rows = []
        self.engine_accuracy = None

    def fit(self, X, y_labels):
        X = np.asarray(X, dtype=np.float32)
        idx = {name: i for i, name in enumerate(self.labels)}
        strs = [str(v) for v in y_labels]
        unknown = sorted({s for s in strs if s not in idx})
        if unknown:
            # A bare KeyError here names only the offending label and leaves the caller unable to
            # tell whether they passed a bad label or constructed the head wrong.
            raise ValueError("label(s) %s are not in this head's label set %s. Fit the head with "
                             "label_ids(question) so the two agree."
                             % (", ".join(repr(u) for u in unknown[:5]), self.labels))
        y = np.array([idx[s] for s in strs])
        self.cross_val_n = 0
        self.cross_val_rows = []
        self.engine_accuracy = None
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.c = Z.mean(0)
        k = min(self.k, max(2, min(Z.shape) - 1))
        _, _, Vt = np.linalg.svd(Z - self.c, full_matrices=False)
        self.P = Vt[:k].T
        Zp = (Z - self.c) @ self.P
        self.s = Zp.std(0) + 1e-6
        self.W, self.b = _fit(Zp / self.s, y, len(self.labels), self.l2)
        self.ready = True
        return self

    def predict(self, x, allowed=None):
        if not self.ready:
            raise RuntimeError("this TTTHead has not been fitted yet; call fit(X, y) first")
        Z = (np.asarray(x, dtype=np.float32).reshape(1, -1) - self.mu) / self.sd
        Zp = ((Z - self.c) @ self.P) / self.s
        p = _softmax(Zp @ self.W + self.b)[0]
        out = {name: float(p[i]) for i, name in enumerate(self.labels)}
        if allowed is not None:         # restrict to the options this case actually offers
            keep = set(allowed)
            out = {k: v for k, v in out.items() if k in keep}
            if not out:
                # Renormalising an empty set hands back {} and the caller's next line is
                # max(probs, key=probs.get), which raises "max() arg is an empty sequence" and
                # names neither the head nor the question.
                raise ValueError("no overlap between the labels this head knows (%s) and the options "
                                 "this question offers (%s). Refit with fit(examples, questions) for "
                                 "the question as it is now written."
                                 % (self.labels, sorted(keep)))
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()}

    def cross_val_accuracy(self, X, y_labels, folds=5):
        """Honest estimate: every example scored by a head that never saw it.

        Folds are stratified by label, so every training fold sees every class with enough examples
        to appear in one. Unstratified folds are cut from a single permutation, so a class with few
        examples can be missing from a training fold entirely -- and the head for that fold is then
        fitted on a label space it cannot predict.

        A class too rare to stratify is dropped rather than allowed to distort the estimate; the
        returned accuracy is over the examples actually scored, and that count is ``cross_val_n``.
        Returns None when there is too little data to say anything, rather than a number that
        flatters the head.
        """
        X = np.asarray(X, dtype=np.float32)
        y = [str(v) for v in y_labels]
        self.cross_val_n = 0
        self.cross_val_rows = []
        if len(y) < 2 * folds:
            return None
        by_class = {}
        for i, lab in enumerate(y):
            by_class.setdefault(lab, []).append(i)
        keep = sorted(lab for lab, members in by_class.items() if len(members) >= folds)
        if len(keep) < 2:
            return None       # no class is common enough for every fold to train on one
        rows, rows_y = [], []
        for lab in keep:
            for i in by_class[lab]:
                rows.append(i)
                rows_y.append(lab)
        rows = np.asarray(rows, dtype=int)
        where = np.empty(len(y), dtype=int)
        where[rows] = np.arange(len(rows))          # original row index -> position within rows
        rng = np.random.default_rng(0)              # fixed seed: the same data always gives the same number
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
                continue          # a single-class training set teaches nothing; do not score it
            h = TTTHead(self.labels, self.k, self.l2).fit(X[rows[tr]], [rows_y[i] for i in tr])
            for i in te:
                p = h.predict(X[rows[i]])
                hit += int(max(p, key=p.get) == rows_y[i])
                scored_rows.append(int(rows[i]))
            scored += len(te)
        self.cross_val_n = scored
        self.cross_val_rows = scored_rows
        return hit / scored if scored else None
