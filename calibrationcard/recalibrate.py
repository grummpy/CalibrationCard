"""Post-hoc recalibration, cross-fitted so every row is scored by a calibrator that never saw it."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from calibrationcard.metrics import count_errors, evaluate

EPS = 1e-12
MIN_ERRORS = 10  # below this a calibrator mostly learns "be more certain", which does not generalize
METHODS = ("temperature", "platt", "isotonic")
LABELS = {"temperature": "Temperature scaling", "platt": "Platt scaling", "isotonic": "Isotonic regression"}


def _logits(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Temperature:
    def fit(self, proba, y):
        z = _logits(proba)

        def nll(log_t):
            p = _softmax(z / np.exp(log_t))
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)))

        self.t = float(np.exp(minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded").x))
        return self

    def transform(self, proba):
        return _softmax(_logits(proba) / self.t)


class Platt:
    """Binary: logistic regression on the logit. Multiclass: one-vs-rest Platt, then rows rescaled to sum to 1."""

    def fit(self, proba, y):
        k = proba.shape[1]
        self.models = []
        columns = [1] if k == 2 else range(k)
        for c in columns:
            target = (y == c).astype(int)
            x = _logit_col(proba[:, c])
            if target.min() == target.max():
                self.models.append(("const", float(target.mean())))
                continue
            model = LogisticRegression(C=1e4, max_iter=1000)
            model.fit(x, target)
            self.models.append(("lr", model))
        return self

    def transform(self, proba):
        k = proba.shape[1]
        if k == 2:
            p1 = _apply(self.models[0], _logit_col(proba[:, 1]))
            return np.column_stack([1 - p1, p1])
        cols = [_apply(m, _logit_col(proba[:, c])) for c, m in enumerate(self.models)]
        return _rows_to_one(np.column_stack(cols))


class Isotonic:
    def fit(self, proba, y):
        k = proba.shape[1]
        self.models = []
        columns = [1] if k == 2 else range(k)
        for c in columns:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(proba[:, c], (y == c).astype(float))
            self.models.append(iso)
        return self

    def transform(self, proba):
        k = proba.shape[1]
        if k == 2:
            p1 = self.models[0].predict(proba[:, 1])
            return np.column_stack([1 - p1, p1])
        return _rows_to_one(np.column_stack([m.predict(proba[:, c]) for c, m in enumerate(self.models)]))


def _logit_col(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _apply(model, x):
    kind, obj = model
    if kind == "const":
        return np.full(len(x), obj)
    return obj.predict_proba(x)[:, 1]


def _rows_to_one(p):
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


MAKERS = {"temperature": Temperature, "platt": Platt, "isotonic": Isotonic}


def cross_fit(proba, y, method: str, folds: int = 2, seed: int = 0):
    """Return recalibrated probabilities for every row plus the fitted parameter summary."""
    counts = np.bincount(y, minlength=proba.shape[1])
    present = counts[counts > 0]
    folds = int(min(folds, present.min())) if len(present) else 0
    if folds < 2:
        raise ValueError("Each class needs at least 2 rows to cross-fit a calibrator.")
    out = np.zeros_like(proba)
    params = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_idx, eval_idx in splitter.split(proba, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MAKERS[method]().fit(proba[fit_idx], y[fit_idx])
        out[eval_idx] = model.transform(proba[eval_idx])
        if method == "temperature":
            params.append(round(model.t, 4))
    return out, {"folds": folds, "temperature": params} if method == "temperature" else {"folds": folds}


def compare(pred, methods=METHODS, folds: int = 2, seed: int = 0, n_bins: int = 15) -> dict:
    if pred.n < 40:
        return {"skipped": f"Only {pred.n} rows. Recalibration needs at least 40 to say anything useful.", "methods": []}
    base = evaluate(pred.proba, pred.y, pred.classes, pred.binary, n_bins=n_bins, seed=seed, bootstrap=False)
    if base["errors"] < MIN_ERRORS:
        return {"skipped": f"Recalibration skipped: with {count_errors(base['errors'])} a calibrator can only "
                           "learn to be more certain, which would likely be overconfident on new data.",
                "methods": [], "before": {k: base[k] for k in ("ece", "adaptive_ece", "brier", "log_loss", "accuracy")}}
    rows = []
    for method in methods:
        try:
            new_p, params = cross_fit(pred.proba, pred.y, method, folds=folds, seed=seed)
        except ValueError as exc:
            rows.append({"method": method, "label": LABELS[method], "error": str(exc)})
            continue
        after = evaluate(new_p, pred.y, pred.classes, pred.binary, n_bins=n_bins, seed=seed, bootstrap=False)
        rows.append({
            "method": method,
            "label": LABELS[method],
            "params": params,
            "ece": after["ece"],
            "adaptive_ece": after["adaptive_ece"],
            "brier": after["brier"],
            "log_loss": after["log_loss"],
            "accuracy": after["accuracy"],
            "bins": after["bins"],
            "positive_bins": after.get("positive_bins"),
        })
    ok = [r for r in rows if "error" not in r]
    # Pick by log loss (a proper scoring rule). ECE alone rewards isotonic's 0/1 jumps that blow up log loss.
    best = min(ok, key=lambda r: (r["log_loss"], r["ece"])) if ok else None
    return {
        "folds": folds,
        "before": {k: base[k] for k in ("ece", "adaptive_ece", "brier", "log_loss", "accuracy")},
        "methods": rows,
        "best": best["method"] if best else None,
        "helps": bool(best and best["ece"] < base["ece"] * 0.8 and best["log_loss"] <= base["log_loss"] * 1.02),
    }
