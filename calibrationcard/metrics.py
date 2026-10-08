"""Calibration metrics: reliability bins, ECE, MCE, adaptive ECE, classwise ECE, Brier score, log loss."""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def top_label(proba: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Confidence of the predicted class and whether that prediction was right."""
    pred = np.argmax(proba, axis=1)
    return proba[np.arange(len(proba)), pred], (pred == y).astype(float)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, *, lower: float = 0.0) -> list[dict]:
    """Equal-width bins on [lower, 1]. Each bin: range, count, mean confidence, accuracy, gap."""
    edges = np.linspace(lower, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    return _bins_from_index(conf, correct, idx, edges)


def adaptive_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Equal-mass bins (about the same number of rows in each)."""
    n = len(conf)
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(conf, kind="mergesort")
    idx = np.empty(n, dtype=int)
    for b, chunk in enumerate(np.array_split(order, n_bins)):
        idx[chunk] = b
    edges = [float(conf[chunk].min()) if len(chunk) else 0.0 for chunk in np.array_split(order, n_bins)]
    edges.append(1.0)
    return _bins_from_index(conf, correct, idx, np.array(edges))


def _bins_from_index(conf, correct, idx, edges) -> list[dict]:
    out = []
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        mean_conf = float(conf[mask].mean()) if count else None
        acc = float(correct[mask].mean()) if count else None
        out.append({
            "low": float(edges[b]),
            "high": float(edges[b + 1]),
            "count": count,
            "confidence": mean_conf,
            "accuracy": acc,
            "gap": None if not count else float(mean_conf - acc),
        })
    return out


def ece_from_bins(bins: list[dict], n: int) -> float:
    return float(sum(b["count"] / n * abs(b["gap"]) for b in bins if b["count"])) if n else 0.0


def mce_from_bins(bins: list[dict], min_count: int = 1) -> float:
    gaps = [abs(b["gap"]) for b in bins if b["count"] >= min_count]
    return float(max(gaps)) if gaps else 0.0


def signed_gap(bins: list[dict], n: int) -> float:
    """Positive means overconfident (confidence above accuracy), negative means underconfident."""
    return float(sum(b["count"] / n * b["gap"] for b in bins if b["count"])) if n else 0.0


def brier(proba: np.ndarray, y: np.ndarray, binary: bool) -> float:
    if binary:
        return float(np.mean((proba[:, 1] - (y == 1)) ** 2))
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def log_loss(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def classwise(proba: np.ndarray, y: np.ndarray, classes: list[str], n_bins: int = 10) -> list[dict]:
    """One-vs-rest calibration for each class: does P(class) match how often it is the class?"""
    rows = []
    for k, name in enumerate(classes):
        p = proba[:, k]
        hit = (y == k).astype(float)
        bins = reliability_bins(p, hit, n_bins)
        rows.append({
            "class": name,
            "support": int(hit.sum()),
            "mean_probability": float(p.mean()),
            "frequency": float(hit.mean()),
            "ece": ece_from_bins(bins, len(p)),
            "signed_gap": signed_gap(bins, len(p)),
            "bins": bins,
        })
    return rows


def histogram(conf: np.ndarray, n_bins: int = 20) -> list[dict]:
    counts, edges = np.histogram(conf, bins=n_bins, range=(0.0, 1.0))
    return [{"low": float(edges[i]), "high": float(edges[i + 1]), "count": int(c)} for i, c in enumerate(counts)]


def bootstrap_ece(conf, correct, n_bins: int, rounds: int = 200, seed: int = 0, lower: float = 0.0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(conf)
    values = []
    for _ in range(rounds):
        pick = rng.integers(0, n, n)
        bins = reliability_bins(conf[pick], correct[pick], n_bins, lower=lower)
        values.append(ece_from_bins(bins, n))
    low, high = np.percentile(values, [5, 95])
    return float(low), float(high)


def noise_floor(conf: np.ndarray, n_bins: int, lower: float = 0.0, rounds: int = 200, seed: int = 0) -> float:
    """Average ECE a perfectly calibrated model with these same confidences would show, from sampling noise alone.

    Each round draws "right or wrong" with probability equal to the stated confidence, then measures ECE.
    Small datasets have a noticeable floor; large ones have almost none.
    """
    n = len(conf)
    if n == 0:
        return 0.0
    rounds = int(max(20, min(rounds, 5_000_000 // max(n, 1))))
    rng = np.random.default_rng(seed)
    edges = np.linspace(lower, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    total = 0.0
    for _ in range(rounds):
        hits = (rng.random(n) < conf).astype(float)
        hit_sum = np.bincount(idx, weights=hits, minlength=n_bins)
        total += float(np.abs(conf_sum - hit_sum).sum() / n)
    return total / rounds


def evaluate(proba: np.ndarray, y: np.ndarray, classes: list[str], binary: bool, *, n_bins: int = 15,
             seed: int = 0, bootstrap: bool = True) -> dict:
    n = len(y)
    conf, correct = top_label(proba, y)
    # Top-label confidence can never be below 1/k, so start the bins there.
    lower = 1.0 / proba.shape[1]
    bins = reliability_bins(conf, correct, n_bins, lower=lower)
    abins = adaptive_bins(conf, correct, n_bins)
    cw = classwise(proba, y, classes, n_bins=10)
    result = {
        "n": int(n),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "ece": ece_from_bins(bins, n),
        "mce": mce_from_bins(bins, min_count=max(1, n // 100)),
        "adaptive_ece": ece_from_bins(abins, n),
        "classwise_ece": float(np.mean([c["ece"] for c in cw])),
        "signed_gap": signed_gap(bins, n),
        "brier": brier(proba, y, binary),
        "log_loss": log_loss(proba, y),
        "errors": int(n - correct.sum()),
        "bins": bins,
        "adaptive_bins": abins,
        "histogram": histogram(conf),
        "classwise": cw,
        "n_bins": n_bins,
        "bin_floor": lower,
    }
    if binary:
        # For a binary model the positive-class reliability curve is the standard view.
        result["positive_bins"] = cw[1]["bins"]
    floor = noise_floor(conf, n_bins, lower=lower, seed=seed)
    result["ece_noise_floor"] = floor
    result["ece_excess"] = max(0.0, result["ece"] - floor)
    if bootstrap and n >= 10:
        result["ece_interval"] = list(bootstrap_ece(conf, correct, n_bins, seed=seed, lower=lower))
    return result


def count_errors(k: int) -> str:
    """'no wrong predictions', 'only 1 wrong prediction', 'only 7 wrong predictions'."""
    if k == 0:
        return "no wrong predictions"
    return f"only {k} wrong prediction" + ("" if k == 1 else "s")
