"""Run the full analysis on a Predictions object."""

from __future__ import annotations

import time

from calibrationcard.grade import grade
from calibrationcard.loader import Predictions
from calibrationcard.metrics import evaluate
from calibrationcard.recalibrate import METHODS, compare


def analyze(pred: Predictions, *, name: str = "predictions", n_bins: int = 15, seed: int = 0,
            recalibrate: bool = True, methods=METHODS, folds: int = 2) -> dict:
    start = time.perf_counter()
    metrics = evaluate(pred.proba, pred.y, pred.classes, pred.binary, n_bins=n_bins, seed=seed)
    recal = compare(pred, methods=methods, folds=folds, seed=seed, n_bins=n_bins) if recalibrate else None
    summary = pred.summary()
    card = grade(metrics, recal, summary)
    return {
        "name": name,
        "summary": summary,
        "metrics": metrics,
        "recalibration": recal,
        "grade": card,
        "seconds": round(time.perf_counter() - start, 3),
        "settings": {"bins": n_bins, "seed": seed, "folds": folds, "methods": list(methods) if recalibrate else []},
    }


def headline_numbers(result: dict) -> dict:
    m = result["metrics"]
    out = {
        "name": result["name"],
        "rows": m["n"],
        "classes": len(result["summary"]["classes"]),
        "grade": result["grade"]["letter"],
        "accuracy": round(m["accuracy"], 6),
        "ece": round(m["ece"], 6),
        "ece_interval": [round(v, 6) for v in m.get("ece_interval", [])],
        "ece_noise_floor": round(m["ece_noise_floor"], 6),
        "ece_excess": round(m["ece_excess"], 6),
        "adaptive_ece": round(m["adaptive_ece"], 6),
        "mce": round(m["mce"], 6),
        "brier": round(m["brier"], 6),
        "log_loss": round(m["log_loss"], 6),
    }
    recal = result.get("recalibration") or {}
    if recal.get("best"):
        best = next(r for r in recal["methods"] if r["method"] == recal["best"])
        out["best_recalibration"] = {"method": best["method"], "ece": round(best["ece"], 6),
                                     "log_loss": round(best["log_loss"], 6), "helps": recal["helps"]}
    return out
