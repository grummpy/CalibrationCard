"""Run the full analysis on a Predictions object."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time

from calibrationcard import __version__
from calibrationcard.grade import grade
from calibrationcard.loader import Predictions
from calibrationcard.metrics import evaluate
from calibrationcard.recalibrate import METHODS, compare, held_out_final_evaluation


def _provenance(pred: Predictions, settings: dict) -> dict:
    """Record enough local evidence to reproduce a report without uploading the predictions."""
    digest = hashlib.sha256()
    digest.update(pred.y.tobytes())
    digest.update(pred.proba.tobytes())
    digest.update(json.dumps(pred.classes, ensure_ascii=False, separators=(",", ":")).encode())
    configuration = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "flask"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    return {
        "input_sha256": digest.hexdigest(),
        "configuration_sha256": hashlib.sha256(configuration).hexdigest(),
        "class_order": list(pred.classes),
        "package_version": __version__,
        "python": platform.python_version(),
        "dependencies": packages,
        "source_revision": os.environ.get("CALIBRATIONCARD_REVISION") or os.environ.get("GITHUB_SHA") or "unavailable",
    }


def analyze(pred: Predictions, *, name: str = "predictions", n_bins: int = 15, seed: int = 0,
            recalibrate: bool = True, methods=METHODS, folds: int = 2,
            final_evaluation_fraction: float | None = None) -> dict:
    start = time.perf_counter()
    settings = {"bins": n_bins, "seed": seed, "folds": folds, "methods": list(methods) if recalibrate else [],
                "final_evaluation_fraction": final_evaluation_fraction}
    if recalibrate and final_evaluation_fraction is not None:
        recal = held_out_final_evaluation(pred, methods=methods, folds=folds, seed=seed, n_bins=n_bins,
                                          final_fraction=final_evaluation_fraction)
        metrics = recal["final_evaluation"]["raw"]
    else:
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
        "settings": settings,
        "provenance": _provenance(pred, settings),
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
