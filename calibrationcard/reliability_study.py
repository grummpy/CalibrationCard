"""Synthetic operating-characteristic checks for CalibrationCard's conventional grade bands.

This is a local teaching/evaluation aid, not an external benchmark or a substitute for a model-specific study.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from calibrationcard.grade import grade
from calibrationcard.loader import Predictions
from calibrationcard.metrics import evaluate


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _synthetic_case(rng, rows: int, prevalence: float, condition: str) -> Predictions:
    """Create a binary calibrated or deliberately sharpened forecast with public synthetic labels."""
    base_logit = np.log(prevalence / (1.0 - prevalence))
    latent = base_logit + rng.normal(0.0, 0.9, size=rows)
    truth_probability = _sigmoid(latent)
    y = (rng.random(rows) < truth_probability).astype(int)
    for _ in range(19):
        if y.min() != y.max():
            break
        y = (rng.random(rows) < truth_probability).astype(int)
    if y.min() == y.max():
        raise ValueError("Synthetic draw has one class; use more rows or a less extreme prevalence.")
    if condition == "calibrated":
        score = truth_probability
    elif condition == "sharpened":
        # The probability is deliberately too extreme while labels still follow truth_probability.
        score = _sigmoid(1.9 * latent)
    else:
        raise ValueError(f"Unknown synthetic condition: {condition}")
    return Predictions(y=y, proba=np.column_stack([1.0 - score, score]), classes=["0", "1"], binary=True,
                       label_column="synthetic_label", prob_columns=["synthetic_score"])


def _mean_ece_bootstrap_interval(eces: list[float], *, rng, rounds: int = 2_000) -> list[float]:
    """Return a 90% bootstrap interval for the mean ECE across synthetic trials."""
    values = np.asarray(eces, dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    samples = values[rng.integers(0, len(values), size=(rounds, len(values)))].mean(axis=1)
    low, high = np.percentile(samples, [5, 95])
    return [float(low), float(high)]


def _grade_summary(grades: list[str], eces: list[float], *, rng) -> dict:
    counts = Counter(item[0] for item in grades)
    return {
        "trials": len(grades),
        "grade_counts": dict(sorted(counts.items())),
        "incomplete_rate": sum(item.startswith("I") for item in grades) / len(grades),
        "mean_ece": float(np.mean(eces)),
        "mean_ece_90_percent_bootstrap_interval": _mean_ece_bootstrap_interval(eces, rng=rng),
    }


def run_grade_reliability_study(*, sample_sizes=(50, 100, 300, 1000), prevalences=(0.1, 0.5),
                                bin_counts=(10, 15, 25), trials: int = 20, seed: int = 0) -> dict:
    """Measure grade behavior on synthetic calibrated and deliberately miscalibrated forecasts.

    `false_alarm_rate` means a calibrated synthetic run receives C/D/F. `missed_gap_rate` means a deliberately
    sharpened synthetic run receives A/B. Both are operating characteristics of these stated simulations, not a
    general reliability guarantee.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    rows: list[dict] = []
    pair_rows: list[dict] = []
    root_rng = np.random.default_rng(seed)
    for n in sample_sizes:
        for prevalence in prevalences:
            if not 0.0 < prevalence < 1.0:
                raise ValueError("prevalences must be strictly between 0 and 1")
            for bins in bin_counts:
                summaries = {}
                for condition in ("calibrated", "sharpened"):
                    grades: list[str] = []
                    eces: list[float] = []
                    for _ in range(trials):
                        pred = _synthetic_case(root_rng, int(n), float(prevalence), condition)
                        metrics = evaluate(pred.proba, pred.y, pred.classes, pred.binary, n_bins=int(bins),
                                           bootstrap_rounds=50, noise_floor_rounds=50,
                                           seed=int(root_rng.integers(0, 2**31 - 1)))
                        card = grade(metrics, None, pred.summary())
                        grades.append(card["letter"])
                        eces.append(metrics["ece"])
                    summary = _grade_summary(grades, eces, rng=root_rng)
                    summary.update({"condition": condition, "rows": int(n), "target_prevalence": float(prevalence),
                                    "bins": int(bins)})
                    rows.append(summary)
                    summaries[condition] = grades
                pair_rows.append({
                    "rows": int(n),
                    "target_prevalence": float(prevalence),
                    "bins": int(bins),
                    "false_alarm_rate": sum(item[0] in "CDF" for item in summaries["calibrated"]) / trials,
                    "missed_gap_rate": sum(item[0] in "AB" for item in summaries["sharpened"]) / trials,
                })
    return {
        "kind": "CalibrationCard synthetic grade-reliability study",
        "synthetic_only": True,
        "seed": seed,
        "parameters": {"sample_sizes": list(sample_sizes), "prevalences": list(prevalences),
                       "bin_counts": list(bin_counts), "trials": trials},
        "definitions": {
            "false_alarm_rate": "Calibrated synthetic run receives C, D, or F.",
            "missed_gap_rate": "Deliberately sharpened synthetic run receives A or B.",
            "incomplete_rate": "Run receives I because evidence is intentionally insufficient.",
            "mean_ece_90_percent_bootstrap_interval": "Percentile bootstrap interval for the mean trial ECE.",
        },
        "conditions": rows,
        "operating_characteristics": pair_rows,
    }
