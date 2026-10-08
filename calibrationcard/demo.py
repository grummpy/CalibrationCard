"""Synthetic predictions with a known amount of overconfidence, for the demo and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ["cat", "dog", "fox", "owl"]


def synthetic_predictions(n: int = 1200, temperature: float = 0.55, seed: int = 7) -> pd.DataFrame:
    """Draw true probabilities, sample labels from them, then sharpen the reported scores (overconfident)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.6, size=(n, len(CLASSES)))
    logits[np.arange(n), rng.integers(0, len(CLASSES), n)] += 1.4
    true_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(len(CLASSES), p=row) for row in true_p])
    shown = np.exp(logits / temperature)
    shown /= shown.sum(axis=1, keepdims=True)
    frame = pd.DataFrame({"label": [CLASSES[i] for i in labels]})
    for k, name in enumerate(CLASSES):
        frame[f"p_{name}"] = shown[:, k].round(6)
    return frame


def synthetic_binary(n: int = 800, seed: int = 3, underconfident: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    shown = 1 / (1 + np.exp(-(z * (0.5 if underconfident else 1.0))))
    return pd.DataFrame({"y_true": y, "score": shown.round(6)})
