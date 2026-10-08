import numpy as np

from calibrationcard.analysis import analyze
from calibrationcard.demo import synthetic_binary, synthetic_predictions
from calibrationcard.loader import load_predictions
from calibrationcard.recalibrate import Temperature, compare, cross_fit


def test_temperature_recovers_sharpening():
    pred = load_predictions(synthetic_predictions(n=3000, temperature=0.5))
    t = Temperature().fit(pred.proba, pred.y).t
    assert 1.6 < t < 2.4  # scores were sharpened by 1/0.5


def test_recalibration_helps_overconfident_and_underconfident_models():
    over = compare(load_predictions(synthetic_predictions()))
    assert over["helps"] and over["best"] in ("temperature", "platt")
    under = compare(load_predictions(synthetic_binary()))
    assert under["helps"]
    best = next(m for m in under["methods"] if m["method"] == under["best"])
    assert best["ece"] < under["before"]["ece"]


def test_cross_fit_never_scores_on_fit_rows():
    pred = load_predictions(synthetic_binary(n=200))
    out, params = cross_fit(pred.proba, pred.y, "isotonic", folds=2, seed=0)
    assert out.shape == pred.proba.shape and np.allclose(out.sum(axis=1), 1)
    assert params["folds"] == 2


def test_tiny_dataset_skips_recalibration():
    pred = load_predictions(synthetic_binary(n=30))
    result = analyze(pred)
    assert result["recalibration"]["skipped"]
    assert any("Recalibration needs" in n for n in result["grade"]["notes"])


def test_already_calibrated_model_is_not_told_to_recalibrate():
    rng = np.random.default_rng(5)
    import pandas as pd

    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)
    result = analyze(load_predictions(pd.DataFrame({"y": y, "score": p})))
    assert result["grade"]["letter"].startswith("A")
    assert not result["recalibration"]["helps"]


def test_no_mistakes_means_no_recalibration_advice():
    import pandas as pd

    y = np.array([0, 1] * 100)
    score = np.where(y == 1, 0.8, 0.2)
    result = analyze(load_predictions(pd.DataFrame({"y": y, "score": score})))
    assert result["recalibration"]["skipped"].startswith("Recalibration skipped")
    assert result["metrics"]["errors"] == 0
