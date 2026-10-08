import numpy as np

from calibrationcard.demo import synthetic_predictions
from calibrationcard.grade import letter
from calibrationcard.loader import load_predictions
from calibrationcard.metrics import adaptive_bins, brier, evaluate, log_loss, reliability_bins, top_label


def test_perfectly_calibrated_binary_has_low_ece():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    proba = np.column_stack([1 - p, p])
    m = evaluate(proba, y, ["0", "1"], True, seed=1)
    assert m["ece"] < 0.015
    assert abs(m["signed_gap"]) < 0.01


def test_known_overconfidence_is_detected():
    pred = load_predictions(synthetic_predictions())
    m = evaluate(pred.proba, pred.y, pred.classes, pred.binary)
    assert m["ece"] > 0.07 and m["signed_gap"] > 0.07
    assert m["ece_interval"][0] <= m["ece"] <= m["ece_interval"][1]


def test_hand_computed_values():
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    y = np.array([0, 1, 1, 0])
    assert np.isclose(brier(proba, y, True), np.mean([0.1**2, 0.8**2, 0.3**2, 0.4**2]))
    assert np.isclose(log_loss(proba, y), -np.mean(np.log([0.9, 0.2, 0.7, 0.6])))
    conf, correct = top_label(proba, y)
    assert conf.tolist() == [0.9, 0.8, 0.7, 0.6] and correct.tolist() == [1, 0, 1, 1]
    bins = reliability_bins(conf, correct, n_bins=5, lower=0.5)
    filled = [b for b in bins if b["count"]]
    assert sum(b["count"] for b in filled) == 4
    # ECE by hand: each bin has one row -> mean |conf - correct| = (0.1 + 0.8 + 0.3 + 0.4) / 4
    m = evaluate(proba, y, ["0", "1"], True, n_bins=5, bootstrap=False)
    assert np.isclose(m["ece"], (0.1 + 0.8 + 0.3 + 0.4) / 4)


def test_multiclass_brier_is_sum_over_classes():
    proba = np.array([[0.5, 0.3, 0.2]])
    assert np.isclose(brier(proba, np.array([0]), False), 0.25 + 0.09 + 0.04)


def test_adaptive_bins_have_equal_mass():
    conf = np.linspace(0.5, 1, 100)
    bins = adaptive_bins(conf, np.ones(100), n_bins=10)
    assert {b["count"] for b in bins} == {10}


def test_letter_grades():
    assert letter(0.004) == "A+"
    assert letter(0.019) == "A-"
    assert letter(0.025) == "B+"
    assert letter(0.12) == "D"
    assert letter(0.3) == "F"


def test_noise_floor_shrinks_with_more_rows():
    from calibrationcard.metrics import noise_floor

    rng = np.random.default_rng(2)
    small = noise_floor(rng.uniform(0.5, 1, 150), 15, lower=0.5)
    large = noise_floor(rng.uniform(0.5, 1, 20000), 15, lower=0.5)
    assert small > 0.03 and large < 0.01


def test_incomplete_grade_for_tiny_or_error_free_sets():
    import pandas as pd

    from calibrationcard.analysis import analyze

    small = analyze(load_predictions(pd.DataFrame({"y": [0, 1] * 30, "score": [0.1, 0.9] * 30})))
    assert small["grade"]["letter"] == "I" and "rows" in small["grade"]["incomplete_reason"]
    sure = analyze(load_predictions(pd.DataFrame({"y": [0, 1] * 100, "score": [0.001, 0.999] * 100})))
    assert sure["grade"]["letter"] == "I" and "wrong predictions" in sure["grade"]["incomplete_reason"]
    # clearly underconfident even with no mistakes: gets a real (bad) grade
    timid = analyze(load_predictions(pd.DataFrame({"y": [0, 1] * 100, "score": [0.35, 0.65] * 100})))
    assert timid["grade"]["letter"] == "F"


def test_biggest_miss_direction_matches_gap():
    import pandas as pd

    from calibrationcard.analysis import analyze

    # says 0.65 but always right -> accuracy is ABOVE the claim
    timid = analyze(load_predictions(pd.DataFrame({"y": [0, 1] * 100, "score": [0.35, 0.65] * 100})))
    miss = next(n for n in timid["grade"]["notes"] if n.startswith("Biggest miss"))
    assert "above what it claims" in miss
    # says ~0.95 but right ~60% -> accuracy is BELOW the claim
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.6).astype(int)
    bold = analyze(load_predictions(pd.DataFrame({"y": y, "score": np.full(400, 0.95)})), recalibrate=False)
    miss = next(n for n in bold["grade"]["notes"] if n.startswith("Biggest miss"))
    assert "below what it claims" in miss


def test_class_direction_labels_mixed_errors():
    from calibrationcard.charts import class_direction

    assert class_direction({"ece": 0.01, "signed_gap": 0.01}) == "on target"
    assert class_direction({"ece": 0.14, "signed_gap": 0.003}) == "mixed"
    assert class_direction({"ece": 0.10, "signed_gap": 0.09}) == "over"
    assert class_direction({"ece": 0.10, "signed_gap": -0.09}) == "under"


def test_error_count_wording():
    from calibrationcard.metrics import count_errors

    assert count_errors(0) == "no wrong predictions"
    assert count_errors(1) == "only 1 wrong prediction"
    assert count_errors(7) == "only 7 wrong predictions"
