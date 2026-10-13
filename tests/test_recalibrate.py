import numpy as np

from calibrationcard.analysis import analyze
from calibrationcard.demo import synthetic_binary, synthetic_predictions
from calibrationcard.loader import load_predictions
from calibrationcard.recalibrate import Temperature, compare, cross_fit, held_out_final_evaluation
from calibrationcard.report import render_html


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


def test_final_evaluation_keeps_selection_labels_separate_and_records_provenance():
    pred = load_predictions(synthetic_predictions(n=600, seed=12))
    result = analyze(pred, methods=("temperature",), seed=7, final_evaluation_fraction=0.25)
    recal = result["recalibration"]
    assert recal["protocol"] == "held-out-final-evaluation"
    assert recal["selection"]["rows"] + recal["final_evaluation"]["rows"] == pred.n
    assert recal["selection"]["candidate_comparison"]["best"] == "temperature"
    assignments = recal["selection"]["candidate_comparison"]["methods"][0]["params"]["fold_assignments"]
    assert len(assignments) == recal["selection"]["rows"] and set(assignments) == {0, 1}
    assert set(recal["selection"]["row_indexes"]).isdisjoint(recal["final_evaluation"]["row_indexes"])
    assert recal["final_evaluation"]["selected_method"] == "temperature"
    assert recal["final_evaluation"]["raw"]["n"] == recal["final_evaluation"]["rows"]
    assert "not used to select it" in " ".join(result["grade"]["notes"])
    provenance = result["provenance"]
    assert len(provenance["input_sha256"]) == 64
    assert len(provenance["configuration_sha256"]) == 64
    assert provenance["class_order"] == pred.classes
    html = render_html(result)
    assert "Recalibration final evaluation" in html
    assert "final-evaluation labels were not used to select a method" in html
    assert "Evidence: input" in html and " · revision " in html


def test_final_evaluation_does_not_claim_selection_when_candidates_are_skipped():
    pred = load_predictions(synthetic_predictions(n=600, seed=12))
    result = analyze(pred, methods=(), seed=7, final_evaluation_fraction=0.25)
    recal = result["recalibration"]
    assert recal["protocol"] == "held-out-final-evaluation"
    assert recal["best"] is None
    html = render_html(result)
    assert "no calibrator selected using" in html
    assert "method selected on" not in html


def test_final_evaluation_refuses_an_unsafe_fraction():
    pred = load_predictions(synthetic_binary(n=100))
    with np.testing.assert_raises_regex(ValueError, "between 0.10 and 0.50"):
        held_out_final_evaluation(pred, methods=("temperature",), final_fraction=0.05)
