import numpy as np
import pandas as pd
import pytest

from calibrationcard.loader import PredictionsError, load_predictions, preview


def test_multiclass_with_prefixed_columns():
    frame = pd.DataFrame({"label": ["a", "b", "c"], "p_a": [0.7, 0.2, 0.1], "p_b": [0.2, 0.7, 0.1], "p_c": [0.1, 0.1, 0.8]})
    pred = load_predictions(frame)
    assert pred.classes == ["a", "b", "c"]
    assert pred.y.tolist() == [0, 1, 2]
    assert not pred.binary


def test_columns_named_after_classes_and_numeric_labels():
    frame = pd.DataFrame(
        {"y_true": [0, 1, 2, 1], "0": [0.6, 0.2, 0.1, 0.3], "1": [0.3, 0.7, 0.2, 0.4], "2": [0.1, 0.1, 0.7, 0.3]}
    )
    pred = load_predictions(frame)
    assert pred.label_column == "y_true"
    assert pred.classes == ["0", "1", "2"]
    assert pred.y.tolist() == [0, 1, 2, 1]


def test_suffix_columns_and_proba_prefix():
    frame = pd.DataFrame({"target": ["x", "y"], "proba_x": [0.9, 0.4], "proba_y": [0.1, 0.6]})
    assert load_predictions(frame).classes == ["x", "y"]
    frame2 = pd.DataFrame({"target": ["x", "y"], "x_prob": [0.9, 0.4], "y_prob": [0.1, 0.6]})
    assert load_predictions(frame2).classes == ["x", "y"]


def test_binary_single_score_and_positive_choice():
    frame = pd.DataFrame({"label": ["spam", "ham", "spam"], "score": [0.9, 0.2, 0.6]})
    pred = load_predictions(frame, positive="spam")
    assert pred.binary and pred.classes == ["ham", "spam"]
    assert np.allclose(pred.proba[:, 1], [0.9, 0.2, 0.6])
    auto = load_predictions(pd.DataFrame({"y": [1, 0, 1], "score": [0.8, 0.1, 0.4]}))
    assert auto.classes == ["0", "1"] and auto.y.tolist() == [1, 0, 1]


def test_two_columns_matched_by_order_for_binary():
    frame = pd.DataFrame({"label": [0, 1, 1], "prob_neg": [0.8, 0.3, 0.1], "prob_pos": [0.2, 0.7, 0.9]})
    pred = load_predictions(frame)
    assert pred.classes == ["0", "1"]
    assert pred.y.tolist() == [0, 1, 1]
    assert any("order" in n for n in pred.notes)


def test_missing_rows_dropped_and_rows_rescaled():
    frame = pd.DataFrame({"label": ["a", "b", None, "a"], "p_a": [0.5, 0.2, 0.5, 0.6], "p_b": [0.4, 0.8, 0.5, 0.4]})
    pred = load_predictions(frame)
    assert pred.n == 3 and pred.dropped_rows == 1
    assert pred.renormalized_rows == 1
    assert np.allclose(pred.proba.sum(axis=1), 1)


def test_rounding_noise_is_silent():
    frame = pd.DataFrame({"label": ["a", "b"], "p_a": [0.333333, 0.5], "p_b": [0.666666, 0.5]})
    pred = load_predictions(frame)
    assert pred.renormalized_rows == 0 and not pred.notes


def test_errors_are_friendly(tmp_path):
    with pytest.raises(PredictionsError, match="outside 0 to 1"):
        load_predictions(pd.DataFrame({"label": [0, 1], "logit": [3.2, -1.0]}), prob_columns=["logit"])
    with pytest.raises(PredictionsError, match="no probability column"):
        load_predictions(pd.DataFrame({"label": ["a", "b", "c"], "p_a": [0.5, 0.5, 0.5], "p_b": [0.5, 0.5, 0.5]}))
    with pytest.raises(PredictionsError, match="classes"):
        load_predictions(pd.DataFrame({"label": ["a", "b", "c"], "score": [0.1, 0.5, 0.9]}))
    with pytest.raises(PredictionsError, match="not found"):
        load_predictions(tmp_path / "missing.csv")


def test_preview_guesses():
    frame = pd.DataFrame({"label": ["a"], "p_a": [1.0], "note": ["x"]})
    info = preview(frame)
    assert info["label_guess"] == "label" and info["prob_guess"] == ["p_a"]
