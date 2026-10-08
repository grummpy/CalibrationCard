"""Read a predictions CSV: one true-label column plus a probability column per class (or one binary score)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAMES = ("label", "y_true", "true_label", "true", "target", "truth", "actual", "y", "class", "gold")
PROB_PATTERNS = (
    re.compile(r"^(?:pred_proba|pred_prob|probability|proba|prob|score|p)[_:\-\s](?P<cls>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<cls>.+?)[_:\-\s](?:probability|proba|prob|score|p)$", re.IGNORECASE),
)
BINARY_SCORE_NAMES = ("score", "prob", "proba", "probability", "p", "y_prob", "y_score", "pred_prob", "p_1", "prob_1",
                      "p_pos", "p_positive", "confidence")


class PredictionsError(ValueError):
    """The CSV cannot be read as classifier predictions."""


@dataclass
class Predictions:
    y: np.ndarray  # integer class index per row
    proba: np.ndarray  # (n, k) probabilities, rows sum to 1
    classes: list[str]
    binary: bool
    label_column: str
    prob_columns: list[str]
    dropped_rows: int = 0
    renormalized_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.y))

    def summary(self) -> dict:
        counts = np.bincount(self.y, minlength=len(self.classes))
        return {
            "rows": self.n,
            "classes": self.classes,
            "class_counts": {c: int(v) for c, v in zip(self.classes, counts, strict=True)},
            "binary": self.binary,
            "label_column": self.label_column,
            "prob_columns": self.prob_columns,
            "dropped_rows": self.dropped_rows,
            "renormalized_rows": self.renormalized_rows,
            "notes": self.notes,
        }


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def guess_label_column(frame: pd.DataFrame) -> str:
    lowered = {_norm(c): c for c in frame.columns}
    for name in LABEL_NAMES:
        if name in lowered:
            return lowered[name]
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            return col
    for col in frame.columns:
        values = frame[col].dropna()
        if len(values) and np.allclose(values, np.round(values)) and values.nunique() <= 50:
            return col
    raise PredictionsError("Could not find the true-label column. Name it 'label' or pick it in the app.")


def _probability_like(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna()
    return len(values) > 0 and float(values.min()) >= -1e-9 and float(values.max()) <= 1 + 1e-9


def _class_from_column(name: str) -> str | None:
    for pattern in PROB_PATTERNS:
        match = pattern.match(str(name).strip())
        if match:
            return match.group("cls").strip()
    return None


def load_predictions(
    source: str | Path | pd.DataFrame,
    *,
    label: str | None = None,
    prob_columns: list[str] | None = None,
    positive: str | None = None,
) -> Predictions:
    frame = source if isinstance(source, pd.DataFrame) else _read_csv(source)
    if frame.empty:
        raise PredictionsError("The file has no rows.")
    label_col = label or guess_label_column(frame)
    if label_col not in frame.columns:
        raise PredictionsError(f"Column '{label_col}' is not in the file.")
    candidates = prob_columns or [c for c in frame.columns if c != label_col and _probability_like(frame[c])]
    missing = [c for c in candidates if c not in frame.columns]
    if missing:
        raise PredictionsError(f"Columns not found: {', '.join(missing)}")
    if not candidates:
        raise PredictionsError("No probability columns found. Probabilities must be numbers between 0 and 1.")
    for col in candidates:
        if not _probability_like(frame[col]):
            raise PredictionsError(f"Column '{col}' has values outside 0 to 1, so it is not a probability.")

    keep = frame[[label_col, *candidates]].copy()
    before = len(keep)
    keep = keep.dropna()
    dropped = before - len(keep)
    if keep.empty:
        raise PredictionsError("Every row is missing a label or a probability.")
    labels = keep[label_col].map(_label_text)
    notes: list[str] = []
    if dropped:
        notes.append(f"Skipped {dropped} rows with a missing label or probability.")

    if len(candidates) == 1:
        return _binary_from_score(keep, labels, label_col, candidates[0], positive, dropped, notes)

    mapping = {}
    label_values = sorted(set(labels))
    for col in candidates:
        cls = _class_from_column(col)
        if cls is not None and cls in label_values:
            mapping[col] = cls
        elif str(col) in label_values:
            mapping[col] = str(col)
        elif cls is not None:
            mapping[col] = cls
        else:
            mapping[col] = str(col)
    classes = [mapping[c] for c in candidates]
    if len(set(classes)) != len(classes):
        raise PredictionsError("Two probability columns map to the same class. Pick the columns by hand.")
    unknown = sorted(set(label_values) - set(classes))
    if unknown:
        if len(candidates) == 2 and len(label_values) == 2:
            # e.g. columns p0/p1 or prob_neg/prob_pos with labels 0/1: order by column order
            mapping = dict(zip(candidates, _sorted_labels(label_values), strict=True))
            classes = [mapping[c] for c in candidates]
            notes.append(f"Matched probability columns to labels by order: {mapping}.")
        else:
            raise PredictionsError(
                f"Labels {unknown[:5]} have no probability column. Columns found: {', '.join(map(str, candidates))}."
            )
    index = {cls: i for i, cls in enumerate(classes)}
    y = np.array([index[v] for v in labels], dtype=int)
    proba = keep[candidates].to_numpy(dtype=float)
    proba, renormalized = _renormalize(proba, notes)
    binary = len(classes) == 2
    return Predictions(y, proba, classes, binary, label_col, list(candidates), dropped, renormalized, notes)


def _sorted_labels(values: list[str]) -> list[str]:
    try:
        return sorted(values, key=float)
    except ValueError:
        return sorted(values)


def _label_text(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    return str(value).strip()


def _binary_from_score(keep, labels, label_col, score_col, positive, dropped, notes) -> Predictions:
    values = sorted(set(labels))
    if len(values) > 2:
        raise PredictionsError(
            f"One score column was found but the labels have {len(values)} classes. Add one probability column per class."
        )
    if positive is None:
        for guess in ("1", "true", "yes", "pos", "positive"):
            match = [v for v in values if v.lower() == guess]
            if match:
                positive = match[0]
                break
        else:
            positive = _sorted_labels(values)[-1]
            notes.append(f"Treated '{positive}' as the positive class. Use --positive to change it.")
    positive = str(positive)
    if positive not in values and len(values) == 2:
        raise PredictionsError(f"Positive class '{positive}' is not one of the labels {values}.")
    negative = next((v for v in values if v != positive), "not " + positive)
    p1 = keep[score_col].to_numpy(dtype=float)
    proba = np.column_stack([1 - p1, p1])
    y = np.array([1 if v == positive else 0 for v in labels], dtype=int)
    return Predictions(y, proba, [negative, positive], True, label_col, [score_col], dropped, 0, notes)


def _renormalize(proba: np.ndarray, notes: list[str]) -> tuple[np.ndarray, int]:
    sums = proba.sum(axis=1)
    if np.any(sums <= 0):
        raise PredictionsError("Some rows have probabilities that add up to zero.")
    error = np.abs(sums - 1)
    count = int((error > 1e-3).sum())  # rounding noise below 0.001 is rescaled silently
    if count:
        worst = float(error.max())
        notes.append(f"{count} rows did not add up to 1 (worst off by {worst:.3f}) and were rescaled.")
        if worst > 0.05:
            notes.append("Large row-sum errors usually mean a column is missing or is not a probability.")
    if np.any(error > 1e-9):
        proba = proba / sums[:, None]
    return proba, count


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.is_file():
        raise PredictionsError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise PredictionsError(f"Could not read {path.name} as CSV: {exc}") from exc


def preview(source: str | Path | pd.DataFrame, rows: int = 5) -> dict:
    frame = source if isinstance(source, pd.DataFrame) else _read_csv(source)
    label = None
    try:
        label = guess_label_column(frame)
    except PredictionsError:
        pass
    probs = [c for c in frame.columns if c != label and _probability_like(frame[c])]
    return {
        "columns": [str(c) for c in frame.columns],
        "label_guess": label,
        "prob_guess": [str(c) for c in probs],
        "rows": int(len(frame)),
        "head": frame.head(rows).astype(str).to_dict(orient="records"),
    }
