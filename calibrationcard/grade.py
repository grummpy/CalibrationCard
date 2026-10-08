"""Turn the metrics into a letter grade and plain-English notes."""

from __future__ import annotations

from calibrationcard.charts import class_direction
from calibrationcard.metrics import count_errors

# ECE cut-offs for each letter. Each letter band is split in thirds for + and -.
BANDS = [("A", 0.0, 0.02), ("B", 0.02, 0.05), ("C", 0.05, 0.10), ("D", 0.10, 0.15), ("F", 0.15, 1.0)]
SMALL_SAMPLE = 300
MIN_ROWS = 100
MIN_ERRORS = 5


def letter(ece: float) -> str:
    for name, low, high in BANDS:
        if ece < high:
            if name == "F":
                return "F"
            third = (high - low) / 3
            if ece < low + third:
                return name + "+"
            if ece >= high - third:
                return name + "-"
            return name
    return "F"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def grade(metrics: dict, recal: dict | None, summary: dict) -> dict:
    ece = metrics["ece"]
    # Grade on the part of ECE that sampling noise cannot explain, so small test sets are not punished for being small.
    graded_on = metrics.get("ece_excess", ece)
    provisional = letter(graded_on)
    notes: list[str] = []
    n = metrics["n"]
    # Overconfidence only shows up through mistakes. With very few rows, or very few mistakes and no clear gap,
    # the honest grade is Incomplete (the numbers are still shown).
    incomplete_reason = ""
    if n < MIN_ROWS:
        incomplete_reason = f"Only {n} rows. At least {MIN_ROWS} are needed for a grade."
    elif metrics["errors"] < MIN_ERRORS and graded_on <= 0.02:
        incomplete_reason = (f"{count_errors(metrics['errors']).capitalize()}, so there is not enough evidence that its "
                             "confidence would hold up when it is wrong.")
    grade_letter = "I" if incomplete_reason else provisional
    if incomplete_reason:
        notes.append(f"Incomplete: {incomplete_reason} On the numbers so far it would get {provisional}.")
    gap = metrics["signed_gap"]
    acc = metrics["accuracy"]
    conf = metrics["mean_confidence"]

    if abs(gap) < 0.01:
        notes.append(f"Confidence tracks accuracy closely: on average the model says {pct(conf)} and is right {pct(acc)} "
                     "of the time.")
    elif gap > 0:
        notes.append(f"Overconfident: average confidence is {pct(conf)} but accuracy is {pct(acc)}. Treat its high "
                     "scores with some caution.")
    else:
        notes.append(f"Underconfident: average confidence is {pct(conf)} but accuracy is {pct(acc)}. It is right more "
                     "often than it claims.")

    worst = max((b for b in metrics["bins"] if b["count"] >= max(5, n // 50)), key=lambda b: abs(b["gap"]), default=None)
    if worst and abs(worst["gap"]) >= 0.05:
        direction = "below" if worst["gap"] > 0 else "above"
        notes.append(f"Biggest miss: when it says {pct(worst['low'])} to {pct(worst['high'])}, it is right "
                     f"{pct(worst['accuracy'])} of the time ({worst['count']} rows), {pct(abs(worst['gap']))} {direction} "
                     "what it claims.")

    floor = metrics.get("ece_noise_floor")
    if floor is not None and floor >= 0.005:
        share = min(1.0, floor / ece) if ece else 1.0
        notes.append(f"With {n:,} rows, even a perfectly calibrated model would show an ECE of about {pct(floor)} from "
                     f"chance alone ({share:.0%} of the measured {pct(ece)}). The grade uses the {pct(graded_on)} above "
                     "that floor.")

    if metrics["errors"] < 10 and not incomplete_reason:
        notes.append(f"{count_errors(metrics['errors']).capitalize()} in {n} rows, so there is little evidence about how "
                     "the model behaves when it is wrong. The grade mostly reflects its high-confidence answers.")
    if n < SMALL_SAMPLE and n >= MIN_ROWS:
        interval = metrics.get("ece_interval")
        extra = f" The 90% bootstrap range for ECE is {pct(interval[0])} to {pct(interval[1])}." if interval else ""
        notes.append(f"Small sample ({n} rows): treat the grade as rough.{extra}")

    cw = metrics.get("classwise") or []
    if len(cw) > 2:
        bad = max(cw, key=lambda c: c["ece"])
        if bad["ece"] > max(0.02, 1.5 * metrics["classwise_ece"]):
            word = {"over": "over-predicted", "under": "under-predicted",
                    "mixed": "too high in some ranges and too low in others"}.get(class_direction(bad), "")
            notes.append(f"Class '{bad['class']}' is the least calibrated ({word}, class ECE {pct(bad['ece'])}).")

    if recal and recal.get("methods"):
        best = next((m for m in recal["methods"] if m["method"] == recal.get("best")), None)
        if best and recal.get("helps"):
            notes.append(f"{best['label']} (cross-fitted) lowers ECE from {pct(recal['before']['ece'])} to "
                         f"{pct(best['ece'])} and log loss from {recal['before']['log_loss']:.3f} to "
                         f"{best['log_loss']:.3f}. Worth applying before trusting the scores.")
        elif best:
            notes.append("Recalibration does not clearly help on held-out rows, so the raw scores are about as good as "
                         "they get without retraining.")
    elif recal and recal.get("skipped"):
        notes.append(recal["skipped"])

    for note in summary.get("notes", []):
        notes.append(note)

    headline = {
        "I": "Incomplete: not enough evidence to grade",
        "A": "Trustworthy confidence",
        "B": "Mostly trustworthy confidence",
        "C": "Confidence is off by a noticeable amount",
        "D": "Confidence is unreliable",
        "F": "Confidence should not be trusted as a probability",
    }[grade_letter[0]]
    return {"letter": grade_letter, "provisional": provisional, "incomplete_reason": incomplete_reason,
            "headline": headline, "notes": notes, "graded_on": graded_on, "scale": [
        {"letter": name, "ece_below": high} for name, _low, high in BANDS
    ]}
