"""Command line: open the UI or grade a predictions CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calibrationcard import __version__
from calibrationcard.analysis import analyze, headline_numbers
from calibrationcard.demo import synthetic_predictions
from calibrationcard.loader import PredictionsError, load_predictions
from calibrationcard.recalibrate import METHODS
from calibrationcard.report import PdfUnavailable, html_to_pdf, render_html
from calibrationcard.server import serve

COMMANDS = {"serve", "grade", "demo", "-h", "--help", "--version"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        argv = ["serve", *argv]
    parser = argparse.ArgumentParser(prog="calibrationcard", description="Grade how trustworthy a classifier's "
                                     "confidence scores are.")
    parser.add_argument("--version", action="version", version=f"calibrationcard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("serve", help="Open the local UI")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--data-dir", type=Path, default=None)

    gp = sub.add_parser("grade", help="Grade a predictions CSV")
    gp.add_argument("csv", type=Path)
    gp.add_argument("--label", help="True-label column (guessed if omitted)")
    gp.add_argument("--prob-cols", nargs="+", help="Probability columns, one per class, or one binary score")
    gp.add_argument("--positive", help="Positive class when there is one score column")
    gp.add_argument("--bins", type=int, default=15)
    gp.add_argument("--seed", type=int, default=0)
    gp.add_argument("--no-recalibration", action="store_true")
    gp.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    gp.add_argument("--name", help="Title for the report (defaults to the file name)")
    gp.add_argument("--html", type=Path, help="Write the report card as HTML")
    gp.add_argument("--pdf", type=Path, help="Write the report card as PDF (needs Chrome, Chromium or Edge)")
    gp.add_argument("--json", type=Path, help="Write all metrics as JSON")

    dp = sub.add_parser("demo", help="Write a synthetic predictions CSV to try the tool")
    dp.add_argument("out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(port=args.port, open_browser=not args.no_browser, data_dir=args.data_dir)
        return 0
    if args.command == "demo":
        synthetic_predictions().to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
        return 0
    try:
        pred = load_predictions(args.csv, label=args.label, prob_columns=args.prob_cols, positive=args.positive)
    except PredictionsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    result = analyze(pred, name=args.name or args.csv.name, n_bins=args.bins, seed=args.seed,
                     recalibrate=not args.no_recalibration, methods=args.methods)
    head = headline_numbers(result)
    print(f"{head['name']}: grade {head['grade']} · ECE {head['ece']:.2%} · Brier {head['brier']:.4f} · "
          f"log loss {head['log_loss']:.4f} · accuracy {head['accuracy']:.1%} · {head['rows']} rows")
    for note in result["grade"]["notes"]:
        print(f"  - {note}")
    html = render_html(result)
    if args.html:
        args.html.write_text(html, encoding="utf-8")
        print(f"Report: {args.html}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.pdf:
        try:
            html_to_pdf(html, args.pdf)
            print(f"PDF: {args.pdf}")
        except PdfUnavailable as exc:
            print(f"PDF skipped: {exc}", file=sys.stderr)
            return 3
    return 0
