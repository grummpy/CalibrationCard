"""Render the one-page HTML report card, and print it to PDF with a local Chrome or Edge if one is installed."""

from __future__ import annotations

import base64
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path

from calibrationcard import __version__
from calibrationcard.charts import BLUE, GREEN, RED, class_bars, confidence_histogram, reliability

WEB = Path(__file__).resolve().parent / "web"


def _pct(x):
    return "n/a" if x is None else f"{abs(x) * 100:.1f}%" if abs(x) < 5e-4 else f"{x * 100:.1f}%"


def _num(x):
    return f"{x + 0.0:.4f}" if abs(x) >= 5e-5 else "0.0000"


def _icon_data_uri() -> str:
    svg = (WEB / "icon.svg").read_bytes() if (WEB / "icon.svg").is_file() else b""
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()


def grade_color(letter: str) -> str:
    return {"A": GREEN, "B": "#4f8a3a", "C": "#c98a16", "D": "#d0662b", "F": RED, "I": "#7a8094"}.get(letter[:1], RED)


def render_html(result: dict, *, generated: datetime | None = None) -> str:
    m = result["metrics"]
    g = result["grade"]
    s = result["summary"]
    recal = result.get("recalibration") or {}
    generated = generated or datetime.now()
    best = next((r for r in recal.get("methods", []) if r.get("method") == recal.get("best")), None)
    overlays = [("Raw model", m["bins"], BLUE)]
    if best and "bins" in best:
        overlays.append((f"After {best['label'].lower()}", best["bins"], GREEN))
    rel = reliability(m["bins"], overlays=overlays, floor=m.get("bin_floor", 0.0),
                      title="Reliability: confidence vs. accuracy", height=330)
    hist = confidence_histogram(m["histogram"])
    classes = class_bars(m["classwise"]) if len(s["classes"]) > 2 else ""
    if s["binary"] and m.get("positive_bins"):
        classes = reliability(m["positive_bins"], title=f"P({s['classes'][1]}) vs. how often it is {s['classes'][1]}",
                              width=420, height=250)
    interval = m.get("ece_interval")
    ece_text = _pct(m["ece"]) + (f' <span class="ci">90% range {_pct(interval[0])}–{_pct(interval[1])}</span>'
                                 if interval else "")
    metric_rows = [
        ("Expected calibration error", ece_text, "Average confidence vs. accuracy gap"),
        ("ECE above chance floor", _pct(m.get("ece_excess", m["ece"])), f"Floor {_pct(m.get('ece_noise_floor', 0))} · graded on this"),
        ("Adaptive ECE", _pct(m["adaptive_ece"]), "Equal-size bins"),
        ("Max calibration error", _pct(m["mce"]), "Worst bin"),
        ("Class-wise ECE", _pct(m["classwise_ece"]), "One-vs-rest, averaged"),
        ("Brier score", _num(m["brier"]), "0 is perfect"),
        ("Log loss", _num(m["log_loss"]), "Punishes confident mistakes"),
        ("Accuracy", _pct(m["accuracy"]), f"{m['errors']:,} wrong of {m['n']:,}"),
        ("Mean confidence", _pct(m["mean_confidence"]), "Average top-class probability"),
    ]
    recal_rows = ""
    if recal.get("methods"):
        before = recal["before"]
        recal_rows = (f"<tr><td>Raw model</td><td>{_pct(before['ece'])}</td><td>{before['brier']:.4f}</td>"
                      f"<td>{_num(before['log_loss'])}</td></tr>")
        for r in recal["methods"]:
            if "error" in r:
                recal_rows += f"<tr><td>{escape(r['label'])}</td><td colspan=3>{escape(r['error'])}</td></tr>"
                continue
            mark = " ★" if r["method"] == recal.get("best") else ""
            extra = ""
            if r["method"] == "temperature" and r.get("params", {}).get("temperature"):
                extra = f' <span class="ci">T = {", ".join(str(t) for t in r["params"]["temperature"])}</span>'
            recal_rows += (f"<tr><td>{escape(r['label'])}{mark}{extra}</td><td>{_pct(r['ece'])}</td>"
                           f"<td>{_num(r['brier'])}</td><td>{_num(r['log_loss'])}</td></tr>")
    elif recal.get("skipped"):
        recal_rows = f"<tr><td colspan=4>{escape(recal['skipped'])}</td></tr>"
    notes = "".join(f"<li>{escape(n)}</li>" for n in g["notes"])
    scale = (" · ".join(f"{b['letter']} &lt; {b['ece_below'] * 100:.0f}%" for b in g["scale"][:-1])
             + " · F otherwise (ECE above the chance floor) · I = not enough evidence")
    color = grade_color(g["letter"])
    title = escape(result.get("name", "predictions"))
    folds = recal.get("folds", 2)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CalibrationCard · {title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
@page {{ size: letter; margin: 0.35in; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #efe8d8; font: 11.5px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f2a44; }}
.card {{ max-width: 8.1in; margin: 18px auto; background: #fbf6ea; border: 1px solid #e0d6bf; border-radius: 14px;
  padding: 20px 24px; box-shadow: 0 8px 30px rgba(60,40,10,.12); }}
header {{ display: flex; align-items: center; gap: 12px; border-bottom: 2px solid #1f2a44; padding-bottom: 10px; }}
header img {{ width: 40px; height: 40px; }}
header h1 {{ font-size: 20px; margin: 0; }} header .meta {{ margin-left: auto; text-align: right; color: #6b6f80; font-size: 11px; }}
.top {{ display: grid; grid-template-columns: 136px 1fr; gap: 16px; margin: 10px 0 4px; align-items: center; }}
.grade {{ width: 124px; height: 124px; border-radius: 50%; border: 6px solid {color}; display: flex; align-items: center;
  justify-content: center; font: 700 64px/1 Georgia, "Times New Roman", serif; color: {color}; transform: rotate(-8deg); }}
.headline {{ font-size: 18px; font-weight: 700; margin-bottom: 6px; }}
ul.notes {{ margin: 0; padding-left: 18px; }} ul.notes li {{ margin: 3px 0; }}
.grid {{ display: grid; grid-template-columns: 1.05fr 1fr; gap: 10px 16px; margin-top: 6px; align-items: start; }}
table {{ width: 100%; border-collapse: collapse; }}
td, th {{ padding: 3px 5px; border-bottom: 1px solid #e6e0d2; text-align: left; vertical-align: top; }}
th {{ font-size: 11px; color: #6b6f80; font-weight: 600; }}
td.v {{ font-weight: 700; white-space: nowrap; }} td.h {{ color: #6b6f80; font-size: 11px; }}
.ci {{ font-weight: 400; color: #6b6f80; font-size: 10px; }}
h2 {{ font-size: 13px; margin: 6px 0 4px; }}
svg {{ width: 100%; height: auto; }}
footer {{ margin-top: 10px; color: #6b6f80; font-size: 10px; display: flex; justify-content: space-between; gap: 10px; }}
@media print {{ body {{ background: #fff; }} .card {{ box-shadow: none; border: 0; margin: 0; padding: 0; max-width: none;
  zoom: 0.94; }} .grid, section, footer {{ break-inside: avoid; }} }}
</style></head>
<body><div class="card">
<header><img src="{_icon_data_uri()}" alt=""><div><h1>CalibrationCard</h1><div>{title}</div></div>
<div class="meta">{m['n']:,} rows · {len(s['classes'])} classes{' (binary)' if s['binary'] else ''}<br>
Generated {generated:%Y-%m-%d %H:%M} · v{__version__}</div></header>
<section class="top"><div class="grade" aria-label="Grade {escape(g['letter'])}">{escape(g['letter'])}</div>
<div><div class="headline">{escape(g['headline'])}</div><ul class="notes">{notes}</ul></div></section>
<section class="grid">
<div>{rel}</div>
<div><table>{''.join(f'<tr><td>{escape(a)}</td><td class="v">{b}</td><td class="h">{escape(c)}</td></tr>' for a, b, c in metric_rows)}</table></div>
<div>{hist}</div>
<div>{classes}</div>
</section>
<section><h2>Recalibration (cross-fitted on {folds} folds, scored only on rows each calibrator never saw)</h2>
<table><tr><th>Method</th><th>ECE</th><th>Brier</th><th>Log loss</th></tr>{recal_rows}</table></section>
<footer><span>Grade scale (ECE): {scale}</span><span>Made locally. Your predictions never left this computer.</span></footer>
</div></body></html>"""


def find_browser() -> str | None:
    names = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge", "chrome"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    candidates = []
    if platform.system() == "Darwin":
        candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                      "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    elif platform.system() == "Windows":
        for base in (os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                     os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                     os.environ.get("LOCALAPPDATA", "")):
            candidates += [os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
                           os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


class PdfUnavailable(RuntimeError):
    pass


def html_to_pdf(html: str, out: Path, browser: str | None = None, timeout: int = 60) -> Path:
    browser = browser or find_browser()
    if not browser:
        raise PdfUnavailable("No Chrome, Chromium or Edge found. Open the HTML report and use Print → Save as PDF.")
    out = Path(out).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        base = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={tmp}/profile",
                f"--print-to-pdf={out}", page.as_uri()]
        for extra in ([], ["--no-sandbox"]):
            command = base[:1] + extra + base[1:]
            try:
                subprocess.run(command, capture_output=True, timeout=timeout, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise PdfUnavailable(f"Could not run {browser}: {exc}") from exc
            if out.is_file() and out.stat().st_size > 1000:
                return out
    raise PdfUnavailable("The browser did not produce a PDF. Open the HTML report and use Print → Save as PDF.")
