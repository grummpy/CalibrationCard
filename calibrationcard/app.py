"""Local web UI. Served on 127.0.0.1 only (see server.py)."""

from __future__ import annotations

import io
import json
import threading
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from calibrationcard import __version__
from calibrationcard.analysis import analyze, headline_numbers
from calibrationcard.demo import synthetic_predictions
from calibrationcard.loader import PredictionsError, load_predictions, preview
from calibrationcard.report import PdfUnavailable, find_browser, html_to_pdf, render_html


def web_dir() -> Path:
    return Path(__file__).resolve().parent / "web"


def create_app(data_dir: Path | None = None) -> Flask:
    folder = Path(data_dir) if data_dir is not None else Path.cwd() / "data"
    folder.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, static_folder=str(web_dir()), static_url_path="/static")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
    results: dict[str, dict] = {}
    frames: dict[str, tuple[str, pd.DataFrame]] = {}
    lock = threading.Lock()

    def keep(store, key, value, limit=10):
        with lock:
            store[key] = value
            while len(store) > limit:
                store.pop(next(iter(store)))

    @app.get("/")
    def index():
        return send_from_directory(web_dir(), "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return send_from_directory(web_dir(), "favicon.ico")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "version": __version__})

    @app.get("/api/meta")
    def meta():
        return jsonify({"version": __version__, "pdf": bool(find_browser())})

    @app.post("/api/upload")
    def upload():
        """Accept a CSV upload (multipart) or a local path (JSON) and return a column preview."""
        try:
            if "file" in request.files:
                f = request.files["file"]
                name = Path(f.filename or "predictions.csv").name
                frame = pd.read_csv(io.BytesIO(f.read()))
            else:
                body = request.get_json(force=True, silent=True) or {}
                if body.get("demo"):
                    name, frame = "Demo: overconfident 4-class model", synthetic_predictions()
                else:
                    path = Path(str(body.get("path", "")).strip()).expanduser()
                    if not path.is_file():
                        return jsonify({"error": f"File not found: {path}"}), 400
                    name, frame = path.name, pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            return jsonify({"error": f"Could not read that file as CSV: {exc}"}), 400
        if frame.empty:
            return jsonify({"error": "The file has no rows."}), 400
        upload_id = uuid.uuid4().hex[:12]
        keep(frames, upload_id, (name, frame))
        info = preview(frame)
        info.update({"upload_id": upload_id, "name": name})
        return jsonify(info)

    @app.post("/api/analyze")
    def analyze_route():
        body = request.get_json(force=True, silent=True) or {}
        item = frames.get(str(body.get("upload_id", "")))
        if item is None:
            return jsonify({"error": "Upload a predictions CSV first."}), 400
        name, frame = item
        try:
            pred = load_predictions(frame, label=body.get("label") or None, prob_columns=body.get("prob_columns") or None,
                                    positive=body.get("positive") or None)
            bins = int(body.get("bins", 15))
            if not 5 <= bins <= 50:
                raise PredictionsError("Bins must be between 5 and 50.")
            result = analyze(pred, name=name, n_bins=bins, seed=int(body.get("seed", 0)),
                             recalibrate=bool(body.get("recalibrate", True)))
        except PredictionsError as exc:
            return jsonify({"error": str(exc)}), 400
        result_id = uuid.uuid4().hex[:12]
        result["result_id"] = result_id
        keep(results, result_id, result)
        return jsonify({"result_id": result_id, "headline": headline_numbers(result), "result": result})

    @app.get("/api/report/<result_id>.html")
    def report_html(result_id: str):
        result = results.get(result_id) or abort(404)
        download = request.args.get("download") == "1"
        headers = {"Content-Disposition": "attachment; filename=calibration-report.html"} if download else {}
        return Response(render_html(result), mimetype="text/html", headers=headers)

    @app.get("/api/report/<result_id>.json")
    def report_json(result_id: str):
        result = results.get(result_id) or abort(404)
        return Response(json.dumps(result, indent=2), mimetype="application/json",
                        headers={"Content-Disposition": "attachment; filename=calibration-report.json"})

    @app.get("/api/report/<result_id>.pdf")
    def report_pdf(result_id: str):
        result = results.get(result_id) or abort(404)
        out = folder / f"report-{result_id}.pdf"
        try:
            html_to_pdf(render_html(result), out)
        except PdfUnavailable as exc:
            return jsonify({"error": str(exc)}), 501
        data = out.read_bytes()
        out.unlink(missing_ok=True)
        return Response(data, mimetype="application/pdf",
                        headers={"Content-Disposition": "attachment; filename=calibration-report.pdf"})

    return app
