import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrationcard import HOST
from calibrationcard.app import create_app
from calibrationcard.cli import main
from calibrationcard.demo import synthetic_binary
from calibrationcard.report import find_browser, grade_color
from calibrationcard.server import make_bound_server


def test_server_on_loopback_returns_200(tmp_path):
    assert HOST == "127.0.0.1"
    server = make_bound_server(create_app(data_dir=tmp_path), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        assert host == "127.0.0.1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            assert response.status == 200
            assert "CalibrationCard" in response.read().decode()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_upload_analyze_and_report(tmp_path):
    client = create_app(data_dir=tmp_path).test_client()
    path = tmp_path / "bin.csv"
    synthetic_binary().to_csv(path, index=False)
    with path.open("rb") as handle:
        up = client.post("/api/upload", data={"file": (handle, "bin.csv")}, content_type="multipart/form-data").get_json()
    assert up["label_guess"] == "y_true" and up["prob_guess"] == ["score"]
    res = client.post("/api/analyze", json={"upload_id": up["upload_id"], "label": "y_true", "prob_columns": ["score"]})
    assert res.status_code == 200, res.get_json()
    data = res.get_json()
    assert data["headline"]["rows"] == 800
    html = client.get(f"/api/report/{data['result_id']}.html")
    assert html.status_code == 200 and b"CalibrationCard" in html.data and b"<svg" in html.data
    assert client.get(f"/api/report/{data['result_id']}.json").status_code == 200
    by_path = client.post("/api/upload", json={"path": str(path)})
    assert by_path.status_code == 200
    assert client.post("/api/upload", json={"path": str(tmp_path / "nope.csv")}).status_code == 400
    bad = client.post("/api/analyze", json={"upload_id": up["upload_id"], "label": "score", "prob_columns": ["y_true"]})
    assert bad.status_code == 400


def test_demo_route(tmp_path):
    client = create_app(data_dir=tmp_path).test_client()
    up = client.post("/api/upload", json={"demo": True}).get_json()
    res = client.post("/api/analyze", json={"upload_id": up["upload_id"]}).get_json()
    assert res["headline"]["grade"][0] in "CDF"


def test_web_incomplete_grade_has_a_neutral_color_mapping():
    script = (Path(__file__).resolve().parents[1] / "calibrationcard" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'I: "#5e6b78"' in script
    assert 'g.setAttribute("aria-label", `Grade ${h.grade}: ${headline}`)' in script
    assert grade_color("I") == "#5e6b78"


def test_incomplete_transitions_stay_neutral_across_api_json_and_html(tmp_path):
    """Each result is independent; a prior grade cannot leak into an Incomplete report."""
    client = create_app(data_dir=tmp_path).test_client()
    rng = np.random.default_rng(5)
    calibrated_scores = rng.random(4000)
    cases = [
        ("f", pd.DataFrame({"y": [0, 1] * 100, "score": [0.35, 0.65] * 100}), "F"),
        ("i", pd.DataFrame({"y": [0, 1] * 30, "score": [0.1, 0.9] * 30}), "I"),
        ("a", pd.DataFrame({"y": (rng.random(4000) < calibrated_scores).astype(int), "score": calibrated_scores}), "A"),
    ]
    seen = []
    for name, frame, expected in cases:
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        upload = client.post("/api/upload", json={"path": str(path)}).get_json()
        response = client.post("/api/analyze", json={
            "upload_id": upload["upload_id"], "label": "y", "prob_columns": ["score"], "recalibrate": False,
        })
        assert response.status_code == 200, response.get_json()
        result = response.get_json()
        assert result["headline"]["grade"].startswith(expected)
        result_id = result["result_id"]
        exported = client.get(f"/api/report/{result_id}.json").get_json()
        html = client.get(f"/api/report/{result_id}.html").get_data(as_text=True)
        assert exported["grade"]["letter"].startswith(expected)
        assert f'aria-label="Grade {exported["grade"]["letter"]}"' in html
        if expected == "I":
            assert exported["grade"]["headline"] == "Incomplete: not enough evidence to grade"
            assert "border: 6px solid #5e6b78" in html
            assert "I = not enough evidence" in html
        seen.append(result_id)
    assert len(set(seen)) == len(cases)


@pytest.mark.skipif(find_browser() is None, reason="needs Chrome, Chromium or Edge for PDF")
def test_pdf_export(tmp_path):
    client = create_app(data_dir=tmp_path).test_client()
    up = client.post("/api/upload", json={"demo": True}).get_json()
    res = client.post("/api/analyze", json={"upload_id": up["upload_id"]}).get_json()
    pdf = client.get(f"/api/report/{res['result_id']}.pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"


def test_cli_grade_and_demo(tmp_path, capsys):
    demo = tmp_path / "demo.csv"
    assert main(["demo", str(demo)]) == 0
    html, js = tmp_path / "r.html", tmp_path / "r.json"
    assert main(["grade", str(demo), "--html", str(html), "--json", str(js), "--bins", "10"]) == 0
    out = capsys.readouterr().out
    assert "grade" in out and "ECE" in out
    assert html.read_text().startswith("<!doctype html>")
    assert json.loads(js.read_text())["settings"]["bins"] == 10
    assert main(["grade", str(tmp_path / "missing.csv")]) == 2
