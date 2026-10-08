import json
import threading
import urllib.request

import pytest

from calibrationcard import HOST
from calibrationcard.app import create_app
from calibrationcard.cli import main
from calibrationcard.demo import synthetic_binary
from calibrationcard.report import find_browser
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
