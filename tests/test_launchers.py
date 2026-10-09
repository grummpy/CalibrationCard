import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_files_exist_and_point_at_the_module():
    command = ROOT / "Launch CalibrationCard.command"
    windows = ROOT / "Launch CalibrationCard.bat"
    shell = ROOT / "launch.sh"
    desktop = ROOT / "calibrationcard.desktop"
    for path in (command, windows, shell, desktop):
        assert path.is_file(), path
    assert os.access(command, os.X_OK)
    assert os.access(shell, os.X_OK)
    for path in (command, windows, shell):
        text = path.read_text(encoding="utf-8")
        assert "calibrationcard" in text
        assert "https://www.python.org/downloads/" in text
        assert "-m calibrationcard" in text
    desktop_text = desktop.read_text(encoding="utf-8")
    assert "launch.sh" in desktop_text
    assert "icon" in desktop_text.lower()
    assert b"\r\n" in windows.read_bytes()


def test_linux_launcher_recovers_dependencies_from_an_existing_venv():
    text = (ROOT / "launch.sh").read_text(encoding="utf-8")
    assert "import flask, numpy, pandas, scipy, sklearn, calibrationcard" in text
    assert "Installing or recovering pinned CalibrationCard dependencies" in text


def test_icons_and_cover_exist():
    for name in ("icon.svg", "icon.png", "icon-512.png", "icon-1024.png", "icon.icns", "icon.ico"):
        assert (ROOT / "assets" / name).is_file(), name
    assert (ROOT / "docs" / "cover.jpg").is_file()
    assert (ROOT / "calibrationcard" / "web" / "favicon.ico").is_file()


