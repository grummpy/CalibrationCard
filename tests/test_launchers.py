import os
import shutil
import subprocess
from pathlib import Path

import pytest

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


def test_native_launchers_recover_dependencies_from_an_existing_venv():
    for name in ("launch.sh", "Launch CalibrationCard.command", "Launch CalibrationCard.bat"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "import flask, numpy, pandas, scipy, sklearn, calibrationcard" in text
        assert "Installing or recovering pinned CalibrationCard dependencies" in text


@pytest.mark.parametrize("existing_venv", [False, True])
def test_macos_launcher_recovers_synthetic_fresh_and_partial_environments(tmp_path, existing_venv):
    """The launcher must not mistake a Python executable for a completed setup."""
    launcher = tmp_path / "Launch CalibrationCard.command"
    shutil.copy2(ROOT / "Launch CalibrationCard.command", launcher)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text("""#!/bin/sh
if [ \"$1\" = \"-c\" ]; then
  case \"$2\" in
    *version_info*) exit 0 ;;
    *) exit 1 ;;
  esac
fi
if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then
  mkdir -p \"$3/bin\"
  cp \"$0\" \"$3/bin/python\"
  exit 0
fi
printf '%s\\n' \"$*\" >> \"$CALIBRATIONCARD_FAKE_LOG\"
exit 0
""", encoding="utf-8")
    fake_python.chmod(0o755)
    for name in ("python3.13", "python3.12", "python3.11", "python"):
        (fake_bin / name).symlink_to(fake_python.name)
    if existing_venv:
        partial_python = tmp_path / ".venv" / "bin" / "python"
        partial_python.parent.mkdir(parents=True)
        shutil.copy2(fake_python, partial_python)
        partial_python.chmod(0o755)
    log = tmp_path / "fake-python.log"
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin", "CALIBRATIONCARD_FAKE_LOG": str(log)}
    run = subprocess.run(["/bin/bash", str(launcher)], cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr
    assert "Installing or recovering pinned CalibrationCard dependencies" in run.stdout
    calls = log.read_text(encoding="utf-8")
    assert "-m pip install -r requirements.txt" in calls
    assert "-m pip install -e . --no-deps" in calls
    assert "-m calibrationcard" in calls


def test_icons_and_cover_exist():
    for name in ("icon.svg", "icon.png", "icon-512.png", "icon-1024.png", "icon.icns", "icon.ico"):
        assert (ROOT / "assets" / name).is_file(), name
    assert (ROOT / "docs" / "cover.jpg").is_file()
    assert (ROOT / "calibrationcard" / "web" / "favicon.ico").is_file()


