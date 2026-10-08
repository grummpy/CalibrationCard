#!/usr/bin/env python3
"""Optional desktop build with PyInstaller.

Run this on the operating system you want a bundle for. It is not part of
the default install, and CI does not run it.

    pip install pyinstaller
    python scripts/build_app.py

macOS writes dist/CalibrationCard (run the binary from Terminal, or wrap it
yourself). On a Mac the icon is assets/icon.icns. Windows uses assets/icon.ico
and writes dist/CalibrationCard.exe. Linux writes dist/CalibrationCard using
the same .ico file.

The window stays a console so the local URL is visible and Ctrl+C stops the
server. The bundle is still a local app: it serves 127.0.0.1 only.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    system = platform.system()
    icon = ROOT / "assets" / ("icon.icns" if system == "Darwin" else "icon.ico")
    if not icon.is_file():
        raise SystemExit(f"Missing icon: {icon}. Run python scripts/render_assets.py first.")
    separator = ";" if system == "Windows" else ":"
    web = ROOT / "calibrationcard" / "web"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        "CalibrationCard",
        "--console",
        "--icon",
        str(icon),
        "--add-data",
        f"{web}{separator}calibrationcard/web",
        "--collect-submodules",
        "sklearn",
        "--collect-submodules",
        "scipy",
        "--collect-submodules",
        "calibrationcard",
        str(ROOT / "scripts" / "pyinstaller_entry.py"),
    ]
    if system == "Darwin":
        command.extend(["--osx-bundle-identifier", "com.grummpy.calibrationcard"])
    print("Running:", " ".join(command))
    subprocess.check_call(command, cwd=ROOT)
    print("Build finished under dist/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
