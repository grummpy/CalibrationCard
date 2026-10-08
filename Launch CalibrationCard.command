#!/bin/bash
# Double-click this file on macOS. It creates .venv on the first run, then opens the app.
set -euo pipefail
cd "$(dirname "$0")"

python_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

PY=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
    PY="$candidate"
    break
  fi
done

if [[ -z "$PY" ]]; then
  echo "CalibrationCard needs Python 3.11 or newer."
  echo "Python was not found, or it is too old."
  echo "Download it from https://www.python.org/downloads/ and then double-click this launcher again."
  read -r -p "Press Enter to close..."
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "First run: creating a virtual environment and installing pinned packages..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install -e . --no-deps
fi

exec .venv/bin/python -m calibrationcard "$@"
