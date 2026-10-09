#!/usr/bin/env bash
# Linux launcher. The macOS .command file and the Windows .bat file do the same job.
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
  echo "Download it from https://www.python.org/downloads/ and then run this launcher again."
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "First run: creating a virtual environment..."
  "$PY" -m venv .venv
fi

if ! .venv/bin/python -c 'import flask, numpy, pandas, scipy, sklearn, calibrationcard' >/dev/null 2>&1; then
  echo "Installing or recovering pinned CalibrationCard dependencies..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install -e . --no-deps
fi

exec .venv/bin/python -m calibrationcard "$@"
