@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 goto missing

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 goto missing

if not exist .venv\Scripts\python.exe (
  echo First run: creating a virtual environment and installing pinned packages...
  python -m venv .venv
  if errorlevel 1 goto failed
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe -m pip install -e . --no-deps
)

.venv\Scripts\python.exe -m calibrationcard
exit /b %errorlevel%

:missing
echo CalibrationCard needs Python 3.11 or newer.
echo Python was not found, or it is too old.
echo Download it from https://www.python.org/downloads/ and then double-click this launcher again.
pause
exit /b 1

:failed
echo Setup failed. See the messages above.
pause
exit /b 1
