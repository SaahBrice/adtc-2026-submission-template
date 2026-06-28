@echo off
REM ============================================================
REM  Docaware - one-click launcher (Windows)
REM  Double-click this file to start the app and open it in your
REM  browser. Requires: models downloaded (bash download_model.sh)
REM  and Python deps installed (pip install -r app/requirements.txt).
REM ============================================================
setlocal
cd /d "%~dp0app"

echo Starting Docaware...
echo (First run loads the model; the page may take a few seconds.)
echo Close the server window to stop Docaware.

start "Docaware server" cmd /k python -m docaware serve --port 8000
timeout /t 5 >nul
start "" http://127.0.0.1:8000
endlocal
