@echo off
REM One-click launcher for the UAV Firmware Anomaly Detection prototype
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\..\UAV_Firmware_Anomaly_Detection\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Starting UAV Firmware Anomaly Detection...
echo Your browser will open at http://localhost:8501  (close this window to stop)
"%PY%" -m streamlit run app.py --server.maxUploadSize 64 --theme.base light
pause
