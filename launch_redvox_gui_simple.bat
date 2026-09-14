@echo off
REM Simple RedVox Django Web Viewer Launcher
REM Update the paths below if needed

set PYTHON_PATH=python
set GUI_DIR=C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk\redvox_gui
set PORT=8000

cd /d "%GUI_DIR%"

echo Starting RedVox Django Web Viewer...
echo Server will be available at: http://127.0.0.1:%PORT%
echo.

start %PYTHON_PATH% manage.py runserver %PORT%

timeout /t 2 /nobreak >nul
start http://127.0.0.1:%PORT%

echo Server started! Press any key to close this window (server continues running)...
pause