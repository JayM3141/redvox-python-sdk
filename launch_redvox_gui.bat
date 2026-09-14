@echo off
REM RedVox Django Web Viewer Launcher
REM This script activates the virtual environment and starts the Django development server

echo ========================================
echo   RedVox Django Web Viewer Launcher
echo ========================================
echo.

REM Set the project directory
set PROJECT_DIR=C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk
set GUI_DIR=%PROJECT_DIR%\redvox_gui

REM Check if project directory exists
if not exist "%PROJECT_DIR%" (
    echo ERROR: Project directory not found: %PROJECT_DIR%
    echo Please update the PROJECT_DIR variable in this script.
    pause
    exit /b 1
)

REM Check if GUI directory exists
if not exist "%GUI_DIR%" (
    echo ERROR: GUI directory not found: %GUI_DIR%
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "%GUI_DIR%\sessions" mkdir "%GUI_DIR%\sessions"
if not exist "%GUI_DIR%\media" mkdir "%GUI_DIR%\media"
if not exist "%GUI_DIR%\media\data_windows" mkdir "%GUI_DIR%\media\data_windows"

REM Set Python executable path (prefer .venv, fallback to system)
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo Using virtual environment: .venv (Python 3.13)
    set PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe
) else (
    echo WARNING: No virtual environment found in .venv
    echo Using system Python. Make sure Django is installed.
    set PYTHON_EXE=python
)

REM Change to GUI directory
cd /d "%GUI_DIR%"

echo Using Python: %PYTHON_EXE%

REM Check if Django is installed
"%PYTHON_EXE%" -c "import django" 2>nul
if errorlevel 1 (
    echo ERROR: Django is not installed in the current Python environment.
    echo Please install Django using: pip install django
    echo Or run: install_dependencies.bat
    pause
    exit /b 1
)

REM Check if lz4 is installed (required for RedVox)
"%PYTHON_EXE%" -c "import lz4" 2>nul
if errorlevel 1 (
    echo ERROR: lz4 module is not installed in the current Python environment.
    echo This is required for RedVox file processing.
    echo.
    echo Please run: quick_fix_lz4.bat
    echo Or install manually: pip install lz4
    echo.
    pause
    exit /b 1
)

REM Check if matplotlib is installed (required for Signal Analysis)
"%PYTHON_EXE%" -c "import matplotlib" 2>nul
if errorlevel 1 (
    echo WARNING: matplotlib module is not installed in the current Python environment.
    echo This is required for the Signal Analysis feature.
    echo.
    echo Please run: pip install matplotlib
    echo Or install manually: pip install matplotlib
    echo.
    echo The Django server will start, but Signal Analysis will not work.
    echo.
)

REM Set environment variables for debugging
set DJANGO_DEBUG=True
set PYTHONUNBUFFERED=1

echo.
echo Starting Django development server...
echo Server will be available at: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

REM Start Django server in background and open browser
start "RedVox Django Server" cmd /k "%PYTHON_EXE% manage.py runserver 127.0.0.1:8000"

REM Wait a moment for server to start
timeout /t 4 /nobreak >nul

REM Open browser
echo Opening web browser...
start http://127.0.0.1:8000

echo.
echo Django server is running in a separate window.
echo Close this window to keep the server running, or press Ctrl+C in the server window to stop it.
echo.
echo If you encounter ERR_FILE_NO_SPACE errors:
echo 1. Check disk space on your C: drive
echo 2. Clear your browser cache
echo 3. Try using a different browser
echo.
pause