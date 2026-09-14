@echo off
REM RedVox Django Diagnostic Script
echo ========================================
echo   RedVox Django Diagnostic Tool
echo ========================================
echo.

set PROJECT_DIR=C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk
set GUI_DIR=%PROJECT_DIR%\redvox_gui

REM Set Python executable path (prefer .venv, fallback to system)
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo Using virtual environment: .venv
    set PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe
) else (
    echo Using system Python
    set PYTHON_EXE=python
)

echo.
echo Checking directories...
if exist "%PROJECT_DIR%" (
    echo [OK] Project directory exists: %PROJECT_DIR%
) else (
    echo [ERROR] Project directory not found: %PROJECT_DIR%
)

if exist "%GUI_DIR%" (
    echo [OK] GUI directory exists: %GUI_DIR%
) else (
    echo [ERROR] GUI directory not found: %GUI_DIR%
)

echo.
echo Checking virtual environments...
if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    echo [OK] Virtual environment found: .venv
) else (
    echo [WARNING] No virtual environment found
)

echo.
echo Checking required directories...
if exist "%GUI_DIR%\sessions" (
    echo [OK] Sessions directory exists
) else (
    echo [INFO] Creating sessions directory...
    mkdir "%GUI_DIR%\sessions"
)

if exist "%GUI_DIR%\media" (
    echo [OK] Media directory exists
) else (
    echo [INFO] Creating media directory...
    mkdir "%GUI_DIR%\media"
)

if exist "%GUI_DIR%\media\data_windows" (
    echo [OK] Data windows directory exists
) else (
    echo [INFO] Creating data windows directory...
    mkdir "%GUI_DIR%\media\data_windows"
)

echo.
echo Checking Python and Django...
"%PYTHON_EXE%" --version
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
) else (
    echo [OK] Python is available
)

"%PYTHON_EXE%" -c "import django; print(f'Django version: {django.VERSION}')" 2>nul
if errorlevel 1 (
    echo [ERROR] Django not installed
    echo Run: pip install django
) else (
    echo [OK] Django is installed
)

"%PYTHON_EXE%" -c "import redvox; print(f'RedVox SDK version: {redvox.VERSION}')" 2>nul
if errorlevel 1 (
    echo [WARNING] RedVox SDK not found in current environment
) else (
    echo [OK] RedVox SDK is available
)

"%PYTHON_EXE%" -c "import lz4; print('lz4 module: OK')" 2>nul
if errorlevel 1 (
    echo [ERROR] lz4 module not found - REQUIRED for RedVox file processing
    echo Run: quick_fix_lz4.bat or install_dependencies.bat
) else (
    echo [OK] lz4 module is available
)

"%PYTHON_EXE%" -c "import numpy; print('numpy version:', numpy.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] numpy not found
) else (
    echo [OK] numpy is available
)

"%PYTHON_EXE%" -c "import matplotlib; print('matplotlib version:', matplotlib.__version__)" 2>nul
if errorlevel 1 (
    echo [WARNING] matplotlib not found - Required for Signal Analysis
    echo Run: pip install matplotlib
) else (
    echo [OK] matplotlib is available
)

"%PYTHON_EXE%" -c "import scipy; print('scipy version:', scipy.__version__)" 2>nul
if errorlevel 1 (
    echo [WARNING] scipy not found - Required for signal processing
) else (
    echo [OK] scipy is available
)

"%PYTHON_EXE%" -c "import pandas; print('pandas version:', pandas.__version__)" 2>nul
if errorlevel 1 (
    echo [WARNING] pandas not found - Required for data analysis
) else (
    echo [OK] pandas is available
)

echo.
echo Checking disk space (simplified)...
dir C:\ | find "bytes free"

echo.
echo Checking Django settings...
"%PYTHON_EXE%" -c "import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'redvox_gui.settings'); import django; django.setup(); from django.conf import settings; print(f'Debug mode: {settings.DEBUG}'); print(f'Allowed hosts: {settings.ALLOWED_HOSTS}'); print(f'Media root: {settings.MEDIA_ROOT}'); print(f'Session engine: {settings.SESSION_ENGINE}')" 2>nul

echo.
echo ========================================
echo   Diagnostic Complete
echo ========================================
echo.
echo If all checks pass, try running: launch_redvox_gui.bat
echo.
pause