@echo off
REM Quick fix for missing lz4 module
echo ========================================
echo   Quick Fix: Installing lz4 module
echo ========================================
echo.

set PROJECT_DIR=C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk

cd /d "%PROJECT_DIR%"

REM Try to activate virtual environment
if exist "%PROJECT_DIR%\.venv312\Scripts\activate.bat" (
    echo Activating virtual environment: .venv312
    call "%PROJECT_DIR%\.venv312\Scripts\activate.bat"
) else if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    echo Activating virtual environment: .venv
    call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
)

echo.
echo Installing lz4 module...
pip install lz4==4.3.3

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install lz4
    echo Trying without version constraint...
    pip install lz4
)

echo.
echo Verifying installation...
python -c "import lz4; print('SUCCESS: lz4 module is now available')" 2>nul
if errorlevel 1 (
    echo ERROR: lz4 module still not available
) else (
    echo.
    echo lz4 module installed successfully!
    echo You can now run the Django application.
)

echo.
pause