@echo off
REM Install RedVox Dependencies
echo ========================================
echo   Installing RedVox Dependencies
echo ========================================
echo.

set PROJECT_DIR=C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk

cd /d "%PROJECT_DIR%"

REM Try to activate virtual environment
if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    echo Activating virtual environment: .venv
    call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
) else (
    echo WARNING: No virtual environment found
    echo Installing to system Python
)

echo.
echo Installing RedVox package with dependencies...
echo This may take a few minutes...
echo.

pip install -e .

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Trying to install missing packages individually...
    echo.
    
    echo Installing lz4...
    pip install lz4==4.3.3
    
    echo Installing Django...
    pip install django
    
    echo Installing other core dependencies...
    pip install numpy pandas scipy pyarrow protobuf requests matplotlib
) else (
    echo.
    echo SUCCESS: Dependencies installed successfully
)

echo.
echo Verifying key installations...
python -c "import lz4; print('lz4: OK')" 2>nul || echo "lz4: FAILED"
python -c "import django; print('django: OK')" 2>nul || echo "django: FAILED"
python -c "import redvox; print('redvox: OK')" 2>nul || echo "redvox: FAILED"
python -c "import numpy; print('numpy: OK')" 2>nul || echo "numpy: FAILED"

echo.
echo ========================================
echo   Installation Complete
echo ========================================
echo.
pause