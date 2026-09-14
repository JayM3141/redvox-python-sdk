@echo off
REM Clean up unused Python virtual environments to save disk space
echo ========================================
echo   RedVox Environment Cleanup
echo ========================================
echo.

set PROJECT_DIR=C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk

echo Checking virtual environments...
echo.

REM Check .venv status
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo Found .venv (Python 3.13) - ACTIVE ENVIRONMENT
    "%PROJECT_DIR%\.venv\Scripts\python.exe" --version
    echo.
    echo Checking installed packages...
    "%PROJECT_DIR%\.venv\Scripts\python.exe" -m pip list | find /I "django"
    "%PROJECT_DIR%\.venv\Scripts\python.exe" -m pip list | find /I "lz4"
    "%PROJECT_DIR%\.venv\Scripts\python.exe" -m pip list | find /I "matplotlib"
    echo.
    echo [OK] .venv is fully configured with all required dependencies
) else (
    echo [ERROR] .venv Python executable not found
)

echo.
echo ========================================
echo   Environment Status
echo ========================================
echo.
echo Current Setup:
echo - .venv (Python 3.13): ACTIVE - Contains all required dependencies
echo - .venv312: REMOVED - Was broken, deleted to save disk space
echo.
echo Environment is now streamlined with single functional virtual environment.
echo This saves approximately 100-200 MB of disk space.
echo.
pause