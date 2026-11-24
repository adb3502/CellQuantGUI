@echo off
setlocal
title CellQuant Local
echo ================================================================
echo   CellQuant v3 Local
echo ================================================================
echo.

set ROOT=%~dp0
set VENV=%ROOT%.venv
set PYTHON=%VENV%\Scripts\python.exe
set PICKER=%ROOT%cellquant-v3\cellquant-picker.py
set APP_ROOT=%ROOT%cellquant-v3
set FRONTEND_BUILD=%APP_ROOT%\frontend\build
set STATIC_DIR=%APP_ROOT%\src\cellquant\static
set STATIC_APP_DIR=%STATIC_DIR%\_app

if not exist "%PYTHON%" (
    echo ERROR: %PYTHON% not found.
    pause
    exit /b 1
)

echo [1/2] Starting native Windows picker...
start "CellQuant Picker" /min cmd /c "set CELLQUANT_PICKER_PORT=7861 && \"%PYTHON%\" \"%PICKER%\""
timeout /t 1 /nobreak >nul

echo [2/2] Starting CellQuant from .venv on http://127.0.0.1:7860 ...
set PYTHONPATH=%APP_ROOT%\src
set CELLQUANT_DB=%ROOT%CellQuant_Output\cellquant-local.db
set CELLQUANT_ADMIN_USER=admin
set CELLQUANT_ADMIN_PASS=changeme

if not exist "%STATIC_APP_DIR%" (
    if exist "%FRONTEND_BUILD%\_app" (
        echo      Frontend build found. Copying static assets...
        if not exist "%STATIC_DIR%" mkdir "%STATIC_DIR%"
        xcopy "%FRONTEND_BUILD%\*" "%STATIC_DIR%\" /E /I /Y >nul
    ) else (
        echo ERROR: Frontend static assets are missing.
        echo.
        echo Run these commands once:
        echo   cd /d "%APP_ROOT%\frontend"
        echo   npm install
        echo   npm run build
        echo.
        pause
        exit /b 1
    )
)

"%PYTHON%" -c "import bcrypt, fastapi, aiosqlite" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Local backend dependencies are missing in .venv.
    echo.
    echo Run this once:
    echo   cd /d "%APP_ROOT%"
    echo   \"%PYTHON%\" -m pip install -e .
    echo.
    pause
    exit /b 1
)

cd /d "%APP_ROOT%"
"%PYTHON%" -m uvicorn cellquant.api.app:create_app --factory --host 127.0.0.1 --port 7860 --reload --reload-dir "%APP_ROOT%\src\cellquant"

taskkill /fi "WindowTitle eq CellQuant Picker" /f >nul 2>&1
endlocal
