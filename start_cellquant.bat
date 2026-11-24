@echo off
title CellQuant
echo ================================================================
echo   CellQuant v3
echo ================================================================
echo.

REM Start the Windows folder-picker agent in a hidden background window
echo [1/2] Starting folder-picker agent on port 7861...
start "CellQuant Picker" /min cmd /c "set CELLQUANT_PICKER_PORT=7861 && python \"%~dp0cellquant-picker.py\""
timeout /t 1 /nobreak >nul

REM Start Docker service
echo [2/2] Starting CellQuant Docker service...
cd /d "%~dp0"
docker compose up

REM When Docker stops, also kill the picker agent
taskkill /fi "WindowTitle eq CellQuant Picker" /f >nul 2>&1
