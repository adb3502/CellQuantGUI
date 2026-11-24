@echo off
:: CellQuant Picker — Install as a startup task for ALL users
:: Run this once as Administrator from the CellQuantGUI folder.

set SCRIPT_DIR=%~dp0
set PICKER=%SCRIPT_DIR%cellquant-v3\cellquant-picker.py

:: Resolve the real Python install, not the Windows Store app alias.
for /f "delims=" %%i in ('powershell -NoProfile -Command "$py=(Get-Command python -ErrorAction SilentlyContinue).Source; if($py){$dir=Split-Path $py; $pyw=Join-Path $dir ''pythonw.exe''; if(Test-Path $pyw){$pyw}else{$py}}"') do set PYTHONW=%%i
if "%PYTHONW%"=="" (
    echo ERROR: Python not found on PATH. Install Python first.
    pause
    exit /b 1
)

echo Installing CellQuant Picker startup task...
echo   Python:  %PYTHONW%
echo   Script:  %PICKER%

:: Delete old task if it exists
schtasks /delete /tn "CellQuantPicker" /f >nul 2>&1

:: Create task: runs at logon for ANY user, in their own session
schtasks /create ^
  /tn "CellQuantPicker" ^
  /tr "\"%PYTHONW%\" \"%PICKER%\"" ^
  /sc ONLOGON ^
  /ru "INTERACTIVE" ^
  /rl HIGHEST ^
  /f

if %errorlevel% == 0 (
    echo.
    echo Success! CellQuant Picker will auto-start when any user logs in.
    echo Running it now for your current session...
    start "" "%PYTHONW%" "%PICKER%"
) else (
    echo.
    echo ERROR: Failed to create scheduled task. Make sure you ran as Administrator.
)
pause
