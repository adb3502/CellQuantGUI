@echo off
:: CellQuant Picker — Per-user auto-start installer
:: Each user runs this ONCE after logging in for the first time.
:: It installs a Task Scheduler job that silently starts the folder-picker
:: agent at login so Browse works from their account.

setlocal

set PYTHONW=D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe
set PICKER=D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3\cellquant-picker.py
set BACKEND=http://localhost:7860
set TASKNAME=CellQuantPicker

if not exist "%PYTHONW%" (
    echo ERROR: Python not found at %PYTHONW%
    pause
    exit /b 1
)

:: Remove old task for this user
schtasks /delete /tn "%TASKNAME%" /f >nul 2>&1

:: Create task: runs silently at logon, no window (pythonw = windowless)
schtasks /create ^
  /tn "%TASKNAME%" ^
  /tr "\"%PYTHONW%\" \"%PICKER%\" --backend %BACKEND%\"" ^
  /sc onlogon ^
  /ru "%USERNAME%" ^
  /rl limited ^
  /f

if %errorlevel% neq 0 (
    echo ERROR: Could not create task. Try running as Administrator once.
    pause
    exit /b 1
)

echo CellQuant Picker installed for: %USERNAME%
echo Starting it now silently...
start "" "%PYTHONW%" "%PICKER%" --backend %BACKEND%
echo Done. The folder picker will auto-start at every login.
timeout /t 3 >nul
