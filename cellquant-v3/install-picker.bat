@echo off
:: CellQuant Picker — Per-user auto-start installer
:: No admin rights needed. Adds a shortcut to the user's Startup folder
:: so the picker launches silently at every login.

setlocal

set "PYTHONW=D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe"
set "PICKER=D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3\cellquant-picker.py"
set "BACKEND=http://localhost:7860"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP%\CellQuantPicker.lnk"

if not exist "%PYTHONW%" (
    echo ERROR: Python not found at %PYTHONW%
    pause
    exit /b 1
)

:: Create a .lnk shortcut in the user's Startup folder via PowerShell
powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%PYTHONW%'; $s.Arguments = '\"%PICKER%\" --backend %BACKEND%'; $s.WindowStyle = 7; $s.Save()"

if not exist "%SHORTCUT%" (
    echo ERROR: Could not create startup shortcut.
    pause
    exit /b 1
)

echo Installed for: %USERNAME%
echo Starting picker now...
start "" "%PYTHONW%" "%PICKER%" --backend %BACKEND%
echo Done. Folder picker will auto-start at every login.
timeout /t 3 >nul
