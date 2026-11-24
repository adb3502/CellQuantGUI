@echo off
:: CellQuant Picker — Per-user auto-start installer
:: No admin rights needed.

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

set /p CQUSER="Enter your CellQuant username (the one you log into the app with): "

powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%PYTHONW%'; $s.Arguments = '\"%PICKER%\" --backend %BACKEND% --username %CQUSER%'; $s.WindowStyle = 7; $s.Save()"

if not exist "%SHORTCUT%" (
    echo ERROR: Could not create startup shortcut.
    pause
    exit /b 1
)

echo Installed for CellQuant user: %CQUSER%
echo Starting picker now...
start "" "%PYTHONW%" "%PICKER%" --backend %BACKEND% --username %CQUSER%
echo Done.
timeout /t 3 >nul
