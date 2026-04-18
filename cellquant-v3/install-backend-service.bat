@echo off
:: CellQuant Backend Service Installer
:: Run ONCE as Administrator to install the backend as a SYSTEM service.
:: SYSTEM account can read all user folders on the workstation.
::
:: After this, the backend starts automatically at boot on port 7860.

setlocal

set PYTHON=D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe
set APP_DIR=D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3
set TASKNAME=CellQuantBackend
set PORT=7860

:: Must be admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Please right-click and "Run as Administrator".
    pause
    exit /b 1
)

if not exist "%PYTHON%" (
    echo ERROR: Python venv not found at %PYTHON%
    pause
    exit /b 1
)

:: Remove old task
schtasks /delete /tn "%TASKNAME%" /f >nul 2>&1

:: Create task running as SYSTEM at boot (SYSTEM reads all user folders)
schtasks /create ^
  /tn "%TASKNAME%" ^
  /tr "\"%PYTHON%\" -m cellquant serve --host 0.0.0.0 --port %PORT% --no-browser\"" ^
  /sc onstart ^
  /ru "SYSTEM" ^
  /rl highest ^
  /sd 01/01/2024 ^
  /f

if %errorlevel% neq 0 (
    echo ERROR: Failed to create backend service task.
    pause
    exit /b 1
)

:: Set working directory via XML patch (schtasks /create doesn't support /wd on all Windows)
schtasks /query /tn "%TASKNAME%" /xml > "%TEMP%\cq_task.xml"
powershell -Command ^
  "(Get-Content '%TEMP%\cq_task.xml') -replace '<WorkingDirectory>.*</WorkingDirectory>', '' | Set-Content '%TEMP%\cq_task2.xml'"
powershell -Command ^
  "(Get-Content '%TEMP%\cq_task2.xml') -replace '<Command>', '<WorkingDirectory>%APP_DIR%</WorkingDirectory><Command>' | Set-Content '%TEMP%\cq_task3.xml'"
schtasks /delete /tn "%TASKNAME%" /f >nul 2>&1
schtasks /create /tn "%TASKNAME%" /xml "%TEMP%\cq_task3.xml" /f >nul 2>&1

echo.
echo Backend service installed. It will start at next boot.
echo Starting it now...
schtasks /run /tn "%TASKNAME%"
timeout /t 4 >nul
curl -s http://localhost:%PORT%/api/health && echo. && echo Backend is running OK.
echo.
echo Next step: each user should run install-picker.bat once to set up their folder picker.
pause
