@echo off
:: CellQuant Backend Service Installer
:: Run ONCE as Administrator to install the backend as a SYSTEM service.
:: SYSTEM account can read all user folders on the workstation.

setlocal

set "PYTHONW=D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe"
set "APP_DIR=D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3"
set "TASKNAME=CellQuantBackend"
set "PORT=7860"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Please right-click and "Run as Administrator".
    pause
    exit /b 1
)

if not exist "%PYTHONW%" (
    echo ERROR: Python venv not found at %PYTHONW%
    pause
    exit /b 1
)

schtasks /delete /tn "%TASKNAME%" /f >nul 2>&1

:: Write the task XML directly - avoids all quoting/continuation issues
set "XMLFILE=%TEMP%\cellquant_backend_task.xml"
(
echo ^<?xml version="1.0" encoding="UTF-16"?^>
echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
echo   ^<Triggers^>
echo     ^<BootTrigger^>^<Enabled^>true^</Enabled^>^</BootTrigger^>
echo   ^</Triggers^>
echo   ^<Principals^>
echo     ^<Principal id="Author"^>
echo       ^<UserId^>S-1-5-18^</UserId^>
echo       ^<RunLevel^>HighestAvailable^</RunLevel^>
echo     ^</Principal^>
echo   ^</Principals^>
echo   ^<Settings^>
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^>
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>
echo     ^<ExecutionTimeLimit^>PT0S^</ExecutionTimeLimit^>
echo     ^<RestartOnFailure^>^<Interval^>PT1M^</Interval^>^<Count^>10^</Count^>^</RestartOnFailure^>
echo   ^</Settings^>
echo   ^<Actions^>
echo     ^<Exec^>
echo       ^<Command^>%PYTHONW%^</Command^>
echo       ^<Arguments^>-m cellquant serve --host 0.0.0.0 --port %PORT% --no-browser^</Arguments^>
echo       ^<WorkingDirectory^>%APP_DIR%^</WorkingDirectory^>
echo     ^</Exec^>
echo   ^</Actions^>
echo ^</Task^>
) > "%XMLFILE%"

schtasks /create /tn "%TASKNAME%" /xml "%XMLFILE%" /f
if %errorlevel% neq 0 (
    echo ERROR: Failed to register scheduled task.
    pause
    exit /b 1
)

del "%XMLFILE%" >nul 2>&1

echo.
echo Backend service installed. Starting now...
schtasks /run /tn "%TASKNAME%"
timeout /t 5 >nul
curl -s http://localhost:%PORT%/api/health
echo.
echo Backend is running. Each user should now run install-picker.bat once.
pause
