@echo off
:: CellQuant Picker — Per-user auto-start installer
:: Each user runs this ONCE. Installs a silent Task Scheduler job that
:: starts their folder-picker agent at every login.

setlocal

set "PYTHONW=D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe"
set "PICKER=D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3\cellquant-picker.py"
set "BACKEND=http://localhost:7860"
set "TASKNAME=CellQuantPicker"

if not exist "%PYTHONW%" (
    echo ERROR: Python not found at %PYTHONW%
    pause
    exit /b 1
)

schtasks /delete /tn "%TASKNAME%" /f >nul 2>&1

set "XMLFILE=%TEMP%\cellquant_picker_task.xml"
(
echo ^<?xml version="1.0" encoding="UTF-16"?^>
echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
echo   ^<Triggers^>
echo     ^<LogonTrigger^>^<Enabled^>true^</Enabled^>^<UserId^>%USERDOMAIN%\%USERNAME%^</UserId^>^</LogonTrigger^>
echo   ^</Triggers^>
echo   ^<Principals^>
echo     ^<Principal id="Author"^>
echo       ^<LogonType^>InteractiveToken^</LogonType^>
echo       ^<RunLevel^>LeastPrivilege^</RunLevel^>
echo     ^</Principal^>
echo   ^</Principals^>
echo   ^<Settings^>
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^>
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>
echo     ^<ExecutionTimeLimit^>PT0S^</ExecutionTimeLimit^>
echo   ^</Settings^>
echo   ^<Actions^>
echo     ^<Exec^>
echo       ^<Command^>%PYTHONW%^</Command^>
echo       ^<Arguments^>"%PICKER%" --backend %BACKEND%^</Arguments^>
echo     ^</Exec^>
echo   ^</Actions^>
echo ^</Task^>
) > "%XMLFILE%"

schtasks /create /tn "%TASKNAME%" /xml "%XMLFILE%" /f
if %errorlevel% neq 0 (
    echo ERROR: Could not create task.
    pause
    exit /b 1
)

del "%XMLFILE%" >nul 2>&1

echo Installed for: %USERNAME%
echo Starting picker now...
start "" "%PYTHONW%" "%PICKER%" --backend %BACKEND%
echo Done. Folder picker will auto-start at every login.
timeout /t 3 >nul
