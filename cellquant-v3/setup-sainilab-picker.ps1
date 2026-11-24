$pythonw = 'D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe'
$picker  = 'D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3\cellquant-picker.py'

if (-not (Test-Path $pythonw)) {
    Write-Host "ERROR: Python not found at $pythonw"
    exit 1
}

$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("C:\Users\Saini Lab\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\CellQuantPicker.lnk")
$s.TargetPath = $pythonw
$s.Arguments = "`"$picker`" --backend http://localhost:7860 --username admin"
$s.WorkingDirectory = 'D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3'
$s.WindowStyle = 7
$s.Save()
Write-Host "Done. Picker will start automatically when Saini Lab logs in."
