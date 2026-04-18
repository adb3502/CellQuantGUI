$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("C:\Users\Saini Lab\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\CellQuantPicker.lnk")
$s.TargetPath = 'D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe'
$s.Arguments = '"D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3\cellquant-picker.py" --backend http://localhost:7860 --username admin'
$s.WindowStyle = 7
$s.Save()
Write-Host "Done. Picker will start automatically when Saini Lab logs in."
