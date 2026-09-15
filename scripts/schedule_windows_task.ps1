# Registers a daily Windows Task Scheduler job for the GH-PM25 operational run.
# Run from an elevated or user PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\schedule_windows_task.ps1
# Remove with:  Unregister-ScheduledTask -TaskName "GH-PM25 daily" -Confirm:$false
param(
  [string]$Time = "07:30",           # local time; CAMS 00 UTC run and NASA POWER updates are available by then
  [string]$Root = (Split-Path -Parent $PSScriptRoot)
)
$python = Join-Path $Root ".venv\Scripts\python.exe"
$script = Join-Path $Root "scripts\run_daily.py"
$action = New-ScheduledTaskAction -Execute $python -Argument "`"$script`"" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3) -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName "GH-PM25 daily" -Action $action -Trigger $trigger -Settings $settings -Description "GH-PM25: refresh inputs, map D-7..D+1, update portal" -Force
Write-Host "Registered 'GH-PM25 daily' at $Time"

# Monthly retraining (first Sunday) with fresh ground data
$action2 = New-ScheduledTaskAction -Execute $python -Argument "`"$script`" --retrain" -WorkingDirectory $Root
$trigger2 = New-ScheduledTaskTrigger -Weekly -WeeksInterval 4 -DaysOfWeek Sunday -At "02:00"
Register-ScheduledTask -TaskName "GH-PM25 retrain" -Action $action2 -Trigger $trigger2 -Settings $settings -Description "GH-PM25: monthly re-calibration and retraining" -Force
Write-Host "Registered 'GH-PM25 retrain' (every 4 weeks, Sunday 02:00)"
