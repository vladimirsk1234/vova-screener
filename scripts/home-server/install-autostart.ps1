# Register Task Scheduler job: start home server at user logon (PC always-on server).
# Run once in an elevated or normal PowerShell from the repo.
param(
  [string]$TaskName = 'VovaHomeServer'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$StartScript = Join-Path $PSScriptRoot 'start-home-server.ps1'

if (-not (Test-Path $StartScript)) {
  throw "Missing $StartScript"
}

$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -NoBrowser -Hidden"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit (New-TimeSpan -Days 0) `
  -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null

Write-Host "Scheduled task '$TaskName' registered (At logon → start-home-server.ps1)."
Write-Host 'Also disable Sleep in Windows Power settings so the PC stays reachable.'
Write-Host 'Optional tunnel autostart: run install-tunnel-autostart.ps1 after cloudflared login.'
