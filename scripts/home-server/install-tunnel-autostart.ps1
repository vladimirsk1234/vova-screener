# Autostart Cloudflare Tunnel at logon (after named config.yml exists, or Quick each time).
param(
  [string]$TaskName = 'VovaHomeTunnel',
  [switch]$Quick
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$StartScript = Join-Path $PSScriptRoot 'start-tunnel.ps1'
$extra = if ($Quick) { ' -Quick' } else { '' }
$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`"$extra"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# Delay so API/web can boot first
$trigger.Delay = 'PT45S'
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -RestartCount 5 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit (New-TimeSpan -Days 0) `
  -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
Write-Host "Scheduled task '$TaskName' registered."
if ($Quick) {
  Write-Host 'Quick tunnel URLs change every restart — prefer named tunnel + config.yml for a stable phone bookmark.'
}
