param([string]$TaskName = 'VovaHomeServer')
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Scheduled task '$TaskName' removed (if it existed)."
