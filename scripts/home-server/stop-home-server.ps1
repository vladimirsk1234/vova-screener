# Stop home-server npm process and listeners on 3001 / 5173.
$ErrorActionPreference = 'Continue'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PidFile = Join-Path $RepoRoot '.data\home-server\home-server.pids.json'

function Stop-PortListeners([int]$Port) {
  $pids = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique)
  foreach ($procId in $pids) {
    if ($procId -and $procId -ne 0) {
      Write-Host "Stopping PID $procId on port $Port"
      Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
  }
}

if (Test-Path $PidFile) {
  try {
    $state = Get-Content $PidFile -Raw | ConvertFrom-Json
    if ($state.npmPid) {
      Write-Host "Stopping npm PID $($state.npmPid)"
      Stop-Process -Id $state.npmPid -Force -ErrorAction SilentlyContinue
      # Also stop child tree if still around
      Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ParentProcessId -eq $state.npmPid } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
  } catch {
    Write-Warning "Could not read PID file: $($_.Exception.Message)"
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

Stop-PortListeners 3001
Stop-PortListeners 5173
# Vite often leaves esbuild helpers
Get-Process -Name esbuild -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host 'Home server stopped.'
