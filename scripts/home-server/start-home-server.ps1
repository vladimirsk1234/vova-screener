# Start NestJS API + React web on this PC (Mongo starts with the API).
# Keeps running until Stop-Home-Server or the window is closed.
param(
  [switch]$NoBrowser,
  [switch]$Hidden,
  [switch]$Detach
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot

$StateDir = Join-Path $RepoRoot '.data\home-server'
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$PidFile = Join-Path $StateDir 'home-server.pids.json'
$LogDir = Join-Path $StateDir 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Test-PortListening([int]$Port) {
  $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  return $null -ne $conn
}

function Stop-PortListeners([int]$Port) {
  $pids = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique)
  foreach ($procId in $pids) {
    if ($procId -and $procId -ne 0) {
      Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
  }
}

Write-Host ''
Write-Host ' ========================================'
Write-Host '  Vova home server (API + Web + Mongo)'
Write-Host ' ========================================'
Write-Host ''

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  throw 'npm not found. Install Node.js 20+ from https://nodejs.org'
}

if (-not (Test-Path (Join-Path $RepoRoot 'node_modules\concurrently'))) {
  Write-Host 'Installing npm dependencies...'
  npm install
  if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
}

Write-Host 'Freeing ports 3001 and 5173 if needed...'
Stop-PortListeners 3001
Stop-PortListeners 5173
Start-Sleep -Seconds 1

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outLog = Join-Path $LogDir "home-server-$stamp.out.log"
$errLog = Join-Path $LogDir "home-server-$stamp.err.log"

$psi = @{
  FilePath               = 'cmd.exe'
  ArgumentList           = @('/c', 'npm run dev')
  WorkingDirectory       = $RepoRoot
  RedirectStandardOutput = $outLog
  RedirectStandardError  = $errLog
  PassThru               = $true
}
if ($Hidden) {
  $psi.WindowStyle = 'Hidden'
} else {
  $psi.WindowStyle = 'Normal'
}

Write-Host "Starting npm run dev (logs: $LogDir)"
$proc = Start-Process @psi

$state = @{
  startedAt   = (Get-Date).ToString('o')
  npmPid      = $proc.Id
  outLog      = $outLog
  errLog      = $errLog
  apiPort     = 3001
  webPort     = 5173
}
$state | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

$deadline = (Get-Date).AddMinutes(3)
$apiReady = $false
$webReady = $false
while ((Get-Date) -lt $deadline) {
  if (-not $apiReady -and (Test-PortListening 3001)) { $apiReady = $true }
  if (-not $webReady -and (Test-PortListening 5173)) { $webReady = $true }
  if ($apiReady -and $webReady) { break }
  if ($proc.HasExited) {
    throw "npm run dev exited early (code $($proc.ExitCode)). See $errLog"
  }
  Start-Sleep -Seconds 2
}

if (-not $apiReady) {
  Write-Warning "API port 3001 not listening yet — check $errLog (Mongo first boot can take minutes)."
}
if (-not $webReady) {
  Write-Warning "Web port 5173 not listening yet — check $errLog"
}

try {
  $health = Invoke-RestMethod -Uri 'http://127.0.0.1:3001/api/health' -TimeoutSec 10
  Write-Host ("API healthy. Universe total={0} cache.series={1}" -f $health.universe.total, $health.cache.series)
} catch {
  Write-Warning "API /health not ready yet: $($_.Exception.Message)"
}

$lan = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } |
  Select-Object -ExpandProperty IPAddress -First 1

Write-Host ''
Write-Host ' Local:  http://localhost:5173'
if ($lan) { Write-Host " LAN:    http://${lan}:5173" }
Write-Host ' API:    http://localhost:3001/api/health'
Write-Host ' Phone from anywhere: run RUN_TUNNEL.bat (Cloudflare Tunnel)'
Write-Host ''
Write-Host " PID file: $PidFile"
Write-Host ' Stop with: powershell -File scripts\home-server\stop-home-server.ps1'
Write-Host ''

if (-not $NoBrowser -and $webReady) {
  Start-Process 'http://localhost:5173'
}

# Keep the script attached when launched interactively (not Task Scheduler / Detach).
if (-not $Hidden -and -not $Detach) {
  Write-Host "Home server running in background npm process."
  Write-Host "Press Enter to detach (server keeps running)..."
  try { [void][Console]::ReadLine() } catch { }
}
