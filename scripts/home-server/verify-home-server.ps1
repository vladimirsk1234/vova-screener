# Verify local home server (+ optional public tunnel URL).
param(
  [string]$PublicBaseUrl,
  [switch]$StartScanSmoke
)

$ErrorActionPreference = 'Stop'
$failed = 0

function Assert-Ok([string]$Name, [scriptblock]$Body) {
  try {
    & $Body
    Write-Host "OK  $Name"
  } catch {
    Write-Host "FAIL $Name - $($_.Exception.Message)"
    $script:failed++
  }
}

Assert-Ok 'API port 3001 listening' {
  if (-not (Get-NetTCPConnection -LocalPort 3001 -State Listen -ErrorAction SilentlyContinue)) {
    throw 'nothing listening'
  }
}

Assert-Ok 'Web port 5173 listening' {
  if (-not (Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue)) {
    throw 'nothing listening'
  }
}

Assert-Ok 'GET /api/health' {
  $h = Invoke-RestMethod -Uri 'http://127.0.0.1:3001/api/health' -TimeoutSec 15
  if (-not $h.ok) { throw 'ok=false' }
  Write-Host ("     universe.total={0} cache.series={1}" -f $h.universe.total, $h.cache.series)
}

Assert-Ok 'Web root responds' {
  $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/' -UseBasicParsing -TimeoutSec 15
  if ($r.StatusCode -ge 400) { throw "status $($r.StatusCode)" }
}

Assert-Ok 'Vite proxies /api/health' {
  $h = Invoke-RestMethod -Uri 'http://127.0.0.1:5173/api/health' -TimeoutSec 15
  if (-not $h.ok) { throw 'ok=false via proxy' }
}

if ($PublicBaseUrl) {
  $base = $PublicBaseUrl.TrimEnd('/')
  Assert-Ok "Public health via $base/api/health" {
    $h = Invoke-RestMethod -Uri "$base/api/health" -TimeoutSec 30
    if (-not $h.ok) { throw 'ok=false' }
  }
  Assert-Ok "Public web $base/" {
    $r = Invoke-WebRequest -Uri "$base/" -UseBasicParsing -TimeoutSec 30
    if ($r.StatusCode -ge 400) { throw "status $($r.StatusCode)" }
  }
}

if ($StartScanSmoke) {
  Assert-Ok 'Smoke scan MANUAL (AAPL) completes' {
    $body = @{
      source        = 'MANUAL SCAN'
      manualTickers = 'AAPL'
      tf            = 'Daily'
      direction     = 'buy'
      minRr         = 1.5
      riskPerTrade  = 100
      noRrReq       = $false
      useLastHlSl   = $true
      newOnly       = $false
      forceRefresh  = $false
    } | ConvertTo-Json
    $start = Invoke-RestMethod -Uri 'http://127.0.0.1:3001/api/scans' -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 30
    $runId = $start.runId
    if (-not $runId) { throw 'no runId' }
    $deadline = (Get-Date).AddMinutes(3)
    $status = 'queued'
    do {
      Start-Sleep -Seconds 2
      $run = Invoke-RestMethod -Uri "http://127.0.0.1:3001/api/scans/$runId" -TimeoutSec 15
      $status = $run.status
    } while ($status -in @('queued', 'running') -and (Get-Date) -lt $deadline)
    if ($status -ne 'completed') { throw "status=$status" }
    Write-Host ("     runId={0} signals={1}" -f $runId, $run.counters.signals)
  }
}

Write-Host ''
if ($failed -gt 0) {
  Write-Host "FAILED checks: $failed"
  Write-Host "Phone off-LAN checklist: start tunnel, pass -PublicBaseUrl https://....trycloudflare.com, test on mobile data."
  exit 1
}

Write-Host "All local checks passed."
Write-Host "Remote phone test: open the tunnel HTTPS URL on mobile data (Wi-Fi off) and run a scan."
exit 0
