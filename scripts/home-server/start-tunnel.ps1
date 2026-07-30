# Expose local React (:5173) via Cloudflare Tunnel for phone access from anywhere.
# Quick mode uses a temporary trycloudflare.com URL (Vite proxies /api to Nest).
# Named mode uses cloudflared/config.yml ingress (routes /api to :3001, / to :5173).
param(
  [switch]$Quick,
  [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LocalBin = Join-Path $PSScriptRoot "bin\cloudflared.exe"

function Get-Cloudflared {
  if (Test-Path $LocalBin) { return $LocalBin }
  $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  Write-Host "cloudflared not found - installing local copy..."
  & (Join-Path $PSScriptRoot "install-cloudflared.ps1")
  if (-not (Test-Path $LocalBin)) { throw "cloudflared install failed" }
  return $LocalBin
}

$cf = Get-Cloudflared

try {
  Invoke-RestMethod -Uri "http://127.0.0.1:3001/api/health" -TimeoutSec 5 | Out-Null
} catch {
  Write-Warning "API not healthy on :3001. Start home server first (RUN_HOME_SERVER.bat)."
}

if ($Quick -or -not $ConfigPath) {
  $defaultConfig = Join-Path $RepoRoot "cloudflared\config.yml"
  if (-not $Quick -and (Test-Path $defaultConfig)) {
    $ConfigPath = $defaultConfig
  }
}

if ($Quick -or -not $ConfigPath -or -not (Test-Path $ConfigPath)) {
  Write-Host ""
  Write-Host " Quick tunnel -> http://127.0.0.1:5173"
  Write-Host " Copy the https://....trycloudflare.com URL onto your phone (use mobile data to test)."
  Write-Host " Ctrl+C stops the tunnel (home server keeps running)."
  Write-Host ""
  & $cf tunnel --url "http://127.0.0.1:5173" --no-autoupdate
  exit $LASTEXITCODE
}

Write-Host "Named tunnel using config: $ConfigPath"
& $cf tunnel --config $ConfigPath run
exit $LASTEXITCODE
