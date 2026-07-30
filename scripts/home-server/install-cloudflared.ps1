# Install cloudflared into scripts/home-server/bin (no admin required).
$ErrorActionPreference = "Stop"
$BinDir = Join-Path $PSScriptRoot "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Exe = Join-Path $BinDir "cloudflared.exe"

if (Test-Path $Exe) {
  Write-Host "Already present: $Exe"
  & $Exe --version
  exit 0
}

$arch = if ($env:PROCESSOR_ARCHITECTURE -match "ARM64") { "arm64" } else { "amd64" }
$url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-$arch.exe"
Write-Host "Downloading cloudflared ($arch)..."
Invoke-WebRequest -Uri $url -OutFile $Exe -UseBasicParsing
Unblock-File $Exe -ErrorAction SilentlyContinue
& $Exe --version
Write-Host ""
Write-Host "Next:"
Write-Host "  1) Quick try (no domain):  .\start-tunnel.ps1 -Quick"
Write-Host "  2) Named tunnel:           cloudflared tunnel login"
Write-Host "     then copy cloudflared\config.example.yml to cloudflared\config.yml and edit hostname"
Write-Host "     cloudflared tunnel create vova-home"
Write-Host "     .\start-tunnel.ps1"
