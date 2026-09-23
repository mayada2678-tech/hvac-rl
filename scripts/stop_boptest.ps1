<#
.SYNOPSIS
    Stoppt den lokalen BOPTEST-Dienst sauber (siehe start_boptest.ps1).
#>
param(
    [string]$BoptestDir = ""
)

if (-not $BoptestDir) {
    $projectRoot = Split-Path $PSScriptRoot -Parent
    $BoptestDir = Join-Path (Split-Path $projectRoot -Parent) "boptest"
}

if (-not (Test-Path $BoptestDir)) {
    Write-Host "Kein BOPTEST-Checkout unter $BoptestDir gefunden — nichts zu stoppen."
    exit 0
}

Push-Location $BoptestDir
try {
    docker compose down
} finally {
    Pop-Location
}
