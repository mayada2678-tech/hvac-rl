<#
.SYNOPSIS
    Klont BOPTEST (falls nötig) und startet den lokalen BOPTEST-Dienst per Docker Compose.

.DESCRIPTION
    BOPTEST läuft nicht als Python-Paket, sondern als eigener Docker-Dienst (REST-API auf
    http://127.0.0.1:8000), siehe workbench.md. Dieses Skript:
      1. klont https://github.com/ibpsa/project1-boptest.git nach ..\boptest (sofern nicht
         schon vorhanden) — bewusst NICHT Teil dieses Projekt-Repos, siehe workbench.md,
      2. patcht docker-compose.yml: MinIO-Images auf quay.io (MinIO hat seine Images 2025 von
         Docker Hub entfernt) und `mc config host add` auf `mc alias set` (das neuere
         `mc`-Client-Image von quay.io hat den alten Befehl entfernt) — betrifft nur diesen
         einen BOPTEST-Commit-Stand, nicht dieses Projekt,
      3. baut und startet die Dienste `web`, `worker`, `provision` im Hintergrund.

    Der erste Start baut mehrere Docker-Images (u. a. eine Conda-Umgebung mit pyfmi für die
    Gebäudesimulation) und kann deutlich über 10 Minuten dauern. Spätere Starts sind schnell.

.PARAMETER BoptestDir
    Zielordner für den BOPTEST-Checkout. Standard: Geschwisterordner ..\boptest (relativ zu
    diesem Projekt-Wurzelverzeichnis).

.PARAMETER Workers
    Anzahl paralleler BOPTEST-Arbeitsprozesse. Jedes laufende Training braucht davon ZWEI
    gleichzeitig (Trainings- und Eval-Umgebung) — mit nur einem Worker (BOPTESTs Standard)
    blockiert das Training beim Aufbau der zweiten Umgebung. Standard hier: 6 (reicht für
    SAC + PPO + TD3 gleichzeitig, je 2 Worker). Kleiner wählen, um Ressourcen zu sparen, wenn
    nur ein Algorithmus gleichzeitig trainiert wird (dann genügt 2).
#>
param(
    [string]$BoptestDir = "",
    [int]$Workers = 6
)

if (-not $BoptestDir) {
    $projectRoot = Split-Path $PSScriptRoot -Parent
    $BoptestDir = Join-Path (Split-Path $projectRoot -Parent) "boptest"
}

if (-not (Test-Path $BoptestDir)) {
    Write-Host "Klone BOPTEST nach $BoptestDir ..."
    git clone --depth 1 https://github.com/ibpsa/project1-boptest.git $BoptestDir
    if ($LASTEXITCODE -ne 0) { throw "git clone fehlgeschlagen." }
}

$compose = Join-Path $BoptestDir "docker-compose.yml"
$needsPatch = (Select-String -Path $compose -Pattern "image: minio/mc" -Quiet -ErrorAction SilentlyContinue) `
    -or (Select-String -Path $compose -Pattern "mc config host add" -Quiet -ErrorAction SilentlyContinue)
if ($needsPatch) {
    Write-Host "Patche docker-compose.yml (veraltete MinIO-Referenzen, alte mc-Syntax) ..."
    (Get-Content $compose) `
        -replace 'image: minio/minio:.*', 'image: quay.io/minio/minio:latest' `
        -replace 'image: minio/mc:.*', 'image: quay.io/minio/mc:latest' `
        -replace '/usr/bin/mc config host add myminio', '/usr/bin/mc alias set myminio' `
        | Set-Content $compose
}

Write-Host "Starte BOPTEST mit $Workers Worker(n) (web, worker, provision) — beim ersten Mal dauert der Build lange ..."
Push-Location $BoptestDir
try {
    docker compose up --scale worker=$Workers web worker provision --build -d
    if ($LASTEXITCODE -ne 0) { throw "docker compose up fehlgeschlagen." }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "BOPTEST laeuft (oder startet gerade) auf http://127.0.0.1:8000"
Write-Host "Stoppen mit: scripts\stop_boptest.ps1"
