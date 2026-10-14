<#
.SYNOPSIS
    Restore a PostgreSQL dump into the Compose `postgres` service.

.DESCRIPTION
    Brings up only the postgres container, waits for it to accept connections,
    copies the dump inside, and restores it. Handles both custom-format dumps
    (pg_dump -Fc, header "PGDMP") and plain SQL dumps -- the format is detected
    from the file itself rather than from the extension, because .backup, .dump
    and .sql are all used inconsistently in the wild.

.EXAMPLE
    .\scripts\restore-db.ps1 -DumpPath "$env:USERPROFILE\Downloads\euroleague.dump"

.EXAMPLE
    # Wipe existing objects before restoring (use when the database is not empty)
    .\scripts\restore-db.ps1 -DumpPath ..\dump.backup -Clean
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DumpPath,

    # Drop existing objects before restoring. Only meaningful for custom-format
    # dumps; a plain SQL dump does whatever its own header says.
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# --- Preconditions ---------------------------------------------------------

if (-not (Test-Path -LiteralPath $DumpPath)) {
    throw "Dump file not found: $DumpPath"
}
$DumpPath = (Resolve-Path -LiteralPath $DumpPath).Path

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is not installed or not on PATH. Install Docker Desktop and restart your shell.'
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker is installed but not running. Start Docker Desktop and try again.'
}

if (-not (Test-Path '.env')) {
    Write-Host 'No .env found - creating one from .env.example' -ForegroundColor Yellow
    Copy-Item '.env.example' '.env'
}

# Values the compose file falls back to when .env does not override them.
$DbUser = if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { 'euroleague' }
$DbName = if ($env:POSTGRES_DB)   { $env:POSTGRES_DB }   else { 'euroleague' }

# --- Detect the dump format ------------------------------------------------

# Custom-format dumps start with the five bytes "PGDMP". Anything else is
# treated as plain SQL and piped through psql.
$header = [System.IO.File]::ReadAllBytes($DumpPath)[0..4]
$isCustom = (-join ($header | ForEach-Object { [char]$_ })) -eq 'PGDMP'
$format = if ($isCustom) { 'custom (pg_restore)' } else { 'plain SQL (psql)' }
Write-Host "Dump format: $format" -ForegroundColor Cyan

# --- Start postgres only ---------------------------------------------------

Write-Host '> docker compose up -d postgres' -ForegroundColor DarkGray
& docker compose up -d postgres
if ($LASTEXITCODE -ne 0) { throw 'Failed to start the postgres container.' }

Write-Host 'Waiting for postgres to accept connections...' -ForegroundColor DarkGray
$ready = $false
foreach ($attempt in 1..60) {
    & docker compose exec -T postgres pg_isready -U $DbUser -d $DbName *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    throw 'postgres did not become ready within 2 minutes. Check: docker compose logs postgres'
}

# --- Copy the dump into the container --------------------------------------

# Restoring from inside the container avoids needing psql/pg_restore installed
# on Windows, and guarantees the client version matches the server.
$inner = '/tmp/restore.dump'
Write-Host "> docker compose cp `"$DumpPath`" postgres:$inner" -ForegroundColor DarkGray
& docker compose cp $DumpPath "postgres:$inner"
if ($LASTEXITCODE -ne 0) { throw 'Failed to copy the dump into the container.' }

# --- Restore ---------------------------------------------------------------

if ($isCustom) {
    $args = @('compose', 'exec', '-T', 'postgres', 'pg_restore',
              '-U', $DbUser, '-d', $DbName,
              # The dump may have been taken as a different role; keeping its
              # ownership and grants would fail on roles this server lacks.
              '--no-owner', '--no-privileges')
    if ($Clean) { $args += @('--clean', '--if-exists') }
    $args += $inner

    Write-Host "> docker $($args -join ' ')" -ForegroundColor DarkGray
    & docker @args
    # pg_restore exits non-zero on non-fatal errors too (a missing extension, a
    # role it could not map). Surface that without pretending the restore died.
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host "pg_restore finished with exit code $LASTEXITCODE - some items were skipped." -ForegroundColor Yellow
        Write-Host 'Read the errors above: missing extensions and unknown roles are usually harmless.' -ForegroundColor Yellow
    }
}
else {
    Write-Host '> docker compose exec -T postgres psql ...' -ForegroundColor DarkGray
    & docker compose exec -T postgres psql -U $DbUser -d $DbName -v ON_ERROR_STOP=1 -f $inner
    if ($LASTEXITCODE -ne 0) { throw 'psql failed to apply the dump.' }
}

& docker compose exec -T postgres rm -f $inner *> $null

# --- Report ----------------------------------------------------------------

Write-Host ''
Write-Host 'Tables now present:' -ForegroundColor Cyan
& docker compose exec -T postgres psql -U $DbUser -d $DbName -c `
    "select schemaname, relname, n_live_tup as approx_rows from pg_stat_user_tables order by n_live_tup desc limit 20;"

Write-Host ''
Write-Host 'Restore complete.' -ForegroundColor Green
Write-Host 'Next:' -ForegroundColor Cyan
Write-Host '  docker compose up -d --build          # start backend + redis'
Write-Host '  .\make.ps1 migrate                    # bring the schema to head, if the dump was older'
Write-Host '  http://localhost:8000/docs'
