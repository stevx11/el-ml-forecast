<#
.SYNOPSIS
    PowerShell equivalent of the Makefile, for Windows.

.DESCRIPTION
    Windows does not ship `make`. This script exposes the same targets so the
    documented workflow works identically on both platforms.

.EXAMPLE
    .\make.ps1 up
    .\make.ps1 ingest -Season E2025
    .\make.ps1 test
    .\make.ps1 help
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'up', 'down', 'logs', 'shell', 'migrate', 'revision',
                 'ingest', 'test', 'lint', 'typecheck', 'fmt', 'check', 'clean')]
    [string]$Target = 'help',

    # Season code for `ingest`.
    [string]$Season = 'E2025',

    # Message for `revision`.
    [string]$Message = ''
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$Compose = @('docker', 'compose')
$AlembicIni = 'backend/alembic.ini'

function Invoke-Step {
    <#
        Run a command and stop the script if it fails. Without this, a failed
        migration would be followed by an ingest against a schema-less database,
        and the real error would scroll away.
    #>
    param([string[]]$Command)

    Write-Host "> $($Command -join ' ')" -ForegroundColor DarkGray
    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

function Test-Prerequisites {
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
}

function Show-Help {
    Write-Host ''
    Write-Host 'EuroLeague AI - available targets' -ForegroundColor Cyan
    Write-Host ''
    $targets = [ordered]@{
        'up'        = 'Start the stack and apply migrations'
        'down'      = 'Stop the stack'
        'logs'      = 'Tail backend logs'
        'shell'     = 'Open a shell in the backend container'
        'migrate'   = 'Apply database migrations'
        'revision'  = 'Autogenerate a migration:  .\make.ps1 revision -Message "add x"'
        'ingest'    = 'Ingest a season:           .\make.ps1 ingest -Season E2025'
        'test'      = 'Run the full test suite'
        'lint'      = 'Run ruff'
        'typecheck' = 'Run mypy'
        'fmt'       = 'Format and autofix with ruff'
        'check'     = 'lint + typecheck + test (what CI would run)'
        'clean'     = 'Stop the stack and delete volumes'
    }
    foreach ($key in $targets.Keys) {
        Write-Host ('  {0,-10} {1}' -f $key, $targets[$key])
    }
    Write-Host ''
    Write-Host 'First run:' -ForegroundColor Cyan
    Write-Host '  .\make.ps1 up'
    Write-Host '  .\make.ps1 ingest -Season E2025'
    Write-Host ''
    Write-Host 'Then open http://localhost:8000/docs'
    Write-Host ''
}

switch ($Target) {
    'help' { Show-Help }

    'up' {
        Test-Prerequisites
        Invoke-Step ($Compose + @('up', '-d', '--build'))
        # Compose waits for the postgres healthcheck, but the backend container
        # can still win the race to the first connection. A short pause here is
        # cheaper than an unclear migration failure.
        Start-Sleep -Seconds 3
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'alembic', '-c', $AlembicIni, 'upgrade', 'head'))
        Write-Host ''
        Write-Host 'Stack is up.' -ForegroundColor Green
        Write-Host '  API docs : http://localhost:8000/docs'
        Write-Host '  Health   : http://localhost:8000/health/ready'
        Write-Host ''
        Write-Host 'Next:  .\make.ps1 ingest -Season E2025' -ForegroundColor Cyan
    }

    'down'    { Invoke-Step ($Compose + @('down')) }
    'logs'    { Invoke-Step ($Compose + @('logs', '-f', 'backend')) }
    'shell'   { Invoke-Step ($Compose + @('exec', 'backend', 'bash')) }
    'migrate' { Invoke-Step ($Compose + @('exec', '-T', 'backend', 'alembic', '-c', $AlembicIni, 'upgrade', 'head')) }

    'revision' {
        if (-not $Message) {
            throw 'revision requires a message:  .\make.ps1 revision -Message "add lineup confidence"'
        }
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'alembic', '-c', $AlembicIni,
                                  'revision', '--autogenerate', '-m', $Message))
        Write-Host ''
        Write-Host 'Read the generated revision before applying it.' -ForegroundColor Yellow
        Write-Host 'Alembic cannot infer intent: a renamed column looks exactly like a drop plus an add.'
    }

    'ingest' {
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'euroleague-ai', 'ingest', 'all', '--season', $Season))
    }

    'test'      { Invoke-Step ($Compose + @('exec', '-T', 'backend', 'pytest')) }
    'lint'      { Invoke-Step ($Compose + @('exec', '-T', 'backend', 'ruff', 'check', 'backend')) }
    'typecheck' { Invoke-Step ($Compose + @('exec', '-T', 'backend', 'mypy')) }

    'fmt' {
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'ruff', 'format', 'backend'))
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'ruff', 'check', '--fix', 'backend'))
    }

    'check' {
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'ruff', 'check', 'backend'))
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'mypy'))
        Invoke-Step ($Compose + @('exec', '-T', 'backend', 'pytest'))
    }

    'clean' { Invoke-Step ($Compose + @('down', '-v')) }
}
