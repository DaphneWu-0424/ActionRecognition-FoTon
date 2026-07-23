$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontend = Join-Path $projectRoot "frontend"
$vite = Join-Path $frontend "node_modules\vite\bin\vite.js"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found: $python"
}
if (-not (Test-Path -LiteralPath $vite)) {
    throw "Frontend dependencies are missing. Run npm install in frontend first."
}

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$projectRoot;$projectRoot\backend"

$processes = @()

try {
    Push-Location $projectRoot
    try {
        & $python -m alembic upgrade head
        if ($LASTEXITCODE -ne 0) {
            throw "Database migration failed."
        }
    }
    finally {
        Pop-Location
    }

    $processes += Start-Process `
        -FilePath $python `
        -ArgumentList @(
            "-m", "uvicorn", "app.main:app",
            "--app-dir", "backend",
            "--host", "127.0.0.1",
            "--port", "8010"
        ) `
        -WorkingDirectory $projectRoot `
        -NoNewWindow `
        -PassThru

    $processes += Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "app.worker") `
        -WorkingDirectory $projectRoot `
        -NoNewWindow `
        -PassThru

    $processes += Start-Process `
        -FilePath (Get-Command node).Source `
        -ArgumentList @(
            $vite,
            "--host", "127.0.0.1",
            "--port", "5173",
            "--strictPort"
        ) `
        -WorkingDirectory $frontend `
        -NoNewWindow `
        -PassThru

    Write-Host ""
    Write-Host "Action Finder is starting:"
    Write-Host "  Web:     http://127.0.0.1:5173"
    Write-Host "  Swagger: http://127.0.0.1:8010/docs"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop API, Worker, and frontend."

    while ($true) {
        Start-Sleep -Seconds 1
        $stopped = $processes | Where-Object { $_.HasExited }
        if ($stopped) {
            throw "One application process exited unexpectedly."
        }
    }
}
finally {
    foreach ($process in $processes) {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
        }
    }
    $env:PYTHONPATH = $previousPythonPath
}
