# Erstellt .venv und installiert alle Pakete (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
& $py -c "import sys; assert sys.version_info >= (3, 10)"
if ($LASTEXITCODE -ne 0) { Write-Error "Python 3.10 oder neuer nötig"; exit 1 }

if (-not (Test-Path .venv)) { & $py -m venv .venv }
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -r requirements.txt
& $vpy -c "import torch; print(f'OK: torch {torch.__version__}, cuda = {torch.cuda.is_available()}')"
Write-Host "Aktivieren mit:  .\.venv\Scripts\Activate.ps1"
