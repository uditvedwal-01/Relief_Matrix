# Beginner-friendly environment setup for Windows PowerShell.
# Run from project root:
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

Write-Host "=== Relief Matrix Python Environment Setup ===" -ForegroundColor Cyan

# 1) Create virtual environment if missing
if (-Not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment already exists." -ForegroundColor Green
}

# 2) Activate virtual environment in current shell
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1

# 3) Upgrade pip/setuptools/wheel before installing dependencies
Write-Host "[3/4] Upgrading pip tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# 4) Install project dependencies
Write-Host "[4/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next checks:" -ForegroundColor Cyan
Write-Host "  python check_env.py"
Write-Host "  python test_ml.py"
