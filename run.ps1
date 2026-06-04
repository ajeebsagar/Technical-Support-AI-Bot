# ─────────────────────────────────────────────────────────────────────────────
# run.ps1 — one-command setup + test for Windows PowerShell / VS Code.
#
# Usage (from the upwork-rag-bot folder):
#   .\run.ps1            # full pipeline: venv -> install -> ingest -> evaluate
#   .\run.ps1 -Serve     # same as above, then launches the Streamlit app
#   .\run.ps1 -SkipInstall   # skip venv/pip (deps already installed) -> ingest+eval
#
# If you hit "running scripts is disabled on this system", run once:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# ─────────────────────────────────────────────────────────────────────────────
param(
    [switch]$Serve,        # launch Streamlit at the end
    [switch]$SkipInstall   # skip venv creation + pip install
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot   # always run relative to this script's folder

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# ── 1. Virtual environment + dependencies ───────────────────────────────────
$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-not $SkipInstall) {
    if (-not (Test-Path $venvPython)) {
        Write-Step "Creating virtual environment (venv)"
        python -m venv venv
    } else {
        Write-Step "Virtual environment already exists - reusing it"
    }

    Write-Step "Upgrading pip"
    & $venvPython -m pip install --upgrade pip

    Write-Step "Installing dependencies (first run downloads PyTorch etc. - be patient)"
    & $venvPython -m pip install -r requirements.txt
} else {
    Write-Step "Skipping install (-SkipInstall)"
    if (-not (Test-Path $venvPython)) {
        Write-Host "venv not found and -SkipInstall was set. Falling back to system 'python'." -ForegroundColor Yellow
        $venvPython = "python"
    }
}

# ── 2. Sanity: confirm the API key is present ───────────────────────────────
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env not found. Copy .env.example to .env and add your key." -ForegroundColor Red
    exit 1
}

# ── 3. Build the knowledge base ─────────────────────────────────────────────
Write-Step "Ingesting PDF -> chunks -> embeddings -> ChromaDB"
& $venvPython ingest.py
if ($LASTEXITCODE -ne 0) { Write-Host "Ingestion failed." -ForegroundColor Red; exit 1 }

# ── 4. Run the ground-truth evaluation (the automated test) ─────────────────
Write-Step "Running ground-truth evaluation"
& $venvPython evaluate.py
if ($LASTEXITCODE -ne 0) { Write-Host "Evaluation failed." -ForegroundColor Red; exit 1 }

# ── 5. Optionally launch the UI ─────────────────────────────────────────────
if ($Serve) {
    Write-Step "Launching Streamlit app (Ctrl+C to stop)"
    & $venvPython -m streamlit run app.py
} else {
    Write-Host "`nAll done. Launch the UI any time with:" -ForegroundColor Green
    Write-Host "  .\run.ps1 -Serve -SkipInstall" -ForegroundColor Green
}
