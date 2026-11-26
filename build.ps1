$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python virtual environment. Install Python 3.10 or later."
    }
}

& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the application dependencies."
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name PersonCutout `
    --collect-all mediapipe `
    --add-data "models\selfie_segmenter.tflite;models" `
    main.py
if ($LASTEXITCODE -ne 0) {
    throw "Could not build the application."
}

$licenseDirectory = Join-Path $PSScriptRoot "dist\PersonCutout\licenses"
New-Item -ItemType Directory -Force -Path $licenseDirectory | Out-Null
Copy-Item "licenses\APACHE-2.0.txt", "licenses\THIRD_PARTY_NOTICES.txt" -Destination $licenseDirectory -Force

Write-Host "Build complete: $(Join-Path $PSScriptRoot 'dist\PersonCutout\PersonCutout.exe')"
