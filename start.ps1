param([switch]$ViewOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($ViewOnly) {
    Start-Process -FilePath (Join-Path $PSScriptRoot 'results\dashboard.html')
    exit
}
$demoPython = $null
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) { $demoPython = $localPython }
elseif (Test-Path -LiteralPath $bundledPython) { $demoPython = $bundledPython }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $demoPython = (Get-Command python).Source }
else { throw 'Install Python 3.11+ and run: python -m pip install -r requirements.txt' }
& $demoPython -c 'import numpy'
if ($LASTEXITCODE -ne 0) { throw 'NumPy is missing. Run: python -m pip install -r requirements.txt' }
& $demoPython -X utf8 (Join-Path $PSScriptRoot 'run_demo.py') --privacy-demo --open
if ($LASTEXITCODE -ne 0) { throw 'Demo failed; check the error above.' }
