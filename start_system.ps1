param(
    [ValidateSet('numpy','torch-cpu','cuda')][string]$Backend = 'numpy',
    [string]$Output = 'system_runs/demo',
    [int]$Rounds = 30,
    [double]$Sigma = 0,
    [int]$MonitorPort = 8767,
    [switch]$Resume,
    [switch]$Open
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv and install requirements-system.txt first (see README).'
}
$runArgs = @('-m','energy_fl.runner','--backend',$Backend,'--workers','1',
             '--rounds',"$Rounds",'--output',$Output,'--sigma',"$Sigma")
if ($Resume) { $runArgs += '--resume' }
$monitor = $null
if ($Open) {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $MonitorPort)
    try { $probe.Start() } catch { throw "Monitor port $MonitorPort is busy; choose another -MonitorPort." }
    finally { $probe.Stop() }
    # Only the local read-only monitor runs in the background; training stays visible.
    $absoluteOutput = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Output))
    $monitorArgs = '-m energy_fl.monitor --output "' + $absoluteOutput + '" --port ' + $MonitorPort + ' --open'
    $monitor = Start-Process -FilePath $python -ArgumentList $monitorArgs -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
}
try {
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE. Use -Resume after fixing the cause." }
    if ($Open) {
        Write-Host 'Monitor remains available during this window. Press Enter to stop it.'
        Read-Host | Out-Null
    }
} finally {
    if ($null -ne $monitor -and -not $monitor.HasExited) { Stop-Process -Id $monitor.Id }
}
