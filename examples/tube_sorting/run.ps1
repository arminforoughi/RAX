# Start the tube-sorting server DETACHED, so it outlives the shell that launched it.
#
#   .\run.ps1            start (refuses if one is already up)
#   .\run.ps1 -Force     replace a running one
#   .\run.ps1 -Stop      stop it (releases the camera and the bus first)
#   .\run.ps1 -Port COM4
#
# ASCII only: Windows PowerShell 5.1 reads .ps1 files as ANSI.
param([switch]$Stop, [switch]$Force, [string]$Port)

if ($Port) { $env:RAX_ARM_PORT = $Port }
elseif (-not $env:RAX_ARM_PORT) { $env:RAX_ARM_PORT = 'COM4' }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent (Split-Path -Parent $root)
$log  = Join-Path $root 'server.log'

function Get-Server {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*tube_sorting*server.py*' }
}

function Stop-Server {
  # POST /shutdown first: a killed process leaves the OAK-D booted with no owner,
  # and the next start then fails with "No available devices" until it is unplugged.
  try { Invoke-RestMethod -Method Post http://127.0.0.1:8486/shutdown -TimeoutSec 5 | Out-Null } catch { }
  Start-Sleep -Seconds 3
  Get-Server | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

if ($Stop) { Stop-Server; 'stopped'; return }

if (Get-Server) {
  if (-not $Force) { 'already running. Use -Force to replace it.'; return }
  Stop-Server
}

# lerobot asks on stdin whether to reuse the stored calibration; blank lines take the
# default (reuse). Detached there is no console, so feed it a file.
$stdin = Join-Path $root '.server_stdin.txt'
if (-not (Test-Path $stdin)) {
  Set-Content -Path $stdin -Value ([string]::Join("`r`n", (1..12 | ForEach-Object { '' }))) -NoNewline -Encoding ascii
}
$env:PYTHONUNBUFFERED = '1'
$p = Start-Process -FilePath 'python' -ArgumentList 'examples\tube_sorting\server.py' `
       -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
       -RedirectStandardInput $stdin -RedirectStandardOutput $log -RedirectStandardError "$log.err"
"started pid=$($p.Id) on $env:RAX_ARM_PORT - log: $log"
