# Start RAX DETACHED, so it outlives the shell that launched it: the general pick UI on
# :8484 and tube sorting on :8486, in one process sharing the arm.
#
#   .\run.ps1 -Arm so101 -Port COM4
#   .\run.ps1 -Arm x250  -Port COM5 -Camera 1 -NoTubes
#   .\run.ps1 -Stop
#
# Stopping is graceful (POST /shutdown releases the camera before exiting): a killed
# process can leave the camera claimed by nobody.
#
# ASCII only: Windows PowerShell 5.1 reads .ps1 files as ANSI.
param([ValidateSet('so101', 'x250')][string]$Arm = 'so101',
      [string]$Port, [int]$Camera = 0, [switch]$NoTubes, [switch]$Stop)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $root
$log  = Join-Path $root 'server.log'

function Get-Servers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'examples.(server|pick_server.server|tube_sorting.server)\.py' }
}

try { Invoke-RestMethod -Method Post "http://127.0.0.1:8484/shutdown" -TimeoutSec 5 | Out-Null } catch { }
Start-Sleep -Seconds 3
Get-Servers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
if ($Stop) { 'stopped'; return }

if (-not $Port) { $Port = if ($Arm -eq 'x250') { 'COM5' } else { 'COM4' } }

# lerobot asks on stdin whether to reuse the stored calibration; blank lines take the
# default (reuse). Detached there is no console, so feed it a file.
$stdin = Join-Path $root '.server_stdin.txt'
if (-not (Test-Path $stdin)) {
  Set-Content -Path $stdin -Value ([string]::Join("`r`n", (1..12 | ForEach-Object { '' }))) -NoNewline -Encoding ascii
}
$env:PYTHONUNBUFFERED = '1'
$pyargs = @('examples\server.py', '--arm', $Arm, '--port', $Port, '--camera', $Camera)
if ($NoTubes) { $pyargs += '--no-tubes' }
$p = Start-Process -FilePath 'python' -ArgumentList $pyargs -WorkingDirectory $repo `
       -WindowStyle Hidden -PassThru -RedirectStandardInput $stdin `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err"
"started $Arm on $Port pid=$($p.Id) - pick http://127.0.0.1:8484/ tubes http://127.0.0.1:8486/ - log $log"
