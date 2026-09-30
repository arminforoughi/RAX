# Start a RAX server DETACHED, so it outlives the shell that launched it.
#
#   .\run.ps1 -App pick -Arm so101 -Port COM4          general pick, UI on :8484
#   .\run.ps1 -App tube -Arm so101 -Port COM4          tube sorting, UI on :8486
#   .\run.ps1 -App pick -Arm x250  -Port COM5 -Camera 1
#   .\run.ps1 -Stop                                    stop whichever is running
#
# Only one server can hold the arm's serial port and camera, so starting one stops
# the other first (gracefully: POST /shutdown releases the camera before exiting).
#
# ASCII only: Windows PowerShell 5.1 reads .ps1 files as ANSI.
param([ValidateSet('pick', 'tube')][string]$App = 'pick',
      [ValidateSet('so101', 'x250')][string]$Arm = 'so101',
      [string]$Port, [int]$Camera = 0, [switch]$Stop)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $root
$apps = @{ pick = @{ dir = 'pick_server'; http = 8484 }; tube = @{ dir = 'tube_sorting'; http = 8486 } }

function Get-Servers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match '(pick_server|tube_sorting).server\.py' }
}

function Stop-Servers {
  foreach ($a in $apps.Values) {
    try { Invoke-RestMethod -Method Post "http://127.0.0.1:$($a.http)/shutdown" -TimeoutSec 5 | Out-Null } catch { }
  }
  Start-Sleep -Seconds 3
  Get-Servers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

Stop-Servers
if ($Stop) { 'stopped'; return }

if (-not $Port) { $Port = if ($Arm -eq 'x250') { 'COM5' } else { 'COM4' } }
$dir = Join-Path $root $apps[$App].dir
$log = Join-Path $dir 'server.log'

# lerobot asks on stdin whether to reuse the stored calibration; blank lines take the
# default (reuse). Detached there is no console, so feed it a file.
$stdin = Join-Path $root '.server_stdin.txt'
if (-not (Test-Path $stdin)) {
  Set-Content -Path $stdin -Value ([string]::Join("`r`n", (1..12 | ForEach-Object { '' }))) -NoNewline -Encoding ascii
}
$env:PYTHONUNBUFFERED = '1'
$pyargs = @("examples\$($apps[$App].dir)\server.py", '--arm', $Arm, '--port', $Port, '--camera', $Camera)
$p = Start-Process -FilePath 'python' -ArgumentList $pyargs -WorkingDirectory $repo `
       -WindowStyle Hidden -PassThru -RedirectStandardInput $stdin `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err"
"started $App on $Arm ($Port) pid=$($p.Id) - UI http://127.0.0.1:$($apps[$App].http)/ - log $log"
