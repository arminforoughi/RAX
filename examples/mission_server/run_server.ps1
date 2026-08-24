# Launch the RAX robot server DETACHED from whatever shell started it.
#
# Why this exists: running `python mission_server.py` inside an agent/CI background
# task ties the server's lifetime to that task. When the task is cleaned up the
# server dies with it - twice this happened MID-PICK, and killing the process while
# the arm is moving is what leaves COM4 in a state that then fails the next connect
# with "Failed to read 'Homing_Offset' ... There is no status packet!".
#
# Start-Process gives it its own process tree, so it keeps running after the
# launching shell exits.
#
#   .\run_server.ps1          start (refuses if one is already up)
#   .\run_server.ps1 -Force   replace a running one
#   .\run_server.ps1 -Stop    stop it
#   .\run_server.ps1 -Status  is it alive?
#
# NOTE: ASCII only. Windows PowerShell 5.1 reads .ps1 as ANSI, so a UTF-8 em dash
# in a string becomes mojibake and throws a ParserError.
param([switch]$Stop, [switch]$Status, [switch]$Force, [string]$Port)

# The arm's serial port. The profile reads RAX_ARM_PORT and has no default, so a
# server started without it connects to '' and dies in robot.connect() - which is
# exactly what happened when this script was first used to relaunch a server that
# had been started by hand in a shell that happened to have the variable set.
# COM4 is the CH343 adapter; -Port overrides, and an already-set env var wins.
if ($Port) { $env:RAX_ARM_PORT = $Port }
elseif (-not $env:RAX_ARM_PORT) { $env:RAX_ARM_PORT = 'COM4' }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = Join-Path $root 'mission_server_stdout.log'

function Get-Server {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*mission_server*' }
}

if ($Status) {
  $p = Get-Server
  if ($p) { "RUNNING  pid=$($p.ProcessId)" } else { 'NOT RUNNING' }
  return
}

if ($Stop) {
  $p = Get-Server
  if (-not $p) { 'not running'; return }
  $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped pid=$($_.ProcessId)" }
  Start-Sleep -Seconds 3
  return
}

$existing = Get-Server
if ($existing -and -not $Force) {
  "already running (pid=$($existing.ProcessId)). Use -Force to replace it."
  return
}
if ($existing) {
  $existing | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  Start-Sleep -Seconds 3          # let COM4 settle before re-opening it
}

$env:PYTHONUNBUFFERED = '1'
"arm port: $env:RAX_ARM_PORT"

# lerobot's bus.connect() asks on STDIN whether to reuse the stored calibration.
# Detached there is no console, so input() raises EOFError and the server dies before
# it ever binds the port. Feed it a file of blank lines: every prompt takes its
# default, which is the stored calibration - the answer we want anyway.
$stdin = Join-Path $root '.server_stdin.txt'
if (-not (Test-Path $stdin)) {
  Set-Content -Path $stdin -Value ([string]::Join("`r`n", (1..12 | ForEach-Object { '' }))) -NoNewline -Encoding ascii
}

$p = Start-Process -FilePath 'python' -ArgumentList 'mission_server.py' `
       -WorkingDirectory $root -WindowStyle Hidden -PassThru `
       -RedirectStandardInput $stdin `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err"
"started pid=$($p.Id) - detached, survives this shell"
"log: $log"
