# Launch the RAX robot server DETACHED from whatever shell started it.
#
# Why this exists: running `python stack_mission2.py` inside an agent/CI background
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
param([switch]$Stop, [switch]$Status, [switch]$Force)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = Join-Path $root 'stack_mission2_stdout.log'

function Get-Server {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*stack_mission2*' }
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
$p = Start-Process -FilePath 'python' -ArgumentList 'stack_mission2.py' `
       -WorkingDirectory $root -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err"
"started pid=$($p.Id) - detached, survives this shell"
"log: $log"
