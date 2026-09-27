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
#   .\run_server.ps1 -SkipPreflight   start even if a preflight check trips
#
# Before launching, it preflights the host-side faults written up in
# TROUBLESHOOTING.md and refuses to start rather than dying ~40s later inside
# cam.connect() with an error that reads like dead hardware.
#
# NOTE: ASCII only. Windows PowerShell 5.1 reads .ps1 as ANSI, so a UTF-8 em dash
# in a string becomes mojibake and throws a ParserError.
param([switch]$Stop, [switch]$Status, [switch]$Force, [string]$Port, [switch]$SkipPreflight)

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

# ---------------------------------------------------------------------------
# PREFLIGHT: the two host-side faults that cost days in Sep 2026.
#
# Neither is a hardware fault and neither travels with the device - both live in
# THIS machine's config, which is why the same OAK-D worked fine on a Mac while
# every connect here failed. Without these checks the only symptom is a ~17s
# stall inside dai.Device() and then
#     RuntimeError: Failed to find device after booting, X_LINK_DEVICE_NOT_FOUND
# which reads like a dead camera and sends you hunting cables for a week.
# See TROUBLESHOOTING.md for the full write-up.
#
#   1. The OAK-D enumerates under TWO ids: 03E7:2485 (Movidius bootloader, what
#      you see while it sits idle) and 03E7:F63B (Luxonis Device, what it becomes
#      AFTER depthai uploads firmware). A disable flag on F63B is INVISIBLE while
#      the camera is idle - Device Manager shows a healthy 2485 - and only bites
#      during the ~6s boot window. ConfigFlags=1 is CONFIGFLAG_DISABLED.
#   2. usbipd bind (STATE=Shared) hands the device toward WSL and swaps its
#      driver. A PERSISTED binding re-applies itself on every replug, so the
#      fault comes back every single time the camera is plugged in.
# ---------------------------------------------------------------------------
$OAK_F63B = 'HKLM:\SYSTEM\CurrentControlSet\Enum\USB\VID_03E7&PID_F63B\19443010F15FF81200'

function Test-RigHealth {
  $problems = @()

  $cf = (Get-ItemProperty $OAK_F63B -Name ConfigFlags -ErrorAction SilentlyContinue).ConfigFlags
  if ($cf -eq 1) {
    $problems += @"
OAK-D booted identity (PID_F63B) is DISABLED (ConfigFlags=1).
    depthai boots the camera, Windows disables it the instant it enumerates, and
    the server dies in cam.connect() with X_LINK_DEVICE_NOT_FOUND.
    Fix from an ADMIN PowerShell, then unplug and replug the camera:
      Set-ItemProperty '$OAK_F63B' -Name ConfigFlags -Value 0 -Type DWord
"@
  }

  $shared = @()
  try {
    foreach ($line in (usbipd list)) {   # try/catch below covers usbipd being absent
      if ($line -match '03e7:(2485|f63b)|1a86:55d3') {
        # -match is CASE-INSENSITIVE and 'Not shared' contains 'shared', so matching
        # on 'Shared' alone flags every healthy device. Exclude the negative first.
        if ($line -notmatch 'Not\s+shared' -and $line -match 'Shared') { $shared += $line.Trim() }
      }
    }
  } catch { }
  if ($shared.Count) {
    $problems += ("usbipd has the rig BOUND (Shared) - handed toward WSL:" + "`n      " +
      ($shared -join "`n      ") + "`n" +
      "    Fix from an ADMIN PowerShell. The busid moves on every replug, so read it" + "`n" +
      "    fresh from 'usbipd list' rather than reusing an old one:" + "`n" +
      "      usbipd unbind --busid <busid>" + "`n" +
      "    If it comes back after a replug, a PERSISTED binding is re-applying it -" + "`n" +
      "    those show under 'Persisted' in usbipd list, and unbind clears them too.")
  }

  $ports = [System.IO.Ports.SerialPort]::getportnames()
  if ($ports -notcontains $env:RAX_ARM_PORT) {
    $seen = 'none'
    if ($ports.Count) { $seen = ($ports -join ', ') }
    $problems += ("arm port $env:RAX_ARM_PORT not present. COM ports seen: $seen" + "`n" +
      "    If the list is EMPTY the CH343 has not enumerated at all - that is a cable" + "`n" +
      "    or port fault, not a config one, and no amount of retrying fixes it." + "`n" +
      "    If it enumerated somewhere else, pass -Port COMn.")
  }
  return $problems
}

if (-not $SkipPreflight) {
  $issues = @(Test-RigHealth)
  if ($issues.Count) {
    ''
    'PREFLIGHT FAILED - the server cannot connect until these are fixed:'
    ''
    foreach ($p in $issues) { "  * $p"; '' }
    'Not starting. See TROUBLESHOOTING.md. Use -SkipPreflight to start anyway.'
    return
  }
  'preflight: OK'
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

# Run from the REPO ROOT, not from this directory. The server resolves its own
# assets off __file__ so it does not care, but ultralytics caches the YOLO-World CLIP
# encoder in a cwd-relative weights/ - start it anywhere else and it re-downloads
# 338 MB into a second cache that nothing will ever use again.
$repo = Split-Path -Parent (Split-Path -Parent $root)
$p = Start-Process -FilePath 'python' `
       -ArgumentList 'examples\mission_server\mission_server.py' `
       -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
       -RedirectStandardInput $stdin `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err"
"started pid=$($p.Id) - detached, survives this shell"
"log: $log"
