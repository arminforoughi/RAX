# Mission server: the rig will not connect

Read this before touching a cable. In Sep 2026 this cost about a week, and the
whole time the hardware was fine.

## The symptom

`run_server.ps1` starts, the arm connects, and then the server dies:

```
overload cleared - 1:0x00 2:0x00 3:0x00 4:0x00 5:0x00 6:0x00   <- arm is FINE
connecting robot + camera...
WARNING [OAK-D] USB3 connect attempt 1 failed (Failed to find device after booting,
                error message: X_LINK_DEVICE_NOT_FOUND)
... 6 attempts, 3x USB3 then 3x USB2, ~17s each ...
RuntimeError: Failed to find device after booting, error message: X_LINK_DEVICE_NOT_FOUND
```

`X_LINK_DEVICE_NOT_FOUND` reads like "no camera attached". It is not. It means
the camera booted and then vanished before depthai could reopen it.

## Why it is so easy to misdiagnose

**The OAK-D has two USB identities.** This is the single most important fact here.

| id | name | when |
|---|---|---|
| `03E7:2485` | Movidius MyriadX | while idle, unbooted (bootloader) |
| `03E7:F63B` | Luxonis Device | after depthai uploads firmware |

`dai.Device()` boots the camera from `2485` into `F63B`, and `F63B` must
re-enumerate as a fresh USB device before depthai can open it. So a fault that
only affects `F63B` is **invisible at rest**: Device Manager shows a healthy
`Movidius MyriadX`, `getAllAvailableDevices()` returns the device with
`state=X_LINK_UNBOOTED, status=X_LINK_SUCCESS`, and everything looks correct
right up until you try to use it.

It also means **the camera works perfectly on another machine.** Both faults
below live in this PC's registry, keyed to the device. They do not travel with
the hardware. A successful test on a Mac tells you nothing about them - it only
rules out the camera, which was never the problem.

## The two root causes

### 1. The booted identity was disabled

`ConfigFlags=1` (`CONFIGFLAG_DISABLED`) on the `F63B` key. Windows disabled the
device the instant it enumerated. Watching the device tree once per second
during a connect attempt showed the whole cycle:

```
 0s  2485 bootloader   OK/present
 2s  2485 disappears                  <- device resets to boot
 3s  F63B appears      Error/present  <- booted, enumerated, then DISABLED
 8s  F63B still        Error/present
 9s  F63B gone, 2485 back             <- gives up, reverts to bootloader
```

The boot works every time. The device is disabled on arrival.

### 2. usbipd had the devices bound

Both the camera and the arm showed `STATE=Shared` in `usbipd list`, with the
OAK-D on the generic Microsoft `WINUSB` driver rather than a Luxonis one.
Binding hands a device toward WSL and swaps its driver.

Worse, these were **persisted** bindings, which re-apply themselves on every
replug. That is the mechanism that made the fault come back every single time
the camera was plugged in, no matter what was done to it.

## Diagnose

None of this needs admin:

```powershell
# is the booted identity disabled?  1 = CONFIGFLAG_DISABLED, 0 = healthy
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Enum\USB\VID_03E7&PID_F63B\19443010F15FF81200' `
  -Name ConfigFlags).ConfigFlags

# is anything holding the devices?  look for STATE=Shared, and the Persisted list
usbipd list

# does the arm's port exist at all?
[System.IO.Ports.SerialPort]::getportnames()

# does the camera boot outside the server?  isolates depthai from lerobot/Flask
python -c "import depthai as dai; d=dai.Device(); print('OK', d.getUsbSpeed())"
```

To watch the boot cycle live, poll during a connect attempt:

```powershell
Get-PnpDevice | Where-Object { $_.InstanceId -match 'VID_03E7' } |
  Select-Object Status, Present, InstanceId
```

Note the busid in `usbipd list` **changes on every replug** (`5-3` one day, `5-2`
the next). Always read it fresh; a stale busid gives
`error: There is no device with busid`.

## Fix

Both need an **administrator** PowerShell. `Enable-PnpDevice` reports
`Generic failure` here because the `F63B` device is only present for ~6 seconds
during a boot attempt; the registry write is what actually lands.

```powershell
# 1. clear the disable flag
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Enum\USB\VID_03E7&PID_F63B\19443010F15FF81200' `
  -Name ConfigFlags -Value 0 -Type DWord

# 2. release both devices from usbipd (read busids fresh from 'usbipd list')
usbipd unbind --busid <camera-busid>
usbipd unbind --busid <arm-busid>
```

Then **unplug and replug the camera** so it re-enumerates with the flag clear.

`fix_oakd.ps1` (next to this file) does all of this in one pass, including taking ownership
of the `Enum` key if it refuses the write, and logs what it did.

Verify: `usbipd list` should show the rig as `Not shared` and, critically, the
`Persisted` section should no longer list the Movidius or the CH343. If a
persisted entry survives, the fault will return on the next replug.

## Prevention

`run_server.ps1` now preflights all three conditions before launching and
refuses to start with the exact remediation command, rather than dying ~40s
later inside `cam.connect()` with an error that reads like dead hardware.
`-SkipPreflight` bypasses it if a check ever misfires.

## Things that are NOT the cause

Ruled out during the investigation, recorded so nobody re-runs them:

- **Hubs / power.** Both devices sit behind two chained Genesys Logic hubs
  (`05E3:0610`), which is a genuinely plausible cause of exactly this failure -
  but the boot trace disproved it. The device re-enumerates cleanly every time.
- **The cable, the port, the camera.** It works on other machines.
- **depthai version.** 3.7.1, unchanged since Jun 2026, working before and after.
- **Process contention.** No Python process was running during the failures.
- **The server, lerobot, Flask.** Plain `dai.Device()` fails identically.
- **WSL holding the device.** Ubuntu was stopped, and no `/dev/ttyACM*` or
  `/dev/ttyUSB*` existed inside it.

## Related: the arm

The arm (`1A86:55D3`, CH343 -> COM4) fails differently and independently. If
`[System.IO.Ports.SerialPort]::getportnames()` returns **nothing**, the adapter
has not enumerated at all - that is a cable or port fault, and no config change
will help. Note `HKLM:\HARDWARE\DEVICEMAP\SERIALCOMM` does not exist when no
serial device is present; Windows creates it on demand, so a missing key means
"none attached", not "broken".

Do not confuse the two `Realtek RealManage COM1/COM2` devices for usable serial
ports. They are class `MultiFunction`, part of the board's IPMI/out-of-band
management, and they are always present.

## Related: the room camera (camsurv)

`mission_server.py` proxies a second camera at `/stream2` from
`CAMSURV = ("http://127.0.0.1:5000", "camsurv123")` - a separate app at
`C:\Users\labot\Documents\camsurv\camserver.py`, not part of this repo. It is
optional: `/stream2` swallows the failure and the robot side is unaffected. If
the room view is blank, check whether anything is listening on :5000.
