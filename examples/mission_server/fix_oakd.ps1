# Fix the OAK-D on LABOT. Requires admin. Writes a log the agent reads back.
$log = "$PSScriptRoot\fix_oakd.log"
function W($m) { $m | Tee-Object -FilePath $log -Append }
Set-Content $log "=== fix_oakd $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

W "admin: $(([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator))"

# 1. Release both devices from usbipd so nothing else claims them.
W "`n--- usbipd unbind ---"
foreach ($vp in '03e7:2485','1a86:55d3') {
  $line = usbipd list | Select-String $vp
  if (-not $line) { W "  $vp : not connected"; continue }
  $busid = $line.ToString().Trim().Split(' ')[0]
  $out = usbipd unbind --busid $busid 2>&1
  W "  $vp busid=$busid -> exit=$LASTEXITCODE $out"
}

# 2. Re-enable the BOOTED OAK-D identity (PID_F63B). It carries ConfigFlags=1
#    (CONFIGFLAG_DISABLED), so Windows disables it the moment depthai boots the
#    camera into it -- which is what surfaces as X_LINK_DEVICE_NOT_FOUND.
W "`n--- enable Luxonis Device (PID_F63B) ---"
$id = 'USB\VID_03E7&PID_F63B\19443010F15FF81200'
try { Enable-PnpDevice -InstanceId $id -Confirm:$false -ErrorAction Stop; W "  Enable-PnpDevice: OK" }
catch { W "  Enable-PnpDevice failed: $($_.Exception.Message)" }

# Fallback: clear the flag directly, taking ownership if the Enum key refuses.
$rk = 'HKLM:\SYSTEM\CurrentControlSet\Enum\USB\VID_03E7&PID_F63B\19443010F15FF81200'
$cf = (Get-ItemProperty $rk -Name ConfigFlags -ErrorAction SilentlyContinue).ConfigFlags
W "  ConfigFlags after enable: $cf"
if ($cf -ne 0) {
  try { Set-ItemProperty $rk -Name ConfigFlags -Value 0 -Type DWord -ErrorAction Stop; W "  registry write: OK" }
  catch {
    W "  registry write blocked, taking ownership..."
    $sub = 'SYSTEM\CurrentControlSet\Enum\USB\VID_03E7&PID_F63B\19443010F15FF81200'
    try {
      $k = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey($sub, [Microsoft.Win32.RegistryKeyPermissionCheck]::ReadWriteSubTree, [System.Security.AccessControl.RegistryRights]::TakeOwnership)
      $acl = $k.GetAccessControl([System.Security.AccessControl.AccessControlSections]::None)
      $acl.SetOwner([System.Security.Principal.NTAccount]"Administrators"); $k.SetAccessControl($acl)
      $k2 = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey($sub, [Microsoft.Win32.RegistryKeyPermissionCheck]::ReadWriteSubTree, [System.Security.AccessControl.RegistryRights]::ChangePermissions)
      $acl2 = $k2.GetAccessControl()
      $acl2.SetAccessRule((New-Object System.Security.AccessControl.RegistryAccessRule("Administrators","FullControl","ContainerInherit","None","Allow")))
      $k2.SetAccessControl($acl2)
      Set-ItemProperty $rk -Name ConfigFlags -Value 0 -Type DWord -ErrorAction Stop
      W "  registry write after ownership: OK"
    } catch { W "  ownership route failed: $($_.Exception.Message)" }
  }
}

W "`n--- final state ---"
foreach ($p in 'VID_03E7&PID_F63B\19443010F15FF81200','VID_03E7&PID_2485\03E72485') {
  $v = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Enum\USB\$p" -Name ConfigFlags -ErrorAction SilentlyContinue).ConfigFlags
  W ("  {0,-42} ConfigFlags={1}" -f $p, $v)
}
W (usbipd list | Out-String)
W "DONE"
