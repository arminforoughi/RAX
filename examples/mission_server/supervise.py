"""Keep the robot server and the room-camera server alive.

WHY A SUPERVISOR. The two die for different reasons and neither can fix itself:
  * stack_mission2.py exits at CONNECT when a servo has latched its overload flag
    (`Failed to read 'Min_Position_Limit' on id_=2 ... Overload error!`). A process
    that has exited cannot retry.
  * camserver.py has been killed externally with nothing in its own log.

WHY PYTHON AND NOT POWERSHELL. The first version of this was supervise.ps1. Run in
the foreground it worked; launched detached with `-File -WindowStyle Hidden` it
stayed alive but silently did nothing - no log, no restarts. Rather than keep
guessing at PS 5.1 plumbing, this owns the children directly: it STARTS them, so
it can just call Popen.poll() and know. No process-table scanning, no window
weirdness, and the health probe still catches a process that is alive but wedged.

    python supervise.py            run it (Ctrl-C stops the supervisor only)
    python supervise.py --stop     stop the supervisor and both servers
    python supervise.py --status   report what is up

Start it detached and it keeps both servers up for as long as it runs:
    Start-Process python -ArgumentList 'supervise.py' -WindowStyle Hidden
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

RAX = os.path.dirname(os.path.abspath(__file__))
CAMSURV = r"C:\Users\labot\Documents\camsurv"
LOG = os.path.join(RAX, "supervise.log")

PERIOD = 10.0          # seconds between checks
BACKOFF_AFTER = 4      # consecutive restarts before standing back
BACKOFF_S = 300.0

SERVICES = {
    "robot": {
        "cmd": [sys.executable, "stack_mission2.py"],
        "cwd": RAX,
        "url": "http://127.0.0.1:8484/status",
        # the robot loads motors, the camera and YOLO before it answers
        "warmup": 150.0,
        # clearing the latched overload FIRST is the whole point: without it a
        # fresh process hits the same unreadable servo and exits again at once
        "pre": [sys.executable, os.path.join(RAX, "clear_motor_overload.py")],
        "log": os.path.join(RAX, "stack_mission2_stdout.log"),
    },
    "camserver": {
        "cmd": [sys.executable, "camserver.py"],
        "cwd": CAMSURV,
        "url": "http://127.0.0.1:5000/",
        "warmup": 25.0,
        "pre": None,
        "log": os.path.join(CAMSURV, "camserver_stdout.log"),
    },
}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def healthy(url, timeout=6.0):
    """Any HTTP answer means the server is alive — camserver's password page is a
    401/302 and that still proves the process is serving."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def kill_strays(name):
    """Kill any instance we did not start, so we never end up with two."""
    match = os.path.basename(SERVICES[name]["cmd"][1])
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return
    for ln in out.splitlines():
        if match in ln:
            pid = ln.rsplit(",", 1)[-1].strip()
            if pid.isdigit() and int(pid) != os.getpid():
                subprocess.run(["taskkill", "/PID", pid, "/F"],
                               capture_output=True, timeout=15)
                log(f"{name}: killed stray pid={pid}")


class Service:
    def __init__(self, name, spec):
        self.name, self.spec = name, spec
        self.proc = None
        self.adopted = False     # running, but started by someone else
        self.started = 0.0
        self.fails = 0
        self.hold_until = 0.0

    def start(self):
        s = self.spec
        if s["pre"]:
            log(f"{self.name}: clearing servo overload before start")
            try:
                subprocess.run(s["pre"], capture_output=True, timeout=90)
            except Exception as e:
                log(f"{self.name}: overload clear failed ({e})")
            time.sleep(2.0)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        out = open(s["log"], "ab", buffering=0)
        self.proc = subprocess.Popen(s["cmd"], cwd=s["cwd"], env=env,
                                     stdout=out, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.started = time.time()
        log(f"{self.name}: started pid={self.proc.pid}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                               capture_output=True, timeout=20)
                log(f"{self.name}: stopped pid={self.proc.pid}")
            except Exception:
                pass

    def check(self):
        now = time.time()
        if now < self.hold_until:
            return
        # An adopted service has no handle to poll, so health is the only signal.
        # Once it fails we start it ourselves and get a handle from then on.
        if self.adopted:
            if healthy(self.spec["url"]):
                self.fails = 0
                return
            log(f"{self.name}: adopted instance stopped answering — taking over")
            kill_strays(self.name)
            self.adopted = False
            self.start()
            self.fails += 1
            return
        # 1. is it even alive? we started it, so this is just a poll
        if self.proc is None or self.proc.poll() is not None:
            code = None if self.proc is None else self.proc.returncode
            log(f"{self.name}: not running (exit={code}) — starting")
            kill_strays(self.name)
            self.start()
            self.fails += 1
        # 2. alive but wedged? only probe once it has had time to boot
        elif now > self.started + self.spec["warmup"]:
            if not healthy(self.spec["url"]):
                log(f"{self.name}: pid={self.proc.pid} alive but not answering "
                    f"{self.spec['url']} — restarting")
                self.stop()
                time.sleep(4.0)          # let COM4 / the USB camera settle
                self.start()
                self.fails += 1
            else:
                if self.fails:
                    log(f"{self.name}: healthy again")
                self.fails = 0
        if self.fails >= BACKOFF_AFTER:
            log(f"{self.name}: failed {self.fails}x — backing off {BACKOFF_S:.0f}s, "
                f"needs a look by hand")
            self.hold_until = now + BACKOFF_S
            self.fails = 0


def main():
    if "--status" in sys.argv:
        for n, s in SERVICES.items():
            print(f"{n:10s} {'UP' if healthy(s['url']) else 'DOWN'}  {s['url']}")
        return
    if "--stop" in sys.argv:
        for n in SERVICES:
            kill_strays(n)
        log("supervisor: stopped both servers")
        return

    log("=" * 46)
    log("supervisor started — watching robot + camserver")
    svcs = [Service(n, s) for n, s in SERVICES.items()]
    # ADOPT a healthy service instead of restarting it. The first version killed
    # it so it could own the child handle - which bounced a perfectly good robot
    # server, and the replacement then hit COM4 still held by the dying one and
    # exited 1. A service we did not start is watched by HTTP alone; the moment it
    # actually fails, we start it ourselves and can poll it from then on.
    for sv in svcs:
        if healthy(sv.spec["url"]):
            log(f"{sv.name}: already up and healthy — adopting (watching by HTTP)")
            sv.adopted = True
            sv.started = time.time()
    try:
        while True:
            for sv in svcs:
                try:
                    sv.check()
                except Exception as e:
                    log(f"{sv.name}: check failed — {type(e).__name__}: {e}")
            time.sleep(PERIOD)
    except KeyboardInterrupt:
        log("supervisor: interrupted (servers left running)")


if __name__ == "__main__":
    main()
