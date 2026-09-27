"""X250 follower: the arm driver `pick.py` expects, talking protocol 2.0 directly.

WHY THIS FILE EXISTS. `pick.py` was written against `~/Documents/lab-robot/x250_driver`,
which lives on the machine the demonstrations were recorded on. Nothing of it is on this
rig, so the script could only ever `--dry-run` here. This is the same interface --
`connect`, `get_observation`, `send_action`, `disconnect`, `cameras` -- implemented on
`rax.robots.arms.dynamixel_bus`, so the controller above it needs no changes.

UNITS, WHICH IS THE PART THAT MATTERS. `poses.json` speaks lerobot's NORMALIZED units:
most joints run -100..100 across the joint's calibrated travel, the gripper 0..100. That
is not a property of the pose, it is a property of the pose PLUS a calibration, and a
calibration is per-robot. Measured on this rig against the demonstrated envelope, the
evidence is unambiguous -- read the arm's ticks and the 0..100 reading lands inside the
envelope for five joints of six, while a degrees-from-centre reading lands inside for
three and puts the gripper at -12.5 against an envelope of [30.2, 76.8].

So the poses cannot be commanded without the X250's calibration, and this machine has
only the SO-101's. Rather than convert with a guess and fly the arm somewhere, an
uncalibrated `send_action` REFUSES. `calibrate()` produces the missing file the way
lerobot does: torque off, a human moves each joint through its range, the extremes are
recorded. Reading works either way, so everything up to the first motion can be
exercised now.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where lerobot keeps calibrations, and where this looks for the X250's.
CALIB_DIR = (Path.home() / ".cache" / "huggingface" / "lerobot" /
             "calibration" / "robots" / "x250_follower")

#: Joint order `pick.py` uses, and the servo ids measured on this arm by broadcast ping
#: (ids 2-5 are model 1020 / XM430-W350, ids 6-7 model 1060 / XL430-W250 -- consistent
#: with the big joints being the strong motors and the tool and gripper the light ones).
MOTOR_IDS = {
    "base": 2,
    "shoulder_2": 3,
    "elbow": 4,
    "wrist": 5,
    "tool": 6,
    "gripper": 7,
}

#: The gripper is normalised 0..100; every other joint -100..100. Same split lerobot
#: uses, and the demonstrated envelope agrees: gripper [30.2, 76.8] is one-sided while
#: base [-40.6, 11.7] is not.
RANGE_0_100 = {"gripper"}

TICKS = 4096


class X250Error(RuntimeError):
    pass


@dataclass
class X250FollowerConfig:
    """Constructor arguments, matching the original driver's shape."""

    id: str = "x250_follower"
    port: str = "COM5"
    cameras: dict = field(default_factory=dict)
    #: Ceiling on how far any one command may move a joint from where it is now, in
    #: normalised units. The original passes this as a string; accepted either way.
    max_relative_target: float | str | None = 8.0
    home_on_connect: bool = False
    baudrate: int = 1_000_000


class _Camera:
    """One OpenCV camera, opened from whatever config object was handed in.

    Takes the fields off lerobot's OpenCVCameraConfig by name rather than importing it,
    so this keeps working if that class moves -- it only ever needed an index and a size.
    """

    def __init__(self, cfg):
        import cv2

        self.index = int(getattr(cfg, "index_or_path", getattr(cfg, "index", 0)))
        self.width = int(getattr(cfg, "width", 640) or 640)
        self.height = int(getattr(cfg, "height", 480) or 480)
        self.fps = int(getattr(cfg, "fps", 30) or 30)
        self._cv = cv2
        self.cap = None

    def open(self):
        cv2 = self._cv
        # CAP_DSHOW on Windows: the default backend opens these UVC devices but often
        # negotiates a mode that then fails to deliver frames.
        cap = cv2.VideoCapture(self.index, getattr(cv2, "CAP_DSHOW", 0))
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise X250Error(f"camera index {self.index} would not open")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cap
        return self

    def read_rgb(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise X250Error(f"camera index {self.index} returned no frame")
        return self._cv.cvtColor(frame, self._cv.COLOR_BGR2RGB)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class X250Follower:
    """The X250, as `pick.py` expects to find it."""

    def __init__(self, config: X250FollowerConfig):
        from rax.robots.arms.dynamixel_bus import DynamixelBus

        self.config = config
        self.bus = DynamixelBus(config.port, config.baudrate)
        self.motors = dict(MOTOR_IDS)
        self.calibration = self._load_calibration()
        try:
            self.max_relative_target = (None if config.max_relative_target is None
                                        else float(config.max_relative_target))
        except (TypeError, ValueError):
            self.max_relative_target = 8.0
        self.cameras = {k: _Camera(v) for k, v in (config.cameras or {}).items()}
        self._connected = False

    # ---- calibration -----------------------------------------------------------
    def _calib_path(self) -> Path:
        return CALIB_DIR / f"{self.config.id}.json"

    def _load_calibration(self):
        p = CALIB_DIR / f"{self.config.id}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception as e:
            logger.warning("could not read %s: %s", p, e)
            return None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and all(m in self.calibration for m in self.motors)

    def _to_normalised(self, motor: str, ticks: int) -> float:
        c = self.calibration[motor]
        lo, hi = float(c["range_min"]), float(c["range_max"])
        span = max(hi - lo, 1.0)
        frac = (float(ticks) - lo) / span
        if motor in RANGE_0_100:
            return frac * 100.0
        return frac * 200.0 - 100.0

    def _to_ticks(self, motor: str, value: float) -> int:
        c = self.calibration[motor]
        lo, hi = float(c["range_min"]), float(c["range_max"])
        frac = (float(value) / 100.0) if motor in RANGE_0_100 \
            else (float(value) + 100.0) / 200.0
        return int(round(lo + max(0.0, min(1.0, frac)) * (hi - lo)))

    def calibrate(self, seconds: float = 30.0) -> dict:
        """Record each joint's travel with the torque OFF, the way lerobot does.

        The arm is limp throughout and this writes no goal positions -- the operator
        moves every joint to both of its extremes while the loop watches the ticks. What
        comes out is the range_min/range_max the normalisation needs, in lerobot's own
        format and location, so the calibration is reusable by anything else here.
        """
        if not self._connected:
            raise X250Error("connect() first")
        self.set_torque(False)
        logger.info("CALIBRATE: torque is OFF. Move every joint through its FULL travel "
                    "for the next %.0f seconds.", seconds)
        lo = {m: 1 << 30 for m in self.motors}
        hi = {m: -(1 << 30) for m in self.motors}
        t0 = time.time()
        while time.time() - t0 < seconds:
            raw = self.bus.read_positions(self.motors.values(), wait=0.05)
            for m, i in self.motors.items():
                if i in raw:
                    lo[m] = min(lo[m], raw[i])
                    hi[m] = max(hi[m], raw[i])
            time.sleep(0.02)
        out = {}
        for m, i in self.motors.items():
            if hi[m] - lo[m] < 50:
                raise X250Error(
                    f"'{m}' only moved {max(0, hi[m] - lo[m])} ticks — it needs to be "
                    f"taken through its whole range, or the calibration will map poses "
                    f"onto a travel the joint does not have")
            out[m] = {"id": i, "drive_mode": 0,
                      "homing_offset": int((lo[m] + hi[m]) // 2 - TICKS // 2),
                      "range_min": int(lo[m]), "range_max": int(hi[m])}
        self._calib_path().parent.mkdir(parents=True, exist_ok=True)
        self._calib_path().write_text(json.dumps(out, indent=4))
        self.calibration = out
        logger.info("CALIBRATE: wrote %s", self._calib_path())
        return out

    # ---- lifecycle -------------------------------------------------------------
    def connect(self) -> None:
        self.bus.open()
        found = self.bus.ping()
        missing = [m for m, i in self.motors.items() if i not in found]
        if missing:
            self.bus.close()
            raise X250Error(
                f"these joints did not answer on {self.config.port}: {missing} "
                f"(ids seen: {sorted(found)})")
        for cam in self.cameras.values():
            cam.open()
        self._connected = True
        logger.info("X250 connected on %s: %d joints, %d camera(s), calibration %s",
                    self.config.port, len(self.motors), len(self.cameras),
                    "loaded" if self.is_calibrated else "MISSING (reads only)")
        if self.config.home_on_connect:
            logger.warning("home_on_connect is ignored: homing needs a calibration and "
                           "a known-safe pose, neither of which this driver assumes")

    def disconnect(self) -> None:
        try:
            self.set_torque(False)
        except Exception:
            pass
        for cam in self.cameras.values():
            cam.close()
        self.bus.close()
        self._connected = False

    def set_torque(self, on: bool) -> None:
        for i in self.motors.values():
            try:
                self.bus.write(i, "torque_enable", 1 if on else 0)
            except Exception as e:
                logger.warning("torque %s failed on id %d: %s", "on" if on else "off", i, e)

    # ---- the interface pick.py uses --------------------------------------------
    def get_observation(self) -> dict:
        """``{motor}.pos`` in normalised units, plus one frame per camera.

        Without a calibration the positions are reported as RAW TICKS and the dict
        carries ``calibrated: False``. That keeps the whole pipeline -- cameras,
        detection, the UI -- exercisable on an uncalibrated arm, while making it
        impossible for a caller to mistake ticks for the units the poses are in.
        """
        raw = self.bus.read_positions(self.motors.values())
        out = {}
        for m, i in self.motors.items():
            if i not in raw:
                raise X250Error(f"joint '{m}' (id {i}) did not answer")
            out[f"{m}.pos"] = (self._to_normalised(m, raw[i]) if self.is_calibrated
                               else float(raw[i]))
        out["calibrated"] = self.is_calibrated
        for name, cam in self.cameras.items():
            out[name] = cam.read_rgb()
        return out

    def send_action(self, action: dict) -> dict:
        """Command ``{motor}.pos`` in normalised units. MOVES THE ARM.

        Refuses outright without a calibration, because the numbers in `poses.json` only
        denote a physical arm shape once a calibration says what this joint's travel is.
        Acting on them regardless would not be approximate, it would be arbitrary.
        """
        if not self.is_calibrated:
            raise X250Error(
                "refusing to move: no calibration at "
                f"{self._calib_path()}. The poses in poses.json are lerobot-normalised "
                "against the arm they were recorded on, so without this file they do "
                "not name a position on this one. Run the calibration (torque off, "
                "move every joint through its range) or copy the file from the machine "
                "the demonstrations came from.")
        present = self.bus.read_positions(self.motors.values())
        sent = {}
        for key, value in action.items():
            if not key.endswith(".pos"):
                continue
            m = key[:-4]
            if m not in self.motors:
                continue
            want = float(value)
            if self.max_relative_target is not None:
                now = self._to_normalised(m, present[self.motors[m]])
                want = max(now - self.max_relative_target,
                           min(now + self.max_relative_target, want))
            self.bus.write(self.motors[m], "goal_position", self._to_ticks(m, want))
            sent[key] = want
        return sent
