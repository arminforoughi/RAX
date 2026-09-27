"""One rig interface, three backends: the X250, the SO-101, and a simulator.

WHY THIS SEAM AND NOT `ServoArm`. `ServoArm` is the *control* seam and it is deliberately
tiny -- four members, no kinematics, no camera model -- because that is all a visual servo
needs. A SERVER needs more than a servo does: it has to draw the arm in 3D, stream a
camera, report joint names to a UI, read a gripper, and know where the racks are. None of
that belongs in `ServoArm` (adding it would make the servo unportable, which is the whole
point of keeping it small), so it lives here.

So the two seams stack rather than compete:

    TubeRig     what a SERVER needs: description, frames, gripper, racks, map
      +-- .servo_arm()  ->  ServoArm      what the APPROACH needs: 4 members

THE SIM BACKEND IS NOT A TOY, and it is not here to fake a result. The X250 is on COM5,
which does not currently enumerate, and the SO-101 owns its camera inside another process.
A UI that can only be developed against absent hardware gets developed against nothing --
which is how the guest page shipped with mojibake emoji and half the camera cropped off.
So the sim implements the same interface, runs the same shared pick loop, and renders a
tube that really does end up in a rack hole. What it is NOT is evidence about the
hardware: every backend reports `simulated` and the UI shows it, because a demo that
cannot be told apart from a real run is worse than no demo.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rax.manipulation.grip import CurrentRise, GripSensor, PositionThreshold
from rax.robots.profiles import ArmProfile, load_profile

logger = logging.getLogger(__name__)

__all__ = ["Tube", "Rack", "TubeRig", "SimRig", "X250Rig", "So101Rig", "make_rig",
           "TUBE_D_M", "TUBE_L_M"]

#: Gripper reading at or above which the X250's jaws are plainly OPEN, so a position
#: reading is not a grasp verdict at all. Between the measured hold band (a 16 mm tube
#: stalls the jaws just over 31.9; pick.py logged 31.98) and every open position on this
#: arm: pick.py verifies a RELEASE at > 45, holds the place-open at 50 and the look-open
#: at 54. 42 sits clear of both sides. See PositionThreshold's docstring for the bug this
#: exists to stop.
X250_OPEN_ABOVE = 42.0

#: A standard lab tube: 16 mm across, 100 mm long. The same figures now in
#: `rax.perception.object_priors`, which is what lets apparent-size ranging judge how far
#: away one is -- before they were added, a tube fell back to the 50.8 mm cube prior and
#: reported itself ~3x too far away.
TUBE_D_M = 0.016
TUBE_L_M = 0.100


@dataclass
class Tube:
    """One tube, wherever it is. Positions are metres in the arm's base frame."""

    id: int
    colour: str
    x: float
    y: float
    z: float = 0.0
    held: bool = False
    #: Name of the rack it is sitting in, and which hole. None when it is on the bench.
    rack: str | None = None
    hole: int | None = None
    #: How the position was obtained: "seen" (this run), "mapped" (a previous look),
    #: or "assumed" (the sim, or a rack slot nobody has looked at). The UI shows this,
    #: because a map that does not distinguish them invites trusting a stale fix.
    source: str = "assumed"

    #: Whether the tube is UPRIGHT. Not cosmetic, and it must not default to True: a rack
    #: hole holds a tube up and nothing else on this bench does, so a loose tube lies down
    #: and rolls. Drawing every tube standing -- which the first version did -- asserts an
    #: orientation nobody measured. The object map already refuses to do that with its
    #: `yaw_known` flag: "drawing that as a definite orientation is a lie".
    standing: bool = False
    #: Long-axis bearing in the base frame, degrees. Meaningless unless `yaw_known`.
    yaw_deg: float = 0.0
    #: True only when something actually measured the orientation.
    yaw_known: bool = False

    def as_json(self) -> dict:
        return {"id": self.id, "colour": self.colour,
                "x": round(self.x, 4), "y": round(self.y, 4), "z": round(self.z, 4),
                "held": self.held, "rack": self.rack, "hole": self.hole,
                "source": self.source, "d": TUBE_D_M, "l": TUBE_L_M,
                "standing": bool(self.standing), "yaw": round(float(self.yaw_deg), 1),
                "yaw_known": bool(self.yaw_known)}


@dataclass
class Rack:
    """A tube rack: where it is, and where its holes are relative to its own centre."""

    name: str
    x: float
    y: float
    yaw_deg: float = 0.0
    #: Hole offsets in the rack's own frame, metres.
    holes: tuple[tuple[float, float], ...] = ()
    colour: str = "#8a93a0"

    def hole_xy(self, i: int) -> tuple[float, float]:
        dx, dy = self.holes[i]
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return (self.x + dx * c - dy * s, self.y + dx * s + dy * c)

    def as_json(self) -> dict:
        return {"name": self.name, "x": round(self.x, 4), "y": round(self.y, 4),
                "yaw": self.yaw_deg, "colour": self.colour,
                "holes": [[round(v, 4) for v in self.hole_xy(i)]
                          for i in range(len(self.holes))]}


def _grid(nx: int, ny: int, pitch: float) -> tuple[tuple[float, float], ...]:
    """A centred nx-by-ny hole grid — the layout of every rack on this bench."""
    return tuple((( i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch)
                 for j in range(ny) for i in range(nx))


class TubeRig(Protocol):
    """Everything the tube server needs from a robot."""

    profile: ArmProfile
    simulated: bool

    def joints_deg(self) -> np.ndarray:
        """Joint angles in DEGREES, for kinematics and the 3D view.

        Degrees even on the X250, whose control path is normalised units: the conversion
        belongs in the backend (see `rax.robots.arms.x250.normalise`), so that everything
        above here speaks one unit and the 3D view needs no per-arm special case.
        """

    def frame(self) -> np.ndarray | None:
        """The latest wrist frame as BGR, or None if there is no camera."""

    def gripper(self) -> float:
        """The gripper's raw reading, in whatever units its sensor uses."""

    def grip_sensor(self) -> GripSensor:
        """How to turn that reading into a held/empty verdict."""

    def tubes(self) -> list[Tube]: ...
    def racks(self) -> list[Rack]: ...


# =================================================================================
# the simulator
# =================================================================================
class SimRig:
    """A rig with no hardware: real kinematics, a scripted arm, and tubes that move.

    The kinematics are NOT faked -- it loads the same URDF and runs the same FK as the
    hardware path, so what the 3D view draws is what the real description produces. What
    is simulated is the arm's response (it goes where it is told, instantly-ish) and the
    camera (there is none).
    """

    simulated = True

    def __init__(self, arm: str = "x250"):
        self.profile = load_profile(arm)
        lo, hi = self.profile.limits()
        self._q = np.array(self.profile.home_deg, dtype=np.float64)
        self._q = np.clip(self._q, lo, hi)
        self._grip = self.profile.gripper.open_pct
        self._holding: int | None = None

        self._racks = [
            Rack("black", 0.20, -0.16, 0.0, _grid(3, 2, 0.022), "#2d323b"),
            Rack("grey", 0.20, 0.16, 0.0, _grid(3, 2, 0.022), "#8a93a0"),
        ]
        # Three tubes LYING on the bench, at plausible reach. Lying, not standing:
        # nothing here holds a tube upright except a rack hole.
        self._tubes = [
            Tube(1, "green", 0.26, 0.02, 0.0, source="assumed",
                 yaw_deg=15.0, yaw_known=True),
            Tube(2, "blue", 0.29, -0.05, 0.0, source="assumed",
                 yaw_deg=-40.0, yaw_known=True),
            Tube(3, "gold", 0.24, 0.07, 0.0, source="assumed",
                 yaw_deg=80.0, yaw_known=True),
        ]

    # ---- the TubeRig interface ---------------------------------------------------
    def joints_deg(self) -> np.ndarray:
        return self._q.copy()

    def frame(self):
        return None

    def gripper(self) -> float:
        return float(self._grip)

    def grip_sensor(self) -> GripSensor:
        g = self.profile.gripper
        # The sim's gripper reports a POSITION, like the X250's, because that is the
        # sensor whose two populations are measured; faking a current draw would be
        # inventing a signal nobody has characterised.
        return PositionThreshold(holding=g.closed_pct + 1.7, empty=g.closed_pct,
                                 open_above=X250_OPEN_ABOVE)

    def tubes(self) -> list[Tube]:
        return list(self._tubes)

    def racks(self) -> list[Rack]:
        return list(self._racks)

    # ---- what the sim additionally offers ---------------------------------------
    def set_joints(self, q_deg) -> None:
        lo, hi = self.profile.limits()
        self._q = np.clip(np.asarray(q_deg, np.float64), lo, hi)

    def set_gripper(self, pct: float) -> None:
        self._grip = float(pct)

    def tube(self, tid: int) -> Tube | None:
        return next((t for t in self._tubes if t.id == tid), None)

    def grasp(self, tid: int) -> bool:
        """Close on a tube. Succeeds only if the tip is actually near it — the sim must
        be able to MISS, or a UI built against it never shows a failure path."""
        t = self.tube(tid)
        if t is None:
            return False
        tip = self.tip_xyz()
        if math.hypot(tip[0] - t.x, tip[1] - t.y) > 0.03:
            self._grip = self.profile.gripper.closed_pct     # closed on air
            return False
        self._holding = tid
        t.held, t.rack, t.hole = True, None, None
        # Upright in the jaws: this gripper closes across the tube and lifts it, which is
        # the one moment the orientation is known without anything having measured it.
        t.standing, t.yaw_known = True, True
        self._grip = self.profile.gripper.closed_pct + 3.5   # clearly holding
        return True

    def release_into(self, rack: str, hole: int) -> bool:
        if self._holding is None:
            return False
        t = self.tube(self._holding)
        r = next((r for r in self._racks if r.name == rack), None)
        if t is None or r is None or not (0 <= hole < len(r.holes)):
            return False
        t.x, t.y = r.hole_xy(hole)
        t.z, t.held, t.rack, t.hole = 0.0, False, rack, hole
        t.standing, t.yaw_known = True, True          # the hole holds it up
        self._holding = None
        self._grip = self.profile.gripper.place_open_pct
        return True

    def free_holes(self, rack: str) -> list[int]:
        r = next((r for r in self._racks if r.name == rack), None)
        if r is None:
            return []
        taken = {t.hole for t in self._tubes if t.rack == rack and t.hole is not None}
        return [i for i in range(len(r.holes)) if i not in taken]

    def holding(self) -> int | None:
        return self._holding

    def tip_xyz(self) -> np.ndarray:
        from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
        if not hasattr(self, "_kin"):
            self._kin = UrdfKinematics(self.profile.urdf_path,
                                       ee_frame=self.profile.ee_frame,
                                       joint_names=list(self.profile.joint_names))
        T = np.asarray(self._kin.forward_kinematics(self._q))
        p = T[:3, 3].copy()
        if self._holding is not None:
            t = self.tube(self._holding)
            if t is not None:
                t.x, t.y, t.z = float(p[0]), float(p[1]), max(0.0, float(p[2]))
        return p


# =================================================================================
# hardware
# =================================================================================
class X250Rig:
    """The real X250: normalised units in, degrees out for the view."""

    simulated = False

    def __init__(self, robot, *, colour: str = "green"):
        from rax.robots.arms.x250.normalise import load_units
        self.profile = load_profile("x250")
        self.robot = robot
        self.colour = colour
        self._units = load_units()
        self._racks = [
            Rack("black", 0.20, -0.16, 0.0, _grid(3, 2, 0.022), "#2d323b"),
            Rack("grey", 0.20, 0.16, 0.0, _grid(3, 2, 0.022), "#8a93a0"),
        ]

    def _pose_norm(self) -> dict:
        o = self.robot.get_observation()
        return {k[:-4]: float(v) for k, v in o.items() if k.endswith(".pos")}

    def joints_deg(self) -> np.ndarray:
        return np.array(self._units.pose_to_deg(self._pose_norm(),
                                                self.profile.joint_names),
                        dtype=np.float64)

    def frame(self):
        import cv2
        o = self.robot.get_observation()
        if "wrist" not in o:
            return None
        return cv2.cvtColor(o["wrist"], cv2.COLOR_RGB2BGR)

    def gripper(self) -> float:
        return float(self._pose_norm().get("gripper", 0.0))

    def grip_sensor(self) -> GripSensor:
        g = self.profile.gripper
        # 31.9 / 30.2, measured over 113 demonstrations with no overlap. The profile
        # carries them so they are stated once.
        return PositionThreshold(holding=31.9, empty=g.closed_pct,
                                 open_above=X250_OPEN_ABOVE)

    def tubes(self) -> list[Tube]:
        """Caps seen in the wrist frame right now.

        DELIBERATELY IMAGE-SPACE-ONLY, reported at the arm's own bearing rather than a
        metric position. This arm has no hand-eye transform and no intrinsics, so a
        base-frame x,y for a detected cap would be fabricated. The 3D view shows these on
        the bearing ray, flagged `source="seen"`, instead of pretending to a fix.
        """
        import sys
        from pathlib import Path
        pick = Path(__file__).resolve().parents[1] / "vla" / "pick"
        if str(pick) not in sys.path:
            sys.path.insert(0, str(pick))
        frame = self.frame()
        if frame is None:
            return []
        try:
            from caps2 import find_caps
        except Exception as e:
            logger.debug("x250: no cap detector (%s)", e)
            return []
        out = []
        for i, c in enumerate(find_caps(frame, restrict_to_mat=True), start=1):
            # No position and no orientation. This arm has no hand-eye transform, so a
            # base-frame pose for a detected cap would be invented; yaw_known stays False.
            out.append(Tube(i, c.colour, 0.0, 0.0, 0.0, source="seen",
                            yaw_known=False))
        return out

    def racks(self) -> list[Rack]:
        return list(self._racks)

    def servo_arm(self, colour: str | None = None):
        import sys
        from pathlib import Path
        pick = Path(__file__).resolve().parents[1] / "vla" / "pick"
        if str(pick) not in sys.path:
            sys.path.insert(0, str(pick))
        from x250_servo import X250ServoArm
        from jaws import exclude_radius, fingers
        # The envelope the servo clamps to is the NORMALISED one — the servo drives the
        # arm, and the arm speaks normalised units.
        from rax.robots.profiles.x250 import ENVELOPE_NORM
        return X250ServoArm(self.robot, colour or self.colour,
                            envelope={k: tuple(v) for k, v in ENVELOPE_NORM.items()},
                            exclude=fingers(), exclude_r=exclude_radius())


class So101Rig:
    """The SO-101 with its OAK-D, driven through whatever observe/command callables the
    host process already has.

    Takes CALLABLES rather than the robot, because on this rig the SO-101's camera and
    bus are owned by the mission server's own threads -- two processes opening the OAK-D
    is how the camera dies. So the tube server either runs inside that process and is
    handed its accessors, or it does not drive this arm at all.
    """

    simulated = False

    def __init__(self, *, joints, frame, gripper, idle_current=0.0, tube_finder=None):
        self.profile = load_profile("so101")
        self._joints, self._frame, self._gripper = joints, frame, gripper
        self._idle = float(idle_current)
        self._finder = tube_finder
        self._racks = [Rack("black", 0.22, -0.14, 0.0, _grid(3, 2, 0.022), "#2d323b")]

    def joints_deg(self) -> np.ndarray:
        return np.asarray(self._joints(), dtype=np.float64)

    def frame(self):
        return self._frame()

    def gripper(self) -> float:
        return float(self._gripper())

    def grip_sensor(self) -> GripSensor:
        # The SO-101's signal is a CURRENT RISE, not a position: a different sensor for
        # the same question, which is exactly what `rax.manipulation.grip` abstracts.
        #
        # It needs no `open_above` guard, and the asymmetry is instructive: current only
        # rises when the motor is WORKING, so an open gripper reads idle and scores as
        # empty all by itself. The position sensor has no such luck — open and holding
        # both read "not shut" — which is why only that one can be asked at the wrong
        # moment and answer confidently wrong.
        return CurrentRise(idle=self._idle,
                           delta=self.profile.gripper.contact_current_delta)

    def tubes(self) -> list[Tube]:
        return list(self._finder() if self._finder else [])

    def racks(self) -> list[Rack]:
        return list(self._racks)


def make_rig(kind: str, **kw) -> TubeRig:
    if kind == "sim":
        return SimRig(kw.pop("arm", "x250"))
    if kind == "x250":
        return X250Rig(**kw)
    if kind == "so101":
        return So101Rig(**kw)
    raise ValueError(f"unknown rig {kind!r}; try sim, x250 or so101")
