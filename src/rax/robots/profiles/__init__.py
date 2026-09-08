"""``ArmProfile`` — an arm described as data, so a new one plugs in without new code.

The pick stack works, but it is welded to one robot: the SO-101's topology is written
into the algorithms as literal joint indices (``q[0]`` is pan, ``q[1]+q[2]+q[3]`` is the
gripper pitch, ``q[4]`` is roll), its joint limits are transcribed by hand, its serial
port is repeated in three files, and its hand-eye transform is a module global.

A profile moves all of that out of the algorithms and into one declarative object. The
IK, the camera geometry and the approach controller then read *which* joint does what
instead of assuming, and pointing the stack at a different arm becomes a matter of
writing one of these — URDF, extrinsics, and a topology description.

Two topologies are supported, selected by :attr:`ArmProfile.ik`:

``"pitch_hold"``
    Arms like the SO-101 whose pitch joints share a parallel axis, so the gripper's
    world pitch is exactly their sum. The last joint of the chain can then be slaved
    algebraically instead of solved, which is what makes the pitch hold exactly rather
    than drift. Needs :attr:`pan_joint` and a :attr:`pitch_chain` of 2+ joints.

``"pose"``
    Arms with a real 6-DOF wrist, solved as a full pose through the ``Kinematics``
    protocol.

Usage::

    from rax.robots.profiles import load_profile
    profile = load_profile("so101")        # limits already read from its URDF
"""

from __future__ import annotations

import importlib
import pathlib
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from rax.robots.profiles.urdf_limits import joint_limits_deg, read_urdf_limits

__all__ = [
    "ArmProfile", "GripperProfile", "CameraProfile", "BusProfile", "CameraKind",
    "load_profile", "available_profiles", "REPO_ROOT", "PACKAGE_ROOT",
]

def _repo_root() -> pathlib.Path:
    """The directory repo-relative asset paths (URDFs, meshes) resolve against.

    Walks up looking for a project marker rather than counting directory levels: the
    count changed once already when the packages moved under ``src/``, and a silently
    wrong root turns every URDF path into a confusing "file not found" far from here.
    Installed (non-editable) there is no repo above the package, so the package
    directory is the honest answer and asset paths must then be absolute.
    """
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return here.parents[2]


REPO_ROOT = _repo_root()

# Assets that ship *with* the library (the SO-101 URDF) resolve against the package,
# not the repo. Those two are the same directory in a clone and different ones after
# ``pip install``, which is exactly the case that used to break: the profile resolved
# fine for the author and could not find its own URDF for anybody who installed it.
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]

CameraMount = Literal["eye_in_hand", "fixed"]
CameraKind = Literal["stereo", "rgbd", "mono"]
IkKind = Literal["pitch_hold", "pose"]

# A seed entry of None means "carry this joint over from the caller's seed"; a number
# overrides it. Lets one seed table describe arms with different joint counts.
SeedValue = float | None


@dataclass(frozen=True)
class GripperProfile:
    """How this arm's gripper opens, closes, and knows it has touched something."""

    joint_name: str = "gripper"
    open_pct: float = 95.0
    closed_pct: float = 2.0
    place_open_pct: float = 60.0
    close_step_pct: float = 5.0
    close_delay_s: float = 0.05
    # Current rise over idle, in raw servo counts, that counts as "fingers on the
    # object". 1.8 was tried and let the cube slip during transit; 8 holds.
    contact_current_delta: float = 8.0
    # Extra squeeze applied once contact is detected.
    squeeze_extra_pct: float = 14.0
    # Opening to relax to after closing on air, rather than staying stalled shut
    # (a stalled close is what trips the servo's overload protection).
    relax_on_miss_pct: float = 40.0
    # Distance from the EE frame origin to the actual grasp centre. For the SO-101 the
    # URDF's gripper_frame_link IS the fingertip, so this is nearly zero — do not
    # assume a large offset without measuring it against the jaw mesh.
    tip_offset_m: float = 0.0
    grasp_roll_deg: float = 0.0
    # Display only: degrees added to the roll joint before drawing the URDF, when the
    # URDF's roll zero is rotated from the servo's zero.
    render_offset_deg: float = 0.0
    # The fingertip's MEASURED pixel in the wrist camera (eye-in-hand only). The
    # hand-eye transform predicts this same pixel independently, so the gap between
    # the two is a live read-out of hand-eye error. None if never measured.
    hand_uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class CameraProfile:
    """Where the camera is, what it sees with, and how its pose is obtained."""

    # What the sensor delivers, which decides how depth is obtained:
    #   "stereo" -> rectified pair, a matcher runs      (OAK-D, ZED, any baseline rig)
    #   "rgbd"   -> metric depth off the device          (RealSense, Femto, Kinect)
    #   "mono"   -> colour only; range comes from the table plane and class sizes
    # Mono is a supported rig, not a degraded one — see perception/camera_interface.py.
    kind: CameraKind = "stereo"
    mount: CameraMount = "eye_in_hand"
    # "x,y,z,rx,ry,rz" with rotation as a rotation vector in radians. For
    # mount="eye_in_hand" this is T_ee_cam; for "fixed" it is T_base_cam.
    extrinsics: str = "0,0,0,0,0,0"
    # Re-fitted extrinsics written by the hand-eye calibration, overriding the above
    # at startup when present. Relative to the repo root.
    calibration_file: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    use_depth: bool = False
    # Used only when the camera cannot report its own intrinsics.
    intrinsics_fallback: tuple[float, float, float, float] = (517.0, 517.0, 329.5, 231.4)
    # Nominal distance from the fingertips back to the camera. Seeds and bounds the
    # hand-eye fit; not used as a correction on its own.
    cam_tip_m: float = 0.10

    @property
    def eye_in_hand(self) -> bool:
        return self.mount == "eye_in_hand"


@dataclass(frozen=True)
class BusProfile:
    """Raw servo-bus access, used to clear a latched overload before the SDK's
    handshake reads hit the error and kill the connect."""

    protocol: str = "feetech"
    baud: int = 1_000_000
    motor_ids: tuple[int, ...] = ()
    torque_register: int = 40
    status_register: int = 56


@dataclass(frozen=True)
class ArmProfile:
    """Everything the arm-agnostic stack needs to know about one arm."""

    name: str
    urdf: str
    ee_frame: str
    joint_names: tuple[str, ...]
    port: str = ""
    # Directory holding the visual meshes for the 3D viewer. Empty = the URDF's own
    # directory. It is separate from `urdf` because mesh loaders impose their own
    # conventions on that directory (lerobot's wants a file named exactly
    # "robot.urdf"), and an arm should be able to say where its meshes are rather than
    # having the layout assumed for it.
    mesh_dir: str = ""

    # --- topology: which joint does what -------------------------------------------
    ik: IkKind = "pose"
    pan_joint: int | None = None
    # Parallel-axis joints whose angles SUM to the gripper's world pitch. The last one
    # is slaved algebraically; the rest help position the tip.
    pitch_chain: tuple[int, ...] = ()
    roll_joint: int | None = None
    # Re-seed poses tried when the IK lands in a bad branch. The SO-101 has a genuine
    # elbow-flip dead band where no single seed converges.
    ik_seeds: tuple[tuple[SeedValue, ...], ...] = ()

    # --- named poses ----------------------------------------------------------------
    home_deg: tuple[float, ...] = ()
    view_deg: tuple[float, ...] = ()
    # Wrist pose (all joints after the pan) held while surveying from a fixed vantage.
    survey_tilt_deg: tuple[float, ...] | None = None

    # --- motion / workspace ---------------------------------------------------------
    joint_rate_max_dps: float = 25.0
    # Per-joint ceilings for smooth transit moves. Per-joint because the joints do not
    # carry equal inertia: the base swings the whole arm, the wrist swings almost
    # nothing. Empty tuples fall back to joint_rate_max_dps for every joint.
    goto_vmax_dps: tuple[float, ...] = ()
    goto_amax_dps2: tuple[float, ...] = ()
    goto_dt_s: float = 0.02
    table_z_m: float = 0.0
    reach_min_m: float = 0.08
    reach_max_m: float = 0.55

    #: How far the FINGERTIP can actually be driven, metres. Distinct from
    #: ``reach_max_m``, which is a plausibility bound on a localization ("a cube at
    #: 92 cm is a broken solve"). This one is a kinematic fact: past it there is no
    #: IK solution, so a target beyond it can be seen and mapped but never picked.
    #:
    #: Derive it, do not guess it -- ``workspace.analyze_workspace(ik, profile)`` then
    #: ``ws.max_reach(pitch_deg, z_m)``. It is strongly pitch-dependent, and the value
    #: here is the best case (a flat wrist); a steep grasp reaches far less.
    reach_grasp_max_m: float = 0.47

    gripper: GripperProfile = field(default_factory=GripperProfile)
    camera: CameraProfile = field(default_factory=CameraProfile)
    bus: BusProfile | None = None

    # Filled by resolve() from the URDF; pass explicitly only to override.
    limits_deg: tuple[np.ndarray, np.ndarray] | None = None

    # --- derived --------------------------------------------------------------------
    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def urdf_path(self) -> str:
        """Absolute URDF path, resolving a repo-relative one against the repo root.

        Always absolute: the server is started from various working directories (and
        as a service), so a path that happens to resolve against the current one is a
        latent failure rather than a convenience.
        """
        p = pathlib.Path(self.urdf).expanduser()
        if p.is_absolute():
            return str(p)
        for root in (PACKAGE_ROOT, REPO_ROOT):
            rooted = root / p
            if rooted.exists():
                return str(rooted)
        return str(p.resolve())

    @property
    def mesh_path(self) -> str:
        """Absolute directory to load visual meshes from; the URDF's own if unset."""
        if not self.mesh_dir:
            return str(pathlib.Path(self.urdf_path).parent)
        p = pathlib.Path(self.mesh_dir).expanduser()
        if p.is_absolute():
            return str(p)
        for root in (PACKAGE_ROOT, REPO_ROOT):
            rooted = root / p
            if rooted.exists():
                return str(rooted)
        return str(p.resolve())

    @property
    def slaved_joint(self) -> int | None:
        """The pitch-chain joint solved algebraically rather than numerically."""
        return self.pitch_chain[-1] if self.pitch_chain else None

    @property
    def positioning_joints(self) -> tuple[int, ...]:
        """Joints the position IK actually drives: the pan plus every pitch-chain
        joint except the slaved one."""
        pan = () if self.pan_joint is None else (self.pan_joint,)
        return pan + tuple(self.pitch_chain[:-1])

    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        if self.limits_deg is None:
            raise RuntimeError(f"profile {self.name!r} is unresolved; call .resolve()")
        return self.limits_deg

    def index(self, joint_name: str) -> int:
        return self.joint_names.index(joint_name)

    # --- lifecycle ------------------------------------------------------------------
    def resolve(self) -> ArmProfile:
        """Return a copy with joint limits read from the URDF, then validated.

        Limits already set on the profile are kept — an arm whose URDF lies about its
        limits (or omits them) can override, but has to say so explicitly.
        """
        prof = self
        if prof.limits_deg is None:
            lo, hi = joint_limits_deg(prof.urdf_path, list(prof.joint_names))
            prof = replace(prof, limits_deg=(lo, hi))
        # An arm that has not been characterized per joint moves every joint at its
        # single overall rate cap — slow and even, rather than guessed and uneven.
        n = prof.n_joints
        if not prof.goto_vmax_dps:
            prof = replace(prof, goto_vmax_dps=(prof.joint_rate_max_dps,) * n)
        if not prof.goto_amax_dps2:
            prof = replace(prof, goto_amax_dps2=(prof.joint_rate_max_dps * 2.0,) * n)
        prof.validate()
        return prof

    def validate(self) -> None:
        n = self.n_joints
        if n == 0:
            raise ValueError(f"profile {self.name!r} lists no joints")

        def _check(idx, what):
            if idx is not None and not 0 <= idx < n:
                raise ValueError(f"{self.name}: {what} index {idx} outside 0..{n - 1}")

        _check(self.pan_joint, "pan_joint")
        _check(self.roll_joint, "roll_joint")
        for i in self.pitch_chain:
            _check(i, "pitch_chain")
        if len(set(self.pitch_chain)) != len(self.pitch_chain):
            raise ValueError(f"{self.name}: pitch_chain repeats a joint: {self.pitch_chain}")

        if self.ik == "pitch_hold":
            if len(self.pitch_chain) < 2:
                raise ValueError(
                    f"{self.name}: ik='pitch_hold' needs a pitch_chain of 2+ parallel "
                    f"joints (the last is slaved to hold the pitch); got {self.pitch_chain}"
                )
            if self.pan_joint is None:
                raise ValueError(f"{self.name}: ik='pitch_hold' needs a pan_joint")
            if self.pan_joint in self.pitch_chain:
                raise ValueError(f"{self.name}: pan_joint is also in pitch_chain")

        for pose, what in ((self.home_deg, "home_deg"), (self.view_deg, "view_deg")):
            if pose and len(pose) != n:
                raise ValueError(f"{self.name}: {what} has {len(pose)} values, need {n}")
        for seed in self.ik_seeds:
            if len(seed) != n:
                raise ValueError(
                    f"{self.name}: ik_seed {seed} has {len(seed)} values, need {n}"
                )
        for lim, what in ((self.goto_vmax_dps, "goto_vmax_dps"),
                          (self.goto_amax_dps2, "goto_amax_dps2")):
            if lim and len(lim) != n:
                raise ValueError(f"{self.name}: {what} has {len(lim)} values, need {n}")
        if self.limits_deg is not None:
            lo, hi = self.limits_deg
            if len(lo) != n or len(hi) != n:
                raise ValueError(f"{self.name}: limits do not cover {n} joints")

    def urdf_limits_report(self) -> str:
        """Human-readable comparison of the URDF's limits against this profile's —
        for checking a hand-transcribed table against its source."""
        urdf = read_urdf_limits(self.urdf_path)
        lo, hi = self.limits()
        rows = [f"{'joint':<16} {'urdf lo':>9} {'urdf hi':>9} {'prof lo':>9} {'prof hi':>9}"]
        for i, n in enumerate(self.joint_names):
            u = urdf.get(n)
            ul, uh = (f"{u[0]:+9.2f}", f"{u[1]:+9.2f}") if u else ("        -", "        -")
            rows.append(f"{n:<16} {ul} {uh} {lo[i]:+9.2f} {hi[i]:+9.2f}")
        return "\n".join(rows)


# --- registry -----------------------------------------------------------------------
# Following the repo's existing backend convention (make_stereo / make_detector):
# import lazily inside the factory so a profile whose deps are missing costs nothing
# until it is actually asked for.
_PROFILES = {
    "so101": ("rax.robots.profiles.so101", "PROFILE"),
    "mock": ("rax.robots.profiles.mock", "PROFILE"),
    "head_mono": ("rax.robots.profiles.head_mono", "PROFILE"),
    "wrist_mono": ("rax.robots.profiles.wrist_mono", "PROFILE"),
}


def available_profiles() -> list[str]:
    return sorted(_PROFILES)


def load_profile(name: str) -> ArmProfile:
    """Look up a named profile and return it resolved (limits read from its URDF)."""
    try:
        module_name, attr = _PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown arm profile {name!r}; available: {available_profiles()}"
        ) from None
    profile = getattr(importlib.import_module(module_name), attr)
    return profile.resolve()
