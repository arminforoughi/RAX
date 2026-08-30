"""``UrdfKinematics`` — FK and IK from a URDF, in numpy, with no robot framework.

This file exists so that "any arm with a URDF" is true without qualification. The
alternative was importing a kinematics solver out of a robot-fleet framework, which
means anyone with a UR5 and a URDF installs a fleet manager to multiply four matrices
together. The maths here is textbook and short; the dependency was not.

``placo`` remains supported and is preferred when present — it is a proper QP-based
solver and handles redundancy better than damped least squares. But it ships no wheels
for some platforms (Windows among them), so it cannot be the only option.
:func:`rax.manipulation.arms.kinematics.make_kinematics` picks between them.

Scope, stated honestly: **serial chains** of revolute, continuous, prismatic and fixed
joints. That covers the SO-100/SO-101 family, most hobby arms, and the common
industrial 6-DOF arms. It does not cover branched trees, mimic joints, or closed
loops — a parallel-jaw *gripper* is fine because the gripper is not part of the chain
to the end-effector frame, but a delta robot is not. If you have one of those, install
placo.

Conventions: URDF angles are radians, this class's API is degrees (matching the rest of
the stack and every servo datasheet); ``<origin rpy>`` is extrinsic X-Y-Z.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["UrdfKinematics", "UrdfJoint"]

_REVOLUTE = ("revolute", "continuous")
_ACTUATED = ("revolute", "continuous", "prismatic")


@dataclass
class UrdfJoint:
    """One joint of the chain, as the URDF describes it."""

    name: str
    type: str
    parent: str
    child: str
    T_origin: np.ndarray       # 4x4 parent -> joint frame, at zero displacement
    axis: np.ndarray           # unit axis in the joint frame
    lower: float               # radians, or metres for prismatic
    upper: float

    @property
    def actuated(self) -> bool:
        return self.type in _ACTUATED

    def motion(self, q: float) -> np.ndarray:
        """The 4x4 this joint contributes at displacement ``q`` (rad, or m)."""
        T = np.eye(4)
        if self.type in _REVOLUTE:
            T[:3, :3] = Rotation.from_rotvec(self.axis * float(q)).as_matrix()
        elif self.type == "prismatic":
            T[:3, 3] = self.axis * float(q)
        return T


def _parse_origin(node) -> np.ndarray:
    o = node.find("origin") if node is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()   # URDF rpy is extrinsic XYZ
    T[:3, 3] = xyz
    return T


def _parse_axis(node) -> np.ndarray:
    a = node.find("axis") if node is not None else None
    v = np.array([float(x) for x in (a.get("xyz", "0 0 1") if a is not None else "0 0 1").split()])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


class UrdfKinematics:
    """Forward and inverse kinematics for a serial chain read from a URDF."""

    def __init__(
        self,
        urdf_path: str,
        ee_frame: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
        *,
        max_iters: int = 80,
        damping: float = 1e-4,
        max_step_rad: float = 0.3,
    ):
        path = self._resolve(urdf_path)
        self.urdf_path = str(path)
        self.urdf_dir = str(path.parent)
        self.ee_frame = ee_frame
        self.max_iters = int(max_iters)
        self.damping = float(damping)
        self.max_step = float(max_step_rad)

        by_child = self._read_joints(path)
        self._chain = self._walk_to_root(by_child, ee_frame, path)
        self.base_link = self._chain[0].parent

        actuated = [j.name for j in self._chain if j.actuated]
        self.joint_names = list(joint_names) if joint_names is not None else actuated
        missing = [n for n in self.joint_names if n not in actuated]
        if missing:
            raise ValueError(
                f"{path}: joints {missing} are not on the chain to {ee_frame!r}. "
                f"That chain actuates {actuated}. A joint off the chain cannot move "
                f"the end effector, so asking the IK to use it would silently do nothing.")

        self._by_name = {j.name: j for j in self._chain}
        self._q = dict.fromkeys(actuated, 0.0)   # radians / metres, all chain joints

    # --- construction helpers -------------------------------------------------
    @staticmethod
    def _resolve(urdf_path: str) -> pathlib.Path:
        """Accept a .urdf file, or a directory holding exactly one."""
        p = pathlib.Path(urdf_path).expanduser()
        if p.is_dir():
            found = sorted(p.glob("*.urdf"))
            if not found:
                raise ValueError(f"no .urdf file in {p}")
            # robot.urdf wins if present — several toolchains require that exact name,
            # so a directory containing it plus a copy is a common, benign layout.
            named = [f for f in found if f.name == "robot.urdf"]
            return named[0] if named else found[0]
        if not p.is_file():
            raise ValueError(f"URDF not found: {p}")
        return p

    @staticmethod
    def _read_joints(path: pathlib.Path) -> dict[str, UrdfJoint]:
        """Every joint in the file, keyed by its CHILD link.

        Keyed by child because that is the direction the chain is walked: each link has
        exactly one parent joint in a tree, so child -> joint is a function, while
        parent -> joint is not.
        """
        root = ET.parse(str(path)).getroot()
        out: dict[str, UrdfJoint] = {}
        for j in root.findall("joint"):
            parent, child = j.find("parent"), j.find("child")
            if parent is None or child is None:
                continue
            jtype = (j.get("type") or "fixed").strip().lower()
            lim = j.find("limit")
            lower = float(lim.get("lower", "-inf")) if lim is not None else -np.inf
            upper = float(lim.get("upper", "inf")) if lim is not None else np.inf
            if jtype == "continuous":
                lower, upper = -np.inf, np.inf   # spins freely; no <limit> to read
            out[child.get("link")] = UrdfJoint(
                name=j.get("name") or "", type=jtype,
                parent=parent.get("link"), child=child.get("link"),
                T_origin=_parse_origin(j), axis=_parse_axis(j),
                lower=lower, upper=upper)
        return out

    @staticmethod
    def _walk_to_root(by_child, ee_frame: str, path) -> list[UrdfJoint]:
        """The serial chain from the base link down to ``ee_frame``."""
        chain: list[UrdfJoint] = []
        link = ee_frame
        seen = set()
        while link in by_child:
            if link in seen:
                raise ValueError(f"{path}: link {link!r} is its own ancestor (loop)")
            seen.add(link)
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        if not chain:
            links = sorted(by_child)
            raise ValueError(
                f"{path}: no joint produces frame {ee_frame!r}. Frames in this URDF: "
                f"{links[:12]}{' ...' if len(links) > 12 else ''}")
        return list(reversed(chain))              # base -> tip

    # --- Kinematics protocol --------------------------------------------------
    def forward_kinematics(self, q_deg: np.ndarray) -> np.ndarray:
        self._set(q_deg)
        return self._chain_transforms()[-1]

    def inverse_kinematics(
        self,
        q_current_deg: np.ndarray,
        T_target: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
    ) -> np.ndarray:
        """Damped least squares, seeded from the current pose and clipped to limits.

        Damped rather than a plain pseudo-inverse because the interesting poses are the
        ones near a singularity — reaching straight out, or straight down — and there
        the pseudo-inverse asks for joint velocities that are effectively infinite. The
        damping term trades a little accuracy for a step the arm can actually take.

        Weights let a caller ask for position only (``orientation_weight=0``), which is
        what the approach controller wants: where the gripper points is decided by the
        grasp planner, not by the IK.
        """
        T_goal = np.asarray(T_target, dtype=np.float64)
        current = np.asarray(q_current_deg, dtype=np.float64).reshape(-1)
        # Solve in NATIVE units — radians for a hinge, metres for a slide — because
        # that is what the Jacobian columns are expressed in and what the URDF limits
        # are stored in. Converting the whole vector as if it were angles silently
        # scales every prismatic joint by 57.3 on the next iteration.
        q = self._to_native(current[:len(self.joint_names)])
        w_p = max(0.0, float(position_weight))
        w_o = max(0.0, float(orientation_weight))
        weights = np.concatenate([np.full(3, w_p), np.full(3, w_o)])
        lo, hi = self._limit_arrays()

        for _ in range(self.max_iters):
            self._set(self._to_api(q))
            frames = self._chain_transforms()
            T_now = frames[-1]
            err_p = T_goal[:3, 3] - T_now[:3, 3]
            err_o = Rotation.from_matrix(T_goal[:3, :3] @ T_now[:3, :3].T).as_rotvec()
            if w_p * np.linalg.norm(err_p) < 1e-5 and w_o * np.linalg.norm(err_o) < 1e-4:
                break
            J = self._jacobian(frames, T_now[:3, 3])
            Jw = J * weights[:, None]
            ew = np.concatenate([err_p, err_o]) * weights
            # (J W Jᵀ + λ²I)⁻¹ applied to the weighted error, then mapped back.
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + self.damping * np.eye(6), ew)
            q = np.clip(q + np.clip(dq, -self.max_step, self.max_step), lo, hi)

        out = current.copy()
        out[:len(self.joint_names)] = self._to_api(q)
        return out            # trailing entries (a gripper) pass through untouched

    def get_link_transforms_chain(self, q_deg: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Base-frame pose of every link along the chain, for 3D visualisation."""
        self._set(q_deg)
        frames = self._chain_transforms()
        return [(self.base_link, np.eye(4))] + [
            (j.child, frames[i]) for i, j in enumerate(self._chain)]

    # --- internals ------------------------------------------------------------
    def _set(self, q_deg) -> None:
        q = np.asarray(q_deg, dtype=np.float64).reshape(-1)
        for i, name in enumerate(self.joint_names):
            if i >= len(q):
                break
            joint = self._by_name[name]
            # Prismatic joints are metres in the URDF and metres here; only rotations
            # are degrees. Converting a slide as if it were an angle is a 57x error.
            self._q[name] = float(q[i]) if joint.type == "prismatic" else float(np.deg2rad(q[i]))

    def _chain_transforms(self) -> list[np.ndarray]:
        """Base-frame transform after each joint in the chain."""
        out: list[np.ndarray] = []
        T = np.eye(4)
        for joint in self._chain:
            T = T @ joint.T_origin
            if joint.actuated:
                T = T @ joint.motion(self._q.get(joint.name, 0.0))
            out.append(T.copy())
        return out

    def _jacobian(self, frames: list[np.ndarray], p_ee: np.ndarray) -> np.ndarray:
        """Geometric Jacobian, 6 x n, for the joints this instance controls."""
        cols = []
        index = {j.name: i for i, j in enumerate(self._chain)}
        for name in self.joint_names:
            i = index[name]
            joint, T = self._chain[i], frames[i]
            axis_world = T[:3, :3] @ joint.axis
            if joint.type == "prismatic":
                cols.append(np.concatenate([axis_world, np.zeros(3)]))
            else:
                cols.append(np.concatenate(
                    [np.cross(axis_world, p_ee - T[:3, 3]), axis_world]))
        return np.stack(cols, axis=1)

    def _unit_scale(self) -> np.ndarray:
        """API units -> native units, per joint. Degrees are radians; metres are metres."""
        return np.array(
            [1.0 if self._by_name[n].type == "prismatic" else np.pi / 180.0
             for n in self.joint_names], dtype=np.float64)

    def _to_native(self, q_api) -> np.ndarray:
        return np.asarray(q_api, dtype=np.float64) * self._unit_scale()

    def _to_api(self, q_native) -> np.ndarray:
        return np.asarray(q_native, dtype=np.float64) / self._unit_scale()

    def _limit_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self._by_name[n].lower for n in self.joint_names], dtype=np.float64)
        hi = np.array([self._by_name[n].upper for n in self.joint_names], dtype=np.float64)
        return lo, hi
