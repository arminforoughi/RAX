"""Gaze engine — locate the object in 3D, approach the committed point, grasp.

The stereo camera already gives the object's position in the **base frame** (the
cloud centroid). So the approach is *model-based*, not a pixel servo:

  1. **Commit** to the object's 3D point from reliable views (enough cloud points,
     depth in a sane band), and stop ingesting new readings once the camera is
     committed-close — stereo and the mask both fall apart at point-blank, where
     depth floors and the centroid drifts behind the camera.
  2. **Servo** the EE so the camera's aim ray (the bottom-centre pixel, where the
     fingertips sit) lands on that point at ``grasp_range`` along a chosen approach
     axis (top-down / horizontal / angled). placo IK solves all joints jointly, so
     there is no pan/tilt cross-coupling to oscillate. Motion is rate-limited.
  3. When the arm **arrives** at the planned standoff, lunge along the aim ray,
     close the gripper, and lift.

This replaced an earlier 2D joint visual servo that drove shoulder_pan + wrist_flex
from pixel error: it cross-coupled tilt into four joints, hunted in pitch, and never
closed depth (it "saw the cube, looked down, and froze"). Driving to the known 3D
point removes that instability. Mock and real SO-101 share this one path.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from manipulation.arms.arm_interface import ArmInterface, Observation
from manipulation.arms.grasp import GraspConfig, close_with_current, release
from manipulation.arms.kinematics import Kinematics
from manipulation.arms.se3 import look_at_R, look_at_ray_R, make_pose, rate_limited_step
from models.depth.stereo import StereoIntrinsics

if TYPE_CHECKING:
    from perception.depth_cloud import CloudTracker, ObjectTrack

logger = logging.getLogger(__name__)

SEARCH, APPROACH, GRASP, PLACE, DONE, FAILED = (
    "SEARCH", "APPROACH", "GRASP", "PLACE", "DONE", "FAILED"
)


@dataclass
class GazeConfig:
    T_ee_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    world_up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    # Where in the image the object should finally sit (gripper aim point).
    # Positive v_offset = below image centre (OpenCV y-down) — where the fingers are.
    # The aim is RAMPED there: the camera first centres the object (offset 0), then
    # walks it down to this offset. Jumping straight to the bottom aim needs more
    # downward pitch than a 5-DoF arm can reach, so the object gets stuck at the top.
    aim_v_offset_px: float = 120.0
    aim_ramp_px_per_tick: float = 2.5   # how fast to walk the object centre -> bottom
    # While the object is not yet on the aim, prioritise *pointing* the camera at it
    # (frame it) over moving toward it. Once framed, the approach weights take over.
    gaze_ik_position_weight: float = 1.0
    gaze_ik_orientation_weight: float = 3.0

    # Camera-to-object distance to stop approaching (m). Fingertips are ~9-10 cm
    # from the camera at this range.
    grasp_range_m: float = 0.10
    final_advance_m: float = 0.02   # last blind inch after reaching grasp_range
    lift_m: float = 0.06

    place_clearance_m: float = 0.04
    place_hover_m: float = 0.10

    # --- joint gaze (SO-101) ------------------------------------------------
    pan_joint: int = 0       # shoulder_pan — yaw in image
    tilt_joint: int = 1      # shoulder_lift — coarse pitch
    pan_sign: float = 1.0
    tilt_sign: float = 1.0   # +1 pitches up on SO-101 (object at top -> look less down)
    gaze_kp_pan: float = 0.22
    gaze_kp_tilt: float = 0.50
    gaze_max_step_pan_deg: float = 0.7
    gaze_max_step_tilt_deg: float = 1.6
    gaze_tilt_boost_px: float = 80.0   # larger |dv| => stronger pitch correction
    gaze_deadband_px: float = 6.0
    vertical_first_px: float = 45.0    # top-down: fix vertical before panning much
    pan_prior_px: float = 100.0
    track_ema_alpha: float = 0.28
    track_max_jump_px: float = 55.0
    gaze_damp_on_approach: float = 0.5
    center_tol_px: float = 55.0
    center_dv_tol_px: float = 70.0
    reach_du_tol_px: float = 55.0
    reach_dv_tol_px: float = 55.0     # object on bottom-centre aim before any reach

    # --- model-based Cartesian approach (SO-101) ---------------------------
    # The object is known in 3D (cloud centroid, base frame). Instead of servoing
    # pixels on coupled joints (unstable: it hunts in pitch and never closes
    # depth), drive the EE so the camera's aim ray lands on the object at
    # grasp_range, coming in along a chosen approach axis. Pixels are confirmation.
    approach_style: str = "angled"   # "topdown" | "horizontal" | "angled"
    servo_ik_position_weight: float = 4.0
    servo_ik_orientation_weight: float = 0.35
    servo_max_ang_vel_deg_s: float = 40.0
    # Two-phase approach: first go to a pre-grasp standoff *back along the approach
    # axis* (for top-down this is straight above the object — the arm comes up and
    # hovers), then descend in along that axis to grasp_range. Without this the arm
    # cuts straight to the final standoff and dives in from the side near the floor.
    pregrasp_standoff_m: float = 0.20  # camera-object distance while still centring
    in_view_margin_frac: float = 0.12  # object must stay this far inside the frame edges
    # Latch the object's 3D position from *reliable* views (enough points, depth in
    # a sane band) and stop updating it once the camera is too close to trust stereo
    # — then approach the committed point and grasp on arrival. This is what lets the
    # pick survive point-blank, where stereo depth and the mask both fall apart.
    latch_ema_alpha: float = 0.3
    latch_freeze_below_m: float = 0.11   # below this raw depth, stop trusting stereo
    latch_max_jump_m: float = 0.06       # reject readings that jump from the estimate
    latch_commit_margin_m: float = 0.06  # once camera within grasp_range+this, commit
    reach_pos_tol_m: float = 0.015       # camera within this of the standoff => arrived

    approach_kp: float = 0.45
    approach_max_step_m: float = 0.005
    approach_close_m: float = 0.14    # below this, reach slows then stops
    approach_close_step_m: float = 0.002
    reach_ik_position_weight: float = 6.0
    reach_ik_orientation_weight: float = 0.25  # hold aim while inching forward
    cloud_min_pts_depth: int = 80      # ignore sparse/garbage clouds for depth

    # Depth: reject garbage (log showed z jumping to -4 m).
    depth_max_m: float = 0.45
    depth_min_m: float = 0.05
    depth_max_jump_m: float = 0.08       # reject sudden *farther* readings only
    depth_ema_alpha: float = 0.15

    max_joint_step_deg: float = 3.0
    ik_position_weight: float = 3.0
    ik_orientation_weight: float = 0.05
    max_lin_vel_m_s: float = 0.04

    center_hold_frames: int = 2
    grasp_range_tol_m: float = 0.015
    overshoot_range_m: float = 0.095   # raw depth below this => passed object, grasp now

    search_sweep: bool = True
    search_yaw_amp_deg: float = 40.0
    search_pitch_amp_deg: float = 12.0
    search_period_s: float = 4.0
    search_timeout_s: float = 25.0
    settle_tol_m: float = 0.01
    settle_max_ticks: int = 120
    loop_hz: float = 20.0
    grasp: GraspConfig = field(default_factory=GraspConfig)


def _camera_pos(obs: Observation) -> np.ndarray:
    return np.asarray(obs.T_base_cam, dtype=np.float64)[:3, 3]


def _project(p_base: np.ndarray, T_base_cam: np.ndarray, intr: StereoIntrinsics) -> tuple[float, float, float]:
    T = np.asarray(T_base_cam, dtype=np.float64)
    p_cam = T[:3, :3].T @ (np.asarray(p_base) - T[:3, 3])
    z = float(p_cam[2]) if abs(p_cam[2]) > 1e-6 else 1e-6
    u = intr.fx * p_cam[0] / z + intr.cx
    v = intr.fy * p_cam[1] / z + intr.cy
    return float(u), float(v), float(z)


def _ray_point_base(
    u: float, v: float, range_m: float, T_base_cam: np.ndarray, intr: StereoIntrinsics
) -> np.ndarray:
    x_c = (float(u) - intr.cx) / intr.fx
    y_c = (float(v) - intr.cy) / intr.fy
    ray = np.array([x_c, y_c, 1.0], dtype=np.float64)
    ray /= max(float(np.linalg.norm(ray)), 1e-9)
    p_cam = ray * float(range_m)
    T = np.asarray(T_base_cam, dtype=np.float64)
    return T[:3, :3] @ p_cam + T[:3, 3]


class GazeEngine:
    def __init__(
        self,
        arm: ArmInterface,
        kinematics: Kinematics,
        cloud: CloudTracker,
        cfg: GazeConfig | None = None,
        *,
        place_on_label: str | None = None,
        cartesian: bool = False,
    ):
        self.arm = arm
        self.kin = kinematics
        self.cloud = cloud
        self.cfg = cfg or GazeConfig()
        self.place_on_label = place_on_label
        self.cartesian = cartesian  # mock: EE pose; real arm: joint gaze
        self.state = SEARCH
        self._t0 = time.time()
        self.last_obs: Observation | None = None
        self.range_m: float | None = None
        self._depth_filt: float | None = None
        self._grasped_extent_m = 0.05
        self._search_fwd0: np.ndarray | None = None
        self._acquired_once = False
        self._center_streak = 0
        self._pan_sign = float(self.cfg.pan_sign)
        self._tilt_sign = float(self.cfg.tilt_sign)
        self._u_filt: float | None = None
        self._v_filt: float | None = None
        self._last_du: float | None = None
        self._last_dv: float | None = None
        self._last_depth: float | None = None
        self._obj_latch: np.ndarray | None = None   # committed object point (base frame)
        self._reach_err_m: float | None = None       # camera distance to grasp standoff
        self._aim_v_now = 0.0                         # current aim offset (ramps 0 -> target)
        self._log_i = 0

    # -- image helpers ------------------------------------------------------

    def _aim_px(self, intr: StereoIntrinsics) -> tuple[float, float]:
        return float(intr.cx), float(intr.cy) + self._aim_v_now

    def _object_uv(self, focus: ObjectTrack, obs: Observation) -> tuple[float, float]:
        """BBox centre, EMA-smoothed (raw box jitters and causes pan oscillation)."""
        u, v = focus.center_px
        if self._u_filt is None:
            self._u_filt, self._v_filt = float(u), float(v)
        else:
            jump = math.hypot(u - self._u_filt, v - self._v_filt)
            a = self.cfg.track_ema_alpha if jump < self.cfg.track_max_jump_px else 0.08
            self._u_filt = (1.0 - a) * self._u_filt + a * float(u)
            self._v_filt = (1.0 - a) * self._v_filt + a * float(v)
        return self._u_filt, self._v_filt

    def _gaze_uv(self, focus: ObjectTrack, obs: Observation) -> tuple[float, float]:
        return self._object_uv(focus, obs)

    def _pixel_err(self, u: float, v: float, intr: StereoIntrinsics) -> tuple[float, float, float]:
        au, av = self._aim_px(intr)
        du, dv = u - au, v - av
        return du, dv, float(math.hypot(du, dv))

    def _filter_depth(self, z_cam: float) -> float | None:
        """Positive camera-frame Z = range; reject garbage and far jumps."""
        z = abs(float(z_cam))
        if not (self.cfg.depth_min_m <= z <= self.cfg.depth_max_m):
            return self._depth_filt
        if self._depth_filt is not None and z > self._depth_filt + self.cfg.depth_max_jump_m:
            return self._depth_filt
        if self._depth_filt is None:
            self._depth_filt = z
        else:
            a = self.cfg.depth_ema_alpha
            if z < self._depth_filt:
                a = min(0.55, self.cfg.depth_ema_alpha * 2.5)
            self._depth_filt = (1.0 - a) * self._depth_filt + a * z
        self.range_m = self._depth_filt
        return self._depth_filt

    def _raw_depth(self, focus: ObjectTrack) -> float | None:
        z = getattr(focus, "range_cam_m", float("nan"))
        return float(z) if np.isfinite(z) else None

    def _depth_from_focus(self, focus: ObjectTrack, obs: Observation) -> float | None:
        if np.isfinite(getattr(focus, "range_cam_m", float("nan"))):
            return self._filter_depth(focus.range_cam_m)
        if focus.has_cloud and focus.points.shape[0] >= self.cfg.cloud_min_pts_depth:
            _, _, z = _project(focus.centroid, obs.T_base_cam, obs.intrinsics)
            return self._filter_depth(z)
        return self._depth_filt

    def _up(self) -> np.ndarray:
        u = np.asarray(self.cfg.world_up, dtype=np.float64)
        n = float(np.linalg.norm(u))
        return u / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    def _clamp_step(self, q_cur: np.ndarray, q_des: np.ndarray) -> np.ndarray:
        cap = float(self.cfg.max_joint_step_deg)
        q_cur = np.asarray(q_cur, dtype=np.float64).reshape(-1)
        q_des = np.asarray(q_des, dtype=np.float64).reshape(-1)
        if cap <= 0.0:
            return q_des
        n = min(len(q_cur), len(q_des))
        out = q_des.copy()
        out[:n] = q_cur[:n] + np.clip(q_des[:n] - q_cur[:n], -cap, cap)
        return out

    def _gaze_deltas(self, obs: Observation, u: float, v: float) -> tuple[float, float, float, float]:
        intr = obs.intrinsics
        au, av = self._aim_px(intr)
        du, dv = u - au, v - av
        d_pan = d_tilt = 0.0
        vertical_first = abs(dv) > self.cfg.vertical_first_px

        if abs(du) > self.cfg.gaze_deadband_px:
            pan_scale = 1.0
            if vertical_first and abs(dv) > abs(du):
                pan_scale = 0.2  # top-down: get object to bottom before panning
            elif abs(du) > self.cfg.pan_prior_px:
                pan_scale = 0.5
            damp = 1.0
            if self._last_du is not None and abs(du) < abs(self._last_du) - 2.0:
                damp = self.cfg.gaze_damp_on_approach
            d_pan = float(np.clip(
                pan_scale * damp * self._pan_sign * self.cfg.gaze_kp_pan
                * math.degrees(math.atan2(du, intr.fx)),
                -self.cfg.gaze_max_step_pan_deg, self.cfg.gaze_max_step_pan_deg,
            ))

        if abs(dv) > self.cfg.gaze_deadband_px:
            kp_t = self.cfg.gaze_kp_tilt
            cap_t = self.cfg.gaze_max_step_tilt_deg
            if abs(dv) > self.cfg.gaze_tilt_boost_px:
                kp_t *= 1.5
                cap_t = min(2.8, cap_t * 1.6)
            damp = 1.0
            if self._last_dv is not None and abs(dv) < abs(self._last_dv) - 2.0:
                damp = self.cfg.gaze_damp_on_approach
            d_tilt = float(np.clip(
                damp * self._tilt_sign * kp_t * math.degrees(math.atan2(dv, intr.fy)),
                -cap_t, cap_t,
            ))

        self._last_du, self._last_dv = du, dv
        return du, dv, d_pan, d_tilt

    def _apply_gaze(self, q: np.ndarray, obs: Observation, u: float, v: float) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
        du, dv, d_pan, d_tilt = self._gaze_deltas(obs, u, v)
        pj, tj = self.cfg.pan_joint, self.cfg.tilt_joint
        if 0 <= pj < len(q):
            q[pj] += d_pan
        if 0 <= tj < len(q):
            q[tj] += d_tilt
        # Top-down pitch: lift + elbow + wrist share the tilt command.
        if not self.cartesian and abs(d_tilt) > 0.02:
            if len(q) > 2:
                q[2] += 0.45 * d_tilt
            if len(q) > 3:
                q[3] += 0.30 * d_tilt
        return q

    # -- model-based Cartesian approach -------------------------------------

    def _object_in_view(self, u: float, v: float, intr: StereoIntrinsics) -> bool:
        """Is the object pixel safely inside the frame (not about to leave view)?"""
        mx = self.cfg.in_view_margin_frac * intr.width
        my = self.cfg.in_view_margin_frac * intr.height
        return (mx <= u <= intr.width - mx) and (my <= v <= intr.height - my)

    def _aim_ray_cam(self, intr: StereoIntrinsics) -> np.ndarray:
        """Unit ray in camera frame through the gripper aim pixel (bottom-centre)."""
        au, av = self._aim_px(intr)
        r = np.array([(au - intr.cx) / intr.fx, (av - intr.cy) / intr.fy, 1.0])
        return r / max(float(np.linalg.norm(r)), 1e-9)

    def _approach_axis(self, obs: Observation, p_obj: np.ndarray) -> np.ndarray:
        """Unit vector FROM the object TO where the gripper comes in (base frame)."""
        up = self._up()
        p_obj = np.asarray(p_obj, dtype=np.float64).reshape(3)
        style = (self.cfg.approach_style or "angled").lower()
        if style == "topdown":
            return up
        if style == "horizontal":
            d = -p_obj                       # object -> base origin
            d = d - float(d @ up) * up       # drop the vertical component
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                return d / n
        # "angled" (and fallbacks): come in along the current view direction.
        d = _camera_pos(obs) - p_obj
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else up

    def _grasp_pose_parts(
        self, obs: Observation, p_obj: np.ndarray, standoff_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """(R_ee, ee_pos) placing the camera's aim ray on the object at ``standoff_m``."""
        intr = obs.intrinsics
        a = self._approach_axis(obs, p_obj)          # object -> EE side
        aim = self._aim_ray_cam(intr)
        R_cur_ee = self.kin.forward_kinematics(obs.joints_deg)[:3, :3]
        R_ee = look_at_ray_R(R_cur_ee, self.cfg.T_ee_cam, -a, aim)  # aim ray -> object
        R_base_cam = R_ee @ np.asarray(self.cfg.T_ee_cam, dtype=np.float64)[:3, :3]
        cam_pos = np.asarray(p_obj, dtype=np.float64).reshape(3) + a * float(standoff_m)
        p_ee_cam = np.asarray(self.cfg.T_ee_cam, dtype=np.float64)[:3, 3]
        ee_pos = cam_pos - R_base_cam @ p_ee_cam
        return R_ee, ee_pos

    def _servo_approach(self, obs: Observation, focus: ObjectTrack, dt: float) -> None:
        """Cartesian visual servo: drive the EE to the grasp standoff pose.

        The target is the known 3D object point (cloud centroid in base frame), not
        a pixel error, so there is no pan/tilt cross-coupling to oscillate. placo IK
        solves all joints jointly; rate limiting keeps the step safe.
        """
        u, v = self._object_uv(focus, obs)
        du, dv, px_err = self._pixel_err(u, v, obs.intrinsics)
        depth = self._depth_from_focus(focus, obs)
        raw = self._raw_depth(focus)

        if focus.has_cloud:
            p_live = np.asarray(focus.centroid, dtype=np.float64).reshape(3)
        else:
            rng = depth if depth is not None else 0.25
            p_live = _ray_point_base(u, v, rng, obs.T_base_cam, obs.intrinsics)

        # Commit to the object's 3D point from reliable views, then stop trusting new
        # readings once we're committed-close — stereo/mask degrade at point-blank, so
        # we execute the planned approach instead of chasing the garbage that close
        # range produces (depth floors, centroid drifts behind the camera).
        cam = _camera_pos(obs)
        committed = (
            self._obj_latch is not None
            and float(np.linalg.norm(cam - self._obj_latch))
            < self.cfg.grasp_range_m + self.cfg.latch_commit_margin_m
        )
        reliable = (
            focus.has_cloud
            and focus.points.shape[0] >= self.cfg.cloud_min_pts_depth
            and (raw is None or raw >= self.cfg.latch_freeze_below_m)
            and not committed
        )
        if self._obj_latch is None:
            if reliable:
                self._obj_latch = p_live.copy()
        elif reliable and float(np.linalg.norm(p_live - self._obj_latch)) <= self.cfg.latch_max_jump_m:
            a = self.cfg.latch_ema_alpha
            self._obj_latch = (1.0 - a) * self._obj_latch + a * p_live
        p_obj = self._obj_latch if self._obj_latch is not None else p_live

        axis = self._approach_axis(obs, p_obj)
        p_obj3 = np.asarray(p_obj, dtype=np.float64).reshape(3)
        cam_grasp = p_obj3 + axis * self.cfg.grasp_range_m
        framed = abs(du) <= self.cfg.center_tol_px and abs(dv) <= self.cfg.center_dv_tol_px
        in_view = self._object_in_view(u, v, obs.intrinsics)

        # Walk the aim from the image centre down to the bottom-centre, but only while
        # the object is actually sitting on the current aim — so it migrates centre ->
        # bottom smoothly instead of being yanked to the bottom (where a 5-DoF arm
        # can't pitch enough and it sticks at the top).
        target = self.cfg.aim_v_offset_px
        if framed and self._aim_v_now < target:
            self._aim_v_now = min(target, self._aim_v_now + self.cfg.aim_ramp_px_per_tick)
        frac = self._aim_v_now / target if target > 1e-6 else 1.0  # 0 centre -> 1 bottom

        # Standoff shrinks as the object descends to the gripper line: centre the
        # object far back at the pre-grasp standoff, close in to grasp_range as it
        # reaches the bottom. This couples "bring it to the bottom" with the approach.
        standoff = (
            self.cfg.pregrasp_standoff_m
            + (self.cfg.grasp_range_m - self.cfg.pregrasp_standoff_m) * frac
        )
        R_ee, ee_pos = self._grasp_pose_parts(obs, p_obj, standoff)

        # Pointing is prioritised over moving until the object is framed: with the
        # object off-aim we use a high orientation / low position weight so the arm
        # rotates to put the object on the aim (fixes "stuck at the top") rather than
        # driving toward a pose it can't point from. Once framed, the approach weights
        # take over and it moves in. Translation is always suppressed if the object
        # would leave the frame (keep it in view), but is otherwise allowed even when
        # not yet framed — top-down has to relocate above the object to reframe it.
        T_cur = self.kin.forward_kinematics(obs.joints_deg)
        if not in_view:
            ee_pos = T_cur[:3, 3]        # hold position, rotate to recover the object
        if framed:
            pw = self.cfg.servo_ik_position_weight
            ow = self.cfg.servo_ik_orientation_weight
        else:
            pw = self.cfg.gaze_ik_position_weight
            ow = self.cfg.gaze_ik_orientation_weight
        T_des = make_pose(R_ee, ee_pos)
        self._reach_err_m = float(np.linalg.norm(cam - cam_grasp))
        T_step = rate_limited_step(
            T_cur, T_des, dt=dt,
            max_lin_vel_m_s=self.cfg.max_lin_vel_m_s,
            max_ang_vel_deg_s=self.cfg.servo_max_ang_vel_deg_s,
        )
        q = self.kin.inverse_kinematics(
            obs.joints_deg, T_step, position_weight=pw, orientation_weight=ow,
        )
        self.arm.send_joint_targets(self._clamp_step(obs.joints_deg, q))

        self._log_i += 1
        if self._log_i % 20 == 0:
            logger.info(
                "[gaze] servo[%s aim=%.0f%s%s] du=%+.0f dv=%+.0f err=%.0fpx depth=%s raw=%s "
                "reach=%.3fm obj=(%+.2f,%+.2f,%+.2f)",
                self.cfg.approach_style, self._aim_v_now,
                "" if framed else " FRAME",
                "" if in_view else " HOLD",
                du, dv, px_err,
                f"{depth:.3f}m" if depth is not None else "n/a",
                f"{raw:.3f}m" if raw is not None else "n/a",
                self._reach_err_m if self._reach_err_m is not None else -1.0,
                p_obj[0], p_obj[1], p_obj[2],
            )

    def _search_step(self, obs: Observation, dt: float) -> None:
        from scipy.spatial.transform import Rotation

        cam_pos = _camera_pos(obs)
        if self._search_fwd0 is None:
            self._search_fwd0 = np.asarray(obs.T_base_cam, dtype=np.float64)[:3, 2]
        up = self._up()
        fwd = self._search_fwd0
        right = np.cross(up, fwd)
        rn = float(np.linalg.norm(right))
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        t = time.time() - self._t0
        yaw = math.radians(self.cfg.search_yaw_amp_deg) * math.sin(2 * math.pi * t / self.cfg.search_period_s)
        pitch = math.radians(self.cfg.search_pitch_amp_deg) * math.sin(math.pi * t / self.cfg.search_period_s)
        look = Rotation.from_rotvec(up * yaw).apply(fwd)
        look = Rotation.from_rotvec(right * pitch).apply(look)
        if self.cartesian:
            T_cur = self.kin.forward_kinematics(obs.joints_deg)
            R_des = look_at_R(T_cur[:3, :3], self.cfg.T_ee_cam, look)
            p_ee_cam = np.asarray(self.cfg.T_ee_cam, dtype=np.float64)[:3, 3]
            T_des = make_pose(R_des, cam_pos - R_des @ p_ee_cam)
            T_step = rate_limited_step(T_cur, T_des, dt=dt, max_lin_vel_m_s=0.0, max_ang_vel_deg_s=30.0)
            q = self.kin.inverse_kinematics(obs.joints_deg, T_step, orientation_weight=0.2)
            self.arm.send_joint_targets(self._clamp_step(obs.joints_deg, q))
        else:
            q = self._apply_gaze(obs.joints_deg, obs, obs.intrinsics.cx + 80 * math.sin(t), obs.intrinsics.cy)
            self.arm.send_joint_targets(self._clamp_step(obs.joints_deg, q))

    def _reset_to_search(self) -> None:
        self.state = SEARCH
        self._t0 = time.time()
        self._search_fwd0 = None
        self._depth_filt = None
        self.range_m = None
        self._center_streak = 0
        self._u_filt = self._v_filt = None
        self._last_du = self._last_dv = None
        self._last_depth = None
        self._obj_latch = None
        self._reach_err_m = None
        self._aim_v_now = 0.0

    def step(self, dt: float = 0.05) -> str:
        obs = self.arm.get_observation()
        self.last_obs = obs
        self.cloud.update(obs)
        focus = self.cloud.focus_track()

        if self.state == SEARCH:
            if focus is not None and (focus.has_cloud or focus.box):
                logger.info("[gaze] acquired tag=%d -> APPROACH", focus.tag)
                if self.place_on_label is None:
                    self.cloud.lock_focus()
                self._depth_filt = None
                if focus.has_cloud:
                    _, _, z = _project(focus.centroid, obs.T_base_cam, obs.intrinsics)
                    self._filter_depth(z)
                self._acquired_once = True
                self._center_streak = 0
                self._u_filt = self._v_filt = None
                self._last_du = self._last_dv = None
                self._last_depth = None
                self._obj_latch = None
                self._reach_err_m = None
                self._aim_v_now = 0.0
                self.arm.set_gripper(self.cfg.grasp.open_pct)
                self.state = APPROACH
            elif len(self.cloud.tracks) == 0 and self.cfg.search_sweep and not self._acquired_once:
                self._search_step(obs, dt)
            if self.state == SEARCH and not self._acquired_once and time.time() - self._t0 > self.cfg.search_timeout_s:
                logger.warning("[gaze] search timed out")
                self.state = FAILED
            return self.state

        if focus is None:
            self._reset_to_search()
            return self.state

        u, v = self._object_uv(focus, obs)
        du, dv, px_err = self._pixel_err(u, v, obs.intrinsics)
        depth = self._depth_from_focus(focus, obs)
        raw = self._raw_depth(focus)

        if self.state == APPROACH:
            self._servo_approach(obs, focus, dt)

            h_ok = abs(du) <= self.cfg.center_tol_px
            v_ok = abs(dv) <= self.cfg.center_dv_tol_px
            # Primary trigger: the object has been walked to the bottom-centre aim
            # (where the fingers are) AND the arm has closed to the grasp standoff.
            at_bottom = self._aim_v_now >= self.cfg.aim_v_offset_px - 1.0
            arrived = (
                at_bottom
                and self._reach_err_m is not None
                and self._reach_err_m <= self.cfg.reach_pos_tol_m
            )
            # Fallback (real hardware where close-range depth stays valid): depth read.
            at_range = (
                depth is not None
                and depth <= self.cfg.grasp_range_m + self.cfg.grasp_range_tol_m
            )
            overshot = raw is not None and raw <= self.cfg.overshoot_range_m and h_ok

            if (arrived or at_range) and h_ok and v_ok:
                self._center_streak += 1
            else:
                self._center_streak = 0

            if overshot or self._center_streak >= self.cfg.center_hold_frames:
                logger.info(
                    "[gaze] APPROACH -> GRASP (%s reach=%.3fm depth=%s du=%+.0f dv=%+.0f)",
                    "overshot" if overshot else "arrived",
                    self._reach_err_m if self._reach_err_m is not None else -1.0,
                    f"{depth:.3f}m" if depth is not None else "n/a", du, dv,
                )
                self.state = GRASP

            if raw is not None:
                self._last_depth = raw
            elif depth is not None:
                self._last_depth = depth
            return self.state

        if self.state == GRASP:
            self._do_grasp(obs, focus)
            self._depth_filt = None
            self.state = PLACE if self.place_on_label else DONE
            return self.state

        if self.state == PLACE:
            self.state = DONE if self._do_place(focus) else FAILED
            return self.state

        return self.state

    def run(self, max_ticks: int = 1000) -> str:
        dt = 1.0 / max(1.0, self.cfg.loop_hz)
        for _ in range(max_ticks):
            if self.step(dt) in (DONE, FAILED):
                return self.state
            time.sleep(dt)
        return self.state

    def _do_grasp(self, obs: Observation, focus: ObjectTrack) -> None:
        """Short final inch along optical axis, close, lift."""
        logger.info("[gaze] GRASP tag=%d", focus.tag)
        raw = self._raw_depth(focus)
        advance = self.cfg.final_advance_m
        if raw is not None and raw < self.cfg.grasp_range_m + 0.025:
            advance = min(advance, 0.008)

        # Advance along the aim ray (the line the fingertips sit on), not the
        # optical axis — the object was centred on that ray, not the image centre.
        T = np.asarray(obs.T_base_cam, dtype=np.float64)
        tool_dir = T[:3, :3] @ self._aim_ray_cam(obs.intrinsics)
        tn = float(np.linalg.norm(tool_dir))
        tool_dir = tool_dir / tn if tn > 1e-6 else T[:3, 2]
        dt = 1.0 / max(1.0, self.cfg.loop_hz)

        n_steps = max(1, int(round(advance / 0.003)))
        for _ in range(n_steps):
            obs = self.arm.get_observation()
            T_cur = self.kin.forward_kinematics(obs.joints_deg)
            T_des = T_cur.copy()
            T_des[:3, 3] = T_cur[:3, 3] + tool_dir * 0.003
            q = self.kin.inverse_kinematics(
                obs.joints_deg, T_des,
                position_weight=self.cfg.ik_position_weight,
                orientation_weight=self.cfg.ik_orientation_weight,
            )
            self.arm.send_joint_targets(self._clamp_step(obs.joints_deg, q))
            time.sleep(dt)

        close_with_current(self.arm, self.cfg.grasp)
        obs = self.arm.get_observation()
        T_cur = self.kin.forward_kinematics(obs.joints_deg)
        T_des = T_cur.copy()
        T_des[:3, 3] = T_cur[:3, 3] + self._up() * self.cfg.lift_m
        q = self.kin.inverse_kinematics(obs.joints_deg, T_des, position_weight=2.0, orientation_weight=0.05)
        self.arm.send_joint_targets(self._clamp_step(obs.joints_deg, q))

    def _find_support(self, focus: ObjectTrack) -> ObjectTrack | None:
        label = (self.place_on_label or "").lower()
        cands = [
            tr for tr in self.cloud.list_tracks()
            if tr.tag != focus.tag and tr.has_cloud and label in tr.label.lower()
        ]
        return max(cands, key=lambda tr: tr.n_cloud_updates) if cands else None

    def _do_place(self, focus: ObjectTrack) -> bool:
        support = self._find_support(focus)
        if support is None:
            logger.warning("[gaze] PLACE: no support %r", self.place_on_label)
            return False
        logger.info("[gaze] PLACE on tag=%d (%s)", support.tag, support.label)
        up = self._up()
        top = support.top_point(up)
        obs = self.arm.get_observation()
        T_cur = self.kin.forward_kinematics(obs.joints_deg)
        target = top + up * (self._grasped_extent_m * 0.5 + self.cfg.place_clearance_m + 0.12)
        T_des = T_cur.copy()
        T_des[:3, 3] = target
        q = self.kin.inverse_kinematics(obs.joints_deg, T_des, position_weight=2.0)
        self.arm.send_joint_targets(q)
        time.sleep(0.5)
        release(self.arm, self.cfg.grasp)
        return True
