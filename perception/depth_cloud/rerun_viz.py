"""Rerun.io live visualization — lerobot-style camera + sim3d scene.

Each tick logs:
  * annotated camera (bbox + crosshair + phase label),
  * optional depth colormap,
  * URDF arm mesh / link chain (via lerobot ``log_manipulation_sim3d``),
  * object point clouds, target cube proxy, camera optical axis, aim ray.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RerunViz:
    def __init__(self, session: str = "rax_gaze", spawn: bool = True):
        import rerun as rr

        self.rr = rr
        rr.init(session, spawn=spawn)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._frame = 0

    @classmethod
    def try_create(cls, **kw) -> "RerunViz | None":
        try:
            return cls(**kw)
        except Exception as e:
            logger.warning("[rerun] unavailable: %s", e)
            return None

    def _annotated_rgb(
        self, obs, cloud, *, state: str, u_aim: float | None, v_aim: float | None,
        aim_v_offset_px: float = 90.0,
    ) -> np.ndarray:
        import cv2

        img = np.ascontiguousarray(obs.left)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = img.copy()
        intr = obs.intrinsics
        cx, cy = int(round(intr.cx)), int(round(intr.cy))
        cv2.drawMarker(img, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
        # Gripper aim point (bottom-centre of frame, not optical centre).
        aim_v = int(round(intr.cy + aim_v_offset_px))
        cv2.drawMarker(img, (cx, aim_v), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        if u_aim is not None and v_aim is not None:
            cv2.drawMarker(
                img, (int(round(u_aim)), int(round(v_aim))),
                (255, 200, 0), cv2.MARKER_TILTED_CROSS, 14, 2,
            )
        for tr in cloud.list_tracks():
            x1, y1, x2, y2 = (int(v) for v in tr.box)
            is_focus = tr.tag == cloud.focus_tag
            color = (60, 60, 255) if is_focus else (60, 200, 60)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if is_focus else 1)
            if np.isfinite(getattr(tr, "range_cam_m", float("nan"))):
                depth_m = tr.range_cam_m
            elif tr.has_cloud:
                depth_m = tr.centroid[2]
            else:
                depth_m = float("nan")
            label = f"{'>' if is_focus else ''}#{tr.tag} {tr.label} z={depth_m:.2f}m"
            cv2.putText(img, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if state:
            cv2.putText(img, f"state={state}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return img

    def log(
        self,
        obs,
        cloud,
        state: str = "",
        *,
        kin=None,
        depth_m: float | None = None,
        u_aim: float | None = None,
        v_aim: float | None = None,
        aim_v_offset_px: float = 90.0,
    ) -> None:
        rr = self.rr
        self._frame += 1
        rr.set_time("frame", sequence=self._frame)

        vis = self._annotated_rgb(
            obs, cloud, state=state, u_aim=u_aim, v_aim=v_aim,
            aim_v_offset_px=aim_v_offset_px,
        )
        rr.log("camera/image", rr.Image(vis))

        focus = cloud.focus_track()
        if focus is not None:
            x1, y1, x2, y2 = focus.box
            rr.log(
                "camera/image/focus_box",
                rr.Boxes2D(
                    mins=[[x1, y1]],
                    sizes=[[x2 - x1, y2 - y1]],
                    labels=[f"#{focus.tag} {focus.label}"],
                    colors=[[255, 60, 60]],
                ),
            )

        if depth_m is not None:
            try:
                rr.log("scalars/range_m", rr.Scalars(float(depth_m)))
            except Exception:
                pass

        # Point clouds + centroids (base frame).
        p_target = None
        for tr in cloud.list_tracks():
            if not tr.has_cloud:
                continue
            is_focus = tr.tag == cloud.focus_tag
            color = [255, 60, 60] if is_focus else [80, 200, 80]
            rr.log(f"world/cloud_{tr.tag}", rr.Points3D(tr.points, colors=color, radii=0.004))
            rr.log(
                f"world/cloud_{tr.tag}/centroid",
                rr.Points3D([tr.centroid], colors=[255, 255, 0], radii=0.012),
            )
            if is_focus:
                p_target = tr.centroid

        T_base_cam = np.asarray(obs.T_base_cam, dtype=np.float64)
        eye = T_base_cam[:3, 3]
        z_ax = T_base_cam[:3, 2] / max(float(np.linalg.norm(T_base_cam[:3, 2])), 1e-9)
        rr.log("world/camera", rr.Points3D([eye], colors=[80, 160, 255], radii=0.015))
        rr.log(
            "world/camera/optical_axis",
            rr.LineStrips3D(
                strips=[np.stack([eye, eye + 0.25 * z_ax])],
                radii=0.003,
                colors=[[255, 255, 80, 255]],
            ),
        )
        if p_target is not None:
            rr.log(
                "world/aim_ray",
                rr.LineStrips3D(
                    strips=[np.stack([eye, p_target])],
                    radii=0.0025,
                    colors=[[255, 60, 200, 255]],
                ),
            )

        # lerobot sim3d: URDF meshes, object cube proxy, ground grid.
        if kin is not None:
            try:
                from lerobot.utils.manipulation_sim3d import log_manipulation_sim3d

                lk = getattr(kin, "_kin", kin)
                centers, halves, labels, focus_i = None, None, None, None
                if p_target is not None:
                    centers = np.asarray(p_target, dtype=np.float64).reshape(1, 3)
                    halves = np.array([[0.025, 0.025, 0.025]], dtype=np.float64)
                    labels = [focus.label if focus else "target"]
                    focus_i = 0
                log_manipulation_sim3d(
                    frame_sequence=self._frame,
                    kinematics=lk,
                    joint_deg=np.asarray(obs.joints_deg, dtype=np.float64),
                    object_centers_base=centers,
                    object_half_sizes_base=halves,
                    object_labels=labels,
                    focus_object_index=focus_i,
                    plan_summary=f"state={state}" + (f" d={depth_m:.2f}m" if depth_m else ""),
                    ground_plane_z_m=0.0,
                )
            except Exception as e:
                logger.debug("[rerun] sim3d: %s", e)
                self._log_arm_chain_fallback(kin, obs)

        try:
            rr.flush()
        except Exception:
            pass

    def _log_arm_chain_fallback(self, kin, obs) -> None:
        """Line-strip arm skeleton when URDF meshes are unavailable."""
        rr = self.rr
        try:
            lk = getattr(kin, "_kin", kin)
            chain = lk.get_link_transforms_chain(obs.joints_deg)
            if not chain:
                return
            pts = np.stack([T[:3, 3] for _, T in chain], axis=0)
            rr.log(
                "sim3d/robot/arm_chain",
                rr.LineStrips3D(
                    strips=[pts], radii=0.004, colors=[[100, 140, 255, 255]],
                ),
            )
            T_ee = kin.forward_kinematics(obs.joints_deg)
            rr.log(
                "sim3d/robot/ee",
                rr.Points3D([T_ee[:3, 3]], colors=[255, 120, 60], radii=0.014),
            )
        except Exception as e:
            logger.debug("[rerun] arm chain fallback: %s", e)
