"""Depth models (stereo): per-pixel distance for the depth cloud.

Public surface:
    StereoDepth / StereoIntrinsics / ROI   the seam (``stereo.py``)
    make_stereo(backend=...)               build a backend with auto-fallback

Backends, heaviest first: ``raft`` (RAFT-Stereo, default), ``foundation``
(NVlabs FoundationStereo), ``sgbm`` (OpenCV, always available). ``auto`` tries
the learned backends and falls back to SGBM so the pipeline runs anywhere.
"""

from __future__ import annotations

import logging

from models.depth.stereo import ROI, StereoDepth, StereoIntrinsics

logger = logging.getLogger(__name__)

__all__ = ["StereoDepth", "StereoIntrinsics", "ROI", "make_stereo"]


def make_stereo(
    backend: str = "auto",
    *,
    max_disp_px: int = 192,
    raft_repo: str = "",
    raft_ckpt: str = "",
    foundation_repo: str = "",
    foundation_ckpt: str = "",
    device: str = "",
) -> StereoDepth:
    """Construct a stereo backend.

    ``backend`` is one of ``"auto" | "raft" | "foundation" | "sgbm"``. ``auto``
    prefers RAFT-Stereo, then FoundationStereo, then SGBM. A named learned backend
    that fails to load falls back to SGBM with a warning (so a missing checkpoint
    never crashes the run).
    """
    want = backend.lower()

    if want in ("auto", "raft"):
        from models.depth.raft_stereo import RaftStereo

        est = RaftStereo.try_create(
            repo_dir=raft_repo, ckpt_path=raft_ckpt, device=device, max_disp_px=max_disp_px
        )
        if est is not None:
            return est
        if want == "raft":
            logger.warning("[stereo] RAFT-Stereo unavailable; using SGBM")

    if want in ("auto", "foundation"):
        from models.depth.foundation_stereo import FoundationStereoBackend

        est = FoundationStereoBackend.try_create(
            repo_dir=foundation_repo, ckpt_path=foundation_ckpt, device=device, max_disp_px=max_disp_px
        )
        if est is not None:
            return est
        if want == "foundation":
            logger.warning("[stereo] FoundationStereo unavailable; using SGBM")

    from models.depth.sgbm_stereo import SgbmStereo

    return SgbmStereo(max_disp_px=max_disp_px)
