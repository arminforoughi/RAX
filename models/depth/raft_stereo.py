"""RAFT-Stereo backend (default learned stereo).

Loads the princeton-vl/RAFT-Stereo model from a local clone + checkpoint, mirroring
the upstream ``demo.py`` inference path. Enabled via ``RAFT_STEREO_REPO`` /
``RAFT_STEREO_CKPT`` (or explicit args); :func:`try_create` returns ``None`` if the
repo, checkpoint, or torch are missing, so the factory falls back gracefully.

Like the other backends it honours ``roi`` by cropping to a disparity-safe window
(see :func:`models.depth.stereo.disparity_window`), so per-object updates stay cheap.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from models.depth.stereo import ROI, StereoIntrinsics, disparity_window

logger = logging.getLogger(__name__)

# Middlebury-style defaults from the RAFT-Stereo argparse; overridable via try_create.
_DEFAULT_ARGS = dict(
    hidden_dims=[128, 128, 128],
    corr_implementation="reg",
    shared_backbone=False,
    corr_levels=4,
    corr_radius=4,
    n_downsample=2,
    context_norm="batch",
    slow_fast_gru=False,
    n_gru_layers=3,
    mixed_precision=False,
)


class RaftStereo:
    name = "raft"

    def __init__(self, model, torch, padder_cls, device, *, iters: int, max_disp_px: int):
        self._model = model
        self._torch = torch
        self._padder_cls = padder_cls
        self._device = device
        self._iters = int(iters)
        self.max_disp = int(max_disp_px)

    @classmethod
    def try_create(
        cls,
        *,
        repo_dir: str = "",
        ckpt_path: str = "",
        iters: int = 32,
        device: str = "",
        max_disp_px: int = 256,
        model_args: dict | None = None,
    ) -> "RaftStereo | None":
        repo = str(repo_dir or os.environ.get("RAFT_STEREO_REPO", "")).strip()
        ckpt = str(ckpt_path or os.environ.get("RAFT_STEREO_CKPT", "")).strip()
        if not repo or not ckpt:
            logger.warning("[raft-stereo] set RAFT_STEREO_REPO / RAFT_STEREO_CKPT to enable")
            return None
        try:
            import torch
        except Exception as e:
            logger.warning("[raft-stereo] torch unavailable: %s", e)
            return None
        try:
            repo_p = Path(repo).expanduser().resolve()
            ckpt_p = Path(ckpt).expanduser().resolve()
            if not repo_p.is_dir() or not ckpt_p.is_file():
                raise FileNotFoundError(f"repo {repo_p} / ckpt {ckpt_p}")
            if str(repo_p) not in sys.path:
                sys.path.insert(0, str(repo_p))
            from core.raft_stereo import RAFTStereo  # type: ignore[import-not-found]
            from core.utils.utils import InputPadder  # type: ignore[import-not-found]

            args = SimpleNamespace(**{**_DEFAULT_ARGS, **(model_args or {})})
            dev = torch.device(device) if device else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model = RAFTStereo(args)
            state = torch.load(str(ckpt_p), map_location=dev)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.to(dev).eval()
            logger.info("[raft-stereo] loaded %s on %s", ckpt_p.name, dev)
            return cls(model, torch, InputPadder, dev, iters=iters, max_disp_px=max_disp_px)
        except Exception as e:
            logger.warning("[raft-stereo] failed to load: %s", e)
            return None

    def _to_tensor(self, img: np.ndarray):
        a = np.asarray(img)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.shape[2] == 4:
            a = a[:, :, :3]
        t = self._torch.as_tensor(a).float().permute(2, 0, 1)[None]  # 1x3xHxW
        return t.to(self._device)

    def depth_meters(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        intr: StereoIntrinsics,
        roi: ROI | None = None,
    ) -> np.ndarray:
        torch = self._torch
        left = np.asarray(left)
        right = np.asarray(right)
        h, w = left.shape[:2]
        depth = np.full((h, w), np.nan, dtype=np.float32)

        if roi is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = disparity_window(roi, self.max_disp, w, h)

        img1 = self._to_tensor(left[y0:y1, x0:x1])
        img2 = self._to_tensor(right[y0:y1, x0:x1])
        padder = self._padder_cls(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        with torch.no_grad():
            _, flow_up = self._model(img1, img2, iters=self._iters, test_mode=True)
        disp = -padder.unpad(flow_up).squeeze().detach().cpu().numpy().astype(np.float64)
        disp = disp.reshape(y1 - y0, x1 - x0)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = intr.fx * intr.baseline_m / disp
        z[disp <= 0.0] = np.nan
        z[~np.isfinite(z)] = np.nan
        depth[y0:y1, x0:x1] = z.astype(np.float32)
        return depth
