"""Link visual geometry from a URDF, as triangles the browser can draw.

WHY THIS IS NOT lerobot's LOADER. lerobot's own route calls
`lerobot.utils.urdf_visual_meshes.load_link_visual_meshes_cached`, which resolves
`<mesh filename=...>` to STL files on disk. That works for the SO-101 and cannot work
for the X250: this repo has no X250 meshes, no Interbotix install to take them from, and
the X250 URDF next door therefore describes itself with `<box>` and `<cylinder>`
primitives instead. A mesh-only loader returns nothing for it, and the 3D view falls
back to the bare-polyline stick figure -- which is exactly the thing the mesh viewer was
written to replace.

So this tessellates PRIMITIVES natively and delegates MESHES to lerobot when it is
installed. It is a superset of what the server used before: the SO-101 keeps its real
STL silhouette, and any arm describable in boxes and cylinders gets a solid body without
shipping a single binary asset.

WINDING IS LOAD-BEARING, and getting it wrong looks like a specific bug. The viewer
backface-culls from the triangle's own normal (`n . viewdir >= 0` is skipped), and it
does that because drawing both sides paints the INSIDE of the arm over the outside and
makes it look like shattered glass -- there is a comment in viewer3d.js about exactly
that, and in `_decimate` about a dedupe that scrambled winding. Every triangle emitted
here is therefore counter-clockwise seen from OUTSIDE the solid, so its right-hand
normal points out. The unit tests check that by summing the divergence: a closed mesh
with consistent outward winding has signed volume > 0.

Output is LINK-LOCAL, which is the contract /geom relies on: this returns each link's
geometry in its own frame, and the server streams a 4x4 per link separately. The
`<visual><origin>` IS baked in here, because that is a property of the geometry within
the link, not of the link's pose in the world.
"""

from __future__ import annotations

import logging
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

__all__ = ["link_visuals", "box_mesh", "cylinder_mesh", "sphere_mesh", "signed_volume",
           "decimate", "MESH_VOXEL_M"]

#: Voxel size the STL path decimates to. 6 mm is what the mission server's viewer settled
#: on for the SO-101: it keeps the silhouette and the gap between the jaws while cutting
#: ~399k triangles to a few thousand. Primitives are already cheap and are never decimated.
MESH_VOXEL_M = 0.006

#: Facets around a cylinder or sphere. 16 is enough that a 3 cm link reads as round at
#: the size these are drawn, and cheap: the whole X250 comes to well under 1k triangles,
#: against the ~6k the decimated SO-101 STLs cost.
SEGMENTS = 16


# ---------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------
def box_mesh(sx: float, sy: float, sz: float) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box centred on the origin, outward-wound."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    V = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float64)
    F = np.array([
        [0, 3, 2], [0, 2, 1],        # -Z
        [4, 5, 6], [4, 6, 7],        # +Z
        [0, 1, 5], [0, 5, 4],        # -Y
        [3, 7, 6], [3, 6, 2],        # +Y
        [0, 4, 7], [0, 7, 3],        # -X
        [1, 2, 6], [1, 6, 5],        # +X
    ], dtype=np.int64)
    return V, F


def cylinder_mesh(radius: float, length: float,
                  segments: int = SEGMENTS) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder about the local Z axis, centred on the origin — the URDF convention."""
    n = max(3, int(segments))
    half = length / 2.0
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    bot = np.column_stack([ring, np.full(n, -half)])
    top = np.column_stack([ring, np.full(n, +half)])
    V = np.vstack([bot, top, [[0.0, 0.0, -half]], [[0.0, 0.0, +half]]])
    cb, ct = 2 * n, 2 * n + 1

    faces = []
    for i in range(n):
        j = (i + 1) % n
        # side: outward
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
        faces.append([cb, j, i])          # bottom cap, normal -Z
        faces.append([ct, n + i, n + j])  # top cap, normal +Z
    return V, np.array(faces, dtype=np.int64)


def sphere_mesh(radius: float, segments: int = SEGMENTS) -> tuple[np.ndarray, np.ndarray]:
    """Lat-long sphere centred on the origin."""
    n = max(4, int(segments))
    rings = max(2, n // 2)
    verts = [[0.0, 0.0, radius]]
    for r in range(1, rings):
        phi = math.pi * r / rings
        z, s = radius * math.cos(phi), radius * math.sin(phi)
        for k in range(n):
            th = 2.0 * math.pi * k / n
            verts.append([s * math.cos(th), s * math.sin(th), z])
    verts.append([0.0, 0.0, -radius])
    V = np.array(verts, dtype=np.float64)
    south = len(V) - 1

    def idx(r, k):
        return 1 + (r - 1) * n + (k % n)

    faces = []
    for k in range(n):
        faces.append([0, idx(1, k), idx(1, k + 1)])
    for r in range(1, rings - 1):
        for k in range(n):
            a, b = idx(r, k), idx(r, k + 1)
            c, d = idx(r + 1, k + 1), idx(r + 1, k)
            faces.append([a, d, c])
            faces.append([a, c, b])
    for k in range(n):
        faces.append([south, idx(rings - 1, k + 1), idx(rings - 1, k)])
    return V, np.array(faces, dtype=np.int64)


def signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Signed volume via the divergence theorem — positive iff wound outward.

    This is the winding check the tests use. A closed triangle soup whose normals all
    point out encloses a positive volume; flip any face and the sum drops.
    """
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


# ---------------------------------------------------------------------------------
# the URDF side
# ---------------------------------------------------------------------------------
def _origin_of(node) -> np.ndarray:
    o = node.find("origin") if node is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _primitive(geom) -> tuple[np.ndarray, np.ndarray] | None:
    box = geom.find("box")
    if box is not None:
        s = [float(v) for v in box.get("size", "0 0 0").split()]
        return box_mesh(*s[:3])
    cyl = geom.find("cylinder")
    if cyl is not None:
        return cylinder_mesh(float(cyl.get("radius", 0.0)), float(cyl.get("length", 0.0)))
    sph = geom.find("sphere")
    if sph is not None:
        return sphere_mesh(float(sph.get("radius", 0.0)))
    return None


def decimate(V: np.ndarray, F: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-cluster a dense STL down to something a browser can draw.

    Snaps vertices to a ``voxel`` grid, drops the triangles that collapse, dedupes. Keeps
    the true silhouette — and the gap between the jaws — unlike a convex hull. The SO-101's
    raw meshes are ~399k triangles, which is four hundred times what this viewer redraws
    comfortably at 5 Hz alongside a camera stream; at 6 mm they come to a few thousand.

    DEDUPE WITHOUT DESTROYING WINDING. Sorting the three indices inside a face (the obvious
    way to dedupe) scrambles its orientation, so half the normals end up pointing inward and
    the shading goes random light/dark — which is what once made the arm look like
    transparent shattered glass, and was blamed on the triangle budget for a while. Rolling
    each face so its smallest index leads is canonical for dedupe AND preserves cyclic
    order, so the backface cull downstream keeps working.
    """
    key = np.floor(V / voxel).astype(np.int64)
    _uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(_uniq)
    Vn = np.zeros((n, 3), np.float64)
    cnt = np.zeros(n, np.float64)
    np.add.at(Vn, inv, V)
    np.add.at(cnt, inv, 1.0)
    Vn /= np.maximum(cnt, 1.0)[:, None]
    Fn = inv[F]
    ok = ((Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2]))
    Fn = Fn[ok]
    if len(Fn) == 0:
        return Vn, Fn.reshape(0, 3)
    roll = np.argmin(Fn, axis=1)
    idx = (np.arange(3)[None, :] + roll[:, None]) % 3
    Fn = np.unique(np.take_along_axis(Fn, idx, axis=1), axis=0)
    return Vn, Fn


def _meshes_via_lerobot(mesh_dir: str) -> dict:
    """The STL path, delegated. Absent lerobot is not an error — just no meshes."""
    try:
        from lerobot.utils.urdf_visual_meshes import load_link_visual_meshes_cached
        return load_link_visual_meshes_cached(mesh_dir) or {}
    except Exception as e:
        logger.debug("urdf_visuals: no mesh loader (%s: %s)", type(e).__name__, e)
        return {}


def link_visuals(urdf_path: str | pathlib.Path, *, mesh_dir: str | None = None,
                 segments: int = SEGMENTS,
                 voxel: float | None = MESH_VOXEL_M
                 ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{link_name: (V, F)}`` in link-local coordinates, ready to draw.

    Primitives are tessellated here; ``<mesh>`` links are filled in from lerobot's loader
    when it is available and DECIMATED to ``voxel`` on the way out — pass ``voxel=None``
    to keep them dense. Decimation is not optional in practice: the SO-101's raw meshes are
    ~399k triangles against a viewer that redraws at 5 Hz next to a camera stream.

    A link with several ``<visual>`` blocks gets them concatenated, so a body built out of
    two boxes draws as both.
    """
    path = pathlib.Path(urdf_path)
    root = ET.parse(path).getroot()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wants_mesh = False

    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        parts: list[tuple[np.ndarray, np.ndarray]] = []
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue
            if geom.find("mesh") is not None:
                wants_mesh = True
                continue
            prim = _primitive(geom)
            if prim is None:
                continue
            V, F = prim
            T = _origin_of(vis)
            Vw = (T[:3, :3] @ V.T).T + T[:3, 3]
            parts.append((Vw, F))
        if parts:
            offs, Vs, Fs, k = [], [], [], 0
            for V, F in parts:
                Vs.append(V)
                Fs.append(F + k)
                k += len(V)
                offs.append(k)
            out[name] = (np.vstack(Vs), np.vstack(Fs))

    if wants_mesh:
        for name, (V, F) in _meshes_via_lerobot(mesh_dir or str(path.parent)).items():
            V, F = np.asarray(V, np.float64), np.asarray(F, np.int64)
            if voxel:
                V, F = decimate(V, F, float(voxel))
            # A mesh wins over a primitive for the same link: if someone drops the real
            # vendor URDF in, its meshes are what they wanted drawn.
            out[name] = (V, F)
    return out
