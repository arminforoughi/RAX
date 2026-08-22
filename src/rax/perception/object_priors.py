"""Per-class size and shape priors for the whole open-vocabulary label set.

These are a starting guess and a fallback, not a measurement: ``perception.measure``
measures the real footprint and yaw off the picture, and the map prefers that whenever
it succeeds. The priors matter when the object is clipped, tiny, or blends into the
table so the silhouette solve fails — and for apparent-size ranging, where the assumed
real width is what converts a bbox width into a distance.

YOLO-World is open-vocabulary, so the label string IS the class key: anything typed in
the query box works, it just gets the generic fallback if it is not listed. All 80 COCO
names are present so "every YOLO class" maps with sane numbers out of the box.

Nothing here depends on the robot, the camera, or the detector — it is a lookup table
and three accessors, which is why it can be shared by every localizer.
"""

from __future__ import annotations

import math

__all__ = [
    "ObjectPriors", "PRIORS", "CLASS_META", "COCO_CLASSES", "TABLE_CLASSES",
    "MAX_TABLE_OBJ_M", "DEFAULT_EDGE_M",
]

# Format: label -> (shape, width_m, depth_m, height_m); width/depth are the footprint
# on the table, height the vertical extent.
_CLASS_TABLE = {
    # --- the original cubes (measured on the real blocks) ---
    "red cube":      ("cube",     0.0508, 0.0508, 0.0508),
    "green cube":    ("cube",     0.0508, 0.0508, 0.0508),
    "blue cube":     ("cube",     0.0508, 0.0508, 0.0508),
    "yellow cube":   ("cube",     0.0508, 0.0508, 0.0508),
    "toy block":     ("cube",     0.0508, 0.0508, 0.0508),
    # --- COCO: people & animals ---
    "person":        ("cylinder", 0.45,  0.30,  1.70),
    "bird":          ("cuboid",   0.10,  0.22,  0.16),
    "cat":           ("cuboid",   0.18,  0.46,  0.25),
    "dog":           ("cuboid",   0.25,  0.70,  0.50),
    "horse":         ("cuboid",   0.60,  2.20,  1.60),
    "sheep":         ("cuboid",   0.40,  1.20,  0.90),
    "cow":           ("cuboid",   0.70,  2.40,  1.50),
    "elephant":      ("cuboid",   1.50,  4.00,  3.00),
    "bear":          ("cuboid",   0.80,  1.80,  1.20),
    "zebra":         ("cuboid",   0.60,  2.20,  1.50),
    "giraffe":       ("cuboid",   0.80,  2.50,  4.50),
    # --- COCO: vehicles & street ---
    "bicycle":       ("cuboid",   0.60,  1.75,  1.10),
    "car":           ("cuboid",   1.80,  4.50,  1.50),
    "motorcycle":    ("cuboid",   0.80,  2.10,  1.20),
    "airplane":      ("cuboid",  30.0,  35.0,  10.0),
    "bus":           ("cuboid",   2.55, 12.0,   3.20),
    "train":         ("cuboid",   3.00, 25.0,   4.00),
    "truck":         ("cuboid",   2.50,  8.00,  3.00),
    "boat":          ("cuboid",   2.00,  6.00,  2.00),
    "traffic light": ("cuboid",   0.30,  0.30,  1.00),
    "fire hydrant":  ("cylinder", 0.30,  0.30,  0.75),
    "stop sign":     ("cuboid",   0.75,  0.05,  2.10),
    "parking meter": ("cuboid",   0.15,  0.15,  1.20),
    "bench":         ("cuboid",   0.55,  1.50,  0.85),
    # --- COCO: accessories & sport ---
    "backpack":      ("cuboid",   0.32,  0.20,  0.45),
    "umbrella":      ("cylinder", 0.06,  0.06,  0.90),
    "handbag":       ("cuboid",   0.32,  0.14,  0.26),
    "tie":           ("cuboid",   0.08,  0.02,  0.55),
    "suitcase":      ("cuboid",   0.45,  0.22,  0.65),
    "frisbee":       ("cylinder", 0.27,  0.27,  0.03),
    "skis":          ("cuboid",   0.12,  1.70,  0.05),
    "snowboard":     ("cuboid",   0.28,  1.50,  0.03),
    "sports ball":   ("sphere",   0.22,  0.22,  0.22),
    "kite":          ("cuboid",   1.00,  0.60,  0.05),
    "baseball bat":  ("cylinder", 0.07,  0.07,  0.85),
    "baseball glove":("cuboid",   0.25,  0.15,  0.30),
    "skateboard":    ("cuboid",   0.21,  0.80,  0.11),
    "surfboard":     ("cuboid",   0.50,  2.10,  0.07),
    "tennis racket": ("cuboid",   0.28,  0.68,  0.03),
    # --- COCO: tabletop (the ones this arm can actually pick) ---
    "bottle":        ("cylinder", 0.068, 0.068, 0.23),
    "wine glass":    ("cylinder", 0.080, 0.080, 0.20),
    "cup":           ("cylinder", 0.080, 0.080, 0.10),
    "pen cup":       ("cylinder", 0.075, 0.075, 0.10),
    "mug":           ("cylinder", 0.085, 0.085, 0.10),
    "fork":          ("cuboid",   0.025, 0.19,  0.012),
    "knife":         ("cuboid",   0.022, 0.22,  0.012),
    "spoon":         ("cuboid",   0.035, 0.18,  0.012),
    "bowl":          ("cylinder", 0.15,  0.15,  0.07),
    "banana":        ("cuboid",   0.045, 0.19,  0.040),
    "apple":         ("sphere",   0.078, 0.078, 0.078),
    "sandwich":      ("cuboid",   0.12,  0.12,  0.05),
    "orange":        ("sphere",   0.075, 0.075, 0.075),
    "broccoli":      ("sphere",   0.12,  0.12,  0.14),
    "carrot":        ("cuboid",   0.035, 0.17,  0.035),
    "hot dog":       ("cuboid",   0.050, 0.16,  0.050),
    "pizza":         ("cylinder", 0.30,  0.30,  0.03),
    "donut":         ("cylinder", 0.095, 0.095, 0.045),
    "cake":          ("cylinder", 0.22,  0.22,  0.10),
    # --- COCO: furniture & appliances ---
    "chair":         ("cuboid",   0.45,  0.45,  0.90),
    "couch":         ("cuboid",   0.90,  2.00,  0.80),
    "potted plant":  ("cylinder", 0.22,  0.22,  0.40),
    "bed":           ("cuboid",   1.50,  2.00,  0.60),
    "dining table":  ("cuboid",   0.90,  1.60,  0.75),
    "toilet":        ("cuboid",   0.38,  0.70,  0.75),
    "microwave":     ("cuboid",   0.50,  0.38,  0.30),
    "oven":          ("cuboid",   0.60,  0.60,  0.85),
    "toaster":       ("cuboid",   0.28,  0.18,  0.20),
    "sink":          ("cuboid",   0.55,  0.45,  0.20),
    "refrigerator":  ("cuboid",   0.70,  0.70,  1.80),
    # --- COCO: electronics & small objects ---
    "tv":            ("cuboid",   1.10,  0.08,  0.65),
    "laptop":        ("cuboid",   0.33,  0.24,  0.02),
    "mouse":         ("cuboid",   0.062, 0.11,  0.038),
    "remote":        ("cuboid",   0.045, 0.16,  0.022),
    "keyboard":      ("cuboid",   0.44,  0.14,  0.025),
    "cell phone":    ("cuboid",   0.072, 0.15,  0.009),
    "book":          ("cuboid",   0.15,  0.22,  0.030),
    "clock":         ("cylinder", 0.25,  0.25,  0.05),
    "vase":          ("cylinder", 0.12,  0.12,  0.25),
    "scissors":      ("cuboid",   0.065, 0.18,  0.010),
    "teddy bear":    ("cuboid",   0.22,  0.15,  0.32),
    "hair drier":    ("cuboid",   0.085, 0.22,  0.22),
    "toothbrush":    ("cuboid",   0.015, 0.19,  0.015),
    # --- handy extras that are not COCO but come up on this table ---
    "pen":           ("cuboid",   0.010, 0.14,  0.010),
    "pencil":        ("cuboid",   0.008, 0.17,  0.008),
    "marker":        ("cylinder", 0.017, 0.017, 0.14),
    "eraser":        ("cuboid",   0.022, 0.055, 0.012),
    "screwdriver":   ("cuboid",   0.028, 0.21,  0.028),
    "tape":          ("cylinder", 0.075, 0.075, 0.025),
    "battery":       ("cylinder", 0.014, 0.014, 0.050),
    "usb stick":     ("cuboid",   0.018, 0.055, 0.009),
    "box":           ("cuboid",   0.10,  0.10,  0.10),
    "can":           ("cylinder", 0.066, 0.066, 0.12),
}

# Objects bigger than this in any footprint dimension cannot be on this table — used
# to reject a nonsense measurement, not to reject the detection.
MAX_TABLE_OBJ_M = 0.45
DEFAULT_EDGE_M = 0.0508     # generic fallback edge for an unlisted label

CLASS_META = {k: {"shape": v[0], "w_m": v[1], "d_m": v[2], "h_m": v[3]}
              for k, v in _CLASS_TABLE.items()}


class ObjectPriors:
    """Label -> size/shape prior, with a tunable fallback for unlisted labels.

    The fallback edge is live-tunable from the UI (/setcubesize), so it is an
    attribute rather than a constant — reading it through the instance is what makes
    a retune take effect everywhere instead of only where the value was first read.
    """

    def __init__(self, table=None, fallback_edge_m: float = DEFAULT_EDGE_M):
        self.table = dict(CLASS_META if table is None else table)
        self.fallback_edge_m = float(fallback_edge_m)

    def meta(self, label) -> dict:
        """Prior for a label: ``{shape, w_m, d_m, h_m}``. Unlisted -> a cube guess."""
        e = self.fallback_edge_m
        return self.table.get(str(label).strip().lower(),
                              {"shape": "cube", "w_m": e, "d_m": e, "h_m": e})

    def size_m(self, label) -> float:
        """Characteristic width for apparent-size ranging (what the bbox width maps to).

        For an object of unknown yaw the bbox width is somewhere between the
        footprint's minor and major axis, so the geometric mean is the least-wrong
        single number.
        """
        m = self.meta(label)
        return float(math.sqrt(max(m["w_m"], 1e-3) * max(m["d_m"], 1e-3)))

    def height_m(self, label) -> float:
        """Vertical extent above the table — used for hover/grasp height."""
        return float(self.meta(label)["h_m"])

    def known(self, label) -> bool:
        return str(label).strip().lower() in self.table


# The instance the stack shares. Rebind attributes on it rather than replacing it, so
# every holder of the reference sees a retune.
PRIORS = ObjectPriors()

# The 80 COCO names, in order — the "all YOLO classes" preset for the query box.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Query presets. "coco" is every class YOLO knows; "table" is the subset a tabletop arm
# can physically pick, which detects faster and keeps street furniture out of the map.
TABLE_CLASSES = [
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "orange", "book", "clock", "vase", "scissors", "teddy bear",
    "cell phone", "mouse", "remote", "keyboard", "laptop", "toothbrush",
    "pen", "pencil", "marker", "tape", "can", "box", "red cube", "green cube",
]
