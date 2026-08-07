"""``ApproachConfig`` — every tunable of the approach, in one place with one idiom.

These knobs were module globals mutated from HTTP handlers, in two competing styles:
some rebound with ``global X``, others held in a one-element list ``X[0]`` precisely so
a rebind would be visible to code that had already imported the name. Both existed for
the same job, and neither survives being moved into a package.

They also form a **public API**. An autotuner drives ``right_trim_cm`` and ``back_cm``
by name through the HTTP routes, within numeric bounds, and the admin UI reads the same
names out of ``/status``. So this exposes get/set by knob name with the bounds attached,
and the routes become thin shims over it.

Knob names and units are the ones already on the wire — centimetres and degrees, not
metres and radians — because renaming them would break the tuner and the UI for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

__all__ = ["ApproachConfig", "Knob", "KNOBS"]


@dataclass(frozen=True)
class Knob:
    """One externally-tunable value: where it lives, what the wire calls it, its units."""

    name: str                  # the wire name (/status key, tuner key)
    attr: str                  # the ApproachConfig field it maps to
    scale: float = 1.0         # wire value * scale = stored value (cm -> m)
    lo: float | None = None    # bounds in WIRE units
    hi: float | None = None
    integer: bool = False
    doc: str = ""


#: The tunable surface. Bounds are the ones the HTTP routes already enforce, not the
#: autotuner's narrower walking range — the tuner clamps itself, and tightening here
#: would silently shrink what an operator can dial in from the UI.
KNOBS: tuple[Knob, ...] = (
    Knob("right_trim_cm", "right_trim_m", 0.01, -10.0, 15.0,
         doc="shift the approach hover this far to the object's RIGHT"),
    Knob("back_cm", "back_m", 0.01, 0.0, 10.0,
         doc="stop the approach hover this far short of the object, radially"),
    Knob("approach_steps", "steps", 1.0, 1.0, 12.0, integer=True,
         doc="how many staged hops close the distance"),
    Knob("aim_du", "aim_du_px", 1.0, -300.0, 300.0,
         doc="lateral pixel trim on the fingertip aim point"),
    Knob("aim_dv", "aim_dv_px", 1.0, -300.0, 300.0,
         doc="vertical pixel trim on the fingertip aim point"),
    Knob("push_out_cm", "push_out_m", 0.01, -5.0, 30.0,
         doc="push every localization radially outward by this much"),
    Knob("range_scale", "range_scale", 1.0, 0.3, 3.0,
         doc="multiplier on every estimated range"),
    Knob("bearing_deg", "bearing_offset_deg", 1.0, -180.0, 180.0,
         doc="rotate the mapped bearing, correcting hand-eye heading error"),
)


@dataclass
class ApproachConfig:
    """Approach, descent, grasp and place geometry.

    Mutable and shared: the UI retunes it live, so holders keep the object rather than
    copying values out of it.
    """

    # --- approach staging ---------------------------------------------------------
    # Shift the hover target to the object's RIGHT so it stays on the LEFT of the
    # camera view during the approach and does not disappear under the gripper.
    right_trim_m: float = 0.050
    # Stop short of the object radially, so the arm does not drive past it.
    back_m: float = 0.010
    # Step 1 closes ~90% of the gap, the rest are small corrections. Tried 2 with a
    # full-distance first move and it missed more: arriving with no margin left means
    # any residual localization error lands as a miss.
    steps: int = 3
    first_step_frac: float = 0.9
    max_first_step_m: float = 0.12
    # Below this remaining distance the hover is already reached.
    arrived_m: float = 0.015
    # A re-localization that jumps further than this is rejected as a bad fix rather
    # than steered toward.
    max_refine_jump_m: float = 0.05

    # --- visual centering ---------------------------------------------------------
    aim_du_px: float = -45.0
    aim_dv_px: float = 0.0
    align_tol_px: float = 12.0
    align_iters: int = 3

    # --- localization corrections -------------------------------------------------
    push_out_m: float = 0.0
    range_scale: float = 1.0
    bearing_offset_deg: float = 0.0

    # --- pick heights -------------------------------------------------------------
    hover_z_m: float = 0.06        # hover this far above the table before descending
    grasp_z_m: float = 0.015       # descend to here (grab the object's lower body)
    lift_m: float = 0.10
    standoff_m: float = 0.05       # park this far above, then descend
    descent_fracs: tuple[float, ...] = (0.4, 0.75, 1.0)

    # --- place / stack ------------------------------------------------------------
    place_clear_m: float = 0.008   # gap left under the carried object at release
    place_hover_m: float = 0.07    # hover this far above the release height
    place_transit_z_m: float = 0.16  # carry height while traversing
    place_retreat_m: float = 0.09  # straight-up retreat after releasing

    _knobs: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self._knobs = {k.name: k for k in KNOBS}

    # --- the knob API -------------------------------------------------------------
    def knob_names(self) -> list[str]:
        return [k.name for k in KNOBS]

    def get_knob(self, name: str) -> float:
        """Current value of a knob, in wire units."""
        k = self._knob(name)
        v = getattr(self, k.attr) / k.scale
        return int(round(v)) if k.integer else float(v)

    def set_knob(self, name: str, value) -> float:
        """Set a knob from a wire value, clamped to its bounds. Returns what was set.

        Clamped rather than rejected: these arrive from a UI text box and an autotuner
        that walks values around, and refusing a slightly out-of-range proposal is a
        worse failure than pinning it at the limit.
        """
        k = self._knob(name)
        v = float(value)
        if k.lo is not None:
            v = max(v, k.lo)
        if k.hi is not None:
            v = min(v, k.hi)
        if k.integer:
            v = round(v)
        stored = v * k.scale
        setattr(self, k.attr, int(stored) if k.integer else float(stored))
        return self.get_knob(name)

    def as_dict(self) -> dict:
        """The /status "tune" block: every knob in wire units."""
        out = {}
        for k in KNOBS:
            v = self.get_knob(k.name)
            out[k.name] = v if k.integer else round(v, 2)
        return out

    def _knob(self, name: str) -> Knob:
        try:
            return self._knobs[name]
        except KeyError:
            raise KeyError(
                f"unknown knob {name!r}; available: {sorted(self._knobs)}") from None

    def describe(self) -> str:
        return "\n".join(f"{k.name:<16} {self.get_knob(k.name):>8}   {k.doc}"
                         for k in KNOBS)
