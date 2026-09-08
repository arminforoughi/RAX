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

from dataclasses import dataclass, field

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
         doc="EXTRA lateral pixel trim, added to the derived grasp bias"),
    Knob("aim_dv", "aim_dv_px", 1.0, -300.0, 300.0,
         doc="vertical pixel trim on the fingertip aim point"),
    Knob("push_out_cm", "push_out_m", 0.01, -5.0, 30.0,
         doc="push every localization radially outward by this much"),
    Knob("range_scale", "range_scale", 1.0, 0.3, 3.0,
         doc="multiplier on every estimated range"),
    Knob("bearing_deg", "bearing_offset_deg", 1.0, -180.0, 180.0,
         doc="rotate the mapped bearing, correcting hand-eye heading error"),
    Knob("survey_pitch_deg", "survey_pitch_deg", 1.0, 0.0, 94.0,
         doc="wrist pitch the survey looks from; higher points the camera further down"),
)


@dataclass
class ApproachConfig:
    """Approach, descent, grasp and place geometry.

    Mutable and shared: the UI retunes it live, so holders keep the object rather than
    copying values out of it.
    """

    # --- where the survey looks from ----------------------------------------------
    # The wrist pitch (SO-101 id4) the survey pose holds. It is the ONE joint that
    # aims the camera without changing the arm's shape, which is why it is the knob:
    # everything the survey measures rides on how steeply the sightline meets the
    # table, so this is the highest-leverage number on the rig and it wants to be
    # dialled against a live map rather than edited and restarted.
    #
    # Measured on the SO-101 with the wrist camera, coverage of the reachable table
    # (r 18-42cm) across a full pan sweep, counting only views that clear
    # MIN_TABLE_INCIDENCE:
    #
    #     33.2 deg -> camera 3.0 deg ABOVE horizontal, 40% covered, blind inside 30cm
    #     53.2     -> +14.4 deg down, 91%
    #     63.2     -> +22.9 deg down, 100%
    #     68.2     -> +27.1 deg down, 100%   <- centre of the plateau
    #     78.2     -> +35.3 deg down, 100%
    #     88.2     -> +43.0 deg down, 96%, and the far edge starts dropping out
    #
    # The server overwrites this from the arm profile at startup; the value here is
    # the fallback for a rig that does not carry one.
    survey_pitch_deg: float = 68.2

    # --- approach staging ---------------------------------------------------------
    # Shift the hover target to the object's RIGHT so it stays on the LEFT of the
    # camera view during the approach and does not disappear under the gripper.
    right_trim_m: float = 0.050
    # Stop short of the object radially, so the arm does not drive past it -- and, more
    # importantly, so the object is still IN FRAME at the hover.
    #
    # Was 0.010. At 1 cm short the object left the camera's view at every single
    # approach on this rig: the centring servo reported "object not in view", fell back
    # to the staged estimate, and the final visual correction never ran once. The saved
    # miss frames show the table and a gripper finger with no object anywhere in them.
    # That is what made the grasp land on the object's centre and shove it, and it is
    # also why a badly wrong aim_du sat unnoticed for so long -- the servo it feeds was
    # never reached. At 4 cm the object stays visible and the servo converges in about
    # two iterations. Measured 2026-09-06 over repeated picks, both values.
    back_m: float = 0.040
    # Step 1 closes ~90% of the gap, the rest are small corrections. Tried 2 with a
    # full-distance first move and it missed more: arriving with no margin left means
    # any residual localization error lands as a miss.
    steps: int = 3
    first_step_frac: float = 0.9
    max_first_step_m: float = 0.12
    # Below this remaining distance the hover is already reached.
    arrived_m: float = 0.015
    # A re-localization that jumps further than this is rejected as a MIS-DETECTION
    # (a different object) rather than steered toward.
    #
    # Was 0.05, and that number silently defeated the whole staged approach. The
    # initial fix comes from the 2D map, whose systematic error on a real table runs
    # 5-10cm; every close-up re-measure therefore disagreed by MORE than 5cm and was
    # discarded, so the cube estimate never changed between stages ("cube=(39.0,9.4)"
    # identical at approach 1, 2 and 3) and the arm drove three blind hops to a target
    # it had already got wrong. The rejected jumps WERE the correction.
    #
    # The bound's real job is only to catch the detector latching a different object,
    # so it is set well outside the localizer's own error and the damping below —
    # not this gate — handles noise.
    max_refine_jump_m: float = 0.12
    # How far to move toward a refined fix, rather than snapping onto it. Damped
    # because a single close-up read is better than the map but not authoritative;
    # over the staged hops this converges (0.3^3 = 3% of the initial error left)
    # while one outlier can only drag the target part of the way.
    refine_gain: float = 0.7
    # The FIRST confirmed close-up read is a different kind of evidence from the ones
    # after it. It is the first look from arm's length, and what it disagrees with is a
    # map fix whose own error runs 5-10cm. Damping it like a routine trim leaves most of
    # that error standing — and the later looks that were supposed to work it off are
    # exactly the ones that fail, because by then the object is under the gripper. So
    # the first one nearly snaps; the rest keep refine_gain.
    #
    # The value only matters when stage 1 is the ONLY look that lands, which is common
    # — once the object is under the gripper the later looks fail. Replayed against a
    # real pick whose map fix was 7.9cm out: one good read then losing sight of it ends
    # 0.8cm off at 0.9 against 2.4cm at 0.7, and one read that is 8cm WRONG then losing
    # sight of it ends 12.0cm off against 11.0cm. It buys 1.6cm on a good read and costs
    # 1.0cm on a bad one, and a bad one has to get past the clipped check, the jump gate
    # and cap_reach first. When several looks land, the gain makes no difference at all
    # (3.0cm either way) — they converge on the same place.
    first_refine_gain: float = 0.9
    # How much FURTHER OUT than the original estimate the refines are allowed to walk
    # the target, in total. Close-up re-measures are biased outward — see cap_reach —
    # and a staged approach compounds that bias into a drive straight past the object.
    # Their bearing is still taken in full; only the reach is held. Sized to cover the
    # initial fix's own short reads without licensing another stage of creep.
    max_refine_out_m: float = 0.03

    # --- visual centering ---------------------------------------------------------
    # Lateral pixel trim on the aim point. NEGATIVE shifts the aim LEFT in the image,
    # which drives the arm RIGHT relative to the object.
    #
    # -45 -> -140 after the centring servo started converging. The trim is really
    # compensating for what HAND_UV is: it was measured with /caltip against ONE black
    # fingertip, not the midpoint between the jaws, so centring the object on it parks
    # that finger on the object instead of straddling it — observed on hardware as the
    # right finger sitting in the middle of the cube.
    #
    # 95 px is 2 cm at the grasp pose: fx=517, camera ~11 cm off the object, so
    # fx * 0.02 / 0.11 = 94 px. Cross-checks against the same run's numbers — the
    # servo accepted du=45px, so its tolerance was the 60px cap, which means the cube's
    # apparent width was >180px, i.e. >35px/cm for a 5.08cm cube.
    #
    # THE PRINCIPLED FIX is to re-measure HAND_UV at the midpoint between the jaws
    # rather than on one fingertip; this knob would then sit near zero. Until then it
    # is a per-rig constant and it belongs on the dial.
    # An ADDITIVE trim on top of the derived grasp bias (derive.grasp_aim_offset_px),
    # not the bias itself. It was -140.0, a hand-dialled pixel count that is 9.5 cm at
    # 35 cm range and 4.8 cm at 18 cm — enough to put a 5 cm cube wholly outside the
    # jaws. It went unnoticed because the centring servo it feeds never converged (the
    # object left frame at the hover), so the value was never applied. With the servo
    # running it closed on air every time. 0.0 = take the derived bias as-is.
    aim_du_px: float = 0.0
    aim_dv_px: float = 0.0
    # Object this close to the aim pixel counts as centred. Deliberately loose: the
    # staged approach already gets close, and chasing a tight pixel tolerance with
    # coarse radial reach moves costs iterations without improving the grasp.
    align_tol_px: float = 40.0
    # Was 3, and 3 could not finish the job it was given. The servo inherits whatever
    # lateral offset the approach parked at — right_trim_m, ~250 px at grasp range —
    # and each iteration is capped at 4.5 deg of pan when close. Measured on a real
    # pick: 268 px of error at 12.8 px/deg needs 21 deg of pan, against a budget of
    # 3 x 4.5 = 13.5 deg. The loop was structurally unable to converge and reported
    # max_iters on every pick in the log, so the grasp always fell back to the
    # uncorrected mapped position. Raised to give the loop more actuation than the
    # error it is handed; the tolerance, not the counter, should be what ends it.
    # (The trim now also decays across the approach — see trim_final_frac — so this
    # budget is sized against a much smaller starting error than the one measured.)
    align_iters: int = 8

    # How much of right_trim_m is left at the LAST approach stage. The trim exists to
    # keep the object off to one side so it stays in frame during transit, and that
    # need is strongest early — far away, swinging — and weakest at the final hop,
    # where it converts directly into pixel error the centring servo must undo. Decay
    # it and the servo inherits an error it can actually close, without giving up the
    # visibility the trim was there to buy. Full trim at stage 1, this fraction at the
    # last stage. 1.0 restores the old fixed-trim behaviour.
    trim_final_frac: float = 0.3

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
