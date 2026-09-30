"""Pick and place any object a camera can detect, with no training.

    from rax.pick import pick, place, scan, ColourTarget, PromptTarget, TOP, SIDE

    target = PromptTarget(prompt="cup", grasp_z=0.04)
    for obj in scan(arm, target):
        pick(arm, target, near_xy=(obj.x, obj.y), grasp=SIDE)
        place(arm, (0.2, -0.15), release_z=0.06, pitch=0, roll=0)

``arm`` is anything implementing :class:`rax.pick.arm.Arm`.
"""

from .arm import Arm
from .pick import SIDE, TOP, Grasp, PickConfig, PickResult, pick
from .place import Found, place, scan
from .targets import ColourTarget, Detection, PromptTarget, Target
from .track import StickyTarget

__all__ = ["Arm", "pick", "place", "scan", "Grasp", "TOP", "SIDE", "PickConfig",
           "PickResult", "Found", "Target", "ColourTarget", "PromptTarget", "StickyTarget",
           "Detection"]
