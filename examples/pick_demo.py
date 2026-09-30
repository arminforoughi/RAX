"""Map, pick and place three tubes on the simulated SO-101. No hardware needed.

    python examples/pick_demo.py
"""

from rax.pick import ColourTarget, pick, place, scan
from rax.pick.sim import SimArm, Tube

arm = SimArm([Tube(0.24, 0.10, 30, "green"), Tube(0.27, 0.02, 120, "blue"),
              Tube(0.21, 0.05, 250, "red")])
tubes = ColourTarget()

found = scan(arm, tubes)
print(f"mapped {len(found)}: " + ", ".join(f"{f.label} ({f.x*100:.0f},{f.y*100:.0f})cm"
                                           for f in found))
for k, f in enumerate(found):
    try:
        res = pick(arm, tubes, near_xy=(f.x, f.y), label=f.label)
        at = place(arm, (0.20, -0.13 - 0.03 * k), release_z=0.10, pitch=0.0, roll=90.0)
        print(f"{f.label}: picked (jaws at {res.grip_pct:.0f}), placed at "
              f"({at[0]*100:.0f},{at[1]*100:.0f})cm")
    except RuntimeError as e:
        print(f"{f.label}: {e}")
