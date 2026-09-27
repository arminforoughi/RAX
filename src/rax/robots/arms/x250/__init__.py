"""The X250 (WidowX-250-class, 5 DoF + parallel gripper).

Only the parts that are about describing and converting this arm live here:

    X250/x250.urdf   a written kinematic sketch — structure and travel measured, link
                     lengths nominal. Read its header before trusting a distance.
    normalise.py     lerobot NORMALISED units <-> degrees, the one place that conversion
                     lives, and honest about which half of it is measured.

The DRIVER is `examples/vla/pick/x250_driver.py` and the servo adapter is
`examples/vla/pick/x250_servo.py`, both still under examples/ because they were written
against one lab's checkout and have not been run on hardware from here. The profile that
ties it all together is `rax.robots.profiles.x250`.

Nothing here is imported at package-import time: `normalise` reads a calibration file and
the URDF path is resolved lazily by the profile, so a machine that has never seen this arm
pays nothing.
"""
