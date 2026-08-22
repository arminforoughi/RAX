"""RAX — approach and grasp, for an arm and a camera you already own.

The stack is four seams and the algorithms between them:

    robots.profiles     the arm as data: URDF, joint topology, limits, gripper
    manipulation.arms   ArmInterface — move joints, report state
    perception.cameras  CameraInterface — hand over a frame (stereo, RGB-D or mono)
    manipulation.arms.rig.Rig   joins one of each into a working rig

Porting to a new robot means writing a profile and, if your arm is not already
supported, a driver with four methods. Nothing in ``perception`` or ``manipulation``
knows what robot it is running on.

    from rax.robots.profiles import load_profile
    from rax.perception.cameras import make_camera
    from rax.manipulation.arms.rig import Rig, geometry_for

See ``docs/porting.md`` for the full contract, and ``python -m rax.grasp --help``
for the entry point.
"""

__version__ = "0.1.0"
