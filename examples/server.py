"""RAX on one arm: the general pick UI on :8484 and tube sorting on :8486, together.

    python examples/server.py --arm so101 --port COM4
    python examples/server.py --arm x250  --port COM5 --camera 1 --no-tubes

The arm is opened once and shared; a job started from either page holds it until done.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "pick_server"), os.path.join(HERE, "tube_sorting")]

from rig_app import Rig, arm_from_args, serve  # noqa: E402


def main():
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    a, arm = arm_from_args(__doc__.split("\n\n")[0])
    rig = Rig(arm)
    pick = load("pick_app", os.path.join(HERE, "pick_server", "server.py")).PickApp(rig)
    pick.start()
    apps = [(pick, 8484)]
    if not a.no_tubes:
        tube = load("tube_app", os.path.join(HERE, "tube_sorting", "server.py")).TubeApp(rig)
        apps.append((tube, 8486))
    serve(rig, apps)


if __name__ == "__main__":
    main()
