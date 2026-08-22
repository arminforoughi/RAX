"""The published entry point, run end to end on every rig it claims to support.

``test_head_camera.py`` proves the *pieces* compose. This proves the thing a stranger
actually types works — ``python -m rax.grasp --arm X --query Y`` — from profile lookup
through detection, localization, staging, descent and a current-sensed close.

Every case here runs with no hardware and no model weights, which is the only reason it
can be a test rather than a demo someone occasionally remembers to run.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rax.grasp import build_parser, build_rig, main  # noqa: E402
from rax.manipulation.arms.mock_arm import table_scene  # noqa: E402

#: Where table_scene puts each object. The CLI is never told these.
TRUTH = {
    "red object": np.array([0.26, -0.07]),
    "green object": np.array([0.30, 0.06]),
    "blue object": np.array([0.22, 0.11]),
}


def _argv(**kw) -> list[str]:
    args = ["--survey-frames", "6", "--settle", "0"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return args


# --- the rigs the README advertises ---------------------------------------------
@pytest.mark.parametrize("arm,camera", [
    ("head_mono", None),      # webcam on a mast — the cheapest supported rig
    ("head_mono", "rgbd"),    # same arm, depth sensor instead
    ("mock", None),           # overhead stereo
])
def test_cli_picks_an_object_on_every_rig(arm, camera, capsys):
    argv = _argv(arm=arm, query="red object")
    if camera:
        argv += ["--camera", camera]
    assert main(argv) == 0, "the pick did not report success"
    out = capsys.readouterr().out
    assert "contact=True" in out, f"the gripper never felt the object:\n{out}"
    assert "holding" in out


@pytest.mark.parametrize("query", list(TRUTH))
def test_cli_locates_each_object_where_it_actually_is(query, capsys):
    """Localization accuracy, read off the CLI's own report.

    A pipeline can 'succeed' while reaching to the wrong place — the gripper closes on
    air and the current never spikes. Checking the reported position against the truth
    catches a sign error or a scale error that a pass/fail result would hide.
    """
    assert main(_argv(arm="head_mono", query=query) + ["--no-move"]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if query in ln and "at (" in ln]
    assert line, "the CLI never reported a position"
    got = np.array([float(v) for v in line[0].split("at (")[1].split(")")[0].split(",")])
    err = float(np.linalg.norm(got - TRUTH[query]))
    assert err < 0.02, f"{query}: reported {got}, actually at {TRUTH[query]} ({err*1000:.0f} mm)"


def test_no_move_never_commands_the_arm():
    """``--no-move`` has to be trustworthy — it is what people run first."""
    from rax.grasp import pick_with_fixed_camera

    args = build_parser().parse_args(
        _argv(arm="head_mono", query="red object") + ["--no-move"])
    profile, rig, kin, geom = build_rig(args)
    commanded = []
    rig.send_joint_targets = lambda q: commanded.append(np.asarray(q).copy())
    rig.set_gripper = lambda pct: commanded.append(("gripper", pct))

    assert pick_with_fixed_camera(rig, kin, geom, profile, args) == 0
    assert commanded == [], f"--no-move commanded the arm: {commanded}"


def test_missing_object_fails_loudly_rather_than_reaching_anyway():
    """Nothing detected must stop the run, not drive to a stale or default position."""
    assert main(_argv(arm="head_mono", query="giraffe")) == 1


def test_a_fixed_camera_rig_is_assembled_as_a_rig():
    """The head-camera path must actually go through the Rig seam, not around it."""
    from rax.manipulation.arms.rig import Rig

    args = build_parser().parse_args(_argv(arm="head_mono", query="red object"))
    profile, rig, _kin, _geom = build_rig(args)
    assert isinstance(rig, Rig)
    assert not profile.camera.eye_in_hand
    # The arm and the camera are separate objects — that is the whole claim.
    assert rig.arm is not rig.camera
    assert not hasattr(rig.arm, "frame"), "the arm should not own a camera"


def test_the_scene_the_mocks_render_matches_the_truth_table():
    """Pin the fixture: if table_scene moves, the accuracy assertions must move too."""
    for obj in table_scene(0.0):
        assert np.allclose(obj.center[:2], TRUTH[obj.label], atol=1e-9)
        # Resting on the table means centre height == radius, which is the exact
        # relationship PlaneRayLocalizer solves for.
        assert obj.center[2] == pytest.approx(obj.radius_m)
