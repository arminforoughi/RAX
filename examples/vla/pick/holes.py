"""Rack holes — moved to ``rax.perception.rack_holes``.

This shim stays so `pick.py` keeps importing `holes` unchanged. The detector itself is
not X250-specific (it is Hough circles plus a darkness test, and it needs no calibration
and no colour model), so it belongs where the SO-101 can reach it too — a rack is a rack
whichever arm is putting a tube in it.
"""

from rax.perception.rack_holes import Hole, draw, find_holes, pick_free_hole

__all__ = ["Hole", "find_holes", "pick_free_hole", "draw"]
