"""Ball gap-filling methods, one file per method (see docs/research/ball-interpolation.md).

Add a method: new file with a BallFill subclass, register it below, run
`scripts/blind_run_public5.py --ball-fill <name>`.
"""
from .base import BallFill
from .cubic import CubicFill
from .kalman import KalmanRTSFill
from .linear import LinearFill
from .physics import PhysicsFill
from .player_anchored import PlayerAnchoredFill

METHODS = {"linear": LinearFill, "cubic": CubicFill, "kalman": KalmanRTSFill, "physics": PhysicsFill,
           "player": PlayerAnchoredFill}


def get(name, **kw):
    """Method by name; 'polyN' gives a degree-N polynomial (poly2, poly4, poly7, ...)."""
    if name in (None, "none"):
        return None
    if name == "log" or (name.startswith("log") and name[3:].replace(".", "").isdigit()):
        from .warped import WarpedLineFill

        return WarpedLineFill(profile="log", c=float(name[3:] or 9), **kw)   # log, log3, log30
    if name.startswith("root") and name[4:].replace(".", "").isdigit():
        from .warped import WarpedLineFill

        return WarpedLineFill(profile="root", root=float(name[4:]), **kw)    # root2, root3
    if name.startswith("poly") and name[4:].isdigit():
        from .poly import PolyFill

        return PolyFill(degree=int(name[4:]), **kw)
    return METHODS[name](**kw)
