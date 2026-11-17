"""Local cubic spline fill.

Not in MatchLens (it only had linear, see linear.py). Per gap, fit a cubic spline through up to
`k` known samples on each side of the gap (default 2 + 2 = an exact cubic), so the fill leaves
the last seen point with the ball's velocity and arrives at the next one with its velocity.
Falls back to linear when either side has fewer than 2 known samples nearby.

Known risk (docs/research/ball-interpolation.md): a touch inside the gap reverses the ball, and a
cubic can overshoot past both ends. Positions are clipped to the pitch (+2 m) in base.py.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from .base import BallFill
from .linear import LinearFill


class CubicFill(BallFill):
    name = "cubic"

    def __init__(self, max_gap=None, k: int = 2, max_reach: int = 12):
        super().__init__(max_gap)
        self.k = k                  # known anchor samples per side
        self.max_reach = max_reach  # how far (samples) to look for anchors beyond the gap edge
        self._linear = LinearFill()

    def _anchors(self, known, start, step, stop):
        out, i = [], start
        while 0 <= i < len(known) and len(out) < self.k and abs(i - start) <= self.max_reach:
            if known[i]:
                out.append(i)
            i += step
        return out

    def fill_gap(self, t, x, y, known, a, b):
        left = self._anchors(known, a - 1, -1, None)[::-1]
        right = self._anchors(known, b, +1, None)
        if len(left) < 2 or len(right) < 2:
            return self._linear.fill_gap(t, x, y, known, a, b)
        idx = left + right
        tt = t[idx].astype(float)
        sx, sy = CubicSpline(tt, x[idx]), CubicSpline(tt, y[idx])
        q = t[a:b].astype(float)
        return sx(q), sy(q)
