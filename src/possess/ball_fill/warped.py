"""Time-warped straight-line fills: logarithmic and root speed profiles.

Same path as MatchLens's linear fill (linear.py: straight line last-known -> next-known), but the
ball does not move at constant speed along it. With tau = fraction of the gap elapsed (0..1):

    linear :  s = tau
    log    :  s = ln(1 + c * tau) / ln(1 + c)        c > 0, larger c = harder deceleration
    rootN  :  s = tau ** (1 / N)                      N = 2 (square root), 3 (cube root), ...

Both put most of the travel early and slow down late: the shape of a struck ball losing speed
(compare physics.py, whose one-sided drag flight is 1 - exp(-k t)). Untuned (tune c and N on
external data only).
"""
import numpy as np

from .base import BallFill


class WarpedLineFill(BallFill):
    def __init__(self, max_gap=None, profile: str = "log", c: float = 9.0, root: float = 2.0):
        super().__init__(max_gap)
        self.profile, self.c, self.root = profile, c, root
        self.name = "log" if profile == "log" else "root%g" % root

    def warp(self, tau: np.ndarray) -> np.ndarray:
        if self.profile == "log":
            return np.log1p(self.c * tau) / np.log1p(self.c)
        return tau ** (1.0 / self.root)

    def fill_gap(self, t, x, y, known, a, b):
        t0, t1 = t[a - 1], t[b]
        s = self.warp((t[a:b] - t0) / float(t1 - t0))
        return x[a - 1] + (x[b] - x[a - 1]) * s, y[a - 1] + (y[b] - y[a - 1]) * s
