"""Polynomial fill of a chosen degree (2, 3, 4, 5, 7, ...).

Not in MatchLens (linear only). Per gap, a least-squares polynomial of degree `degree` is fitted
per axis through the known samples on both sides of the gap (enough anchors for the degree:
ceil((degree + 1) / 2) + 1 per side), then evaluated inside the gap. Time is centred and scaled
to [-1, 1] for numerical stability.

Expected behaviour (Runge's phenomenon): higher degrees bend harder between the anchors and
overshoot more inside a long gap; positions are clipped to the pitch (+2 m) in base.py. Falls
back to linear when a side lacks anchors.
"""
import math

import numpy as np

from .base import BallFill
from .linear import LinearFill


class PolyFill(BallFill):
    def __init__(self, max_gap=None, degree: int = 3, max_reach: int = 25):
        super().__init__(max_gap)
        self.degree = degree
        self.per_side = math.ceil((degree + 1) / 2) + 1
        self.max_reach = max_reach
        self.name = "poly%d" % degree
        self._linear = LinearFill()

    def _anchors(self, known, start, step):
        out, i = [], start
        while 0 <= i < len(known) and len(out) < self.per_side and abs(i - start) <= self.max_reach:
            if known[i]:
                out.append(i)
            i += step
        return out

    def fill_gap(self, t, x, y, known, a, b):
        left, right = self._anchors(known, a - 1, -1), self._anchors(known, b, +1)
        if len(left) < 2 or len(right) < 2 or len(left) + len(right) < self.degree + 1:
            return self._linear.fill_gap(t, x, y, known, a, b)
        idx = sorted(left + right)
        tt = t[idx].astype(float)
        c, h = (tt[0] + tt[-1]) / 2.0, max((tt[-1] - tt[0]) / 2.0, 1.0)
        u, q = (tt - c) / h, (t[a:b].astype(float) - c) / h
        px = np.polynomial.polynomial.polyfit(u, x[idx], self.degree)
        py = np.polynomial.polynomial.polyfit(u, y[idx], self.degree)
        return (np.polynomial.polynomial.polyval(q, px), np.polynomial.polynomial.polyval(q, py))
