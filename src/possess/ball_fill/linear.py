"""Two-sided linear fill — exact port of MatchLens.

Source: footballtactics events/eval/metrica.py:MetricaMatch.iter_frames(interpolate_ball=True):
straight line from the last known position to the next known one, i / (n + 1) along it.
"""
import numpy as np

from .base import BallFill


class LinearFill(BallFill):
    name = "linear"

    def fill_gap(self, t, x, y, known, a, b):
        t0, t1 = t[a - 1], t[b]
        s = (t[a:b] - t0) / float(t1 - t0)
        return x[a - 1] + (x[b] - x[a - 1]) * s, y[a - 1] + (y[b] - y[a - 1]) * s
