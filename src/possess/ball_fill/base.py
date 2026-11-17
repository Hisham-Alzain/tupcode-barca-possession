"""Shared pieces for ball gap filling: find gaps, apply a per-gap fill, keep the ghost contract.

Every method works offline on one half at a time (arrays, look-ahead is free) and returns the
filled positions plus a mask. Filled frames are passed on with conf=None, MatchLens's
"Kalman-coasted ghost" convention (footballtactics events/eval/metrica.py), so touch detection
never fires on a guessed position.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

# Pitch in metres, centred (MatchLens convention), plus a small margin for balls just out.
X_LIM, Y_LIM = 52.5 + 2.0, 34.0 + 2.0


def gaps(known: np.ndarray) -> Iterator[Tuple[int, int]]:
    """(start, stop) index ranges of consecutive missing samples; stop is exclusive."""
    i, n = 0, len(known)
    while i < n:
        if known[i]:
            i += 1
            continue
        j = i
        while j < n and not known[j]:
            j += 1
        yield i, j
        i = j


class BallFill:
    """Base class. Subclasses implement `fill_gap`; `fill` handles the bookkeeping."""

    name = "base"

    def __init__(self, max_gap: Optional[int] = None):
        self.max_gap = max_gap  # frames; longer gaps stay empty. None = fill every gap

    def fill(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, known: np.ndarray):
        """Return (fx, fy, filled) for one half. t = frame ids (sorted), known = ball observed."""
        fx, fy = x.astype(float).copy(), y.astype(float).copy()
        filled = np.zeros(len(t), dtype=bool)
        for a, b in gaps(known):
            if self.max_gap is not None and b - a > self.max_gap:
                continue
            has_prev, has_next = a > 0, b < len(t)
            if not has_next:
                continue  # gap runs to the end of the half: nothing to aim at (MatchLens: no fill)
            if not has_prev:
                fx[a:b], fy[a:b] = x[b], y[b]  # gap at the start of a half: MatchLens holds next
            else:
                gx, gy = self.fill_gap(t, x, y, known, a, b)
                fx[a:b], fy[a:b] = np.clip(gx, -X_LIM, X_LIM), np.clip(gy, -Y_LIM, Y_LIM)
            filled[a:b] = True
        return fx, fy, filled

    def fill_gap(self, t, x, y, known, a: int, b: int):
        """Positions for samples a..b-1. Samples a-1 and b are known."""
        raise NotImplementedError
