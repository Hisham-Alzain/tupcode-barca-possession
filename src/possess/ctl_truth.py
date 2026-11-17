"""Controlled-possession relabelling of H / A / D label runs (envelope L4-control-definition).

The organisers' definition (CHALLENGE.md, "Change of possession"): possession changes only when
the other team gains CONTROLLED possession; a simple touch without control, a deflection of a pass
that does not intercept it and an aerial duel won without subsequent control are NOT a change;
while the ball is loose or contested possession stays with the last team that had it; after a
stoppage there is a clear possessor. Proxies they name for "controlled": a player keeps the ball a
few seconds; a team completes more than two consecutive passes.

The external truth (Metrica event feed, DFL BallPossession flag) gives every one-touch deflection
or tackle a team spell of its own. `relabel` rewrites such label runs to the written definition:

    a team spell is CONTROLLED when it lasts >= min_s seconds, or (min_pass > 0) when that team
    starts >= min_pass completed passes inside it;
    an uncontrolled spell that directly follows a live spell of the other team goes back to that
    team:  A b A -> A A A,  A b D -> A A D  (never controlled before the ball went dead);
    the first spell after a stoppage (D b ...) or at the start of a half is kept: the restart
    taker is the clear possessor;
    the shortest uncontrolled spell is absorbed first and the runs are merged before the next
    look, so  B a B  with a one-touch `a` becomes one B possession whose length then counts;
    a controlled spell keeps its first frame: a kept interception starts at its first touch.

The same function is the opt-in decoder post-pass `ctl_s` (decode.decode_halves): on predicted
labels it is the user's rule "a deflection that returns to the passing team is no change; an
interception that is kept changes possession from the first touch".

matchlens_start: nothing in MatchLens relabels finished spells (checked possession.py,
impulse.py, deadball.py, detectors/set_piece_detector.py; its minimum hold is the frame-level
min_hold_s 0.8 / HYSTERESIS_S 0.5). Closest in this repo: cleanup.py R1 flicker + R4 sliver
(cl_dir=2), which absorb in a fixed rule order instead of shortest first and need both neighbours.
"""
from typing import Optional

import numpy as np

TEAMS = ("H", "A")


def _block(lab: np.ndarray, a: int, b: int, min_len: float, n_ph: Optional[np.ndarray],
           n_pa: Optional[np.ndarray], min_pass: int, counts: Optional[dict]) -> None:
    seg = lab[a:b]
    brk = (np.flatnonzero(seg[1:] != seg[:-1]) + 1).tolist()
    bounds = [a] + [a + i for i in brk] + [b]
    runs = [[bounds[i], bounds[i + 1], str(lab[bounds[i]])] for i in range(len(bounds) - 1)]

    def controlled(s, e, team):
        if e - s >= min_len:
            return True
        if min_pass > 0 and n_ph is not None:
            c = n_ph if team == "H" else n_pa
            return c[e] - c[s] >= min_pass
        return False

    while True:
        best = -1
        for i in range(1, len(runs)):
            s, e, cur = runs[i]
            prev = runs[i - 1][2]
            if cur in TEAMS and prev in TEAMS and prev != cur and not controlled(s, e, cur) \
                    and (best < 0 or e - s < runs[best][1] - runs[best][0]):
                best = i
        if best < 0:
            break
        if counts is not None:
            counts["relabelled"] = counts.get("relabelled", 0) + 1
            counts["frames"] = counts.get("frames", 0) + runs[best][1] - runs[best][0]
        new = runs[best - 1][2]
        runs[best - 1][1] = runs[best][1]                    # merge into the previous spell
        del runs[best]
        if best < len(runs) and runs[best][2] == new:        # and with the next one of that team
            runs[best - 1][1] = runs[best][1]
            del runs[best]
    for s, e, k in runs:
        lab[s:e] = k


def relabel(lab: np.ndarray, period: np.ndarray, min_s: float = 2.0, fps: float = 25.0,
            passes: np.ndarray = None, min_pass: int = 0, counts: dict = None,
            free: np.ndarray = None) -> np.ndarray:
    """Copy of `lab` ('H','A','D','X' per frame) with uncontrolled team spells given back to the
    team in possession before them (module docstring). Each contiguous block of one period > 0 is
    treated alone; D and X runs are never written or removed.

    min_s: a spell at least this long is controlled; 0 with min_pass 0 returns the input.
    passes: per frame +1 / -1 where a completed home / away pass starts, 0 elsewhere (external
    event data); read only when min_pass > 0.
    counts: optional dict, receives "relabelled" (spells) and "frames".
    """
    if min_s <= 0 and min_pass <= 0:
        return lab
    out = np.array(lab, dtype="<U1")
    period = np.asarray(period)
    n = len(out)
    n_ph = n_pa = None
    if min_pass > 0 and passes is not None:
        n_ph = np.concatenate([[0], np.cumsum(np.asarray(passes) > 0)])
        n_pa = np.concatenate([[0], np.cumsum(np.asarray(passes) < 0)])
        if free is not None:                               # frames that count for either team (envelope L8)
            add = np.concatenate([[0], np.cumsum(np.asarray(free, bool))]) * min_pass
            n_ph, n_pa = n_ph + add, n_pa + add
    i = 0
    while i < n:
        if period[i] <= 0:
            i += 1
            continue
        j = i
        while j < n and period[j] == period[i]:
            j += 1
        _block(out, i, j, min_s * fps, n_ph, n_pa, min_pass, counts)
        i = j
    return out


def n_possessions(lab: np.ndarray, period: np.ndarray) -> int:
    """Team spells (maximal H or A runs) inside the played periods."""
    lab, play = np.asarray(lab), np.asarray(period) > 0
    start = np.ones(len(lab), bool)
    start[1:] = (lab[1:] != lab[:-1]) | (np.asarray(period)[1:] != np.asarray(period)[:-1])
    return int((start & play & np.isin(lab, TEAMS)).sum())
