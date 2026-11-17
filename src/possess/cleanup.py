"""Segment clean-up on finished H / A / D label runs (envelope L2-17-cleanup-fable).

The decoder prices every switch the same (c_switch, c_stop); it cannot say "a 0.5 s away spell
inside a home possession is less believable than a 0.5 s away spell after a stoppage". These
rules act on the finished runs of one half and depend on what flanks a run, how long it is,
whether the ball was hidden under it and whether a clean touch backs it. Nothing is learned and
no ball-in-stoppage feature is used.

    R1 flicker   team spell shorter than m_flicker whose two neighbours are the same other team
                 -> that team                                   (A h A  -> A A A)
    R2 bridge    D run shorter than m_bridge whose two neighbours are the same team -> that team
                                                                (H d H  -> H H H)
    R3 island    team spell shorter than m_island with D on both sides -> D   (D h D -> D D D)
    R4 sliver    team spell shorter than m_sliver between a D run and a spell of the other team
                 -> the other team                              (D h A -> D A A,  A h D -> A A D)
    protect      a team spell with an uncontested touch of its own team inside it
                 (decode.touch_hold) is exempt from R1 / R3 / R4 (Link & Hoernig 2017: contact,
                 not distance, makes a possession)

Order R3, R2, R4, R1, repeated until nothing changes, at most 3 passes. Thresholds in seconds;
0 = rule off; all 0 returns the input unchanged. Runs never cross a change of `period`, the first
and last run of a half have one neighbour only and are never touched, and X is never written.

matchlens_start: MatchLens has no segment-level merge pass (checked in the ported files:
stoppage_ml.py DeadBallDetector / PlacedBallDetector, team_control.py, deadball2.py, decode.py
touch_hold; the originals are possession.py, deadball.py, impulse.py, set_piece_detector.py).
Its minimum durations are frame-level hysteresis: HYSTERESIS_S 0.5, STATIONARY_DURATION_S 2.0,
min_hold_s 0.8, k = 5 control frames (0.2 s), deadball2 on 2 s / off 0.2 s; those values anchor
the grid (0.4, 0.8, 2 s). Li & Link (2026) clean a frame classifier's output with a 3 s floor.
"""
from typing import Dict, Optional

import numpy as np

RULES = ("island", "bridge", "sliver", "flicker")           # order of application (R3, R2, R4, R1)
KEYS = ("m_flicker", "m_bridge", "m_island", "m_sliver", "cl_protect", "cl_vis", "cl_dir")


def active(kw: dict) -> bool:
    """True when any rule threshold in `kw` is > 0."""
    return any(kw.get(k, 0) > 0 for k in KEYS[:4])


def _clean_block(lab: np.ndarray, a: int, b: int, n_hid: np.ndarray, n_th: Optional[np.ndarray],
                 n_ta: Optional[np.ndarray], thr: Dict[str, float], protect: int, cl_vis: int,
                 cl_dir: int, max_pass: int, counts: Optional[dict]) -> None:
    seg = lab[a:b]
    brk = (np.flatnonzero(seg[1:] != seg[:-1]) + 1).tolist()
    bounds = [a] + [a + i for i in brk] + [b]
    runs = [[bounds[i], bounds[i + 1], str(lab[bounds[i]])] for i in range(len(bounds) - 1)]

    def touched(s, e, team):
        if not protect or n_th is None:
            return False
        c = n_th if team == "H" else n_ta
        return c[e] - c[s] > 0

    def target(rule, i):
        s, e, cur = runs[i]
        prev, nxt = runs[i - 1][2], runs[i + 1][2]
        if e - s >= thr[rule]:
            return None
        if rule == "bridge":
            return prev if cur == "D" and prev == nxt and prev in ("H", "A") else None
        if cur not in ("H", "A"):
            return None
        other = "A" if cur == "H" else "H"
        if rule == "island":
            new = "D" if prev == "D" and nxt == "D" else None
        elif rule == "sliver":
            ok = (prev == "D" and nxt == other and cl_dir in (0, 1)) or \
                 (prev == other and nxt == "D" and cl_dir in (0, 2))
            new = other if ok else None
        else:
            new = other if prev == other and nxt == other else None
            if new and cl_vis:                               # 1: hidden-majority runs only, 2: visible-majority only
                hid = 2 * (n_hid[e] - n_hid[s]) > e - s
                if hid != (cl_vis == 1):
                    new = None
        if new and touched(s, e, cur):
            return None
        return new

    for _ in range(max_pass):
        changed = False
        for rule in RULES:
            if thr[rule] <= 0:
                continue
            i = 1
            while i < len(runs) - 1:
                new = target(rule, i)
                if new is None:
                    i += 1
                    continue
                changed = True
                if counts is not None:
                    counts[rule] = counts.get(rule, 0) + 1
                runs[i][2] = new
                if runs[i + 1][2] == new:                    # merge with the right neighbour
                    runs[i][1] = runs[i + 1][1]
                    del runs[i + 1]
                if runs[i - 1][2] == new:                    # and with the left one
                    runs[i - 1][1] = runs[i][1]
                    del runs[i]
                    i -= 1
                i = max(i, 1)                                # the merged run is looked at again
        if not changed:
            break
    for s, e, k in runs:
        lab[s:e] = k


def cleanup(lab: np.ndarray, period: np.ndarray, visible: np.ndarray, touch: np.ndarray = None,
            fps: float = 25.0, m_flicker: float = 0.0, m_bridge: float = 0.0, m_island: float = 0.0,
            m_sliver: float = 0.0, cl_protect: int = 0, cl_vis: int = 0, cl_dir: int = 0,
            max_pass: int = 3, counts: dict = None) -> np.ndarray:
    """Cleaned copy of `lab` ('H','A','D','X' per frame). See the module docstring for the rules.

    period: per frame, > 0 inside a half; each contiguous block of one period is cleaned alone.
    visible: ball seen on the frame (cl_vis: R1 on hidden-majority (1) / visible-majority (2) runs).
    touch: +1 / -1 on frames with an uncontested home / away touch (decode.touch_hold with
    max_hold_s=0), 0 elsewhere; only read when cl_protect=1.
    cl_dir: R4 direction, 0 both, 1 only D h A (after a stoppage), 2 only A h D (before one).
    counts: optional dict, incremented per rule with the number of runs it relabelled.
    """
    thr = {"flicker": m_flicker * fps, "bridge": m_bridge * fps, "island": m_island * fps,
           "sliver": m_sliver * fps}
    if not any(v > 0 for v in thr.values()):
        return lab
    out = np.array(lab, dtype="<U1")
    n = len(out)
    period = np.asarray(period)
    n_hid = np.concatenate([[0], np.cumsum(~np.asarray(visible, bool))])
    n_th = n_ta = None
    if cl_protect and touch is not None:
        n_th = np.concatenate([[0], np.cumsum(np.asarray(touch) > 0)])
        n_ta = np.concatenate([[0], np.cumsum(np.asarray(touch) < 0)])
    i = 0
    while i < n:
        if period[i] <= 0:
            i += 1
            continue
        j = i
        while j < n and period[j] == period[i]:
            j += 1
        _clean_block(out, i, j, n_hid, n_th, n_ta, thr, cl_protect, cl_vis, cl_dir, max_pass, counts)
        i = j
    return out
