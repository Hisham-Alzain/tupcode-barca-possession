"""Team-level possession from ball-to-player distances (loop iteration 2).

matchlens_start: MatchLens possession.py (footballtactics events/possession.py, Vidal-Codina §2.3 port; not vendored: only the zone constants below are used). Kept: the Possession Zone
(R_pz = 1.8 m around a player) and Duel Zone (R_dz = 2.3 m around the ball) and their meaning.
Changed, because the task is team-level, offline and without IDs:
  * zones are evaluated per TEAM (nearest player of each team), not per player;
  * a gain is committed after the challenging team has the ball alone in its zone on `k` visible
    frames; frames with no clear control (duel, nobody in a zone, ball hidden) neither add to nor
    reset that count, only control by the current team resets it (retention, CHALLENGE.md's
    "controlled possession"); the switch is dated at the k-th frame, instead of MatchLens's touch
    confirmation, which left 18-20% of visible frames without a possessor (bench iteration 2);
  * duel (both teams inside R_dz) or nobody in a zone keeps the current team;
  * ball not visible keeps the current team (possession survives ~75% of gaps, swarm F2).
Dead balls come from elsewhere (MatchLens DeadBallDetector, ghost rule off).
"""
from typing import Optional

import numpy as np


def team_from_distances(d_home: np.ndarray, d_away: np.ndarray, visible: np.ndarray,
                        r_pz: float = 1.8, r_dz: float = 2.3, k: int = 3,
                        dead: Optional[np.ndarray] = None, restart_flip: bool = False,
                        backdate: bool = False, period: Optional[np.ndarray] = None,
                        max_gap: Optional[int] = None) -> np.ndarray:
    """Per-frame team label 'H'/'A' ('X' before the first gain), 'D' where `dead`.

    d_home / d_away: distance (m) from the ball to the nearest home / away player (nan if unknown).
    restart_flip (plan P1): when play resumes after a stoppage, start with the OTHER team than the
      one before it (true after 78-90% of stoppages on the external truth); control evidence can
      still switch it back.
    backdate (plan P2): a committed switch is dated at the challenger's first control frame, not
      at the k-th.
    period: optional per-frame period; all state resets when it changes (no carry-over across
      half-time).
    max_gap: optional; a challenge is dropped when two of its control frames are more than
      `max_gap` frames apart (Fable QA 2026-11-13 edge case: one control frame before a 60 s ball
      gap lets back-dating repaint the whole gap). Off by default: on the tune split a cap costs
      score (none 0.3724, 250 frames 0.3707, 125 0.3694, 50 0.3669) because the turnover usually did
      happen inside the gap.
    """
    n = len(d_home)
    out = np.array(["X"] * n, dtype="<U1")
    cur, cand, run, cand_start, last_ctrl = "X", None, 0, 0, -10**9
    flip = {"H": "A", "A": "H", "X": "X"}
    was_dead = False
    for i in range(n):
        if period is not None and i > 0 and period[i] != period[i - 1]:
            cur, cand, run, was_dead = "X", None, 0, False
        if dead is not None and dead[i]:
            out[i] = "D"
            cand, run = None, 0
            was_dead = True
            continue
        if was_dead:
            was_dead = False
            if restart_flip:
                cur = flip[cur]
        if visible[i]:
            h, a = d_home[i], d_away[i]
            h_in = h == h and h <= r_pz
            a_in = a == a and a <= r_pz
            duel = (h == h and h <= r_dz) and (a == a and a <= r_dz)
            ctrl = None
            if not duel:
                if h_in and not a_in:
                    ctrl = "H"
                elif a_in and not h_in:
                    ctrl = "A"
            if ctrl is not None and ctrl != cur:
                if cand != ctrl or (max_gap is not None and i - last_ctrl > max_gap):
                    run, cand_start = 0, i
                run += 1
                last_ctrl = i
                cand = ctrl
                if run >= k:
                    if backdate:
                        seg = out[cand_start:i]
                        seg[seg != "D"] = ctrl
                    cur, cand, run = ctrl, None, 0
            elif ctrl == cur:
                cand, run = None, 0
        out[i] = cur
    return out
