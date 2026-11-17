"""Clean the ball track before the touch detector reads it (plan step 2, envelope L2-9).

`decode.touch_hold` finds touches as ball impulses next to a player, so every tracking fault that
looks like an impulse becomes a false touch (Fable QA of L2-8: a 1-frame 5-10 m teleport adds
~3.7 false touch frames). Three faults seen in real tracking (docs/research/swarm-findings.md
"Data problems") are removed here:

  teleport   a run of <= max_tp frames that leaves the track faster than v_max and comes back
             (the frame after the run is reachable from the frame before it at v_max), lies
             >= dev_min off the line between those two frames and has no player within touch
             range. A jump that does not come back is a real change of track and is kept; so is
             a hard kick (one long step that stays near the line, next to a player).
  stale      the exact same position repeated for <= stale_max frames (2 s) while the ball was
             moving before the run and is moving again after it (a frozen reading). A ball that
             was already at rest, or whose motion before the run is unknown, is kept. Longer
             freezes are kept too: on the tune split (IDSSE, 16,600 such frames) they are
             stoppages whose end, the jump to the restart position, is a touch the detector
             needs (removing them all: tune 0.5657 -> 0.5600).
  off pitch  a reading more than off_m outside the touchlines / goal lines.

Removed frames are treated as hidden (nan); runs of <= max_fill removed frames with a good frame
on both sides are bridged by straight-line motion instead, so one bad frame does not cut a touch
hold or leave a hole in the smoothing window.

matchlens_start: the "coast through a bad detection" idea of the MatchLens core ball tracker
(reject a detection the ball cannot have reached, keep the last motion for a few frames) and
events/kinematics.py's maximum plausible ball speed gate. Changed: offline (the track after the
fault is known, so a teleport is told from a real jump by whether it comes back) and the gap is
bridged from both sides rather than extrapolated. Nothing in MatchLens covers stale repeats.
Constants: v_max 35 m/s, the fastest ball in the external truth data apart from jumps (99.9th
percentile of the per-frame speed on the tune split is 46-53 m/s, almost all of it single-frame
jumps); no constant was fitted to challenge data. dev_min 1.5 m and stale_max 50 frames were
chosen on the tune split with injected faults (envelope L2-9 "v2 after QA": dev_min 2.0 -> 1.5
recovers more of 1.4-5 m jumps, stale_max 10 -> 50 more of 0.4-2 s freezes, clean score equal).
"""
from typing import Optional, Tuple

import numpy as np


def _teleports(x: np.ndarray, y: np.ndarray, ok: np.ndarray, near: np.ndarray, step: float,
               max_tp: int, dev_min: float) -> np.ndarray:
    """One greedy pass in index order: runs of <= max_tp frames that left the track and came back.

    A run is removed only if every frame of it lies >= dev_min from the straight line between the
    frame before and the frame after it and no player is within touch range of it (`near`): a
    hard kick also makes one step longer than v_max allows, but it stays close to that line and
    starts at a player (Fable QA of L2-9). A run cut off by hidden frames has no frame after it;
    it is removed when it lies >= dev_min beyond what the ball could have reached, again with no
    player near."""
    n = len(x)
    bad = np.zeros(n, bool)
    g = -1                              # last frame on the track
    pend = []                           # frames off the track since g
    for i in np.flatnonzero(ok).tolist() + [n + max_tp + 2]:
        if g < 0 or i - g > max_tp + 1:
            # track start, or the gap since g is too long to judge. Frames still pending are an
            # isolated blip only if the visible stretch ended right after them; otherwise the
            # ball really jumped and they are the new track.
            if pend and len(pend) <= max_tp and i - pend[-1] > 1 and g >= 0 and not near[pend].any() and all(
                    np.hypot(x[p] - x[g], y[p] - y[g]) - step * (p - g) >= dev_min for p in pend):
                bad[pend] = True
            g, pend = i, []
            continue
        if np.hypot(x[i] - x[g], y[i] - y[g]) <= step * (i - g):
            if pend and not near[pend].any():
                t = (np.array(pend) - g) / float(i - g)
                dev = np.hypot(x[pend] - (x[g] + t * (x[i] - x[g])), y[pend] - (y[g] + t * (y[i] - y[g])))
                if (dev >= dev_min).all():
                    bad[pend] = True
            g, pend = i, []
        else:
            pend.append(i)
    return bad


def _stale(x: np.ndarray, y: np.ndarray, ok: np.ndarray, step: float, max_len: int) -> np.ndarray:
    """Exact repeats of the previous frame, in runs of <= max_len entered and left by a moving ball."""
    n = len(x)
    bad = np.zeros(n, bool)
    rep = np.zeros(n, bool)
    rep[1:] = ok[1:] & ok[:-1] & (x[1:] == x[:-1]) & (y[1:] == y[:-1])
    idx = np.flatnonzero(rep)
    if not len(idx):
        return bad
    starts = idx[np.r_[True, np.diff(idx) > 1]]
    ends = idx[np.r_[np.diff(idx) > 1, True]]
    for s, e in zip(starts.tolist(), ends.tolist()):
        a = s - 1                                         # the frame being repeated
        if e - s + 1 > max_len:
            continue                                      # a long freeze: kept (see module docstring)
        if e + 1 >= n or not ok[e + 1]:
            continue                                      # no exit to judge
        if np.hypot(x[e + 1] - x[e], y[e + 1] - y[e]) <= step:
            continue                                      # still at rest afterwards
        if a < 1 or not ok[a - 1] or np.hypot(x[a] - x[a - 1], y[a] - y[a - 1]) <= step:
            continue                                      # at rest before (a real stop), or unknown
        bad[s:e + 1] = True
    return bad


def clean_ball(ball_x: np.ndarray, ball_y: np.ndarray, d_near: Optional[np.ndarray] = None,
               r_touch: float = 1.0, fps: float = 25.0, v_max: float = 35.0, max_tp: int = 3,
               dev_min: float = 1.5, max_fill: int = 10, stale_v: float = 1.0, stale_max: int = 10,
               off_m: Optional[float] = None, half_l: float = 52.5, half_w: float = 34.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, removed) with teleports, stale repeats and off-pitch readings taken out.

    `removed` marks every frame judged faulty; in x / y those frames are nan, except runs of
    <= max_fill frames (stale runs: <= stale_max) between two good frames, which are linearly
    interpolated. d_near: distance (m) from the ball to the nearest
    player of either team; a teleport candidate with a player within r_touch is kept (it may be a
    kick). Positions are metres, origin at the centre spot. The inputs are not modified.
    """
    x, y = np.array(ball_x, float), np.array(ball_y, float)
    n = len(x)
    removed = np.zeros(n, bool)
    if n == 0:
        return x, y, removed
    ok = ~(np.isnan(x) | np.isnan(y))
    with np.errstate(invalid="ignore"):
        if off_m is not None:            # off by default (Fable re-QA: removed the real ball crossing the line)
            removed |= ok & ((np.abs(x) > half_l + off_m) | (np.abs(y) > half_w + off_m))
    ok &= ~removed
    stale = _stale(x, y, ok, stale_v / fps, stale_max)
    removed |= stale
    ok &= ~removed
    step = v_max / fps
    with np.errstate(invalid="ignore"):
        near = np.zeros(n, bool) if d_near is None else np.asarray(d_near) <= r_touch
    tp = _teleports(x, y, ok, near, step, max_tp, dev_min)
    rev = _teleports(x[::-1], y[::-1], ok[::-1], near[::-1], step, max_tp, dev_min)[::-1]   # blips at a stretch start
    removed |= tp | rev
    ok &= ~removed
    x[removed], y[removed] = np.nan, np.nan
    idx = np.flatnonzero(removed)
    if len(idx):
        starts = idx[np.r_[True, np.diff(idx) > 1]]
        ends = idx[np.r_[np.diff(idx) > 1, True]]
        for s, e in zip(starts.tolist(), ends.tolist()):
            a, b = s - 1, e + 1
            limit = stale_max if stale[s:e + 1].all() else max_fill      # a frozen reading is always bridged
            if e - s + 1 <= limit and a >= 0 and b < n and ok[a] and ok[b]:
                t = (np.arange(s, e + 1) - a) / float(b - a)
                x[s:e + 1] = x[a] + t * (x[b] - x[a])
                y[s:e + 1] = y[a] + t * (y[b] - y[a])
    return x, y, removed
