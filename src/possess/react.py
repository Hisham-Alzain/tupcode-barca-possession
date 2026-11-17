"""Player reaction to a change of possession while the ball is hidden (envelope L5-hidden-team).

When possession changes, the 22 players turn: the team that won the ball moves towards the
opponent's goal and the team that lost it retreats, so ALL tracked players flow the same way along
the pitch, and the flow reverses 1-2 s after a turnover (counter-pressing / recovery runs: Bauer &
Anzer 2021; team centroids move together and follow the ball: Frencken et al. 2011; the latest
player positions and velocities carry most of the ball information: Kim et al. 2023, Ballradar).
The shape model (shape.py) already holds the 1 s centroid velocity as two of 19 linear columns,
next to territory columns that change slowly; this module reads the reversal itself:

  flow      mean velocity of all tracked players along the pitch, positive towards the goal the
            home team attacks (m/s). Players are matched over a short lag by nearest neighbour
            inside their team (no identities; a player who appears or vanishes has no velocity,
            so a changing player count does not move the flow as it moves a centroid).
  reversal  flow(t + L) - flow(t - L): positive = the play turns towards home's attack.

Two opt-in uses (decode.decode_halves; both off by default, labels byte-identical):
  gap_cp    post-pass: a predicted H <-> A switch on a hidden frame is moved to the change point
            of the flow inside its hidden block (`place_switches`).
  w_rev     decoder transition: a switch on a hidden frame costs less where the flow reverses
            towards the team that takes the ball (`switch_discount`).

matchlens_start: footballtactics backend/events attacking_direction.py (per-period attacking side
from the goalkeepers; ported as shape.attack_dir, reused here) and possession.py (needs the ball
on every frame: nothing there or in impulse.py / kinematics.py / deadball.py reads player
reaction). The flow, the change point and the discount are new.
"""
from typing import Tuple

import numpy as np

from .shape import MIN_PLAYERS, V_MAX, attack_dir, smooth

FPS = 25.0
LAG_S = 0.8              # velocity lag = shape.SPEED_LAG_S (jitter-robust; not tuned)
REV_S = 1.0              # half-width of the reversal difference (s)


def player_vel(p: np.ndarray, fps: float = FPS, lag_s: float = LAG_S, chunk: int = 20000) -> np.ndarray:
    """x-velocity (m/s) of every player of one team, [n, SLOTS, 2] -> [n, SLOTS] (nan = empty slot
    or no match). As shape._player_speed: the player at t + h is matched to the nearest player of
    his team at t - h; a displacement above V_MAX is a new or swapped track."""
    n = len(p)
    h = max(1, int(round(lag_s * fps / 2)))
    dt = 2 * h / fps
    out = np.full((n, p.shape[1]), np.nan, dtype=np.float32)
    for s in range(h, n - h, chunk):
        e = min(n - h, s + chunk)
        a, b = p[s + h:e + h], p[s - h:e - h]
        d = np.hypot(a[:, :, None, 0] - b[:, None, :, 0], a[:, :, None, 1] - b[:, None, :, 1])
        d = np.where(np.isnan(d), np.inf, d)
        j = d.argmin(axis=2)
        dm = np.take_along_axis(d, j[:, :, None], axis=2)[:, :, 0]
        bx = np.take_along_axis(b[:, :, 0], j, axis=1)
        v = (a[:, :, 0] - bx) / dt
        v[(dm > V_MAX * dt) | np.isnan(a[:, :, 0])] = np.nan
        out[s:e] = v
    return out


def flow(pos: np.ndarray, period: np.ndarray, fps: float = FPS, smooth_s: float = 1.0) -> np.ndarray:
    """Mean x-velocity of all tracked players, signed so that positive = towards the goal the home
    team attacks; centred mean over `smooth_s`. nan where either team has < MIN_PLAYERS matched
    players, outside the periods, or where the attacking direction is not clear (shape.attack_dir)."""
    d = attack_dir(pos, period).astype(float)
    vh, va = player_vel(pos[:, 0], fps), player_vel(pos[:, 1], fps)
    ch, ca = np.isfinite(vh).sum(axis=1), np.isfinite(va).sum(axis=1)
    tot = np.nansum(vh, axis=1) + np.nansum(va, axis=1)
    ok = (ch >= MIN_PLAYERS) & (ca >= MIN_PLAYERS) & (d != 0) & (np.asarray(period) > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(ok, d * tot / (ch + ca), np.nan)
    return smooth(f, max(1, int(round(smooth_s * fps))))


def reversal(fl: np.ndarray, fps: float = FPS, half_s: float = REV_S) -> np.ndarray:
    """flow(t + L) - flow(t - L), 0 where either side is missing."""
    k = max(1, int(round(half_s * fps)))
    r = np.zeros(len(fl))
    r[k:-k] = fl[2 * k:] - fl[:-2 * k]
    return np.nan_to_num(r)


def place_switches(lab: np.ndarray, visible: np.ndarray, fl: np.ndarray, fps: float = FPS,
                   win_s: float = 4.0, lag_s: float = 0.0, min_side_s: float = 0.0) -> Tuple[np.ndarray, int]:
    """Move every predicted H <-> A switch that sits on a hidden frame to the change point of the
    flow. Returns (labels, number of switches moved).

    Search range: frames of the two spells around the switch, inside the same hidden block, at
    most `win_s` from the predicted switch, leaving `min_side_s` of each spell. The change point q
    maximises  sum_{t<q} s_old * flow + sum_{t>=q} s_new * flow  (s = +1 home, -1 away): the frame
    after which the players flow towards the new team's attack. `lag_s` is subtracted (players
    react after the turnover). A switch with no finite flow in its range stays where it is."""
    lab = np.array(lab, dtype="<U1")
    n = len(lab)
    vis = np.asarray(visible, bool)
    f = np.nan_to_num(fl)
    has = np.isfinite(fl)
    sw = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"]) & np.isin(lab[:-1], ["H", "A"])) + 1
    w, lag, ms = int(win_s * fps), int(round(lag_s * fps)), int(min_side_s * fps)
    moved = 0
    for p in sw:
        if vis[p] or lab[p] == lab[p - 1]:
            continue
        old, new = lab[p - 1], lab[p]
        lo = p
        while lo > 0 and lab[lo - 1] == old and not vis[lo - 1] and p - lo < w:
            lo -= 1
        hi = p
        while hi < n and lab[hi] == new and not vis[hi] and hi - p < w:
            hi += 1
        lo, hi = lo + ms, hi - ms                       # q in [lo, hi]
        if hi <= lo or not has[lo:hi].any():
            continue
        s_new = 1.0 if new == "H" else -1.0
        c = np.concatenate([[0.0], np.cumsum(f[lo:hi])])     # c[k] = sum of flow over lo .. lo+k-1
        q = lo + int(np.argmin(s_new * c))              # old before q: -s_new * c[k]; new after: s_new * (C - c[k])
        q = min(max(q - lag, lo), hi)
        if q != p:
            lab[min(p, q):max(p, q)] = new if q < p else old
            moved += 1
    return lab, moved


def switch_discount(fl: np.ndarray, visible: np.ndarray, fps: float = FPS, half_s: float = REV_S,
                    thr: float = 1.0, lag_s: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """(to_home, to_away): per-frame cue >= 0 for the decoder's switch cost on hidden frames.
    to_home[t] = max(0, reversal(t + lag) - thr): the flow turns towards home's attack `lag_s`
    after t by more than `thr` m/s; to_away the mirror image. 0 on visible frames."""
    r = reversal(fl, fps, half_s)
    k = int(round(lag_s * fps))
    if k > 0:
        r = np.concatenate([r[k:], np.zeros(k)])
    hid = ~np.asarray(visible, bool)
    return np.where(hid, np.maximum(0.0, r - thr), 0.0), np.where(hid, np.maximum(0.0, -r - thr), 0.0)
