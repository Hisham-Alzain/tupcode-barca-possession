"""Learned ball-aware team evidence (envelope L8-autonomous-fable round 5, T125; opt-in via
decode.decode_halves w_bt, 0 = off).

Which team holds the visible ball, from the context BEFORE the frame: who is at the ball now, who
was nearest in the last 0.5-4 s, which team stood at the last ball impulse, the flight direction,
the players near the ball, the pitch zone, and the ball-free shape evidence. One antisymmetric
logistic regression (shape.fit_logistic, no intercept: swapping the teams flips the sign) on 21
home-minus-away features; tanh(logit / scale) enters the decoder as one more team emission on
visible frames, D-neutral (it says WHICH team, not that the ball is live: without that it lost on
the bench whose dead ball is visible, real1 -0.004..-0.029).

Why: on the external tune matches (ctl T=2 truth) the true possessions the decoder gives to the
other team start in a duel and the ball then travels with neither team near; no single rule reads
them (kick from the speed jump, receiver, Laws 15-17: T122-T124 rejected), but the per-frame model
alone is as often right as the whole decoder (0.931 vs 0.933 on real2, 0.954 vs 0.953 on real1)
and right on 0.43-0.45 of the frames where the decoder is wrong (scripts/a8b_ml.py probe).
Reported scores never use MODEL: they use leave-one-match-out fits (scripts/a8b_ml.py srcgate).

matchlens_start: footballtactics events/possession.py (control zone: a player within r_pz of the
ball, last-touch memory), events/impulse.py + events/kinematics.py (ball velocity change as a
touch), events/attacking_direction.py (-> shape.attack_dir). MatchLens weighs none of these
against a truth; possession.py is a fixed state machine. Kept: the 1 m control radius, the 1.5 m
duel radius, finite-difference velocity. New: the weights are fitted on external data.
"""
from typing import Dict, Optional

import numpy as np

FPS = 25.0
INF = 1e9
SCALE = 2.0
FEATS = ("g_now", "in1_now", "past13", "past25", "past50", "past100", "since_alone", "imp_last", "fly48", "fly816",
         "fly_past", "n3", "n6", "shape", "free_past50", "free_past100", "free_fly", "fast_past50", "xatt", "xatt_free",
         "xatt_fast")


def past_min(x: np.ndarray, w: int) -> np.ndarray:
    """Minimum of x over the w frames ending at each frame."""
    cur, s = np.asarray(x, float).copy(), 1
    while s < w:
        st = min(s, w - s)
        cur = np.minimum(cur, np.concatenate([np.full(st, INF), cur[:-st]]))
        s += st
    return cur


def since(mask: np.ndarray) -> np.ndarray:
    """Frames since the last True (INF before the first)."""
    k = np.arange(len(mask))
    idx = np.where(mask, k, -10**9)
    return np.minimum(k - np.maximum.accumulate(idx), INF)


def ball_vel(ball_x: np.ndarray, ball_y: np.ndarray, lag: int = 8, fps: float = FPS):
    """Centred finite-difference ball velocity over `lag` frames (nan where an end is hidden).
    Port of footballtactics events/kinematics.py finite differences; lag 8 = decode.SMOOTH_TOUCH lag."""
    n, h = len(ball_x), lag // 2
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    if n > lag:
        vx[h:n - lag + h] = (ball_x[lag:] - ball_x[:-lag]) * fps / lag
        vy[h:n - lag + h] = (ball_y[lag:] - ball_y[:-lag]) * fps / lag
    return vx, vy


def impulses(ball_x: np.ndarray, ball_y: np.ndarray, thr: float = 4.0, h: int = 6, lag: int = 8, fps: float = FPS) -> np.ndarray:
    """Frames where the ball's velocity changes by >= thr m/s between h frames before and h frames
    after (local maxima of the change, at least h frames apart). The speed-change test of
    footballtactics events/impulse.py:impulse_verdict without the player-distance condition."""
    vx, vy = ball_vel(ball_x, ball_y, lag, fps)
    n = len(ball_x)
    dv = np.zeros(n)
    if n > 2 * h:
        with np.errstate(invalid="ignore"):
            dv[h:n - h] = np.nan_to_num(np.hypot(vx[2 * h:] - vx[:-2 * h], vy[2 * h:] - vy[:-2 * h]))
    out = []
    for i in np.flatnonzero(dv >= thr):
        if dv[i] >= dv[max(0, i - h):i + h + 1].max() and (not out or i - out[-1] > h):
            out.append(int(i))
    return np.array(out, int)


def features(ball_x: np.ndarray, ball_y: np.ndarray, d_home: np.ndarray, d_away: np.ndarray, pos: np.ndarray,
             period: np.ndarray, shape_team: Optional[np.ndarray] = None, fps: float = FPS) -> np.ndarray:
    """[frames, len(FEATS)] home-minus-away features (0 where undefined). d_home / d_away = distance
    of each team's nearest player to the ball (99 where the ball is hidden); pos [frames, 2, players, 2]."""
    from .decode import flight_dir
    from .shape import attack_dir

    bx, by = np.asarray(ball_x, float), np.asarray(ball_y, float)
    dh, da = np.nan_to_num(np.asarray(d_home, float), nan=99.0), np.nan_to_num(np.asarray(d_away, float), nan=99.0)
    n = len(bx)
    g = lambda d: np.exp(-np.minimum(d, 50.0))   # noqa: E731
    c: Dict[str, np.ndarray] = {}
    c["g_now"] = g(dh) - g(da)
    c["in1_now"] = (dh < 1.0).astype(float) - (da < 1.0)
    for w in (13, 25, 50, 100):
        c["past%d" % w] = g(past_min(dh, w)) - g(past_min(da, w))
    c["since_alone"] = np.exp(-since((dh < 1.0) & (da >= 1.5)) / 50.0) - np.exp(-since((da < 1.0) & (dh >= 1.5)) / 50.0)
    imp = impulses(bx, by, 4.0, fps=fps)
    last = np.zeros(n)
    if len(imp):
        att = np.array([g(dh[max(0, i - 3):i + 4].min()) - g(da[max(0, i - 3):i + 4].min()) for i in imp])
        k = np.arange(n)
        j = np.searchsorted(imp, k, side="right") - 1          # last impulse at or before the frame
        jj = np.maximum(j, 0)
        last = np.where(j >= 0, att[jj] * np.exp(-(k - imp[jj]) / 50.0), 0.0)
    c["imp_last"] = last
    fly = flight_dir(bx, by, pos, period, 4.0, 8.0, fps)
    c["fly48"] = fly
    c["fly816"] = flight_dir(bx, by, pos, period, 8.0, 16.0, fps)
    sm = np.convolve(np.nan_to_num(fly), np.ones(25) / 25, mode="same")
    c["fly_past"] = np.concatenate([np.zeros(12), sm[:-12]]) if n > 12 else np.zeros(n)
    with np.errstate(invalid="ignore"):
        dist = np.hypot(pos[:, :, :, 0] - bx[:, None, None], pos[:, :, :, 1] - by[:, None, None])
        n3 = (dist < 3.0).sum(axis=2).astype(float)
        n6 = (dist < 6.0).sum(axis=2).astype(float)
    c["n3"] = n3[:, 0] - n3[:, 1]
    c["n6"] = n6[:, 0] - n6[:, 1]
    c["shape"] = np.zeros(n) if shape_team is None else np.asarray(shape_team, float)
    free = ((dh >= 1.5) & (da >= 1.5)).astype(float)
    vx, vy = ball_vel(bx, by, fps=fps)
    with np.errstate(invalid="ignore"):
        fast = np.nan_to_num((np.hypot(vx, vy) > 8.0).astype(float))
    c["free_past50"] = free * c["past50"]
    c["free_past100"] = free * c["past100"]
    c["free_fly"] = free * fly
    c["fast_past50"] = fast * c["past50"]
    xatt = np.nan_to_num(bx) * attack_dir(pos, np.asarray(period)).astype(float) / 52.5
    c["xatt"] = xatt
    c["xatt_free"] = xatt * free
    c["xatt_fast"] = xatt * fast
    return np.stack([np.nan_to_num(np.asarray(c[k], float)) for k in FEATS], axis=1)


def evidence(X: np.ndarray, visible: np.ndarray, model: Optional[Dict[str, np.ndarray]] = None, scale: float = SCALE) -> np.ndarray:
    """tanh(logit / scale) in [-1, 1], positive = home; 0 where the ball is hidden."""
    m = model or MODEL
    z = np.tanh((((X - m["mu"]) / m["sd"]) @ m["w"] + m.get("b", 0.0)) / scale)
    return np.where(np.asarray(visible, bool), z, 0.0)


def from_feats(feats: Dict[str, np.ndarray], period: np.ndarray, model: Optional[Dict[str, np.ndarray]] = None,
               fps: float = FPS) -> np.ndarray:
    """Evidence from a decoder `feats` dict (needs ball_x, ball_y, d_home, d_away, visible, pos)."""
    X = features(feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"], feats["pos"], period,
                 feats.get("shape_team"), fps)
    return evidence(X, feats["visible"], model)


# Fitted on the three external tune matches, each on the benches orig / real1 / real2 seed 0 (ctl T=2
# truth, visible live frames, every 3rd frame, l2 = 10): `scripts/a8b_ml.py fit`. Column order = FEATS.
# Largest standardised weights: shape +1.58, imp_last +1.40, g_now +1.39, past100 +0.57, fast_past50 +0.35.
MODEL: Dict[str, np.ndarray] = {
    "mu": np.zeros(21),
    "sd": np.array([0.52199, 0.6756, 0.60049, 0.64731, 0.67869, 0.66113, 0.75339, 0.40948, 0.49237, 0.22474, 0.43619, 0.8009,
                    0.98157, 2.8998, 0.39735, 0.40096, 0.43911, 0.38727, 0.46891, 0.299, 0.27437]),
    "w": np.array([1.3937, -0.11464, -0.23661, 0.20536, 0.20859, 0.56963, -0.068176, 1.4034, 0.21486, 0.20238, 0.10494,
                   -0.014131, -0.024657, 1.5824, 0.058786, 0.33933, 0.014422, 0.35347, 0.30442, -0.22346, 0.010223]),
    "b": 0.0,
}
