"""Learned ball-aware stoppage evidence on visible frames (envelope L8-autonomous-fable round 6, T137;
opt-in via decode.decode_halves w_bd, 0 = off).

Is the visible ball dead? One logistic regression (shape.fit_logistic, with intercept) on 21 features
that do not know the teams: the ball-free stoppage log-odds (shape_dead), the ball's speed and how long
it has been slow, the distance to the nearest line and whether it is beyond one, the nearest player and
whether the ball moves with him, the fast players' speed, time since the last impulse / touch / hidden
frame. tanh((logit - base) / 2), base = the training share of D as log-odds, is added to the D-vs-live
log-odds of visible frames (through shape_dead, i.e. next to the detector and the ball-free model that
decode.viterbi already weighs there); hidden frames are not touched.

Why: with T125 in, the true possessions swallowed by a long predicted spell are the largest unmatched
class, and the stoppage most often swallowed with them lasts 5-7 s with the ball visible on 60 % of
it (scripts/a8b_r6.py diag). The decoder's visible-frame stoppage evidence is a hand-set sum (detector
+-1, ball-free model x 0.75); the learned combination is right on 0.926 of real1's visible frames
against 0.914 (the decoder), and equal on real2 / orig.
Reported scores never use MODEL: they use leave-one-match-out fits (scripts/a8b_r6.py dead).

matchlens_start: footballtactics events/deadball.py (ball out of play: beyond a line; stopped ball:
speed under a threshold for a hold time; restart on a kick), events/kinematics.py (finite-difference
velocity), events/possession.py (1 m control zone). MatchLens applies these as fixed thresholds in a
state machine; kept: the out-of-pitch test, the slow-ball idea, the 1 m / 1.5 m / 3 m radii. New: the
weights are fitted on external data and the ball-free stoppage model enters as a feature.
"""
from typing import Dict, Optional

import numpy as np

from .ballteam import ball_vel, impulses, since

FPS = 25.0
SCALE = 2.0
HALF_LEN, HALF_WID = 52.5, 34.0
FEATS = ("shape_dead", "speed", "slow", "still_time", "dmin", "alone3", "line_dist", "outside", "carried", "carried_slow",
         "p90", "goal_area_slow", "e_dmin", "e_dmax", "e_speed", "since_imp", "free", "duel", "fast", "since_hidden",
         "since_touch")


def _coupling(pos_team: np.ndarray, bx: np.ndarray, by: np.ndarray, vx: np.ndarray, vy: np.ndarray, k: int = 5,
              fps: float = FPS) -> np.ndarray:
    """exp(-|v_ball - v_player| / 3) for the team's player nearest to the ball when he is within 3 m, else 0.
    No player identity is needed: his position k frames earlier is the nearest team-mate position then."""
    n = len(bx)
    P = np.asarray(pos_team, float)
    with np.errstate(invalid="ignore"):
        rx, ry = P[:, :, 0] - bx[:, None], P[:, :, 1] - by[:, None]
        dist = np.where(np.isnan(rx) | np.isnan(ry), 1e9, np.hypot(rx, ry))
    i0 = dist.argmin(axis=1)
    px, py = P[np.arange(n), i0, 0], P[np.arange(n), i0, 1]
    qx, qy = np.full(n, np.nan), np.full(n, np.nan)
    if n > k:
        with np.errstate(invalid="ignore"):
            d2 = np.hypot(P[:-k, :, 0] - px[k:, None], P[:-k, :, 1] - py[k:, None])
        j = np.where(np.isnan(d2), 1e9, d2).argmin(axis=1)
        qx[k:], qy[k:] = P[np.arange(n - k), j, 0], P[np.arange(n - k), j, 1]
    with np.errstate(invalid="ignore"):
        c = np.exp(-np.hypot(vx - (px - qx) * fps / k, vy - (py - qy) * fps / k) / 3.0) * (dist.min(axis=1) < 3.0)
    return np.nan_to_num(c)


def features(ball_x: np.ndarray, ball_y: np.ndarray, d_home: np.ndarray, d_away: np.ndarray, visible: np.ndarray,
             pos: np.ndarray, shape_dead: Optional[np.ndarray] = None, p90: Optional[np.ndarray] = None,
             fps: float = FPS) -> np.ndarray:
    """[frames, len(FEATS)] (0 where undefined). d_home / d_away = distance of each team's nearest player
    to the ball; pos [frames, 2, players, 2]; shape_dead = ball-free stoppage log-odds; p90 = raw speed of
    the fast players (shape.decoder_inputs edge_p90)."""
    bx, by = np.asarray(ball_x, float), np.asarray(ball_y, float)
    dh, da = np.nan_to_num(np.asarray(d_home, float), nan=99.0), np.nan_to_num(np.asarray(d_away, float), nan=99.0)
    vis = np.asarray(visible, bool)
    n = len(bx)
    vx, vy = ball_vel(bx, by, fps=fps)
    with np.errstate(invalid="ignore"):
        sp = np.nan_to_num(np.hypot(vx, vy))
        line = np.nan_to_num(np.minimum(HALF_LEN - np.abs(bx), HALF_WID - np.abs(by)), nan=10.0)
        ga = np.nan_to_num((np.abs(bx) > 46.0).astype(float))
    dmin, dmax = np.minimum(dh, da), np.maximum(dh, da)
    im = np.zeros(n, bool)
    im[impulses(bx, by, 4.0, fps=fps)] = True
    carried = np.abs(_coupling(pos[:, 0], bx, by, vx, vy, fps=fps) - _coupling(pos[:, 1], bx, by, vx, vy, fps=fps))
    c: Dict[str, np.ndarray] = {
        "shape_dead": np.zeros(n) if shape_dead is None else np.nan_to_num(np.asarray(shape_dead, float)),
        "speed": np.minimum(sp, 25.0) / 25.0,
        "slow": (sp < 1.0).astype(float),
        "still_time": 1.0 - np.exp(-since(sp > 2.0) / 75.0),
        "dmin": np.minimum(dmin, 10.0) / 10.0,
        "alone3": (dmin > 3.0).astype(float),
        "line_dist": np.clip(line, 0.0, 10.0) / 10.0,
        "outside": (line < 0).astype(float),
        "carried": carried,
        "carried_slow": carried * (sp < 3.0),
        "p90": np.zeros(n) if p90 is None else np.minimum(np.nan_to_num(np.asarray(p90, float)), 8.0) / 8.0,
        "goal_area_slow": ga * (sp < 1.0),
        "e_dmin": np.minimum(dmin, 10.0) / 10.0,
        "e_dmax": np.minimum(dmax, 10.0) / 10.0,
        "e_speed": np.minimum(sp, 25.0) / 25.0,
        "since_imp": np.exp(-since(im) / 50.0),
        "free": ((dh >= 1.5) & (da >= 1.5)).astype(float),
        "duel": ((dh < 1.5) & (da < 1.5)).astype(float),
        "fast": (sp > 8.0).astype(float),
        "since_hidden": np.exp(-since(~vis) / 50.0),
        "since_touch": np.exp(-since(dmin < 1.0) / 50.0),
    }
    return np.stack([np.nan_to_num(np.asarray(c[k], float)) for k in FEATS], axis=1)


def evidence(X: np.ndarray, visible: np.ndarray, model: Optional[Dict[str, np.ndarray]] = None, scale: float = SCALE) -> np.ndarray:
    """tanh((logit - base) / scale) in [-1, 1], positive = dead; 0 where the ball is hidden."""
    m = model or MODEL
    lo = ((X - m["mu"]) / m["sd"]) @ m["w"] + float(m["b"])
    return np.where(np.asarray(visible, bool), np.tanh((lo - float(m["base"])) / scale), 0.0)


def from_feats(feats: Dict[str, np.ndarray], p90: Optional[np.ndarray] = None, model: Optional[Dict[str, np.ndarray]] = None,
               fps: float = FPS) -> np.ndarray:
    """Evidence from a decoder `feats` dict (needs ball_x, ball_y, d_home, d_away, visible, pos, shape_dead)."""
    X = features(feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"], feats["visible"], feats["pos"],
                 feats.get("shape_dead"), p90, fps)
    return evidence(X, feats["visible"], model)


# Fitted on the three external tune matches, each on the benches orig / real1 / real2 seed 0 (ctl T=2 truth,
# visible frames, every 3rd frame, l2 = 10): `scripts/a8b_r6.py deadfit`. Column order = FEATS. Largest
# standardised weights: shape_dead +1.69, carried -1.40, since_touch -0.89, outside +0.80, carried_slow +0.74,
# since_imp +0.65, slow +0.59, p90 -0.56.
MODEL: Dict[str, np.ndarray] = {
    "mu": np.array([-1.1861, 0.23233, 0.21557, 0.17633, 0.26968, 0.35405, 0.69865, 0.1337, 0.29396, 0.17248, 0.37627,
                    0.069808, 0.26968, 0.64886, 0.23233, 0.52333, 0.53996, 0.046789, 0.29096, 0.071351, 0.64622]),
    "sd": np.array([2.6792, 0.20997, 0.41122, 0.34667, 0.27765, 0.47823, 0.39325, 0.34033, 0.36609, 0.33241, 0.17172,
                    0.25482, 0.27765, 0.30227, 0.20997, 0.33833, 0.4984, 0.21119, 0.4542, 0.1772, 0.38726]),
    "w": np.array([1.6871, -0.14876, 0.58653, -0.02876, -0.0094981, -0.45617, -0.28003, 0.7973, -1.3997, 0.74044,
                   -0.56249, -0.035614, -0.0094981, 0.048502, -0.14876, 0.65425, 0.14042, -0.00098783, -0.28362,
                   -0.061285, -0.8934]),
    "b": -2.1137,
    "base": -0.81419,
}
