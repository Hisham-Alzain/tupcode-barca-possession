"""Dead-ball detection from ID-free signals (loop iteration 3).

matchlens_start: footballtactics events/deadball.py DeadBallDetector. Kept: combine several dead
signals with hysteresis (MatchLens's leaky bucket) and treat "ball missing" as no evidence (ghost
rule off, ADR-002). Its own D output is kept as one input. Added, from the swarm (F3, F4):
  * far     - distance from the ball to the nearest player of either team (m);
  * line    - distance from the ball to the nearest touch/goal line (m, negative = outside);
  * slow    - both teams' centroid speeds (m/s, 1 s median) are low: a collective stop;
  * bspeed  - ball speed (m/s, 1 s median).
A frame is dead-evidence when (far > r_far and (line < d_line or slow)) or ball outside the pitch
by more than `out_m`; play restarts when someone is within r_near of a moving ball. Frames with the
ball hidden carry no evidence and keep the current state.

Envelope L2-11 (restart timing, survey C2 / C4), both off by default in `detect`:
  * still_v  - a ball slower than still_v (m/s, `vlag`, see `lag_speed`) with no player within
               still_far is dead evidence. With the hysteresis back-dating this is MatchLens's
               STATIONARY rule (deadball.py: < 0.3 m/s for 2 s, nobody within 2 m) started at the
               frame the ball stopped, not 1.4 s later when MatchLens confirms it.
  * v_kick   - once dead, play resumes on a ball moving faster than v_kick (the restart kick or
               throw; IFAB Laws 13, 15-17: the ball is in play when it is kicked and clearly
               moves), not on a player standing within r_near of it (the taker waiting).
"""
from typing import Dict

import numpy as np

HALF_L, HALF_W = 52.5, 34.0


def rolling_median(v: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(v), np.nan)
    h = w // 2
    for i in range(len(v)):
        seg = v[max(0, i - h): i + h + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def short_median(v: np.ndarray, w: int = 5) -> np.ndarray:
    """Centred median over w frames, nan ignored (nan where the window is empty). Vectorised."""
    h = w // 2
    pad = np.concatenate([np.full(h, np.nan), np.asarray(v, float), np.full(h, np.nan)])
    win = np.stack([pad[i:i + len(v)] for i in range(w)], axis=1)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(win).any(axis=1)
    out[ok] = np.nanmedian(win[ok], axis=1)
    return out


def lag_speed(ball_x: np.ndarray, ball_y: np.ndarray, fps: float = 25, lag_s: float = 0.4) -> np.ndarray:
    """Ball speed (m/s) that survives tracking jitter and keeps sharp edges: displacement of the
    5-frame median position over `lag_s`, and per frame the smaller of the window that starts at
    it and the window that ends at it. A ball that stops at t0 is slow from t0 on, and a ball
    kicked at t1 is slow up to t1 (a centred speed would blur both by lag_s / 2). Frame-to-frame
    speed turns 0.05 m of jitter into ~1.8 m/s; this turns it into ~0.1 m/s. nan where neither
    window is fully visible."""
    k = max(1, int(round(lag_s * fps)))
    xs, ys = short_median(ball_x, 5), short_median(ball_y, 5)
    xs, ys = np.where(np.isnan(ball_x), np.nan, xs), np.where(np.isnan(ball_y), np.nan, ys)
    fwd = np.full(len(xs), np.nan)
    fwd[:-k] = np.hypot(xs[k:] - xs[:-k], ys[k:] - ys[:-k]) * fps / k
    bwd = np.full(len(xs), np.nan)
    bwd[k:] = fwd[:-k]
    return np.fmin(fwd, bwd)


def features(ball_x, ball_y, d_home, d_away, cen_speed_home, cen_speed_away, fps=25) -> Dict[str, np.ndarray]:
    far = np.fmin(d_home, d_away)
    line = np.fmin(HALF_L - np.abs(ball_x), HALF_W - np.abs(ball_y))
    bs = np.hypot(np.diff(ball_x, prepend=np.nan), np.diff(ball_y, prepend=np.nan)) * fps
    w = int(fps)
    return {"far": far, "line": line, "bspeed": rolling_median(bs, w), "vlag": lag_speed(ball_x, ball_y, fps),
            "team_speed": np.fmax(rolling_median(cen_speed_home, w), rolling_median(cen_speed_away, w))}


def detect(feat: Dict[str, np.ndarray], base_dead: np.ndarray, r_far=6.0, d_line=3.0, v_slow=1.0,
           out_m=1.0, r_near=2.0, on_frames=25, off_frames=10, use_base=True, still_v=0.0,
           still_far=2.0, v_kick=0.0) -> np.ndarray:
    """Boolean dead mask. Hysteresis: `on_frames` of dead evidence to stop play, `off_frames` of
    restart evidence to resume. MatchLens's own D (`base_dead`) also counts as dead evidence.

    still_v > 0: ball slower than still_v with nobody within still_far is dead evidence (needs
    feat["vlag"]). v_kick > 0: restart evidence is a ball faster than v_kick inside the pitch
    instead of a player within r_near (module docstring, envelope L2-11)."""
    far, line, slow = feat["far"], feat["line"], feat["team_speed"] < v_slow
    vis = ~np.isnan(far)
    ev_dead = vis & (((far > r_far) & ((line < d_line) | slow)) | (line < -out_m))
    if use_base:
        ev_dead |= base_dead
    if still_v:
        ev_dead |= vis & (feat["vlag"] < still_v) & (far > still_far)
    if v_kick:
        ev_play = vis & (feat["vlag"] > v_kick) & ~(line < -out_m) & ~base_dead & ~ev_dead
    else:
        ev_play = vis & (far < r_near) & ~(line < -out_m) & ~base_dead
    n = len(far)
    out = np.zeros(n, bool)
    state, run_d, run_p = False, 0, 0
    start = 0
    for i in range(n):
        run_d = run_d + 1 if ev_dead[i] else 0
        run_p = run_p + 1 if ev_play[i] else 0
        if not state and run_d >= on_frames:
            state, start = True, i - on_frames + 1
            out[start:i] = True          # back-date to when the evidence began
        elif state and run_p >= off_frames:
            state = False
            out[i - off_frames + 1:i] = False
        out[i] = state
    return out
