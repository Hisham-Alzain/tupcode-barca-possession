"""One pass over the frames -> per-frame arrays every later step needs.

Works on any frame source yielding `(frame_id, period, tracks, ball_present)` (possess.epts.AtdMatch,
possess.metrica_csv.MetricaMatch, scripts/idsse_cache.IdsseMatch). Arrays are indexed by frame_id.

  d_home, d_away   distance (m) from the ball to the nearest home / away player (nan: no ball)
  cen_home, cen_away   team centroid speed (m/s), median position of >= 6 players (count-robust)
  ml_dead          MatchLens dead-ball state (possess.stoppage_ml, ghost rule off)
  ball_x, ball_y   the ball actually used (nan where missing)
  pos              (positions=True only) anonymous player positions for possess.shape
"""
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .stoppage_ml import MatchLensStoppage

FPS = 25.0


def compute(frames: Iterable[Tuple[int, int, dict, bool]], n: int, ball_x: Optional[np.ndarray] = None,
            ball_y: Optional[np.ndarray] = None, fps: float = FPS, positions: bool = False) -> Dict[str, np.ndarray]:
    """`ball_x/ball_y` override the frames' own ball (the bench passes its degraded ball).

    positions=True adds "pos" [n, 2, shape.SLOTS, 2] (anonymous player positions, the input of
    possess.shape) in the same pass; off by default so existing feature caches keep their keys.
    """
    if positions:
        from .shape import SLOTS

        pos_arr = np.full((n, 2, SLOTS, 2), np.nan, dtype=np.float32)
    own_ball = ball_x is None
    bx = np.full(n, np.nan) if own_ball else ball_x
    by = np.full(n, np.nan) if own_ball else ball_y
    dh, da = np.full(n, np.nan), np.full(n, np.nan)
    cx = {1: np.full(n, np.nan), 2: np.full(n, np.nan)}
    cy = {1: np.full(n, np.nan), 2: np.full(n, np.nan)}
    ml_dead = np.zeros(n, bool)
    stop = MatchLensStoppage(fps)
    last_period = None
    for fid, per, tracks, present in frames:
        if fid >= n:
            continue
        if per != last_period:                  # a new half: MatchLens resets its engine state
            stop = MatchLensStoppage(fps)
            last_period = per
        if own_ball and present:
            bx[fid], by[fid] = tracks["ball"][1]["map_position"]
        pos = None if np.isnan(bx[fid]) else (float(bx[fid]), float(by[fid]))
        players = {}
        pts = {1: [], 2: []}
        best = {1: np.inf, 2: np.inf}
        for grp in ("players", "goalkeepers"):
            for k, r in tracks[grp].items():
                players[k] = r
                t = r.get("team")
                if t in pts:
                    p = r["map_position"]
                    if positions and len(pts[t]) < SLOTS:
                        pos_arr[fid, t - 1, len(pts[t])] = p
                    pts[t].append(p)
                    if pos is not None:
                        best[t] = min(best[t], (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2)
        if pos is not None:
            dh[fid], da[fid] = np.sqrt(best[1]), np.sqrt(best[2])
        for t in (1, 2):
            if len(pts[t]) >= 6:
                a = np.asarray(pts[t])
                cx[t][fid], cy[t][fid] = np.median(a[:, 0]), np.median(a[:, 1])
        ml_dead[fid] = stop.update(fid, pos, players)
    sp = {t: np.hypot(np.diff(cx[t], prepend=np.nan), np.diff(cy[t], prepend=np.nan)) * fps for t in (1, 2)}
    out = {"ball_x": bx, "ball_y": by, "d_home": dh, "d_away": da,
           "cen_home": sp[1], "cen_away": sp[2], "ml_dead": ml_dead}
    if positions:
        out["pos"] = pos_arr
    return out
