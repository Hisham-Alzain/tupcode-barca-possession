"""Edges of predicted stoppages (D runs) inside hidden-ball blocks (envelope L2-18-stop-edges-fable).

The decoder decides WHETHER a stoppage exists; with constant transition costs its edges sit where
the stoppage log-odds cross zero, and those log-odds are built from centred 1-15 s windows. This
module is a post-pass on finished labels that only MOVES an existing D edge: it never writes X,
never creates or deletes a run, works per period, and reads player columns only (no ball).

matchlens_start: stoppage_ml.py DeadBallDetector._commit_flip (port of MatchLens deadball.py):
a committed flip is BACK-STAMPED to the candidacy anchor, the first supporting frame, so the dead
interval starts when the transition began, not when it was confirmed. `snap_edges` (M1) is the
offline, player-only form of that idea for both edges: the edge goes to the step of the fast
players' speed inside a search window. Anchors kept from that file: STATIONARY_DURATION_S 2.0 (the
search half-window `ed_win`), STATIONARY_DISP_WINDOW_S 1.0 (the step window `ed_w`). `shift_edges`
(M2) is a constant per-edge shift. MatchLens detectors/set_piece_detector.py and deadball.py read
the ball and cannot be used on hidden frames.

All parameters off (the default) returns the input unchanged. Swapping home and away swaps the
output (the columns are pooled over both teams and the labels are only copied from neighbours).
"""
from typing import Dict, Optional

import numpy as np

FPS = 25.0
FAST_Q = 0.9             # as shape.FAST_Q: the fast tail of the player speeds
FAST_V = 2.0             # m/s; a player faster than this is "running" (count column, diagnosis only)
KEYS = ("ed_exit", "ed_entry", "ed_w", "ed_win", "ed_win_e", "ed_theta", "ed_vis", "ed_min_s", "ed_dx", "ed_de")


def active(kw: dict) -> bool:
    return any(kw.get(k, 0) for k in ("ed_exit", "ed_entry", "ed_dx", "ed_de"))


def edge_columns(pos: np.ndarray, fps: float = FPS) -> Dict[str, np.ndarray]:
    """{"p90": raw 90th percentile of the speeds of all tracked players (both teams, no 1 s / 5 s
    smoothing; the speed itself is shape._player_speed's centred 0.8 s displacement), "nfast":
    players faster than FAST_V}. nan / 0 where nobody is tracked."""
    from . import shape

    sp = np.concatenate([shape._player_speed(pos[:, t], fps, each=True)[1] for t in (0, 1)], axis=1)
    with np.errstate(invalid="ignore"):
        nfast = (sp > FAST_V).sum(axis=1).astype(np.float32)
    return {"p90": shape._row_quantile(sp, FAST_Q).astype(np.float32), "nfast": nfast}


def taker_frames(pos: np.ndarray, fps: float = FPS, band: float = 1.5, corner: float = 2.0,
                 v_slow: float = 1.0, hold_s: float = 1.0, r_free: float = 2.0, chunk: int = 20000) -> np.ndarray:
    """M5 cue (Laws 15 / 17, geometry only, no identity): True on frames where, for the last
    `hold_s` seconds without a break, some player stood within `band` m of a touchline (or beyond
    it) or within `corner` m of a corner, slower than `v_slow`, with no opponent within `r_free`.
    Anchors (stoppage_ml.py, port of MatchLens deadball.py): hold_s = STATIONARY_DISP_WINDOW_S 1.0,
    r_free = NO_PLAYER_RADIUS_M 2.0 (= the opponents' distance of Law 15). Goal kicks are left out
    (a goalkeeper always stands in his goal area). MatchLens detectors/set_piece_detector.py was
    re-checked for a taker / wall formation test: none in the ported material (stoppage_ml.py
    PlacedBallDetector is its ball-based placed-ball rule: a still ball with one player next to
    it); nothing there works without the ball."""
    from . import shape

    n = len(pos)
    h = max(1, int(round(shape.SPEED_LAG_S * fps / 2)))       # _player_speed indexes slots at t + h
    flag = np.zeros(n, bool)
    for t in (0, 1):
        sp = shape._player_speed(pos[:, t], fps, each=True)[1]
        for s in range(0, n - h, chunk):
            e = min(n - h, s + chunk)
            p, o = pos[s + h:e + h, t], pos[s + h:e + h, 1 - t]
            with np.errstate(invalid="ignore"):
                dy = shape.HALF_W - np.abs(p[:, :, 1])
                cand = ((dy <= band) | (np.hypot(shape.HALF_L - np.abs(p[:, :, 0]), dy) <= corner)) & (sp[s:e] < v_slow)
                d = np.hypot(p[:, :, None, 0] - o[:, None, :, 0], p[:, :, None, 1] - o[:, None, :, 1])
                free = ~(d <= r_free).any(axis=2)
            flag[s:e] |= (cand & free).any(axis=1)
    k = max(1, int(round(hold_s * fps)))
    c = np.concatenate([[0], np.cumsum(flag)])
    out = np.zeros(n, bool)
    out[k - 1:] = (c[k:] - c[:-k]) == k
    return out


def step_score(v: np.ndarray, period: np.ndarray, w: int) -> np.ndarray:
    """s[t] = mean(v[t : t + w]) - mean(v[t - w : t]): positive where v steps up at frame t.
    nan where a window leaves the array, crosses a period break or has under half its frames."""
    trail, lead = side_means(v, period, w)
    return lead - trail


def side_means(v: np.ndarray, period: np.ndarray, w: int):
    """(trailing mean of v[t - w : t], leading mean of v[t : t + w]); nan as in step_score."""
    n = len(v)
    ok = np.isfinite(v)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    t = np.arange(n)
    lo, hi = t - w, t + w
    good = (lo >= 0) & (hi <= n)
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    period = np.asarray(period)
    good &= (period[lo] == period) & (period[np.clip(hi - 1, 0, n - 1)] == period) & (period > 0)
    n_lead, n_trail = k[hi] - k[t], k[t] - k[lo]
    good &= (n_lead * 2 >= w) & (n_trail * 2 >= w)
    with np.errstate(invalid="ignore", divide="ignore"):
        lead, trail = (c[hi] - c[t]) / n_lead, (c[t] - c[lo]) / n_trail
    return np.where(good, trail, np.nan), np.where(good, lead, np.nan)


def _segments(lab: np.ndarray, period: np.ndarray):
    """[(a, b)] maximal runs of one label inside one period (b exclusive)."""
    n = len(lab)
    cut = np.flatnonzero((lab[1:] != lab[:-1]) | (np.asarray(period)[1:] != np.asarray(period)[:-1])) + 1
    return list(zip(np.concatenate([[0], cut]).tolist(), np.concatenate([cut, [n]]).tolist()))


def _move(lab, period, visible, fps, min_s, vis_too, do_exit, do_entry, target, moves):
    """Shared walk over D runs >= min_s. target(kind, e, lo, hi) -> new edge frame in [lo, hi] or
    None. Edge frame of an exit = first frame after the run, of an entry = first frame of the run.
    Guards: an edge stays in its period, never passes the middle of the D run or of the
    neighbouring run, and is only moved against a team neighbour (H / A) of the same period."""
    out = lab.copy()
    segs = _segments(lab, period)
    n = len(lab)
    for i, (a, b) in enumerate(segs):
        if lab[a] != "D" or period[a] <= 0 or b - a < min_s * fps:
            continue
        mid = (a + b) // 2
        if do_exit and i + 1 < len(segs):
            na, nb = segs[i + 1]
            if period[na] == period[a] and lab[na] in ("H", "A") and (vis_too or not visible[min(b, n - 1)]):
                lo, hi = mid + 1, max(mid + 1, (na + nb) // 2)
                e = target("exit", b, lo, hi)
                if e is not None and lo <= e <= hi and e != b:
                    out[min(e, b):max(e, b)] = "D" if e > b else lab[na]
                    moves.append(("exit", b, e))
        if do_entry and i > 0:
            pa, pb = segs[i - 1]
            if period[pa] == period[a] and lab[pa] in ("H", "A") and (vis_too or not visible[a]):
                lo, hi = min(mid, (pa + pb) // 2 + 1), mid
                e = target("entry", a, lo, hi)
                if e is not None and lo <= e <= hi and e != a:
                    out[min(e, a):max(e, a)] = "D" if e < a else lab[pa]
                    moves.append(("entry", a, e))
    return out


def snap_edges(lab: np.ndarray, period: np.ndarray, visible: np.ndarray, p90: np.ndarray,
               ed_exit: int = 0, ed_entry: int = 0, ed_w: float = 1.0, ed_win: float = 2.0,
               ed_theta: float = 0.0, ed_vis: int = 0, ed_min_s: float = 3.0, fps: float = FPS,
               moves: Optional[list] = None, ed_win_e: float = 0.0) -> np.ndarray:
    """M1. Every predicted D run >= ed_min_s whose exit frame is hidden (ed_vis=1: any) gets its
    exit moved to the largest step UP of p90 (step_score, window ed_w s) within +-ed_win s, if
    that step is >= ed_theta m/s and > 0; its entry to the largest step DOWN (within +-ed_win_e s
    when that is set). Frames D gives up
    take the neighbouring team's label, frames gained become D. moves: list filled with
    (kind, old frame, new frame)."""
    moves = [] if moves is None else moves
    if not (ed_exit or ed_entry):
        return lab
    s = step_score(np.asarray(p90, float), period, max(1, int(round(ed_w * fps))))
    wins = {"exit": int(round(ed_win * fps)), "entry": int(round((ed_win_e or ed_win) * fps))}

    def target(kind, e, lo, hi):
        lo, hi = max(lo, e - wins[kind]), min(hi, e + wins[kind], len(s) - 1)
        if hi < lo:
            return None
        seg = s[lo:hi + 1] if kind == "exit" else -s[lo:hi + 1]
        if not np.isfinite(seg).any():
            return None
        j = int(np.nanargmax(seg))
        return lo + j if seg[j] > 0 and seg[j] >= ed_theta else None

    return _move(lab, period, visible, fps, ed_min_s, ed_vis, ed_exit, ed_entry, target, moves)


def shift_edges(lab: np.ndarray, period: np.ndarray, visible: np.ndarray, ed_dx: float = 0.0,
                ed_de: float = 0.0, ed_vis: int = 0, ed_min_s: float = 3.0, fps: float = FPS,
                moves: Optional[list] = None) -> np.ndarray:
    """M2. Hidden exits of D runs >= ed_min_s move by ed_dx seconds, hidden entries by ed_de
    (positive = later), clipped by the same guards as snap_edges."""
    moves = [] if moves is None else moves
    if not (ed_dx or ed_de):
        return lab
    d = {"exit": int(round(ed_dx * fps)), "entry": int(round(ed_de * fps))}

    def target(kind, e, lo, hi):
        return int(np.clip(e + d[kind], lo, hi)) if d[kind] else None

    return _move(lab, period, visible, fps, ed_min_s, ed_vis, bool(d["exit"]), bool(d["entry"]), target, moves)


def apply(lab: np.ndarray, period: np.ndarray, visible: np.ndarray, p90: Optional[np.ndarray] = None,
          fps: float = FPS, moves: Optional[list] = None, **kw) -> np.ndarray:
    """snap_edges (needs p90) then shift_edges, parameters as in KEYS. Nothing set: `lab` itself."""
    if not active(kw):
        return lab
    com = {k: kw[k] for k in ("ed_vis", "ed_min_s") if k in kw}
    if (kw.get("ed_exit") or kw.get("ed_entry")) and p90 is not None:
        lab = snap_edges(lab, period, visible, p90, fps=fps, moves=moves,
                         **{k: kw[k] for k in ("ed_exit", "ed_entry", "ed_w", "ed_win", "ed_win_e", "ed_theta") if k in kw}, **com)
    if kw.get("ed_dx") or kw.get("ed_de"):
        lab = shift_edges(lab, period, visible, kw.get("ed_dx", 0.0), kw.get("ed_de", 0.0), fps=fps, moves=moves, **com)
    return lab


def cross_entry(lab: np.ndarray, period: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray,
                back_s: float = 3.0, off: int = 5, fps: float = FPS, moves: Optional[list] = None,
                strict: int = 0) -> np.ndarray:
    """Stoppage entry at the line crossing (envelope L8 round 3, opt-in via decode_halves ed_cross).

    Law 9: the ball is out of play when it has wholly crossed a boundary line. For each predicted
    stoppage that follows a team spell, the nearest frame within `back_s` seconds of its first
    frame where the visible ball goes from inside the pitch to beyond a line (deadball2.HALF_L /
    HALF_W, the pitch all inputs are mapped to) dates the entry: the stoppage starts `off` frames
    after it. The spell before it is cut or extended to that frame; it is never removed (one frame
    at least stays) and the stoppage keeps one frame at least. A stoppage with no such crossing
    (foul, ball hidden before the line) is untouched.
    strict=1: an entry is moved EARLIER only when the ball is not seen inside the pitch again
    between the crossing and our own entry (a ball that came back in was in play again, so that
    crossing belongs to an earlier stoppage).
    Measured on the external tune matches (scripts/a8b_r3.py entry; ctl T=2 truth): the crossing
    sits 0.20 s (real2) / 0.16 s (real1) before the true stoppage start, quartiles -0.28 / -0.08 s
    on both, against -0.28 / +0.74 s and -0.24 / +1.00 s for the decoder's own entry; hence
    `off` = 5 frames (the centre of the ball crosses before the whole ball does).
    matchlens_start: footballtactics events/deadball.py (ported in stoppage_ml.py / deadball2.py):
    the ball's distance to the nearest line is dead-ball EVIDENCE there (d_line), smoothed by
    hysteresis; nothing in MatchLens dates the stoppage at the crossing itself."""
    from .deadball2 import HALF_L, HALF_W

    lab = np.asarray(lab)
    out = lab.copy()
    n, back = len(lab), int(back_s * fps)
    with np.errstate(invalid="ignore"):
        ld = np.minimum(HALF_L - np.abs(ball_x), HALF_W - np.abs(ball_y))      # nan where the ball is hidden
        cross = np.flatnonzero((ld[:-1] >= 0) & (ld[1:] < 0)) + 1          # first frame beyond the line
    if not len(cross):
        return out
    for a, b in _segments(lab, period):
        if lab[a] != "D" or a == 0 or period[a - 1] != period[a] or lab[a - 1] not in ("H", "A"):
            continue
        j = np.searchsorted(cross, a)
        cand = [c for c in cross[max(0, j - 1):j + 1] if abs(c - a) <= back and period[c] == period[a]
                and not (strict and c < a and (ld[c:a] >= 0).any())]
        if not cand:
            continue
        cf = min(cand, key=lambda c: (abs(c - a), c))
        a0 = a - 1                                       # start of the team spell before the stoppage
        while a0 > 0 and lab[a0 - 1] == lab[a - 1] and period[a0 - 1] == period[a]:
            a0 -= 1
        ne = min(max(cf + off, a0 + 1), b - 1)
        if ne < a:
            out[ne:a] = "D"
        elif ne > a:
            out[a:ne] = lab[a - 1]
        if moves is not None and ne != a:
            moves.append(("cross", int(a), int(ne)))
    return out


def cross_dead(lab: np.ndarray, period: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray,
               margin: float = 0.5, off: int = 5, pre_s: float = 1.0, post_s: float = 3.0,
               max_s: float = 30.0, fps: float = FPS, moves: Optional[list] = None, min_s: float = 0.0) -> np.ndarray:
    """Stoppage at a line crossing the decoder missed (envelope L8 round 3, opt-in via decode_halves
    ed_xdead). Law 9 again: a visible ball that goes from inside the pitch to more than `margin`
    metres beyond a line is out of play. Where no predicted stoppage lies within -pre_s..+post_s
    seconds of such a crossing, one starts `off` frames after it and lasts until the ball is seen
    inside the pitch again (hidden frames in between stay dead), at most `max_s` seconds and never
    past the period. A stoppage that would last less than `min_s` seconds is not added (a ball
    back inside within 2 s is a touch on the line or tracking noise more often than a restart:
    scripts/a8b_r3.py xdrun, real1 -0.0026 -> +0.0027 with min_s = 2, real2 unchanged).
    Measured on the external tune matches (scripts/a8b_r3.py crossmiss, ctl T=2 truth, labels
    T95 + T100): 24 (real2) / 80 (real1) crossings have no stoppage of ours nearby; with the 0.5 m
    margin 18 of 18 / 58 of 60 are true stoppages (median 4 s). The ball is seen inside again at
    the true restart on real2 (median +0.0 s) but 2 s before it on real1 (its dead ball moves).
    matchlens_start: footballtactics events/deadball.py (stoppage_ml.py / deadball2.py): distance
    to the line is one dead-ball evidence among others, behind hysteresis and the decoder's
    stoppage cost; no rule there makes a crossing a stoppage by itself."""
    from .deadball2 import HALF_L, HALF_W

    lab = np.asarray(lab)
    out = lab.copy()
    n = len(lab)
    period = np.asarray(period)
    with np.errstate(invalid="ignore"):
        ld = np.minimum(HALF_L - np.abs(ball_x), HALF_W - np.abs(ball_y))
        cross = np.flatnonzero((ld[:-1] >= -margin) & (ld[1:] < -margin) & (period[1:] > 0)) + 1
    for cf in cross:
        if (out[max(0, cf - int(pre_s * fps)):min(n, cf + int(post_s * fps))] == "D").any():
            continue
        k = cf
        while k < n and period[k] == period[cf] and k - cf < max_s * fps and not ld[k] >= 0:
            k += 1
        a = min(cf + off, k)
        if k - a < max(1.0, min_s * fps):
            continue
        out[a:k] = "D"
        if moves is not None:
            moves.append(("xdead", int(a), int(k)))
    return out


def _speed(ball_x: np.ndarray, ball_y: np.ndarray, lag: int, fps: float) -> np.ndarray:
    n, h = len(ball_x), lag // 2
    s = np.full(n, np.nan)
    if n > lag:
        s[h:n - lag + h] = np.hypot(ball_x[lag:] - ball_x[:-lag], ball_y[lag:] - ball_y[:-lag]) * fps / lag
    return s


def restart_dead(lab: np.ndarray, period: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray, v_k: float = 5.0,
                 line: float = 0.0, margin: float = 0.5, off: int = 5, v_set: float = 2.0, set_s: float = 0.4,
                 spot: float = 0.5, max_s: float = 20.0, lag: int = 8, fps: float = FPS, reapp: bool = False, law: bool = False,
                 moves: Optional[list] = None) -> np.ndarray:
    """Out of play until the restart kick (envelope L11-A, opt-in via decode_halves ed_rkick = v_k).
    Law 9 / Laws 15-17: after the ball leaves the pitch, play restarts only when the ball is thrown or
    kicked from a restart spot. Our stoppages ended when the ball was seen inside again (cross_dead) or
    when the decoder's ball-free evidence turned, so a dead ball carried or rolled back in and placed
    (goal kick in the goal area, ball fetched for a throw-in) was labelled as the team that fetched it.
    For each visible crossing more than `margin` m beyond a line (and, with `line` > 0, for each end of a
    predicted stoppage where the visible ball lies within `line` m of a line), the frames from `off`
    frames after the crossing (from the stoppage end) up to the restart become D. Restart = the first
    visible frame moving at `v_k` m/s or more after the ball has settled (slower than `v_set` m/s for
    `set_s` s) on a restart spot (within `spot` m of a line or beyond it, a corner, a goal area), or after a hidden
    stretch (a ball back in view away from a restart spot ends it there). No restart within `max_s` s:
    nothing changes. Speed = displacement over `lag` frames as in
    still_dead. Constants: v_k 5 m/s from T116 (kick_exit), the rest a-priori geometry; chosen on the
    external benches only (scripts/a10_vdead.py bench).
    matchlens_start: footballtactics events/deadball.py (ported as deadball2: v_kick restart
    detection, line distance as dead-ball evidence) and edges.cross_dead; MatchLens ends a stoppage at
    the first kick of the ball at rest but never ties it to the ball having left the pitch."""
    from .deadball2 import HALF_L, HALF_W

    lab = np.asarray(lab)
    out = lab.copy()
    n = len(lab)
    period = np.asarray(period)
    bx, by = np.asarray(ball_x, float), np.asarray(ball_y, float)
    hid = np.isnan(bx)
    s = _speed(bx, by, lag, fps)
    with np.errstate(invalid="ignore"):
        ld = np.minimum(HALF_L - np.abs(bx), HALF_W - np.abs(by))
        # restart spots: on / beyond a line (throw-in, ball fetched), corner quadrant, goal area (+ 1 m)
        at_spot = ((ld < spot) | ((np.abs(bx) > HALF_L - spot - 1.0) & (np.abs(by) > HALF_W - spot - 1.0))
                   | ((np.abs(bx) > HALF_L - 6.5) & (np.abs(by) < 10.2)))
        starts = [int(c) + off for c in np.flatnonzero((ld[:-1] >= -margin) & (ld[1:] < -margin) & (period[1:] > 0)) + 1]
        if reapp:       # the crossing itself was hidden: the ball comes back into view beyond a line
            starts += [int(c) for c in np.flatnonzero(hid[:-1] & (ld[1:] < -margin) & (period[1:] > 0)) + 1]
    if line > 0:
        isd = lab == "D"
        for e in np.flatnonzero(isd[:-1] & ~isd[1:] & (period[1:] > 0)) + 1:
            if period[e] == period[e - 1] and not hid[e] and ld[e] < line:
                starts.append(int(e))
    nset = max(1, int(round(set_s * fps)))
    with np.errstate(invalid="ignore"):
        sp_goal = (ld >= -0.5) & (((np.abs(bx) > HALF_L - 6.5) & (np.abs(by) < 10.2))       # goal area (Law 16)
                   | ((np.abs(bx) > HALF_L - spot - 1.0) & (np.abs(by) > HALF_W - spot - 1.0)))  # corner (Law 17)
        sp_touch = np.abs(by) > HALF_W - spot                                                  # throw-in (Law 15)
    for a in sorted(set(starts)):
        if a >= n or period[a] <= 0:
            continue
        if law:     # the line the ball left decides where the restart can be taken
            vv = np.flatnonzero(~hid[max(0, a - off - 1):a + 1])
            j = max(0, a - off - 1) + (vv[-1] if len(vv) else 0)
            goal_line = (np.abs(bx[j]) - HALF_L) > (np.abs(by[j]) - HALF_W) if not hid[j] else False
            at_spot = sp_goal if goal_line else sp_touch
        k, cnt, ok, found = a, 0, False, False
        while k < n and period[k] == period[a] and k - a < max_s * fps:
            if hid[k]:
                ok, cnt = True, 0
            elif k > a and hid[k - 1] and not at_spot[k]:
                found = True            # back in view away from a restart spot: the restart was hidden
                break
            else:
                sk = s[k] if np.isfinite(s[k]) else 0.0
                if sk < v_set:
                    cnt += 1
                    if cnt >= nset and at_spot[k]:
                        ok = True
                else:
                    cnt = 0
                if ok and sk >= v_k and (not law or at_spot[max(a, k - lag // 2)]):
                    found = True
                    break
            k += 1
        if not found:
            continue
        if moves is not None and (out[a:k] != "D").any():
            moves.append(("rkick", int(a), int(k)))
        out[a:k] = "D"
    return out


def still_dead(lab: np.ndarray, period: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray, d_min: np.ndarray,
               alone: float = 1.0, v: float = 0.5, min_s: float = 3.0, lag: int = 8, fps: float = FPS,
               moves: Optional[list] = None) -> np.ndarray:
    """A ball at rest that nobody plays is dead (envelope L8 round 4, opt-in via decode_halves ed_still).
    A run of `min_s` seconds or more where the visible ball moves slower than `v` m/s (displacement
    over `lag` frames) and the median distance of the nearest player over the run is above `alone`
    metres becomes D. In open play no ball lies still for 3 s with nobody on it; a goalkeeper holding
    the ball has it within 1 m and is left alone. The run ends where the ball moves again (the kick).
    Measured on the external tune matches (scripts/a8b_r4.py still / var cfg=st1; ctl T=2 truth, on top of
    T95 + T100 + T112 + T113): orig (dead ball left where it lies) 213 such runs, truly D on 0.94 of
    their frames, 22 runs we call live of which 19 are truly D: +0.0101 [+0.0060, +0.0146]; real1 +0.0016;
    real2 4 seeds +0.0007 / -0.0003 / +0.0002 / +0.0007 (both benches move or hide every dead ball, so
    the rule has nothing to act on there). Without the `alone` condition real2 loses 0.0027.
    matchlens_start: footballtactics events/deadball.py still-ball state (ported as deadball2 still_v /
    v_kick, off since T36 / ADR-010): there a still ball is detector evidence at 0.3 m/s with no
    duration or isolation condition; here only a ball left alone for 3 s, as a post-pass."""
    lab = np.asarray(lab)
    out = lab.copy()
    n, h = len(lab), lag // 2
    s = np.full(n, np.nan)
    if n > lag:
        s[h:n - lag + h] = np.hypot(ball_x[lag:] - ball_x[:-lag], ball_y[lag:] - ball_y[:-lag]) * fps / lag
    with np.errstate(invalid="ignore"):
        m = np.isfinite(s) & (s < v) & ~np.isnan(ball_x) & (np.asarray(period) > 0)
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j] and period[j] == period[i]:
            j += 1
        if j - i >= min_s * fps and (alone <= 0 or np.nanmedian(d_min[i:j]) > alone):
            if moves is not None and (out[i:j] != "D").any():
                moves.append(("still", int(i), int(j)))
            out[i:j] = "D"
        i = j
    return out
