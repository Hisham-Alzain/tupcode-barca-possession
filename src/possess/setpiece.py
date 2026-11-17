"""Tag-gated set-piece dead ball (envelope L11-C; opt-in, OFF by default).

A Smart Tagging SET PIECES clip of a known type (Goal Kick, Corner Kick, Free Kick, Throw-in) says
that a restart of that type happens near the clip start. The tag is used as a GATE, not as a clock
(T114: the clip start is 3-9 s before the restart, a different lead per type): inside the window
[clip start - pre_s, clip start + post_s] the restart is found PHYSICALLY on the ball track, and

  * the frames from the moment the ball went dead (the end of our last stoppage before the restart,
    or the ball's last frame beyond a pitch line, whichever is later) up to the restart are D
    (Law 9: the ball is out of play until the restart is taken; CHALLENGE.md: stoppages are D);
  * with a tagged taker team, D frames right after the kick (`take_s`) go to the taker (Laws 13, 15,
    16, 17: the restart is taken by that team).

No restart found, or no stoppage / line crossing within `gap_s` before it -> nothing changes.

Restart = the ball at rest (speed < v_rest for >= rest_s, visible) and then kicked (speed > v_kick
within kick_s after the rest ends), with the rest position at the legal spot of the type (Laws):
  goal kick   inside the goal area (5.5 m from the goal line, 9.16 m from the goal centre) + tol
  corner      inside the corner arc (1 m from the corner flag) + tol
  throw-in    on the touch line (|y| = 34) +- tol; the ball is held, then thrown
  free kick   anywhere on the pitch
`hid=1` also accepts a restart that is seen only as the ball REAPPEARING at the legal spot after a
hidden block of >= rest_s and moving faster than v_kick (the ball is often hidden while it is dead).

Constants: Laws of the Game geometry; the speed thresholds are physical (a ball at rest vs a kicked
ball), `tol` is the tracking error; window and gap are simulator-checked on external data only
(scripts/a12_setpiece.py). Nothing is fitted on the provided match.

matchlens_start: detectors/set_piece_detector.py classifies restarts from geometry (ball still at
a corner arc / goal area / touch line, nearest player as taker); the rest-then-kick test is the
same idea, gated here by the tag instead of a formation classifier. deadball.py (dead from a line
crossing) gives the dead-start evidence. Ported as geometry only (no code import).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

FPS = 25.0
HALF_L, HALF_W = 52.5, 34.0
GA_DEPTH, GA_HALF_W = 5.5, 9.16 + 5.5          # goal area: 5.5 m deep, 18.32 + 2 x 5.5 m wide
SP = dict(pre_s=3.0, post_s=15.0, gap_s=15.0, v_rest=1.0, rest_s=0.4, v_kick=4.0, kick_s=1.0,
          tol=2.0, take_s=2.0, hid=0, line_m=0.5, fk=0)
# fk=0: free kicks are skipped. A free kick has no fixed spot, so "rest then kick anywhere" also
# matches a slow ball in the first seconds of live play after a quick free kick: on real2 seed 2 the
# kick was found 4-8 s after the true restart in 5 of 8 simulated free kicks and live play was filled
# with D (scripts/a12_setpiece.py diag, external truth). Types with a Laws spot only.
TYPES = ("goal kick", "corner kick", "free kick", "throw-in")


def norm_type(t: str) -> Optional[str]:
    t = (t or "").strip().lower()
    for k in TYPES:
        if t.startswith(k):
            return k
    return None


def spot_ok(typ: str, x: float, y: float, tol: float) -> bool:
    """Is (x, y) the legal spot of a restart of type `typ` (Laws 13, 15, 16, 17), within `tol` m?"""
    if np.isnan(x) or np.isnan(y):
        return False
    ax, ay = abs(x), abs(y)
    if typ == "goal kick":
        return ax >= HALF_L - GA_DEPTH - tol and ay <= GA_HALF_W + tol
    if typ == "corner kick":
        return np.hypot(HALF_L - ax, HALF_W - ay) <= 1.0 + tol
    if typ == "throw-in":
        return ay >= HALF_W - tol and ax <= HALF_L + tol
    return ax <= HALF_L + tol and ay <= HALF_W + tol


def speed(bx: np.ndarray, by: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Ball speed (m/s) from a centred 2-frame difference; NaN where the ball is hidden."""
    v = np.full(len(bx), np.nan)
    if len(bx) > 2:
        v[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * fps / 2.0
    return v


def find_restart(bx, by, v, lo: int, hi: int, typ: str, kw: dict, fps: float = FPS) -> Optional[Tuple[int, str]]:
    """(kick frame, how) of the first restart of type `typ` in [lo, hi), or None (module docstring)."""
    n = len(bx)
    lo, hi = max(lo, 1), min(hi, n - 1)
    rest_n, kick_n = max(1, int(round(kw["rest_s"] * fps))), max(1, int(round(kw["kick_s"] * fps)))
    vis = ~np.isnan(bx)
    rest = vis & (np.nan_to_num(v, nan=99.0) < kw["v_rest"])
    run = 0
    for i in range(max(lo - rest_n, 1), hi):
        if rest[i]:
            run += 1
            continue
        if run >= rest_n and i >= lo:              # rest ended at i: kicked soon after, from the legal spot?
            m = slice(i - run, i)
            if spot_ok(typ, float(np.nanmedian(bx[m])), float(np.nanmedian(by[m])), kw["tol"]):
                w = np.nan_to_num(v[i:min(i + kick_n, n)], nan=0.0)
                if len(w) and w.max() > kw["v_kick"]:
                    return i, "rest"
        run = 0
    if kw.get("hid"):                              # ball reappears at the spot and is moving
        for i in range(max(lo, rest_n), hi):
            if vis[i] and not vis[i - rest_n:i].any() and spot_ok(typ, bx[i], by[i], kw["tol"]):
                w = np.nan_to_num(v[i:min(i + kick_n, n)], nan=0.0)
                if len(w) and w.max() > kw["v_kick"]:
                    return i, "reappear"
    return None


def apply(lab: np.ndarray, bx: np.ndarray, by: np.ndarray, period: np.ndarray, events: List[dict],
          sp_kw: Dict[str, float] = None, home_code: str = "H", away_code: str = "A",
          fps: float = FPS) -> Tuple[np.ndarray, List[dict]]:
    """Labels with the tag-gated set-piece rule applied, and one info row per event.

    events: [{"a": clip start (array index), "type": tag Type, "sign": +1 home / -1 away / 0}].
    """
    kw = dict(SP, **(sp_kw or {}))
    out = np.array(lab, copy=True)
    v = speed(bx, by, fps)
    period = np.asarray(period)
    out_of = (np.abs(np.nan_to_num(bx)) > HALF_L + kw["line_m"]) | (np.abs(np.nan_to_num(by)) > HALF_W + kw["line_m"])
    gap_n, take_n = int(round(kw["gap_s"] * fps)), int(round(kw["take_s"] * fps))
    info = []
    for ev in sorted(events, key=lambda e: e["a"]):
        typ = norm_type(ev.get("type", ""))
        row = {"a": int(ev["a"]), "type": typ, "sign": int(ev.get("sign", 0)), "kick": None}
        info.append(row)
        if typ is None or (typ == "free kick" and not kw["fk"]):
            continue
        lo, hi = ev["a"] - int(round(kw["pre_s"] * fps)), ev["a"] + int(round(kw["post_s"] * fps))
        r = find_restart(bx, by, v, lo, hi, typ, kw, fps)
        if r is None:
            continue
        k, how = r
        row.update(kick=int(k), how=how)
        if period[k] <= 0:
            continue
        b0 = max(k - gap_n, 0)
        seg = slice(b0, k)
        same = period[seg] == period[k]
        dead_ev = ((out[seg] == "D") | out_of[seg]) & same
        j = np.flatnonzero(dead_ev)
        if len(j) == 0:
            row["no_dead"] = True
            continue
        e = b0 + int(j[-1])                        # last dead evidence before the kick
        row.update(dead_from=e, filled=int((out[e + 1:k] != "D").sum()),
                   before=str("".join(sorted(set(out[e + 1:k].tolist())))))
        out[e + 1:k] = "D"
        if row["sign"] and take_n:
            code = home_code if row["sign"] > 0 else away_code
            t = k
            while t < min(k + take_n, len(out)) and out[t] == "D" and period[t] == period[k]:
                out[t] = code
                t += 1
            row["taken"] = t - k
    return out, info
