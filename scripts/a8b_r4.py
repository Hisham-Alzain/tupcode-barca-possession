"""Envelope L8-autonomous-fable, round 4: the exit side of the stoppage (physical restart date),
order of the edge passes, Smart Tagging transition clips.

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r4.py rprof bench=real2 seed=0 cfg=Xd2
    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r4.py kick bench=real2 seed=0 cfg=Xd2

rprof: ball speed / visibility / nearest player around the TRUE restart of the stoppages we hold (ctl T=2 truth)
kick:  candidate physical restart dates near our exit against the true restart
Tune matches only; no held-out, no public truth, nothing of the provided match.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a8b_run as R  # noqa: E402
import a8b_r3 as R3  # noqa: E402

a4_short, tune_shape = R.a4_short, R.tune_shape
FPS = 25.0
blocks = R3.blocks


def speed(bx, by, lag=8):
    n, h = len(bx), lag // 2
    v = np.full(n, np.nan)
    v[h:n - lag + h] = np.hypot(bx[lag:] - bx[:-lag], by[lag:] - by[:-lag]) * FPS / lag
    return v


def exits(lab, t, period, min_s=3.0):
    """(p0, q, te): our stoppage [p0, q) of >= min_s followed by a team spell, te = end of the true stoppage it overlaps most."""
    n = len(t)
    truns = blocks(t == "D", period)
    for p0, q in blocks(lab == "D", period):
        if q >= n or period[q] != period[p0] or lab[q] not in ("H", "A") or (q - p0) < min_s * FPS:
            continue
        ov = [(min(b, q) - max(a, p0), a, b) for a, b in truns if min(b, q) > max(a, p0)]
        if ov:
            yield p0, q, max(ov)[2]


def cmd_rprof(opt):
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "Xd2")
    v = "" if variant == "orig" else variant
    offs = [-3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]
    sp, vi, dm = {o: [] for o in offs}, {o: [] for o in offs}, {o: [] for o in offs}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        s = speed(feats["ball_x"], feats["ball_y"])
        d = np.minimum(feats["d_home"], feats["d_away"])
        n = len(t)
        for p0, q, te in exits(lab, t, period):
            for o in offs:
                k = te + int(o * FPS)
                if 0 <= k < n:
                    vi[o].append(vis[k])
                    if vis[k] and np.isfinite(s[k]):
                        sp[o].append(s[k])
                        dm[o].append(d[k])
        del feats
    print("%s s%d cfg %s: around the TRUE restart of the stoppages we hold (n %d)" % (variant, seed, c, len(vi[0.0])))
    for o in offs:
        x, dd = np.array(sp[o]), np.array(dm[o])
        print("  %+5.1f s  visible %.2f | speed p25 %5.1f med %5.1f p75 %5.1f | < 1 m/s %.2f  > 4 m/s %.2f | nearest player med %4.1f m" % (
            o, np.mean(vi[o]), *np.percentile(x, [25, 50, 75]), (x < 1).mean(), (x > 4).mean(), np.median(dd)))


def kick_frame(s, vis, lo, hi, v_k, still_v, still_n):
    """First frame f in [lo, hi) where the visible ball reaches v_k after `still_n` visible frames below still_v
    somewhere in the still_n * 3 frames before; -1 if none."""
    for f in range(lo, hi):
        if not (vis[f] and s[f] >= v_k):
            continue
        a = max(0, f - 3 * still_n)
        w = (s[a:f] < still_v) & vis[a:f]
        if w.sum() >= still_n:
            return f
    return -1


def cmd_kick(opt):
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "Xd2")
    v = "" if variant == "orig" else variant
    back, fwd = float(opt.pop("back", 4.0)), float(opt.pop("fwd", 2.0))
    grid = [(vk, sv) for vk in (3.0, 5.0, 8.0) for sv in (1.0, 2.0)]
    res = {g: [] for g in grid}
    ours = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        s = np.nan_to_num(speed(feats["ball_x"], feats["ball_y"]), nan=0.0)
        n = len(t)
        for p0, q, te in exits(lab, t, period):
            ours.append((q - te) / FPS)
            for g in grid:
                f = kick_frame(s, vis, max(p0, q - int(back * FPS)), min(n, q + int(fwd * FPS)), g[0], g[1], 5)
                res[g].append(((f - te) / FPS if f >= 0 else np.nan, (q - te) / FPS))
        del feats
    ours = np.array(ours)
    print("%s s%d cfg %s exits %d: ours - true: p25 %+.2f med %+.2f p75 %+.2f | abs med %.2f mean(min(abs,5)) %.2f" % (
        variant, seed, c, len(ours), *np.percentile(ours, [25, 50, 75]), np.median(np.abs(ours)), np.minimum(np.abs(ours), 5).mean()))
    for g in grid:
        x = np.array(res[g])
        ok = np.isfinite(x[:, 0])
        k, o = x[ok, 0], x[ok, 1]
        print("  v_k %.0f still %.0f: found %.2f | kick - true: p25 %+.2f med %+.2f p75 %+.2f abs med %.2f | ours there: p25 %+.2f med %+.2f p75 %+.2f abs med %.2f | closer %.2f, mean abs (cap 5) %.2f vs %.2f" % (
            g[0], g[1], ok.mean(), *np.percentile(k, [25, 50, 75]), np.median(np.abs(k)), *np.percentile(o, [25, 50, 75]), np.median(np.abs(o)),
            (np.abs(k) < np.abs(o)).mean(), np.minimum(np.abs(k), 5).mean(), np.minimum(np.abs(o), 5).mean()))


def cmd_reap(opt):
    """Exits where the ball is seen at our exit and was hidden earlier in the stoppage: reappearance frame r
    (start of the visible stretch that holds our exit) against the true restart and our exit."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "Xd2")
    v = "" if variant == "orig" else variant
    rows = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        s = np.nan_to_num(speed(feats["ball_x"], feats["ball_y"]), nan=0.0)
        for p0, q, te in exits(lab, t, period):
            if not vis[q]:
                continue
            r = q
            while r > p0 and vis[r - 1]:
                r -= 1
            if r == p0:
                continue                                   # visible through the whole stoppage
            rows.append(((q - r) / FPS, (r - te) / FPS, (q - te) / FPS, s[min(r + 4, q)], np.median(s[r:q + 1])))
        del feats
    x = np.array(rows)
    print("%s s%d cfg %s: exits with the ball seen at the exit after a hidden stretch: %d" % (variant, seed, c, len(x)))
    for lo, hi in ((0, 0.2), (0.2, 0.6), (0.6, 1.5), (1.5, 3.0), (3.0, 99)):
        m = (x[:, 0] >= lo) & (x[:, 0] < hi)
        if m.sum() < 3:
            continue
        for nm, mm in (("all", m), ("fast at r (>= 4 m/s)", m & (x[:, 3] >= 4)), ("slow at r", m & (x[:, 3] < 4))):
            if mm.sum() >= 3:
                print("  seen %.1f-%.1f s before our exit, %-20s n %3d | reappearance - true: p25 %+.2f med %+.2f p75 %+.2f abs med %.2f | ours - true: p25 %+.2f med %+.2f p75 %+.2f abs med %.2f" % (
                    lo, hi, nm, mm.sum(), *np.percentile(x[mm, 1], [25, 50, 75]), np.median(np.abs(x[mm, 1])), *np.percentile(x[mm, 2], [25, 50, 75]), np.median(np.abs(x[mm, 2]))))


def cmd_kprov(opt):
    """Provided match INPUT only (no truth): is there a kick from a still ball near our stoppage exits, and does it
    sit tighter against the SET PIECES clip starts than our own exit does."""
    import a8_tags
    from possess import pipeline

    c = opt.pop("cfg", "Xd2")
    vk, sv = float(opt.pop("vk", 5.0)), float(opt.pop("sv", 1.0))
    back, fwd = float(opt.pop("back", 4.0)), float(opt.pop("fwd", 2.0))
    tags, home, f, per, _ = a8_tags.challenge_default()
    lab = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
    vis = ~np.isnan(f["ball_x"])
    s = np.nan_to_num(speed(f["ball_x"], f["ball_y"]), nan=0.0)
    n = len(lab)
    sp_starts = np.array(sorted(t["a"] + 1 for t in tags if t["code"] == "SET PIECES" and not t["type"].startswith("Kick Off (Start")))
    rows = []
    for p0, q in blocks(lab == "D", per):
        if q >= n or per[q] != per[p0] or lab[q] not in ("H", "A") or (q - p0) < 3 * FPS:
            continue
        k = kick_frame(s, vis, max(p0, q - int(back * FPS)), min(n, q + int(fwd * FPS)), vk, sv, 5)
        j = np.searchsorted(sp_starts, q)
        near = [x for x in sp_starts[max(0, j - 2):j + 1] if -2 * FPS <= q - x <= 12 * FPS]
        rows.append((q, k, vis[max(p0, q - 25):q].mean(), vis[q], near[-1] if near else -1, np.median(s[max(p0, q - 50):q][vis[max(p0, q - 50):q]]) if vis[max(p0, q - 50):q].any() else np.nan))
    r = np.array(rows, float)
    fd = r[:, 1] >= 0
    print("provided input cfg %s: stoppage exits (D >= 3 s) %d | ball visible in the last second: all of it %.2f, none %.2f | visible at the exit %.2f | median speed in the last 2 s where seen: p25 %.1f med %.1f p75 %.1f" % (
        c, len(r), (r[:, 2] == 1).mean(), (r[:, 2] == 0).mean(), r[:, 3].mean(), *np.nanpercentile(r[:, 5], [25, 50, 75])))
    d = (r[fd, 1] - r[fd, 0]) / FPS
    print("  kick (v_k %.0f, still %.0f) found at %d exits (%.2f): kick - our exit s: p10 %+.2f p25 %+.2f med %+.2f p75 %+.2f p90 %+.2f | abs > 0.5 s %.2f" % (
        vk, sv, fd.sum(), fd.mean(), *np.percentile(d, [10, 25, 50, 75, 90]), (np.abs(d) > 0.5).mean()))
    sp = fd & (r[:, 4] >= 0)
    if sp.any():
        a, b = (r[sp, 0] - r[sp, 4]) / FPS, (r[sp, 1] - r[sp, 4]) / FPS
        print("  with a SET PIECES clip start nearby (n %d): our exit - clip start: p25 %.2f med %.2f p75 %.2f (IQR %.2f, sd %.2f) | kick - clip start: p25 %.2f med %.2f p75 %.2f (IQR %.2f, sd %.2f)" % (
            sp.sum(), *np.percentile(a, [25, 50, 75]), np.subtract(*np.percentile(a, [75, 25])), a.std(), *np.percentile(b, [25, 50, 75]), np.subtract(*np.percentile(b, [75, 25])), b.std()))
    al = r[:, 4] >= 0
    a = (r[al, 0] - r[al, 4]) / FPS
    print("  all exits with a clip start nearby (n %d): our exit - clip start p25 %.2f med %.2f p75 %.2f" % (al.sum(), *np.percentile(a, [25, 50, 75])))


def cmd_trans(opt):
    """Provided match INPUT only (no truth): our labels around the start of each clip, per code: share of the
    clip's team / the other team / D per second from -6 s to +12 s, and where our nearest change INTO the
    clip's team sits against the clip start."""
    import a8_tags
    from possess import pipeline, smarttags as st

    c = opt.pop("cfg", "Xd2")
    tags, home, f, per, _ = a8_tags.challenge_default()
    lab = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
    n = len(lab)
    sgn = np.where(lab == "H", 1, np.where(lab == "A", -1, 0))
    into = {1: np.flatnonzero((sgn[1:] == 1) & (sgn[:-1] != 1)) + 1, -1: np.flatnonzero((sgn[1:] == -1) & (sgn[:-1] != -1)) + 1}
    by = {}
    for t in tags:
        s = st.team_sign(t, home)
        if s:
            by.setdefault(t["code"], []).append((t["a"] + 1, t["b"] + 1, s, t["type"]))
    offs = list(range(-6, 13))
    for code, ts in sorted(by.items(), key=lambda kv: -len(kv[1])):
        L = np.array([(b - a) / FPS for a, b, _, _ in ts])
        print("%s n %d, length med %.1f s (p10 %.1f, p90 %.1f)" % (code, len(ts), np.median(L), *np.percentile(L, [10, 90])))
        own, oth, dd = [], [], []
        for o in offs:
            x = np.array([sgn[a + o * 25:a + o * 25 + 25].mean() * 0 + np.array([(sgn[a + o * 25:a + o * 25 + 25] == s).mean(), (sgn[a + o * 25:a + o * 25 + 25] == -s).mean(),
                          (lab[a + o * 25:a + o * 25 + 25] == "D").mean()]) for a, b, s, _ in ts if a + o * 25 >= 0 and a + o * 25 + 25 <= n])
            own.append(x[:, 0].mean()), oth.append(x[:, 1].mean()), dd.append(x[:, 2].mean())
        print("   s    " + " ".join("%4d" % o for o in offs))
        print("   own  " + " ".join("%4.2f" % x for x in own))
        print("   other" + " ".join("%4.2f" % x for x in oth))
        print("   D    " + " ".join("%4.2f" % x for x in dd))
        d = []
        for a, b, s, _ in ts:
            k = into[s]
            k = k[(k >= a - 10 * 25) & (k <= a + 15 * 25)]
            if len(k):
                d.append((k[np.argmin(np.abs(k - a))] - a) / FPS)
        d = np.array(d)
        if len(d):
            print("   nearest change into the clip's team - clip start: n %d of %d | p10 %+.1f p25 %+.1f med %+.1f p75 %+.1f p90 %+.1f | within 1 s %.2f, 2 s %.2f" % (
                len(d), len(ts), *np.percentile(d, [10, 25, 50, 75, 90]), (np.abs(d) <= 1).mean(), (np.abs(d) <= 2).mean()))


# --- rows 53-55: the two exit passes (rejected by the gate, so not kept in src/possess; they ran as the last
# steps of decode_halves behind ed_kick / ed_reap, configs K5 / P15 / PK in a8b_run.CONFIGS) ---
from typing import Optional  # noqa: E402
from possess.edges import _segments  # noqa: E402
def kick_exit(lab: np.ndarray, period: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray,
              back_s: float = 4.0, fwd_s: float = 2.0, v_k: float = 5.0, still_v: float = 1.0, still_n: int = 5,
              min_s: float = 3.0, lag: int = 8, fps: float = FPS, moves: Optional[list] = None) -> np.ndarray:
    """Stoppage exit at the kick (envelope L8 round 4, opt-in via decode_halves ed_kick). A restart is a
    kick of a ball at rest: for each predicted stoppage of `min_s` seconds or more that a team spell
    follows, the first frame from `back_s` seconds before to `fwd_s` seconds after our exit where the
    visible ball reaches `v_k` m/s (centred over `lag` frames) after at least `still_n` visible frames
    under `still_v` m/s in the 3 * still_n frames before it dates the restart. The stoppage is cut or
    extended to that frame (one frame of each side stays). No such kick (ball hidden, carried): untouched.
    Measured on the external tune matches (scripts/a8b_r4.py kick; ctl T=2 truth, labels T95 + T100 +
    T112 + T113): found at 0.11 (real2) / 0.35 (real1) of the exits, kick minus true restart median
    +0.00 s (abs 0.16 / 0.04 s) against abs 0.96 / 0.84 s for our own exit there.
    matchlens_start: footballtactics events/deadball.py v_kick (ported as deadball2 v_kick, off since
    T36): there a kick ends the detector's still-ball state; here it only dates the exit of a stoppage
    the decoder already holds (T36 made every still ball a stoppage, which did not transfer)."""
    lab = np.asarray(lab)
    out = lab.copy()
    n, h = len(lab), lag // 2
    s = np.zeros(n)
    if n > lag:
        s[h:n - lag + h] = np.nan_to_num(np.hypot(ball_x[lag:] - ball_x[:-lag], ball_y[lag:] - ball_y[:-lag]) * fps / lag, nan=0.0)
    vis = ~np.isnan(ball_x)
    slow = (vis & (s < still_v)).astype(int)
    cs = np.concatenate([[0], np.cumsum(slow)])
    for a, b in _segments(lab, period):
        if lab[a] != "D" or b >= n or period[b] != period[a] or lab[b] not in ("H", "A") or b - a < min_s * fps:
            continue
        e = b
        while e < n and lab[e] == lab[b] and period[e] == period[b]:
            e += 1
        lo, hi = max(a + 1, b - int(back_s * fps)), min(e - 1, b + int(fwd_s * fps))
        f = next((k for k in range(lo, hi) if vis[k] and s[k] >= v_k and cs[k] - cs[max(0, k - 3 * still_n)] >= still_n), -1)
        if f < 0 or f == b:
            continue
        if f < b:
            out[f:b] = lab[b]
        else:
            out[b:f] = "D"
        if moves is not None:
            moves.append(("kick", int(b), int(f)))
    return out


def reap_exit(lab: np.ndarray, period: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray,
              max_s: float = 3.0, v_r: float = 4.0, min_s: float = 3.0, lag: int = 8, fps: float = FPS,
              moves: Optional[list] = None) -> np.ndarray:
    """Stoppage exit at the ball's reappearance in flight (envelope L8 round 4, opt-in via decode_halves
    ed_reap). For each predicted stoppage of `min_s` seconds or more that a team spell follows: when the
    ball is visible at our exit, was hidden earlier in the stoppage, and came back into view inside the
    pitch at most `max_s` seconds before our exit already moving at `v_r` m/s or more (centred over `lag`
    frames from its first visible frame), the stoppage ends at that first visible frame: a ball that
    comes back into view in flight inside the pitch has been kicked. A ball that comes back slow
    (placed, carried) leaves the exit where it is.
    Measured on the external tune matches (scripts/a8b_r4.py reap; ctl T=2 truth, labels T95 + T100 +
    T112 + T113): real2, reappearance 0.6-3 s before our exit and fast: 40 of 250 exits, reappearance
    minus true restart median +0.02..+0.04 s against +1.1..+2.2 s for our exit. CAUTION: real2 hides the
    ball "from a slow ball to the restart kick" by construction (ADR-016); on real1 (hiding independent of
    play, dead ball moves) the 6 such exits are 3.4 s early. See the log row for the transfer argument.
    matchlens_start: nothing in footballtactics events/ (deadball.py, possession.py, kinematics.py
    checked) uses the reappearance of the ball; MatchLens assumes a ball track without long gaps."""
    from .deadball2 import HALF_L, HALF_W

    lab = np.asarray(lab)
    out = lab.copy()
    n, h = len(lab), lag // 2
    vis = ~np.isnan(ball_x)
    for a, b in _segments(lab, period):
        if lab[a] != "D" or b >= n or period[b] != period[a] or lab[b] not in ("H", "A") or b - a < min_s * fps or not vis[b]:
            continue
        r = b
        while r > a and vis[r - 1]:
            r -= 1
        if r <= a or b - r > max_s * fps or b - r < 2 or r + lag >= n or not vis[r + lag]:
            continue
        if abs(ball_x[r]) > HALF_L or abs(ball_y[r]) > HALF_W:
            continue
        if np.hypot(ball_x[r + lag] - ball_x[r], ball_y[r + lag] - ball_y[r]) * fps / lag < v_r:
            continue
        out[r:b] = lab[b]
        if moves is not None:
            moves.append(("reap", int(b), int(r)))
    return out


def touch_ev(feats, period, kw):
    """decode_halves' touch events for the ctl_hi pass (same call, per period)."""
    from possess import decode

    n = len(period)
    ev = np.zeros(n)
    for i, j in blocks(np.asarray(period) > 0, period):
        ev[i:j] = decode.touch_events(**{k: feats[k][i:j] for k in ("ball_x", "ball_y", "d_home", "d_away", "visible")},
                                      pos=feats["pos"][i:j] if feats.get("pos") is not None else None,
                                      **{k: kw[k] for k in ("r_touch", "lag", "ball_clean", "fps", "smooth_touch", "vtx", "vtx_raw", "duel") if k in kw})
    return ev


def variants(feats, period, which):
    """Labels of the post-pass variants named in `which` (dict name -> labels); Xd2 is the baseline."""
    from possess import ctl_truth, edges, pipeline

    kw = dict(pipeline.DEC, **R.CONFIGS["Xd2"])
    out = {"Xd2": a4_short.decode_one(feats, period, R.CONFIGS["Xd2"])}
    bx, by = feats["ball_x"], feats["ball_y"]
    ev = touch_ev(feats, period, kw) if any(w in which for w in ("ord", "post")) else None

    def ctl(lab):
        if kw.get("ctl_s"):
            lab = ctl_truth.relabel(lab, period, float(kw["ctl_s"]), FPS)
        return ctl_truth.relabel(lab, period, float(kw["ctl_hi"]), FPS, passes=ev, min_pass=int(kw["ctl_touch"]))

    def cross(lab):
        lab = edges.cross_entry(lab, period, bx, by, kw["ed_cross"], 5, FPS, strict=kw["ed_cross_strict"])
        return edges.cross_dead(lab, period, bx, by, margin=0.5, fps=FPS, min_s=kw["ed_xdead_min"])

    for w in which:                                       # stA<r>: ball at rest >= 3 s with no player within r m (median) -> D
        if w.startswith("st"):
            out[w] = still_dead(out["Xd2"], period, bx, by, np.minimum(feats["d_home"], feats["d_away"]), alone=float(w[2:]))
    if "ord" in which or "chk" in which:
        raw = a4_short.decode_one(feats, period, dict(R.CONFIGS["Xd2"], ctl_s=0, ctl_hi=0, ed_cross=0, ed_xdead=0))
    if "chk" in which:                                    # must reproduce Xd2 exactly
        ev = touch_ev(feats, period, kw) if ev is None else ev
        out["chk"] = cross(ctl(raw))
    if "ord" in which:                                    # edge passes first, then the ctl passes
        out["ord"] = ctl(cross(raw))
    if "post" in which:                                   # ctl passes once more after the edge passes
        out["post"] = ctl(out["Xd2"])
    return out


def cmd_var(opt):
    """Scores post-pass variants built outside the decoder against Xd2 (same table as a8b_run run)."""
    import json

    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    which = str(opt.pop("cfg", "chk,ord,post")).split(",")
    tag = opt.pop("tag", "r4v")
    v = "" if variant == "orig" else variant
    labs, tv = {}, {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        for k, lab in variants(feats, period, which).items():
            labs[k, name] = lab
        del feats
    cfgs = ["Xd2"] + which
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  spellF1 mean of the 3 tune matches" % (variant, seed))
    with open(os.path.join(ROOT, "eval", "results", "a8b_%s_%s_s%d.json" % (tag, variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "variants": which, "results": res, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def still_runs(bx, by, period, v=0.5, min_s=3.0, lag=8):
    """[a, b) runs where the visible ball moves slower than v m/s (centred over lag frames) for min_s seconds or more."""
    s = speed(bx, by, lag)
    with np.errstate(invalid="ignore"):
        m = np.isfinite(s) & (s < v) & ~np.isnan(bx)
    return [(a, b) for a, b in blocks(m, period) if b - a >= min_s * FPS]


def still_dead(lab, period, bx, by, dmin, alone=2.0, v=0.5, min_s=3.0, moves=None):
    """Row 57 candidate: a visible ball at rest (< v m/s) for min_s seconds or more with no player within `alone`
    metres (median over the run; 0 = no such condition) is a dead ball: the run becomes D."""
    out = np.asarray(lab).copy()
    for a, b in still_runs(bx, by, period, v, min_s):
        if alone > 0 and not np.nanmedian(dmin[a:b]) > alone:
            continue
        if moves is not None and (out[a:b] != "D").any():
            moves.append((int(a), int(b), float((out[a:b] != "D").sum() / FPS)))
        out[a:b] = "D"
    return out


def cmd_still(opt):
    """Visible ball at rest for >= min_s seconds: how often, what we call it, what it truly is (external),
    or where it sits against the SET PIECES clips (bench=prov: provided INPUT only, no truth)."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "Xd2")
    v, ms = float(opt.pop("v", 0.5)), float(opt.pop("min", 3.0))
    if variant == "prov":
        import a8_tags
        from possess import pipeline
        from possess.deadball2 import HALF_L, HALF_W

        tags, home, f, per, _ = a8_tags.challenge_default()
        lab = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
        sp = np.array(sorted(t["a"] + 1 for t in tags if t["code"] == "SET PIECES"))
        runs = still_runs(f["ball_x"], f["ball_y"], per, v, ms)
        play = per > 0
        tot = sum(b - a for a, b in runs)
        dsh = np.array([(lab[a:b] == "D").mean() for a, b in runs])
        print("provided input cfg %s: still runs (< %.1f m/s for >= %.0f s) %d, %.1f s in all (%.2f %% of graded frames); ours: D on %.2f of their frames; runs mostly live (< 0.5 D) %d" % (
            c, v, ms, len(runs), tot / FPS, 100 * tot / play.sum(), sum(((lab[a:b] == "D").sum()) for a, b in runs) / max(tot, 1), (dsh < 0.5).sum()))
        for (a, b), d in zip(runs, dsh):
            if d < 0.5 or opt.get("all"):
                j = np.searchsorted(sp, a)
                near = [(a - x) / FPS for x in sp[max(0, j - 1):j + 1] if -10 * FPS <= a - x <= 20 * FPS]
                ld = min(HALF_L - abs(f["ball_x"][a]), HALF_W - abs(f["ball_y"][a]))
                dm = min(f["d_home"][a], f["d_away"][a])
                print("   frame %6d  %.1f s  D share %.2f  x %+5.1f y %+5.1f  to line %4.1f m  nearest player %4.1f m  before: %s after: %s | set-piece clip start %s" % (
                    a, (b - a) / FPS, d, f["ball_x"][a], f["ball_y"][a], ld, dm, lab[max(a - 1, 0)], lab[min(b, len(lab) - 1)], ("%+.1f s before the run" % near[0]) if near else "none near"))
        return
    vv = "" if variant == "orig" else variant
    n_r = tot = dfr = tdfr = 0
    live_runs, live_true_d = 0, 0
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, vv, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        for a, b in still_runs(feats["ball_x"], feats["ball_y"], period, v, ms):
            n_r += 1
            tot += b - a
            dfr += (lab[a:b] == "D").sum()
            tdfr += (t[a:b] == "D").sum()
            if (lab[a:b] == "D").mean() < 0.5:
                live_runs += 1
                live_true_d += (t[a:b] == "D").mean() >= 0.5
        del feats
    print("%s s%d cfg %s: still runs (< %.1f m/s for >= %.0f s) %d, %.1f s; truly D on %.2f of their frames, ours D on %.2f; runs we call mostly live %d, of which truly D %d" % (
        variant, seed, c, v, ms, n_r, tot / FPS, tdfr / max(tot, 1), dfr / max(tot, 1), live_runs, live_true_d))


def cmd_stprov(opt):
    """Provided match INPUT only (no truth): what still_dead changes against cfg (our own output)."""
    import a8_tags
    from possess import ctl_truth, pipeline

    c, alone = opt.pop("cfg", "Xd2"), float(opt.pop("alone", 1.0))
    _, _, f, per, _ = a8_tags.challenge_default()
    lab0 = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
    mv = []
    lab = still_dead(lab0, per, f["ball_x"], f["ball_y"], np.minimum(f["d_home"], f["d_away"]), alone=alone, moves=mv)
    play = per > 0
    print("provided input, %s + still_dead(alone %.1f): frames changed %.4f (%.1f s), runs touched %d: %s | D runs %d -> %d, possessions %d -> %d" % (
        c, alone, ((lab != lab0) & play).sum() / play.sum(), (lab != lab0).sum() / FPS, len(mv), [(a, round(s, 1)) for a, b, s in mv],
        len(blocks(lab0 == "D", per)), len(blocks(lab == "D", per)), ctl_truth.n_possessions(lab0, per), ctl_truth.n_possessions(lab, per)))


def cmd_tprof(opt):
    """External analogue of `trans`: our labels per second around TRUE open-play turnovers (ctl T=2 truth; a spell
    of >= pre s of one team directly followed by >= post s of the other), on the clip clock of an ATTACKING
    TRANSITION clip that starts `lead` s before the turnover; and our nearest change into the new team."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "Xd2")
    pre, post, lead = float(opt.pop("pre", 3.0)), float(opt.pop("post", 8.0)), float(opt.pop("lead", 5.0))
    v = "" if variant == "orig" else variant
    offs = list(range(-6, 13))
    own, oth, dd, d = {o: [] for o in offs}, {o: [] for o in offs}, {o: [] for o in offs}, []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        del feats
        n = len(t)
        sg = np.where(lab == "H", 1, np.where(lab == "A", -1, 0))
        segs = [(a, b) for a, b in R3.blocks(np.ones(n, bool), period)]
        runs = []
        for a, b in segs:
            i = a
            while i < b:
                j = i
                while j < b and t[j] == t[i]:
                    j += 1
                runs.append((i, j, t[i]))
                i = j
        for (a0, b0, x0), (a1, b1, x1) in zip(runs[:-1], runs[1:]):
            if b0 != a1 or period[a0] != period[a1] or {x0, x1} != {"H", "A"} or b0 - a0 < pre * FPS or b1 - a1 < post * FPS:
                continue
            s = 1 if x1 == "H" else -1
            st = a1 - int(lead * FPS)                    # clip start
            for o in offs:
                w = slice(st + o * 25, st + o * 25 + 25)
                if w.start < 0 or w.stop > n:
                    continue
                own[o].append((sg[w] == s).mean()), oth[o].append((sg[w] == -s).mean()), dd[o].append((lab[w] == "D").mean())
            k = np.flatnonzero((sg[1:] == s) & (sg[:-1] != s)) + 1
            k = k[(k >= st - 250) & (k <= st + 375)]
            if len(k):
                d.append((k[np.argmin(np.abs(k - st))] - st) / FPS)
    d = np.array(d)
    print("%s s%d cfg %s: true turnovers (>= %.0f s before, >= %.0f s after) %d; clip clock = turnover at +%.0f s" % (variant, seed, c, pre, post, len(own[0]), lead))
    print("   s    " + " ".join("%4d" % o for o in offs))
    print("   own  " + " ".join("%4.2f" % np.mean(own[o]) for o in offs))
    print("   other" + " ".join("%4.2f" % np.mean(oth[o]) for o in offs))
    print("   D    " + " ".join("%4.2f" % np.mean(dd[o]) for o in offs))
    print("   nearest change into the new team - clip start: n %d | p10 %+.1f p25 %+.1f med %+.1f p75 %+.1f p90 %+.1f" % (len(d), *np.percentile(d, [10, 25, 50, 75, 90])))


def cmd_splead(opt):
    """Provided match INPUT only (no truth): per SET PIECES clip, physical restart candidates on the clip clock
    against our stoppage exit: kick of a ball at rest; first visible frame at >= 6 m/s after rest or after a
    hidden stretch; and what the ball does in the clip's first 12 s."""
    import a8_tags
    from possess import pipeline

    c = opt.pop("cfg", "Xd2")
    tags, home, f, per, _ = a8_tags.challenge_default()
    lab = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
    bx, by = f["ball_x"], f["ball_y"]
    vis = ~np.isnan(bx)
    s = np.nan_to_num(speed(bx, by), nan=0.0)
    n = len(lab)
    rows = {}
    for t in tags:
        if t["code"] != "SET PIECES":
            continue
        a = t["a"] + 1
        lo, hi = max(a - 50, 0), min(a + 300, n)
        ex = [k for k in range(lo + 1, hi) if lab[k - 1] == "D" and lab[k] in ("H", "A")]
        ex = min(ex, key=lambda k: abs(k - a - 100)) if ex else -1
        kf = kick_frame(s, vis, lo, hi, 5.0, 1.0, 5)
        fast = -1
        for k in range(lo + 1, hi):
            if vis[k] and s[k] >= 6.0 and (not vis[max(k - 25, 0):k].all() or (s[max(k - 25, 0):k] < 2.0).any()):
                fast = k
                break
        rows.setdefault(t["type"], []).append((a, (ex - a) / FPS if ex >= 0 else np.nan, (kf - a) / FPS if kf >= 0 else np.nan,
                                               (fast - a) / FPS if fast >= 0 else np.nan, vis[a:a + 125].mean()))
    fm = lambda x: " ".join("%5.1f" % v if np.isfinite(v) else "    -" for v in x)   # noqa: E731
    for k, r in rows.items():
        r = np.array(r)
        print("%-22s n %2d | ball visible in the first 5 s %.2f | our exit: n %2d med %.1f (IQR %.1f-%.1f) | kick from rest: n %2d med %.1f (IQR %.1f-%.1f) | first fast ball: n %2d med %.1f (IQR %.1f-%.1f)" % (
            k, len(r), r[:, 4].mean(), np.isfinite(r[:, 1]).sum(), *np.nanpercentile(r[:, 1], [50, 25, 75]),
            np.isfinite(r[:, 2]).sum(), *(np.nanpercentile(r[:, 2], [50, 25, 75]) if np.isfinite(r[:, 2]).any() else [np.nan] * 3),
            np.isfinite(r[:, 3]).sum(), *(np.nanpercentile(r[:, 3], [50, 25, 75]) if np.isfinite(r[:, 3]).any() else [np.nan] * 3)))
        if opt.get("list"):
            print("     exit: " + fm(r[:, 1]))
            print("     kick: " + fm(r[:, 2]))
            print("     fast: " + fm(r[:, 3]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:])
    globals()["cmd_" + cmd](opt)
