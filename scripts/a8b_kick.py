"""Envelope L8-autonomous-fable round 2: flight-direction evidence for the team in possession
(next step (a) of .planning/research/autonomous-L8.md: one-kick possessions with no detected touch).

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_kick.py flight bench=real2 seed=0 [cfg=hi2t1]
    uv run python scripts/ramlock.py -- uv run python scripts/a8b_kick.py starts bench=real2 seed=0 [cfg=hi2t1]

flight: on visible frames where the ball flies (speed and distance to the nearest player above
        thresholds), how often the team attacking in the ball's direction is the true team (ctl T=2
        truth), against our own label on the same frames.
starts: what the inputs show at the START of the true possessions we give to the other team.
Tune matches only; no held-out, no public truth, nothing of the provided match.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a8b_run as R  # noqa: E402
import tune_shape  # noqa: E402
from possess import decode, shape  # noqa: E402

FPS = 25.0


def ball_vel(bx, by, lag=8):
    """Centred ball velocity over `lag` frames (m/s); nan where either end is hidden."""
    n, h = len(bx), lag // 2
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    vx[h:n - h] = (bx[lag:] - bx[:-lag]) * FPS / lag
    vy[h:n - h] = (by[lag:] - by[:-lag]) * FPS / lag
    return vx, vy


def cmd_flight(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "hi2t1")
    acc = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        ad = shape.attack_dir(feats["pos"], period).astype(float)
        vx, vy = ball_vel(feats["ball_x"], feats["ball_y"])
        sp = np.hypot(vx, vy)
        dmin = np.fmin(feats["d_home"], feats["d_away"])
        live = np.isin(t, ["H", "A"]) & np.isfinite(sp) & (ad != 0) & (period > 0)
        sg_t = np.where(t == "H", 1.0, -1.0)
        sg_p = np.where(lab == "H", 1.0, np.where(lab == "A", -1.0, 0.0))
        fwd = np.sign(vx * ad)                          # +1: the ball flies the way home attacks
        with np.errstate(invalid="ignore"):
            ax = np.abs(vx) / sp
        for v0 in (4, 8, 12, 16):
            for d0 in (0, 2, 4):
                for a0 in (0.0, 0.7):
                    m = live & (sp >= v0) & (dmin >= d0) & (ax >= a0)
                    a = acc.setdefault((v0, d0, a0), np.zeros(6))
                    a += [m.sum(), (fwd[m] == sg_t[m]).sum(), (sg_p[m] == sg_t[m]).sum(),
                          ((fwd[m] == sg_t[m]) & (sg_p[m] == -sg_t[m])).sum(), ((fwd[m] != sg_t[m]) & (sg_p[m] == sg_t[m])).sum(),
                          (live).sum()]
        del feats
    print("%s s%d cfg=%s  flight direction vs ctl2 truth on visible live frames" % (variant, seed, c))
    print("speed>= dmin>= |vx|/v>= | share of live | dir right | ours right | dir right & ours wrong | dir wrong & ours right")
    for (v0, d0, a0), a in acc.items():
        n = max(a[0], 1)
        print("%5g %5g %5.1f | %.3f | %.3f | %.3f | %.4f | %.4f" % (v0, d0, a0, a[0] / a[5], a[1] / n, a[2] / n, a[3] / n, a[4] / n))


def cmd_starts(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "hi2t1")
    from possess import pipeline
    from possess.ballclean import clean_ball

    rows = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        ad = shape.attack_dir(feats["pos"], period).astype(float)
        bx, by, dh, da = feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"]
        sm = decode.ball_smoothed(bx, by)
        lag, r = (8, 2.0) if sm else (2, 1.0)
        cx, cy, _ = clean_ball(bx, by, np.minimum(dh, da), r, FPS)
        imp = decode.ball_impulse(cx, cy, lag=lag)["impulse"]
        vx, vy = ball_vel(bx, by)
        n = len(t)
        for s, e, k in R._spells(t, period):
            seg = lab[s:e]
            if (seg == k).mean() >= 0.5 or (seg == "D").mean() >= (np.isin(seg, ["H", "A"]) & (seg != k)).mean():
                continue
            sg = 1.0 if k == "H" else -1.0
            a, b = max(s - 15, 0), min(s + 16, n)
            do, dp = (dh, da) if k == "H" else (da, dh)
            w = vis[a:b]
            f1 = slice(s, min(s + 25, e))
            with np.errstate(invalid="ignore"):
                mo, mp = (np.nanmin(do[a:b][w]), np.nanmin(dp[a:b][w])) if w.any() else (np.nan, np.nan)
                io = imp[a:b] & (do[a:b] <= r)
                ip = imp[a:b] & (dp[a:b] <= r)
                rows.append(dict(L=(e - s) / FPS, vis=float(w.mean()), visall=float(vis[s:e].mean()), mo=mo, mp=mp, imp=bool(imp[a:b].any()),
                                 own_only=bool((io & ~ip).any()), opp_only=bool((ip & ~io).any()), duel=bool((io & ip).any()),
                                 fwd=float(np.nanmean(vx[f1] * ad[f1] * sg)) if np.isfinite(vx[f1]).any() else np.nan,
                                 spd=float(np.nanmean(np.hypot(vx[f1], vy[f1]))) if np.isfinite(vx[f1]).any() else np.nan,
                                 prevD=bool(s and t[s - 1] == "D")))
        del feats
    N = len(rows)
    g = lambda k: np.array([x[k] for x in rows], float)   # noqa: E731
    print("%s s%d cfg=%s: %d true possessions given to the other team (any visibility)" % (variant, seed, c, N))
    print("length med %.1f s; after a stoppage %.2f; start window (+-0.6 s) ball visible share: 0 -> %.2f, < 0.5 -> %.2f, 1 -> %.2f" % (
        np.median(g("L")), g("prevD").mean(), (g("vis") == 0).mean(), (g("vis") < 0.5).mean(), (g("vis") == 1).mean()))
    sel = g("vis") >= 0.5
    print("start window mostly visible: %d" % sel.sum())
    for nm, m in (("vis>=0.5", sel), ("all", np.ones(N, bool))):
        print("  [%s] impulse in window %.2f | own-only touch %.2f, opp-only %.2f, duel %.2f | own within 1 m %.2f, 2 m %.2f; opp within 1 m %.2f, 2 m %.2f" % (
            nm, g("imp")[m].mean(), g("own_only")[m].mean(), g("opp_only")[m].mean(), g("duel")[m].mean(),
            (g("mo")[m] <= 1).mean(), (g("mo")[m] <= 2).mean(), (g("mp")[m] <= 1).mean(), (g("mp")[m] <= 2).mean()))
        f, sp = g("fwd")[m], g("spd")[m]
        ok = np.isfinite(f)
        print("  [%s] first second: measurable %.2f; ball speed med %.1f m/s; flies own team's way %.2f (speed >= 8: n %d, %.2f; |fwd| >= 5: n %d, %.2f)" % (
            nm, ok.mean(), np.nanmedian(sp), (f[ok] > 0).mean(), (ok & (sp >= 8)).sum(), (f[ok & (sp >= 8)] > 0).mean(),
            (ok & (np.abs(f) >= 5)).sum(), (f[ok & (np.abs(f) >= 5)] > 0).mean()))


def cmd_frames(opt):
    """Where the wrong-team frames are (both truth and we name a team): by ball state."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "F")
    acc = {}
    tot = np.zeros(2)
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        vx, vy = ball_vel(feats["ball_x"], feats["ball_y"])
        sp = np.hypot(vx, vy)
        ad = shape.attack_dir(feats["pos"], period).astype(float)
        both = np.isin(t, ["H", "A"]) & np.isin(lab, ["H", "A"]) & (period > 0)
        wrong = both & (t != lab)
        own = np.where(t == "H", feats["d_home"], feats["d_away"])
        opp = np.where(t == "H", feats["d_away"], feats["d_home"])
        fwd = vx * ad * np.where(t == "H", 1.0, -1.0)   # ball's x speed in the TRUE team's attacking direction
        n = len(t)
        since = np.zeros(n)                              # seconds since the ball was last visible
        k = 0
        for i in range(n):
            k = 0 if vis[i] else k + 1
            since[i] = k / FPS
        # length of the true spell each frame belongs to
        L = np.zeros(n)
        for s0, e0, _ in R._spells(t, period):
            L[s0:e0] = (e0 - s0) / FPS
        with np.errstate(invalid="ignore"):
            st = {"vis own<=1 opp>1.5": vis & (own <= 1) & (opp > 1.5), "vis opp<=1 own>1.5": vis & (opp <= 1) & (own > 1.5),
                  "vis both<=1.5": vis & (own <= 1.5) & (opp <= 1.5),
                  "vis free slow(<4)": vis & (np.fmin(own, opp) > 1) & ~((own <= 1.5) & (opp <= 1.5)) & ~(sp >= 4) & ~((own <= 1) | (opp <= 1)),
                  "vis free fwd(>=4 own way)": vis & (np.fmin(own, opp) > 1) & (sp >= 4) & (fwd > 0),
                  "vis free back(>=4 other way)": vis & (np.fmin(own, opp) > 1) & (sp >= 4) & (fwd <= 0),
                  "hid <=1s": ~vis & (since <= 1), "hid 1-3s": ~vis & (since > 1) & (since <= 3), "hid >3s": ~vis & (since > 3),
                  "true spell <3s": L < 3, "true spell 3-8s": (L >= 3) & (L < 8), "true spell >=8s": L >= 8}
        for kk, m in st.items():
            a = acc.setdefault(kk, np.zeros(2))
            a += [(both & m).sum(), (wrong & m).sum()]
        tot += [both.sum(), wrong.sum()]
        del feats
    print("%s s%d cfg=%s: frames where truth and we both name a team: wrong team %.4f" % (variant, seed, c, tot[1] / tot[0]))
    print("%-30s share of frames | error rate | share of all wrong frames" % "state")
    for kk, a in acc.items():
        print("%-30s %.3f | %.3f | %.3f" % (kk, a[0] / tot[0], a[1] / max(a[0], 1), a[1] / tot[1]))


def cmd_pred(opt):
    """Prediction side: why each PREDICTED possession is not matched at IoU 0.5 (R.classify with the roles swapped)."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "F")
    cats = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        del feats
        for cat, L, vs in R.classify(lab, t, period, vis):
            b = "<3s" if L < 3 else ("3-8s" if L < 8 else (">=8s" if L < 20 else ">=20s"))
            d = cats.setdefault(cat, {"n": 0, "<3s": 0, "3-8s": 0, ">=8s": 0, ">=20s": 0, "hid": 0, "sec": 0.0})
            d["n"] += 1
            d[b] += 1
            d["hid"] += vs < 0.5
            d["sec"] += L
    tot = sum(d["n"] for d in cats.values())
    print("%s s%d cfg=%s: %d predicted possessions; by cause (n, share, < 3 s, 3-8 s, 8-20 s, >= 20 s, mostly hidden, total s)" % (variant, seed, c, tot))
    print("  (roles swapped: 'dead' = we say team where the truth is mostly D; 'merged_long' = the true spell is far longer than ours, i.e. we split it;")
    print("   'split_by_*' = our spell covers several true spells of that team)")
    for cat, d in sorted(cats.items(), key=lambda kv: -kv[1]["n"]):
        print("  %-14s %4d %.3f | %4d %4d %4d %4d | %4d | %6.0f" % (cat, d["n"], d["n"] / tot, d["<3s"], d["3-8s"], d[">=8s"], d[">=20s"], d["hid"], d["sec"]))


def sandwich_rows(variant, seed, c):
    """One row per PREDICTED team spell < 8 s that sits between two live spells of the other team (X k X):
    features of the inputs inside it and y = the truth's frames in it are mostly team k."""
    from possess import pipeline

    v = "" if variant == "orig" else variant
    rows = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        kw = {k: pipeline.DEC[k] for k in ("r_touch", "lag", "ball_clean", "smooth_touch", "vtx", "vtx_raw") if k in pipeline.DEC}
        ev = decode.touch_events(feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"], pos=feats.get("pos"), **kw)
        fly = decode.flight_dir(feats["ball_x"], feats["ball_y"], feats["pos"], period, 4.0, 8.0)
        dh, da = feats["d_home"], feats["d_away"]
        r = R._runs(lab, period)
        for i in range(1, len(r) - 1):
            s0, e0, k, p = r[i]
            a, b = r[i - 1], r[i + 1]
            if k not in "HA" or p <= 0 or e0 - s0 >= 8 * FPS or a[3] != p or b[3] != p or a[2] != b[2] or a[2] not in "HA":
                continue
            sg = 1.0 if k == "H" else -1.0
            own, opp = (dh, da) if k == "H" else (da, dh)
            sl = slice(s0, e0)
            w = vis[sl]
            with np.errstate(invalid="ignore"):
                rows.append(dict(name=name, y=float((t[sl] == k).mean() >= 0.5), yD=float((t[sl] == "D").mean() >= 0.5), L=(e0 - s0) / FPS, vis=float(w.mean()),
                                 own_t=int((ev[sl] == sg).sum()), opp_t=int((ev[sl] == -sg).sum()),
                                 fly=float((fly[sl] * sg).sum() / FPS), fly_abs=float(np.abs(fly[sl]).sum() / FPS),
                                 own1=float(((own[sl] <= 1) & ~(opp[sl] <= 1.5) & w).sum() / FPS), opp1=float(((opp[sl] <= 1) & ~(own[sl] <= 1.5) & w).sum() / FPS),
                                 nb=min(a[1] - a[0], b[1] - b[0]) / FPS))
        del feats
    return rows


def cmd_sand(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    c = opt.pop("cfg", "F")
    rows = sandwich_rows(variant, seed, c)
    g = lambda k: np.array([x[k] for x in rows], float)   # noqa: E731
    y = g("y")
    print("%s s%d cfg=%s: %d predicted sandwiched spells < 8 s; real (truth mostly that team) %.3f; truth mostly D %.3f" % (variant, seed, c, len(y), y.mean(), g("yD").mean()))

    def tab(nm, val, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (val >= lo) & (val < hi)
            out.append("[%g,%g) n %d real %.2f" % (lo, hi, m.sum(), y[m].mean() if m.any() else float("nan")))
        print("  %-8s " % nm + " | ".join(out))

    tab("L", g("L"), [0, 2, 3, 4, 6, 8])
    tab("own_t", g("own_t"), [0, 1, 2, 3, 99])
    tab("opp_t", g("opp_t"), [0, 1, 2, 99])
    tab("fly s", g("fly"), [-99, -0.5, -0.1, 0.1, 0.5, 1, 99])
    tab("own1 s", g("own1"), [0, 0.1, 0.5, 1, 2, 99])
    tab("opp1 s", g("opp1"), [0, 0.1, 0.5, 99])
    tab("vis", g("vis"), [0, 0.5, 0.9, 1.01])
    tab("nb s", g("nb"), [0, 2, 4, 8, 999])
    m = (g("own_t") == 0)
    tab("no-touch: fly", g("fly")[m] if False else np.where(m, g("fly"), np.nan), [-99, 0.1, 0.5, 1, 99])
    tab("no-touch: own1", np.where(m, g("own1"), np.nan), [0, 0.1, 0.5, 1, 99])
    tab("no-touch: L", np.where(m, g("L"), np.nan), [0, 2, 3, 4, 6, 8])
    m2 = m & (g("own1") < 0.5)
    tab("no-touch & own1<0.5: L", np.where(m2, g("L"), np.nan), [0, 2, 3, 4, 6, 8])
    tab("no-touch & own1<0.5: fly", np.where(m2, g("fly"), np.nan), [-99, 0.1, 0.5, 1, 99])


def cmd_deadposs(opt):
    """True possessions we call D (cause 'dead'): where they sit against our D run and the true stoppages."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "F")
    rows = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        sd = feats.get("shape_dead")
        del feats
        n = len(t)
        for s, e, k in R._spells(t, period):
            seg = lab[s:e]
            dead = (seg == "D").mean()
            if (seg == k).mean() >= 0.5 or dead < (np.isin(seg, ["H", "A"]) & (seg != k)).mean():
                continue
            i, j = s, e                                   # our D run around the spell
            while i > 0 and lab[i - 1] == "D":
                i -= 1
            while j < n and lab[j] == "D":
                j += 1
            tb, ta = (t[s - 1] if s else "X"), (t[e] if e < n else "X")
            pos = "inside" if (i < s - 12 and j > e + 12) else ("our D starts in it" if i >= s - 12 and j > e + 12 else ("our D ends in it" if j <= e + 12 and i < s - 12 else "D within"))
            rows.append(dict(L=(e - s) / FPS, vis=float(vis[s:e].mean()), pos=pos, tb=str(tb), ta=str(ta), dlen=(j - i) / FPS,
                             before=(s - i) / FPS, after=(j - e) / FPS, sd=float(np.nanmean(sd[s:e])) if sd is not None else np.nan))
    N = len(rows)
    print("%s s%d cfg=%s: %d true possessions we call D; length med %.1f s; mostly hidden %.2f" % (variant, seed, c, N, np.median([r["L"] for r in rows]), np.mean([r["vis"] < 0.5 for r in rows])))
    for pos in ("inside", "our D starts in it", "our D ends in it", "D within"):
        sel = [r for r in rows if r["pos"] == pos]
        if not sel:
            continue
        print("  %-20s n %3d | len med %.1f s | hidden %.2f | truth before D %.2f, after D %.2f | our D run med %.0f s; our D before the spell med %.1f s, after med %.1f s | stoppage log-odds in the spell med %+.2f" % (
            pos, len(sel), np.median([r["L"] for r in sel]), np.mean([r["vis"] < 0.5 for r in sel]), np.mean([r["tb"] == "D" for r in sel]),
            np.mean([r["ta"] == "D" for r in sel]), np.median([r["dlen"] for r in sel]), np.median([r["before"] for r in sel]), np.median([r["after"] for r in sel]),
            np.nanmedian([r["sd"] for r in sel])))


def cmd_flow(opt):
    """All-player flow direction (x velocity of the centroid of both teams, 1 s) as team evidence, by ball visibility."""
    import warnings

    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "F")
    lag = int(opt.pop("lag", 25))
    acc = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        pos = feats["pos"]
        ad = shape.attack_dir(pos, period).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cx = np.nanmean(pos[:, :, :, 0].reshape(len(t), -1), axis=1)
        n, h = len(t), lag // 2
        vx = np.full(n, np.nan)
        vx[h:n - lag + h] = (cx[lag:] - cx[:-lag]) * FPS / lag
        live = np.isin(t, ["H", "A"]) & np.isfinite(vx) & (ad != 0) & (period > 0)
        sg_t = np.where(t == "H", 1.0, -1.0)
        sg_p = np.where(lab == "H", 1.0, np.where(lab == "A", -1.0, 0.0))
        fwd = np.sign(vx * ad)
        for nm, mk in (("visible", vis), ("hidden", ~vis)):
            for v0 in (0.5, 1.0, 1.5, 2.0, 3.0):
                m = live & mk & (np.abs(vx) >= v0)
                a = acc.setdefault((nm, v0), np.zeros(6))
                a += [m.sum(), (fwd[m] == sg_t[m]).sum(), (sg_p[m] == sg_t[m]).sum(), ((fwd[m] == sg_t[m]) & (sg_p[m] == -sg_t[m])).sum(),
                      ((fwd[m] != sg_t[m]) & (sg_p[m] == sg_t[m])).sum(), (live & mk).sum()]
        del feats
    print("%s s%d cfg=%s  player flow direction (lag %d fr) vs ctl2 truth on true-live frames" % (variant, seed, c, lag))
    print("ball     |vx|>= | share | flow right | ours right | flow right & ours wrong | flow wrong & ours right")
    for (nm, v0), a in acc.items():
        n = max(a[0], 1)
        print("%-8s %4.1f | %.3f | %.3f | %.3f | %.4f | %.4f" % (nm, v0, a[0] / a[5], a[1] / n, a[2] / n, a[3] / n, a[4] / n))


def cmd_keep(opt):
    """Row 36: T95's touch requirement, with a sustained own-way flight counting like a touch. Done outside the
    decoder (T95's code path is under QA): decode with the flight only, then ctl_truth.relabel with the touch
    events plus one event per run of >= `run` s of full-strength flight."""
    from possess import ctl_truth, pipeline

    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    runs = [float(x) for x in str(opt.pop("run", "0.3,0.5,1.0")).split(",")]
    labs, tv = {}, {}
    cfgs = ["F", "check"] + ["keep%g" % r for r in runs]
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        labs["F", name] = a4_short.decode_one(feats, period, R.CONFIGS["F"])
        raw = a4_short.decode_one(feats, period, R.FLY)
        kw = {k: pipeline.DEC[k] for k in ("r_touch", "lag", "ball_clean", "smooth_touch", "vtx", "vtx_raw", "duel") if k in pipeline.DEC}
        ev = np.zeros(len(period))
        fly = decode.flight_dir(feats["ball_x"], feats["ball_y"], feats["pos"], period, 4.0, 8.0)
        i, n = 0, len(period)
        while i < n:                                     # per period block, as decode_halves does
            if period[i] <= 0:
                i += 1
                continue
            j = i
            while j < n and period[j] == period[i]:
                j += 1
            ev[i:j] = decode.touch_events(**{k: feats[k][i:j] for k in ("ball_x", "ball_y", "d_home", "d_away", "visible")}, pos=feats["pos"][i:j], **kw)
            i = j
        labs["check", name] = ctl_truth.relabel(raw, period, 2.0, FPS, passes=ev, min_pass=1)
        for r in runs:
            ev2 = ev.copy()
            for sg in (1.0, -1.0):
                full = fly == sg
                c = np.zeros(n, int)
                for i in range(1, n):
                    c[i] = c[i - 1] + 1 if full[i] else 0
                ev2[(c == int(r * FPS)) & (ev2 == 0)] = sg
            labs["keep%g" % r, name] = ctl_truth.relabel(raw, period, 2.0, FPS, passes=ev2, min_pass=1)
        print(name, "check == F:", bool((labs["check", name] == labs["F", name]).all()), flush=True)
        del feats
    R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  flight counts as a touch in the T95 pass" % (variant, seed))


def flight_segments(fly, vis, dh, da, max_wait=1.5, r_in=1.0, r_out=1.5):
    """[(start, end, sign, receiver)]: runs of flight evidence of one sign and the team (+1 / -1, 0 = none) that
    stands alone within r_in of the ball (other team beyond r_out) first after the run, within max_wait s, ball visible."""
    n = len(fly)
    sg = np.sign(fly)
    brk = np.flatnonzero(sg[1:] != sg[:-1]) + 1
    b = np.concatenate([[0], brk, [n]])
    out = []
    with np.errstate(invalid="ignore"):
        near = np.fmin(dh, da) <= r_in
        team = np.where((dh <= r_in) & ~(da <= r_out), 1.0, np.where((da <= r_in) & ~(dh <= r_out), -1.0, 0.0))
    for s0, e0 in zip(b[:-1], b[1:]):
        if sg[s0] == 0:
            continue
        rec = 0.0
        for j in range(e0, min(n, e0 + int(max_wait * FPS))):
            if not vis[j]:
                break
            if near[j]:
                rec = team[j]
                break
        out.append((int(s0), int(e0), float(sg[s0]), float(rec)))
    return out


def cmd_recv(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    acc = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        vis = np.asarray(feats["visible"], bool)
        fly = decode.flight_dir(feats["ball_x"], feats["ball_y"], feats["pos"], period, 4.0, 8.0)
        sg_t = np.where(t == "H", 1.0, np.where(t == "A", -1.0, 0.0))
        for s0, e0, sg, rec in flight_segments(fly, vis, feats["d_home"], feats["d_away"]):
            L = (e0 - s0) / FPS
            m = sg_t[s0:e0]
            if (m != 0).mean() < 0.5 or L < 0.3:
                continue
            tt = 1.0 if (m == 1).sum() >= (m == -1).sum() else -1.0
            key = ("none" if rec == 0 else ("receiver = flight team" if rec == sg else "receiver = other team"), "<1s" if L < 1 else ">=1s")
            a = acc.setdefault(key, np.zeros(3))
            a += [1, tt == sg, tt == rec]
        del feats
    print("%s s%d: flight runs >= 0.3 s (ctl2 truth of the run = majority team)" % (variant, seed))
    for k, a in sorted(acc.items()):
        print("  %-24s %-5s n %4d | truth = flight team %.3f | truth = receiver %.3f" % (k[0], k[1], a[0], a[1] / a[0], a[2] / a[0]))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], a4_short.kv(sys.argv[2:])
    {"flight": cmd_flight, "starts": cmd_starts, "frames": cmd_frames, "pred": cmd_pred, "sand": cmd_sand, "deadposs": cmd_deadposs, "flow": cmd_flow, "keep": cmd_keep, "recv": cmd_recv}[cmd](rest)
