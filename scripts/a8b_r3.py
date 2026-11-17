"""Envelope L8-autonomous-fable round 3: the uncertain zone (hidden blocks of 3-8 s) and short possessions.

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r3.py hid bench=real2 seed=0 [cfg=F]
    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r3.py hidrule bench=real2 seed=0 [cfg=F]

hid:     hidden-ball blocks by length: true D share (ctl T=2 truth), our D share, D precision / recall,
         and for the 3-8 s blocks the confusion by what surrounds the block.
hidrule: scores label post-passes on short hidden blocks (see RULES) against cfg.
Tune matches only; no held-out, no public truth, nothing of the provided match.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a8b_run as R  # noqa: E402
import tune_shape  # noqa: E402

FPS = 25.0
BINS = [(0, 1), (1, 3), (3, 8), (8, 20), (20, 1e9)]


def blocks(mask, period):
    """[a, b) runs of True inside one period (> 0)."""
    m = np.asarray(mask, bool) & (np.asarray(period) > 0)
    brk = np.flatnonzero(np.diff(np.asarray(period, int)) != 0) + 1
    out = []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        cuts = [a] + [c for c in brk if a < c < b] + [b]
        out += list(zip(cuts[:-1], cuts[1:]))
    return out


def line_dist(x, y):
    return min(52.5 - abs(x), 34.0 - abs(y))


def block_rows(variant, seed, c):
    """One row per hidden block of the three tune matches."""
    rows = []
    v = "" if variant == "orig" else variant
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t2 = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis, bx, by = feats["visible"], feats["ball_x"], feats["ball_y"]
        sd = feats.get("shape_dead")
        n = len(lab)
        for a, b in blocks(~vis, period):
            L = (b - a) / FPS
            td, pd = t2[a:b] == "D", lab[a:b] == "D"
            pa, pb = (a - 1 if a > 0 and period[a - 1] == period[a] else None), (b if b < n and period[b] == period[a] else None)
            row = {"m": name, "a": int(a), "b": int(b), "L": L, "n": b - a, "td": int(td.sum()), "pd": int(pd.sum()), "tp": int((td & pd).sum()),
                   "lab_before": lab[pa] if pa is not None else "X", "lab_after": lab[pb] if pb is not None else "X",
                   "tr_before": t2[pa] if pa is not None else "X", "tr_after": t2[pb] if pb is not None else "X",
                   "ld_in": line_dist(bx[pa], by[pa]) if pa is not None else np.nan, "ld_out": line_dist(bx[pb], by[pb]) if pb is not None else np.nan,
                   "disp": float(np.hypot(bx[pb] - bx[pa], by[pb] - by[pa])) if pa is not None and pb is not None else np.nan,
                   "sd": float(np.nanmean(sd[a:b])) if sd is not None else np.nan,
                   "team_ok": int(((t2[a:b] == lab[a:b]) & ~td & ~pd).sum()), "both_live": int((~td & ~pd).sum())}
            rows.append(row)
        del feats
    return rows


def cmd_hid(opt):
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "F")
    rows = block_rows(variant, seed, c)
    tot = sum(r["n"] for r in rows)
    print("%s s%d cfg=%s  hidden blocks (ctl2 truth)" % (variant, seed, c))
    print("%-8s %5s %6s | %6s %6s | %5s %5s | %7s" % ("len s", "n", "frames", "true D", "our D", "prec", "rec", "team ok"))
    for lo, hi in BINS:
        rs = [r for r in rows if lo <= r["L"] < hi]
        n = sum(r["n"] for r in rs)
        if not n:
            continue
        td, pd, tp = (sum(r[k] for r in rs) for k in ("td", "pd", "tp"))
        bl = sum(r["both_live"] for r in rs)
        print("%-8s %5d %6.3f | %6.3f %6.3f | %5.2f %5.2f | %7.3f" % ("%g-%g" % (lo, min(hi, 999)), len(rs), n / tot, td / n, pd / n,
                                                                    tp / max(pd, 1), tp / max(td, 1), sum(r["team_ok"] for r in rs) / max(bl, 1)))
    for lo, hi in ((1, 3), (3, 8), (8, 20)):
        rs = [r for r in rows if lo <= r["L"] < hi]
        print("blocks %g-%g s: block-level (majority) confusion and context" % (lo, hi))
        for key, f in (("all", lambda r: True),
                       ("live both sides (ours)", lambda r: r["lab_before"] in "HA" and r["lab_after"] in "HA"),
                       ("D one side (ours)", lambda r: (r["lab_before"] == "D") != (r["lab_after"] == "D")),
                       ("D both sides (ours)", lambda r: r["lab_before"] == "D" and r["lab_after"] == "D"),
                       ("interior in+out (>5 m), live both", lambda r: r["lab_before"] in "HA" and r["lab_after"] in "HA" and r["ld_in"] > 5 and r["ld_out"] > 5),
                       ("near line in or out (<=5 m), live both", lambda r: r["lab_before"] in "HA" and r["lab_after"] in "HA" and not (r["ld_in"] > 5 and r["ld_out"] > 5)),
                       ("live both, disp/L >= 3 m/s", lambda r: r["lab_before"] in "HA" and r["lab_after"] in "HA" and r["disp"] / r["L"] >= 3),
                       ("live both, disp/L < 3 m/s", lambda r: r["lab_before"] in "HA" and r["lab_after"] in "HA" and r["disp"] / r["L"] < 3),
                       ("live both, same team", lambda r: r["lab_before"] in "HA" and r["lab_after"] == r["lab_before"]),
                       ("live both, other team", lambda r: r["lab_before"] in "HA" and r["lab_after"] in "HA" and r["lab_after"] != r["lab_before"])):
            q = [r for r in rs if f(r)]
            pD = [r for r in q if r["pd"] > 0.5 * r["n"]]
            pL = [r for r in q if r["pd"] <= 0.5 * r["n"]]
            tD = lambda rr: sum(1 for r in rr if r["td"] > 0.5 * r["n"])   # noqa: E731
            print("  %-40s n %4d | we say D %4d (truly D %4d) | we say live %4d (truly D %4d) | D frames ours %5d true %5d" % (
                key, len(q), len(pD), tD(pD), len(pL), tD(pL), sum(r["pd"] for r in q), sum(r["td"] for r in q)))
    with open(os.path.join(ROOT, "eval", "results", "a8b_r3hid_%s_s%d_%s.json" % (variant, seed, c)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "cfg": c, "rows": rows}, fh, default=float)


def cmd_hidprov(opt):
    """Provided match INPUT only (no truth): hidden blocks by length, our D share, and where the ball
    was last / first seen (distance to the nearest line, displacement speed across the block)."""
    import a8_tags
    from possess import pipeline

    c = opt.pop("cfg", "F")
    _, _, f, per, _ = a8_tags.challenge_default()
    lab = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
    bx, by = f["ball_x"], f["ball_y"]
    vis, n = ~np.isnan(bx), len(per)
    rows = []
    for a, b in blocks(~vis, per):
        pa, pb = (a - 1 if a > 0 and per[a - 1] == per[a] else None), (b if b < n and per[b] == per[a] else None)
        rows.append({"L": (b - a) / FPS, "n": b - a, "pd": int((lab[a:b] == "D").sum()),
                     "ld_in": line_dist(bx[pa], by[pa]) if pa is not None else np.nan, "ld_out": line_dist(bx[pb], by[pb]) if pb is not None else np.nan,
                     "spd": float(np.hypot(bx[pb] - bx[pa], by[pb] - by[pa])) / ((b - a) / FPS) if pa is not None and pb is not None else np.nan})
    print("provided input, cfg=%s: hidden blocks" % c)
    for lo, hi in BINS:
        rs = [r for r in rows if lo <= r["L"] < hi]
        if not rs:
            continue
        nn = sum(r["n"] for r in rs)
        sayD = [r for r in rs if r["pd"] > 0.5 * r["n"]]
        cnt = lambda rr, k, x0, x1: sum(1 for r in rr if x0 <= r[k] < x1)   # noqa: E731
        print("%g-%g s: n %d, our D share %.3f, blocks we call D %d | ld_in <0 / 0-1 / 1-2 / 2-5 / >=5: %s | ld_out: %s | spd <0.5 / 0.5-1 / 1-2 / 2-3 / >=3: %s | of those we call live: ld_in %s ld_out %s spd %s" % (
            lo, min(hi, 999), len(rs), sum(r["pd"] for r in rs) / nn, len(sayD),
            [cnt(rs, "ld_in", x0, x1) for x0, x1 in ((-99, 0), (0, 1), (1, 2), (2, 5), (5, 99))],
            [cnt(rs, "ld_out", x0, x1) for x0, x1 in ((-99, 0), (0, 1), (1, 2), (2, 5), (5, 99))],
            [cnt(rs, "spd", x0, x1) for x0, x1 in ((0, .5), (.5, 1), (1, 2), (2, 3), (3, 999))],
            [cnt([r for r in rs if r not in sayD], "ld_in", x0, x1) for x0, x1 in ((-99, 0), (0, 1), (1, 2), (2, 5), (5, 99))],
            [cnt([r for r in rs if r not in sayD], "ld_out", x0, x1) for x0, x1 in ((-99, 0), (0, 1), (1, 2), (2, 5), (5, 99))],
            [cnt([r for r in rs if r not in sayD], "spd", x0, x1) for x0, x1 in ((0, .5), (.5, 1), (1, 2), (2, 3), (3, 999))]))


def behind_ball(pos, bx, ad):
    """(fa - fh): share of away players goal-side of the ball minus the same for home; ad = home attack sign.
    Positive = home ahead of the ball more than away = home has it (the defending team stands behind the ball)."""
    with np.errstate(invalid="ignore"):
        rel_h = (bx[:, None] - pos[:, 0, :, 0]) * ad[:, None]      # > 0: home player between the ball and his own goal
        rel_a = (pos[:, 1, :, 0] - bx[:, None]) * ad[:, None]
        fh = np.nansum(rel_h > 0, axis=1) / np.maximum(np.isfinite(rel_h).sum(axis=1), 1)
        fa = np.nansum(rel_a > 0, axis=1) / np.maximum(np.isfinite(rel_a).sum(axis=1), 1)
    return np.where(np.isfinite(bx) & (ad != 0), fa - fh, 0.0)


def cmd_side(opt):
    """Is 'the defending team stands goal-side of the ball' team evidence the decoder lacks (visible frames)."""
    from possess import shape
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "F")
    v = "" if variant == "orig" else variant
    acc = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        ad = shape.attack_dir(feats["pos"], period).astype(float)
        g = behind_ball(feats["pos"], feats["ball_x"], ad)
        sgn = np.where(t == "H", 1.0, -1.0)
        both = np.isin(t, ["H", "A"]) & np.isin(lab, ["H", "A"]) & (period > 0) & feats["visible"]
        ok = both & (t == lab)
        L = np.zeros(len(t))
        age = np.zeros(len(t))
        for s0, e0, _ in R._spells(t, period):
            L[s0:e0] = (e0 - s0) / FPS
            age[s0:e0] = np.arange(e0 - s0) / FPS
        gs = g * sgn                                     # > 0: the cue names the true team
        dh, da = feats["d_home"], feats["d_away"]
        lag = 8
        sp = np.full(len(t), np.nan)
        sp[lag // 2:len(t) - lag // 2] = np.hypot(feats["ball_x"][lag:] - feats["ball_x"][:-lag], feats["ball_y"][lag:] - feats["ball_y"][:-lag]) * FPS / lag
        duel = (dh <= 1.5) & (da <= 1.5)
        alone = ((dh <= 1) | (da <= 1)) & ~duel
        free = ~duel & ~alone
        for key, m in (("one team alone <=1 m", both & alone), ("both <=1.5 m", both & duel), ("free, < 4 m/s", both & free & ~(sp >= 4)), ("free, >= 4 m/s", both & free & (sp >= 4)),
                       ("all", both), ("true spell <3s", both & (L < 3)), ("3-8s", both & (L >= 3) & (L < 8)), (">=8s", both & (L >= 8)),
                       ("first 2 s of a true spell", both & (age < 2)), ("after 2 s", both & (age >= 2))):
            for thr in (0.0, 0.2, 0.4):
                a = acc.setdefault((key, thr), np.zeros(6))
                fire = m & (np.abs(g) > thr)
                a += [m.sum(), fire.sum(), (fire & (gs > 0)).sum(), (fire & ~ok).sum(), (fire & ~ok & (gs > 0)).sum(), (fire & ok & (gs > 0)).sum()]
        del feats
    print("%s s%d cfg=%s  goal-side cue on visible frames where truth and we name a team" % (variant, seed, c))
    print("%-26s thr | fires | cue right | when we are WRONG: cue right | when we are right: cue agrees" % "frames")
    for (key, thr), a in acc.items():
        print("%-26s %.1f | %.3f | %.3f | %.3f (n %d) | %.3f" % (key, thr, a[1] / a[0], a[2] / max(a[1], 1), a[4] / max(a[3], 1), a[3], a[5] / max(a[1] - a[3], 1)))


SIDE = {  # name: (weight, ramp start, ramp end, 0 = every visible frame / 1 = only when no player is within 1 m)
    "s1": (0.1, 0.2, 0.6, 0), "s2": (0.2, 0.2, 0.6, 0), "s4": (0.4, 0.2, 0.6, 0), "s2f": (0.2, 0.2, 0.6, 1), "s4f": (0.4, 0.2, 0.6, 1),
    "s7f": (0.7, 0.2, 0.6, 1), "s2a": (0.2, 0.0, 0.4, 0), "s4a": (0.4, 0.0, 0.4, 0), "s2b": (0.2, 0.4, 0.8, 0), "s4b": (0.4, 0.4, 0.8, 0),
}


def side_feats(feats, period, w, g0, g1, mode):
    """Prototype through the opt-in tag hook (viterbi tag_team / w_tag / tag_vis = 1): goal-side cue as team evidence."""
    from possess import shape
    g = behind_ball(feats["pos"], feats["ball_x"], shape.attack_dir(feats["pos"], period).astype(float))
    e = np.sign(g) * np.clip((np.abs(g) - g0) / (g1 - g0), 0.0, 1.0)
    if mode == 1:
        e = np.where((feats["d_home"] > 1) & (feats["d_away"] > 1), e, 0.0)
    return dict(feats, tag_team=e), dict(w_tag=w, tag_vis=1.0)


def cmd_siderun(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    base = opt.pop("base", "F")
    cfgs = [base] + str(opt.pop("cfg", ",".join(SIDE))).split(",")
    tag = opt.pop("tag", "r3side")
    labs, tv = {}, {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        for c in cfgs:
            if c in SIDE:
                f2, kw = side_feats(feats, period, *SIDE[c])
                labs[c, name] = a4_short.decode_one(f2, period, dict(R.CONFIGS[base], **kw))
            else:
                labs[c, name] = a4_short.decode_one(feats, period, R.CONFIGS[c])
        del feats
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  goal-side cue on top of %s" % (variant, seed, base))
    with open(os.path.join(ROOT, "eval", "results", "a8b_%s_%s_s%d.json" % (tag, variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "base": base, "configs": {c: SIDE.get(c, c) for c in cfgs}, "results": res, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_doracle(opt):
    """Oracle: give the true D to the stoppages of one length bin only (and split visible / hidden): where is the missed-D loss."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "F")
    v = "" if variant == "orig" else variant
    bins = [(0, 1), (1, 2), (2, 5), (5, 15), (15, 1e9)]
    labs, tv, names = {}, {}, list(tune_shape.TUNE)
    keys = ["F"] + ["D %g-%g" % (a, min(b, 999)) for a, b in bins] + ["D>=2 vis", "D>=2 hid", "D>=2 all", "noFalseD"]
    cnt = {k: 0 for k in keys}
    for name in names:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        t = tv[name]["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        labs["F", name] = lab
        runs = blocks(t == "D", period)
        for k in keys[1:]:
            labs[k, name] = lab.copy()
        for a, b in runs:
            L = (b - a) / FPS
            hid = (~vis[a:b]).mean() > 0.5
            for (lo, hi), k in zip(bins, keys[1:]):
                if lo <= L < hi:
                    labs[k, name][a:b] = "D"
                    cnt[k] += 1
            if L >= 2:
                labs["D>=2 all", name][a:b] = "D"
                labs["D>=2 hid" if hid else "D>=2 vis", name][a:b] = "D"
                cnt["D>=2 hid" if hid else "D>=2 vis"] += 1
        m = (lab == "D") & np.isin(t, ["H", "A"])
        labs["noFalseD", name] = np.where(m, t, lab)
        del feats
    print("stoppage counts per oracle:", cnt)
    R.score_table(labs, tv, keys, names, "%s s%d  missed-D oracles by stoppage length (labels %s)" % (variant, seed, c))


def cmd_edges(opt):
    """Entry / exit lag of our D runs against the true stoppages (>= min_s, matched by largest overlap), split by
    whether the ball is visible at the true edge, with an oracle that repairs one kind of edge only."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "F")
    min_s, win = float(opt.pop("min", 5.0)), int(float(opt.pop("win", 4.0)) * FPS)
    v = "" if variant == "orig" else variant
    kinds = ["entry vis", "entry hid", "exit vis", "exit hid"]
    labs, tv, names, lags = {}, {}, list(tune_shape.TUNE), {k: [] for k in kinds}
    for name in names:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        t = tv[name]["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        labs["F", name] = lab
        for k in kinds + ["all edges"]:
            labs[k, name] = lab.copy()
        pruns = blocks(lab == "D", period)
        for a, b in blocks(t == "D", period):
            if (b - a) / FPS < min_s:
                continue
            ov = [(min(b, q) - max(a, p0), p0, q) for p0, q in pruns if min(b, q) > max(a, p0)]
            if not ov:
                continue
            _, p0, q = max(ov)
            for kind, te, pe in (("entry", a, p0), ("exit", b, q)):
                if abs(pe - te) > win or pe == te:
                    if pe == te:
                        lags[kind + (" vis" if vis[min(te, len(vis) - 1)] else " hid")].append(0.0)
                    continue
                k = kind + (" vis" if vis[min(te, len(vis) - 1)] else " hid")
                lags[k].append((pe - te) / FPS)
                lo, hi = min(te, pe), max(te, pe)
                if period[lo] == period[hi - 1]:
                    labs[k, name][lo:hi] = t[lo:hi]
                    labs["all edges", name][lo:hi] = t[lo:hi]
        del feats
    for k in kinds:
        x = np.array(lags[k])
        if len(x):
            print("%-10s n %4d  lag s (ours - true) p10 %+.2f p25 %+.2f median %+.2f p75 %+.2f p90 %+.2f | exact %.2f, late > 0.5 s %.2f, early < -0.5 s %.2f" % (
                k, len(x), *np.percentile(x, [10, 25, 50, 75, 90]), (x == 0).mean(), (x > 0.5).mean(), (x < -0.5).mean()))
    R.score_table(labs, tv, ["F"] + kinds + ["all edges"], names, "%s s%d  stoppage-edge oracles (true stoppages >= %g s, edges within %g s)" % (variant, seed, min_s, win / FPS))


def cross_frame(bx, by, pe, back, margin=0.0):
    """Last frame f in (pe - back, pe + back] where the visible ball goes from inside the pitch to beyond a line (by `margin` m); -1 if none."""
    lo, hi = max(1, pe - back), min(len(bx) - 1, pe + back)
    with np.errstate(invalid="ignore"):
        ld = np.minimum(52.5 - np.abs(bx[lo - 1:hi + 1]), 34.0 - np.abs(by[lo - 1:hi + 1]))
    c = np.flatnonzero((ld[:-1] >= -margin) & (ld[1:] < -margin))
    return lo + int(c[np.argmin(np.abs(lo + c - pe))]) if len(c) else -1


def cmd_entry(opt):
    """True stoppage entries with the ball visible: does the frame where the ball crosses a line date them better than we do."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "F")
    v = "" if variant == "orig" else variant
    back = int(float(opt.pop("back", 3.0)) * FPS)
    rows = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        bx, by = feats["ball_x"], feats["ball_y"]
        truns = blocks(t == "D", period)
        for p0, q in blocks(lab == "D", period):
            ov = [(min(b, q) - max(a, p0), a, b) for a, b in truns if min(b, q) > max(a, p0)]
            cf = cross_frame(bx, by, p0, back)
            te = max(ov)[1] if ov else None
            rows.append((cf >= 0, te is not None, (p0 - te) / FPS if te is not None else np.nan, (cf - te) / FPS if te is not None and cf >= 0 else np.nan,
                         (cf - p0) / FPS if cf >= 0 else np.nan, (q - p0) / FPS))
        del feats
    r = np.array(rows, float)
    has = r[:, 0] > 0
    print("%s s%d cfg=%s: our D runs %d; with a line crossing within %.1f s of our entry %d (%.2f); of those overlapping a true stoppage %.2f (others %.2f)" % (
        variant, seed, c, len(r), back / FPS, has.sum(), has.mean(), r[has, 1].mean(), r[~has, 1].mean()))
    m = has & (r[:, 1] > 0)
    q = lambda x: "p10 %+.2f p25 %+.2f med %+.2f p75 %+.2f p90 %+.2f | mean abs %.2f" % (*np.percentile(x, [10, 25, 50, 75, 90]), np.abs(x).mean())   # noqa: E731
    print("  our entry - true entry :", q(r[m, 2]))
    print("  crossing  - true entry :", q(r[m, 3]))
    print("  crossing  - our entry  :", q(r[m, 4]))
    print("  without a crossing, our entry - true entry:", q(r[~has & (r[:, 1] > 0), 2]))


def cross_snap(lab, period, bx, by, back_s=3.0, off=5, stats=None):
    """Prototype post-pass: the entry of each of our D runs moves to the frame where the visible ball crossed a
    line (nearest crossing within back_s of our entry) plus `off` frames. Never removes a whole spell or D run."""
    out = lab.copy()
    back = int(back_s * FPS)
    for p0, q in blocks(lab == "D", period):
        if p0 == 0 or period[p0 - 1] != period[p0] or lab[p0 - 1] not in ("H", "A"):
            continue
        cf = cross_frame(bx, by, p0, back)
        if cf < 0:
            continue
        ne = cf + off
        a0 = p0 - 1                                      # start of the team spell before the D run
        while a0 > 0 and lab[a0 - 1] == lab[p0 - 1] and period[a0 - 1] == period[p0]:
            a0 -= 1
        ne = min(max(ne, a0 + 1), q - 1)
        if ne < p0:
            out[ne:p0] = "D"
        elif ne > p0:
            out[p0:ne] = lab[p0 - 1]
        if stats is not None:
            stats.append((ne - p0) / FPS)
    return out


CROSS = {"c3o5": (3.0, 5), "c3o0": (3.0, 0), "c15o5": (1.5, 5), "c3o3": (3.0, 3), "c5o5": (5.0, 5), "c1o5": (1.0, 5)}


def cmd_crossrun(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    base = opt.pop("base", "F")
    cfgs = [base] + str(opt.pop("cfg", ",".join(CROSS))).split(",")
    labs, tv = {}, {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        labs[base, name] = a4_short.decode_one(feats, period, R.CONFIGS[base])
        for c in cfgs[1:]:
            labs[c, name] = cross_snap(labs[base, name], period, feats["ball_x"], feats["ball_y"], *CROSS[c])
        del feats
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  D entry at the line crossing, on top of %s" % (variant, seed, base))
    with open(os.path.join(ROOT, "eval", "results", "a8b_%s_%s_s%d.json" % (opt.pop("tag", "r3cross"), variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "base": base, "configs": {c: CROSS.get(c, c) for c in cfgs}, "results": res, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_crossmiss(opt):
    """Visible line crossings with NO stoppage of ours nearby: are they true stoppages we miss, and how long."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "F")
    v = "" if variant == "orig" else variant
    for margin in (0.0, 0.5, 1.0):
        tot = cov = unc = uncD = 0
        dl, outl = [], []
        for name in tune_shape.TUNE:
            truth, period, feats = a4_short.load(name, v, seed)
            t = R.truths(truth, period)["ctl2"]
            lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
            bx, by = feats["ball_x"], feats["ball_y"]
            n = len(t)
            with np.errstate(invalid="ignore"):
                ld = np.fmin(52.5 - np.abs(bx), 34.0 - np.abs(by))
                cr = np.flatnonzero((ld[:-1] >= -margin) & (ld[1:] < -margin) & (period[1:] > 0)) + 1
            for cf in cr:
                tot += 1
                if (lab[max(0, cf - 25):min(n, cf + 75)] == "D").any():
                    cov += 1
                    continue
                unc += 1
                isD = (t[cf:min(n, cf + 25)] == "D").any()
                uncD += isD
                if isD:
                    a = cf + int(np.argmax(t[cf:cf + 25] == "D"))
                    b = a
                    while b < n and t[b] == "D":
                        b += 1
                    dl.append((b - a) / FPS)
                    k = cf                                # frames until the ball is seen inside again
                    while k < n and not (ld[k] >= 0):
                        k += 1
                    outl.append((k - b) / FPS)
            del feats
        print("%s s%d margin %.1f m: crossings %d, with our D nearby %d, without %d of which truly a stoppage %d (%.2f)" % (variant, seed, margin, tot, cov, unc, uncD, uncD / max(unc, 1)), end="")
        if dl:
            print(" | true D length p25 %.1f med %.1f p75 %.1f s | ball seen inside again minus true restart: p25 %+.1f med %+.1f p75 %+.1f s" % (
                *np.percentile(dl, [25, 50, 75]), *np.percentile(outl, [25, 50, 75])))
        else:
            print()


def cmd_xtransfer(opt):
    """Provided match INPUT only (no truth): what the crossing snap changes against cfg `base` (our own output)."""
    import a8_tags
    from possess import ctl_truth, edges, pipeline

    base = opt.pop("base", "F")
    _, _, f, per, _ = a8_tags.challenge_default()
    play = per > 0
    lab0 = pipeline.label(f, period=per, dec_kw=R.CONFIGS[base])
    nD = len(blocks(lab0 == "D", per))
    print("provided input, base %s: D runs %d, possessions %d" % (base, nD, ctl_truth.n_possessions(lab0, per)))
    for c in str(opt.pop("cfg", "Fx3,Fx5")).split(","):
        lab = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
        mv = []
        if R.CONFIGS[c].get("ed_xdead"):
            edges.cross_dead(lab0, per, f["ball_x"], f["ball_y"], moves=mv, min_s=R.CONFIGS[c].get("ed_xdead_min", 0.0))
            print("   new stoppages (frame, seconds):", [(a, round((k - a) / FPS, 1)) for _, a, k in mv])
        else:
            edges.cross_entry(lab0, per, f["ball_x"], f["ball_y"], R.CONFIGS[c]["ed_cross"], R.CONFIGS[c].get("ed_cross_off", 5), moves=mv)
        d = np.array([(ne - a) / FPS for _, a, ne in mv]) if mv else np.zeros(1)
        print("%s: frames changed %.4f, D runs moved %d of %d, shift s p10 %+.2f p25 %+.2f med %+.2f p75 %+.2f p90 %+.2f, possessions %d, window possessions %d -> %d" % (
            c, ((lab != lab0) & play).sum() / play.sum(), len(mv), nD, *np.percentile(d, [10, 25, 50, 75, 90]), ctl_truth.n_possessions(lab, per),
            ctl_truth.n_possessions(lab0[11186:18686], per[11186:18686]), ctl_truth.n_possessions(lab[11186:18686], per[11186:18686])))


def cmd_rexit(opt):
    """Our stoppage exits by what the ball does there: exit exactly at the ball's reappearance (hidden just before,
    visible at the exit), exit on a hidden frame, exit on a visible frame; lag against the true restart."""
    variant, seed, c = opt.pop("bench", "real2"), int(opt.pop("seed", 0)), opt.pop("cfg", "F")
    v = "" if variant == "orig" else variant
    lags = {"at reappearance": [], "hidden": [], "visible": []}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = R.truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, R.CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        truns = blocks(t == "D", period)
        n = len(t)
        for p0, q in blocks(lab == "D", period):
            if q >= n or period[q] != period[p0] or lab[q] not in ("H", "A") or (q - p0) < 3 * FPS:
                continue
            ov = [(min(b, q) - max(a, p0), a, b) for a, b in truns if min(b, q) > max(a, p0)]
            if not ov:
                continue
            te = max(ov)[2]
            k = "hidden" if not vis[q] else ("at reappearance" if not vis[max(q - 13, p0):q].all() else "visible")
            lags[k].append((q - te) / FPS)
        del feats
    for k, x in lags.items():
        x = np.array(x)
        if len(x):
            print("%s s%d exit %-16s n %4d  ours - true restart: p10 %+.2f p25 %+.2f med %+.2f p75 %+.2f p90 %+.2f | late > 0.5 s %.2f, early < -0.5 s %.2f" % (
                variant, seed, k, len(x), *np.percentile(x, [10, 25, 50, 75, 90]), (x > 0.5).mean(), (x < -0.5).mean()))


def cmd_spd(opt):
    """Provided match INPUT only (no truth): how much of each SET PIECES clip's first seconds we already call D."""
    import a8_tags
    from possess import pipeline

    c = opt.pop("cfg", "F")
    tags, home, f, per, _ = a8_tags.challenge_default()
    lab = pipeline.label(f, period=per, dec_kw=R.CONFIGS[c])
    vis = ~np.isnan(f["ball_x"])
    n = len(lab)
    rows = {}
    for t in tags:
        if t["code"] != "SET PIECES" or t["type"].startswith("Kick Off (Start"):
            continue
        a = t["a"] + 1
        w = slice(max(a, 0), min(a + 150, n))
        dsh = (lab[w] == "D").mean()
        pre = (lab[max(a - 250, 0):a] == "D").mean()
        r = rows.setdefault(t["type"], [])
        r.append((dsh, pre, vis[w].mean(), a))
    print("cfg=%s SET PIECES clips: share of D in the first 6 s of the clip (and in the 10 s before it)" % c)
    for k, r in rows.items():
        r = np.array(r)
        print("%-22s n %2d | D share in first 6 s: none (<0.1) %2d, some %2d, most (>0.5) %2d | of the 'none': D in the 10 s before > 0.3: %d, ball visible share %.2f, frames %s" % (
            k, len(r), (r[:, 0] < 0.1).sum(), ((r[:, 0] >= 0.1) & (r[:, 0] <= 0.5)).sum(), (r[:, 0] > 0.5).sum(),
            ((r[:, 0] < 0.1) & (r[:, 1] > 0.3)).sum(), r[r[:, 0] < 0.1, 2].mean() if (r[:, 0] < 0.1).any() else 0, r[r[:, 0] < 0.1, 3].astype(int).tolist()))


def cross_dead(lab, period, bx, by, margin=0.5, off=5, hid_s=30.0, max_s=30.0, pre_s=1.0, post_s=3.0, stats=None, vk=0.0, lag=8, min_s=0.0):
    """Prototype post-pass (T113): a visible ball that goes beyond a line by `margin` m with none of our stoppages
    within -pre_s..+post_s starts a stoppage `off` frames later; it lasts until the ball is seen inside the pitch
    again, at most hid_s seconds of hidden ball and max_s seconds in all, and never past the period."""
    out = lab.copy()
    n = len(lab)
    with np.errstate(invalid="ignore"):
        ld = np.fmin(52.5 - np.abs(bx), 34.0 - np.abs(by))
        cr = np.flatnonzero((ld[:-1] >= -margin) & (ld[1:] < -margin) & (period[1:] > 0)) + 1
    sp = np.full(n, np.nan)                              # ball speed over `lag` frames, centred
    sp[lag // 2:n - lag + lag // 2] = np.hypot(bx[lag:] - bx[:-lag], by[lag:] - by[:-lag]) * FPS / lag
    for cf in cr:
        if (out[max(0, cf - int(pre_s * FPS)):min(n, cf + int(post_s * FPS))] == "D").any():
            continue
        k, hid = cf, 0
        while k < n and period[k] == period[cf] and k - cf < max_s * FPS and not (ld[k] >= 0 and (vk <= 0 or sp[k] >= vk)):
            hid = hid + 1 if np.isnan(ld[k]) else 0
            if hid > hid_s * FPS:
                break
            k += 1
        a = min(cf + off, k)
        if k - a < max(1, min_s * FPS):
            continue
        out[a:k] = "D"
        if stats is not None:
            stats.append((k - a) / FPS)
    return out


XD = {"n1": dict(min_s=1.0), "n2": dict(min_s=2.0), "n3": dict(min_s=3.0), "d30": dict(hid_s=30.0), "d3": dict(hid_s=3.0), "d0": dict(hid_s=0.0), "d30m0": dict(hid_s=30.0, margin=0.0), "d30m1": dict(hid_s=30.0, margin=1.0),
      "d8": dict(hid_s=8.0), "d30c10": dict(hid_s=30.0, max_s=10.0),
      "k2": dict(vk=2.0), "k3": dict(vk=3.0), "k5": dict(vk=5.0), "k8": dict(vk=8.0), "k5h3": dict(vk=5.0, hid_s=3.0), "k5c15": dict(vk=5.0, max_s=15.0)}


def cmd_xdrun(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    base = opt.pop("base", "Fx5s")
    cfgs = [base] + str(opt.pop("cfg", ",".join(XD))).split(",")
    labs, tv, st = {}, {}, {c: [] for c in cfgs}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        labs[base, name] = a4_short.decode_one(feats, period, R.CONFIGS[base])
        for c in cfgs[1:]:
            labs[c, name] = cross_dead(labs[base, name], period, feats["ball_x"], feats["ball_y"], stats=st[c], **XD[c])
        del feats
    print({c: (len(x), round(float(np.median(x)), 1)) for c, x in st.items() if x})
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  stoppage at an uncovered line crossing, on top of %s" % (variant, seed, base))
    with open(os.path.join(ROOT, "eval", "results", "a8b_%s_%s_s%d.json" % (opt.pop("tag", "r3xd0"), variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "base": base, "configs": {c: XD.get(c, c) for c in cfgs}, "results": res, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def tag_dead(lab, period, bx, by, starts, lead_s=3.0, free_s=6.0, back_s=20.0, fwd_s=8.0, v_slow=2.0, lag=8, stats=None):
    """Prototype post-pass (T114): a SET PIECES clip that starts at frame a says the ball is dead for its first
    lead_s seconds (the restart sits at clip start + 3.1..8.5 s on the provided input, median 6 s). Only where we
    hold no D in the clip's first free_s seconds: D on [a, a + lead_s), extended backwards while the ball is hidden,
    slower than v_slow or beyond a line (at most back_s), and forwards while it is visible and slower than v_slow
    (at most to a + fwd_s)."""
    out = lab.copy()
    n = len(lab)
    sp = np.full(n, np.nan)
    sp[lag // 2:n - lag + lag // 2] = np.hypot(bx[lag:] - bx[:-lag], by[lag:] - by[:-lag]) * FPS / lag
    with np.errstate(invalid="ignore"):
        ld = np.minimum(52.5 - np.abs(bx), 34.0 - np.abs(by))
        still = np.isnan(bx) | (sp < v_slow) | (ld < 0)
        slow_vis = ~np.isnan(bx) & (sp < v_slow)
    for a in starts:
        a = int(a)
        if a < 0 or a + int(free_s * FPS) >= n or period[a] <= 0 or (lab[a:a + int(free_s * FPS)] == "D").any():
            continue
        s = a
        while s > 0 and a - s < back_s * FPS and period[s - 1] == period[a] and still[s - 1] and lab[s - 1] != "D":
            s -= 1
        k = a + int(lead_s * FPS)
        while k < n and k - a < fwd_s * FPS and period[k] == period[a] and slow_vis[k]:
            k += 1
        out[s:k] = "D"
        if stats is not None:
            stats.append(((s - a) / FPS, (k - a) / FPS))
    return out


def sim_setpieces(truth, period, seed, rate=16.3, jitter=0.0):
    """Simulated SET PIECES clip starts: true restarts after stoppages >= 1 s, `rate` per half hour (57 clips in 3.5
    half hours on the provided match), clip start = restart - offset, offset drawn from the provided INPUT quantiles
    of (our stoppage exit - clip start): p10 3.2, p25 4.9, p50 6.1, p75 7.2, p90 9.0 s (a8b_run.py sp)."""
    rng = np.random.default_rng(3000 + seed)
    rs = [b for a, b in blocks(truth == "D", period) if b - a >= FPS and b < len(truth) and period[b] == period[a] and truth[b] in ("H", "A")]
    k = min(len(rs), int(round(rate * (period > 0).sum() / FPS / 1800.0)))
    off = np.interp(rng.uniform(0, 1, k), [0, .1, .25, .5, .75, .9, 1], [0.7, 3.2, 4.9, 6.1, 7.2, 9.0, 17.0]) + (rng.normal(0, jitter, k) if jitter else 0)
    return [int(rs[i] - o * FPS) for i, o in zip(sorted(rng.choice(len(rs), k, replace=False)), off)]


TD = {"t3": {}, "t2": dict(lead_s=2.0), "t4": dict(lead_s=4.0), "t3n": dict(back_s=0.0, fwd_s=3.0), "t3b": dict(fwd_s=3.0), "t3all": dict(rate=99.0), "t3j": dict(jitter=2.0)}


def cmd_tdrun(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    base = opt.pop("base", "Xd")
    cfgs = [base] + str(opt.pop("cfg", ",".join(TD))).split(",")
    labs, tv, st = {}, {}, {c: [] for c in cfgs}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        labs[base, name] = a4_short.decode_one(feats, period, R.CONFIGS[base])
        for c in cfgs[1:]:
            kw = dict(TD[c])
            starts = sim_setpieces(tv[name]["ctl2"], period, seed, **{k: kw.pop(k) for k in ("rate", "jitter") if k in kw})
            st[c].append(len(starts))
            labs[c, name] = tag_dead(labs[base, name], period, feats["ball_x"], feats["ball_y"], starts, stats=st[c], **kw)
        del feats
    print({c: "clips %d, acted on %d" % (sum(x for x in v2 if isinstance(x, int)), sum(1 for x in v2 if not isinstance(x, int))) for c, v2 in st.items() if v2})
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  set-piece clip start as stoppage evidence (simulated clips), on top of %s" % (variant, seed, base))
    with open(os.path.join(ROOT, "eval", "results", "a8b_%s_%s_s%d.json" % (opt.pop("tag", "r3td"), variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "base": base, "configs": {c: TD.get(c, c) for c in cfgs}, "results": res, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_tdprov(opt):
    """Provided match INPUT only (no truth): what tag_dead does with the real SET PIECES clips."""
    import a8_tags
    from possess import ctl_truth, pipeline

    base = opt.pop("base", "Xd")
    tags, home, f, per, _ = a8_tags.challenge_default()
    lab0 = pipeline.label(f, period=per, dec_kw=R.CONFIGS[base])
    starts = [t["a"] + 1 for t in tags if t["code"] == "SET PIECES" and not t["type"].startswith("Kick Off (Start")]
    st = []
    lab = tag_dead(lab0, per, f["ball_x"], f["ball_y"], starts, stats=st)
    play = per > 0
    print("provided input, base %s: clips %d, acted on %d, D from / to (s rel. clip start): %s; frames changed %.4f, possessions %d -> %d" % (
        base, len(starts), len(st), [(round(a, 1), round(b, 1)) for a, b in st], ((lab != lab0) & play).sum() / play.sum(),
        ctl_truth.n_possessions(lab0, per), ctl_truth.n_possessions(lab, per)))


if __name__ == "__main__":
    cmd, opt = sys.argv[1], a4_short.kv(sys.argv[2:])
    {"hid": cmd_hid, "hidprov": cmd_hidprov, "side": cmd_side, "siderun": cmd_siderun, "doracle": cmd_doracle, "edges": cmd_edges, "entry": cmd_entry, "crossrun": cmd_crossrun, "crossmiss": cmd_crossmiss, "xtransfer": cmd_xtransfer, "rexit": cmd_rexit, "spd": cmd_spd, "xdrun": cmd_xdrun, "tdrun": cmd_tdrun, "tdprov": cmd_tdprov}[cmd](opt)
