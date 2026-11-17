"""Envelope L3-A4 (short possessions): why true possessions are missed, and decoder variants on saved inputs.

    uv run python scripts/ramlock.py -- uv run python scripts/a4_short.py save [bench=real2] [seed=0]
    uv run python scripts/ramlock.py -- uv run python scripts/a4_short.py diag [bench=real2] [seed=0] [key=val ...]
    uv run python scripts/ramlock.py -- uv run python scripts/a4_short.py run [bench=real2] [seed=0] key=val ...

save: decoder inputs of the tune matches (bench features, leave-one-match-out shape emissions, exactly
      what pipeline.label hands to decode.decode_halves) -> data/external/_bench/a4_<match>_<bench>_s<seed>.npz
      (seed > 0: real2 built in memory, as diagnose.py prior check).
diag: missed true possessions (IoU 0.5) by length, ball visibility, touch evidence of their team and
      what the default predicted instead; predicted short runs and their match rate.
run:  default vs default + overrides per match (spellF1, counts) and the paired 2-min block bootstrap
      (diagnose._pr_boot). Tune matches only; nothing reads held-out, public or challenge data.
"""
import os
import sys

from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench  # noqa: E402
import diagnose as dg  # noqa: E402
import tune_shape  # noqa: E402
from possess import decode, pipeline, shape  # noqa: E402

pm = bench.pm
FPS = 25.0
KEYS = ("d_home", "d_away", "visible", "dead", "ball_x", "ball_y")


def _path(name, variant, seed):
    return os.path.join(bench.CACHE, "a4_%s_%s_s%d.npz" % (name, variant or "orig", seed))


def kv(args):
    out = {}
    for a in args:
        k, v = a.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def cmd_save(opt):
    from possess.deadball2 import detect, features as dead_features

    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    variant = "" if variant == "orig" else variant
    clean = {n: tune_shape.shape_features(n, bench.get_case(n)) for n in tune_shape.TUNE}
    truths = {n: bench.get_case(n)["truth"] for n in tune_shape.TUNE}
    for name in tune_shape.TUNE:
        lomo = shape.fit_models([clean[m] for m in clean if m != name], [truths[m] for m in clean if m != name], l2=tune_shape.L2)
        case, f = dg._pr_case(name, variant, seed)
        df = dead_features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["cen_home"], f["cen_away"])
        dead = detect(df, f["ml_dead"], **pipeline.DEAD2)
        feats = {"d_home": np.nan_to_num(f["d_home"], nan=99.0), "d_away": np.nan_to_num(f["d_away"], nan=99.0),
                 "visible": ~np.isnan(f["ball_x"]), "dead": dead, "ball_x": f["ball_x"], "ball_y": f["ball_y"]}
        feats.update(shape.decoder_inputs(f["pos"], case["period"], lomo))
        feats = {k: v for k, v in feats.items() if v is not None}
        if seed:
            feats["pos"] = f["pos"]
        np.savez_compressed(_path(name, variant, seed), truth=case["truth"], period=case["period"],
                            **{"f_" + k: np.asarray(v) for k, v in feats.items()})
        print("saved", _path(name, variant, seed), flush=True)


def load(name, variant, seed):
    z = np.load(_path(name, variant, seed))
    feats = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
    if "sdead_pi" in feats:
        feats["sdead_pi"] = float(feats["sdead_pi"])
    if "pos" not in feats:
        feats["pos"] = np.load(os.path.join(bench.CACHE, "shapepos_%s_s%d_%s.npz" % (name, seed, variant)))["pos"] if variant == "real2"             else tune_shape.positions(name, None, variant)
    return z["truth"], z["period"], feats


def decode_one(feats, period, over=None):
    return decode.decode_halves(period, feats=dict(feats), **dict(pipeline.DEC, **(over or {})))


def runs_of(codes, runs):
    return pm.possession_spells(codes, runs)


def touch_team(feats):
    """+1 / -1 on uncontested touch frames, as the default decoder's touch emission sees them
    (smoothed regime when the track is low-passed), 0 elsewhere."""
    bx, by, dh, da = feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"]
    lag, peak, r = 2, 0, 1.0
    if decode.ball_smoothed(bx, by):
        lag, peak, r = decode.SMOOTH_TOUCH["lag"], decode.SMOOTH_TOUCH["peak"], decode.SMOOTH_TOUCH["r_touch"]
    from possess.ballclean import clean_ball

    bx, by, _ = clean_ball(bx, by, np.minimum(dh, da), r, FPS)
    return decode.touch_hold(bx, by, dh, da, r, 0.0, lag=lag, peak=peak), r


def cmd_diag(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    variant = "" if variant == "orig" else variant
    for name in tune_shape.TUNE:
        truth, period, feats = load(name, variant, seed)
        lab = decode_one(feats, period, opt)
        g = pm.label_codes(truth)
        runs = pm.valid_runs(g)
        p = pm.fill_abstentions(pm.label_codes(lab), runs)
        gs, ps = runs_of(g, runs), runs_of(p, runs)
        g_ok, p_ok = dg.matched_flags(gs, ps)
        vis = feats["visible"]
        tt, r = touch_team(feats)
        dh, da = feats["d_home"], feats["d_away"]
        r_ = pm.evaluate(g, p, bootstrap_n=0)
        print("\n== %s %.4f pred %d true %d | touch r %.1f" % (name, r_["spell_f1_mean"], len(ps), len(gs), r))
        cat = {}
        for (a, b, team), ok in zip(gs, g_ok):
            L = (b - a) / FPS
            lb = "<2" if L < 2 else "2-4" if L < 4 else "4-10" if L < 10 else ">=10"
            v = vis[a:b].mean()
            vb = "hid" if v < 0.1 else "vis" if v > 0.9 else "mix"
            sgn = 1.0 if team == pm.H else -1.0
            dteam = dh if team == pm.H else da
            dopp = da if team == pm.H else dh
            touched = (tt[max(0, a - 12):b] == sgn).any()
            near1 = (vis[a:b] & (dteam[a:b] <= 1.0)).any()
            near2 = (vis[a:b] & (dteam[a:b] <= 2.0)).any()
            seg = p[a:b]
            maj = "same" if (seg == team).mean() >= 0.5 else "D" if (seg == pm.D).mean() >= 0.5 else "other"
            key = (lb, vb)
            c = cat.setdefault(key, np.zeros(9, int))
            c += np.array([1, ok, touched, touched and ok, near1, near2, maj == "same", maj == "D", maj == "other"])
        print("%-5s %-4s %5s %5s | %7s %6s | %5s %5s | missed: pred mostly same / D / other" % ("len", "ball", "n", "found", "touched", "t&fnd", "<=1m", "<=2m"))
        for key in sorted(cat, key=lambda k: (["<2", "2-4", "4-10", ">=10"].index(k[0]), k[1])):
            c = cat[key]
            print("%-5s %-4s %5d %5d | %7d %6d | %5d %5d | %d / %d / %d" % (*key, *c))
        # predicted runs by length: matched share
        PL = np.array([(b - a) / FPS for a, b, _ in ps])
        for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 10), (10, 1e9)]:
            m = (PL >= lo) & (PL < hi)
            print("pred %g-%g s: %d, matched %.2f" % (lo, hi, m.sum(), p_ok[m].mean() if m.any() else np.nan))


def touch_peaks(feats):
    """Frames where the default's touch test fires on the ball alone (impulse, peak-thinned in the
    smoothed regime), before the radius / team test; and the radius."""
    bx, by, dh, da = feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"]
    lag, peak, r = 2, 0, 1.0
    if decode.ball_smoothed(bx, by):
        lag, peak, r = decode.SMOOTH_TOUCH["lag"], decode.SMOOTH_TOUCH["peak"], decode.SMOOTH_TOUCH["r_touch"]
    from possess.ballclean import clean_ball

    bx, by, _ = clean_ball(bx, by, np.minimum(dh, da), r, FPS)
    imp = decode.ball_impulse(bx, by, lag=lag)
    t = imp["impulse"].copy()
    dv = np.nan_to_num(np.hypot(imp["vox"] - imp["vix"], imp["voy"] - imp["viy"]), nan=-1.0)
    if peak:
        top = dv.copy()
        for s in range(1, lag + 1):
            top[s:] = np.maximum(top[s:], dv[:-s])
            top[:-s] = np.maximum(top[:-s], dv[s:])
        t &= dv >= top
    return t, dv, r


def cmd_touches(opt):
    """Ball-impulse peaks near players: contested share and who is right, all live frames and inside
    missed short visible true possessions."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    variant = "" if variant == "orig" else variant
    for name in tune_shape.TUNE:
        truth, period, feats = load(name, variant, seed)
        lab = decode_one(feats, period, opt)
        g = pm.label_codes(truth)
        runs = pm.valid_runs(g)
        p = pm.fill_abstentions(pm.label_codes(lab), runs)
        gs = runs_of(g, runs)
        g_ok, _ = dg.matched_flags(gs, runs_of(p, runs))
        t, dv, r = touch_peaks(feats)
        dh, da = feats["d_home"], feats["d_away"]
        nxt = np.concatenate([g[4:], [pm.D] * 4])     # truth a few frames after the touch
        live = t & np.isin(nxt, [pm.H, pm.A]) & feats["visible"]
        hin, ain = dh <= r, da <= r
        one, both, none = live & (hin != ain), live & hin & ain, live & ~hin & ~ain
        nearer_h = dh < da
        right = nearer_h == (nxt == pm.H)
        print("\n== %s r %.1f | live peaks %d: uncontested %d right %.3f | contested %d nearer right %.3f | nobody within r %d nearer right %.3f"
              % (name, r, live.sum(), one.sum(), right[one].mean(), both.sum(), right[both].mean(), none.sum(), right[none].mean()))
        for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]:
            m = both & (np.abs(dh - da) >= lo) & (np.abs(dh - da) < hi)
            print("   contested |dh-da| %.1f-%.1f: n %d nearer right %.3f" % (lo, hi, m.sum(), right[m].mean() if m.any() else np.nan))
        # inside missed short visible true spells
        c = Counter()
        for (a, b, team), ok in zip(gs, g_ok):
            if ok or b - a >= 4 * FPS or feats["visible"][a:b].mean() <= 0.9:
                continue
            sl = slice(max(0, a - 12), b)
            dT = (dh if team == pm.H else da)[sl]
            dO = (da if team == pm.H else dh)[sl]
            pk = t[sl]
            if not pk.any():
                c["no peak at all"] += 1
            elif not (pk & (dT <= r)).any():
                c["peaks, team never within r at a peak"] += 1
            elif (pk & (dT <= r) & (dO > r)).any():
                c["uncontested own peak (decoder overruled)"] += 1
            elif (pk & (dT <= r) & (dT < dO)).any():
                c["contested own peak, team nearer"] += 1
            else:
                c["contested own peak, team farther"] += 1
        print("   missed short visible:", dict(c))


def cmd_run(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    variant = "" if variant == "orig" else variant
    names = opt.pop("names", None)
    names = names.split(",") if isinstance(names, str) else tune_shape.TUNE
    diffs, so, sn = [], [], []
    for name in names:
        truth, period, feats = load(name, variant, seed)
        off = decode_one(feats, period)
        on = decode_one(feats, period, opt)
        g = pm.label_codes(truth)
        ra, rb = (pm.evaluate(g, pm.label_codes(x), bootstrap_n=0) for x in (off, on))
        print("%s off %.4f %d/%d on %.4f %d/%d  frames changed %.4f" % (
            name, ra["spell_f1_mean"], ra["n_pred_spells"], ra["n_gt_spells"], rb["spell_f1_mean"], rb["n_pred_spells"],
            rb["n_gt_spells"], (off != on)[period > 0].mean()), flush=True)
        bo, bn = dg._pr_boot(truth, off, on)
        diffs.append(bn - bo); so.append(ra["spell_f1_mean"]); sn.append(rb["spell_f1_mean"])
    d = np.mean(diffs, axis=0)
    print("%s s%d %s | mean off %.4f on %.4f (%+.4f) | bootstrap %+.4f CI90 [%+.4f, %+.4f]" % (
        variant or "orig", seed, " ".join("%s=%s" % kv_ for kv_ in opt.items()), np.mean(so), np.mean(sn),
        np.mean(sn) - np.mean(so), d.mean(), np.percentile(d, 5), np.percentile(d, 95)), flush=True)


def vertex(bx, by, lag, fps=FPS):
    """Per frame i: contact point and contact frame of a straight-in / straight-out ball through a
    touch between i-lag and i+lag (velocities from the lag frames outside that window; time between
    the two anchor positions conserved, least squares). (vx, vy, k) with nan where not measurable."""
    n = len(bx)
    L = lag
    P = np.stack([bx, by], 1)
    b0 = np.full((n, 2), np.nan); b1 = np.full((n, 2), np.nan)
    vin = np.full((n, 2), np.nan); vout = np.full((n, 2), np.nan)
    b0[L:] = P[:-L]; b1[:-L] = P[L:]
    vin[2 * L:] = (P[L:-L] - P[:-2 * L]) * fps / L
    vout[:-2 * L] = (P[2 * L:] - P[L:-L]) * fps / L
    T = 2.0 * L / fps
    dv = vin - vout
    rhs = b1 - b0 - vout * T
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = np.clip((dv * rhs).sum(1) / (dv * dv).sum(1), 0.0, T)
    v = b0 + vin * t1[:, None]
    k = np.arange(n) - L + np.round(np.nan_to_num(t1) * fps).astype(int)
    return v[:, 0], v[:, 1], np.clip(k, 0, n - 1)


def team_dist_to(pos, px, py, k):
    """[n, 2] distance of each team's nearest player at frame k to point (px, py)."""
    d = np.hypot(pos[k, :, :, 0] - px[:, None, None], pos[k, :, :, 1] - py[:, None, None])
    return np.nanmin(np.where(np.isnan(d), 1e9, d), axis=2)


def cmd_vertex(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    variant = "" if variant == "orig" else variant
    for name in tune_shape.TUNE:
        truth, period, feats = load(name, variant, seed)
        pos = np.load(os.path.join(bench.CACHE, "shapepos_%s_s%d_%s.npz" % (name, seed, variant)))["pos"]
        t, dv, r = touch_peaks(feats)
        from possess.ballclean import clean_ball

        bx, by, _ = clean_ball(feats["ball_x"], feats["ball_y"], np.minimum(feats["d_home"], feats["d_away"]), r, FPS)
        lag = decode.SMOOTH_TOUCH["lag"] if decode.ball_smoothed(feats["ball_x"], feats["ball_y"]) else 2
        g = pm.label_codes(truth)
        nxt = np.concatenate([g[4:], [pm.D] * 4])
        live = t & np.isin(nxt, [pm.H, pm.A]) & feats["visible"]
        idx = np.flatnonzero(live)
        chk = team_dist_to(pos, bx, by, np.arange(len(bx)))[idx]
        print("\n== %s check d_home vs pos[:,0]: median abs diff %.3f" % (name, np.nanmedian(np.abs(chk[:, 0] - feats["d_home"][idx]))))
        vx, vy, k = vertex(bx, by, lag)
        vx, vy, k = vx[idx], vy[idx], k[idx]
        ok = np.isfinite(vx)
        dT = np.full((len(idx), 2), 1e9)
        for s in (-1, 0, 1):
            dT = np.fmin(dT, team_dist_to(pos, np.nan_to_num(vx), np.nan_to_num(vy), np.clip(k + s, 0, len(bx) - 1)))
        dT[~ok] = 1e9
        tru_h = nxt[idx] == pm.H
        dh, da = feats["d_home"][idx], feats["d_away"][idx]
        cur_one = (dh <= r) != (da <= r)
        cur_right = ((dh <= r) == tru_h)[cur_one].mean()
        print("current r %.1f: uncontested %d of %d, right %.3f" % (r, cur_one.sum(), len(idx), cur_right))
        for R in (1.0, 1.5, 2.0, 3.0):
            one = (dT[:, 0] <= R) != (dT[:, 1] <= R)
            print("vertex R %.1f: uncontested %d, right %.3f | new (not current-uncontested) %d right %.3f | contested %d" % (
                R, one.sum(), ((dT[:, 0] <= R) == tru_h)[one].mean(), (one & ~cur_one).sum(),
                ((dT[:, 0] <= R) == tru_h)[one & ~cur_one].mean(), ((dT[:, 0] <= R) & (dT[:, 1] <= R)).sum()))
        near = np.argmin(dT, 1) == 0
        mg = np.abs(dT[:, 0] - dT[:, 1])
        for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 1e8)]:
            m = (mg >= lo) & (mg < hi) & (dT.min(1) <= 3.0)
            print("   nearest-to-vertex within 3 m, margin %.1f-%.1f: n %d right %.3f" % (lo, hi, m.sum(), (near == tru_h)[m].mean()))


def emission_parts(feats, period, over=None):
    """Per-frame team log-score parts of the default decoder (H positive), per half: dist, rule, touch, shape."""
    from possess.ballclean import clean_ball
    from possess.team_control import team_from_distances

    kw = dict(pipeline.DEC, **(over or {}))
    n = len(period)
    parts = {k: np.zeros(n) for k in ("dist", "rule", "touch", "shape")}
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(period, int), [0]])) != 0)
    for a, b in zip(edges[:-1], edges[1:]):
        if period[a] <= 0:
            continue
        sl = slice(a, b)
        dh, da, vis, dead = feats["d_home"][sl], feats["d_away"][sl], feats["visible"][sl], feats["dead"][sl]
        bx, by = feats["ball_x"][sl], feats["ball_y"][sl]
        parts["dist"][sl] = decode.team_emission(dh, da, vis, soft=1, d0=kw["d0"], cap=kw["cap"])
        rule = team_from_distances(dh, da, vis, dead=dead, **pipeline.TEAM)
        parts["rule"][sl] = kw["w_rule"] * vis * ((rule == "H").astype(float) - (rule == "A"))
        lag, peak, r = kw.get("lag", 2), 0, kw.get("r_touch", 1.0)
        if decode.ball_smoothed(bx, by):
            lag, peak, r = decode.SMOOTH_TOUCH["lag"], decode.SMOOTH_TOUCH["peak"], decode.SMOOTH_TOUCH["r_touch"]
        cx, cy, _ = clean_ball(bx, by, np.minimum(dh, da), r, FPS)
        parts["touch"][sl] = kw["w_touch"] * vis * decode.touch_hold(cx, cy, dh, da, r, 4.0, duel=kw["duel"], lag=lag, peak=peak,
                                                                     reappear=kw["reappear"], tw=kw.get("tw", 0))
        parts["shape"][sl] = kw["w_shape"] * np.where(vis, kw["shape_vis"], 1.0) * feats["shape_team"][sl]
    return parts


def cmd_overruled(opt):
    """Missed short (< 4 s) visible true possessions: summed team evidence for the true team, by part."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    variant = "" if variant == "orig" else variant
    big = {}
    for name in tune_shape.TUNE:
        truth, period, feats = load(name, variant, seed)
        lab = decode_one(feats, period, opt)
        parts = emission_parts(feats, period, opt)
        g = pm.label_codes(truth)
        runs = pm.valid_runs(g)
        p = pm.fill_abstentions(pm.label_codes(lab), runs)
        gs, ps = runs_of(g, runs), runs_of(p, runs)
        g_ok, _ = dg.matched_flags(gs, ps)
        rows, offs = [], []
        for (a, b, team), ok in zip(gs, g_ok):
            if b - a >= 4 * FPS or feats["visible"][a:b].mean() <= 0.9:
                continue
            sg = 1.0 if team == pm.H else -1.0
            s = {k: sg * v[a:b].sum() for k, v in parts.items()}
            hold = (sg * parts["touch"][a:b] > 0).mean()
            wrong = (sg * parts["touch"][a:b] < 0).mean()
            rows.append((ok, s["dist"], s["rule"], s["touch"], s["shape"], hold, wrong, (b - a) / FPS, (p[a:b] == team).mean()))
            if not ok and sum(s.values()) > 8:
                c_ = Counter()
                c_["same" if (p[a:b] == team).mean() > 0.5 else "D" if (p[a:b] == pm.D).mean() > 0.5 else "other"] += 1
                prev = g[a - 1] if a > 0 else -1
                nxt = g[b] if b < len(g) else -1
                ctx = ("prevD" if prev == pm.D else "prevOpp") + ("/nextD" if nxt == pm.D else "/nextOpp" if nxt != team else "/nextSame")
                big.setdefault(name, Counter())[(list(c_)[0], ctx)] += 1
                if "same" in c_:
                    for pa, pb, pt in ps:
                        if pt == team and pa < b and pb > a:
                            offs.append(((pa - a) / FPS, (pb - b) / FPS, (b - a) / FPS))
                            break
        R = np.array(rows, float)
        for ok in (1, 0):
            m = R[:, 0] == ok
            print("%s %s n %d | median sums dist %+.1f rule %+.1f touch %+.1f shape %+.1f | own hold %.2f wrong hold %.2f | len %.1f s | pred same %.2f | total>8: %d"
                  % (name, "found " if ok else "missed", m.sum(), *np.median(R[m, 1:9], axis=0), (R[m, 1:5].sum(1) > 8).sum()))
        m = R[:, 0] == 0
        tot = R[m, 1:5].sum(1)
        print("   missed: total evidence quartiles %s; own hold == 0: %d, wrong hold > 0.5: %d" % (
            np.round(np.percentile(tot, [10, 25, 50, 75, 90]), 1), (R[m, 5] == 0).sum(), (R[m, 6] > 0.5).sum()))
        print("   missed with evidence > 8 (pred majority, true context):", dict(big.get(name, {})))
        if offs:
            O = np.array(offs)
            print("   'same' cases: pred start - true start q25/50/75 %s s; pred end - true end %s s; true len %s s" % (
                np.round(np.percentile(O[:, 0], [25, 50, 75]), 2), np.round(np.percentile(O[:, 1], [25, 50, 75]), 2),
                np.round(np.percentile(O[:, 2], [25, 50, 75]), 2)))


def cmd_transfer(opt):
    """Provided match INPUT only (cached features + positions; the ground truth is never opened):
    possessions per half hour, D share and frames changed, default vs default + overrides
    -> eval/results/a4_transfer.json. Nothing is chosen on it."""
    import json

    f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"),
                                    os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz"))
    per = pipeline.challenge_period(len(f["ball_x"]))
    play = per > 0
    labs = {"off": pipeline.label(f, period=per), "on": pipeline.label(f, dec_kw=opt, period=per)}
    hh = play.sum() / FPS / 1800.0
    res = {"overrides": opt, "frames changed": float((labs["off"] != labs["on"])[play].mean())}
    for t, lab in labs.items():
        n_pos = sum(int(((lab[1:] == c) & (lab[:-1] != c) & play[1:]).sum()) for c in "HA")
        res[t + " possessions / half hour"] = n_pos / hh
        res[t + " D share"] = float((lab == "D")[play].mean())
        res[t + " H share of H+A"] = float((lab == "H")[play].sum() / max(1, np.isin(lab, ["H", "A"])[play].sum()))
    for k, v in res.items():
        print(k, v)
    with open(os.path.join(ROOT, "eval", "results", "a4_transfer.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    cmd, rest = sys.argv[1], kv(sys.argv[2:])
    {"save": cmd_save, "diag": cmd_diag, "run": cmd_run, "touches": cmd_touches, "overruled": cmd_overruled, "vertex": cmd_vertex, "transfer": cmd_transfer}[cmd](rest)
