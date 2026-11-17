"""Envelope L8-autonomous-fable, round 5: team identity on visible frames, one-kick spells, and why
frame accuracy and spellF1 part ways. Tune matches only (cached decoder inputs), ctl T=2 truth.

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r5.py tid bench=real2 seed=0
    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r5.py price bench=real2 seed=0

tid:    oracle split of the wrong-team frames of the default labels (whole spell wrong by true
        length, late / early switch inside a spell we mostly get right, duel frames; visible / hidden)
price:  what the metric pays: boundary jitter, dropped short spells, inserted false spells applied to
        the TRUTH (accuracy vs spellF1), and on the default labels the gain of inserting one missed
        true spell against the cost of one false spell (break-even precision of a proposal)
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

pm = R.pm
FPS = 25.0
OUT = os.path.join(ROOT, "eval", "results")


def f1(truth, lab):
    return pm.evaluate(pm.label_codes(truth), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]


def load_all(variant, seed, cfg="default"):
    v = "" if variant == "orig" else variant
    out = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        out[name] = dict(truth=R.truths(truth, period)["ctl2"], period=period, feats=feats,
                         pred=a4_short.decode_one(feats, period, R.CONFIGS[cfg]))
    return out


def spell_index(truth, period):
    """Per frame: index of its true run, and per run (start, end, label)."""
    runs = R._runs(np.asarray(truth), np.asarray(period))
    idx = np.zeros(len(truth), int)
    for i, (s, e, k, p) in enumerate(runs):
        idx[s:e] = i
    return idx, runs


def cmd_tid(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    data = load_all(variant, seed, opt.pop("cfg", "default"))
    names = list(data)
    classes = ["all", "vis", "hid", "vis_whole_lt3", "vis_whole_3_8", "vis_whole_ge8", "vis_late", "vis_early", "vis_duel",
               "vis_free", "vis_opp1m", "hid_whole", "hid_edge"]
    share = {c: [] for c in classes}
    gain = {c: [] for c in classes}
    base = []
    lag_late, lag_early = [], []
    for n in names:
        d = data[n]
        t, p, per, f = d["truth"], d["pred"], d["period"], d["feats"]
        vis = np.asarray(f["visible"], bool)
        team_t, team_p = np.isin(t, ["H", "A"]), np.isin(p, ["H", "A"])
        wrong = team_t & team_p & (t != p) & (per > 0)
        idx, runs = spell_index(t, per)
        same = np.zeros(len(runs))
        ln = np.zeros(len(runs))
        for i, (s, e, k, pp) in enumerate(runs):
            same[i] = (p[s:e] == k).mean()
            ln[i] = (e - s) / FPS
        whole = same[idx] < 0.5
        pos = np.zeros(len(t))                                # relative position of the frame in its true spell
        for i, (s, e, k, pp) in enumerate(runs):
            pos[s:e] = (np.arange(e - s) + 0.5) / (e - s)
        L = ln[idx]
        dmin_own = np.where(t == "H", f["d_home"], f["d_away"])
        dmin_opp = np.where(t == "H", f["d_away"], f["d_home"])
        duel = (dmin_own < 1.5) & (dmin_opp < 1.5)
        free = (dmin_own >= 1.5) & (dmin_opp >= 1.5)
        opp1 = (dmin_opp < 1.0) & (dmin_own >= 1.5)
        masks = {"all": wrong, "vis": wrong & vis, "hid": wrong & ~vis,
                 "vis_whole_lt3": wrong & vis & whole & (L < 3), "vis_whole_3_8": wrong & vis & whole & (L >= 3) & (L < 8),
                 "vis_whole_ge8": wrong & vis & whole & (L >= 8),
                 "vis_late": wrong & vis & ~whole & (pos < 0.5), "vis_early": wrong & vis & ~whole & (pos >= 0.5),
                 "vis_duel": wrong & vis & duel, "vis_free": wrong & vis & free, "vis_opp1m": wrong & vis & opp1,
                 "hid_whole": wrong & ~vis & whole, "hid_edge": wrong & ~vis & ~whole}
        b = f1(t, p)
        base.append(b)
        live = (team_t & team_p & (per > 0)).sum()
        for c, m in masks.items():
            lab = p.copy()
            lab[m] = t[m]
            share[c].append(m.sum() / live)
            gain[c].append(f1(t, lab) - b)
        # switch lag on spells we mostly get right: length of the wrong run at the start / end
        for i, (s, e, k, pp) in enumerate(runs):
            if k in "HA" and pp > 0 and same[i] >= 0.5:
                w = wrong[s:e]
                a = int(np.argmin(w)) if not w.all() else e - s
                z = int(np.argmin(w[::-1])) if not w.all() else e - s
                lag_late.append(a / FPS)
                lag_early.append(z / FPS)
    print("%s s%d  default ctl2 %.4f (%s)" % (variant, seed, np.mean(base), " ".join("%.3f" % x for x in base)))
    print("%-15s %8s %8s   per match" % ("class", "frames%", "d ctl2"))
    res = {"base": base}
    for c in classes:
        print("%-15s %8.2f %+8.4f   %s" % (c, 100 * np.mean(share[c]), np.mean(gain[c]), " ".join("%+.3f" % x for x in gain[c])), flush=True)
        res[c] = {"share": float(np.mean(share[c])), "gain": float(np.mean(gain[c])), "per_match": gain[c]}
    la, le = np.array(lag_late), np.array(lag_early)
    for nm, x in (("late", la), ("early", le)):
        print("%s switch on mostly-right spells: > 0 in %.2f, > 0.5 s %.2f, > 1 s %.2f, > 2 s %.2f; mean %.2f s" % (
            nm, (x > 0).mean(), (x > 0.5).mean(), (x > 1).mean(), (x > 2).mean(), x.mean()))
    with open(os.path.join(OUT, "a8b_r5_tid_%s_s%d.json" % (variant, seed)), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


def _team_spells(lab, period):
    return [(s, e, k) for s, e, k, p in R._runs(np.asarray(lab), np.asarray(period)) if p > 0 and k in "HA"]


def cmd_price(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    data = load_all(variant, seed, opt.pop("cfg", "default"))
    rng = np.random.default_rng(0)
    rows = {}

    def add(key, acc, sc):
        rows.setdefault(key, []).append((acc, sc))

    for n, d in data.items():
        t, p, per = d["truth"], d["pred"], d["period"]
        live = per > 0
        acc = lambda lab: float((lab[live] == t[live]).mean())   # noqa: E731
        add("pred", acc(p), f1(t, p))
        runs = R._runs(t, per)
        # 1. boundary jitter on the truth (team <-> team and team <-> D boundaries alike)
        for sg in (0.25, 0.5, 1.0, 2.0):
            lab = t.copy()
            for (s, e, k, pp), (s2, e2, k2, pp2) in zip(runs[:-1], runs[1:]):
                if pp != pp2 or pp <= 0:
                    continue
                sh = int(round(rng.normal(0, sg) * FPS))
                sh = max(-(e - s) // 2 + 1, min((e2 - s2) // 2 - 1, sh))
                if sh > 0:
                    lab[e:e + sh] = k
                elif sh < 0:
                    lab[e + sh:e] = k2
            add("jitter %.2f s" % sg, acc(lab), f1(t, lab))
        # 2. drop true team spells shorter than L (take the previous label)
        for Lm in (3.0, 5.0, 8.0):
            lab = t.copy()
            for i, (s, e, k, pp) in enumerate(runs):
                if k in "HA" and pp > 0 and e - s < Lm * FPS and i > 0 and runs[i - 1][3] == pp:
                    lab[s:e] = lab[s - 1]
            add("drop true < %g s" % Lm, acc(lab), f1(t, lab))
        # 3. insert false spells of 2.5 s of the other team inside true team spells >= 10 s, one per k such spells
        long_ = [(s, e, k) for s, e, k, pp in runs if k in "HA" and pp > 0 and e - s >= 10 * FPS]
        for frac in (0.25, 0.5, 1.0):
            lab = t.copy()
            for s, e, k in long_:
                if rng.random() < frac:
                    a = int(rng.integers(s + 50, e - 50 - 62))
                    lab[a:a + 62] = "A" if k == "H" else "H"
            add("false 2.5 s in %.2f of long" % frac, acc(lab), f1(t, lab))
        # 4. on the default labels: insert each missed true spell (pred mostly the other team) / a false spell
        ts = _team_spells(t, per)
        missed = [(s, e, k) for s, e, k in ts if (p[s:e] == k).mean() < 0.5 and (p[s:e] != "D").mean() > 0.5]
        b = f1(t, p)
        lab = p.copy()
        for s, e, k in missed:
            lab[s:e] = k
        add("pred + all missed true spells (n=%d)" % len(missed), acc(lab), f1(t, lab))
        for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 1e9)):
            lab = p.copy()
            m = [(s, e, k) for s, e, k in missed if lo * FPS <= e - s < hi * FPS]
            for s, e, k in m:
                lab[s:e] = k
            add("pred + missed %g-%g s: per spell" % (lo, hi), len(m), (f1(t, lab) - b) / max(1, len(m)))
        ps = [(s, e, k) for s, e, k in _team_spells(p, per) if e - s >= 8 * FPS and (t[s:e] == k).mean() > 0.9]
        for dur in (2.0, 3.0, 4.0):
            lab = p.copy()
            w = int(dur * FPS)
            pick = [ps[i] for i in rng.permutation(len(ps))[:len(missed)]]
            for s, e, k in pick:
                a = int(rng.integers(s + 25, e - 25 - w))
                lab[a:a + w] = "A" if k == "H" else "H"
            add("pred + false %g s spells: per spell" % dur, len(pick), (f1(t, lab) - b) / max(1, len(pick)))
        # 5. jitter of the default labels' own boundaries, and snapping them to the truth where a true boundary is near
        tb = np.flatnonzero(t[1:] != t[:-1]) + 1
        pr = R._runs(p, per)
        for W in (0.5, 1.0, 2.0):
            lab = p.copy()
            for (s, e, k, pp), (s2, e2, k2, pp2) in zip(pr[:-1], pr[1:]):
                if pp != pp2 or pp <= 0 or len(tb) == 0:
                    continue
                j = tb[np.argmin(np.abs(tb - e))]
                if abs(j - e) <= W * FPS and t[j - 1] == k and t[j] == k2:
                    if j > e:
                        lab[e:min(j, e2)] = k
                    else:
                        lab[max(j, s):e] = k2
            add("pred boundaries snapped to truth within %g s" % W, acc(lab), f1(t, lab))
    print("%s s%d (mean of the tune matches)" % (variant, seed))
    res = {}
    for k, v in rows.items():
        a, s = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        res[k] = [float(a), float(s)]
        if "per spell" in k:
            print("%-48s n %6.1f  d ctl2 per spell %+.5f" % (k, a, s))
        else:
            print("%-48s acc %.4f  ctl2 %.4f" % (k, a, s))
    with open(os.path.join(OUT, "a8b_r5_price_%s_s%d.json" % (variant, seed)), "w") as fh:
        json.dump(res, fh, indent=1)


def insert_from(base, alt, period, min_s=2.0, max_s=1e9, purity=0.9):
    """T122: base labels plus the team spells of `alt` (>= min_s) that lie inside ONE base spell of the
    other team (sandwiched: base names the other team just before, just after and on >= purity of it).
    Nothing else of `alt` is taken (no boundary moves, no D changes). Returns (labels, [(s, e, k)])."""
    lab = np.array(base)
    added = []
    n = len(base)
    for s, e, k in _team_spells(alt, period):
        if not (min_s * FPS <= e - s < max_s * FPS) or s == 0 or e >= n:
            continue
        o = "A" if k == "H" else "H"
        if base[s - 1] == o and base[e] == o and period[s - 1] == period[s] == period[e] and (base[s:e] == o).mean() >= purity:
            lab[s:e] = k
            added.append((s, e, k))
    return lab, added


ALT = {"cs3": dict(c_switch=3.0), "cs2": dict(c_switch=2.0), "cs1": dict(c_switch=1.0), "wt10": dict(w_touch=1.0),
       "cs2wt10": dict(c_switch=2.0, w_touch=1.0), "wf10": dict(w_fly=1.0), "cs2wf10": dict(c_switch=2.0, w_fly=1.0),
       "wc16": dict(w_ctrl=1.6), "cs2wc16": dict(c_switch=2.0, w_ctrl=1.6), "cs3wc16": dict(c_switch=3.0, w_ctrl=1.6),
       "cs3wf10": dict(c_switch=3.0, w_fly=1.0), "mh6": dict(max_hold_s=6.0), "cs2x": dict(c_switch=2.0, ctl_hi=0, ctl_s=0),
       "cs15": dict(c_switch=1.5), "cs25": dict(c_switch=2.5)}


def cmd_ins(opt):
    """Selective insertion: precision of the sandwiched spells a cheaper-switch decode adds, and the score."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    alts = str(opt.pop("alt", "cs3,cs2,cs1")).split(",")
    mins = [float(x) for x in str(opt.pop("min", "2")).split(",")]
    tag = opt.pop("tag", "ins")
    v = "" if variant == "orig" else variant
    labs, tv, stat = {}, {}, {}
    cfgs = ["default"]
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = R.truths(truth, period)
        t = tv[name]["ctl2"]
        base = a4_short.decode_one(feats, period, {})
        labs["default", name] = base
        for a in alts:
            alt = a4_short.decode_one(feats, period, ALT[a])
            labs[a, name] = alt
            for m in mins:
                key = "%s>%g" % (a, m)
                lab, added = insert_from(base, alt, period, m)
                labs[key, name] = lab
                st = stat.setdefault(key, {"n": 0, "real": 0, "by": {}})
                for s, e, k in added:
                    r = (t[s:e] == k).mean() >= 0.5
                    st["n"] += 1
                    st["real"] += r
                    b = "<3" if e - s < 75 else "3-5" if e - s < 125 else "5-8" if e - s < 200 else ">=8"
                    c = st["by"].setdefault(b, [0, 0])
                    c[0] += 1
                    c[1] += r
        del feats
    for a in alts:
        cfgs += [a] + ["%s>%g" % (a, m) for m in mins]
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  selective insertion (T122)" % (variant, seed))
    for k, st in stat.items():
        print("%-12s added %3d real %.2f | " % (k, st["n"], st["real"] / max(1, st["n"])) +
              "  ".join("%s %d (%.2f)" % (b, c[0], c[1] / c[0]) for b, c in sorted(st["by"].items())))
    with open(os.path.join(OUT, "a8b_r5_%s_%s_s%d.json" % (tag, variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "results": res, "stat": stat, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def ball_vel(bx, by, lag=8):
    n, h = len(bx), lag // 2
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    vx[h:n - lag + h] = (bx[lag:] - bx[:-lag]) * FPS / lag
    vy[h:n - lag + h] = (by[lag:] - by[:-lag]) * FPS / lag
    return vx, vy


def impulses(bx, by, thr, h=6, lag=8):
    """Frames where the ball's velocity changes by >= thr m/s between h frames before and h after
    (local maxima of that change within +-h frames)."""
    vx, vy = ball_vel(bx, by, lag)
    n = len(bx)
    dv = np.full(n, 0.0)
    with np.errstate(invalid="ignore"):
        d = np.hypot(vx[2 * h:] - vx[:-2 * h], vy[2 * h:] - vy[:-2 * h])
    dv[h:n - h] = np.nan_to_num(d)
    out = []
    for i in np.flatnonzero(dv >= thr):
        if dv[i] >= dv[max(0, i - h):i + h + 1].max():
            if not out or i - out[-1] > h:
                out.append(int(i))
    return np.array(out, int), dv


def kick_proposals(pred, period, feats, thr=5.0, margin=1.0, rmax=3.0, lo=2.0, hi=8.0, win=3, end="imp"):
    """T123 proposals: inside a predicted spell of team A, a ball impulse nearest to team B (by `margin`
    m, within `rmax`) starts a B spell that runs to the next impulse nearest to A (end="imp") or the
    next frame A is within 1 m and B is not within 1.5 m (end="ctl"); kept if lo <= length < hi s."""
    bx, by = feats["ball_x"], feats["ball_y"]
    dh, da = np.asarray(feats["d_home"], float), np.asarray(feats["d_away"], float)
    imp, dv = impulses(bx, by, thr)
    n = len(pred)
    team = np.zeros(len(imp), int)                           # +1 home, -1 away, 0 unclear
    for j, i in enumerate(imp):
        a, b = max(0, i - win), min(n, i + win + 1)
        h_, a_ = dh[a:b].min(), da[a:b].min()
        if min(h_, a_) <= rmax and abs(h_ - a_) >= margin:
            team[j] = 1 if h_ < a_ else -1
    ctl = np.where((dh < 1.0) & (da >= 1.5), 1, np.where((da < 1.0) & (dh >= 1.5), -1, 0))
    out = []
    code = {"H": 1, "A": -1}
    for j, i in enumerate(imp):
        k = pred[i]
        if k not in code or team[j] != -code[k] or period[i] <= 0:
            continue
        A = code[k]
        e = None
        if end == "imp":
            for jj in range(j + 1, len(imp)):
                if imp[jj] - i >= hi * FPS:
                    break
                if team[jj] == A:
                    e = int(imp[jj])
                    break
                if team[jj] == -A:
                    continue
        else:
            w = np.flatnonzero(ctl[i + 1:i + int(hi * FPS)] == A)
            e = i + 1 + int(w[0]) if len(w) else None
        if e is None or not (lo * FPS <= e - i < hi * FPS) or period[e] != period[i]:
            continue
        if (pred[i:e] == k).mean() < 0.9 or pred[e] != k or pred[i - 1] != k:
            continue
        if np.isnan(bx[i:e]).mean() > 0.2:
            continue
        if out and i < out[-1][1]:
            continue
        out.append((int(i), e, "A" if k == "H" else "H"))
    return out


def cmd_kick(opt):
    """Precision / recall of the kick proposals against the ctl2 truth, and the score with them inserted."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    data = load_all(variant, seed)
    grid = []
    for thr in [float(x) for x in str(opt.pop("thr", "4,6,9")).split(",")]:
        for margin in [float(x) for x in str(opt.pop("margin", "0.5,1.5") if "margin" in opt else "0.5,1.5").split(",")]:
            for rmax in (2.0, 4.0):
                for end in ("imp", "ctl"):
                    grid.append(dict(thr=thr, margin=margin, rmax=rmax, end=end))
    print("%s s%d  kick proposals (T123)" % (variant, seed))
    print("thr margin rmax end |   n  real  | missed found |  d ctl2  per match")
    for g in grid:
        n = real = found = nm = 0
        ds = []
        for name, d in data.items():
            t, p, per = d["truth"], d["pred"], d["period"]
            props = kick_proposals(p, per, d["feats"], **g)
            lab = p.copy()
            for s, e, k in props:
                lab[s:e] = k
                n += 1
                real += (t[s:e] == k).mean() >= 0.5
            missed = [(s, e, k) for s, e, k in _team_spells(t, per) if (p[s:e] == k).mean() < 0.5 and (p[s:e] != "D").mean() > 0.5]
            nm += len(missed)
            for s, e, k in missed:
                found += (lab[s:e] == k).mean() >= 0.5
            ds.append(f1(t, lab) - f1(t, p))
        print("%3g %6g %4g %3s | %3d  %.2f  | %3d / %3d   | %+.4f  %s" % (
            g["thr"], g["margin"], g["rmax"], g["end"], n, real / max(1, n), found, nm, np.mean(ds), " ".join("%+.3f" % x for x in ds)), flush=True)


def cmd_missed(opt):
    """What the inputs show in the true possessions the default gives to the other team."""
    from possess import decode

    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    show = int(opt.pop("show", 0))
    data = load_all(variant, seed)
    rows = []
    for name, d in data.items():
        t, p, per, f = d["truth"], d["pred"], d["period"], d["feats"]
        bx, by = f["ball_x"], f["ball_y"]
        fly = decode.flight_dir(bx, by, f["pos"], per, 4.0, 8.0)
        vx, vy = ball_vel(bx, by)
        sp = np.hypot(vx, vy)
        _, dv = impulses(bx, by, 0.0)
        runs = R._runs(t, per)
        for i, (s, e, k, pp) in enumerate(runs):
            if k not in "HA" or pp <= 0 or (p[s:e] == k).mean() >= 0.5 or (p[s:e] != "D").mean() <= 0.5:
                continue
            sg = 1.0 if k == "H" else -1.0
            own = f["d_home"] if k == "H" else f["d_away"]
            opp = f["d_away"] if k == "H" else f["d_home"]
            prev = runs[i - 1][2] if i and runs[i - 1][3] == pp else "X"
            nxt = runs[i + 1][2] if i + 1 < len(runs) and runs[i + 1][3] == pp else "X"
            a, b = max(0, s - 15), s + 15
            vis = ~np.isnan(bx[s:e])
            rows.append(dict(L=(e - s) / FPS, vis=vis.mean(), prev=prev, nxt=nxt,
                             own_min=float(own[s:e].min()), opp_min=float(opp[s:e].min()),
                             near_own=float((own[s:e] < opp[s:e]).mean()),
                             own1=float((own[s:e] < 1.0).mean()), opp1=float((opp[s:e] < 1.0).mean()),
                             fly=float(np.nanmean(fly[s:e] * sg)) if vis.any() else 0.0,
                             vmax=float(np.nanmax(sp[s:e])) if np.isfinite(sp[s:e]).any() else 0.0,
                             dv0=float(dv[a:b].max()), own0=float(own[a:b].min()), opp0=float(opp[a:b].min()),
                             dv1=float(dv[max(0, e - 15):e + 15].max()),
                             pD_after=float((p[e:e + 50] == "D").mean()), pD_before=float((p[max(0, s - 50):s] == "D").mean()),
                             name=name, s=s))
    n = len(rows)
    q = lambda key, m=None: "%.2f / %.2f / %.2f" % tuple(np.percentile([r[key] for r in rows if m is None or m(r)], [25, 50, 75]))   # noqa: E731
    sh = lambda m: "%d (%.2f)" % (sum(1 for r in rows if m(r)), sum(1 for r in rows if m(r)) / max(1, n))                    # noqa: E731
    print("%s s%d  missed true possessions (pred mostly the other team): %d" % (variant, seed, n))
    print("length s %s | visible share %s" % (q("L"), q("vis")))
    print("truth before: other team %s, D %s, same/X %s" % (sh(lambda r: r["prev"] in "HA"), sh(lambda r: r["prev"] == "D"), sh(lambda r: r["prev"] not in "HAD")))
    print("truth after : other team %s, D %s" % (sh(lambda r: r["nxt"] in "HA"), sh(lambda r: r["nxt"] == "D")))
    print("pred D within 2 s after %s, before %s" % (sh(lambda r: r["pD_after"] > 0.2), sh(lambda r: r["pD_before"] > 0.2)))
    print("mostly visible (>= 0.8): %s" % sh(lambda r: r["vis"] >= 0.8))
    v = lambda r: r["vis"] >= 0.8   # noqa: E731
    print("visible ones: own_min %s | opp_min %s | nearest is own, share of frames %s" % (q("own_min", v), q("opp_min", v), q("near_own", v)))
    print("  own within 1 m share %s | opp within 1 m share %s" % (q("own1", v), q("opp1", v)))
    print("  flight toward own (mean, -1..1) %s | vmax %s" % (q("fly", v), q("vmax", v)))
    print("  start: dv %s own %s opp %s | end: dv %s" % (q("dv0", v), q("own0", v), q("opp0", v), q("dv1", v)))
    for lbl, m in (("own never within 1.5 m", lambda r: v(r) and r["own_min"] >= 1.5), ("own within 1 m at the start, opp not within 1.5", lambda r: v(r) and r["own0"] < 1.0 and r["opp0"] >= 1.5),
                   ("both within 1.5 m at the start", lambda r: v(r) and r["own0"] < 1.5 and r["opp0"] < 1.5), ("opp only at the start", lambda r: v(r) and r["own0"] >= 1.5 and r["opp0"] < 1.5),
                   ("nobody within 1.5 m at the start", lambda r: v(r) and r["own0"] >= 1.5 and r["opp0"] >= 1.5),
                   ("flight toward own > 0.3", lambda r: v(r) and r["fly"] > 0.3), ("flight against own < -0.3", lambda r: v(r) and r["fly"] < -0.3),
                   ("own holds within 1 m >= 30 % of the spell", lambda r: v(r) and r["own1"] >= 0.3)):
        print("  %-50s %s" % (lbl, sh(m)))
    for r in rows[:show]:
        print("   %s %6d L %.1f vis %.2f %s>%s own_min %.1f opp_min %.1f near_own %.2f own1 %.2f opp1 %.2f fly %+.2f vmax %.0f | start dv %.1f own %.1f opp %.1f" % (
            r["name"][:8], r["s"], r["L"], r["vis"], r["prev"], r["nxt"], r["own_min"], r["opp_min"], r["near_own"], r["own1"], r["opp1"], r["fly"], r["vmax"], r["dv0"], r["own0"], r["opp0"]))


def line_cross(bx, by, period, margin=0.0):
    """(first frames beyond a line, signed distance inside the pitch per frame)."""
    from possess.deadball2 import HALF_L, HALF_W

    with np.errstate(invalid="ignore"):
        ld = np.minimum(HALF_L - np.abs(bx), HALF_W - np.abs(by))
        cross = np.flatnonzero((ld[:-1] >= -margin) & (ld[1:] < -margin) & (period[1:] > 0)) + 1
    return cross, ld


def crossing_stops(pred, period, bx, by, min_s=2.0, near=1.0):
    """Predicted stoppages (>= min_s, team spell on both sides, same period) with a visible line crossing
    within `near` s of the entry: (a, b, team before, team after, crossing frame, is_end_line)."""
    from possess.deadball2 import HALF_L

    cross, ld = line_cross(bx, by, period)
    runs = R._runs(pred, period)
    out = []
    for i in range(1, len(runs) - 1):
        a, b, k, p = runs[i]
        if k != "D" or p <= 0 or b - a < min_s * FPS:
            continue
        kb, ka = runs[i - 1], runs[i + 1]
        if kb[3] != p or ka[3] != p or kb[2] not in "HA" or ka[2] not in "HA":
            continue
        c = cross[(cross >= a - near * FPS) & (cross <= a + near * FPS)]
        if not len(c):
            continue
        c = int(c[np.argmin(np.abs(c - a))])
        out.append((a, b, kb[2], ka[2], c, bool(abs(bx[c]) > HALF_L), i))
    return out, runs


def cmd_law(opt):
    """Laws 15-17: after the ball crosses a line the OTHER team restarts. Where the default says the same
    team before and after such a stoppage, what does the truth hold?"""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    data = load_all(variant, seed)
    tot = {"same": 0, "other": 0}
    pat = {}
    lens = {"before_B": [], "after_B": []}
    kickgap = []
    tsame = tn = 0
    for name, d in data.items():
        t, p, per, f = d["truth"], d["pred"], d["period"], d["feats"]
        bx, by = f["ball_x"], f["ball_y"]
        truns = R._runs(t, per)
        tidx = np.zeros(len(t), int)
        for i, r in enumerate(truns):
            tidx[r[0]:r[1]] = i
        imp, dv = impulses(bx, by, 4.0)
        stops, _ = crossing_stops(p, per, bx, by)
        for a, b, kb, ka, c, endl, _i in stops:
            # the true stoppage that overlaps ours most
            cand = {}
            for j in np.unique(tidx[a:b]):
                if truns[j][2] == "D":
                    cand[j] = min(b, truns[j][1]) - max(a, truns[j][0])
            if not cand:
                pat_key = "no true D"
                if kb == ka:
                    pat[pat_key] = pat.get(pat_key, 0) + 1
                    tot["same"] += 1
                else:
                    tot["other"] += 1
                continue
            j = max(cand, key=cand.get)
            tb = truns[j - 1] if j > 0 else None
            ta = truns[j + 1] if j + 1 < len(truns) else None
            tbk = tb[2] if tb and tb[2] in "HA" else "?"
            tak = ta[2] if ta and ta[2] in "HA" else "?"
            if tbk in "HA" and tak in "HA":
                tn += 1
                tsame += tbk == tak
            if kb != ka:
                tot["other"] += 1
                continue
            tot["same"] += 1
            A = kb
            key = "before %s after %s" % ("A" if tbk == A else "B" if tbk in "HA" else "?", "A" if tak == A else "B" if tak in "HA" else "?")
            pat[key] = pat.get(key, 0) + 1
            if tbk in "HA" and tbk != A:
                lens["before_B"].append((tb[1] - tb[0]) / FPS)
                pi = imp[(imp < c) & (imp > c - 10 * FPS)]
                kickgap.append(((c - pi[-1]) / FPS if len(pi) else np.nan, (c - tb[0]) / FPS))
            if tak in "HA" and tak != A:
                lens["after_B"].append((ta[1] - ta[0]) / FPS)
    print("%s s%d  predicted stoppages >= 2 s with a visible crossing at the entry: same team before and after %d, other team %d" % (
        variant, seed, tot["same"], tot["other"]))
    print("truth at those stoppages: same team before and after %d of %d (%.2f)" % (tsame, tn, tsame / max(1, tn)))
    print("predicted A-D-A, truth pattern (A = our team on both sides):")
    for k, v in sorted(pat.items(), key=lambda x: -x[1]):
        print("   %-22s %d" % (k, v))
    for k, v in lens.items():
        if v:
            print("   %s: true B spell length quartiles %.1f / %.1f / %.1f s" % ((k,) + tuple(np.percentile(v, [25, 50, 75]))))
    if kickgap:
        g = np.array(kickgap)
        ok = np.isfinite(g[:, 0])
        print("   before_B: last impulse (>= 4 m/s) before the crossing found %.2f; impulse-to-crossing %.1f / %.1f / %.1f s; true B start-to-crossing %.1f / %.1f / %.1f s; abs diff median %.2f s" % (
            (ok.mean(),) + tuple(np.percentile(g[ok, 0], [25, 50, 75])) + tuple(np.percentile(g[:, 1], [25, 50, 75])) + (np.median(np.abs(g[ok, 0] - g[ok, 1])),)))


def law_last_kick(pred, period, bx, by, thr=4.0, lo=2.0, hi=8.0, min_s=2.0, vis_min=0.8, added=None):
    """T124: Laws 15-17. A predicted stoppage that starts at a visible line crossing and has the SAME
    team before and after it: the other team played the ball out. If the last ball impulse (>= thr m/s)
    before the crossing lies lo..hi s before it, the frames from that impulse to the stoppage go to the
    other team."""
    lab = np.array(pred)
    imp, _ = impulses(bx, by, thr)
    stops, runs = crossing_stops(pred, period, bx, by, min_s=min_s)
    for a, b, kb, ka, c, endl, i in stops:
        if kb != ka:
            continue
        pi = imp[(imp < c) & (imp >= c - hi * FPS)]
        if not len(pi):
            continue
        s = int(pi[-1])
        if c - s < lo * FPS or s <= runs[i - 1][0] or np.isnan(bx[s:c]).mean() > 1 - vis_min:
            continue
        lab[s:a] = "A" if kb == "H" else "H"
        if added is not None:
            added.append((s, a, lab[s]))
    return lab


def cmd_lawrun(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    data = load_all(variant, seed)
    print("%s s%d  T124 last kick before a crossing stoppage with the same team on both sides" % (variant, seed))
    for g in (dict(thr=4.0, lo=2.0), dict(thr=4.0, lo=1.5), dict(thr=4.0, lo=3.0), dict(thr=6.0, lo=2.0), dict(thr=3.0, lo=2.0),
              dict(thr=4.0, lo=2.0, hi=5.0), dict(thr=4.0, lo=2.0, hi=12.0)):
        n = real = 0
        ds = []
        for name, d in data.items():
            t, p, per, f = d["truth"], d["pred"], d["period"], d["feats"]
            added = []
            lab = law_last_kick(p, per, f["ball_x"], f["ball_y"], added=added, **g)
            n += len(added)
            real += sum((t[s:e] == k).mean() >= 0.5 for s, e, k in added)
            ds.append(f1(t, lab) - f1(t, p))
        print("%-32s n %3d real %.2f  d ctl2 %+.4f  %s" % (" ".join("%s=%g" % kv for kv in g.items()), n, real / max(1, n), np.mean(ds), " ".join("%+.3f" % x for x in ds)), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    globals()["cmd_" + cmd](a4_short.kv(sys.argv[2:]))
