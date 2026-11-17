"""Envelope L8-autonomous-fable, round 5: learned ball-aware team evidence (T125).

A leave-one-match-out logistic model (shape.fit_logistic, antisymmetric: swapping the teams flips the
sign) of "which team holds the visible ball" from ball / player context around the frame (who was
near before, who is near after, the last and next ball impulse, flight, counts near the ball). Its
logit enters the decoder as one more team emission through the existing tag hook (tag_team / w_tag /
tag_vis; no change in src). Tune matches only, ctl T=2 truth; never the provided match's truth.

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_ml.py probe bench=real2 seed=0
    uv run python scripts/ramlock.py -- uv run python scripts/a8b_ml.py run bench=real2 seed=0 w=0.3,0.7 set=ctx
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a8b_run as R  # noqa: E402
import a8b_r5 as R5  # noqa: E402
import tune_shape  # noqa: E402
from possess import decode, shape  # noqa: E402

FPS = 25.0
OUT = os.path.join(ROOT, "eval", "results")
INF = 1e9


def past_min(x, w):
    """min of x over the w frames ending at each frame."""
    cur, s = x.copy(), 1
    while s < w:
        st = min(s, w - s)
        cur = np.minimum(cur, np.concatenate([np.full(st, INF), cur[:-st]]))
        s += st
    return cur


def fut_min(x, w):
    return past_min(x[::-1], w)[::-1]


def since(mask):
    """frames since the last True (INF before the first)."""
    idx = np.where(mask, np.arange(len(mask)), -10**9)
    return np.minimum(np.arange(len(mask)) - np.maximum.accumulate(idx), INF)


def until(mask):
    return since(mask[::-1])[::-1]


def smooth(x, w):
    return np.convolve(np.nan_to_num(x), np.ones(w) / w, mode="same")


def team_feats(feats, period, dec=None):
    """Antisymmetric per-frame features (home minus away). Returns (X, names)."""
    bx, by = np.asarray(feats["ball_x"], float), np.asarray(feats["ball_y"], float)
    dh, da = np.asarray(feats["d_home"], float), np.asarray(feats["d_away"], float)
    pos = feats["pos"]
    n = len(bx)
    g = lambda d: np.exp(-np.minimum(d, 50.0))   # noqa: E731
    cols, names = [], []

    def add(name, v):
        cols.append(np.nan_to_num(np.asarray(v, float)))
        names.append(name)

    add("g_now", g(dh) - g(da))
    add("in1_now", (dh < 1.0).astype(float) - (da < 1.0))
    P, F = {}, {}
    for w in (13, 25, 50, 100):
        P[w] = g(past_min(dh, w)) - g(past_min(da, w))
        F[w] = g(fut_min(dh, w)) - g(fut_min(da, w))
        add("past%d" % w, P[w])
        add("fut%d" % w, F[w])
    al_h, al_a = (dh < 1.0) & (da >= 1.5), (da < 1.0) & (dh >= 1.5)
    add("since_alone", np.exp(-since(al_h) / 50.0) - np.exp(-since(al_a) / 50.0))
    add("until_alone", np.exp(-until(al_h) / 50.0) - np.exp(-until(al_a) / 50.0))
    imp, _ = R5.impulses(bx, by, 4.0)
    if len(imp):
        att = np.array([g(dh[max(0, i - 3):i + 4].min()) - g(da[max(0, i - 3):i + 4].min()) for i in imp])
        j = np.searchsorted(imp, np.arange(n), side="right") - 1          # last impulse at or before t
        last = np.where(j >= 0, att[np.maximum(j, 0)] * np.exp(-(np.arange(n) - imp[np.maximum(j, 0)]) / 50.0), 0.0)
        j2 = np.minimum(j + 1, len(imp) - 1)
        nxt = np.where(j + 1 < len(imp), att[j2] * np.exp(-np.abs(imp[j2] - np.arange(n)) / 50.0), 0.0)
    else:
        last = nxt = np.zeros(n)
    add("imp_last", last)
    add("imp_next", nxt)
    fly = decode.flight_dir(bx, by, pos, period, 4.0, 8.0)
    fly2 = decode.flight_dir(bx, by, pos, period, 8.0, 16.0)
    flyp = np.concatenate([np.zeros(12), smooth(fly, 25)[:-12]])
    flyf = np.concatenate([smooth(fly, 25)[12:], np.zeros(12)])
    add("fly48", fly)
    add("fly816", fly2)
    add("fly_past", flyp)
    add("fly_fut", flyf)
    with np.errstate(invalid="ignore"):
        dist = np.hypot(pos[:, :, :, 0] - bx[:, None, None], pos[:, :, :, 1] - by[:, None, None])
        n3 = np.nan_to_num((dist < 3.0).sum(axis=2).astype(float))
        n6 = np.nan_to_num((dist < 6.0).sum(axis=2).astype(float))
    add("n3", n3[:, 0] - n3[:, 1])
    add("n6", n6[:, 0] - n6[:, 1])
    add("shape", feats["shape_team"] if feats.get("shape_team") is not None else np.zeros(n))
    free = ((dh >= 1.5) & (da >= 1.5)).astype(float)
    duelp = ((past_min(dh, 25) < 1.5) & (past_min(da, 25) < 1.5)).astype(float)
    vx, vy = R5.ball_vel(bx, by)
    fast = np.nan_to_num((np.hypot(vx, vy) > 8.0).astype(float))
    add("free_fut50", free * F[50])
    add("free_past50", free * P[50])
    add("free_fut100", free * F[100])
    add("free_past100", free * P[100])
    add("free_fly", free * fly)
    add("duel_fut50", duelp * F[50])
    add("duel_fut100", duelp * F[100])
    add("duel_flyfut", duelp * flyf)
    add("fast_past50", fast * P[50])
    add("fast_fut50", fast * F[50])
    ad = shape.attack_dir(pos, np.asarray(period)).astype(float)
    xatt = np.nan_to_num(bx) * ad / 52.5
    add("xatt", xatt)
    add("xatt_free", xatt * free)
    add("xatt_fast", xatt * fast)
    nctx = len(cols)
    if dec is not None:
        dv = np.where(dec == "H", 1.0, np.where(dec == "A", -1.0, 0.0))
        add("dec", dv)
        for s in (25, 50):
            add("dec_m%d" % s, np.concatenate([np.zeros(s), dv[:-s]]))
            add("dec_p%d" % s, np.concatenate([dv[s:], np.zeros(s)]))
    return np.stack(cols, axis=1), names, nctx


def logit(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def load_bench(variant, seed):
    v = "" if variant == "orig" else variant
    out = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        dec = a4_short.decode_one(feats, period, {})
        X, names, nctx = team_feats(feats, period, dec)
        t2 = R.truths(truth, period)
        vis = np.asarray(feats["visible"], bool)
        out[name] = dict(truth=t2, period=period, feats=feats, dec=dec, X=X.astype(np.float32), vis=vis, names=names, nctx=nctx)
    return out


def fit_lomo(data, name, cols, l2, others=()):
    """Model for `name` from the other tune matches (and the same matches of the benches in `others`)."""
    Xs, ys = [], []
    for src in (data,) + tuple(others):
        for m, d in src.items():
            if m == name:
                continue
            t = d["truth"]["ctl2"]
            k = d["vis"] & np.isin(t, ["H", "A"]) & (d["period"] > 0)
            Xs.append(d["X"][k][::3][:, cols].astype(float))
            ys.append((t[k][::3] == "H").astype(float))
    return shape.fit_logistic(np.concatenate(Xs), np.concatenate(ys), l2=l2, intercept=False)


def cmd_probe(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    l2 = float(opt.pop("l2", 10.0))
    data = load_bench(variant, seed)
    names = next(iter(data.values()))["names"]
    nctx = next(iter(data.values()))["nctx"]
    sets = {"ctx": list(range(nctx)), "ctx+dec": list(range(len(names)))}
    print("%s s%d  learned team evidence, leave-one-match-out (visible frames where truth and decoder both name a team)" % (variant, seed))
    for sn, cols in sets.items():
        tot = np.zeros(6)
        for name, d in data.items():
            m = fit_lomo(data, name, cols, l2)
            z = logit(m, d["X"][:, cols].astype(float))
            t, p = d["truth"]["ctl2"], d["dec"]
            k = d["vis"] & np.isin(t, ["H", "A"]) & np.isin(p, ["H", "A"]) & (d["period"] > 0)
            th, ph, zh = t[k] == "H", p[k] == "H", z[k] > 0
            wrong = th != ph
            conf = np.abs(z[k]) > 2.0
            tot += [k.sum(), (zh == th).sum(), (ph == th).sum(), (zh[wrong] == th[wrong]).sum(), wrong.sum(),
                    ((zh != th) & ~wrong & conf).sum()]
            if sn == "ctx" and name == list(data)[0]:
                order = np.argsort(-np.abs(m["w"]))[:14]
                print("   largest weights (%s held out): " % name + ", ".join("%s %+.2f" % (names[cols[i]], m["w"][i]) for i in order))
        print("%-8s model right %.4f | decoder right %.4f | model right where the decoder is wrong %.3f (n %d) | model confidently wrong (abs logit > 2) where the decoder is right: %d frames" % (
            sn, tot[1] / tot[0], tot[2] / tot[0], tot[3] / tot[4], tot[4], tot[5]), flush=True)


def cmd_run(opt):
    """Decoder with the model's logit as one more team emission on visible frames (tag hook)."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    ws = [float(x) for x in str(opt.pop("w", "0.3,0.7")).split(",")]
    scale = float(opt.pop("scale", 2.0))
    l2 = float(opt.pop("l2", 10.0))
    tag = opt.pop("tag", "ml")
    neutral = int(opt.pop("neutral", 0))
    drop = [x for x in str(opt.pop("drop", "")).split(",") if x]
    data = load_bench(variant, seed)
    names = next(iter(data.values()))["names"]
    nctx = next(iter(data.values()))["nctx"]
    cols = [i for i in range(nctx) if not any(names[i].startswith(dp) for dp in drop)]
    labs, tv = {}, {}
    cfgs = ["default"] + ["ml%g" % w for w in ws]
    for name, d in data.items():
        m = fit_lomo(data, name, cols, l2)
        z = np.tanh(logit(m, d["X"][:, cols].astype(float)) / scale)
        z[~d["vis"]] = 0.0
        tv[name] = d["truth"]
        labs["default", name] = d["dec"]
        f2 = dict(d["feats"], tag_team=z)
        for w in ws:
            labs["ml%g" % w, name] = a4_short.decode_one(f2, d["period"], dict(w_tag=w, tag_vis=1.0, tag_neutral=neutral))
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T125 learned team emission (scale %g, l2 %g, drop %s)" % (variant, seed, scale, l2, drop))
    with open(os.path.join(OUT, "a8b_%s_%s_s%d.json" % (tag, variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "w": ws, "scale": scale, "l2": l2, "drop": drop, "neutral": neutral, "results": res, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)

def _bs(x):
    b, sd = x.split(":") if ":" in x else (x, "0")
    return b, int(sd)


def cmd_gate(opt):
    """Pooled leave-one-match-out: the model for a match is fitted on the OTHER tune matches of every
    bench in `train` (minus the test bench itself with excl=1: the transfer check) and scored on `test`."""
    train = [_bs(x) for x in str(opt.pop("train", "real2:0,real1:0,orig:0")).split(",")]
    test = [_bs(x) for x in str(opt.pop("test", "real2:0,real1:0,orig:0")).split(",")]
    ws = [float(x) for x in str(opt.pop("w", "0.5")).split(",")]
    scale, l2, excl = float(opt.pop("scale", 2.0)), float(opt.pop("l2", 10.0)), int(opt.pop("excl", 0))
    neutral = int(opt.pop("neutral", 1))
    drop = [x for x in str(opt.pop("drop", "")).split(",") if x]
    tag = opt.pop("tag", "mlgate")
    tr = {b: load_bench(*b) for b in train}
    d0 = next(iter(tr.values()))
    names, nctx = next(iter(d0.values()))["names"], next(iter(d0.values()))["nctx"]
    cols = [i for i in range(nctx) if not any(names[i].startswith(dp) for dp in drop)]
    out = {}
    for b in test:
        data = tr[b] if b in tr else load_bench(*b)
        src = [v for k, v in tr.items() if not (excl and k[0] == b[0])]
        labs, tv = {}, {}
        cfgs = ["default"] + ["ml%g" % w for w in ws]
        for name, d in data.items():
            m = fit_lomo(src[0], name, cols, l2, others=tuple(src[1:]))
            z = np.tanh(logit(m, d["X"][:, cols].astype(float)) / scale)
            z[~d["vis"]] = 0.0
            tv[name] = d["truth"]
            labs["default", name] = d["dec"]
            f2 = dict(d["feats"], tag_team=z)
            for w in ws:
                labs["ml%g" % w, name] = a4_short.decode_one(f2, d["period"], dict(w_tag=w, tag_vis=1.0, tag_neutral=neutral))
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T125 pooled LOMO (train %s, excl %d, drop %s, scale %g, l2 %g)" % (
            b[0], b[1], ",".join("%s:%d" % x for x in train), excl, drop, scale, l2))
        if b not in tr:
            del data
    with open(os.path.join(OUT, "a8b_%s.json" % tag), "w") as fh:
        json.dump({"train": train, "test": test, "w": ws, "scale": scale, "l2": l2, "excl": excl, "drop": drop, "neutral": neutral,
                   "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)

def load_src(variant, seed, decode_default=True):
    """As load_bench, with the features of src/possess/ballteam.py."""
    from possess import ballteam

    v = "" if variant == "orig" else variant
    out = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        X = ballteam.features(feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"], feats["pos"], period, feats.get("shape_team"))
        out[name] = dict(truth=R.truths(truth, period), period=period, feats=feats, X=X.astype(np.float32),
                         vis=np.asarray(feats["visible"], bool), dec=a4_short.decode_one(feats, period, {}) if decode_default else None)
    return out


def cmd_srcgate(opt):
    """The gate through the src flag (w_bt, feats["bt"] = leave-one-match-out evidence). excl=1: the model for a
    test bench is fitted without that bench (transfer check); excl=0: pooled over all `train` benches."""
    from possess import ballteam

    train = [_bs(x) for x in str(opt.pop("train", "real2:0,real1:0,orig:0")).split(",")]
    test = [_bs(x) for x in str(opt.pop("test", "real2:0,real1:0,orig:0")).split(",")]
    ws = [float(x) for x in str(opt.pop("w", "0.5")).split(",")]
    l2, excl = float(opt.pop("l2", 10.0)), int(opt.pop("excl", 0))
    tag = opt.pop("tag", "a8_T125_src")
    tr = {b: load_src(*b) for b in train}
    cols = list(range(len(ballteam.FEATS)))
    out = {}
    for b in test:
        data = tr[b] if b in tr else load_src(*b)
        src = [v for k, v in tr.items() if not (excl and k[0] == b[0])]
        labs, tv = {}, {}
        cfgs = ["default"] + ["bt%g" % w for w in ws]
        for name, d in data.items():
            m = fit_lomo(src[0], name, cols, l2, others=tuple(src[1:]))
            tv[name] = d["truth"]
            labs["default", name] = d["dec"]
            f2 = dict(d["feats"], bt=ballteam.evidence(d["X"].astype(float), d["vis"], m))
            for w in ws:
                labs["bt%g" % w, name] = a4_short.decode_one(f2, d["period"], dict(w_bt=w))
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T125 w_bt, leave-one-match-out (train %s, excl %d, l2 %g)" % (
            b[0], b[1], ",".join("%s:%d" % x for x in train), excl, l2))
        if b not in tr:
            del data
    with open(os.path.join(OUT, "%s.json" % tag), "w") as fh:
        json.dump({"train": train, "test": test, "w": ws, "l2": l2, "excl": excl, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_fit(opt):
    """ballteam.MODEL: all tune matches on every bench in `train`. Prints the constants."""
    from possess import ballteam

    train = [_bs(x) for x in str(opt.pop("train", "real2:0,real1:0,orig:0")).split(",")]
    tr = [load_src(*b, decode_default=False) for b in train]
    m = fit_lomo(tr[0], "", list(range(len(ballteam.FEATS))), float(opt.pop("l2", 10.0)), others=tuple(tr[1:]))
    fmt = lambda v: "[" + ", ".join("%.5g" % x for x in np.atleast_1d(v)) + "]"   # noqa: E731
    print("MODEL: Dict[str, np.ndarray] = {")
    for k in ("mu", "sd", "w"):
        print('    "%s": np.array(%s),' % (k, fmt(m[k])))
    print('    "b": %.5g,' % float(np.atleast_1d(m["b"])[0]))
    print("}")
    for k, w in sorted(zip(ballteam.FEATS, m["w"]), key=lambda x: -abs(x[1])):
        print("   %-14s %+.3f" % (k, w))

def cmd_prov(opt):
    """Provided match INPUT only (no truth): how the evidence of ballteam.MODEL looks there against the
    external benches (same model): strength, and agreement of its sign with the default team label."""
    import a8_tags
    from possess import ballteam, pipeline

    def stats(z, lab, vis, per):
        k = vis & (per > 0) & np.isin(lab, ["H", "A"])
        s = np.where(lab[k] == "H", 1.0, -1.0)
        return "visible team frames %7d | mean abs %.3f | abs > 0.5: %.3f | sign agrees with the default label %.3f | agrees where abs > 0.5: %.3f | share named home by the model %.3f, by the label %.3f" % (
            k.sum(), np.abs(z[k]).mean(), (np.abs(z[k]) > 0.5).mean(), (np.sign(z[k]) == s).mean(),
            (np.sign(z[k]) == s)[np.abs(z[k]) > 0.5].mean(), (z[k] > 0).mean(), (s > 0).mean())

    _, _, f, per, base = a8_tags.challenge_default()
    vis = ~np.isnan(f["ball_x"])
    feats = {"d_home": np.nan_to_num(f["d_home"], nan=99.0), "d_away": np.nan_to_num(f["d_away"], nan=99.0), "visible": vis,
             "ball_x": f["ball_x"], "ball_y": f["ball_y"], "pos": f["pos"]}
    feats.update(shape.decoder_inputs(f["pos"], per))
    print("provided  " + stats(ballteam.from_feats(feats, per), base, vis, per), flush=True)
    for b in ("real2", "real1", "orig"):
        for name in tune_shape.TUNE:
            truth, period, ft = a4_short.load(name, "" if b == "orig" else b, 0)
            lab = a4_short.decode_one(ft, period, {})
            print("%-5s %-9s" % (b, name[:9]) + stats(ballteam.from_feats(ft, period), lab, np.asarray(ft["visible"], bool), period), flush=True)

def cmd_tagcheck(opt):
    """Provided match INPUT only (no truth): Smart Tagging one-team clips as an independent check of what
    w_bt changes (as row 31 did for the flight evidence). Nothing is chosen on it."""
    import a8_tags
    from possess import pipeline, smarttags as st

    tags, home, f, per, base = a8_tags.challenge_default()
    prior = st.phase_prior(tags, len(base), home)
    one = (np.abs(prior) == 1) & (per > 0)
    sg = lambda lab: np.where(lab == "H", 1, np.where(lab == "A", -1, 0))   # noqa: E731
    b = sg(base)
    lv = one & (b != 0)
    print("default: agreement with the one-team tag frames where we name a team %.4f (n %d)" % ((b[lv] == prior[lv]).mean(), lv.sum()))
    for w in [float(x) for x in str(opt.pop("w", "0.3,0.5,0.7")).split(",")]:
        lab = pipeline.label(f, period=per, dec_kw=dict(w_bt=w))
        c = sg(lab)
        lv2 = one & (c != 0)
        flip = one & (b != 0) & (c != 0) & (b != c)
        print("w_bt %.1f: agreement %.4f (n %d) | tagged frames where the team changed: %d, tag sides with the new label %.3f" % (
            w, (c[lv2] == prior[lv2]).mean(), lv2.sum(), flip.sum(), (c[flip] == prior[flip]).mean()), flush=True)

def hidden_feats(z, vis, period, shape_team):
    """T126 features on hidden-ball frames: the visible evidence z just before and just after the hidden
    block, decayed with the distance to that edge, and the ball-free shape evidence. Antisymmetric."""
    n = len(z)
    X = np.zeros((n, 7))
    hid = ~vis & (period > 0)
    e = np.flatnonzero(np.diff(np.concatenate([[0], hid.astype(int), [0]])))
    for a, b in zip(e[::2], e[1::2]):
        pa, pb = max(0, a - 25), min(n, b + 25)
        zb = z[pa:a][vis[pa:a] & (period[pa:a] == period[a])]
        za = z[b:pb][vis[b:pb] & (period[b:pb] == period[a])]
        zb, za = (zb.mean() if len(zb) else 0.0), (za.mean() if len(za) else 0.0)
        t = np.arange(a, b)
        for i, tau in enumerate((25.0, 100.0, 400.0)):
            X[a:b, i] = zb * np.exp(-(t - a) / tau)
            X[a:b, 3 + i] = za * np.exp(-(b - 1 - t) / tau)
    X[:, 6] = np.where(hid, np.nan_to_num(shape_team), 0.0)
    return X, hid


def cmd_hid(opt):
    """T126: learned team evidence on HIDDEN frames (leave-one-match-out, same bench), on top of w_bt."""
    from possess import ballteam

    variant, seed = opt.pop("bench", "real1"), int(opt.pop("seed", 0))
    w = float(opt.pop("w", 0.5))
    whs = [float(x) for x in str(opt.pop("wh", "0.3,0.6,1.0")).split(",")]
    data = load_src(variant, seed)
    cols = list(range(len(ballteam.FEATS)))
    labs, tv = {}, {}
    cfgs = ["bt"] + ["h%g" % x for x in whs]
    tot = np.zeros(5)
    for name, d in data.items():
        m = fit_lomo(data, name, cols, 10.0)
        Xs, ys, H = [], [], {}
        for nm, dd in data.items():
            z = ballteam.evidence(dd["X"].astype(float), dd["vis"], m)
            Xh, hid = hidden_feats(z, dd["vis"], dd["period"], dd["feats"].get("shape_team"))
            H[nm] = (z, Xh, hid)
            if nm != name:
                t = dd["truth"]["ctl2"]
                k = hid & np.isin(t, ["H", "A"])
                Xs.append(Xh[k][::3])
                ys.append((t[k][::3] == "H").astype(float))
        mh = shape.fit_logistic(np.concatenate(Xs), np.concatenate(ys), l2=10.0, intercept=False)
        z, Xh, hid = H[name]
        zh = np.tanh(logit(mh, Xh) / 2.0) * hid
        tv[name] = d["truth"]
        base = a4_short.decode_one(dict(d["feats"], bt=z), d["period"], dict(w_bt=w))
        labs["bt", name] = base
        t = d["truth"]["ctl2"]
        k = hid & np.isin(t, ["H", "A"]) & np.isin(base, ["H", "A"])
        th, ph, mhh = t[k] == "H", base[k] == "H", zh[k] > 0
        tot += [k.sum(), (mhh == th).sum(), (ph == th).sum(), (mhh[ph != th] == th[ph != th]).sum(), (ph != th).sum()]
        for x in whs:
            labs["h%g" % x, name] = a4_short.decode_one(dict(d["feats"], bt=z + zh * (x / w)), d["period"], dict(w_bt=w))
        if name == list(data)[0]:
            print("   hidden model weights (zb 1 / 4 / 16 s, za 1 / 4 / 16 s, shape): " + " ".join("%+.2f" % x for x in mh["w"]))
    print("%s s%d hidden frames where truth and decoder name a team: model right %.4f, decoder right %.4f, model right where the decoder is wrong %.3f (n %d)" % (
        variant, seed, tot[1] / tot[0], tot[2] / tot[0], tot[3] / max(1, tot[4]), tot[4]))
    res = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T126 hidden-frame evidence on top of w_bt=%g (baseline = bt)" % (variant, seed, w))
    with open(os.path.join(OUT, "a8b_T126_%s_s%d.json" % (variant, seed)), "w") as fh:
        json.dump({"bench": variant, "seed": seed, "w": w, "wh": whs, "results": res, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]](a4_short.kv(sys.argv[2:]))
