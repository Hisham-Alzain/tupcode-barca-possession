"""Envelope L8-autonomous-fable, round 7. T139: learned re-dating of the boundaries we already have.

`a8b_r7.py edge` shows that moving each predicted boundary onto its true partner (same from / to labels,
within 3 s) is worth +0.043..+0.052 ctl2 (team -> team +0.026..+0.031, D -> team +0.012..+0.017). Here a
ridge regression of (our boundary - true boundary) on local ball / touch features, one model per kind of
boundary, leave-one-match-out on the external tune matches, moves the boundary by its prediction.

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r7b.py bd test=real2:0,real1:0,orig:0
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a8b_r6 as R6  # noqa: E402
import a8b_r7 as R7  # noqa: E402
import a8b_run as R  # noqa: E402
import tune_shape  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results")
BDF = ("old_last", "old_miss", "new_first", "new_miss", "new1_first", "new1_miss", "imp_last", "imp_miss", "imp_next", "impn_miss",
       "z_cross", "z_miss", "vis_p", "vis_win", "speed_p", "hid_before", "hid_after", "len_prev", "len_next", "fast_first", "fast_miss")


def _kind(fr, to):
    return "TT" if "D" not in (fr, to) else ("TD" if to == "D" else "DT")


def _first(m, p, last=False):
    """Offset (s) from index p of the first (last) True of the window m; [0, 1] when none."""
    i = np.flatnonzero(m)
    if not len(i):
        return [0.0, 1.0]
    return [((i[-1] if last else i[0]) - p) / 25.0, 0.0]


def bd_prepare(feats, z):
    from possess import ballteam
    bx, by = np.asarray(feats["ball_x"], float), np.asarray(feats["ball_y"], float)
    vis = np.asarray(feats["visible"], bool)
    dh = np.where(vis, np.nan_to_num(np.asarray(feats["d_home"], float), nan=99.0), 99.0)
    da = np.where(vis, np.nan_to_num(np.asarray(feats["d_away"], float), nan=99.0), 99.0)
    vx, vy = ballteam.ball_vel(bx, by)
    with np.errstate(invalid="ignore"):
        sp = np.nan_to_num(np.hypot(vx, vy))
    im = np.zeros(len(bx), bool)
    im[ballteam.impulses(bx, by, 4.0)] = True
    return dict(dh=dh, da=da, dmin=np.minimum(dh, da), far=np.full(len(bx), 99.0), vis=vis, sp=sp, im=im, z=R7._box(z, 6))


def bd_feats(P, period, bounds):
    """One row per boundary (frame, from, to); times in seconds relative to the boundary, window -3..+2 s."""
    n = len(period)
    fr_all = [x[0] for x in bounds]
    X = np.zeros((len(bounds), len(BDF)))
    for r, (p, fr, to) in enumerate(bounds):
        lo, hi = max(0, p - 75), min(n, p + 50)
        w = slice(lo, hi)
        dn = P["dmin"] if to == "D" else (P["dh"] if to == "H" else P["da"])
        do = P["dmin"] if fr == "D" else (P["dh"] if fr == "H" else P["da"])
        on = P["far"] if to == "D" else (P["da"] if to == "H" else P["dh"])
        oo = P["far"] if fr == "D" else (P["da"] if fr == "H" else P["dh"])
        new_t = (dn[w] < 1.5) & (dn[w] < on[w])
        sgn = 1.0 if to == "H" else (-1.0 if to == "A" else (-1.0 if fr == "H" else 1.0))
        v = _first((do[w] < 1.5) & (do[w] < oo[w]), p - lo, last=True)
        v += _first(new_t, p - lo)
        v += _first(new_t & (dn[w] < 1.0), p - lo)
        v += _first(P["im"][lo:min(n, p + 13)], p - lo, last=True)
        v += _first(P["im"][p:hi], 0)
        v += _first((P["z"][w] * sgn < 0) & P["vis"][w], p - lo, last=True)
        v += [float(P["vis"][p]), float(P["vis"][w].mean()), min(P["sp"][p], 25.0) / 25.0,
              float((~P["vis"][max(0, p - 25):p]).mean()) if p > 0 else 0.0, float((~P["vis"][p:min(n, p + 25)]).mean())]
        pa = fr_all[r - 1] if r > 0 and period[fr_all[r - 1]] == period[p] else p - 250
        pb = fr_all[r + 1] if r + 1 < len(bounds) and period[fr_all[r + 1]] == period[p] else p + 250
        v += [min((p - pa) / 25.0, 10.0) / 10.0, min((pb - p) / 25.0, 10.0) / 10.0]
        v += _first(P["sp"][w] > 6.0, p - lo)
        X[r] = v
    return X


def _ridge(X, y, l2=3.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1)
    A = Z.T @ Z + l2 * np.diag([1.0] * X.shape[1] + [0.0])
    return dict(mu=mu, sd=sd, w=np.linalg.solve(A, Z.T @ y))


def _ridge_pred(m, X):
    return np.concatenate([(X - m["mu"]) / m["sd"], np.ones((len(X), 1))], axis=1) @ m["w"]


def _partner(tb, tf, p, fr, to, win):
    lo, hi = np.searchsorted(tf, p - win), np.searchsorted(tf, p + win + 1)
    c = [tb[j][0] for j in range(lo, hi) if tb[j][1] == fr and tb[j][2] == to]
    return min(c, key=lambda v: abs(v - p)) if c else None


def move_bounds(lab, period, bounds, shift, sel, margin=5):
    """Boundary r moves by -shift[r] frames (positive shift = ours was late), kept `margin` frames inside its neighbours."""
    out = np.array(lab)
    fr_all = [x[0] for x in bounds]
    for r, (p, fr, to) in enumerate(bounds):
        if not sel[r]:
            continue
        pa = fr_all[r - 1] if r > 0 and period[fr_all[r - 1]] == period[p] else p - 250
        pb = fr_all[r + 1] if r + 1 < len(bounds) and period[fr_all[r + 1]] == period[p] else p + 250
        q = int(np.clip(p - int(round(shift[r])), max(pa + margin, 0), min(pb - margin, len(lab) - 1)))
        while q < p and period[q] != period[p]:
            q += 1
        while q > p and period[q - 1] != period[p]:
            q -= 1
        if q < p:
            out[q:p] = to
        elif q > p:
            out[p:q] = fr
    return out


def cmd_bd(opt):
    """excl=1: the model for a bench is fitted without that bench (transfer). gain scales the predicted move."""
    pool = R6.benches(opt.pop("pool", R6.POOL3))
    test = R6.benches(opt.pop("test", R6.POOL3))
    kinds = str(opt.pop("kinds", "TT,DT,TD")).split(",")
    excl, l2, win = int(opt.pop("excl", 0)), float(opt.pop("l2", 3.0)), int(float(opt.pop("win", 3.0)) * 25)
    gain = float(opt.pop("gain", 1.0))
    tag = opt.pop("tag", "T139_bd")
    bt_models = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]

    def prep(b):
        data = R6.load(*b)
        for name, d in data.items():
            z = np.where(d["vis"], np.tanh(R6.logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            d["lab"] = a4_short.decode_one(dict(d["feats"], bt=z), d["period"], dict(w_bt=0.5))
            d["bounds"] = R7._bounds(d["lab"], d["period"])
            d["BF"] = bd_feats(bd_prepare(d["feats"], z), d["period"], d["bounds"])
            tb = R7._bounds(d["truth"]["ctl2"], d["period"])
            tf = np.array([x[0] for x in tb])
            q = [_partner(tb, tf, p, fr, to, win) for p, fr, to in d["bounds"]]
            d["off"] = np.array([np.nan if v is None else float(x[0] - v) for v, x in zip(q, d["bounds"])])
            d["kind"] = np.array([_kind(fr, to) for _, fr, to in d["bounds"]])
            d["feats"] = None
        return data

    tr, keep = {}, {}
    for b in pool:
        data = prep(b)
        for name, d in data.items():
            tr[b, name] = (d["BF"], d["off"], d["kind"])
        if b in test:
            keep[b] = data
        del data
    out = {}
    for b in test:
        data = keep.pop(b) if b in keep else prep(b)
        labs, tv = {}, {}
        cfgs = ["bt"] + kinds + ["all", "oracle"]
        err = {k: [0.0, 0.0, 0] for k in kinds}
        for name, d in data.items():
            tv[name] = d["truth"]
            labs["bt", name] = d["lab"]
            shift = np.zeros(len(d["bounds"]))
            for k in kinds:
                rows = [v for (bb, mm), v in tr.items() if mm != name and not (excl and bb[0] == b[0])]
                Xs = np.concatenate([v[0][(v[2] == k) & ~np.isnan(v[1])] for v in rows])
                ys = np.concatenate([v[1][(v[2] == k) & ~np.isnan(v[1])] for v in rows])
                m = _ridge(Xs, ys, l2)
                sel = d["kind"] == k
                shift[sel] = gain * _ridge_pred(m, d["BF"][sel])
                ok = sel & ~np.isnan(d["off"])
                err[k][0] += np.abs(d["off"][ok]).sum() / 25.0
                err[k][1] += np.abs(d["off"][ok] - shift[ok]).sum() / 25.0
                err[k][2] += int(ok.sum())
                labs[k, name] = move_bounds(d["lab"], d["period"], d["bounds"], shift, sel)
                if name == tune_shape.TUNE[0] and b == test[0]:
                    order = np.argsort(-np.abs(m["w"][:-1]))[:8]
                    print("   %s model (n %d, without %s), frames per sd: " % (k, len(ys), name) + ", ".join("%s %+.1f" % (BDF[i], m["w"][i]) for i in order) + " | intercept %+.1f" % m["w"][-1], flush=True)
            labs["all", name] = move_bounds(d["lab"], d["period"], d["bounds"], shift, np.isin(d["kind"], kinds))
            labs["oracle", name] = move_bounds(d["lab"], d["period"], d["bounds"], np.nan_to_num(d["off"]), ~np.isnan(d["off"]), margin=0)
        print("%s s%d mean abs date error of partnered boundaries (s), before -> after: " % b + ", ".join("%s %.3f -> %.3f (n %d)" % (k, v[0] / v[2], v[1] / v[2], v[2]) for k, v in err.items()), flush=True)
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T139 learned boundary re-dating (pool %s, excl %d, l2 %g, gain %g)" % (
            b[0], b[1], ",".join("%s:%d" % x for x in pool), excl, l2, gain))
        del data
    with open(os.path.join(OUT, "a8_r7_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "kinds": kinds, "excl": excl, "l2": l2, "gain": gain, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_snap(opt):
    """T140: are the true team -> team dates ball events? Share of true / our TT boundaries within 0.2 s of a ball
    impulse and of the new side's first touch; rule: our TT boundary snaps to the nearest candidate within w s.
    cand = imp (any impulse), impn (impulse with the new side nearer), touch (onset of an uncontested touch of the new side)."""
    test = R6.benches(opt.pop("test", R6.POOL3))
    ws = [float(x) for x in str(opt.pop("w", "0.5,1.0")).split(",")]
    cands = str(opt.pop("cand", "imp,impn,touch")).split(",")
    bt_models = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]
    out = {}
    for b in test:
        data = R6.load(*b)
        labs, tv = {}, {}
        cfgs = ["bt"] + ["%s%g" % (c, w) for c in cands for w in ws]
        st = np.zeros(7)
        for name, d in data.items():
            z = np.where(d["vis"], np.tanh(R6.logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            lab = a4_short.decode_one(dict(d["feats"], bt=z), d["period"], dict(w_bt=0.5))
            P = bd_prepare(d["feats"], z)
            period, n = d["period"], len(lab)
            tv[name], labs["bt", name] = d["truth"], lab
            bounds = R7._bounds(lab, period)
            imf = np.flatnonzero(P["im"])
            near = lambda arr, p: (np.abs(arr - p).min() if len(arr) else 10 ** 6)   # noqa: E731
            tb = [x for x in R7._bounds(d["truth"]["ctl2"], period) if "D" not in x[1:]]
            vis = P["vis"]
            for p, fr, to in tb:
                if vis[p]:
                    dn, do = (P["dh"], P["da"]) if to == "H" else (P["da"], P["dh"])
                    on = np.flatnonzero((dn[max(0, p - 50):p + 50] < 1.5) & (dn[max(0, p - 50):p + 50] < do[max(0, p - 50):p + 50])) + max(0, p - 50)
                    st[:3] += [1, near(imf, p) <= 5, bool(len(on)) and abs(on[0] - p) <= 5]
            for p, fr, to in bounds:
                if "D" not in (fr, to) and vis[p]:
                    st[3:5] += [1, near(imf, p) <= 5]
            for c in cands:
                for w in ws:
                    shift, sel = np.zeros(len(bounds)), np.zeros(len(bounds), bool)
                    for r, (p, fr, to) in enumerate(bounds):
                        if "D" in (fr, to):
                            continue
                        dn, do = (P["dh"], P["da"]) if to == "H" else (P["da"], P["dh"])
                        lo, hi = max(0, p - int(w * 25)), min(n, p + int(w * 25) + 1)
                        if c == "imp":
                            k = imf[(imf >= lo) & (imf < hi)]
                        elif c == "impn":
                            k = imf[(imf >= lo) & (imf < hi)]
                            k = k[dn[k] < do[k]]
                        else:
                            t = (dn[lo:hi] < 1.5) & (dn[lo:hi] < do[lo:hi])
                            k = np.flatnonzero(t[1:] & ~t[:-1]) + lo + 1
                        if len(k):
                            q = k[np.abs(k - p).argmin()]
                            shift[r], sel[r] = p - q, True
                    labs["%s%g" % (c, w), name] = move_bounds(lab, period, bounds, shift, sel)
                    if c == cands[0] and w == ws[0]:
                        st[5:7] += [sel.sum(), np.abs(shift[sel]).sum() / 25.0]
        print("%s s%d visible true team -> team boundaries %d: within 0.2 s of a ball impulse %.3f, of the new side's first touch in +-2 s %.3f | ours %d: of an impulse %.3f | %s%g moves %d boundaries by %.2f s on average" % (
            b[0], b[1], st[0], st[1] / st[0], st[2] / st[0], st[3], st[4] / st[3], cands[0], ws[0], st[5], st[6] / max(st[5], 1)), flush=True)
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T140 team -> team boundary snapped to a ball event" % b)
        del data
    with open(os.path.join(OUT, "a8_r7_T140_snap.json"), "w") as fh:
        json.dump({"w": ws, "cand": cands, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]](a4_short.kv(sys.argv[2:]))
