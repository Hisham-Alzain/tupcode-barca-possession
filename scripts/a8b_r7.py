"""Envelope L8-autonomous-fable, round 7.

T138 `hd`: learned stoppage evidence on HIDDEN-ball frames from the context of the hidden block (how the
ball left view, how it came back, how long the block is, the ball-free stoppage log-odds around it).
Leave-one-match-out on the external tune matches; reaches the decoder as feats["hd"] (w_hd hook). The
baseline of every table is the main default with the leave-one-match-out w_bt evidence (T125).

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r7.py hd w=0.5,1.0 test=real2:0,real1:0,orig:0
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a8b_r6 as R6  # noqa: E402
import a8b_run as R  # noqa: E402
import tune_shape  # noqa: E402
from possess import shape  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results")
HDF = ("sd", "sd1", "sd3", "sd_blk", "p90", "p90_blk", "len", "short", "long", "t_in", "t_out", "pre_ok", "pre_line",
       "pre_out", "pre_speed", "pre_outward", "pre_slow", "pre_dmin", "pre_out_dec", "pre_outward_dec", "post_ok",
       "post_line", "post_out", "post_speed", "post_slow", "post_dmin", "post_edge", "post_goal", "post_corner",
       "jump_v", "jump_d", "post_slow_dec", "post_edge_dec", "hid10")


def _box(x, w):
    k = np.ones(2 * w + 1)
    return np.convolve(np.nan_to_num(x), k, "same") / np.convolve(np.ones(len(x)), k, "same")


def _edge(bx, by, dmin, i, j):
    """State of the visible ball at frame i (j = the frame 5 further away from the block): line distance,
    outside, speed, speed towards the nearest line, slow, nearest player."""
    x, y = bx[i], by[i]
    lx, ly = 52.5 - abs(x), 34.0 - abs(y)
    line = min(lx, ly)
    sp = outward = 0.0
    if j is not None and not np.isnan(bx[j]):
        vx, vy = (x - bx[j]) * 25.0 / abs(i - j), (y - by[j]) * 25.0 / abs(i - j)
        sp = float(np.hypot(vx, vy))
        outward = float(vx * np.sign(x) if lx < ly else vy * np.sign(y))
    return line, sp, outward, float(dmin[i])


def hd_feats(feats, period):
    """[frames, len(HDF)], zero on visible frames; mask of hidden in-play frames."""
    bx, by = np.asarray(feats["ball_x"], float), np.asarray(feats["ball_y"], float)
    vis = np.asarray(feats["visible"], bool) & ~np.isnan(bx)
    dmin = np.minimum(np.nan_to_num(np.asarray(feats["d_home"], float), nan=99.0), np.nan_to_num(np.asarray(feats["d_away"], float), nan=99.0))
    n = len(bx)
    sd = np.nan_to_num(np.asarray(feats["shape_dead"], float)) if feats.get("shape_dead") is not None else np.zeros(n)
    p90 = np.minimum(np.nan_to_num(np.asarray(feats["edge_p90"], float)), 8.0) / 8.0 if feats.get("edge_p90") is not None else np.zeros(n)
    hid = ~vis & (period > 0)
    X = np.zeros((n, len(HDF)))
    c = {k: i for i, k in enumerate(HDF)}
    X[:, c["sd"]], X[:, c["sd1"]], X[:, c["sd3"]], X[:, c["p90"]] = sd, _box(sd, 25), _box(sd, 75), p90
    X[:, c["hid10"]] = _box(hid.astype(float), 125)
    e = np.flatnonzero(np.diff(np.concatenate([[0], hid.astype(int), [0]])))
    for a, b in zip(e[::2], e[1::2]):
        t = np.arange(a, b)
        L = (b - a) / 25.0
        din, dout = np.exp(-(t - a) / 75.0), np.exp(-(b - 1 - t) / 75.0)
        X[a:b, c["sd_blk"]], X[a:b, c["p90_blk"]] = sd[a:b].mean(), p90[a:b].mean()
        X[a:b, c["len"]], X[a:b, c["short"]], X[a:b, c["long"]] = min(L, 30.0) / 30.0, L < 3.0, L > 15.0
        X[a:b, c["t_in"]], X[a:b, c["t_out"]] = din, dout
        if a > 0 and vis[a - 1] and period[a - 1] == period[a]:
            line, sp, ow, dm = _edge(bx, by, dmin, a - 1, a - 6 if a >= 6 else None)
            X[a:b, c["pre_ok"]] = 1.0
            X[a:b, c["pre_line"]], X[a:b, c["pre_out"]] = np.clip(line, 0, 10) / 10.0, line < 0
            X[a:b, c["pre_speed"]], X[a:b, c["pre_outward"]] = min(sp, 25.0) / 25.0, np.clip(ow, 0, 20) / 20.0
            X[a:b, c["pre_slow"]], X[a:b, c["pre_dmin"]] = sp < 1.0, min(dm, 10.0) / 10.0
            X[a:b, c["pre_out_dec"]], X[a:b, c["pre_outward_dec"]] = (line < 1.0) * din, np.clip(ow, 0, 20) / 20.0 * (line < 10.0) * din
        if b < n and vis[b] and period[b] == period[a]:
            line, sp, ow, dm = _edge(bx, by, dmin, b, b + 5 if b + 5 < n else None)
            X[a:b, c["post_ok"]] = 1.0
            X[a:b, c["post_line"]], X[a:b, c["post_out"]] = np.clip(line, 0, 10) / 10.0, line < 0
            X[a:b, c["post_speed"]], X[a:b, c["post_slow"]] = min(sp, 25.0) / 25.0, sp < 1.0
            X[a:b, c["post_dmin"]], X[a:b, c["post_edge"]] = min(dm, 10.0) / 10.0, line < 2.0
            X[a:b, c["post_goal"]] = (abs(bx[b]) > 46.0) and (abs(by[b]) < 10.0)
            X[a:b, c["post_corner"]] = (52.5 - abs(bx[b]) < 2.0) and (34.0 - abs(by[b]) < 2.0)
            X[a:b, c["post_slow_dec"]], X[a:b, c["post_edge_dec"]] = (sp < 1.0) * dout, (line < 2.0) * dout
            if a > 0 and vis[a - 1] and period[a - 1] == period[a]:
                j = float(np.hypot(bx[b] - bx[a - 1], by[b] - by[a - 1]))
                X[a:b, c["jump_v"]], X[a:b, c["jump_d"]] = min(j / L, 15.0) / 15.0, min(j, 50.0) / 50.0
    X[~hid] = 0.0
    return X, hid


def _rows(b):
    data = R6.load(*b)
    for name, d in data.items():
        d["HX"], d["hid"] = hd_feats(d["feats"], d["period"])
    return data


def _fit(rows, l2=10.0):
    Xs, ys = np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])
    m = shape.fit_logistic(Xs, ys, l2=l2, intercept=True)
    m["base"] = float(np.log(ys.mean() / (1 - ys.mean())))
    return m


def cmd_hd(opt):
    """T138. excl=1: the model for a bench is fitted without that bench (transfer)."""
    pool = R6.benches(opt.pop("pool", R6.POOL3))
    test = R6.benches(opt.pop("test", R6.POOL3))
    ws = [float(x) for x in str(opt.pop("w", "0.5,1.0")).split(",")]
    excl, l2, scale = int(opt.pop("excl", 0)), float(opt.pop("l2", 10.0)), float(opt.pop("scale", 2.0))
    wbd = float(opt.pop("wbd", 0.0))
    tag = opt.pop("tag", "T138_hd")
    drop = [x for x in str(opt.pop("drop", "")).split(",") if x]
    cols = [i for i, nm in enumerate(HDF) if nm not in drop]
    bt_models = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]
    tr, keep = {}, {}
    for b in pool:
        data = _rows(b)
        for name, d in data.items():
            k = d["hid"]
            tr[b, name] = (d["HX"][k][::3][:, cols], (d["truth"]["ctl2"][k][::3] == "D").astype(float))
        if b in test:
            keep[b] = data
        del data
    out = {}
    for b in test:
        data = keep.pop(b) if b in keep else _rows(b)
        labs, tv = {}, {}
        cfgs = ["bt"] + ["hd%g" % w for w in ws]
        st, nb = np.zeros(3), {}
        for name, d in data.items():
            rows = [v for (bb, mm), v in tr.items() if mm != name and not (excl and bb[0] == b[0])]
            if not sum(len(r[1]) for r in rows) or not d["hid"].any():
                m, ev = None, np.zeros(len(d["period"]))
            else:
                m = _fit(rows, l2)
                lo = ((d["HX"][:, cols] - m["mu"]) / m["sd"]) @ m["w"] + float(m["b"])
                ev = np.where(d["hid"], np.tanh((lo - m["base"]) / scale), 0.0)
            tv[name] = d["truth"]
            f0 = dict(d["feats"])
            f0["bt"] = np.where(d["vis"], np.tanh(R6.logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            bkw = dict(w_bt=0.5)
            if wbd:
                bkw["w_bd"] = wbd
            labs["bt", name] = a4_short.decode_one(dict(f0), d["period"], dict(bkw))
            for w in ws:
                labs["hd%g" % w, name] = a4_short.decode_one(dict(f0, hd=ev), d["period"], dict(bkw, w_hd=w))
            k = d["hid"]
            if m is not None and k.any():
                t = d["truth"]["ctl2"][k] == "D"
                st += [((lo[k] > 0) == t).mean(), ((labs["bt", name][k] == "D") == t).mean(), t.mean()]
                Lb = d["HX"][k][:, HDF.index("len")] * 30.0
                for nm, lo_, hi_ in (("<3s", 0, 3), ("3-8s", 3, 8), ("8-15s", 8, 15), (">=15s", 15, 99)):
                    q = (Lb >= lo_) & (Lb < hi_)
                    if q.any():
                        v = nb.setdefault(nm, np.zeros(5))
                        v += [q.sum(), ((lo[k][q] > 0) == t[q]).sum(), ((labs["bt", name][k][q] == "D") == t[q]).sum(), t[q].sum(),
                              ((labs["hd%g" % ws[-1], name][k][q] == "D") == t[q]).sum()]
                if name == tune_shape.TUNE[0] and b == test[0]:
                    order = np.argsort(-np.abs(m["w"]))[:14]
                    print("   hidden stoppage model (without %s), largest weights: " % name + ", ".join("%s %+.2f" % (HDF[cols[i]], m["w"][i]) for i in order), flush=True)
        st /= len(data)
        print("%s s%d hidden frames: model right on D-vs-live %.4f | decoder %.4f | share of D %.3f" % (b[0], b[1], st[0], st[1], st[2]), flush=True)
        for nm, v in nb.items():
            print("   block %-6s frames %7d | model %.3f | decoder %.3f | decoder + hd%g %.3f | share of D %.3f" % (
                nm, v[0], v[1] / v[0], v[2] / v[0], ws[-1], v[4] / v[0], v[3] / v[0]))
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T138 learned hidden stoppage evidence (pool %s, excl %d, w_bd %g, drop %s)" % (
            b[0], b[1], ",".join("%s:%d" % x for x in pool), excl, wbd, drop))
        del data
    with open(os.path.join(OUT, "a8_r7_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "w": ws, "excl": excl, "w_bd": wbd, "drop": drop, "scale": scale, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def _bounds(lab, period):
    """[(frame, from, to)] for label changes inside one period."""
    i = np.flatnonzero((lab[1:] != lab[:-1]) & (period[1:] == period[:-1]) & (period[1:] > 0)) + 1
    return [(int(k), lab[k - 1], lab[k]) for k in i]


def cmd_edge(opt):
    """Diagnosis: how much of the loss is the DATE of a boundary we do have. Each predicted boundary with a true
    boundary of the same (from, to) pair within `win` s is moved onto it (oracle), by kind of boundary."""
    win = int(float(opt.pop("win", 3.0)) * 25)
    bt_models = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]
    for b in R6.benches(opt.pop("test", "real2:0,real1:0,orig:0")):
        data = R6.load(*b)
        res, offs = {}, {}
        for name, d in data.items():
            z = np.where(d["vis"], np.tanh(R6.logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            lab = a4_short.decode_one(dict(d["feats"], bt=z), d["period"], dict(w_bt=0.5))
            t, period = d["truth"]["ctl2"], d["period"]
            f1 = R6.Fast(t)
            tb = _bounds(t, period)
            tf = np.array([x[0] for x in tb])
            o = {"score": f1(lab)}
            outs = {k: np.array(lab) for k in ("all", "TT", "TD", "DT")}
            for p, fr, to in _bounds(lab, period):
                kind = "TT" if "D" not in (fr, to) else ("TD" if to == "D" else "DT")
                lo, hi = np.searchsorted(tf, p - win), np.searchsorted(tf, p + win + 1)
                c = [tb[j][0] for j in range(lo, hi) if tb[j][1] == fr and tb[j][2] == to]
                st = offs.setdefault(kind, [0, []])
                st[0] += 1
                if not c:
                    continue
                q = min(c, key=lambda v: abs(v - p))
                st[1].append((p - q) / 25.0)
                for k in ("all", kind):
                    if q < p:
                        outs[k][q:p] = to
                    elif q > p:
                        outs[k][p:q] = fr
            for k, x in outs.items():
                o[k] = f1(x)
            res[name] = o
        keys = list(next(iter(res.values())))
        mean = {k: float(np.mean([res[n][k] for n in res])) for k in keys}
        print("%s s%d score %.4f | boundary-date oracle (within %.1f s): " % (b[0], b[1], mean["score"], win / 25.0) + ", ".join("%s %+.4f" % (k, mean[k] - mean["score"]) for k in keys[1:]), flush=True)
        for kind, (n, v) in offs.items():
            v = np.array(v)
            print("   %s: predicted boundaries %d, with a true partner %.2f | ours - true (s): median %+.2f, mean %+.2f, mean abs %.2f, within 0.5 s %.2f, p10 %+.2f p90 %+.2f" % (
                kind, n, len(v) / max(n, 1), np.median(v), v.mean(), np.abs(v).mean(), (np.abs(v) <= 0.5).mean(), np.percentile(v, 10), np.percentile(v, 90)), flush=True)
        del data


BUNDLES = {"bd": dict(w_bd=1.0), "rev": dict(w_rev=2.0), "still": dict(ed_still=1.0),
           "bd+rev": dict(w_bd=1.0, w_rev=2.0), "bd+still": dict(w_bd=1.0, ed_still=1.0),
           "bd+rev+still": dict(w_bd=1.0, w_rev=2.0, ed_still=1.0), "rev+still": dict(w_rev=2.0, ed_still=1.0),
           "bd+rev1": dict(w_bd=1.0, w_rev=1.0), "bd+rev3": dict(w_bd=1.0, w_rev=3.0), "bd0.5+rev": dict(w_bd=0.5, w_rev=2.0),
           "bd1.5+rev": dict(w_bd=1.5, w_rev=2.0)}


def cmd_bundle(opt):
    """T141: the opt-in flags that gain on real1 / orig and are neutral on real2 (w_bd T137, w_rev T89, ed_still
    T120), alone and together, on top of the default with leave-one-match-out w_bt and w_bd evidence.
    excl=1: the w_bd model for a bench is fitted without that bench."""
    from possess import balldead
    pool = R6.benches(opt.pop("pool", R6.POOL3))
    test = R6.benches(opt.pop("test", R6.POOL3))
    names = str(opt.pop("cfg", "bd,rev,bd+rev,bd+rev+still")).split(",")
    excl = int(opt.pop("excl", 0))
    tag = opt.pop("tag", "T141_bundle")
    bt_models = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]
    tr, keep = {}, {}
    for b in pool:
        data = R6._src_rows(b)
        for name, d in data.items():
            k = d["vis"] & (d["period"] > 0)
            tr[b, name] = (d["BX"][k][::3], (d["truth"]["ctl2"][k][::3] == "D").astype(float))
        if b in test:
            keep[b] = data
        del data
    out = {}
    for b in test:
        data = keep.pop(b) if b in keep else R6._src_rows(b)
        labs, tv = {}, {}
        for name, d in data.items():
            m = R6._fit_bd([v for (bb, mm), v in tr.items() if mm != name and not (excl and bb[0] == b[0])])
            tv[name] = d["truth"]
            f0 = dict(d["feats"])
            f0["bt"] = np.where(d["vis"], np.tanh(R6.logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            f0["bd"] = balldead.evidence(d["BX"], d["vis"], m)
            labs["bt", name] = a4_short.decode_one(dict(f0), d["period"], dict(w_bt=0.5))
            for c in names:
                labs[c, name] = a4_short.decode_one(dict(f0), d["period"], dict(BUNDLES[c], w_bt=0.5))
        out["%s:%d" % b] = R.score_table(labs, tv, ["bt"] + names, list(tune_shape.TUNE), "%s s%d  T141 bundles on top of w_bt (leave-one-match-out, excl %d)" % (b[0], b[1], excl))
        del data
    with open(os.path.join(OUT, "a8_r7_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "cfg": {c: BUNDLES[c] for c in names}, "excl": excl, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_prov(opt):
    """Provided match INPUT only (no truth): what the T141 bundles change in our own output (balldead.MODEL)."""
    R.CONFIGS["BD1"], R.CONFIGS["REV"], R.CONFIGS["BDREV"], R.CONFIGS["BDREVST"] = BUNDLES["bd"], BUNDLES["rev"], BUNDLES["bd+rev"], BUNDLES["bd+rev+still"]
    R.cmd_transfer({"cfg": "BD1,REV,BDREV,BDREVST", "tag": "T141"})


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]](a4_short.kv(sys.argv[2:]))
