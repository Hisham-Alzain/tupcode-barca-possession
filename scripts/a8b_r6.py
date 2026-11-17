"""Envelope L8-autonomous-fable, round 6: follow-ups of T125 (learned ball-aware team evidence).

Every model here is fitted strictly leave-one-match-out on the external tune matches (the model that
scores a match never saw that match on ANY bench or seed) and reaches the decoder as feats["bt"]
(the w_bt hook) or as a new opt-in array; ballteam.MODEL and the w_bt code path are not touched.

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r6.py nl kinds=lin,mlp,gl test=real2:0,real1:0,orig:0
    uv run python scripts/ramlock.py -- uv run python scripts/a8b_r6.py scan test=real2:0,real1:0 cfgs=wt0.5,mh3
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a8b_ml as ML  # noqa: E402
import a8b_run as R  # noqa: E402
import tune_shape  # noqa: E402
from possess import ballteam, shape  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results")
POOL3 = "real2:0,real1:0,orig:0"
POOL6 = "real2:0,real2:1,real2:2,real2:3,real1:0,orig:0"
ALL6 = POOL6


def benches(s):
    return [ML._bs(x) for x in str(s).split(",") if x]


def even_feats(feats, period):
    """Team-symmetric context (unchanged when the teams are swapped): how contested / fast / fresh the
    ball is. Gates for the nonlinear models."""
    bx, by = np.asarray(feats["ball_x"], float), np.asarray(feats["ball_y"], float)
    dh = np.nan_to_num(np.asarray(feats["d_home"], float), nan=99.0)
    da = np.nan_to_num(np.asarray(feats["d_away"], float), nan=99.0)
    vis = np.asarray(feats["visible"], bool)
    n = len(bx)
    vx, vy = ballteam.ball_vel(bx, by)
    with np.errstate(invalid="ignore"):
        sp = np.nan_to_num(np.hypot(vx, vy))
    imp = ballteam.impulses(bx, by, 4.0)
    im = np.zeros(n, bool)
    im[imp] = True
    dmin, dmax = np.minimum(dh, da), np.maximum(dh, da)
    cols = [np.minimum(dmin, 10.0) / 10.0, np.minimum(dmax, 10.0) / 10.0, np.minimum(sp, 25.0) / 25.0,
            np.exp(-ballteam.since(im) / 50.0), ((dh >= 1.5) & (da >= 1.5)).astype(float),
            ((dh < 1.5) & (da < 1.5)).astype(float), (sp > 8.0).astype(float),
            np.exp(-ballteam.since(~vis) / 50.0), np.exp(-ballteam.since(dmin < 1.0) / 50.0)]
    return np.stack(cols, axis=1)


NEWF = ("couple", "approach", "ray", "goalside", "ray_fut", "dec", "dec_m25", "dec_p25", "dec_m50", "dec_p50")
DEC5 = "dec+dec_m25+dec_p25+dec_m50+dec_p50"          # second stage: the default decoder's own label now and 1 / 2 s before / after


def new_feats(feats, period):
    """T134: new home-minus-away feature families for the T125 model, all free of player identity (the
    provided match has none): couple = the nearest player of the team moves as the ball does; approach =
    the team closes in on the ball; ray = a player of the team stands on the ball's line of travel (the
    intended receiver); goalside = players between the ball and their own goal; ray_fut = ray seen 0.5 s later."""
    bx, by = np.asarray(feats["ball_x"], float), np.asarray(feats["ball_y"], float)
    dh = np.nan_to_num(np.asarray(feats["d_home"], float), nan=99.0)
    da = np.nan_to_num(np.asarray(feats["d_away"], float), nan=99.0)
    pos = np.asarray(feats["pos"], float)
    n, k = len(bx), 5
    vx, vy = ballteam.ball_vel(bx, by)
    with np.errstate(invalid="ignore"):
        sp = np.hypot(vx, vy)
    ad = shape.attack_dir(pos, np.asarray(period)).astype(float)
    couple, ray, gs = [], [], []
    for t in (0, 1):
        P = pos[:, t]                                            # [n, players, 2]
        with np.errstate(invalid="ignore"):
            rx, ry = P[:, :, 0] - bx[:, None], P[:, :, 1] - by[:, None]
            dist = np.where(np.isnan(rx) | np.isnan(ry), 1e9, np.hypot(rx, ry))
        i0 = dist.argmin(axis=1)
        px, py = P[np.arange(n), i0, 0], P[np.arange(n), i0, 1]
        qx, qy = np.full(n, np.nan), np.full(n, np.nan)          # the same player k frames earlier = the nearest team-mate position then
        with np.errstate(invalid="ignore"):
            d2 = np.hypot(P[:-k, :, 0] - px[k:, None], P[:-k, :, 1] - py[k:, None])
        d2 = np.where(np.isnan(d2), 1e9, d2)
        j = d2.argmin(axis=1)
        qx[k:], qy[k:] = P[np.arange(n - k), j, 0], P[np.arange(n - k), j, 1]
        pvx, pvy = (px - qx) * 25.0 / k, (py - qy) * 25.0 / k
        dmin = dist.min(axis=1)
        with np.errstate(invalid="ignore"):
            c = np.exp(-np.hypot(vx - pvx, vy - pvy) / 3.0) * (dmin < 3.0)
        couple.append(np.nan_to_num(c))
        with np.errstate(invalid="ignore", divide="ignore"):
            along = (rx * vx[:, None] + ry * vy[:, None]) / sp[:, None]
            perp = np.sqrt(np.maximum(0.0, rx ** 2 + ry ** 2 - along ** 2))
            sc = np.where((along > 1.0) & (along < 40.0), np.exp(-np.arctan2(perp, along) / 0.15), 0.0)
        ray.append(np.nan_to_num(np.nanmax(np.nan_to_num(sc), axis=1)) * np.nan_to_num((sp > 4.0).astype(float)))
        with np.errstate(invalid="ignore"):
            behind = ((P[:, :, 0] - bx[:, None]) * ad[:, None] * (1 if t == 0 else -1) < 0).sum(axis=1).astype(float)
        gs.append(behind)
    ap = []
    for d in (dh, da):
        prev = np.concatenate([np.full(12, np.nan), d[:-12]])
        with np.errstate(invalid="ignore"):
            ap.append(np.nan_to_num(np.tanh((prev - d) * 25.0 / 12.0 / 4.0) * ((d < 10.0) & (prev < 30.0))))
    vis = ~np.isnan(bx)
    r = ray[0] - ray[1]
    cols = [couple[0] - couple[1], ap[0] - ap[1], r, np.where(vis, (gs[0] - gs[1]) / 10.0, 0.0), np.concatenate([r[12:], np.zeros(12)])]
    return np.stack([np.nan_to_num(c) for c in cols], axis=1)


EVEN = ("dmin", "dmax", "speed", "since_imp", "free", "duel", "fast", "since_hidden", "since_touch")


def load(variant, seed, keep_feats=True):
    d = ML.load_src(variant, seed, decode_default=True)
    for name, x in d.items():
        x["E"] = even_feats(x["feats"], x["period"]).astype(np.float32)
        dv = np.where(x["dec"] == "H", 1.0, np.where(x["dec"] == "A", -1.0, 0.0))
        sh = [dv] + [_shift(dv, k) for k in (-25, 25, -50, 50)]
        x["N"] = np.concatenate([new_feats(x["feats"], x["period"]), np.stack(sh, axis=1)], axis=1).astype(np.float32)
        if not keep_feats:
            x["feats"] = None
    return d


def spell_len(t):
    """Length (frames) of the run of equal labels each frame sits in."""
    e = np.flatnonzero(np.concatenate([[True], t[1:] != t[:-1], [True]]))
    return np.repeat(np.diff(e), np.diff(e))


def train_xy(pool, name, target="ctl2", with_len=False):
    """Rows of every pool bench except match `name` (strict leave-one-match-out): visible frames where
    the truth names a team, every 3rd. with_len: also the length of the true spell of each row."""
    Xs, Es, ys, Ls, Ns = [], [], [], [], []
    for d in pool.values():
        for m, x in d.items():
            if m == name:
                continue
            t = x["truth"][target]
            k = x["vis"] & np.isin(t, ["H", "A"]) & (x["period"] > 0)
            Xs.append(x["X"][k][::3].astype(float))
            Es.append(x["E"][k][::3].astype(float))
            ys.append((t[k][::3] == "H").astype(float))
            Ls.append(spell_len(t)[k][::3].astype(float))
            Ns.append(x["N"][k][::3].astype(float))
    if with_len:
        return np.concatenate(Xs), np.concatenate(Es), np.concatenate(ys), np.concatenate(Ls), np.concatenate(Ns)
    return np.concatenate(Xs), np.concatenate(Es), np.concatenate(ys)


# ---- models: each returns a json-able dict; logit_of(model, X, E) evaluates it -------------------------------

def fit_lin(X, E, y, l2=10.0):
    m = shape.fit_logistic(X, y, l2=l2, intercept=False)
    return {"kind": "lin", "sd": m["sd"].tolist(), "w": m["w"].tolist()}


def fit_wlin(X, E, y, sw, l2=10.0, iters=30):
    """fit_lin with a weight per row (shape.fit_logistic's Newton steps, weighted; weights scaled to mean 1)."""
    sw = sw / sw.mean()
    sd = np.sqrt((X ** 2).mean(axis=0)) + 1e-9
    Z = X / sd
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (sw * (p - y)) + l2 * w
        Hs = (Z * (sw * p * (1 - p))[:, None]).T @ Z + np.diag(np.full(len(w), l2 + 1e-9))
        step = np.linalg.solve(Hs, g)
        w = w - step
        if np.abs(step).max() < 1e-7:
            break
    return {"kind": "lin", "sd": sd.tolist(), "w": w.tolist()}


def _gl_expand(X, E):
    """Gated linear: every odd feature alone and times every even gate (odd x even = odd)."""
    return np.concatenate([X] + [X * E[:, j:j + 1] for j in range(E.shape[1])], axis=1)


def fit_gl(X, E, y, l2=30.0):
    m = shape.fit_logistic(_gl_expand(X, E), y, l2=l2, intercept=False)
    return {"kind": "gl", "sd": m["sd"].tolist(), "w": m["w"].tolist()}


def _mlp_fwd(p, Zo, Ze, nh):
    nin = Zo.shape[1] + Ze.shape[1]
    W1 = p[:nin * nh].reshape(nin, nh)
    b1 = p[nin * nh:nin * nh + nh]
    w2 = p[nin * nh + nh:]
    I1, I2 = np.concatenate([Zo, Ze], axis=1), np.concatenate([-Zo, Ze], axis=1)
    h1, h2 = np.tanh(I1 @ W1 + b1), np.tanh(I2 @ W1 + b1)
    return (h1 - h2) @ w2, (I1, I2, h1, h2, W1, w2)


def fit_mlp(X, E, y, nh=12, l2=1e-4, seed=0, iters=300):
    """Antisymmetric one-hidden-layer network g(x, e) - g(-x, e) (swapping the teams flips the sign),
    L-BFGS on the logistic loss, fixed seed."""
    sd = np.sqrt((X ** 2).mean(axis=0)) + 1e-9
    emu, esd = E.mean(axis=0), E.std(axis=0) + 1e-9
    Zo, Ze = X / sd, (E - emu) / esd
    nin = Zo.shape[1] + Ze.shape[1]
    rng = np.random.default_rng(seed)
    p0 = np.concatenate([rng.normal(0, 0.3 / np.sqrt(nin), nin * nh), np.zeros(nh), rng.normal(0, 0.3, nh)])
    N = len(y)

    def loss(p):
        z, (I1, I2, h1, h2, W1, w2) = _mlp_fwd(p, Zo, Ze, nh)
        z = np.clip(z, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        L = -np.mean(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12)) + l2 * (p ** 2).sum()
        dz = (pr - y) / N
        gw2 = (h1 - h2).T @ dz
        dA1 = dz[:, None] * w2 * (1 - h1 ** 2)
        dA2 = -dz[:, None] * w2 * (1 - h2 ** 2)
        g = np.concatenate([(I1.T @ dA1 + I2.T @ dA2).ravel(), (dA1 + dA2).sum(axis=0), gw2]) + 2 * l2 * p
        return L, g

    r = minimize(loss, p0, jac=True, method="L-BFGS-B", options={"maxiter": iters})
    return {"kind": "mlp", "nh": nh, "sd": sd.tolist(), "emu": emu.tolist(), "esd": esd.tolist(), "p": r.x.tolist(), "loss": float(r.fun)}


def logit_of(m, X, E, N=None):
    X, E = np.asarray(X, float), np.asarray(E, float)
    if m["kind"] == "linx":
        return (np.concatenate([X, np.asarray(N, float)[:, m["nc"]]], axis=1) / np.array(m["sd"])) @ np.array(m["w"])
    if m["kind"] == "lin":
        return (X / np.array(m["sd"])) @ np.array(m["w"])
    if m["kind"] == "gl":
        return (_gl_expand(X, E) / np.array(m["sd"])) @ np.array(m["w"])
    if m["kind"] == "mlp":
        return _mlp_fwd(np.array(m["p"]), X / np.array(m["sd"]), (E - np.array(m["emu"])) / np.array(m["esd"]), m["nh"])[0]
    raise ValueError(m["kind"])


FIT = {"lin": fit_lin, "gl": fit_gl, "mlp": fit_mlp}


def lomo_models(pool_spec, kinds, cache=None, **kw):
    """{kind: {match: model}} fitted leave-one-match-out on the pool; cached as json."""
    path = os.path.join(OUT, "a8_r6_models_%s.json" % cache) if cache else None
    have = json.load(open(path)) if path and os.path.exists(path) else {}
    todo = [k for k in kinds if k not in have]
    if todo:
        pool = {b: load(*b, keep_feats=False) for b in benches(pool_spec)}
        for name in tune_shape.TUNE:
            X, E, y, L, N = train_xy(pool, name, with_len=True)
            for k in todo:
                kind, _, arg = k.partition("-")
                extra = dict(kw)
                if kind == "linx":       # T134: the 21 features plus new families (arg = names joined by +, default all)
                    nc = [NEWF.index(a) for a in arg.split("+")] if arg else list(range(len(NEWF)))
                    m = fit_lin(np.concatenate([X, N[:, nc]], axis=1), E, y)
                    have.setdefault(k, {})[name] = dict(m, kind="linx", nc=nc)
                    continue
                if kind == "linw":       # spell-balanced: rows of a true spell longer than `arg` s weigh arg / length
                    have.setdefault(k, {})[name] = fit_wlin(X, E, y, np.minimum(1.0, float(arg) * 25.0 / L))
                    continue
                if kind == "linwp":      # weight = length ** -arg (arg 1 = every true spell weighs the same)
                    have.setdefault(k, {})[name] = fit_wlin(X, E, y, L ** -float(arg))
                    continue
                if arg:
                    extra["nh" if kind == "mlp" else "l2"] = float(arg) if kind != "mlp" else int(arg)
                have.setdefault(k, {})[name] = FIT[kind](X, E, y, **extra)
                print("   fitted %s without %s (n %d)" % (k, name, len(y)), flush=True)
        del pool
        if path:
            with open(path, "w") as fh:
                json.dump(have, fh)
    return have


def frame_stats(z, d):
    t, p = d["truth"]["ctl2"], d["dec"]
    k = d["vis"] & np.isin(t, ["H", "A"]) & np.isin(p, ["H", "A"]) & (d["period"] > 0)
    th, ph, zh = t[k] == "H", p[k] == "H", z[k] > 0
    wrong = th != ph
    pr = 1.0 / (1.0 + np.exp(-np.clip(z[k], -30, 30)))
    ll = -np.mean(np.where(th, np.log(pr + 1e-12), np.log(1 - pr + 1e-12)))
    return np.array([k.sum(), (zh == th).sum(), (ph == th).sum(), (zh[wrong] == th[wrong]).sum(), wrong.sum(), ll * k.sum()])


def cmd_nl(opt):
    """Nonlinear versions of the T125 model against the linear one: per-frame numbers, then the decoder."""
    pool = opt.pop("pool", POOL3)
    kinds = str(opt.pop("kinds", "lin,gl,mlp")).split(",")
    ws = [float(x) for x in str(opt.pop("w", "0.5")).split(",")]
    scale = float(opt.pop("scale", 2.0))
    tag = opt.pop("tag", "nl")
    cache = opt.pop("cache", "pool3" if pool == POOL3 else ("pool6" if pool == POOL6 else None))
    models = lomo_models(pool, kinds, cache)
    out = {}
    for b in benches(opt.pop("test", POOL3)):
        data = load(*b)
        labs, tv = {}, {}
        cfgs = ["default"] + ["%s@%g" % (k, w) for k in kinds for w in ws]
        st = {k: np.zeros(6) for k in kinds}
        for name, d in data.items():
            tv[name] = d["truth"]
            labs["default", name] = d["dec"]
            for k in kinds:
                z = logit_of(models[k][name], d["X"], d["E"], d["N"])
                st[k] += frame_stats(z, d)
                if k.startswith("linx") and name == tune_shape.TUNE[0] and b == benches(pool)[0]:
                    mm = models[k][name]
                    print("   %s weights of the new features (without %s): " % (k, name) + ", ".join("%s %+.2f" % (NEWF[c], w) for c, w in zip(mm["nc"], mm["w"][21:])), flush=True)
                f2 = dict(d["feats"], bt=np.where(d["vis"], np.tanh(z / scale), 0.0))
                for w in ws:
                    labs["%s@%g" % (k, w), name] = a4_short.decode_one(f2, d["period"], dict(w_bt=w))
        for k in kinds:
            s = st[k]
            print("%s s%d %-7s per frame: model right %.4f | decoder %.4f | right where the decoder is wrong %.3f | log-loss %.4f" % (
                b[0], b[1], k, s[1] / s[0], s[2] / s[0], s[3] / s[4], s[5] / s[0]), flush=True)
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  round 6 models, leave-one-match-out (pool %s)" % (b[0], b[1], pool))
        del data
    with open(os.path.join(OUT, "a8_r6_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "kinds": kinds, "w": ws, "scale": scale, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


CFG = {
    "wt0": dict(w_touch=0.0), "wt0.4": dict(w_touch=0.4), "wt1": dict(w_touch=1.0),
    "mh2": dict(max_hold_s=2.0), "mh6": dict(max_hold_s=6.0),
    "wr0": dict(w_rule=0.0), "wr0.6": dict(w_rule=0.6),
    "cs3": dict(c_switch=3.0), "cs5": dict(c_switch=5.0),
    "wf0.4": dict(w_fly=0.4), "wf1": dict(w_fly=1.0),
}


def cmd_scan(opt):
    """Decoder parameters re-scanned WITH w_bt on (baseline = default + w_bt; pooled leave-one-match-out model)."""
    pool = opt.pop("pool", POOL3)
    kind = opt.pop("kind", "lin")
    w = float(opt.pop("w", 0.5))
    names = str(opt.pop("cfgs", "wt0.5,wt1.5,mh3,mh6")).split(",")
    tag = opt.pop("tag", "scan")
    models = lomo_models(pool, [kind], opt.pop("cache", "pool3" if pool == POOL3 else ("pool6" if pool == POOL6 else None)))
    from possess import pipeline
    print("defaults: " + ", ".join("%s=%s" % (k, pipeline.DEC.get(k)) for k in ("w_touch", "max_hold_s", "w_rule", "c_switch", "w_shape")))
    out = {}
    for b in benches(opt.pop("test", "real2:0,real1:0")):
        data = load(*b)
        labs, tv = {}, {}
        cfgs = ["bt"] + names
        for name, d in data.items():
            tv[name] = d["truth"]
            z = logit_of(models[kind][name], d["X"], d["E"])
            f2 = dict(d["feats"], bt=np.where(d["vis"], np.tanh(z / 2.0), 0.0))
            labs["bt", name] = a4_short.decode_one(f2, d["period"], dict(w_bt=w))
            for c in names:
                labs[c, name] = a4_short.decode_one(f2, d["period"], dict(CFG[c], w_bt=w))
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  re-scan on top of w_bt=%g (%s, pool %s); baseline = bt" % (b[0], b[1], w, kind, pool))
        del data
    with open(os.path.join(OUT, "a8_r6_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "kind": kind, "w": w, "cfgs": names, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)

# ---- T130: learned per-frame switch cost ------------------------------------------------------------------------

SWF = ("dv", "dv_max", "since_imp", "until_imp", "duel", "dmin", "speed", "g_change", "near_change25", "near_change50",
       "z_change", "z_flip", "hidden_near")


def _shift(x, k):
    """x[t + k] (0 beyond the ends)."""
    out = np.zeros_like(x)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def switch_feats(feats, X, z):
    """Team-symmetric features of "a turnover happens around this frame" (offline: both sides of the frame).
    X = ballteam.features, z = the leave-one-match-out T125 evidence (tanh, 0 where hidden)."""
    bx, by = np.asarray(feats["ball_x"], float), np.asarray(feats["ball_y"], float)
    dh = np.nan_to_num(np.asarray(feats["d_home"], float), nan=99.0)
    da = np.nan_to_num(np.asarray(feats["d_away"], float), nan=99.0)
    vis = np.asarray(feats["visible"], bool)
    n, h = len(bx), 6
    vx, vy = ballteam.ball_vel(bx, by)
    dv = np.zeros(n)
    with np.errstate(invalid="ignore"):
        dv[h:n - h] = np.nan_to_num(np.hypot(vx[2 * h:] - vx[:-2 * h], vy[2 * h:] - vy[:-2 * h]))
        sp = np.nan_to_num(np.hypot(vx, vy))
    dv = np.minimum(dv, 20.0) / 20.0
    imp = ballteam.impulses(bx, by, 4.0)
    im = np.zeros(n, bool)
    im[imp] = True
    X = np.asarray(X, float)
    g, p25, p50 = X[:, 0], X[:, 3], X[:, 4]
    cols = [dv, _shift(-ballteam.past_min(-dv, 13), 6), np.exp(-ballteam.since(im) / 12.0), np.exp(-ballteam.since(im[::-1])[::-1] / 12.0),
            ((dh < 1.5) & (da < 1.5)).astype(float), np.minimum(np.minimum(dh, da), 10.0) / 10.0, np.minimum(sp, 25.0) / 25.0,
            np.abs(_shift(g, 12) - _shift(g, -12)), np.abs(_shift(p25, 24) - p25), np.abs(_shift(p50, 49) - p50),
            np.abs(_shift(z, 12) - _shift(z, -12)) / 2.0, np.maximum(0.0, -_shift(z, 12) * _shift(z, -12)),
            1.0 - np.exp(-np.minimum(ballteam.since(~vis), ballteam.since(~vis[::-1])[::-1]) / 12.0)]
    return np.stack(cols, axis=1)


def turnover_target(t, win=12):
    """1 within `win` frames of a direct change of the truth's team (H <-> A, no stoppage between)."""
    ch = np.zeros(len(t), bool)
    ch[1:] = np.isin(t[1:], ["H", "A"]) & np.isin(t[:-1], ["H", "A"]) & (t[1:] != t[:-1])
    return ballteam.since(ch) <= win, ballteam.since(ch[::-1])[::-1] < win, ch


def cmd_sw(opt):
    """T130: learned per-frame switch cost (w_sw), on the default and on top of w_bt. Leave-one-match-out.
    REJECTED (flat), so the decoder hook is not in src: apply scripts/a8b_r6_sw.patch (git apply) to run this."""
    import inspect
    from possess import decode
    if "w_sw" not in inspect.signature(decode.viterbi).parameters:
        sys.exit("decode.viterbi has no w_sw: git apply scripts/a8b_r6_sw.patch first (T130 was rejected; the hook is not in src)")
    pool = benches(opt.pop("pool", POOL3))
    wsw = [float(x) for x in str(opt.pop("wsw", "0.5,1.0")).split(",")]
    wbt = float(opt.pop("w", 0.5))
    l2 = float(opt.pop("l2", 10.0))
    win = int(opt.pop("win", 12))
    lo, hi = float(opt.pop("lo", 1.0)), float(opt.pop("hi", 8.0))
    tag = opt.pop("tag", "T130_sw")
    test = benches(opt.pop("test", POOL3))
    bt_models = lomo_models(",".join("%s:%d" % b for b in pool), ["lin"], opt.pop("cache", "pool3"))["lin"]
    tr = {}
    for b in pool:                       # rows for the switch model: visible frames where the truth names a team
        d = load(*b, keep_feats=True)
        for name, x in d.items():
            z = np.where(x["vis"], np.tanh(logit_of(bt_models[name], x["X"], x["E"]) / 2.0), 0.0)
            S = switch_feats(x["feats"], x["X"], z)
            t = x["truth"]["ctl2"]
            a, c, _ = turnover_target(t, win)
            k = x["vis"] & np.isin(t, ["H", "A"]) & (x["period"] > 0)
            tr[b, name] = (S[k][::3], (a | c)[k][::3].astype(float))
        del d
    sm = {}
    for name in tune_shape.TUNE:
        Xs = np.concatenate([v[0] for (b, m), v in tr.items() if m != name])
        ys = np.concatenate([v[1] for (b, m), v in tr.items() if m != name])
        m = shape.fit_logistic(Xs, ys, l2=l2, intercept=True)
        m["base"] = float(np.log(ys.mean() / (1 - ys.mean())))
        sm[name] = m
        if name == tune_shape.TUNE[0]:
            print("   switch model without %s: base rate %.3f; weights: " % (name, ys.mean()) + ", ".join("%s %+.2f" % (k, w) for k, w in zip(SWF, m["w"])), flush=True)
    del tr
    out = {}
    for b in test:
        data = load(*b)
        labs, tv = {}, {}
        cfgs = ["default"] + ["sw%g" % w for w in wsw] + ["bt"] + ["bt+sw%g" % w for w in wsw]
        st = np.zeros(4)
        for name, d in data.items():
            tv[name] = d["truth"]
            labs["default", name] = d["dec"]
            z = np.where(d["vis"], np.tanh(logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            m = sm[name]
            l = ((switch_feats(d["feats"], d["X"], z) - m["mu"]) / m["sd"]) @ m["w"] + float(m["b"])
            s = np.where(d["vis"], m["base"] - l, 0.0)
            a, c, ch = turnover_target(d["truth"]["ctl2"], win)
            k = d["vis"] & (d["period"] > 0) & np.isin(d["truth"]["ctl2"], ["H", "A"])
            y = (a | c)[k]
            st += [l[k][y].mean(), l[k][~y].mean(), (l[k][y] > m["base"]).mean(), (l[k][~y] > m["base"]).mean()]
            f2 = dict(d["feats"], sw=s)
            f3 = dict(f2, bt=z)
            labs["bt", name] = a4_short.decode_one(f3, d["period"], dict(w_bt=wbt))
            for w in wsw:
                labs["sw%g" % w, name] = a4_short.decode_one(f2, d["period"], dict(w_sw=w, sw_lo=lo, sw_hi=hi))
                labs["bt+sw%g" % w, name] = a4_short.decode_one(f3, d["period"], dict(w_bt=wbt, w_sw=w, sw_lo=lo, sw_hi=hi))
        st /= len(data)
        print("%s s%d switch log-odds: mean at turnover frames %+.2f, elsewhere %+.2f; above the base rate: %.3f of turnover frames, %.3f of the others" % (
            b[0], b[1], st[0], st[1], st[2], st[3]), flush=True)
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T130 learned switch cost (clip %g..%g, win %d); d vs default" % (b[0], b[1], lo, hi, win))
        del data
    with open(os.path.join(OUT, "a8_r6_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "wsw": wsw, "w_bt": wbt, "win": win, "clip": [lo, hi], "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


# ---- T132: learned accept / reject of the relabels a liberal decode proposes, target = the metric's own change ----

pm = R.pm


class Fast:
    """vendor possession_metrics spell F1 (mean over IoU 0.3-0.7) with the overlap list built by search
    instead of the double loop; same greedy matching. check() compares with pm.evaluate."""

    def __init__(self, truth):
        self.gt = pm.label_codes(truth)
        self.runs = pm.valid_runs(self.gt)
        self.gs = pm.possession_spells(self.gt, self.runs)

    def __call__(self, lab):
        ps = pm.possession_spells(pm.fill_abstentions(pm.label_codes(lab), self.runs), self.runs)
        if not ps or not self.gs:
            return 0.0
        st, en, cd = np.array([x[0] for x in ps]), np.array([x[1] for x in ps]), np.array([x[2] for x in ps])
        ov = []
        for i, (a, b, k) in enumerate(self.gs):
            for j in range(int(np.searchsorted(en, a, side="right")), int(np.searchsorted(st, b, side="left"))):
                if cd[j] == k:
                    ov.append(((min(b, en[j]) - max(a, st[j])) / float(max(b, en[j]) - min(a, st[j])), i, j))
        ov.sort(key=lambda t: (-t[0], t[1], t[2]))
        f = []
        for thr in pm.SPELL_IOUS:
            tp = int(pm.matched_gt_flags(ov, len(self.gs), thr).sum())
            f.append(2.0 * tp / (len(self.gs) + len(ps)))
        return float(np.mean(f))

    def check(self, lab):
        a, b = self(lab), pm.evaluate(self.gt, pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]
        assert abs(a - b) < 1e-12, (a, b)
        return a


PROP = {"cs2": dict(c_switch=2.0), "cs1": dict(c_switch=1.0), "cs3": dict(c_switch=3.0),
        "cs2bt1": dict(c_switch=2.0, w_bt=1.0), "bt1": dict(w_bt=1.0), "cs6": dict(c_switch=6.0)}


def candidates(base, alt, period, team_only=True):
    """Maximal runs where alt differs from base (constant base label, alt label and period): (start, stop, new)."""
    diff = (alt != base) & (period > 0)
    if team_only:
        diff &= np.isin(alt, ["H", "A"]) & np.isin(base, ["H", "A"])
    n = len(base)
    brk = np.ones(n + 1, bool)
    brk[1:n] = (diff[1:] != diff[:-1]) | (alt[1:] != alt[:-1]) | (base[1:] != base[:-1]) | (period[1:] != period[:-1])
    e = np.flatnonzero(brk)
    return [(int(a), int(b), str(alt[a])) for a, b in zip(e[:-1], e[1:]) if diff[a]]


def cand_feats(c, base, period, X, E, z, vis):
    """Features of one proposed relabel, oriented to the proposed team (team-symmetric)."""
    a, b, k = c
    sg = 1.0 if k == "H" else -1.0
    n = len(base)
    dur = (b - a) / 25.0
    zz = sg * z[a:b]
    pl = base[a - 1] if a > 0 and period[a - 1] == period[a] else "X"
    nl = base[b] if b < n and period[b] == period[a] else "X"
    i = a
    while i > 0 and base[i - 1] == base[a] and period[i - 1] == period[a]:
        i -= 1
    j = b
    while j < n and base[j] == base[a] and period[j] == period[a]:
        j += 1
    left, right = (a - i) / 25.0, (j - b) / 25.0
    f = [np.log(dur), float(dur < 2.0), float(dur < 1.0), zz.mean(), zz.min(), zz.max(), (zz > 0).mean(), vis[a:b].mean(),
         float(pl == k), float(nl == k), float(pl == "D" or pl == "X"), float(nl == "D" or nl == "X"),
         float(left > 0 and right > 0), np.log((j - i) / 25.0), np.log1p(left), np.log1p(right),
         float(0 < left < 2.0), float(0 < right < 2.0)]
    return np.concatenate([f, sg * X[a:b].mean(axis=0), E[a:b].mean(axis=0)])


def apply_cands(base, cands, keep):
    lab = base.copy()
    for (a, b, k), ok in zip(cands, keep):
        if ok:
            lab[a:b] = k
    return lab


def build_cands(b, bt_models, prop, base_kind, wbt, team_only):
    """Per match of one bench: base and proposer labels, candidates, their features and the metric's change
    (ctl2 and old truth) when each one alone is applied to the base."""
    out = {}
    data = load(*b)
    for name, d in data.items():
        z = np.where(d["vis"], np.tanh(logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
        f2 = dict(d["feats"], bt=z)
        bkw = dict(w_bt=wbt) if base_kind == "bt" else {}
        base = a4_short.decode_one(f2, d["period"], bkw) if bkw else d["dec"]
        if prop.startswith("del"):       # del<max seconds>: every base team spell up to that length goes to its neighbour team
            alt = base.copy()
            for a, e, k in R._spells(base, d["period"]):
                if e - a > float(prop[3:]) * 25:
                    continue
                pl = base[a - 1] if a > 0 and d["period"][a - 1] == d["period"][a] else "X"
                nl = base[e] if e < len(base) and d["period"][e] == d["period"][a] else "X"
                tgt = pl if pl in "HA" and (nl == pl or nl in "DX") else (nl if nl in "HA" and pl in "DX" else None)
                if tgt:
                    alt[a:e] = tgt
            cs = candidates(base, alt, d["period"], True)
        elif prop.startswith("z"):       # z<min seconds>: runs where the smoothed model evidence opposes the base team
            zs = np.convolve(z, np.ones(13) / 13.0, mode="same")
            opp = d["vis"] & (((base == "H") & (zs < -0.2)) | ((base == "A") & (zs > 0.2)))
            alt = base.copy()
            alt[opp] = np.where(base[opp] == "H", "A", "H")
            cs = [c for c in candidates(base, alt, d["period"], True) if c[1] - c[0] >= float(prop[1:]) * 25]
            alt = apply_cands(base, cs, [True] * len(cs))
        else:
            alt = a4_short.decode_one(f2, d["period"], dict(bkw, **PROP[prop]))
            cs = candidates(base, alt, d["period"], team_only)
        sc = {k: Fast(d["truth"][k]) for k in ("ctl2", "old")}
        s0 = {k: sc[k].check(base) for k in sc}
        F = np.array([cand_feats(c, base, d["period"], d["X"].astype(float), d["E"].astype(float), z, d["vis"]) for c in cs]).reshape(len(cs), -1)
        D = np.array([[sc[k](apply_cands(base, [c], [True])) - s0[k] for k in ("ctl2", "old")] for c in cs]).reshape(len(cs), 2)
        out[name] = dict(truth=d["truth"], dec=d["dec"], base=base, alt=alt, cands=cs, F=F, D=D)
        print("   %s s%d %-9s candidates %4d | better %4d, worse %4d, same %4d | sum of gains %+.4f, of losses %+.4f" % (
            b[0], b[1], name[:9], len(cs), (D[:, 0] > 0).sum(), (D[:, 0] < 0).sum(), (D[:, 0] == 0).sum(),
            D[:, 0][D[:, 0] > 0].sum(), D[:, 0][D[:, 0] < 0].sum()), flush=True)
    del data
    return out


def cmd_cand(opt):
    """T132. Leave-one-match-out: the accept model for a match is fitted on the candidates of the other matches
    (all pool benches). Rows: default, base, the proposer decode, the oracle (every candidate that helps alone),
    and the learned accept at each threshold."""
    pool_s = opt.pop("pool", POOL3)
    prop, base_kind = opt.pop("prop", "cs2"), opt.pop("base", "bt")
    wbt, l2 = float(opt.pop("w", 0.5)), float(opt.pop("l2", 10.0))
    thrs = [float(x) for x in str(opt.pop("thr", "0.5,0.6")).split(",")]
    team_only = int(opt.pop("team_only", 1))
    target = opt.pop("target", "pos")            # pos: change > 0; nonneg: change >= 0
    nf = int(opt.pop("nf", 0))                   # 0 = all features; 18 = duration / evidence / structure only
    tag = opt.pop("tag", "T132_%s_%s" % (prop, base_kind))
    test = benches(opt.pop("test", pool_s))
    bt_models = lomo_models(pool_s, ["lin"], opt.pop("cache", "pool3" if pool_s == POOL3 else "pool6"))["lin"]
    C = {b: build_cands(b, bt_models, prop, base_kind, wbt, team_only) for b in benches(pool_s)}
    for b in test:
        if b not in C:
            C[b] = build_cands(b, bt_models, prop, base_kind, wbt, team_only)
    out = {}
    for b in test:
        labs, tv = {}, {}
        cfgs = ["default", "base", "prop", "oracle"] + ["acc%g" % t for t in thrs]
        nacc = {t: [0, 0] for t in thrs}
        for name, d in C[b].items():
            Fs = np.concatenate([x["F"] for bb in benches(pool_s) for m, x in C[bb].items() if m != name and len(x["cands"])])
            Ds = np.concatenate([x["D"][:, 0] for bb in benches(pool_s) for m, x in C[bb].items() if m != name and len(x["cands"])])
            y = (Ds > 0).astype(float) if target == "pos" else (Ds >= 0).astype(float)
            if nf:
                Fs = Fs[:, :nf]
            Ft = d["F"][:, :nf] if nf and len(d["cands"]) else d["F"]
            if target == "reg":          # ridge regression of the change itself (x 1000); thr is then in thousandths
                mu, sd = Fs.mean(axis=0), Fs.std(axis=0) + 1e-9
                Z = (Fs - mu) / sd
                wr = np.linalg.solve(Z.T @ Z + l2 * np.eye(Z.shape[1]), Z.T @ (1000.0 * (Ds - Ds.mean())))
                p = ((Ft - mu) / sd) @ wr + 1000.0 * Ds.mean() if len(d["cands"]) else np.zeros(0)
            else:
                m = shape.fit_logistic(Fs, y, l2=l2, intercept=True)
                p = 1.0 / (1.0 + np.exp(-(((Ft - m["mu"]) / m["sd"]) @ m["w"] + float(m["b"])))) if len(d["cands"]) else np.zeros(0)
            tv[name] = d["truth"]
            labs["default", name], labs["base", name], labs["prop", name] = d["dec"], d["base"], d["alt"]
            labs["oracle", name] = apply_cands(d["base"], d["cands"], d["D"][:, 0] > 0)
            for t in thrs:
                labs["acc%g" % t, name] = apply_cands(d["base"], d["cands"], p > t)
                nacc[t][0] += int((p > t).sum())
                nacc[t][1] += int(((p > t) & (d["D"][:, 0] > 0)).sum())
        print("%s s%d accepted (helping alone): " % b + ", ".join("thr %g: %d (%d)" % (t, v[0], v[1]) for t, v in nacc.items()), flush=True)
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T132 learned accept of %s proposals on base %s (pool %s, l2 %g)" % (b[0], b[1], prop, base_kind, pool_s, l2))
    with open(os.path.join(OUT, "a8_r6_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool_s, "prop": prop, "base": base_kind, "w_bt": wbt, "l2": l2, "thr": thrs, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_diag(opt):
    """Where the loss sits with w_bt on (a8b_run.py diag on the labels default + w_bt, pooled leave-one-match-out)."""
    wbt = float(opt.pop("w", 0.5))
    bt_models = lomo_models(POOL3, ["lin"], "pool3")["lin"]
    for b in benches(opt.pop("test", "real2:0,real1:0")):
        data = load(*b)
        res, cats, merged = {}, {}, {}
        for name, d in data.items():
            z = np.where(d["vis"], np.tanh(logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            lab = a4_short.decode_one(dict(d["feats"], bt=z), d["period"], dict(w_bt=wbt))
            t, period, vis = d["truth"]["ctl2"], d["period"], d["vis"]
            f1 = Fast(t)
            play = period > 0
            tD, pD = (t == "D") & play, (lab == "D") & play
            o = {"score": f1(lab)}
            x = R._ffill_team(lab, pD & ~tD)
            x[tD] = "D"
            o["D"] = f1(x)
            x = np.array(lab)
            x[tD] = "D"
            o["missedD"] = f1(x)
            o["falseD"] = f1(R._ffill_team(lab, pD & ~tD))
            x = np.array(lab)
            m = play & ~pD & ~tD
            x[m] = t[m]
            o["team"] = f1(x)
            for nm, mk in (("team_vis", vis), ("team_hid", ~vis)):
                x = np.array(lab)
                x[m & mk] = t[m & mk]
                o[nm] = f1(x)
            for nm, mk in (("D_vis", vis), ("D_hid", ~vis)):        # true D / live on the frames of one kind, our team elsewhere
                x = R._ffill_team(lab, pD & ~tD & mk)
                x[tD & mk] = "D"
                o[nm] = f1(x)
            res[name] = o
            ps = R._spells(lab, period)
            for (s0, e0, k0), (cat, L, vs) in zip(R._spells(t, period), R.classify(t, lab, period, vis)):
                if cat != "merged_long":
                    continue
                ov = max(((min(e0, q) - max(s0, p0), p0, q) for p0, q, kk in ps if kk == k0 and p0 < e0 and q > s0), default=None)
                if ov is None:
                    continue
                rest = np.concatenate([t[ov[1]:max(ov[1], s0)], t[min(ov[2], e0):ov[2]]])
                rv = np.concatenate([vis[ov[1]:max(ov[1], s0)], vis[min(ov[2], e0):ov[2]]])
                nD, nO, nS = (rest == "D").sum(), ((rest != "D") & (rest != k0)).sum(), (rest == k0).sum()
                kind = "rest mostly same team" if nS >= max(nD, nO) * 4 else ("stoppage" if nD >= nO else "other team")
                mm = merged.setdefault(kind, [0, 0.0, 0.0, 0.0])
                mm[0] += 1
                mm[1] += (nD if kind == "stoppage" else nO) / 25.0
                mm[2] += float(rv[(rest == "D") if kind == "stoppage" else ((rest != "D") & (rest != k0))].mean()) if kind != "rest mostly same team" else 0.0
                mm[3] += L
            for cat, L, vs in R.classify(t, lab, period, vis):
                bb = "<3s" if L < 3 else ("3-8s" if L < 8 else ">=8s")
                c = cats.setdefault(cat, {"n": 0, "<3s": 0, "3-8s": 0, ">=8s": 0, "hid": 0})
                c["n"] += 1
                c[bb] += 1
                c["hid"] += vs < 0.5
        keys = list(next(iter(res.values())))
        mean = {k: float(np.mean([res[n][k] for n in res])) for k in keys}
        print("%s s%d default + w_bt %g: score %.4f | oracle gains: " % (b[0], b[1], wbt, mean["score"]) + ", ".join("%s %+.4f" % (k, mean[k] - mean["score"]) for k in keys[1:]), flush=True)
        tot = sum(c["n"] for c in cats.values())
        print("   true possessions %d; unmatched at IoU 0.5 by cause (n, share | <3 s, 3-8 s, >= 8 s | mostly hidden):" % tot)
        for cat, c in sorted(cats.items(), key=lambda kv: -kv[1]["n"]):
            print("   %-14s %4d %.3f | %4d %4d %4d | %4d" % (cat, c["n"], c["n"] / tot, c["<3s"], c["3-8s"], c[">=8s"], c["hid"]))
        print("   swallowed true spells, by what else the predicted spell covers (n | mean seconds of it | share of it visible | mean length of the swallowed spell):")
        for kind, mm in sorted(merged.items(), key=lambda kv: -kv[1][0]):
            print("   %-22s %4d | %5.1f s | %.2f | %4.1f s" % (kind, mm[0], mm[1] / mm[0], mm[2] / mm[0], mm[3] / mm[0]))
        del data


# ---- T137: learned ball-aware stoppage evidence on visible frames (the D-side mirror of T125) --------------------

DEADF = ("shape_dead", "speed", "slow", "still_time", "dmin", "alone3", "line_dist", "outside", "carried", "carried_slow",
         "p90", "goal_area_slow") + EVEN


def dead_feats(feats, E, N_couple_abs):
    bx, by = np.asarray(feats["ball_x"], float), np.asarray(feats["ball_y"], float)
    dh = np.nan_to_num(np.asarray(feats["d_home"], float), nan=99.0)
    da = np.nan_to_num(np.asarray(feats["d_away"], float), nan=99.0)
    n = len(bx)
    vx, vy = ballteam.ball_vel(bx, by)
    with np.errstate(invalid="ignore"):
        sp = np.nan_to_num(np.hypot(vx, vy))
        line = np.nan_to_num(np.minimum(52.5 - np.abs(bx), 34.0 - np.abs(by)), nan=10.0)
        ga = np.nan_to_num((np.abs(bx) > 46.0).astype(float))
    dmin = np.minimum(dh, da)
    sd = np.nan_to_num(np.asarray(feats["shape_dead"], float)) if feats.get("shape_dead") is not None else np.zeros(n)
    p90 = np.nan_to_num(np.asarray(feats["edge_p90"], float)) if feats.get("edge_p90") is not None else np.zeros(n)
    cols = [sd, np.minimum(sp, 25.0) / 25.0, (sp < 1.0).astype(float), 1.0 - np.exp(-ballteam.since(sp > 2.0) / 75.0),
            np.minimum(dmin, 10.0) / 10.0, (dmin > 3.0).astype(float), np.clip(line, 0.0, 10.0) / 10.0, (line < 0).astype(float),
            N_couple_abs, N_couple_abs * (sp < 3.0), np.minimum(p90, 8.0) / 8.0, ga * (sp < 1.0)]
    return np.concatenate([np.stack(cols, axis=1), np.asarray(E, float)], axis=1)


def cmd_dead(opt):
    """T137. The evidence is added to the D-vs-live log-odds of visible frames through feats["shape_dead"]
    (decode.viterbi: sdead_vis * w_sdead * shape_dead), so no src change. excl=1: the model for a bench is
    fitted without that bench."""
    from possess import pipeline
    pool = benches(opt.pop("pool", POOL3))
    ws = [float(x) for x in str(opt.pop("w", "0.5,1.0,2.0")).split(",")]
    excl, l2 = int(opt.pop("excl", 0)), float(opt.pop("l2", 10.0))
    wbt = float(opt.pop("wbt", 0.0))
    tag = opt.pop("tag", "T137_dead")
    test = benches(opt.pop("test", POOL3))
    unit = float(pipeline.DEC["sdead_vis"]) * float(pipeline.DEC["w_sdead"])
    bt_models = lomo_models(POOL3, ["lin"], "pool3")["lin"]

    drop = [x for x in str(opt.pop("drop", "")).split(",") if x]
    cols = [i for i, nm in enumerate(DEADF) if nm not in drop]

    def feats_of(d):
        with np.errstate(invalid="ignore"):
            cabs = np.abs(d["N"][:, 0]).astype(float)          # coupling of the ball with the nearer team's nearest player
        return dead_feats(d["feats"], d["E"], cabs)[:, cols]

    tr, keep = {}, {}
    for b in pool:
        data = load(*b)
        for name, d in data.items():
            t = d["truth"]["ctl2"]
            k = d["vis"] & (d["period"] > 0)
            tr[b, name] = (feats_of(d)[k][::3], (t[k][::3] == "D").astype(float))
        if b in test:
            keep[b] = data
        else:
            del data
    out = {}
    for b in test:
        data = keep[b] if b in keep else load(*b)
        labs, tv = {}, {}
        cfgs = ["default"] + ["dead%g" % w for w in ws]
        st = np.zeros(3)
        for name, d in data.items():
            rows = [v for (bb, m), v in tr.items() if m != name and not (excl and bb[0] == b[0])]
            Xs, ys = np.concatenate([v[0] for v in rows]), np.concatenate([v[1] for v in rows])
            m = shape.fit_logistic(Xs, ys, l2=l2, intercept=True)
            base = float(np.log(ys.mean() / (1 - ys.mean())))
            l = ((feats_of(d) - m["mu"]) / m["sd"]) @ m["w"] + float(m["b"])
            ev = np.where(d["vis"], np.tanh((l - base) / 2.0), 0.0)
            t = d["truth"]["ctl2"]
            k = d["vis"] & (d["period"] > 0)
            st += [((l[k] > 0) == (t[k] == "D")).mean(), ((d["dec"][k] == "D") == (t[k] == "D")).mean(), (t[k] == "D").mean()]
            tv[name] = d["truth"]
            f0 = dict(d["feats"])
            bkw = {}
            if wbt:
                f0["bt"] = np.where(d["vis"], np.tanh(logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
                bkw = dict(w_bt=wbt)
            labs["default", name] = a4_short.decode_one(f0, d["period"], bkw) if wbt else d["dec"]
            sd0 = np.nan_to_num(np.asarray(d["feats"]["shape_dead"], float))
            for w in ws:
                labs["dead%g" % w, name] = a4_short.decode_one(dict(f0, shape_dead=sd0 + w * ev / unit), d["period"], bkw)
            if name == tune_shape.TUNE[0] and b == test[0]:
                order = np.argsort(-np.abs(m["w"]))[:12]
                print("   stoppage model (without %s), largest weights: " % name + ", ".join("%s %+.2f" % (DEADF[cols[i]], m["w"][i]) for i in order), flush=True)
        st /= len(data)
        print("%s s%d visible frames: model right on D-vs-live %.4f | decoder %.4f | share of D %.3f" % (b[0], b[1], st[0], st[1], st[2]), flush=True)
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T137 learned visible stoppage evidence (pool %s, excl %d, w_bt %g, drop %s)" % (
            b[0], b[1], ",".join("%s:%d" % x for x in pool), excl, wbt, drop))
        del data
    with open(os.path.join(OUT, "a8_r6_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "w": ws, "excl": excl, "w_bt": wbt, "drop": drop, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def _src_rows(b):
    """Rows of balldead.features for one bench: {match: (X, y, vis mask data)}."""
    from possess import balldead
    out = {}
    data = load(*b)
    for name, d in data.items():
        f = d["feats"]
        X = balldead.features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["visible"], f["pos"], f.get("shape_dead"), f.get("edge_p90"))
        out[name] = dict(d, BX=X)
    return out


def _fit_bd(rows, l2=10.0):
    Xs, ys = np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])
    m = shape.fit_logistic(Xs, ys, l2=l2, intercept=True)
    m["base"] = float(np.log(ys.mean() / (1 - ys.mean())))
    return m


def cmd_deadsrc(opt):
    """T137 through the src flag (w_bd, feats["bd"] = leave-one-match-out evidence of balldead.features).
    excl=1: the model for a bench is fitted without that bench. wbt > 0: on top of w_bt (pooled LOMO)."""
    from possess import balldead
    pool = benches(opt.pop("pool", POOL3))
    ws = [float(x) for x in str(opt.pop("w", "1.0")).split(",")]
    excl, wbt = int(opt.pop("excl", 0)), float(opt.pop("wbt", 0.0))
    tag = opt.pop("tag", "T137_src")
    test = benches(opt.pop("test", POOL3))
    bt_models = lomo_models(POOL3, ["lin"], "pool3")["lin"]
    tr, keep = {}, {}
    for b in pool:
        data = _src_rows(b)
        for name, d in data.items():
            k = d["vis"] & (d["period"] > 0)
            tr[b, name] = (d["BX"][k][::3], (d["truth"]["ctl2"][k][::3] == "D").astype(float))
        if b in test:
            keep[b] = data
        del data
    out = {}
    for b in test:
        data = keep.pop(b) if b in keep else _src_rows(b)
        labs, tv = {}, {}
        cfgs = ["default"] + (["bt"] if wbt else []) + ["bd%g" % w for w in ws]
        for name, d in data.items():
            m = _fit_bd([v for (bb, mm), v in tr.items() if mm != name and not (excl and bb[0] == b[0])])
            tv[name] = d["truth"]
            labs["default", name] = d["dec"]
            f0, bkw = dict(d["feats"]), {}
            if wbt:
                f0["bt"] = np.where(d["vis"], np.tanh(logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
                bkw = dict(w_bt=wbt)
                labs["bt", name] = a4_short.decode_one(f0, d["period"], bkw)
            f0["bd"] = balldead.evidence(d["BX"], d["vis"], m)
            for w in ws:
                labs["bd%g" % w, name] = a4_short.decode_one(f0, d["period"], dict(bkw, w_bd=w))
        out["%s:%d" % b] = R.score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  T137 w_bd, leave-one-match-out (pool %s, excl %d, w_bt %g)" % (
            b[0], b[1], ",".join("%s:%d" % x for x in pool), excl, wbt))
        del data
    with open(os.path.join(OUT, "a8_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "w": ws, "excl": excl, "w_bt": wbt, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_deadfit(opt):
    """balldead.MODEL: all tune matches on every bench of the pool. Prints the constants."""
    from possess import balldead
    rows = []
    for b in benches(opt.pop("pool", POOL3)):
        for name, d in _src_rows(b).items():
            k = d["vis"] & (d["period"] > 0)
            rows.append((d["BX"][k][::3], (d["truth"]["ctl2"][k][::3] == "D").astype(float)))
    m = _fit_bd(rows)
    fmt = lambda v: "[" + ", ".join("%.5g" % x for x in np.atleast_1d(v)) + "]"   # noqa: E731
    print("MODEL: Dict[str, np.ndarray] = {")
    for k in ("mu", "sd", "w"):
        print('    "%s": np.array(%s),' % (k, fmt(m[k])))
    print('    "b": %.5g,' % float(np.atleast_1d(m["b"])[0]))
    print('    "base": %.5g,' % m["base"])
    print("}")
    for k, w in sorted(zip(balldead.FEATS, m["w"]), key=lambda x: -abs(x[1])):
        print("#   %-14s %+.3f" % (k, w))


def cmd_deadprov(opt):
    """Provided match INPUT only (no truth): what w_bd (balldead.MODEL) changes in our own output, and its
    evidence on visible frames against the external benches."""
    from possess import balldead
    R.CONFIGS["BD05"], R.CONFIGS["BD1"], R.CONFIGS["BT5"], R.CONFIGS["BT5BD1"] = dict(w_bd=0.5), dict(w_bd=1.0), dict(w_bt=0.5), dict(w_bt=0.5, w_bd=1.0)
    R.cmd_transfer({"cfg": "BD05,BD1,BT5,BT5BD1", "tag": "T137"})


def cmd_deadtag(opt):
    """Provided match INPUT only (no truth): Smart Tagging SET PIECES clips as an independent check of what
    w_bd changes (as rows 48-49 did for T113). A stoppage is expected from before the clip start to the restart
    (clip start + 3..9 s, row 51), so window = clip start -2..+6 s. Nothing is chosen on it."""
    import a8_tags
    from possess import pipeline
    tags, home, f, per, base = a8_tags.challenge_default()
    play = per > 0
    win = np.zeros(len(per), bool)
    starts = [t["a"] + 1 for t in tags if t["code"] == "SET PIECES"]
    for a in starts:
        win[max(0, a - 50):a + 150] = True
    vis = ~np.isnan(f["ball_x"])
    print("SET PIECES clips %d; window share of play frames %.4f, of visible play frames %.4f" % (len(starts), win[play].mean(), win[play & vis].mean()))
    noD = lambda lab: sum(1 for a in starts if not (lab[a:a + 150] == "D").any())   # noqa: E731
    print("default: clips with no D of ours in their first 6 s: %d" % noD(base))
    for name, kw in (("w_bd 0.5", dict(w_bd=0.5)), ("w_bd 1.0", dict(w_bd=1.0)), ("w_bd 1.5", dict(w_bd=1.5))):
        lab = pipeline.label(f, period=per, dec_kw=kw)
        toD, toL = play & (lab == "D") & (base != "D"), play & (lab != "D") & (base == "D")
        runs = lambda m: int((np.diff(np.concatenate([[0], m.astype(int)])) == 1).sum())   # noqa: E731
        print("%s: frames live -> D %d in %d runs, share inside a window %.3f | D -> live %d in %d runs, share inside a window %.3f | clips with no D in their first 6 s: %d" % (
            name, toD.sum(), runs(toD), win[toD].mean() if toD.any() else 0.0, toL.sum(), runs(toL), win[toL].mean() if toL.any() else 0.0, noD(lab)), flush=True)


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]](a4_short.kv(sys.argv[2:]))
