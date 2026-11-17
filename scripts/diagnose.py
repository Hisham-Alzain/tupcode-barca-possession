"""Where does the current default lose spellF1? (envelope L2-8-segment-fable, tune split only)

    uv run python scripts/diagnose.py                      # the default (pipeline.DEC)
    uv run python scripts/diagnose.py w_touch=0.7          # the default with decoder overrides
    uv run python scripts/diagnose.py w_touch=0.7 ball_noise=0.1 noise=0.1   # + sigma 0.1 m on ball / players

Runs the default decoder on the tune matches the way scripts/evaluate.py does (leave-one-match-out
shape models, cached bench features) and prints:
  (a) bench.explain(): cause of every missed true possession / spurious predicted one (IoU 0.5)
  (b) oracle ceilings (scripts/oracle.py variants): one part of the prediction replaced by truth
  (c) possession length histograms, true vs predicted, with recall / precision per length bin
  (d) frame confusion (visible / hidden ball) and boundary recall by type (within +-1 s)
  (e) stoppage timing per match (envelope L2-11): predicted minus true stop / restart frame for
      visible true boundaries with a predicted one within 10 s, and how the bench ball behaves on
      true stoppage frames (IDSSE freezes it, Metrica's is the bench fill line: a ball-speed
      stoppage rule cannot be validated cleanly on either)
Tracking faults on the ball the touch detector reads (envelope L2-9; distances unchanged, as
ball_noise): teleport=<per second> one-frame jumps of tp_lo..tp_hi m (5-10), stale=<per second>
the same reading repeated for st_lo..st_hi frames (3-8). On the faulted frames the distances to
the nearest players are recomputed from the faulted ball (real_d=0: left as they were, the L2-9 v1
harness). quick=1 prints the tune line only. ball_clean=0 turns the cleaner off.

    uv run python scripts/diagnose.py quick=1 teleport=0.5                 # 1 per 2 s, cleaner on
    uv run python scripts/diagnose.py quick=1 teleport=0.5 ball_clean=0    # same faults, cleaner off
ctx=0: shape team model without the lead / lag context columns (shape.team_context, envelope L2-10).
fast=0: stoppage model without the fast-player columns (shape.FAST_FEATS, envelope L2-16).
bench=real1: the realistic bench ball (bench.VARIANTS, envelope L2-12) instead of the original.

    uv run python scripts/diagnose.py transfer             # envelope L2-13 transfer check

`transfer` (LOOP.md protocol amendment 2026-11-14): the stoppage signals of the default on the
provided match's INPUT (cached features and player positions of the automatic tracking; the ground
truth is never opened) next to the tune matches on the original and the realistic bench: how often
the player-only stoppage model says "stopped", the share and run lengths of predicted D for the
L2-10, L2-13 and current defaults, and where the ball is on predicted-D frames; since L2-14 also
how often the ball is within 1 m of one team with the other 1-1.5 m away (where the distance cap
acts), and since L2-16 the speed of the fast players (shape.FAST_FEATS) and the labels with and
without it. Aggregates only; writes eval/results/transfer_hidden_stop.json (L2-13's run:
transfer_stoppage.json, L2-14 / L2-15: transfer_decoder.json). Nothing is tuned or chosen on the provided match.
`segment` (envelope L2-15): short possessions, turnover timing against the touch detector and
who is right on contested touches, per tune match: `diagnose.py segment bench=real1 duel=0`.
`dhide`: sensitivity that goes with it (see cmd_dhide): the same two defaults on the realistic bench
when the ball is also hidden inside the true stoppages, as the provided match's input suggests.
`cleanup` (envelope L2-17, possess/cleanup.py: minimum-duration / merge rules on finished runs):
    diagnose.py cleanup save bench=real1      # one decode per tune match -> cleanup_<match>_<bench>_v1.npz
    diagnose.py cleanup step0 bench=real1     # predicted runs by length x flank x ball: match rates
    diagnose.py cleanup grid bench=real1      # the fixed grid on the saved labels -> tune_cleanup_real1.json
    diagnose.py cleanup check m_flicker=0.4   # per match, paired bootstrap, counts, noise gate (saved labels)
    diagnose.py cleanup transfer m_flicker=0.4   # provided-match INPUT aggregates -> transfer_cleanup.json
Result: negative, the default has no clean-up (short runs are rare after the decoder).
`edges` (envelope L2-18, possess/edges.py: move the edges of predicted stoppages in hidden blocks):
    diagnose.py edges save bench=real1 [noise=0.05 | drop=1] [tag=m6a c_out_hid=8.0]   # default WITHOUT the edge pass
    diagnose.py edges d0 | profiles | taker | m3      # step 0: edge table + oracle, aligned profiles, taker cue, edge AUC
    diagnose.py edges grid | m4 | row ed_exit=1 ...   # the fixed grid -> eval/results/tune_edges.json
    diagnose.py edges check|dhide|transfer ed_exit=1 ed_w=1.0 ed_win=2.0   # gates (bootstrap, noise, dhide, provided INPUT)
Result: exit snap adopted (shape.DEC_SHAPE ed_exit=1); `--dec ed_exit=0` gives the L2-16 default.
`prior` (envelope L2-21, shape.prior_shift: share of D among the hidden frames, estimated without labels):
    diagnose.py prior check [bench=real2] [seed=1 names=metrica1] [noise=0.1] [drop=1]   # off vs on, hidden D share, bootstrap
    diagnose.py prior null | transfer      # shift under random hiding; provided INPUT -> transfer_sd_prior.json
Result: adopted (shape.DEC_SHAPE sd_prior=1); `--dec sd_prior=0` gives the L2-20 default.
Otherwise never run on held-out matches, the public window or challenge data.
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench  # noqa: E402
import oracle  # noqa: E402
import tune_shape  # noqa: E402
from possess import shape  # noqa: E402

pm = bench.pm
BINS = [0, 2, 4, 6, 10, 15, 25, 40, 1e9]            # seconds
ED_OFF = dict(ed_exit=0, ed_entry=0, ed_dx=0.0, ed_de=0.0)   # the default before the L2-18 edge post-pass (possess/edges.py)
FPS = 25.0


def inject_faults(x, y, teleport=0.0, stale=0.0, seed=2, tp_lo=5.0, tp_hi=10.0, st_lo=3, st_hi=8):
    """(x, y, changed) copy of the ball with faults at the given rates per second of visible ball
    (fixed seed). teleport: one frame moved tp_lo..tp_hi m in a random direction. stale: a reading
    repeated for the next st_lo..st_hi frames (only where those frames are visible)."""
    rng = np.random.default_rng(seed)
    x, y = x.copy(), y.copy()
    changed = np.zeros(len(x), bool)
    vis = np.flatnonzero(~np.isnan(x))
    if teleport:
        idx = vis[rng.random(len(vis)) < teleport / FPS]
        r, th = rng.uniform(tp_lo, tp_hi, len(idx)), rng.uniform(0, 2 * np.pi, len(idx))
        x[idx] += r * np.cos(th)
        y[idx] += r * np.sin(th)
        changed[idx] = True
    if stale:
        for i in vis[rng.random(len(vis)) < stale / FPS].tolist():
            k = int(rng.integers(int(st_lo), int(st_hi) + 1))
            if i + k < len(x) and not np.isnan(x[i:i + k + 1]).any():
                x[i + 1:i + k + 1], y[i + 1:i + k + 1] = x[i], y[i]
                changed[i + 1:i + k + 1] = True
    return x, y, changed


def faulted(name, case, feats, real_d=1, **fk):
    """feats with the faults in the touch ball and, with real_d=1, in the distances to the nearest
    home / away player on the faulted frames too (recomputed from the cached player positions, as
    a real fault would be seen; the stoppage detector and the shape evidence are left alone)."""
    x, y, ch = inject_faults(feats["ball_x"], feats["ball_y"], **fk)
    out = dict(feats, ball_x=x, ball_y=y)
    if real_d and ch.any():
        pos = tune_shape.positions(name, case)[ch]                    # [m, 2, slots, 2]
        with np.errstate(invalid="ignore"), shape._quiet():
            d = np.nanmin(np.hypot(pos[:, :, :, 0] - x[ch][:, None, None], pos[:, :, :, 1] - y[ch][:, None, None]), axis=2)
        for t, key in enumerate(("d_home", "d_away")):
            out[key] = feats[key].copy()
            out[key][ch] = np.nan_to_num(d[:, t], nan=99.0)
    return out


FAULT_KEYS = ("teleport", "stale", "tp_lo", "tp_hi", "st_lo", "st_hi", "real_d")


def default_run(kw=None, **deg):
    """[(name, case, feats with shape emissions, labels)] for the tune matches, default + kw."""
    fk = {k: deg.pop(k) for k in FAULT_KEYS if k in deg}
    ctx = deg.pop("ctx", None)                 # ctx=0: team model without lead / lag context (before L2-10)
    fast = deg.pop("fast", None)               # fast=0: stoppage model without the fast-player columns (before L2-16)
    data = tune_shape.load_all(**deg)
    clean = tune_shape.load_all() if any(deg.values()) else data     # models are fitted on clean data
    if fk.get("teleport") or fk.get("stale"):
        data = [(n, c, faulted(n, c, f, **fk), s) for n, c, f, s in data]
    data_e = tune_shape.with_emissions(data, tune_shape.lomo_models(clean), ctx=ctx, fast=fast)
    res = tune_shape.run(data_e, dict(shape.DEC_SHAPE, **(kw or {})))
    return [(n, c, f, r[4]) for (n, c, f), r in zip(data_e, res)], res


def matched_flags(gs, ps, thr=0.5):
    """One-to-one greedy IoU matching as in the scorer: (gt matched?, pred matched?)."""
    ov = pm.spell_overlaps(gs, ps)
    g, p = np.zeros(len(gs), bool), np.zeros(len(ps), bool)
    for iou, gi, pj in ov:
        if iou < thr:
            break
        if g[gi] or p[pj]:
            continue
        g[gi] = p[pj] = True
    return g, p


def length_table(spells, ok):
    L = np.array([(b - a) / FPS for a, b, _ in spells])
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (L >= lo) & (L < hi)
        rows.append((m.sum(), ok[m].mean() if m.any() else float("nan")))
    return rows


def boundaries(codes, runs):
    """[(frame, kind)] kind in 'turnover', 'stop' (live -> D), 'restart' (D -> live)."""
    out = []
    for a, b in runs:
        seg = codes[a:b]
        for i in np.flatnonzero(seg[1:] != seg[:-1]):
            x, y = seg[i], seg[i + 1]
            kind = "stop" if y == pm.D else "restart" if x == pm.D else "turnover"
            out.append((a + i + 1, kind))
    return out


def stoppage_timing(g, p, runs, hidden, feats):
    """One line: quartiles of predicted - true stop and restart time (s, visible true boundaries
    with a predicted one of the same kind within 10 s), and the share of visible true-D frames on
    which the ball is slower than 0.3 m/s (possess.deadball2.lag_speed)."""
    from possess.deadball2 import lag_speed

    out = []
    pb = boundaries(p, runs)
    for kind in ("stop", "restart"):
        pf = np.array([f for f, k in pb if k == kind])
        d = []
        for f, k in boundaries(g, runs):
            if k == kind and not hidden[f] and len(pf):
                c = pf[np.abs(pf - f) <= 250] - f
                if len(c):
                    d.append(c[np.argmin(np.abs(c))] / FPS)
        q = np.percentile(d, [25, 50, 75]) if d else [float("nan")] * 3
        out.append("%s n %d q25/50/75 %+.2f %+.2f %+.2f s" % ((kind, len(d)) + tuple(q)))
    v = lag_speed(feats["ball_x"], feats["ball_y"], FPS)
    m = (g == pm.D) & np.isfinite(v)
    out.append("ball < 0.3 m/s on %.2f of visible true-D frames" % ((v[m] < 0.3).mean() if m.any() else float("nan")))
    return "  |  ".join(out)


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def stoppage_signals(f, period, sd, labs, truth=None, ps90=None):
    """Aggregates of the stoppage signals for one match. f: features.compute output (ball, distances),
    sd: player-only stoppage log-odds, labs: {tag: labels}. truth only for the external matches.
    ps90: shape.FAST_FEATS[0] per frame (L2-16: speed of the fast players, 1 s mean)."""
    play = np.asarray(period) > 0
    vis = play & ~np.isnan(f["ball_x"])
    hid = play & ~vis
    far = np.fmin(f["d_home"], f["d_away"])
    step = np.hypot(np.diff(f["ball_x"], prepend=np.nan), np.diff(f["ball_y"], prepend=np.nan)) * FPS
    half_hours = play.sum() / FPS / 1800.0
    near, oth = np.fmin(f["d_home"], f["d_away"])[vis], np.fmax(f["d_home"], f["d_away"])[vis]
    out = {"vis: a player <= 1 m": float((near <= 1.0).mean()),      # L2-14: where the distance cap acts
           "vis: one team <= 1 m, other 1-1.5 m": float(((near <= 1.0) & (oth > 1.0) & (oth <= 1.5)).mean()),
           "vis: both teams <= 1.5 m": float((oth <= 1.5).mean()),
           "shape_dead>0 visible": float((sd[vis] > 0).mean()), "shape_dead>0 hidden": float((sd[hid] > 0).mean()),
           "shape_dead p50 visible": float(np.median(sd[vis])), "shape_dead p90 visible": float(np.percentile(sd[vis], 90)),
           "shape no-evidence share": float((sd[play] == 0).mean())}
    if ps90 is not None:                                 # L2-16: the fast-player feature itself
        for tag, m in (("visible", vis & np.isfinite(ps90)), ("hidden", hid & np.isfinite(ps90))):
            out["ps90 median m/s " + tag] = float(np.median(ps90[m]))
            out["ps90 < 2 m/s " + tag] = float((ps90[m] < 2.0).mean())
            out["ps90 > 4 m/s " + tag] = float((ps90[m] > 4.0).mean())
    # L2-15: what the touch detector sees (duel rule of decode.touch_hold): ball noise, uncontested
    # and contested (both teams within 1 m) impulse onsets per minute of visible ball
    from possess.ballclean import clean_ball
    from possess.decode import SG7, ball_impulse, ball_noise

    dh, da = np.nan_to_num(f["d_home"], nan=99.0), np.nan_to_num(f["d_away"], nan=99.0)
    bx, by, _ = clean_ball(f["ball_x"], f["ball_y"], np.minimum(dh, da), 1.0, FPS)
    imp = ball_impulse(bx, by)["impulse"] & vis
    both, one = imp & (dh <= 1.0) & (da <= 1.0), imp & ((dh <= 1.0) != (da <= 1.0))
    mins = vis.sum() / FPS / 60.0
    out.update({"touch: ball noise sigma m": ball_noise(bx, by, np.convolve(bx, SG7, "same"), np.convolve(by, SG7, "same")),
                "touch: uncontested onsets / min": float(len(_runs(one)) / mins),
                "touch: contested onsets / min": float(len(_runs(both)) / mins),
                "touch: contested share of touch frames": float(both.sum() / max(1, both.sum() + one.sum()))})
    for tag, lab in labs.items():
        d = lab == "D"
        r = _runs(d & play)
        dv = d & vis
        out.update({
            tag + " D share visible": float(d[vis].mean()), tag + " D share hidden": float(d[hid].mean()),
            tag + " D runs / half hour": float(len(r) / half_hours),
            tag + " D run median s": float(np.median(r) / FPS) if len(r) else 0.0,
            tag + " D run max s": float(r.max() / FPS) if len(r) else 0.0,
            tag + " possessions / half hour": float(len(_runs((lab == "H") & play)) + len(_runs((lab == "A") & play))) / half_hours,
            tag + " D vis: player <= 1 m": float((far[dv] <= 1.0).mean()) if dv.any() else 0.0,
            tag + " D vis: ball > 9 m from all": float((far[dv] > 9.0).mean()) if dv.any() else 0.0,
            tag + " D vis: ball step > 2 m/s": float((step[dv] > 2.0).mean()) if dv.any() else 0.0})
    if truth is not None:
        td = truth == "D"
        out["truth D share visible"], out["truth D share hidden"] = float(td[vis].mean()), float(td[hid].mean())
        out["truth D runs / half hour"] = float(len(_runs(td & play)) / half_hours)
    return out


def cmd_transfer():
    import json

    from possess import pipeline

    # tag -> (decoder overrides, shape.DEAD_FAST). L2-16: "L2-15" = the current decoder with the
    # stoppage model before the fast-player columns. (L2-10 / L2-13 columns: transfer_stoppage.json.)
    cfgs = {"L2-14": (dict(shape.DEC_SHAPE_L2_14, **ED_OFF), 0), "L2-15": (dict(ED_OFF), 0), "L2-16": (dict(ED_OFF), 1),
            "now": ({}, 1)}                           # L2-18: "now" = L2-16 + the edge post-pass

    def labels(f, period, **k):
        out = {}
        for t, (kw, fast) in cfgs.items():
            shape.DEAD_FAST = fast
            try:
                out[t] = pipeline.label(f, dec_kw=kw, period=period, **k)
            finally:
                shape.DEAD_FAST = 1
        return out

    res = {}
    cache = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")
    if os.path.exists(cache) and os.path.exists(os.path.join(os.path.dirname(cache), "positions_v1.npz")):
        f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"), cache)   # input caches only
        per = pipeline.challenge_period(len(f["ball_x"]))
        sf = shape.features(f["pos"], per)
        sd = shape.emissions(sf, shape.MODELS)[1]
        res["provided"] = stoppage_signals(f, per, sd, labels(f, per), ps90=sf["fast"][:, 0])
    else:
        print("provided match: input caches missing, skipped (run the pipeline once first)")
    for variant in ("", "real1"):
        data = tune_shape.load_all(variant=variant)
        models = tune_shape.lomo_models(data)
        for name, case, feats, sf in data:
            sd = shape.emissions(sf, models[name])[1]
            f = dict(bench.case_features(name, case, variant=variant), pos=tune_shape.positions(name, case))
            labs = labels(f, case["period"], shape_models=models[name])
            res["%s%s" % (name[:1] + name[-1], "r" if variant else "o")] = stoppage_signals(f, case["period"], sd, labs, case["truth"], ps90=sf["fast"][:, 0])
    names = list(res)
    print("o = original bench, r = realistic bench (real1); models leave-one-match-out on the bench, shape.MODELS on the provided match")
    print("%-34s" % "signal" + "".join("%9s" % n for n in names))
    for k in res[names[-1]]:
        print("%-34s" % k + "".join("%9s" % ("%.3f" % res[n][k] if k in res[n] else "") for n in names))
    # L2-13's run: transfer_stoppage.json; L2-14 / L2-15: transfer_decoder.json
    with open(os.path.join(ROOT, "eval", "results", "transfer_hidden_stop.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def cmd_dhide(ps=(0.0, 0.5, 0.8, 0.95), variant="real1", post=None):
    """Sensitivity for the transfer check (envelope L2-13). On the provided match's input the
    player-only model says 'stopped' on 16% of visible and 60% of hidden frames (bench: ~40% on
    both): the real ball is mostly lost while play is dead, which the bench's hidden blocks (drawn
    independently of the truth) do not reproduce. Here the ball is also hidden inside a share p of
    the true stoppages >= 3 s (from 0.5-2 s after the stop to 0-1 s before the restart), in memory
    only, and the L2-13 default, the default before L2-16 ("L2-15": current decoder, stoppage
    model without the fast-player columns) and the current default are scored on the same input.
    L2-16 also prints the share of hidden frames labelled D next to the true share: on the
    provided match the fast-player columns raise it (0.673 -> 0.711) although they leave it flat
    on the plain bench, where hidden blocks are drawn independently of the truth."""
    from possess import pipeline
    from possess.deadball2 import detect, features as dead_features
    from possess.decode import decode_halves

    data = tune_shape.load_all(variant=variant)
    models = tune_shape.lomo_models(data)
    for p in ps:
        rows, info, dsh = {}, [], {}
        for name, case, _, sf in data:
            rng = np.random.default_rng(7)
            t = case["truth"]
            td = t == "D"
            f = bench.case_features(name, case, variant=variant)
            extra = np.zeros(len(t), bool)
            d = np.diff(np.concatenate([[False], td, [False]]).astype(np.int8))
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a >= 75 and rng.random() < p:
                    extra[a + int(rng.uniform(0.5, 2.0) * FPS): b - int(rng.uniform(0.0, 1.0) * FPS)] = True
            bx, by = np.where(extra, np.nan, f["ball_x"]), np.where(extra, np.nan, f["ball_y"])
            dh, da = np.where(extra, np.nan, f["d_home"]), np.where(extra, np.nan, f["d_away"])
            df = dead_features(bx, by, dh, da, f["cen_home"], f["cen_away"])
            vis = ~np.isnan(bx)
            st, sd = shape.emissions(sf, models[name])
            sd0 = shape.emissions(sf, models[name], fast=0)[1]
            feats = {"d_home": np.nan_to_num(dh, nan=99.0), "d_away": np.nan_to_num(da, nan=99.0), "visible": vis,
                     "dead": detect(df, f["ml_dead"] & ~extra, **pipeline.DEAD2), "ball_x": bx, "ball_y": by,
                     "shape_team": st, "shape_dead": sd, "edge_p90": sf["p90"]}
            ok = np.isin(t, ["H", "A", "D"])
            info.append("%s visible %.2f, true D on visible %.2f hidden %.2f, players 'stopped' visible %.2f hidden %.2f" % (
                name, vis[ok].mean(), td[ok & vis].mean(), td[ok & ~vis].mean(), (sd[ok & vis] > 0).mean(),
                (sd[ok & ~vis] > 0).mean()))
            for tag, kw, s_dead in (("L2-13", shape.DEC_SHAPE_L2_13, sd0), ("L2-15", shape.DEC_SHAPE_L2_16, sd0),
                                    ("L2-16", shape.DEC_SHAPE_L2_16, sd), ("now", shape.DEC_SHAPE, sd)):
                lab = decode_halves(case["period"], feats=dict(feats, shape_dead=s_dead), **dict(pipeline.DEC_BASE, **kw))
                rows.setdefault(tag, []).append(pm.evaluate(pm.label_codes(t), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"])
                dsh.setdefault(tag, []).append((lab == "D")[ok & ~vis].mean())
                if post and tag == "L2-16":                # L2-18: the edge post-pass on the same decode
                    lab = post(name, lab, case["period"], vis)
                    rows.setdefault("L2-16+edges", []).append(pm.evaluate(pm.label_codes(t), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"])
                    dsh.setdefault("L2-16+edges", []).append((lab == "D")[ok & ~vis].mean())
        print("p = %.2f" % p)
        for line in info:
            print("   " + line)
        for tag, v in rows.items():
            print("   %-6s %.4f  (%s)   D share of hidden frames %s" % (
                tag, np.mean(v), "  ".join("%.3f" % x for x in v), "  ".join("%.3f" % x for x in dsh[tag])), flush=True)


def cmd_segment(kw=None):
    """Envelope L2-15: why short possessions are missed, per tune match (bench= as usual).
    (1) true possessions < 4 s: how many, how many with the ball hidden, how many found (IoU 0.5);
    (2) timing: nearest touch of the new team (decode.touch_hold, uncontested) minus the true
        switch, for true turnovers with the ball visible +-3 s: do the sources date a switch alike?
    (3) who touched the ball: share of touch frames (truth live 2 frames later) on which the team
        within 1 m (uncontested) / the nearer team (contested, both within 1 m) is the true team."""
    from possess.ballclean import clean_ball
    from possess.decode import ball_impulse, touch_hold

    rows, res = default_run(kw)
    print("tune", tune_shape.fmt(res))
    for name, case, feats, lab in rows:
        g = pm.label_codes(case["truth"])
        runs = pm.valid_runs(g)
        p = pm.fill_abstentions(pm.label_codes(lab), runs)
        gs = pm.possession_spells(g, runs)
        g_ok, _ = matched_flags(gs, pm.possession_spells(p, runs))
        dh, da, vis = feats["d_home"], feats["d_away"], feats["visible"]
        short = [(vis[a:b].mean() < 0.1, ok) for (a, b, _), ok in zip(gs, g_ok) if b - a < 4 * FPS]
        bx, by, _ = clean_ball(feats["ball_x"], feats["ball_y"], np.minimum(dh, da), 1.0, FPS)
        t0 = touch_hold(bx, by, dh, da, 1.0, 0.0)                # touch frames only
        off, w = [], int(3 * FPS)
        for f, kind in boundaries(g, runs):
            if kind != "turnover" or not vis[max(0, f - w):f + w].all():
                continue
            tt = np.flatnonzero(t0[max(0, f - w):f + w] == (1.0 if g[f] == pm.H else -1.0)) + max(0, f - w)
            if len(tt):
                off.append((tt[np.argmin(np.abs(tt - f))] - f) / FPS)
        nxt = np.concatenate([g[2:], [pm.D, pm.D]])
        live = ball_impulse(bx, by)["impulse"] & vis & np.isin(nxt, [pm.H, pm.A])
        near_h = (dh < da) == (nxt == pm.H)
        one, both = live & ((dh <= 1.0) != (da <= 1.0)), live & (dh <= 1.0) & (da <= 1.0)
        print("%-9s true < 4 s: %d of %d (hidden %d, found %d; visible or mixed %d, found %d) | new team's touch - true "
              "switch, n %d: q10/50/90 %+.2f %+.2f %+.2f s | nearer team right: uncontested %.3f (n %d), contested %.3f (n %d)" % (
                  name, len(short), len(gs), sum(h for h, _ in short), sum(h and ok for h, ok in short),
                  sum(not h for h, _ in short), sum((not h) and ok for h, ok in short), len(off),
                  *np.percentile(off, [10, 50, 90]), near_h[one].mean(), one.sum(), near_h[both].mean(), both.sum()))


# ---- envelope L2-17: segment clean-up on saved labels (possess/cleanup.py) -------------------
CL_OFF = dict(m_flicker=0.0, m_bridge=0.0, m_island=0.0, m_sliver=0.0)
CL_BUCKETS = [0, 0.4, 0.8, 1.2, 2, 3, 5, 1e9]        # seconds
CL_GRID = {"A": [dict(m_flicker=v) for v in (0.4, 0.8, 1.2, 2.0)],
           "B": [dict(m_bridge=v) for v in (1.0, 2.0, 3.0, 5.0)],
           "C": [dict(m_island=v) for v in (0.6, 1.0, 2.0, 3.0)],
           "D": [dict(m_sliver=v) for v in (0.4, 0.8, 1.2)]}


def _cl_path(name, variant, deg):
    tag = "".join("_%s%g" % (k, v) for k, v in sorted(deg.items()) if v)
    return os.path.join(bench.CACHE, "cleanup_%s_%s%s_v1.npz" % (name, variant or "orig", tag))


def _touch(feats, period):
    """decode.touch_frames per half, as decode_halves builds it for cl_protect."""
    from possess.decode import touch_frames

    out = np.zeros(len(period))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(period, int), [0]])) != 0)
    for a, b in zip(edges[:-1], edges[1:]):
        if period[a] > 0:
            out[a:b] = touch_frames(feats["ball_x"][a:b], feats["ball_y"][a:b], feats["d_home"][a:b], feats["d_away"][a:b])
    return out


def cl_save(deg):
    """One decode of the default WITHOUT clean-up per tune match; labels + truth + period +
    visible + touch frames go to a new cache (cleanup_<match>_<bench>[_deg]_v1.npz)."""
    rows, res = default_run(dict(CL_OFF, **ED_OFF), **dict(deg))
    print("saved", tune_shape.VARIANT or "orig", deg, tune_shape.fmt(res))
    for name, case, feats, lab in rows:
        np.savez_compressed(_cl_path(name, tune_shape.VARIANT, deg), lab=lab, truth=case["truth"],
                            period=case["period"], visible=feats["visible"], touch=_touch(feats, case["period"]))


def cl_load(variant="real1", deg=None):
    out = []
    for name in tune_shape.TUNE:
        z = np.load(_cl_path(name, variant, deg or {}))
        out.append((name, {k: z[k] for k in z.files}))
    return out


def cl_apply(d, kw, counts=None):
    from possess.cleanup import cleanup

    return cleanup(d["lab"], d["period"], d["visible"], d["touch"], counts=counts, **kw)


def cl_score(data, kw):
    """[(name, spellF1, n_pred, n_gt, labels)] of the saved default labels after clean-up `kw`."""
    res = []
    for name, d in data:
        lab = cl_apply(d, kw)
        r = pm.evaluate(pm.label_codes(d["truth"]), pm.label_codes(lab), bootstrap_n=0)
        res.append((name, r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"], lab))
    return res


def cl_step0(variant):
    """Step 0 of the envelope: predicted team spells and D runs by length x flank context x
    hidden / visible majority: count, share matched at IoU 0.5 (team spells) or share with
    >= 50% true D (D runs), and the match rate of the neighbouring team spells."""
    import json

    tab = {}
    for name, d in cl_load(variant):
        g = pm.label_codes(d["truth"])
        runs = pm.valid_runs(g)
        p = pm.fill_abstentions(pm.label_codes(d["lab"]), runs)
        ps = pm.possession_spells(p, runs)
        _, p_ok = matched_flags(pm.possession_spells(g, runs), ps)
        ok = {(a, b): o for (a, b, _), o in zip(ps, p_ok)}
        segs = pm.segments(p, runs)
        for i, (a, b, k) in enumerate(segs):
            prev = segs[i - 1] if i and segs[i - 1][1] == a else None
            nxt = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == b else None
            pk, nk = (prev[2] if prev else None), (nxt[2] if nxt else None)
            if k == pm.D:
                kind = "D"
                ctx = "edge" if pk is None or nk is None else "same team both sides" if pk == nk else "two teams"
                good = (g[a:b] == pm.D).mean() >= 0.5
            else:
                kind = "team"
                ctx = ("edge" if pk is None or nk is None else "D both sides" if pk == nk == pm.D else
                       "other team both sides" if pk == nk else "D before, other after" if pk == pm.D else
                       "other before, D after")
                good = ok[(a, b)]
            nb = [ok[(s[0], s[1])] for s in (prev, nxt) if s is not None and s[2] != pm.D]
            L = (b - a) / FPS
            bucket = max(j for j, lo in enumerate(CL_BUCKETS[:-1]) if L >= lo)
            hid = "hidden" if (~d["visible"][a:b]).mean() > 0.5 else "visible"
            tch = bool(kind == "team" and (d["touch"][a:b] == (1.0 if k == pm.H else -1.0)).any())
            for key in ((kind, ctx, bucket, hid), (kind, ctx, bucket, "all")):
                t = tab.setdefault(key, {"n": 0, "good": 0, "nb": 0, "nb_ok": 0, "per_match": Counter(),
                                         "touched": 0, "touched_good": 0})
                t["n"] += 1; t["good"] += bool(good); t["nb"] += len(nb); t["nb_ok"] += int(sum(nb))
                t["per_match"][name] += 1; t["touched"] += tch; t["touched_good"] += tch and bool(good)
    print("step 0 (%s): kind | context | length s | ball | n (per match) | matched@0.5 or D>=50%% true | neighbours matched | "
          "touched n / matched" % variant)
    rows = []
    for key in sorted(tab):
        t = tab[key]
        kind, ctx, bucket, hid = key
        row = dict(kind=kind, context=ctx, lo=CL_BUCKETS[bucket], hi=CL_BUCKETS[bucket + 1], ball=hid, n=t["n"],
                   rate=t["good"] / t["n"], nb_rate=t["nb_ok"] / t["nb"] if t["nb"] else None,
                   per_match=dict(t["per_match"]), touched=t["touched"],
                   touched_rate=t["touched_good"] / t["touched"] if t["touched"] else None)
        rows.append(row)
        if t["n"] >= 15 and ctx != "edge" and (hid == "all" or bucket <= 4):
            print("  %-4s %-22s %3g-%-4s %-7s %4d (%s)  %.2f  nb %s  touched %d %s%s" % (
                kind, ctx, row["lo"], "%g" % row["hi"] if row["hi"] < 1e8 else "inf", hid, t["n"],
                "/".join(str(t["per_match"][n]) for n in tune_shape.TUNE), row["rate"],
                "%.2f" % row["nb_rate"] if t["nb"] else "-", t["touched"],
                "%.2f" % row["touched_rate"] if t["touched"] else "-", "   <-- below 0.4" if row["rate"] < 0.4 else ""))
    with open(os.path.join(ROOT, "eval", "results", "cleanup_step0_%s.json" % variant), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    hit = [r for r in rows if r["n"] >= 15 and r["rate"] < 0.4 and r["context"] != "edge" and r["ball"] == "all"]
    print("STOP RULE: %d buckets (ball = all) with >= 15 runs and rate < 0.4 -> %s" % (len(hit), "run the grid" if hit else "NEGATIVE, no grid"))


def cl_grid(variant, cl_dir=0):
    """The fixed 25-config grid of the envelope on the saved labels (no decoding)."""
    import json

    data = cl_load(variant)
    other = cl_load("" if variant else "real1")           # the other bench, reported next to it
    base = cl_score(data, {})
    b_mean = float(np.mean([r[1] for r in base]))
    rows = []

    def add(tag, kw):
        res, o = cl_score(data, kw), cl_score(other, kw)
        row = dict(tag=tag, kw=kw, spellF1=float(np.mean([r[1] for r in res])), per_match=[round(r[1], 4) for r in res],
                   n_pred=[r[2] for r in res], other_bench=float(np.mean([r[1] for r in o])))
        rows.append(row)
        print("%-3s %-58s %s | other bench %.4f" % (tag, kw, tune_shape.fmt(res), row["other_bench"]), flush=True)
        return row

    add("off", {})
    best = {}
    for g, cfgs in CL_GRID.items():
        got = [add(g, dict(c, cl_dir=cl_dir) if g == "D" and cl_dir else c) for c in cfgs]
        top = max(r["spellF1"] for r in got)
        pick = next(r for r in got if r["spellF1"] >= top - 0.002)      # smaller threshold inside a plateau
        if pick["spellF1"] > b_mean:                                    # the plateau pick itself must beat off
            best[g] = pick["kw"]
        print("   %s: best %s, %s" % (g, pick["kw"], "kept" if g in best else "does not beat off, dropped"))
    for g in ("A", "D"):                                                # E: protect
        add("E", dict(best.get(g, CL_GRID[g][0]), cl_protect=1))
    for v in (1, 2):                                                    # F: hidden / visible-majority flickers only
        add("F", dict(best.get("A", CL_GRID["A"][0]), cl_vis=v))
    combos = [("A", "B"), ("A", "C"), ("B", "C"), ("A", "B", "C"), ("A", "B", "C", "D")]
    got = []
    for c in combos:
        kw = {}
        for g in c:
            kw.update(best.get(g, {}))
        got.append(add("G", kw) if kw and not any(kw == r["kw"] for r in rows) else None)
    got = [r for r in got if r]
    if got:
        add("G", dict(max(got, key=lambda r: r["spellF1"])["kw"], cl_protect=1))
    # gate 4: the config picked on two matches must not lose on the third
    for k, name in enumerate(tune_shape.TUNE):
        two = [i for i in range(len(tune_shape.TUNE)) if i != k]
        pick = max(rows, key=lambda r: np.mean([r["per_match"][i] for i in two]))
        print("gate 4: picked without %-9s -> %s %s: there %.4f vs off %.4f" % (
            name, pick["tag"], pick["kw"], pick["per_match"][k], rows[0]["per_match"][k]))
    with open(os.path.join(ROOT, "eval", "results", "tune_cleanup_%s.json" % (variant or "orig")), "w") as fh:
        json.dump(dict(protocol="envelope L2-17 fixed grid on saved default labels (diagnose.py cleanup save)",
                       bench=variant or "orig", rows=rows), fh, indent=1)


def _cl_boot(d, base, new, n_draws=2000, seed=0):
    """Paired block bootstrap over TRUE possessions of one match: every predicted spell is
    attached to the true spell it overlaps most (nearest by midpoint when none), a draw resamples
    true spells with their matched flags (IoU 0.3..0.7) and attached predictions.
    Returns per-draw mean-F1 of (base, new)."""
    g = pm.label_codes(d["truth"])
    runs = pm.valid_runs(g)
    gs = pm.possession_spells(g, runs)
    ga, gb = np.array([s[0] for s in gs]), np.array([s[1] for s in gs])
    picks = np.random.default_rng(seed).integers(0, len(gs), (n_draws, len(gs)))
    out = []
    for lab in (base, new):
        ps = pm.possession_spells(pm.fill_abstentions(pm.label_codes(lab), runs), runs)
        cnt = np.zeros(len(gs))
        for a, b, _ in ps:
            ov = np.minimum(gb, b) - np.maximum(ga, a)
            cnt[np.argmax(ov) if ov.max() > 0 else np.argmin(np.abs((ga + gb) / 2 - (a + b) / 2))] += 1
        f1 = np.zeros(n_draws)
        for thr in pm.SPELL_IOUS:
            tp = matched_flags(gs, ps, thr)[0][picks].sum(axis=1)
            f1 += 2.0 * tp / (cnt[picks].sum(axis=1) + len(gs))
        out.append(f1 / len(pm.SPELL_IOUS))
    return out


def cl_check(kw):
    """Gates of the envelope for one clean-up config, all from saved labels."""
    for variant in ("real1", ""):
        data = cl_load(variant)
        b, c = cl_score(data, {}), cl_score(data, kw)
        print("%-5s off %s\n      on  %s" % (variant or "orig", tune_shape.fmt(b), tune_shape.fmt(c)))
        if variant:
            diffs = []
            for (name, d), rb, rc in zip(data, b, c):
                x, y = _cl_boot(d, rb[4], rc[4])
                diffs.append(y - x)
                print("      %-9s paired bootstrap gain %+.4f CI90 [%+.4f, %+.4f]" % ((name, np.mean(y - x)) + tuple(np.percentile(y - x, [5, 95]))))
            m = np.mean(diffs, axis=0)
            print("      mean of 3 paired bootstrap gain %+.4f CI90 [%+.4f, %+.4f]  P(gain <= 0) %.3f" % (
                (m.mean(),) + tuple(np.percentile(m, [5, 95])) + ((m <= 0).mean(),)))
            for name, d in data:
                counts = {}
                lab = cl_apply(d, kw, counts)
                n_runs = int((d["lab"][1:] != d["lab"][:-1]).sum()) + 1
                print("      %-9s runs %d, relabelled by rule %s, frames changed %.4f" % (name, n_runs, counts, (lab != d["lab"]).mean()))
    for deg in (dict(noise=0.05), dict(noise=0.1), dict(drop=1)):
        try:
            data = cl_load("real1", deg)
        except FileNotFoundError:
            print("noise gate %s: labels not saved" % deg)
            continue
        print("noise gate real1 %-16s off %s\n%34s on  %s" % (deg, tune_shape.fmt(cl_score(data, {})), "", tune_shape.fmt(cl_score(data, kw))))


def cl_transfer(kw):
    """Transfer check on the provided match's INPUT (cached features + positions, aggregates only;
    the ground truth is never opened): the default with and without the clean-up, next to the
    three tune matches on real1. Nothing is chosen here."""
    import json

    from possess import pipeline

    def stats(lab, period):
        play = np.asarray(period) > 0
        hh = play.sum() / FPS / 1800.0
        r = _runs((lab == "D") & play)
        return {"possessions / half hour": (len(_runs((lab == "H") & play)) + len(_runs((lab == "A") & play))) / hh,
                "D runs / half hour": len(r) / hh, "D run median s": float(np.median(r) / FPS) if len(r) else 0.0,
                "D share": float((lab == "D")[play].mean())}

    def one(lab, period, visible, touch, check=None):
        from possess.cleanup import cleanup

        counts = {}
        new = cleanup(lab, period, visible, touch, counts=counts, **kw)
        if check is not None:
            assert (new == check).all(), "clean-up on saved labels differs from the pipeline's own post-pass"
        a, b = stats(lab, period), stats(new, period)
        n_runs = int(((lab[1:] != lab[:-1]) & (np.asarray(period)[1:] > 0)).sum()) + 1
        hh = (np.asarray(period) > 0).sum() / FPS / 1800.0
        out = {"before " + k: v for k, v in a.items()}
        out.update({"after " + k: v for k, v in b.items()})
        out["possessions change"] = b["possessions / half hour"] / a["possessions / half hour"] - 1.0
        out["runs"] = n_runs
        for rule in ("flicker", "bridge", "island", "sliver"):
            out["share of runs relabelled: " + rule] = counts.get(rule, 0) / n_runs
            out["relabelled / half hour: " + rule] = counts.get(rule, 0) / hh
        # census of the runs each rule looks at, before any rule (same thresholds)
        from possess.cleanup import cleanup as cu
        for rule, key in (("flicker", "m_flicker"), ("bridge", "m_bridge"), ("island", "m_island"), ("sliver", "m_sliver")):
            if kw.get(key, 0) > 0:
                c1 = {}
                cu(lab, period, visible, touch, counts=c1, max_pass=1, **{key: kw[key]})
                out["eligible / half hour (rule alone, no protect): " + rule] = c1.get(rule, 0) / hh
        return out

    res = {}
    cache = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")
    f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"), cache)       # input caches only
    per = pipeline.challenge_period(len(f["ball_x"]))
    base = pipeline.label(f, dec_kw=dict(CL_OFF, **ED_OFF), period=per)
    feats = {"ball_x": f["ball_x"], "ball_y": f["ball_y"], "d_home": np.nan_to_num(f["d_home"], nan=99.0),
             "d_away": np.nan_to_num(f["d_away"], nan=99.0)}
    res["provided"] = one(base, per, ~np.isnan(f["ball_x"]), _touch(feats, per),
                          check=pipeline.label(f, dec_kw=dict(CL_OFF, **ED_OFF, **kw), period=per))
    for name, d in cl_load("real1"):
        res[name] = one(d["lab"], d["period"], d["visible"], d["touch"])
    names = list(res)
    print("clean-up %s" % kw)
    print("%-58s" % "aggregate" + "".join("%10s" % n for n in names))
    for k in res["provided"]:
        print("%-58s" % k + "".join("%10s" % ("%.3f" % res[n][k]) for n in names))
    ch = [res[n]["possessions change"] for n in names[1:]]
    lo, hi = min(ch), max(ch)
    pr = res["provided"]["possessions change"]
    a_lo, a_hi = (2 * lo if lo < 0 else 0.5 * lo), (0.5 * hi if hi < 0 else 2 * hi)
    print("possessions change: provided %+.4f, bench %+.4f..%+.4f; allowed %+.4f..%+.4f (0.5x..2x the bench range) and not below -0.10 -> %s" % (
        pr, lo, hi, a_lo, a_hi, "PASS" if a_lo <= pr <= a_hi and pr >= -0.10 else "FAIL"))
    with open(os.path.join(ROOT, "eval", "results", "transfer_cleanup.json"), "w") as fh:
        json.dump(dict(kw=kw, res=res), fh, indent=1, default=float)


def cmd_cleanup(args):
    """diagnose.py cleanup save|step0|grid|check|transfer [bench=real1] [noise=0.05] [drop=1] [m_flicker=0.8 ...]"""
    sub, variant, rest = args[0], "real1", []
    for it in args[1:]:
        if it.startswith("bench="):
            variant = "" if it[6:] == "orig" else it[6:]
        else:
            rest.append(it)
    kw = tune_shape.kv(rest)
    if sub == "save":
        tune_shape.VARIANT = variant
        return cl_save({k: kw[k] for k in ("noise", "drop") if k in kw})
    if sub == "step0":
        return cl_step0(variant)
    if sub == "grid":
        return cl_grid(variant, int(kw.get("cl_dir", 0)))
    if sub == "check":
        return cl_check(kw)
    if sub == "transfer":
        return cl_transfer(kw)
    raise SystemExit(cmd_cleanup.__doc__)


# ---- envelope L2-18: edges of stoppages inside hidden blocks (possess/edges.py) ----------------
ED_MIN_S = 3.0                                       # stoppages / D runs looked at (envelope)
ED_COLS = ("sdead", "p90", "ps90", "ps90_5", "ps15", "nfast")


def _ed_path(name, variant, deg):
    tag = "".join("_%s%g" % (k, v) for k, v in sorted(deg.items()) if v)
    return os.path.join(bench.CACHE, "edges_%s_%s%s_v1.npz" % (name, variant or "orig", tag))


def ed_save(deg, kw=None, tag=""):
    """One decode of the default (+ decoder overrides kw, saved under `tag`) per tune match;
    labels + truth + period + visible + the stoppage log-odds the decoder read + raw p90 / nfast
    (edges.edge_columns) + the smoothed speed columns go to edges_<match>_<bench>[_deg]_v1.npz."""
    from possess import edges

    rows, res = default_run(dict(ED_OFF, **(kw or {})), **dict(deg))
    print("saved", tune_shape.VARIANT or "orig", deg, kw or "", tune_shape.fmt(res))
    for name, case, feats, lab in rows:
        path = _ed_path(name, tune_shape.VARIANT, deg)
        if tag:                                          # another decode of the same input: labels only
            np.savez_compressed(path.replace("_v1.npz", "_%s_v1.npz" % tag), lab=lab)
            continue
        pos = tune_shape.positions(name, case)
        if deg.get("drop"):
            pos = tune_shape.drop_players(pos)
        if deg.get("noise"):
            pos = tune_shape.add_noise(pos, deg["noise"])
        cols = edges.edge_columns(pos)
        del pos
        sf = tune_shape.shape_features(name, bench.get_case(name), 0, deg.get("noise", 0.0), deg.get("drop", 0))
        np.savez_compressed(path, lab=lab, truth=case["truth"], period=case["period"], visible=feats["visible"],
                            sdead=feats["shape_dead"], p90=cols["p90"], nfast=cols["nfast"], ps90=sf["fast"][:, 0],
                            ps90_5=sf["fast"][:, 1], ps15=sf["dead"][:, shape.DEAD_FEATS.index("ps15")])


def ed_load(variant="real1", deg=None, tag=""):
    out = []
    for name in tune_shape.TUNE:
        z = np.load(_ed_path(name, variant, deg or {}))
        d = {k: z[k] for k in z.files}
        if tag:
            d["lab"] = np.load(_ed_path(name, variant, deg or {}).replace("_v1.npz", "_%s_v1.npz" % tag))["lab"]
        out.append((name, d))
    return out


def _d_runs(is_d, period, min_s=ED_MIN_S):
    """[(a, b)] runs of D inside one period, at least min_s long (b exclusive)."""
    per = np.asarray(period)
    m = np.concatenate([[False], np.asarray(is_d, bool) & (per > 0), [False]])
    d = np.diff(m.astype(np.int8))
    out = []
    for a, b in zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()):
        for c in [a] + (np.flatnonzero(per[a + 1:b] != per[a:b - 1]) + a + 1).tolist() + [b]:
            if c > a and c - a >= min_s * FPS:
                out.append((a, c))
            a = c
    return out


def ed_apply(d, kw, moves=None):
    from possess import edges

    return edges.apply(d["lab"], d["period"], d["visible"], d["p90"], moves=moves, **kw)


def ed_offsets(d, lab):
    """Per true stoppage >= 3 s with a live truth frame on that side: (kind, hidden?, offset s or
    None when no predicted D run overlaps it). Offset = predicted - true edge of the predicted D
    run with the largest overlap."""
    t, per, vis = d["truth"], d["period"], d["visible"]
    n = len(t)
    pr = _d_runs(lab == "D", per, 0.0)
    pa, pb = np.array([r[0] for r in pr]), np.array([r[1] for r in pr])
    out = []
    for a, b in _d_runs(t == "D", per):
        ov = np.minimum(pb, b) - np.maximum(pa, a) if len(pr) else np.array([0])
        j = int(np.argmax(ov)) if ov.max() > 0 else None
        if a > 0 and per[a - 1] == per[a] and t[a - 1] in ("H", "A"):
            out.append(("entry", not vis[a], None if j is None else (pa[j] - a) / FPS, a))
        if b < n and per[b] == per[a] and t[b] in ("H", "A"):
            out.append(("exit", not vis[b], None if j is None else (pb[j] - b) / FPS, b))
    return out


def ed_stats(d, lab):
    """{(kind, 'hidden' | 'visible'): dict(n, med, q25, q75, in05, unmatched)}, hidden D recall / precision."""
    out = {}
    off = ed_offsets(d, lab)
    for kind in ("entry", "exit"):
        for hid in (True, False):
            v = [o for k, h, o, _ in off if k == kind and h == hid]
            x = np.array([o for o in v if o is not None])
            q = np.percentile(x, [25, 50, 75]) if len(x) else [np.nan] * 3
            out[(kind, "hidden" if hid else "visible")] = dict(
                n=len(v), med=float(q[1]), q25=float(q[0]), q75=float(q[2]),
                in05=float((np.abs(x) <= 0.5).sum() / max(1, len(v))), unmatched=float(1 - len(x) / max(1, len(v))))
    ok = np.isin(d["truth"], ["H", "A", "D"]) & ~d["visible"]
    td, pd_ = (d["truth"] == "D") & ok, (lab == "D") & ok
    out["hidden D recall"] = float((td & pd_).sum() / max(1, td.sum()))
    out["hidden D precision"] = float((td & pd_).sum() / max(1, pd_.sum()))
    return out


def _sf1(d, lab):
    r = pm.evaluate(pm.label_codes(d["truth"]), pm.label_codes(lab), bootstrap_n=0)
    return r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]


def ed_table(variant, kw=None, data=None, quiet=False):
    """D0a: the edge table per match (+ after `kw` when given) and the hidden-edge oracle."""
    data = data or ed_load(variant)
    res = {}
    for name, d in data:
        labs = {"default": d["lab"]}
        if kw:
            labs["config"] = ed_apply(d, kw[name] if name in kw else kw)
        for tag, lab in labs.items():
            st = ed_stats(d, lab)
            res[(name, tag)] = st
            if quiet:
                continue
            print("%-9s %-7s spellF1 %.4f  hidden D recall %.3f precision %.3f" % (
                name, tag, _sf1(d, lab)[0], st["hidden D recall"], st["hidden D precision"]))
            for key in (("entry", "hidden"), ("exit", "hidden"), ("entry", "visible"), ("exit", "visible")):
                s = st[key]
                print("    %-5s %-7s n %3d  offset pred - true: median %+.2f s  IQR %+.2f..%+.2f  within +-0.5 s %.2f  unmatched %.2f" % (
                    key + (s["n"], s["med"], s["q25"], s["q75"], s["in05"], s["unmatched"])))
    return res


def ed_oracle(variant):
    """Hidden-edge oracle (as in L2-16): hidden true-D frames of stoppages >= 3 s set to D in the
    default's labels, by zone: first 5 s, last 5 s, the middle; 'edge' = first + last;
    'edge2' = every hidden frame within +-5 s of a true edge of those stoppages set to the truth."""
    tot = {}
    for name, d in ed_load(variant):
        t, lab, hid = d["truth"], d["lab"], ~d["visible"]
        z = {k: np.zeros(len(t), bool) for k in ("first5", "last5", "middle", "edge2")}
        w = int(5 * FPS)
        for a, b in _d_runs(t == "D", d["period"]):
            z["first5"][a:min(b, a + w)] = True
            z["last5"][max(a, b - w):b] = True
            z["middle"][a + w:b - w] = True
            for e in (a, b):
                z["edge2"][max(0, e - w):e + w] = True
        z["edge"] = z["first5"] | z["last5"]
        z["edge2"] &= np.isin(t, ["H", "A", "D"]) & (d["period"] > 0)
        base = _sf1(d, lab)[0]
        line = "%-9s default %.4f" % (name, base)
        for k in ("first5", "last5", "middle", "edge", "edge2"):
            new = lab.copy()
            m = z[k] & hid if k == "edge2" else z[k] & hid & (t == "D")
            new[m] = t[m]
            g = _sf1(d, new)[0] - base
            tot.setdefault(k, []).append(g)
            line += "  %s %+.4f" % (k, g)
        print(line)
    print("mean      " + "  ".join("%s %+.4f" % (k, np.mean(v)) for k, v in tot.items()))
    return {k: float(np.mean(v)) for k, v in tot.items()}


def ed_profiles(variant, half_s=8.0):
    """D0b: mean of the columns around true HIDDEN exits / entries of stoppages >= 3 s, per match.
    Zero crossing of the mean stoppage log-odds nearest the edge, and the half-rise time of raw
    p90 (midpoint between its minimum within 4 s inside the stoppage and its mean 3-6 s outside)."""
    import json

    h = int(half_s * FPS)
    tt = np.arange(-h, h + 1) / FPS
    out = {}
    for name, d in ed_load(variant):
        per = d["period"]
        for kind in ("exit", "entry"):
            ev = [e for k, hid, _, e in ed_offsets(d, d["lab"]) if k == kind and hid
                  and e - h >= 0 and e + h < len(per) and per[e - h] == per[e] == per[e + h]]
            prof = {}
            with shape._quiet():
                for c in ED_COLS:
                    prof[c] = np.nanmean(np.stack([d[c][e - h:e + h + 1] for e in ev]).astype(float), axis=0)
            sg = 1 if kind == "exit" else -1             # time axis pointing out of the stoppage
            u = sg * tt
            lo_m = (u >= -4) & (u <= 0)
            res = {}
            for c in ("p90", "ps90", "nfast"):
                p = shape.smooth(prof[c], 5)
                i0 = np.flatnonzero(lo_m)[np.argmin(p[lo_m])]
                mid = 0.5 * (p[i0] + p[(u >= 3) & (u <= 6)].mean())
                idx = np.arange(i0, len(p)) if sg > 0 else np.arange(i0, -1, -1)
                hit = idx[p[idx] >= mid]
                res[c + " half-rise s"] = float(tt[hit[0]]) if len(hit) else None
            s = prof["sdead"]
            cr = np.flatnonzero((s[:-1] > 0) != (s[1:] > 0))
            res["log-odds zero crossing s"] = float(tt[cr[np.argmin(np.abs(tt[cr]))]]) if len(cr) else None
            res["n"] = len(ev)
            grid = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
            res["profile"] = {c: [round(float(prof[c][h + int(g * FPS)]), 2) for g in grid] for c in ED_COLS}
            out["%s %s" % (name, kind)] = res
            print("%-9s true hidden %-5s n %3d  log-odds zero crossing %s s | half-rise: raw p90 %s s, ps90 (1 s mean) %s s, n > 2 m/s %s s" % (
                name, kind, len(ev), res["log-odds zero crossing s"], res["p90 half-rise s"], res["ps90 half-rise s"],
                res["nfast half-rise s"]))
            print("      t (s)     " + " ".join("%6g" % g for g in grid))
            for c in ED_COLS:
                print("      %-9s " % c + " ".join("%6.2f" % v for v in res["profile"][c]))
    with open(os.path.join(ROOT, "eval", "results", "edges_profiles_%s.json" % (variant or "orig")), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def ed_census(lab, period, visible):
    """Input-only aggregates of one match's labels (D0d transfer baseline)."""
    play = np.asarray(period) > 0
    hh = play.sum() / FPS / 1800.0
    r = _runs((lab == "D") & play)
    long_ = _d_runs(lab == "D", period)
    n = len(lab)
    hid_edges = [not visible[a] for a, b in long_] + [not visible[min(b, n - 1)] for a, b in long_]
    return {"D runs >= 3 s / half hour": len(long_) / hh, "share of their edges on hidden frames": float(np.mean(hid_edges)) if long_ else 0.0,
            "hidden D share": float((lab == "D")[play & ~visible].mean()), "D share": float((lab == "D")[play].mean()),
            "possessions / half hour": (len(_runs((lab == "H") & play)) + len(_runs((lab == "A") & play))) / hh,
            "D runs / half hour": len(r) / hh, "D run median s": float(np.median(r) / FPS) if len(r) else 0.0}


def ed_transfer(kw=None):
    """D0d / transfer gate: aggregates of the default (and of `kw` on top) on the provided match's
    INPUT (cached features + positions; the ground truth is never opened) next to the tune
    matches on real1. Nothing is chosen here. Writes eval/results/transfer_edges.json."""
    import json

    from possess import edges, pipeline

    res = {}
    cache = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")
    f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"), cache)       # input caches only
    per = pipeline.challenge_period(len(f["ball_x"]))
    vis = ~np.isnan(f["ball_x"])
    base = pipeline.label(f, dec_kw=dict(ED_OFF), period=per)
    if kw:                                               # the post-pass on saved labels = the pipeline's own
        assert (edges.apply(base, per, vis, edges.edge_columns(f["pos"])["p90"], **kw) ==
                pipeline.label(f, dec_kw=dict(ED_OFF, **kw), period=per)).all()
    items = [("provided", base, per, vis, edges.edge_columns(f["pos"])["p90"] if kw else None)]
    del f
    items += [(name, d["lab"], d["period"], d["visible"], d["p90"]) for name, d in ed_load("real1")]
    for name, lab, period, visible, p90 in items:
        out = {"before " + k: v for k, v in ed_census(lab, period, visible).items()}
        if kw:
            moves = []
            new = edges.apply(lab, period, visible, p90, moves=moves, **kw)
            out.update({"after " + k: v for k, v in ed_census(new, period, visible).items()})
            n_edges = 2 * len(_d_runs(lab == "D", period, 0.0))
            out["share of D edges moved"] = len(moves) / max(1, n_edges)
            for kind in ("exit", "entry"):
                mv = [(e - o) / FPS for k, o, e in moves if k == kind]
                out["mean signed move of %s s" % ("exits" if kind == "exit" else "entries")] = float(np.mean(mv)) if mv else 0.0
                out["%s moved" % kind] = len(mv)
            out["hidden D share change"] = out["after hidden D share"] - out["before hidden D share"]
            out["possessions change"] = out["after possessions / half hour"] / out["before possessions / half hour"] - 1.0
        res[name] = out
    names = list(res)
    print("edges %s" % (kw or "(default only)"))
    print("%-44s" % "aggregate" + "".join("%10s" % n for n in names))
    for k in res["provided"]:
        print("%-44s" % k + "".join("%10s" % ("%.3f" % res[n][k]) for n in names))
    with open(os.path.join(ROOT, "eval", "results", "transfer_edges.json"), "w") as fh:
        json.dump(dict(kw=kw, res=res), fh, indent=1, default=float)


def ed_lomo_shift(data, kind="exit"):
    """M2 scalar per match: minus the median hidden offset (s) of `kind` on the OTHER tune
    matches' default labels (the scored match never enters), rounded to a frame."""
    off = {name: [o for k, h, o, _ in ed_offsets(d, d["lab"]) if k == kind and h and o is not None] for name, d in data}
    return {name: -round(float(np.median(sum((v for m, v in off.items() if m != name), []))) * FPS) / FPS for name in off}


def ed_row(tag, kw, data, other=None, labs=None):
    """One grid row: per-match spellF1 / counts and the hidden edge statistics. kw: edges.apply
    parameters, or {match: parameters} (leave-one-match-out scalars); labs: {match: labels} of
    another decode instead (M4 / M6)."""
    def lab_of(name, d, k):
        return ed_apply(d, k[name] if name in k else k)

    res = [(name,) + _sf1(d, labs[name] if labs else lab_of(name, d, kw)) for name, d in data]
    st = [ed_stats(d, labs[name] if labs else lab_of(name, d, kw)) for name, d in data]
    row = dict(tag=tag, kw=kw, spellF1=float(np.mean([r[1] for r in res])), per_match=[round(r[1], 4) for r in res],
               n_pred=[r[2] for r in res], n_gt=[r[3] for r in res],
               exit_med=[round(s[("exit", "hidden")]["med"], 2) for s in st],
               exit_in05=[round(s[("exit", "hidden")]["in05"], 2) for s in st],
               entry_med=[round(s[("entry", "hidden")]["med"], 2) for s in st],
               entry_in05=[round(s[("entry", "hidden")]["in05"], 2) for s in st],
               hid_D_recall=[round(s["hidden D recall"], 3) for s in st],
               hid_D_precision=[round(s["hidden D precision"], 3) for s in st])
    if other is not None and not labs:
        okw = kw if other[1] is None else other[1]       # leave-one-match-out scalars of the other bench
        row["other_bench"] = float(np.mean([_sf1(d, lab_of(name, d, okw))[0] for name, d in other[0]]))
    print("%-4s %-62s %.4f (%s) | exit med %s in05 %s | entry med %s in05 %s | hid D R %s P %s%s" % (
        tag, kw if not labs else kw, row["spellF1"], " ".join("%.3f %d" % (r[1], r[2]) for r in res),
        row["exit_med"], row["exit_in05"], row["entry_med"], row["entry_in05"], row["hid_D_recall"], row["hid_D_precision"],
        " | other bench %.4f" % row["other_bench"] if "other_bench" in row else ""), flush=True)
    return row


def ed_grid(variant="real1"):
    """The fixed grid of the envelope on the saved default labels (M1, M2: post-passes; M6: the
    saved control decodes). Writes eval/results/tune_edges.json."""
    import json

    data = ed_load(variant)
    oth = ed_load("" if variant else "real1")
    rows = []

    def add(tag, kw, lomo_other=None):
        rows.append(ed_row(tag, kw, data, (oth, lomo_other)))
        return rows[-1]

    def pick(got, base):
        """Smallest change inside a 0.002 plateau (rows are listed smallest first); None if it does not beat off."""
        top = max(r["spellF1"] for r in got)
        p = next(r for r in got if r["spellF1"] >= top - 0.002)
        return p if p["spellF1"] > base else None

    off = add("off", {})
    for t, c in (("m6a", 8.0), ("m6b", 16.0)):
        try:
            rows.append(ed_row("M6", dict(c_out_hid=c), data, labs={n: d["lab"] for n, d in ed_load(variant, tag=t)}))
        except FileNotFoundError:
            print("M6 %s: labels not saved" % t)
    # M1 exits: w 1 / 2 x win 2 / 4, theta 0
    got = [add("M1", dict(ed_exit=1, ed_w=w, ed_win=win)) for win in (2.0, 4.0) for w in (1.0, 2.0)]
    bad = [sum(abs(r["exit_med"][i]) >= abs(off["exit_med"][i]) or r["exit_in05"][i] <= off["exit_in05"][i] for i in range(3))
           for r in got if r["kw"]["ed_win"] == 2.0]
    print("   M1 first diagnostic (win 2 s; w 1, 2): matches where the hidden exit median does not move toward 0 or the "
          "share within +-0.5 s does not rise: %s -> %s" % (bad, "STOP M1" if min(bad) >= 2 else "continue"))
    best = max(got, key=lambda r: r["spellF1"])
    bk = best["kw"]
    add("M1", dict(bk, ed_theta=0.5))
    for win in (2.0, 4.0):
        add("M1", dict(ed_entry=1, ed_w=bk["ed_w"], ed_win=win))
    add("M1", dict(bk, ed_entry=1))
    add("M1", dict(bk, ed_vis=1))
    # M2 exits (leave-one-match-out scalar); the entry shift is stopped by its rule (sign differs)
    sx, sx_o = ed_lomo_shift(data), ed_lomo_shift(oth)
    se = ed_lomo_shift(data, "entry")
    print("   M2 scalars (s, leave-one-match-out): exit %s, entry %s (other bench exit %s)" % (sx, se, sx_o))
    add("M2", {n: dict(ed_dx=v) for n, v in sx.items()}, {n: dict(ed_dx=v) for n, v in sx_o.items()})
    add("M2", {n: dict(ed_dx=round(v * FPS / 2) / FPS) for n, v in sx.items()},
        {n: dict(ed_dx=round(v * FPS / 2) / FPS) for n, v in sx_o.items()})
    p = pick(got, off["spellF1"])
    print("   M1 exit plateau pick: %s" % (p["kw"] if p else "none beats off"))
    for k, name in enumerate(tune_shape.TUNE):          # gate 5: picked on two matches, read on the third
        two = [i for i in range(3) if i != k]
        pk = max(rows, key=lambda r: np.mean([r["per_match"][i] for i in two]))
        print("gate 5: picked without %-9s -> %s %s: there %.4f vs off %.4f" % (name, pk["tag"], pk["kw"], pk["per_match"][k], off["per_match"][k]))
    with open(os.path.join(ROOT, "eval", "results", "tune_edges.json"), "w") as fh:
        json.dump(dict(protocol="envelope L2-18 fixed grid on saved default labels (diagnose.py edges save); M2 scalars leave-one-match-out",
                       bench=variant or "orig", rows=rows), fh, indent=1, default=float)


def _ed_append(rows, variant):
    import json

    path = os.path.join(ROOT, "eval", "results", "tune_edges.json")
    doc = json.load(open(path))
    assert doc["bench"] == (variant or "orig")
    doc["rows"] = [r for r in doc["rows"] if all(r["tag"] != n["tag"] or r["kw"] != n["kw"] for n in rows)] + rows
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, default=float)


def ed_cues(saved):
    """{match: (cue_up, cue_dn)} for M4: positive step up / down of raw p90 (edges.step_score, 1 s)
    in SD units, clipped to [0, 3]; the SD comes from the OTHER tune matches (leave-one-match-out)."""
    from possess import edges

    s = {name: edges.step_score(d["p90"].astype(float), d["period"], int(FPS)) for name, d in saved}
    out = {}
    for name in s:
        sd = float(np.nanstd(np.concatenate([v for m, v in s.items() if m != name])))
        z = np.nan_to_num(s[name] / sd)
        out[name] = (np.clip(z, 0, 3), np.clip(-z, 0, 3), sd)
    return out


def ed_m4(variant, cfgs):
    """M4: the step cue as a per-frame transition cost in the decoder (decode.viterbi w_r / w_s).
    One decode of the three tune matches per config; rows appended to tune_edges.json."""
    tune_shape.VARIANT = variant
    data = tune_shape.load_all()
    saved = ed_load(variant)
    cues = ed_cues(saved)
    print("   cue SD (m/s, leave-one-match-out): %s" % {n: round(c[2], 3) for n, c in cues.items()})
    data_e = [(n, c, dict(f, cue_up=cues[n][0], cue_dn=cues[n][1]))
              for n, c, f in tune_shape.with_emissions(data, tune_shape.lomo_models(data))]
    rows = []
    for kw in cfgs:
        res = tune_shape.run(data_e, dict(shape.DEC_SHAPE, **dict(ED_OFF, **kw)))
        labs = {r[0]: r[4] for r in res}
        rows.append(ed_row("M4", kw, saved, labs=labs))
        for n, lab in labs.items():
            np.savez_compressed(_ed_path(n, variant, {}).replace("_v1.npz", "_m4_%s_v1.npz" % "_".join("%s%g" % i for i in sorted(kw.items()))), lab=lab)
    _ed_append(rows, variant)


def ed_check(kw):
    """Gates for one post-pass config, all from saved labels: both benches, paired bootstrap over
    true spells (real1), edges moved, noise gate (labels saved with noise= / drop=)."""
    for variant in ("real1", ""):
        data = ed_load(variant)
        b = [(n,) + _sf1(d, d["lab"]) for n, d in data]
        labs = {n: ed_apply(d, kw) for n, d in data}
        c = [(n,) + _sf1(d, labs[n]) for n, d in data]
        print("%-5s off %s\n      on  %s" % (variant or "orig", tune_shape.fmt(b), tune_shape.fmt(c)))
        if variant:
            diffs = []
            for name, d in data:
                x, y = _cl_boot(d, d["lab"], labs[name])
                diffs.append(y - x)
                moves = []
                ed_apply(d, kw, moves)
                mv = {k: [(e - o) / FPS for kk, o, e in moves if kk == k] for k in ("exit", "entry")}
                print("      %-9s paired bootstrap gain %+.4f CI90 [%+.4f, %+.4f] | moved exits %d (mean %+.2f s) entries %d (mean %+.2f s), frames changed %.4f" % (
                    (name, np.mean(y - x)) + tuple(np.percentile(y - x, [5, 95])) + (
                        len(mv["exit"]), np.mean(mv["exit"]) if mv["exit"] else 0, len(mv["entry"]),
                        np.mean(mv["entry"]) if mv["entry"] else 0, (labs[name] != d["lab"]).mean())))
            m = np.mean(diffs, axis=0)
            print("      mean of 3 paired bootstrap gain %+.4f CI90 [%+.4f, %+.4f]  P(gain <= 0) %.3f" % (
                (m.mean(),) + tuple(np.percentile(m, [5, 95])) + ((m <= 0).mean(),)))
    for deg in (dict(noise=0.05), dict(noise=0.1), dict(drop=1)):
        try:
            data = ed_load("real1", deg)
        except FileNotFoundError:
            print("noise gate %s: labels not saved" % deg)
            continue
        print("noise gate real1 %-16s off %s\n%34s on  %s" % (
            deg, tune_shape.fmt([(n,) + _sf1(d, d["lab"]) for n, d in data]), "",
            tune_shape.fmt([(n,) + _sf1(d, ed_apply(d, kw)) for n, d in data])))


def ed_m3(variant="real1"):
    """M3 first diagnostic (no decode): leave-one-match-out stoppage logistic with one-sided 3 s
    means of raw p90 against dead_fast, judged by the AUC on HIDDEN frames within +-3 s of the
    true edges of stoppages >= 3 s (where T40, judged on all frames, could not see a gain)."""
    from possess import edges

    data = ed_load(variant)
    w = int(3 * FPS)
    X, near, y, fit = {}, {}, {}, {}
    for name, d in data:
        sf = tune_shape.shape_features(name, bench.get_case(name))
        trail, lead = edges.side_means(d["p90"].astype(float), d["period"], w)
        base = np.hstack([sf["dead"], sf["fast"]])
        i5 = base.shape[1] - 1                           # ps90_5
        ext = np.stack([trail, lead], axis=1)
        inter = (sf["fast"][:, 0] * np.abs(lead - trail))[:, None]
        X[name] = {"dead_fast": base, "one-sided": np.hstack([base, ext]), "+ interaction": np.hstack([base, ext, inter]),
                   "one-sided replacing ps90_5": np.hstack([np.delete(base, i5, axis=1), ext])}
        t = d["truth"]
        z = np.zeros(len(t), bool)
        for k, hid, _, e in ed_offsets(d, d["lab"]):
            z[max(0, e - w):e + w] = True
        near[name] = z & ~d["visible"] & np.isin(t, ["H", "A", "D"]) & sf["ok"]
        y[name] = t == "D"
        fit[name] = sf["ok"] & np.isin(t, ["H", "A", "D"]) & (np.arange(len(t)) % 5 == 0)
    for col in X[data[0][0]]:
        line = "%-28s" % col
        for name, _ in data:
            tr = [m for m, _ in data if m != name]
            ok = {m: fit[m] & np.isfinite(X[m][col]).all(axis=1) for m in tr}
            mdl = shape.fit_logistic(np.vstack([X[m][col][ok[m]] for m in tr]), np.concatenate([y[m][ok[m]] for m in tr]).astype(float), tune_shape.L2)
            m = near[name] & np.isfinite(X[name][col]).all(axis=1) & np.isfinite(X[name]["+ interaction"]).all(axis=1)
            hid_all = ~dict(data)[name]["visible"] & fit[name] & np.isfinite(X[name]["+ interaction"]).all(axis=1)
            z = shape.log_odds(mdl, X[name][col])
            line += "  %s edge AUC %.4f (n %d) all-hidden %.4f" % (name, shape.auc(z[m], y[name][m]), m.sum(), shape.auc(z[hid_all], y[name][hid_all]))
        print(line, flush=True)


def ed_taker(variant="real1"):
    """D0c: prevalence of the M5 'lone taker' cue (edges.taker_frames). Bench: share of true hidden
    exits of stoppages >= 3 s with a taker in the 3 s before the exit, share of true live frames
    with a taker. Provided match (INPUT only, no truth): share of hidden / visible frames with one.
    The column is cached per bench match as edgetaker_<match>_v1.npz."""
    from possess import edges, pipeline

    w = int(3 * FPS)
    for name, d in ed_load(variant):
        path = os.path.join(bench.CACHE, "edgetaker_%s_v1.npz" % name)
        if os.path.exists(path):
            tk = np.load(path)["taker"]
        else:
            tk = edges.taker_frames(tune_shape.positions(name, bench.get_case(name)))
            np.savez_compressed(path, taker=tk)
        t, vis, play = d["truth"], d["visible"], d["period"] > 0
        off = ed_offsets(d, d["lab"])
        ex = [e for k, hid, _, e in off if k == "exit" and hid]
        exv = [e for k, hid, _, e in off if k == "exit" and not hid]
        live = np.isin(t, ["H", "A"]) & play
        print("%-9s taker in the 3 s before a true hidden exit %.2f (n %d), visible exit %.2f (n %d) | taker on true live frames %.3f, "
              "true D frames %.3f | all frames hidden %.3f visible %.3f" % (
                  name, np.mean([tk[max(0, e - w):e].any() for e in ex]), len(ex),
                  np.mean([tk[max(0, e - w):e].any() for e in exv]), len(exv), tk[live].mean(),
                  tk[(t == "D") & play].mean(), tk[play & ~vis].mean(), tk[play & vis].mean()))
    cache = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")
    f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"), cache)       # input caches only
    per = pipeline.challenge_period(len(f["ball_x"]))
    play, vis = per > 0, ~np.isnan(f["ball_x"])
    tk = edges.taker_frames(f["pos"])
    print("provided  (input only) taker on hidden frames %.3f, visible frames %.3f, all %.3f" % (
        tk[play & ~vis].mean(), tk[play & vis].mean(), tk[play].mean()))


def cmd_edges(args):
    """diagnose.py edges save|d0|profiles|transfer [bench=real1] [noise=0.05] [drop=1] [ed_exit=1 ...]"""
    sub, variant, rest, tag = args[0], "real1", [], ""
    for it in args[1:]:
        if it.startswith("bench="):
            variant = "" if it[6:] == "orig" else it[6:]
        elif it.startswith("tag="):
            tag = it[4:]
        else:
            rest.append(it)
    kw = tune_shape.kv(rest)
    if sub == "save":                                    # save [tag=m6a c_out_hid=8.0]: decoder overrides, labels only
        tune_shape.VARIANT = variant
        deg = {k: kw.pop(k) for k in ("noise", "drop") if k in kw}
        return ed_save(deg, kw, tag)
    if sub == "d0":
        if tag:                                          # edge table of another saved decode
            return ed_table(variant, data=ed_load(variant, tag=tag))
        ed_table(variant, kw or None)
        return ed_oracle(variant)
    if sub == "profiles":
        return ed_profiles(variant)
    if sub == "grid":
        return ed_grid(variant)
    if sub == "m3":
        return ed_m3(variant)
    if sub == "check":
        return ed_check(kw)
    if sub == "dhide":                                   # gate 7: dhide 0.8 / 0.95 with the post-pass on top
        from possess import edges

        p90 = {n: d["p90"] for n, d in ed_load(variant)}
        return cmd_dhide((0.8, 0.95), variant, post=lambda name, lab, per, vis: edges.apply(lab, per, vis, p90[name], **kw))
    if sub == "row":                                     # one more post-pass config on the saved labels -> tune_edges.json
        if kw.pop("lomo_dx", 0):                         # + the leave-one-match-out exit shift of M2
            sx, so = ed_lomo_shift(ed_load(variant)), ed_lomo_shift(ed_load("" if variant else "real1"))
            row = ed_row("C", {n: dict(kw, ed_dx=v) for n, v in sx.items()}, ed_load(variant),
                         (ed_load("" if variant else "real1"), {n: dict(kw, ed_dx=v) for n, v in so.items()}))
        else:
            row = ed_row("C", kw, ed_load(variant), (ed_load("" if variant else "real1"), None))
        return _ed_append([row], variant)
    if sub == "m4":                                      # m4 w_r=2.0 | m4 w_s=4.0 | m4 w_r=4.0 w_s=4.0
        return ed_m4(variant, [kw] if kw else [dict(w_r=2.0), dict(w_r=4.0), dict(w_r=8.0), dict(w_s=4.0)])
    if sub == "taker":
        return ed_taker(variant)
    if sub == "transfer":
        return ed_transfer(kw or None)
    raise SystemExit(cmd_edges.__doc__)


# ----------------------------------------------------------------------------- hidden-frame prior (L2-21)
def _pr_runs(m):
    return int((np.diff(np.concatenate([[0], np.asarray(m, np.int8), [0]])) == 1).sum())


def _pr_shift(sd, k, pi_tr):
    pi = shape.em_prior(sd[k], pi_tr)
    return pi, float(np.log(pi / (1 - pi)) - np.log(pi_tr / (1 - pi_tr)))


def _pr_boot(truth, base, new, n_draws=2000, seed=0, block=3000):
    """Paired bootstrap over 2-min blocks of TRUE possessions (by midpoint; predictions attached as
    in _cl_boot). Returns per-draw mean-F1 of (base, new)."""
    g = pm.label_codes(truth)
    runs = pm.valid_runs(g)
    gs = pm.possession_spells(g, runs)
    ga, gb = np.array([s[0] for s in gs]), np.array([s[1] for s in gs])
    blk = ((ga + gb) // 2) // block
    ub = np.unique(blk)
    picks = np.random.default_rng(seed).integers(0, len(ub), (n_draws, len(ub)))
    wts = np.zeros((n_draws, len(ub)))
    for d in range(n_draws):
        np.add.at(wts[d], picks[d], 1.0)
    w = wts @ (blk[None, :] == ub[:, None]).astype(float)        # [draws, true spells] multiplicity
    out = []
    for lab in (base, new):
        ps = pm.possession_spells(pm.fill_abstentions(pm.label_codes(lab), runs), runs)
        cnt = np.zeros(len(gs))
        for a, b, _ in ps:
            ov = np.minimum(gb, b) - np.maximum(ga, a)
            cnt[np.argmax(ov) if ov.max() > 0 else np.argmin(np.abs((ga + gb) / 2 - (a + b) / 2))] += 1
        f1 = np.zeros(n_draws)
        for thr in pm.SPELL_IOUS:
            f1 += 2.0 * (w @ matched_flags(gs, ps, thr)[0].astype(float)) / (w @ cnt + w.sum(axis=1))
        out.append(f1 / len(pm.SPELL_IOUS))
    return out


def _pr_case(name, variant, seed=0):
    """(case, features.compute output with "pos") of a tune match; seed > 0: real2 built in memory."""
    if seed == 0:
        case = bench.get_case(name, variant=variant)
        return case, dict(bench.case_features(name, case, variant=variant), pos=tune_shape.positions(name, case, variant))
    from possess import features as feat_mod

    truth, present, period, vx, vy = bench.base_ball(name)
    case, pos = bench.real2_case(name, truth, present, period, vx, vy, seed)
    f = feat_mod.compute(bench.load_match(name).iter_frames(), n=len(truth), ball_x=case["ball_x"], ball_y=case["ball_y"])
    return case, dict(f, pos=pos)


def cmd_prior(args):
    """Envelope L2-21 (shape.prior_shift, DEC_SHAPE sd_prior=1): the share of D among the hidden frames.
    prior check [bench=real2] [seed=1] [names=metrica1,...] [noise=0.1] [drop=1]   off vs on per match:
        score, hidden D share true / predicted, estimated share and shift, paired 2-min block bootstrap
    prior null       estimated shift under RANDOM hiding, 20 hide masks per tune match (no decode)
    prior transfer   provided-match INPUT aggregates off / on -> eval/results/transfer_sd_prior.json"""
    import json

    from possess import pipeline

    cmd = args[0] if args else "check"
    opt = dict(a.split("=") for a in args[1:])
    clean = {n: tune_shape.shape_features(n, bench.get_case(n)) for n in tune_shape.TUNE}
    truths = {n: bench.get_case(n)["truth"] for n in tune_shape.TUNE}
    lomo = {n: shape.fit_models([clean[m] for m in clean if m != n], [truths[m] for m in clean if m != n], l2=tune_shape.L2)
            for n in tune_shape.TUNE}
    if cmd == "null":
        for name in tune_shape.TUNE:
            per, sf = bench.get_case(name)["period"], clean[name]
            sd, pt = shape.emissions(sf, lomo[name])[1], shape.dead_prior(lomo[name])
            v = [_pr_shift(sd, bench.hide_mask(len(per), s) & (per > 0) & sf["ok"], pt)[1] for s in range(20)]
            print("%s estimated shift, 20 random masks: min %+.2f max %+.2f mean %+.2f sd %.2f" % (name, min(v), max(v), np.mean(v), np.std(v)))
    elif cmd == "transfer":
        cache = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")
        f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"), cache)   # input caches only
        per = pipeline.challenge_period(len(f["ball_x"]))
        play = per > 0
        hid = play & np.isnan(f["ball_x"])
        sf = shape.features(f["pos"], per)
        pi, sh = _pr_shift(shape.emissions(sf, shape.MODELS)[1], hid & sf["ok"], shape.dead_prior(shape.MODELS))
        res = {"estimated hidden D share": pi, "shift": sh, "hidden frames without shape evidence": float((~sf["ok"])[hid].mean())}
        labs = {"off": pipeline.label(f, dec_kw=dict(sd_prior=0), period=per), "on": pipeline.label(f, period=per)}
        hh = play.sum() / FPS / 1800.0
        for t, lab in labs.items():
            res[t + " D share hidden"] = float((lab == "D")[hid].mean())
            res[t + " D share visible"] = float((lab == "D")[play & ~hid].mean())
            res[t + " possessions / half hour"] = (_pr_runs((lab == "H") & play) + _pr_runs((lab == "A") & play)) / hh
            res[t + " D runs / half hour"] = _pr_runs((lab == "D") & play) / hh
        res["frames changed"] = float((labs["off"] != labs["on"])[play].mean())
        for k, v in res.items():
            print("%-40s %.4f" % (k, v))
        with open(os.path.join(ROOT, "eval", "results", "transfer_sd_prior.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    else:
        variant, seed = opt.get("bench", "real2"), int(opt.get("seed", 0))
        variant = "" if variant == "orig" else variant
        diffs, so, sn = [], [], []
        for name in opt.get("names", ",".join(tune_shape.TUNE)).split(","):
            case, f = _pr_case(name, variant, seed)
            if int(opt.get("drop", 0)):
                f["pos"] = tune_shape.drop_players(f["pos"])
            if float(opt.get("noise", 0)):
                f["pos"] = tune_shape.add_noise(f["pos"], float(opt["noise"]))
            t, per = case["truth"], case["period"]
            hid = np.isnan(f["ball_x"]) & (per > 0)
            sf = shape.features(f["pos"], per)
            pi, sh = _pr_shift(shape.emissions(sf, lomo[name])[1], hid & sf["ok"], shape.dead_prior(lomo[name]))
            off = pipeline.label(f, dec_kw=dict(sd_prior=0), period=per, shape_models=lomo[name])
            on = pipeline.label(f, period=per, shape_models=lomo[name])
            ra, rb = (pm.evaluate(pm.label_codes(t), pm.label_codes(x), bootstrap_n=0) for x in (off, on))
            hs = hid & np.isin(t, ["H", "A", "D"])
            td = t == "D"
            print("%s off %.4f %d/%d on %.4f %d/%d | hidden D true %.3f est %.3f (shift %+.2f) pred %.3f -> %.3f recall %.3f -> %.3f "
                  "precision %.3f -> %.3f | no-evidence share %.3f" % (
                      name, ra["spell_f1_mean"], ra["n_pred_spells"], ra["n_gt_spells"], rb["spell_f1_mean"], rb["n_pred_spells"],
                      rb["n_gt_spells"], td[hs].mean(), pi, sh, (off == "D")[hs].mean(), (on == "D")[hs].mean(),
                      ((off == "D") & td & hs).sum() / td[hs].sum(), ((on == "D") & td & hs).sum() / td[hs].sum(),
                      ((off == "D") & td & hs).sum() / max(1, ((off == "D") & hs).sum()),
                      ((on == "D") & td & hs).sum() / max(1, ((on == "D") & hs).sum()), (~sf["ok"])[hid].mean()), flush=True)
            bo, bn = _pr_boot(t, off, on)
            diffs.append(bn - bo); so.append(ra["spell_f1_mean"]); sn.append(rb["spell_f1_mean"])
        d = np.mean(diffs, axis=0)
        print("mean off %.4f on %.4f | paired 2-min block bootstrap %+.4f CI90 [%+.4f, %+.4f], draws <= 0: %d of %d" % (
            np.mean(so), np.mean(sn), d.mean(), np.percentile(d, 5), np.percentile(d, 95), (d <= 0).sum(), len(d)))


# ----------------------------------------------------------------------------- hidden-block team carry (L2-22)
CY_BINS = [1, 5, 10, 20, 40, 1e9]                   # hidden-block length bins (s)


def _cy_path(name, variant, seed, tag=""):
    return os.path.join(bench.CACHE, "carry_%s_%s_s%d%s_v1.npz" % (name, variant or "orig", seed, tag))


def cy_save(variant, seed, kw=None, tag=""):
    """One default decode per tune match (leave-one-match-out shape models) -> carry_*.npz."""
    from possess import pipeline

    clean = {n: tune_shape.shape_features(n, bench.get_case(n)) for n in tune_shape.TUNE}
    truths = {n: bench.get_case(n)["truth"] for n in tune_shape.TUNE}
    for name in tune_shape.TUNE:
        models = shape.fit_models([clean[m] for m in clean if m != name], [truths[m] for m in clean if m != name], l2=tune_shape.L2)
        case, f = _pr_case(name, variant, seed)
        lab = pipeline.label(f, dec_kw=kw, period=case["period"], shape_models=models)
        np.savez_compressed(_cy_path(name, variant, seed, tag), truth=case["truth"], period=case["period"], lab=lab,
                            ball_x=f["ball_x"], ball_y=f["ball_y"], d_home=f["d_home"], d_away=f["d_away"])
        print(name, "saved, spellF1 %.4f" % _cy_f1(case["truth"], lab), flush=True)


def cy_load(variant, seed, tag=""):
    out = {}
    for name in tune_shape.TUNE:
        z = np.load(_cy_path(name, variant, seed, tag))
        out[name] = {k: z[k] for k in z.files}
    return out


def _cy_f1(truth, lab):
    return pm.evaluate(pm.label_codes(truth), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]


def _cy_blocks(d, min_s=1.0):
    """[(i, j)] hidden-ball blocks [i, j) of >= min_s inside one period, with a visible frame on both sides."""
    hid = np.isnan(d["ball_x"]) & (d["period"] > 0)
    e = np.diff(np.concatenate([[0], hid.astype(np.int8), [0]]))
    out = []
    for i, j in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if j - i >= min_s * FPS and i > 0 and j < len(hid) and d["period"][i - 1] == d["period"][i] == d["period"][j]:
            out.append((int(i), int(j)))
    return out


def cy_oracles(d):
    """{name: labels}: the default with one part of the hidden blocks replaced.
    O1 true team at disappearance on the truly live hidden frames (the envelope's stop-rule oracle)
    O2 true team at disappearance on the hidden frames the default calls live
    O3 true team per frame on hidden frames that are live in truth and in the default
    O4 truth on every truly live hidden frame
    P1 (no truth) the default's team at disappearance on the hidden frames it calls live"""
    t, lab = d["truth"], d["lab"]
    live_t, live_p = np.isin(t, ["H", "A"]), np.isin(lab, ["H", "A"])
    o = {k: lab.copy() for k in ("O1", "O2", "O3", "O4", "P1", "O5", "O6", "P2")}
    w = int(3 * FPS)
    for i, j in _cy_blocks(d):
        s = slice(i, j)
        # reappearance half of the hypothesis. O5 / O6: truth on the 3 s after / before the block
        # (ceiling of any edge fix). P2 (no truth): one team within 1 m of the reappearing ball and the
        # default says something else -> that team until the default agrees, at most 3 s.
        o["O5"][j:j + w] = np.where(live_t[j:j + w], t[j:j + w], lab[j:j + w])
        o["O6"][max(0, i - w):i] = np.where(live_t[max(0, i - w):i], t[max(0, i - w):i], lab[max(0, i - w):i])
        dh, da = d["d_home"][j], d["d_away"][j]
        if min(dh, da) <= 1.0 and max(dh, da) > 1.0:
            team = "H" if dh < da else "A"
            k = j
            while k < min(j + w, len(lab)) and lab[k] != team and d["period"][k] == d["period"][j]:
                k += 1
            o["P2"][j:k] = team
        if t[i - 1] in "HA":
            o["O1"][s] = np.where(live_t[s], t[i - 1], lab[s])
            o["O2"][s] = np.where(live_p[s], t[i - 1], lab[s])
        o["O3"][s] = np.where(live_t[s] & live_p[s], t[s], lab[s])
        o["O4"][s] = np.where(live_t[s], t[s], lab[s])
        if lab[i - 1] in "HA":
            o["P1"][s] = np.where(live_p[s], lab[i - 1], lab[s])
    return o


def cy_d0(variant, seed):
    """Step 0: team errors on hidden frames and around block edges, and the carry oracles."""
    data = cy_load(variant, seed)
    w = int(3 * FPS)
    tot = {}
    for name, d in data.items():
        t, lab, per = d["truth"], d["lab"], d["period"]
        live_t = np.isin(t, ["H", "A"])
        vis = ~np.isnan(d["ball_x"]) & (per > 0)
        blocks = _cy_blocks(d)
        base = _cy_f1(t, lab)
        print("\n%s  default %.4f  blocks >= 1 s: %d" % (name, base, len(blocks)))
        print("  truly live hidden frames by block length: n, right team / wrong team / D | truly dead: n, pred D")
        for lo, hi in zip(CY_BINS[:-1], CY_BINS[1:]):
            m = np.zeros(len(t), bool)
            for i, j in blocks:
                if lo * FPS <= j - i < hi * FPS:
                    m[i:j] = True
            lv, dd = m & live_t, m & (t == "D")
            if lv.sum():
                print("   %3g-%-4g s  %6d  %.3f / %.3f / %.3f | %6d %.3f" % (
                    lo, min(hi, 999), lv.sum(), (lab == t)[lv].mean(), (np.isin(lab, ["H", "A"]) & (lab != t))[lv].mean(),
                    (lab == "D")[lv].mean(), dd.sum(), (lab == "D")[dd].mean() if dd.sum() else np.nan))
        # edges: visible live frames in the 3 s before / after a block against all other visible live frames
        pre, post = np.zeros(len(t), bool), np.zeros(len(t), bool)
        for i, j in blocks:
            pre[max(0, i - w):i] = True
            post[j:j + w] = True
        for tag, m in (("3 s before a block", pre & vis & live_t), ("3 s after a block", post & vis & live_t),
                       ("other visible", ~pre & ~post & vis & live_t)):
            print("  %-20s live frames %6d: right %.3f wrong team %.3f D %.3f" % (
                tag, m.sum(), (lab == t)[m].mean(), (np.isin(lab, ["H", "A"]) & (lab != t))[m].mean(), (lab == "D")[m].mean()))
        # what carries: state at disappearance against the block's live frames
        c = Counter()
        for i, j in blocks:
            a_t, a_p, b_t = t[i - 1], lab[i - 1], t[j]
            lv = live_t[i:j]
            kind = "dead-in" if a_t == "D" else "live-in" if a_t in "HA" else "x"
            c[kind] += 1
            c[kind + " live frames"] += int(lv.sum())
            if a_t in "HA":
                c["live-in frames same team as at disappearance"] += int((t[i:j] == a_t)[lv].sum())
                c["live-in, default right at disappearance"] += int(a_p == a_t)
                c["live-in, all hidden frames dead after 2 s"] += int(not lv[int(2 * FPS):].any())
                c["live-in, block all live"] += int(lv.all())
                if b_t in "HA":
                    c["live-in live-out"] += 1
                    c["live-in live-out same team"] += int(b_t == a_t)
            if b_t in "HA":
                c["live-out"] += 1
                near = np.sign(d["d_away"][j] - d["d_home"][j]) if min(d["d_home"][j], d["d_away"][j]) <= 1.0 else 0
                c["live-out, a player within 1 m"] += int(near != 0)
                c["live-out, within 1 m and that team is the true team"] += int(near != 0 and (b_t == "H") == (near > 0))
                c["live-out, default right at reappearance"] += int(lab[j] == b_t)
        for k in sorted(c):
            print("  %-55s %d" % (k, c[k]))
        for k, o in cy_oracles(d).items():
            f1 = _cy_f1(t, o)
            tot.setdefault(k, []).append(f1 - base)
            print("  %s %.4f (%+.4f), frames changed %d" % (k, f1, f1 - base, (o != lab).sum()))
    print("\nmean gain over the tune matches: " + "  ".join("%s %+.4f" % (k, np.mean(v)) for k, v in tot.items()))


def _cy_edges(f, per, lab, r, truth=None):
    """Block-edge aggregates of one labelling: blocks >= 1 s, how often the reappearing ball has
    exactly one team within r (the L2-22 rule's trigger), label there, team switch across the block."""
    d = {"ball_x": f["ball_x"], "period": per}
    c = Counter()
    for i, j in _cy_blocks(d):
        c["blocks"] += 1
        dh, da = np.nan_to_num(f["d_home"][j], nan=99.0), np.nan_to_num(f["d_away"][j], nan=99.0)
        team = ("H" if dh < da else "A") if min(dh, da) <= r < max(dh, da) else ""
        c["fires"] += int(team != "")
        if team:
            c["fires, label at reappearance = that team"] += int(lab[j] == team)
            c["fires, label D"] += int(lab[j] == "D")
            if truth is not None:
                c["fires, truth live"] += int(truth[j] in "HA")
                c["fires, truth = that team"] += int(truth[j] == team)
        if lab[i - 1] in "HA" and lab[j] in "HA":
            c["team before and after"] += 1
            c["team switched across the block"] += int(lab[i - 1] != lab[j])
    return c


def cy_check(opt):
    """reappear off vs on per match: score, counts, rule trigger statistics, paired block bootstrap."""
    from possess import pipeline

    variant, seed, r = opt.get("bench", "real2"), int(opt.get("seed", 0)), float(opt.get("reappear", 1.0))
    variant = "" if variant == "orig" else variant
    clean = {n: tune_shape.shape_features(n, bench.get_case(n)) for n in tune_shape.TUNE}
    truths = {n: bench.get_case(n)["truth"] for n in tune_shape.TUNE}
    diffs, so, sn = [], [], []
    for name in opt.get("names", ",".join(tune_shape.TUNE)).split(","):
        models = shape.fit_models([clean[m] for m in clean if m != name], [truths[m] for m in clean if m != name], l2=tune_shape.L2)
        case, f = _pr_case(name, variant, seed)
        if int(opt.get("drop", 0)):
            f["pos"] = tune_shape.drop_players(f["pos"])
        if float(opt.get("noise", 0)):                    # players AND the distances the rule reads
            f["pos"] = tune_shape.add_noise(f["pos"], float(opt["noise"]))
            rng = np.random.default_rng(5)
            f["d_home"] = np.abs(f["d_home"] + rng.normal(0, float(opt["noise"]), len(f["d_home"])))
            f["d_away"] = np.abs(f["d_away"] + rng.normal(0, float(opt["noise"]), len(f["d_away"])))
        t, per = case["truth"], case["period"]
        off = pipeline.label(f, dec_kw=dict(reappear=0.0), period=per, shape_models=models)
        on = pipeline.label(f, dec_kw=dict(reappear=r), period=per, shape_models=models)
        ra, rb = (pm.evaluate(pm.label_codes(t), pm.label_codes(x), bootstrap_n=0) for x in (off, on))
        ca, cb = _cy_edges(f, per, off, r, t), _cy_edges(f, per, on, r, t)
        print("%s off %.4f %d/%d on %.4f %d/%d | blocks %d fires %d (truth live %d, truth that team %d) label = that team %d -> %d, "
              "label D %d -> %d | switch across block %d/%d -> %d/%d | frames changed %d" % (
                  name, ra["spell_f1_mean"], ra["n_pred_spells"], ra["n_gt_spells"], rb["spell_f1_mean"], rb["n_pred_spells"],
                  rb["n_gt_spells"], ca["blocks"], ca["fires"], ca["fires, truth live"], ca["fires, truth = that team"],
                  ca["fires, label at reappearance = that team"], cb["fires, label at reappearance = that team"],
                  ca["fires, label D"], cb["fires, label D"], ca["team switched across the block"], ca["team before and after"],
                  cb["team switched across the block"], cb["team before and after"], (off != on).sum()), flush=True)
        bo, bn = _pr_boot(t, off, on)
        diffs.append(bn - bo); so.append(ra["spell_f1_mean"]); sn.append(rb["spell_f1_mean"])
    dd = np.mean(diffs, axis=0)
    print("mean off %.4f on %.4f | paired 2-min block bootstrap %+.4f CI90 [%+.4f, %+.4f], draws <= 0: %d of %d" % (
        np.mean(so), np.mean(sn), dd.mean(), np.percentile(dd, 5), np.percentile(dd, 95), (dd <= 0).sum(), len(dd)))


def cy_transfer(opt):
    """Provided-match INPUT aggregates, reappear off / on -> eval/results/transfer_reappear.json."""
    import json

    from possess import pipeline

    r = float(opt.get("reappear", 1.0))
    f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"),
                                    os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz"))   # input caches only
    per = pipeline.challenge_period(len(f["ball_x"]))
    play = per > 0
    hh = play.sum() / FPS / 1800.0
    res = {}
    labs = {"off": pipeline.label(f, dec_kw=dict(reappear=0.0), period=per), "on": pipeline.label(f, dec_kw=dict(reappear=r), period=per)}
    for tg, lab in labs.items():
        c = _cy_edges(f, per, lab, r)
        res[tg] = {"possessions / half hour": (_pr_runs((lab == "H") & play) + _pr_runs((lab == "A") & play)) / hh,
                   "D runs / half hour": _pr_runs((lab == "D") & play) / hh, "D share": float((lab == "D")[play].mean()),
                   "H share of H+A": float((lab == "H")[play].sum() / np.isin(lab, ["H", "A"])[play].sum()), **c}
    res["frames changed"] = float((labs["off"] != labs["on"])[play].mean())
    for k, v in res.items():
        print(k, v)
    with open(os.path.join(ROOT, "eval", "results", "transfer_reappear.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


def cmd_carry(args):
    """Envelope L2-22 (which team during and around hidden-ball blocks):
    carry save [bench=real2] [seed=0]   one default decode per tune match -> carry_*.npz
    carry d0 [bench=real2] [seed=0]     step 0: team errors on hidden frames / block edges, carry oracles
    carry check [bench=real2] [seed=1 names=metrica1] [noise=0.1 drop=1] [reappear=1.0]   decode.touch_hold
                                        reappear off vs on: scores, trigger statistics, block bootstrap
    carry transfer [reappear=1.0]       provided-match INPUT aggregates off / on -> transfer_reappear.json"""
    cmd = args[0] if args else "d0"
    opt = dict(a.split("=") for a in args[1:])
    variant, seed = opt.get("bench", "real2"), int(opt.get("seed", 0))
    variant = "" if variant == "orig" else variant
    if cmd == "save":
        return cy_save(variant, seed)
    if cmd == "d0":
        return cy_d0(variant, seed)
    if cmd == "check":
        return cy_check(opt)
    if cmd == "transfer":
        return cy_transfer(opt)
    raise SystemExit(cmd_carry.__doc__)


def main():
    if sys.argv[1:2] == ["carry"]:
        return cmd_carry(sys.argv[2:])
    if sys.argv[1:2] == ["prior"]:
        return cmd_prior(sys.argv[2:])
    if sys.argv[1:2] == ["edges"]:
        return cmd_edges(sys.argv[2:])
    if sys.argv[1:2] == ["cleanup"]:
        return cmd_cleanup(sys.argv[2:])
    if sys.argv[1:2] == ["transfer"]:
        return cmd_transfer()
    if sys.argv[1:2] == ["segment"]:                     # segment [bench=real1] [duel=0 ...]
        for it in list(sys.argv[2:]):
            if it.startswith("bench="):
                tune_shape.VARIANT = "" if it[6:] == "orig" else it[6:]
                sys.argv.remove(it)
        return cmd_segment(tune_shape.kv(sys.argv[2:]))
    if sys.argv[1:2] == ["dhide"]:
        return cmd_dhide(tuple(float(x) for x in sys.argv[2:]) or (0.0, 0.5, 0.8, 0.95))   # dhide 0.8 0.95
    for it in list(sys.argv[1:]):
        if it.startswith("bench="):
            tune_shape.VARIANT = "" if it[6:] == "orig" else it[6:]
            sys.argv.remove(it)
    kw = tune_shape.kv(sys.argv[1:])
    deg = {k: kw.pop(k) for k in ("view", "noise", "drop", "ball_noise", "ctx", "fast") + FAULT_KEYS if k in kw}
    quick = kw.pop("quick", 0)
    rows, res = default_run(kw, **deg)
    print("config: default DEC_SHAPE + %s  degradation %s" % (kw, deg))
    print("tune", tune_shape.fmt(res))
    if quick:
        return
    miss, spur = Counter(), Counter()
    timing = []
    orc = {}
    hist = {"true": np.zeros((len(BINS) - 1, 2)), "pred": np.zeros((len(BINS) - 1, 2))}
    conf = {"visible": np.zeros((3, 3)), "hidden": np.zeros((3, 3))}
    brec = Counter()
    for name, case, feats, lab in rows:
        truth, hidden = case["truth"], case["hidden"]
        g = pm.label_codes(truth)
        runs = pm.valid_runs(g)
        p = pm.fill_abstentions(pm.label_codes(lab), runs)
        m, s = bench.explain(g, p, runs, hidden)
        miss.update(m); spur.update(s)
        timing.append("  %-9s %s" % (name, stoppage_timing(g, p, runs, hidden, feats)))
        print("\n%s  missed %d  spurious %d" % (name, sum(m.values()), sum(s.values())))
        for k, v in oracle.variants(truth, lab, case).items():
            r = pm.evaluate(g, pm.label_codes(v), bootstrap_n=0)
            orc.setdefault(k, []).append((r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))
        gs, ps = pm.possession_spells(g, runs), pm.possession_spells(p, runs)
        g_ok, p_ok = matched_flags(gs, ps)
        for key, sp, ok in (("true", gs, g_ok), ("pred", ps, p_ok)):
            t = length_table(sp, ok)
            hist[key] += np.array([(n, n * (r if n else 0)) for n, r in t])
            print("  %s lengths  " % key + "  ".join("%d:%.2f" % (n, r) for n, r in t))
        # frame confusion on scored frames
        idx = {pm.H: 0, pm.A: 1, pm.D: 2}
        for a, b in runs:
            for key, mask in (("visible", ~hidden[a:b]), ("hidden", hidden[a:b])):
                for gc in idx:
                    for pc in idx:
                        conf[key][idx[gc], idx[pc]] += ((g[a:b] == gc) & (p[a:b] == pc) & mask).sum()
        # boundary recall by type within +-1 s (same destination label)
        pb = boundaries(p, runs)
        for f, kind in boundaries(g, runs):
            hit = any(k2 == kind and abs(f2 - f) <= 25 and p[min(f2, len(p) - 1)] == g[f] for f2, k2 in pb)
            hid = "hidden" if hidden[f] else "visible"
            brec[(kind, hid, "n")] += 1
            brec[(kind, hid, "hit")] += hit
        gb = boundaries(g, runs)
        for f, kind in pb:
            hit = any(k2 == kind and abs(f2 - f) <= 25 and g[min(f2, len(g) - 1)] == p[f] for f2, k2 in gb)
            hid = "hidden" if hidden[f] else "visible"
            brec[("pred " + kind, hid, "n")] += 1
            brec[("pred " + kind, hid, "hit")] += hit

    print("\n(a) why true possessions were missed (IoU 0.5), all tune matches")
    for k, v in miss.most_common():
        print("  %5d  %s" % (v, k))
    print("    why predicted possessions were spurious")
    for k, v in spur.most_common():
        print("  %5d  %s" % (v, k))
    print("\n(b) oracle ceilings: spellF1 (pred/true) per match, mean")
    for k, v in orc.items():
        print("  %-14s %s   mean %.4f" % (k, "  ".join("%.4f (%d/%d)" % x for x in v), np.mean([x[0] for x in v])))
    print("\n(c) possession lengths, all tune matches: bin [s]  count  share matched at IoU 0.5")
    for key in ("true", "pred"):
        print("  %s (%s)" % (key, "recall" if key == "true" else "precision"))
        for (lo, hi), (n, k) in zip(zip(BINS[:-1], BINS[1:]), hist[key]):
            print("    %4.0f-%-4s %5d  %.2f" % (lo, "%.0f" % hi if hi < 1e8 else "inf", n, k / n if n else float("nan")))
    print("\n(d) frame confusion, rows truth H/A/D, columns prediction H/A/D (share of all scored frames)")
    tot = conf["visible"].sum() + conf["hidden"].sum()
    for key in ("visible", "hidden"):
        print("  ball %s" % key)
        for r in conf[key]:
            print("    " + "  ".join("%.3f" % (x / tot) for x in r))
        c = conf[key]                                    # envelope L2-16: stoppage frames found / right
        print("    D recall %.3f  precision %.3f" % (c[2, 2] / c[2].sum(), c[2, 2] / c[:, 2].sum()))
    print("    boundary recall / precision within +-1 s")
    for kind in ("turnover", "stop", "restart", "pred turnover", "pred stop", "pred restart"):
        for hid in ("visible", "hidden"):
            n = brec[(kind, hid, "n")]
            print("    %-14s %-8s %4d  %.2f" % (kind, hid, n, brec[(kind, hid, "hit")] / n if n else float("nan")))
    print("\n(e) stoppage timing, predicted - true (visible true boundaries), and the bench ball on true stoppages")
    print("\n".join(timing))


if __name__ == "__main__":
    main()
