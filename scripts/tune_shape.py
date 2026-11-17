"""Ball-free shape evidence (possess/shape.py, plan P5) on the TUNE split only, leave-one-match-out.

    uv run python scripts/tune_shape.py collect          # one pass per tune match -> position + feature caches
    uv run python scripts/tune_shape.py auc              # leave-one-match-out AUC of the two models, per feature
    uv run python scripts/tune_shape.py grid             # decoder grid (<= 30 configs) -> eval/results/tune_shape.json
    uv run python scripts/tune_shape.py eval w_shape=0.05 w_sdead=0.1 c_stop=6.0   # one config, per match
    uv run python scripts/tune_shape.py explain w_shape=... # bench.explain() causes, baseline vs config
    uv run python scripts/tune_shape.py fit              # final models on all tune matches (paste into shape.MODELS)
    uv run python scripts/tune_shape.py eval ... view=25 # robustness: players outside a +-25 m camera window dropped
    uv run python scripts/tune_shape.py eval_mismatch ... view=25   # same, models fitted on full 11 v 11
    uv run python scripts/tune_shape.py grid 5 bench=real1   # L2-13 grid on the realistic bench (bench.VARIANTS)
    uv run python scripts/tune_shape.py grid 6 bench=real1   # L2-14 grid (frame rule weight, distance cap)
    uv run python scripts/tune_shape.py grid 7 bench=real1   # L2-15 grid (contested touches)
    uv run python scripts/tune_shape.py gate             # noise gate: sigma 0 / 0.05 / 0.10 m and 7-12 players,
                                                         # adopted config vs team-only, models fitted on clean data

Every score is leave-one-match-out: the shape models used on a match are fitted on the other two
tune matches (the decoder weights themselves are grid-chosen on the tune mean, as in
tune_decoder.py). Never run on held-out matches, the public window or challenge data.
Caches: data/external/_bench/shapepos_<name>_v1.npz (positions), shape_<name>_s0_v<FEAT_V>.npz.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench  # noqa: E402
from possess import pipeline, shape  # noqa: E402
from possess.deadball2 import detect, features as dead_features  # noqa: E402
from possess.decode import decode_halves  # noqa: E402

pm = bench.pm
TUNE = bench.SPLITS["tune"]                 # never the held-out matches
FEAT_V = 4          # v2: lagged player speed, goalkeeper rule, voted attacking direction (Fable QA)
                    # v3 (L2-16): + "fast" columns (shape.FAST_FEATS); team / dead columns as in v2
                    # v4 (L2-18): + "p90" (raw fast-player speed for possess/edges.py); the rest as in v3
L2 = 10.0


def positions(name, case, variant=""):
    """Player positions of a tune match. variant: a bench variant that loses players (real2, envelope
    L2-20) has its own positions, written by bench.get_case next to the case; every other variant
    shares the full positions."""
    if variant and bench.VARIANTS[variant].get("real2"):
        return np.load(os.path.join(bench.CACHE, "shapepos_%s_s0_%s.npz" % (name, variant)))["pos"]
    path = os.path.join(bench.CACHE, "shapepos_%s_v1.npz" % name)
    if os.path.exists(path):
        return np.load(path)["pos"]
    pos = shape.collect(bench.load_match(name).iter_frames(), n=len(case["truth"]))
    np.savez_compressed(path, pos=pos)
    return pos


def camera_view(pos, half_width):
    """Robustness degradation (swarm F7: the challenge tracks 7-12 players per team): keep only
    players within `half_width` metres (along x) of the 1 s smoothed mean x of all players, as a
    broadcast camera following play would."""
    x = pos[:, :, :, 0].reshape(len(pos), -1)
    with shape._quiet():
        c = shape.smooth(np.nanmean(x, axis=1), 25)
    out = pos.copy()
    out[np.abs(pos[:, :, :, 0] - c[:, None, None]) > half_width] = np.nan
    return out


def add_noise(pos, sigma, seed=0):
    """White position noise (m) on every player, every frame."""
    return pos + np.random.default_rng(seed).normal(0, sigma, pos.shape).astype(np.float32)


def drop_players(pos, seed=0, lo=7, hi=12, seg_s=(2, 10)):
    """7-12 tracked players per team (swarm F7): per team, a count drawn uniformly from lo..hi
    for random 2-10 s segments; the players furthest (along x) from the mean x of all players
    are dropped, as a camera following play loses them."""
    rng = np.random.default_rng(seed)
    n = len(pos)
    out = pos.copy()
    with shape._quiet():
        c = shape.smooth(np.nanmean(pos[:, :, :, 0].reshape(n, -1), axis=1), 25)
    for t in (0, 1):
        keep = np.empty(n, int)
        i = 0
        while i < n:
            L = int(rng.uniform(*seg_s) * 25)
            keep[i:i + L] = rng.integers(lo, hi + 1)
            i += L
        d = np.abs(pos[:, t, :, 0] - c[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        rank = np.argsort(np.argsort(d, axis=1), axis=1)
        out[:, t][rank >= keep[:, None]] = np.nan
    return out


def shape_features(name, case, view=0, noise=0.0, drop=0):
    tag = ("_view%d" % view if view else "") + ("_noise%g" % noise if noise else "") + ("_drop" if drop else "")
    path = os.path.join(bench.CACHE, "shape_%s_s0_v%d%s.npz" % (name, FEAT_V, tag))
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    pos = positions(name, case)
    if view:
        pos = camera_view(pos, view)
    if drop:
        pos = drop_players(pos)
    if noise:
        pos = add_noise(pos, noise)
    if not tag:
        print(name, "attacking direction (period: sign, vote share)", shape.attack_votes(pos, case["period"]))
    f = shape.features(pos, case["period"])
    np.savez_compressed(path, **f)
    return f


VARIANT = ""        # bench case variant for load_all ("" = original bench, "real1" = realistic, L2-12)


def load_all(view=0, noise=0.0, drop=0, ball_noise=0.0, variant=None):
    """[(name, case, decoder feats, shape feats)] for the tune matches; detector run once each.
    variant (default VARIANT; set by `bench=real1` on the command line): the ball of bench.VARIANTS.
    Truth, hidden mask and player positions are the same in every variant, so are the shape features."""
    variant = VARIANT if variant is None else variant
    data = []
    for name in TUNE:
        case = bench.get_case(name, variant=variant)
        f = bench.case_features(name, case, variant=variant)
        df = dead_features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["cen_home"], f["cen_away"])
        feats = {"d_home": np.nan_to_num(f["d_home"], nan=99.0), "d_away": np.nan_to_num(f["d_away"], nan=99.0),
                 "visible": ~np.isnan(f["ball_x"]), "dead": detect(df, f["ml_dead"], **pipeline.DEAD2),
                 "ball_x": f["ball_x"], "ball_y": f["ball_y"]}        # for decode.touch_hold (w_touch > 0)
        if ball_noise:        # white noise (m) on the ball the touch detector sees (distances unchanged)
            rng = np.random.default_rng(1)
            feats["ball_x"] = f["ball_x"] + rng.normal(0, ball_noise, len(f["ball_x"]))
            feats["ball_y"] = f["ball_y"] + rng.normal(0, ball_noise, len(f["ball_y"]))
        data.append((name, case, feats, shape_features(name, bench.get_case(name) if variant else case,
                                                       view, noise, drop)))
    return data


def lomo_models(data, l2=L2):
    """{name: models fitted on the other tune matches}."""
    return {name: shape.fit_models([s for n, _, _, s in data if n != name],
                                   [c["truth"] for n, c, _, _ in data if n != name], l2=l2)
            for name, _, _, _ in data}


def with_emissions(data, models, clip=4.0, ctx=None, fast=None):
    """ctx=0: base team model only (before L2-10); None = shape.TEAM_CTX.
    fast=0: stoppage model without the fast-player columns (before L2-16); None = shape.DEAD_FAST."""
    out = []
    for name, case, feats, sf in data:
        st, sd = shape.emissions(sf, models[name], clip, ctx, fast=fast)
        e = dict(feats, shape_team=st, shape_dead=sd, edge_p90=sf["p90"], shape_ok=sf["ok"])   # edge_p90: L2-18
        pi = shape.dead_prior(models[name]) if fast is None or fast else None                  # L2-21 (prior_shift)
        if pi is not None:
            e["sdead_pi"] = pi
        out.append((name, case, e))
    return out


def run(data_e, kw):
    res = []
    for name, case, feats in data_e:
        lab = decode_halves(case["period"], feats=dict(feats), **dict(pipeline.DEC_BASE, **kw))
        r = pm.evaluate(pm.label_codes(case["truth"]), pm.label_codes(lab), bootstrap_n=0)
        res.append((name, r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"], lab))
    return res


def fmt(res):
    return "%.4f  (%s)" % (np.mean([r[1] for r in res]), "  ".join("%s %.3f %d/%d" % r[:4] for r in res))


def kv(items):
    out = {}
    for it in items:
        k, v = it.split("=")
        out[k] = float(v) if "." in v or "e" in v else int(v)
    return out


def cmd_auc(data):
    models = lomo_models(data)
    for name, case, feats, sf in data:
        truth, ok = case["truth"], sf["ok"]
        hid = ~feats["visible"]
        st, sd = shape.emissions(sf, models[name], clip=50.0)
        live = ok & np.isin(truth, ["H", "A"])
        alld = ok & np.isin(truth, ["H", "A", "D"])
        print("%s  ok %.3f  TEAM auc all %.3f hidden %.3f | DEAD auc all %.3f hidden %.3f" % (
            name, ok[case["period"] > 0].mean(), shape.auc(st[live], truth[live] == "H"),
            shape.auc(st[live & hid], truth[live & hid] == "H"), shape.auc(sd[alld], truth[alld] == "D"),
            shape.auc(sd[alld & hid], truth[alld & hid] == "D")))
        print("   team feats: " + "  ".join("%s %.2f" % (k, shape.auc(sf["team"][live, i], truth[live] == "H"))
                                            for i, k in enumerate(shape.TEAM_FEATS)))
        print("   dead feats: " + "  ".join("%s %.2f" % (k, shape.auc(sf["dead"][alld, i], truth[alld] == "D"))
                                            for i, k in enumerate(shape.DEAD_FEATS)))


GRID = [
    {},                                                       # baseline (pipeline.DEC_BASE)
    dict(w_shape=0.02), dict(w_shape=0.05), dict(w_shape=0.1), dict(w_shape=0.2),
    dict(w_shape=0.05, gap_pos=-1.0), dict(w_shape=0.1, gap_pos=-1.0),
    dict(w_sdead=0.05, c_stop=6.0), dict(w_sdead=0.1, c_stop=6.0), dict(w_sdead=0.2, c_stop=6.0),
    dict(w_sdead=0.1, c_stop=3.0), dict(w_sdead=0.1, c_stop=12.0), dict(w_sdead=0.2, c_stop=12.0),
    dict(w_sdead=0.4, c_stop=12.0), dict(w_sdead=0.4, c_stop=25.0), dict(w_sdead=1.0, c_stop=25.0),
    dict(w_shape=0.05, w_sdead=0.1, c_stop=6.0), dict(w_shape=0.05, w_sdead=0.2, c_stop=12.0),
    dict(w_shape=0.1, w_sdead=0.2, c_stop=12.0), dict(w_shape=0.1, w_sdead=0.4, c_stop=25.0),
]

C2 = dict(w_shape=0.05, w_sdead=0.2, c_stop=12.0)             # best of GRID
GRID2 = [dict(C2, **d) for d in (
    {}, dict(gap_pos=-1.0), dict(shape_vis=0.5), dict(shape_vis=1.0), dict(shape_vis=1.0, gap_pos=-1.0),
    dict(w_shape=0.03), dict(w_shape=0.07), dict(w_sdead=0.3), dict(w_sdead=0.15), dict(c_stop=8.0),
    dict(c_stop=16.0), dict(w_sdead=0.3, c_stop=16.0), dict(c_switch=3.0), dict(c_switch=5.0),
    dict(c_switch=6.0), dict(c_same=0.0), dict(c_same=3.0), dict(w_rule=0.0), dict(w_rule=0.6),
    dict(cap=2.0), dict(shape_vis=1.0, gap_pos=-1.0, w_shape=0.03), dict(shape_vis=1.0, gap_pos=-1.0, c_switch=5.0),
)]
C3 = dict(C2, gap_pos=-1.0, shape_vis=1.0)                     # best region of GRID2
GRID3 = [dict(C3, **d) for d in (
    {}, dict(sdead_vis=1.0, w_dead=1.0), dict(sdead_vis=1.0, w_dead=2.0), dict(sdead_vis=1.0, w_dead=4.0),
    dict(sdead_vis=1.0, w_dead=8.0), dict(sdead_vis=0.5, w_dead=2.0), dict(sdead_vis=0.5, w_dead=4.0),
    dict(sdead_vis=2.0, w_dead=2.0), dict(sdead_vis=1.0, w_dead=0.5), dict(sdead_vis=1.0, w_dead=0.0001),
    dict(sdead_vis=1.0, w_dead=2.0, c_stop=8.0), dict(sdead_vis=1.0, w_dead=2.0, c_stop=20.0),
    dict(sdead_vis=1.0, w_dead=2.0, w_sdead=0.3), dict(sdead_vis=1.0, w_dead=2.0, w_shape=0.03),
)]
# after the Fable QA fixes (features v2): one-at-a-time around the v1 best, 14 configs
C4 = dict(w_shape=0.05, w_sdead=0.2, c_stop=12.0, gap_pos=-1.0, shape_vis=1.0, sdead_vis=1.0, w_dead=1.0)
GRID4 = [dict(C4, **d) for d in (
    {}, dict(w_shape=0.03), dict(w_shape=0.07), dict(w_sdead=0.15), dict(w_sdead=0.3), dict(w_dead=0.5),
    dict(w_dead=1.5), dict(c_stop=8.0), dict(c_stop=16.0), dict(sdead_vis=0.5), dict(sdead_vis=1.5),
    dict(c_switch=5.0), dict(w_sdead=0.0, sdead_vis=0.0), dict(w_shape=0.0),
)]
# L2-13: stoppages when the dead ball moves. Run with bench=real1 (the deciding bench), 18 configs.
# Base = the L2-10 default; w_live = score of the detector's "live" verdict (decode.viterbi).
GRID5 = [dict(shape.DEC_SHAPE_L2_10, **d) for d in (
    {}, dict(w_live=0.5), dict(w_live=0.25), dict(w_live=0.0),
    dict(sdead_vis=1.5), dict(sdead_vis=2.0),
    dict(w_live=0.5, sdead_vis=1.5), dict(w_live=0.25, sdead_vis=1.5), dict(w_live=0.0, sdead_vis=1.5),
    dict(w_live=0.5, sdead_vis=2.0), dict(w_live=0.25, sdead_vis=2.0), dict(w_live=0.0, sdead_vis=2.0),
    dict(w_live=0.0, sdead_vis=2.5), dict(w_live=0.0, sdead_vis=3.0),
    dict(w_live=0.0, sdead_vis=2.0, w_dead=2.0), dict(w_live=0.0, sdead_vis=2.0, c_stop=16.0),
    dict(w_live=0.0, sdead_vis=2.0, c_stop=8.0), dict(w_live=0.0, sdead_vis=2.0, w_sdead=0.45),
)]
# L2-14: ball proximity counted three times on visible frames (distance, frame rule, touch).
# Run with bench=real1, 11 configs. Base = the L2-13 default (w_rule 0.3, cap 1.5).
# Found by a one-at-a-time scan of 24 decoder constants around the L2-13 default on real1
# (envelope L2-14-next-fable.out.md): only w_rule=0 and cap=1.0 raised all three matches.
# Adopted: cap=1.0 alone (w_rule=0 failed the transfer check, see shape.DEC_SHAPE).
C6 = dict(shape.DEC_SHAPE_L2_13)
GRID6 = [dict(C6, **d) for d in (
    {}, dict(w_rule=0.0), dict(cap=1.0), dict(w_rule=0.0, cap=1.0), dict(w_rule=0.15, cap=1.0),
    dict(w_rule=0.0, cap=1.25), dict(w_rule=0.0, cap=1.0, w_touch=0.5), dict(w_rule=0.0, cap=1.0, c_switch=5.0),
    dict(w_rule=0.0, cap=1.0, w_shape=0.08), dict(w_rule=0.0, cap=0.75), dict(w_rule=0.0, cap=1.0, w_sdead=0.4),
)]
# L2-15: contested touches (both teams within r_touch) in decode.touch_hold. Run with bench=real1.
# Base = the L2-14 default (duel=0: the hold is cleared). Adopted: duel=2 (the holder keeps the ball).
# A one-at-a-time scan of 22 other constants around L2-14 found nothing above 0.5874 (envelope
# L2-15-segment-fable.out.md): r_touch 1.5 / 2.0, max_hold_s 2 / 8 / 15, c_switch 2 / 3 / 5,
# w_touch 1.0 / 1.5, lag 1 / 3, d0 0.15 / 0.5, w_ctrl 0.5 / 1.5, w_shape 0.08, c_same 0, c_stop 9.
GRID7 = [dict(shape.DEC_SHAPE_L2_14, **d) for d in (
    {}, dict(duel=1), dict(duel=2), dict(duel=2, c_switch=3.0), dict(duel=2, max_hold_s=6.0),
    dict(duel=2, w_touch=1.0), dict(duel=2, lag=1), dict(duel=2, w_shape=0.08), dict(duel=2, w_ctrl=0.5),
    dict(duel=2, r_touch=1.5),
)]
GRIDS = {"1": GRID, "2": GRID2, "3": GRID3, "4": GRID4, "5": GRID5, "6": GRID6, "7": GRID7}


def cmd_grid(data, grid=GRID, tag="tune_shape"):
    data_e = with_emissions(data, lomo_models(data))
    rows = []
    for kw in grid[:30]:
        res = run(data_e, kw)
        rows.append(dict(kw=kw, per_match=[round(r[1], 4) for r in res], n_pred=[r[2] for r in res],
                         spellF1=float(np.mean([r[1] for r in res]))))
        print(len(rows), kw, fmt(res), flush=True)
    rows.sort(key=lambda r: -r["spellF1"])
    with open(os.path.join(ROOT, "eval", "results", tag + ".json"), "w") as fh:
        json.dump(dict(protocol="leave-one-match-out shape models, tune split", l2=L2, rows=rows), fh, indent=1)


def cmd_explain(data, kw):
    data_e = with_emissions(data, lomo_models(data))
    for label, k in (("baseline", {}), ("config", kw)):
        miss, spur = Counter(), Counter()
        res = run(data_e, k)
        for (name, case, feats), r in zip(data_e, res):
            g = pm.label_codes(case["truth"])
            runs = pm.valid_runs(g)
            m, s = bench.explain(g, pm.fill_abstentions(pm.label_codes(r[4]), runs), runs, case["hidden"])
            miss.update(m); spur.update(s)
        print(label, fmt(res))
        for title, c in (("missed", miss), ("spurious", spur)):
            for key, v in c.most_common(8):
                print("   %-8s %4d  %s" % (title, v, key))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "grid"
    global VARIANT
    deg = dict(view=0, noise=0.0, drop=0, ball_noise=0.0)  # robustness degradations of the positions / ball
    for it in list(sys.argv[2:]):
        k, _, v = it.partition("=")
        if k == "bench":                                   # bench=real1 | orig
            VARIANT = "" if v == "orig" else v
            sys.argv.remove(it)
        if k in deg:
            deg[k] = float(v) if k in ("noise", "ball_noise") else int(v)
            sys.argv.remove(it)
    view = deg["view"]
    data = load_all(**deg)
    if cmd == "collect":
        for name, _, _, sf in data:
            print(name, "frames", len(sf["ok"]), "usable %.3f" % sf["ok"].mean(), "(view=%d)" % view)
    elif cmd == "auc":
        cmd_auc(data)
    elif cmd == "grid":
        g = sys.argv[2] if len(sys.argv) > 2 else "1"
        cmd_grid(data, GRIDS[g], ("tune_shape" if g == "1" else "tune_shape_grid" + g) + ("_" + VARIANT if VARIANT else ""))
    elif cmd == "eval":
        print(fmt(run(with_emissions(data, lomo_models(data)), kv(sys.argv[2:]))))
    elif cmd == "fit":                                    # final models on all tune matches -> shape.MODELS
        m = shape.fit_models([d[3] for d in data], [d[1]["truth"] for d in data], l2=L2)
        for name, d in m.items():
            print('    "%s": {' % name)
            for k in ("mu", "sd", "w", "b") + (("pi",) if "pi" in d else ()):
                print('        "%s": %s,' % (k, [float("%.5g" % v) for v in np.atleast_1d(d[k])]))
            print("    },")
    elif cmd == "gate":
        models = lomo_models(data)                        # fitted on clean positions (as shipped)
        cfg = dict(shape.DEC_SHAPE, **kv(sys.argv[2:]))
        team_only = dict(cfg, w_sdead=0.0, sdead_vis=0.0)
        out = {}
        for tag, d in (("sigma0", {}), ("sigma0.05", dict(noise=0.05)), ("sigma0.10", dict(noise=0.1)),
                       ("players7-12", dict(drop=1)), ("players7-12+sigma0.10", dict(drop=1, noise=0.1))):
            de = with_emissions(load_all(**d) if d else data, models)
            a, b = run(de, cfg), run(de, team_only)
            out[tag] = dict(adopted=float(np.mean([r[1] for r in a])), team_only=float(np.mean([r[1] for r in b])))
            print("%-22s adopted   %s" % (tag, fmt(a)), flush=True)
            print("%-22s team-only %s" % ("", fmt(b)), flush=True)
        with open(os.path.join(ROOT, "eval", "results", "tune_shape_gate%s.json" % ("_" + VARIANT if VARIANT else "")), "w") as fh:
            json.dump(dict(config=cfg, gate=out), fh, indent=1)
    elif cmd == "eval_mismatch":                          # models fitted on full 11 v 11, applied to view=
        models = lomo_models(load_all(0))
        print(fmt(run(with_emissions(data, models), kv(sys.argv[2:]))))
    elif cmd == "explain":
        cmd_explain(data, kv(sys.argv[2:]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
