"""Re-score the adopted steps (ADR-005..009) one by one on a bench variant (envelope L2-12, audit).

    uv run python scripts/audit_bench.py build --variant real1 --match metrica1   # case + feature cache
    uv run python scripts/audit_bench.py score --variant real1                    # tune split, all steps
    uv run python scripts/audit_bench.py score                                    # the protocol bench
    uv run python scripts/audit_bench.py score --variant real1 --split heldout    # a held-out LOOK

Variants: bench.VARIANTS ("" = protocol bench; dead1 = realistic dead ball; real1 = dead1 + faults).
Truth, hidden-block mask and player positions are identical across variants; only the ball differs.
Shape models: leave-one-match-out on the tune split (fitted from positions + truth, so the same fit
serves every variant: no refit is needed); held-out uses shape.MODELS.
Changes no pipeline default. Writes eval/results/audit_steps_<variant>[_HELDOUT].json.
real2 (L2-20) is the exception: its hidden mask and player positions differ; `score` is not meant for it
(use scripts/evaluate.py --bench real2).

    uv run python scripts/audit_bench.py inputs --variant real2             # envelope L2-20
    uv run python scripts/audit_bench.py inputs --variant real2 --kw p_hide=0.6 smooth_sigma=4.0   # uncached scan

`inputs`: INPUT aggregates (ball track, gap lengths, player counts, fast-player speed; the metrics of
.planning/research/provided-vs-bench-gaps.md) of the provided match next to the tune matches of a bench
variant. The provided match is read through its feature / position caches only (ball and player
positions): no label of it, and never the public ground truth. Writes eval/results/audit_inputs_<variant>.json.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench  # noqa: E402
import tune_shape  # noqa: E402
from possess import pipeline, shape  # noqa: E402
from possess.deadball2 import detect, features as dead_features  # noqa: E402
from possess.decode import decode_halves  # noqa: E402

pm = bench.pm
FULL = dict(pipeline.DEC_BASE, **shape.DEC_SHAPE)
# (step, ADR, uses shape, team-context, decoder kwargs)   None kwargs = frame rule
STEPS = [
    ("rule (before ADR-005)", None, False, 0, None),
    ("ADR-005 decoder", "005", False, 0, dict(pipeline.DEC_BASE, w_touch=0.0)),
    ("ADR-006 + team shape", "006", True, 0, dict(FULL, w_touch=0.0)),
    ("ADR-007 + touch", "007", True, 0, dict(FULL, ball_clean=0)),
    ("ADR-008 + ball cleaning", "008", True, 0, dict(FULL)),
    ("ADR-009 + team context (default)", "009", True, None, dict(FULL)),
    # leave-one-out from the default: what each step is worth inside the final pipeline
    ("default minus touch", "-007", True, None, dict(FULL, w_touch=0.0)),
    ("default minus shape", "-006", False, 0, dict(pipeline.DEC_BASE)),
    ("default minus shape stoppage (w_sdead=0)", "-006d", True, None, dict(FULL, w_sdead=0.0)),
]


def load(name, variant):
    case = bench.get_case(name, variant=variant)
    f = bench.case_features(name, case, variant=variant)
    df = dead_features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["cen_home"], f["cen_away"])
    dead = detect(df, f["ml_dead"], **pipeline.DEAD2)
    feats = {"d_home": np.nan_to_num(f["d_home"], nan=99.0), "d_away": np.nan_to_num(f["d_away"], nan=99.0),
             "visible": ~np.isnan(f["ball_x"]), "dead": dead, "ball_x": f["ball_x"], "ball_y": f["ball_y"]}
    return case, f, feats


def detector_quality(case, feats):
    """Stoppage detector against the external truth on visible frames (the bench-sensitive part)."""
    t, vis, dead = case["truth"], feats["visible"], feats["dead"]
    ok = vis & np.isin(t, ["H", "A", "D"])
    td = t == "D"
    tp = (dead & td & ok).sum()
    return {"D_recall_visible": float(tp / max(1, (td & ok).sum())),
            "D_precision_visible": float(tp / max(1, (dead & ok).sum())),
            "dead_share_visible": float(dead[ok].mean()), "truth_D_share_visible": float(td[ok].mean())}


def _q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else float("nan")


def input_stats(period, bx, by, pos, fps=25.0):
    """Input-only aggregates of one match: period per frame, the ball the pipeline sees (nan = hidden),
    player positions [n, 2, SLOTS, 2]. No label is read."""
    from possess.ballclean import clean_ball
    from possess.decode import touch_hold
    from possess.edges import edge_columns

    runs = bench._runs
    o = {}
    n = len(bx)
    inp = period > 0
    vis = inp & np.isfinite(bx)
    hid = inp & ~vis
    mins, vmin = inp.sum() / fps / 60, vis.sum() / fps / 60
    o["hidden share"] = hid.sum() / inp.sum()
    w, sh = int(300 * fps), []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        sh += [hid[idx[s:s + w]].mean() for s in range(0, len(idx) - w + 1, w)]
    o["hidden share 5-min win min"], o["hidden share 5-min win max"] = min(sh), max(sh)
    o["hidden share 5-min win sd"] = float(np.std(sh))
    gs, ge = runs(hid)
    gl = (ge - gs) / fps
    big = gl >= 5
    o["gaps >= 5 s per half hour"] = big.sum() / mins * 30
    o["gap >= 5 s median s"], o["gap max s"] = _q(gl[big], 50), float(gl.max())
    o["hidden frames in gaps > 60 s"] = (ge - gs)[gl > 60].sum() / max(1, hid.sum())
    o["hidden frames in gaps 40-60 s"] = (ge - gs)[(gl > 40) & (gl <= 60)].sum() / max(1, hid.sum())
    o["gaps < 1 s per half hour"] = (gl < 1).sum() / mins * 30
    vs, ve = runs(vis)
    vl = (ve - vs) / fps
    o["visible run median s"], o["visible runs < 5 s"] = _q(vl, 50), float((vl < 5).mean())
    bnd = np.minimum(52.5 - np.abs(bx), 34.0 - np.abs(by))
    with np.errstate(invalid="ignore"):
        o["visible ball outside pitch"] = float((bnd[vis] < 0).mean())
        o["visible ball < 2 m of a line or out"] = float((bnd[vis] < 2).mean())
    dteam = []
    for t in (0, 1):
        d = np.hypot(pos[:, t, :, 0] - bx[:, None], pos[:, t, :, 1] - by[:, None])
        dteam.append(np.where(np.isnan(d), np.inf, d).min(axis=1))
    dn = np.minimum(dteam[0], dteam[1])
    k = 5
    v = np.full(n, np.nan)
    v[k:] = np.hypot(bx[k:] - bx[:-k], by[k:] - by[:-k]) / (k / fps)
    vx, vy = np.diff(bx, prepend=np.nan) * fps, np.diff(by, prepend=np.nan) * fps
    acc = np.hypot(np.diff(vx, prepend=np.nan), np.diff(vy, prepend=np.nan))[vis] * fps
    o["ball accel p50 m/s2"], o["ball accel p90 m/s2"], o["ball accel p99 m/s2"] = _q(acc, 50), _q(acc, 90), _q(acc, 99)
    d1x, d1y, d2x, d2y = (np.full(n, np.nan) for _ in range(4))
    d1x[k:], d1y[k:] = bx[k:] - bx[:-k], by[k:] - by[:-k]
    d2x[:-k], d2y[:-k] = d1x[k:], d1y[k:]
    s1, s2 = np.hypot(d1x, d1y) / (k / fps), np.hypot(d2x, d2y) / (k / fps)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (d1x * d2x + d1y * d2y) / (np.hypot(d1x, d1y) * np.hypot(d2x, d2y))
        kink = (cosang < np.cos(np.radians(45))) & (np.fmax(s1, s2) > 3) & vis
    o["ball kinks (> 45 deg, > 3 m/s) per visible min"] = len(runs(kink)[0]) / vmin
    o["speed change over 0.4 s p90 m/s"] = _q(np.abs(s2 - s1)[vis], 90)
    with np.errstate(invalid="ignore"):
        o["steps > 35 m/s per visible min"] = float((np.hypot(vx, vy)[vis] > 35).sum() / vmin)
        o["ball speed median m/s (0.2 s lag)"] = _q(v[vis], 50)
        o["visible: nearest player <= 1 m"] = float((dn[vis] <= 1).mean())
    dh, da = (np.where(np.isfinite(d) & np.isfinite(bx), d, 99.0) for d in dteam)
    cx, cy, _ = clean_ball(bx, by, np.minimum(dh, da))
    th = touch_hold(cx, cy, dh, da)
    on = (th != 0) & (np.concatenate([[0.0], th[:-1]]) != th)
    o["touch onsets per visible min"] = float((on & vis).sum() / vmin)
    o["touch-hold share of visible frames"] = float((th[vis] != 0).mean())
    bi = np.flatnonzero(big)
    S, E = gs[bi], ge[bi]
    ok = (S > 0) & (E < n)
    S, E = S[ok], E[ok]
    ok = vis[S - 1] & vis[E] & (period[S - 1] == period[E])
    S, E = S[ok], E[ok]
    o["entry: ball speed median m/s"] = _q(v[S - 1], 50)
    o["exit: ball speed median m/s (0.2 s after)"] = _q(v[np.minimum(E + k, n - 1)], 50)
    o["entry: nearest player <= 1 m"] = float((dn[S - 1] <= 1).mean())
    o["exit: nearest player <= 1 m"] = float((dn[E] <= 1).mean())
    disp = np.hypot(bx[E] - bx[S - 1], by[E] - by[S - 1])
    o["block: ball displacement median m"] = _q(disp, 50)
    o["block: reappears < 5 m away"] = float((disp < 5).mean())
    p90 = edge_columns(pos)["p90"].astype(float)
    o["fast-player speed median, visible"], o["fast-player speed median, hidden"] = _q(p90[vis], 50), _q(p90[hid], 50)
    with np.errstate(invalid="ignore"):
        slow = p90 < 2.0
    o["visible frames with p90 < 2"] = float(slow[vis].mean())
    o["blocks with p90 < 2 on >= 50% of frames"] = float(np.mean([slow[s:e].mean() >= 0.5 for s, e in zip(S, E)]))
    o["blocks with p90 < 2 on < 10% of frames"] = float(np.mean([slow[s:e].mean() < 0.1 for s, e in zip(S, E)]))
    ss, se = runs(slow & inp)
    keep = (se - ss) / fps >= 5
    hs = np.array([hid[a:b].mean() for a, b in zip(ss[keep], se[keep])])
    o["slow runs >= 5 s: hidden share inside"], o["slow runs >= 5 s: fully visible"] = float(hs.mean()), float((hs < 0.05).mean())
    ch, ca = np.isfinite(pos[:, 0, :, 0]).sum(1), np.isfinite(pos[:, 1, :, 0]).sum(1)
    tot = ch + ca
    o["players tracked mean, visible"], o["players tracked mean, hidden"] = float(tot[vis].mean()), float(tot[hid].mean())
    o["hidden frames with < 16 players"] = float((tot[hid] < 16).mean())
    o["visible frames with < 16 players"] = float((tot[vis] < 16).mean())
    o["hidden frames with a team < 6"] = float(((ch < 6) | (ca < 6))[hid].mean())
    o["hidden frames with a team < 8 (no shape evidence)"] = float(((ch < 8) | (ca < 8))[hid].mean())
    o["player count changes per s"] = float(np.abs(np.diff(tot.astype(float)))[inp[1:]].sum() / (inp.sum() / fps))
    return o


def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("ml", ctypes.c_ulong)] + [(c, ctypes.c_ulonglong) for c in "abcdefg"]
    m = MS()
    m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.b / 2 ** 30


def cmd_inputs(variant, kw):
    res = {}
    run_dir = os.path.join(ROOT, "data", "challenge", "_runs")
    f = np.load(os.path.join(run_dir, "features_v1.npz"))               # provided INPUT only (ball, players)
    res["provided"] = input_stats(pipeline.challenge_period(len(f["ball_x"])), f["ball_x"], f["ball_y"],
                                  np.load(os.path.join(run_dir, "positions_v1.npz"))["pos"])
    for name in bench.SPLITS["tune"]:
        if os.name == "nt" and free_gb() < 3:
            sys.exit("free RAM below 3 GB")
        if kw and bench.VARIANTS.get(variant, {}).get("real2"):
            case, pos = bench.real2_case(name, *bench.base_ball(name), **kw)
        else:
            case = bench.get_case(name, variant=variant)
            pos = tune_shape.positions(name, case, variant)
        res[name] = input_stats(case["period"], case["ball_x"], case["ball_y"], pos)
        t, play = case["truth"], case["period"] > 0
        res[name]["(bench truth) D share of hidden / visible frames"] = "%.2f / %.2f" % (
            (t[case["hidden"] & play] == "D").mean(), (t[~case["hidden"] & play] == "D").mean())
    names = list(res)
    print("%-52s" % "metric" + " ".join("%11s" % x for x in names))
    for key in res[names[-1]]:
        print("%-52s" % key[:52] + " ".join("%11s" % (
            "-" if key not in res[x] else res[x][key] if isinstance(res[x][key], str) else "%.3f" % res[x][key])
            for x in names))
    path = os.path.join(ROOT, "eval", "results", "audit_inputs_%s%s.json" % (variant or "orig", "_scan" if kw else ""))
    with open(path, "w") as fh:
        json.dump({"variant": variant, "kw": kw, "real2": bench.REAL2, "stats": res}, fh, indent=1, default=float)
    print("wrote", os.path.relpath(path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score", "inputs"])
    ap.add_argument("--kw", nargs="*", help="inputs only: REAL2 overrides (uncached scan), e.g. p_hide=0.6")
    ap.add_argument("--variant", default="")
    ap.add_argument("--match", default=None)
    ap.add_argument("--split", default="tune", choices=["tune", "heldout"])
    a = ap.parse_args()
    names = bench.SPLITS[a.split]
    if a.cmd == "inputs":
        return cmd_inputs(a.variant, tune_shape.kv(a.kw or []))
    if a.cmd == "build":
        for name in ([a.match] if a.match else names):
            case = bench.get_case(name, variant=a.variant)
            f = bench.case_features(name, case, variant=a.variant)
            print("built", name, a.variant or "orig", "visible %.3f" % (~np.isnan(f["ball_x"][case["period"] > 0])).mean())
        return
    data = {n: load(n, a.variant) for n in names}
    sf = {n: tune_shape.shape_features(n, data[n][0]) for n in names}
    if a.split == "tune":
        models = {n: shape.fit_models([sf[m] for m in names if m != n], [data[m][0]["truth"] for m in names if m != n],
                                      l2=tune_shape.L2) for n in names}
    else:
        models = {n: shape.MODELS for n in names}
    out = {"variant": a.variant or "orig", "split": a.split, "steps": [],
           "detector": {n: detector_quality(data[n][0], data[n][2]) for n in names},
           "dead_model": bench.DEAD_MODEL if a.variant else None, "faults": bench.FAULTS if a.variant else None}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        out["commit"] = None
    for n in names:
        print("detector %-9s " % n + "  ".join("%s %.3f" % kv for kv in out["detector"][n].items()))
    for label, adr, use_shape, ctx, kw in STEPS:
        row = {"step": label, "adr": adr, "matches": {}}
        for n in names:
            case, f, feats = data[n]
            if kw is None:
                lab = pipeline.label(dict(f), decoder="rule", period=case["period"])
            else:
                fe = dict(feats)
                if use_shape:
                    st, sd = shape.emissions(sf[n], models[n], 4.0, ctx)
                    fe.update(shape_team=st, shape_dead=sd)
                lab = decode_halves(case["period"], feats=fe, **kw)
            r = pm.evaluate(pm.label_codes(case["truth"]), pm.label_codes(lab), bootstrap_n=0)
            row["matches"][n] = {"spellF1": r["spell_f1_mean"], "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
                                 "aliveDeadF1": r["alive_dead_f1"], "macroF1": r["macro_f1"]}
        row["mean"] = float(np.mean([m["spellF1"] for m in row["matches"].values()]))
        out["steps"].append(row)
        print("%-42s %.4f  (%s)" % (label, row["mean"], "  ".join(
            "%s %.3f %d/%d ad %.2f" % (n, m["spellF1"], m["n_pred"], m["n_gt"], m["aliveDeadF1"]) for n, m in row["matches"].items())))
    path = os.path.join(ROOT, "eval", "results", "audit_steps_%s%s.json" % (
        a.variant or "orig", "" if a.split == "tune" else "_HELDOUT"))
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
