"""Masking test: hide ball stretches that were actually seen, fill them, compare to the truth.

For each gap length bucket, sample stretches where the ball is known, hide the middle L frames,
run every fill method on a local window, and measure the distance (m) between the filled and the
real (hidden) ball positions. Players are passed to methods that need them (player-anchored).

    python scripts/eval_ball_fill.py --source g3          # Metrica Sample Game 3 (external)
    python scripts/eval_ball_fill.py --source challenge   # BAR-ATH tactical (automatic check)

Choosing a method or tuning its parameters must use the external result (RULES.md).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from possess import ball_fill  # noqa: E402
from possess.epts import AtdMatch  # noqa: E402

METHODS = ["linear", "cubic", "kalman", "physics", "player", "poly2", "poly3", "poly4", "poly5", "poly7",
           "log3", "log", "log30", "root2", "root3"]
BUCKETS_S = [0.5, 1, 2.5, 5, 10, 20]
FPS = 25
CTX = 60    # known frames kept on each side of the hidden stretch
EDGE = 5    # require this many known frames right next to the hidden stretch


def load(source):
    if source == "g3":
        g = os.path.join(ROOT, "data", "external", "metrica", "_sd", "data", "Sample_Game_3")
        m = AtdMatch(os.path.join(g, "Sample_Game_3_metadata.xml"),
                     os.path.join(g, "Sample_Game_3_tracking.txt"), "g3")
    else:
        d = os.path.join(ROOT, "data", "challenge", "tactical")
        graded = [(1, 11185, 90651), (2, 110808, 190197)]

        def period_of(fid):
            for per, a, b in graded:
                if a <= fid - 1 <= b:
                    return per
            return 0

        m = AtdMatch(os.path.join(d, "BAR-ATH_tactical_atd.xml"), os.path.join(d, "BAR-ATH_tactical_atd.txt"),
                     "bar-ath", period_of=period_of, frame_filter=lambda f: period_of(f) > 0)
    halves = {}
    for fid, per, tracks, present in m._raw_frames():
        h = halves.setdefault(per, {"t": [], "x": [], "y": [], "k": [], "p": []})
        h["t"].append(fid)
        h["k"].append(present)
        bx, by = tracks["ball"][1]["map_position"] if present else (np.nan, np.nan)
        h["x"].append(bx); h["y"].append(by)
        h["p"].append({k: (*r["map_position"], r["team"])
                       for grp in ("players", "goalkeepers") for k, r in tracks[grp].items()})
    for h in halves.values():
        for key in ("t", "x", "y", "k"):
            h[key] = np.asarray(h[key])
    return halves


def candidate_starts(known, L):
    """Starts s where known[s-EDGE-CTX//2 : s+L+EDGE] are all known (a real seen stretch)."""
    starts, n, i = [], len(known), 0
    while i < n:
        if not known[i]:
            i += 1
            continue
        j = i
        while j < n and known[j]:
            j += 1
        lo, hi = i + EDGE, j - L - EDGE
        if hi > lo:
            starts.extend(range(lo, hi, max(1, L // 2)))
        i = j
    return starts


def run(source, n_per_bucket, seed):
    rng = np.random.default_rng(seed)
    halves = load(source)
    res = {}
    for L_s in BUCKETS_S:
        L = int(round(L_s * FPS))
        pool = [(per, s) for per, h in halves.items() for s in candidate_starts(h["k"], L)]
        if not pool:
            res[L_s] = {"n": 0}
            continue
        picks = [pool[i] for i in rng.choice(len(pool), size=min(n_per_bucket, len(pool)), replace=False)]
        errs = {m: [] for m in METHODS}
        for per, s in picks:
            h = halves[per]
            lo, hi = max(0, s - CTX), min(len(h["t"]), s + L + CTX)
            t, x, y = h["t"][lo:hi], h["x"][lo:hi].copy(), h["y"][lo:hi].copy()
            k = h["k"][lo:hi].copy()
            a, b = s - lo, s - lo + L
            tx, ty = x[a:b].copy(), y[a:b].copy()
            k[a:b] = False
            x[a:b] = y[a:b] = np.nan
            for name in METHODS:
                f = ball_fill.get(name)
                if hasattr(f, "set_context"):
                    f.set_context(h["p"][lo:hi])
                fx, fy, filled = f.fill(t, x, y, k)
                e = np.hypot(fx[a:b] - tx, fy[a:b] - ty)
                errs[name].append(float(np.nanmean(e)))
        res[L_s] = {"n": len(picks), "pool": len(pool),
                    "mean_err_m": {m: float(np.mean(v)) for m, v in errs.items()},
                    "median_err_m": {m: float(np.median(v)) for m, v in errs.items()},
                    "p90_err_m": {m: float(np.percentile(v, 90)) for m, v in errs.items()}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["g3", "challenge"], default="g3")
    ap.add_argument("--n", type=int, default=200, help="masked stretches per gap-length bucket")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    res = run(a.source, a.n, a.seed)
    out = os.path.join(ROOT, "eval", "results", "ball_fill_masked_%s.json" % a.source)
    with open(out, "w") as fh:
        json.dump({"source": a.source, "n_per_bucket": a.n, "seed": a.seed, "ctx": CTX, "results": res},
                  fh, indent=2)
    print("source=%s  (%.0f s)  mean error in metres per hidden stretch; n = stretches tested" %
          (a.source, time.time() - t0))
    print("%-8s" % "gap" + "".join("%9s" % m for m in METHODS) + "      n")
    for L_s, r in res.items():
        if not r.get("n"):
            print("%-8s no seen stretch long enough" % ("%gs" % L_s))
            continue
        best = min(r["mean_err_m"], key=r["mean_err_m"].get)
        row = "".join(("%8.2f*" if m == best else "%9.2f") % r["mean_err_m"][m] for m in METHODS)
        print("%-8s" % ("%gs" % L_s) + row + "%7d" % r["n"])
    print("* = lowest mean error in that row; wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
