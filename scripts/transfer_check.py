"""Input-level transfer check (envelope L2-12, LOOP.md protocol amendment after L2-11).

    uv run python scripts/transfer_check.py                    # provided match + bench (original cases)
    uv run python scripts/transfer_check.py --variant real1    # ... + the realistic bench cases
    uv run python scripts/transfer_check.py --no-challenge     # bench only

For every ball-based signal the default pipeline uses, the same statistic on the provided match
(INPUT only: the cached features of the automatic tracking; the ground truth is never opened here)
and on each external bench match. Prints aggregates only; writes eval/results/audit_transfer*.json.
A signal is flagged when the provided match lies outside the bench range by more than `FLAG_REL`.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench  # noqa: E402
from possess import pipeline  # noqa: E402
from possess.ballclean import clean_ball  # noqa: E402
from possess.deadball2 import detect, features as dead_features  # noqa: E402
from possess.decode import SG7, ball_noise, touch_hold  # noqa: E402

FPS = 25.0
FLAG_REL = 0.5          # provided < bench min / 1.5 or > bench max * 1.5 (and off by > 0.02 absolute)
CHALLENGE_CACHE = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")
MATCHES = bench.SPLITS["tune"] + bench.SPLITS["heldout"]


def runs_of(mask):
    """[(start, length)] of the True runs of a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e - s


def signal_stats(f, period):
    """Aggregate statistics of the ball-based signals, inside the played halves only."""
    play = np.asarray(period) > 0
    x, y = np.asarray(f["ball_x"], float), np.asarray(f["ball_y"], float)
    vis = play & ~np.isnan(x)
    nvis = max(1, int(vis.sum()))
    mins = nvis / FPS / 60.0
    out = {"visible_share": vis.sum() / max(1, play.sum())}
    # gaps (missing ball inside a half)
    _, glen = runs_of(play & np.isnan(x))
    gs = glen / FPS
    miss = max(1, int(glen.sum()))
    out["gaps_per_half_hour"] = len(glen) / (play.sum() / FPS / 1800.0)
    out["gap_frames_in_<1s"] = glen[gs < 1].sum() / miss
    out["gap_frames_in_1-5s"] = glen[(gs >= 1) & (gs < 5)].sum() / miss
    out["gap_frames_in_>=5s"] = glen[gs >= 5].sum() / miss
    out["gap_median_s"] = float(np.median(gs)) if len(gs) else 0.0
    out["gap_p90_s"] = float(np.percentile(gs, 90)) if len(gs) else 0.0
    # speed
    df = dead_features(x, y, f["d_home"], f["d_away"], f["cen_home"], f["cen_away"])
    vlag = df["vlag"]
    ok = vis & np.isfinite(vlag)
    out["still_share_<0.3"] = float((vlag[ok] < 0.3).mean())
    out["slow_share_<1.0"] = float((vlag[ok] < 1.0).mean())
    for p in (10, 50, 90, 99):
        out["vlag_p%d" % p] = float(np.percentile(vlag[ok], p))
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    sok = vis & np.isfinite(step)
    out["frame_speed_p50"] = float(np.median(step[sok]) * FPS)
    out["frame_speed_p99"] = float(np.percentile(step[sok], 99) * FPS)
    out["jumps_>35mps_per_min"] = float((step[sok] * FPS > 35).sum() / mins)
    out["jumps_>60mps_per_min"] = float((step[sok] * FPS > 60).sum() / mins)
    # sample-and-hold: exact repeats of the previous reading
    rep = sok & (step == 0)
    out["exact_repeat_share"] = float(rep.sum() / max(1, sok.sum()))
    _, rlen = runs_of(rep)
    out["repeat_run_mean_frames"] = float(rlen.mean()) if len(rlen) else 0.0
    out["repeat_run_p90_frames"] = float(np.percentile(rlen, 90)) if len(rlen) else 0.0
    out["freezes_>=0.4s_per_min"] = float((rlen >= 10).sum() / mins)
    out["freezes_>=2s_per_min"] = float((rlen >= 50).sum() / mins)
    out["freeze_frames_share_>=0.4s"] = float(rlen[rlen >= 10].sum() / nvis)
    # piecewise-linear track (a low-rate ball upsampled to 25 fps shows as straight runs between kinks)
    acc = np.hypot(np.diff(x, 2, prepend=[np.nan, np.nan]), np.diff(y, 2, prepend=[np.nan, np.nan]))
    aok = vis & np.isfinite(acc) & (step * FPS > 0.5)
    out["linear_share_moving"] = float((acc[aok] < 1e-3).mean()) if aok.any() else 0.0
    _, llen = runs_of(aok & (acc < 1e-3))
    out["linear_run_median_frames"] = float(np.median(llen)) if len(llen) else 0.0
    out["linear_run_p90_frames"] = float(np.percentile(llen, 90)) if len(llen) else 0.0
    # position noise as the touch detector estimates it
    xs, ys = np.convolve(x, SG7, "same"), np.convolve(y, SG7, "same")
    out["ball_noise_m"] = ball_noise(x[vis], y[vis], xs[vis], ys[vis])
    # distance to the nearest player
    far = np.fmin(f["d_home"], f["d_away"])[vis]
    far = far[np.isfinite(far)]
    for p in (10, 50, 90):
        out["dnear_p%d" % p] = float(np.percentile(far, p))
    out["dnear_share_<=1m"] = float((far <= 1.0).mean())
    out["dnear_share_>9m"] = float((far > 9.0).mean())
    # touches (the emission of ADR-007), raw and after ball cleaning (ADR-008)
    dh, da = np.nan_to_num(f["d_home"], nan=99.0), np.nan_to_num(f["d_away"], nan=99.0)
    for tag, (bx, by) in (("raw", (x, y)), ("clean", clean_ball(x, y, np.minimum(dh, da))[:2])):
        th = touch_hold(bx, by, dh, da)
        on = (th != 0) & (np.concatenate([[0.0], th[:-1]]) != th)
        out["touch_onsets_per_min_%s" % tag] = float((on & vis).sum() / mins)
        out["touch_hold_share_%s" % tag] = float((th[vis] != 0).mean())
    out["ballclean_removed_share"] = float(clean_ball(x, y, np.minimum(dh, da))[2][vis].mean())
    # stoppage detector
    dead = detect(df, f["ml_dead"], **pipeline.DEAD2)
    out["dead_share_all"] = float(dead[play].mean())
    out["dead_share_visible"] = float(dead[vis].mean())
    out["ml_dead_share_visible"] = float(np.asarray(f["ml_dead"], bool)[vis].mean())
    _, dlen = runs_of(dead & play)
    out["dead_runs_per_half_hour"] = len(dlen) / (play.sum() / FPS / 1800.0)
    out["dead_run_median_s"] = float(np.median(dlen) / FPS) if len(dlen) else 0.0
    out["dead_run_max_s"] = float(dlen.max() / FPS) if len(dlen) else 0.0
    dv = dead & ok
    out["still_share_in_detector_dead"] = float((vlag[dv] < 0.3).mean()) if dv.any() else 0.0
    return out


def truth_stats(f, case):
    """Bench only: the dead ball as drawn by the bench (needs external truth)."""
    t, x = case["truth"], np.asarray(f["ball_x"], float)
    d, vis = t == "D", ~np.isnan(x)
    df = dead_features(x, np.asarray(f["ball_y"], float), f["d_home"], f["d_away"], f["cen_home"], f["cen_away"])
    v = df["vlag"]
    ok = d & vis & np.isfinite(v)
    return {"truth_D_share": float(d[case["period"] > 0].mean()),
            "truth_D_ball_visible": float(vis[d].mean()) if d.any() else 0.0,
            "truth_D_still_<0.3": float((v[ok] < 0.3).mean()) if ok.any() else 0.0,
            "truth_D_vlag_p50": float(np.median(v[ok])) if ok.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", help="bench case variant (bench.get_case), e.g. real1")
    ap.add_argument("--no-challenge", action="store_true")
    ap.add_argument("--matches", nargs="*", default=MATCHES)
    a = ap.parse_args()
    res = {}
    if not a.no_challenge:
        z = np.load(CHALLENGE_CACHE)                       # input features only (no ground truth)
        f = {k: z[k] for k in z.files}
        res["provided"] = signal_stats(f, pipeline.challenge_period(len(f["ball_x"])))
    extra = {}
    for name in a.matches:
        case = bench.get_case(name, variant=a.variant)
        f = bench.case_features(name, case, variant=a.variant)
        res[name] = signal_stats(f, case["period"])
        extra[name] = truth_stats(f, case)
    names = list(res)
    keys = list(res[names[0]])
    flags = []
    print("%-30s" % "signal" + "".join("%10s" % n[:9] for n in names) + "  flag")
    for k in keys:
        row = [res[n][k] for n in names]
        flag = ""
        if "provided" in res and len(names) > 1:
            b = row[1:]
            lo, hi, p = min(b), max(b), row[0]
            if (p < lo / (1 + FLAG_REL) - 0.005 or p > hi * (1 + FLAG_REL) + 0.005) and abs(p - np.clip(p, lo, hi)) > 0.02:
                flag = "DIFFERS"
                flags.append(k)
        print("%-30s" % k + "".join("%10.3f" % v for v in row) + "  " + flag)
    print()
    for k in list(extra[a.matches[0]]) if a.matches else []:
        print("%-30s" % k + ("%10s" % "" if "provided" in res else "") + "".join("%10.3f" % extra[n][k] for n in a.matches))
    out = {"variant": a.variant or "orig", "signals": res, "bench_truth": extra, "flagged": flags,
           "flag_rule": "provided < bench min / %.1f or > bench max * %.1f, and > 0.02 outside" % (1 + FLAG_REL, 1 + FLAG_REL)}
    path = os.path.join(ROOT, "eval", "results", "audit_transfer%s.json" % ("_" + a.variant if a.variant else ""))
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
