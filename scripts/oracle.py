"""Oracle ablations on cached bench predictions: which component limits spellF1?

Each variant replaces ONE part of the prediction with the external truth (tune split only; this
is diagnosis on external data, never a submission):
  as_is          the prediction
  dead_oracle    stoppages perfect: truth D where D; on true live frames the predicted team,
                 forward-filled where the prediction said D
  team_visible   team perfect where the ball is visible (live frames), prediction elsewhere
  team_hidden    team perfect where the ball is hidden (live frames), prediction elsewhere
  team_all       team perfect on all live frames, predicted stoppages kept
  dead+team_vis  dead_oracle and team_visible together

    python scripts/oracle.py --pipeline matchlens_noghost
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench  # noqa: E402

pm = bench.pm


def ffill_team(lab):
    out, last = lab.copy(), "X"
    for i, v in enumerate(lab):
        if v in ("H", "A"):
            last = v
        elif v in ("D", "X") and last != "X":
            out[i] = last
    return out


def variants(truth, pred, case):
    live = np.isin(truth, ["H", "A"])
    vis = ~np.isnan(case["ball_x"])
    v = {"as_is": pred}
    d = np.where(truth == "D", "D", ffill_team(np.where(truth == "D", "X", pred)))
    d = np.where(live, np.where(np.isin(d, ["H", "A"]), d, "X"), d)
    v["dead_oracle"] = d
    tv = pred.copy(); m = live & vis & (pred != "D"); tv[m] = truth[m]
    v["team_visible"] = tv
    th = pred.copy(); m = live & ~vis & (pred != "D"); th[m] = truth[m]
    v["team_hidden"] = th
    ta = pred.copy(); m = live & (pred != "D"); ta[m] = truth[m]
    v["team_all"] = ta
    dt = d.copy(); m = live & vis; dt[m] = truth[m]
    v["dead+team_vis"] = dt
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="matchlens_noghost")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = {}
    for name in bench.SPLITS["tune"]:
        case = bench.get_case(name, a.seed)
        pred = np.load(os.path.join(bench.CACHE, "pred_%s_%s_s%d.npy" % (a.pipeline, name, a.seed)))
        g = pm.label_codes(case["truth"])
        for k, lab in variants(case["truth"], pred, case).items():
            r = pm.evaluate(g, pm.label_codes(lab), bootstrap_n=0)
            rows.setdefault(k, []).append((r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))
    print("pipeline=%s  tune split, spellF1 per match and mean" % a.pipeline)
    for k, v in rows.items():
        print("  %-14s %s   mean %.4f" % (k, "  ".join("%.4f (%d/%d)" % x for x in v),
                                          np.mean([x[0] for x in v])))


if __name__ == "__main__":
    main()
