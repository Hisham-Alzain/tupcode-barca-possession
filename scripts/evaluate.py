"""Score one pipeline variant everywhere it can be scored, in one table.

    uv run python scripts/evaluate.py --variant base
    uv run python scripts/evaluate.py --variant p1p2 --team restart_flip=1 backdate=1

Columns:
  tune      Metrica G1, G2, IDSSE 1 (external truth, ball hidden like the challenge). THE number used
            to keep or drop a change (LOOP.md protocol, RULES.md: tune on external data only).
  public5   Barcelona-Athletic public 5-minute window (organisers' public ground truth, 17
            possessions). Reported for every variant as a sanity check, NEVER used to choose:
            the organisers show it ranks versions the wrong way round, and tuning on it is
            forbidden.
Held-out matches (Metrica G3, IDSSE 2) are not touched here; see scripts/bench.py --split heldout.

Shape evidence (possess/shape.py, default since L2-7): tune matches get their player positions from
the shapepos cache and LEAVE-ONE-MATCH-OUT shape models (fitted on the other two tune matches), so
the tune number never uses a model that saw the scored match. Held-out and public5 use the shipped
shape.MODELS (fitted on the three tune matches). --no-shape gives the decoder without shape.
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

pm = bench.pm
PUBLIC_GT = os.path.join(ROOT, "third_party", "mtah", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
CHALLENGE_DIR = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data", "challenge"))
CHALLENGE_CACHE = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")


def kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = float(v) if "." in v else int(v)
    return out


def score(truth_codes, pred_labels):
    r = pm.evaluate(truth_codes, pm.label_codes(pred_labels), bootstrap_n=0)
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "n_pred": r["n_pred_spells"],
            "n_gt": r["n_gt_spells"], "aliveDeadF1": r["alive_dead_f1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--team", nargs="*", help="team_control overrides, e.g. restart_flip=1 k=5")
    ap.add_argument("--dead", nargs="*", help="deadball2 overrides, e.g. r_far=6.0")
    ap.add_argument("--decoder", default=None, help="viterbi (default, pipeline.DEC) or rule (frame rule only)")
    ap.add_argument("--dec", nargs="*", help="decoder overrides, e.g. c_switch=4.0")
    ap.add_argument("--no-public", action="store_true")
    ap.add_argument("--no-shape", action="store_true", help="decoder without shape evidence (pipeline.DEC_BASE)")
    ap.add_argument("--split", default="tune", choices=["tune", "heldout"],
                    help="heldout = a held-out LOOK: run only for a clear tune win, count it in LOOP.md")
    ap.add_argument("--bench", default="orig", choices=["orig"] + sorted(bench.VARIANTS),
                    help="tune-split ball: orig = protocol bench; real1 = realistic bench (moving dead ball + "
                         "small jumps, envelope L2-12), THE deciding bench since 2026-11-14 (tune split only); "
                         "real2 = real1 + smoothed ball, state-coupled hiding, player loss (envelope L2-20)")
    a = ap.parse_args()
    variant = "" if a.bench == "orig" else a.bench
    team_kw, dead_kw, dec_kw = kv(a.team), kv(a.dead), kv(a.dec)

    res = {"variant": a.variant, "team_kw": team_kw, "dead_kw": dead_kw, "decoder": a.decoder,
           "dec_kw": dec_kw, "bench": a.bench, "tune": {}}
    try:
        res["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                                text=True).strip()
    except Exception:
        res["commit"] = None
    use_shape = not a.no_shape and a.decoder in (None, "viterbi")
    lomo = {}
    if use_shape and a.split == "tune":                   # leave-one-match-out shape models
        sf = {n: tune_shape.shape_features(n, bench.get_case(n)) for n in tune_shape.TUNE}
        tr = {n: bench.get_case(n)["truth"] for n in tune_shape.TUNE}
        lomo = {n: shape.fit_models([sf[m] for m in sf if m != n], [tr[m] for m in sf if m != n], l2=tune_shape.L2)
                for n in tune_shape.TUNE}
    res["shape"] = "off" if not use_shape else "leave-one-match-out models" if lomo else "shape.MODELS"
    res["attack_votes"] = {}
    for name in bench.SPLITS[a.split]:
        case = bench.get_case(name, variant=variant)      # truth, period: same in every variant (hidden mask: not real2)
        f = dict(bench.case_features(name, case, variant=variant))
        if use_shape:
            f["pos"] = tune_shape.positions(name, case, variant)   # real2 (L2-20) loses players; models stay clean-fitted
            res["attack_votes"][name] = shape.attack_votes(f["pos"], case["period"])
        res["tune"][name] = score(pm.label_codes(case["truth"]), pipeline.label(
            f, dead_kw, team_kw, a.decoder, dec_kw, period=case["period"], shape_models=lomo.get(name)))
    res["tune_mean"] = float(np.mean([m["spellF1"] for m in res["tune"].values()]))

    if not a.no_public and os.path.exists(PUBLIC_GT):
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        f = pipeline.challenge_features(CHALLENGE_DIR, CHALLENGE_CACHE)
        per = pipeline.challenge_period(len(f["ball_x"]))
        if use_shape:                                     # sanity: a share near 0.5 = direction not trustworthy
            res["attack_votes"]["public5_match"] = shape.attack_votes(f["pos"], per)
            print("challenge attacking direction {period: (home sign, vote share)}:", res["attack_votes"]["public5_match"])
        else:
            f.pop("pos", None)
        rows = pipeline.to_rows(pipeline.label(f, dead_kw, team_kw, a.decoder, dec_kw, period=per))
        res["public5"] = score(gt, rows)

    line = ("%-14s " + a.split + ("" if not variant else "[" + variant + "]") + " %.4f  (%s)") % (a.variant, res["tune_mean"], "  ".join(
        "%s %.3f %d/%d" % (n, m["spellF1"], m["n_pred"], m["n_gt"]) for n, m in res["tune"].items()))
    if "public5" in res:
        p = res["public5"]
        line += "  | public5 (sanity) %.4f %d/%d" % (p["spellF1"], p["n_pred"], p["n_gt"])
    print(line)
    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "eval", "results", "eval_%s%s%s.json" % (a.variant, "_" + variant if variant else "",
                                                                         "" if a.split == "tune" else "_HELDOUT")), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
