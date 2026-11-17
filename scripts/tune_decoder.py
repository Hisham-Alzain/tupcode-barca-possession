"""Grid over the Viterbi decoder (decode.py, plan P3) on the TUNE split only (external data).

    uv run python scripts/tune_decoder.py          # v2 grid -> eval/results/tune_decoder_v2.json
    uv run python scripts/tune_decoder.py --v1     # the first grid -> eval/results/tune_decoder.json

v2 (envelope L2-6-decoder-fable): stoppages hard (as the frame rule), graded distance emission,
switches inside hidden-ball blocks placed at `gap_pos`. One-at-a-time around a centre, <= 30
configs, features loaded once (low-RAM laptop). Never run on the public window or challenge data.
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench  # noqa: E402
from possess import pipeline  # noqa: E402
from possess.deadball2 import detect, features as dead_features  # noqa: E402
from possess.decode import decode_halves  # noqa: E402

GRID_V1 = dict(w_dead=[1.0, 3.0], c_switch=[2.0, 4.0, 8.0], c_stop=[3.0, 8.0], c_same=[1.0, 3.0],
               r_pz=[1.0])
CENTRE = dict(hard_dead=1, soft=1, cap=1.5, d0=0.25, c_switch=4.0, c_same=1.0, gap_pos=0.5)
AROUND = dict(c_switch=[3.0, 5.0], cap=[1.0, 1.25, 2.0], d0=[0.1, 0.5], c_same=[0.0, 3.0],
              gap_pos=[-1.0, 0.0, 0.3, 0.7, 1.0], w_rule=[0.3, 1.0], hard_dead=[0])
EXTRA = [  # diagnosis: v1 emission with and without the two v2 fixes; the frame rule as the only emission
    dict(w_dead=1.0, c_switch=4.0, c_stop=8.0, c_same=1.0),
    dict(w_dead=1.0, c_switch=4.0, c_stop=8.0, c_same=1.0, hard_dead=1),
    dict(w_dead=1.0, c_switch=4.0, c_stop=8.0, c_same=1.0, hard_dead=1, gap_pos=0.5),
    dict(hard_dead=1, c_same=1.0, gap_pos=0.5, w_ctrl=0.0, w_rule=1.0, c_switch=0.5),
]


def configs(v1: bool):
    if v1:
        for vals in itertools.product(*GRID_V1.values()):
            kw = dict(zip(GRID_V1, vals))
            kw["r_dz"] = kw["r_pz"]
            yield kw
        return
    yield dict(CENTRE)
    for k, vals in AROUND.items():
        for v in vals:
            kw = dict(CENTRE, **{k: v})
            if k == "hard_dead":                       # soft stoppages need their own costs
                kw.update(w_dead=1.0, c_stop=8.0)
            yield kw
    yield from EXTRA


def main():
    v1 = "--v1" in sys.argv
    # Same inputs as pipeline.label(decoder="viterbi"), but the stoppage detector (slow, and the
    # same for every config) runs once per match instead of once per config.
    data = []
    for name in bench.SPLITS["tune"]:
        case = bench.get_case(name)
        f = bench.case_features(name, case)
        df = dead_features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["cen_home"], f["cen_away"])
        feats = {"d_home": np.nan_to_num(f["d_home"], nan=99.0), "d_away": np.nan_to_num(f["d_away"], nan=99.0),
                 "visible": ~np.isnan(f["ball_x"]), "dead": detect(df, f["ml_dead"], **pipeline.DEAD2)}
        data.append((name, case, feats))
    rows = []
    for kw in configs(v1):
        res = []
        for name, case, f in data:
            lab = decode_halves(case["period"], feats=dict(f), **kw)
            r = bench.pm.evaluate(bench.pm.label_codes(case["truth"]), bench.pm.label_codes(lab), bootstrap_n=0)
            res.append((round(r["spell_f1_mean"], 4), r["n_pred_spells"]))
        rows.append(dict(kw=kw, per_match=[x for x, _ in res], n_pred=[p for _, p in res],
                         spellF1=float(np.mean([x for x, _ in res]))))
        print(len(rows), kw, round(rows[-1]["spellF1"], 4), res, flush=True)
    rows.sort(key=lambda r: -r["spellF1"])
    out = os.path.join(ROOT, "eval", "results", "tune_decoder.json" if v1 else "tune_decoder_v2.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("TOP")
    for r in rows[:6]:
        print(r)


if __name__ == "__main__":
    main()
