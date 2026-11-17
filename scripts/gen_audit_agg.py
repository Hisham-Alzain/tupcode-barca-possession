"""L24-generality-method: tables from eval/results/gen_audit_*.json (scripts/gen_audit_stress.py) and the
per-match numbers of eval/results/overfit_audit_summary.json -> eval/results/gen_audit_summary.json.
Reads result files only (no data loading).    uv run python scripts/gen_audit_agg.py
"""
import glob
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "results")
STEPS = [("T95", "BASE", "T95"), ("T100", "T95", "T100"), ("T112+T113", "T100", "T112"), ("w_bt", "T112", "T125"),
         ("w_bd+ed_still", "T125", "BD"), ("set-piece", "BD", "FINAL"), ("FINAL-BASE", "BASE", "FINAL")]
SRC = {"Metrica": ["metrica1", "metrica2"], "IDSSE": ["idsse1"], "all": ["metrica1", "metrica2", "idsse1"]}
TRUTHS = ("old", "ctl1", "ctl2", "ctl3")


def d(r, a, b, k, ms):
    return float(np.mean([r["scores"][b][k][m] - r["scores"][a][k][m] for m in ms]))


def main():
    out = {"per_source_fresh_seeds": {}, "runs": {}, "label_conventions": {}, "loso": {}}
    # (1) sources, from the L21 fresh-seed audit (7 benches, LOMO models), bench types weighted equally
    pm = json.load(open(os.path.join(RES, "overfit_audit_summary.json")))["per_match"]
    names = {"T95": "ADR-023 T95", "T100": "ADR-024 T100", "T112+T113": "ADR-025 T112+T113", "w_bt": "ADR-026 w_bt (LOMO)",
             "w_bd+ed_still": "ADR-028 w_bd+ed_still (LOMO)", "set-piece": "ADR-029 set-piece", "FINAL-BASE": "FINAL vs BASE (LOMO)"}
    print("(1) per source, ctl T=2, fresh seeds (L21 benches)")
    for s, _, _ in STEPS:
        row = {}
        for src, ms in SRC.items():
            cells = [pm["%s %s" % (ty, m)][names[s]]["d"] for ty in ("real2", "real1", "orig") for m in ms]
            row[src] = float(np.mean(cells))
            row[src + "_min_cell"] = float(np.min(cells))
            row[src + "_neg_cells"] = "%d/%d" % (sum(c < 0 for c in cells), len(cells))
        out["per_source_fresh_seeds"][s] = row
        print("%-14s Metrica %+.4f (min %+.4f, neg %s) | IDSSE %+.4f (min %+.4f, neg %s)" % (
            s, row["Metrica"], row["Metrica_min_cell"], row["Metrica_neg_cells"], row["IDSSE"], row["IDSSE_min_cell"], row["IDSSE_neg_cells"]))
    # (2) + (3) own runs
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, "gen_audit_*.json"))):
        r = json.load(open(p))
        if "scores" in r:
            runs[r["tag"]] = r
    print("\n(2) conditions, ctl T=2 (mean of 3 matches; M = Metrica mean, I = IDSSE 1)")
    for tag, r in runs.items():
        row = {"hidden_share": float(np.mean(list(r["hidden_share"].values()))),
               "BASE": float(np.mean(list(r["scores"]["BASE"]["ctl2"].values()))),
               "FINAL": float(np.mean(list(r["scores"]["FINAL"]["ctl2"].values()))),
               "ci90_final_vs_base": r["boot_final_vs_base_ctl2"]["avg"], "steps": {}}
        for s, a, b in STEPS:
            row["steps"][s] = {src: d(r, a, b, "ctl2", ms) for src, ms in SRC.items()}
            row["steps"][s]["per_match"] = {m: d(r, a, b, "ctl2", [m]) for m in SRC["all"]}
        out["runs"][tag] = row
        print("%-16s hid %.2f BASE %.4f FINAL %.4f [%+.3f,%+.3f] | " % (tag, row["hidden_share"], row["BASE"], row["FINAL"], *row["ci90_final_vs_base"])
              + " ".join("%s %+.4f (M %+.4f I %+.4f)" % (s, x["all"], x["Metrica"], x["IDSSE"]) for s, x in row["steps"].items()))
    print("\n(3) label conventions: all-match mean per truth")
    for tag, r in runs.items():
        out["label_conventions"][tag] = {s: {k: d(r, a, b, k, SRC["all"]) for k in TRUTHS} for s, a, b in STEPS}
        out["label_conventions"][tag]["FINAL-BASE min match"] = {k: min(d(r, "BASE", "FINAL", k, [m]) for m in SRC["all"]) for k in TRUTHS}
        print("%-16s " % tag + " | ".join("%s " % s + "/".join("%+.4f" % x[k] for k in TRUTHS) for s, x in out["label_conventions"][tag].items()))
    # (4) leave-one-source-out (Metrica matches decoded with IDSSE-only models; IDSSE 1 LOMO = Metrica-only)
    print("\n(4) leave-one-source-out, ctl T=2")
    for tag, r in runs.items():
        if not r.get("loso"):
            continue
        row = {}
        for m in SRC["all"]:
            row[m] = {"w_bt_lomo": d(r, "T112", "T125", "ctl2", [m]), "w_bt_other_source": d(r, "T112", "T125_so", "ctl2", [m]),
                      "w_bd_lomo": d(r, "T125", "BD", "ctl2", [m]), "w_bd_other_source": d(r, "T125_so", "BD_so", "ctl2", [m]),
                      "final_lomo": d(r, "BASE", "FINAL", "ctl2", [m]), "final_other_source": d(r, "BASE", "FINAL_so", "ctl2", [m])}
        out["loso"][tag] = row
        print("%-16s " % tag + " | ".join("%s bt %+.4f->%+.4f bd %+.4f->%+.4f FINAL %+.4f->%+.4f" % (
            m, x["w_bt_lomo"], x["w_bt_other_source"], x["w_bd_lomo"], x["w_bd_other_source"], x["final_lomo"], x["final_other_source"]) for m, x in row.items()))
    json.dump(out, open(os.path.join(RES, "gen_audit_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
