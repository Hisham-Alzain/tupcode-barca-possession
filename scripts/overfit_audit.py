"""Work package L21-overfit-audit: BASE (15 Nov default, shape.DEC_SHAPE_L4, tag flags off) vs each
cumulative step of 17 Nov up to FINAL, on degradation seeds never used for any choice.

    uv run python scripts/ramlock.py -- uv run python scripts/overfit_audit.py prep
    uv run python scripts/ramlock.py -- uv run python scripts/overfit_audit.py bench real2 5
    uv run python scripts/overfit_audit.py agg

prep:  leave-one-match-out balldead models on the gate pool (real2:0, real1:0, orig:0), as
       scripts/a10_vdead.py cmd_bench -> data/external/_bench/overfit_audit_bd_lomo.json
bench: builds the bench inputs if missing (a4_short.cmd_save; real1 / orig seeds > 0 through
       bench.get_case), decodes BASE, T95, T100, T112, T125, BD (= pipeline.DEC) with LOMO w_bt
       (a8_r6_models_pool3.json, the gate's models) and LOMO w_bd, and T125 / BD again with the committed
       MODEL constants (in-sample); set-piece rule on simulated tags as scripts/a12_setpiece.py (sim=base,
       cfg bh_nt = pipeline.TAGS). Scores on old truth and ctl T=2; paired 2-min block bootstrap draws.
       -> eval/results/overfit_audit_bench_<variant>_s<seed>.json (+ draws npz in the bench cache)
agg:   pooled tables -> eval/results/overfit_audit_summary.json
Tune matches only; never held-out; the public truth is never opened; nothing written under data/challenge.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "eval", "results")
CFGS = ["BASE", "T95", "T100", "T112", "T125", "BD", "FINAL", "T125_is", "BD_is", "FINAL_is"]
STEPS = [("ADR-023 T95", "BASE", "T95"), ("ADR-024 T100", "T95", "T100"), ("ADR-025 T112+T113", "T100", "T112"),
         ("ADR-026 w_bt (LOMO)", "T112", "T125"), ("ADR-028 w_bd+ed_still (LOMO)", "T125", "BD"),
         ("ADR-029 set-piece", "BD", "FINAL"), ("FINAL vs BASE (LOMO)", "BASE", "FINAL"),
         ("ADR-026 w_bt (in-sample)", "T112", "T125_is"), ("ADR-028 w_bd+ed_still (in-sample)", "T125_is", "BD_is"),
         ("FINAL vs BASE (in-sample)", "BASE", "FINAL_is")]
# ctl T=2 gains written in the ADRs (real2 = mean of the seeds quoted, real1 s0, orig s0)
CLAIMED = {"ADR-023 T95": dict(real2=0.0114, real1=0.0017, orig=0.0035),
           "ADR-024 T100": dict(real2=0.0152, real1=0.0159, orig=0.0150),
           "ADR-025 T112+T113": dict(real2=0.0141, real1=0.0105, orig=0.0132),
           "ADR-026 w_bt (LOMO)": dict(real2=0.0120, real1=0.0089, orig=0.0034),
           "ADR-028 w_bd+ed_still (LOMO)": dict(real2=0.0027, real1=0.0187, orig=0.0121),
           "ADR-029 set-piece": dict(real2=0.0019, real1=0.0100, orig=0.0021)}


def _bd_path():
    import bench
    return os.path.join(bench.CACHE, "overfit_audit_bd_lomo.json")


def cmd_prep(args):
    import a8b_r6 as R6
    import tune_shape
    tr = {}
    for b in R6.benches(R6.POOL3):
        data = R6._src_rows(b)
        for name, d in data.items():
            k = d["vis"] & (d["period"] > 0)
            tr[b, name] = (d["BX"][k][::3], (d["truth"]["ctl2"][k][::3] == "D").astype(float))
        del data
    out = {}
    for name in tune_shape.TUNE:
        m = R6._fit_bd([v for (bb, mm), v in tr.items() if mm != name])
        out[name] = {k: np.asarray(v).tolist() for k, v in m.items()}
        print("fitted bd without", name, flush=True)
    json.dump(out, open(_bd_path(), "w"))


def cmd_bench(args):
    import a4_short
    import a5_ctl
    import a8b_r6 as R6
    import a12_setpiece as A12
    import bench
    import diagnose as dg
    import tune_shape
    from possess import balldead, ballteam, decode, pipeline, setpiece as sp, shape

    variant, seed = args[0], int(args[1])
    assert variant in ("real2", "real1", "orig") and seed not in (0, 1) and not (variant == "real2" and seed < 5)
    assert pipeline.DEC == dict(pipeline.DEC_BASE, **shape.DEC_SHAPE)
    assert (pipeline.TAGS["sp"], pipeline.TAGS["sp_hid"], pipeline.TAGS["sp_take_s"]) == (1, 1, 0)
    v = "" if variant == "orig" else variant
    orig_case = dg._pr_case

    def _case(name, var, sd):
        if var == "real2":
            return orig_case(name, var, sd)
        case = bench.get_case(name, sd, var)
        return case, dict(bench.case_features(name, case, sd, var), pos=tune_shape.positions(name, case, var))

    dg._pr_case = _case
    if not all(os.path.exists(a4_short._path(n, v, seed)) for n in tune_shape.TUNE):
        a4_short.cmd_save({"bench": variant, "seed": seed})
    bdm = {n: {k: np.asarray(x) for k, x in m.items()} for n, m in json.load(open(_bd_path())).items()}
    btm = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]
    kws = {"BASE": shape.DEC_SHAPE_L4, "T95": shape.DEC_SHAPE_L8_T95, "T100": shape.DEC_SHAPE_L8_T100,
           "T112": shape.DEC_SHAPE_L8_T112, "T125": shape.DEC_SHAPE_L8_T125, "BD": shape.DEC_SHAPE}
    rates = A12.provided_rates()
    pm = a5_ctl.pm
    data = R6._src_rows((variant, seed))
    res = {"bench": variant, "seed": seed, "scores": {}, "n_pred": {}, "sp": {}, "commit": a5_ctl._commit()}
    draws = {}
    for name, d in data.items():
        per, vis, f = d["period"], d["vis"], d["feats"]
        f = {k: x for k, x in f.items() if k not in ("bt", "bd")}
        bt = np.where(vis, np.tanh(R6.logit_of(btm[name], d["X"], d["E"]) / ballteam.SCALE), 0.0)
        bd = balldead.evidence(d["BX"], vis, bdm[name])

        def dec(c, **extra):
            return decode.decode_halves(per, feats=dict(f, **extra), **dict(pipeline.DEC_BASE, **kws[c]))

        labs = {c: dec(c) for c in ("BASE", "T95", "T100", "T112")}
        labs["T125"], labs["BD"] = dec("T125", bt=bt), dec("BD", bt=bt, bd=bd)
        labs["T125_is"], labs["BD_is"] = dec("T125"), dec("BD")        # committed MODEL constants
        truth = d["truth"]["old"]
        clean = np.load(a5_ctl._path(name, "", 0))
        evs = A12.sim_events(A12.true_restarts(truth, per, clean["f_ball_x"], clean["f_ball_y"]), per, seed, rates,
                             **A12.SIMS["base"])
        for a, b in (("BD", "FINAL"), ("BD_is", "FINAL_is")):
            labs[b], info = sp.apply(labs[a], f["ball_x"], f["ball_y"], per, evs, dict(hid=1, take_s=0.0))
        res["sp"][name] = {"events": len(evs), "changed": int((labs["FINAL"] != labs["BD"]).sum())}
        for k in ("old", "ctl2"):
            t = d["truth"][k]
            for c in CFGS:
                r = pm.evaluate(pm.label_codes(t), pm.label_codes(labs[c]), bootstrap_n=0)
                res["scores"].setdefault(c, {}).setdefault(k, {})[name] = r["spell_f1_mean"]
                res["n_pred"].setdefault(c, {})[name] = r["n_pred_spells"]
                if c != "BASE":
                    b0, b1 = dg._pr_boot(t, labs["BASE"], labs[c])
                    draws["BASE|%s|%s" % (k, name)] = b0.astype(np.float32)
                    draws["%s|%s|%s" % (c, k, name)] = b1.astype(np.float32)
        print(name, " ".join("%s %.4f" % (c, res["scores"][c]["ctl2"][name]) for c in CFGS), flush=True)
    tag = "%s_s%d" % (variant, seed)
    np.savez_compressed(os.path.join(bench.CACHE, "overfit_audit_draws_%s.npz" % tag), **draws)
    json.dump(res, open(os.path.join(RES, "overfit_audit_bench_%s.json" % tag), "w"), indent=1, default=float)


def cmd_agg(args):
    import bench
    import tune_shape
    names = list(tune_shape.TUNE)
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, "overfit_audit_bench_*.json"))):
        r = json.load(open(p))
        tag = "%s_s%d" % (r["bench"], r["seed"])
        runs[tag] = (r, np.load(os.path.join(bench.CACHE, "overfit_audit_draws_%s.npz" % tag)))
    types = sorted({t.split("_")[0] for t in runs}, reverse=True)
    out = {"benches": sorted(runs), "claimed_ctl2": CLAIMED, "steps": {}, "per_bench": {}, "per_match": {}}

    def delta(tags, a, b, k, match=None):
        ms = [match] if match else names
        pt = np.mean([[runs[t][0]["scores"][b][k][m] - runs[t][0]["scores"][a][k][m] for m in ms] for t in tags])
        dr = np.mean([[runs[t][1]["%s|%s|%s" % (b, k, m)] - runs[t][1]["%s|%s|%s" % (a, k, m)] for m in ms] for t in tags], axis=(0, 1))
        return float(pt), dr

    for label, a, b in STEPS:
        row = {}
        for k in ("ctl2", "old"):
            per_type = {}
            for ty in types:
                tags = [t for t in runs if t.startswith(ty + "_")]
                pt, dr = delta(tags, a, b, k)
                per_type[ty] = (pt, dr)
                row["%s %s" % (ty, k)] = {"d": pt, "ci90": [float(np.percentile(dr, 5)), float(np.percentile(dr, 95))],
                                          "p_le0": float((dr <= 0).mean()), "n_benches": len(tags)}
            pt = float(np.mean([x[0] for x in per_type.values()]))
            dr = np.mean([x[1] for x in per_type.values()], axis=0)
            row["avg %s" % k] = {"d": pt, "ci90": [float(np.percentile(dr, 5)), float(np.percentile(dr, 95))], "p_le0": float((dr <= 0).mean())}
        if label in CLAIMED:
            row["claimed ctl2"] = dict(CLAIMED[label], avg=float(np.mean([CLAIMED[label][t] for t in types])))
        out["steps"][label] = row
        print("%-34s " % label + " | ".join("%s %+.4f [%+.4f,%+.4f] p%.3f" % (c.replace(" ctl2", ""), x["d"], x["ci90"][0], x["ci90"][1], x["p_le0"])
                                            for c, x in row.items() if c.endswith("ctl2") and c != "claimed ctl2")
              + (" | claimed " + " ".join("%s %+.4f" % (t, row["claimed ctl2"][t]) for t in types + ["avg"]) if label in CLAIMED else ""), flush=True)
        print("%-34s " % "   old truth" + " | ".join("%s %+.4f [%+.4f,%+.4f]" % (c.replace(" old", ""), x["d"], x["ci90"][0], x["ci90"][1])
                                                     for c, x in row.items() if c.endswith(" old")))
    for t in sorted(runs):
        r = runs[t][0]
        out["per_bench"][t] = {c: {k: float(np.mean([r["scores"][c][k][m] for m in names])) for k in ("ctl2", "old")} for c in CFGS}
        print("%-10s ctl2 " % t + " ".join("%s %.4f" % (c, out["per_bench"][t][c]["ctl2"]) for c in CFGS))
    for ty in types:
        tags = [t for t in runs if t.startswith(ty + "_")]
        for m in names:
            row = {}
            for label, a, b in STEPS:
                pt, dr = delta(tags, a, b, "ctl2", m)
                row[label] = {"d": pt, "ci90": [float(np.percentile(dr, 5)), float(np.percentile(dr, 95))]}
            out["per_match"]["%s %s" % (ty, m)] = row
            print("%-6s %-9s " % (ty, m) + " ".join("%+.4f" % row[s[0]]["d"] for s in STEPS[:7]) + "  | in-sample bt %+.4f bd %+.4f" % (
                row[STEPS[7][0]]["d"], row[STEPS[8][0]]["d"]))
    # Holm on the six adoption steps (one-sided bootstrap p, averaged over bench types)
    ps = sorted((out["steps"][s[0]]["avg ctl2"]["p_le0"], s[0]) for s in STEPS[:6])
    holm, ok = {}, True
    for i, (p, s) in enumerate(ps):
        ok = ok and p <= 0.05 / (len(ps) - i)
        holm[s] = {"p": p, "threshold": 0.05 / (len(ps) - i), "survives": bool(ok)}
    out["holm_avg_ctl2"] = holm
    print("Holm:", json.dumps(holm))
    json.dump(out, open(os.path.join(RES, "overfit_audit_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"prep": cmd_prep, "bench": cmd_bench, "agg": cmd_agg}[sys.argv[1]](sys.argv[2:])
