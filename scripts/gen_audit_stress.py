"""Work package L24-generality-method: does the METHOD generalise across sources, conditions and label
conventions? BASE (shape.DEC_SHAPE_L4, tags off) vs every cumulative step of 17 Nov up to FINAL, as
scripts/overfit_audit.py cmd_bench (same LOMO models, same simulated set-piece tags), extended with
  - all four truths (old, ctl T=1, T=2, T=3),
  - leave-one-SOURCE-out learned models (ballteam, balldead fitted on IDSSE 1 only -> used on Metrica 1, 2;
    Metrica-only -> IDSSE 1 is the existing leave-one-match-out model of IDSSE 1),
  - shifted degradations of the real2 bench built in memory (seed 5, own cache files gen_<tag>_*).

    uv run python scripts/ramlock.py -- uv run python scripts/gen_audit_stress.py loso
    uv run python scripts/ramlock.py -- uv run python scripts/gen_audit_stress.py ctl real2 5
    uv run python scripts/ramlock.py -- uv run python scripts/gen_audit_stress.py stress hide60
Tune matches only; never held-out; the public truth is never opened; nothing written under data/challenge.
-> eval/results/gen_audit_<tag>.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "eval", "results")
CFGS = ["BASE", "T95", "T100", "T112", "T125", "BD", "FINAL", "T125_so", "BD_so", "FINAL_so"]
TRUTHS = ("old", "ctl1", "ctl2", "ctl3")
POOL = [("real2", 0), ("real1", 0), ("orig", 0)]
SEED = 5
# shifted conditions: overrides of bench.REAL2 (+ module patches in _apply)
STRESS = {
    "hide20": dict(hide_frac=0.20),
    "hide60": dict(hide_frac=0.60),                    # gap-length pool doubled so the fill can reach 60%
    "short": dict(p_hide=0.0, p_hide_long=0.0),        # every hidden block 1-3 s, none tied to a whole stoppage
    "rawjit": dict(noise_m=0.004),                     # no ball low-pass, white jitter x2, teleports x2
}


def _so_path():
    import bench
    return os.path.join(bench.CACHE, "gen_audit_loso_models.json")


def cmd_loso(args):
    """ballteam + balldead fitted on IDSSE 1 rows only (three gate benches), as R6.train_xy / overfit_audit.cmd_prep."""
    import a4_short
    import a5_ctl  # noqa: F401  (sets a4_short._path for real1 / orig)
    import a8b_r6 as R6
    import a8b_run
    from possess import balldead, ballteam

    bt_x, bt_y, bd_rows = [], [], []
    for variant, seed in POOL:
        truth, period, f = a4_short.load("idsse1", "" if variant == "orig" else variant, seed)
        t = a8b_run.truths(truth, period)["ctl2"]
        vis = np.asarray(f["visible"], bool)
        X = ballteam.features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["pos"], period, f.get("shape_team"))
        BX = balldead.features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["visible"], f["pos"],
                               f.get("shape_dead"), f.get("edge_p90"))
        k = vis & np.isin(t, ["H", "A"]) & (period > 0)
        bt_x.append(X[k][::3].astype(float))
        bt_y.append((t[k][::3] == "H").astype(float))
        k = vis & (period > 0)
        bd_rows.append((BX[k][::3], (t[k][::3] == "D").astype(float)))
        print("rows", variant, seed, len(bt_y[-1]), len(bd_rows[-1][1]), flush=True)
    bt = R6.fit_lin(np.concatenate(bt_x), None, np.concatenate(bt_y))
    bd = R6._fit_bd(bd_rows)
    json.dump({"train": "idsse1 only, benches %s" % POOL, "bt": bt, "bd": {k: np.asarray(v).tolist() for k, v in bd.items()},
               "n_bt": int(sum(map(len, bt_y))), "n_bd": int(sum(len(r[1]) for r in bd_rows))}, open(_so_path(), "w"))
    print("saved", _so_path())


def _apply(tag):
    """Shift the real2 degradation in this process only (nothing in scripts/bench.py or its cache changes)."""
    import bench
    bench.REAL2.update(STRESS[tag])
    if tag == "hide60":
        bench.GAP_FRAMES = list(bench.GAP_FRAMES) + [g for g in bench.GAP_FRAMES if 25 <= g <= 1000] * 2
    if tag == "short":
        n = int(0.407 * bench.GAP_N / 50) + 200
        bench.GAP_FRAMES = [25 + (7 * i) % 51 for i in range(n)]          # 25..75 frames = 1-3 s
    if tag == "rawjit":
        bench.smooth_ball = lambda x, y, sigma: (x.copy(), y.copy())
        bench.FAULTS["jump_per_min"] = 2.0 * bench.FAULTS["jump_per_min"]


def run(variant, seed, tag, stress=None):
    import a4_short
    import a5_ctl
    import a8b_r6 as R6
    import a12_setpiece as A12
    import bench
    import diagnose as dg
    import tune_shape
    from possess import balldead, ballteam, decode, pipeline, setpiece as sp, shape

    assert pipeline.DEC == dict(pipeline.DEC_BASE, **shape.DEC_SHAPE)
    assert (pipeline.TAGS["sp"], pipeline.TAGS["sp_hid"], pipeline.TAGS["sp_take_s"]) == (1, 1, 0)
    v = "" if variant == "orig" else variant
    if stress:
        assert variant == "real2" and seed == SEED
        _apply(stress)
        a4_short._path = lambda n, vv, s: os.path.join(bench.CACHE, "gen_%s_%s_s%d.npz" % (stress, n, s))
        if not all(os.path.exists(a4_short._path(n, v, seed)) for n in tune_shape.TUNE):
            a4_short.cmd_save({"bench": variant, "seed": seed})
    else:
        assert all(os.path.exists(a4_short._path(n, v, seed)) for n in tune_shape.TUNE), "control runs reuse the audit's inputs"
    bdm = {n: {k: np.asarray(x) for k, x in m.items()} for n, m in
           json.load(open(os.path.join(bench.CACHE, "overfit_audit_bd_lomo.json"))).items()}
    btm = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]
    so = json.load(open(_so_path())) if os.path.exists(_so_path()) else None
    so_bd = {k: np.asarray(x) for k, x in so["bd"].items()} if so else None
    kws = {"BASE": shape.DEC_SHAPE_L4, "T95": shape.DEC_SHAPE_L8_T95, "T100": shape.DEC_SHAPE_L8_T100,
           "T112": shape.DEC_SHAPE_L8_T112, "T125": shape.DEC_SHAPE_L8_T125, "BD": shape.DEC_SHAPE}
    rates = A12.provided_rates()
    pm = a5_ctl.pm
    data = R6._src_rows((variant, seed))
    res = {"tag": tag, "bench": variant, "seed": seed, "stress": stress, "real2_overrides": STRESS.get(stress),
           "scores": {}, "n_pred": {}, "sp": {}, "hidden_share": {}, "boot_final_vs_base_ctl2": {},
           "loso": bool(so), "commit": a5_ctl._commit()}
    cfgs = [c for c in CFGS if so or not c.endswith("_so")]
    d0, d1 = [], []
    for name, d in data.items():
        per, vis, f = d["period"], d["vis"], d["feats"]
        f = {k: x for k, x in f.items() if k not in ("bt", "bd")}
        res["hidden_share"][name] = float(1.0 - vis[per > 0].mean())

        def dec(c, **extra):
            return decode.decode_halves(per, feats=dict(f, **extra), **dict(pipeline.DEC_BASE, **kws[c]))

        def bt_of(m):
            return np.where(vis, np.tanh(R6.logit_of(m, d["X"], d["E"]) / ballteam.SCALE), 0.0)

        bt, bd = bt_of(btm[name]), balldead.evidence(d["BX"], vis, bdm[name])
        labs = {c: dec(c) for c in ("BASE", "T95", "T100", "T112")}
        labs["T125"], labs["BD"] = dec("T125", bt=bt), dec("BD", bt=bt, bd=bd)
        truth = d["truth"]["old"]
        clean = np.load(a5_ctl._path(name, "", 0))
        evs = A12.sim_events(A12.true_restarts(truth, per, clean["f_ball_x"], clean["f_ball_y"]), per, seed, rates,
                             **A12.SIMS["base"])
        pairs = [("BD", "FINAL")]
        if so:
            # leave-one-source-out: Metrica matches get the IDSSE-only models; IDSSE 1 keeps its LOMO models
            # (which are Metrica-only by construction)
            if name.startswith("metrica"):
                bt_s, bd_s = bt_of(so["bt"]), balldead.evidence(d["BX"], vis, so_bd)
            else:
                bt_s, bd_s = bt, bd
            labs["T125_so"], labs["BD_so"] = dec("T125", bt=bt_s), dec("BD", bt=bt_s, bd=bd_s)
            pairs.append(("BD_so", "FINAL_so"))
        for a, b in pairs:
            labs[b], _ = sp.apply(labs[a], f["ball_x"], f["ball_y"], per, evs, dict(hid=1, take_s=0.0))
        res["sp"][name] = {"events": len(evs), "changed": int((labs["FINAL"] != labs["BD"]).sum())}
        for k in TRUTHS:
            t = d["truth"][k]
            for c in cfgs:
                r = pm.evaluate(pm.label_codes(t), pm.label_codes(labs[c]), bootstrap_n=0)
                res["scores"].setdefault(c, {}).setdefault(k, {})[name] = r["spell_f1_mean"]
                if k == "ctl2":
                    res["n_pred"].setdefault(c, {})[name] = r["n_pred_spells"]
        b0, b1 = dg._pr_boot(d["truth"]["ctl2"], labs["BASE"], labs["FINAL"], n_draws=1000)
        d0.append(b0)
        d1.append(b1)
        dr = b1 - b0
        res["boot_final_vs_base_ctl2"][name] = [float(np.percentile(dr, 5)), float(np.percentile(dr, 95))]
        print(name, "hid %.3f" % res["hidden_share"][name],
              " ".join("%s %.4f" % (c, res["scores"][c]["ctl2"][name]) for c in cfgs), flush=True)
    dr = np.mean(d1, axis=0) - np.mean(d0, axis=0)
    res["boot_final_vs_base_ctl2"]["avg"] = [float(np.percentile(dr, 5)), float(np.percentile(dr, 95))]
    json.dump(res, open(os.path.join(RES, "gen_audit_%s.json" % tag), "w"), indent=1, default=float)
    print("saved gen_audit_%s.json" % tag, flush=True)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "loso":
        cmd_loso(a)
    elif cmd == "ctl":
        run(a[0], int(a[1]), "ctl_%s_s%s" % (a[0], a[1]))
    elif cmd == "stress":
        run("real2", SEED, "stress_%s" % a[0], stress=a[0])
