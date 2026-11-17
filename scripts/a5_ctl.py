"""Envelope L4-control-definition: external truth relabelled to the organisers' written definition
of a change of possession (possess/ctl_truth.py), and decoder switches scored on both truths.

    uv run python scripts/ramlock.py -- uv run python scripts/a5_ctl.py truth
    uv run python scripts/ramlock.py -- uv run python scripts/a5_ctl.py save bench=real1|orig
    uv run python scripts/ramlock.py -- uv run python scripts/a5_ctl.py run bench=real2 seed=0 [cfg=a,b,...]
    uv run python scripts/ramlock.py -- uv run python scripts/a5_ctl.py transfer [cfg=a,b,...]

truth:    per tune match, possessions in the old truth and in the ctl truth (T = 1, 2, 3 s; T = 2 s
          or one completed pass, Metrica event feed only) -> eval/results/a5_ctl_truth.json
save:     decoder inputs of the tune matches for real1 / orig (a4_short.cmd_save, files a5_*);
          real2 seeds 0-3 are read from the a4_* files of envelope L3-A4 (never rewritten).
run:      every config in CONFIGS decoded once per match and scored against every truth; paired
          2-min block bootstrap against the default on the old truth and on ctl T = 2 s
          -> eval/results/a5_ctl_<bench>_s<seed>.json
transfer: provided match INPUT only (the ground truth is never opened): possessions per half
          hour, D share, frames changed -> eval/results/a5_ctl_transfer.json. Nothing is chosen on it.
Tune matches only; no held-out, no public truth.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import bench  # noqa: E402
import diagnose as dg  # noqa: E402
import tune_shape  # noqa: E402
from possess import ctl_truth, pipeline  # noqa: E402

pm = bench.pm
FPS = 25.0
TS = (1.0, 2.0, 3.0)
CONFIGS = {
    "default": {},
    "fl1": dict(m_flicker=1.0),                                   # T58 R1 flicker
    "fl2": dict(m_flicker=2.0),
    "fl2p": dict(m_flicker=2.0, cl_protect=1),                    # T58 with touch-backed spells protected
    "fl2sl": dict(m_flicker=2.0, m_sliver=2.0, cl_dir=2),         # + A b D -> A A D
    "cs6": dict(c_switch=6.0),
    "cs8": dict(c_switch=8.0),
    "cf1": dict(confirm_s=1.0),                                   # touch that returns is ignored
    "cf2": dict(confirm_s=2.0),
    "cf4": dict(confirm_s=4.0),
    "ctl1": dict(ctl_s=1.0),                                      # ctl relabelling on the predicted labels
    "ctl2": dict(ctl_s=2.0),
    "ctl3": dict(ctl_s=3.0),
    "cf2ctl2": dict(confirm_s=2.0, ctl_s=2.0),
    "cs6ctl2": dict(c_switch=6.0, ctl_s=2.0),
}


def _path(name, variant, seed):
    """real2: the a4_* inputs of envelope L3-A4 (read only); real1 / orig: this envelope's a5_* files."""
    return os.path.join(bench.CACHE, "%s_%s_%s_s%d.npz" % ("a4" if variant == "real2" else "a5", name, variant or "orig", seed))


a4_short._path = _path


def passes_of(name, n):
    """+1 / -1 on the start frame of every completed home / away pass (Metrica event feed: a PASS
    row is a completed pass, a failed one is BALL LOST); None for matches without frame-indexed events."""
    if not name.startswith("metrica") or name == "metrica3":
        return None
    from possess.metrica_csv import _load_events

    g = int(name[-1])
    ev = _load_events(os.path.join(bench.METRICA, "Sample_Game_%d" % g, "Sample_Game_%d_RawEventsData.csv" % g))
    out = np.zeros(n)
    for e in ev:
        if e.raw_type == "PASS" and e.team_id in (1, 2) and 0 <= e.start_frame < n:
            out[e.start_frame] = 1.0 if e.team_id == 1 else -1.0
    return out


def truths_of(name, truth, period):
    """{"old", "ctl1", "ctl2", "ctl3", "pass"}: label arrays ("pass" only with event data)."""
    out = {"old": truth}
    for t in TS:
        out["ctl%d" % t] = ctl_truth.relabel(truth, period, t, FPS)
    p = passes_of(name, len(truth))
    if p is not None:
        out["pass"] = ctl_truth.relabel(truth, period, 2.0, FPS, passes=p, min_pass=1)
    return out


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _dump(tag, res):
    res["commit"] = _commit()
    with open(os.path.join(ROOT, "eval", "results", "a5_ctl_%s.json" % tag), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


def cmd_truth(opt):
    res = {}
    for name in tune_shape.TUNE:
        truth, period, _ = a4_short.load(name, "real2", 0)
        hh = (period > 0).sum() / FPS / 1800.0
        tv = truths_of(name, truth, period)
        runs = pm.valid_runs(pm.label_codes(truth))
        L = np.array([(b - a) / FPS for a, b, _ in pm.possession_spells(pm.label_codes(truth), runs)])
        r = {"half_hours": hh, "old_lt1": int((L < 1).sum()), "old_lt2": int((L < 2).sum()), "old_lt3": int((L < 3).sum())}
        for k, v in tv.items():
            r[k] = ctl_truth.n_possessions(v, period)
            r[k + "_per_half_hour"] = r[k] / hh
            r[k + "_frames_changed"] = float((v != truth)[period > 0].mean())
        p = passes_of(name, len(truth))
        if p is not None:                                        # literal second proxy: > 2 consecutive passes
            c = {}
            ctl_truth.relabel(truth, period, 2.0, FPS, counts=c)
            c3 = {}
            ctl_truth.relabel(truth, period, 2.0, FPS, passes=p, min_pass=3, counts=c3)
            r["lt2s_spells_saved_by_3_passes"] = c.get("relabelled", 0) - c3.get("relabelled", 0)
        res[name] = r
        print(name, " ".join("%s %s" % (k, ("%.3f" % v) if isinstance(v, float) else v) for k, v in r.items()), flush=True)
    _dump("truth", res)


def cmd_run(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    cfgs = opt.pop("cfg", None)
    cfgs = cfgs.split(",") if isinstance(cfgs, str) else list(CONFIGS)
    if "default" not in cfgs:
        cfgs = ["default"] + cfgs
    v = "" if variant == "orig" else variant
    res = {"bench": variant, "seed": seed, "configs": {c: CONFIGS[c] for c in cfgs}, "scores": {}, "boot": {}}
    labs, tv, per = {}, {}, {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name], per[name] = truths_of(name, truth, period), period
        for c in cfgs:
            labs[c, name] = a4_short.decode_one(feats, period, CONFIGS[c])
        del feats
    keys = ["old", "ctl1", "ctl2", "ctl3", "pass"]
    print("%s s%d  spellF1 mean of 3 tune matches (pass: Metrica 1, 2 only) | predicted possessions | bootstrap vs default" % (variant, seed))
    print("%-8s %6s %6s %6s %6s %6s | %5s | %-26s | %-26s" % ("cfg", *keys, "npred", "d old [CI90]", "d ctl2 [CI90]"))
    for c in cfgs:
        sc = {}
        for name in tune_shape.TUNE:
            for k, t in tv[name].items():
                r = pm.evaluate(pm.label_codes(t), pm.label_codes(labs[c, name]), bootstrap_n=0)
                sc.setdefault(k, {})[name] = {"spellF1": r["spell_f1_mean"], "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
        res["scores"][c] = sc
        means = {k: float(np.mean([m["spellF1"] for m in sc[k].values()])) for k in sc}
        npred = sum(m["n_pred"] for m in sc["old"].values())
        bt = {}
        if c != "default":
            for k in ("old", "ctl2"):
                d = np.mean([np.subtract(*dg._pr_boot(tv[n][k], labs["default", n], labs[c, n])[::-1]) for n in tune_shape.TUNE], axis=0)
                bt[k] = [float(d.mean()), float(np.percentile(d, 5)), float(np.percentile(d, 95))]
        res["boot"][c] = bt
        res["scores"][c]["mean"] = means
        fb = lambda k: "%+.4f [%+.4f, %+.4f]" % tuple(bt[k]) if bt else ""   # noqa: E731
        print("%-8s %s | %5d | %-26s | %-26s" % (c, " ".join("%.4f" % means[k] for k in keys), npred, fb("old"), fb("ctl2")), flush=True)
    res["n_gt"] = {k: {n: res["scores"]["default"][k][n]["n_gt"] for n in res["scores"]["default"][k] if n != "mean"} for k in keys}
    print("true possessions:", {k: sum(v.values()) for k, v in res["n_gt"].items()})
    _dump("%s_s%d" % (variant, seed), res)


def cmd_transfer(opt):
    cfgs = opt.pop("cfg", None)
    cfgs = cfgs.split(",") if isinstance(cfgs, str) else ["ctl1", "ctl2", "ctl3", "cf2", "cf2ctl2"]
    f = pipeline.challenge_features(os.path.join(ROOT, "data", "challenge"),
                                    os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz"))
    per = pipeline.challenge_period(len(f["ball_x"]))
    play = per > 0
    hh = play.sum() / FPS / 1800.0
    res = {}
    base = None
    for c in ["default"] + [c for c in cfgs if c != "default"]:
        lab = pipeline.label(f, dec_kw=CONFIGS[c], period=per)
        base = lab if base is None else base
        n = ctl_truth.n_possessions(lab, per)
        res[c] = {"possessions": n, "per_half_hour": n / hh, "D_share": float((lab == "D")[play].mean()),
                  "H_share_of_HA": float((lab == "H")[play].sum() / max(1, np.isin(lab, ["H", "A"])[play].sum())),
                  "frames_changed": float((lab != base)[play].mean())}
        print(c, " ".join("%s %.4f" % kv_ for kv_ in res[c].items()), flush=True)
    _dump("transfer", res)


if __name__ == "__main__":
    cmd, rest = sys.argv[1], a4_short.kv(sys.argv[2:])
    {"truth": cmd_truth, "save": a4_short.cmd_save, "run": cmd_run, "transfer": cmd_transfer}[cmd](rest)
