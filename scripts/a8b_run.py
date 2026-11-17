"""Envelope L8-autonomous-fable: generic opt-in config runner on the cached decoder inputs
(a4_* real2 seeds 0-3, a5_* real1 / orig; scripts/a4_short.py, a5_ctl.py), plus the Q2 count.

    uv run python scripts/ramlock.py -- uv run python scripts/a8b_run.py restart
    uv run python scripts/ramlock.py -- uv run python scripts/a8b_run.py run bench=real2 seed=0 cfg=a,b tag=q2

restart: share of stoppages after which the OTHER team restarts, on the old external truth and on
         the ctl truth (T = 1, 2, 3 s) -> eval/results/a8b_restart.json   (naive review gap 2)
run:     default vs CONFIGS, spellF1 on old / ctl2 / ctl3 truth, per match, predicted possessions,
         paired 2-min block bootstrap vs default -> eval/results/a8b_<tag>_<bench>_s<seed>.json
Tune matches only; no held-out, no public truth, nothing of the provided match.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a5_ctl  # noqa: E402  (sets a4_short._path)
import diagnose as dg  # noqa: E402
import tune_shape  # noqa: E402
from possess import ctl_truth  # noqa: E402

pm = a5_ctl.pm
FPS = 25.0

HI = dict(ctl_hi=2.0, ctl_touch=1)                       # T95 candidate
FLY = dict(w_fly=0.7, fly_v0=4.0, fly_v1=8.0)                # T100 candidate (round 2)
HF = dict(HI, **FLY)                                         # T95 + T100
XD2 = dict(HF, ed_cross=5.0, ed_cross_strict=1, ed_xdead=1, ed_xdead_min=2.0)   # round-4 baseline (T112 + T113 bundle)
CONFIGS = {
    "default": {},
    # Q2 restart team (naive review gap 2)
    "cs0": dict(c_same=0.0),
    "nf": dict(rule_flip=0),
    "cs0nf": dict(c_same=0.0, rule_flip=0),
    "csm1": dict(c_same=-1.0),                          # mild prior FOR the same team
    "csm1nf": dict(c_same=-1.0, rule_flip=0),
    "cs05nf": dict(c_same=0.5, rule_flip=0),
    # Q3 touch hold cap (review row 11)
    # Q3 pass proxy: spells of ctl_s..ctl_hi seconds need >= ctl_touch distinct touches
    "hi2t2": dict(ctl_hi=2.0, ctl_touch=2), "hi3t2": dict(ctl_hi=3.0, ctl_touch=2),
    "hi2t1": dict(ctl_hi=2.0, ctl_touch=1), "hi2t3": dict(ctl_hi=2.0, ctl_touch=3),
    "hi15t2": dict(ctl_hi=1.5, ctl_touch=2), "hi3t3": dict(ctl_hi=3.0, ctl_touch=3),
    "hi3t1": dict(ctl_hi=3.0, ctl_touch=1), "hi4t1": dict(ctl_hi=4.0, ctl_touch=1), "hi6t1": dict(ctl_hi=6.0, ctl_touch=1),
    "hi2t1h": dict(ctl_hi=2.0, ctl_touch=1, ctl_hid=1), "hi3t1h": dict(ctl_hi=3.0, ctl_touch=1, ctl_hid=1),
    "hi4t1h": dict(ctl_hi=4.0, ctl_touch=1, ctl_hid=1), "hi2t2h": dict(ctl_hi=2.0, ctl_touch=2, ctl_hid=1),
    # Q4 (diag: under-segmentation on visible frames, missed D): on top of hi2t1
    "cs2h": dict(HI, c_switch=2.0), "cs3h": dict(HI, c_switch=3.0), "wt10h": dict(HI, w_touch=1.0),
    "wt14h": dict(HI, w_touch=1.4), "cs3wt10h": dict(HI, c_switch=3.0, w_touch=1.0),
    "st8h": dict(HI, c_stop=8.0), "st10h": dict(HI, c_stop=10.0), "sd4h": dict(HI, w_sdead=0.4), "sd5h": dict(HI, w_sdead=0.5),
    # round 2: flight direction (decode.flight_dir), on the default and on top of T95
    "fly3": dict(w_fly=0.3), "fly5": dict(w_fly=0.5), "fly10": dict(w_fly=1.0),
    "fly3h": dict(HI, w_fly=0.3), "fly5h": dict(HI, w_fly=0.5), "fly10h": dict(HI, w_fly=1.0),
    "fly5hv12": dict(HI, w_fly=0.5, fly_v0=12.0, fly_v1=20.0), "fly5hv6": dict(HI, w_fly=0.5, fly_v0=6.0, fly_v1=12.0),
    "fly10hv6": dict(HI, w_fly=1.0, fly_v0=6.0, fly_v1=12.0), "fly20h": dict(HI, w_fly=2.0), "fly20hv6": dict(HI, w_fly=2.0, fly_v0=6.0, fly_v1=12.0),
    "fly10hv4": dict(HI, w_fly=1.0, fly_v0=4.0, fly_v1=12.0), "fly10hv48": dict(HI, w_fly=1.0, fly_v0=4.0, fly_v1=8.0),
    "fly10v6": dict(w_fly=1.0, fly_v0=6.0, fly_v1=12.0), "fly10v4": dict(w_fly=1.0, fly_v0=4.0, fly_v1=12.0),
    "fly7hv48": dict(HI, w_fly=0.7, fly_v0=4.0, fly_v1=8.0), "fly15hv48": dict(HI, w_fly=1.5, fly_v0=4.0, fly_v1=8.0),
    "fly10hv26": dict(HI, w_fly=1.0, fly_v0=2.0, fly_v1=6.0), "fly10hv36": dict(HI, w_fly=1.0, fly_v0=3.0, fly_v1=6.0),
    "fly10hv610": dict(HI, w_fly=1.0, fly_v0=6.0, fly_v1=10.0), "fly10hv58": dict(HI, w_fly=1.0, fly_v0=5.0, fly_v1=8.0),
    "fly10v48": dict(w_fly=1.0, fly_v0=4.0, fly_v1=8.0), "fly7v48": dict(w_fly=0.7, fly_v0=4.0, fly_v1=8.0),
    "fly5hv48": dict(HI, w_fly=0.5, fly_v0=4.0, fly_v1=8.0), "fly5v48": dict(w_fly=0.5, fly_v0=4.0, fly_v1=8.0),
    "F": dict(HI, w_fly=0.7, fly_v0=4.0, fly_v1=8.0),                # T95 + T100 = round-2 baseline
    "F3": dict(ctl_hi=3.0, ctl_touch=1, w_fly=0.7, fly_v0=4.0, fly_v1=8.0),
    "Fn": dict(HI, w_fly=0.7, fly_v0=4.0, fly_v1=8.0, fly_neutral=1),   # needs the fly_neutral code of row 24 (not kept)
    # constants on top of F (row 26)
    "Fcs3": dict(HF, c_switch=3.0), "Fcs5": dict(HF, c_switch=5.0), "Fwt5": dict(HF, w_touch=0.5), "Fwt10": dict(HF, w_touch=1.0),
    "Fwr0": dict(HF, w_rule=0.0), "Fwr6": dict(HF, w_rule=0.6), "Fst10": dict(HF, c_stop=10.0), "Fst14": dict(HF, c_stop=14.0),
    "Fmh6": dict(HF, max_hold_s=6.0), "Fws10": dict(HF, w_shape=0.10),
    # launch gate (row 29; needs decode.launch_gate, rejected and not kept)
    "Fg1": dict(HF, fly_gate=1), "Fg2": dict(HF, fly_gate=2), "Fg1s2": dict(HF, fly_gate=1, fly_gate_s=2.0), "Fg1s8": dict(HF, fly_gate=1, fly_gate_s=8.0),
    "Fg1w10": dict(HF, fly_gate=1, w_fly=1.0), "Fg2w10": dict(HF, fly_gate=2, w_fly=1.0), "Fg1w15": dict(HF, fly_gate=1, w_fly=1.5),
    # player flow direction (row 32; needs decode.flow_dir, rejected and not kept)
    "W1": dict(HF, w_flow=0.1), "W2": dict(HF, w_flow=0.2), "W4": dict(HF, w_flow=0.4), "W8": dict(HF, w_flow=0.8),
    "W2v": dict(HF, w_flow=0.2, flow_vis=1.0), "W2a": dict(HF, w_flow=0.2, flow_v0=0.5, flow_v1=2.0), "W2b": dict(HF, w_flow=0.2, flow_v0=2.0, flow_v1=4.0),
    "W4b": dict(HF, w_flow=0.4, flow_v0=2.0, flow_v1=4.0), "W2c": dict(HF, w_flow=0.2, flow_v0=3.0, flow_v1=5.0), "W4c": dict(HF, w_flow=0.4, flow_v0=3.0, flow_v1=5.0),
    "W8c": dict(HF, w_flow=0.8, flow_v0=3.0, flow_v1=5.0),
    # second tier for fast balls (row 35; flat, code not kept)
    "X4": dict(HF, fly_x=0.43), "X4b": dict(HF, fly_x=0.43, fly_v2=8.0, fly_v3=16.0), "X8": dict(HF, fly_x=0.86), "X15": dict(HF, fly_x=1.5),
    # receive gate on the flight evidence (row 37; needs decode.receive_gate, rejected and not kept)
    "R1": dict(HF, fly_recv=1), "R2": dict(HF, fly_recv=2), "R3": dict(HF, fly_recv=3),
    "R1w3": dict(HF, fly_recv=1, fly_recv_s=3.0), "R3w3": dict(HF, fly_recv=3, fly_recv_s=3.0), "R3w10": dict(HF, fly_recv=3, w_fly=1.0),
    # distance-emission constants on top of F (row 38)
    "Fcap075": dict(HF, cap=0.75), "Fcap15": dict(HF, cap=1.5), "Fd015": dict(HF, d0=0.15), "Fd04": dict(HF, d0=0.4),
    "Fwc07": dict(HF, w_ctrl=0.7), "Fwc13": dict(HF, w_ctrl=1.3), "Fcs1": dict(HF, c_same=0.0), "Fgp": dict(HF, gap_pos=0.5),
    "Fwc16": dict(HF, w_ctrl=1.6), "Fwc20": dict(HF, w_ctrl=2.0),
    # on top of G = T95 + T100 + T108 (row 40)
    "G": dict(HF, w_ctrl=1.6), "Gf10": dict(HF, w_ctrl=1.6, w_fly=1.0), "Gf5": dict(HF, w_ctrl=1.6, w_fly=0.5), "Gwt10": dict(HF, w_ctrl=1.6, w_touch=1.0),
    "Gwt5": dict(HF, w_ctrl=1.6, w_touch=0.5), "Gcs5": dict(HF, w_ctrl=1.6, c_switch=5.0), "Gcap15": dict(HF, w_ctrl=1.6, cap=1.5),
    # round 3: kick out of a duel credited by its outgoing direction (row 42; decode.touch_hold duel_fly)
    "Fdf2": dict(HF, duel_fly=2.0), "Fdf4": dict(HF, duel_fly=4.0), "Fdf8": dict(HF, duel_fly=8.0),
    # round 3: stoppage entry at the line crossing (row 46; edges.cross_entry)
    "Fx3": dict(HF, ed_cross=3.0), "Fx5": dict(HF, ed_cross=5.0), "Fx3o0": dict(HF, ed_cross=3.0, ed_cross_off=0), "Fx2": dict(HF, ed_cross=2.0), "Fx8": dict(HF, ed_cross=8.0),
    "Fx3s": dict(HF, ed_cross=3.0, ed_cross_strict=1), "Fx5s": dict(HF, ed_cross=5.0, ed_cross_strict=1), "Fx8s": dict(HF, ed_cross=8.0, ed_cross_strict=1),
    "Fx15s": dict(HF, ed_cross=15.0, ed_cross_strict=1),
    "Gx5s": dict(HF, w_ctrl=1.6, ed_cross=5.0, ed_cross_strict=1), "x5s": dict(ed_cross=5.0, ed_cross_strict=1),
    "X": dict(HF, ed_cross=5.0, ed_cross_strict=1),                      # T112
    "Xd": dict(HF, ed_cross=5.0, ed_cross_strict=1, ed_xdead=1), "Fd": dict(HF, ed_xdead=1),   # T113 (row 48; edges.cross_dead)
    "Xd2": dict(HF, ed_cross=5.0, ed_cross_strict=1, ed_xdead=1, ed_xdead_min=2.0),              # T113 with the 2 s minimum (row 52)
    # round 4: on top of Xd2 (T95 + T100 + T112 + T113). ed_kick / ed_reap: rows 53-55, rejected, code in a8b_r4.py (not in src)
    "K5": dict(XD2, ed_kick=5.0), "K3": dict(XD2, ed_kick=3.0), "K8": dict(XD2, ed_kick=8.0), "FK5": dict(HF, ed_kick=5.0),
    "PK": dict(XD2, ed_reap=1.5, ed_kick=5.0), "P3": dict(XD2, ed_reap=3.0), "P15": dict(XD2, ed_reap=1.5), "P3v8": dict(XD2, ed_reap=3.0, ed_reap_v=8.0), "P3K": dict(XD2, ed_reap=3.0, ed_kick=5.0),
    "S1": dict(XD2, ed_still=1.0), "FS1": dict(HF, ed_still=1.0),          # row 57 (edges.still_dead)
    "B5": dict(w_bt=0.5), "B3": dict(w_bt=0.3), "B7": dict(w_bt=0.7),   # round 5, T125 (ballteam.py; bench scores need LOMO evidence: a8b_ml.py srcgate)
    "x3": dict(ed_cross=3.0), "Gx3": dict(HF, w_ctrl=1.6, ed_cross=3.0), "Gx5": dict(HF, w_ctrl=1.6, ed_cross=5.0),
    "ctl2": dict(ctl_s=2.0), "ctl15": dict(ctl_s=1.5),             # T85 reference rows
    "mh2": dict(max_hold_s=2.0), "mh3": dict(max_hold_s=3.0), "mh1": dict(max_hold_s=1.0),
}


def _runs(lab, period):
    brk = np.flatnonzero((lab[1:] != lab[:-1]) | (period[1:] != period[:-1])) + 1
    b = np.concatenate([[0], brk, [len(lab)]])
    return [(int(s), int(e), str(lab[s]), int(period[s])) for s, e in zip(b[:-1], b[1:])]


def restart_share(lab, period, min_d=0.0):
    """(other-team restarts, stoppages flanked by team spells in the same period)."""
    r = _runs(np.asarray(lab), np.asarray(period))
    oth = n = 0
    for i in range(1, len(r) - 1):
        s, e, k, p = r[i]
        if k != "D" or p <= 0 or e - s < min_d * FPS:
            continue
        a, b = r[i - 1], r[i + 1]
        if a[3] == p and b[3] == p and a[2] in "HA" and b[2] in "HA":
            n += 1
            oth += a[2] != b[2]
    return oth, n


def cmd_restart(opt):
    res = {}
    print("%-10s %-5s %-14s %-14s %-14s" % ("match", "truth", "all D", "D >= 1 s", "D >= 5 s"))
    for name in tune_shape.TUNE:
        z = np.load(a4_short._path(name, "", 0))
        truth, period = z["truth"], z["period"]
        res[name] = {}
        for key, T in (("old", 0.0), ("ctl1", 1.0), ("ctl2", 2.0), ("ctl3", 3.0)):
            t = ctl_truth.relabel(truth, period, T, FPS) if T else truth
            row = {}
            for md in (0.0, 1.0, 5.0):
                o, n = restart_share(t, period, md)
                row["min_d_%g" % md] = {"other": o, "n": n, "share": o / max(1, n)}
            res[name][key] = row
            print("%-10s %-5s " % (name, key) + " ".join("%4d/%4d %.3f" % (v["other"], v["n"], v["share"]) for v in row.values()), flush=True)
    res["commit"] = a5_ctl._commit()
    with open(os.path.join(ROOT, "eval", "results", "a8b_restart.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def score_table(labs, tv, cfgs, names, title):
    """Prints and returns {cfg: {scores, boot}} for labels labs[cfg, name] against truths tv[name][key]."""
    keys = list(tv[names[0]])
    out = {}
    print(title)
    print("%-9s " % "cfg" + " ".join("%6s" % k for k in keys) + " | %-17s | %5s | %-26s | %-26s" % ("ctl2 per match", "npred", "d old [CI90]", "d ctl2 [CI90]"))
    for c in cfgs:
        sc = {}
        for n in names:
            for k, t in tv[n].items():
                r = pm.evaluate(pm.label_codes(t), pm.label_codes(labs[c, n]), bootstrap_n=0)
                sc.setdefault(k, {})[n] = {"spellF1": r["spell_f1_mean"], "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
        means = {k: float(np.mean([m["spellF1"] for m in sc[k].values()])) for k in keys}
        npred = sum(m["n_pred"] for m in sc["old"].values())
        bt = {}
        if c != cfgs[0]:
            for k in ("old", "ctl2"):
                d = np.mean([np.subtract(*dg._pr_boot(tv[n][k], labs[cfgs[0], n], labs[c, n])[::-1]) for n in names], axis=0)
                bt[k] = [float(d.mean()), float(np.percentile(d, 5)), float(np.percentile(d, 95))]
        sc["mean"] = means
        out[c] = {"scores": sc, "boot": bt, "n_pred": npred}
        fb = lambda k: "%+.4f [%+.4f, %+.4f]" % tuple(bt[k]) if bt else ""   # noqa: E731
        print("%-9s " % c + " ".join("%.4f" % means[k] for k in keys) + " | %-17s | %5d | %-26s | %-26s" % (
            " ".join("%.3f" % sc["ctl2"][n]["spellF1"] for n in names), npred, fb("old"), fb("ctl2")), flush=True)
    return out


def truths(truth, period):
    return {"old": truth, "ctl2": ctl_truth.relabel(truth, period, 2.0, FPS), "ctl3": ctl_truth.relabel(truth, period, 3.0, FPS),
            "ctl1": ctl_truth.relabel(truth, period, 1.0, FPS)}


def cmd_run(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    base = opt.pop("base", "default")                   # round 2: base=hi2t1 scores candidates against T95
    cfgs = [c for c in str(opt.pop("cfg", ",".join(CONFIGS))).split(",") if c != base]
    cfgs = [base] + cfgs
    tag = opt.pop("tag", "x")
    labs, tv = {}, {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = truths(truth, period)
        for c in cfgs:
            labs[c, name] = a4_short.decode_one(feats, period, CONFIGS[c])
        del feats
    res = score_table(labs, tv, cfgs, list(tune_shape.TUNE), "%s s%d  spellF1 mean of the 3 tune matches" % (variant, seed))
    out = {"bench": variant, "seed": seed, "configs": {c: CONFIGS[c] for c in cfgs}, "results": res, "commit": a5_ctl._commit()}
    with open(os.path.join(ROOT, "eval", "results", "a8b_%s_%s_s%d.json" % (tag, variant, seed)), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


def _ffill_team(lab, mask):
    """Frames in `mask` take the last H / A label before them (the next one at a block start)."""
    out = np.array(lab)
    team = np.isin(out, ["H", "A"]) & ~mask
    idx = np.where(team, np.arange(len(out)), -1)
    idx = np.maximum.accumulate(idx)
    nxt = np.where(team, np.arange(len(out)), len(out))
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    src = np.where(idx >= 0, idx, np.minimum(nxt, len(out) - 1))
    out[mask] = out[src[mask]]
    return out


def _spells(lab, period):
    return [(s, e, k) for s, e, k, p in _runs(np.asarray(lab), np.asarray(period)) if p > 0 and k in "HA"]


def classify(truth, pred, period, vis, thr=0.5):
    """Why each true possession is not matched at IoU `thr` (same-team predicted spell)."""
    ps = _spells(pred, period)
    out = []
    for s, e, k in _spells(truth, period):
        seg = pred[s:e]
        L = e - s
        ov = [(min(e, b) - max(s, a), a, b) for a, b, kk in ps if kk == k and a < e and b > s]
        best = max(ov, default=(0, 0, 0))
        iou = best[0] / (max(e, best[2]) - min(s, best[1])) if best[0] else 0.0
        if iou >= thr:
            cat = "matched"
        else:
            same, dead = (seg == k).mean(), (seg == "D").mean()
            other = 1.0 - same - dead
            if same < 0.5:
                cat = "dead" if dead >= other else "wrong_team"
            elif len(ov) >= 2 and best[0] < thr * L:
                inter = seg[seg != k]
                cat = "split_by_D" if (inter == "D").mean() >= 0.5 else "split_by_team"
            elif best[2] - best[1] > L / thr:
                cat = "merged_long"                      # the predicted spell runs far beyond the true one
            else:
                cat = "edges"
        out.append((cat, L / FPS, float(vis[s:e].mean())))
    return out


def cmd_diag(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "hi2t1")
    key = opt.pop("truth", "ctl2")
    res = {}
    cats = {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = truths(truth, period)[key]
        lab = a4_short.decode_one(feats, period, CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        del feats
        play = period > 0
        f1 = lambda p: pm.evaluate(pm.label_codes(t), pm.label_codes(p), bootstrap_n=0)["spell_f1_mean"]   # noqa: E731
        tD, pD = (t == "D") & play, (lab == "D") & play
        o = {"score": f1(lab)}
        x = _ffill_team(lab, pD & ~tD)                  # D oracle: true stoppages, our team elsewhere
        x[tD] = "D"
        o["oracle_D"] = f1(x)
        x = np.array(lab)                               # D-start oracle / D-end oracle are inside oracle_D; split false / missed
        x[tD] = "D"
        o["oracle_missedD"] = f1(x)                     # only: every true D frame is D
        o["oracle_falseD"] = f1(_ffill_team(lab, pD & ~tD))   # only: no D where the truth is live
        x = np.array(lab)
        m = play & ~pD & ~tD
        x[m] = t[m]
        o["oracle_team"] = f1(x)                        # true team wherever both say live
        for nm, mk in (("vis", vis), ("hid", ~vis)):
            x = np.array(lab)
            x[play & mk] = t[play & mk]
            o["oracle_" + nm] = f1(x)
        for nm, cap in (("restart4", 4.0), ("restart_all", 1e9)):   # true team in the first possession after a true stoppage
            x = np.array(lab)
            for s0, e0, k0 in _spells(t, period):
                if s0 > 0 and t[s0 - 1] == "D":
                    e1 = min(e0, s0 + int(cap * FPS))
                    m = np.isin(x[s0:e1], ["H", "A"])
                    x[s0:e1][m] = k0
            o["oracle_" + nm] = f1(x)
        res[name] = o
        for cat, L, vs in classify(t, lab, period, vis):
            b = "<3s" if L < 3 else ("3-8s" if L < 8 else ">=8s")
            d = cats.setdefault(cat, {"n": 0, "<3s": 0, "3-8s": 0, ">=8s": 0, "hid": 0})
            d["n"] += 1
            d[b] += 1
            d["hid"] += vs < 0.5
    keys = list(res[tune_shape.TUNE[0]])
    print("%s s%d cfg=%s truth=%s" % (variant, seed, c, key))
    print("%-10s " % "match" + " ".join("%14s" % k for k in keys))
    for n in res:
        print("%-10s " % n + " ".join("%14.4f" % res[n][k] for k in keys))
    print("%-10s " % "mean" + " ".join("%14.4f" % np.mean([res[n][k] for n in res]) for k in keys))
    tot = sum(d["n"] for d in cats.values())
    print("true possessions %d; unmatched at IoU 0.5 by cause (n, share, <3 s, 3-8 s, >= 8 s, mostly hidden):" % tot)
    for cat, d in sorted(cats.items(), key=lambda kv: -kv[1]["n"]):
        print("  %-14s %4d %.3f | %4d %4d %4d | %4d" % (cat, d["n"], d["n"] / tot, d["<3s"], d["3-8s"], d[">=8s"], d["hid"]))
    with open(os.path.join(ROOT, "eval", "results", "a8b_diag_%s_s%d_%s_%s.json" % (variant, seed, c, key)), "w") as fh:
        json.dump({"oracles": res, "causes": cats, "commit": a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_miss(opt):
    """True possessions we give to the other team (cause wrong_team), mostly visible: what the inputs showed."""
    from possess import decode, pipeline

    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "hi2t1")
    rows = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        kw = {k: pipeline.DEC[k] for k in ("r_touch", "lag", "ball_clean", "smooth_touch", "vtx", "vtx_raw") if k in pipeline.DEC}
        ev = decode.touch_events(feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"], pos=feats.get("pos"), **kw)
        hold = decode.touch_hold(feats["ball_x"], feats["ball_y"], feats["d_home"], feats["d_away"], 1.0, 4.0, duel=2)
        for s, e, k in _spells(t, period):
            seg = lab[s:e]
            sg = 1.0 if k == "H" else -1.0
            if (seg == k).mean() >= 0.5 or (seg == "D").mean() >= (np.isin(seg, ["H", "A"]) & (seg != k)).mean() or vis[s:e].mean() < 0.8:
                continue
            d_own = feats["d_home" if k == "H" else "d_away"][s:e]
            d_opp = feats["d_away" if k == "H" else "d_home"][s:e]
            rows.append(((e - s) / FPS, int((ev[s:e] == sg).sum()), int((ev[s:e] == -sg).sum()), float(np.nanmin(d_own)),
                         float((d_own < 1.0).mean()), float((d_opp < 1.0).mean()), float((d_own < d_opp).mean()),
                         float((hold[s:e] == sg).mean()), str(t[s - 1]) if s else "X"))
        del feats
    r = np.array([x[:8] for x in rows])
    print("%s s%d cfg=%s: %d visible true possessions given to the other team" % (variant, seed, c, len(r)))
    print("length s p25/med/p75 %.1f %.1f %.1f" % tuple(np.percentile(r[:, 0], [25, 50, 75])))
    print("own touches detected: 0 -> %.2f, 1 -> %.2f, >=2 -> %.2f | opponent touches: 0 -> %.2f, >=1 -> %.2f" % (
        (r[:, 1] == 0).mean(), (r[:, 1] == 1).mean(), (r[:, 1] >= 2).mean(), (r[:, 2] == 0).mean(), (r[:, 2] >= 1).mean()))
    print("own team ever within 1 m %.2f; share of frames own < 1 m med %.2f, opponent < 1 m med %.2f; own nearer med %.2f" % (
        (r[:, 3] < 1.0).mean(), np.median(r[:, 4]), np.median(r[:, 5]), np.median(r[:, 6])))
    print("after: " + str({k: sum(1 for x in rows if x[8] == k) for k in "HAD"}) + " (label before the spell: D = restart)")


def cmd_rst(opt):
    """At true restarts (D >= 2 s -> team): who is right about the taker - our labels, or the ball-player geometry."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "hi2t1")
    rows = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        t = truths(truth, period)["ctl2"]
        lab = a4_short.decode_one(feats, period, CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        dh, da = feats["d_home"], feats["d_away"]
        r = _runs(t, period)
        for i in range(1, len(r)):
            s0, e0, k, p = r[i]
            if k not in "HA" or r[i - 1][2] != "D" or r[i - 1][3] != p or r[i - 1][1] - r[i - 1][0] < 2 * FPS:
                continue
            sg = 1 if k == "H" else -1
            seg = lab[s0:min(e0, s0 + 50)]
            live = seg[np.isin(seg, ["H", "A"])]
            ours = 0 if len(live) == 0 else (1 if (live == "H").mean() >= 0.5 else -1)
            row = {"sg": sg, "ours": ours, "len": (e0 - s0) / FPS, "vis0": bool(vis[max(s0 - 12, 0):s0 + 13].any())}
            for nm, a, b in (("pre1", s0 - 25, s0), ("at", s0 - 5, s0 + 6), ("post1", s0, s0 + 25), ("pre3", s0 - 75, s0)):
                a = max(a, 0)
                m = vis[a:b]
                if m.sum() < 3:
                    row[nm] = (0, 0.0)
                    continue
                h, w = np.median(dh[a:b][m]), np.median(da[a:b][m])
                row[nm] = (int(np.sign(w - h)), float(abs(w - h)), float(min(h, w)))
            rows.append(row)
        del feats
    n = len(rows)
    print("%s s%d cfg=%s: %d true restarts after D >= 2 s; first possession median %.1f s" % (variant, seed, c, n, np.median([x["len"] for x in rows])))
    print("ours: named %.2f, right when named %.3f" % (np.mean([x["ours"] != 0 for x in rows]), np.mean([x["ours"] == x["sg"] for x in rows if x["ours"] != 0])))
    for nm in ("pre3", "pre1", "at", "post1"):
        for near, marg in ((99, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 2.0), (2.0, 4.0)):
            sel = [x for x in rows if x[nm][0] != 0 and x[nm][2] <= near and x[nm][1] >= marg]
            if sel:
                ok = np.mean([x[nm][0] == x["sg"] for x in sel])
                dis = [x for x in sel if x["ours"] != 0 and x["ours"] != x[nm][0]]
                print("  %-5s nearest <= %4.1f m, margin >= %.0f m: n %3d (%.2f) right %.3f | disagrees with ours %3d, geometry right in %.2f of them" % (
                    nm, near, marg, len(sel), len(sel) / n, ok, len(dis), np.mean([x[nm][0] == x["sg"] for x in dis]) if dis else float("nan")))


def cmd_dead(opt):
    """True stoppages by length: how much of them we call D, fully missed runs, entry / exit lag; and our false D runs."""
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    c = opt.pop("cfg", "hi2t1")
    B = ((0, 2), (2, 5), (5, 15), (15, 1e9))
    acc = {b: [] for b in B}
    fd = []
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        lab = a4_short.decode_one(feats, period, CONFIGS[c])
        vis = np.asarray(feats["visible"], bool)
        del feats
        pd_ = lab == "D"
        for s0, e0, k, p in _runs(truth, period):
            if k != "D" or p <= 0:
                continue
            L = (e0 - s0) / FPS
            b = next(x for x in B if x[0] <= L < x[1])
            d = np.flatnonzero(pd_[s0:e0])
            ent = ext = np.nan
            if len(d):
                i, j = s0 + d[0], s0 + d[-1] + 1
                while i > 0 and pd_[i - 1]:
                    i -= 1
                while j < len(lab) and pd_[j]:
                    j += 1
                ent, ext = (i - s0) / FPS, (j - e0) / FPS
            acc[b].append((L, len(d) / (e0 - s0), len(d) == 0, ent, ext, vis[s0:e0].mean()))
        for s0, e0, k, p in _runs(lab, period):
            if k == "D" and p > 0:
                fd.append(((e0 - s0) / FPS, (truth[s0:e0] == "D").mean()))
    print("%s s%d cfg=%s  true stoppages" % (variant, seed, c))
    print("length s    n  frames  D-recall  missed  missed_fr  visible | entry lag p25/med/p75 | exit lag p25/med/p75")
    tot = sum(sum(x[0] for x in a) for a in acc.values())
    for b, a in acc.items():
        a = np.array(a)
        ok = ~np.isnan(a[:, 3])
        print("%4g-%-5g %4d  %.3f   %.3f    %.3f   %.3f     %.2f   | %+5.1f %+5.1f %+5.1f | %+5.1f %+5.1f %+5.1f" % (
            b[0], min(b[1], 999), len(a), a[:, 0].sum() / tot, (a[:, 0] * a[:, 1]).sum() / a[:, 0].sum(), a[:, 2].mean(),
            (a[:, 0] * a[:, 2]).sum() / tot, a[:, 5].mean(), *np.percentile(a[ok, 3], [25, 50, 75]), *np.percentile(a[ok, 4], [25, 50, 75])))
    fd = np.array(fd)
    print("our D runs: %d; with < 10 %% true D: %d (%.1f s total, median %.1f s); frame precision %.3f" % (
        len(fd), (fd[:, 1] < 0.1).sum(), fd[fd[:, 1] < 0.1, 0].sum(), np.median(fd[fd[:, 1] < 0.1, 0]) if (fd[:, 1] < 0.1).any() else 0,
        (fd[:, 0] * fd[:, 1]).sum() / fd[:, 0].sum()))


def cmd_sp(opt):
    """Provided INPUT only (no truth): where our D -> live transitions sit against the START of each SET PIECES
    clip, and whether our first team after it is the tagged taker (anchor = nearest transition, not the
    last stoppage in the clip as in smarttags.restart_frame)."""
    import a8_tags
    from possess import smarttags as st

    c = opt.pop("cfg", "default")
    tags, home, f, per, base = a8_tags.challenge_default()
    from possess import pipeline
    lab = base if c == "default" else pipeline.label(f, period=per, dec_kw=CONFIGS[c])
    w = float(opt.pop("w", 0))
    if w:                                               # tagged restart taker with the clip-start anchor (row 18)
        extra, dec = pipeline.challenge_tag_inputs(a8_tags.D, len(per), dict(w_tagr=w, tagr_hold_s=float(opt.pop("hold", 1.5))))
        lab = pipeline.label(f, period=per, dec_kw=dict(CONFIGS[c], **dec, tagr_anchor=1), extra=extra)
        play = per > 0
        print("w_tagr=%g: frames changed %.4f, possessions %d -> %d, home share %.4f -> %.4f" % (
            w, ((lab != base) & play).sum() / play.sum(), ctl_truth.n_possessions(base, per), ctl_truth.n_possessions(lab, per),
            (base == "H")[play].sum() / np.isin(base, ["H", "A"])[play].sum(), (lab == "H")[play].sum() / np.isin(lab, ["H", "A"])[play].sum()))
    vis = ~np.isnan(f["ball_x"])
    ex = np.flatnonzero((lab[:-1] == "D") & np.isin(lab[1:], ["H", "A"])) + 1      # first live index after a D run
    rows = {}
    for t in tags:
        if t["code"] != "SET PIECES":
            continue
        a = t["a"] + 1
        sg = st.team_sign(t, home)
        near = ex[(ex >= a - 10 * 25) & (ex <= a + 20 * 25)]
        r = rows.setdefault(t["type"] or "?", {"n": 0, "off": [], "agree": [], "vis": [], "len": [], "live_at": 0})
        r["n"] += 1
        r["live_at"] += lab[min(a + 75, len(lab) - 1)] != "D"
        if len(near) == 0:
            continue
        j = int(near[np.argmin(np.abs(near - (a + 75)))])
        r["off"].append((j - a) / FPS)
        r["vis"].append(bool(vis[j]))
        e = j
        while e < len(lab) and lab[e] == lab[j]:
            e += 1
        r["len"].append((e - j) / FPS)
        if sg:
            r["agree"].append((1 if lab[j] == "H" else -1) == sg)
    print("cfg=%s  SET PIECES clips: our D->live transition nearest to clip start + 3 s (search -10..+20 s)" % c)
    print("%-22s %3s %5s | offset from clip start p10/p25/med/p75/p90 s | ball visible | first spell med s | first team = taker" % ("type", "n", "found"))
    for k, r in sorted(rows.items()):
        o = r["off"]
        print("%-22s %3d %5d | %s | %.2f | %.1f | %s" % (
            k, r["n"], len(o), " ".join("%+5.1f" % x for x in np.percentile(o, [10, 25, 50, 75, 90])) if o else "-",
            np.mean(r["vis"]) if o else 0, np.median(r["len"]) if o else 0,
            "%d / %d" % (sum(r["agree"]), len(r["agree"])) if r["agree"] else "-"))


def cmd_transfer(opt):
    """Provided match INPUT only (no truth): what a config changes in our own output."""
    import a8_tags
    from possess import pipeline

    cfgs = str(opt.pop("cfg")).split(",")
    tag = opt.pop("tag", "x")
    _, _, f, per, base = a8_tags.challenge_default()
    play = per > 0
    vis = ~np.isnan(f["ball_x"])
    hh = play.sum() / FPS / 1800.0

    def row(lab):
        n = ctl_truth.n_possessions(lab, per)
        ch = (lab != base) & play
        return {"possessions": n, "per_half_hour": n / hh, "mean_s": float(np.isin(lab, ["H", "A"])[play].sum() / FPS / max(1, n)),
                "D_share": float((lab == "D")[play].mean()),
                "home_share_of_HA": float((lab == "H")[play].sum() / max(1, np.isin(lab, ["H", "A"])[play].sum())),
                # our own output inside the public 5-min window (frames 11185-18684; the truth is NOT opened; reported, never used to choose)
                "window_possessions": ctl_truth.n_possessions(lab[11186:18686], per[11186:18686]),
                "window_D_frames": int((lab[11186:18686] == "D").sum()),
                "frames_changed": float(ch.sum() / play.sum()), "changed_on_hidden": float((ch & ~vis).sum() / max(1, ch.sum()))}

    res = {"default": row(base)}
    print("default", json.dumps(res["default"]))
    for c in cfgs:
        res[c] = row(pipeline.label(f, period=per, dec_kw=CONFIGS[c]))
        print(c, json.dumps(res[c]), flush=True)
    res["commit"] = a5_ctl._commit()
    with open(os.path.join(ROOT, "eval", "results", "a8b_%s_transfer.json" % tag), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    cmd, rest = sys.argv[1], a4_short.kv(sys.argv[2:])
    {"restart": cmd_restart, "run": cmd_run, "transfer": cmd_transfer, "diag": cmd_diag, "miss": cmd_miss, "rst": cmd_rst, "dead": cmd_dead, "sp": cmd_sp}[cmd](rest)
