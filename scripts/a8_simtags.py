"""Envelope L6-smart-tagging: a Smart Tagging SIMULATOR for the external matches, and the sweep of
the three opt-in tag mechanisms on it (the external data has no tags; the provided match's truth
is never opened, nothing is tuned on the provided match).

    uv run python scripts/ramlock.py -- uv run python scripts/a8_simtags.py stats [bench=real2] [seed=0] [sim=a,b]
    uv run python scripts/ramlock.py -- uv run python scripts/a8_simtags.py run bench=real2 seed=0 [sim=base] [cfg=a,b] [tag=name]
    uv run python scripts/ramlock.py -- uv run python scripts/a8_simtags.py transfer [cfg=a,b]

Simulator (simulate_phases): team-phase clips are drawn until they cover `cover` of the played
frames (provided file: 0.68). A clip is anchored on a random live frame of the external truth (so
long possessions are tagged more often), speaks for the team in possession there, and has a length
resampled from the provided file's own team-phase clip lengths (an INPUT statistic). Knobs, every
one a simulator ASSUMPTION that the report varies:
  anchor   uniform: the anchor sits anywhere in the clip (clip padding before and after, edges are
           not possession edges); center: in the middle; phase: the clip is the true team phase
           plus `pad` seconds on both sides (optimistic: clip edges = possession edges)
  merge    the truth the tagger "sees": spells shorter than `merge` s absorbed (ctl_truth.relabel)
  p_wrong  share of clips given to the wrong team
  jitter   clip shifted by N(0, jitter) seconds (mis-timing)
  cover    lower = dropped tags
simulate_restarts: set-piece clips with a taker around true restarts (D -> live in the truth),
`rate` per half hour (provided file: 27 team-tagged set pieces), the restart placed inside the
clip as measured on the provided input (a8_tags.py: restart found by our stoppages from the clip
start), wrong taker with p_wrong, shifted with jitter.

stats:    per simulator setting, the same INPUT statistics a8_tags.py prints for the provided file
          (coverage, both-team share, agreement with OUR default labels on visible / hidden
          frames, our D share inside clips) + what only the external truth can tell (clip purity).
          The setting whose input statistics are closest to the provided file is the calibrated one.
run:      default vs every config, old truth and ctl T = 2 s truth (ADR-022), paired 2-min block
          bootstrap vs default -> eval/results/a8_sim_<bench>_s<seed>_<sim>.json
transfer: provided INPUT only, real tags: frames changed, possessions per half hour, home share.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a5_ctl  # noqa: E402  (sets a4_short._path: a4_* real2 inputs, a5_* real1 / orig inputs)
import diagnose as dg  # noqa: E402
import tune_shape  # noqa: E402
from possess import ctl_truth, smarttags as st  # noqa: E402

pm = a5_ctl.pm
FPS = 25.0
TAGS_XML = os.path.join(ROOT, "data", "challenge", "tactical", "BAR-ATH_tactical_smart_tagging.xml")
INV = os.path.join(ROOT, "eval", "results", "a8_tags_inventory.json")

SIMS = {
    "base": dict(),                                              # uniform anchor, ctl 2 s truth, no errors
    "center": dict(anchor="center"),
    "phase": dict(anchor="phase"),                               # optimistic: clip = true phase + 3 s pads
    "raw": dict(merge=0.0),                                      # tagger sees every one-touch spell
    "wrong10": dict(p_wrong=0.10),
    "wrong20": dict(p_wrong=0.20),
    "jit3": dict(jitter=3.0),
    "jit6": dict(jitter=6.0),
    "drop": dict(cover=0.45),
    "pess": dict(p_wrong=0.20, jitter=3.0),
    "cal": dict(),                                               # filled after `stats` (see CAL below)
}
# Calibrated setting: TO BE chosen by `stats` so that the simulated clips reproduce the provided file's
# INPUT statistics (agreement with our default labels on visible frames, both-team share).
# PLACEHOLDER below: `stats` and a8_tags.py have not run yet (machine short of RAM, 2026-11-16).
CAL = dict(p_wrong=0.10, jitter=3.0)
SIMS["cal"] = CAL

CONFIGS = {
    "default": {},
    # (a) team-phase prior, hidden frames only
    "h02": dict(w_tag=0.02), "h05": dict(w_tag=0.05), "h10": dict(w_tag=0.10), "h20": dict(w_tag=0.20),
    # (a) everywhere
    "v02": dict(w_tag=0.02, tag_vis=1.0), "v05": dict(w_tag=0.05, tag_vis=1.0), "v10": dict(w_tag=0.10, tag_vis=1.0),
    # (a) clip ends trimmed by 3 s
    "h05t": dict(w_tag=0.05, trim=3.0), "v05t": dict(w_tag=0.05, tag_vis=1.0, trim=3.0),
    # (b)+(c) tagged restart taker
    "r05": dict(w_tagr=0.5), "r10": dict(w_tagr=1.0), "r20": dict(w_tagr=2.0),
    "rall10": dict(w_tagr=1.0, sp_rate=1e9),                     # every true restart tagged (upper bound)
    # L8: D-neutral tag term (tag_neutral=1)
    "h05n": dict(w_tag=0.05, tag_neutral=1), "h10n": dict(w_tag=0.10, tag_neutral=1), "h20n": dict(w_tag=0.20, tag_neutral=1),
    "h40n": dict(w_tag=0.40, tag_neutral=1),
    "v02n": dict(w_tag=0.02, tag_vis=1.0, tag_neutral=1), "v05n": dict(w_tag=0.05, tag_vis=1.0, tag_neutral=1),
    "v10n": dict(w_tag=0.10, tag_vis=1.0, tag_neutral=1), "v20n": dict(w_tag=0.20, tag_vis=1.0, tag_neutral=1),
    "r10n": dict(w_tagr=1.0, tag_neutral=1), "rall10n": dict(w_tagr=1.0, sp_rate=1e9, tag_neutral=1),
    # L8: shorter / softer restart hold, every true restart tagged (upper bound of the restart mechanism)
    "rallh1": dict(w_tagr=1.0, sp_rate=1e9, tagr_hold_s=1.0), "rallw3h1": dict(w_tagr=0.3, sp_rate=1e9, tagr_hold_s=1.0),
    "rallw3h2": dict(w_tagr=0.3, sp_rate=1e9, tagr_hold_s=2.0), "rallw1h4": dict(w_tagr=0.1, sp_rate=1e9),
    # L8 row 18: anchor on the clip start; simulated clips open 3-8 s before the true restart (sp_lohi)
    "a03h15": dict(w_tagr=0.3, tagr_hold_s=1.5, tagr_anchor=1, sp_lohi=(3.0, 8.0)),
    "a10h15": dict(w_tagr=1.0, tagr_hold_s=1.5, tagr_anchor=1, sp_lohi=(3.0, 8.0)),
    "a03h4": dict(w_tagr=0.3, tagr_anchor=1, sp_lohi=(3.0, 8.0)),
    "a10h4": dict(w_tagr=1.0, tagr_anchor=1, sp_lohi=(3.0, 8.0)),
    "aall03h15": dict(w_tagr=0.3, tagr_hold_s=1.5, tagr_anchor=1, sp_lohi=(3.0, 8.0), sp_rate=1e9),
    "aall10h15": dict(w_tagr=1.0, tagr_hold_s=1.5, tagr_anchor=1, sp_lohi=(3.0, 8.0), sp_rate=1e9),
    "aall10h4": dict(w_tagr=1.0, tagr_anchor=1, sp_lohi=(3.0, 8.0), sp_rate=1e9),
    # combined
    "h05r10": dict(w_tag=0.05, w_tagr=1.0), "v02r10": dict(w_tag=0.02, tag_vis=1.0, w_tagr=1.0),
}


def provided_lens():
    """Lengths (frames) of the provided file's team-phase clips, and the coverage target."""
    tags = st.load_tags(TAGS_XML)
    return np.array([t["b"] - t["a"] for t in tags if t["code"] in st.IN_POSS + st.OUT_POSS and t["team"]])


def provided_restart():
    """(clip length frames, lo, hi seconds of the restart from the clip start, rate per half hour)
    for team-tagged set pieces of the provided file (a8_tags.py inventory, INPUT only)."""
    if not os.path.exists(INV):                    # a8_tags.py not run yet: stated fallback, not measured
        return 875, 5.0, 25.0, 7.6                 # 35 s clip (provided SET PIECES length), 27 takers / 3.5 half hours
    inv = json.load(open(INV))
    sp = [v for k, v in inv["set_pieces"].items() if v["with_team"] > 0]
    lo = float(np.mean([v["restart_from_start_s"][0] for v in sp]))
    hi = float(np.mean([v["restart_from_start_s"][2] for v in sp]))
    ln = float(np.median([v["len_med"] for v in sp]))
    n = sum(inv["restart_events_with_team"].values())
    hh = sum(b - a + 1 for _, a, b in __import__("possess.pipeline", fromlist=["x"]).HALVES) / FPS / 1800.0
    return int(ln * FPS), lo, hi, n / hh


def _runs(lab, period):
    """[(a, b, label)] maximal runs inside each period block."""
    brk = np.flatnonzero((lab[1:] != lab[:-1]) | (period[1:] != period[:-1])) + 1
    bounds = np.concatenate([[0], brk, [len(lab)]])
    return [(int(a), int(b), str(lab[a])) for a, b in zip(bounds[:-1], bounds[1:]) if period[a] > 0]


def simulate_phases(truth, period, seed, lens, cover=0.68, anchor="uniform", merge=2.0, p_wrong=0.0,
                    jitter=0.0, pad=3.0):
    """[(first frame, last frame, sign)] simulated team-phase clips (module docstring)."""
    rng = np.random.default_rng(1000 + seed)
    ph = ctl_truth.relabel(truth, period, merge, FPS) if merge > 0 else np.asarray(truth)
    play = np.asarray(period) > 0
    live = np.flatnonzero(play & np.isin(ph, ["H", "A"]))
    n = len(ph)
    run_of = np.zeros(n, int)
    runs = _runs(ph, np.asarray(period))
    for i, (a, b, _) in enumerate(runs):
        run_of[a:b] = i
    covered = np.zeros(n, bool)
    target, clips = cover * play.sum(), []
    while covered[play].sum() < target and len(clips) < 5000:
        t = int(rng.choice(live))
        s = 1 if ph[t] == "H" else -1
        L = int(rng.choice(lens))
        if anchor == "phase":
            ra, rb, _ = runs[run_of[t]]
            a, L = ra - int(pad * FPS), rb - ra + 2 * int(pad * FPS)
        elif anchor == "center":
            a = t - L // 2
        else:
            a = t - int(rng.uniform(0, 1) * L)
        if jitter:
            a += int(rng.normal(0, jitter * FPS))
        blk = np.flatnonzero(np.asarray(period) == period[t])
        a, b = max(a, int(blk[0])), min(a + L, int(blk[-1]))
        if b <= a:
            continue
        if rng.uniform() < p_wrong:
            s = -s
        clips.append((a, b, s))
        covered[a:b + 1] = True
    return clips


def simulate_restarts(truth, period, seed, clip_len, lo, hi, rate, p_wrong=0.0, jitter=0.0, **_):
    """[{a, b, sign}] simulated set-piece clips with a taker around true restarts."""
    rng = np.random.default_rng(2000 + seed)
    truth, period = np.asarray(truth), np.asarray(period)
    rs = [(b, 1 if truth[b] == "H" else -1) for a, b, k in _runs(truth, period)
          if k == "D" and b - a >= FPS and b < len(truth) and period[b] == period[a] and truth[b] in ("H", "A")]
    hh = (period > 0).sum() / FPS / 1800.0
    k = min(len(rs), int(round(rate * hh)))
    out = []
    for i in sorted(rng.choice(len(rs), k, replace=False)) if k else []:
        j, s = rs[i]
        a = j - int(rng.uniform(lo, hi) * FPS) + (int(rng.normal(0, jitter * FPS)) if jitter else 0)
        out.append({"a": a, "b": a + clip_len, "sign": -s if rng.uniform() < p_wrong else s})
    return out


def sim_feats(truth, period, seed, sim, lens, rst, trim=0.0, sp_rate=None):
    kw = dict(SIMS[sim])
    clips = simulate_phases(truth, period, seed, lens, **kw)
    ev = simulate_restarts(truth, period, seed, rst[0], rst[1], rst[2], sp_rate or rst[3], **kw)
    return clips, {"tag_team": st.prior_from_clips(clips, len(truth), 0, trim), "tag_events": ev}


def decode_cfg(feats, period, truth, seed, sim, lens, rst, cfg):
    over = dict(CONFIGS[cfg])
    trim, sp_rate = over.pop("trim", 0.0), over.pop("sp_rate", None)
    lohi = over.pop("sp_lohi", None)
    if lohi:
        rst = (rst[0], lohi[0], lohi[1], rst[3])
    _, extra = sim_feats(truth, period, seed, sim, lens, rst, trim, sp_rate)
    return a4_short.decode_one(dict(feats, **extra), period, over)


def _opts(opt):
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    return variant, ("" if variant == "orig" else variant), seed


def cmd_stats(opt):
    variant, v, seed = _opts(opt)
    sims = opt.pop("sim", None)
    sims = sims.split(",") if isinstance(sims, str) else [s for s in SIMS]
    lens, res = provided_lens(), {}
    acc = {s: [] for s in sims}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        lab = a4_short.decode_one(feats, period)
        vis = np.asarray(feats["visible"], bool)
        del feats
        play = period > 0
        ctl = ctl_truth.relabel(truth, period, 2.0, FPS)
        ours = np.where(lab == "H", 1, np.where(lab == "A", -1, 0))
        tru = np.where(ctl == "H", 1, np.where(ctl == "A", -1, 0))
        for s in sims:
            clips = simulate_phases(truth, period, seed, lens, **SIMS[s])
            pr = st.prior_from_clips(clips, len(truth))
            pos = st.prior_from_clips([c for c in clips if c[2] > 0], len(truth)) != 0
            neg = st.prior_from_clips([c for c in clips if c[2] < 0], len(truth)) != 0
            one = (np.abs(pr) == 1) & play
            r = {"clips": len(clips), "coverage": float((pos | neg)[play].mean()), "both": float((pos & neg)[play].mean())}
            for nm, m in (("vis", vis), ("hid", ~vis)):
                lv = one & m & (ours != 0)
                tv = one & m & (tru != 0)
                r["agree_ours_" + nm] = float((ours[lv] == pr[lv]).mean())
                r["purity_truth_" + nm] = float((tru[tv] == pr[tv]).mean())
                r["oursD_in_clip_" + nm] = float((lab == "D")[one & m].mean())
                r["ours_right_" + nm] = float((ours[lv & (tru != 0)] == tru[lv & (tru != 0)]).mean())
            acc[s].append(r)
    keys = list(acc[sims[0]][0])
    print("%s s%d  mean of the 3 tune matches" % (variant, seed))
    print("%-8s " % "sim" + " ".join("%9s" % k[-9:] for k in keys))
    for s in sims:
        res[s] = {k: float(np.mean([r[k] for r in acc[s]])) for k in keys}
        res[s]["per_match"] = acc[s]
        print("%-8s " % s + " ".join("%9.3f" % res[s][k] for k in keys), flush=True)
    with open(os.path.join(ROOT, "eval", "results", "a8_simstats_%s_s%d.json" % (variant, seed)), "w") as fh:
        json.dump({"commit": a5_ctl._commit(), "sims": {s: SIMS[s] for s in sims}, "stats": res}, fh, indent=1, default=float)


def cmd_run(opt):
    variant, v, seed = _opts(opt)
    sim = opt.pop("sim", "cal")
    cfgs = opt.pop("cfg", None)
    cfgs = cfgs.split(",") if isinstance(cfgs, str) else list(CONFIGS)
    if "default" not in cfgs:
        cfgs = ["default"] + cfgs
    tag = opt.pop("tag", "")
    lens, rst = provided_lens(), provided_restart()
    res = {"bench": variant, "seed": seed, "sim": sim, "sim_kw": SIMS[sim], "restart_sim": rst,
           "configs": {c: CONFIGS[c] for c in cfgs}, "scores": {}, "boot": {}}
    labs, tv = {}, {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        tv[name] = {"old": truth, "ctl2": ctl_truth.relabel(truth, period, 2.0, FPS)}
        for c in cfgs:
            labs[c, name] = decode_cfg(feats, period, truth, seed, sim, lens, rst, c)
        del feats
    print("%s s%d sim=%s  spellF1 (3 tune matches) | predicted possessions | paired bootstrap vs default" % (variant, seed, sim))
    print("%-8s %6s %6s | %-20s | %5s | %-26s | %-26s" % ("cfg", "old", "ctl2", "ctl2 per match", "npred", "d old [CI90]", "d ctl2 [CI90]"))
    for c in cfgs:
        sc = {}
        for name in tune_shape.TUNE:
            for k, t in tv[name].items():
                r = pm.evaluate(pm.label_codes(t), pm.label_codes(labs[c, name]), bootstrap_n=0)
                sc.setdefault(k, {})[name] = {"spellF1": r["spell_f1_mean"], "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
        means = {k: float(np.mean([m["spellF1"] for m in sc[k].values()])) for k in sc}
        npred = sum(m["n_pred"] for m in sc["old"].values())
        bt = {}
        if c != "default":
            for k in ("old", "ctl2"):
                d = np.mean([np.subtract(*dg._pr_boot(tv[n][k], labs["default", n], labs[c, n])[::-1]) for n in tune_shape.TUNE], axis=0)
                bt[k] = [float(d.mean()), float(np.percentile(d, 5)), float(np.percentile(d, 95))]
        sc["mean"] = means
        res["scores"][c], res["boot"][c] = sc, bt
        fb = lambda k: "%+.4f [%+.4f, %+.4f]" % tuple(bt[k]) if bt else ""   # noqa: E731
        print("%-8s %.4f %.4f | %-20s | %5d | %-26s | %-26s" % (
            c, means["old"], means["ctl2"], " ".join("%.3f" % sc["ctl2"][n]["spellF1"] for n in tune_shape.TUNE),
            npred, fb("old"), fb("ctl2")), flush=True)
    res["commit"] = a5_ctl._commit()
    with open(os.path.join(ROOT, "eval", "results", "a8_sim_%s_s%d_%s%s.json" % (variant, seed, sim, tag)), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


def cmd_transfer(opt):
    import a8_tags
    from possess import pipeline

    cfgs = opt.pop("cfg", None)
    cfgs = cfgs.split(",") if isinstance(cfgs, str) else ["h05", "v02", "r10", "h05r10"]
    D = os.path.join(ROOT, "data", "challenge")
    tags, home, f, per, base = a8_tags.challenge_default()
    play = per > 0
    vis = ~np.isnan(f["ball_x"])
    hh = play.sum() / FPS / 1800.0

    def row(lab):
        n = ctl_truth.n_possessions(lab, per)
        return {"possessions": n, "per_half_hour": n / hh, "D_share": float((lab == "D")[play].mean()),
                "home_share_of_HA": float((lab == "H")[play].sum() / max(1, np.isin(lab, ["H", "A"])[play].sum())),
                "frames_changed": float((lab != base)[play].mean()),
                "changed_on_hidden": float((lab != base)[play & ~vis].sum() / max(1, (lab != base)[play].sum()))}

    res = {"default": row(base)}
    print("default", json.dumps(res["default"]))
    for c in cfgs:
        over = dict(CONFIGS[c])
        over.pop("sp_rate", None)
        tkw = dict(w_tag=over.get("w_tag", 0.0), tag_vis=over.get("tag_vis", 0.0), trim_s=over.get("trim", 0.0),
                   w_tagr=over.get("w_tagr", 0.0))
        extra, dec = pipeline.challenge_tag_inputs(D, len(per), tkw)
        lab = pipeline.label(f, period=per, dec_kw=dec, extra=extra)
        res[c] = row(lab)
        if "tag_events" in extra:
            pr = st.restart_prior(extra["tag_events"], base == "D", 4.0, offset=0)
            m = pr != 0
            res[c]["restart_events"] = len(extra["tag_events"])
            res[c]["restart_frames_anchored"] = int(m.sum())
            res[c]["default_agrees_on_restart_frames"] = float(
                (np.where(base == "H", 1, np.where(base == "A", -1, 0))[m] == pr[m]).mean()) if m.any() else None
        print(c, json.dumps(res[c]), flush=True)
    res["commit"] = a5_ctl._commit()
    with open(os.path.join(ROOT, "eval", "results", "a8_transfer.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    cmd, rest = sys.argv[1], a4_short.kv(sys.argv[2:])
    {"stats": cmd_stats, "run": cmd_run, "transfer": cmd_transfer}[cmd](rest)
