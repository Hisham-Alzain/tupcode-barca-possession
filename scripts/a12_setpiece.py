"""Envelope L11-C: tag-gated set-piece dead ball (src/possess/setpiece.py).

    uv run python scripts/ramlock.py -- uv run python scripts/a12_setpiece.py inventory
    uv run python scripts/ramlock.py -- uv run python scripts/a12_setpiece.py run bench=real2 seed=0 [sim=base] [cfg=a,b]
    uv run python scripts/ramlock.py -- uv run python scripts/a12_setpiece.py transfer [cfg=a,b]

inventory: provided INPUT only (no truth): per SET PIECES clip of type Goal Kick / Corner Kick /
           Free Kick / Throw-in, the physical restart found in a wide window [start - 10 s, start + 20 s],
           where it sits relative to the clip start, and what our default labels say before it.
           -> eval/results/a12_inventory.json
run:       external tune matches. They have no tags, so set-piece clips are SIMULATED around the true
           restarts (truth D -> live after >= 1 s): the type from the clean ball position at the
           restart (Laws geometry), the taker from the truth, the clip start = restart - lead(type)
           + N(0, jitter), tags dropped so that the rate per half hour per type matches the provided
           file (INPUT statistic), throw-ins without a team (as in the provided file). Default vs the
           rule on old truth and ctl T = 2 truth, paired 2-min block bootstrap.
           -> eval/results/a12_<bench>_s<seed>_<sim>.json
transfer:  provided INPUT only: frames changed, D added, possessions, changed restarts by type.
Tune matches only; never held-out; the public truth is never opened.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from possess import ctl_truth, setpiece as sp, smarttags as st  # noqa: E402

FPS = 25.0
TAGS_XML = os.path.join(ROOT, "data", "challenge", "tactical", "BAR-ATH_tactical_smart_tagging.xml")
RES = os.path.join(ROOT, "eval", "results")
# Lead of the clip start before the restart, by type: T114 (autonomous-L8.md; provided INPUT, our
# stoppage exits from the clip start): goal kick ~4.3 s, throw-in ~6 s, corner 6.9 s. Free kick: not
# measured in T114; the inventory below gives it (median over the provided clips, INPUT only).
LEAD = {"goal kick": 4.3, "corner kick": 6.9, "throw-in": 6.0, "free kick": 6.0}
SIMS = {
    "base": dict(jitter=1.5),
    "jit0": dict(jitter=0.0),
    "jit3": dict(jitter=3.0),
    "lead-2": dict(jitter=1.5, dlead=-2.0),
    "lead+2": dict(jitter=1.5, dlead=2.0),
    "all": dict(jitter=1.5, keep=1.0),             # every true restart tagged
}
CONFIGS = {
    "default": None,
    # round 1 (free kicks included; real2 s2 failed the gate: false kicks after quick free kicks)
    "sp": dict(fk=1),
    "sph": dict(hid=1, fk=1),
    "sp_g10": dict(gap_s=10.0, fk=1),
    "sp_g20": dict(gap_s=20.0, fk=1),
    "sp_nt": dict(take_s=0.0, fk=1),
    "sph_nt": dict(hid=1, take_s=0.0, fk=1),
    "sp_w8": dict(pre_s=8.0, fk=1),
    # round 2: types with a Laws spot only (setpiece.SP fk=0)
    "b": dict(),
    "bh": dict(hid=1),
    "b_nt": dict(take_s=0.0),
    "bh_nt": dict(hid=1, take_s=0.0),
    "bh_g10": dict(hid=1, gap_s=10.0),
}


def kv(args):
    out = {}
    for a in args:
        k, v = a.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def _commit():
    import subprocess
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provided_rates():
    """Tagged set pieces per half hour by type in the provided file (INPUT)."""
    from possess import pipeline
    tags = st.load_tags(TAGS_XML)
    hh = sum(b - a + 1 for _, a, b in pipeline.HALVES) / FPS / 1800.0
    out = {}
    for t in tags:
        k = sp.norm_type(t["type"]) if t["code"] == "SET PIECES" else None
        if k:
            out[k] = out.get(k, 0) + 1
    return {k: v / hh for k, v in out.items()}


def challenge_events(tags, home, offset=1):
    """Every SET PIECES clip of the four types: {a (frame_id), type, sign}."""
    return [{"a": t["a"] + offset, "b": t["b"] + offset, "type": t["type"], "sign": st.team_sign(t, home)}
            for t in tags if t["code"] == "SET PIECES" and sp.norm_type(t["type"])]


# ---------------------------------------------------------------------------------------------- inventory
def cmd_inventory(opt):
    import a8_tags
    tags, home, f, per, lab = a8_tags.challenge_default()
    bx, by = f["ball_x"], f["ball_y"]
    v = sp.speed(bx, by)
    evs = challenge_events(tags, home)
    rows = []
    for hid in (0, 1):
        kw = dict(sp.SP, hid=hid)
        for ev in evs:
            typ = sp.norm_type(ev["type"])
            r = sp.find_restart(bx, by, v, ev["a"] - int(10 * FPS), ev["a"] + int(20 * FPS), typ, kw)
            row = {"a": ev["a"], "type": typ, "sign": ev["sign"], "hid": hid, "found": r is not None}
            a = ev["a"]
            row["vis_in_clip"] = float((~np.isnan(bx[a:a + 875])).mean())
            if r is not None:
                k, how = r
                row.update(kick=k, how=how, rel_s=(k - a) / FPS)
                ds = np.flatnonzero(lab[max(k - 25 * 20, 0):k] == "D")
                row["ours_D_at_kick-1"] = bool(lab[k - 1] == "D")
                row["last_D_before_s"] = (k - (max(k - 500, 0) + ds[-1])) / FPS if len(ds) else None
                if len(ds):
                    e = max(k - 500, 0) + int(ds[-1])
                    seg = lab[e + 1:k]
                    s = 1 if ev["sign"] > 0 else -1
                    row["between"] = {c: int((seg == c).sum()) for c in "HAD"}
                    row["between_taker"] = int((seg == ("H" if s > 0 else "A")).sum()) if ev["sign"] else None
                row["label_after"] = "".join(lab[k:k + 50:10].tolist())
            rows.append(row)
    summ = {}
    for hid in (0, 1):
        for typ in sp.TYPES:
            rs = [r for r in rows if r["hid"] == hid and r["type"] == typ]
            fd = [r for r in rs if r["found"]]
            early = [r["last_D_before_s"] for r in fd if r.get("last_D_before_s") and r["last_D_before_s"] > 0.2]
            summ["%s hid=%d" % (typ, hid)] = {
                "clips": len(rs), "found": len(fd),
                "rel_s_q10_50_90": [float(np.percentile([r["rel_s"] for r in fd], p)) for p in (10, 50, 90)] if fd else None,
                "D_until_kick": sum(1 for r in fd if r["ours_D_at_kick-1"]),
                "D_ends_early": len(early), "early_s_q50_max": [float(np.median(early)), float(max(early))] if early else None,
                "early_taker_frames": sum(r.get("between_taker") or 0 for r in fd),
                "early_frames": sum(sum((r.get("between") or {}).get(c, 0) for c in "HA") for r in fd if r.get("last_D_before_s", 0) and r["last_D_before_s"] <= 15),
            }
            print("%-24s %s" % ("%s hid=%d" % (typ, hid), json.dumps(summ["%s hid=%d" % (typ, hid)])), flush=True)
    json.dump({"commit": _commit(), "summary": summ, "rows": rows}, open(os.path.join(RES, "a12_inventory.json"), "w"),
              indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


# ---------------------------------------------------------------------------------------------- external
def _runs(lab, period):
    brk = np.flatnonzero((lab[1:] != lab[:-1]) | (period[1:] != period[:-1])) + 1
    bounds = np.concatenate([[0], brk, [len(lab)]])
    return [(int(a), int(b), str(lab[a])) for a, b in zip(bounds[:-1], bounds[1:]) if period[a] > 0]


def true_restarts(truth, period, cx, cy):
    """[(restart frame, type, sign)]: truth D run >= 1 s followed by a team; the type from the clean
    ball at the restart (Laws geometry, 3 m tolerance); kick-offs (ball within 3 m of the centre) out."""
    out = []
    for a, b, k in _runs(truth, period):
        if k != "D" or b - a < FPS or b >= len(truth) or period[b] != period[a] or truth[b] not in ("H", "A"):
            continue
        j = b
        while j < min(b + 13, len(cx)) and np.isnan(cx[j]):
            j += 1
        x, y = (cx[j], cy[j]) if j < len(cx) else (np.nan, np.nan)
        if np.isnan(x) or np.hypot(x, y) < 3.0:
            continue
        typ = ("goal kick" if sp.spot_ok("goal kick", x, y, 3.0) else "corner kick" if sp.spot_ok("corner kick", x, y, 3.0)
               else "throw-in" if abs(y) >= sp.HALF_W - 3.0 else "free kick")
        out.append((b, typ, 1 if truth[b] == "H" else -1))
    return out


def sim_events(rs, period, seed, rates, jitter=1.5, dlead=0.0, keep=None):
    rng = np.random.default_rng(3000 + seed)
    hh = (np.asarray(period) > 0).sum() / FPS / 1800.0
    out = []
    for typ in sp.TYPES:
        ix = [i for i, r in enumerate(rs) if r[1] == typ]
        if not ix:
            continue
        p = 1.0 if keep else min(1.0, rates.get(typ, 0.0) * hh / len(ix))
        for i in ix:
            if rng.uniform() >= p:
                continue
            b, _, s = rs[i]
            a = b - int(round((LEAD[typ] + dlead) * FPS)) + (int(rng.normal(0, jitter * FPS)) if jitter else 0)
            out.append({"a": a, "type": typ, "sign": 0 if typ == "throw-in" else s, "true": b})
    return out


def cmd_run(opt):
    import a4_short
    import a5_ctl  # noqa: F401  (a4_short._path for real1 / orig)
    import diagnose as dg
    import tune_shape
    pm = a5_ctl.pm
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    sim = opt.pop("sim", "base")
    cfgs = opt.pop("cfg", None)
    cfgs = ["default"] + [c for c in (cfgs.split(",") if isinstance(cfgs, str) else CONFIGS) if c != "default"]
    rates = provided_rates()
    res = {"bench": variant, "seed": seed, "sim": sim, "sim_kw": SIMS[sim], "lead": LEAD, "rates": rates,
           "configs": {c: CONFIGS[c] for c in cfgs}, "scores": {}, "boot": {}, "fired": {}}
    labs, tv = {}, {}
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        clean = np.load(a5_ctl._path(name, "", 0))
        rs = true_restarts(truth, period, clean["f_ball_x"], clean["f_ball_y"])
        evs = sim_events(rs, period, seed, rates, **SIMS[sim])
        tv[name] = {"old": truth, "ctl2": ctl_truth.relabel(truth, period, 2.0, FPS)}
        base = a4_short.decode_one(feats, period)
        for c in cfgs:
            if CONFIGS[c] is None:
                labs[c, name] = base
                continue
            lab, info = sp.apply(base, feats["ball_x"], feats["ball_y"], period, evs, CONFIGS[c])
            labs[c, name] = lab
            ok = [r for r in info if r["kick"] is not None]
            true_of = {e["a"]: e["true"] for e in evs}
            res["fired"].setdefault(c, {})[name] = {
                "events": len(evs), "types": {t: sum(1 for e in evs if e["type"] == t) for t in sp.TYPES},
                "kick_found": len(ok), "kick_err_s_q50": float(np.median([abs(r["kick"] - true_of[r["a"]]) / FPS for r in ok])) if ok else None,
                "kick_within_1s": sum(1 for r in ok if abs(r["kick"] - true_of[r["a"]]) <= FPS),
                "filled_frames": int(sum(r.get("filled", 0) for r in ok)),
                "filled_true_D": float((tv[name]["old"][(lab != base)] == "D").mean()) if (lab != base).any() else None,
                "changed": int((lab != base).sum())}
        del feats
    print("%s s%d sim=%s" % (variant, seed, sim))
    for c in cfgs:
        sc = {}
        for name in tune_shape.TUNE:
            for k, t in tv[name].items():
                r = pm.evaluate(pm.label_codes(t), pm.label_codes(labs[c, name]), bootstrap_n=0)
                sc.setdefault(k, {})[name] = r["spell_f1_mean"]
        means = {k: float(np.mean(list(sc[k].values()))) for k in sc}
        bt = {}
        if c != "default":
            for k in ("old", "ctl2"):
                d = np.mean([np.subtract(*dg._pr_boot(tv[n][k], labs["default", n], labs[c, n])[::-1]) for n in tune_shape.TUNE], axis=0)
                bt[k] = [float(d.mean()), float(np.percentile(d, 5)), float(np.percentile(d, 95))]
        sc["mean"] = means
        res["scores"][c], res["boot"][c] = sc, bt
        fr = res["fired"].get(c, {})
        fb = lambda k: "%+.4f [%+.4f, %+.4f]" % tuple(bt[k]) if bt else ""  # noqa: E731
        print("%-8s old %.4f ctl2 %.4f | d old %-26s | d ctl2 %-26s | ev %s kick %s ok1s %s chg %s trueD %s" % (
            c, means["old"], means["ctl2"], fb("old"), fb("ctl2"),
            sum(x["events"] for x in fr.values()) if fr else "-", sum(x["kick_found"] for x in fr.values()) if fr else "-",
            sum(x["kick_within_1s"] for x in fr.values()) if fr else "-", sum(x["changed"] for x in fr.values()) if fr else "-",
            ",".join("%.2f" % x["filled_true_D"] if x["filled_true_D"] is not None else "-" for x in fr.values()) if fr else "-"),
            flush=True)
    res["commit"] = _commit()
    json.dump(res, open(os.path.join(RES, "a12_%s_s%d_%s.json" % (variant, seed, sim)), "w"), indent=1, default=float)


def cmd_diag(opt):
    """Per simulated event on one external bench: kick vs true restart, frames filled, their truth."""
    import a4_short
    import a5_ctl
    import tune_shape
    variant, seed = opt.pop("bench", "real2"), int(opt.pop("seed", 0))
    v = "" if variant == "orig" else variant
    cfg = CONFIGS[opt.pop("cfg", "sp")]
    for name in tune_shape.TUNE:
        truth, period, feats = a4_short.load(name, v, seed)
        clean = np.load(a5_ctl._path(name, "", 0))
        evs = sim_events(true_restarts(truth, period, clean["f_ball_x"], clean["f_ball_y"]), period, seed, provided_rates())
        base = a4_short.decode_one(feats, period)
        _, info = sp.apply(base, feats["ball_x"], feats["ball_y"], period, evs, cfg)
        tr = {e["a"]: e["true"] for e in evs}
        for r in info:
            if r["kick"] is None or not r.get("filled"):
                continue
            e, k, t = r["dead_from"], r["kick"], tr[r["a"]]
            seg = truth[e + 1:k]
            # true D run containing / before the true restart
            j = t - 1
            while j > 0 and truth[j] == "D":
                j -= 1
            print("%-9s %-11s kick-true %+6.1fs filled %5d trueD %.2f ourDend-trueDstart %+6.1fs before %s vis_rest %.2f" % (
                name, r["type"], (k - t) / FPS, r["filled"], float((seg == "D").mean()), (e - j) / FPS, r["before"],
                float((~np.isnan(feats["ball_x"][max(k - 25, 0):k])).mean())), flush=True)
        del feats


# ---------------------------------------------------------------------------------------------- transfer
def cmd_transfer(opt):
    import a8_tags
    cfgs = opt.pop("cfg", None)
    cfgs = cfgs.split(",") if isinstance(cfgs, str) else ["sp", "sph"]
    tags, home, f, per, base = a8_tags.challenge_default()
    evs = challenge_events(tags, home)
    play = per > 0
    hh = play.sum() / FPS / 1800.0
    res = {"default": {"possessions": ctl_truth.n_possessions(base, per), "D_share": float((base == "D")[play].mean())}}
    print("default", json.dumps(res["default"]))
    for c in cfgs:
        lab, info = sp.apply(base, f["ball_x"], f["ball_y"], per, evs, CONFIGS[c])
        ch = (lab != base) & play
        res[c] = {"possessions": ctl_truth.n_possessions(lab, per), "per_half_hour": ctl_truth.n_possessions(lab, per) / hh,
                  "frames_changed": int(ch.sum()), "share": float(ch[play].mean()),
                  "D_added": int(((lab == "D") & (base != "D") & play).sum()),
                  "changed_restarts": [{"type": r["type"], "clip_a_row": r["a"] - 1, "kick_row": r["kick"] - 1, "rel_s": (r["kick"] - r["a"]) / FPS,
                                        "how": r.get("how"), "filled": r.get("filled", 0), "before": r.get("before"), "taken": r.get("taken", 0)}
                                       for r in info if r["kick"] is not None and (r.get("filled", 0) or r.get("taken", 0))],
                  "found_by_type": {t: [sum(1 for r in info if r["type"] == t and r["kick"] is not None),
                                        sum(1 for r in info if r["type"] == t)] for t in sp.TYPES}}
        print(c, json.dumps({k: res[c][k] for k in res[c] if k != "changed_restarts"}), flush=True)
        for r in res[c]["changed_restarts"]:
            print("   ", json.dumps(r))
    res["commit"] = _commit()
    json.dump(res, open(os.path.join(RES, "a12_transfer.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    cmd, rest = sys.argv[1], kv(sys.argv[2:])
    {"inventory": cmd_inventory, "run": cmd_run, "transfer": cmd_transfer, "diag": cmd_diag}[cmd](rest)
