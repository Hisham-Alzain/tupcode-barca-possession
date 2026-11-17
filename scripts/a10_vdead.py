"""Envelope L11-A: visible dead ball labelled as a team.

prov   provided match INPUT only (no truth): how visible / how the ball behaves in stoppages we are confident
       about (Smart Tagging SET PIECES clip starts, our own predicted D runs); traces at catalogue frames.
bench  external benches: the same statistic on truth-D (ctl T=2) frames and on predicted D runs; the error
       class "visible true-D frame labelled as a team" (frames, oracle spellF1 cost); candidate flags scored
       on top of the default with leave-one-match-out w_bt / w_bd evidence (as a8b_r7.py bundle).

    uv run python scripts/ramlock.py -- uv run python scripts/a10_vdead.py prov
    uv run python scripts/ramlock.py -- uv run python scripts/a10_vdead.py bench test=real2:0,real1:0,orig:0
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import a4_short  # noqa: E402
import a8b_r6 as R6  # noqa: E402
import a8b_run as R  # noqa: E402
import tune_shape  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results")
FPS = 25.0
STATES = ("hidden", "rest", "carried", "slow_free", "fast")


def ball_state(bx, by, d_home, d_away, fps=FPS, lag=8):
    """Per frame: 0 hidden, 1 rest (< 0.5 m/s), 2 carried (0.5-4 m/s, a player within 1.5 m),
    3 slow free (0.5-4 m/s, nobody within 1.5 m), 4 fast (>= 4 m/s). Speed as edges.still_dead."""
    bx, by = np.asarray(bx, float), np.asarray(by, float)
    n, h = len(bx), lag // 2
    s = np.full(n, np.nan)
    if n > lag:
        s[h:n - lag + h] = np.hypot(bx[lag:] - bx[:-lag], by[lag:] - by[:-lag]) * fps / lag
    dmin = np.fmin(np.asarray(d_home, float), np.asarray(d_away, float))
    st = np.zeros(n, np.int8)
    vis = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        sv = np.where(np.isnan(s), 0.0, s)
        st[vis & (sv < 0.5)] = 1
        st[vis & (sv >= 0.5) & (sv < 4.0) & (dmin < 1.5)] = 2
        st[vis & (sv >= 0.5) & (sv < 4.0) & ~(dmin < 1.5)] = 3
        st[vis & (sv >= 4.0)] = 4
    return st


def runs(mask, period, min_len=1):
    out, n, i = [], len(mask), 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j] and period[j] == period[i]:
            j += 1
        if j - i >= min_len:
            out.append((i, j))
        i = j
    return out


def describe(st, m):
    """Share of frames per state in mask m (and the visible share)."""
    k = int(m.sum())
    if not k:
        return {"frames": 0}
    r = {"frames": k, "visible": float((st[m] > 0).mean())}
    for i, nme in enumerate(STATES):
        r[nme] = float((st[m] == i).mean())
    vis = m & (st > 0)
    r["vis_rest_or_carried"] = float(np.isin(st[vis], (1, 2)).mean()) if vis.any() else 0.0
    return r


def fmt(r):
    if not r.get("frames"):
        return "frames 0"
    return "frames %6d (%6.0f s) visible %.3f | hidden %.3f rest %.3f carried %.3f slow_free %.3f fast %.3f" % (
        r["frames"], r["frames"] / FPS, r["visible"], r["hidden"], r["rest"], r["carried"], r["slow_free"], r["fast"])


def run_stats(st, lab, period, min_s=2.0):
    """Predicted (or true) D runs of >= min_s: share of runs with any visible frame, mean visible share."""
    rr = runs((lab == "D") & (period > 0), period, int(min_s * FPS))
    if not rr:
        return {"runs": 0}
    vs = np.array([(st[a:b] > 0).mean() for a, b in rr])
    return {"runs": len(rr), "any_visible": float((vs > 0).mean()), "mostly_visible": float((vs > 0.5).mean()),
            "fully_hidden": float((vs == 0).mean()), "mean_visible": float(vs.mean())}


def cmd_prov(opt):
    """Provided match INPUT only. Nothing is chosen on it."""
    import a8_tags
    tags, home, f, per, base = a8_tags.challenge_default()
    play = per > 0
    st = ball_state(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"])
    res = {}
    sp = [t for t in tags if t["code"] == "SET PIECES"]
    core, wide = np.zeros(len(per), bool), np.zeros(len(per), bool)
    by_type = {}
    for t in sp:
        a = t["a"] + 1
        core[a:a + 75] = True            # clip start .. +3 s: before the earliest restart (p10 clip start + 3.2 s, row 51)
        wide[max(0, a - 50):a + 150] = True
        ty = (t.get("type") or t["labels"].get("set piece") or t["labels"].get("set pieces") or "?")
        m = np.zeros(len(per), bool)
        m[a:a + 75] = True
        by_type.setdefault(ty, []).append(m)
    print("SET PIECES clips %d; label groups seen: %s" % (len(sp), sorted({k for t in sp for k in t["labels"]})))
    res["setpiece_core_0_3s"] = describe(st, core & play)
    print("set-piece clip start..+3 s :", fmt(res["setpiece_core_0_3s"]))
    res["setpiece_wide_m2_6s"] = describe(st, wide & play)
    print("set-piece clip -2..+6 s    :", fmt(res["setpiece_wide_m2_6s"]))
    for ty, ms in sorted(by_type.items()):
        m = np.any(ms, axis=0) & play
        res["type_" + ty] = describe(st, m)
        res["type_" + ty]["clips"] = len(ms)
        print("  %-22s n=%2d %s" % (ty[:22], len(ms), fmt(res["type_" + ty])))
    # per clip: is the ball visible at all during the first 3 s
    anyv = [bool((st[t["a"] + 1:t["a"] + 76] > 0).any()) for t in sp]
    res["clips_with_visible_ball_0_3s"] = float(np.mean(anyv))
    print("clips with a visible ball in their first 3 s: %.3f (%d of %d)" % (np.mean(anyv), sum(anyv), len(anyv)))
    res["pred_D"] = describe(st, (base == "D") & play)
    print("our predicted D frames      :", fmt(res["pred_D"]))
    res["pred_D_runs2s"] = run_stats(st, base, per)
    print("our predicted D runs >= 2 s :", json.dumps(res["pred_D_runs2s"]))
    res["all_play"] = describe(st, play)
    print("all play frames             :", fmt(res["all_play"]))
    # catalogue frames (public window, INPUT only): trace at 0.5 s steps
    for a, b in ((15860, 15960), (16320, 16420), (17360, 17480), (18100, 18400)):
        dmin = np.fmin(f["d_home"], f["d_away"])
        row = []
        for i in range(a, b, 12):
            row.append("%d:%s%d/%.1f" % (i, base[i], st[i], dmin[i] if np.isfinite(dmin[i]) else -1))
        print("trace %d-%d (frame:label state/dmin): %s" % (a, b, " ".join(row)))
        bx, by = f["ball_x"][a:b], f["ball_y"][a:b]
        print("   ball x %s | y %s" % (np.round(bx[::25], 1).tolist(), np.round(by[::25], 1).tolist()))
    res["commit"] = R.a5_ctl._commit()
    with open(os.path.join(OUT, "a10_L11A_prov.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


def vdead_oracle(lab, truth, period, vis):
    """Labels with every visible true-D frame that we call a team set to D (the error-class bound)."""
    o = np.asarray(lab).copy()
    m = (truth == "D") & vis & np.isin(o, ["H", "A"]) & (period > 0)
    o[m] = "D"
    return o, m


CFGS = {"bd": dict(w_bd=1.0), "still": dict(ed_still=1.0), "bd+still": dict(w_bd=1.0, ed_still=1.0),
        "rk5": dict(ed_rkick=5.0), "rk4": dict(ed_rkick=4.0), "rk7": dict(ed_rkick=7.0),
        "rk5L": dict(ed_rkick=5.0, ed_rk_line=2.5), "bd+rk5": dict(w_bd=1.0, ed_rkick=5.0),
        "bd+st+rk5": dict(w_bd=1.0, ed_still=1.0, ed_rkick=5.0),
        "rk5law": dict(ed_rkick=5.0, ed_rk_law=1), "rk5rlaw": dict(ed_rkick=5.0, ed_rk_law=1, ed_rk_reapp=1),
        "bd+st+rk5law": dict(w_bd=1.0, ed_still=1.0, ed_rkick=5.0, ed_rk_law=1),
        "rk5r": dict(ed_rkick=5.0, ed_rk_reapp=1), "bd+st+rk5r": dict(w_bd=1.0, ed_still=1.0, ed_rkick=5.0, ed_rk_reapp=1)}


def cmd_bench(opt):
    from possess import balldead
    pool = R6.benches(opt.pop("pool", R6.POOL3))
    test = R6.benches(opt.pop("test", R6.POOL3))
    names = [c for c in str(opt.pop("cfg", ",".join(CFGS))).split(",") if c]
    tag = opt.pop("tag", "bench")
    extra = json.loads(opt.pop("extra", "{}"))         # {"name": {kw}} more configs
    CFGS.update(extra)
    bt_models = R6.lomo_models(R6.POOL3, ["lin"], "pool3")["lin"]
    tr, keep = {}, {}
    for b in pool:
        data = R6._src_rows(b)
        for name, d in data.items():
            k = d["vis"] & (d["period"] > 0)
            tr[b, name] = (d["BX"][k][::3], (d["truth"]["ctl2"][k][::3] == "D").astype(float))
        if b in test:
            keep[b] = data
        del data
    out = {}
    for b in test:
        data = keep.pop(b) if b in keep else R6._src_rows(b)
        labs, tv, stats = {}, {}, {}
        for name, d in data.items():
            m = R6._fit_bd([v for (bb, mm), v in tr.items() if mm != name])
            tv[name] = d["truth"]
            f0 = dict(d["feats"])
            f0["bt"] = np.where(d["vis"], np.tanh(R6.logit_of(bt_models[name], d["X"], d["E"]) / 2.0), 0.0)
            f0["bd"] = balldead.evidence(d["BX"], d["vis"], m)
            per = d["period"]
            labs["bt", name] = a4_short.decode_one(dict(f0), per, dict(w_bt=0.5))
            for c in names:
                labs[c, name] = a4_short.decode_one(dict(f0), per, dict(CFGS[c], w_bt=0.5))
            f = d["feats"]
            bx = np.where(d["vis"], np.asarray(f["ball_x"], float), np.nan)
            st = ball_state(bx, f["ball_y"], f["d_home"], f["d_away"])
            t2 = np.asarray(d["truth"]["ctl2"])
            play = per > 0
            lab0 = labs["bt", name]
            orc, em = vdead_oracle(lab0, t2, per, d["vis"])
            labs["oracle", name] = orc
            stats[name] = {"true_D": describe(st, (t2 == "D") & play), "pred_D": describe(st, (lab0 == "D") & play),
                           "true_D_runs2s": run_stats(st, t2, per), "pred_D_runs2s": run_stats(st, lab0, per),
                           "err_vis_D_as_team": describe(st, em), "play_s": float(play.sum() / FPS)}
            for c in names:
                stats[name]["fix_" + c] = int((em & (labs[c, name] == "D")).sum())
                stats[name]["new_err_" + c] = int(((t2 != "D") & (labs[c, name] == "D") & (lab0 != "D") & play).sum())
        key = "%s:%d" % b
        print("== %s" % key)
        for name, s in stats.items():
            print("  %s true D : %s" % (name, fmt(s["true_D"])))
            print("  %s pred D : %s" % (name, fmt(s["pred_D"])))
            print("  %s true D runs>=2s %s | pred %s" % (name, json.dumps(s["true_D_runs2s"]), json.dumps(s["pred_D_runs2s"])))
            e = s["err_vis_D_as_team"]
            print("  %s ERROR visible true-D called team: %.1f s per 90 min (%d frames) %s | fixed by %s | new false D %s" % (
                name, e.get("frames", 0) / FPS / (s["play_s"] / 5400.0), e.get("frames", 0), fmt(e),
                {c: s["fix_" + c] for c in names}, {c: s["new_err_" + c] for c in names}))
        sc = R.score_table(labs, tv, ["bt", "oracle"] + names, list(tune_shape.TUNE), "%s  L11-A on top of w_bt (LOMO w_bt / w_bd)" % key)
        out[key] = {"scores": sc, "stats": stats}
        del data
    with open(os.path.join(OUT, "a10_L11A_%s.json" % tag), "w") as fh:
        json.dump({"pool": pool, "cfg": {c: CFGS[c] for c in names}, "results": out, "commit": R.a5_ctl._commit()}, fh, indent=1, default=float)


def cmd_diag(opt):
    """External bench only: where the false D frames of a config come from (late exit after a true
    stoppage vs early entry vs isolated), with the ball state at the end of each run."""
    b = R6.benches(opt.pop("test", "real2:0"))[0]
    c = opt.pop("cfg", "rk5law")
    data = R6.load(*b)
    for name, d in data.items():
        per, t2 = d["period"], np.asarray(d["truth"]["ctl2"])
        f0 = dict(d["feats"])
        l0 = a4_short.decode_one(dict(f0), per, {})
        l1 = a4_short.decode_one(dict(f0), per, dict(CFGS[c]))
        fd = (l1 == "D") & (l0 != "D") & (per > 0)
        rows = []
        for a, e in runs(fd, per):
            bad = int((t2[a:e] != "D").sum())
            if not bad:
                continue
            kind = "late_exit" if t2[a] == "D" else ("early_entry" if (t2[e - 1] == "D" or (e < len(t2) and t2[e] == "D")) else "isolated")
            rows.append((kind, bad, (e - a) / FPS))
        tot = {}
        for k, bad, ln in rows:
            tot.setdefault(k, [0, 0])
            tot[k][0] += bad
            tot[k][1] += 1
        good = int(((t2 == "D") & fd).sum())
        print(name, c, "fixed", good, "| false-D frames [frames, runs] by kind:", tot,
              "| isolated run lengths s:", sorted(round(r[2], 1) for r in rows if r[0] == "isolated")[-10:], flush=True)


CAT = ((15889, 15946), (16338, 16400), (17380, 17461), (18127, 18389))


def cmd_transfer(opt):
    """Provided match INPUT only (no truth): what each config changes in our own output. Never used to choose."""
    import a8_tags
    from possess import ctl_truth, pipeline
    extra = json.loads(opt.pop("extra", "{}"))
    CFGS.update(extra)
    names = [c for c in str(opt.pop("cfg", ",".join(CFGS))).split(",") if c]
    tags, home, f, per, base = a8_tags.challenge_default()
    play = per > 0
    st = ball_state(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"])
    win = np.zeros(len(per), bool)
    for t in tags:
        if t["code"] == "SET PIECES":
            win[max(0, t["a"] + 1 - 50):t["a"] + 1 + 150] = True
    res = {"default": {"possessions": ctl_truth.n_possessions(base, per), "D_share": float((base == "D")[play].mean()),
                       "catalogue_D": {"%d-%d" % s: int((base[s[0]:s[1]] == "D").sum()) for s in CAT}}}
    print("default", json.dumps(res["default"]))
    for c in names:
        lab = pipeline.label(f, period=per, dec_kw=CFGS[c])
        toD, toL = play & (lab == "D") & (base != "D"), play & (lab != "D") & (base == "D")
        rr = runs(toD, per)
        newD = [r for r in runs((lab == "D") & play, per) if (base[r[0]:r[1]] != "D").all()]
        res[c] = {"possessions": ctl_truth.n_possessions(lab, per), "D_share": float((lab == "D")[play].mean()),
                  "frames_changed": float(((lab != base) & play).sum() / play.sum()), "live_to_D": int(toD.sum()),
                  "live_to_D_runs": len(rr), "live_to_D_in_setpiece_window": float(win[toD].mean()) if toD.any() else 0.0,
                  "live_to_D_visible": float((st[toD] > 0).mean()) if toD.any() else 0.0,
                  "D_to_live": int(toL.sum()), "new_D_runs": len(newD),
                  "window_D_frames": int((lab[11186:18686] == "D").sum()),
                  "runs_sec": [round((b - a) / FPS, 1) for a, b in rr],
                  # our own D frames on the ADR-027 catalogue spans (diagnosis only, never used to choose)
                  "catalogue_D": {"%d-%d" % s: int((lab[s[0]:s[1]] == "D").sum()) for s in CAT}}
        print(c, json.dumps(res[c]), flush=True)
    res["setpiece_window_share_of_play"] = float(win[play].mean())
    res["commit"] = R.a5_ctl._commit()
    with open(os.path.join(OUT, "a10_L11A_transfer_%s.json" % opt.pop("tag", "rk5r")), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]](a4_short.kv(sys.argv[2:]))
