"""Envelope L6-smart-tagging: the Smart Tagging file against the provided INPUT only.

    uv run python scripts/ramlock.py -- uv run python scripts/a8_tags.py

Inventory (codes, clip lengths, coverage, overlap) and alignment with the default decoded output
(agreement of the phase clips with our H / A call on visible and hidden frames; where the restart
sits inside a SET PIECES clip; restart team vs our first team after the stoppage).
The ground truth is never opened. Nothing is chosen here. -> eval/results/a8_tags_inventory.json
"""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from possess import ctl_truth, pipeline, smarttags as st  # noqa: E402
from possess.kickoff import pre_kickoff_spans  # noqa: E402

FPS = 25.0
D = os.path.join(ROOT, "data", "challenge")


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def challenge_default():
    """(tags, home, features, period with pre-kick-off removed, default labels per frame_id)."""
    xml = glob.glob(os.path.join(D, "**", "*tactical_smart_tagging.xml"), recursive=True)[0]
    home = st.home_team_name(glob.glob(os.path.join(D, "**", "*tactical_atd.xml"), recursive=True)[0])
    tags = st.load_tags(xml)
    f = pipeline.challenge_features(D, os.path.join(D, "_runs", "features_v1.npz"))
    per = pipeline.challenge_period(len(f["ball_x"]))
    for a, k in pre_kickoff_spans(xml, f["ball_x"], f["ball_y"], pipeline.HALVES):
        per[a + 1:k + 1] = 0
    return tags, home, f, per, pipeline.label(f, period=per)


def main():
    tags, home, f, per, lab = challenge_default()
    n = len(f["ball_x"])
    play = per > 0
    res = {"home": home, "n_tags": len(tags), "codes": {}}
    by = defaultdict(list)
    for t in tags:
        by[t["code"]].append(t)
    print("home =", home, "| tags", len(tags))
    print("%-30s %4s %5s %5s %5s %8s | teams" % ("code", "n", "p10", "med", "p90", "mode"))
    for c, ts in sorted(by.items(), key=lambda kv: -len(kv[1])):
        L = np.array([(t["b"] - t["a"]) / FPS for t in ts])
        mode = Counter(np.round(L, 1)).most_common(1)[0]
        tm = Counter(t["team"] or "N/A" for t in ts)
        res["codes"][c] = {"n": len(ts), "len_p10": q(L, 10), "len_med": q(L, 50), "len_p90": q(L, 90),
                           "mode_len": mode[0], "mode_share": mode[1] / len(ts), "teams": dict(tm)}
        print("%-30s %4d %5.1f %5.1f %5.1f %3.0f%%@%.0fs | %s" % (c, len(ts), q(L, 10), q(L, 50), q(L, 90),
                                                                  100 * mode[1] / len(ts), mode[0], dict(tm)))
    pos, neg = np.zeros(n), np.zeros(n)
    for t in tags:
        s = st.phase_sign(t, home)
        if s:
            (pos if s > 0 else neg)[max(t["a"] + 1, 0):t["b"] + 2] += 1
    cov, both = (pos + neg) > 0, (pos > 0) & (neg > 0)
    res["coverage"] = float(cov[play].mean())
    res["overlap_both"] = float(both[play].mean())
    res["home_share_of_one_team_frames"] = float(((pos > 0) & ~both)[play].sum() / max(1, (cov & ~both)[play].sum()))
    print("coverage of graded frames %.3f | both teams %.3f | home share of one-team frames %.3f" % (
        res["coverage"], res["overlap_both"], res["home_share_of_one_team_frames"]))
    inn = [t for t in tags if t["code"] in st.IN_POSS]
    out = [t for t in tags if t["code"] in st.OUT_POSS]
    same = sum(any(abs(o["a"] - i["a"]) <= 25 and abs(o["b"] - i["b"]) <= 25 and st.phase_sign(o, home) == st.phase_sign(i, home)
                   for i in inn) for o in out)
    res["out_clips"] = len(out)
    res["out_clips_mirroring_an_in_clip"] = same
    print("out-of-possession clips %d, mirroring an in-possession clip of the other team (+-1 s): %d" % (len(out), same))
    # union runs of one-team evidence: the simulator's unit
    prior = st.phase_prior(tags, n, home)
    for nm, m in (("one-team runs", (np.abs(prior) == 1) & play), ("uncovered runs", (~cov) & play), ("both-team runs", both & play)):
        e = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
        L = (e[1::2] - e[::2]) / FPS
        res[nm.replace(" ", "_")] = {"n": len(L), "p10": q(L, 10), "med": q(L, 50), "p90": q(L, 90), "mean": float(L.mean())}
        print("%-15s n %d  p10 %.1f med %.1f p90 %.1f mean %.1f s" % (nm, len(L), q(L, 10), q(L, 50), q(L, 90), L.mean()))

    vis = ~np.isnan(f["ball_x"])
    live = np.isin(lab, ["H", "A"]) & play
    ours = np.where(lab == "H", 1, np.where(lab == "A", -1, 0))
    al = {}
    for nm, m in (("all", play), ("visible", play & vis), ("hidden", play & ~vis)):
        one = m & (np.abs(prior) == 1)
        lv = one & live
        al[nm] = {"frames": int(m.sum()), "one_team_clip_share": float(one.sum() / max(1, m.sum())),
                  "ours_D_inside_clip": float((lab == "D")[one].mean()),
                  "agree_when_we_name_a_team": float((ours[lv] == prior[lv]).mean()),
                  "ours_D_outside_clips": float((lab == "D")[m & ~cov].mean()),
                  "live_frames_outside_clips_share": float((live & m & ~cov).sum() / max(1, (live & m).sum()))}
        print(nm, json.dumps(al[nm]))
    res["alignment"] = al
    dis = live & (np.abs(prior) == 1) & (ours != prior)
    e = np.flatnonzero(np.diff(np.concatenate([[0], dis.astype(int), [0]])))
    L = (e[1::2] - e[::2]) / FPS
    hid_share = np.array([1 - vis[a:b].mean() for a, b in zip(e[::2], e[1::2])])
    res["disagree_runs"] = {"n": len(L), "len_med": q(L, 50), "len_p90": q(L, 90), "frames": int(dis.sum()),
                            "frames_in_runs_ge_4s": int(sum((b - a) for a, b in zip(e[::2], e[1::2]) if b - a >= 100)),
                            "mostly_hidden_runs": int((hid_share > 0.5).sum()),
                            "frames_hidden": int((dis & ~vis).sum())}
    print("disagreement runs", json.dumps(res["disagree_runs"]))
    print("%-30s %7s %6s %6s" % ("code", "frames", "agree", "ourD"))
    res["by_code"] = {}
    for c, ts in sorted(by.items(), key=lambda kv: -len(kv[1])):
        m = np.zeros(n, bool)
        sg = np.zeros(n)
        for t in ts:
            s = st.phase_sign(t, home)
            if s:
                m[t["a"] + 1:t["b"] + 2] = True
                sg[t["a"] + 1:t["b"] + 2] = s
        m &= play
        lv = m & live
        if not lv.any():
            continue
        res["by_code"][c] = {"frames": int(m.sum()), "agree": float((ours[lv] == sg[lv]).mean()), "ours_D": float((lab == "D")[m].mean())}
        print("%-30s %7d %6.3f %6.3f" % (c, m.sum(), res["by_code"][c]["agree"], res["by_code"][c]["ours_D"]))
    rel = defaultdict(lambda: [0, 0, 0])
    for t in tags:
        s = st.phase_sign(t, home)
        if not s or t["b"] - t["a"] < 50:
            continue
        for i in range(max(t["a"] + 1, 0), min(t["b"] + 2, n)):
            if play[i]:
                k = min(4, int(5 * (i - t["a"] - 1) / (t["b"] - t["a"] + 1)))
                rel[k][2] += 1
                if live[i]:
                    rel[k][0] += ours[i] == s
                    rel[k][1] += 1
    res["agree_by_clip_fifth"] = [float(rel[k][0] / max(1, rel[k][1])) for k in range(5)]
    res["live_by_clip_fifth"] = [float(rel[k][1] / max(1, rel[k][2])) for k in range(5)]
    print("agreement by fifth of the clip", np.round(res["agree_by_clip_fifth"], 3), "| we say live", np.round(res["live_by_clip_fifth"], 3))

    dead = lab == "D"
    sp = defaultdict(list)
    for t in by.get("SET PIECES", []):
        a, b = max(t["a"] + 1, 0), min(t["b"] + 2, n)
        if not play[min(a, n - 1)]:
            continue
        j = st.restart_frame(t, dead)
        r = {"len": (b - a) / FPS, "dead_share": float(dead[a:b].mean()), "sign": st.team_sign(t, home)}
        if j is not None:
            r["rel_start"] = (j - a) / FPS
            r["rel_end"] = (j - b) / FPS
            r["first_team"] = int(ours[j])
        sp[t["type"] or "?"].append(r)
    res["set_pieces"] = {}
    print("%-22s %3s %5s %6s | restart from clip start p10/med/p90 s | after clip end | our first team = tag team" % ("type", "n", "anch", "dead%"))
    for ty, rs in sorted(sp.items()):
        an = [r for r in rs if "rel_start" in r]
        x = [r["rel_start"] for r in an]
        late = sum(r["rel_end"] > 0 for r in an)
        tm = [r for r in an if r["sign"]]
        ok = sum(r["first_team"] == r["sign"] for r in tm)
        res["set_pieces"][ty] = {"n": len(rs), "anchored": len(an), "dead_share": float(np.mean([r["dead_share"] for r in rs])),
                                 "len_med": q([r["len"] for r in rs], 50),
                                 "restart_from_start_s": [q(x, 10), q(x, 50), q(x, 90)], "restart_after_clip_end": late,
                                 "with_team": len(tm), "our_first_team_equals_tag": ok}
        print("%-22s %3d %5d %6.2f | %5.1f %5.1f %5.1f | %d | %d / %d" % (ty, len(rs), len(an), res["set_pieces"][ty]["dead_share"],
                                                                         q(x, 10), q(x, 50), q(x, 90), late, ok, len(tm)))
    ev = st.restart_events(tags, home)
    res["restart_events_with_team"] = dict(Counter(e_["type"] for e_ in ev))
    print("restart events with a taker:", res["restart_events_with_team"])
    res["goals"] = [{"team": g["team"], "a": g["a"], "b": g["b"], "half": g["half"]} for g in by.get("GOALS", [])]
    print("goals:", [(g["team"], g["a"], g["b"]) for g in by.get("GOALS", [])])
    print("kick-off tags:", [(t["type"], t["a"]) for t in by.get("SET PIECES", []) if t["type"].lower().startswith("kick")])
    hh = play.sum() / FPS / 1800.0
    res["possessions_per_half_hour"] = float(ctl_truth.n_possessions(lab, per) / hh)
    res["stoppages_per_half_hour"] = float(((lab[1:] == "D") & (lab[:-1] != "D") & play[1:]).sum() / hh)
    print("possessions / half hour %.1f | stoppages / half hour %.1f" % (res["possessions_per_half_hour"], res["stoppages_per_half_hour"]))
    with open(os.path.join(ROOT, "eval", "results", "a8_tags_inventory.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
