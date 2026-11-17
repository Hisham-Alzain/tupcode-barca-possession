"""Cut the tactical tracking into readable windows for the reverse-engineering agent swarm (ADR-001).

Each window -> one CSV at 5 Hz (every 5th frame), metres, pitch centred (x along length, home
side per half printed in the header). Columns:
  frame, t_s, gt (public window only, else blank), ball_x, ball_y (blank = missing),
  home = "x:y|x:y|..." sorted by x, away = same.
Plus the Smart Tagging intervals overlapping the window, as a header block.

    python scripts/make_agent_slices.py            # writes data/challenge/_agent_slices/
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
MTAH = os.path.join(ROOT, "third_party", "mtah")
DATA = os.path.join(ROOT, "data", "challenge", "tactical")
OUT = os.path.join(ROOT, "data", "challenge", "_agent_slices")
STEP = 5
HALVES = [(1, 11185, 90651), (2, 110808, 190197)]

# (name, first video frame, last video frame, has_gt). Public GT covers 11185..18684.
WINDOWS = [
    ("w1_public_a", 11185, 14934, True),
    ("w2_public_b", 14935, 18684, True),
    ("w3_h1_mid", 45000, 48749, False),
    ("w4_h1_late", 78000, 81749, False),
    ("w5_h2_early", 120000, 123749, False),
    ("w6_h2_mid", 150000, 153749, False),
    ("w7_h2_late", 182000, 185749, False),
]


def load_gt():
    gt = {}
    with open(os.path.join(MTAH, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")) as fh:
        for row in csv.DictReader(fh):
            gt[int(row["frame"])] = row["label"].strip()
    return gt


def load_tags():
    sys.path.insert(0, os.path.join(MTAH, "starter"))
    sys.path.insert(0, MTAH)
    try:
        import load_smart_tagging as lst  # official starter, stdlib only

        return lst.load(data_dir=os.path.dirname(DATA))
    except Exception as exc:  # keep going without tags
        print("tags unavailable:", type(exc).__name__)
        return []


def main():
    from kloppy import metrica
    from possess.epts import to_pitch_m

    os.makedirs(OUT, exist_ok=True)
    ds = metrica.load_tracking_epts(meta_data=os.path.join(DATA, "BAR-ATH_tactical_atd.xml"),
                                    raw_data=os.path.join(DATA, "BAR-ATH_tactical_atd.txt"))
    team_of = {}
    for team in ds.metadata.teams:
        tid = "home" if str(team.ground).lower().endswith("home") else "away"
        for p in team.players:
            team_of[p.player_id] = tid
    by_row = {int(f.frame_id) - 1: f for f in ds.frames}
    gt = load_gt()
    tags = load_tags()

    for name, a, b, has_gt in WINDOWS:
        path = os.path.join(OUT, name + ".csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("# window %s video frames %d-%d, 5 Hz, metres, x in [-52.5,52.5], y in [-34,34]\n"
                     % (name, a, b))
            fh.write("# home = Barcelona (H), away = Athletic (A). gt column: %s\n"
                     % ("public ground truth H/A/D" if has_gt else "none (no labels here)"))
            for tg in tags:
                try:
                    s, e = int(tg.get("start_frame")), int(tg.get("end_frame"))
                except Exception:
                    continue
                if e >= a and s <= b:
                    fh.write("# tag frames %d-%d %s %s\n" % (s, e, tg.get("code", ""), tg.get("tags", {})))
            w = csv.writer(fh)
            w.writerow(["frame", "t_s", "gt", "ball_x", "ball_y", "home", "away"])
            for row in range(a, b + 1, STEP):
                f = by_row.get(row)
                if f is None:
                    continue
                bc = f.ball_coordinates
                bx = by = ""
                if bc is not None and bc.x == bc.x and bc.y == bc.y:
                    bx, by = ("%.1f" % v for v in to_pitch_m(bc.x, bc.y))
                pos = {"home": [], "away": []}
                for pl, pd in f.players_data.items():
                    c = pd.coordinates
                    if c is None or c.x != c.x or c.y != c.y or pl.player_id not in team_of:
                        continue
                    pos[team_of[pl.player_id]].append(to_pitch_m(c.x, c.y))
                enc = {k: "|".join("%.1f:%.1f" % p for p in sorted(v)) for k, v in pos.items()}
                w.writerow([row, "%.1f" % (row / 25.0), gt.get(row, "") if has_gt else "",
                            bx, by, enc["home"], enc["away"]])
        print("wrote", name, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
