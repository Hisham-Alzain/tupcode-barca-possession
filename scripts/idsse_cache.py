"""IDSSE (DFL) matches for the external bench: stream-parse once, cache compact arrays.

Started from MatchLens eval/sportec.py (footballtactics, not in this repo) (SportecMatch.iter_frames, _is_alive,
_is_goalkeeper, _team_id) and footballtactics evaluation/adapters/metrica_replay.py
build_clip(dataset="sportec"). MatchLens reads these files through kloppy, which builds every
frame object in memory (~400 MB XML each) and maps the ball state only to ALIVE/DEAD; here the
raw DFL XML is stream-parsed with iterparse so BallPossession is kept as per-frame truth and the
parse runs once. Same conventions as sportec.py: pitch-centred metres, x along the length (DFL raw
positions already are), conf 1.0, goalkeepers by position "TW", team 1 = home.

Row layout: first half rows 0..n1-1 (frame N - first ball N of the half), then GAP_ROWS rows of
"X" (never yielded, so no possession spell can cross half time), then the second half.
IDSSE is 25 Hz (T steps 0.04 s), the bench rate, so no resampling.

Truth per row: BallStatus 0 -> "D"; else BallPossession 1 -> "H", 2 -> "A"; rows outside the two
halves -> "X".
"""
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE_DIR = os.path.join(ROOT, "data", "external", "idsse")
CACHE = os.path.join(ROOT, "data", "external", "_bench")
MATCHES = {"idsse1": "J03WMX", "idsse2": "J03WPY"}
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
GAP_ROWS = 25
FPS = 25
VERSION = 1


def _find(kind, mid):
    pref = {"meta": "DFL_02_01_matchinformation", "pos": "DFL_04_03_positions_raw_observed"}[kind]
    for f in os.listdir(IDSSE_DIR):
        if f.startswith(pref) and f.endswith("DFL-MAT-%s.xml" % mid):
            return os.path.join(IDSSE_DIR, f)
    raise FileNotFoundError("%s for %s in %s" % (kind, mid, IDSSE_DIR))


def read_matchinfo(path):
    """(home TeamId, guest TeamId, set of goalkeeper PersonIds)."""
    home = guest = None
    gks = set()
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "Team":
            if el.get("Role") == "home":
                home = el.get("TeamId")
            elif el.get("Role") == "guest":
                guest = el.get("TeamId")
        elif el.tag == "Player" and el.get("PlayingPosition") == "TW":
            gks.add(el.get("PersonId"))
    if home is None or guest is None:
        raise ValueError("matchinformation without home/guest roles: %s" % path)
    return home, guest, gks


def truth_labels(status, possession):
    """BallStatus 0 -> D; else BallPossession 1 -> H, 2 -> A; anything else -> X."""
    status, possession = np.asarray(status), np.asarray(possession)
    lab = np.full(len(status), "X", dtype="<U1")
    alive = status != 0
    lab[alive & (possession == 1)] = "H"
    lab[alive & (possession == 2)] = "A"
    lab[~alive] = "D"
    return lab


def parse(pos_path, meta_path):
    """Stream-parse a DFL positions XML into row-indexed arrays (see module docstring)."""
    home, guest, gks = read_matchinfo(meta_path)
    persons, sets = {}, []          # PersonId -> column; (section, kind, col, N, x, y, extra...)
    cur = None
    ctx = ET.iterparse(pos_path, events=("start", "end"))
    _, root = next(ctx)
    for ev, el in ctx:
        tag = el.tag
        if ev == "start":
            if tag == "FrameSet":
                team, pid = el.get("TeamId"), el.get("PersonId")
                sec = SECTIONS.get(el.get("GameSection"))
                if sec is None or team not in (home, guest, "BALL"):
                    cur = None       # referees, extra time sections: skipped
                else:
                    cur = dict(sec=sec, team=team, pid=pid, n=[], x=[], y=[], st=[], po=[])
            continue
        if tag == "Frame":
            if cur is not None:
                a = el.attrib
                cur["n"].append(int(a["N"])); cur["x"].append(float(a["X"])); cur["y"].append(float(a["Y"]))
                if cur["team"] == "BALL":
                    cur["st"].append(int(a.get("BallStatus", 1)))
                    cur["po"].append(int(a.get("BallPossession", 0) or 0))
            el.clear()
        elif tag == "FrameSet":
            if cur is not None:
                rec = dict(sec=cur["sec"], team=cur["team"], n=np.array(cur["n"], np.int64),
                           x=np.array(cur["x"], np.float32), y=np.array(cur["y"], np.float32))
                if cur["team"] == "BALL":
                    rec["st"] = np.array(cur["st"], np.int8); rec["po"] = np.array(cur["po"], np.int8)
                else:
                    rec["col"] = persons.setdefault(cur["pid"], len(persons))
                sets.append(rec)
            cur = None
            el.clear()
            root.clear()
    balls = {s["sec"]: s for s in sets if s["team"] == "BALL"}
    if set(balls) != {1, 2}:
        raise ValueError("expected a BALL frameset for both halves, got %s" % sorted(balls))
    start, offset = {}, {}
    row = 0
    for sec in (1, 2):
        n0, n1 = int(balls[sec]["n"].min()), int(balls[sec]["n"].max())
        start[sec], offset[sec] = (n0, n1), row
        row += n1 - n0 + 1 + (GAP_ROWS if sec == 1 else 0)
    n = row
    P = len(persons)
    period = np.zeros(n, np.int8)
    status = np.zeros(n, np.int8); poss = np.zeros(n, np.int8)
    seen = np.zeros(n, bool)
    bx = np.full(n, np.nan, np.float32); by = np.full(n, np.nan, np.float32)
    pos = np.full((n, P, 2), np.nan, np.float32)
    team = np.zeros(P, np.int8); gk = np.zeros(P, bool)
    for pid, c in persons.items():
        gk[c] = pid in gks

    def rows_of(s):
        n0, n1 = start[s["sec"]]
        ok = (s["n"] >= n0) & (s["n"] <= n1)
        return offset[s["sec"]] + (s["n"][ok] - n0), ok

    for sec in (1, 2):
        n0, n1 = start[sec]
        period[offset[sec]:offset[sec] + n1 - n0 + 1] = sec
    for s in sets:
        r, ok = rows_of(s)
        if s["team"] == "BALL":
            seen[r] = True
            bx[r], by[r] = s["x"][ok], s["y"][ok]
            status[r], poss[r] = s["st"][ok], s["po"][ok]
        else:
            c = s["col"]
            team[c] = 1 if s["team"] == home else 2
            pos[r, c, 0], pos[r, c, 1] = s["x"][ok], s["y"][ok]
    truth = truth_labels(status, poss)
    truth[~seen] = "X"
    return dict(truth=truth, period=period, present=seen, ball_x=bx, ball_y=by, pos=pos,
                team=team, gk=gk, status=status, possession=poss)


def load(name):
    """Cached arrays for "idsse1" / "idsse2"; parses the XML on the first call."""
    path = os.path.join(CACHE, "idsse_%s_v%d.npz" % (name, VERSION))
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    mid = MATCHES[name]
    t0 = time.time()
    d = parse(_find("pos", mid), _find("meta", mid))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **d)
    print("parsed %s (%s) in %.0fs -> %s" % (name, mid, time.time() - t0, os.path.basename(path)))
    return d


class IdsseMatch:
    """Duck-typed like AtdMatch / MatchLens SportecMatch for replay_match and the bench:
    `.name`, `.fps`, `.events`, `.iter_frames(max_frames, interpolate_ball)`."""

    def __init__(self, name, data=None):
        self.name, self.fps, self.events = name, FPS, []
        self.d = data if data is not None else load(name)

    def iter_frames(self, max_frames=None, interpolate_ball=False):
        d = self.d
        pos, team, gk = d["pos"], d["team"], d["gk"]
        count = 0
        for f in np.flatnonzero(d["period"] > 0):
            players, keepers = {}, {}
            p = pos[f]
            for c in np.flatnonzero(~np.isnan(p[:, 0])):
                rec = {"map_position": (float(p[c, 0]), float(p[c, 1])), "team": int(team[c]), "conf": 1.0}
                if gk[c]:
                    rec["role"] = "goalkeeper"
                    keepers[int(c) + 1] = rec
                else:
                    players[int(c) + 1] = rec
            ok = bool(d["present"][f]) and not np.isnan(d["ball_x"][f])
            ball = {1: {"map_position": (float(d["ball_x"][f]), float(d["ball_y"][f])), "conf": 1.0}} if ok else {}
            yield int(f), int(d["period"][f]), {"players": players, "goalkeepers": keepers, "ball": ball}, ok
            count += 1
            if max_frames and count >= max_frames:
                return


def sanity(truth, fps=FPS):
    """Share of H/A/D, number of possessions (maximal H or A runs) and their mean length in s."""
    t = np.asarray(truth)
    n = max(1, int(np.isin(t, ["H", "A", "D"]).sum()))
    runs, cur, L = [], None, 0
    for c in t:
        if c == cur:
            L += 1
            continue
        if cur in ("H", "A"):
            runs.append(L)
        cur, L = c, 1
    if cur in ("H", "A"):
        runs.append(L)
    return {"H": float((t == "H").sum() / n), "A": float((t == "A").sum() / n),
            "D": float((t == "D").sum() / n), "possessions": len(runs),
            "mean_s": float(np.mean(runs) / fps) if runs else 0.0}


if __name__ == "__main__":
    import sys
    for nm in sys.argv[1:] or list(MATCHES):
        t0 = time.time()
        dd = load(nm)
        print(nm, "rows", len(dd["truth"]), "persons", dd["pos"].shape[1], "load %.1fs" % (time.time() - t0),
              sanity(dd["truth"]))
