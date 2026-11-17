"""Local possession viewer: the tactical video next to the pipeline's per-frame labels.

    uv run python viewer/server.py --video "C:/path/to/BAR-ATH_tactical.mp4"
    then open http://127.0.0.1:8765

Shows where the pipeline changes possession (every spell start, turnovers highlighted), the tracking
(players + ball) on a mini pitch, the distance evidence the decoder uses, and lets you run the
pipeline (`possess`) from the page and load its output. Binds to 127.0.0.1 only: the match data is
licensed to enrolled participants and must never leave this machine (docs/RULES.md). Nothing here
tunes or fits anything; the public 5-minute truth is shown only as a check (organisers' rules).
Standard library + numpy only.
"""
import argparse
import datetime as dt
import glob
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")              # analysis caches (scripts/viewer_cache.py, git-ignored)
SUB_DIR = os.path.join(ROOT, "submissions")
RUNS = os.path.join(ROOT, "data", "challenge", "_runs")
PUBLIC_GT = os.path.join(ROOT, "third_party", "mtah", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
TAGS_XML = os.path.join(ROOT, "data", "challenge", "tactical", "BAR-ATH_tactical_smart_tagging.xml")
FPS = 25
HALVES = [(1, 11185, 90651), (2, 110808, 190197)]       # video frames, mirrors pipeline.HALVES

_cache = {}
_run = {"proc": None, "out": None, "log": None, "started": None}
_lock = threading.Lock()


def read_labels(path):
    """Submission CSV (frame,label) -> numpy array of 'H','A','D','X' indexed by video frame."""
    key = ("lab", path, os.path.getmtime(path))
    if key not in _cache:
        with open(path) as fh:
            next(fh)
            lab = np.array([ln.rstrip("\n").split(",")[1] for ln in fh], dtype="<U1")
        _cache[key] = lab
    return _cache[key]


def segments(lab, start=0):
    """Run-length encode -> [[first_frame, last_frame, label], ...]."""
    if len(lab) == 0:
        return []
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut - 1, len(lab) - 1]
    return [[int(a) + start, int(b) + start, str(lab[a])] for a, b in zip(starts, ends)]


def changes(segs):
    """Possession-change events. kind: 'turnover' (team spell H<->A, through D or not),
    'dead' (play stops), 'restart' (play resumes with the same team), 'start' (first team spell)."""
    out, last_team = [], None
    for a, b, l in segs:
        if l in "HA":
            kind = "start" if last_team is None else "turnover" if l != last_team else "restart"
            out.append({"frame": a, "label": l, "kind": kind, "dur": (b - a + 1) / FPS})
            last_team = l
        elif l == "D":
            out.append({"frame": a, "label": l, "kind": "dead", "dur": (b - a + 1) / FPS})
        elif l == "X":
            last_team = None                       # halves reset
    return out


def features():
    if "feat" not in _cache:
        f = np.load(os.path.join(RUNS, "features_v1.npz"))
        p = np.load(os.path.join(RUNS, "positions_v1.npz"))["pos"]
        _cache["feat"] = {k: f[k] for k in ("ball_x", "ball_y", "d_home", "d_away")}
        _cache["pos"] = p
    return _cache["feat"], _cache["pos"]


def smart_tags():
    """Smart Tagging clips -> [{a, b, code, team, type}] in video frames (frame = round(seconds * 25),
    as src/possess/kickoff.py). team: 'H' Barcelona, 'A' Athletic, '' otherwise; type: SET PIECES only."""
    key = ("tags", os.path.getmtime(TAGS_XML))
    if key not in _cache:
        out = []
        for inst in ET.parse(TAGS_XML).getroot().iter("instance"):
            lab = {}
            for l in inst.findall("label"):
                g, t = l.findtext("group"), l.findtext("text")
                if g and t:
                    lab[g.strip().lower()] = t.strip()
            code = (inst.findtext("code") or "").strip()
            team = {"barcelona": "H", "atletic club": "A"}.get(lab.get("team", "").lower(), "")
            out.append({"a": int(round(float(inst.findtext("start")) * FPS)),
                        "b": int(round(float(inst.findtext("end")) * FPS)), "code": code, "team": team,
                        "type": lab.get("type", "") if code == "SET PIECES" else ""})
        out.sort(key=lambda t: (t["a"], t["b"]))
        _cache[key] = out
    return _cache[key]


def r1(a, nd=1):
    """Round to `nd` decimals and turn NaN into None for JSON."""
    a = np.round(np.asarray(a, dtype=float), nd)
    return [None if np.isnan(v) else float(v) for v in a.ravel()]


def cached(name, load):
    """viewer/cache/<name> through `load(path)`, memoised by mtime; None when the file is missing or
    unreadable (scripts/viewer_cache.py / viewer_evidence.py build them; every row fails soft)."""
    p = os.path.join(CACHE, name)
    try:
        key = ("cache", name, os.path.getmtime(p))
        if key not in _cache:
            _cache[key] = load(p)
        return _cache[key]
    except (OSError, ValueError, KeyError):
        return None


def campan():
    """Camera pan speed (m/s x 10, rounded; -1 = unknown) as means over 5-frame blocks (5 Hz)."""
    def load(p):
        pan = np.load(p)["pan"].astype(float)
        k = len(pan) // 5 * 5
        m = np.nanmean(pan[:k].reshape(-1, 5), 1)        # all-NaN block -> NaN (warning silenced)
        return {"step": 5, "pan10": [-1 if not np.isfinite(v) else int(round(v * 10)) for v in m]}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return cached("aim.npz", load)


def read_json(p):
    with open(p) as fh:
        return json.load(fh)


def evidence(frame, half):
    z = cached("evidence.npz", lambda p: (lambda z: {k: z[k] for k in z.files})(np.load(p)))
    if z is None:
        return {"n": 0}
    n = len(z["stop"])
    a, b = max(0, frame - half), min(n, frame + half + 1)
    names = [k for k in ("bt", "touch", "dist", "rule", "fly", "shape", "tag", "tagr") if k in z]
    return {"start": a, "n": max(0, b - a), "names": names,
            "terms": {k: r1(z[k][a:b], 2) for k in names}, "stop": r1(z["stop"][a:b], 2),
            "vit": z["vit"][a:b].tolist()}


def submission_files():
    files = sorted(glob.glob(os.path.join(SUB_DIR, "*.csv")), key=os.path.getmtime, reverse=True)
    return [os.path.basename(p) for p in files]


def safe_sub(name):
    if not name or not re.fullmatch(r"[\w.\-]+\.csv", name):
        raise ValueError("bad file name")
    p = os.path.join(SUB_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(name)
    return p


def run_status():
    with _lock:
        pr = _run["proc"]
        st = {"running": pr is not None and pr.poll() is None, "out": _run["out"], "started": _run["started"]}
        if pr is not None and pr.poll() is not None:
            st["returncode"] = pr.returncode
        if _run["log"] and os.path.exists(_run["log"]):
            with open(_run["log"], errors="replace") as fh:
                st["log_tail"] = fh.read()[-2000:]
    return st


def start_run():
    with _lock:
        if _run["proc"] is not None and _run["proc"].poll() is None:
            return {"error": "a run is already going"}
        os.makedirs(SUB_DIR, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(SUB_DIR, "viewer_run_%s.csv" % stamp)
        log = os.path.join(SUB_DIR, "viewer_run_%s.log" % stamp)
        cmd = ["uv", "run", "possess", "--data", os.path.join("data", "challenge"), "--out", out]
        fh = open(log, "w")
        _run.update(proc=subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT),
                    out=os.path.basename(out), log=log, started=stamp)
    return run_status()


class Handler(SimpleHTTPRequestHandler):
    video = None

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):         # quiet: video range requests are many
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def local_only(self):
        """Refuse other host names (DNS rebinding) and cross-site requests: the data must stay local."""
        host = self.headers.get("Host", "").rsplit(":", 1)[0]
        origin = self.headers.get("Origin")
        ok = host in ("127.0.0.1", "localhost") and (
            origin is None or urllib.parse.urlparse(origin).hostname in ("127.0.0.1", "localhost"))
        if not ok:
            self.send_json({"error": "local requests only"}, 403)
        return ok

    def do_POST(self):
        if not self.local_only():
            return
        if self.path == "/api/run":
            try:
                return self.send_json(start_run())
            except OSError as e:                    # e.g. uv not on PATH
                return self.send_json({"error": "could not start the pipeline: %s" % e}, 500)
        self.send_json({"error": "not found"}, 404)

    def do_GET(self):
        if not self.local_only():
            return
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == "/video":
                return self.send_video()
            if u.path == "/api/info":
                return self.send_json({"files": submission_files(), "fps": FPS, "halves": HALVES,
                                       "video": bool(self.video and os.path.exists(self.video)),
                                       "video_path": self.video, "gt": os.path.exists(PUBLIC_GT),
                                       "run": run_status()})
            if u.path == "/api/labels":
                lab = read_labels(safe_sub(q.get("file")))
                segs = segments(lab)
                return self.send_json({"n": int(len(lab)), "segments": segs, "changes": changes(segs)})
            if u.path == "/api/score":              # official scorer on the public 5-min window (check only)
                if not os.path.exists(PUBLIC_GT):
                    return self.send_json({"error": "public truth not found"}, 404)
                sys.path.insert(0, os.path.join(ROOT, "vendor", "mtah"))
                import possession_metrics as pm
                gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
                pred = pm.label_codes(read_labels(safe_sub(q.get("file"))))
                r = pm.evaluate(gt, pred, bootstrap_n=0)
                return self.send_json({"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                                       "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]})
            if u.path == "/api/gt":                 # public 5 min: a correctness check only
                if not os.path.exists(PUBLIC_GT):
                    return self.send_json({"segments": []})
                with open(PUBLIC_GT) as fh:
                    head = next(fh).strip().split(",")
                    rows = [ln.rstrip("\n").split(",") for ln in fh if ln.strip()]
                fi, li = head.index("frame") if "frame" in head else 0, len(head) - 1
                fr = np.array([int(r[fi]) for r in rows])
                lab = np.array([r[li] for r in rows], dtype="<U1")
                return self.send_json({"segments": segments(lab, int(fr[0])) if len(fr) else []})
            if u.path == "/api/track":              # video frames [start, start+n)
                feat, pos = features()
                s, n = max(0, int(q.get("start", 0))), max(0, min(int(q.get("n", 250)), 2500))
                i0 = min(s + 1, len(feat["ball_x"]))                  # cache index = frame_id = frame + 1
                i1 = min(s + 1 + n, len(feat["ball_x"]))
                return self.send_json({"start": s, "n": i1 - i0,
                                       "ball": r1(np.stack([feat["ball_x"][i0:i1], feat["ball_y"][i0:i1]], 1)),
                                       "pos": r1(pos[i0:i1]), "d_home": r1(feat["d_home"][i0:i1]),
                                       "d_away": r1(feat["d_away"][i0:i1])})
            if u.path == "/api/ballgaps":            # [first, last] video frames with no ball position
                feat, _ = features()
                miss = np.isnan(feat["ball_x"][1:]).astype(np.int8)  # cache index i = video frame i - 1
                d = np.diff(np.r_[0, miss, 0])
                st, en = np.where(d == 1)[0], np.where(d == -1)[0] - 1
                return self.send_json({"gaps": [[int(a), int(b)] for a, b in zip(st, en)]})
            if u.path == "/api/tags":               # Smart Tagging clips (video frames)
                if not os.path.exists(TAGS_XML):
                    return self.send_json([])
                return self.send_json(smart_tags())
            if u.path == "/api/campan":             # camera pan speed, whole match, 5 Hz ({} = no cache)
                return self.send_json(campan() or {})
            if u.path == "/api/marks":              # ball line crossings + detected set-piece restarts
                m = cached("marks.json", read_json) or {}
                return self.send_json({"out": m.get("out", []), "restarts": cached("restarts.json", read_json) or []})
            if u.path == "/api/evidence":           # decision evidence around a frame ({"n": 0} = no cache)
                return self.send_json(evidence(max(0, int(q.get("frame", 0))),
                                               max(0, min(int(q.get("half_window", 750)), 1500))))
            if u.path == "/api/run":
                return self.send_json(run_status())
        except (ValueError, FileNotFoundError) as e:
            return self.send_json({"error": str(e)}, 400)
        return super().do_GET()

    def send_video(self):
        if not self.video or not os.path.exists(self.video):
            return self.send_json({"error": "no video; start the server with --video PATH"}, 404)
        size = os.path.getsize(self.video)
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        a, b = 0, size - 1
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else size - 1
            else:                                                  # suffix range: the last N bytes
                a = max(0, size - int(m.group(2))) if int(m.group(2)) else size
            if a >= size:                                          # unsatisfiable
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                return self.end_headers()
            if b < a:                                              # malformed: ignore the Range header
                m, a, b = None, 0, size - 1
        if m and (m.group(1) or m.group(2)):
            b = min(b, a + 8 * 1024 * 1024 - 1, size - 1)          # 8 MB chunks
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(self.video)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        try:
            with open(self.video, "rb") as fh:
                fh.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = os.path.join(ROOT, "data", "challenge", "tactical", "BAR-ATH_tactical.mp4")
    ap.add_argument("--video", default=os.environ.get("MTAH_VIDEO", default))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    Handler.video = a.video
    if not os.path.exists(a.video):
        print("warning: video not found at %s (labels and pitch still work)" % a.video, file=sys.stderr)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print("possession viewer on http://127.0.0.1:%d  (Ctrl+C to stop)" % a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
