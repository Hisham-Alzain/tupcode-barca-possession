"""Precompute the viewer's analysis caches (viewer/cache/, git-ignored; reads data/, never writes there).

    uv run python scripts/viewer_cache.py

aim.npz        camera aim per video frame from the tactical homography (DATA.md "Homography"): the
               inverse homography applied to the image centre (960, 540) -> normalised pitch (0-1),
               aim_x in metres = (x - 0.5) * 105; pan = |change of the 1 s-smoothed aim x over 1 s| (m/s).
marks.json     "out": runs of video frames where the visible ball is more than 0.5 m beyond a touchline
               or goal line (pitch 105 x 68, ball from the viewer's features cache, index = frame + 1),
               [first, last, side]; side x+ / x- (goal lines) or y+ / y- (touchlines).
restarts.json  physically detected set-piece restarts from another agent's inventory
               (bp-wt/a12/eval/results/a12_inventory.json, frames there = frame_id = video frame + 1):
               [{frame, type, team ('H'/'A'/''), clip, how, hid}]; skipped when the file is missing.
Display only: nothing here feeds the pipeline or chooses a parameter.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "viewer", "cache")
HOMOG = os.path.join(ROOT, "data", "challenge", "tactical", "BAR-ATH_tactical_homography.csv")
FEAT = os.path.join(ROOT, "data", "challenge", "_runs", "features_v1.npz")
INVENTORY = os.path.join(os.path.dirname(ROOT), "bp-wt", "a12", "eval", "results", "a12_inventory.json")
FPS = 25
HALVES = [(1, 11185, 90651), (2, 110808, 190197)]       # video frames, mirrors pipeline.HALVES
LENGTH, WIDTH, OUT_M = 105.0, 68.0, 0.5


def camera_aim(path=HOMOG):
    """(aim_x, aim_y) in metres (pitch centre = 0) per video frame; NaN where the matrix is invalid."""
    import pandas as pd

    d = pd.read_csv(path)
    n = int(d["frame"].max()) + 1
    M = np.full((n, 3, 3), np.nan)
    M[d["frame"].to_numpy()] = d[["h%d%d" % (i, j) for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    ok = np.isfinite(M).all((1, 2)) & (np.abs(np.linalg.det(np.nan_to_num(M))) > 1e-12)
    p = np.full((n, 3), np.nan)
    p[ok] = np.linalg.solve(M[ok], np.broadcast_to([960.0, 540.0, 1.0], (int(ok.sum()), 3))[..., None])[..., 0]
    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    return (x - 0.5) * LENGTH, (y - 0.5) * WIDTH


def pan_speed(aim_x, fps=FPS):
    """|change of the 1 s moving-average aim x over 1 s| in m/s, centred; NaN where undefined."""
    k = np.ones(fps) / fps
    good = np.isfinite(aim_x)
    s = np.convolve(np.where(good, aim_x, 0.0), k, "same") / np.maximum(np.convolve(good, k, "same"), 1e-9)
    s[np.convolve(good, k, "same") < 0.5] = np.nan
    v = np.full(len(s), np.nan)
    h = fps // 2
    v[h:len(s) - fps + h] = np.abs(s[fps:] - s[:-fps])      # change over fps frames = 1 s
    return v


def out_runs(ball_x, ball_y):
    """[[first, last, side], ...] video frames inside the halves where the ball is > OUT_M beyond a line."""
    bx, by = ball_x[1:], ball_y[1:]                        # cache index = video frame + 1
    with np.errstate(invalid="ignore"):
        ox, oy = np.abs(bx) > LENGTH / 2 + OUT_M, np.abs(by) > WIDTH / 2 + OUT_M
    out = ox | oy
    inh = np.zeros(len(out), bool)
    for _, a, b in HALVES:
        inh[a:b + 1] = True
    out &= inh
    d = np.diff(np.r_[0, out.astype(np.int8), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    runs = []
    for a, b in zip(st, en):
        if ox[a]:
            side = "x+" if bx[a] > 0 else "x-"
        else:
            side = "y+" if by[a] > 0 else "y-"
        if runs and a - runs[-1][1] <= FPS and runs[-1][2] == side:   # one exit flickering at the line = one run
            runs[-1][1] = int(b)
        else:
            runs.append([int(a), int(b), side])
    return runs


def restarts(path=INVENTORY):
    """Detected restart per tagged set piece (hid=0 row if found there, else the hid=1 row)."""
    with open(path) as fh:
        rows = json.load(fh)["rows"]
    best = {}
    for r in sorted(rows, key=lambda r: r.get("hid", 0)):
        if r.get("found") and "kick" in r and r["a"] not in best:
            best[r["a"]] = r
    team = {1: "H", -1: "A"}
    return sorted(({"frame": int(r["kick"]) - 1, "type": r["type"], "team": team.get(r.get("sign"), ""),
                    "clip": int(r["a"]) - 1, "how": r.get("how", ""), "hid": int(r.get("hid", 0))}
                   for r in best.values()), key=lambda e: e["frame"])


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(HOMOG):
        ax, ay = camera_aim()
        pan = pan_speed(ax)
        np.savez_compressed(os.path.join(OUT, "aim.npz"), aim_x=ax.astype(np.float32),
                            aim_y=ay.astype(np.float32), pan=pan.astype(np.float32))
        print("aim.npz: %d frames, pan median %.2f m/s" % (len(ax), np.nanmedian(pan)))
    else:
        print("homography not found: aim.npz skipped", file=sys.stderr)
    f = np.load(FEAT)
    marks = {"out": out_runs(f["ball_x"], f["ball_y"])}
    with open(os.path.join(OUT, "marks.json"), "w") as fh:
        json.dump(marks, fh)
    print("marks.json: %d line crossings" % len(marks["out"]))
    if os.path.exists(INVENTORY):
        try:
            rs = restarts()
            with open(os.path.join(OUT, "restarts.json"), "w") as fh:
                json.dump(rs, fh)
            print("restarts.json: %d detected restarts" % len(rs))
        except (KeyError, ValueError, TypeError) as e:
            print("inventory unusable (%s): restarts.json skipped" % e, file=sys.stderr)
    else:
        print("inventory not found: restarts.json skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
