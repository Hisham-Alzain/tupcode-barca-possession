"""Per-frame decision evidence of the default pipeline for the viewer's WHY row (viewer/cache/evidence.npz).

    uv run python scripts/ramlock.py -- uv run python scripts/viewer_evidence.py

Runs the challenge pipeline once (pipeline.run_challenge, default flags, feature cache
data/challenge/_runs/features_v1.npz read only) with the opt-in hook decode.DEBUG switched on and
saves, per VIDEO frame (float16):
  <term>   team margin H minus A of each Viterbi emission term (= 2 x its H score; > 0 favours
           Barcelona): dist, rule, touch, shape, fly, bt (and tag / tagr when switched on)
  stop     D score minus the better team score (> 0 = the emissions favour a stoppage); clipped +-20
  vit      Viterbi labels before the post-passes (int8: 0 H, 1 A, 2 D, 3 X)
and checks that the labels equal the given submission (default the final draft). Display only.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from possess import decode, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "challenge"))
    ap.add_argument("--check", default=os.path.join(ROOT, "submissions", "draft_2026-11-17d.csv"))
    a = ap.parse_args()
    decode.DEBUG = {}
    try:
        rows = pipeline.run_challenge(a.data, cache=os.path.join(a.data, "_runs", "features_v1.npz"))
        dbg = decode.DEBUG
    finally:
        decode.DEBUG = None
    n = pipeline.N_FRAMES
    names = []
    for _, d in dbg.get("parts", []):
        names += [k for k in d if k not in names]
    out = {k: np.zeros(n, np.float32) for k in names}
    for off, d in dbg["parts"]:
        for k, v in d.items():
            vf = np.asarray(v, np.float64) * (1.0 if k == "stop" else 2.0)   # H - A = 2 x H score
            lo = off - 1                                                     # array index = video frame + 1
            s0 = max(0, -lo)
            m = min(len(vf), n - lo)
            out[k][lo + s0:lo + m] = vf[s0:m]
    out["stop"] = np.clip(out["stop"], -20, 20)
    vit = np.full(n, 3, np.int8)
    v = dbg["vit"][1:n + 1]
    vit[:len(v)] = np.select([v == "H", v == "A", v == "D"], [0, 1, 2], 3)
    os.makedirs(os.path.join(ROOT, "viewer", "cache"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "viewer", "cache", "evidence.npz"), vit=vit,
                        **{k: v.astype(np.float16) for k, v in out.items()})
    print("evidence.npz: terms %s" % ", ".join(names))
    if a.check and os.path.exists(a.check):
        with open(a.check) as fh:
            next(fh)
            ref = np.array([ln.rstrip("\n").split(",")[1] for ln in fh], dtype="<U1")
        print("labels vs %s: %d of %d frames differ" % (os.path.basename(a.check), int((ref != rows).sum()), n))


if __name__ == "__main__":
    main()
