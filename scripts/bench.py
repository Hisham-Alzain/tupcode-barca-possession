"""External evaluation bench for possession (LOOP.md protocol).

    python scripts/bench.py --pipeline possess --split tune
    python scripts/bench.py --pipeline possess --split heldout     # counts as a held-out look

Per external match:
  truth   H/A/D per frame. Metrica: ball absent = dead (Metrica convention), otherwise the team of
          the last event (loose events BALL LOST / BALL OUT / CHALLENGE keep the previous team:
          "a loose ball stays with the last team"), ported from footballtactics
          evaluation/adapters/metrica_replay.py:_ground_truth_possession.
          IDSSE (idsse1/idsse2, scripts/idsse_cache.py): BallStatus 0 = dead, else BallPossession
          1 = H, 2 = A; the ball is IDSSE's own (tracked through stoppages, no show_dead_ball).
  degrade ball hidden in blocks drawn from the challenge gap-length histogram
          (eval/results/matchlens_physics_public5.json) until ~40% of frames are hidden.
  score   official possession_metrics.evaluate (spellF1 mean + bootstrap CI, macroF1, counts).
  explain every unmatched true possession and every spurious predicted one, by cause.
Caches in data/external/_bench/ (gitignored).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "mtah"))   # official scorer (MIT), vendored
import possession_metrics as pm  # noqa: E402
from possess.epts import AtdMatch  # noqa: E402

CACHE = os.path.join(ROOT, "data", "external", "_bench")
METRICA = os.path.join(ROOT, "data", "external", "metrica")
G3 = os.path.join(METRICA, "_sd", "data", "Sample_Game_3")
SPLITS = {"tune": ["metrica1", "metrica2", "idsse1"], "heldout": ["metrica3", "idsse2"]}
LOOSE = {"BALL LOST", "BALL OUT", "CHALLENGE"}
# challenge gap histogram: (lo_s, hi_s, n_gaps)
GAP_HIST = [(0.04, 0.5, 15), (0.5, 1.0, 1), (5, 10, 53), (10, 30, 59), (30, 60, 19)]
HIDE_FRAC = 0.40
FPS = 25


# ----------------------------------------------------------------------------- matches
def load_match(name):
    if name.startswith("idsse"):
        from idsse_cache import IdsseMatch

        return IdsseMatch(name)
    if name.startswith("metrica") and name != "metrica3":
        from possess.metrica_csv import load_metrica_match

        return load_metrica_match(METRICA, int(name[-1]))
    return AtdMatch(os.path.join(G3, "Sample_Game_3_metadata.xml"),
                    os.path.join(G3, "Sample_Game_3_tracking.txt"), "metrica3")


def g3_events():
    d = json.load(open(os.path.join(G3, "Sample_Game_3_events.json")))
    out = []
    for e in d["data"]:
        tid = (e.get("team") or {}).get("id")
        team = 1 if tid == "FIFATMA" else 2 if tid == "FIFATMB" else None
        out.append((int(e["start"]["frame"]), e["type"]["name"], team))
    return sorted(out)


def build_truth(name, match):
    """Return frame-indexed arrays: truth labels (U1), ball_present, period, ball x, ball y."""
    fids, present, period, bxs, bys = [], [], [], [], []
    for fid, per, tracks, bp in match.iter_frames():
        fids.append(fid); present.append(bp); period.append(per)
        bx, by = tracks["ball"][1]["map_position"] if bp else (np.nan, np.nan)
        bxs.append(bx); bys.append(by)
    n = max(fids) + 1
    bp_arr = np.zeros(n, bool); per_arr = np.zeros(n, np.int8); seen = np.zeros(n, bool)
    bp_arr[fids] = present; per_arr[fids] = period; seen[fids] = True
    bx_arr = np.full(n, np.nan); by_arr = np.full(n, np.nan)
    bx_arr[fids] = bxs; by_arr[fids] = bys
    if name == "metrica3":
        marks = [(f, ("contested" if t in LOOSE else (team or 0))) for f, t, team in g3_events()]
    else:
        marks = [(int(e.start_frame), "contested" if e.raw_type in LOOSE else (e.team_id or 0))
                 for e in match.events]
    marks.sort(key=lambda m: m[0])
    lab = np.array(["X"] * n, dtype="<U1")
    cur, last_team, j = None, None, 0
    for f in range(n):
        while j < len(marks) and marks[j][0] <= f:
            s = marks[j][1]
            if s in (1, 2):
                cur = last_team = s
            elif s == "contested":
                cur = last_team
            j += 1
        if not seen[f]:
            continue
        if not bp_arr[f]:
            lab[f] = "D"
        elif cur in (1, 2):
            lab[f] = "H" if cur == 1 else "A"
    return lab, bp_arr, per_arr, bx_arr, by_arr


def show_dead_ball(bx, by, period):
    """Metrica drops the ball while play is dead; the challenge tracking keeps it ~72% of D frames
    (swarm F1). Make it visible: straight line from where it was last seen to where play restarts,
    within a period. Same idea as footballtactics metrica.py iter_frames(interpolate_ball=True)."""
    vx, vy = bx.copy(), by.copy()
    known = ~np.isnan(bx)
    i, n = 0, len(bx)
    while i < n:
        if known[i]:
            i += 1
            continue
        j = i
        while j < n and not known[j]:
            j += 1
        if i > 0 and j < n and period[i - 1] == period[j] and period[j] > 0:
            s = np.arange(1, j - i + 1) / (j - i + 1)
            vx[i:j] = bx[i - 1] + (bx[j] - bx[i - 1]) * s
            vy[i:j] = by[i - 1] + (by[j] - by[i - 1]) * s
        i = j
    return vx, vy


def hide_mask(n, seed):
    rng = np.random.default_rng(seed)
    lens = []
    w = np.array([g[2] for g in GAP_HIST], float); w /= w.sum()
    while sum(lens) < HIDE_FRAC * n:
        lo, hi, _ = GAP_HIST[rng.choice(len(GAP_HIST), p=w)]
        lens.append(int(rng.uniform(lo, hi) * FPS) + 1)
    mask = np.zeros(n, bool)
    for L in lens:
        for _ in range(50):
            s = int(rng.integers(0, max(1, n - L)))
            if not mask[max(0, s - 5):s + L + 5].any():
                mask[s:s + L] = True
                break
    return mask


# ----------------------------------------------------------------------------- realistic bench (L2-12)
# Envelope L2-12 (audit): the default bench draws the dead ball as a straight fill line (Metrica) or
# keeps IDSSE's ball frozen at speed 0, and its visible ball has no tracking faults. The variants
# below replace ONLY the ball the pipeline sees (truth and the hidden-block mask are unchanged, so a
# score difference comes from ball behaviour alone). Default behaviour (variant "") is untouched.
#   dead1  dead-ball model only: roll-out, carried on a wandering path, placed shortly before the restart
#   real1  dead1 + challenge-like tracking faults (sample-and-hold, long freezes, jumps, noise)
# FAULTS is calibrated on INPUT statistics of the provided match only (scripts/transfer_check.py:
# exact_repeat_share, repeat runs, freezes per minute, jumps per minute, ball_noise_m); no ground truth.
# DEAD_MODEL wander chosen so the bench INPUT speed statistics meet the provided match's (still < 0.3 m/s
# 0.045 of visible frames, < 1 m/s 0.133, median 4.1 m/s): idsse1 0.045 / 0.127 / 4.13, metrica1 0.082 /
# 0.168 / 3.83 (wander 2 m: 0.07-0.10 still; 6 m: 0.03-0.07). eval/results/audit_transfer*.json.
DEAD_MODEL = dict(roll_tau_s=0.8, roll_s=(0.5, 2.0), place_s=(0.3, 1.5), wander_m=4.0, wander_corr_s=1.0,
                  max_frac=0.3)
# Measured on the provided match's input: exact repeats 0.1% of frames, no freeze >= 0.4 s, not piecewise
# linear, so sample-and-hold and long freezes stay OFF (options kept); steps > 35 m/s 4.8 per minute vs
# 1.1-3.2 on the bench -> about one small jump per minute added (each jump makes two fast steps).
FAULTS = dict(hold_share=0.0, hold_block_s=(2.0, 10.0), hold_k=5,       # sample-and-hold at fps / hold_k
              freeze_per_min=0.0, freeze_s=(0.4, 3.0),                  # long freezes
              jump_per_min=1.0, jump_m=(1.5, 2.4), jump_frames=(1, 3),  # teleports
              noise_m=0.0)                                              # white position noise per axis
VARIANTS = {"dead1": dict(dead=True, faults=False), "real1": dict(dead=True, faults=True),
            "fault1": dict(dead=False, faults=True)}


# ----------------------------------------------------------------------------- realistic bench 2 (L2-20)
# Envelope L2-20, .planning/research/provided-vs-bench-gaps.md: real1 hides the ball in blocks placed at
# random, shows an unsmoothed ball and never loses a player. On the provided match's INPUT the hidden
# blocks start on a slow ball, end on a kick and sit on slow-player (stoppage) stretches; the ball is a
# smoothed track; long blackouts exist; players go missing in hidden frames. `real2` = real1's dead-ball
# model and jumps, plus:
#   hide_mask_coupled   blocks tied to the bench's own true stoppages + live blocks, lengths from the
#                       provided gap-length list (GAP_FRAMES), entry on a slow ball, exit on a kick
#   smooth_ball         Gaussian low-pass of the visible ball inside each visible run
#   lose_players        close-up camera in hidden stoppage blocks + count flicker
# Every constant in REAL2 is set by matching INPUT aggregates of the provided match (ball track, gap
# lengths, player counts, fast-player speed; scripts/audit_bench.py inputs); no label of the provided
# match is read anywhere. The bench's own (external) truth places the blocks; truth itself is unchanged.
# Gap lengths (frames at 25 fps) of the provided ball inside the graded halves (158857 frames).
GAP_FRAMES = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 10, 12, 22, 122, 129, 132, 132, 132, 134, 135, 135, 136,
              137, 137, 140, 144, 146, 146, 148, 148, 149, 152, 152, 152, 153, 159, 163, 164, 169, 169, 169, 170,
              176, 177, 177, 182, 183, 183, 187, 190, 194, 194, 200, 201, 205, 207, 220, 222, 223, 229, 229, 232,
              234, 239, 240, 242, 245, 260, 265, 266, 268, 268, 269, 273, 275, 275, 280, 287, 289, 289, 292, 297,
              298, 301, 305, 311, 319, 320, 350, 366, 376, 380, 390, 406, 407, 414, 418, 430, 432, 462, 485, 488,
              502, 509, 511, 512, 521, 522, 528, 529, 544, 558, 573, 577, 577, 603, 604, 606, 611, 625, 627, 639,
              722, 729, 743, 743, 744, 751, 779, 788, 836, 929, 976, 1055, 1073, 1124, 1141, 1260, 1278, 1324,
              1353, 1391, 1535, 1737, 3466, 5537]
GAP_N = 158857
REAL2 = dict(
    hide_frac=0.407,            # hidden share of the played frames (provided: 0.407 in both halves)
    smooth_sigma=7.0,           # frames; ball low-pass (targets: accel p90 18 / p99 37 m/s2, kinks 8 / visible min)
    noise_m=0.002,              # white jitter per axis after smoothing (target: accel p50 5.7 m/s2)
    tail_s=40.0, tail_cand=20,  # gaps longer than this are placed first, on the best of tail_cand windows (most true D)
    stop_min_s=3.0, p_hide=0.9,   # each true stoppage >= stop_min_s becomes a hidden block with this probability
    stop_long_s=30.0, p_hide_long=0.5,           # ... stoppages longer than stop_long_s with this one (the tail covers them)
    fill_cand=6,                # a fill block takes the window with the most true D of fill_cand random ones
    margin_s=0.5,               # shortest visible stretch left between two blocks
    entry_v=1.5, entry_max_s=2.0, q_entry=0.85,  # block start snapped to a slow ball (m/s, search, probability)
    kick_v=13.0, kick_max_s=3.0, q_exit=0.8,     # block end snapped to the frame before the ball flies (m/s over 0.2 s)
    flick_per_s=0.06, flick_hid=2.0, flick_mean_s=8.5, flick_far_m=15.0,   # single-player dropouts (x flick_hid when hidden)
    close_min_s=10.0, p_close=0.27, close_part=(0.4, 1.0), close_r_m=(8.0, 20.0),   # close-up camera in hidden stoppage blocks
)
VARIANTS["real2"] = dict(dead=True, faults=True, real2=True)


def bench_positions(name, n):
    """Player positions [n, 2, SLOTS, 2] of an external match (same cache as tune_shape.positions)."""
    from possess import shape

    path = os.path.join(CACHE, "shapepos_%s_v1.npz" % name)
    if os.path.exists(path):
        return np.load(path)["pos"]
    pos = shape.collect(load_match(name).iter_frames(), n=n)
    np.savez_compressed(path, pos=pos)
    return pos


def _runs(mask):
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def nearest_player(pos, bx, by):
    """Distance (m) from the ball to the nearest player of either team (nan: no ball)."""
    d = np.hypot(pos[:, :, :, 0] - bx[:, None, None], pos[:, :, :, 1] - by[:, None, None])
    d = np.where(np.isnan(d), np.inf, d).reshape(len(bx), -1).min(axis=1)
    return np.where(np.isfinite(d), d, np.nan)


def hide_mask_coupled(truth, period, bx, by, seed, fps=FPS, **kw):
    """Hidden-ball mask tied to the state of play (envelope L2-20). bx, by: the full ball (before hiding).

    1. tail: gaps longer than tail_s go first, each on the window (of tail_cand random ones) holding the
       most true stoppage time;
    2. stoppages: every true stoppage >= stop_min_s not yet half hidden becomes a block with probability
       p_hide, from the first slow-ball frame after the stop to the restart kick;
    3. fill: the remaining gap lengths (the list minus the nearest length to every stoppage block) at
       random free places until hide_frac of the played frames is hidden;
    4. the < 1 s gaps at random.
    Block ends are snapped (q_exit) to the frame before the ball exceeds kick_v, block starts (q_entry) to
    a ball slower than entry_v. Lengths are GAP_FRAMES scaled by the match length."""
    p = dict(REAL2, **kw)
    rng = np.random.default_rng(3000 + seed)
    n = len(truth)
    inp = period > 0
    k = 5
    v = np.full(n, np.nan)
    v[k:] = np.hypot(bx[k:] - bx[:-k], by[k:] - by[:-k]) * fps / k      # speed over the last 0.2 s
    vf = np.full(n, np.nan)
    vf[:-k] = v[k:]                                                     # speed over the next 0.2 s
    with np.errstate(invalid="ignore"):
        slow = v < p["entry_v"]
        kick = (vf >= p["kick_v"]) & (v < 0.5 * p["kick_v"])
    isd = (truth == "D") & inp
    cd = np.concatenate([[0], np.cumsum(isd)])
    mask = np.zeros(n, bool)
    scale = inp.sum() / GAP_N
    lens = [max(1, int(round(g * scale))) for g in GAP_FRAMES]
    short = [g for g in lens if g < fps]
    tail = sorted([g for g in lens if g > p["tail_s"] * fps], reverse=True)
    pool = [g for g in lens if fps <= g <= p["tail_s"] * fps]
    target = p["hide_frac"] * inp.sum()

    def first(m, a, b):
        i = np.flatnonzero(m[max(a, 0):max(b, 0)])
        return max(a, 0) + int(i[0]) if len(i) else -1

    def snap_exit(e):
        if rng.random() < p["q_exit"]:
            j = first(kick, e, e + int(p["kick_max_s"] * fps))
            if j >= 0:
                return j
        return e

    def snap_entry(s, slack):
        if rng.random() < p["q_entry"]:
            lo, hi = s - slack, s + slack
            idx = np.flatnonzero(slow[max(lo, 1):max(hi, 1)]) + max(lo, 1)
            if len(idx):
                return int(idx[np.argmin(np.abs(idx - s))])
        return s

    def free(s, e):
        return (0 < s < e < n and period[s] == period[e] and period[s] > 0 and (period[s:e + 1] == period[s]).all()
                and not mask[max(0, s - int(p["margin_s"] * fps)):e + int(p["margin_s"] * fps)].any())

    def place(L, cand, best):
        """Best (most true D) or first valid of `cand` random windows of about L frames."""
        got, score = None, -1.0
        frames = np.flatnonzero(inp)
        for _ in range(cand):
            e = snap_exit(int(rng.choice(frames)))
            s = snap_entry(e - L, max(k, int(0.15 * L)))
            if not free(s, e):
                continue
            sc = (cd[e] - cd[s]) / (e - s)
            if not best:
                return s, e
            if sc > score:
                got, score = (s, e), sc
        return got

    for L in tail:
        w = place(L, p["tail_cand"], True)
        if w:
            mask[w[0]:w[1]] = True
    ds, de = _runs(isd)
    order = rng.permutation(len(ds))
    for i in order:
        a, b = int(ds[i]), int(de[i])
        ph = p["p_hide_long"] if b - a > p["stop_long_s"] * fps else p["p_hide"]
        if b - a < p["stop_min_s"] * fps or b >= n - 2 * k or mask[a:b].mean() >= 0.5 or rng.random() > ph:
            continue
        if mask.sum() >= target:
            break
        s = a
        if rng.random() < p["q_entry"]:
            j = first(slow, a, min(b, a + int(p["entry_max_s"] * fps)))
            s = j if j >= 0 else a
        e = snap_exit(b - k)                               # the restart kick, searched from just before the restart
        if e == b - k:
            e = b + int(rng.uniform(-1.0, 1.5) * fps)
        if e - s < fps or not free(s, e):
            continue
        mask[s:e] = True
        if pool:
            pool.pop(int(np.argmin(np.abs(np.asarray(pool) - (e - s)))))
    pool = [pool[i] for i in rng.permutation(len(pool))]
    for L in pool:
        if mask.sum() >= target:
            break
        w = place(L, p["fill_cand"], True) or place(L, 50, False)
        if w:
            mask[w[0]:w[1]] = True
    for L in short:
        for _ in range(50):
            s = int(rng.choice(np.flatnonzero(inp)))
            if 0 < s and s + L < n and not mask[s - 10:s + L + 10].any() and period[s] == period[s + L]:
                mask[s:s + L] = True
                break
    return mask


def smooth_ball(x, y, sigma):
    """Gaussian low-pass (sigma frames) of the visible ball inside each visible run; the run ends keep
    their position and trend (odd reflection), nan stays nan."""
    r = int(np.ceil(4 * sigma))
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    g /= g.sum()
    xs, ys = x.copy(), y.copy()
    for a, b in zip(*_runs(~np.isnan(x))):
        if b - a < 3:
            continue
        for src, dst in ((x, xs), (y, ys)):
            dst[a:b] = np.convolve(np.pad(src[a:b], r, mode="reflect", reflect_type="odd"), g, "valid")
    return xs, ys


def lose_players(pos, truth, period, hidden, bx, by, seed, fps=FPS, **kw):
    """Players missing as on the provided input: single-player dropouts far from the ball (more often
    while the ball is hidden), and a close-up camera in hidden stoppage blocks (only the players within
    close_r_m of the ball are kept for part of the block). bx, by: the full ball (before hiding)."""
    p = dict(REAL2, **kw)
    rng = np.random.default_rng(4000 + seed)
    out = pos.copy()
    n = len(pos)
    inp = np.flatnonzero(period > 0)
    d = np.hypot(pos[:, :, :, 0] - bx[:, None, None], pos[:, :, :, 1] - by[:, None, None])   # nan: no player / ball
    w = np.where(hidden[inp], p["flick_hid"], 1.0)
    n_ev = int(p["flick_per_s"] * (w.sum() / fps))
    for s in rng.choice(inp, n_ev, p=w / w.sum()):
        far = np.argwhere(np.nan_to_num(d[s], nan=-1.0) > p["flick_far_m"])
        if len(far):
            t, j = far[rng.integers(len(far))]
            out[s:s + 1 + int(rng.exponential(p["flick_mean_s"]) * fps), t, j] = np.nan
    isd = truth == "D"
    for a, b in zip(*_runs(hidden & (period > 0))):
        if b - a < p["close_min_s"] * fps or isd[a:b].mean() < 0.5 or rng.random() > p["p_close"]:
            continue
        L = int(rng.uniform(*p["close_part"]) * (b - a))
        s = a + int(rng.integers(0, b - a - L + 1))
        r = rng.uniform(*p["close_r_m"])
        seg = out[s:s + L]
        seg[~(d[s:s + L] <= r)] = np.nan                    # nan distance (no ball) drops the player too
    return out


def realistic_dead_ball(vx, vy, truth, period, seed, fps=FPS, **kw):
    """Ball during every true stoppage re-drawn: it rolls on with its last velocity (decaying), is
    carried towards the restart spot along a wandering path (never a straight constant-speed line,
    never frozen), and lies at the restart spot only for the last moments."""
    p = dict(DEAD_MODEL, **kw)
    rng = np.random.default_rng(1000 + seed)
    x, y = vx.copy(), vy.copy()
    n = len(x)
    d = truth == "D"
    i = 0
    while i < n:
        if not d[i]:
            i += 1
            continue
        j = i
        while j < n and d[j]:
            j += 1
        L = j - i
        if (i >= 3 and j < n and period[i - 1] == period[j] and period[j] > 0 and L >= 5
                and not np.isnan(vx[i - 3:i]).any() and not np.isnan(vx[j])):
            p0 = np.array([vx[i - 1], vy[i - 1]])
            v0 = (p0 - np.array([vx[i - 3], vy[i - 3]])) / 2.0            # m per frame
            sp = np.hypot(*v0) * fps
            if sp > 30.0:
                v0 *= 30.0 / sp
            p1 = np.array([vx[j], vy[j]])
            n_roll = int(min(p["max_frac"] * L, rng.uniform(*p["roll_s"]) * fps))
            n_place = int(min(p["max_frac"] * L, rng.uniform(*p["place_s"]) * fps))
            n_mid = L - n_roll - n_place
            tau = p["roll_tau_s"] * fps
            t = np.arange(1, n_roll + 1)
            roll = p0[None, :] + v0[None, :] * (tau * (1 - np.exp(-t / tau)))[:, None]
            pr = roll[-1] if n_roll else p0
            s = (np.arange(1, n_mid + 1) / (n_mid + 1))
            ease = s * s * (3 - 2 * s)
            mid = pr[None, :] + (p1 - pr)[None, :] * ease[:, None]
            if n_mid > 10:                                               # smooth wander, zero at both ends
                w = max(3, int(p["wander_corr_s"] * fps))
                k = np.hanning(w); k /= k.sum()
                for c in (0, 1):
                    g = np.convolve(rng.normal(0, 1, n_mid + 2 * w), k, "same")[w:-w]
                    g = g / (g.std() + 1e-9) * rng.uniform(0.3, 1.0) * p["wander_m"]
                    mid[:, c] += g * np.sin(np.pi * s)
            seg = np.vstack([roll.reshape(-1, 2), mid.reshape(-1, 2), np.repeat(p1[None, :], n_place, axis=0)])
            x[i:j], y[i:j] = seg[:, 0], seg[:, 1]
        i = j
    return x, y


def tracking_faults(vx, vy, period, seed, fps=FPS, **kw):
    """Challenge-like faults on the ball: sample-and-hold blocks, long freezes, jumps, white noise."""
    p = dict(FAULTS, **kw)
    rng = np.random.default_rng(2000 + seed)
    x, y = vx.copy(), vy.copy()
    n = len(x)
    play = np.flatnonzero((period > 0) & ~np.isnan(x))
    mins = len(play) / fps / 60.0
    if p["noise_m"]:
        x = x + rng.normal(0, p["noise_m"], n)
        y = y + rng.normal(0, p["noise_m"], n)
    if p["jump_per_min"]:
        for s in rng.choice(play, int(p["jump_per_min"] * mins), replace=False):
            L = int(rng.integers(p["jump_frames"][0], p["jump_frames"][1] + 1))
            r, th = rng.uniform(*p["jump_m"]), rng.uniform(0, 2 * np.pi)
            x[s:s + L] += r * np.cos(th)
            y[s:s + L] += r * np.sin(th)
    if p["hold_share"]:
        held, k = 0, int(p["hold_k"])
        while held < p["hold_share"] * len(play):
            L = int(rng.uniform(*p["hold_block_s"]) * fps)
            s = int(rng.choice(play))
            for a in range(s, min(n, s + L), k):
                if not np.isnan(x[a]):
                    x[a:a + k] = np.where(np.isnan(x[a:a + k]), np.nan, x[a])
                    y[a:a + k] = np.where(np.isnan(y[a:a + k]), np.nan, y[a])
            held += L
    if p["freeze_per_min"]:
        for s in rng.choice(play, int(p["freeze_per_min"] * mins), replace=False):
            L = int(rng.uniform(*p["freeze_s"]) * fps)
            x[s:s + L] = np.where(np.isnan(x[s:s + L]), np.nan, x[s])
            y[s:s + L] = np.where(np.isnan(y[s:s + L]), np.nan, y[s])
    return x, y


def get_case(name, seed=0, variant=""):
    """Truth + the ball the pipeline will see (dead balls visible, then hidden blocks).

    variant "" = the protocol bench (cache *_v2.npz, unchanged). Other names: VARIANTS (L2-12 audit),
    cached under their own name."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "%s_s%d_%s.npz" % (name, seed, variant or "v2"))
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    truth, present, period, vx, vy = base_ball(name)
    if variant and VARIANTS[variant].get("real2"):
        case, pos = real2_case(name, truth, present, period, vx, vy, seed)
        np.savez_compressed(os.path.join(CACHE, "shapepos_%s_s%d_%s.npz" % (name, seed, variant)), pos=pos)
        np.savez_compressed(path, **case)
        return case
    if variant:
        v = VARIANTS[variant]
        if v["dead"]:
            vx, vy = realistic_dead_ball(vx, vy, truth, period, seed)
        if v["faults"]:
            vx, vy = tracking_faults(vx, vy, period, seed)
    hidden = hide_mask(len(truth), seed)
    seen_x, seen_y = vx.copy(), vy.copy()
    seen_x[hidden] = seen_y[hidden] = np.nan
    case = dict(truth=truth, present=present, period=period, hidden=hidden, ball_x=seen_x, ball_y=seen_y)
    np.savez_compressed(path, **case)
    return case


def real2_case(name, truth, present, period, vx, vy, seed=0, pos=None, **kw):
    """(case, player positions) of the `real2` variant (envelope L2-20): dead-ball model as real1, then
    state-coupled hiding, smoothing inside the visible runs, real1's jumps on the smoothed ball, and
    player loss. kw overrides REAL2 (scripts/audit_bench.py inputs scans them; the cache uses REAL2)."""
    p = dict(REAL2, **kw)
    vx, vy = realistic_dead_ball(vx, vy, truth, period, seed)
    hidden = hide_mask_coupled(truth, period, vx, vy, seed, **p)
    seen_x, seen_y = vx.copy(), vy.copy()
    seen_x[hidden] = seen_y[hidden] = np.nan
    seen_x, seen_y = smooth_ball(seen_x, seen_y, p["smooth_sigma"])
    seen_x, seen_y = tracking_faults(seen_x, seen_y, period, seed, noise_m=p["noise_m"])
    pos = lose_players(bench_positions(name, len(truth)) if pos is None else pos, truth, period, hidden, vx, vy,
                       seed, **p)
    return dict(truth=truth, present=present, period=period, hidden=hidden, ball_x=seen_x, ball_y=seen_y), pos


def base_ball(name):
    """(truth, present, period, ball x, ball y) of an external match before any degradation."""
    if name.startswith("idsse"):
        # IDSSE: truth from BallStatus/BallPossession; its own ball is tracked through stoppages,
        # so show_dead_ball is not applied (scripts/idsse_cache.py).
        from idsse_cache import load as idsse_load

        d = idsse_load(name)
        truth, present, period = d["truth"], d["present"], d["period"]
        vx, vy = d["ball_x"].astype(float), d["ball_y"].astype(float)
    else:
        match = load_match(name)
        truth, present, period, bx, by = build_truth(name, match)
        vx, vy = show_dead_ball(bx, by, period)
    return truth, present, period, vx, vy


def case_features(name, case, seed=0, variant=""):
    """possess.features.compute on the bench ball (dead balls visible, blocks hidden). Cached."""
    from possess import features as feat_mod

    path = os.path.join(CACHE, "feat_%s_s%d_%s.npz" % (name, seed, variant or "v1"))
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    f = feat_mod.compute(load_match(name).iter_frames(), n=len(case["truth"]),
                         ball_x=case["ball_x"], ball_y=case["ball_y"])
    np.savez_compressed(path, **f)
    return f


def pipeline_possess(name, n, case, seed=0, dead_kw=None, team_kw=None):
    """The submission pipeline (possess.pipeline.label) on a bench case."""
    from possess.pipeline import label

    return label(case_features(name, case, seed), dead_kw=dead_kw, team_kw=team_kw)


PIPELINES = {
    # Self-contained pipeline (no outside repo). Replaces the loop-3 "dead2_team" pipeline, which read
    # MatchLens's dead-ball state from a cached run of the external MatchLens Eventer.
    "possess": pipeline_possess,
}


# ----------------------------------------------------------------------------- explain
def spells(codes, runs):
    return pm.possession_spells(codes, runs)


def explain(gt, pred_filled, runs, hidden):
    """Cause of every unmatched true possession and every spurious predicted one (IoU 0.5)."""
    gs, ps = spells(gt, runs), spells(pred_filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    g_ok = pm.matched_gt_flags(ov, len(gs), 0.5)
    matched_p = set()
    for iou, gi, pj in ov:
        if iou < 0.5:
            break
        if g_ok[gi] and gi not in [x for x in ()]:
            pass
    # recompute one-to-one pairs to know which predictions matched
    used_g = set()
    for iou, gi, pj in ov:
        if iou < 0.5:
            break
        if gi in used_g or pj in matched_p:
            continue
        used_g.add(gi); matched_p.add(pj)
    D, H, A = pm.D, pm.H, pm.A
    miss, spur = Counter(), Counter()
    for gi, (a, b, code) in enumerate(gs):
        if g_ok[gi]:
            continue
        seg = pred_filled[a:b]
        other = A if code == H else H
        f_same, f_d, f_oth = (seg == code).mean(), (seg == D).mean(), (seg == other).mean()
        inside = [p for p in ps if p[2] == code and p[0] < b and p[1] > a]
        gap = hidden[a:b].mean() > 0.5
        if f_oth >= 0.5:
            cause = "wrong team for the whole possession"
        elif f_d >= 0.5:
            cause = "predicted dead ball during a live possession"
        elif len(inside) >= 2:
            # what separates the pieces?
            cuts = seg[(seg != code)]
            cause = ("split by an invented stoppage" if (cuts == D).sum() >= (cuts == other).sum()
                     else "split by a spurious turnover")
        elif inside and (inside[0][1] - inside[0][0]) > 1.5 * (b - a):
            cause = "merged with a neighbour (missed stoppage or turnover)"
        else:
            cause = "boundary too late/early"
        miss[cause + (" [ball hidden]" if gap else "")] += 1
    for pj, (a, b, code) in enumerate(ps):
        if pj in matched_p:
            continue
        seg = gt[a:b]
        other = A if code == H else H
        if (seg == D).mean() >= 0.5:
            cause = "possession invented during a true dead ball"
        elif (seg == other).mean() >= 0.5:
            cause = "spell for the wrong team"
        elif (seg == code).mean() >= 0.5:
            cause = "fragment of a true possession (over-splitting)"
        else:
            cause = "mixed / straddles a boundary"
        spur[cause + (" [ball hidden]" if hidden[a:b].mean() > 0.5 else "")] += 1
    return miss, spur


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="possess")
    ap.add_argument("--split", choices=list(SPLITS), default="tune")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--tag", default=None, help="name for the results file")
    a = ap.parse_args()
    fn = PIPELINES[a.pipeline]
    out = {"pipeline": a.pipeline, "split": a.split, "seed": a.seed, "matches": {}}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                                text=True).strip()
    except Exception:
        out["commit"] = None
    tot_miss, tot_spur = Counter(), Counter()
    for name in SPLITS[a.split]:
        t0 = time.time()
        case = get_case(name, a.seed)
        truth, hidden = case["truth"], case["hidden"]
        pred = fn(name, len(truth), case)
        np.save(os.path.join(CACHE, "pred_%s_%s_s%d.npy" % (a.pipeline, name, a.seed)), pred)
        gt_c, pr_c = pm.label_codes(truth), pm.label_codes(pred)
        r = pm.evaluate(gt_c, pr_c, bootstrap_n=a.bootstrap)
        runs = pm.valid_runs(gt_c)
        miss, spur = explain(gt_c, pm.fill_abstentions(pr_c, runs), runs, hidden)
        tot_miss.update(miss); tot_spur.update(spur)
        live = np.isin(truth, ["H", "A"])
        out["matches"][name] = {
            "spellF1": r["spell_f1_mean"], "ci": [r["spell_bootstrap_low"], r["spell_bootstrap_high"]],
            "macroF1": r["macro_f1"], "alive_dead_f1": r["alive_dead_f1"],
            "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
            "gt_mean_s": r["gt_mean_spell_duration_s"], "pred_mean_s": r["mean_spell_duration_s"],
            "hidden_frac_all": float(hidden.mean()), "hidden_frac_live": float(hidden[live].mean()),
            "missed": dict(miss), "spurious": dict(spur), "secs": round(time.time() - t0),
        }
        m = out["matches"][name]
        ci = "[%.3f, %.3f]" % tuple(m["ci"]) if m["ci"][0] is not None else "[no CI]"
        print("%-9s spellF1 %.4f %s  macroF1 %.4f  aliveDeadF1 %.3f  possessions %d -> %d  "
              "mean %.1fs -> %.1fs  ball hidden %.0f%% of live (%ds)"
              % (name, m["spellF1"], ci, m["macroF1"], m["alive_dead_f1"],
                 m["n_gt"], m["n_pred"], m["gt_mean_s"], m["pred_mean_s"],
                 100 * m["hidden_frac_live"], m["secs"]))
    out["mean_spellF1"] = float(np.mean([m["spellF1"] for m in out["matches"].values()]))
    out["missed_total"], out["spurious_total"] = dict(tot_miss), dict(tot_spur)
    print("MEAN spellF1 %.4f" % out["mean_spellF1"])
    print("\nWhy true possessions were missed (IoU 0.5):")
    for k, v in tot_miss.most_common():
        print("  %5d  %s" % (v, k))
    print("Why predicted possessions were spurious:")
    for k, v in tot_spur.most_common():
        print("  %5d  %s" % (v, k))
    tag = a.tag or "%s_%s_s%d" % (a.pipeline, a.split, a.seed)
    path = os.path.join(ROOT, "eval", "results", "bench_%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
