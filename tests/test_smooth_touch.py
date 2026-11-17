"""Envelope L2-20: touch detection on a low-passed ball track, and the bench variant real2."""
import os
import sys

import numpy as np

from possess import pipeline
from possess.decode import SMOOTH_TOUCH, ball_smoothed, touch_hold, viterbi
from possess.shape import DEC_SHAPE_L2_16, DEC_SHAPE_L2_18

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import bench  # noqa: E402


def _zigzag(n=2000, step=40, v=6.0, fps=25.0):
    """Ball at 6 m/s turning 90 degrees every 40 frames; a home player stands on every corner."""
    x, y = np.zeros(n), np.zeros(n)
    corners = []
    for i in range(1, n):
        k = (i - 1) // step
        d = (1.0, 0.0) if k % 2 == 0 else (0.0, 1.0)
        x[i], y[i] = x[i - 1] + d[0] * v / fps, y[i - 1] + d[1] * v / fps
        if i % step == 0:
            corners.append((x[i], y[i]))
    c = np.array(corners)
    return x, y, c


def _dist(x, y, pts):
    return np.hypot(x[:, None] - pts[None, :, 0], y[:, None] - pts[None, :, 1]).min(axis=1)


def test_ball_smoothed_reads_the_track():
    x, y, _ = _zigzag()
    assert not ball_smoothed(x, y)                          # sharp corners: 212 m/s2 on 2.5% of frames
    xs, ys = bench.smooth_ball(x, y, 7.0)
    assert ball_smoothed(xs, ys)
    assert not ball_smoothed(xs[:100], ys[:100])            # too few frames to tell
    rng = np.random.default_rng(0)
    assert not ball_smoothed(xs + rng.normal(0, 0.05, len(xs)), ys + rng.normal(0, 0.05, len(xs)))


def test_wide_touch_detector_finds_touches_on_a_smoothed_ball():
    x, y, corners = _zigzag()
    away = np.array([[500.0, 500.0]])
    xs, ys = bench.smooth_ball(x, y, 7.0)
    dh, da = _dist(xs, ys, corners), _dist(xs, ys, away)
    old = touch_hold(xs, ys, dh, da, duel=2)
    new = touch_hold(xs, ys, dh, da, duel=2, **SMOOTH_TOUCH)
    assert (old[100:] != 0).mean() < 0.1                    # the 2-frame impulse test is blind here
    assert (new[100:-50] == 1).mean() > 0.9                 # every corner is a home touch
    assert (touch_hold(xs, ys, da, dh, duel=2, **SMOOTH_TOUCH) == -new).all()
    # peak=1 keeps one frame per touch: fewer touch frames than the plain wide test, same holder
    raw = touch_hold(x, y, _dist(x, y, corners), _dist(x, y, away))
    assert (raw[100:] == 1).mean() > 0.9                    # unsmoothed track: the default detector works


def test_smooth_touch_is_inert_on_an_unsmoothed_ball_and_default_on():
    x, y, corners = _zigzag(800)
    away = np.array([[3.0, 40.0], [60.0, 2.0]])
    dh, da = _dist(x, y, corners), _dist(x, y, away)
    vis, dead = np.ones(len(x), bool), np.zeros(len(x), bool)
    kw = dict(pipeline.DEC_BASE, ball_x=x, ball_y=y, duel=2)
    a = viterbi(dh, da, vis, dead, smooth_touch=0, **kw)
    b = viterbi(dh, da, vis, dead, smooth_touch=1, **kw)
    assert (a == b).all()
    assert pipeline.DEC["smooth_touch"] == 1 and "smooth_touch" not in pipeline.DEC_BASE
    assert DEC_SHAPE_L2_16["smooth_touch"] == DEC_SHAPE_L2_18["smooth_touch"] == 0


def test_smooth_ball_and_player_loss_keep_what_is_missing():
    x, y, _ = _zigzag(600)
    x[200:260] = y[200:260] = np.nan
    xs, ys = bench.smooth_ball(x, y, 7.0)
    assert (np.isnan(xs) == np.isnan(x)).all()
    assert abs(xs[0] - x[0]) < 1e-9 and abs(xs[199] - x[199]) < 1e-9 and abs(ys[260] - y[260]) < 1e-9   # run ends kept
    rng = np.random.default_rng(0)
    pos = rng.uniform(-30, 30, (600, 2, 14, 2)).astype(np.float32)
    pos[:, :, 11:] = np.nan
    truth = np.array(["H"] * 600, dtype="<U1")
    truth[200:560] = "D"
    hidden = np.zeros(600, bool)
    hidden[200:560] = True
    out = bench.lose_players(pos, truth, np.ones(600, np.int8), hidden, np.nan_to_num(x), np.nan_to_num(y), 0, p_close=1.0)
    assert (np.isnan(out) | (out == pos)).all()                         # players are only ever removed
    assert np.isfinite(out[:, :, :, 0]).sum(axis=(1, 2))[hidden].min() < 16   # the close-up
    assert bench.VARIANTS["real1"] == dict(dead=True, faults=True)


def test_coupled_hiding_follows_the_stoppages():
    n, fps = 60000, 25
    rng = np.random.default_rng(0)
    truth = np.array(["H"] * n, dtype="<U1")
    for s in range(1000, n - 2000, 1500):                               # a 20 s stoppage every minute
        truth[s:s + 500] = "D"
    period = np.ones(n, np.int8)
    t = np.arange(n) / fps
    bx, by = 30 * np.sin(t / 7.0), 20 * np.cos(t / 5.0)
    bx[truth == "D"], by[truth == "D"] = 10.0, 10.0                    # the dead ball lies still
    m = bench.hide_mask_coupled(truth, period, bx, by, 0)
    assert 0.3 < m.mean() < 0.5
    assert (truth[m] == "D").mean() > (truth[~m] == "D").mean() + 0.2
    assert (m == bench.hide_mask_coupled(truth, period, bx, by, 0)).all()   # seeded
