"""Envelope L3-A4: the touch is attributed at the contact point rebuilt from a low-passed ball track."""
import os
import sys

import numpy as np

from possess.decode import touch_vertex, vertex_team_dist, viterbi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import bench  # noqa: E402


def _deflection(n=200, k=100, fps=25.0):
    """Ball at 15 m/s along +x, deflected at frame k (at (15*k/fps, 0)) into +y at 12 m/s."""
    t = np.arange(n) / fps
    tk = k / fps
    x = np.where(t <= tk, 15.0 * t, 15.0 * tk)
    y = np.where(t <= tk, 0.0, 12.0 * (t - tk))
    return x, y, (15.0 * tk, 0.0)


def test_vertex_recovers_the_contact_point_of_a_smoothed_deflection():
    x, y, (cx, cy) = _deflection()
    xs, ys = bench.smooth_ball(x, y, 7.0)
    vx, vy, kc = touch_vertex(xs, ys, 8)
    assert np.hypot(xs[100] - cx, ys[100] - cy) > 2.0       # the smoothed ball sits 2.1 m off the toucher
    assert np.hypot(vx[100] - cx, vy[100] - cy) < 0.8       # the vertex 0.65 m
    assert abs(kc[100] - 100) <= 2
    assert np.isnan(vx[:16]).all()                          # needs 2 lag frames on each side


def test_vertex_team_dist_and_missing():
    n = 5
    pos = np.full((n, 2, 3, 2), np.nan, dtype=np.float32)
    pos[:, 0, 0] = [1.0, 0.0]
    pos[:, 1, 0] = [0.0, 3.0]
    vx, vy = np.zeros(n), np.zeros(n)
    vx[2] = np.nan
    dh, da = vertex_team_dist(pos, vx, vy, np.arange(n))
    assert np.allclose(dh[[0, 1, 3, 4]], 1.0) and np.allclose(da[[0, 1, 3, 4]], 3.0)
    assert dh[2] == 99.0 and da[2] == 99.0


def test_vtx_off_or_unsmoothed_is_byte_identical():
    rng = np.random.default_rng(1)
    n = 600
    x, y = np.cumsum(rng.normal(0, 0.3, n)), np.cumsum(rng.normal(0, 0.3, n))
    dh, da = rng.uniform(0, 3, n), rng.uniform(0, 3, n)
    pos = rng.uniform(-5, 5, (n, 2, 3, 2)).astype(np.float32)
    kw = dict(d_home=dh, d_away=da, visible=np.ones(n, bool), dead=np.zeros(n, bool), soft=1, cap=1.0, w_touch=0.7,
              ball_x=x, ball_y=y, duel=2, smooth_touch=1)
    base = viterbi(**kw)
    assert (viterbi(**kw, pos=pos, vtx=0.0) == base).all()
    assert (viterbi(**kw, pos=pos, vtx=2.0) == base).all()   # a sharp track: vtx applies to low-passed tracks only
