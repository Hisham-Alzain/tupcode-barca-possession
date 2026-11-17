"""Envelope L8 round 2: flight-direction evidence (decode.flight_dir, w_fly; opt-in)."""
import numpy as np

from possess import decode


def _case(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.full((n, 2, 11, 2), np.nan)
    pos[:, 0, :, 0] = -20 + rng.normal(0, 5, (n, 11))    # home stands on the -x side -> attacks +x
    pos[:, 1, :, 0] = 20 + rng.normal(0, 5, (n, 11))
    pos[:, :, :, 1] = rng.normal(0, 10, (n, 2, 11))
    period = np.ones(n, dtype=np.int8)
    bx = np.concatenate([np.zeros(100), np.linspace(0, 40, 100), np.full(100, 40.0), np.linspace(40, 0, 100)])   # +10 m/s, then -10 m/s
    by = np.zeros(n)
    return bx, by, pos, period


def test_flight_dir_sign_and_ramp():
    bx, by, pos, period = _case()
    f = decode.flight_dir(bx, by, pos, period, 4.0, 8.0)
    assert (f[110:190] == 1.0).all()                     # ball flies +x at 10 m/s: home attacks +x
    assert (f[310:390] == -1.0).all()
    assert (f[10:90] == 0.0).all() and (f[210:290] == 0.0).all()
    half = decode.flight_dir(bx, by, pos, period, 8.0, 12.0)
    assert np.allclose(half[110:190], 0.5, atol=0.03)               # ramp: 10 m/s between 8 and 12


def test_flight_dir_swaps_with_the_teams_and_ignores_hidden_and_teleports():
    bx, by, pos, period = _case()
    f = decode.flight_dir(bx, by, pos, period, 4.0, 8.0)
    assert np.array_equal(decode.flight_dir(bx, by, pos[:, ::-1], period, 4.0, 8.0), -f)
    hid = bx.copy()
    hid[140:150] = np.nan
    g = decode.flight_dir(hid, by, pos, period, 4.0, 8.0)
    assert (g[140:150] == 0.0).all() and np.isfinite(g).all()
    tp = bx.copy()
    tp[50:] += 60.0                                      # teleport: 60 m in one frame
    assert (decode.flight_dir(tp, by, pos, period, 4.0, 8.0)[46:54] == 0.0).all()
    assert (decode.flight_dir(bx, by, pos, np.zeros(len(bx), dtype=np.int8), 4.0, 8.0) == 0.0).all()


def test_w_fly_off_is_identical_and_on_moves_the_team():
    bx, by, pos, period = _case()
    n = len(bx)
    feats = dict(d_home=np.full(n, 30.0), d_away=np.full(n, 30.0), visible=np.ones(n, bool), dead=np.zeros(n, bool),
                 ball_x=bx, ball_y=by, pos=pos)
    kw = dict(hard_dead=1, soft=1, cap=1.0, c_switch=4.0, w_touch=0.7)
    base = decode.decode_halves(period, feats=dict(feats), **kw)
    assert np.array_equal(base, decode.decode_halves(period, feats=dict(feats), w_fly=0.0, fly_v0=4.0, fly_v1=8.0, **kw))
    on = decode.decode_halves(period, feats=dict(feats), w_fly=0.7, fly_v0=4.0, fly_v1=8.0, **kw)
    assert (on[120:180] == "H").all() and (on[320:380] == "A").all()
