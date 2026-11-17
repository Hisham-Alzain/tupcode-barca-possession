"""Envelope L8 round 5: learned ball-aware team evidence (ballteam.py, w_bt; opt-in)."""
import numpy as np

from possess import ballteam, decode


def _case(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.full((n, 2, 11, 2), np.nan)
    pos[:, 0, :, 0] = -20 + rng.normal(0, 5, (n, 11))    # home stands on the -x side -> attacks +x
    pos[:, 1, :, 0] = 20 + rng.normal(0, 5, (n, 11))
    pos[:, :, :, 1] = rng.normal(0, 10, (n, 2, 11))
    period = np.ones(n, dtype=np.int8)
    bx = np.concatenate([np.zeros(100), np.linspace(0, 40, 100), np.full(100, 40.0), np.linspace(40, 0, 100)])
    by = np.zeros(n)
    dh = np.concatenate([np.full(100, 0.5), np.linspace(0.5, 20, 100), np.full(200, 20.0)])    # home at the ball, then it leaves
    da = np.concatenate([np.full(200, 20.0), np.full(100, 0.4), np.linspace(0.4, 20, 100)])
    return bx, by, dh, da, pos, period


def test_helpers():
    x = np.array([5.0, 3.0, 4.0, 1.0, 6.0, 7.0])
    assert np.array_equal(ballteam.past_min(x, 3), [5, 3, 3, 1, 1, 1])
    assert np.array_equal(ballteam.past_min(x, 1), x)
    assert np.array_equal(ballteam.since(np.array([0, 1, 0, 0, 1, 0], bool))[1:], [0, 1, 2, 0, 1])
    bx = np.concatenate([np.zeros(50), np.linspace(0, 20, 50), np.full(50, 20.0)])           # starts at 10 m/s, stops
    imp = ballteam.impulses(bx, np.zeros(150), 4.0)
    assert len(imp) == 2 and abs(imp[0] - 50) <= 6 and abs(imp[1] - 100) <= 6


def test_features_flip_with_the_teams():
    bx, by, dh, da, pos, period = _case()
    st = np.linspace(-1, 1, len(bx))
    X = ballteam.features(bx, by, dh, da, pos, period, st)
    assert X.shape == (len(bx), len(ballteam.FEATS)) and np.isfinite(X).all()
    Y = ballteam.features(bx, by, da, dh, pos[:, ::-1], period, -st)
    assert np.allclose(X, -Y)
    i = ballteam.FEATS.index
    assert (X[10:90, i("g_now")] > 0).all() and (X[210:290, i("g_now")] < 0).all()
    assert (X[120:140, i("past50")] > 0).all()           # home was at the ball within the last 2 s
    assert (X[130:190, i("fly48")] == 1.0).all()


def test_evidence_range_and_hidden_frames():
    bx, by, dh, da, pos, period = _case()
    X = ballteam.features(bx, by, dh, da, pos, period)
    m = {"mu": np.zeros(X.shape[1]), "sd": np.ones(X.shape[1]), "w": np.ones(X.shape[1]), "b": 0.0}
    vis = np.ones(len(bx), bool)
    vis[50:60] = False
    z = ballteam.evidence(X, vis, m)
    assert (np.abs(z) <= 1).all() and (z[50:60] == 0).all() and (z[10:40] > 0).all() and (z[220:280] < 0).all()
    assert np.allclose(ballteam.evidence(-X, vis, m), -z)


def test_w_bt_off_is_identical_on_names_the_team_and_keeps_the_stoppage():
    bx, by, dh, da, pos, period = _case()
    n = len(bx)
    dead = np.zeros(n, bool)
    dead[300:] = True
    feats = dict(d_home=np.full(n, 30.0), d_away=np.full(n, 30.0), visible=np.ones(n, bool), dead=dead,
                 ball_x=np.zeros(n), ball_y=by, pos=pos)
    bt = np.concatenate([np.full(150, 1.0), np.full(250, -1.0)])
    kw = dict(hard_dead=1, soft=1, cap=1.0, c_switch=4.0, w_touch=0.7)
    base = decode.decode_halves(period, feats=dict(feats), **kw)
    assert np.array_equal(base, decode.decode_halves(period, feats=dict(feats, bt=bt), **kw))            # weight 0 = off
    assert np.array_equal(base, decode.decode_halves(period, feats=dict(feats), w_bt=0.0, **kw))
    on = decode.decode_halves(period, feats=dict(feats, bt=bt), w_bt=0.5, **kw)
    assert (on[20:140] == "H").all() and (on[170:290] == "A").all()
    assert (on[300:] == "D").all()                       # D-neutral: the evidence does not lift a team against a stoppage
