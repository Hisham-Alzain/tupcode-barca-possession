"""Envelope L8 round 6: learned ball-aware stoppage evidence on visible frames (balldead.py, w_bd; opt-in)."""
import numpy as np

from possess import balldead, decode
from possess.shape import DEC_SHAPE


def _case(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.full((n, 2, 11, 2), np.nan)
    pos[:, 0, :, 0] = -20 + rng.normal(0, 5, (n, 11))
    pos[:, 1, :, 0] = 20 + rng.normal(0, 5, (n, 11))
    pos[:, :, :, 1] = rng.normal(0, 10, (n, 2, 11))
    period = np.ones(n, dtype=np.int8)
    bx = np.concatenate([np.zeros(100), np.linspace(0, 40, 100), np.full(100, 40.0), np.linspace(40, 60, 100)])   # ends beyond the goal line
    by = np.zeros(n)
    dh = np.concatenate([np.full(100, 0.5), np.linspace(0.5, 20, 100), np.full(200, 20.0)])
    da = np.concatenate([np.full(200, 20.0), np.full(100, 0.4), np.linspace(0.4, 20, 100)])
    vis = np.ones(n, bool)
    vis[150:160] = False
    return bx, by, dh, da, vis, pos, period


def test_features_do_not_know_the_teams():
    bx, by, dh, da, vis, pos, period = _case()
    sd, p90 = np.linspace(-2, 2, len(bx)), np.full(len(bx), 3.0)
    X = balldead.features(bx, by, dh, da, vis, pos, sd, p90)
    assert X.shape == (len(bx), len(balldead.FEATS)) and np.isfinite(X).all()
    assert np.allclose(X, balldead.features(bx, by, da, dh, vis, pos[:, ::-1], sd, p90))     # teams swapped: the same
    i = balldead.FEATS.index
    assert (X[220:290, i("slow")] == 1).all() and (X[110:190, i("slow")] == 0).all()
    assert (X[380:, i("outside")] == 1).all() and (X[:300, i("outside")] == 0).all()
    assert np.allclose(X[:, i("shape_dead")], sd) and np.allclose(X[:, i("p90")], 3.0 / 8.0)
    assert X[250, i("still_time")] > X[205, i("still_time")] > 0


def test_evidence_range_and_hidden_frames():
    bx, by, dh, da, vis, pos, period = _case()
    X = balldead.features(bx, by, dh, da, vis, pos)
    m = {"mu": np.zeros(X.shape[1]), "sd": np.ones(X.shape[1]), "w": np.zeros(X.shape[1]), "b": 0.0, "base": 0.0}
    m["w"][balldead.FEATS.index("outside")] = 4.0
    z = balldead.evidence(X, vis, m)
    assert (np.abs(z) <= 1).all() and (z[150:160] == 0).all() and (z[385:] > 0.9).all() and (z[:150] == 0).all()
    z2 = balldead.evidence(X, vis, dict(m, base=1.0))
    assert (z2[:150] < 0).all() and (z2[150:160] == 0).all()                                  # below the base rate = live
    assert balldead.MODEL and all(len(balldead.MODEL[k]) == len(balldead.FEATS) for k in ("mu", "sd", "w"))
    zm = balldead.from_feats(dict(ball_x=bx, ball_y=by, d_home=dh, d_away=da, visible=vis, pos=pos, shape_dead=np.zeros(len(bx))))
    assert np.isfinite(zm).all() and (np.abs(zm) <= 1).all() and (zm[150:160] == 0).all()


def _feats(n=400):
    bx, by, dh, da, vis, pos, period = _case(n)
    dh = np.concatenate([np.full(200, 0.5), np.full(200, 20.0)])                             # home holds the ball, then nobody
    da = np.full(n, 20.0)
    feats = dict(d_home=dh, d_away=da, visible=np.ones(n, bool), dead=np.zeros(n, bool), ball_x=np.zeros(n), ball_y=np.zeros(n),
                 pos=pos, shape_team=np.zeros(n), shape_dead=np.full(n, -1.0))
    return feats, period


def test_w_bd_off_is_identical_and_on_adds_a_stoppage():
    feats, period = _feats()
    n = len(period)
    kw = dict(DEC_SHAPE, w_bd=0.0, ed_still=0.0, hard_dead=1, soft=1, cap=1.0, d0=0.25, c_switch=4.0, c_same=1.0, w_rule=0.3, w_touch=0.7)
    base = decode.decode_halves(period, feats=feats, **kw)
    assert np.array_equal(base, decode.decode_halves(period, feats=feats, **dict(kw, w_bd=0.0)))
    assert np.array_equal(base, decode.decode_halves(period, feats=dict(feats, bd=np.zeros(n)), **dict(kw, w_bd=1.0)))
    assert np.array_equal(base, decode.decode_halves(period, feats=dict(feats, bd=np.ones(n)), **dict(kw, w_bd=0.0)))   # bd without the flag: ignored
    assert (base[:150] == "H").all()
    bd = np.zeros(n)
    bd[60:140] = 1.0                                                                          # the model says: dead here
    lab = decode.decode_halves(period, feats=dict(feats, bd=bd), **dict(kw, w_bd=40.0))
    assert (lab[80:120] == "D").all() and (lab[:40] == "H").all()
    hid = dict(feats, visible=np.zeros(n, bool))                                              # hidden frames carry no bd evidence in practice;
    assert np.array_equal(decode.decode_halves(period, feats=hid, **kw),                      # the flag alone (zero evidence) changes nothing
                          decode.decode_halves(period, feats=dict(hid, bd=np.zeros(n)), **dict(kw, w_bd=1.0)))
