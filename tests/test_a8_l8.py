"""Envelope L8: opt-in flags rule_flip, ctl_hi / ctl_touch (decode.py). Synthetic data only."""
import numpy as np

from possess import decode


def _scene(n=400):
    """Ball at home feet, then a pass with two away touches, then home again; no gaps."""
    rng = np.random.default_rng(0)
    bx = np.cumsum(np.where(np.arange(n) % 60 < 30, 0.3, -0.3)) + rng.normal(0, 0.001, n)
    by = np.zeros(n)
    dh = np.where((np.arange(n) // 60) % 2 == 0, 0.3, 5.0).astype(float)
    da = np.where((np.arange(n) // 60) % 2 == 0, 5.0, 0.3).astype(float)
    return bx, by, dh, da


def test_touch_events_one_per_touch():
    bx, by, dh, da = _scene()
    ev = decode.touch_events(bx, by, dh, da)
    raw = decode.touch_hold(bx, by, dh, da, 1.0, 0.0)
    assert set(np.unique(ev)) <= {-1.0, 0.0, 1.0}
    assert 0 < (ev != 0).sum() <= (raw != 0).sum()
    idx = np.flatnonzero(ev != 0)
    same = ev[idx[1:]] == ev[idx[:-1]]
    assert (np.diff(idx)[same] > 5).all()              # same-team events are distinct touches


def test_flags_off_identical_and_on_runs():
    bx, by, dh, da = _scene()
    n = len(bx)
    feats = {"d_home": dh, "d_away": da, "visible": np.ones(n, bool), "dead": np.zeros(n, bool), "ball_x": bx, "ball_y": by}
    period = np.ones(n, int)
    kw = dict(hard_dead=1, soft=1, cap=1.0, d0=0.25, c_switch=4.0, c_same=1.0, w_rule=0.3, w_touch=0.7, ctl_s=1.0)
    base = decode.decode_halves(period, feats=dict(feats), **kw)
    assert (decode.decode_halves(period, feats=dict(feats), rule_flip=-1, ctl_hi=0, **kw) == base).all()
    hi = decode.decode_halves(period, feats=dict(feats), ctl_hi=3.0, ctl_touch=99, **kw)
    nb = (base[1:] != base[:-1]).sum()
    assert (hi[1:] != hi[:-1]).sum() <= nb            # the pass can only merge spells
    nf = decode.decode_halves(period, feats=dict(feats), rule_flip=0, **kw)
    assert set(np.unique(nf)) <= {"H", "A", "D"}
