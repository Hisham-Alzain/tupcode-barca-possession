"""edges.still_dead (envelope L8 round 4, opt-in ed_still): a ball at rest and alone for 3 s is dead."""
import numpy as np

from possess import edges


def _case(n=400):
    period = np.ones(n, dtype=int)
    bx = np.cumsum(np.full(n, 0.4))            # 10 m/s
    by = np.zeros(n)
    bx[100:200] = bx[100]                      # 4 s at rest
    bx[200:] = bx[100] + np.cumsum(np.full(n - 200, 0.4))
    lab = np.array(["H"] * n, dtype="<U1")
    return lab, period, bx, by


def test_alone_ball_at_rest_becomes_dead():
    lab, period, bx, by = _case()
    out = edges.still_dead(lab, period, bx, by, np.full(len(lab), 5.0))
    d = np.flatnonzero(out == "D")
    assert len(d) >= 75 and d[0] >= 96 and d[-1] <= 204
    assert (out[:90] == "H").all() and (out[210:] == "H").all()


def test_held_ball_is_left_alone():
    lab, period, bx, by = _case()
    out = edges.still_dead(lab, period, bx, by, np.full(len(lab), 0.3))
    assert (out == lab).all()


def test_short_rest_and_hidden_ball_untouched():
    lab, period, bx, by = _case()
    bx2 = bx.copy()
    bx2[150:160] = np.nan                      # two rests under 3 s, split by a hidden stretch
    out = edges.still_dead(lab, period, bx2, by, np.full(len(lab), 5.0))
    assert (out == lab).all()
