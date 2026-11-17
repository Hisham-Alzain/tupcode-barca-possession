import numpy as np

from possess import setpiece as sp


def _case():
    n = 1000
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    bx[:300], by[:300] = np.linspace(0, 54, 300), 0.0          # ball runs over the goal line
    bx[500:600], by[500:600] = 48.5, 3.0                       # placed in the goal area
    bx[600:], by[600:] = 48.5 - np.arange(400) * 0.4, 3.0      # kicked (10 m/s)
    lab = np.array(["A"] * n)
    lab[290:400] = "D"
    lab[400:] = "H"
    lab[600:620] = "D"
    return lab, bx, by, np.ones(n, int)


def test_goal_kick_dead_until_kick_then_taker():
    lab, bx, by, per = _case()
    out, info = sp.apply(lab, bx, by, per, [{"a": 580, "type": "Goal Kick", "sign": 1}])
    assert info[0]["kick"] == 600
    assert (out[400:600] == "D").all() and (out[600:620] == "H").all()
    assert (out[:400] == lab[:400]).all() and (out[620:] == lab[620:]).all()


def test_wrong_type_or_far_window_changes_nothing():
    lab, bx, by, per = _case()
    for ev in ({"a": 580, "type": "Corner Kick", "sign": 1}, {"a": 100, "type": "Goal Kick", "sign": 1},
               {"a": 580, "type": "Kick Off (Start)", "sign": 0}):
        out, _ = sp.apply(lab, bx, by, per, [ev])
        assert (out == lab).all()


def test_no_dead_evidence_within_gap_changes_nothing():
    lab, bx, by, per = _case()
    out, info = sp.apply(lab, bx, by, per, [{"a": 580, "type": "Goal Kick", "sign": 1}], {"gap_s": 4.0})
    assert info[0].get("no_dead") and (out[:600] == lab[:600]).all()


def test_spots():
    assert sp.spot_ok("corner kick", 52.0, -33.5, 1.0) and not sp.spot_ok("corner kick", 40.0, -33.5, 1.0)
    assert sp.spot_ok("throw-in", 10.0, 34.2, 1.0) and not sp.spot_ok("throw-in", 10.0, 20.0, 1.0)
    assert sp.spot_ok("goal kick", -48.0, 8.0, 0.0) and not sp.spot_ok("goal kick", -30.0, 0.0, 2.0)
