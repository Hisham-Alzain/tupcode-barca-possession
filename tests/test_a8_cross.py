"""Envelope L8 round 3: stoppage entry at the line crossing (edges.cross_entry, decode_halves ed_cross)."""
import numpy as np

from possess import decode, edges


def _case(n=400, cross_at=200, d_from=230):
    """Ball runs along x and leaves the pitch over the goal line at `cross_at`; labels switch to D at `d_from`."""
    period = np.ones(n, int)
    bx = np.full(n, 40.0)
    bx[:cross_at] = np.linspace(20.0, 52.4, cross_at)
    bx[cross_at:] = 53.5
    by = np.zeros(n)
    lab = np.array(["H"] * n, dtype="<U1")
    lab[d_from:] = "D"
    return lab, period, bx, by


def test_late_entry_moves_back_to_the_crossing():
    lab, period, bx, by = _case()
    out = edges.cross_entry(lab, period, bx, by, back_s=3.0, off=5)
    assert (out[:205] == "H").all() and (out[205:] == "D").all()


def test_early_entry_gives_the_frames_back_to_the_team_before():
    lab, period, bx, by = _case(d_from=180)
    out = edges.cross_entry(lab, period, bx, by, back_s=3.0, off=5)
    assert (out[:205] == "H").all() and (out[205:] == "D").all()


def test_no_crossing_or_out_of_window_changes_nothing():
    lab, period, bx, by = _case()
    assert (edges.cross_entry(lab, period, bx, by, back_s=1.0, off=5) == lab).all()      # crossing 1.2 s before our entry
    bx2 = np.where(np.arange(len(bx)) >= 200, np.nan, bx)                                 # ball hidden before the line
    assert (edges.cross_entry(lab, period, bx2, by, back_s=3.0, off=5) == lab).all()
    inside = np.full(len(bx), 10.0)
    assert (edges.cross_entry(lab, period, inside, by, back_s=3.0, off=5) == lab).all()


def test_never_removes_the_spell_before_nor_crosses_a_period():
    lab, period, bx, by = _case()
    lab[:198] = "A"                                    # H spell 198..229, crossing at 200: entry 205, spell kept
    out = edges.cross_entry(lab, period, bx, by, back_s=3.0, off=5)
    assert (out[198:205] == "H").all() and (out[:198] == "A").all()
    lab2, _, _, _ = _case()
    lab2[:228] = "A"                                   # H spell 228..229 starts after the crossing: one frame stays
    out2 = edges.cross_entry(lab2, period, bx, by, back_s=3.0, off=5)
    assert out2[228] == "H" and (out2[229:] == "D").all() and (out2[:228] == "A").all()
    period3 = np.ones(len(lab), int)
    period3[210:] = 2                                  # the crossing lies in another period
    lab3, _, _, _ = _case()
    assert (edges.cross_entry(lab3, period3, bx, by, back_s=3.0, off=5) == lab3).all()


def test_off_by_default_in_decode_halves():
    rng = np.random.default_rng(0)
    n = 600
    period = np.ones(n, int)
    feats = {"d_home": rng.uniform(0.2, 6, n), "d_away": rng.uniform(0.2, 6, n), "visible": np.ones(n, bool),
             "dead": np.arange(n) > 400, "ball_x": np.linspace(0, 60, n), "ball_y": np.zeros(n)}
    kw = dict(soft=1, w_rule=0.3, hard_dead=1)
    base = decode.decode_halves(period, feats=dict(feats), **kw)
    assert (decode.decode_halves(period, feats=dict(feats), ed_cross=0, **kw) == base).all()
    on = decode.decode_halves(period, feats=dict(feats), ed_cross=3.0, **kw)
    assert on.shape == base.shape and set(on) <= {"H", "A", "D"}


def test_strict_ignores_a_crossing_after_which_the_ball_came_back_in():
    lab, period, bx, by = _case()
    bx[215:] = 50.0                                    # back inside the pitch from 215 on, our entry at 230
    assert (edges.cross_entry(lab, period, bx, by, back_s=3.0, off=5, strict=1) == lab).all()
    assert (edges.cross_entry(lab, period, bx, by, back_s=3.0, off=5, strict=0) != lab).any()
    lab2, _, bx2, _ = _case()                          # ball stays out: strict moves the entry as before
    assert (edges.cross_entry(lab2, period, bx2, by, back_s=3.0, off=5, strict=1)[205:] == "D").all()


def test_cross_dead_adds_a_stoppage_only_where_none_is_near():
    n = 600
    period = np.ones(n, int)
    by = np.zeros(n)
    bx = np.full(n, 30.0)
    bx[200:300] = 53.5                                 # beyond the goal line by 1 m for 4 s, then back inside
    bx[250:280] = np.nan                               # hidden in between: stays dead
    lab = np.array(["H"] * n, dtype="<U1")
    out = edges.cross_dead(lab, period, bx, by)
    assert (out[205:300] == "D").all() and (out[:205] == "H").all() and (out[300:] == "H").all()
    lab2 = lab.copy()
    lab2[240:330] = "D"                                # our own stoppage within 3 s of the crossing: untouched
    assert (edges.cross_dead(lab2, period, bx, by) == lab2).all()
    bx3 = np.full(n, 30.0)
    bx3[200:300] = 52.8                                # 0.3 m beyond the line: inside the margin
    assert (edges.cross_dead(lab, period, bx3, by) == lab).all()


def test_cross_dead_is_capped_and_stays_in_its_period():
    n = 2000
    by = np.zeros(n)
    bx = np.full(n, 30.0)
    bx[200:] = np.nan
    bx[200:210] = 54.0                                 # out, then hidden for good
    lab = np.array(["A"] * n, dtype="<U1")
    out = edges.cross_dead(lab, np.ones(n, int), bx, by, max_s=30.0)
    assert (out[205:950] == "D").all() and (out[950:] == "A").all()
    period = np.ones(n, int)
    period[400:] = 2
    out2 = edges.cross_dead(lab, period, bx, by)
    assert (out2[205:400] == "D").all() and (out2[400:] == "A").all()


def test_cross_dead_minimum_length():
    n = 600
    by = np.zeros(n)
    bx = np.full(n, 30.0)
    bx[200:240] = 53.5                                 # out for 1.6 s
    lab = np.array(["H"] * n, dtype="<U1")
    assert (edges.cross_dead(lab, np.ones(n, int), bx, by, min_s=2.0) == lab).all()
    assert (edges.cross_dead(lab, np.ones(n, int), bx, by, min_s=1.0)[205:240] == "D").all()

