"""Ball cleaning before the touch detector (ballclean.py, envelope L2-9) on synthetic tracks."""
import numpy as np

from possess.ballclean import clean_ball


def _line(n=200, v=10.0, fps=25.0, x0=-40.0):
    """A ball rolling along +x at v m/s from x0, y = 1 (stays on the pitch)."""
    return x0 + np.arange(n) * v / fps, np.ones(n)


def test_a_good_track_is_returned_unchanged():
    x, y = _line()
    x[50:60] = y[50:60] = np.nan                        # hidden frames stay hidden
    cx, cy, rem = clean_ball(x, y)
    assert not rem.any()
    assert np.array_equal(cx, x, equal_nan=True) and np.array_equal(cy, y, equal_nan=True)
    assert cx is not x                                  # a copy: the features keep their ball


def test_empty_input():
    cx, cy, rem = clean_ball(np.array([]), np.array([]))
    assert len(cx) == len(cy) == len(rem) == 0


def test_teleport_that_comes_back_is_bridged():
    x, y = _line()
    tx, ty = x.copy(), y.copy()
    tx[80] += 6.0                                       # one frame
    ty[120:123] -= 8.0                                  # three frames
    cx, cy, rem = clean_ball(tx, ty)
    assert set(np.flatnonzero(rem)) == {80, 120, 121, 122}
    assert np.allclose(cx, x) and np.allclose(cy, y)
    assert tx[80] == x[80] + 6.0                        # input not modified


def test_jump_that_does_not_come_back_is_kept():
    x, y = _line()
    x[100:] -= 20.0                                     # the track continues somewhere else
    cx, cy, rem = clean_ball(x, y)
    assert not rem.any() and np.array_equal(cx, x)


def test_fast_ball_is_not_a_teleport():
    x, y = _line(n=60, v=33.0)                          # a hard shot, below v_max
    assert not clean_ball(x, y)[2].any()
    x, y = _line(n=50, v=45.0, x0=-45.0)                # faster than v_max but it keeps going
    assert not clean_ball(x, y)[2].any()


def test_blip_at_the_edge_of_a_hidden_block_is_removed():
    x, y = _line()
    x[60:90] = y[60:90] = np.nan
    x[90] += 9.0                                        # first frame after the block
    x[59] -= 9.0                                        # last frame before it
    cx, _, rem = clean_ball(x, y)
    assert set(np.flatnonzero(rem)) == {59, 90}
    assert np.isnan(cx[59]) and np.isnan(cx[90])        # nothing to bridge from: hidden


def test_stale_repeat_of_a_moving_ball_is_bridged():
    x, y = _line()
    sx = x.copy()
    sx[101:107] = sx[100]                               # six frames frozen, then it catches up
    cx, cy, rem = clean_ball(sx, y)
    assert set(np.flatnonzero(rem)) == set(range(101, 107))
    assert np.allclose(cx, x) and np.allclose(cy, y)


def test_ball_at_rest_and_long_freezes_are_kept():
    n = 200
    f = np.arange(n)
    x = np.where(f < 100, 5.0, 5.0 + 0.4 * (f - 99))    # at rest, then kicked at frame 100
    y = np.zeros(n)
    assert not clean_ball(x, y)[2].any()
    x2, _ = _line()
    x2[50:140] = x2[50]                                 # frozen for 3.6 s: kept (a stoppage on tune data)
    assert not clean_ball(x2, np.ones(n))[2].any()


def test_freeze_of_up_to_two_seconds_is_bridged():
    x, y = _line()
    sx = x.copy()
    sx[51:91] = sx[50]                                  # 40 frames
    cx, _, rem = clean_ball(sx, y, stale_max=50)
    assert rem.sum() == 40 and np.allclose(cx, x)


def test_hard_kick_is_not_a_teleport():
    """Fable QA of L2-9: one step longer than v_max allows, then slower motion along the same
    line, is a kick. It stays near the line between its neighbours, so it is kept."""
    n = 100
    x = np.full(n, -10.0)
    x[50] = -10.0 + 1.6                                 # 40 m/s over the first frame
    x[51:] = x[50] + 0.5 * np.arange(1, n - 50)         # then 12.5 m/s
    cx, _, rem = clean_ball(x, np.zeros(n))
    assert not rem.any() and np.array_equal(cx, x)


def test_jump_next_to_a_player_is_kept():
    x, y = _line()
    tx = x.copy()
    tx[80] += 6.0
    d = np.full(len(x), 5.0)
    assert clean_ball(tx, y, d)[2][80]                  # nobody near: a teleport
    d[80] = 0.6
    assert not clean_ball(tx, y, d)[2].any()            # a player within touch range: left alone
    assert clean_ball(tx, y, d, r_touch=0.5)[2][80]


def test_small_wobble_is_kept():
    x, y = _line(v=34.0, n=60)
    ty = y.copy()
    ty[30] += 1.0                                       # over v_max for one frame, 1 m off the line
    assert not clean_ball(x, ty)[2].any()


def test_short_run_before_a_hidden_block_near_the_track_is_kept():
    x, y = _line()
    x[60:90] = y[60:90] = np.nan
    x[58:60] += 2.0                                     # a real 2 m jump, then the ball is hidden
    assert not clean_ball(x, y)[2].any()


def test_off_pitch_readings_become_hidden():
    x, y = _line(v=2.0)
    y[40:70] = 34.0 + 5.0                               # 5 m outside the touchline for 30 frames
    x[150] = 52.5 + 6.0                                 # one reading behind the goal line
    cx, cy, rem = clean_ball(x, y, off_m=3.0)
    assert rem[40:70].all() and rem[150] and rem.sum() == 31
    assert np.isnan(cy[40:70]).all()                    # too long to bridge
    assert np.isclose(cx[150], -40 + 150 * 2.0 / 25.0)   # a single reading is bridged
    x, _ = _line(v=2.0)
    assert not clean_ball(x, np.full(len(x), 35.5), off_m=3.0)[2].any()   # 1.5 m outside: a ball going out of play


def test_defaults_keep_ball_crossing_the_line_and_long_freezes():
    """Fable re-QA: off-pitch rule off by default; stale bridging limited to 10 frames."""
    import numpy as np
    from possess.ballclean import clean_ball

    x = np.linspace(50.0, 56.0, 40); y = np.zeros(40)              # ball runs over the goal line
    assert not clean_ball(x, y)[2].any()
