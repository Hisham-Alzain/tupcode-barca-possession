import numpy as np

from possess.team_control import team_from_distances

NAN = np.nan


def run(dh, da, vis=None, **kw):
    dh, da = np.array(dh, float), np.array(da, float)
    vis = np.ones(len(dh), bool) if vis is None else np.array(vis, bool)
    return "".join(team_from_distances(dh, da, vis, **kw))


def test_gain_after_k_frames_of_control():
    assert run([1, 1, 1, 9, 9, 9, 9], [9, 9, 9, 1, 1, 1, 1], k=3) == "XXHHHAA"


def test_short_touch_is_not_a_turnover():
    # away player within the zone for 2 frames only (a deflection) -> home keeps it
    assert run([1, 1, 1, 9, 9, 1, 1], [9, 9, 9, 1, 1, 9, 9], k=3) == "XXHHHHH"


def test_duel_keeps_current_team():
    assert run([1, 1, 1, 2, 2, 2, 2], [9, 9, 9, 2, 2, 2, 2], k=3) == "XXHHHHH"


def test_hidden_ball_keeps_team():
    assert run([1, 1, 1, NAN, NAN], [9, 9, 9, NAN, NAN], vis=[1, 1, 1, 0, 0], k=3) == "XXHHH"


def test_dead_frames_and_restart():
    dead = np.array([0, 0, 0, 1, 1, 0, 0, 0], bool)
    assert run([1, 1, 1, 1, 1, 9, 9, 9], [9, 9, 9, 9, 9, 1, 1, 1], k=3, dead=dead) == "XXHDDHHA"


def test_restart_flip_gives_ball_to_other_team_after_stoppage():
    dead = np.array([0, 0, 0, 1, 1, 0, 0], bool)
    assert run([1, 1, 1, 1, 1, 9, 9], [9, 9, 9, 9, 9, 9, 9], k=3, dead=dead, restart_flip=True) == "XXHDDAA"


def test_backdate_dates_switch_at_first_control_frame():
    assert run([1, 1, 1, 9, 9, 9, 9], [9, 9, 9, 1, 1, 1, 1], k=3, backdate=True) == "HHHAAAA"


def test_backdate_never_overwrites_dead_frames():
    dead = np.array([0, 0, 0, 0, 1, 0, 0], bool)
    out = run([1, 1, 1, 9, 9, 9, 9], [9, 9, 9, 1, 1, 1, 1], k=3, backdate=True, dead=dead)
    assert out[4] == "D"


def test_challenge_does_not_survive_a_long_ball_gap():
    # H holds; one A control frame; 60 frames hidden; then A controls 4 frames (k=3)
    dh = [1, 1, 1, 9] + [NAN] * 60 + [9, 9, 9, 9]
    da = [9, 9, 9, 1] + [NAN] * 60 + [1, 1, 1, 1]
    vis = [1, 1, 1, 1] + [0] * 60 + [1, 1, 1, 1]
    out = run(dh, da, vis=vis, k=3, backdate=True, max_gap=50)
    assert out[4:64] == "H" * 60                 # gap not repainted to A
    assert out[-2:] == "AA"


def test_period_change_resets_state():
    period = np.array([1, 1, 1, 2, 2], np.int8)
    out = run([1, 1, 1, 9, 9], [9, 9, 9, 9, 9], k=3, period=period)
    assert out == "XXHXX"


def test_ball_in_flight_keeps_team():
    # nobody within 1.8 m (a long pass in the air) -> no change
    assert run([1, 1, 1, 6, 6, 6], [9, 9, 9, 7, 7, 7], k=3) == "XXHHHH"
