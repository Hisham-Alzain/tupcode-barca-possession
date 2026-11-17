"""Synthetic tests for src/possess/ball_fill (no challenge data)."""
import numpy as np
import pytest

from possess.ball_fill import CubicFill, KalmanRTSFill, LinearFill, PhysicsFill, get


def _series(fn, n=30, hole=(10, 16)):
    t = np.arange(n)
    x = np.array([fn(i)[0] for i in t], float)
    y = np.array([fn(i)[1] for i in t], float)
    known = np.ones(n, bool)
    known[hole[0]:hole[1]] = False
    xm, ym = x.copy(), y.copy()
    xm[~known] = ym[~known] = np.nan
    return t, x, y, xm, ym, known


def test_linear_matches_matchlens_formula():
    t, x, y, xm, ym, known = _series(lambda i: (i * 0.5, -i * 0.2))
    fx, fy, filled = LinearFill().fill(t, xm, ym, known)
    assert filled[10:16].all() and filled.sum() == 6
    n = 6
    for k, i in enumerate(range(10, 16), start=1):  # i / (n + 1) along the line (metrica.py)
        s = k / (n + 1)
        assert fx[i] == pytest.approx(x[9] + (x[16] - x[9]) * s)


def test_cubic_exact_on_cubic_motion():
    f = lambda i: (0.01 * (i - 12) ** 3 / 10, 0.3 * i)  # noqa: E731  (within pitch)
    t, x, y, xm, ym, known = _series(f)
    fx, fy, filled = CubicFill().fill(t, xm, ym, known)
    assert np.allclose(fx[10:16], x[10:16]) and np.allclose(fy[10:16], y[10:16])


def test_cubic_beats_linear_on_curved_flight():
    f = lambda i: (10 * np.sin(i / 6.0), 0.5 * i)  # noqa: E731
    t, x, y, xm, ym, known = _series(f)
    lx, _, _ = LinearFill().fill(t, xm, ym, known)
    cx, _, _ = CubicFill().fill(t, xm, ym, known)
    assert np.abs(cx[10:16] - x[10:16]).max() < np.abs(lx[10:16] - x[10:16]).max()


def test_edges_and_max_gap():
    t, x, y, xm, ym, known = _series(lambda i: (i, 0), n=20, hole=(0, 3))
    known[17:] = False
    fx, fy, filled = CubicFill().fill(t, xm, ym, known)
    assert (fx[0:3] == x[3]).all()          # start of half: hold next (MatchLens)
    assert not filled[17:].any()            # end of half: no target, no fill
    _, _, filled2 = LinearFill(max_gap=2).fill(*_series(lambda i: (i, 0))[0:1],
                                               *_series(lambda i: (i, 0))[3:6])
    assert not filled2.any()                # 6-frame gap > max_gap


def test_cubic_falls_back_to_linear_with_one_anchor():
    t, x, y, xm, ym, known = _series(lambda i: (i * 1.0, 0.0), n=20, hole=(1, 5))
    known[0] = True
    cx, _, f1 = CubicFill().fill(t, xm, ym, known)
    lx, _, f2 = LinearFill().fill(t, xm, ym, known)
    assert np.allclose(cx[1:5], lx[1:5])


def test_clip_to_pitch():
    t, x, y, xm, ym, known = _series(lambda i: (0.0, 0.0))
    xm[8], xm[9], xm[16], xm[17] = 0, 50, 50, 0  # violent reversal -> big overshoot
    fx, _, _ = CubicFill().fill(t, xm, ym, known)
    assert np.nanmax(np.abs(fx)) <= 54.5


def test_registry():
    assert get("none") is None and get("linear").name == "linear" and get("cubic").name == "cubic"
    assert get("kalman").name == "kalman"


def test_kalman_recovers_constant_velocity():
    t, x, y, xm, ym, known = _series(lambda i: (0.6 * i - 5, -0.3 * i), n=60, hole=(25, 37))
    fx, fy, filled = KalmanRTSFill().fill(t, xm, ym, known)
    assert filled[25:37].all()
    assert np.abs(fx[25:37] - x[25:37]).max() < 0.05 and np.abs(fy[25:37] - y[25:37]).max() < 0.05


def _drag_flight(i, v0=(20.0, 6.0), p0=(-30.0, -10.0), k=0.6, fps=25.0):
    t = i / fps
    f = (1 - np.exp(-k * t)) / k
    return (p0[0] + v0[0] * f, p0[1] + v0[1] * f)


def test_physics_beats_linear_on_decelerating_pass():
    t, x, y, xm, ym, known = _series(_drag_flight, n=80, hole=(10, 60))
    pf = PhysicsFill()
    px, py, filled = pf.fill(t, xm, ym, known)
    lx, ly, _ = LinearFill().fill(t, xm, ym, known)
    perr = np.hypot(px[10:60] - x[10:60], py[10:60] - y[10:60]).max()
    lerr = np.hypot(lx[10:60] - x[10:60], ly[10:60] - y[10:60]).max()
    assert filled[10:60].all() and perr < lerr
    assert pf.gap_info[-1]["disagreement"] < 2.0 and not pf.gap_info[-1]["leaves_pitch"]


def test_physics_flags_touch_and_ball_going_out():
    # ball runs right towards the touchline at y=34, then (after the gap) comes back: a touch
    f = lambda i: (0.0, -10 + 1.2 * i) if i < 20 else (0.0, -10 + 1.2 * 19 - 1.2 * (i - 19))  # noqa
    t, x, y, xm, ym, known = _series(f, n=60, hole=(20, 45))
    pf = PhysicsFill(k=0.0)
    pf.fill(t, xm, ym, known)
    info = pf.gap_info[-1]
    assert info["leaves_pitch"] and info["disagreement"] > 5


def _players(n, carrier, receiver=None):
    """Per-frame player dicts: slot 1 (team 1) runs along y=0; slot 2 (team 2) stands at (20, 5)."""
    return [{1: (carrier(i), 0.0, 1), 2: (20.0, 5.0, 2)} for i in range(n)]


def test_player_anchored_dribble_then_pass():
    from possess.ball_fill import PlayerAnchoredFill

    n, hole = 80, (10, 70)
    carrier = lambda i: 0.2 * i  # noqa: E731  dribbling player, 5 m/s
    t = np.arange(n)
    x = np.full(n, np.nan); y = np.full(n, np.nan)
    known = np.zeros(n, bool); known[:10] = known[70:] = True
    x[:10] = [carrier(i) for i in range(10)]; y[:10] = 0.0          # ball at the dribbler
    x[70:], y[70:] = 20.0, 5.0                                       # reappears at the opponent
    pf = PlayerAnchoredFill()
    pf.set_context(_players(n, carrier))
    fx, fy, filled = pf.fill(t, x, y, known)
    assert filled[10:70].all()
    assert np.allclose(fx[10:40], [carrier(i) for i in range(10, 40)])   # rides with the dribbler
    assert pf.gap_info[-1] == {"len": 60, "A": 1, "B": 2}                # team changed inside gap
    assert "team changed inside gap 1" in pf.summary()


def test_player_anchored_reanchors_when_slot_vanishes():
    from possess.ball_fill import PlayerAnchoredFill

    n = 40
    players = [{1: (float(i) * 0.2, 0.0, 1), 3: (30.0, 0.0, 1)} for i in range(n)]
    for i in range(15, n):  # slot 1 lost; same player reappears as slot 7
        players[i] = {7: (i * 0.2, 0.0, 1), 3: (30.0, 0.0, 1)}
    t = np.arange(n); x = np.full(n, np.nan); y = np.zeros(n)
    known = np.zeros(n, bool); known[:5] = known[35:] = True
    x[:5] = [i * 0.2 for i in range(5)]; x[35:] = [i * 0.2 for i in range(35, n)]
    pf = PlayerAnchoredFill(); pf.set_context(players)
    fx, _, _ = pf.fill(t, x, y, known)
    assert np.allclose(fx[5:35], [i * 0.2 for i in range(5, 35)], atol=0.5)


def test_player_anchored_without_context_is_linear():
    from possess.ball_fill import PlayerAnchoredFill

    t, x, y, xm, ym, known = _series(lambda i: (i * 0.5, 0.0))
    px, _, _ = PlayerAnchoredFill().fill(t, xm, ym, known)
    lx, _, _ = LinearFill().fill(t, xm, ym, known)
    assert np.allclose(px, lx, equal_nan=True)


@pytest.mark.parametrize("deg", [2, 3, 4, 5, 7])
def test_poly_exact_on_matching_degree(deg):
    f = lambda i: (sum(((i - 20) / 20.0) ** p for p in range(deg + 1)), 0.1 * i)  # noqa: E731
    t, x, y, xm, ym, known = _series(f, n=40, hole=(18, 22))
    fx, fy, filled = get("poly%d" % deg).fill(t, xm, ym, known)
    assert filled[18:22].all() and np.allclose(fx[18:22], x[18:22], atol=1e-6)


@pytest.mark.parametrize("name", ["log", "log3", "root2", "root3"])
def test_warped_hits_both_ends_and_front_loads(name):
    t, x, y, xm, ym, known = _series(lambda i: (float(i), 0.0), n=30, hole=(10, 20))
    fx, _, filled = get(name).fill(t, xm, ym, known)
    lx, _, _ = LinearFill().fill(t, xm, ym, known)
    assert filled[10:20].all()
    assert (np.diff(fx[9:21]) > 0).all()                 # monotone between the anchors
    assert (fx[10:20] >= lx[10:20] - 1e-9).all()         # ahead of constant speed (decelerating)
    assert np.diff(fx[9:21])[0] > np.diff(fx[9:21])[-1]  # fast early, slow late


def test_warped_matches_drag_flight_better_than_linear():
    k, fps = 0.8, 25.0
    f = lambda i: (30 * (1 - np.exp(-k * i / fps)) / k * 0.4, 0.0)  # noqa: E731
    t, x, y, xm, ym, known = _series(f, n=90, hole=(5, 80))
    lerr = np.abs(LinearFill().fill(t, xm, ym, known)[0][5:80] - x[5:80]).mean()
    werr = np.abs(get("log").fill(t, xm, ym, known)[0][5:80] - x[5:80]).mean()
    assert werr < lerr


def test_kalman_smooths_noise_better_than_linear_endpoints():
    rng = np.random.default_rng(0)
    t, x, y, xm, ym, known = _series(lambda i: (0.5 * i, 0.0), n=80, hole=(35, 50))
    noisy = xm + rng.normal(0, 0.5, len(xm))
    kx, _, _ = KalmanRTSFill(meas_std=0.5).fill(t, noisy, ym, known)
    lx, _, _ = LinearFill().fill(t, noisy, ym, known)
    assert np.abs(kx[35:50] - x[35:50]).mean() < np.abs(lx[35:50] - x[35:50]).mean()
