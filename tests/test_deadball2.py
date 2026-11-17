import numpy as np

from possess.deadball2 import detect, features

N = 200


def _feat(far, line, team_speed=None):
    far = np.asarray(far, float)
    line = np.asarray(line, float)
    ts = np.full(len(far), 2.5) if team_speed is None else np.asarray(team_speed, float)
    return {"far": far, "line": line, "team_speed": ts, "bspeed": np.zeros(len(far))}


def test_ball_far_and_near_line_stops_play_then_restarts():
    far = np.r_[np.full(50, 0.5), np.full(80, 10.0), np.full(70, 0.5)]
    line = np.r_[np.full(50, 20.0), np.full(80, 1.0), np.full(70, 20.0)]
    dead = detect(_feat(far, line), np.zeros(N, bool), r_far=6, d_line=2, on_frames=25, off_frames=5)
    assert not dead[:50].any()
    assert dead[55:125].all()          # back-dated to when the evidence began
    assert not dead[140:].any()


def test_long_pass_mid_pitch_is_not_dead():
    far = np.r_[np.full(50, 0.5), np.full(40, 15.0), np.full(110, 0.5)]   # 1.6 s in the air
    line = np.full(N, 25.0)
    dead = detect(_feat(far, line), np.zeros(N, bool), r_far=6, d_line=2, on_frames=25)
    assert not dead.any()


def test_hidden_ball_gives_no_evidence():
    far = np.full(N, np.nan)
    line = np.full(N, np.nan)
    assert not detect(_feat(far, line), np.zeros(N, bool)).any()


def test_matchlens_dead_counts_as_evidence():
    base = np.zeros(N, bool); base[60:120] = True
    dead = detect(_feat(np.full(N, 0.5), np.full(N, 20.0)), base, on_frames=10, off_frames=5)
    assert dead[70:119].all()


def test_features_shapes_and_line_sign():
    bx = np.array([0.0, 53.0, 10.0]); by = np.array([0.0, 0.0, 35.0])
    f = features(bx, by, np.ones(3), np.ones(3), np.ones(3), np.ones(3))
    assert f["line"][1] < 0 and f["line"][2] < 0 and f["line"][0] == 34.0


# ---- envelope L2-11: still ball = dead from the frame it stopped; play resumes on the kick ----
def _ball(n=300, stop=100, kick=220, noise=0.0, seed=0):
    """Ball rolling at 8 m/s along x, still from `stop`, kicked at 12 m/s from `kick`."""
    t = np.arange(n)
    x = np.where(t < stop, (t - stop) * 8 / 25.0, np.where(t < kick, 0.0, (t - kick) * 12 / 25.0))
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, noise, n), rng.normal(0, noise, n)


def _detect(bx, by, d_near, **kw):
    f = features(bx, by, d_near, d_near + 5.0, np.full(len(bx), 3.0), np.full(len(bx), 3.0))
    return detect(f, np.zeros(len(bx), bool), r_far=9.0, d_line=2.0, v_slow=1.5, on_frames=50, off_frames=5, **kw)


def test_lag_speed_has_sharp_edges():
    from possess.deadball2 import lag_speed

    bx, by = _ball()
    v = lag_speed(bx, by)
    assert (v[102:218] < 0.3).all()                 # slow over the whole still stretch
    assert (v[:95] > 3).all() and (v[226:280] > 3).all()


def test_still_ball_is_dead_from_the_frame_it_stopped_until_the_kick():
    bx, by = _ball()
    taker = np.full(300, 0.5)                       # a player stands at the ball all the time
    dead = _detect(bx, by, taker, still_v=0.3, still_far=0.0, v_kick=3.0)
    assert not dead[:97].any() and dead[103:218].all() and not dead[226:].any()
    # old rule: the player standing next to the ball keeps play alive
    assert not _detect(bx, by, taker).any()
    # MatchLens's own radius: a player within 2 m of the still ball -> no evidence
    assert not _detect(bx, by, taker, still_v=0.3, still_far=2.0, v_kick=3.0).any()


def test_still_ball_shorter_than_on_frames_stays_live():
    bx, by = _ball(stop=100, kick=130)              # 1.2 s on the ball, then a pass
    dead = _detect(bx, by, np.full(300, 0.5), still_v=0.3, still_far=0.0, v_kick=3.0)
    assert not dead[:200].any()                     # (after frame ~240 the test ball leaves the pitch)


def test_still_and_kick_survive_ball_jitter():
    bx, by = _ball(noise=0.05)
    dead = _detect(bx, by, np.full(300, 0.5), still_v=0.3, still_far=0.0, v_kick=3.0)
    assert dead[110:210].all() and not dead[:90].any() and not dead[235:].any()


def test_walking_taker_does_not_restart_play():
    # ball out (far from everyone, at the line) then a player walks up and stands by it for 3 s
    n = 300
    bx, by = np.full(n, 0.0), np.full(n, 33.5)
    far = np.r_[np.full(120, 12.0), np.full(180, 0.5)]
    dead_new = _detect(bx, by, far, still_v=0.3, still_far=0.0, v_kick=3.0)
    dead_old = _detect(bx, by, far)
    assert dead_new[10:290].all()
    assert dead_old[10:115].all() and not dead_old[130:].any()
