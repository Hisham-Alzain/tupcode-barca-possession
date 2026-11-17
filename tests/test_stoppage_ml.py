"""Ported MatchLens dead-ball detector (possess.stoppage_ml): synthetic checks, no data needed."""
import math

import numpy as np

from possess.stoppage_ml import BallSpeed, DeadBallDetector, MatchLensStoppage, PlacedBallDetector

FPS = 25.0
NEAR = {1: {"map_position": (0.0, 0.0), "team": 1}, 2: {"map_position": (3.0, 0.0), "team": 2}}


def _run(ball_positions, players=NEAR):
    s = MatchLensStoppage(FPS)
    return [s.update(i, p, players) for i, p in enumerate(ball_positions)]


def test_ghost_rule_is_off_by_default():
    assert DeadBallDetector.GHOST_BALL_DEAD_S is None
    moving = [(i * 0.3, 0.0) for i in range(25)]
    dead = _run(moving + [None] * (30 * 25))          # 30 s without a ball
    assert not any(dead)


def test_ball_out_of_bounds_stops_play():
    path = [(40 + i * 0.5, 0.0) for i in range(60)]   # runs over the goal line at x = 52.5
    dead = _run(path)
    assert not dead[0] and dead[-1]


def test_resting_ball_stops_play_after_hysteresis():
    far = {1: {"map_position": (20.0, 0.0), "team": 1}}
    dead = _run([(0.0, 0.0)] * (5 * 25), players=far)
    first = dead.index(True)
    # speed needs 7 samples, the placed-ball signal 0.8 s at rest, then 0.5 s of hysteresis;
    # it fires before the 2 s "stationary" signal (MatchLens behaviour)
    assert 25 <= first <= 45 and all(dead[first:])


def test_ball_speed_matches_constant_velocity():
    sp = BallSpeed(FPS)
    out = [sp.update(i, (i * 0.4, i * 0.3)) for i in range(10)]   # 0.5 m/frame = 12.5 m/s
    assert out[5] is None and math.isclose(out[-1], 12.5, rel_tol=1e-9)


def test_ball_speed_skips_track_blinks():
    sp = BallSpeed(FPS)
    frames = [0, 1, 2, 3, 4, 5, 20]                     # span 20 > 1.3 * 6
    assert [sp.update(f, (f * 0.1, 0.0)) for f in frames][-1] is None


def test_placed_ball_after_hold_time():
    pb = PlacedBallDetector(FPS)
    res = [pb.update(i, (10.0, 5.0), 0.1) for i in range(30)]
    assert not res[18] and res[19]                      # 0.8 s = 20 frames
    assert not pb.update(31, (10.0, 5.0), 3.0)          # moving again
