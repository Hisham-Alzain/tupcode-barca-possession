"""Kalman filter + Rauch-Tung-Striebel (RTS) smoother fill.

Not in MatchLens as a gap filler: MatchLens's AdvancedBallTracker (core/trackers) is a forward
Kalman that *coasts* a lost ball (positions marked conf=None); offline we can also run backwards,
so the RTS smoother uses the frames after the gap as well as before.

Model: constant velocity per axis, state [p, v], dt = 1/25 s, white-noise acceleration with std
`accel_std` (m/s^2), measurement noise `meas_std` (m). Per gap, the filter runs over a local
window of up to `context` known frames on each side; gap frames get a predict step but no update,
then the RTS pass smooths the whole window. Defaults are physical guesses, NOT tuned (tune on
degraded external matches only).
"""
import numpy as np

from .base import BallFill


def _cv_rts(z: np.ndarray, obs: np.ndarray, dt: float, q: float, r: float) -> np.ndarray:
    """1-D constant-velocity Kalman filter + RTS smoother. z: measurements, obs: mask."""
    n = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    H = np.array([1.0, 0.0])
    R = r * r
    xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    first = int(np.argmax(obs))
    x = np.array([z[first], 0.0]); P = np.diag([R, 25.0])  # unknown start speed: 5 m/s std
    for k in range(n):
        if k > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if obs[k]:
            S = H @ P @ H + R
            K = P @ H / S
            x = x + K * (z[k] - H @ x)
            P = P - np.outer(K, H @ P)
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0]


class KalmanRTSFill(BallFill):
    name = "kalman"

    def __init__(self, max_gap=None, context: int = 25, accel_std: float = 10.0,
                 meas_std: float = 0.3, fps: float = 25.0):
        super().__init__(max_gap)
        self.context, self.q, self.r, self.dt = context, accel_std, meas_std, 1.0 / fps

    def fill_gap(self, t, x, y, known, a, b):
        lo, hi = max(0, a - self.context), min(len(t), b + self.context)
        obs = known[lo:hi].copy()
        zx = np.where(obs, x[lo:hi], 0.0)
        zy = np.where(obs, y[lo:hi], 0.0)
        sx = _cv_rts(zx, obs, self.dt, self.q, self.r)
        sy = _cv_rts(zy, obs, self.dt, self.q, self.r)
        return sx[a - lo:b - lo], sy[a - lo:b - lo]
