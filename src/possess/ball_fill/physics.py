"""Physics fill: two-sided ball flight with drag in the ground plane.

Idea (docs/research/ball-interpolation.md, method 4). A struck ball leaves the last seen point
with a velocity and slows down (air drag / rolling friction). Model per axis:

    p(t) = p0 + v0 * (1 - exp(-k t)) / k          k = decay rate (1/s)

Run it forward from the last known point (exit velocity) and backward from the next known point
(arrival velocity, time reversed), then blend with a smoothstep so the path leaves the left anchor
on the forward model and reaches the right anchor on the backward one. Both ends are matched.

MatchLens start: velocity comes from a least-squares fit over the known samples next to the gap,
the same role as MatchLens PitchKinematics (footballtactics events/kinematics.py; ported in possess/stoppage_ml.py BallSpeed) (Savitzky-Golay, WINDOW = 7,
POLY_ORDER = 2); here the samples need not be contiguous, so a degree-1 fit over up to `window`
anchors is used. Speeds are capped at `vmax` (a hard shot is ~35 m/s).

Beyond the fill, each gap also yields two physics signals for later stages (`gap_info`):
  * leaves_pitch   - the forward flight crosses a touchline/goal line inside the gap:
                     evidence the ball went OUT (a real dead ball), not just hidden;
  * disagreement   - max distance between forward and backward flights (m). Large means the
                     ball changed direction inside the gap: a touch, so a possible change of
                     possession hidden in the gap.
Defaults are physical guesses, NOT tuned (tune on degraded external matches only).
"""
import numpy as np

from .base import BallFill
from .linear import LinearFill

HALF_L, HALF_W = 52.5, 34.0


class PhysicsFill(BallFill):
    name = "physics"

    def __init__(self, max_gap=None, k: float = 0.6, window: int = 5, max_reach: int = 12,
                 vmax: float = 35.0, fps: float = 25.0, max_flight_s: float = 2.5):
        super().__init__(max_gap)
        self.k, self.window, self.max_reach, self.vmax, self.fps = k, window, max_reach, vmax, fps
        # One flight lasts a couple of seconds; beyond that the ball has been touched/stopped
        # and a drag model run backwards blows up exponentially. Longer gaps -> linear.
        self.max_flight = int(round(max_flight_s * fps))
        self._linear = LinearFill()
        self.gap_info = []  # one dict per filled gap (aggregates only are ever printed)

    def _anchors(self, known, start, step):
        out, i = [], start
        while 0 <= i < len(known) and len(out) < self.window and abs(i - start) <= self.max_reach:
            if known[i]:
                out.append(i)
            i += step
        return sorted(out)

    def _velocity(self, t, v, idx):
        tt = t[idx] / self.fps
        slope = np.polyfit(tt - tt.mean(), v[idx], 1)[0]
        return float(np.clip(slope, -self.vmax, self.vmax))

    def _flight(self, p0, v0, dt):
        """Forward: position dt seconds after leaving p0 with velocity v0."""
        k = self.k
        return p0 + v0 * (1.0 - np.exp(-k * dt)) / k if k > 0 else p0 + v0 * dt

    def _flight_back(self, p1, v1, tau):
        """Backward: where the ball was tau seconds before arriving at p1 with velocity v1.
        Under drag the ball was faster earlier: v(t1 - tau) = v1 * exp(k tau)."""
        k = self.k
        return p1 - v1 * (np.exp(k * tau) - 1.0) / k if k > 0 else p1 - v1 * tau

    def fill_gap(self, t, x, y, known, a, b):
        left, right = self._anchors(known, a - 1, -1), self._anchors(known, b, +1)
        if len(left) < 2 or len(right) < 2 or b - a > self.max_flight:
            gx, gy = self._linear.fill_gap(t, x, y, known, a, b)
            self.gap_info.append({"len": b - a, "model": "linear", "leaves_pitch": False,
                                  "disagreement": 0.0})
            return gx, gy
        t0, t1 = t[a - 1], t[b]
        q = t[a:b].astype(float)
        dt_f = (q - t0) / self.fps          # seconds since last seen
        dt_b = (t1 - q) / self.fps          # seconds until seen again
        out = []
        fwd_paths = []
        for v in (x, y):
            vin, vout = self._velocity(t, v, left), self._velocity(t, v, right)
            fwd = self._flight(v[a - 1], vin, dt_f)
            bwd = self._flight_back(v[b], vout, dt_b)  # time-reversed from the right anchor
            s = (q - t0) / float(t1 - t0)
            w = s * s * (3 - 2 * s)                   # smoothstep: forward early, backward late
            out.append((1 - w) * fwd + w * bwd)
            fwd_paths.append((fwd, bwd))
        (fx, bx), (fy, by) = fwd_paths
        self.gap_info.append({
            "len": b - a, "model": "physics",
            "leaves_pitch": bool(np.any((np.abs(fx) > HALF_L) | (np.abs(fy) > HALF_W))),
            "disagreement": float(np.max(np.hypot(fx - bx, fy - by))),
        })
        return out[0], out[1]

    def summary(self):
        g = [i for i in self.gap_info if i["model"] == "physics"]
        if not self.gap_info:
            return "no gaps"
        dis = np.array([i["disagreement"] for i in g]) if g else np.zeros(1)
        lens = np.array([i["len"] for i in self.gap_info]) / self.fps
        edges = [0, 0.5, 1, 2.5, 5, 10, 30, 1e9]
        hist = np.histogram(lens, bins=edges)[0]
        labels = ["<0.5s", "0.5-1s", "1-2.5s", "2.5-5s", "5-10s", "10-30s", ">30s"]
        frames = [int(sum(i["len"] for i in self.gap_info if lo <= i["len"] / self.fps < hi))
                  for lo, hi in zip(edges[:-1], edges[1:])]
        return ("gaps %d (physics %d, linear fallback %d); gap lengths [count/frames] %s; "
                "physics gaps: forward flight leaves pitch in %d, fwd/bwd disagreement "
                "median %.1f m, >5 m in %d"
                % (len(self.gap_info), len(g), len(self.gap_info) - len(g),
                   " ".join("%s:%d/%d" % (l, c, f) for l, c, f in zip(labels, hist, frames)),
                   sum(i["leaves_pitch"] for i in g), float(np.median(dis)), int((dis > 5).sum())))
