"""Player-anchored fill: while the ball is unseen, it is at a player's feet.

Idea (docs/research/ball-interpolation.md, method 6). Most long gaps are crowds, duels and
dribbles, so instead of drawing a line through empty space:

  * A = the player nearest the last seen ball (within `r_attach` m): the carrier when it vanished.
  * B = the player nearest the ball where it reappears: the receiver.
  * The ball rides with A, then travels to B as a pass at `pass_speed` m/s, arriving exactly when
    the ball is seen again, so the hand-off happens as late as the evidence allows.
  * Only A: ride with A, then a pass to the reappearance point. Only B: pass from the last seen
    point to B, then ride with B. Neither: linear (MatchLens).
  * A's track slot can vanish (slots are not identities): re-anchor to the nearest player of the
    same team to the ball's previous position.

MatchLens start: the nearest-player-to-ball rule is the possessor rule of
MatchLens possession.py (footballtactics events/possession.py, Vidal-Codina port; not in this repo), whose control radius is the same idea as
`r_attach`; the hold-then-pass shape is new. Untuned guesses; tune on degraded external data only.

Signals kept per gap (`gap_info`, aggregates only are printed): A's and B's teams. A gap that
starts with one team and ends with the other has a change of possession hidden inside it.
"""
import numpy as np

from .base import BallFill
from .linear import LinearFill


class PlayerAnchoredFill(BallFill):
    name = "player"

    def __init__(self, max_gap=None, r_attach: float = 3.0, pass_speed: float = 15.0,
                 fps: float = 25.0):
        super().__init__(max_gap)
        self.r, self.vpass, self.fps = r_attach, pass_speed, fps
        self._linear = LinearFill()
        self.players = None        # set per half by the adapter: list of {slot: (x, y, team)}
        self.gap_info = []

    def set_context(self, players):
        self.players = players

    def _nearest(self, i, pos, team=None):
        best, bd = None, np.inf
        for slot, (px, py, tm) in self.players[i].items():
            if team is not None and tm != team:
                continue
            d = (px - pos[0]) ** 2 + (py - pos[1]) ** 2
            if d < bd:
                best, bd = slot, d
        return best, np.sqrt(bd)

    def _ride(self, start, stop, slot, team, pos0):
        """Ball at the anchor player's feet for samples start..stop-1; re-anchor if slot vanishes."""
        xs, ys, pos = [], [], pos0
        for i in range(start, stop):
            p = self.players[i].get(slot)
            if p is None:
                slot, _ = self._nearest(i, pos, team)
                p = self.players[i].get(slot) if slot is not None else None
            if p is not None:
                pos = (p[0], p[1])
            xs.append(pos[0]); ys.append(pos[1])
        return np.array(xs), np.array(ys), pos

    def fill_gap(self, t, x, y, known, a, b):
        if self.players is None:
            return self._linear.fill_gap(t, x, y, known, a, b)
        p_last, p_next = (x[a - 1], y[a - 1]), (x[b], y[b])
        sa, da = self._nearest(a - 1, p_last)
        sb, db = self._nearest(b, p_next)
        A = (sa, self.players[a - 1][sa][2]) if sa is not None and da <= self.r else None
        B = (sb, self.players[b][sb][2]) if sb is not None and db <= self.r else None
        self.gap_info.append({"len": b - a, "A": A[1] if A else None, "B": B[1] if B else None})
        if A is None and B is None:
            return self._linear.fill_gap(t, x, y, known, a, b)

        n = b - a
        gx, gy = np.empty(n), np.empty(n)
        if A is not None:
            # ride with A, then pass to (B's position or the reappearance point) arriving at b
            target = p_next
            rx, ry, _ = self._ride(a, b, A[0], A[1], p_last)
            # pass length from where A is near the end of the gap
            d = np.hypot(rx[-1] - target[0], ry[-1] - target[1])
            k = int(min(n, np.ceil(d / self.vpass * self.fps)))
            m = n - k
            gx[:m], gy[:m] = rx[:m], ry[:m]
            if k:
                sx, sy = (rx[m - 1], ry[m - 1]) if m > 0 else p_last
                s = np.arange(1, k + 1) / (k + 1)
                gx[m:], gy[m:] = sx + (target[0] - sx) * s, sy + (target[1] - sy) * s
        else:
            # pass from the last seen point to B, then ride with B until the ball reappears
            bx0, by0 = self.players[a][B[0]][:2] if B[0] in self.players[a] else p_next
            d = np.hypot(bx0 - p_last[0], by0 - p_last[1])
            k = int(min(n, np.ceil(d / self.vpass * self.fps)))
            rx, ry, _ = self._ride(a, b, B[0], B[1], p_next)
            s = np.arange(1, k + 1) / (k + 1)
            gx[:k] = p_last[0] + (rx[:k] - p_last[0]) * s
            gy[:k] = p_last[1] + (ry[:k] - p_last[1]) * s
            gx[k:], gy[k:] = rx[k:], ry[k:]
        return gx, gy

    def summary(self):
        g = self.gap_info
        if not g:
            return "no gaps"
        both = [i for i in g if i["A"] and i["B"]]
        return ("gaps %d: carrier+receiver %d (same team %d, team changed inside gap %d), "
                "carrier only %d, receiver only %d, neither (linear) %d"
                % (len(g), len(both), sum(i["A"] == i["B"] for i in both),
                   sum(i["A"] != i["B"] for i in both),
                   sum(bool(i["A"]) and not i["B"] for i in g),
                   sum(bool(i["B"]) and not i["A"] for i in g),
                   sum(not i["A"] and not i["B"] for i in g)))
