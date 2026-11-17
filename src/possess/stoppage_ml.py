"""
Composite dead-ball detector — ported from MatchLens (footballtactics backend/events/deadball.py and
the PlacedBallDetector in backend/events/ball_strike.py), kept inside this repo so the pipeline has
no outside dependency.

Changes from MatchLens, each measured on the external bench (.planning/LOOP.md):
  * GHOST signal OFF by default (GHOST_BALL_DEAD_S = None, ADR-002): on this data a missing ball is
    not a stoppage (41% of graded frames have no ball, mostly in 5-30 s blocks). Missing ball is
    UNKNOWN evidence and only holds the current state.
  * Goalkeeper-holding override is not wired: it needs MatchLens's per-player possessor and
    attacking-direction modules, which this team-level pipeline does not have. `update()` still
    accepts `last_possessor` so it can be wired later.
  * Ball speed and the placed-ball flag, which MatchLens's Eventer computed upstream, are computed
    here by `ball_speed_sg()` and `PlacedBallDetector` (speed-only path: tracking has no ball height).
  * `run_detector()` drives the detector over whole-match arrays.

Original MatchLens description follows.

Vidal-Codina §2.3.4 assumes the in-play / dead-ball boolean is provided by the
tracking system. MatchLens does not have that signal — we derive it from a
combination of:

    1. OUT-OF-BOUNDS: ball map_position outside [0, 105] x [0, 68]
       with 0.5 m tolerance. Note: the FIFA coord system used by the project
       (mapper.py) places origin at center, so bounds are:
           x ∈ [-52.5, 52.5], y ∈ [-34, 34]
       Strongest signal — near-zero false-positive rate.

    2. STATIONARY: ball speed_mps < 0.3 sustained for ≥ 2.0 s AND no player
       within 2 m of ball. Catches throw-in setups, foul stoppages.

    3. GHOST: ball missing (conf is None or AdvancedBallTracker has no track)
       for ≥ 0.5 s. Tracker has lost the ball; assume dead.

To prevent jitter, all transitions go through LEAKY-BUCKET HYSTERESIS: each
frame's evidence is classified as DEAD (any dead signal), PLAY (healthy
in-bounds ball), or UNKNOWN (ball missing but ghost not yet armed). A flip
candidacy opens only on direct contradiction of the current state and commits
once net support reaches 0.5 s (15 frames @ 30 Hz):

    - toward dead:  DEAD +1, UNKNOWN +1 (ball absence is never evidence of
      play — an out ball that the tracker then loses keeps its candidacy),
      PLAY −1
    - toward play:  PLAY +1, UNKNOWN holds, DEAD −1
    - support <= 0 drops the candidacy; 50/50 boundary jitter self-cancels
      and the state conservatively stays put, while a 2:1 dead-majority
      still commits in ~3x the window.

Committed flips are BACK-STAMPED to the candidacy anchor (the first
supporting frame), so `dead_ball_start_frame` and the dead interval reflect
when the transition actually began, not when it was confirmed 0.5 s later.

Known false positives (accepted in v1):
    - Goalkeeper holds ball >6 s before drop kick → ghost+stationary fires.
      Mitigated: if last possessor is a goalkeeper inside their own pen area,
      the "GK_HOLDING" sub-state suppresses both ghost and stationary (never
      out-of-bounds) until the 6 s cap expires.
    - Quick free kicks taken before referee whistle: short stoppage missed
      entirely. Acceptable error.
    - Midfield foul stoppages where players crowd the ball: the stationary
      signal requires no player within 2 m, so these are invisible. A 4th
      "all-stop" signal (ball AND players slow) was considered and deferred:
      the Metrica benchmark can't measure it (its ball data is absent while
      dead, so ghost already covers benchmark stoppages) and it can only add
      unmeasurable false positives during slow open play.
"""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import Any, Deque, Dict, Optional, Tuple

# FIFA pitch bounds (matches mapper.py FIFA_PITCH_METERS convention)
PITCH_X_HALF = 52.5
PITCH_Y_HALF = 34.0

# Per-frame evidence classes for the leaky-bucket hysteresis
_EV_DEAD = "dead"
_EV_PLAY = "play"
_EV_UNKNOWN = "unknown"  # ball missing, ghost not yet armed


class DeadBallDetector:
    """Frame-level composite ball-in-play / dead-ball state machine."""

    # ---- Tunables ----
    #
    # Durations are declared in SECONDS and converted to frames against the
    # feed's actual fps at construction. They used to be frame counts sized for
    # 30 Hz, which silently changed meaning on any other feed: at 25 Hz the
    # 15-frame hysteresis became 0.6 s and the 60-frame stationary window 2.4 s,
    # so the detector was measurably more sluggish on 25 Hz sources — including
    # the Metrica benchmark data. Distances stay in metres and are fps-invariant.
    OOB_TOLERANCE_M = 0.5
    STATIONARY_VEL_THRESHOLD_MPS = 0.3
    STATIONARY_DURATION_S = 2.0
    NO_PLAYER_RADIUS_M = 2.0
    GHOST_BALL_DEAD_S: Optional[float] = None  # MatchLens: 0.5. None = ghost never means dead (ADR-002)
    HYSTERESIS_S = 0.5  # net supporting evidence before flipping state

    # Displacement fallback for the stationary signal when speed_mps is
    # unavailable (SG filter needs a 7-frame window; detection gaps starve it):
    # the ball counts as slow if the median position of the older half of a
    # 1 s window is within 0.5 m of the newer half's median. Medians suppress
    # single-frame homography jitter that raw endpoint deltas would not.
    STATIONARY_DISP_WINDOW_S = 1.0
    STATIONARY_DISP_M = 0.5

    # GK-holding suppression: if last in-play possessor was a GK in own box,
    # don't commit dead-ball; the GK is just holding the ball pre-distribution
    GK_HOLDING_MAX_S = 6.0  # beyond this it's actually dead

    def __init__(self, fps: float = 30.0):
        self.fps = float(fps) if fps and fps > 0 else 30.0

        # Seconds -> frames, once, against this feed's real rate.
        self.stationary_duration_frames = max(1, round(self.STATIONARY_DURATION_S * self.fps))
        self.ghost_ball_dead_frames = (
            None if self.GHOST_BALL_DEAD_S is None else max(1, round(self.GHOST_BALL_DEAD_S * self.fps))
        )
        self.hysteresis_frames = max(1, round(self.HYSTERESIS_S * self.fps))
        self.gk_holding_max_frames = max(1, round(self.GK_HOLDING_MAX_S * self.fps))

        # Public state
        self.in_play: bool = True
        self.dead_ball_start_frame: Optional[int] = None
        self.last_dead_ball_interval: Optional[Tuple[int, int]] = None

        # Cause of the most recent dead-ball flip (set-piece priors):
        #   "oob"        — ball observed out of bounds (exit_pos = first OOB fix)
        #   "stationary" — ball visibly stopped in-field
        #   "ghost"      — ball simply vanished (ambiguous: may have gone out)
        self.last_dead_cause: Optional[str] = None
        self.last_oob_exit_pos: Optional[Tuple[float, float]] = None
        self.last_pos_before_dead: Optional[Tuple[float, float]] = None

        # Internal hysteresis state (leaky bucket)
        self._flip_target: Optional[bool] = None
        self._flip_target_since_frame: Optional[int] = None  # candidacy anchor
        self._flip_support: int = 0
        # True once the dead candidacy has seen a HARD dead signal (OOB or
        # stationary). Purely ghost-anchored candidacies are drained harder by
        # real ball detections: a flickery-but-tracked ball is in play, while
        # an out-ball the tracker lost keeps full UNKNOWN support.
        self._flip_has_hard_dead: bool = False
        # Cause detail for the current toward-dead candidacy
        self._flip_oob_pos: Optional[Tuple[float, float]] = None
        self._flip_saw_stationary: bool = False
        self._flip_saw_placed: bool = False

        # Internal signal accumulators
        self._stationary_since_frame: Optional[int] = None
        self._ghost_since_frame: Optional[int] = None

        # Ball position ring for the displacement fallback
        self.disp_window_frames = max(2, round(self.STATIONARY_DISP_WINDOW_S * self.fps))
        self._pos_hist: Deque[Tuple[int, float, float]] = deque(
            maxlen=self.disp_window_frames
        )

        # GK-holding suppression
        self._last_possessor_was_gk: bool = False
        self._gk_holding_since_frame: Optional[int] = None

    # -------------------------------------------------------------------- API

    def update(
        self,
        frame_id: int,
        ball: Optional[Dict[str, Any]],
        players: Dict[int, Dict[str, Any]],
        last_possessor: Optional[Dict[str, Any]] = None,
        placed: bool = False,
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Tick the state machine.

        Args:
            frame_id: current frame
            ball: the ball track dict (with map_position, speed_mps, conf), or None
            players: tracks["players"] | tracks["goalkeepers"] merged
            last_possessor: dict like {"track_id", "team", "role", "in_own_box"}
                or None. Used to suppress dead-ball during GK holding.

        Returns:
            (in_play, last_dead_ball_interval)
            in_play is the public hysteresis-stable state.
            last_dead_ball_interval is (start_frame, end_frame) of the most
            recent dead-ball interval, or None if no dead ball has yet occurred.
        """
        # 1. Gather raw signals
        signals = self._gather_signals(frame_id, ball, players)
        # A ball resting on the ground is dead however many players stand over
        # it. The stationary signal cannot see this: it requires nobody within
        # 2 m, which the taker of the coming restart defeats by definition.
        # Measured on IDSSE (ball tracked while dead, as on our own footage),
        # dead recall was 81.5% without it.
        signals["placed"] = bool(placed)

        # 2. GK-holding override
        gk_holding = self._update_gk_holding(frame_id, last_possessor)

        # 3. Combine signals into proposed state
        any_dead_signal = (
            signals["oob"]
            or signals["stationary"]
            or signals["ghost"]
            or signals["placed"]
        )

        gk_suppressed = (
            gk_holding
            and (signals["ghost"] or signals["stationary"])
            and not signals["oob"]
        )
        if gk_suppressed:
            # GK holding the ball — suppress dead unless out of bounds.
            # Stationary is included: a GK standing with the ball and no
            # teammate within 2 m must not read as a dead ball.
            any_dead_signal = False

        # 4. Classify this frame's evidence
        if any_dead_signal:
            evidence = _EV_DEAD
        elif gk_suppressed:
            # Active suppression is positive evidence of play (drains any
            # pending dead candidacy), even while the ball reads as ghost.
            evidence = _EV_PLAY
        elif signals["missing"]:
            evidence = _EV_UNKNOWN
        else:
            evidence = _EV_PLAY

        # A placed ball is as hard a signal as OOB: it is a direct observation,
        # not an absence of evidence.
        hard_dead = signals["oob"] or signals["stationary"] or signals["placed"]

        # 5. Leaky-bucket hysteresis
        if self._flip_target is None:
            # A candidacy opens only on direct contradiction of the current
            # state — UNKNOWN alone never opens one, which preserves the
            # 0.5 s ghost-arming tolerance for brief occlusions.
            if evidence == _EV_DEAD and self.in_play:
                self._flip_target = False
                self._flip_target_since_frame = frame_id
                self._flip_support = 1
                self._flip_has_hard_dead = hard_dead
                self._note_dead_evidence(signals)
            elif evidence == _EV_PLAY and not self.in_play:
                self._flip_target = True
                self._flip_target_since_frame = frame_id
                self._flip_support = 1
        else:
            if self._flip_target is False:
                # Toward dead: absence of the ball is never evidence of play,
                # so UNKNOWN keeps supporting (an out ball the tracker then
                # loses must not reset the clock).
                if evidence in (_EV_DEAD, _EV_UNKNOWN):
                    self._flip_support += 1
                    if hard_dead:
                        self._flip_has_hard_dead = True
                    if evidence == _EV_DEAD:
                        self._note_dead_evidence(signals)
                else:
                    # A real in-bounds detection drains a purely ghost-anchored
                    # candidacy at double rate: a ball tracked ≥1/3 of the time
                    # is being played, whereas OOB/stationary-anchored
                    # candidacies stay robust to single-frame jitter.
                    self._flip_support -= 1 if self._flip_has_hard_dead else 2
            else:
                # Toward play: only a healthy ball is evidence; UNKNOWN holds.
                if evidence == _EV_PLAY:
                    self._flip_support += 1
                elif evidence == _EV_DEAD:
                    self._flip_support -= 1

            if self._flip_support <= 0:
                self._flip_target = None
                self._flip_target_since_frame = None
                self._flip_support = 0
                self._flip_has_hard_dead = False
                self._flip_oob_pos = None
                self._flip_saw_stationary = False
                self._flip_saw_placed = False
            elif self._flip_support >= self.hysteresis_frames:
                # Back-stamp the flip to where the candidacy began
                self._commit_flip(self._flip_target_since_frame, self._flip_target)

        return self.in_play, self.last_dead_ball_interval

    def _note_dead_evidence(self, signals: Dict[str, bool]) -> None:
        """Record cause detail while a toward-dead candidacy is supported."""
        if signals["oob"] and self._flip_oob_pos is None and self._pos_hist:
            _, bx, by = self._pos_hist[-1]
            self._flip_oob_pos = (bx, by)
        if signals["stationary"]:
            self._flip_saw_stationary = True
        if signals.get("placed"):
            self._flip_saw_placed = True

    def reset(self) -> None:
        self.in_play = True
        self.dead_ball_start_frame = None
        self.last_dead_ball_interval = None
        self.last_dead_cause = None
        self.last_oob_exit_pos = None
        self.last_pos_before_dead = None
        self._flip_oob_pos = None
        self._flip_saw_stationary = False
        self._flip_saw_placed = False
        self._flip_target = None
        self._flip_target_since_frame = None
        self._flip_support = 0
        self._flip_has_hard_dead = False
        self._stationary_since_frame = None
        self._ghost_since_frame = None
        self._pos_hist.clear()
        self._last_possessor_was_gk = False
        self._gk_holding_since_frame = None

    # ------------------------------------------------------------- internals

    def _gather_signals(
        self,
        frame_id: int,
        ball: Optional[Dict[str, Any]],
        players: Dict[int, Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Compute the three primary dead-ball signals for this frame."""
        signals = {
            "oob": False,
            "stationary": False,
            "ghost": False,
            "missing": False,
            "placed": False,  # supplied by the caller, not derived here
        }

        # --- Ghost ball: missing or coasted ---
        if ball is None or ball.get("map_position") is None:
            signals["missing"] = True
            if self._ghost_since_frame is None:
                self._ghost_since_frame = frame_id
            elif (self.ghost_ball_dead_frames is not None
                  and frame_id - self._ghost_since_frame >= self.ghost_ball_dead_frames):
                signals["ghost"] = True
            return signals  # Can't evaluate other signals without ball position
        else:
            # Ghost frame inside AdvancedBallTracker (conf is None) still counts
            # as ghost for our purposes — the ball position is Kalman-coasted
            ball_conf = ball.get("conf")
            if ball_conf is None:
                signals["missing"] = True
                if self._ghost_since_frame is None:
                    self._ghost_since_frame = frame_id
                elif (self.ghost_ball_dead_frames is not None
                      and frame_id - self._ghost_since_frame >= self.ghost_ball_dead_frames):
                    signals["ghost"] = True
            else:
                self._ghost_since_frame = None

        # --- Out of bounds (using FIFA-pitch centered coord system) ---
        bx, by = ball["map_position"]
        if (
            abs(bx) > PITCH_X_HALF + self.OOB_TOLERANCE_M
            or abs(by) > PITCH_Y_HALF + self.OOB_TOLERANCE_M
        ):
            signals["oob"] = True

        # --- Stationary: slow ball AND no player nearby ---
        self._pos_hist.append((frame_id, bx, by))
        slow = self._slow_verdict(ball.get("speed_mps"))
        if slow is True:
            # Check no player within 2 m of ball
            nearest_dist = self._nearest_player_dist(bx, by, players)
            if nearest_dist is None or nearest_dist > self.NO_PLAYER_RADIUS_M:
                if self._stationary_since_frame is None:
                    self._stationary_since_frame = frame_id
                elif (
                    frame_id - self._stationary_since_frame
                    >= self.stationary_duration_frames
                ):
                    signals["stationary"] = True
            else:
                self._stationary_since_frame = None
        elif slow is False:
            self._stationary_since_frame = None
        # slow is None (indeterminate): hold the accumulator — a speed gap is
        # not evidence of movement, and the since-frame arithmetic means held
        # frames still count toward the duration.

        return signals

    def _slow_verdict(self, ball_speed: Optional[float]) -> Optional[bool]:
        """
        Three-way slow classification:
            True  — speed reading says slow, or (no speed reading) the
                    displacement fallback confirms the ball is static
            False — demonstrably fast or displaced
            None  — speed unavailable and history too thin to judge
        """
        if ball_speed is not None:
            return ball_speed < self.STATIONARY_VEL_THRESHOLD_MPS

        if len(self._pos_hist) < 2:
            return None
        span = self._pos_hist[-1][0] - self._pos_hist[0][0]
        if span < 0.8 * self.disp_window_frames:
            return None

        half = len(self._pos_hist) // 2
        older = list(self._pos_hist)[:half]
        newer = list(self._pos_hist)[half:]
        dx = median(p[1] for p in newer) - median(p[1] for p in older)
        dy = median(p[2] for p in newer) - median(p[2] for p in older)
        return (dx * dx + dy * dy) ** 0.5 < self.STATIONARY_DISP_M

    @staticmethod
    def _nearest_player_dist(
        bx: float, by: float, players: Dict[int, Dict[str, Any]]
    ) -> Optional[float]:
        """Returns the smallest Euclidean distance from ball to any player, or None."""
        min_d = None
        for pdict in players.values():
            mp = pdict.get("map_position")
            if mp is None:
                continue
            d = ((mp[0] - bx) ** 2 + (mp[1] - by) ** 2) ** 0.5
            if min_d is None or d < min_d:
                min_d = d
        return min_d

    def _update_gk_holding(
        self, frame_id: int, last_possessor: Optional[Dict[str, Any]]
    ) -> bool:
        """Track whether a GK in own box has been holding the ball."""
        if last_possessor is None:
            self._last_possessor_was_gk = False
            self._gk_holding_since_frame = None
            return False

        is_gk = last_possessor.get("role") == "goalkeeper"
        in_own_box = last_possessor.get("in_own_box", False)

        if is_gk and in_own_box:
            if not self._last_possessor_was_gk:
                self._gk_holding_since_frame = frame_id
                self._last_possessor_was_gk = True
            elif self._gk_holding_since_frame is not None:
                # Cap GK-holding suppression at 6 s — beyond that it really is dead
                if frame_id - self._gk_holding_since_frame > self.gk_holding_max_frames:
                    return False
            return True
        else:
            self._last_possessor_was_gk = False
            self._gk_holding_since_frame = None
            return False

    def _commit_flip(self, flip_frame: int, new_state: bool) -> None:
        """
        Atomically transition the public state.

        `flip_frame` is the candidacy anchor — the first frame that supported
        this flip — so interval boundaries reflect when the transition began,
        not when hysteresis confirmed it 0.5 s later.
        """
        if new_state and not self.in_play:
            # Transitioning to in-play: close the dead-ball interval
            if self.dead_ball_start_frame is not None:
                self.last_dead_ball_interval = (self.dead_ball_start_frame, flip_frame)
                self.dead_ball_start_frame = None
        elif not new_state and self.in_play:
            # Transitioning to dead-ball: open a new interval and publish the
            # cause gathered during the candidacy (OOB wins over stationary;
            # neither seen → the ball simply vanished)
            self.dead_ball_start_frame = flip_frame
            if self._flip_oob_pos is not None:
                self.last_dead_cause = "oob"
            elif self._flip_saw_placed:
                self.last_dead_cause = "placed"
            elif self._flip_saw_stationary:
                self.last_dead_cause = "stationary"
            else:
                self.last_dead_cause = "ghost"
            self.last_oob_exit_pos = self._flip_oob_pos
            self.last_pos_before_dead = (
                (self._pos_hist[-1][1], self._pos_hist[-1][2])
                if self._pos_hist
                else None
            )

        self.in_play = new_state
        self._flip_target = None
        self._flip_target_since_frame = None
        self._flip_support = 0
        self._flip_has_hard_dead = False
        self._flip_oob_pos = None
        self._flip_saw_stationary = False
        self._flip_saw_placed = False

        # Signal accumulators deliberately survive the flip. They reset when
        # their own condition breaks (ball moves / player approaches / ball
        # reappears). Resetting them here forced armed signals to re-arm from
        # scratch, during which the state read as PLAY and flipped straight
        # back — a ~3 s dead/alive oscillation for stationary dead balls.


# ---------------------------------------------------------------------------
# Inputs MatchLens's Eventer computed upstream of the detector (ported)
# ---------------------------------------------------------------------------
import math  # noqa: E402

import numpy as np  # noqa: E402


class BallSpeed:
    """MatchLens PitchKinematics ball speed (footballtactics events/kinematics.py): Savitzky-Golay
    (window 7, order 2) smoothed central difference smooth[mid+1] - smooth[mid-1] over the last 7
    observed samples; skipped when those samples span more than 1.3 x 6 frames (a track blink)."""

    WINDOW, POLY = 7, 2

    def __init__(self, fps: float = 25.0):
        self.fps = float(fps)
        k = np.arange(self.WINDOW) - self.WINDOW // 2
        A = np.vander(k, self.POLY + 1, increasing=True)
        S = A @ np.linalg.pinv(A)                      # smoothing (hat) matrix
        mid = self.WINDOW // 2
        self._w = S[mid + 1] - S[mid - 1]
        self._buf: Deque[Tuple[int, float, float]] = deque(maxlen=self.WINDOW)

    def update(self, frame_id: int, pos: Optional[Tuple[float, float]]) -> Optional[float]:
        if pos is None:
            return None
        self._buf.append((frame_id, pos[0], pos[1]))
        if len(self._buf) < self.WINDOW:
            return None
        if self._buf[-1][0] - self._buf[0][0] > (self.WINDOW - 1) * 1.3:
            return None
        mid = self.WINDOW // 2
        dframes = self._buf[mid + 1][0] - self._buf[mid - 1][0]
        if dframes <= 0:
            return None
        dt = dframes / self.fps
        xs = np.array([b[1] for b in self._buf]); ys = np.array([b[2] for b in self._buf])
        return float(math.hypot(self._w @ xs / dt, self._w @ ys / dt))


class PlacedBallDetector:
    """MatchLens PlacedBallDetector (footballtactics events/ball_strike.py), speed-only path: a ball
    resting on the ground (speed <= 0.4 m/s, drifting <= 0.6 m) for >= 0.8 s is placed for a restart.
    The height branch is dropped: tracking here has no ball height."""

    def __init__(self, fps: float = 25.0, max_speed_mps: float = 0.4, max_drift_m: float = 0.6,
                 min_hold_s: float = 0.8):
        self.max_speed_mps, self.max_drift_m = max_speed_mps, max_drift_m
        self.min_hold_frames = max(1, round(min_hold_s * fps))
        self._since: Optional[int] = None
        self._anchor: Optional[Tuple[float, float]] = None

    def update(self, frame_id: int, pos: Optional[Tuple[float, float]], speed: Optional[float]) -> bool:
        if pos is None or speed is None or speed > self.max_speed_mps:
            self._since, self._anchor = None, None
            return False
        if self._anchor is None or math.hypot(pos[0] - self._anchor[0], pos[1] - self._anchor[1]) > self.max_drift_m:
            self._since, self._anchor = frame_id, pos
        return frame_id - self._since + 1 >= self.min_hold_frames


class MatchLensStoppage:
    """Drives speed -> placed -> DeadBallDetector frame by frame, the way MatchLens's Eventer did
    (stage 4), minus the GK-holding input. Feed every frame in order; returns True while dead."""

    def __init__(self, fps: float = 25.0):
        self.speed = BallSpeed(fps)
        self.placed = PlacedBallDetector(fps)
        self.det = DeadBallDetector(fps)

    def update(self, frame_id: int, ball_pos: Optional[Tuple[float, float]], players: Dict[int, Dict[str, Any]]) -> bool:
        sp = self.speed.update(frame_id, ball_pos)
        placed = self.placed.update(frame_id, ball_pos, sp)
        ball = None
        if ball_pos is not None:
            ball = {"map_position": ball_pos, "conf": 1.0}
            if sp is not None:
                ball["speed_mps"] = sp
        in_play, _ = self.det.update(frame_id=frame_id, ball=ball, players=players, last_possessor=None,
                                     placed=placed)
        return not in_play
