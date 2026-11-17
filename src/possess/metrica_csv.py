"""
Metrica Sports open-data loader.

Source: https://github.com/metrica-sports/sample-data (Sample Games 1 & 2).
Two matches of 25 Hz 2D tracking for all players + ball, with a synchronized
manually-annotated event feed. This is the closest public analogue to the
in-stadium EPTS data Vidal-Codina et al. used, which is why Phase 0 starts here.

Format
------
Tracking CSVs (one per team, ball columns duplicated in both):

    row 0:  ,,,Home,,Home,, ...              team labels
    row 1:  ,,,11,,1,,2,, ...                shirt numbers
    row 2:  Period,Frame,Time [s],Player11,,Player1,, ... ,Ball,
    row 3+: 1,1,0.04,0.00082,0.48238, ...    normalised [0,1] coordinates

Coordinates are normalised to the unit square over a 105 x 68 m pitch, origin
top-left. MatchLens uses FIFA metres centred on the pitch, so the conversion is
an exact affine map with no scaling ambiguity — see `to_pitch_m`.

The first player column of each team file is the goalkeeper (Metrica's column
ordering convention). Substituted-off players read NaN.

Ball position is NaN whenever the game is dead. This is the same convention the
paper describes for Track160/Hawk-Eye ("indirectly with missing ball data when
the game is dead") and it gives us a free ground-truth in-play boolean — see
`MetricaMatch.ball_present_mask`.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Metrica's pitch. Matches the FIFA dimensions the event module already assumes
# (x in [-52.5, 52.5], y in [-34, 34]), so the conversion is a pure recentring.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

FPS = 25.0

HOME_TEAM_ID = 1
AWAY_TEAM_ID = 2


def to_pitch_m(x_norm: float, y_norm: float) -> Tuple[float, float]:
    """Normalised Metrica coordinates -> FIFA metres, origin at pitch centre.

    The y axis is *not* flipped. Every y-dependent test in the event module is
    symmetric about the halfway line (corner marks at (+-52.5, +-34), the cross
    channel at |y| > 20.15, carry zones at |y| > 18), so the sign convention is
    unobservable downstream. Documented rather than "corrected" to avoid a
    future reader introducing a mismatch while fixing a non-problem.
    """
    return (x_norm - 0.5) * PITCH_LENGTH_M, (y_norm - 0.5) * PITCH_WIDTH_M


def _f(v: str) -> Optional[float]:
    """Parse a Metrica cell. Empty and NaN both mean 'not tracked this frame'."""
    if v is None:
        return None
    v = v.strip()
    if not v or v == "NaN":
        return None
    try:
        out = float(v)
    except ValueError:
        return None
    return None if math.isnan(out) else out


def _player_num(name: str) -> Optional[int]:
    """'Player11' -> 11. Home players are 1-14, away 15-28, so ids never collide."""
    name = name.strip()
    if not name.lower().startswith("player"):
        return None
    try:
        return int(name[6:])
    except ValueError:
        return None


@dataclass
class GroundTruthEvent:
    """One row of the Metrica event feed, normalised to MatchLens conventions."""

    team_id: Optional[int]
    raw_type: str  # e.g. "PASS", "SHOT", "SET PIECE"
    raw_subtype: str  # e.g. "ON TARGET-GOAL", "CORNER KICK"
    period: int
    start_frame: int
    start_time_s: float
    end_frame: int
    end_time_s: float
    player_id: Optional[int]  # "From" — the executor
    receiver_id: Optional[int]  # "To"
    start_x_m: Optional[float]
    start_y_m: Optional[float]
    end_x_m: Optional[float]
    end_y_m: Optional[float]

    @property
    def subtype_tokens(self) -> List[str]:
        """'HEAD-ON TARGET-GOAL' -> ['HEAD', 'ON TARGET', 'GOAL']."""
        return [t for t in self.raw_subtype.split("-") if t]


@dataclass
class _TeamFile:
    """Parsed column layout of one team's tracking CSV."""

    team_id: int
    path: str
    # (player_id, x_col, y_col), goalkeeper first
    player_cols: List[Tuple[int, int, int]] = field(default_factory=list)
    gk_id: Optional[int] = None
    ball_cols: Optional[Tuple[int, int]] = None


def _parse_header(path: str, team_id: int) -> _TeamFile:
    with open(path, "r", encoding="utf-8-sig") as fh:
        rdr = csv.reader(fh)
        next(rdr)  # team labels
        next(rdr)  # shirt numbers
        names = next(rdr)  # column names

    tf = _TeamFile(team_id=team_id, path=path)
    for idx, raw in enumerate(names):
        if idx < 3:
            continue
        label = (raw or "").strip()
        if not label:
            continue
        if label.lower() == "ball":
            tf.ball_cols = (idx, idx + 1)
            continue
        pid = _player_num(label)
        if pid is not None:
            tf.player_cols.append((pid, idx, idx + 1))

    if not tf.player_cols:
        raise ValueError(f"No player columns found in {path}")
    # Metrica lists the goalkeeper first.
    tf.gk_id = tf.player_cols[0][0]
    return tf


@dataclass
class MetricaMatch:
    """A loaded Metrica match: tracking on disk, events in memory."""

    name: str
    home: _TeamFile
    away: _TeamFile
    events: List[GroundTruthEvent]
    fps: float = FPS

    # ---------------------------------------------------------------- tracking

    def iter_frames(
        self,
        max_frames: Optional[int] = None,
        interpolate_ball: bool = False,
    ) -> Iterator[Tuple[int, int, Dict[str, Any], bool]]:
        """Yield `(frame_id, period, tracks, ball_present)` per frame.

        `tracks` is shaped exactly like the pipeline's per-frame dict so the
        Eventer cannot tell it came from a CSV:

            {"players": {tid: {...}}, "goalkeepers": {tid: {...}},
             "ball": {1: {...}} }

        `ball_present` is the ground-truth in-play signal (Metrica drops ball
        data when the game is dead).

        `interpolate_ball=False` passes the gap straight through, so the ball
        genuinely vanishes — matching Metrica's own convention.

        `interpolate_ball=True` linearly fills the gap and marks the filled
        frames `conf=None` (the module's "Kalman-coasted ghost" convention).
        This is the honest setting for tuning: on real footage YOLO keeps
        seeing the ball during a dead ball, so a detector tuned against a
        genuinely-absent ball learns a signal it will not have in production.
        Report both; treat the absent-ball number as an upper bound.
        """
        h_rows = self._iter_raw(self.home)
        a_rows = self._iter_raw(self.away)

        pending: List[Tuple[int, int, Dict[str, Any], Optional[Tuple[float, float]]]] = []
        count = 0

        for (fid, period, h) , (fid2, _p2, a) in zip(h_rows, a_rows):
            if fid != fid2:
                raise ValueError(f"Tracking files out of sync at frame {fid} vs {fid2}")

            players: Dict[int, Dict[str, Any]] = {}
            keepers: Dict[int, Dict[str, Any]] = {}
            for tf, row in ((self.home, h), (self.away, a)):
                for pid, xc, yc in tf.player_cols:
                    xn, yn = _f(row[xc]), _f(row[yc])
                    if xn is None or yn is None:
                        continue  # substituted off / not tracked this frame
                    rec: Dict[str, Any] = {
                        "map_position": to_pitch_m(xn, yn),
                        "team": tf.team_id,
                        # Ground truth: a row that has coordinates is a real
                        # observation. The ball already carried conf=1.0 here;
                        # players did not, which left them indistinguishable
                        # from "entity not present at all" under the conf
                        # convention (Atlas 0.4) and made any consumer that
                        # gates on a real detection — the heat-map accumulator —
                        # silently see an empty match.
                        "conf": 1.0,
                    }
                    if pid == tf.gk_id:
                        rec["role"] = "goalkeeper"
                        keepers[pid] = rec
                    else:
                        players[pid] = rec

            bx = by = None
            if self.home.ball_cols:
                bxn = _f(h[self.home.ball_cols[0]])
                byn = _f(h[self.home.ball_cols[1]])
                if bxn is not None and byn is not None:
                    bx, by = to_pitch_m(bxn, byn)

            ball_present = bx is not None
            tracks: Dict[str, Any] = {"players": players, "goalkeepers": keepers, "ball": {}}
            if ball_present:
                tracks["ball"] = {1: {"map_position": (bx, by), "conf": 1.0}}

            if not interpolate_ball:
                yield fid, period, tracks, ball_present
                count += 1
                if max_frames and count >= max_frames:
                    return
                continue

            # --- interpolating variant: buffer gaps until the ball returns ---
            if ball_present:
                if pending:
                    prev = self._last_known
                    for i, (pfid, pper, ptracks, _) in enumerate(pending, start=1):
                        if prev is not None:
                            t = i / (len(pending) + 1)
                            ix = prev[0] + (bx - prev[0]) * t
                            iy = prev[1] + (by - prev[1]) * t
                        else:
                            ix, iy = bx, by
                        # conf=None marks a coasted position, per the module's
                        # ghost convention.
                        ptracks["ball"] = {1: {"map_position": (ix, iy), "conf": None}}
                        yield pfid, pper, ptracks, False
                        count += 1
                        if max_frames and count >= max_frames:
                            return
                    pending.clear()
                self._last_known = (bx, by)
                yield fid, period, tracks, True
                count += 1
                if max_frames and count >= max_frames:
                    return
            else:
                pending.append((fid, period, tracks, None))

        # Trailing dead-ball tail: emit with the last known position held.
        for pfid, pper, ptracks, _ in pending:
            if getattr(self, "_last_known", None) is not None:
                ptracks["ball"] = {1: {"map_position": self._last_known, "conf": None}}
            yield pfid, pper, ptracks, False

    _last_known: Optional[Tuple[float, float]] = None

    def _iter_raw(self, tf: _TeamFile) -> Iterator[Tuple[int, int, List[str]]]:
        with open(tf.path, "r", encoding="utf-8-sig") as fh:
            rdr = csv.reader(fh)
            for _ in range(3):
                next(rdr)
            for row in rdr:
                if len(row) < 3 or not row[1].strip():
                    continue
                yield int(row[1]), int(row[0]), row

    # ------------------------------------------------------------------ truth

    def ball_present_mask(self, max_frames: Optional[int] = None) -> Dict[int, bool]:
        """frame_id -> ball tracked this frame. Ground truth for `ball_in_play`."""
        mask: Dict[int, bool] = {}
        if not self.home.ball_cols:
            return mask
        bx_col, by_col = self.home.ball_cols
        for i, (fid, _period, row) in enumerate(self._iter_raw(self.home)):
            mask[fid] = _f(row[bx_col]) is not None and _f(row[by_col]) is not None
            if max_frames and i + 1 >= max_frames:
                break
        return mask

    def goalkeeper_ids(self) -> Dict[int, Optional[int]]:
        return {HOME_TEAM_ID: self.home.gk_id, AWAY_TEAM_ID: self.away.gk_id}


def _load_events(path: str) -> List[GroundTruthEvent]:
    out: List[GroundTruthEvent] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            team_raw = (row.get("Team") or "").strip().lower()
            team_id = (
                HOME_TEAM_ID if team_raw == "home" else AWAY_TEAM_ID if team_raw == "away" else None
            )

            sx, sy = _f(row.get("Start X", "")), _f(row.get("Start Y", ""))
            ex, ey = _f(row.get("End X", "")), _f(row.get("End Y", ""))
            sxm, sym = to_pitch_m(sx, sy) if sx is not None and sy is not None else (None, None)
            exm, eym = to_pitch_m(ex, ey) if ex is not None and ey is not None else (None, None)

            def _i(key: str) -> int:
                v = _f(row.get(key, ""))
                return int(v) if v is not None else 0

            out.append(
                GroundTruthEvent(
                    team_id=team_id,
                    raw_type=(row.get("Type") or "").strip().upper(),
                    raw_subtype=(row.get("Subtype") or "").strip().upper(),
                    period=_i("Period") or 1,
                    start_frame=_i("Start Frame"),
                    start_time_s=_f(row.get("Start Time [s]", "")) or 0.0,
                    end_frame=_i("End Frame"),
                    end_time_s=_f(row.get("End Time [s]", "")) or 0.0,
                    player_id=_player_num(row.get("From") or ""),
                    receiver_id=_player_num(row.get("To") or ""),
                    start_x_m=sxm,
                    start_y_m=sym,
                    end_x_m=exm,
                    end_y_m=eym,
                )
            )
    return out


def load_metrica_match(root: str, game: int = 1) -> MetricaMatch:
    """Load `Sample_Game_<game>` from a directory of Metrica sample data."""
    d = os.path.join(root, f"Sample_Game_{game}")
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"Metrica sample game not found at {d}. "
            "Fetch it from https://github.com/metrica-sports/sample-data"
        )
    home = _parse_header(
        os.path.join(d, f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"), HOME_TEAM_ID
    )
    away = _parse_header(
        os.path.join(d, f"Sample_Game_{game}_RawTrackingData_Away_Team.csv"), AWAY_TEAM_ID
    )
    events = _load_events(os.path.join(d, f"Sample_Game_{game}_RawEventsData.csv"))
    return MetricaMatch(name=f"Sample_Game_{game}", home=home, away=away, events=events)
