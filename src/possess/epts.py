"""Read FIFA/EPTS tracking (Metrica format: the challenge match and Metrica Sample Game 3).

Reads the tracking with kloppy, treats every track slot as an anonymous player of its team (no
identities), guesses one goalkeeper per team per frame (deepest player on the team's own side),
and yields frames as `(frame_id, period, tracks, ball_present)` with
`tracks = {"players": {...}, "goalkeepers": {...}, "ball": {1: {"map_position", "conf"}}}`,
positions in pitch-centred metres (the frame shape MatchLens used, so the ported modules read it
unchanged). Optional offline ball gap filling via `possess.ball_fill`.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, Iterator, Optional, Tuple

PITCH_L, PITCH_W = 105.0, 68.0
FPS = 25.0


def to_pitch_m(xn: float, yn: float) -> Tuple[float, float]:
    """Metrica 0-1 (origin top-left) -> FIFA metres centred on the pitch (MatchLens convention)."""
    return (xn - 0.5) * PITCH_L, (0.5 - yn) * PITCH_W


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


class AtdMatch:
    """Frame source for EPTS tracking; same interface as possess.metrica_csv.MetricaMatch."""

    def __init__(self, meta_xml: str, raw_txt: str, name: str = "atd",
                 period_of: Optional[callable] = None, frame_filter: Optional[callable] = None):
        """Streams the raw file (kloppy's EPTS line reader) instead of building kloppy's whole
        TrackingDataset in memory: the full match is 199k frames x 46 slots. Coordinates are the
        raw Metrica values (0-1, origin top-left), identical to kloppy's default output."""
        from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata

        self.name, self.fps = name, FPS
        self._raw_txt = raw_txt
        with open(meta_xml, "rb") as fh:
            self._meta = load_metadata(fh)
        self.team_of: Dict[str, int] = {}
        for team in self._meta.teams:
            tid = 1 if str(team.ground).lower().endswith("home") else 2
            for p in team.players:
                self.team_of[p.player_id] = tid
        self._int_id = {pid: i + 1 for i, pid in enumerate(sorted(self.team_of))}
        self._keys = [(pid, "player_%s_x" % pid, "player_%s_y" % pid) for pid in sorted(self.team_of)]
        self._period_of = period_of
        self._frame_filter = frame_filter
        self.frames_per_period: Dict[int, int] = {}
        self._own_side = self._estimate_sides()

    def _frames(self):
        """Yield (frame_id, period, [(player_id, x01, y01)], (bx01, by01) or None), streamed."""
        from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data

        with open(self._raw_txt, "rb") as fh:
            for row in read_raw_data(fh, self._meta, sensor_ids=["position"]):
                fid = int(row["frame_id"])
                if self._frame_filter and not self._frame_filter(fid):
                    continue
                per = self._period_of(fid) if self._period_of else int(row.get("period_id") or 0)
                pts = [(pid, row.get(kx), row.get(ky)) for pid, kx, ky in self._keys]
                pts = [(p, x, y) for p, x, y in pts if _ok(x) and _ok(y)]
                bx, by = row.get("ball_x"), row.get("ball_y")
                yield fid, per, pts, ((bx, by) if _ok(bx) and _ok(by) else None)

    def _estimate_sides(self) -> Dict[Tuple[int, int], float]:
        """Sign of each team's mean x per period: which half it defends (algorithmic, no labels)."""
        acc: Dict[Tuple[int, int], list] = {}
        for i, (fid, per, pts, _) in enumerate(self._frames()):
            if i % 25:
                continue
            for pid, x, y in pts:
                key = (per, self.team_of.get(pid, 0))
                s = acc.setdefault(key, [0.0, 0])
                s[0] += x - 0.5
                s[1] += 1
        return {k: (1.0 if s[0] / max(s[1], 1) > 0 else -1.0) for k, s in acc.items()}

    ball_fill = None   # a possess.ball_fill.BallFill, or None for no gap filling
    filled_frames: int = 0

    def iter_frames(self, max_frames: Optional[int] = None, interpolate_ball: bool = False
                    ) -> Iterator[Tuple[int, int, dict, bool]]:
        """Yield frames for MatchLens. With `ball_fill` set, each half is buffered and its ball
        gaps filled offline by that method (filled frames carry conf=None, MatchLens ghosts)."""
        raw = self._raw_frames(max_frames)
        if self.ball_fill is None:
            yield from raw
            return
        half: list = []
        for item in raw:
            if half and item[1] != half[-1][1]:
                yield from self._fill_half(half)  # never fill across a half boundary
                half = []
            half.append(item)
        yield from self._fill_half(half)

    def _fill_half(self, half):
        import numpy as np

        if not half:
            return
        t = np.array([h[0] for h in half])
        known = np.array([h[3] for h in half])
        x = np.array([h[2]["ball"][1]["map_position"][0] if h[3] else np.nan for h in half])
        y = np.array([h[2]["ball"][1]["map_position"][1] if h[3] else np.nan for h in half])
        if hasattr(self.ball_fill, "set_context"):  # methods that need the players (player-anchored)
            self.ball_fill.set_context([
                {k: (*r["map_position"], r["team"])
                 for grp in ("players", "goalkeepers") for k, r in h[2][grp].items()}
                for h in half])
        fx, fy, filled = self.ball_fill.fill(t, x, y, known)
        for i, (fid, per, tracks, present) in enumerate(half):
            if filled[i]:
                tracks["ball"] = {1: {"map_position": (float(fx[i]), float(fy[i])), "conf": None}}
                self.filled_frames += 1
            yield fid, per, tracks, present

    def _raw_frames(self, max_frames: Optional[int] = None):
        count = 0
        for fid, per, pts, ball in self._frames():
            self.frames_per_period[per] = self.frames_per_period.get(per, 0) + 1
            by_team: Dict[int, Dict[int, dict]] = {1: {}, 2: {}}
            for pid, x, y in pts:
                team = self.team_of.get(pid)
                if team not in (1, 2):
                    continue
                by_team[team][self._int_id[pid]] = {
                    "map_position": to_pitch_m(x, y), "team": team, "conf": 1.0}
            players, keepers = {}, {}
            for team, recs in by_team.items():
                side = self._own_side.get((per, team))
                gk = None
                if recs and side is not None:
                    # own goal at x = side * 52.5 m; deepest player there is the keeper guess
                    gk = max(recs, key=lambda k: side * recs[k]["map_position"][0])
                for k, r in recs.items():
                    if k == gk:
                        r["role"] = "goalkeeper"
                        keepers[k] = r
                    else:
                        players[k] = r
            present = ball is not None
            tracks = {"players": players, "goalkeepers": keepers, "ball": {}}
            if present:
                tracks["ball"] = {1: {"map_position": to_pitch_m(*ball), "conf": 1.0}}
            yield fid, per, tracks, present
            count += 1
            if max_frames and count >= max_frames:
                return


def write_submission(labels: Iterable[str], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("frame,label\n")
        for i, lab in enumerate(labels):
            fh.write(f"{i},{lab}\n")
