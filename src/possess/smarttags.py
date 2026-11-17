"""Smart Tagging as decoder input (envelope L6-smart-tagging; opt-in, every flag off by default).

The organisers list the Smart Tagging file among the allowed INPUTS (docs/RULES.md "Using as input
the video, the automatic tracking data, the smart tagging ..."). It is a list of clips
(`instance`: start / end in seconds of the video, a `code`, labels Team / Type / Half ...):

  team-phase codes   in possession: PROGRESSION, BUILD UP, CREATING CHANCES, ATTACKING TRANSITION,
                     BALL IN FINAL THIRD, BALL IN THE BOX, PLAYERS IN THE BOX, LONG BALL;
                     out of possession: DEFENDING IN MIDDLE / DEFENSIVE / ATTACKING THIRD,
                     DEFENSIVE TRANSITION. The Team label is the team the phase belongs to, so an
                     out-of-possession clip of team T is evidence for the OTHER team.
  SET PIECES         Type = Corner Kick / Goal Kick / Free Kick / Throw-in / Kick Off ...; the Team
                     label (when not N/A) is the team taking the restart.
  GOALS              Team = the scoring team; the next kick-off belongs to the other team (Law 8).

Clips are padded (several codes have one fixed length), so a clip edge is NOT a possession edge and
one clip can hold several possessions: the tags are used as soft evidence only.

Three mechanisms (decode.viterbi / decode_halves; all need the arrays built here in `feats`):
  w_tag      `tag_team` (per frame, in [-1, 1]: + home, - away) added to the team emission,
             w_tag * tag_team on hidden-ball frames and w_tag * tag_vis * tag_team on visible ones.
  w_tagr     `tag_restart` (per frame, +1 / -1 for `restart_hold_s` after a tagged restart of a
             team, 0 elsewhere) added to the team emission on every frame: the restart taker is the
             clear possessor after a stoppage (CHALLENGE.md), the tag names him.
  goals      folded into `tag_restart`: after a GOALS clip of team T the next kick-off tag (or,
             without one, nothing) is a restart of the other team.

Seconds -> video frame round(s * 25); ATD frame_id = video frame + 1 (third_party/mtah/DATA.md,
kickoff.py).

matchlens_start: nothing in MatchLens reads Smart Tagging or any tag prior. Checked
footballtactics backend/events/possession.py, impulse.py, deadball.py, attacking_direction.py,
detectors/set_piece_detector.py (restart type and taker from geometry: ball still at a corner arc /
goal area / touch line and the nearest player; no external tags). The set-piece taker idea is the
same one, with the tag in place of the geometry.
"""
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np

FPS = 25.0
IN_POSS = ("PROGRESSION", "BUILD UP", "CREATING CHANCES", "ATTACKING TRANSITION", "BALL IN FINAL THIRD",
           "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL")
OUT_POSS = ("DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN ATTACKING THIRD",
            "DEFENSIVE TRANSITION")


def load_tags(xml_path: str, fps: float = FPS) -> List[dict]:
    """[{code, a, b (video frames, inclusive), team (label text or None), type, half, labels}]."""
    out = []
    for inst in ET.parse(xml_path).getroot().iter("instance"):
        lab = {}
        for l in inst.findall("label"):
            g, t = l.findtext("group"), l.findtext("text")
            if g and t:
                lab[g.strip().lower()] = t.strip()
        team = lab.get("team")
        m = re.match(r"(\d)", lab.get("half", ""))
        out.append({"code": (inst.findtext("code") or "").strip().upper(),
                    "a": int(round(float(inst.findtext("start")) * fps)),
                    "b": int(round(float(inst.findtext("end")) * fps)),
                    "team": None if team is None or team.upper() in ("N/A", "NA", "") else team,
                    "type": lab.get("type", ""), "half": int(m.group(1)) if m else 0, "labels": lab})
    return sorted(out, key=lambda t: (t["a"], t["b"]))


def home_team_name(atd_xml: str) -> Optional[str]:
    """Name of the local (home) team from the ATD metadata: Score@idLocalTeam -> Teams/Team/Name."""
    root = ET.parse(atd_xml).getroot()
    score = next(iter(root.iter("Score")), None)
    if score is None:
        return None
    for t in root.iter("Team"):
        if t.get("id") == score.get("idLocalTeam"):
            return (t.findtext("Name") or "").strip() or None
    return None


def team_sign(tag: dict, home: str) -> int:
    """+1 when the tag's team is `home`, -1 for the other team, 0 without a team."""
    if tag["team"] is None:
        return 0
    return 1 if tag["team"].strip().lower() == home.strip().lower() else -1


def phase_sign(tag: dict, home: str) -> int:
    """Team in possession a phase clip speaks for: +1 home, -1 away, 0 = not a team-phase clip."""
    s = team_sign(tag, home)
    if tag["code"] in IN_POSS:
        return s
    if tag["code"] in OUT_POSS:
        return -s
    return 0


def phase_prior(tags: List[dict], n: int, home: str, offset: int = 1, trim_s: float = 0.0,
                fps: float = FPS) -> np.ndarray:
    """Per-frame team evidence in [-1, 1] from the team-phase clips (index = video frame + offset).

    Every clip speaks for its team in possession on its frames (minus `trim_s` seconds cut from
    both ends: clip padding); frames where clips of both teams overlap, and frames without a clip,
    carry 0 (prior_from_clips).
    """
    return prior_from_clips([(t["a"], t["b"], phase_sign(t, home)) for t in tags], n, offset, trim_s, fps)


def prior_from_clips(clips, n: int, offset: int = 0, trim_s: float = 0.0, fps: float = FPS) -> np.ndarray:
    """`phase_prior` for clips given as (first frame, last frame, sign of the team in possession).

    Clips that mirror each other (an in-possession clip of one team and the out-of-possession clip
    of the other on the same window) count once: a frame's value is the sign of the teams present,
    +1 / -1 when only one team is named there, 0 when both or none are.
    """
    pos, neg = np.zeros(n, bool), np.zeros(n, bool)
    k = int(round(trim_s * fps))
    for a, b, s in clips:
        a, b = a + offset + k, b + offset + 1 - k
        if s == 0 or b <= a:
            continue
        (pos if s > 0 else neg)[max(a, 0):min(b, n)] = True
    return pos.astype(float) - neg.astype(float)


def restart_events(tags: List[dict], home: str) -> List[dict]:
    """Tagged restarts with a known taker: [{a, b, sign, type}] (video frames).

    SET PIECES clips with a team; plus kick-offs after a GOALS clip: the first SET PIECES clip of a
    kick-off type that starts after the goal clip starts gets the conceding team (Law 8) when it
    has no team of its own.
    """
    out = []
    used = set()
    for t in tags:
        if t["code"] == "SET PIECES" and team_sign(t, home):
            out.append({"a": t["a"], "b": t["b"], "sign": team_sign(t, home), "type": t["type"]})
    for g in tags:
        if g["code"] != "GOALS" or not team_sign(g, home):
            continue
        for i, t in enumerate(tags):
            if t["code"] == "SET PIECES" and t["type"].lower().startswith("kick off") and t["a"] >= g["a"] \
                    and t["half"] == g["half"] and i not in used and not team_sign(t, home):
                used.add(i)
                out.append({"a": t["a"], "b": t["b"], "sign": -team_sign(g, home), "type": "Kick Off (After Goal)"})
                break
    return sorted(out, key=lambda e: e["a"])


def restart_frame(ev: dict, dead: np.ndarray, offset: int = 1) -> Optional[int]:
    """Index of the restart inside a tagged clip: the first live frame after the LAST dead run that
    starts in the clip (`dead`: per-index bool, e.g. the first-pass labels == 'D'). None when the
    clip holds no dead frame: the clip is then not anchored and gives no evidence.
    """
    a, b = max(ev["a"] + offset, 0), min(ev["b"] + offset + 1, len(dead))
    if b <= a:
        return None
    d = np.flatnonzero(dead[a:b])
    if len(d) == 0:
        return None
    j = a + int(d[-1]) + 1
    while j < len(dead) and dead[j]:                   # the stoppage may end after the clip
        j += 1
    return j if j < len(dead) else None


def restart_frame_start(ev: dict, dead: np.ndarray, offset: int = 1, lo_s: float = 2.0, hi_s: float = 15.0,
                        fps: float = FPS) -> Optional[int]:
    """Envelope L8 anchor: the first D -> live transition from `lo_s` seconds before the clip START to
    `hi_s` seconds after it. A SET PIECES clip opens a few seconds before its restart (on the provided
    input our stoppage exits sit 5-7 s after the clip start, autonomous-L8.md row 17), so the restart is
    the first one after the clip opens; `restart_frame` (the last stoppage that starts anywhere in the
    35 s clip) picks a later stoppage in most clips. None when no stoppage ends in the window."""
    a = ev["a"] + offset
    lo, hi = max(a - int(lo_s * fps), 1), min(a + int(hi_s * fps), len(dead) - 1)
    if hi <= lo:
        return None
    j = np.flatnonzero(dead[lo - 1:hi] & ~dead[lo:hi + 1])
    return lo + int(j[0]) if len(j) else None


def restart_prior(events: List[dict], dead: np.ndarray, hold_s: float = 4.0, offset: int = 1,
                  max_after_s: float = 15.0, fps: float = FPS, anchor: int = 0) -> np.ndarray:
    """Per-frame +1 / -1 for `hold_s` seconds from each tagged restart (restart_frame), 0 elsewhere.

    A restart found more than `max_after_s` seconds after the clip's end is dropped (the stoppage
    that started in the clip is a long one: injury, substitution; the clip is still the right
    stoppage, but an unbounded search would attach a tag to anything)."""
    out = np.zeros(len(dead))
    for ev in events:
        if anchor:                                     # L8: first restart after the clip opens
            j = restart_frame_start(ev, dead, offset, fps=fps)
            if j is None:
                continue
        else:
            j = restart_frame(ev, dead, offset)
            if j is None or j > ev["b"] + offset + max_after_s * fps:
                continue
        out[j:j + int(round(hold_s * fps))] = ev["sign"]
    return out


def challenge_inputs(xml_path: str, n: int, home: str, dead: np.ndarray = None, trim_s: float = 0.0,
                     hold_s: float = 4.0) -> Dict[str, np.ndarray]:
    """{"tag_team", "tag_restart"} for the provided match (index = frame_id). `dead` (first-pass
    labels == 'D') is needed for tag_restart only."""
    tags = load_tags(xml_path)
    out = {"tag_team": phase_prior(tags, n, home, trim_s=trim_s)}
    if dead is not None:
        out["tag_restart"] = restart_prior(restart_events(tags, home), dead, hold_s)
    return out
