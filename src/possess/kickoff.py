"""Kick-off frame of each half from the Smart Tagging "Kick Off (Start)" tags (definition alignment).

Organisers (third_party/mtah/FAQ.md "Which frames are actually graded?"): a half can be graded from
before its kick-off, and every graded frame before the kick-off is `D` (ball not in play; for the
provided match the second half is graded from 110,808 but kicks off at 113,742). Review:
docs/organiser-definitions-review.md item 2.

The tag is a window of a few seconds (start/end in seconds from the start of the video,
video_frame = round(seconds * 25), third_party/mtah/DATA.md "Frame alignment") that contains the
kick-off. A kick-off is taken from the centre spot, so the kick-off frame is the frame inside the
window where the ball is nearest the centre spot (0, 0). No constant is fitted: the window comes
from the tag and the spot from the Laws of the Game. With no ball in the window, the window start
is used.

matchlens_start: MatchLens detectors/set_piece_detector.py classifies kick-offs from player
formation, not from tags; nothing there reads Smart Tagging. Checked set_piece_detector.py,
deadball.py, attacking_direction.py.
"""
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np

FPS = 25.0


def kickoff_windows(xml_path: str, fps: float = FPS) -> List[Tuple[int, int, int]]:
    """[(half, first_video_frame, last_video_frame)] of the "Kick Off (Start)" tags, by half."""
    out = []
    for inst in ET.parse(xml_path).getroot().iter("instance"):
        lab = {}
        for l in inst.findall("label"):
            g, t = l.findtext("group"), l.findtext("text")
            if g and t:
                lab[g.strip().lower()] = t.strip().lower()
        if lab.get("type") != "kick off (start)":
            continue
        m = re.match(r"(\d)", lab.get("half", ""))
        if not m:
            continue
        a = int(round(float(inst.findtext("start")) * fps))
        b = int(round(float(inst.findtext("end")) * fps))
        out.append((int(m.group(1)), a, b))
    return sorted(out)


def kickoff_frames(xml_path: str, ball_x: np.ndarray, ball_y: np.ndarray,
                   fps: float = FPS) -> Dict[int, int]:
    """{half: kick-off video frame}. `ball_x/ball_y` are indexed by frame_id = video frame + 1."""
    res = {}
    for half, a, b in kickoff_windows(xml_path, fps):   # sorted: the earliest tag of a half wins
        if half in res:
            continue
        i0, i1 = max(a + 1, 0), min(b + 2, len(ball_x))
        if i1 <= i0:
            continue
        d = np.hypot(ball_x[i0:i1], ball_y[i0:i1])
        res[half] = a if np.all(np.isnan(d)) else int(np.nanargmin(d)) + i0 - 1
    return res


def pre_kickoff_spans(xml_path: str, ball_x: np.ndarray, ball_y: np.ndarray,
                      halves, fps: float = FPS) -> List[Tuple[int, int]]:
    """[(graded_start, kick_off)] video-frame spans [a, k) that are graded but before the kick-off.

    Only when the half's kick-off tag window starts after the graded start: a half graded from its
    kick-off (FAQ.md: the first half of the provided match) gets nothing, and a tag outside the
    half is ignored (QA L4 ADR-021: a late tag must never turn a whole half into D).
    """
    wins, ko = {}, kickoff_frames(xml_path, ball_x, ball_y, fps)
    for half, a, _ in kickoff_windows(xml_path, fps):
        wins.setdefault(half, a)
    spans = []
    for per, a, b in halves:
        if per not in ko or not a < wins[per] <= b:
            continue
        if a < ko[per] <= b:
            spans.append((a, ko[per]))
    return spans
