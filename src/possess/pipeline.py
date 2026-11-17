"""The possession pipeline: frames -> H / A / D per frame.

    python -m possess.pipeline --data data/challenge --out submissions/<team>.csv

Steps (each documented in its own module; loop history in .planning/LOOP.md):
  1. features.compute        nearest home/away player to the ball, team centroid speeds,
                             MatchLens dead-ball state (ported, ghost rule off: ADR-002)
  2. deadball2.detect        ID-free stoppage evidence + MatchLens state, hysteresis (loop 3)
  3. team_control            team-level possession/duel zones with retention (loop 2, ADR-003)
  4. decode                  whole-half Viterbi over H / A / D (ADR-005)
  5. shape                   ball-free team and stoppage evidence from player positions, added to
                             the decoder when the features carry "pos" (plan P5, L2-7)
Frames outside the graded halves are written as X.
"""
import argparse
import glob
import os
from typing import Dict

import numpy as np

from . import features as feat_mod
from .deadball2 import detect, features as dead_features
from .shape import DEC_SHAPE
from .team_control import team_from_distances

# Parameters chosen on the external tune split only (Metrica G1/G2; .planning/LOOP.md).
# still_v / still_far / v_kick: envelope L2-11 (still ball = dead from the frame it stopped; play resumes on the kick)
DEAD2 = dict(r_far=9.0, d_line=2.0, v_slow=1.5, on_frames=50, off_frames=5,   # loop 3
             still_v=0.0, v_kick=0.0)   # L2-11 still/kick rule OFF: bench-artefact gain, does not transfer (Fable QA)
TEAM = dict(r_pz=1.0, r_dz=1.0, k=5, restart_flip=True, backdate=True)        # loop 2 + loop 4 (P1, P2)
# Whole-half Viterbi decoder (plan P3, Fable design, Fable QA, ADR-005). Uses TEAM's rule as one emission.
DECODER = "viterbi"
DEC_BASE = dict(hard_dead=1, soft=1, cap=1.5, d0=0.25, c_switch=4.0, c_same=1.0, gap_pos=0.5, w_rule=0.3, w_touch=0.7)   # w_touch: loop L2-8 (touch_hold, MatchLens impulse port)
# Default: the same decoder plus ball-free shape evidence (shape.py, plan P5, envelope L2-7). Used
# whenever the features carry player positions ("pos"); without them DEC_BASE applies.
DEC = dict(DEC_BASE, **DEC_SHAPE)

N_FRAMES = 199319                                  # rows in a submission (DATA.md)
HALVES = [(1, 11185, 90651), (2, 110808, 190197)]  # graded video frames (DATA.md)


def challenge_period(n: int = N_FRAMES + 1) -> np.ndarray:
    """Period per frame_id for the provided match (0 outside the graded halves)."""
    p = np.zeros(n, dtype=np.int8)
    for per, a, b in HALVES:
        p[a + 1:b + 2] = per
    return p


def label(f: Dict[str, np.ndarray], dead_kw=None, team_kw=None, decoder: str = DECODER, dec_kw=None,
          period: np.ndarray = None, shape_models=None, extra: Dict[str, np.ndarray] = None) -> np.ndarray:
    """Per-frame labels ('H','A','D', 'X' before the first gain) from `features.compute` output.

    decoder="viterbi" (default): whole-half decoding (decode.py, plan P3), parameters DEC updated
    by dec_kw; needs `period` (per frame index, 0 = not played). When `f` has "pos"
    (features.compute(..., positions=True)) the shape evidence is added and DEC applies; without
    it DEC_BASE applies. shape_models: other models than shape.MODELS (the bench passes
    leave-one-match-out fits). decoder="rule": the frame-by-frame team rule (team_control) alone.
    extra: more decoder inputs (per-frame arrays), e.g. the Smart Tagging arrays of smarttags.py
    ("tag_team", "tag_events"); they do nothing unless dec_kw switches their weight on.
    """
    if decoder is None:
        decoder = DECODER
    use_shape = decoder == "viterbi" and f.get("pos") is not None
    dec_kw = dict(DEC if use_shape else DEC_BASE, **(dec_kw or {})) if decoder == "viterbi" else dec_kw
    df = dead_features(f["ball_x"], f["ball_y"], f["d_home"], f["d_away"], f["cen_home"], f["cen_away"])
    dead = detect(df, f["ml_dead"], **dict(DEAD2, **(dead_kw or {})))
    visible = ~np.isnan(f["ball_x"])
    if decoder == "viterbi":
        from .decode import decode_halves

        if period is None:
            period = np.ones(len(dead), dtype=np.int8)
        feats = {"d_home": np.nan_to_num(f["d_home"], nan=99.0), "d_away": np.nan_to_num(f["d_away"], nan=99.0),
                 "visible": visible, "dead": dead, "ball_x": f["ball_x"], "ball_y": f["ball_y"]}
        if use_shape:
            from .shape import decoder_inputs

            feats.update(decoder_inputs(f["pos"], period, shape_models))
            if dec_kw.get("vtx"):                       # touch at the contact point needs the players (L3-A4)
                feats["pos"] = f["pos"]
        feats.update(extra or {})
        return decode_halves(period, feats=feats, **(dec_kw or {}))
    return team_from_distances(f["d_home"], f["d_away"], visible, dead=dead, period=period, **dict(TEAM, **(team_kw or {})))


def _period_of(frame_id: int) -> int:
    row = frame_id - 1                             # ATD frame_id is 1-based (DATA.md)
    for per, a, b in HALVES:
        if a <= row <= b:
            return per
    return 0


def challenge_features(data_dir: str, cache: str = None) -> Dict[str, np.ndarray]:
    """features.compute on the provided match (graded halves only), with player positions ("pos").

    Optional npz cache; the positions live in their own file next to it (positions_v1.npz), so an
    existing feature cache is never rewritten: when only the positions are missing they are
    collected in one extra pass over the tracking file.
    """
    pos_cache = os.path.join(os.path.dirname(cache), "positions_v1.npz") if cache else None
    f = None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        f = {k: z[k] for k in z.files}
        if os.path.exists(pos_cache):
            f["pos"] = np.load(pos_cache)["pos"]
            return f
    from .epts import AtdMatch

    def find(pattern):
        hits = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
        if not hits:
            raise FileNotFoundError("%s not found under %s" % (pattern, data_dir))
        return hits[0]

    match = AtdMatch(find("*tactical_atd.xml"), find("*tactical_atd.txt"), name="BAR-ATH",
                     period_of=_period_of, frame_filter=lambda fid: _period_of(fid) > 0)
    if f is None:
        f = feat_mod.compute(match.iter_frames(), n=N_FRAMES + 1, positions=True)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez_compressed(cache, **{k: v for k, v in f.items() if k != "pos"})
    else:
        from .shape import collect

        f["pos"] = collect(match.iter_frames(), n=N_FRAMES + 1)
    if cache:
        np.savez_compressed(pos_cache, pos=f["pos"])
    return f


def to_rows(lab: np.ndarray) -> np.ndarray:
    """Per-frame_id labels -> the 199,319 submission rows (X outside the graded halves)."""
    rows = np.array(["X"] * N_FRAMES, dtype="<U1")
    for per, a, b in HALVES:
        rows[a:b + 1] = lab[a + 1:b + 2]           # row = frame_id - 1
    return rows


# Smart Tagging evidence (envelope L6-smart-tagging, smarttags.py). OFF by default: NOT YET SCORED
# (the external tag-simulator sweep had not run when this was committed; no weight is chosen;
# .planning/research/smart-tagging.md).
# w_tag / tag_vis: team-phase clips as a team prior (hidden frames / visible scale); trim_s: seconds
# cut from both clip ends; w_tagr / tagr_hold_s: tagged set-piece taker (and the conceding team's
# kick-off after a tagged goal) for the seconds after the restart.
# L11-C (ADR-029, user decision 2026-11-17, cold QA ADOPT as a recorded gate exception): sp=1 sp_hid=1
# sp_take_s=0 - tag-gated set-piece dead ball (setpiece.py): inside a tagged goal kick / corner /
# throw-in window the frames from our last dead evidence to the physical restart are D.
TAGS = dict(w_tag=0.0, tag_vis=0.0, trim_s=0.0, w_tagr=0.0, tagr_hold_s=4.0, sp=1, sp_hid=1, sp_take_s=0)


def challenge_tag_inputs(data_dir: str, n: int, tag_kw: Dict[str, float]):
    """(extra decoder inputs, decoder overrides) for the Smart Tagging flags; ({}, {}) when all are off
    or the tag file / home team name is missing."""
    from . import smarttags as st

    kw = dict(TAGS, **(tag_kw or {}))
    tags = glob.glob(os.path.join(data_dir, "**", "*tactical_smart_tagging.xml"), recursive=True)
    meta = glob.glob(os.path.join(data_dir, "**", "*tactical_atd.xml"), recursive=True)
    home = st.home_team_name(meta[0]) if meta else None
    if not (kw["w_tag"] or kw["w_tagr"]) or not tags or not home:
        return {}, {}
    tg = st.load_tags(tags[0])
    extra, dec = {}, {}
    if kw["w_tag"]:
        extra["tag_team"] = st.phase_prior(tg, n, home, offset=1, trim_s=kw["trim_s"])
        dec.update(w_tag=kw["w_tag"], tag_vis=kw["tag_vis"])
    if kw["w_tagr"]:                               # array index = frame_id = video frame + 1
        extra["tag_events"] = [dict(e, a=e["a"] + 1, b=e["b"] + 1) for e in st.restart_events(tg, home)]
        dec.update(w_tagr=kw["w_tagr"], tagr_hold_s=kw["tagr_hold_s"])
    return extra, dec


def run_challenge(data_dir: str, cache: str = None, tag_kw: Dict[str, float] = None) -> np.ndarray:
    """Labels for all 199,319 submission rows of the provided match (tactical tracking only).
    tag_kw: Smart Tagging flags overriding TAGS (default: set-piece dead-ball rule on, ADR-029)."""
    tag_kw = dict(TAGS, **(tag_kw or {}))
    f = challenge_features(data_dir, cache)
    period = challenge_period(len(f["ball_x"]))
    pre = []                                       # graded frames before a half's kick-off
    tags = glob.glob(os.path.join(data_dir, "**", "*tactical_smart_tagging.xml"), recursive=True)
    if tags:                                       # FAQ.md: graded frames before kick-off are D
        from .kickoff import pre_kickoff_spans

        pre = pre_kickoff_spans(tags[0], f["ball_x"], f["ball_y"], HALVES)
    extra, dec = challenge_tag_inputs(data_dir, len(f["ball_x"]), tag_kw)
    rows = label_with_kickoffs(f, period, pre, dec, extra)
    if tags and tag_kw.get("sp"):                  # L11-C tag-gated set-piece dead ball (ADR-029, default on)
        rows = setpiece_rows(rows, f, period, tags[0], data_dir, tag_kw)
    return rows


def setpiece_rows(rows: np.ndarray, f, period: np.ndarray, xml: str, data_dir: str, tag_kw) -> np.ndarray:
    """setpiece.apply on the submission rows (row = video frame = frame_id - 1). Flags: sp=1 switches
    it on; sp_<key> overrides setpiece.SP (e.g. sp_hid=1, sp_take_s=0)."""
    from . import setpiece, smarttags as st
    meta = glob.glob(os.path.join(data_dir, "**", "*tactical_atd.xml"), recursive=True)
    home = st.home_team_name(meta[0]) if meta else None
    if not home:
        return rows
    evs = [{"a": t["a"], "type": t["type"], "sign": st.team_sign(t, home)}
           for t in st.load_tags(xml) if t["code"] == "SET PIECES" and setpiece.norm_type(t["type"])]
    kw = {k[3:]: v for k, v in tag_kw.items() if k.startswith("sp_")}
    n = len(rows)
    out, _ = setpiece.apply(rows, f["ball_x"][1:n + 1], f["ball_y"][1:n + 1], period[1:n + 1], evs, kw)
    return out


def label_with_kickoffs(f, period: np.ndarray, pre, dec_kw=None, extra=None) -> np.ndarray:
    """Submission rows with the graded pre-kick-off spans [a, k) (video frames) written as D.

    The decoder starts the half at the kick-off (period 0 before it). Note: `period` also feeds the
    match-level hidden-frame prior and the attack-direction votes, so a few frames elsewhere can
    change (QA L4 ADR-021: 5 isolated frames on the provided match).
    """
    period = period.copy()
    for a, k in pre:
        period[a + 1:k + 1] = 0                    # frame_id = video frame + 1
    tag = dict(dec_kw=dec_kw, extra=extra) if dec_kw or extra else {}    # Smart Tagging flags (off: the old call)
    rows = to_rows(label(f, period=period, **tag))
    for a, k in pre:
        rows[a:k] = "D"
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", os.path.join("data", "challenge")))
    ap.add_argument("--out", required=True, help="submission CSV path")
    ap.add_argument("--tags", nargs="*", default=[], help="Smart Tagging flags (pipeline.TAGS, default off), "
                                                         "e.g. w_tag=0.05 tag_vis=0 w_tagr=1.0")
    a = ap.parse_args()
    from .epts import write_submission

    rows = run_challenge(a.data, tag_kw={k: float(v) for k, v in (it.split("=") for it in a.tags)})
    write_submission(rows, a.out)
    print("wrote %s (%d rows)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
