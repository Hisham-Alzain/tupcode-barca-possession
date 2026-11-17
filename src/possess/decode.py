"""Whole-match decoding of H / A / D (research plan P3).

Instead of committing frame by frame, find the single best label sequence for a whole half:
the Viterbi path of a 3-state model whose per-frame scores come from the existing rules.

  Emissions (log-scores, additive):
    v1 (soft=0)  ball visible, one team alone in its control zone -> that team +w_ctrl, other -w_ctrl;
                 duel / nobody in a zone / ball hidden -> 0 (no evidence)
    v2 (soft=1)  every visible frame: e = w_ctrl * (g(d_away) - g(d_home)), g(d) = ln(1 + min(d, cap) / d0);
                 H +e, A -e. The nearer team scores, in proportion to how much nearer it is, and a
                 ball at a player's feet counts far more than a ball 1 m away (measured on the tune
                 split: log-odds 3.0 for < 0.5 m vs 0.5-1 m on Metrica, yet the zone rule treats
                 both alike; envelope L2-6-decoder-fable.out.md).
    w_rule > 0   the frame rule's own label (team_control, pipeline defaults) as one more
                 emission: its team +w_rule, the other -w_rule, on visible live frames.
    stoppages    soft (hard_dead=0): detector says dead -> D +w_dead, else D -w_dead.
                 hard (hard_dead=1): D exactly where the detector says dead, as in the frame rule.
  Transitions (costs subtracted):
    H <-> A         c_switch
    H/A -> D        c_stop
    D -> H/A        c_stop, plus c_same when the restart team equals the team before the stoppage
                    (the other team restarts after 78-90% of stoppages on external truth)
    shape        (w_shape, w_sdead > 0; arrays from possess.shape.emissions) on frames with the
                 ball hidden: H +w_shape * shape_team, A the negative; and, with hard_dead=1, the
                 held detector state is replaced by D +w_sdead/2 * shape_dead, teams the negative.
                 shape_vis scales the team term on visible frames (0 = hidden frames only).
                 sdead_vis > 0 also makes visible-frame stoppages soft: D-vs-live log-odds
                 w_dead * (+1 detector dead, -1 live) + sdead_vis * w_sdead * shape_dead.
                 w_live >= 0 (envelope L2-13): the detector's "live" verdict scores -w_live
                 instead of -w_dead. The detector needs the ball far from every player or
                 outside the pitch, so it misses a dead ball that is carried or thrown back to
                 the spot (D recall on visible frames 0.4-0.6 on the realistic bench): its
                 "dead" is reliable (precision 0.72-0.98), its "live" is weak evidence, and
                 the player-only stoppage model (which never sees the ball) decides there.
    touch        (w_touch > 0, needs ball_x / ball_y in the features) on visible frames: the team
                 of the last touch (`touch_hold`: ball impulse within r_touch of a player) scores
                 +w_touch, the other -w_touch, for up to max_hold_s after the touch. Envelope
                 L2-8: 49% of true possessions are shorter than 4 s (one-touch deflections,
                 tackles, clearances) and the distance emission sees them for a few frames only.
                 ball_clean=1 (default): the touch detector reads the ball after
                 ballclean.clean_ball (teleports, stale repeats, off-pitch readings removed,
                 short holes bridged; envelope L2-9). Distances and `visible` are not changed.
  Ball hidden: no team evidence (unless shape emissions are given), so a switch inside a hidden block can sit anywhere in it at the
    same score. gap_pos in [0, 1] places it (0 = block start, 0.5 = middle, 1 = end);
    gap_pos < 0 keeps the Viterbi tie-break (v1 behaviour).

Sources: sticky HMM (Fox et al. 2011), PathCRF (2026), Li & Link (2026) — see
docs/research/literature/B_segmentation/. Zone constants and the frame rule: team_control.py
(MatchLens possession.py, Vidal-Codina §2.3). The "team before the stoppage" is tracked in an
extended state (D_from_H, D_from_A), so the model has 4 states internally: H, A, DH, DA.
All parameters tuned on the external tune split only.
"""
import numpy as np

H, A, DH, DA = 0, 1, 2, 3
NEG = -1e9
# Opt-in evidence hook for the local viewer (scripts/viewer_evidence.py). None (default) = off and
# nothing is collected. Set to {} before a run: `viterbi` then appends (offset, {term: H score per
# frame, "stop": D minus the best team score}) to DEBUG["parts"] and `decode_halves` stores the
# Viterbi labels before its post-passes in DEBUG["vit"]. Labels are the same either way.
DEBUG = None


def team_emission(d_home: np.ndarray, d_away: np.ndarray, visible: np.ndarray, r_pz: float = 1.0,
                  r_dz: float = 1.0, w_ctrl: float = 1.0, soft: int = 0, d0: float = 0.25,
                  cap: float = 6.0) -> np.ndarray:
    """Per-frame score for H (the score for A is its negative). 0 where the ball is hidden."""
    if soft:
        g_h = np.log1p(np.minimum(d_home, cap) / d0)
        g_a = np.log1p(np.minimum(d_away, cap) / d0)
        return np.where(visible, w_ctrl * (g_a - g_h), 0.0)
    h_in = visible & (d_home <= r_pz)
    a_in = visible & (d_away <= r_pz)
    duel = visible & (d_home <= r_dz) & (d_away <= r_dz)
    e = np.zeros(len(d_home))
    e[h_in & ~a_in & ~duel] = w_ctrl
    e[a_in & ~h_in & ~duel] = -w_ctrl
    return e


SG7 = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0   # Savitzky-Golay, window 7, order 2


def ball_noise(ball_x: np.ndarray, ball_y: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """White position noise of the ball (m per axis), from the residual to its SG7-smoothed track
    on moving-ball frames: median |residual| of a 2-D Gaussian = 1.177 sd, and SG7 leaves
    sqrt(2/3) = 0.816 of white noise in the residual. 0 when there is nothing to measure."""
    r = np.hypot(ball_x - xs, ball_y - ys)
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.median(r) / (1.177 * 0.816)) if len(r) else 0.0


def ball_impulse(ball_x: np.ndarray, ball_y: np.ndarray, cos_thr: float = 0.85, dv_thr: float = 1.5,
                 lag: int = 2, v_min: float = 1.0, fps: float = 25.0, noise_k: float = 3.0) -> dict:
    """Ball impulse per frame (the kinematic half of `touch_hold`, see there for the source):
    {"impulse": bool, "vix", "viy": incoming velocity over `lag` frames (m/s), "vox", "voy": outgoing}."""
    n = len(ball_x)
    xs, ys = np.convolve(ball_x, SG7, "same"), np.convolve(ball_y, SG7, "same")
    xs, ys = np.where(np.isnan(xs), ball_x, xs), np.where(np.isnan(ys), ball_y, ys)
    vix, viy = np.full(n, np.nan), np.full(n, np.nan)
    vox, voy = np.full(n, np.nan), np.full(n, np.nan)
    k = fps / lag
    vix[lag:], viy[lag:] = (xs[lag:] - xs[:-lag]) * k, (ys[lag:] - ys[:-lag]) * k
    vox[:-lag], voy[:-lag] = vix[lag:], viy[lag:]
    si, so = np.hypot(vix, viy), np.hypot(vox, voy)
    sd_v = ball_noise(ball_x, ball_y, xs, ys) * 0.816 * k      # velocity noise (m/s per axis) after SG7
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vix * vox + viy * voy) / (si * so)
        impulse = ((cos < cos_thr) & (np.fmax(si, so) > v_min) & (np.fmin(si, so) > noise_k * 2 * sd_v)) |                   (np.abs(so - si) > max(dv_thr, noise_k * 1.414 * sd_v))
    return {"impulse": impulse, "vix": vix, "viy": viy, "vox": vox, "voy": voy}


# Smoothed ball track (envelope L2-20). A touch changes the ball's velocity within one or two
# frames; a provider that low-passes the track spreads it over ~0.5 s, and the 2-frame impulse test
# of `touch_hold` (MatchLens impulse.py thresholds) then fires on the hardest kicks only. Whether a
# track is low-passed is read from the track itself, no labels: the 99th percentile of the frame-
# to-frame acceleration. An unsmoothed 25 fps ball has 204-619 m/s2 there per half (all 5 external matches,
# every bench variant), a Gaussian low-pass of 3 / 5 / 7 frames 100 / 55 / 38 m/s2; SMOOTH_ACC sits
# between (6 m/s of velocity change per frame). On a low-passed track `touch_hold` runs with
# SMOOTH_TOUCH: velocities over 8 frames (0.32 s) on each side, one touch per velocity-change peak
# (`peak`), and a 2 m touch radius, because smoothing also pulls the ball off the toucher's feet.
# Chosen on the external tune matches under the bench variant `real2` (scripts/bench.py; ball
# smoothed to the provided match's INPUT statistics): 0.4303 -> 0.4833, lag 6 / 8 / 10 the same
# within 0.003; tracks that are not low-passed keep the old detector and byte-identical labels.
SMOOTH_ACC = 150.0
# L3-A4 QA: the vertex touch (vtx) gains only on HEAVILY low-passed tracks. External matches smoothed
# with Gaussian sigma 7 / 5 / 3 frames: gain +0.029 / +0.008 (n.s.) / -0.031; their acceleration
# p99 per half is <= 41 / <= 57 / >= 89. VTX_ACC sits in the 57-89 gap, so vtx stays on at sigma
# 5-7 and switches off at sigma 3 (labels there equal vtx=0). Chosen on external data only.
VTX_ACC = 70.0
SMOOTH_TOUCH = dict(lag=8, peak=1, r_touch=2.0)


def ball_smoothed(ball_x: np.ndarray, ball_y: np.ndarray, fps: float = 25.0, acc_thr: float = SMOOTH_ACC) -> bool:
    """True when the visible ball track is low-passed: 99th percentile of the frame-to-frame
    acceleration (m/s2) below `acc_thr`. False when fewer than 250 accelerations can be measured."""
    a = np.hypot(np.diff(ball_x, 2), np.diff(ball_y, 2)) * fps * fps
    a = a[np.isfinite(a)]
    return bool(len(a) >= 250 and np.percentile(a, 99) < acc_thr)


def touch_vertex(ball_x: np.ndarray, ball_y: np.ndarray, lag: int, fps: float = 25.0):
    """Contact point of a touch centred on each frame i (envelope L3-A4): the ball is taken to fly
    straight in, from x[i-lag] with the velocity of the `lag` frames before it, and straight out, to
    x[i+lag] with the velocity of the `lag` frames after it, and the flight time between the two
    anchors (2 lag / fps) is conserved; the contact time t1 solves that in least squares (clipped to
    the window). A low-passed track rounds a deflection off and pulls the ball metres away from the
    toucher's feet; the vertex puts it back. Returns (x, y, contact frame), nan where not measurable.
    Source: Vidal-Codina et al. §2.3.3 (a gain = a change of the incoming vs outgoing direction /
    speed over the whole control interval, not one frame; "if the ball positional data has been
    smoothed, the speed may be computed using finite differences")."""
    n, L = len(ball_x), lag
    P = np.stack([ball_x, ball_y], 1)
    b0, b1 = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    vin, vout = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    if n > 2 * L:
        b0[L:], b1[:-L] = P[:-L], P[L:]
        vin[2 * L:] = (P[L:-L] - P[:-2 * L]) * fps / L
        vout[:-2 * L] = (P[2 * L:] - P[L:-L]) * fps / L
    T = 2.0 * L / fps
    dv = vin - vout
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = np.clip((dv * (b1 - b0 - vout * T)).sum(1) / (dv * dv).sum(1), 0.0, T)
    v = b0 + vin * t1[:, None]
    k = np.clip(np.arange(n) - L + np.round(np.nan_to_num(t1) * fps).astype(int), 0, max(n - 1, 0))
    return v[:, 0], v[:, 1], k


def vertex_team_dist(pos: np.ndarray, vx: np.ndarray, vy: np.ndarray, k: np.ndarray, spread: int = 1):
    """(d_home, d_away): distance of each team's nearest player to the point (vx, vy), minimum over
    the frames k-spread .. k+spread; 99 where the point or the team is missing. pos [n, 2, slots, 2]."""
    n = len(vx)
    px, py = np.nan_to_num(vx, nan=1e6), np.nan_to_num(vy, nan=1e6)
    out = np.full((n, 2), 99.0)
    for s in range(-spread, spread + 1):
        kk = np.clip(k + s, 0, n - 1)
        with np.errstate(invalid="ignore"):
            d = np.hypot(pos[kk, :, :, 0] - px[:, None, None], pos[kk, :, :, 1] - py[:, None, None])
        out = np.fmin(out, np.nanmin(np.where(np.isnan(d), 99.0, d), axis=2))
    out[~np.isfinite(vx)] = 99.0
    return out[:, 0], out[:, 1]


def rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    """Minimum of x over frames i-w .. i+w (NaN ignored where a finite value exists)."""
    out = np.asarray(x, float).copy()
    for s in range(1, w + 1):
        out[s:] = np.fmin(out[s:], x[:-s])
        out[:-s] = np.fmin(out[:-s], x[s:])
    return out


def touch_hold(ball_x: np.ndarray, ball_y: np.ndarray, d_home: np.ndarray, d_away: np.ndarray,
               r_touch: float = 1.0, max_hold_s: float = 4.0, cos_thr: float = 0.85,
               dv_thr: float = 1.5, lag: int = 2, v_min: float = 1.0, duel: int = 0,
               fps: float = 25.0, noise_k: float = 3.0, peak: int = 0, reappear: float = 0.0,
               reappear_gap_s: float = 1.0, tw: int = 0, lose: float = 0.0, lose_mode: int = 1,
               dt_home: np.ndarray = None, dt_away: np.ndarray = None, confirm_s: float = 0.0) -> np.ndarray:
    """Per-frame "team of the last touch": +1 home, -1 away, 0 unknown (envelope L2-8).

    A touch is a ball impulse next to a player: the ball's direction or speed changes at a frame
    where a player stands within `r_touch`. The team of that player then holds the ball until the
    next touch, a hidden frame, or `max_hold_s` (a loose ball stays with the last team, as in the
    external truth: a one-touch deflection or tackle is a possession of its own although the ball
    is near that team for only a few frames, which the distance emission cannot see).

    matchlens_start: footballtactics events/impulse.py:impulse_verdict + kin_from_series and
    events/possession.py PossessionConfig (direction_change_cos_threshold 0.85,
    speed_change_delta_mps 1.5, Savitzky-Golay window 7 / order 2 on the ball positions; Vidal-
    Codina §2.3.3). Changed: team level, no player identity (nearest player of either team);
    incoming / outgoing velocity over `lag` frames instead of one; offline (centred).
    Both teams within r_touch: duel=0 -> unknown (the hold is cleared), duel=1 -> the nearer team,
    duel=2 -> the touch is ignored and the team of the last uncontested touch keeps the ball, its
    hold running on (envelope L2-15). This is the duel rule of MatchLens possession.py (Vidal-
    Codina 2.3 duel zone: possession is not reassigned while both are at the ball), already ported
    in team_control.team_from_distances ("duel keeps the current team"); the touch hold was the
    one place that dropped the holder instead. Who touched the ball in a duel cannot be read from
    two distances: on the realistic bench the nearer team is right on 0.77-0.84 of duel touches
    against 0.97-0.98 of uncontested ones.
    Noisy ball: the thresholds rise with the ball's own position noise (`ball_noise`, estimated
    from the data, no labels): speed leg max(dv_thr, noise_k sigma of a speed difference),
    direction leg only when both speeds exceed 2 noise_k sigma of a velocity component.
    reappear > 0 (envelope L2-22, opt-in): the first visible frame after a hidden block of at least
    `reappear_gap_s` counts as a touch when exactly one team has a player within `reappear` metres
    of the ball. The impulse test is blind there (it needs `lag` visible frames on both sides), yet
    a ball that comes back at a player's feet is that player's: restart taker or carrier. Radius =
    the control zone of team_control (MatchLens possession.py r_pz, Vidal-Codina 2.3).
    confirm_s > 0 (envelope L4-control-definition, opt-in): an uncontested touch by the team that
    does NOT hold the ball is ignored when the next uncontested touch, within `confirm_s`, is by
    the holder again (a deflection that returns: no change, the holder's hold runs on). A touch
    followed by a touch of its own team, or by no touch for `confirm_s`, takes the ball from its
    own frame (a kept interception starts at the first touch). Organisers' definition: "a simple
    touch of the ball without control" is not a change of possession.
    """
    n = len(ball_x)
    out = np.zeros(n)
    if n < 2 * lag + 7:
        return out
    imp = ball_impulse(ball_x, ball_y, cos_thr, dv_thr, lag, v_min, fps, noise_k)
    impulse = imp["impulse"]
    dh_t, da_t = (rolling_min(d_home, tw), rolling_min(d_away, tw)) if tw > 0 else (d_home, d_away)
    if dt_home is not None:                         # touch test on other distances (vertex, L3-A4)
        dh_t, da_t = dt_home, dt_away
    with np.errstate(invalid="ignore"):
        h_in, a_in = dh_t <= r_touch, da_t <= r_touch
        team = np.where(h_in & ~a_in, 1.0, np.where(a_in & ~h_in, -1.0, 0.0))
        if duel == 1:
            team = np.where(h_in & a_in, np.sign(da_t - dh_t), team)
    touch = impulse & (h_in | a_in)
    if peak:
        # a smoothed ball spreads one touch over many frames: keep only the frame where the velocity
        # change (outgoing minus incoming, over `lag` frames each) is largest within +-lag frames
        dv = np.nan_to_num(np.hypot(imp["vox"] - imp["vix"], imp["voy"] - imp["viy"]), nan=-1.0)
        top = dv.copy()
        for s in range(1, lag + 1):
            top[s:] = np.maximum(top[s:], dv[:-s])
            top[:-s] = np.maximum(top[:-s], dv[s:])
        touch &= dv >= top
    keep = duel == 2
    vis = ~np.isnan(ball_x)
    revert = np.zeros(n, bool)                          # touch answered by the other team within confirm_s
    if confirm_s > 0:
        idx = np.flatnonzero(touch & (team != 0.0) & vis)
        if len(idx) > 1:
            # one touch fires on a few neighbouring frames: frames of one team at most 2 lag + 1 apart are one touch
            new = np.concatenate([[True], (team[idx[1:]] != team[idx[:-1]]) | (np.diff(idx) > 2 * lag + 1)])
            cid = np.cumsum(new) - 1
            first, last = idx[new], idx[np.concatenate([new[1:], [True]])]
            t_c = team[first]
            rev = np.zeros(len(first), bool)
            rev[:-1] = (t_c[1:] == -t_c[:-1]) & (first[1:] - last[:-1] <= int(confirm_s * fps))
            revert[idx] = rev[cid]
    cur, left = 0.0, 0
    max_hold = int(max_hold_s * fps)
    if reappear > 0:
        with np.errstate(invalid="ignore"):
            h_r, a_r = d_home <= reappear, d_away <= reappear
        team_r = np.where(h_r & ~a_r, 1.0, np.where(a_r & ~h_r, -1.0, 0.0))
    hid_run, min_gap = 0, int(reappear_gap_s * fps)
    if lose > 0:
        with np.errstate(invalid="ignore"):
            h_l, a_l = d_home <= lose, d_away <= lose
        team_l = np.where(h_l & ~a_l, 1.0, np.where(a_l & ~h_l, -1.0, 0.0))
    for i in range(n):
        if not vis[i]:
            cur, left = 0.0, 0
            hid_run += 1
            continue
        back, hid_run = hid_run >= min_gap, 0
        if touch[i] and not (keep and team[i] == 0.0) and not (revert[i] and cur == -team[i]):
            # duel=2: a contested touch changes nothing; confirm_s: nor does a touch that returns
            cur, left = team[i], max_hold
        elif reappear > 0 and back and team_r[i] != 0.0:  # ball back at one team's feet (L2-22)
            cur, left = team_r[i], max_hold
        elif lose > 0 and cur != 0.0 and team_l[i] == -cur:   # ball in the opponent's zone alone (L3-A4)
            cur, left = (-cur, max_hold) if lose_mode == 2 else (0.0, 0)
        elif left > 0:
            left -= 1
        else:
            cur = 0.0
        out[i] = cur
    return out


def place_gap_switches(lab: np.ndarray, visible: np.ndarray, gap_pos: float) -> np.ndarray:
    """Move every H<->A switch that falls in a hidden block to `gap_pos` of that block.

    Only blocks whose labels are one team, then the other (no D, one switch) are touched; the
    switch may sit at the block's first frame or just after its last one.
    """
    n = len(lab)
    i = 0
    while i < n:
        if visible[i]:
            i += 1
            continue
        j = i
        while j < n and not visible[j]:
            j += 1
        lo, hi = max(i - 1, 0), min(j, n - 1)          # last visible frame before, first after
        a, b = lab[lo], lab[hi]
        seg = lab[lo:hi + 1]
        if a != b and a in "HA" and b in "HA" and np.isin(seg, [a, b]).all() \
                and (seg[1:] != seg[:-1]).sum() == 1:
            p = i + int(np.floor(gap_pos * (j - i) + 0.5))   # round half up (QA #5)
            lab[i:p] = a
            lab[p:j] = b
        i = j
    return lab


def viterbi(d_home: np.ndarray, d_away: np.ndarray, visible: np.ndarray, dead: np.ndarray,
            r_pz: float = 1.0, r_dz: float = 1.0, w_ctrl: float = 1.0, w_dead: float = 2.0,
            c_switch: float = 4.0, c_stop: float = 6.0, c_same: float = 3.0,
            soft: int = 0, d0: float = 0.25, cap: float = 6.0, w_rule: float = 0.0,
            hard_dead: int = 0, gap_pos: float = -1.0, shape_team: np.ndarray = None,
            shape_dead: np.ndarray = None, w_shape: float = 0.0, w_sdead: float = 0.0,
            shape_vis: float = 0.0, sdead_vis: float = 0.0, ball_x: np.ndarray = None,
            ball_y: np.ndarray = None, w_touch: float = 0.0, r_touch: float = 1.0,
            max_hold_s: float = 4.0, duel: int = 0, lag: int = 2, ball_clean: int = 1,
            fps: float = 25.0, w_live: float = -1.0, c_out_hid: float = -1.0, peak: int = 0,
            smooth_touch: int = 0, cue_up: np.ndarray = None, cue_dn: np.ndarray = None, w_r: float = 0.0,
            w_s: float = 0.0, reappear: float = 0.0, tw: int = 0, lose: float = 0.0,
            lose_mode: int = 1, pos: np.ndarray = None, vtx: float = 0.0, vtx_raw: int = 0,
            confirm_s: float = 0.0, tag_team: np.ndarray = None, w_tag: float = 0.0,
            tag_vis: float = 0.0, tag_restart: np.ndarray = None, w_tagr: float = 0.0,
            rule_flip: int = -1, tag_neutral: int = 0, fly: np.ndarray = None, w_fly: float = 0.0,
            bt: np.ndarray = None, w_bt: float = 0.0, sw_h: np.ndarray = None, sw_a: np.ndarray = None,
            w_rev: float = 0.0) -> np.ndarray:
    """Best 'H'/'A'/'D' sequence (numpy array of 1-char strings). Defaults reproduce v1.

    w_rev > 0 with sw_h / sw_a (envelope L5-hidden-team, react.switch_discount, ported from agent/a7 47ebbe8
    in L8 round 7; off by default, labels byte-identical): the switch A -> H at frame i costs
    c_switch - w_rev * sw_h[i] and H -> A costs c_switch - w_rev * sw_a[i], floored at 0 (cue arrays >= 0,
    0 on visible frames).

    Smart Tagging evidence (envelope L6, smarttags.py; off by default, labels byte-identical):
    w_tag * tag_team is added to the team emission on hidden-ball frames (times tag_vis on visible
    ones); w_tagr * tag_restart on every frame (the tagged restart taker, for a few seconds).

    Per-frame stoppage costs on hidden frames (envelope L2-18, all off by default = the constant
    c_stop, labels byte-identical): c_out_hid >= 0 replaces the cost of LEAVING D at a hidden
    frame (control M6); w_r * cue_up lowers the cost of leaving D and w_s * cue_dn the cost of
    entering D at a hidden frame (M4: an event cue on the transition; cue arrays >= 0), never
    below 0.

    Tie-breaking favours H (lowest state index). With soft=1, visible frames where both teams are
    beyond `cap` carry zero evidence, so without w_rule > 0 those stretches lean home (Fable QA
    2026-11-13: swapping home/away changed 5-8% of frames). Use w_rule > 0 with soft=1; the
    pipeline default does, and tests/test_decode.py checks home/away symmetry for it.
    """
    n = len(d_home)
    if n == 0:                                          # QA #3
        return np.array([], dtype="<U1")
    visible = np.asarray(visible, bool)
    dead = np.asarray(dead, bool)
    e = team_emission(d_home, d_away, visible, r_pz, r_dz, w_ctrl, soft, d0, cap)
    dbg = {"dist": e} if DEBUG is not None else None
    if w_rule:
        from .team_control import team_from_distances

        from .pipeline import TEAM                      # same frame rule as the pipeline default (QA #4)

        # rule_flip (envelope L8, opt-in): 0 / 1 overrides TEAM's restart_flip in the frame-rule emission; -1 = TEAM
        rule = team_from_distances(d_home, d_away, visible, dead=dead,
                                   **(TEAM if rule_flip < 0 else dict(TEAM, restart_flip=bool(rule_flip))))
        t = w_rule * visible * ((rule == "H").astype(float) - (rule == "A"))
        e = e + t
        if dbg is not None:
            dbg["rule"] = t
    if w_touch and ball_x is not None:                  # team of the last touch (touch_hold, L2-8)
        smoothed = smooth_touch and ball_smoothed(ball_x, ball_y, fps)
        if smoothed:                                    # low-passed ball track (envelope L2-20)
            lag, peak, r_touch = SMOOTH_TOUCH["lag"], SMOOTH_TOUCH["peak"], SMOOTH_TOUCH["r_touch"]
        # NOTE (L2-20 QA): in the smoothed regime the 2 m r_touch above is also the keep radius of
        # clean_ball below; that is part of the scored configuration (real2 0.4833).
        if ball_clean:                                 # teleports / stale repeats / off-pitch out (L2-9)
            from .ballclean import clean_ball

            ball_x, ball_y, _ = clean_ball(ball_x, ball_y, np.minimum(d_home, d_away), r_touch, fps)
        dt_h = dt_a = None
        r_t = r_touch
        heavy = smoothed and ball_smoothed(ball_x, ball_y, fps, VTX_ACC)
        if vtx > 0 and pos is not None and (heavy or vtx_raw):       # toucher at the contact point (L3-A4)
            dt_h, dt_a = vertex_team_dist(pos, *touch_vertex(ball_x, ball_y, lag, fps))
            r_t = vtx
        t = w_touch * visible * touch_hold(ball_x, ball_y, d_home, d_away, r_t, max_hold_s,
                                           duel=duel, lag=lag, fps=fps, peak=peak, reappear=reappear, tw=tw,
                                           lose=lose, lose_mode=lose_mode, dt_home=dt_h, dt_away=dt_a,
                                           confirm_s=confirm_s)
        e = e + t
        if dbg is not None:
            dbg["touch"] = t
    if w_shape and shape_team is not None:              # ball-free team evidence (shape.py, plan P5)
        t = w_shape * np.where(visible, shape_vis, 1.0) * shape_team
        e = e + t
        if dbg is not None:
            dbg["shape"] = t
    if w_fly and fly is not None:                       # flight direction (envelope L8 round 2, flight_dir; opt-in)
        t = w_fly * np.nan_to_num(fly)
        e = e + t
        if dbg is not None:
            dbg["fly"] = t
    tag_abs = np.zeros(n)                               # |tag evidence| per frame (tag_neutral, L8)
    if w_tag and tag_team is not None:                  # Smart Tagging team-phase clips (L6)
        t = w_tag * np.where(visible, tag_vis, 1.0) * np.nan_to_num(tag_team)
        e, tag_abs = e + t, tag_abs + np.abs(t)
        if dbg is not None:
            dbg["tag"] = t
    if w_tagr and tag_restart is not None:              # Smart Tagging restart taker (L6)
        t = w_tagr * np.nan_to_num(tag_restart)
        e, tag_abs = e + t, tag_abs + np.abs(t)
        if dbg is not None:
            dbg["tagr"] = t
    bt_abs = None
    if w_bt and bt is not None:                         # learned ball-aware team evidence (L8 round 5, ballteam.py; opt-in)
        t = w_bt * np.nan_to_num(bt)
        e, bt_abs = e + t, np.abs(t)
        if dbg is not None:
            dbg["bt"] = t
    if hard_dead:
        e_team = np.where(dead, NEG, e)
        e_teamA = np.where(dead, NEG, -e)
        e_d = np.where(dead, 0.0, NEG)
        if w_sdead and shape_dead is not None:
            # ball hidden: the detector only holds its last state there, so replace it by the
            # ball-free stoppage log-odds (soft; c_stop prices entering / leaving D)
            half = 0.5 * w_sdead * shape_dead
            hid = ~visible
            e_team = np.where(hid, e - half, e_team)
            e_teamA = np.where(hid, -e - half, e_teamA)
            e_d = np.where(hid, half, e_d)
            if sdead_vis:
                # ball visible: detector and shape both soft, D-vs-live log-odds
                # w_dead * (+-1) + sdead_vis * w_sdead * shape_dead (replaces the hard copy)
                hv = 0.5 * (np.where(dead, w_dead, -(w_dead if w_live < 0 else w_live))
                            + sdead_vis * w_sdead * shape_dead)
                e_team = np.where(hid, e_team, e - hv)
                e_teamA = np.where(hid, e_teamA, -e - hv)
                e_d = np.where(hid, e_d, hv)
    else:
        e_team, e_teamA = e, -e
        e_d = np.where(dead, w_dead, -w_dead)
    if tag_neutral and tag_abs.any():
        # envelope L8 (opt-in): the tags say WHICH team, not that the ball is live. Adding the tag
        # term to the team score also lifts the named team against D (measured: every w_tag config
        # lost, even with error-free simulated tags). Give D the same amount, so the named team
        # keeps its score against D and only the other team is pushed down (by twice the term).
        e_d = np.where(e_d > NEG / 2, e_d + tag_abs, e_d)
    if bt_abs is not None:
        # the ball-team model is fitted on live frames only: it names a team, it does not say the
        # ball is live. D gets the same amount (as tag_neutral), so only the other team is pushed down.
        e_d = np.where(e_d > NEG / 2, e_d + bt_abs, e_d)
    if dbg is not None:
        dbg["stop"] = e_d - np.maximum(e_team, e_teamA)
        DEBUG.setdefault("parts", []).append((DEBUG.get("_off", 0), dbg))
    eh, ea, ed = e_team.tolist(), e_teamA.tolist(), e_d.tolist()

    # cost of leaving D (c_out) and of entering D (c_in) at frame i; constants unless L2-18 options
    c_out, c_in = np.full(n, float(c_stop)), np.full(n, float(c_stop))
    if c_out_hid >= 0:
        c_out = np.where(visible, c_out, float(c_out_hid))
    if w_r and cue_up is not None:
        c_out = np.maximum(0.0, c_out - w_r * np.where(visible, 0.0, np.nan_to_num(cue_up)))
    if w_s and cue_dn is not None:
        c_in = np.maximum(0.0, c_in - w_s * np.where(visible, 0.0, np.nan_to_num(cue_dn)))
    co, ci = c_out.tolist(), c_in.tolist()
    c_ah, c_ha = np.full(n, float(c_switch)), np.full(n, float(c_switch))    # switch to H / to A at frame i
    if w_rev and sw_h is not None and sw_a is not None:
        # floor 0; the 1e-3 term only breaks ties inside a saturated stretch towards the cue's peak
        c_ah = np.maximum(0.0, c_ah - w_rev * np.nan_to_num(sw_h)) - 1e-3 * np.nan_to_num(sw_h)
        c_ha = np.maximum(0.0, c_ha - w_rev * np.nan_to_num(sw_a)) - 1e-3 * np.nan_to_num(sw_a)
    c_ah, c_ha = c_ah.tolist(), c_ha.tolist()
    sh, sa, sdh, sda = eh[0], ea[0], ed[0], ed[0]
    back = np.zeros((n, 4), dtype=np.int8)
    bk = back.tolist()
    for i in range(1, n):
        c_other, c_stop = co[i], ci[i]
        c_back = c_other + c_same
        # ties go to the lowest state index (H, A, DH, DA), as np.argmax did in v1
        bh, vh = H, sh
        v = sa - c_ah[i]
        if v > vh:
            bh, vh = A, v
        v = sdh - c_back
        if v > vh:
            bh, vh = DH, v
        v = sda - c_other
        if v > vh:
            bh, vh = DA, v
        ba, va = H, sh - c_ha[i]
        if sa > va:
            ba, va = A, sa
        v = sdh - c_other
        if v > va:
            ba, va = DH, v
        v = sda - c_back
        if v > va:
            ba, va = DA, v
        v = sh - c_stop
        if v >= sdh:
            bdh, vdh = H, v
        else:
            bdh, vdh = DH, sdh
        v = sa - c_stop
        if v >= sda:
            bda, vda = A, v
        else:
            bda, vda = DA, sda
        m = max(vh, va)                                  # keep the scores bounded
        sh, sa, sdh, sda = vh - m + eh[i], va - m + ea[i], vdh - m + ed[i], vda - m + ed[i]
        bk[i] = (bh, ba, bdh, bda)
    path = np.empty(n, dtype=np.int8)
    s = int(np.argmax([sh, sa, sdh, sda]))
    path[-1] = s
    for i in range(n - 1, 0, -1):
        s = bk[i][s]
        path[i - 1] = s
    lab = np.array(["H", "A", "D", "D"], dtype="<U1")[path]
    if gap_pos >= 0:
        lab = place_gap_switches(lab, visible, gap_pos)
    return lab


def decode_halves(period: np.ndarray, **kw) -> np.ndarray:
    """Run viterbi separately on each contiguous block of the same period (> 0); 'X' elsewhere."""
    from . import cleanup as cl
    from . import edges as ed

    feats = dict(kw.pop("feats"))
    ckw = {k: kw.pop(k) for k in cl.KEYS if k in kw}     # segment clean-up post-pass (envelope L2-17)
    # stoppage-edge post-pass (envelope L2-18, edges.py): runs on the finished labels, before the
    # clean-up rules. feats["edge_p90"] = raw fast-player speed (shape.decoder_inputs); without it
    # the snap is skipped (a decoder without player positions has no edge evidence).
    ekw = {k: kw.pop(k) for k in ed.KEYS if k in kw}
    # controlled-possession post-pass (envelope L4-control-definition, ctl_truth.relabel; opt-in, 0 = off):
    # a predicted team spell shorter than ctl_s seconds that follows a live spell of the other team
    # goes back to that team (deflection that returns = no change; a kept interception keeps its start).
    ctl_s = kw.pop("ctl_s", 0)
    # pass proxy (envelope L8, opt-in, 0 = off): after the ctl_s pass, a team spell shorter than
    # ctl_hi seconds that follows the other team's live spell is kept only when it holds at least
    # ctl_touch distinct uncontested touches of its own team (receive + release = the organisers'
    # "controlled": a completed action, not one touch); otherwise it goes back (ctl_truth.relabel).
    ctl_hi, ctl_touch = kw.pop("ctl_hi", 0), int(kw.pop("ctl_touch", 2))
    # stoppage entry at the line crossing (envelope L8 round 3, edges.cross_entry; opt-in, 0 = off):
    # ed_cross = search window in seconds around our own entry, ed_cross_off = frames after the crossing.
    ed_cross, ed_cross_off = float(kw.pop("ed_cross", 0)), int(kw.pop("ed_cross_off", 5))
    ed_cross_strict = int(kw.pop("ed_cross_strict", 0))
    # stoppage at a line crossing with none of ours nearby (edges.cross_dead; opt-in, 0 = off), margin in metres
    ed_xdead, ed_xdead_m = int(kw.pop("ed_xdead", 0)), float(kw.pop("ed_xdead_m", 0.5))
    ed_xdead_min = float(kw.pop("ed_xdead_min", 0.0))      # seconds; shorter new stoppages are not added
    ed_still = float(kw.pop("ed_still", 0))                # ball at rest and alone for 3 s = dead (edges.still_dead), radius in m
    # out of play until the restart kick (envelope L11-A, edges.restart_dead; opt-in, 0 = off): ed_rkick = kick speed m/s
    ed_rkick, ed_rk_line = float(kw.pop("ed_rkick", 0)), float(kw.pop("ed_rk_line", 0))
    ed_rk_reapp = int(kw.pop("ed_rk_reapp", 0))            # 1: also a hidden crossing (ball back in view beyond a line)
    ed_rk_law = int(kw.pop("ed_rk_law", 0))                # 1: restart spot by the line left (goal area / corner vs touchline)
    ctl_hid = kw.pop("ctl_hid", 0)       # 1: a spell with any hidden-ball frame is exempt (no touch can be seen there)
    # player reaction to a hidden turnover (envelope L5-hidden-team, react.py, T89; ported from agent/a7 47ebbe8
    # in L8 round 7; opt-in, 0 = off): w_rev > 0 makes a hidden H<->A switch cheaper where the all-player flow
    # reverses towards the team that takes the ball (rev_thr m/s; rev_lag s = reaction lag). The flow is
    # feats["flow"] or computed from feats["pos"]; without either it is skipped.
    w_rev, rev_thr, rev_lag = kw.pop("w_rev", 0.0), kw.pop("rev_thr", 1.0), kw.pop("rev_lag", 1.0)
    gap_cp, cp_win, cp_lag = kw.pop("gap_cp", 0), kw.pop("cp_win", 4.0), kw.pop("cp_lag", 0.0)
    flow = feats.pop("flow", None)
    if (gap_cp or w_rev) and flow is None and feats.get("pos") is not None:
        from .react import flow as _flow

        flow = _flow(feats["pos"], np.asarray(period), kw.get("fps", 25.0))
    if w_rev and flow is not None:
        from .react import switch_discount

        feats["sw_h"], feats["sw_a"] = switch_discount(flow, feats["visible"], kw.get("fps", 25.0), thr=rev_thr, lag_s=rev_lag)
        kw["w_rev"] = w_rev
    p90 = feats.pop("edge_p90", None)
    # learned ball-aware stoppage evidence on visible frames (envelope L8 round 6, balldead.py; opt-in, w_bd = 0
    # = off). Added to the stoppage log-odds the decoder already weighs on visible frames (sdead_vis * w_sdead *
    # shape_dead), in units of that product, before the hidden-frame prior shift (which reads hidden frames only).
    # feats["bd"] given (the bench passes leave-one-match-out evidence) is used as it is; else balldead.MODEL.
    w_bd, bd = float(kw.pop("w_bd", 0)), feats.pop("bd", None)
    unit_bd = float(kw.get("sdead_vis", 0) or 0) * float(kw.get("w_sdead", 0) or 0)
    if w_bd and unit_bd > 0 and feats.get("shape_dead") is not None:
        if bd is None and feats.get("pos") is not None and feats.get("ball_x") is not None:
            from . import balldead

            if balldead.MODEL:
                bd = balldead.from_feats(feats, p90, fps=kw.get("fps", 25.0))
        if bd is not None:
            feats["shape_dead"] = np.nan_to_num(np.asarray(feats["shape_dead"], float)) + w_bd * np.nan_to_num(bd) / unit_bd
    # hidden-frame prior of the stoppage log-odds (envelope L2-21, shape.prior_shift): estimated once
    # per match from the hidden frames of all its periods, no labels. Needs the shape arrays.
    sd_prior, ok, pi = kw.pop("sd_prior", 0), feats.pop("shape_ok", None), feats.pop("sdead_pi", None)
    if sd_prior and ok is not None and pi is not None and feats.get("shape_dead") is not None:
        from .shape import prior_shift

        feats["shape_dead"] = prior_shift(feats["shape_dead"], np.asarray(ok, bool),
                                          ~np.asarray(feats["visible"], bool) & (np.asarray(period) > 0), float(pi))
    # Smart Tagging restart taker (envelope L6, smarttags.restart_prior): the restart of a tagged
    # set piece is the end of the last stoppage that starts inside its clip, so the stoppages of a
    # first decoding pass (without the term) anchor the clip; feats["tag_events"] = [{a, b, sign}]
    # in array indices. Without w_tagr or events nothing runs twice.
    tag_ev = feats.pop("tag_events", None)
    if kw.get("w_tagr", 0) and tag_ev is not None and len(tag_ev):
        from .smarttags import restart_prior

        first = decode_halves(period, feats=dict(feats, **({} if p90 is None else {"edge_p90": p90})),
                              **dict({k: v for k, v in kw.items() if k != "tagr_anchor"}, w_tagr=0.0, **ekw))
        feats["tag_restart"] = restart_prior(tag_ev, first == "D", float(kw.pop("tagr_hold_s", 4.0)), offset=0,
                                             fps=kw.get("fps", 25.0), anchor=int(kw.get("tagr_anchor", 0)))
    kw.pop("tagr_hold_s", None)
    kw.pop("tagr_anchor", None)                         # L8: 1 = anchor on the clip start (smarttags.restart_frame_start)
    # flight direction (envelope L8 round 2, opt-in, w_fly = 0 = off): see flight_dir
    fly_v0, fly_v1 = float(kw.pop("fly_v0", 8.0)), float(kw.pop("fly_v1", 16.0))
    if kw.get("w_fly", 0) and feats.get("pos") is not None and feats.get("ball_x") is not None:
        feats["fly"] = flight_dir(feats["ball_x"], feats["ball_y"], feats["pos"], period, fly_v0, fly_v1, kw.get("fps", 25.0))
    # learned ball-aware team evidence (envelope L8 round 5, ballteam.py; opt-in, w_bt = 0 = off). feats["bt"]
    # given (the bench passes leave-one-match-out evidence) is used as it is; else ballteam.MODEL.
    if kw.get("w_bt", 0) and feats.get("bt") is None and feats.get("pos") is not None and feats.get("ball_x") is not None:
        from . import ballteam

        if ballteam.MODEL:
            feats["bt"] = ballteam.from_feats(feats, period, fps=kw.get("fps", 25.0))
    protect = cl.active(ckw) and ckw.get("cl_protect", 0) and feats.get("ball_x") is not None
    n = len(period)
    out = np.array(["X"] * n, dtype="<U1")
    touch = np.zeros(n) if protect else None
    i = 0
    while i < n:
        if period[i] <= 0:
            i += 1
            continue
        j = i
        while j < n and period[j] == period[i]:
            j += 1
        sl = slice(i, j)
        if DEBUG is not None:
            DEBUG["_off"] = i
        out[sl] = viterbi(**{k: v[sl] for k, v in feats.items()}, **kw)
        if gap_cp and flow is not None:                  # T88 (rejected; kept so that react.py stays whole)
            from .react import place_switches

            out[sl] = place_switches(out[sl], feats["visible"][sl], flow[sl], kw.get("fps", 25.0), cp_win, cp_lag)[0]
        if protect:
            touch[sl] = touch_frames(feats["ball_x"][sl], feats["ball_y"][sl], feats["d_home"][sl],
                                     feats["d_away"][sl], kw.get("r_touch", 1.0), kw.get("lag", 2),
                                     kw.get("ball_clean", 1), kw.get("fps", 25.0))
        i = j
    if DEBUG is not None:
        DEBUG["vit"] = out.copy()
    if ed.active(ekw):
        out = ed.apply(out, period, feats["visible"], p90, fps=kw.get("fps", 25.0), **ekw)
    if cl.active(ckw):
        out = cl.cleanup(out, period, feats["visible"], touch, fps=kw.get("fps", 25.0), **ckw)
    if ctl_s:
        from .ctl_truth import relabel

        out = relabel(out, period, float(ctl_s), kw.get("fps", 25.0))
    if ctl_hi and feats.get("ball_x") is not None:
        from .ctl_truth import relabel

        ev = np.zeros(n)
        i = 0
        while i < n:
            if period[i] <= 0:
                i += 1
                continue
            j = i
            while j < n and period[j] == period[i]:
                j += 1
            ev[i:j] = touch_events(**{k: feats[k][i:j] for k in ("ball_x", "ball_y", "d_home", "d_away", "visible")},
                                   pos=feats["pos"][i:j] if feats.get("pos") is not None else None,
                                   **{k: kw[k] for k in ("r_touch", "lag", "ball_clean", "fps", "smooth_touch", "vtx", "vtx_raw", "duel") if k in kw})
            i = j
        out = relabel(out, period, float(ctl_hi), kw.get("fps", 25.0), passes=ev, min_pass=ctl_touch,
                      free=~np.asarray(feats["visible"], bool) if ctl_hid else None)
    if ed_cross > 0 and feats.get("ball_x") is not None:
        out = ed.cross_entry(out, period, feats["ball_x"], feats["ball_y"], ed_cross, ed_cross_off, kw.get("fps", 25.0),
                             strict=ed_cross_strict)
    if ed_xdead and feats.get("ball_x") is not None:
        out = ed.cross_dead(out, period, feats["ball_x"], feats["ball_y"], margin=ed_xdead_m, fps=kw.get("fps", 25.0),
                            min_s=ed_xdead_min)
    if ed_still > 0 and feats.get("ball_x") is not None:     # L8 round 4 (edges.still_dead; opt-in, 0 = off): ed_still = alone radius in m
        out = ed.still_dead(out, period, feats["ball_x"], feats["ball_y"], np.minimum(feats["d_home"], feats["d_away"]),
                            alone=ed_still, fps=kw.get("fps", 25.0))
    if ed_rkick > 0 and feats.get("ball_x") is not None:     # L11-A (edges.restart_dead; opt-in, 0 = off)
        out = ed.restart_dead(out, period, feats["ball_x"], feats["ball_y"], v_k=ed_rkick, line=ed_rk_line, reapp=bool(ed_rk_reapp), law=bool(ed_rk_law),
                              fps=kw.get("fps", 25.0))
    return out


def flight_dir(ball_x: np.ndarray, ball_y: np.ndarray, pos: np.ndarray, period: np.ndarray,
               v0: float = 8.0, v1: float = 16.0, fps: float = 25.0, lag: int = 8, v_max: float = 40.0) -> np.ndarray:
    """Per-frame team evidence from the ball's flight (envelope L8 round 2): in [-1, 1], positive =
    home. A ball that travels fast along the pitch was played by the team attacking that way
    (long ball, clearance, through pass, shot): on the external tune matches (bench real2, ctl T=2
    truth; scripts/a8b_kick.py flight) that team is the true team on 0.87 of the frames with a
    speed of 12 m/s or more mostly along x and on 0.93 at 16 m/s, against 0.88-0.90 for the
    decoder's own label there, and the two err on different frames. The value is the sign of the
    ball's x velocity times home's attacking direction (shape.attack_dir), scaled by a ramp of the
    x speed from 0 at `v0` to 1 at `v1` m/s; velocity centred over `lag` frames (the smoothed-touch
    lag, SMOOTH_TOUCH); 0 where the ball is hidden at either end, above `v_max` (a teleport) or
    the attacking direction is unknown.
    matchlens_start: footballtactics events/attacking_direction.py (side a team defends from the
    mean x of its players) -> shape.attack_dir; events/kinematics.py finite-difference ball velocity.
    Nothing in MatchLens uses the flight direction as possession evidence (possession.py, impulse.py checked)."""
    from .shape import attack_dir

    n, h = len(ball_x), lag // 2
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    if n > lag:
        vx[h:n - lag + h] = (ball_x[lag:] - ball_x[:-lag]) * fps / lag
        vy[h:n - lag + h] = (ball_y[lag:] - ball_y[:-lag]) * fps / lag
    ad = attack_dir(pos, np.asarray(period)).astype(float)
    with np.errstate(invalid="ignore"):
        ramp = np.clip((np.abs(vx) - v0) / max(v1 - v0, 1e-9), 0.0, 1.0)
        out = np.where(np.hypot(vx, vy) <= v_max, np.sign(vx) * ad * ramp, 0.0)
    return np.nan_to_num(out)


def touch_events(ball_x: np.ndarray, ball_y: np.ndarray, d_home: np.ndarray, d_away: np.ndarray,
                 visible: np.ndarray = None, pos: np.ndarray = None, r_touch: float = 1.0, lag: int = 2,
                 ball_clean: int = 1, fps: float = 25.0, smooth_touch: int = 0, vtx: float = 0.0,
                 vtx_raw: int = 0, duel: int = 0) -> np.ndarray:
    """+1 / -1 on the FIRST frame of each distinct uncontested home / away touch, 0 elsewhere
    (envelope L8, for the ctl_hi pass proxy). Same ball preparation as the touch emission in
    `viterbi` (smoothed-track parameters, cleaning, vertex attribution); a touch that fires on
    neighbouring frames (within 2 lag + 1 frames, same team) is one touch."""
    peak = 0
    smoothed = smooth_touch and ball_smoothed(ball_x, ball_y, fps)
    if smoothed:
        lag, peak, r_touch = SMOOTH_TOUCH["lag"], SMOOTH_TOUCH["peak"], SMOOTH_TOUCH["r_touch"]
    if ball_clean:
        from .ballclean import clean_ball

        ball_x, ball_y, _ = clean_ball(ball_x, ball_y, np.minimum(d_home, d_away), r_touch, fps)
    dt_h = dt_a = None
    r_t = r_touch
    heavy = smoothed and ball_smoothed(ball_x, ball_y, fps, VTX_ACC)
    if vtx > 0 and pos is not None and (heavy or vtx_raw):
        dt_h, dt_a = vertex_team_dist(pos, *touch_vertex(ball_x, ball_y, lag, fps))
        r_t = vtx
    t = touch_hold(ball_x, ball_y, d_home, d_away, r_t, 0.0, duel=0, lag=lag, fps=fps, peak=peak,
                   dt_home=dt_h, dt_away=dt_a)
    out = np.zeros(len(t))
    last = {1.0: -10**9, -1.0: -10**9}
    for i in np.flatnonzero(t != 0):
        if i - last[t[i]] > 2 * lag + 1:
            out[i] = t[i]
        last[t[i]] = i
    return out


def touch_frames(ball_x: np.ndarray, ball_y: np.ndarray, d_home: np.ndarray, d_away: np.ndarray,
                 r_touch: float = 1.0, lag: int = 2, ball_clean: int = 1, fps: float = 25.0) -> np.ndarray:
    """+1 / -1 on the frames of an uncontested home / away touch, 0 elsewhere: `touch_hold` with
    no hold, on the ball the decoder's touch emission reads (cleaned when ball_clean). Used by
    cleanup.cleanup(cl_protect=1)."""
    if ball_clean:
        from .ballclean import clean_ball

        ball_x, ball_y, _ = clean_ball(ball_x, ball_y, np.minimum(d_home, d_away), r_touch, fps)
    return touch_hold(ball_x, ball_y, d_home, d_away, r_touch, 0.0, duel=0, lag=lag, fps=fps)
