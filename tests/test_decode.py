"""Whole-half decoder (decode.py) on synthetic distances (no data needed)."""
import numpy as np

from possess.decode import decode_halves, place_gap_switches, team_emission, viterbi

V2 = dict(hard_dead=1, soft=1, cap=1.5, d0=0.25, c_switch=4.0, c_same=1.0, gap_pos=0.5)


def _case(n=400, turnover=150, hidden=(0, 0), dead=(0, 0)):
    """Home on the ball until `turnover`, away after; the other team 5 m away."""
    d_home = np.where(np.arange(n) < turnover, 0.3, 5.0)
    d_away = np.where(np.arange(n) < turnover, 5.0, 0.3)
    visible = np.ones(n, bool)
    visible[hidden[0]:hidden[1]] = False
    dd = np.zeros(n, bool)
    dd[dead[0]:dead[1]] = True
    return dict(d_home=d_home, d_away=d_away, visible=visible, dead=dd)


def _reference_v1(d_home, d_away, visible, dead, r_pz=1.0, r_dz=1.0, w_ctrl=1.0, w_dead=2.0,
                  c_switch=4.0, c_stop=6.0, c_same=3.0):
    """The first (numpy argmax) implementation, kept here to pin the defaults' behaviour."""
    n = len(d_home)
    h_in, a_in = visible & (d_home <= r_pz), visible & (d_away <= r_pz)
    duel = visible & (d_home <= r_dz) & (d_away <= r_dz)
    hc, ac = h_in & ~a_in & ~duel, a_in & ~h_in & ~duel
    em = np.zeros((n, 4))
    em[hc, 0] += w_ctrl; em[hc, 1] -= w_ctrl
    em[ac, 1] += w_ctrl; em[ac, 0] -= w_ctrl
    dd = np.where(dead, w_dead, -w_dead)
    em[:, 2] += dd; em[:, 3] += dd
    T = np.zeros((4, 4))
    T[0, 1] = T[1, 0] = -c_switch
    T[0, 3] = T[1, 2] = T[2, 3] = T[3, 2] = -np.inf
    T[0, 2] = T[1, 3] = T[2, 1] = T[3, 0] = -c_stop
    T[2, 0] = T[3, 1] = -c_stop - c_same
    score, back = em[0].copy(), np.zeros((n, 4), dtype=np.int8)
    for i in range(1, n):
        cand = score[:, None] + T
        back[i] = np.argmax(cand, axis=0)
        score = cand[back[i], np.arange(4)] + em[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return np.array(["H", "A", "D", "D"], dtype="<U1")[path]


def test_defaults_match_the_first_implementation():
    rng = np.random.default_rng(0)
    n = 3000
    f = dict(d_home=rng.uniform(0, 3, n), d_away=rng.uniform(0, 3, n), visible=rng.random(n) > 0.3,
             dead=np.repeat(rng.random(n // 100) > 0.8, 100))
    for kw in ({}, dict(w_dead=1.0, c_switch=2.0, c_stop=8.0, c_same=1.0)):
        assert (viterbi(**f, **kw) == _reference_v1(**f, **kw)).all()


def test_turnover_is_dated_where_control_changes():
    for kw in ({}, V2):
        lab = "".join(viterbi(**_case(), **kw))
        assert lab == "H" * 150 + "A" * 250


def test_short_touch_by_the_other_team_is_not_a_possession():
    f = _case(turnover=400)
    f["d_away"][200:202] = 0.2          # two frames nearer than the carrier
    f["d_home"][200:202] = 0.6
    assert set(viterbi(**f, **V2)) == {"H"}


def test_soft_emission_grades_distance_and_ignores_far_balls():
    vis = np.ones(4, bool)
    e = team_emission(np.array([0.2, 0.8, 3.0, 9.0]), np.array([5.0, 5.0, 5.0, 2.0]), vis, soft=1, cap=1.5)
    assert e[0] > e[1] > 0            # at the feet beats 0.8 m away
    assert e[2] == 0 and e[3] == 0    # both beyond the cap: no evidence
    assert (team_emission(np.array([0.2]), np.array([5.0]), np.zeros(1, bool), soft=1) == 0).all()


def test_hard_dead_copies_the_detector_and_other_team_restarts():
    f = _case(turnover=400, dead=(100, 200), hidden=(200, 230))
    lab = "".join(viterbi(**f, **V2))
    assert lab[:100] == "H" * 100 and lab[100:200] == "D" * 100
    assert lab[230:] == "H" * 170       # control evidence wins over the restart prior
    f["visible"][:] = False             # no evidence after the stoppage: the other team restarts
    lab = "".join(viterbi(**f, **dict(V2, c_same=1.0)))
    assert lab[100:200] == "D" * 100
    assert {lab[99], lab[200]} == {"H", "A"} and len(set(lab[200:])) == 1


def test_switch_inside_a_hidden_block_goes_to_gap_pos():
    f = _case(turnover=150, hidden=(100, 200))
    for pos, cut in ((0.0, 100), (0.5, 150), (1.0, 200)):
        lab = "".join(viterbi(**f, **dict(V2, gap_pos=pos)))
        assert lab == "H" * cut + "A" * (400 - cut)


def test_place_gap_switches_leaves_stoppages_and_plain_gaps_alone():
    vis = np.ones(12, bool)
    vis[3:9] = False
    lab = np.array(list("HHHDDDAAAAAA"))
    assert "".join(place_gap_switches(lab.copy(), vis, 0.5)) == "HHHDDDAAAAAA"
    lab = np.array(list("HHHHHHHHHHHH"))
    assert "".join(place_gap_switches(lab.copy(), vis, 0.5)) == "HHHHHHHHHHHH"
    lab = np.array(list("HHHAAAAAAAAA"))      # switch at the block's first frame
    assert "".join(place_gap_switches(lab.copy(), vis, 0.5)) == "HHHHHHAAAAAA"
    lab = np.array(list("HHHHHHHHHAAA"))      # switch just after the block
    assert "".join(place_gap_switches(lab.copy(), vis, 0.5)) == "HHHHHHAAAAAA"


def test_decode_halves_marks_unplayed_frames_and_decodes_each_half():
    f = _case(n=300, turnover=100)
    period = np.array([0] * 20 + [1] * 130 + [0] * 20 + [2] * 130)
    lab = "".join(decode_halves(period, feats=f, **V2))
    assert lab[:20] == "X" * 20 and lab[150:170] == "X" * 20
    assert lab[20:100] == "H" * 80 and lab[100:150] == "A" * 50 and lab[170:] == "A" * 130


def test_default_config_is_home_away_symmetric():
    """Fable QA #1: with w_rule > 0, swapping home and away mirrors the labels."""
    import numpy as np
    from possess.decode import viterbi

    rng = np.random.default_rng(1)
    n = 600
    dh = rng.uniform(0.1, 8, n); da = rng.uniform(0.1, 8, n)
    dh[200:260] = 0.2; da[400:460] = 0.2
    vis = np.ones(n, bool); vis[300:350] = False
    dead = np.zeros(n, bool); dead[500:540] = True
    kw = dict(hard_dead=1, soft=1, cap=1.5, d0=0.25, c_switch=4.0, c_same=1.0, gap_pos=0.5, w_rule=0.3)
    a = viterbi(dh, da, vis, dead, **kw)
    b = viterbi(da, dh, vis, dead, **kw)
    swap = {"H": "A", "A": "H", "D": "D", "X": "X"}
    mirrored = np.array([swap[x] for x in b])
    assert (a == mirrored).mean() > 0.99


def test_shape_arrays_without_weights_change_nothing():
    f = _case(hidden=(100, 250))
    base = viterbi(**f, **V2)
    st = np.full(400, -3.0)
    assert (viterbi(**f, shape_team=st, shape_dead=st, **V2) == base).all()


def test_shape_team_evidence_dates_a_switch_inside_a_hidden_block():
    """Ball hidden 100-300, turnover at 150 by distance is invisible; shape says away from 220."""
    f = _case(turnover=150, hidden=(100, 300))
    st = np.zeros(400)
    st[100:220], st[220:300] = 2.0, -2.0
    lab = "".join(viterbi(**f, shape_team=st, w_shape=0.5, **dict(V2, gap_pos=-1.0)))
    assert lab == "H" * 220 + "A" * 180


def test_shape_dead_evidence_finds_a_stoppage_inside_a_hidden_block():
    f = _case(turnover=1000, hidden=(100, 300))             # home throughout, detector never dead
    sd = np.full(400, -2.0)
    sd[150:250] = 3.0
    kw = dict(V2, gap_pos=-1.0, c_stop=6.0)
    lab = "".join(viterbi(**f, shape_dead=sd, w_sdead=1.0, **kw))
    assert lab == "H" * 150 + "D" * 100 + "H" * 150
    # visible frames stay hard: shape cannot create a stoppage there unless sdead_vis > 0
    f2 = _case(turnover=1000)
    assert set(viterbi(**f2, shape_dead=np.full(400, 4.0), w_sdead=1.0, **kw)) == {"H"}
    soft = "".join(viterbi(**f2, shape_dead=sd, w_sdead=1.0, sdead_vis=1.0, w_dead=1.0, **kw))
    assert soft == "H" * 150 + "D" * 100 + "H" * 150


def test_w_live_lets_the_player_model_find_a_stoppage_the_ball_detector_misses():
    """Envelope L2-13: a dead ball that moves is never 'dead' for the ball detector. Its 'live'
    verdict then scores -w_live (0 = no evidence) instead of -w_dead, and the player-only
    stoppage log-odds decide on visible frames."""
    n = 400
    f = dict(d_home=np.full(n, 5.0), d_away=np.full(n, 5.0), visible=np.ones(n, bool), dead=np.zeros(n, bool))
    sd = np.full(n, -4.0)
    sd[150:250] = 4.0
    kw = dict(V2, gap_pos=-1.0, c_stop=12.0, w_sdead=0.3, sdead_vis=1.0, w_dead=1.0)
    old = viterbi(**f, shape_dead=sd, **kw)
    assert set(old) == {"H"}                                 # detector 'live' outvotes the players
    assert (viterbi(**f, shape_dead=sd, w_live=-1.0, **kw) == old).all()   # < 0 = -w_dead (L2-10 behaviour)
    assert (viterbi(**f, shape_dead=sd, w_live=1.0, **kw) == old).all()
    new = "".join(viterbi(**f, shape_dead=sd, w_live=0.0, **kw))
    assert new[150:250] == "D" * 100 and "D" not in new[:150] + new[250:]
    # the detector's 'dead' verdict keeps its weight: it wins against a player model leaning live
    f2 = dict(f, dead=np.arange(n) // 100 == 1)
    lab = "".join(viterbi(**f2, shape_dead=np.full(n, -1.0), w_live=0.0, **dict(kw, w_dead=4.0)))
    assert lab[100:200] == "D" * 100 and "D" not in lab[:100] + lab[200:]


def test_default_decoder_carries_the_l2_13_stoppage_weights():
    from possess import pipeline
    from possess.shape import DEC_SHAPE, DEC_SHAPE_L2_10

    assert pipeline.DEC["w_live"] == 0.0 and pipeline.DEC["sdead_vis"] == 2.5
    assert DEC_SHAPE_L2_10["w_live"] < 0 and DEC_SHAPE_L2_10["sdead_vis"] == 1.0
    from possess.shape import DEC_SHAPE_L2_13

    assert {k: v for k, v in DEC_SHAPE_L2_13.items() if k not in ("w_live", "sdead_vis")} == \
           {k: v for k, v in DEC_SHAPE_L2_10.items() if k not in ("w_live", "sdead_vis")}
    new = ("cap", "duel", "ed_exit", "ed_w", "ed_win", "smooth_touch", "sd_prior", "ctl_s", "reappear", "vtx", "ctl_hi", "ctl_touch", "w_fly", "fly_v0", "fly_v1", "ed_cross", "ed_cross_strict", "ed_xdead", "ed_xdead_min", "w_bt", "w_bd", "ed_still")   # L2-14, L2-15, L2-18 (edge post-pass), L2-20, L2-21, L2-22
    assert {k: v for k, v in DEC_SHAPE.items() if k not in new} == {k: v for k, v in DEC_SHAPE_L2_13.items() if k not in new}
    assert all(DEC_SHAPE_L2_13[k] == pipeline.DEC_BASE[k] for k in ("w_rule", "cap"))   # old configs reproduce
    assert "w_live" not in pipeline.DEC_BASE                 # no-shape decoder unchanged (hard stoppages)


def test_default_decoder_caps_the_distance_emission_at_the_control_zone():
    """Envelope L2-14: with shape evidence the distance emission saturates at the control-zone
    radius; the frame rule stays (w_rule=0 failed the transfer check) and so does the no-shape decoder."""
    from possess import pipeline

    assert pipeline.DEC["cap"] == pipeline.TEAM["r_pz"] == 1.0 and pipeline.DEC["w_rule"] == 0.3
    assert pipeline.DEC_BASE["w_rule"] == 0.3 and pipeline.DEC_BASE["cap"] == 1.5
    # an opponent standing 1.2 m from a ball at a team-mate's feet is no evidence against him
    e = team_emission(np.array([0.2, 0.2]), np.array([1.2, 5.0]), np.ones(2, bool), soft=1, cap=1.0)
    assert e[0] == e[1] > 0
    e15 = team_emission(np.array([0.2, 0.2]), np.array([1.2, 5.0]), np.ones(2, bool), soft=1, cap=1.5)
    assert e15[0] < e15[1]


def test_default_shape_decoder_is_home_away_symmetric():
    """Swapping the teams (distances and the sign of the shape team log-odds) mirrors every label
    with the default shape decoder."""
    from possess import pipeline

    rng = np.random.default_rng(4)
    n = 800
    dh, da = rng.uniform(0.1, 8, n), rng.uniform(0.1, 8, n)
    dh[200:260] = 0.2; da[400:460] = 0.2
    vis = np.ones(n, bool); vis[300:380] = False
    dead = np.zeros(n, bool); dead[500:540] = True
    st = np.cumsum(rng.normal(0, 0.3, n)).clip(-4, 4) + 1e-3
    sd = np.full(n, -2.0); sd[500:560] = 3.0
    from possess.decode import decode_halves                 # pipeline.DEC carries post-pass keys (L2-18)

    per = np.ones(n, dtype=np.int8)
    p90 = rng.uniform(0.5, 5, n).astype(np.float32)
    f = dict(visible=vis, dead=dead, shape_dead=sd, edge_p90=p90)
    a = decode_halves(per, feats=dict(f, d_home=dh, d_away=da, shape_team=st), **pipeline.DEC)
    b = decode_halves(per, feats=dict(f, d_home=da, d_away=dh, shape_team=-st), **pipeline.DEC)
    swap = {"H": "A", "A": "H", "D": "D"}
    assert (a == np.array([swap[x] for x in b])).all() and {"H", "A", "D"} <= set(a)


def test_per_frame_stoppage_costs_off_are_byte_identical_and_cues_move_hidden_edges():
    """Envelope L2-18 (M4 / M6): the per-frame cost arrays default to the constant c_stop; a cue
    on a hidden frame makes leaving D there cheaper, a visible-frame cue changes nothing."""
    from possess import pipeline
    from possess.shape import DEC_SHAPE_L2_16

    rng = np.random.default_rng(7)
    n = 1500
    dh, da = rng.uniform(0.1, 8, n), rng.uniform(0.1, 8, n)
    vis = np.ones(n, bool); vis[400:900] = False
    dead = np.zeros(n, bool)
    st = np.cumsum(rng.normal(0, 0.3, n)).clip(-4, 4) + 1e-3
    sd = np.full(n, -2.0); sd[500:700] = 3.0; sd[700:760] = 0.05      # weak evidence at the end of the stoppage
    kw = dict(pipeline.DEC_BASE, **DEC_SHAPE_L2_16)
    base = viterbi(dh, da, vis, dead, shape_team=st, shape_dead=sd, **kw)
    zero = np.zeros(n)
    assert (viterbi(dh, da, vis, dead, shape_team=st, shape_dead=sd, cue_up=zero, cue_dn=zero, w_r=4.0, w_s=4.0,
                    **kw) == base).all()
    assert (viterbi(dh, da, vis, dead, shape_team=st, shape_dead=sd, c_out_hid=kw["c_stop"], **kw) == base).all()
    cue = zero.copy(); cue[705] = 3.0
    moved = viterbi(dh, da, vis, dead, shape_team=st, shape_dead=sd, cue_up=cue, w_r=4.0, **kw)
    end = lambda lab: int(np.flatnonzero(lab == "D").max()) + 1
    assert end(base) > 705 and end(moved) == 705
    cue_v = zero.copy(); cue_v[100] = 3.0                    # visible frame: ignored
    assert (viterbi(dh, da, vis, dead, shape_team=st, shape_dead=sd, cue_up=cue_v, w_r=4.0, **kw) == base).all()


# ----------------------------------------------------------------------------- touch_hold (L2-8)
def _touch_case(n=220):
    """Home kicks at frame 10 (+x, 10 m/s); the ball flies past an away player (0.5 m, no touch) at
    frame 40; a second away player deflects it at frame 70 (+y, 8 m/s); a home player stops it at 164."""
    from possess.decode import touch_hold  # noqa: F401  (import check)

    f = np.arange(n)
    x = np.where(f < 10, 0.0, np.where(f < 70, 0.4 * (f - 10), 24.0))
    y = np.clip(0.32 * (f - 70), 0.0, 30.0)
    return (x, y) + _touch_d(x, y)


def _touch_d(x, y):
    """Distances from the ball to the nearest home / away player of `_touch_case`."""
    home = np.array([[0.0, 0.0], [24.0, 30.0]])
    away = np.array([[12.0, 0.5], [24.0, 0.0]])
    d = lambda p: np.hypot(x[:, None] - p[None, :, 0], y[:, None] - p[None, :, 1]).min(axis=1)  # noqa: E731
    return d(home), d(away)


def test_touch_hold_follows_touches_not_proximity():
    from possess.decode import touch_hold

    x, y, dh, da = _touch_case()
    h = touch_hold(x, y, dh, da)
    assert h[5] == 0                                   # nobody has touched it yet
    assert (h[12:66] == 1).all()                       # home kick; the fly-by at frame 40 is no touch
    assert (h[73:160] == -1).all()                     # away deflection: away until the next touch
    assert (h[168:200] == 1).all()                     # home stops the ball
    assert (touch_hold(x, y, da, dh) == -h).all()      # swapping the teams flips the sign


def test_touch_hold_expires_and_clears_on_hidden_frames():
    from possess.decode import touch_hold

    x, y, dh, da = _touch_case()
    assert (touch_hold(x, y, dh, da, max_hold_s=1.0)[100:160] == 0).all()   # 1 s after the deflection
    x2, y2 = x.copy(), y.copy()
    x2[80:90] = y2[80:90] = np.nan
    h = touch_hold(x2, y2, dh, da)
    assert (h[80:160] == 0).all() and (h[168:200] == 1).all()


def test_touch_hold_reappear_is_opt_in_and_needs_one_team_and_a_long_gap():
    """Envelope L2-22: the first visible frame after a hidden block >= 1 s is a touch of the only
    team within `reappear` metres; off by default, not after a short gap, not in a duel."""
    from possess.decode import touch_hold

    n = 200
    x, y = np.full(n, 5.0), np.zeros(n)
    dh, da = np.full(n, 0.4), np.full(n, 6.0)            # ball still at a home player's feet
    x[50:90] = y[50:90] = np.nan                         # hidden 1.6 s
    x[150:160] = y[150:160] = np.nan                     # hidden 0.4 s: a tracking hole
    s = slice(50, 190)                                   # the array ends carry the SG7 edge artefact
    assert (touch_hold(x, y, dh, da)[s] == 0).all()      # default: no impulse, no touch
    h = touch_hold(x, y, dh, da, reappear=1.0, max_hold_s=1.0)
    assert (h[50:90] == 0).all() and (h[90:115] == 1).all() and (h[117:190] == 0).all()
    assert (touch_hold(x, y, da, dh, reappear=1.0, max_hold_s=1.0) == -h).all()
    assert (touch_hold(x, y, dh, np.full(n, 0.8), reappear=1.0)[s] == 0).all()   # both teams at the ball


def test_touch_hold_survives_ball_noise():
    """sigma 0.1 m white noise: the thresholds rise with the measured noise, no touch storm."""
    from possess.decode import touch_hold

    x, y, dh, da = _touch_case()
    rng = np.random.default_rng(0)
    h = touch_hold(x + rng.normal(0, 0.1, len(x)), y + rng.normal(0, 0.1, len(x)), dh, da)
    assert (h[20:66] == 1).all() and (h[76:160] == -1).all()


def test_touch_emission_gives_a_one_touch_possession():
    x, y, dh, da = _touch_case()
    f = dict(d_home=dh, d_away=da, visible=np.ones(len(x), bool), dead=np.zeros(len(x), bool))
    base = viterbi(**f, **V2)
    assert set(base[80:150]) == {"H"}                  # distances alone: the deflection is invisible
    lab = viterbi(**f, ball_x=x, ball_y=y, w_touch=0.7, **V2)
    assert set(lab[80:150]) == {"A"} and lab[30] == "H" and lab[190] == "H"
    assert (viterbi(**f, ball_x=x, ball_y=y, **V2) == base).all()           # w_touch = 0: unchanged


def test_touch_hold_ignores_tracking_faults_after_cleaning():
    """Envelope L2-9: a one-frame teleport or a stale repeat next to a player is an impulse, i.e.
    a false touch; after ballclean.clean_ball the hold is the fault-free one."""
    from possess.ballclean import clean_ball
    from possess.decode import touch_hold

    x, y, dh, da = _touch_case()
    good = touch_hold(x, y, dh, da)
    for kind in ("teleport", "stale"):
        fx, fy = x.copy(), y.copy()
        if kind == "teleport":
            fy[39] += 7.0                              # as the ball passes the away player (frame 40)
        else:
            fx[38:43] = fx[37]
        assert (touch_hold(fx, fy, dh, da)[45:66] != 1).any(), kind       # the fault steals the hold
        fh, fa = _touch_d(fx, fy)                      # distances as the tracking would give them
        cx, cy, _ = clean_ball(fx, fy, np.minimum(fh, fa))
        assert (touch_hold(cx, cy, fh, fa)[45:66] == 1).all(), kind
        assert (touch_hold(cx, cy, dh, da) == good).all(), kind


def test_decoder_cleans_the_ball_by_default():
    x, y, dh, da = _touch_case()
    f = dict(d_home=dh, d_away=da, visible=np.ones(len(x), bool), dead=np.zeros(len(x), bool))
    ref = viterbi(**f, ball_x=x, ball_y=y, w_touch=0.7, **V2)
    fy = y.copy()
    fy[39] += 7.0
    fy[110] -= 7.0
    fh, fa = _touch_d(x, fy)
    ff = dict(f, d_home=fh, d_away=fa)
    assert (viterbi(**ff, ball_x=x, ball_y=fy, w_touch=0.7, **V2) == ref).all()
    assert (viterbi(**f, ball_x=x, ball_y=y, w_touch=0.7, ball_clean=0, **V2) == ref).all()   # clean ball: no change


def test_touch_hold_duel_2_keeps_the_holder_through_a_contested_touch():
    """Envelope L2-15: an impulse with both teams within r_touch says nothing about who touched the
    ball. duel=0 drops the holder, duel=1 takes the nearer team, duel=2 leaves the last uncontested
    touch in force (the MatchLens duel rule, as in team_control), its hold still running out."""
    from possess.decode import touch_hold

    x, y, _, _ = _touch_case()
    home = np.array([[0.0, 0.0], [24.0, 30.0], [23.5, 0.4]])      # a home player 0.6 m from the deflection
    away = np.array([[12.0, 0.5], [24.0, 0.0]])
    d = lambda p: np.hypot(x[:, None] - p[None, :, 0], y[:, None] - p[None, :, 1]).min(axis=1)  # noqa: E731
    dh, da = d(home), d(away)
    kw = dict(r_touch=2.0)                             # both teams within reach on every impulse frame
    assert (touch_hold(x, y, dh, da, duel=0, **kw)[75:160] == 0).all()        # holder dropped
    assert (touch_hold(x, y, dh, da, duel=1, **kw)[75:160] != 0).all()        # the nearer team at the last impulse frame
    h = touch_hold(x, y, dh, da, duel=2, **kw)
    assert (h[12:100] == 1).all()                      # home kicked at frame 10 and keeps the ball
    assert (h[120:160] == 0).all()                     # ... until its 4 s hold runs out (not renewed)
    assert (h[168:200] == 1).all()                     # the next uncontested touch counts as before
    assert (touch_hold(x, y, da, dh, duel=2, **kw) == -h).all()
    # no contested touch in the plain case: the three rules agree
    x, y, dh, da = _touch_case()
    assert (touch_hold(x, y, dh, da, duel=2) == touch_hold(x, y, dh, da)).all()


def test_default_decoder_keeps_the_holder_in_a_duel():
    """Envelope L2-15: duel=2 is the default with shape evidence; the older configs spell out
    duel=0, so they still reproduce through pipeline.label(dec_kw=...); DEC_BASE is unchanged."""
    from possess import pipeline
    from possess.shape import DEC_SHAPE_L2_10, DEC_SHAPE_L2_13, DEC_SHAPE_L2_14

    assert pipeline.DEC["duel"] == 2 and "duel" not in pipeline.DEC_BASE
    assert DEC_SHAPE_L2_10["duel"] == DEC_SHAPE_L2_13["duel"] == DEC_SHAPE_L2_14["duel"] == 0
    assert dict(pipeline.DEC, **DEC_SHAPE_L2_14)["duel"] == 0 and DEC_SHAPE_L2_14["cap"] == 1.0
