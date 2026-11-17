"""Player reaction to a hidden turnover (possess/react.py, envelope L5-hidden-team) on synthetic
players: no data needed."""
import numpy as np

from possess import react
from possess.decode import decode_halves, viterbi
from possess.shape import SLOTS

V2 = dict(hard_dead=1, soft=1, cap=1.5, d0=0.25, c_switch=4.0, c_same=1.0, gap_pos=-1.0)
N, T = 500, 250


def _players(n=N, turn=T, v=2.0, fps=25.0):
    """10 players per team, home on the -x side (attacks +x); everybody runs +v until `turn`, -v after."""
    rng = np.random.default_rng(0)
    pos = np.full((n, 2, SLOTS, 2), np.nan, dtype=np.float32)
    t = np.arange(n)
    shift = np.where(t < turn, t, 2 * turn - t) * v / fps
    for team, cx in ((0, -15.0), (1, 15.0)):
        base = np.stack([cx + rng.uniform(-8, 8, 10), rng.uniform(-25, 25, 10)], axis=1)
        pos[:, team, :10, 0] = base[None, :, 0] + shift[:, None]
        pos[:, team, :10, 1] = base[None, :, 1]
    return pos


def _case(n=N, turnover=T, hidden=(100, 400)):
    d_home = np.where(np.arange(n) < turnover, 0.3, 5.0)
    d_away = np.where(np.arange(n) < turnover, 5.0, 0.3)
    visible = np.ones(n, bool)
    visible[hidden[0]:hidden[1]] = False
    return dict(d_home=d_home, d_away=d_away, visible=visible, dead=np.zeros(n, bool))


def test_flow_sign_and_team_swap():
    pos, per = _players(), np.ones(N, np.int8)
    fl = react.flow(pos, per)
    assert abs(np.nanmean(fl[50:200]) - 2.0) < 0.05 and abs(np.nanmean(fl[300:450]) + 2.0) < 0.05
    fl_sw = react.flow(pos[:, ::-1], per)                  # away is now "home" and attacks -x
    np.testing.assert_allclose(fl_sw[50:450], -fl[50:450], atol=1e-5)
    assert np.isnan(react.flow(pos, np.zeros(N, np.int8))).all()   # outside the periods: no flow


def test_flow_ignores_players_who_vanish():
    pos, per = _players(), np.ones(N, np.int8)
    pos[120:180, 0, 0] = np.nan                            # one home player lost for 2.4 s
    fl = react.flow(pos, per)
    assert np.nanmax(np.abs(fl[60:200] - 2.0)) < 0.05       # a centroid would jump here; the flow does not


def test_too_few_players_gives_no_flow():
    pos, per = _players(), np.ones(N, np.int8)
    pos[200:300, 1, 3:] = np.nan                           # 3 away players left
    assert np.isnan(react.flow(pos, per, smooth_s=0.04)[220:280]).all()


def test_reversal_and_discount():
    pos, per = _players(), np.ones(N, np.int8)
    fl = react.flow(pos, per)
    r = react.reversal(fl)
    assert abs(int(np.argmin(r)) - T) <= 3 and r.min() < -3.5
    vis = _case()["visible"]
    to_h, to_a = react.switch_discount(fl, vis, thr=1.0, lag_s=0.0)
    assert to_h.max() == 0.0 and to_a[vis].max() == 0.0     # only towards away, only on hidden frames
    assert abs(int(np.argmax(to_a)) - T) <= 3 and to_a.max() > 2.5
    _, to_a1 = react.switch_discount(fl, vis, thr=1.0, lag_s=1.0)
    assert abs(int(np.argmax(to_a1)) - (T - 25)) <= 3       # the reaction is read 1 s later


def test_place_switches_moves_hidden_switch_only():
    pos, per = _players(), np.ones(N, np.int8)
    fl = react.flow(pos, per)
    vis = _case()["visible"]
    lab = np.array(["H"] * 310 + ["A"] * (N - 310))
    out, moved = react.place_switches(lab, vis, fl, win_s=4.0)
    sw = int(np.flatnonzero(out[1:] != out[:-1])[0]) + 1
    assert moved == 1 and abs(sw - T) <= 3
    out2, _ = react.place_switches(lab, vis, fl, win_s=4.0, lag_s=1.0)
    assert int(np.flatnonzero(out2[1:] != out2[:-1])[0]) + 1 == sw - 25
    lab_v = np.array(["H"] * 450 + ["A"] * (N - 450))      # switch on a visible frame: untouched
    out3, moved3 = react.place_switches(lab_v, vis, fl)
    assert moved3 == 0 and (out3 == lab_v).all()
    out4, moved4 = react.place_switches(lab, vis, np.full(N, np.nan))   # no flow: untouched
    assert moved4 == 0 and (out4 == lab).all()


def test_decoder_off_is_identical_and_keys_are_absorbed():
    case, per = _case(), np.ones(N, np.int8)
    fl = react.flow(_players(), per)
    base = decode_halves(per, feats=dict(case), **V2)
    assert (decode_halves(per, feats=dict(case, flow=fl), **V2) == base).all()
    assert (decode_halves(per, feats=dict(case, flow=fl), **V2, gap_cp=0, w_rev=0.0, rev_thr=2.0, cp_lag=1.0) == base).all()
    assert (decode_halves(per, feats=dict(case), **V2, gap_cp=1, w_rev=2.0) == base).all()   # no flow, no positions: skipped
    z = np.zeros(N)
    assert (viterbi(**case, **V2, sw_h=z + 1.0, sw_a=z + 1.0, w_rev=0.0) == viterbi(**case, **V2)).all()


def test_w_rev_puts_the_hidden_switch_at_the_reversal():
    case, per = _case(), np.ones(N, np.int8)
    fl = react.flow(_players(), per)
    lab = decode_halves(per, feats=dict(case, flow=fl), **V2, w_rev=2.0, rev_lag=0.0)
    sw = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    assert len(sw) == 1 and abs(int(sw[0]) - T) <= 3 and lab[0] == "H" and lab[-1] == "A"
    lab1 = decode_halves(per, feats=dict(case, flow=fl), **V2, w_rev=2.0)      # default reaction lag 1 s
    sw1 = np.flatnonzero(lab1[1:] != lab1[:-1]) + 1
    assert len(sw1) == 1 and abs(int(sw1[0]) - (T - 25)) <= 3
    # a reversal towards HOME gives no discount to a switch towards away: nothing is invented
    lab_h = decode_halves(per, feats=dict(_case(turnover=N), flow=fl), **V2, w_rev=3.9)
    assert (lab_h == "H").all()


def test_gap_cp_in_decoder_and_period_blocks():
    case = _case()
    per = np.ones(N, np.int8)
    fl = react.flow(_players(), per)
    kw = dict(V2, gap_pos=0.8)                             # the decoder leaves the switch at frame 340
    assert int(np.flatnonzero(np.diff((decode_halves(per, feats=dict(case), **kw) == "A").astype(int)))[0]) + 1 == 340
    lab = decode_halves(per, feats=dict(case, flow=fl), **kw, gap_cp=1)
    sw = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    assert len(sw) == 1 and abs(int(sw[0]) - T) <= 3
    pos = _players()
    lab_p = decode_halves(per, feats=dict(case, pos=pos), **kw, gap_cp=1)     # flow computed from the positions
    assert (lab_p == lab).all()
