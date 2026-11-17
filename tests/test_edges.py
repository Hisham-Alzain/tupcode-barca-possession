"""possess/edges.py: the stoppage-edge post-pass (envelope L2-18-stop-edges-fable)."""
import os
import sys

import numpy as np

from possess import edges, pipeline
from possess.decode import decode_halves
from possess.shape import DEC_SHAPE, DEC_SHAPE_L2_16

FPS = 25


def _case(step_at=420, n=1000):
    """H 0-200, D 200-400 (predicted exit at 400), A after; ball hidden 150-600; speed steps up at `step_at`."""
    lab = np.array(["H"] * 200 + ["D"] * 200 + ["A"] * (n - 400), dtype="<U1")
    period = np.ones(n, dtype=np.int8)
    vis = np.ones(n, bool)
    vis[150:600] = False
    p90 = np.where(np.arange(n) < step_at, 1.5, 4.0).astype(np.float32)
    p90[:160] = 4.0                                      # play before the stoppage: steps down at 160
    return lab, period, vis, p90


def _runs(lab):
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return [(a, lab[a]) for a in np.concatenate([[0], cut])]


def test_off_is_identity_and_the_same_object():
    lab, period, vis, p90 = _case()
    assert edges.apply(lab, period, vis, p90) is lab
    assert edges.apply(lab, period, vis, p90, ed_exit=0, ed_entry=0, ed_dx=0.0, ed_de=0.0, ed_w=2.0) is lab
    assert not edges.active({}) and edges.active(dict(ed_exit=1))


def test_snap_finds_a_step_at_a_known_frame_for_exit_and_entry():
    lab, period, vis, p90 = _case(step_at=420)
    moves = []
    out = edges.snap_edges(lab, period, vis, p90, ed_exit=1, moves=moves)
    assert _runs(out) == [(0, "H"), (200, "D"), (420, "A")] and moves == [("exit", 400, 420)]
    assert (lab == np.array(["H"] * 200 + ["D"] * 200 + ["A"] * 600)).all()       # input not written
    out = edges.snap_edges(lab, period, vis, p90, ed_exit=1, ed_entry=1)
    assert _runs(out) == [(0, "H"), (160, "D"), (420, "A")]                       # entry: step down at 160
    early = edges.snap_edges(lab, period, vis, _case(step_at=370)[3], ed_exit=1)  # D gives frames to the next team
    assert _runs(early) == [(0, "H"), (200, "D"), (370, "A")]
    # a step outside the +-2 s search window, or below theta, moves nothing
    assert (edges.snap_edges(lab, period, vis, _case(step_at=480)[3], ed_exit=1) == lab).all()
    assert (edges.snap_edges(lab, period, vis, p90, ed_exit=1, ed_theta=5.0) == lab).all()
    assert _runs(edges.snap_edges(lab, period, vis, _case(step_at=480)[3], ed_exit=1, ed_win=4.0))[-1] == (480, "A")


def test_snap_is_idempotent_and_only_touches_hidden_edges_of_long_runs():
    lab, period, vis, p90 = _case()
    once = edges.snap_edges(lab, period, vis, p90, ed_exit=1, ed_entry=1)
    twice = edges.snap_edges(once, period, vis, p90, ed_exit=1, ed_entry=1)
    assert (once != twice).sum() <= 1
    allvis = np.ones(len(lab), bool)
    assert (edges.snap_edges(lab, period, allvis, p90, ed_exit=1, ed_entry=1) == lab).all()
    assert _runs(edges.snap_edges(lab, period, allvis, p90, ed_exit=1, ed_vis=1))[-1] == (420, "A")
    short = lab.copy()
    short[200:340] = "H"                                 # D run of 60 frames < 3 s
    assert (edges.snap_edges(short, period, vis, p90, ed_exit=1, ed_entry=1) == short).all()


def test_edges_never_cross_a_period_break_or_the_middle_of_a_neighbour_and_keep_every_run():
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = 3000
        lab = np.empty(n, dtype="<U1")
        i = 0
        while i < n:
            L = int(rng.integers(10, 400))
            lab[i:i + L] = rng.choice(["H", "A", "D"])
            i += L
        period = np.array([1] * 1500 + [2] * 1500, dtype=np.int8)
        period[:20] = 0
        lab[period == 0] = "X"
        vis = rng.random(n) > 0.6
        p90 = rng.uniform(0.5, 5.0, n).astype(np.float32)
        p90[rng.random(n) < 0.05] = np.nan
        for kw in (dict(ed_exit=1, ed_entry=1, ed_win=4.0), dict(ed_dx=3.0, ed_de=-3.0), dict(ed_exit=1, ed_vis=1, ed_w=2.0)):
            out = edges.apply(lab, period, vis, p90, **kw)
            assert (out[period == 0] == "X").all() and not (out[period > 0] == "X").any()
            segs = lambda x: [(a, b, x[a]) for a, b in edges._segments(x, period)]
            before, after = segs(lab), segs(out)
            assert [s[2] for s in before] == [s[2] for s in after]            # no run created or deleted
            for (a0, b0, k), (a1, b1, _) in zip(before, after):
                assert a1 < b1
                if k != "D":                                                  # a team run keeps its middle
                    assert a1 <= (a0 + b0) // 2 < b1 or b0 - a0 == 1
                else:
                    assert a1 <= (a0 + b0) // 2 <= b1
            for a, b, k in after:                                             # no run spans the period break
                assert period[a] == period[b - 1]


def test_period_break_guards_the_step_window():
    v = np.r_[np.full(100, 1.0), np.full(100, 5.0)]
    period = np.r_[np.ones(100), np.full(100, 2)].astype(np.int8)
    s = edges.step_score(v, period, 25)
    assert np.isnan(s[76:125]).all()                     # a window reading across the break is not used
    assert np.nanmax(np.abs(s)) == 0.0
    one = edges.step_score(v, np.ones(200, dtype=np.int8), 25)
    assert int(np.nanargmax(one)) == 100 and abs(one[100] - 4.0) < 1e-9


def test_shift_moves_hidden_edges_by_a_constant_within_the_guards():
    lab, period, vis, _ = _case()
    out = edges.shift_edges(lab, period, vis, ed_dx=1.0, ed_de=-0.4)
    assert _runs(out) == [(0, "H"), (190, "D"), (425, "A")]
    far = edges.shift_edges(lab, period, vis, ed_dx=100.0)                        # clipped at the neighbour's middle
    assert _runs(far) == [(0, "H"), (200, "D"), (700, "A")]


def test_home_away_swap_is_symmetric():
    lab, period, vis, p90 = _case()
    swap = np.array([{"H": "A", "A": "H", "D": "D"}[x] for x in lab])
    for kw in (dict(ed_exit=1, ed_entry=1), dict(ed_dx=0.8, ed_de=0.4)):
        a, b = edges.apply(lab, period, vis, p90, **kw), edges.apply(swap, period, vis, p90, **kw)
        assert (a == np.array([{"H": "A", "A": "H", "D": "D"}[x] for x in b])).all()


def test_default_carries_the_exit_snap_and_off_reproduces_the_previous_default_byte_for_byte():
    assert pipeline.DEC["ed_exit"] == 1 and pipeline.DEC["ed_w"] == 1.0 and pipeline.DEC["ed_win"] == 2.0
    assert not any(k in DEC_SHAPE_L2_16 for k in edges.KEYS) and not any(k in pipeline.DEC_BASE for k in edges.KEYS)
    assert {k: v for k, v in DEC_SHAPE.items() if k not in edges.KEYS} == dict(DEC_SHAPE_L2_16, smooth_touch=1, sd_prior=1, reappear=1.0, vtx=2.0, ctl_s=0.6, ctl_hi=2.0, ctl_touch=1, w_fly=0.7, fly_v0=4.0, fly_v1=8.0, ed_cross=5.0, ed_cross_strict=1, ed_xdead=1, ed_xdead_min=2.0, w_bt=0.5, w_bd=1.0, ed_still=1.0)   # L2-20..22, L3-A4, L4 ADR-022
    rng = np.random.default_rng(3)
    n = 3000
    dh, da = rng.uniform(0.1, 8, n), rng.uniform(0.1, 8, n)
    vis = np.ones(n, bool); vis[900:1700] = False
    sd = np.full(n, -2.0); sd[1000:1300] = 3.0
    p90 = np.where((np.arange(n) >= 1000) & (np.arange(n) < 1320), 1.5, 4.0).astype(np.float32)
    f = dict(d_home=dh, d_away=da, visible=vis, dead=np.zeros(n, bool), shape_team=np.zeros(n) + 1e-3, shape_dead=sd)
    period = np.ones(n, dtype=np.int8)
    old = decode_halves(period, feats=dict(f), **dict(pipeline.DEC_BASE, **DEC_SHAPE_L2_16))
    off = decode_halves(period, feats=dict(f, edge_p90=p90), **dict(pipeline.DEC, ed_exit=0, reappear=0.0, ctl_s=0))
    assert (old == off).all()
    assert (decode_halves(period, feats=dict(f), **dict(pipeline.DEC, reappear=0.0, ctl_s=0)) == old).all()    # no edge column: snap skipped
    new = decode_halves(period, feats=dict(f, edge_p90=p90), **dict(pipeline.DEC, reappear=0.0, ctl_s=0))   # reappear (L2-22) tested in test_decode; ctl_s (ADR-022) in test_ctl_truth
    end = lambda lab: int(np.flatnonzero(lab == "D").max()) + 1
    assert end(old) == 1300 and end(new) == 1320 and (new[:1300] == old[:1300]).all() and (new[1320:] == old[1320:]).all()


def test_lomo_shift_never_sees_the_scored_match():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import diagnose

    def match(offset_frames):
        n = 4000
        truth = np.array(["H"] * n, dtype="<U1")
        lab = truth.copy()
        for a in (500, 1500, 2500):
            truth[a:a + 300] = "D"
            lab[a:a + 300 + offset_frames] = "D"
        return dict(truth=truth, lab=lab, period=np.ones(n, dtype=np.int8), visible=np.zeros(n, bool))

    data = [("a", match(-25)), ("b", match(-25)), ("c", match(75))]
    s = diagnose.ed_lomo_shift(data)
    assert s == {"a": -1.0, "b": -1.0, "c": 1.0}         # c's own +3 s offset never enters c's scalar
