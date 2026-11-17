"""Controlled-possession relabelling (possess/ctl_truth.py) and its two opt-in decoder flags
(ctl_s, confirm_s; envelope L4-control-definition)."""
import numpy as np

from possess import pipeline
from possess.ctl_truth import n_possessions, relabel
from possess.decode import decode_halves, touch_hold

FPS = 25.0


def seq(*parts):
    return np.array([k for k, n in parts for _ in range(n)], dtype="<U1")


def run(lab, period=None, **kw):
    period = np.ones(len(lab), dtype=np.int8) if period is None else period
    return "".join(relabel(lab, period, fps=FPS, **kw))


def swap(lab):
    return np.array([{"H": "A", "A": "H"}.get(k, k) for k in lab], dtype="<U1")


def test_short_spell_goes_back_to_the_previous_team():
    lab = seq(("H", 100), ("A", 49), ("H", 100))
    assert run(lab, min_s=2.0) == "H" * 249                       # deflection that returns
    lab = seq(("H", 100), ("A", 50), ("H", 100))
    assert run(lab, min_s=2.0) == "".join(lab)                    # 2 s is controlled
    assert run(lab, min_s=0.0) == "".join(lab)                    # off


def test_uncontrolled_before_a_stoppage_and_restart_possessor_kept():
    lab = seq(("H", 100), ("A", 10), ("D", 50), ("A", 10), ("H", 100))
    # A h D -> the 10 frames before the stoppage go back to H; the restart taker (D a H) is kept
    assert run(lab, min_s=2.0) == "H" * 110 + "D" * 50 + "A" * 10 + "H" * 100
    lab = seq(("A", 10), ("H", 100))                              # first spell of a half is kept
    assert run(lab, min_s=2.0) == "".join(lab)


def test_shortest_first_lets_a_one_touch_interruption_merge():
    # H | a 1.5 s | h 0.2 s | a 1.5 s | H : the one-touch h is absorbed first, A then holds 3.2 s
    lab = seq(("H", 100), ("A", 38), ("H", 5), ("A", 37), ("H", 100))
    assert run(lab, min_s=2.0) == "H" * 100 + "A" * 80 + "H" * 100
    # pinball: nothing reaches 2 s -> all H
    lab = seq(("H", 100), ("A", 8), ("H", 5), ("A", 8), ("H", 5), ("A", 8), ("H", 100))
    assert run(lab, min_s=2.0) == "H" * 234


def test_kept_interception_keeps_its_first_frame_and_d_is_untouched():
    lab = seq(("H", 100), ("A", 60), ("D", 30), ("H", 5), ("A", 100))
    out = run(lab, min_s=2.0)
    assert out == "".join(lab)
    assert out.index("A") == 100


def test_pass_proxy_and_counts():
    lab = seq(("H", 100), ("A", 20), ("H", 100))
    passes = np.zeros(len(lab))
    passes[105] = -1                                              # completed away pass inside the spell
    assert run(lab, min_s=2.0, passes=passes, min_pass=1) == "".join(lab)
    passes[105] = 1                                               # a home pass does not make it controlled
    c = {}
    assert run(lab, min_s=2.0, passes=passes, min_pass=1, counts=c) == "H" * 220
    assert c == {"relabelled": 1, "frames": 20}


def test_periods_are_separate_and_teams_symmetric():
    lab = seq(("H", 100), ("A", 10), ("H", 100), ("A", 10), ("H", 30))
    period = np.array([1] * 105 + [2] * 145, dtype=np.int8)
    out = relabel(lab, period, 2.0, FPS)
    # the split puts the first a across the half boundary: 5 frames end half 1 (go back to H),
    # 5 frames open half 2 (first spell of a half, kept)
    assert "".join(out) == "H" * 105 + "A" * 5 + "H" * 140
    rng = np.random.default_rng(0)
    lab = np.repeat(rng.choice(["H", "A", "D"], 300, p=[0.45, 0.45, 0.1]), rng.integers(3, 120, 300))
    per = np.ones(len(lab), np.int8)
    a = relabel(lab, per, 2.0, FPS)
    assert (swap(relabel(swap(lab), per, 2.0, FPS)) == a).all()
    assert ((a == "D") == (lab == "D")).all()
    assert n_possessions(a, per) <= n_possessions(lab, per)
    assert (relabel(a, per, 2.0, FPS) == a).all()                 # idempotent


def test_n_possessions():
    lab = seq(("H", 10), ("D", 5), ("H", 10), ("A", 3), ("X", 4))
    assert n_possessions(lab, np.ones(len(lab), np.int8)) == 3


def _kick_track():
    """Ball: home kick at 30, away deflection at 50, home touch at 60, then away touch at 150 and
    another away touch at 170. Players stand on the touch points."""
    n = 260
    x = np.zeros(n)
    y = np.zeros(n)
    v = np.zeros(n)
    v[30:50], v[50:60], v[60:150], v[150:170], v[170:] = 10.0, -6.0, 8.0, -9.0, 7.0
    x = np.cumsum(v) / FPS
    dh = np.full(n, 30.0)
    da = np.full(n, 30.0)
    for i, home in ((30, True), (50, False), (60, True), (150, False), (170, False)):
        (dh if home else da)[i - 3:i + 4] = 0.3
    return x, y, dh, da


def test_confirm_ignores_a_touch_that_returns_and_backdates_a_kept_one():
    x, y, dh, da = _kick_track()
    old = touch_hold(x, y, dh, da, max_hold_s=10.0)
    new = touch_hold(x, y, dh, da, max_hold_s=10.0, confirm_s=2.0)
    assert (old[52:57] == -1).all()                               # default: the deflection takes the ball
    assert (new[35:147] == 1).all()                               # confirm: home keeps it through the deflection
    assert (new[153:259] == -1).all() and (old[153:259] == -1).all()   # kept interception: from its first touch
    assert (touch_hold(x, y, dh, da, max_hold_s=10.0, confirm_s=0.0) == old).all()
    assert (touch_hold(x, y, da, dh, max_hold_s=10.0, confirm_s=2.0) == -new).all()


def test_decoder_flags_default_off_and_ctl_s_post_pass():
    assert pipeline.DEC["ctl_s"] == 0.6 and "confirm_s" not in pipeline.DEC   # ADR-022 default
    n = 400
    dh = np.full(n, 0.3)
    da = np.full(n, 20.0)
    dh[200:220], da[200:220] = 20.0, 0.3                          # away at the ball for 0.8 s
    feats = dict(d_home=dh, d_away=da, visible=np.ones(n, bool), dead=np.zeros(n, bool))
    per = np.ones(n, np.int8)
    kw = dict(pipeline.DEC_BASE, w_touch=0.0)
    off = decode_halves(per, feats=feats, **kw)
    on = decode_halves(per, feats=feats, **dict(kw, ctl_s=2.0))
    assert (off[203:217] == "A").all()
    assert (on == "H").all()
    assert (decode_halves(per, feats=feats, **dict(kw, ctl_s=0)) == off).all()
