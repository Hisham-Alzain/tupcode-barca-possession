"""Segment clean-up rules (possess/cleanup.py, envelope L2-17)."""
import numpy as np

from possess import pipeline
from possess.cleanup import KEYS, cleanup
from possess.decode import decode_halves

FPS = 25.0


def seq(*parts):
    """seq(("H", 100), ("A", 5)) -> label array."""
    return np.array([k for k, n in parts for _ in range(n)], dtype="<U1")


def run(lab, period=None, visible=None, touch=None, **kw):
    period = np.ones(len(lab), dtype=np.int8) if period is None else period
    visible = np.ones(len(lab), bool) if visible is None else visible
    return cleanup(lab, period, visible, touch, fps=FPS, **kw)


def swap(lab):
    return np.array([{"H": "A", "A": "H"}.get(k, k) for k in lab], dtype="<U1")


def text(lab):
    return "".join(lab)


def test_flicker_absorbed_only_below_threshold():
    lab = seq(("H", 100), ("A", 9), ("H", 100))
    assert text(run(lab, m_flicker=0.4)) == "H" * 209            # 9 frames < 10
    lab = seq(("H", 100), ("A", 10), ("H", 100))
    assert text(run(lab, m_flicker=0.4)) == text(lab)            # 10 frames is not shorter than 0.4 s


def test_flicker_needs_same_other_team_on_both_sides():
    lab = seq(("H", 100), ("A", 5), ("D", 100))
    assert text(run(lab, m_flicker=2.0)) == text(lab)


def test_bridge():
    lab = seq(("H", 100), ("D", 20), ("H", 100))
    assert text(run(lab, m_bridge=1.0)) == "H" * 220
    lab = seq(("H", 100), ("D", 20), ("A", 100))                 # a stoppage between two teams stays
    assert text(run(lab, m_bridge=1.0)) == text(lab)


def test_island():
    lab = seq(("D", 100), ("A", 12), ("D", 100))
    assert text(run(lab, m_island=0.6)) == "D" * 212
    assert text(run(lab, m_flicker=5.0, m_bridge=5.0, m_sliver=5.0)) == text(lab)


def test_sliver_both_directions_and_cl_dir():
    after = seq(("D", 100), ("H", 8), ("A", 100))                # D h A -> D A
    before = seq(("A", 100), ("H", 8), ("D", 100))               # A h D -> A D
    assert text(run(after, m_sliver=0.4)) == "D" * 100 + "A" * 108
    assert text(run(before, m_sliver=0.4)) == "A" * 108 + "D" * 100
    assert text(run(after, m_sliver=0.4, cl_dir=2)) == text(after)
    assert text(run(before, m_sliver=0.4, cl_dir=1)) == text(before)
    assert text(run(after, m_sliver=0.4, cl_dir=1)) == "D" * 100 + "A" * 108
    assert text(run(before, m_sliver=0.4, cl_dir=2)) == "A" * 108 + "D" * 100


def test_thresholds_zero_is_identity():
    rng = np.random.default_rng(0)
    lab = rng.choice(np.array(["H", "A", "D"]), 2000)
    out = run(lab, cl_protect=1, cl_vis=1, cl_dir=1)
    assert out is lab and text(out) == text(lab)


def test_idempotent_and_never_more_runs():
    rng = np.random.default_rng(1)
    kw = dict(m_flicker=0.8, m_bridge=1.0, m_island=0.6, m_sliver=0.4)
    for _ in range(20):
        parts = [(str(rng.choice(["H", "A", "D"])), int(rng.integers(1, 60))) for _ in range(80)]
        lab = seq(*parts)
        once = run(lab, max_pass=50, **kw)
        assert text(run(once, max_pass=50, **kw)) == text(once)
        assert (once[1:] != once[:-1]).sum() <= (lab[1:] != lab[:-1]).sum()
        assert set(once) <= {"H", "A", "D"}
        assert text(lab) == text(seq(*parts))                    # the input is not modified


def test_home_away_swap_symmetry():
    rng = np.random.default_rng(2)
    kw = dict(m_flicker=0.8, m_bridge=1.0, m_island=0.6, m_sliver=0.4, cl_protect=1)
    for _ in range(20):
        lab = seq(*[(str(rng.choice(["H", "A", "D"])), int(rng.integers(1, 60))) for _ in range(80)])
        touch = rng.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0], len(lab))
        assert text(swap(run(lab, touch=touch, **kw))) == text(run(swap(lab), touch=-touch, **kw))


def test_half_boundary_untouched_and_x_never_written():
    # A h | h A: each short home run is the first / last run of its half, so no rule may touch it
    lab = np.concatenate([seq(("A", 100), ("H", 5)), seq(("X", 10)), seq(("H", 5), ("A", 100))])
    period = np.array([1] * 105 + [0] * 10 + [2] * 105, dtype=np.int8)
    kw = dict(m_flicker=5.0, m_bridge=5.0, m_island=5.0, m_sliver=5.0)
    assert text(run(lab, period=period, **kw)) == text(lab)
    # two halves that touch (no gap): the run pair straddling the boundary is not one flicker
    lab = seq(("A", 100), ("H", 5), ("H", 3), ("A", 100))
    period = np.array([1] * 105 + [2] * 103, dtype=np.int8)
    assert text(run(lab, period=period, **kw)) == text(lab)
    # frames outside the halves keep whatever they hold
    lab = seq(("X", 20), ("H", 100), ("A", 5), ("H", 100), ("X", 20))
    period = np.array([0] * 20 + [1] * 205 + [0] * 20, dtype=np.int8)
    out = run(lab, period=period, **kw)
    assert text(out) == "X" * 20 + "H" * 205 + "X" * 20


def test_protect_keeps_a_touched_spell():
    lab = seq(("H", 100), ("A", 5), ("H", 100))
    touch = np.zeros(len(lab))
    assert text(run(lab, touch=touch, m_flicker=0.4, cl_protect=1)) == "H" * 205
    touch[102] = -1.0                                            # an away touch inside the away spell
    assert text(run(lab, touch=touch, m_flicker=0.4, cl_protect=1)) == text(lab)
    assert text(run(lab, touch=touch, m_flicker=0.4, cl_protect=0)) == "H" * 205
    touch[102] = 1.0                                             # a home touch does not protect it
    assert text(run(lab, touch=touch, m_flicker=0.4, cl_protect=1)) == "H" * 205
    isl = seq(("D", 100), ("A", 5), ("D", 100))
    touch[102] = -1.0
    assert text(run(isl, touch=touch, m_island=0.6, cl_protect=1)) == text(isl)
    brd = seq(("H", 100), ("D", 5), ("H", 100))                  # R2 is never protected
    assert text(run(brd, touch=touch, m_bridge=1.0, cl_protect=1)) == "H" * 205


def test_cl_vis_selects_hidden_or_visible_majority_flickers():
    lab = seq(("H", 100), ("A", 6), ("H", 100))
    vis = np.ones(len(lab), bool)
    assert text(run(lab, visible=vis, m_flicker=0.4, cl_vis=1)) == text(lab)
    assert text(run(lab, visible=vis, m_flicker=0.4, cl_vis=2)) == "H" * 206
    vis[100:104] = False                                         # 4 of 6 frames hidden
    assert text(run(lab, visible=vis, m_flicker=0.4, cl_vis=1)) == "H" * 206
    assert text(run(lab, visible=vis, m_flicker=0.4, cl_vis=2)) == text(lab)


def test_order_island_before_flicker_and_repeat():
    # D a D h: the island goes first (-> D), nothing is left for the flicker rule
    lab = seq(("H", 100), ("D", 100), ("A", 5), ("D", 100), ("H", 100))
    assert text(run(lab, m_island=0.6, m_flicker=0.4)) == "H" * 100 + "D" * 205 + "H" * 100
    # H d H a H: the bridge merges the home runs, then the flicker is absorbed
    lab = seq(("H", 100), ("D", 5), ("H", 3), ("A", 4), ("H", 100))
    assert text(run(lab, m_bridge=1.0, m_flicker=0.4)) == "H" * 212


def _feats(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    bx, by = 52 + 30 * np.sin(t / 7) + rng.normal(0, 0.3, n).cumsum() * 0.05, 34 + 20 * np.cos(t / 5)
    blocks = rng.integers(0, 30, n // 10).repeat(10)[:n]
    dh = np.where(blocks % 2 == 0, 0.3, 3.0) + rng.random(n) * 0.5
    da = np.where(blocks % 2 == 1, 0.3, 3.0) + rng.random(n) * 0.5
    vis = np.ones(n, bool)
    vis[700:1100] = False
    bx, by = np.where(vis, bx, np.nan), np.where(vis, by, np.nan)
    dead = np.zeros(n, bool)
    dead[1500:1700] = True
    return {"d_home": np.where(vis, dh, 99.0), "d_away": np.where(vis, da, 99.0), "visible": vis, "dead": dead,
            "ball_x": bx, "ball_y": by}


def test_default_has_no_cleanup_and_rules_off_is_byte_identical():
    assert not any(k in pipeline.DEC for k in KEYS)              # negative result: the default is unchanged
    f = _feats()
    period = np.array([1] * 1500 + [2] * 1500, dtype=np.int8)
    base = decode_halves(period, feats=dict(f), **pipeline.DEC_BASE)
    off = decode_halves(period, feats=dict(f), **dict(pipeline.DEC_BASE, m_flicker=0.0, m_bridge=0.0, m_island=0.0,
                                                    m_sliver=0.0, cl_protect=1, cl_vis=0, cl_dir=0))
    assert base.tobytes() == off.tobytes()


def test_decode_halves_post_pass_equals_cleanup_on_its_output():
    f = _feats()
    period = np.array([1] * 1500 + [2] * 1500, dtype=np.int8)
    base = decode_halves(period, feats=dict(f), **pipeline.DEC_BASE)
    kw = dict(m_flicker=1.2, m_bridge=1.0, m_island=1.0, m_sliver=0.8)
    hooked = decode_halves(period, feats=dict(f), **dict(pipeline.DEC_BASE, **kw))
    assert text(hooked) == text(cleanup(base, period, f["visible"], None, fps=FPS, **kw))
    assert (hooked != base).any()                                # the synthetic sequence has something to clean
    prot = decode_halves(period, feats=dict(f), **dict(pipeline.DEC_BASE, cl_protect=1, **kw))
    assert set(prot) <= {"H", "A", "D"} and len(prot) == len(base)
