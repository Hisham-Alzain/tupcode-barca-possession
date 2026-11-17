"""End-to-end pipeline on a synthetic frame stream (no data needed)."""
import numpy as np

from possess import features
from possess.pipeline import label


def _frames(n=400):
    """Home dribbles for 6 s, away takes it at 6 s and dribbles on; ball hidden 10-12 s."""
    for f in range(n):
        x = -10 + f * 0.05
        home_at_ball = f < 150
        players = {}
        for k in range(11):
            players[k] = {"map_position": (x - 15 + k * 3.0, -22 + k * 4.0), "team": 1}  # none on the ball line
            players[100 + k] = {"map_position": (x - 14 + k * 3.0, -18 + k * 4.0), "team": 2}
        carrier = 0 if home_at_ball else 100
        players[carrier] = {"map_position": (x, 0.3), "team": 1 if home_at_ball else 2}
        hidden = 250 <= f < 300
        tracks = {"players": players, "goalkeepers": {}, "ball": {} if hidden else
                  {1: {"map_position": (x, 0.0), "conf": 1.0}}}
        yield f, 1, tracks, not hidden


def test_pipeline_follows_a_turnover_and_keeps_team_through_a_gap():
    f = features.compute(_frames(), n=400)
    lab = "".join(label(f))
    assert set(lab) <= {"H", "A", "D", "X"}
    assert lab[100] == "H" and lab[200] == "A"
    assert lab[260:300] == "A" * 40           # ball hidden: team kept, not dead
    assert "D" not in lab                     # nothing here is a stoppage


def test_features_shapes():
    f = features.compute(_frames(50), n=50)
    assert all(len(v) == 50 for v in f.values())
    assert not np.isnan(f["d_home"][10]) and f["ml_dead"].dtype == bool


def test_label_adds_shape_evidence_only_when_positions_are_present():
    from possess import pipeline

    f = features.compute(_frames(), n=400)
    fp = features.compute(_frames(), n=400, positions=True)
    assert fp["pos"].shape[0] == 400
    base = label(f)
    assert (label({k: v for k, v in fp.items() if k != "pos"}) == base).all()   # no pos -> DEC_BASE
    assert (label(f, dec_kw=pipeline.DEC_BASE) == base).all()
    lab = label(fp)                                        # pos -> DEC (shape config), still valid labels
    assert set(lab) <= {"H", "A", "D", "X"} and len(lab) == 400
    assert pipeline.DEC["w_shape"] > 0 and "w_shape" not in pipeline.DEC_BASE
