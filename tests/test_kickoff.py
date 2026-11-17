import numpy as np

from possess.kickoff import kickoff_frames, kickoff_windows

XML = """<?xml version="1.0" ?><file><ALL_INSTANCES>
<instance><ID>1</ID><start>4.0</start><end>8.0</end><code>SET PIECES</code>
<label><group>Type</group><text>Kick Off (Start)</text></label><label><group>Half</group><text>2nd Half</text></label></instance>
<instance><ID>2</ID><start>1.0</start><end>2.0</end><code>SET PIECES</code>
<label><group>Type</group><text>Kick Off (After Goal)</text></label><label><group>Half</group><text>1st Half</text></label></instance>
</ALL_INSTANCES></file>"""


def test_kickoff_window_and_frame(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(XML)
    assert kickoff_windows(str(p)) == [(2, 100, 200)]       # after-goal kick-offs ignored
    n = 400
    bx = np.full(n, np.nan)
    by = np.zeros(n)
    bx[1:] = np.linspace(-20, 20, n - 1)                    # frame_id = video frame + 1
    bx[:120] = np.nan                                        # ball hidden at the window start
    k = kickoff_frames(str(p), bx, by)[2]
    assert 100 <= k <= 200 and abs(bx[k + 1]) == np.nanmin(np.abs(bx[101:202]))


def test_pre_kickoff_spans_rules(tmp_path):
    from possess.kickoff import pre_kickoff_spans

    p = tmp_path / "t.xml"
    p.write_text(XML)                                        # 2nd-half window 100-200
    bx = np.full(400, np.nan)
    by = bx.copy()
    assert pre_kickoff_spans(str(p), bx, by, [(2, 50, 390)]) == [(50, 100)]
    assert pre_kickoff_spans(str(p), bx, by, [(2, 120, 390)]) == []    # graded from the kick-off
    assert pre_kickoff_spans(str(p), bx, by, [(2, 10, 90)]) == []      # tag after the half
    assert pre_kickoff_spans(str(p), bx, by, [(1, 50, 390)]) == []     # no tag for that half


def test_label_with_kickoffs_writes_d(monkeypatch):
    from possess import pipeline as pl

    monkeypatch.setattr(pl, "HALVES", [(1, 10, 99)])
    monkeypatch.setattr(pl, "N_FRAMES", 120)
    seen = {}

    def fake_label(f, period):
        seen["period"] = period
        return np.array(["H"] * 121)

    monkeypatch.setattr(pl, "label", fake_label)
    per = np.ones(121, dtype=np.int8)
    rows = pl.label_with_kickoffs({}, per, [(10, 30)])
    assert (rows[10:30] == "D").all() and rows[30] == "H" and rows[9] == "X"
    assert (seen["period"][11:31] == 0).all() and seen["period"][31] == 1 and per[11] == 1


def test_no_ball_uses_window_start(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(XML)
    bx = np.full(400, np.nan)
    assert kickoff_frames(str(p), bx, bx.copy()) == {2: 100}
