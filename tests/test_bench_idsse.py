"""IDSSE truth mapping and row layout on a synthetic DFL snippet (scripts/idsse_cache.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import idsse_cache as ic  # noqa: E402

META = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation><Teams>
<Team TeamId="T_AWAY" Role="guest"><Players>
<Player PersonId="P_AGK" PlayingPosition="TW"/><Player PersonId="P_A1" PlayingPosition="STZ"/>
</Players></Team>
<Team TeamId="T_HOME" Role="home"><Players><Player PersonId="P_H1" PlayingPosition="IVZ"/></Players></Team>
</Teams></MatchInformation></PutDataRequest>
"""


def _fs(sec, team, pid, frames):
    rows = "\n".join('<Frame N="%d" X="%.2f" Y="%.2f"%s/>' % f for f in frames)
    return '<FrameSet GameSection="%s" TeamId="%s" PersonId="%s">\n%s\n</FrameSet>' % (sec, team, pid, rows)


def _ball(n, st, po):
    return (n, 1.0, 2.0, ' Z="0.00" BallPossession="%d" BallStatus="%d"' % (po, st))


def _write(tmp_path):
    sets = [
        _fs("firstHalf", "referee", "R1", [(10000, 0, 0, ""), (10001, 0, 0, "")]),
        _fs("firstHalf", "T_HOME", "P_H1", [(10000, -5, 1, ""), (10001, -5, 1, ""), (10002, -5, 1, "")]),
        _fs("firstHalf", "T_AWAY", "P_AGK", [(10001, 50, 0, "")]),
        _fs("firstHalf", "BALL", "B", [_ball(10000, 1, 1), _ball(10001, 1, 2), _ball(10002, 0, 2)]),
        _fs("secondHalf", "T_AWAY", "P_A1", [(100000, 3, 3, ""), (100001, 3, 3, "")]),
        _fs("secondHalf", "BALL", "B", [_ball(100000, 1, 2), _ball(100001, 1, 1)]),
    ]
    pos = '<?xml version="1.0"?>\n<PutDataRequest><Positions>\n%s\n</Positions></PutDataRequest>\n' % "\n".join(sets)
    p, m = tmp_path / "pos.xml", tmp_path / "meta.xml"
    p.write_text(pos); m.write_text(META)
    return str(p), str(m)


def test_truth_labels():
    lab = ic.truth_labels([1, 1, 0, 0, 1], [1, 2, 1, 2, 0])
    assert list(lab) == ["H", "A", "D", "D", "X"]


def test_parse_rows_truth_and_tracks(tmp_path):
    d = ic.parse(*_write(tmp_path))
    g = ic.GAP_ROWS
    assert len(d["truth"]) == 3 + g + 2
    assert list(d["truth"][:3]) == ["H", "A", "D"]
    assert set(d["truth"][3:3 + g]) == {"X"}
    assert list(d["truth"][3 + g:]) == ["A", "H"]
    assert list(d["period"][:3]) == [1, 1, 1] and list(d["period"][3 + g:]) == [2, 2]
    assert not d["period"][3:3 + g].any()
    assert d["ball_x"][0] == 1.0 and np.isnan(d["ball_x"][3])
    m = ic.IdsseMatch("synthetic", data=d)
    frames = list(m.iter_frames())
    assert [f[0] for f in frames] == [0, 1, 2, 3 + g, 4 + g]
    _, per, tr, ok = frames[1]
    assert per == 1 and ok and tr["ball"][1]["map_position"] == (1.0, 2.0)
    (home,) = tr["players"].values()
    assert home["team"] == 1 and home["map_position"] == (-5.0, 1.0)
    (keeper,) = tr["goalkeepers"].values()          # TW -> goalkeeper, guest -> team 2
    assert keeper["team"] == 2 and keeper["role"] == "goalkeeper"
    _, per, tr, _ = frames[3]
    assert per == 2 and [r["team"] for r in tr["players"].values()] == [2]   # referee never appears


def test_sanity_counts():
    s = ic.sanity(np.array(list("HHHDAAXHH")), fps=1)
    assert s["possessions"] == 3 and abs(s["mean_s"] - 7 / 3) < 1e-9
