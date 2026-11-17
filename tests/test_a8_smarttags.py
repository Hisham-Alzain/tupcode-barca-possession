"""Smart Tagging inputs (envelope L6): parsing, the per-frame arrays and the opt-in decoder terms."""
import numpy as np

from possess import smarttags as st
from possess.decode import decode_halves
from possess.pipeline import DEC_BASE, TAGS

XML = """<?xml version="1.0" ?>
<file><ALL_INSTANCES>
<instance><ID>1</ID><start>4.0</start><end>8.0</end><code>PROGRESSION</code>
 <label><group>Team</group><text>Home FC</text></label><label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>2</ID><start>4.0</start><end>8.0</end><code>DEFENDING IN MIDDLE THIRD</code>
 <label><group>Team</group><text>Away FC</text></label><label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>3</ID><start>6.0</start><end>12.0</end><code>BUILD UP</code>
 <label><group>Team</group><text>Away FC</text></label><label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>4</ID><start>20.0</start><end>24.0</end><code>SET PIECES</code>
 <label><group>Team</group><text>Away FC</text></label><label><group>Type</group><text>Corner Kick</text></label>
 <label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>5</ID><start>30.0</start><end>34.0</end><code>SET PIECES</code>
 <label><group>Team</group><text>N/A</text></label><label><group>Type</group><text>Throw-in</text></label>
 <label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>6</ID><start>40.0</start><end>44.0</end><code>GOALS</code>
 <label><group>Team</group><text>Home FC</text></label><label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>7</ID><start>60.0</start><end>64.0</end><code>SET PIECES</code>
 <label><group>Team</group><text>N/A</text></label><label><group>Type</group><text>Kick Off (After Goal)</text></label>
 <label><group>Half</group><text>1st Half</text></label></instance>
</ALL_INSTANCES></file>"""


def _tags(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(XML)
    return st.load_tags(str(p))


def test_load_and_signs(tmp_path):
    tags = _tags(tmp_path)
    assert len(tags) == 7
    t = tags[0]
    assert (t["a"], t["b"], t["half"]) == (100, 200, 1)
    by = {(t["code"], t["a"]): t for t in tags}
    assert st.phase_sign(by["PROGRESSION", 100], "Home FC") == 1
    assert st.phase_sign(by["DEFENDING IN MIDDLE THIRD", 100], "home fc") == 1      # away defends = home has it
    assert st.phase_sign(by["BUILD UP", 150], "Home FC") == -1
    assert st.phase_sign(by["SET PIECES", 500], "Home FC") == 0                    # not a phase clip
    assert by["SET PIECES", 750]["team"] is None


def test_phase_prior_overlap_offset_trim(tmp_path):
    tags = _tags(tmp_path)
    p = st.phase_prior(tags, 400, "Home FC", offset=1)
    assert p[100] == 0 and p[101] == 1 and p[150] == 1         # frame_id = video frame + 1
    assert (p[151:202] == 0).all()                             # both teams named: no evidence
    assert (p[202:302] == -1).all() and p[302] == 0
    q = st.phase_prior(tags, 400, "Home FC", offset=0, trim_s=1.0)
    assert q[124] == 0 and q[125] == 1 and (q[276:] == 0).all() and q[275] == -1


def test_restart_events_and_goal_kickoff(tmp_path):
    ev = st.restart_events(_tags(tmp_path), "Home FC")
    assert [(e["a"], e["sign"], e["type"]) for e in ev] == [(500, -1, "Corner Kick"), (1500, -1, "Kick Off (After Goal)")]


def test_restart_frame_and_prior():
    dead = np.zeros(1000, bool)
    dead[90:130] = True          # starts before the clip: still a stoppage inside it
    dead[180:260] = True         # last stoppage that starts in the clip; ends after it
    ev = {"a": 100, "b": 200, "sign": -1}
    assert st.restart_frame(ev, dead, offset=0) == 260
    pr = st.restart_prior([ev], dead, hold_s=2.0, offset=0)
    assert (pr[260:310] == -1).all() and pr[259] == 0 and pr[310] == 0
    assert st.restart_frame({"a": 400, "b": 500, "sign": 1}, dead, offset=0) is None
    assert not st.restart_prior([ev], dead, hold_s=2.0, offset=0, max_after_s=1.0).any()


def test_home_team_name(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text('<r><Score idLocalTeam="b" idVisitingTeam="a"/><Teams><Team id="a"><Name>Away FC</Name></Team>'
                 '<Team id="b"><Name>Home FC</Name></Team></Teams></r>')
    assert st.home_team_name(str(p)) == "Home FC"


def _feats(n=1500):
    rng = np.random.default_rng(0)
    vis = np.ones(n, bool)
    vis[600:1100] = False
    dh = np.where(np.arange(n) < 600, 0.3, 5.0) + rng.uniform(0, 0.05, n)       # home on the ball, then nobody
    da = np.full(n, 5.0) + rng.uniform(0, 0.05, n)
    dead = np.zeros(n, bool)
    dead[1100:1250] = True
    return {"d_home": dh, "d_away": da, "visible": vis, "dead": dead,
            "ball_x": np.where(vis, 0.0, np.nan), "ball_y": np.where(vis, 0.0, np.nan)}


def test_flags_off_are_identical_and_default_off():
    assert TAGS["w_tag"] == 0 and TAGS["w_tagr"] == 0
    f, per = _feats(), np.ones(1500, np.int8)
    base = decode_halves(per, feats=dict(f), **DEC_BASE)
    extra = {"tag_team": -np.ones(1500), "tag_events": [{"a": 1050, "b": 1150, "sign": -1}]}
    same = decode_halves(per, feats=dict(f, **extra), **DEC_BASE)
    assert (base == same).all()


def test_phase_prior_names_the_team_on_hidden_frames():
    f, per = _feats(), np.ones(1500, np.int8)
    tag = np.zeros(1500)
    tag[700:1100] = -1
    base = decode_halves(per, feats=dict(f), **DEC_BASE)
    lab = decode_halves(per, feats=dict(f, tag_team=tag), **dict(DEC_BASE, w_tag=0.05))
    assert (base[700:1100] == "H").all()
    assert (lab[800:1100] == "A").all() and (lab[:600] == "H").all()
    assert (lab[1100:1250] == "D").all()


def test_restart_taker_two_pass():
    f, per = _feats(), np.ones(1500, np.int8)
    ev = [{"a": 1050, "b": 1150, "sign": -1}]
    base = decode_halves(per, feats=dict(f), **DEC_BASE)
    lab = decode_halves(per, feats=dict(f, tag_events=ev), **dict(DEC_BASE, w_tagr=1.0, tagr_hold_s=4.0))
    assert (lab[1100:1250] == "D").all()
    assert (lab[1250:1350] == "A").all()
    assert (lab[:1100] == base[:1100]).all()
