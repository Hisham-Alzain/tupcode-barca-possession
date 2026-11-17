"""L11-A: edges.restart_dead (out of play until the restart kick) and its opt-in decoder flag."""
import numpy as np

from possess import edges


def _case():
    fps = 25
    n = 20 * fps
    x = np.full(n, 30.0)
    y = np.zeros(n)
    # 0-2 s: ball travels to the goal line and out (x 30 -> 55)
    x[:50] = np.linspace(30.0, 55.0, 50)
    # 2-4 s: beyond the line, slow; 4-6 s: carried back into the goal area (x 55 -> 49)
    x[50:100] = 55.0
    x[100:150] = np.linspace(55.0, 49.0, 50)
    # 6-10 s: placed (at rest) in the goal area; 10 s: goal kick at 15 m/s
    x[150:250] = 49.0
    x[250:] = 49.0 - (np.arange(n - 250) * 15.0 / fps)
    x = np.maximum(x, -50.0)
    lab = np.array(["H"] * n)
    lab[55:100] = "D"            # our stoppage ends when the ball comes back in
    return lab, np.ones(n, int), x, y


def test_restart_dead_extends_to_the_kick():
    lab, per, x, y = _case()
    out = edges.restart_dead(lab, per, x, y, v_k=5.0)
    assert (out[100:248] == "D").all()          # carried back + placed = dead
    assert (out[262:] == "H").all()             # after the goal kick: live
    assert (out[:45] == "H").all()              # before the crossing: unchanged


def test_restart_dead_no_kick_no_change():
    lab, per, x, y = _case()
    x[250:] = 49.0                               # never kicked within max_s
    out = edges.restart_dead(lab, per, x, y, v_k=5.0, max_s=8.0)
    assert (out == lab).all()


def test_flag_off_is_identity_in_decoder():
    import inspect

    from possess import decode
    src = inspect.getsource(decode.decode_halves)
    assert 'kw.pop("ed_rkick", 0)' in src and "if ed_rkick > 0" in src


def test_law_goal_line_ignores_a_toss_from_beyond_the_line():
    lab, per, x, y = _case()
    x[100:110] = np.linspace(55.0, 51.0, 10)     # tossed back in fast (10 m/s) from beyond the goal line
    x[110:150] = np.linspace(51.0, 49.0, 40)
    out = edges.restart_dead(lab, per, x, y, v_k=5.0, law=True)
    assert (out[100:248] == "D").all()          # the toss is not the restart; the goal kick is
    out2 = edges.restart_dead(lab, per, x, y, v_k=5.0)
    assert (out2[110:248] != "D").any()         # without law the toss ends the stoppage
