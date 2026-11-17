"""Ball-free shape evidence (shape.py) on synthetic positions (no data needed)."""
import numpy as np

from possess import shape


def _match(n=600, seed=0):
    """Home (attacks +x, keeper at -45) wide and advancing until n/2, then away wide; nobody moves
    in the last 100 frames. Returns positions [n, 2, SLOTS, 2] and the period."""
    rng = np.random.default_rng(seed)
    pos = np.full((n, 2, shape.SLOTS, 2), np.nan, dtype=np.float32)
    base_y = np.linspace(-1, 1, 10)
    for i in range(n):
        home_ball = i < n // 2
        drift = 0.0 if i >= n - 100 else 0.05 * (i % 100) * (1 if home_ball else -1)
        for t, sign in ((0, 1), (1, -1)):
            wide = 25.0 if (t == 0) == home_ball else 10.0
            x = sign * -10 + drift + rng.normal(0, 3, 10) * (0 if i >= n - 100 else 1)
            pos[i, t, :10, 0] = x
            pos[i, t, :10, 1] = base_y * wide
            pos[i, t, 10] = (sign * -45, 0)
    return pos, np.ones(n, dtype=np.int8)


def test_collect_fills_slots_by_team_without_ids():
    frames = [(0, 1, {"players": {7: {"team": 1, "map_position": (1.0, 2.0)},
                                  9: {"team": 2, "map_position": (3.0, 4.0)}},
                      "goalkeepers": {1: {"team": 1, "map_position": (-40.0, 0.0)}}, "ball": {}}, False)]
    pos = shape.collect(frames, n=2)
    assert pos.shape == (2, 2, shape.SLOTS, 2)
    assert np.isfinite(pos[0, 0, :, 0]).sum() == 2 and np.isfinite(pos[0, 1, :, 0]).sum() == 1
    assert np.isnan(pos[1]).all()


def test_attack_dir_follows_the_side_each_team_stands_on():
    pos, period = _match()
    assert (shape.attack_dir(pos, period) == 1).all()
    assert (shape.attack_dir(pos[:, ::-1], period) == -1).all()
    period2 = period.copy(); period2[:10] = 0
    assert (shape.attack_dir(pos, period2)[:10] == 0).all()


def test_smooth_ignores_nan():
    v = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan])
    s = shape.smooth(v, 3)
    assert s[1] == 2.0 and s[0] == 1.0 and np.isnan(s[4])


def test_team_features_are_antisymmetric_and_the_wider_team_scores_positive():
    pos, period = _match()
    f = shape.features(pos, period)
    g = shape.features(pos[:, ::-1].copy(), period)        # swap home and away
    ok = f["ok"] & g["ok"]
    assert ok.mean() > 0.9
    assert np.allclose(f["team"][ok], -g["team"][ok], atol=1e-4)
    assert np.allclose(f["dead"][ok], g["dead"][ok], atol=1e-4)
    sy = shape.TEAM_FEATS.index("sy")
    assert np.nanmean(f["team"][50:250, sy]) > 5 and np.nanmean(f["team"][350:450, sy]) < -5


def test_too_few_players_gives_no_evidence():
    pos, period = _match()
    pos[100:200, 1, 5:] = np.nan                           # away drops to 5 players
    f = shape.features(pos, period)
    assert not f["ok"][120:180].any()
    st, sd = shape.emissions(f, shape.MODELS)
    assert (st[120:180] == 0).all() and (sd[120:180] == 0).all()


def test_fit_models_learns_width_and_stillness_and_is_symmetric():
    pos, period = _match()
    n = len(period)
    truth = np.array(["H"] * (n // 2) + ["A"] * (n // 2 - 100) + ["D"] * 100)
    f = shape.features(pos, period)
    m = shape.fit_models([f], [truth], l2=1.0, step=1)
    assert float(m["team"]["b"]) == 0.0 and not m["team"]["mu"].any()
    st, sd = shape.emissions(f, m)
    live = f["ok"] & (truth != "D")
    assert shape.auc(st[live], truth[live] == "H") > 0.95
    assert shape.auc(sd[f["ok"]], truth[f["ok"]] == "D") > 0.95
    g = shape.features(pos[:, ::-1].copy(), period)
    assert np.allclose(shape.emissions(g, m)[0], -st, atol=1e-3)


def test_shipped_models_match_feature_lists_and_decoder_inputs():
    assert len(shape.MODELS["team"]["w"]) == len(shape.TEAM_FEATS)
    assert len(shape.MODELS["dead"]["w"]) == len(shape.DEAD_FEATS)
    pos, period = _match()
    d = shape.decoder_inputs(pos, period)
    assert set(d) == {"shape_team", "shape_dead", "edge_p90", "shape_ok", "sdead_pi"} and len(d["shape_team"]) == len(period)
    assert np.isfinite(d["shape_team"]).all() and np.abs(d["shape_dead"]).max() <= 4.0


def test_features_compute_positions_option_matches_collect():
    from possess import features as feat_mod

    def frames():
        for i in range(5):
            yield i, 1, {"players": {k: {"team": 1 + k % 2, "map_position": (float(k), float(i))} for k in range(8)},
                         "goalkeepers": {}, "ball": {}}, False

    f = feat_mod.compute(frames(), n=5, positions=True)
    assert np.array_equal(f["pos"], shape.collect(frames(), n=5), equal_nan=True)
    assert "pos" not in feat_mod.compute(frames(), n=5)


def _still_team(n=200, speed=0.0, sigma=0.0, fps=25.0, seed=0):
    """10 players in a 2 x 5 grid, 8 m apart, all moving along x at `speed`, plus position noise."""
    rng = np.random.default_rng(seed)
    p = np.full((n, shape.SLOTS, 2), np.nan, dtype=np.float32)
    gx, gy = np.meshgrid(np.arange(5) * 8.0 - 16, np.array([-8.0, 8.0]))
    t = np.arange(n)[:, None] / fps
    p[:, :10, 0] = gx.ravel()[None] + speed * t + rng.normal(0, sigma, (n, 10))
    p[:, :10, 1] = gy.ravel()[None] + rng.normal(0, sigma, (n, 10))
    return p


def test_player_speed_is_robust_to_position_jitter():
    """Fable QA: 0.05-0.10 m white noise must not read as running (frame-to-frame it was > 2 m/s)."""
    for sigma in (0.05, 0.10):
        still = shape._player_speed(_still_team(sigma=sigma), 25.0)
        assert np.nanmedian(still) < 0.4
        run = shape._player_speed(_still_team(speed=4.0, sigma=sigma), 25.0)
        assert abs(np.nanmedian(run) - 4.0) < 0.3


def test_player_speed_uses_the_frame_rate():
    for fps in (10.0, 25.0, 50.0):
        v = shape._player_speed(_still_team(n=int(8 * fps), speed=3.0, fps=fps), fps)
        assert abs(np.nanmedian(v) - 3.0) < 0.05


def test_deepest_player_is_dropped_only_inside_his_own_penalty_area():
    n = 60
    p = _still_team(n)                                     # 10 outfield players, x from -16 to 16
    sign = np.ones(n, dtype=np.int8)
    no_gk = shape.team_frame(p, sign)
    assert np.isfinite(no_gk["_u"]).sum(axis=1).min() == 10          # nobody dropped
    assert np.allclose(no_gk["lo_u"][5:-5], -16.0, atol=1e-4)
    p[:, 10] = (-45.0, 3.0)                                # a keeper in his own box
    with_gk = shape.team_frame(p, sign)
    assert np.isfinite(with_gk["_u"]).sum(axis=1).min() == 10         # 11 tracked, keeper dropped
    assert np.allclose(with_gk["dline"][5:-5], no_gk["dline"][5:-5], atol=1e-4)
    p[:, 10] = (-30.0, 3.0)                                # deepest player outside the box: a defender
    assert np.isfinite(shape.team_frame(p, sign)["_u"]).sum(axis=1).min() == 11


def test_unclear_attacking_direction_gives_no_evidence_and_is_reported():
    pos, period = _match()
    votes = shape.attack_votes(pos, period)
    assert votes[1][0] == 1 and votes[1][1] > 0.95
    mixed = pos.copy()
    mixed[:300] = mixed[:300, ::-1]                        # teams swap sides half way: a 50/50 vote
    v = shape.attack_votes(mixed, period)[1][1]
    assert v < shape.ATTACK_MIN_VOTE
    assert (shape.attack_dir(mixed, period) == 0).all()
    assert not shape.features(mixed, period)["ok"].any()


def test_team_context_adds_lead_lag_columns_and_stays_antisymmetric():
    # envelope L2-10: vu 2 s ahead / behind, change of width and length across +-2 s
    n, k = 200, int(shape.CTX_S * shape.FPS)
    T = np.random.default_rng(0).normal(size=(n, len(shape.TEAM_FEATS)))
    C = shape.team_context(T)
    assert C.shape == (n, len(shape.TEAM_FEATS) + len(shape.CTX_FEATS))
    assert np.array_equal(C[:, :T.shape[1]], T)
    vu, sy = T[:, shape.TEAM_FEATS.index("vu")], T[:, shape.TEAM_FEATS.index("sy")]
    assert np.allclose(C[k:-k, -4], vu[2 * k:]) and np.allclose(C[k:-k, -3], vu[:-2 * k])
    assert np.allclose(C[k:-k, -2], sy[2 * k:] - sy[:-2 * k])
    assert np.isnan(C[:k, -3]).all() and np.isnan(C[-k:, -4]).all()      # no neighbour: nan, not a guess
    assert np.allclose(shape.team_context(-T)[k:-k], -C[k:-k])


def test_context_model_is_shipped_used_by_default_and_falls_back_to_the_base_model():
    assert len(shape.MODELS["team_ctx"]["w"]) == len(shape.TEAM_FEATS) + len(shape.CTX_FEATS)
    assert shape.MODELS["team_ctx"]["b"] == 0 and not shape.MODELS["team_ctx"]["mu"].any()   # antisymmetric
    pos, period = _match()
    f = shape.features(pos, period)
    k = int(shape.CTX_S * shape.FPS)
    base = shape.emissions(f, shape.MODELS, ctx=0)[0]
    ctx = shape.emissions(f, shape.MODELS)[0]
    old = shape.emissions(f, {m: shape.MODELS[m] for m in ("team", "dead")})[0]   # models without team_ctx
    assert np.array_equal(old, base)
    assert np.array_equal(ctx[:k], base[:k]) and np.array_equal(ctx[-k:], base[-k:])   # edges: base model
    assert not np.array_equal(ctx[k:-k], base[k:-k])
    assert np.array_equal(shape.emissions(f, shape.MODELS)[1], shape.emissions(f, shape.MODELS, ctx=0)[1])


def test_context_model_sees_a_change_of_possession_earlier_than_the_base_model():
    # home has the ball until n/2, then away; with models fitted on that match the context model is
    # right on at least as many frames around the change
    pos, period = _match()
    f = shape.features(pos, period)
    truth = np.array(["H"] * 300 + ["A"] * 200 + ["D"] * 100)
    m = shape.fit_models([f], [truth], l2=1.0, step=1)
    assert "team_ctx" in m
    live = f["ok"] & (truth != "D")
    y = truth[live] == "H"
    right = [((shape.emissions(f, m, ctx=c)[0][live] > 0) == y).mean() for c in (0, 1)]
    assert right[1] >= right[0] and right[1] > 0.9


# ---------------------------------------------------------------- fast players (envelope L2-16)
def _two_teams(n=150, sprinters=0, speed=5.0):
    """Two standing teams of 10 (home on the -x side); `sprinters` home players run along +y."""
    pos = np.full((n, 2, shape.SLOTS, 2), np.nan, dtype=np.float32)
    pos[:, 0], pos[:, 1] = _still_team(n), _still_team(n)
    pos[:, 0, :, 0] -= 25.0
    pos[:, 1, :, 0] += 25.0
    pos[:, 0, 5:5 + sprinters, 1] += (speed * np.arange(n) / 25.0)[:, None]     # row y = +8, moving away
    return pos, np.ones(n, dtype=np.int8)


def test_row_quantile_matches_nanpercentile():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 28))
    v[rng.random(v.shape) < 0.3] = np.nan
    v[7] = np.nan
    got = shape._row_quantile(v, 0.9)
    assert np.isnan(got[7])
    ok = np.arange(50) != 7
    assert np.allclose(got[ok], np.nanpercentile(v[ok], 90, axis=1))


def test_fast_columns_see_three_sprinters_the_median_player_does_not():
    pos, period = _two_teams(sprinters=3)
    f = shape.features(pos, period)
    assert f["fast"].shape == (len(period), len(shape.FAST_FEATS))
    mid = slice(50, 100)
    assert np.all(np.abs(f["fast"][mid, 0] - 5.0) < 0.3)                 # ps90 = the sprinters
    assert np.all(f["dead"][mid, shape.DEAD_FEATS.index("ps_mean")] < 0.2)   # median player stands
    still = shape.features(*_two_teams(sprinters=0))
    assert np.all(still["fast"][mid] < 0.2)
    assert np.array_equal(still["dead"][mid], shape.features(*_two_teams(sprinters=0))["dead"][mid])


def test_fast_columns_do_not_depend_on_which_team_is_home():
    pos, period = _two_teams(sprinters=3)
    f, g = shape.features(pos, period), shape.features(pos[:, ::-1].copy(), period)
    assert np.allclose(f["fast"], g["fast"], equal_nan=True)


def test_fast_stoppage_model_default_fallbacks_and_direction():
    assert len(shape.MODELS["dead_fast"]["w"]) == len(shape.DEAD_FEATS) + len(shape.FAST_FEATS)
    assert (shape.MODELS["dead_fast"]["w"][-2:] < 0).all()               # fast players = play is live
    run, period = _two_teams(sprinters=3)
    f, s = shape.features(run, period), shape.features(*_two_teams(sprinters=0))
    mid = slice(50, 100)
    base = shape.emissions(f, shape.MODELS, fast=0)[1]
    assert np.array_equal(base, np.clip(shape.log_odds(shape.MODELS["dead"], f["dead"]), -4, 4))
    old = {k: v for k, v in shape.MODELS.items() if k != "dead_fast"}
    assert np.array_equal(shape.emissions(f, old)[1], base)              # models before L2-16
    assert np.array_equal(shape.emissions({k: v for k, v in f.items() if k != "fast"}, shape.MODELS)[1], base)
    now = shape.emissions(f, shape.MODELS, clip=50.0)[1]
    assert np.all(now[mid] < shape.emissions(s, shape.MODELS, clip=50.0)[1][mid])   # sprinters -> less "stopped"
    nan_fast = dict(f, fast=np.full_like(f["fast"], np.nan))             # no fast columns -> base model
    assert np.array_equal(shape.emissions(nan_fast, shape.MODELS)[1], base)


def test_fit_models_adds_the_fast_stoppage_model_only_with_fast_columns():
    pos, period = _match()
    f = shape.features(pos, period)
    truth = np.array(["H"] * 300 + ["A"] * 200 + ["D"] * 100)
    m = shape.fit_models([f], [truth], step=1)
    assert len(m["dead_fast"]["w"]) == len(shape.DEAD_FEATS) + len(shape.FAST_FEATS)
    m_old = shape.fit_models([{k: v for k, v in f.items() if k != "fast"}], [truth], step=1)
    assert "dead_fast" not in m_old and np.allclose(m_old["dead"]["w"], m["dead"]["w"])


# ----------------------------------------------------------------------------- L2-21 hidden-frame prior
def _posteriors(n, share, pi_tr, seed=0):
    """Log-odds of a calibrated model fitted with class share pi_tr, on a population with `share`."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < share
    x = rng.normal(np.where(y, 1.0, -1.0), 1.0)              # likelihood ratio log-odds = 2x
    return 2.0 * x + np.log(pi_tr / (1 - pi_tr)), y


def test_em_prior_recovers_the_class_share_without_labels():
    for share in (0.2, 0.4, 0.8):
        z, y = _posteriors(20000, share, 0.4)
        assert abs(shape.em_prior(z, 0.4) - y.mean()) < 0.02


def test_prior_shift_moves_hidden_frames_only_and_fills_frames_without_evidence(monkeypatch):
    monkeypatch.setattr(shape, "PRIOR_MIN_FRAMES", 1500)     # synthetic input is shorter than a match
    n = 12000
    z, _ = _posteriors(n, 0.8, 0.4)
    sd = np.clip(z, -4, 4)
    hidden = np.arange(n) < 6000
    ok = np.arange(n) % 10 != 0
    sd[~ok] = 0.0
    out = shape.prior_shift(sd, ok, hidden, 0.4)
    assert np.array_equal(out[~hidden], sd[~hidden])                       # visible frames untouched
    k = hidden & ok
    shift = (out[k] - sd[k])[np.abs(out[k]) < 4]
    assert shift.min() > 1.5 and np.ptp(shift) < 1e-9                      # logit(0.8) - logit(0.4) = 1.79
    assert np.allclose(out[hidden & ~ok], np.log(0.8 / 0.2), atol=0.15)    # no evidence -> the population prior
    assert np.abs(out).max() <= 4.0


def test_prior_shift_is_off_for_small_shifts_few_frames_and_old_models():
    z, _ = _posteriors(12000, 0.45, 0.4)                                   # shift ~0.2 < PRIOR_MIN_SHIFT
    sd = np.clip(z, -4, 4)
    ok, hidden = np.ones(12000, bool), np.ones(12000, bool)
    assert shape.prior_shift(sd, ok, hidden, 0.4) is sd
    z, _ = _posteriors(shape.PRIOR_MIN_FRAMES - 1, 0.9, 0.4)
    sd = np.clip(z, -4, 4)
    assert shape.prior_shift(sd, np.ones(len(sd), bool), np.ones(len(sd), bool), 0.4) is sd
    assert shape.dead_prior({"dead": {"w": 0}}) is None                    # models fitted before L2-21
    assert 0.3 < shape.dead_prior(shape.MODELS) < 0.5
    pos, period = _match()
    f = shape.features(pos, period)
    m = shape.fit_models([f], [np.array(["H"] * 300 + ["A"] * 200 + ["D"] * 100)], step=1)
    assert 0.0 < float(m["dead_fast"]["pi"]) < 1.0 and 0.0 < float(m["dead"]["pi"]) < 1.0


def test_decode_halves_applies_the_prior_only_with_sd_prior_and_the_shape_arrays(monkeypatch):
    from possess.decode import decode_halves

    monkeypatch.setattr(shape, "PRIOR_MIN_FRAMES", 1500)     # synthetic input is shorter than a match
    n = 6000
    z, _ = _posteriors(n, 0.85, 0.4, seed=3)
    sd = np.clip(np.convolve(z, np.ones(25) / 25, "same") - 1.2, -4, 4)    # mostly a little below 0: live
    feats = {"d_home": np.full(n, 30.0), "d_away": np.full(n, 30.0), "visible": np.zeros(n, bool),
             "dead": np.zeros(n, bool), "shape_team": np.zeros(n), "shape_dead": sd}
    kw = dict(hard_dead=1, soft=1, w_sdead=0.3, c_stop=12.0)
    period = np.ones(n, dtype=np.int8)
    base = decode_halves(period, feats=dict(feats), **kw)
    extra = dict(feats, shape_ok=np.ones(n, bool), sdead_pi=0.4)
    assert np.array_equal(decode_halves(period, feats=dict(extra), sd_prior=0, **kw), base)
    assert np.array_equal(decode_halves(period, feats=dict(feats), sd_prior=1, **kw), base)   # no arrays: skipped
    on = decode_halves(period, feats=dict(extra), sd_prior=1, **kw)
    assert (on == "D").mean() > (base == "D").mean() + 0.2
    assert shape.DEC_SHAPE["sd_prior"] == 1 and "sd_prior" not in shape.DEC_SHAPE_L2_20
    monkeypatch.setattr(shape, "PRIOR_MIN_FRAMES", 40000)    # QA L2-21: short inputs get no estimate
    assert np.array_equal(decode_halves(period, feats=dict(extra), sd_prior=1, **kw), base)
