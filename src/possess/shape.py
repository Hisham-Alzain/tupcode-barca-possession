"""Ball-free evidence from player positions only (research plan P5, envelope L2-7-shape-fable).

While the ball is hidden (41% of the challenge match, blocks of 5-60 s) the decoder has no team
evidence and the stoppage detector just holds its last state. This module turns the 22 player
positions into two per-frame log-odds a decoder can add as emissions on hidden frames:

  team   log-odds "home has the ball" vs "away has it" (given play is live)
  dead   log-odds "play is stopped" vs "play is live"

matchlens_start: MatchLens attacking_direction.py (footballtactics backend/events, not in this
repo): per-period attacking direction from which side each team's goalkeeper stands on. Ported
idea, changed for anonymous tracks: `attack_dir` uses the sign of the mean home-minus-away mean x
over the period (the goalkeepers alone shift it by ~8 m), no goalkeeper identity needed.
Nothing in MatchLens does team-shape possession evidence (checked possession.py, impulse.py,
kinematics.py, deadball.py: all need the ball).

Features (swarm F6, F7: relative, count-robust, smoothed over seconds; frames with < MIN_PLAYERS
of either team are ignored). Each team is described in its own attacking frame (u = x toward the
opponent's goal), outfield players only (the deepest player is dropped as the goalkeeper when
he stands in his own penalty area):
  cu, su, sy     centroid advancement, length (std u) and width (std y)
  dline, aline   defensive line (mean of the 3 deepest) and attacking line (mean of the 2 highest)
  vu, cs         centroid velocity along u and centroid speed
  ps             median player speed over a 0.8 s lag (players matched by nearest neighbour: no IDs)
  enc            share of the opponent's outfield players inside this team's bounding box
  box, border    players in the opponent's penalty area; smallest distance of a player to a line
Team model inputs are home-minus-away differences (antisymmetric: swapping the teams flips the
sign, so the model needs no intercept and cannot favour a side). Stoppage model inputs are
symmetric (sums, maxima) at two time scales; since envelope L2-16 also the speed of the fast
players (FAST_FEATS: 90th percentile over all tracked players, model "dead_fast").
matchlens_start for L2-16: MatchLens deadball.py / detectors/set_piece_detector.py were checked
(the port is stoppage_ml.py): both read the ball (stationary ball, ball out, restart kick) and
have no player-only stoppage evidence, so the fast-player columns extend this module's own model.

Models: logistic regression in numpy (no new dependency), fitted on the external tune matches
only (scripts/tune_shape.py; leave-one-match-out for every reported number).
"""
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

FPS = 25.0
SLOTS = 14               # players kept per team per frame (extra ones are dropped)
MIN_PLAYERS = 8          # swarm F7: tracking degrades where the ball is lost
HALF_L, HALF_W = 52.5, 34.0
BOX_L, BOX_W = 16.5, 20.16
ATTACK_MIN_VOTE = 0.65   # share of frames that must agree on the side a team defends
SPEED_LAG_S = 0.8        # player speed = displacement over this lag (robust to position jitter)
V_MAX = 12.0             # m/s; a larger matched displacement is a new or swapped track


def collect(frames: Iterable[Tuple[int, int, dict, bool]], n: int) -> np.ndarray:
    """One pass over the frames -> positions [n, 2 teams, SLOTS, 2] float32 (nan = empty slot).

    Slot order carries no identity. Same frame sources as possess.features.compute.
    """
    pos = np.full((n, 2, SLOTS, 2), np.nan, dtype=np.float32)
    for fid, _per, tracks, _present in frames:
        if fid >= n:
            continue
        k = [0, 0]
        for grp in ("players", "goalkeepers"):
            for r in tracks[grp].values():
                t = r.get("team")
                if t in (1, 2) and k[t - 1] < SLOTS:
                    pos[fid, t - 1, k[t - 1]] = r["map_position"]
                    k[t - 1] += 1
    return pos


def attack_votes(pos: np.ndarray, period: np.ndarray) -> Dict[int, Tuple[int, float]]:
    """{period: (home attacking direction +1 / -1, vote share)}: every frame votes with the sign
    of mean home x minus mean away x (the side a team stands on, goalkeepers included; same idea
    as epts.AtdMatch._estimate_sides, per frame instead of one mean so a few bad frames cannot
    flip it). The vote share (0.5-1) shows how clear the half is: print it, a share near 0.5 means
    the sign cannot be trusted."""
    with np.errstate(invalid="ignore"), _quiet():
        diff = np.nanmean(pos[:, 0, :, 0], axis=1) - np.nanmean(pos[:, 1, :, 0], axis=1)
    out = {}
    for p in np.unique(period[period > 0]):
        d = diff[(period == p) & np.isfinite(diff)]
        share = float((d < 0).mean()) if len(d) else 0.5     # home on the -x side -> attacks +x
        out[int(p)] = (1 if share >= 0.5 else -1, max(share, 1.0 - share))
    return out


def attack_dir(pos: np.ndarray, period: np.ndarray, min_vote: float = None) -> np.ndarray:
    """+1 where home attacks +x, -1 where it attacks -x (per frame); 0 outside the periods and in
    a period whose vote share is below `min_vote` (no shape evidence there rather than a guess)."""
    min_vote = ATTACK_MIN_VOTE if min_vote is None else min_vote
    out = np.zeros(len(period), dtype=np.int8)
    for p, (sign, share) in attack_votes(pos, period).items():
        if share >= min_vote:
            out[period == p] = sign
    return out


class _quiet:
    def __enter__(self):
        import warnings

        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._c.__exit__(*a)


def smooth(v: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling mean over w frames ignoring nan (nan where the window has no value)."""
    ok = np.isfinite(v)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    n = len(v)
    lo = np.clip(np.arange(n) - w // 2, 0, n)
    hi = np.clip(np.arange(n) + w // 2 + 1, 0, n)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)


def _player_speed(p: np.ndarray, fps: float, lag_s: float = SPEED_LAG_S, chunk: int = 20000,
                  each: bool = False):
    """Median player speed (m/s) per frame for one team, [n, SLOTS, 2] -> [n]. No identities:
    each player at t + h is matched to the nearest player of the same team at t - h, and the speed
    is that displacement over the lag 2h ~ `lag_s` (Fable QA 2026-11-14: frame-to-frame speed
    turned 0.05 m of position jitter into > 1 m/s; over 0.8 s the same jitter is < 0.2 m/s).
    each=True: (median [n], every player's speed [n, SLOTS], nan = empty slot or no match)."""
    n = len(p)
    h = max(1, int(round(lag_s * fps / 2)))
    dt = 2 * h / fps
    out = np.full(n, np.nan)
    sp = np.full((n, p.shape[1]), np.nan, dtype=np.float32) if each else None
    for s in range(h, n - h, chunk):
        e = min(n - h, s + chunk)
        a, b = p[s + h:e + h], p[s - h:e - h]
        d = np.hypot(a[:, :, None, 0] - b[:, None, :, 0], a[:, :, None, 1] - b[:, None, :, 1])
        d = np.where(np.isnan(d), np.inf, d).min(axis=2)
        d[(d > V_MAX * dt) | np.isnan(a[:, :, 0])] = np.nan
        with _quiet():
            out[s:e] = np.nanmedian(d, axis=1) / dt
        if each:
            sp[s:e] = d / dt
    return (out, sp) if each else out


def _row_quantile(v: np.ndarray, q: float) -> np.ndarray:
    """Quantile q of every row ignoring nan (linear interpolation, as np.nanpercentile; nan for
    an all-nan row)."""
    s = np.sort(np.asarray(v, float), axis=1)                 # nan last
    c = np.isfinite(s).sum(axis=1)
    x = q * np.maximum(c - 1, 0)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, np.maximum(c - 1, 0))
    rows = np.arange(len(s))
    out = s[rows, lo] + (x - lo) * (s[rows, hi] - s[rows, lo])
    return np.where(c > 0, out, np.nan)


def team_frame(p: np.ndarray, sign: np.ndarray, fps: float = FPS) -> Dict[str, np.ndarray]:
    """Per-frame descriptors of one team. p [n, SLOTS, 2]; sign +1/-1/0 = its attacking direction."""
    n = len(p)
    u = p[:, :, 0] * sign[:, None].astype(np.float32)
    y = p[:, :, 1]
    cnt = np.isfinite(u).sum(axis=1)
    ok = (cnt >= MIN_PLAYERS) & (sign != 0)
    rows = np.arange(n)
    # goalkeeper = the deepest player, dropped only when he stands in his own penalty area
    # (Fable QA: when the keeper is not tracked the deepest player is a defender and stays)
    gk = np.argmin(np.where(np.isfinite(u), u, np.inf), axis=1)
    is_gk = (u[rows, gk] < -(HALF_L - BOX_L)) & (np.abs(y[rows, gk]) < BOX_W)
    uo, yo = u.copy(), y.copy()
    uo[rows[is_gk], gk[is_gk]] = np.nan
    yo[rows[is_gk], gk[is_gk]] = np.nan
    with _quiet():
        f = {"cu": np.nanmean(uo, axis=1), "su": np.nanstd(uo, axis=1), "sy": np.nanstd(yo, axis=1),
             "cy": np.nanmean(yo, axis=1)}
        uso = np.sort(uo, axis=1)                          # outfield only, nan last
        f["dline"] = uso[:, 0:3].mean(axis=1)
        co = np.isfinite(uo).sum(axis=1)
        top = np.take_along_axis(uso, np.clip(co[:, None] - np.array([[1, 2]]), 0, SLOTS - 1), axis=1)
        f["aline"] = top.mean(axis=1)
        f["box"] = ((uo > HALF_L - BOX_L) & (np.abs(yo) < BOX_W)).sum(axis=1).astype(float)
        f["border"] = np.nanmin(np.minimum(HALF_L - np.abs(p[:, :, 0]), HALF_W - np.abs(y)), axis=1)
        f["lo_u"], f["hi_u"] = np.nanmin(uo, axis=1), np.nanmax(uo, axis=1)
        f["lo_y"], f["hi_y"] = np.nanmin(yo, axis=1), np.nanmax(yo, axis=1)
    w = int(fps)
    cu_s, cy_s = smooth(np.where(ok, f["cu"], np.nan), w), smooth(np.where(ok, f["cy"], np.nan), w)
    f["vu"] = np.gradient(cu_s) * fps
    f["cs"] = np.hypot(f["vu"], np.gradient(cy_s) * fps)
    f["ps"], sp = _player_speed(p, fps, each=True)
    for k in f:
        f[k] = np.where(ok, f[k], np.nan)
    f["_u"], f["_y"], f["ok"], f["_sp"] = uo, yo, ok, sp
    return f


def _enclosed(box: Dict[str, np.ndarray], other: Dict[str, np.ndarray]) -> np.ndarray:
    """Share of `other`'s outfield players inside `box` team's bounding box (opposite u frames)."""
    u, y = -other["_u"], other["_y"]
    inside = (u >= box["lo_u"][:, None]) & (u <= box["hi_u"][:, None]) & \
             (y >= box["lo_y"][:, None]) & (y <= box["hi_y"][:, None])
    c = np.isfinite(u).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, inside.sum(axis=1) / c, np.nan)


TEAM_FEATS = ["cu", "vu", "su", "sy", "area", "dline", "aline", "ps", "cs", "enc", "box", "border",
              "vu5", "cu_x_sy", "cu_x_su"]
DEAD_FEATS = ["ps_mean", "ps_max", "cs_max", "ps5", "ps15", "border_min", "border5", "box_max", "su_mean",
              "sy_mean", "abs_cu", "sep", "ps_slope", "ps_min5", "ps_x_border"]
# L2-16: the fast tail of the player speeds. DEAD_FEATS only know the median player: during a
# stoppage most players walk, but so they do in slow build-up play; what a stoppage lacks is the
# two or three players who sprint (carrier, presser, runner). ps90 = 90th percentile of the speeds
# of all tracked players of both teams (about the third fastest of 22), over 1 s and 5 s.
FAST_FEATS = ["ps90", "ps90_5"]
FAST_Q = 0.9             # set before any scoring run (p75, top-3 mean, max: same score within noise)
DEAD_FAST = 1            # default of `emissions`: stoppage model with FAST_FEATS when the models carry it


def features(pos: np.ndarray, period: np.ndarray, fps: float = FPS) -> Dict[str, np.ndarray]:
    """{"team": [n, len(TEAM_FEATS)], "dead": [n, len(DEAD_FEATS)], "fast": [n, len(FAST_FEATS)],
    "ok": [n] bool}.

    Team columns are home-minus-away (antisymmetric), smoothed over 1 s (vu5: 5 s). nan rows where
    either team has < MIN_PLAYERS or the frame is outside the periods.
    """
    d = attack_dir(pos, period)
    h, a = team_frame(pos[:, 0], d, fps), team_frame(pos[:, 1], -d, fps)
    h["enc"], a["enc"] = _enclosed(h, a), _enclosed(a, h)
    for t in (h, a):
        t["area"] = t["su"] * t["sy"]
    ok = h["ok"] & a["ok"]
    w1, w5, w15 = int(fps), int(5 * fps), int(15 * fps)

    def sm(v, w=w1):
        return smooth(np.where(ok, v, np.nan), w)

    T = {k: sm(h[k] - a[k]) for k in ("cu", "vu", "su", "sy", "area", "dline", "aline", "ps", "cs", "enc",
                                      "box", "border")}
    T["vu5"] = sm(h["vu"] - a["vu"], w5)
    # width / length of the team that is further up the pitch (territory x shape interaction)
    T["cu_x_sy"] = T["cu"] * sm(h["sy"] + a["sy"]) / 20.0
    T["cu_x_su"] = T["cu"] * sm(h["su"] + a["su"]) / 20.0
    ps_mean = sm((h["ps"] + a["ps"]) / 2)
    border = np.fmin(h["border"], a["border"])
    D = {"ps_mean": ps_mean, "ps_max": sm(np.fmax(h["ps"], a["ps"])), "cs_max": sm(np.fmax(h["cs"], a["cs"])),
         "ps5": sm((h["ps"] + a["ps"]) / 2, w5), "ps15": sm((h["ps"] + a["ps"]) / 2, w15),
         "border_min": sm(border), "border5": sm(border, w5),
         "box_max": sm(np.fmax(h["box"] + _mirror_box(a), a["box"] + _mirror_box(h))),
         "su_mean": sm((h["su"] + a["su"]) / 2), "sy_mean": sm((h["sy"] + a["sy"]) / 2),
         "abs_cu": np.abs(sm((h["cu"] - a["cu"]) / 2)), "sep": sm(h["cu"] + a["cu"])}
    D["ps_slope"] = np.gradient(D["ps5"]) * fps
    D["ps_min5"] = _rolling_min(ps_mean, w5)
    D["ps_x_border"] = D["ps5"] * np.clip(D["border5"], -2, 5)
    out_ok = ok & (period > 0)
    Tm = np.stack([T[k] for k in TEAM_FEATS], axis=1)
    Dm = np.stack([D[k] for k in DEAD_FEATS], axis=1)
    out_ok &= np.isfinite(Tm).all(axis=1) & np.isfinite(Dm).all(axis=1)
    p90 = _row_quantile(np.concatenate([h["_sp"], a["_sp"]], axis=1), FAST_Q)
    Fm = np.stack([sm(p90), sm(p90, w5)], axis=1)             # FAST_FEATS (nan rows fall back to the base model)
    # "p90" (L2-18): the same fast-player speed with no 1 s / 5 s smoothing and no `ok` mask, read by
    # the edge post-pass (possess/edges.py, = edges.edge_columns(pos)["p90"]); not a model column
    return {"team": np.where(out_ok[:, None], Tm, np.nan), "dead": np.where(out_ok[:, None], Dm, np.nan),
            "fast": np.where(out_ok[:, None], Fm, np.nan), "ok": out_ok, "p90": p90.astype(np.float32)}


def _mirror_box(t: Dict[str, np.ndarray]) -> np.ndarray:
    """Players of team t inside their OWN penalty area (defending a set piece)."""
    return ((t["_u"] < -(HALF_L - BOX_L)) & (np.abs(t["_y"]) < BOX_W)).sum(axis=1).astype(float)


def _rolling_min(v: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling minimum, nan ignored (nan where the window is all nan). Subsampled x5."""
    step = 5
    x = np.where(np.isfinite(v), v, np.inf)
    n = len(x)
    m = np.full(n, np.inf)
    for s in range(-(w // 2), w // 2 + 1, step):
        lo, hi = max(0, -s), min(n, n - s)
        m[lo:hi] = np.minimum(m[lo:hi], x[lo + s:hi + s])
    return np.where(np.isfinite(m), m, np.nan)


# ----------------------------------------------------------------------------- logistic regression
def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, intercept: bool = True,
                 iters: int = 30) -> Dict[str, np.ndarray]:
    """L2 logistic regression by Newton steps on standardised columns. Returns mu, sd, w, b.

    intercept=False also keeps the columns un-centred (mu = 0), so f(-x) = -f(x): used for the
    antisymmetric team model.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = X.mean(axis=0) if intercept else np.zeros(X.shape[1])
    sd = np.sqrt(((X - mu) ** 2).mean(axis=0)) + 1e-9
    Z = (X - mu) / sd
    if intercept:
        Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    if intercept:
        reg[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg * w
        Hs = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
        step = np.linalg.solve(Hs, g)
        w = w - step
        if np.abs(step).max() < 1e-7:
            break
    return {"mu": mu, "sd": sd, "w": w[:X.shape[1]], "b": np.array(w[-1] if intercept else 0.0)}


def log_odds(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """Model log-odds per row; 0 (no evidence) where a row has nan."""
    z = ((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"]
    return np.where(np.isfinite(z), z, 0.0)


CTX_S = 2.0              # s; lead / lag of the context columns (set before any scoring run, not tuned)
CTX_FEATS = ["vu_lead", "vu_lag", "dsy", "dsu"]
TEAM_CTX = 1             # default of `emissions`: use the context team model when the models carry it


def _shift(v: np.ndarray, k: int) -> np.ndarray:
    """out[t] = v[t + k] (nan where t + k is outside the array)."""
    o = np.full(len(v), np.nan)
    if k > 0:
        o[:-k] = v[k:]
    elif k < 0:
        o[-k:] = v[:k]
    else:
        o[:] = v
    return o


def team_context(T: np.ndarray, fps: float = FPS, lag_s: float = CTX_S) -> np.ndarray:
    """Team columns [n, len(TEAM_FEATS)] plus what the teams do `lag_s` later and earlier
    (envelope L2-10): the players react to a change of possession 1-2 s after it, so the centroid
    velocity difference 2 s ahead (vu_lead) and behind (vu_lag), and the change of width (dsy)
    and length (dsu) across +-2 s say who has the ball now. Offline only (reads the future).
    Leave-one-match-out on the tune split, ball-hidden live frames: right team 0.854 -> 0.886
    (true possessions of 4-10 s: 0.759 -> 0.824). Computed from the cached base columns, so no
    feature cache changes. Rows whose +-lag neighbours are missing are nan (the caller falls
    back to the base model there). A lag that crosses a period break reads the other half for
    `lag_s` seconds; frames outside the periods are nan anyway.
    """
    k = max(1, int(round(lag_s * fps)))
    vu, su, sy = (T[:, TEAM_FEATS.index(c)] for c in ("vu", "su", "sy"))
    ctx = [_shift(vu, k), _shift(vu, -k), _shift(sy, k) - _shift(sy, -k), _shift(su, k) - _shift(su, -k)]
    return np.hstack([T, np.stack(ctx, axis=1)])


def fit_models(feats: Iterable[Dict[str, np.ndarray]], truths: Iterable[np.ndarray], l2: float = 10.0,
               step: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit the team and stoppage models on labelled matches (every `step`-th valid frame).

    team: live frames, y = truth is H; each row is also added with flipped sign and label, so the
    fit is exactly antisymmetric. team_ctx: the same on `team_context` columns (L2-10).
    dead: all H/A/D frames, y = truth is D. dead_fast: the same on DEAD_FEATS + FAST_FEATS (L2-16;
    only when the feature dicts carry "fast").
    """
    Xt, yt, Xc, yc, Xd, yd, Xf, yf = [], [], [], [], [], [], [], []
    for f, truth in zip(feats, truths):
        ok = f["ok"].copy()
        ok[np.arange(len(ok)) % step != 0] = False
        live = ok & np.isin(truth, ["H", "A"])
        Xt.append(f["team"][live]); yt.append(truth[live] == "H")
        C = team_context(f["team"])
        lc = live & np.isfinite(C).all(axis=1)
        Xc.append(C[lc]); yc.append(truth[lc] == "H")
        m = ok & np.isin(truth, ["H", "A", "D"])
        Xd.append(f["dead"][m]); yd.append(truth[m] == "D")
        if "fast" in f:
            F = np.hstack([f["dead"], f["fast"]])
            mf = m & np.isfinite(F).all(axis=1)
            Xf.append(F[mf]); yf.append(truth[mf] == "D")

    def antisym(X, y):
        X, y = np.vstack(X), np.concatenate(y).astype(float)
        return fit_logistic(np.vstack([X, -X]), np.concatenate([y, 1 - y]), l2, intercept=False)

    out = {"team": antisym(Xt, yt), "team_ctx": antisym(Xc, yc),
           "dead": fit_logistic(np.vstack(Xd), np.concatenate(yd).astype(float), l2, intercept=True)}
    out["dead"]["pi"] = np.array(float(np.concatenate(yd).mean()))       # training share of D (L2-21, prior_shift)
    if Xf:
        out["dead_fast"] = fit_logistic(np.vstack(Xf), np.concatenate(yf).astype(float), l2, intercept=True)
        out["dead_fast"]["pi"] = np.array(float(np.concatenate(yf).mean()))
    return out


# ----------------------------------------------------------------------------- hidden-frame prior (L2-21)
# The stoppage models are fitted on ALL frames of the tune matches (share of D = "pi", 0.40), so
# their log-odds carry that prior. Where the ball is hidden the true share can be very different:
# a broadcast loses the ball mostly in stoppages (bench variant real2, state-coupled hiding: 0.71-
# 0.83 of the hidden frames are D), and the uncorrected model then under-calls D there (predicted
# 0.55-0.66; observed log-odds on real2's hidden frames sit 1.5-2.5 above the model's). The share
# of D among the hidden frames of the match at hand is estimated from the model's own posteriors,
# with no labels (EM of Saerens, Latinne & Decaestecker 2002, "Adjusting the outputs of a classifier
# to new a priori probabilities"), and the log-odds of the hidden frames move by
# logit(new share) - logit(pi). On real2 the estimate is 0.700 / 0.803 / 0.805 against a true
# 0.695 / 0.807 / 0.797. Hidden frames with no shape evidence (a team with < MIN_PLAYERS tracked:
# camera close-ups, log-odds 0 until now) get the hidden population's own prior, logit(new share).
# Under random hiding (real1, original bench) the estimate only wobbles around pi (shifts +0.29 /
# -0.08 / -0.00, costing 0.004): a shift smaller than PRIOR_MIN_SHIFT is estimation noise and is
# not applied (labels byte-identical there). 0.6 was set as twice the largest of those three,
# before any gated run; over 20 random masks per tune match the largest |shift| is 0.49 (sd 0.11-
# 0.16), real2 gives 1.20-1.82 (two seeds, also with position noise and player loss >= 0.97).
# One estimate per match (all periods): per half the random-hiding shifts reach 1.4.
# matchlens_start: nothing in MatchLens estimates a class prior (deadball.py, stoppage_ml.py,
# detectors/set_piece_detector.py checked: fixed thresholds on the ball); extends this module's model.
PRIOR_MIN_SHIFT = 0.6     # log-odds; smaller estimated shifts are not applied
# Validated on FULL matches only (L2-21 QA finding 1): per half the random-hiding shifts already
# reach 1.4 > PRIOR_MIN_SHIFT, so shorter inputs (one half, a clip) must not get an estimate.
# 40000 frames (~27 min of hidden play with evidence) is above a typical half's share and below
# every full match (external tune matches and the provided match all exceed it).
PRIOR_MIN_FRAMES = 40000  # hidden frames with shape evidence needed for an estimate


def em_prior(z: np.ndarray, pi_tr: float, iters: int = 200) -> float:
    """Share of the positive class in the population whose posterior log-odds are `z`, for a model
    fitted with class share `pi_tr` (EM re-estimation, Saerens et al. 2002; no labels)."""
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
    pi = float(pi_tr)
    for _ in range(iters):
        a = pi / pi_tr * p
        b = (1.0 - pi) / (1.0 - pi_tr) * (1.0 - p)
        new = float((a / (a + b)).mean())
        if abs(new - pi) < 1e-6:
            break
        pi = new
    return float(min(max(pi, 0.01), 0.99))


def prior_shift(sdead: np.ndarray, ok: np.ndarray, hidden: np.ndarray, pi_tr: float,
                min_shift: float = None, clip: float = 4.0) -> np.ndarray:
    """Stoppage log-odds with the class prior of the hidden frames re-estimated (see above).
    sdead: `emissions` output; ok: frames with shape evidence; hidden: ball hidden, inside a period.
    Returned unchanged when there are too few frames or the estimated shift is below `min_shift`."""
    min_shift = PRIOR_MIN_SHIFT if min_shift is None else min_shift
    k = hidden & ok
    if k.sum() < PRIOR_MIN_FRAMES or not 0.0 < pi_tr < 1.0:
        return sdead
    pi = em_prior(sdead[k], pi_tr)
    shift = np.log(pi / (1.0 - pi)) - np.log(pi_tr / (1.0 - pi_tr))
    if abs(shift) < min_shift:
        return sdead
    out = np.array(sdead, float)
    out[k] = np.clip(out[k] + shift, -clip, clip)
    out[hidden & ~ok] = np.clip(np.log(pi / (1.0 - pi)), -clip, clip)
    return out


def emissions(f: Dict[str, np.ndarray], models: Dict[str, Dict[str, np.ndarray]],
              clip: float = 4.0, ctx: int = None, fps: float = FPS,
              fast: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """(shape_team, shape_dead) per-frame log-odds for decode.viterbi, clipped to +-clip.

    ctx (default TEAM_CTX): team log-odds from the context model (`team_context`, models
    ["team_ctx"]) where its lead / lag rows exist, from the base model elsewhere. ctx=0 or models
    without "team_ctx": the base model only (L2-7 behaviour).
    fast (default DEAD_FAST): stoppage log-odds from the model with the fast-player columns
    (models["dead_fast"], f["fast"]; envelope L2-16) where those are finite, from the base
    stoppage model elsewhere. fast=0 or no "dead_fast" / "fast": the base model only (before L2-16).
    """
    team = log_odds(models["team"], f["team"])
    if (TEAM_CTX if ctx is None else ctx) and "team_ctx" in models:
        m = models["team_ctx"]
        z = ((team_context(f["team"], fps) - m["mu"]) / m["sd"]) @ m["w"] + m["b"]
        team = np.where(np.isfinite(z), z, team)
    dead = log_odds(models["dead"], f["dead"])
    if (DEAD_FAST if fast is None else fast) and "dead_fast" in models and "fast" in f:
        m = models["dead_fast"]
        z = ((np.hstack([f["dead"], f["fast"]]) - m["mu"]) / m["sd"]) @ m["w"] + m["b"]
        dead = np.where(np.isfinite(z), z, dead)
    return (np.clip(team, -clip, clip), np.clip(dead, -clip, clip))


# Final models: fitted on all three tune matches (Metrica G1, G2, IDSSE 1) by
# `scripts/tune_shape.py fit` (l2 = 10, every 5th frame; features v2 after the Fable QA fixes). Column order = TEAM_FEATS / DEAD_FEATS.
# Reported scores never use these: they use leave-one-match-out fits (scripts/tune_shape.py).
MODELS = {
    "team": {
        "mu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "sd": [31.209, 3.3786, 6.4182, 3.9593, 109.29, 22.456, 33.747, 0.58476, 0.69448, 0.35982, 1.2772, 5.3736, 3.0024, 36.816, 43.629],
        "w": [6.0852, 0.10263, -0.45284, 0.64839, 1.055, -2.9208, -2.0641, -0.48155, -0.1885, 0.23883, -0.024825, -0.49181, 0.76747, 0.7334, -2.0407],
        "b": [0.0],
    },
    "team_ctx": {      # TEAM_FEATS + CTX_FEATS (envelope L2-10), same fit
        "mu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "sd": [31.216, 3.3791, 6.4196, 3.9587, 109.3, 22.461, 33.755, 0.58483, 0.6944, 0.35989, 1.2775, 5.3744, 3.0024, 36.825, 43.64, 3.3972, 3.279, 1.8534, 2.6904],
        "w": [4.7695, 0.0077204, -0.50582, 1.0574, 1.1842, -1.3811, -2.4177, -0.2274, -0.30759, 0.2117, -0.096668, -0.42939, 1.2476, 0.34606, -1.109, 0.47679, -1.0587, 0.66141, 0.41312],
        "b": [0.0],
    },
    "dead": {
        "mu": [1.6233, 1.8288, 1.6743, 1.6233, 1.6235, 2.3739, 2.3739, 2.4023, 12.684, 12.246, 14.952, -9.1784, -0.00019226, 1.1833, 4.0637],
        "sd": [0.93247, 1.0093, 1.1488, 0.85398, 0.69643, 2.6184, 2.4471, 4.2092, 2.7406, 2.6432, 10.802, 4.6167, 0.19776, 0.63295, 4.5282],
        "w": [-0.88502, 0.41148, 0.11908, -1.9644, -1.796, -0.38307, 1.0674, 0.029198, 0.4793, -0.77012, -0.25851, 0.13585, -0.66227, 1.9778, -1.9041],
        "b": [-1.4514],
        "pi": [0.40538],   # share of D among the fitted frames (L2-21, prior_shift)
    },
    "dead_fast": {     # DEAD_FEATS + FAST_FEATS (envelope L2-16), same fit
        "mu": [1.6233, 1.8288, 1.6743, 1.6233, 1.6235, 2.3739, 2.3739, 2.4023, 12.684, 12.246, 14.952, -9.1784, -0.00019226, 1.1833, 4.0637, 2.8319, 2.832],
        "sd": [0.93247, 1.0093, 1.1488, 0.85398, 0.69643, 2.6184, 2.4471, 4.2092, 2.7406, 2.6432, 10.802, 4.6167, 0.19776, 0.63295, 4.5282, 1.3579, 1.2498],
        "w": [-1.5623, 1.3918, 0.43304, -0.2205, -1.4246, -0.31564, 0.91753, 0.096262, 0.48485, -0.77081, -0.14714, 0.1088, -0.58128, 1.7254, -1.7233, -0.62171, -1.816],
        "b": [-1.3068],
        "pi": [0.40538],
    },
}
MODELS = {m: {k: (np.asarray(v, float) if k not in ("b", "pi") else np.asarray(v[0], float)) for k, v in d.items()}
          for m, d in MODELS.items()}
# Decoder overrides on top of the no-shape decoder (pipeline.DEC_BASE), chosen on the tune split
# with leave-one-match-out models after the Fable QA fixes: 0.4932 vs 0.3976 without shape
# (eval/results/tune_shape_grid4.json); noise / player-dropout gate in tune_shape_gate.json.
# L2-13 (decided on the realistic bench `real1`, leave-one-match-out; scripts/tune_shape.py grid 5):
# w_live=0.0, sdead_vis=2.5. The ball detector's "live" verdict is no evidence (it misses a dead
# ball that moves) and the player-only stoppage model, which never reads the ball, weighs 2.5x on
# visible frames. real1 0.5276 -> 0.5731, original bench 0.5845 -> 0.6002. Before: DEC_SHAPE_L2_10.
DEC_SHAPE_L2_10 = dict(w_shape=0.05, w_sdead=0.3, c_stop=12.0, gap_pos=-1.0, shape_vis=1.0, sdead_vis=1.0,
                       w_dead=1.0, w_live=-1.0, duel=0, smooth_touch=0,
                       w_rule=0.3, cap=1.5)    # = pipeline.DEC_BASE, spelled out so the old configs still reproduce
DEC_SHAPE_L2_13 = dict(DEC_SHAPE_L2_10, sdead_vis=2.5, w_live=0.0)
# L2-14 (decided on `real1`, leave-one-match-out shape models; scripts/tune_shape.py grid 6,
# envelope L2-14-next-fable.out.md): cap=1.0. Ball proximity is counted three times on visible
# frames (distance emission, frame rule, last touch) and produces possessions that do not exist:
# of 347 spurious predicted possessions on real1, 101 sat inside a true stoppage (a player walking
# the dead ball to the spot), 83 were fragments and 63 the wrong team. The distance emission now
# stops grading beyond cap = 1 m = the control-zone radius (team_control r_pz, MatchLens
# possession.py / Vidal-Codina 2.3) instead of 1.5 m: an opponent 1.2 m from a ball at a player's
# feet is no evidence. real1 0.5731 -> 0.5834, original bench 0.6002 -> 0.6110, every match up.
# NOT adopted: w_rule=0.0 (frame-rule emission off). On the bench it adds as much again (with
# cap=1.0: real1 0.5950, original 0.6192, labels still home / away symmetric), but it fails the
# transfer check: on the provided match's input it removes 14% of the predicted possessions
# (24% with cap=1.0) against 0-3% (1-7%) on the bench; cap=1.0 alone removes 3% on both.
# The no-shape decoder (pipeline.DEC_BASE) keeps cap=1.5.
DEC_SHAPE_L2_14 = dict(DEC_SHAPE_L2_13, cap=1.0)
# L2-15 (decided on `real1`, leave-one-match-out shape models; envelope L2-15-segment-fable.out.md):
# duel=2. A ball impulse with both teams within r_touch no longer clears the touch hold: the team
# of the last uncontested touch keeps it (decode.touch_hold; the MatchLens / Vidal-Codina duel
# rule that team_control already applies). Diagnosis on idsse1 (356 predicted vs 511 true
# possessions): 293 of the 511 are shorter than 4 s, 105 of those hidden; the visible ones are
# found at the right frame by the touch detector (new team's touch - true switch: median 0.00 s,
# as on Metrica: the two sources define the switch alike) but contested touches wiped the
# holder's evidence. Truth-team oracle on duel touches only: +0.025. real1 0.5834 -> 0.5935,
# original bench 0.6110 -> 0.6205, every match up on both. Nothing is learned.
# NOT adopted (same score within noise, more parts): a learned touch-team model (0.596-0.603),
# a distance margin inside the duel (0.595-0.596), the nearer team (duel=1, 0.5873).
DEC_SHAPE_L2_16 = dict(DEC_SHAPE_L2_14, duel=2)       # L2-15 decoder (+ the L2-16 stoppage model, DEAD_FAST)
# L2-18 (decided on `real1`; envelope L2-18-stop-edges-fable.out.md): ed_exit=1. Post-pass on the
# finished labels (possess/edges.py snap_edges, applied in decode.decode_halves): the EXIT of every
# predicted stoppage >= 3 s that ends on a hidden-ball frame moves to the largest step up of the
# fast players' raw speed (p90, 1 s before against 1 s after) within +-2 s. The decoder puts an
# edge where the stoppage log-odds cross zero, and those are built from centred 1-15 s windows:
# at true hidden restarts they cross 0.6-1.0 s early while the raw speed rises 0.6-0.9 s AFTER
# the restart (predicted hidden exits: median -1.0 s on all three tune matches, after: +0.40 /
# -0.18 / +0.16 s). Nothing is learned; window and search range are the MatchLens dead-ball
# anchors (stoppage_ml.py STATIONARY_DISP_WINDOW_S 1.0, STATIONARY_DURATION_S 2.0), not tuned.
# real1 0.6061 -> 0.6081, original bench 0.6283 -> 0.6305; small (paired bootstrap +0.0020,
# CI90 [+0.0002, +0.0041]). NOT adopted: entries as well (ed_entry=1: real1 0.6102, original
# 0.6332, but hidden D precision on idsse1 falls 0.015 > the 0.01 gate), +-4 s search (lengthens
# D), a constant exit shift (median fixed, score +0.001), the cue as a transition cost, an
# edge-aware stoppage model (edge AUC down), a lone-taker hold (taker on 3-11% of live frames).
DEC_SHAPE_L2_18 = dict(DEC_SHAPE_L2_16, ed_exit=1, ed_w=1.0, ed_win=2.0)
# L2-20 (decided on the bench variant `real2` = real1 + ball smoothed to the provided match's INPUT
# statistics, state-coupled hiding, player loss; envelope L2-20-real2-fable.out.md): smooth_touch=1.
# When the ball track is low-passed (decode.ball_smoothed: acceleration p99 < 150 m/s2, read from
# the track, no labels) the touch detector runs with decode.SMOOTH_TOUCH (8-frame velocities, one
# touch per velocity-change peak, 2 m radius). On real2 the old 2-frame test fires on the hardest
# kicks only (touch-hold share 0.18-0.34 of visible frames, right team 0.81-0.87) and the touch
# emission LOSES 0.024; with SMOOTH_TOUCH the hold covers 0.66-0.75 (right team 0.89-0.94):
# real2 0.4303 -> 0.4833, every match up. real1 and the original bench are not low-passed
# (p99 204-619 per half; any threshold 48-204 gives identical output on every dataset): labels byte-identical (0.6081 / 0.6305). Old default: DEC_SHAPE_L2_18 or
# `--dec smooth_touch=0`.
DEC_SHAPE_L2_20 = dict(DEC_SHAPE_L2_18, smooth_touch=1)
# L2-21 (real2 and real1 co-decide; envelope L2-21-hidden-d-real2-fable.out.md): sd_prior=1. The
# stoppage log-odds of the hidden frames are corrected for the share of D among the hidden frames
# of the match itself, estimated without labels (`prior_shift` above, applied in
# decode.decode_halves). real2 0.4833 -> 0.5173 (hidden D share predicted 0.55 / 0.62 / 0.66 ->
# 0.73 / 0.82 / 0.84, true 0.71 / 0.83 / 0.80); real1 and the original bench: estimated shift below
# PRIOR_MIN_SHIFT, labels byte-identical (0.6081 / 0.6305). Old default: DEC_SHAPE_L2_20 (no key = off) or
# `--dec sd_prior=0`.
DEC_SHAPE_L2_21 = dict(DEC_SHAPE_L2_20, sd_prior=1)
# L2-22 (envelope L2-22-hidden-carry-fable, ADR-019): the first visible frame after a hidden block
# >= 1 s starts a touch hold for the only team within 1 m (MatchLens control zone). real2 0.5173 ->
# 0.5326; Fable QA over 5 real2 seeds: pooled +0.0090 CI90 [+0.0055, +0.0126], median seed +0.0079.
# Recorded cost: real1 -0.0013, orig -0.0023 (inside tolerances). Old default: DEC_SHAPE_L2_21.
DEC_SHAPE_L2_22 = dict(DEC_SHAPE_L2_21, reappear=1.0)
# L3-A4 (envelope L3-A4-short-possessions): vtx=2.0. On a low-passed ball track (smooth_touch regime
# only) the toucher's team is read at the contact point rebuilt from the straight incoming and
# outgoing flight (decode.touch_vertex; Vidal-Codina §2.3.3 compares incoming vs outgoing over the
# whole control interval), not at the smoothed ball, which sits ~2 m off the feet on a hard
# deflection. Radius = SMOOTH_TOUCH r_touch (2 m; new, unswept: spread=1, the 2-lag flight window, the contact-time clip; none from the provided match). Touch-team accuracy on live
# peaks 0.94-0.97 -> 0.98-0.99 with more uncontested touches. real2 0.5326 -> 0.5618; seeds 1-3
# +0.033 / +0.031 / +0.026 (bootstrap CI90 lower bounds > 0); real1 0.6068 and orig 0.6283
# byte-identical (not low-passed). Old default: DEC_SHAPE_L2_22 or `--dec vtx=0`.
DEC_SHAPE_L3_A4 = dict(DEC_SHAPE_L2_22, vtx=2.0)
# L4 (ADR-022, user decision 2026-11-15): organisers' control definition (CHALLENGE.md 53-80: a
# deflection / touch without control is not a change of possession). ctl_s=1.0: a predicted team
# spell under 1 s right after the other team's live spell goes back to that team; a kept
# interception starts at its first touch (ctl_truth.relabel). Judged on ctl truth T=2 (pooled CI90
# +0.0163 [+0.0106, +0.0219]) and T=3; costs -0.0066 pooled on the old truth (definition change,
# not a gate pass). 1.0 rather than 2.0 from external numbers only (2.0 loses on T=1 truth).
# Old default: DEC_SHAPE_L3_A4 or `--dec ctl_s=0`.
DEC_SHAPE_L4 = dict(DEC_SHAPE_L3_A4, ctl_s=1.0)
# L8 T95 (ADR-023, user decision 2026-11-17, cold QA MERGE): after ctl_s, a team spell under 2 s that
# follows the other team's live spell and holds no uncontested touch of its own team goes back to
# that team (`decode` ctl_hi / ctl_touch). ctl T=2: real2 4 seeds +0.007..+0.016 (median +0.011),
# real1 +0.002, orig +0.004; old truth -0.001 (neutral). Old default: DEC_SHAPE_L4 or `--dec ctl_hi=0`.
DEC_SHAPE_L8_T95 = dict(DEC_SHAPE_L4, ctl_hi=2.0, ctl_touch=1)
# L8 T100 (ADR-024, cold QA ADOPT): ball flight direction - on visible frames the team attacking in
# the ball's direction of travel gets w_fly evidence, ramped 4 -> 8 m/s (decode.flight_dir).
# On top of T95, ctl T=2 / old truth: real2 s0 +0.0133/+0.0137, s1 +0.0204/+0.0179, real1
# +0.0159/+0.0137, all CI90 > 0; unused seeds s2/s3/orig +0.012/+0.015/+0.015. Old default:
# DEC_SHAPE_L8_T95 or `--dec w_fly=0`.
DEC_SHAPE_L8_T100 = dict(DEC_SHAPE_L8_T95, w_fly=0.7, fly_v0=4.0, fly_v1=8.0)
# L8 T112+T113 (ADR-025, cold QA ADOPT): Law 9 on the visible ball. T112: a predicted stoppage's
# entry moves to the nearest line crossing within 5 s (edges.cross_entry). T113: ball > 0.5 m
# beyond a line, no stoppage nearby, for >= 2 s -> D until seen inside (edges.cross_dead).
# vs T95+T100, ctl T=2: real2 s0-s3 +0.017/+0.014/+0.012/+0.014, real1 +0.011, orig +0.013, all
# CI90 > 0; old truth +0.008..+0.013. Old default: DEC_SHAPE_L8_T100 or `--dec ed_cross=0 ed_xdead=0`.
DEC_SHAPE_L8_T112 = dict(DEC_SHAPE_L8_T100, ed_cross=5.0, ed_cross_strict=1, ed_xdead=1, ed_xdead_min=2.0)
# L8 T125 (ADR-026, cold QA ADOPT): learned ball-team evidence on visible frames (ballteam.MODEL,
# logistic, 21 input-only features, fitted leave-one-match-out on external tune matches only;
# constants in ballteam.py). w_bt=0.5, D-neutral. vs T112 default, ctl T=2: real2 s0-s3
# +0.019/+0.002/+0.012/+0.014 (median +0.013; seed-pooled +0.012 [+0.004, +0.019]; unused seeds
# s1-s3 +0.0095 [+0.0016, +0.0172]), real1 +0.009 [+0.002, +0.016], orig +0.003. Old default:
# DEC_SHAPE_L8_T112 or `--dec w_bt=0`.
DEC_SHAPE_L8_T125 = dict(DEC_SHAPE_L8_T112, w_bt=0.5)
# L11-A (ADR-028, user decision 2026-11-17, cold QA ADOPT): visible dead ball. w_bd=1.0: learned
# "the visible ball is dead" evidence (balldead.MODEL, 21 team-blind features, external tune matches
# only) added to the visible-frame stoppage log-odds; ed_still=1.0: a ball at rest >= 3 s with no
# player within 1 m is D (edges.still_dead). ctl T=2 vs T125 default: real2 s0-s2 +0.003/+0.001/
# +0.004, real1 +0.019 [+0.012, +0.026], orig +0.012 [+0.007, +0.018]; unused seeds real1 s1
# +0.022 [+0.015, +0.029], orig s1 +0.008 [+0.003, +0.013]. Old default: DEC_SHAPE_L8_T125.
DEC_SHAPE_L11 = dict(DEC_SHAPE_L8_T125, w_bd=1.0, ed_still=1.0)
# L30 hedge (ADR-030, user decision 2026-11-17): ctl_s 1.0 -> 0.6. Both-readings principle on external
# data only: scored under four label readings (old truth = every short possession kept; ctl T=1/2/3
# = controlled possession only). vs ctl_s=1.0: old +0.0050*, T=1 +0.0010, T=2 -0.0015, T=3 -0.0027*;
# worst-case regret 0.0102 -> 0.0052. ctl_hi=2.0 / ctl_touch=1 unchanged. Old default: DEC_SHAPE_L11.
DEC_SHAPE = dict(DEC_SHAPE_L11, ctl_s=0.6)


def decoder_inputs(pos: np.ndarray, period: np.ndarray, models=None, fps: float = FPS) -> Dict[str, np.ndarray]:
    """{"shape_team", "shape_dead", "edge_p90", "shape_ok", "sdead_pi"} for the `feats` dict of
    decode.decode_halves (use with DEC_SHAPE). edge_p90 = raw speed of the fast players for the
    edge post-pass (edges.py); shape_ok (frames with shape evidence) and sdead_pi (the stoppage
    model's training share of D; absent for models fitted before L2-21) feed `prior_shift`.

    pos from `collect` (or features.compute(..., positions=True)["pos"]); period per frame index.
    """
    f = features(pos, period, fps)
    models = models or MODELS
    st, sd = emissions(f, models, fps=fps)
    out = {"shape_team": st, "shape_dead": sd, "edge_p90": f["p90"], "shape_ok": f["ok"]}
    pi = dead_prior(models)
    if pi is not None:
        out["sdead_pi"] = pi
    return out


def dead_prior(models: Dict[str, Dict[str, np.ndarray]]) -> Optional[float]:
    """Training share of D of the stoppage model `emissions` uses (None: models without "pi")."""
    m = models.get("dead_fast") if DEAD_FAST and "dead_fast" in models else models.get("dead")
    return float(m["pi"]) if m is not None and "pi" in m else None


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (rank statistic; ties broken arbitrarily)."""
    y = np.asarray(y, bool)
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def save_models(path: str, models: Dict[str, Dict[str, np.ndarray]]) -> None:
    np.savez(path, **{"%s_%s" % (m, k): v for m, d in models.items() for k, v in d.items()})


def load_models(path: str) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    z = np.load(path)
    return {m: {k: z["%s_%s" % (m, k)] for k in ("mu", "sd", "w", "b", "pi") if "%s_%s" % (m, k) in z.files}
            for m in ("team", "team_ctx", "dead", "dead_fast") if "%s_w" % m in z.files}
