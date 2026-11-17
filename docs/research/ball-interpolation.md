# Ball gap filling — candidate methods (research note, 2026-11-11)

Problem: tactical ATD has no ball on 41% of graded frames, concentrated in crowds, duels and long balls. MatchLens v0 treats "no ball" as "play stopped", which split 17 true possessions into 30 on the public window. We are offline, so every gap has a known start and end position (look-ahead is free).

## Where MatchLens ended (matchlens_start)
| File | What it already does | Verdict |
|---|---|---|
| `footballtactics/backend/events/eval/metrica.py:MetricaMatch.iter_frames(interpolate_ball=True)` | Buffers a gap until the ball returns, fills **linearly** between last-known and next-known, marks filled frames `conf=None` ("Kalman-coasted ghost") | **Port** — method 2 below, our v1 |
| `matchlens_ref/kinematics.py:PitchKinematics` (SG window 7, poly 2) and `impulse.py:SmoothedBallSeries` | Savitzky-Golay velocity on the ball series; ghost frames are skipped as a time base | **Keep** — downstream of any fill |
| `matchlens_ref/ball_utils.py:ball_airborne_confident / BallFlightDetector` | Flight state; returns `None` on a ghost frame so touch logic doesn't fire on filled positions | **Keep** — fill must preserve the `conf=None` flag |
| `impulse.py:contact_thresholds` | Treats `airborne is None` (ghost) as unknown, not as a touch | **Keep** |
Not in MatchLens: spline, Kalman/RTS, player-anchored fill, learned imputation.

## Methods
| # | Method | Idea | Good for | Weak at | Cost |
|---|---|---|---|---|---|
| 1 | **No fill, abstain** | Pass the gap through; emit `X` mid-possession (free on spellF1); stop the dead-ball detector from reading absence as dead | Any gap length, zero risk of inventing motion | Gives the touch detector nothing; hand-overs inside the gap are missed | none |
| 2 | **Linear, two-sided** (MatchLens) | Straight line last-known → next-known | Short gaps (≤ 0.5 s), aerial long balls (ground-plane path of a kicked ball is nearly straight) | Gaps containing a touch (direction change) — the line cuts the corner; crowds | trivial |
| 3 | **Cubic spline / PCHIP** | Fit through neighbouring known points; PCHIP is monotone, no overshoot | Smooth curved flight, short–medium gaps | Oscillates across a touch; needs ≥ 2 known points each side | low |
| 4 | **Physics (ballistic + drag)** | Constant-velocity with exponential deceleration in the ground plane; velocity from SG at the gap edges | Long balls, clearances, goal kicks (exactly where the ball is lost) | Rolling balls that get touched; needs flight state | low |
| 5 | **Kalman + RTS smoother** | Constant-velocity or constant-acceleration state; forward filter then backward smoothing uses future frames | Medium gaps with noisy edges; gives a covariance = how much to trust the fill | Erratic bounces, touches inside the gap; one more thing to tune (Q, R) | medium |
| 6 | **Player-anchored** | If the gap is in a crowd (k players within r m of both endpoints), the ball is at someone's feet: carry it with the nearest player (dribble) or along the line from the last toucher to the next receiver | Crowds and duels — the main gap cause — and it feeds possession directly | Wrong player picked = wrong team; needs a crowd test | low–medium |
| 7 | **Learned imputation from player tracks** | Set-Transformer + bi-LSTM (Zhan et al. 2023), TranSPORTmer, masked multi-modal pre-training (2512.19528): infer the ball from the 22 players | Long gaps, no ball needed at all | Training data (Metrica G1–G3 + IDSSE only), a day or more of work, GPU; risk of a worse result than 6 in 7 days | high |
| 8 | **Hybrid by gap length and context** (recommended v2) | ≤ 0.5 s: 2 or 3 · 0.5–3 s in a crowd: 6 · 0.5–3 s in open space / airborne: 4 or 5 · > 3 s: 1 | Matches each failure mode to its cause | Thresholds to tune (on external data only) | sum of parts |

## Design constraints
- Filled frames keep `conf=None` so MatchLens touch/flight logic treats them as ghosts (no fake impulses).
- A fill must never cross a gap that contains a likely restart (ball reappears at a restart spot, collective slowdown, Smart Tagging set piece).
- The dead-ball detector gets a separate signal: `ball_missing` ≠ `ball_dead`.

## How to choose (tune on external data only)
1. Take Metrica G3 and IDSSE (clean ball). Mask the ball with a gap generator calibrated on the challenge **input** stats (gap-length histogram, computed blind) and biased to crowds/long balls.
2. For each method: position RMSE vs the hidden truth, per gap-length bucket; touch recall/precision downstream; possession spellF1 on the degraded match.
3. Pick per bucket; ties → simpler. Then one blind public-window sanity run, logged, never used to choose.

## Sources
- Zhan et al., *Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM*, KDD 2023 — https://arxiv.org/pdf/2306.08206
- *Multi-Modal Soccer Scene Analysis with Masked Pre-Training* (2025) — https://arxiv.org/abs/2512.19528
- Kontos & Karlis, interpolation methods for broadcast tracking gaps, IJAS 35(2) 2023 — https://oajournals.fupress.net/index.php/ijas/article/view/15707
- Training-free off-screen player imputation (Metrica data, CPU) — https://arxiv.org/abs/2607.11548
- Kalman tracking limits on fast erratic objects (2025) — https://arxiv.org/html/2509.18451v1
- Kalman smoother filling whole missing frames (HyperTools tutorial) — https://hypertools.readthedocs.io/en/latest/tutorials/projectile_kalman.html
- SkillCorner open data (uncorrected broadcast tracking with gaps) — https://github.com/SkillCorner/opendata
