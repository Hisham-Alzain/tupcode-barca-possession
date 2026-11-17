```yaml
id: L2-11-next-fable
status: win   # default switched (pipeline.DEAD2); tune only; held-out / public / challenge NOT looked at; not committed
matchlens_delta: MatchLens deadball.py STATIONARY rule (0.3 m/s, 2 s) kept, back-dated to the frame the ball stopped and applied with a player at the ball; restart on ball movement (set_piece_detector idea), not on a player being near
```
**Diagnosis (tune, default 0.5845):** restart team is already right (340 of 421 restarts, 25 wrong team). The loss is stoppage timing with the ball visible:

| | m1 | m2 | i1 |
|---|---|---|---|
| stop, predicted - true, median | +0.44 s | +0.72 s | +1.36 s |
| restart, median | -1.88 s | -0.96 s | 0.00 s |
| oracle: truth D added on visible frames | 0.685 | 0.694 | 0.574 |

Stops are late: MatchLens confirms a still ball after 1.4 s. Restarts are early: a player within 2 m of a dead ball counted as play.

**Change (`deadball2.detect`, rule, nothing learned):** `still_v=0.3`: a ball below 0.3 m/s for 2 s is dead from its first slow frame. `v_kick=3.0`: play resumes when the ball exceeds 3 m/s. `still_far=0` (MatchLens 2 m: 0.6062).

**Tune:** 0.5845 -> **0.6125** (m1 0.630 -> 0.649 318/357, m2 0.585 -> 0.629 282/330, i1 0.539 -> 0.559 368/511). Possessions invented in visible stoppages 103 -> 60. **IDSSE only: +0.020.** `v_kick` 2 / 3 / 5: 0.6058 / 0.6062 / 0.6047. 34 configs.

**Noise:** detector ball sigma 0.05 / 0.10 m: 0.6108 / 0.6053. Players 0.10 m + 7-12 + touch ball 0.10 m: 0.5431 -> 0.5705.

**Reproduce:** `uv run python scripts/evaluate.py --variant L2-11-restart --no-public` -> 0.6125; `scripts/diagnose.py` (new section e); `uv run pytest` (105 pass).

**Risks:**
- **Neither bench ball is real during stoppages.** IDSSE freezes it (speed exactly 0 on 98% of true-D frames); Metrica's is the fill line. The still-ball gain is not validated on a real dead ball. Clean evidence only for false alarms: IDSSE live ball is still >= 1 s on 0.17% of frames.

**Not adopted:** `w_dead=2` 0.6196 (all three up); `out_m=0` 0.6180 (Metrica only, IDSSE -0.001); `still_v=1.0` 0.6263 (fits the fill line).
