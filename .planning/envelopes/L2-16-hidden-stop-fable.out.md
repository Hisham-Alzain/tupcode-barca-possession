```yaml
id: L2-16-hidden-stop-fable
status: win on the bench gates (real1 0.5935 -> 0.6061, orig 0.6205 -> 0.6283); transfer mixed, see below. Tune only, no held-out, public truth never opened, nothing committed.
new_angle: speed of the FAST players (90th percentile over all tracked players), not the median player the stoppage model knew. Player-only, no ball.
matchlens_start: shape.py stoppage model. MatchLens deadball.py / set_piece_detector.py (port: stoppage_ml.py) checked: ball-based only.
files: src/possess/shape.py (FAST_FEATS ps90, ps90_5; model "dead_fast"; DEAD_FAST=1; MODELS refit, "dead" unchanged), scripts/tune_shape.py (FEAT_V 3 = new cache names; fast=), scripts/diagnose.py (fast=0, transfer, dhide, D recall line), tests/test_shape.py (+5; 116 pass)
```
**Diagnosis (real1).** Perfect hidden stoppages +0.040, almost all from missed D (+0.037), and only at the edges of stoppages >= 3 s (last 5 s +0.022, first 5 s +0.017, middle +0.002; < 3 s +0.004). Hidden restarts come 1.2-1.7 s early.

**Change.** Two columns added to the LOMO stoppage logistic. Hidden-frame AUC 0.907/0.945/0.955 -> 0.914/0.958/0.962.

| tune | m1 | m2 | i1 | mean |
|---|---|---|---|---|
| real1 before | 0.659 312/357 | 0.598 272/330 | 0.523 361/511 | 0.5935 |
| real1 now | 0.665 308 | 0.615 274 | 0.538 368 | **0.6061** |
| orig now | 0.685 302 | 0.647 262 | 0.553 371 | **0.6283** |

D recall/precision: hidden 0.792/0.872 -> 0.804/0.889; visible 0.788/0.932 -> 0.818/0.927. Most of the gain is on visible frames (hidden frames only: 0.5973).

**Gate (real1, off -> on).** sigma 0.05: 0.5928 -> 0.6046; 0.10: 0.5914 -> 0.6037; 7-12 players: 0.5257 -> 0.5333; both: 0.5247 -> 0.5323.

**Transfer (`transfer_hidden_stop.json`).** Possessions / half hour: provided 112.46 -> 112.46; real1 -1.3..+1.9%. DIFFERS: hidden D share provided 0.673 -> 0.711, real1 flat (0.437, 0.357, 0.363). With the ball hidden inside true stoppages (`dhide` 0.8 / 0.95) the bench also rises (0.53 -> 0.55, truth 0.60) and the gain holds: 0.5890 -> 0.6013, 0.5899 -> 0.6004.

**Commands.** `evaluate.py --variant L2-16 --bench real1 --no-public` (and without `--bench`); `diagnose.py bench=real1 [fast=0]`; `diagnose.py transfer`; `diagnose.py dhide 0.8 0.95`.

**Risks.** ~20 features screened by AUC, 13 decodes; p75 / top-3 / max variants all 0.602-0.606. Not adopted: w_sdead 0.4 (0.6102, one run). Hidden-edge oracle still +0.036. Transfer run was auto-backgrounded (alone). Needs Fable QA and a new draft.
