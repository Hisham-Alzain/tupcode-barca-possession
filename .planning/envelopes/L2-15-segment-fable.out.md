```yaml
id: L2-15-segment-fable
status: partial (bench gates met, idsse1 up; near-inert on the provided match). Tune only, no held-out, nothing committed, public ground truth never opened.
matchlens_start: possession.py duel zone (Vidal-Codina 2.3), already ported in team_control ("duel keeps the current team"); now also applied in decode.touch_hold.
files: src/possess/decode.py (duel=2, ball_impulse split out), src/possess/shape.py (DEC_SHAPE, DEC_SHAPE_L2_14; duel=0 spelled out in old configs), scripts/diagnose.py (segment; transfer: L2-14 column + touch stats), scripts/tune_shape.py (GRID7), tests/test_decode.py (+2; 111 pass)
```
**Diagnosis (real1, 0.5834).** idsse1: 293 of 511 true possessions < 4 s; 105 hidden (2 found), 188 visible (69 found). Sources date a switch alike: new team's touch minus true switch, median 0.00 s on all three, so no definition fix exists. Constants are flat (22 one-at-a-time, none above 0.5874; lower switch cost adds possessions, not score). Lever: touch team. Nearer team right on 0.97-0.98 of uncontested touch frames, 0.77-0.84 of contested; oracle on contested touches +0.025. Missed IDSSE flickers (opponent pokes the ball in a duel; 1 s for the other team before a stoppage) have no kinematic signature.

**Change (rule, nothing learned).** `duel=2`: a contested touch no longer clears the hold; the last uncontested toucher keeps it.

| tune | m1 | m2 | i1 | mean |
|---|---|---|---|---|
| real1 before | 0.649 314/357 | 0.591 279/330 | 0.510 356/511 | 0.5834 |
| real1 now | 0.659 312 | 0.598 272 | 0.523 361 | **0.5935** |
| orig before | 0.665 | 0.627 | 0.541 | 0.6110 |
| orig now | 0.675 295 | 0.633 253 | 0.553 352 | **0.6205** |

Gate (real1, sigma 0 / 0.05 / 0.10 / 7-12 players / both): 0.5935, 0.5928, 0.5914, 0.5257, 0.5247 (before 0.5834, 0.5823, 0.5813, 0.5190, 0.5175).

**Transfer (input only, `transfer_decoder.json`).** Possessions / half hour, L2-14 -> now: provided 112.46 -> 112.46 (0%); bench -2.7..+2.0%. Contested touch onsets: provided 0.16 / min, bench 1.6-3.0; uncontested 6.5 vs 20-46. Expect ~0 on the real match.

**Not adopted.** Learned touch-team model 0.596-0.603; duel margin 0.595-0.596; duel=1 0.5873.

**Commands.** `evaluate.py --variant L2-15 --bench real1 --no-public` (and without `--bench`); `tune_shape.py grid 7 bench=real1`; `tune_shape.py gate bench=real1`; `diagnose.py segment bench=real1 [duel=0]`; `diagnose.py transfer`.

**Risks.** Ball noise on the touch ball reverses it (0.1 m: 0.5812 vs 0.5839; 0.2 m: 0.5392 vs 0.5470); provided ball sigma 0.000. ~45 candidates tried. Transfer runs were auto-backgrounded (alone). A frame-diff count on the provided input was blocked by the permission classifier; PowerShell RAM check blocked, `/proc/meminfo` used (>= 4.9 GB).
