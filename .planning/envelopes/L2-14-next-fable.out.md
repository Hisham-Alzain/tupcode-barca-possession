```yaml
id: L2-14-next-fable
status: win, small (real1 +0.010, orig +0.011; gates met). The larger half of the bench gain (w_rule=0) is NOT adopted: it fails the transfer check. Tune only, no held-out, nothing committed, public ground truth never opened.
matchlens_start: nothing new ported. cap now equals the MatchLens control-zone radius (possession.py, team_control r_pz = 1.0 m).
files: src/possess/shape.py (DEC_SHAPE, DEC_SHAPE_L2_13), scripts/tune_shape.py (GRID6, gate file per bench), scripts/diagnose.py (transfer: L2-13 column + cap-band stats -> transfer_decoder.json; dhide <p...>), tests/test_decode.py (+2; 109 pass)
```
**Diagnosis (real1, 0.5731).** Oracles: stoppages 0.695, team hidden 0.709, team visible 0.691; restart team +0.002 (not a cause). 347 spurious possessions: 101 inside a true stoppage, 83 fragments, 63 wrong team. Cause targeted: ball proximity counted three times on visible frames (distance, frame rule, touch).

**Change.** `cap` 1.5 -> 1.0 (distance emission stops grading beyond the control zone). Scan of 24 constants + grid 6 (11).

| tune | m1 | m2 | i1 | mean |
|---|---|---|---|---|
| real1 before | 0.629 325/357 | 0.583 288/330 | 0.507 360/511 | 0.5731 |
| real1 now | 0.649 314 | 0.591 279 | 0.510 356 | **0.5834** |
| orig before | 0.647 | 0.615 | 0.538 | 0.6002 |
| orig now | 0.665 300 | 0.627 260 | 0.541 345 | **0.6110** |

5 of 6 halves up (idsse1 half 1 -0.006). Spurious 347 -> 318, missed 572 -> 567. Gate (real1, sigma 0 / 0.05 / 0.10 / 7-12 players / both): 0.5834, 0.5823, 0.5813, 0.5190, 0.5175 vs 0.5731, 0.5729, 0.5707, 0.5146, 0.5130.

**Transfer (input only, `eval/results/transfer_decoder.json`).**
- cap band (one team <= 1 m, other 1-1.5 m): provided 1.4% of visible frames, bench 1.8-2.7%.
- Predicted possessions / half hour: provided 116.1 -> 112.5 (-3%); bench -1..-4%. Resembles.
- `dhide` p 0.8 / 0.95: +0.007 / +0.007.
- w_rule=0 + cap=1.0: real1 0.5950, orig 0.6192, but provided possessions -24% (bench -1..-7%); w_rule=0 alone -14% vs -3..+1%. DIFFERS -> rejected.

**Negative.** Lead/lag stoppage model (+-2, +-4 s): AUC +0.003, 0.5725. Removed.

**Commands.** `evaluate.py --variant L2-14 --bench real1 --no-public` (and without `--bench`); `tune_shape.py grid 6 bench=real1`; `tune_shape.py gate bench=real1`; `diagnose.py transfer`; `diagnose.py dhide 0.8 0.95`.

**Risks.** Gain is Metrica-heavy (IDSSE +0.003) and gone at 0.2 m distance noise (0.5407 vs 0.5415). IDSSE under-segmentation unchanged. Runs over 180 s were auto-backgrounded by the harness (one at a time). CLAUDE.md still forbids reading data/challenge (ADR-001 allows input); only cached input aggregates read.
