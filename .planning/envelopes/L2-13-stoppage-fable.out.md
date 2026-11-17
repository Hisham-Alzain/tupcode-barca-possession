```yaml
id: L2-13-stoppage-fable
status: partial (bench gates met; transfer check mixed: the gain is expected to shrink to 0..+0.01 on the provided match). Tune only, no held-out, nothing committed, public5 not run.
matchlens_start: nothing new ported; deadball2 detector (MatchLens deadball port) untouched. Changed only how the decoder weighs it.
files: src/possess/decode.py (w_live), src/possess/shape.py (DEC_SHAPE, DEC_SHAPE_L2_10), scripts/evaluate.py (--bench), scripts/tune_shape.py (bench=, GRID5), scripts/diagnose.py (bench=, transfer, dhide), tests/test_decode.py (+2; 107 pass)
```
**Change (18 configs, decided on real1, LOMO).** Player-only stoppage model has AUC 0.90-0.96 on visible frames but the detector's "live" verdict (-1 per frame) vetoed it. New default: `w_live=0.0` (detector "live" = no evidence, "dead" keeps +1), `sdead_vis=2.5`. No ball signal added.

| tune | m1 | m2 | i1 | mean |
|---|---|---|---|---|
| real1 before | 0.585 | 0.525 | 0.472 | 0.5276 |
| real1 now | 0.629 | 0.583 | 0.507 | **0.5731** |
| orig before | 0.630 | 0.585 | 0.539 | 0.5845 |
| orig now | 0.647 | 0.615 | 0.538 | **0.6002** |

Final D recall/precision, visible frames: real1 0.38/0.77, 0.41/0.99, 0.55/1.00 -> 0.77/0.84, 0.78/0.98, 0.78/0.98; orig 0.84/0.84, 0.87/0.98, 0.90/0.97.

**Transfer (input only, `eval/results/transfer_stoppage.json`).**
- Resembles: D runs per half hour 25.8 (real1 23-27), median 16.6 s (15.6-20.7), max 169 s (129-212, unchanged), possessions 116 (92-111).
- DIFFERS: players "stopped" on 15.8% of visible / 59.8% of hidden frames; bench ~40% on both. Predicted D on visible frames 6.6% -> 12.7% (real1 17-23% -> 33%). The real ball is mostly lost while play is dead; the bench hides it independently of the truth.
- `dhide` (ball also hidden inside a share p of true stoppages, real1): gain +0.046 (p=0), +0.026 (0.5), +0.009 (0.8), -0.001 (0.95). The provided match sits between p 0.8 and 0.95. New default stays 0.571-0.574 at every p.

**Commands.** `evaluate.py --variant L2-13 --bench real1 --no-public` (and without `--bench`); `tune_shape.py grid 5 bench=real1`; `diagnose.py transfer`; `diagnose.py dhide`.

**Risks.** Bench overstates visible stoppages, so L2-12's exposure and this gain are both smaller on the real match. `audit_bench.py` FULL now carries the new default. shape.MODELS unchanged. CLAUDE.md still forbids reading data/challenge (ADR-001 allows input); only cached input aggregates were read.
