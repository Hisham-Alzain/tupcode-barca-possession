```yaml
id: L2-18-stop-edges-fable
status: SMALL WIN. One grid config (M1 exit snap) passes all 8 gates; default changed in the working tree, NOT committed. real1 0.6061 -> 0.6081, orig 0.6283 -> 0.6305. Tune only, no held-out, public truth never opened.
new_angle: constant costs cannot move an edge (M6 confirmed, median moves <= 0.12 s); edges judged by signed offset and moved by a post-pass.
matchlens_start: stoppage_ml.py _commit_flip (back-stamp); anchors 1.0 s step window, 2.0 s search. set_piece_detector / PlacedBallDetector re-checked - ball-based, no taker test.
matchlens_delta: edges.py snap_edges, offline, player-only; shape.DEC_SHAPE ed_exit=1, ed_w=1.0, ed_win=2.0 (old default - DEC_SHAPE_L2_16 or `--dec ed_exit=0`).
files: src/possess/edges.py (new), shape.py, decode.py, tune_shape.py (FEAT_V 4), diagnose.py (`edges ...`), tests (140 pass)
```
**Step 0 (m1 / m2 / i1).** Hidden exits median -1.02 / -0.94 / -1.12 s; entries +0.48 / -0.02 / -0.32 (wide). Oracle +0.021. Log-odds cross 1.04 / 0.56 / 0.92 s early; raw p90 half-rise 0.80 / 0.92 / 0.56 s AFTER the restart: H1 (smear).

**Stops.** M5 (taker on 10.6 / 3.1 / 5.0% of live frames), M3 (edge AUC down), M2 entry (sign differs).

**Grid, 21 configs (`eval/results/tune_edges.json`).** M2 exit shift zeroes the median, only +0.0012: the loss is spread, not bias. M4 0.6058-0.6086. Exit + entry: 0.6102, orig 0.6332, CI90 [+0.0014, +0.0070], rejected by gate 4 (i1 hidden D precision -0.015). +-4 s: 0.6117, lengthens D.

**Chosen.** 0.670 / 0.615 / 0.540, counts unchanged (308 / 274 / 368). Exit median +0.40 / -0.18 / +0.16 s; hidden D recall +0.017 / +0.009 / +0.024, precision -0.007 / -0.003 / -0.006. Bootstrap +0.0020, CI90 [+0.0002, +0.0041]. Noise 0.6068 / 0.6060 / 0.5349 (off 0.6046 / 0.6037 / 0.5333). dhide +0.0042 / +0.0042.

**Transfer (`transfer_edges.json`).** Provided hidden D share 0.711 -> 0.724 (bench +0.005..+0.014); possessions unchanged; exits +1.08 s (bench +0.48..+1.19).

**Commands.** `evaluate.py --variant L2-18 [--bench real1] --no-public`; `diagnose.py edges save|d0|profiles|grid|m4|check|dhide|transfer`.

**Risks.** Gain m1-heavy (m2 flat); CI barely above 0 after ~120 candidates; raises D on the provided match. Needs Fable QA.
