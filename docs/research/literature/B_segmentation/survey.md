# Cluster B survey: whole-match segmentation and ball-free possession inference

Date: 2026-11-12. Method: WebSearch only (WebFetch blocked), so every number below comes from abstracts, search snippets or secondary summaries; none was checked against a paper's tables. "Not found" means the search text did not contain it.

Bench reference (from .planning/LOOP.md and docs/possession-state-machine.html): tune spellF1 0.356 (iteration 2) -> 0.375 (iteration 3, deadball2). Oracle lifts on the iteration-2 base: perfect visible-ball team -> ~0.50-0.61, perfect hidden-ball team -> ~0.53, perfect stoppages -> ~0.56, all three -> ~0.75+. Known failure modes: k-frame retention switches late and dates turnovers late; a turnover inside a ball gap is dated when the ball returns; IDSSE under-switching (286 predicted vs 511 true possessions); possession survives ~75% of gaps (swarm F2).

## Sources

| Note | Year, venue | Method | Data needed | Reported result | Relevance |
|---|---|---|---|---|---|
| [kim2026-pathcrf](kim2026-pathcrf.md) | 2026 KDD | Per-frame edge scores + Dynamic Masked CRF + Viterbi; context-dependent stay probability | Player tracking + events, no ball | 69.6% possession edge acc. (676 classes); 75.7% event F1; dynamic stay prob. fixes event count | **High** (design pattern) |
| [kim2023-ballradar](kim2023-ballradar.md) | 2023 KDD | Set Transformer + hierarchical Bi-LSTM, possession then ball | 15 GPS + 3 Metrica matches | Team-in-possession 85.8% from players only; ball err ~3.65 m | **High** (ceiling for players-only team) |
| [capellera2024-transportmer](capellera2024-transportmer.md) | 2024 ACCV | Masked set-attention transformer; per-frame state (pass/possession/uncontrolled/out of play) | Proprietary soccer data, size not found | ~80% state accuracy, ball unobserved | High (closest task) |
| [peral2025-cropdrop](peral2025-cropdrop.md) | 2026 WACV | + image crops, masked pre-training | Large proprietary + video | State 85.9%, possessor 89.6% | Low (needs video) |
| [li-link2026-intention-phases](li-link2026-intention-phases.md) | 2026 arXiv | Temporal graph attention + post-processing | 7 Bundesliga matches | Segment F1 0.68->0.79 from post-processing alone | High (evidence that decoding adds ~+0.1) |
| [perse2009-basketball-phases](perse2009-basketball-phases.md) | 2009 CVIU | GMM per phase on team configuration + temporal smoothing, no ball | Player trajectories, small labels | Offense/defense/time-out 85-95% (basketball), 90-97% (handball) | **High** (cheap, ball-free) |
| [rothe2026-tactical-states](rothe2026-tactical-states.md) | 2026 journal | Configuration clusters per ball zone -> lookup table to in/out-of-possession states | 3 training games | F1 > 0.9 most states | Medium (needs ball zone) |
| [fox2011-sticky-hdp-hmm](fox2011-sticky-hdp-hmm.md) | 2011 AoAS | Self-transition bias kappa against over-segmentation | Unsupervised | SOTA diarization | High (cheap HSMM substitute) |
| [gudmundsson2017-survey-hervieu-hsmm](gudmundsson2017-survey-hervieu-hsmm.md) | 2017 CSUR | Survey; Hervieu HSMM with GMM durations for handball phases | Labelled trajectories | not found | Medium |
| [killick2012-pelt](killick2012-pelt.md) | 2012 JASA | Penalised exact change-point search | Signal only | O(n), exact | Medium (date turnovers in gaps) |
| [farha2019-mstcn](farha2019-mstcn.md) | 2019 CVPR | Multi-stage TCN + smoothing loss vs over-segmentation | Frame labels | SOTA segmental F1 (non-sport) | Medium-low |
| [bauer-anzer2021-counterpressing](bauer-anzer2021-counterpressing.md) | 2021 DMKD | 134 features + XGBoost | 97 matches, 20k labels | AUC 0.874 | Low-medium (pressing features) |
| [fassmeyer2021-labeling-situations](fassmeyer2021-labeling-situations.md) | 2021 Frontiers | VAE + large-margin classifier | Large unlabelled corpus | >90% AUC corners/crosses/counters | Low-medium (set pieces) |
| [bialkowski2014-formations](bialkowski2014-formations.md) | 2014 ICDM / 2016 TKDE | Role assignment by entropy minimisation | Season of tracking | Team identity (no possession numbers found) | Low-medium |
| [kim2022-soccercpd](kim2022-soccercpd.md) | 2022 KDD | Formation/role change points | Tracking with IDs | Matches experts (no numbers found) | Low |
| [anon2026-gnn-match-phases](anon2026-gnn-match-phases.md) | 2026 arXiv | Delaunay GNN + sequence model, 7 phases | not found | not found (unverified) | Low |
| [link-hoernig2017-individual-ball-possession](link-hoernig2017-individual-ball-possession.md) | 2017 PLOS ONE | Sphere-of-influence individual possession | Ball + players | not found | Low for B (cluster A) |

## What the literature agrees on
1. Every strong system is **per-frame evidence + a temporal decoder**; the decoder is where segment-level scores are won (Li & Link: +0.10 segment F1 from post-processing; PathCRF: dynamic stay probability fixes event count; Fox: without a self-transition bias, HMMs over-segment).
2. **Players-only team-in-possession is ~80-86% per frame** (Ballradar 85.8%, TranSPORTmer ~80% state accuracy; Perše 85-95% offense/defense in basketball). Above our ~75% carry-forward-through-gaps baseline, but only usable through a sticky decoder.
3. A fixed switch penalty is the failure point: PathCRF's gain comes from making the stay probability context-dependent. That is our IDSSE 286-vs-511 symptom.

## Gaps
- No published system found that does **H/A/D labelling with the ball hidden in long blocks and no player IDs**; TranSPORTmer is closest but its data and per-class numbers were not found.
- No sports paper found that tunes the switch penalty against an IoU-matched segment metric like spellF1; we must tune it ourselves.
- No HSMM for football possession found (only Hervieu, handball phases).
- Accuracy of shape-only team classification **inside 5-30 s ball gaps of football** specifically: not found. Ballradar's 85.8% is averaged over all frames, mostly with the ball near players.
- Training cost/hardware for Ballradar, TranSPORTmer, PathCRF: not found. Ballradar/PathCRF have public code (PathCRF repo confirmed; Ballradar repo unconfirmed).

## Candidate fixes (testable hypotheses)

Effects below are estimates from the oracle gaps, not from any paper. All measured as tune spellF1 vs the current base (0.375), then held-out only if CIs separate.

### C1. Whole-match Viterbi over {H, A, D} with a tuned switch penalty (PathCRF pattern, Fox kappa) — rank 1
- Hypothesis: replacing k-frame retention + forward-fill with Viterbi over per-frame log-scores (existing team_control zone evidence -> log P(H)/P(A); deadball2 score -> log P(D); ball-hidden frames get a flat team emission, so they inherit through the transition cost) removes late switches and re-dates them to the best boundary.
- Prediction: closes part of the **visible-ball team gap** (late/early switch dating; flicker) and the fragmentation part of the stoppage gap. Tune spellF1 +0.02 to +0.05 (0.375 -> ~0.40-0.42). Possession count should move toward truth on IDSSE if penalty is tuned per frame rate, not per frame.
- Smallest implementation: one function `viterbi(logE[T,3], switch_cost, dead_entry_cost, dead_exit_cost)` in numpy (3x3, O(T)); emissions as clipped log-ratios of the features we already compute; tune 3 costs by grid on tune split, choose the one closest to the true possession count as tie-break.
- Cost: 0.5-1 day.

### C2. Shape-based team emission for ball-hidden frames (Perše, Ballradar, swarm F6) — rank 2
- Hypothesis: a small classifier P(H | team shape) using relative, count-normalised features (home-minus-away centroid x toward each attack direction, each team's y-spread and hull area, defensive-line heights, team speeds, ball's last-seen zone and time since last seen), trained per frame on Metrica + IDSSE truth (tune split only), fed as emission into C1 for frames without a ball, lets Viterbi find turnovers inside gaps and date them.
- Prediction: closes **10-30% of the hidden-ball oracle gap** (that gap is ~+0.17), i.e. +0.02 to +0.05 on top of C1. Without C1 it should *hurt* (per-frame 80% accuracy fragments possessions); test both to confirm.
- Smallest implementation: ~12 features, logistic regression or sklearn HistGradientBoosting, 5-fold by match; trains in minutes on CPU on ~5 matches (subsample to 5 Hz). Temper the emission (multiply log-ratio by w < 1, tune w).
- Cost: 1-1.5 days (features 0.5, training/calibration 0.5, tuning in C1 0.5).

### C3. Context-dependent switch cost (PathCRF "dynamic masked CRF") — rank 3
- Hypothesis: a single switch cost cannot fit both Metrica and IDSSE (IDSSE under-switches 286/511). Make the cost a function of context: lower when the ball is visible, fast, and closer to the other team, at gap boundaries, and right after a D segment; higher mid-gap and when the current team is in the zone.
- Prediction: possession count error halves on IDSSE; tune spellF1 +0.01 to +0.03 over C1. Mostly visible-ball team gap.
- Smallest implementation: cost_t = c0 - c1 * sigmoid(feature score) inside the same Viterbi; tune c0, c1 by grid (or fit a logistic "switch here" model on true boundaries).
- Cost: 0.5 day after C1.

### C4. Restart-aware D state (HSMM-lite with D durations, links to cluster C) — rank 4
- Hypothesis: allowing H -> D -> A in the decoder with a restart-team emission at the first live frames (team nearer the ball / set-piece configuration, Fassmeyer) fixes "after a stoppage the old team is kept".
- Prediction: closes a slice of the **stoppage gap**: +0.01 to +0.02. Minimum D duration (e.g. 2 s) enforced by expanding D into a chain of 2 s x fps states (cheap HSMM substitute).
- Cost: 0.5 day on top of C1.

### C5. Small learned sequence model (Ballradar-lite / MS-TCN smoothing loss) — rank 5, only if C1-C3 stall
- Hypothesis: a BiGRU or 1-stage TCN over per-frame pooled team features (DeepSets mean/max over players per team, no IDs), trained with random long-block ball masking (TranSPORTmer idea) and a smoothing loss, beats hand emissions.
- Prediction: uncertain; with 5 matches (only 3 in tune) overfitting risk is high; at best equal to C2 plus a few points per-frame accuracy. Still needs C1 decoding for spellF1.
- Smallest implementation: PyTorch, ~50k parameters, 5 Hz, sequences of 60 s; CPU training ~1-2 h per run.
- Cost: 2 days; not recommended with 5 days left unless C1-C3 are done by day 3.

### Not recommended now
Full Ballradar / TranSPORTmer / CropDrop reproduction (data and compute beyond us, no time to validate); SoccerCPD and Bialkowski role assignment (minutes-scale changes, need IDs); PELT as a separate stage (subsumed by C1+C2).

## Order of work suggested
Day 1: C1. Day 2: C2. Day 3: C3 (+C4 if time). Days 4-5: held-out look, freeze, submission. Each as one loop iteration with an oracle re-run after it to see which gap moved.
