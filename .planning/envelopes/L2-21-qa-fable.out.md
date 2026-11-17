```yaml
id: L2-21-qa-fable
status: PASS, adopt with changes (findings 1-2) | compliance: defensible, disclose; residual risk is the user's call | transfer: plausible, no cap needed
```
**Reproduced** (`evaluate.py --variant L2-21-qa[-off] [--bench real2|real1] --no-public [--dec sd_prior=0]`): real2 0.5173, off 0.4833, real1 0.6081, orig 0.6305; pytest passes. No held-out run; public truth never opened by me.

**EM correct.** Standard Saerens update, starts at pi, converges in 20-27 iterations to 1e-9 (200 allowed). Hidden frames with evidence only, one estimate per match. `pi` comes from the leave-one-out fit (`tune_shape.py:157`). Shifted hidden log-odds are calibrated against tune truth in every bin (e.g. predicted 0.62 / true 0.58-0.65).

**Gate.** Not verifiable from history (uncommitted). But any threshold in 0.49-0.97 gives identical output on all benches and on the provided match (2.27), so 0.6 is not fitted. Ungated cost is small (builder: -0.004 / -0.005).

**Compliance.** No label, constant or threshold comes from the provided match; the estimate is recomputed from each match's own tracking input, identically everywhere, like the attacking-direction vote. That is inference, not fitting. Weak point: the method is literally "prior recalibration", and the rule text says "calibrating". Record an ADR and disclose it in the submission notes, with smooth_touch.

**Transfer.** Input aggregates, real2 range vs provided: hidden share 0.41-0.42 / 0.41; median block 10 s / 10.7 s; fast-player speed on hidden frames 1.66-1.80 / 1.61; raw model D probability on hidden frames 0.60-0.66 / 0.68; no-evidence 0.10-0.17 / 0.18. The provided match is slightly beyond real2 on every indicator, in the same direction, so 0.869 is a mild extrapolation. Overestimation is cheap on external data: real2 with the estimate biased +0.5 / +1.0 / -0.5 log-odds (shifts up to 2.82) scores 0.5198 / 0.5146 / 0.5139; capped at 1.2: 0.5140. No cap recommended. Unverifiable: the provided posteriors are less saturated (5% clipped vs 7-12%).

**Findings**
1. `shape.py:415` PRIOR_MIN_FRAMES=1500 is unvalidated: the gate was tested on full matches only, and per-half shifts already reach 1.4 (`shape.py:411`). On short inputs (clips) it would fire on noise. Raise it to the validated scale or document full-match-only.
2. `eval_L2-21_repro.json` and the overwritten committed `eval_L2-20.json` contain public5 scores (0.5059 both), against this envelope's "--no-public always". Confirm provenance; it must not inform the decision.
3. `shape.py:448`: no-evidence fill is on/off with the gate (discontinuity at 0.6).
4. Per-half estimates disagree (idsse1 0.654 / 0.895); pooled is right.

**Files written:** this envelope; `eval/results/eval_L2-21-qa*.json` (8, by evaluate.py).
