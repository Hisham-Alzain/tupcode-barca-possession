```yaml
id: L2-21-hidden-d-real2-fable
status: WIN, decision rule met. Default changed in the working tree (sd_prior=1), NOT committed. Tune only; no held-out; public truth never opened. Needs Fable QA.
new_angle: label-free prior-shift correction (EM, Saerens 2002) of the stoppage log-odds on hidden frames; not T27/T56 (weight), not T55/T59.
matchlens_start: shape.py stoppage model. deadball.py, stoppage_ml.py, set_piece_detector.py checked: fixed ball thresholds, no class prior.
files: src/possess/{shape,decode}.py, scripts/{tune_shape,diagnose}.py, tests (149 pass)
```
**Diagnosis.** The stoppage model is fitted on all frames (D share 0.40). On real2 hidden frames D is 0.71-0.83: observed log-odds sit 1.5-2.5 above the model's. Also 10-17% of hidden frames have no shape evidence (close-ups; D 0.82-0.99, recall 0.28-0.57). Oracle +0.066.

**Change.** `shape.prior_shift`: hidden D share estimated per match from the model's own posteriors (0.700 / 0.803 / 0.805; true 0.695 / 0.807 / 0.797), hidden log-odds shifted; no-evidence hidden frames get that prior. Applied only when |shift| >= 0.6.

| | m1 | m2 | i1 | mean |
|---|---|---|---|---|
| real2 before | 0.553 302/357 | 0.504 283/330 | 0.393 328/511 | 0.4833 |
| real2 now | 0.569 307 | 0.554 287 | 0.429 321 | **0.5173** |
| hidden D pred (true) | 0.551 -> 0.727 (0.714) | 0.622 -> 0.819 (0.826) | 0.662 -> 0.838 (0.800) | |

real1 0.6081, orig 0.6305: gate shut, counts unchanged. Ungated: real1 0.6039, orig 0.6258.

**Evidence.** Bootstrap +0.0339, CI90 [+0.0236, +0.0448]. Seed 1: 0.4824 -> 0.5152, CI90 [+0.0230, +0.0429], all matches up. Noise 0.05 / 0.10 / 7-12 players + 0.10: +0.034 / +0.032 / +0.035; real1 unchanged under both.

**Gate.** Random hiding, 60 masks: max |shift| 0.49; real2 0.97-1.82. 0.6 fixed before the provided estimate was read.

**Transfer** (`transfer_sd_prior.json`): provided estimate 0.869 (shift +2.27); hidden D 0.724 -> 0.897 (real2 +0.18..+0.20); possessions per half hour 119.0 -> 117.8; D runs 24.4 -> 30.0 (real2 +14..+35%); 7.5% of frames change.

**Commands.** `evaluate.py --variant L2-21 [--bench real2|real1] --no-public [--dec sd_prior=0]`; `diagnose.py prior check [seed=1 names=X] [noise=0.1 drop=1] | null | transfer`.

**Risks.** Provided hidden D 0.897 exceeds every real2 match; no-evidence frames (18% of its hidden frames) become 99.7% D. Gate margin 0.49 vs 0.6. ~11 configs; visible-frame EM (0.5200) not taken.
