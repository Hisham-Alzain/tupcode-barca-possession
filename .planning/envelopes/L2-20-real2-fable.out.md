```yaml
id: L2-20-real2-fable
status: real2 built + audited; fix = ADOPTION CANDIDATE (default changed in working tree, NOT committed). Tune only; public truth never opened. Needs Fable QA.
new_angle: first bench with a smoothed ball and state-coupled hiding.
matchlens_start: decode.touch_hold (impulse.py port); bench: nothing in MatchLens.
matchlens_delta: thresholds kept; low-passed track -> 8-frame velocities, one touch per peak, r 2 m.
files: scripts/{bench,audit_bench,evaluate,tune_shape}.py, src/possess/{decode,shape}.py, tests (145 pass)
```
**A. Audit** (`audit_inputs_real2.json`; provided | real2 range)
- accel p90, p99: 18, 37 | 19-20, 37-39
- touch-hold share: 0.27 | 0.17-0.33
- entry / exit ball speed: 2.1 / 12.3 | 2.4-3.3 / 10.2-12.8
- fast-player speed hidden: 1.57 | 1.65-1.77
- gaps > 60 s: 0.19 | 0.19-0.23
- players hidden: 19.2 | 18.4-19.4

Not matched: kinks 12.5 (8.1); reappears < 5 m 0.12-0.16 (0.28).

**B. Switch gains** (old default 0.4303 / 0.6081 / 0.6305)
| | real2 | real1 | orig |
|---|---|---|---|
| touch | **-0.024** | +0.072 | +0.094 |
| duel | -0.005 | +0.009 | +0.009 |
| ADR-011 | +0.002 | +0.055 | +0.015 |
| dead_fast | +0.003 | +0.012 | +0.007 |
| ed_exit | 0.000 | +0.002 | +0.002 |
| shape stoppage | +0.031 | +0.105 | +0.097 |

Smoothing is the loss: real2 unsmoothed scores 0.6376.

**C. Fix** `smooth_touch=1`: accel p99 < 150 (bench 280-570; provided 30 / 47) -> `SMOOTH_TOUCH`. real2 **0.4833**, every match up; real1, orig byte-identical. Lag 6 / 10: 0.4830 / 0.4808. Noise gate +0.041..+0.053; ball noise 0.05 m switches it off. ~20 configs.

**Transfer** (`transfer_smooth_touch.json`): provided hold share 0.27 -> 0.67 (real2 0.66-0.75); possessions per half hour 112.5 -> 119.0; hidden D 0.724 unchanged; 4.8% of frames change.

**Not calibration:** REAL2 constants match bench INPUT to provided input aggregates; no provided label read; the algorithm is chosen on external truth only.

**Commands:** `audit_bench.py inputs --variant real2`; `evaluate.py --variant X --bench real2 --no-public [--dec smooth_touch=0]`.

**Risks:** acts only on the provided match, unverified there; one seed; real2 hidden D predicted 0.55-0.66 vs truth 0.71-0.83 (next lever). Suggest real2 co-decides with real1.
