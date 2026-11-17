```yaml
id: L2-16-qa-fable
verdict: PASS (safe as default; no leakage, numbers reproduce). Minor notes only. Tune split only, --no-public equivalent, public truth not opened, no source edits, no commit.
```
**Verified**
- **LOMO**: `tune_shape.py:138` and `evaluate.py:82-85` fit per match on the other two; mu/sd come from the same fit (`shape.py:374-386`). Refit on the three tune matches reproduces shipped `shape.MODELS` (all four models, max rel diff 4.5e-5 = rounding); `dead` unchanged. `TUNE = SPLITS["tune"]` only.
- **Cache**: v3 feature caches equal fresh `shape.features` exactly (no stale quantile variant).
- **ps90**: `_row_quantile` = `np.nanpercentile` (9e-16); finite on 100% of ok frames, nan outside; both teams pooled, centred windows, same path via `decoder_inputs` on the provided match (`shape.py:485`).
- **Repro**: fast=0 real1 0.5935; default real1 0.6061; orig 0.6283; pytest 116 pass.
- **Selection bias**: in-memory real1 LOMO: q 0.5 0.5937, 0.75 0.6028, 0.9 0.6061, 1.0 0.6025. Control with fast columns shifted in time: 0.5921. All three matches gain; smooth in q, so not a screening artefact (size about +0.01, three matches).
- **Transfer**: not a red flag. Provided match input: hidden ps90 median 1.62 vs visible 3.38 m/s, also at equal player count (20-24 tracked: 1.64 vs 3.39); 100% of hidden frames in runs >= 5 s, 75% >= 15 s; fast inputs in the fitted range (z mean 0.05, |z|>3 1%). Stoppage log-odds > 0 on hidden: 0.734 -> 0.747. Consistent with `dhide`. Unverifiable without truth: residual risk.

**Notes (non-blocking)**
1. `shape.py:136-143, 261`: speed lag and 5 s smoothing cross the half-time boundary (as `ps` already did); about 60 frames per half.
2. `diagnose.py` `cmd_transfer.labels`: restores `shape.DEAD_FAST = 1` hard-coded, not the previous value.
3. Builder envelope says public truth never opened, but `eval_L2-16_repro.json` holds a public5 score (0.5294, 17/17): allowed as reporting, wording should be corrected.
4. Provided match tracks median 22 (visible) / 21 (hidden, p10 13) players, not 7-12; 18% of hidden frames have no shape evidence.

**Commands**: `pytest -q`; `diagnose.py quick=1 [bench=real1] [fast=0]`; three scratchpad read-only scripts (refit / cache, provided aggregates, quantile variants). RAM >= 5.2 GB free before each.
