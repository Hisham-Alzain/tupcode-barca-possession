id: L3-A4-qa-fable — **PASS, adopt with changes** (default is safe for heavily smoothed tracks only; say so)

**Reproduced.** real2 0.5618, real1 0.6068, orig 0.6283; `vtx=0` 0.5326; pytest passes; real1/orig labels identical on/off.

**Seeds (real2, sigma 7, mean gain; metrica1 / metrica2 / idsse1).**
| s0 | s1 | s2 | s3 | s4 (new) | s5 (new) |
|---|---|---|---|---|---|
| +.029 | +.033 | +.031 | +.026 | +.030 (-.006/+.028/+.068) | +.026 (-.000/+.025/+.054) |

Pooled 6-seed bootstrap: +0.029, CI90 [+0.017, +0.042]. metrica1 -0.001 [-0.017, +0.016] (neutral), metrica2 +0.031 [+0.015, +0.047], idsse1 +0.057 [+0.029, +0.086].

**Sensitivity (F1, the main finding).** The gain depends on smoothing strength:
- sigma 3: **-0.031** [-0.043, -0.021], s0 and s4, all matches lose
- sigma 5: +0.008 [-0.004, +0.021], not significant
- sigma 7: +0.029; sigma 10: +0.073

A NaN-vertex fallback to ball distances does not repair sigma 3.

**Findings.**
1. decode.py:351 — vtx fires on any track with acc p99 < 150, including sigma 3 (p99 ~90), where it loses. Recommended guard, external-only: p99 threshold in the 57-89 gap. Not blocking.
2. Not a filter inverse: synthetic isolated touches recover to 0.0-0.65 m at sigma 3-7. Touches < 0.5 s apart land 1.6-3.7 m off (worse than the ball).
3. decode.py:149-153 — peaks within 16 frames of a gap have no vertex: 4-5% of peaks, ~2.5% of old touches dropped.
4. shape.py:590 "no new constant" is inaccurate: `spread=1`, the 2-lag window and the clip are new, unswept, external-only.
5. Geometry sound at sigma 7: offset max < 6 m, t1 clipped ~1%.

**Transfer (input only).** Provided vertex-ball offset at hard peaks 2.6 m (bench sigma 7: 1.7-2.0; sigma 3: 1.0): heavy regime. Counts and 3.2% changed match bench direction.

**Compliance.** Clean; inherits ADR-017 disclosure.

**Commands.** `evaluate.py --bench real2|real1|orig --no-public [--dec vtx=0]`; scratchpad `qa_a4.py`, `qa_offs.py`; all through ramlock.
