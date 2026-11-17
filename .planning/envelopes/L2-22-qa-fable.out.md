```yaml
id: L2-22-qa-fable
verdict: PASS (adopt reappear=1.0), with recorded costs. Tune only; no held-out; public truth never opened; no source edited; nothing committed.
```
**Reproduced.** real2 0.5173 -> 0.5326, real1 0.6081 -> 0.6068, orig 0.6305 -> 0.6283. pytest 150 pass. No-flag labels == `reappear=0` on all 15 seed x match decodes.

**Seeds (real2, in memory, gain m1 / m2 / i1 = mean, own CI90).**

| seed | m1 | m2 | i1 | mean | CI90 |
|---|---|---|---|---|---|
| 0 | +.0110 | +.0085 | +.0266 | +.0154 | [+.0095, +.0215] |
| 1 | +.0016 | +.0002 | +.0090 | +.0036 | [-.0020, +.0094] |
| 2 | +.0206 | +.0026 | +.0153 | +.0128 | [+.0062, +.0199] |
| 3 | +.0080 | +.0044 | +.0114 | +.0079 | [+.0030, +.0136] |
| 4 | +.0081 | +.0025 | +.0056 | +.0054 | [+.0001, +.0106] |

**Pooled paired 2-min block bootstrap.** Seeds 0-4: +0.0090 [+0.0055, +0.0126], median +0.0079. Seeds 1-4 (not seen when built): +0.0075 [+0.0039, +0.0111], median +0.0067. 0 of 2000 draws <= 0. Both criteria met; 15 of 15 cells >= 0.

**Costs.**
- orig -0.0023, CI90 [-0.0048, -0.0004], 2000/2000 draws <= 0: a real loss, inside the 0.005 tolerance. real1 -0.0013 [-0.0040, +0.0009].
- metrica2 alone: pooled CI90 [-0.0007, +0.0082].
- Dead balls are not overridden: truly dead fires labelled D fall 41 -> 27 of 68 (seed 0), 42 -> 35 of 70, 42 -> 30 of 72, 31 -> 18 of 59. Live fires right rise 87 -> 109 of 139 (seed 0).
- Noise gate not re-run (builder: CI includes 0).

**Logic.**
- `decode.py:194-195` NaN distance = team absent, so the rule can fire with one team untracked. 0 cases in about 1 900 blocks; untested.
- `decode.py:281-290` smooth regime: touch radius 2 m, reappear 1 m; frames removed by `clean_ball` count as hidden. Unmeasured.
- `decode.py:203-205` real touch wins; both teams inside never fires (duel=2 safe).
- Periods: `pipeline.py:79` decodes per half, counter resets. A half starting hidden fires at kick-off, uncounted by `_cy_blocks`.
- Stop rule: concerned carry; one config, no grid; fresh seeds confirm.

**Transfer.** 77/131 (59%) vs bench 44-61%; 0.73% of frames changed vs 264-1260 frames per bench match (same order, match length not checked): consistent.

**To adopt:** `reappear=1.0` is not yet in `shape.py` DEC.

**Commands.** `scratchpad/qa_seed.py <root> <seed>` (0-4); `diagnose.py carry check bench=real1|orig`; `uv run pytest`.
